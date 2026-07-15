#!/usr/bin/env python3
"""Publish exactly one tournament winner after a one-shot sealed evaluation.

This stage consumes the candidate-only tournament's authenticated winner.  It
does not retrain or rerank candidates.  The sealed suite is evaluated exactly
once, behind an exclusive durable consumption marker; post-bake validation uses
only fit and candidate.  Existing target-native restore, composite bake, and
fresh-reload certification primitives own all model mutation and publication.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import importlib.util
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

import retain_qwen_exact_target_candidates as retention
import run_qwen_exact_target_native_guided_candidate_tournament as tournament
import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_guided_checkpoint_recovery import (
    GuidedCheckpointRecoverySink,
    canonical_json_bytes,
    canonical_sha256,
    portable_v3_factor_state_sha256,
    rebind_recovered_v3_factor_payload,
)
from neural_forge.hf_progressive_target_bridge import ExternalBehaviorGateReceipt
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    CausalBehaviorMetrics,
    authenticate_guided_factor_sources,
    capture_target_native_factor_state,
)


PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-guided-winner-publication-preregistration-v2"
)
RUN_SCHEMA = "neural-forge-qwen-guided-winner-publication-v2"
SEALED_CONSUMPTION_SCHEMA = "neural-forge-guided-sealed-consumption-v1"
BUILD_FILENAME = "prepare_qwen_exact_target_native_guided_winner_publication.py"
EXPECTED_SPLIT_COUNTS = {"fit": 168, "candidate": 24, "sealed": 32}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GuidedWinnerPublicationError(RuntimeError):
    """Winner provenance, sealed lifecycle, bake, or publication failed."""


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GuidedWinnerPublicationError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise GuidedWinnerPublicationError(
            f"{name} fields changed; expected={sorted(fields)!r}, actual={actual!r}."
        )
    return value


def _resolve(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise GuidedWinnerPublicationError(f"{name} must be a nonempty path.")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else base / raw
    cursor = candidate.absolute()
    while True:
        if cursor.is_symlink():
            raise GuidedWinnerPublicationError(f"{name} cannot traverse a symlink.")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    return candidate.resolve()


def _bound_file(
    path: Path | str,
    expected_sha256: str,
    name: str,
) -> Path:
    source = _resolve(path, base=Path.cwd(), name=name)
    expected = _sha256(expected_sha256, f"{name}.sha256")
    if not source.is_file() or stage._file_sha256(source) != expected:
        raise GuidedWinnerPublicationError(f"{name} is absent or changed.")
    return source


def _bound_json(
    path: Path | str,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    source = _bound_file(path, expected_sha256, name)
    payload = stage._strict_json(source)
    if not isinstance(payload, dict):
        raise GuidedWinnerPublicationError(f"{name} must contain one JSON object.")
    canonical_json_bytes(payload)
    return payload


def _module_file(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise GuidedWinnerPublicationError(
            f"Required module {module_name!r} is absent."
        )
    return Path(spec.origin).resolve()


def _implementation_hashes(*, builder_path: Path) -> dict[str, str]:
    builder = _resolve(builder_path, base=Path.cwd(), name="builder implementation")
    if not builder.is_file():
        raise GuidedWinnerPublicationError("Builder implementation is absent.")
    return {
        "runner_sha256": stage._file_sha256(Path(__file__).resolve()),
        "builder_sha256": stage._file_sha256(builder),
        "tournament_runner_sha256": stage._file_sha256(
            Path(tournament.__file__).resolve()
        ),
        "guided_runner_sha256": stage._file_sha256(Path(guided.__file__).resolve()),
        "exact_stage_runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
        "trainer_sha256": stage._file_sha256(
            _module_file(guided.GUIDED_TRAINER_MODULE)
        ),
        "checkpoint_recovery_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_guided_checkpoint_recovery")
        ),
        "progressive_bridge_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_progressive_target_bridge")
        ),
        "guided_adapter_sha256": stage._file_sha256(
            _module_file(guided.GUIDED_ADAPTER_MODULE)
        ),
        "independent_authenticator_sha256": stage._file_sha256(
            Path(retention.__file__).resolve()
        ),
    }


def _split_semantics(binding: stage.SplitBinding) -> dict[str, Any]:
    return {
        "path": str(binding.path),
        "sha256": binding.sha256,
        "expected_count": binding.expected_count,
        "row_ids_sha256": canonical_sha256(list(binding.row_ids)),
        "prompt_sha256s_sha256": canonical_sha256(list(binding.prompt_sha256s)),
        "payload_sha256": canonical_sha256(binding.payload),
    }


def _frontier_metrics(value: Any, name: str) -> CausalBehaviorMetrics:
    fields = set(CausalBehaviorMetrics.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GuidedWinnerPublicationError(f"{name} has a malformed metric schema.")
    integer_fields = {
        "action_count",
        "action_agreement_count",
        "parent_action_worst_rank",
        "rollout_count",
        "exact_rollout_count",
        "rollout_token_count",
        "token_match_count",
        "lcp_token_count",
        "lcp_at_least_4_count",
        "longest_run_excess",
    }
    numeric_fields = fields - integer_fields - {"split", "report_sha256"}
    if any(
        isinstance(value[field], bool) or not isinstance(value[field], int)
        for field in integer_fields
    ) or any(
        isinstance(value[field], bool)
        or not isinstance(value[field], (int, float))
        or not math.isfinite(float(value[field]))
        for field in numeric_fields
    ):
        raise GuidedWinnerPublicationError(f"{name} contains invalid metric types.")
    if value["split"] != "calibration-selection":
        raise GuidedWinnerPublicationError(f"{name} is not candidate-only behavior.")
    _sha256(value["report_sha256"], f"{name}.report_sha256")
    if (
        value["action_count"] <= 0
        or value["rollout_count"] <= 0
        or value["rollout_token_count"] <= 0
        or not 0 <= value["action_agreement_count"] <= value["action_count"]
        or not 0 <= value["exact_rollout_count"] <= value["rollout_count"]
        or not 0 <= value["token_match_count"] <= value["rollout_token_count"]
        or not 0 <= value["lcp_token_count"] <= value["rollout_token_count"]
        or value["lcp_token_count"] > value["token_match_count"]
        or not 0 <= value["lcp_at_least_4_count"] <= value["rollout_count"]
        or value["parent_action_worst_rank"] < 1
        or value["longest_run_excess"] < 0
    ):
        raise GuidedWinnerPublicationError(f"{name} contains impossible metric counts.")
    ratios = (
        ("action_accuracy", "action_agreement_count", "action_count"),
        ("exact_rollout_fraction", "exact_rollout_count", "rollout_count"),
        ("token_match_fraction", "token_match_count", "rollout_token_count"),
        ("lcp_fraction", "lcp_token_count", "rollout_token_count"),
    )
    if any(
        not math.isclose(
            float(value[fraction]),
            value[numerator] / value[denominator],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for fraction, numerator, denominator in ratios
    ):
        raise GuidedWinnerPublicationError(f"{name} contains inconsistent fractions.")
    return CausalBehaviorMetrics(**dict(value))


def _authenticated_frontier_binding(
    contract: tournament.AuthenticatedCandidateTournament,
    run_report: Mapping[str, Any],
    winner_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = winner_selection.get("candidate_frontier")
    required = {
        "schema",
        "path",
        "sha256",
        "observed_callback_count",
        "pending_callback_count",
        "completed_arms",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or canonical_json_bytes(raw)
        != canonical_json_bytes(run_report.get("candidate_frontier"))
        or raw.get("schema") != tournament.FRONTIER_SCHEMA
        or raw.get("observed_callback_count") != tournament.EXPECTED_CALLBACK_COUNT
        or raw.get("pending_callback_count") != 0
        or raw.get("completed_arms") != list(tournament.ARM_IDS)
    ):
        raise GuidedWinnerPublicationError(
            "Tournament winner is not bound to a complete callback frontier."
        )
    path = _resolve(raw["path"], base=contract.path.parent, name="candidate frontier")
    digest = _sha256(raw["sha256"], "candidate_frontier.sha256")
    frontier = _bound_json(path, digest, "candidate frontier")
    observations = frontier.get("observed_callbacks")
    retained = frontier.get("retained")
    frontier_fields = {
        "schema",
        "evaluation_contract_sha256",
        "completed_arms",
        "retained_count",
        "retained",
        "observed_callback_count",
        "pending_callback_count",
        "observed_callbacks",
        "distinct_full_v3_states_only",
        "portable_content_identity_used",
        "step_zero_ranked",
        "later_states_require_pareto_safety_vs_step_zero",
        "sealed_metrics_used",
        "internal_numeric_matching_used",
        "callback_metrics_are_durable_without_factor_pack_retention",
        "maximum_retained_factor_packs",
    }
    if (
        set(frontier) != frontier_fields
        or frontier.get("schema") != tournament.FRONTIER_SCHEMA
        or frontier.get("evaluation_contract_sha256")
        != contract.evaluation_contract_sha256
        or frontier.get("completed_arms") != list(tournament.ARM_IDS)
        or frontier.get("observed_callback_count") != tournament.EXPECTED_CALLBACK_COUNT
        or frontier.get("pending_callback_count") != 0
        or not isinstance(observations, list)
        or len(observations) != tournament.EXPECTED_CALLBACK_COUNT
        or not isinstance(retained, list)
        or frontier.get("retained_count") != tournament.MAX_RETAINED_STATES
        or len(retained) != tournament.MAX_RETAINED_STATES
        or frontier.get("sealed_metrics_used") is not False
        or frontier.get("internal_numeric_matching_used") is not False
        or frontier.get("callback_metrics_are_durable_without_factor_pack_retention")
        is not True
        or frontier.get("maximum_retained_factor_packs")
        != tournament.MAX_RETAINED_STATES
        or frontier.get("distinct_full_v3_states_only") is not True
        or frontier.get("portable_content_identity_used") is not True
        or frontier.get("step_zero_ranked") is not True
        or frontier.get("later_states_require_pareto_safety_vs_step_zero") is not True
    ):
        raise GuidedWinnerPublicationError(
            "Candidate frontier is incomplete, unlocked, or malformed."
        )
    observation_fields = {
        "ordinal",
        "arm_id",
        "arm_index",
        "step",
        "checkpoint_sha256",
        "runtime_factor_state_sha256",
        "portable_factor_state_sha256",
        "pareto_safe_vs_step0",
        "metrics",
        "metrics_sha256",
        "rank",
        "eligible_for_retention",
        "disposition",
        "retained_after_callback",
        "factor_pack_created",
        "evicted_portable_factor_state_sha256s",
        "terminal_verified",
        "terminal_pareto_safe_vs_step0",
    }
    active: dict[str, dict[str, Any]] = {}
    metric_objects: dict[int, CausalBehaviorMetrics] = {}
    for ordinal, row in enumerate(observations):
        expected_arm_index = ordinal // len(tournament.CHECKPOINT_STEPS)
        expected_step = tournament.CHECKPOINT_STEPS[
            ordinal % len(tournament.CHECKPOINT_STEPS)
        ]
        expected_arm = tournament.ARM_IDS[expected_arm_index]
        if (
            not isinstance(row, Mapping)
            or set(row) != observation_fields
            or row.get("ordinal") != ordinal
            or row.get("arm_id") != expected_arm
            or row.get("arm_index") != expected_arm_index
            or row.get("step") != expected_step
            or row.get("disposition") not in tournament.FINAL_CALLBACK_DISPOSITIONS
            or row.get("disposition") == "error-inconsistent-duplicate"
            or not isinstance(row.get("pareto_safe_vs_step0"), bool)
            or (expected_step == 0 and row.get("pareto_safe_vs_step0") is not True)
            or row.get("terminal_verified") is not True
            or row.get("terminal_pareto_safe_vs_step0")
            != row.get("pareto_safe_vs_step0")
            or not isinstance(row.get("metrics"), Mapping)
            or row.get("metrics_sha256") != canonical_sha256(row["metrics"])
            or not isinstance(row.get("rank"), list)
            or not isinstance(row.get("eligible_for_retention"), bool)
            or not isinstance(row.get("retained_after_callback"), bool)
            or not isinstance(row.get("factor_pack_created"), bool)
            or not isinstance(row.get("evicted_portable_factor_state_sha256s"), list)
        ):
            raise GuidedWinnerPublicationError(
                "Candidate frontier contains unauthenticated callback evidence."
            )
        for field in (
            "checkpoint_sha256",
            "runtime_factor_state_sha256",
            "portable_factor_state_sha256",
        ):
            _sha256(row[field], f"candidate_frontier[{ordinal}].{field}")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in row["evicted_portable_factor_state_sha256s"]
        ):
            raise GuidedWinnerPublicationError(
                "Candidate frontier contains an invalid eviction identity."
            )
        metrics = _frontier_metrics(row["metrics"], f"callback[{ordinal}]")
        metric_objects[ordinal] = metrics
        rank = tuple(tournament.tournament_behavior_rank(row["metrics"]))
        eligible = expected_step == 0 or row["pareto_safe_vs_step0"]
        if row["rank"] != list(rank) or row["eligible_for_retention"] is not eligible:
            raise GuidedWinnerPublicationError(
                "Candidate frontier rank or retention eligibility was forged."
            )
        portable = row["portable_factor_state_sha256"]
        disposition = row["disposition"]
        evicted = row["evicted_portable_factor_state_sha256s"]
        if not eligible:
            if (
                disposition != "ineligible-non-pareto"
                or row["retained_after_callback"] is not False
                or row["factor_pack_created"] is not False
                or evicted
            ):
                raise GuidedWinnerPublicationError(
                    "Candidate frontier non-Pareto disposition was forged."
                )
            continue
        previous = active.get(portable)
        if previous is not None:
            if (
                disposition != "duplicate-retained-state"
                or row["retained_after_callback"] is not True
                or row["factor_pack_created"] is not False
                or evicted
                or previous["rank"] != row["rank"]
                or previous["metrics"] != row["metrics"]
            ):
                raise GuidedWinnerPublicationError(
                    "Candidate frontier duplicate-state disposition was forged."
                )
            continue
        candidate_order = (*rank, expected_arm_index, expected_step, portable)
        keep = sorted(
            (
                *((value["order"], key) for key, value in active.items()),
                (candidate_order, portable),
            ),
            key=lambda value: value[0],
        )[: tournament.MAX_RETAINED_STATES]
        keep_hashes = {value[1] for value in keep}
        if portable not in keep_hashes:
            if (
                disposition != "not-retained-rank-cutoff"
                or row["retained_after_callback"] is not False
                or row["factor_pack_created"] is not False
                or evicted
            ):
                raise GuidedWinnerPublicationError(
                    "Candidate frontier rank-cutoff disposition was forged."
                )
            continue
        active[portable] = {
            "order": candidate_order,
            "observation": dict(row),
            "rank": list(row["rank"]),
            "metrics": dict(row["metrics"]),
        }
        expected_evicted = [key for key in tuple(active) if key not in keep_hashes]
        if (
            disposition != "retained-top-three"
            or row["retained_after_callback"] is not True
            or row["factor_pack_created"] is not True
            or evicted != expected_evicted
        ):
            raise GuidedWinnerPublicationError(
                "Candidate frontier top-three disposition was forged."
            )
        for evicted_digest in expected_evicted:
            del active[evicted_digest]
    retained_fields = {
        "arm_id",
        "arm_index",
        "step",
        "rank",
        "runtime_factor_state_sha256",
        "portable_factor_state_sha256",
        "checkpoint_sha256",
        "metrics",
        "factor_pack",
        "factor_pack_sha256",
    }
    ordered_active = sorted(active.items(), key=lambda value: value[1]["order"])
    if len(ordered_active) != tournament.MAX_RETAINED_STATES:
        raise GuidedWinnerPublicationError("Candidate frontier did not reconstruct.")
    retained_observations: list[Mapping[str, Any]] = []
    for index, (portable, state) in enumerate(ordered_active):
        row = retained[index]
        observation = state["observation"]
        if (
            not isinstance(row, Mapping)
            or set(row) != retained_fields
            or row["portable_factor_state_sha256"] != portable
            or row["arm_id"] != observation["arm_id"]
            or row["arm_index"] != observation["arm_index"]
            or row["step"] != observation["step"]
            or row["rank"] != observation["rank"]
            or row["runtime_factor_state_sha256"]
            != observation["runtime_factor_state_sha256"]
            or row["checkpoint_sha256"] != observation["checkpoint_sha256"]
            or row["metrics"] != observation["metrics"]
        ):
            raise GuidedWinnerPublicationError(
                "Retained frontier state differs from its callback evidence."
            )
        pack_path = _resolve(
            row["factor_pack"],
            base=contract.path.parent,
            name=f"candidate frontier retained[{index}] factor pack",
        )
        pack_sha = _sha256(
            row["factor_pack_sha256"], f"retained[{index}].factor_pack_sha256"
        )
        _bound_file(pack_path, pack_sha, f"retained[{index}] factor pack")
        retained_observations.append(observation)
    forbidden_keys = {
        "factor_payload",
        "factor_state",
        "factor_tensors",
        "model",
        "model_state",
        "optimizer",
        "optimizer_state",
        "parent",
        "parent_state",
        "model_checkpoint",
        "sealed_metrics",
    }

    def contains_forbidden(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                str(key) in forbidden_keys or contains_forbidden(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_forbidden(item) for item in value)
        return False

    if contains_forbidden(frontier):
        raise GuidedWinnerPublicationError(
            "Candidate frontier contains forbidden model or sealed state."
        )
    winner_observation = retained_observations[0]
    step_zero_matches = [
        (ordinal, row)
        for ordinal, row in enumerate(observations)
        if row["arm_id"] == winner_observation["arm_id"] and row["step"] == 0
    ]
    if len(step_zero_matches) != 1:
        raise GuidedWinnerPublicationError(
            "Winner does not have exactly one authenticated step-zero baseline."
        )
    step_zero_ordinal, step_zero = step_zero_matches[0]
    winner_ordinal = int(winner_observation["ordinal"])
    winner_metrics = metric_objects[winner_ordinal]
    step_zero_metrics = metric_objects[step_zero_ordinal]
    pairs = tuple(
        zip(
            winner_metrics.pareto_objectives,
            step_zero_metrics.pareto_objectives,
            strict=True,
        )
    )
    objectives_equal = all(left == right for left, right in pairs)
    dominates_step_zero = all(left >= right for left, right in pairs) and any(
        left > right for left, right in pairs
    )
    declared_safe = winner_observation["pareto_safe_vs_step0"] is True
    independently_safe = objectives_equal or dominates_step_zero
    admission = {
        "passed": declared_safe and independently_safe,
        "winner_arm_id": winner_observation["arm_id"],
        "winner_step": winner_observation["step"],
        "winner_callback_ordinal": winner_ordinal,
        "step_zero_callback_ordinal": step_zero_ordinal,
        "declared_pareto_safe_vs_step0": declared_safe,
        "objectives_equal_to_step_zero": objectives_equal,
        "dominates_step_zero": dominates_step_zero,
        "winner_metrics_sha256": winner_observation["metrics_sha256"],
        "step_zero_metrics_sha256": step_zero["metrics_sha256"],
        "recomputed_from_authenticated_output_metrics": True,
        "internal_numeric_matching_consulted": False,
    }
    admission["decision_sha256"] = canonical_sha256(admission)
    binding = {
        "schema": tournament.FRONTIER_SCHEMA,
        "path": str(path),
        "sha256": digest,
        "observed_callback_count": tournament.EXPECTED_CALLBACK_COUNT,
        "pending_callback_count": 0,
        "completed_arms": list(tournament.ARM_IDS),
    }
    return binding, admission


def _winner_payload(
    contract: tournament.AuthenticatedCandidateTournament,
    run_report: Mapping[str, Any],
    winner_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        run_report.get("schema") != tournament.RUN_SCHEMA
        or run_report.get("status") != "candidate-winner-selected"
        or run_report.get("evaluation_contract_sha256")
        != contract.evaluation_contract_sha256
        or run_report.get("sealed_forward_count") != 0
        or run_report.get("sealed_evaluation_accessed") is not False
        or run_report.get("model_checkpoint_written") is not False
        or run_report.get("publication_attempted") is not False
        or run_report.get("stop_after_candidate_winner") is not True
        or canonical_json_bytes(run_report.get("winner_selection"))
        != canonical_json_bytes(winner_selection)
    ):
        raise GuidedWinnerPublicationError(
            "Tournament terminal report is not the sealed-free winner boundary."
        )
    winner = winner_selection.get("winner")
    required_winner = {
        "arm_id",
        "arm_index",
        "step",
        "rank",
        "runtime_factor_state_sha256",
        "portable_factor_state_sha256",
        "checkpoint_sha256",
        "metrics",
        "factor_pack",
        "factor_pack_sha256",
    }
    required_selection = {
        "schema",
        "evaluation_contract_sha256",
        "winner",
        "top_three_factor_state_sha256s",
        "top_three_portable_factor_state_sha256s",
        "candidate_only",
        "sealed_evaluation_accessed",
        "model_checkpoint_written",
        "publication_attempted",
        "stop_reason",
        "candidate_frontier",
    }
    if (
        set(winner_selection) != required_selection
        or winner_selection.get("schema") != tournament.WINNER_SCHEMA
        or winner_selection.get("evaluation_contract_sha256")
        != contract.evaluation_contract_sha256
        or winner_selection.get("candidate_only") is not True
        or winner_selection.get("sealed_evaluation_accessed") is not False
        or winner_selection.get("model_checkpoint_written") is not False
        or winner_selection.get("publication_attempted") is not False
        or winner_selection.get("stop_reason")
        != "candidate-winner-selected-before-sealed"
        or not isinstance(winner, Mapping)
        or set(winner) != required_winner
    ):
        raise GuidedWinnerPublicationError("Winner selection is malformed or unlocked.")
    frontier_binding, pareto_admission = _authenticated_frontier_binding(
        contract,
        run_report,
        winner_selection,
    )
    frontier = _bound_json(
        Path(frontier_binding["path"]),
        frontier_binding["sha256"],
        "candidate frontier",
    )
    retained = frontier["retained"]
    if (
        canonical_json_bytes(retained[0]) != canonical_json_bytes(winner)
        or [row.get("runtime_factor_state_sha256") for row in retained]
        != winner_selection.get("top_three_factor_state_sha256s")
        or [row.get("portable_factor_state_sha256") for row in retained]
        != winner_selection.get("top_three_portable_factor_state_sha256s")
    ):
        raise GuidedWinnerPublicationError(
            "Winner selection differs from the authenticated callback frontier."
        )
    factor_path = _resolve(
        winner["factor_pack"],
        base=contract.path.parent,
        name="winner.factor_pack",
    )
    factor_sha = _sha256(winner["factor_pack_sha256"], "winner.factor_pack_sha256")
    _bound_file(factor_path, factor_sha, "winner.factor_pack")
    try:
        loaded = torch.load(factor_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise GuidedWinnerPublicationError(
            "Winner factor pack cannot load safely."
        ) from error
    if not isinstance(loaded, Mapping):
        raise GuidedWinnerPublicationError("Winner factor pack is not a mapping.")
    factor_state_sha = _sha256(
        winner["runtime_factor_state_sha256"],
        "winner.runtime_factor_state_sha256",
    )
    try:
        GuidedCheckpointRecoverySink._validate_v3_factor_payload(
            loaded,
            expected_factor_state_sha256=factor_state_sha,
        )
    except Exception as error:
        raise GuidedWinnerPublicationError(
            "Winner factor pack failed v3 authentication."
        ) from error
    portable_sha = _sha256(
        winner["portable_factor_state_sha256"],
        "winner.portable_factor_state_sha256",
    )
    if portable_v3_factor_state_sha256(loaded) != portable_sha:
        raise GuidedWinnerPublicationError("Winner portable factor identity changed.")
    metrics = winner.get("metrics")
    if not isinstance(metrics, Mapping):
        raise GuidedWinnerPublicationError("Winner candidate metrics are absent.")
    return (
        {
            "arm_id": str(winner["arm_id"]),
            "arm_index": int(winner["arm_index"]),
            "step": int(winner["step"]),
            "rank": list(winner["rank"]),
            "runtime_factor_state_sha256": factor_state_sha,
            "portable_factor_state_sha256": portable_sha,
            "checkpoint_sha256": _sha256(
                winner["checkpoint_sha256"], "winner.checkpoint_sha256"
            ),
            "metrics": dict(metrics),
            "metrics_sha256": canonical_sha256(metrics),
            "factor_pack": str(factor_path),
            "factor_pack_sha256": factor_sha,
            "candidate_frontier": frontier_binding,
            "tournament_pareto_admission": pareto_admission,
        },
        dict(loaded),
    )


def build_winner_publication_preregistration_payload(
    *,
    tournament_preregistration: Path,
    tournament_preregistration_sha256: str,
    tournament_run_report: Path,
    tournament_run_report_sha256: str,
    winner_selection: Path,
    winner_selection_sha256: str,
    sealed_consumption_marker: Path,
    dtype: str,
    parent_device_map: str,
    target_device_map: str,
    builder_path: Path,
) -> tuple[
    dict[str, Any],
    tournament.AuthenticatedCandidateTournament,
    dict[str, Any],
    dict[str, Any],
]:
    tournament_path = _bound_file(
        tournament_preregistration,
        tournament_preregistration_sha256,
        "tournament preregistration",
    )
    tournament_sha = _sha256(
        tournament_preregistration_sha256,
        "tournament_preregistration_sha256",
    )
    source = tournament.authenticate_candidate_tournament_preregistration(
        tournament_path,
        tournament_sha,
    )
    run_path = _bound_file(
        tournament_run_report,
        tournament_run_report_sha256,
        "tournament run report",
    )
    run_sha = _sha256(
        tournament_run_report_sha256,
        "tournament_run_report_sha256",
    )
    run_report = _bound_json(run_path, run_sha, "tournament run report")
    winner_path = _bound_file(
        winner_selection,
        winner_selection_sha256,
        "winner selection",
    )
    winner_sha = _sha256(winner_selection_sha256, "winner_selection_sha256")
    winner_selection_payload = _bound_json(winner_path, winner_sha, "winner selection")
    winner, factor_payload = _winner_payload(
        source,
        run_report,
        winner_selection_payload,
    )
    source_runtime = source.runtime
    if (
        dtype != source_runtime["dtype"]
        or parent_device_map != source_runtime["parent_device_map"]
        or target_device_map != source_runtime["target_device_map"]
    ):
        raise GuidedWinnerPublicationError(
            "Winner publication runtime must exactly match the authenticated "
            "tournament runtime."
        )
    marker = _resolve(
        sealed_consumption_marker,
        base=Path.cwd(),
        name="sealed consumption marker",
    )
    if marker.name != "sealed_consumption.json":
        raise GuidedWinnerPublicationError(
            "Sealed marker must be a real path named sealed_consumption.json."
        )
    sealed = source.splits["sealed"]
    if sealed.expected_count != 32:
        raise GuidedWinnerPublicationError(
            "Winner publication requires exact sealed32."
        )
    splits = {
        name: _split_semantics(source.splits[name])
        for name in ("fit", "candidate", "sealed")
    }
    implementation = _implementation_hashes(builder_path=builder_path)
    runtime = {
        "dtype": dtype,
        "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
        "parent_device_map": parent_device_map,
        "target_device_map": target_device_map,
        "explicit_execute_required": True,
    }
    semantic_contract = {
        "tournament_evaluation_contract_sha256": source.evaluation_contract_sha256,
        "source_guided_evaluation_contract_sha256": (
            source.source.evaluation_contract_sha256
        ),
        "winner_portable_factor_state_sha256": winner["portable_factor_state_sha256"],
        "winner_candidate_metrics_sha256": winner["metrics_sha256"],
        "candidate_frontier_sha256": winner["candidate_frontier"]["sha256"],
        "observed_callback_count": winner["candidate_frontier"][
            "observed_callback_count"
        ],
        "tournament_pareto_admission_decision_sha256": winner[
            "tournament_pareto_admission"
        ]["decision_sha256"],
        "splits": splits,
        "sealed_forward_limit": 1,
        "post_bake_split_order": ["fit", "candidate"],
        "post_bake_sealed_forbidden": True,
        "runtime": runtime,
        "implementation": implementation,
    }
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "tournament": {
            "preregistration": str(tournament_path),
            "preregistration_sha256": tournament_sha,
            "run_report": str(run_path),
            "run_report_sha256": run_sha,
            "winner_selection": str(winner_path),
            "winner_selection_sha256": winner_sha,
            "candidate_frontier": dict(winner["candidate_frontier"]),
        },
        "source_guided": {
            "preregistration": str(source.source.path),
            "preregistration_sha256": source.source.file_sha256,
            "evaluation_contract_sha256": source.source.evaluation_contract_sha256,
        },
        "winner": winner,
        "splits": splits,
        "sealed_consumption": {
            "marker": str(marker),
            "schema": SEALED_CONSUMPTION_SCHEMA,
            "must_be_absent_at_preflight": True,
            "exclusive_create_immediately_before_forward": True,
            "maximum_forward_count": 1,
        },
        "runtime": runtime,
        "implementation": implementation,
        "semantic_contract": semantic_contract,
        "semantic_contract_sha256": canonical_sha256(semantic_contract),
        "model_free_preregistration": True,
        "model_weights_loaded": False,
        "gpu_work_launched": False,
        "publication_authorized_before_sealed_pass": False,
    }
    return payload, source, winner, factor_payload


@dataclass(frozen=True, slots=True)
class AuthenticatedWinnerPublication:
    path: Path
    file_sha256: str
    tournament: tournament.AuthenticatedCandidateTournament
    winner: Mapping[str, Any]
    factor_payload: Mapping[str, Any]
    sealed_marker: Path
    runtime: Mapping[str, Any]
    raw: Mapping[str, Any]

    def preflight_report(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "status": "preflight-passed",
            "preregistration": str(self.path),
            "preregistration_sha256": self.file_sha256,
            "semantic_contract_sha256": self.raw["semantic_contract_sha256"],
            "evaluation_contract_sha256": self.raw["semantic_contract_sha256"],
            "winner": dict(self.winner),
            "sealed": dict(self.raw["splits"]["sealed"]),
            "sealed_consumption_marker": str(self.sealed_marker),
            "sealed_marker_absent": not self.sealed_marker.exists(),
            "runtime": dict(self.runtime),
            "models_loaded": False,
            "gpu_work_launched": False,
            "sealed_forward_count": 0,
            "publication_attempted": False,
        }


def authenticate_winner_publication_preregistration(
    path: Path | str,
    expected_sha256: str,
) -> AuthenticatedWinnerPublication:
    source = _bound_file(path, expected_sha256, "winner publication preregistration")
    expected = _sha256(expected_sha256, "preregistration_sha256")
    raw = _bound_json(source, expected, "winner publication preregistration")
    _strict(
        raw,
        {
            "schema",
            "tournament",
            "source_guided",
            "winner",
            "splits",
            "sealed_consumption",
            "runtime",
            "implementation",
            "semantic_contract",
            "semantic_contract_sha256",
            "model_free_preregistration",
            "model_weights_loaded",
            "gpu_work_launched",
            "publication_authorized_before_sealed_pass",
        },
        "winner publication preregistration",
    )
    if raw.get("schema") != PREREGISTRATION_SCHEMA:
        raise GuidedWinnerPublicationError("Winner publication schema changed.")
    tournament_row = _strict(
        raw["tournament"],
        {
            "preregistration",
            "preregistration_sha256",
            "run_report",
            "run_report_sha256",
            "winner_selection",
            "winner_selection_sha256",
            "candidate_frontier",
        },
        "tournament",
    )
    marker_row = _strict(
        raw["sealed_consumption"],
        {
            "marker",
            "schema",
            "must_be_absent_at_preflight",
            "exclusive_create_immediately_before_forward",
            "maximum_forward_count",
        },
        "sealed_consumption",
    )
    builder_path = Path(__file__).resolve().with_name(BUILD_FILENAME)
    expected_payload, tournament_contract, winner, factor_payload = (
        build_winner_publication_preregistration_payload(
            tournament_preregistration=Path(str(tournament_row["preregistration"])),
            tournament_preregistration_sha256=str(
                tournament_row["preregistration_sha256"]
            ),
            tournament_run_report=Path(str(tournament_row["run_report"])),
            tournament_run_report_sha256=str(tournament_row["run_report_sha256"]),
            winner_selection=Path(str(tournament_row["winner_selection"])),
            winner_selection_sha256=str(tournament_row["winner_selection_sha256"]),
            sealed_consumption_marker=Path(str(marker_row["marker"])),
            dtype=str(raw["runtime"]["dtype"]),
            parent_device_map=str(raw["runtime"]["parent_device_map"]),
            target_device_map=str(raw["runtime"]["target_device_map"]),
            builder_path=builder_path,
        )
    )
    if canonical_json_bytes(raw) != canonical_json_bytes(expected_payload):
        raise GuidedWinnerPublicationError(
            "Winner publication preregistration differs from authenticated sources."
        )
    marker = _resolve(
        marker_row["marker"],
        base=source.parent,
        name="sealed consumption marker",
    )
    return AuthenticatedWinnerPublication(
        path=source,
        file_sha256=expected,
        tournament=tournament_contract,
        winner=MappingProxyType(dict(winner)),
        factor_payload=MappingProxyType(dict(factor_payload)),
        sealed_marker=marker,
        runtime=MappingProxyType(dict(raw["runtime"])),
        raw=MappingProxyType(dict(raw)),
    )


def _consume_sealed_marker(
    contract: AuthenticatedWinnerPublication,
    output_root: Path,
) -> dict[str, Any]:
    marker = contract.sealed_marker
    if (
        marker.parent != contract.path.parent
        or marker.name != "sealed_consumption.json"
    ):
        raise GuidedWinnerPublicationError(
            "Sealed marker escaped the preregistration evidence directory."
        )
    payload = {
        "schema": SEALED_CONSUMPTION_SCHEMA,
        "status": "consumed-before-forward",
        "preregistration_sha256": contract.file_sha256,
        "semantic_contract_sha256": contract.raw["semantic_contract_sha256"],
        "winner_portable_factor_state_sha256": contract.winner[
            "portable_factor_state_sha256"
        ],
        "sealed_split_sha256": contract.tournament.splits["sealed"].sha256,
        "sealed_forward_limit": 1,
        "output_root": str(output_root),
    }
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise GuidedWinnerPublicationError(
            "Sealed suite was already consumed for this preregistration."
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            parent_descriptor = os.open(marker.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError:
            pass
    except BaseException:
        # Once exclusive creation succeeds, never remove the marker: a crash or
        # partial write burns sealed access fail-closed.
        raise
    return {
        **payload,
        "marker": str(marker),
        "marker_file_sha256": stage._file_sha256(marker),
        "exclusive_create_succeeded": True,
    }


def _method_gate(
    metrics: Any,
    *,
    winner_step: int,
    tournament_pareto_admission: bool,
) -> tuple[bool, dict[str, Any]]:
    payload = stage._metrics_dict(metrics)
    checks = {
        "winner_is_post_step_zero": winner_step > 0,
        "candidate_has_one_lcp_at_least_4_rollout": int(payload["lcp_at_least_4_count"])
        >= 1,
        "tournament_later_state_required_pareto_safety_vs_step_zero": (
            tournament_pareto_admission
        ),
    }
    decision = {
        "passed": all(checks.values()),
        "checks": checks,
        "winner_step": winner_step,
        "candidate_lcp_at_least_4_count": payload["lcp_at_least_4_count"],
        "internal_numeric_matching_consulted": False,
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    return bool(decision["passed"]), decision


class ProductionWinnerPublicationRuntime(guided.ProductionGuidedRuntime):
    """Production heavy boundary with an auditable sealed forward counter."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sealed_forward_count = 0

    def evaluate_publication_split(
        self,
        model: Any,
        parent: Any,
        contract: AuthenticatedWinnerPublication,
        split_name: str,
    ) -> Any:
        if split_name not in {"fit", "candidate", "sealed"}:
            raise GuidedWinnerPublicationError("Unknown publication split.")
        if split_name == "sealed":
            self.sealed_forward_count += 1
            if self.sealed_forward_count != 1:
                raise GuidedWinnerPublicationError(
                    "Sealed evaluation exceeded one call."
                )
        module = importlib.import_module(guided.GUIDED_TRAINER_MODULE)
        loader = getattr(module, "load_behavior_domain_suite")
        evaluator = getattr(module, "evaluate_behavior_domain")
        binding = contract.tournament.splits[split_name]
        suite = loader(
            split_name,
            binding.payload,
            artifact_sha256=binding.sha256,
        )
        schedule = contract.tournament.schedule
        return evaluator(
            parent,
            model,
            suite,
            horizon=schedule.evaluation_horizon,
            top_k=schedule.top_k,
            split={
                "fit": "calibration-exploratory",
                "candidate": "calibration-selection",
                "sealed": "heldout-evaluation",
            }[split_name],
        )

    def independent_reload_certify(
        self,
        child_path: Path,
        exact_unique_parameters: int,
    ) -> Mapping[str, Any]:
        return retention._independent_accepted_child_authenticator(
            child_path,
            exact_unique_parameters,
        )


def execute_winner_publication(
    contract: AuthenticatedWinnerPublication,
    output_root: Path | str,
    *,
    dtype: str,
    parent_device_map: str,
    target_device_map: str,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Execute the exact winner once; sealed access is permanently one-shot."""

    if type(contract) is not AuthenticatedWinnerPublication:
        raise GuidedWinnerPublicationError(
            "Execution requires the canonical authenticated winner contract."
        )
    refreshed = authenticate_winner_publication_preregistration(
        contract.path,
        contract.file_sha256,
    )
    if canonical_json_bytes(refreshed.raw) != canonical_json_bytes(contract.raw):
        raise GuidedWinnerPublicationError("Winner sources changed before execution.")
    expected_runtime = refreshed.runtime
    if (
        dtype != expected_runtime["dtype"]
        or parent_device_map != expected_runtime["parent_device_map"]
        or target_device_map != expected_runtime["target_device_map"]
    ):
        raise GuidedWinnerPublicationError(
            "Execution runtime differs from preregistration."
        )
    if refreshed.sealed_marker.exists():
        raise GuidedWinnerPublicationError(
            "Sealed suite was already consumed; refusing model access."
        )
    cuda_preflight = tournament.validate_candidate_tournament_cuda_preflight(
        refreshed.tournament,
        dtype=dtype,
        parent_device_map=parent_device_map,
        target_device_map=target_device_map,
    )
    runner = (
        ProductionWinnerPublicationRuntime(
            dtype=dtype,
            parent_device_map=parent_device_map,
            target_device_map=target_device_map,
        )
        if runtime is None
        else runtime
    )
    base = dataclasses.replace(
        refreshed.tournament.source.execution_contract,
        splits=refreshed.tournament.splits,
        training=refreshed.tournament.schedule.as_dict(),
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise GuidedWinnerPublicationError("output_root must not already exist.")
    output.mkdir(parents=True)
    parent: Any = None
    target: Any = None
    tokenizer: Any = None
    compiled: Any = None
    transaction: Any = None
    compiled_sha: str | None = None
    published = False
    marker_receipt: Mapping[str, Any] | None = None
    report: dict[str, Any] = {
        **refreshed.preflight_report(),
        "status": "running",
        "output_root": str(output),
        "cuda_preflight": dict(cuda_preflight),
    }
    try:
        prepare_cache = getattr(runner, "prepare_compiler_cache", None)
        if callable(prepare_cache):
            cache = prepare_cache(base)
            if cache is not None:
                report["compiled_child_cache"] = dict(cache)
        parent, target, tokenizer = runner.load_models(base)
        report.update({"models_loaded": True, "gpu_work_launched": True})
        compiled = runner.compile(parent, target, tokenizer, base)
        compiler_report = stage._compiled_report_dict(compiled)
        compiled_sha = runner.state_sha256(compiled.child_model)
        stage._validate_continuation_compilation(base, compiler_report, compiled_sha)
        finalize = getattr(runner, "finalize_compiled_base", None)
        if callable(finalize):
            compiled = finalize(compiled, base)
            if runner.state_sha256(compiled.child_model) != compiled_sha:
                raise GuidedWinnerPublicationError(
                    "Compiler finalization changed the compiled child."
                )
        transaction = runner.install_transaction(compiled)
        inherited_install = transaction.installation_report()
        source_restore = runner.restore_source_v2(
            transaction, refreshed.tournament.source
        )
        guided_install = runner.install_guided_transaction(
            transaction,
            refreshed.tournament.source,
        )
        trainer = importlib.import_module(guided.GUIDED_TRAINER_MODULE)
        source_v2_state = capture_target_native_factor_state(transaction)
        live_binding = authenticate_guided_factor_sources(
            transaction,
            runner._guided_tranche,
            source_v2_state,
        )
        del source_v2_state
        gc.collect()
        rebound = rebind_recovered_v3_factor_payload(
            refreshed.factor_payload,
            live_binding,
        )
        set_stage = getattr(runner._guided_tranche, "set_training_stage", None)
        if not callable(set_stage):
            raise GuidedWinnerPublicationError("Guided tranche has no joint stage.")
        set_stage("joint")
        restore = getattr(trainer, "restore_target_native_factor_payload", None)
        if not callable(restore):
            raise GuidedWinnerPublicationError("Public full-v3 restore is unavailable.")
        restored = restore(
            transaction,
            rebound,
            guided_tranche=runner._guided_tranche,
            expected_guided_source_binding=live_binding,
        )
        if (
            getattr(restored, "schema", None) != BEHAVIOR_FACTOR_PACK_V3_SCHEMA
            or getattr(restored, "factor_state_sha256", None)
            != rebound["factor_state_sha256"]
            or portable_v3_factor_state_sha256(rebound, live_binding)
            != refreshed.winner["portable_factor_state_sha256"]
        ):
            raise GuidedWinnerPublicationError(
                "Live winner restore changed factor state."
            )
        report.update(
            {
                "compiler_report": compiler_report,
                "inherited_installation": inherited_install,
                "source_v2_restore": dict(source_restore),
                "guided_installation": dict(guided_install),
                "runtime_rebound_factor_state_sha256": rebound["factor_state_sha256"],
                "winner_portable_factor_state_sha256": refreshed.winner[
                    "portable_factor_state_sha256"
                ],
            }
        )

        candidate_metrics = runner.evaluate_publication_split(
            compiled.child_model,
            parent,
            refreshed,
            "candidate",
        )
        candidate_payload = stage._metrics_dict(candidate_metrics)
        if canonical_json_bytes(candidate_payload) != canonical_json_bytes(
            refreshed.winner["metrics"]
        ):
            raise GuidedWinnerPublicationError(
                "Restored winner candidate behavior differs from tournament evidence."
            )
        candidate_passed, candidate_decision = stage._behavior_gate(
            candidate_metrics,
            base.thresholds["candidate"],
        )
        method_passed, method_decision = _method_gate(
            candidate_metrics,
            winner_step=int(refreshed.winner["step"]),
            tournament_pareto_admission=bool(
                refreshed.winner["tournament_pareto_admission"]["passed"]
            ),
        )
        report["pre_sealed_candidate_decision"] = candidate_decision
        report["guided_method_decision"] = method_decision
        if not candidate_passed or not method_passed:
            runner.rollback_guided_installation(transaction)
            transaction.rollback()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_sha:
                raise GuidedWinnerPublicationError(
                    "Candidate rejection rollback failed."
                )
            report.update(
                {
                    "status": "rejected",
                    "reason": "winner-candidate-recheck-rejected",
                    "sealed_forward_count": 0,
                    "sealed_evaluation_accessed": False,
                    "sealed_marker_consumed": False,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report

        if getattr(runner, "sealed_forward_count", 0) != 0:
            raise GuidedWinnerPublicationError("Sealed forward occurred before marker.")
        marker_receipt = _consume_sealed_marker(refreshed, output)
        sealed_metrics = runner.evaluate_publication_split(
            compiled.child_model,
            parent,
            refreshed,
            "sealed",
        )
        if getattr(runner, "sealed_forward_count", None) != 1:
            raise GuidedWinnerPublicationError(
                "Sealed forward count is not exactly one."
            )
        sealed_passed, sealed_decision = stage._behavior_gate(
            sealed_metrics,
            base.thresholds["sealed"],
        )
        pre_bake_decisions = {
            "fit": None,
            "candidate": candidate_decision,
            "sealed": sealed_decision,
        }
        report.update(
            {
                "sealed_evaluation": {
                    "forward_count": 1,
                    "one_shot_marker_consumed": True,
                    "marker_receipt": dict(marker_receipt),
                    "post_bake_sealed_forward_count": 0,
                },
                "pre_bake_behavior_decisions": pre_bake_decisions,
                "sealed_evaluation_accessed": True,
            }
        )
        if not sealed_passed:
            runner.rollback_guided_installation(transaction)
            transaction.rollback()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_sha:
                raise GuidedWinnerPublicationError("Sealed rejection rollback failed.")
            report.update(
                {
                    "status": "rejected",
                    "reason": "one-shot-sealed-gate-rejected",
                    "sealed_forward_count": 1,
                    "sealed_marker_consumed": True,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report

        sealed_payload = stage._metrics_dict(sealed_metrics)
        gate_evidence = {
            "preregistration_sha256": refreshed.file_sha256,
            "winner_portable_factor_state_sha256": refreshed.winner[
                "portable_factor_state_sha256"
            ],
            "candidate_decision_sha256": candidate_decision["decision_sha256"],
            "method_decision_sha256": method_decision["decision_sha256"],
            "sealed_decision_sha256": sealed_decision["decision_sha256"],
            "sealed_marker_file_sha256": marker_receipt["marker_file_sha256"],
        }
        gate_evidence_sha = canonical_sha256(gate_evidence)
        receipt = ExternalBehaviorGateReceipt(
            evidence_sha256=gate_evidence_sha,
            evaluation_contract_sha256=refreshed.raw["semantic_contract_sha256"],
            heldout_action_count=int(sealed_payload["action_count"]),
            free_running_rollout_count=int(sealed_payload["rollout_count"]),
            passed=True,
            internal_similarity_only=False,
        )
        handoff = runner.handoff_guided_for_publication(
            transaction,
            refreshed.tournament.source,
        )
        publication = transaction.bake_for_publication(receipt)
        report.update(
            {
                "external_behavior_gate_evidence": gate_evidence,
                "external_behavior_gate_evidence_sha256": gate_evidence_sha,
                "guided_publication_handoff": dict(handoff),
                "publication": publication,
            }
        )

        post_decisions: dict[str, Any] = {
            "fit": None,
            "candidate": None,
            "sealed": None,
        }
        post_candidate_metrics: Any = None
        for name in ("fit", "candidate"):
            metrics = runner.evaluate_publication_split(
                compiled.child_model,
                parent,
                refreshed,
                name,
            )
            passed, decision = stage._behavior_gate(metrics, base.thresholds[name])
            post_decisions[name] = decision
            if not passed:
                transaction.restore_prepublication_native()
                rollback_sha = runner.state_sha256(compiled.child_model)
                if rollback_sha != compiled_sha:
                    raise GuidedWinnerPublicationError("Post-bake rollback failed.")
                report.update(
                    {
                        "status": "rejected",
                        "reason": f"post-bake-{name}-gate-rejected",
                        "post_bake_behavior_decisions": post_decisions,
                        "sealed_forward_count": 1,
                        "sealed_marker_consumed": True,
                        "rollback_exact": True,
                        "rollback_state_sha256": rollback_sha,
                        "standalone_compaction_passed": False,
                        "model_checkpoint_written": False,
                    }
                )
                stage._atomic_json(output / "run_report.json", report)
                return report
            if name == "candidate":
                post_candidate_metrics = metrics
        post_method_passed, post_method_decision = _method_gate(
            post_candidate_metrics,
            winner_step=int(refreshed.winner["step"]),
            tournament_pareto_admission=bool(
                refreshed.winner["tournament_pareto_admission"]["passed"]
            ),
        )
        report["post_bake_behavior_decisions"] = post_decisions
        report["post_bake_guided_method_decision"] = post_method_decision
        if not post_method_passed or getattr(runner, "sealed_forward_count", None) != 1:
            transaction.restore_prepublication_native()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_sha:
                raise GuidedWinnerPublicationError("Post-bake method rollback failed.")
            report.update(
                {
                    "status": "rejected",
                    "reason": "post-bake-guided-method-gate-rejected",
                    "sealed_forward_count": 1,
                    "sealed_marker_consumed": True,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report

        certificate = runner.save_reload_certify(
            compiled.child_model,
            tokenizer,
            output,
        )
        published = True
        child_path = Path(str(certificate.get("path", ""))).expanduser().resolve()
        if child_path.parent != output or child_path.name != "accepted_child":
            raise GuidedWinnerPublicationError(
                "Fresh reload certificate does not bind the direct accepted child."
            )
        independent = runner.independent_reload_certify(
            child_path,
            stage.QWEN3_1P4B_EXACT_PARAMETERS,
        )
        independent_tree = independent.get("artifact_tree")
        certificate_tree = certificate.get("artifact_tree")
        if (
            certificate.get("fresh_reload_performed") is not True
            or certificate.get("exact_budget_match") is not True
            or certificate.get("exact_unique_parameters")
            != stage.QWEN3_1P4B_EXACT_PARAMETERS
            or certificate.get("state_keys_match_pre_save") is not True
            or certificate.get("state_bytes_match_pre_save") is not True
            or certificate.get("published_adapter_parameter_count") != 0
            or certificate.get("published_parent_runtime_reference_count") != 0
            or certificate.get("published_source_sized_sidecar_parameter_count") != 0
            or independent.get("fresh_reload_performed") is not True
            or independent.get("loaded_from_candidate_path") is not True
            or independent.get("exact_unique_parameters")
            != stage.QWEN3_1P4B_EXACT_PARAMETERS
            or independent.get("state_dict_sha256")
            != certificate.get("state_dict_sha256")
            or independent.get("published_adapter_parameter_count") != 0
            or independent.get("parent_runtime_reference_count") != 0
            or independent.get("source_sized_sidecar_parameter_count") != 0
            or not isinstance(independent_tree, Mapping)
            or not isinstance(certificate_tree, Mapping)
            or any(
                independent_tree.get(key) != certificate_tree.get(key)
                for key in ("file_count", "total_bytes", "tree_sha256")
            )
        ):
            raise GuidedWinnerPublicationError(
                "Fresh/independent reload did not prove a standalone exact child."
            )
        report.update(
            {
                "status": "accepted",
                "reason": None,
                "sealed_forward_count": 1,
                "sealed_marker_consumed": True,
                "post_bake_sealed_forward_count": 0,
                "standalone_compaction_passed": True,
                "model_checkpoint_written": True,
                "model_save_count": 1,
                "fresh_reload_certificate": certificate,
                "independent_reload_certificate": dict(independent),
                "fresh_reload_guided_method_gate": {
                    "passed": True,
                    "post_bake_decision_sha256": post_method_decision[
                        "decision_sha256"
                    ],
                    "state_bytes_match_pre_save": certificate[
                        "state_bytes_match_pre_save"
                    ],
                    "candidate_behavior_certified_before_save": True,
                    "serialized_bytes_equal_behavior_certified_bytes": True,
                    "internal_numeric_matching_consulted": False,
                },
                "rollback_exact": None,
                "retained_artifacts": ["accepted_child", "run_report.json"],
            }
        )
        stage._atomic_json(output / "run_report.json", report)
        return report
    except BaseException as error:
        rollback_error: str | None = None
        restored = False
        if transaction is not None:
            try:
                rollback_guided = getattr(runner, "rollback_guided_installation", None)
                if callable(rollback_guided):
                    rollback_guided(transaction)
                restore = getattr(transaction, "restore_prepublication_native", None)
                rollback = getattr(transaction, "rollback", None)
                if callable(restore):
                    restore()
                    restored = True
                elif callable(rollback) and getattr(transaction, "is_live", False):
                    rollback()
                    restored = True
                if (
                    restored
                    and compiled is not None
                    and compiled_sha is not None
                    and runner.state_sha256(compiled.child_model) != compiled_sha
                ):
                    raise GuidedWinnerPublicationError(
                        "Failure rollback changed child."
                    )
            except BaseException as rollback_failure:
                rollback_error = repr(rollback_failure)
        if published or (output / "accepted_child").exists():
            try:
                stage._remove_published_child_tree(output / "accepted_child", output)
            except BaseException as cleanup_failure:
                rollback_error = f"{rollback_error or ''}; {cleanup_failure!r}"
        failure = {
            **report,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "sealed_forward_count": getattr(runner, "sealed_forward_count", 0),
            "sealed_marker_consumed": marker_receipt is not None,
            "post_bake_sealed_forward_count": 0,
            "rollback_error": rollback_error,
            "prepublication_native_restored": restored,
            "standalone_compaction_passed": False,
            "model_checkpoint_written": False,
        }
        try:
            if not (output / "run_report.json").exists():
                stage._atomic_json(output / "run_report.json", failure)
        except BaseException:
            pass
        raise
    finally:
        parent = None
        target = None
        tokenizer = None
        compiled = None
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--dtype", choices=("float16",), required=True)
    parser.add_argument("--parent-device-map", required=True)
    parser.add_argument("--target-device-map", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_winner_publication_preregistration(
        args.preregistration,
        args.preregistration_sha256,
    )
    if args.preflight:
        if args.output_root is not None:
            raise SystemExit("--output-root is only valid with --execute.")
        cuda_preflight = tournament.validate_candidate_tournament_cuda_preflight(
            contract.tournament,
            dtype=args.dtype,
            parent_device_map=args.parent_device_map,
            target_device_map=args.target_device_map,
        )
        report = contract.preflight_report()
        report["cuda_preflight"] = cuda_preflight
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        raise SystemExit("--execute requires --output-root.")
    result = execute_winner_publication(
        contract,
        args.output_root,
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
