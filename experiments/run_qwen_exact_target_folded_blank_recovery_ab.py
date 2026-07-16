#!/usr/bin/env python3
"""Authenticate a sealed-free folded-vs-blank recovery A/B scaffold.

Preregistration and preflight are deliberately model-free.  They authenticate
one authenticated retained V3 or V4 candidate factor state, the exact compiled
blank cache that candidate started from, a frozen replay bank, and one recovery plan shared
byte-for-byte by both arms.  Folded materialization authentication is stricter:
an explicit production callback bundle must independently fresh-reload the
atomic experimental child artifact before zero-step evaluation or training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import authenticate_qwen_exact_target_native_guided_gud_candidate_tournament_history as tournament_history
import materialize_qwen3_stage1_blank_target as blank
import prepare_qwen_exact_target_folded_blank_recovery_inputs as recovery_inputs
import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament
import run_qwen_exact_target_native_guided_gud_winner_publication as publication
import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_compiled_child_cache import (
    CACHE_RECEIPT_FILENAME,
    CompilerIdentity,
    verify_compiled_child_cache,
)
from neural_forge import hf_recovery
from neural_forge import hf_teacher_token_replay
from neural_forge import hf_progressive_target_bridge as progressive_bridge
from neural_forge.hf_recovery import recover_cross_tokenizer_causal_lm
from neural_forge.hf_guided_checkpoint_recovery import (
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_experimental_child_artifact import (
    ARTIFACT_RECEIPT_FILENAME,
    BRIDGE_REPORT_FILENAME,
    ExperimentalChildArtifactError,
    ExperimentalChildArtifactExpectations,
    verify_experimental_child_artifact,
)
from neural_forge.hf_progressive_target_bridge import (
    EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA,
    EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA,
    EXPERIMENTAL_RECOVERY_MATERIALIZATION_RECEIPT_SCHEMA,
)


PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-folded-blank-recovery-ab-preregistration-v2"
)
RUN_SCHEMA = "neural-forge-qwen-exact-target-folded-blank-recovery-ab-run-v2"
REPLAY_SCHEMA = "neural-forge-frozen-recovery-replay-bank-v5"
PLAN_SCHEMA = "neural-forge-folded-blank-recovery-plan-v2"
TEACHER_SOURCE_ARTIFACT_SCHEMA = "neural-forge-teacher-source-artifact-binding-v1"
MATERIALIZATION_RECEIPT_SCHEMA = (
    "neural-forge-folded-blank-recovery-materialization-receipt-v1"
)
ZERO_STEP_RECEIPT_SCHEMA = "neural-forge-folded-blank-recovery-zero-step-receipt-v1"
FOLD_SOURCE_SCHEMA = "neural-forge-folded-recovery-source-binding-v1"
FOLD_SOURCE_AUTHENTICATION_SCHEMA = (
    "neural-forge-folded-recovery-source-authentication-receipt-v1"
)
FoldSourceKind = Literal["historical-guided-qkvo-v3", "guided-gud-v4"]
FOLD_SOURCE_KINDS: tuple[FoldSourceKind, ...] = (
    "historical-guided-qkvo-v3",
    "guided-gud-v4",
)
BUILD_FILENAME = "prepare_qwen_exact_target_folded_blank_recovery_ab.py"
EXECUTOR_FILENAME = "execute_qwen_exact_target_folded_blank_recovery_ab.py"
RECOVERY_INPUT_BUILDER_FILENAME = (
    "prepare_qwen_exact_target_folded_blank_recovery_inputs.py"
)
ARM_IDS = ("folded", "blank")
EXACT_UNIQUE_PARAMETERS = 1_410_567_168
MAX_RETAINED_CHECKPOINTS_PER_ARM = 3
MAX_TOTAL_RETAINED_CHECKPOINTS = 6
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BEHAVIOR_CHECKPOINT_FIELDS = frozenset(
    {
        "step",
        "loss",
        "correct",
        "total",
        "pass_rate",
        "generation_health_correct",
        "generation_health_total",
        "generation_health_rate",
        "behavior_category_breakdown",
        "generation_category_breakdown",
        "behavior_category_summary",
        "generation_category_summary",
        "passed_ids",
        "generation_passed_prompts",
    }
)
_CATEGORY_BREAKDOWN_ROW_FIELDS = frozenset(
    {"correct", "total", "pass_rate", "failed_ids", "passed_ids"}
)
_CATEGORY_SUMMARY_FIELDS = frozenset(
    {
        "category_count",
        "covered_category_count",
        "min_pass_rate",
        "mean_pass_rate",
        "hard_categories",
        "hard_category_correct",
        "hard_category_total",
        "hard_category_pass_rate",
        "hard_category_breakdown",
    }
)
_HARD_CATEGORY_ROW_FIELDS = frozenset({"correct", "total", "pass_rate"})
_REPLAY_RECOVERY_PARAMETER_NAMES = (
    "steps",
    "batch_size",
    "max_length",
    "response_loss_mode",
    "calibration_order",
    "stepwise_donor_max_new_tokens",
    "on_policy_max_new_tokens",
    "teacher_generate_max_new_tokens",
    "on_policy_interval",
)
_FORBIDDEN_RECOVERY_KWARGS = {
    "source_model",
    "candidate_model",
    "candidate_tokenizer_model",
    "output_dir",
    "calibration_texts",
    "behavior_eval_suite",
    "precomputed_teacher_generated_texts",
    "on_policy_replay_bank",
    "stepwise_replay_bank",
    "behavior_checkpoint_observer",
}


class FoldedBlankRecoveryABError(RuntimeError):
    """The retained candidate, neutral blank, or symmetric A/B contract failed."""


@dataclass(frozen=True, slots=True)
class FoldedArtifactVerificationHooks:
    """Explicit production callbacks required for one independent folded reload."""

    model_loader: Callable[[Path], Any]
    native_certifier: Callable[[Any], Mapping[str, Any] | Any]
    runtime_auditor: Callable[[Any, Path], Mapping[str, Any]]
    dense_state_hasher: Callable[[Any], str] | None = None

    def __post_init__(self) -> None:
        required = (
            self.model_loader,
            self.native_certifier,
            self.runtime_auditor,
        )
        if any(not callable(value) for value in required) or (
            self.dense_state_hasher is not None
            and not callable(self.dense_state_hasher)
        ):
            raise FoldedBlankRecoveryABError(
                "Folded artifact verification hooks must all be callable."
            )


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FoldedBlankRecoveryABError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise FoldedBlankRecoveryABError(
            f"{name} fields changed; expected={sorted(fields)!r}, actual={actual!r}."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FoldedBlankRecoveryABError(f"{name} must be a nonnegative integer.")
    return int(value)


def _finite_rate(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise FoldedBlankRecoveryABError(f"{name} must be a finite rate in [0, 1].")
    return float(value)


def _exact_rate(*, correct: int, total: int, rate: Any, name: str) -> float:
    normalized = _finite_rate(rate, name)
    expected = float(correct / max(1, total))
    if abs(normalized - expected) > 1e-12:
        raise FoldedBlankRecoveryABError(
            f"{name} differs from its authenticated correct/total counts."
        )
    return normalized


def _string_list(value: Any, name: str, *, unique: bool) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise FoldedBlankRecoveryABError(
            f"{name} must be an array of nonblank strings."
        )
    if unique and len(value) != len(set(value)):
        raise FoldedBlankRecoveryABError(f"{name} must not contain duplicates.")
    return list(value)


def _behavior_category_breakdown(value: Any, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise FoldedBlankRecoveryABError(f"{name} must be an object.")
    normalized: dict[str, dict[str, Any]] = {}
    for category, raw_row in value.items():
        if (
            not isinstance(category, str)
            or not category
            or category != category.strip().lower().replace("-", "_")
        ):
            raise FoldedBlankRecoveryABError(
                f"{name} contains a noncanonical category name."
            )
        row = _strict(
            raw_row,
            set(_CATEGORY_BREAKDOWN_ROW_FIELDS),
            f"{name}.{category}",
        )
        correct = _nonnegative_int(row["correct"], f"{name}.{category}.correct")
        total = _nonnegative_int(row["total"], f"{name}.{category}.total")
        if correct > total:
            raise FoldedBlankRecoveryABError(
                f"{name}.{category}.correct exceeds total."
            )
        rate = _exact_rate(
            correct=correct,
            total=total,
            rate=row["pass_rate"],
            name=f"{name}.{category}.pass_rate",
        )
        passed_ids = _string_list(
            row["passed_ids"],
            f"{name}.{category}.passed_ids",
            unique=True,
        )
        failed_ids = _string_list(
            row["failed_ids"],
            f"{name}.{category}.failed_ids",
            unique=True,
        )
        if set(passed_ids).intersection(failed_ids):
            raise FoldedBlankRecoveryABError(
                f"{name}.{category} marks one identifier both passed and failed."
            )
        if len(passed_ids) > min(correct, 20) or len(failed_ids) > min(
            total - correct, 20
        ):
            raise FoldedBlankRecoveryABError(
                f"{name}.{category} contains more IDs than its counts permit."
            )
        normalized[category] = {
            "correct": correct,
            "total": total,
            "pass_rate": rate,
            "failed_ids": failed_ids,
            "passed_ids": passed_ids,
        }
    if list(normalized) != sorted(normalized):
        raise FoldedBlankRecoveryABError(f"{name} categories must be sorted.")
    return normalized


def _behavior_category_summary(
    value: Any,
    *,
    breakdown: Mapping[str, Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    summary = _strict(value, set(_CATEGORY_SUMMARY_FIELDS), name)
    hard_categories = _string_list(
        summary["hard_categories"],
        f"{name}.hard_categories",
        unique=True,
    )
    hard_breakdown = summary["hard_category_breakdown"]
    if not isinstance(hard_breakdown, Mapping):
        raise FoldedBlankRecoveryABError(
            f"{name}.hard_category_breakdown must be an object."
        )
    for category, row in hard_breakdown.items():
        if category not in hard_categories:
            raise FoldedBlankRecoveryABError(
                f"{name}.hard_category_breakdown contains an undeclared category."
            )
        strict_row = _strict(
            row,
            set(_HARD_CATEGORY_ROW_FIELDS),
            f"{name}.hard_category_breakdown.{category}",
        )
        correct = _nonnegative_int(
            strict_row["correct"],
            f"{name}.hard_category_breakdown.{category}.correct",
        )
        total = _nonnegative_int(
            strict_row["total"],
            f"{name}.hard_category_breakdown.{category}.total",
        )
        if correct > total:
            raise FoldedBlankRecoveryABError(
                f"{name}.hard_category_breakdown.{category}.correct exceeds total."
            )
        _exact_rate(
            correct=correct,
            total=total,
            rate=strict_row["pass_rate"],
            name=f"{name}.hard_category_breakdown.{category}.pass_rate",
        )
    expected = hf_recovery._category_breakdown_summary(dict(breakdown))
    if canonical_json_bytes(summary) != canonical_json_bytes(expected):
        raise FoldedBlankRecoveryABError(
            f"{name} does not exactly summarize its authenticated category rows."
        )
    return dict(summary)


def _behavior_checkpoint_metrics(
    value: Any,
    *,
    expected_step: int,
    require_loss_none: bool,
    name: str,
) -> dict[str, Any]:
    metrics = _strict(value, set(_BEHAVIOR_CHECKPOINT_FIELDS), name)
    step = _nonnegative_int(metrics["step"], f"{name}.step")
    if step != expected_step:
        raise FoldedBlankRecoveryABError(f"{name}.step differs from its checkpoint.")
    loss = metrics["loss"]
    if require_loss_none:
        if loss is not None:
            raise FoldedBlankRecoveryABError(f"{name}.loss must be null at step zero.")
    elif (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(float(loss))
    ):
        raise FoldedBlankRecoveryABError(f"{name}.loss must be finite after step zero.")

    correct = _nonnegative_int(metrics["correct"], f"{name}.correct")
    total = _nonnegative_int(metrics["total"], f"{name}.total")
    generation_correct = _nonnegative_int(
        metrics["generation_health_correct"],
        f"{name}.generation_health_correct",
    )
    generation_total = _nonnegative_int(
        metrics["generation_health_total"],
        f"{name}.generation_health_total",
    )
    if total <= 0 or correct > total or generation_correct > generation_total:
        raise FoldedBlankRecoveryABError(
            f"{name} contains impossible behavior or generation counts."
        )
    _exact_rate(
        correct=correct,
        total=total,
        rate=metrics["pass_rate"],
        name=f"{name}.pass_rate",
    )
    _exact_rate(
        correct=generation_correct,
        total=generation_total,
        rate=metrics["generation_health_rate"],
        name=f"{name}.generation_health_rate",
    )
    behavior_breakdown = _behavior_category_breakdown(
        metrics["behavior_category_breakdown"],
        f"{name}.behavior_category_breakdown",
    )
    generation_breakdown = _behavior_category_breakdown(
        metrics["generation_category_breakdown"],
        f"{name}.generation_category_breakdown",
    )
    if (
        sum(row["correct"] for row in behavior_breakdown.values()) != correct
        or sum(row["total"] for row in behavior_breakdown.values()) != total
        or sum(row["correct"] for row in generation_breakdown.values())
        != generation_correct
        or sum(row["total"] for row in generation_breakdown.values())
        != generation_total
    ):
        raise FoldedBlankRecoveryABError(
            f"{name} category totals differ from its aggregate counts."
        )
    _behavior_category_summary(
        metrics["behavior_category_summary"],
        breakdown=behavior_breakdown,
        name=f"{name}.behavior_category_summary",
    )
    _behavior_category_summary(
        metrics["generation_category_summary"],
        breakdown=generation_breakdown,
        name=f"{name}.generation_category_summary",
    )
    passed_ids = _string_list(
        metrics["passed_ids"],
        f"{name}.passed_ids",
        unique=True,
    )
    generation_passed_prompts = _string_list(
        metrics["generation_passed_prompts"],
        f"{name}.generation_passed_prompts",
        unique=False,
    )
    if (
        len(passed_ids) != correct
        or len(generation_passed_prompts) != generation_correct
    ):
        raise FoldedBlankRecoveryABError(
            f"{name} passed identifiers differ from its correct counts."
        )
    return dict(metrics)


def _bound_json(path: Path | str, expected_sha256: str, name: str) -> dict[str, Any]:
    source = publication._bound_file(path, expected_sha256, name)
    payload = stage._strict_json(source)
    if not isinstance(payload, dict):
        raise FoldedBlankRecoveryABError(f"{name} must contain one JSON object.")
    canonical_json_bytes(payload)
    return payload


def _module_file(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise FoldedBlankRecoveryABError(
            f"Required implementation module {module_name!r} is absent."
        )
    return Path(spec.origin).resolve()


def _implementation_hashes(*, builder_path: Path) -> dict[str, str]:
    builder = publication._resolve(
        builder_path,
        base=Path.cwd(),
        name="A/B builder implementation",
    )
    if not builder.is_file():
        raise FoldedBlankRecoveryABError("A/B builder implementation is absent.")
    executor = Path(__file__).resolve().with_name(EXECUTOR_FILENAME)
    recovery_input_builder = (
        Path(__file__).resolve().with_name(RECOVERY_INPUT_BUILDER_FILENAME)
    )
    if not executor.is_file() or not recovery_input_builder.is_file():
        raise FoldedBlankRecoveryABError(
            "Production recovery executor or frozen-input builder is absent."
        )
    result = {
        "runner_sha256": stage._file_sha256(Path(__file__).resolve()),
        "builder_sha256": stage._file_sha256(builder),
        "production_executor_sha256": stage._file_sha256(executor),
        "recovery_input_builder_sha256": stage._file_sha256(recovery_input_builder),
        "v4_candidate_tournament_runner_sha256": stage._file_sha256(
            Path(tournament.__file__).resolve()
        ),
        "v3_candidate_tournament_runner_sha256": stage._file_sha256(
            Path(tournament.v3_tournament.__file__).resolve()
        ),
        "v3_winner_publication_runtime_sha256": stage._file_sha256(
            Path(tournament.v3_publication.__file__).resolve()
        ),
        "guided_continuation_runtime_sha256": stage._file_sha256(
            Path(tournament.v3_publication.guided.__file__).resolve()
        ),
        "guided_qkvo_adapter_sha256": stage._file_sha256(
            _module_file(tournament.v3_publication.guided.GUIDED_ADAPTER_MODULE)
        ),
        "guided_gud_adapter_sha256": stage._file_sha256(
            Path(publication.tournament.gud_adapter.__file__).resolve()
        ),
        "v4_candidate_authenticator_sha256": stage._file_sha256(
            Path(publication.__file__).resolve()
        ),
        "v4_historical_authenticator_sha256": stage._file_sha256(
            Path(tournament_history.__file__).resolve()
        ),
        "exact_stage_runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
        "compiled_blank_cache_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_compiled_child_cache")
        ),
        "progressive_bridge_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_progressive_target_bridge")
        ),
        "recovery_implementation_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_recovery")
        ),
        "teacher_token_replay_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_teacher_token_replay")
        ),
        "experimental_child_artifact_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_experimental_child_artifact")
        ),
        "behavior_tools_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_tools")
        ),
        "blank_materializer_sha256": stage._file_sha256(Path(blank.__file__).resolve()),
        "checkpoint_authenticator_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_guided_checkpoint_recovery")
        ),
        "factor_authenticator_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_target_native_behavior_distillation")
        ),
    }
    result["behavior_evaluator_implementation_sha256"] = canonical_sha256(
        {
            "hf_recovery_sha256": result["recovery_implementation_sha256"],
            "hf_tools_sha256": result["behavior_tools_sha256"],
        }
    )
    return result


def _collect_sha256_roots(value: Any, *, path: str = "$") -> dict[str, str]:
    roots: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            roots.update(_collect_sha256_roots(value[key], path=f"{path}.{str(key)}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            roots.update(_collect_sha256_roots(item, path=f"{path}[{index}]"))
    elif isinstance(value, str) and _SHA256.fullmatch(value):
        roots[path] = value
    return roots


@dataclass(frozen=True, slots=True)
class _V4CandidateBinding:
    contract: tournament.AuthenticatedGuidedGUDCandidateTournament
    preregistration_authentication_receipt: Mapping[str, Any]
    preregistration_path: Path
    preregistration_sha256: str
    preregistration: Mapping[str, Any]
    run_report_path: Path
    run_report_sha256: str
    run_report: Mapping[str, Any]
    winner_selection_path: Path
    winner_selection_sha256: str
    winner_selection: Mapping[str, Any]
    winner: Mapping[str, Any]
    factor_payload: Mapping[str, Any]


def _authenticate_v4_candidate(
    *,
    tournament_preregistration: Path,
    tournament_preregistration_sha256: str,
    tournament_run_report: Path,
    tournament_run_report_sha256: str,
    winner_selection: Path,
    winner_selection_sha256: str,
) -> _V4CandidateBinding:
    prereg_path = publication._bound_file(
        tournament_preregistration,
        tournament_preregistration_sha256,
        "v4 candidate tournament preregistration",
    )
    prereg_sha = _sha256(
        tournament_preregistration_sha256,
        "v4 candidate tournament preregistration SHA",
    )
    prereg = _bound_json(prereg_path, prereg_sha, "v4 candidate preregistration")
    try:
        authenticated = tournament_history.authenticate_guided_gud_candidate_preregistration_with_history(
            prereg_path,
            prereg_sha,
        )
        contract = authenticated.contract
        authentication_receipt = authenticated.receipt.as_dict()
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "V4 candidate tournament failed transitive authentication."
        ) from error
    run_path = publication._bound_file(
        tournament_run_report,
        tournament_run_report_sha256,
        "v4 candidate tournament run report",
    )
    run_sha = _sha256(tournament_run_report_sha256, "v4 run report SHA")
    run_report = _bound_json(run_path, run_sha, "v4 candidate run report")
    winner_path = publication._bound_file(
        winner_selection,
        winner_selection_sha256,
        "v4 retained winner selection",
    )
    winner_sha = _sha256(winner_selection_sha256, "v4 winner selection SHA")
    winner_payload = _bound_json(
        winner_path,
        winner_sha,
        "v4 retained winner selection",
    )
    try:
        winner, factor_payload = publication._winner_payload(
            contract,
            run_report,
            winner_payload,
        )
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "Retained v4 factor pack/frontier failed full candidate authentication."
        ) from error
    if (
        run_report.get("sealed_forward_count") != 0
        or run_report.get("sealed_evaluation_accessed") is not False
        or run_report.get("publication_attempted") is not False
        or winner_payload.get("candidate_only") is not True
        or winner_payload.get("sealed_evaluation_accessed") is not False
        or winner_payload.get("publication_attempted") is not False
    ):
        raise FoldedBlankRecoveryABError(
            "A/B source must remain at the sealed-free candidate boundary."
        )
    return _V4CandidateBinding(
        contract=contract,
        preregistration_authentication_receipt=MappingProxyType(
            dict(authentication_receipt)
        ),
        preregistration_path=prereg_path,
        preregistration_sha256=prereg_sha,
        preregistration=MappingProxyType(dict(prereg)),
        run_report_path=run_path,
        run_report_sha256=run_sha,
        run_report=MappingProxyType(dict(run_report)),
        winner_selection_path=winner_path,
        winner_selection_sha256=winner_sha,
        winner_selection=MappingProxyType(dict(winner_payload)),
        winner=MappingProxyType(dict(winner)),
        factor_payload=MappingProxyType(dict(factor_payload)),
    )


@dataclass(frozen=True, slots=True)
class _FoldSourceBinding:
    kind: FoldSourceKind
    tournament_contract: Any
    v3_tournament_contract: Any
    source_guided: Any
    preregistration_path: Path
    preregistration_sha256: str
    preregistration: Mapping[str, Any]
    run_report_path: Path
    run_report_sha256: str
    run_report: Mapping[str, Any]
    winner_selection_path: Path
    winner_selection_sha256: str
    winner_selection: Mapping[str, Any]
    winner: Mapping[str, Any]
    factor_payload: Mapping[str, Any]
    authentication_receipt: Mapping[str, Any]
    historical_source_commit: str | None


def _factor_payload_binding(
    factor_payload: Mapping[str, Any],
    *,
    kind: FoldSourceKind,
    winner: Mapping[str, Any],
) -> dict[str, Any]:
    expected_schema = (
        publication.BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        if kind == "historical-guided-qkvo-v3"
        else publication.BEHAVIOR_FACTOR_PACK_V4_SCHEMA
    )
    rows = factor_payload.get("factors")
    if (
        factor_payload.get("schema") != expected_schema
        or factor_payload.get("factor_state_sha256")
        != winner["runtime_factor_state_sha256"]
        or isinstance(factor_payload.get("factor_count"), bool)
        or not isinstance(factor_payload.get("factor_count"), int)
        or not isinstance(rows, list)
        or factor_payload.get("factor_count") != len(rows)
        or not rows
    ):
        raise FoldedBlankRecoveryABError(
            "Fold source factor payload metadata differs from its authenticated winner."
        )
    tensor_hashes: dict[str, str] = {}
    for index, raw_row in enumerate(rows):
        row = _strict(
            raw_row,
            {"name", "shape", "dtype", "tensor_sha256"},
            f"fold source factor manifest[{index}]",
        )
        name = row["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in tensor_hashes
            or not isinstance(row["shape"], list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in row["shape"]
            )
            or not isinstance(row["dtype"], str)
            or not row["dtype"]
        ):
            raise FoldedBlankRecoveryABError(
                "Fold source factor manifest names, shapes, or dtypes are malformed."
            )
        tensor_hashes[name] = _sha256(
            row["tensor_sha256"],
            f"fold source factor manifest[{index}].tensor_sha256",
        )
    tensor_hashes = {name: tensor_hashes[name] for name in sorted(tensor_hashes)}
    guided_source_sha = _sha256(
        factor_payload.get("guided_source_binding_sha256"),
        "fold source guided source binding SHA",
    )
    guided_gud_source_sha: str | None = None
    if kind == "guided-gud-v4":
        guided_gud_source_sha = _sha256(
            factor_payload.get("guided_gud_source_binding_sha256"),
            "fold source guided-GUD source binding SHA",
        )
    elif "guided_gud_source_binding_sha256" in factor_payload:
        raise FoldedBlankRecoveryABError(
            "Historical guided-QKVO source cannot contain a guided-GUD binding."
        )
    factor_manifest = {
        key: value for key, value in factor_payload.items() if key != "factor_tensors"
    }
    return {
        "factor_pack_schema": expected_schema,
        "factor_count": len(rows),
        "factor_payload_manifest_sha256": canonical_sha256(factor_manifest),
        "factor_rows_sha256": canonical_sha256(rows),
        "factor_tensor_sha256s": tensor_hashes,
        "factor_tensor_sha256s_sha256": canonical_sha256(tensor_hashes),
        "guided_source_binding_sha256": guided_source_sha,
        "guided_gud_source_binding_sha256": guided_gud_source_sha,
    }


def _source_authentication_receipt(
    *,
    kind: FoldSourceKind,
    preregistration_sha256: str,
    run_report_sha256: str,
    winner_selection_sha256: str,
    evaluation_contract_sha256: str,
    candidate_frontier_sha256: str,
    factor_pack_sha256: str,
    historical_source_commit: str | None,
    upstream_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": FOLD_SOURCE_AUTHENTICATION_SCHEMA,
        "kind": kind,
        "authentication_mode": (
            "historical-exact-artifact-v1"
            if kind == "historical-guided-qkvo-v3"
            else "live-v4-transitive-history-authentication"
        ),
        "preregistration_sha256": _sha256(
            preregistration_sha256,
            "fold source authentication preregistration SHA",
        ),
        "run_report_sha256": _sha256(
            run_report_sha256,
            "fold source authentication run report SHA",
        ),
        "winner_selection_sha256": _sha256(
            winner_selection_sha256,
            "fold source authentication winner selection SHA",
        ),
        "evaluation_contract_sha256": _sha256(
            evaluation_contract_sha256,
            "fold source authentication evaluation contract SHA",
        ),
        "candidate_frontier_sha256": _sha256(
            candidate_frontier_sha256,
            "fold source authentication frontier SHA",
        ),
        "factor_pack_sha256": _sha256(
            factor_pack_sha256,
            "fold source authentication factor pack SHA",
        ),
        "historical_source_commit": historical_source_commit,
        "nonretained_callback_preimages_complete": (
            False if kind == "historical-guided-qkvo-v3" else None
        ),
        "upstream_receipt": dict(upstream_receipt),
        "upstream_receipt_sha256": canonical_sha256(dict(upstream_receipt)),
        "candidate_only": True,
        "sealed_evaluation_accessed": False,
        "sealed_data_accessed": False,
        "publication_attempted": False,
        "publication_authorized": False,
        "accepted_artifact": False,
        "behavior_capability_proven": False,
        "coherent_continuation_proven": False,
        "standalone_intelligence_claim": False,
    }
    historical_commit_valid = (
        historical_source_commit == tournament.v3_tournament.HISTORICAL_V3_SOURCE_COMMIT
        if kind == "historical-guided-qkvo-v3"
        else historical_source_commit is None
    )
    if not historical_commit_valid:
        raise FoldedBlankRecoveryABError(
            "Fold source historical commit is inconsistent with its source kind."
        )
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _normalize_v3_winner(
    source: tournament.AuthenticatedV3WinnerSource,
    winner_selection: Mapping[str, Any],
) -> dict[str, Any]:
    raw_winner = winner_selection.get("winner")
    if not isinstance(raw_winner, Mapping):
        raise FoldedBlankRecoveryABError(
            "Historical V3 winner selection has no authenticated winner."
        )
    metrics = raw_winner.get("metrics")
    frontier = source.candidate_frontier
    if not isinstance(metrics, Mapping) or not isinstance(frontier, Mapping):
        raise FoldedBlankRecoveryABError(
            "Historical V3 winner metrics or frontier binding are malformed."
        )
    return {
        "arm_id": source.arm_id,
        "arm_index": int(raw_winner["arm_index"]),
        "step": source.step,
        "rank": list(raw_winner["rank"]),
        "runtime_factor_state_sha256": source.runtime_factor_state_sha256,
        "portable_factor_state_sha256": source.portable_factor_state_sha256,
        "checkpoint_sha256": _sha256(
            raw_winner["checkpoint_sha256"],
            "historical V3 winner checkpoint SHA",
        ),
        "metrics": dict(metrics),
        "metrics_sha256": canonical_sha256(metrics),
        "factor_pack": str(source.factor_pack_path),
        "factor_pack_sha256": source.factor_pack_file_sha256,
        "candidate_frontier": dict(frontier),
    }


def _authenticate_v3_fold_source(
    *,
    tournament_preregistration: Path,
    tournament_preregistration_sha256: str,
    tournament_run_report: Path,
    tournament_run_report_sha256: str,
    winner_selection: Path,
    winner_selection_sha256: str,
) -> _FoldSourceBinding:
    source_value = {
        "tournament_preregistration": {
            "path": str(tournament_preregistration),
            "sha256": tournament_preregistration_sha256,
        },
        "tournament_run_report": {
            "path": str(tournament_run_report),
            "sha256": tournament_run_report_sha256,
        },
        "winner_selection": {
            "path": str(winner_selection),
            "sha256": winner_selection_sha256,
        },
        "expected_winner": {
            "arm_id": tournament.EXPECTED_SOURCE_ARM_ID,
            "step": tournament.EXPECTED_SOURCE_STEP,
            "runtime_factor_state_sha256": tournament.EXPECTED_SOURCE_RUNTIME_V3_SHA256,
            "portable_factor_state_sha256": tournament.EXPECTED_SOURCE_PORTABLE_V3_SHA256,
            "factor_pack_file_sha256": tournament.EXPECTED_SOURCE_FACTOR_PACK_FILE_SHA256,
        },
    }
    try:
        source = tournament.authenticate_v3_winner_source(
            source_value,
            base=Path.cwd(),
        )
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "Historical V3 candidate failed exact-artifact authentication."
        ) from error
    prereg = _bound_json(
        source.tournament_preregistration_path,
        source.tournament_preregistration_sha256,
        "historical V3 candidate preregistration",
    )
    run_report = _bound_json(
        source.run_report_path,
        source.run_report_sha256,
        "historical V3 candidate run report",
    )
    selection = _bound_json(
        source.winner_selection_path,
        source.winner_selection_sha256,
        "historical V3 winner selection",
    )
    winner = _normalize_v3_winner(source, selection)
    evaluation_sha = source.tournament_contract.evaluation_contract_sha256
    upstream = {
        "schema": "neural-forge-historical-v3-source-audit-v1",
        "historical_source_commit": tournament.v3_tournament.HISTORICAL_V3_SOURCE_COMMIT,
        "callback_preimages_persisted": False,
        "candidate_frontier": dict(source.candidate_frontier),
        "exact_artifact_authenticator": "authenticate_v3_winner_source",
        "candidate_only": True,
        "incoherent_candidate_label_preserved": True,
        "publication_authorized": False,
    }
    receipt = _source_authentication_receipt(
        kind="historical-guided-qkvo-v3",
        preregistration_sha256=source.tournament_preregistration_sha256,
        run_report_sha256=source.run_report_sha256,
        winner_selection_sha256=source.winner_selection_sha256,
        evaluation_contract_sha256=evaluation_sha,
        candidate_frontier_sha256=winner["candidate_frontier"]["sha256"],
        factor_pack_sha256=source.factor_pack_file_sha256,
        historical_source_commit=tournament.v3_tournament.HISTORICAL_V3_SOURCE_COMMIT,
        upstream_receipt=upstream,
    )
    return _FoldSourceBinding(
        kind="historical-guided-qkvo-v3",
        tournament_contract=source.tournament_contract,
        v3_tournament_contract=source.tournament_contract,
        source_guided=source.tournament_contract.source,
        preregistration_path=source.tournament_preregistration_path,
        preregistration_sha256=source.tournament_preregistration_sha256,
        preregistration=MappingProxyType(dict(prereg)),
        run_report_path=source.run_report_path,
        run_report_sha256=source.run_report_sha256,
        run_report=MappingProxyType(dict(run_report)),
        winner_selection_path=source.winner_selection_path,
        winner_selection_sha256=source.winner_selection_sha256,
        winner_selection=MappingProxyType(dict(selection)),
        winner=MappingProxyType(winner),
        factor_payload=MappingProxyType(dict(source.factor_payload)),
        authentication_receipt=MappingProxyType(receipt),
        historical_source_commit=tournament.v3_tournament.HISTORICAL_V3_SOURCE_COMMIT,
    )


def _authenticate_fold_source(
    *,
    kind: FoldSourceKind,
    tournament_preregistration: Path,
    tournament_preregistration_sha256: str,
    tournament_run_report: Path,
    tournament_run_report_sha256: str,
    winner_selection: Path,
    winner_selection_sha256: str,
) -> _FoldSourceBinding:
    if kind not in FOLD_SOURCE_KINDS:
        raise FoldedBlankRecoveryABError(
            f"Unknown fold source kind {kind!r}; expected one of {FOLD_SOURCE_KINDS!r}."
        )
    if kind == "historical-guided-qkvo-v3":
        return _authenticate_v3_fold_source(
            tournament_preregistration=tournament_preregistration,
            tournament_preregistration_sha256=tournament_preregistration_sha256,
            tournament_run_report=tournament_run_report,
            tournament_run_report_sha256=tournament_run_report_sha256,
            winner_selection=winner_selection,
            winner_selection_sha256=winner_selection_sha256,
        )
    v4 = _authenticate_v4_candidate(
        tournament_preregistration=tournament_preregistration,
        tournament_preregistration_sha256=tournament_preregistration_sha256,
        tournament_run_report=tournament_run_report,
        tournament_run_report_sha256=tournament_run_report_sha256,
        winner_selection=winner_selection,
        winner_selection_sha256=winner_selection_sha256,
    )
    receipt = _source_authentication_receipt(
        kind="guided-gud-v4",
        preregistration_sha256=v4.preregistration_sha256,
        run_report_sha256=v4.run_report_sha256,
        winner_selection_sha256=v4.winner_selection_sha256,
        evaluation_contract_sha256=v4.contract.evaluation_contract_sha256,
        candidate_frontier_sha256=v4.winner["candidate_frontier"]["sha256"],
        factor_pack_sha256=v4.winner["factor_pack_sha256"],
        historical_source_commit=None,
        upstream_receipt=v4.preregistration_authentication_receipt,
    )
    return _FoldSourceBinding(
        kind="guided-gud-v4",
        tournament_contract=v4.contract,
        v3_tournament_contract=v4.contract.source.tournament_contract,
        source_guided=v4.contract.source.tournament_contract.source,
        preregistration_path=v4.preregistration_path,
        preregistration_sha256=v4.preregistration_sha256,
        preregistration=v4.preregistration,
        run_report_path=v4.run_report_path,
        run_report_sha256=v4.run_report_sha256,
        run_report=v4.run_report,
        winner_selection_path=v4.winner_selection_path,
        winner_selection_sha256=v4.winner_selection_sha256,
        winner_selection=v4.winner_selection,
        winner=v4.winner,
        factor_payload=v4.factor_payload,
        authentication_receipt=MappingProxyType(receipt),
        historical_source_commit=None,
    )


def _compiled_blank_binding(fold_source: _FoldSourceBinding) -> dict[str, Any]:
    try:
        guided_source = fold_source.source_guided
        cache = guided_source.execution_contract.compiler_cache
    except AttributeError as error:
        raise FoldedBlankRecoveryABError(
            "Fold source does not expose its transitive compiled blank cache."
        ) from error
    required_pins = (
        "expected_compiler_identity_sha256",
        "expected_receipt_sha256",
        "expected_compiler_report_sha256",
        "expected_state_dict_sha256",
        "expected_artifact_tree_sha256",
    )
    if (
        cache is None
        or cache.require_hit is not True
        or any(getattr(cache, name, None) is None for name in required_pins)
    ):
        raise FoldedBlankRecoveryABError(
            "Fold source does not require one fully pinned compiled blank cache hit."
        )
    identity_sha = _sha256(
        cache.expected_compiler_identity_sha256,
        "compiled blank identity SHA",
    )
    cache_path = (cache.root / identity_sha).resolve()
    receipt_path = cache_path / CACHE_RECEIPT_FILENAME
    receipt_raw = stage._strict_json(receipt_path)
    if not isinstance(receipt_raw, Mapping):
        raise FoldedBlankRecoveryABError("Compiled blank receipt is malformed.")
    try:
        identity = CompilerIdentity.from_dict(receipt_raw["compiler_identity"])
        verified = verify_compiled_child_cache(
            cache_path,
            expected_identity=identity,
            expected_receipt_sha256=cache.expected_receipt_sha256,
            expected_compiler_report_sha256=(cache.expected_compiler_report_sha256),
            expected_state_dict_sha256=cache.expected_state_dict_sha256,
        )
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "Compiled blank cache failed model-free fresh-reload receipt verification."
        ) from error
    receipt = verified.receipt.as_dict()
    state = receipt.get("model_state")
    tree = receipt.get("artifact_tree")
    if (
        identity.compiler_identity_sha256 != identity_sha
        or not isinstance(state, Mapping)
        or not isinstance(tree, Mapping)
        or state.get("unique_parameter_count") != EXACT_UNIQUE_PARAMETERS
        or state.get("logical_parameter_count") != EXACT_UNIQUE_PARAMETERS
        or receipt.get("fresh_reload_verified") is not True
        or tree.get("tree_sha256") != cache.expected_artifact_tree_sha256
    ):
        raise FoldedBlankRecoveryABError(
            "Compiled blank is not the exact untied 1,410,567,168 state."
        )
    config_path = cache_path / "config.json"
    config = stage._strict_json(config_path)
    if not isinstance(config, Mapping) or any(
        config.get(name) != expected
        for name, expected in blank.CANONICAL_CONFIG.items()
    ):
        raise FoldedBlankRecoveryABError(
            "Compiled blank config differs from the exact stage-1 Qwen shape."
        )
    return {
        "cache_path": str(cache_path),
        "compiler_identity_sha256": identity_sha,
        "parent_artifact_tree_sha256": identity.as_dict()[
            "parent_artifact_tree_sha256"
        ],
        "receipt_sha256": verified.receipt.receipt_sha256,
        "compiler_report_sha256": cache.expected_compiler_report_sha256,
        "artifact_tree": dict(tree),
        "artifact_tree_sha256": tree["tree_sha256"],
        "state": dict(state),
        "state_dict_sha256": state["state_dict_sha256"],
        "compiler_state_sha256": state["compiler_state_sha256"],
        "state_key_sha256": state["state_key_sha256"],
        "state_manifest_sha256": state["state_manifest_sha256"],
        "parameter_topology_sha256": state["parameter_topology_sha256"],
        "buffer_topology_sha256": state["buffer_topology_sha256"],
        "config_file_sha256": stage._file_sha256(config_path),
        "config_payload_sha256": canonical_sha256(config),
        "config": {name: config[name] for name in blank.CANONICAL_CONFIG},
        "exact_unique_parameters": EXACT_UNIQUE_PARAMETERS,
        "fresh_reload_verified": True,
        "model_loaded_for_authentication": False,
        "capability_proven": False,
        "publication_receipt": False,
    }


def _nonblank_texts(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise FoldedBlankRecoveryABError(f"{name} must be a nonempty text list.")
    if any(not isinstance(item, str) for item in value):
        raise FoldedBlankRecoveryABError(f"{name} must contain only strings.")
    result = tuple(value)
    if any(not item.strip() for item in result):
        raise FoldedBlankRecoveryABError(f"{name} contains a blank text.")
    return result


def _ordered_texts_sha256(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(len(texts).to_bytes(8, byteorder="big", signed=False))
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _ordered_replay_bank_sha256(bank: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    digest.update(len(bank).to_bytes(8, byteorder="big", signed=False))
    for batch in bank:
        digest.update(len(batch).to_bytes(8, byteorder="big", signed=False))
        digest.update(bytes.fromhex(_ordered_texts_sha256(batch)))
    return digest.hexdigest()


def _frozen_stepwise_replay_bank(
    value: Any,
) -> tuple[hf_recovery._FrozenStepwiseReplayEvent, ...]:
    try:
        return hf_recovery._freeze_stepwise_replay_bank(value)
    except (TypeError, ValueError) as error:
        raise FoldedBlankRecoveryABError(
            "stepwise_replay_bank is not a valid immutable token replay bank."
        ) from error


def _replay_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    source = publication._bound_file(path, expected_sha256, "frozen replay bank")
    file_sha = _sha256(expected_sha256, "frozen replay bank SHA")
    raw = _bound_json(source, file_sha, "frozen replay bank")
    replay_fields = {
        "schema",
        "status",
        "teacher_token_replay",
        "teacher_source_artifact",
        "behavior_eval_suite",
        "frozen_before_preregistration",
        "contains_sealed_data",
        "publication_authorized",
        "replay_payload_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != replay_fields:
        raise FoldedBlankRecoveryABError(
            "frozen replay bank fields changed or are incomplete."
        )
    body = dict(raw)
    embedded = _sha256(body.pop("replay_payload_sha256"), "replay payload SHA")
    try:
        verified_teacher_replay = (
            hf_teacher_token_replay.verify_teacher_token_replay_payload(
                raw["teacher_token_replay"]
            )
        )
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "Teacher-token replay failed strict model-free authentication."
        ) from error
    calibration = _nonblank_texts(
        verified_teacher_replay["calibration_texts"],
        "teacher_token_replay.calibration_texts",
    )
    teacher = _nonblank_texts(
        verified_teacher_replay["precomputed_teacher_generated_texts"],
        "teacher_token_replay.precomputed_teacher_generated_texts",
    )
    bank_raw = verified_teacher_replay["on_policy_replay_bank"]
    if isinstance(bank_raw, (str, bytes)) or not isinstance(bank_raw, Sequence):
        raise FoldedBlankRecoveryABError("on_policy_replay_bank must be a list.")
    bank = tuple(
        _nonblank_texts(event, f"on_policy_replay_bank[{index}]")
        for index, event in enumerate(bank_raw)
    )
    stepwise_bank = _frozen_stepwise_replay_bank(
        verified_teacher_replay["stepwise_replay_bank"]
    )
    suite = raw["behavior_eval_suite"]
    if not isinstance(suite, Mapping) or not suite:
        raise FoldedBlankRecoveryABError("behavior_eval_suite must be an object.")
    try:
        authenticated_behavior = recovery_inputs.verify_behavior_suite_binding(suite)
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "behavior_eval_suite binding failed strict adapter authentication."
        ) from error
    if (
        authenticated_behavior.get("mode") != recovery_inputs.BEHAVIOR_MODE_FRESH
        or authenticated_behavior.get("causal_ab_eligible") is not True
        or len(calibration) != 640
        or authenticated_behavior["adapter_provenance"]["selection"][
            "recovery_calibration"
        ]["ordered_text_manifest_sha256"]
        != canonical_sha256(list(calibration))
    ):
        raise FoldedBlankRecoveryABError(
            "Production recovery A/B requires the fresh disjoint causal adapter; "
            "explicit JSON is diagnostic-only."
        )
    try:
        loaded_suite, behavior_probes, _generation_prompts = (
            hf_recovery._recovery_behavior_eval_suite(dict(suite))
        )
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "behavior_eval_suite is not a valid recovery evaluator suite."
        ) from error
    adapter = authenticated_behavior["adapter_provenance"]
    generation_by_id = {
        prompt["id"]: prompt for prompt in loaded_suite["generation_prompts"]
    }
    adapter_targets = adapter["targets"]
    selection = adapter["selection"]
    try:
        reservoir_rows = recovery_inputs._verify_fresh_candidate_reservoir_payload(
            selection["candidate_reservoir"]
        )
        recomputed_derivation = recovery_inputs._derive_fresh_candidate_selection(
            reservoir_rows=reservoir_rows,
            teacher_payload=verified_teacher_replay,
        )
        recomputed_disjointness = (
            recovery_inputs._fresh_candidate_replay_disjointness_audit(
                decoded_prompts=[
                    generation_by_id[target["generation_prompt_id"]]["prompt"]
                    for target in adapter_targets
                ],
                prompt_token_ids=[
                    target["prompt_token_ids"] for target in adapter_targets
                ],
                teacher_payload=verified_teacher_replay,
                expected_count=recovery_inputs.FRESH_CANDIDATE_COUNT,
                audit_scope="final-balanced-candidate",
            )
        )
        pool_indices = recomputed_derivation["selected_reservoir_indices"]
        recomputed_pool_disjointness = (
            recovery_inputs._fresh_candidate_replay_disjointness_audit(
                decoded_prompts=[
                    reservoir_rows[index]["decoded_prompt"] for index in pool_indices
                ],
                prompt_token_ids=[
                    reservoir_rows[index]["token_ids"] for index in pool_indices
                ],
                teacher_payload=verified_teacher_replay,
                expected_count=(recovery_inputs.FRESH_PARENT_CANDIDATE_POOL_COUNT),
                audit_scope="parent-candidate-pool",
            )
        )
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "Fresh behavior reservoir derivation or prompts fail the complete "
            "frozen replay audit."
        ) from error
    structural = selection.get("parent_structural_selection")
    selected_rows = selection["selected_candidate"]["prompt_rows"]
    pool_rows = selection["parent_candidate_pool"]["prompt_rows"]
    if (
        not isinstance(structural, Mapping)
        or selection.get("replay_exclusion_derivation") != recomputed_derivation
        or selection.get("final_replay_disjointness_audit_sha256")
        != recomputed_disjointness["payload_sha256"]
        or selection.get("parent_candidate_pool_replay_disjointness_audit_sha256")
        != recomputed_pool_disjointness["payload_sha256"]
        or [target.get("prompt_id") for target in adapter_targets]
        != structural.get("accepted_prompt_ids")
        or [row.get("reservoir_index") for row in selected_rows]
        != structural.get("accepted_reservoir_indices")
        or [row.get("pool_order") for row in selected_rows]
        != structural.get("accepted_pool_orders")
        or [row.get("category_id") for row in selected_rows]
        != structural.get("accepted_category_ids_in_selection_order")
        or [row.get("reservoir_index") for row in pool_rows] != pool_indices
        or [row.get("prompt_id") for row in pool_rows]
        != recomputed_derivation["selected_prompt_ids"]
        or structural.get("accepted_category_histogram")
        != {
            category: recovery_inputs.FRESH_CATEGORY_QUOTA
            for category in structural.get("authenticated_category_ids", [])
        }
        or structural.get("selection_is_child_blind") is not True
        or structural.get("selection_is_not_parent_capability_ranked") is not True
        or structural.get("parent_health_used_as_fixed_binary_gate_only") is not True
        or structural.get("eligible_rows_sorted_or_ranked_after_gate") is not False
        or structural.get("child_or_fold_outputs_accessed") is not False
        or adapter.get("recovery_replay_disjointness_audit") != recomputed_disjointness
        or adapter.get("recovery_replay_disjointness_audit_sha256")
        != recomputed_disjointness["payload_sha256"]
    ):
        raise FoldedBlankRecoveryABError(
            "Fresh behavior reservoir selection or replay-disjointness evidence "
            "changed after freezing."
        )
    parameters = dict(verified_teacher_replay["parameters"])
    tokenizer_fingerprints = verified_teacher_replay["tokenizer_fingerprints"]
    source_model_fingerprint = verified_teacher_replay["source_model_fingerprint"]
    source_model_state = source_model_fingerprint["model_state_fingerprint"]
    teacher_source_artifact = _strict(
        raw["teacher_source_artifact"],
        {
            "schema",
            "expected_parent_artifact_tree_sha256",
            "actual_parent_artifact_tree_sha256",
            "artifact_tree_file_count",
            "artifact_tree_total_bytes",
            "tree_match_verified_before_model_load",
            "source_model_state_dict_sha256",
            "sha256",
        },
        "teacher_source_artifact",
    )
    teacher_source_body = dict(teacher_source_artifact)
    teacher_source_sha = _sha256(
        teacher_source_body.pop("sha256"),
        "teacher_source_artifact.sha256",
    )
    expected_parent_tree_sha = _sha256(
        teacher_source_artifact["expected_parent_artifact_tree_sha256"],
        "teacher_source_artifact.expected_parent_artifact_tree_sha256",
    )
    actual_parent_tree_sha = _sha256(
        teacher_source_artifact["actual_parent_artifact_tree_sha256"],
        "teacher_source_artifact.actual_parent_artifact_tree_sha256",
    )
    teacher_source_state_sha = _sha256(
        teacher_source_artifact["source_model_state_dict_sha256"],
        "teacher_source_artifact.source_model_state_dict_sha256",
    )
    teacher_source_file_count = _nonnegative_int(
        teacher_source_artifact["artifact_tree_file_count"],
        "teacher_source_artifact.artifact_tree_file_count",
    )
    teacher_source_total_bytes = _nonnegative_int(
        teacher_source_artifact["artifact_tree_total_bytes"],
        "teacher_source_artifact.artifact_tree_total_bytes",
    )
    if (
        raw.get("schema") != REPLAY_SCHEMA
        or raw.get("status") != "frozen"
        or raw.get("frozen_before_preregistration") is not True
        or raw.get("contains_sealed_data") is not False
        or raw.get("publication_authorized") is not False
        or not isinstance(suite.get("split"), str)
        or suite.get("split", "").strip().casefold() != "candidate"
        or suite.get("sealed_data_accessed") is not False
        or suite.get("contains_sealed_data", False) is not False
        or suite.get("publication_authorized", False) is not False
        or loaded_suite is None
        or not behavior_probes
        or tokenizer_fingerprints.get("source")
        != tokenizer_fingerprints.get("candidate")
        or authenticated_behavior["adapter_provenance"].get(
            "source_model_fingerprint_sha256"
        )
        != source_model_fingerprint["sha256"]
        or authenticated_behavior["adapter_provenance"].get(
            "source_model_state_dict_sha256"
        )
        != source_model_state["state_dict_sha256"]
        or authenticated_behavior["adapter_provenance"].get(
            "identity_tokenizer_fingerprint_sha256"
        )
        != tokenizer_fingerprints["source"]["sha256"]
        or teacher_source_artifact.get("schema") != TEACHER_SOURCE_ARTIFACT_SCHEMA
        or expected_parent_tree_sha != actual_parent_tree_sha
        or teacher_source_artifact.get("tree_match_verified_before_model_load")
        is not True
        or teacher_source_file_count <= 0
        or teacher_source_total_bytes <= 0
        or teacher_source_state_sha != source_model_state["state_dict_sha256"]
        or canonical_sha256(teacher_source_body) != teacher_source_sha
        or canonical_sha256(body) != embedded
    ):
        raise FoldedBlankRecoveryABError(
            "Replay bank is not frozen, sealed-free, and self-authenticating."
        )
    event_rows = [
        {
            "event_index": index,
            "text_count": len(event),
            "ordered_text_sha256": _ordered_texts_sha256(event),
        }
        for index, event in enumerate(bank)
    ]
    stepwise_event_rows = [
        {
            "event_index": index,
            "optimizer_step": event.step,
            "row_count": len(event.rows),
            "prompt_token_count": sum(len(row.prompt_input_ids) for row in event.rows),
            "rollout_token_count": sum(
                len(row.rollout_input_ids) for row in event.rows
            ),
            "continuation_token_count": sum(
                len(row.rollout_input_ids) - len(row.prompt_input_ids)
                for row in event.rows
            ),
            "event_sha256": hf_recovery._stepwise_replay_event_sha256(event),
        }
        for index, event in enumerate(stepwise_bank)
    ]
    return {
        "path": str(source),
        "file_sha256": file_sha,
        "payload_sha256": embedded,
        "teacher_token_replay_schema": verified_teacher_replay["schema"],
        "teacher_token_replay_payload_sha256": verified_teacher_replay[
            "payload_sha256"
        ],
        "teacher_token_replay_parameters": parameters,
        "teacher_token_replay_parameters_sha256": canonical_sha256(parameters),
        "teacher_source_artifact_sha256": teacher_source_sha,
        "teacher_source_expected_parent_artifact_tree_sha256": (
            expected_parent_tree_sha
        ),
        "teacher_source_actual_parent_artifact_tree_sha256": (actual_parent_tree_sha),
        "teacher_source_artifact_tree_file_count": teacher_source_file_count,
        "teacher_source_artifact_tree_total_bytes": teacher_source_total_bytes,
        "source_tokenizer_fingerprint_sha256": tokenizer_fingerprints["source"][
            "sha256"
        ],
        "candidate_tokenizer_fingerprint_sha256": tokenizer_fingerprints["candidate"][
            "sha256"
        ],
        "source_model_fingerprint_sha256": source_model_fingerprint["sha256"],
        "source_model_state_dict_sha256": source_model_state["state_dict_sha256"],
        "calibration_text_count": len(calibration),
        "calibration_texts_sha256": _ordered_texts_sha256(calibration),
        "teacher_generated_text_count": len(teacher),
        "teacher_generated_texts_sha256": _ordered_texts_sha256(teacher),
        "on_policy_event_count": len(bank),
        "on_policy_events": event_rows,
        "on_policy_replay_bank_sha256": _ordered_replay_bank_sha256(bank),
        "stepwise_event_count": len(stepwise_bank),
        "stepwise_event_steps": [event.step for event in stepwise_bank],
        "stepwise_events": stepwise_event_rows,
        "stepwise_replay_bank_schema": hf_recovery._STEPWISE_REPLAY_SCHEMA,
        "stepwise_replay_bank_sha256": (
            hf_recovery._stepwise_replay_bank_sha256(stepwise_bank)
        ),
        "stepwise_replay_bank_supplied": True,
        "behavior_eval_suite_sha256": canonical_sha256(suite),
        "behavior_eval_suite_binding_payload_sha256": authenticated_behavior[
            "payload_sha256"
        ],
        "behavior_mode": authenticated_behavior["mode"],
        "causal_ab_eligible": True,
        "behavior_adapter_payload_sha256": authenticated_behavior["adapter_provenance"][
            "payload_sha256"
        ],
        "recovery_replay_disjointness_audit_sha256": recomputed_disjointness[
            "payload_sha256"
        ],
        "frozen_before_preregistration": True,
        "contains_sealed_data": False,
        "publication_authorized": False,
    }


def _plan_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    source = publication._bound_file(path, expected_sha256, "recovery plan")
    file_sha = _sha256(expected_sha256, "recovery plan SHA")
    raw = _bound_json(source, file_sha, "recovery plan")
    _strict(
        raw,
        {
            "schema",
            "status",
            "recovery_kwargs",
            "seeds",
            "checkpoint_steps",
            "retained_checkpoint_steps",
            "teacher_token_replay_payload_sha256",
            "zero_step_before_optimizer",
            "reset_rng_before_each_arm",
            "publication_authorized",
            "sealed_data_accessed",
            "plan_payload_sha256",
        },
        "recovery plan",
    )
    body = dict(raw)
    embedded = _sha256(body.pop("plan_payload_sha256"), "plan payload SHA")
    kwargs = raw["recovery_kwargs"]
    seeds = raw["seeds"]
    checkpoints = raw["checkpoint_steps"]
    retained = raw["retained_checkpoint_steps"]
    teacher_token_replay_payload_sha = _sha256(
        raw["teacher_token_replay_payload_sha256"],
        "recovery plan teacher-token replay payload SHA",
    )
    if not isinstance(kwargs, Mapping) or not kwargs:
        raise FoldedBlankRecoveryABError("recovery_kwargs must be nonempty.")
    supported_recovery_kwargs = set(
        inspect.signature(recover_cross_tokenizer_causal_lm).parameters
    )
    unknown_recovery_kwargs = set(kwargs) - supported_recovery_kwargs
    if unknown_recovery_kwargs:
        raise FoldedBlankRecoveryABError(
            "recovery_kwargs contains unsupported names: "
            f"{sorted(unknown_recovery_kwargs)!r}."
        )
    if set(kwargs).intersection(_FORBIDDEN_RECOVERY_KWARGS):
        raise FoldedBlankRecoveryABError(
            "recovery_kwargs cannot replace frozen A/B data/model/output inputs."
        )
    steps = kwargs.get("steps")
    interval = kwargs.get("on_policy_interval")
    stepwise_weight = kwargs.get("stepwise_donor_kd_weight", 0.0)
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
        or isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
        or kwargs.get("adamw_foreach") is not False
        or kwargs.get("local_files_only") is not True
        or kwargs.get("trust_remote_code", False) is not False
        or kwargs.get("save_best_loss", False) is not False
        or kwargs.get("save_best_behavior", False) is not False
        or kwargs.get("behavior_max_probes") != recovery_inputs.BEHAVIOR_PROBE_COUNT
        or kwargs.get("behavior_max_generation_prompts")
        != recovery_inputs.GENERATION_PROMPT_COUNT
        or kwargs.get("behavior_max_new_tokens")
        != recovery_inputs.BEHAVIOR_MAX_NEW_TOKENS
        or isinstance(stepwise_weight, bool)
        or not isinstance(stepwise_weight, (int, float))
        or float(stepwise_weight) < 0.0
    ):
        raise FoldedBlankRecoveryABError(
            "Recovery plan must pin positive steps/interval, local-only trusted "
            "model loading, AdamW foreach=False, no implicit best-checkpoint "
            "retention, the full 48/24/16 behavior contract, and a nonnegative "
            "stepwise donor weight."
        )
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or any(
            isinstance(step, bool) or not isinstance(step, int) for step in checkpoints
        )
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[0] != 0
        or checkpoints[-1] != steps
        or kwargs.get("behavior_eval_checkpoint_steps") != checkpoints
        or not isinstance(retained, list)
        or retained != sorted(set(retained))
        or any(step not in checkpoints or step == 0 for step in retained)
        or len(retained) > MAX_RETAINED_CHECKPOINTS_PER_ARM
    ):
        raise FoldedBlankRecoveryABError(
            "Checkpoint schedule/retention is not exact, ordered, and capped."
        )
    _strict(seeds, {"python", "torch", "cuda", "data_order"}, "seeds")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in seeds.values()
    ):
        raise FoldedBlankRecoveryABError("Every recovery seed must be nonnegative.")
    if (
        raw.get("schema") != PLAN_SCHEMA
        or raw.get("status") != "preregistered"
        or raw.get("zero_step_before_optimizer") is not True
        or raw.get("reset_rng_before_each_arm") is not True
        or raw.get("publication_authorized") is not False
        or raw.get("sealed_data_accessed") is not False
        or canonical_sha256(body) != embedded
    ):
        raise FoldedBlankRecoveryABError(
            "Recovery plan is not a sealed-free self-authenticating A/B plan."
        )
    on_policy_enabled = any(
        float(kwargs.get(name, 0.0)) > 0.0
        for name in (
            "on_policy_anchor_kd_weight",
            "on_policy_unlikelihood_weight",
            "on_policy_hidden_kd_weight",
        )
    )
    events = (
        [step for step in range(1, steps + 1) if step == 1 or step % interval == 0]
        if on_policy_enabled
        else []
    )
    stepwise_events = list(range(1, steps + 1)) if float(stepwise_weight) > 0.0 else []
    return {
        "path": str(source),
        "file_sha256": file_sha,
        "payload_sha256": embedded,
        "recovery_kwargs": dict(kwargs),
        "recovery_kwargs_sha256": canonical_sha256(kwargs),
        "seeds": dict(seeds),
        "seeds_sha256": canonical_sha256(seeds),
        "checkpoint_steps": list(checkpoints),
        "checkpoint_steps_sha256": canonical_sha256(checkpoints),
        "retained_checkpoint_steps": list(retained),
        "teacher_token_replay_payload_sha256": teacher_token_replay_payload_sha,
        "maximum_retained_checkpoints_per_arm": MAX_RETAINED_CHECKPOINTS_PER_ARM,
        "maximum_total_retained_checkpoints": MAX_TOTAL_RETAINED_CHECKPOINTS,
        "on_policy_event_steps": events,
        "on_policy_event_count": len(events),
        "stepwise_donor_kd_weight": float(stepwise_weight),
        "stepwise_event_steps": stepwise_events,
        "stepwise_event_count": len(stepwise_events),
        "stepwise_replay_bank_required": bool(stepwise_events),
        "zero_step_before_optimizer": True,
        "reset_rng_before_each_arm": True,
        "adamw_foreach": False,
        "local_files_only": True,
        "trust_remote_code": False,
        "dynamic_stepwise_donor_generation_forbidden": True,
        "publication_authorized": False,
        "sealed_data_accessed": False,
    }


def build_recovery_ab_preregistration_payload(
    *,
    fold_source_kind: FoldSourceKind,
    tournament_preregistration: Path,
    tournament_preregistration_sha256: str,
    tournament_run_report: Path,
    tournament_run_report_sha256: str,
    winner_selection: Path,
    winner_selection_sha256: str,
    replay_bank: Path,
    replay_bank_sha256: str,
    recovery_plan: Path,
    recovery_plan_sha256: str,
    builder_path: Path,
) -> tuple[dict[str, Any], _FoldSourceBinding]:
    fold_source = _authenticate_fold_source(
        kind=fold_source_kind,
        tournament_preregistration=tournament_preregistration,
        tournament_preregistration_sha256=tournament_preregistration_sha256,
        tournament_run_report=tournament_run_report,
        tournament_run_report_sha256=tournament_run_report_sha256,
        winner_selection=winner_selection,
        winner_selection_sha256=winner_selection_sha256,
    )
    compiled_blank = _compiled_blank_binding(fold_source)
    replay = _replay_binding(replay_bank, replay_bank_sha256)
    plan = _plan_binding(recovery_plan, recovery_plan_sha256)
    replay_parameters = replay["teacher_token_replay_parameters"]
    mismatched_replay_parameters = {
        name: {
            "replay": replay_parameters.get(name),
            "recovery_plan": plan["recovery_kwargs"].get(name),
        }
        for name in _REPLAY_RECOVERY_PARAMETER_NAMES
        if name not in plan["recovery_kwargs"]
        or replay_parameters.get(name) != plan["recovery_kwargs"].get(name)
    }
    if mismatched_replay_parameters:
        raise FoldedBlankRecoveryABError(
            "Frozen teacher replay parameters differ from recovery_kwargs: "
            f"{mismatched_replay_parameters!r}."
        )
    if (
        plan["teacher_token_replay_payload_sha256"]
        != replay["teacher_token_replay_payload_sha256"]
    ):
        raise FoldedBlankRecoveryABError(
            "Recovery plan does not bind the exact frozen teacher-token replay."
        )
    if (
        replay["teacher_source_expected_parent_artifact_tree_sha256"]
        != compiled_blank["parent_artifact_tree_sha256"]
        or replay["teacher_source_actual_parent_artifact_tree_sha256"]
        != compiled_blank["parent_artifact_tree_sha256"]
    ):
        raise FoldedBlankRecoveryABError(
            "Teacher source artifact is not the canonical parent pinned by the "
            "compiled child lineage."
        )
    if replay["on_policy_event_count"] != plan["on_policy_event_count"]:
        raise FoldedBlankRecoveryABError(
            "Frozen on-policy replay event count differs from the recovery schedule."
        )
    if (
        replay["stepwise_event_count"] != plan["stepwise_event_count"]
        or replay["stepwise_event_steps"] != plan["stepwise_event_steps"]
    ):
        raise FoldedBlankRecoveryABError(
            "Frozen stepwise replay bank must contain exactly one ordered event "
            "per scheduled stepwise donor optimizer step."
        )
    implementation = _implementation_hashes(builder_path=builder_path)
    shared = {
        "replay_bank_file_sha256": replay["file_sha256"],
        "replay_bank_payload_sha256": replay["payload_sha256"],
        "teacher_token_replay_payload_sha256": replay[
            "teacher_token_replay_payload_sha256"
        ],
        "teacher_token_replay_parameters_sha256": replay[
            "teacher_token_replay_parameters_sha256"
        ],
        "teacher_source_artifact_sha256": replay["teacher_source_artifact_sha256"],
        "source_tokenizer_fingerprint_sha256": replay[
            "source_tokenizer_fingerprint_sha256"
        ],
        "candidate_tokenizer_fingerprint_sha256": replay[
            "candidate_tokenizer_fingerprint_sha256"
        ],
        "source_model_fingerprint_sha256": replay["source_model_fingerprint_sha256"],
        "source_model_state_dict_sha256": replay["source_model_state_dict_sha256"],
        "canonical_parent_artifact_tree_sha256": compiled_blank[
            "parent_artifact_tree_sha256"
        ],
        "calibration_texts_sha256": replay["calibration_texts_sha256"],
        "teacher_generated_texts_sha256": replay["teacher_generated_texts_sha256"],
        "on_policy_replay_bank_sha256": replay["on_policy_replay_bank_sha256"],
        "behavior_eval_suite_sha256": replay["behavior_eval_suite_sha256"],
        "behavior_eval_suite_binding_payload_sha256": replay[
            "behavior_eval_suite_binding_payload_sha256"
        ],
        "behavior_mode": replay["behavior_mode"],
        "causal_ab_eligible": replay["causal_ab_eligible"],
        "behavior_adapter_payload_sha256": replay["behavior_adapter_payload_sha256"],
        "recovery_replay_disjointness_audit_sha256": replay[
            "recovery_replay_disjointness_audit_sha256"
        ],
        "recovery_plan_file_sha256": plan["file_sha256"],
        "recovery_plan_payload_sha256": plan["payload_sha256"],
        "recovery_kwargs_sha256": plan["recovery_kwargs_sha256"],
        "seeds_sha256": plan["seeds_sha256"],
        "checkpoint_steps_sha256": plan["checkpoint_steps_sha256"],
        "stepwise_replay_bank_schema": replay["stepwise_replay_bank_schema"],
        "stepwise_replay_bank_sha256": replay["stepwise_replay_bank_sha256"],
        "stepwise_event_count": replay["stepwise_event_count"],
        "stepwise_replay_bank_required": plan["stepwise_replay_bank_required"],
        "same_stepwise_replay_bank_required_for_both_arms": True,
        "zero_step_before_optimizer": True,
        "reset_rng_before_each_arm": True,
        "adamw_foreach": False,
        "local_files_only": True,
        "trust_remote_code": False,
        "dynamic_stepwise_donor_generation_forbidden": True,
    }
    shared_sha = canonical_sha256(shared)
    factor_payload_binding = _factor_payload_binding(
        fold_source.factor_payload,
        kind=fold_source.kind,
        winner=fold_source.winner,
    )
    factor_source_binding = {
        "kind": fold_source.kind,
        "factor_pack_sha256": fold_source.winner["factor_pack_sha256"],
        "portable_factor_state_sha256": fold_source.winner[
            "portable_factor_state_sha256"
        ],
        "runtime_factor_state_sha256": fold_source.winner[
            "runtime_factor_state_sha256"
        ],
        "checkpoint_sha256": fold_source.winner["checkpoint_sha256"],
        "metrics_sha256": fold_source.winner["metrics_sha256"],
        "candidate_frontier_sha256": fold_source.winner["candidate_frontier"]["sha256"],
        "winner_selection_sha256": fold_source.winner_selection_sha256,
        **factor_payload_binding,
    }
    factor_source_binding_sha = canonical_sha256(factor_source_binding)
    arms = [
        {
            "arm_id": "folded",
            "initialization": "authenticated-fold-source-materialization",
            "source_portable_factor_state_sha256": fold_source.winner[
                "portable_factor_state_sha256"
            ],
            "source_factor_pack_sha256": fold_source.winner["factor_pack_sha256"],
            "expected_initial_state_sha256": None,
            "shared_execution_contract_sha256": shared_sha,
        },
        {
            "arm_id": "blank",
            "initialization": "authenticated-compiled-blank-state",
            "source_portable_factor_state_sha256": None,
            "source_factor_pack_sha256": None,
            "expected_initial_state_sha256": compiled_blank["state_dict_sha256"],
            "shared_execution_contract_sha256": shared_sha,
        },
    ]
    source_roots = {
        "fold_source_preregistration": _collect_sha256_roots(
            fold_source.preregistration
        ),
        "fold_source_run_report": _collect_sha256_roots(fold_source.run_report),
        "fold_source_winner_selection": _collect_sha256_roots(
            fold_source.winner_selection
        ),
        "fold_source_winner": _collect_sha256_roots(fold_source.winner),
        "fold_source_authentication": _collect_sha256_roots(
            fold_source.authentication_receipt
        ),
        "fold_source_factor_binding": _collect_sha256_roots(factor_source_binding),
        "compiled_blank": _collect_sha256_roots(compiled_blank),
        "replay_bank": _collect_sha256_roots(replay),
        "recovery_plan": _collect_sha256_roots(plan),
        "implementation": _collect_sha256_roots(implementation),
    }
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "status": "experimental-unaccepted",
        "fold_source": {
            "schema": FOLD_SOURCE_SCHEMA,
            "kind": fold_source.kind,
            "historical_source_commit": fold_source.historical_source_commit,
            "nonretained_callback_preimages_complete": (
                False if fold_source.kind == "historical-guided-qkvo-v3" else None
            ),
            "tournament_preregistration": str(fold_source.preregistration_path),
            "tournament_preregistration_sha256": fold_source.preregistration_sha256,
            "tournament_run_report": str(fold_source.run_report_path),
            "tournament_run_report_sha256": fold_source.run_report_sha256,
            "winner_selection": str(fold_source.winner_selection_path),
            "winner_selection_sha256": fold_source.winner_selection_sha256,
            "factor_pack": str(fold_source.winner["factor_pack"]),
            "factor_pack_sha256": fold_source.winner["factor_pack_sha256"],
            "candidate_frontier": str(fold_source.winner["candidate_frontier"]["path"]),
            "candidate_frontier_sha256": fold_source.winner["candidate_frontier"][
                "sha256"
            ],
            "evaluation_contract_sha256": (
                fold_source.tournament_contract.evaluation_contract_sha256
            ),
            "authentication_receipt": dict(fold_source.authentication_receipt),
            "winner": dict(fold_source.winner),
            "factor_pack_schema": str(fold_source.factor_payload.get("schema", "")),
            "factor_source_binding": factor_source_binding,
            "factor_source_binding_sha256": factor_source_binding_sha,
            "candidate_only": True,
            "sealed_evaluation_accessed": False,
            "sealed_data_accessed": False,
            "publication_attempted": False,
            "publication_authorized": False,
            "accepted_artifact": False,
            "behavior_capability_proven": False,
            "coherent_continuation_proven": False,
            "standalone_intelligence_claim": False,
        },
        "compiled_blank": compiled_blank,
        "replay_bank": replay,
        "recovery_plan": plan,
        "shared_execution_contract": shared,
        "shared_execution_contract_sha256": shared_sha,
        "arms": arms,
        "transitive_sha256_roots": source_roots,
        "transitive_sha256_roots_sha256": canonical_sha256(source_roots),
        "implementation": implementation,
        "exact_unique_parameters": EXACT_UNIQUE_PARAMETERS,
        "exact_parameter_count_required_for_both_arms": True,
        "identical_config_and_topology_required_for_both_arms": True,
        "identical_schedule_seeds_checkpoints_for_both_arms": True,
        "symmetric_frozen_replay_bank": True,
        "model_free_preregistration": True,
        "model_weights_loaded": False,
        "gpu_work_launched": False,
        "training_launched": False,
        "materialization_launched": False,
        "zero_step_evaluation_launched": False,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
        "maximum_retained_checkpoints_per_arm": MAX_RETAINED_CHECKPOINTS_PER_ARM,
        "maximum_total_retained_checkpoints": MAX_TOTAL_RETAINED_CHECKPOINTS,
        "production_model_hook_implemented": True,
        "explicit_production_hook_required": False,
    }
    payload["semantic_contract_sha256"] = canonical_sha256(
        {
            key: payload[key]
            for key in (
                "fold_source",
                "compiled_blank",
                "replay_bank",
                "recovery_plan",
                "shared_execution_contract",
                "arms",
                "transitive_sha256_roots_sha256",
                "implementation",
                "exact_unique_parameters",
            )
        }
    )
    return payload, fold_source


@dataclass(frozen=True, slots=True)
class AuthenticatedFoldedBlankRecoveryAB:
    path: Path
    file_sha256: str
    fold_source: _FoldSourceBinding
    raw: Mapping[str, Any]

    def materialization_plan(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "status": "materialization-planned",
            "preregistration": str(self.path),
            "preregistration_sha256": self.file_sha256,
            "semantic_contract_sha256": self.raw["semantic_contract_sha256"],
            "arms": [dict(value) for value in self.raw["arms"]],
            "exact_unique_parameters": EXACT_UNIQUE_PARAMETERS,
            "compiled_blank_state_sha256": self.raw["compiled_blank"][
                "state_dict_sha256"
            ],
            "fold_source_kind": self.raw["fold_source"]["kind"],
            "folded_factor_state_sha256": self.raw["fold_source"]["winner"][
                "portable_factor_state_sha256"
            ],
            "fresh_copy_per_arm_required": True,
            "zero_step_before_optimizer": True,
            "model_weights_loaded": False,
            "gpu_work_launched": False,
            "materialization_launched": False,
            "sealed_data_accessed": False,
            "publication_authorized": False,
            "accepted_artifact": False,
        }

    def preflight_report(self) -> dict[str, Any]:
        return {
            **self.materialization_plan(),
            "status": "preflight-passed",
            "shared_execution_contract": dict(self.raw["shared_execution_contract"]),
            "replay_bank": dict(self.raw["replay_bank"]),
            "recovery_plan": dict(self.raw["recovery_plan"]),
            "maximum_retained_checkpoints_per_arm": (MAX_RETAINED_CHECKPOINTS_PER_ARM),
            "maximum_total_retained_checkpoints": MAX_TOTAL_RETAINED_CHECKPOINTS,
            "production_model_hook_implemented": True,
            "explicit_production_hook_required": False,
        }


def authenticate_recovery_ab_preregistration(
    path: Path | str,
    expected_sha256: str,
) -> AuthenticatedFoldedBlankRecoveryAB:
    source = publication._bound_file(
        path, expected_sha256, "recovery A/B preregistration"
    )
    expected = _sha256(expected_sha256, "recovery A/B preregistration SHA")
    raw = _bound_json(source, expected, "recovery A/B preregistration")
    expected_fields = {
        "schema",
        "status",
        "fold_source",
        "compiled_blank",
        "replay_bank",
        "recovery_plan",
        "shared_execution_contract",
        "shared_execution_contract_sha256",
        "arms",
        "transitive_sha256_roots",
        "transitive_sha256_roots_sha256",
        "implementation",
        "exact_unique_parameters",
        "exact_parameter_count_required_for_both_arms",
        "identical_config_and_topology_required_for_both_arms",
        "identical_schedule_seeds_checkpoints_for_both_arms",
        "symmetric_frozen_replay_bank",
        "model_free_preregistration",
        "model_weights_loaded",
        "gpu_work_launched",
        "training_launched",
        "materialization_launched",
        "zero_step_evaluation_launched",
        "sealed_data_accessed",
        "publication_authorized",
        "accepted_artifact",
        "maximum_retained_checkpoints_per_arm",
        "maximum_total_retained_checkpoints",
        "production_model_hook_implemented",
        "explicit_production_hook_required",
        "semantic_contract_sha256",
    }
    _strict(raw, expected_fields, "recovery A/B preregistration")
    fold_source_row = raw.get("fold_source")
    replay_row = raw.get("replay_bank")
    plan_row = raw.get("recovery_plan")
    if not all(
        isinstance(value, Mapping) for value in (fold_source_row, replay_row, plan_row)
    ):
        raise FoldedBlankRecoveryABError("Recovery A/B source rows are malformed.")
    assert isinstance(fold_source_row, Mapping)
    builder_path = Path(__file__).resolve().with_name(BUILD_FILENAME)
    expected_payload, fold_source = build_recovery_ab_preregistration_payload(
        fold_source_kind=str(fold_source_row["kind"]),
        tournament_preregistration=Path(
            str(fold_source_row["tournament_preregistration"])
        ),
        tournament_preregistration_sha256=str(
            fold_source_row["tournament_preregistration_sha256"]
        ),
        tournament_run_report=Path(str(fold_source_row["tournament_run_report"])),
        tournament_run_report_sha256=str(
            fold_source_row["tournament_run_report_sha256"]
        ),
        winner_selection=Path(str(fold_source_row["winner_selection"])),
        winner_selection_sha256=str(fold_source_row["winner_selection_sha256"]),
        replay_bank=Path(str(replay_row["path"])),
        replay_bank_sha256=str(replay_row["file_sha256"]),
        recovery_plan=Path(str(plan_row["path"])),
        recovery_plan_sha256=str(plan_row["file_sha256"]),
        builder_path=builder_path,
    )
    if canonical_json_bytes(raw) != canonical_json_bytes(expected_payload):
        raise FoldedBlankRecoveryABError(
            "Recovery A/B preregistration differs from freshly authenticated roots."
        )
    return AuthenticatedFoldedBlankRecoveryAB(
        path=source,
        file_sha256=expected,
        fold_source=fold_source,
        raw=MappingProxyType(dict(raw)),
    )


def _self_hashed_receipt(
    value: Any,
    *,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    row = _strict(value, fields | {"receipt_sha256"}, name)
    body = dict(row)
    claimed = _sha256(body.pop("receipt_sha256"), f"{name}.receipt_sha256")
    if canonical_sha256(body) != claimed:
        raise FoldedBlankRecoveryABError(f"{name} self-hash changed.")
    return row


def _authenticate_folded_artifact_publication(
    contract: AuthenticatedFoldedBlankRecoveryAB,
    row: Mapping[str, Any],
    *,
    initial_state_sha256: str,
    hooks: FoldedArtifactVerificationHooks,
) -> dict[str, Any]:
    publication_root = row.get("artifact_publication_root")
    artifact_receipt_sha256 = row.get("artifact_receipt_sha256")
    if not isinstance(publication_root, str) or not publication_root.strip():
        raise FoldedBlankRecoveryABError(
            "Folded materialization must bind one atomic artifact publication root."
        )
    if type(hooks) is not FoldedArtifactVerificationHooks:
        raise FoldedBlankRecoveryABError(
            "Folded materialization requires explicit fresh-reload verification hooks."
        )
    expected_artifact_receipt_sha = _sha256(
        artifact_receipt_sha256,
        "folded experimental child artifact receipt SHA",
    )
    blank_row = contract.raw["compiled_blank"]
    expectations = ExperimentalChildArtifactExpectations(
        preregistration_sha256=contract.file_sha256,
        shared_execution_contract_sha256=contract.raw[
            "shared_execution_contract_sha256"
        ],
        config_payload_sha256=blank_row["config_payload_sha256"],
        state_key_sha256=blank_row["state_key_sha256"],
        parameter_topology_sha256=blank_row["parameter_topology_sha256"],
        buffer_topology_sha256=blank_row["buffer_topology_sha256"],
        exact_unique_parameters=EXACT_UNIQUE_PARAMETERS,
        exact_logical_parameters=EXACT_UNIQUE_PARAMETERS,
        arm_id="folded",
    )
    try:
        verified = verify_experimental_child_artifact(
            Path(publication_root),
            expectations=expectations,
            expected_receipt_sha256=expected_artifact_receipt_sha,
            model_loader=hooks.model_loader,
            native_certifier=hooks.native_certifier,
            runtime_auditor=hooks.runtime_auditor,
            dense_state_hasher=hooks.dense_state_hasher,
        )
    except ExperimentalChildArtifactError as error:
        raise FoldedBlankRecoveryABError(
            "Folded experimental child artifact failed independent fresh reload."
        ) from error
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "Folded experimental child artifact verifier failed closed."
        ) from error
    expected_root = Path(publication_root).expanduser().resolve(strict=False)
    artifact_receipt = verified.receipt.as_dict()
    report = dict(verified.bridge_report)
    report_path = verified.publication_path / BRIDGE_REPORT_FILENAME
    report_record = artifact_receipt.get("bridge_report")
    artifact_record = artifact_receipt.get("artifact")
    fingerprint = artifact_receipt.get("fresh_reload_model_state_fingerprint")
    if (
        verified.publication_path != expected_root
        or verified.receipt.receipt_sha256 != expected_artifact_receipt_sha
        or not (verified.publication_path / ARTIFACT_RECEIPT_FILENAME).is_file()
        or not isinstance(report_record, Mapping)
        or not isinstance(artifact_record, Mapping)
        or not isinstance(fingerprint, Mapping)
        or artifact_receipt.get("fresh_reload_state_dict_sha256")
        != initial_state_sha256
        or fingerprint.get("state_dict_sha256") != initial_state_sha256
        or fingerprint.get("state_key_sha256") != blank_row["state_key_sha256"]
        or fingerprint.get("parameter_topology_sha256")
        != blank_row["parameter_topology_sha256"]
        or fingerprint.get("buffer_topology_sha256")
        != blank_row["buffer_topology_sha256"]
        or report_record.get("materialized_bridge_state_sha256")
        != artifact_receipt.get("fresh_reload_bridge_state_sha256")
        or artifact_receipt.get("fresh_reload_performed") is not True
        or artifact_receipt.get("artifact_tree_unchanged_by_fresh_reload") is not True
    ):
        raise FoldedBlankRecoveryABError(
            "Folded artifact receipt does not bind the fresh-reloaded initial state."
        )
    try:
        canonical_report_bytes = progressive_bridge._canonical_json_report_bytes(
            report,
            hash_field="materialization_transaction_sha256",
        )
    except Exception as error:
        raise FoldedBlankRecoveryABError(
            "Folded materialization report is not a canonical bridge transaction."
        ) from error
    if report_path.read_bytes() != canonical_report_bytes or report_record.get(
        "file_sha256"
    ) != stage._file_sha256(report_path):
        raise FoldedBlankRecoveryABError(
            "Folded artifact does not contain the exact canonical bridge report bytes."
        )
    transaction_body = dict(report)
    transaction_sha = _sha256(
        transaction_body.pop("materialization_transaction_sha256", None),
        "folded materialization transaction SHA",
    )
    if canonical_sha256(transaction_body) != transaction_sha:
        raise FoldedBlankRecoveryABError(
            "Folded materialization report transaction hash changed."
        )

    experiment_contract = _strict(
        report.get("experimental_recovery_contract"),
        {
            "schema",
            "recovery_ab_preregistration_sha256",
            "recovery_ab_semantic_contract_sha256",
            "shared_execution_contract_sha256",
            "factor_state_sha256",
            "factor_payload_schema",
            "native_child_state_sha256",
            "blank_control_state_sha256",
            "compiled_blank_cache_state_dict_sha256",
            "restored_runtime_factor_state_sha256",
            "restored_portable_factor_state_sha256",
            "bridge_native_child_state_sha256",
            "bridge_live_factor_state_sha256",
            "exact_unique_parameters",
            "publication_authorized",
            "sealed_data_accessed",
        },
        "folded materialization experiment contract",
    )
    embedded_receipt = _self_hashed_receipt(
        report.get("experimental_recovery_receipt"),
        fields={
            "schema",
            "experiment_contract_sha256",
            "factor_state_sha256",
            "live_factor_state_sha256",
            "native_child_state_sha256",
            "exact_unique_parameters",
            "factor_payload_schema",
            "status",
            "publication_authorized",
            "sealed_data_accessed",
            "behavior_capability_proven",
            "recovery_training_required",
            "external_experiment_contract_revalidated_by_bridge",
            "external_factor_pack_revalidated_by_bridge",
            "live_factor_tensor_state_revalidated_by_bridge",
            "native_child_state_revalidated_by_bridge",
        },
        name="folded bridge materialization receipt",
    )
    winner = contract.raw["fold_source"]["winner"]
    expected_factor_schema = contract.raw["fold_source"]["factor_pack_schema"]
    evidence_sha = _sha256(
        row.get("materialization_evidence_sha256"),
        "folded materialization evidence SHA",
    )
    bridge_native_sha = _sha256(
        row.get("bridge_native_child_state_sha256"),
        "folded bridge native child state SHA",
    )
    bridge_live_factor_sha = _sha256(
        row.get("bridge_live_factor_state_sha256"),
        "folded bridge live factor state SHA",
    )
    materialized_bridge_sha = _sha256(
        row.get("materialized_bridge_state_sha256"),
        "folded materialized bridge state SHA",
    )
    if (
        report.get("schema")
        != EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA
        or report.get("status") != "experimental-unaccepted"
        or report.get("exact_unique_parameters") != EXACT_UNIQUE_PARAMETERS
        or report.get("exact_budget_match") is not True
        or report.get("materialized_native_state_sha256") != materialized_bridge_sha
        or report.get("native_child_state_sha256") != bridge_native_sha
        or report.get("live_factor_state_sha256") != bridge_live_factor_sha
        or report.get("materialized_registered_adapter_parameter_count") != 0
        or report.get("materialized_optimizer_parameter_count") != 0
        or report.get("materialized_registered_source_sized_sidecar_parameter_count")
        != 0
        or report.get("standalone_structure_materialized") is not True
        or report.get("serialization_only_context") is not True
        or report.get("serialized_state_must_match_materialized_state_sha256")
        is not True
        or report.get("fresh_reload_and_recovery_required") is not True
        or report.get("publication_authorized") is not False
        or report.get("accepted_artifact") is not False
        or report.get("sealed_data_accessed") is not False
        or report.get("behavior_capability_proven") is not False
        or report.get("external_factor_pack_revalidated_by_bridge") is not False
        or report.get("external_factor_pack_receipt_bound") is not False
        or report.get("contract_factor_state_sha256_is_external_label_only") is not True
        or report.get("live_factor_state_sha256_is_bridge_authenticated") is not True
        or experiment_contract.get("schema") != EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA
        or experiment_contract.get("recovery_ab_preregistration_sha256")
        != contract.file_sha256
        or experiment_contract.get("recovery_ab_semantic_contract_sha256")
        != contract.raw["semantic_contract_sha256"]
        or experiment_contract.get("shared_execution_contract_sha256")
        != contract.raw["shared_execution_contract_sha256"]
        or experiment_contract.get("factor_state_sha256")
        != winner["portable_factor_state_sha256"]
        or experiment_contract.get("factor_payload_schema") != expected_factor_schema
        or experiment_contract.get("native_child_state_sha256") != bridge_native_sha
        or experiment_contract.get("blank_control_state_sha256") != bridge_native_sha
        or experiment_contract.get("compiled_blank_cache_state_dict_sha256")
        != blank_row["state_dict_sha256"]
        or experiment_contract.get("restored_runtime_factor_state_sha256")
        != winner["runtime_factor_state_sha256"]
        or experiment_contract.get("restored_portable_factor_state_sha256")
        != winner["portable_factor_state_sha256"]
        or experiment_contract.get("bridge_native_child_state_sha256")
        != bridge_native_sha
        or experiment_contract.get("bridge_live_factor_state_sha256")
        != bridge_live_factor_sha
        or experiment_contract.get("exact_unique_parameters") != EXACT_UNIQUE_PARAMETERS
        or experiment_contract.get("publication_authorized") is not False
        or experiment_contract.get("sealed_data_accessed") is not False
        or embedded_receipt.get("schema")
        != EXPERIMENTAL_RECOVERY_MATERIALIZATION_RECEIPT_SCHEMA
        or embedded_receipt.get("experiment_contract_sha256")
        != canonical_sha256(experiment_contract)
        or embedded_receipt.get("factor_state_sha256")
        != winner["portable_factor_state_sha256"]
        or embedded_receipt.get("live_factor_state_sha256") != bridge_live_factor_sha
        or embedded_receipt.get("native_child_state_sha256") != bridge_native_sha
        or embedded_receipt.get("exact_unique_parameters") != EXACT_UNIQUE_PARAMETERS
        or embedded_receipt.get("factor_payload_schema") != expected_factor_schema
        or embedded_receipt.get("status") != "experimental-unaccepted"
        or embedded_receipt.get("publication_authorized") is not False
        or embedded_receipt.get("sealed_data_accessed") is not False
        or embedded_receipt.get("behavior_capability_proven") is not False
        or embedded_receipt.get("recovery_training_required") is not True
        or embedded_receipt.get("external_experiment_contract_revalidated_by_bridge")
        is not False
        or embedded_receipt.get("external_factor_pack_revalidated_by_bridge")
        is not False
        or embedded_receipt.get("live_factor_tensor_state_revalidated_by_bridge")
        is not True
        or embedded_receipt.get("native_child_state_revalidated_by_bridge") is not True
        or embedded_receipt.get("receipt_sha256") != evidence_sha
        or row.get("materialization_transaction_sha256") != transaction_sha
    ):
        raise FoldedBlankRecoveryABError(
            "Folded materialization report does not bind the authenticated factor, "
            "blank state, A/B contract, and dense materialized bytes."
        )
    return {
        "publication_root": str(verified.publication_path),
        "artifact_receipt_path": str(
            verified.publication_path / ARTIFACT_RECEIPT_FILENAME
        ),
        "artifact_receipt_sha256": expected_artifact_receipt_sha,
        "artifact_tree_sha256": artifact_record["tree"]["tree_sha256"],
        "bridge_report_path": str(report_path),
        "bridge_report_file_sha256": report_record["file_sha256"],
        "transaction_sha256": transaction_sha,
        "evidence_sha256": evidence_sha,
        "materialized_bridge_state_sha256": materialized_bridge_sha,
        "fresh_reload_state_dict_sha256": initial_state_sha256,
        "bridge_native_child_state_sha256": bridge_native_sha,
        "bridge_live_factor_state_sha256": bridge_live_factor_sha,
        "fresh_reload_performed": True,
        "artifact_tree_unchanged_by_fresh_reload": True,
    }


def authenticate_materialization_receipts(
    contract: AuthenticatedFoldedBlankRecoveryAB,
    receipts: Mapping[str, Any],
    *,
    folded_artifact_hooks: FoldedArtifactVerificationHooks,
) -> dict[str, Any]:
    if type(contract) is not AuthenticatedFoldedBlankRecoveryAB or set(receipts) != set(
        ARM_IDS
    ):
        raise FoldedBlankRecoveryABError(
            "Materialization requires the canonical contract and both exact arms."
        )
    fields = {
        "schema",
        "status",
        "arm_id",
        "initial_state_sha256",
        "exact_unique_parameters",
        "config_payload_sha256",
        "parameter_topology_sha256",
        "buffer_topology_sha256",
        "shared_execution_contract_sha256",
        "source_portable_factor_state_sha256",
        "source_factor_pack_sha256",
        "factor_source_binding_sha256",
        "factor_pack_authenticated_separately",
        "materialization_evidence_sha256",
        "materialization_transaction_sha256",
        "artifact_publication_root",
        "artifact_receipt_sha256",
        "bridge_native_child_state_sha256",
        "bridge_live_factor_state_sha256",
        "materialized_bridge_state_sha256",
        "optimizer_steps",
        "sealed_data_accessed",
        "publication_authorized",
        "accepted_artifact",
    }
    normalized: dict[str, dict[str, Any]] = {}
    report_bindings: dict[str, dict[str, Any] | None] = {}
    blank_row = contract.raw["compiled_blank"]
    winner = contract.raw["fold_source"]["winner"]
    for arm_id in ARM_IDS:
        row = _self_hashed_receipt(
            receipts[arm_id],
            fields=fields,
            name=f"{arm_id} materialization receipt",
        )
        initial_sha = _sha256(row["initial_state_sha256"], "initial state SHA")
        expected_portable = (
            winner["portable_factor_state_sha256"] if arm_id == "folded" else None
        )
        expected_pack = winner["factor_pack_sha256"] if arm_id == "folded" else None
        expected_source_binding = (
            contract.raw["fold_source"]["factor_source_binding_sha256"]
            if arm_id == "folded"
            else None
        )
        evidence_sha = _sha256(
            row["materialization_evidence_sha256"],
            f"{arm_id} materialization evidence SHA",
        )
        transaction_sha = row["materialization_transaction_sha256"]
        report_binding = None
        if arm_id == "folded":
            report_binding = _authenticate_folded_artifact_publication(
                contract,
                row,
                initial_state_sha256=initial_sha,
                hooks=folded_artifact_hooks,
            )
        if (
            row.get("schema") != MATERIALIZATION_RECEIPT_SCHEMA
            or row.get("status") != "experimental-unaccepted"
            or row.get("arm_id") != arm_id
            or row.get("exact_unique_parameters") != EXACT_UNIQUE_PARAMETERS
            or row.get("config_payload_sha256") != blank_row["config_payload_sha256"]
            or row.get("parameter_topology_sha256")
            != blank_row["parameter_topology_sha256"]
            or row.get("buffer_topology_sha256") != blank_row["buffer_topology_sha256"]
            or row.get("shared_execution_contract_sha256")
            != contract.raw["shared_execution_contract_sha256"]
            or row.get("source_portable_factor_state_sha256") != expected_portable
            or row.get("source_factor_pack_sha256") != expected_pack
            or row.get("factor_source_binding_sha256") != expected_source_binding
            or row.get("factor_pack_authenticated_separately")
            is not (arm_id == "folded")
            or (arm_id == "blank" and evidence_sha != blank_row["receipt_sha256"])
            or (arm_id == "blank" and transaction_sha is not None)
            or (
                arm_id == "blank"
                and row.get("bridge_native_child_state_sha256") is not None
            )
            or (
                arm_id == "blank"
                and row.get("bridge_live_factor_state_sha256") is not None
            )
            or (
                arm_id == "blank"
                and row.get("materialized_bridge_state_sha256") is not None
            )
            or (arm_id == "blank" and row.get("artifact_publication_root") is not None)
            or (arm_id == "blank" and row.get("artifact_receipt_sha256") is not None)
            or row.get("optimizer_steps") != 0
            or row.get("sealed_data_accessed") is not False
            or row.get("publication_authorized") is not False
            or row.get("accepted_artifact") is not False
            or (arm_id == "blank" and initial_sha != blank_row["state_dict_sha256"])
            or (arm_id == "folded" and initial_sha == blank_row["state_dict_sha256"])
        ):
            raise FoldedBlankRecoveryABError(
                f"{arm_id} materialization receipt violates the symmetric contract."
            )
        normalized[arm_id] = dict(row)
        report_bindings[arm_id] = report_binding
    return {
        "schema": RUN_SCHEMA,
        "status": "materialization-authenticated",
        "preregistration_sha256": contract.file_sha256,
        "receipts": normalized,
        "authenticated_materialization_reports": report_bindings,
        "folded_and_blank_initial_states_distinct": (
            normalized["folded"]["initial_state_sha256"]
            != normalized["blank"]["initial_state_sha256"]
        ),
        "optimizer_steps": 0,
        "folded_artifact_independent_fresh_reload_verified": True,
        "model_loading_performed_by_this_authenticator": True,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
    }


def authenticate_zero_step_receipts(
    contract: AuthenticatedFoldedBlankRecoveryAB,
    materialization: Mapping[str, Any],
    receipts: Mapping[str, Any],
    *,
    folded_artifact_hooks: FoldedArtifactVerificationHooks,
) -> dict[str, Any]:
    if type(contract) is not AuthenticatedFoldedBlankRecoveryAB or set(receipts) != set(
        ARM_IDS
    ):
        raise FoldedBlankRecoveryABError(
            "Zero-step authentication requires both materialized A/B arms."
        )
    embedded_materialization_receipts = materialization.get("receipts")
    if not isinstance(embedded_materialization_receipts, Mapping):
        raise FoldedBlankRecoveryABError(
            "Zero-step materialization evidence does not contain both receipts."
        )
    expected_materialization = authenticate_materialization_receipts(
        contract,
        embedded_materialization_receipts,
        folded_artifact_hooks=folded_artifact_hooks,
    )
    if canonical_json_bytes(materialization) != canonical_json_bytes(
        expected_materialization
    ):
        raise FoldedBlankRecoveryABError(
            "Zero-step materialization wrapper differs from fresh authentication."
        )
    fields = {
        "schema",
        "status",
        "arm_id",
        "initial_state_sha256",
        "shared_execution_contract_sha256",
        "replay_bank_payload_sha256",
        "checkpoint_step",
        "optimizer_steps",
        "metrics",
        "metrics_sha256",
        "evaluator_implementation_sha256",
        "sealed_data_accessed",
        "publication_authorized",
        "accepted_artifact",
    }
    normalized: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        row = _self_hashed_receipt(
            receipts[arm_id],
            fields=fields,
            name=f"{arm_id} zero-step receipt",
        )
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping):
            _behavior_checkpoint_metrics(
                metrics,
                expected_step=0,
                require_loss_none=True,
                name=f"{arm_id} zero-step metrics",
            )
        if (
            row.get("schema") != ZERO_STEP_RECEIPT_SCHEMA
            or row.get("status") != "experimental-unaccepted"
            or row.get("arm_id") != arm_id
            or row.get("initial_state_sha256")
            != materialization["receipts"][arm_id]["initial_state_sha256"]
            or row.get("shared_execution_contract_sha256")
            != contract.raw["shared_execution_contract_sha256"]
            or row.get("replay_bank_payload_sha256")
            != contract.raw["replay_bank"]["payload_sha256"]
            or row.get("checkpoint_step") != 0
            or row.get("optimizer_steps") != 0
            or not isinstance(metrics, Mapping)
            or row.get("metrics_sha256") != canonical_sha256(metrics)
            or row.get("evaluator_implementation_sha256")
            != contract.raw["implementation"][
                "behavior_evaluator_implementation_sha256"
            ]
            or row.get("sealed_data_accessed") is not False
            or row.get("publication_authorized") is not False
            or row.get("accepted_artifact") is not False
        ):
            raise FoldedBlankRecoveryABError(
                f"{arm_id} zero-step receipt violates the symmetric contract."
            )
        normalized[arm_id] = dict(row)
    return {
        "schema": RUN_SCHEMA,
        "status": "zero-step-authenticated",
        "preregistration_sha256": contract.file_sha256,
        "materialization_receipts": dict(materialization["receipts"]),
        "zero_step_receipts": normalized,
        "next_action": "authenticated-production-recovery-executor-ready",
        "training_launched": False,
        "optimizer_steps": 0,
        "maximum_retained_checkpoints_per_arm": (MAX_RETAINED_CHECKPOINTS_PER_ARM),
        "maximum_total_retained_checkpoints": MAX_TOTAL_RETAINED_CHECKPOINTS,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_recovery_ab_preregistration(
        args.preregistration,
        args.preregistration_sha256,
    )
    print(json.dumps(contract.preflight_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
