#!/usr/bin/env python3
"""Run the candidate-only guided continuation tournament.

The tournament is deliberately not a publication gate.  Three authenticated
starting trajectories are reset onto the same compiled blank child, trained on
the exact same fit/candidate schedule, and ranked only by causal output actions
and free-running rollouts.  The best three distinct full v3 factor states are
retained; one candidate winner is written and execution stops before sealed
evaluation or model publication.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import importlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import prepare_qwen_exact_target_native_continuation_splits as split_tools
import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_exact_candidate_frontier import rollout_first_rank
from neural_forge.hf_guided_attention_site_mask import (
    GradientSiteMaskGuard,
    GuidedAttentionSiteMask,
    build_guided_attention_site_mask,
)
from neural_forge.hf_guided_checkpoint_recovery import (
    GUIDED_CHECKPOINT_CANDIDATE_SCHEMA,
    GuidedCheckpointRecoverySink,
    canonical_json_bytes,
    canonical_sha256,
    portable_guided_source_binding_sha256,
    portable_v3_factor_state_sha256,
    rebind_recovered_v3_factor_payload,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BehaviorDistillationSchedule,
    DurableBehaviorCheckpoint,
    authenticate_guided_factor_sources,
    capture_target_native_factor_state,
)


PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-candidate-tournament-"
    "preregistration-v1"
)
RUN_SCHEMA = "neural-forge-qwen-exact-target-native-guided-candidate-tournament-v2"
WINNER_SCHEMA = "neural-forge-guided-candidate-tournament-winner-v2"
FRONTIER_SCHEMA = "neural-forge-guided-candidate-tournament-frontier-v2"
ARM_IDS = (
    "step40-v3-trajectory",
    "step64-v3-trajectory",
    "source-v2-trajectory-site-mask",
)
CHECKPOINT_STEPS = (0, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64)
SPLIT_COUNTS = {"fit": 168, "candidate": 24, "sealed": 32}
MAX_RETAINED_STATES = 3
EXPECTED_CALLBACK_COUNT = len(ARM_IDS) * len(CHECKPOINT_STEPS)
FINAL_CALLBACK_DISPOSITIONS = frozenset(
    {
        "retained-top-three",
        "not-retained-rank-cutoff",
        "ineligible-non-pareto",
        "duplicate-retained-state",
        "error-inconsistent-duplicate",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CUDA_DEVICE_ORDER = "PCI_BUS_ID"
EXPECTED_CUDA_VISIBLE_DEVICES = "4,6"
EXPECTED_PARENT_DEVICE_MAP = "cuda:0"
EXPECTED_TARGET_DEVICE_MAP = "cuda:1"
EXPECTED_PARENT_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
EXPECTED_TARGET_DEVICE_NAME = "Quadro RTX 8000"
EXPECTED_PARENT_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_TARGET_COMPUTE_CAPABILITY = (7, 5)
HISTORICAL_V3_SOURCE_COMMIT = "565aad1b618df3f65dc5b9a25d25eb1a0f90ef5b"
HISTORICAL_TOURNAMENT_IMPLEMENTATION_PATHS = {
    "runner_sha256": (
        "experiments/run_qwen_exact_target_native_guided_candidate_tournament.py"
    ),
    "builder_sha256": (
        "experiments/prepare_qwen_exact_target_native_guided_candidate_tournament.py"
    ),
}


class CandidateTournamentError(RuntimeError):
    """A tournament contract, reset, callback, or candidate-only gate failed."""


def candidate_tournament_runtime_contract() -> dict[str, Any]:
    """Return the one admissible production GPU/runtime contract."""

    return {
        "dtype": "float16",
        "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
        "cuda_device_order": EXPECTED_CUDA_DEVICE_ORDER,
        "cuda_visible_devices": EXPECTED_CUDA_VISIBLE_DEVICES,
        "parent_device_map": EXPECTED_PARENT_DEVICE_MAP,
        "parent_device_name": EXPECTED_PARENT_DEVICE_NAME,
        "parent_compute_capability": list(EXPECTED_PARENT_COMPUTE_CAPABILITY),
        "target_device_map": EXPECTED_TARGET_DEVICE_MAP,
        "target_device_name": EXPECTED_TARGET_DEVICE_NAME,
        "target_compute_capability": list(EXPECTED_TARGET_COMPUTE_CAPABILITY),
        "exact_source_reset_between_arms": True,
        "durable_callback_every_arm": True,
        "mask_verify_after_each_optimizer_step": True,
    }


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CandidateTournamentError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise CandidateTournamentError(
            f"{name} must contain exactly {sorted(keys)}; got {actual!r}."
        )
    return value


def _resolve(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CandidateTournamentError(f"{name} must be a nonempty path.")
    raw = Path(value).expanduser()
    candidate = base / raw if not raw.is_absolute() else raw
    cursor = candidate.absolute()
    while True:
        if cursor.exists() and cursor.is_symlink():
            raise CandidateTournamentError(f"{name} cannot traverse a symlink.")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    result = candidate.resolve()
    return result


def _file_binding(value: Any, *, base: Path, name: str) -> tuple[Path, str]:
    row = _strict(value, {"path", "sha256"}, name)
    path = _resolve(base, row["path"], f"{name}.path")
    digest = _sha256(row["sha256"], f"{name}.sha256")
    if not path.is_file() or stage._file_sha256(path) != digest:
        raise CandidateTournamentError(f"{name} is absent or changed.")
    return path, digest


def candidate_tournament_schedule() -> BehaviorDistillationSchedule:
    """Return the one admissible schedule for all tournament arms."""

    return BehaviorDistillationSchedule(
        steps=64,
        learning_rate=5e-5,
        checkpoint_steps=CHECKPOINT_STEPS,
        fit_horizon=16,
        evaluation_horizon=8,
        fit_prompts_per_step=8,
        repetition_excess_loss_weight=0.4,
        parent_lane_loss_weight=2.0,
        child_lane_loss_weight=2.0,
        parent_only_steps=4,
        warmup_fit_horizon=8,
        late_position_loss_multiplier=2.0,
        divergence_loss_multiplier=6.0,
        post_divergence_loss_multiplier=3.0,
        post_divergence_positions=4,
        child_lane_starts_at_divergence=True,
        max_child_lane_prefixes=8,
        child_repeat_stop_run_length=0,
        guided_qk_learning_rate_scale=0.5,
        guided_vo_learning_rate_scale=1.0,
        guided_right_factor_learning_rate_scale=0.125,
        guided_joint_inherited_learning_rate_scale=0.25,
        guided_continuation_enabled=True,
        guided_gud_unlock_step=0,
        guided_lexical_embedding_unlock_step=0,
        guided_lexical_head_unlock_step=0,
        max_prompt_tokens=split_tools.MAX_PROMPT_TOKENS,
        max_train_prefixes=20_000,
        max_evaluation_prefixes=20_000,
        max_checkpoints=16,
        selection_policy="rollout-first-acquisition",
        sealed_evaluation_policy="disabled",
    )


def tournament_behavior_rank(metrics: Mapping[str, Any]) -> tuple[int, ...]:
    """Ascending, behavior-only rank shared by every arm."""

    return rollout_first_rank(metrics)


def _prompt_records(binding: stage.SplitBinding) -> tuple[Any, ...]:
    values = binding.payload.get("prompts")
    if not isinstance(values, list):
        raise CandidateTournamentError(f"{binding.name} has no prompt list.")
    try:
        return tuple(
            split_tools._prompt_record(value, f"{binding.name}.prompts[{index}]")
            for index, value in enumerate(values)
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise CandidateTournamentError("Split prompt authentication failed.") from error


def _authenticate_split_evolution(
    splits: Mapping[str, stage.SplitBinding],
    source: guided.AuthenticatedGuidedContinuation,
) -> None:
    if {name: value.expected_count for name, value in splits.items()} != SPLIT_COUNTS:
        raise CandidateTournamentError("Tournament splits are not exact 168/24/32.")
    old = source.execution_contract.splits
    if {name: old[name].expected_count for name in old} != {
        "fit": 144,
        "candidate": 24,
        "sealed": 32,
    }:
        raise CandidateTournamentError("Guided source is not exact 144/24/32.")
    old_records = {name: _prompt_records(old[name]) for name in old}
    new_records = {name: _prompt_records(splits[name]) for name in splits}
    expected_fit = (*old_records["fit"], *old_records["candidate"])
    if [row.payload for row in new_records["fit"]] != [
        row.payload for row in expected_fit
    ]:
        raise CandidateTournamentError(
            "Tournament fit must be old fit followed by old candidate, byte for byte."
        )
    if (
        splits["sealed"].path != old["sealed"].path
        or splits["sealed"].sha256 != old["sealed"].sha256
        or [row.payload for row in new_records["sealed"]]
        != [row.payload for row in old_records["sealed"]]
    ):
        raise CandidateTournamentError("Sealed32 must remain exact and locked.")
    try:
        split_tools._assert_globally_disjoint(new_records, label="candidate tournament")
        split_tools._assert_globally_disjoint(
            {
                "historical": tuple(
                    row
                    for name in ("fit", "candidate", "sealed")
                    for row in old_records[name]
                ),
                "new-candidate": new_records["candidate"],
            },
            label="candidate tournament historical exclusion",
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise CandidateTournamentError(
            "New candidate24 is not exact/prefix-disjoint from historical prompts."
        ) from error


@dataclass(frozen=True, slots=True)
class V3InitialState:
    arm_id: str
    source_step: int
    path: Path
    file_sha256: str
    factor_state_sha256: str
    portable_factor_state_sha256: str
    payload: Mapping[str, Any]
    recovery_record_path: Path
    recovery_record_sha256: str

    def as_report(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "source_step": self.source_step,
            "path": str(self.path),
            "sha256": self.file_sha256,
            "factor_state_sha256": self.factor_state_sha256,
            "portable_factor_state_sha256": self.portable_factor_state_sha256,
            "recovery_record": str(self.recovery_record_path),
            "recovery_record_sha256": self.recovery_record_sha256,
            "schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            "authenticated_before_model_access": True,
        }


@dataclass(frozen=True, slots=True)
class TournamentArm:
    arm_id: str
    initial_schema: str
    initial_factor_state_sha256: str
    initial_portable_factor_state_sha256: str | None
    site_mask_enabled: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class AuthenticatedCandidateTournament:
    path: Path
    file_sha256: str
    source: guided.AuthenticatedGuidedContinuation
    splits: Mapping[str, stage.SplitBinding]
    initial_states: Mapping[str, V3InitialState]
    site_mask: GuidedAttentionSiteMask
    site_mask_path: Path
    site_mask_file_sha256: str
    arms: tuple[TournamentArm, ...]
    schedule: BehaviorDistillationSchedule
    runtime: Mapping[str, Any]
    implementation: Mapping[str, str]
    evaluation_contract_sha256: str

    def preflight_report(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "status": "preflight-passed",
            "preregistration": str(self.path),
            "preregistration_sha256": self.file_sha256,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "source_guided_preregistration_sha256": self.source.file_sha256,
            "splits": {
                name: {
                    "path": str(value.path),
                    "sha256": value.sha256,
                    "expected_count": value.expected_count,
                }
                for name, value in self.splits.items()
            },
            "initial_states": {
                name: value.as_report() for name, value in self.initial_states.items()
            },
            "site_mask": {
                "path": str(self.site_mask_path),
                "sha256": self.site_mask_file_sha256,
                "contract_sha256": self.site_mask.mask_contract_sha256,
                "selected_site_counts": {
                    row.role: row.selected_site_count
                    for row in self.site_mask.role_masks
                },
            },
            "arms": [value.as_dict() for value in self.arms],
            "training": self.schedule.as_dict(),
            "runtime": dict(self.runtime),
            "implementation": dict(self.implementation),
            "models_loaded": False,
            "gpu_work_launched": False,
            "sealed_suite_supplied_to_trainer": False,
            "sealed_evaluation_locked": True,
            "selection_uses_internal_numeric_targets": False,
            "publication_authorized": False,
        }


def _load_v3_initial_state(
    row: Any,
    *,
    base: Path,
    arm_id: str,
    expected_step: int,
    expected_source_v2_sha256: str,
    expected_source_v2_tensor_sha256s: Mapping[str, str],
    expected_audit_sha256: str,
    expected_basis_sha256: str,
    expected_basis_payload_sha256: str,
    expected_basis_manifest_sha256: str,
    expected_basis_tensor_sha256s: Mapping[str, str],
) -> V3InitialState:
    value = _strict(
        row,
        {
            "path",
            "sha256",
            "factor_state_sha256",
            "portable_factor_state_sha256",
            "source_step",
            "recovery_record",
        },
        f"initial_states.{arm_id}",
    )
    source_step = value["source_step"]
    if source_step != expected_step:
        raise CandidateTournamentError(f"{arm_id} source step changed.")
    path, digest = _file_binding(
        {"path": value["path"], "sha256": value["sha256"]},
        base=base,
        name=f"initial_states.{arm_id}.pack",
    )
    record_path, record_sha = _file_binding(
        value["recovery_record"],
        base=base,
        name=f"initial_states.{arm_id}.recovery_record",
    )
    factor_sha = _sha256(
        value["factor_state_sha256"], f"initial_states.{arm_id}.factor_state_sha256"
    )
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CandidateTournamentError(
            f"Cannot safely load {arm_id} v3 pack."
        ) from error
    payload: Any = loaded
    if isinstance(loaded, Mapping) and isinstance(loaded.get("factor_pack"), Mapping):
        payload = loaded["factor_pack"]
    if not isinstance(payload, Mapping):
        raise CandidateTournamentError(f"{arm_id} v3 payload is absent.")
    try:
        GuidedCheckpointRecoverySink._validate_v3_factor_payload(
            payload, expected_factor_state_sha256=factor_sha
        )
    except Exception as error:
        raise CandidateTournamentError(
            f"{arm_id} v3 payload failed authentication."
        ) from error
    binding = payload.get("guided_source_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("source_v2_factor_state_sha256") != expected_source_v2_sha256
        or binding.get("source_v2_tensor_sha256s")
        != dict(expected_source_v2_tensor_sha256s)
        or binding.get("audit_report_sha256") != expected_audit_sha256
        or binding.get("basis_envelope_file_sha256") != expected_basis_sha256
        or binding.get("basis_payload_sha256") != expected_basis_payload_sha256
        or binding.get("basis_manifest_sha256") != expected_basis_manifest_sha256
        or binding.get("source_basis_tensor_sha256s")
        != dict(expected_basis_tensor_sha256s)
    ):
        raise CandidateTournamentError(f"{arm_id} is bound to another guided source.")
    record = stage._strict_json(record_path)
    if not isinstance(record, Mapping):
        raise CandidateTournamentError(f"{arm_id} recovery record is not a mapping.")
    declared_record_sha = _sha256(
        record.get("candidate_record_sha256"),
        f"initial_states.{arm_id}.candidate_record_sha256",
    )
    if (
        canonical_sha256(
            {
                name: value
                for name, value in record.items()
                if name != "candidate_record_sha256"
            }
        )
        != declared_record_sha
    ):
        raise CandidateTournamentError(f"{arm_id} recovery record self-hash changed.")
    portable_sha = _sha256(
        value["portable_factor_state_sha256"],
        f"initial_states.{arm_id}.portable_factor_state_sha256",
    )
    payload_portable_sha = portable_v3_factor_state_sha256(payload)
    events = record.get("events")
    retained_steps = record.get("retained_steps")
    matching_events = (
        [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("step") == expected_step
        ]
        if isinstance(events, list)
        else []
    )
    if (
        record.get("schema") != GUIDED_CHECKPOINT_CANDIDATE_SCHEMA
        or record.get("factor_state_sha256") != factor_sha
        or record.get("portable_factor_state_sha256") != portable_sha
        or payload_portable_sha != portable_sha
        or record.get("portable_guided_source_binding_sha256")
        != portable_guided_source_binding_sha256(payload["guided_source_binding"])
        or record.get("runtime_identity_excluded_from_portable_content")
        != ["inherited_installation_sha256"]
        or record.get("content_address") != factor_sha
        or record.get("contains_parent_state") is not False
        or record.get("contains_native_child_state") is not False
        or record.get("contains_optimizer_state") is not False
        or record.get("contains_model_checkpoint") is not False
        or record.get("factor_pack_relative_path") != path.name
        or record_path.parent / str(record.get("factor_pack_relative_path")) != path
        or record.get("factor_pack_file_sha256") != digest
        or record.get("factor_pack_size_bytes") != path.stat().st_size
        or not isinstance(events, list)
        or record.get("event_count") != len(events)
        or retained_steps
        != [event.get("step") for event in events if isinstance(event, Mapping)]
        or expected_step not in retained_steps
        or len(matching_events) != 1
        or matching_events[0].get("factor_state_sha256") != factor_sha
        or matching_events[0].get("portable_factor_state_sha256") != portable_sha
    ):
        raise CandidateTournamentError(f"{arm_id} recovery record disagrees with pack.")
    return V3InitialState(
        arm_id=arm_id,
        source_step=source_step,
        path=path,
        file_sha256=digest,
        factor_state_sha256=factor_sha,
        portable_factor_state_sha256=portable_sha,
        payload=dict(payload),
        recovery_record_path=record_path,
        recovery_record_sha256=record_sha,
    )


def _authenticate_tournament_implementation(
    value: Any,
    *,
    historical_git_revision: str | None = None,
) -> dict[str, Any]:
    implementation = _strict(
        value,
        set(HISTORICAL_TOURNAMENT_IMPLEMENTATION_PATHS),
        "implementation",
    )
    if historical_git_revision is not None:
        try:
            guided._authenticate_historical_git_blob_pins(
                implementation,
                historical_git_revision,
                paths=HISTORICAL_TOURNAMENT_IMPLEMENTATION_PATHS,
            )
        except guided.GuidedContinuationError as error:
            raise CandidateTournamentError(
                "Historical tournament implementation authentication failed."
            ) from error
        return dict(implementation)
    prepare_path = (
        Path(__file__)
        .with_name("prepare_qwen_exact_target_native_guided_candidate_tournament.py")
        .resolve()
    )
    live = {
        "runner_sha256": stage._file_sha256(Path(__file__).resolve()),
        "builder_sha256": stage._file_sha256(prepare_path),
    }
    if dict(implementation) != live:
        raise CandidateTournamentError("Tournament implementation hashes changed.")
    return live


def authenticate_candidate_tournament_preregistration(
    path: Path | str,
    expected_sha256: str,
    *,
    historical_git_revision: str | None = None,
) -> AuthenticatedCandidateTournament:
    prereg_path = Path(path).expanduser().resolve()
    expected = _sha256(expected_sha256, "preregistration_sha256")
    if prereg_path.is_symlink() or not prereg_path.is_file():
        raise CandidateTournamentError(
            "Tournament preregistration is absent/symlinked."
        )
    if stage._file_sha256(prereg_path) != expected:
        raise CandidateTournamentError("Tournament preregistration SHA changed.")
    raw = stage._strict_json(prereg_path)
    prereg = _strict(
        raw,
        {
            "schema",
            "sealed_before_model_access",
            "source_guided",
            "splits",
            "initial_states",
            "site_mask",
            "arms",
            "training",
            "selection",
            "runtime",
            "implementation",
            "evaluation_contract_sha256",
        },
        "preregistration",
    )
    if (
        prereg["schema"] != PREREGISTRATION_SCHEMA
        or prereg["sealed_before_model_access"] is not True
    ):
        raise CandidateTournamentError("Tournament schema/seal changed.")
    source_path, source_sha = _file_binding(
        prereg["source_guided"], base=prereg_path.parent, name="source_guided"
    )
    source = (
        guided.authenticate_guided_preregistration(source_path, source_sha)
        if historical_git_revision is None
        else guided.authenticate_guided_preregistration(
            source_path,
            source_sha,
            historical_git_revision=historical_git_revision,
        )
    )
    split_rows = _strict(prereg["splits"], {"fit", "candidate", "sealed"}, "splits")
    try:
        splits = {
            name: stage._load_split(
                split_rows[name], name=name, base=prereg_path.parent
            )
            for name in ("fit", "candidate", "sealed")
        }
    except stage.ExactTargetNativeStageError as error:
        raise CandidateTournamentError("Tournament split binding failed.") from error
    _authenticate_split_evolution(splits, source)
    continuation = source.execution_contract.continuation
    if continuation is None or continuation.trainer_factor_pack_schema != (
        BEHAVIOR_FACTOR_PACK_V2_SCHEMA
    ):
        raise CandidateTournamentError("Source guided trajectory is not full v2.")
    initial_rows = _strict(
        prereg["initial_states"],
        {ARM_IDS[0], ARM_IDS[1]},
        "initial_states",
    )
    source_v2_rows = continuation.factor_payload.get("factors")
    if not isinstance(source_v2_rows, list):
        raise CandidateTournamentError("Source v2 factor manifest disappeared.")
    source_v2_tensor_sha256s = {
        str(row.get("name")): str(row.get("tensor_sha256"))
        for row in source_v2_rows
        if isinstance(row, Mapping)
    }
    initial_states = {
        ARM_IDS[0]: _load_v3_initial_state(
            initial_rows[ARM_IDS[0]],
            base=prereg_path.parent,
            arm_id=ARM_IDS[0],
            expected_step=40,
            expected_source_v2_sha256=continuation.factor_state_sha256,
            expected_source_v2_tensor_sha256s=source_v2_tensor_sha256s,
            expected_audit_sha256=source.artifacts.audit_report_self_sha256,
            expected_basis_sha256=source.artifacts.basis_envelope_file_sha256,
            expected_basis_payload_sha256=source.artifacts.basis_payload_sha256,
            expected_basis_manifest_sha256=source.artifacts.basis_manifest_sha256,
            expected_basis_tensor_sha256s=source.artifacts.basis_tensor_sha256s,
        ),
        ARM_IDS[1]: _load_v3_initial_state(
            initial_rows[ARM_IDS[1]],
            base=prereg_path.parent,
            arm_id=ARM_IDS[1],
            expected_step=64,
            expected_source_v2_sha256=continuation.factor_state_sha256,
            expected_source_v2_tensor_sha256s=source_v2_tensor_sha256s,
            expected_audit_sha256=source.artifacts.audit_report_self_sha256,
            expected_basis_sha256=source.artifacts.basis_envelope_file_sha256,
            expected_basis_payload_sha256=source.artifacts.basis_payload_sha256,
            expected_basis_manifest_sha256=source.artifacts.basis_manifest_sha256,
            expected_basis_tensor_sha256s=source.artifacts.basis_tensor_sha256s,
        ),
    }
    mask_path, mask_sha = _file_binding(
        prereg["site_mask"], base=prereg_path.parent, name="site_mask"
    )
    computed_mask = build_guided_attention_site_mask(
        source.artifacts.audit_report,
        expected_report_sha256=source.artifacts.audit_report_self_sha256,
    )
    mask_payload = stage._strict_json(mask_path)
    if canonical_json_bytes(mask_payload) != canonical_json_bytes(
        computed_mask.as_dict()
    ):
        raise CandidateTournamentError("Site-mask contract differs from audit replay.")
    arms = (
        TournamentArm(
            ARM_IDS[0],
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            initial_states[ARM_IDS[0]].factor_state_sha256,
            initial_states[ARM_IDS[0]].portable_factor_state_sha256,
            False,
        ),
        TournamentArm(
            ARM_IDS[1],
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            initial_states[ARM_IDS[1]].factor_state_sha256,
            initial_states[ARM_IDS[1]].portable_factor_state_sha256,
            False,
        ),
        TournamentArm(
            ARM_IDS[2],
            BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            continuation.factor_state_sha256,
            None,
            True,
        ),
    )
    if prereg["arms"] != [value.as_dict() for value in arms]:
        raise CandidateTournamentError("Tournament arm order or source changed.")
    schedule = candidate_tournament_schedule()
    if canonical_json_bytes(prereg["training"]) != canonical_json_bytes(
        schedule.as_dict()
    ):
        raise CandidateTournamentError("Tournament schedule changed.")
    selection = _strict(
        prereg["selection"],
        {
            "domain",
            "rank_order",
            "retain_distinct_full_v3_states",
            "distinct_identity",
            "rank_step_zero",
            "later_states_require_pareto_safety_vs_step_zero",
            "sealed_disabled",
            "stop_after_candidate_winner",
        },
        "selection",
    )
    expected_selection = {
        "domain": "candidate-output-actions-and-free-rollouts-only",
        "rank_order": [
            "exact_rollout_count",
            "lcp_at_least_4_count",
            "lcp_token_count",
            "token_match_count",
            "repeat_fraction_excess",
            "adjacent_repeat_fraction_excess",
            "longest_run_excess",
            "action_agreement_count",
        ],
        "retain_distinct_full_v3_states": 3,
        "distinct_identity": "portable-v3-factor-content",
        "rank_step_zero": True,
        "later_states_require_pareto_safety_vs_step_zero": True,
        "sealed_disabled": True,
        "stop_after_candidate_winner": True,
    }
    if dict(selection) != expected_selection:
        raise CandidateTournamentError("Behavior-only selection contract changed.")
    runtime = _strict(
        prereg["runtime"],
        set(candidate_tournament_runtime_contract()),
        "runtime",
    )
    expected_runtime = candidate_tournament_runtime_contract()
    if dict(runtime) != expected_runtime:
        raise CandidateTournamentError("Tournament runtime contract changed.")
    expected_implementation = _authenticate_tournament_implementation(
        prereg["implementation"],
        historical_git_revision=historical_git_revision,
    )
    preimage = {
        "source_guided_preregistration_sha256": source.file_sha256,
        "split_sha256s": {name: value.sha256 for name, value in splits.items()},
        "initial_state_sha256s": {
            name: {
                "file": value.file_sha256,
                "factor": value.factor_state_sha256,
                "portable_factor": value.portable_factor_state_sha256,
                "record": value.recovery_record_sha256,
            }
            for name, value in initial_states.items()
        },
        "site_mask_file_sha256": mask_sha,
        "site_mask_contract_sha256": computed_mask.mask_contract_sha256,
        "arms": [value.as_dict() for value in arms],
        "training": schedule.as_dict(),
        "selection": expected_selection,
        "runtime": expected_runtime,
        "implementation": expected_implementation,
    }
    evaluation_sha = canonical_sha256(preimage)
    if prereg["evaluation_contract_sha256"] != evaluation_sha:
        raise CandidateTournamentError("Evaluation contract SHA changed.")
    return AuthenticatedCandidateTournament(
        path=prereg_path,
        file_sha256=expected,
        source=source,
        splits=splits,
        initial_states=initial_states,
        site_mask=computed_mask,
        site_mask_path=mask_path,
        site_mask_file_sha256=mask_sha,
        arms=arms,
        schedule=schedule,
        runtime=expected_runtime,
        implementation=expected_implementation,
        evaluation_contract_sha256=evaluation_sha,
    )


def validate_candidate_tournament_cuda_preflight(
    contract: AuthenticatedCandidateTournament,
    *,
    dtype: str,
    parent_device_map: str,
    target_device_map: str,
    torch_module: Any = torch,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Authenticate the exact logical GPU mapping without loading either model."""

    if type(contract) is not AuthenticatedCandidateTournament:
        raise CandidateTournamentError(
            "CUDA preflight requires an authenticated tournament contract."
        )
    expected = dict(contract.runtime)
    if (
        dtype != expected["dtype"]
        or parent_device_map != expected["parent_device_map"]
        or target_device_map != expected["target_device_map"]
        or environ.get("CUDA_DEVICE_ORDER") != expected["cuda_device_order"]
        or environ.get("CUDA_VISIBLE_DEVICES") != expected["cuda_visible_devices"]
    ):
        raise CandidateTournamentError(
            "Tournament CLI runtime or CUDA visibility differs from preregistration."
        )
    cuda = getattr(torch_module, "cuda", None)
    if (
        cuda is None
        or not callable(getattr(cuda, "is_available", None))
        or not cuda.is_available()
        or not callable(getattr(cuda, "device_count", None))
        or cuda.device_count() != 2
    ):
        raise CandidateTournamentError(
            "Tournament preflight requires exactly two visible CUDA devices."
        )
    observed = []
    for logical_index, role in ((0, "parent"), (1, "target")):
        name = cuda.get_device_name(logical_index)
        capability = tuple(cuda.get_device_capability(logical_index))
        if (
            name != expected[f"{role}_device_name"]
            or list(capability) != expected[f"{role}_compute_capability"]
        ):
            raise CandidateTournamentError(
                f"Logical CUDA {logical_index} is not the preregistered {role} GPU."
            )
        observed.append(
            {
                "role": role,
                "logical_index": logical_index,
                "device_map": expected[f"{role}_device_map"],
                "name": name,
                "compute_capability": list(capability),
            }
        )
    return {
        "status": "passed",
        "cuda_device_order": environ["CUDA_DEVICE_ORDER"],
        "cuda_visible_devices": environ["CUDA_VISIBLE_DEVICES"],
        "devices": observed,
        "models_loaded": False,
        "model_forward_calls": 0,
        "gpu_kernel_work_launched": False,
    }


@dataclass(slots=True)
class _RetainedState:
    arm_id: str
    arm_index: int
    step: int
    rank: tuple[int, ...]
    runtime_factor_state_sha256: str
    portable_factor_state_sha256: str
    checkpoint_sha256: str
    metrics: Mapping[str, Any]
    pack_path: Path
    pack_sha256: str

    @property
    def total_order(self) -> tuple[Any, ...]:
        return (
            *self.rank,
            self.arm_index,
            self.step,
            self.portable_factor_state_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_index": self.arm_index,
            "step": self.step,
            "rank": list(self.rank),
            "runtime_factor_state_sha256": self.runtime_factor_state_sha256,
            "portable_factor_state_sha256": self.portable_factor_state_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metrics": dict(self.metrics),
            "factor_pack": str(self.pack_path),
            "factor_pack_sha256": self.pack_sha256,
        }


class TournamentTopThreeSink:
    """Durably retain the globally strongest three distinct non-step-zero states."""

    def __init__(self, output_root: Path | str, *, evaluation_contract_sha256: str):
        self.output_root = Path(output_root).expanduser().resolve()
        if not self.output_root.is_dir() or self.output_root.is_symlink():
            raise CandidateTournamentError("Sink output root must already be real.")
        self.evaluation_contract_sha256 = _sha256(
            evaluation_contract_sha256, "evaluation_contract_sha256"
        )
        self.retained_root = self.output_root / "retained"
        self.retained_root.mkdir()
        self._retained: dict[str, _RetainedState] = {}
        self._events: dict[str, list[DurableBehaviorCheckpoint]] = {}
        self._completed: set[str] = set()
        self._observed_callbacks: list[dict[str, Any]] = []
        self._write_frontier()

    def _append_pending_observation(
        self,
        *,
        arm_id: str,
        arm_index: int,
        event: DurableBehaviorCheckpoint,
        metrics: Mapping[str, Any],
        rank: tuple[int, ...],
        runtime_digest: str,
        portable_digest: str,
    ) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "ordinal": len(self._observed_callbacks),
            "arm_id": arm_id,
            "arm_index": arm_index,
            "step": event.step,
            "checkpoint_sha256": event.checkpoint_sha256,
            "runtime_factor_state_sha256": runtime_digest,
            "portable_factor_state_sha256": portable_digest,
            "pareto_safe_vs_step0": event.pareto_safe_vs_step0,
            "metrics": dict(metrics),
            "metrics_sha256": canonical_sha256(metrics),
            "rank": list(rank),
            "eligible_for_retention": (event.step == 0 or event.pareto_safe_vs_step0),
            "disposition": "observed-pending",
            "retained_after_callback": None,
            "factor_pack_created": None,
            "evicted_portable_factor_state_sha256s": None,
            "terminal_verified": False,
            "terminal_pareto_safe_vs_step0": None,
        }
        self._observed_callbacks.append(observation)
        self._write_frontier()
        return observation

    def _finalize_observation(
        self,
        observation: dict[str, Any],
        *,
        disposition: str,
        retained_after_callback: bool,
        factor_pack_created: bool,
        evicted: Sequence[str] = (),
    ) -> None:
        if (
            observation.get("disposition") != "observed-pending"
            or disposition not in FINAL_CALLBACK_DISPOSITIONS
        ):
            raise CandidateTournamentError(
                "Callback observation cannot transition to that disposition."
            )
        observation.update(
            {
                "disposition": disposition,
                "retained_after_callback": retained_after_callback,
                "factor_pack_created": factor_pack_created,
                "evicted_portable_factor_state_sha256s": list(evicted),
            }
        )
        self._write_frontier()

    def callback_for_arm(
        self, arm_id: str, arm_index: int
    ) -> Callable[[DurableBehaviorCheckpoint], None]:
        if arm_id not in ARM_IDS or arm_index != ARM_IDS.index(arm_id):
            raise CandidateTournamentError("Unknown or reordered tournament arm.")
        if arm_id in self._events:
            raise CandidateTournamentError("Tournament arm callback already opened.")
        self._events[arm_id] = []

        def callback(event: DurableBehaviorCheckpoint) -> None:
            if not isinstance(event, DurableBehaviorCheckpoint):
                raise CandidateTournamentError("Callback event is not durable.")
            events = self._events[arm_id]
            expected_step = (
                CHECKPOINT_STEPS[len(events)]
                if len(events) < len(CHECKPOINT_STEPS)
                else None
            )
            if event.step != expected_step:
                raise CandidateTournamentError(
                    f"{arm_id} callback order expected {expected_step}, got {event.step}."
                )
            payload = event.factor_state.as_factor_payload()
            try:
                GuidedCheckpointRecoverySink._validate_v3_factor_payload(
                    payload,
                    expected_factor_state_sha256=event.factor_state.factor_state_sha256,
                )
            except Exception as error:
                raise CandidateTournamentError(
                    "Callback state is not full v3."
                ) from error
            metrics = event.metrics.as_dict()
            rank = tournament_behavior_rank(metrics)
            runtime_digest = event.factor_state.factor_state_sha256
            digest = portable_v3_factor_state_sha256(payload)
            events.append(event)
            observation = self._append_pending_observation(
                arm_id=arm_id,
                arm_index=arm_index,
                event=event,
                metrics=metrics,
                rank=rank,
                runtime_digest=runtime_digest,
                portable_digest=digest,
            )
            if event.step != 0 and not event.pareto_safe_vs_step0:
                self._finalize_observation(
                    observation,
                    disposition="ineligible-non-pareto",
                    retained_after_callback=False,
                    factor_pack_created=False,
                )
                return
            previous = self._retained.get(digest)
            if previous is not None:
                if previous.rank != rank or dict(previous.metrics) != metrics:
                    self._finalize_observation(
                        observation,
                        disposition="error-inconsistent-duplicate",
                        retained_after_callback=False,
                        factor_pack_created=False,
                    )
                    raise CandidateTournamentError(
                        "One portable factor state produced inconsistent candidate behavior."
                    )
                self._finalize_observation(
                    observation,
                    disposition="duplicate-retained-state",
                    retained_after_callback=True,
                    factor_pack_created=False,
                )
                return
            directory = self.retained_root / digest
            pack_path = directory / "factor_pack.pt"
            candidate = _RetainedState(
                arm_id=arm_id,
                arm_index=arm_index,
                step=event.step,
                rank=rank,
                runtime_factor_state_sha256=runtime_digest,
                portable_factor_state_sha256=digest,
                checkpoint_sha256=event.checkpoint_sha256,
                metrics=metrics,
                pack_path=pack_path,
                pack_sha256="",
            )
            keep = sorted(
                (*self._retained.values(), candidate),
                key=lambda value: value.total_order,
            )[:MAX_RETAINED_STATES]
            keep_hashes = {value.portable_factor_state_sha256 for value in keep}
            if digest not in keep_hashes:
                self._finalize_observation(
                    observation,
                    disposition="not-retained-rank-cutoff",
                    retained_after_callback=False,
                    factor_pack_created=False,
                )
                return
            directory.mkdir()
            _atomic_torch_save(pack_path, payload)
            candidate.pack_sha256 = stage._file_sha256(pack_path)
            self._retained[digest] = candidate
            evicted_digests: list[str] = []
            for evicted in tuple(self._retained):
                if evicted not in keep_hashes:
                    owned = self.retained_root / evicted
                    if owned.parent != self.retained_root or owned.name != evicted:
                        raise CandidateTournamentError(
                            "Refusing unsafe retained cleanup."
                        )
                    shutil.rmtree(owned)
                    del self._retained[evicted]
                    evicted_digests.append(evicted)
            _atomic_json(
                directory / "candidate_record.json",
                candidate.as_dict(),
            )
            self._finalize_observation(
                observation,
                disposition="retained-top-three",
                retained_after_callback=True,
                factor_pack_created=True,
                evicted=evicted_digests,
            )

        return callback

    def complete_arm(self, arm_id: str, result: Any) -> None:
        events = self._events.get(arm_id)
        if events is None or tuple(value.step for value in events) != CHECKPOINT_STEPS:
            raise CandidateTournamentError(
                f"{arm_id} durable checkpoint series incomplete."
            )
        if (
            getattr(result, "sealed_metrics", None) is not None
            or getattr(result, "sealed_evaluation_unlocked", None) is not False
            or getattr(result, "sealed_evaluation_policy", None) != "disabled"
        ):
            raise CandidateTournamentError("Tournament arm accessed sealed behavior.")
        terminal = getattr(result, "checkpoints", None)
        if not isinstance(terminal, tuple) or tuple(
            value.step for value in terminal
        ) != (CHECKPOINT_STEPS):
            raise CandidateTournamentError(
                "Terminal checkpoints disagree with callbacks."
            )
        for event, row in zip(events, terminal, strict=True):
            if (
                event.factor_state.factor_state_sha256 != row.factor_state_sha256
                or event.metrics.as_dict() != row.metrics.as_dict()
                or event.pareto_safe_vs_step0 != row.pareto_safe_vs_step0
            ):
                raise CandidateTournamentError("Terminal checkpoint evidence changed.")
        observations = [
            value for value in self._observed_callbacks if value["arm_id"] == arm_id
        ]
        if len(observations) != len(CHECKPOINT_STEPS) or any(
            value["disposition"] == "observed-pending" for value in observations
        ):
            raise CandidateTournamentError(
                "Arm terminal verification found incomplete callback evidence."
            )
        for observation, row in zip(observations, terminal, strict=True):
            observation["terminal_verified"] = True
            observation["terminal_pareto_safe_vs_step0"] = row.pareto_safe_vs_step0
        self._completed.add(arm_id)
        self._write_frontier()

    def _write_frontier(self) -> None:
        ordered = sorted(self._retained.values(), key=lambda value: value.total_order)
        payload = {
            "schema": FRONTIER_SCHEMA,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "completed_arms": [name for name in ARM_IDS if name in self._completed],
            "retained_count": len(ordered),
            "retained": [value.as_dict() for value in ordered],
            "observed_callback_count": len(self._observed_callbacks),
            "pending_callback_count": sum(
                value["disposition"] == "observed-pending"
                for value in self._observed_callbacks
            ),
            "observed_callbacks": [dict(value) for value in self._observed_callbacks],
            "distinct_full_v3_states_only": True,
            "portable_content_identity_used": True,
            "step_zero_ranked": True,
            "later_states_require_pareto_safety_vs_step_zero": True,
            "sealed_metrics_used": False,
            "internal_numeric_matching_used": False,
            "callback_metrics_are_durable_without_factor_pack_retention": True,
            "maximum_retained_factor_packs": MAX_RETAINED_STATES,
        }
        _atomic_json(self.output_root / "candidate_frontier.json", payload)

    def frontier_binding(self) -> dict[str, Any]:
        path = self.output_root / "candidate_frontier.json"
        if not path.is_file():
            raise CandidateTournamentError("Candidate frontier is absent.")
        return {
            "schema": FRONTIER_SCHEMA,
            "path": str(path),
            "sha256": stage._file_sha256(path),
            "observed_callback_count": len(self._observed_callbacks),
            "pending_callback_count": sum(
                value["disposition"] == "observed-pending"
                for value in self._observed_callbacks
            ),
            "completed_arms": [name for name in ARM_IDS if name in self._completed],
        }

    def finalize(self) -> tuple[dict[str, Any], tuple[_RetainedState, ...]]:
        if self._completed != set(ARM_IDS):
            raise CandidateTournamentError("Cannot select a winner before all arms.")
        if len(self._observed_callbacks) != EXPECTED_CALLBACK_COUNT or any(
            value["disposition"] == "observed-pending"
            or value["disposition"] not in FINAL_CALLBACK_DISPOSITIONS
            or value["terminal_verified"] is not True
            or value["terminal_pareto_safe_vs_step0"] != value["pareto_safe_vs_step0"]
            for value in self._observed_callbacks
        ):
            raise CandidateTournamentError(
                "Cannot select a winner from incomplete callback evidence."
            )
        ordered = tuple(
            sorted(self._retained.values(), key=lambda value: value.total_order)
        )
        if len(ordered) != MAX_RETAINED_STATES:
            raise CandidateTournamentError("Tournament produced no complete top three.")
        winner = ordered[0]
        self._write_frontier()
        frontier_binding = self.frontier_binding()
        payload = {
            "schema": WINNER_SCHEMA,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "winner": winner.as_dict(),
            "top_three_factor_state_sha256s": [
                value.runtime_factor_state_sha256 for value in ordered
            ],
            "top_three_portable_factor_state_sha256s": [
                value.portable_factor_state_sha256 for value in ordered
            ],
            "candidate_only": True,
            "sealed_evaluation_accessed": False,
            "model_checkpoint_written": False,
            "publication_attempted": False,
            "stop_reason": "candidate-winner-selected-before-sealed",
            "candidate_frontier": frontier_binding,
        }
        _atomic_json(
            self.output_root / "winner_selection.json", payload, exclusive=True
        )
        return payload, ordered


def _atomic_json(
    path: Path, payload: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload), allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise CandidateTournamentError(f"Refusing to overwrite {path}.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise CandidateTournamentError(f"Refusing to overwrite {path}.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ArmExecution:
    result: Any
    reset_state_sha256: str
    rollback_state_sha256: str
    stored_initial_factor_state_sha256: str
    portable_initial_factor_state_sha256: str | None
    runtime_rebound_initial_factor_state_sha256: str
    trainer_initial_factor_state_sha256: str
    site_mask_enabled: bool
    post_optimizer_mask_verification_count: int
    sealed_forward_count: int


class ProductionCandidateTournamentRuntime(guided.ProductionGuidedRuntime):
    """Load/compile once; install, authenticate, train, and roll back each arm."""

    def __init__(
        self,
        *,
        dtype: str = "float16",
        parent_device_map: str = EXPECTED_PARENT_DEVICE_MAP,
        target_device_map: str = EXPECTED_TARGET_DEVICE_MAP,
        cuda_preflight_report: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            dtype != "float16"
            or parent_device_map != EXPECTED_PARENT_DEVICE_MAP
            or target_device_map != EXPECTED_TARGET_DEVICE_MAP
        ):
            raise CandidateTournamentError(
                "Production tournament runtime must use the preregistered FP16 "
                "3090/RTX8000 logical device mapping."
            )
        super().__init__(
            dtype=dtype,
            parent_device_map=parent_device_map,
            target_device_map=target_device_map,
        )
        self.cuda_preflight_report = (
            None if cuda_preflight_report is None else dict(cuda_preflight_report)
        )
        self.parent: Any = None
        self.compiled: Any = None
        self.compiled_state_sha256: str | None = None

    def prepare(self, contract: AuthenticatedCandidateTournament) -> Mapping[str, Any]:
        base = contract.source.execution_contract
        cache = self.prepare_compiler_cache(base)
        parent, target, tokenizer = self.load_models(base)
        compiled = self.compile(parent, target, tokenizer, base)
        compiler_report = stage._compiled_report_dict(compiled)
        compiled_sha = self.state_sha256(compiled.child_model)
        stage._validate_continuation_compilation(base, compiler_report, compiled_sha)
        compiled = self.finalize_compiled_base(compiled, base)
        if self.state_sha256(compiled.child_model) != compiled_sha:
            raise CandidateTournamentError("Compiler finalization changed child bytes.")
        self.parent = parent
        self.compiled = compiled
        self.compiled_state_sha256 = compiled_sha
        del target, tokenizer
        report = {
            "compiler_report": compiler_report,
            "compiled_state_sha256": compiled_sha,
            "compiler_cache": cache,
            "models_loaded": True,
            "gpu_work_launched": True,
        }
        if self.cuda_preflight_report is not None:
            report["cuda_preflight"] = dict(self.cuda_preflight_report)
        return report

    def run_arm(
        self,
        contract: AuthenticatedCandidateTournament,
        arm: TournamentArm,
        callback: Callable[[DurableBehaviorCheckpoint], None],
    ) -> ArmExecution:
        if (
            self.parent is None
            or self.compiled is None
            or self.compiled_state_sha256 is None
        ):
            raise CandidateTournamentError("Runtime is not prepared.")
        if self.state_sha256(self.compiled.child_model) != self.compiled_state_sha256:
            raise CandidateTournamentError("Compiled child was not reset between arms.")
        transaction: Any = None
        try:
            transaction = self.install_transaction(self.compiled)
            self.restore_source_v2(transaction, contract.source)
            self.install_guided_transaction(transaction, contract.source)
            source_v2_state = capture_target_native_factor_state(transaction)
            live_source_binding = authenticate_guided_factor_sources(
                transaction,
                self._guided_tranche,
                source_v2_state,
            )
            del source_v2_state
            gc.collect()
            if arm.arm_id in contract.initial_states:
                stored = contract.initial_states[arm.arm_id]
                initial_payload = rebind_recovered_v3_factor_payload(
                    stored.payload,
                    live_source_binding,
                )
                portable_initial_sha = portable_v3_factor_state_sha256(
                    initial_payload,
                    live_source_binding,
                )
                if portable_initial_sha != stored.portable_factor_state_sha256:
                    raise CandidateTournamentError(
                        "Runtime rebinding changed the preregistered portable v3 state."
                    )
                runtime_initial_sha = _sha256(
                    initial_payload.get("factor_state_sha256"),
                    "runtime rebound initial factor state",
                )
            else:
                continuation = contract.source.execution_contract.continuation
                if continuation is None:
                    raise CandidateTournamentError(
                        "Source v2 continuation disappeared."
                    )
                initial_payload = continuation.factor_payload
                portable_initial_sha = None
                runtime_initial_sha = continuation.factor_state_sha256
            module = importlib.import_module(guided.GUIDED_TRAINER_MODULE)
            loader = getattr(module, "load_behavior_domain_suite")
            domains = {
                name: loader(
                    name,
                    contract.splits[name].payload,
                    artifact_sha256=contract.splits[name].sha256,
                )
                for name in ("fit", "candidate")
            }
            threshold_type = getattr(module, "CandidateBehaviorThresholds")
            thresholds = threshold_type(
                **contract.source.execution_contract.thresholds["candidate"].as_dict()
            )
            trainer = getattr(module, "train_target_native_behavior_distillation")
            mask_count = 0
            guard: GradientSiteMaskGuard | None = None

            def verify_mask(_step: int) -> None:
                nonlocal mask_count
                if guard is None:
                    raise CandidateTournamentError("Site-mask guard is absent.")
                guard.verify_step()
                mask_count += 1

            guard_context: Any = contextlib.nullcontext()
            post_step: Callable[[int], None] | None = None
            if arm.site_mask_enabled:
                registry = getattr(self._guided_tranche, "guided_registry", None)
                guard = GradientSiteMaskGuard(registry, contract.site_mask)
                guard_context = guard
                post_step = verify_mask
            with guard_context:
                result = trainer(
                    transaction=transaction,
                    parent_model=self.parent,
                    domains=domains,
                    schedule=contract.schedule,
                    candidate_thresholds=thresholds,
                    seed=contract.source.execution_contract.seed,
                    initial_factor_payload=initial_payload,
                    guided_tranche=self._guided_tranche,
                    durable_checkpoint_callback=callback,
                    post_optimizer_step_callback=post_step,
                )
            if arm.site_mask_enabled and mask_count != contract.schedule.steps:
                raise CandidateTournamentError(
                    "Mask was not verified after every step."
                )
            initial_sha = getattr(result, "initial_factor_state_sha256", None)
            if initial_sha != runtime_initial_sha:
                raise CandidateTournamentError("Trainer restored another arm source.")
            self.rollback_guided_installation(transaction)
            transaction.rollback()
            rollback_sha = self.state_sha256(self.compiled.child_model)
            if rollback_sha != self.compiled_state_sha256:
                raise CandidateTournamentError("Arm rollback changed compiled bytes.")
            return ArmExecution(
                result=result,
                reset_state_sha256=self.compiled_state_sha256,
                rollback_state_sha256=rollback_sha,
                stored_initial_factor_state_sha256=arm.initial_factor_state_sha256,
                portable_initial_factor_state_sha256=portable_initial_sha,
                runtime_rebound_initial_factor_state_sha256=runtime_initial_sha,
                trainer_initial_factor_state_sha256=initial_sha,
                site_mask_enabled=arm.site_mask_enabled,
                post_optimizer_mask_verification_count=mask_count,
                sealed_forward_count=0,
            )
        except BaseException:
            if transaction is not None:
                try:
                    self.rollback_guided_installation(transaction)
                    if getattr(transaction, "is_live", False):
                        transaction.rollback()
                finally:
                    if self.state_sha256(self.compiled.child_model) != (
                        self.compiled_state_sha256
                    ):
                        raise CandidateTournamentError(
                            "Failed arm could not restore exact compiled bytes."
                        )
            raise

    def close(self) -> None:
        self.parent = None
        self.compiled = None
        gc.collect()


def execute_candidate_tournament(
    contract: AuthenticatedCandidateTournament,
    output_root: Path | str,
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Run all arms, write one candidate winner, and stop before sealed."""

    if type(contract) is not AuthenticatedCandidateTournament:
        raise CandidateTournamentError(
            "execute_candidate_tournament requires an authenticated tournament contract."
        )
    contract = authenticate_candidate_tournament_preregistration(
        contract.path, contract.file_sha256
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise CandidateTournamentError("output_root must not already exist.")
    cuda_preflight = None
    if runtime is None:
        cuda_preflight = validate_candidate_tournament_cuda_preflight(
            contract,
            dtype=str(contract.runtime["dtype"]),
            parent_device_map=str(contract.runtime["parent_device_map"]),
            target_device_map=str(contract.runtime["target_device_map"]),
        )
        runner = ProductionCandidateTournamentRuntime(
            dtype=str(contract.runtime["dtype"]),
            parent_device_map=str(contract.runtime["parent_device_map"]),
            target_device_map=str(contract.runtime["target_device_map"]),
            cuda_preflight_report=cuda_preflight,
        )
    else:
        runner = runtime
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        **contract.preflight_report(),
        "status": "running",
        "output_root": str(output),
        "arm_executions": [],
    }
    if cuda_preflight is not None:
        report["cuda_preflight"] = cuda_preflight
    sink = TournamentTopThreeSink(
        output, evaluation_contract_sha256=contract.evaluation_contract_sha256
    )
    try:
        preparation = runner.prepare(contract)
        report["runtime_preparation"] = dict(preparation)
        compiled_sha = preparation.get("compiled_state_sha256")
        for index, arm in enumerate(contract.arms):
            execution = runner.run_arm(
                contract, arm, sink.callback_for_arm(arm.arm_id, index)
            )
            if (
                execution.sealed_forward_count != 0
                or execution.reset_state_sha256 != compiled_sha
                or execution.rollback_state_sha256 != compiled_sha
                or execution.stored_initial_factor_state_sha256
                != arm.initial_factor_state_sha256
                or execution.portable_initial_factor_state_sha256
                != arm.initial_portable_factor_state_sha256
                or execution.trainer_initial_factor_state_sha256
                != execution.runtime_rebound_initial_factor_state_sha256
                or execution.site_mask_enabled != arm.site_mask_enabled
                or (
                    arm.site_mask_enabled
                    and execution.post_optimizer_mask_verification_count != 64
                )
                or (
                    not arm.site_mask_enabled
                    and execution.post_optimizer_mask_verification_count != 0
                )
            ):
                raise CandidateTournamentError("Arm execution receipt failed.")
            sink.complete_arm(arm.arm_id, execution.result)
            report["arm_executions"].append(
                {
                    "arm_id": arm.arm_id,
                    "reset_state_sha256": execution.reset_state_sha256,
                    "rollback_state_sha256": execution.rollback_state_sha256,
                    "stored_initial_factor_state_sha256": (
                        execution.stored_initial_factor_state_sha256
                    ),
                    "portable_initial_factor_state_sha256": (
                        execution.portable_initial_factor_state_sha256
                    ),
                    "runtime_rebound_initial_factor_state_sha256": (
                        execution.runtime_rebound_initial_factor_state_sha256
                    ),
                    "trainer_initial_factor_state_sha256": (
                        execution.trainer_initial_factor_state_sha256
                    ),
                    "site_mask_enabled": execution.site_mask_enabled,
                    "post_optimizer_mask_verification_count": (
                        execution.post_optimizer_mask_verification_count
                    ),
                    "sealed_forward_count": 0,
                    "sealed_suite_supplied": False,
                    "candidate_evidence_sha256": getattr(
                        execution.result, "evidence_sha256", None
                    ),
                }
            )
        winner, retained = sink.finalize()
        if not (output / "winner_selection.json").is_file():
            raise CandidateTournamentError("Winner was not durably selected.")
        frontier_binding = sink.frontier_binding()
        if (
            winner.get("candidate_frontier") != frontier_binding
            or frontier_binding["observed_callback_count"] != EXPECTED_CALLBACK_COUNT
            or frontier_binding["pending_callback_count"] != 0
            or frontier_binding["completed_arms"] != list(ARM_IDS)
        ):
            raise CandidateTournamentError(
                "Winner is not bound to the complete callback frontier."
            )
        report.update(
            {
                "status": "candidate-winner-selected",
                "winner_selection": winner,
                "top_three_factor_state_sha256s": [
                    value.runtime_factor_state_sha256 for value in retained
                ],
                "top_three_portable_factor_state_sha256s": [
                    value.portable_factor_state_sha256 for value in retained
                ],
                "sealed_forward_count": 0,
                "sealed_evaluation_accessed": False,
                "model_checkpoint_written": False,
                "publication_attempted": False,
                "stop_after_candidate_winner": True,
                "candidate_frontier": frontier_binding,
            }
        )
        _atomic_json(output / "run_report.json", report, exclusive=True)
        return report
    except BaseException as error:
        try:
            failure_frontier = sink.frontier_binding()
        except Exception:
            failure_frontier = None
        failure = {
            **report,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "sealed_forward_count": 0,
            "sealed_evaluation_accessed": False,
            "model_checkpoint_written": False,
            "publication_attempted": False,
            "candidate_frontier": failure_frontier,
        }
        if not (output / "run_report.json").exists():
            _atomic_json(output / "run_report.json", failure, exclusive=True)
        raise
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--parent-device-map", default=EXPECTED_PARENT_DEVICE_MAP)
    parser.add_argument("--target-device-map", default=EXPECTED_TARGET_DEVICE_MAP)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_candidate_tournament_preregistration(
        args.preregistration, args.preregistration_sha256
    )
    cuda_preflight = validate_candidate_tournament_cuda_preflight(
        contract,
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
    )
    if args.preflight:
        report = contract.preflight_report()
        report["cuda_preflight"] = cuda_preflight
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        raise CandidateTournamentError("--execute requires --output-root.")
    runtime = ProductionCandidateTournamentRuntime(
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
        cuda_preflight_report=cuda_preflight,
    )
    report = execute_candidate_tournament(contract, args.output_root, runtime=runtime)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
