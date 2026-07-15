#!/usr/bin/env python3
"""Run the candidate-only v4 guided-GUD basis tournament.

This is a continuation of the completed v3 residual-attention tournament, not
a new compiler run.  Preparation authenticates the retained step-40/step-12
v3 winner and a separately produced output-Jacobian GUD basis audit.  Execution
installs one exact-zero Gate/Up/Down product on that same recovered QKVO state,
compares four basis methods with and without a coupled G/U + independent Down
mask, retains at most three full v4 factor packs, and stops before sealed data.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

import prepare_qwen_exact_target_native_continuation_splits as split_tools
import run_qwen_exact_target_native_guided_candidate_tournament as v3_tournament
import run_qwen_exact_target_native_guided_winner_publication as v3_publication
import run_qwen_exact_target_native_stage as stage
import neural_forge.hf_gud_output_jacobian_audit as gud_audit
import neural_forge.hf_gud_output_jacobian_replay as gud_replay
import neural_forge.hf_guided_checkpoint_recovery as checkpoint_recovery
import neural_forge.hf_target_native_behavior_distillation as behavior
from neural_forge import hf_dense_swiglu_projection_residual_adapter as gud_adapter
from neural_forge.hf_guided_checkpoint_recovery import (
    GuidedCheckpointRecoverySink,
    canonical_json_bytes,
    canonical_sha256,
    portable_v3_factor_state_sha256,
    portable_v4_factor_state_sha256,
    rebind_recovered_v3_factor_payload,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BehaviorDistillationSchedule,
    DurableBehaviorCheckpoint,
    authenticate_guided_factor_sources,
    capture_target_native_factor_state,
)


PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-gud-candidate-tournament-"
    "preregistration-v1"
)
RUN_SCHEMA = "neural-forge-qwen-exact-target-native-guided-gud-candidate-tournament-v1"
WINNER_SCHEMA = "neural-forge-guided-gud-candidate-tournament-winner-v1"
FRONTIER_SCHEMA = "neural-forge-guided-gud-candidate-tournament-frontier-v1"
BASIS_BUNDLE_SCHEMA = "neural-forge-guided-gud-candidate-basis-bundle-v1"
SITE_MASK_SCHEMA = "neural-forge-guided-gud-layer-mask-v1"
SITE_MASK_DERIVATION_SCHEMA = "neural-forge-guided-gud-layer-mask-derivation-v1"
MASK_SELECTION_POLICY = (
    "top-half-by-min-guided-capture-fraction-then-min-guided-over-random-"
    "then-min-normal-fraction-then-layer-ascending"
)

BasisMethod = Literal[
    "output-jacobian", "balanced", "procrustes-hybrid", "random-control"
]
MaskMode = Literal["all-site", "masked"]
BASIS_METHODS: tuple[BasisMethod, ...] = (
    "output-jacobian",
    "balanced",
    "procrustes-hybrid",
    "random-control",
)
MASK_MODES: tuple[MaskMode, ...] = ("all-site", "masked")
ARM_IDS = tuple(f"{method}-{mode}" for method in BASIS_METHODS for mode in MASK_MODES)
CHECKPOINT_STEPS = (0, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64)
EXPECTED_CALLBACK_COUNT = len(ARM_IDS) * len(CHECKPOINT_STEPS)
MAX_RETAINED_STATES = 3
SPLIT_COUNTS = {"fit": 192, "candidate": 24, "sealed": 32}
GUD_ROLES = ("gate", "up", "down")
EXPECTED_GEOMETRY = gud_audit.GUDStageGeometry(30, 1536, 5120).to_dict()
EXPECTED_RANK = 8
EXPECTED_GUIDED_GUD_PARAMETER_COUNT = 4_792_320
EXPECTED_SOURCE_ARM_ID = "step40-v3-trajectory"
EXPECTED_SOURCE_STEP = 12
EXPECTED_SOURCE_PORTABLE_V3_SHA256 = (
    "4108d205826b71d1b324082d781b31662318eda83c1e57086f931dc234f11227"
)
EXPECTED_SOURCE_TOURNAMENT_PREREGISTRATION_SHA256 = (
    "5a603e0c90f5db2d1a891bbf59a8cbb6116d19e98c00b11b62ed7cacd1aa1fed"
)
EXPECTED_SOURCE_FACTOR_PACK_FILE_SHA256 = (
    "ec18c6e1b671a7bfe2a8ea4557fb94e53abd990072bf0f647a15eab80fff27c1"
)
EXPECTED_SOURCE_RUNTIME_V3_SHA256 = (
    "557f7c52369d63958b7dd8ffce95ce18cab88289b9b287bc04426763a9a686b7"
)
EXPECTED_SOURCE_EVALUATION_CONTRACT_SHA256 = (
    "0f99a02f9711ac8633c763caa6b01cbb2bbcfb6e3fc91de44edda8f517f2e587"
)
EXPECTED_SOURCE_RUN_REPORT_SHA256 = (
    "86d47fa3a4e65d9c22efa6e8188bf85eba3c2d312c0dd969c67e82fd5446a902"
)
EXPECTED_SOURCE_WINNER_SELECTION_SHA256 = (
    "b432e6bd3c6ef8434ff8745361e95cc7baae502a51f2d434fc150de38d85e263"
)
EXPECTED_SOURCE_CANDIDATE_FRONTIER_SHA256 = (
    "cb0068bb4d1a646d1368600394b9e276ca15ada12606ce803f16a25e9b25b937"
)
HISTORICAL_V3_RUN_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-candidate-tournament-v1"
)
HISTORICAL_V3_WINNER_SCHEMA = "neural-forge-guided-candidate-tournament-winner-v1"
HISTORICAL_V3_FRONTIER_SCHEMA = (
    "neural-forge-guided-candidate-tournament-frontier-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GuidedGUDCandidateTournamentError(v3_tournament.CandidateTournamentError):
    """A v4 source, audit, reset, callback, or candidate-only gate failed."""


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GuidedGUDCandidateTournamentError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise GuidedGUDCandidateTournamentError(
            f"{name} must contain exactly {sorted(keys)}; got {actual!r}."
        )
    return value


def _resolve(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise GuidedGUDCandidateTournamentError(f"{name} must be a nonempty path.")
    raw = Path(value).expanduser()
    candidate = base / raw if not raw.is_absolute() else raw
    cursor = candidate.absolute()
    while True:
        if cursor.exists() and cursor.is_symlink():
            raise GuidedGUDCandidateTournamentError(
                f"{name} cannot traverse a symlink."
            )
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    return candidate.resolve()


def _file_binding(value: Any, *, base: Path, name: str) -> tuple[Path, str]:
    row = _strict(value, {"path", "sha256"}, name)
    path = _resolve(base, row["path"], f"{name}.path")
    digest = _sha256(row["sha256"], f"{name}.sha256")
    if not path.is_file() or stage._file_sha256(path) != digest:
        raise GuidedGUDCandidateTournamentError(f"{name} is absent or changed.")
    return path, digest


def _canonical_report_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    embedded = payload.pop(field, None)
    expected = _sha256(embedded, field)
    if canonical_sha256(payload) != expected:
        raise GuidedGUDCandidateTournamentError(
            f"{field} does not authenticate its report."
        )
    return expected


def guided_gud_candidate_schedule() -> BehaviorDistillationSchedule:
    """Return the one fixed 64-step v4 behavior schedule."""

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


def guided_gud_basis_audit_schedule() -> BehaviorDistillationSchedule:
    """Use the tournament loss contract at the replay audit's exact horizon."""

    return dataclasses.replace(
        guided_gud_candidate_schedule(),
        fit_horizon=8,
        warmup_fit_horizon=8,
    )


def guided_gud_basis_audit_config(seed: int) -> gud_replay.GUDOutputJacobianReplayConfig:
    """Return the complete pinned production replay configuration."""

    return gud_replay.GUDOutputJacobianReplayConfig(
        prompt_count=24,
        horizon=8,
        rank=8,
        sketch_width=16,
        balanced_envelope_width=32,
        balanced_stream_rows=16,
        fingerprint_lanes=8,
        fingerprint_chunk_elements=32 * 1024,
        max_callback_rows=256,
        seed=seed,
        min_normal_fraction=0.0,
        min_guided_over_random_ratio=0.0,
        max_bytes=gud_audit.HARD_MAX_AUXILIARY_BYTES,
    )


def guided_gud_stage_contract() -> list[dict[str, Any]]:
    return [
        {
            "start_step": 1,
            "end_step": 8,
            "guided_gud_stage": "guided-left",
            "trainable_products": ["guided-gud-left"],
            "guided_gud_right_frozen": True,
            "recovered_v3_qkvo_frozen": True,
        },
        {
            "start_step": 9,
            "end_step": 24,
            "guided_gud_stage": "guided",
            "trainable_products": ["guided-gud-left", "guided-gud-right"],
            "guided_gud_right_frozen": False,
            "recovered_v3_qkvo_frozen": True,
        },
        {
            "start_step": 25,
            "end_step": 64,
            "guided_gud_stage": "joint",
            "trainable_products": ["guided-gud", "recovered-v3-qkvo"],
            "guided_gud_right_frozen": False,
            "recovered_v3_qkvo_frozen": False,
        },
    ]


def guided_gud_runtime_contract() -> dict[str, Any]:
    result = dict(v3_tournament.candidate_tournament_runtime_contract())
    result.update(
        {
            "source_v3_reset_between_arms": True,
            "guided_gud_exact_zero_install_each_arm": True,
            "guided_gud_parameter_count": EXPECTED_GUIDED_GUD_PARAMETER_COUNT,
            "basis_methods": list(BASIS_METHODS),
            "mask_modes": list(MASK_MODES),
            "qkvo_frozen_through_step": 24,
            "guided_gud_right_frozen_through_step": 8,
            "full_v4_callback_only": True,
            "maximum_retained_factor_packs": MAX_RETAINED_STATES,
            "sealed_forward_count": 0,
        }
    )
    return result


def guided_gud_selection_contract() -> dict[str, Any]:
    return {
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
        "retain_distinct_full_v4_states": MAX_RETAINED_STATES,
        "distinct_identity": "portable-v4-factor-content",
        "rank_step_zero": True,
        "later_states_require_pareto_safety_vs_step_zero": True,
        "sealed_disabled": True,
        "stop_after_candidate_winner": True,
    }


def guided_gud_basis_bundle_from_audit(
    result: gud_replay.GUDOutputJacobianReplayResult,
) -> dict[str, Any]:
    """Package the four production replay packs without model or replay state."""

    if not isinstance(result, gud_replay.GUDOutputJacobianReplayResult):
        raise GuidedGUDCandidateTournamentError(
            "basis bundle requires GUDOutputJacobianReplayResult."
        )
    replay_report = result.report.to_dict()
    _canonical_report_sha256(replay_report, "report_sha256")
    geometry = _strict(
        replay_report.get("geometry"),
        set(EXPECTED_GEOMETRY),
        "replay audit geometry",
    )
    if (
        dict(geometry) != EXPECTED_GEOMETRY
        or replay_report.get("schema")
        != gud_replay.GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA
        or replay_report.get("basis_pack_schema")
        != gud_replay.GUD_COMPLETE_BASIS_PACK_SCHEMA
        or replay_report.get("candidate_order") != list(BASIS_METHODS)
        or replay_report.get("two_pass_replay_identical") is not True
        or replay_report.get("gate_up_separate_hooks") is not True
        or replay_report.get("full_downstream_loss_output_delta_captured") is not True
        or replay_report.get("role_local_projection_input_captured") is not True
        or replay_report.get("internal_weight_matching_used") is not False
        or replay_report.get("source_v3_state_unchanged") is not True
        or replay_report.get("source_v3_factor_count")
        != len(checkpoint_recovery._V3_FACTOR_NAMES)
        or isinstance(replay_report.get("source_v3_lexical_rank"), bool)
        or not isinstance(replay_report.get("source_v3_lexical_rank"), int)
        or replay_report.get("source_v3_lexical_rank") <= 0
    ):
        raise GuidedGUDCandidateTournamentError(
            "Basis result is not the complete production GUD replay audit."
        )
    config = replay_report.get("config")
    try:
        authenticated_config = gud_replay.GUDOutputJacobianReplayConfig(
            **dict(config)
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise GuidedGUDCandidateTournamentError(
            "Production GUD replay configuration is malformed."
        ) from error
    if (
        not isinstance(config, Mapping)
        or authenticated_config.to_dict() != dict(config)
        or authenticated_config.prompt_count != 24
        or authenticated_config.horizon != 8
        or authenticated_config.rank != EXPECTED_RANK
        or replay_report.get("config_sha256") != authenticated_config.sha256()
    ):
        raise GuidedGUDCandidateTournamentError(
            "Production GUD replay dimensions changed."
        )
    runtime_before = _sha256(
        replay_report.get("source_v3_runtime_factor_state_sha256_before"),
        "replay source v3 runtime SHA before",
    )
    runtime_after = _sha256(
        replay_report.get("source_v3_runtime_factor_state_sha256_after"),
        "replay source v3 runtime SHA after",
    )
    portable_sha = _sha256(
        replay_report.get("source_v3_portable_factor_state_sha256"),
        "replay source v3 portable SHA",
    )
    factor_manifest_sha = _sha256(
        replay_report.get("source_v3_factor_manifest_sha256"),
        "replay source v3 factor manifest SHA",
    )
    binding_sha = _sha256(
        replay_report.get("source_guided_qkvo_binding_sha256"),
        "replay guided QKVO binding SHA",
    )
    raw_source_binding = replay_report.get("source_guided_qkvo_binding")
    try:
        source_binding = checkpoint_recovery._source_binding(
            raw_source_binding,
            "replay guided QKVO binding",
        )
    except Exception as error:
        raise GuidedGUDCandidateTournamentError(
            "Replay guided QKVO source binding is malformed."
        ) from error
    source_tensor_hashes = replay_report.get("source_v3_tensor_sha256s")
    if (
        runtime_before != runtime_after
        or source_binding.binding_sha256 != binding_sha
        or not isinstance(source_tensor_hashes, Mapping)
        or set(source_tensor_hashes) != checkpoint_recovery._V3_FACTOR_NAMES
        or any(
            _SHA256.fullmatch(value) is None
            for value in source_tensor_hashes.values()
            if isinstance(value, str)
        )
        or any(not isinstance(value, str) for value in source_tensor_hashes.values())
    ):
        raise GuidedGUDCandidateTournamentError(
            "Replay did not preserve and authenticate the complete v3 source."
        )
    layers = int(geometry["logical_layers"])
    widths = {
        "gate": int(geometry["hidden_size"]),
        "up": int(geometry["hidden_size"]),
        "down": int(geometry["intermediate_size"]),
    }
    rows: dict[str, dict[str, Tensor]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for method in BASIS_METHODS:
        source_pack = result.basis_packs.get(method)
        if not isinstance(source_pack, Mapping) or set(source_pack) != set(GUD_ROLES):
            raise GuidedGUDCandidateTournamentError(
                f"Replay basis is incomplete for {method}."
            )
        per_role: dict[str, Tensor] = {}
        for role in GUD_ROLES:
            value = source_pack[role]
            expected_shape = (layers, EXPECTED_RANK, widths[role])
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or not value.is_contiguous()
                or tuple(value.shape) != expected_shape
                or not bool(torch.isfinite(value).all())
            ):
                raise GuidedGUDCandidateTournamentError(
                    f"Replay basis {method}/{role} is not finite contiguous CPU "
                    f"FP32 {expected_shape}."
                )
            per_role[role] = value.detach().clone().contiguous()
        rows[method] = per_role
        hashes[method] = {
            role: gud_adapter._tensor_sha256(per_role[role]) for role in GUD_ROLES
        }
    manifest = [
        {
            "candidate": method,
            "role": role,
            "shape": list(rows[method][role].shape),
            "dtype": str(rows[method][role].dtype),
            "sha256": hashes[method][role],
        }
        for method in BASIS_METHODS
        for role in GUD_ROLES
    ]
    manifest_sha = canonical_sha256(manifest)
    if (
        replay_report.get("basis_pack_manifest") != manifest
        or replay_report.get("basis_pack_manifest_sha256") != manifest_sha
    ):
        raise GuidedGUDCandidateTournamentError(
            "Replay basis tensors differ from the authenticated pack manifest."
        )
    tensor_count = len(BASIS_METHODS) * len(GUD_ROLES)
    total_numel = sum(
        value.numel() for method in rows.values() for value in method.values()
    )
    total_nbytes = sum(
        value.numel() * value.element_size()
        for method in rows.values()
        for value in method.values()
    )
    metadata = {
        "schema": BASIS_BUNDLE_SCHEMA,
        "audit_schema": gud_replay.GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA,
        "basis_pack_schema": gud_replay.GUD_COMPLETE_BASIS_PACK_SCHEMA,
        "geometry": dict(geometry),
        "rank": EXPECTED_RANK,
        "candidate_order": list(BASIS_METHODS),
        "replay_audit_report": replay_report,
        "audit_source_v3_runtime_factor_state_sha256_before": runtime_before,
        "audit_source_v3_runtime_factor_state_sha256_after": runtime_after,
        "source_v3_portable_factor_state_sha256": portable_sha,
        "source_v3_factor_manifest_sha256": factor_manifest_sha,
        "source_v3_tensor_sha256s": dict(source_tensor_hashes),
        "source_guided_qkvo_binding": source_binding.as_dict(),
        "source_guided_qkvo_binding_sha256": binding_sha,
        "source_v3_state_unchanged": True,
        "source_basis_tensor_sha256s": hashes,
        "basis_manifest_sha256": manifest_sha,
        "tensor_count": tensor_count,
        "total_numel": total_numel,
        "total_nbytes": total_nbytes,
        "contains_model_state": False,
        "contains_parent_state": False,
        "contains_optimizer_state": False,
        "contains_gradients": False,
        "contains_activations": False,
        "contains_hidden_states": False,
        "contains_replay_blocks": False,
    }
    payload_sha = canonical_sha256(metadata)
    return {
        **metadata,
        "basis_payload_sha256": payload_sha,
        "guided_right_rows": rows,
    }


@dataclass(frozen=True, slots=True)
class AuthenticatedGuidedGUDBasisBundle:
    path: Path
    file_sha256: str
    payload_sha256: str
    audit_report_sha256: str
    replay_audit_report: Mapping[str, Any]
    basis_manifest_sha256: str
    audit_source_v3_runtime_factor_state_sha256_before: str
    audit_source_v3_runtime_factor_state_sha256_after: str
    source_v3_portable_factor_state_sha256: str
    source_v3_factor_manifest_sha256: str
    source_v3_tensor_sha256s: Mapping[str, str]
    source_guided_qkvo_binding: Mapping[str, Any]
    source_guided_qkvo_binding_sha256: str
    source_v3_state_unchanged: bool
    geometry: Mapping[str, int]
    rank: int
    rows: Mapping[str, Mapping[str, Tensor]]
    source_basis_tensor_sha256s: Mapping[str, Mapping[str, str]]

    def rows_for(self, method: BasisMethod) -> Mapping[str, Tensor]:
        if method not in BASIS_METHODS:
            raise GuidedGUDCandidateTournamentError("Unknown GUD basis method.")
        return self.rows[method]


def authenticate_guided_gud_basis_bundle(
    path: Path | str,
    expected_sha256: str,
) -> AuthenticatedGuidedGUDBasisBundle:
    bundle_path = Path(path).expanduser().resolve()
    digest = _sha256(expected_sha256, "basis_bundle.sha256")
    if (
        bundle_path.is_symlink()
        or not bundle_path.is_file()
        or stage._file_sha256(bundle_path) != digest
    ):
        raise GuidedGUDCandidateTournamentError("Basis bundle is absent or changed.")
    try:
        raw = torch.load(bundle_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise GuidedGUDCandidateTournamentError(
            "Basis bundle cannot load safely."
        ) from error
    fields = {
        "schema",
        "audit_schema",
        "basis_pack_schema",
        "geometry",
        "rank",
        "candidate_order",
        "replay_audit_report",
        "audit_source_v3_runtime_factor_state_sha256_before",
        "audit_source_v3_runtime_factor_state_sha256_after",
        "source_v3_portable_factor_state_sha256",
        "source_v3_factor_manifest_sha256",
        "source_v3_tensor_sha256s",
        "source_guided_qkvo_binding",
        "source_guided_qkvo_binding_sha256",
        "source_v3_state_unchanged",
        "source_basis_tensor_sha256s",
        "basis_manifest_sha256",
        "tensor_count",
        "total_numel",
        "total_nbytes",
        "contains_model_state",
        "contains_parent_state",
        "contains_optimizer_state",
        "contains_gradients",
        "contains_activations",
        "contains_hidden_states",
        "contains_replay_blocks",
        "basis_payload_sha256",
        "guided_right_rows",
    }
    payload = _strict(raw, fields, "basis bundle")
    geometry = _strict(
        payload["geometry"],
        set(EXPECTED_GEOMETRY),
        "basis geometry",
    )
    if (
        payload["schema"] != BASIS_BUNDLE_SCHEMA
        or payload["audit_schema"] != gud_replay.GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA
        or payload["basis_pack_schema"]
        != gud_replay.GUD_COMPLETE_BASIS_PACK_SCHEMA
        or dict(geometry) != EXPECTED_GEOMETRY
        or payload["rank"] != EXPECTED_RANK
        or payload["candidate_order"] != list(BASIS_METHODS)
        or any(
            payload[name] is not False
            for name in (
                "contains_model_state",
                "contains_parent_state",
                "contains_optimizer_state",
                "contains_gradients",
                "contains_activations",
                "contains_hidden_states",
                "contains_replay_blocks",
            )
        )
    ):
        raise GuidedGUDCandidateTournamentError(
            "Basis bundle geometry, method order, or state exclusion changed."
        )
    metadata = {
        name: payload[name]
        for name in fields - {"basis_payload_sha256", "guided_right_rows"}
    }
    payload_sha = _sha256(payload["basis_payload_sha256"], "basis_payload_sha256")
    if canonical_sha256(metadata) != payload_sha:
        raise GuidedGUDCandidateTournamentError("Basis payload SHA changed.")
    replay_report = _strict(
        payload["replay_audit_report"],
        set(payload["replay_audit_report"]),
        "replay audit report",
    )
    audit_sha = _canonical_report_sha256(replay_report, "report_sha256")
    try:
        replay_config = gud_replay.GUDOutputJacobianReplayConfig(
            **dict(replay_report.get("config"))
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise GuidedGUDCandidateTournamentError(
            "Basis replay configuration is malformed."
        ) from error
    if (
        replay_report.get("schema") != gud_replay.GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA
        or replay_report.get("basis_pack_schema")
        != gud_replay.GUD_COMPLETE_BASIS_PACK_SCHEMA
        or replay_report.get("geometry") != dict(geometry)
        or replay_report.get("site_count") != EXPECTED_GEOMETRY["logical_layers"] * 3
        or replay_report.get("separate_projection_hook_count")
        != EXPECTED_GEOMETRY["logical_layers"] * 3
        or replay_report.get("candidate_order") != list(BASIS_METHODS)
        or replay_report.get("two_pass_replay_identical") is not True
        or replay_report.get("gate_up_separate_hooks") is not True
        or replay_report.get("full_downstream_loss_output_delta_captured") is not True
        or replay_report.get("role_local_projection_input_captured") is not True
        or replay_report.get("internal_weight_matching_used") is not False
        or replay_report.get("source_v3_state_unchanged") is not True
        or replay_report.get("source_v3_factor_count")
        != len(checkpoint_recovery._V3_FACTOR_NAMES)
        or isinstance(replay_report.get("source_v3_lexical_rank"), bool)
        or not isinstance(replay_report.get("source_v3_lexical_rank"), int)
        or replay_report.get("source_v3_lexical_rank") <= 0
        or replay_report.get("config") != replay_config.to_dict()
        or replay_report.get("config_sha256") != replay_config.sha256()
        or replay_config.prompt_count != 24
        or replay_config.horizon != 8
        or replay_config.rank != EXPECTED_RANK
    ):
        raise GuidedGUDCandidateTournamentError(
            "Basis bundle is not the complete Qwen GUD replay audit."
        )
    runtime_before = _sha256(
        payload["audit_source_v3_runtime_factor_state_sha256_before"],
        "basis audit source v3 runtime SHA before",
    )
    runtime_after = _sha256(
        payload["audit_source_v3_runtime_factor_state_sha256_after"],
        "basis audit source v3 runtime SHA after",
    )
    portable_sha = _sha256(
        payload["source_v3_portable_factor_state_sha256"],
        "basis source v3 portable SHA",
    )
    factor_manifest_sha = _sha256(
        payload["source_v3_factor_manifest_sha256"],
        "basis source v3 factor manifest SHA",
    )
    binding_sha = _sha256(
        payload["source_guided_qkvo_binding_sha256"],
        "basis source guided QKVO binding SHA",
    )
    try:
        source_binding = checkpoint_recovery._source_binding(
            payload["source_guided_qkvo_binding"],
            "basis source guided QKVO binding",
        )
    except Exception as error:
        raise GuidedGUDCandidateTournamentError(
            "Basis source guided QKVO binding is malformed."
        ) from error
    source_tensor_hashes = _strict(
        payload["source_v3_tensor_sha256s"],
        set(payload["source_v3_tensor_sha256s"]),
        "source v3 tensor hashes",
    )
    if (
        set(source_tensor_hashes) != checkpoint_recovery._V3_FACTOR_NAMES
        or any(
            _sha256(value, f"source v3 tensor {name}") != value
            for name, value in source_tensor_hashes.items()
        )
        or runtime_before != runtime_after
        or source_binding.binding_sha256 != binding_sha
        or payload["source_v3_state_unchanged"] is not True
        or replay_report.get("source_v3_runtime_factor_state_sha256_before")
        != runtime_before
        or replay_report.get("source_v3_runtime_factor_state_sha256_after")
        != runtime_after
        or replay_report.get("source_v3_portable_factor_state_sha256")
        != portable_sha
        or replay_report.get("source_v3_factor_manifest_sha256")
        != factor_manifest_sha
        or replay_report.get("source_v3_tensor_sha256s") != dict(source_tensor_hashes)
        or replay_report.get("source_guided_qkvo_binding")
        != source_binding.as_dict()
        or replay_report.get("source_guided_qkvo_binding_sha256") != binding_sha
    ):
        raise GuidedGUDCandidateTournamentError(
            "Basis replay source provenance changed."
        )
    rows = _strict(payload["guided_right_rows"], set(BASIS_METHODS), "guided rows")
    hashes = _strict(
        payload["source_basis_tensor_sha256s"],
        set(BASIS_METHODS),
        "basis tensor hashes",
    )
    expected_widths = {
        "gate": EXPECTED_GEOMETRY["hidden_size"],
        "up": EXPECTED_GEOMETRY["hidden_size"],
        "down": EXPECTED_GEOMETRY["intermediate_size"],
    }
    tensor_count = 0
    total_numel = 0
    total_nbytes = 0
    authenticated_rows: dict[str, dict[str, Tensor]] = {}
    authenticated_hashes: dict[str, dict[str, str]] = {}
    for method in BASIS_METHODS:
        method_rows = _strict(rows[method], set(GUD_ROLES), f"rows.{method}")
        method_hashes = _strict(hashes[method], set(GUD_ROLES), f"hashes.{method}")
        authenticated_rows[method] = {}
        authenticated_hashes[method] = {}
        for role in GUD_ROLES:
            value = method_rows[role]
            expected_shape = (
                EXPECTED_GEOMETRY["logical_layers"],
                EXPECTED_RANK,
                expected_widths[role],
            )
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or not value.is_contiguous()
                or tuple(value.shape) != expected_shape
                or not bool(torch.isfinite(value).all())
            ):
                raise GuidedGUDCandidateTournamentError(
                    f"{method}/{role} is not finite contiguous CPU FP32 {expected_shape}."
                )
            tensor_sha = gud_adapter._tensor_sha256(value)
            if tensor_sha != _sha256(method_hashes[role], f"{method}.{role}.sha256"):
                raise GuidedGUDCandidateTournamentError(
                    f"{method}/{role} failed tensor authentication."
                )
            authenticated_rows[method][role] = value
            authenticated_hashes[method][role] = tensor_sha
            tensor_count += 1
            total_numel += int(value.numel())
            total_nbytes += int(value.numel() * value.element_size())
    if (
        payload["tensor_count"] != tensor_count
        or payload["total_numel"] != total_numel
        or payload["total_nbytes"] != total_nbytes
    ):
        raise GuidedGUDCandidateTournamentError("Basis tensor accounting changed.")
    manifest = [
        {
            "candidate": method,
            "role": role,
            "shape": list(authenticated_rows[method][role].shape),
            "dtype": str(authenticated_rows[method][role].dtype),
            "sha256": authenticated_hashes[method][role],
        }
        for method in BASIS_METHODS
        for role in GUD_ROLES
    ]
    manifest_sha = canonical_sha256(manifest)
    if (
        replay_report.get("basis_pack_manifest") != manifest
        or replay_report.get("basis_pack_manifest_sha256") != manifest_sha
        or payload["basis_manifest_sha256"] != manifest_sha
    ):
        raise GuidedGUDCandidateTournamentError("Basis pack manifest changed.")
    return AuthenticatedGuidedGUDBasisBundle(
        path=bundle_path,
        file_sha256=digest,
        payload_sha256=payload_sha,
        audit_report_sha256=audit_sha,
        replay_audit_report=dict(replay_report),
        basis_manifest_sha256=str(payload["basis_manifest_sha256"]),
        audit_source_v3_runtime_factor_state_sha256_before=runtime_before,
        audit_source_v3_runtime_factor_state_sha256_after=runtime_after,
        source_v3_portable_factor_state_sha256=portable_sha,
        source_v3_factor_manifest_sha256=factor_manifest_sha,
        source_v3_tensor_sha256s=dict(source_tensor_hashes),
        source_guided_qkvo_binding=source_binding.as_dict(),
        source_guided_qkvo_binding_sha256=binding_sha,
        source_v3_state_unchanged=True,
        geometry=dict(geometry),
        rank=EXPECTED_RANK,
        rows=authenticated_rows,
        source_basis_tensor_sha256s=authenticated_hashes,
    )


def guided_gud_layer_mask_from_replay_report(
    report: Mapping[str, Any],
) -> tuple[GuidedGUDLayerMask, dict[str, Any]]:
    """Derive a deterministic half-site mask from fit-only replay diagnostics."""

    if not isinstance(report, Mapping):
        raise GuidedGUDCandidateTournamentError(
            "Mask derivation requires a replay report."
        )
    audit_sha = _canonical_report_sha256(report, "report_sha256")
    diagnostics = report.get("site_diagnostics")
    diagnostics_sha = _sha256(
        report.get("site_diagnostics_sha256"),
        "site_diagnostics_sha256",
    )
    if canonical_sha256(diagnostics) != diagnostics_sha:
        raise GuidedGUDCandidateTournamentError(
            "Replay site diagnostics failed report authentication."
        )
    layers = EXPECTED_GEOMETRY["logical_layers"]
    if not isinstance(diagnostics, list) or len(diagnostics) != layers * 3:
        raise GuidedGUDCandidateTournamentError(
            "Replay diagnostics cannot derive the complete GUD mask."
        )
    by_site: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in diagnostics:
        if not isinstance(row, Mapping):
            raise GuidedGUDCandidateTournamentError(
                "Replay site diagnostic is malformed."
            )
        layer = row.get("layer")
        role = row.get("role")
        key = (layer, role)
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or layer >= layers
            or role not in GUD_ROLES
            or key in by_site
        ):
            raise GuidedGUDCandidateTournamentError(
                "Replay site diagnostics changed topology or order."
            )
        for name in (
            "guided_capture_fraction",
            "guided_over_random_ratio",
            "normal_fraction",
        ):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not torch.isfinite(torch.tensor(float(value)))
            ):
                raise GuidedGUDCandidateTournamentError(
                    f"Replay diagnostic {name} is not finite."
                )
        by_site[key] = row
    expected = {(layer, role) for layer in range(layers) for role in GUD_ROLES}
    if set(by_site) != expected:
        raise GuidedGUDCandidateTournamentError(
            "Replay site diagnostics do not cover every GUD projection."
        )

    def score(rows: Sequence[Mapping[str, Any]], layer: int) -> tuple[float, ...]:
        return (
            min(float(row["guided_capture_fraction"]) for row in rows),
            min(float(row["guided_over_random_ratio"]) for row in rows),
            min(float(row["normal_fraction"]) for row in rows),
            -float(layer),
        )

    gate_up_scores = {
        layer: score(
            [by_site[(layer, "gate")], by_site[(layer, "up")]], layer
        )
        for layer in range(layers)
    }
    down_scores = {
        layer: score([by_site[(layer, "down")]], layer) for layer in range(layers)
    }
    selected_count = layers // 2
    if selected_count <= 0 or selected_count >= layers:
        raise GuidedGUDCandidateTournamentError(
            "Mask derivation requires at least two logical layers."
        )
    selected_gate_up = set(
        sorted(gate_up_scores, key=gate_up_scores.get, reverse=True)[:selected_count]
    )
    selected_down = set(
        sorted(down_scores, key=down_scores.get, reverse=True)[:selected_count]
    )
    derivation = {
        "schema": SITE_MASK_DERIVATION_SCHEMA,
        "source_audit_report_sha256": audit_sha,
        "source_site_diagnostics_sha256": diagnostics_sha,
        "selection_policy": MASK_SELECTION_POLICY,
        "gate_up_coupled": True,
        "selected_layer_count_per_branch": selected_count,
        "gate_up_selected_layers": sorted(selected_gate_up),
        "down_selected_layers": sorted(selected_down),
        "gate_up_scores": {
            str(layer): list(gate_up_scores[layer]) for layer in range(layers)
        },
        "down_scores": {
            str(layer): list(down_scores[layer]) for layer in range(layers)
        },
    }
    derivation["derivation_sha256"] = canonical_sha256(derivation)
    mask_payload = {
        "schema": SITE_MASK_SCHEMA,
        "logical_layers": layers,
        "gate_up_layer_mask": [layer in selected_gate_up for layer in range(layers)],
        "down_layer_mask": [layer in selected_down for layer in range(layers)],
        "gate_up_coupled": True,
        "gate_up_selected_layer_count": selected_count,
        "down_selected_layer_count": selected_count,
        "source_audit_report_sha256": audit_sha,
        "source_site_diagnostics_sha256": diagnostics_sha,
        "derivation_report": derivation,
        "derivation_sha256": derivation["derivation_sha256"],
    }
    mask_payload["mask_contract_sha256"] = canonical_sha256(mask_payload)
    mask = GuidedGUDLayerMask.from_dict(mask_payload)
    return mask, derivation


def guided_gud_layer_mask_from_replay(
    result: gud_replay.GUDOutputJacobianReplayResult,
) -> tuple[GuidedGUDLayerMask, dict[str, Any]]:
    if not isinstance(result, gud_replay.GUDOutputJacobianReplayResult):
        raise GuidedGUDCandidateTournamentError(
            "Mask derivation requires GUDOutputJacobianReplayResult."
        )
    return guided_gud_layer_mask_from_replay_report(result.report.to_dict())


@dataclass(frozen=True, slots=True)
class GuidedGUDLayerMask:
    logical_layers: int
    gate_up_layer_mask: tuple[bool, ...]
    down_layer_mask: tuple[bool, ...]
    source_audit_report_sha256: str
    source_site_diagnostics_sha256: str
    derivation_report: Mapping[str, Any]
    derivation_sha256: str
    mask_contract_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GuidedGUDLayerMask:
        row = _strict(
            value,
            {
                "schema",
                "logical_layers",
                "gate_up_layer_mask",
                "down_layer_mask",
                "gate_up_coupled",
                "gate_up_selected_layer_count",
                "down_selected_layer_count",
                "source_audit_report_sha256",
                "source_site_diagnostics_sha256",
                "derivation_report",
                "derivation_sha256",
                "mask_contract_sha256",
            },
            "guided GUD site mask",
        )
        payload = dict(row)
        embedded = payload.pop("mask_contract_sha256", None)
        mask_sha = _sha256(embedded, "mask_contract_sha256")
        layers = row["logical_layers"]
        gate_up = row["gate_up_layer_mask"]
        down = row["down_layer_mask"]
        derivation = row["derivation_report"]
        expected_derivation_fields = {
            "schema",
            "source_audit_report_sha256",
            "source_site_diagnostics_sha256",
            "selection_policy",
            "gate_up_coupled",
            "selected_layer_count_per_branch",
            "gate_up_selected_layers",
            "down_selected_layers",
            "gate_up_scores",
            "down_scores",
            "derivation_sha256",
        }
        if not isinstance(derivation, Mapping) or set(derivation) != (
            expected_derivation_fields
        ):
            raise GuidedGUDCandidateTournamentError(
                "Guided GUD mask derivation report is malformed."
            )
        derivation_payload = dict(derivation)
        embedded_derivation_sha = derivation_payload.pop("derivation_sha256", None)
        derivation_sha = _sha256(
            embedded_derivation_sha,
            "mask derivation SHA",
        )
        audit_sha = _sha256(
            row["source_audit_report_sha256"],
            "site_mask.source_audit_report_sha256",
        )
        diagnostics_sha = _sha256(
            row["source_site_diagnostics_sha256"],
            "site_mask.source_site_diagnostics_sha256",
        )
        if (
            row["schema"] != SITE_MASK_SCHEMA
            or isinstance(layers, bool)
            or not isinstance(layers, int)
            or layers != EXPECTED_GEOMETRY["logical_layers"]
            or not isinstance(gate_up, list)
            or not isinstance(down, list)
            or len(gate_up) != layers
            or len(down) != layers
            or any(not isinstance(value, bool) for value in (*gate_up, *down))
            or not any(gate_up)
            or all(gate_up)
            or not any(down)
            or all(down)
            or row["gate_up_coupled"] is not True
            or row["gate_up_selected_layer_count"] != sum(gate_up)
            or row["down_selected_layer_count"] != sum(down)
            or sum(gate_up) != layers // 2
            or sum(down) != layers // 2
            or row["derivation_sha256"] != derivation_sha
            or canonical_sha256(derivation_payload) != derivation_sha
            or derivation.get("schema") != SITE_MASK_DERIVATION_SCHEMA
            or derivation.get("source_audit_report_sha256") != audit_sha
            or derivation.get("source_site_diagnostics_sha256") != diagnostics_sha
            or derivation.get("selection_policy") != MASK_SELECTION_POLICY
            or derivation.get("gate_up_coupled") is not True
            or derivation.get("selected_layer_count_per_branch") != layers // 2
            or derivation.get("gate_up_selected_layers")
            != [index for index, selected in enumerate(gate_up) if selected]
            or derivation.get("down_selected_layers")
            != [index for index, selected in enumerate(down) if selected]
            or canonical_sha256(payload) != mask_sha
        ):
            raise GuidedGUDCandidateTournamentError(
                "Guided GUD mask is not an authenticated coupled G/U mask."
            )
        return cls(
            logical_layers=layers,
            gate_up_layer_mask=tuple(gate_up),
            down_layer_mask=tuple(down),
            source_audit_report_sha256=audit_sha,
            source_site_diagnostics_sha256=diagnostics_sha,
            derivation_report=dict(derivation),
            derivation_sha256=derivation_sha,
            mask_contract_sha256=mask_sha,
        )

    def selected_for_role(self, role: str) -> tuple[bool, ...]:
        if role in {"gate", "up"}:
            return self.gate_up_layer_mask
        if role == "down":
            return self.down_layer_mask
        raise GuidedGUDCandidateTournamentError("Unknown GUD mask role.")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SITE_MASK_SCHEMA,
            "logical_layers": self.logical_layers,
            "gate_up_layer_mask": list(self.gate_up_layer_mask),
            "down_layer_mask": list(self.down_layer_mask),
            "gate_up_coupled": True,
            "gate_up_selected_layer_count": sum(self.gate_up_layer_mask),
            "down_selected_layer_count": sum(self.down_layer_mask),
            "source_audit_report_sha256": self.source_audit_report_sha256,
            "source_site_diagnostics_sha256": (
                self.source_site_diagnostics_sha256
            ),
            "derivation_report": dict(self.derivation_report),
            "derivation_sha256": self.derivation_sha256,
        }
        payload["mask_contract_sha256"] = canonical_sha256(payload)
        return payload


def authenticate_guided_gud_layer_mask(
    path: Path | str,
    expected_sha256: str,
    *,
    expected_audit_report_sha256: str,
) -> tuple[GuidedGUDLayerMask, Path, str]:
    mask_path = Path(path).expanduser().resolve()
    digest = _sha256(expected_sha256, "site_mask.sha256")
    if (
        mask_path.is_symlink()
        or not mask_path.is_file()
        or stage._file_sha256(mask_path) != digest
    ):
        raise GuidedGUDCandidateTournamentError("Guided GUD mask is absent or changed.")
    raw = stage._strict_json(mask_path)
    if not isinstance(raw, Mapping):
        raise GuidedGUDCandidateTournamentError("Guided GUD mask is not a mapping.")
    mask = GuidedGUDLayerMask.from_dict(raw)
    if mask.source_audit_report_sha256 != _sha256(
        expected_audit_report_sha256, "expected audit report SHA"
    ):
        raise GuidedGUDCandidateTournamentError(
            "Guided GUD mask is bound to another basis audit."
        )
    return mask, mask_path, digest


def authenticate_guided_gud_mask_derivation(
    mask: GuidedGUDLayerMask,
    basis: AuthenticatedGuidedGUDBasisBundle,
) -> dict[str, Any]:
    """Re-derive the preregistered mask from authenticated fit-only evidence."""

    expected, derivation = guided_gud_layer_mask_from_replay_report(
        basis.replay_audit_report
    )
    if (
        mask.as_dict() != expected.as_dict()
        or mask.derivation_sha256 != derivation["derivation_sha256"]
        or sum(mask.gate_up_layer_mask)
        != EXPECTED_GEOMETRY["logical_layers"] // 2
        or sum(mask.down_layer_mask) != EXPECTED_GEOMETRY["logical_layers"] // 2
    ):
        raise GuidedGUDCandidateTournamentError(
            "Guided GUD mask differs from its deterministic replay derivation."
        )
    return derivation


class GuidedGUDGradientMaskGuard:
    """Mask unselected G/U/D layer rows and verify them after every step."""

    def __init__(self, registry: nn.Module, mask: GuidedGUDLayerMask) -> None:
        if not isinstance(
            registry, gud_adapter.GuidedSparseDenseSwiGLUProjectionResidualRegistry
        ):
            raise GuidedGUDCandidateTournamentError(
                "GUD mask requires the authenticated guided registry."
            )
        if not isinstance(mask, GuidedGUDLayerMask):
            raise GuidedGUDCandidateTournamentError(
                "GUD mask requires an authenticated layer mask."
            )
        self.registry = registry
        self.mask = mask
        self._handles: list[RemovableHandle] = []
        self._snapshots: dict[
            str, tuple[Tensor, str, tuple[int, int, tuple[int, ...]]]
        ] = {}
        self._unselected: dict[str, tuple[int, ...]] = {}
        self._entered = False

    def _hook(self, parameter: nn.Parameter, rows: tuple[int, ...]) -> RemovableHandle:
        index = torch.tensor(rows, dtype=torch.long, device=parameter.device)

        def mask_gradient(gradient: Tensor) -> Tensor:
            if not isinstance(gradient, Tensor) or tuple(gradient.shape) != tuple(
                parameter.shape
            ):
                raise GuidedGUDCandidateTournamentError(
                    "Guided GUD gradient shape changed under mask."
                )
            result = gradient.clone(memory_format=torch.preserve_format)
            result.index_fill_(0, index, 0)
            return result

        requires_grad = bool(parameter.requires_grad)
        if not requires_grad:
            parameter.requires_grad_(True)
        try:
            return parameter.register_hook(mask_gradient)
        finally:
            parameter.requires_grad_(requires_grad)

    def __enter__(self) -> GuidedGUDGradientMaskGuard:
        if self._entered:
            raise GuidedGUDCandidateTournamentError("GUD mask guard is one-shot.")
        self.registry.assert_finite()
        if (
            self.registry.site_count != self.mask.logical_layers
            or self.registry.rank != EXPECTED_RANK
            or self.registry.audit_report_sha256 != self.mask.source_audit_report_sha256
        ):
            raise GuidedGUDCandidateTournamentError(
                "GUD registry topology or audit differs from its mask."
            )
        try:
            for role in GUD_ROLES:
                selected = self.mask.selected_for_role(role)
                unselected = tuple(
                    index for index, enabled in enumerate(selected) if not enabled
                )
                self._unselected[role] = unselected
                cpu_index = torch.tensor(unselected, dtype=torch.long)
                for side in ("left", "right"):
                    name = f"{role}_{side}"
                    parameter = getattr(self.registry, name)
                    if (
                        not isinstance(parameter, nn.Parameter)
                        or int(parameter.shape[0]) != self.mask.logical_layers
                    ):
                        raise GuidedGUDCandidateTournamentError(
                            "Guided GUD factor topology changed before masking."
                        )
                    snapshot = (
                        parameter.detach().cpu().index_select(0, cpu_index).contiguous()
                    )
                    self._snapshots[name] = (
                        snapshot,
                        gud_adapter._tensor_sha256(snapshot),
                        (
                            id(parameter),
                            int(parameter.data_ptr()),
                            tuple(parameter.shape),
                        ),
                    )
                    self._handles.append(self._hook(parameter, unselected))
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        self._entered = True
        return self

    def verify_step(self) -> None:
        if not self._entered or len(self._handles) != len(GUD_ROLES) * 2:
            raise GuidedGUDCandidateTournamentError("GUD mask guard is not active.")
        for role in GUD_ROLES:
            index = torch.tensor(self._unselected[role], dtype=torch.long)
            for side in ("left", "right"):
                name = f"{role}_{side}"
                parameter = getattr(self.registry, name)
                snapshot, digest, signature = self._snapshots[name]
                if (
                    id(parameter),
                    int(parameter.data_ptr()),
                    tuple(parameter.shape),
                ) != signature:
                    raise GuidedGUDCandidateTournamentError(
                        "A masked GUD factor changed ownership."
                    )
                observed = parameter.detach().cpu().index_select(0, index).contiguous()
                if gud_adapter._tensor_sha256(observed) != digest or not torch.equal(
                    observed, snapshot
                ):
                    raise GuidedGUDCandidateTournamentError(
                        "An unselected GUD row changed after an optimizer step."
                    )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._entered = False


@dataclass(frozen=True, slots=True)
class AuthenticatedV3WinnerSource:
    tournament_contract: v3_tournament.AuthenticatedCandidateTournament
    tournament_preregistration_path: Path
    tournament_preregistration_sha256: str
    run_report_path: Path
    run_report_sha256: str
    winner_selection_path: Path
    winner_selection_sha256: str
    arm_id: str
    step: int
    runtime_factor_state_sha256: str
    portable_factor_state_sha256: str
    factor_pack_path: Path
    factor_pack_file_sha256: str
    factor_payload: Mapping[str, Any]
    candidate_frontier: Mapping[str, Any]


def _authenticate_historical_v1_winner_payload(
    contract: v3_tournament.AuthenticatedCandidateTournament,
    *,
    run_path: Path,
    run_sha256: str,
    run_report: Mapping[str, Any],
    winner_path: Path,
    winner_sha256: str,
    winner_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the exact frozen v1 evidence without pretending it is v2."""

    if (
        run_sha256 != EXPECTED_SOURCE_RUN_REPORT_SHA256
        or winner_sha256 != EXPECTED_SOURCE_WINNER_SELECTION_SHA256
        or contract.evaluation_contract_sha256
        != EXPECTED_SOURCE_EVALUATION_CONTRACT_SHA256
    ):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 run, winner, or evaluation identity changed."
        )
    if (
        run_path.name != "run_report.json"
        or winner_path.name != "winner_selection.json"
        or run_path.parent != winner_path.parent
    ):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 run and winner must be exact sibling artifacts."
        )
    evaluation_sha = contract.evaluation_contract_sha256
    if (
        run_report.get("schema") != HISTORICAL_V3_RUN_SCHEMA
        or run_report.get("status") != "candidate-winner-selected"
        or run_report.get("evaluation_contract_sha256") != evaluation_sha
        or run_report.get("sealed_forward_count") != 0
        or run_report.get("sealed_evaluation_accessed") is not False
        or run_report.get("sealed_evaluation_locked") is not True
        or run_report.get("sealed_suite_supplied_to_trainer") is not False
        or run_report.get("selection_uses_internal_numeric_targets") is not False
        or run_report.get("model_checkpoint_written") is not False
        or run_report.get("publication_attempted") is not False
        or run_report.get("publication_authorized") is not False
        or run_report.get("stop_after_candidate_winner") is not True
        or canonical_json_bytes(run_report.get("winner_selection"))
        != canonical_json_bytes(winner_selection)
    ):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 terminal report is not the sealed-free v1 boundary."
        )

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
    selection = _strict(
        winner_selection,
        {
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
        },
        "historical v3 winner selection",
    )
    winner = _strict(selection["winner"], required_winner, "historical v3 winner")
    metrics = winner.get("metrics")
    if (
        selection.get("schema") != HISTORICAL_V3_WINNER_SCHEMA
        or selection.get("evaluation_contract_sha256") != evaluation_sha
        or selection.get("candidate_only") is not True
        or selection.get("sealed_evaluation_accessed") is not False
        or selection.get("model_checkpoint_written") is not False
        or selection.get("publication_attempted") is not False
        or selection.get("stop_reason")
        != "candidate-winner-selected-before-sealed"
        or winner.get("arm_id") != EXPECTED_SOURCE_ARM_ID
        or winner.get("arm_index") != 0
        or winner.get("step") != EXPECTED_SOURCE_STEP
        or winner.get("runtime_factor_state_sha256")
        != EXPECTED_SOURCE_RUNTIME_V3_SHA256
        or winner.get("portable_factor_state_sha256")
        != EXPECTED_SOURCE_PORTABLE_V3_SHA256
        or winner.get("factor_pack_sha256")
        != EXPECTED_SOURCE_FACTOR_PACK_FILE_SHA256
        or not isinstance(metrics, Mapping)
        or winner.get("rank") != list(v3_tournament.tournament_behavior_rank(metrics))
    ):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 winner is malformed, unlocked, or relabeled."
        )
    for field in (
        "runtime_factor_state_sha256",
        "portable_factor_state_sha256",
        "checkpoint_sha256",
        "factor_pack_sha256",
    ):
        _sha256(winner[field], f"historical v3 winner.{field}")

    frontier_path = _resolve(
        run_path.parent,
        "candidate_frontier.json",
        "historical v3 candidate frontier",
    )
    if (
        not frontier_path.is_file()
        or stage._file_sha256(frontier_path)
        != EXPECTED_SOURCE_CANDIDATE_FRONTIER_SHA256
    ):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 candidate frontier is absent or changed."
        )
    frontier = _strict(
        stage._strict_json(frontier_path),
        {
            "schema",
            "evaluation_contract_sha256",
            "completed_arms",
            "retained_count",
            "retained",
            "distinct_full_v3_states_only",
            "portable_content_identity_used",
            "step_zero_ranked",
            "later_states_require_pareto_safety_vs_step_zero",
            "sealed_metrics_used",
            "internal_numeric_matching_used",
        },
        "historical v3 candidate frontier",
    )
    retained = frontier.get("retained")
    if (
        frontier.get("schema") != HISTORICAL_V3_FRONTIER_SCHEMA
        or frontier.get("evaluation_contract_sha256") != evaluation_sha
        or frontier.get("completed_arms") != list(v3_tournament.ARM_IDS)
        or frontier.get("retained_count") != v3_tournament.MAX_RETAINED_STATES
        or not isinstance(retained, list)
        or len(retained) != v3_tournament.MAX_RETAINED_STATES
        or frontier.get("distinct_full_v3_states_only") is not True
        or frontier.get("portable_content_identity_used") is not True
        or frontier.get("step_zero_ranked") is not True
        or frontier.get("later_states_require_pareto_safety_vs_step_zero") is not True
        or frontier.get("sealed_metrics_used") is not False
        or frontier.get("internal_numeric_matching_used") is not False
    ):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 candidate frontier is incomplete or unlocked."
        )

    retained_rows: list[Mapping[str, Any]] = []
    portable_identities: set[str] = set()
    for index, raw in enumerate(retained):
        row = _strict(raw, required_winner, f"historical v3 retained[{index}]")
        row_metrics = row.get("metrics")
        arm_index = row.get("arm_index")
        portable = _sha256(
            row.get("portable_factor_state_sha256"),
            f"historical v3 retained[{index}].portable_factor_state_sha256",
        )
        if (
            not isinstance(arm_index, int)
            or isinstance(arm_index, bool)
            or not 0 <= arm_index < len(v3_tournament.ARM_IDS)
            or row.get("arm_id") != v3_tournament.ARM_IDS[arm_index]
            or row.get("step") not in v3_tournament.CHECKPOINT_STEPS
            or not isinstance(row_metrics, Mapping)
            or row.get("rank")
            != list(v3_tournament.tournament_behavior_rank(row_metrics))
            or portable in portable_identities
        ):
            raise GuidedGUDCandidateTournamentError(
                "Historical v3 retained frontier row is inconsistent."
            )
        portable_identities.add(portable)
        for field in (
            "runtime_factor_state_sha256",
            "checkpoint_sha256",
            "factor_pack_sha256",
        ):
            _sha256(row.get(field), f"historical v3 retained[{index}].{field}")
        pack_path = _resolve(
            run_path.parent,
            row.get("factor_pack"),
            f"historical v3 retained[{index}] factor pack",
        )
        if (
            not pack_path.is_file()
            or stage._file_sha256(pack_path) != row.get("factor_pack_sha256")
        ):
            raise GuidedGUDCandidateTournamentError(
                "A historical v3 retained factor pack is absent or changed."
            )
        record_path = _resolve(
            pack_path.parent,
            "candidate_record.json",
            f"historical v3 retained[{index}] candidate record",
        )
        if (
            not record_path.is_file()
            or canonical_json_bytes(stage._strict_json(record_path))
            != canonical_json_bytes(row)
        ):
            raise GuidedGUDCandidateTournamentError(
                "A historical v3 retained candidate record is absent or changed."
            )
        retained_rows.append(row)

    runtime_ids = [row["runtime_factor_state_sha256"] for row in retained_rows]
    portable_ids = [row["portable_factor_state_sha256"] for row in retained_rows]
    if (
        canonical_json_bytes(retained_rows[0]) != canonical_json_bytes(winner)
        or selection.get("top_three_factor_state_sha256s") != runtime_ids
        or selection.get("top_three_portable_factor_state_sha256s") != portable_ids
        or run_report.get("top_three_factor_state_sha256s") != runtime_ids
        or run_report.get("top_three_portable_factor_state_sha256s") != portable_ids
    ):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 winner, frontier, and terminal top three disagree."
        )

    factor_path = _resolve(
        run_path.parent,
        winner["factor_pack"],
        "historical v3 winner factor pack",
    )
    try:
        loaded = torch.load(factor_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 winner factor pack cannot load safely."
        ) from error
    if not isinstance(loaded, Mapping):
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 winner factor pack is not a mapping."
        )
    try:
        GuidedCheckpointRecoverySink._validate_v3_factor_payload(
            loaded,
            expected_factor_state_sha256=EXPECTED_SOURCE_RUNTIME_V3_SHA256,
        )
    except Exception as error:
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 winner is not a complete authenticated 26-factor state."
        ) from error
    if portable_v3_factor_state_sha256(loaded) != EXPECTED_SOURCE_PORTABLE_V3_SHA256:
        raise GuidedGUDCandidateTournamentError(
            "Historical v3 winner portable factor identity changed."
        )
    frontier_binding = {
        "schema": HISTORICAL_V3_FRONTIER_SCHEMA,
        "path": str(frontier_path),
        "sha256": EXPECTED_SOURCE_CANDIDATE_FRONTIER_SHA256,
        "retained_count": v3_tournament.MAX_RETAINED_STATES,
        "completed_arms": list(v3_tournament.ARM_IDS),
        "callback_preimages_persisted": False,
        "historical_source_commit": v3_tournament.HISTORICAL_V3_SOURCE_COMMIT,
        "authentication_mode": "exact-artifact-sha-plus-historical-v1-structure",
        "publication_authorized": False,
    }
    return (
        {
            "arm_id": str(winner["arm_id"]),
            "arm_index": int(winner["arm_index"]),
            "step": int(winner["step"]),
            "rank": list(winner["rank"]),
            "runtime_factor_state_sha256": EXPECTED_SOURCE_RUNTIME_V3_SHA256,
            "portable_factor_state_sha256": EXPECTED_SOURCE_PORTABLE_V3_SHA256,
            "checkpoint_sha256": str(winner["checkpoint_sha256"]),
            "metrics": dict(metrics),
            "metrics_sha256": canonical_sha256(metrics),
            "factor_pack": str(factor_path),
            "factor_pack_sha256": EXPECTED_SOURCE_FACTOR_PACK_FILE_SHA256,
            "candidate_frontier": frontier_binding,
        },
        dict(loaded),
    )


def authenticate_v3_winner_source(
    value: Mapping[str, Any],
    *,
    base: Path,
) -> AuthenticatedV3WinnerSource:
    row = _strict(
        value,
        {
            "tournament_preregistration",
            "tournament_run_report",
            "winner_selection",
            "expected_winner",
        },
        "source_v3_winner",
    )
    prereg_path, prereg_sha = _file_binding(
        row["tournament_preregistration"],
        base=base,
        name="source tournament preregistration",
    )
    run_path, run_sha = _file_binding(
        row["tournament_run_report"],
        base=base,
        name="source tournament run report",
    )
    winner_path, winner_sha = _file_binding(
        row["winner_selection"],
        base=base,
        name="source winner selection",
    )
    expected = _strict(
        row["expected_winner"],
        {
            "arm_id",
            "step",
            "runtime_factor_state_sha256",
            "portable_factor_state_sha256",
            "factor_pack_file_sha256",
        },
        "source expected winner",
    )
    try:
        contract = v3_tournament.authenticate_candidate_tournament_preregistration(
            prereg_path,
            prereg_sha,
            historical_git_revision=v3_tournament.HISTORICAL_V3_SOURCE_COMMIT,
        )
        run_report = stage._strict_json(run_path)
        winner_selection = stage._strict_json(winner_path)
        if not isinstance(run_report, Mapping) or not isinstance(
            winner_selection, Mapping
        ):
            raise GuidedGUDCandidateTournamentError(
                "Source tournament reports are not mappings."
            )
        winner, payload = _authenticate_historical_v1_winner_payload(
            contract,
            run_path=run_path,
            run_sha256=run_sha,
            run_report=run_report,
            winner_path=winner_path,
            winner_sha256=winner_sha,
            winner_selection=winner_selection,
        )
    except Exception as error:
        if isinstance(error, GuidedGUDCandidateTournamentError):
            raise
        raise GuidedGUDCandidateTournamentError(
            "Completed v3 winner/frontier authentication failed."
        ) from error
    expected_runtime = _sha256(
        expected["runtime_factor_state_sha256"],
        "source expected runtime factor SHA",
    )
    expected_portable = _sha256(
        expected["portable_factor_state_sha256"],
        "source expected portable factor SHA",
    )
    expected_pack = _sha256(
        expected["factor_pack_file_sha256"],
        "source expected factor pack SHA",
    )
    if (
        prereg_sha != EXPECTED_SOURCE_TOURNAMENT_PREREGISTRATION_SHA256
        or expected_pack != EXPECTED_SOURCE_FACTOR_PACK_FILE_SHA256
        or expected["arm_id"] != EXPECTED_SOURCE_ARM_ID
        or expected["step"] != EXPECTED_SOURCE_STEP
        or expected_portable != EXPECTED_SOURCE_PORTABLE_V3_SHA256
        or winner["arm_id"] != expected["arm_id"]
        or winner["step"] != expected["step"]
        or winner["runtime_factor_state_sha256"] != expected_runtime
        or winner["portable_factor_state_sha256"] != expected_portable
        or winner["factor_pack_sha256"] != expected_pack
        or portable_v3_factor_state_sha256(payload) != expected_portable
    ):
        raise GuidedGUDCandidateTournamentError(
            "Source is not the authenticated best retained v3 step40/step12 state."
        )
    return AuthenticatedV3WinnerSource(
        tournament_contract=contract,
        tournament_preregistration_path=prereg_path,
        tournament_preregistration_sha256=prereg_sha,
        run_report_path=run_path,
        run_report_sha256=run_sha,
        winner_selection_path=winner_path,
        winner_selection_sha256=winner_sha,
        arm_id=str(winner["arm_id"]),
        step=int(winner["step"]),
        runtime_factor_state_sha256=expected_runtime,
        portable_factor_state_sha256=expected_portable,
        factor_pack_path=Path(winner["factor_pack"]),
        factor_pack_file_sha256=expected_pack,
        factor_payload=payload,
        candidate_frontier=dict(winner["candidate_frontier"]),
    )


def authenticate_guided_gud_basis_source(
    basis: AuthenticatedGuidedGUDBasisBundle,
    source: AuthenticatedV3WinnerSource,
) -> dict[str, Any]:
    """Bind replay factors, QKVO content, fit-only prompts, and loss schedule."""

    payload = source.factor_payload
    try:
        checkpoint_recovery.GuidedCheckpointRecoverySink._validate_v3_factor_payload(
            payload,
            expected_factor_state_sha256=source.runtime_factor_state_sha256,
        )
    except Exception as error:
        raise GuidedGUDCandidateTournamentError(
            "The authenticated winner is not a complete 26-factor v3 payload."
        ) from error
    rows = payload.get("factors")
    if not isinstance(rows, list):
        raise GuidedGUDCandidateTournamentError("Winner factor manifest is absent.")
    winner_tensor_hashes = {
        row.get("name"): row.get("tensor_sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    winner_manifest_sha = canonical_sha256(rows)
    winner_portable_sha = portable_v3_factor_state_sha256(payload)
    if (
        set(winner_tensor_hashes) != checkpoint_recovery._V3_FACTOR_NAMES
        or winner_tensor_hashes != dict(basis.source_v3_tensor_sha256s)
        or winner_manifest_sha != basis.source_v3_factor_manifest_sha256
        or winner_portable_sha != source.portable_factor_state_sha256
        or winner_portable_sha != basis.source_v3_portable_factor_state_sha256
    ):
        raise GuidedGUDCandidateTournamentError(
            "GUD replay factors differ from the authenticated v3 winner content."
        )
    try:
        historical_binding = checkpoint_recovery._source_binding(
            payload.get("guided_source_binding"),
            "historical winner guided QKVO binding",
        )
        audit_binding = checkpoint_recovery._source_binding(
            basis.source_guided_qkvo_binding,
            "basis audit guided QKVO binding",
        )
    except Exception as error:
        raise GuidedGUDCandidateTournamentError(
            "Guided QKVO source binding failed portable authentication."
        ) from error
    historical_portable_binding = checkpoint_recovery.portable_guided_source_binding(
        historical_binding
    )
    audit_portable_binding = checkpoint_recovery.portable_guided_source_binding(
        audit_binding
    )
    if canonical_json_bytes(historical_portable_binding) != canonical_json_bytes(
        audit_portable_binding
    ):
        raise GuidedGUDCandidateTournamentError(
            "GUD replay is bound to different portable QKVO source content."
        )

    candidate = source.tournament_contract.splits["candidate"]
    sealed = source.tournament_contract.splits["sealed"]
    try:
        split_tools._assert_globally_disjoint(
            {
                "exposed-candidate24-fit-basis": _prompt_records(candidate),
                "sealed32-never-replayed": _prompt_records(sealed),
            },
            label="guided-GUD replay basis versus sealed",
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise GuidedGUDCandidateTournamentError(
            "The replay basis is not exact/prefix-disjoint from sealed32."
        ) from error
    expected_domain = behavior.load_behavior_domain_suite(
        "fit",
        candidate.payload,
        artifact_sha256=candidate.sha256,
    )
    expected_schedule = guided_gud_basis_audit_schedule()
    report = basis.replay_audit_report
    if (
        report.get("basis_domain") != expected_domain.as_dict()
        or report.get("basis_domain_sha256") != expected_domain.domain_sha256
        or report.get("basis_artifact_sha256") != candidate.sha256
        or report.get("behavior_schedule") != expected_schedule.as_dict()
        or report.get("behavior_schedule_sha256")
        != canonical_sha256(expected_schedule.as_dict())
        or report.get("config")
        != guided_gud_basis_audit_config(
            source.tournament_contract.source.execution_contract.seed
        ).to_dict()
        or report.get("config_sha256")
        != guided_gud_basis_audit_config(
            source.tournament_contract.source.execution_contract.seed
        ).sha256()
        or report.get("source_v3_factor_count")
        != len(checkpoint_recovery._V3_FACTOR_NAMES)
        or report.get("source_v3_lexical_rank") != payload.get("lexical_rank")
    ):
        raise GuidedGUDCandidateTournamentError(
            "GUD replay did not use the exact old candidate24 as fit-only data "
            "under the authenticated audit schedule."
        )
    return {
        "source_v3_factor_manifest_sha256": winner_manifest_sha,
        "source_v3_portable_factor_state_sha256": winner_portable_sha,
        "portable_guided_qkvo_binding_sha256": (
            checkpoint_recovery.portable_guided_source_binding_sha256(
                historical_binding
            )
        ),
        "basis_domain_sha256": expected_domain.domain_sha256,
        "basis_artifact_sha256": candidate.sha256,
        "basis_prompt_count": len(expected_domain.prompts),
        "behavior_schedule_sha256": canonical_sha256(expected_schedule.as_dict()),
        "sealed_prompt_count_replayed": 0,
        "basis_exact_and_prefix_disjoint_from_sealed": True,
    }


def _prompt_records(binding: stage.SplitBinding) -> tuple[Any, ...]:
    return v3_tournament._prompt_records(binding)


def _authenticate_v4_split_evolution(
    splits: Mapping[str, stage.SplitBinding],
    source: AuthenticatedV3WinnerSource,
) -> None:
    if {name: value.expected_count for name, value in splits.items()} != SPLIT_COUNTS:
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD splits are not exact 192/24/32."
        )
    old = source.tournament_contract.splits
    if {name: old[name].expected_count for name in old} != {
        "fit": 168,
        "candidate": 24,
        "sealed": 32,
    }:
        raise GuidedGUDCandidateTournamentError(
            "Source tournament is not exact 168/24/32."
        )
    old_records = {name: _prompt_records(old[name]) for name in old}
    new_records = {name: _prompt_records(splits[name]) for name in splits}
    expected_fit = (*old_records["fit"], *old_records["candidate"])
    if [row.payload for row in new_records["fit"]] != [
        row.payload for row in expected_fit
    ]:
        raise GuidedGUDCandidateTournamentError(
            "V4 fit must be old fit followed by exposed candidate, byte for byte."
        )
    if (
        splits["sealed"].path != old["sealed"].path
        or splits["sealed"].sha256 != old["sealed"].sha256
        or [row.payload for row in new_records["sealed"]]
        != [row.payload for row in old_records["sealed"]]
    ):
        raise GuidedGUDCandidateTournamentError(
            "Sealed32 must remain exact and locked."
        )
    try:
        split_tools._assert_globally_disjoint(
            new_records,
            label="guided GUD tournament",
        )
        split_tools._assert_globally_disjoint(
            {
                "historical": tuple(
                    row
                    for name in ("fit", "candidate", "sealed")
                    for row in old_records[name]
                ),
                "new-candidate": new_records["candidate"],
            },
            label="guided GUD historical exclusion",
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise GuidedGUDCandidateTournamentError(
            "Fresh candidate24 is not exact/prefix-disjoint from history."
        ) from error


@dataclass(frozen=True, slots=True)
class GuidedGUDTournamentArm:
    arm_id: str
    arm_index: int
    basis_method: BasisMethod
    mask_mode: MaskMode
    source_v3_schema: str
    source_v3_factor_state_sha256: str
    source_v3_portable_factor_state_sha256: str
    basis_payload_sha256: str
    source_basis_tensor_sha256s: Mapping[str, str]
    site_mask_contract_sha256: str | None

    @property
    def site_mask_enabled(self) -> bool:
        return self.mask_mode == "masked"

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["source_basis_tensor_sha256s"] = dict(self.source_basis_tensor_sha256s)
        payload["site_mask_enabled"] = self.site_mask_enabled
        return payload


def expected_guided_gud_arms(
    source: AuthenticatedV3WinnerSource,
    basis: AuthenticatedGuidedGUDBasisBundle,
    mask: GuidedGUDLayerMask,
) -> tuple[GuidedGUDTournamentArm, ...]:
    values = []
    for index, arm_id in enumerate(ARM_IDS):
        method = BASIS_METHODS[index // len(MASK_MODES)]
        mode = MASK_MODES[index % len(MASK_MODES)]
        values.append(
            GuidedGUDTournamentArm(
                arm_id=arm_id,
                arm_index=index,
                basis_method=method,
                mask_mode=mode,
                source_v3_schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
                source_v3_factor_state_sha256=(source.runtime_factor_state_sha256),
                source_v3_portable_factor_state_sha256=(
                    source.portable_factor_state_sha256
                ),
                basis_payload_sha256=basis.payload_sha256,
                source_basis_tensor_sha256s=dict(
                    basis.source_basis_tensor_sha256s[method]
                ),
                site_mask_contract_sha256=(
                    mask.mask_contract_sha256 if mode == "masked" else None
                ),
            )
        )
    return tuple(values)


@dataclass(frozen=True, slots=True)
class AuthenticatedGuidedGUDCandidateTournament:
    path: Path
    file_sha256: str
    source: AuthenticatedV3WinnerSource
    splits: Mapping[str, stage.SplitBinding]
    basis: AuthenticatedGuidedGUDBasisBundle
    site_mask: GuidedGUDLayerMask
    site_mask_path: Path
    site_mask_file_sha256: str
    arms: tuple[GuidedGUDTournamentArm, ...]
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
            "source_v3": {
                "arm_id": self.source.arm_id,
                "step": self.source.step,
                "runtime_factor_state_sha256": (
                    self.source.runtime_factor_state_sha256
                ),
                "portable_factor_state_sha256": (
                    self.source.portable_factor_state_sha256
                ),
                "factor_pack": str(self.source.factor_pack_path),
                "factor_pack_sha256": self.source.factor_pack_file_sha256,
                "candidate_frontier": dict(self.source.candidate_frontier),
            },
            "splits": {
                name: {
                    "path": str(value.path),
                    "sha256": value.sha256,
                    "expected_count": value.expected_count,
                }
                for name, value in self.splits.items()
            },
            "basis_bundle": {
                "path": str(self.basis.path),
                "sha256": self.basis.file_sha256,
                "payload_sha256": self.basis.payload_sha256,
                "audit_report_sha256": self.basis.audit_report_sha256,
                "basis_manifest_sha256": self.basis.basis_manifest_sha256,
                "methods": list(BASIS_METHODS),
            },
            "site_mask": {
                "path": str(self.site_mask_path),
                "sha256": self.site_mask_file_sha256,
                "contract_sha256": self.site_mask.mask_contract_sha256,
                "gate_up_coupled": True,
            },
            "arms": [arm.as_dict() for arm in self.arms],
            "training": {
                "schedule": self.schedule.as_dict(),
                "stages": guided_gud_stage_contract(),
            },
            "runtime": dict(self.runtime),
            "implementation": dict(self.implementation),
            "models_loaded": False,
            "gpu_work_launched": False,
            "sealed_suite_supplied_to_trainer": False,
            "sealed_evaluation_locked": True,
            "selection_uses_internal_numeric_targets": False,
            "publication_authorized": False,
        }


def _implementation_contract() -> dict[str, str]:
    files = {
        "runner_sha256": Path(__file__).resolve(),
        "builder_sha256": Path(__file__)
        .with_name(
            "prepare_qwen_exact_target_native_guided_gud_candidate_tournament.py"
        )
        .resolve(),
        "v3_tournament_runner_sha256": Path(v3_tournament.__file__).resolve(),
        "v3_winner_authenticator_sha256": Path(v3_publication.__file__).resolve(),
        "gud_audit_sha256": Path(gud_audit.__file__).resolve(),
        "gud_replay_sha256": Path(gud_replay.__file__).resolve(),
        "guided_checkpoint_recovery_sha256": Path(
            checkpoint_recovery.__file__
        ).resolve(),
        "behavior_trainer_sha256": Path(behavior.__file__).resolve(),
        "guided_gud_adapter_sha256": Path(gud_adapter.__file__).resolve(),
    }
    return {name: stage._file_sha256(path) for name, path in files.items()}


def guided_gud_evaluation_preimage(
    *,
    source: AuthenticatedV3WinnerSource,
    splits: Mapping[str, stage.SplitBinding],
    basis: AuthenticatedGuidedGUDBasisBundle,
    mask: GuidedGUDLayerMask,
    mask_file_sha256: str,
    arms: Sequence[GuidedGUDTournamentArm],
    training: Mapping[str, Any],
    selection: Mapping[str, Any],
    runtime: Mapping[str, Any],
    implementation: Mapping[str, str],
) -> dict[str, Any]:
    """Build the single canonical model-free tournament contract preimage."""

    return {
        "source_v3": {
            "tournament_preregistration_sha256": (
                source.tournament_preregistration_sha256
            ),
            "run_report_sha256": source.run_report_sha256,
            "winner_selection_sha256": source.winner_selection_sha256,
            "runtime_factor_state_sha256": source.runtime_factor_state_sha256,
            "portable_factor_state_sha256": source.portable_factor_state_sha256,
            "factor_pack_file_sha256": source.factor_pack_file_sha256,
            "candidate_frontier": dict(source.candidate_frontier),
        },
        "split_sha256s": {name: value.sha256 for name, value in splits.items()},
        "basis_bundle_file_sha256": basis.file_sha256,
        "basis_payload_sha256": basis.payload_sha256,
        "audit_report_sha256": basis.audit_report_sha256,
        "basis_manifest_sha256": basis.basis_manifest_sha256,
        "site_mask_file_sha256": _sha256(
            mask_file_sha256,
            "mask_file_sha256",
        ),
        "site_mask_contract_sha256": mask.mask_contract_sha256,
        "arms": [arm.as_dict() for arm in arms],
        "training": dict(training),
        "selection": dict(selection),
        "runtime": dict(runtime),
        "implementation": dict(implementation),
    }


def authenticate_guided_gud_candidate_preregistration(
    path: Path | str,
    expected_sha256: str,
) -> AuthenticatedGuidedGUDCandidateTournament:
    prereg_path = Path(path).expanduser().resolve()
    expected = _sha256(expected_sha256, "preregistration_sha256")
    if (
        prereg_path.is_symlink()
        or not prereg_path.is_file()
        or stage._file_sha256(prereg_path) != expected
    ):
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD preregistration is absent or changed."
        )
    raw = stage._strict_json(prereg_path)
    prereg = _strict(
        raw,
        {
            "schema",
            "sealed_before_model_access",
            "source_v3_winner",
            "splits",
            "basis_bundle",
            "site_mask",
            "arms",
            "training",
            "selection",
            "runtime",
            "implementation",
            "evaluation_contract_sha256",
        },
        "guided-GUD preregistration",
    )
    if (
        prereg["schema"] != PREREGISTRATION_SCHEMA
        or prereg["sealed_before_model_access"] is not True
    ):
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD preregistration schema/seal changed."
        )
    source = authenticate_v3_winner_source(
        prereg["source_v3_winner"],
        base=prereg_path.parent,
    )
    split_rows = _strict(prereg["splits"], set(SPLIT_COUNTS), "splits")
    try:
        splits = {
            name: stage._load_split(
                split_rows[name],
                name=name,
                base=prereg_path.parent,
            )
            for name in ("fit", "candidate", "sealed")
        }
    except stage.ExactTargetNativeStageError as error:
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD split binding failed."
        ) from error
    _authenticate_v4_split_evolution(splits, source)
    basis_path, basis_sha = _file_binding(
        prereg["basis_bundle"],
        base=prereg_path.parent,
        name="basis bundle",
    )
    basis = authenticate_guided_gud_basis_bundle(basis_path, basis_sha)
    if (
        basis.source_v3_portable_factor_state_sha256
        != source.portable_factor_state_sha256
    ):
        raise GuidedGUDCandidateTournamentError(
            "GUD basis audit is bound to another recovered v3 state."
        )
    authenticate_guided_gud_basis_source(basis, source)
    mask_path, mask_sha = _file_binding(
        prereg["site_mask"],
        base=prereg_path.parent,
        name="guided GUD site mask",
    )
    mask, _, _ = authenticate_guided_gud_layer_mask(
        mask_path,
        mask_sha,
        expected_audit_report_sha256=basis.audit_report_sha256,
    )
    authenticate_guided_gud_mask_derivation(mask, basis)
    arms = expected_guided_gud_arms(source, basis, mask)
    if prereg["arms"] != [arm.as_dict() for arm in arms]:
        raise GuidedGUDCandidateTournamentError(
            "Eight-arm basis/mask order or source changed."
        )
    schedule = guided_gud_candidate_schedule()
    expected_training = {
        "schedule": schedule.as_dict(),
        "stages": guided_gud_stage_contract(),
    }
    if canonical_json_bytes(prereg["training"]) != canonical_json_bytes(
        expected_training
    ):
        raise GuidedGUDCandidateTournamentError("Guided-GUD stage schedule changed.")
    selection = guided_gud_selection_contract()
    if canonical_json_bytes(prereg["selection"]) != canonical_json_bytes(selection):
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD behavior-only selection changed."
        )
    runtime = guided_gud_runtime_contract()
    if canonical_json_bytes(prereg["runtime"]) != canonical_json_bytes(runtime):
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD runtime/reset contract changed."
        )
    implementation = _implementation_contract()
    if canonical_json_bytes(prereg["implementation"]) != canonical_json_bytes(
        implementation
    ):
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD implementation hashes changed."
        )
    preimage = guided_gud_evaluation_preimage(
        source=source,
        splits=splits,
        basis=basis,
        mask=mask,
        mask_file_sha256=mask_sha,
        arms=arms,
        training=expected_training,
        selection=selection,
        runtime=runtime,
        implementation=implementation,
    )
    evaluation_sha = canonical_sha256(preimage)
    if prereg["evaluation_contract_sha256"] != evaluation_sha:
        raise GuidedGUDCandidateTournamentError(
            "Guided-GUD evaluation contract SHA changed."
        )
    return AuthenticatedGuidedGUDCandidateTournament(
        path=prereg_path,
        file_sha256=expected,
        source=source,
        splits=splits,
        basis=basis,
        site_mask=mask,
        site_mask_path=mask_path,
        site_mask_file_sha256=mask_sha,
        arms=arms,
        schedule=schedule,
        runtime=runtime,
        implementation=implementation,
        evaluation_contract_sha256=evaluation_sha,
    )


@dataclass(frozen=True, slots=True)
class _DurableV4CheckpointReceipt:
    step: int
    checkpoint_sha256: str
    factor_state_sha256: str
    metrics: Mapping[str, Any]
    pareto_safe_vs_step0: bool


class GuidedGUDTopThreeSink(v3_tournament.TournamentTopThreeSink):
    """Retain at most three distinct full v4 states across all eight arms."""

    def __init__(self, output_root: Path | str, *, evaluation_contract_sha256: str):
        super().__init__(
            output_root,
            evaluation_contract_sha256=evaluation_contract_sha256,
        )

    def callback_for_arm(
        self,
        arm_id: str,
        arm_index: int,
    ) -> Callable[[DurableBehaviorCheckpoint], None]:
        if arm_id not in ARM_IDS or arm_index != ARM_IDS.index(arm_id):
            raise GuidedGUDCandidateTournamentError(
                "Unknown or reordered guided-GUD arm."
            )
        if arm_id in self._events:
            raise GuidedGUDCandidateTournamentError(
                "Guided-GUD arm callback already opened."
            )
        self._events[arm_id] = []

        def callback(event: DurableBehaviorCheckpoint) -> None:
            if not isinstance(event, DurableBehaviorCheckpoint):
                raise GuidedGUDCandidateTournamentError(
                    "Guided-GUD callback event is not durable."
                )
            events = self._events[arm_id]
            expected_step = (
                CHECKPOINT_STEPS[len(events)]
                if len(events) < len(CHECKPOINT_STEPS)
                else None
            )
            if event.step != expected_step:
                raise GuidedGUDCandidateTournamentError(
                    f"{arm_id} callback expected {expected_step}, got {event.step}."
                )
            payload = event.factor_state.as_factor_payload()
            try:
                GuidedCheckpointRecoverySink._validate_v4_factor_payload(
                    payload,
                    expected_factor_state_sha256=(
                        event.factor_state.factor_state_sha256
                    ),
                )
            except Exception as error:
                raise GuidedGUDCandidateTournamentError(
                    "Callback state is not complete authenticated v4."
                ) from error
            metrics = event.metrics.as_dict()
            rank = v3_tournament.tournament_behavior_rank(metrics)
            runtime_digest = event.factor_state.factor_state_sha256
            portable_digest = portable_v4_factor_state_sha256(payload)
            observation = self._append_pending_observation(
                arm_id=arm_id,
                arm_index=arm_index,
                event=event,
                metrics=metrics,
                rank=rank,
                runtime_digest=runtime_digest,
                portable_digest=portable_digest,
            )
            events.append(
                _DurableV4CheckpointReceipt(
                    step=event.step,
                    checkpoint_sha256=event.checkpoint_sha256,
                    factor_state_sha256=runtime_digest,
                    metrics=dict(metrics),
                    pareto_safe_vs_step0=event.pareto_safe_vs_step0,
                )
            )
            if event.step != 0 and not event.pareto_safe_vs_step0:
                self._finalize_observation(
                    observation,
                    disposition="ineligible-non-pareto",
                    retained_after_callback=False,
                    factor_pack_created=False,
                )
                return
            previous = self._retained.get(portable_digest)
            if previous is not None:
                if previous.rank != rank or dict(previous.metrics) != metrics:
                    self._finalize_observation(
                        observation,
                        disposition="error-inconsistent-duplicate",
                        retained_after_callback=False,
                        factor_pack_created=False,
                    )
                    raise GuidedGUDCandidateTournamentError(
                        "One portable v4 state produced inconsistent behavior."
                    )
                self._finalize_observation(
                    observation,
                    disposition="duplicate-retained-state",
                    retained_after_callback=True,
                    factor_pack_created=False,
                )
                return
            directory = self.retained_root / portable_digest
            pack_path = directory / "factor_pack.pt"
            candidate = v3_tournament._RetainedState(
                arm_id=arm_id,
                arm_index=arm_index,
                step=event.step,
                rank=rank,
                runtime_factor_state_sha256=runtime_digest,
                portable_factor_state_sha256=portable_digest,
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
            if portable_digest not in keep_hashes:
                self._finalize_observation(
                    observation,
                    disposition="not-retained-rank-cutoff",
                    retained_after_callback=False,
                    factor_pack_created=False,
                )
                return
            directory.mkdir()
            v3_tournament._atomic_torch_save(pack_path, payload)
            candidate.pack_sha256 = stage._file_sha256(pack_path)
            self._retained[portable_digest] = candidate
            evicted_digests: list[str] = []
            for evicted in tuple(self._retained):
                if evicted not in keep_hashes:
                    owned = self.retained_root / evicted
                    if owned.parent != self.retained_root or owned.name != evicted:
                        raise GuidedGUDCandidateTournamentError(
                            "Refusing unsafe guided-GUD retained cleanup."
                        )
                    shutil.rmtree(owned)
                    del self._retained[evicted]
                    evicted_digests.append(evicted)
            v3_tournament._atomic_json(
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
            raise GuidedGUDCandidateTournamentError(
                f"{arm_id} durable v4 checkpoint series is incomplete."
            )
        if (
            getattr(result, "sealed_metrics", None) is not None
            or getattr(result, "sealed_evaluation_unlocked", None) is not False
            or getattr(result, "sealed_evaluation_policy", None) != "disabled"
        ):
            raise GuidedGUDCandidateTournamentError(
                "Guided-GUD arm accessed sealed behavior."
            )
        terminal = getattr(result, "checkpoints", None)
        if (
            not isinstance(terminal, tuple)
            or tuple(value.step for value in terminal) != CHECKPOINT_STEPS
        ):
            raise GuidedGUDCandidateTournamentError(
                "Terminal v4 checkpoints disagree with callbacks."
            )
        for event, row in zip(events, terminal, strict=True):
            if (
                event.factor_state_sha256 != row.factor_state_sha256
                or event.metrics != row.metrics.as_dict()
                or event.pareto_safe_vs_step0 != row.pareto_safe_vs_step0
            ):
                raise GuidedGUDCandidateTournamentError(
                    "Terminal v4 checkpoint evidence changed."
                )
        observations = [
            value for value in self._observed_callbacks if value["arm_id"] == arm_id
        ]
        if len(observations) != len(CHECKPOINT_STEPS) or any(
            value["disposition"] == "observed-pending" for value in observations
        ):
            raise GuidedGUDCandidateTournamentError(
                "Arm terminal verification found incomplete callback evidence."
            )
        for observation, row in zip(observations, terminal, strict=True):
            observation["terminal_verified"] = True
            observation["terminal_pareto_safe_vs_step0"] = row.pareto_safe_vs_step0
        self._completed.add(arm_id)
        self._events[arm_id].clear()
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
            "distinct_full_v4_states_only": True,
            "portable_content_identity_used": True,
            "step_zero_ranked": True,
            "later_states_require_pareto_safety_vs_step_zero": True,
            "sealed_metrics_used": False,
            "internal_numeric_matching_used": False,
            "callback_metrics_are_durable_without_factor_pack_retention": True,
            "maximum_retained_factor_packs": MAX_RETAINED_STATES,
        }
        v3_tournament._atomic_json(
            self.output_root / "candidate_frontier.json",
            payload,
        )

    def frontier_binding(self) -> dict[str, Any]:
        path = self.output_root / "candidate_frontier.json"
        if not path.is_file():
            raise GuidedGUDCandidateTournamentError("V4 candidate frontier is absent.")
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

    def _reauthenticate_retained_state(self, value: Any) -> None:
        expected_directory = (
            self.retained_root / value.portable_factor_state_sha256
        ).resolve()
        expected_pack = (expected_directory / "factor_pack.pt").resolve()
        expected_record = (expected_directory / "candidate_record.json").resolve()
        if (
            value.pack_path.resolve() != expected_pack
            or expected_directory.parent != self.retained_root.resolve()
            or not expected_pack.is_file()
            or expected_pack.is_symlink()
            or stage._file_sha256(expected_pack) != value.pack_sha256
            or not expected_record.is_file()
            or expected_record.is_symlink()
            or stage._strict_json(expected_record) != value.as_dict()
        ):
            raise GuidedGUDCandidateTournamentError(
                "A retained v4 factor pack or record changed before selection."
            )
        try:
            payload = torch.load(expected_pack, map_location="cpu", weights_only=True)
            GuidedCheckpointRecoverySink._validate_v4_factor_payload(
                payload,
                expected_factor_state_sha256=value.runtime_factor_state_sha256,
            )
            portable_sha = portable_v4_factor_state_sha256(payload)
        except Exception as error:
            raise GuidedGUDCandidateTournamentError(
                "A retained v4 factor pack failed final authentication."
            ) from error
        if portable_sha != value.portable_factor_state_sha256:
            raise GuidedGUDCandidateTournamentError(
                "A retained v4 factor pack changed portable identity."
            )

    def finalize(self) -> tuple[dict[str, Any], tuple[Any, ...]]:
        if self._completed != set(ARM_IDS):
            raise GuidedGUDCandidateTournamentError(
                "Cannot select a v4 winner before all eight arms."
            )
        if len(self._observed_callbacks) != EXPECTED_CALLBACK_COUNT or any(
            value["disposition"] == "observed-pending"
            or value["disposition"] not in v3_tournament.FINAL_CALLBACK_DISPOSITIONS
            or value["terminal_verified"] is not True
            or value["terminal_pareto_safe_vs_step0"] != value["pareto_safe_vs_step0"]
            for value in self._observed_callbacks
        ):
            raise GuidedGUDCandidateTournamentError(
                "Cannot select a v4 winner from incomplete callback evidence."
            )
        ordered = tuple(
            sorted(self._retained.values(), key=lambda value: value.total_order)
        )
        if len(ordered) != MAX_RETAINED_STATES:
            raise GuidedGUDCandidateTournamentError(
                "Guided-GUD tournament produced no complete top three."
            )
        for value in ordered:
            self._reauthenticate_retained_state(value)
        self._write_frontier()
        frontier = self.frontier_binding()
        payload = {
            "schema": WINNER_SCHEMA,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "winner": ordered[0].as_dict(),
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
            "stop_reason": "guided-gud-candidate-winner-selected-before-sealed",
            "candidate_frontier": frontier,
        }
        v3_tournament._atomic_json(
            self.output_root / "winner_selection.json",
            payload,
            exclusive=True,
        )
        return payload, ordered


@dataclass(frozen=True, slots=True)
class GuidedGUDArmExecution:
    result: Any
    reset_state_sha256: str
    rollback_state_sha256: str
    stored_source_v3_factor_state_sha256: str
    runtime_rebound_source_v3_factor_state_sha256: str
    trainer_initial_factor_state_sha256: str
    basis_method: str
    site_mask_enabled: bool
    post_optimizer_mask_verification_count: int
    guided_gud_parameter_count: int
    sealed_forward_count: int


class ProductionGuidedGUDCandidateTournamentRuntime(
    v3_tournament.ProductionCandidateTournamentRuntime
):
    """Compile once; restore exact v3, install audited GUD, train, roll back."""

    def prepare(
        self,
        contract: AuthenticatedGuidedGUDCandidateTournament,
    ) -> Mapping[str, Any]:
        if type(contract) is not AuthenticatedGuidedGUDCandidateTournament:
            raise GuidedGUDCandidateTournamentError(
                "Production v4 preparation requires its authenticated contract."
            )
        return super().prepare(contract.source.tournament_contract)

    def _rollback_guided_gud_arm(
        self,
        transaction: Any,
        guided_gud_tranche: Any | None,
    ) -> str:
        """Attempt every rollback layer independently, then prove exact bytes."""

        failures: list[tuple[str, BaseException]] = []
        if guided_gud_tranche is not None:
            try:
                guided_gud_tranche.rollback_guided_only()
            except BaseException as error:
                failures.append(("guided-GUD rollback_guided_only", error))
                try:
                    if getattr(guided_gud_tranche, "is_applied", False):
                        guided_gud_tranche.rollback()
                except BaseException as fallback_error:
                    failures.append(("guided-GUD rollback fallback", fallback_error))
        try:
            self.rollback_guided_installation(transaction)
        except BaseException as error:
            failures.append(("guided-QKVO rollback", error))
        try:
            transaction.rollback()
        except BaseException as error:
            failures.append(("base transaction rollback", error))
        try:
            rollback_sha = self.state_sha256(self.compiled.child_model)
        except BaseException as error:
            failures.append(("compiled-state authentication", error))
            rollback_sha = ""
        if rollback_sha != self.compiled_state_sha256:
            failures.append(
                (
                    "compiled-state mismatch",
                    GuidedGUDCandidateTournamentError(
                        "Rollback did not restore the exact compiled child bytes."
                    ),
                )
            )
        if failures:
            combined = GuidedGUDCandidateTournamentError(
                "One or more guided-GUD rollback layers failed."
            )
            for label, error in failures:
                combined.add_note(f"{label}: {type(error).__name__}: {error}")
            raise combined
        return rollback_sha

    def run_arm(
        self,
        contract: AuthenticatedGuidedGUDCandidateTournament,
        arm: GuidedGUDTournamentArm,
        callback: Callable[[DurableBehaviorCheckpoint], None],
    ) -> GuidedGUDArmExecution:
        if (
            self.parent is None
            or self.compiled is None
            or self.compiled_state_sha256 is None
        ):
            raise GuidedGUDCandidateTournamentError("Runtime is not prepared.")
        if self.state_sha256(self.compiled.child_model) != self.compiled_state_sha256:
            raise GuidedGUDCandidateTournamentError(
                "Compiled child was not reset before a guided-GUD arm."
            )
        transaction: Any = None
        guided_gud_tranche: Any = None
        try:
            transaction = self.install_transaction(self.compiled)
            source_guided = contract.source.tournament_contract.source
            self.restore_source_v2(transaction, source_guided)
            self.install_guided_transaction(transaction, source_guided)
            source_v2_state = capture_target_native_factor_state(transaction)
            live_qkvo_binding = authenticate_guided_factor_sources(
                transaction,
                self._guided_tranche,
                source_v2_state,
            )
            runtime_v3_payload = rebind_recovered_v3_factor_payload(
                contract.source.factor_payload,
                live_qkvo_binding,
            )
            if (
                portable_v3_factor_state_sha256(
                    runtime_v3_payload,
                    live_qkvo_binding,
                )
                != contract.source.portable_factor_state_sha256
            ):
                raise GuidedGUDCandidateTournamentError(
                    "Runtime rebinding changed the retained v3 portable identity."
                )
            runtime_v3_sha = _sha256(
                runtime_v3_payload.get("factor_state_sha256"),
                "runtime rebound v3 state",
            )
            self._guided_tranche.set_training_stage("joint")
            if self._guided_tranche.training_stage != "joint":
                raise GuidedGUDCandidateTournamentError(
                    "QKVO tranche did not enter joint before exact v3 restore."
                )
            restored_v3 = behavior.restore_target_native_factor_payload(
                transaction,
                runtime_v3_payload,
                guided_tranche=self._guided_tranche,
                expected_guided_source_binding=live_qkvo_binding,
            )
            if (
                restored_v3.factor_state_sha256 != runtime_v3_sha
                or restored_v3.schema != BEHAVIOR_FACTOR_PACK_V3_SCHEMA
            ):
                raise GuidedGUDCandidateTournamentError(
                    "Exact retained v3 restore changed before GUD installation."
                )
            recaptured_v3 = capture_target_native_factor_state(
                transaction,
                guided_tranche=self._guided_tranche,
                guided_source_binding=live_qkvo_binding,
            )
            if recaptured_v3.factor_state_sha256 != runtime_v3_sha:
                raise GuidedGUDCandidateTournamentError(
                    "Live 26-factor state differs from the rebound v3 payload."
                )
            rows = contract.basis.rows_for(arm.basis_method)
            guided_gud_tranche = (
                gud_adapter.install_guided_dense_swiglu_projection_residual_tranche(
                    transaction.swiglu_handle,
                    guided_right_rows=rows,
                    source_basis_tensor_sha256s=(
                        contract.basis.source_basis_tensor_sha256s[arm.basis_method]
                    ),
                    expected_audit_report_sha256=(contract.basis.audit_report_sha256),
                    expected_basis_envelope_file_sha256=(contract.basis.file_sha256),
                    expected_basis_payload_sha256=(contract.basis.payload_sha256),
                    expected_basis_manifest_sha256=(
                        contract.basis.basis_manifest_sha256
                    ),
                    owner_name=(
                        f"guided_dense_swiglu_{arm.basis_method.replace('-', '_')}_"
                        f"{arm.mask_mode.replace('-', '_')}_registry"
                    ),
                )
            )
            if (
                guided_gud_tranche.report.guided_adapter_parameter_count
                != EXPECTED_GUIDED_GUD_PARAMETER_COUNT
                or guided_gud_tranche.report.model_parameter_count_before_guided_installation
                + EXPECTED_GUIDED_GUD_PARAMETER_COUNT
                != guided_gud_tranche.report.model_parameter_count_installed
            ):
                raise GuidedGUDCandidateTournamentError(
                    "Guided-GUD installation is not exactly 4,792,320 parameters."
                )
            post_install_v3 = capture_target_native_factor_state(
                transaction,
                guided_tranche=self._guided_tranche,
                guided_source_binding=live_qkvo_binding,
            )
            if post_install_v3.factor_state_sha256 != runtime_v3_sha:
                raise GuidedGUDCandidateTournamentError(
                    "Exact-zero GUD installation changed recovered v3 bytes."
                )
            domains = {
                name: behavior.load_behavior_domain_suite(
                    name,
                    contract.splits[name].payload,
                    artifact_sha256=contract.splits[name].sha256,
                )
                for name in ("fit", "candidate")
            }
            thresholds = behavior.CandidateBehaviorThresholds(
                **source_guided.execution_contract.thresholds["candidate"].as_dict()
            )
            mask_count = 0
            guard: GuidedGUDGradientMaskGuard | None = None

            def verify_mask(_step: int) -> None:
                nonlocal mask_count
                if guard is None:
                    raise GuidedGUDCandidateTournamentError(
                        "Guided-GUD mask callback has no guard."
                    )
                guard.verify_step()
                mask_count += 1

            guard_context: Any = contextlib.nullcontext()
            post_step: Callable[[int], None] | None = None
            if arm.site_mask_enabled:
                guard = GuidedGUDGradientMaskGuard(
                    guided_gud_tranche.guided_registry,
                    contract.site_mask,
                )
                guard_context = guard
                post_step = verify_mask
            with guard_context:
                result = behavior.train_target_native_behavior_distillation(
                    transaction=transaction,
                    parent_model=self.parent,
                    domains=domains,
                    schedule=contract.schedule,
                    candidate_thresholds=thresholds,
                    seed=source_guided.execution_contract.seed,
                    initial_factor_payload=runtime_v3_payload,
                    guided_tranche=self._guided_tranche,
                    guided_gud_tranche=guided_gud_tranche,
                    durable_checkpoint_callback=callback,
                    post_optimizer_step_callback=post_step,
                )
            if arm.site_mask_enabled and mask_count != contract.schedule.steps:
                raise GuidedGUDCandidateTournamentError(
                    "Masked GUD arm was not verified after every optimizer step."
                )
            if not arm.site_mask_enabled and mask_count != 0:
                raise GuidedGUDCandidateTournamentError(
                    "All-site GUD arm unexpectedly used a mask."
                )
            initial_sha = getattr(result, "initial_factor_state_sha256", None)
            if initial_sha != runtime_v3_sha:
                raise GuidedGUDCandidateTournamentError(
                    "Trainer did not authenticate the exact pre-restored v3 source."
                )
            cleanup_transaction = transaction
            transaction = None
            rollback_sha = self._rollback_guided_gud_arm(
                cleanup_transaction,
                guided_gud_tranche,
            )
            guided_gud_tranche = None
            return GuidedGUDArmExecution(
                result=result,
                reset_state_sha256=self.compiled_state_sha256,
                rollback_state_sha256=rollback_sha,
                stored_source_v3_factor_state_sha256=(
                    contract.source.runtime_factor_state_sha256
                ),
                runtime_rebound_source_v3_factor_state_sha256=runtime_v3_sha,
                trainer_initial_factor_state_sha256=initial_sha,
                basis_method=arm.basis_method,
                site_mask_enabled=arm.site_mask_enabled,
                post_optimizer_mask_verification_count=mask_count,
                guided_gud_parameter_count=EXPECTED_GUIDED_GUD_PARAMETER_COUNT,
                sealed_forward_count=0,
            )
        except BaseException as error:
            if transaction is not None:
                cleanup_transaction = transaction
                transaction = None
                try:
                    self._rollback_guided_gud_arm(
                        cleanup_transaction,
                        guided_gud_tranche,
                    )
                except BaseException as rollback_error:
                    rollback_error.add_note(
                        f"Original arm failure: {type(error).__name__}: {error}"
                    )
                    raise rollback_error from error
            raise


def execute_guided_gud_candidate_tournament(
    contract: AuthenticatedGuidedGUDCandidateTournament,
    output_root: Path | str,
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Run eight preregistered arms and stop at a candidate-only v4 winner."""

    if type(contract) is not AuthenticatedGuidedGUDCandidateTournament:
        raise GuidedGUDCandidateTournamentError(
            "execute requires an authenticated guided-GUD contract."
        )
    contract = authenticate_guided_gud_candidate_preregistration(
        contract.path,
        contract.file_sha256,
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise GuidedGUDCandidateTournamentError("output_root must not already exist.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    sink = GuidedGUDTopThreeSink(
        output,
        evaluation_contract_sha256=contract.evaluation_contract_sha256,
    )
    runner = (
        ProductionGuidedGUDCandidateTournamentRuntime() if runtime is None else runtime
    )
    prepare_report: Mapping[str, Any] | None = None
    arm_reports: list[dict[str, Any]] = []
    try:
        prepare_report = runner.prepare(contract)
        for arm in contract.arms:
            callback = sink.callback_for_arm(arm.arm_id, arm.arm_index)
            execution = runner.run_arm(contract, arm, callback)
            if (
                not isinstance(execution, GuidedGUDArmExecution)
                or execution.reset_state_sha256 != execution.rollback_state_sha256
                or execution.stored_source_v3_factor_state_sha256
                != arm.source_v3_factor_state_sha256
                or execution.basis_method != arm.basis_method
                or execution.site_mask_enabled is not arm.site_mask_enabled
                or execution.guided_gud_parameter_count
                != EXPECTED_GUIDED_GUD_PARAMETER_COUNT
                or execution.sealed_forward_count != 0
                or (
                    arm.site_mask_enabled
                    and execution.post_optimizer_mask_verification_count
                    != contract.schedule.steps
                )
                or (
                    not arm.site_mask_enabled
                    and execution.post_optimizer_mask_verification_count != 0
                )
            ):
                raise GuidedGUDCandidateTournamentError(
                    "Guided-GUD arm execution receipt failed."
                )
            sink.complete_arm(arm.arm_id, execution.result)
            arm_reports.append(
                {
                    "arm": arm.as_dict(),
                    "reset_state_sha256": execution.reset_state_sha256,
                    "rollback_state_sha256": execution.rollback_state_sha256,
                    "runtime_rebound_source_v3_factor_state_sha256": (
                        execution.runtime_rebound_source_v3_factor_state_sha256
                    ),
                    "trainer_initial_factor_state_sha256": (
                        execution.trainer_initial_factor_state_sha256
                    ),
                    "post_optimizer_mask_verification_count": (
                        execution.post_optimizer_mask_verification_count
                    ),
                    "guided_gud_parameter_count": (
                        execution.guided_gud_parameter_count
                    ),
                    "sealed_forward_count": 0,
                }
            )
        winner, retained = sink.finalize()
        frontier = sink.frontier_binding()
        if (
            frontier["observed_callback_count"] != EXPECTED_CALLBACK_COUNT
            or frontier["pending_callback_count"] != 0
            or frontier["completed_arms"] != list(ARM_IDS)
            or len(retained) != MAX_RETAINED_STATES
        ):
            raise GuidedGUDCandidateTournamentError(
                "Final guided-GUD frontier is incomplete."
            )
        report = {
            "schema": RUN_SCHEMA,
            "status": "candidate-winner-selected",
            "evaluation_contract_sha256": contract.evaluation_contract_sha256,
            "preregistration_sha256": contract.file_sha256,
            "source_v3_runtime_factor_state_sha256": (
                contract.source.runtime_factor_state_sha256
            ),
            "source_v3_portable_factor_state_sha256": (
                contract.source.portable_factor_state_sha256
            ),
            "basis_payload_sha256": contract.basis.payload_sha256,
            "audit_report_sha256": contract.basis.audit_report_sha256,
            "prepare": dict(prepare_report),
            "arms": arm_reports,
            "winner_selection": winner,
            "candidate_frontier": frontier,
            "sealed_forward_count": 0,
            "sealed_evaluation_accessed": False,
            "model_checkpoint_written": False,
            "publication_attempted": False,
            "stop_after_candidate_winner": True,
        }
        v3_tournament._atomic_json(
            output / "run_report.json",
            report,
            exclusive=True,
        )
        return report
    except BaseException as error:
        failure = {
            "schema": RUN_SCHEMA,
            "status": "failed-before-sealed",
            "evaluation_contract_sha256": contract.evaluation_contract_sha256,
            "preregistration_sha256": contract.file_sha256,
            "prepare": None if prepare_report is None else dict(prepare_report),
            "arms": arm_reports,
            "candidate_frontier": sink.frontier_binding(),
            "error_type": type(error).__name__,
            "error": str(error),
            "sealed_forward_count": 0,
            "sealed_evaluation_accessed": False,
            "model_checkpoint_written": False,
            "publication_attempted": False,
        }
        try:
            v3_tournament._atomic_json(output / "run_report.json", failure)
        except Exception as report_error:
            error.add_note(f"Failure report also failed: {report_error!r}")
        raise
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_guided_gud_candidate_preregistration(
        args.preregistration,
        args.preregistration_sha256,
    )
    if args.preflight:
        print(json.dumps(contract.preflight_report(), indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        raise GuidedGUDCandidateTournamentError("--execute requires --output-root.")
    report = execute_guided_gud_candidate_tournament(
        contract,
        args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
