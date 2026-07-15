"""Real-model behavior replay collector for guided Gate/Up/Down bases.

The model/loss replay itself is reused from the QKVO audit.  This module owns
only the GUD projection hooks and bounded streaming state.  Gate, Up, and Down
are observed at their separate live projection boundaries; no shared MLP
input hook or pre-collected full-width replay table exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neural_forge import hf_target_native_behavior_distillation as _trainer
from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    DenseSwiGLUResidualProjection,
)
from neural_forge.hf_gud_output_jacobian_audit import (
    GUD_ROLE_ORDER,
    HARD_MAX_AUXILIARY_BYTES,
    GUDOutputJacobianAuditConfig,
    GUDRoleSpec,
    GUDSiteAuditReport,
    GUDStageGeometry,
    finalize_streamed_gud_site_basis,
    finalize_streaming_gud_envelope,
    validate_qwen_30_layer_stage_geometry,
)
from neural_forge.hf_guided_checkpoint_recovery import (
    portable_v3_factor_state_sha256,
)
from neural_forge.hf_output_jacobian_sketch import (
    OUTPUT_JACOBIAN_SKETCH_SCHEMA,
    OutputJacobianBlockIdentity,
    OutputJacobianSketchSpec,
    TwoPassOutputJacobianSketch,
)
from neural_forge.hf_progressive_target_bridge import ExactTargetCompilerTransaction
from neural_forge.hf_qkvo_output_jacobian_audit import (
    QKVOOutputJacobianAuditConfig,
    _build_replay_plans,
    _LossContext,
    _replay_block_count,
    _replay_loss_pass,
    _tensor_sha256,
    _validate_domain_and_schedule,
    _validate_exact_topology,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BehaviorDistillationSchedule,
    BehaviorDomainSuite,
    GuidedFactorSourceBinding,
    TargetNativeFactorState,
    capture_target_native_factor_state,
)


GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA = (
    "neural-forge-gud-output-jacobian-real-replay-v1"
)
GUD_COMPLETE_BASIS_PACK_SCHEMA = "neural-forge-gud-four-candidate-basis-pack-v1"
_CANDIDATES = (
    "output-jacobian",
    "balanced",
    "procrustes-hybrid",
    "random-control",
)
_SITE_DIAGNOSTIC_FIELDS = (
    "sample_count",
    "block_count",
    "normal_effective_rank",
    "selected_singular_values",
    "raw_frobenius_sq_estimate",
    "normal_frobenius_sq_estimate",
    "normal_fraction",
    "guided_capture_energy",
    "guided_capture_fraction",
    "random_capture_energy",
    "random_capture_fraction",
    "guided_over_random_ratio",
    "normal_fraction_gate_passed",
    "guided_capture_gate_passed",
    "guided_over_random_gate_passed",
    "gate_passed",
)
_FP32_BYTES = 4
_FP64_BYTES = 8
_FINGERPRINT_STATS = 6
_IDENTITY_CHAIN_HEADER = (
    f"{OUTPUT_JACOBIAN_SKETCH_SCHEMA}:ordered-block-identities-v1"
).encode()


class GUDOutputJacobianReplayError(RuntimeError):
    """Base failure for an uncertifiable real-model GUD replay."""


class GUDOutputJacobianReplayTopologyError(GUDOutputJacobianReplayError):
    """The live projection topology is not the declared GUD stage."""


class GUDOutputJacobianReplayMismatchError(GUDOutputJacobianReplayError):
    """Pass-one and pass-two model/loss evidence changed."""


class GUDOutputJacobianReplayLimitError(GUDOutputJacobianReplayError):
    """The explicit auxiliary-memory plan exceeds 256 MiB."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GUDOutputJacobianReplayError(
            "Replay evidence must be canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def role_local_mixed_precision_input_vjp_prevalidated(
    native_output_delta: Tensor,
    fp32_output_delta: Tensor,
    base_weight: Tensor,
    inherited_left: Tensor,
    inherited_right: Tensor,
) -> Tensor:
    """Reproduce the live wrapper's mixed-precision input VJP exactly.

    The dense branch remains in the native model dtype.  The inherited branch
    follows the wrapper's explicit FP32 input/factor path, is cast back through
    the FP32-to-native forward cast, and is added in the native input dtype.
    Only the completed sensitivity is promoted to FP32 for sketching.  This is
    a hot-path helper: topology and dtype metadata are validated by the caller.
    """

    base_input_vjp = native_output_delta @ base_weight
    inherited_input_vjp = (
        fp32_output_delta @ inherited_left
    ) @ inherited_right
    return (
        base_input_vjp + inherited_input_vjp.to(dtype=base_input_vjp.dtype)
    ).to(dtype=torch.float32)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise GUDOutputJacobianReplayError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GUDOutputJacobianReplayError(f"{name} must be positive.") from error
    if result <= 0:
        raise GUDOutputJacobianReplayError(f"{name} must be positive.")
    return result


def _derived_seed(seed: int, *parts: object) -> int:
    payload = ":".join(
        (GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA, str(seed), *(str(part) for part in parts))
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


@dataclass(frozen=True, slots=True)
class GUDOutputJacobianReplayConfig:
    """Replay, streaming-sketch, fingerprint, and memory bounds."""

    prompt_count: int = 24
    horizon: int = 8
    rank: int = 8
    sketch_width: int = 16
    balanced_envelope_width: int = 32
    balanced_stream_rows: int = 16
    fingerprint_lanes: int = 8
    fingerprint_chunk_elements: int = 32 * 1024
    max_callback_rows: int = 256
    seed: int = 0
    min_normal_fraction: float = 0.0
    min_guided_over_random_ratio: float = 0.0
    max_bytes: int = HARD_MAX_AUXILIARY_BYTES

    def __post_init__(self) -> None:
        for name in (
            "prompt_count",
            "horizon",
            "rank",
            "sketch_width",
            "balanced_envelope_width",
            "balanced_stream_rows",
            "fingerprint_lanes",
            "fingerprint_chunk_elements",
            "max_callback_rows",
            "max_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if isinstance(self.seed, bool):
            raise GUDOutputJacobianReplayError("seed must be an integer.")
        try:
            object.__setattr__(self, "seed", operator.index(self.seed))
        except (TypeError, ValueError, OverflowError) as error:
            raise GUDOutputJacobianReplayError("seed must be an integer.") from error
        for name in ("min_normal_fraction", "min_guided_over_random_ratio"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise GUDOutputJacobianReplayError(
                    f"{name} must be finite and nonnegative."
                ) from error
            if not math.isfinite(value) or value < 0.0:
                raise GUDOutputJacobianReplayError(
                    f"{name} must be finite and nonnegative."
                )
            object.__setattr__(self, name, value)
        if self.rank > self.sketch_width:
            raise GUDOutputJacobianReplayError("rank cannot exceed sketch_width.")
        if (
            self.balanced_envelope_width < 2 * self.rank
            or self.balanced_envelope_width % 2
        ):
            raise GUDOutputJacobianReplayError(
                "balanced_envelope_width must be even and at least twice rank."
            )
        if self.balanced_stream_rows < self.rank:
            raise GUDOutputJacobianReplayError(
                "balanced_stream_rows cannot be smaller than rank."
            )
        if self.fingerprint_lanes < 2 or self.fingerprint_lanes > 32:
            raise GUDOutputJacobianReplayError(
                "fingerprint_lanes must be between 2 and 32."
            )
        if self.fingerprint_chunk_elements > 256 * 1024:
            raise GUDOutputJacobianReplayLimitError(
                "fingerprint_chunk_elements exceeds its bounded chunk limit."
            )
        if self.max_bytes > HARD_MAX_AUXILIARY_BYTES:
            raise GUDOutputJacobianReplayLimitError(
                "max_bytes exceeds the hard 256 MiB ceiling."
            )

    @property
    def fingerprint_values_per_tensor(self) -> int:
        return _FINGERPRINT_STATS + self.fingerprint_lanes

    @property
    def fingerprint_values_per_wrapper(self) -> int:
        return 2 * self.fingerprint_values_per_tensor

    def basis_config(self) -> GUDOutputJacobianAuditConfig:
        return GUDOutputJacobianAuditConfig(
            rank=self.rank,
            sketch_width=self.sketch_width,
            balanced_envelope_width=self.balanced_envelope_width,
            seed=self.seed,
            max_blocks_per_site=max(1, self.prompt_count * self.horizon * 2),
            max_rows_per_site=max(1, self.max_callback_rows * 2),
            min_normal_fraction=self.min_normal_fraction,
            min_guided_over_random_ratio=self.min_guided_over_random_ratio,
            max_bytes=self.max_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return _canonical_sha256(
            {"schema": GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA, "config": self.to_dict()}
        )


@dataclass(frozen=True, slots=True)
class GUDOutputJacobianReplayReport:
    payload: Mapping[str, Any]

    @property
    def report_sha256(self) -> str:
        return str(self.payload["report_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class GUDOutputJacobianReplayResult:
    """Four complete candidate packs, each mapping Gate/Up/Down to rank rows."""

    basis_packs: Mapping[str, Mapping[str, Tensor]]
    site_reports: tuple[GUDSiteAuditReport, ...]
    report: GUDOutputJacobianReplayReport

    def candidate(self, name: str) -> Mapping[str, Tensor]:
        if name not in _CANDIDATES:
            raise GUDOutputJacobianReplayError("Unknown GUD candidate name.")
        return self.basis_packs[name]


@dataclass(frozen=True, slots=True)
class _LiveV3Provenance:
    runtime_factor_state_sha256: str
    portable_factor_state_sha256: str
    factor_manifest_sha256: str
    tensor_sha256s: tuple[tuple[str, str], ...]
    lexical_rank: int
    parameter_ids: tuple[int, ...]
    parameter_data_ptrs: tuple[int, ...]
    parameter_versions: tuple[tuple[str, int], ...]
    total_nbytes: int
    largest_factor_nbytes: int


def _validate_live_guided_qkvo_binding(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    binding: GuidedFactorSourceBinding,
) -> None:
    if not isinstance(binding, GuidedFactorSourceBinding):
        raise GUDOutputJacobianReplayTopologyError(
            "source_guided_qkvo_binding must be authenticated."
        )
    assertion = getattr(guided_tranche, "assert_live", None)
    if not callable(assertion):
        raise GUDOutputJacobianReplayTopologyError(
            "guided_tranche does not expose a live-state assertion."
        )
    assertion()
    if (
        getattr(guided_tranche, "inherited_handle", None)
        is not transaction.qkvo_handle
        or getattr(guided_tranche, "training_stage", None) != "joint"
    ):
        raise GUDOutputJacobianReplayTopologyError(
            "The production GUD audit requires this transaction's live joint "
            "guided-QKVO tranche."
        )
    source_accessor = getattr(guided_tranche, "factor_source_binding", None)
    if not callable(source_accessor):
        raise GUDOutputJacobianReplayTopologyError(
            "The live guided-QKVO tranche exposes no source binding."
        )
    raw = source_accessor()
    expected = {
        "audit_report_sha256": binding.audit_report_sha256,
        "basis_envelope_file_sha256": binding.basis_envelope_file_sha256,
        "basis_payload_sha256": binding.basis_payload_sha256,
        "basis_manifest_sha256": binding.basis_manifest_sha256,
        "source_basis_tensor_sha256s": dict(
            binding.source_basis_tensor_sha256s
        ),
        "inherited_installation_sha256": binding.inherited_installation_sha256,
        "inherited_factor_state_sha256_at_install": (
            binding.inherited_factor_state_sha256_at_install
        ),
        "inherited_handle_identity": id(transaction.qkvo_handle),
        "inherited_registry_identity": id(
            transaction.qkvo_handle.adapter_registry
        ),
        "adapter_site_indices_sha256": binding.adapter_site_indices_sha256,
        "audit_site_indices_sha256": binding.audit_site_indices_sha256,
    }
    if not isinstance(raw, Mapping) or _trainer._canonical_sha256(
        raw
    ) != _trainer._canonical_sha256(expected):
        raise GUDOutputJacobianReplayTopologyError(
            "The supplied guided-QKVO binding does not authenticate the live "
            "tranche installation."
        )


def _capture_live_v3_provenance(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    binding: GuidedFactorSourceBinding,
) -> _LiveV3Provenance:
    """Capture and immediately reduce a complete live 26-factor v3 state."""

    _validate_live_guided_qkvo_binding(transaction, guided_tranche, binding)
    entries = _trainer._factor_parameters(
        transaction,
        guided_tranche=guided_tranche,
    )
    versions = tuple((name, int(value._version)) for name, value in entries)
    state: TargetNativeFactorState = capture_target_native_factor_state(
        transaction,
        guided_tranche=guided_tranche,
        guided_source_binding=binding,
    )
    if (
        state.schema != BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        or len(state.names) != 26
        or state.guided_source_binding != binding
        or state.lexical_rank is None
    ):
        raise GUDOutputJacobianReplayTopologyError(
            "The live source is not one complete authenticated v3 factor state."
        )
    manifest = state.as_manifest()
    factors = manifest.get("factors")
    if not isinstance(factors, list) or len(factors) != 26:
        raise GUDOutputJacobianReplayTopologyError(
            "The captured v3 factor manifest is incomplete."
        )
    tensor_hashes = tuple(
        zip(state.names, state.tensor_sha256s, strict=True)
    )
    result = _LiveV3Provenance(
        runtime_factor_state_sha256=state.factor_state_sha256,
        portable_factor_state_sha256=portable_v3_factor_state_sha256(
            manifest,
            binding,
        ),
        factor_manifest_sha256=_canonical_sha256(factors),
        tensor_sha256s=tensor_hashes,
        lexical_rank=int(state.lexical_rank),
        parameter_ids=state.parameter_ids,
        parameter_data_ptrs=state.parameter_data_ptrs,
        parameter_versions=versions,
        total_nbytes=sum(
            int(value.numel() * value.element_size()) for value in state.tensors
        ),
        largest_factor_nbytes=max(
            int(value.numel() * value.element_size()) for value in state.tensors
        ),
    )
    del state
    return result


def _source_v3_report_fields(
    before: _LiveV3Provenance,
    after: _LiveV3Provenance,
    binding: GuidedFactorSourceBinding,
) -> dict[str, Any]:
    def runtime_identity(source: _LiveV3Provenance) -> str:
        return _canonical_sha256(
            {
                "parameter_ids": list(source.parameter_ids),
                "parameter_data_ptrs": list(source.parameter_data_ptrs),
                "parameter_versions": [
                    list(row) for row in source.parameter_versions
                ],
            }
        )

    return {
        "source_v3_runtime_factor_state_sha256_before": (
            before.runtime_factor_state_sha256
        ),
        "source_v3_runtime_factor_state_sha256_after": (
            after.runtime_factor_state_sha256
        ),
        "source_v3_portable_factor_state_sha256": (
            before.portable_factor_state_sha256
        ),
        "source_v3_factor_manifest_sha256": before.factor_manifest_sha256,
        "source_v3_tensor_sha256s": dict(before.tensor_sha256s),
        "source_v3_lexical_rank": before.lexical_rank,
        "source_v3_factor_count": len(before.tensor_sha256s),
        "source_v3_total_nbytes": before.total_nbytes,
        "source_v3_runtime_parameter_identity_sha256_before": runtime_identity(
            before
        ),
        "source_v3_runtime_parameter_identity_sha256_after": runtime_identity(
            after
        ),
        "source_guided_qkvo_binding": binding.as_dict(),
        "source_guided_qkvo_binding_sha256": binding.binding_sha256,
        "source_v3_state_unchanged": before == after,
    }


def _role_dimensions(role_spec: GUDRoleSpec) -> tuple[int, int]:
    return role_spec.output_width, role_spec.input_width


def _sketch_memory_components(
    role_spec: GUDRoleSpec,
    config: GUDOutputJacobianReplayConfig,
    max_rows: int,
) -> tuple[int, int]:
    out_features, in_features = _role_dimensions(role_spec)
    occupied = config.rank
    width = config.sketch_width
    rank = config.rank
    range_width = min(width, in_features - occupied)
    occupied_bytes = 4 * occupied * (out_features + in_features)
    pass1 = occupied_bytes + 4 * out_features * width + 8 * in_features * width
    pass2 = occupied_bytes + 4 * (in_features + out_features) * (
        range_width + rank
    )
    factor_analysis = (
        8 * occupied * (out_features + in_features)
        + 32 * occupied * occupied
        + 16
        * (
            out_features * min(out_features, occupied)
            + in_features * min(in_features, occupied)
            + 2 * occupied * occupied
        )
    )
    pass1_block = 4 * max_rows * (
        3 * out_features + 3 * in_features + 2 * occupied
    ) + 4 * (max_rows * width + in_features * width)
    transition = (
        24 * in_features * width + 16 * in_features * rank + 8 * width * width
    )
    pass2_block = 4 * max_rows * (
        3 * out_features + 3 * in_features + 2 * occupied
    ) + 4 * (
        max_rows * (range_width + rank)
        + out_features * max(range_width, rank)
    )
    finalize = (
        16 * out_features * range_width
        + 8 * range_width * range_width
        + 8 * in_features * (range_width + rank)
        + 12 * rank * in_features
    )
    return max(pass1, pass2), max(
        factor_analysis, pass1_block, transition, pass2_block, finalize
    )


def plan_gud_real_replay_memory(
    geometry: GUDStageGeometry,
    *,
    replay_blocks: int,
    max_rows: int,
    config: GUDOutputJacobianReplayConfig,
) -> dict[str, Any]:
    """Certify simultaneous all-site sketches and bounded streamed evidence."""

    blocks = _positive_int(replay_blocks, "replay_blocks")
    rows = _positive_int(max_rows, "max_rows")
    if rows > config.max_callback_rows:
        raise GUDOutputJacobianReplayLimitError(
            "A replay prefix exceeds max_callback_rows."
        )
    persistent_sketches = 0
    largest_sketch_operation = 0
    input_width_sum = 0
    output_width_max = 0
    input_width_max = 0
    for role_spec in geometry.role_specs:
        persistent, transient = _sketch_memory_components(role_spec, config, rows)
        persistent_sketches += geometry.logical_layers * persistent
        largest_sketch_operation = max(largest_sketch_operation, transient)
        input_width_sum += role_spec.input_width
        output_width_max = max(output_width_max, role_spec.output_width)
        input_width_max = max(input_width_max, role_spec.input_width)
    wrappers = geometry.logical_layers * len(GUD_ROLE_ORDER)
    fingerprint_buffer = (
        blocks * wrappers * config.fingerprint_values_per_wrapper * _FP32_BYTES
    )
    fingerprint_host_clone = fingerprint_buffer
    fingerprint_chunk = (
        config.fingerprint_lanes
        * config.fingerprint_chunk_elements
        * _FP64_BYTES
        + config.fingerprint_chunk_elements * _FP64_BYTES
    )
    envelope_pack = (
        geometry.logical_layers
        * config.balanced_envelope_width
        * input_width_sum
        * _FP32_BYTES
    )
    # The core keeps a CPU FP64 envelope for balanced finalization and one
    # FP32 device copy for pass-two reduction.
    envelope_cpu_and_device = 3 * envelope_pack
    reduced_buckets = (
        geometry.logical_layers
        * len(GUD_ROLE_ORDER)
        * 2
        * config.balanced_stream_rows
        * config.balanced_envelope_width
        * _FP32_BYTES
    )
    final_four_packs = (
        geometry.logical_layers
        * len(_CANDIDATES)
        * config.rank
        * input_width_sum
        * _FP32_BYTES
    )
    wrapper_cast_and_vjp = rows * _FP32_BYTES * (
        4 * input_width_max + 2 * output_width_max + config.rank
    )
    balanced_finalize = _FP64_BYTES * (
        32
        * config.balanced_stream_rows
        * config.balanced_envelope_width
        + 96 * config.balanced_envelope_width**2
        + 16 * config.rank * input_width_max
    )
    tensor_hash_staging = 4 * 1024 * 1024
    components = {
        "all_site_output_jacobian_persistent_bytes": persistent_sketches,
        "largest_output_jacobian_operation_bytes": largest_sketch_operation,
        "pass_fingerprint_device_buffer_bytes": fingerprint_buffer,
        "pass_fingerprint_host_clone_bytes": fingerprint_host_clone,
        "chunked_fingerprint_temporary_bytes": fingerprint_chunk,
        "pass1_envelope_cpu_and_device_bytes": envelope_cpu_and_device,
        "pass2_reduced_bucket_bytes": reduced_buckets,
        "four_complete_candidate_pack_bytes": final_four_packs,
        "wrapper_cast_and_role_local_vjp_bytes": wrapper_cast_and_vjp,
        "largest_balanced_finalize_bytes": balanced_finalize,
        "bounded_tensor_hash_staging_bytes": tensor_hash_staging,
    }
    peak = sum(components.values())
    payload: dict[str, Any] = {
        "schema": f"{GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA}:memory-plan-v1",
        **components,
        "accounted_auxiliary_peak_bytes": peak,
        "max_auxiliary_bytes": config.max_bytes,
        "hard_max_auxiliary_bytes": HARD_MAX_AUXILIARY_BYTES,
        "within_limit": peak <= config.max_bytes <= HARD_MAX_AUXILIARY_BYTES,
        "site_count": wrappers,
        "replay_blocks": blocks,
        "maximum_callback_rows": rows,
        "full_all_sites_replay_tensor_table_bytes": 0,
        "full_lanes_rows_width_coefficient_table_bytes": 0,
        "fingerprint_mode": (
            "separable-chunked-full-element-order-sensitive-projection"
        ),
        "fingerprint_collision_free": False,
        "fingerprint_binds_every_element": True,
        "balanced_snapshot_mode": "bounded-streamed-row-projection-buckets",
        "memory_scope": {
            "includes": [
                "all-90-output-jacobian-sketches",
                "pass-one-streamed-full-width-envelope-sketches",
                "pass-two-reduced-coordinate-buckets",
                "separable-replay-fingerprint-buffers-and-chunks",
                "four-complete-final-candidate-packs",
                "balanced-finalization-buffers",
            ],
            "excludes": [
                "parent-and-child-model-weights",
                "standard-forward-backward-autograd-activations",
                "framework-allocator-reservations",
                "python-object-overhead",
            ],
        },
    }
    payload["memory_plan_sha256"] = _canonical_sha256(payload)
    if not payload["within_limit"]:
        raise GUDOutputJacobianReplayLimitError(
            "Real GUD replay exceeds the configured/hard auxiliary-memory ceiling."
        )
    return payload


def _bind_source_v3_capture_memory(
    memory_plan: Mapping[str, Any],
    source: _LiveV3Provenance,
    config: GUDOutputJacobianReplayConfig,
) -> dict[str, Any]:
    """Add the disjoint before/after source-capture phases to the peak plan."""

    payload = dict(memory_plan)
    payload.pop("memory_plan_sha256", None)
    replay_peak = int(payload["accounted_auxiliary_peak_bytes"])
    final_packs = int(payload["four_complete_candidate_pack_bytes"])
    hash_staging = int(payload["bounded_tensor_hash_staging_bytes"])
    before_peak = source.total_nbytes + source.largest_factor_nbytes + hash_staging
    after_peak = before_peak + final_packs
    payload.update(
        {
            "replay_phase_accounted_auxiliary_peak_bytes": replay_peak,
            "source_v3_cpu_clone_bytes": source.total_nbytes,
            "source_v3_largest_hash_temporary_bytes": (
                source.largest_factor_nbytes
            ),
            "source_v3_capture_before_replay_peak_bytes": before_peak,
            "source_v3_capture_after_pack_peak_bytes": after_peak,
            "source_v3_payload_tensor_reclone_bytes": 0,
            "source_v3_capture_uses_manifest_without_payload_reclone": True,
            "accounted_auxiliary_peak_bytes": max(replay_peak, before_peak, after_peak),
        }
    )
    payload["within_limit"] = bool(
        int(payload["accounted_auxiliary_peak_bytes"])
        <= config.max_bytes
        <= HARD_MAX_AUXILIARY_BYTES
    )
    memory_scope = dict(payload["memory_scope"])
    memory_scope["includes"] = [
        *list(memory_scope["includes"]),
        "complete-live-v3-source-capture-before-and-after-replay",
    ]
    payload["memory_scope"] = memory_scope
    payload["memory_plan_sha256"] = _canonical_sha256(payload)
    if not payload["within_limit"]:
        raise GUDOutputJacobianReplayLimitError(
            "Live v3 source capture plus GUD replay exceeds the hard auxiliary "
            "memory ceiling."
        )
    return payload


def _mixing_matrix(
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    row_ids = torch.arange(rows, dtype=torch.int64, device=device).reshape(-1, 1)
    column_ids = torch.arange(
        columns, dtype=torch.int64, device=device
    ).reshape(1, -1)
    values = row_ids + 1 + (seed % 1_000_003) + 104_729 * (column_ids + 1)
    values = values ^ (values << 13)
    values = values ^ (values >> 7)
    values = values ^ (values << 17)
    signs = values.bitwise_and(1).to(torch.float32).mul_(2.0).sub_(1.0)
    return signs.mul_(1.0 / math.sqrt(max(rows, 1)))


def _write_tensor_fingerprint(
    destination: Tensor,
    value: Tensor,
    *,
    seed: int,
    config: GUDOutputJacobianReplayConfig,
) -> None:
    flat = value.reshape(-1)
    if (
        flat.dtype != torch.float32
        or destination.dtype != torch.float32
        or destination.device != flat.device
        or destination.shape != (config.fingerprint_values_per_tensor,)
    ):
        raise GUDOutputJacobianReplayTopologyError(
            "Fingerprint tensors have incompatible metadata."
        )
    stats = torch.stack(
        (
            torch.isfinite(flat).all().to(torch.float32),
            flat.sum(dtype=torch.float32),
            flat.square().sum(dtype=torch.float32),
            flat.abs().sum(dtype=torch.float32),
            flat.min(),
            flat.max(),
        )
    )
    projections = torch.zeros(
        (config.fingerprint_lanes,), dtype=torch.float64, device=flat.device
    )
    lane_ids = torch.arange(
        config.fingerprint_lanes, dtype=torch.int64, device=flat.device
    ).reshape(1, -1)
    for start in range(0, int(flat.numel()), config.fingerprint_chunk_elements):
        segment = flat[start : start + config.fingerprint_chunk_elements]
        positions = torch.arange(
            start,
            start + int(segment.numel()),
            dtype=torch.int64,
            device=flat.device,
        ).reshape(-1, 1)
        mixed = positions + 1 + (seed % 1_000_003) + 104_729 * (lane_ids + 1)
        mixed = mixed ^ (mixed << 13)
        mixed = mixed ^ (mixed >> 7)
        mixed = mixed ^ (mixed << 17)
        coefficients = mixed.bitwise_and(1).to(torch.float64).mul_(2.0).sub_(1.0)
        projections.add_(
            coefficients.T @ segment.to(dtype=torch.float64), alpha=1.0
        )
    destination[:_FINGERPRINT_STATS].copy_(stats)
    destination[_FINGERPRINT_STATS:].copy_(projections.to(torch.float32))


def _fingerprint_digest(
    values: Tensor,
    *,
    inputs_shape: tuple[int, int],
    delta_shape: tuple[int, int],
    config: GUDOutputJacobianReplayConfig,
) -> str:
    width = config.fingerprint_values_per_tensor

    def part(offset: int, shape: tuple[int, int]) -> dict[str, Any]:
        row = values[offset : offset + width]
        return {
            "shape": list(shape),
            "finite": bool(row[0] == 1.0),
            "sum": float(row[1]),
            "square_sum": float(row[2]),
            "abs_sum": float(row[3]),
            "minimum": float(row[4]),
            "maximum": float(row[5]),
            "order_sensitive_projection_lanes": [
                float(item) for item in row[_FINGERPRINT_STATS:].tolist()
            ],
        }

    return _canonical_sha256(
        {
            "kind": "separable-chunked-full-element-order-sensitive-projection-v1",
            "collision_free": False,
            "binds_every_element": True,
            "inputs": part(0, inputs_shape),
            "delta": part(width, delta_shape),
        }
    )


class _GUDReplayCollector:
    """Separate live projection hooks with bounded pass-one/pass-two state."""

    def __init__(
        self,
        *,
        transaction: Any,
        geometry: GUDStageGeometry,
        sketches: Mapping[tuple[str, int], TwoPassOutputJacobianSketch],
        config: GUDOutputJacobianReplayConfig,
    ) -> None:
        self.transaction = transaction
        self.geometry = geometry
        self.sketches = dict(sketches)
        self.config = config
        self.phase: Literal["pass1", "pass2"] = "pass1"
        self.current: _LossContext | None = None
        self.handles: list[Any] = []
        self.wrapper_count = geometry.logical_layers * len(GUD_ROLE_ORDER)
        self.observations: dict[tuple[str, int], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        self._buffer: Tensor | None = None
        self._seen = bytearray()
        self._contexts: list[_LossContext | None] = []
        self._rows: list[int] = []
        self._expected_blocks = 0
        self._pass_authenticators: dict[str, str] = {}
        self._identity_chains: dict[tuple[str, int, int], str] = {}
        self._callback_count = 0
        self._host_transfer_count = 0
        handle = transaction.swiglu_handle
        first = handle.mlp_modules[0]
        first_wrapper = getattr(first, handle.report.projection_names["gate"])
        self.device = first_wrapper.weight.device
        half = config.balanced_envelope_width // 2
        self._activation_sketches = {
            (role, layer): torch.zeros(
                (half, geometry.role_spec(role).input_width),
                dtype=torch.float32,
                device=self.device,
            )
            for layer in range(geometry.logical_layers)
            for role in GUD_ROLE_ORDER
        }
        self._sensitivity_sketches = {
            key: torch.zeros_like(value)
            for key, value in self._activation_sketches.items()
        }
        self.envelopes_cpu: dict[tuple[str, int], Tensor] = {}
        self.envelopes_device: dict[tuple[str, int], Tensor] = {}
        self.reduced_activations: dict[tuple[str, int], Tensor] = {}
        self.reduced_sensitivities: dict[tuple[str, int], Tensor] = {}

    def install(self) -> None:
        handle = self.transaction.swiglu_handle
        expected = 0
        for layer in range(self.geometry.logical_layers):
            if layer not in handle.mlp_modules:
                raise GUDOutputJacobianReplayTopologyError(
                    "A required live MLP module is missing."
                )
            mlp = handle.mlp_modules[layer]
            for role in GUD_ROLE_ORDER:
                wrapper = getattr(mlp, handle.report.projection_names[role], None)
                if (
                    not isinstance(wrapper, DenseSwiGLUResidualProjection)
                    or wrapper.layer_index != layer
                    or wrapper.role != role
                    or wrapper.guided_registry is not None
                ):
                    self.remove()
                    raise GUDOutputJacobianReplayTopologyError(
                        "A required separate live Gate/Up/Down wrapper is missing "
                        "or already has a guided GUD tranche."
                    )
                self.handles.append(wrapper.register_forward_hook(self._forward_hook))
                expected += 1
        if len(self.handles) != expected or expected != self.wrapper_count:
            self.remove()
            raise GUDOutputJacobianReplayTopologyError(
                "The collector did not install one hook per GUD projection."
            )

    def remove(self) -> None:
        while self.handles:
            self.handles.pop().remove()
        self._buffer = None
        self._seen = bytearray()
        self._contexts = []
        self._rows = []

    @property
    def active_hook_count(self) -> int:
        return len(self.handles)

    def begin_pass(self, phase: Literal["pass1", "pass2"], block_count: int) -> None:
        if phase not in {"pass1", "pass2"} or self._buffer is not None:
            raise GUDOutputJacobianReplayMismatchError(
                "A replay pass is malformed or already active."
            )
        blocks = _positive_int(block_count, "block_count")
        if phase == "pass2" and not self.envelopes_device:
            raise GUDOutputJacobianReplayMismatchError(
                "Pass two cannot start before pass-one envelopes are fixed."
            )
        self.phase = phase
        self._expected_blocks = blocks
        self._buffer = torch.zeros(
            (
                blocks,
                self.wrapper_count,
                self.config.fingerprint_values_per_wrapper,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        slots = blocks * self.wrapper_count
        self._seen = bytearray(slots)
        self._contexts = [None] * blocks
        self._rows = [0] * slots
        if phase == "pass2":
            self.reduced_activations = {
                key: torch.zeros(
                    (
                        self.config.balanced_stream_rows,
                        int(envelope.shape[0]),
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                for key, envelope in self.envelopes_device.items()
            }
            self.reduced_sensitivities = {
                key: torch.zeros_like(value)
                for key, value in self.reduced_activations.items()
            }

    def prepare_context(self, context: _LossContext) -> None:
        if (
            self._buffer is None
            or context.block_ordinal < 0
            or context.block_ordinal >= self._expected_blocks
            or context.prefix_length <= 0
            or context.prefix_length > self.config.max_callback_rows
        ):
            raise GUDOutputJacobianReplayMismatchError(
                "Loss context is outside the active bounded replay pass."
            )

    def _wrapper_index(self, layer: int, role: str) -> int:
        return layer * len(GUD_ROLE_ORDER) + GUD_ROLE_ORDER.index(role)  # type: ignore[arg-type]

    def _record_fingerprint(
        self,
        context: _LossContext,
        *,
        layer: int,
        role: str,
        inputs: Tensor,
        delta: Tensor,
    ) -> None:
        buffer = self._buffer
        if buffer is None:
            raise GUDOutputJacobianReplayMismatchError(
                "A GUD callback occurred outside an active pass."
            )
        wrapper_index = self._wrapper_index(layer, role)
        linear = context.block_ordinal * self.wrapper_count + wrapper_index
        if self._seen[linear]:
            raise GUDOutputJacobianReplayMismatchError(
                "One GUD projection was observed twice in one loss block."
            )
        destination = buffer[context.block_ordinal, wrapper_index]
        width = self.config.fingerprint_values_per_tensor
        _write_tensor_fingerprint(
            destination[:width],
            inputs,
            seed=_derived_seed(self.config.seed, layer, role, "inputs"),
            config=self.config,
        )
        _write_tensor_fingerprint(
            destination[width:],
            delta,
            seed=_derived_seed(self.config.seed, layer, role, "delta"),
            config=self.config,
        )
        previous = self._contexts[context.block_ordinal]
        if previous is None:
            self._contexts[context.block_ordinal] = context
        elif previous != context:
            raise GUDOutputJacobianReplayMismatchError(
                "GUD projections disagreed on one loss context."
            )
        self._seen[linear] = 1
        self._rows[linear] = context.prefix_length
        self._callback_count += 1

    def _stream_rows(
        self,
        destination: Tensor,
        rows: Tensor,
        *,
        context: _LossContext,
        layer: int,
        role: str,
        purpose: str,
    ) -> None:
        mixing = _mixing_matrix(
            int(rows.shape[0]),
            int(destination.shape[0]),
            seed=_derived_seed(
                self.config.seed,
                context.block_ordinal,
                layer,
                role,
                purpose,
            ),
            device=rows.device,
        )
        destination.add_(mixing.T @ rows)

    def _forward_hook(
        self, module: nn.Module, arguments: tuple[Any, ...], output: Any
    ) -> None:
        context = self.current
        if (
            context is None
            or not isinstance(module, DenseSwiGLUResidualProjection)
            or len(arguments) != 1
            or not isinstance(arguments[0], Tensor)
            or not isinstance(output, Tensor)
            or not output.requires_grad
        ):
            raise GUDOutputJacobianReplayTopologyError(
                "A live GUD wrapper did not expose one authenticated tensor boundary."
            )
        inputs = arguments[0].detach()
        layer = module.layer_index
        role = module.role

        def output_gradient_hook(delta: Tensor) -> Tensor:
            flat_inputs = inputs.to(torch.float32).reshape(-1, int(inputs.shape[-1]))
            flat_delta = delta.detach().to(torch.float32).reshape(
                -1, int(delta.shape[-1])
            )
            if (
                int(flat_inputs.shape[0]) != context.prefix_length
                or int(flat_delta.shape[0]) != context.prefix_length
            ):
                raise GUDOutputJacobianReplayTopologyError(
                    "A hooked GUD prefix shape changed."
                )
            self._record_fingerprint(
                context,
                layer=layer,
                role=role,
                inputs=flat_inputs,
                delta=flat_delta,
            )
            row = module.registry._row(layer)
            weight = module.base_projection.weight.detach()
            left = getattr(module.registry, f"{role}_left")[row].detach()
            right = getattr(module.registry, f"{role}_right")[row].detach()
            native_delta = delta.detach().reshape(-1, int(delta.shape[-1]))
            if (
                native_delta.dtype != weight.dtype
                or inputs.dtype != weight.dtype
                or flat_delta.dtype != torch.float32
                or left.dtype != torch.float32
                or right.dtype != torch.float32
            ):
                raise GUDOutputJacobianReplayTopologyError(
                    "The live GUD VJP is not the declared native-base plus FP32 "
                    "residual-factor path."
                )
            input_vjp = role_local_mixed_precision_input_vjp_prevalidated(
                native_delta,
                flat_delta,
                weight,
                left,
                right,
            )
            key = (role, layer)
            sketch = self.sketches[key]
            if self.phase == "pass1":
                sketch.add_prevalidated_pass1(flat_delta, flat_inputs)
                self._stream_rows(
                    self._activation_sketches[key],
                    flat_inputs,
                    context=context,
                    layer=layer,
                    role=role,
                    purpose="pass1-activation",
                )
                self._stream_rows(
                    self._sensitivity_sketches[key],
                    input_vjp,
                    context=context,
                    layer=layer,
                    role=role,
                    purpose="pass1-sensitivity",
                )
                self.observations[key][0] += 1
            else:
                sketch.add_prevalidated_pass2(flat_delta, flat_inputs)
                envelope = self.envelopes_device[key]
                reduced_inputs = flat_inputs @ envelope.T
                reduced_vjp = input_vjp @ envelope.T
                self._stream_rows(
                    self.reduced_activations[key],
                    reduced_inputs,
                    context=context,
                    layer=layer,
                    role=role,
                    purpose="pass2-activation",
                )
                self._stream_rows(
                    self.reduced_sensitivities[key],
                    reduced_vjp,
                    context=context,
                    layer=layer,
                    role=role,
                    purpose="pass2-sensitivity",
                )
                self.observations[key][1] += 1
            return delta

        output.register_hook(output_gradient_hook)

    def finish_pass(self) -> str:
        buffer = self._buffer
        if buffer is None or any(value != 1 for value in self._seen) or any(
            context is None for context in self._contexts
        ):
            raise GUDOutputJacobianReplayMismatchError(
                "The replay pass did not observe every separate GUD projection once."
            )
        staged = buffer.detach().to(device="cpu").contiguous()
        self._host_transfer_count += 1
        try:
            if not bool(torch.isfinite(staged).all()):
                raise GUDOutputJacobianReplayMismatchError(
                    "A GUD fingerprint is non-finite."
                )
            numeric: list[list[str]] = []
            overall = hashlib.sha256(
                b"neural-forge:gud-separable-chunked-replay-authenticator-v1"
            )
            for ordinal in range(self._expected_blocks):
                context = self._contexts[ordinal]
                assert context is not None
                overall.update(_canonical_json(asdict(context)))
                block_hashes = []
                for wrapper_index in range(self.wrapper_count):
                    layer = wrapper_index // len(GUD_ROLE_ORDER)
                    role = GUD_ROLE_ORDER[wrapper_index % len(GUD_ROLE_ORDER)]
                    role_spec = self.geometry.role_spec(role)
                    rows = self._rows[ordinal * self.wrapper_count + wrapper_index]
                    digest = _fingerprint_digest(
                        staged[ordinal, wrapper_index],
                        inputs_shape=(rows, role_spec.input_width),
                        delta_shape=(rows, role_spec.output_width),
                        config=self.config,
                    )
                    block_hashes.append(digest)
                    overall.update(bytes.fromhex(digest))
                numeric.append(block_hashes)
            pass_number = 1 if self.phase == "pass1" else 2
            for layer in range(self.geometry.logical_layers):
                for role in GUD_ROLE_ORDER:
                    wrapper_index = self._wrapper_index(layer, role)
                    chain = hashlib.sha256(_IDENTITY_CHAIN_HEADER)
                    for ordinal, context in enumerate(self._contexts):
                        assert context is not None
                        identity = OutputJacobianBlockIdentity(
                            semantic_identity=context.semantic_identity(
                                layer=layer,
                                role=role,
                                group=0,
                                site=layer,
                            ),
                            numeric_fingerprint_sha256=numeric[ordinal][wrapper_index],
                        )
                        chain.update(bytes.fromhex(identity.sha256()))
                    chain_sha = chain.hexdigest()
                    self.sketches[(role, layer)].bind_prevalidated_identity_chain(
                        pass_number, chain_sha
                    )
                    self._identity_chains[(role, layer, pass_number)] = chain_sha
            digest = overall.hexdigest()
            self._pass_authenticators[self.phase] = digest
            return digest
        finally:
            self._buffer = None
            self._seen = bytearray()
            self._contexts = []
            self._rows = []
            self._expected_blocks = 0

    def finalize_pass1_envelopes(self) -> None:
        if self._buffer is not None or self.envelopes_cpu:
            raise GUDOutputJacobianReplayMismatchError(
                "Pass-one envelopes were finalized in the wrong phase."
            )
        for key in sorted(self._activation_sketches):
            role, _layer = key
            registry = self.transaction.swiglu_handle.registry
            row = registry._row(key[1])
            right = getattr(registry, f"{role}_right")[row]
            envelope = finalize_streaming_gud_envelope(
                self._activation_sketches.pop(key),
                self._sensitivity_sketches.pop(key),
                right,
                self.config.basis_config(),
            )
            self.envelopes_cpu[key] = envelope
            self.envelopes_device[key] = envelope.to(
                device=self.device, dtype=torch.float32
            )

    def release_finalization_buffers(self) -> None:
        """Drop replay-derived tensors after all 90 site reports are complete."""

        if self._buffer is not None or self.active_hook_count:
            raise GUDOutputJacobianReplayMismatchError(
                "Cannot release collector tensors during an active replay."
            )
        self._activation_sketches.clear()
        self._sensitivity_sketches.clear()
        self.envelopes_cpu.clear()
        self.envelopes_device.clear()
        self.reduced_activations.clear()
        self.reduced_sensitivities.clear()

    def site_source_manifest(
        self,
        role: str,
        layer: int,
        *,
        pass1_loss_sha256: str,
        pass2_loss_sha256: str,
    ) -> dict[str, Any]:
        return {
            "schema": f"{GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA}:site-source-v1",
            "layer": layer,
            "role": role,
            "pass1_identity_chain_sha256": self._identity_chains[(role, layer, 1)],
            "pass2_identity_chain_sha256": self._identity_chains[(role, layer, 2)],
            "pass1_replay_authenticator_sha256": self._pass_authenticators["pass1"],
            "pass2_replay_authenticator_sha256": self._pass_authenticators["pass2"],
            "pass1_loss_replay_sha256": pass1_loss_sha256,
            "pass2_loss_replay_sha256": pass2_loss_sha256,
            "observation_counts": list(self.observations[(role, layer)]),
            "real_projection_output_gradient_hook": True,
            "role_local_projection_input": True,
            "role_local_input_vjp_precision": (
                "base=(D_native@W_native); inherited=((D_fp32@L_fp32)@R_fp32); "
                "sum_native=base+cast_native(inherited); sensitivity=cast_fp32("
                "sum_native)"
            ),
            "shared_mlp_input_hook_used": False,
            "full_width_replay_table_stored": False,
        }

    @property
    def instrumentation(self) -> dict[str, Any]:
        return {
            "gradient_callback_count": self._callback_count,
            "pass_boundary_host_transfer_count": self._host_transfer_count,
            "active_hook_count": self.active_hook_count,
            "full_lanes_rows_width_coefficient_table_bytes": 0,
            "full_all_sites_replay_tensor_table_bytes": 0,
            "fingerprint_values_per_wrapper": (
                self.config.fingerprint_values_per_wrapper
            ),
            "gate_up_separate_wrapper_indices": True,
        }


def _validate_production_topology(
    transaction: ExactTargetCompilerTransaction,
) -> GUDStageGeometry:
    if not isinstance(transaction, ExactTargetCompilerTransaction):
        raise GUDOutputJacobianReplayTopologyError(
            "transaction must be ExactTargetCompilerTransaction."
        )
    transaction.assert_live()
    handle = transaction.swiglu_handle
    report = handle.report
    geometry = GUDStageGeometry(
        logical_layers=len(report.layer_indices),
        hidden_size=report.hidden_size,
        intermediate_size=report.intermediate_size,
    )
    validate_qwen_30_layer_stage_geometry(geometry)
    if (
        tuple(report.layer_indices) != tuple(range(30))
        or report.rank != 8
        or handle.registry.rank != 8
        or int(handle.registry.layer_indices.numel()) != 30
    ):
        raise GUDOutputJacobianReplayTopologyError(
            "The production audit requires all 30 inherited rank-eight GUD sites."
        )
    return geometry


def _registry_state_sha256(registry: nn.Module) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "shape": list(value.shape),
                "sha256": _tensor_sha256(value),
            }
            for name, value in registry.named_parameters(recurse=False)
        ]
    )


def _build_sketches(
    transaction: Any,
    geometry: GUDStageGeometry,
    config: GUDOutputJacobianReplayConfig,
) -> dict[tuple[str, int], TwoPassOutputJacobianSketch]:
    registry = transaction.swiglu_handle.registry
    result = {}
    for layer in range(geometry.logical_layers):
        row = registry._row(layer)
        for role in GUD_ROLE_ORDER:
            role_spec = geometry.role_spec(role)
            left = getattr(registry, f"{role}_left")[row]
            right = getattr(registry, f"{role}_right")[row]
            result[(role, layer)] = TwoPassOutputJacobianSketch(
                OutputJacobianSketchSpec(
                    out_features=role_spec.output_width,
                    in_features=role_spec.input_width,
                    rank=config.rank,
                    sketch_width=config.sketch_width,
                    seed=_derived_seed(config.seed, layer, role, "output-jacobian"),
                    accumulator_device=str(left.device),
                    max_bytes=config.max_bytes,
                    min_normal_fraction=config.min_normal_fraction,
                    min_guided_over_random_ratio=(
                        config.min_guided_over_random_ratio
                    ),
                ),
                left,
                right,
                accumulator_device=left.device,
            )
    return result


def _behavior_replay_report_fields(
    basis_domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    plans: Any,
    *,
    replay_blocks: int,
    pass1_loss_sha256: str,
    pass2_loss_sha256: str,
    pass1_authenticator_sha256: str,
    pass2_authenticator_sha256: str,
) -> dict[str, Any]:
    replay_plan_manifest = [
        {
            "prompt_id": plan.prompt.prompt_id,
            "prompt_token_ids_sha256": plan.prompt.token_ids_sha256,
            "parent_tokens": list(plan.parent_tokens),
            "child_tokens": list(plan.child_tokens),
            "position_weights": list(plan.position_weights),
            "child_indices": list(plan.child_indices),
        }
        for plan in plans
    ]
    basis_domain_identity = basis_domain.as_dict()
    schedule_identity = schedule.as_dict()
    replay_plan_sha256 = _canonical_sha256(replay_plan_manifest)
    schedule_sha256 = _canonical_sha256(schedule_identity)
    behavior_replay_identity = {
        "basis_domain_sha256": basis_domain.domain_sha256,
        "basis_artifact_sha256": basis_domain.artifact_sha256,
        "schedule_sha256": schedule_sha256,
        "replay_plan_sha256": replay_plan_sha256,
        "replay_block_count": replay_blocks,
        "pass1_loss_replay_sha256": pass1_loss_sha256,
        "pass2_loss_replay_sha256": pass2_loss_sha256,
        "pass1_full_element_replay_authenticator_sha256": (
            pass1_authenticator_sha256
        ),
        "pass2_full_element_replay_authenticator_sha256": (
            pass2_authenticator_sha256
        ),
    }
    return {
        "basis_domain": basis_domain_identity,
        "basis_domain_sha256": basis_domain.domain_sha256,
        "basis_artifact_sha256": basis_domain.artifact_sha256,
        "behavior_schedule": schedule_identity,
        "behavior_schedule_sha256": schedule_sha256,
        "replay_plan_manifest": replay_plan_manifest,
        "replay_plan_sha256": replay_plan_sha256,
        "behavior_replay_identity": behavior_replay_identity,
        "behavior_replay_identity_sha256": _canonical_sha256(
            behavior_replay_identity
        ),
    }


def _ordered_site_report_evidence(
    site_reports: Mapping[tuple[str, int], GUDSiteAuditReport],
    geometry: GUDStageGeometry,
) -> tuple[tuple[GUDSiteAuditReport, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {
        (role, layer)
        for layer in range(geometry.logical_layers)
        for role in GUD_ROLE_ORDER
    }
    if set(site_reports) != expected:
        raise GUDOutputJacobianReplayMismatchError(
            "The completed site-report map is not the exact GUD topology."
        )
    ordered = tuple(
        site_reports[(role, layer)]
        for layer in range(geometry.logical_layers)
        for role in GUD_ROLE_ORDER
    )
    payloads: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for layer in range(geometry.logical_layers):
        for role in GUD_ROLE_ORDER:
            report = site_reports[(role, layer)]
            payload = report.to_dict()
            declared_sha = payload.pop("report_sha256", None)
            if (
                payload.get("layer_index") != layer
                or payload.get("role") != role
                or declared_sha != report.report_sha256
                or declared_sha != _canonical_sha256(payload)
            ):
                raise GUDOutputJacobianReplayMismatchError(
                    "A completed GUD site report failed self-authentication."
                )
            payload["report_sha256"] = declared_sha
            dependency = payload.get("output_jacobian_dependency_report")
            if not isinstance(dependency, Mapping) or any(
                name not in dependency for name in _SITE_DIAGNOSTIC_FIELDS
            ):
                raise GUDOutputJacobianReplayMismatchError(
                    "A GUD site report lacks output-Jacobian diagnostics."
                )
            dependency_payload = dict(dependency)
            dependency_sha = dependency_payload.pop("report_sha256", None)
            if dependency_sha != _canonical_sha256(dependency_payload):
                raise GUDOutputJacobianReplayMismatchError(
                    "A site's output-Jacobian dependency failed authentication."
                )
            payloads.append(payload)
            diagnostics.append(
                {
                    "layer": layer,
                    "role": role,
                    "site_report_sha256": report.report_sha256,
                    "output_jacobian_report_sha256": dependency[
                        "report_sha256"
                    ],
                    **{
                        name: dependency[name]
                        for name in _SITE_DIAGNOSTIC_FIELDS
                    },
                }
            )
    return ordered, payloads, diagnostics


def run_gud_output_jacobian_replay_audit(
    *,
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    source_guided_qkvo_binding: GuidedFactorSourceBinding,
    parent_model: nn.Module,
    basis_domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    config: GUDOutputJacobianReplayConfig | None = None,
) -> GUDOutputJacobianReplayResult:
    """Run two real behavior-loss passes and return four complete GUD packs."""

    selected = config or GUDOutputJacobianReplayConfig()
    if not isinstance(selected, GUDOutputJacobianReplayConfig):
        raise GUDOutputJacobianReplayError("config has the wrong type.")
    if (
        selected.prompt_count != 24
        or selected.horizon != 8
        or selected.rank != 8
        or selected.sketch_width < 8
    ):
        raise GUDOutputJacobianReplayError(
            "Production replay requires 24 prompts, horizon 8, and rank 8."
        )
    geometry = _validate_production_topology(transaction)
    _validate_exact_topology(transaction)
    if not isinstance(parent_model, nn.Module) or parent_model is transaction.model:
        raise GUDOutputJacobianReplayTopologyError(
            "parent_model must be a distinct live teacher module."
        )
    qkvo_config = QKVOOutputJacobianAuditConfig(
        prompt_count=selected.prompt_count,
        horizon=selected.horizon,
        rank=selected.rank,
        sketch_width=selected.sketch_width,
        seed=selected.seed,
        min_normal_fraction=max(0.20, selected.min_normal_fraction),
        min_guided_over_random_ratio=max(
            1.50, selected.min_guided_over_random_ratio
        ),
        max_bytes=selected.max_bytes,
    )
    _validate_domain_and_schedule(basis_domain, schedule, qkvo_config)
    source_v3_before = _capture_live_v3_provenance(
        transaction,
        guided_tranche,
        source_guided_qkvo_binding,
    )
    factor_entries = _trainer._factor_parameters(
        transaction,
        guided_tranche=guided_tranche,
    )
    if any(value.grad is not None for _name, value in factor_entries):
        raise GUDOutputJacobianReplayMismatchError(
            "The live v3 source must enter the audit without residual gradients."
        )
    inherited_qkvo_installation = transaction.qkvo_handle.report.as_dict()
    inherited_gud_installation = transaction.swiglu_handle.report.as_dict()
    guided_qkvo_installation = guided_tranche.report.as_dict()
    if source_guided_qkvo_binding.inherited_installation_sha256 != (
        inherited_qkvo_installation.get("installation_sha256")
    ):
        raise GUDOutputJacobianReplayTopologyError(
            "The v3 source binding names a different inherited QKVO installation."
        )
    lexical_handle = getattr(transaction, "lexical_handle", None)
    lexical_installation = (
        None if lexical_handle is None else lexical_handle.report.as_dict()
    )
    installation_identity = {
        "inherited_qkvo_installation_sha256": inherited_qkvo_installation[
            "installation_sha256"
        ],
        "inherited_gud_installation_sha256": inherited_gud_installation[
            "installation_sha256"
        ],
        "guided_qkvo_installation_sha256": guided_qkvo_installation[
            "installation_sha256"
        ],
        "lexical_installation_sha256": (
            None
            if lexical_installation is None
            else lexical_installation["installation_sha256"]
        ),
        "transaction_identity": id(transaction),
        "inherited_qkvo_handle_identity": id(transaction.qkvo_handle),
        "inherited_qkvo_registry_identity": id(
            transaction.qkvo_handle.adapter_registry
        ),
        "inherited_gud_handle_identity": id(transaction.swiglu_handle),
        "inherited_gud_registry_identity": id(transaction.swiglu_handle.registry),
        "guided_qkvo_handle_identity": id(guided_tranche),
        "guided_qkvo_registry_identity": id(guided_tranche.guided_registry),
        "guided_qkvo_training_stage": guided_tranche.training_stage,
    }
    installation_identity["runtime_installation_identity_sha256"] = (
        _canonical_sha256(installation_identity)
    )
    plans = _build_replay_plans(
        transaction, parent_model, basis_domain, schedule, qkvo_config
    )
    replay_blocks = _replay_block_count(plans)
    max_rows = max(
        len(plan.prompt.token_ids) + selected.horizon for plan in plans
    )
    memory_plan = plan_gud_real_replay_memory(
        geometry,
        replay_blocks=replay_blocks,
        max_rows=max_rows,
        config=selected,
    )
    memory_plan = _bind_source_v3_capture_memory(
        memory_plan,
        source_v3_before,
        selected,
    )
    registry = transaction.swiglu_handle.registry
    inherited_before = _registry_state_sha256(registry)
    versions_before = tuple(
        (name, int(value._version))
        for name, value in registry.named_parameters(recurse=False)
    )
    sketches = _build_sketches(transaction, geometry, selected)
    collector = _GUDReplayCollector(
        transaction=transaction,
        geometry=geometry,
        sketches=sketches,
        config=selected,
    )
    parent_modes = tuple((module, bool(module.training)) for module in parent_model.modules())
    pass1_loss_sha = ""
    pass2_loss_sha = ""
    pass1_auth = ""
    pass2_auth = ""
    try:
        parent_model.eval()
        with transaction.output_training_scope() as factors:
            collector.install()
            try:
                collector.begin_pass("pass1", replay_blocks)
                pass1_loss_sha, pass1_blocks, _ = _replay_loss_pass(
                    transaction=transaction,
                    parent_model=parent_model,
                    plans=plans,
                    schedule=schedule,
                    collector=collector,
                    factors=factors,
                )
                pass1_auth = collector.finish_pass()
                if pass1_blocks != replay_blocks:
                    raise GUDOutputJacobianReplayMismatchError(
                        "Pass one did not execute every replay block."
                    )
                for sketch in sketches.values():
                    sketch.finish_pass1()
                collector.finalize_pass1_envelopes()
                collector.begin_pass("pass2", replay_blocks)
                pass2_loss_sha, pass2_blocks, _ = _replay_loss_pass(
                    transaction=transaction,
                    parent_model=parent_model,
                    plans=plans,
                    schedule=schedule,
                    collector=collector,
                    factors=factors,
                )
                pass2_auth = collector.finish_pass()
                if (
                    pass2_blocks != replay_blocks
                    or pass2_loss_sha != pass1_loss_sha
                    or pass2_auth != pass1_auth
                ):
                    raise GUDOutputJacobianReplayMismatchError(
                        "The two real behavior-loss passes did not replay identically."
                    )
            finally:
                collector.remove()
                for factor in factors:
                    factor.grad = None
    finally:
        collector.remove()
        for module, training in parent_modes:
            module.training = training
    expected_observations = {
        (role, layer)
        for layer in range(geometry.logical_layers)
        for role in GUD_ROLE_ORDER
    }
    if set(collector.observations) != expected_observations or any(
        counts != [replay_blocks, replay_blocks]
        for counts in collector.observations.values()
    ):
        raise GUDOutputJacobianReplayMismatchError(
            "Not all 90 GUD sites were observed exactly once per replay block/pass."
        )
    pack_storage: dict[str, dict[str, Tensor]] = {
        candidate: {
            role: torch.empty(
                (
                    geometry.logical_layers,
                    selected.rank,
                    geometry.role_spec(role).input_width,
                ),
                dtype=torch.float32,
                device="cpu",
            )
            for role in GUD_ROLE_ORDER
        }
        for candidate in _CANDIDATES
    }
    site_reports: dict[tuple[str, int], GUDSiteAuditReport] = {}
    basis_config = selected.basis_config()
    for layer in range(geometry.logical_layers):
        row = registry._row(layer)
        for role in GUD_ROLE_ORDER:
            key = (role, layer)
            site_result = finalize_streamed_gud_site_basis(
                layer_index=layer,
                role_spec=geometry.role_spec(role),
                inherited_right=getattr(registry, f"{role}_right")[row],
                output_jacobian_result=sketches.pop(key).finalize(),
                balanced_envelope=collector.envelopes_cpu[key],
                reduced_activation_rows=collector.reduced_activations[key],
                reduced_sensitivity_rows=collector.reduced_sensitivities[key],
                config=basis_config,
                source_manifest=collector.site_source_manifest(
                    role,
                    layer,
                    pass1_loss_sha256=pass1_loss_sha,
                    pass2_loss_sha256=pass2_loss_sha,
                ),
                memory_plan_sha256=memory_plan["memory_plan_sha256"],
            )
            site_reports[key] = site_result.report
            pack_storage["output-jacobian"][role][layer].copy_(
                site_result.output_jacobian_right_rows
            )
            pack_storage["balanced"][role][layer].copy_(
                site_result.balanced_right_rows
            )
            pack_storage["procrustes-hybrid"][role][layer].copy_(
                site_result.procrustes_hybrid_right_rows
            )
            pack_storage["random-control"][role][layer].copy_(
                site_result.random_control_right_rows
            )
            del site_result
    (
        ordered_site_reports,
        ordered_site_report_payloads,
        site_diagnostics,
    ) = _ordered_site_report_evidence(site_reports, geometry)
    collector_instrumentation = collector.instrumentation
    collector_hook_count = collector.active_hook_count
    collector.release_finalization_buffers()
    packs = {
        candidate: MappingProxyType(pack_storage[candidate])
        for candidate in _CANDIDATES
    }
    expected_shapes = {
        "gate": (30, 8, 1536),
        "up": (30, 8, 1536),
        "down": (30, 8, 5120),
    }
    pack_manifest = []
    for candidate in _CANDIDATES:
        for role in GUD_ROLE_ORDER:
            value = packs[candidate][role]
            if tuple(value.shape) != expected_shapes[role]:
                raise GUDOutputJacobianReplayTopologyError(
                    "A completed GUD basis pack has the wrong exact shape."
                )
            pack_manifest.append(
                {
                    "candidate": candidate,
                    "role": role,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": _tensor_sha256(value),
                }
            )
    source_v3_after = _capture_live_v3_provenance(
        transaction,
        guided_tranche,
        source_guided_qkvo_binding,
    )
    source_v3_state_unchanged = bool(source_v3_after == source_v3_before)
    inherited_after = _registry_state_sha256(registry)
    versions_after = tuple(
        (name, int(value._version))
        for name, value in registry.named_parameters(recurse=False)
    )
    transaction.assert_live()
    gradients_cleared = all(value.grad is None for _name, value in factor_entries)
    if (
        inherited_after != inherited_before
        or versions_after != versions_before
        or not source_v3_state_unchanged
        or not gradients_cleared
        or collector_hook_count != 0
    ):
        raise GUDOutputJacobianReplayMismatchError(
            "Replay mutated the live v3 source or leaked hooks/gradients."
        )
    site_manifest = [
        {
            "layer": layer,
            "role": role,
            "report_sha256": site_reports[(role, layer)].report_sha256,
        }
        for layer in range(30)
        for role in GUD_ROLE_ORDER
    ]
    behavior_identity_fields = _behavior_replay_report_fields(
        basis_domain,
        schedule,
        plans,
        replay_blocks=replay_blocks,
        pass1_loss_sha256=pass1_loss_sha,
        pass2_loss_sha256=pass2_loss_sha,
        pass1_authenticator_sha256=pass1_auth,
        pass2_authenticator_sha256=pass2_auth,
    )
    source_v3_fields = _source_v3_report_fields(
        source_v3_before,
        source_v3_after,
        source_guided_qkvo_binding,
    )
    report_payload: dict[str, Any] = {
        "schema": GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA,
        "basis_pack_schema": GUD_COMPLETE_BASIS_PACK_SCHEMA,
        "config": selected.to_dict(),
        "config_sha256": selected.sha256(),
        "geometry": geometry.to_dict(),
        **behavior_identity_fields,
        "pass1_loss_replay_sha256": pass1_loss_sha,
        "pass2_loss_replay_sha256": pass2_loss_sha,
        "pass1_full_element_replay_authenticator_sha256": pass1_auth,
        "pass2_full_element_replay_authenticator_sha256": pass2_auth,
        "two_pass_replay_identical": True,
        "replay_block_count": replay_blocks,
        "site_count": 90,
        "separate_projection_hook_count": 90,
        "gate_up_separate_hooks": True,
        "full_downstream_loss_output_delta_captured": True,
        "role_local_projection_input_captured": True,
        "role_local_input_vjp_formula": (
            "cast_fp32((D_native@W_native) + cast_native((D_fp32@L_fp32)"
            "@R_fp32))"
        ),
        "native_base_weights_preserved_without_fp32_full_weight_clones": True,
        "shared_mlp_input_hook_used": False,
        "full_all_sites_replay_tensor_table_stored": False,
        "full_lanes_rows_width_coefficient_table_stored": False,
        "memory_plan": memory_plan,
        "memory_plan_sha256": memory_plan["memory_plan_sha256"],
        "instrumentation": collector_instrumentation,
        "runtime_installation_identity": installation_identity,
        "runtime_installation_identity_sha256": installation_identity[
            "runtime_installation_identity_sha256"
        ],
        "inherited_qkvo_installation_sha256": inherited_qkvo_installation[
            "installation_sha256"
        ],
        "inherited_gud_installation_sha256": inherited_gud_installation[
            "installation_sha256"
        ],
        "guided_qkvo_installation_sha256": guided_qkvo_installation[
            "installation_sha256"
        ],
        **source_v3_fields,
        "inherited_factor_state_sha256_before": inherited_before,
        "inherited_factor_state_sha256_after": inherited_after,
        "factor_gradients_cleared": gradients_cleared,
        "hooks_removed": collector_hook_count == 0,
        "optimizer_created": False,
        "internal_weight_matching_used": False,
        "site_manifest": site_manifest,
        "site_manifest_sha256": _canonical_sha256(site_manifest),
        "ordered_site_reports": ordered_site_report_payloads,
        "ordered_site_reports_sha256": _canonical_sha256(
            ordered_site_report_payloads
        ),
        "site_diagnostics": site_diagnostics,
        "site_diagnostics_sha256": _canonical_sha256(site_diagnostics),
        "basis_pack_manifest": pack_manifest,
        "basis_pack_manifest_sha256": _canonical_sha256(pack_manifest),
        "candidate_order": list(_CANDIDATES),
    }
    report_payload["report_sha256"] = _canonical_sha256(report_payload)
    immutable_packs = MappingProxyType(
        {candidate: packs[candidate] for candidate in _CANDIDATES}
    )
    return GUDOutputJacobianReplayResult(
        basis_packs=immutable_packs,
        site_reports=ordered_site_reports,
        report=GUDOutputJacobianReplayReport(
            MappingProxyType(dict(report_payload))
        ),
    )


__all__ = [
    "GUD_COMPLETE_BASIS_PACK_SCHEMA",
    "GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA",
    "GUDOutputJacobianReplayConfig",
    "GUDOutputJacobianReplayError",
    "GUDOutputJacobianReplayLimitError",
    "GUDOutputJacobianReplayMismatchError",
    "GUDOutputJacobianReplayReport",
    "GUDOutputJacobianReplayResult",
    "GUDOutputJacobianReplayTopologyError",
    "plan_gud_real_replay_memory",
    "role_local_mixed_precision_input_vjp_prevalidated",
    "run_gud_output_jacobian_replay_audit",
]
