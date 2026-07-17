"""Cached second-order attention folding into explicit CREVF payloads.

This module is deliberately model-free.  It consumes authenticated tensors
captured before the solve, builds damped second-order statistics, and emits
fully encoded CREVF records.  The source tensors are never mutated and every
successive-rounding candidate is compared with a round-to-nearest rollback.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

from .hf_lfm2_causal_residual_entropy_value_fold import (
    CREVF_RECORD_ENVELOPE_BYTES,
    CrevfExactValue,
    CrevfTensorSpec,
    bf16_bits,
    decode_crevf_tensor,
    encode_crevf_tensor,
    inspect_crevf_tensor,
)


_EVIDENCE_SCHEMA = "neural-forge.lfm2.cached-second-order-evidence.v1"
_ATTENTION_BRIDGE_SCHEMA = "neural-forge.lfm2.cached-attention-bridge.v1"
_STATS_SCHEMA = "neural-forge.lfm2.cached-second-order-statistics.v1"
_ALLOWED_GROUP_SIZES = frozenset({32, 64, 128, 256})
_ALLOWED_BIT_WIDTHS = frozenset(range(4, 9))
_RECORD_PREFIX = struct.Struct("<Q32s")


class Lfm2CachedSecondOrderError(ValueError):
    """Cached evidence, geometry, or a requested solve is invalid."""


@dataclass(frozen=True, slots=True)
class Lfm2CachedSecondOrderEvidence:
    """Sealed CPU float64 tensors captured before the solver is entered."""

    schema: str
    site_id: str
    tensor_role: str
    parent_inputs: Tensor = field(repr=False, compare=False)
    child_inputs: Tensor = field(repr=False, compare=False)
    parent_outputs: Tensor = field(repr=False, compare=False)
    child_outputs: Tensor = field(repr=False, compare=False)
    output_sensitivities: Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    tensor_fingerprints: tuple[tuple[str, str], ...] = ()
    evidence_sha256: str = ""

    @property
    def sample_count(self) -> int:
        return int(self.parent_inputs.shape[0])

    @property
    def input_features(self) -> int:
        return int(self.parent_inputs.shape[1])

    @property
    def output_features(self) -> int:
        return int(self.parent_outputs.shape[1])


@dataclass(frozen=True, slots=True)
class Lfm2SealedAttentionBridge:
    """Fixed child-attention map for exact V-delta propagation into O."""

    schema: str
    site_id: str
    batch_size: int
    query_tokens: int
    key_tokens: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    child_attention_probabilities: Tensor = field(repr=False, compare=False)
    attention_mask: Tensor | None = field(default=None, repr=False, compare=False)
    trace_attention_config_sha256: str = ""
    attention_mask_sha256: str = ""
    tensor_fingerprints: tuple[tuple[str, str], ...] = ()
    bridge_sha256: str = ""

    @property
    def query_hidden_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def value_hidden_size(self) -> int:
        return self.key_value_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class Lfm2SecondOrderStatistics:
    """Damped input/output Hessians and parent-child mismatch covariance."""

    schema: str
    evidence_sha256: str
    parent_hessian: Tensor = field(repr=False, compare=False)
    child_hessian: Tensor = field(repr=False, compare=False)
    h_in: Tensor = field(repr=False, compare=False)
    damped_child_hessian: Tensor = field(repr=False, compare=False)
    mismatch_cross_covariance: Tensor = field(repr=False, compare=False)
    h_out: Tensor | None = field(default=None, repr=False, compare=False)
    input_damping: float = 0.0
    output_damping: float | None = None
    parent_input_blend: float = 0.5
    tensor_fingerprints: tuple[tuple[str, str], ...] = ()
    statistics_sha256: str = ""


@dataclass(frozen=True, slots=True)
class Lfm2SecondOrderOpsAudit:
    """Explicit audit counters for the zero-training solve boundary."""

    cached_tensors_only: bool = True
    parent_model_calls: int = 0
    child_model_calls: int = 0
    forward_calls: int = 0
    generation_calls: int = 0
    gradient_calls: int = 0
    optimizer_steps: int = 0
    training_steps: int = 0
    input_hessian_successive_rounding: bool = True
    output_hessian_row_compensation: bool = False
    fixed_attention_probability_bridge: bool = False
    nonlinear_query_key_coupling: bool = False
    source_rollback_verified: bool = True


@dataclass(frozen=True, slots=True)
class Lfm2SecondOrderQuantizerConfig:
    """One deterministic point in the second-order quantizer grid."""

    bit_width: int
    group_size: int
    clipping_ratio_ppm: int
    rounding_offset_ppm: int
    outliers_per_group: int
    mismatch_coefficient_ppm: int

    @property
    def clipping_ratio(self) -> float:
        return self.clipping_ratio_ppm / 1_000_000

    @property
    def rounding_offset(self) -> float:
        return self.rounding_offset_ppm / 1_000_000

    @property
    def mismatch_coefficient(self) -> float:
        return self.mismatch_coefficient_ppm / 1_000_000

    @property
    def option_suffix(self) -> str:
        return (
            f"b{self.bit_width}-g{self.group_size}"
            f"-c{self.clipping_ratio_ppm}"
            f"-r{self.rounding_offset_ppm:+d}"
            f"-o{self.outliers_per_group}"
            f"-m{self.mismatch_coefficient_ppm:+d}"
        )


@dataclass(frozen=True, slots=True)
class Lfm2CrevfSolvedTensor:
    """One reconstructed tensor and its exact CREVF serialization seam."""

    name: str
    role: str
    shape: tuple[int, ...]
    bit_width: int
    group_size: int
    tensor_blob: bytes = field(repr=False, compare=False)
    group_codes: Tensor = field(repr=False, compare=False)
    scales: Tensor = field(repr=False, compare=False)
    exact_outliers: tuple[CrevfExactValue, ...]
    reconstructed_tensor: Tensor = field(repr=False, compare=False)
    source_parameter_count: int
    dense_bf16_bytes: int
    tensor_blob_bytes: int
    physical_record_bytes: int
    physical_byte_savings: int
    persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    decoded_code_transient_bytes: int
    source_sha256: str
    reconstruction_sha256: str
    tensor_blob_sha256: str
    physical_payload_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2SecondOrderQuantizedOption:
    """One chargeable grid option after safe RTN rollback."""

    option_id: str
    config: Lfm2SecondOrderQuantizerConfig
    payload: Lfm2CrevfSolvedTensor = field(compare=False)
    selected_method: str
    current_objective: float
    full_precision_objective: float
    rtn_objective: float
    raw_successive_objective: float
    selected_objective: float
    improvement_vs_current: float
    improvement_vs_rtn: float
    evidence_sha256: str
    statistics_sha256: str
    source_sha256: str
    config_sha256: str
    rtn_physical_payload_sha256: str
    raw_successive_physical_payload_sha256: str
    option_sha256: str
    ops_audit: Lfm2SecondOrderOpsAudit

    @property
    def physical_record_bytes(self) -> int:
        return self.payload.physical_record_bytes


@dataclass(frozen=True, slots=True)
class Lfm2SecondOrderGridResult:
    """All valid CREVF options and the best objective under an optional cap."""

    evidence_sha256: str
    statistics_sha256: str
    source_sha256: str
    grid_sha256: str
    current_objective: float
    options: tuple[Lfm2SecondOrderQuantizedOption, ...]
    best_option: Lfm2SecondOrderQuantizedOption
    max_tensor_physical_bytes: int | None
    ops_audit: Lfm2SecondOrderOpsAudit


@dataclass(frozen=True, slots=True)
class Lfm2JointAttentionConfig:
    """One coupled V/O attention grid point."""

    quantizer: Lfm2SecondOrderQuantizerConfig
    propagation_coefficient_ppm: int
    value_objective_weight_ppm: int

    @property
    def propagation_coefficient(self) -> float:
        return self.propagation_coefficient_ppm / 1_000_000

    @property
    def value_objective_weight(self) -> float:
        return self.value_objective_weight_ppm / 1_000_000


@dataclass(frozen=True, slots=True)
class Lfm2JointAttentionOption:
    """A physically sealed V/O pair selected as one rollback unit."""

    option_id: str
    config: Lfm2JointAttentionConfig
    value_payload: Lfm2CrevfSolvedTensor = field(compare=False)
    output_payload: Lfm2CrevfSolvedTensor = field(compare=False)
    selected_method: str
    current_joint_objective: float
    rtn_joint_objective: float
    raw_successive_joint_objective: float
    selected_joint_objective: float
    value_selected_objective: float
    output_selected_objective: float
    total_physical_record_bytes: int
    total_dense_bf16_bytes: int
    total_physical_byte_savings: int
    evidence_bundle_sha256: str
    config_sha256: str
    physical_bundle_sha256: str
    option_sha256: str
    ops_audit: Lfm2SecondOrderOpsAudit


@dataclass(frozen=True, slots=True)
class Lfm2JointAttentionResult:
    """All coupled V/O payloads and the best option under an optional cap."""

    evidence_bundle_sha256: str
    source_bundle_sha256: str
    grid_sha256: str
    current_joint_objective: float
    options: tuple[Lfm2JointAttentionOption, ...]
    best_option: Lfm2JointAttentionOption
    max_bundle_physical_bytes: int | None
    ops_audit: Lfm2SecondOrderOpsAudit


@dataclass(frozen=True, slots=True)
class _QuantizedCandidate:
    payload: Lfm2CrevfSolvedTensor
    objective: float


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_canonical(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _tensor_raw_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def _tensor_fingerprint(value: Tensor) -> str:
    header = _canonical_bytes(
        {
            "dtype": str(value.dtype),
            "shape": tuple(int(part) for part in value.shape),
        }
    )
    return _sha256_bytes(header + b"\0" + _tensor_raw_bytes(value))


def _storage_identity(value: Tensor) -> tuple[str, int]:
    if value.device.type == "meta":
        raise Lfm2CachedSecondOrderError("Meta tensors are not cached evidence.")
    return (str(value.device), int(value.untyped_storage().data_ptr()))


def _require_plain_finite_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise Lfm2CachedSecondOrderError(f"{name} must be a Tensor.")
    if isinstance(value, nn.Parameter):
        raise Lfm2CachedSecondOrderError(f"{name} cannot be an nn.Parameter.")
    if value.requires_grad:
        raise Lfm2CachedSecondOrderError(f"{name} cannot require gradients.")
    if value.device.type == "meta" or not value.dtype.is_floating_point:
        raise Lfm2CachedSecondOrderError(
            f"{name} must be a materialized floating-point Tensor."
        )
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise Lfm2CachedSecondOrderError(f"{name} must be nonempty and finite.")
    return value


def _flatten_cached(value: Tensor, name: str) -> Tensor:
    if value.ndim < 2:
        raise Lfm2CachedSecondOrderError(
            f"{name} must have sample axes and one feature axis."
        )
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, int(value.shape[-1]))
        .contiguous()
        .clone()
    )


def _fingerprint_pairs(
    values: Iterable[tuple[str, Tensor | None]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, "none" if value is None else _tensor_fingerprint(value))
        for name, value in values
    )


def _evidence_seal_payload(
    *,
    schema: str,
    site_id: str,
    tensor_role: str,
    fingerprints: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "schema": schema,
        "site_id": site_id,
        "tensor_role": tensor_role,
        "tensor_fingerprints": tuple(fingerprints),
    }


def seal_lfm2_cached_second_order_evidence(
    *,
    site_id: str,
    tensor_role: str,
    parent_inputs: Tensor,
    child_inputs: Tensor,
    parent_outputs: Tensor,
    child_outputs: Tensor,
    output_sensitivities: Tensor | None = None,
) -> Lfm2CachedSecondOrderEvidence:
    """Validate, detach, normalize, and content-seal cached tensor evidence."""

    if not isinstance(site_id, str) or not site_id.strip():
        raise Lfm2CachedSecondOrderError("site_id must be a nonempty string.")
    if not isinstance(tensor_role, str) or not tensor_role.strip():
        raise Lfm2CachedSecondOrderError("tensor_role must be a nonempty string.")
    named: list[tuple[str, Tensor]] = []
    for name, value in (
        ("parent_inputs", parent_inputs),
        ("child_inputs", child_inputs),
        ("parent_outputs", parent_outputs),
        ("child_outputs", child_outputs),
    ):
        named.append((name, _require_plain_finite_tensor(value, name)))
    if output_sensitivities is not None:
        named.append(
            (
                "output_sensitivities",
                _require_plain_finite_tensor(
                    output_sensitivities,
                    "output_sensitivities",
                ),
            )
        )
    identities = [_storage_identity(value) for _name, value in named]
    if len(identities) != len(set(identities)):
        raise Lfm2CachedSecondOrderError(
            "Evidence tensors must not alias the same storage."
        )

    flattened = {name: _flatten_cached(value, name) for name, value in named}
    parent_x = flattened["parent_inputs"]
    child_x = flattened["child_inputs"]
    parent_y = flattened["parent_outputs"]
    child_y = flattened["child_outputs"]
    sensitivity = flattened.get("output_sensitivities")
    if parent_x.shape != child_x.shape:
        raise Lfm2CachedSecondOrderError(
            "Parent and child cached inputs must have identical geometry."
        )
    if parent_y.shape != child_y.shape:
        raise Lfm2CachedSecondOrderError(
            "Parent and child cached outputs must have identical geometry."
        )
    sample_count = int(parent_x.shape[0])
    if sample_count < 2 or int(parent_y.shape[0]) != sample_count:
        raise Lfm2CachedSecondOrderError(
            "Every evidence tensor must describe the same two or more samples."
        )
    if sensitivity is not None and sensitivity.shape != parent_y.shape:
        raise Lfm2CachedSecondOrderError(
            "Output sensitivities must match cached output geometry."
        )

    pairs = _fingerprint_pairs(
        (
            ("parent_inputs", parent_x),
            ("child_inputs", child_x),
            ("parent_outputs", parent_y),
            ("child_outputs", child_y),
            ("output_sensitivities", sensitivity),
        )
    )
    seal = _sha256_canonical(
        _evidence_seal_payload(
            schema=_EVIDENCE_SCHEMA,
            site_id=site_id.strip(),
            tensor_role=tensor_role.strip(),
            fingerprints=pairs,
        )
    )
    return Lfm2CachedSecondOrderEvidence(
        schema=_EVIDENCE_SCHEMA,
        site_id=site_id.strip(),
        tensor_role=tensor_role.strip(),
        parent_inputs=parent_x,
        child_inputs=child_x,
        parent_outputs=parent_y,
        child_outputs=child_y,
        output_sensitivities=sensitivity,
        tensor_fingerprints=pairs,
        evidence_sha256=seal,
    )


def verify_lfm2_cached_second_order_evidence(
    evidence: Lfm2CachedSecondOrderEvidence,
) -> None:
    """Fail closed if a sealed evidence tensor or its metadata changed."""

    if not isinstance(evidence, Lfm2CachedSecondOrderEvidence):
        raise Lfm2CachedSecondOrderError(
            "evidence must be Lfm2CachedSecondOrderEvidence."
        )
    if evidence.schema != _EVIDENCE_SCHEMA:
        raise Lfm2CachedSecondOrderError("Unsupported cached evidence schema.")
    values = (
        ("parent_inputs", evidence.parent_inputs),
        ("child_inputs", evidence.child_inputs),
        ("parent_outputs", evidence.parent_outputs),
        ("child_outputs", evidence.child_outputs),
        ("output_sensitivities", evidence.output_sensitivities),
    )
    identities: list[tuple[str, int]] = []
    for name, value in values:
        if value is None:
            continue
        _require_plain_finite_tensor(value, name)
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or value.ndim != 2
            or not value.is_contiguous()
        ):
            raise Lfm2CachedSecondOrderError(
                "Sealed evidence tensors must remain contiguous CPU float64."
            )
        identities.append(_storage_identity(value))
    if len(identities) != len(set(identities)):
        raise Lfm2CachedSecondOrderError("Sealed evidence storage is aliased.")
    observed_pairs = _fingerprint_pairs(values)
    observed_seal = _sha256_canonical(
        _evidence_seal_payload(
            schema=evidence.schema,
            site_id=evidence.site_id,
            tensor_role=evidence.tensor_role,
            fingerprints=observed_pairs,
        )
    )
    if (
        observed_pairs != evidence.tensor_fingerprints
        or observed_seal != evidence.evidence_sha256
    ):
        raise Lfm2CachedSecondOrderError(
            "Cached evidence content authentication failed."
        )


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedSecondOrderError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedSecondOrderError(f"{name} must be a positive integer.")
    return value


def _mask_clone(value: Tensor | None) -> Tensor | None:
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise Lfm2CachedSecondOrderError("attention_mask must be a Tensor or None.")
    if isinstance(value, nn.Parameter) or value.requires_grad:
        raise Lfm2CachedSecondOrderError(
            "attention_mask cannot be a Parameter or require gradients."
        )
    if value.device.type == "meta" or value.numel() == 0:
        raise Lfm2CachedSecondOrderError(
            "attention_mask must be a nonempty materialized Tensor."
        )
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise Lfm2CachedSecondOrderError("attention_mask must be finite.")
    if value.dtype.is_complex:
        raise Lfm2CachedSecondOrderError("attention_mask cannot be complex.")
    return value.detach().to(device="cpu").contiguous().clone()


def _attention_bridge_body(
    bridge: Lfm2SealedAttentionBridge,
    fingerprints: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "schema": bridge.schema,
        "site_id": bridge.site_id,
        "batch_size": bridge.batch_size,
        "query_tokens": bridge.query_tokens,
        "key_tokens": bridge.key_tokens,
        "query_heads": bridge.query_heads,
        "key_value_heads": bridge.key_value_heads,
        "head_dim": bridge.head_dim,
        "trace_attention_config_sha256": (bridge.trace_attention_config_sha256),
        "attention_mask_sha256": bridge.attention_mask_sha256,
        "tensor_fingerprints": tuple(fingerprints),
    }


def seal_lfm2_attention_bridge(
    *,
    site_id: str,
    child_attention_probabilities: Tensor,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    trace_attention_config_sha256: str,
    attention_mask: Tensor | None = None,
) -> Lfm2SealedAttentionBridge:
    """Seal fixed child probabilities and GQA geometry for V-to-O propagation."""

    if not isinstance(site_id, str) or not site_id.strip():
        raise Lfm2CachedSecondOrderError("site_id must be a nonempty string.")
    batch_size = _positive_dimension(batch_size, "batch_size")
    query_tokens = _positive_dimension(query_tokens, "query_tokens")
    key_tokens = _positive_dimension(key_tokens, "key_tokens")
    query_heads = _positive_dimension(query_heads, "query_heads")
    key_value_heads = _positive_dimension(
        key_value_heads,
        "key_value_heads",
    )
    head_dim = _positive_dimension(head_dim, "head_dim")
    if query_heads % key_value_heads:
        raise Lfm2CachedSecondOrderError(
            "query_heads must be divisible by key_value_heads for GQA."
        )
    trace_config = _require_sha256(
        trace_attention_config_sha256,
        "trace_attention_config_sha256",
    )
    probabilities = _require_plain_finite_tensor(
        child_attention_probabilities,
        "child_attention_probabilities",
    )
    if attention_mask is not None:
        if not isinstance(attention_mask, Tensor):
            raise Lfm2CachedSecondOrderError("attention_mask must be a Tensor or None.")
        if _storage_identity(probabilities) == _storage_identity(attention_mask):
            raise Lfm2CachedSecondOrderError(
                "Attention probabilities and mask must not alias storage."
            )
    if probabilities.ndim != 4 or tuple(probabilities.shape) != (
        batch_size,
        query_heads,
        query_tokens,
        key_tokens,
    ):
        raise Lfm2CachedSecondOrderError(
            "child_attention_probabilities must have exact "
            "[batch, query_heads, query_tokens, key_tokens] geometry."
        )
    probabilities = (
        probabilities.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if bool(probabilities.lt(0).any()) or bool(probabilities.gt(1).any()):
        raise Lfm2CachedSecondOrderError(
            "Child attention probabilities must be in [0, 1]."
        )
    row_sums = probabilities.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=0.0,
        atol=5e-3,
    ):
        raise Lfm2CachedSecondOrderError(
            "Child attention probabilities must be normalized per query."
        )
    mask = _mask_clone(attention_mask)
    if mask is not None:
        try:
            torch.broadcast_shapes(
                tuple(mask.shape),
                (
                    batch_size,
                    query_heads,
                    query_tokens,
                    key_tokens,
                ),
            )
        except RuntimeError as exc:
            raise Lfm2CachedSecondOrderError(
                "attention_mask cannot broadcast to attention geometry."
            ) from exc
    mask_sha = (
        _sha256_canonical({"attention_mask": "none"})
        if mask is None
        else _tensor_fingerprint(mask)
    )
    fingerprints = _fingerprint_pairs(
        (
            ("child_attention_probabilities", probabilities),
            ("attention_mask", mask),
        )
    )
    provisional = Lfm2SealedAttentionBridge(
        schema=_ATTENTION_BRIDGE_SCHEMA,
        site_id=site_id.strip(),
        batch_size=batch_size,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        child_attention_probabilities=probabilities,
        attention_mask=mask,
        trace_attention_config_sha256=trace_config,
        attention_mask_sha256=mask_sha,
        tensor_fingerprints=fingerprints,
    )
    seal = _sha256_canonical(_attention_bridge_body(provisional, fingerprints))
    return Lfm2SealedAttentionBridge(
        schema=provisional.schema,
        site_id=provisional.site_id,
        batch_size=provisional.batch_size,
        query_tokens=provisional.query_tokens,
        key_tokens=provisional.key_tokens,
        query_heads=provisional.query_heads,
        key_value_heads=provisional.key_value_heads,
        head_dim=provisional.head_dim,
        child_attention_probabilities=probabilities,
        attention_mask=mask,
        trace_attention_config_sha256=trace_config,
        attention_mask_sha256=mask_sha,
        tensor_fingerprints=fingerprints,
        bridge_sha256=seal,
    )


def verify_lfm2_attention_bridge(
    bridge: Lfm2SealedAttentionBridge,
) -> None:
    """Authenticate fixed attention probabilities, mask, and GQA metadata."""

    if not isinstance(bridge, Lfm2SealedAttentionBridge):
        raise Lfm2CachedSecondOrderError(
            "attention_bridge must be Lfm2SealedAttentionBridge."
        )
    if bridge.schema != _ATTENTION_BRIDGE_SCHEMA:
        raise Lfm2CachedSecondOrderError("Unsupported attention bridge schema.")
    _require_sha256(
        bridge.trace_attention_config_sha256,
        "trace_attention_config_sha256",
    )
    for name in (
        "batch_size",
        "query_tokens",
        "key_tokens",
        "query_heads",
        "key_value_heads",
        "head_dim",
    ):
        _positive_dimension(getattr(bridge, name), name)
    if bridge.query_heads % bridge.key_value_heads:
        raise Lfm2CachedSecondOrderError("Sealed bridge has unsupported GQA geometry.")
    probabilities = bridge.child_attention_probabilities
    _require_plain_finite_tensor(
        probabilities,
        "child_attention_probabilities",
    )
    if (
        probabilities.device.type != "cpu"
        or probabilities.dtype != torch.float64
        or not probabilities.is_contiguous()
        or tuple(probabilities.shape)
        != (
            bridge.batch_size,
            bridge.query_heads,
            bridge.query_tokens,
            bridge.key_tokens,
        )
    ):
        raise Lfm2CachedSecondOrderError(
            "Sealed attention probabilities changed geometry or dtype."
        )
    if bool(probabilities.lt(0).any()) or bool(probabilities.gt(1).any()):
        raise Lfm2CachedSecondOrderError("Sealed attention probabilities left [0, 1].")
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(
            (
                bridge.batch_size,
                bridge.query_heads,
                bridge.query_tokens,
            ),
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=5e-3,
    ):
        raise Lfm2CachedSecondOrderError(
            "Sealed attention probabilities are not normalized."
        )
    mask = bridge.attention_mask
    if mask is not None:
        _mask_clone(mask)
    mask_sha = (
        _sha256_canonical({"attention_mask": "none"})
        if mask is None
        else _tensor_fingerprint(mask)
    )
    fingerprints = _fingerprint_pairs(
        (
            ("child_attention_probabilities", probabilities),
            ("attention_mask", mask),
        )
    )
    seal = _sha256_canonical(_attention_bridge_body(bridge, fingerprints))
    if (
        mask_sha != bridge.attention_mask_sha256
        or fingerprints != bridge.tensor_fingerprints
        or seal != bridge.bridge_sha256
    ):
        raise Lfm2CachedSecondOrderError(
            "Attention bridge content authentication failed."
        )


def _validate_ratio(
    value: float,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CachedSecondOrderError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise Lfm2CachedSecondOrderError(
            f"{name} must be finite and in [{minimum}, {maximum}]."
        )
    return result


def _positive_damping(hessian: Tensor, ratio: float) -> float:
    dimension = int(hessian.shape[0])
    mean_diagonal = float(torch.trace(hessian) / dimension)
    scale = max(mean_diagonal, torch.finfo(torch.float64).eps)
    return max(ratio * scale, torch.finfo(torch.float64).eps)


def _require_positive_definite(value: Tensor, name: str) -> None:
    info = torch.linalg.cholesky_ex(value, check_errors=False).info
    if bool(info.ne(0).any()):
        raise Lfm2CachedSecondOrderError(f"{name} is not positive definite.")


def compute_lfm2_second_order_statistics(
    evidence: Lfm2CachedSecondOrderEvidence,
    *,
    damping_ratio: float = 1e-4,
    parent_input_blend: float = 0.5,
    output_damping_ratio: float = 1e-4,
) -> Lfm2SecondOrderStatistics:
    """Build damped second moments without consulting either model."""

    verify_lfm2_cached_second_order_evidence(evidence)
    damping_ratio = _validate_ratio(
        damping_ratio,
        "damping_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    parent_input_blend = _validate_ratio(
        parent_input_blend,
        "parent_input_blend",
        minimum=0.0,
        maximum=1.0,
    )
    output_damping_ratio = _validate_ratio(
        output_damping_ratio,
        "output_damping_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    sample_count = evidence.sample_count
    parent_h = evidence.parent_inputs.T @ evidence.parent_inputs / sample_count
    child_h = evidence.child_inputs.T @ evidence.child_inputs / sample_count
    blended = parent_input_blend * parent_h + (1.0 - parent_input_blend) * child_h
    input_damping = _positive_damping(blended, damping_ratio)
    identity = torch.eye(
        evidence.input_features,
        dtype=torch.float64,
        device="cpu",
    )
    h_in = blended + input_damping * identity
    damped_child = child_h + input_damping * identity
    mismatch = (
        evidence.child_inputs.T
        @ (evidence.parent_inputs - evidence.child_inputs)
        / sample_count
    )
    h_out: Tensor | None = None
    output_damping: float | None = None
    if evidence.output_sensitivities is not None:
        sensitivity_h = (
            evidence.output_sensitivities.T
            @ evidence.output_sensitivities
            / sample_count
        )
        output_damping = _positive_damping(
            sensitivity_h,
            output_damping_ratio,
        )
        output_identity = torch.eye(
            evidence.output_features,
            dtype=torch.float64,
            device="cpu",
        )
        h_out = sensitivity_h + output_damping * output_identity
    _require_positive_definite(h_in, "h_in")
    _require_positive_definite(damped_child, "damped_child_hessian")
    if h_out is not None:
        _require_positive_definite(h_out, "h_out")

    pairs = _fingerprint_pairs(
        (
            ("parent_hessian", parent_h),
            ("child_hessian", child_h),
            ("h_in", h_in),
            ("damped_child_hessian", damped_child),
            ("mismatch_cross_covariance", mismatch),
            ("h_out", h_out),
        )
    )
    payload = {
        "schema": _STATS_SCHEMA,
        "evidence_sha256": evidence.evidence_sha256,
        "input_damping": input_damping,
        "output_damping": output_damping,
        "parent_input_blend": parent_input_blend,
        "tensor_fingerprints": pairs,
    }
    seal = _sha256_canonical(payload)
    return Lfm2SecondOrderStatistics(
        schema=_STATS_SCHEMA,
        evidence_sha256=evidence.evidence_sha256,
        parent_hessian=parent_h,
        child_hessian=child_h,
        h_in=h_in,
        damped_child_hessian=damped_child,
        mismatch_cross_covariance=mismatch,
        h_out=h_out,
        input_damping=input_damping,
        output_damping=output_damping,
        parent_input_blend=parent_input_blend,
        tensor_fingerprints=pairs,
        statistics_sha256=seal,
    )


def _verify_statistics(
    statistics: Lfm2SecondOrderStatistics,
    evidence: Lfm2CachedSecondOrderEvidence,
) -> None:
    if not isinstance(statistics, Lfm2SecondOrderStatistics):
        raise Lfm2CachedSecondOrderError(
            "statistics must be Lfm2SecondOrderStatistics."
        )
    if (
        statistics.schema != _STATS_SCHEMA
        or statistics.evidence_sha256 != evidence.evidence_sha256
    ):
        raise Lfm2CachedSecondOrderError(
            "Second-order statistics do not belong to this evidence."
        )
    pairs = _fingerprint_pairs(
        (
            ("parent_hessian", statistics.parent_hessian),
            ("child_hessian", statistics.child_hessian),
            ("h_in", statistics.h_in),
            ("damped_child_hessian", statistics.damped_child_hessian),
            (
                "mismatch_cross_covariance",
                statistics.mismatch_cross_covariance,
            ),
            ("h_out", statistics.h_out),
        )
    )
    seal = _sha256_canonical(
        {
            "schema": statistics.schema,
            "evidence_sha256": statistics.evidence_sha256,
            "input_damping": statistics.input_damping,
            "output_damping": statistics.output_damping,
            "parent_input_blend": statistics.parent_input_blend,
            "tensor_fingerprints": pairs,
        }
    )
    if pairs != statistics.tensor_fingerprints or seal != statistics.statistics_sha256:
        raise Lfm2CachedSecondOrderError(
            "Second-order statistics content authentication failed."
        )


def _weighted_objective(error: Tensor, h_out: Tensor | None) -> float:
    if error.ndim != 2 or error.numel() == 0:
        raise Lfm2CachedSecondOrderError("Objective error geometry is invalid.")
    if h_out is None:
        result = error.square().mean()
    else:
        mean_diagonal = torch.trace(h_out) / int(h_out.shape[0])
        result = torch.sum((error @ h_out) * error) / error.numel() / mean_diagonal
    observed = float(result)
    if not math.isfinite(observed):
        raise Lfm2CachedSecondOrderError("A nonfinite objective was produced.")
    return observed


def _candidate_objective(
    source_weight: Tensor,
    candidate_weight: Tensor,
    evidence: Lfm2CachedSecondOrderEvidence,
    h_out: Tensor | None,
    *,
    child_inputs: Tensor | None = None,
    child_outputs: Tensor | None = None,
) -> float:
    x = evidence.child_inputs if child_inputs is None else child_inputs
    y = evidence.child_outputs if child_outputs is None else child_outputs
    delta = candidate_weight.to(torch.float64) - source_weight
    prediction = y + x @ delta.T
    return _weighted_objective(prediction - evidence.parent_outputs, h_out)


def _corrected_target(
    source_weight: Tensor,
    evidence: Lfm2CachedSecondOrderEvidence,
    statistics: Lfm2SecondOrderStatistics,
    *,
    mismatch_coefficient: float,
    row_block_size: int,
) -> Tensor:
    if (
        isinstance(row_block_size, bool)
        or not isinstance(row_block_size, int)
        or row_block_size <= 0
    ):
        raise Lfm2CachedSecondOrderError("row_block_size must be positive.")
    residual = evidence.parent_outputs - evidence.child_outputs
    target = source_weight.clone()
    sample_count = evidence.sample_count
    for start in range(0, int(source_weight.shape[0]), row_block_size):
        stop = min(start + row_block_size, int(source_weight.shape[0]))
        rows = source_weight[start:stop]
        rhs = evidence.child_inputs.T @ residual[:, start:stop] / sample_count
        rhs = rhs + (
            mismatch_coefficient * statistics.mismatch_cross_covariance @ rows.T
        )
        delta_transpose = torch.linalg.solve(
            statistics.damped_child_hessian,
            rhs,
        )
        target[start:stop] = rows + delta_transpose.T
    if not bool(torch.isfinite(target).all()):
        raise Lfm2CachedSecondOrderError(
            "The cached mismatch correction produced nonfinite values."
        )
    return target


def _ppm(
    value: float,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> int:
    number = _validate_ratio(
        value,
        name,
        minimum=minimum,
        maximum=maximum,
    )
    ppm = int(round(number * 1_000_000))
    if not math.isclose(number, ppm / 1_000_000, abs_tol=1e-12):
        raise Lfm2CachedSecondOrderError(
            f"{name} must have an exact parts-per-million representation."
        )
    return ppm


def _validated_axis(
    values: Sequence[Any],
    name: str,
) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Lfm2CachedSecondOrderError(f"{name} must be a sequence.")
    result = tuple(values)
    if not result or len(result) != len(set(result)):
        raise Lfm2CachedSecondOrderError(
            f"{name} must be nonempty and contain no duplicates."
        )
    return result


def _grid_configs(
    *,
    input_features: int,
    bit_widths: Sequence[int],
    group_sizes: Sequence[int],
    clipping_ratios: Sequence[float],
    rounding_offsets: Sequence[float],
    outliers_per_group: Sequence[int],
    mismatch_coefficients: Sequence[float],
    second_input_features: int | None = None,
) -> tuple[Lfm2SecondOrderQuantizerConfig, ...]:
    bits = _validated_axis(bit_widths, "bit_widths")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in _ALLOWED_BIT_WIDTHS
        for value in bits
    ):
        raise Lfm2CachedSecondOrderError(
            "bit_widths must contain only integers from four through eight."
        )
    requested_groups = _validated_axis(group_sizes, "group_sizes")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in _ALLOWED_GROUP_SIZES
        for value in requested_groups
    ):
        raise Lfm2CachedSecondOrderError(
            "group_sizes must contain only 32, 64, 128, or 256."
        )
    compatible_groups = tuple(
        value
        for value in requested_groups
        if input_features % value == 0
        and (second_input_features is None or second_input_features % value == 0)
    )
    if not compatible_groups:
        raise Lfm2CachedSecondOrderError(
            "No requested row-aligned group size fits the tensor geometry."
        )
    clips = tuple(
        _ppm(value, "clipping_ratio", minimum=0.25, maximum=1.0)
        for value in _validated_axis(clipping_ratios, "clipping_ratios")
    )
    offsets = tuple(
        _ppm(value, "rounding_offset", minimum=-0.49, maximum=0.49)
        for value in _validated_axis(rounding_offsets, "rounding_offsets")
    )
    outliers = _validated_axis(outliers_per_group, "outliers_per_group")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in outliers
    ):
        raise Lfm2CachedSecondOrderError(
            "outliers_per_group must contain nonnegative integers."
        )
    mismatch = tuple(
        _ppm(value, "mismatch_coefficient", minimum=-2.0, maximum=2.0)
        for value in _validated_axis(
            mismatch_coefficients,
            "mismatch_coefficients",
        )
    )
    configs = []
    for bit_width in sorted(bits):
        for group_size in sorted(compatible_groups):
            for clip in sorted(clips):
                for offset in sorted(offsets):
                    for outlier_count in sorted(outliers):
                        if outlier_count >= group_size:
                            raise Lfm2CachedSecondOrderError(
                                "outliers_per_group must be smaller than every "
                                "compatible group size."
                            )
                        for mismatch_ppm in sorted(mismatch):
                            configs.append(
                                Lfm2SecondOrderQuantizerConfig(
                                    bit_width=bit_width,
                                    group_size=group_size,
                                    clipping_ratio_ppm=clip,
                                    rounding_offset_ppm=offset,
                                    outliers_per_group=outlier_count,
                                    mismatch_coefficient_ppm=mismatch_ppm,
                                )
                            )
    return tuple(configs)


def _source_weight(value: Any, name: str) -> tuple[Tensor, str]:
    tensor = _require_plain_finite_tensor(value, name)
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        raise Lfm2CachedSecondOrderError(
            f"{name} must be a materialized two-dimensional BF16 Tensor."
        )
    fingerprint = _tensor_fingerprint(tensor)
    return (
        tensor.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .to(torch.float64),
        fingerprint,
    )


def _stable_outlier_mask(
    target_bf16: Tensor,
    *,
    group_size: int,
    outliers_per_group: int,
) -> Tensor:
    rows, columns = (int(value) for value in target_bf16.shape)
    mask = torch.zeros((rows, columns), dtype=torch.bool)
    if outliers_per_group == 0:
        return mask
    for start in range(0, columns, group_size):
        stop = start + group_size
        magnitudes = target_bf16[:, start:stop].to(torch.float64).abs()
        selected = torch.argsort(
            magnitudes,
            dim=1,
            descending=True,
            stable=True,
        )[:, :outliers_per_group]
        mask[:, start:stop].scatter_(1, selected, True)
    return mask


def _minimum_positive_fp16() -> Tensor:
    zero = torch.tensor(0.0, dtype=torch.float16)
    one = torch.tensor(1.0, dtype=torch.float16)
    return torch.nextafter(zero, one)


def _quantize_solved_tensor(
    *,
    source_weight: Tensor,
    target_weight: Tensor,
    h_in: Tensor,
    h_out: Tensor | None,
    output_row_block_size: int,
    name: str,
    role: str,
    config: Lfm2SecondOrderQuantizerConfig,
    successive: bool,
    source_sha256: str,
    prepared_input_inverse_cholesky: Tensor | None = None,
    prepared_output_inverse_cholesky_blocks: (
        Sequence[tuple[int, int, Tensor]] | None
    ) = None,
) -> Lfm2CrevfSolvedTensor:
    rows, columns = (int(value) for value in target_weight.shape)
    group_size = config.group_size
    bit_width = config.bit_width
    if columns % group_size:
        raise Lfm2CachedSecondOrderError(
            "Quantizer groups must be aligned within every output row."
        )
    if h_in.shape != (columns, columns):
        raise Lfm2CachedSecondOrderError(
            "Input Hessian geometry does not match the target tensor."
        )
    if h_out is not None and h_out.shape != (rows, rows):
        raise Lfm2CachedSecondOrderError(
            "Output Hessian geometry does not match the target tensor."
        )
    if (
        isinstance(output_row_block_size, bool)
        or not isinstance(output_row_block_size, int)
        or output_row_block_size <= 0
    ):
        raise Lfm2CachedSecondOrderError("output_row_block_size must be positive.")
    qmax = (1 << (bit_width - 1)) - 1
    target_bf16 = target_weight.to(torch.bfloat16).contiguous()
    target = target_bf16.to(torch.float64)
    working = target.clone()
    outlier_mask = _stable_outlier_mask(
        target_bf16,
        group_size=group_size,
        outliers_per_group=config.outliers_per_group,
    )
    codes = torch.full((rows, columns), qmax, dtype=torch.uint8)
    reconstruction = torch.empty((rows, columns), dtype=torch.float64)
    scales = torch.empty(
        (rows, columns // group_size),
        dtype=torch.float16,
    )
    inverse_cholesky: Tensor | None = None
    if successive:
        if prepared_input_inverse_cholesky is None:
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_in))
            inverse_cholesky = torch.linalg.cholesky(inverse, upper=True)
        else:
            inverse_cholesky = prepared_input_inverse_cholesky
            if (
                inverse_cholesky.shape != h_in.shape
                or inverse_cholesky.dtype != torch.float64
                or inverse_cholesky.device.type != "cpu"
                or not bool(torch.isfinite(inverse_cholesky).all())
                or bool(torch.diag(inverse_cholesky).le(0).any())
            ):
                raise Lfm2CachedSecondOrderError(
                    "Prepared input inverse-Cholesky factor is invalid."
                )
    minimum_scale = _minimum_positive_fp16()

    def quantize_row(row: int) -> tuple[Tensor, Tensor, Tensor]:
        row_working = working[row].clone()
        row_target = target[row]
        row_mask = outlier_mask[row]
        row_reconstruction = torch.empty(columns, dtype=torch.float64)
        row_codes = torch.full((columns,), qmax, dtype=torch.uint8)
        row_scales = torch.empty(
            columns // group_size,
            dtype=torch.float16,
        )
        rounding_offset = config.rounding_offset if successive else 0.0
        for group_index, start in enumerate(range(0, columns, group_size)):
            stop = start + group_size
            included = ~row_mask[start:stop]
            maximum = torch.where(
                included,
                row_target[start:stop].abs(),
                torch.zeros(group_size, dtype=torch.float64),
            ).amax()
            raw_scale = maximum * config.clipping_ratio / qmax
            if float(maximum) == 0.0:
                raw_scale = torch.tensor(1.0, dtype=torch.float64)
            fp16_scale = raw_scale.to(torch.float16)
            if float(maximum) > 0.0 and float(fp16_scale) == 0.0:
                fp16_scale = minimum_scale
            if not bool(torch.isfinite(fp16_scale)) or float(fp16_scale) <= 0:
                raise Lfm2CachedSecondOrderError(
                    "A quantization scale is not finite positive FP16."
                )
            row_scales[group_index] = fp16_scale
            scale64 = fp16_scale.to(torch.float64)
            for column in range(start, stop):
                value = row_working[column]
                magnitude = torch.floor(
                    value.abs() / scale64 + 0.5 + rounding_offset
                ).clamp_(0, qmax)
                signed = torch.sign(value) * magnitude
                quantized = signed * scale64
                raw_code = signed.to(torch.int16) + qmax
                if bool(row_mask[column]):
                    quantized = row_target[column]
                    raw_code = torch.tensor(qmax, dtype=torch.int16)
                row_reconstruction[column] = quantized
                row_codes[column] = raw_code.to(torch.uint8)
                if inverse_cholesky is not None:
                    diagonal = inverse_cholesky[column, column]
                    if not bool(torch.isfinite(diagonal)) or float(diagonal) <= 0:
                        raise Lfm2CachedSecondOrderError(
                            "Successive rounding encountered an invalid inverse "
                            "input-Hessian diagonal."
                        )
                    error = (value - quantized) / diagonal
                    row_working[column:] -= error * inverse_cholesky[column, column:]
        return row_reconstruction, row_codes, row_scales

    prepared_output_blocks: dict[tuple[int, int], Tensor] | None = None
    if prepared_output_inverse_cholesky_blocks is not None:
        prepared_output_blocks = {
            (start, stop): factor
            for start, stop, factor in (
                prepared_output_inverse_cholesky_blocks
            )
        }
        if len(prepared_output_blocks) != len(
            prepared_output_inverse_cholesky_blocks
        ):
            raise Lfm2CachedSecondOrderError(
                "Prepared output inverse-Cholesky blocks repeat geometry."
            )
    for block_start in range(0, rows, output_row_block_size):
        block_stop = min(block_start + output_row_block_size, rows)
        output_inverse_cholesky: Tensor | None = None
        if successive and h_out is not None:
            if prepared_output_blocks is None:
                block_hessian = h_out[
                    block_start:block_stop,
                    block_start:block_stop,
                ]
                _require_positive_definite(
                    block_hessian,
                    "output Hessian row block",
                )
                block_inverse = torch.cholesky_inverse(
                    torch.linalg.cholesky(block_hessian)
                )
                output_inverse_cholesky = torch.linalg.cholesky(
                    block_inverse,
                    upper=True,
                )
            else:
                try:
                    output_inverse_cholesky = prepared_output_blocks[
                        (block_start, block_stop)
                    ]
                except KeyError as exc:
                    raise Lfm2CachedSecondOrderError(
                        "Prepared output inverse-Cholesky blocks do not "
                        "cover quantizer row geometry."
                    ) from exc
                block_size = block_stop - block_start
                if (
                    output_inverse_cholesky.shape
                    != (block_size, block_size)
                    or output_inverse_cholesky.dtype != torch.float64
                    or output_inverse_cholesky.device.type != "cpu"
                    or not bool(
                        torch.isfinite(output_inverse_cholesky).all()
                    )
                    or bool(
                        torch.diag(output_inverse_cholesky).le(0).any()
                    )
                ):
                    raise Lfm2CachedSecondOrderError(
                        "Prepared output inverse-Cholesky block is invalid."
                    )
        for row in range(block_start, block_stop):
            row_reconstruction, row_codes, row_scales = quantize_row(row)
            reconstruction[row] = row_reconstruction
            codes[row] = row_codes
            scales[row] = row_scales
            if output_inverse_cholesky is not None:
                local_row = row - block_start
                diagonal = output_inverse_cholesky[local_row, local_row]
                if not bool(torch.isfinite(diagonal)) or float(diagonal) <= 0:
                    raise Lfm2CachedSecondOrderError(
                        "Successive rounding encountered an invalid inverse "
                        "output-Hessian diagonal."
                    )
                row_error = (working[row] - row_reconstruction) / diagonal
                working[row:block_stop] -= (
                    output_inverse_cholesky[
                        local_row,
                        local_row:,
                    ][:, None]
                    * row_error[None, :]
                )

    flat_mask = outlier_mask.reshape(-1)
    flat_indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1).tolist()
    words = bf16_bits(target_bf16.reshape(-1))
    exact_outliers = tuple(
        CrevfExactValue(int(index), int(words[index])) for index in flat_indices
    )
    group_count = rows * (columns // group_size)
    group_codes = codes.reshape(group_count, group_size).contiguous()
    flat_scales = scales.reshape(-1).contiguous()
    spec = CrevfTensorSpec(
        name=name,
        role=role,
        shape=(rows, columns),
        group_size=group_size,
        group_bit_widths=(bit_width,) * group_count,
        group_codes=group_codes,
        scales=flat_scales,
        exact_outliers=exact_outliers,
    )
    blob = encode_crevf_tensor(spec)
    packed = decode_crevf_tensor(blob)
    decoded = packed.dequantize().to(device="cpu").contiguous()
    del packed
    expected = reconstruction.to(torch.bfloat16)
    if not torch.equal(decoded, expected):
        raise Lfm2CachedSecondOrderError(
            "CREVF reconstruction diverged from solved quantizer codes."
        )
    metadata = inspect_crevf_tensor(blob)
    record = (
        _RECORD_PREFIX.pack(
            len(blob),
            hashlib.sha256(blob).digest(),
        )
        + blob
    )
    if len(record) != CREVF_RECORD_ENVELOPE_BYTES + len(blob):
        raise Lfm2CachedSecondOrderError("CREVF record accounting diverged.")
    dense_bytes = int(source_weight.numel()) * 2
    return Lfm2CrevfSolvedTensor(
        name=name,
        role=role,
        shape=(rows, columns),
        bit_width=bit_width,
        group_size=group_size,
        tensor_blob=blob,
        group_codes=group_codes.clone(),
        scales=flat_scales.clone(),
        exact_outliers=exact_outliers,
        reconstructed_tensor=decoded,
        source_parameter_count=int(source_weight.numel()),
        dense_bf16_bytes=dense_bytes,
        tensor_blob_bytes=len(blob),
        physical_record_bytes=len(record),
        physical_byte_savings=dense_bytes - len(record),
        persistent_encoded_bytes=metadata.persistent_encoded_bytes,
        decoded_full_state_bytes=metadata.decoded_full_state_bytes,
        decoded_code_transient_bytes=metadata.decoded_code_transient_bytes,
        source_sha256=source_sha256,
        reconstruction_sha256=_tensor_fingerprint(decoded),
        tensor_blob_sha256=_sha256_bytes(blob),
        physical_payload_sha256=_sha256_bytes(record),
    )


def _configuration_sha256(config: Any) -> str:
    if isinstance(config, Lfm2SecondOrderQuantizerConfig):
        value = {
            "bit_width": config.bit_width,
            "group_size": config.group_size,
            "clipping_ratio_ppm": config.clipping_ratio_ppm,
            "rounding_offset_ppm": config.rounding_offset_ppm,
            "outliers_per_group": config.outliers_per_group,
            "mismatch_coefficient_ppm": config.mismatch_coefficient_ppm,
        }
    elif isinstance(config, Lfm2JointAttentionConfig):
        value = {
            "quantizer_sha256": _configuration_sha256(config.quantizer),
            "propagation_coefficient_ppm": config.propagation_coefficient_ppm,
            "value_objective_weight_ppm": config.value_objective_weight_ppm,
        }
    else:
        raise Lfm2CachedSecondOrderError("Unsupported configuration type.")
    return _sha256_canonical(value)


def _assert_source_unchanged(
    original: Tensor,
    initial_sha256: str,
    name: str,
) -> None:
    if _tensor_fingerprint(original) != initial_sha256:
        raise Lfm2CachedSecondOrderError(f"{name} changed during the solve.")


def solve_lfm2_crevf_second_order_grid(
    weight: Tensor,
    evidence: Lfm2CachedSecondOrderEvidence,
    *,
    name: str,
    role: str,
    bit_widths: Sequence[int] = (4, 5, 6, 7, 8),
    group_sizes: Sequence[int] = (32, 64, 128, 256),
    clipping_ratios: Sequence[float] = (0.9, 1.0),
    rounding_offsets: Sequence[float] = (0.0,),
    outliers_per_group: Sequence[int] = (0,),
    mismatch_coefficients: Sequence[float] = (0.0, 1.0),
    damping_ratio: float = 1e-4,
    parent_input_blend: float = 0.5,
    output_damping_ratio: float = 1e-4,
    row_block_size: int = 32,
    max_tensor_physical_bytes: int | None = None,
) -> Lfm2SecondOrderGridResult:
    """Solve and serialize a deterministic Hessian-weighted CREVF grid."""

    verify_lfm2_cached_second_order_evidence(evidence)
    source, source_sha = _source_weight(weight, "weight")
    if source.shape != (
        evidence.output_features,
        evidence.input_features,
    ):
        raise Lfm2CachedSecondOrderError(
            "Weight geometry does not match the sealed evidence."
        )
    if not isinstance(name, str) or not name.strip():
        raise Lfm2CachedSecondOrderError("name must be a nonempty string.")
    if not isinstance(role, str) or not role.strip():
        raise Lfm2CachedSecondOrderError("role must be a nonempty string.")
    if max_tensor_physical_bytes is not None and (
        isinstance(max_tensor_physical_bytes, bool)
        or not isinstance(max_tensor_physical_bytes, int)
        or max_tensor_physical_bytes <= 0
    ):
        raise Lfm2CachedSecondOrderError(
            "max_tensor_physical_bytes must be positive or None."
        )
    configs = _grid_configs(
        input_features=evidence.input_features,
        bit_widths=bit_widths,
        group_sizes=group_sizes,
        clipping_ratios=clipping_ratios,
        rounding_offsets=rounding_offsets,
        outliers_per_group=outliers_per_group,
        mismatch_coefficients=mismatch_coefficients,
    )
    statistics = compute_lfm2_second_order_statistics(
        evidence,
        damping_ratio=damping_ratio,
        parent_input_blend=parent_input_blend,
        output_damping_ratio=output_damping_ratio,
    )
    _verify_statistics(statistics, evidence)
    current = _candidate_objective(
        source,
        source,
        evidence,
        statistics.h_out,
    )
    targets: dict[int, Tensor] = {}
    full_precision: dict[int, float] = {}
    options: list[Lfm2SecondOrderQuantizedOption] = []
    audit = Lfm2SecondOrderOpsAudit(
        output_hessian_row_compensation=statistics.h_out is not None,
    )
    for config in configs:
        mismatch_ppm = config.mismatch_coefficient_ppm
        if mismatch_ppm not in targets:
            target = _corrected_target(
                source,
                evidence,
                statistics,
                mismatch_coefficient=config.mismatch_coefficient,
                row_block_size=row_block_size,
            )
            targets[mismatch_ppm] = target
            full_precision[mismatch_ppm] = _candidate_objective(
                source,
                target,
                evidence,
                statistics.h_out,
            )
        target = targets[mismatch_ppm]
        rtn_payload = _quantize_solved_tensor(
            source_weight=source,
            target_weight=target,
            h_in=statistics.h_in,
            h_out=statistics.h_out,
            output_row_block_size=row_block_size,
            name=name.strip(),
            role=role.strip(),
            config=config,
            successive=False,
            source_sha256=source_sha,
        )
        rtn_objective = _candidate_objective(
            source,
            rtn_payload.reconstructed_tensor,
            evidence,
            statistics.h_out,
        )
        successive_payload = _quantize_solved_tensor(
            source_weight=source,
            target_weight=target,
            h_in=statistics.h_in,
            h_out=statistics.h_out,
            output_row_block_size=row_block_size,
            name=name.strip(),
            role=role.strip(),
            config=config,
            successive=True,
            source_sha256=source_sha,
        )
        successive_objective = _candidate_objective(
            source,
            successive_payload.reconstructed_tensor,
            evidence,
            statistics.h_out,
        )
        tolerance = 1e-12 * max(1.0, abs(rtn_objective))
        if successive_objective <= rtn_objective + tolerance:
            selected_method = "successive"
            selected_payload = successive_payload
            selected_objective = successive_objective
        else:
            selected_method = "rtn_rollback"
            selected_payload = rtn_payload
            selected_objective = rtn_objective
        config_sha = _configuration_sha256(config)
        option_id = f"{name.strip()}:{config.option_suffix}"
        option_sha = _sha256_canonical(
            {
                "option_id": option_id,
                "config_sha256": config_sha,
                "evidence_sha256": evidence.evidence_sha256,
                "statistics_sha256": statistics.statistics_sha256,
                "source_sha256": source_sha,
                "physical_payload_sha256": (selected_payload.physical_payload_sha256),
                "selected_method": selected_method,
                "selected_objective": selected_objective,
                "rtn_physical_payload_sha256": (rtn_payload.physical_payload_sha256),
                "raw_successive_physical_payload_sha256": (
                    successive_payload.physical_payload_sha256
                ),
            }
        )
        options.append(
            Lfm2SecondOrderQuantizedOption(
                option_id=option_id,
                config=config,
                payload=selected_payload,
                selected_method=selected_method,
                current_objective=current,
                full_precision_objective=full_precision[mismatch_ppm],
                rtn_objective=rtn_objective,
                raw_successive_objective=successive_objective,
                selected_objective=selected_objective,
                improvement_vs_current=current - selected_objective,
                improvement_vs_rtn=rtn_objective - selected_objective,
                evidence_sha256=evidence.evidence_sha256,
                statistics_sha256=statistics.statistics_sha256,
                source_sha256=source_sha,
                config_sha256=config_sha,
                rtn_physical_payload_sha256=(rtn_payload.physical_payload_sha256),
                raw_successive_physical_payload_sha256=(
                    successive_payload.physical_payload_sha256
                ),
                option_sha256=option_sha,
                ops_audit=audit,
            )
        )
    options.sort(key=lambda item: item.option_id)
    eligible = [
        item
        for item in options
        if max_tensor_physical_bytes is None
        or item.physical_record_bytes <= max_tensor_physical_bytes
    ]
    if not eligible:
        raise Lfm2CachedSecondOrderError(
            "No solved CREVF tensor fits max_tensor_physical_bytes."
        )
    best = min(
        eligible,
        key=lambda item: (
            item.selected_objective,
            item.physical_record_bytes,
            item.option_id,
        ),
    )
    _assert_source_unchanged(weight, source_sha, "weight")
    verify_lfm2_cached_second_order_evidence(evidence)
    grid_sha = _sha256_canonical(
        {
            "evidence_sha256": evidence.evidence_sha256,
            "statistics_sha256": statistics.statistics_sha256,
            "source_sha256": source_sha,
            "option_sha256": tuple(item.option_sha256 for item in options),
            "max_tensor_physical_bytes": max_tensor_physical_bytes,
            "best_option": best.option_id,
        }
    )
    return Lfm2SecondOrderGridResult(
        evidence_sha256=evidence.evidence_sha256,
        statistics_sha256=statistics.statistics_sha256,
        source_sha256=source_sha,
        grid_sha256=grid_sha,
        current_objective=current,
        options=tuple(options),
        best_option=best,
        max_tensor_physical_bytes=max_tensor_physical_bytes,
        ops_audit=audit,
    )


def _attention_context_from_values(
    values: Tensor,
    bridge: Lfm2SealedAttentionBridge,
) -> Tensor:
    expected_shape = (
        bridge.batch_size * bridge.key_tokens,
        bridge.value_hidden_size,
    )
    if values.shape != expected_shape:
        raise Lfm2CachedSecondOrderError(
            "Value projection evidence does not match the attention bridge."
        )
    value_heads = values.reshape(
        bridge.batch_size,
        bridge.key_tokens,
        bridge.key_value_heads,
        bridge.head_dim,
    ).permute(0, 2, 1, 3)
    repeats = bridge.query_heads // bridge.key_value_heads
    query_value_heads = value_heads.repeat_interleave(repeats, dim=1)
    context_heads = torch.einsum(
        "bhqk,bhkd->bhqd",
        bridge.child_attention_probabilities,
        query_value_heads,
    )
    return (
        context_heads.permute(0, 2, 1, 3)
        .reshape(
            bridge.batch_size * bridge.query_tokens,
            bridge.query_hidden_size,
        )
        .contiguous()
    )


def _joint_branch(
    *,
    successive: bool,
    value_source: Tensor,
    output_source: Tensor,
    value_target: Tensor,
    value_statistics: Lfm2SecondOrderStatistics,
    value_evidence: Lfm2CachedSecondOrderEvidence,
    output_evidence: Lfm2CachedSecondOrderEvidence,
    attention_bridge: Lfm2SealedAttentionBridge,
    config: Lfm2JointAttentionConfig,
    value_name: str,
    value_role: str,
    output_name: str,
    output_role: str,
    value_source_sha: str,
    output_source_sha: str,
    damping_ratio: float,
    parent_input_blend: float,
    output_damping_ratio: float,
    row_block_size: int,
    prepared_value_input_inverse_cholesky: Tensor | None = None,
    prepared_value_output_inverse_cholesky_blocks: (
        Sequence[tuple[int, int, Tensor]] | None
    ) = None,
) -> tuple[Lfm2CrevfSolvedTensor, Lfm2CrevfSolvedTensor, float, float]:
    value_payload = _quantize_solved_tensor(
        source_weight=value_source,
        target_weight=value_target,
        h_in=value_statistics.h_in,
        h_out=value_statistics.h_out,
        output_row_block_size=row_block_size,
        name=value_name,
        role=value_role,
        config=config.quantizer,
        successive=successive,
        source_sha256=value_source_sha,
        prepared_input_inverse_cholesky=(
            prepared_value_input_inverse_cholesky
        ),
        prepared_output_inverse_cholesky_blocks=(
            prepared_value_output_inverse_cholesky_blocks
        ),
    )
    value_objective = _candidate_objective(
        value_source,
        value_payload.reconstructed_tensor,
        value_evidence,
        value_statistics.h_out,
    )
    value_delta = value_payload.reconstructed_tensor.to(torch.float64) - value_source
    projected_value_delta = value_evidence.child_inputs @ value_delta.T
    attention_context_delta = _attention_context_from_values(
        projected_value_delta,
        attention_bridge,
    )
    joint_context = output_evidence.child_inputs + (
        config.propagation_coefficient * attention_context_delta
    )
    joint_child_outputs = output_evidence.child_outputs + (
        (joint_context - output_evidence.child_inputs) @ output_source.T
    )
    branch_evidence = seal_lfm2_cached_second_order_evidence(
        site_id=(
            f"{output_evidence.site_id}:"
            f"{'successive' if successive else 'rtn'}:"
            f"{config.quantizer.option_suffix}:"
            f"p{config.propagation_coefficient_ppm:+d}"
        ),
        tensor_role=output_evidence.tensor_role,
        parent_inputs=output_evidence.parent_inputs.clone(),
        child_inputs=joint_context.clone(),
        parent_outputs=output_evidence.parent_outputs.clone(),
        child_outputs=joint_child_outputs.clone(),
        output_sensitivities=(
            None
            if output_evidence.output_sensitivities is None
            else output_evidence.output_sensitivities.clone()
        ),
    )
    output_statistics = compute_lfm2_second_order_statistics(
        branch_evidence,
        damping_ratio=damping_ratio,
        parent_input_blend=parent_input_blend,
        output_damping_ratio=output_damping_ratio,
    )
    output_target = _corrected_target(
        output_source,
        branch_evidence,
        output_statistics,
        mismatch_coefficient=config.quantizer.mismatch_coefficient,
        row_block_size=row_block_size,
    )
    output_payload = _quantize_solved_tensor(
        source_weight=output_source,
        target_weight=output_target,
        h_in=output_statistics.h_in,
        h_out=output_statistics.h_out,
        output_row_block_size=row_block_size,
        name=output_name,
        role=output_role,
        config=config.quantizer,
        successive=successive,
        source_sha256=output_source_sha,
    )
    output_objective = _candidate_objective(
        output_source,
        output_payload.reconstructed_tensor,
        branch_evidence,
        output_statistics.h_out,
    )
    joint_objective = output_objective + config.value_objective_weight * value_objective
    return (
        value_payload,
        output_payload,
        value_objective,
        joint_objective,
    )


def solve_lfm2_joint_attention_vo(
    value_weight: Tensor,
    output_weight: Tensor,
    value_evidence: Lfm2CachedSecondOrderEvidence,
    output_evidence: Lfm2CachedSecondOrderEvidence,
    *,
    attention_bridge: Lfm2SealedAttentionBridge,
    value_name: str,
    output_name: str,
    value_role: str = "attention.v",
    output_role: str = "attention.o",
    bit_widths: Sequence[int] = (4, 5, 6, 7, 8),
    group_sizes: Sequence[int] = (32, 64, 128, 256),
    clipping_ratios: Sequence[float] = (0.9, 1.0),
    rounding_offsets: Sequence[float] = (0.0,),
    outliers_per_group: Sequence[int] = (0,),
    mismatch_coefficients: Sequence[float] = (0.0, 1.0),
    propagation_coefficients: Sequence[float] = (0.5, 1.0),
    value_objective_weight: float = 0.1,
    damping_ratio: float = 1e-4,
    parent_input_blend: float = 0.5,
    output_damping_ratio: float = 1e-4,
    row_block_size: int = 32,
    max_bundle_physical_bytes: int | None = None,
) -> Lfm2JointAttentionResult:
    """Jointly solve attention V/O payloads with cached mismatch propagation."""

    verify_lfm2_cached_second_order_evidence(value_evidence)
    verify_lfm2_cached_second_order_evidence(output_evidence)
    verify_lfm2_attention_bridge(attention_bridge)
    value_source, value_source_sha = _source_weight(
        value_weight,
        "value_weight",
    )
    output_source, output_source_sha = _source_weight(
        output_weight,
        "output_weight",
    )
    if value_source.shape != (
        value_evidence.output_features,
        value_evidence.input_features,
    ):
        raise Lfm2CachedSecondOrderError(
            "Value weight geometry does not match its evidence."
        )
    if output_source.shape != (
        output_evidence.output_features,
        output_evidence.input_features,
    ):
        raise Lfm2CachedSecondOrderError(
            "Output weight geometry does not match its evidence."
        )
    if (
        value_evidence.sample_count
        != attention_bridge.batch_size * attention_bridge.key_tokens
        or output_evidence.sample_count
        != attention_bridge.batch_size * attention_bridge.query_tokens
        or value_evidence.output_features != attention_bridge.value_hidden_size
        or output_evidence.input_features != attention_bridge.query_hidden_size
    ):
        raise Lfm2CachedSecondOrderError(
            "Joint V/O evidence does not match the sealed attention/GQA "
            "bridge geometry."
        )
    reproduced_child_context = _attention_context_from_values(
        value_evidence.child_outputs,
        attention_bridge,
    )
    if not torch.allclose(
        reproduced_child_context,
        output_evidence.child_inputs,
        rtol=5e-3,
        atol=5e-4,
    ):
        maximum_error = float(
            (reproduced_child_context - output_evidence.child_inputs).abs().amax()
        )
        raise Lfm2CachedSecondOrderError(
            "Sealed attention probabilities do not reproduce the cached "
            f"child O input (maximum error {maximum_error:.8g})."
        )
    for name, value in (
        ("value_name", value_name),
        ("output_name", output_name),
        ("value_role", value_role),
        ("output_role", output_role),
    ):
        if not isinstance(value, str) or not value.strip():
            raise Lfm2CachedSecondOrderError(f"{name} must be a nonempty string.")
    if max_bundle_physical_bytes is not None and (
        isinstance(max_bundle_physical_bytes, bool)
        or not isinstance(max_bundle_physical_bytes, int)
        or max_bundle_physical_bytes <= 0
    ):
        raise Lfm2CachedSecondOrderError(
            "max_bundle_physical_bytes must be positive or None."
        )
    quantizer_configs = _grid_configs(
        input_features=value_evidence.input_features,
        second_input_features=output_evidence.input_features,
        bit_widths=bit_widths,
        group_sizes=group_sizes,
        clipping_ratios=clipping_ratios,
        rounding_offsets=rounding_offsets,
        outliers_per_group=outliers_per_group,
        mismatch_coefficients=mismatch_coefficients,
    )
    propagation_ppm = tuple(
        _ppm(value, "propagation_coefficient", minimum=0.0, maximum=2.0)
        for value in _validated_axis(
            propagation_coefficients,
            "propagation_coefficients",
        )
    )
    objective_weight_ppm = _ppm(
        value_objective_weight,
        "value_objective_weight",
        minimum=0.0,
        maximum=10.0,
    )
    value_statistics = compute_lfm2_second_order_statistics(
        value_evidence,
        damping_ratio=damping_ratio,
        parent_input_blend=parent_input_blend,
        output_damping_ratio=output_damping_ratio,
    )
    output_statistics = compute_lfm2_second_order_statistics(
        output_evidence,
        damping_ratio=damping_ratio,
        parent_input_blend=parent_input_blend,
        output_damping_ratio=output_damping_ratio,
    )
    current_value = _candidate_objective(
        value_source,
        value_source,
        value_evidence,
        value_statistics.h_out,
    )
    current_output = _candidate_objective(
        output_source,
        output_source,
        output_evidence,
        output_statistics.h_out,
    )
    current_joint = current_output + (objective_weight_ppm / 1_000_000 * current_value)
    value_targets: dict[int, Tensor] = {}
    options: list[Lfm2JointAttentionOption] = []
    audit = Lfm2SecondOrderOpsAudit(
        output_hessian_row_compensation=(
            value_statistics.h_out is not None or output_statistics.h_out is not None
        ),
        fixed_attention_probability_bridge=True,
    )
    for quantizer in quantizer_configs:
        if quantizer.mismatch_coefficient_ppm not in value_targets:
            value_targets[quantizer.mismatch_coefficient_ppm] = _corrected_target(
                value_source,
                value_evidence,
                value_statistics,
                mismatch_coefficient=quantizer.mismatch_coefficient,
                row_block_size=row_block_size,
            )
        for propagation in sorted(propagation_ppm):
            config = Lfm2JointAttentionConfig(
                quantizer=quantizer,
                propagation_coefficient_ppm=propagation,
                value_objective_weight_ppm=objective_weight_ppm,
            )
            rtn = _joint_branch(
                successive=False,
                value_source=value_source,
                output_source=output_source,
                value_target=value_targets[quantizer.mismatch_coefficient_ppm],
                value_statistics=value_statistics,
                value_evidence=value_evidence,
                output_evidence=output_evidence,
                attention_bridge=attention_bridge,
                config=config,
                value_name=value_name.strip(),
                value_role=value_role.strip(),
                output_name=output_name.strip(),
                output_role=output_role.strip(),
                value_source_sha=value_source_sha,
                output_source_sha=output_source_sha,
                damping_ratio=damping_ratio,
                parent_input_blend=parent_input_blend,
                output_damping_ratio=output_damping_ratio,
                row_block_size=row_block_size,
            )
            successive = _joint_branch(
                successive=True,
                value_source=value_source,
                output_source=output_source,
                value_target=value_targets[quantizer.mismatch_coefficient_ppm],
                value_statistics=value_statistics,
                value_evidence=value_evidence,
                output_evidence=output_evidence,
                attention_bridge=attention_bridge,
                config=config,
                value_name=value_name.strip(),
                value_role=value_role.strip(),
                output_name=output_name.strip(),
                output_role=output_role.strip(),
                value_source_sha=value_source_sha,
                output_source_sha=output_source_sha,
                damping_ratio=damping_ratio,
                parent_input_blend=parent_input_blend,
                output_damping_ratio=output_damping_ratio,
                row_block_size=row_block_size,
            )
            tolerance = 1e-12 * max(1.0, abs(rtn[3]))
            if successive[3] <= rtn[3] + tolerance:
                selected = successive
                selected_method = "successive"
            else:
                selected = rtn
                selected_method = "rtn_rollback"
            config_sha = _configuration_sha256(config)
            evidence_bundle_sha = _sha256_canonical(
                {
                    "value": value_evidence.evidence_sha256,
                    "output": output_evidence.evidence_sha256,
                    "attention_bridge": attention_bridge.bridge_sha256,
                }
            )
            physical_bundle_sha = _sha256_canonical(
                {
                    "value": selected[0].physical_payload_sha256,
                    "output": selected[1].physical_payload_sha256,
                    "config_sha256": config_sha,
                }
            )
            option_id = (
                f"{value_name.strip()}+{output_name.strip()}:"
                f"{quantizer.option_suffix}:p{propagation:+d}:"
                f"vw{objective_weight_ppm}"
            )
            option_sha = _sha256_canonical(
                {
                    "option_id": option_id,
                    "evidence_bundle_sha256": evidence_bundle_sha,
                    "physical_bundle_sha256": physical_bundle_sha,
                    "selected_method": selected_method,
                    "selected_joint_objective": selected[3],
                }
            )
            total_physical = (
                selected[0].physical_record_bytes + selected[1].physical_record_bytes
            )
            total_dense = selected[0].dense_bf16_bytes + selected[1].dense_bf16_bytes
            output_selected = selected[3] - config.value_objective_weight * selected[2]
            options.append(
                Lfm2JointAttentionOption(
                    option_id=option_id,
                    config=config,
                    value_payload=selected[0],
                    output_payload=selected[1],
                    selected_method=selected_method,
                    current_joint_objective=current_joint,
                    rtn_joint_objective=rtn[3],
                    raw_successive_joint_objective=successive[3],
                    selected_joint_objective=selected[3],
                    value_selected_objective=selected[2],
                    output_selected_objective=output_selected,
                    total_physical_record_bytes=total_physical,
                    total_dense_bf16_bytes=total_dense,
                    total_physical_byte_savings=total_dense - total_physical,
                    evidence_bundle_sha256=evidence_bundle_sha,
                    config_sha256=config_sha,
                    physical_bundle_sha256=physical_bundle_sha,
                    option_sha256=option_sha,
                    ops_audit=audit,
                )
            )
    options.sort(key=lambda item: item.option_id)
    eligible = [
        item
        for item in options
        if max_bundle_physical_bytes is None
        or item.total_physical_record_bytes <= max_bundle_physical_bytes
    ]
    if not eligible:
        raise Lfm2CachedSecondOrderError(
            "No joint V/O payload fits max_bundle_physical_bytes."
        )
    best = min(
        eligible,
        key=lambda item: (
            item.selected_joint_objective,
            item.total_physical_record_bytes,
            item.option_id,
        ),
    )
    _assert_source_unchanged(
        value_weight,
        value_source_sha,
        "value_weight",
    )
    _assert_source_unchanged(
        output_weight,
        output_source_sha,
        "output_weight",
    )
    verify_lfm2_cached_second_order_evidence(value_evidence)
    verify_lfm2_cached_second_order_evidence(output_evidence)
    verify_lfm2_attention_bridge(attention_bridge)
    source_bundle_sha = _sha256_canonical(
        {"value": value_source_sha, "output": output_source_sha}
    )
    evidence_bundle_sha = _sha256_canonical(
        {
            "value": value_evidence.evidence_sha256,
            "output": output_evidence.evidence_sha256,
            "attention_bridge": attention_bridge.bridge_sha256,
        }
    )
    grid_sha = _sha256_canonical(
        {
            "evidence_bundle_sha256": evidence_bundle_sha,
            "source_bundle_sha256": source_bundle_sha,
            "option_sha256": tuple(item.option_sha256 for item in options),
            "max_bundle_physical_bytes": max_bundle_physical_bytes,
            "best_option": best.option_id,
        }
    )
    return Lfm2JointAttentionResult(
        evidence_bundle_sha256=evidence_bundle_sha,
        source_bundle_sha256=source_bundle_sha,
        grid_sha256=grid_sha,
        current_joint_objective=current_joint,
        options=tuple(options),
        best_option=best,
        max_bundle_physical_bytes=max_bundle_physical_bytes,
        ops_audit=audit,
    )


__all__ = [
    "Lfm2CachedSecondOrderError",
    "Lfm2CachedSecondOrderEvidence",
    "Lfm2CrevfSolvedTensor",
    "Lfm2JointAttentionConfig",
    "Lfm2JointAttentionOption",
    "Lfm2JointAttentionResult",
    "Lfm2SealedAttentionBridge",
    "Lfm2SecondOrderGridResult",
    "Lfm2SecondOrderOpsAudit",
    "Lfm2SecondOrderQuantizedOption",
    "Lfm2SecondOrderQuantizerConfig",
    "Lfm2SecondOrderStatistics",
    "compute_lfm2_second_order_statistics",
    "seal_lfm2_attention_bridge",
    "seal_lfm2_cached_second_order_evidence",
    "solve_lfm2_crevf_second_order_grid",
    "solve_lfm2_joint_attention_vo",
    "verify_lfm2_attention_bridge",
    "verify_lfm2_cached_second_order_evidence",
]
