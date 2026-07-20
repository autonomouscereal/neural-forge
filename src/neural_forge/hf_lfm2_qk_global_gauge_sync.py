"""Fit-only global Q/K gauge and RMSNorm-radius synchronization.

The CPCF attention-dimension compiler intentionally exposes per-case balanced
score factors before fitting one shared child projection.  A general score
factorization is defined up to a full invertible bilinear gauge,
``Q -> QG`` and ``K -> K G^-T``.  Restricting that ambiguity to an orthogonal
rotation is only valid when both representations are equally balanced.
Orienting every case against a different local anchor therefore creates
mutually contradictory regression targets even when one dense child Q/K pair
exists.

This module closes that specific analytical gap without gradients, optimizer
state, model forwards, selection evidence, or validation evidence.  A fixed
number of analytical operations:

1. fits a globally coherent shared initialization from the sealed raw-linear
   source anchors around the supplied source-derived Q/K priors;
2. exactly replays that shared target-dtype Q/K pair;
3. jointly aligns each case/GQA group to that shared replay with a bounded
   full-rank bilinear gauge whose nullspace is completed toward identity;
4. maps the aligned post-RoPE factors into target RMSNorm directions;
5. solves positive informative-row RMSNorm radii in closed form while keeping
   numerically unidentifiable centered-score rows exactly zero; and
6. solves one global weighted ridge system around source-derived Q/K priors.

The result remains only a FIT compiler primitive.  Its error ladder is
diagnostic and cannot establish held-out routing quality or model coherence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (
    Lfm2CpcfAttentionDimensionFoldError,
    Lfm2CpcfAttentionFactorSufficientState,
    invert_lfm2_rope,
    lfm2_runtime_rms_norm,
)


_SCHEMA = "neural-forge-lfm2-qk-global-gauge-sync-v2"
_GAIN_DESIGN_CHUNK_ELEMENTS = 1_048_576
_SCORE_ACTIVITY_ROUNDOFF_MULTIPLIER = 1.0


class Lfm2QkGlobalGaugeSyncError(Lfm2CpcfAttentionDimensionFoldError):
    """Raised when the bounded analytical synchronization cannot be proven."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment failure
        raise Lfm2QkGlobalGaugeSyncError("PyTorch is required.") from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(name): _deep_freeze(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _deep_thaw(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2QkGlobalGaugeSyncError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _exact_float_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.dtype
        not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    ):
        raise Lfm2QkGlobalGaugeSyncError(
            f"{name} must be a nonempty dense real rank-{ndim} floating tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2QkGlobalGaugeSyncError(f"{name} contains NaN or infinity.")
    return result


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2QkGlobalGaugeSyncError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2QkGlobalGaugeSyncError(
            f"{name} must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2QkGlobalGaugeSyncError(f"{name} must be finite and positive.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2QkGlobalGaugeSyncError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2QkGlobalGaugeSyncError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2QkGlobalGaugeSyncError(f"{name} must be finite and nonnegative.")
    return result


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_rope(value: Any, cosine: Any, sine: Any) -> Any:
    return (
        value * cosine.unsqueeze(1) + _rotate_half(value) * sine.unsqueeze(1)
    ).contiguous()


def _state_bundle_sha256(
    states: Sequence[Lfm2CpcfAttentionFactorSufficientState],
) -> str:
    return _canonical_sha256([state.as_digest_dict() for state in states])


def _prior_bundle_sha256(query_prior: Any, key_prior: Any) -> str:
    return _canonical_sha256(
        {
            "source_query_weight_prior_sha256": _tensor_sha256(query_prior),
            "source_key_weight_prior_sha256": _tensor_sha256(key_prior),
        }
    )


def _runtime_bundle_sha256(
    query_norm_weight: Any,
    key_norm_weight: Any,
    query_template: Any,
    key_template: Any,
    norm_epsilon: float,
) -> str:
    return _canonical_sha256(
        {
            "target_query_norm_weight_sha256": _tensor_sha256(query_norm_weight),
            "target_key_norm_weight_sha256": _tensor_sha256(key_norm_weight),
            "target_query_weight_template_sha256": _tensor_sha256(query_template),
            "target_key_weight_template_sha256": _tensor_sha256(key_template),
            "norm_epsilon": norm_epsilon,
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2QkGlobalGaugeSyncPolicy:
    """Fixed ceilings and analytical regularizers for the FIT-only solve."""

    sweeps: int = 3
    source_anchor_prior_ridge: float = 1e-10
    global_prior_ridge: float = 1e-10
    radius_prior_strength: float = 1e-8
    gauge_map_ridge: float = 1e-10
    gauge_query_reconciliation_fraction: float = 0.5
    maximum_gauge_condition_number: float = 4096.0
    minimum_radius: float = 2.0**-24
    maximum_radius: float = 2.0**16
    gauge_rank_relative_tolerance: float = 1e-10
    maximum_fit_cases: int = 256
    maximum_total_rows: int = 262_144
    maximum_total_score_elements: int = 16_777_216
    maximum_total_factor_elements: int = 67_108_864
    maximum_total_output_elements: int = 67_108_864
    maximum_total_svd_workspace_elements: int = 16_777_216
    maximum_normal_equation_elements: int = 16_777_216

    def __post_init__(self) -> None:
        if isinstance(self.sweeps, bool) or self.sweeps not in {2, 3}:
            raise Lfm2QkGlobalGaugeSyncError("sweeps must be exactly 2 or 3.")
        for name in (
            "source_anchor_prior_ridge",
            "global_prior_ridge",
            "radius_prior_strength",
            "gauge_map_ridge",
            "gauge_query_reconciliation_fraction",
            "gauge_rank_relative_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        if self.gauge_map_ridge == 0.0:
            raise Lfm2QkGlobalGaugeSyncError(
                "gauge_map_ridge must be positive for deterministic nullspace "
                "completion."
            )
        if self.gauge_query_reconciliation_fraction > 1.0:
            raise Lfm2QkGlobalGaugeSyncError(
                "gauge_query_reconciliation_fraction must not exceed one."
            )
        for name in ("minimum_radius", "maximum_radius"):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), name),
            )
        if self.minimum_radius >= self.maximum_radius:
            raise Lfm2QkGlobalGaugeSyncError(
                "minimum_radius must be smaller than maximum_radius."
            )
        object.__setattr__(
            self,
            "maximum_gauge_condition_number",
            _finite_positive(
                self.maximum_gauge_condition_number,
                "maximum_gauge_condition_number",
            ),
        )
        if self.maximum_gauge_condition_number < 1.0:
            raise Lfm2QkGlobalGaugeSyncError(
                "maximum_gauge_condition_number must be at least one."
            )
        for name in (
            "maximum_fit_cases",
            "maximum_total_rows",
            "maximum_total_score_elements",
            "maximum_total_factor_elements",
            "maximum_total_output_elements",
            "maximum_total_svd_workspace_elements",
            "maximum_normal_equation_elements",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Lfm2QkGlobalGaugeSyncError(f"{name} must be a positive integer.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sweeps": self.sweeps,
            "source_anchor_prior_ridge": self.source_anchor_prior_ridge,
            "global_prior_ridge": self.global_prior_ridge,
            "radius_prior_strength": self.radius_prior_strength,
            "gauge_map_ridge": self.gauge_map_ridge,
            "gauge_query_reconciliation_fraction": (
                self.gauge_query_reconciliation_fraction
            ),
            "maximum_gauge_condition_number": (self.maximum_gauge_condition_number),
            "minimum_radius": self.minimum_radius,
            "maximum_radius": self.maximum_radius,
            "gauge_rank_relative_tolerance": self.gauge_rank_relative_tolerance,
            "maximum_fit_cases": self.maximum_fit_cases,
            "maximum_total_rows": self.maximum_total_rows,
            "maximum_total_score_elements": self.maximum_total_score_elements,
            "maximum_total_factor_elements": self.maximum_total_factor_elements,
            "maximum_total_output_elements": self.maximum_total_output_elements,
            "maximum_total_svd_workspace_elements": (
                self.maximum_total_svd_workspace_elements
            ),
            "maximum_normal_equation_elements": (self.maximum_normal_equation_elements),
        }


@dataclass(frozen=True, slots=True)
class Lfm2QkGlobalGaugeSyncProvenance:
    """Seal binding the literal FIT factors, source priors, and target runtime."""

    fit_receipt_sha256: str
    source_runtime_receipt_sha256: str
    factor_state_bundle_sha256: str
    source_prior_bundle_sha256: str
    target_runtime_bundle_sha256: str
    provenance_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "fit_receipt_sha256",
            "source_runtime_receipt_sha256",
            "factor_state_bundle_sha256",
            "source_prior_bundle_sha256",
            "target_runtime_bundle_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _required_sha256(getattr(self, name), name),
            )
        body = self.body()
        computed = _canonical_sha256(body)
        if self.provenance_sha256 and self.provenance_sha256 != computed:
            raise Lfm2QkGlobalGaugeSyncError(
                "Global Q/K synchronization provenance seal is invalid."
            )
        object.__setattr__(self, "provenance_sha256", computed)

    def body(self) -> dict[str, str]:
        return {
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "source_runtime_receipt_sha256": (self.source_runtime_receipt_sha256),
            "factor_state_bundle_sha256": self.factor_state_bundle_sha256,
            "source_prior_bundle_sha256": self.source_prior_bundle_sha256,
            "target_runtime_bundle_sha256": self.target_runtime_bundle_sha256,
        }

    def as_dict(self) -> dict[str, str]:
        return {**self.body(), "provenance_sha256": self.provenance_sha256}

    def verify(
        self,
        *,
        states: Sequence[Lfm2CpcfAttentionFactorSufficientState],
        query_prior: Any,
        key_prior: Any,
        query_norm_weight: Any,
        key_norm_weight: Any,
        query_template: Any,
        key_template: Any,
        norm_epsilon: float,
    ) -> str:
        if (
            self.factor_state_bundle_sha256 != _state_bundle_sha256(states)
            or self.source_prior_bundle_sha256
            != _prior_bundle_sha256(query_prior, key_prior)
            or self.target_runtime_bundle_sha256
            != _runtime_bundle_sha256(
                query_norm_weight,
                key_norm_weight,
                query_template,
                key_template,
                norm_epsilon,
            )
            or _canonical_sha256(self.body()) != self.provenance_sha256
        ):
            raise Lfm2QkGlobalGaugeSyncError(
                "Global Q/K synchronization inputs do not match sealed provenance."
            )
        return self.provenance_sha256


def seal_lfm2_qk_global_gauge_sync_provenance(
    *,
    fit_receipt_sha256: str,
    source_runtime_receipt_sha256: str,
    factor_states: Sequence[Lfm2CpcfAttentionFactorSufficientState],
    source_query_weight_prior: Any,
    source_key_weight_prior: Any,
    target_query_norm_weight: Any,
    target_key_norm_weight: Any,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    norm_epsilon: float,
) -> Lfm2QkGlobalGaugeSyncProvenance:
    """Create the typed seal required by the analytical compiler."""

    states = tuple(factor_states)
    if len(states) < 2:
        raise Lfm2QkGlobalGaugeSyncError(
            "At least two FIT factor states are required for global synchronization."
        )
    epsilon = _finite_positive(norm_epsilon, "norm_epsilon")
    return Lfm2QkGlobalGaugeSyncProvenance(
        fit_receipt_sha256=_required_sha256(
            fit_receipt_sha256,
            "fit_receipt_sha256",
        ),
        source_runtime_receipt_sha256=_required_sha256(
            source_runtime_receipt_sha256,
            "source_runtime_receipt_sha256",
        ),
        factor_state_bundle_sha256=_state_bundle_sha256(states),
        source_prior_bundle_sha256=_prior_bundle_sha256(
            source_query_weight_prior,
            source_key_weight_prior,
        ),
        target_runtime_bundle_sha256=_runtime_bundle_sha256(
            target_query_norm_weight,
            target_key_norm_weight,
            target_query_weight_template,
            target_key_weight_template,
            epsilon,
        ),
    )


@dataclass(frozen=True, slots=True)
class _Geometry:
    query_heads: int
    key_heads: int
    head_dim: int
    hidden_width: int
    target_dtype: Any
    total_rows: int
    total_score_elements: int
    total_factor_elements: int
    total_output_elements: int
    total_svd_workspace_elements: int

    @property
    def queries_per_key(self) -> int:
        return self.query_heads // self.key_heads


@dataclass(frozen=True, slots=True)
class _Replay:
    query_raw: Any
    key_raw: Any
    query_post: Any
    key_post: Any


@dataclass(frozen=True, slots=True)
class _SynchronizedCase:
    aligned_query_post: Any
    aligned_key_post: Any
    query_raw_target: Any
    key_raw_target: Any
    norm_image_query_post: Any
    norm_image_key_post: Any
    gauge_records: tuple[Mapping[str, Any], ...]
    radius_record: Mapping[str, Any]


def _validate_inputs(
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    query_norm: Any,
    key_norm: Any,
    query_template: Any,
    key_template: Any,
    query_prior: Any,
    key_prior: Any,
    policy: Lfm2QkGlobalGaugeSyncPolicy,
) -> _Geometry:
    torch = _torch()
    if len(states) < 2:
        raise Lfm2QkGlobalGaugeSyncError(
            "At least two FIT states are required for global synchronization."
        )
    if len(states) > policy.maximum_fit_cases:
        raise Lfm2QkGlobalGaugeSyncError("FIT state count exceeds the hard ceiling.")
    if not all(
        isinstance(state, Lfm2CpcfAttentionFactorSufficientState) for state in states
    ):
        raise TypeError(
            "factor_states must contain Lfm2CpcfAttentionFactorSufficientState."
        )
    if any(state.partition != "fit" for state in states):
        raise Lfm2QkGlobalGaugeSyncError(
            "The synchronization primitive accepts literal FIT states only."
        )
    ordinals = [state.case_ordinal for state in states]
    if len(set(ordinals)) != len(ordinals):
        raise Lfm2QkGlobalGaugeSyncError(
            "FIT factor-state case ordinals must be unique."
        )
    evidence_receipts = [state.evidence_sha256 for state in states]
    if len(set(evidence_receipts)) != len(evidence_receipts):
        raise Lfm2QkGlobalGaugeSyncError(
            "FIT factor-state evidence receipts must be unique."
        )
    first = states[0]
    query_heads = int(first.local_query_post_rope_factor.shape[1])
    key_heads = int(first.local_key_post_rope_factor.shape[1])
    head_dim = int(first.local_query_post_rope_factor.shape[-1])
    hidden_width = int(first.target_query_inputs.shape[-1])
    if query_heads <= 0 or key_heads <= 0 or query_heads % key_heads:
        raise Lfm2QkGlobalGaugeSyncError(
            "Q/K heads must define a positive integral GQA ratio."
        )
    queries_per_key = query_heads // key_heads
    if head_dim <= 0 or head_dim % 2:
        raise Lfm2QkGlobalGaugeSyncError(
            "The synchronized target head dimension must be positive and even."
        )
    expected_q_weight = (query_heads * head_dim, hidden_width)
    expected_k_weight = (key_heads * head_dim, hidden_width)
    if (
        tuple(query_norm.shape) != (head_dim,)
        or tuple(key_norm.shape) != (head_dim,)
        or tuple(query_template.shape) != expected_q_weight
        or tuple(query_prior.shape) != expected_q_weight
        or tuple(key_template.shape) != expected_k_weight
        or tuple(key_prior.shape) != expected_k_weight
    ):
        raise Lfm2QkGlobalGaugeSyncError(
            "Norm, template, prior, factor, and child-input geometry disagree."
        )
    if (
        query_template.dtype != torch.bfloat16
        or key_template.dtype != torch.bfloat16
        or query_prior.dtype != torch.bfloat16
        or key_prior.dtype != torch.bfloat16
        or query_norm.dtype != torch.bfloat16
        or key_norm.dtype != torch.bfloat16
    ):
        raise Lfm2QkGlobalGaugeSyncError(
            "Version 1 requires exact BF16 target templates, priors, and Q/K norms."
        )
    if bool(
        (query_norm.to(torch.float64).abs() <= torch.finfo(torch.float64).tiny)
        .any()
        .item()
    ) or bool(
        (key_norm.to(torch.float64).abs() <= torch.finfo(torch.float64).tiny)
        .any()
        .item()
    ):
        raise Lfm2QkGlobalGaugeSyncError("Target Q/K RMSNorm weights must be nonzero.")
    total_rows = 0
    total_score_elements = 0
    total_factor_elements = 0
    total_output_elements = 0
    total_svd_workspace_elements = 0
    for index, state in enumerate(states):
        query = state.local_query_post_rope_factor
        key = state.local_key_post_rope_factor
        batch, state_q_heads, query_length, state_dim = map(int, query.shape)
        key_batch, state_k_heads, key_length, key_dim = map(int, key.shape)
        total_rows += batch * (query_length + key_length)
        total_score_elements += batch * query_heads * query_length * key_length
        state_factor_elements = int(query.numel()) + int(key.numel())
        total_factor_elements += state_factor_elements
        # Raw, aligned, normalized-image, and exact replay factor states are
        # materialized during a sweep.
        total_output_elements += 4 * state_factor_elements
        total_svd_workspace_elements += (
            batch
            * key_heads
            * (
                6 * head_dim * head_dim
                + (queries_per_key * query_length + key_length) * head_dim
            )
        )
        if (
            batch <= 0
            or query_length <= 0
            or key_length <= 0
            or (state_q_heads, state_k_heads, state_dim, key_dim)
            != (query_heads, key_heads, head_dim, head_dim)
            or key_batch != batch
            or tuple(state.query_post_rope_anchor.shape) != tuple(query.shape)
            or tuple(state.key_post_rope_anchor.shape) != tuple(key.shape)
            or tuple(state.query_raw_linear_anchor.shape) != tuple(query.shape)
            or tuple(state.key_raw_linear_anchor.shape) != tuple(key.shape)
            or tuple(state.target_query_inputs.shape)
            != (batch, query_length, hidden_width)
            or tuple(state.target_key_inputs.shape) != (batch, key_length, hidden_width)
            or tuple(state.query_row_sqrt_weights.shape) != (batch, query_length)
            or tuple(state.key_row_sqrt_weights.shape) != (batch, key_length)
            or tuple(state.target_query_rope_cos.shape)
            != (batch, query_length, head_dim)
            or tuple(state.target_query_rope_sin.shape)
            != (batch, query_length, head_dim)
            or tuple(state.target_key_rope_cos.shape) != (batch, key_length, head_dim)
            or tuple(state.target_key_rope_sin.shape) != (batch, key_length, head_dim)
        ):
            raise Lfm2QkGlobalGaugeSyncError(
                f"factor_states[{index}] has inconsistent Q/K geometry."
            )
        exact_runtime_values = (
            state.target_query_inputs,
            state.target_key_inputs,
            state.target_query_rope_cos,
            state.target_query_rope_sin,
            state.target_key_rope_cos,
            state.target_key_rope_sin,
        )
        if any(value.dtype != torch.bfloat16 for value in exact_runtime_values):
            raise Lfm2QkGlobalGaugeSyncError(
                "Exact child inputs and target RoPE terms must be BF16."
            )
        for role, weights in (
            ("query", state.query_row_sqrt_weights),
            ("key", state.key_row_sqrt_weights),
        ):
            weights64 = weights.to(torch.float64)
            if bool((weights64 < 0.0).any().item()) or not bool(
                (weights64 > 0.0).any().item()
            ):
                raise Lfm2QkGlobalGaugeSyncError(
                    f"factor_states[{index}] {role} weights must be nonnegative "
                    "with positive mass."
                )
    if total_rows > policy.maximum_total_rows:
        raise Lfm2QkGlobalGaugeSyncError("FIT rows exceed the hard ceiling.")
    if total_score_elements > policy.maximum_total_score_elements:
        raise Lfm2QkGlobalGaugeSyncError("FIT score elements exceed the hard ceiling.")
    if total_factor_elements > policy.maximum_total_factor_elements:
        raise Lfm2QkGlobalGaugeSyncError("FIT factor elements exceed the hard ceiling.")
    if total_output_elements > policy.maximum_total_output_elements:
        raise Lfm2QkGlobalGaugeSyncError(
            "FIT analytical output elements exceed the hard ceiling."
        )
    if total_svd_workspace_elements > policy.maximum_total_svd_workspace_elements:
        raise Lfm2QkGlobalGaugeSyncError("FIT SVD workspace exceeds the hard ceiling.")
    gauge_parameter_count = head_dim * (head_dim + 1) // 2
    if (
        hidden_width * hidden_width > policy.maximum_normal_equation_elements
        or gauge_parameter_count * gauge_parameter_count
        > policy.maximum_normal_equation_elements
    ):
        raise Lfm2QkGlobalGaugeSyncError(
            "Global projection or symmetric-gauge normal equation exceeds the "
            "hard element ceiling."
        )
    return _Geometry(
        query_heads=query_heads,
        key_heads=key_heads,
        head_dim=head_dim,
        hidden_width=hidden_width,
        target_dtype=torch.bfloat16,
        total_rows=total_rows,
        total_score_elements=total_score_elements,
        total_factor_elements=total_factor_elements,
        total_output_elements=total_output_elements,
        total_svd_workspace_elements=total_svd_workspace_elements,
    )


def _exact_replay(
    state: Lfm2CpcfAttentionFactorSufficientState,
    query_weight: Any,
    key_weight: Any,
    query_norm: Any,
    key_norm: Any,
    epsilon: float,
    geometry: _Geometry,
) -> _Replay:
    dtype = geometry.target_dtype
    batch, query_length, _hidden = map(int, state.target_query_inputs.shape)
    key_length = int(state.target_key_inputs.shape[1])
    query_raw = (
        (state.target_query_inputs.to(dtype=dtype) @ query_weight.to(dtype=dtype).T)
        .reshape(
            batch,
            query_length,
            geometry.query_heads,
            geometry.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_raw = (
        (state.target_key_inputs.to(dtype=dtype) @ key_weight.to(dtype=dtype).T)
        .reshape(
            batch,
            key_length,
            geometry.key_heads,
            geometry.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    query_pre = lfm2_runtime_rms_norm(query_raw, query_norm, epsilon)
    key_pre = lfm2_runtime_rms_norm(key_raw, key_norm, epsilon)
    return _Replay(
        query_raw=query_raw,
        key_raw=key_raw,
        query_post=_apply_rope(
            query_pre,
            state.target_query_rope_cos,
            state.target_query_rope_sin,
        ),
        key_post=_apply_rope(
            key_pre,
            state.target_key_rope_cos,
            state.target_key_rope_sin,
        ),
    )


def _continuous_replay(
    state: Lfm2CpcfAttentionFactorSufficientState,
    query_weight: Any,
    key_weight: Any,
    query_norm: Any,
    key_norm: Any,
    epsilon: float,
    geometry: _Geometry,
) -> _Replay:
    torch = _torch()
    batch, query_length, _hidden = map(int, state.target_query_inputs.shape)
    key_length = int(state.target_key_inputs.shape[1])
    query_raw = (
        (state.target_query_inputs.to(torch.float64) @ query_weight.to(torch.float64).T)
        .reshape(
            batch,
            query_length,
            geometry.query_heads,
            geometry.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_raw = (
        (state.target_key_inputs.to(torch.float64) @ key_weight.to(torch.float64).T)
        .reshape(
            batch,
            key_length,
            geometry.key_heads,
            geometry.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    def normalize(raw: Any, norm: Any) -> Any:
        variance = raw.square().mean(dim=-1, keepdim=True)
        return (
            raw
            * torch.rsqrt(variance + epsilon)
            * norm.to(torch.float64).reshape(1, 1, 1, -1)
        )

    query_pre = normalize(query_raw, query_norm)
    key_pre = normalize(key_raw, key_norm)
    return _Replay(
        query_raw=query_raw,
        key_raw=key_raw,
        query_post=_apply_rope(
            query_pre,
            state.target_query_rope_cos.to(torch.float64),
            state.target_query_rope_sin.to(torch.float64),
        ),
        key_post=_apply_rope(
            key_pre,
            state.target_key_rope_cos.to(torch.float64),
            state.target_key_rope_sin.to(torch.float64),
        ),
    )


def _weighted_ridge_map_to_shared_replay(
    source_query: Any,
    target_query: Any,
    query_sqrt_weights: Any,
    *,
    ridge: float,
    role: str,
) -> Any:
    torch = _torch()
    source = source_query.to(torch.float64)
    target = target_query.to(torch.float64)
    weights = query_sqrt_weights.to(torch.float64).reshape(-1, 1)
    dimension = int(source.shape[1])
    weighted_source = source * weights
    weighted_target = target * weights
    identity = torch.eye(dimension, dtype=torch.float64)
    gram = weighted_source.T @ weighted_source
    right = weighted_source.T @ weighted_target
    try:
        result = torch.linalg.solve(
            gram + ridge * identity,
            right + ridge * identity,
        )
    except RuntimeError as exc:
        raise Lfm2QkGlobalGaugeSyncError(
            f"{role} bilinear gauge ridge solve failed."
        ) from exc
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2QkGlobalGaugeSyncError(
            f"{role} bilinear gauge ridge solve produced non-finite state."
        )
    return result.contiguous()


def _condition_bounded_invertible(
    value: Any,
    *,
    maximum_condition_number: float,
) -> tuple[Any, dict[str, Any]]:
    """Return a deterministic invertible GL(d) map with bounded singular values."""

    torch = _torch()
    matrix = value.to(torch.float64)
    left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
    limit = math.sqrt(maximum_condition_number)
    lower = 1.0 / limit
    upper = limit
    bounded = singular.clamp(min=lower, max=upper)
    result = (left @ torch.diag(bounded) @ right_h).contiguous()
    condition = float((bounded.max() / bounded.min()).item())
    return result, {
        "input_singular_values": [float(item.item()) for item in singular],
        "bounded_singular_values": [float(item.item()) for item in bounded],
        "lower_singular_bound": lower,
        "upper_singular_bound": upper,
        "singular_value_clamp_count": int((bounded != singular).sum().item()),
        "condition_number": condition,
    }


def _divided_rmsnorm_rows(
    post_rows: Any,
    norm_weight: Any,
    cosine_rows: Any,
    sine_rows: Any,
) -> Any:
    rows = post_rows.to(_torch().float64)
    length, dimension = map(int, rows.shape)
    unrotated = invert_lfm2_rope(
        rows.reshape(1, 1, length, dimension),
        cosine_rows.to(_torch().float64).reshape(1, length, dimension),
        sine_rows.to(_torch().float64).reshape(1, length, dimension),
    )
    return (
        unrotated.reshape(length, dimension)
        / norm_weight.to(_torch().float64).reshape(1, dimension)
    ).contiguous()


def _divided_rmsnorm_adjoint_rows(
    divided_rows: Any,
    norm_weight: Any,
    cosine_rows: Any,
    sine_rows: Any,
) -> Any:
    """Apply the adjoint of :func:`_divided_rmsnorm_rows` row-wise.

    Materializing one dense ``d x d`` inverse-RoPE/RMSNorm map per factor row
    makes an otherwise bounded FIT state require ``O(rows * d**2)`` FP64
    storage.  RoPE's half-rotation is skew-adjoint, so the same adjoint is
    evaluated exactly with elementwise operations in ``O(rows * d)`` storage.
    """

    torch = _torch()
    rows = divided_rows.to(torch.float64)
    cosine = cosine_rows.to(torch.float64)
    sine = sine_rows.to(torch.float64)
    rope_radius = (cosine.square() + sine.square()).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    scaled = rows / norm_weight.to(torch.float64).reshape(1, -1) / rope_radius
    return (scaled * cosine + _rotate_half(scaled * sine)).contiguous()


def _gauge_invariant_score_activity(
    query_rows: Any,
    key_rows: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Identify numerically resolvable score rows and columns.

    CPCF factors are produced *after* an FP64 SVD of centered scores.  An
    exactly zero centered-score row can consequently contain nonzero factor
    dust at machine precision.  Testing factor entries for exact zero is both
    production-incorrect and GL-gauge-dependent.  The product ``Q @ K.T`` is
    gauge invariant, so activity is instead decided in score space using the
    standard matrix-size-scaled FP64 roundoff floor.  No fixed absolute floor
    is used: uniformly tiny but resolvable score matrices retain their rows.

    A one-key centered-score group has no identifiable score contrast by
    construction and is therefore always inactive, even if an upstream SVD
    emits roundoff dust.
    """

    torch = _torch()
    query = query_rows.to(torch.float64)
    key = key_rows.to(torch.float64)
    scores = query @ key.T
    query_count, key_count = map(int, scores.shape)
    dimension = int(query.shape[1])
    absolute_scores = scores.abs()
    score_scale = float(absolute_scores.max().item())
    comparison_dimension = max(query_count, key_count, dimension)
    tolerance = (
        _SCORE_ACTIVITY_ROUNDOFF_MULTIPLIER
        * float(torch.finfo(torch.float64).eps)
        * comparison_dimension
        * score_scale
    )
    if key_count == 1:
        query_activity = torch.zeros(
            (query_count,),
            dtype=torch.bool,
        )
        key_activity = torch.zeros(
            (key_count,),
            dtype=torch.bool,
        )
        rule = "one_key_centered_score_has_no_identifiable_contrast"
    else:
        query_activity = absolute_scores.amax(dim=-1) > tolerance
        key_activity = absolute_scores.amax(dim=0) > tolerance
        rule = "gauge_invariant_score_max_exceeds_scaled_fp64_roundoff"
    return (
        query_activity,
        key_activity,
        {
            "score_maximum_absolute_value": score_scale,
            "score_activity_absolute_tolerance": tolerance,
            "score_activity_relative_roundoff_multiplier": (
                _SCORE_ACTIVITY_ROUNDOFF_MULTIPLIER
            ),
            "score_activity_comparison_dimension": comparison_dimension,
            "score_activity_rule": rule,
            "fixed_absolute_score_floor_used": False,
        },
    )


def _rmsnorm_image_balanced_gauge(
    source_query: Any,
    source_key: Any,
    gauge: Any,
    query_norm: Any,
    key_norm: Any,
    query_cosine_rows: Any,
    query_sine_rows: Any,
    key_cosine_rows: Any,
    key_sine_rows: Any,
    query_sqrt_weights: Any,
    key_sqrt_weights: Any,
    *,
    maximum_condition_number: float,
) -> tuple[Any, list[dict[str, Any]]]:
    """Synchronize every gauge-row gain with the target RMSNorm image.

    A fixed three-step regularized solve acts only on the head-dimensional
    symmetric GL gauge.  It linearizes ``log(RMS(inverse_rope(row)/norm))``
    around the current ``G``/``G^-T`` pair and solves the minimum-norm
    symmetric update in the dual when rows are fewer than gauge parameters.
    No target projection weights, gradients, backwards, or optimizer state
    participate.  Applying reciprocal matrix exponentials preserves the
    local QK score product at every step.
    """

    torch = _torch()
    current = gauge.to(torch.float64)
    best = current
    best_objective = float("inf")
    best_step = 0
    frontier: list[dict[str, Any]] = []
    query_weights = query_sqrt_weights.to(torch.float64).reshape(-1)
    key_weights = key_sqrt_weights.to(torch.float64).reshape(-1)
    query_informative, key_informative, score_activity = (
        _gauge_invariant_score_activity(source_query, source_key)
    )
    dimension = int(current.shape[0])
    upper_i, upper_j = torch.triu_indices(dimension, dimension)
    off_diagonal = upper_i != upper_j
    parameter_count = int(upper_i.numel())
    for step in range(4):
        inverse_transpose = torch.linalg.inv(current).T
        aligned_query = source_query.to(torch.float64) @ current
        aligned_key = source_key.to(torch.float64) @ inverse_transpose
        divided_query = _divided_rmsnorm_rows(
            aligned_query,
            query_norm,
            query_cosine_rows,
            query_sine_rows,
        )
        divided_key = _divided_rmsnorm_rows(
            aligned_key,
            key_norm,
            key_cosine_rows,
            key_sine_rows,
        )
        query_gain = torch.sqrt(divided_query.square().mean(dim=-1))
        key_gain = torch.sqrt(divided_key.square().mean(dim=-1))
        positive_query_gain = query_gain[query_informative]
        positive_key_gain = key_gain[key_informative]
        joint_positive_gain = torch.cat(
            (positive_query_gain, positive_key_gain),
            dim=0,
        )
        minimum_query_gain = (
            float(positive_query_gain.min().item())
            if positive_query_gain.numel()
            else 0.0
        )
        maximum_query_gain = (
            float(positive_query_gain.max().item())
            if positive_query_gain.numel()
            else 0.0
        )
        minimum_key_gain = (
            float(positive_key_gain.min().item()) if positive_key_gain.numel() else 0.0
        )
        maximum_key_gain = (
            float(positive_key_gain.max().item()) if positive_key_gain.numel() else 0.0
        )
        maximum_log_gain_error = (
            float(joint_positive_gain.log().abs().max().item())
            if joint_positive_gain.numel()
            else 0.0
        )
        objective = maximum_log_gain_error
        if objective < best_objective:
            best_objective = objective
            best = current
            best_step = step
        frontier.append(
            {
                "step": step,
                "minimum_query_divided_rms": minimum_query_gain,
                "maximum_query_divided_rms": maximum_query_gain,
                "minimum_key_divided_rms": minimum_key_gain,
                "maximum_key_divided_rms": maximum_key_gain,
                "maximum_joint_divided_rms": max(
                    maximum_query_gain,
                    maximum_key_gain,
                ),
                "maximum_absolute_log_gain_error": maximum_log_gain_error,
                "query_factor_row_count": int(query_gain.numel()),
                "key_factor_row_count": int(key_gain.numel()),
                "informative_query_factor_row_count": int(
                    query_informative.sum().item()
                ),
                "informative_key_factor_row_count": int(key_informative.sum().item()),
                "zero_or_score_uninformative_query_factor_row_count": int(
                    query_gain.numel() - query_informative.sum().item()
                ),
                "zero_or_score_uninformative_key_factor_row_count": int(
                    key_gain.numel() - key_informative.sum().item()
                ),
                "zero_factor_rows_excluded_from_log_gain_solve": True,
                "score_uninformative_rows_excluded_from_log_gain_solve": True,
                "score_activity_is_full_gl_gauge_invariant": True,
                "score_activity": score_activity,
                "gain_design_solver_executed": False,
                "gain_design_streamed": False,
                "gain_design_peak_elements": 0,
                "primal_gain_design_chunk_element_ceiling": (
                    _GAIN_DESIGN_CHUNK_ELEMENTS
                ),
                "gauge_sha256": _tensor_sha256(current),
            }
        )
        if step == 3 or not joint_positive_gain.numel():
            break

        def gain_design(
            aligned: Any,
            divided: Any,
            norm: Any,
            cosine: Any,
            sine: Any,
            *,
            sign: float,
        ) -> Any:
            adjoint = _divided_rmsnorm_adjoint_rows(
                divided,
                norm,
                cosine,
                sine,
            )
            denominator = (
                divided.square()
                .sum(dim=-1, keepdim=True)
                .clamp_min(torch.finfo(torch.float64).tiny)
            )
            design = aligned[:, upper_i] * adjoint[:, upper_j]
            design[:, off_diagonal] += (
                aligned[:, upper_j[off_diagonal]] * adjoint[:, upper_i[off_diagonal]]
            )
            return sign * design / denominator

        design_inputs = (
            (
                aligned_query[query_informative],
                divided_query[query_informative],
                query_norm,
                query_cosine_rows[query_informative],
                query_sine_rows[query_informative],
                query_weights[query_informative],
                positive_query_gain,
                1.0,
            ),
            (
                aligned_key[key_informative],
                divided_key[key_informative],
                key_norm,
                key_cosine_rows[key_informative],
                key_sine_rows[key_informative],
                key_weights[key_informative],
                positive_key_gain,
                -1.0,
            ),
        )
        informative_row_count = int(joint_positive_gain.numel())
        ridge = 1e-6
        materialize_design = (
            informative_row_count <= parameter_count
            or informative_row_count * parameter_count <= _GAIN_DESIGN_CHUNK_ELEMENTS
        )
        if materialize_design:
            designs: list[Any] = []
            targets: list[Any] = []
            for (
                aligned,
                divided,
                norm,
                cosine,
                sine,
                weights,
                gains,
                sign,
            ) in design_inputs:
                designs.append(
                    gain_design(
                        aligned,
                        divided,
                        norm,
                        cosine,
                        sine,
                        sign=sign,
                    )
                    * weights.reshape(-1, 1)
                )
                targets.append(-gains.log() * weights)
            design = torch.cat(designs, dim=0)
            target = torch.cat(targets, dim=0)
            peak_design_elements = int(design.numel())
        else:
            design = None
            target = None
            peak_design_elements = 0
        if informative_row_count <= parameter_count:
            assert design is not None and target is not None
            dual_identity = torch.eye(int(design.shape[0]), dtype=torch.float64)
            parameters = design.T @ torch.linalg.solve(
                design @ design.T + ridge * dual_identity,
                target,
            )
        elif design is not None and target is not None:
            parameter_identity = torch.eye(parameter_count, dtype=torch.float64)
            parameters = torch.linalg.solve(
                design.T @ design + ridge * parameter_identity,
                design.T @ target,
            )
        else:
            gram = torch.zeros(
                (parameter_count, parameter_count),
                dtype=torch.float64,
            )
            right = torch.zeros((parameter_count,), dtype=torch.float64)
            chunk_rows = max(
                1,
                _GAIN_DESIGN_CHUNK_ELEMENTS // parameter_count,
            )
            for (
                aligned,
                divided,
                norm,
                cosine,
                sine,
                weights,
                gains,
                sign,
            ) in design_inputs:
                for start in range(0, int(aligned.shape[0]), chunk_rows):
                    stop = min(start + chunk_rows, int(aligned.shape[0]))
                    chunk_weight = weights[start:stop]
                    design_chunk = gain_design(
                        aligned[start:stop],
                        divided[start:stop],
                        norm,
                        cosine[start:stop],
                        sine[start:stop],
                        sign=sign,
                    ) * chunk_weight.reshape(-1, 1)
                    target_chunk = -gains[start:stop].log() * chunk_weight
                    gram += design_chunk.T @ design_chunk
                    right += design_chunk.T @ target_chunk
                    peak_design_elements = max(
                        peak_design_elements,
                        int(design_chunk.numel()),
                    )
            parameter_identity = torch.eye(parameter_count, dtype=torch.float64)
            parameters = torch.linalg.solve(
                gram + ridge * parameter_identity,
                right,
            )
        frontier[-1].update(
            {
                "gain_design_solver_executed": True,
                "gain_design_streamed": not materialize_design,
                "gain_design_peak_elements": peak_design_elements,
            }
        )
        symmetric_update = torch.zeros(
            (dimension, dimension),
            dtype=torch.float64,
        )
        symmetric_update[upper_i, upper_j] = parameters
        symmetric_update[upper_j, upper_i] = parameters
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_update)
        bounded_eigenvalues = eigenvalues.clamp(min=-0.25, max=0.25)
        damped_update = (
            eigenvectors @ torch.diag(torch.exp(bounded_eigenvalues)) @ eigenvectors.T
        )
        current, _bound = _condition_bounded_invertible(
            current @ damped_update,
            maximum_condition_number=maximum_condition_number,
        )
    for record in frontier:
        record["selected"] = int(record["step"]) == best_step
    return best.contiguous(), frontier


def _bilinear_gauge_to_shared_replay(
    source_query: Any,
    source_key: Any,
    target_query: Any,
    target_key: Any,
    query_sqrt_weights: Any,
    key_sqrt_weights: Any,
    query_norm: Any,
    key_norm: Any,
    query_cosine_rows: Any,
    query_sine_rows: Any,
    key_cosine_rows: Any,
    key_sine_rows: Any,
    *,
    policy: Lfm2QkGlobalGaugeSyncPolicy,
) -> tuple[Any, Any, dict[str, Any]]:
    """Align one score factor with a bounded full GL(d) bilinear gauge.

    ``Q @ G`` and ``K @ G^-T`` preserve ``Q @ K.T`` exactly.  Independent
    weighted ridge maps estimate ``G`` from the query rows and ``G^-T`` from
    the key rows.  Their fixed arithmetic mean is projected onto a bounded
    invertible set.  Ridge-to-identity supplies a deterministic completion for
    directions not identified by short or rank-deficient cases.
    """

    torch = _torch()
    source_q = source_query.to(torch.float64)
    source_k = source_key.to(torch.float64)
    target_q = target_query.to(torch.float64)
    target_k = target_key.to(torch.float64)
    q_weight = query_sqrt_weights.to(torch.float64).reshape(-1, 1)
    k_weight = key_sqrt_weights.to(torch.float64).reshape(-1, 1)
    query_map = _weighted_ridge_map_to_shared_replay(
        source_q,
        target_q,
        q_weight,
        ridge=policy.gauge_map_ridge,
        role="query",
    )
    key_inverse_transpose_map = _weighted_ridge_map_to_shared_replay(
        source_k,
        target_k,
        k_weight,
        ridge=policy.gauge_map_ridge,
        role="key",
    )
    bounded_query_map, query_bound = _condition_bounded_invertible(
        query_map,
        maximum_condition_number=policy.maximum_gauge_condition_number,
    )
    bounded_key_map, key_bound = _condition_bounded_invertible(
        key_inverse_transpose_map,
        maximum_condition_number=policy.maximum_gauge_condition_number,
    )
    key_derived_query_map = torch.linalg.inv(bounded_key_map).T
    query_fraction = policy.gauge_query_reconciliation_fraction
    reconciled = (
        query_fraction * bounded_query_map
        + (1.0 - query_fraction) * key_derived_query_map
    )
    gauge, gauge_bound = _condition_bounded_invertible(
        reconciled,
        maximum_condition_number=policy.maximum_gauge_condition_number,
    )
    gauge, image_balance_frontier = _rmsnorm_image_balanced_gauge(
        source_q,
        source_k,
        gauge,
        query_norm,
        key_norm,
        query_cosine_rows,
        query_sine_rows,
        key_cosine_rows,
        key_sine_rows,
        q_weight,
        k_weight,
        maximum_condition_number=policy.maximum_gauge_condition_number,
    )
    gauge, gauge_bound = _condition_bounded_invertible(
        gauge,
        maximum_condition_number=policy.maximum_gauge_condition_number,
    )
    inverse_transpose = torch.linalg.inv(gauge).T.contiguous()
    aligned_q = (source_q @ gauge).contiguous()
    aligned_k = (source_k @ inverse_transpose).contiguous()

    weighted_rows = torch.cat((source_q * q_weight, source_k * k_weight), dim=0)
    row_singular = torch.linalg.svdvals(weighted_rows)
    row_maximum = float(row_singular.max().item()) if row_singular.numel() else 0.0
    row_threshold = row_maximum * policy.gauge_rank_relative_tolerance
    identifiable_rank = int((row_singular > row_threshold).sum().item())
    dimension = int(gauge.shape[0])
    residual_num = ((aligned_q - target_q) * q_weight).square().sum() + (
        (aligned_k - target_k) * k_weight
    ).square().sum()
    residual_den = (
        (target_q * q_weight).square().sum() + (target_k * k_weight).square().sum()
    ).clamp_min(torch.finfo(torch.float64).tiny)
    source_scores = source_q @ source_k.T
    aligned_scores = aligned_q @ aligned_k.T
    score_denominator = (
        source_scores.square().sum().clamp_min(torch.finfo(torch.float64).tiny)
    )
    score_invariance = float(
        torch.sqrt(
            (aligned_scores - source_scores).square().sum() / score_denominator
        ).item()
    )
    if not math.isfinite(score_invariance):
        raise Lfm2QkGlobalGaugeSyncError(
            "The bounded bilinear gauge produced non-finite score state."
        )
    record = {
        "gauge_family": "full_gl_bilinear_qg_k_inverse_transpose",
        "identifiable_singular_values": [float(value.item()) for value in row_singular],
        "identifiable_rank": identifiable_rank,
        "gauge_dimension": dimension,
        "full_rank_gauge_identified": identifiable_rank == dimension,
        "nullspace_dimension": dimension - identifiable_rank,
        "nullspace_completion": "weighted_ridge_to_identity",
        "gauge_reconciliation": "fixed_convex_query_key_map_reconciliation",
        "gauge_query_reconciliation_fraction": query_fraction,
        "query_map_bounding": query_bound,
        "key_inverse_transpose_map_bounding": key_bound,
        "reconciled_gauge_bounding": gauge_bound,
        "rmsnorm_image_balance_frontier": image_balance_frontier,
        "rmsnorm_image_balance_selected_maximum_gain": next(
            float(value["maximum_joint_divided_rms"])
            for value in image_balance_frontier
            if bool(value["selected"])
        ),
        "weighted_alignment_relative_rmse": float(
            torch.sqrt(residual_num / residual_den).item()
        ),
        "score_invariance_relative_rmse": score_invariance,
        "gauge_sha256": _tensor_sha256(gauge),
        "inverse_transpose_sha256": _tensor_sha256(inverse_transpose),
    }
    return gauge, inverse_transpose, record


def _align_case(
    state: Lfm2CpcfAttentionFactorSufficientState,
    replay: _Replay,
    query_norm: Any,
    key_norm: Any,
    geometry: _Geometry,
    policy: Lfm2QkGlobalGaugeSyncPolicy,
) -> tuple[Any, Any, tuple[Mapping[str, Any], ...]]:
    torch = _torch()
    local_query = state.local_query_post_rope_factor.to(torch.float64)
    local_key = state.local_key_post_rope_factor.to(torch.float64)
    aligned_query = torch.empty_like(local_query)
    aligned_key = torch.empty_like(local_key)
    batch = int(local_query.shape[0])
    query_length = int(local_query.shape[2])
    records: list[Mapping[str, Any]] = []
    offsets = geometry.queries_per_key
    for batch_index in range(batch):
        for group in range(geometry.key_heads):
            start = group * offsets
            stop = start + offsets
            query_rows = local_query[batch_index, start:stop].reshape(
                offsets * query_length,
                geometry.head_dim,
            )
            replay_query_rows = replay.query_post[
                batch_index,
                start:stop,
            ].reshape(offsets * query_length, geometry.head_dim)
            key_rows = local_key[batch_index, group]
            replay_key_rows = replay.key_post[batch_index, group]
            query_weights = state.query_row_sqrt_weights[batch_index].repeat(offsets)
            key_weights = state.key_row_sqrt_weights[batch_index]
            query_cosine_rows = state.target_query_rope_cos[batch_index].repeat(
                offsets, 1
            )
            query_sine_rows = state.target_query_rope_sin[batch_index].repeat(
                offsets, 1
            )
            key_cosine_rows = state.target_key_rope_cos[batch_index]
            key_sine_rows = state.target_key_rope_sin[batch_index]
            gauge, inverse_transpose, diagnostic = _bilinear_gauge_to_shared_replay(
                query_rows,
                key_rows,
                replay_query_rows,
                replay_key_rows,
                query_weights,
                key_weights,
                query_norm,
                key_norm,
                query_cosine_rows,
                query_sine_rows,
                key_cosine_rows,
                key_sine_rows,
                policy=policy,
            )
            aligned_query[batch_index, start:stop] = (query_rows @ gauge).reshape(
                offsets, query_length, geometry.head_dim
            )
            aligned_key[batch_index, group] = key_rows @ inverse_transpose
            records.append(
                MappingProxyType(
                    {
                        "batch_index": batch_index,
                        "target_key_value_group": group,
                        **diagnostic,
                    }
                )
            )
    return aligned_query, aligned_key, tuple(records)


def _positive_radial_preimage(
    desired_post: Any,
    current_raw: Any,
    norm_weight: Any,
    cosine: Any,
    sine: Any,
    *,
    score_informative_rows: Any | None = None,
    epsilon: float,
    policy: Lfm2QkGlobalGaugeSyncPolicy,
) -> tuple[Any, Any, dict[str, Any]]:
    """Invert reachable RMSNorm gain and evaluate an exact BF16 frontier.

    For ``x = rho*u`` with ``RMS(u)=1``, RMSNorm before its learned weight has
    gain ``a = rho / sqrt(rho**2 + epsilon)``.  Therefore every finite
    ``0 < a < 1`` has the unique positive radius

    ``rho = sqrt(epsilon) * a / sqrt(1 - a**2)``.

    An exactly zero synchronized factor row is different: it is the exact
    image of the zero raw row, and often represents a causal-centered score
    row with no identifying information.  Such rows remain exactly zero and
    are excluded from positive-radius statistics.

    The previous implementation computed ``a`` and then discarded it, which
    silently normalized every desired row toward gain one.  Here the exact
    continuous inverse is retained, bounded only by the explicit policy, and
    neighboring BF16 radii plus the current shared preimage are replayed
    through the literal target-dtype RMSNorm/RoPE chain.  The lowest exact
    post-RoPE row error is published deterministically.
    """

    torch = _torch()
    target_dtype = current_raw.dtype
    desired_input = desired_post.to(torch.float64).contiguous()
    if score_informative_rows is None:
        informative_rows = torch.ones(
            desired_input.shape[:-1],
            dtype=torch.bool,
        )
    elif (
        not isinstance(score_informative_rows, torch.Tensor)
        or score_informative_rows.dtype != torch.bool
        or tuple(score_informative_rows.shape) != tuple(desired_input.shape[:-1])
    ):
        raise Lfm2QkGlobalGaugeSyncError(
            "score_informative_rows must be a bool mask over desired factor rows."
        )
    else:
        informative_rows = score_informative_rows.detach().to(device="cpu").contiguous()
    input_exact_zero = desired_input.eq(0.0).all(dim=-1, keepdim=True)
    score_uninformative = (~informative_rows).unsqueeze(-1)
    desired_exact = torch.where(
        score_uninformative,
        torch.zeros_like(desired_input),
        desired_input,
    )
    unrotated = invert_lfm2_rope(
        desired_exact,
        cosine.to(torch.float64),
        sine.to(torch.float64),
    )
    divided = unrotated / norm_weight.to(torch.float64).reshape(1, 1, 1, -1)
    desired_gain = torch.sqrt(divided.square().mean(dim=-1, keepdim=True))
    zero_target = desired_exact.eq(0.0).all(dim=-1, keepdim=True)
    positive_target = ~zero_target
    safe_gain = torch.where(
        positive_target,
        desired_gain,
        torch.ones_like(desired_gain),
    )
    unit_direction = torch.where(
        positive_target,
        divided / safe_gain,
        torch.zeros_like(divided),
    )
    current_exact_raw = current_raw.to(dtype=target_dtype)
    current_radius_unclipped = torch.sqrt(
        current_exact_raw.to(torch.float64).square().mean(dim=-1, keepdim=True)
    )
    prior_radius = torch.where(
        positive_target,
        current_radius_unclipped.clamp(
            min=policy.minimum_radius,
            max=policy.maximum_radius,
        ),
        torch.zeros_like(current_radius_unclipped),
    )
    reachable = positive_target & (desired_gain < 1.0)
    unreachable = positive_target & ~reachable
    safe_denominator = (1.0 - desired_gain.square()).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    unbounded_radius = math.sqrt(epsilon) * desired_gain / torch.sqrt(safe_denominator)
    unbounded_radius = torch.where(
        reachable,
        unbounded_radius,
        torch.where(
            unreachable,
            torch.full_like(unbounded_radius, policy.maximum_radius),
            torch.zeros_like(unbounded_radius),
        ),
    )
    analytical_radius = torch.where(
        positive_target,
        unbounded_radius.clamp(
            min=policy.minimum_radius,
            max=policy.maximum_radius,
        ),
        torch.zeros_like(unbounded_radius),
    )
    strength = policy.radius_prior_strength
    regularized_radius = torch.where(
        positive_target,
        ((analytical_radius + strength * prior_radius) / (1.0 + strength)).clamp(
            min=policy.minimum_radius,
            max=policy.maximum_radius,
        ),
        torch.zeros_like(analytical_radius),
    )
    analytical_gain = torch.where(
        positive_target,
        unbounded_radius / torch.sqrt(unbounded_radius.square() + epsilon),
        torch.zeros_like(unbounded_radius),
    )
    reachable_gain_error = torch.where(
        reachable,
        (analytical_gain - desired_gain).abs(),
        torch.zeros_like(desired_gain),
    )

    central = analytical_radius.to(target_dtype)
    lower = torch.where(
        positive_target,
        torch.nextafter(
            central,
            torch.zeros_like(central),
        )
        .to(torch.float64)
        .clamp(
            min=policy.minimum_radius,
            max=policy.maximum_radius,
        ),
        torch.zeros_like(analytical_radius),
    )
    upper = torch.where(
        positive_target,
        torch.nextafter(
            central,
            torch.full_like(central, float("inf")),
        )
        .to(torch.float64)
        .clamp(
            min=policy.minimum_radius,
            max=policy.maximum_radius,
        ),
        torch.zeros_like(analytical_radius),
    )
    scalar_candidates = (
        # Try the coherent shared-replay radius first.  For unreachable gains
        # all sufficiently nonzero radii approach the same RMSNorm boundary;
        # this ordering avoids choosing a projection-destroying maximum radius
        # for a sub-BF16 post-RoPE improvement.
        prior_radius,
        analytical_radius,
        lower,
        upper,
        regularized_radius,
    )

    def exact_candidate(radius: Any) -> tuple[Any, Any, Any]:
        raw = (unit_direction * radius).to(dtype=target_dtype).contiguous()
        pre = lfm2_runtime_rms_norm(raw, norm_weight, epsilon)
        post = _apply_rope(
            pre,
            cosine.to(dtype=target_dtype),
            sine.to(dtype=target_dtype),
        )
        error = (
            (post.to(torch.float64) - desired_exact).square().mean(dim=-1, keepdim=True)
        )
        return raw, post, error

    current_exact_pre = lfm2_runtime_rms_norm(
        current_exact_raw,
        norm_weight,
        epsilon,
    )
    current_exact_post = _apply_rope(
        current_exact_pre,
        cosine.to(dtype=target_dtype),
        sine.to(dtype=target_dtype),
    )
    current_error = (
        (current_exact_post.to(torch.float64) - desired_exact)
        .square()
        .mean(dim=-1, keepdim=True)
    )
    current_is_bounded = (
        positive_target
        & (current_radius_unclipped >= policy.minimum_radius)
        & (current_radius_unclipped <= policy.maximum_radius)
    )
    infinity = torch.full_like(current_error, float("inf"))
    best_raw = current_exact_raw
    best_post = current_exact_post
    best_error = torch.where(current_is_bounded, current_error, infinity)
    best_index = torch.zeros_like(best_error, dtype=torch.int64)
    # Exact BF16 replay cannot resolve improvements below its own local
    # quantization scale.  Within that tie band retain the current coherent
    # shared raw preimage instead of replacing it with a numerically closer
    # but projection-incoherent saturation radius.
    bf16_tie_error = (
        4.0 * float(torch.finfo(target_dtype).eps)
    ) ** 2 * desired_exact.square().mean(dim=-1, keepdim=True).clamp_min(1.0)
    for candidate_index, candidate_radius in enumerate(scalar_candidates, start=1):
        candidate_raw, candidate_post, candidate_error = exact_candidate(
            candidate_radius
        )
        improve = candidate_error < (best_error - bf16_tie_error)
        best_raw = torch.where(improve, candidate_raw, best_raw)
        best_post = torch.where(improve, candidate_post, best_post)
        best_error = torch.where(improve, candidate_error, best_error)
        best_index = torch.where(
            improve,
            torch.full_like(best_index, candidate_index),
            best_index,
        )

    best_raw = torch.where(zero_target, torch.zeros_like(best_raw), best_raw)
    best_post = torch.where(zero_target, torch.zeros_like(best_post), best_post)
    best_error = torch.where(zero_target, torch.zeros_like(best_error), best_error)
    best_index = torch.where(
        zero_target,
        torch.full_like(best_index, -1),
        best_index,
    )
    raw_target = best_raw.to(torch.float64)
    selected_radius = torch.sqrt(raw_target.square().mean(dim=-1, keepdim=True))
    lower_hits = int(
        (positive_target & (selected_radius <= policy.minimum_radius)).sum().item()
    )
    upper_hits = int(
        (positive_target & (selected_radius >= policy.maximum_radius)).sum().item()
    )
    continuous_bound_clamps = int(
        (
            reachable
            & (
                (unbounded_radius < policy.minimum_radius)
                | (unbounded_radius > policy.maximum_radius)
            )
        )
        .sum()
        .item()
    )

    def extrema(value: Any, mask: Any) -> tuple[float, float]:
        selected = value[mask]
        if not selected.numel():
            return 0.0, 0.0
        return float(selected.min().item()), float(selected.max().item())

    minimum_positive_gain, maximum_positive_gain = extrema(
        desired_gain,
        positive_target,
    )
    minimum_positive_analytical, maximum_positive_analytical = extrema(
        analytical_radius,
        positive_target,
    )
    minimum_positive_prior, maximum_positive_prior = extrema(
        prior_radius,
        positive_target,
    )
    minimum_positive_solved, maximum_positive_solved = extrema(
        selected_radius,
        positive_target,
    )
    positive_solved = selected_radius[positive_target]
    record = {
        "row_count": int(desired_gain.numel()),
        "minimum_desired_gain": float(desired_gain.min().item()),
        "maximum_desired_gain": float(desired_gain.max().item()),
        "minimum_positive_desired_gain": minimum_positive_gain,
        "maximum_positive_desired_gain": maximum_positive_gain,
        "input_exact_zero_target_row_count": int(input_exact_zero.sum().item()),
        "score_uninformative_target_row_count": int(score_uninformative.sum().item()),
        "nonzero_svd_dust_rows_forced_to_zero": int(
            (score_uninformative & ~input_exact_zero).sum().item()
        ),
        "zero_target_row_count": int(zero_target.sum().item()),
        "positive_target_row_count": int(positive_target.sum().item()),
        "continuous_reachable_row_count": int(reachable.sum().item()),
        "continuous_unreachable_row_count": int(unreachable.sum().item()),
        "minimum_analytical_radius": float(analytical_radius.min().item()),
        "maximum_analytical_radius": float(analytical_radius.max().item()),
        "minimum_positive_analytical_radius": minimum_positive_analytical,
        "maximum_positive_analytical_radius": maximum_positive_analytical,
        "continuous_radius_bound_clamp_count": continuous_bound_clamps,
        "maximum_reachable_continuous_gain_absolute_error": float(
            reachable_gain_error.max().item()
        ),
        "minimum_prior_radius": float(prior_radius.min().item()),
        "maximum_prior_radius": float(prior_radius.max().item()),
        "minimum_positive_prior_radius": minimum_positive_prior,
        "maximum_positive_prior_radius": maximum_positive_prior,
        "minimum_solved_radius": float(selected_radius.min().item()),
        "maximum_solved_radius": float(selected_radius.max().item()),
        "minimum_positive_solved_radius": minimum_positive_solved,
        "maximum_positive_solved_radius": maximum_positive_solved,
        "mean_solved_radius": (
            float(positive_solved.mean().item()) if positive_solved.numel() else 0.0
        ),
        "lower_bound_hits": lower_hits,
        "upper_bound_hits": upper_hits,
        "closed_form_reachable_rmsnorm_gain_inverse": True,
        "exact_bf16_radial_candidate_count": 1 + len(scalar_candidates),
        "exact_zero_preimage_rows": int((best_index == -1).sum().item()),
        "exact_shared_preimage_rows_retained": int(
            (positive_target & (best_index == 0)).sum().item()
        ),
        "prior_radius_rows_retained": int(
            (positive_target & (best_index == 1)).sum().item()
        ),
        "analytical_inverse_rows_retained": int(
            (positive_target & (best_index == 2)).sum().item()
        ),
        "lower_bf16_neighbor_rows_retained": int(
            (positive_target & (best_index == 3)).sum().item()
        ),
        "upper_bf16_neighbor_rows_retained": int(
            (positive_target & (best_index == 4)).sum().item()
        ),
        "regularized_radius_rows_retained": int(
            (positive_target & (best_index == 5)).sum().item()
        ),
        "finite_preimage_frontier_uses_fit_scores": False,
        "shared_preimage_bound_rejections": int(
            (positive_target & ~current_is_bounded).sum().item()
        ),
        "positive_radius_enforced_for_nonzero_rows": True,
        "exact_zero_radius_permitted": True,
    }
    return raw_target, best_post, record


def _synchronize_case(
    state: Lfm2CpcfAttentionFactorSufficientState,
    replay: _Replay,
    query_norm: Any,
    key_norm: Any,
    epsilon: float,
    geometry: _Geometry,
    policy: Lfm2QkGlobalGaugeSyncPolicy,
) -> _SynchronizedCase:
    query_activity, key_activity = _factor_score_activity(state, geometry)
    aligned_q, aligned_k, gauge_records = _align_case(
        state,
        replay,
        query_norm,
        key_norm,
        geometry,
        policy,
    )
    q_raw, q_post, q_radius = _positive_radial_preimage(
        aligned_q,
        replay.query_raw,
        query_norm,
        state.target_query_rope_cos,
        state.target_query_rope_sin,
        score_informative_rows=query_activity,
        epsilon=epsilon,
        policy=policy,
    )
    k_raw, k_post, k_radius = _positive_radial_preimage(
        aligned_k,
        replay.key_raw,
        key_norm,
        state.target_key_rope_cos,
        state.target_key_rope_sin,
        score_informative_rows=key_activity,
        epsilon=epsilon,
        policy=policy,
    )
    return _SynchronizedCase(
        aligned_query_post=aligned_q,
        aligned_key_post=aligned_k,
        query_raw_target=q_raw,
        key_raw_target=k_raw,
        norm_image_query_post=q_post,
        norm_image_key_post=k_post,
        gauge_records=gauge_records,
        radius_record=MappingProxyType({"query": q_radius, "key": k_radius}),
    )


def _weighted_prior_ridge(
    inputs: Any,
    targets: Any,
    sqrt_weights: Any,
    prior: Any,
    ridge: float,
    *,
    role: str,
) -> Any:
    torch = _torch()
    x = inputs.to(torch.float64)
    y = targets.to(torch.float64)
    weights = sqrt_weights.to(torch.float64).reshape(-1, 1)
    if (
        x.ndim != 2
        or y.ndim != 2
        or int(x.shape[0]) != int(y.shape[0])
        or tuple(prior.shape) != (int(y.shape[1]), int(x.shape[1]))
    ):
        raise Lfm2QkGlobalGaugeSyncError(
            f"{role} global ridge rows or prior are incompatible."
        )
    weighted_x = x * weights
    weighted_y = y * weights
    gram = weighted_x.T @ weighted_x
    identity = torch.eye(int(gram.shape[0]), dtype=torch.float64)
    right = weighted_x.T @ weighted_y + ridge * prior.to(torch.float64).T
    try:
        solution = torch.linalg.solve(gram + ridge * identity, right)
    except RuntimeError as exc:
        raise Lfm2QkGlobalGaugeSyncError(
            f"{role} global prior-ridge solve failed."
        ) from exc
    result = solution.T.contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2QkGlobalGaugeSyncError(
            f"{role} global prior-ridge solve produced non-finite state."
        )
    return result


def _factor_score_activity(
    state: Lfm2CpcfAttentionFactorSufficientState,
    geometry: _Geometry,
) -> tuple[Any, Any]:
    """Return numerically resolvable rows in each score-factor GQA group."""

    torch = _torch()
    query_factor = state.local_query_post_rope_factor
    key_factor = state.local_key_post_rope_factor
    query_activity = torch.zeros(
        query_factor.shape[:-1],
        dtype=torch.bool,
    )
    key_activity = torch.zeros(
        key_factor.shape[:-1],
        dtype=torch.bool,
    )
    group_width = geometry.queries_per_key
    query_length = int(query_factor.shape[2])
    for batch_index in range(int(query_factor.shape[0])):
        for group in range(geometry.key_heads):
            start = group * group_width
            stop = start + group_width
            query_rows = query_factor[batch_index, start:stop].reshape(
                group_width * query_length,
                geometry.head_dim,
            )
            key_rows = key_factor[batch_index, group]
            query_rows_active, key_rows_active, _diagnostic = (
                _gauge_invariant_score_activity(query_rows, key_rows)
            )
            query_activity[batch_index, start:stop] = query_rows_active.reshape(
                group_width,
                query_length,
            )
            key_activity[batch_index, group] = key_rows_active
    return query_activity, key_activity


def _factor_activity_counts(
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    geometry: _Geometry,
) -> dict[str, Any]:
    query_rows = 0
    key_rows = 0
    zero_query_rows = 0
    zero_key_rows = 0
    informative_query_rows = 0
    informative_key_rows = 0
    informative_score_cells = 0
    score_activity_thresholds: list[float] = []
    score_activity_scales: list[float] = []
    one_key_centered_groups = 0
    for state in states:
        query_nonzero = state.local_query_post_rope_factor.ne(0.0).any(dim=-1)
        key_nonzero = state.local_key_post_rope_factor.ne(0.0).any(dim=-1)
        query_activity, key_activity = _factor_score_activity(state, geometry)
        query_length = int(state.local_query_post_rope_factor.shape[2])
        group_width = geometry.queries_per_key
        for batch_index in range(int(query_nonzero.shape[0])):
            for group in range(geometry.key_heads):
                start = group * group_width
                stop = start + group_width
                _query_active, _key_active, diagnostic = (
                    _gauge_invariant_score_activity(
                        state.local_query_post_rope_factor[
                            batch_index, start:stop
                        ].reshape(
                            group_width * query_length,
                            geometry.head_dim,
                        ),
                        state.local_key_post_rope_factor[batch_index, group],
                    )
                )
                score_activity_thresholds.append(
                    float(diagnostic["score_activity_absolute_tolerance"])
                )
                score_activity_scales.append(
                    float(diagnostic["score_maximum_absolute_value"])
                )
                one_key_centered_groups += int(
                    diagnostic["score_activity_rule"]
                    == "one_key_centered_score_has_no_identifiable_contrast"
                )
        repeated_key_activity = key_activity.repeat_interleave(
            geometry.queries_per_key,
            dim=1,
        )
        informative_cells = query_activity.unsqueeze(-1) & (
            repeated_key_activity.unsqueeze(-2)
        )
        query_rows += int(query_nonzero.numel())
        key_rows += int(key_nonzero.numel())
        zero_query_rows += int((~query_nonzero).sum().item())
        zero_key_rows += int((~key_nonzero).sum().item())
        informative_query_rows += int(query_activity.sum().item())
        informative_key_rows += int(key_activity.sum().item())
        informative_score_cells += int(informative_cells.sum().item())
    return {
        "query_factor_row_count": query_rows,
        "key_factor_row_count": key_rows,
        "exact_zero_query_factor_row_count": zero_query_rows,
        "exact_zero_key_factor_row_count": zero_key_rows,
        "informative_query_factor_row_count": informative_query_rows,
        "informative_key_factor_row_count": informative_key_rows,
        "score_uninformative_query_factor_row_count": (
            query_rows - informative_query_rows
        ),
        "score_uninformative_key_factor_row_count": key_rows - informative_key_rows,
        "total_score_cell_count": geometry.total_score_elements,
        "informative_score_cell_count": informative_score_cells,
        "score_uninformative_cell_count": (
            geometry.total_score_elements - informative_score_cells
        ),
        "activity_rule": (
            "gauge_invariant_score_row_or_column_above_a_matrix_scale_aware_"
            "fp64_roundoff_floor; one_key_centered_groups_are_unidentifiable"
        ),
        "fixed_absolute_score_floor_used": False,
        "score_activity_relative_roundoff_multiplier": (
            _SCORE_ACTIVITY_ROUNDOFF_MULTIPLIER
        ),
        "minimum_score_activity_absolute_tolerance": min(score_activity_thresholds),
        "maximum_score_activity_absolute_tolerance": max(score_activity_thresholds),
        "minimum_group_score_scale": min(score_activity_scales),
        "maximum_group_score_scale": max(score_activity_scales),
        "one_key_centered_group_count": one_key_centered_groups,
        "counts_authenticated_by_factor_state_and_report_sha256": True,
    }


def _fit_source_anchor_initialization(
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    query_prior: Any,
    key_prior: Any,
    geometry: _Geometry,
    ridge: float,
) -> tuple[Any, Any, dict[str, float]]:
    """Fit one globally coherent shared replay from raw-linear source anchors."""

    torch = _torch()
    query_inputs = torch.cat(
        [
            state.target_query_inputs.to(torch.float64).reshape(
                -1,
                geometry.hidden_width,
            )
            for state in states
        ],
        dim=0,
    )
    query_targets = torch.cat(
        [
            state.query_raw_linear_anchor.to(torch.float64)
            .permute(0, 2, 1, 3)
            .reshape(-1, geometry.query_heads * geometry.head_dim)
            for state in states
        ],
        dim=0,
    )
    query_sqrt = torch.cat(
        [state.query_row_sqrt_weights.reshape(-1, 1) for state in states],
        dim=0,
    )
    key_inputs = torch.cat(
        [
            state.target_key_inputs.to(torch.float64).reshape(
                -1,
                geometry.hidden_width,
            )
            for state in states
        ],
        dim=0,
    )
    key_targets = torch.cat(
        [
            state.key_raw_linear_anchor.to(torch.float64)
            .permute(0, 2, 1, 3)
            .reshape(-1, geometry.key_heads * geometry.head_dim)
            for state in states
        ],
        dim=0,
    )
    key_sqrt = torch.cat(
        [state.key_row_sqrt_weights.reshape(-1, 1) for state in states],
        dim=0,
    )
    query_weight = _weighted_prior_ridge(
        query_inputs,
        query_targets,
        query_sqrt,
        query_prior,
        ridge,
        role="query source-anchor initialization",
    )
    key_weight = _weighted_prior_ridge(
        key_inputs,
        key_targets,
        key_sqrt,
        key_prior,
        ridge,
        role="key source-anchor initialization",
    )

    def target_error(
        inputs: Any,
        targets: Any,
        sqrt_weights: Any,
        weight: Any,
    ) -> float:
        prediction = inputs @ weight.T
        scale = sqrt_weights.square()
        numerator = (scale * (prediction - targets).square()).sum()
        denominator = (
            (scale * targets.square()).sum().clamp_min(torch.finfo(torch.float64).tiny)
        )
        return float(torch.sqrt(numerator / denominator).item())

    return (
        query_weight,
        key_weight,
        {
            "query_raw_anchor_relative_rmse": target_error(
                query_inputs,
                query_targets,
                query_sqrt,
                query_weight,
            ),
            "key_raw_anchor_relative_rmse": target_error(
                key_inputs,
                key_targets,
                key_sqrt,
                key_weight,
            ),
        },
    )


def _fit_global_weights(
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    synchronized: tuple[_SynchronizedCase, ...],
    query_prior: Any,
    key_prior: Any,
    geometry: _Geometry,
    ridge: float,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = _torch()
    query_inputs = torch.cat(
        [
            state.target_query_inputs.to(torch.float64).reshape(
                -1,
                geometry.hidden_width,
            )
            for state in states
        ],
        dim=0,
    )
    query_targets_by_head = torch.cat(
        [
            value.query_raw_target.permute(0, 2, 1, 3).reshape(
                -1, geometry.query_heads, geometry.head_dim
            )
            for value in synchronized
        ],
        dim=0,
    )
    query_sqrt = torch.cat(
        [state.query_row_sqrt_weights.reshape(-1, 1) for state in states],
        dim=0,
    )
    key_inputs = torch.cat(
        [
            state.target_key_inputs.to(torch.float64).reshape(
                -1,
                geometry.hidden_width,
            )
            for state in states
        ],
        dim=0,
    )
    key_targets_by_head = torch.cat(
        [
            value.key_raw_target.permute(0, 2, 1, 3).reshape(
                -1, geometry.key_heads, geometry.head_dim
            )
            for value in synchronized
        ],
        dim=0,
    )
    key_sqrt = torch.cat(
        [state.key_row_sqrt_weights.reshape(-1, 1) for state in states],
        dim=0,
    )
    activities = tuple(_factor_score_activity(state, geometry) for state in states)
    query_activity = torch.cat(
        [
            query.permute(0, 2, 1).reshape(-1, geometry.query_heads)
            for query, _key in activities
        ],
        dim=0,
    )
    key_activity = torch.cat(
        [
            key.permute(0, 2, 1).reshape(-1, geometry.key_heads)
            for _query, key in activities
        ],
        dim=0,
    )
    query_blocks: list[Any] = []
    query_solved_heads = 0
    for head in range(geometry.query_heads):
        start = head * geometry.head_dim
        stop = start + geometry.head_dim
        head_activity = query_activity[:, head].reshape(-1, 1)
        head_prior = query_prior[start:stop]
        if bool(head_activity.any().item()):
            query_blocks.append(
                _weighted_prior_ridge(
                    query_inputs,
                    query_targets_by_head[:, head],
                    query_sqrt * head_activity,
                    head_prior,
                    ridge,
                    role=f"query head {head}",
                )
            )
            query_solved_heads += 1
        else:
            query_blocks.append(head_prior.to(torch.float64).clone())
    key_blocks: list[Any] = []
    key_solved_heads = 0
    for head in range(geometry.key_heads):
        start = head * geometry.head_dim
        stop = start + geometry.head_dim
        head_activity = key_activity[:, head].reshape(-1, 1)
        head_prior = key_prior[start:stop]
        if bool(head_activity.any().item()):
            key_blocks.append(
                _weighted_prior_ridge(
                    key_inputs,
                    key_targets_by_head[:, head],
                    key_sqrt * head_activity,
                    head_prior,
                    ridge,
                    role=f"key head {head}",
                )
            )
            key_solved_heads += 1
        else:
            key_blocks.append(head_prior.to(torch.float64).clone())
    query_weight = torch.cat(query_blocks, dim=0).contiguous()
    key_weight = torch.cat(key_blocks, dim=0).contiguous()

    def target_error(
        inputs: Any,
        targets: Any,
        sqrt_weights: Any,
        activity: Any,
        weight: Any,
        head_count: int,
    ) -> float:
        prediction = (inputs @ weight.T).reshape(
            -1,
            head_count,
            geometry.head_dim,
        )
        scale = sqrt_weights.square().reshape(-1, 1, 1) * activity.to(
            torch.float64
        ).unsqueeze(-1)
        numerator = (scale * (prediction - targets).square()).sum()
        denominator = (
            (scale * targets.square()).sum().clamp_min(torch.finfo(torch.float64).tiny)
        )
        return float(torch.sqrt(numerator / denominator).item())

    return (
        query_weight,
        key_weight,
        {
            "query_raw_target_relative_rmse": target_error(
                query_inputs,
                query_targets_by_head,
                query_sqrt,
                query_activity,
                query_weight,
                geometry.query_heads,
            ),
            "key_raw_target_relative_rmse": target_error(
                key_inputs,
                key_targets_by_head,
                key_sqrt,
                key_activity,
                key_weight,
                geometry.key_heads,
            ),
            "query_head_ridge_solve_count": query_solved_heads,
            "key_head_ridge_solve_count": key_solved_heads,
            "query_heads_retained_from_prior_without_factor_evidence": (
                geometry.query_heads - query_solved_heads
            ),
            "key_heads_retained_from_prior_without_factor_evidence": (
                geometry.key_heads - key_solved_heads
            ),
            "zero_or_score_uninformative_factor_rows_used_in_global_ridge": False,
        },
    )


def _score_rows(query: Any, key: Any, geometry: _Geometry) -> Any:
    repeated_key = key.repeat_interleave(geometry.queries_per_key, dim=1)
    return _torch().matmul(
        query.to(_torch().float64),
        repeated_key.to(_torch().float64).transpose(-2, -1),
    ) / math.sqrt(geometry.head_dim)


def _accumulate_score_error(
    prediction_query: Any,
    prediction_key: Any,
    target_query: Any,
    target_key: Any,
    state: Lfm2CpcfAttentionFactorSufficientState,
    geometry: _Geometry,
) -> tuple[Any, Any]:
    torch = _torch()
    prediction = _score_rows(prediction_query, prediction_key, geometry)
    target = _score_rows(target_query, target_key, geometry)
    weights = state.query_row_sqrt_weights.to(torch.float64).square().reshape(
        int(target.shape[0]), 1, int(target.shape[2]), 1
    ) * state.key_row_sqrt_weights.to(torch.float64).square().reshape(
        int(target.shape[0]), 1, 1, int(target.shape[3])
    )
    weights = weights.expand_as(target)
    query_activity, key_activity = _factor_score_activity(state, geometry)
    repeated_key_activity = key_activity.repeat_interleave(
        geometry.queries_per_key,
        dim=1,
    )
    informative_cells = query_activity.unsqueeze(-1) & (
        repeated_key_activity.unsqueeze(-2)
    )
    weights = weights * informative_cells.to(torch.float64)
    numerator = (weights * (prediction - target).square()).sum()
    denominator = (weights * target.square()).sum()
    return numerator, denominator


def _aggregate_score_rrmse(
    predictions: Sequence[tuple[Any, Any]],
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    geometry: _Geometry,
) -> float:
    torch = _torch()
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for (query, key), state in zip(predictions, states, strict=True):
        add_num, add_den = _accumulate_score_error(
            query,
            key,
            state.local_query_post_rope_factor,
            state.local_key_post_rope_factor,
            state,
            geometry,
        )
        numerator += add_num
        denominator += add_den
    return float(
        torch.sqrt(
            numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)
        ).item()
    )


def _error_ladder(
    *,
    states: tuple[Lfm2CpcfAttentionFactorSufficientState, ...],
    synchronized: tuple[_SynchronizedCase, ...],
    continuous_replays: tuple[_Replay, ...],
    exact_replays: tuple[_Replay, ...],
    geometry: _Geometry,
    rank_error: float | None,
) -> dict[str, float | None]:
    return {
        "E_rank": rank_error,
        "E_norm_image": _aggregate_score_rrmse(
            [
                (value.norm_image_query_post, value.norm_image_key_post)
                for value in synchronized
            ],
            states,
            geometry,
        ),
        "E_projection": _aggregate_score_rrmse(
            [(value.query_post, value.key_post) for value in continuous_replays],
            states,
            geometry,
        ),
        "E_runtime": _aggregate_score_rrmse(
            [(value.query_post, value.key_post) for value in exact_replays],
            states,
            geometry,
        ),
    }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledQkGlobalGaugeSync:
    """Dense BF16 Q/K result plus a sealed FIT-only analytical receipt."""

    query_weight: Any
    key_weight: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        query = _exact_float_tensor(self.query_weight, "query_weight", ndim=2)
        key = _exact_float_tensor(self.key_weight, "key_weight", ndim=2)
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
            raise Lfm2QkGlobalGaugeSyncError(
                "Published synchronized Q/K weights must be dense BF16 tensors."
            )
        object.__setattr__(self, "query_weight", query)
        object.__setattr__(self, "key_weight", key)
        report_copy = json.loads(
            json.dumps(
                _deep_thaw(self.report),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        object.__setattr__(self, "report", _deep_freeze(report_copy))
        self.verify_authentication()

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().clone(),
                "k_proj.weight": self.key_weight.detach().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)

    def verify_authentication(
        self,
        *,
        expected_fit_receipt_sha256: str | None = None,
        expected_source_runtime_receipt_sha256: str | None = None,
        expected_factor_state_bundle_sha256: str | None = None,
    ) -> str:
        body = self.as_dict()
        sealed = _required_sha256(body.pop("report_sha256", None), "report_sha256")
        if (
            _canonical_sha256(body) != sealed
            or body.get("query_weight_sha256") != _tensor_sha256(self.query_weight)
            or body.get("key_weight_sha256") != _tensor_sha256(self.key_weight)
        ):
            raise Lfm2QkGlobalGaugeSyncError(
                "Published global Q/K synchronization receipt is invalid."
            )
        provenance = body.get("provenance")
        if not isinstance(provenance, dict):
            raise Lfm2QkGlobalGaugeSyncError(
                "The receipt lacks typed global synchronization provenance."
            )
        provenance_body = dict(provenance)
        provenance_seal = _required_sha256(
            provenance_body.pop("provenance_sha256", None),
            "provenance_sha256",
        )
        for name in (
            "fit_receipt_sha256",
            "source_runtime_receipt_sha256",
            "factor_state_bundle_sha256",
            "source_prior_bundle_sha256",
            "target_runtime_bundle_sha256",
        ):
            _required_sha256(provenance_body.get(name), name)
        if _canonical_sha256(provenance_body) != provenance_seal:
            raise Lfm2QkGlobalGaugeSyncError(
                "The embedded global synchronization provenance is invalid."
            )
        expected_receipts = (
            (
                expected_fit_receipt_sha256,
                "fit_receipt_sha256",
            ),
            (
                expected_source_runtime_receipt_sha256,
                "source_runtime_receipt_sha256",
            ),
            (
                expected_factor_state_bundle_sha256,
                "factor_state_bundle_sha256",
            ),
        )
        for expected, name in expected_receipts:
            if expected is not None and provenance_body[name] != _required_sha256(
                expected,
                f"expected_{name}",
            ):
                raise Lfm2QkGlobalGaugeSyncError(
                    f"The embedded {name} differs from the authoritative receipt."
                )
        audit = body.get("operation_audit")
        if not isinstance(audit, dict) or any(
            audit.get(name) != 0
            for name in (
                "model_forward_calls",
                "backward_calls",
                "optimizer_steps",
                "gradient_updates",
                "training_examples",
                "selection_case_count",
                "validation_case_count",
                "fit_check_case_count",
                "trace_capture_calls",
            )
        ):
            raise Lfm2QkGlobalGaugeSyncError(
                "The receipt does not prove zero-training FIT-only execution."
            )
        return sealed


def compile_lfm2_qk_global_gauge_sync(
    *,
    factor_states: Sequence[Lfm2CpcfAttentionFactorSufficientState],
    target_query_norm_weight: Any,
    target_key_norm_weight: Any,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    source_query_weight_prior: Any,
    source_key_weight_prior: Any,
    norm_epsilon: float,
    provenance: Lfm2QkGlobalGaugeSyncProvenance,
    policy: Lfm2QkGlobalGaugeSyncPolicy | None = None,
) -> Lfm2CompiledQkGlobalGaugeSync:
    """Compile one shared dense BF16 Q/K pair from authenticated FIT factors."""

    torch = _torch()
    states = tuple(factor_states)
    query_norm = _exact_float_tensor(
        target_query_norm_weight,
        "target_query_norm_weight",
        ndim=1,
    )
    key_norm = _exact_float_tensor(
        target_key_norm_weight,
        "target_key_norm_weight",
        ndim=1,
    )
    query_template = _exact_float_tensor(
        target_query_weight_template,
        "target_query_weight_template",
        ndim=2,
    )
    key_template = _exact_float_tensor(
        target_key_weight_template,
        "target_key_weight_template",
        ndim=2,
    )
    query_prior = _exact_float_tensor(
        source_query_weight_prior,
        "source_query_weight_prior",
        ndim=2,
    )
    key_prior = _exact_float_tensor(
        source_key_weight_prior,
        "source_key_weight_prior",
        ndim=2,
    )
    epsilon = _finite_positive(norm_epsilon, "norm_epsilon")
    policy_value = policy or Lfm2QkGlobalGaugeSyncPolicy()
    if not isinstance(policy_value, Lfm2QkGlobalGaugeSyncPolicy):
        raise TypeError("policy must be Lfm2QkGlobalGaugeSyncPolicy.")
    if not isinstance(provenance, Lfm2QkGlobalGaugeSyncProvenance):
        raise TypeError("provenance must be Lfm2QkGlobalGaugeSyncProvenance.")
    geometry = _validate_inputs(
        states,
        query_norm,
        key_norm,
        query_template,
        key_template,
        query_prior,
        key_prior,
        policy_value,
    )
    factor_activity = _factor_activity_counts(states, geometry)
    provenance.verify(
        states=states,
        query_prior=query_prior,
        key_prior=key_prior,
        query_norm_weight=query_norm,
        key_norm_weight=key_norm,
        query_template=query_template,
        key_template=key_template,
        norm_epsilon=epsilon,
    )
    # The sufficient state contains only already-truncated d-dimensional
    # factors.  Without the discarded singular spectrum or parent score
    # matrix, parent-to-rank-d loss is unknowable rather than approximately
    # zero.
    rank_error: float | None = None
    source_prior_replays = tuple(
        _exact_replay(
            state,
            query_prior,
            key_prior,
            query_norm,
            key_norm,
            epsilon,
            geometry,
        )
        for state in states
    )
    source_prior_runtime_error = _aggregate_score_rrmse(
        [(value.query_post, value.key_post) for value in source_prior_replays],
        states,
        geometry,
    )
    (
        anchor_query_weight64,
        anchor_key_weight64,
        source_anchor_projection_fit,
    ) = _fit_source_anchor_initialization(
        states,
        query_prior,
        key_prior,
        geometry,
        policy_value.source_anchor_prior_ridge,
    )
    query_weight = anchor_query_weight64.to(torch.bfloat16).contiguous()
    key_weight = anchor_key_weight64.to(torch.bfloat16).contiguous()
    source_anchor_replays = tuple(
        _exact_replay(
            state,
            query_weight,
            key_weight,
            query_norm,
            key_norm,
            epsilon,
            geometry,
        )
        for state in states
    )
    source_anchor_runtime_error = _aggregate_score_rrmse(
        [(value.query_post, value.key_post) for value in source_anchor_replays],
        states,
        geometry,
    )
    initialization = {
        "method": "global_weighted_raw_linear_anchor_ridge_around_source_prior",
        "uses_balanced_score_factor_gauges": False,
        "fixed_no_candidate_selection": True,
        "source_prior_runtime_score_relative_rmse": source_prior_runtime_error,
        "source_anchor_runtime_score_relative_rmse": source_anchor_runtime_error,
        "source_anchor_projection_fit": source_anchor_projection_fit,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
    }
    frontier: list[dict[str, Any]] = []
    final_synchronized: tuple[_SynchronizedCase, ...] = ()
    final_ladder: dict[str, float | None] = {}
    best_query_weight: Any = query_weight
    best_key_weight: Any = key_weight
    best_runtime_error = source_anchor_runtime_error
    published_sweep = 0
    for sweep in range(1, policy_value.sweeps + 1):
        shared_replays = tuple(
            _exact_replay(
                state,
                query_weight,
                key_weight,
                query_norm,
                key_norm,
                epsilon,
                geometry,
            )
            for state in states
        )
        synchronized = tuple(
            _synchronize_case(
                state,
                replay,
                query_norm,
                key_norm,
                epsilon,
                geometry,
                policy_value,
            )
            for state, replay in zip(states, shared_replays, strict=True)
        )
        query_weight64, key_weight64, projection_fit = _fit_global_weights(
            states,
            synchronized,
            query_prior,
            key_prior,
            geometry,
            policy_value.global_prior_ridge,
        )
        candidate_query_weight = query_weight64.to(torch.bfloat16).contiguous()
        candidate_key_weight = key_weight64.to(torch.bfloat16).contiguous()
        damping_frontier: list[dict[str, Any]] = []
        selected_query_weight = query_weight
        selected_key_weight = key_weight
        selected_query_weight64 = query_weight.to(torch.float64)
        selected_key_weight64 = key_weight.to(torch.float64)
        selected_exact_replays = shared_replays
        selected_runtime_error = _aggregate_score_rrmse(
            [(value.query_post, value.key_post) for value in shared_replays],
            states,
            geometry,
        )
        selected_fraction = 0.0
        for fraction in (0.125, 0.25, 0.5, 1.0):
            blended_query64 = (1.0 - fraction) * query_weight.to(
                torch.float64
            ) + fraction * query_weight64
            blended_key64 = (1.0 - fraction) * key_weight.to(
                torch.float64
            ) + fraction * key_weight64
            blended_query = blended_query64.to(torch.bfloat16).contiguous()
            blended_key = blended_key64.to(torch.bfloat16).contiguous()
            blended_replays = tuple(
                _exact_replay(
                    state,
                    blended_query,
                    blended_key,
                    query_norm,
                    key_norm,
                    epsilon,
                    geometry,
                )
                for state in states
            )
            blended_error = _aggregate_score_rrmse(
                [(value.query_post, value.key_post) for value in blended_replays],
                states,
                geometry,
            )
            damping_frontier.append(
                {
                    "fraction": fraction,
                    "runtime_score_relative_rmse": blended_error,
                    "query_weight_sha256": _tensor_sha256(blended_query),
                    "key_weight_sha256": _tensor_sha256(blended_key),
                }
            )
            if blended_error < selected_runtime_error:
                selected_query_weight = blended_query
                selected_key_weight = blended_key
                selected_query_weight64 = blended_query64
                selected_key_weight64 = blended_key64
                selected_exact_replays = blended_replays
                selected_runtime_error = blended_error
                selected_fraction = fraction
        continuous_replays = tuple(
            _continuous_replay(
                state,
                selected_query_weight64,
                selected_key_weight64,
                query_norm,
                key_norm,
                epsilon,
                geometry,
            )
            for state in states
        )
        ladder = _error_ladder(
            states=states,
            synchronized=synchronized,
            continuous_replays=continuous_replays,
            exact_replays=selected_exact_replays,
            geometry=geometry,
            rank_error=rank_error,
        )
        gauge_records = [
            dict(record) for value in synchronized for record in value.gauge_records
        ]
        radius_records = [dict(value.radius_record) for value in synchronized]
        candidate_runtime_error = selected_runtime_error
        accepted_into_frontier = candidate_runtime_error < best_runtime_error
        if accepted_into_frontier:
            best_runtime_error = candidate_runtime_error
            best_query_weight = selected_query_weight
            best_key_weight = selected_key_weight
            final_synchronized = synchronized
            final_ladder = ladder
            published_sweep = sweep
        elif not final_synchronized:
            # Preserve a truthful ladder even when the source-anchor
            # initialization already dominates every analytical update.
            final_synchronized = synchronized
            final_ladder = {
                **ladder,
                "E_runtime": best_runtime_error,
            }
        frontier.append(
            {
                "sweep": sweep,
                "fixed_schedule": True,
                "candidate_error_ladder": ladder,
                "error_ladder": ladder,
                "accepted_into_monotone_frontier": accepted_into_frontier,
                "selected_damping_fraction": selected_fraction,
                "fixed_damping_frontier": damping_frontier,
                "published_sweep_after_step": published_sweep,
                "published_runtime_score_relative_rmse": best_runtime_error,
                "projection_fit": projection_fit,
                "maximum_gauge_alignment_relative_rmse": max(
                    float(record["weighted_alignment_relative_rmse"])
                    for record in gauge_records
                ),
                "maximum_gauge_score_invariance_relative_rmse": max(
                    float(record["score_invariance_relative_rmse"])
                    for record in gauge_records
                ),
                "maximum_gauge_condition_number": max(
                    float(record["reconciled_gauge_bounding"]["condition_number"])
                    for record in gauge_records
                ),
                "minimum_identifiable_gauge_rank": min(
                    int(record["identifiable_rank"]) for record in gauge_records
                ),
                "maximum_gauge_nullspace_dimension": max(
                    int(record["nullspace_dimension"]) for record in gauge_records
                ),
                "gauge_records": gauge_records,
                "radius_records": radius_records,
                "candidate_query_weight_sha256": _tensor_sha256(candidate_query_weight),
                "candidate_key_weight_sha256": _tensor_sha256(candidate_key_weight),
                "published_query_weight_sha256": _tensor_sha256(best_query_weight),
                "published_key_weight_sha256": _tensor_sha256(best_key_weight),
            }
        )
        # Rejected sweeps cannot move the next shared replay away from the
        # best FIT state already established by the fixed analytical frontier.
        query_weight = best_query_weight
        key_weight = best_key_weight
    if not final_synchronized:  # pragma: no cover - policy guarantees sweeps
        raise AssertionError("The fixed synchronization schedule did not execute.")
    if not all(
        value is None or math.isfinite(float(value)) for value in final_ladder.values()
    ):
        raise Lfm2QkGlobalGaugeSyncError(
            "The global Q/K synchronization error ladder is non-finite."
        )
    operation_audit = {
        "model_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "gradient_updates": 0,
        "training_examples": 0,
        "selection_case_count": 0,
        "validation_case_count": 0,
        "fit_check_case_count": 0,
        "trace_capture_calls": 0,
        "source_anchor_initialization_ridge_solves": 2,
        "analytical_fixed_sweeps": policy_value.sweeps,
        "bounded_full_bilinear_gauge_solves": sum(
            len(value["gauge_records"]) for value in frontier
        ),
        "symmetric_rmsnorm_gain_gauge_solves": sum(
            max(
                len(record["rmsnorm_image_balance_frontier"]) - 1,
                0,
            )
            for value in frontier
            for record in value["gauge_records"]
        ),
        "dense_rmsnorm_linear_maps_materialized": 0,
        "dense_rmsnorm_linear_map_elements_materialized": 0,
        "rmsnorm_rowwise_analytic_adjoint": True,
        "streamed_symmetric_gain_design_solves": sum(
            bool(step["gain_design_streamed"])
            for value in frontier
            for record in value["gauge_records"]
            for step in record["rmsnorm_image_balance_frontier"]
        ),
        "positive_radius_closed_form_solves": sum(
            int(record[role]["positive_target_row_count"])
            for value in frontier
            for record in value["radius_records"]
            for role in ("query", "key")
        ),
        "exact_zero_radial_preimages": sum(
            int(record[role]["input_exact_zero_target_row_count"])
            for value in frontier
            for record in value["radius_records"]
            for role in ("query", "key")
        ),
        "zero_or_score_uninformative_radial_preimages": sum(
            int(record[role]["zero_target_row_count"])
            for value in frontier
            for record in value["radius_records"]
            for role in ("query", "key")
        ),
        "nonzero_svd_dust_radial_preimages": sum(
            int(record[role]["nonzero_svd_dust_rows_forced_to_zero"])
            for value in frontier
            for record in value["radius_records"]
            for role in ("query", "key")
        ),
        "global_prior_ridge_solves": sum(
            int(value["projection_fit"]["query_head_ridge_solve_count"])
            + int(value["projection_fit"]["key_head_ridge_solve_count"])
            for value in frontier
        ),
        "analytical_frontier_state_comparisons": 5 * policy_value.sweeps,
        "zero_or_score_uninformative_factor_rows_used_in_global_ridge": False,
    }
    body = {
        "schema": _SCHEMA,
        "solver": "global_gauge_radius_synchronized",
        "scientific_scope": "fit_only_qk_analytical_compiler",
        "gauge_identifiability": (
            "bounded_full_gl_relative_to_source_prior_shared_replay_with_"
            "ridge_identity_nullspace_completion"
        ),
        "dense_parameter_state": True,
        "target_dtype": "torch.bfloat16",
        "query_heads": geometry.query_heads,
        "key_value_heads": geometry.key_heads,
        "queries_per_key_value": geometry.queries_per_key,
        "head_dim": geometry.head_dim,
        "hidden_width": geometry.hidden_width,
        "fit_case_count": len(states),
        "fit_case_identities": [
            {
                "case_ordinal": state.case_ordinal,
                "evidence_sha256": state.evidence_sha256,
            }
            for state in states
        ],
        "provenance": provenance.as_dict(),
        "provenance_scope": {
            "fit_receipt_authority": "caller_supplied_authenticated_fit_partition",
            "source_runtime_receipt_authority": (
                "caller_supplied_authenticated_source_runtime"
            ),
            "factor_state_tensors_bound_by_sha256": True,
            "factor_activity_counts_derived_from_authenticated_tensors": True,
            "source_prior_tensors_bound_by_sha256": True,
            "target_runtime_tensors_bound_by_sha256": True,
            "independent_upstream_receipt_discovery_performed": False,
        },
        "policy": policy_value.as_dict(),
        "resource_usage": {
            "total_rows": geometry.total_rows,
            "total_score_elements": geometry.total_score_elements,
            "total_factor_elements": geometry.total_factor_elements,
            "total_output_elements": geometry.total_output_elements,
            "total_svd_workspace_elements": (geometry.total_svd_workspace_elements),
            "dense_rmsnorm_linear_map_elements_materialized": 0,
            "rmsnorm_transform_storage_complexity": "O(total_factor_elements)",
            "primal_gain_design_chunk_element_ceiling": (_GAIN_DESIGN_CHUNK_ELEMENTS),
            "maximum_gain_design_elements_materialized": max(
                int(step["gain_design_peak_elements"])
                for value in frontier
                for record in value["gauge_records"]
                for step in record["rmsnorm_image_balance_frontier"]
            ),
        },
        "factor_activity": factor_activity,
        "fit_score_evidence": {
            "has_informative_factor_score_cells": bool(
                factor_activity["informative_score_cell_count"]
            ),
            "zero_information_error_ladder": not bool(
                factor_activity["informative_score_cell_count"]
            ),
            "zero_information_state_can_retain_source_initialization": True,
        },
        "target_runtime": {
            "query_norm_weight_sha256": _tensor_sha256(query_norm),
            "key_norm_weight_sha256": _tensor_sha256(key_norm),
            "query_weight_template_sha256": _tensor_sha256(query_template),
            "key_weight_template_sha256": _tensor_sha256(key_template),
            "norm_epsilon": epsilon,
        },
        "source_prior": {
            "query_weight_sha256": _tensor_sha256(query_prior),
            "key_weight_sha256": _tensor_sha256(key_prior),
        },
        "source_anchor_initialization": initialization,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
        "error_ladder": final_ladder,
        "error_ladder_meaning": {
            "E_rank": (
                "unavailable: sufficient states do not contain discarded "
                "singular energy or authoritative pre-truncation parent scores"
            ),
            "E_norm_image": (
                "weighted score RRMSE after exact positive radial preimages "
                "and target-dtype RMSNorm/RoPE, before a shared projection"
            ),
            "E_projection": (
                "weighted score RRMSE of the shared analytical projection "
                "under continuous FP64 RMSNorm/RoPE"
            ),
            "E_runtime": (
                "weighted score RRMSE after dense BF16 materialization and "
                "exact target-runtime projection/RMSNorm/RoPE replay"
            ),
            "factor_cell_scope": (
                "only cells joining score rows and columns above the "
                "gauge-invariant matrix-scale FP64 roundoff floor in the same "
                "GQA group; one-key centered groups and numerically "
                "unidentifiable SVD dust are excluded"
            ),
        },
        "deterministic_frontier": frontier,
        "frontier_selection": (
            "minimum_fit_runtime_error_on_fixed_analytical_sweep_and_damping_schedule"
        ),
        "published_sweep": published_sweep,
        "operation_audit": operation_audit,
        "not_claimed": [
            "heldout_fit_check_quality",
            "validation_quality",
            "whole_model_coherence",
            "parent_pretruncation_rank_retention",
            "absolute_gauge_identification_without_a_source_prior",
            "recovery_when_no_single_dense_qk_state_satisfies_the_fit_scores",
            "causal_masked_parent_score_error_from_factor_state_alone",
            "direct_score_residual_gauss_newton",
        ],
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    return Lfm2CompiledQkGlobalGaugeSync(
        query_weight=query_weight,
        key_weight=key_weight,
        report=report,
    )


__all__ = [
    "Lfm2CompiledQkGlobalGaugeSync",
    "Lfm2QkGlobalGaugeSyncError",
    "Lfm2QkGlobalGaugeSyncPolicy",
    "Lfm2QkGlobalGaugeSyncProvenance",
    "compile_lfm2_qk_global_gauge_sync",
    "seal_lfm2_qk_global_gauge_sync_provenance",
]
