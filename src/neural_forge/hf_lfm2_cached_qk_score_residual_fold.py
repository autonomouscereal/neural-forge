"""Bounded analytical Q/K score-residual folding for cached LFM2 evidence.

This module deliberately operates below a model runtime.  It consumes sealed
cached attention cases, an authenticated dense BF16 Q/K seed produced by an
earlier structural synchronization stage, and target-head centered scores.
It then performs at most two deterministic Gauss-Newton linearizations.  Each
linearized ridge system is solved by a hand-written matrix-free PCG operator.

There is no autograd graph, backward call, optimizer, model forward, teacher
rerun, pruning, quantization, or stochastic training loop.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Barrier
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (
    Lfm2CompiledCpcfAttentionDimensionFold,
    Lfm2CpcfAttentionFactorSufficientState,
    Lfm2CpcfAttentionDimensionProductionCase,
)
from neural_forge.hf_lfm2_qk_global_gauge_sync import (
    Lfm2CompiledQkGlobalGaugeSync,
)


LFM2_CACHED_QK_SCORE_RESIDUAL_SCHEMA = (
    "neural-forge-lfm2-cached-qk-score-residual-fold-v1"
)
LFM2_QK_TARGET_SCORE_TRANSFORM_SCHEMA = "neural-forge-lfm2-qk-target-score-transform-v1"
LFM2_QK_OBSERVED_EXECUTION_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-qk-score-observed-execution-v1"
)
LFM2_QK_LINEARIZATION_CACHE_RESOURCE_SCHEMA = (
    "neural-forge-lfm2-qk-fixed-linearization-cache-resource-v2"
)

_MAXIMUM_PCG_ITERATIONS = 4_096
_MAXIMUM_FRONTIER_POINTS = 9
_MAXIMUM_FIT_CASES = 4_096
_MAXIMUM_FIT_CHECK_CASES = 2_048
_MAXIMUM_SCORE_ELEMENTS_PER_CASE = 8_388_608
_MAXIMUM_AGGREGATE_SCORE_ELEMENTS = 268_435_456
_MAXIMUM_AGGREGATE_EVIDENCE_BYTES = 8 * 1024**3
_MAXIMUM_TRUST_RADIUS = 2.0
_HARD_MAXIMUM_SCORE_RRMSE = 0.35
_HARD_MAXIMUM_PROBABILITY_RRMSE = 0.20
_HARD_MAXIMUM_JS_MEAN = 0.05
_HARD_MAXIMUM_JS_WORST = 0.10
_HARD_MINIMUM_TOP1 = 0.90


class Lfm2CachedQkScoreResidualFoldError(ValueError):
    """Cached evidence, synchronized state, or solve policy is invalid."""


class Lfm2CachedQkScoreResidualAuthenticationError(Lfm2CachedQkScoreResidualFoldError):
    """A synchronized seed, evidence bank, or result failed authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for cached LFM2 Q/K score-residual folding."
        ) from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Tensor hashing requires a dense non-meta tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_nbytes(value: Any) -> int:
    return int(value.numel()) * int(value.element_size())


def _float_tensor(
    value: Any,
    name: str,
    *,
    ndim: int,
    exact_dtype: Any | None = None,
) -> Any:
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
        or (exact_dtype is not None and value.dtype != exact_dtype)
    ):
        dtype_clause = (
            "" if exact_dtype is None else f" with exact dtype {str(exact_dtype)}"
        )
        raise Lfm2CachedQkScoreResidualFoldError(
            f"{name} must be a finite dense real rank-{ndim} tensor{dtype_clause}."
        )
    result = value.detach().contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CachedQkScoreResidualFoldError(f"{name} contains NaN or infinity.")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedQkScoreResidualFoldError(f"{name} must be a positive integer.")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CachedQkScoreResidualFoldError(
            f"{name} must be a finite nonnegative number."
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CachedQkScoreResidualFoldError(
            f"{name} must be a finite nonnegative number."
        )
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise Lfm2CachedQkScoreResidualFoldError(f"{name} must be strictly positive.")
    return result


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def apply_lfm2_finite_radius_rope(
    value: Any,
    cosine: Any,
    sine: Any,
) -> Any:
    """Apply duplicated-half RoPE without assuming a unit finite radius."""

    return value * cosine.unsqueeze(1) + _rotate_half(value) * sine.unsqueeze(1)


def apply_lfm2_finite_radius_rope_transpose(
    cotangent: Any,
    cosine: Any,
    sine: Any,
) -> Any:
    """Apply the exact transpose of duplicated-half finite-radius RoPE."""

    return cotangent * cosine.unsqueeze(1) - _rotate_half(cotangent) * sine.unsqueeze(1)


def lfm2_continuous_rmsnorm_jvp(
    raw_rows: Any,
    tangent_rows: Any,
    norm_weight: Any,
    epsilon: float,
) -> Any:
    """Exact JVP of the continuous RMSNorm map around ``raw_rows``.

    The actual BF16 runtime contains rounding plateaus.  The compiler uses
    this continuous FP32 shadow only for the analytical direction and performs
    candidate selection through exact BF16 replay.
    """

    torch = _torch()
    raw = raw_rows.to(torch.float32)
    tangent = tangent_rows.to(torch.float32)
    gain = norm_weight.to(device=raw.device, dtype=torch.float32)
    if raw.shape != tangent.shape or tuple(gain.shape) != (int(raw.shape[-1]),):
        raise Lfm2CachedQkScoreResidualFoldError(
            "RMSNorm JVP rows, tangents, and gain have incompatible shapes."
        )
    epsilon_value = _finite_positive(epsilon, "epsilon")
    width = int(raw.shape[-1])
    radius_squared = raw.square().mean(dim=-1, keepdim=True) + epsilon_value
    radius = torch.sqrt(radius_squared)
    radial = (raw * tangent).sum(dim=-1, keepdim=True) / (width * radius_squared)
    return (gain / radius) * (tangent - raw * radial)


def lfm2_continuous_rmsnorm_vjp(
    raw_rows: Any,
    cotangent_rows: Any,
    norm_weight: Any,
    epsilon: float,
) -> Any:
    """Exact transpose-Jacobian product of the continuous RMSNorm map."""

    torch = _torch()
    raw = raw_rows.to(torch.float32)
    cotangent = cotangent_rows.to(torch.float32)
    gain = norm_weight.to(device=raw.device, dtype=torch.float32)
    if raw.shape != cotangent.shape or tuple(gain.shape) != (int(raw.shape[-1]),):
        raise Lfm2CachedQkScoreResidualFoldError(
            "RMSNorm VJP rows, cotangents, and gain have incompatible shapes."
        )
    epsilon_value = _finite_positive(epsilon, "epsilon")
    width = int(raw.shape[-1])
    radius_squared = raw.square().mean(dim=-1, keepdim=True) + epsilon_value
    radius = torch.sqrt(radius_squared)
    gained = cotangent * gain
    radial = (raw * gained).sum(dim=-1, keepdim=True) / (width * radius_squared)
    return (gained - raw * radial) / radius


def center_lfm2_causal_scores(value: Any, mask: Any) -> Any:
    """Project scores onto the zero-mean subspace of every causal row."""

    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (value * mask).sum(dim=-1, keepdim=True) / count
    return _torch().where(mask, value - mean, _torch().zeros_like(value))


def _masked_probabilities(scores: Any, mask: Any) -> Any:
    torch = _torch()
    masked = scores.to(torch.float64).masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked, dim=-1, dtype=torch.float64)
    return torch.where(mask, probabilities, torch.zeros_like(probabilities))


def _factor_state_bundle_sha256(
    states: Sequence[Lfm2CpcfAttentionFactorSufficientState],
) -> str:
    return _canonical_sha256([state.as_digest_dict() for state in states])


def _authenticate_dense_qk_seed_artifacts(
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
    global_sync: Lfm2CompiledQkGlobalGaugeSync,
) -> dict[str, str]:
    torch = _torch()
    if not isinstance(
        upstream_fold,
        Lfm2CompiledCpcfAttentionDimensionFold,
    ):
        raise TypeError("upstream_fold must be Lfm2CompiledCpcfAttentionDimensionFold.")
    if not isinstance(global_sync, Lfm2CompiledQkGlobalGaugeSync):
        raise TypeError("global_sync must be Lfm2CompiledQkGlobalGaugeSync.")
    try:
        upstream_sha256 = upstream_fold.verify_authentication()
    except Exception as exc:
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The seed's upstream CPCF fold failed authentication."
        ) from exc
    runtime = upstream_fold.runtime
    if runtime is None:
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The dense seed requires a production CPCF runtime."
        )
    fit_states = tuple(
        state
        for state in upstream_fold.factor_sufficient_states
        if state.partition == "fit"
    )
    if len(fit_states) < 2:
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The dense seed requires at least two authenticated upstream FIT "
            "factor states."
        )
    fit_bundle_sha256 = _factor_state_bundle_sha256(fit_states)
    all_bundle_sha256 = _factor_state_bundle_sha256(
        upstream_fold.factor_sufficient_states
    )
    if (
        upstream_fold.report.get("factor_sufficient_state_bundle_sha256")
        != all_bundle_sha256
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The upstream all-state factor bundle changed after authentication."
        )
    try:
        global_sha256 = global_sync.verify_authentication(
            expected_fit_receipt_sha256=upstream_sha256,
            expected_source_runtime_receipt_sha256=(runtime.source_runtime_sha256),
            expected_factor_state_bundle_sha256=fit_bundle_sha256,
        )
    except Exception as exc:
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The dense seed's global Q/K synchronization artifact is not "
            "cross-bound to the upstream CPCF FIT factors and runtime."
        ) from exc
    report = global_sync.as_dict()
    target_runtime = report.get("target_runtime")
    topology = upstream_fold.topology
    if (
        not isinstance(target_runtime, dict)
        or report.get("target_dtype") != "torch.bfloat16"
        or runtime.target_dtype != torch.bfloat16
        or report.get("query_heads") != topology.target_query_heads
        or report.get("key_value_heads") != topology.target_key_value_heads
        or report.get("queries_per_key_value") != topology.target_queries_per_key_value
        or report.get("head_dim") != topology.target_head_dim
        or report.get("hidden_width") != runtime.target_hidden_width
        or target_runtime.get("query_norm_weight_sha256")
        != _tensor_sha256(runtime.target_query_norm_weight)
        or target_runtime.get("key_norm_weight_sha256")
        != _tensor_sha256(runtime.target_key_norm_weight)
        or target_runtime.get("query_weight_template_sha256")
        != _tensor_sha256(runtime.target_query_weight_template)
        or target_runtime.get("key_weight_template_sha256")
        != _tensor_sha256(runtime.target_key_weight_template)
        or target_runtime.get("norm_epsilon") != runtime.norm_epsilon
        or tuple(global_sync.query_weight.shape)
        != (
            topology.target_query_heads * topology.target_head_dim,
            runtime.target_hidden_width,
        )
        or tuple(global_sync.key_weight.shape)
        != (
            topology.target_key_value_heads * topology.target_head_dim,
            runtime.target_hidden_width,
        )
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The global Q/K state does not match the authenticated CPCF target "
            "topology, templates, RMSNorm state, dtype, or epsilon."
        )
    return {
        "upstream_fold_report_sha256": upstream_sha256,
        "upstream_all_factor_state_bundle_sha256": all_bundle_sha256,
        "upstream_fit_factor_state_bundle_sha256": fit_bundle_sha256,
        "global_sync_report_sha256": global_sha256,
        "source_runtime_sha256": runtime.source_runtime_sha256,
    }


@dataclass(frozen=True, slots=True)
class Lfm2AuthenticatedDenseQkSeed:
    """Dense Q/K seed derived only from typed CPCF and global-sync artifacts."""

    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold
    global_sync: Lfm2CompiledQkGlobalGaugeSync
    seed_sha256: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        authenticated = _authenticate_dense_qk_seed_artifacts(
            self.upstream_fold,
            self.global_sync,
        )
        body = {
            **authenticated,
            "query_weight_sha256": _tensor_sha256(self.query_weight),
            "key_weight_sha256": _tensor_sha256(self.key_weight),
            "query_norm_weight_sha256": _tensor_sha256(self.query_norm_weight),
            "key_norm_weight_sha256": _tensor_sha256(self.key_norm_weight),
            "hidden_size": self.hidden_size,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "norm_epsilon": self.norm_epsilon,
        }
        computed = _canonical_sha256(body)
        object.__setattr__(self, "seed_sha256", computed)

    @property
    def query_weight(self) -> Any:
        return self.global_sync.query_weight

    @property
    def key_weight(self) -> Any:
        return self.global_sync.key_weight

    @property
    def query_norm_weight(self) -> Any:
        assert self.upstream_fold.runtime is not None
        return self.upstream_fold.runtime.target_query_norm_weight

    @property
    def key_norm_weight(self) -> Any:
        assert self.upstream_fold.runtime is not None
        return self.upstream_fold.runtime.target_key_norm_weight

    @property
    def head_dim(self) -> int:
        return self.upstream_fold.topology.target_head_dim

    @property
    def norm_epsilon(self) -> float:
        assert self.upstream_fold.runtime is not None
        return self.upstream_fold.runtime.norm_epsilon

    @property
    def source_artifact_sha256(self) -> str:
        return str(self.global_sync.report["report_sha256"])

    @property
    def source_runtime_sha256(self) -> str:
        assert self.upstream_fold.runtime is not None
        return self.upstream_fold.runtime.source_runtime_sha256

    @property
    def hidden_size(self) -> int:
        return int(self.query_weight.shape[1])

    @property
    def query_heads(self) -> int:
        return int(self.query_weight.shape[0]) // self.head_dim

    @property
    def key_value_heads(self) -> int:
        return int(self.key_weight.shape[0]) // self.head_dim

    @property
    def queries_per_key_value(self) -> int:
        return self.query_heads // self.key_value_heads

    def verify(self) -> str:
        rebuilt = build_lfm2_authenticated_dense_qk_seed(
            upstream_fold=self.upstream_fold,
            global_sync=self.global_sync,
        )
        if rebuilt.seed_sha256 != self.seed_sha256:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Dense Q/K synchronization seed changed after sealing."
            )
        return self.seed_sha256


def build_lfm2_authenticated_dense_qk_seed(
    *,
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
    global_sync: Lfm2CompiledQkGlobalGaugeSync,
) -> Lfm2AuthenticatedDenseQkSeed:
    """Build a dense seed only from cross-authenticated analytical artifacts."""

    return Lfm2AuthenticatedDenseQkSeed(
        upstream_fold=upstream_fold,
        global_sync=global_sync,
    )


@dataclass(frozen=True, slots=True)
class Lfm2QkTargetScoreTransformReceipt:
    """Typed receipt for one factor-state-to-target-score transformation."""

    source_split_sha256: str
    source_runtime_sha256: str
    trajectory_sha256: str
    partition: Literal["fit", "fit_check"]
    case_ordinal: int
    upstream_fold_report_sha256: str
    upstream_factor_state_bundle_sha256: str
    upstream_factor_state_index: int
    production_case_sha256: str
    factor_state_sha256: str
    query_heads: int
    key_value_heads: int
    head_dim: int
    transform_input_bundle_sha256: str
    transform_output_bundle_sha256: str
    solver_content_sha256: str
    solver_row_bundle_sha256: str
    trajectory_row_bundle_sha256: str
    transform_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "source_split_sha256",
            "source_runtime_sha256",
            "trajectory_sha256",
            "upstream_fold_report_sha256",
            "upstream_factor_state_bundle_sha256",
            "production_case_sha256",
            "factor_state_sha256",
            "transform_input_bundle_sha256",
            "transform_output_bundle_sha256",
            "solver_content_sha256",
            "solver_row_bundle_sha256",
            "trajectory_row_bundle_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _required_sha256(getattr(self, name), name),
            )
        if self.partition not in {"fit", "fit_check"}:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Target-score transform partition must be FIT or FIT-CHECK."
            )
        if (
            isinstance(self.case_ordinal, bool)
            or not isinstance(self.case_ordinal, int)
            or self.case_ordinal < 0
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Target-score transform case_ordinal must be nonnegative."
            )
        if (
            isinstance(self.upstream_factor_state_index, bool)
            or not isinstance(self.upstream_factor_state_index, int)
            or self.upstream_factor_state_index < 0
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Target-score upstream factor-state index must be nonnegative."
            )
        for name in ("query_heads", "key_value_heads", "head_dim"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.head_dim % 2 or self.query_heads % self.key_value_heads:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Target-score transform does not define even-head-dimension GQA."
            )
        computed = _canonical_sha256(self.as_dict(include_seal=False))
        if self.transform_receipt_sha256 and self.transform_receipt_sha256 != computed:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Target-score transform receipt does not match its seal."
            )
        object.__setattr__(self, "transform_receipt_sha256", computed)

    def as_dict(self, *, include_seal: bool = True) -> dict[str, Any]:
        result = {
            "schema": LFM2_QK_TARGET_SCORE_TRANSFORM_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "transform_receipt_sha256"
            },
        }
        if include_seal:
            result["transform_receipt_sha256"] = self.transform_receipt_sha256
        return result

    def verify(self) -> str:
        computed = _canonical_sha256(self.as_dict(include_seal=False))
        if computed != self.transform_receipt_sha256:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Target-score transform receipt changed after sealing."
            )
        return computed


def _same_numerical_tensor(left: Any, right: Any) -> bool:
    torch = _torch()
    return tuple(left.shape) == tuple(right.shape) and bool(
        torch.equal(left.to(torch.float64), right.to(torch.float64))
    )


def _authenticate_upstream_factor_member(
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
    factor_state: Lfm2CpcfAttentionFactorSufficientState,
    production_case: Lfm2CpcfAttentionDimensionProductionCase,
) -> dict[str, Any]:
    if not isinstance(
        upstream_fold,
        Lfm2CompiledCpcfAttentionDimensionFold,
    ):
        raise TypeError("upstream_fold must be Lfm2CompiledCpcfAttentionDimensionFold.")
    if not isinstance(
        factor_state,
        Lfm2CpcfAttentionFactorSufficientState,
    ):
        raise TypeError("factor_state must be Lfm2CpcfAttentionFactorSufficientState.")
    if not isinstance(
        production_case,
        Lfm2CpcfAttentionDimensionProductionCase,
    ):
        raise TypeError(
            "production_case must be Lfm2CpcfAttentionDimensionProductionCase."
        )
    try:
        report_sha256 = upstream_fold.verify_authentication()
    except Exception as exc:
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The upstream CPCF fold failed production authentication."
        ) from exc
    exact_member_indices = tuple(
        index
        for index, member in enumerate(upstream_fold.factor_sufficient_states)
        if member is factor_state
    )
    if len(exact_member_indices) != 1:
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The factor state must be the exact member selected from the "
            "authenticated upstream CPCF fold."
        )
    report = upstream_fold.report
    factor_bundle_sha256 = _canonical_sha256(
        [member.as_digest_dict() for member in upstream_fold.factor_sufficient_states]
    )
    provenance_report = report.get("provenance")
    provenance = production_case.provenance
    if (
        report.get("factor_sufficient_state_bundle_sha256") != factor_bundle_sha256
        or not isinstance(provenance_report, Mapping)
        or provenance_report.get("source_split_sha256")
        != provenance.source_split_sha256
        or provenance_report.get("source_runtime_sha256")
        != provenance.source_runtime_sha256
        or upstream_fold.runtime is None
        or upstream_fold.runtime.source_runtime_sha256
        != provenance.source_runtime_sha256
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "The production case provenance is not cross-bound to the "
            "authenticated upstream CPCF fold and factor bundle."
        )
    return {
        "report_sha256": report_sha256,
        "factor_state_bundle_sha256": factor_bundle_sha256,
        "factor_state_index": exact_member_indices[0],
    }


def _derive_target_score_payload(
    production_case: Lfm2CpcfAttentionDimensionProductionCase,
    factor_state: Lfm2CpcfAttentionFactorSufficientState,
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
) -> dict[str, Any]:
    torch = _torch()
    upstream = _authenticate_upstream_factor_member(
        upstream_fold,
        factor_state,
        production_case,
    )
    production_sha = production_case.verify()
    provenance = production_case.provenance
    if (
        factor_state.partition != provenance.partition
        or factor_state.case_ordinal != provenance.case_ordinal
        or factor_state.evidence_sha256 != production_sha
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Factor state partition/ordinal/evidence identity does not match "
            "the authenticated production case."
        )
    if not (
        _same_numerical_tensor(
            factor_state.target_query_inputs,
            production_case.target_query_residual_inputs,
        )
        and _same_numerical_tensor(
            factor_state.target_key_inputs,
            production_case.target_key_value_residual_inputs,
        )
        and _same_numerical_tensor(
            factor_state.query_row_sqrt_weights,
            production_case.query_row_sqrt_weights,
        )
        and _same_numerical_tensor(
            factor_state.key_row_sqrt_weights,
            production_case.key_value_row_sqrt_weights,
        )
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Factor-state target inputs or row weights do not match production "
            "evidence."
        )
    assert upstream_fold.runtime is not None
    target_dtype = upstream_fold.runtime.target_dtype
    query_factor = factor_state.local_query_post_rope_factor.to(
        dtype=target_dtype
    ).contiguous()
    key_factor = factor_state.local_key_post_rope_factor.to(
        dtype=target_dtype
    ).contiguous()
    batch, query_heads, query_length, head_dim = map(int, query_factor.shape)
    key_batch, key_value_heads, key_length, key_dim = map(int, key_factor.shape)
    if (
        batch <= 0
        or key_batch != batch
        or query_length > key_length
        or head_dim <= 0
        or head_dim % 2
        or key_dim != head_dim
        or query_heads <= 0
        or key_value_heads <= 0
        or query_heads % key_value_heads
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Factor state does not define a valid target GQA score geometry."
        )
    topology = upstream_fold.topology
    if (
        query_heads != topology.target_query_heads
        or key_value_heads != topology.target_key_value_heads
        or head_dim != topology.target_head_dim
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Factor-state Q/K geometry does not match the authenticated "
            "upstream target topology."
        )
    query_inputs = factor_state.target_query_inputs.to(torch.bfloat16).contiguous()
    key_inputs = factor_state.target_key_inputs.to(torch.bfloat16).contiguous()
    if tuple(query_inputs.shape) != (
        batch,
        query_length,
        int(query_inputs.shape[-1]),
    ) or tuple(key_inputs.shape) != (batch, key_length, int(query_inputs.shape[-1])):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Factor-state child input geometry differs across Q/K."
        )
    query_cos, query_sin = _validated_rope_pair(
        factor_state.target_query_rope_cos,
        factor_state.target_query_rope_sin,
        name="target query",
        expected_shape=(batch, query_length, head_dim),
        device=torch.device("cpu"),
    )
    key_cos, key_sin = _validated_rope_pair(
        factor_state.target_key_rope_cos,
        factor_state.target_key_rope_sin,
        name="target key",
        expected_shape=(batch, key_length, head_dim),
        device=torch.device("cpu"),
    )
    mask = _target_mask(
        production_case.causal_mask,
        batch=batch,
        query_heads=query_heads,
        query_length=query_length,
        key_length=key_length,
        device=torch.device("cpu"),
    )
    repeated_key = key_factor.repeat_interleave(
        query_heads // key_value_heads,
        dim=1,
    )
    scores = center_lfm2_causal_scores(
        torch.matmul(query_factor, repeated_key.transpose(-2, -1))
        / math.sqrt(head_dim),
        mask,
    ).to(torch.float32)
    query_weights = factor_state.query_row_sqrt_weights.to(torch.float32).contiguous()
    key_weights = factor_state.key_row_sqrt_weights.to(torch.float32).contiguous()
    if (
        tuple(query_weights.shape) != (batch, query_length)
        or tuple(key_weights.shape) != (batch, key_length)
        or bool((query_weights <= 0.0).any().item())
        or bool((key_weights <= 0.0).any().item())
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Factor-state solver row weights must be finite positive."
        )
    transform_tensors = {
        "target_query_inputs_sha256": _tensor_sha256(query_inputs),
        "target_key_inputs_sha256": _tensor_sha256(key_inputs),
        "target_query_rope_cos_sha256": _tensor_sha256(query_cos),
        "target_query_rope_sin_sha256": _tensor_sha256(query_sin),
        "target_key_rope_cos_sha256": _tensor_sha256(key_cos),
        "target_key_rope_sin_sha256": _tensor_sha256(key_sin),
        "query_row_sqrt_weights_sha256": _tensor_sha256(query_weights),
        "key_row_sqrt_weights_sha256": _tensor_sha256(key_weights),
        "target_causal_mask_sha256": _tensor_sha256(mask),
        "target_centered_scores_sha256": _tensor_sha256(scores),
    }
    informative_query_rows = mask.sum(dim=-1) > 1
    informative_cells = mask & informative_query_rows.unsqueeze(-1)
    informative_query_tokens = informative_query_rows.any(dim=1)
    informative_key_tokens = informative_cells.any(dim=(1, 2))
    solver_consumed = {
        "informative_query_rows_sha256": _tensor_sha256(informative_query_rows),
        "informative_cells_sha256": _tensor_sha256(informative_cells),
        "informative_query_inputs_sha256": _tensor_sha256(
            query_inputs[informative_query_tokens]
        ),
        "informative_key_inputs_sha256": _tensor_sha256(
            key_inputs[informative_key_tokens]
        ),
        "informative_query_rope_cos_sha256": _tensor_sha256(
            query_cos[informative_query_tokens]
        ),
        "informative_query_rope_sin_sha256": _tensor_sha256(
            query_sin[informative_query_tokens]
        ),
        "informative_key_rope_cos_sha256": _tensor_sha256(
            key_cos[informative_key_tokens]
        ),
        "informative_key_rope_sin_sha256": _tensor_sha256(
            key_sin[informative_key_tokens]
        ),
        "informative_query_row_sqrt_weights_sha256": _tensor_sha256(
            query_weights[informative_query_tokens]
        ),
        "informative_key_row_sqrt_weights_sha256": _tensor_sha256(
            key_weights[informative_key_tokens]
        ),
        "informative_target_centered_scores_sha256": _tensor_sha256(
            scores.masked_select(informative_cells)
        ),
    }
    numerical_rows: list[str] = []
    trajectory_rows: list[str] = []
    for batch_index in range(batch):
        for token in range(query_length):
            informative_heads = informative_query_rows[batch_index, :, token]
            if bool(informative_heads.any().item()):
                numerical_rows.append(
                    _canonical_sha256(
                        {
                            "role": "query",
                            "input": _tensor_sha256(query_inputs[batch_index, token]),
                            "rope_cos": _tensor_sha256(query_cos[batch_index, token]),
                            "rope_sin": _tensor_sha256(query_sin[batch_index, token]),
                            "weight": _tensor_sha256(
                                query_weights[batch_index, token].reshape(1)
                            ),
                            "informative_heads": _tensor_sha256(informative_heads),
                            "scores": _tensor_sha256(
                                scores[
                                    batch_index,
                                    informative_heads,
                                    token,
                                    :,
                                ]
                            ),
                            "mask": _tensor_sha256(
                                mask[
                                    batch_index,
                                    informative_heads,
                                    token,
                                    :,
                                ]
                            ),
                        }
                    )
                )
            trajectory_rows.append(
                _canonical_sha256(
                    {
                        "role": "query",
                        "trajectory_sha256": provenance.trajectory_sha256,
                        "batch_index": batch_index,
                        "token": token,
                    }
                )
            )
        for token in range(key_length):
            informative_key_cells = informative_cells[
                batch_index,
                :,
                :,
                token,
            ]
            if bool(informative_key_cells.any().item()):
                numerical_rows.append(
                    _canonical_sha256(
                        {
                            "role": "key",
                            "input": _tensor_sha256(key_inputs[batch_index, token]),
                            "rope_cos": _tensor_sha256(key_cos[batch_index, token]),
                            "rope_sin": _tensor_sha256(key_sin[batch_index, token]),
                            "weight": _tensor_sha256(
                                key_weights[batch_index, token].reshape(1)
                            ),
                            "informative_cells": _tensor_sha256(informative_key_cells),
                            "scores": _tensor_sha256(
                                scores[batch_index, :, :, token].masked_select(
                                    informative_key_cells
                                )
                            ),
                        }
                    )
                )
            trajectory_rows.append(
                _canonical_sha256(
                    {
                        "role": "key",
                        "trajectory_sha256": provenance.trajectory_sha256,
                        "batch_index": batch_index,
                        "token": token,
                    }
                )
            )
    factor_digest = _canonical_sha256(factor_state.as_digest_dict())
    input_bundle = _canonical_sha256(
        {
            "production_case_sha256": production_sha,
            "factor_state_sha256": factor_digest,
            "query_factor_sha256": _tensor_sha256(
                factor_state.local_query_post_rope_factor
            ),
            "key_factor_sha256": _tensor_sha256(
                factor_state.local_key_post_rope_factor
            ),
            **{
                key: value
                for key, value in transform_tensors.items()
                if "scores" not in key
            },
        }
    )
    output_bundle = _canonical_sha256(
        {
            "target_centered_scores_sha256": transform_tensors[
                "target_centered_scores_sha256"
            ],
            "target_causal_mask_sha256": transform_tensors["target_causal_mask_sha256"],
        }
    )
    return {
        "upstream_fold_report_sha256": upstream["report_sha256"],
        "upstream_factor_state_bundle_sha256": upstream["factor_state_bundle_sha256"],
        "upstream_factor_state_index": upstream["factor_state_index"],
        "production_case_sha256": production_sha,
        "factor_state_sha256": factor_digest,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
        "query_inputs": query_inputs,
        "key_inputs": key_inputs,
        "query_cosine": query_cos,
        "query_sine": query_sin,
        "key_cosine": key_cos,
        "key_sine": key_sin,
        "query_weights": query_weights,
        "key_weights": key_weights,
        "mask": mask,
        "scores": scores,
        "transform_input_bundle_sha256": input_bundle,
        "transform_output_bundle_sha256": output_bundle,
        "solver_content_sha256": _canonical_sha256(solver_consumed),
        "solver_row_sha256s": tuple(numerical_rows),
        "trajectory_row_sha256s": tuple(trajectory_rows),
        "solver_row_bundle_sha256": _canonical_sha256(numerical_rows),
        "trajectory_row_bundle_sha256": _canonical_sha256(trajectory_rows),
    }


@dataclass(frozen=True, slots=True)
class Lfm2QkScoreResidualCase:
    """Builder-derived production score evidence; raw score authoring is invalid."""

    production_case: Lfm2CpcfAttentionDimensionProductionCase
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold
    upstream_factor_state_index: int
    transform_receipt: Lfm2QkTargetScoreTransformReceipt
    target_query_residual_inputs: Any
    target_key_value_residual_inputs: Any
    target_centered_scores: Any
    target_causal_mask: Any
    query_rope_cos: Any
    query_rope_sin: Any
    key_rope_cos: Any
    key_rope_sin: Any
    query_row_sqrt_weights: Any
    key_value_row_sqrt_weights: Any
    solver_row_sha256s: tuple[str, ...]
    trajectory_row_sha256s: tuple[str, ...]
    content_sha256: str = ""
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.production_case,
            Lfm2CpcfAttentionDimensionProductionCase,
        ):
            raise TypeError(
                "production_case must be Lfm2CpcfAttentionDimensionProductionCase."
            )
        if not isinstance(
            self.upstream_fold,
            Lfm2CompiledCpcfAttentionDimensionFold,
        ):
            raise TypeError(
                "upstream_fold must be Lfm2CompiledCpcfAttentionDimensionFold."
            )
        if (
            isinstance(self.upstream_factor_state_index, bool)
            or not isinstance(self.upstream_factor_state_index, int)
            or self.upstream_factor_state_index < 0
            or self.upstream_factor_state_index
            >= len(self.upstream_fold.factor_sufficient_states)
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "upstream_factor_state_index does not select an authenticated "
                "factor state."
            )
        factor_state = self.upstream_fold.factor_sufficient_states[
            self.upstream_factor_state_index
        ]
        if not isinstance(
            self.transform_receipt,
            Lfm2QkTargetScoreTransformReceipt,
        ):
            raise TypeError(
                "transform_receipt must be Lfm2QkTargetScoreTransformReceipt."
            )
        payload = _derive_target_score_payload(
            self.production_case,
            factor_state,
            self.upstream_fold,
        )
        receipt = self.transform_receipt
        receipt.verify()
        provenance = self.production_case.provenance
        expected_receipt = Lfm2QkTargetScoreTransformReceipt(
            source_split_sha256=provenance.source_split_sha256,
            source_runtime_sha256=provenance.source_runtime_sha256,
            trajectory_sha256=provenance.trajectory_sha256,
            partition=provenance.partition,
            case_ordinal=provenance.case_ordinal,
            upstream_fold_report_sha256=payload["upstream_fold_report_sha256"],
            upstream_factor_state_bundle_sha256=payload[
                "upstream_factor_state_bundle_sha256"
            ],
            upstream_factor_state_index=payload["upstream_factor_state_index"],
            production_case_sha256=payload["production_case_sha256"],
            factor_state_sha256=payload["factor_state_sha256"],
            query_heads=payload["query_heads"],
            key_value_heads=payload["key_value_heads"],
            head_dim=payload["head_dim"],
            transform_input_bundle_sha256=payload["transform_input_bundle_sha256"],
            transform_output_bundle_sha256=payload["transform_output_bundle_sha256"],
            solver_content_sha256=payload["solver_content_sha256"],
            solver_row_bundle_sha256=payload["solver_row_bundle_sha256"],
            trajectory_row_bundle_sha256=payload["trajectory_row_bundle_sha256"],
        )
        supplied_tensors = (
            self.target_query_residual_inputs,
            self.target_key_value_residual_inputs,
            self.target_centered_scores,
            self.target_causal_mask,
            self.query_rope_cos,
            self.query_rope_sin,
            self.key_rope_cos,
            self.key_rope_sin,
            self.query_row_sqrt_weights,
            self.key_value_row_sqrt_weights,
        )
        expected_tensors = (
            payload["query_inputs"],
            payload["key_inputs"],
            payload["scores"],
            payload["mask"],
            payload["query_cosine"],
            payload["query_sine"],
            payload["key_cosine"],
            payload["key_sine"],
            payload["query_weights"],
            payload["key_weights"],
        )
        if (
            expected_receipt.transform_receipt_sha256
            != receipt.transform_receipt_sha256
            or any(
                not _same_numerical_tensor(supplied, expected)
                for supplied, expected in zip(
                    supplied_tensors,
                    expected_tensors,
                    strict=True,
                )
            )
            or tuple(self.solver_row_sha256s) != payload["solver_row_sha256s"]
            or tuple(self.trajectory_row_sha256s) != payload["trajectory_row_sha256s"]
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Raw or altered target-score evidence cannot satisfy the typed "
                "factor-state transform receipt."
            )
        content_sha = payload["solver_content_sha256"]
        if self.content_sha256 and self.content_sha256 != content_sha:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Q/K solver-consumed content fingerprint does not match."
            )
        body = {
            "transform_receipt_sha256": receipt.transform_receipt_sha256,
            "solver_content_sha256": content_sha,
            "solver_row_sha256s": list(payload["solver_row_sha256s"]),
            "trajectory_row_sha256s": list(payload["trajectory_row_sha256s"]),
        }
        computed = _canonical_sha256(body)
        if self.evidence_sha256 and self.evidence_sha256 != computed:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Q/K score-residual case does not match its seal."
            )
        for name, value in zip(
            (
                "target_query_residual_inputs",
                "target_key_value_residual_inputs",
                "target_centered_scores",
                "target_causal_mask",
                "query_rope_cos",
                "query_rope_sin",
                "key_rope_cos",
                "key_rope_sin",
                "query_row_sqrt_weights",
                "key_value_row_sqrt_weights",
            ),
            expected_tensors,
            strict=True,
        ):
            object.__setattr__(self, name, value.detach().cpu().contiguous().clone())
        object.__setattr__(self, "solver_row_sha256s", payload["solver_row_sha256s"])
        object.__setattr__(
            self,
            "trajectory_row_sha256s",
            payload["trajectory_row_sha256s"],
        )
        object.__setattr__(self, "content_sha256", content_sha)
        object.__setattr__(self, "evidence_sha256", computed)

    @property
    def factor_state(self) -> Lfm2CpcfAttentionFactorSufficientState:
        return self.upstream_fold.factor_sufficient_states[
            self.upstream_factor_state_index
        ]

    @property
    def transform_receipt_sha256(self) -> str:
        return self.transform_receipt.transform_receipt_sha256

    def verify(self) -> str:
        rebuilt = build_lfm2_qk_score_residual_case(
            production_case=self.production_case,
            upstream_fold=self.upstream_fold,
            factor_state=self.factor_state,
        )
        if rebuilt.evidence_sha256 != self.evidence_sha256:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Q/K score-residual case changed after sealing."
            )
        return self.evidence_sha256


def build_lfm2_qk_score_residual_case(
    *,
    production_case: Lfm2CpcfAttentionDimensionProductionCase,
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
    factor_state: Lfm2CpcfAttentionFactorSufficientState,
) -> Lfm2QkScoreResidualCase:
    """Derive and seal target scores from an authenticated CPCF factor state."""

    upstream = _authenticate_upstream_factor_member(
        upstream_fold,
        factor_state,
        production_case,
    )
    payload = _derive_target_score_payload(
        production_case,
        factor_state,
        upstream_fold,
    )
    provenance = production_case.provenance
    receipt = Lfm2QkTargetScoreTransformReceipt(
        source_split_sha256=provenance.source_split_sha256,
        source_runtime_sha256=provenance.source_runtime_sha256,
        trajectory_sha256=provenance.trajectory_sha256,
        partition=provenance.partition,
        case_ordinal=provenance.case_ordinal,
        upstream_fold_report_sha256=upstream["report_sha256"],
        upstream_factor_state_bundle_sha256=upstream["factor_state_bundle_sha256"],
        upstream_factor_state_index=upstream["factor_state_index"],
        production_case_sha256=payload["production_case_sha256"],
        factor_state_sha256=payload["factor_state_sha256"],
        query_heads=payload["query_heads"],
        key_value_heads=payload["key_value_heads"],
        head_dim=payload["head_dim"],
        transform_input_bundle_sha256=payload["transform_input_bundle_sha256"],
        transform_output_bundle_sha256=payload["transform_output_bundle_sha256"],
        solver_content_sha256=payload["solver_content_sha256"],
        solver_row_bundle_sha256=payload["solver_row_bundle_sha256"],
        trajectory_row_bundle_sha256=payload["trajectory_row_bundle_sha256"],
    )
    return Lfm2QkScoreResidualCase(
        production_case=production_case,
        upstream_fold=upstream_fold,
        upstream_factor_state_index=upstream["factor_state_index"],
        transform_receipt=receipt,
        target_query_residual_inputs=payload["query_inputs"],
        target_key_value_residual_inputs=payload["key_inputs"],
        target_centered_scores=payload["scores"],
        target_causal_mask=payload["mask"],
        query_rope_cos=payload["query_cosine"],
        query_rope_sin=payload["query_sine"],
        key_rope_cos=payload["key_cosine"],
        key_rope_sin=payload["key_sine"],
        query_row_sqrt_weights=payload["query_weights"],
        key_value_row_sqrt_weights=payload["key_weights"],
        solver_row_sha256s=payload["solver_row_sha256s"],
        trajectory_row_sha256s=payload["trajectory_row_sha256s"],
    )


QkEvidencePartition = Literal["fit", "fit_check"]


@dataclass(frozen=True, slots=True)
class Lfm2QkScoreResidualEvidenceBank:
    """One sealed local partition of the literal upstream FIT trajectories."""

    partition: QkEvidencePartition
    cases: tuple[Lfm2QkScoreResidualCase, ...]
    source_split_sha256: str
    source_runtime_sha256: str
    source_artifact_sha256: str
    transform_receipt_sha256: str = ""
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if self.partition not in {"fit", "fit_check"}:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Evidence partition must be exactly 'fit' or 'fit_check'."
            )
        values = tuple(self.cases)
        if not values or not all(
            isinstance(value, Lfm2QkScoreResidualCase) for value in values
        ):
            raise Lfm2CachedQkScoreResidualFoldError(
                "Evidence bank requires at least one sealed score-residual case."
            )
        source_split = _required_sha256(
            self.source_split_sha256,
            "source_split_sha256",
        )
        source_runtime = _required_sha256(
            self.source_runtime_sha256,
            "source_runtime_sha256",
        )
        source_artifact = _required_sha256(
            self.source_artifact_sha256,
            "source_artifact_sha256",
        )
        phase_receipts: list[tuple[str, int]] = []
        rows: list[dict[str, Any]] = []
        upstream_fold_receipts: set[str] = set()
        upstream_factor_bundles: set[str] = set()
        for case in values:
            case_sha = case.verify()
            provenance = case.production_case.provenance
            if (
                provenance.partition != self.partition
                or provenance.upstream_split != "fit"
                or provenance.source_split_sha256 != source_split
                or provenance.source_runtime_sha256 != source_runtime
                or case.transform_receipt.source_split_sha256 != source_split
                or case.transform_receipt.source_runtime_sha256 != source_runtime
            ):
                raise Lfm2CachedQkScoreResidualAuthenticationError(
                    "Local evidence partition disagrees with production provenance."
                )
            upstream_fold_receipts.add(
                case.transform_receipt.upstream_fold_report_sha256
            )
            upstream_factor_bundles.add(
                case.transform_receipt.upstream_factor_state_bundle_sha256
            )
            phase_receipts.append(
                (
                    provenance.trajectory_sha256,
                    provenance.case_ordinal,
                )
            )
            rows.append(
                {
                    "case_ordinal": provenance.case_ordinal,
                    "trajectory_sha256": provenance.trajectory_sha256,
                    "case_sha256": case_sha,
                    "content_sha256": case.content_sha256,
                    "transform_receipt_sha256": case.transform_receipt_sha256,
                }
            )
        if len(upstream_fold_receipts) != 1 or len(upstream_factor_bundles) != 1:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "One evidence bank cannot mix authenticated upstream CPCF "
                "folds or factor bundles."
            )
        if len(set(phase_receipts)) != len(phase_receipts):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "A local evidence bank repeats one trajectory-phase receipt."
            )
        transform_receipt = _canonical_sha256(
            [case.transform_receipt_sha256 for case in values]
        )
        if (
            self.transform_receipt_sha256
            and self.transform_receipt_sha256 != transform_receipt
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Q/K evidence transform-receipt bundle does not match."
            )
        body = {
            "partition": self.partition,
            "upstream_split": "fit",
            "source_split_sha256": source_split,
            "source_runtime_sha256": source_runtime,
            "source_artifact_sha256": source_artifact,
            "upstream_fold_report_sha256": next(iter(upstream_fold_receipts)),
            "upstream_factor_state_bundle_sha256": next(iter(upstream_factor_bundles)),
            "transform_receipt_sha256": transform_receipt,
            "rows": rows,
        }
        computed = _canonical_sha256(body)
        if self.evidence_sha256 and self.evidence_sha256 != computed:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Q/K evidence bank does not match its seal."
            )
        object.__setattr__(self, "cases", values)
        object.__setattr__(self, "source_split_sha256", source_split)
        object.__setattr__(self, "source_runtime_sha256", source_runtime)
        object.__setattr__(self, "source_artifact_sha256", source_artifact)
        object.__setattr__(self, "transform_receipt_sha256", transform_receipt)
        object.__setattr__(self, "evidence_sha256", computed)

    @property
    def trajectory_sha256s(self) -> tuple[str, ...]:
        return tuple(
            value.production_case.provenance.trajectory_sha256 for value in self.cases
        )

    @property
    def content_sha256s(self) -> tuple[str, ...]:
        return tuple(value.content_sha256 for value in self.cases)

    @property
    def solver_row_sha256s(self) -> tuple[str, ...]:
        return tuple(row for case in self.cases for row in case.solver_row_sha256s)

    @property
    def trajectory_row_sha256s(self) -> tuple[str, ...]:
        return tuple(row for case in self.cases for row in case.trajectory_row_sha256s)

    @property
    def upstream_fold_report_sha256(self) -> str:
        return self.cases[0].transform_receipt.upstream_fold_report_sha256

    @property
    def upstream_factor_state_bundle_sha256(self) -> str:
        return self.cases[0].transform_receipt.upstream_factor_state_bundle_sha256

    def verify(self) -> str:
        rebuilt = Lfm2QkScoreResidualEvidenceBank(
            partition=self.partition,
            cases=self.cases,
            source_split_sha256=self.source_split_sha256,
            source_runtime_sha256=self.source_runtime_sha256,
            source_artifact_sha256=self.source_artifact_sha256,
            transform_receipt_sha256=self.transform_receipt_sha256,
        )
        if rebuilt.evidence_sha256 != self.evidence_sha256:
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Q/K evidence bank changed after sealing."
            )
        return self.evidence_sha256


def seal_lfm2_qk_score_residual_fit_bank(
    cases: Sequence[Lfm2QkScoreResidualCase],
    *,
    source_split_sha256: str,
    source_runtime_sha256: str,
    source_artifact_sha256: str,
) -> Lfm2QkScoreResidualEvidenceBank:
    return Lfm2QkScoreResidualEvidenceBank(
        partition="fit",
        cases=tuple(cases),
        source_split_sha256=source_split_sha256,
        source_runtime_sha256=source_runtime_sha256,
        source_artifact_sha256=source_artifact_sha256,
    )


def seal_lfm2_qk_score_residual_fit_check_bank(
    cases: Sequence[Lfm2QkScoreResidualCase],
    *,
    source_split_sha256: str,
    source_runtime_sha256: str,
    source_artifact_sha256: str,
) -> Lfm2QkScoreResidualEvidenceBank:
    return Lfm2QkScoreResidualEvidenceBank(
        partition="fit_check",
        cases=tuple(cases),
        source_split_sha256=source_split_sha256,
        source_runtime_sha256=source_runtime_sha256,
        source_artifact_sha256=source_artifact_sha256,
    )


def _evidence_case_nbytes(case: Lfm2QkScoreResidualCase) -> int:
    return sum(
        _tensor_nbytes(getattr(case, name))
        for name in (
            "target_query_residual_inputs",
            "target_key_value_residual_inputs",
            "target_centered_scores",
            "target_causal_mask",
            "query_rope_cos",
            "query_rope_sin",
            "key_rope_cos",
            "key_rope_sin",
            "query_row_sqrt_weights",
            "key_value_row_sqrt_weights",
        )
    )


def _informative_query_row_count(
    cases: Sequence[Lfm2QkScoreResidualCase],
) -> int:
    return sum(
        int((case.target_causal_mask.sum(dim=-1) > 1).sum().item()) for case in cases
    )


@dataclass(frozen=True, slots=True)
class Lfm2QkScoreResidualFoldPolicy:
    """Fixed bounded analytical solve and exact-dtype selection policy."""

    ridge: float = 1e-3
    trust_radius: float = _MAXIMUM_TRUST_RADIUS
    maximum_sweeps: int = 2
    maximum_pcg_iterations: int = 128
    minimum_pcg_iterations: int = 8
    pcg_restart_iterations: int = 64
    pcg_relative_tolerance: float = 1e-5
    alpha_frontier: tuple[float, ...] = (
        0.0,
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    route_multiplier_clip: float = 8.0
    metric_regression_tolerance: float = 1e-5
    minimum_fit_score_improvement: float = 1e-8
    minimum_fit_check_score_improvement: float = 0.0
    maximum_fit_centered_score_relative_rmse: float = 0.35
    maximum_fit_probability_relative_rmse: float = 0.20
    maximum_fit_js_divergence_mean: float = 0.05
    maximum_fit_js_divergence_worst: float = 0.10
    minimum_fit_top1_route_agreement: float = 0.90
    maximum_fit_check_centered_score_relative_rmse: float = 0.35
    maximum_fit_check_probability_relative_rmse: float = 0.20
    maximum_fit_check_js_divergence_mean: float = 0.05
    maximum_fit_check_js_divergence_worst: float = 0.10
    minimum_fit_check_top1_route_agreement: float = 0.90
    maximum_fit_cases: int = _MAXIMUM_FIT_CASES
    maximum_fit_check_cases: int = _MAXIMUM_FIT_CHECK_CASES
    maximum_score_elements_per_case: int = _MAXIMUM_SCORE_ELEMENTS_PER_CASE
    maximum_aggregate_score_elements: int = _MAXIMUM_AGGREGATE_SCORE_ELEMENTS
    maximum_aggregate_evidence_bytes: int = _MAXIMUM_AGGREGATE_EVIDENCE_BYTES
    solver_device: str = "cpu"
    compute_devices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ridge", _finite_positive(self.ridge, "ridge"))
        object.__setattr__(
            self,
            "trust_radius",
            _finite_positive(self.trust_radius, "trust_radius"),
        )
        if self.trust_radius > _MAXIMUM_TRUST_RADIUS:
            raise Lfm2CachedQkScoreResidualFoldError(
                f"trust_radius exceeds the hard {_MAXIMUM_TRUST_RADIUS:g} ceiling."
            )
        maximum_sweeps = _positive_int(self.maximum_sweeps, "maximum_sweeps")
        if maximum_sweeps > 2:
            raise Lfm2CachedQkScoreResidualFoldError(
                "The analytical Q/K correction permits at most two sweeps."
            )
        object.__setattr__(self, "maximum_sweeps", maximum_sweeps)
        for name in (
            "maximum_pcg_iterations",
            "minimum_pcg_iterations",
            "pcg_restart_iterations",
            "maximum_fit_cases",
            "maximum_fit_check_cases",
            "maximum_score_elements_per_case",
            "maximum_aggregate_score_elements",
            "maximum_aggregate_evidence_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        hard_integer_ceilings = {
            "maximum_pcg_iterations": _MAXIMUM_PCG_ITERATIONS,
            "minimum_pcg_iterations": _MAXIMUM_PCG_ITERATIONS,
            "pcg_restart_iterations": _MAXIMUM_PCG_ITERATIONS,
            "maximum_fit_cases": _MAXIMUM_FIT_CASES,
            "maximum_fit_check_cases": _MAXIMUM_FIT_CHECK_CASES,
            "maximum_score_elements_per_case": _MAXIMUM_SCORE_ELEMENTS_PER_CASE,
            "maximum_aggregate_score_elements": _MAXIMUM_AGGREGATE_SCORE_ELEMENTS,
            "maximum_aggregate_evidence_bytes": _MAXIMUM_AGGREGATE_EVIDENCE_BYTES,
        }
        for name, ceiling in hard_integer_ceilings.items():
            if getattr(self, name) > ceiling:
                raise Lfm2CachedQkScoreResidualFoldError(
                    f"{name} exceeds its hard analytical ceiling {ceiling}."
                )
        if (
            self.minimum_pcg_iterations > self.maximum_pcg_iterations
            or self.pcg_restart_iterations > self.maximum_pcg_iterations
        ):
            raise Lfm2CachedQkScoreResidualFoldError(
                "PCG minimum/restart iterations exceed the hard iteration cap."
            )
        object.__setattr__(
            self,
            "pcg_relative_tolerance",
            _finite_positive(
                self.pcg_relative_tolerance,
                "pcg_relative_tolerance",
            ),
        )
        alphas = tuple(
            _finite_nonnegative(value, "alpha_frontier")
            for value in self.alpha_frontier
        )
        if (
            not alphas
            or alphas[0] != 0.0
            or any(value > 1.0 for value in alphas)
            or tuple(sorted(set(alphas))) != alphas
            or len(alphas) > _MAXIMUM_FRONTIER_POINTS
        ):
            raise Lfm2CachedQkScoreResidualFoldError(
                "alpha_frontier must be sorted, unique, start at zero, and "
                f"remain inside [0,1] with at most {_MAXIMUM_FRONTIER_POINTS} "
                "points."
            )
        object.__setattr__(self, "alpha_frontier", alphas)
        object.__setattr__(
            self,
            "route_multiplier_clip",
            _finite_positive(
                self.route_multiplier_clip,
                "route_multiplier_clip",
            ),
        )
        for name in (
            "metric_regression_tolerance",
            "minimum_fit_score_improvement",
            "minimum_fit_check_score_improvement",
            "maximum_fit_centered_score_relative_rmse",
            "maximum_fit_probability_relative_rmse",
            "maximum_fit_js_divergence_mean",
            "maximum_fit_js_divergence_worst",
            "minimum_fit_top1_route_agreement",
            "maximum_fit_check_centered_score_relative_rmse",
            "maximum_fit_check_probability_relative_rmse",
            "maximum_fit_check_js_divergence_mean",
            "maximum_fit_check_js_divergence_worst",
            "minimum_fit_check_top1_route_agreement",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        for name in (
            "minimum_fit_top1_route_agreement",
            "minimum_fit_check_top1_route_agreement",
        ):
            if getattr(self, name) > 1.0 or getattr(self, name) < _HARD_MINIMUM_TOP1:
                raise Lfm2CachedQkScoreResidualFoldError(
                    f"{name} cannot relax the hard production floor "
                    f"{_HARD_MINIMUM_TOP1:g}."
                )
        hard_metric_ceilings = {
            "maximum_fit_centered_score_relative_rmse": (_HARD_MAXIMUM_SCORE_RRMSE),
            "maximum_fit_check_centered_score_relative_rmse": (
                _HARD_MAXIMUM_SCORE_RRMSE
            ),
            "maximum_fit_probability_relative_rmse": (_HARD_MAXIMUM_PROBABILITY_RRMSE),
            "maximum_fit_check_probability_relative_rmse": (
                _HARD_MAXIMUM_PROBABILITY_RRMSE
            ),
            "maximum_fit_js_divergence_mean": _HARD_MAXIMUM_JS_MEAN,
            "maximum_fit_check_js_divergence_mean": _HARD_MAXIMUM_JS_MEAN,
            "maximum_fit_js_divergence_worst": _HARD_MAXIMUM_JS_WORST,
            "maximum_fit_check_js_divergence_worst": _HARD_MAXIMUM_JS_WORST,
        }
        for name, ceiling in hard_metric_ceilings.items():
            if getattr(self, name) > ceiling:
                raise Lfm2CachedQkScoreResidualFoldError(
                    f"{name} cannot relax the hard production ceiling {ceiling:g}."
                )
        if not isinstance(self.solver_device, str) or not self.solver_device:
            raise Lfm2CachedQkScoreResidualFoldError(
                "solver_device must be a nonempty torch device string."
            )
        try:
            solver_device = _torch().device(self.solver_device)
        except (RuntimeError, ValueError) as exc:
            raise Lfm2CachedQkScoreResidualFoldError(
                "solver_device is not a valid torch device string."
            ) from exc
        if solver_device.type == "cuda" and solver_device.index is None:
            raise Lfm2CachedQkScoreResidualFoldError(
                "CUDA solver_device must name an explicit device index."
            )
        canonical_solver = str(solver_device)
        object.__setattr__(self, "solver_device", canonical_solver)
        raw_compute_devices = self.compute_devices or (canonical_solver,)
        if (
            not isinstance(raw_compute_devices, tuple)
            or not raw_compute_devices
            or not all(
                isinstance(value, str) and value for value in raw_compute_devices
            )
        ):
            raise Lfm2CachedQkScoreResidualFoldError(
                "compute_devices must be a nonempty tuple of torch device strings."
            )
        try:
            compute_devices = tuple(
                _torch().device(value) for value in raw_compute_devices
            )
        except (RuntimeError, ValueError) as exc:
            raise Lfm2CachedQkScoreResidualFoldError(
                "compute_devices contains an invalid torch device string."
            ) from exc
        if any(
            value.type == "cuda" and value.index is None for value in compute_devices
        ):
            raise Lfm2CachedQkScoreResidualFoldError(
                "CUDA compute_devices must name explicit device indices."
            )
        canonical_compute = tuple(str(value) for value in compute_devices)
        if len(set(canonical_compute)) != len(canonical_compute):
            raise Lfm2CachedQkScoreResidualFoldError(
                "compute_devices must not repeat a device."
            )
        object.__setattr__(self, "compute_devices", canonical_compute)

    def as_dict(self) -> dict[str, Any]:
        return {
            name: (
                list(value) if name in {"alpha_frontier", "compute_devices"} else value
            )
            for name, value in (
                (field, getattr(self, field)) for field in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class Lfm2QkRoutingMetrics:
    case_count: int
    eligible_query_rows: int
    centered_score_relative_rmse: float
    probability_relative_rmse: float
    js_divergence_mean: float
    js_divergence_worst: float
    top1_route_agreement_fraction: float
    maximum_causal_probability_leakage: float

    def __post_init__(self) -> None:
        for name in ("case_count", "eligible_query_rows"):
            _positive_int(getattr(self, name), name)
        for name in (
            "centered_score_relative_rmse",
            "probability_relative_rmse",
            "js_divergence_mean",
            "js_divergence_worst",
            "maximum_causal_probability_leakage",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        top1 = _finite_nonnegative(
            self.top1_route_agreement_fraction,
            "top1_route_agreement_fraction",
        )
        if top1 > 1.0:
            raise Lfm2CachedQkScoreResidualFoldError(
                "top1_route_agreement_fraction must remain inside [0,1]."
            )
        object.__setattr__(self, "top1_route_agreement_fraction", top1)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Lfm2QkPcgReport:
    converged: bool
    iterations: int
    normal_operator_calls: int
    relative_normal_residual: float
    termination: str

    def __post_init__(self) -> None:
        if not isinstance(self.converged, bool):
            raise Lfm2CachedQkScoreResidualFoldError(
                "PCG converged flag must be boolean."
            )
        for name in ("iterations", "normal_operator_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Lfm2CachedQkScoreResidualFoldError(
                    f"{name} must be a nonnegative integer."
                )
        residual = float(self.relative_normal_residual)
        if math.isnan(residual) or residual < 0.0:
            raise Lfm2CachedQkScoreResidualFoldError(
                "relative_normal_residual must be nonnegative and not NaN."
            )
        object.__setattr__(self, "relative_normal_residual", residual)
        if not isinstance(self.termination, str) or not self.termination:
            raise Lfm2CachedQkScoreResidualFoldError(
                "PCG termination must be a nonempty string."
            )

    def as_dict(self) -> dict[str, Any]:
        body = {name: getattr(self, name) for name in self.__dataclass_fields__}
        residual_is_finite = math.isfinite(self.relative_normal_residual)
        body["relative_normal_residual"] = (
            self.relative_normal_residual if residual_is_finite else None
        )
        body["relative_normal_residual_finite"] = residual_is_finite
        return body


@dataclass(slots=True)
class _PreparedCase:
    source: Lfm2QkScoreResidualCase
    query_inputs: Any
    key_inputs: Any
    target_scores: Any
    target_probabilities: Any
    mask: Any
    informative_query_rows: Any
    query_cosine: Any
    query_sine: Any
    key_cosine: Any
    key_sine: Any
    cell_weights: Any
    query_linear: Any | None = None
    key_linear: Any | None = None
    query_post_rope: Any | None = None
    key_post_rope: Any | None = None
    baseline_scores: Any | None = None


@dataclass(frozen=True, slots=True)
class _StaticDeviceShard:
    device: Any
    indexed_cases: tuple[tuple[int, _PreparedCase], ...]


def _static_device_shards(
    cases: Sequence[_PreparedCase],
) -> tuple[_StaticDeviceShard, ...]:
    """Group round-robin cases by their already-authenticated device assignment."""

    ordered_cases = tuple(cases)
    devices: list[str] = []
    for case in ordered_cases:
        logical_device = str(case.query_inputs.device)
        if logical_device not in devices:
            devices.append(logical_device)
    shards = []
    for logical_device in devices:
        indexed_cases = tuple(
            (case_index, case)
            for case_index, case in enumerate(ordered_cases)
            if str(case.query_inputs.device) == logical_device
        )
        if not indexed_cases:
            raise Lfm2CachedQkScoreResidualFoldError(
                "A static score device shard is unexpectedly empty."
            )
        if any(
            str(case.key_inputs.device) != logical_device
            for _case_index, case in indexed_cases
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Prepared score cases do not remain on one assigned device."
            )
        shards.append(
            _StaticDeviceShard(
                device=_torch().device(logical_device),
                indexed_cases=indexed_cases,
            )
        )
    if sum(len(shard.indexed_cases) for shard in shards) != len(ordered_cases):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Static score device shards do not cover every prepared case."
        )
    return tuple(shards)


def _supports_concurrent_static_cuda_devices(devices: Sequence[Any]) -> bool:
    """Return whether distinct CUDA devices can execute independent shard work."""

    torch = _torch()
    resolved = tuple(torch.device(device) for device in devices)
    return (
        len(resolved) > 1
        and all(device.type == "cuda" for device in resolved)
        and len({str(device) for device in resolved}) == len(resolved)
    )


def _uses_concurrent_static_cuda_shards(
    shards: Sequence[_StaticDeviceShard],
) -> bool:
    return _supports_concurrent_static_cuda_devices(
        tuple(shard.device for shard in shards)
    )


def _execute_static_device_shards(
    shards: Sequence[_StaticDeviceShard],
    operation: Any,
) -> tuple[Any, ...]:
    """Launch independent CUDA shards together and return shard-order results."""

    ordered = tuple(shards)
    if not _uses_concurrent_static_cuda_shards(ordered):
        return tuple(operation(shard) for shard in ordered)

    torch = _torch()
    start_barrier = Barrier(len(ordered))

    def _worker(shard: _StaticDeviceShard) -> Any:
        start_barrier.wait()
        with torch.no_grad():
            if shard.device.type != "cuda":  # test-only forced-concurrency path
                return operation(shard)
            with torch.cuda.device(shard.device):
                result = operation(shard)
                # Queued CUDA work must finish before canonical solver reduction.
                torch.cuda.synchronize(shard.device)
                return result

    with ThreadPoolExecutor(
        max_workers=len(ordered),
        thread_name_prefix="lfm2-score-shard",
    ) as executor:
        futures = tuple(executor.submit(_worker, shard) for shard in ordered)
        # Completion order cannot influence the later global-case-order reduction.
        return tuple(future.result() for future in futures)


def _resolve_execution_devices(
    policy: Lfm2QkScoreResidualFoldPolicy,
) -> tuple[Any, tuple[Any, ...]]:
    torch = _torch()
    solver = torch.device(policy.solver_device)
    workers = tuple(torch.device(value) for value in policy.compute_devices)
    requested_cuda = tuple(
        value for value in (solver, *workers) if value.type == "cuda"
    )
    if requested_cuda and not torch.cuda.is_available():
        raise Lfm2CachedQkScoreResidualFoldError(
            "Requested CUDA execution devices are unavailable."
        )
    cuda_count = torch.cuda.device_count() if requested_cuda else 0
    if any(
        value.index is None or value.index < 0 or value.index >= cuda_count
        for value in requested_cuda
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "A requested CUDA execution device index is unavailable."
        )
    return solver, workers


def _observed_device_identity(device: Any) -> dict[str, Any]:
    torch = _torch()
    observed = torch.device(device)
    cuda_uuid: str | None = None
    cuda_name: str | None = None
    if observed.type == "cuda":
        if observed.index is None:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Observed CUDA execution must name an explicit logical device."
            )
        try:
            properties = torch.cuda.get_device_properties(observed.index)
            raw_uuid = getattr(properties, "uuid")
            raw_name = getattr(properties, "name")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Observed CUDA execution lacks a readable hardware UUID or name."
            ) from exc
        cuda_uuid = str(raw_uuid)
        cuda_name = str(raw_name)
        if not cuda_uuid or cuda_uuid == "None" or not cuda_name:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Observed CUDA execution lacks a nonempty hardware UUID or name."
            )
    return {
        "observed_logical_device": str(observed),
        "observed_device_type": observed.type,
        "observed_device_index": observed.index,
        "cuda_uuid": cuda_uuid,
        "cuda_name": cuda_name,
    }


def _force_observed_positive_scalar(
    value: Any,
    *,
    intended_device: Any,
    operation_name: str,
) -> tuple[float, dict[str, Any]]:
    torch = _torch()
    intended = torch.device(intended_device)
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.numel() != 1
        or value.device.type == "meta"
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            f"{operation_name} did not produce one dense observable tensor scalar."
        )
    observed = torch.device(value.device)
    if str(observed) != str(intended):
        raise Lfm2CachedQkScoreResidualFoldError(
            f"{operation_name} ran on {observed}, not requested device {intended}."
        )
    scalar = float(value.detach().item())
    if intended.type == "cuda":
        torch.cuda.synchronize(intended)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise Lfm2CachedQkScoreResidualFoldError(
            f"{operation_name} did not produce a finite positive execution witness."
        )
    return scalar, _observed_device_identity(observed)


def _build_observed_execution_receipt(
    *,
    seed: Lfm2AuthenticatedDenseQkSeed,
    fit_cases: Sequence[Lfm2QkScoreResidualCase],
    solver_device: Any,
    compute_devices: Sequence[Any],
) -> dict[str, Any]:
    """Force and seal tensor work on every requested logical execution device."""

    torch = _torch()
    solver = torch.device(solver_device)
    workers = tuple(torch.device(value) for value in compute_devices)
    if not workers:
        raise Lfm2CachedQkScoreResidualFoldError(
            "Observed execution requires at least one compute device."
        )
    if len(fit_cases) < len(workers):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Observed execution requires one distinct authenticated FIT case "
            "per requested compute device."
        )
    assigned_cases = tuple(fit_cases[index] for index in range(len(workers)))
    assigned_case_receipts = tuple(case.evidence_sha256 for case in assigned_cases)
    if any(not receipt for receipt in assigned_case_receipts) or len(
        set(assigned_case_receipts)
    ) != len(assigned_case_receipts):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Observed compute execution requires distinct nonempty FIT case receipts."
        )

    compute_observations: list[dict[str, Any]] = []
    with torch.no_grad():
        for assignment_index, (device, case) in enumerate(
            zip(workers, assigned_cases, strict=True)
        ):
            scores = case.target_centered_scores.detach().to(
                device=device,
                dtype=torch.float32,
            )
            mask = case.target_causal_mask.detach().to(device=device)
            recentered_scores = center_lfm2_causal_scores(scores, mask)
            witness_tensor = recentered_scores.square().sum(
                dtype=torch.float64
            ) + mask.sum(dtype=torch.float64)
            witness, identity = _force_observed_positive_scalar(
                witness_tensor,
                intended_device=device,
                operation_name="cached_centered_score_energy",
            )
            observation_body = {
                "assignment_index": assignment_index,
                "intended_logical_device": str(device),
                **identity,
                "cached_evidence_sha256": case.evidence_sha256,
                "cached_content_sha256": case.content_sha256,
                "cached_transform_receipt_sha256": (case.transform_receipt_sha256),
                "cached_trajectory_sha256": (
                    case.production_case.provenance.trajectory_sha256
                ),
                "cached_case_ordinal": (case.production_case.provenance.case_ordinal),
                "analytical_operation": (
                    "masked_centered_score_energy_plus_valid_cell_count"
                ),
                "analytical_tensor_element_count": int(scores.numel()),
                "analytical_scalar_dtype": str(witness_tensor.dtype),
                "analytical_scalar": witness,
                "forced_host_scalar_read": True,
            }
            compute_observations.append(
                {
                    **observation_body,
                    "case_execution_receipt_sha256": _canonical_sha256(
                        observation_body
                    ),
                }
            )

        query = seed.query_weight.detach().to(
            device=solver,
            dtype=torch.float32,
        )
        key = seed.key_weight.detach().to(
            device=solver,
            dtype=torch.float32,
        )
        solver_witness_tensor = (
            query.square().sum(dtype=torch.float64)
            + key.square().sum(dtype=torch.float64)
            + query.new_tensor(
                int(query.numel()) + int(key.numel()),
                dtype=torch.float64,
            )
        )
        solver_witness, solver_identity = _force_observed_positive_scalar(
            solver_witness_tensor,
            intended_device=solver,
            operation_name="solver_seed_qk_energy",
        )
    solver_body = {
        "intended_logical_device": str(solver),
        **solver_identity,
        "source_seed_sha256": seed.seed_sha256,
        "source_query_weight_sha256": _tensor_sha256(seed.query_weight),
        "source_key_weight_sha256": _tensor_sha256(seed.key_weight),
        "analytical_operation": "dense_seed_qk_energy_plus_parameter_count",
        "analytical_tensor_element_count": int(query.numel()) + int(key.numel()),
        "analytical_scalar_dtype": str(solver_witness_tensor.dtype),
        "analytical_scalar": solver_witness,
        "forced_host_scalar_read": True,
    }
    solver_observation = {
        **solver_body,
        "solver_execution_receipt_sha256": _canonical_sha256(solver_body),
    }

    observed_devices = [
        value["observed_logical_device"] for value in compute_observations
    ]
    requested_devices = [str(value) for value in workers]
    case_execution_receipts = [
        value["case_execution_receipt_sha256"] for value in compute_observations
    ]
    cuda_uuids = [
        value["cuda_uuid"]
        for value in compute_observations
        if value["observed_device_type"] == "cuda"
    ]
    if (
        observed_devices != requested_devices
        or len(set(case_execution_receipts)) != len(case_execution_receipts)
        or len(set(cuda_uuids)) != len(cuda_uuids)
        or solver_observation["observed_logical_device"] != str(solver)
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Observed tensor execution did not uniquely cover every requested "
            "logical device and authenticated FIT assignment."
        )
    body = {
        "schema": LFM2_QK_OBSERVED_EXECUTION_RECEIPT_SCHEMA,
        "requested_solver_device": str(solver),
        "requested_compute_devices": requested_devices,
        "requested_compute_device_count": len(requested_devices),
        "observed_compute_device_count": len(observed_devices),
        "every_requested_compute_device_observed": True,
        "solver_device_observed": True,
        "cached_evidence_analytical_tensor_operation_count": len(compute_observations),
        "solver_tensor_operation_count": 1,
        "distinct_case_receipts": list(assigned_case_receipts),
        "distinct_case_receipt_count": len(assigned_case_receipts),
        "distinct_case_execution_receipt_count": len(case_execution_receipts),
        "per_device_case_counts": {device: 1 for device in requested_devices},
        "compute_observations": compute_observations,
        "solver_observation": solver_observation,
        "operation_audit": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def _execution_report(
    solver_device: Any,
    compute_devices: Sequence[Any],
    observed_execution_receipt: Mapping[str, Any],
    *,
    fit_case_count: int,
) -> dict[str, Any]:
    return {
        "solver_device": str(solver_device),
        "compute_devices": [str(value) for value in compute_devices],
        "case_assignment": "authenticated_bank_order_round_robin",
        "compute_device_concurrency": "serial_round_robin",
        "streamed_evidence": False,
        "fixed_linearization_cache": True,
        "linearization_cache_scope": "one_gauss_newton_sweep",
        "maximum_logically_resident_prepared_cases": fit_case_count,
        "jvp_vjp_dtype": "torch.float32",
        "pcg_vector_dtype": "torch.float32",
        "scalar_reduction_dtype": "torch.float64",
        "observed_execution_receipt": dict(observed_execution_receipt),
    }


def _target_mask(
    mask: Any,
    *,
    batch: int,
    query_heads: int,
    query_length: int,
    key_length: int,
    device: Any,
) -> Any:
    torch = _torch()
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 4
        or tuple(mask.shape[0:1]) != (batch,)
        or tuple(mask.shape[2:]) != (query_length, key_length)
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Production causal mask has an incompatible shape."
        )
    value = mask.detach().to(device=device)
    if int(value.shape[1]) == query_heads:
        result = value
    elif int(value.shape[1]) == 1:
        result = value.expand(-1, query_heads, -1, -1)
    else:
        reference = value[:, :1]
        if not bool(torch.equal(value, reference.expand_as(value))):
            raise Lfm2CachedQkScoreResidualFoldError(
                "Source-head masks differ and cannot be projected to target heads."
            )
        result = reference.expand(-1, query_heads, -1, -1)
    offset = key_length - query_length
    query_position = torch.arange(query_length, device=device).reshape(1, 1, -1, 1)
    key_position = torch.arange(key_length, device=device).reshape(1, 1, 1, -1)
    if (
        offset < 0
        or bool((result & (key_position > offset + query_position)).any().item())
        or bool((~result.any(dim=-1)).any().item())
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Production mask exposes future keys or contains an empty causal row."
        )
    return result.contiguous()


def _validated_rope_pair(
    cosine: Any,
    sine: Any,
    *,
    name: str,
    expected_shape: tuple[int, int, int],
    device: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    if tuple(cosine.shape) != expected_shape or tuple(sine.shape) != expected_shape:
        raise Lfm2CachedQkScoreResidualFoldError(
            f"{name} RoPE terms differ from target batch/length/head_dim."
        )
    cos = cosine.to(device=device, dtype=torch.bfloat16).contiguous()
    sin = sine.to(device=device, dtype=torch.bfloat16).contiguous()
    half = expected_shape[-1] // 2
    if not torch.equal(cos[..., :half], cos[..., half:]) or not torch.equal(
        sin[..., :half], sin[..., half:]
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            f"{name} RoPE terms do not preserve duplicated-half frequency pairs."
        )
    radius = (
        cos[..., :half].to(torch.float32).square()
        + sin[..., :half].to(torch.float32).square()
    )
    if not bool(torch.isfinite(radius).all().item()) or bool(
        (radius <= torch.finfo(torch.float32).tiny).any().item()
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            f"{name} RoPE contains a zero or non-finite finite-precision radius."
        )
    return cos, sin


def _prepare_evidence_case(
    evidence: Lfm2QkScoreResidualCase,
    seed: Lfm2AuthenticatedDenseQkSeed,
    policy: Lfm2QkScoreResidualFoldPolicy,
    *,
    device: Any,
    authenticate: bool = True,
) -> _PreparedCase:
    torch = _torch()
    if authenticate:
        evidence.verify()
    batch, query_length, hidden = map(
        int,
        evidence.target_query_residual_inputs.shape,
    )
    key_length = int(evidence.target_key_value_residual_inputs.shape[1])
    expected_scores = (
        batch,
        seed.query_heads,
        query_length,
        key_length,
    )
    if (
        hidden != seed.hidden_size
        or tuple(evidence.target_key_value_residual_inputs.shape)
        != (batch, key_length, seed.hidden_size)
        or tuple(evidence.target_centered_scores.shape) != expected_scores
        or int(math.prod(expected_scores)) > policy.maximum_score_elements_per_case
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Target inputs/scores differ from the synchronized seed topology or "
            "exceed the per-case analytical ceiling."
        )
    mask = _target_mask(
        evidence.target_causal_mask,
        batch=batch,
        query_heads=seed.query_heads,
        query_length=query_length,
        key_length=key_length,
        device=device,
    )
    target_scores = evidence.target_centered_scores.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()
    if bool((target_scores.masked_select(~mask) != 0.0).any().item()):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Authenticated target scores contain a nonzero masked cell."
        )
    target_probabilities = _masked_probabilities(target_scores, mask)
    informative_query_rows = mask.sum(dim=-1) > 1
    query_cosine, query_sine = _validated_rope_pair(
        evidence.query_rope_cos,
        evidence.query_rope_sin,
        name="query",
        expected_shape=(batch, query_length, seed.head_dim),
        device=device,
    )
    key_cosine, key_sine = _validated_rope_pair(
        evidence.key_rope_cos,
        evidence.key_rope_sin,
        name="key",
        expected_shape=(batch, key_length, seed.head_dim),
        device=device,
    )
    query_sqrt = evidence.query_row_sqrt_weights.to(
        device=device,
        dtype=torch.float32,
    )
    key_sqrt = evidence.key_value_row_sqrt_weights.to(
        device=device,
        dtype=torch.float32,
    )
    if (
        tuple(query_sqrt.shape) != (batch, query_length)
        or tuple(key_sqrt.shape) != (batch, key_length)
        or bool((query_sqrt <= 0).any().item())
        or bool((key_sqrt <= 0).any().item())
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Q/K score residual folding requires positive aligned row weights."
        )
    route = target_probabilities * (1.0 - target_probabilities)
    route = torch.where(mask, route, torch.zeros_like(route))
    route_mean = route.sum(dim=-1, keepdim=True) / mask.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1)
    normalized_route = torch.where(
        route_mean > torch.finfo(torch.float64).tiny,
        route / route_mean.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(route),
    ).clamp(max=policy.route_multiplier_clip)
    normalized_mean = normalized_route.sum(dim=-1, keepdim=True) / mask.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1)
    route_multiplier = (1.0 + normalized_route) / (1.0 + normalized_mean)
    cell_weights = (
        query_sqrt.square().reshape(batch, 1, query_length, 1)
        * key_sqrt.square().reshape(batch, 1, 1, key_length)
        * route_multiplier.to(torch.float32)
    ).expand(-1, seed.query_heads, -1, -1)
    informative_cells = mask & informative_query_rows.unsqueeze(-1)
    cell_weights = torch.where(
        informative_cells,
        cell_weights,
        torch.zeros_like(cell_weights),
    )
    return _PreparedCase(
        source=evidence,
        query_inputs=evidence.target_query_residual_inputs.to(
            device=device,
            dtype=torch.bfloat16,
        ),
        key_inputs=evidence.target_key_value_residual_inputs.to(
            device=device,
            dtype=torch.bfloat16,
        ),
        target_scores=target_scores,
        target_probabilities=target_probabilities,
        mask=mask,
        informative_query_rows=informative_query_rows,
        query_cosine=query_cosine,
        query_sine=query_sine,
        key_cosine=key_cosine,
        key_sine=key_sine,
        cell_weights=cell_weights,
    )


def _iter_prepared_evidence_cases(
    evidence_cases: Sequence[Lfm2QkScoreResidualCase],
    seed: Lfm2AuthenticatedDenseQkSeed,
    policy: Lfm2QkScoreResidualFoldPolicy,
    compute_devices: Sequence[Any],
) -> Iterable[_PreparedCase]:
    devices = tuple(compute_devices)
    if not devices:
        raise Lfm2CachedQkScoreResidualFoldError(
            "At least one compute device is required for streamed evidence."
        )
    for index, evidence in enumerate(evidence_cases):
        yield _prepare_evidence_case(
            evidence,
            seed,
            policy,
            device=devices[index % len(devices)],
            authenticate=False,
        )


def _exact_runtime_rmsnorm(
    raw_rows: Any,
    norm_weight: Any,
    epsilon: float,
) -> Any:
    torch = _torch()
    dtype = raw_rows.dtype
    direction = raw_rows.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon)
    return (
        direction.to(dtype=dtype) * norm_weight.to(device=raw_rows.device, dtype=dtype)
    ).contiguous()


def _exact_replay(
    case: _PreparedCase,
    seed: Lfm2AuthenticatedDenseQkSeed,
    query_weight: Any,
    key_weight: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    torch = _torch()
    batch, query_length, _hidden = map(int, case.query_inputs.shape)
    key_length = int(case.key_inputs.shape[1])
    query = query_weight.to(
        device=case.query_inputs.device,
        dtype=torch.bfloat16,
    )
    key = key_weight.to(
        device=case.key_inputs.device,
        dtype=torch.bfloat16,
    )
    query_linear = (
        (case.query_inputs @ query.T)
        .reshape(
            batch,
            query_length,
            seed.query_heads,
            seed.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_linear = (
        (case.key_inputs @ key.T)
        .reshape(
            batch,
            key_length,
            seed.key_value_heads,
            seed.head_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    query_pre = _exact_runtime_rmsnorm(
        query_linear,
        seed.query_norm_weight,
        seed.norm_epsilon,
    )
    key_pre = _exact_runtime_rmsnorm(
        key_linear,
        seed.key_norm_weight,
        seed.norm_epsilon,
    )
    query_post = apply_lfm2_finite_radius_rope(
        query_pre,
        case.query_cosine,
        case.query_sine,
    )
    key_post = apply_lfm2_finite_radius_rope(
        key_pre,
        case.key_cosine,
        case.key_sine,
    )
    repeated_key = key_post.repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    raw_scores = torch.matmul(
        query_post,
        repeated_key.transpose(-2, -1),
    ) / math.sqrt(seed.head_dim)
    centered = center_lfm2_causal_scores(raw_scores, case.mask).to(torch.float32)
    return query_linear, key_linear, query_post, key_post, centered


def _attach_linearization(
    cases: Sequence[_PreparedCase],
    seed: Lfm2AuthenticatedDenseQkSeed,
    query_weight: Any,
    key_weight: Any,
) -> None:
    for case in cases:
        (
            case.query_linear,
            case.key_linear,
            case.query_post_rope,
            case.key_post_rope,
            case.baseline_scores,
        ) = _exact_replay(case, seed, query_weight, key_weight)


def _score_jvp(
    case: _PreparedCase,
    seed: Lfm2AuthenticatedDenseQkSeed,
    delta_query: Any,
    delta_key: Any,
) -> Any:
    torch = _torch()
    assert case.query_linear is not None
    assert case.key_linear is not None
    assert case.query_post_rope is not None
    assert case.key_post_rope is not None
    batch, query_length, _hidden = map(int, case.query_inputs.shape)
    key_length = int(case.key_inputs.shape[1])
    query_delta = delta_query.reshape(
        seed.query_heads,
        seed.head_dim,
        seed.hidden_size,
    )
    key_delta = delta_key.reshape(
        seed.key_value_heads,
        seed.head_dim,
        seed.hidden_size,
    )
    delta_query_linear = torch.einsum(
        "bqf,hdf->bhqd",
        case.query_inputs.to(torch.float32),
        query_delta,
    )
    delta_key_linear = torch.einsum(
        "bkf,gdf->bgkd",
        case.key_inputs.to(torch.float32),
        key_delta,
    )
    delta_query_pre = lfm2_continuous_rmsnorm_jvp(
        case.query_linear,
        delta_query_linear,
        seed.query_norm_weight,
        seed.norm_epsilon,
    )
    delta_key_pre = lfm2_continuous_rmsnorm_jvp(
        case.key_linear,
        delta_key_linear,
        seed.key_norm_weight,
        seed.norm_epsilon,
    )
    delta_query_post = apply_lfm2_finite_radius_rope(
        delta_query_pre,
        case.query_cosine.to(torch.float32),
        case.query_sine.to(torch.float32),
    )
    delta_key_post = apply_lfm2_finite_radius_rope(
        delta_key_pre,
        case.key_cosine.to(torch.float32),
        case.key_sine.to(torch.float32),
    )
    repeated_key = case.key_post_rope.to(torch.float32).repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    repeated_delta_key = delta_key_post.repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    delta_raw = (
        torch.matmul(
            delta_query_post,
            repeated_key.transpose(-2, -1),
        )
        + torch.matmul(
            case.query_post_rope.to(torch.float32),
            repeated_delta_key.transpose(-2, -1),
        )
    ) / math.sqrt(seed.head_dim)
    if tuple(delta_raw.shape) != (
        batch,
        seed.query_heads,
        query_length,
        key_length,
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "Internal score JVP produced an invalid target shape."
        )
    return center_lfm2_causal_scores(delta_raw, case.mask)


def _score_vjp(
    case: _PreparedCase,
    seed: Lfm2AuthenticatedDenseQkSeed,
    centered_score_cotangent: Any,
) -> tuple[Any, Any]:
    """Apply A^T; centering occurs after caller-supplied score weighting."""

    torch = _torch()
    assert case.query_linear is not None
    assert case.key_linear is not None
    assert case.query_post_rope is not None
    assert case.key_post_rope is not None
    cotangent = center_lfm2_causal_scores(
        centered_score_cotangent.to(torch.float32),
        case.mask,
    )
    repeated_key = case.key_post_rope.to(torch.float32).repeat_interleave(
        seed.queries_per_key_value,
        dim=1,
    )
    query_post = case.query_post_rope.to(torch.float32)
    query_post_cotangent = torch.matmul(cotangent, repeated_key) / math.sqrt(
        seed.head_dim
    )
    repeated_key_cotangent = torch.matmul(
        cotangent.transpose(-2, -1),
        query_post,
    ) / math.sqrt(seed.head_dim)
    batch, _heads, key_length, _dim = repeated_key_cotangent.shape
    key_post_cotangent = repeated_key_cotangent.reshape(
        batch,
        seed.key_value_heads,
        seed.queries_per_key_value,
        key_length,
        seed.head_dim,
    ).sum(dim=2)
    query_pre_cotangent = apply_lfm2_finite_radius_rope_transpose(
        query_post_cotangent,
        case.query_cosine.to(torch.float32),
        case.query_sine.to(torch.float32),
    )
    key_pre_cotangent = apply_lfm2_finite_radius_rope_transpose(
        key_post_cotangent,
        case.key_cosine.to(torch.float32),
        case.key_sine.to(torch.float32),
    )
    query_linear_cotangent = lfm2_continuous_rmsnorm_vjp(
        case.query_linear,
        query_pre_cotangent,
        seed.query_norm_weight,
        seed.norm_epsilon,
    )
    key_linear_cotangent = lfm2_continuous_rmsnorm_vjp(
        case.key_linear,
        key_pre_cotangent,
        seed.key_norm_weight,
        seed.norm_epsilon,
    )
    query_gradient = torch.einsum(
        "bhqd,bqf->hdf",
        query_linear_cotangent,
        case.query_inputs.to(torch.float32),
    ).reshape_as(
        seed.query_weight.to(
            device=query_linear_cotangent.device,
            dtype=torch.float32,
        )
    )
    key_gradient = torch.einsum(
        "bgkd,bkf->gdf",
        key_linear_cotangent,
        case.key_inputs.to(torch.float32),
    ).reshape_as(
        seed.key_weight.to(
            device=key_linear_cotangent.device,
            dtype=torch.float32,
        )
    )
    return query_gradient, key_gradient


class Lfm2QkScoreResidualLinearOperator:
    """Public matrix-free weighted score Jacobian and ridge normal operator."""

    def __init__(
        self,
        *,
        seed: Lfm2AuthenticatedDenseQkSeed,
        prepared_cases: Sequence[_PreparedCase],
        current_query_weight: Any,
        current_key_weight: Any,
        ridge: float,
    ) -> None:
        torch = _torch()
        self.seed = seed
        self.cases = tuple(prepared_cases)
        self.device = current_query_weight.device
        self.query_shape = tuple(seed.query_weight.shape)
        self.key_shape = tuple(seed.key_weight.shape)
        self.query_numel = int(seed.query_weight.numel())
        self.key_numel = int(seed.key_weight.numel())
        self.ridge = _finite_positive(ridge, "ridge")
        self.query_scale_squared = max(
            float(current_query_weight.to(torch.float64).square().sum().item()),
            self.query_numel * torch.finfo(torch.float32).eps,
        )
        self.key_scale_squared = max(
            float(current_key_weight.to(torch.float64).square().sum().item()),
            self.key_numel * torch.finfo(torch.float32).eps,
        )
        energy = 0.0
        for case in self.cases:
            assert case.baseline_scores is not None
            residual = case.target_scores - case.baseline_scores
            energy += float(
                (
                    case.cell_weights.to(torch.float64)
                    * residual.to(torch.float64).square()
                )
                .sum()
                .item()
            )
        self.residual_energy = max(energy, torch.finfo(torch.float64).tiny)
        self.normal_operator_calls = 0

    def pack(self, query: Any, key: Any) -> Any:
        return _torch().cat((query.reshape(-1), key.reshape(-1)))

    def unpack(self, vector: Any) -> tuple[Any, Any]:
        if int(vector.numel()) != self.query_numel + self.key_numel:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Packed Q/K correction has the wrong parameter count."
            )
        return (
            vector[: self.query_numel].reshape(self.query_shape),
            vector[self.query_numel :].reshape(self.key_shape),
        )

    def jvp(self, vector: Any) -> tuple[Any, ...]:
        delta_query, delta_key = self.unpack(vector)
        return tuple(
            _score_jvp(case, self.seed, delta_query, delta_key) for case in self.cases
        )

    def vjp(self, score_cotangents: Sequence[Any]) -> Any:
        torch = _torch()
        values = tuple(score_cotangents)
        if len(values) != len(self.cases):
            raise Lfm2CachedQkScoreResidualFoldError(
                "Score cotangents do not align with prepared FIT cases."
            )
        query = torch.zeros(
            self.query_shape,
            device=self.device,
            dtype=torch.float32,
        )
        key = torch.zeros(
            self.key_shape,
            device=self.device,
            dtype=torch.float32,
        )
        for case, cotangent in zip(self.cases, values, strict=True):
            query_delta, key_delta = _score_vjp(
                case,
                self.seed,
                cotangent,
            )
            query.add_(query_delta)
            key.add_(key_delta)
        return self.pack(query, key)

    def right_hand_side(self) -> Any:
        cotangents = []
        for case in self.cases:
            assert case.baseline_scores is not None
            residual = case.target_scores - case.baseline_scores
            cotangents.append(case.cell_weights * residual / self.residual_energy)
        return self.vjp(cotangents)

    def normal(self, vector: Any) -> Any:
        self.normal_operator_calls += 1
        delta_query, delta_key = self.unpack(vector)
        query = _torch().zeros_like(delta_query, dtype=_torch().float32)
        key = _torch().zeros_like(delta_key, dtype=_torch().float32)
        for case in self.cases:
            delta_scores = _score_jvp(
                case,
                self.seed,
                delta_query,
                delta_key,
            )
            query_delta, key_delta = _score_vjp(
                case,
                self.seed,
                case.cell_weights * delta_scores / self.residual_energy,
            )
            query.add_(query_delta)
            key.add_(key_delta)
        query.add_(
            delta_query,
            alpha=self.ridge / self.query_scale_squared,
        )
        key.add_(
            delta_key,
            alpha=self.ridge / self.key_scale_squared,
        )
        return self.pack(query, key)


class _StreamingQkScoreResidualLinearOperator:
    """Reuse one device-resident fixed linearization within a matrix-free sweep."""

    def __init__(
        self,
        *,
        seed: Lfm2AuthenticatedDenseQkSeed,
        evidence_cases: Sequence[Lfm2QkScoreResidualCase],
        current_query_weight: Any,
        current_key_weight: Any,
        ridge: float,
        policy: Lfm2QkScoreResidualFoldPolicy,
        solver_device: Any,
        compute_devices: Sequence[Any],
    ) -> None:
        torch = _torch()
        self.seed = seed
        self.evidence_cases = tuple(evidence_cases)
        self.current_query_weight = current_query_weight
        self.current_key_weight = current_key_weight
        self.policy = policy
        self.device = solver_device
        self.compute_devices = tuple(compute_devices)
        self.query_shape = tuple(seed.query_weight.shape)
        self.key_shape = tuple(seed.key_weight.shape)
        self.query_numel = int(seed.query_weight.numel())
        self.key_numel = int(seed.key_weight.numel())
        self.ridge = _finite_positive(ridge, "ridge")
        self.query_scale_squared = max(
            float(current_query_weight.to(torch.float64).square().sum().item()),
            self.query_numel * torch.finfo(torch.float32).eps,
        )
        self.key_scale_squared = max(
            float(current_key_weight.to(torch.float64).square().sum().item()),
            self.key_numel * torch.finfo(torch.float32).eps,
        )
        self.normal_operator_calls = 0
        self._successful_right_hand_side_calls = 0
        self._successful_normal_operator_calls = 0
        self._concurrent_right_hand_side_calls = 0
        self._concurrent_normal_operator_calls = 0
        self._linearization_cache = tuple(self._build_linearized_cases())
        if len(self._linearization_cache) != len(self.evidence_cases):
            raise Lfm2CachedQkScoreResidualFoldError(
                "Fixed-linearization cache does not cover every FIT case."
            )
        self.shards = _static_device_shards(self._linearization_cache)
        self.linearization_cache_resource_receipt = (
            self._build_linearization_cache_resource_receipt()
        )
        energy = 0.0
        for case in self._linearized_cases():
            assert case.baseline_scores is not None
            residual = case.target_scores - case.baseline_scores
            energy += float(
                (
                    case.cell_weights.to(torch.float64)
                    * residual.to(torch.float64).square()
                )
                .sum()
                .item()
            )
        self.residual_energy = max(energy, torch.finfo(torch.float64).tiny)

    def _build_linearized_cases(self) -> Iterable[_PreparedCase]:
        for case in _iter_prepared_evidence_cases(
            self.evidence_cases,
            self.seed,
            self.policy,
            self.compute_devices,
        ):
            _attach_linearization(
                (case,),
                self.seed,
                self.current_query_weight,
                self.current_key_weight,
            )
            yield case

    def _linearized_cases(self) -> Iterable[_PreparedCase]:
        yield from self._linearization_cache

    @staticmethod
    def _case_cached_tensors(case: _PreparedCase) -> tuple[Any, ...]:
        return (
            case.query_inputs,
            case.key_inputs,
            case.target_scores,
            case.target_probabilities,
            case.mask,
            case.informative_query_rows,
            case.query_cosine,
            case.query_sine,
            case.key_cosine,
            case.key_sine,
            case.cell_weights,
            case.query_linear,
            case.key_linear,
            case.query_post_rope,
            case.key_post_rope,
            case.baseline_scores,
        )

    def _build_linearization_cache_resource_receipt(self) -> dict[str, Any]:
        torch = _torch()
        requested_devices = [str(value) for value in self.compute_devices]
        per_device_case_counts = {device: 0 for device in requested_devices}
        per_device_logical_tensor_bytes = {device: 0 for device in requested_devices}
        for case_index, case in enumerate(self._linearization_cache):
            expected_device = requested_devices[case_index % len(requested_devices)]
            tensors = self._case_cached_tensors(case)
            if any(not isinstance(value, torch.Tensor) for value in tensors):
                raise Lfm2CachedQkScoreResidualFoldError(
                    "Fixed-linearization cache contains an absent tensor."
                )
            observed_devices = {str(value.device) for value in tensors}
            if observed_devices != {expected_device}:
                raise Lfm2CachedQkScoreResidualFoldError(
                    "Fixed-linearization cache differs from round-robin device "
                    "assignment."
                )
            per_device_case_counts[expected_device] += 1
            per_device_logical_tensor_bytes[expected_device] += sum(
                _tensor_nbytes(value) for value in tensors
            )
        total_bytes = sum(per_device_logical_tensor_bytes.values())
        if total_bytes <= 0 or sum(per_device_case_counts.values()) != len(
            self.evidence_cases
        ):
            raise Lfm2CachedQkScoreResidualFoldError(
                "Fixed-linearization cache resource accounting is empty."
            )
        successful_gradient_calls = (
            self._successful_right_hand_side_calls
            + self._successful_normal_operator_calls
        )
        concurrent_gradient_calls = (
            self._concurrent_right_hand_side_calls
            + self._concurrent_normal_operator_calls
        )
        concurrent_path_executed = concurrent_gradient_calls > 0
        if concurrent_path_executed:
            concurrency_mode = "thread_barrier_concurrent_static_cuda_shards"
        elif successful_gradient_calls > 0:
            concurrency_mode = "serial_round_robin"
        else:
            concurrency_mode = "not_executed"
        body = {
            "schema": LFM2_QK_LINEARIZATION_CACHE_RESOURCE_SCHEMA,
            "cache_scope": "one_gauss_newton_sweep",
            "resident_storage": "compute_device",
            "case_assignment": "authenticated_bank_order_round_robin",
            "static_device_shards": [
                {
                    "logical_device": str(shard.device),
                    "case_assignment_indices": [
                        case_index for case_index, _case in shard.indexed_cases
                    ],
                    "case_count": len(shard.indexed_cases),
                }
                for shard in self.shards
            ],
            "source_fit_case_count": len(self.evidence_cases),
            "resident_prepared_case_count": len(self._linearization_cache),
            "linearization_replay_call_count": len(self._linearization_cache),
            "per_device_case_counts": per_device_case_counts,
            "per_device_logical_tensor_bytes": per_device_logical_tensor_bytes,
            "total_logical_tensor_bytes": total_bytes,
            "maximum_device_logical_tensor_bytes": max(
                per_device_logical_tensor_bytes.values()
            ),
            "cache_reused_for_right_hand_side": True,
            "cache_reused_for_normal_operator_calls": True,
            "cache_invalidated_at_sweep_end": True,
            "compute_device_concurrency": concurrency_mode,
            "actual_multi_cuda_concurrent_path_executed": (concurrent_path_executed),
            "successful_right_hand_side_calls": (
                self._successful_right_hand_side_calls
            ),
            "successful_normal_operator_calls": (
                self._successful_normal_operator_calls
            ),
            "concurrent_right_hand_side_calls": (
                self._concurrent_right_hand_side_calls
            ),
            "concurrent_normal_operator_calls": (
                self._concurrent_normal_operator_calls
            ),
            "concurrent_per_case_gradients_retained_until_reduction": True,
            "solver_gradient_reduction_order": "original_global_fit_case_order",
            "nondeterministic_timing_fields_included": False,
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }
        return {
            **body,
            "resource_receipt_sha256": _canonical_sha256(body),
        }

    def _record_gradient_execution(
        self,
        *,
        operation: Literal["right_hand_side", "normal"],
        actual_multi_cuda_concurrent_path_executed: bool,
    ) -> None:
        if operation == "right_hand_side":
            self._successful_right_hand_side_calls += 1
            if actual_multi_cuda_concurrent_path_executed:
                self._concurrent_right_hand_side_calls += 1
        else:
            self._successful_normal_operator_calls += 1
            if actual_multi_cuda_concurrent_path_executed:
                self._concurrent_normal_operator_calls += 1
        self.linearization_cache_resource_receipt = (
            self._build_linearization_cache_resource_receipt()
        )

    def pack(self, query: Any, key: Any) -> Any:
        return _torch().cat((query.reshape(-1), key.reshape(-1)))

    def unpack(self, vector: Any) -> tuple[Any, Any]:
        if int(vector.numel()) != self.query_numel + self.key_numel:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Packed Q/K correction has the wrong parameter count."
            )
        return (
            vector[: self.query_numel].reshape(self.query_shape),
            vector[self.query_numel :].reshape(self.key_shape),
        )

    def _empty_gradient(self) -> tuple[Any, Any]:
        torch = _torch()
        return (
            torch.zeros(
                self.query_shape,
                device=self.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                self.key_shape,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def _canonical_reduce_per_case_gradients(
        self,
        shard_gradients: Sequence[Mapping[int, tuple[Any, Any]]],
    ) -> tuple[Any, Any]:
        by_case: dict[int, tuple[Any, Any]] = {}
        for shard_result in shard_gradients:
            for case_index, gradients in shard_result.items():
                if case_index in by_case:
                    raise Lfm2CachedQkScoreResidualFoldError(
                        "Concurrent score shards returned a duplicate case gradient."
                    )
                by_case[case_index] = gradients
        expected_indices = set(range(len(self._linearization_cache)))
        if set(by_case) != expected_indices:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Concurrent score shards did not return every case gradient."
            )
        query, key = self._empty_gradient()
        # This is intentionally identical to the former global case-order
        # accumulation.  Shard completion order cannot alter FP association.
        for case_index in range(len(self._linearization_cache)):
            query_delta, key_delta = by_case[case_index]
            query.add_(query_delta.to(device=self.device, dtype=_torch().float32))
            key.add_(key_delta.to(device=self.device, dtype=_torch().float32))
        return query, key

    def right_hand_side(self) -> Any:
        torch = _torch()
        if _uses_concurrent_static_cuda_shards(self.shards):

            def _case_gradients(
                shard: _StaticDeviceShard,
            ) -> dict[int, tuple[Any, Any]]:
                gradients = {}
                for case_index, case in shard.indexed_cases:
                    assert case.baseline_scores is not None
                    residual = case.target_scores - case.baseline_scores
                    query_delta, key_delta = _score_vjp(
                        case,
                        self.seed,
                        case.cell_weights * residual / self.residual_energy,
                    )
                    if (
                        query_delta.device != shard.device
                        or key_delta.device != shard.device
                    ):
                        raise Lfm2CachedQkScoreResidualAuthenticationError(
                            "A score RHS gradient left its static device shard."
                        )
                    gradients[case_index] = (query_delta, key_delta)
                return gradients

            shard_gradients = _execute_static_device_shards(
                self.shards,
                _case_gradients,
            )
            query, key = self._canonical_reduce_per_case_gradients(shard_gradients)
            self._record_gradient_execution(
                operation="right_hand_side",
                actual_multi_cuda_concurrent_path_executed=(
                    _supports_concurrent_static_cuda_devices(
                        tuple(shard.device for shard in self.shards)
                    )
                ),
            )
            return self.pack(query, key)

        query, key = self._empty_gradient()
        for case in self._linearized_cases():
            assert case.baseline_scores is not None
            residual = case.target_scores - case.baseline_scores
            query_delta, key_delta = _score_vjp(
                case,
                self.seed,
                case.cell_weights * residual / self.residual_energy,
            )
            query.add_(
                query_delta.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            key.add_(
                key_delta.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        self._record_gradient_execution(
            operation="right_hand_side",
            actual_multi_cuda_concurrent_path_executed=False,
        )
        return self.pack(query, key)

    def normal(self, vector: Any) -> Any:
        torch = _torch()
        self.normal_operator_calls += 1
        delta_query, delta_key = self.unpack(vector)
        if _uses_concurrent_static_cuda_shards(self.shards):

            def _case_gradients(
                shard: _StaticDeviceShard,
            ) -> dict[int, tuple[Any, Any]]:
                worker_query = delta_query.to(
                    device=shard.device,
                    dtype=torch.float32,
                )
                worker_key = delta_key.to(
                    device=shard.device,
                    dtype=torch.float32,
                )
                gradients = {}
                for case_index, case in shard.indexed_cases:
                    delta_scores = _score_jvp(
                        case,
                        self.seed,
                        worker_query,
                        worker_key,
                    )
                    query_delta, key_delta = _score_vjp(
                        case,
                        self.seed,
                        case.cell_weights * delta_scores / self.residual_energy,
                    )
                    if (
                        query_delta.device != shard.device
                        or key_delta.device != shard.device
                    ):
                        raise Lfm2CachedQkScoreResidualAuthenticationError(
                            "A score normal gradient left its static device shard."
                        )
                    gradients[case_index] = (query_delta, key_delta)
                return gradients

            shard_gradients = _execute_static_device_shards(
                self.shards,
                _case_gradients,
            )
            query, key = self._canonical_reduce_per_case_gradients(shard_gradients)
            query.add_(
                delta_query,
                alpha=self.ridge / self.query_scale_squared,
            )
            key.add_(
                delta_key,
                alpha=self.ridge / self.key_scale_squared,
            )
            self._record_gradient_execution(
                operation="normal",
                actual_multi_cuda_concurrent_path_executed=(
                    _supports_concurrent_static_cuda_devices(
                        tuple(shard.device for shard in self.shards)
                    )
                ),
            )
            return self.pack(query, key)

        query, key = self._empty_gradient()
        for case in self._linearized_cases():
            worker_device = case.query_inputs.device
            worker_query = delta_query.to(
                device=worker_device,
                dtype=torch.float32,
            )
            worker_key = delta_key.to(
                device=worker_device,
                dtype=torch.float32,
            )
            delta_scores = _score_jvp(
                case,
                self.seed,
                worker_query,
                worker_key,
            )
            query_delta, key_delta = _score_vjp(
                case,
                self.seed,
                case.cell_weights * delta_scores / self.residual_energy,
            )
            query.add_(
                query_delta.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            key.add_(
                key_delta.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        query.add_(
            delta_query,
            alpha=self.ridge / self.query_scale_squared,
        )
        key.add_(
            delta_key,
            alpha=self.ridge / self.key_scale_squared,
        )
        self._record_gradient_execution(
            operation="normal",
            actual_multi_cuda_concurrent_path_executed=False,
        )
        return self.pack(query, key)


def _dot(left: Any, right: Any) -> float:
    return float((left.to(_torch().float64) * right.to(_torch().float64)).sum().item())


def _pcg(
    operator: Any,
    right_hand_side: Any,
    policy: Lfm2QkScoreResidualFoldPolicy,
) -> tuple[Any, Lfm2QkPcgReport]:
    torch = _torch()
    solution = torch.zeros_like(right_hand_side, dtype=torch.float32)
    right_norm = math.sqrt(max(_dot(right_hand_side, right_hand_side), 0.0))
    if not math.isfinite(right_norm):
        return solution, Lfm2QkPcgReport(
            converged=False,
            iterations=0,
            normal_operator_calls=operator.normal_operator_calls,
            relative_normal_residual=math.inf,
            termination="nonfinite_right_hand_side",
        )
    if right_norm <= torch.finfo(torch.float64).tiny:
        return solution, Lfm2QkPcgReport(
            converged=True,
            iterations=0,
            normal_operator_calls=operator.normal_operator_calls,
            relative_normal_residual=0.0,
            termination="zero_score_residual_gradient",
        )
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = _dot(residual, residual)
    relative = 1.0
    for iteration in range(1, policy.maximum_pcg_iterations + 1):
        normal_direction = operator.normal(direction)
        curvature = _dot(direction, normal_direction)
        if not math.isfinite(curvature) or curvature <= 0.0:
            if math.isfinite(relative) and relative <= policy.pcg_relative_tolerance:
                return solution, Lfm2QkPcgReport(
                    converged=True,
                    iterations=iteration - 1,
                    normal_operator_calls=operator.normal_operator_calls,
                    relative_normal_residual=relative,
                    termination="converged_before_curvature_check",
                )
            return solution, Lfm2QkPcgReport(
                converged=False,
                iterations=iteration - 1,
                normal_operator_calls=operator.normal_operator_calls,
                relative_normal_residual=relative,
                termination="nonpositive_or_nonfinite_curvature",
            )
        step = residual_squared / curvature
        if not math.isfinite(step):
            return solution, Lfm2QkPcgReport(
                converged=False,
                iterations=iteration - 1,
                normal_operator_calls=operator.normal_operator_calls,
                relative_normal_residual=relative,
                termination="nonfinite_step",
            )
        solution.add_(direction, alpha=step)
        if iteration % policy.pcg_restart_iterations == 0:
            residual = right_hand_side - operator.normal(solution)
        else:
            residual.add_(normal_direction, alpha=-step)
        next_squared = max(_dot(residual, residual), 0.0)
        relative = math.sqrt(next_squared) / right_norm
        if relative == 0.0 or (
            iteration >= policy.minimum_pcg_iterations
            and relative <= policy.pcg_relative_tolerance
        ):
            return solution, Lfm2QkPcgReport(
                converged=True,
                iterations=iteration,
                normal_operator_calls=operator.normal_operator_calls,
                relative_normal_residual=relative,
                termination=(
                    "exact_normal_residual"
                    if relative == 0.0
                    else "relative_normal_residual"
                ),
            )
        if not math.isfinite(next_squared):
            return solution, Lfm2QkPcgReport(
                converged=False,
                iterations=iteration,
                normal_operator_calls=operator.normal_operator_calls,
                relative_normal_residual=math.inf,
                termination="nonfinite_residual",
            )
        if iteration % policy.pcg_restart_iterations == 0:
            direction = residual.clone()
        else:
            beta = next_squared / residual_squared
            direction.mul_(beta).add_(residual)
        residual_squared = next_squared
    return solution, Lfm2QkPcgReport(
        converged=False,
        iterations=policy.maximum_pcg_iterations,
        normal_operator_calls=operator.normal_operator_calls,
        relative_normal_residual=relative,
        termination="maximum_iterations",
    )


def _pcg_iterate_assessment(
    direction: Any,
    report: Lfm2QkPcgReport,
) -> dict[str, Any]:
    """Separate numerical convergence from safe functional evaluation.

    Reaching the fixed PCG iteration ceiling is not, by itself, evidence that
    the accumulated analytical direction is behaviorally useless.  A finite,
    nonzero ceiling iterate may therefore enter the already-fixed exact-BF16
    frontier.  Numerical breakdown terminations and zero directions remain
    fail-closed and never open that frontier.
    """

    torch = _torch()
    all_finite = bool(torch.isfinite(direction).all().item())
    l2_norm = (
        float(torch.linalg.vector_norm(direction.to(torch.float64)).item())
        if all_finite
        else math.inf
    )
    l2_norm_is_finite = math.isfinite(l2_norm)
    nonzero = (
        all_finite and l2_norm_is_finite and l2_norm > torch.finfo(torch.float64).tiny
    )
    termination_permits_functional_frontier = (
        report.converged or report.termination == "maximum_iterations"
    )
    return {
        "all_finite": all_finite,
        "nonzero": nonzero,
        "l2_norm": l2_norm if l2_norm_is_finite else None,
        "l2_norm_finite": l2_norm_is_finite,
        "termination_permits_functional_frontier": (
            termination_permits_functional_frontier
        ),
        "functional_frontier_permitted": (
            nonzero and termination_permits_functional_frontier
        ),
    }


@dataclass(slots=True)
class _RoutingMetricAccumulator:
    case_count: int = 0
    score_error: float = 0.0
    score_target_energy: float = 0.0
    probability_error: float = 0.0
    probability_target_energy: float = 0.0
    js_sum: float = 0.0
    js_worst: float = 0.0
    top1_matches: int = 0
    eligible_rows: int = 0
    leakage: float = 0.0

    @staticmethod
    def _row_routing_metrics(
        prediction_probabilities: Any,
        target_probabilities: Any,
        mask: Any,
    ) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        """Evaluate every informative row without a device sync per row.

        Rows are grouped by their exact number of valid keys before reduction.
        Consequently every FP64 row reduction has the same ordered operands and
        reduction extent as the former scalar implementation.  The results are
        scattered back into original row order before batched host transfer so
        the accumulator's Python FP64 additions also retain their prior order.
        """

        torch = _torch()
        key_width = int(prediction_probabilities.shape[-1])
        flat_prediction = prediction_probabilities.reshape(-1, key_width)
        flat_target = target_probabilities.reshape_as(flat_prediction)
        flat_mask = mask.reshape_as(flat_prediction)
        valid_counts = flat_mask.sum(dim=-1)
        count_histogram = tuple(
            int(value)
            for value in torch.bincount(
                valid_counts,
                minlength=key_width + 1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        eligible_rows = sum(count_histogram[2:])
        if eligible_rows == 0:
            return (), ()

        eligible_indices = torch.nonzero(valid_counts > 1, as_tuple=False).flatten()
        eligible_counts = valid_counts.index_select(0, eligible_indices)
        count_order = torch.argsort(eligible_counts, stable=True)
        grouped_indices = eligible_indices.index_select(0, count_order)
        grouped_js = torch.empty(
            (eligible_rows,),
            device=flat_prediction.device,
            dtype=torch.float64,
        )
        grouped_top1 = torch.empty(
            (eligible_rows,),
            device=flat_prediction.device,
            dtype=torch.bool,
        )
        tiny = torch.finfo(torch.float64).tiny
        cursor = 0
        for valid_count in range(2, key_width + 1):
            group_size = count_histogram[valid_count]
            if group_size == 0:
                continue
            row_indices = grouped_indices.narrow(0, cursor, group_size)
            group_mask = flat_mask.index_select(0, row_indices)
            prediction = (
                flat_prediction.index_select(0, row_indices)
                .masked_select(group_mask)
                .reshape(group_size, valid_count)
            )
            target = (
                flat_target.index_select(0, row_indices)
                .masked_select(group_mask)
                .reshape(group_size, valid_count)
            )
            midpoint = 0.5 * (prediction + target)
            js = (
                0.5
                * (
                    (
                        target
                        * (
                            torch.log(target.clamp_min(tiny))
                            - torch.log(midpoint.clamp_min(tiny))
                        )
                    ).sum(dim=-1)
                    + (
                        prediction
                        * (
                            torch.log(prediction.clamp_min(tiny))
                            - torch.log(midpoint.clamp_min(tiny))
                        )
                    ).sum(dim=-1)
                )
                / math.log(2.0)
            )
            grouped_js.narrow(0, cursor, group_size).copy_(js.clamp_min(0.0))
            target_winners = torch.isclose(
                target,
                target.max(dim=-1, keepdim=True).values,
                rtol=1e-10,
                atol=1e-12,
            )
            grouped_top1.narrow(0, cursor, group_size).copy_(
                target_winners.gather(
                    dim=-1,
                    index=prediction.argmax(dim=-1, keepdim=True),
                ).squeeze(-1)
            )
            cursor += group_size

        js_by_row = torch.empty_like(grouped_js)
        top1_by_row = torch.empty_like(grouped_top1)
        js_by_row.scatter_(0, count_order, grouped_js)
        top1_by_row.scatter_(0, count_order, grouped_top1)
        return (
            tuple(float(value) for value in js_by_row.detach().cpu().tolist()),
            tuple(bool(value) for value in top1_by_row.detach().cpu().tolist()),
        )

    def add(self, case: _PreparedCase, prediction_scores: Any) -> None:
        torch = _torch()
        self.case_count += 1
        prediction_probabilities = _masked_probabilities(
            prediction_scores,
            case.mask,
        )
        weights64 = case.cell_weights.to(torch.float64)
        score_delta = prediction_scores.to(torch.float64) - case.target_scores.to(
            torch.float64
        )
        self.score_error += float((weights64 * score_delta.square()).sum().item())
        self.score_target_energy += float(
            (weights64 * case.target_scores.to(torch.float64).square()).sum().item()
        )
        probability_delta = prediction_probabilities - case.target_probabilities
        self.probability_error += float(
            (weights64 * probability_delta.square()).sum().item()
        )
        self.probability_target_energy += float(
            (weights64 * case.target_probabilities.square()).sum().item()
        )
        if bool((~case.mask).any().item()):
            self.leakage = max(
                self.leakage,
                float(prediction_probabilities[~case.mask].abs().max().item()),
            )
        js_values, top1_matches = self._row_routing_metrics(
            prediction_probabilities,
            case.target_probabilities,
            case.mask,
        )
        for js_value in js_values:
            self.js_sum += js_value
            self.js_worst = max(self.js_worst, js_value)
        self.top1_matches += sum(top1_matches)
        self.eligible_rows += len(js_values)

    def finish(self) -> Lfm2QkRoutingMetrics:
        torch = _torch()
        if self.eligible_rows == 0:
            raise Lfm2CachedQkScoreResidualFoldError(
                "Q/K evidence contains no eligible causal query row."
            )
        tiny = torch.finfo(torch.float64).tiny
        return Lfm2QkRoutingMetrics(
            case_count=self.case_count,
            eligible_query_rows=self.eligible_rows,
            centered_score_relative_rmse=math.sqrt(
                self.score_error / max(self.score_target_energy, tiny)
            ),
            probability_relative_rmse=math.sqrt(
                self.probability_error / max(self.probability_target_energy, tiny)
            ),
            js_divergence_mean=self.js_sum / self.eligible_rows,
            js_divergence_worst=self.js_worst,
            top1_route_agreement_fraction=(self.top1_matches / self.eligible_rows),
            maximum_causal_probability_leakage=self.leakage,
        )


def _evaluate(
    cases: Iterable[_PreparedCase],
    seed: Lfm2AuthenticatedDenseQkSeed,
    query_weight: Any,
    key_weight: Any,
) -> Lfm2QkRoutingMetrics:
    accumulator = _RoutingMetricAccumulator()
    for case in cases:
        *_unused, prediction_scores = _exact_replay(
            case,
            seed,
            query_weight,
            key_weight,
        )
        accumulator.add(case, prediction_scores)
    return accumulator.finish()


def _evaluate_pair_streamed(
    evidence_cases: Sequence[Lfm2QkScoreResidualCase],
    seed: Lfm2AuthenticatedDenseQkSeed,
    policy: Lfm2QkScoreResidualFoldPolicy,
    compute_devices: Sequence[Any],
    first_query_weight: Any,
    first_key_weight: Any,
    second_query_weight: Any,
    second_key_weight: Any,
) -> tuple[Lfm2QkRoutingMetrics, Lfm2QkRoutingMetrics]:
    first = _RoutingMetricAccumulator()
    second = _RoutingMetricAccumulator()
    for case in _iter_prepared_evidence_cases(
        evidence_cases,
        seed,
        policy,
        compute_devices,
    ):
        *_unused, first_prediction = _exact_replay(
            case,
            seed,
            first_query_weight,
            first_key_weight,
        )
        *_unused, second_prediction = _exact_replay(
            case,
            seed,
            second_query_weight,
            second_key_weight,
        )
        first.add(case, first_prediction)
        second.add(case, second_prediction)
    return first.finish(), second.finish()


def _exact_seed_target_proof(
    evidence_cases: Sequence[Lfm2QkScoreResidualCase],
    seed: Lfm2AuthenticatedDenseQkSeed,
    policy: Lfm2QkScoreResidualFoldPolicy,
    compute_devices: Sequence[Any],
) -> bool:
    for case in _iter_prepared_evidence_cases(
        evidence_cases,
        seed,
        policy,
        compute_devices,
    ):
        *_unused, prediction_scores = _exact_replay(
            case,
            seed,
            seed.query_weight,
            seed.key_weight,
        )
        informative_cells = case.mask & case.informative_query_rows.unsqueeze(-1)
        if not _torch().equal(
            prediction_scores.masked_select(informative_cells),
            case.target_scores.masked_select(informative_cells),
        ):
            return False
    return True


def _metrics_eligible(
    candidate: Lfm2QkRoutingMetrics,
    baseline: Lfm2QkRoutingMetrics,
    *,
    tolerance: float,
    minimum_score_improvement: float,
) -> bool:
    return (
        math.isfinite(candidate.centered_score_relative_rmse)
        and candidate.centered_score_relative_rmse
        <= baseline.centered_score_relative_rmse - minimum_score_improvement
        and candidate.probability_relative_rmse
        <= baseline.probability_relative_rmse + tolerance
        and candidate.js_divergence_mean <= baseline.js_divergence_mean + tolerance
        and candidate.js_divergence_worst <= baseline.js_divergence_worst + tolerance
        and candidate.top1_route_agreement_fraction + tolerance
        >= baseline.top1_route_agreement_fraction
        and candidate.maximum_causal_probability_leakage == 0.0
    )


def _absolute_quality_eligible(
    metrics: Lfm2QkRoutingMetrics,
    policy: Lfm2QkScoreResidualFoldPolicy,
    *,
    partition: QkEvidencePartition,
) -> bool:
    if partition == "fit":
        limits = (
            policy.maximum_fit_centered_score_relative_rmse,
            policy.maximum_fit_probability_relative_rmse,
            policy.maximum_fit_js_divergence_mean,
            policy.maximum_fit_js_divergence_worst,
            policy.minimum_fit_top1_route_agreement,
        )
    else:
        limits = (
            policy.maximum_fit_check_centered_score_relative_rmse,
            policy.maximum_fit_check_probability_relative_rmse,
            policy.maximum_fit_check_js_divergence_mean,
            policy.maximum_fit_check_js_divergence_worst,
            policy.minimum_fit_check_top1_route_agreement,
        )
    (
        maximum_score,
        maximum_probability,
        maximum_js_mean,
        maximum_js_worst,
        minimum_top1,
    ) = limits
    return (
        metrics.centered_score_relative_rmse <= maximum_score
        and metrics.probability_relative_rmse <= maximum_probability
        and metrics.js_divergence_mean <= maximum_js_mean
        and metrics.js_divergence_worst <= maximum_js_worst
        and metrics.top1_route_agreement_fraction >= minimum_top1
        and metrics.maximum_causal_probability_leakage == 0.0
    )


def _trust_scale(
    delta_query: Any,
    delta_key: Any,
    current_query: Any,
    current_key: Any,
    trust_radius: float,
) -> tuple[Any, Any, float, float, float]:
    torch = _torch()
    tiny = torch.finfo(torch.float64).tiny
    query_relative = float(
        torch.linalg.vector_norm(delta_query.to(torch.float64)).item()
    ) / max(
        float(torch.linalg.vector_norm(current_query.to(torch.float64)).item()),
        tiny,
    )
    key_relative = float(
        torch.linalg.vector_norm(delta_key.to(torch.float64)).item()
    ) / max(
        float(torch.linalg.vector_norm(current_key.to(torch.float64)).item()),
        tiny,
    )
    scale = min(
        1.0,
        trust_radius / max(query_relative, tiny),
        trust_radius / max(key_relative, tiny),
    )
    return (
        delta_query * scale,
        delta_key * scale,
        query_relative,
        key_relative,
        scale,
    )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledQkScoreResidualFold:
    """Frozen exact-dtype result or authenticated synchronized-state fallback."""

    query_weight: Any
    key_weight: Any
    seed: Lfm2AuthenticatedDenseQkSeed
    accepted: bool
    changed: bool
    status: str
    fit_metrics: Lfm2QkRoutingMetrics
    fit_check_metrics: Lfm2QkRoutingMetrics | None
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if not isinstance(self.seed, Lfm2AuthenticatedDenseQkSeed):
            raise TypeError("seed must be Lfm2AuthenticatedDenseQkSeed.")
        if not isinstance(self.fit_metrics, Lfm2QkRoutingMetrics):
            raise TypeError("fit_metrics must be Lfm2QkRoutingMetrics.")
        if self.fit_check_metrics is not None and not isinstance(
            self.fit_check_metrics,
            Lfm2QkRoutingMetrics,
        ):
            raise TypeError("fit_check_metrics must be Lfm2QkRoutingMetrics or None.")
        query = _float_tensor(
            self.query_weight,
            "query_weight",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        key = _float_tensor(
            self.key_weight,
            "key_weight",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        if tuple(query.shape) != tuple(self.seed.query_weight.shape) or tuple(
            key.shape
        ) != tuple(self.seed.key_weight.shape):
            raise Lfm2CachedQkScoreResidualFoldError(
                "Compiled Q/K result differs from synchronized target shapes."
            )
        report = dict(self.report)
        seal = report.pop("report_sha256", None)
        computed = _canonical_sha256(report)
        if (
            seal != computed
            or report.get("query_weight_sha256") != _tensor_sha256(query)
            or report.get("key_weight_sha256") != _tensor_sha256(key)
            or report.get("seed_sha256") != self.seed.verify()
            or report.get("accepted") != self.accepted
            or report.get("changed") != self.changed
            or report.get("status") != self.status
            or report.get("fit_metrics") != self.fit_metrics.as_dict()
            or report.get("fit_check_metrics")
            != (
                None
                if self.fit_check_metrics is None
                else self.fit_check_metrics.as_dict()
            )
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Compiled Q/K score-residual result failed authentication."
            )
        object.__setattr__(self, "query_weight", query)
        object.__setattr__(self, "key_weight", key)
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "report_sha256": computed}),
        )

    def verify(self) -> str:
        report = dict(self.report)
        seal = report.pop("report_sha256")
        if (
            _canonical_sha256(report) != seal
            or report["query_weight_sha256"] != _tensor_sha256(self.query_weight)
            or report["key_weight_sha256"] != _tensor_sha256(self.key_weight)
            or report["seed_sha256"] != self.seed.verify()
            or report["accepted"] != self.accepted
            or report["changed"] != self.changed
            or report["status"] != self.status
            or report["fit_metrics"] != self.fit_metrics.as_dict()
            or report["fit_check_metrics"]
            != (
                None
                if self.fit_check_metrics is None
                else self.fit_check_metrics.as_dict()
            )
        ):
            raise Lfm2CachedQkScoreResidualAuthenticationError(
                "Compiled Q/K score-residual result changed after sealing."
            )
        return seal

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
            }
        )


def _result(
    *,
    query_weight: Any,
    key_weight: Any,
    seed: Lfm2AuthenticatedDenseQkSeed,
    accepted: bool,
    changed: bool,
    status: str,
    fit_metrics: Lfm2QkRoutingMetrics,
    fit_check_metrics: Lfm2QkRoutingMetrics | None,
    report_body: Mapping[str, Any],
) -> Lfm2CompiledQkScoreResidualFold:
    body = {
        **dict(report_body),
        "accepted": accepted,
        "changed": changed,
        "status": status,
        "seed_sha256": seed.seed_sha256,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
        "fit_metrics": fit_metrics.as_dict(),
        "fit_check_metrics": (
            None if fit_check_metrics is None else fit_check_metrics.as_dict()
        ),
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    return Lfm2CompiledQkScoreResidualFold(
        query_weight=query_weight,
        key_weight=key_weight,
        seed=seed,
        accepted=accepted,
        changed=changed,
        status=status,
        fit_metrics=fit_metrics,
        fit_check_metrics=fit_check_metrics,
        report=report,
    )


def compile_lfm2_cached_qk_score_residual_fold(
    *,
    seed: Lfm2AuthenticatedDenseQkSeed,
    fit_bank: Lfm2QkScoreResidualEvidenceBank,
    fit_check_bank: Lfm2QkScoreResidualEvidenceBank,
    policy: Lfm2QkScoreResidualFoldPolicy | None = None,
) -> Lfm2CompiledQkScoreResidualFold:
    """Compile one bounded dense Q/K correction from cached FIT evidence."""

    torch = _torch()
    if not isinstance(seed, Lfm2AuthenticatedDenseQkSeed):
        raise TypeError("seed must be Lfm2AuthenticatedDenseQkSeed.")
    if not isinstance(fit_bank, Lfm2QkScoreResidualEvidenceBank):
        raise TypeError("fit_bank must be Lfm2QkScoreResidualEvidenceBank.")
    if not isinstance(fit_check_bank, Lfm2QkScoreResidualEvidenceBank):
        raise TypeError("fit_check_bank must be Lfm2QkScoreResidualEvidenceBank.")
    if fit_bank.partition != "fit" or fit_check_bank.partition != "fit_check":
        raise Lfm2CachedQkScoreResidualFoldError(
            "Compiler requires exact FIT and FIT-CHECK banks."
        )
    seed.verify()
    fit_before = fit_bank.verify()
    fit_check_before = fit_check_bank.verify()
    seed_upstream_fold_sha256 = seed.upstream_fold.verify_authentication()
    seed_upstream_factor_bundle_sha256 = _factor_state_bundle_sha256(
        seed.upstream_fold.factor_sufficient_states
    )
    if (
        fit_bank.upstream_fold_report_sha256 != seed_upstream_fold_sha256
        or fit_check_bank.upstream_fold_report_sha256 != seed_upstream_fold_sha256
        or fit_bank.upstream_factor_state_bundle_sha256
        != seed_upstream_factor_bundle_sha256
        or fit_check_bank.upstream_factor_state_bundle_sha256
        != seed_upstream_factor_bundle_sha256
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Dense seed and evidence banks name different authenticated "
            "upstream CPCF folds or all-state factor bundles."
        )
    fit_informative_query_rows = _informative_query_row_count(fit_bank.cases)
    fit_check_informative_query_rows = _informative_query_row_count(
        fit_check_bank.cases
    )
    if fit_informative_query_rows == 0 or fit_check_informative_query_rows == 0:
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "FIT and FIT-CHECK must independently contain at least one causal "
            "query row with two or more valid keys."
        )
    if (
        fit_bank.source_split_sha256 != fit_check_bank.source_split_sha256
        or fit_bank.upstream_fold_report_sha256
        != fit_check_bank.upstream_fold_report_sha256
        or fit_bank.upstream_factor_state_bundle_sha256
        != fit_check_bank.upstream_factor_state_bundle_sha256
        or set(fit_bank.trajectory_sha256s) & set(fit_check_bank.trajectory_sha256s)
        or set(fit_bank.content_sha256s) & set(fit_check_bank.content_sha256s)
        or set(fit_bank.solver_row_sha256s) & set(fit_check_bank.solver_row_sha256s)
        or set(fit_bank.trajectory_row_sha256s)
        & set(fit_check_bank.trajectory_row_sha256s)
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "FIT/FIT-CHECK provenance or numerical content overlaps, or names "
            "different source splits/upstream CPCF folds."
        )
    if (
        len(
            {
                seed.source_runtime_sha256,
                fit_bank.source_runtime_sha256,
                fit_check_bank.source_runtime_sha256,
            }
        )
        != 1
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Dense seed and FIT/FIT-CHECK runtime receipts differ."
        )
    if (
        seed.source_artifact_sha256 != fit_bank.source_artifact_sha256
        or seed.source_artifact_sha256 != fit_check_bank.source_artifact_sha256
    ):
        raise Lfm2CachedQkScoreResidualAuthenticationError(
            "Dense seed and evidence banks name different authenticated source "
            "artifacts."
        )
    policy_value = policy or Lfm2QkScoreResidualFoldPolicy()
    if not isinstance(policy_value, Lfm2QkScoreResidualFoldPolicy):
        raise TypeError("policy must be Lfm2QkScoreResidualFoldPolicy.")
    if (
        len(fit_bank.cases) > policy_value.maximum_fit_cases
        or len(fit_check_bank.cases) > policy_value.maximum_fit_check_cases
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "FIT/FIT-CHECK banks exceed the bounded analytical case ceilings."
        )
    all_cases = (*fit_bank.cases, *fit_check_bank.cases)
    aggregate_score_elements = sum(
        int(case.target_centered_scores.numel()) for case in all_cases
    )
    aggregate_evidence_bytes = sum(_evidence_case_nbytes(case) for case in all_cases)
    if (
        aggregate_score_elements > policy_value.maximum_aggregate_score_elements
        or aggregate_evidence_bytes > policy_value.maximum_aggregate_evidence_bytes
    ):
        raise Lfm2CachedQkScoreResidualFoldError(
            "FIT/FIT-CHECK banks exceed aggregate analytical evidence ceilings."
        )
    solver_device, compute_devices = _resolve_execution_devices(policy_value)
    observed_execution_receipt = _build_observed_execution_receipt(
        seed=seed,
        fit_cases=fit_bank.cases,
        solver_device=solver_device,
        compute_devices=compute_devices,
    )
    execution = _execution_report(
        solver_device,
        compute_devices,
        observed_execution_receipt,
        fit_case_count=len(fit_bank.cases),
    )
    current_query = seed.query_weight.to(device=solver_device).clone()
    current_key = seed.key_weight.to(device=solver_device).clone()
    initial_fit_metrics = _evaluate(
        _iter_prepared_evidence_cases(
            fit_bank.cases,
            seed,
            policy_value,
            compute_devices,
        ),
        seed,
        current_query,
        current_key,
    )
    current_fit_metrics = initial_fit_metrics
    sweeps: list[dict[str, Any]] = []
    changed = False
    zero_bilinear_jacobian = False
    exact_quality_seed_no_op = _absolute_quality_eligible(
        initial_fit_metrics,
        policy_value,
        partition="fit",
    ) and _exact_seed_target_proof(
        fit_bank.cases,
        seed,
        policy_value,
        compute_devices,
    )
    solve_sweeps = 0 if exact_quality_seed_no_op else policy_value.maximum_sweeps
    for sweep in range(solve_sweeps):
        operator = _StreamingQkScoreResidualLinearOperator(
            seed=seed,
            evidence_cases=fit_bank.cases,
            current_query_weight=current_query.to(torch.float32),
            current_key_weight=current_key.to(torch.float32),
            ridge=policy_value.ridge,
            policy=policy_value,
            solver_device=solver_device,
            compute_devices=compute_devices,
        )
        direction, pcg_report = _pcg(
            operator,
            operator.right_hand_side(),
            policy_value,
        )
        linearization_resource_receipt = dict(
            operator.linearization_cache_resource_receipt
        )
        if linearization_resource_receipt["actual_multi_cuda_concurrent_path_executed"]:
            execution["compute_device_concurrency"] = linearization_resource_receipt[
                "compute_device_concurrency"
            ]
        pcg_iterate = _pcg_iterate_assessment(direction, pcg_report)
        sweep_report: dict[str, Any] = {
            "sweep": sweep,
            "pcg": pcg_report.as_dict(),
            "pcg_numerically_converged": pcg_report.converged,
            "pcg_iterate": pcg_iterate,
            "residual_energy": operator.residual_energy,
            "linearization_cache_resource_receipt": (linearization_resource_receipt),
        }
        zero_bilinear_jacobian = (
            zero_bilinear_jacobian
            or pcg_report.termination == "zero_score_residual_gradient"
        )
        if not pcg_iterate["functional_frontier_permitted"]:
            sweep_report["functional_frontier_evaluated"] = False
            sweeps.append(sweep_report)
            fallback_body = {
                "schema": LFM2_CACHED_QK_SCORE_RESIDUAL_SCHEMA,
                "method": "continuous_rmsnorm_rope_gqa_score_jacobian_pcg",
                "policy": policy_value.as_dict(),
                "fit_evidence_sha256": fit_before,
                "fit_check_evidence_sha256": fit_check_before,
                "source_split_sha256": fit_bank.source_split_sha256,
                "source_runtime_sha256": fit_bank.source_runtime_sha256,
                "source_artifact_sha256": fit_bank.source_artifact_sha256,
                "upstream_fold_report_sha256": (fit_bank.upstream_fold_report_sha256),
                "upstream_factor_state_bundle_sha256": (
                    fit_bank.upstream_factor_state_bundle_sha256
                ),
                "transform_receipt_sha256": fit_bank.transform_receipt_sha256,
                "execution": execution,
                "sweeps": sweeps,
                "initial_fit_metrics": initial_fit_metrics.as_dict(),
                "fit_check_evaluated": False,
                "operation_audit": {
                    "fit_cases_consumed": len(fit_bank.cases),
                    "fit_check_cases_consumed": 0,
                    "selection_cases_consumed": 0,
                    "true_validation_cases_consumed": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "training_steps": 0,
                    "model_forward_calls": 0,
                    "model_generate_calls": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                    "target_mutations_during_compile": 0,
                },
            }
            return _result(
                query_weight=seed.query_weight,
                key_weight=seed.key_weight,
                seed=seed,
                accepted=False,
                changed=False,
                status=(
                    "unidentifiable_zero_bilinear_jacobian"
                    if zero_bilinear_jacobian
                    else (
                        "pcg_nonconverged_seed_fallback"
                        if not pcg_report.converged
                        else "pcg_unusable_seed_fallback"
                    )
                ),
                fit_metrics=initial_fit_metrics,
                fit_check_metrics=None,
                report_body=fallback_body,
            )
        delta_query, delta_key = operator.unpack(direction)
        del operator
        (
            delta_query,
            delta_key,
            query_relative,
            key_relative,
            trust_scale,
        ) = _trust_scale(
            delta_query,
            delta_key,
            current_query.to(torch.float32),
            current_key.to(torch.float32),
            policy_value.trust_radius,
        )
        frontier: list[dict[str, Any]] = []
        selected: (
            tuple[
                tuple[float, float, float, float, float, str],
                float,
                Any,
                Any,
                Lfm2QkRoutingMetrics,
            ]
            | None
        ) = None
        for alpha in policy_value.alpha_frontier:
            query_candidate = (
                current_query.to(torch.float32) + alpha * delta_query
            ).to(torch.bfloat16)
            key_candidate = (current_key.to(torch.float32) + alpha * delta_key).to(
                torch.bfloat16
            )
            metrics = _evaluate(
                _iter_prepared_evidence_cases(
                    fit_bank.cases,
                    seed,
                    policy_value,
                    compute_devices,
                ),
                seed,
                query_candidate,
                key_candidate,
            )
            state_sha = _canonical_sha256(
                {
                    "query": _tensor_sha256(query_candidate),
                    "key": _tensor_sha256(key_candidate),
                }
            )
            eligible = alpha == 0.0 or _metrics_eligible(
                metrics,
                current_fit_metrics,
                tolerance=policy_value.metric_regression_tolerance,
                minimum_score_improvement=(policy_value.minimum_fit_score_improvement),
            )
            frontier.append(
                {
                    "alpha": alpha,
                    "eligible": eligible,
                    "state_sha256": state_sha,
                    "metrics": metrics.as_dict(),
                }
            )
            if eligible:
                key_value = (
                    metrics.centered_score_relative_rmse,
                    metrics.probability_relative_rmse,
                    metrics.js_divergence_worst,
                    -metrics.top1_route_agreement_fraction,
                    alpha,
                    state_sha,
                )
                candidate = (
                    key_value,
                    alpha,
                    query_candidate,
                    key_candidate,
                    metrics,
                )
                if selected is None or candidate[0] < selected[0]:
                    selected = candidate
        assert selected is not None
        (
            _selection_key,
            selected_alpha,
            selected_query,
            selected_key,
            selected_metrics,
        ) = selected
        sweep_report.update(
            {
                "unscaled_query_relative_delta": query_relative,
                "unscaled_key_relative_delta": key_relative,
                "trust_scale": trust_scale,
                "functional_frontier_evaluated": True,
                "functional_frontier_materialization_dtype": "torch.bfloat16",
                "frontier": frontier,
                "selected_alpha": selected_alpha,
                "selected_query_weight_sha256": _tensor_sha256(selected_query),
                "selected_key_weight_sha256": _tensor_sha256(selected_key),
            }
        )
        sweeps.append(sweep_report)
        same_state = torch.equal(selected_query, current_query) and torch.equal(
            selected_key, current_key
        )
        functional_candidate_eligible = selected_alpha != 0.0 and not same_state
        sweep_report["functional_candidate_eligible"] = functional_candidate_eligible
        sweep_report["functionally_selected_approximate_pcg_iterate"] = (
            functional_candidate_eligible and not pcg_report.converged
        )
        if selected_alpha == 0.0 or same_state:
            break
        current_query = selected_query
        current_key = selected_key
        current_fit_metrics = selected_metrics
        changed = True
    fit_absolute_quality = _absolute_quality_eligible(
        current_fit_metrics,
        policy_value,
        partition="fit",
    )
    fit_exact_seed_target_proof = (
        _exact_seed_target_proof(
            fit_bank.cases,
            seed,
            policy_value,
            compute_devices,
        )
        if not changed
        else False
    )
    if not changed and (not fit_absolute_quality or not fit_exact_seed_target_proof):
        no_op_body = {
            "schema": LFM2_CACHED_QK_SCORE_RESIDUAL_SCHEMA,
            "method": "continuous_rmsnorm_rope_gqa_score_jacobian_pcg",
            "policy": policy_value.as_dict(),
            "fit_evidence_sha256": fit_before,
            "fit_check_evidence_sha256": fit_check_before,
            "source_split_sha256": fit_bank.source_split_sha256,
            "source_runtime_sha256": fit_bank.source_runtime_sha256,
            "source_artifact_sha256": fit_bank.source_artifact_sha256,
            "upstream_fold_report_sha256": (fit_bank.upstream_fold_report_sha256),
            "upstream_factor_state_bundle_sha256": (
                fit_bank.upstream_factor_state_bundle_sha256
            ),
            "transform_receipt_sha256": fit_bank.transform_receipt_sha256,
            "execution": execution,
            "sweeps": sweeps,
            "initial_fit_metrics": initial_fit_metrics.as_dict(),
            "fit_absolute_quality_eligible": fit_absolute_quality,
            "fit_exact_seed_target_proof": fit_exact_seed_target_proof,
            "fit_check_evaluated": False,
            "operation_audit": {
                "fit_cases_consumed": len(fit_bank.cases),
                "fit_check_cases_consumed": 0,
                "selection_cases_consumed": 0,
                "true_validation_cases_consumed": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "training_steps": 0,
                "model_forward_calls": 0,
                "model_generate_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "target_mutations_during_compile": 0,
            },
        }
        return _result(
            query_weight=seed.query_weight,
            key_weight=seed.key_weight,
            seed=seed,
            accepted=False,
            changed=False,
            status=(
                "unidentifiable_zero_bilinear_jacobian"
                if zero_bilinear_jacobian
                else "poor_quality_seed_fallback"
            ),
            fit_metrics=initial_fit_metrics,
            fit_check_metrics=None,
            report_body=no_op_body,
        )
    if changed and not fit_absolute_quality:
        poor_fit_body = {
            "schema": LFM2_CACHED_QK_SCORE_RESIDUAL_SCHEMA,
            "method": "continuous_rmsnorm_rope_gqa_score_jacobian_pcg",
            "analytical_fold_not_training": True,
            "policy": policy_value.as_dict(),
            "fit_evidence_sha256": fit_before,
            "fit_check_evidence_sha256": fit_check_before,
            "source_split_sha256": fit_bank.source_split_sha256,
            "source_runtime_sha256": fit_bank.source_runtime_sha256,
            "source_artifact_sha256": fit_bank.source_artifact_sha256,
            "upstream_fold_report_sha256": (fit_bank.upstream_fold_report_sha256),
            "upstream_factor_state_bundle_sha256": (
                fit_bank.upstream_factor_state_bundle_sha256
            ),
            "transform_receipt_sha256": fit_bank.transform_receipt_sha256,
            "execution": execution,
            "sweeps": sweeps,
            "initial_fit_metrics": initial_fit_metrics.as_dict(),
            "proposed_fit_metrics": current_fit_metrics.as_dict(),
            "fit_absolute_quality_eligible": False,
            "fit_check_evaluated": False,
            "operation_audit": {
                "fit_cases_consumed": len(fit_bank.cases),
                "fit_check_cases_consumed": 0,
                "selection_cases_consumed": 0,
                "true_validation_cases_consumed": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "training_steps": 0,
                "model_forward_calls": 0,
                "model_generate_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "target_mutations_during_compile": 0,
            },
        }
        return _result(
            query_weight=seed.query_weight,
            key_weight=seed.key_weight,
            seed=seed,
            accepted=False,
            changed=False,
            status="poor_quality_seed_fallback",
            fit_metrics=initial_fit_metrics,
            fit_check_metrics=None,
            report_body=poor_fit_body,
        )

    # The correction is now frozen. FIT-CHECK tensors were authenticated before
    # the solve, but are numerically evaluated only here for acceptance and can
    # never trigger a refit or another frontier.
    (
        seed_fit_check_metrics,
        proposed_fit_check_metrics,
    ) = _evaluate_pair_streamed(
        fit_check_bank.cases,
        seed,
        policy_value,
        compute_devices,
        seed.query_weight.to(solver_device),
        seed.key_weight.to(solver_device),
        current_query,
        current_key,
    )
    fit_check_accepted = _metrics_eligible(
        proposed_fit_check_metrics,
        seed_fit_check_metrics,
        tolerance=0.0,
        minimum_score_improvement=(policy_value.minimum_fit_check_score_improvement),
    ) and _absolute_quality_eligible(
        proposed_fit_check_metrics,
        policy_value,
        partition="fit_check",
    )
    fit_check_exact_seed_target_proof = (
        _exact_seed_target_proof(
            fit_check_bank.cases,
            seed,
            policy_value,
            compute_devices,
        )
        if not changed
        else False
    )
    if not changed:
        fit_check_accepted = (
            fit_check_accepted
            and fit_exact_seed_target_proof
            and fit_check_exact_seed_target_proof
        )
    fit_after = fit_bank.verify()
    fit_check_after = fit_check_bank.verify()
    source_unchanged = fit_before == fit_after and fit_check_before == fit_check_after
    accepted = fit_check_accepted and source_unchanged
    functionally_selected_approximate_pcg_iterate = any(
        bool(sweep.get("functionally_selected_approximate_pcg_iterate", False))
        for sweep in sweeps
    )
    published_query = current_query.detach().cpu() if accepted else seed.query_weight
    published_key = current_key.detach().cpu() if accepted else seed.key_weight
    final_body = {
        "schema": LFM2_CACHED_QK_SCORE_RESIDUAL_SCHEMA,
        "method": "continuous_rmsnorm_rope_gqa_score_jacobian_pcg",
        "analytical_fold_not_training": True,
        "policy": policy_value.as_dict(),
        "fit_evidence_sha256": fit_before,
        "fit_check_evidence_sha256": fit_check_before,
        "source_split_sha256": fit_bank.source_split_sha256,
        "source_runtime_sha256": fit_bank.source_runtime_sha256,
        "source_artifact_sha256": fit_bank.source_artifact_sha256,
        "upstream_fold_report_sha256": fit_bank.upstream_fold_report_sha256,
        "upstream_factor_state_bundle_sha256": (
            fit_bank.upstream_factor_state_bundle_sha256
        ),
        "transform_receipt_sha256": fit_bank.transform_receipt_sha256,
        "execution": execution,
        "source_evidence_unchanged": source_unchanged,
        "sweeps": sweeps,
        "initial_fit_metrics": initial_fit_metrics.as_dict(),
        "fit_absolute_quality_eligible": fit_absolute_quality,
        "fit_exact_seed_target_proof": fit_exact_seed_target_proof,
        "seed_fit_check_metrics": seed_fit_check_metrics.as_dict(),
        "proposed_fit_check_metrics": proposed_fit_check_metrics.as_dict(),
        "fit_check_absolute_quality_eligible": _absolute_quality_eligible(
            proposed_fit_check_metrics,
            policy_value,
            partition="fit_check",
        ),
        "fit_check_exact_seed_target_proof": (fit_check_exact_seed_target_proof),
        "fit_check_evaluated": True,
        "fit_check_triggered_refit": False,
        "functionally_accepted_approximate_pcg_iterate": (
            accepted and functionally_selected_approximate_pcg_iterate
        ),
        "operation_audit": {
            "fit_cases_consumed": len(fit_bank.cases),
            "fit_check_cases_consumed": len(fit_check_bank.cases),
            "selection_cases_consumed": 0,
            "true_validation_cases_consumed": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "target_mutations_during_compile": 0,
            "published_state_cast_before_fit_check": True,
        },
    }
    return _result(
        query_weight=published_query,
        key_weight=published_key,
        seed=seed,
        accepted=accepted,
        changed=accepted and changed,
        status=(
            ("quality_gated_seed_no_op" if not changed else "fit_check_accepted")
            if accepted
            else "fit_check_rejected_seed_fallback"
        ),
        fit_metrics=(
            current_fit_metrics if accepted and changed else initial_fit_metrics
        ),
        fit_check_metrics=(
            proposed_fit_check_metrics if accepted else seed_fit_check_metrics
        ),
        report_body=final_body,
    )


__all__ = [
    "LFM2_CACHED_QK_SCORE_RESIDUAL_SCHEMA",
    "LFM2_QK_LINEARIZATION_CACHE_RESOURCE_SCHEMA",
    "LFM2_QK_OBSERVED_EXECUTION_RECEIPT_SCHEMA",
    "LFM2_QK_TARGET_SCORE_TRANSFORM_SCHEMA",
    "Lfm2AuthenticatedDenseQkSeed",
    "Lfm2CachedQkScoreResidualAuthenticationError",
    "Lfm2CachedQkScoreResidualFoldError",
    "Lfm2CompiledQkScoreResidualFold",
    "Lfm2QkPcgReport",
    "Lfm2QkRoutingMetrics",
    "Lfm2QkScoreResidualCase",
    "Lfm2QkScoreResidualEvidenceBank",
    "Lfm2QkScoreResidualFoldPolicy",
    "Lfm2QkScoreResidualLinearOperator",
    "Lfm2QkTargetScoreTransformReceipt",
    "apply_lfm2_finite_radius_rope",
    "apply_lfm2_finite_radius_rope_transpose",
    "build_lfm2_authenticated_dense_qk_seed",
    "center_lfm2_causal_scores",
    "build_lfm2_qk_score_residual_case",
    "compile_lfm2_cached_qk_score_residual_fold",
    "lfm2_continuous_rmsnorm_jvp",
    "lfm2_continuous_rmsnorm_vjp",
    "seal_lfm2_qk_score_residual_fit_bank",
    "seal_lfm2_qk_score_residual_fit_check_bank",
]
