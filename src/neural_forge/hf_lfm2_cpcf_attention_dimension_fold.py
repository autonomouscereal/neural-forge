"""Closed-form LFM2 CPCF attention head-dimension contraction.

This bounded primitive compiles authenticated parent attention routes into
ordinary dense Q/K projections with a smaller, even RoPE head dimension.  It
does not read a model or a blank target.  Callers supply the actual
child-normalized residual inputs for disjoint fit and validation evidence.

The compiler:

* verifies parent pre-RoPE Q/K, RoPE, centered causal scores, and probabilities;
* assigns every source Q and KV head to explicit positive barycentric groups;
* selects complete source RoPE frequency pairs from fit-only score/route
  contribution;
* factors each desired target-GQA score tensor at the requested head dimension;
* unrotates those target states and ridge-fits target-shaped Q/K projections;
* replays target RoPE, GQA expansion, causal masking, and softmax; and
* fails closed when spectral capacity or held-out routing gates do not pass.

Only a fixed bounded sequence of SVD and ridge solves is used.  There is no
optimizer, backward call, training, pruning, quantization, model execution, or
model mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


_SCHEMA = "neural-forge-lfm2-cpcf-attention-dimension-fold-v1"


class Lfm2CpcfAttentionDimensionFoldError(ValueError):
    """The supplied topology or authenticated evidence cannot be folded."""


class Lfm2CpcfAttentionDimensionCapacityError(Lfm2CpcfAttentionDimensionFoldError):
    """The requested head dimension fails the sealed spectral-capacity gate."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = MappingProxyType(dict(report))


class Lfm2CpcfAttentionDimensionValidationError(Lfm2CpcfAttentionDimensionFoldError):
    """The fitted projection fails held-out causal-routing gates."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = MappingProxyType(dict(report))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for CPCF attention-dimension folding."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfAttentionDimensionFoldError(f"{name} must be a positive integer.")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{name} must be finite and nonnegative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{name} must be finite and nonnegative."
        )
    return result


def _unit_fraction(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise Lfm2CpcfAttentionDimensionFoldError(f"{name} must be within [0, 1].")
    return result


def _optional_nonnegative(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, name)


def _optional_fraction(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _unit_fraction(value, name)


def _float_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{name} must be a nonempty dense real rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfAttentionDimensionFoldError(f"{name} contains NaN or infinity.")
    return result


def _mask_tensor(
    value: Any,
    name: str,
    *,
    batch: int,
    heads: int,
    query_length: int,
    key_length: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.bool
        or value.ndim != 4
        or int(value.shape[0]) != batch
        or int(value.shape[1]) not in {1, heads}
        or tuple(int(size) for size in value.shape[2:]) != (query_length, key_length)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(f"{name} must be bool [B,1|Hq,Q,K].")
    result = value.detach().to(device="cpu").contiguous()
    if int(result.shape[1]) == 1:
        result = result.expand(batch, heads, query_length, key_length).clone()
    else:
        result = result.clone()
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_rope(value: Any, cosine: Any, sine: Any) -> Any:
    return value * cosine.unsqueeze(1) + _rotate_half(value) * sine.unsqueeze(1)


def _undo_rope(value: Any, cosine: Any, sine: Any) -> Any:
    return value * cosine.unsqueeze(1) - _rotate_half(value) * sine.unsqueeze(1)


def _center_scores(scores: Any, mask: Any) -> Any:
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (scores * mask).sum(dim=-1, keepdim=True) / count
    return _torch().where(mask, scores - mean, _torch().zeros_like(scores))


def _masked_softmax(scores: Any, mask: Any) -> Any:
    torch = _torch()
    masked = scores.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked, dim=-1, dtype=torch.float64)
    return torch.where(mask, probabilities, torch.zeros_like(probabilities))


def _relative_rmse(
    prediction: Any,
    target: Any,
    mask: Any,
    weights: Any | None = None,
) -> float:
    torch = _torch()
    scale = mask.to(torch.float64)
    if weights is not None:
        scale = scale * weights
    numerator = (scale * (prediction - target).square()).sum()
    denominator = (
        (scale * target.square()).sum().clamp_min(torch.finfo(torch.float64).tiny)
    )
    return float(torch.sqrt(numerator / denominator).item())


def _retained_energy(
    prediction: Any,
    target: Any,
    mask: Any,
    weights: Any | None = None,
) -> float:
    torch = _torch()
    scale = mask.to(torch.float64)
    if weights is not None:
        scale = scale * weights
    denominator = (
        (scale * target.square()).sum().clamp_min(torch.finfo(torch.float64).tiny)
    )
    error = (scale * (prediction - target).square()).sum()
    return float((1.0 - error / denominator).item())


def _ridge_fit(inputs: Any, targets: Any, ridge: float, role: str) -> Any:
    torch = _torch()
    if (
        inputs.ndim != 2
        or targets.ndim != 2
        or int(inputs.shape[0]) != int(targets.shape[0])
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{role} ridge rows are incompatible."
        )
    gram = inputs.T @ inputs
    regularized = gram + ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64)
    try:
        coefficients = torch.linalg.solve(
            regularized,
            inputs.T @ targets,
        )
    except RuntimeError as exc:
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{role} ridge solve failed."
        ) from exc
    if not bool(torch.isfinite(coefficients).all().item()):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{role} ridge solve produced non-finite coefficients."
        )
    return coefficients.T.contiguous()


def _select_pair_coordinates(value: Any, selected: tuple[int, ...]) -> Any:
    torch = _torch()
    half = int(value.shape[-1]) // 2
    indices = torch.tensor(selected, dtype=torch.long)
    first = value.index_select(-1, indices)
    second = value.index_select(-1, indices + half)
    return torch.cat((first, second), dim=-1).contiguous()


def _select_rope_terms(value: Any, selected: tuple[int, ...]) -> Any:
    torch = _torch()
    indices = torch.tensor(selected, dtype=torch.long)
    half_values = value.index_select(-1, indices)
    return torch.cat((half_values, half_values), dim=-1).contiguous()


@dataclass(frozen=True, slots=True)
class Lfm2CpcfBarycentricHeadGroup:
    """One target head receiving positive mass from explicit source heads."""

    source_indices: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        indices = tuple(self.source_indices)
        weights = tuple(float(value) for value in self.weights)
        if (
            not indices
            or len(indices) != len(weights)
            or len(set(indices)) != len(indices)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in indices
            )
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "A barycentric head group must contain unique nonnegative "
                "source indices with finite positive weights."
            )
        total = math.fsum(weights)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Barycentric head-group weights must sum to one."
            )
        object.__setattr__(self, "source_indices", indices)
        object.__setattr__(
            self,
            "weights",
            tuple(value / total for value in weights),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_indices": list(self.source_indices),
            "weights": list(self.weights),
            "all_members_positive": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionDimensionTopology:
    source_query_heads: int
    source_key_value_heads: int
    target_query_heads: int
    target_key_value_heads: int
    source_head_dim: int
    target_head_dim: int

    def __post_init__(self) -> None:
        for name in (
            "source_query_heads",
            "source_key_value_heads",
            "target_query_heads",
            "target_key_value_heads",
            "source_head_dim",
            "target_head_dim",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        if (
            self.source_head_dim % 2
            or self.target_head_dim % 2
            or self.target_head_dim > self.source_head_dim
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Source and target RoPE dimensions must be even and target "
                "head dimension cannot exceed source head dimension."
            )
        if (
            self.source_query_heads % self.source_key_value_heads
            or self.target_query_heads % self.target_key_value_heads
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Source and target query-head counts must divide into GQA "
                "key/value groups."
            )

    @property
    def source_queries_per_key_value(self) -> int:
        return self.source_query_heads // self.source_key_value_heads

    @property
    def target_queries_per_key_value(self) -> int:
        return self.target_query_heads // self.target_key_value_heads

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_query_heads": self.source_query_heads,
            "source_key_value_heads": self.source_key_value_heads,
            "target_query_heads": self.target_query_heads,
            "target_key_value_heads": self.target_key_value_heads,
            "source_head_dim": self.source_head_dim,
            "target_head_dim": self.target_head_dim,
            "source_queries_per_key_value": (self.source_queries_per_key_value),
            "target_queries_per_key_value": (self.target_queries_per_key_value),
            "complete_source_rope_pairs": self.source_head_dim // 2,
            "complete_target_rope_pairs": self.target_head_dim // 2,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionDimensionCase:
    """One authenticated parent route aligned to actual child inputs.

    ``target_*_residual_inputs`` are the actual normalized child residual rows
    that will feed the target Q/K projections.  Parent Q/K are normalized
    pre-RoPE states; the supplied RoPE terms, centered scores, probabilities,
    and causal mask form a functionally authenticated chain.
    """

    target_query_residual_inputs: Any
    target_key_value_residual_inputs: Any
    parent_query_pre_rope: Any
    parent_key_pre_rope: Any
    parent_centered_scores: Any
    parent_probabilities: Any
    query_rope_cos: Any
    query_rope_sin: Any
    key_rope_cos: Any
    key_rope_sin: Any
    causal_mask: Any
    route_contributions: Any | None = None


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionDimensionPolicy:
    """Capacity and held-out gates for one analytical contraction."""

    minimum_score_spectral_energy_fraction: float = 0.99
    maximum_fit_centered_score_relative_rmse: float | None = 0.10
    maximum_validation_centered_score_relative_rmse: float | None = 0.20
    maximum_validation_js_divergence: float | None = 0.05
    maximum_validation_route_probability_relative_rmse: float | None = 0.20
    minimum_validation_top1_route_agreement: float | None = 0.90
    authentication_rtol: float = 2e-5
    authentication_atol: float = 2e-6
    rank_relative_tolerance: float = 1e-10
    anchor_exact_relative_rmse: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_score_spectral_energy_fraction",
            _unit_fraction(
                self.minimum_score_spectral_energy_fraction,
                "minimum_score_spectral_energy_fraction",
            ),
        )
        object.__setattr__(
            self,
            "maximum_fit_centered_score_relative_rmse",
            _optional_nonnegative(
                self.maximum_fit_centered_score_relative_rmse,
                "maximum_fit_centered_score_relative_rmse",
            ),
        )
        object.__setattr__(
            self,
            "maximum_validation_centered_score_relative_rmse",
            _optional_nonnegative(
                self.maximum_validation_centered_score_relative_rmse,
                "maximum_validation_centered_score_relative_rmse",
            ),
        )
        object.__setattr__(
            self,
            "maximum_validation_js_divergence",
            _optional_fraction(
                self.maximum_validation_js_divergence,
                "maximum_validation_js_divergence",
            ),
        )
        object.__setattr__(
            self,
            "maximum_validation_route_probability_relative_rmse",
            _optional_nonnegative(
                self.maximum_validation_route_probability_relative_rmse,
                "maximum_validation_route_probability_relative_rmse",
            ),
        )
        object.__setattr__(
            self,
            "minimum_validation_top1_route_agreement",
            _optional_fraction(
                self.minimum_validation_top1_route_agreement,
                "minimum_validation_top1_route_agreement",
            ),
        )
        for name in (
            "authentication_rtol",
            "authentication_atol",
            "rank_relative_tolerance",
            "anchor_exact_relative_rmse",
        ):
            value = _finite_nonnegative(getattr(self, name), name)
            if value == 0.0 and name == "rank_relative_tolerance":
                raise Lfm2CpcfAttentionDimensionFoldError(
                    "rank_relative_tolerance must be positive."
                )
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionDimensionLimits:
    """Hard ceilings preventing accidental unbounded analytical solves."""

    maximum_fit_cases: int = 64
    maximum_validation_cases: int = 64
    maximum_batch: int = 128
    maximum_query_tokens: int = 512
    maximum_key_tokens: int = 4096
    maximum_source_query_heads: int = 128
    maximum_source_key_value_heads: int = 128
    maximum_source_head_dim: int = 256
    maximum_target_input_width: int = 16384
    maximum_score_matrix_elements_per_group: int = 8_388_608

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionRoutingMetrics:
    case_count: int
    eligible_query_rows: int
    centered_score_relative_rmse: float
    centered_score_retained_energy_fraction: float
    probability_relative_rmse: float
    route_probability_relative_rmse: float
    probability_barycenter_relative_rmse: float
    js_divergence_mean: float
    js_divergence_worst: float
    top1_route_agreement_fraction: float
    maximum_causal_probability_leakage: float

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionDimensionReplay:
    centered_scores: Any
    probabilities: Any
    causal_mask: Any
    query_rope_cos: Any
    query_rope_sin: Any
    key_rope_cos: Any
    key_rope_sin: Any

    def __post_init__(self) -> None:
        for name in (
            "centered_scores",
            "probabilities",
            "query_rope_cos",
            "query_rope_sin",
            "key_rope_cos",
            "key_rope_sin",
        ):
            object.__setattr__(
                self,
                name,
                getattr(self, name).detach().cpu().clone(),
            )
        object.__setattr__(
            self,
            "causal_mask",
            self.causal_mask.detach().cpu().clone(),
        )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledCpcfAttentionDimensionFold:
    topology: Lfm2CpcfAttentionDimensionTopology
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...]
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...]
    selected_source_rope_frequencies: tuple[int, ...]
    query_weight: Any
    key_weight: Any
    query_input_width: int
    key_value_input_width: int
    fit_metrics: Lfm2CpcfAttentionRoutingMetrics
    validation_metrics: Lfm2CpcfAttentionRoutingMetrics
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        query = _float_tensor(self.query_weight, "query_weight", ndim=2)
        key = _float_tensor(self.key_weight, "key_weight", ndim=2)
        expected_query = (
            self.topology.target_query_heads * self.topology.target_head_dim,
            self.query_input_width,
        )
        expected_key = (
            self.topology.target_key_value_heads * self.topology.target_head_dim,
            self.key_value_input_width,
        )
        if tuple(query.shape) != expected_query or tuple(key.shape) != expected_key:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Compiled Q/K projections do not match the target topology."
            )
        object.__setattr__(self, "query_weight", query.clone())
        object.__setattr__(self, "key_weight", key.clone())
        object.__setattr__(
            self,
            "selected_source_rope_frequencies",
            tuple(self.selected_source_rope_frequencies),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
            }
        )

    def replay(
        self,
        case: Lfm2CpcfAttentionDimensionCase,
    ) -> Lfm2CpcfAttentionDimensionReplay:
        """Authenticate and replay one case without refitting or mutation."""

        normalized = _authenticate_case(
            case,
            self.topology,
            case_name="replay",
            limits=Lfm2CpcfAttentionDimensionLimits(),
            policy=Lfm2CpcfAttentionDimensionPolicy(
                minimum_score_spectral_energy_fraction=0.0,
                maximum_fit_centered_score_relative_rmse=None,
                maximum_validation_centered_score_relative_rmse=None,
                maximum_validation_js_divergence=None,
                maximum_validation_route_probability_relative_rmse=None,
                minimum_validation_top1_route_agreement=None,
            ),
        )
        _attach_target_barycenters(
            normalized,
            self.topology,
            self.query_groups,
            self.key_value_groups,
            self.selected_source_rope_frequencies,
        )
        return _replay_normalized(
            normalized,
            self.topology,
            self.selected_source_rope_frequencies,
            self.query_weight,
            self.key_weight,
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


@dataclass(slots=True)
class _NormalizedCase:
    query_inputs: Any
    key_inputs: Any
    parent_query_pre: Any
    parent_key_pre: Any
    parent_query_post: Any
    parent_key_post: Any
    parent_scores: Any
    parent_probabilities: Any
    query_cos: Any
    query_sin: Any
    key_cos: Any
    key_sin: Any
    mask: Any
    route: Any
    evidence_sha256: str
    desired_scores: Any | None = None
    desired_probabilities: Any | None = None
    probability_barycenter: Any | None = None
    desired_mask: Any | None = None
    desired_route: Any | None = None
    query_anchor: Any | None = None
    key_anchor: Any | None = None
    target_query_cos: Any | None = None
    target_query_sin: Any | None = None
    target_key_cos: Any | None = None
    target_key_sin: Any | None = None


def _case_sha256(case: Lfm2CpcfAttentionDimensionCase) -> str:
    digest = hashlib.sha256()
    for name in case.__dataclass_fields__:
        value = getattr(case, name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if value is None:
            digest.update(b"none")
        else:
            digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


def _evidence_bundle_sha256(
    fit_cases: Sequence[Lfm2CpcfAttentionDimensionCase],
    validation_cases: Sequence[Lfm2CpcfAttentionDimensionCase],
) -> str:
    digest = hashlib.sha256()
    for split, cases in (("fit", fit_cases), ("validation", validation_cases)):
        for index, case in enumerate(cases):
            digest.update(split.encode("ascii"))
            digest.update(str(index).encode("ascii"))
            digest.update(bytes.fromhex(_case_sha256(case)))
    return digest.hexdigest()


def _validate_limits(
    topology: Lfm2CpcfAttentionDimensionTopology,
    fit_count: int,
    validation_count: int,
    limits: Lfm2CpcfAttentionDimensionLimits,
) -> None:
    if (
        fit_count > limits.maximum_fit_cases
        or validation_count > limits.maximum_validation_cases
        or topology.source_query_heads > limits.maximum_source_query_heads
        or topology.source_key_value_heads > limits.maximum_source_key_value_heads
        or topology.source_head_dim > limits.maximum_source_head_dim
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "The requested fold exceeds a configured analytical ceiling."
        )


def _validate_groups(
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> None:
    if (
        len(query_groups) != topology.target_query_heads
        or len(key_value_groups) != topology.target_key_value_heads
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Barycentric group counts must equal target Q and KV head counts."
        )

    def assignment(
        groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
        source_count: int,
        role: str,
    ) -> tuple[int, ...]:
        destinations = [-1] * source_count
        for target, group in enumerate(groups):
            for source in group.source_indices:
                if source >= source_count or destinations[source] != -1:
                    raise Lfm2CpcfAttentionDimensionFoldError(
                        f"{role} barycentric groups must assign every source "
                        "head exactly once."
                    )
                destinations[source] = target
        if any(value < 0 for value in destinations):
            raise Lfm2CpcfAttentionDimensionFoldError(
                f"{role} barycentric groups dropped a source head."
            )
        return tuple(destinations)

    query_assignment = assignment(query_groups, topology.source_query_heads, "Query")
    key_assignment = assignment(
        key_value_groups, topology.source_key_value_heads, "Key/value"
    )
    for source_query, target_query in enumerate(query_assignment):
        source_kv = source_query // topology.source_queries_per_key_value
        target_kv = target_query // topology.target_queries_per_key_value
        if key_assignment[source_kv] != target_kv:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Query and key/value barycentric groups violate GQA coupling."
            )


def _authenticate_case(
    case: Lfm2CpcfAttentionDimensionCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    *,
    case_name: str,
    limits: Lfm2CpcfAttentionDimensionLimits,
    policy: Lfm2CpcfAttentionDimensionPolicy,
) -> _NormalizedCase:
    torch = _torch()
    if not isinstance(case, Lfm2CpcfAttentionDimensionCase):
        raise TypeError(f"{case_name} must be Lfm2CpcfAttentionDimensionCase.")
    query_inputs = _float_tensor(
        case.target_query_residual_inputs,
        f"{case_name}.target_query_residual_inputs",
        ndim=3,
    )
    key_inputs = _float_tensor(
        case.target_key_value_residual_inputs,
        f"{case_name}.target_key_value_residual_inputs",
        ndim=3,
    )
    batch, query_length, query_width = map(int, query_inputs.shape)
    key_batch, key_length, key_width = map(int, key_inputs.shape)
    if (
        key_batch != batch
        or query_length > key_length
        or batch > limits.maximum_batch
        or query_length > limits.maximum_query_tokens
        or key_length > limits.maximum_key_tokens
        or query_width > limits.maximum_target_input_width
        or key_width > limits.maximum_target_input_width
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} exceeds shape or causal-length ceilings."
        )
    source_dim = topology.source_head_dim
    query_pre = _float_tensor(
        case.parent_query_pre_rope,
        f"{case_name}.parent_query_pre_rope",
        ndim=4,
    )
    key_pre = _float_tensor(
        case.parent_key_pre_rope,
        f"{case_name}.parent_key_pre_rope",
        ndim=4,
    )
    expected_query = (
        batch,
        topology.source_query_heads,
        query_length,
        source_dim,
    )
    expected_key = (
        batch,
        topology.source_key_value_heads,
        key_length,
        source_dim,
    )
    if tuple(query_pre.shape) != expected_query or tuple(key_pre.shape) != expected_key:
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} parent pre-RoPE Q/K shapes differ from topology."
        )
    query_cos = _float_tensor(
        case.query_rope_cos, f"{case_name}.query_rope_cos", ndim=3
    )
    query_sin = _float_tensor(
        case.query_rope_sin, f"{case_name}.query_rope_sin", ndim=3
    )
    key_cos = _float_tensor(case.key_rope_cos, f"{case_name}.key_rope_cos", ndim=3)
    key_sin = _float_tensor(case.key_rope_sin, f"{case_name}.key_rope_sin", ndim=3)
    if (
        tuple(query_cos.shape) != (batch, query_length, source_dim)
        or tuple(query_sin.shape) != tuple(query_cos.shape)
        or tuple(key_cos.shape) != (batch, key_length, source_dim)
        or tuple(key_sin.shape) != tuple(key_cos.shape)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} RoPE shapes differ from Q/K evidence."
        )
    half = source_dim // 2
    rope_values = (query_cos, query_sin, key_cos, key_sin)
    if any(
        not torch.allclose(
            value[..., :half],
            value[..., half:],
            rtol=policy.authentication_rtol,
            atol=policy.authentication_atol,
        )
        for value in rope_values
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} RoPE terms do not preserve complete frequency pairs."
        )
    radius_errors = (
        query_cos[..., :half].square() + query_sin[..., :half].square() - 1.0,
        key_cos[..., :half].square() + key_sin[..., :half].square() - 1.0,
    )
    if any(
        float(value.abs().max().item()) > max(policy.authentication_atol * 4.0, 1e-8)
        for value in radius_errors
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} RoPE sine/cosine radius is not one."
        )
    if not (
        torch.allclose(
            query_cos,
            key_cos[:, -query_length:],
            rtol=policy.authentication_rtol,
            atol=policy.authentication_atol,
        )
        and torch.allclose(
            query_sin,
            key_sin[:, -query_length:],
            rtol=policy.authentication_rtol,
            atol=policy.authentication_atol,
        )
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} Q/K RoPE terms do not identify aligned causal positions."
        )
    mask = _mask_tensor(
        case.causal_mask,
        f"{case_name}.causal_mask",
        batch=batch,
        heads=topology.source_query_heads,
        query_length=query_length,
        key_length=key_length,
    )
    offset = key_length - query_length
    query_positions = torch.arange(query_length).reshape(1, 1, query_length, 1)
    key_positions = torch.arange(key_length).reshape(1, 1, 1, key_length)
    future = key_positions > (offset + query_positions)
    if bool((mask & future).any().item()) or bool((~mask.any(dim=-1)).any().item()):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} mask exposes a future key or has an empty route."
        )
    query_post = _apply_rope(query_pre, query_cos, query_sin)
    key_post = _apply_rope(key_pre, key_cos, key_sin)
    repeated_key = key_post.repeat_interleave(
        topology.source_queries_per_key_value,
        dim=1,
    )
    raw_scores = torch.matmul(
        query_post,
        repeated_key.transpose(-2, -1),
    ) / math.sqrt(source_dim)
    reconstructed_scores = _center_scores(raw_scores, mask)
    supplied_scores = _float_tensor(
        case.parent_centered_scores,
        f"{case_name}.parent_centered_scores",
        ndim=4,
    )
    supplied_probabilities = _float_tensor(
        case.parent_probabilities,
        f"{case_name}.parent_probabilities",
        ndim=4,
    )
    expected_score_shape = (
        batch,
        topology.source_query_heads,
        query_length,
        key_length,
    )
    if (
        tuple(supplied_scores.shape) != expected_score_shape
        or tuple(supplied_probabilities.shape) != expected_score_shape
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} score/probability evidence has the wrong shape."
        )
    reconstructed_probabilities = _masked_softmax(
        reconstructed_scores,
        mask,
    )
    if not (
        torch.allclose(
            reconstructed_scores,
            supplied_scores,
            rtol=policy.authentication_rtol,
            atol=policy.authentication_atol,
        )
        and torch.allclose(
            reconstructed_probabilities,
            supplied_probabilities,
            rtol=policy.authentication_rtol,
            atol=policy.authentication_atol,
        )
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} failed QK→centered-score→masked-softmax authentication."
        )
    if case.route_contributions is None:
        route = supplied_probabilities * (1.0 - supplied_probabilities)
    else:
        route = _float_tensor(
            case.route_contributions,
            f"{case_name}.route_contributions",
            ndim=4,
        )
        if tuple(route.shape) != expected_score_shape or bool(
            (route < 0.0).any().item()
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                f"{case_name} route contributions must be nonnegative [B,Hq,Q,K]."
            )
        route = torch.where(mask, route, torch.zeros_like(route))
    return _NormalizedCase(
        query_inputs=query_inputs,
        key_inputs=key_inputs,
        parent_query_pre=query_pre,
        parent_key_pre=key_pre,
        parent_query_post=query_post,
        parent_key_post=key_post,
        parent_scores=supplied_scores,
        parent_probabilities=supplied_probabilities,
        query_cos=query_cos,
        query_sin=query_sin,
        key_cos=key_cos,
        key_sin=key_sin,
        mask=mask,
        route=route,
        evidence_sha256=_case_sha256(case),
    )


def _source_group_weights(
    groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    count: int,
) -> Any:
    torch = _torch()
    result = torch.zeros(count, dtype=torch.float64)
    for group in groups:
        for source, weight in zip(
            group.source_indices,
            group.weights,
            strict=True,
        ):
            result[source] = weight
    return result


def _frequency_importance(
    cases: Sequence[_NormalizedCase],
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> tuple[Any, tuple[int, ...]]:
    torch = _torch()
    pair_count = topology.source_head_dim // 2
    importance = torch.zeros(pair_count, dtype=torch.float64)
    query_weight = _source_group_weights(query_groups, topology.source_query_heads)
    key_weight = _source_group_weights(
        key_value_groups, topology.source_key_value_heads
    )
    source_kv = (
        torch.arange(topology.source_query_heads, dtype=torch.long)
        // topology.source_queries_per_key_value
    )
    head_weight = query_weight * key_weight.index_select(0, source_kv)
    for case in cases:
        repeated_key = case.parent_key_post.repeat_interleave(
            topology.source_queries_per_key_value,
            dim=1,
        )
        for pair in range(pair_count):
            contribution = (
                case.parent_query_post[..., pair].unsqueeze(-1)
                * repeated_key[..., pair].unsqueeze(-2)
                + case.parent_query_post[..., pair + pair_count].unsqueeze(-1)
                * repeated_key[..., pair + pair_count].unsqueeze(-2)
            ) / math.sqrt(topology.source_head_dim)
            centered = _center_scores(contribution, case.mask)
            score_alignment = centered.square() + (centered * case.parent_scores).abs()
            importance[pair] += (
                score_alignment * case.route * head_weight.reshape(1, -1, 1, 1)
            ).sum()
    ranked = sorted(
        range(pair_count),
        key=lambda index: (-float(importance[index].item()), index),
    )
    selected = tuple(sorted(ranked[: topology.target_head_dim // 2]))
    return importance, selected


def _weighted_head_barycenter(
    value: Any,
    groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> Any:
    torch = _torch()
    output = []
    for group in groups:
        indices = torch.tensor(group.source_indices, dtype=torch.long)
        weights = torch.tensor(group.weights, dtype=torch.float64)
        selected = value.index_select(1, indices)
        shape = (1, -1, *([1] * (value.ndim - 2)))
        output.append((selected * weights.reshape(shape)).sum(dim=1))
    return torch.stack(output, dim=1)


def _group_mask_barycenter(
    mask: Any,
    groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> Any:
    torch = _torch()
    output = []
    for group in groups:
        indices = torch.tensor(group.source_indices, dtype=torch.long)
        selected = mask.index_select(1, indices)
        reference = selected[:, :1]
        if not bool(torch.equal(selected, reference.expand_as(selected))):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Source heads merged into one target head have different causal masks."
            )
        output.append(reference[:, 0])
    return torch.stack(output, dim=1)


def _attach_target_barycenters(
    case: _NormalizedCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    selected: tuple[int, ...],
) -> None:
    scale = (topology.target_head_dim / topology.source_head_dim) ** 0.25
    case.desired_scores = _weighted_head_barycenter(case.parent_scores, query_groups)
    case.probability_barycenter = _weighted_head_barycenter(
        case.parent_probabilities, query_groups
    )
    case.desired_mask = _group_mask_barycenter(case.mask, query_groups)
    case.desired_probabilities = _masked_softmax(
        case.desired_scores,
        case.desired_mask,
    )
    case.desired_route = _weighted_head_barycenter(case.route, query_groups)
    case.query_anchor = (
        _weighted_head_barycenter(
            _select_pair_coordinates(case.parent_query_post, selected),
            query_groups,
        )
        * scale
    )
    case.key_anchor = (
        _weighted_head_barycenter(
            _select_pair_coordinates(case.parent_key_post, selected),
            key_value_groups,
        )
        * scale
    )
    case.target_query_cos = _select_rope_terms(case.query_cos, selected)
    case.target_query_sin = _select_rope_terms(case.query_sin, selected)
    case.target_key_cos = _select_rope_terms(case.key_cos, selected)
    case.target_key_sin = _select_rope_terms(case.key_sin, selected)


def _case_capacity_and_factor_targets(
    case: _NormalizedCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    policy: Lfm2CpcfAttentionDimensionPolicy,
    limits: Lfm2CpcfAttentionDimensionLimits,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    torch = _torch()
    assert case.desired_scores is not None
    assert case.desired_mask is not None
    assert case.query_anchor is not None
    assert case.key_anchor is not None
    assert case.target_query_cos is not None
    assert case.target_query_sin is not None
    assert case.target_key_cos is not None
    assert case.target_key_sin is not None
    batch, _heads, query_length, key_length = map(int, case.desired_scores.shape)
    offsets = topology.target_queries_per_key_value
    target_dim = topology.target_head_dim
    query_factor = torch.zeros(
        (batch, topology.target_query_heads, query_length, target_dim),
        dtype=torch.float64,
    )
    key_factor = torch.zeros(
        (batch, topology.target_key_value_heads, key_length, target_dim),
        dtype=torch.float64,
    )
    capacity: list[dict[str, Any]] = []
    for batch_index in range(batch):
        for group in range(topology.target_key_value_heads):
            start = group * offsets
            stop = start + offsets
            matrix = case.desired_scores[batch_index, start:stop].reshape(
                offsets * query_length, key_length
            )
            if matrix.numel() > limits.maximum_score_matrix_elements_per_group:
                raise Lfm2CpcfAttentionDimensionFoldError(
                    "A target GQA score matrix exceeds the SVD ceiling."
                )
            left, singular, right_h = torch.linalg.svd(
                matrix * math.sqrt(target_dim),
                full_matrices=False,
            )
            total_energy = (
                singular.square().sum().clamp_min(torch.finfo(torch.float64).tiny)
            )
            kept = min(target_dim, int(singular.numel()))
            retained = float((singular[:kept].square().sum() / total_energy).item())
            maximum = float(singular.max().item()) if singular.numel() else 0.0
            numerical_rank = int(
                (singular > maximum * policy.rank_relative_tolerance).sum().item()
            )
            q_anchor = case.query_anchor[batch_index, start:stop].reshape(
                offsets * query_length, target_dim
            )
            k_anchor = case.key_anchor[batch_index, group]
            anchor_scores = (q_anchor @ k_anchor.T / math.sqrt(target_dim)).reshape(
                offsets, query_length, key_length
            )
            anchor_mask = case.desired_mask[batch_index, start:stop]
            anchor_scores = _center_scores(anchor_scores, anchor_mask)
            target_scores = case.desired_scores[batch_index, start:stop]
            anchor_error = _relative_rmse(
                anchor_scores,
                target_scores,
                anchor_mask,
            )
            if anchor_error <= policy.anchor_exact_relative_rmse:
                q_rows = q_anchor
                k_rows = k_anchor
                gauge = "selected_source_pair_anchor_exact"
            else:
                q_rows = torch.zeros(
                    (offsets * query_length, target_dim),
                    dtype=torch.float64,
                )
                k_rows = torch.zeros(
                    (key_length, target_dim),
                    dtype=torch.float64,
                )
                root = torch.sqrt(singular[:kept].clamp_min(0.0))
                q_rows[:, :kept] = left[:, :kept] * root
                k_rows[:, :kept] = right_h[:kept].T * root
                cross = q_rows.T @ q_anchor + k_rows.T @ k_anchor
                orient_left, _orient_singular, orient_right_h = torch.linalg.svd(
                    cross, full_matrices=False
                )
                rotation = orient_left @ orient_right_h
                q_rows = q_rows @ rotation
                k_rows = k_rows @ rotation
                gauge = "svd_balanced_selected_pair_procrustes"
            query_factor[batch_index, start:stop] = q_rows.reshape(
                offsets, query_length, target_dim
            )
            key_factor[batch_index, group] = k_rows
            capacity.append(
                {
                    "batch_index": batch_index,
                    "target_key_value_group": group,
                    "matrix_rows": offsets * query_length,
                    "matrix_columns": key_length,
                    "numerical_rank": numerical_rank,
                    "target_rank": target_dim,
                    "exact_rank_representable": numerical_rank <= target_dim,
                    "retained_spectral_energy_fraction": retained,
                    "selected_anchor_relative_rmse": anchor_error,
                    "factor_gauge": gauge,
                }
            )
    query_pre = _undo_rope(
        query_factor,
        case.target_query_cos,
        case.target_query_sin,
    )
    key_pre = _undo_rope(
        key_factor,
        case.target_key_cos,
        case.target_key_sin,
    )
    return query_pre, key_pre, capacity


def _replay_normalized(
    case: _NormalizedCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    selected: tuple[int, ...],
    query_weight: Any,
    key_weight: Any,
) -> Lfm2CpcfAttentionDimensionReplay:
    del selected  # The selected terms are already attached and authenticated.
    assert case.desired_mask is not None
    assert case.target_query_cos is not None
    assert case.target_query_sin is not None
    assert case.target_key_cos is not None
    assert case.target_key_sin is not None
    batch, query_length, _query_width = map(int, case.query_inputs.shape)
    key_length = int(case.key_inputs.shape[1])
    target_dim = topology.target_head_dim
    query_pre = (
        (case.query_inputs @ query_weight.T)
        .reshape(
            batch,
            query_length,
            topology.target_query_heads,
            target_dim,
        )
        .permute(0, 2, 1, 3)
    )
    key_pre = (
        (case.key_inputs @ key_weight.T)
        .reshape(
            batch,
            key_length,
            topology.target_key_value_heads,
            target_dim,
        )
        .permute(0, 2, 1, 3)
    )
    query_post = _apply_rope(
        query_pre,
        case.target_query_cos,
        case.target_query_sin,
    )
    key_post = _apply_rope(
        key_pre,
        case.target_key_cos,
        case.target_key_sin,
    )
    repeated_key = key_post.repeat_interleave(
        topology.target_queries_per_key_value,
        dim=1,
    )
    raw_scores = _torch().matmul(
        query_post, repeated_key.transpose(-2, -1)
    ) / math.sqrt(target_dim)
    centered = _center_scores(raw_scores, case.desired_mask)
    probabilities = _masked_softmax(centered, case.desired_mask)
    return Lfm2CpcfAttentionDimensionReplay(
        centered_scores=centered,
        probabilities=probabilities,
        causal_mask=case.desired_mask,
        query_rope_cos=case.target_query_cos,
        query_rope_sin=case.target_query_sin,
        key_rope_cos=case.target_key_cos,
        key_rope_sin=case.target_key_sin,
    )


def _routing_metrics(
    cases: Sequence[_NormalizedCase],
    replays: Sequence[Lfm2CpcfAttentionDimensionReplay],
) -> Lfm2CpcfAttentionRoutingMetrics:
    torch = _torch()
    score_predictions = []
    score_targets = []
    probability_predictions = []
    probability_targets = []
    probability_barycenters = []
    masks = []
    routes = []
    js_values: list[float] = []
    top_matches = 0
    eligible = 0
    leakage = 0.0
    for case, replay in zip(cases, replays, strict=True):
        assert case.desired_scores is not None
        assert case.desired_probabilities is not None
        assert case.probability_barycenter is not None
        assert case.desired_mask is not None
        assert case.desired_route is not None
        prediction_scores = replay.centered_scores.to(torch.float64)
        prediction_probabilities = replay.probabilities.to(torch.float64)
        score_predictions.append(prediction_scores)
        score_targets.append(case.desired_scores)
        probability_predictions.append(prediction_probabilities)
        probability_targets.append(case.desired_probabilities)
        probability_barycenters.append(case.probability_barycenter)
        masks.append(case.desired_mask)
        routes.append(case.desired_route)
        if bool((~case.desired_mask).any().item()):
            leakage = max(
                leakage,
                float(prediction_probabilities[~case.desired_mask].abs().max().item()),
            )
        flat_prediction = prediction_probabilities.reshape(
            -1, int(prediction_probabilities.shape[-1])
        )
        flat_target = case.desired_probabilities.reshape_as(flat_prediction)
        flat_mask = case.desired_mask.reshape_as(flat_prediction)
        for row in range(int(flat_prediction.shape[0])):
            valid = flat_mask[row]
            if int(valid.sum().item()) < 1:
                continue
            reference = flat_target[row, valid]
            candidate = flat_prediction[row, valid]
            midpoint = 0.5 * (reference + candidate)
            tiny = torch.finfo(torch.float64).tiny
            js = (
                0.5
                * (
                    (
                        reference
                        * (
                            torch.log(reference.clamp_min(tiny))
                            - torch.log(midpoint.clamp_min(tiny))
                        )
                    ).sum()
                    + (
                        candidate
                        * (
                            torch.log(candidate.clamp_min(tiny))
                            - torch.log(midpoint.clamp_min(tiny))
                        )
                    ).sum()
                )
                / math.log(2.0)
            )
            js_values.append(float(js.clamp_min(0.0).item()))
            reference_max = reference.max()
            reference_winners = torch.isclose(
                reference,
                reference_max,
                rtol=1e-10,
                atol=1e-12,
            )
            top_matches += int(
                bool(reference_winners[int(candidate.argmax().item())].item())
            )
            eligible += 1
    score_prediction = torch.cat([value.reshape(-1) for value in score_predictions])
    score_target = torch.cat([value.reshape(-1) for value in score_targets])
    probability_prediction = torch.cat(
        [value.reshape(-1) for value in probability_predictions]
    )
    probability_target = torch.cat([value.reshape(-1) for value in probability_targets])
    probability_barycenter = torch.cat(
        [value.reshape(-1) for value in probability_barycenters]
    )
    mask = torch.cat([value.reshape(-1) for value in masks])
    route = torch.cat([value.reshape(-1) for value in routes])
    if eligible == 0:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Routing evidence has no eligible causal query row."
        )
    return Lfm2CpcfAttentionRoutingMetrics(
        case_count=len(cases),
        eligible_query_rows=eligible,
        centered_score_relative_rmse=_relative_rmse(
            score_prediction,
            score_target,
            mask,
        ),
        centered_score_retained_energy_fraction=_retained_energy(
            score_prediction,
            score_target,
            mask,
        ),
        probability_relative_rmse=_relative_rmse(
            probability_prediction,
            probability_target,
            mask,
        ),
        route_probability_relative_rmse=_relative_rmse(
            probability_prediction,
            probability_target,
            mask,
            route,
        ),
        probability_barycenter_relative_rmse=_relative_rmse(
            probability_prediction,
            probability_barycenter,
            mask,
        ),
        js_divergence_mean=math.fsum(js_values) / len(js_values),
        js_divergence_worst=max(js_values),
        top1_route_agreement_fraction=top_matches / eligible,
        maximum_causal_probability_leakage=leakage,
    )


def _projection_fit_metrics(
    inputs: Any,
    targets: Any,
    weight: Any,
) -> dict[str, Any]:
    torch = _torch()
    prediction = inputs @ weight.T
    denominator = torch.linalg.vector_norm(targets).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    relative = float(
        (torch.linalg.vector_norm(prediction - targets) / denominator).item()
    )
    rank = int(torch.linalg.matrix_rank(inputs).item())
    return {
        "rows": int(inputs.shape[0]),
        "input_width": int(inputs.shape[1]),
        "output_width": int(targets.shape[1]),
        "design_rank": rank,
        "design_full_column_rank": rank == int(inputs.shape[1]),
        "target_state_relative_rmse": relative,
    }


def _checks(
    fit: Lfm2CpcfAttentionRoutingMetrics,
    validation: Lfm2CpcfAttentionRoutingMetrics,
    policy: Lfm2CpcfAttentionDimensionPolicy,
) -> dict[str, bool | None]:
    return {
        "maximum_fit_centered_score_relative_rmse": (
            None
            if policy.maximum_fit_centered_score_relative_rmse is None
            else fit.centered_score_relative_rmse
            <= policy.maximum_fit_centered_score_relative_rmse
        ),
        "maximum_validation_centered_score_relative_rmse": (
            None
            if policy.maximum_validation_centered_score_relative_rmse is None
            else validation.centered_score_relative_rmse
            <= policy.maximum_validation_centered_score_relative_rmse
        ),
        "maximum_validation_js_divergence": (
            None
            if policy.maximum_validation_js_divergence is None
            else validation.js_divergence_worst
            <= policy.maximum_validation_js_divergence
        ),
        "maximum_validation_route_probability_relative_rmse": (
            None
            if policy.maximum_validation_route_probability_relative_rmse is None
            else validation.route_probability_relative_rmse
            <= policy.maximum_validation_route_probability_relative_rmse
        ),
        "minimum_validation_top1_route_agreement": (
            None
            if policy.minimum_validation_top1_route_agreement is None
            else validation.top1_route_agreement_fraction
            >= policy.minimum_validation_top1_route_agreement
        ),
        "zero_causal_probability_leakage": (
            fit.maximum_causal_probability_leakage == 0.0
            and validation.maximum_causal_probability_leakage == 0.0
        ),
    }


def compile_lfm2_cpcf_attention_dimension_fold(
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    fit_cases: Sequence[Lfm2CpcfAttentionDimensionCase],
    validation_cases: Sequence[Lfm2CpcfAttentionDimensionCase],
    query_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    key_value_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    policy: Lfm2CpcfAttentionDimensionPolicy | None = None,
    limits: Lfm2CpcfAttentionDimensionLimits | None = None,
    ridge: float = 1e-8,
) -> Lfm2CompiledCpcfAttentionDimensionFold:
    """Compile a smaller target-head Q/K route from sealed causal evidence."""

    torch = _torch()
    if not isinstance(topology, Lfm2CpcfAttentionDimensionTopology):
        raise TypeError("topology must be Lfm2CpcfAttentionDimensionTopology.")
    fit_values = tuple(fit_cases)
    validation_values = tuple(validation_cases)
    if not fit_values or not validation_values:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Disjoint nonempty fit and validation cases are required."
        )
    fit_digests = {_case_sha256(value) for value in fit_values}
    validation_digests = {_case_sha256(value) for value in validation_values}
    if fit_digests & validation_digests:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Fit and validation evidence must be disjoint."
        )
    policy_value = policy or Lfm2CpcfAttentionDimensionPolicy()
    limits_value = limits or Lfm2CpcfAttentionDimensionLimits()
    if not isinstance(policy_value, Lfm2CpcfAttentionDimensionPolicy):
        raise TypeError("policy must be Lfm2CpcfAttentionDimensionPolicy.")
    if not isinstance(limits_value, Lfm2CpcfAttentionDimensionLimits):
        raise TypeError("limits must be Lfm2CpcfAttentionDimensionLimits.")
    ridge_value = _finite_nonnegative(ridge, "ridge")
    if ridge_value == 0.0:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "ridge must be positive for a bounded deterministic solve."
        )
    _validate_limits(
        topology,
        len(fit_values),
        len(validation_values),
        limits_value,
    )
    query_group_values = tuple(query_groups)
    key_group_values = tuple(key_value_groups)
    if not all(
        isinstance(value, Lfm2CpcfBarycentricHeadGroup)
        for value in (*query_group_values, *key_group_values)
    ):
        raise TypeError(
            "query_groups and key_value_groups must contain "
            "Lfm2CpcfBarycentricHeadGroup values."
        )
    _validate_groups(topology, query_group_values, key_group_values)
    source_before = _evidence_bundle_sha256(fit_values, validation_values)
    normalized_fit = tuple(
        _authenticate_case(
            case,
            topology,
            case_name=f"fit_cases[{index}]",
            limits=limits_value,
            policy=policy_value,
        )
        for index, case in enumerate(fit_values)
    )
    normalized_validation = tuple(
        _authenticate_case(
            case,
            topology,
            case_name=f"validation_cases[{index}]",
            limits=limits_value,
            policy=policy_value,
        )
        for index, case in enumerate(validation_values)
    )
    query_widths = {
        int(value.query_inputs.shape[-1])
        for value in (*normalized_fit, *normalized_validation)
    }
    key_widths = {
        int(value.key_inputs.shape[-1])
        for value in (*normalized_fit, *normalized_validation)
    }
    if len(query_widths) != 1 or len(key_widths) != 1:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "All fit and validation cases must share child Q/K input widths."
        )
    importance, selected = _frequency_importance(
        normalized_fit,
        topology,
        query_group_values,
        key_group_values,
    )
    for value in (*normalized_fit, *normalized_validation):
        _attach_target_barycenters(
            value,
            topology,
            query_group_values,
            key_group_values,
            selected,
        )
    fit_q_targets = []
    fit_k_targets = []
    fit_capacity = []
    validation_capacity = []
    for index, value in enumerate(normalized_fit):
        query_target, key_target, capacity = _case_capacity_and_factor_targets(
            value,
            topology,
            policy_value,
            limits_value,
        )
        fit_q_targets.append(query_target)
        fit_k_targets.append(key_target)
        fit_capacity.extend({"case_index": index, **record} for record in capacity)
    for index, value in enumerate(normalized_validation):
        _query_target, _key_target, capacity = _case_capacity_and_factor_targets(
            value,
            topology,
            policy_value,
            limits_value,
        )
        validation_capacity.extend(
            {"case_index": index, **record} for record in capacity
        )
    all_capacity = (*fit_capacity, *validation_capacity)
    minimum_retained = min(
        float(value["retained_spectral_energy_fraction"]) for value in all_capacity
    )
    capacity_report = {
        "target_head_dim": topology.target_head_dim,
        "target_rope_pair_count": topology.target_head_dim // 2,
        "maximum_required_numerical_rank": max(
            int(value["numerical_rank"]) for value in all_capacity
        ),
        "all_score_matrices_exact_rank_representable": all(
            bool(value["exact_rank_representable"]) for value in all_capacity
        ),
        "minimum_retained_spectral_energy_fraction": minimum_retained,
        "required_minimum_retained_spectral_energy_fraction": (
            policy_value.minimum_score_spectral_energy_fraction
        ),
        "fit_matrices": fit_capacity,
        "validation_matrices": validation_capacity,
        "accepted": (
            minimum_retained >= policy_value.minimum_score_spectral_energy_fraction
        ),
    }
    if not capacity_report["accepted"]:
        body = {
            "schema": _SCHEMA,
            "accepted": False,
            "failure": "target_head_dimension_spectral_capacity",
            "topology": topology.as_dict(),
            "capacity": capacity_report,
        }
        report = {**body, "report_sha256": _canonical_sha256(body)}
        raise Lfm2CpcfAttentionDimensionCapacityError(
            "Target head dimension fails the spectral-capacity gate.",
            report,
        )
    query_inputs = torch.cat(
        [
            value.query_inputs.reshape(-1, int(value.query_inputs.shape[-1]))
            for value in normalized_fit
        ],
        dim=0,
    )
    key_inputs = torch.cat(
        [
            value.key_inputs.reshape(-1, int(value.key_inputs.shape[-1]))
            for value in normalized_fit
        ],
        dim=0,
    )
    query_targets = torch.cat(
        [
            value.permute(0, 2, 1, 3).reshape(
                -1,
                topology.target_query_heads * topology.target_head_dim,
            )
            for value in fit_q_targets
        ],
        dim=0,
    )
    key_targets = torch.cat(
        [
            value.permute(0, 2, 1, 3).reshape(
                -1,
                topology.target_key_value_heads * topology.target_head_dim,
            )
            for value in fit_k_targets
        ],
        dim=0,
    )
    query_weight = _ridge_fit(query_inputs, query_targets, ridge_value, "query")
    key_weight = _ridge_fit(key_inputs, key_targets, ridge_value, "key")
    fit_replays = tuple(
        _replay_normalized(
            value,
            topology,
            selected,
            query_weight,
            key_weight,
        )
        for value in normalized_fit
    )
    validation_replays = tuple(
        _replay_normalized(
            value,
            topology,
            selected,
            query_weight,
            key_weight,
        )
        for value in normalized_validation
    )
    fit_metrics = _routing_metrics(normalized_fit, fit_replays)
    validation_metrics = _routing_metrics(normalized_validation, validation_replays)
    checks = _checks(fit_metrics, validation_metrics, policy_value)
    source_after = _evidence_bundle_sha256(fit_values, validation_values)
    source_unchanged = source_before == source_after
    operation_audit = {
        "source_evidence_unchanged": source_unchanged,
        "blank_target_values_read": False,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "all_source_query_heads_contribute": True,
        "all_source_key_value_heads_contribute": True,
        "source_query_heads_dropped": 0,
        "source_key_value_heads_dropped": 0,
        "dense_target_projection_state": True,
    }
    body = {
        "schema": _SCHEMA,
        "method": (
            "fit_only_route_weighted_complete_rope_pair_selection_then_"
            "target_gqa_score_svd_unrotation_and_ridge"
        ),
        "accepted": (
            all(value is not False for value in checks.values()) and source_unchanged
        ),
        "topology": topology.as_dict(),
        "query_input_width": next(iter(query_widths)),
        "key_value_input_width": next(iter(key_widths)),
        "query_groups": [value.as_dict() for value in query_group_values],
        "key_value_groups": [value.as_dict() for value in key_group_values],
        "full_source_assignment_fraction": 1.0,
        "selected_source_rope_frequencies": list(selected),
        "selected_complete_source_rope_coordinate_pairs": [
            [value, value + topology.source_head_dim // 2] for value in selected
        ],
        "fit_only_frequency_score_route_contributions": [
            float(value.item()) for value in importance
        ],
        "capacity": capacity_report,
        "query_projection_fit": _projection_fit_metrics(
            query_inputs, query_targets, query_weight
        ),
        "key_projection_fit": _projection_fit_metrics(
            key_inputs, key_targets, key_weight
        ),
        "fit_routing": fit_metrics.as_dict(),
        "heldout_routing": validation_metrics.as_dict(),
        "checks": checks,
        "policy": policy_value.as_dict(),
        "limits": limits_value.as_dict(),
        "ridge": ridge_value,
        "fit_case_count": len(normalized_fit),
        "validation_case_count": len(normalized_validation),
        "fit_evidence_sha256": _canonical_sha256(
            [value.evidence_sha256 for value in normalized_fit]
        ),
        "validation_evidence_sha256": _canonical_sha256(
            [value.evidence_sha256 for value in normalized_validation]
        ),
        "source_evidence_sha256_before": source_before,
        "source_evidence_sha256_after": source_after,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
        "analytical_solvers": {
            "score_factorization": "bounded_truncated_svd",
            "factor_orientation": (
                "selected_pair_exact_anchor_or_orthogonal_procrustes"
            ),
            "projection_fit": "positive_ridge_normal_equations",
            "frequency_selection_uses_validation": False,
        },
        "operation_audit": operation_audit,
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    if not body["accepted"]:
        raise Lfm2CpcfAttentionDimensionValidationError(
            "Compiled attention dimension fails held-out routing gates.",
            report,
        )
    return Lfm2CompiledCpcfAttentionDimensionFold(
        topology=topology,
        query_groups=query_group_values,
        key_value_groups=key_group_values,
        selected_source_rope_frequencies=selected,
        query_weight=query_weight,
        key_weight=key_weight,
        query_input_width=next(iter(query_widths)),
        key_value_input_width=next(iter(key_widths)),
        fit_metrics=fit_metrics,
        validation_metrics=validation_metrics,
        report=report,
    )


__all__ = [
    "Lfm2CompiledCpcfAttentionDimensionFold",
    "Lfm2CpcfAttentionDimensionCapacityError",
    "Lfm2CpcfAttentionDimensionCase",
    "Lfm2CpcfAttentionDimensionFoldError",
    "Lfm2CpcfAttentionDimensionLimits",
    "Lfm2CpcfAttentionDimensionPolicy",
    "Lfm2CpcfAttentionDimensionReplay",
    "Lfm2CpcfAttentionDimensionTopology",
    "Lfm2CpcfAttentionDimensionValidationError",
    "Lfm2CpcfAttentionRoutingMetrics",
    "Lfm2CpcfBarycentricHeadGroup",
    "compile_lfm2_cpcf_attention_dimension_fold",
]
