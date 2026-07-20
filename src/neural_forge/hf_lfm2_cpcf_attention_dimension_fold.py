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
from typing import Any, Literal


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


def _exact_float_tensor(
    value: Any,
    name: str,
    *,
    ndim: int | None,
) -> Any:
    """Validate a finite floating tensor without changing its runtime dtype."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or (ndim is not None and value.ndim != ndim)
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.dtype
        not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{name} must be a nonempty dense real FP16/BF16/FP32/FP64 "
            f"{'tensor' if ndim is None else f'rank-{ndim} tensor'}."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
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


def _rope_low_precision_reconstruction_error_bound(dtype: Any) -> float | None:
    torch = _torch()
    if dtype not in {torch.float16, torch.bfloat16}:
        return None
    unit_roundoff = float(torch.finfo(dtype).eps) / 2.0
    return (6.0 * unit_roundoff) / (1.0 - 6.0 * unit_roundoff)


def _rope_low_precision_geometry_error_bound(dtype: Any) -> float | None:
    torch = _torch()
    if dtype not in {torch.float16, torch.bfloat16}:
        return None
    return float(2.0 * torch.finfo(dtype).eps)


def lfm2_runtime_rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    """Replay the finite-precision LFM2 Q/K RMSNorm operation.

    LFM2 forms its RMS statistic in FP32, casts the normalized direction back
    to the projection dtype, and multiplies by the learned weight in that same
    dtype.  Keeping these two low-precision boundaries is necessary when the
    evidence and target templates are BF16.
    """

    torch = _torch()
    raw = _exact_float_tensor(value, "value", ndim=None)
    norm_weight = _exact_float_tensor(weight, "weight", ndim=1)
    if int(raw.shape[-1]) != int(norm_weight.shape[0]):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "RMSNorm weight width differs from the runtime rows."
        )
    epsilon_value = _finite_nonnegative(epsilon, "epsilon")
    if epsilon_value == 0.0:
        raise Lfm2CpcfAttentionDimensionFoldError("epsilon must be positive.")
    input_dtype = raw.dtype
    direction = raw.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon_value)
    return (
        norm_weight.to(dtype=input_dtype) * direction.to(dtype=input_dtype)
    ).contiguous()


def invert_lfm2_rope(rotated: Any, cosine: Any, sine: Any) -> Any:
    """Invert duplicated-half LFM2 RoPE at its exact finite radius."""

    torch = _torch()
    value = _exact_float_tensor(rotated, "rotated", ndim=4)
    cos = _exact_float_tensor(cosine, "cosine", ndim=3)
    sin = _exact_float_tensor(sine, "sine", ndim=3)
    if (
        value.dtype != cos.dtype
        or value.dtype != sin.dtype
        or int(value.shape[-1]) % 2
        or tuple(cos.shape)
        != (int(value.shape[0]), int(value.shape[2]), int(value.shape[3]))
        or tuple(sin.shape) != tuple(cos.shape)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Rotated rows and same-dtype RoPE terms have incompatible shapes."
        )
    half = int(cos.shape[-1]) // 2
    if not torch.equal(cos[..., :half], cos[..., half:]) or not torch.equal(
        sin[..., :half],
        sin[..., half:],
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "LFM2 RoPE terms do not have duplicated-half pair structure."
        )
    radius = cos.square() + sin.square()
    if bool((radius.to(torch.float32) <= torch.finfo(torch.float32).tiny).any().item()):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Cannot invert a zero-radius RoPE row."
        )
    numerator = value * cos.unsqueeze(1) - _rotate_half(value) * sin.unsqueeze(1)
    result = numerator / radius.unsqueeze(1)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Inverse RoPE produced a non-finite row."
        )
    return result.to(dtype=value.dtype).contiguous()


def lfm2_rmsnorm_preimage(
    normalized: Any,
    weight: Any,
    *,
    epsilon: float,
) -> Any:
    """Choose a deterministic finite-precision preimage of an RMSNorm row."""

    torch = _torch()
    values = _exact_float_tensor(normalized, "normalized", ndim=4)
    norm_weight = _exact_float_tensor(weight, "weight", ndim=1)
    width = int(values.shape[-1])
    if tuple(norm_weight.shape) != (width,):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "RMSNorm weight width differs from normalized rows."
        )
    epsilon_value = _finite_nonnegative(epsilon, "epsilon")
    if epsilon_value == 0.0:
        raise Lfm2CpcfAttentionDimensionFoldError("epsilon must be positive.")
    weight64 = norm_weight.to(torch.float64)
    if bool((weight64.abs() <= torch.finfo(torch.float64).tiny).any().item()):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Cannot recover an RMSNorm preimage through a zero learned weight."
        )
    rows = values.reshape(-1, width)
    direction = rows.to(torch.float64) / weight64.reshape(1, width)
    mean_square = direction.square().mean(dim=-1, keepdim=True)
    denominator = 1.0 - mean_square
    ideal = torch.sqrt(
        torch.as_tensor(epsilon_value, dtype=torch.float64)
        / denominator.clamp_min(torch.finfo(torch.float64).eps)
    )
    ideal = torch.where(
        denominator > torch.finfo(torch.float32).eps,
        ideal,
        torch.full_like(ideal, 128.0),
    ).clamp(min=2.0**-8, max=2.0**16)
    scales = torch.stack(
        (
            ideal,
            torch.ones_like(ideal),
            torch.full_like(ideal, 16.0),
            torch.full_like(ideal, 128.0),
            torch.full_like(ideal, 1024.0),
        ),
        dim=0,
    )
    candidates = (direction.unsqueeze(0) * scales).to(dtype=values.dtype)
    predictions = lfm2_runtime_rms_norm(
        candidates,
        norm_weight,
        epsilon_value,
    )
    errors = (
        (predictions.to(torch.float64) - rows.unsqueeze(0).to(torch.float64))
        .square()
        .mean(dim=-1)
    )
    choice = errors.argmin(dim=0)
    row_indices = torch.arange(int(rows.shape[0]))
    selected = candidates[choice, row_indices, :]
    return selected.reshape_as(values).contiguous()


def lfm2_projection_target_from_runtime(
    runtime_rows: Any,
    anchor_linear_rows: Any,
    *,
    norm_weight: Any,
    epsilon: float,
    cosine: Any,
    sine: Any,
) -> Any:
    """Map desired post-RoPE rows to raw target-projection regression rows."""

    torch = _torch()
    runtime = _float_tensor(runtime_rows, "runtime_rows", ndim=4)
    anchors = _float_tensor(anchor_linear_rows, "anchor_linear_rows", ndim=4)
    weight = _float_tensor(norm_weight, "norm_weight", ndim=1)
    cos = _float_tensor(cosine, "cosine", ndim=3)
    sin = _float_tensor(sine, "sine", ndim=3)
    if (
        tuple(runtime.shape) != tuple(anchors.shape)
        or tuple(weight.shape) != (int(runtime.shape[-1]),)
        or tuple(cos.shape)
        != (int(runtime.shape[0]), int(runtime.shape[2]), int(runtime.shape[3]))
        or tuple(sin.shape) != tuple(cos.shape)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Projection-target runtime rows, anchors, norm, and RoPE disagree."
        )
    radius = cos.square() + sin.square()
    if bool((radius <= torch.finfo(torch.float64).tiny).any().item()):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Projection target cannot invert zero-radius RoPE."
        )
    unrotated = (
        runtime * cos.unsqueeze(1) - _rotate_half(runtime) * sin.unsqueeze(1)
    ) / radius.unsqueeze(1)
    safe_weight = torch.where(
        weight.abs() > torch.finfo(torch.float64).eps,
        weight,
        torch.ones_like(weight),
    )
    direction = unrotated / safe_weight.reshape(1, 1, 1, -1)
    direction_rms = torch.sqrt(direction.square().mean(dim=-1, keepdim=True))
    direction = direction / direction_rms.clamp_min(torch.finfo(torch.float64).eps)
    epsilon_value = _finite_nonnegative(epsilon, "epsilon")
    if epsilon_value == 0.0:
        raise Lfm2CpcfAttentionDimensionFoldError("epsilon must be positive.")
    anchor_rms = torch.sqrt(anchors.square().mean(dim=-1, keepdim=True)).clamp_min(
        math.sqrt(epsilon_value)
    )
    return (direction * anchor_rms).contiguous()


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


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionCaseProvenance:
    """Authenticated location of one case inside the upstream FIT receipt.

    Both compiler partitions originate from the literal upstream ``fit``
    split.  ``partition`` only describes the deterministic local split used
    to fit a candidate and then gate that frozen candidate; it is never a
    macrocell-selection or true-validation label.
    """

    partition: Literal["fit", "fit_check"]
    trajectory_sha256: str
    source_split_sha256: str
    source_runtime_sha256: str
    case_ordinal: int
    upstream_split: Literal["fit"] = "fit"
    provenance_sha256: str = ""

    def __post_init__(self) -> None:
        if self.partition not in {"fit", "fit_check"}:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Production case partition must be 'fit' or 'fit_check'."
            )
        if self.upstream_split != "fit":
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Production Q/K evidence must retain literal upstream split 'fit'."
            )
        if (
            isinstance(self.case_ordinal, bool)
            or not isinstance(self.case_ordinal, int)
            or self.case_ordinal < 0
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "case_ordinal must be a nonnegative integer."
            )
        trajectory = _required_sha256(
            self.trajectory_sha256,
            "trajectory_sha256",
        )
        source_split = _required_sha256(
            self.source_split_sha256,
            "source_split_sha256",
        )
        source_runtime = _required_sha256(
            self.source_runtime_sha256,
            "source_runtime_sha256",
        )
        body = {
            "partition": self.partition,
            "trajectory_sha256": trajectory,
            "source_split_sha256": source_split,
            "source_runtime_sha256": source_runtime,
            "case_ordinal": self.case_ordinal,
            "upstream_split": self.upstream_split,
        }
        computed = _canonical_sha256(body)
        if self.provenance_sha256 and self.provenance_sha256 != computed:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Production case provenance does not match its seal."
            )
        object.__setattr__(self, "trajectory_sha256", trajectory)
        object.__setattr__(self, "source_split_sha256", source_split)
        object.__setattr__(self, "source_runtime_sha256", source_runtime)
        object.__setattr__(self, "provenance_sha256", computed)

    def verify(self) -> str:
        body = {
            "partition": self.partition,
            "trajectory_sha256": self.trajectory_sha256,
            "source_split_sha256": self.source_split_sha256,
            "source_runtime_sha256": self.source_runtime_sha256,
            "case_ordinal": self.case_ordinal,
            "upstream_split": self.upstream_split,
        }
        computed = _canonical_sha256(body)
        if computed != self.provenance_sha256:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Production case provenance changed after sealing."
            )
        return computed

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "trajectory_sha256": self.trajectory_sha256,
            "source_split_sha256": self.source_split_sha256,
            "source_runtime_sha256": self.source_runtime_sha256,
            "case_ordinal": self.case_ordinal,
            "upstream_split": self.upstream_split,
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionDimensionRuntime:
    """Exact source/target QK runtime contract for production compilation."""

    source_query_norm_weight: Any
    source_key_norm_weight: Any
    target_query_norm_weight: Any
    target_key_norm_weight: Any
    target_query_weight_template: Any
    target_key_weight_template: Any
    norm_epsilon: float = 1e-5
    maximum_fit_cases: int = 256
    maximum_fit_check_cases: int = 128
    maximum_aggregate_query_rows: int = 262_144
    maximum_aggregate_key_rows: int = 1_048_576
    maximum_aggregate_evidence_bytes: int = 8 * 1024**3
    solver_mode: Literal[
        "per_case_local_gauge",
        "global_gauge_radius_synchronized",
    ] = "per_case_local_gauge"

    def __post_init__(self) -> None:
        for name in (
            "source_query_norm_weight",
            "source_key_norm_weight",
            "target_query_norm_weight",
            "target_key_norm_weight",
        ):
            object.__setattr__(
                self,
                name,
                _exact_float_tensor(getattr(self, name), name, ndim=1),
            )
        for name in (
            "target_query_weight_template",
            "target_key_weight_template",
        ):
            object.__setattr__(
                self,
                name,
                _exact_float_tensor(getattr(self, name), name, ndim=2),
            )
        if (
            self.target_query_weight_template.dtype
            != self.target_key_weight_template.dtype
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Target Q/K projection templates must share one exact dtype."
            )
        epsilon = _finite_nonnegative(self.norm_epsilon, "norm_epsilon")
        if epsilon == 0.0:
            raise Lfm2CpcfAttentionDimensionFoldError("norm_epsilon must be positive.")
        object.__setattr__(self, "norm_epsilon", epsilon)
        for name in (
            "maximum_fit_cases",
            "maximum_fit_check_cases",
            "maximum_aggregate_query_rows",
            "maximum_aggregate_key_rows",
            "maximum_aggregate_evidence_bytes",
        ):
            value = _positive_int(getattr(self, name), name)
            hard_ceiling = {
                "maximum_fit_cases": 4096,
                "maximum_fit_check_cases": 4096,
                "maximum_aggregate_query_rows": 16_777_216,
                "maximum_aggregate_key_rows": 67_108_864,
                "maximum_aggregate_evidence_bytes": 64 * 1024**3,
            }[name]
            if value > hard_ceiling:
                raise Lfm2CpcfAttentionDimensionFoldError(
                    f"{name} exceeds the production analytical hard ceiling."
                )
            object.__setattr__(self, name, value)
        if self.solver_mode not in {
            "per_case_local_gauge",
            "global_gauge_radius_synchronized",
        }:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "solver_mode must be 'per_case_local_gauge' or "
                "'global_gauge_radius_synchronized'."
            )

    @property
    def target_dtype(self) -> Any:
        return self.target_query_weight_template.dtype

    @property
    def target_hidden_width(self) -> int:
        return int(self.target_query_weight_template.shape[1])

    @property
    def source_runtime_sha256(self) -> str:
        return _canonical_sha256(
            {
                "source_query_norm_weight_sha256": _tensor_sha256(
                    self.source_query_norm_weight
                ),
                "source_key_norm_weight_sha256": _tensor_sha256(
                    self.source_key_norm_weight
                ),
                "source_query_norm_weight_dtype": str(
                    self.source_query_norm_weight.dtype
                ),
                "source_key_norm_weight_dtype": str(self.source_key_norm_weight.dtype),
                "norm_epsilon": self.norm_epsilon,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_query_norm_weight_sha256": _tensor_sha256(
                self.source_query_norm_weight
            ),
            "source_key_norm_weight_sha256": _tensor_sha256(
                self.source_key_norm_weight
            ),
            "target_query_norm_weight_sha256": _tensor_sha256(
                self.target_query_norm_weight
            ),
            "target_key_norm_weight_sha256": _tensor_sha256(
                self.target_key_norm_weight
            ),
            "target_query_weight_template_sha256": _tensor_sha256(
                self.target_query_weight_template
            ),
            "target_key_weight_template_sha256": _tensor_sha256(
                self.target_key_weight_template
            ),
            "target_dtype": str(self.target_dtype),
            "target_hidden_width": self.target_hidden_width,
            "source_runtime_sha256": self.source_runtime_sha256,
            "norm_epsilon": self.norm_epsilon,
            "maximum_fit_cases": self.maximum_fit_cases,
            "maximum_fit_check_cases": self.maximum_fit_check_cases,
            "maximum_aggregate_query_rows": self.maximum_aggregate_query_rows,
            "maximum_aggregate_key_rows": self.maximum_aggregate_key_rows,
            "maximum_aggregate_evidence_bytes": self.maximum_aggregate_evidence_bytes,
            "solver_mode": self.solver_mode,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionDimensionProductionCase:
    """One exact cached Q/K route with raw-linear and row-weight provenance."""

    target_query_residual_inputs: Any
    target_key_value_residual_inputs: Any
    parent_query_linear: Any
    parent_key_linear: Any
    parent_query_post_rope: Any
    parent_key_post_rope: Any
    parent_centered_scores: Any
    parent_probabilities: Any
    query_rope_cos: Any
    query_rope_sin: Any
    key_rope_cos: Any
    key_rope_sin: Any
    causal_mask: Any
    query_row_sqrt_weights: Any
    key_value_row_sqrt_weights: Any
    provenance: Lfm2CpcfAttentionCaseProvenance
    route_contributions: Any | None = None
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        for name, ndim in (
            ("target_query_residual_inputs", 3),
            ("target_key_value_residual_inputs", 3),
            ("parent_query_linear", 4),
            ("parent_key_linear", 4),
            ("parent_query_post_rope", 4),
            ("parent_key_post_rope", 4),
            ("parent_centered_scores", 4),
            ("parent_probabilities", 4),
            ("query_rope_cos", 3),
            ("query_rope_sin", 3),
            ("key_rope_cos", 3),
            ("key_rope_sin", 3),
            ("query_row_sqrt_weights", 2),
            ("key_value_row_sqrt_weights", 2),
        ):
            object.__setattr__(
                self,
                name,
                _exact_float_tensor(getattr(self, name), name, ndim=ndim),
            )
        torch = _torch()
        if (
            not isinstance(self.causal_mask, torch.Tensor)
            or self.causal_mask.layout != torch.strided
            or self.causal_mask.device.type == "meta"
            or self.causal_mask.dtype != torch.bool
            or self.causal_mask.ndim != 4
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "causal_mask must be a dense bool rank-4 tensor."
            )
        object.__setattr__(
            self,
            "causal_mask",
            self.causal_mask.detach().to(device="cpu").contiguous().clone(),
        )
        if self.route_contributions is not None:
            object.__setattr__(
                self,
                "route_contributions",
                _exact_float_tensor(
                    self.route_contributions,
                    "route_contributions",
                    ndim=4,
                ),
            )
        if not isinstance(self.provenance, Lfm2CpcfAttentionCaseProvenance):
            raise TypeError("provenance must be Lfm2CpcfAttentionCaseProvenance.")
        self.provenance.verify()
        computed = _production_case_sha256(self, include_seal=False)
        if self.evidence_sha256 and self.evidence_sha256 != computed:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Production attention evidence does not match its seal."
            )
        object.__setattr__(self, "evidence_sha256", computed)

    def verify(self) -> str:
        self.provenance.verify()
        computed = _production_case_sha256(self, include_seal=False)
        if computed != self.evidence_sha256:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Production attention evidence changed after sealing."
            )
        return computed


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionFactorSufficientState:
    """Sealed per-case state needed by a future globally coupled Q/K gauge."""

    partition: Literal["fit", "fit_check"]
    case_ordinal: int
    evidence_sha256: str
    local_query_post_rope_factor: Any
    local_key_post_rope_factor: Any
    query_post_rope_anchor: Any
    key_post_rope_anchor: Any
    query_raw_linear_anchor: Any
    key_raw_linear_anchor: Any
    target_query_inputs: Any
    target_key_inputs: Any
    query_row_sqrt_weights: Any
    key_row_sqrt_weights: Any
    target_query_rope_cos: Any
    target_query_rope_sin: Any
    target_key_rope_cos: Any
    target_key_rope_sin: Any

    def __post_init__(self) -> None:
        if self.partition not in {"fit", "fit_check"}:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Factor sufficient state has an invalid partition."
            )
        if (
            isinstance(self.case_ordinal, bool)
            or not isinstance(self.case_ordinal, int)
            or self.case_ordinal < 0
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Factor sufficient-state ordinal must be nonnegative."
            )
        object.__setattr__(
            self,
            "evidence_sha256",
            _required_sha256(self.evidence_sha256, "evidence_sha256"),
        )
        for name, ndim in (
            ("local_query_post_rope_factor", 4),
            ("local_key_post_rope_factor", 4),
            ("query_post_rope_anchor", 4),
            ("key_post_rope_anchor", 4),
            ("query_raw_linear_anchor", 4),
            ("key_raw_linear_anchor", 4),
            ("target_query_inputs", 3),
            ("target_key_inputs", 3),
            ("query_row_sqrt_weights", 2),
            ("key_row_sqrt_weights", 2),
            ("target_query_rope_cos", 3),
            ("target_query_rope_sin", 3),
            ("target_key_rope_cos", 3),
            ("target_key_rope_sin", 3),
        ):
            object.__setattr__(
                self,
                name,
                _exact_float_tensor(getattr(self, name), name, ndim=ndim),
            )

    def as_digest_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "case_ordinal": self.case_ordinal,
            "evidence_sha256": self.evidence_sha256,
            **{
                name: _tensor_sha256(getattr(self, name))
                for name in self.__dataclass_fields__
                if name
                not in {
                    "partition",
                    "case_ordinal",
                    "evidence_sha256",
                }
            },
        }


def _production_case_sha256(
    case: Lfm2CpcfAttentionDimensionProductionCase,
    *,
    include_seal: bool,
) -> str:
    digest = hashlib.sha256()
    for name in case.__dataclass_fields__:
        if name == "evidence_sha256" and not include_seal:
            continue
        value = getattr(case, name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if value is None:
            digest.update(b"none")
        elif isinstance(value, Lfm2CpcfAttentionCaseProvenance):
            digest.update(bytes.fromhex(value.verify()))
        elif name == "evidence_sha256":
            digest.update(value.encode("ascii"))
        else:
            digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


def build_lfm2_cpcf_attention_dimension_production_case(
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    runtime: Lfm2CpcfAttentionDimensionRuntime,
    target_query_residual_inputs: Any,
    target_key_value_residual_inputs: Any,
    parent_query_linear: Any,
    parent_key_linear: Any,
    parent_query_post_rope: Any,
    parent_key_post_rope: Any,
    query_rope_cos: Any,
    query_rope_sin: Any,
    key_rope_cos: Any,
    key_rope_sin: Any,
    causal_mask: Any,
    provenance: Lfm2CpcfAttentionCaseProvenance,
    query_row_sqrt_weights: Any | None = None,
    key_value_row_sqrt_weights: Any | None = None,
    route_contributions: Any | None = None,
) -> Lfm2CpcfAttentionDimensionProductionCase:
    """Build and seal one real projection→RMSNorm→RoPE production case."""

    torch = _torch()
    if not isinstance(topology, Lfm2CpcfAttentionDimensionTopology):
        raise TypeError("topology must be Lfm2CpcfAttentionDimensionTopology.")
    if not isinstance(runtime, Lfm2CpcfAttentionDimensionRuntime):
        raise TypeError("runtime must be Lfm2CpcfAttentionDimensionRuntime.")
    if not isinstance(provenance, Lfm2CpcfAttentionCaseProvenance):
        raise TypeError("provenance must be Lfm2CpcfAttentionCaseProvenance.")
    provenance.verify()
    if provenance.source_runtime_sha256 != runtime.source_runtime_sha256:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production case provenance does not bind the supplied source "
            "RMSNorm runtime."
        )
    query_linear = _exact_float_tensor(
        parent_query_linear,
        "parent_query_linear",
        ndim=4,
    )
    key_linear = _exact_float_tensor(
        parent_key_linear,
        "parent_key_linear",
        ndim=4,
    )
    query_post = _exact_float_tensor(
        parent_query_post_rope,
        "parent_query_post_rope",
        ndim=4,
    )
    key_post = _exact_float_tensor(
        parent_key_post_rope,
        "parent_key_post_rope",
        ndim=4,
    )
    if (
        query_linear.dtype != key_linear.dtype
        or query_linear.dtype != query_post.dtype
        or query_linear.dtype != key_post.dtype
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Raw, normalized, and post-RoPE Q/K evidence must share one dtype."
        )
    batch, _query_heads, query_length, _head_dim = map(int, query_post.shape)
    key_length = int(key_post.shape[2])
    if (
        not isinstance(causal_mask, torch.Tensor)
        or causal_mask.dtype != torch.bool
        or causal_mask.device.type == "meta"
        or causal_mask.layout != torch.strided
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "causal_mask must be a dense bool tensor."
        )
    mask = causal_mask.detach().to(device="cpu").contiguous()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if (
        mask.ndim != 4
        or int(mask.shape[0]) != batch
        or int(mask.shape[1]) not in {1, topology.source_query_heads}
        or tuple(int(value) for value in mask.shape[2:]) != (query_length, key_length)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "causal_mask must be [B,1|source_query_heads,Q,K]."
        )
    expanded_mask = (
        mask.expand(
            batch,
            topology.source_query_heads,
            query_length,
            key_length,
        )
        .clone()
        .contiguous()
    )
    repeated_key = key_post.repeat_interleave(
        topology.source_queries_per_key_value,
        dim=1,
    )
    raw_scores = torch.matmul(query_post, repeated_key.transpose(-2, -1)) / math.sqrt(
        topology.source_head_dim
    )
    centered_scores = _center_scores(raw_scores, expanded_mask)
    probabilities = _masked_softmax(
        centered_scores.to(torch.float64),
        expanded_mask,
    )
    query_weights = (
        torch.ones(
            (batch, query_length),
            dtype=query_linear.dtype,
        )
        if query_row_sqrt_weights is None
        else _exact_float_tensor(
            query_row_sqrt_weights,
            "query_row_sqrt_weights",
            ndim=2,
        )
    )
    key_weights = (
        torch.ones(
            (batch, key_length),
            dtype=key_linear.dtype,
        )
        if key_value_row_sqrt_weights is None
        else _exact_float_tensor(
            key_value_row_sqrt_weights,
            "key_value_row_sqrt_weights",
            ndim=2,
        )
    )
    return Lfm2CpcfAttentionDimensionProductionCase(
        target_query_residual_inputs=target_query_residual_inputs,
        target_key_value_residual_inputs=target_key_value_residual_inputs,
        parent_query_linear=query_linear,
        parent_key_linear=key_linear,
        parent_query_post_rope=query_post,
        parent_key_post_rope=key_post,
        parent_centered_scores=centered_scores,
        parent_probabilities=probabilities,
        query_rope_cos=query_rope_cos,
        query_rope_sin=query_rope_sin,
        key_rope_cos=key_rope_cos,
        key_rope_sin=key_rope_sin,
        causal_mask=mask.clone(),
        query_row_sqrt_weights=query_weights,
        key_value_row_sqrt_weights=key_weights,
        provenance=provenance,
        route_contributions=route_contributions,
    )


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
    runtime: Lfm2CpcfAttentionDimensionRuntime | None = None
    factor_sufficient_states: tuple[
        Lfm2CpcfAttentionFactorSufficientState,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if self.runtime is None:
            query = _float_tensor(self.query_weight, "query_weight", ndim=2)
            key = _float_tensor(self.key_weight, "key_weight", ndim=2)
        else:
            if not isinstance(self.runtime, Lfm2CpcfAttentionDimensionRuntime):
                raise TypeError(
                    "runtime must be Lfm2CpcfAttentionDimensionRuntime or None."
                )
            query = _exact_float_tensor(self.query_weight, "query_weight", ndim=2)
            key = _exact_float_tensor(self.key_weight, "key_weight", ndim=2)
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
        states = tuple(self.factor_sufficient_states)
        if not all(
            isinstance(value, Lfm2CpcfAttentionFactorSufficientState)
            for value in states
        ):
            raise TypeError(
                "factor_sufficient_states must contain public sufficient-state records."
            )
        object.__setattr__(self, "factor_sufficient_states", states)
        if self.runtime is not None:
            if (
                query.dtype != self.runtime.target_dtype
                or key.dtype != self.runtime.target_dtype
            ):
                raise Lfm2CpcfAttentionDimensionFoldError(
                    "Published Q/K state does not preserve the exact target "
                    "template dtype."
                )
            self.verify_authentication()

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().clone(),
            }
        )

    def replay(
        self,
        case: (
            Lfm2CpcfAttentionDimensionCase | Lfm2CpcfAttentionDimensionProductionCase
        ),
    ) -> Lfm2CpcfAttentionDimensionReplay:
        """Authenticate and replay one case without refitting or mutation."""

        if self.runtime is not None:
            if not isinstance(case, Lfm2CpcfAttentionDimensionProductionCase):
                raise TypeError(
                    "Production replay requires a production attention case."
                )
            normalized = _authenticate_production_case(
                case,
                self.topology,
                self.runtime,
                case_name="replay",
                limits=_production_limits(self.runtime),
                policy=Lfm2CpcfAttentionDimensionPolicy(
                    minimum_score_spectral_energy_fraction=0.0,
                    maximum_fit_centered_score_relative_rmse=None,
                    maximum_validation_centered_score_relative_rmse=None,
                    maximum_validation_js_divergence=None,
                    maximum_validation_route_probability_relative_rmse=None,
                    minimum_validation_top1_route_agreement=None,
                ),
                expected_partition=None,
            )
            _attach_target_barycenters_production(
                normalized,
                self.topology,
                self.query_groups,
                self.key_value_groups,
                self.selected_source_rope_frequencies,
            )
            return _replay_normalized_production(
                normalized,
                self.topology,
                self.query_weight,
                self.key_weight,
                self.runtime,
            )
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

    @property
    def fit_check_metrics(self) -> Lfm2CpcfAttentionRoutingMetrics:
        """Production name for the legacy-compatible second metric field."""

        return self.validation_metrics

    def verify_authentication(self) -> str:
        """Verify report seal, exact dense state, and no-training semantics."""

        if self.runtime is None:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Semantic production authentication requires a runtime contract."
            )
        report = json.loads(
            json.dumps(
                dict(self.report),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        sealed = _required_sha256(
            report.pop("report_sha256", None),
            "report_sha256",
        )
        if _canonical_sha256(report) != sealed:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Compiled production report seal is invalid."
            )
        if (
            report.get("production_runtime_contract") is not True
            or report.get("accepted") is not True
            or report.get("topology") != self.topology.as_dict()
            or report.get("query_groups")
            != [value.as_dict() for value in self.query_groups]
            or report.get("key_value_groups")
            != [value.as_dict() for value in self.key_value_groups]
            or report.get("selected_source_rope_frequencies")
            != list(self.selected_source_rope_frequencies)
            or report.get("query_input_width") != self.query_input_width
            or report.get("key_value_input_width") != self.key_value_input_width
            or report.get("query_weight_sha256") != _tensor_sha256(self.query_weight)
            or report.get("key_weight_sha256") != _tensor_sha256(self.key_weight)
            or report.get("runtime") != self.runtime.as_dict()
            or report.get("fit_routing") != self.fit_metrics.as_dict()
            or report.get("fit_check_routing") != self.fit_check_metrics.as_dict()
            or report.get("factor_sufficient_state_count")
            != len(self.factor_sufficient_states)
            or report.get("factor_sufficient_state_bundle_sha256")
            != _canonical_sha256(
                [value.as_digest_dict() for value in self.factor_sufficient_states]
            )
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Compiled production report does not bind its exact typed state."
            )
        operation = report.get("operation_audit")
        if not isinstance(operation, dict):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Compiled production report lacks an operation audit."
            )
        zero_fields = (
            "model_forward_calls",
            "model_generate_calls",
            "backward_calls",
            "optimizer_steps",
            "training_steps",
            "pruned_parameters",
            "quantized_parameters",
            "target_parameter_mutations_during_fit_check",
            "selection_cases_consumed",
            "true_validation_cases_consumed",
        )
        if (
            not bool(operation.get("source_evidence_unchanged"))
            or not bool(operation.get("dense_target_projection_state"))
            or not bool(operation.get("published_state_cast_before_fit_check"))
            or any(int(operation.get(name, -1)) != 0 for name in zero_fields)
        ):
            raise Lfm2CpcfAttentionDimensionFoldError(
                "Compiled production report violates the analytical no-training "
                "or frozen-fit-check boundary."
            )
        return sealed


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
    query_linear_anchor: Any | None = None
    key_linear_anchor: Any | None = None
    target_query_cos: Any | None = None
    target_query_sin: Any | None = None
    target_key_cos: Any | None = None
    target_key_sin: Any | None = None
    parent_query_linear: Any | None = None
    parent_key_linear: Any | None = None
    query_sqrt_weights: Any | None = None
    key_sqrt_weights: Any | None = None
    provenance: Lfm2CpcfAttentionCaseProvenance | None = None
    production: bool = False


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


def _production_limits(
    runtime: Lfm2CpcfAttentionDimensionRuntime,
) -> Lfm2CpcfAttentionDimensionLimits:
    """Materialize explicit bounded ceilings for the production branch."""

    return Lfm2CpcfAttentionDimensionLimits(
        maximum_fit_cases=runtime.maximum_fit_cases,
        maximum_validation_cases=runtime.maximum_fit_check_cases,
    )


def _maximum_rope_pair_relative_error(
    normalized: Any,
    rotated: Any,
    reconstruction: Any,
) -> float:
    torch = _torch()
    half = int(normalized.shape[-1]) // 2
    error = reconstruction.to(torch.float64) - rotated.to(torch.float64)
    normalized64 = normalized.to(torch.float64)
    rotated64 = rotated.to(torch.float64)
    error_pairs = torch.stack(
        (error[..., :half], error[..., half:]),
        dim=-1,
    )
    normalized_pairs = torch.stack(
        (normalized64[..., :half], normalized64[..., half:]),
        dim=-1,
    )
    rotated_pairs = torch.stack(
        (rotated64[..., :half], rotated64[..., half:]),
        dim=-1,
    )
    scale = torch.maximum(
        torch.linalg.vector_norm(normalized_pairs, dim=-1),
        torch.linalg.vector_norm(rotated_pairs, dim=-1),
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return float((torch.linalg.vector_norm(error_pairs, dim=-1) / scale).max().item())


def _authenticate_production_case(
    case: Lfm2CpcfAttentionDimensionProductionCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    runtime: Lfm2CpcfAttentionDimensionRuntime,
    *,
    case_name: str,
    limits: Lfm2CpcfAttentionDimensionLimits,
    policy: Lfm2CpcfAttentionDimensionPolicy,
    expected_partition: Literal["fit", "fit_check"] | None,
) -> _NormalizedCase:
    torch = _torch()
    if not isinstance(case, Lfm2CpcfAttentionDimensionProductionCase):
        raise TypeError(
            f"{case_name} must be Lfm2CpcfAttentionDimensionProductionCase."
        )
    case.verify()
    if (
        expected_partition is not None
        and case.provenance.partition != expected_partition
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} provenance must identify partition '{expected_partition}'."
        )
    if case.provenance.source_runtime_sha256 != runtime.source_runtime_sha256:
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} provenance does not bind the supplied source RMSNorm runtime."
        )
    query_inputs_exact = _exact_float_tensor(
        case.target_query_residual_inputs,
        f"{case_name}.target_query_residual_inputs",
        ndim=3,
    )
    key_inputs_exact = _exact_float_tensor(
        case.target_key_value_residual_inputs,
        f"{case_name}.target_key_value_residual_inputs",
        ndim=3,
    )
    batch, query_length, query_width = map(int, query_inputs_exact.shape)
    key_batch, key_length, key_width = map(int, key_inputs_exact.shape)
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
    query_linear = _exact_float_tensor(
        case.parent_query_linear,
        f"{case_name}.parent_query_linear",
        ndim=4,
    )
    key_linear = _exact_float_tensor(
        case.parent_key_linear,
        f"{case_name}.parent_key_linear",
        ndim=4,
    )
    query_post = _exact_float_tensor(
        case.parent_query_post_rope,
        f"{case_name}.parent_query_post_rope",
        ndim=4,
    )
    key_post = _exact_float_tensor(
        case.parent_key_post_rope,
        f"{case_name}.parent_key_post_rope",
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
    if (
        tuple(query_linear.shape) != expected_query
        or tuple(query_post.shape) != expected_query
        or tuple(key_linear.shape) != expected_key
        or tuple(key_post.shape) != expected_key
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} raw/pre/post Q/K shapes differ from topology."
        )
    evidence_dtypes = {
        query_linear.dtype,
        key_linear.dtype,
        query_post.dtype,
        key_post.dtype,
        case.query_rope_cos.dtype,
        case.query_rope_sin.dtype,
        case.key_rope_cos.dtype,
        case.key_rope_sin.dtype,
    }
    if len(evidence_dtypes) != 1:
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} raw/norm/rotary evidence must preserve one exact dtype."
        )
    evidence_dtype = next(iter(evidence_dtypes))
    if tuple(runtime.source_query_norm_weight.shape) != (source_dim,) or tuple(
        runtime.source_key_norm_weight.shape
    ) != (source_dim,):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Source Q/K RMSNorm weights differ from the source head dimension."
        )
    query_pre = lfm2_runtime_rms_norm(
        query_linear,
        runtime.source_query_norm_weight,
        runtime.norm_epsilon,
    )
    key_pre = lfm2_runtime_rms_norm(
        key_linear,
        runtime.source_key_norm_weight,
        runtime.norm_epsilon,
    )
    query_cos = _exact_float_tensor(
        case.query_rope_cos,
        f"{case_name}.query_rope_cos",
        ndim=3,
    )
    query_sin = _exact_float_tensor(
        case.query_rope_sin,
        f"{case_name}.query_rope_sin",
        ndim=3,
    )
    key_cos = _exact_float_tensor(
        case.key_rope_cos,
        f"{case_name}.key_rope_cos",
        ndim=3,
    )
    key_sin = _exact_float_tensor(
        case.key_rope_sin,
        f"{case_name}.key_rope_sin",
        ndim=3,
    )
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
    if any(
        not torch.equal(value[..., :half], value[..., half:])
        for value in (query_cos, query_sin, key_cos, key_sin)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} RoPE terms do not preserve exact duplicated halves."
        )
    radius_error = max(
        float(
            (
                query_cos[..., :half].to(torch.float64).square()
                + query_sin[..., :half].to(torch.float64).square()
                - 1.0
            )
            .abs()
            .max()
            .item()
        ),
        float(
            (
                key_cos[..., :half].to(torch.float64).square()
                + key_sin[..., :half].to(torch.float64).square()
                - 1.0
            )
            .abs()
            .max()
            .item()
        ),
    )
    geometry_bound = _rope_low_precision_geometry_error_bound(evidence_dtype)
    if geometry_bound is None:
        geometry_bound = max(policy.authentication_atol * 4.0, 1e-7)
    position_chord = torch.sqrt(
        (
            query_cos[..., :half].to(torch.float64)
            - key_cos[:, -query_length:, :half].to(torch.float64)
        ).square()
        + (
            query_sin[..., :half].to(torch.float64)
            - key_sin[:, -query_length:, :half].to(torch.float64)
        ).square()
    )
    if (
        not math.isfinite(radius_error)
        or radius_error > geometry_bound
        or float(position_chord.max().item()) > geometry_bound
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} RoPE geometry is illegal for {evidence_dtype}."
        )
    reconstructed_query_post = _apply_rope(query_pre, query_cos, query_sin)
    reconstructed_key_post = _apply_rope(key_pre, key_cos, key_sin)
    reconstruction_bound = _rope_low_precision_reconstruction_error_bound(
        evidence_dtype
    )
    if reconstruction_bound is None:
        post_legal = bool(
            torch.allclose(
                reconstructed_query_post,
                query_post,
                rtol=policy.authentication_rtol,
                atol=policy.authentication_atol,
            )
            and torch.allclose(
                reconstructed_key_post,
                key_post,
                rtol=policy.authentication_rtol,
                atol=policy.authentication_atol,
            )
        )
    else:
        post_legal = (
            _maximum_rope_pair_relative_error(
                query_pre,
                query_post,
                reconstructed_query_post,
            )
            <= reconstruction_bound
            and _maximum_rope_pair_relative_error(
                key_pre,
                key_post,
                reconstructed_key_post,
            )
            <= reconstruction_bound
        )
    if not post_legal:
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} failed raw-linear→RMSNorm→RoPE authentication."
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
    repeated_key = query_post.new_empty(0)
    repeated_key = key_post.repeat_interleave(
        topology.source_queries_per_key_value,
        dim=1,
    )
    raw_scores = torch.matmul(query_post, repeated_key.transpose(-2, -1)) / math.sqrt(
        source_dim
    )
    reconstructed_scores_exact = _center_scores(raw_scores, mask)
    supplied_scores_exact = _exact_float_tensor(
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
        tuple(supplied_scores_exact.shape) != expected_score_shape
        or tuple(supplied_probabilities.shape) != expected_score_shape
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} score/probability evidence has the wrong shape."
        )
    score_rtol = max(
        policy.authentication_rtol,
        0.0 if reconstruction_bound is None else reconstruction_bound,
    )
    score_atol = max(
        policy.authentication_atol,
        float(torch.finfo(evidence_dtype).eps)
        if evidence_dtype in {torch.float16, torch.bfloat16}
        else policy.authentication_atol,
    )
    reconstructed_scores = reconstructed_scores_exact.to(torch.float64)
    reconstructed_probabilities = _masked_softmax(reconstructed_scores, mask)
    if not (
        torch.allclose(
            reconstructed_scores,
            supplied_scores_exact.to(torch.float64),
            rtol=score_rtol,
            atol=score_atol,
        )
        and torch.allclose(
            reconstructed_probabilities,
            supplied_probabilities,
            rtol=score_rtol,
            atol=score_atol,
        )
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} failed authenticated QK score/softmax replay."
        )
    query_sqrt = _float_tensor(
        case.query_row_sqrt_weights,
        f"{case_name}.query_row_sqrt_weights",
        ndim=2,
    )
    key_sqrt = _float_tensor(
        case.key_value_row_sqrt_weights,
        f"{case_name}.key_value_row_sqrt_weights",
        ndim=2,
    )
    if (
        tuple(query_sqrt.shape) != (batch, query_length)
        or tuple(key_sqrt.shape) != (batch, key_length)
        or bool((query_sqrt <= 0.0).any().item())
        or bool((key_sqrt <= 0.0).any().item())
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            f"{case_name} row sqrt weights must be finite positive [B,Q]/[B,K]."
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
        query_inputs=query_inputs_exact,
        key_inputs=key_inputs_exact,
        parent_query_pre=query_pre.to(torch.float64),
        parent_key_pre=key_pre.to(torch.float64),
        parent_query_post=query_post.to(torch.float64),
        parent_key_post=key_post.to(torch.float64),
        parent_scores=supplied_scores_exact.to(torch.float64),
        parent_probabilities=supplied_probabilities,
        query_cos=query_cos,
        query_sin=query_sin,
        key_cos=key_cos,
        key_sin=key_sin,
        mask=mask,
        route=route,
        evidence_sha256=case.evidence_sha256,
        parent_query_linear=query_linear.to(torch.float64),
        parent_key_linear=key_linear.to(torch.float64),
        query_sqrt_weights=query_sqrt,
        key_sqrt_weights=key_sqrt,
        provenance=case.provenance,
        production=True,
    )


def _production_row_identities(
    case: Lfm2CpcfAttentionDimensionProductionCase,
) -> set[str]:
    """Return packaging-invariant query/key row identities for overlap checks."""

    identities: set[str] = set()
    trajectory = case.provenance.trajectory_sha256
    batch = int(case.target_query_residual_inputs.shape[0])
    for batch_index in range(batch):
        for token in range(int(case.target_query_residual_inputs.shape[1])):
            body = {
                "role": "query",
                "trajectory_sha256": trajectory,
                "target_input": _tensor_sha256(
                    case.target_query_residual_inputs[batch_index, token]
                ),
                "parent_linear": _tensor_sha256(
                    case.parent_query_linear[batch_index, :, token]
                ),
                "parent_post_rope": _tensor_sha256(
                    case.parent_query_post_rope[batch_index, :, token]
                ),
                "rope_cos": _tensor_sha256(case.query_rope_cos[batch_index, token]),
                "rope_sin": _tensor_sha256(case.query_rope_sin[batch_index, token]),
            }
            identities.add(_canonical_sha256(body))
        for token in range(int(case.target_key_value_residual_inputs.shape[1])):
            body = {
                "role": "key",
                "trajectory_sha256": trajectory,
                "target_input": _tensor_sha256(
                    case.target_key_value_residual_inputs[batch_index, token]
                ),
                "parent_linear": _tensor_sha256(
                    case.parent_key_linear[batch_index, :, token]
                ),
                "parent_post_rope": _tensor_sha256(
                    case.parent_key_post_rope[batch_index, :, token]
                ),
                "rope_cos": _tensor_sha256(case.key_rope_cos[batch_index, token]),
                "rope_sin": _tensor_sha256(case.key_rope_sin[batch_index, token]),
            }
            identities.add(_canonical_sha256(body))
    return identities


def _authenticate_production_partitions(
    fit_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
    fit_check_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
) -> dict[str, Any]:
    all_cases = (*fit_cases, *fit_check_cases)
    source_receipts = {case.provenance.source_split_sha256 for case in all_cases}
    if len(source_receipts) != 1:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "FIT and FIT-CHECK cases must share one authenticated upstream FIT "
            "split receipt."
        )
    source_runtime_receipts = {
        case.provenance.source_runtime_sha256 for case in all_cases
    }
    if len(source_runtime_receipts) != 1:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "FIT and FIT-CHECK cases must share one authenticated source "
            "RMSNorm runtime receipt."
        )
    fit_rows: set[str] = set()
    for case in fit_cases:
        fit_rows.update(_production_row_identities(case))
    fit_check_rows: set[str] = set()
    for case in fit_check_cases:
        fit_check_rows.update(_production_row_identities(case))
    overlap = fit_rows & fit_check_rows
    if overlap:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "FIT and FIT-CHECK contain overlapping query/key rows, including "
            "a repacked or subset overlap."
        )
    fit_trajectories = {case.provenance.trajectory_sha256 for case in fit_cases}
    fit_check_trajectories = {
        case.provenance.trajectory_sha256 for case in fit_check_cases
    }
    shared_trajectories = fit_trajectories & fit_check_trajectories
    if shared_trajectories:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "FIT and FIT-CHECK must not share a source trajectory, including "
            "distinct phases or repacked subsets of that trajectory."
        )
    return {
        "upstream_split": "fit",
        "source_split_sha256": next(iter(source_receipts)),
        "source_runtime_sha256": next(iter(source_runtime_receipts)),
        "fit_trajectory_count": len(fit_trajectories),
        "fit_check_trajectory_count": len(fit_check_trajectories),
        "shared_trajectory_count": 0,
        "fit_row_identity_count": len(fit_rows),
        "fit_check_row_identity_count": len(fit_check_rows),
        "overlap_row_identity_count": 0,
        "fit_provenance_sha256": _canonical_sha256(
            [case.provenance.as_dict() for case in fit_cases]
        ),
        "fit_check_provenance_sha256": _canonical_sha256(
            [case.provenance.as_dict() for case in fit_check_cases]
        ),
    }


def _production_evidence_bundle_sha256(
    fit_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
    fit_check_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
) -> str:
    digest = hashlib.sha256()
    for partition, cases in (
        ("fit", fit_cases),
        ("fit_check", fit_check_cases),
    ):
        for index, case in enumerate(cases):
            digest.update(partition.encode("ascii"))
            digest.update(str(index).encode("ascii"))
            digest.update(bytes.fromhex(case.verify()))
    return digest.hexdigest()


def _production_resource_usage(
    cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
) -> dict[str, int]:
    torch = _torch()
    query_rows = 0
    key_rows = 0
    evidence_bytes = 0
    for case in cases:
        query_rows += int(case.target_query_residual_inputs.shape[0]) * int(
            case.target_query_residual_inputs.shape[1]
        )
        key_rows += int(case.target_key_value_residual_inputs.shape[0]) * int(
            case.target_key_value_residual_inputs.shape[1]
        )
        for name in case.__dataclass_fields__:
            value = getattr(case, name)
            if isinstance(value, torch.Tensor):
                evidence_bytes += int(value.numel()) * int(value.element_size())
    return {
        "aggregate_query_rows": query_rows,
        "aggregate_key_rows": key_rows,
        "aggregate_evidence_bytes": evidence_bytes,
    }


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


def _frequency_importance_production(
    cases: Sequence[_NormalizedCase],
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
) -> tuple[Any, tuple[int, ...]]:
    torch = _torch()
    pair_count = topology.source_head_dim // 2
    importance = torch.zeros(pair_count, dtype=torch.float64)
    query_head_weight = _source_group_weights(
        query_groups,
        topology.source_query_heads,
    )
    key_head_weight = _source_group_weights(
        key_value_groups,
        topology.source_key_value_heads,
    )
    source_kv = (
        torch.arange(topology.source_query_heads, dtype=torch.long)
        // topology.source_queries_per_key_value
    )
    head_weight = query_head_weight * key_head_weight.index_select(0, source_kv)
    for case in cases:
        assert case.query_sqrt_weights is not None
        assert case.key_sqrt_weights is not None
        repeated_key = case.parent_key_post.repeat_interleave(
            topology.source_queries_per_key_value,
            dim=1,
        )
        cell_weight = case.query_sqrt_weights.square().reshape(
            int(case.query_sqrt_weights.shape[0]),
            1,
            int(case.query_sqrt_weights.shape[1]),
            1,
        ) * case.key_sqrt_weights.square().reshape(
            int(case.key_sqrt_weights.shape[0]),
            1,
            1,
            int(case.key_sqrt_weights.shape[1]),
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
                score_alignment
                * case.route
                * cell_weight
                * head_weight.reshape(1, -1, 1, 1)
            ).sum()
    ranked = sorted(
        range(pair_count),
        key=lambda index: (-float(importance[index].item()), index),
    )
    selected = tuple(sorted(ranked[: topology.target_head_dim // 2]))
    return importance, selected


def _attach_target_barycenters_production(
    case: _NormalizedCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    key_value_groups: tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    selected: tuple[int, ...],
) -> None:
    assert case.parent_query_linear is not None
    assert case.parent_key_linear is not None
    scale = (topology.target_head_dim / topology.source_head_dim) ** 0.25
    case.desired_scores = _weighted_head_barycenter(
        case.parent_scores,
        query_groups,
    )
    case.probability_barycenter = _weighted_head_barycenter(
        case.parent_probabilities,
        query_groups,
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
    case.query_linear_anchor = (
        _weighted_head_barycenter(
            _select_pair_coordinates(case.parent_query_linear, selected),
            query_groups,
        )
        * scale
    )
    case.key_linear_anchor = (
        _weighted_head_barycenter(
            _select_pair_coordinates(case.parent_key_linear, selected),
            key_value_groups,
        )
        * scale
    )
    case.target_query_cos = _select_rope_terms(case.query_cos, selected)
    case.target_query_sin = _select_rope_terms(case.query_sin, selected)
    case.target_key_cos = _select_rope_terms(case.key_cos, selected)
    case.target_key_sin = _select_rope_terms(case.key_sin, selected)


def _case_capacity_and_factor_targets_production(
    case: _NormalizedCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    policy: Lfm2CpcfAttentionDimensionPolicy,
    limits: Lfm2CpcfAttentionDimensionLimits,
    runtime: Lfm2CpcfAttentionDimensionRuntime,
) -> tuple[
    Any,
    Any,
    list[dict[str, Any]],
    Lfm2CpcfAttentionFactorSufficientState,
]:
    torch = _torch()
    assert case.desired_scores is not None
    assert case.desired_mask is not None
    assert case.query_anchor is not None
    assert case.key_anchor is not None
    assert case.query_linear_anchor is not None
    assert case.key_linear_anchor is not None
    assert case.target_query_cos is not None
    assert case.target_query_sin is not None
    assert case.target_key_cos is not None
    assert case.target_key_sin is not None
    assert case.query_sqrt_weights is not None
    assert case.key_sqrt_weights is not None
    assert case.provenance is not None
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
        query_sqrt = case.query_sqrt_weights[batch_index]
        key_sqrt = case.key_sqrt_weights[batch_index]
        for group in range(topology.target_key_value_heads):
            start = group * offsets
            stop = start + offsets
            matrix = case.desired_scores[batch_index, start:stop].reshape(
                offsets * query_length,
                key_length,
            )
            if matrix.numel() > limits.maximum_score_matrix_elements_per_group:
                raise Lfm2CpcfAttentionDimensionFoldError(
                    "A target GQA score matrix exceeds the SVD ceiling."
                )
            query_factor_sqrt = query_sqrt.repeat(offsets)
            weighted_matrix = (
                matrix * query_factor_sqrt.reshape(-1, 1) * key_sqrt.reshape(1, -1)
            )
            left, singular, right_h = torch.linalg.svd(
                weighted_matrix * math.sqrt(target_dim),
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
                offsets * query_length,
                target_dim,
            )
            k_anchor = case.key_anchor[batch_index, group]
            anchor_scores = (q_anchor @ k_anchor.T / math.sqrt(target_dim)).reshape(
                offsets, query_length, key_length
            )
            anchor_mask = case.desired_mask[batch_index, start:stop]
            anchor_scores = _center_scores(anchor_scores, anchor_mask)
            target_scores = case.desired_scores[batch_index, start:stop]
            cell_weights = (
                query_sqrt.square().reshape(1, query_length, 1)
                * key_sqrt.square().reshape(1, 1, key_length)
            ).expand(offsets, -1, -1)
            anchor_error = _relative_rmse(
                anchor_scores,
                target_scores,
                anchor_mask,
                cell_weights,
            )
            if anchor_error <= policy.anchor_exact_relative_rmse:
                q_rows = q_anchor
                k_rows = k_anchor
                gauge = "weighted_selected_source_pair_anchor_exact"
            else:
                q_weighted = torch.zeros(
                    (offsets * query_length, target_dim),
                    dtype=torch.float64,
                )
                k_weighted = torch.zeros(
                    (key_length, target_dim),
                    dtype=torch.float64,
                )
                root = torch.sqrt(singular[:kept].clamp_min(0.0))
                q_weighted[:, :kept] = left[:, :kept] * root
                k_weighted[:, :kept] = right_h[:kept].T * root
                q_anchor_weighted = q_anchor * query_factor_sqrt.reshape(-1, 1)
                k_anchor_weighted = k_anchor * key_sqrt.reshape(-1, 1)
                cross = (
                    q_weighted.T @ q_anchor_weighted + k_weighted.T @ k_anchor_weighted
                )
                orient_left, _orient_singular, orient_right_h = torch.linalg.svd(
                    cross, full_matrices=False
                )
                rotation = orient_left @ orient_right_h
                q_rows = (q_weighted @ rotation) / query_factor_sqrt.reshape(-1, 1)
                k_rows = (k_weighted @ rotation) / key_sqrt.reshape(-1, 1)
                gauge = "weighted_svd_balanced_selected_pair_procrustes"
            query_factor[batch_index, start:stop] = q_rows.reshape(
                offsets,
                query_length,
                target_dim,
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
                    "query_row_weighted": True,
                    "key_row_weighted": True,
                }
            )
    query_linear_target = lfm2_projection_target_from_runtime(
        query_factor,
        case.query_linear_anchor,
        norm_weight=runtime.target_query_norm_weight,
        epsilon=runtime.norm_epsilon,
        cosine=case.target_query_cos,
        sine=case.target_query_sin,
    )
    key_linear_target = lfm2_projection_target_from_runtime(
        key_factor,
        case.key_linear_anchor,
        norm_weight=runtime.target_key_norm_weight,
        epsilon=runtime.norm_epsilon,
        cosine=case.target_key_cos,
        sine=case.target_key_sin,
    )
    sufficient_state = Lfm2CpcfAttentionFactorSufficientState(
        partition=case.provenance.partition,
        case_ordinal=case.provenance.case_ordinal,
        evidence_sha256=case.evidence_sha256,
        local_query_post_rope_factor=query_factor,
        local_key_post_rope_factor=key_factor,
        query_post_rope_anchor=case.query_anchor,
        key_post_rope_anchor=case.key_anchor,
        query_raw_linear_anchor=case.query_linear_anchor,
        key_raw_linear_anchor=case.key_linear_anchor,
        target_query_inputs=case.query_inputs,
        target_key_inputs=case.key_inputs,
        query_row_sqrt_weights=case.query_sqrt_weights,
        key_row_sqrt_weights=case.key_sqrt_weights,
        target_query_rope_cos=case.target_query_cos,
        target_query_rope_sin=case.target_query_sin,
        target_key_rope_cos=case.target_key_cos,
        target_key_rope_sin=case.target_key_sin,
    )
    return (
        query_linear_target,
        key_linear_target,
        capacity,
        sufficient_state,
    )


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


def _replay_normalized_production(
    case: _NormalizedCase,
    topology: Lfm2CpcfAttentionDimensionTopology,
    query_weight: Any,
    key_weight: Any,
    runtime: Lfm2CpcfAttentionDimensionRuntime,
) -> Lfm2CpcfAttentionDimensionReplay:
    """Replay the exact published target dtype through Q/K norm and RoPE."""

    torch = _torch()
    assert case.desired_mask is not None
    assert case.target_query_cos is not None
    assert case.target_query_sin is not None
    assert case.target_key_cos is not None
    assert case.target_key_sin is not None
    dtype = runtime.target_dtype
    query_template = _exact_float_tensor(
        query_weight,
        "query_weight",
        ndim=2,
    )
    key_template = _exact_float_tensor(
        key_weight,
        "key_weight",
        ndim=2,
    )
    if query_template.dtype != dtype or key_template.dtype != dtype:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production replay requires already-cast exact target Q/K state."
        )
    batch, query_length, _query_width = map(int, case.query_inputs.shape)
    key_length = int(case.key_inputs.shape[1])
    target_dim = topology.target_head_dim
    query_linear = (
        (case.query_inputs.to(dtype=dtype) @ query_template.T)
        .reshape(
            batch,
            query_length,
            topology.target_query_heads,
            target_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key_linear = (
        (case.key_inputs.to(dtype=dtype) @ key_template.T)
        .reshape(
            batch,
            key_length,
            topology.target_key_value_heads,
            target_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    query_pre = lfm2_runtime_rms_norm(
        query_linear,
        runtime.target_query_norm_weight,
        runtime.norm_epsilon,
    )
    key_pre = lfm2_runtime_rms_norm(
        key_linear,
        runtime.target_key_norm_weight,
        runtime.norm_epsilon,
    )
    query_cos = case.target_query_cos.to(dtype=dtype)
    query_sin = case.target_query_sin.to(dtype=dtype)
    key_cos = case.target_key_cos.to(dtype=dtype)
    key_sin = case.target_key_sin.to(dtype=dtype)
    query_post = _apply_rope(query_pre, query_cos, query_sin)
    key_post = _apply_rope(key_pre, key_cos, key_sin)
    repeated_key = key_post.repeat_interleave(
        topology.target_queries_per_key_value,
        dim=1,
    )
    raw_scores = torch.matmul(query_post, repeated_key.transpose(-2, -1)) / math.sqrt(
        target_dim
    )
    centered = _center_scores(
        raw_scores,
        case.desired_mask,
    ).to(torch.float64)
    probabilities = _masked_softmax(centered, case.desired_mask)
    return Lfm2CpcfAttentionDimensionReplay(
        centered_scores=centered,
        probabilities=probabilities,
        causal_mask=case.desired_mask,
        query_rope_cos=query_cos,
        query_rope_sin=query_sin,
        key_rope_cos=key_cos,
        key_rope_sin=key_sin,
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


def _routing_metrics_production(
    cases: Sequence[_NormalizedCase],
    replays: Sequence[Lfm2CpcfAttentionDimensionReplay],
) -> Lfm2CpcfAttentionRoutingMetrics:
    """Compute Q/K row-weighted routing metrics for frozen production state."""

    torch = _torch()
    score_predictions: list[Any] = []
    score_targets: list[Any] = []
    probability_predictions: list[Any] = []
    probability_targets: list[Any] = []
    probability_barycenters: list[Any] = []
    masks: list[Any] = []
    routes: list[Any] = []
    cell_weights: list[Any] = []
    weighted_js_sum = 0.0
    js_weight_sum = 0.0
    js_worst = 0.0
    top_match_weight = 0.0
    eligible_weight = 0.0
    eligible = 0
    leakage = 0.0
    for case, replay in zip(cases, replays, strict=True):
        assert case.desired_scores is not None
        assert case.desired_probabilities is not None
        assert case.probability_barycenter is not None
        assert case.desired_mask is not None
        assert case.desired_route is not None
        assert case.query_sqrt_weights is not None
        assert case.key_sqrt_weights is not None
        prediction_scores = replay.centered_scores.to(torch.float64)
        prediction_probabilities = replay.probabilities.to(torch.float64)
        target_heads = int(case.desired_scores.shape[1])
        weights = (
            case.query_sqrt_weights.square().reshape(
                int(case.query_sqrt_weights.shape[0]),
                1,
                int(case.query_sqrt_weights.shape[1]),
                1,
            )
            * case.key_sqrt_weights.square().reshape(
                int(case.key_sqrt_weights.shape[0]),
                1,
                1,
                int(case.key_sqrt_weights.shape[1]),
            )
        ).expand(-1, target_heads, -1, -1)
        score_predictions.append(prediction_scores)
        score_targets.append(case.desired_scores)
        probability_predictions.append(prediction_probabilities)
        probability_targets.append(case.desired_probabilities)
        probability_barycenters.append(case.probability_barycenter)
        masks.append(case.desired_mask)
        routes.append(case.desired_route)
        cell_weights.append(weights)
        if bool((~case.desired_mask).any().item()):
            leakage = max(
                leakage,
                float(prediction_probabilities[~case.desired_mask].abs().max().item()),
            )
        flat_prediction = prediction_probabilities.reshape(
            -1,
            int(prediction_probabilities.shape[-1]),
        )
        flat_target = case.desired_probabilities.reshape_as(flat_prediction)
        flat_mask = case.desired_mask.reshape_as(flat_prediction)
        flat_weights = weights.reshape_as(flat_prediction)
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
            js_value = float(js.clamp_min(0.0).item())
            row_weight = float(flat_weights[row, valid].mean().item())
            weighted_js_sum += js_value * row_weight
            js_weight_sum += row_weight
            js_worst = max(js_worst, js_value)
            reference_max = reference.max()
            reference_winners = torch.isclose(
                reference,
                reference_max,
                rtol=1e-10,
                atol=1e-12,
            )
            match = bool(reference_winners[int(candidate.argmax().item())].item())
            top_match_weight += float(match) * row_weight
            eligible_weight += row_weight
            eligible += 1
    if eligible == 0 or eligible_weight <= 0.0 or js_weight_sum <= 0.0:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Weighted routing evidence has no eligible causal query row."
        )
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
    weights = torch.cat([value.reshape(-1) for value in cell_weights])
    return Lfm2CpcfAttentionRoutingMetrics(
        case_count=len(cases),
        eligible_query_rows=eligible,
        centered_score_relative_rmse=_relative_rmse(
            score_prediction,
            score_target,
            mask,
            weights,
        ),
        centered_score_retained_energy_fraction=_retained_energy(
            score_prediction,
            score_target,
            mask,
            weights,
        ),
        probability_relative_rmse=_relative_rmse(
            probability_prediction,
            probability_target,
            mask,
            weights,
        ),
        route_probability_relative_rmse=_relative_rmse(
            probability_prediction,
            probability_target,
            mask,
            weights * route,
        ),
        probability_barycenter_relative_rmse=_relative_rmse(
            probability_prediction,
            probability_barycenter,
            mask,
            weights,
        ),
        js_divergence_mean=weighted_js_sum / js_weight_sum,
        js_divergence_worst=js_worst,
        top1_route_agreement_fraction=top_match_weight / eligible_weight,
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


def _projection_fit_metrics_production(
    inputs: Any,
    targets: Any,
    weight: Any,
    sqrt_weights: Any,
) -> dict[str, Any]:
    torch = _torch()
    prediction = inputs @ weight.to(torch.float64).T
    weighted_error = (prediction - targets) * sqrt_weights
    weighted_target = targets * sqrt_weights
    denominator = torch.linalg.vector_norm(weighted_target).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    relative = float((torch.linalg.vector_norm(weighted_error) / denominator).item())
    weighted_inputs = inputs * sqrt_weights
    rank = int(torch.linalg.matrix_rank(weighted_inputs).item())
    return {
        "rows": int(inputs.shape[0]),
        "input_width": int(inputs.shape[1]),
        "output_width": int(targets.shape[1]),
        "weighted_design_rank": rank,
        "weighted_design_full_column_rank": rank == int(inputs.shape[1]),
        "weighted_target_state_relative_rmse": relative,
        "query_or_key_row_sqrt_weighted": True,
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


def _checks_production(
    fit: Lfm2CpcfAttentionRoutingMetrics,
    fit_check: Lfm2CpcfAttentionRoutingMetrics,
    policy: Lfm2CpcfAttentionDimensionPolicy,
) -> dict[str, bool | None]:
    legacy = _checks(fit, fit_check, policy)
    return {
        "maximum_fit_centered_score_relative_rmse": legacy[
            "maximum_fit_centered_score_relative_rmse"
        ],
        "maximum_fit_check_centered_score_relative_rmse": legacy[
            "maximum_validation_centered_score_relative_rmse"
        ],
        "maximum_fit_check_js_divergence": legacy["maximum_validation_js_divergence"],
        "maximum_fit_check_route_probability_relative_rmse": legacy[
            "maximum_validation_route_probability_relative_rmse"
        ],
        "minimum_fit_check_top1_route_agreement": legacy[
            "minimum_validation_top1_route_agreement"
        ],
        "zero_causal_probability_leakage": legacy["zero_causal_probability_leakage"],
    }


def _compile_lfm2_cpcf_attention_dimension_fold_legacy(
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


def _compile_lfm2_cpcf_attention_dimension_fold_production(
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    fit_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
    fit_check_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
    query_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    key_value_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    runtime: Lfm2CpcfAttentionDimensionRuntime,
    policy: Lfm2CpcfAttentionDimensionPolicy | None,
    limits: Lfm2CpcfAttentionDimensionLimits | None,
    ridge: float,
) -> Lfm2CompiledCpcfAttentionDimensionFold:
    torch = _torch()
    if runtime.solver_mode == "global_gauge_radius_synchronized":
        raise Lfm2CpcfAttentionDimensionFoldError(
            "The global_gauge_radius_synchronized production solver mode is "
            "reserved by the public runtime contract but is not implemented "
            "yet; per-case local gauges must not be represented as that solver."
        )
    fit_values = tuple(fit_cases)
    fit_check_values = tuple(fit_check_cases)
    if not fit_values or not fit_check_values:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Disjoint nonempty FIT and FIT-CHECK cases are required."
        )
    if not all(
        isinstance(value, Lfm2CpcfAttentionDimensionProductionCase)
        for value in (*fit_values, *fit_check_values)
    ):
        raise TypeError("Production compilation requires production attention cases.")
    resource_usage = _production_resource_usage((*fit_values, *fit_check_values))
    if (
        resource_usage["aggregate_query_rows"] > runtime.maximum_aggregate_query_rows
        or resource_usage["aggregate_key_rows"] > runtime.maximum_aggregate_key_rows
        or resource_usage["aggregate_evidence_bytes"]
        > runtime.maximum_aggregate_evidence_bytes
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production Q/K evidence exceeds an aggregate row or byte ceiling."
        )
    provenance_report = _authenticate_production_partitions(
        fit_values,
        fit_check_values,
    )
    policy_value = policy or Lfm2CpcfAttentionDimensionPolicy()
    limits_value = limits or _production_limits(runtime)
    if not isinstance(policy_value, Lfm2CpcfAttentionDimensionPolicy):
        raise TypeError("policy must be Lfm2CpcfAttentionDimensionPolicy.")
    if not isinstance(limits_value, Lfm2CpcfAttentionDimensionLimits):
        raise TypeError("limits must be Lfm2CpcfAttentionDimensionLimits.")
    if (
        len(fit_values) > runtime.maximum_fit_cases
        or len(fit_check_values) > runtime.maximum_fit_check_cases
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "The production FIT/FIT-CHECK partitions exceed the runtime "
            "contract ceilings."
        )
    _validate_limits(
        topology,
        len(fit_values),
        len(fit_check_values),
        limits_value,
    )
    ridge_value = _finite_nonnegative(ridge, "ridge")
    if ridge_value == 0.0:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "ridge must be positive for a bounded deterministic solve."
        )
    expected_query_template = (
        topology.target_query_heads * topology.target_head_dim,
        runtime.target_hidden_width,
    )
    expected_key_template = (
        topology.target_key_value_heads * topology.target_head_dim,
        runtime.target_hidden_width,
    )
    if (
        tuple(runtime.target_query_weight_template.shape) != expected_query_template
        or tuple(runtime.target_key_weight_template.shape) != expected_key_template
        or tuple(runtime.target_query_norm_weight.shape) != (topology.target_head_dim,)
        or tuple(runtime.target_key_norm_weight.shape) != (topology.target_head_dim,)
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Exact target Q/K templates or RMSNorm weights differ from the "
            "target topology and hidden width."
        )
    if {
        runtime.source_query_norm_weight.dtype,
        runtime.source_key_norm_weight.dtype,
        runtime.target_query_norm_weight.dtype,
        runtime.target_key_norm_weight.dtype,
    } != {runtime.target_dtype}:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production Q/K projection templates and all source/target RMSNorm "
            "weights must preserve one exact floating dtype."
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
    source_before = _production_evidence_bundle_sha256(
        fit_values,
        fit_check_values,
    )
    normalized_fit = tuple(
        _authenticate_production_case(
            case,
            topology,
            runtime,
            case_name=f"fit_cases[{index}]",
            limits=limits_value,
            policy=policy_value,
            expected_partition="fit",
        )
        for index, case in enumerate(fit_values)
    )
    normalized_fit_check = tuple(
        _authenticate_production_case(
            case,
            topology,
            runtime,
            case_name=f"fit_check_cases[{index}]",
            limits=limits_value,
            policy=policy_value,
            expected_partition="fit_check",
        )
        for index, case in enumerate(fit_check_values)
    )
    all_normalized = (*normalized_fit, *normalized_fit_check)
    query_widths = {int(value.query_inputs.shape[-1]) for value in all_normalized}
    key_widths = {int(value.key_inputs.shape[-1]) for value in all_normalized}
    evidence_dtypes = {value.query_cos.dtype for value in all_normalized}
    if (
        query_widths != {runtime.target_hidden_width}
        or key_widths != {runtime.target_hidden_width}
        or query_widths != key_widths
    ):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production Q/K inputs must have equal width matching the exact "
            "target hidden width."
        )
    if evidence_dtypes != {runtime.target_dtype}:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production RoPE evidence dtype must equal the exact target Q/K "
            "template dtype."
        )
    importance, selected = _frequency_importance_production(
        normalized_fit,
        topology,
        query_group_values,
        key_group_values,
    )
    for value in all_normalized:
        _attach_target_barycenters_production(
            value,
            topology,
            query_group_values,
            key_group_values,
            selected,
        )
    fit_q_targets: list[Any] = []
    fit_k_targets: list[Any] = []
    fit_capacity: list[dict[str, Any]] = []
    fit_check_capacity: list[dict[str, Any]] = []
    sufficient_states: list[Lfm2CpcfAttentionFactorSufficientState] = []
    for index, value in enumerate(normalized_fit):
        query_target, key_target, capacity, sufficient_state = (
            _case_capacity_and_factor_targets_production(
                value,
                topology,
                policy_value,
                limits_value,
                runtime,
            )
        )
        fit_q_targets.append(query_target)
        fit_k_targets.append(key_target)
        sufficient_states.append(sufficient_state)
        fit_capacity.extend({"case_index": index, **record} for record in capacity)
    for index, value in enumerate(normalized_fit_check):
        _query_target, _key_target, capacity, sufficient_state = (
            _case_capacity_and_factor_targets_production(
                value,
                topology,
                policy_value,
                limits_value,
                runtime,
            )
        )
        fit_check_capacity.extend(
            {"case_index": index, **record} for record in capacity
        )
        sufficient_states.append(sufficient_state)
    all_capacity = (*fit_capacity, *fit_check_capacity)
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
        "fit_check_matrices": fit_check_capacity,
        "accepted": (
            minimum_retained >= policy_value.minimum_score_spectral_energy_fraction
        ),
    }
    if not capacity_report["accepted"]:
        body = {
            "schema": _SCHEMA,
            "production_runtime_contract": True,
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
            value.query_inputs.to(torch.float64).reshape(
                -1,
                runtime.target_hidden_width,
            )
            for value in normalized_fit
        ],
        dim=0,
    )
    key_inputs = torch.cat(
        [
            value.key_inputs.to(torch.float64).reshape(
                -1,
                runtime.target_hidden_width,
            )
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
    query_fit_sqrt = torch.cat(
        [
            value.query_sqrt_weights.reshape(-1, 1)
            for value in normalized_fit
            if value.query_sqrt_weights is not None
        ],
        dim=0,
    )
    key_fit_sqrt = torch.cat(
        [
            value.key_sqrt_weights.reshape(-1, 1)
            for value in normalized_fit
            if value.key_sqrt_weights is not None
        ],
        dim=0,
    )
    query_weight64 = _ridge_fit(
        query_inputs * query_fit_sqrt,
        query_targets * query_fit_sqrt,
        ridge_value,
        "query",
    )
    key_weight64 = _ridge_fit(
        key_inputs * key_fit_sqrt,
        key_targets * key_fit_sqrt,
        ridge_value,
        "key",
    )
    query_weight = query_weight64.to(dtype=runtime.target_dtype).contiguous()
    key_weight = key_weight64.to(dtype=runtime.target_dtype).contiguous()
    fit_replays = tuple(
        _replay_normalized_production(
            value,
            topology,
            query_weight,
            key_weight,
            runtime,
        )
        for value in normalized_fit
    )
    fit_check_replays = tuple(
        _replay_normalized_production(
            value,
            topology,
            query_weight,
            key_weight,
            runtime,
        )
        for value in normalized_fit_check
    )
    fit_metrics = _routing_metrics_production(
        normalized_fit,
        fit_replays,
    )
    fit_check_metrics = _routing_metrics_production(
        normalized_fit_check,
        fit_check_replays,
    )
    checks = _checks_production(
        fit_metrics,
        fit_check_metrics,
        policy_value,
    )
    source_after = _production_evidence_bundle_sha256(
        fit_values,
        fit_check_values,
    )
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
        "target_parameter_mutations_during_fit_check": 0,
        "selection_cases_consumed": 0,
        "true_validation_cases_consumed": 0,
        "published_state_cast_before_fit_check": True,
        "all_source_query_heads_contribute": True,
        "all_source_key_value_heads_contribute": True,
        "source_query_heads_dropped": 0,
        "source_key_value_heads_dropped": 0,
        "dense_target_projection_state": True,
    }
    body = {
        "schema": _SCHEMA,
        "method": (
            "fit_only_row_weighted_complete_rope_pair_selection_then_"
            "weighted_target_gqa_score_svd_projection_preimage_and_"
            "weighted_ridge_exact_dtype_fit_check"
        ),
        "production_runtime_contract": True,
        "accepted": (
            all(value is not False for value in checks.values()) and source_unchanged
        ),
        "topology": topology.as_dict(),
        "runtime": runtime.as_dict(),
        "solver_contract": {
            "requested_mode": runtime.solver_mode,
            "executed_mode": "per_case_local_gauge",
            "global_gauge_radius_synchronized_available": False,
            "current_mode_reuses_per_case_structural_factor_route": True,
            "current_mode_is_not_claimed_as_qk_collapse_solution": True,
            "rope_frequency_contraction_active": (
                topology.target_head_dim < topology.source_head_dim
            ),
            "factor_sufficient_state_exposed": True,
        },
        "solver_error_ladder": [
            {
                "stage": "exact_runtime_chain_and_provenance",
                "status": "implemented_and_authenticated",
            },
            {
                "stage": "row_weighted_local_factor_and_ridge",
                "status": "implemented_diagnostic_baseline",
            },
            {
                "stage": "cross_case_global_gauge_radius_synchronization",
                "status": "required_not_implemented",
            },
            {
                "stage": "exact_runtime_score_residual_correction",
                "status": "follow_on_if_global_synchronization_is_insufficient",
            },
            {
                "stage": "qk_collapse_solution",
                "status": "not_claimed_by_current_mode",
            },
        ],
        "query_input_width": runtime.target_hidden_width,
        "key_value_input_width": runtime.target_hidden_width,
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
        "query_projection_fit": _projection_fit_metrics_production(
            query_inputs,
            query_targets,
            query_weight,
            query_fit_sqrt,
        ),
        "key_projection_fit": _projection_fit_metrics_production(
            key_inputs,
            key_targets,
            key_weight,
            key_fit_sqrt,
        ),
        "fit_routing": fit_metrics.as_dict(),
        "fit_check_routing": fit_check_metrics.as_dict(),
        "checks": checks,
        "policy": policy_value.as_dict(),
        "limits": limits_value.as_dict(),
        "production_case_limits": {
            "maximum_fit_cases": runtime.maximum_fit_cases,
            "maximum_fit_check_cases": runtime.maximum_fit_check_cases,
            "maximum_aggregate_query_rows": runtime.maximum_aggregate_query_rows,
            "maximum_aggregate_key_rows": runtime.maximum_aggregate_key_rows,
            "maximum_aggregate_evidence_bytes": (
                runtime.maximum_aggregate_evidence_bytes
            ),
        },
        "production_resource_usage": resource_usage,
        "ridge": ridge_value,
        "fit_case_count": len(normalized_fit),
        "fit_check_case_count": len(normalized_fit_check),
        "fit_evidence_sha256": _canonical_sha256(
            [value.evidence_sha256 for value in normalized_fit]
        ),
        "fit_check_evidence_sha256": _canonical_sha256(
            [value.evidence_sha256 for value in normalized_fit_check]
        ),
        "factor_sufficient_state_count": len(sufficient_states),
        "factor_sufficient_state_bundle_sha256": _canonical_sha256(
            [value.as_digest_dict() for value in sufficient_states]
        ),
        "provenance": provenance_report,
        "source_evidence_sha256_before": source_before,
        "source_evidence_sha256_after": source_after,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
        "analytical_solvers": {
            "score_factorization": "bounded_row_weighted_truncated_svd",
            "factor_orientation": (
                "weighted_selected_pair_exact_anchor_or_orthogonal_procrustes"
            ),
            "projection_target": (
                "exact_radius_inverse_rope_target_rmsnorm_direction_"
                "raw_linear_anchor_scale"
            ),
            "projection_fit": "positive_row_weighted_ridge_normal_equations",
            "frequency_selection_uses_fit_check": False,
            "candidate_fit_uses_fit_check": False,
        },
        "operation_audit": operation_audit,
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    if not body["accepted"]:
        raise Lfm2CpcfAttentionDimensionValidationError(
            "Compiled attention dimension fails frozen FIT-CHECK routing gates.",
            report,
        )
    return Lfm2CompiledCpcfAttentionDimensionFold(
        topology=topology,
        query_groups=query_group_values,
        key_value_groups=key_group_values,
        selected_source_rope_frequencies=selected,
        query_weight=query_weight,
        key_weight=key_weight,
        query_input_width=runtime.target_hidden_width,
        key_value_input_width=runtime.target_hidden_width,
        fit_metrics=fit_metrics,
        validation_metrics=fit_check_metrics,
        report=report,
        runtime=runtime,
        factor_sufficient_states=tuple(sufficient_states),
    )


def compile_lfm2_cpcf_attention_dimension_fold(
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    fit_cases: Sequence[
        Lfm2CpcfAttentionDimensionCase | Lfm2CpcfAttentionDimensionProductionCase
    ],
    validation_cases: Sequence[Lfm2CpcfAttentionDimensionCase] = (),
    query_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    key_value_groups: Sequence[Lfm2CpcfBarycentricHeadGroup],
    policy: Lfm2CpcfAttentionDimensionPolicy | None = None,
    limits: Lfm2CpcfAttentionDimensionLimits | None = None,
    ridge: float = 1e-8,
    runtime: Lfm2CpcfAttentionDimensionRuntime | None = None,
    fit_check_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase] | None = None,
) -> Lfm2CompiledCpcfAttentionDimensionFold:
    """Compile a smaller target-head Q/K route without optimization.

    Calls without ``runtime`` dispatch byte-for-byte through the historical
    primitive.  Production calls provide exact target templates and a
    disjoint ``fit_check_cases`` partition derived from the upstream FIT
    receipt; macrocell selection remains outside this compiler.
    """

    if not isinstance(topology, Lfm2CpcfAttentionDimensionTopology):
        raise TypeError("topology must be Lfm2CpcfAttentionDimensionTopology.")
    if runtime is None:
        if fit_check_cases is not None:
            raise Lfm2CpcfAttentionDimensionFoldError(
                "fit_check_cases requires a production runtime contract."
            )
        return _compile_lfm2_cpcf_attention_dimension_fold_legacy(
            topology=topology,
            fit_cases=fit_cases,
            validation_cases=validation_cases,
            query_groups=query_groups,
            key_value_groups=key_value_groups,
            policy=policy,
            limits=limits,
            ridge=ridge,
        )
    if not isinstance(runtime, Lfm2CpcfAttentionDimensionRuntime):
        raise TypeError("runtime must be Lfm2CpcfAttentionDimensionRuntime or None.")
    if tuple(validation_cases):
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production compilation does not consume validation or macrocell "
            "selection cases; use fit_check_cases from the upstream FIT split."
        )
    if fit_check_cases is None:
        raise Lfm2CpcfAttentionDimensionFoldError(
            "Production compilation requires disjoint fit_check_cases."
        )
    return _compile_lfm2_cpcf_attention_dimension_fold_production(
        topology=topology,
        fit_cases=fit_cases,  # type: ignore[arg-type]
        fit_check_cases=fit_check_cases,
        query_groups=query_groups,
        key_value_groups=key_value_groups,
        runtime=runtime,
        policy=policy,
        limits=limits,
        ridge=ridge,
    )


__all__ = [
    "Lfm2CompiledCpcfAttentionDimensionFold",
    "Lfm2CpcfAttentionCaseProvenance",
    "Lfm2CpcfAttentionDimensionCapacityError",
    "Lfm2CpcfAttentionDimensionCase",
    "Lfm2CpcfAttentionDimensionFoldError",
    "Lfm2CpcfAttentionDimensionLimits",
    "Lfm2CpcfAttentionDimensionPolicy",
    "Lfm2CpcfAttentionDimensionProductionCase",
    "Lfm2CpcfAttentionDimensionReplay",
    "Lfm2CpcfAttentionDimensionRuntime",
    "Lfm2CpcfAttentionDimensionTopology",
    "Lfm2CpcfAttentionDimensionValidationError",
    "Lfm2CpcfAttentionFactorSufficientState",
    "Lfm2CpcfAttentionRoutingMetrics",
    "Lfm2CpcfBarycentricHeadGroup",
    "build_lfm2_cpcf_attention_dimension_production_case",
    "compile_lfm2_cpcf_attention_dimension_fold",
    "invert_lfm2_rope",
    "lfm2_projection_target_from_runtime",
    "lfm2_rmsnorm_preimage",
    "lfm2_runtime_rms_norm",
]
