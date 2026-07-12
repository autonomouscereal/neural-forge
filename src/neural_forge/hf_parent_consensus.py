"""Deterministic activation consensus for aligned cross-family parents.

Every parent must already be transported into the same row/feature space.  The
weighted-mean mode preserves explicit parent weights exactly.  The robust mode
uses a deterministic weighted-median anchor followed by bounded Huber IRLS on
each row.  Both modes use non-negative normalized coefficients, so every
consensus coordinate remains inside the corresponding parent coordinate hull.

This module never constructs an optimizer, invokes backward, or mutates parent
tensors.  Consensus tensors and diagnostics are detached from autograd.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


ConsensusMethod = Literal["weighted_mean", "robust"]


class ParentConsensusError(ValueError):
    """Base class for fail-closed parent-consensus errors."""


class ParentConsensusShapeError(ParentConsensusError):
    """Parent rows, labels, or an algorithm control has an invalid shape."""


class ParentConsensusFiniteError(ParentConsensusError):
    """An input, intermediate, or consensus result is non-finite."""


class ParentConsensusWeightError(ParentConsensusError):
    """Explicit parent weights are malformed or non-positive."""


@dataclass(frozen=True, slots=True)
class ParentContributionReport:
    """Contribution and final disagreement evidence for one parent."""

    parent_index: int
    parent_name: str
    requested_weight: float
    normalized_prior_weight: float
    mean_contribution_fraction: float
    minimum_contribution_fraction: float
    maximum_contribution_fraction: float
    disagreement_rms: float
    mean_row_disagreement_rms: float
    maximum_row_disagreement_rms: float
    robust_multiplier_mean: float
    robust_multiplier_minimum: float
    robust_multiplier_maximum: float
    suppressed_row_fraction: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence for experiment reports."""

        return {
            "parent_index": self.parent_index,
            "parent_name": self.parent_name,
            "requested_weight": self.requested_weight,
            "normalized_prior_weight": self.normalized_prior_weight,
            "mean_contribution_fraction": self.mean_contribution_fraction,
            "minimum_contribution_fraction": self.minimum_contribution_fraction,
            "maximum_contribution_fraction": self.maximum_contribution_fraction,
            "disagreement_rms": self.disagreement_rms,
            "mean_row_disagreement_rms": self.mean_row_disagreement_rms,
            "maximum_row_disagreement_rms": self.maximum_row_disagreement_rms,
            "robust_multiplier_mean": self.robust_multiplier_mean,
            "robust_multiplier_minimum": self.robust_multiplier_minimum,
            "robust_multiplier_maximum": self.robust_multiplier_maximum,
            "suppressed_row_fraction": self.suppressed_row_fraction,
        }


@dataclass(frozen=True, slots=True)
class ParentConsensusResult:
    """Fused rows, row-wise weights, and auditable consensus diagnostics."""

    consensus: Any
    effective_weights: Any
    row_disagreement_rms: Any
    parent_reports: tuple[ParentContributionReport, ...]
    method: ConsensusMethod
    parent_count: int
    row_count: int
    feature_width: int
    iterations: int
    converged: bool
    maximum_relative_update: float
    robust_cutoff: float | None
    scale_floor: float | None
    deterministic: bool = True
    no_training: bool = True
    convex_hull_bounded: bool = True
    bounded_influence: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return scalar, JSON-compatible diagnostics without large tensors."""

        return {
            "method": self.method,
            "parent_count": self.parent_count,
            "row_count": self.row_count,
            "feature_width": self.feature_width,
            "iterations": self.iterations,
            "converged": self.converged,
            "maximum_relative_update": self.maximum_relative_update,
            "robust_cutoff": self.robust_cutoff,
            "scale_floor": self.scale_floor,
            "deterministic": self.deterministic,
            "no_training": self.no_training,
            "convex_hull_bounded": self.convex_hull_bounded,
            "bounded_influence": self.bounded_influence,
            "parents": [report.as_dict() for report in self.parent_reports],
        }


def parent_activation_consensus(
    parent_rows: Sequence[Any],
    parent_weights: Sequence[float] | Any,
    *,
    method: ConsensusMethod = "robust",
    parent_names: Sequence[str] | None = None,
    robust_cutoff: float = 1.5,
    scale_floor: float = 1e-6,
    max_iterations: int = 32,
    tolerance: float = 1e-6,
) -> ParentConsensusResult:
    """Fuse aligned ``[rows, features]`` tensors without training.

    Robust consensus estimates a location independently for every aligned row.
    A parent's Huber multiplier is ``min(1, cutoff * scale / distance)``.
    Consequently, the residual magnitude entering that row's weighted update
    is capped by ``cutoff * scale`` before normalization.
    """

    torch = _require_torch()
    rows = _validate_parent_rows(torch, parent_rows)
    names = _validate_parent_names(parent_names, len(rows))
    requested_weights = _validate_parent_weights(torch, parent_weights, len(rows))
    method = _validate_method(method)
    robust_cutoff = _positive_float(robust_cutoff, "robust_cutoff")
    scale_floor = _positive_float(scale_floor, "scale_floor")
    max_iterations = _positive_int(max_iterations, "max_iterations")
    tolerance = _nonnegative_float(tolerance, "tolerance")

    output_dtype = rows[0].dtype
    for value in rows[1:]:
        output_dtype = torch.promote_types(output_dtype, value.dtype)
    work_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    device = rows[0].device

    with torch.no_grad():
        stacked = torch.stack(
            [value.detach().to(dtype=work_dtype) for value in rows],
            dim=0,
        )
        weights = torch.tensor(
            requested_weights,
            dtype=work_dtype,
            device=device,
        )
        normalized_weights = weights / weights.sum()
        row_count = int(stacked.shape[1])

        if method == "weighted_mean":
            effective_weights = normalized_weights[:, None].expand(
                -1, row_count
            ).clone()
            consensus = torch.einsum("pr,prf->rf", effective_weights, stacked)
            multipliers = torch.ones_like(effective_weights)
            iterations = 1
            converged = True
            maximum_relative_update = 0.0
            result_cutoff = None
            result_scale_floor = None
        else:
            (
                consensus,
                effective_weights,
                multipliers,
                iterations,
                converged,
                maximum_relative_update,
            ) = _robust_consensus(
                torch,
                stacked,
                normalized_weights,
                robust_cutoff=robust_cutoff,
                scale_floor=scale_floor,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            result_cutoff = robust_cutoff
            result_scale_floor = scale_floor

        minimum = stacked.amin(dim=0)
        maximum = stacked.amax(dim=0)
        consensus = torch.maximum(torch.minimum(consensus, maximum), minimum)
        _require_finite(torch, consensus, "consensus")
        _require_finite(torch, effective_weights, "effective_weights")
        _require_finite(torch, multipliers, "robust multipliers")

        final_distances = _row_rms(torch, stacked - consensus.unsqueeze(0))
        row_disagreement_rms = torch.sqrt(
            torch.sum(
                normalized_weights[:, None] * final_distances.square(),
                dim=0,
            )
        )
        _require_finite(torch, row_disagreement_rms, "row disagreement")
        reports = _build_parent_reports(
            torch,
            stacked,
            consensus,
            final_distances,
            effective_weights,
            multipliers,
            requested_weights,
            normalized_weights,
            names,
        )

        consensus = consensus.to(dtype=output_dtype).detach().contiguous()
        _require_finite(torch, consensus, "cast consensus")
        return ParentConsensusResult(
            consensus=consensus,
            effective_weights=effective_weights.detach().contiguous(),
            row_disagreement_rms=row_disagreement_rms.detach().contiguous(),
            parent_reports=reports,
            method=method,
            parent_count=len(rows),
            row_count=int(consensus.shape[0]),
            feature_width=int(consensus.shape[1]),
            iterations=iterations,
            converged=converged,
            maximum_relative_update=maximum_relative_update,
            robust_cutoff=result_cutoff,
            scale_floor=result_scale_floor,
            bounded_influence=method == "robust",
        )


def weighted_mean_parent_consensus(
    parent_rows: Sequence[Any],
    parent_weights: Sequence[float] | Any,
    *,
    parent_names: Sequence[str] | None = None,
) -> ParentConsensusResult:
    """Convenience wrapper for exact explicit-weight averaging."""

    return parent_activation_consensus(
        parent_rows,
        parent_weights,
        method="weighted_mean",
        parent_names=parent_names,
    )


def robust_parent_consensus(
    parent_rows: Sequence[Any],
    parent_weights: Sequence[float] | Any,
    *,
    parent_names: Sequence[str] | None = None,
    robust_cutoff: float = 1.5,
    scale_floor: float = 1e-6,
    max_iterations: int = 32,
    tolerance: float = 1e-6,
) -> ParentConsensusResult:
    """Convenience wrapper for bounded disagreement-aware consensus."""

    return parent_activation_consensus(
        parent_rows,
        parent_weights,
        method="robust",
        parent_names=parent_names,
        robust_cutoff=robust_cutoff,
        scale_floor=scale_floor,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _robust_consensus(
    torch,
    stacked,
    normalized_weights,
    *,
    robust_cutoff: float,
    scale_floor: float,
    max_iterations: int,
    tolerance: float,
):
    anchor = _weighted_median(torch, stacked, normalized_weights)
    anchor_distances = _row_rms(torch, stacked - anchor.unsqueeze(0))
    median_distance = _weighted_median(
        torch,
        anchor_distances,
        normalized_weights,
    )
    anchor_magnitude = _row_rms(torch, anchor)
    scale = torch.maximum(
        median_distance,
        float(scale_floor)
        * torch.maximum(anchor_magnitude, torch.ones_like(anchor_magnitude)),
    )
    _require_finite(torch, scale, "robust disagreement scale")

    center = anchor
    effective_weights = normalized_weights[:, None].expand(
        -1, int(stacked.shape[1])
    ).clone()
    multipliers = torch.ones_like(effective_weights)
    converged = False
    maximum_relative_update = math.inf

    for iteration in range(1, max_iterations + 1):
        distances = _row_rms(torch, stacked - center.unsqueeze(0))
        threshold = float(robust_cutoff) * scale
        multipliers = torch.minimum(
            torch.ones_like(distances),
            threshold.unsqueeze(0) / distances.clamp_min(torch.finfo(distances.dtype).tiny),
        )
        raw_weights = normalized_weights[:, None] * multipliers
        denominators = raw_weights.sum(dim=0)
        if bool((denominators <= 0.0).any()):
            raise ParentConsensusWeightError(
                "Robust weighting removed every parent from at least one row."
            )
        effective_weights = raw_weights / denominators.unsqueeze(0)
        updated = torch.einsum("pr,prf->rf", effective_weights, stacked)
        minimum = stacked.amin(dim=0)
        maximum = stacked.amax(dim=0)
        updated = torch.maximum(torch.minimum(updated, maximum), minimum)
        update_rms = _row_rms(torch, updated - center)
        center_rms = _row_rms(torch, center)
        relative_update = update_rms / torch.maximum(
            center_rms,
            torch.ones_like(center_rms),
        )
        _require_finite(torch, updated, "robust consensus update")
        _require_finite(torch, relative_update, "robust relative update")
        maximum_relative_update = float(relative_update.max().item())
        center = updated
        if maximum_relative_update <= tolerance:
            converged = True
            break

    return (
        center,
        effective_weights,
        multipliers,
        iteration,
        converged,
        maximum_relative_update,
    )


def _weighted_median(torch, values, normalized_weights):
    sorted_values, order = torch.sort(values, dim=0, stable=True)
    weight_shape = (int(normalized_weights.shape[0]),) + (1,) * (values.ndim - 1)
    expanded_weights = normalized_weights.reshape(weight_shape).expand_as(values)
    sorted_weights = torch.gather(expanded_weights, 0, order)
    cumulative = sorted_weights.cumsum(dim=0)
    median_index = (cumulative >= 0.5).to(torch.int64).argmax(dim=0)
    index = median_index.unsqueeze(0)
    lower = torch.gather(sorted_values, 0, index).squeeze(0)
    cumulative_at_lower = torch.gather(cumulative, 0, index).squeeze(0)
    next_index = (median_index + 1).clamp_max(int(values.shape[0]) - 1)
    upper = torch.gather(sorted_values, 0, next_index.unsqueeze(0)).squeeze(0)
    tolerance = torch.finfo(values.dtype).eps * 8.0
    has_interval = (median_index < int(values.shape[0]) - 1) & torch.isclose(
        cumulative_at_lower,
        torch.full_like(cumulative_at_lower, 0.5),
        rtol=0.0,
        atol=tolerance,
    )
    return torch.where(has_interval, (lower + upper) * 0.5, lower)


def _build_parent_reports(
    torch,
    stacked,
    consensus,
    final_distances,
    effective_weights,
    multipliers,
    requested_weights,
    normalized_weights,
    names,
) -> tuple[ParentContributionReport, ...]:
    reports = []
    suppression_tolerance = torch.finfo(multipliers.dtype).eps * 8.0
    for index in range(int(stacked.shape[0])):
        contribution = effective_weights[index]
        multiplier = multipliers[index]
        disagreement = _tensor_rms(torch, stacked[index] - consensus)
        row_distances = final_distances[index]
        reports.append(
            ParentContributionReport(
                parent_index=index,
                parent_name=names[index],
                requested_weight=requested_weights[index],
                normalized_prior_weight=float(normalized_weights[index].item()),
                mean_contribution_fraction=float(contribution.mean().item()),
                minimum_contribution_fraction=float(contribution.min().item()),
                maximum_contribution_fraction=float(contribution.max().item()),
                disagreement_rms=float(disagreement.item()),
                mean_row_disagreement_rms=float(row_distances.mean().item()),
                maximum_row_disagreement_rms=float(row_distances.max().item()),
                robust_multiplier_mean=float(multiplier.mean().item()),
                robust_multiplier_minimum=float(multiplier.min().item()),
                robust_multiplier_maximum=float(multiplier.max().item()),
                suppressed_row_fraction=float(
                    (multiplier < 1.0 - suppression_tolerance)
                    .to(torch.float64)
                    .mean()
                    .item()
                ),
            )
        )
    return tuple(reports)


def _row_rms(torch, value):
    if value.ndim == 2:
        return torch.sqrt(torch.mean(value.square(), dim=-1))
    if value.ndim == 3:
        return torch.sqrt(torch.mean(value.square(), dim=-1))
    raise ParentConsensusShapeError("RMS input must have two or three dimensions.")


def _tensor_rms(torch, value):
    return torch.sqrt(torch.mean(value.square()))


def _validate_parent_rows(torch, parent_rows: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(parent_rows, (str, bytes)) or not isinstance(
        parent_rows, Sequence
    ):
        raise ParentConsensusShapeError("parent_rows must be a sequence of tensors.")
    rows = tuple(parent_rows)
    if len(rows) < 2:
        raise ParentConsensusShapeError("At least two parent row tensors are required.")

    shape = None
    device = None
    for index, value in enumerate(rows):
        name = f"parent_rows[{index}]"
        if not isinstance(value, torch.Tensor):
            raise ParentConsensusShapeError(f"{name} must be a torch.Tensor.")
        if value.layout != torch.strided or value.device.type == "meta":
            raise ParentConsensusShapeError(
                f"{name} must be a dense strided non-meta tensor."
            )
        if value.ndim != 2:
            raise ParentConsensusShapeError(
                f"{name} must have shape [rows, features]."
            )
        if any(int(size) <= 0 for size in value.shape):
            raise ParentConsensusShapeError(f"{name} cannot have an empty dimension.")
        if not torch.is_floating_point(value) or torch.is_complex(value):
            raise ParentConsensusShapeError(
                f"{name} must have a real floating-point dtype."
            )
        if shape is None:
            shape = tuple(value.shape)
            device = value.device
        elif tuple(value.shape) != shape:
            raise ParentConsensusShapeError(
                "All parent row tensors must have the same aligned shape."
            )
        elif value.device != device:
            raise ParentConsensusShapeError(
                "All parent row tensors must be on the same device."
            )
        if not bool(torch.isfinite(value.detach()).all()):
            raise ParentConsensusFiniteError(f"{name} contains NaN or infinity.")
    return rows


def _validate_parent_weights(torch, parent_weights: Any, count: int) -> tuple[float, ...]:
    if isinstance(parent_weights, torch.Tensor):
        if (
            parent_weights.layout != torch.strided
            or parent_weights.device.type == "meta"
            or parent_weights.ndim != 1
            or int(parent_weights.shape[0]) != count
            or not torch.is_floating_point(parent_weights)
            or torch.is_complex(parent_weights)
        ):
            raise ParentConsensusWeightError(
                f"parent_weights must be a real floating tensor with shape [{count}]."
            )
        values = tuple(
            float(value)
            for value in parent_weights.detach().to(device="cpu", dtype=torch.float64)
        )
    else:
        if isinstance(parent_weights, (str, bytes)) or not isinstance(
            parent_weights, Sequence
        ):
            raise ParentConsensusWeightError(
                "parent_weights must be an explicit sequence of positive weights."
            )
        if len(parent_weights) != count:
            raise ParentConsensusWeightError(
                f"parent_weights must contain exactly {count} values."
            )
        values = tuple(
            _positive_float(value, f"parent_weights[{index}]")
            for index, value in enumerate(parent_weights)
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ParentConsensusWeightError(
            "Every parent weight must be finite and greater than zero."
        )
    return values


def _validate_parent_names(
    parent_names: Sequence[str] | None,
    count: int,
) -> tuple[str, ...]:
    if parent_names is None:
        return tuple(f"parent_{index}" for index in range(count))
    if isinstance(parent_names, (str, bytes)) or not isinstance(
        parent_names, Sequence
    ):
        raise ParentConsensusShapeError("parent_names must be a sequence of strings.")
    if len(parent_names) != count:
        raise ParentConsensusShapeError(
            f"parent_names must contain exactly {count} values."
        )
    names = tuple(parent_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ParentConsensusShapeError("Every parent name must be a non-empty string.")
    if len(set(names)) != count:
        raise ParentConsensusShapeError("Parent names must be unique.")
    return names


def _validate_method(value: Any) -> ConsensusMethod:
    if value not in ("weighted_mean", "robust"):
        raise ParentConsensusShapeError(
            "method must be either 'weighted_mean' or 'robust'."
        )
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ParentConsensusWeightError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParentConsensusWeightError(
            f"{name} must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ParentConsensusWeightError(f"{name} must be finite and positive.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ParentConsensusShapeError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParentConsensusShapeError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ParentConsensusShapeError(f"{name} must be finite and non-negative.")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParentConsensusShapeError(f"{name} must be a positive integer.")
    return value


def _require_finite(torch, value, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ParentConsensusFiniteError(f"{name} contains NaN or infinity.")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for neural_forge.hf_parent_consensus."
        ) from exc
    return torch


__all__ = [
    "ConsensusMethod",
    "ParentConsensusError",
    "ParentConsensusFiniteError",
    "ParentConsensusResult",
    "ParentConsensusShapeError",
    "ParentConsensusWeightError",
    "ParentContributionReport",
    "parent_activation_consensus",
    "robust_parent_consensus",
    "weighted_mean_parent_consensus",
]
