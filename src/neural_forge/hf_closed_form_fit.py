"""Deterministic closed-form fitting for blank-target transport operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal


NormKind = Literal["layer_norm", "rms_norm"]


class ClosedFormFitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FitMetrics:
    rows: int
    output_width: int
    rmse: float
    target_rms: float
    relative_rmse: float
    mean_cosine_similarity: float
    finite: bool


@dataclass(frozen=True, slots=True)
class LinearFitResult:
    weight: Any
    bias: Any | None
    ridge: float
    train_metrics: FitMetrics
    validation_metrics: FitMetrics | None


@dataclass(frozen=True, slots=True)
class NormFitResult:
    kind: NormKind
    weight: Any
    bias: Any | None
    epsilon: float
    ridge: float
    train_metrics: FitMetrics
    validation_metrics: FitMetrics | None


@dataclass(frozen=True, slots=True)
class FitGate:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: FitMetrics


def fit_ridge_linear(
    input_rows: Any,
    target_rows: Any,
    *,
    ridge: float = 1e-6,
    fit_bias: bool = True,
    validation_input_rows: Any | None = None,
    validation_target_rows: Any | None = None,
) -> LinearFitResult:
    """Fit ``target = input @ weight.T + bias`` without autograd."""

    torch = _require_torch()
    with torch.no_grad():
        inputs, targets = _validated_pair(
            torch,
            input_rows,
            target_rows,
            "input_rows",
            "target_rows",
        )
        ridge = _nonnegative_float(ridge, "ridge")
        if not isinstance(fit_bias, bool):
            raise ClosedFormFitError("fit_bias must be boolean.")
        design = inputs
        if fit_bias:
            design = torch.cat(
                [inputs, torch.ones((inputs.shape[0], 1), dtype=torch.float64)],
                dim=1,
            )
        gram = design.T @ design
        regularizer = torch.eye(gram.shape[0], dtype=torch.float64) * ridge
        if fit_bias:
            regularizer[-1, -1] = 0.0
        right = design.T @ targets
        try:
            solution = torch.linalg.solve(gram + regularizer, right)
        except RuntimeError:
            solution = torch.linalg.lstsq(gram + regularizer, right).solution
        if not bool(torch.isfinite(solution).all()):
            raise ClosedFormFitError("Ridge solve produced non-finite coefficients.")
        if fit_bias:
            weight = solution[:-1].T.contiguous()
            bias = solution[-1].contiguous()
        else:
            weight = solution.T.contiguous()
            bias = None
        train_prediction = inputs @ weight.T
        if bias is not None:
            train_prediction = train_prediction + bias
        train_metrics = measure_fit(train_prediction, targets)

        validation_metrics = None
        if (validation_input_rows is None) != (validation_target_rows is None):
            raise ClosedFormFitError(
                "Validation input and target rows must be supplied together."
            )
        if validation_input_rows is not None:
            validation_inputs, validation_targets = _validated_pair(
                torch,
                validation_input_rows,
                validation_target_rows,
                "validation_input_rows",
                "validation_target_rows",
            )
            if validation_inputs.shape[1] != inputs.shape[1]:
                raise ClosedFormFitError(
                    "Validation input width does not match training input width."
                )
            if validation_targets.shape[1] != targets.shape[1]:
                raise ClosedFormFitError(
                    "Validation target width does not match training target width."
                )
            prediction = validation_inputs @ weight.T
            if bias is not None:
                prediction = prediction + bias
            validation_metrics = measure_fit(prediction, validation_targets)
        return LinearFitResult(
            weight=weight.detach(),
            bias=None if bias is None else bias.detach(),
            ridge=ridge,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )


def fit_diagonal_norm(
    input_rows: Any,
    target_rows: Any,
    *,
    kind: NormKind,
    epsilon: float,
    has_bias: bool,
    ridge: float = 1e-6,
    validation_input_rows: Any | None = None,
    validation_target_rows: Any | None = None,
) -> NormFitResult:
    """Fit only the legal per-channel affine parameters of LayerNorm/RMSNorm."""

    torch = _require_torch()
    with torch.no_grad():
        inputs, targets = _validated_pair(
            torch,
            input_rows,
            target_rows,
            "input_rows",
            "target_rows",
        )
        if inputs.shape[1] != targets.shape[1]:
            raise ClosedFormFitError(
                "A diagonal norm fit requires equal input and target widths."
            )
        if kind not in {"layer_norm", "rms_norm"}:
            raise ClosedFormFitError("kind must be layer_norm or rms_norm.")
        if not isinstance(has_bias, bool):
            raise ClosedFormFitError("has_bias must be boolean.")
        if kind == "rms_norm" and has_bias:
            raise ClosedFormFitError("RMSNorm does not support a fitted bias.")
        epsilon = _positive_float(epsilon, "epsilon")
        ridge = _nonnegative_float(ridge, "ridge")
        normalized = normalize_rows(inputs, kind=kind, epsilon=epsilon)
        weight, bias = _fit_channel_affine(
            torch,
            normalized,
            targets,
            has_bias=has_bias,
            ridge=ridge,
        )
        train_prediction = normalized * weight
        if bias is not None:
            train_prediction = train_prediction + bias
        train_metrics = measure_fit(train_prediction, targets)

        validation_metrics = None
        if (validation_input_rows is None) != (validation_target_rows is None):
            raise ClosedFormFitError(
                "Validation input and target rows must be supplied together."
            )
        if validation_input_rows is not None:
            validation_inputs, validation_targets = _validated_pair(
                torch,
                validation_input_rows,
                validation_target_rows,
                "validation_input_rows",
                "validation_target_rows",
            )
            if validation_inputs.shape[1] != inputs.shape[1]:
                raise ClosedFormFitError(
                    "Validation norm width does not match training norm width."
                )
            normalized_validation = normalize_rows(
                validation_inputs,
                kind=kind,
                epsilon=epsilon,
            )
            prediction = normalized_validation * weight
            if bias is not None:
                prediction = prediction + bias
            validation_metrics = measure_fit(prediction, validation_targets)
        return NormFitResult(
            kind=kind,
            weight=weight.detach(),
            bias=None if bias is None else bias.detach(),
            epsilon=epsilon,
            ridge=ridge,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )


def normalize_rows(input_rows: Any, *, kind: NormKind, epsilon: float) -> Any:
    torch = _require_torch()
    with torch.no_grad():
        rows = _matrix(torch, input_rows, "input_rows")
        epsilon = _positive_float(epsilon, "epsilon")
        if kind == "layer_norm":
            centered = rows - rows.mean(dim=-1, keepdim=True)
            variance = centered.square().mean(dim=-1, keepdim=True)
            result = centered * torch.rsqrt(variance + epsilon)
        elif kind == "rms_norm":
            variance = rows.square().mean(dim=-1, keepdim=True)
            result = rows * torch.rsqrt(variance + epsilon)
        else:
            raise ClosedFormFitError("kind must be layer_norm or rms_norm.")
        if not bool(torch.isfinite(result).all()):
            raise ClosedFormFitError("Normalized rows are non-finite.")
        return result.detach().contiguous()


def measure_fit(prediction_rows: Any, target_rows: Any) -> FitMetrics:
    torch = _require_torch()
    with torch.no_grad():
        prediction, target = _validated_pair(
            torch,
            prediction_rows,
            target_rows,
            "prediction_rows",
            "target_rows",
        )
        error = prediction - target
        mse = error.square().mean()
        target_energy = target.square().mean()
        rmse = float(torch.sqrt(mse).item())
        target_rms = float(torch.sqrt(target_energy).item())
        relative_rmse = rmse / max(target_rms, torch.finfo(torch.float64).eps)
        numerator = (prediction * target).sum(dim=-1)
        denominator = torch.linalg.vector_norm(
            prediction, dim=-1
        ) * torch.linalg.vector_norm(
            target,
            dim=-1,
        )
        cosine = torch.where(
            denominator > 0.0,
            numerator / denominator,
            torch.where(
                (torch.linalg.vector_norm(prediction, dim=-1) == 0.0)
                & (torch.linalg.vector_norm(target, dim=-1) == 0.0),
                torch.ones_like(denominator),
                torch.zeros_like(denominator),
            ),
        )
        mean_cosine = float(cosine.mean().item())
        values = (rmse, target_rms, relative_rmse, mean_cosine)
        finite = all(math.isfinite(value) for value in values)
        if not finite:
            raise ClosedFormFitError("Fit metrics are non-finite.")
        return FitMetrics(
            rows=int(target.shape[0]),
            output_width=int(target.shape[1]),
            rmse=rmse,
            target_rms=target_rms,
            relative_rmse=relative_rmse,
            mean_cosine_similarity=mean_cosine,
            finite=True,
        )


def gate_fit(
    metrics: FitMetrics,
    *,
    max_relative_rmse: float,
    min_mean_cosine_similarity: float,
) -> FitGate:
    if not isinstance(metrics, FitMetrics) or not metrics.finite:
        raise ClosedFormFitError("metrics must be finite FitMetrics.")
    max_relative_rmse = _nonnegative_float(
        max_relative_rmse,
        "max_relative_rmse",
    )
    min_cosine = float(min_mean_cosine_similarity)
    if not math.isfinite(min_cosine) or not -1.0 <= min_cosine <= 1.0:
        raise ClosedFormFitError(
            "min_mean_cosine_similarity must be finite and in [-1, 1]."
        )
    reasons = []
    if metrics.relative_rmse > max_relative_rmse:
        reasons.append(
            f"relative_rmse {metrics.relative_rmse:.6g} exceeds {max_relative_rmse:.6g}"
        )
    if metrics.mean_cosine_similarity < min_cosine:
        reasons.append(
            f"mean_cosine_similarity {metrics.mean_cosine_similarity:.6g} is below "
            f"{min_cosine:.6g}"
        )
    return FitGate(accepted=not reasons, reasons=tuple(reasons), metrics=metrics)


def _fit_channel_affine(
    torch: Any,
    normalized: Any,
    targets: Any,
    *,
    has_bias: bool,
    ridge: float,
) -> tuple[Any, Any | None]:
    xx = normalized.square().sum(dim=0) + ridge
    xy = (normalized * targets).sum(dim=0)
    if not has_bias:
        weight = torch.where(xx > 0.0, xy / xx, torch.zeros_like(xy))
        return weight.contiguous(), None

    x_sum = normalized.sum(dim=0)
    y_sum = targets.sum(dim=0)
    count = float(normalized.shape[0])
    determinant = xx * count - x_sum.square()
    if bool((torch.abs(determinant) <= torch.finfo(torch.float64).eps).any()):
        raise ClosedFormFitError("LayerNorm affine solve is rank deficient.")
    weight = (xy * count - x_sum * y_sum) / determinant
    bias = (xx * y_sum - x_sum * xy) / determinant
    if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all()):
        raise ClosedFormFitError("Norm affine solve produced non-finite values.")
    return weight.contiguous(), bias.contiguous()


def _validated_pair(
    torch: Any,
    left: Any,
    right: Any,
    left_name: str,
    right_name: str,
) -> tuple[Any, Any]:
    left_matrix = _matrix(torch, left, left_name)
    right_matrix = _matrix(torch, right, right_name)
    if left_matrix.shape[0] != right_matrix.shape[0]:
        raise ClosedFormFitError(
            f"{left_name} and {right_name} must have the same row count."
        )
    return left_matrix, right_matrix


def _matrix(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise ClosedFormFitError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.ndim != 2:
        raise ClosedFormFitError(f"{name} must be a dense two-dimensional tensor.")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ClosedFormFitError(f"{name} cannot have an empty dimension.")
    if not torch.is_floating_point(value):
        raise ClosedFormFitError(f"{name} must have a floating-point dtype.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ClosedFormFitError(f"{name} contains NaN or infinity.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ClosedFormFitError(f"{name} must be a finite non-negative number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ClosedFormFitError(
            f"{name} must be a finite non-negative number."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ClosedFormFitError(f"{name} must be a finite non-negative number.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result == 0.0:
        raise ClosedFormFitError(f"{name} must be greater than zero.")
    return result


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for closed-form fitting.") from exc
    return torch


__all__ = [
    "ClosedFormFitError",
    "FitGate",
    "FitMetrics",
    "LinearFitResult",
    "NormFitResult",
    "fit_diagonal_norm",
    "fit_ridge_linear",
    "gate_fit",
    "measure_fit",
    "normalize_rows",
]
