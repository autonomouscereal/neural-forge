"""Zero-gradient, model-visible fits for residual-attention folding.

Raw Q/K/V reconstruction is not the operator consumed by attention.  Keys are
observed through query-key scores modulo row-constant softmax shifts, values are
observed through the attention-weighted readout, and O writes that readout into
the residual stream.  This module provides deterministic diagnostics and
bounded ridge solves in those spaces.

The functions here do not trace a model or decide which paths to merge.  A
caller must collect causal, post-RoPE, cache-aware evidence and must evaluate a
tentative edit in closed loop.  These primitives only solve the fixed-evidence
subproblems; they never construct an optimizer, call ``backward()``, or add a
runtime sidecar.
"""

from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np


class ResidualAttentionFitError(ValueError):
    """Raised when model-visible evidence or a closed-form fit is malformed."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionFitError(f"{name} must be a positive integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionFitError(
            f"{name} must be a positive integer."
        ) from error
    if result <= 0:
        raise ResidualAttentionFitError(f"{name} must be a positive integer.")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ResidualAttentionFitError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ResidualAttentionFitError(f"{name} must be finite.") from error
    if not math.isfinite(result):
        raise ResidualAttentionFitError(f"{name} must be finite.")
    return result


def _array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    if value is None:
        raise ResidualAttentionFitError(f"{name} is required.")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            value = numpy_method()
    try:
        result = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ResidualAttentionFitError(
            f"{name} must be a finite numeric tensor."
        ) from error
    if result.size == 0 or (ndim is not None and result.ndim != ndim):
        suffix = "" if ndim is None else f" with rank {ndim}"
        raise ResidualAttentionFitError(f"{name} must be non-empty{suffix}.")
    if not bool(np.isfinite(result).all()):
        raise ResidualAttentionFitError(f"{name} contains NaN or infinity.")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _relative_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    error = float(np.sqrt(np.mean(np.square(actual - predicted))))
    scale = float(np.sqrt(np.mean(np.square(actual))))
    return error / max(scale, np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class ModelVisibleError:
    """Mean and worst model-visible error over causal attention rows."""

    mean: float
    worst: float
    row_count: int
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "worst": self.worst,
            "row_count": self.row_count,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionFitLimits:
    """Explicit memory/shape bounds for one dense closed-form solve."""

    max_rows: int = 65_536
    max_input_dimensions: int = 4_096
    max_output_dimensions: int = 8_192
    max_gram_elements: int = 16_777_216
    max_rhs_elements: int = 33_554_432

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class ResidualAttentionRidgeReport:
    method: str
    row_count: int
    input_dimensions: int
    output_dimensions: int
    ridge: float
    weighted: bool
    condition_number: float
    before_relative_rmse: float
    after_relative_rmse: float
    relative_anchor_drift: float
    design_sha256: str
    target_sha256: str
    anchor_sha256: str
    fitted_sha256: str
    optimizer_steps: int = 0
    backward_calls: int = 0

    @property
    def zero_gradient(self) -> bool:
        return self.optimizer_steps == 0 and self.backward_calls == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        } | {
            "zero_gradient": self.zero_gradient,
            "training_used": False,
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionRidgeFit:
    weight: np.ndarray
    report: ResidualAttentionRidgeReport


@dataclass(frozen=True, slots=True)
class ResidualScaleFit:
    scale: float
    unclipped_scale: float
    minimum_scale: float
    maximum_scale: float
    before_relative_rmse: float
    after_relative_rmse: float
    optimizer_steps: int = 0
    backward_calls: int = 0

    @property
    def zero_gradient(self) -> bool:
        return self.optimizer_steps == 0 and self.backward_calls == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        } | {
            "zero_gradient": self.zero_gradient,
            "training_used": False,
        }


@dataclass(frozen=True, slots=True)
class ScoreSpaceMedoid:
    medoid_index: int
    total_costs: tuple[float, ...]
    pairwise_costs: np.ndarray
    method: str = "softmax-fisher-score-space-real-medoid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "medoid_index": self.medoid_index,
            "total_costs": list(self.total_costs),
            "pairwise_costs_sha256": _sha256(self.pairwise_costs),
            "method": self.method,
            "real_medoid": True,
            "tie_break": "lowest candidate index",
        }


def softmax_fisher_score_error(
    reference_scores: Any,
    folded_scores: Any,
    *,
    reference_probabilities: Any | None = None,
) -> ModelVisibleError:
    """Measure key distortion in the score space actually read by softmax.

    For score perturbation ``d``, each row uses
    ``sum(p*d**2) - sum(p*d)**2``.  The second term removes the score-constant
    direction that cannot change softmax probabilities.
    """

    reference = _array(reference_scores, "reference_scores")
    folded = _array(folded_scores, "folded_scores")
    if reference.shape != folded.shape or reference.shape[-1] <= 0:
        raise ResidualAttentionFitError(
            "reference_scores and folded_scores must share a non-empty key axis."
        )
    probabilities = (
        _softmax(reference)
        if reference_probabilities is None
        else _array(reference_probabilities, "reference_probabilities")
    )
    if probabilities.shape != reference.shape:
        raise ResidualAttentionFitError(
            "reference_probabilities must have the same shape as scores."
        )
    if bool((probabilities < 0.0).any()):
        raise ResidualAttentionFitError(
            "reference_probabilities cannot contain negative values."
        )
    mass = probabilities.sum(axis=-1, keepdims=True)
    if bool((mass <= 0.0).any()):
        raise ResidualAttentionFitError(
            "Every reference-probability row must have positive mass."
        )
    probabilities = probabilities / mass
    delta = folded - reference
    mean_delta = np.sum(probabilities * delta, axis=-1)
    row_error = np.sum(probabilities * np.square(delta), axis=-1) - np.square(
        mean_delta
    )
    row_error = np.maximum(row_error, 0.0)
    return ModelVisibleError(
        mean=float(np.mean(row_error)),
        worst=float(np.max(row_error)),
        row_count=int(row_error.size),
        method="softmax-fisher-score-error-modulo-constant-shifts",
    )


def attention_weighted_value_error(
    reference_values: Any,
    folded_values: Any,
    attention_probabilities: Any,
) -> ModelVisibleError:
    """Measure V distortion after the fixed causal attention readout."""

    reference = _array(reference_values, "reference_values")
    folded = _array(folded_values, "folded_values")
    attention = _array(attention_probabilities, "attention_probabilities")
    if reference.shape != folded.shape or reference.ndim < 2:
        raise ResidualAttentionFitError(
            "reference_values and folded_values must share [..., keys, width]."
        )
    if attention.ndim != reference.ndim or attention.shape[:-2] != reference.shape[:-2]:
        raise ResidualAttentionFitError(
            "attention_probabilities must share value leading dimensions."
        )
    if attention.shape[-1] != reference.shape[-2]:
        raise ResidualAttentionFitError(
            "Attention key count must equal the value sequence length."
        )
    if bool((attention < 0.0).any()):
        raise ResidualAttentionFitError(
            "attention_probabilities cannot contain negative values."
        )
    mass = attention.sum(axis=-1, keepdims=True)
    if bool((mass <= 0.0).any()):
        raise ResidualAttentionFitError("Every attention row must have positive mass.")
    normalized = attention / mass
    delta_readout = np.matmul(normalized, folded - reference)
    row_error = np.mean(np.square(delta_readout), axis=-1)
    return ModelVisibleError(
        mean=float(np.mean(row_error)),
        worst=float(np.max(row_error)),
        row_count=int(row_error.size),
        method="fixed-attention-value-readout-error",
    )


def fit_residual_branch_scale(
    candidate_delta: Any,
    target_delta: Any,
    *,
    ridge_to_one: float = 1e-6,
    minimum_scale: float = 0.0,
    maximum_scale: float = 4.0,
) -> ResidualScaleFit:
    """Fit one bounded scale for an attention delta before residual addition."""

    candidate = _array(candidate_delta, "candidate_delta")
    target = _array(target_delta, "target_delta")
    if candidate.shape != target.shape:
        raise ResidualAttentionFitError(
            "candidate_delta and target_delta must have identical shapes."
        )
    ridge = _finite(ridge_to_one, "ridge_to_one")
    minimum = _finite(minimum_scale, "minimum_scale")
    maximum = _finite(maximum_scale, "maximum_scale")
    if ridge < 0.0 or minimum > maximum:
        raise ResidualAttentionFitError(
            "ridge_to_one must be non-negative and scale bounds must be ordered."
        )
    x = candidate.reshape(-1)
    y = target.reshape(-1)
    denominator = float(np.dot(x, x)) + ridge
    if denominator <= 0.0:
        raise ResidualAttentionFitError(
            "Cannot fit a scale from a zero candidate with zero ridge."
        )
    raw = (float(np.dot(x, y)) + ridge) / denominator
    scale = min(max(raw, minimum), maximum)
    return ResidualScaleFit(
        scale=scale,
        unclipped_scale=raw,
        minimum_scale=minimum,
        maximum_scale=maximum,
        before_relative_rmse=_relative_rmse(target, candidate),
        after_relative_rmse=_relative_rmse(target, candidate * scale),
    )


def fit_attention_output_ridge(
    context_rows: Any,
    target_delta_rows: Any,
    anchor_weight: Any,
    *,
    ridge: float = 1e-3,
    row_weights: Any | None = None,
    limits: ResidualAttentionFitLimits | None = None,
    method: str = "causal-attention-context-to-residual-ridge",
) -> ResidualAttentionRidgeFit:
    """Fit O (or V) around a real anchor with one bounded ridge solve.

    The orientation follows Hugging Face linear weights: ``weight`` is
    ``[output, input]`` and predictions are ``context_rows @ weight.T``.
    """

    design = _array(context_rows, "context_rows", ndim=2)
    target = _array(target_delta_rows, "target_delta_rows", ndim=2)
    anchor = _array(anchor_weight, "anchor_weight", ndim=2)
    if design.shape[0] != target.shape[0]:
        raise ResidualAttentionFitError(
            "context_rows and target_delta_rows must have equal row counts."
        )
    if anchor.shape != (target.shape[1], design.shape[1]):
        raise ResidualAttentionFitError(
            "anchor_weight must have shape [target width, context width]."
        )
    regularization = _finite(ridge, "ridge")
    if regularization <= 0.0:
        raise ResidualAttentionFitError("ridge must be positive.")
    bounds = limits or ResidualAttentionFitLimits()
    rows, input_dimensions = map(int, design.shape)
    output_dimensions = int(target.shape[1])
    if (
        rows > bounds.max_rows
        or input_dimensions > bounds.max_input_dimensions
        or output_dimensions > bounds.max_output_dimensions
        or input_dimensions * input_dimensions > bounds.max_gram_elements
        or input_dimensions * output_dimensions > bounds.max_rhs_elements
    ):
        raise ResidualAttentionFitError(
            "Ridge evidence exceeds the declared row/dimension/workspace bounds."
        )
    weighted = row_weights is not None
    if row_weights is None:
        design_fit = design
        target_fit = target
    else:
        weights = _array(row_weights, "row_weights")
        if weights.shape not in {(rows,), (rows, 1)}:
            raise ResidualAttentionFitError(
                "row_weights must have shape [rows] or [rows, 1]."
            )
        weights = weights.reshape(rows)
        if bool((weights < 0.0).any()) or float(weights.sum()) <= 0.0:
            raise ResidualAttentionFitError(
                "row_weights must be non-negative with positive total mass."
            )
        square_root = np.sqrt(weights).reshape(rows, 1)
        design_fit = design * square_root
        target_fit = target * square_root
    gram = design_fit.T @ design_fit
    gram.flat[:: input_dimensions + 1] += regularization
    rhs = design_fit.T @ target_fit + regularization * anchor.T
    try:
        fitted_transposed = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as error:
        raise ResidualAttentionFitError(
            "Regularized attention ridge system could not be solved."
        ) from error
    fitted = fitted_transposed.T
    before = design @ anchor.T
    after = design @ fitted.T
    anchor_norm = float(np.linalg.norm(anchor))
    drift = float(np.linalg.norm(fitted - anchor)) / max(
        anchor_norm, np.finfo(np.float64).eps
    )
    report = ResidualAttentionRidgeReport(
        method=str(method),
        row_count=rows,
        input_dimensions=input_dimensions,
        output_dimensions=output_dimensions,
        ridge=regularization,
        weighted=weighted,
        condition_number=float(np.linalg.cond(gram)),
        before_relative_rmse=_relative_rmse(target, before),
        after_relative_rmse=_relative_rmse(target, after),
        relative_anchor_drift=drift,
        design_sha256=_sha256(design),
        target_sha256=_sha256(target),
        anchor_sha256=_sha256(anchor),
        fitted_sha256=_sha256(fitted),
    )
    return ResidualAttentionRidgeFit(_readonly(fitted), report)


def fit_fixed_attention_value_ridge(
    attention_mixed_input_rows: Any,
    target_context_rows: Any,
    anchor_value_weight: Any,
    **kwargs: Any,
) -> ResidualAttentionRidgeFit:
    """Fit V after Q/K fix the causal attention-mixed input rows."""

    kwargs.setdefault("method", "fixed-qk-attention-readout-value-ridge")
    return fit_attention_output_ridge(
        attention_mixed_input_rows,
        target_context_rows,
        anchor_value_weight,
        **kwargs,
    )


def select_score_space_qk_medoid(
    score_traces: Any,
    *,
    probability_traces: Any | None = None,
    max_candidates: int = 256,
    max_trace_values: int = 33_554_432,
) -> ScoreSpaceMedoid:
    """Select a real Q/K addressing anchor by softmax-Fisher distortion.

    ``score_traces`` has shape ``[candidates, ..., keys]``.  Cost ``i,j`` is
    the model-visible error when candidate ``j`` replaces source ``i``, using
    source ``i``'s attention distribution.  The lowest-index minimum is chosen.
    """

    scores = _array(score_traces, "score_traces")
    if scores.ndim < 2 or scores.shape[-1] <= 0:
        raise ResidualAttentionFitError(
            "score_traces must have shape [candidates, ..., keys]."
        )
    candidates = int(scores.shape[0])
    if candidates <= 0 or candidates > _positive_int(max_candidates, "max_candidates"):
        raise ResidualAttentionFitError("score_traces candidate count is out of bounds.")
    if scores.size > _positive_int(max_trace_values, "max_trace_values"):
        raise ResidualAttentionFitError("score_traces exceeds max_trace_values.")
    probabilities = (
        _softmax(scores)
        if probability_traces is None
        else _array(probability_traces, "probability_traces")
    )
    if probabilities.shape != scores.shape:
        raise ResidualAttentionFitError(
            "probability_traces must have the same shape as score_traces."
        )
    costs = np.zeros((candidates, candidates), dtype=np.float64)
    for source in range(candidates):
        for replacement in range(candidates):
            costs[source, replacement] = softmax_fisher_score_error(
                scores[source],
                scores[replacement],
                reference_probabilities=probabilities[source],
            ).mean
    totals = costs.sum(axis=0)
    minimum = float(np.min(totals))
    tied = np.flatnonzero(np.isclose(totals, minimum, rtol=1e-12, atol=1e-15))
    medoid = int(tied[0])
    readonly_costs = _readonly(costs)
    return ScoreSpaceMedoid(
        medoid_index=medoid,
        total_costs=tuple(float(value) for value in totals.tolist()),
        pairwise_costs=readonly_costs,
    )


__all__ = [
    "ModelVisibleError",
    "ResidualAttentionFitError",
    "ResidualAttentionFitLimits",
    "ResidualAttentionRidgeFit",
    "ResidualAttentionRidgeReport",
    "ResidualScaleFit",
    "ScoreSpaceMedoid",
    "attention_weighted_value_error",
    "fit_attention_output_ridge",
    "fit_fixed_attention_value_ridge",
    "fit_residual_branch_scale",
    "select_score_space_qk_medoid",
    "softmax_fisher_score_error",
]
