"""Closed-loop evidence and gates for atomic residual-attention folds.

An attention fold is not merely a similarity between Q, K, V, or O weight
slices.  During cached decoding it is a state transition::

    (r_t, K_{<t}, V_{<t})
        --Q/K/V, attention, O, residual add-->
    (r'_t, K_{<=t}, V_{<=t}, logits_t)

This module scores that complete transition.  It deliberately does not edit a
model, fit a projection, call ``backward()``, or recover weights.  Callers run
the parent and a tentatively folded model in closed loop, canonicalize their
head spaces if widths differ, and provide the resulting evidence here.  The
scorer then supplies a deterministic, inference-only gate for an atomic
Q/K/V/O surgery transaction.

The distinction between full-cache and cache-transition error matters.  A
candidate can match the first action while writing a bad K/V entry; that entry
is only consumed on a later decode step.  The transition metric exposes this
failure before a one-token score can certify the fold.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


class ResidualAttentionFoldError(ValueError):
    """Raised when residual-attention evidence or a gate is malformed."""


RESIDUAL_ATTENTION_METRICS = (
    "query_error",
    "key_error",
    "value_error",
    "output_context_error",
    "output_delta_error",
    "attention_score_geometry_error",
    "fixed_attention_value_error",
    "qkvo_bundle_error",
    "attention_js_divergence",
    "residual_transition_error",
    "residual_state_error",
    "key_cache_error",
    "value_cache_error",
    "cache_bundle_error",
    "cache_transition_error",
    "action_margin_damage",
    "topk_damage",
    "action_mismatch",
    "atomic_attention_damage",
)

_DEFAULT_WEIGHTS = {
    "attention_score_geometry_error": 0.10,
    "attention_js_divergence": 0.15,
    "fixed_attention_value_error": 0.05,
    "qkvo_bundle_error": 0.05,
    "residual_transition_error": 0.15,
    "residual_state_error": 0.05,
    "cache_bundle_error": 0.05,
    "cache_transition_error": 0.15,
    "action_margin_damage": 0.10,
    "topk_damage": 0.05,
    "action_mismatch": 0.10,
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionFoldError(f"{name} must be a positive integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionFoldError(
            f"{name} must be a positive integer."
        ) from error
    if result <= 0:
        raise ResidualAttentionFoldError(f"{name} must be a positive integer.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionFoldError(f"{name} must be a non-negative integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionFoldError(
            f"{name} must be a non-negative integer."
        ) from error
    if result < 0:
        raise ResidualAttentionFoldError(f"{name} must be a non-negative integer.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ResidualAttentionFoldError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ResidualAttentionFoldError(f"{name} must be finite.") from error
    if not math.isfinite(result):
        raise ResidualAttentionFoldError(f"{name} must be finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ResidualAttentionFoldError(f"{name} cannot be negative.")
    return result


def _probability(value: Any, name: str, *, inclusive_zero: bool = True) -> float:
    result = _finite_float(value, name)
    minimum = 0.0 if inclusive_zero else np.nextafter(0.0, 1.0)
    if result < minimum or result > 1.0:
        raise ResidualAttentionFoldError(f"{name} must be in [0, 1].")
    return result


def _array(value: Any, name: str) -> np.ndarray:
    """Copy one tensor-like value without retaining an autograd graph."""

    if value is None:
        raise ResidualAttentionFoldError(f"{name} is required.")
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
        raise ResidualAttentionFoldError(
            f"{name} must be a finite numeric tensor."
        ) from error
    if result.ndim == 0 or result.size == 0:
        raise ResidualAttentionFoldError(
            f"{name} must have at least one dimension and one value."
        )
    if not bool(np.isfinite(result).all()):
        raise ResidualAttentionFoldError(f"{name} contains NaN or infinity.")
    result.setflags(write=False)
    return result


def _token_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.int64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionFoldError(
            f"{name} must contain integer token IDs."
        ) from error
    if shape == () and result.shape == (1,):
        result = result.reshape(())
    if result.shape != shape:
        raise ResidualAttentionFoldError(
            f"{name} must have shape {shape}, got {result.shape}."
        )
    result.setflags(write=False)
    return result


def _mapping(
    values: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool,
    valid_metrics_only: bool,
) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        if valid_metrics_only and key not in RESIDUAL_ATTENTION_METRICS:
            raise ResidualAttentionFoldError(f"{name} contains unknown metric {key!r}.")
        value = _finite_float(raw_value, f"{name}[{key!r}]")
        if value < 0.0:
            raise ResidualAttentionFoldError(f"{name}[{key!r}] cannot be negative.")
        normalized[key] = value
    if not normalized and not allow_empty:
        raise ResidualAttentionFoldError(f"{name} cannot be empty.")
    return MappingProxyType(dict(sorted(normalized.items())))


def _same_shape(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if left.shape != right.shape:
        raise ResidualAttentionFoldError(
            f"{name} compares shapes {left.shape} and {right.shape}; "
            "canonicalize folded heads into a common comparison space first."
        )


@dataclass(frozen=True, slots=True)
class ResidualAttentionStepEvidence:
    """One cached-decode step for a complete residual-attention transition.

    ``attention_scores`` contains finite, valid-position pre-softmax scores.
    They are compared after subtracting each query row's mean because softmax
    is invariant to a row-wise additive shift.  ``fixed_attention_value_readout``
    is V read through the *reference* attention distribution; for folded
    evidence the caller must apply the parent's frozen distribution to folded
    V.  This separates content error from changed addressing.

    ``attention_output_input`` is the candidate's ordinary concatenated
    context entering O and ``attention_delta`` is O's residual-width output.
    Head-width-changing folds must map parent and candidate tensors into the
    same comparison chart before constructing this object.  Q and K remain
    addressing anchors; this module never averages them before softmax.
    """

    step_index: int
    pre_attention_residual: Any
    query: Any
    key: Any
    value: Any
    attention_scores: Any
    attention_probabilities: Any
    fixed_attention_value_readout: Any
    attention_output_input: Any
    attention_delta: Any
    post_attention_residual: Any
    key_cache: Any
    value_cache: Any
    action_logits: Any
    selected_token_ids: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_index", _nonnegative_int(self.step_index, "step_index")
        )
        tensor_fields = (
            "pre_attention_residual",
            "query",
            "key",
            "value",
            "attention_scores",
            "attention_probabilities",
            "fixed_attention_value_readout",
            "attention_output_input",
            "attention_delta",
            "post_attention_residual",
            "key_cache",
            "value_cache",
            "action_logits",
        )
        for name in tensor_fields:
            object.__setattr__(self, name, _array(getattr(self, name), name))

        pre = self.pre_attention_residual
        delta = self.attention_delta
        post = self.post_attention_residual
        if pre.shape != delta.shape or pre.shape != post.shape:
            raise ResidualAttentionFoldError(
                "pre_attention_residual, attention_delta, and "
                "post_attention_residual must have identical shapes."
            )
        if not bool(np.allclose(post, pre + delta, rtol=5e-4, atol=5e-5)):
            maximum = float(np.max(np.abs(post - (pre + delta))))
            raise ResidualAttentionFoldError(
                "post_attention_residual must equal pre_attention_residual + "
                f"attention_delta (maximum error {maximum:.6g})."
            )
        if self.key_cache.shape != self.value_cache.shape:
            raise ResidualAttentionFoldError(
                "key_cache and value_cache must have identical canonical shapes."
            )
        probabilities = self.attention_probabilities
        scores = self.attention_scores
        if scores.shape != probabilities.shape:
            raise ResidualAttentionFoldError(
                "attention_scores and attention_probabilities must have "
                "identical canonical shapes."
            )
        if probabilities.ndim < 1 or probabilities.shape[-1] <= 0:
            raise ResidualAttentionFoldError(
                "attention_probabilities must have a non-empty key axis."
            )
        if bool((probabilities < 0.0).any()):
            raise ResidualAttentionFoldError(
                "attention_probabilities cannot contain negative values."
            )
        row_sums = probabilities.sum(axis=-1)
        if bool((row_sums <= 0.0).any()):
            raise ResidualAttentionFoldError(
                "Every attention-probability row must have positive mass."
            )
        logits = self.action_logits
        if logits.shape[-1] < 2:
            raise ResidualAttentionFoldError(
                "action_logits needs at least two vocabulary entries."
            )
        selected = self.selected_token_ids
        if selected is None:
            selected = np.argmax(logits, axis=-1).astype(np.int64, copy=False)
            selected = np.array(selected, copy=True)
            selected.setflags(write=False)
        else:
            selected = _token_array(
                selected,
                logits.shape[:-1],
                "selected_token_ids",
            )
        if bool((selected < 0).any()) or bool((selected >= logits.shape[-1]).any()):
            raise ResidualAttentionFoldError(
                "selected_token_ids contains an out-of-vocabulary index."
            )
        object.__setattr__(self, "selected_token_ids", selected)


@dataclass(frozen=True, slots=True)
class ResidualAttentionRollout:
    """A single prompt's closed-loop cached-decode trajectory."""

    sequence_id: str
    steps: tuple[ResidualAttentionStepEvidence, ...]

    def __post_init__(self) -> None:
        sequence_id = str(self.sequence_id).strip()
        if not sequence_id:
            raise ResidualAttentionFoldError("sequence_id cannot be blank.")
        steps = tuple(self.steps)
        if not steps:
            raise ResidualAttentionFoldError(
                "A rollout must contain at least one step."
            )
        for expected, step in enumerate(steps):
            if not isinstance(step, ResidualAttentionStepEvidence):
                raise ResidualAttentionFoldError(
                    "steps must contain ResidualAttentionStepEvidence values."
                )
            if step.step_index != expected:
                raise ResidualAttentionFoldError(
                    "Rollout step indices must be contiguous and start at zero."
                )
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True, slots=True)
class ResidualAttentionRolloutPair:
    """Parent and tentatively folded rollouts for the same prompt."""

    reference: ResidualAttentionRollout
    folded: ResidualAttentionRollout

    def __post_init__(self) -> None:
        if self.reference.sequence_id != self.folded.sequence_id:
            raise ResidualAttentionFoldError(
                "Reference and folded rollouts must share a sequence_id."
            )
        if len(self.reference.steps) != len(self.folded.steps):
            raise ResidualAttentionFoldError(
                "Reference and folded rollouts must have the same horizon."
            )

    @property
    def sequence_id(self) -> str:
        return self.reference.sequence_id

    @property
    def horizon(self) -> int:
        return len(self.reference.steps)


@dataclass(frozen=True, slots=True)
class ResidualAttentionCandidateEvidence:
    """Measured evidence for one indivisible fold or interacting fold group."""

    candidate_ids: tuple[str, ...]
    rollouts: tuple[ResidualAttentionRolloutPair, ...]
    token_lane: str = "unspecified"

    def __post_init__(self) -> None:
        candidate_ids = tuple(str(value).strip() for value in self.candidate_ids)
        if not candidate_ids or any(not value for value in candidate_ids):
            raise ResidualAttentionFoldError(
                "candidate_ids must contain at least one non-blank ID."
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ResidualAttentionFoldError("candidate_ids must be unique.")
        rollouts = tuple(self.rollouts)
        if not rollouts:
            raise ResidualAttentionFoldError(
                "Candidate evidence must contain at least one rollout pair."
            )
        if any(
            not isinstance(value, ResidualAttentionRolloutPair) for value in rollouts
        ):
            raise ResidualAttentionFoldError(
                "rollouts must contain ResidualAttentionRolloutPair values."
            )
        sequence_ids = tuple(value.sequence_id for value in rollouts)
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ResidualAttentionFoldError("Rollout sequence IDs must be unique.")
        token_lane = str(self.token_lane).strip().lower()
        valid_lanes = {
            "unspecified",
            "forced-shared-token",
            "greedy-feedback-aligned-prefix",
        }
        if token_lane not in valid_lanes:
            raise ResidualAttentionFoldError(
                f"token_lane must be one of {sorted(valid_lanes)}."
            )
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "rollouts", rollouts)
        object.__setattr__(self, "token_lane", token_lane)

    @property
    def atomic_group(self) -> bool:
        return len(self.candidate_ids) > 1


@dataclass(frozen=True, slots=True)
class ResidualAttentionScoringConfig:
    """Deterministic closed-loop metric and risk aggregation controls."""

    quantile: float = 0.95
    top_k: int = 5
    cache_sequence_axis: int = -2
    epsilon: float = 1e-12
    worst_weight: float = 0.5
    metric_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_WEIGHTS)
    )

    def __post_init__(self) -> None:
        quantile = _probability(self.quantile, "quantile")
        top_k = _positive_int(self.top_k, "top_k")
        if isinstance(self.cache_sequence_axis, bool):
            raise ResidualAttentionFoldError("cache_sequence_axis must be an integer.")
        try:
            cache_axis = operator.index(self.cache_sequence_axis)
        except (TypeError, ValueError, OverflowError) as error:
            raise ResidualAttentionFoldError(
                "cache_sequence_axis must be an integer."
            ) from error
        epsilon = _finite_float(self.epsilon, "epsilon")
        if epsilon <= 0.0:
            raise ResidualAttentionFoldError("epsilon must be positive.")
        worst_weight = _probability(self.worst_weight, "worst_weight")
        weights = _mapping(
            self.metric_weights,
            "metric_weights",
            allow_empty=False,
            valid_metrics_only=True,
        )
        if sum(weights.values()) <= 0.0:
            raise ResidualAttentionFoldError(
                "metric_weights must contain positive total weight."
            )
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "cache_sequence_axis", cache_axis)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "worst_weight", worst_weight)
        object.__setattr__(self, "metric_weights", weights)


@dataclass(frozen=True, slots=True)
class ResidualAttentionStepDamage:
    sequence_id: str
    step_index: int
    metrics: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "step_index": self.step_index,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionMetricSummary:
    mean: float
    quantile: float
    worst: float
    first_step_worst: float
    final_step_worst: float
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "quantile": self.quantile,
            "worst": self.worst,
            "first_step_worst": self.first_step_worst,
            "final_step_worst": self.final_step_worst,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionCandidateDamage:
    candidate_ids: tuple[str, ...]
    sequence_count: int
    minimum_horizon: int
    maximum_horizon: int
    step_damages: tuple[ResidualAttentionStepDamage, ...]
    metrics: Mapping[str, ResidualAttentionMetricSummary]
    risk_score: float
    quantile: float
    worst_weight: float
    token_lane: str

    @property
    def atomic_group(self) -> bool:
        return len(self.candidate_ids) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "closed-loop-residual-attention-fold-v1",
            "candidate_ids": list(self.candidate_ids),
            "atomic_group": self.atomic_group,
            "sequence_count": self.sequence_count,
            "minimum_horizon": self.minimum_horizon,
            "maximum_horizon": self.maximum_horizon,
            "step_damages": [value.as_dict() for value in self.step_damages],
            "metrics": {
                name: summary.as_dict() for name, summary in self.metrics.items()
            },
            "risk_score": self.risk_score,
            "quantile": self.quantile,
            "worst_weight": self.worst_weight,
            "qkvo_residual_cache_atomic": True,
            "cache_closed_loop": True,
            "token_lane": self.token_lane,
            "autoregressive_closed_loop": (
                self.token_lane == "greedy-feedback-aligned-prefix"
            ),
            "closed_loop": self.token_lane == "greedy-feedback-aligned-prefix",
            "kv_cache_transition_scored": True,
            "gradient_free": True,
            "training_used": False,
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionGatePolicy:
    """Worst-case and quantile limits for promotion of a tentative fold."""

    quantile_limits: Mapping[str, float] = field(default_factory=dict)
    worst_limits: Mapping[str, float] = field(default_factory=dict)
    max_risk_score: float | None = None
    minimum_horizon: int = 2

    def __post_init__(self) -> None:
        quantile_limits = _mapping(
            self.quantile_limits,
            "quantile_limits",
            allow_empty=True,
            valid_metrics_only=True,
        )
        worst_limits = _mapping(
            self.worst_limits,
            "worst_limits",
            allow_empty=True,
            valid_metrics_only=True,
        )
        maximum = self.max_risk_score
        if maximum is not None:
            maximum = _finite_float(maximum, "max_risk_score")
            if maximum < 0.0:
                raise ResidualAttentionFoldError("max_risk_score cannot be negative.")
        horizon = _positive_int(self.minimum_horizon, "minimum_horizon")
        if not quantile_limits and not worst_limits and maximum is None:
            raise ResidualAttentionFoldError(
                "A residual-attention gate must declare at least one quantile, "
                "worst-case, or aggregate-risk limit."
            )
        object.__setattr__(self, "quantile_limits", quantile_limits)
        object.__setattr__(self, "worst_limits", worst_limits)
        object.__setattr__(self, "max_risk_score", maximum)
        object.__setattr__(self, "minimum_horizon", horizon)


@dataclass(frozen=True, slots=True)
class ResidualAttentionGateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    risk_score: float
    maximum_limit_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "risk_score": self.risk_score,
            "maximum_limit_ratio": self.maximum_limit_ratio,
        }


@dataclass(frozen=True, slots=True)
class RankedResidualAttentionCandidate:
    rank: int
    damage: ResidualAttentionCandidateDamage
    decision: ResidualAttentionGateDecision


@dataclass(frozen=True, slots=True)
class ResidualAttentionInteractionMetric:
    group_quantile: float
    independent_quantile_sum: float
    quantile_excess: float
    group_worst: float
    independent_worst_sum: float
    worst_excess: float

    def as_dict(self) -> dict[str, float]:
        return {
            "group_quantile": self.group_quantile,
            "independent_quantile_sum": self.independent_quantile_sum,
            "quantile_excess": self.quantile_excess,
            "group_worst": self.group_worst,
            "independent_worst_sum": self.independent_worst_sum,
            "worst_excess": self.worst_excess,
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionInteractionReport:
    candidate_ids: tuple[str, ...]
    metrics: Mapping[str, ResidualAttentionInteractionMetric]
    group_risk_score: float
    independent_risk_sum: float
    risk_excess: float

    @property
    def destructive_synergy(self) -> bool:
        return self.risk_excess > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "measured-atomic-fold-interaction-v1",
            "candidate_ids": list(self.candidate_ids),
            "metrics": {name: value.as_dict() for name, value in self.metrics.items()},
            "group_risk_score": self.group_risk_score,
            "independent_risk_sum": self.independent_risk_sum,
            "risk_excess": self.risk_excess,
            "destructive_synergy": self.destructive_synergy,
            "measured_jointly": True,
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionBranchScaleFit:
    """Closed-form scalar compensation for the complete O residual branch."""

    scale: float
    prior_scale: float
    ridge: float
    effective_ridge: float
    row_count: int
    baseline_relative_l2: float
    fitted_relative_l2: float
    clipped_to_bounds: bool

    @property
    def improved(self) -> bool:
        return self.fitted_relative_l2 < self.baseline_relative_l2

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "closed-form-residual-attention-branch-scale-v1",
            "scale": self.scale,
            "prior_scale": self.prior_scale,
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "row_count": self.row_count,
            "baseline_relative_l2": self.baseline_relative_l2,
            "fitted_relative_l2": self.fitted_relative_l2,
            "improved": self.improved,
            "clipped_to_bounds": self.clipped_to_bounds,
            "gradient_free": True,
            "training_used": False,
        }


@dataclass(frozen=True, slots=True)
class FixedAttentionOutputRidgeFit:
    """Global fixed-attention V/O solve minimizing residual-output error."""

    output_matrix: np.ndarray
    ridge: float
    effective_ridge: float
    row_count: int
    input_width: int
    output_width: int
    prior_used: bool
    baseline_relative_l2: float
    fitted_relative_l2: float
    gram_condition_number: float

    @property
    def improved(self) -> bool:
        return self.fitted_relative_l2 < self.baseline_relative_l2

    def as_dict(self, *, include_matrix: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": "global-fixed-attention-vo-ridge-v1",
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "row_count": self.row_count,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "prior_used": self.prior_used,
            "baseline_relative_l2": self.baseline_relative_l2,
            "fitted_relative_l2": self.fitted_relative_l2,
            "improved": self.improved,
            "gram_condition_number": self.gram_condition_number,
            "fixed_reference_attention": True,
            "qk_edited": False,
            "gradient_free": True,
            "training_used": False,
        }
        if include_matrix:
            result["output_matrix"] = self.output_matrix.tolist()
        return result


def _symmetric_relative_l2(
    reference: np.ndarray, folded: np.ndarray, epsilon: float, name: str
) -> float:
    _same_shape(reference, folded, name)
    reference_flat = reference.reshape(-1)
    folded_flat = folded.reshape(-1)
    numerator = 2.0 * float(np.linalg.norm(reference_flat - folded_flat))
    denominator = (
        float(np.linalg.norm(reference_flat))
        + float(np.linalg.norm(folded_flat))
        + epsilon
    )
    return numerator / denominator


def _attention_js_divergence(
    reference: np.ndarray, folded: np.ndarray, epsilon: float
) -> float:
    _same_shape(reference, folded, "attention distributions")
    reference_rows = reference.reshape(-1, reference.shape[-1])
    folded_rows = folded.reshape(-1, folded.shape[-1])
    reference_rows = reference_rows / reference_rows.sum(axis=-1, keepdims=True)
    folded_rows = folded_rows / folded_rows.sum(axis=-1, keepdims=True)
    midpoint = 0.5 * (reference_rows + folded_rows)
    reference_terms = np.where(
        reference_rows > 0.0,
        reference_rows
        * (np.log(reference_rows + epsilon) - np.log(midpoint + epsilon)),
        0.0,
    )
    folded_terms = np.where(
        folded_rows > 0.0,
        folded_rows * (np.log(folded_rows + epsilon) - np.log(midpoint + epsilon)),
        0.0,
    )
    divergences = 0.5 * (reference_terms.sum(axis=-1) + folded_terms.sum(axis=-1))
    # Dividing by log(2) makes JSD finite in [0, 1].  Use the worst row so a
    # single destroyed head/query cannot be hidden by an average.
    return float(np.max(divergences) / math.log(2.0))


def _attention_score_geometry_error(
    reference: np.ndarray,
    folded: np.ndarray,
    epsilon: float,
) -> float:
    """Compare addressing logits modulo softmax-invisible row translations."""

    _same_shape(reference, folded, "attention score geometry")
    reference_rows = reference.reshape(-1, reference.shape[-1])
    folded_rows = folded.reshape(-1, folded.shape[-1])
    reference_centered = reference_rows - reference_rows.mean(axis=-1, keepdims=True)
    folded_centered = folded_rows - folded_rows.mean(axis=-1, keepdims=True)
    row_errors = []
    for reference_row, folded_row in zip(reference_centered, folded_centered):
        row_errors.append(
            _symmetric_relative_l2(
                reference_row,
                folded_row,
                epsilon,
                "centered attention score row",
            )
        )
    return max(row_errors)


def _normalize_axis(axis: int, ndim: int, name: str) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ResidualAttentionFoldError(
            f"cache_sequence_axis={axis} is invalid for {name} with {ndim} axes."
        )
    return normalized


def _cache_increment(
    current: np.ndarray,
    previous: np.ndarray | None,
    axis: int,
    name: str,
) -> np.ndarray:
    sequence_axis = _normalize_axis(axis, current.ndim, name)
    if previous is None:
        return current
    if current.ndim != previous.ndim:
        raise ResidualAttentionFoldError(
            f"{name} changed rank between cached decode steps."
        )
    previous_axis = _normalize_axis(axis, previous.ndim, name)
    for dimension, (current_size, previous_size) in enumerate(
        zip(current.shape, previous.shape)
    ):
        if dimension == sequence_axis:
            if current_size < previous_size:
                raise ResidualAttentionFoldError(
                    f"{name} shrank along its cache sequence axis."
                )
        elif current_size != previous_size:
            raise ResidualAttentionFoldError(
                f"{name} changed non-sequence shape between decode steps."
            )
    if previous_axis != sequence_axis:  # pragma: no cover - defensive invariant
        raise ResidualAttentionFoldError(f"{name} cache axis was inconsistent.")
    slices = [slice(None)] * current.ndim
    slices[sequence_axis] = slice(previous.shape[sequence_axis], None)
    increment = current[tuple(slices)]
    if increment.shape[sequence_axis] <= 0:
        raise ResidualAttentionFoldError(
            f"{name} did not append a cache entry at this decode step."
        )
    return increment


def _action_damage(
    reference: ResidualAttentionStepEvidence,
    folded: ResidualAttentionStepEvidence,
    *,
    top_k: int,
    epsilon: float,
) -> tuple[float, float, float]:
    reference_logits = reference.action_logits
    folded_logits = folded.action_logits
    _same_shape(reference_logits, folded_logits, "action logits")
    rows = int(np.prod(reference_logits.shape[:-1], dtype=np.int64))
    vocabulary = reference_logits.shape[-1]
    reference_rows = reference_logits.reshape(rows, vocabulary)
    folded_rows = folded_logits.reshape(rows, vocabulary)
    reference_actions = reference.selected_token_ids.reshape(rows)
    folded_actions = folded.selected_token_ids.reshape(rows)

    row_indices = np.arange(rows)
    reference_winner = reference_rows[row_indices, reference_actions]
    folded_reference_winner = folded_rows[row_indices, reference_actions]
    reference_competitors = reference_rows.copy()
    folded_competitors = folded_rows.copy()
    reference_competitors[row_indices, reference_actions] = -np.inf
    folded_competitors[row_indices, reference_actions] = -np.inf
    reference_margin = reference_winner - reference_competitors.max(axis=-1)
    folded_margin = folded_reference_winner - folded_competitors.max(axis=-1)
    margin_damage = np.maximum(0.0, reference_margin - folded_margin) / (
        np.abs(reference_margin) + epsilon
    )

    effective_top_k = min(top_k, vocabulary)
    reference_top = np.argsort(-reference_rows, axis=-1, kind="stable")[
        :, :effective_top_k
    ]
    folded_top = np.argsort(-folded_rows, axis=-1, kind="stable")[:, :effective_top_k]
    overlaps = np.array(
        [
            len(set(reference_top[index]).intersection(folded_top[index]))
            for index in range(rows)
        ],
        dtype=np.float64,
    )
    topk_damage = 1.0 - overlaps / effective_top_k
    mismatch = reference_actions != folded_actions
    return (
        float(np.max(margin_damage)),
        float(np.max(topk_damage)),
        float(np.mean(mismatch, dtype=np.float64)),
    )


def _score_step(
    reference: ResidualAttentionStepEvidence,
    folded: ResidualAttentionStepEvidence,
    previous_reference: ResidualAttentionStepEvidence | None,
    previous_folded: ResidualAttentionStepEvidence | None,
    config: ResidualAttentionScoringConfig,
) -> Mapping[str, float]:
    if reference.step_index != folded.step_index:
        raise ResidualAttentionFoldError(
            "Reference and folded evidence has mismatched step indices."
        )
    epsilon = config.epsilon
    query_error = _symmetric_relative_l2(
        reference.query, folded.query, epsilon, "query"
    )
    key_error = _symmetric_relative_l2(reference.key, folded.key, epsilon, "key")
    value_error = _symmetric_relative_l2(
        reference.value, folded.value, epsilon, "value"
    )
    context_error = _symmetric_relative_l2(
        reference.attention_output_input,
        folded.attention_output_input,
        epsilon,
        "attention output context",
    )
    delta_error = _symmetric_relative_l2(
        reference.attention_delta,
        folded.attention_delta,
        epsilon,
        "attention output delta",
    )
    score_geometry_error = _attention_score_geometry_error(
        reference.attention_scores,
        folded.attention_scores,
        epsilon,
    )
    fixed_attention_value_error = _symmetric_relative_l2(
        reference.fixed_attention_value_readout,
        folded.fixed_attention_value_readout,
        epsilon,
        "fixed-attention value readout",
    )
    # Raw Q/K/V distances remain useful diagnostics, but the atomic promotion
    # metric follows what attention actually observes: QK score geometry,
    # fixed-attention V readout, ordinary context, and the O residual update.
    # This intentionally never averages Q or K before softmax.
    qkvo_error = max(
        score_geometry_error,
        fixed_attention_value_error,
        context_error,
        delta_error,
    )
    attention_js = _attention_js_divergence(
        reference.attention_probabilities,
        folded.attention_probabilities,
        epsilon,
    )
    reference_update = (
        reference.post_attention_residual - reference.pre_attention_residual
    )
    folded_update = folded.post_attention_residual - folded.pre_attention_residual
    transition_error = _symmetric_relative_l2(
        reference_update,
        folded_update,
        epsilon,
        "residual attention update",
    )
    state_error = _symmetric_relative_l2(
        reference.post_attention_residual,
        folded.post_attention_residual,
        epsilon,
        "post-attention residual",
    )
    key_cache_error = _symmetric_relative_l2(
        reference.key_cache,
        folded.key_cache,
        epsilon,
        "key cache",
    )
    value_cache_error = _symmetric_relative_l2(
        reference.value_cache,
        folded.value_cache,
        epsilon,
        "value cache",
    )
    reference_key_increment = _cache_increment(
        reference.key_cache,
        None if previous_reference is None else previous_reference.key_cache,
        config.cache_sequence_axis,
        "reference key cache",
    )
    folded_key_increment = _cache_increment(
        folded.key_cache,
        None if previous_folded is None else previous_folded.key_cache,
        config.cache_sequence_axis,
        "folded key cache",
    )
    reference_value_increment = _cache_increment(
        reference.value_cache,
        None if previous_reference is None else previous_reference.value_cache,
        config.cache_sequence_axis,
        "reference value cache",
    )
    folded_value_increment = _cache_increment(
        folded.value_cache,
        None if previous_folded is None else previous_folded.value_cache,
        config.cache_sequence_axis,
        "folded value cache",
    )
    cache_transition_error = max(
        _symmetric_relative_l2(
            reference_key_increment,
            folded_key_increment,
            epsilon,
            "key cache increment",
        ),
        _symmetric_relative_l2(
            reference_value_increment,
            folded_value_increment,
            epsilon,
            "value cache increment",
        ),
    )
    margin_damage, topk_damage, mismatch = _action_damage(
        reference,
        folded,
        top_k=config.top_k,
        epsilon=epsilon,
    )
    cache_bundle_error = max(key_cache_error, value_cache_error)
    atomic_damage = max(
        qkvo_error,
        attention_js,
        transition_error,
        state_error,
        cache_bundle_error,
        cache_transition_error,
        margin_damage,
        topk_damage,
        mismatch,
    )
    values = {
        "query_error": query_error,
        "key_error": key_error,
        "value_error": value_error,
        "output_context_error": context_error,
        "output_delta_error": delta_error,
        "attention_score_geometry_error": score_geometry_error,
        "fixed_attention_value_error": fixed_attention_value_error,
        "qkvo_bundle_error": qkvo_error,
        "attention_js_divergence": attention_js,
        "residual_transition_error": transition_error,
        "residual_state_error": state_error,
        "key_cache_error": key_cache_error,
        "value_cache_error": value_cache_error,
        "cache_bundle_error": cache_bundle_error,
        "cache_transition_error": cache_transition_error,
        "action_margin_damage": margin_damage,
        "topk_damage": topk_damage,
        "action_mismatch": mismatch,
        "atomic_attention_damage": atomic_damage,
    }
    return MappingProxyType(values)


def score_residual_attention_candidate(
    evidence: ResidualAttentionCandidateEvidence,
    config: ResidualAttentionScoringConfig | None = None,
) -> ResidualAttentionCandidateDamage:
    """Score one individually measured fold or atomic interacting fold group."""

    if not isinstance(evidence, ResidualAttentionCandidateEvidence):
        raise ResidualAttentionFoldError(
            "evidence must be ResidualAttentionCandidateEvidence."
        )
    controls = config or ResidualAttentionScoringConfig()
    if not isinstance(controls, ResidualAttentionScoringConfig):
        raise ResidualAttentionFoldError(
            "config must be ResidualAttentionScoringConfig."
        )
    step_damages: list[ResidualAttentionStepDamage] = []
    horizons: list[int] = []
    for rollout_pair in evidence.rollouts:
        horizons.append(rollout_pair.horizon)
        previous_reference = None
        previous_folded = None
        for reference, folded in zip(
            rollout_pair.reference.steps, rollout_pair.folded.steps
        ):
            metrics = _score_step(
                reference,
                folded,
                previous_reference,
                previous_folded,
                controls,
            )
            step_damages.append(
                ResidualAttentionStepDamage(
                    sequence_id=rollout_pair.sequence_id,
                    step_index=reference.step_index,
                    metrics=metrics,
                )
            )
            previous_reference = reference
            previous_folded = folded

    summaries: dict[str, ResidualAttentionMetricSummary] = {}
    for metric in RESIDUAL_ATTENTION_METRICS:
        values = np.array(
            [step.metrics[metric] for step in step_damages], dtype=np.float64
        )
        first_values = np.array(
            [step.metrics[metric] for step in step_damages if step.step_index == 0],
            dtype=np.float64,
        )
        final_values = np.array(
            [
                step.metrics[metric]
                for step in step_damages
                if step.step_index
                == next(
                    horizon
                    for sequence_id, horizon in (
                        (pair.sequence_id, pair.horizon) for pair in evidence.rollouts
                    )
                    if sequence_id == step.sequence_id
                )
                - 1
            ],
            dtype=np.float64,
        )
        summaries[metric] = ResidualAttentionMetricSummary(
            mean=float(np.mean(values)),
            quantile=float(np.quantile(values, controls.quantile, method="higher")),
            worst=float(np.max(values)),
            first_step_worst=float(np.max(first_values)),
            final_step_worst=float(np.max(final_values)),
            sample_count=int(values.size),
        )
    total_weight = sum(controls.metric_weights.values())
    risk_score = 0.0
    for metric, weight in controls.metric_weights.items():
        summary = summaries[metric]
        aggregate = (
            1.0 - controls.worst_weight
        ) * summary.quantile + controls.worst_weight * summary.worst
        risk_score += weight * aggregate
    risk_score /= total_weight
    return ResidualAttentionCandidateDamage(
        candidate_ids=evidence.candidate_ids,
        sequence_count=len(evidence.rollouts),
        minimum_horizon=min(horizons),
        maximum_horizon=max(horizons),
        step_damages=tuple(step_damages),
        metrics=MappingProxyType(summaries),
        risk_score=float(risk_score),
        quantile=controls.quantile,
        worst_weight=controls.worst_weight,
        token_lane=evidence.token_lane,
    )


def gate_residual_attention_candidate(
    damage: ResidualAttentionCandidateDamage,
    policy: ResidualAttentionGatePolicy,
) -> ResidualAttentionGateDecision:
    """Apply explicit horizon, quantile, worst-case, and aggregate risk gates."""

    if not isinstance(damage, ResidualAttentionCandidateDamage):
        raise ResidualAttentionFoldError(
            "damage must be ResidualAttentionCandidateDamage."
        )
    if not isinstance(policy, ResidualAttentionGatePolicy):
        raise ResidualAttentionFoldError("policy must be ResidualAttentionGatePolicy.")
    reasons: list[str] = []
    ratios: list[float] = []
    if damage.minimum_horizon < policy.minimum_horizon:
        reasons.append(
            f"minimum_horizon={damage.minimum_horizon} is below "
            f"{policy.minimum_horizon}"
        )
    for metric, limit in policy.quantile_limits.items():
        measured = damage.metrics[metric].quantile
        ratios.append(
            measured / limit if limit > 0.0 else math.inf if measured else 0.0
        )
        if measured > limit:
            reasons.append(f"{metric}.quantile={measured:.12g} exceeds {limit:.12g}")
    for metric, limit in policy.worst_limits.items():
        measured = damage.metrics[metric].worst
        ratios.append(
            measured / limit if limit > 0.0 else math.inf if measured else 0.0
        )
        if measured > limit:
            reasons.append(f"{metric}.worst={measured:.12g} exceeds {limit:.12g}")
    if policy.max_risk_score is not None:
        limit = policy.max_risk_score
        ratios.append(
            damage.risk_score / limit
            if limit > 0.0
            else math.inf
            if damage.risk_score
            else 0.0
        )
        if damage.risk_score > limit:
            reasons.append(f"risk_score={damage.risk_score:.12g} exceeds {limit:.12g}")
    return ResidualAttentionGateDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        risk_score=damage.risk_score,
        maximum_limit_ratio=max(ratios, default=0.0),
    )


def rank_residual_attention_candidates(
    damages: Sequence[ResidualAttentionCandidateDamage],
    policy: ResidualAttentionGatePolicy,
) -> tuple[RankedResidualAttentionCandidate, ...]:
    """Rank accepted candidates first, then by limit ratio and atomic risk."""

    values = tuple(damages)
    if not values:
        return ()
    identities = [value.candidate_ids for value in values]
    if len(set(identities)) != len(identities):
        raise ResidualAttentionFoldError("Candidate damage identities must be unique.")
    decisions = [gate_residual_attention_candidate(value, policy) for value in values]
    ordered = sorted(
        zip(values, decisions),
        key=lambda item: (
            not item[1].accepted,
            item[1].maximum_limit_ratio,
            item[0].risk_score,
            item[0].candidate_ids,
        ),
    )
    return tuple(
        RankedResidualAttentionCandidate(rank, damage, decision)
        for rank, (damage, decision) in enumerate(ordered, start=1)
    )


def _evidence_rows(
    evidence: ResidualAttentionCandidateEvidence,
    reference_attribute: str,
    folded_attribute: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference_rows: list[np.ndarray] = []
    folded_rows: list[np.ndarray] = []
    for rollout in evidence.rollouts:
        for reference, folded in zip(rollout.reference.steps, rollout.folded.steps):
            reference_value = getattr(reference, reference_attribute)
            folded_value = getattr(folded, folded_attribute)
            reference_matrix = reference_value.reshape(-1, reference_value.shape[-1])
            folded_matrix = folded_value.reshape(-1, folded_value.shape[-1])
            if reference_matrix.shape[0] != folded_matrix.shape[0]:
                raise ResidualAttentionFoldError(
                    "Closed-form residual-attention rows must align exactly."
                )
            reference_rows.append(reference_matrix)
            folded_rows.append(folded_matrix)
    return np.concatenate(reference_rows, axis=0), np.concatenate(folded_rows, axis=0)


def fit_residual_attention_branch_scale_rows(
    reference_deltas: Any,
    folded_deltas: Any,
    *,
    ridge: float = 1e-6,
    prior_scale: float = 1.0,
    minimum_scale: float = 0.0,
    maximum_scale: float = 4.0,
) -> ResidualAttentionBranchScaleFit:
    """Fit one bounded scalar from explicit parent/folded O-delta rows.

    The solve minimizes ``||alpha * folded_delta - parent_delta||^2`` plus a
    trace-scaled penalty around ``prior_scale``.  It is an atomic downstream
    compensation: it never rescales Q, K, or V independently and never changes
    attention addressing.
    """

    ridge_value = _nonnegative_float(ridge, "ridge")
    prior = _finite_float(prior_scale, "prior_scale")
    minimum = _finite_float(minimum_scale, "minimum_scale")
    maximum = _finite_float(maximum_scale, "maximum_scale")
    if maximum < minimum:
        raise ResidualAttentionFoldError(
            "maximum_scale cannot be smaller than minimum_scale."
        )
    reference = _array(reference_deltas, "reference_deltas")
    folded = _array(folded_deltas, "folded_deltas")
    if reference.ndim != 2 or folded.ndim != 2:
        raise ResidualAttentionFoldError(
            "reference_deltas and folded_deltas must be [rows, hidden] matrices."
        )
    if reference.shape != folded.shape:
        raise ResidualAttentionFoldError(
            "Residual branch scaling requires matching parent/folded delta shapes."
        )
    reference_vector = reference.reshape(-1)
    folded_vector = folded.reshape(-1)
    energy = float(folded_vector @ folded_vector)
    effective_ridge = ridge_value * max(energy, np.finfo(np.float64).tiny)
    denominator = energy + effective_ridge
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ResidualAttentionFoldError(
            "Residual branch scale is undefined for a zero-energy folded branch "
            "without positive ridge."
        )
    unconstrained = float(
        (folded_vector @ reference_vector + effective_ridge * prior) / denominator
    )
    scale = min(max(unconstrained, minimum), maximum)
    baseline_error = _symmetric_relative_l2(
        reference,
        prior * folded,
        np.finfo(np.float64).eps,
        "prior-scaled residual branch",
    )
    fitted_error = _symmetric_relative_l2(
        reference,
        scale * folded,
        np.finfo(np.float64).eps,
        "fitted residual branch",
    )
    return ResidualAttentionBranchScaleFit(
        scale=scale,
        prior_scale=prior,
        ridge=ridge_value,
        effective_ridge=effective_ridge,
        row_count=int(reference.shape[0]),
        baseline_relative_l2=baseline_error,
        fitted_relative_l2=fitted_error,
        clipped_to_bounds=scale != unconstrained,
    )


def fit_residual_attention_branch_scale(
    evidence: ResidualAttentionCandidateEvidence,
    *,
    ridge: float = 1e-6,
    prior_scale: float = 1.0,
    minimum_scale: float = 0.0,
    maximum_scale: float = 4.0,
) -> ResidualAttentionBranchScaleFit:
    """Fit one complete O-branch scalar from candidate rollout evidence."""

    if not isinstance(evidence, ResidualAttentionCandidateEvidence):
        raise ResidualAttentionFoldError(
            "evidence must be ResidualAttentionCandidateEvidence."
        )
    reference, folded = _evidence_rows(
        evidence,
        "attention_delta",
        "attention_delta",
    )
    return fit_residual_attention_branch_scale_rows(
        reference,
        folded,
        ridge=ridge,
        prior_scale=prior_scale,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
    )


def fit_fixed_attention_output_ridge_rows(
    design_rows: Any,
    target_rows: Any,
    *,
    ridge: float = 1e-6,
    prior_output_matrix: Any | None = None,
) -> FixedAttentionOutputRidgeFit:
    """Fit O globally from explicit fixed-attention readout/target rows.

    ``design_rows`` should be ``A_parent @ V_folded`` (including any immutable
    logical-site scale) and ``target_rows`` the parent group contribution.  Q
    and K are never averaged or edited.
    """

    ridge_value = _nonnegative_float(ridge, "ridge")
    design = _array(design_rows, "design_rows")
    targets = _array(target_rows, "target_rows")
    if design.ndim != 2 or targets.ndim != 2 or design.shape[0] != targets.shape[0]:
        raise ResidualAttentionFoldError(
            "design_rows and target_rows must be aligned two-dimensional matrices."
        )
    input_width = int(design.shape[1])
    output_width = int(targets.shape[1])
    if prior_output_matrix is None:
        prior = np.zeros((input_width, output_width), dtype=np.float64)
        prior_used = False
    else:
        prior = _array(prior_output_matrix, "prior_output_matrix")
        if prior.shape != (input_width, output_width):
            raise ResidualAttentionFoldError(
                "prior_output_matrix must have shape "
                f"{(input_width, output_width)}, got {prior.shape}."
            )
        prior_used = True
    gram = design.T @ design
    trace_scale = float(np.trace(gram)) / max(input_width, 1)
    effective_ridge = ridge_value * max(trace_scale, np.finfo(np.float64).tiny)
    right = design.T @ targets + effective_ridge * prior
    system = gram + effective_ridge * np.eye(input_width, dtype=np.float64)
    if effective_ridge == 0.0:
        solution = np.linalg.lstsq(design, targets, rcond=None)[0]
        condition_number = float(np.linalg.cond(gram))
    else:
        try:
            solution = np.linalg.solve(system, right)
        except np.linalg.LinAlgError as error:
            raise ResidualAttentionFoldError(
                "Fixed-attention V/O ridge system could not be solved."
            ) from error
        condition_number = float(np.linalg.cond(system))
    if not bool(np.isfinite(solution).all()) or not math.isfinite(condition_number):
        raise ResidualAttentionFoldError(
            "Fixed-attention V/O ridge produced a non-finite result."
        )
    solution = np.array(solution, dtype=np.float64, copy=True, order="C")
    solution.setflags(write=False)
    baseline_error = _symmetric_relative_l2(
        targets,
        design @ prior,
        np.finfo(np.float64).eps,
        "prior fixed-attention output",
    )
    fitted_error = _symmetric_relative_l2(
        targets,
        design @ solution,
        np.finfo(np.float64).eps,
        "fitted fixed-attention output",
    )
    return FixedAttentionOutputRidgeFit(
        output_matrix=solution,
        ridge=ridge_value,
        effective_ridge=effective_ridge,
        row_count=int(design.shape[0]),
        input_width=input_width,
        output_width=output_width,
        prior_used=prior_used,
        baseline_relative_l2=baseline_error,
        fitted_relative_l2=fitted_error,
        gram_condition_number=condition_number,
    )


def fit_fixed_attention_output_ridge(
    evidence: ResidualAttentionCandidateEvidence,
    *,
    ridge: float = 1e-6,
    prior_output_matrix: Any | None = None,
) -> FixedAttentionOutputRidgeFit:
    """Fit O globally from folded V read through frozen parent attention."""

    if not isinstance(evidence, ResidualAttentionCandidateEvidence):
        raise ResidualAttentionFoldError(
            "evidence must be ResidualAttentionCandidateEvidence."
        )
    targets, design = _evidence_rows(
        evidence,
        "attention_delta",
        "fixed_attention_value_readout",
    )
    return fit_fixed_attention_output_ridge_rows(
        design,
        targets,
        ridge=ridge,
        prior_output_matrix=prior_output_matrix,
    )


def score_residual_attention_interaction(
    group: ResidualAttentionCandidateDamage,
    singletons: Sequence[ResidualAttentionCandidateDamage],
) -> ResidualAttentionInteractionReport:
    """Measure excess joint damage beyond the sum of individual fold damage.

    Positive excess is destructive synergy: candidates that looked safe alone
    are more harmful when installed together.  Negative excess indicates that
    their measured effects compensate.  The group must come from its own joint
    closed-loop rollout; this function never fabricates group evidence from
    singleton sketches.
    """

    if len(group.candidate_ids) < 2:
        raise ResidualAttentionFoldError(
            "Interaction scoring requires an atomic group of at least two candidates."
        )
    singleton_values = tuple(singletons)
    if len(singleton_values) != len(group.candidate_ids):
        raise ResidualAttentionFoldError(
            "Provide exactly one singleton damage report per group candidate."
        )
    singleton_ids: list[str] = []
    for singleton in singleton_values:
        if len(singleton.candidate_ids) != 1:
            raise ResidualAttentionFoldError(
                "Interaction baselines must be singleton damage reports."
            )
        singleton_ids.append(singleton.candidate_ids[0])
        if (
            singleton.sequence_count != group.sequence_count
            or singleton.minimum_horizon != group.minimum_horizon
            or singleton.maximum_horizon != group.maximum_horizon
        ):
            raise ResidualAttentionFoldError(
                "Group and singleton interaction evidence must use matching "
                "sequence counts and horizons."
            )
    if set(singleton_ids) != set(group.candidate_ids):
        raise ResidualAttentionFoldError(
            "Singleton candidate IDs must exactly cover the atomic group."
        )
    metrics: dict[str, ResidualAttentionInteractionMetric] = {}
    for metric in RESIDUAL_ATTENTION_METRICS:
        group_summary = group.metrics[metric]
        independent_quantile = sum(
            value.metrics[metric].quantile for value in singleton_values
        )
        independent_worst = sum(
            value.metrics[metric].worst for value in singleton_values
        )
        metrics[metric] = ResidualAttentionInteractionMetric(
            group_quantile=group_summary.quantile,
            independent_quantile_sum=independent_quantile,
            quantile_excess=group_summary.quantile - independent_quantile,
            group_worst=group_summary.worst,
            independent_worst_sum=independent_worst,
            worst_excess=group_summary.worst - independent_worst,
        )
    independent_risk = sum(value.risk_score for value in singleton_values)
    return ResidualAttentionInteractionReport(
        candidate_ids=group.candidate_ids,
        metrics=MappingProxyType(metrics),
        group_risk_score=group.risk_score,
        independent_risk_sum=independent_risk,
        risk_excess=group.risk_score - independent_risk,
    )


__all__ = [
    "FixedAttentionOutputRidgeFit",
    "RESIDUAL_ATTENTION_METRICS",
    "RankedResidualAttentionCandidate",
    "ResidualAttentionCandidateDamage",
    "ResidualAttentionCandidateEvidence",
    "ResidualAttentionBranchScaleFit",
    "ResidualAttentionFoldError",
    "ResidualAttentionGateDecision",
    "ResidualAttentionGatePolicy",
    "ResidualAttentionInteractionMetric",
    "ResidualAttentionInteractionReport",
    "ResidualAttentionMetricSummary",
    "ResidualAttentionRollout",
    "ResidualAttentionRolloutPair",
    "ResidualAttentionScoringConfig",
    "ResidualAttentionStepDamage",
    "ResidualAttentionStepEvidence",
    "fit_fixed_attention_output_ridge",
    "fit_fixed_attention_output_ridge_rows",
    "fit_residual_attention_branch_scale",
    "fit_residual_attention_branch_scale_rows",
    "gate_residual_attention_candidate",
    "rank_residual_attention_candidates",
    "score_residual_attention_candidate",
    "score_residual_attention_interaction",
]
