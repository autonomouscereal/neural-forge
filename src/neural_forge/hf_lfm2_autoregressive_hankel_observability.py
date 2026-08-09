"""Future-causal observability for fold-only LFM2 residual reduction.

This module turns authenticated, cached autoregressive Hankel sufficient
statistics into a residual-space observability Gram.  It is a closed-form
compiler operation: no model is constructed or executed, and no parameter is
optimized.  Each non-zero future lag is solved as a regularized, whitened
least-squares map from one residual boundary to a future LM-head state.  Lags
and boundaries are trace-normalized before aggregation so row count, raw
activation variance, or layer count cannot silently dominate the fold.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_balanced_circuit_basis import (
    PrecomputedDownstreamObservability,
    _tensor_sha256,
)
from .dynamic_circuit_ir import CircuitKind, canonical_sha256
from .hf_lfm2_autoregressive_hankel_statistics import (
    LFM2_AUTOREGRESSIVE_HANKEL_STATISTICS_SCHEMA,
    Lfm2AutoregressiveHankelStatistics,
    _tensor_payload_receipt,
)


LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROVENANCE_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-provenance-v1"
)
_ROLE = "autoregressive_hankel_future_causal_geometry"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2AutoregressiveHankelObservabilityError(ValueError):
    """Authenticated trajectory statistics cannot define AHB geometry."""


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{name} must be a finite positive number."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{name} must be a finite positive number."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{name} must be a positive integer."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{name} must be nonempty text."
        )
    return value.strip()


def _receipt_body(receipt: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{name} receipt must be a mapping."
        )
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{name} receipt has no valid SHA-256 seal."
        )
    if claimed != canonical_sha256(body):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{name} receipt seal is invalid."
        )
    return {**body, "receipt_sha256": claimed}


def _statistics_tensors(
    value: Lfm2AutoregressiveHankelStatistics,
) -> dict[str, torch.Tensor]:
    return {
        "lag_pair_counts": value.lag_pair_counts,
        "right_censor_counts": value.right_censor_counts,
        "state_sum": value.state_sum,
        "future_sum": value.future_sum,
        "state_second_moment": value.state_second_moment,
        "future_second_moment": value.future_second_moment,
        "state_future_cross": value.state_future_cross,
    }


@dataclass(frozen=True, slots=True)
class _VerifiedStatistics:
    value: Lfm2AutoregressiveHankelStatistics
    receipt: Mapping[str, Any]
    boundary_count: int
    lag_count: int
    hidden_size: int
    future_lags: tuple[int, ...]


def _verify_statistics(
    value: Any,
    *,
    index: int,
) -> _VerifiedStatistics:
    label = f"statistics[{index}]"
    if not isinstance(value, Lfm2AutoregressiveHankelStatistics):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} has the wrong type."
        )
    receipt = _receipt_body(value.receipt, name=label)
    if receipt.get("schema") != LFM2_AUTOREGRESSIVE_HANKEL_STATISTICS_SCHEMA:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} has the wrong schema."
        )
    source_sha256 = receipt.get("source_receipt_sha256")
    if not isinstance(source_sha256, str) or _SHA256.fullmatch(source_sha256) is None:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} is not bound to an authenticated source batch."
        )
    try:
        future_lags = tuple(int(item) for item in receipt["future_lags"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} has an invalid future-lag schedule."
        ) from exc
    if (
        len(future_lags) < 2
        or future_lags[0] != 0
        or tuple(sorted(set(future_lags))) != future_lags
        or not any(lag > 0 for lag in future_lags)
    ):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} must contain lag zero and at least one future lag."
        )
    boundary_count = _positive_int(receipt.get("boundary_count"), "boundary_count")
    lag_count = _positive_int(receipt.get("future_lag_count"), "future_lag_count")
    hidden_size = _positive_int(receipt.get("hidden_size"), "hidden_size")
    if lag_count != len(future_lags):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} future-lag count disagrees with its schedule."
        )
    tensors = _statistics_tensors(value)
    expected_shapes = {
        "lag_pair_counts": (lag_count,),
        "right_censor_counts": (lag_count,),
        "state_sum": (boundary_count, lag_count, hidden_size),
        "future_sum": (lag_count, hidden_size),
        "state_second_moment": (
            boundary_count,
            lag_count,
            hidden_size,
            hidden_size,
        ),
        "future_second_moment": (lag_count, hidden_size, hidden_size),
        "state_future_cross": (
            boundary_count,
            lag_count,
            hidden_size,
            hidden_size,
        ),
    }
    for name, tensor in tensors.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or tuple(int(size) for size in tensor.shape) != expected_shapes[name]
            or (
                name in {"lag_pair_counts", "right_censor_counts"}
                and tensor.dtype != torch.int64
            )
            or (
                name not in {"lag_pair_counts", "right_censor_counts"}
                and tensor.dtype != torch.float64
            )
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise Lfm2AutoregressiveHankelObservabilityError(
                f"{label}.{name} violates its sealed CPU tensor contract."
            )
    if value.lag_pair_counts.tolist() != receipt.get("lag_pair_counts"):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} pair counts disagree with its receipt."
        )
    if value.right_censor_counts.tolist() != receipt.get("right_censor_counts"):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} censor counts disagree with its receipt."
        )
    if receipt.get("tensor_payload") != _tensor_payload_receipt(tensors):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} tensor payload changed after sealing."
        )
    audit = receipt.get("operation_audit")
    if not isinstance(audit, Mapping) or any(
        value not in {0, False} for value in audit.values()
    ):
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} does not carry a zero-execution audit."
        )
    return _VerifiedStatistics(
        value=value,
        receipt=MappingProxyType(receipt),
        boundary_count=boundary_count,
        lag_count=lag_count,
        hidden_size=hidden_size,
        future_lags=future_lags,
    )


def _symmetrize(value: torch.Tensor) -> torch.Tensor:
    return ((value + value.T) * 0.5).contiguous()


def _regularized_psd_inverse(
    covariance: torch.Tensor,
    *,
    ridge_fraction: float,
    maximum_condition_number: float,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    matrix = _symmetrize(covariance)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    maximum = float(eigenvalues[-1].item())
    trace = float(torch.trace(matrix).item())
    width = int(matrix.shape[0])
    if not math.isfinite(maximum) or not math.isfinite(trace) or maximum <= 0.0:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} covariance has no positive finite energy."
        )
    relative_tolerance = max(
        1e-12,
        64.0 * torch.finfo(torch.float64).eps * width,
    )
    negative_tolerance = relative_tolerance * maximum
    minimum = float(eigenvalues[0].item())
    if minimum < -negative_tolerance:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} covariance is materially indefinite."
        )
    nonnegative = torch.clamp(eigenvalues, min=0.0)
    support = nonnegative > negative_tolerance
    effective_rank = int(torch.count_nonzero(support).item())
    if effective_rank <= 0:
        raise Lfm2AutoregressiveHankelObservabilityError(
            f"{label} covariance has no empirical support."
        )
    supported = nonnegative[support]
    mean_energy = float(supported.mean().item())
    ridge = max(
        mean_energy * ridge_fraction,
        maximum / (maximum_condition_number - 1.0),
        torch.finfo(torch.float64).tiny,
    )
    inverse_scales = torch.zeros_like(nonnegative)
    inverse_sqrt_scales = torch.zeros_like(nonnegative)
    inverse_scales[support] = (supported + ridge).reciprocal()
    inverse_sqrt_scales[support] = (supported + ridge).rsqrt()
    inverse = (
        (eigenvectors * inverse_scales.unsqueeze(0))
        @ eigenvectors.T
    ).contiguous()
    inverse_sqrt = (
        (eigenvectors * inverse_sqrt_scales.unsqueeze(0))
        @ eigenvectors.T
    ).contiguous()
    report = {
        "label": label,
        "width": width,
        "raw_trace_energy": trace,
        "raw_minimum_eigenvalue": minimum,
        "raw_maximum_eigenvalue": maximum,
        "effective_rank": effective_rank,
        "support_relative_tolerance": relative_tolerance,
        "support_absolute_tolerance": negative_tolerance,
        "support_only_inverse": True,
        "ridge": ridge,
        "ridge_fraction_of_supported_mean_energy": ridge_fraction,
        "maximum_condition_number": maximum_condition_number,
        "regularized_condition_number": float(
            (supported[-1].item() + ridge)
            / (supported[0].item() + ridge)
        ),
    }
    return inverse, inverse_sqrt, report


def _centered_scatter(
    verified: Sequence[_VerifiedStatistics],
    *,
    boundary_index: int,
    lag_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    hidden_size = verified[0].hidden_size
    state_scatter = torch.zeros(
        (hidden_size, hidden_size), dtype=torch.float64
    )
    future_scatter = torch.zeros_like(state_scatter)
    cross_scatter = torch.zeros_like(state_scatter)
    state_sum = torch.zeros(hidden_size, dtype=torch.float64)
    future_sum = torch.zeros(hidden_size, dtype=torch.float64)
    pair_count = 0
    for item in verified:
        count = int(item.value.lag_pair_counts[lag_index].item())
        if count <= 0:
            continue
        state_sum.add_(item.value.state_sum[boundary_index, lag_index])
        future_sum.add_(item.value.future_sum[lag_index])
        state_scatter.add_(
            item.value.state_second_moment[boundary_index, lag_index]
        )
        future_scatter.add_(item.value.future_second_moment[lag_index])
        cross_scatter.add_(item.value.state_future_cross[boundary_index, lag_index])
        pair_count += count
    centered_degrees = pair_count - 1
    if centered_degrees <= 0:
        raise Lfm2AutoregressiveHankelObservabilityError(
            "Future-causal fitting has no pooled centered degrees of freedom."
        )
    state_scatter.sub_(torch.outer(state_sum, state_sum) / pair_count)
    future_scatter.sub_(torch.outer(future_sum, future_sum) / pair_count)
    cross_scatter.sub_(torch.outer(state_sum, future_sum) / pair_count)
    scale = 1.0 / centered_degrees
    return (
        _symmetrize(state_scatter * scale),
        _symmetrize(future_scatter * scale),
        (cross_scatter * scale).contiguous(),
        pair_count,
        centered_degrees,
    )


@dataclass(frozen=True, slots=True)
class Lfm2AutoregressiveHankelObservabilityResult:
    """Authenticated AHB evidence plus reusable per-boundary Grams."""

    evidence: PrecomputedDownstreamObservability
    boundary_grams: tuple[torch.Tensor, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.evidence, PrecomputedDownstreamObservability)
            or not self.evidence.verify()
            or not self.boundary_grams
            or not isinstance(self.report, Mapping)
        ):
            raise Lfm2AutoregressiveHankelObservabilityError(
                "The AHB observability result is inconsistent."
            )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def gram(self) -> torch.Tensor:
        return self.evidence.gram

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def build_lfm2_autoregressive_hankel_observability(
    statistics: Sequence[Lfm2AutoregressiveHankelStatistics],
    *,
    boundary_names: Sequence[str],
    state_ridge_fraction: float = 1e-8,
    future_ridge_fraction: float = 1e-8,
    maximum_condition_number: float = 1e10,
) -> Lfm2AutoregressiveHankelObservabilityResult:
    """Build a globally balanced future-causal residual observability Gram.

    Valid causal pairs are pooled across trajectories and centered once; no
    pair crosses a trajectory boundary.  Lag zero is authenticated but
    deliberately excluded: the existing structural
    and cached-state lanes already cover contemporaneous geometry, while this
    evidence exists specifically to preserve multi-token causal behavior.
    """

    values = tuple(statistics)
    if not values:
        raise Lfm2AutoregressiveHankelObservabilityError(
            "statistics must contain at least one authenticated trajectory."
        )
    verified = tuple(
        _verify_statistics(value, index=index)
        for index, value in enumerate(values)
    )
    geometry = {
        (item.boundary_count, item.lag_count, item.hidden_size, item.future_lags)
        for item in verified
    }
    if len(geometry) != 1:
        raise Lfm2AutoregressiveHankelObservabilityError(
            "All AHB statistics must share boundary, hidden, and lag geometry."
        )
    source_receipts = tuple(
        str(item.receipt["source_receipt_sha256"]) for item in verified
    )
    if len(set(source_receipts)) != len(source_receipts):
        raise Lfm2AutoregressiveHankelObservabilityError(
            "AHB statistics must not repeat an authenticated source batch."
        )
    names = tuple(_text(value, "boundary_name") for value in boundary_names)
    boundary_count = verified[0].boundary_count
    hidden_size = verified[0].hidden_size
    future_lags = verified[0].future_lags
    if len(names) != boundary_count or len(set(names)) != len(names):
        raise Lfm2AutoregressiveHankelObservabilityError(
            "boundary_names must uniquely name every residual boundary."
        )
    state_ridge = _finite_positive(state_ridge_fraction, "state_ridge_fraction")
    future_ridge = _finite_positive(
        future_ridge_fraction, "future_ridge_fraction"
    )
    condition_limit = _finite_positive(
        maximum_condition_number, "maximum_condition_number"
    )
    if condition_limit <= 1.0:
        raise Lfm2AutoregressiveHankelObservabilityError(
            "maximum_condition_number must be greater than one."
        )

    future_whiteners: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
    for lag_index, lag in enumerate(future_lags):
        if lag == 0:
            continue
        _state, future_cov, _cross, _pairs, _degrees = _centered_scatter(
            verified,
            boundary_index=0,
            lag_index=lag_index,
        )
        _future_inverse, future_inverse_sqrt, future_report = (
            _regularized_psd_inverse(
                future_cov,
                ridge_fraction=future_ridge,
                maximum_condition_number=condition_limit,
                label=f"future.lag.{lag}",
            )
        )
        future_whiteners[lag_index] = (
            future_inverse_sqrt,
            future_report,
        )

    boundary_grams: list[torch.Tensor] = []
    boundary_reports: list[dict[str, Any]] = []
    with torch.inference_mode():
        for boundary_index, boundary_name in enumerate(names):
            accumulated = torch.zeros(
                (hidden_size, hidden_size), dtype=torch.float64
            )
            lag_reports: list[dict[str, Any]] = []
            contributing_lags = 0
            for lag_index, lag in enumerate(future_lags):
                if lag == 0:
                    continue
                state_cov, _future_cov, cross_cov, pairs, degrees = (
                    _centered_scatter(
                        verified,
                        boundary_index=boundary_index,
                        lag_index=lag_index,
                    )
                )
                _state_inverse, state_inverse_sqrt, state_report = (
                    _regularized_psd_inverse(
                        state_cov,
                        ridge_fraction=state_ridge,
                        maximum_condition_number=condition_limit,
                        label=f"{boundary_name}.lag.{lag}.state",
                    )
                )
                future_inverse_sqrt, future_report = future_whiteners[lag_index]
                canonical_factor = (
                    state_inverse_sqrt @ cross_cov @ future_inverse_sqrt
                ).contiguous()
                predictive_factor = (
                    state_inverse_sqrt @ canonical_factor
                ).contiguous()
                lag_gram = _symmetrize(
                    predictive_factor @ predictive_factor.T
                )
                lag_trace = float(torch.trace(lag_gram).item())
                if not math.isfinite(lag_trace) or lag_trace <= 0.0:
                    raise Lfm2AutoregressiveHankelObservabilityError(
                        f"{boundary_name} lag {lag} has no predictive energy."
                    )
                accumulated.add_(lag_gram)
                contributing_lags += 1
                lag_reports.append(
                    {
                        "lag": lag,
                        "pair_count": pairs,
                        "pooled_centered_degrees": degrees,
                        "raw_predictive_trace_energy": lag_trace,
                        "canonical_correlation_squared_sum": float(
                            canonical_factor.square().sum().item()
                        ),
                        "variance_independent_equal_horizon_weight": (
                            1.0 / len(future_whiteners)
                        ),
                        "trace_normalized_before_horizon_aggregation": False,
                        "gram_sha256": _tensor_sha256(lag_gram),
                        "state_covariance": state_report,
                        "future_covariance": future_report,
                    }
                )
            if contributing_lags == 0:
                raise Lfm2AutoregressiveHankelObservabilityError(
                    f"{boundary_name} has no non-zero future-lag evidence."
                )
            boundary_gram = _symmetrize(accumulated / contributing_lags)
            boundary_trace = float(torch.trace(boundary_gram).item())
            if not math.isfinite(boundary_trace) or boundary_trace <= 0.0:
                raise Lfm2AutoregressiveHankelObservabilityError(
                    f"{boundary_name} aggregate has no predictive energy."
                )
            boundary_grams.append(boundary_gram)
            boundary_reports.append(
                {
                    "boundary_index": boundary_index,
                    "boundary_name": boundary_name,
                    "future_lag_count": contributing_lags,
                    "equal_horizon_weighting": True,
                    "predictive_strength_preserved": True,
                    "trace_normalized_before_boundary_aggregation": False,
                    "gram_sha256": _tensor_sha256(boundary_gram),
                    "lags": lag_reports,
                }
            )
        global_gram = _symmetrize(sum(boundary_grams) / boundary_count)
        global_trace = float(torch.trace(global_gram).item())
        if not math.isfinite(global_trace) or global_trace <= 0.0:
            raise Lfm2AutoregressiveHankelObservabilityError(
                "The global AHB observability Gram has no predictive energy."
            )

    config = {
        "state_ridge_fraction": state_ridge,
        "future_ridge_fraction": future_ridge,
        "maximum_condition_number": condition_limit,
        "pooled_pair_centering": True,
        "cross_trajectory_pairs_created": False,
        "lag_zero_used_for_selection": False,
        "nonzero_lags": [lag for lag in future_lags if lag > 0],
        "equal_horizon_weighting": True,
        "horizon_trace_normalization": False,
        "equal_boundary_weighting": True,
        "boundary_trace_normalization": False,
    }
    provenance_body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROVENANCE_SCHEMA,
        "method": "regularized-whitened-future-state-predictive-geometry",
        "source_statistics_receipt_sha256s": [
            item.receipt["receipt_sha256"] for item in verified
        ],
        "source_batch_receipt_sha256s": list(source_receipts),
        "source_tensor_payload_sha256s": [
            item.receipt["tensor_payload"]["payload_sha256"]
            for item in verified
        ],
        "boundary_names": list(names),
        "future_lags": list(future_lags),
        "configuration": config,
        "boundary_gram_sha256s": [
            _tensor_sha256(value) for value in boundary_grams
        ],
        "global_gram_sha256": _tensor_sha256(global_gram),
        "fit_only_selection_evidence": True,
        "parent_forward_calls": 0,
        "child_forward_calls": 0,
        "trace_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
    }
    provenance = {
        **provenance_body,
        "provenance_sha256": canonical_sha256(provenance_body),
    }
    evidence_id = (
        "interface.text.logits.autoregressive_hankel."
        + provenance["provenance_sha256"][:16]
    )
    evidence = PrecomputedDownstreamObservability(
        evidence_id=evidence_id,
        node_id="output.lm_head",
        circuit_kind=CircuitKind.OUTPUT_HEAD,
        role=_ROLE,
        gram=global_gram,
        provenance=provenance,
        interface_ids=("text.logits.output",),
    )
    report_body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_SCHEMA,
        "method": "autoregressive-hankel-bisimulation-future-causal-v1",
        "source_trajectory_count": len(verified),
        "source_width": hidden_size,
        "boundary_count": boundary_count,
        "boundary_names": list(names),
        "future_lags": list(future_lags),
        "configuration": config,
        "boundaries": boundary_reports,
        "global_gram_sha256": _tensor_sha256(global_gram),
        "precomputed_observability": evidence.as_dict(),
        "empirical_on_trajectory_not_action_complete_bisimulation": True,
        "target_token_counterfactual_successors_required_for_action_completion": True,
        "model_forward_calls": 0,
        "parent_trace_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
    }
    report = {
        **report_body,
        "report_sha256": canonical_sha256(report_body),
    }
    return Lfm2AutoregressiveHankelObservabilityResult(
        evidence=evidence,
        boundary_grams=tuple(boundary_grams),
        report=report,
    )


__all__ = [
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_SCHEMA",
    "Lfm2AutoregressiveHankelObservabilityError",
    "Lfm2AutoregressiveHankelObservabilityResult",
    "build_lfm2_autoregressive_hankel_observability",
]
