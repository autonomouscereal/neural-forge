"""Fold-only moving-gauge residual reclosure for an already folded child.

The ordinary residual skip is an identity map, while independently folded
boundaries need not use the same child coordinate chart.  The mature CPCF
boundary primitive already gives the compositionally correct local target::

    child_write = encoded_parent_endpoint - actual_child_pre_state

This module adds the missing whole-stack pieces without executing either
model: a cycle-coupled encoder/decoder at every cached parent/live-child
boundary, explicit ``E_(l+1) D_l`` chart transitions, polar separation of the
orthogonal factor that may cross RMSNorm from the non-orthogonal factor that
must be absorbed by physical writers, and one forward plus one reverse
closed-form writer reconciliation sweep.  Publication consists only of dense
replacement tensors in the seed target dtype.  No runtime sidecar remains.

This is analytic folding, not training.  There is no autograd graph, backward
call, optimizer, parent recapture, model forward, pruning, quantization, or
numeric quality gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_fold_executor import run_fold_only_callback
from neural_forge.hf_closed_form_causal_circuit_fold import PredictiveBoundaryMap
from neural_forge.hf_cpcf_boundary_chart import (
    CpcfBoundaryChart,
    fit_cpcf_orthogonal_chart_transport,
)
from neural_forge.hf_cpcf_sequential_boundary_transport import (
    fit_cpcf_sequential_downstream_chart_transport,
)


LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-residual-reclosure-v1"
)
_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_moving_gauge_residual_reclosure."
    "cycle-coupled-polar-forward-reverse-dense-replacement.v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INTERFACE_ROLES = frozenset({"input", "intermediate", "terminal"})
_TRANSITION_MODES = frozenset({"moving_gauge", "forced_identity_diagnostic"})
_SUPPORTED_TARGET_DTYPES = frozenset({"float32", "bfloat16"})
_WRITER_PUBLICATION_KINDS = frozenset(
    {"live_physical_parameter", "deferred_dense_prior"}
)
_BOUNDARY_CHILD_ROW_EVIDENCE_KINDS = frozenset(
    {
        "live_child_recapture",
        "deferred_prior_analytic_macrocell",
    }
)
_ZERO_EXECUTION = {
    "parent_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "parent_model_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "parent_cache_regenerations": 0,
    "target_model_forward_calls": 0,
    "target_layer_forward_calls": 0,
    "generation_calls": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2MovingGaugeResidualReclosureError(ValueError):
    """Moving-gauge evidence, geometry, or publication is invalid."""


class Lfm2MovingGaugeResidualAuthenticationError(Lfm2MovingGaugeResidualReclosureError):
    """A sealed moving-gauge input or output failed authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for moving-gauge residual reclosure."
        ) from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2MovingGaugeResidualReclosureError(
            "Moving-gauge metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _sha256(detached)})


def verify_lfm2_moving_gauge_report(value: Mapping[str, Any]) -> bool:
    """Verify one canonical report emitted by this module."""

    if not isinstance(value, Mapping):
        return False
    try:
        body = dict(value)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _sha256(body)
    except (TypeError, ValueError):
        return False


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2MovingGaugeResidualAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2MovingGaugeResidualReclosureError(f"{name} must be nonempty text.")
    return value.strip()


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2MovingGaugeResidualReclosureError(
            f"{name} must be a finite nonnegative number."
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2MovingGaugeResidualReclosureError(
            f"{name} must be a finite nonnegative number."
        )
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise Lfm2MovingGaugeResidualReclosureError(f"{name} must be positive.")
    return result


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def fingerprint_lfm2_moving_gauge_tensor(value: Any) -> str:
    """Hash tensor dtype, shape, and exact dense bytes."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Only materialized dense tensors can be authenticated."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _matrix(
    value: Any,
    name: str,
    *,
    dtype: Any | None = None,
    require_plain: bool = True,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or (require_plain and (value.requires_grad or value.grad_fn is not None))
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            f"{name} must be a detached nonempty dense real floating matrix."
        )
    result = value.detach().to(device="cpu", dtype=dtype).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2MovingGaugeResidualReclosureError(f"{name} contains NaN or infinity.")
    return result


def _weights(value: Any | None, rows: int, name: str) -> Any:
    torch = _torch()
    if value is None:
        return torch.ones(rows, dtype=torch.float64)
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or int(value.numel()) != rows
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            f"{name} must contain one detached positive floating value per row."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()) or bool(
        (result <= 0.0).any().item()
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            f"{name} must contain finite positive values."
        )
    return result


def _shares_storage(left: Any, right: Any) -> bool:
    torch = _torch()
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return left is right
    if left is right or left.device != right.device:
        return left is right
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except RuntimeError:
        return False


def _weighted_rmse(prediction: Any, target: Any, weights: Any) -> float:
    torch = _torch()
    energy = (weights.unsqueeze(1) * (prediction - target).square()).sum()
    denominator = weights.sum() * int(target.shape[1])
    return float(torch.sqrt(energy / denominator).item())


def _relative_rmse(prediction: Any, target: Any, weights: Any) -> float:
    torch = _torch()
    difference = _weighted_rmse(prediction, target, weights)
    denominator = float(
        torch.sqrt(
            (weights.unsqueeze(1) * target.square()).sum()
            / (weights.sum() * int(target.shape[1]))
        )
        .clamp_min(1e-30)
        .item()
    )
    return difference / denominator


def _ridge_scale(design: Any, weights: Any) -> float:
    torch = _torch()
    root = torch.sqrt(weights).unsqueeze(1)
    weighted = design * root
    spectral = torch.linalg.matrix_norm(weighted, ord=2)
    scale = float((spectral * spectral).item())
    return max(scale, torch.finfo(torch.float64).tiny)


def _anchored_ridge_replacement(
    design: Any,
    target: Any,
    anchor: Any,
    weights: Any,
    *,
    ridge_ratio: float,
    absolute_ridge_floor: float,
) -> tuple[Any, Mapping[str, Any]]:
    """Solve ``min ||XW-Y||_w^2 + ridge ||W-anchor||^2`` exactly."""

    torch = _torch()
    x = _matrix(design, "ridge design", dtype=torch.float64)
    y = _matrix(target, "ridge target", dtype=torch.float64)
    a = _matrix(anchor, "ridge anchor", dtype=torch.float64)
    if (
        int(x.shape[0]) != int(y.shape[0])
        or int(x.shape[1]) != int(a.shape[0])
        or int(y.shape[1]) != int(a.shape[1])
        or int(weights.numel()) != int(x.shape[0])
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Closed-form ridge geometry is inconsistent."
        )
    ratio = _finite_nonnegative(ridge_ratio, "ridge_ratio")
    floor = _finite_positive(absolute_ridge_floor, "absolute_ridge_floor")
    scale = _ridge_scale(x, weights)
    ridge = max(floor, ratio * scale)
    root = torch.sqrt(weights).unsqueeze(1)
    weighted_x = x * root
    weighted_residual = (y - x @ a) * root
    if int(x.shape[1]) <= int(x.shape[0]):
        normal = weighted_x.T @ weighted_x
        delta = torch.linalg.solve(
            normal + ridge * torch.eye(int(normal.shape[0]), dtype=torch.float64),
            weighted_x.T @ weighted_residual,
        )
        solve_form = "primal"
    else:
        dual = weighted_x @ weighted_x.T
        delta = weighted_x.T @ torch.linalg.solve(
            dual + ridge * torch.eye(int(dual.shape[0]), dtype=torch.float64),
            weighted_residual,
        )
        solve_form = "dual"
    replacement = (a + delta).contiguous()
    if not bool(torch.isfinite(replacement).all().item()):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Closed-form ridge replacement is non-finite."
        )
    report = _seal(
        {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "anchored_closed_form_ridge_replacement",
            "solve_form": solve_form,
            "rows": int(x.shape[0]),
            "input_width": int(x.shape[1]),
            "output_width": int(y.shape[1]),
            "ridge_ratio": ratio,
            "ridge_scale": scale,
            "effective_ridge": ridge,
            "design_sha256": fingerprint_lfm2_moving_gauge_tensor(x),
            "target_sha256": fingerprint_lfm2_moving_gauge_tensor(y),
            "anchor_sha256": fingerprint_lfm2_moving_gauge_tensor(a),
            "replacement_sha256": fingerprint_lfm2_moving_gauge_tensor(replacement),
            "closed_form_linear_solve": True,
            "autograd_used": False,
            "quality_gate": False,
        }
    )
    return replacement, report


def _explicit_nonnegative_frozen_row_weights(
    value: Any,
    rows: int,
) -> Any:
    """Return caller-supplied FP64 row weights without changing their mass."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or int(value.numel()) != rows
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "row_weights must be an explicit detached dense real floating "
            "vector with one value per feature row."
        )
    weights = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(weights).all().item()) or bool(
        (weights < 0.0).any().item()
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "row_weights must contain finite nonnegative values."
        )
    total_mass = float(weights.sum().item())
    if not math.isfinite(total_mass):
        raise Lfm2MovingGaugeResidualReclosureError(
            "row_weights total mass must be finite."
        )
    if total_mass <= 0.0:
        raise Lfm2MovingGaugeResidualReclosureError(
            "row_weights must have positive total mass."
        )
    return weights


def solve_lfm2_trace_scaled_weighted_anchored_ridge(
    feature_rows: Any,
    target_rows: Any,
    anchor: Any,
    row_weights: Any,
    *,
    output_dtype: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Solve one frozen-feature, trace-scaled anchored ridge system.

    The returned matrix has equation orientation ``[feature_width,
    output_width]``.  In FP64 this computes

    ``V = (F.T D F + lambda I)^-1 (F.T D Y + lambda V0)``

    with ``lambda = max(1e-12, 1e-8 * trace(F.T D F) / d)``.  Caller-supplied
    row weights are used exactly: zero-weight rows are permitted, but the
    total mass must be positive.  The implementation always solves the primal
    regularized normal equation with one Cholesky path.  It never selects a
    dual system, pseudoinverse, least-squares fallback, or iterative optimizer.
    The FP64 solution is cast exactly once to the requested publication dtype.
    """

    torch = _torch()
    if output_dtype not in {torch.float32, torch.bfloat16}:
        raise Lfm2MovingGaugeResidualReclosureError(
            "output_dtype must be torch.float32 or torch.bfloat16."
        )
    source_features = _matrix(feature_rows, "feature_rows")
    source_targets = _matrix(target_rows, "target_rows")
    source_anchor = _matrix(anchor, "anchor")
    features = source_features.to(dtype=torch.float64).contiguous()
    targets = source_targets.to(dtype=torch.float64).contiguous()
    anchor64 = source_anchor.to(dtype=torch.float64).contiguous()
    weights = _explicit_nonnegative_frozen_row_weights(
        row_weights,
        int(features.shape[0]),
    )
    rows = int(features.shape[0])
    feature_width = int(features.shape[1])
    output_width = int(targets.shape[1])
    if int(targets.shape[0]) != rows or tuple(anchor64.shape) != (
        feature_width,
        output_width,
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Trace-scaled anchored ridge geometry is inconsistent."
        )

    with torch.no_grad():
        root_weights = torch.sqrt(weights).unsqueeze(1)
        weighted_features = features * root_weights
        weighted_targets = targets * root_weights
        gram = (weighted_features.T @ weighted_features).contiguous()
        cross = (weighted_features.T @ weighted_targets).contiguous()
        gram_trace = float(torch.trace(gram).item())
        if not math.isfinite(gram_trace) or gram_trace < 0.0:
            raise Lfm2MovingGaugeResidualReclosureError(
                "The weighted feature Gram trace is invalid."
            )
        ridge = max(1e-12, 1e-8 * gram_trace / feature_width)
        identity = torch.eye(feature_width, dtype=torch.float64)
        system = (gram + ridge * identity).contiguous()
        right_hand_side = (cross + ridge * anchor64).contiguous()
        finite_before_solve = {
            "weighted_features": bool(torch.isfinite(weighted_features).all().item()),
            "weighted_targets": bool(torch.isfinite(weighted_targets).all().item()),
            "gram": bool(torch.isfinite(gram).all().item()),
            "cross_product": bool(torch.isfinite(cross).all().item()),
            "regularized_system": bool(torch.isfinite(system).all().item()),
            "right_hand_side": bool(torch.isfinite(right_hand_side).all().item()),
        }
        if not all(finite_before_solve.values()):
            raise Lfm2MovingGaugeResidualReclosureError(
                "The FP64 weighted normal equation contains NaN or infinity."
            )

        factor, factor_info = torch.linalg.cholesky_ex(
            system,
            check_errors=False,
        )
        cholesky_info = int(factor_info.max().item())
        if cholesky_info != 0:
            raise Lfm2MovingGaugeResidualReclosureError(
                "The FP64 primal regularized system is not positive definite; "
                "no fallback solve is permitted."
            )
        solution64 = torch.cholesky_solve(
            right_hand_side,
            factor,
        ).contiguous()
        if not bool(torch.isfinite(solution64).all().item()):
            raise Lfm2MovingGaugeResidualReclosureError(
                "The FP64 anchored ridge solution contains NaN or infinity."
            )
        normal_residual = system @ solution64 - right_hand_side
        normal_residual_max_abs = float(normal_residual.abs().max().item())

        # This is the sole publication-dtype cast in the solver.
        solution = solution64.to(dtype=output_dtype).contiguous().clone()
        if not bool(torch.isfinite(solution).all().item()):
            raise Lfm2MovingGaugeResidualReclosureError(
                "The requested output-dtype cast produced NaN or infinity."
            )

    report = _seal(
        {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "trace_scaled_fp64_weighted_anchored_ridge",
            "equation": ("V=(F^T D F+lambda I)^-1(F^T D Y+lambda V0)"),
            "rows": rows,
            "feature_width": feature_width,
            "output_width": output_width,
            "feature_rows_source_dtype": _dtype_name(source_features),
            "target_rows_source_dtype": _dtype_name(source_targets),
            "anchor_source_dtype": _dtype_name(source_anchor),
            "accumulation_dtype": "float64",
            "solve_dtype": "float64",
            "output_dtype": str(output_dtype).removeprefix("torch."),
            "output_dtype_cast_count": 1,
            "feature_rows_source_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(source_features)
            ),
            "target_rows_source_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(source_targets)
            ),
            "anchor_source_sha256": fingerprint_lfm2_moving_gauge_tensor(source_anchor),
            "feature_rows_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(features)
            ),
            "target_rows_fp64_sha256": (fingerprint_lfm2_moving_gauge_tensor(targets)),
            "anchor_fp64_sha256": fingerprint_lfm2_moving_gauge_tensor(anchor64),
            "row_weights_fp64_sha256": (fingerprint_lfm2_moving_gauge_tensor(weights)),
            "fp64_solution_sha256": fingerprint_lfm2_moving_gauge_tensor(solution64),
            "output_solution_sha256": fingerprint_lfm2_moving_gauge_tensor(solution),
            "explicit_row_weights": True,
            "row_weights_frozen": True,
            "row_weights_nonnegative": True,
            "row_weight_sum": float(weights.sum().item()),
            "row_weight_minimum": float(weights.min().item()),
            "row_weight_maximum": float(weights.max().item()),
            "positive_weight_row_count": int((weights > 0.0).sum().item()),
            "zero_weight_row_count": int((weights == 0.0).sum().item()),
            "ridge_formula": ("max(1e-12,1e-8*trace(F^T D F)/feature_width)"),
            "ridge_absolute_floor": 1e-12,
            "ridge_trace_ratio": 1e-8,
            "weighted_gram_trace": gram_trace,
            "ridge_trace_denominator": feature_width,
            "effective_ridge": ridge,
            "solver_path": "primal_weighted_normal_equation_cholesky",
            "primal_system": True,
            "dual_system": False,
            "pseudoinverse_used": False,
            "fallback_solve_used": False,
            "cholesky_info": cholesky_info,
            "gram_rank": None,
            "gram_rank_computed": False,
            "gram_rank_reason": ("not_computed_to_avoid_full_o_d_cubed_decomposition"),
            "regularized_system_rank": feature_width,
            "regularized_system_rank_evidence": (
                "successful_fp64_cholesky_factorization"
            ),
            "regularized_system_rank_numerically_estimated": False,
            "regularized_system_full_rank": True,
            "regularized_system_full_rank_proven_by_cholesky": True,
            "rank_diagnostic_svd_calls": 0,
            "finite_checks": {
                **finite_before_solve,
                "fp64_solution": True,
                "output_solution": True,
            },
            "normal_equation_residual_max_abs": normal_residual_max_abs,
            "closed_form_linear_solve": True,
            "autograd_used": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "quality_gate": False,
        }
    )
    return solution, report


def solve_lfm2_moving_gauge_anchored_writer_replacement(
    fit_input_rows: Any,
    desired_output_rows: Any,
    seed_weight: Any,
    fit_weights: Any,
    *,
    ridge_ratio: float = 1e-8,
    absolute_ridge_floor: float = 1e-12,
) -> tuple[Any, Mapping[str, Any]]:
    """Solve one physical dense writer from fresh primals without optimization.

    ``seed_weight`` follows the physical ``torch.nn.Linear`` layout
    ``[output, input]``.  The returned tensor preserves that exact shape and
    target dtype.  The solve is an anchored closed-form ridge projection; it
    performs no autograd, backward pass, optimizer step, or iterative training.
    """

    torch = _torch()
    seed = _matrix(seed_weight, "seed_weight", require_plain=True)
    if _dtype_name(seed) not in _SUPPORTED_TARGET_DTYPES:
        raise Lfm2MovingGaugeResidualReclosureError(
            "Physical writer seed dtype must be FP32 or BF16."
        )
    design = _matrix(
        fit_input_rows,
        "fit_input_rows",
        dtype=torch.float64,
    )
    target = _matrix(
        desired_output_rows,
        "desired_output_rows",
        dtype=torch.float64,
    )
    weights = _weights(fit_weights, int(design.shape[0]), "fit_weights")
    if (
        int(design.shape[0]) != int(target.shape[0])
        or int(design.shape[1]) != int(seed.shape[1])
        or int(target.shape[1]) != int(seed.shape[0])
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Fresh-primal writer rows do not match the physical dense geometry."
        )
    replacement_columns, solve_report = _anchored_ridge_replacement(
        design,
        target,
        seed.detach().to(device="cpu", dtype=torch.float64).T,
        weights,
        ridge_ratio=ridge_ratio,
        absolute_ridge_floor=absolute_ridge_floor,
    )
    replacement = replacement_columns.T.to(dtype=seed.dtype).contiguous().clone()
    prediction_before = (
        design
        @ seed.detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .T
    )
    prediction_after = design @ replacement.to(dtype=torch.float64).T
    report = _seal(
        {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "fresh_primal_anchored_physical_writer_replacement",
            "rows": int(design.shape[0]),
            "input_width": int(design.shape[1]),
            "output_width": int(target.shape[1]),
            "target_dtype": _dtype_name(seed),
            "fit_input_rows_sha256": fingerprint_lfm2_moving_gauge_tensor(design),
            "desired_output_rows_sha256": fingerprint_lfm2_moving_gauge_tensor(target),
            "fit_weights_sha256": fingerprint_lfm2_moving_gauge_tensor(weights),
            "seed_weight_sha256": fingerprint_lfm2_moving_gauge_tensor(seed),
            "replacement_weight_sha256": fingerprint_lfm2_moving_gauge_tensor(
                replacement
            ),
            "closed_form_solve_sha256": solve_report["report_sha256"],
            "fit_relative_rmse_before": _relative_rmse(
                prediction_before,
                target,
                weights,
            ),
            "fit_relative_rmse_after_target_dtype_cast": _relative_rmse(
                prediction_after,
                target,
                weights,
            ),
            "fresh_primal_rows_consumed": True,
            "target_dtype_cast_measured": True,
            "closed_form_linear_solve": True,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "quality_gate": False,
        }
    )
    return replacement, report


def _regularized_reciprocal(
    mapping: Any,
    *,
    ridge_ratio: float,
    absolute_ridge_floor: float,
) -> tuple[Any, Mapping[str, Any]]:
    """Return a deterministic ridge reciprocal without ``pinv`` thresholds."""

    torch = _torch()
    source_to_target = _matrix(
        mapping,
        "mapping",
        dtype=torch.float64,
    )
    source_width = int(source_to_target.shape[0])
    weights = torch.ones(source_width, dtype=torch.float64)
    reciprocal, report = _anchored_ridge_replacement(
        source_to_target,
        torch.eye(source_width, dtype=torch.float64),
        torch.zeros(
            int(source_to_target.shape[1]),
            source_width,
            dtype=torch.float64,
        ),
        weights,
        ridge_ratio=ridge_ratio,
        absolute_ridge_floor=absolute_ridge_floor,
    )
    return reciprocal, report


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeReclosurePolicy:
    """Fixed analytic policy; scalar diagnostics never select a child."""

    chart_ridge_ratio: float = 1e-8
    cycle_consistency_weight: float = 1e-2
    writer_ridge_ratio: float = 1e-8
    reverse_transport_ridge_ratio: float = 1e-8
    absolute_ridge_floor: float = 1e-12
    orthogonality_tolerance: float = 1e-10
    transition_mode: str = "moving_gauge"
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "chart_ridge_ratio",
            "cycle_consistency_weight",
            "writer_ridge_ratio",
            "reverse_transport_ridge_ratio",
            "orthogonality_tolerance",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_nonnegative(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "absolute_ridge_floor",
            _finite_positive(
                self.absolute_ridge_floor,
                "absolute_ridge_floor",
            ),
        )
        mode = str(self.transition_mode).strip().lower()
        if mode not in _TRANSITION_MODES:
            raise Lfm2MovingGaugeResidualReclosureError(
                "transition_mode must be moving_gauge or forced_identity_diagnostic."
            )
        object.__setattr__(self, "transition_mode", mode)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Moving-gauge policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "moving_gauge_reclosure_policy",
            "chart_ridge_ratio": self.chart_ridge_ratio,
            "cycle_consistency_weight": self.cycle_consistency_weight,
            "writer_ridge_ratio": self.writer_ridge_ratio,
            "reverse_transport_ridge_ratio": (self.reverse_transport_ridge_ratio),
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "orthogonality_tolerance": self.orthogonality_tolerance,
            "transition_mode": self.transition_mode,
            "forward_sweep_count": 1,
            "reverse_reconciliation_sweep_count": 1,
            "closed_form_solves_only": True,
            "numeric_quality_gate": False,
            "execution": dict(_ZERO_EXECUTION),
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBoundaryEvidence:
    """One sealed cached-parent/live-child full-prefix boundary pair."""

    boundary_id: str
    interface_role: str
    interface_locked: bool
    base_encoder: Any = field(repr=False, compare=False)
    fit_parent_rows: Any = field(repr=False, compare=False)
    fit_child_rows: Any = field(repr=False, compare=False)
    fit_weights: Any = field(repr=False, compare=False)
    fit_provenance: str
    row_alignment_sha256: str
    cached_parent_evidence_sha256: str
    live_child_recapture_sha256: str
    child_row_evidence_kind: str = "live_child_recapture"
    child_row_evidence_sha256: str | None = None
    heldout_parent_rows: Any | None = field(default=None, repr=False, compare=False)
    heldout_child_rows: Any | None = field(default=None, repr=False, compare=False)
    heldout_weights: Any | None = field(default=None, repr=False, compare=False)
    heldout_provenance: str | None = None
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        torch = _torch()
        boundary_id = _text(self.boundary_id, "boundary_id")
        role = str(self.interface_role).strip().lower()
        if role not in _INTERFACE_ROLES:
            raise Lfm2MovingGaugeResidualReclosureError(
                f"interface_role must be one of {sorted(_INTERFACE_ROLES)}."
            )
        if not isinstance(self.interface_locked, bool):
            raise TypeError("interface_locked must be boolean.")
        if role in {"input", "terminal"} and not self.interface_locked:
            raise Lfm2MovingGaugeResidualReclosureError(
                "Input and terminal boundaries must remain interface locked."
            )
        encoder = _matrix(self.base_encoder, "base_encoder", dtype=torch.float64)
        parent = _matrix(
            self.fit_parent_rows,
            "fit_parent_rows",
            dtype=torch.float64,
        )
        child = _matrix(
            self.fit_child_rows,
            "fit_child_rows",
            dtype=torch.float64,
        )
        if (
            int(parent.shape[0]) != int(child.shape[0])
            or int(parent.shape[1]) != int(encoder.shape[0])
            or int(child.shape[1]) != int(encoder.shape[1])
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Boundary parent, child, and base-encoder geometry differs."
            )
        weight = _weights(self.fit_weights, int(parent.shape[0]), "fit_weights")
        heldout_values = (
            self.heldout_parent_rows,
            self.heldout_child_rows,
            self.heldout_weights,
            self.heldout_provenance,
        )
        heldout_requested = any(value is not None for value in heldout_values)
        if heldout_requested and not all(value is not None for value in heldout_values):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Heldout boundary evidence must supply rows, weights, and provenance."
            )
        heldout_parent = None
        heldout_child = None
        heldout_weight = None
        heldout_provenance = None
        if heldout_requested:
            if (
                _shares_storage(self.fit_parent_rows, self.heldout_parent_rows)
                or _shares_storage(self.fit_child_rows, self.heldout_child_rows)
                or _shares_storage(self.fit_weights, self.heldout_weights)
            ):
                raise Lfm2MovingGaugeResidualReclosureError(
                    "Fit and heldout boundary evidence must have disjoint storage."
                )
            heldout_parent = _matrix(
                self.heldout_parent_rows,
                "heldout_parent_rows",
                dtype=torch.float64,
            )
            heldout_child = _matrix(
                self.heldout_child_rows,
                "heldout_child_rows",
                dtype=torch.float64,
            )
            heldout_weight = _weights(
                self.heldout_weights,
                int(heldout_parent.shape[0]),
                "heldout_weights",
            )
            heldout_provenance = _text(
                self.heldout_provenance,
                "heldout_provenance",
            )
            if (
                int(heldout_parent.shape[0]) != int(heldout_child.shape[0])
                or int(heldout_parent.shape[1]) != int(parent.shape[1])
                or int(heldout_child.shape[1]) != int(child.shape[1])
            ):
                raise Lfm2MovingGaugeResidualReclosureError(
                    "Heldout boundary geometry differs from FIT geometry."
                )
            if heldout_provenance == self.fit_provenance:
                raise Lfm2MovingGaugeResidualReclosureError(
                    "FIT and heldout provenance identifiers must differ."
                )
        fit_provenance = _text(self.fit_provenance, "fit_provenance")
        row_alignment = _digest(
            self.row_alignment_sha256,
            "row_alignment_sha256",
        )
        cached_parent = _digest(
            self.cached_parent_evidence_sha256,
            "cached_parent_evidence_sha256",
        )
        live_child = _digest(
            self.live_child_recapture_sha256,
            "live_child_recapture_sha256",
        )
        child_row_evidence_kind = str(self.child_row_evidence_kind).strip().lower()
        if child_row_evidence_kind not in _BOUNDARY_CHILD_ROW_EVIDENCE_KINDS:
            raise Lfm2MovingGaugeResidualReclosureError(
                "child_row_evidence_kind must identify a live child recapture or "
                "a deferred-prior analytic macrocell."
            )
        if (
            child_row_evidence_kind == "deferred_prior_analytic_macrocell"
            and role != "terminal"
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Only the locked terminal boundary may use deferred-prior "
                "analytic child rows."
            )
        if self.child_row_evidence_sha256 is None:
            if child_row_evidence_kind != "live_child_recapture":
                raise Lfm2MovingGaugeResidualReclosureError(
                    "Deferred-prior analytic child rows require their own "
                    "authenticated evidence digest."
                )
            child_row_evidence = live_child
        else:
            child_row_evidence = _digest(
                self.child_row_evidence_sha256,
                "child_row_evidence_sha256",
            )
        body = {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "paired_full_prefix_boundary_evidence",
            "boundary_id": boundary_id,
            "interface_role": role,
            "interface_locked": self.interface_locked,
            "parent_width": int(parent.shape[1]),
            "child_width": int(child.shape[1]),
            "fit_rows": int(parent.shape[0]),
            "fit_provenance": fit_provenance,
            "fit_provenance_sha256": _sha256(fit_provenance),
            "row_alignment_sha256": row_alignment,
            "cached_parent_evidence_sha256": cached_parent,
            "live_child_recapture_sha256": live_child,
            "child_row_evidence_kind": child_row_evidence_kind,
            "child_row_evidence_sha256": child_row_evidence,
            "base_encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(encoder),
            "fit_parent_rows_sha256": fingerprint_lfm2_moving_gauge_tensor(parent),
            "fit_child_rows_sha256": fingerprint_lfm2_moving_gauge_tensor(child),
            "fit_weights_sha256": fingerprint_lfm2_moving_gauge_tensor(weight),
            "heldout": (
                None
                if heldout_parent is None
                else {
                    "rows": int(heldout_parent.shape[0]),
                    "provenance": heldout_provenance,
                    "provenance_sha256": _sha256(heldout_provenance),
                    "parent_rows_sha256": (
                        fingerprint_lfm2_moving_gauge_tensor(heldout_parent)
                    ),
                    "child_rows_sha256": (
                        fingerprint_lfm2_moving_gauge_tensor(heldout_child)
                    ),
                    "weights_sha256": (
                        fingerprint_lfm2_moving_gauge_tensor(heldout_weight)
                    ),
                    "used_by_solver": False,
                }
            ),
            "complete_full_prefix_rows": True,
            "cached_parent_only_no_recapture": True,
            "live_child_rows_already_recaptured": (
                child_row_evidence_kind == "live_child_recapture"
            ),
            "deferred_prior_analytic_macrocell_rows": (
                child_row_evidence_kind == "deferred_prior_analytic_macrocell"
            ),
            "execution": dict(_ZERO_EXECUTION),
        }
        supplied_report = dict(self.report)
        if supplied_report:
            if supplied_report != dict(_seal(body)):
                raise Lfm2MovingGaugeResidualAuthenticationError(
                    "Boundary evidence report is stale or malformed."
                )
            report = MappingProxyType(supplied_report)
        else:
            report = _seal(body)
        object.__setattr__(self, "boundary_id", boundary_id)
        object.__setattr__(self, "interface_role", role)
        object.__setattr__(self, "base_encoder", encoder)
        object.__setattr__(self, "fit_parent_rows", parent)
        object.__setattr__(self, "fit_child_rows", child)
        object.__setattr__(self, "fit_weights", weight)
        object.__setattr__(self, "fit_provenance", fit_provenance)
        object.__setattr__(self, "row_alignment_sha256", row_alignment)
        object.__setattr__(self, "cached_parent_evidence_sha256", cached_parent)
        object.__setattr__(self, "live_child_recapture_sha256", live_child)
        object.__setattr__(self, "child_row_evidence_kind", child_row_evidence_kind)
        object.__setattr__(self, "child_row_evidence_sha256", child_row_evidence)
        object.__setattr__(self, "heldout_parent_rows", heldout_parent)
        object.__setattr__(self, "heldout_child_rows", heldout_child)
        object.__setattr__(self, "heldout_weights", heldout_weight)
        object.__setattr__(self, "heldout_provenance", heldout_provenance)
        object.__setattr__(self, "report", report)

    @property
    def parent_width(self) -> int:
        return int(self.fit_parent_rows.shape[1])

    @property
    def child_width(self) -> int:
        return int(self.fit_child_rows.shape[1])

    def verify_fit_authentication(self) -> bool:
        """Verify the sealed report and FIT tensors without reading heldout rows."""

        try:
            return (
                verify_lfm2_moving_gauge_report(self.report)
                and self.report["base_encoder_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.base_encoder)
                and self.report["fit_parent_rows_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.fit_parent_rows)
                and self.report["fit_child_rows_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.fit_child_rows)
                and self.report["fit_weights_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.fit_weights)
            )
        except (KeyError, TypeError, ValueError):
            return False

    def verify_authentication(self) -> bool:
        """Verify FIT plus every supplied heldout tensor."""

        try:
            if not self.verify_fit_authentication():
                return False
            heldout_report = self.report.get("heldout")
            if self.heldout_parent_rows is None:
                return heldout_report is None
            return (
                isinstance(heldout_report, Mapping)
                and heldout_report.get("parent_rows_sha256")
                == fingerprint_lfm2_moving_gauge_tensor(self.heldout_parent_rows)
                and heldout_report.get("child_rows_sha256")
                == fingerprint_lfm2_moving_gauge_tensor(self.heldout_child_rows)
                and heldout_report.get("weights_sha256")
                == fingerprint_lfm2_moving_gauge_tensor(self.heldout_weights)
            )
        except (KeyError, TypeError, ValueError):
            return False


def lfm2_moving_gauge_sequential_fit_provenance(
    upstream_evidence: Lfm2MovingGaugeBoundaryEvidence,
    downstream_evidence: Lfm2MovingGaugeBoundaryEvidence,
) -> str:
    """Return the canonical lineage id for one sequential chart transport."""

    if (
        not isinstance(upstream_evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not upstream_evidence.verify_authentication()
        or not isinstance(downstream_evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not downstream_evidence.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Sequential moving-gauge provenance requires authenticated boundary "
            "evidence."
        )
    return f"{upstream_evidence.fit_provenance}:to:{downstream_evidence.boundary_id}"


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBoundaryChart:
    """One cycle-coupled CPCF chart bound to exact boundary evidence."""

    evidence_sha256: str
    chart: CpcfBoundaryChart
    orthogonal_rotation: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence_sha = _digest(self.evidence_sha256, "evidence_sha256")
        if not isinstance(self.chart, CpcfBoundaryChart):
            raise TypeError("chart must be CpcfBoundaryChart.")
        rotation = _matrix(
            self.orthogonal_rotation,
            "orthogonal_rotation",
            dtype=_torch().float64,
        )
        if tuple(rotation.shape) != (
            self.chart.target_width,
            self.chart.target_width,
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Boundary orthogonal rotation has incorrect geometry."
            )
        if not verify_lfm2_moving_gauge_report(self.report):
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Boundary chart report seal is invalid."
            )
        if (
            self.report.get("evidence_sha256") != evidence_sha
            or self.report.get("encoder_sha256")
            != fingerprint_lfm2_moving_gauge_tensor(self.chart.mapping.encoder)
            or self.report.get("decoder_sha256")
            != fingerprint_lfm2_moving_gauge_tensor(self.chart.mapping.decoder)
            or self.report.get("orthogonal_rotation_sha256")
            != fingerprint_lfm2_moving_gauge_tensor(rotation)
        ):
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Boundary chart tensors differ from their report."
            )
        object.__setattr__(self, "evidence_sha256", evidence_sha)
        object.__setattr__(self, "orthogonal_rotation", rotation)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def encoder(self) -> Any:
        return self.chart.mapping.encoder

    @property
    def decoder(self) -> Any:
        return self.chart.mapping.decoder

    def verify_authentication(self) -> bool:
        try:
            return (
                verify_lfm2_moving_gauge_report(self.report)
                and self.report["encoder_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.encoder)
                and self.report["decoder_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.decoder)
                and self.report["orthogonal_rotation_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.orthogonal_rotation)
            )
        except (KeyError, TypeError, ValueError):
            return False


def _fp64_svd_anchored_chart_solve(
    design: Any,
    target: Any,
    anchor: Any,
    weights: Any,
    *,
    ridge_ratio: float,
    absolute_ridge_floor: float,
    candidate_kind: str,
) -> tuple[Any, Mapping[str, Any]]:
    """Solve one diagnostic chart around a frozen anchor in FP64.

    The thin-SVD form leaves every unobserved design-null direction exactly at
    the anchor.  It is intentionally separate from physical writer publication:
    this helper produces diagnostic endpoint charts only.
    """

    torch = _torch()
    x = _matrix(design, "chart design", dtype=torch.float64)
    y = _matrix(target, "chart target", dtype=torch.float64)
    a = _matrix(anchor, "chart anchor", dtype=torch.float64)
    w = _weights(weights, int(x.shape[0]), "chart weights")
    if (
        int(x.shape[0]) != int(y.shape[0])
        or int(x.shape[1]) != int(a.shape[0])
        or int(y.shape[1]) != int(a.shape[1])
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Generalized chart solve geometry is inconsistent."
        )
    ratio = _finite_nonnegative(ridge_ratio, "ridge_ratio")
    floor = _finite_positive(absolute_ridge_floor, "absolute_ridge_floor")
    with torch.no_grad():
        root = torch.sqrt(w).unsqueeze(1)
        weighted_design = (x * root).contiguous()
        weighted_residual = ((y - x @ a) * root).contiguous()
        left, singular_values, right_h = torch.linalg.svd(
            weighted_design,
            full_matrices=False,
        )
        gram_trace = float(singular_values.square().sum().item())
        ridge = max(
            floor,
            ratio * gram_trace / max(1, int(x.shape[1])),
        )
        gains = singular_values / (singular_values.square() + ridge)
        projected_residual = left.T @ weighted_residual
        correction = (
            right_h.T @ (gains.unsqueeze(1) * projected_residual)
        ).contiguous()
        solution = (a + correction).contiguous()
        weighted_target = (y * root).contiguous()
        normal_residual = weighted_design.T @ (
            weighted_design @ solution - weighted_target
        ) + ridge * (solution - a)
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Generalized FP64 chart solution is non-finite."
        )
    largest = (
        float(singular_values.max().item()) if int(singular_values.numel()) else 0.0
    )
    rank_tolerance = (
        max(int(x.shape[0]), int(x.shape[1])) * torch.finfo(torch.float64).eps * largest
    )
    rank = int((singular_values > rank_tolerance).sum().item())
    full_column_rank = rank == int(x.shape[1])
    smallest = (
        float(singular_values.min().item())
        if full_column_rank and int(singular_values.numel())
        else 0.0
    )
    regularized_condition = (largest * largest + ridge) / (smallest * smallest + ridge)
    body = {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "fp64_thin_svd_anchored_chart_solve",
        "candidate_kind": _text(candidate_kind, "candidate_kind"),
        "equation": (
            "anchor+V*diag(s/(s^2+lambda))*U.T*sqrt(D)*(target-design@anchor)"
        ),
        "rows": int(x.shape[0]),
        "input_width": int(x.shape[1]),
        "output_width": int(y.shape[1]),
        "ridge_ratio": ratio,
        "ridge_absolute_floor": floor,
        "weighted_gram_trace": gram_trace,
        "effective_ridge": ridge,
        "effective_rank": rank,
        "rank_tolerance": rank_tolerance,
        "full_column_rank": full_column_rank,
        "largest_singular_value": largest,
        "smallest_full_column_rank_singular_value": (
            smallest if full_column_rank else None
        ),
        "design_condition_number": (
            largest / smallest if full_column_rank and smallest > 0.0 else None
        ),
        "regularized_normal_condition_number": regularized_condition,
        "design_sha256": fingerprint_lfm2_moving_gauge_tensor(x),
        "target_sha256": fingerprint_lfm2_moving_gauge_tensor(y),
        "anchor_sha256": fingerprint_lfm2_moving_gauge_tensor(a),
        "weights_sha256": fingerprint_lfm2_moving_gauge_tensor(w),
        "singular_values_sha256": fingerprint_lfm2_moving_gauge_tensor(singular_values),
        "correction_sha256": fingerprint_lfm2_moving_gauge_tensor(correction),
        "solution_sha256": fingerprint_lfm2_moving_gauge_tensor(solution),
        "correction_frobenius_norm": float(torch.linalg.matrix_norm(correction).item()),
        "anchor_frobenius_norm": float(torch.linalg.matrix_norm(a).item()),
        "anchored_normal_equation_residual_max_abs": float(
            normal_residual.abs().max().item()
        ),
        "null_directions_preserve_anchor_analytically": True,
        "bit_exact_null_direction_claimed": False,
        "fit_rows_only": True,
        "heldout_used_by_solver": False,
        "accumulation_dtype": "float64",
        "solve_dtype": "float64",
        "output_dtype": "float64",
        "closed_form_linear_solve": True,
        "autograd_used": False,
        "quality_gate": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    return solution, _seal(body)


def _weighted_row_gram_difference(
    left: Any,
    right: Any,
    weights: Any,
) -> Mapping[str, Any]:
    """Return a memory-bounded row-Gram invariant diagnostic."""

    torch = _torch()
    root = torch.sqrt(weights).unsqueeze(1)
    lhs = (left * root).contiguous()
    rhs = (right * root).contiguous()
    # If [lhs rhs] = Q R, orthogonal invariance gives
    # ||lhs lhs.T-rhs rhs.T||_F = ||R_l R_l.T-R_r R_r.T||_F.  Computing that
    # signed difference directly avoids subtracting three large scalar fourth-
    # order terms.  The reduced square dimension is k=min(n, 2d), so this
    # avoids an n-by-n matrix only when n > 2d; otherwise k equals n.
    _, reduced = torch.linalg.qr(torch.cat((lhs, rhs), dim=1), mode="reduced")
    width = int(lhs.shape[1])
    rows = int(lhs.shape[0])
    concatenated_width = int(lhs.shape[1] + rhs.shape[1])
    reduced_dimension = int(reduced.shape[0])
    n_by_n_materialized = reduced_dimension == rows
    reduced_lhs = reduced[:, :width]
    reduced_rhs = reduced[:, width:]
    signed_gram = (
        reduced_lhs @ reduced_lhs.T - reduced_rhs @ reduced_rhs.T
    ).contiguous()
    rhs_gram = (reduced_rhs @ reduced_rhs.T).contiguous()
    difference = float(torch.linalg.matrix_norm(signed_gram).item())
    reference = float(torch.linalg.matrix_norm(rhs_gram).item())
    return MappingProxyType(
        {
            "necessary_condition": "(parent@encoder)(parent@encoder).T=child@child.T",
            "weighted_row_gram_difference_frobenius": difference,
            "weighted_row_gram_relative_frobenius": difference
            / max(reference, torch.finfo(torch.float64).tiny),
            "maximum_row_norm_difference": float(
                (
                    torch.linalg.vector_norm(lhs, dim=1)
                    - torch.linalg.vector_norm(rhs, dim=1)
                )
                .abs()
                .max()
                .item()
            ),
            "row_count": rows,
            "concatenated_feature_width": concatenated_width,
            "reduced_qr_dimension": reduced_dimension,
            "materialized_square_dimension": int(signed_gram.shape[0]),
            "n_by_n_row_gram_materialized": n_by_n_materialized,
            "n_by_n_matrix_avoided": not n_by_n_materialized,
            "materialized_matrix_kind": "reduced_coordinate_signed_gram",
            "computation": "reduced_qr_direct_signed_row_gram_frobenius",
            "certificate_interpretation": (
                "a nonzero FP64 value is a necessary-invariant mismatch "
                "diagnostic; exact-arithmetic impossibility is not claimed "
                "without a separate numerical error proof"
            ),
            "formal_exact_arithmetic_certificate_claimed": False,
            "diagnostic_only": True,
            "numeric_gate": False,
        }
    )


def _chart_candidate_metrics(
    parent: Any,
    child: Any,
    weights: Any,
    encoder: Any,
) -> Mapping[str, Any]:
    torch = _torch()
    prediction = (parent @ encoder).contiguous()
    tiny = torch.finfo(torch.float64).tiny

    def rms_direction(rows: Any) -> Any:
        scale = torch.sqrt(rows.square().mean(dim=1, keepdim=True)).clamp_min(tiny)
        return (rows / scale).contiguous()

    prediction_direction = rms_direction(prediction)
    child_direction = rms_direction(child)
    return MappingProxyType(
        {
            "weighted_rmse": _weighted_rmse(prediction, child, weights),
            "relative_rmse": _relative_rmse(prediction, child, weights),
            "pre_gain_rms_direction_weighted_rmse": _weighted_rmse(
                prediction_direction,
                child_direction,
                weights,
            ),
            "pre_gain_rms_direction_relative_rmse": _relative_rmse(
                prediction_direction,
                child_direction,
                weights,
            ),
            "rms_direction_definition": "row/sqrt(mean(row^2));zero-clamped",
            "learned_rmsnorm_gain_applied": False,
            "prediction_sha256": fingerprint_lfm2_moving_gauge_tensor(prediction),
        }
    )


def _encoder_geometry(value: Any) -> Mapping[str, Any]:
    torch = _torch()
    singular = torch.linalg.svdvals(value)
    largest = float(singular.max().item())
    tolerance = (
        max(int(value.shape[0]), int(value.shape[1]))
        * torch.finfo(torch.float64).eps
        * largest
    )
    rank = int((singular > tolerance).sum().item())
    full_column_rank = rank == int(value.shape[1])
    smallest = float(singular.min().item()) if full_column_rank else None
    return MappingProxyType(
        {
            "rank": rank,
            "rank_tolerance": tolerance,
            "full_column_rank": full_column_rank,
            "largest_singular_value": largest,
            "smallest_singular_value": smallest,
            "condition_number": (
                largest / smallest if smallest is not None and smallest > 0.0 else None
            ),
            "singular_values_sha256": fingerprint_lfm2_moving_gauge_tensor(singular),
        }
    )


def _chart_correction_decomposition(
    base_encoder: Any,
    orthogonal_encoder: Any,
    candidate_encoder: Any,
) -> Mapping[str, Any]:
    """Separate correction inside and outside the base source subspace."""

    torch = _torch()
    base_left, singular, _ = torch.linalg.svd(
        base_encoder,
        full_matrices=False,
    )
    largest = float(singular.max().item())
    tolerance = (
        max(int(base_encoder.shape[0]), int(base_encoder.shape[1]))
        * torch.finfo(torch.float64).eps
        * largest
    )
    rank = int((singular > tolerance).sum().item())
    basis = base_left[:, :rank]
    correction = (candidate_encoder - orthogonal_encoder).contiguous()
    inside = (basis @ (basis.T @ correction)).contiguous()
    outside = (correction - inside).contiguous()
    total_norm = float(torch.linalg.matrix_norm(correction).item())
    inside_norm = float(torch.linalg.matrix_norm(inside).item())
    outside_norm = float(torch.linalg.matrix_norm(outside).item())
    denominator = max(total_norm, torch.finfo(torch.float64).tiny)
    return MappingProxyType(
        {
            "base_encoder_rank": rank,
            "correction_frobenius_norm": total_norm,
            "inside_base_source_subspace_frobenius_norm": inside_norm,
            "outside_base_source_subspace_frobenius_norm": outside_norm,
            "inside_base_source_subspace_fraction": inside_norm / denominator,
            "outside_base_source_subspace_fraction": outside_norm / denominator,
            "correction_sha256": fingerprint_lfm2_moving_gauge_tensor(correction),
            "inside_correction_sha256": fingerprint_lfm2_moving_gauge_tensor(inside),
            "outside_correction_sha256": fingerprint_lfm2_moving_gauge_tensor(outside),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeGeneralizedChartDiagnostic:
    """Parallel intermediate-chart candidates with no publication authority."""

    evidence_sha256: str
    orthogonal_chart: Lfm2MovingGaugeBoundaryChart
    scale_shear_transform: Any = field(repr=False, compare=False)
    scale_shear_encoder: Any = field(repr=False, compare=False)
    generalized_encoder: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence_sha = _digest(self.evidence_sha256, "evidence_sha256")
        if (
            not isinstance(self.orthogonal_chart, Lfm2MovingGaugeBoundaryChart)
            or not self.orthogonal_chart.verify_authentication()
            or self.orthogonal_chart.evidence_sha256 != evidence_sha
        ):
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Generalized diagnostic requires an authenticated orthogonal chart."
            )
        torch = _torch()
        transform = _matrix(
            self.scale_shear_transform,
            "scale_shear_transform",
            dtype=torch.float64,
        )
        scale_shear = _matrix(
            self.scale_shear_encoder,
            "scale_shear_encoder",
            dtype=torch.float64,
        )
        generalized = _matrix(
            self.generalized_encoder,
            "generalized_encoder",
            dtype=torch.float64,
        )
        if (
            tuple(transform.shape)
            != (
                self.orthogonal_chart.chart.target_width,
                self.orthogonal_chart.chart.target_width,
            )
            or tuple(scale_shear.shape) != tuple(self.orthogonal_chart.encoder.shape)
            or tuple(generalized.shape) != tuple(self.orthogonal_chart.encoder.shape)
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Generalized diagnostic candidate geometry is inconsistent."
            )
        if not verify_lfm2_moving_gauge_report(self.report):
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Generalized diagnostic report seal is invalid."
            )
        if (
            self.report.get("evidence_sha256") != evidence_sha
            or self.report.get("orthogonal_chart_sha256")
            != self.orthogonal_chart.report["report_sha256"]
            or self.report.get("scale_shear_transform_sha256")
            != fingerprint_lfm2_moving_gauge_tensor(transform)
            or self.report.get("scale_shear_encoder_sha256")
            != fingerprint_lfm2_moving_gauge_tensor(scale_shear)
            or self.report.get("generalized_encoder_sha256")
            != fingerprint_lfm2_moving_gauge_tensor(generalized)
        ):
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Generalized diagnostic tensors differ from their report."
            )
        object.__setattr__(self, "evidence_sha256", evidence_sha)
        object.__setattr__(self, "scale_shear_transform", transform)
        object.__setattr__(self, "scale_shear_encoder", scale_shear)
        object.__setattr__(self, "generalized_encoder", generalized)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        try:
            return (
                verify_lfm2_moving_gauge_report(self.report)
                and self.orthogonal_chart.verify_authentication()
                and self.orthogonal_chart.evidence_sha256 == self.evidence_sha256
                and self.report["orthogonal_chart_sha256"]
                == self.orthogonal_chart.report["report_sha256"]
                and self.report["scale_shear_transform_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.scale_shear_transform)
                and self.report["scale_shear_encoder_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.scale_shear_encoder)
                and self.report["generalized_encoder_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.generalized_encoder)
            )
        except (KeyError, TypeError, ValueError):
            return False


def _predictive_mapping(
    encoder: Any,
    decoder: Any,
    *,
    kind: str,
    lineage: Mapping[str, Any],
) -> PredictiveBoundaryMap:
    body = {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "component": kind,
        "source_width": int(encoder.shape[0]),
        "target_width": int(encoder.shape[1]),
        "encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(encoder),
        "decoder_sha256": fingerprint_lfm2_moving_gauge_tensor(decoder),
        "feature_predictors": {},
        "lineage": dict(lineage),
        "fit_only": True,
        "training_steps": 0,
    }
    return PredictiveBoundaryMap(
        encoder=encoder,
        decoder=decoder,
        feature_predictors={},
        report={**body, "report_sha256": _sha256(body)},
    )


def _cpcf_chart(
    name: str,
    mapping: PredictiveBoundaryMap,
    *,
    kind: str,
    lineage: Mapping[str, Any],
) -> CpcfBoundaryChart:
    body = {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "name": name,
        "component": kind,
        "source_width": mapping.source_width,
        "target_width": mapping.target_width,
        "chart_transition_residual_formula": (
            "encoded_parent_endpoint_minus_actual_child_pre_state"
        ),
        "identity_skip_path_accounted_for": True,
        "predictive_mapping": mapping.as_dict(),
        "lineage": dict(lineage),
        "training_steps": 0,
    }
    return CpcfBoundaryChart(
        name=name,
        mapping=mapping,
        report={**body, "report_sha256": _sha256(body)},
    )


def fit_lfm2_cycle_coupled_moving_gauge_boundary(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    *,
    policy: Lfm2MovingGaugeReclosurePolicy | None = None,
    evaluate_heldout: bool = True,
) -> Lfm2MovingGaugeBoundaryChart:
    """Fit one input/output chart pair using only sealed FIT evidence.

    The base parent-to-child encoder is changed only by the mature CPCF
    orthogonal Procrustes transport.  A separate child-to-parent decoder is a
    deterministic ridge fit with a regularized reciprocal anchor.  That anchor
    is the closed-form cycle-consistency term; no gradient optimization occurs.
    """

    torch = _torch()
    if not isinstance(evaluate_heldout, bool):
        raise TypeError("evaluate_heldout must be boolean.")
    evidence_authenticated = isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence) and (
        evidence.verify_authentication()
        if evaluate_heldout
        else evidence.verify_fit_authentication()
    )
    if not evidence_authenticated:
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Boundary evidence failed authentication."
        )
    active_policy = policy or Lfm2MovingGaugeReclosurePolicy()
    if not isinstance(active_policy, Lfm2MovingGaugeReclosurePolicy) or not (
        active_policy.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Moving-gauge policy failed authentication."
        )
    base_encoded = evidence.fit_parent_rows @ evidence.base_encoder
    base_reciprocal, reciprocal_report = _regularized_reciprocal(
        evidence.base_encoder,
        ridge_ratio=active_policy.chart_ridge_ratio,
        absolute_ridge_floor=active_policy.absolute_ridge_floor,
    )
    base_decoder, base_decoder_report = _anchored_ridge_replacement(
        base_encoded,
        evidence.fit_parent_rows,
        base_reciprocal,
        evidence.fit_weights,
        ridge_ratio=(
            active_policy.chart_ridge_ratio + active_policy.cycle_consistency_weight
        ),
        absolute_ridge_floor=active_policy.absolute_ridge_floor,
    )
    lineage = {
        "boundary_evidence_sha256": evidence.report["report_sha256"],
        "cached_parent_evidence_sha256": (evidence.cached_parent_evidence_sha256),
        "live_child_recapture_sha256": evidence.live_child_recapture_sha256,
    }
    base_mapping = _predictive_mapping(
        evidence.base_encoder,
        base_decoder,
        kind="cycle_anchored_base_boundary_mapping",
        lineage=lineage,
    )
    base_chart = _cpcf_chart(
        evidence.boundary_id,
        base_mapping,
        kind="cycle_anchored_base_boundary_chart",
        lineage=lineage,
    )
    if evidence.interface_locked:
        rotation = torch.eye(evidence.child_width, dtype=torch.float64)
        transported_encoder = evidence.base_encoder
        transport_report = None
    else:
        transport = fit_cpcf_orthogonal_chart_transport(
            base_chart,
            fit_actual_child_rows=evidence.fit_child_rows,
            fit_provenance=evidence.fit_provenance,
            fit_parent_states=evidence.fit_parent_rows,
            fit_weights=evidence.fit_weights,
            **(
                {
                    "heldout_actual_child_rows": evidence.heldout_child_rows,
                    "heldout_provenance": evidence.heldout_provenance,
                    "heldout_parent_states": evidence.heldout_parent_rows,
                    "heldout_weights": evidence.heldout_weights,
                }
                if evidence.heldout_parent_rows is not None and evaluate_heldout
                else {}
            ),
            orthogonality_tolerance=active_policy.orthogonality_tolerance,
        )
        rotation = transport.rotation
        transported_encoder = transport.mapping.encoder
        transport_report = dict(transport.report)
    reciprocal, final_reciprocal_report = _regularized_reciprocal(
        transported_encoder,
        ridge_ratio=active_policy.chart_ridge_ratio,
        absolute_ridge_floor=active_policy.absolute_ridge_floor,
    )
    uncoupled_decoder, uncoupled_report = _anchored_ridge_replacement(
        evidence.fit_child_rows,
        evidence.fit_parent_rows,
        torch.zeros_like(reciprocal),
        evidence.fit_weights,
        ridge_ratio=active_policy.chart_ridge_ratio,
        absolute_ridge_floor=active_policy.absolute_ridge_floor,
    )
    data_scale = _ridge_scale(evidence.fit_child_rows, evidence.fit_weights)
    chart_ridge = max(
        active_policy.absolute_ridge_floor,
        active_policy.chart_ridge_ratio * data_scale,
    )
    cycle_ridge = active_policy.cycle_consistency_weight * data_scale
    total_ridge = chart_ridge + cycle_ridge
    reciprocal_fraction = cycle_ridge / total_ridge
    coupled_anchor = reciprocal_fraction * reciprocal
    coupled_decoder, coupled_report = _anchored_ridge_replacement(
        evidence.fit_child_rows,
        evidence.fit_parent_rows,
        coupled_anchor,
        evidence.fit_weights,
        ridge_ratio=total_ridge / data_scale,
        absolute_ridge_floor=active_policy.absolute_ridge_floor,
    )
    final_mapping = _predictive_mapping(
        transported_encoder,
        coupled_decoder,
        kind="orthogonal_encoder_cycle_coupled_decoder_mapping",
        lineage=lineage,
    )
    final_chart = _cpcf_chart(
        evidence.boundary_id,
        final_mapping,
        kind="moving_gauge_cycle_coupled_boundary_chart",
        lineage=lineage,
    )

    def cycle_metrics(
        parent: Any, child: Any, weights: Any, decoder: Any
    ) -> dict[str, Any]:
        encoded = parent @ transported_encoder
        return {
            "parent_cycle_relative_rmse": _relative_rmse(
                encoded @ decoder,
                parent,
                weights,
            ),
            "child_cycle_relative_rmse": _relative_rmse(
                (child @ decoder) @ transported_encoder,
                child,
                weights,
            ),
            "parent_to_child_relative_rmse": _relative_rmse(
                encoded,
                child,
                weights,
            ),
            "child_to_parent_relative_rmse": _relative_rmse(
                child @ decoder,
                parent,
                weights,
            ),
        }

    fit_uncoupled = cycle_metrics(
        evidence.fit_parent_rows,
        evidence.fit_child_rows,
        evidence.fit_weights,
        uncoupled_decoder,
    )
    fit_coupled = cycle_metrics(
        evidence.fit_parent_rows,
        evidence.fit_child_rows,
        evidence.fit_weights,
        coupled_decoder,
    )
    heldout_metrics = None
    if evidence.heldout_parent_rows is not None and evaluate_heldout:
        heldout_metrics = {
            "uncoupled": cycle_metrics(
                evidence.heldout_parent_rows,
                evidence.heldout_child_rows,
                evidence.heldout_weights,
                uncoupled_decoder,
            ),
            "coupled": cycle_metrics(
                evidence.heldout_parent_rows,
                evidence.heldout_child_rows,
                evidence.heldout_weights,
                coupled_decoder,
            ),
            "used_by_solver": False,
            "read_after_chart_freeze": True,
        }
    identity = torch.eye(evidence.child_width, dtype=torch.float64)
    orthogonality_error = float((rotation.T @ rotation - identity).abs().max().item())
    body = {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "cycle_coupled_moving_gauge_boundary_chart",
        "implementation": _IMPLEMENTATION,
        "boundary_id": evidence.boundary_id,
        "interface_role": evidence.interface_role,
        "interface_locked": evidence.interface_locked,
        "evidence_sha256": evidence.report["report_sha256"],
        "policy_sha256": active_policy.authentication_sha256,
        "source_width": evidence.parent_width,
        "target_width": evidence.child_width,
        "row_convention": "parent_rows@encoder;child_rows@decoder",
        "separate_input_output_maps": True,
        "encoder_role": "parent_boundary_to_child_boundary",
        "decoder_role": "child_boundary_to_parent_boundary",
        "encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(transported_encoder),
        "decoder_sha256": fingerprint_lfm2_moving_gauge_tensor(coupled_decoder),
        "base_encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(
            evidence.base_encoder
        ),
        "orthogonal_rotation_sha256": fingerprint_lfm2_moving_gauge_tensor(rotation),
        "orthogonality_max_abs_error": orthogonality_error,
        "encoder_change_from_base": (
            "none_interface_locked"
            if evidence.interface_locked
            else "right_orthogonal_procrustes_only"
        ),
        "cpcf_chart_transition_residual_formula": (
            "encoded_parent_endpoint_minus_actual_child_pre_state"
        ),
        "base_reciprocal_report_sha256": reciprocal_report["report_sha256"],
        "base_decoder_report_sha256": base_decoder_report["report_sha256"],
        "transport_report": transport_report,
        "final_reciprocal_report_sha256": final_reciprocal_report["report_sha256"],
        "uncoupled_decoder_report_sha256": uncoupled_report["report_sha256"],
        "coupled_decoder_report_sha256": coupled_report["report_sha256"],
        "cycle_consistency": {
            "method": "closed_form_regularized_reciprocal_anchor",
            "weight": active_policy.cycle_consistency_weight,
            "data_ridge": chart_ridge,
            "cycle_ridge": cycle_ridge,
            "reciprocal_anchor_fraction": reciprocal_fraction,
            "gradient_optimization": False,
            "fit_uncoupled": fit_uncoupled,
            "fit_coupled": fit_coupled,
            "heldout": heldout_metrics,
            "all_metrics_diagnostic_only": True,
        },
        "fit_only": True,
        "heldout_used_by_solver": False,
        "heldout_evaluation_requested": evaluate_heldout,
        "numeric_quality_gate": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    return Lfm2MovingGaugeBoundaryChart(
        evidence_sha256=evidence.report["report_sha256"],
        chart=final_chart,
        orthogonal_rotation=rotation,
        report=_seal(body),
    )


def fit_lfm2_generalized_moving_gauge_chart_diagnostic(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    *,
    policy: Lfm2MovingGaugeReclosurePolicy | None = None,
    orthogonal_chart: Lfm2MovingGaugeBoundaryChart | None = None,
) -> Lfm2MovingGaugeGeneralizedChartDiagnostic:
    """Fit parallel non-orthogonal intermediate charts without selecting one.

    The existing right-orthogonal chart remains the default and is retained as
    the immutable anchor.  Candidate one keeps the base encoder's source
    subspace and admits a full child-coordinate scale/shear transform
    (``E0@A``).  Candidate two admits a full source-to-child map.  Both are
    deterministic FP64 anchored solves over FIT rows only.  Heldout rows are
    read only after candidate tensors and solver receipts are frozen.
    """

    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not evidence.verify_fit_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized chart evidence failed authentication."
        )
    if evidence.interface_locked or evidence.interface_role != "intermediate":
        raise Lfm2MovingGaugeResidualReclosureError(
            "Generalized charts are diagnostic-only for unlocked intermediate "
            "boundaries; input and terminal interfaces remain locked."
        )
    active_policy = policy or Lfm2MovingGaugeReclosurePolicy()
    if (
        not isinstance(active_policy, Lfm2MovingGaugeReclosurePolicy)
        or not active_policy.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized chart policy failed authentication."
        )
    orthogonal = orthogonal_chart or fit_lfm2_cycle_coupled_moving_gauge_boundary(
        evidence,
        policy=active_policy,
        evaluate_heldout=False,
    )
    if (
        not isinstance(orthogonal, Lfm2MovingGaugeBoundaryChart)
        or not orthogonal.verify_authentication()
        or orthogonal.evidence_sha256 != evidence.report["report_sha256"]
        or orthogonal.report.get("policy_sha256") != active_policy.authentication_sha256
        or orthogonal.report.get("heldout_evaluation_requested") is not False
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Supplied orthogonal chart is not bound to this evidence and policy."
        )

    # FIT-only solver boundary.  Do not access any heldout tensor above the
    # candidate/hash freeze immediately below.
    parent = evidence.fit_parent_rows
    child = evidence.fit_child_rows
    weights = evidence.fit_weights
    base = evidence.base_encoder
    base_encoded = (parent @ base).contiguous()
    scale_shear_transform, scale_shear_solve = _fp64_svd_anchored_chart_solve(
        base_encoded,
        child,
        orthogonal.orthogonal_rotation,
        weights,
        ridge_ratio=active_policy.chart_ridge_ratio,
        absolute_ridge_floor=active_policy.absolute_ridge_floor,
        candidate_kind="fixed_source_subspace_scale_shear",
    )
    scale_shear_encoder = (base @ scale_shear_transform).contiguous()
    generalized_encoder, generalized_solve = _fp64_svd_anchored_chart_solve(
        parent,
        child,
        orthogonal.encoder,
        weights,
        ridge_ratio=active_policy.chart_ridge_ratio,
        absolute_ridge_floor=active_policy.absolute_ridge_floor,
        candidate_kind="unrestricted_source_to_child",
    )
    frozen_candidate_hashes = {
        "orthogonal": fingerprint_lfm2_moving_gauge_tensor(orthogonal.encoder),
        "scale_shear_transform": fingerprint_lfm2_moving_gauge_tensor(
            scale_shear_transform
        ),
        "scale_shear": fingerprint_lfm2_moving_gauge_tensor(scale_shear_encoder),
        "generalized": fingerprint_lfm2_moving_gauge_tensor(generalized_encoder),
    }
    fit_diagnostics = {
        "orthogonal": dict(
            _chart_candidate_metrics(
                parent,
                child,
                weights,
                orthogonal.encoder,
            )
        ),
        "scale_shear": dict(
            _chart_candidate_metrics(
                parent,
                child,
                weights,
                scale_shear_encoder,
            )
        ),
        "generalized": dict(
            _chart_candidate_metrics(
                parent,
                child,
                weights,
                generalized_encoder,
            )
        ),
    }
    gram_certificate = dict(_weighted_row_gram_difference(base_encoded, child, weights))
    rotated_gram_preservation = dict(
        _weighted_row_gram_difference(
            base_encoded,
            parent @ orthogonal.encoder,
            weights,
        )
    )

    # The explicit solver barrier is above this line.  Heldout tensors affect
    # diagnostics and the outer report seal only; they cannot affect candidates.
    heldout_diagnostics = None
    if evidence.heldout_parent_rows is not None:
        if not evidence.verify_authentication():
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Heldout boundary tensors changed after evidence authentication."
            )
        heldout_parent = _matrix(
            evidence.heldout_parent_rows,
            "heldout_parent_rows_snapshot",
            dtype=_torch().float64,
        )
        heldout_child = _matrix(
            evidence.heldout_child_rows,
            "heldout_child_rows_snapshot",
            dtype=_torch().float64,
        )
        heldout_weights = _weights(
            evidence.heldout_weights,
            int(heldout_parent.shape[0]),
            "heldout_weights_snapshot",
        )
        heldout_diagnostics = {
            "provenance": evidence.heldout_provenance,
            "orthogonal": dict(
                _chart_candidate_metrics(
                    heldout_parent,
                    heldout_child,
                    heldout_weights,
                    orthogonal.encoder,
                )
            ),
            "scale_shear": dict(
                _chart_candidate_metrics(
                    heldout_parent,
                    heldout_child,
                    heldout_weights,
                    scale_shear_encoder,
                )
            ),
            "generalized": dict(
                _chart_candidate_metrics(
                    heldout_parent,
                    heldout_child,
                    heldout_weights,
                    generalized_encoder,
                )
            ),
            "used_by_solver": False,
            "read_after_candidate_hash_freeze": True,
        }

    body = {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "parallel_generalized_moving_gauge_chart_diagnostic",
        "implementation": _IMPLEMENTATION,
        "boundary_id": evidence.boundary_id,
        "interface_role": evidence.interface_role,
        "interface_locked": evidence.interface_locked,
        "evidence_sha256": evidence.report["report_sha256"],
        "policy_sha256": active_policy.authentication_sha256,
        "orthogonal_chart_sha256": orthogonal.report["report_sha256"],
        "orthogonal_encoder_sha256": frozen_candidate_hashes["orthogonal"],
        "scale_shear_transform_sha256": frozen_candidate_hashes[
            "scale_shear_transform"
        ],
        "scale_shear_encoder_sha256": frozen_candidate_hashes["scale_shear"],
        "generalized_encoder_sha256": frozen_candidate_hashes["generalized"],
        "source_width": evidence.parent_width,
        "target_width": evidence.child_width,
        "default_chart_mode": "right_orthogonal_procrustes_only",
        "orthogonal_default_replaced": False,
        "candidate_publication_authority": False,
        "candidate_modes": {
            "scale_shear": "base_encoder@full_child_coordinate_transform",
            "generalized": "full_source_to_child_encoder",
        },
        "solver_reports": {
            "scale_shear": dict(scale_shear_solve),
            "generalized": dict(generalized_solve),
        },
        "gram_invariant_certificate": gram_certificate,
        "orthogonal_gram_preservation_certificate": rotated_gram_preservation,
        "encoder_geometry": {
            "orthogonal": dict(_encoder_geometry(orthogonal.encoder)),
            "scale_shear": dict(_encoder_geometry(scale_shear_encoder)),
            "generalized": dict(_encoder_geometry(generalized_encoder)),
        },
        "correction_decomposition": {
            "scale_shear": dict(
                _chart_correction_decomposition(
                    base,
                    orthogonal.encoder,
                    scale_shear_encoder,
                )
            ),
            "generalized": dict(
                _chart_correction_decomposition(
                    base,
                    orthogonal.encoder,
                    generalized_encoder,
                )
            ),
        },
        "diagnostics": {
            "fit": fit_diagnostics,
            "heldout": heldout_diagnostics,
            "all_metrics_diagnostic_only": True,
            "manual_semantic_review_remains_authoritative": True,
        },
        "fit_only_solver_inputs": {
            "parent_rows_sha256": fingerprint_lfm2_moving_gauge_tensor(parent),
            "child_rows_sha256": fingerprint_lfm2_moving_gauge_tensor(child),
            "weights_sha256": fingerprint_lfm2_moving_gauge_tensor(weights),
        },
        "candidate_hashes_frozen_before_heldout_read": frozen_candidate_hashes,
        "heldout_used_by_solver": False,
        "heldout_read_after_candidate_freeze": heldout_diagnostics is not None,
        "runtime_sidecar_required": False,
        "physical_jacobian_claimed": False,
        "closed_form_solves_only": True,
        "numeric_quality_gate": False,
        "decoded_intelligence_gate": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    return Lfm2MovingGaugeGeneralizedChartDiagnostic(
        evidence_sha256=evidence.report["report_sha256"],
        orthogonal_chart=orthogonal,
        scale_shear_transform=scale_shear_transform,
        scale_shear_encoder=scale_shear_encoder,
        generalized_encoder=generalized_encoder,
        report=_seal(body),
    )


def fit_lfm2_moving_gauge_residual_chart_adapter(
    *,
    boundary_id: str,
    interface_role: str,
    interface_locked: bool,
    base_producer_encoder: Any,
    fit_parent_rows: Any,
    fit_live_child_rows: Any,
    fit_weights: Any,
    fit_provenance: str,
    row_alignment_sha256: str,
    cached_parent_evidence_sha256: str,
    live_child_recapture_sha256: str,
    heldout_parent_rows: Any | None = None,
    heldout_live_child_rows: Any | None = None,
    heldout_weights: Any | None = None,
    heldout_provenance: str | None = None,
    policy: Lfm2MovingGaugeReclosurePolicy | None = None,
) -> Lfm2MovingGaugeBoundaryChart:
    """Additive adapter for ``_fit_residual_chart`` in the stack compiler.

    The caller supplies the already selected cached-parent and fresh live-child
    full-prefix rows.  This function neither imports the semantic adapter nor
    mutates the child, so integration can replace the forced-identity branch
    without a dependency cycle.
    """

    evidence = Lfm2MovingGaugeBoundaryEvidence(
        boundary_id=boundary_id,
        interface_role=interface_role,
        interface_locked=interface_locked,
        base_encoder=base_producer_encoder,
        fit_parent_rows=fit_parent_rows,
        fit_child_rows=fit_live_child_rows,
        fit_weights=fit_weights,
        fit_provenance=fit_provenance,
        row_alignment_sha256=row_alignment_sha256,
        cached_parent_evidence_sha256=cached_parent_evidence_sha256,
        live_child_recapture_sha256=live_child_recapture_sha256,
        heldout_parent_rows=heldout_parent_rows,
        heldout_child_rows=heldout_live_child_rows,
        heldout_weights=heldout_weights,
        heldout_provenance=heldout_provenance,
    )
    return fit_lfm2_cycle_coupled_moving_gauge_boundary(
        evidence,
        policy=policy,
    )


def transport_lfm2_cycle_coupled_boundary_from_upstream(
    upstream: Lfm2MovingGaugeBoundaryChart,
    downstream: Lfm2MovingGaugeBoundaryChart,
    *,
    downstream_evidence: Lfm2MovingGaugeBoundaryEvidence,
    fit_actual_child_upstream_rows: Any,
    fit_weights: Any,
    fit_provenance: str,
) -> Lfm2MovingGaugeBoundaryChart:
    """Apply the mature CPCF sequential gauge transport to one boundary.

    The input and terminal interfaces stay pinned.  Intermediate encoder and
    decoder maps are transported together, so decode/cycle behavior is
    invariant while the downstream endpoint chart is aligned to the actual
    preceding live-child full-prefix rows.
    """

    torch = _torch()
    if (
        not isinstance(upstream, Lfm2MovingGaugeBoundaryChart)
        or not upstream.verify_authentication()
        or not isinstance(downstream, Lfm2MovingGaugeBoundaryChart)
        or not downstream.verify_authentication()
        or not isinstance(
            downstream_evidence,
            Lfm2MovingGaugeBoundaryEvidence,
        )
        or not downstream_evidence.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Sequential moving-gauge inputs failed authentication."
        )
    if downstream.evidence_sha256 != downstream_evidence.report["report_sha256"]:
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Sequential downstream chart names different boundary evidence."
        )
    if downstream_evidence.interface_locked:
        return downstream
    actual = _matrix(
        fit_actual_child_upstream_rows,
        "fit_actual_child_upstream_rows",
        dtype=torch.float64,
    )
    weights = _weights(fit_weights, int(actual.shape[0]), "fit_weights")
    sequential = fit_cpcf_sequential_downstream_chart_transport(
        upstream.chart,
        downstream.chart,
        fit_parent_downstream_states=downstream_evidence.fit_parent_rows,
        fit_actual_child_upstream_rows=actual,
        fit_provenance=fit_provenance,
        fit_weights=weights,
    )
    rotation = (
        downstream.orthogonal_rotation @ sequential.downstream_transport.rotation
    ).contiguous()
    source_report = dict(downstream.report)
    source_cycle = source_report.get("cycle_consistency")
    body = {
        **{
            key: value
            for key, value in source_report.items()
            if key not in {"report_sha256", "encoder_sha256", "decoder_sha256"}
        },
        "kind": "sequentially_transported_cycle_coupled_boundary_chart",
        "encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(
            sequential.downstream_chart.mapping.encoder
        ),
        "decoder_sha256": fingerprint_lfm2_moving_gauge_tensor(
            sequential.downstream_chart.mapping.decoder
        ),
        "orthogonal_rotation_sha256": fingerprint_lfm2_moving_gauge_tensor(rotation),
        "source_boundary_chart_sha256": downstream.report["report_sha256"],
        "sequential_transport_report_sha256": sequential.report["report_sha256"],
        "sequential_transport_report": dict(sequential.report),
        "sequential_transport_reused_mature_cpcf_primitive": True,
        "separate_input_output_maps_transported_together": True,
        "cycle_consistency": {
            "source_chart": source_cycle,
            "orthogonal_transport_preserves_encoder_decoder_composition": True,
            "gradient_optimization": False,
        },
    }
    return Lfm2MovingGaugeBoundaryChart(
        evidence_sha256=downstream.evidence_sha256,
        chart=sequential.downstream_chart,
        orthogonal_rotation=rotation,
        report=_seal(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeWriterEvidence:
    """One dense writer (live or deferred) and its frozen FIT input rows."""

    parameter_id: str
    transition_index: int
    seed_weight: Any = field(repr=False, compare=False)
    fit_input_rows: Any = field(repr=False, compare=False)
    row_alignment_sha256: str
    live_child_recapture_sha256: str
    boundary_evidence_sha256: str | None = None
    publication_kind: str = "live_physical_parameter"
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        torch = _torch()
        parameter_id = _text(self.parameter_id, "parameter_id")
        if (
            isinstance(self.transition_index, bool)
            or not isinstance(self.transition_index, int)
            or self.transition_index < 0
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "transition_index must be a nonnegative integer."
            )
        seed = _matrix(self.seed_weight, "seed_weight", require_plain=True)
        if _dtype_name(seed) not in _SUPPORTED_TARGET_DTYPES:
            raise Lfm2MovingGaugeResidualReclosureError(
                "Physical writer seed dtype must be FP32 or BF16."
            )
        rows = _matrix(
            self.fit_input_rows,
            "fit_input_rows",
            dtype=torch.float64,
        )
        if int(rows.shape[1]) != int(seed.shape[1]):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Writer input rows and seed input width differ."
            )
        alignment = _digest(self.row_alignment_sha256, "row_alignment_sha256")
        live_child = _digest(
            self.live_child_recapture_sha256,
            "live_child_recapture_sha256",
        )
        boundary_evidence = (
            None
            if self.boundary_evidence_sha256 is None
            else _digest(
                self.boundary_evidence_sha256,
                "boundary_evidence_sha256",
            )
        )
        publication_kind = str(self.publication_kind).strip().lower()
        if publication_kind not in _WRITER_PUBLICATION_KINDS:
            raise Lfm2MovingGaugeResidualReclosureError(
                "publication_kind must be live_physical_parameter or "
                "deferred_dense_prior."
            )
        body = {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "physical_writer_fit_evidence",
            "parameter_id": parameter_id,
            "transition_index": self.transition_index,
            "output_width": int(seed.shape[0]),
            "input_width": int(seed.shape[1]),
            "fit_rows": int(rows.shape[0]),
            "target_dtype": _dtype_name(seed),
            "row_alignment_sha256": alignment,
            "live_child_recapture_sha256": live_child,
            "boundary_evidence_sha256": boundary_evidence,
            "publication_kind": publication_kind,
            "seed_weight_sha256": fingerprint_lfm2_moving_gauge_tensor(seed),
            "fit_input_rows_sha256": fingerprint_lfm2_moving_gauge_tensor(rows),
            "existing_physical_parameter": (
                publication_kind == "live_physical_parameter"
            ),
            "deferred_dense_prior": publication_kind == "deferred_dense_prior",
            "runtime_sidecar": False,
            "execution": dict(_ZERO_EXECUTION),
        }
        supplied = dict(self.report)
        if supplied:
            if supplied != dict(_seal(body)):
                raise Lfm2MovingGaugeResidualAuthenticationError(
                    "Writer evidence report is stale or malformed."
                )
            report = MappingProxyType(supplied)
        else:
            report = _seal(body)
        object.__setattr__(self, "parameter_id", parameter_id)
        object.__setattr__(self, "seed_weight", seed)
        object.__setattr__(self, "fit_input_rows", rows)
        object.__setattr__(self, "row_alignment_sha256", alignment)
        object.__setattr__(self, "live_child_recapture_sha256", live_child)
        object.__setattr__(self, "boundary_evidence_sha256", boundary_evidence)
        object.__setattr__(self, "publication_kind", publication_kind)
        object.__setattr__(self, "report", report)

    def verify_authentication(self) -> bool:
        try:
            return (
                verify_lfm2_moving_gauge_report(self.report)
                and self.report["seed_weight_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.seed_weight)
                and self.report["fit_input_rows_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(self.fit_input_rows)
            )
        except (KeyError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeTransition:
    """Explicit ``E_(l+1) D_l`` plus its norm-safe polar split."""

    transition_index: int
    full_transition: Any = field(repr=False, compare=False)
    orthogonal_factor: Any = field(repr=False, compare=False)
    positive_input_factor: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        full = _matrix(
            self.full_transition,
            "full_transition",
            dtype=torch.float64,
        )
        orthogonal = _matrix(
            self.orthogonal_factor,
            "orthogonal_factor",
            dtype=torch.float64,
        )
        positive = _matrix(
            self.positive_input_factor,
            "positive_input_factor",
            dtype=torch.float64,
        )
        if tuple(orthogonal.shape) != tuple(full.shape) or tuple(positive.shape) != (
            int(full.shape[1]),
            int(full.shape[1]),
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Polar transition tensors have incompatible geometry."
            )
        if not verify_lfm2_moving_gauge_report(self.report):
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Transition report seal is invalid."
            )
        for name, value in (
            ("full_transition", full),
            ("orthogonal_factor", orthogonal),
            ("positive_input_factor", positive),
        ):
            if self.report.get(f"{name}_sha256") != (
                fingerprint_lfm2_moving_gauge_tensor(value)
            ):
                raise Lfm2MovingGaugeResidualAuthenticationError(
                    f"Transition {name} differs from its report."
                )
        object.__setattr__(self, "full_transition", full)
        object.__setattr__(self, "orthogonal_factor", orthogonal)
        object.__setattr__(self, "positive_input_factor", positive)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        try:
            return verify_lfm2_moving_gauge_report(self.report) and all(
                self.report[f"{name}_sha256"]
                == fingerprint_lfm2_moving_gauge_tensor(value)
                for name, value in (
                    ("full_transition", self.full_transition),
                    ("orthogonal_factor", self.orthogonal_factor),
                    ("positive_input_factor", self.positive_input_factor),
                )
            )
        except (KeyError, TypeError, ValueError):
            return False


def _moving_transition(
    upstream: Lfm2MovingGaugeBoundaryChart,
    downstream: Lfm2MovingGaugeBoundaryChart,
    *,
    transition_index: int,
    policy: Lfm2MovingGaugeReclosurePolicy,
) -> Lfm2MovingGaugeTransition:
    torch = _torch()
    if upstream.decoder.shape[1] != downstream.encoder.shape[0]:
        raise Lfm2MovingGaugeResidualReclosureError(
            "Adjacent charts use different parent residual widths."
        )
    # Column convention requested by the compiler design: E_(l+1) D_l.
    full = (downstream.encoder.T @ upstream.decoder.T).contiguous()
    left, singular, right_h = torch.linalg.svd(full, full_matrices=False)
    orthogonal = (left @ right_h).contiguous()
    positive = (right_h.T @ torch.diag(singular) @ right_h).contiguous()
    reconstruction_error = float((orthogonal @ positive - full).abs().max().item())
    rows, columns = int(full.shape[0]), int(full.shape[1])
    if rows >= columns:
        identity = torch.eye(columns, dtype=torch.float64)
        partial_error = float((orthogonal.T @ orthogonal - identity).abs().max().item())
        partial_isometry_side = "columns"
    else:
        identity = torch.eye(rows, dtype=torch.float64)
        partial_error = float((orthogonal @ orthogonal.T - identity).abs().max().item())
        partial_isometry_side = "rows"
    if partial_error > policy.orthogonality_tolerance:
        raise Lfm2MovingGaugeResidualReclosureError(
            "Polar transition is not a partial isometry within tolerance."
        )
    body = {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "moving_gauge_boundary_transition_polar_split",
        "transition_index": transition_index,
        "formula": "E_(l+1)@D_l",
        "column_convention": True,
        "upstream_boundary_sha256": upstream.report["report_sha256"],
        "downstream_boundary_sha256": downstream.report["report_sha256"],
        "input_child_width": columns,
        "output_child_width": rows,
        "full_transition_sha256": fingerprint_lfm2_moving_gauge_tensor(full),
        "orthogonal_factor_sha256": fingerprint_lfm2_moving_gauge_tensor(orthogonal),
        "positive_input_factor_sha256": fingerprint_lfm2_moving_gauge_tensor(positive),
        "polar_formula": "full_transition=orthogonal_factor@positive_input_factor",
        "polar_reconstruction_max_abs_error": reconstruction_error,
        "partial_isometry_side": partial_isometry_side,
        "partial_isometry_max_abs_error": partial_error,
        "rmsnorm_crossing_factor": "no_unconditional_factor_crossing_claim",
        "rmsnorm_gain_commutation_must_be_authenticated_by_integrating_compiler": (
            True
        ),
        "noncommuting_rmsnorm_gain_requires_adjacent_input_producer_absorption": (True),
        "nonorthogonal_positive_factor_crosses_rmsnorm": False,
        "nonorthogonal_positive_factor_absorption": (
            "adjacent_existing_physical_writers"
        ),
        "square_rmsnorm_orthogonal_conjugacy_claim": False,
        "rectangular_rmsnorm_conjugacy_claim": False,
        "arbitrary_map_conjugacy_through_rmsnorm_claimed": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    return Lfm2MovingGaugeTransition(
        transition_index=transition_index,
        full_transition=full,
        orthogonal_factor=orthogonal,
        positive_input_factor=positive,
        report=_seal(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeDenseReplacementState:
    """Target-dtype dense writer replacements; no inference sidecar."""

    parameter_ids: tuple[str, ...]
    replacement_weights: tuple[Any, ...] = field(repr=False, compare=False)
    boundaries: tuple[Lfm2MovingGaugeBoundaryChart, ...]
    transitions: tuple[Lfm2MovingGaugeTransition, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        ids = tuple(self.parameter_ids)
        if not ids or len(ids) != len(set(ids)):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Replacement parameter_ids must be nonempty and unique."
            )
        values = tuple(
            _matrix(value, f"replacement_weights.{parameter_id}")
            for parameter_id, value in zip(ids, self.replacement_weights, strict=True)
        )
        if not verify_lfm2_moving_gauge_report(self.report):
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Dense replacement report seal is invalid."
            )
        if self.report.get("parameter_ids") != list(ids) or self.report.get(
            "replacement_weight_sha256"
        ) != [fingerprint_lfm2_moving_gauge_tensor(value) for value in values]:
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Dense replacement tensors differ from their report."
            )
        object.__setattr__(self, "parameter_ids", ids)
        object.__setattr__(self, "replacement_weights", values)
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def replacements_by_parameter_id(self) -> Mapping[str, Any]:
        return MappingProxyType(
            dict(zip(self.parameter_ids, self.replacement_weights, strict=True))
        )

    def verify_authentication(self) -> bool:
        try:
            return (
                verify_lfm2_moving_gauge_report(self.report)
                and self.report["replacement_weight_sha256"]
                == [
                    fingerprint_lfm2_moving_gauge_tensor(value)
                    for value in self.replacement_weights
                ]
                and all(value.verify_authentication() for value in self.boundaries)
                and all(value.verify_authentication() for value in self.transitions)
            )
        except (KeyError, TypeError, ValueError):
            return False


def _writer_group(
    sites: Sequence[Lfm2MovingGaugeWriterEvidence],
    current: Mapping[str, Any],
) -> tuple[Any, Any, tuple[int, ...]]:
    torch = _torch()
    features = torch.cat(tuple(site.fit_input_rows for site in sites), dim=1)
    widths = tuple(int(site.seed_weight.shape[1]) for site in sites)
    weights = torch.cat(
        tuple(current[site.parameter_id].to(dtype=torch.float64) for site in sites),
        dim=1,
    )
    return features, weights, widths


def _split_cast_replacements(
    sites: Sequence[Lfm2MovingGaugeWriterEvidence],
    combined: Any,
    widths: Sequence[int],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    cursor = 0
    for site, width in zip(sites, widths, strict=True):
        value = combined[:, cursor : cursor + width]
        cursor += width
        cast = value.to(dtype=site.seed_weight.dtype).contiguous().clone()
        if not bool(_torch().isfinite(cast).all().item()):
            raise Lfm2MovingGaugeResidualReclosureError(
                f"Target-dtype replacement for {site.parameter_id} is non-finite."
            )
        result[site.parameter_id] = cast
    if cursor != int(combined.shape[1]):
        raise AssertionError("Internal writer split lost coordinates.")
    return MappingProxyType(result)


def compile_lfm2_moving_gauge_residual_reclosure(
    boundary_evidence: Sequence[Lfm2MovingGaugeBoundaryEvidence],
    writer_evidence: Sequence[Lfm2MovingGaugeWriterEvidence],
    *,
    policy: Lfm2MovingGaugeReclosurePolicy | None = None,
) -> Lfm2MovingGaugeDenseReplacementState:
    """Compile one complete target-dtype forward/reverse reclosure transaction."""

    torch = _torch()
    active_policy = policy or Lfm2MovingGaugeReclosurePolicy()
    boundaries_input = tuple(boundary_evidence)
    sites_input = tuple(writer_evidence)
    if (
        not isinstance(active_policy, Lfm2MovingGaugeReclosurePolicy)
        or not active_policy.verify_authentication()
        or len(boundaries_input) < 2
        or not all(
            isinstance(value, Lfm2MovingGaugeBoundaryEvidence)
            and value.verify_authentication()
            for value in boundaries_input
        )
        or not sites_input
        or not all(
            isinstance(value, Lfm2MovingGaugeWriterEvidence)
            and value.verify_authentication()
            for value in sites_input
        )
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Moving-gauge compile inputs failed authentication."
        )
    layer_count = len(boundaries_input) - 1
    if (
        boundaries_input[0].interface_role != "input"
        or boundaries_input[-1].interface_role != "terminal"
        or len({value.boundary_id for value in boundaries_input})
        != len(boundaries_input)
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Boundary series must have unique input-to-terminal ordering."
        )
    alignment = boundaries_input[0].row_alignment_sha256
    row_count = int(boundaries_input[0].fit_parent_rows.shape[0])
    parent_width = boundaries_input[0].parent_width
    if any(
        value.row_alignment_sha256 != alignment
        or int(value.fit_parent_rows.shape[0]) != row_count
        or value.parent_width != parent_width
        for value in boundaries_input
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Boundary series does not share one aligned full-prefix row contract."
        )
    grouped: dict[int, list[Lfm2MovingGaugeWriterEvidence]] = {
        index: [] for index in range(layer_count)
    }
    for site in sites_input:
        if (
            site.transition_index not in grouped
            or site.row_alignment_sha256 != alignment
            or int(site.fit_input_rows.shape[0]) != row_count
            or int(site.seed_weight.shape[0])
            != boundaries_input[site.transition_index + 1].child_width
            or (
                site.boundary_evidence_sha256 is not None
                and site.boundary_evidence_sha256
                != boundaries_input[site.transition_index].report["report_sha256"]
            )
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                f"Writer {site.parameter_id} does not match its boundary transition."
            )
        grouped[site.transition_index].append(site)
    if any(not value for value in grouped.values()):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Every boundary transition requires at least one physical writer."
        )
    for values in grouped.values():
        values.sort(key=lambda value: value.parameter_id)
    if len({site.parameter_id for site in sites_input}) != len(sites_input):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Physical writer parameter ids must be globally unique."
        )

    def operation() -> tuple[
        tuple[Lfm2MovingGaugeBoundaryChart, ...],
        tuple[Lfm2MovingGaugeTransition, ...],
        Mapping[str, Any],
        Mapping[str, Any],
    ]:
        initial_charts = tuple(
            fit_lfm2_cycle_coupled_moving_gauge_boundary(
                value,
                policy=active_policy,
            )
            for value in boundaries_input
        )
        chart_series = [initial_charts[0]]
        for index in range(layer_count):
            chart_series.append(
                transport_lfm2_cycle_coupled_boundary_from_upstream(
                    chart_series[index],
                    initial_charts[index + 1],
                    downstream_evidence=boundaries_input[index + 1],
                    fit_actual_child_upstream_rows=(
                        boundaries_input[index].fit_child_rows
                    ),
                    fit_weights=boundaries_input[index].fit_weights,
                    fit_provenance=lfm2_moving_gauge_sequential_fit_provenance(
                        boundaries_input[index],
                        boundaries_input[index + 1],
                    ),
                )
            )
        charts = tuple(chart_series)
        transitions = tuple(
            _moving_transition(
                charts[index],
                charts[index + 1],
                transition_index=index,
                policy=active_policy,
            )
            for index in range(layer_count)
        )
        current: dict[str, Any] = {
            site.parameter_id: site.seed_weight.clone() for site in sites_input
        }
        state = boundaries_input[0].fit_child_rows.clone()
        forward_steps = []
        for index in range(layer_count):
            upstream = boundaries_input[index]
            downstream = boundaries_input[index + 1]
            sites = tuple(grouped[index])
            features, seed_group, widths = _writer_group(sites, current)
            existing_writer_output = features @ seed_group.T
            live_branch_write = downstream.fit_child_rows - upstream.fit_child_rows
            fixed_other_write = live_branch_write - existing_writer_output
            moving_endpoint, moving_write = charts[
                index + 1
            ].chart.residual_write_target(
                actual_child_pre_state=state,
                parent_endpoint=downstream.fit_parent_rows,
            )
            if active_policy.transition_mode == "moving_gauge":
                desired_endpoint = moving_endpoint
                desired_total_write = moving_write
            else:
                if upstream.child_width != downstream.child_width:
                    raise Lfm2MovingGaugeResidualReclosureError(
                        "Forced-identity diagnostic requires equal adjacent child widths."
                    )
                parent_delta = downstream.fit_parent_rows - upstream.fit_parent_rows
                desired_total_write = charts[index].chart.encode(parent_delta)
                desired_endpoint = state + desired_total_write
            target_writer_output = desired_total_write - fixed_other_write
            replacement_columns, solve_report = _anchored_ridge_replacement(
                features,
                target_writer_output,
                seed_group.T,
                upstream.fit_weights,
                ridge_ratio=active_policy.writer_ridge_ratio,
                absolute_ridge_floor=active_policy.absolute_ridge_floor,
            )
            split = _split_cast_replacements(
                sites,
                replacement_columns.T,
                widths,
            )
            previous = {site.parameter_id: current[site.parameter_id] for site in sites}
            current.update(split)
            features_after, group_after, _ = _writer_group(sites, current)
            actual_writer_output = features_after @ group_after.T
            state = (state + fixed_other_write + actual_writer_output).contiguous()
            full = transitions[index].full_transition
            orthogonal = transitions[index].orthogonal_factor
            factor_defect = upstream.fit_child_rows @ (full - orthogonal).T
            forward_steps.append(
                {
                    "transition_index": index,
                    "boundary_id": downstream.boundary_id,
                    "parameter_ids": [site.parameter_id for site in sites],
                    "solve_report_sha256": solve_report["report_sha256"],
                    "desired_endpoint_relative_rmse": _relative_rmse(
                        state,
                        desired_endpoint,
                        upstream.fit_weights,
                    ),
                    "moving_endpoint_relative_rmse": _relative_rmse(
                        state,
                        moving_endpoint,
                        upstream.fit_weights,
                    ),
                    "requested_nonorthogonal_factor_defect_l2_norm": float(
                        torch.linalg.vector_norm(factor_defect).item()
                    ),
                    "nonorthogonal_factor_absorbed_by_writer_target": True,
                    "changed_coordinate_count": sum(
                        int(
                            torch.count_nonzero(
                                current[site.parameter_id]
                                != previous[site.parameter_id]
                            ).item()
                        )
                        for site in sites
                    ),
                }
            )

        terminal_target = charts[-1].chart.encode(boundaries_input[-1].fit_parent_rows)
        forward_terminal = state.clone()
        terminal_residual = (terminal_target - state).contiguous()
        physical_widths = {value.child_width for value in boundaries_input}
        if len(physical_widths) != 1:
            raise Lfm2MovingGaugeResidualReclosureError(
                "The frozen-writer-input reverse approximation requires the "
                "physical residual skip to preserve one child width."
            )
        # A chart transition describes how two *desired parent encodings* are
        # related.  It is not a Jacobian of the installed child.  Under this
        # solver's explicit frozen-writer-input approximation, a local writer
        # correction travels through every later residual skip unchanged.
        # Therefore the only honest closed-form downstream transport available
        # here is the exact physical identity.  Nonlinear downstream response is
        # measured by the later causal recapture/reclosure stage instead of being
        # invented from E/D chart rotations.
        physical_identity = torch.eye(
            boundaries_input[-1].child_width,
            dtype=torch.float64,
        )
        downstream_rows = [physical_identity] * layer_count
        reverse_steps = []
        for index in range(layer_count - 1, -1, -1):
            sites = tuple(grouped[index])
            features, current_group, widths = _writer_group(sites, current)
            row_transport = downstream_rows[index]
            local_requested = terminal_residual
            current_output = features @ current_group.T
            replacement_columns, solve_report = _anchored_ridge_replacement(
                features,
                current_output + local_requested,
                current_group.T,
                boundaries_input[index].fit_weights,
                ridge_ratio=active_policy.writer_ridge_ratio,
                absolute_ridge_floor=active_policy.absolute_ridge_floor,
            )
            previous_group = current_group
            split = _split_cast_replacements(
                sites,
                replacement_columns.T,
                widths,
            )
            current.update(split)
            _, new_group, _ = _writer_group(sites, current)
            local_realized = features @ (new_group - previous_group).T
            terminal_realized = local_realized @ row_transport
            terminal_residual = (terminal_residual - terminal_realized).contiguous()
            reverse_steps.append(
                {
                    "transition_index": index,
                    "parameter_ids": [site.parameter_id for site in sites],
                    "transport_reciprocal_report_sha256": None,
                    "downstream_transport_kind": (
                        "exact_physical_identity_skip_under_frozen_writer_inputs"
                    ),
                    "chart_transition_used_as_physical_jacobian": False,
                    "solve_report_sha256": solve_report["report_sha256"],
                    "downstream_transport_sha256": (
                        fingerprint_lfm2_moving_gauge_tensor(row_transport)
                    ),
                    "requested_local_l2_norm": float(
                        torch.linalg.vector_norm(local_requested).item()
                    ),
                    "realized_terminal_l2_norm": float(
                        torch.linalg.vector_norm(terminal_realized).item()
                    ),
                    "remaining_terminal_l2_norm": float(
                        torch.linalg.vector_norm(terminal_residual).item()
                    ),
                }
            )
        endpoint = terminal_target - terminal_residual
        diagnostics = _seal(
            {
                "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
                "kind": "moving_gauge_forward_reverse_diagnostics",
                "transition_mode": active_policy.transition_mode,
                "forward_steps": forward_steps,
                "reverse_steps": reverse_steps,
                "reverse_downstream_transport_kind": (
                    "exact_physical_identity_skip_under_frozen_writer_inputs"
                ),
                "chart_transition_used_as_physical_jacobian": False,
                "nonlinear_downstream_jacobian_claimed": False,
                "reverse_transport_ridge_ratio_used": False,
                "forward_terminal_relative_rmse": _relative_rmse(
                    forward_terminal,
                    terminal_target,
                    boundaries_input[-1].fit_weights,
                ),
                "reconciled_terminal_relative_rmse": _relative_rmse(
                    endpoint,
                    terminal_target,
                    boundaries_input[-1].fit_weights,
                ),
                "forward_sweep_count": 1,
                "reverse_reconciliation_sweep_count": 1,
                "all_metrics_diagnostic_only": True,
                "quality_gate": False,
            }
        )
        return charts, transitions, MappingProxyType(current), diagnostics

    (charts, transitions, replacements, diagnostics), guard = run_fold_only_callback(
        operation,
        phase="lfm2-moving-gauge-residual-reclosure",
        protected_models=(),
    )
    ordered_sites = tuple(sorted(sites_input, key=lambda value: value.parameter_id))
    ids = tuple(site.parameter_id for site in ordered_sites)
    seeds = tuple(site.seed_weight for site in ordered_sites)
    values = tuple(replacements[parameter_id] for parameter_id in ids)
    changed_counts = tuple(
        int(torch.count_nonzero(value != seed).item())
        for seed, value in zip(seeds, values, strict=True)
    )
    if sum(changed_counts) <= 0:
        raise Lfm2MovingGaugeResidualReclosureError(
            "Moving-gauge publication is structurally ineffective in the target "
            "dtype: every dense writer is byte-identical to its seed."
        )
    publication_kinds = {
        site.parameter_id: site.publication_kind for site in ordered_sites
    }
    body = {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "moving_gauge_target_dtype_dense_replacement_state",
        "implementation": _IMPLEMENTATION,
        "policy_sha256": active_policy.authentication_sha256,
        "boundary_evidence_sha256": [
            value.report["report_sha256"] for value in boundaries_input
        ],
        "boundary_chart_sha256": [value.report["report_sha256"] for value in charts],
        "transition_sha256": [value.report["report_sha256"] for value in transitions],
        "writer_evidence_sha256": [
            site.report["report_sha256"] for site in ordered_sites
        ],
        "parameter_ids": list(ids),
        "seed_weight_sha256": [
            fingerprint_lfm2_moving_gauge_tensor(value) for value in seeds
        ],
        "replacement_weight_sha256": [
            fingerprint_lfm2_moving_gauge_tensor(value) for value in values
        ],
        "target_dtypes": [_dtype_name(value) for value in values],
        "changed_coordinate_count_by_parameter": dict(zip(ids, changed_counts)),
        "changed_coordinate_count": sum(changed_counts),
        "changed_parameter_count": sum(value > 0 for value in changed_counts),
        "publication_kind_by_parameter": publication_kinds,
        "live_physical_parameter_ids": [
            site.parameter_id
            for site in ordered_sites
            if site.publication_kind == "live_physical_parameter"
        ],
        "deferred_dense_prior_parameter_ids": [
            site.parameter_id
            for site in ordered_sites
            if site.publication_kind == "deferred_dense_prior"
        ],
        "target_dtype_structural_effectfulness_required": True,
        "direct_target_dtype_dense_replacement": True,
        "analytic_small_additive_ray_materialized": False,
        "global_ray_multiplier": None,
        "nextafter_weight_injection": False,
        "forward_then_reverse_target_dtype_reconciliation": True,
        "physical_seed_tensors_mutated": False,
        "physical_parameter_sharing": False,
        "runtime_sidecar_required": False,
        "parent_sidecar_required": False,
        "diagnostics_sha256": diagnostics["report_sha256"],
        "diagnostics": dict(diagnostics),
        "fold_only_guard": guard.as_dict(),
        "execution": {
            **_ZERO_EXECUTION,
            "optimizer_constructions": guard.optimizer_constructions,
            "optimizer_steps": guard.optimizer_steps,
            "backward_calls": guard.backward_calls,
            "autograd_grad_calls": guard.autograd_grad_calls,
            "parameters_pruned": guard.pruning_operations,
            "parameters_quantized": guard.quantization_operations,
            "guard_clean": guard.clean,
            "closed_form_linear_solves": True,
        },
        "numeric_quality_gate": False,
        "decoded_intelligence_gate": False,
    }
    return Lfm2MovingGaugeDenseReplacementState(
        parameter_ids=ids,
        replacement_weights=values,
        boundaries=charts,
        transitions=transitions,
        report=_seal(body),
    )


__all__ = [
    "LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA",
    "Lfm2MovingGaugeBoundaryChart",
    "Lfm2MovingGaugeBoundaryEvidence",
    "Lfm2MovingGaugeDenseReplacementState",
    "Lfm2MovingGaugeGeneralizedChartDiagnostic",
    "Lfm2MovingGaugeReclosurePolicy",
    "Lfm2MovingGaugeResidualAuthenticationError",
    "Lfm2MovingGaugeResidualReclosureError",
    "Lfm2MovingGaugeTransition",
    "Lfm2MovingGaugeWriterEvidence",
    "compile_lfm2_moving_gauge_residual_reclosure",
    "fingerprint_lfm2_moving_gauge_tensor",
    "fit_lfm2_cycle_coupled_moving_gauge_boundary",
    "fit_lfm2_generalized_moving_gauge_chart_diagnostic",
    "fit_lfm2_moving_gauge_residual_chart_adapter",
    "lfm2_moving_gauge_sequential_fit_provenance",
    "solve_lfm2_trace_scaled_weighted_anchored_ridge",
    "transport_lfm2_cycle_coupled_boundary_from_upstream",
    "verify_lfm2_moving_gauge_report",
]
