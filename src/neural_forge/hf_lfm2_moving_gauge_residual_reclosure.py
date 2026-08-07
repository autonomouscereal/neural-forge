"""Fold-only moving-gauge residual reclosure for an already folded child.

The ordinary residual skip is an identity map, while independently folded
boundaries need not use the same child coordinate chart.  The mature CPCF
boundary primitive already gives the compositionally correct local target::

    child_write = encoded_parent_endpoint - actual_child_pre_state

This module adds the missing whole-stack pieces without executing the parent
model: a cycle-coupled encoder/decoder at every cached parent/live-child
boundary, explicit ``E_(l+1) D_l`` chart transitions, polar separation of the
orthogonal factor that may cross RMSNorm from the non-orthogonal factor that
must be absorbed by physical writers, and one forward plus one reverse
closed-form writer reconciliation sweep.  Publication consists only of dense
replacement tensors in the seed target dtype.  No runtime sidecar remains.

The default reverse sweep transports a local writer correction to the terminal
through the exact physical identity skip, which is only valid under the
explicit frozen-writer-input approximation.  ``reverse_sweep_mode`` selects the
alternative: the caller hands in one mid-sweep child recapture provider, the
forward replacements are measured on the actual child between the sweeps, and
the reverse solves are anchored on that measured downstream response instead of
the identity assumption.  That mode is a distinct sealed policy; the sealed
digest of every prior forward-only run is unchanged.

Both sweeps optimize objectives that are not the boundary's own write, so
either can replace a healthy writer with one that agrees with the child's own
downstream error instead.  ``writer_reclosure_acceptance`` selects the guard:
before any proposed writer is kept, incumbent and proposal are scored against
the same branch-write target frame the boundary's own solve used, on the same
rows the sweep being guarded solves on, and a proposal that regresses that
boundary beyond ``boundary_regression_tolerance`` is refused - the incumbent
writer is kept and the refusal is recorded with both metrics and both write
norms.  The reclosure becomes per-boundary monotone instead of free to trade a
first-layer catastrophe for a terminal objective.  ``unguarded`` is the default
and is a distinct sealed policy from either guarded mode, so the sealed digest
of every prior unguarded run is unchanged.

That guard measures one boundary's own write error and nothing else, so a
proposal is free to improve its local frame while multiplying the write it
publishes - NOTES-W13M section 1.2 measured 82 accepted rewrites, 81 of them
upward, median 33x and one at 974x, with the guard refusing nothing.
``writer_reclosure_norm_ratio_ceiling`` is the second clause: a proposal whose
write-norm ratio exceeds the ceiling is refused whatever its boundary metric
did, recorded under its own refusal reason.  ``None`` is the default and keeps
the sealed digest of every guarded run that predates it.

This is analytic folding, not training.  There is no autograd graph, backward
call, optimizer, parent recapture, parent model forward, pruning, quantization,
or numeric quality gate.  The optional mid-sweep child recapture is executed by
the caller's provider, never by this module, and its declared child replay and
forward counts are sealed into the receipt.
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
LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-residual-reclosure-recaptured-reverse-v2"
)
LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-residual-reclosure-boundary-acceptance-v1"
)
LFM2_MOVING_GAUGE_WRITE_NORM_CEILING_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-residual-reclosure-write-norm-ceiling-v1"
)
_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_moving_gauge_residual_reclosure."
    "cycle-coupled-polar-forward-reverse-dense-replacement.v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INTERFACE_ROLES = frozenset({"input", "intermediate", "terminal"})
_TRANSITION_MODES = frozenset({"moving_gauge", "forced_identity_diagnostic"})
_REVERSE_SWEEP_MODES = frozenset(
    {"forward_only_identity_transport", "recaptured_terminal_anchored"}
)
LFM2_MOVING_GAUGE_WRITER_RECLOSURE_UNGUARDED = "unguarded"
LFM2_MOVING_GAUGE_WRITER_RECLOSURE_REQUIRE_IMPROVEMENT = (
    "require_boundary_improvement"
)
LFM2_MOVING_GAUGE_WRITER_RECLOSURE_FAIL_CLOSED = (
    "require_boundary_improvement_fail_closed"
)
LFM2_MOVING_GAUGE_WRITER_RECLOSURE_ACCEPTANCE_MODES = (
    LFM2_MOVING_GAUGE_WRITER_RECLOSURE_UNGUARDED,
    LFM2_MOVING_GAUGE_WRITER_RECLOSURE_REQUIRE_IMPROVEMENT,
    LFM2_MOVING_GAUGE_WRITER_RECLOSURE_FAIL_CLOSED,
)
_ACCEPTANCE_UNGUARDED = LFM2_MOVING_GAUGE_WRITER_RECLOSURE_UNGUARDED
_ACCEPTANCE_SKIP = LFM2_MOVING_GAUGE_WRITER_RECLOSURE_REQUIRE_IMPROVEMENT
_ACCEPTANCE_FAIL_CLOSED = LFM2_MOVING_GAUGE_WRITER_RECLOSURE_FAIL_CLOSED
_WRITER_RECLOSURE_ACCEPTANCE_MODES = frozenset(
    LFM2_MOVING_GAUGE_WRITER_RECLOSURE_ACCEPTANCE_MODES
)
_BOUNDARY_ACCEPTANCE_METRIC = "weighted_relative_l2_branch_write_error"
_WRITE_NORM_RATIO_METRIC = "proposed_over_incumbent_branch_write_l2_norm"
# Writer evidence carries exactly one aligned design-row set per site, so the
# acceptance measurement is taken on the same rows the sweep it guards solves
# on: the sealed FIT rows for the forward sweep and the authenticated mid-sweep
# recapture for the recaptured reverse sweep.  Naming the frame in every
# decision keeps the receipt honest about which rows produced the number.
_FIT_ROW_FRAME = "aligned_fit_writer_input_rows"
_RECAPTURED_ROW_FRAME = "recaptured_writer_input_rows"
_FORWARD_SWEEP_NAME = "forward"
_REVERSE_SWEEP_NAME = "reverse"
# NOTES-W13M section 1.2: the boundary metric is local to one solve frame, so a
# proposal can cut its own write error while multiplying the write it publishes
# by 974x.  The two refusal reasons are kept apart so a receipt reader can count
# exactly what the norm ceiling added on top of the boundary clause.
_REFUSAL_BOUNDARY_REGRESSION = "boundary_write_error_regressed"
_REFUSAL_WRITE_NORM_CEILING = "write_norm_ratio_above_ceiling"
_IDENTITY_TRANSPORT_KIND = "exact_physical_identity_skip_under_frozen_writer_inputs"
_TERMINAL_BOUNDARY_TRANSPORT_KIND = "exact_physical_identity_at_terminal_boundary"
_MEASURED_TRANSPORT_KIND = "measured_recaptured_child_downstream_response"
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


@dataclass(frozen=True, slots=True, init=False)
class PreparedAnchoredWriterProblem:
    """Authenticated immutable design side of one physical writer solve.

    The prepared object owns detached CPU copies of every target-invariant
    tensor used by :func:`_anchored_ridge_replacement`.  A later solve still
    builds the target residual and calls the same unfactored
    ``torch.linalg.solve`` path.  In particular, this object is not a cached
    factorization and does not change the numerical algorithm.

    Tensor fields are deliberately private.  ``frozen=True`` prevents field
    replacement, while :meth:`verify_authentication` detects any in-place
    mutation of the retained tensors before they can be used.
    """

    _design_fp64: Any = field(repr=False, compare=False)
    _seed_weight: Any = field(repr=False, compare=False)
    _anchor_columns_fp64: Any = field(repr=False, compare=False)
    _seed_prediction_fp64: Any = field(repr=False, compare=False)
    _fit_weights_fp64: Any = field(repr=False, compare=False)
    _root_weights_fp64: Any = field(repr=False, compare=False)
    _weighted_design_fp64: Any = field(repr=False, compare=False)
    _regularized_system_fp64: Any = field(repr=False, compare=False)
    ridge_ratio: float
    absolute_ridge_floor: float
    ridge_scale: float
    effective_ridge: float
    solve_form: str
    report: Mapping[str, Any]

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "prepared_anchored_writer_problem",
            "rows": int(self._design_fp64.shape[0]),
            "input_width": int(self._design_fp64.shape[1]),
            "output_width": int(self._seed_weight.shape[0]),
            "target_dtype": _dtype_name(self._seed_weight),
            "ridge_ratio": self.ridge_ratio,
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "ridge_scale": self.ridge_scale,
            "effective_ridge": self.effective_ridge,
            "solve_form": self.solve_form,
            "fit_input_rows_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._design_fp64)
            ),
            "seed_weight_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._seed_weight)
            ),
            "anchor_columns_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._anchor_columns_fp64)
            ),
            "seed_prediction_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._seed_prediction_fp64)
            ),
            "fit_weights_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._fit_weights_fp64)
            ),
            "root_weights_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._root_weights_fp64)
            ),
            "weighted_design_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._weighted_design_fp64)
            ),
            "regularized_system_fp64_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(self._regularized_system_fp64)
            ),
            "target_rows_consumed": False,
            "linear_factorization_precomputed": False,
            "closed_form_linear_solve": True,
            "autograd_used": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }

    def verify_authentication(self) -> bool:
        try:
            return dict(self.report) == dict(_seal(self.authentication_body()))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


def _make_prepared_anchored_writer_problem(
    *,
    design_fp64: Any,
    seed_weight: Any,
    anchor_columns_fp64: Any,
    seed_prediction_fp64: Any,
    fit_weights_fp64: Any,
    root_weights_fp64: Any,
    weighted_design_fp64: Any,
    regularized_system_fp64: Any,
    ridge_ratio: float,
    absolute_ridge_floor: float,
    ridge_scale: float,
    effective_ridge: float,
    solve_form: str,
) -> PreparedAnchoredWriterProblem:
    problem = object.__new__(PreparedAnchoredWriterProblem)
    for name, value in (
        ("_design_fp64", design_fp64),
        ("_seed_weight", seed_weight),
        ("_anchor_columns_fp64", anchor_columns_fp64),
        ("_seed_prediction_fp64", seed_prediction_fp64),
        ("_fit_weights_fp64", fit_weights_fp64),
        ("_root_weights_fp64", root_weights_fp64),
        ("_weighted_design_fp64", weighted_design_fp64),
        ("_regularized_system_fp64", regularized_system_fp64),
        ("ridge_ratio", ridge_ratio),
        ("absolute_ridge_floor", absolute_ridge_floor),
        ("ridge_scale", ridge_scale),
        ("effective_ridge", effective_ridge),
        ("solve_form", solve_form),
    ):
        object.__setattr__(problem, name, value)
    object.__setattr__(problem, "report", MappingProxyType({}))
    object.__setattr__(
        problem,
        "report",
        _seal(problem.authentication_body()),
    )
    return problem


def prepare_lfm2_moving_gauge_anchored_writer_problem(
    fit_input_rows: Any,
    seed_weight: Any,
    fit_weights: Any,
    *,
    ridge_ratio: float = 1e-8,
    absolute_ridge_floor: float = 1e-12,
) -> PreparedAnchoredWriterProblem:
    """Prepare target-invariant FP64 math for repeated physical-writer solves."""

    torch = _torch()
    seed = _matrix(seed_weight, "seed_weight", require_plain=True)
    if _dtype_name(seed) not in _SUPPORTED_TARGET_DTYPES:
        raise Lfm2MovingGaugeResidualReclosureError(
            "Physical writer seed dtype must be FP32 or BF16."
        )
    # Keep the same two-stage materialization performed by the legacy public
    # solver and its private anchored-ridge implementation.
    physical_design = _matrix(
        fit_input_rows,
        "fit_input_rows",
        dtype=torch.float64,
    )
    weights = _weights(fit_weights, int(physical_design.shape[0]), "fit_weights")
    design = _matrix(physical_design, "ridge design", dtype=torch.float64)
    anchor = _matrix(
        seed.detach().to(device="cpu", dtype=torch.float64).T,
        "ridge anchor",
        dtype=torch.float64,
    )
    if int(design.shape[1]) != int(anchor.shape[0]):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Fresh-primal writer rows do not match the physical dense geometry."
        )
    ratio = _finite_nonnegative(ridge_ratio, "ridge_ratio")
    floor = _finite_positive(absolute_ridge_floor, "absolute_ridge_floor")
    scale = _ridge_scale(design, weights)
    ridge = max(floor, ratio * scale)
    root = torch.sqrt(weights).unsqueeze(1)
    weighted_design = design * root
    seed_prediction = (design @ anchor).contiguous()
    if int(design.shape[1]) <= int(design.shape[0]):
        normal = weighted_design.T @ weighted_design
        system = (
            normal + ridge * torch.eye(int(normal.shape[0]), dtype=torch.float64)
        ).contiguous()
        solve_form = "primal"
    else:
        dual = weighted_design @ weighted_design.T
        system = (
            dual + ridge * torch.eye(int(dual.shape[0]), dtype=torch.float64)
        ).contiguous()
        solve_form = "dual"
    if not bool(torch.isfinite(system).all().item()):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Prepared anchored-writer system is non-finite."
        )
    return _make_prepared_anchored_writer_problem(
        design_fp64=design,
        seed_weight=seed,
        anchor_columns_fp64=anchor,
        seed_prediction_fp64=seed_prediction,
        fit_weights_fp64=weights,
        root_weights_fp64=root,
        weighted_design_fp64=weighted_design,
        regularized_system_fp64=system,
        ridge_ratio=ratio,
        absolute_ridge_floor=floor,
        ridge_scale=scale,
        effective_ridge=ridge,
        solve_form=solve_form,
    )


def solve_lfm2_prepared_anchored_writer_replacement(
    problem: PreparedAnchoredWriterProblem,
    desired_output_rows: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Solve one new target against an authenticated prepared writer problem.

    The returned tensor and report intentionally match
    :func:`solve_lfm2_moving_gauge_anchored_writer_replacement` byte for byte.
    The prepared system is passed to ``torch.linalg.solve`` afresh; no LU,
    Cholesky, inverse, or other factorization is retained or reused.
    """

    torch = _torch()
    if not isinstance(problem, PreparedAnchoredWriterProblem) or not (
        problem.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Prepared anchored-writer problem failed exact authentication."
        )
    # Match the legacy public/private pair's two target materializations.
    physical_target = _matrix(
        desired_output_rows,
        "desired_output_rows",
        dtype=torch.float64,
    )
    target = _matrix(physical_target, "ridge target", dtype=torch.float64)
    design = problem._design_fp64
    seed = problem._seed_weight
    anchor = problem._anchor_columns_fp64
    weights = problem._fit_weights_fp64
    if int(design.shape[0]) != int(target.shape[0]) or int(target.shape[1]) != int(
        anchor.shape[1]
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Fresh-primal writer rows do not match the physical dense geometry."
        )

    weighted_residual = (
        target - problem._seed_prediction_fp64
    ) * problem._root_weights_fp64
    if problem.solve_form == "primal":
        delta = torch.linalg.solve(
            problem._regularized_system_fp64,
            problem._weighted_design_fp64.T @ weighted_residual,
        )
    elif problem.solve_form == "dual":
        delta = problem._weighted_design_fp64.T @ torch.linalg.solve(
            problem._regularized_system_fp64,
            weighted_residual,
        )
    else:  # pragma: no cover - authenticated reports cannot reach this branch
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Prepared anchored-writer solve form is invalid."
        )
    replacement_columns = (anchor + delta).contiguous()
    if not bool(torch.isfinite(replacement_columns).all().item()):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Closed-form ridge replacement is non-finite."
        )
    solve_report = _seal(
        {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "anchored_closed_form_ridge_replacement",
            "solve_form": problem.solve_form,
            "rows": int(design.shape[0]),
            "input_width": int(design.shape[1]),
            "output_width": int(target.shape[1]),
            "ridge_ratio": problem.ridge_ratio,
            "ridge_scale": problem.ridge_scale,
            "effective_ridge": problem.effective_ridge,
            "design_sha256": problem.report["fit_input_rows_fp64_sha256"],
            "target_sha256": fingerprint_lfm2_moving_gauge_tensor(target),
            "anchor_sha256": problem.report["anchor_columns_fp64_sha256"],
            "replacement_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(replacement_columns)
            ),
            "closed_form_linear_solve": True,
            "autograd_used": False,
            "quality_gate": False,
        }
    )

    replacement = replacement_columns.T.to(dtype=seed.dtype).contiguous().clone()
    prediction_before = problem._seed_prediction_fp64
    prediction_after = design @ replacement.to(dtype=torch.float64).T
    report = _seal(
        {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "fresh_primal_anchored_physical_writer_replacement",
            "rows": int(design.shape[0]),
            "input_width": int(design.shape[1]),
            "output_width": int(target.shape[1]),
            "target_dtype": _dtype_name(seed),
            "fit_input_rows_sha256": problem.report["fit_input_rows_fp64_sha256"],
            "desired_output_rows_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(target)
            ),
            "fit_weights_sha256": problem.report["fit_weights_fp64_sha256"],
            "seed_weight_sha256": problem.report["seed_weight_sha256"],
            "replacement_weight_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(replacement)
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
    reverse_sweep_mode: str = "forward_only_identity_transport"
    recaptured_step_ridge_ratio: float = 1e-2
    writer_reclosure_acceptance: str = _ACCEPTANCE_UNGUARDED
    boundary_regression_tolerance: float = 0.0
    # NOTES-W13M section 1.2: ``None`` is every lane sealed before this ceiling
    # existed and is byte-identical to it.  A float refuses any proposed writer
    # whose write-norm ratio exceeds it, whatever its boundary metric did.
    writer_reclosure_norm_ratio_ceiling: float | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "chart_ridge_ratio",
            "cycle_consistency_weight",
            "writer_ridge_ratio",
            "reverse_transport_ridge_ratio",
            "recaptured_step_ridge_ratio",
            "orthogonality_tolerance",
            "boundary_regression_tolerance",
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
        reverse_mode = str(self.reverse_sweep_mode).strip().lower()
        if reverse_mode not in _REVERSE_SWEEP_MODES:
            raise Lfm2MovingGaugeResidualReclosureError(
                "reverse_sweep_mode must be forward_only_identity_transport or "
                "recaptured_terminal_anchored."
            )
        object.__setattr__(self, "reverse_sweep_mode", reverse_mode)
        acceptance = str(self.writer_reclosure_acceptance).strip().lower()
        if acceptance not in _WRITER_RECLOSURE_ACCEPTANCE_MODES:
            raise Lfm2MovingGaugeResidualReclosureError(
                "writer_reclosure_acceptance must be unguarded, "
                "require_boundary_improvement, or "
                "require_boundary_improvement_fail_closed."
            )
        object.__setattr__(self, "writer_reclosure_acceptance", acceptance)
        if (
            acceptance == _ACCEPTANCE_UNGUARDED
            and self.boundary_regression_tolerance != 0.0
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "boundary_regression_tolerance only applies to a guarded "
                "writer_reclosure_acceptance mode."
            )
        if self.writer_reclosure_norm_ratio_ceiling is not None:
            object.__setattr__(
                self,
                "writer_reclosure_norm_ratio_ceiling",
                _finite_positive(
                    self.writer_reclosure_norm_ratio_ceiling,
                    "writer_reclosure_norm_ratio_ceiling",
                ),
            )
            if acceptance == _ACCEPTANCE_UNGUARDED:
                # The unguarded lane never scores a proposal, so a ceiling there
                # would claim a refusal clause that can never run.
                raise Lfm2MovingGaugeResidualReclosureError(
                    "writer_reclosure_norm_ratio_ceiling only applies to a "
                    "guarded writer_reclosure_acceptance mode."
                )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2MovingGaugeResidualAuthenticationError(
                "Moving-gauge policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        body = {
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
        if self.reverse_sweep_mode != "forward_only_identity_transport":
            body = {
                **body,
                "policy_schema": LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA,
                "reverse_sweep_mode": self.reverse_sweep_mode,
                "recaptured_step_ridge_ratio": self.recaptured_step_ridge_ratio,
                "mid_sweep_child_recapture_count": 1,
                "reverse_downstream_transport_kind": _MEASURED_TRANSPORT_KIND,
                "frozen_writer_input_identity_transport": False,
                "reverse_writer_inputs_measured_after_forward_publication": (
                    True
                ),
            }
        if self.writer_reclosure_acceptance == _ACCEPTANCE_UNGUARDED:
            # Both default lanes are pinned byte-for-byte: the forward-only
            # body keeps the policy digest of every reclosure sealed before the
            # recaptured mode existed, and an unguarded body keeps the digest of
            # every reclosure sealed before this acceptance gate existed.
            return body
        body = {
            **body,
            "acceptance_schema": LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA,
            "writer_reclosure_acceptance": self.writer_reclosure_acceptance,
            "boundary_regression_tolerance": (self.boundary_regression_tolerance),
            "boundary_acceptance_metric": _BOUNDARY_ACCEPTANCE_METRIC,
            "boundary_acceptance_guards_every_writer_overwrite": True,
            "boundary_acceptance_refusal_keeps_incumbent_writer": (
                self.writer_reclosure_acceptance == _ACCEPTANCE_SKIP
            ),
            "boundary_acceptance_refusal_aborts_reclosure": (
                self.writer_reclosure_acceptance == _ACCEPTANCE_FAIL_CLOSED
            ),
            "per_boundary_monotone_writer_reclosure": True,
            "terminal_objective_may_overwrite_a_regressed_boundary": False,
        }
        if self.writer_reclosure_norm_ratio_ceiling is None:
            # Same discipline one level down: an unset ceiling keeps the policy
            # digest of every guarded reclosure sealed before it existed.
            return body
        return {
            **body,
            "norm_ceiling_schema": LFM2_MOVING_GAUGE_WRITE_NORM_CEILING_SCHEMA,
            "writer_reclosure_norm_ratio_ceiling": (
                self.writer_reclosure_norm_ratio_ceiling
            ),
            "write_norm_ratio_ceiling_metric": _WRITE_NORM_RATIO_METRIC,
            "write_norm_ratio_ceiling_refuses_any_boundary_metric": True,
            "write_norm_ratio_ceiling_refusal_keeps_incumbent_writer": (
                self.writer_reclosure_acceptance == _ACCEPTANCE_SKIP
            ),
            "write_norm_ratio_ceiling_refusal_aborts_reclosure": (
                self.writer_reclosure_acceptance == _ACCEPTANCE_FAIL_CLOSED
            ),
            "unmeasurable_write_norm_ratio_is_not_above_the_ceiling": True,
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


def _boundary_chart_exact_equal(
    actual: Lfm2MovingGaugeBoundaryChart,
    expected: Lfm2MovingGaugeBoundaryChart,
) -> bool:
    torch = _torch()
    return bool(
        isinstance(actual, Lfm2MovingGaugeBoundaryChart)
        and actual.verify_authentication()
        and dict(actual.report) == dict(expected.report)
        and actual.chart.name == expected.chart.name
        and dict(actual.chart.report) == dict(expected.chart.report)
        and dict(actual.chart.mapping.report) == dict(expected.chart.mapping.report)
        and not actual.chart.mapping.feature_predictors
        and torch.equal(actual.encoder, expected.encoder)
        and torch.equal(actual.decoder, expected.decoder)
        and torch.equal(actual.orthogonal_rotation, expected.orthogonal_rotation)
    )


def _recompute_generalized_fit_authority(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
) -> Mapping[str, Any]:
    """Reproduce every FIT-only candidate from retained authenticated inputs."""

    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not evidence.verify_fit_authentication()
        or evidence.interface_locked
        or evidence.interface_role != "intermediate"
        or not isinstance(policy, Lfm2MovingGaugeReclosurePolicy)
        or not policy.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized FIT authority requires authenticated intermediate "
            "evidence and policy."
        )
    orthogonal = fit_lfm2_cycle_coupled_moving_gauge_boundary(
        evidence,
        policy=policy,
        evaluate_heldout=False,
    )
    parent = evidence.fit_parent_rows
    child = evidence.fit_child_rows
    weights = evidence.fit_weights
    base = evidence.base_encoder
    base_encoded = (parent @ base).contiguous()
    transform, scale_shear_solve = _fp64_svd_anchored_chart_solve(
        base_encoded,
        child,
        orthogonal.orthogonal_rotation,
        weights,
        ridge_ratio=policy.chart_ridge_ratio,
        absolute_ridge_floor=policy.absolute_ridge_floor,
        candidate_kind="fixed_source_subspace_scale_shear",
    )
    scale_shear = (base @ transform).contiguous()
    generalized, generalized_solve = _fp64_svd_anchored_chart_solve(
        parent,
        child,
        orthogonal.encoder,
        weights,
        ridge_ratio=policy.chart_ridge_ratio,
        absolute_ridge_floor=policy.absolute_ridge_floor,
        candidate_kind="unrestricted_source_to_child",
    )
    return MappingProxyType(
        {
            "orthogonal_chart": orthogonal,
            "scale_shear_transform": transform,
            "scale_shear_encoder": scale_shear,
            "generalized_encoder": generalized,
            "scale_shear_solver_report": scale_shear_solve,
            "generalized_solver_report": generalized_solve,
        }
    )


def _orthogonal_causal_endpoint_body(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    boundary_chart: Lfm2MovingGaugeBoundaryChart,
) -> dict[str, Any]:
    locked = evidence.interface_locked
    return {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "orthogonal_causal_endpoint_chart",
        "implementation": _IMPLEMENTATION,
        "boundary_id": evidence.boundary_id,
        "interface_role": evidence.interface_role,
        "interface_locked": locked,
        "evidence_sha256": evidence.report["report_sha256"],
        "policy_sha256": policy.authentication_sha256,
        "selected_candidate_mode": (
            "locked_base_e0" if locked else "right_orthogonal_procrustes"
        ),
        "source_width": boundary_chart.chart.source_width,
        "target_width": boundary_chart.chart.target_width,
        "encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(
            boundary_chart.encoder
        ),
        "decoder_sha256": fingerprint_lfm2_moving_gauge_tensor(
            boundary_chart.decoder
        ),
        "orthogonal_rotation_sha256": fingerprint_lfm2_moving_gauge_tensor(
            boundary_chart.orthogonal_rotation
        ),
        "boundary_chart_sha256": boundary_chart.report["report_sha256"],
        "cpcf_chart_sha256": boundary_chart.chart.report["report_sha256"],
        "predictive_mapping_sha256": boundary_chart.chart.mapping.report[
            "report_sha256"
        ],
        "causal_endpoint_encoder_authority": True,
        "physical_writer_publication_authority": False,
        "locked_chart_claimed": locked,
        "right_orthogonal_chart_claimed": not locked,
        "generalized_chart_claimed": False,
        "heldout_used_by_solver": False,
        "heldout_evaluation_requested": False,
        "fit_only": True,
        "closed_form_solves_only": True,
        "numeric_quality_gate": False,
        "execution": dict(_ZERO_EXECUTION),
    }


def _authenticate_orthogonal_causal_endpoint_authority(
    *,
    evidence_sha256: str,
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    boundary_chart: Lfm2MovingGaugeBoundaryChart,
    report: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    evidence_sha = _digest(evidence_sha256, "evidence_sha256")
    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not evidence.verify_fit_authentication()
        or evidence.report["report_sha256"] != evidence_sha
        or not isinstance(policy, Lfm2MovingGaugeReclosurePolicy)
        or not policy.verify_authentication()
        or not isinstance(boundary_chart, Lfm2MovingGaugeBoundaryChart)
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Orthogonal causal endpoint requires exact authenticated evidence, "
            "policy, and mature boundary chart."
        )
    expected_chart = fit_lfm2_cycle_coupled_moving_gauge_boundary(
        evidence,
        policy=policy,
        evaluate_heldout=False,
    )
    if not _boundary_chart_exact_equal(boundary_chart, expected_chart):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Orthogonal causal endpoint chart does not recursively reproduce "
            "from its retained FIT evidence and policy."
        )
    expected_report = _seal(
        _orthogonal_causal_endpoint_body(evidence, policy, expected_chart)
    )
    if dict(report) != dict(expected_report):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Orthogonal causal endpoint report is stale or malformed."
        )
    # The solve above is FIT-only.  Authenticate any supplied heldout tensors
    # only after the endpoint tensor and report have been reconstructed.
    if not evidence.verify_authentication():
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Orthogonal causal endpoint retained evidence failed authentication."
        )
    return evidence_sha, expected_report


def _verify_orthogonal_causal_endpoint_snapshot(
    *,
    evidence_sha256: str,
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    boundary_chart: Lfm2MovingGaugeBoundaryChart,
    report: Mapping[str, Any],
) -> bool:
    """Cheap recheck after construction's full deterministic reconstruction."""

    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not evidence.verify_authentication()
        or evidence.report["report_sha256"] != evidence_sha256
        or not isinstance(policy, Lfm2MovingGaugeReclosurePolicy)
        or not policy.verify_authentication()
        or not isinstance(boundary_chart, Lfm2MovingGaugeBoundaryChart)
        or not boundary_chart.verify_authentication()
    ):
        return False
    chart = boundary_chart.chart
    mapping = chart.mapping
    lineage = {
        "boundary_evidence_sha256": evidence.report["report_sha256"],
        "cached_parent_evidence_sha256": evidence.cached_parent_evidence_sha256,
        "live_child_recapture_sha256": evidence.live_child_recapture_sha256,
    }
    if (
        chart.name != evidence.boundary_id
        or not verify_lfm2_moving_gauge_report(mapping.report)
        or not verify_lfm2_moving_gauge_report(chart.report)
        or mapping.feature_predictors
        or mapping.report.get("lineage") != lineage
        or chart.report.get("lineage") != lineage
        or chart.report.get("predictive_mapping") != mapping.as_dict()
        or mapping.report.get("encoder_sha256")
        != fingerprint_lfm2_moving_gauge_tensor(mapping.encoder)
        or mapping.report.get("decoder_sha256")
        != fingerprint_lfm2_moving_gauge_tensor(mapping.decoder)
        or boundary_chart.report.get("evidence_sha256") != evidence_sha256
        or boundary_chart.report.get("policy_sha256") != policy.authentication_sha256
        or boundary_chart.report.get("boundary_id") != evidence.boundary_id
        or boundary_chart.report.get("interface_role") != evidence.interface_role
        or boundary_chart.report.get("interface_locked") != evidence.interface_locked
        or boundary_chart.report.get("heldout_evaluation_requested") is not False
        or dict(report)
        != dict(_seal(_orthogonal_causal_endpoint_body(evidence, policy, boundary_chart)))
    ):
        return False
    return True


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeOrthogonalCausalEndpointChart:
    """Evidence-retaining mature locked/right-orthogonal causal endpoint."""

    evidence_sha256: str
    evidence: Lfm2MovingGaugeBoundaryEvidence = field(repr=False, compare=False)
    policy: Lfm2MovingGaugeReclosurePolicy = field(repr=False, compare=False)
    boundary_chart: Lfm2MovingGaugeBoundaryChart = field(
        repr=False,
        compare=False,
    )
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence_sha, expected = _authenticate_orthogonal_causal_endpoint_authority(
            evidence_sha256=self.evidence_sha256,
            evidence=self.evidence,
            policy=self.policy,
            boundary_chart=self.boundary_chart,
            report=self.report,
        )
        object.__setattr__(self, "evidence_sha256", evidence_sha)
        object.__setattr__(self, "report", expected)

    @property
    def chart(self) -> CpcfBoundaryChart:
        return self.boundary_chart.chart

    @property
    def encoder(self) -> Any:
        return self.boundary_chart.encoder

    @property
    def decoder(self) -> Any:
        return self.boundary_chart.decoder

    def verify_authentication(self) -> bool:
        try:
            return _verify_orthogonal_causal_endpoint_snapshot(
                evidence_sha256=self.evidence_sha256,
                evidence=self.evidence,
                policy=self.policy,
                boundary_chart=self.boundary_chart,
                report=self.report,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False


def bind_lfm2_orthogonal_causal_endpoint_chart(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    *,
    policy: Lfm2MovingGaugeReclosurePolicy | None = None,
) -> Lfm2MovingGaugeOrthogonalCausalEndpointChart:
    """Bind a mature locked/right-orthogonal chart to causal authority."""

    active_policy = policy or Lfm2MovingGaugeReclosurePolicy()
    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not evidence.verify_fit_authentication()
        or not isinstance(active_policy, Lfm2MovingGaugeReclosurePolicy)
        or not active_policy.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Orthogonal causal endpoint binding requires authenticated evidence "
            "and policy."
        )
    boundary_chart = fit_lfm2_cycle_coupled_moving_gauge_boundary(
        evidence,
        policy=active_policy,
        evaluate_heldout=False,
    )
    report = _seal(
        _orthogonal_causal_endpoint_body(
            evidence,
            active_policy,
            boundary_chart,
        )
    )
    return Lfm2MovingGaugeOrthogonalCausalEndpointChart(
        evidence_sha256=evidence.report["report_sha256"],
        evidence=evidence,
        policy=active_policy,
        boundary_chart=boundary_chart,
        report=report,
    )


def _generalized_chart_diagnostic_body(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the complete canonical generalized diagnostic report body."""

    torch = _torch()
    orthogonal = authority["orthogonal_chart"]
    scale_shear_transform = authority["scale_shear_transform"]
    scale_shear_encoder = authority["scale_shear_encoder"]
    generalized_encoder = authority["generalized_encoder"]
    scale_shear_solve = authority["scale_shear_solver_report"]
    generalized_solve = authority["generalized_solver_report"]
    parent = evidence.fit_parent_rows
    child = evidence.fit_child_rows
    weights = evidence.fit_weights
    base = evidence.base_encoder
    base_encoded = (parent @ base).contiguous()
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
            _chart_candidate_metrics(parent, child, weights, orthogonal.encoder)
        ),
        "scale_shear": dict(
            _chart_candidate_metrics(parent, child, weights, scale_shear_encoder)
        ),
        "generalized": dict(
            _chart_candidate_metrics(parent, child, weights, generalized_encoder)
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

    # Candidate tensors and their hashes are already reconstructed above.  Full
    # evidence authentication, including heldout tensors, deliberately happens
    # only after that freeze and before any heldout diagnostic is read.
    if not evidence.verify_authentication():
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized diagnostic retained evidence failed full authentication."
        )
    heldout_diagnostics = None
    if evidence.heldout_parent_rows is not None:
        heldout_parent = _matrix(
            evidence.heldout_parent_rows,
            "heldout_parent_rows_snapshot",
            dtype=torch.float64,
        )
        heldout_child = _matrix(
            evidence.heldout_child_rows,
            "heldout_child_rows_snapshot",
            dtype=torch.float64,
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

    return {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "parallel_generalized_moving_gauge_chart_diagnostic",
        "implementation": _IMPLEMENTATION,
        "boundary_id": evidence.boundary_id,
        "interface_role": evidence.interface_role,
        "interface_locked": evidence.interface_locked,
        "evidence_sha256": evidence.report["report_sha256"],
        "policy_sha256": policy.authentication_sha256,
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


def _authenticate_generalized_diagnostic_authority(
    *,
    evidence_sha256: str,
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    orthogonal_chart: Lfm2MovingGaugeBoundaryChart,
    scale_shear_transform: Any,
    scale_shear_encoder: Any,
    generalized_encoder: Any,
    report: Mapping[str, Any],
) -> tuple[str, Any, Any, Any]:
    torch = _torch()
    evidence_sha = _digest(evidence_sha256, "evidence_sha256")
    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or evidence.report["report_sha256"] != evidence_sha
        or not isinstance(policy, Lfm2MovingGaugeReclosurePolicy)
        or not policy.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized diagnostic does not retain its exact evidence and policy."
        )
    transform = _matrix(
        scale_shear_transform,
        "scale_shear_transform",
        dtype=torch.float64,
    )
    scale_shear = _matrix(
        scale_shear_encoder,
        "scale_shear_encoder",
        dtype=torch.float64,
    )
    generalized = _matrix(
        generalized_encoder,
        "generalized_encoder",
        dtype=torch.float64,
    )
    authority = _recompute_generalized_fit_authority(evidence, policy)
    expected_orthogonal = authority["orthogonal_chart"]
    expected_transform = authority["scale_shear_transform"]
    expected_scale_shear = authority["scale_shear_encoder"]
    expected_generalized = authority["generalized_encoder"]
    if (
        not _boundary_chart_exact_equal(orthogonal_chart, expected_orthogonal)
        or not torch.equal(transform, expected_transform)
        or not torch.equal(scale_shear, expected_scale_shear)
        or not torch.equal(scale_shear, evidence.base_encoder @ transform)
        or not torch.equal(generalized, expected_generalized)
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized diagnostic candidates do not reproduce from retained "
            "FIT evidence and policy."
        )
    expected_report = _seal(
        _generalized_chart_diagnostic_body(evidence, policy, authority)
    )
    if dict(report) != dict(expected_report):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized diagnostic report differs from its complete recomputed "
            "FIT authority and diagnostic evidence."
        )
    return evidence_sha, transform, scale_shear, generalized


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeGeneralizedChartDiagnostic:
    """Parallel intermediate-chart candidates with reproducible FIT authority."""

    evidence_sha256: str
    evidence: Lfm2MovingGaugeBoundaryEvidence = field(repr=False, compare=False)
    policy: Lfm2MovingGaugeReclosurePolicy = field(repr=False, compare=False)
    orthogonal_chart: Lfm2MovingGaugeBoundaryChart
    scale_shear_transform: Any = field(repr=False, compare=False)
    scale_shear_encoder: Any = field(repr=False, compare=False)
    generalized_encoder: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence_sha, transform, scale_shear, generalized = (
            _authenticate_generalized_diagnostic_authority(
                evidence_sha256=self.evidence_sha256,
                evidence=self.evidence,
                policy=self.policy,
                orthogonal_chart=self.orthogonal_chart,
                scale_shear_transform=self.scale_shear_transform,
                scale_shear_encoder=self.scale_shear_encoder,
                generalized_encoder=self.generalized_encoder,
                report=self.report,
            )
        )
        object.__setattr__(self, "evidence_sha256", evidence_sha)
        object.__setattr__(self, "scale_shear_transform", transform)
        object.__setattr__(self, "scale_shear_encoder", scale_shear)
        object.__setattr__(self, "generalized_encoder", generalized)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        try:
            _authenticate_generalized_diagnostic_authority(
                evidence_sha256=self.evidence_sha256,
                evidence=self.evidence,
                policy=self.policy,
                orthogonal_chart=self.orthogonal_chart,
                scale_shear_transform=self.scale_shear_transform,
                scale_shear_encoder=self.scale_shear_encoder,
                generalized_encoder=self.generalized_encoder,
                report=self.report,
            )
            return True
        except (KeyError, TypeError, ValueError):
            return False


def _generalized_e0a_causal_endpoint_body(
    diagnostic: Lfm2MovingGaugeGeneralizedChartDiagnostic,
    chart: CpcfBoundaryChart,
) -> dict[str, Any]:
    frozen = dict(diagnostic.report["candidate_hashes_frozen_before_heldout_read"])
    return {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "generalized_e0a_causal_endpoint_chart",
        "implementation": _IMPLEMENTATION,
        "boundary_id": diagnostic.report["boundary_id"],
        "interface_role": "intermediate",
        "interface_locked": False,
        "evidence_sha256": diagnostic.evidence_sha256,
        "policy_sha256": diagnostic.policy.authentication_sha256,
        "generalized_diagnostic_sha256": diagnostic.report["report_sha256"],
        "orthogonal_anchor_chart_sha256": (
            diagnostic.orthogonal_chart.report["report_sha256"]
        ),
        "selected_candidate_mode": "scale_shear_e0a",
        "selected_candidate_encoder_sha256": frozen["scale_shear"],
        "candidate_hashes_frozen_before_heldout_read": frozen,
        "heldout_used_by_solver": False,
        "heldout_read_after_candidate_freeze": diagnostic.report[
            "heldout_read_after_candidate_freeze"
        ],
        "source_width": chart.source_width,
        "target_width": chart.target_width,
        "encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(chart.mapping.encoder),
        "decoder_sha256": fingerprint_lfm2_moving_gauge_tensor(chart.mapping.decoder),
        "cpcf_chart_sha256": chart.report["report_sha256"],
        "decoder_role": (
            "orthogonal_anchor_structural_compatibility_only_not_used_by_"
            "causal_endpoint_targets"
        ),
        "causal_endpoint_encoder_authority": True,
        "physical_writer_publication_authority": False,
        "orthogonal_chart_claimed": False,
        "locked_chart_claimed": False,
        "original_storage_dtype_authentication_authority": (
            "upstream_hmac_boundary_phase_not_this_fp64_evidence_layer"
        ),
        "original_storage_dtype_claimed_by_this_layer": False,
        "canonical_promoted_computation_dtype": "float64",
        "dtype_promotion_explicit": True,
        "fit_only": True,
        "closed_form_solves_only": True,
        "numeric_quality_gate": False,
        "execution": dict(_ZERO_EXECUTION),
    }


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeGeneralizedCausalEndpointChart:
    """Authenticated E0A endpoint projection for causal-target construction.

    This is deliberately a distinct type from
    :class:`Lfm2MovingGaugeBoundaryChart`: selecting a FIT-frozen E0A
    candidate must not relabel it as an orthogonal transport or as a locked
    interface chart.  It authorizes endpoint projection only; physical writer
    publication remains a later, separately authenticated operation.
    """

    evidence_sha256: str
    diagnostic: Lfm2MovingGaugeGeneralizedChartDiagnostic = field(
        repr=False,
        compare=False,
    )
    chart: CpcfBoundaryChart = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence_sha, expected = _authenticate_generalized_e0a_endpoint_authority(
            evidence_sha256=self.evidence_sha256,
            diagnostic=self.diagnostic,
            chart=self.chart,
            report=self.report,
        )
        object.__setattr__(self, "evidence_sha256", evidence_sha)
        object.__setattr__(self, "report", expected)

    @property
    def encoder(self) -> Any:
        return self.chart.mapping.encoder

    @property
    def decoder(self) -> Any:
        return self.chart.mapping.decoder

    def verify_authentication(self) -> bool:
        try:
            _authenticate_generalized_e0a_endpoint_authority(
                evidence_sha256=self.evidence_sha256,
                diagnostic=self.diagnostic,
                chart=self.chart,
                report=self.report,
            )
            return True
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


def _generalized_e0a_endpoint_lineage(
    diagnostic: Lfm2MovingGaugeGeneralizedChartDiagnostic,
) -> dict[str, Any]:
    frozen = diagnostic.report["candidate_hashes_frozen_before_heldout_read"]
    return {
        "boundary_evidence_sha256": diagnostic.evidence_sha256,
        "generalized_diagnostic_sha256": diagnostic.report["report_sha256"],
        "orthogonal_anchor_chart_sha256": (
            diagnostic.orthogonal_chart.report["report_sha256"]
        ),
        "selected_candidate_mode": "scale_shear_e0a",
        "selected_candidate_encoder_sha256": frozen["scale_shear"],
    }


def _reconstruct_generalized_e0a_endpoint_chart(
    diagnostic: Lfm2MovingGaugeGeneralizedChartDiagnostic,
) -> CpcfBoundaryChart:
    lineage = _generalized_e0a_endpoint_lineage(diagnostic)
    mapping = _predictive_mapping(
        diagnostic.scale_shear_encoder,
        diagnostic.orthogonal_chart.decoder,
        kind="generalized_e0a_causal_endpoint_mapping",
        lineage=lineage,
    )
    return _cpcf_chart(
        diagnostic.evidence.boundary_id,
        mapping,
        kind="generalized_e0a_causal_endpoint_chart",
        lineage=lineage,
    )


def _authenticate_generalized_e0a_endpoint_authority(
    *,
    evidence_sha256: str,
    diagnostic: Lfm2MovingGaugeGeneralizedChartDiagnostic,
    chart: CpcfBoundaryChart,
    report: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    torch = _torch()
    evidence_sha = _digest(evidence_sha256, "evidence_sha256")
    if (
        not isinstance(diagnostic, Lfm2MovingGaugeGeneralizedChartDiagnostic)
        or not diagnostic.verify_authentication()
        or diagnostic.evidence_sha256 != evidence_sha
        or diagnostic.evidence.report["report_sha256"] != evidence_sha
        or not isinstance(chart, CpcfBoundaryChart)
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized causal endpoint requires its exact authenticated "
            "diagnostic and evidence."
        )
    expected_chart = _reconstruct_generalized_e0a_endpoint_chart(diagnostic)
    if (
        chart.name != expected_chart.name
        or dict(chart.mapping.report) != dict(expected_chart.mapping.report)
        or dict(chart.report) != dict(expected_chart.report)
        or chart.mapping.feature_predictors
        or not torch.equal(chart.mapping.encoder, expected_chart.mapping.encoder)
        or not torch.equal(chart.mapping.decoder, expected_chart.mapping.decoder)
        or not torch.equal(chart.mapping.decoder, diagnostic.orthogonal_chart.decoder)
        or fingerprint_lfm2_moving_gauge_tensor(chart.mapping.encoder)
        != diagnostic.report["candidate_hashes_frozen_before_heldout_read"][
            "scale_shear"
        ]
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized causal endpoint nested CPCF chart does not reconstruct "
            "from its frozen encoder and exact orthogonal-anchor decoder."
        )
    expected_report = _seal(
        _generalized_e0a_causal_endpoint_body(diagnostic, expected_chart)
    )
    if dict(report) != dict(expected_report):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized causal endpoint report is stale or malformed."
        )
    return evidence_sha, expected_report


def bind_lfm2_generalized_e0a_causal_endpoint_chart(
    diagnostic: Lfm2MovingGaugeGeneralizedChartDiagnostic,
    evidence: Lfm2MovingGaugeBoundaryEvidence,
) -> Lfm2MovingGaugeGeneralizedCausalEndpointChart:
    """Promote one frozen E0A candidate to causal endpoint authority.

    The candidate tensor was solved and frozen inside ``diagnostic`` before
    heldout rows were read.  This adapter performs no new solve.  It binds that
    exact tensor and exact diagnostic report to the fully authenticated paired
    boundary evidence.  The inherited decoder only satisfies the CPCF chart
    container's structural contract; causal targets consume the encoder only.
    """

    if (
        not isinstance(diagnostic, Lfm2MovingGaugeGeneralizedChartDiagnostic)
        or not diagnostic.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized causal endpoint diagnostic failed authentication."
        )
    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or evidence is not diagnostic.evidence
        or not evidence.verify_authentication()
        or evidence.report["report_sha256"] != diagnostic.evidence_sha256
        or evidence.interface_role != "intermediate"
        or evidence.interface_locked
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized causal endpoint evidence is not the authenticated "
            "unlocked intermediate boundary used by the diagnostic."
        )
    frozen = diagnostic.report["candidate_hashes_frozen_before_heldout_read"]
    if diagnostic.report.get("heldout_used_by_solver") is not False or frozen.get(
        "scale_shear"
    ) != fingerprint_lfm2_moving_gauge_tensor(diagnostic.scale_shear_encoder):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Generalized E0A candidate lacks an authenticated FIT-only freeze."
        )
    chart = _reconstruct_generalized_e0a_endpoint_chart(diagnostic)
    report = _seal(_generalized_e0a_causal_endpoint_body(diagnostic, chart))
    return Lfm2MovingGaugeGeneralizedCausalEndpointChart(
        evidence_sha256=diagnostic.evidence_sha256,
        diagnostic=diagnostic,
        chart=chart,
        report=report,
    )


def _base_e0_endpoint_lineage(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    orthogonal_anchor: Lfm2MovingGaugeBoundaryChart,
) -> dict[str, Any]:
    return {
        "boundary_evidence_sha256": evidence.report["report_sha256"],
        "policy_sha256": policy.authentication_sha256,
        "orthogonal_anchor_chart_sha256": orthogonal_anchor.report["report_sha256"],
        "selected_candidate_mode": "base_e0",
        "selected_candidate_encoder_sha256": (
            fingerprint_lfm2_moving_gauge_tensor(evidence.base_encoder)
        ),
    }


def _reconstruct_base_e0_endpoint_chart(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    orthogonal_anchor: Lfm2MovingGaugeBoundaryChart,
) -> CpcfBoundaryChart:
    lineage = _base_e0_endpoint_lineage(evidence, policy, orthogonal_anchor)
    mapping = _predictive_mapping(
        evidence.base_encoder,
        orthogonal_anchor.decoder,
        kind="base_e0_causal_endpoint_mapping",
        lineage=lineage,
    )
    return _cpcf_chart(
        evidence.boundary_id,
        mapping,
        kind="base_e0_causal_endpoint_chart",
        lineage=lineage,
    )


def _base_e0_causal_endpoint_body(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    orthogonal_anchor: Lfm2MovingGaugeBoundaryChart,
    chart: CpcfBoundaryChart,
) -> dict[str, Any]:
    encoder_sha = fingerprint_lfm2_moving_gauge_tensor(evidence.base_encoder)
    return {
        "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
        "kind": "base_e0_causal_endpoint_chart",
        "implementation": _IMPLEMENTATION,
        "boundary_id": evidence.boundary_id,
        "interface_role": "intermediate",
        "interface_locked": False,
        "evidence_sha256": evidence.report["report_sha256"],
        "policy_sha256": policy.authentication_sha256,
        "orthogonal_anchor_chart_sha256": orthogonal_anchor.report["report_sha256"],
        "selected_candidate_mode": "base_e0",
        "selected_candidate_encoder_sha256": encoder_sha,
        "base_encoder_sha256": encoder_sha,
        "source_width": chart.source_width,
        "target_width": chart.target_width,
        "encoder_sha256": fingerprint_lfm2_moving_gauge_tensor(chart.mapping.encoder),
        "decoder_sha256": fingerprint_lfm2_moving_gauge_tensor(chart.mapping.decoder),
        "cpcf_chart_sha256": chart.report["report_sha256"],
        "encoder_fit_performed": False,
        "encoder_transform_applied": False,
        "decoder_role": (
            "deterministic_fit_only_orthogonal_anchor_structural_compatibility_"
            "only_not_used_by_causal_endpoint_targets"
        ),
        "causal_endpoint_encoder_authority": True,
        "physical_writer_publication_authority": False,
        "locked_chart_claimed": False,
        "right_orthogonal_chart_claimed": False,
        "generalized_chart_claimed": False,
        "original_storage_dtype_authentication_authority": (
            "upstream_hmac_boundary_phase_not_this_fp64_evidence_layer"
        ),
        "original_storage_dtype_claimed_by_this_layer": False,
        "canonical_promoted_computation_dtype": "float64",
        "dtype_promotion_explicit": True,
        "fit_only": True,
        "closed_form_solves_only": True,
        "numeric_quality_gate": False,
        "execution": dict(_ZERO_EXECUTION),
    }


def _authenticate_base_e0_endpoint_authority(
    *,
    evidence_sha256: str,
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    policy: Lfm2MovingGaugeReclosurePolicy,
    orthogonal_anchor: Lfm2MovingGaugeBoundaryChart,
    chart: CpcfBoundaryChart,
    report: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    torch = _torch()
    evidence_sha = _digest(evidence_sha256, "evidence_sha256")
    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not evidence.verify_fit_authentication()
        or evidence.report["report_sha256"] != evidence_sha
        or evidence.interface_role != "intermediate"
        or evidence.interface_locked
        or not isinstance(policy, Lfm2MovingGaugeReclosurePolicy)
        or not policy.verify_authentication()
        or not isinstance(chart, CpcfBoundaryChart)
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Base-E0 causal endpoint requires exact authenticated unlocked "
            "intermediate evidence and policy."
        )
    expected_anchor = fit_lfm2_cycle_coupled_moving_gauge_boundary(
        evidence,
        policy=policy,
        evaluate_heldout=False,
    )
    if not _boundary_chart_exact_equal(orthogonal_anchor, expected_anchor):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Base-E0 structural decoder anchor does not reproduce from FIT "
            "evidence and policy."
        )
    expected_chart = _reconstruct_base_e0_endpoint_chart(
        evidence,
        policy,
        expected_anchor,
    )
    if (
        chart.name != expected_chart.name
        or dict(chart.mapping.report) != dict(expected_chart.mapping.report)
        or dict(chart.report) != dict(expected_chart.report)
        or chart.mapping.feature_predictors
        or not torch.equal(chart.mapping.encoder, expected_chart.mapping.encoder)
        or not torch.equal(chart.mapping.encoder, evidence.base_encoder)
        or not torch.equal(chart.mapping.decoder, expected_chart.mapping.decoder)
        or not torch.equal(chart.mapping.decoder, expected_anchor.decoder)
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Base-E0 causal endpoint nested CPCF chart does not reconstruct "
            "from the bit-exact base encoder and anchor decoder."
        )
    expected_report = _seal(
        _base_e0_causal_endpoint_body(
            evidence,
            policy,
            expected_anchor,
            expected_chart,
        )
    )
    if dict(report) != dict(expected_report):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Base-E0 causal endpoint report is stale or malformed."
        )
    # As with E0A, heldout evidence is authenticated only after every FIT-only
    # tensor and report has been reproduced.
    if not evidence.verify_authentication():
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Base-E0 retained evidence failed full authentication."
        )
    return evidence_sha, expected_report


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBaseCausalEndpointChart:
    """Unlocked intermediate causal endpoint using literal base E0."""

    evidence_sha256: str
    evidence: Lfm2MovingGaugeBoundaryEvidence = field(repr=False, compare=False)
    policy: Lfm2MovingGaugeReclosurePolicy = field(repr=False, compare=False)
    orthogonal_anchor: Lfm2MovingGaugeBoundaryChart = field(
        repr=False,
        compare=False,
    )
    chart: CpcfBoundaryChart = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence_sha, expected = _authenticate_base_e0_endpoint_authority(
            evidence_sha256=self.evidence_sha256,
            evidence=self.evidence,
            policy=self.policy,
            orthogonal_anchor=self.orthogonal_anchor,
            chart=self.chart,
            report=self.report,
        )
        object.__setattr__(self, "evidence_sha256", evidence_sha)
        object.__setattr__(self, "report", expected)

    @property
    def encoder(self) -> Any:
        return self.chart.mapping.encoder

    @property
    def decoder(self) -> Any:
        return self.chart.mapping.decoder

    def verify_authentication(self) -> bool:
        try:
            _authenticate_base_e0_endpoint_authority(
                evidence_sha256=self.evidence_sha256,
                evidence=self.evidence,
                policy=self.policy,
                orthogonal_anchor=self.orthogonal_anchor,
                chart=self.chart,
                report=self.report,
            )
            return True
        except (KeyError, TypeError, ValueError):
            return False


def bind_lfm2_base_e0_causal_endpoint_chart(
    evidence: Lfm2MovingGaugeBoundaryEvidence,
    *,
    policy: Lfm2MovingGaugeReclosurePolicy | None = None,
) -> Lfm2MovingGaugeBaseCausalEndpointChart:
    """Bind literal base E0 to one authenticated intermediate endpoint."""

    active_policy = policy or Lfm2MovingGaugeReclosurePolicy()
    if (
        not isinstance(evidence, Lfm2MovingGaugeBoundaryEvidence)
        or not evidence.verify_fit_authentication()
        or evidence.interface_role != "intermediate"
        or evidence.interface_locked
        or not isinstance(active_policy, Lfm2MovingGaugeReclosurePolicy)
        or not active_policy.verify_authentication()
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "Base-E0 endpoint binding requires authenticated unlocked "
            "intermediate evidence and policy."
        )
    anchor = fit_lfm2_cycle_coupled_moving_gauge_boundary(
        evidence,
        policy=active_policy,
        evaluate_heldout=False,
    )
    chart = _reconstruct_base_e0_endpoint_chart(evidence, active_policy, anchor)
    report = _seal(
        _base_e0_causal_endpoint_body(
            evidence,
            active_policy,
            anchor,
            chart,
        )
    )
    return Lfm2MovingGaugeBaseCausalEndpointChart(
        evidence_sha256=evidence.report["report_sha256"],
        evidence=evidence,
        policy=active_policy,
        orthogonal_anchor=anchor,
        chart=chart,
        report=report,
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
        evidence=evidence,
        policy=active_policy,
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
class Lfm2MovingGaugeRecapturedChildState:
    """One fresh child recapture measured between the two writer sweeps.

    The compiler never executes a model.  The caller publishes the forward
    replacements onto its live child, replays the same aligned FIT rows through
    the adapter's existing internal recapture, and hands the measured boundary
    states and writer input rows back here.  ``request_sha256`` binds the
    recapture to the exact forward replacement tensors it was measured on, so a
    stale or foreign capture fails closed instead of silently anchoring the
    reverse sweep on the wrong child.
    """

    request_sha256: str
    boundary_child_rows: tuple[Any, ...] = field(repr=False, compare=False)
    writer_input_rows: Mapping[str, Any] = field(repr=False, compare=False)
    row_alignment_sha256: str = ""
    recapture_provenance: str = ""
    child_state_sha256: str = ""
    capture_sha256: str = ""
    child_replay_capture_calls: int = 1
    child_model_forward_calls: int = 0
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        torch = _torch()
        request = _digest(self.request_sha256, "request_sha256")
        rows_input = tuple(self.boundary_child_rows)
        if len(rows_input) < 2:
            raise Lfm2MovingGaugeResidualReclosureError(
                "A mid-sweep recapture must measure every boundary state."
            )
        boundary_rows = tuple(
            _matrix(value, f"boundary_child_rows.{index}", dtype=torch.float64)
            for index, value in enumerate(rows_input)
        )
        row_count = int(boundary_rows[0].shape[0])
        if any(int(value.shape[0]) != row_count for value in boundary_rows):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Recaptured boundary rows do not share one aligned row contract."
            )
        if not isinstance(self.writer_input_rows, Mapping) or not (
            self.writer_input_rows
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "Recaptured writer input rows must be a nonempty mapping."
            )
        writer_rows = {}
        for parameter_id, value in self.writer_input_rows.items():
            name = _text(parameter_id, "writer_input_rows key")
            rows = _matrix(
                value,
                f"writer_input_rows.{name}",
                dtype=torch.float64,
            )
            if int(rows.shape[0]) != row_count:
                raise Lfm2MovingGaugeResidualReclosureError(
                    "Recaptured writer input rows do not share the boundary "
                    "row contract."
                )
            writer_rows[name] = rows
        alignment = _digest(self.row_alignment_sha256, "row_alignment_sha256")
        child_state = _digest(self.child_state_sha256, "child_state_sha256")
        capture = _digest(self.capture_sha256, "capture_sha256")
        provenance = _text(self.recapture_provenance, "recapture_provenance")
        replay_calls = self.child_replay_capture_calls
        forward_calls = self.child_model_forward_calls
        if (
            isinstance(replay_calls, bool)
            or not isinstance(replay_calls, int)
            or replay_calls < 1
            or isinstance(forward_calls, bool)
            or not isinstance(forward_calls, int)
            or forward_calls < 0
        ):
            raise Lfm2MovingGaugeResidualReclosureError(
                "A mid-sweep recapture must declare at least one child replay "
                "capture and nonnegative child forward calls."
            )
        body = {
            "schema": LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA,
            "kind": "mid_sweep_recaptured_child_state",
            "request_sha256": request,
            "row_alignment_sha256": alignment,
            "recapture_provenance": provenance,
            "recapture_provenance_sha256": _sha256(provenance),
            "child_state_sha256": child_state,
            "capture_sha256": capture,
            "rows": row_count,
            "boundary_count": len(boundary_rows),
            "boundary_child_widths": [
                int(value.shape[1]) for value in boundary_rows
            ],
            "boundary_child_rows_sha256": [
                fingerprint_lfm2_moving_gauge_tensor(value)
                for value in boundary_rows
            ],
            "writer_parameter_ids": sorted(writer_rows),
            "writer_input_rows_sha256": {
                name: fingerprint_lfm2_moving_gauge_tensor(writer_rows[name])
                for name in sorted(writer_rows)
            },
            "measured_on_live_child_after_forward_publication": True,
            "child_replay_capture_calls": replay_calls,
            "child_model_forward_calls": forward_calls,
            "parent_model_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "quality_gate": False,
        }
        supplied = dict(self.report)
        if supplied:
            if supplied != dict(_seal(body)):
                raise Lfm2MovingGaugeResidualAuthenticationError(
                    "Recaptured child-state report is stale or malformed."
                )
            report = MappingProxyType(supplied)
        else:
            report = _seal(body)
        object.__setattr__(self, "request_sha256", request)
        object.__setattr__(self, "boundary_child_rows", boundary_rows)
        object.__setattr__(self, "writer_input_rows", MappingProxyType(writer_rows))
        object.__setattr__(self, "row_alignment_sha256", alignment)
        object.__setattr__(self, "recapture_provenance", provenance)
        object.__setattr__(self, "child_state_sha256", child_state)
        object.__setattr__(self, "capture_sha256", capture)
        object.__setattr__(self, "child_replay_capture_calls", replay_calls)
        object.__setattr__(self, "child_model_forward_calls", forward_calls)
        object.__setattr__(self, "report", report)

    def verify_authentication(self) -> bool:
        try:
            return (
                verify_lfm2_moving_gauge_report(self.report)
                and self.report["boundary_child_rows_sha256"]
                == [
                    fingerprint_lfm2_moving_gauge_tensor(value)
                    for value in self.boundary_child_rows
                ]
                and self.report["writer_input_rows_sha256"]
                == {
                    name: fingerprint_lfm2_moving_gauge_tensor(
                        self.writer_input_rows[name]
                    )
                    for name in sorted(self.writer_input_rows)
                }
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


def _receipt_float(value: float) -> float | None:
    """Canonical JSON forbids non-finite numbers; record them as ``None``."""

    result = float(value)
    return result if math.isfinite(result) else None


def _receipt_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator == 0.0:
        return None
    return numerator / denominator


def _weight_frobenius_norms(
    sites: Sequence[Lfm2MovingGaugeWriterEvidence],
    weights: Mapping[str, Any],
) -> dict[str, float | None]:
    torch = _torch()
    return {
        site.parameter_id: _receipt_float(
            float(
                torch.linalg.matrix_norm(
                    weights[site.parameter_id].to(dtype=torch.float64)
                ).item()
            )
        )
        for site in sites
    }


def _boundary_acceptance_decision(
    sites: Sequence[Lfm2MovingGaugeWriterEvidence],
    *,
    sweep: str,
    transition_index: int,
    boundary_id: str,
    row_frame: str,
    incumbent_stage: str,
    features: Any,
    target: Any,
    row_weights: Any,
    incumbent_group: Any,
    proposed_group: Any,
    incumbent_weights: Mapping[str, Any],
    proposed_weights: Mapping[str, Any],
    policy: Lfm2MovingGaugeReclosurePolicy,
) -> tuple[bool, dict[str, Any]]:
    """Score one proposed writer overwrite against the boundary it overwrites.

    Both writers are measured against the *same* branch-write target frame the
    boundary's own solve used, on the same rows the sweep being guarded solves
    on, with that boundary's own row weights.  No new evidence path is opened:
    the target comes from the module's existing residual-write machinery and the
    metric is the module's existing weighted relative error.

    The optional norm-ratio ceiling is a second, independent clause on the same
    two writes.  It reads the ``write_norm_ratio`` this decision already records
    and refuses whatever the boundary metric decided, which is the only clause
    that can catch a locally-improving, globally-destructive rewrite.
    """

    torch = _torch()
    incumbent_write = features @ incumbent_group.T
    proposed_write = features @ proposed_group.T
    incumbent_metric = _relative_rmse(incumbent_write, target, row_weights)
    proposed_metric = _relative_rmse(proposed_write, target, row_weights)
    tolerance = policy.boundary_regression_tolerance
    threshold = incumbent_metric * (1.0 + tolerance)
    boundary_accepted = bool(
        math.isfinite(proposed_metric)
        and (
            not math.isfinite(incumbent_metric) or proposed_metric <= threshold
        )
    )
    incumbent_norm = float(torch.linalg.matrix_norm(incumbent_write).item())
    proposed_norm = float(torch.linalg.matrix_norm(proposed_write).item())
    write_norm_ratio = _receipt_ratio(proposed_norm, incumbent_norm)
    # A ratio the receipt could not measure -- an incumbent that writes exactly
    # zero, or a nonfinite norm -- is not above any ceiling.  Refusing it would
    # freeze a from-zero seed writer permanently; the ledger counts these
    # separately so the clause can never quietly skip a boundary.
    ceiling = policy.writer_reclosure_norm_ratio_ceiling
    ceiling_exceeded = bool(
        ceiling is not None
        and write_norm_ratio is not None
        and write_norm_ratio > ceiling
    )
    accepted = bool(boundary_accepted and not ceiling_exceeded)
    refusal_reason = (
        None
        if accepted
        else (
            _REFUSAL_BOUNDARY_REGRESSION
            if not boundary_accepted
            else _REFUSAL_WRITE_NORM_CEILING
        )
    )
    incumbent_frobenius = _weight_frobenius_norms(sites, incumbent_weights)
    proposed_frobenius = _weight_frobenius_norms(sites, proposed_weights)
    decision = {
        "sweep": sweep,
        "transition_index": int(transition_index),
        "boundary_id": boundary_id,
        "parameter_ids": [site.parameter_id for site in sites],
        "metric": _BOUNDARY_ACCEPTANCE_METRIC,
        "row_frame": row_frame,
        "rows": int(features.shape[0]),
        "incumbent_stage": incumbent_stage,
        "accepted": accepted,
        "refused": not accepted,
        "regression_tolerance": tolerance,
        "acceptance_threshold": _receipt_float(threshold),
        "incumbent_relative_write_error": _receipt_float(incumbent_metric),
        "proposed_relative_write_error": _receipt_float(proposed_metric),
        "accepted_relative_write_error": _receipt_float(
            proposed_metric if accepted else incumbent_metric
        ),
        "relative_write_error_ratio": _receipt_ratio(
            proposed_metric,
            incumbent_metric,
        ),
        "incumbent_write_l2_norm": _receipt_float(incumbent_norm),
        "proposed_write_l2_norm": _receipt_float(proposed_norm),
        "accepted_write_l2_norm": _receipt_float(
            proposed_norm if accepted else incumbent_norm
        ),
        "write_norm_ratio": write_norm_ratio,
        "incumbent_weight_frobenius_norm_by_parameter": incumbent_frobenius,
        "proposed_weight_frobenius_norm_by_parameter": proposed_frobenius,
        "accepted_weight_frobenius_norm_by_parameter": (
            proposed_frobenius if accepted else incumbent_frobenius
        ),
        "weight_frobenius_norm_ratio_by_parameter": {
            site.parameter_id: (
                None
                if incumbent_frobenius[site.parameter_id] is None
                or proposed_frobenius[site.parameter_id] is None
                else _receipt_ratio(
                    proposed_frobenius[site.parameter_id],
                    incumbent_frobenius[site.parameter_id],
                )
            )
            for site in sites
        },
    }
    if ceiling is not None:
        # Additive: an unset ceiling leaves every decision byte-identical to the
        # ledger the boundary clause sealed before this clause existed.
        decision.update(
            {
                "norm_ceiling_schema": (
                    LFM2_MOVING_GAUGE_WRITE_NORM_CEILING_SCHEMA
                ),
                "write_norm_ratio_ceiling": ceiling,
                "write_norm_ratio_metric": _WRITE_NORM_RATIO_METRIC,
                "write_norm_ratio_measured": write_norm_ratio is not None,
                "write_norm_ratio_ceiling_exceeded": ceiling_exceeded,
                "boundary_metric_accepted": boundary_accepted,
                "refusal_reason": refusal_reason,
            }
        )
    if not accepted and policy.writer_reclosure_acceptance == (
        _ACCEPTANCE_FAIL_CLOSED
    ):
        if refusal_reason == _REFUSAL_WRITE_NORM_CEILING:
            raise Lfm2MovingGaugeResidualReclosureError(
                "The moving-gauge "
                f"{sweep} sweep inflated boundary {boundary_id}: write-norm "
                f"ratio {write_norm_ratio} exceeds the ceiling {ceiling}."
            )
        raise Lfm2MovingGaugeResidualReclosureError(
            "The moving-gauge "
            f"{sweep} sweep regressed boundary {boundary_id}: relative write "
            f"error {proposed_metric} exceeds the incumbent {incumbent_metric} "
            f"at tolerance {tolerance}."
        )
    return accepted, decision


def _recaptured_boundary_write_target(
    charts: Sequence[Lfm2MovingGaugeBoundaryChart],
    boundaries: Sequence[Lfm2MovingGaugeBoundaryEvidence],
    index: int,
    *,
    recaptured_pre_state: Any,
    recaptured_post_state: Any,
    incumbent_writer_output: Any,
    transition_mode: str,
) -> Any:
    """Rebuild the forward sweep's branch-write target on recaptured states.

    Identical arithmetic to the forward sweep, with every child-side quantity
    replaced by the mid-sweep recapture the reverse sweep already solves
    against: the pre-state, the realized branch write, and the incumbent
    writer's own share of it.  Parent endpoints are cached evidence and do not
    move, so the desired total write is the same chart target as before.
    """

    upstream = boundaries[index]
    downstream = boundaries[index + 1]
    if transition_mode == "moving_gauge":
        _endpoint, desired_total_write = charts[
            index + 1
        ].chart.residual_write_target(
            actual_child_pre_state=recaptured_pre_state,
            parent_endpoint=downstream.fit_parent_rows,
        )
    else:
        desired_total_write = charts[index].chart.encode(
            downstream.fit_parent_rows - upstream.fit_parent_rows
        )
    live_branch_write = recaptured_post_state - recaptured_pre_state
    fixed_other_write = live_branch_write - incumbent_writer_output
    return (desired_total_write - fixed_other_write).contiguous()


def _terminal_anchored_composed_ridge(
    design: Any,
    downstream_transport: Any,
    terminal_residual: Any,
    anchor: Any,
    weights: Any,
    *,
    ridge_ratio: float,
    absolute_ridge_floor: float,
) -> tuple[Any, Mapping[str, Any]]:
    """Solve ``min ||X(W-anchor)J - R||_w^2 + ridge ||W-anchor||^2`` exactly.

    ``R`` is the remaining terminal residual and ``J`` the measured downstream
    response, so only the writer *delta* is fitted: the writer keeps every part
    of its current output and adds the correction that reaches the terminal.

    The reverse sweep must not invert the measured downstream response: a
    direction the recaptured child barely propagates would otherwise be
    requested back at enormous magnitude, far outside the linearization the
    recapture supports.  Solving through ``J`` instead leaves such directions
    unrequested, because the composed normal equation weights each of them by
    its own measured gain.

    ``(X^T D X) delta (J J^T) + ridge*delta = X^T D R J^T`` is separated by one
    symmetric eigendecomposition per side and solved coordinate-wise.  There is
    no iterative optimizer, pseudoinverse threshold, or autograd.  At ``J = I``
    this reduces exactly to the anchored ridge the forward-only sweep uses.
    """

    torch = _torch()
    x = _matrix(design, "composed ridge design", dtype=torch.float64)
    j = _matrix(downstream_transport, "composed ridge transport", dtype=torch.float64)
    r = _matrix(terminal_residual, "composed ridge terminal residual", dtype=torch.float64)
    a = _matrix(anchor, "composed ridge anchor", dtype=torch.float64)
    if (
        int(x.shape[0]) != int(r.shape[0])
        or int(x.shape[1]) != int(a.shape[0])
        or int(a.shape[1]) != int(j.shape[0])
        or int(j.shape[1]) != int(r.shape[1])
        or int(weights.numel()) != int(x.shape[0])
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Terminal-anchored composed ridge geometry is inconsistent."
        )
    ratio = _finite_nonnegative(ridge_ratio, "ridge_ratio")
    floor = _finite_positive(absolute_ridge_floor, "absolute_ridge_floor")
    transport_spectral = float(torch.linalg.matrix_norm(j, ord=2).item())
    design_scale = _ridge_scale(x, weights)
    scale = design_scale * transport_spectral * transport_spectral
    ridge = max(floor, ratio * scale)
    root = torch.sqrt(weights).unsqueeze(1)
    weighted_x = x * root
    # Only the delta is fitted: the writer's current output is preserved and
    # the terminal residual is what the correction has to deliver.
    weighted_residual = r * root
    gram = weighted_x.T @ weighted_x
    kernel = j @ j.T
    cross = weighted_x.T @ weighted_residual @ j.T
    design_values, design_basis = torch.linalg.eigh(gram)
    transport_values, transport_basis = torch.linalg.eigh(kernel)
    design_values = design_values.clamp_min(0.0)
    transport_values = transport_values.clamp_min(0.0)
    rotated = design_basis.T @ cross @ transport_basis
    denominator = design_values.unsqueeze(1) * transport_values.unsqueeze(0) + ridge
    delta = design_basis @ (rotated / denominator) @ transport_basis.T
    replacement = (a + delta).contiguous()
    if not bool(torch.isfinite(replacement).all().item()):
        raise Lfm2MovingGaugeResidualReclosureError(
            "Terminal-anchored composed ridge replacement is non-finite."
        )
    report = _seal(
        {
            "schema": LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA,
            "kind": "terminal_anchored_composed_downstream_ridge",
            "equation": ("(X^T D X)delta(J J^T)+ridge*delta=X^T D R J^T"),
            "fitted_quantity": "writer_delta_against_remaining_terminal_residual",
            "solve_form": "paired_symmetric_eigenbasis",
            "rows": int(x.shape[0]),
            "input_width": int(x.shape[1]),
            "downstream_width": int(j.shape[0]),
            "terminal_width": int(j.shape[1]),
            "ridge_ratio": ratio,
            "design_ridge_scale": design_scale,
            "downstream_transport_spectral_norm": transport_spectral,
            "ridge_scale": scale,
            "effective_ridge": ridge,
            "design_gram_eigenvalue_max": float(design_values.max().item()),
            "design_gram_eigenvalue_min": float(design_values.min().item()),
            "downstream_gram_eigenvalue_max": float(transport_values.max().item()),
            "downstream_gram_eigenvalue_min": float(transport_values.min().item()),
            "design_sha256": fingerprint_lfm2_moving_gauge_tensor(x),
            "downstream_transport_sha256": fingerprint_lfm2_moving_gauge_tensor(j),
            "terminal_residual_sha256": fingerprint_lfm2_moving_gauge_tensor(r),
            "anchor_sha256": fingerprint_lfm2_moving_gauge_tensor(a),
            "replacement_sha256": fingerprint_lfm2_moving_gauge_tensor(replacement),
            "downstream_transport_inverted": False,
            "pseudoinverse_used": False,
            "closed_form_linear_solve": True,
            "autograd_used": False,
            "quality_gate": False,
        }
    )
    return replacement, report


def _recaptured_writer_group(
    sites: Sequence[Lfm2MovingGaugeWriterEvidence],
    current: Mapping[str, Any],
    recaptured: Lfm2MovingGaugeRecapturedChildState,
) -> tuple[Any, Any, tuple[int, ...]]:
    """Group one transition's writers against freshly recaptured input rows."""

    torch = _torch()
    features = torch.cat(
        tuple(recaptured.writer_input_rows[site.parameter_id] for site in sites),
        dim=1,
    )
    widths = tuple(int(site.seed_weight.shape[1]) for site in sites)
    weights = torch.cat(
        tuple(current[site.parameter_id].to(dtype=torch.float64) for site in sites),
        dim=1,
    )
    return features, weights, widths


def _mid_sweep_recapture_request(
    boundaries: Sequence[Lfm2MovingGaugeBoundaryEvidence],
    sites: Sequence[Lfm2MovingGaugeWriterEvidence],
    current: Mapping[str, Any],
    *,
    policy: Lfm2MovingGaugeReclosurePolicy,
    row_alignment_sha256: str,
) -> Mapping[str, Any]:
    """Seal the exact forward publication the recapture must be measured on."""

    ordered = tuple(sorted(sites, key=lambda value: value.parameter_id))
    replacements = {
        site.parameter_id: current[site.parameter_id] for site in ordered
    }
    body = {
        "schema": LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA,
        "kind": "mid_sweep_recapture_request",
        "policy_sha256": policy.authentication_sha256,
        "row_alignment_sha256": row_alignment_sha256,
        "rows": int(boundaries[0].fit_parent_rows.shape[0]),
        "boundary_ids": [value.boundary_id for value in boundaries],
        "boundary_child_widths": [value.child_width for value in boundaries],
        "parameter_ids": [site.parameter_id for site in ordered],
        "writer_input_widths": {
            site.parameter_id: int(site.seed_weight.shape[1]) for site in ordered
        },
        "forward_replacement_weight_sha256": {
            site.parameter_id: fingerprint_lfm2_moving_gauge_tensor(
                replacements[site.parameter_id]
            )
            for site in ordered
        },
        "forward_sweep_completed": True,
        "reverse_sweep_pending": True,
        "recapture_measures_live_child_after_forward_publication": True,
        "parent_recapture_requested": False,
    }
    return MappingProxyType(
        {
            "request_body": MappingProxyType(body),
            "request_sha256": _sha256(body),
            "forward_replacement_weights": MappingProxyType(replacements),
        }
    )


def _authenticate_mid_sweep_recapture(
    value: Any,
    *,
    request: Mapping[str, Any],
    boundaries: Sequence[Lfm2MovingGaugeBoundaryEvidence],
    sites: Sequence[Lfm2MovingGaugeWriterEvidence],
) -> Lfm2MovingGaugeRecapturedChildState:
    """Fail closed unless the recapture measured this exact publication."""

    if (
        not isinstance(value, Lfm2MovingGaugeRecapturedChildState)
        or not value.verify_authentication()
        or value.request_sha256 != request["request_sha256"]
    ):
        raise Lfm2MovingGaugeResidualAuthenticationError(
            "The mid-sweep child recapture does not authenticate against the "
            "forward writer publication it must be measured on."
        )
    body = request["request_body"]
    measured_widths = [
        int(rows.shape[1]) for rows in value.boundary_child_rows
    ]
    if (
        value.row_alignment_sha256 != body["row_alignment_sha256"]
        or len(value.boundary_child_rows) != len(boundaries)
        or measured_widths != list(body["boundary_child_widths"])
        or int(value.boundary_child_rows[0].shape[0]) != int(body["rows"])
        or sorted(value.writer_input_rows) != sorted(body["parameter_ids"])
        or any(
            int(value.writer_input_rows[site.parameter_id].shape[1])
            != int(site.seed_weight.shape[1])
            for site in sites
        )
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "The mid-sweep child recapture does not match the boundary, row, "
            "or writer geometry of this reclosure."
        )
    return value


def _recaptured_terminal_anchored_reverse_sweep(
    boundaries: Sequence[Lfm2MovingGaugeBoundaryEvidence],
    grouped: Mapping[int, Sequence[Lfm2MovingGaugeWriterEvidence]],
    current: dict[str, Any],
    *,
    charts: Sequence[Lfm2MovingGaugeBoundaryChart],
    terminal_target: Any,
    recaptured: Lfm2MovingGaugeRecapturedChildState,
    policy: Lfm2MovingGaugeReclosurePolicy,
    acceptance_decisions: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Any]:
    """Re-solve writers back-to-front against the measured downstream response.

    The forward-only sweep transports every local correction to the terminal
    through the exact physical identity, which is only true while all later
    writer inputs stay frozen.  Here the child was actually replayed after the
    forward publication, so three approximations are replaced by measurements:
    the terminal residual is the one the recaptured child actually has, each
    writer is solved on its recaptured input rows, and the downstream response
    of the terminal state to the state at each boundary is measured on those
    rows by one anchored ridge whose anchor is the identity skip itself.  When
    the recaptured rows carry no downstream signal the fit stays at that anchor,
    which is exactly the forward-only approximation, so the mode degrades to the
    old behavior instead of inventing a rotation.  Each writer is then solved
    *through* the measured response rather than through its inverse.
    """

    torch = _torch()
    layer_count = len(boundaries) - 1
    measured = recaptured.boundary_child_rows
    terminal_measured = measured[-1]
    terminal_weights = boundaries[-1].fit_weights
    terminal_width = int(terminal_measured.shape[1])
    terminal_residual = (terminal_target - terminal_measured).contiguous()
    reverse_steps: list[dict[str, Any]] = []
    for index in range(layer_count - 1, -1, -1):
        sites = tuple(grouped[index])
        features, current_group, widths = _recaptured_writer_group(
            sites,
            current,
            recaptured,
        )
        downstream_rows = measured[index + 1]
        downstream_width = int(downstream_rows.shape[1])
        anchor = torch.eye(downstream_width, terminal_width, dtype=torch.float64)
        if index == layer_count - 1:
            # The downstream boundary IS the terminal boundary: the physical
            # relation is the identity, not something to be estimated, so this
            # step keeps the ordinary writer ridge instead of the trust region.
            row_transport = anchor
            transport_report = None
            transport_kind = _TERMINAL_BOUNDARY_TRANSPORT_KIND
            step_ridge_ratio = policy.writer_ridge_ratio
        else:
            row_transport, transport_report = _anchored_ridge_replacement(
                downstream_rows,
                terminal_measured,
                anchor,
                boundaries[index].fit_weights,
                ridge_ratio=policy.reverse_transport_ridge_ratio,
                absolute_ridge_floor=policy.absolute_ridge_floor,
            )
            transport_kind = _MEASURED_TRANSPORT_KIND
            # The measured response is a linearization taken at the recaptured
            # point.  Its low-gain directions would otherwise buy a negligible
            # terminal improvement with an enormous local step, which is exactly
            # where the linearization stops describing the child.
            step_ridge_ratio = policy.recaptured_step_ridge_ratio
        replacement_columns, solve_report = _terminal_anchored_composed_ridge(
            features,
            row_transport,
            terminal_residual,
            current_group.T,
            boundaries[index].fit_weights,
            ridge_ratio=step_ridge_ratio,
            absolute_ridge_floor=policy.absolute_ridge_floor,
        )
        previous_group = current_group
        previous_weights = {
            site.parameter_id: current[site.parameter_id] for site in sites
        }
        split = _split_cast_replacements(
            sites,
            replacement_columns.T,
            widths,
        )
        current.update(split)
        _, new_group, _ = _recaptured_writer_group(sites, current, recaptured)
        if acceptance_decisions is not None:
            # Both writers are scored on the recaptured rows this sweep already
            # solves against, so the guard and the solve see one child state.
            accepted, decision = _boundary_acceptance_decision(
                sites,
                sweep=_REVERSE_SWEEP_NAME,
                transition_index=index,
                boundary_id=boundaries[index + 1].boundary_id,
                row_frame=_RECAPTURED_ROW_FRAME,
                incumbent_stage="post_forward_sweep",
                features=features,
                target=_recaptured_boundary_write_target(
                    charts,
                    boundaries,
                    index,
                    recaptured_pre_state=measured[index],
                    recaptured_post_state=measured[index + 1],
                    incumbent_writer_output=features @ previous_group.T,
                    transition_mode=policy.transition_mode,
                ),
                row_weights=boundaries[index].fit_weights,
                incumbent_group=previous_group,
                proposed_group=new_group,
                incumbent_weights=previous_weights,
                proposed_weights=current,
                policy=policy,
            )
            acceptance_decisions.append(decision)
            if not accepted:
                current.update(previous_weights)
                _, new_group, _ = _recaptured_writer_group(
                    sites,
                    current,
                    recaptured,
                )
        local_realized = features @ (new_group - previous_group).T
        terminal_realized = local_realized @ row_transport
        local_norm = float(torch.linalg.vector_norm(local_realized).item())
        realized_norm = float(torch.linalg.vector_norm(terminal_realized).item())
        requested_norm = float(torch.linalg.vector_norm(terminal_residual).item())
        relative_before = _relative_rmse(
            terminal_target - terminal_residual,
            terminal_target,
            terminal_weights,
        )
        terminal_residual = (terminal_residual - terminal_realized).contiguous()
        reverse_steps.append(
            {
                "transition_index": index,
                "parameter_ids": [site.parameter_id for site in sites],
                "transport_reciprocal_report_sha256": None,
                "downstream_transport_inverted": False,
                "downstream_transport_kind": transport_kind,
                "chart_transition_used_as_physical_jacobian": False,
                "solve_report_sha256": solve_report["report_sha256"],
                "downstream_transport_sha256": (
                    fingerprint_lfm2_moving_gauge_tensor(row_transport)
                ),
                "requested_terminal_l2_norm": requested_norm,
                "realized_local_l2_norm": local_norm,
                "realized_terminal_l2_norm": realized_norm,
                "remaining_terminal_l2_norm": float(
                    torch.linalg.vector_norm(terminal_residual).item()
                ),
                "downstream_transport_report_sha256": (
                    None
                    if transport_report is None
                    else transport_report["report_sha256"]
                ),
                "recaptured_downstream_rows_sha256": (
                    fingerprint_lfm2_moving_gauge_tensor(downstream_rows)
                ),
                "recaptured_writer_input_rows_sha256": (
                    fingerprint_lfm2_moving_gauge_tensor(features)
                ),
                "terminal_amplification_ratio": (
                    None if local_norm == 0.0 else realized_norm / local_norm
                ),
                "terminal_relative_rmse_before": relative_before,
                "terminal_relative_rmse_after": _relative_rmse(
                    terminal_target - terminal_residual,
                    terminal_target,
                    terminal_weights,
                ),
                "writer_inputs_measured_after_forward_publication": True,
            }
        )
    return reverse_steps, terminal_residual


def compile_lfm2_moving_gauge_residual_reclosure(
    boundary_evidence: Sequence[Lfm2MovingGaugeBoundaryEvidence],
    writer_evidence: Sequence[Lfm2MovingGaugeWriterEvidence],
    *,
    policy: Lfm2MovingGaugeReclosurePolicy | None = None,
    mid_sweep_recapture_provider: Any | None = None,
) -> Lfm2MovingGaugeDenseReplacementState:
    """Compile one complete target-dtype forward/reverse reclosure transaction."""

    torch = _torch()
    active_policy = policy or Lfm2MovingGaugeReclosurePolicy()
    boundaries_input = tuple(boundary_evidence)
    sites_input = tuple(writer_evidence)
    recaptured_reverse = (
        isinstance(active_policy, Lfm2MovingGaugeReclosurePolicy)
        and active_policy.reverse_sweep_mode == "recaptured_terminal_anchored"
    )
    guarded_acceptance = (
        isinstance(active_policy, Lfm2MovingGaugeReclosurePolicy)
        and active_policy.writer_reclosure_acceptance != _ACCEPTANCE_UNGUARDED
    )
    if recaptured_reverse != (mid_sweep_recapture_provider is not None) or (
        mid_sweep_recapture_provider is not None
        and not callable(mid_sweep_recapture_provider)
    ):
        raise Lfm2MovingGaugeResidualReclosureError(
            "The recaptured reverse sweep requires exactly one callable "
            "mid-sweep child recapture provider; the forward-only sweep "
            "accepts none."
        )
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
        Lfm2MovingGaugeRecapturedChildState | None,
        Mapping[str, Any] | None,
    ]:
        acceptance_decisions: list[dict[str, Any]] | None = (
            [] if guarded_acceptance else None
        )
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
        # The boundary each transition's writers are solved against, kept in the
        # module's own target frame so the reverse sweep is scored against the
        # exact frame the boundary's own solve used.
        boundary_write_targets: dict[int, Any] = {}
        seed_boundary_error: dict[int, float] = {}
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
            boundary_write_targets[index] = target_writer_output
            if acceptance_decisions is not None:
                seed_boundary_error[index] = _relative_rmse(
                    existing_writer_output,
                    target_writer_output,
                    upstream.fit_weights,
                )
                accepted, decision = _boundary_acceptance_decision(
                    sites,
                    sweep=_FORWARD_SWEEP_NAME,
                    transition_index=index,
                    boundary_id=downstream.boundary_id,
                    row_frame=_FIT_ROW_FRAME,
                    incumbent_stage="pre_reclosure_seed",
                    features=features,
                    target=target_writer_output,
                    row_weights=upstream.fit_weights,
                    incumbent_group=seed_group,
                    proposed_group=group_after,
                    incumbent_weights=previous,
                    proposed_weights=current,
                    policy=active_policy,
                )
                acceptance_decisions.append(decision)
                if not accepted:
                    current.update(previous)
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
        recaptured: Lfm2MovingGaugeRecapturedChildState | None = None
        if recaptured_reverse:
            request = _mid_sweep_recapture_request(
                boundaries_input,
                sites_input,
                current,
                policy=active_policy,
                row_alignment_sha256=alignment,
            )
            recaptured = _authenticate_mid_sweep_recapture(
                mid_sweep_recapture_provider(request),
                request=request,
                boundaries=boundaries_input,
                sites=sites_input,
            )
            reverse_steps, terminal_residual = (
                _recaptured_terminal_anchored_reverse_sweep(
                    boundaries_input,
                    grouped,
                    current,
                    charts=charts,
                    terminal_target=terminal_target,
                    recaptured=recaptured,
                    policy=active_policy,
                    acceptance_decisions=acceptance_decisions,
                )
            )
        else:
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
            # here is the exact physical identity.  Nonlinear downstream response
            # is measured by the recaptured reverse sweep or the later causal
            # recapture/reclosure stage instead of being invented from E/D chart
            # rotations.
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
                previous_weights = {
                    site.parameter_id: current[site.parameter_id]
                    for site in sites
                }
                split = _split_cast_replacements(
                    sites,
                    replacement_columns.T,
                    widths,
                )
                current.update(split)
                _, new_group, _ = _writer_group(sites, current)
                if acceptance_decisions is not None:
                    # The terminal objective may not pay for itself with this
                    # boundary's own write: score the proposal against the frame
                    # the forward solve targeted, on the same FIT rows.
                    accepted, decision = _boundary_acceptance_decision(
                        sites,
                        sweep=_REVERSE_SWEEP_NAME,
                        transition_index=index,
                        boundary_id=boundaries_input[index + 1].boundary_id,
                        row_frame=_FIT_ROW_FRAME,
                        incumbent_stage="post_forward_sweep",
                        features=features,
                        target=boundary_write_targets[index],
                        row_weights=boundaries_input[index].fit_weights,
                        incumbent_group=previous_group,
                        proposed_group=new_group,
                        incumbent_weights=previous_weights,
                        proposed_weights=current,
                        policy=active_policy,
                    )
                    acceptance_decisions.append(decision)
                    if not accepted:
                        current.update(previous_weights)
                        _, new_group, _ = _writer_group(sites, current)
                local_realized = features @ (new_group - previous_group).T
                terminal_realized = local_realized @ row_transport
                terminal_residual = (terminal_residual - terminal_realized).contiguous()
                reverse_steps.append(
                    {
                        "transition_index": index,
                        "parameter_ids": [site.parameter_id for site in sites],
                        "transport_reciprocal_report_sha256": None,
                        "downstream_transport_kind": _IDENTITY_TRANSPORT_KIND,
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
        acceptance: Mapping[str, Any] | None = None
        if acceptance_decisions is not None:
            # One end-to-end line per boundary, always in the FIT frame the
            # boundary's own solve used, so a receipt reader can check the
            # monotonicity claim without replaying either sweep.
            per_boundary = []
            for index in range(layer_count):
                sites = tuple(grouped[index])
                features, final_group, _widths = _writer_group(sites, current)
                final_error = _relative_rmse(
                    features @ final_group.T,
                    boundary_write_targets[index],
                    boundaries_input[index].fit_weights,
                )
                seed_error = seed_boundary_error[index]
                per_boundary.append(
                    {
                        "transition_index": index,
                        "boundary_id": boundaries_input[index + 1].boundary_id,
                        "parameter_ids": [site.parameter_id for site in sites],
                        "row_frame": _FIT_ROW_FRAME,
                        "seed_relative_write_error": _receipt_float(seed_error),
                        "published_relative_write_error": _receipt_float(
                            final_error
                        ),
                        "relative_write_error_ratio": _receipt_ratio(
                            final_error,
                            seed_error,
                        ),
                        "published_no_worse_than_seed": bool(
                            math.isfinite(final_error)
                            and (
                                not math.isfinite(seed_error)
                                or final_error
                                <= seed_error
                                * (
                                    1.0
                                    + active_policy.boundary_regression_tolerance
                                )
                                ** 2
                            )
                        ),
                    }
                )
            refused = [
                value for value in acceptance_decisions if not value["accepted"]
            ]
            acceptance_body = {
                "schema": LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA,
                "kind": "moving_gauge_boundary_acceptance_ledger",
                "writer_reclosure_acceptance": (
                    active_policy.writer_reclosure_acceptance
                ),
                "boundary_regression_tolerance": (
                    active_policy.boundary_regression_tolerance
                ),
                "metric": _BOUNDARY_ACCEPTANCE_METRIC,
                "reverse_sweep_mode": active_policy.reverse_sweep_mode,
                "reverse_row_frame": (
                    _RECAPTURED_ROW_FRAME
                    if recaptured_reverse
                    else _FIT_ROW_FRAME
                ),
                "evaluated_writer_overwrites": len(acceptance_decisions),
                "accepted_writer_overwrites": (
                    len(acceptance_decisions) - len(refused)
                ),
                "refused_writer_overwrites": len(refused),
                "refused_by_sweep": {
                    _FORWARD_SWEEP_NAME: sum(
                        1
                        for value in refused
                        if value["sweep"] == _FORWARD_SWEEP_NAME
                    ),
                    _REVERSE_SWEEP_NAME: sum(
                        1
                        for value in refused
                        if value["sweep"] == _REVERSE_SWEEP_NAME
                    ),
                },
                "refused_transition_indices": sorted(
                    {int(value["transition_index"]) for value in refused}
                ),
                "decisions": list(acceptance_decisions),
                "per_boundary": per_boundary,
                "every_published_boundary_no_worse_than_seed": all(
                    value["published_no_worse_than_seed"]
                    for value in per_boundary
                ),
                "quality_gate": False,
            }
            if active_policy.writer_reclosure_norm_ratio_ceiling is not None:
                # Additive, and split by reason so the ceiling's own effect is
                # countable: how many overwrites it refused that the boundary
                # metric would have kept, and how many ratios it could not
                # measure at all.
                ceiling_refused = [
                    value
                    for value in refused
                    if value["refusal_reason"] == _REFUSAL_WRITE_NORM_CEILING
                ]
                acceptance_body = {
                    **acceptance_body,
                    "norm_ceiling_schema": (
                        LFM2_MOVING_GAUGE_WRITE_NORM_CEILING_SCHEMA
                    ),
                    "writer_reclosure_norm_ratio_ceiling": (
                        active_policy.writer_reclosure_norm_ratio_ceiling
                    ),
                    "write_norm_ratio_metric": _WRITE_NORM_RATIO_METRIC,
                    "norm_ratio_ceiling_refused_writer_overwrites": len(
                        ceiling_refused
                    ),
                    "norm_ratio_ceiling_refused_transition_indices": sorted(
                        {
                            int(value["transition_index"])
                            for value in ceiling_refused
                        }
                    ),
                    "boundary_metric_refused_writer_overwrites": (
                        len(refused) - len(ceiling_refused)
                    ),
                    "unmeasured_write_norm_ratio_writer_overwrites": sum(
                        1
                        for value in acceptance_decisions
                        if not value["write_norm_ratio_measured"]
                    ),
                    "maximum_accepted_write_norm_ratio": max(
                        (
                            value["write_norm_ratio"]
                            for value in acceptance_decisions
                            if value["accepted"]
                            and value["write_norm_ratio"] is not None
                        ),
                        default=None,
                    ),
                }
            acceptance = _seal(acceptance_body)
        diagnostics_body = {
            "schema": LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA,
            "kind": "moving_gauge_forward_reverse_diagnostics",
            "transition_mode": active_policy.transition_mode,
            "forward_steps": forward_steps,
            "reverse_steps": reverse_steps,
            "reverse_downstream_transport_kind": (
                _MEASURED_TRANSPORT_KIND
                if recaptured_reverse
                else _IDENTITY_TRANSPORT_KIND
            ),
            "chart_transition_used_as_physical_jacobian": False,
            "nonlinear_downstream_jacobian_claimed": False,
            "reverse_transport_ridge_ratio_used": recaptured_reverse,
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
        if recaptured is not None:
            diagnostics_body.update(
                {
                    "policy_schema": LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA,
                    "reverse_sweep_mode": active_policy.reverse_sweep_mode,
                    "mid_sweep_child_recapture_count": 1,
                    "mid_sweep_recapture_sha256": (
                        recaptured.report["report_sha256"]
                    ),
                    "mid_sweep_recapture_request_sha256": (
                        recaptured.request_sha256
                    ),
                    "mid_sweep_recaptured_child_state_sha256": (
                        recaptured.child_state_sha256
                    ),
                    "frozen_writer_input_identity_transport": False,
                    "downstream_response_measured_on_recaptured_child_rows": True,
                    "measured_downstream_transport_anchor": (
                        "exact_physical_identity_skip"
                    ),
                    "measured_forward_terminal_relative_rmse": _relative_rmse(
                        recaptured.boundary_child_rows[-1],
                        terminal_target,
                        boundaries_input[-1].fit_weights,
                    ),
                    "terminal_amplification_ratio_by_transition": {
                        str(step["transition_index"]): (
                            step["terminal_amplification_ratio"]
                        )
                        for step in reverse_steps
                    },
                }
            )
        if acceptance is not None:
            diagnostics_body["boundary_acceptance"] = dict(acceptance)
        diagnostics = _seal(diagnostics_body)
        return (
            charts,
            transitions,
            MappingProxyType(current),
            diagnostics,
            recaptured,
            acceptance,
        )

    (
        (charts, transitions, replacements, diagnostics, recaptured, acceptance),
        guard,
    ) = run_fold_only_callback(
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
    if recaptured is not None:
        body.update(
            {
                "policy_schema": LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA,
                "reverse_sweep_mode": active_policy.reverse_sweep_mode,
                "mid_sweep_recapture_sha256": recaptured.report["report_sha256"],
                "mid_sweep_recapture": dict(recaptured.report),
                "forward_then_recaptured_reverse_reconciliation": True,
                "frozen_writer_input_identity_transport": False,
                # The recapture is executed by the caller's provider; the module
                # still never constructs, loads, or runs a model itself.
                "execution": {
                    **body["execution"],
                    "target_model_forward_calls": (
                        recaptured.child_model_forward_calls
                    ),
                    "child_replay_capture_calls": (
                        recaptured.child_replay_capture_calls
                    ),
                    "child_recapture_executed_by_caller_provider": True,
                },
            }
        )
    if acceptance is not None:
        # The full per-boundary ledger lives once, inside the sealed
        # diagnostics; the publication body carries its digest and the counts a
        # receipt reader needs before deciding to open it.
        body.update(
            {
                "acceptance_schema": (
                    LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA
                ),
                "writer_reclosure_acceptance": (
                    active_policy.writer_reclosure_acceptance
                ),
                "boundary_regression_tolerance": (
                    active_policy.boundary_regression_tolerance
                ),
                "boundary_acceptance_sha256": acceptance["report_sha256"],
                "boundary_acceptance_summary": {
                    key: acceptance[key]
                    for key in (
                        "metric",
                        "reverse_row_frame",
                        "evaluated_writer_overwrites",
                        "accepted_writer_overwrites",
                        "refused_writer_overwrites",
                        "refused_by_sweep",
                        "refused_transition_indices",
                        "every_published_boundary_no_worse_than_seed",
                    )
                },
                "per_boundary_monotone_writer_reclosure": True,
                "terminal_objective_may_overwrite_a_regressed_boundary": False,
            }
        )
        if active_policy.writer_reclosure_norm_ratio_ceiling is not None:
            body.update(
                {
                    "norm_ceiling_schema": (
                        LFM2_MOVING_GAUGE_WRITE_NORM_CEILING_SCHEMA
                    ),
                    "writer_reclosure_norm_ratio_ceiling": (
                        active_policy.writer_reclosure_norm_ratio_ceiling
                    ),
                    "boundary_acceptance_summary": {
                        **body["boundary_acceptance_summary"],
                        **{
                            key: acceptance[key]
                            for key in (
                                "norm_ratio_ceiling_refused_writer_overwrites",
                                (
                                    "norm_ratio_ceiling_refused_transition_"
                                    "indices"
                                ),
                                "boundary_metric_refused_writer_overwrites",
                                (
                                    "unmeasured_write_norm_ratio_writer_"
                                    "overwrites"
                                ),
                                "maximum_accepted_write_norm_ratio",
                            )
                        },
                    },
                    "write_norm_ratio_ceiling_refuses_any_boundary_metric": (
                        True
                    ),
                }
            )
    return Lfm2MovingGaugeDenseReplacementState(
        parameter_ids=ids,
        replacement_weights=values,
        boundaries=charts,
        transitions=transitions,
        report=_seal(body),
    )


__all__ = [
    "LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA",
    "LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA",
    "LFM2_MOVING_GAUGE_RESIDUAL_RECLOSURE_SCHEMA",
    "LFM2_MOVING_GAUGE_WRITER_RECLOSURE_ACCEPTANCE_MODES",
    "LFM2_MOVING_GAUGE_WRITER_RECLOSURE_FAIL_CLOSED",
    "LFM2_MOVING_GAUGE_WRITER_RECLOSURE_REQUIRE_IMPROVEMENT",
    "LFM2_MOVING_GAUGE_WRITER_RECLOSURE_UNGUARDED",
    "LFM2_MOVING_GAUGE_WRITE_NORM_CEILING_SCHEMA",
    "Lfm2MovingGaugeBaseCausalEndpointChart",
    "Lfm2MovingGaugeBoundaryChart",
    "Lfm2MovingGaugeBoundaryEvidence",
    "Lfm2MovingGaugeDenseReplacementState",
    "Lfm2MovingGaugeGeneralizedCausalEndpointChart",
    "Lfm2MovingGaugeGeneralizedChartDiagnostic",
    "Lfm2MovingGaugeOrthogonalCausalEndpointChart",
    "PreparedAnchoredWriterProblem",
    "Lfm2MovingGaugeRecapturedChildState",
    "Lfm2MovingGaugeReclosurePolicy",
    "Lfm2MovingGaugeResidualAuthenticationError",
    "Lfm2MovingGaugeResidualReclosureError",
    "Lfm2MovingGaugeTransition",
    "Lfm2MovingGaugeWriterEvidence",
    "bind_lfm2_base_e0_causal_endpoint_chart",
    "bind_lfm2_generalized_e0a_causal_endpoint_chart",
    "bind_lfm2_orthogonal_causal_endpoint_chart",
    "compile_lfm2_moving_gauge_residual_reclosure",
    "fingerprint_lfm2_moving_gauge_tensor",
    "fit_lfm2_cycle_coupled_moving_gauge_boundary",
    "fit_lfm2_generalized_moving_gauge_chart_diagnostic",
    "fit_lfm2_moving_gauge_residual_chart_adapter",
    "lfm2_moving_gauge_sequential_fit_provenance",
    "prepare_lfm2_moving_gauge_anchored_writer_problem",
    "solve_lfm2_moving_gauge_anchored_writer_replacement",
    "solve_lfm2_prepared_anchored_writer_replacement",
    "solve_lfm2_trace_scaled_weighted_anchored_ridge",
    "transport_lfm2_cycle_coupled_boundary_from_upstream",
    "verify_lfm2_moving_gauge_report",
]
