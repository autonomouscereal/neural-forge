"""Numerical primitives for endpoint-aware LFM2 CPCF short-conv folding.

This module is intentionally independent from the production compiler.  It
contains deterministic, closed-form building blocks that a later compiler can
compose without granting fit or selection code access to heldout validation
evidence.

The implementation performs no gradient calculation, optimizer step, pruning,
or quantization.  Parent tensors are transported into a smaller, blank target
through authenticated policies, fit-only activation geometry, and bounded
closed-form projections.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable


LFM2_CPCF_SHORT_CONV_V4_SCHEMA = "neural-forge-lfm2-cpcf-short-conv-v4"
_ALLOWED_SPLITS = frozenset({"fit", "selection", "validation"})
_ALLOWED_TARGET_DTYPES = frozenset({"bfloat16", "float16", "float32", "float64"})


class Lfm2CpcfShortConvV4Error(ValueError):
    """V4 short-convolution folding evidence or policy is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF short-conv v4 folding."
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
        raise Lfm2CpcfShortConvV4Error(
            "V4 metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _sha256(detached)})


def _verify_sealed_report(report: Mapping[str, Any]) -> bool:
    if not isinstance(report, Mapping):
        return False
    body = dict(report)
    supplied = body.pop("report_sha256", None)
    return isinstance(supplied, str) and supplied == _sha256(body)


def _sha256_text(value: Any, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise Lfm2CpcfShortConvV4Error(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfShortConvV4Error(
            f"{name} must be a materialized dense tensor for authentication."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    byte_view = tensor.reshape(-1).contiguous().view(torch.uint8)
    digest.update(byte_view.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_tensor(
    value: Any,
    name: str,
    *,
    minimum_dimensions: int,
    maximum_dimensions: int | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim < minimum_dimensions
        or (maximum_dimensions is not None and value.ndim > maximum_dimensions)
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        upper = (
            "" if maximum_dimensions is None else f" through rank-{maximum_dimensions}"
        )
        raise Lfm2CpcfShortConvV4Error(
            f"{name} must be a nonempty dense real floating rank-"
            f"{minimum_dimensions}{upper} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfShortConvV4Error(f"{name} contains NaN or infinity.")
    return result


def _snapshot_dense_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
    ):
        raise Lfm2CpcfShortConvV4Error(
            f"{name} must be a nonempty materialized dense tensor."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _snapshot_optional_mask(value: Any | None, name: str) -> Any | None:
    if value is None:
        return None
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
    ):
        raise Lfm2CpcfShortConvV4Error(
            f"{name} must be a nonempty materialized dense mask."
        )
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()


def _canonical_unique_cases(cases: Sequence[Any], name: str) -> tuple[Any, ...]:
    values = tuple(cases)
    if not values:
        raise Lfm2CpcfShortConvV4Error(f"{name} requires at least one case.")
    case_ids = [getattr(value, "case_id", None) for value in values]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise Lfm2CpcfShortConvV4Error(f"{name} cases require nonempty case ids.")
    if len(set(case_ids)) != len(case_ids):
        raise Lfm2CpcfShortConvV4Error(f"{name} case ids must be unique.")
    return tuple(sorted(values, key=lambda value: value.case_id))


def _state_sha256(values: Mapping[str, Any], name: str) -> str:
    return _sha256(
        {
            key: _tensor_sha256(value, f"{name}.{key}")
            for key, value in sorted(values.items())
            if value is not None
        }
    )


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CpcfShortConvV4Error(f"{name} must be finite and positive.")
    result = float(value)
    minimum_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not minimum_ok:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise Lfm2CpcfShortConvV4Error(f"{name} must be a finite {qualifier} number.")
    return result


def _normalize_split(value: Any) -> str:
    split = str(value).strip().lower()
    if split not in _ALLOWED_SPLITS:
        raise Lfm2CpcfShortConvV4Error(
            "split must be 'fit', 'selection', or 'validation'."
        )
    return split


def _normalize_target_dtype(value: Any) -> str:
    name = str(value).strip().lower().removeprefix("torch.")
    if name not in _ALLOWED_TARGET_DTYPES:
        raise Lfm2CpcfShortConvV4Error(
            "target_dtype must be bfloat16, float16, float32, or float64."
        )
    return name


def _torch_dtype(name: str) -> Any:
    torch = _torch()
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvV4SolverPolicy:
    """Authenticated, finite search and trust policy for the v4 primitives."""

    ridge_alphas: tuple[float, ...] = (1e-8, 1e-6, 1e-4)
    predictor_ridge_alpha: float = 1e-6
    rank_relative_tolerance: float = 1e-10
    participation_relative_tolerance: float = 1e-12
    variable_projection_passes: int = 3
    writer_trust_radius: float = 4.0
    fir_trust_radius: float = 1.0
    bcx_trust_radius: float = 0.25
    correction_rms_floor: float = 1e-6
    context_objective_weight: float = 0.05
    target_dtype: str = "bfloat16"
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ridge_alphas, tuple) or not self.ridge_alphas:
            raise Lfm2CpcfShortConvV4Error("ridge_alphas must be a nonempty tuple.")
        alphas = tuple(
            _positive_float(value, "ridge_alphas", allow_zero=True)
            for value in self.ridge_alphas
        )
        if tuple(sorted(set(alphas))) != alphas:
            raise Lfm2CpcfShortConvV4Error(
                "ridge_alphas must be unique and strictly increasing."
            )
        predictor = _positive_float(
            self.predictor_ridge_alpha,
            "predictor_ridge_alpha",
            allow_zero=True,
        )
        rank_tolerance = _positive_float(
            self.rank_relative_tolerance,
            "rank_relative_tolerance",
        )
        participation_tolerance = _positive_float(
            self.participation_relative_tolerance,
            "participation_relative_tolerance",
        )
        if isinstance(
            self.variable_projection_passes, bool
        ) or self.variable_projection_passes not in {2, 3}:
            raise Lfm2CpcfShortConvV4Error(
                "variable_projection_passes must be exactly 2 or 3."
            )
        writer_radius = _positive_float(
            self.writer_trust_radius,
            "writer_trust_radius",
        )
        fir_radius = _positive_float(
            self.fir_trust_radius,
            "fir_trust_radius",
        )
        bcx_radius = _positive_float(
            self.bcx_trust_radius,
            "bcx_trust_radius",
        )
        rms_floor = _positive_float(
            self.correction_rms_floor,
            "correction_rms_floor",
        )
        context_weight = _positive_float(
            self.context_objective_weight,
            "context_objective_weight",
            allow_zero=True,
        )
        if context_weight > 0.1:
            raise Lfm2CpcfShortConvV4Error(
                "context_objective_weight must be between 0.0 and 0.1."
            )
        target_dtype = _normalize_target_dtype(self.target_dtype)
        object.__setattr__(self, "ridge_alphas", alphas)
        object.__setattr__(self, "predictor_ridge_alpha", predictor)
        object.__setattr__(self, "rank_relative_tolerance", rank_tolerance)
        object.__setattr__(
            self,
            "participation_relative_tolerance",
            participation_tolerance,
        )
        object.__setattr__(self, "writer_trust_radius", writer_radius)
        object.__setattr__(self, "fir_trust_radius", fir_radius)
        object.__setattr__(self, "bcx_trust_radius", bcx_radius)
        object.__setattr__(self, "correction_rms_floor", rms_floor)
        object.__setattr__(self, "context_objective_weight", context_weight)
        object.__setattr__(self, "target_dtype", target_dtype)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV4Error(
                "solver policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "authenticated_solver_policy",
            "ridge_alphas": list(self.ridge_alphas),
            "predictor_ridge_alpha": self.predictor_ridge_alpha,
            "rank_relative_tolerance": self.rank_relative_tolerance,
            "participation_relative_tolerance": (self.participation_relative_tolerance),
            "variable_projection_passes": self.variable_projection_passes,
            "writer_trust_radius": self.writer_trust_radius,
            "fir_trust_radius": self.fir_trust_radius,
            "bcx_trust_radius": self.bcx_trust_radius,
            "correction_rms_floor": self.correction_rms_floor,
            "context_objective_weight": self.context_objective_weight,
            "target_dtype": self.target_dtype,
            "execution": {
                "closed_form_only": True,
                "gradients": 0,
                "optimizer_steps": 0,
                "training_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
            },
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4RidgeCase:
    """One independently weighted matrix case for a closed-form solve."""

    case_id: str
    split: str
    design: Any
    target: Any
    valid_rows: Any | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise Lfm2CpcfShortConvV4Error("case_id must be nonempty text.")
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "split", _normalize_split(self.split))
        object.__setattr__(
            self,
            "design",
            _snapshot_dense_tensor(self.design, f"{self.case_id}.design"),
        )
        object.__setattr__(
            self,
            "target",
            _snapshot_dense_tensor(self.target, f"{self.case_id}.target"),
        )
        object.__setattr__(
            self,
            "valid_rows",
            _snapshot_optional_mask(
                self.valid_rows,
                f"{self.case_id}.valid_rows",
            ),
        )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV4Error(
                "ridge case authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        tensors = {
            "design": _tensor_sha256(self.design, f"{self.case_id}.design"),
            "target": _tensor_sha256(self.target, f"{self.case_id}.target"),
        }
        if self.valid_rows is not None:
            tensors["valid_rows"] = _tensor_sha256(
                self.valid_rows,
                f"{self.case_id}.valid_rows",
            )
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "ridge_case",
            "case_id": self.case_id,
            "split": self.split,
            "tensors": tensors,
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4RidgeResult:
    solution: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )


def affine_penalty_mask(input_width: int, *, fit_bias: bool) -> Any:
    """Return a ridge mask whose affine intercept is never penalized."""

    torch = _torch()
    if (
        isinstance(input_width, bool)
        or not isinstance(input_width, int)
        or input_width <= 0
    ):
        raise Lfm2CpcfShortConvV4Error("input_width must be a positive integer.")
    values = torch.ones(
        (input_width + int(fit_bias),),
        dtype=torch.float64,
    )
    if fit_bias:
        values[-1] = 0.0
    return values


def fir_penalty_mask(fir_length: int, *, fit_bias: bool) -> Any:
    """Return a ridge mask whose depthwise FIR bias is unpenalized."""

    return affine_penalty_mask(fir_length, fit_bias=fit_bias)


def _normalize_ridge_case(
    case: Lfm2CpcfV4RidgeCase,
    *,
    columns: int | None,
    outputs: int | None,
    required_split: str,
) -> tuple[Any, Any]:
    torch = _torch()
    if not isinstance(case, Lfm2CpcfV4RidgeCase):
        raise Lfm2CpcfShortConvV4Error(
            "ridge cases must be Lfm2CpcfV4RidgeCase instances."
        )
    if not case.verify_authentication():
        raise Lfm2CpcfShortConvV4Error(
            f"ridge case {case.case_id!r} changed after authentication."
        )
    if case.split != required_split:
        raise Lfm2CpcfShortConvV4Error(
            f"ridge solve requires {required_split!r} cases only; received "
            f"{case.split!r}."
        )
    design = _finite_tensor(
        case.design,
        f"{case.case_id}.design",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    target = _finite_tensor(
        case.target,
        f"{case.case_id}.target",
        minimum_dimensions=1,
        maximum_dimensions=2,
    )
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    if int(design.shape[0]) != int(target.shape[0]):
        raise Lfm2CpcfShortConvV4Error(
            f"{case.case_id} design and target row counts differ."
        )
    if columns is not None and int(design.shape[1]) != columns:
        raise Lfm2CpcfShortConvV4Error("ridge cases have inconsistent design widths.")
    if outputs is not None and int(target.shape[1]) != outputs:
        raise Lfm2CpcfShortConvV4Error("ridge cases have inconsistent target widths.")
    if case.valid_rows is not None:
        mask = case.valid_rows
        if (
            not isinstance(mask, torch.Tensor)
            or mask.layout != torch.strided
            or mask.device.type == "meta"
            or mask.ndim != 1
            or tuple(mask.shape) != (int(design.shape[0]),)
        ):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id}.valid_rows must be a dense rank-1 row mask."
            )
        mask = mask.detach().to(device="cpu", dtype=torch.bool)
        design = design[mask]
        target = target[mask]
    if int(design.shape[0]) == 0:
        raise Lfm2CpcfShortConvV4Error(f"{case.case_id} contains no valid solve rows.")
    return design.contiguous(), target.contiguous()


def _effective_rank(values: Any, relative_tolerance: float) -> int:
    if int(values.numel()) == 0:
        return 0
    maximum = float(values.max().item())
    if maximum <= 0.0:
        return 0
    return int((values > maximum * relative_tolerance).sum().item())


def case_balanced_trace_scaled_prior_ridge(
    cases: Sequence[Lfm2CpcfV4RidgeCase],
    *,
    prior: Any,
    alpha: float,
    penalty_mask: Any,
    required_split: str = "fit",
    rank_relative_tolerance: float = 1e-10,
) -> Lfm2CpcfV4RidgeResult:
    """Solve an equal-case-weighted, trace-scaled ridge around ``prior``.

    Each case contributes its *mean* Gram and cross-product.  Repeating rows
    inside one trajectory therefore cannot increase that trajectory's weight.
    The effective ridge is ``alpha * trace(G) / penalized_columns`` and scales
    with a global rescaling of the design.  A zero penalty-mask entry excludes
    that coefficient from both shrinkage and the prior, which is how affine
    and FIR biases remain truly unpenalized.
    """

    torch = _torch()
    ordered_cases = _canonical_unique_cases(cases, "case-balanced ridge")
    split = _normalize_split(required_split)
    if split != "fit":
        raise Lfm2CpcfShortConvV4Error(
            "closed-form coefficient solves accept fit cases only; selection "
            "may rank candidates and validation may evaluate the winner."
        )
    alpha_value = _positive_float(alpha, "alpha", allow_zero=True)
    tolerance = _positive_float(
        rank_relative_tolerance,
        "rank_relative_tolerance",
    )
    columns: int | None = None
    outputs: int | None = None
    gram = None
    right = None
    case_receipts = []
    for case in ordered_cases:
        design, target = _normalize_ridge_case(
            case,
            columns=columns,
            outputs=outputs,
            required_split=split,
        )
        columns = int(design.shape[1])
        outputs = int(target.shape[1])
        row_count = int(design.shape[0])
        case_gram = (design.T @ design) / row_count
        case_right = (design.T @ target) / row_count
        gram = case_gram if gram is None else gram + case_gram
        right = case_right if right is None else right + case_right
        case_receipts.append(
            {
                "case_id": case.case_id,
                "authentication_sha256": case.authentication_sha256,
                "rows": row_count,
                "design_sha256": _tensor_sha256(
                    design,
                    f"{case.case_id}.normalized_design",
                ),
                "target_sha256": _tensor_sha256(
                    target,
                    f"{case.case_id}.normalized_target",
                ),
            }
        )
    assert columns is not None and outputs is not None
    prior_value = _finite_tensor(
        prior,
        "prior",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if tuple(prior_value.shape) != (columns, outputs):
        raise Lfm2CpcfShortConvV4Error(
            "prior must have shape [design_columns, target_outputs]."
        )
    mask = _finite_tensor(
        penalty_mask,
        "penalty_mask",
        minimum_dimensions=1,
        maximum_dimensions=1,
    )
    if tuple(mask.shape) != (columns,) or bool((mask < 0.0).any().item()):
        raise Lfm2CpcfShortConvV4Error(
            "penalty_mask must be a nonnegative vector over design columns."
        )
    assert gram is not None and right is not None
    gram = gram / len(ordered_cases)
    right = right / len(ordered_cases)
    penalized_count = int((mask > 0.0).sum().item())
    trace_denominator = max(penalized_count, 1)
    total_trace = float(torch.trace(gram).item())
    trace = float((torch.diag(gram) * (mask > 0.0).to(torch.float64)).sum().item())
    effective_ridge = alpha_value * trace / trace_denominator
    penalty = torch.diag(mask.square())
    system = gram + effective_ridge * penalty
    shifted_right = right + effective_ridge * penalty @ prior_value
    singular_values = torch.linalg.svdvals(gram)
    system_singular_values = torch.linalg.svdvals(system)
    try:
        solution = torch.linalg.solve(system, shifted_right)
        solve_method = "symmetric_solve"
    except RuntimeError:
        solution = (
            torch.linalg.pinv(
                system,
                rtol=tolerance,
                hermitian=True,
            )
            @ shifted_right
        )
        solve_method = "hermitian_pseudoinverse"
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2CpcfShortConvV4Error("case-balanced ridge produced NaN or infinity.")
    maximum_system = float(system_singular_values.max().item())
    positive_system = system_singular_values[
        system_singular_values > maximum_system * tolerance
    ]
    condition = (
        float(maximum_system / positive_system.min().item())
        if int(positive_system.numel()) > 0 and maximum_system > 0.0
        else 0.0
    )
    correction = solution - prior_value
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "case_balanced_trace_scaled_prior_ridge",
        "required_split": split,
        "case_balanced": True,
        "row_duplication_invariant_objective": True,
        "case_count": len(ordered_cases),
        "case_receipts": case_receipts,
        "design_columns": columns,
        "target_outputs": outputs,
        "alpha": alpha_value,
        "gram_trace": trace,
        "total_gram_trace": total_trace,
        "trace_uses_penalized_columns_only": True,
        "trace_denominator": trace_denominator,
        "effective_ridge": effective_ridge,
        "penalty_mask": [float(value) for value in mask.tolist()],
        "unpenalized_columns": [
            index for index, value in enumerate(mask.tolist()) if float(value) == 0.0
        ],
        "gram_singular_values": [float(value) for value in singular_values.tolist()],
        "gram_effective_rank": _effective_rank(
            singular_values,
            tolerance,
        ),
        "regularized_system_condition": condition,
        "solve_method": solve_method,
        "prior_rms": float(prior_value.square().mean().sqrt().item()),
        "solution_rms": float(solution.square().mean().sqrt().item()),
        "correction_rms": float(correction.square().mean().sqrt().item()),
        "prior_sha256": _tensor_sha256(prior_value, "ridge.prior"),
        "solution_sha256": _tensor_sha256(solution, "ridge.solution"),
        "execution": {
            "closed_form_linear_solve": True,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    return Lfm2CpcfV4RidgeResult(
        solution=solution.detach().contiguous(),
        diagnostics=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4BasisCase:
    """Fit-only evidence for predictable and endpoint-observable channels."""

    case_id: str
    split: str
    child_inputs: Any
    parent_contexts: Any
    endpoint_writes: Any
    valid_rows: Any | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise Lfm2CpcfShortConvV4Error("case_id must be nonempty text.")
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "split", _normalize_split(self.split))
        object.__setattr__(
            self,
            "child_inputs",
            _snapshot_dense_tensor(
                self.child_inputs,
                f"{self.case_id}.child_inputs",
            ),
        )
        object.__setattr__(
            self,
            "parent_contexts",
            _snapshot_dense_tensor(
                self.parent_contexts,
                f"{self.case_id}.parent_contexts",
            ),
        )
        object.__setattr__(
            self,
            "endpoint_writes",
            _snapshot_dense_tensor(
                self.endpoint_writes,
                f"{self.case_id}.endpoint_writes",
            ),
        )
        object.__setattr__(
            self,
            "valid_rows",
            _snapshot_optional_mask(
                self.valid_rows,
                f"{self.case_id}.valid_rows",
            ),
        )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV4Error(
                "basis case authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        tensors = {
            "child_inputs": _tensor_sha256(
                self.child_inputs,
                f"{self.case_id}.child_inputs",
            ),
            "parent_contexts": _tensor_sha256(
                self.parent_contexts,
                f"{self.case_id}.parent_contexts",
            ),
            "endpoint_writes": _tensor_sha256(
                self.endpoint_writes,
                f"{self.case_id}.endpoint_writes",
            ),
        }
        if self.valid_rows is not None:
            tensors["valid_rows"] = _tensor_sha256(
                self.valid_rows,
                f"{self.case_id}.valid_rows",
            )
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "endpoint_aware_basis_case",
            "case_id": self.case_id,
            "split": self.split,
            "tensors": tensors,
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4ChannelBasis:
    """Signed source-to-target encoder and its exact Euclidean dual."""

    predictor_weight: Any
    predictor_bias: Any
    encoder: Any
    decoder: Any
    endpoint_right_basis: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def source_hidden_size(self) -> int:
        return int(self.encoder.shape[0])

    @property
    def target_hidden_size(self) -> int:
        return int(self.encoder.shape[1])

    def verify_authentication(self) -> bool:
        if not _verify_sealed_report(self.report):
            return False
        state = {
            "predictor_weight": self.predictor_weight,
            "predictor_bias": self.predictor_bias,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "endpoint_right_basis": self.endpoint_right_basis,
        }
        return self.report.get("state_sha256") == _state_sha256(
            state,
            "basis",
        )


def _basis_rows(case: Lfm2CpcfV4BasisCase) -> tuple[Any, Any, Any]:
    torch = _torch()
    if not isinstance(case, Lfm2CpcfV4BasisCase):
        raise Lfm2CpcfShortConvV4Error(
            "basis cases must be Lfm2CpcfV4BasisCase instances."
        )
    if case.split != "fit":
        raise Lfm2CpcfShortConvV4Error(
            "endpoint-aware channel basis accepts fit cases only."
        )
    if not case.verify_authentication():
        raise Lfm2CpcfShortConvV4Error(
            f"basis case {case.case_id!r} changed after authentication."
        )
    child = _finite_tensor(
        case.child_inputs,
        f"{case.case_id}.child_inputs",
        minimum_dimensions=2,
    )
    parent = _finite_tensor(
        case.parent_contexts,
        f"{case.case_id}.parent_contexts",
        minimum_dimensions=2,
    )
    endpoint = _finite_tensor(
        case.endpoint_writes,
        f"{case.case_id}.endpoint_writes",
        minimum_dimensions=2,
    )
    prefix = tuple(int(size) for size in child.shape[:-1])
    if (
        tuple(int(size) for size in parent.shape[:-1]) != prefix
        or tuple(int(size) for size in endpoint.shape[:-1]) != prefix
    ):
        raise Lfm2CpcfShortConvV4Error(
            f"{case.case_id} basis tensors have different row geometry."
        )
    child = child.reshape(-1, int(child.shape[-1]))
    parent = parent.reshape(-1, int(parent.shape[-1]))
    endpoint = endpoint.reshape(-1, int(endpoint.shape[-1]))
    if case.valid_rows is not None:
        mask = case.valid_rows
        if (
            not isinstance(mask, torch.Tensor)
            or mask.layout != torch.strided
            or mask.device.type == "meta"
            or tuple(int(size) for size in mask.shape) != prefix
        ):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id}.valid_rows has the wrong geometry."
            )
        mask = mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
        child = child[mask]
        parent = parent[mask]
        endpoint = endpoint[mask]
    if int(child.shape[0]) == 0:
        raise Lfm2CpcfShortConvV4Error(f"{case.case_id} contains no valid basis rows.")
    return child.contiguous(), parent.contiguous(), endpoint.contiguous()


def _case_balanced_covariance(
    pairs: Sequence[tuple[Any, Any]],
    *,
    center_each_case: bool,
) -> Any:
    torch = _torch()
    result = None
    for left, right in pairs:
        if center_each_case:
            left = left - left.mean(dim=0, keepdim=True)
            right = right - right.mean(dim=0, keepdim=True)
        value = (left.T @ right) / int(left.shape[0])
        result = value if result is None else result + value
    assert result is not None
    return (result / len(pairs)).to(torch.float64).contiguous()


def _canonicalize_svd_pair(left: Any, right_t: Any) -> tuple[Any, Any]:
    left = left.clone()
    right_t = right_t.clone()
    for column in range(int(left.shape[1])):
        vector = left[:, column]
        pivot = int(vector.abs().argmax().item())
        if float(vector[pivot].item()) < 0.0:
            left[:, column].neg_()
            right_t[column, :].neg_()
    return left.contiguous(), right_t.contiguous()


def _canonicalize_vector(value: Any) -> Any:
    vector = value.clone()
    pivot = int(vector.abs().argmax().item())
    if float(vector[pivot].item()) < 0.0:
        vector.neg_()
    return vector.contiguous()


def _append_orthonormal(
    columns: list[Any],
    candidate: Any,
    *,
    tolerance: float,
) -> bool:
    torch = _torch()
    value = candidate.clone()
    for _ in range(2):
        for existing in columns:
            value = value - existing * torch.dot(existing, value)
    norm = torch.linalg.vector_norm(value)
    if float(norm.item()) <= tolerance:
        return False
    columns.append(_canonicalize_vector(value / norm))
    return True


def build_signed_endpoint_aware_channel_basis(
    cases: Sequence[Lfm2CpcfV4BasisCase],
    *,
    target_channels: int,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> Lfm2CpcfV4ChannelBasis:
    """Build a fit-only predictable/observable signed channel basis.

    Parent contexts are first projected onto the span predictable from actual
    child inputs.  The PLS cross-covariance between those predictable contexts
    and residual endpoint writes then supplies signed behavior directions.
    Rank-deficient endpoint directions are completed deterministically using
    predictable-context coordinate residuals and finally coordinate axes.
    """

    torch = _torch()
    if not isinstance(policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    if not policy.verify_authentication():
        raise Lfm2CpcfShortConvV4Error("solver policy failed authentication.")
    if (
        isinstance(target_channels, bool)
        or not isinstance(target_channels, int)
        or target_channels <= 0
    ):
        raise Lfm2CpcfShortConvV4Error("target_channels must be a positive integer.")
    ordered_cases = _canonical_unique_cases(cases, "endpoint-aware basis")
    rows = [_basis_rows(case) for case in ordered_cases]
    child_width = int(rows[0][0].shape[1])
    source_width = int(rows[0][1].shape[1])
    endpoint_width = int(rows[0][2].shape[1])
    if target_channels > source_width:
        raise Lfm2CpcfShortConvV4Error(
            "target_channels cannot exceed source context width."
        )
    for child, parent, endpoint in rows:
        if (
            int(child.shape[1]) != child_width
            or int(parent.shape[1]) != source_width
            or int(endpoint.shape[1]) != endpoint_width
        ):
            raise Lfm2CpcfShortConvV4Error(
                "basis cases have inconsistent feature widths."
            )
    predictor_cases = []
    for case, (child, parent, _endpoint) in zip(ordered_cases, rows):
        design = torch.cat(
            (
                child,
                torch.ones(
                    (int(child.shape[0]), 1),
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )
        predictor_cases.append(
            Lfm2CpcfV4RidgeCase(
                case_id=case.case_id,
                split="fit",
                design=design,
                target=parent,
            )
        )
    predictor = case_balanced_trace_scaled_prior_ridge(
        predictor_cases,
        prior=torch.zeros(
            (child_width + 1, source_width),
            dtype=torch.float64,
        ),
        alpha=policy.predictor_ridge_alpha,
        penalty_mask=affine_penalty_mask(child_width, fit_bias=True),
        required_split="fit",
        rank_relative_tolerance=policy.rank_relative_tolerance,
    )
    predictor_weight = predictor.solution[:-1, :].contiguous()
    predictor_bias = predictor.solution[-1, :].contiguous()
    predicted_rows = [
        child @ predictor_weight + predictor_bias for child, _parent, _endpoint in rows
    ]
    predictability_covariance = _case_balanced_covariance(
        [(value, value) for value in predicted_rows],
        center_each_case=True,
    )
    raw_parent_covariance = _case_balanced_covariance(
        [(parent, parent) for _child, parent, _endpoint in rows],
        center_each_case=True,
    )
    cross_covariance = _case_balanced_covariance(
        [
            (predicted, endpoint)
            for predicted, (_child, _parent, endpoint) in zip(
                predicted_rows,
                rows,
            )
        ],
        center_each_case=True,
    )
    observability_gram = cross_covariance @ cross_covariance.T
    left, balanced_values, right_t = torch.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )
    left, right_t = _canonicalize_svd_pair(left, right_t)
    maximum_balanced = (
        float(balanced_values.max().item()) if int(balanced_values.numel()) > 0 else 0.0
    )
    balanced_rank = (
        int(
            (balanced_values > maximum_balanced * policy.rank_relative_tolerance)
            .sum()
            .item()
        )
        if maximum_balanced > 0.0
        else 0
    )
    columns: list[Any] = []
    endpoint_columns = min(balanced_rank, target_channels)
    orthogonal_tolerance = max(
        policy.rank_relative_tolerance,
        torch.finfo(torch.float64).eps * source_width,
    )
    for index in range(endpoint_columns):
        _append_orthonormal(
            columns,
            left[:, index],
            tolerance=orthogonal_tolerance,
        )
    completion_sources: list[dict[str, Any]] = []
    predictable_diagonal = torch.diag(predictability_covariance)
    coordinate_order = sorted(
        range(source_width),
        key=lambda index: (
            -float(predictable_diagonal[index].item()),
            index,
        ),
    )
    identity = torch.eye(source_width, dtype=torch.float64)
    for source in coordinate_order:
        if len(columns) >= target_channels:
            break
        if _append_orthonormal(
            columns,
            identity[:, source],
            tolerance=orthogonal_tolerance,
        ):
            completion_sources.append(
                {
                    "source_index": source,
                    "predictable_variance": float(predictable_diagonal[source].item()),
                    "kind": "predictability_weighted_coordinate_residual",
                }
            )
    if len(columns) != target_channels:
        raise Lfm2CpcfShortConvV4Error(
            "deterministic rank completion could not fill target channels."
        )
    encoder = torch.stack(columns, dim=1).contiguous()
    decoder = encoder.T.contiguous()
    dual_error = torch.max(
        torch.abs(decoder @ encoder - torch.eye(target_channels, dtype=torch.float64))
    )
    predictability_spectrum = torch.linalg.eigvalsh(predictability_covariance).flip(0)
    raw_parent_spectrum = torch.linalg.eigvalsh(raw_parent_covariance).flip(0)
    observability_spectrum = torch.linalg.eigvalsh(observability_gram).flip(0)
    source_input_rms = torch.sqrt(torch.diag(raw_parent_covariance).clamp(min=0))
    source_predictable_rms = torch.sqrt(
        torch.diag(predictability_covariance).clamp(min=0)
    )
    source_observability = torch.sqrt(torch.diag(observability_gram).clamp(min=0))
    encoder_energy = encoder.square().sum(dim=1).sqrt()
    decoder_energy = decoder.square().sum(dim=0).sqrt()
    maximum_encoder_energy = max(
        float(encoder_energy.max().item()),
        torch.finfo(torch.float64).eps,
    )
    numeric_threshold = maximum_encoder_energy * policy.participation_relative_tolerance
    numeric_participants = [
        index
        for index, value in enumerate(encoder_energy.tolist())
        if float(value) > numeric_threshold
    ]
    source_participation = [
        {
            "source_index": index,
            "raw_context_rms": float(source_input_rms[index].item()),
            "predictable_context_rms": float(source_predictable_rms[index].item()),
            "endpoint_observability_rms": float(source_observability[index].item()),
            "encoder_row_norm": float(encoder_energy[index].item()),
            "decoder_column_norm": float(decoder_energy[index].item()),
            "numeric_basis_participation": index in numeric_participants,
        }
        for index in range(source_width)
    ]
    prediction_errors = []
    raw_energies = []
    for predicted, (_child, parent, _endpoint) in zip(predicted_rows, rows):
        prediction_errors.append(float((predicted - parent).square().mean().item()))
        raw_energies.append(float(parent.square().mean().item()))
    predictability_fraction = 1.0 - (
        sum(prediction_errors) / len(prediction_errors)
    ) / max(
        sum(raw_energies) / len(raw_energies),
        torch.finfo(torch.float64).eps,
    )
    endpoint_right_basis = right_t[:endpoint_columns, :].T.contiguous()
    state = {
        "predictor_weight": predictor_weight,
        "predictor_bias": predictor_bias,
        "encoder": encoder,
        "decoder": decoder,
        "endpoint_right_basis": endpoint_right_basis,
    }
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "signed_predictability_observability_channel_basis",
        "policy_sha256": policy.authentication_sha256,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "fit_case_authentication_sha256": [
            case.authentication_sha256 for case in ordered_cases
        ],
        "fit_case_count": len(ordered_cases),
        "child_input_width": child_width,
        "source_hidden_size": source_width,
        "target_hidden_size": target_channels,
        "hidden_width_compression_ratio": source_width / target_channels,
        "endpoint_width": endpoint_width,
        "predictor": dict(predictor.diagnostics),
        "predictability_fraction": predictability_fraction,
        "predictability_controllability_spectrum": [
            float(value) for value in predictability_spectrum.tolist()
        ],
        "raw_parent_context_spectrum": [
            float(value) for value in raw_parent_spectrum.tolist()
        ],
        "endpoint_observability_spectrum": [
            float(value) for value in observability_spectrum.tolist()
        ],
        "balanced_endpoint_cross_covariance_singular_values": [
            float(value) for value in balanced_values.tolist()
        ],
        "balanced_endpoint_effective_rank": balanced_rank,
        "endpoint_basis_columns": endpoint_columns,
        "rank_completion_columns": completion_sources,
        "canonical_svd_signs": True,
        "degenerate_svd_subspace_rotation_canonicalized": False,
        "dual_decoder": "euclidean_left_inverse_of_orthonormal_encoder",
        "maximum_dual_identity_error": float(dual_error.item()),
        "all_source_channels_consumed": list(range(source_width)),
        "source_channels_selected": len(numeric_participants),
        "source_channels_dropped": source_width - len(numeric_participants),
        "source_channels_pruned": 0,
        "full_source_coverage": len(numeric_participants) == source_width,
        "source_participation": source_participation,
        "numeric_participating_source_indices": numeric_participants,
        "full_source_numeric_participation": (
            len(numeric_participants) == source_width
        ),
        "state_sha256": _sha256(
            {
                name: _tensor_sha256(value, f"basis.{name}")
                for name, value in state.items()
            }
        ),
        "execution": {
            "fit_only_predictor": True,
            "closed_form_linear_solve": True,
            "deterministic_svd": False,
            "determinism_claim": (
                "canonical signs and case order only; repeated singular-value "
                "subspace rotations are not claimed cross-backend deterministic"
            ),
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    return Lfm2CpcfV4ChannelBasis(
        predictor_weight=predictor_weight.detach(),
        predictor_bias=predictor_bias.detach(),
        encoder=encoder.detach(),
        decoder=decoder.detach(),
        endpoint_right_basis=endpoint_right_basis.detach(),
        report=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4TransportedPriors:
    in_proj_weight: Any
    in_proj_bias: Any | None
    fir_kernel: Any
    fir_bias: Any | None
    out_proj_weight: Any
    out_proj_bias: Any | None
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        if not _verify_sealed_report(self.report):
            return False
        state = {
            "in_proj_weight": self.in_proj_weight,
            "in_proj_bias": self.in_proj_bias,
            "fir_kernel": self.fir_kernel,
            "fir_bias": self.fir_bias,
            "out_proj_weight": self.out_proj_weight,
            "out_proj_bias": self.out_proj_bias,
        }
        return self.report.get("state_sha256") == _state_sha256(
            state,
            "transport",
        )


def _resize_fir_rows(kernel: Any, target_length: int) -> Any:
    torch = _torch()
    source_length = int(kernel.shape[1])
    if target_length == source_length:
        return kernel.clone().contiguous()
    if target_length == 1:
        return kernel.mean(dim=1, keepdim=True)
    if source_length == 1:
        return kernel.expand(-1, target_length).clone().contiguous()
    return (
        torch.nn.functional.interpolate(
            kernel.unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=True,
        )
        .squeeze(0)
        .contiguous()
    )


def transport_parent_short_conv_priors(
    *,
    basis: Lfm2CpcfV4ChannelBasis,
    source_in_proj_weight: Any,
    source_fir_kernel: Any,
    source_out_proj_weight: Any,
    target_fir_length: int,
    source_in_proj_bias: Any | None = None,
    source_fir_bias: Any | None = None,
    source_out_proj_bias: Any | None = None,
    input_lift: Any | None = None,
    input_lift_bias: Any | None = None,
    output_encoder: Any | None = None,
) -> Lfm2CpcfV4TransportedPriors:
    """Transport actual parent B/C/x, FIR, and writer tensors into priors."""

    torch = _torch()
    if not isinstance(basis, Lfm2CpcfV4ChannelBasis):
        raise TypeError("basis must be Lfm2CpcfV4ChannelBasis.")
    if not basis.verify_authentication():
        raise Lfm2CpcfShortConvV4Error("channel basis changed after authentication.")
    source_hidden = basis.source_hidden_size
    target_hidden = basis.target_hidden_size
    if (
        isinstance(target_fir_length, bool)
        or not isinstance(target_fir_length, int)
        or target_fir_length <= 0
    ):
        raise Lfm2CpcfShortConvV4Error("target_fir_length must be a positive integer.")
    source_in = _finite_tensor(
        source_in_proj_weight,
        "source_in_proj_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    source_out = _finite_tensor(
        source_out_proj_weight,
        "source_out_proj_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    source_kernel = _finite_tensor(
        source_fir_kernel,
        "source_fir_kernel",
        minimum_dimensions=2,
        maximum_dimensions=3,
    )
    if source_kernel.ndim == 3:
        if int(source_kernel.shape[1]) != 1:
            raise Lfm2CpcfShortConvV4Error(
                "source_fir_kernel rank-3 input must be depthwise [S,1,L]."
            )
        source_kernel = source_kernel[:, 0, :]
    source_input_width = int(source_in.shape[1])
    source_output_width = int(source_out.shape[0])
    if (
        tuple(source_in.shape[:1]) != (3 * source_hidden,)
        or int(source_out.shape[1]) != source_hidden
        or int(source_kernel.shape[0]) != source_hidden
    ):
        raise Lfm2CpcfShortConvV4Error(
            "parent operator tensors disagree with basis source width."
        )
    if input_lift is None:
        if tuple(basis.predictor_weight.shape) == (
            target_hidden,
            source_input_width,
        ):
            input_lift_value = basis.predictor_weight.to(torch.float64).clone()
            default_input_lift = "fit_predictor"
        elif source_input_width == source_hidden:
            input_lift_value = basis.decoder.to(torch.float64).clone()
            default_input_lift = "euclidean_basis_decoder"
        else:
            raise Lfm2CpcfShortConvV4Error(
                "input_lift is required when parent input width differs "
                "from source channel width."
            )
    else:
        input_lift_value = _finite_tensor(
            input_lift,
            "input_lift",
            minimum_dimensions=2,
            maximum_dimensions=2,
        )
        if tuple(input_lift_value.shape) != (
            target_hidden,
            source_input_width,
        ):
            raise Lfm2CpcfShortConvV4Error(
                "input_lift must map target residual rows to parent input rows."
            )
        default_input_lift = "explicit"
    if input_lift_bias is None:
        if default_input_lift == "fit_predictor" and tuple(
            basis.predictor_bias.shape
        ) == (source_input_width,):
            input_lift_bias_value = basis.predictor_bias.to(torch.float64).clone()
            default_input_lift_bias = "fit_predictor"
        else:
            input_lift_bias_value = torch.zeros(
                (source_input_width,),
                dtype=torch.float64,
            )
            default_input_lift_bias = "zero"
    else:
        input_lift_bias_value = _finite_tensor(
            input_lift_bias,
            "input_lift_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(input_lift_bias_value.shape) != (source_input_width,):
            raise Lfm2CpcfShortConvV4Error(
                "input_lift_bias must be a parent-input-width vector."
            )
        default_input_lift_bias = "explicit"
    if output_encoder is None:
        if source_output_width != source_hidden:
            raise Lfm2CpcfShortConvV4Error(
                "output_encoder is required when parent output width differs "
                "from source channel width."
            )
        output_encoder_value = basis.encoder
    else:
        output_encoder_value = _finite_tensor(
            output_encoder,
            "output_encoder",
            minimum_dimensions=2,
            maximum_dimensions=2,
        )
        if tuple(output_encoder_value.shape) != (
            source_output_width,
            target_hidden,
        ):
            raise Lfm2CpcfShortConvV4Error(
                "output_encoder must map parent output rows to target rows."
            )
    encoder = basis.encoder.to(torch.float64)
    decoder = basis.decoder.to(torch.float64)
    role_weights = source_in.reshape(
        3,
        source_hidden,
        source_input_width,
    )
    transported_roles = [encoder.T @ role @ input_lift_value.T for role in role_weights]
    in_weight = torch.cat(transported_roles, dim=0).contiguous()
    source_bias_roles = torch.zeros(
        (3, source_hidden),
        dtype=torch.float64,
    )
    if source_in_proj_bias is not None:
        source_bias = _finite_tensor(
            source_in_proj_bias,
            "source_in_proj_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(source_bias.shape) != (3 * source_hidden,):
            raise Lfm2CpcfShortConvV4Error("source_in_proj_bias has the wrong width.")
        source_bias_roles = source_bias.reshape(3, source_hidden)
    transported_bias_roles = [
        encoder.T @ (source_role_bias + role @ input_lift_bias_value)
        for role, source_role_bias in zip(role_weights, source_bias_roles)
    ]
    in_bias_candidate = torch.cat(transported_bias_roles).contiguous()
    in_bias = (
        in_bias_candidate
        if source_in_proj_bias is not None
        or bool((input_lift_bias_value != 0.0).any().item())
        else None
    )
    resized_kernel = _resize_fir_rows(source_kernel, target_fir_length)
    diagonal_transport = decoder.T * encoder
    fir_kernel = (diagonal_transport.T @ resized_kernel).contiguous()
    noncommutation = []
    for lag in range(target_fir_length):
        projected = decoder @ (resized_kernel[:, lag].reshape(-1, 1) * encoder)
        diagonal = torch.diag(torch.diag(projected))
        denominator = float(
            torch.linalg.matrix_norm(projected)
            .clamp_min(torch.finfo(torch.float64).eps)
            .item()
        )
        noncommutation.append(
            {
                "lag": lag,
                "projected_operator_frobenius": denominator,
                "off_diagonal_frobenius": float(
                    torch.linalg.matrix_norm(projected - diagonal).item()
                ),
                "off_diagonal_relative_frobenius": float(
                    torch.linalg.matrix_norm(projected - diagonal).item() / denominator
                ),
            }
        )
    fir_bias = None
    if source_fir_bias is not None:
        source_conv_bias = _finite_tensor(
            source_fir_bias,
            "source_fir_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(source_conv_bias.shape) != (source_hidden,):
            raise Lfm2CpcfShortConvV4Error("source_fir_bias has the wrong width.")
        fir_bias = (diagonal_transport.T @ source_conv_bias).contiguous()
    out_weight = (output_encoder_value.T @ source_out @ decoder.T).contiguous()
    out_bias = None
    if source_out_proj_bias is not None:
        source_writer_bias = _finite_tensor(
            source_out_proj_bias,
            "source_out_proj_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(source_writer_bias.shape) != (source_output_width,):
            raise Lfm2CpcfShortConvV4Error("source_out_proj_bias has the wrong width.")
        out_bias = (output_encoder_value.T @ source_writer_bias).contiguous()
    state = {
        "in_proj_weight": in_weight,
        "fir_kernel": fir_kernel,
        "out_proj_weight": out_weight,
    }
    if in_bias is not None:
        state["in_proj_bias"] = in_bias
    if fir_bias is not None:
        state["fir_bias"] = fir_bias
    if out_bias is not None:
        state["out_proj_bias"] = out_bias
    participation = diagonal_transport.abs().sum(dim=1)
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "transported_parent_short_conv_priors",
        "basis_state_sha256": basis.report["state_sha256"],
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "source_input_width": source_input_width,
        "source_output_width": source_output_width,
        "source_fir_length": int(source_kernel.shape[1]),
        "target_fir_length": target_fir_length,
        "transport": {
            "input_roles": "encoder_transpose_parent_role_input_lift",
            "input_bias": (
                "encoder_transpose_parent_bias_plus_parent_role_input_lift_bias"
            ),
            "fir": "dual_diagonal_source_fir_projection",
            "output": "output_encoder_transpose_parent_writer_decoder_transpose",
        },
        "input_lift_source": default_input_lift,
        "input_lift_bias_source": default_input_lift_bias,
        "input_lift_sha256": _tensor_sha256(
            input_lift_value,
            "transport.input_lift",
        ),
        "input_lift_bias_sha256": _tensor_sha256(
            input_lift_bias_value,
            "transport.input_lift_bias",
        ),
        "output_encoder_sha256": _tensor_sha256(
            output_encoder_value,
            "transport.output_encoder",
        ),
        "fir_diagonal_projection_is_linear_surrogate_for_cubic_operator": True,
        "fir_projected_operator_noncommutation": noncommutation,
        "maximum_fir_off_diagonal_relative_frobenius": max(
            value["off_diagonal_relative_frobenius"] for value in noncommutation
        ),
        "all_parent_tensors_consumed": True,
        "source_fir_transport_participation": [
            float(value) for value in participation.tolist()
        ],
        "source_channels_dropped": basis.report["source_channels_dropped"],
        "source_channels_pruned": 0,
        "source_channels_quantized": 0,
        "state_sha256": _sha256(
            {
                name: _tensor_sha256(value, f"transport.{name}")
                for name, value in sorted(state.items())
            }
        ),
        "execution": {
            "closed_form_transport": True,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    return Lfm2CpcfV4TransportedPriors(
        in_proj_weight=in_weight.detach(),
        in_proj_bias=None if in_bias is None else in_bias.detach(),
        fir_kernel=fir_kernel.detach(),
        fir_bias=None if fir_bias is None else fir_bias.detach(),
        out_proj_weight=out_weight.detach(),
        out_proj_bias=None if out_bias is None else out_bias.detach(),
        report=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4ProjectionCase:
    """One fit-only case for bounded variable-projection primitives."""

    case_id: str
    split: str
    child_input_history: Any
    desired_context_history: Any
    desired_write_suffix: Any
    suffix_start: int
    valid_prefix_mask: Any | None = None
    valid_suffix_mask: Any | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise Lfm2CpcfShortConvV4Error("case_id must be nonempty text.")
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "split", _normalize_split(self.split))
        if (
            isinstance(self.suffix_start, bool)
            or not isinstance(self.suffix_start, int)
            or self.suffix_start < 0
        ):
            raise Lfm2CpcfShortConvV4Error(
                "suffix_start must be a nonnegative integer."
            )
        for attribute in (
            "child_input_history",
            "desired_context_history",
            "desired_write_suffix",
        ):
            object.__setattr__(
                self,
                attribute,
                _snapshot_dense_tensor(
                    getattr(self, attribute),
                    f"{self.case_id}.{attribute}",
                ),
            )
        for attribute in ("valid_prefix_mask", "valid_suffix_mask"):
            object.__setattr__(
                self,
                attribute,
                _snapshot_optional_mask(
                    getattr(self, attribute),
                    f"{self.case_id}.{attribute}",
                ),
            )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV4Error(
                "projection case authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        tensors = {
            "child_input_history": _tensor_sha256(
                self.child_input_history,
                f"{self.case_id}.child_input_history",
            ),
            "desired_context_history": _tensor_sha256(
                self.desired_context_history,
                f"{self.case_id}.desired_context_history",
            ),
            "desired_write_suffix": _tensor_sha256(
                self.desired_write_suffix,
                f"{self.case_id}.desired_write_suffix",
            ),
        }
        for name in ("valid_prefix_mask", "valid_suffix_mask"):
            value = getattr(self, name)
            if value is not None:
                tensors[name] = _tensor_sha256(value, f"{self.case_id}.{name}")
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "projection_case",
            "case_id": self.case_id,
            "split": self.split,
            "suffix_start": self.suffix_start,
            "tensors": tensors,
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4BoundedStep:
    value: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )


def bound_prior_correction(
    candidate: Any,
    prior: Any,
    *,
    relative_radius: float,
    rms_floor: float,
    reference_rms: float | None = None,
) -> Lfm2CpcfV4BoundedStep:
    """Clip a closed-form correction to an authenticated RMS trust region."""

    candidate_value = _finite_tensor(
        candidate,
        "candidate",
        minimum_dimensions=1,
    )
    prior_value = _finite_tensor(
        prior,
        "prior",
        minimum_dimensions=1,
    )
    if tuple(candidate_value.shape) != tuple(prior_value.shape):
        raise Lfm2CpcfShortConvV4Error(
            "candidate and prior must have identical shapes."
        )
    radius = _positive_float(relative_radius, "relative_radius")
    floor = _positive_float(rms_floor, "rms_floor")
    prior_rms = float(prior_value.square().mean().sqrt().item())
    if reference_rms is None:
        reference = max(prior_rms, floor)
    else:
        reference = max(
            _positive_float(
                reference_rms,
                "reference_rms",
                allow_zero=True,
            ),
            floor,
        )
    limit = radius * reference
    correction = candidate_value - prior_value
    correction_rms = float(correction.square().mean().sqrt().item())
    scale = min(1.0, limit / max(correction_rms, floor))
    bounded = (prior_value + scale * correction).contiguous()
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "bounded_prior_correction",
        "relative_radius": radius,
        "reference_rms": reference,
        "correction_rms_before": correction_rms,
        "correction_rms_after": float(
            (bounded - prior_value).square().mean().sqrt().item()
        ),
        "clip_scale": scale,
        "clipped": scale < 1.0,
        "value_sha256": _tensor_sha256(bounded, "bounded.value"),
    }
    return Lfm2CpcfV4BoundedStep(
        value=bounded.detach(),
        diagnostics=_seal_report(body),
    )


def causal_depthwise_context(
    b_history: Any,
    c_history: Any,
    x_history: Any,
    kernel: Any,
    bias: Any | None,
) -> Any:
    """Replay the complete causal target-native depthwise context."""

    torch = _torch()
    b_value = _finite_tensor(
        b_history,
        "b_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    c_value = _finite_tensor(
        c_history,
        "c_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    x_value = _finite_tensor(
        x_history,
        "x_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    kernel_value = _finite_tensor(
        kernel,
        "kernel",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if tuple(b_value.shape) != tuple(c_value.shape) or tuple(b_value.shape) != tuple(
        x_value.shape
    ):
        raise Lfm2CpcfShortConvV4Error(
            "B, C, and x histories must have identical shapes."
        )
    batch, sequence, hidden = map(int, b_value.shape)
    if int(kernel_value.shape[0]) != hidden:
        raise Lfm2CpcfShortConvV4Error("kernel width differs from B/C/x histories.")
    bias_value = None
    if bias is not None:
        bias_value = _finite_tensor(
            bias,
            "bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(bias_value.shape) != (hidden,):
            raise Lfm2CpcfShortConvV4Error(
                "depthwise bias width differs from histories."
            )
    fir_length = int(kernel_value.shape[1])
    gated = (b_value * x_value).transpose(1, 2).contiguous()
    padded = torch.nn.functional.pad(gated, (fir_length - 1, 0))
    filtered = torch.nn.functional.conv1d(
        padded,
        kernel_value.reshape(hidden, 1, fir_length),
        bias=bias_value,
        groups=hidden,
    )
    if tuple(filtered.shape) != (batch, hidden, sequence):
        raise Lfm2CpcfShortConvV4Error(
            "causal depthwise replay changed sequence geometry."
        )
    return (c_value * filtered.transpose(1, 2)).contiguous()


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4VolterraReplay:
    """A sampled coupled-CP replay without materializing its cubic tensor."""

    b_history: Any
    c_history: Any
    x_history: Any
    context_history: Any
    write_history: Any
    arithmetic_report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arithmetic_report",
            MappingProxyType(dict(self.arithmetic_report)),
        )

    def verify_authentication(self) -> bool:
        if not _verify_sealed_report(self.arithmetic_report):
            return False
        state = {
            "b_history": self.b_history,
            "c_history": self.c_history,
            "x_history": self.x_history,
            "context_history": self.context_history,
            "write_history": self.write_history,
        }
        return self.arithmetic_report.get("replay_state_sha256") == _state_sha256(
            state,
            "replay",
        )


def replay_sampled_cp_volterra(
    child_input_history: Any,
    *,
    in_proj_weight: Any,
    in_proj_bias: Any | None,
    fir_kernel: Any,
    fir_bias: Any | None,
    writer_weight: Any,
    writer_bias: Any | None,
    target_dtype: str | None = None,
    valid_prefix_mask: Any | None = None,
) -> Lfm2CpcfV4VolterraReplay:
    """Replay the target's sampled lagged cubic/Volterra factorization.

    For each channel and lag, the replay evaluates the coupled rank-1 factor
    ``C_j(h_t) * K[j,l] * B_j(h_t-l) * X_j(h_t-l) * O[:,j]``.  It does so
    through ordinary affine projections and a grouped causal convolution, not
    by constructing the enormous order-four coefficient tensor.
    """

    torch = _torch()
    child_fp64 = _finite_tensor(
        child_input_history,
        "child_input_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    weight = _finite_tensor(
        in_proj_weight,
        "in_proj_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    target_hidden = int(child_fp64.shape[-1])
    if tuple(weight.shape) != (3 * target_hidden, target_hidden):
        raise Lfm2CpcfShortConvV4Error(
            "in_proj_weight must be [3T,T] for sampled CP replay."
        )
    dtype_name = (
        "float64" if target_dtype is None else _normalize_target_dtype(target_dtype)
    )
    compute_dtype = _torch_dtype(dtype_name)
    child = child_fp64.to(dtype=compute_dtype)
    weight_value = weight.to(dtype=compute_dtype)
    mask_value = None
    if valid_prefix_mask is not None:
        mask_value = _snapshot_optional_mask(
            valid_prefix_mask,
            "valid_prefix_mask",
        )
        if tuple(mask_value.shape) != tuple(child.shape[:2]):
            raise Lfm2CpcfShortConvV4Error(
                "valid_prefix_mask must cover batch and sequence rows."
            )
        child = child * mask_value.unsqueeze(-1).to(dtype=compute_dtype)
    projection = child @ weight_value.T
    if in_proj_bias is not None:
        bias = _finite_tensor(
            in_proj_bias,
            "in_proj_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(bias.shape) != (3 * target_hidden,):
            raise Lfm2CpcfShortConvV4Error("in_proj_bias has the wrong width.")
        projection = projection + bias.to(dtype=compute_dtype)
    b_value, c_value, x_value = projection.chunk(3, dim=-1)
    kernel_value = _finite_tensor(
        fir_kernel,
        "fir_kernel",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if int(kernel_value.shape[0]) != target_hidden:
        raise Lfm2CpcfShortConvV4Error("fir_kernel width differs from target channels.")
    fir_bias_value = None
    if fir_bias is not None:
        fir_bias_value = _finite_tensor(
            fir_bias,
            "fir_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(fir_bias_value.shape) != (target_hidden,):
            raise Lfm2CpcfShortConvV4Error("fir_bias has the wrong width.")
    fir_length = int(kernel_value.shape[1])
    gated = (b_value * x_value).transpose(1, 2).contiguous()
    padded = torch.nn.functional.pad(gated, (fir_length - 1, 0))
    filtered = torch.nn.functional.conv1d(
        padded,
        kernel_value.to(dtype=compute_dtype).reshape(
            target_hidden,
            1,
            fir_length,
        ),
        bias=(
            None if fir_bias_value is None else fir_bias_value.to(dtype=compute_dtype)
        ),
        groups=target_hidden,
    )
    context = c_value * filtered.transpose(1, 2)
    writer = _finite_tensor(
        writer_weight,
        "writer_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if int(writer.shape[1]) != target_hidden:
        raise Lfm2CpcfShortConvV4Error(
            "writer_weight input width differs from target channels."
        )
    write = context @ writer.to(dtype=compute_dtype).T
    if writer_bias is not None:
        out_bias = _finite_tensor(
            writer_bias,
            "writer_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(out_bias.shape) != (int(writer.shape[0]),):
            raise Lfm2CpcfShortConvV4Error("writer_bias has the wrong width.")
        write = write + out_bias.to(dtype=compute_dtype)
    state = {
        "b_history": b_value.contiguous(),
        "c_history": c_value.contiguous(),
        "x_history": x_value.contiguous(),
        "context_history": context.contiguous(),
        "write_history": write.contiguous(),
    }
    arithmetic = _seal_report(
        {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "sampled_cp_target_dtype_replay",
            "compute_device": "cpu",
            "parameter_storage_dtype": dtype_name,
            "intermediate_arithmetic_dtype": dtype_name,
            "parameters_materialized_before_replay": True,
            "intermediates_computed_at_declared_dtype": True,
            "final_output_cast_only": False,
            "valid_prefix_mask_applied_before_projection": mask_value is not None,
            "invalid_projected_history_zeroed_after_affine": False,
            "biased_projection_matches_lfm2_mask_before_linear": True,
            "parameter_state_sha256": _state_sha256(
                {
                    "in_proj_weight": weight_value,
                    "in_proj_bias": (
                        None if in_proj_bias is None else bias.to(dtype=compute_dtype)
                    ),
                    "fir_kernel": kernel_value.to(dtype=compute_dtype),
                    "fir_bias": (
                        None
                        if fir_bias_value is None
                        else fir_bias_value.to(dtype=compute_dtype)
                    ),
                    "writer_weight": writer.to(dtype=compute_dtype),
                    "writer_bias": (
                        None
                        if writer_bias is None
                        else out_bias.to(dtype=compute_dtype)
                    ),
                },
                "replay.parameters",
            ),
            "replay_state_sha256": _state_sha256(state, "replay"),
        }
    )
    return Lfm2CpcfV4VolterraReplay(
        **state,
        arithmetic_report=arithmetic,
    )


def _projection_case_values(
    case: Lfm2CpcfV4ProjectionCase,
    *,
    required_split: str,
) -> tuple[Any, Any, Any, Any, Any]:
    torch = _torch()
    if not isinstance(case, Lfm2CpcfV4ProjectionCase):
        raise Lfm2CpcfShortConvV4Error(
            "projection cases must be Lfm2CpcfV4ProjectionCase instances."
        )
    if case.split != required_split:
        raise Lfm2CpcfShortConvV4Error(
            f"projection requires {required_split!r} cases only."
        )
    if not case.verify_authentication():
        raise Lfm2CpcfShortConvV4Error(
            f"projection case {case.case_id!r} changed after authentication."
        )
    child = _finite_tensor(
        case.child_input_history,
        f"{case.case_id}.child_input_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    desired_context = _finite_tensor(
        case.desired_context_history,
        f"{case.case_id}.desired_context_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    desired_write = _finite_tensor(
        case.desired_write_suffix,
        f"{case.case_id}.desired_write_suffix",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    if tuple(child.shape) != tuple(desired_context.shape):
        raise Lfm2CpcfShortConvV4Error(
            f"{case.case_id} child and desired context shapes differ."
        )
    batch, sequence, _hidden = map(int, child.shape)
    if case.suffix_start >= sequence:
        raise Lfm2CpcfShortConvV4Error(
            f"{case.case_id}.suffix_start must be inside the sequence."
        )
    suffix_length = sequence - case.suffix_start
    if tuple(desired_write.shape[:2]) != (batch, suffix_length):
        raise Lfm2CpcfShortConvV4Error(
            f"{case.case_id} desired write does not cover its suffix."
        )
    if case.valid_prefix_mask is None:
        prefix_mask = torch.ones((batch, sequence), dtype=torch.bool)
    else:
        prefix_mask = case.valid_prefix_mask
        if (
            not isinstance(prefix_mask, torch.Tensor)
            or prefix_mask.layout != torch.strided
            or prefix_mask.device.type == "meta"
            or tuple(prefix_mask.shape) != (batch, sequence)
        ):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id}.valid_prefix_mask has the wrong shape."
            )
        prefix_mask = prefix_mask.detach().to(device="cpu", dtype=torch.bool)
    if case.valid_suffix_mask is None:
        suffix_mask = prefix_mask[:, case.suffix_start :].clone()
    else:
        suffix_mask = case.valid_suffix_mask
        if (
            not isinstance(suffix_mask, torch.Tensor)
            or suffix_mask.layout != torch.strided
            or suffix_mask.device.type == "meta"
            or tuple(suffix_mask.shape) != (batch, suffix_length)
        ):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id}.valid_suffix_mask has the wrong shape."
            )
        suffix_mask = suffix_mask.detach().to(device="cpu", dtype=torch.bool)
    if not bool(prefix_mask.any().item()) or not bool(suffix_mask.any().item()):
        raise Lfm2CpcfShortConvV4Error(f"{case.case_id} has no valid projection rows.")
    child = child * prefix_mask.unsqueeze(-1).to(torch.float64)
    return (
        child,
        desired_context,
        desired_write,
        prefix_mask.contiguous(),
        suffix_mask.contiguous(),
    )


def sampled_cp_volterra_objective(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: Sequence[Lfm2CpcfV4VolterraReplay],
    *,
    required_split: str = "fit",
    context_objective_weight: float = 0.05,
) -> Mapping[str, Any]:
    """Measure the equal-case sampled input-to-residual Volterra objective."""

    split = _normalize_split(required_split)
    if not cases or len(cases) != len(replays):
        raise Lfm2CpcfShortConvV4Error(
            "sampled CP objective requires aligned nonempty cases and replays."
        )
    context_weight = _positive_float(
        context_objective_weight,
        "context_objective_weight",
        allow_zero=True,
    )
    if context_weight > 0.1:
        raise Lfm2CpcfShortConvV4Error(
            "context_objective_weight must be between 0.0 and 0.1."
        )
    _canonical_unique_cases(cases, "sampled CP objective")
    ordered_pairs = sorted(
        zip(cases, replays),
        key=lambda value: value[0].case_id,
    )
    reports = []
    for case, replay in ordered_pairs:
        (
            _child,
            desired_context,
            desired_write,
            prefix_mask,
            suffix_mask,
        ) = _projection_case_values(case, required_split=split)
        if not isinstance(replay, Lfm2CpcfV4VolterraReplay):
            raise Lfm2CpcfShortConvV4Error(
                "sampled CP objective replays have the wrong type."
            )
        if not replay.verify_authentication():
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id} replay changed after authentication."
            )
        prediction = _finite_tensor(
            replay.write_history,
            f"{case.case_id}.write_history",
            minimum_dimensions=3,
            maximum_dimensions=3,
        )[:, case.suffix_start :, :]
        if tuple(prediction.shape) != tuple(desired_write.shape):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id} replay and desired write shapes differ."
            )
        prediction_rows = prediction.reshape(
            -1,
            int(prediction.shape[-1]),
        )[suffix_mask.reshape(-1)]
        target_rows = desired_write.reshape(
            -1,
            int(desired_write.shape[-1]),
        )[suffix_mask.reshape(-1)]
        context_prediction = _finite_tensor(
            replay.context_history,
            f"{case.case_id}.context_history",
            minimum_dimensions=3,
            maximum_dimensions=3,
        )
        if tuple(context_prediction.shape) != tuple(desired_context.shape):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id} replay and desired context shapes differ."
            )
        context_rows = context_prediction.reshape(
            -1,
            int(context_prediction.shape[-1]),
        )[prefix_mask.reshape(-1)]
        context_target_rows = desired_context.reshape(
            -1,
            int(desired_context.shape[-1]),
        )[prefix_mask.reshape(-1)]
        difference = prediction_rows - target_rows
        context_difference = context_rows - context_target_rows
        reports.append(
            {
                "case_id": case.case_id,
                "rows": int(prediction_rows.shape[0]),
                "mean_squared_error": float(difference.square().mean().item()),
                "relative_rmse": _relative_rmse(
                    prediction_rows,
                    target_rows,
                ),
                "cosine_similarity": _cosine_similarity(
                    prediction_rows,
                    target_rows,
                ),
                "context_mean_squared_error": float(
                    context_difference.square().mean().item()
                ),
                "context_relative_rmse": _relative_rmse(
                    context_rows,
                    context_target_rows,
                ),
                "context_cosine_similarity": _cosine_similarity(
                    context_rows,
                    context_target_rows,
                ),
            }
        )
    mean_endpoint = sum(float(value["relative_rmse"]) for value in reports) / len(
        reports
    )
    mean_context = sum(
        float(value["context_relative_rmse"]) for value in reports
    ) / len(reports)
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "sampled_coupled_cp_volterra_objective",
        "split": split,
        "case_balanced": True,
        "case_count": len(reports),
        "cases": reports,
        "mean_case_squared_error": sum(
            float(value["mean_squared_error"]) for value in reports
        )
        / len(reports),
        "mean_case_relative_rmse": mean_endpoint,
        "maximum_case_relative_rmse": max(
            float(value["relative_rmse"]) for value in reports
        ),
        "mean_case_cosine_similarity": sum(
            float(value["cosine_similarity"]) for value in reports
        )
        / len(reports),
        "minimum_case_cosine_similarity": min(
            float(value["cosine_similarity"]) for value in reports
        ),
        "context_objective_weight": context_weight,
        "mean_case_context_relative_rmse": mean_context,
        "maximum_case_context_relative_rmse": max(
            float(value["context_relative_rmse"]) for value in reports
        ),
        "mean_case_context_cosine_similarity": sum(
            float(value["context_cosine_similarity"]) for value in reports
        )
        / len(reports),
        "combined_fit_objective": mean_endpoint + context_weight * mean_context,
        "operator_objective": (
            "sampled_child_input_to_residual_lagged_cubic_coupled_cp"
        ),
        "huge_volterra_tensor_materialized": False,
    }
    return _seal_report(body)


def _lag_windows(value: Any, fir_length: int) -> Any:
    torch = _torch()
    padded = torch.nn.functional.pad(value, (fir_length - 1, 0))
    return padded.unfold(dimension=1, size=fir_length, step=1)


def _causal_depthwise_channel_from_values(
    b_channel: Any,
    c_channel: Any,
    x_channel: Any,
    kernel_row: Any,
    bias_scalar: Any | None,
) -> Any:
    """Replay one already-normalized channel without a grouped-conv launch."""

    windows = _lag_windows(b_channel * x_channel, int(kernel_row.shape[0]))
    filtered = (windows * kernel_row.reshape(1, 1, -1)).sum(dim=-1)
    if bias_scalar is not None:
        filtered = filtered + bias_scalar
    return (c_channel * filtered).contiguous()


def _linearized_bcx_affine_design_from_values(
    child: Any,
    b_channel: Any,
    c_channel: Any,
    x_channel: Any,
    kernel_row: Any,
    *,
    conv_bias_scalar: Any | None,
    fit_bias: bool,
) -> Any:
    """Build one channel Jacobian from already-normalized FP64 state."""

    torch = _torch()
    fir_length = int(kernel_row.shape[0])
    child_features = child
    if fit_bias:
        child_features = torch.cat(
            (
                child,
                torch.ones(
                    (*child.shape[:-1], 1),
                    dtype=torch.float64,
                ),
            ),
            dim=-1,
        )
    child_windows = (
        torch.nn.functional.pad(
            child_features.transpose(1, 2),
            (fir_length - 1, 0),
        )
        .unfold(dimension=2, size=fir_length, step=1)
        .permute(0, 2, 3, 1)
    )
    x_windows = _lag_windows(x_channel, fir_length)
    b_windows = _lag_windows(b_channel, fir_length)
    weighted_b = (
        child_windows * x_windows.unsqueeze(-1) * kernel_row.reshape(1, 1, -1, 1)
    ).sum(dim=2) * c_channel.unsqueeze(-1)
    weighted_x = (
        child_windows * b_windows.unsqueeze(-1) * kernel_row.reshape(1, 1, -1, 1)
    ).sum(dim=2) * c_channel.unsqueeze(-1)
    filtered = (x_windows * b_windows * kernel_row.reshape(1, 1, -1)).sum(dim=2)
    if conv_bias_scalar is not None:
        filtered = filtered + conv_bias_scalar
    weighted_c = child_features * filtered.unsqueeze(-1)
    return torch.cat((weighted_b, weighted_c, weighted_x), dim=-1).contiguous()


def linearized_bcx_affine_design(
    child_input_history: Any,
    b_history: Any,
    c_history: Any,
    x_history: Any,
    kernel: Any,
    conv_bias: Any | None,
    *,
    channel: int,
    fit_bias: bool,
) -> Any:
    """Return the per-channel Jacobian for affine B/C/x row corrections."""

    child = _finite_tensor(
        child_input_history,
        "child_input_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    b_value = _finite_tensor(
        b_history,
        "b_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    c_value = _finite_tensor(
        c_history,
        "c_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    x_value = _finite_tensor(
        x_history,
        "x_history",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    kernel_value = _finite_tensor(
        kernel,
        "kernel",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if (
        tuple(child.shape[:-1]) != tuple(b_value.shape[:-1])
        or tuple(b_value.shape) != tuple(c_value.shape)
        or tuple(b_value.shape) != tuple(x_value.shape)
    ):
        raise Lfm2CpcfShortConvV4Error(
            "linearized B/C/x tensors have inconsistent geometry."
        )
    hidden = int(b_value.shape[-1])
    if (
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or channel < 0
        or channel >= hidden
        or tuple(kernel_value.shape[:1]) != (hidden,)
    ):
        raise Lfm2CpcfShortConvV4Error("linearized channel or kernel width is invalid.")
    conv_bias_scalar = None
    if conv_bias is not None:
        bias_value = _finite_tensor(
            conv_bias,
            "conv_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(bias_value.shape) != (hidden,):
            raise Lfm2CpcfShortConvV4Error("conv_bias has the wrong width.")
        conv_bias_scalar = bias_value[channel]
    design = _linearized_bcx_affine_design_from_values(
        child,
        b_value[..., channel],
        c_value[..., channel],
        x_value[..., channel],
        kernel_value[channel],
        conv_bias_scalar=conv_bias_scalar,
        fit_bias=fit_bias,
    )
    expected = 3 * (int(child.shape[-1]) + int(fit_bias))
    if int(design.shape[-1]) != expected:
        raise Lfm2CpcfShortConvV4Error("linearized B/C/x design has the wrong width.")
    return design.contiguous()


def collapse_endpoint_context_sufficient_statistics(
    *,
    case_id: str,
    split: str,
    context_design: Any,
    endpoint_residual: Any,
    writer_column: Any,
    valid_rows: Any,
    context_residual: Any | None = None,
    context_coefficient: float = 0.0,
) -> tuple[Lfm2CpcfV4RidgeCase, Mapping[str, Any]]:
    """Collapse an endpoint output axis without changing ridge sufficient statistics.

    For ``J[n,d]``, endpoint residual ``R[n,o]`` and writer column ``w[o]``,
    the expanded design contains ``w[o] * J[n,d]``.  This helper returns an
    ``n``-row scalar-target case with the identical per-case mean Gram and
    cross-product.  An optional scalar context residual is incorporated with
    its independently normalized coefficient.
    """

    torch = _torch()
    design = _finite_tensor(
        context_design,
        "context_design",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    endpoint = _finite_tensor(
        endpoint_residual,
        "endpoint_residual",
        minimum_dimensions=3,
        maximum_dimensions=3,
    )
    writer = _finite_tensor(
        writer_column,
        "writer_column",
        minimum_dimensions=1,
        maximum_dimensions=1,
    )
    mask = _snapshot_optional_mask(valid_rows, "valid_rows")
    if mask is None or tuple(mask.shape) != tuple(design.shape[:2]):
        raise Lfm2CpcfShortConvV4Error(
            "valid_rows must cover the context design batch and sequence."
        )
    if tuple(endpoint.shape[:2]) != tuple(design.shape[:2]) or int(
        endpoint.shape[-1]
    ) != int(writer.shape[0]):
        raise Lfm2CpcfShortConvV4Error(
            "endpoint residual, writer, and context design are misaligned."
        )
    coefficient = _positive_float(
        context_coefficient,
        "context_coefficient",
        allow_zero=True,
    )
    context_value = None
    if context_residual is not None:
        context_value = _finite_tensor(
            context_residual,
            "context_residual",
            minimum_dimensions=2,
            maximum_dimensions=2,
        )
        if tuple(context_value.shape) != tuple(design.shape[:2]):
            raise Lfm2CpcfShortConvV4Error(
                "context_residual must cover the context design rows."
            )
    elif coefficient != 0.0:
        raise Lfm2CpcfShortConvV4Error(
            "a positive context_coefficient requires context_residual."
        )
    design_rows = design.reshape(-1, int(design.shape[-1]))[mask.reshape(-1)]
    endpoint_rows = endpoint.reshape(-1, int(endpoint.shape[-1]))[mask.reshape(-1)]
    context_rows = (
        None if context_value is None else context_value.reshape(-1)[mask.reshape(-1)]
    )
    output_width = int(writer.shape[0])
    writer_squared_norm = float(writer.square().sum().item())
    aggregate_coefficient = writer_squared_norm / output_width + coefficient
    if aggregate_coefficient == 0.0:
        reduced_design = torch.zeros_like(design_rows)
        reduced_target = torch.zeros(
            (int(design_rows.shape[0]), 1),
            dtype=torch.float64,
        )
    else:
        root = math.sqrt(aggregate_coefficient)
        right = (endpoint_rows @ writer.reshape(-1, 1)).reshape(-1)
        right = right / output_width
        if context_rows is not None:
            right = right + coefficient * context_rows
        reduced_design = design_rows * root
        reduced_target = (right / root).reshape(-1, 1)
    ridge_case = Lfm2CpcfV4RidgeCase(
        case_id=case_id,
        split=split,
        design=reduced_design,
        target=reduced_target,
    )
    report = _seal_report(
        {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "collapsed_endpoint_context_sufficient_statistics",
            "case_id": ridge_case.case_id,
            "split": ridge_case.split,
            "source_rows": int(design_rows.shape[0]),
            "design_columns": int(design_rows.shape[1]),
            "endpoint_output_width": output_width,
            "expanded_rows_avoided": int(design_rows.shape[0])
            * max(output_width - 1, 0),
            "writer_squared_norm": writer_squared_norm,
            "context_coefficient": coefficient,
            "aggregate_design_coefficient": aggregate_coefficient,
            "zero_information_noop": aggregate_coefficient == 0.0,
            "ridge_case_authentication_sha256": (ridge_case.authentication_sha256),
            "exact_per_case_mean_gram_and_right_preserved": True,
        }
    )
    return ridge_case, report


def fixed_variable_projection_schedule(
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[str, ...]:
    """Return the fixed sampled CP/Volterra alternating-factor schedule."""

    if not isinstance(policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    phases = (
        "sampled_cp_writer_factor",
        "sampled_cp_fir_factor",
        "sampled_cp_linearized_bcx_factors",
    )
    return phases * policy.variable_projection_passes


def solve_bounded_affine_projection(
    cases: Sequence[Lfm2CpcfV4RidgeCase],
    *,
    prior_weight: Any,
    prior_bias: Any | None,
    alpha: float,
    relative_radius: float,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
    required_split: str = "fit",
    reference_rms: float | None = None,
) -> tuple[Any, Any | None, Mapping[str, Any]]:
    """Solve and trust-bound one writer or affine producer projection."""

    torch = _torch()
    prior_weight_value = _finite_tensor(
        prior_weight,
        "prior_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    fit_bias = prior_bias is not None
    if fit_bias:
        prior_bias_value = _finite_tensor(
            prior_bias,
            "prior_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if int(prior_bias_value.shape[0]) != int(prior_weight_value.shape[0]):
            raise Lfm2CpcfShortConvV4Error(
                "prior affine bias width differs from weight outputs."
            )
        prior_solution = torch.cat(
            (
                prior_weight_value.T,
                prior_bias_value.reshape(1, -1),
            ),
            dim=0,
        )
    else:
        prior_solution = prior_weight_value.T
    result = case_balanced_trace_scaled_prior_ridge(
        cases,
        prior=prior_solution,
        alpha=alpha,
        penalty_mask=affine_penalty_mask(
            int(prior_weight_value.shape[1]),
            fit_bias=fit_bias,
        ),
        required_split=required_split,
        rank_relative_tolerance=policy.rank_relative_tolerance,
    )
    bounded = bound_prior_correction(
        result.solution,
        prior_solution,
        relative_radius=relative_radius,
        rms_floor=policy.correction_rms_floor,
        reference_rms=reference_rms,
    )
    value = bounded.value
    weight = value[: int(prior_weight_value.shape[1]), :].T.contiguous()
    bias = value[-1, :].contiguous() if fit_bias else None
    report = _seal_report(
        {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "bounded_affine_projection",
            "policy_sha256": policy.authentication_sha256,
            "ridge": dict(result.diagnostics),
            "trust_region": dict(bounded.diagnostics),
            "weight_sha256": _tensor_sha256(
                weight,
                "bounded_affine.weight",
            ),
            "bias_sha256": (
                None if bias is None else _tensor_sha256(bias, "bounded_affine.bias")
            ),
        }
    )
    return weight.detach(), None if bias is None else bias.detach(), report


def solve_bounded_writer_projection(
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    context_histories: Sequence[Any],
    prior_weight: Any,
    prior_bias: Any | None,
    alpha: float,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[Any, Any | None, Mapping[str, Any]]:
    """Solve the writer factor against the sampled residual endpoint."""

    torch = _torch()
    if not cases or len(cases) != len(context_histories):
        raise Lfm2CpcfShortConvV4Error(
            "writer projection requires aligned nonempty fit evidence."
        )
    ordered_pairs = sorted(
        zip(cases, context_histories),
        key=lambda value: value[0].case_id,
    )
    ordered_cases = _canonical_unique_cases(
        tuple(value[0] for value in ordered_pairs),
        "writer projection",
    )
    ordered_contexts = tuple(value[1] for value in ordered_pairs)
    fit_bias = prior_bias is not None
    ridge_cases = []
    target_rms_values = []
    for case, context_history in zip(ordered_cases, ordered_contexts):
        (
            _child,
            _desired_context,
            desired_write,
            _prefix_mask,
            suffix_mask,
        ) = _projection_case_values(case, required_split="fit")
        context = _finite_tensor(
            context_history,
            f"{case.case_id}.context_history",
            minimum_dimensions=3,
            maximum_dimensions=3,
        )
        suffix_context = context[:, case.suffix_start :, :]
        if tuple(suffix_context.shape[:2]) != tuple(desired_write.shape[:2]):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id} writer context has the wrong suffix geometry."
            )
        design = suffix_context.reshape(
            -1,
            int(suffix_context.shape[-1]),
        )
        if fit_bias:
            design = torch.cat(
                (
                    design,
                    torch.ones(
                        (int(design.shape[0]), 1),
                        dtype=torch.float64,
                    ),
                ),
                dim=1,
            )
        target = desired_write.reshape(
            -1,
            int(desired_write.shape[-1]),
        )
        ridge_cases.append(
            Lfm2CpcfV4RidgeCase(
                case_id=case.case_id,
                split="fit",
                design=design,
                target=target,
                valid_rows=suffix_mask.reshape(-1),
            )
        )
        target_rms_values.append(float(target.square().mean().sqrt().item()))
    weight, bias, generic_report = solve_bounded_affine_projection(
        ridge_cases,
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        alpha=alpha,
        relative_radius=policy.writer_trust_radius,
        policy=policy,
        required_split="fit",
        reference_rms=sum(target_rms_values) / len(target_rms_values),
    )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "sampled_cp_writer_factor_projection",
        "policy_sha256": policy.authentication_sha256,
        "operator_objective": (
            "sampled_child_input_to_residual_lagged_cubic_coupled_cp"
        ),
        "factor_held_fixed": ["B", "C", "X", "FIR"],
        "factor_solved": "residual_writer",
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "affine_projection": dict(generic_report),
    }
    return weight, bias, _seal_report(body)


def solve_bounded_fir_projection(
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    current_bcx: Sequence[tuple[Any, Any, Any]],
    current_kernel: Any,
    current_bias: Any | None,
    writer_weight: Any,
    writer_bias: Any | None,
    alpha: float,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[Any, Any | None, Mapping[str, Any]]:
    """Block-solve FIR factors against the sampled endpoint objective."""

    torch = _torch()
    if not cases or len(cases) != len(current_bcx):
        raise Lfm2CpcfShortConvV4Error(
            "FIR projection requires aligned nonempty fit evidence."
        )
    ordered_pairs = sorted(
        zip(cases, current_bcx),
        key=lambda value: value[0].case_id,
    )
    ordered_cases = _canonical_unique_cases(
        tuple(value[0] for value in ordered_pairs),
        "FIR projection",
    )
    ordered_bcx = tuple(value[1] for value in ordered_pairs)
    values = [
        _projection_case_values(case, required_split="fit") for case in ordered_cases
    ]
    kernel = _finite_tensor(
        current_kernel,
        "current_kernel",
        minimum_dimensions=2,
        maximum_dimensions=2,
    ).clone()
    target_hidden, fir_length = map(int, kernel.shape)
    bias = None
    if current_bias is not None:
        bias = _finite_tensor(
            current_bias,
            "current_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        ).clone()
        if tuple(bias.shape) != (target_hidden,):
            raise Lfm2CpcfShortConvV4Error("current FIR bias has the wrong width.")
    writer = _finite_tensor(
        writer_weight,
        "writer_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if int(writer.shape[1]) != target_hidden:
        raise Lfm2CpcfShortConvV4Error("writer width differs from FIR channels.")
    out_bias = None
    if writer_bias is not None:
        out_bias = _finite_tensor(
            writer_bias,
            "writer_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(out_bias.shape) != (int(writer.shape[0]),):
            raise Lfm2CpcfShortConvV4Error("writer bias has the wrong width.")
    normalized_bcx = []
    for case, (child, _desired_context, _desired_write, _pm, _sm), bcx in zip(
        ordered_cases,
        values,
        ordered_bcx,
    ):
        if not isinstance(bcx, tuple) or len(bcx) != 3:
            raise Lfm2CpcfShortConvV4Error("current_bcx must contain B/C/x tuples.")
        normalized = tuple(
            _finite_tensor(
                value,
                f"{case.case_id}.current_bcx",
                minimum_dimensions=3,
                maximum_dimensions=3,
            )
            for value in bcx
        )
        if any(tuple(value.shape) != tuple(child.shape) for value in normalized):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id} B/C/x geometry differs from child input."
            )
        normalized_bcx.append(normalized)
    channel_reports = []
    context_coefficient = policy.context_objective_weight / target_hidden
    contexts = [
        causal_depthwise_context(
            *bcx,
            kernel,
            bias,
        )
        for bcx in normalized_bcx
    ]
    writes = [context @ writer.T for context in contexts]
    if out_bias is not None:
        writes = [write + out_bias for write in writes]
    for channel in range(target_hidden):
        writer_column = writer[:, channel]
        if (
            float(writer_column.square().sum().item()) == 0.0
            and context_coefficient == 0.0
        ):
            channel_reports.append(
                {
                    "channel": channel,
                    "zero_information_noop": True,
                    "collapsed_case_reports": [],
                }
            )
            continue
        ridge_cases = []
        collapse_reports = []
        channel_designs = []
        for (
            case,
            (
                _child,
                desired_context,
                desired_write,
                prefix_mask,
                suffix_mask,
            ),
            (b_value, c_value, x_value),
            context,
            write,
        ) in zip(ordered_cases, values, normalized_bcx, contexts, writes):
            current_channel_write = context[..., channel].unsqueeze(-1) * writer[
                :, channel
            ].reshape(1, 1, -1)
            other_write = write - current_channel_write
            target = desired_write - other_write[:, case.suffix_start :, :]
            bx_windows = _lag_windows(
                b_value[..., channel] * x_value[..., channel],
                fir_length,
            )
            base_design = bx_windows * c_value[..., channel].unsqueeze(-1)
            if bias is not None:
                base_design = torch.cat(
                    (
                        base_design,
                        c_value[..., channel].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            channel_designs.append(base_design)
            suffix_design = base_design[:, case.suffix_start :, :]
            endpoint_case, collapse_report = (
                collapse_endpoint_context_sufficient_statistics(
                    case_id=f"{case.case_id}::endpoint",
                    split="fit",
                    context_design=suffix_design,
                    endpoint_residual=target,
                    writer_column=writer_column,
                    valid_rows=suffix_mask,
                )
            )
            ridge_cases.append(endpoint_case)
            collapse_reports.append(dict(collapse_report))
            if context_coefficient > 0.0:
                root = math.sqrt(context_coefficient)
                context_rows = base_design.reshape(
                    -1,
                    int(base_design.shape[-1]),
                )[prefix_mask.reshape(-1)]
                context_target = desired_context[..., channel].reshape(-1)[
                    prefix_mask.reshape(-1)
                ]
                ridge_cases.append(
                    Lfm2CpcfV4RidgeCase(
                        case_id=f"{case.case_id}::context",
                        split="fit",
                        design=context_rows * root,
                        target=(context_target * root).reshape(-1, 1),
                    )
                )
        prior_parts = [kernel[channel]]
        if bias is not None:
            prior_parts.append(bias[channel].reshape(1))
        prior = torch.cat(prior_parts).reshape(-1, 1)
        solve = case_balanced_trace_scaled_prior_ridge(
            ridge_cases,
            prior=prior,
            alpha=alpha,
            penalty_mask=fir_penalty_mask(
                fir_length,
                fit_bias=bias is not None,
            ),
            required_split="fit",
            rank_relative_tolerance=policy.rank_relative_tolerance,
        )
        bounded = bound_prior_correction(
            solve.solution,
            prior,
            relative_radius=policy.fir_trust_radius,
            rms_floor=policy.correction_rms_floor,
        )
        kernel[channel] = bounded.value[:fir_length, 0]
        if bias is not None:
            bias[channel] = bounded.value[-1, 0]
        for case_index, base_design in enumerate(channel_designs):
            new_context = (base_design @ bounded.value[:, 0]).contiguous()
            old_context = contexts[case_index][..., channel]
            delta = new_context - old_context
            contexts[case_index][..., channel] = new_context
            writes[case_index] = (
                writes[case_index]
                + delta.unsqueeze(-1) * writer_column.reshape(1, 1, -1)
            ).contiguous()
        channel_reports.append(
            {
                "channel": channel,
                "ridge_report_sha256": solve.diagnostics["report_sha256"],
                "trust_report_sha256": bounded.diagnostics["report_sha256"],
                "zero_information_noop": False,
                "collapsed_case_reports": collapse_reports,
            }
        )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "sampled_cp_fir_factor_projection",
        "policy_sha256": policy.authentication_sha256,
        "operator_objective": (
            "sampled_child_input_to_residual_lagged_cubic_coupled_cp"
        ),
        "factor_held_fixed": ["B", "C", "X", "residual_writer"],
        "factor_solved": "depthwise_FIR",
        "huge_volterra_tensor_materialized": False,
        "full_operator_replays_per_channel": 0,
        "cached_full_context_replays": len(contexts),
        "single_channel_state_updates": target_hidden,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "channel_reports": channel_reports,
        "kernel_sha256": _tensor_sha256(kernel, "sampled_cp_fir.kernel"),
        "bias_sha256": (
            None if bias is None else _tensor_sha256(bias, "sampled_cp_fir.bias")
        ),
    }
    return (
        kernel.detach(),
        None if bias is None else bias.detach(),
        _seal_report(body),
    )


def solve_bounded_linearized_bcx_projection(
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    kernel: Any,
    conv_bias: Any | None,
    writer_weight: Any,
    writer_bias: Any | None,
    current_in_proj_weight: Any,
    current_in_proj_bias: Any | None,
    alpha: float,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[Any, Any | None, Mapping[str, Any]]:
    """Block-solve linearized B/C/x factors on endpoint residuals.

    The Jacobian is evaluated from the sampled coupled CP/Volterra replay and
    multiplied by the current writer factor before solving.  It therefore
    targets residual endpoint error directly instead of independently matching
    B, C, or x coordinates.
    """

    torch = _torch()
    ordered_cases = _canonical_unique_cases(
        cases,
        "linearized B/C/x projection",
    )
    values = [
        _projection_case_values(case, required_split="fit") for case in ordered_cases
    ]
    weight = _finite_tensor(
        current_in_proj_weight,
        "current_in_proj_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    ).clone()
    target_hidden = int(weight.shape[1])
    if tuple(weight.shape) != (3 * target_hidden, target_hidden):
        raise Lfm2CpcfShortConvV4Error("current_in_proj_weight must be [3T,T].")
    fit_bias = current_in_proj_bias is not None
    bias = None
    if fit_bias:
        bias = _finite_tensor(
            current_in_proj_bias,
            "current_in_proj_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(bias.shape) != (3 * target_hidden,):
            raise Lfm2CpcfShortConvV4Error("current_in_proj_bias has the wrong shape.")
        bias = bias.clone()
    kernel_value = _finite_tensor(
        kernel,
        "kernel",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if int(kernel_value.shape[0]) != target_hidden:
        raise Lfm2CpcfShortConvV4Error("kernel width differs from target channels.")
    conv_bias_value = None
    if conv_bias is not None:
        conv_bias_value = _finite_tensor(
            conv_bias,
            "conv_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(conv_bias_value.shape) != (target_hidden,):
            raise Lfm2CpcfShortConvV4Error("conv_bias has the wrong width.")
    writer = _finite_tensor(
        writer_weight,
        "writer_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    )
    if int(writer.shape[1]) != target_hidden:
        raise Lfm2CpcfShortConvV4Error("writer width differs from target channels.")
    writer_bias_value = None
    if writer_bias is not None:
        writer_bias_value = _finite_tensor(
            writer_bias,
            "writer_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        )
        if tuple(writer_bias_value.shape) != (int(writer.shape[0]),):
            raise Lfm2CpcfShortConvV4Error("writer_bias has the wrong width.")
    channel_reports = []
    feature_width = target_hidden + int(fit_bias)
    penalty = torch.cat(
        (
            affine_penalty_mask(target_hidden, fit_bias=fit_bias),
            affine_penalty_mask(target_hidden, fit_bias=fit_bias),
            affine_penalty_mask(target_hidden, fit_bias=fit_bias),
        )
    )
    context_coefficient = policy.context_objective_weight / target_hidden
    cached_b = []
    cached_c = []
    cached_x = []
    cached_contexts = []
    cached_writes = []
    for child, _desired_context, _desired_write, prefix_mask, _suffix_mask in values:
        replay = replay_sampled_cp_volterra(
            child,
            in_proj_weight=weight,
            in_proj_bias=bias,
            fir_kernel=kernel_value,
            fir_bias=conv_bias_value,
            writer_weight=writer,
            writer_bias=writer_bias_value,
            valid_prefix_mask=prefix_mask,
        )
        cached_b.append(replay.b_history)
        cached_c.append(replay.c_history)
        cached_x.append(replay.x_history)
        cached_contexts.append(replay.context_history)
        cached_writes.append(replay.write_history)
    for channel in range(target_hidden):
        writer_column = writer[:, channel]
        if (
            float(writer_column.square().sum().item()) == 0.0
            and context_coefficient == 0.0
        ):
            channel_reports.append(
                {
                    "channel": channel,
                    "zero_information_noop": True,
                    "collapsed_case_reports": [],
                }
            )
            continue
        ridge_cases = []
        collapse_reports = []
        for case_index, (
            case,
            (
                child,
                desired_context,
                desired_write,
                prefix_mask,
                suffix_mask,
            ),
        ) in enumerate(zip(ordered_cases, values)):
            full_context_design = _linearized_bcx_affine_design_from_values(
                child,
                cached_b[case_index][..., channel],
                cached_c[case_index][..., channel],
                cached_x[case_index][..., channel],
                kernel_value[channel],
                conv_bias_scalar=(
                    None if conv_bias_value is None else conv_bias_value[channel]
                ),
                fit_bias=fit_bias,
            )
            context_design = full_context_design[:, case.suffix_start :, :]
            target = (
                desired_write - cached_writes[case_index][:, case.suffix_start :, :]
            )
            endpoint_case, collapse_report = (
                collapse_endpoint_context_sufficient_statistics(
                    case_id=f"{case.case_id}::endpoint",
                    split="fit",
                    context_design=context_design,
                    endpoint_residual=target,
                    writer_column=writer_column,
                    valid_rows=suffix_mask,
                )
            )
            ridge_cases.append(endpoint_case)
            collapse_reports.append(dict(collapse_report))
            if context_coefficient > 0.0:
                root = math.sqrt(context_coefficient)
                context_rows = full_context_design.reshape(
                    -1,
                    int(full_context_design.shape[-1]),
                )[prefix_mask.reshape(-1)]
                context_residual = (
                    desired_context[..., channel]
                    - cached_contexts[case_index][..., channel]
                ).reshape(-1)[prefix_mask.reshape(-1)]
                ridge_cases.append(
                    Lfm2CpcfV4RidgeCase(
                        case_id=f"{case.case_id}::context",
                        split="fit",
                        design=context_rows * root,
                        target=(context_residual * root).reshape(-1, 1),
                    )
                )
        solve = case_balanced_trace_scaled_prior_ridge(
            ridge_cases,
            prior=torch.zeros(
                (3 * feature_width, 1),
                dtype=torch.float64,
            ),
            alpha=alpha,
            penalty_mask=penalty,
            required_split="fit",
            rank_relative_tolerance=policy.rank_relative_tolerance,
        )
        bounded = bound_prior_correction(
            solve.solution,
            torch.zeros_like(solve.solution),
            relative_radius=policy.bcx_trust_radius,
            rms_floor=policy.correction_rms_floor,
            reference_rms=float(
                weight[[channel, target_hidden + channel, 2 * target_hidden + channel]]
                .square()
                .mean()
                .sqrt()
                .item()
            ),
        )
        correction = bounded.value.reshape(3, feature_width)
        for role in range(3):
            row = role * target_hidden + channel
            weight[row] += correction[role, :target_hidden]
            if bias is not None:
                bias[row] += correction[role, -1]
        role_rows = weight[
            [channel, target_hidden + channel, 2 * target_hidden + channel]
        ]
        role_bias = (
            None
            if bias is None
            else bias[[channel, target_hidden + channel, 2 * target_hidden + channel]]
        )
        for case_index, (
            child,
            _desired_context,
            _desired_write,
            prefix_mask,
            _suffix_mask,
        ) in enumerate(values):
            projected = child @ role_rows.T
            if role_bias is not None:
                projected = projected + role_bias
            new_b, new_c, new_x = projected.unbind(dim=-1)
            new_context = _causal_depthwise_channel_from_values(
                new_b,
                new_c,
                new_x,
                kernel_value[channel],
                (None if conv_bias_value is None else conv_bias_value[channel]),
            )
            old_context = cached_contexts[case_index][..., channel]
            delta = new_context - old_context
            cached_b[case_index][..., channel] = new_b
            cached_c[case_index][..., channel] = new_c
            cached_x[case_index][..., channel] = new_x
            cached_contexts[case_index][..., channel] = new_context
            cached_writes[case_index] = (
                cached_writes[case_index]
                + delta.unsqueeze(-1) * writer_column.reshape(1, 1, -1)
            ).contiguous()
        channel_reports.append(
            {
                "channel": channel,
                "ridge_report_sha256": solve.diagnostics["report_sha256"],
                "trust_report_sha256": bounded.diagnostics["report_sha256"],
                "zero_information_noop": False,
                "collapsed_case_reports": collapse_reports,
            }
        )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "sampled_cp_linearized_bcx_factor_projection",
        "policy_sha256": policy.authentication_sha256,
        "operator_objective": (
            "sampled_child_input_to_residual_lagged_cubic_coupled_cp"
        ),
        "factor_held_fixed": ["depthwise_FIR", "residual_writer"],
        "factor_solved": ["B", "C", "X"],
        "endpoint_jacobian_used": True,
        "huge_volterra_tensor_materialized": False,
        "full_operator_replays_per_channel": 0,
        "cached_full_operator_replays": len(cached_contexts),
        "single_channel_exact_state_updates": target_hidden,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "channel_reports": channel_reports,
        "weight_sha256": _tensor_sha256(
            weight,
            "linearized_bcx.weight",
        ),
        "bias_sha256": (
            None if bias is None else _tensor_sha256(bias, "linearized_bcx.bias")
        ),
        "execution": {
            "fixed_blockwise_closed_form": True,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    return (
        weight.detach(),
        None if bias is None else bias.detach(),
        _seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4VariableProjectionResult:
    """State produced by a fixed number of sampled CP factor sweeps."""

    in_proj_weight: Any
    in_proj_bias: Any | None
    fir_kernel: Any
    fir_bias: Any | None
    writer_weight: Any
    writer_bias: Any | None
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        if not _verify_sealed_report(self.report):
            return False
        state = {
            "in_proj_weight": self.in_proj_weight,
            "in_proj_bias": self.in_proj_bias,
            "fir_kernel": self.fir_kernel,
            "fir_bias": self.fir_bias,
            "writer_weight": self.writer_weight,
            "writer_bias": self.writer_bias,
        }
        return self.report.get("state_sha256") == _state_sha256(
            state,
            "variable_projection",
        )


def run_fixed_sampled_cp_variable_projection(
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    initial_in_proj_weight: Any,
    initial_in_proj_bias: Any | None,
    initial_fir_kernel: Any,
    initial_fir_bias: Any | None,
    initial_writer_weight: Any,
    initial_writer_bias: Any | None,
    alpha: float,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> Lfm2CpcfV4VariableProjectionResult:
    """Run exactly two or three bounded sampled CP/Volterra factor sweeps."""

    if not isinstance(policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    if not policy.verify_authentication():
        raise Lfm2CpcfShortConvV4Error("solver policy failed authentication.")
    ordered_cases = _canonical_unique_cases(cases, "variable projection")
    values = [
        _projection_case_values(case, required_split="fit") for case in ordered_cases
    ]
    in_weight = _finite_tensor(
        initial_in_proj_weight,
        "initial_in_proj_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    ).clone()
    in_bias = (
        None
        if initial_in_proj_bias is None
        else _finite_tensor(
            initial_in_proj_bias,
            "initial_in_proj_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        ).clone()
    )
    kernel = _finite_tensor(
        initial_fir_kernel,
        "initial_fir_kernel",
        minimum_dimensions=2,
        maximum_dimensions=2,
    ).clone()
    fir_bias = (
        None
        if initial_fir_bias is None
        else _finite_tensor(
            initial_fir_bias,
            "initial_fir_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        ).clone()
    )
    writer = _finite_tensor(
        initial_writer_weight,
        "initial_writer_weight",
        minimum_dimensions=2,
        maximum_dimensions=2,
    ).clone()
    writer_bias = (
        None
        if initial_writer_bias is None
        else _finite_tensor(
            initial_writer_bias,
            "initial_writer_bias",
            minimum_dimensions=1,
            maximum_dimensions=1,
        ).clone()
    )

    def replay_all() -> list[Lfm2CpcfV4VolterraReplay]:
        return [
            replay_sampled_cp_volterra(
                child,
                in_proj_weight=in_weight,
                in_proj_bias=in_bias,
                fir_kernel=kernel,
                fir_bias=fir_bias,
                writer_weight=writer,
                writer_bias=writer_bias,
                valid_prefix_mask=prefix,
            )
            for child, _context, _write, prefix, _suffix in values
        ]

    def objective() -> Mapping[str, Any]:
        return sampled_cp_volterra_objective(
            ordered_cases,
            replay_all(),
            required_split="fit",
            context_objective_weight=policy.context_objective_weight,
        )

    def accept_objective(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> bool:
        before_value = float(before["combined_fit_objective"])
        after_value = float(after["combined_fit_objective"])
        tolerance = 1e-12 * max(1.0, abs(before_value))
        return math.isfinite(after_value) and after_value <= before_value + tolerance

    def rejected_phase_report(phase: str, error: Exception) -> Mapping[str, Any]:
        return _seal_report(
            {
                "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
                "kind": "nonfinite_or_numeric_phase_rejection",
                "phase": phase,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "candidate_state_installed": False,
                "fit_state_reverted": True,
            }
        )

    sweep_reports = []
    initial_replays = replay_all()
    initial_objective = sampled_cp_volterra_objective(
        ordered_cases,
        initial_replays,
        required_split="fit",
        context_objective_weight=policy.context_objective_weight,
    )
    for pass_index in range(policy.variable_projection_passes):
        before_replays = replay_all()
        before_objective = sampled_cp_volterra_objective(
            ordered_cases,
            before_replays,
            required_split="fit",
            context_objective_weight=policy.context_objective_weight,
        )
        previous_writer = writer.clone()
        previous_writer_bias = None if writer_bias is None else writer_bias.clone()
        try:
            writer_candidate, writer_bias_candidate, writer_report = (
                solve_bounded_writer_projection(
                    cases=ordered_cases,
                    context_histories=[
                        replay.context_history for replay in before_replays
                    ],
                    prior_weight=writer,
                    prior_bias=writer_bias,
                    alpha=alpha,
                    policy=policy,
                )
            )
            writer = writer_candidate
            writer_bias = writer_bias_candidate
            writer_objective = objective()
            writer_accepted = accept_objective(before_objective, writer_objective)
        except (Lfm2CpcfShortConvV4Error, RuntimeError) as exc:
            writer = previous_writer
            writer_bias = previous_writer_bias
            writer_objective = before_objective
            writer_accepted = False
            writer_report = rejected_phase_report("writer", exc)
        if not writer_accepted:
            writer = previous_writer
            writer_bias = previous_writer_bias
            writer_objective = before_objective
        writer_replays = replay_all()
        previous_kernel = kernel.clone()
        previous_fir_bias = None if fir_bias is None else fir_bias.clone()
        try:
            kernel_candidate, fir_bias_candidate, fir_report = (
                solve_bounded_fir_projection(
                    cases=ordered_cases,
                    current_bcx=[
                        (
                            replay.b_history,
                            replay.c_history,
                            replay.x_history,
                        )
                        for replay in writer_replays
                    ],
                    current_kernel=kernel,
                    current_bias=fir_bias,
                    writer_weight=writer,
                    writer_bias=writer_bias,
                    alpha=alpha,
                    policy=policy,
                )
            )
            kernel = kernel_candidate
            fir_bias = fir_bias_candidate
            fir_objective = objective()
            fir_accepted = accept_objective(writer_objective, fir_objective)
        except (Lfm2CpcfShortConvV4Error, RuntimeError) as exc:
            kernel = previous_kernel
            fir_bias = previous_fir_bias
            fir_objective = writer_objective
            fir_accepted = False
            fir_report = rejected_phase_report("fir", exc)
        if not fir_accepted:
            kernel = previous_kernel
            fir_bias = previous_fir_bias
            fir_objective = writer_objective
        previous_in_weight = in_weight.clone()
        previous_in_bias = None if in_bias is None else in_bias.clone()
        try:
            in_weight_candidate, in_bias_candidate, bcx_report = (
                solve_bounded_linearized_bcx_projection(
                    cases=ordered_cases,
                    kernel=kernel,
                    conv_bias=fir_bias,
                    writer_weight=writer,
                    writer_bias=writer_bias,
                    current_in_proj_weight=in_weight,
                    current_in_proj_bias=in_bias,
                    alpha=alpha,
                    policy=policy,
                )
            )
            in_weight = in_weight_candidate
            in_bias = in_bias_candidate
            bcx_objective = objective()
            bcx_accepted = accept_objective(fir_objective, bcx_objective)
        except (Lfm2CpcfShortConvV4Error, RuntimeError) as exc:
            in_weight = previous_in_weight
            in_bias = previous_in_bias
            bcx_objective = fir_objective
            bcx_accepted = False
            bcx_report = rejected_phase_report("bcx", exc)
        if not bcx_accepted:
            in_weight = previous_in_weight
            in_bias = previous_in_bias
            bcx_objective = fir_objective
        after_objective = bcx_objective
        sweep_reports.append(
            {
                "pass_index": pass_index,
                "before_objective_sha256": before_objective["report_sha256"],
                "after_objective_sha256": after_objective["report_sha256"],
                "before_mean_case_relative_rmse": before_objective[
                    "mean_case_relative_rmse"
                ],
                "after_mean_case_relative_rmse": after_objective[
                    "mean_case_relative_rmse"
                ],
                "before_combined_fit_objective": before_objective[
                    "combined_fit_objective"
                ],
                "after_combined_fit_objective": after_objective[
                    "combined_fit_objective"
                ],
                "writer_phase_accepted": writer_accepted,
                "fir_phase_accepted": fir_accepted,
                "bcx_phase_accepted": bcx_accepted,
                "writer_report_sha256": writer_report["report_sha256"],
                "fir_report_sha256": fir_report["report_sha256"],
                "bcx_report_sha256": bcx_report["report_sha256"],
            }
        )
    final_replays = replay_all()
    final_objective = sampled_cp_volterra_objective(
        ordered_cases,
        final_replays,
        required_split="fit",
        context_objective_weight=policy.context_objective_weight,
    )
    if not accept_objective(initial_objective, final_objective):
        raise Lfm2CpcfShortConvV4Error(
            "fit-only accept/revert invariant allowed final objective regression."
        )
    state = {
        "in_proj_weight": in_weight,
        "fir_kernel": kernel,
        "writer_weight": writer,
    }
    if in_bias is not None:
        state["in_proj_bias"] = in_bias
    if fir_bias is not None:
        state["fir_bias"] = fir_bias
    if writer_bias is not None:
        state["writer_bias"] = writer_bias
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "fixed_bounded_sampled_cp_variable_projection",
        "policy_sha256": policy.authentication_sha256,
        "alpha": _positive_float(alpha, "alpha", allow_zero=True),
        "passes": policy.variable_projection_passes,
        "schedule": list(fixed_variable_projection_schedule(policy)),
        "operator_objective": (
            "sampled_child_input_to_residual_lagged_cubic_coupled_cp"
        ),
        "coupled_cp_rank": int(kernel.shape[0]),
        "huge_volterra_tensor_materialized": False,
        "initial_objective": dict(initial_objective),
        "final_objective": dict(final_objective),
        "sweeps": sweep_reports,
        "fit_only_accept_revert": True,
        "final_objective_not_worse_than_initial": True,
        "fit_case_authentication_sha256": [
            case.authentication_sha256 for case in ordered_cases
        ],
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "state_sha256": _sha256(
            {
                name: _tensor_sha256(value, f"variable_projection.{name}")
                for name, value in sorted(state.items())
            }
        ),
        "execution": {
            "fixed_passes": True,
            "closed_form_factor_solves": True,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    return Lfm2CpcfV4VariableProjectionResult(
        in_proj_weight=in_weight.detach(),
        in_proj_bias=None if in_bias is None else in_bias.detach(),
        fir_kernel=kernel.detach(),
        fir_bias=None if fir_bias is None else fir_bias.detach(),
        writer_weight=writer.detach(),
        writer_bias=None if writer_bias is None else writer_bias.detach(),
        report=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV4EvaluationCase:
    case_id: str
    split: str
    prediction: Any
    target: Any
    valid_rows: Any | None = None
    candidate_state_sha256: str = ""
    projection_case_authentication_sha256: str = ""
    context_prediction: Any | None = None
    context_target: Any | None = None
    valid_context_rows: Any | None = None
    arithmetic_report: Mapping[str, Any] | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise Lfm2CpcfShortConvV4Error("case_id must be nonempty text.")
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "split", _normalize_split(self.split))
        object.__setattr__(
            self,
            "prediction",
            _snapshot_dense_tensor(
                self.prediction,
                f"{self.case_id}.prediction",
            ),
        )
        object.__setattr__(
            self,
            "target",
            _snapshot_dense_tensor(
                self.target,
                f"{self.case_id}.target",
            ),
        )
        object.__setattr__(
            self,
            "valid_rows",
            _snapshot_optional_mask(
                self.valid_rows,
                f"{self.case_id}.valid_rows",
            ),
        )
        context_values = (self.context_prediction, self.context_target)
        if (context_values[0] is None) != (context_values[1] is None):
            raise Lfm2CpcfShortConvV4Error(
                "context_prediction and context_target must either both be present "
                "or both be absent."
            )
        if context_values[0] is not None:
            object.__setattr__(
                self,
                "context_prediction",
                _snapshot_dense_tensor(
                    context_values[0],
                    f"{self.case_id}.context_prediction",
                ),
            )
            object.__setattr__(
                self,
                "context_target",
                _snapshot_dense_tensor(
                    context_values[1],
                    f"{self.case_id}.context_target",
                ),
            )
            object.__setattr__(
                self,
                "valid_context_rows",
                _snapshot_optional_mask(
                    self.valid_context_rows,
                    f"{self.case_id}.valid_context_rows",
                ),
            )
        elif self.valid_context_rows is not None:
            raise Lfm2CpcfShortConvV4Error(
                "valid_context_rows requires context prediction and target evidence."
            )
        state_sha = str(self.candidate_state_sha256).strip().lower()
        if not state_sha:
            state_sha = _state_sha256(
                {"prediction": self.prediction},
                f"{self.case_id}.candidate",
            )
        object.__setattr__(
            self,
            "candidate_state_sha256",
            _sha256_text(state_sha, "candidate_state_sha256"),
        )
        projection_sha = str(self.projection_case_authentication_sha256).strip().lower()
        if projection_sha:
            projection_sha = _sha256_text(
                projection_sha,
                "projection_case_authentication_sha256",
            )
        object.__setattr__(
            self,
            "projection_case_authentication_sha256",
            projection_sha,
        )
        if self.arithmetic_report is not None:
            if not isinstance(self.arithmetic_report, Mapping):
                raise Lfm2CpcfShortConvV4Error(
                    "arithmetic_report must be a sealed mapping."
                )
            object.__setattr__(
                self,
                "arithmetic_report",
                MappingProxyType(dict(self.arithmetic_report)),
            )
            if not _verify_sealed_report(self.arithmetic_report):
                raise Lfm2CpcfShortConvV4Error(
                    "arithmetic_report authentication failed."
                )
            if (
                self.arithmetic_report.get("parameter_state_sha256")
                != self.candidate_state_sha256
            ):
                raise Lfm2CpcfShortConvV4Error(
                    "arithmetic_report is not bound to the candidate state."
                )
            if projection_sha and (
                self.arithmetic_report.get("projection_case_authentication_sha256")
                != projection_sha
            ):
                raise Lfm2CpcfShortConvV4Error(
                    "arithmetic_report is not bound to the projection evidence."
                )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV4Error(
                "evaluation case authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def evidence_body(self) -> dict[str, Any]:
        tensors = {
            "target": _tensor_sha256(self.target, f"{self.case_id}.target"),
        }
        if self.valid_rows is not None:
            tensors["valid_rows"] = _tensor_sha256(
                self.valid_rows,
                f"{self.case_id}.valid_rows",
            )
        if self.context_target is not None:
            tensors["context_target"] = _tensor_sha256(
                self.context_target,
                f"{self.case_id}.context_target",
            )
            if self.valid_context_rows is not None:
                tensors["valid_context_rows"] = _tensor_sha256(
                    self.valid_context_rows,
                    f"{self.case_id}.valid_context_rows",
                )
        return {
            "case_id": self.case_id,
            "split": self.split,
            "target_shape": [int(value) for value in self.target.shape],
            "context_target_shape": (
                None
                if self.context_target is None
                else [int(value) for value in self.context_target.shape]
            ),
            "projection_case_authentication_sha256": (
                self.projection_case_authentication_sha256 or None
            ),
            "tensors": tensors,
        }

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "kind": "evaluation_case",
            **self.evidence_body(),
            "prediction_sha256": _tensor_sha256(
                self.prediction,
                f"{self.case_id}.prediction",
            ),
            "context_prediction_sha256": (
                None
                if self.context_prediction is None
                else _tensor_sha256(
                    self.context_prediction,
                    f"{self.case_id}.context_prediction",
                )
            ),
            "candidate_state_sha256": self.candidate_state_sha256,
            "arithmetic_report_sha256": (
                None
                if self.arithmetic_report is None
                else self.arithmetic_report.get("report_sha256")
            ),
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())


def _relative_rmse(prediction: Any, target: Any) -> float:
    torch = _torch()
    error = (prediction - target).square().mean().sqrt()
    scale = target.square().mean().sqrt()
    return float(
        (error / torch.clamp(scale, min=torch.finfo(torch.float64).eps)).item()
    )


def _cosine_similarity(prediction: Any, target: Any) -> float:
    torch = _torch()
    left = prediction.reshape(-1)
    right = target.reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    floor = torch.finfo(torch.float64).eps
    if float(left_norm.item()) <= floor and float(right_norm.item()) <= floor:
        return 1.0
    if float(left_norm.item()) <= floor or float(right_norm.item()) <= floor:
        return 0.0
    return float(torch.dot(left, right).div(left_norm * right_norm).item())


def evaluate_target_dtype_predictions(
    cases: Sequence[Lfm2CpcfV4EvaluationCase],
    *,
    target_dtype: str,
    required_split: str,
    policy_sha256: str,
) -> Mapping[str, Any]:
    """Evaluate already-replayed predictions after target-dtype materialization."""

    torch = _torch()
    ordered_cases = _canonical_unique_cases(cases, "target-dtype evaluation")
    split = _normalize_split(required_split)
    dtype_name = _normalize_target_dtype(target_dtype)
    dtype = _torch_dtype(dtype_name)
    reports = []
    evidence_cases = []
    candidate_state_sha256 = None
    actual_arithmetic = True
    context_evidence_present: bool | None = None
    for case in ordered_cases:
        if not isinstance(case, Lfm2CpcfV4EvaluationCase):
            raise Lfm2CpcfShortConvV4Error(
                "evaluation cases must be Lfm2CpcfV4EvaluationCase instances."
            )
        if case.split != split:
            raise Lfm2CpcfShortConvV4Error(
                f"target-dtype evaluation requires {split!r} cases only."
            )
        if not case.verify_authentication():
            raise Lfm2CpcfShortConvV4Error(
                f"evaluation case {case.case_id!r} changed after authentication."
            )
        if candidate_state_sha256 is None:
            candidate_state_sha256 = case.candidate_state_sha256
        elif candidate_state_sha256 != case.candidate_state_sha256:
            raise Lfm2CpcfShortConvV4Error(
                "one evaluation report must describe one candidate state."
            )
        raw_prediction = case.prediction
        prediction = _finite_tensor(
            case.prediction,
            f"{case.case_id}.prediction",
            minimum_dimensions=1,
        )
        target = _finite_tensor(
            case.target,
            f"{case.case_id}.target",
            minimum_dimensions=1,
        )
        if tuple(prediction.shape) != tuple(target.shape):
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id} prediction and target shapes differ."
            )
        prediction = prediction.reshape(-1, int(prediction.shape[-1]))
        target = target.reshape(-1, int(target.shape[-1]))
        if case.valid_rows is not None:
            mask = case.valid_rows
            if (
                not isinstance(mask, torch.Tensor)
                or mask.layout != torch.strided
                or mask.device.type == "meta"
                or int(mask.numel()) != int(prediction.shape[0])
            ):
                raise Lfm2CpcfShortConvV4Error(
                    f"{case.case_id}.valid_rows has the wrong row count."
                )
            mask = mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
            prediction = prediction[mask]
            target = target[mask]
        if int(prediction.shape[0]) == 0:
            raise Lfm2CpcfShortConvV4Error(
                f"{case.case_id} contains no valid evaluation rows."
            )
        has_context = case.context_prediction is not None
        if context_evidence_present is None:
            context_evidence_present = has_context
        elif context_evidence_present != has_context:
            raise Lfm2CpcfShortConvV4Error(
                "one evaluation report cannot mix cases with and without "
                "context-state evidence."
            )
        raw_context_prediction = case.context_prediction
        context_prediction = None
        context_target = None
        materialized_context = None
        if has_context:
            context_prediction = _finite_tensor(
                case.context_prediction,
                f"{case.case_id}.context_prediction",
                minimum_dimensions=1,
            )
            context_target = _finite_tensor(
                case.context_target,
                f"{case.case_id}.context_target",
                minimum_dimensions=1,
            )
            if tuple(context_prediction.shape) != tuple(context_target.shape):
                raise Lfm2CpcfShortConvV4Error(
                    f"{case.case_id} context prediction and target shapes differ."
                )
            context_prediction = context_prediction.reshape(
                -1,
                int(context_prediction.shape[-1]),
            )
            context_target = context_target.reshape(
                -1,
                int(context_target.shape[-1]),
            )
            materialized_context = (
                raw_context_prediction.detach()
                .to(device="cpu", dtype=dtype)
                .to(torch.float64)
                .reshape(-1, int(raw_context_prediction.shape[-1]))
            )
            if case.valid_context_rows is not None:
                context_mask = case.valid_context_rows
                if (
                    not isinstance(context_mask, torch.Tensor)
                    or context_mask.layout != torch.strided
                    or context_mask.device.type == "meta"
                    or int(context_mask.numel()) != int(context_prediction.shape[0])
                ):
                    raise Lfm2CpcfShortConvV4Error(
                        f"{case.case_id}.valid_context_rows has the wrong row count."
                    )
                context_mask = (
                    context_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
                )
                context_prediction = context_prediction[context_mask]
                context_target = context_target[context_mask]
                materialized_context = materialized_context[context_mask]
            if int(context_prediction.shape[0]) == 0:
                raise Lfm2CpcfShortConvV4Error(
                    f"{case.case_id} contains no valid context evaluation rows."
                )
        arithmetic = case.arithmetic_report
        case_actual_arithmetic = (
            arithmetic is not None
            and _verify_sealed_report(arithmetic)
            and arithmetic.get("parameter_storage_dtype") == dtype_name
            and arithmetic.get("intermediate_arithmetic_dtype") == dtype_name
            and arithmetic.get("parameters_materialized_before_replay") is True
            and arithmetic.get("intermediates_computed_at_declared_dtype") is True
            and arithmetic.get("final_output_cast_only") is False
            and _dtype_name(raw_prediction) == dtype_name
            and arithmetic.get("evaluated_prediction_sha256")
            == _tensor_sha256(
                raw_prediction,
                f"{case.case_id}.evaluated_prediction",
            )
            and (
                not has_context
                or (
                    _dtype_name(raw_context_prediction) == dtype_name
                    and arithmetic.get("evaluated_context_prediction_sha256")
                    == _tensor_sha256(
                        raw_context_prediction,
                        f"{case.case_id}.evaluated_context_prediction",
                    )
                )
            )
        )
        actual_arithmetic = actual_arithmetic and case_actual_arithmetic
        materialized = (
            raw_prediction.detach().to(device="cpu", dtype=dtype).to(torch.float64)
        )
        materialized = materialized.reshape(
            -1,
            int(materialized.shape[-1]),
        )
        if case.valid_rows is not None:
            materialized = materialized[case.valid_rows.reshape(-1)]
        difference = materialized - target
        context_difference = (
            None
            if materialized_context is None
            else materialized_context - context_target
        )
        evidence_cases.append(case.evidence_body())
        reports.append(
            {
                "case_id": case.case_id,
                "rows": int(prediction.shape[0]),
                "width": int(prediction.shape[1]),
                "relative_rmse": _relative_rmse(materialized, target),
                "cosine_similarity": _cosine_similarity(
                    materialized,
                    target,
                ),
                "maximum_absolute_error": float(difference.abs().max().item()),
                "mean_absolute_error": float(difference.abs().mean().item()),
                "target_dtype_drift_relative_rmse": _relative_rmse(
                    materialized,
                    prediction,
                ),
                "fp64_prediction_sha256": _tensor_sha256(
                    prediction,
                    f"{case.case_id}.fp64_prediction",
                ),
                "materialized_prediction_sha256": _tensor_sha256(
                    materialized,
                    f"{case.case_id}.materialized_prediction",
                ),
                "target_sha256": _tensor_sha256(
                    target,
                    f"{case.case_id}.evaluation_target",
                ),
                "valid_rows_sha256": (
                    None
                    if case.valid_rows is None
                    else _tensor_sha256(
                        case.valid_rows,
                        f"{case.case_id}.evaluation_valid_rows",
                    )
                ),
                "projection_case_authentication_sha256": (
                    case.projection_case_authentication_sha256 or None
                ),
                "context_rows": (
                    None
                    if context_prediction is None
                    else int(context_prediction.shape[0])
                ),
                "context_width": (
                    None
                    if context_prediction is None
                    else int(context_prediction.shape[1])
                ),
                "context_relative_rmse": (
                    None
                    if materialized_context is None
                    else _relative_rmse(materialized_context, context_target)
                ),
                "context_cosine_similarity": (
                    None
                    if materialized_context is None
                    else _cosine_similarity(materialized_context, context_target)
                ),
                "context_maximum_absolute_error": (
                    None
                    if context_difference is None
                    else float(context_difference.abs().max().item())
                ),
                "context_target_dtype_drift_relative_rmse": (
                    None
                    if materialized_context is None
                    else _relative_rmse(
                        materialized_context,
                        context_prediction,
                    )
                ),
                "context_target_sha256": (
                    None
                    if context_target is None
                    else _tensor_sha256(
                        context_target,
                        f"{case.case_id}.evaluation_context_target",
                    )
                ),
                "valid_context_rows_sha256": (
                    None
                    if case.valid_context_rows is None
                    else _tensor_sha256(
                        case.valid_context_rows,
                        f"{case.case_id}.evaluation_valid_context_rows",
                    )
                ),
                "case_authentication_sha256": case.authentication_sha256,
                "candidate_state_sha256": case.candidate_state_sha256,
                "actual_parameter_and_intermediate_target_dtype_replay": (
                    case_actual_arithmetic
                ),
                "arithmetic_report_sha256": (
                    None if arithmetic is None else arithmetic.get("report_sha256")
                ),
            }
        )
    evidence_sha256 = _sha256(
        {
            "split": split,
            "cases": evidence_cases,
        }
    )
    policy_digest = _sha256_text(policy_sha256, "policy_sha256")
    context_summary = {
        "context_evidence_present": bool(context_evidence_present),
        "mean_case_context_relative_rmse": (
            None
            if not context_evidence_present
            else sum(float(value["context_relative_rmse"]) for value in reports)
            / len(reports)
        ),
        "maximum_case_context_relative_rmse": (
            None
            if not context_evidence_present
            else max(float(value["context_relative_rmse"]) for value in reports)
        ),
        "mean_case_context_cosine_similarity": (
            None
            if not context_evidence_present
            else sum(float(value["context_cosine_similarity"]) for value in reports)
            / len(reports)
        ),
        "minimum_case_context_cosine_similarity": (
            None
            if not context_evidence_present
            else min(float(value["context_cosine_similarity"]) for value in reports)
        ),
        "maximum_case_context_absolute_error": (
            None
            if not context_evidence_present
            else max(
                float(value["context_maximum_absolute_error"]) for value in reports
            )
        ),
        "mean_case_context_target_dtype_drift_relative_rmse": (
            None
            if not context_evidence_present
            else sum(
                float(value["context_target_dtype_drift_relative_rmse"])
                for value in reports
            )
            / len(reports)
        ),
    }
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "target_dtype_evaluation",
        "policy_sha256": policy_digest,
        "split": split,
        "target_dtype": dtype_name,
        "case_balanced": True,
        "case_count": len(reports),
        "cases": reports,
        "evidence_sha256": evidence_sha256,
        "candidate_state_sha256": candidate_state_sha256,
        "mean_case_relative_rmse": sum(
            float(value["relative_rmse"]) for value in reports
        )
        / len(reports),
        "maximum_case_relative_rmse": max(
            float(value["relative_rmse"]) for value in reports
        ),
        "mean_case_cosine_similarity": sum(
            float(value["cosine_similarity"]) for value in reports
        )
        / len(reports),
        "minimum_case_cosine_similarity": min(
            float(value["cosine_similarity"]) for value in reports
        ),
        "maximum_case_absolute_error": max(
            float(value["maximum_absolute_error"]) for value in reports
        ),
        "mean_case_target_dtype_drift_relative_rmse": sum(
            float(value["target_dtype_drift_relative_rmse"]) for value in reports
        )
        / len(reports),
        **context_summary,
        "actual_parameter_and_intermediate_target_dtype_replay": actual_arithmetic,
        "final_output_materialization_only": not actual_arithmetic,
        "selection_used_for_parameter_fitting": False,
        "validation_used_for_parameter_fitting": False,
    }
    return _seal_report(body)


def evaluate_candidate_state_target_dtype(
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    in_proj_weight: Any,
    in_proj_bias: Any | None,
    fir_kernel: Any,
    fir_bias: Any | None,
    writer_weight: Any,
    writer_bias: Any | None,
    required_split: str,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> Mapping[str, Any]:
    """Materialize a complete candidate state and replay every intermediate."""

    if not isinstance(policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    if not policy.verify_authentication():
        raise Lfm2CpcfShortConvV4Error("solver policy failed authentication.")
    split = _normalize_split(required_split)
    dtype = _torch_dtype(policy.target_dtype)
    state = {
        "in_proj_weight": _finite_tensor(
            in_proj_weight,
            "in_proj_weight",
            minimum_dimensions=2,
            maximum_dimensions=2,
        ).to(dtype=dtype),
        "in_proj_bias": (
            None
            if in_proj_bias is None
            else _finite_tensor(
                in_proj_bias,
                "in_proj_bias",
                minimum_dimensions=1,
                maximum_dimensions=1,
            ).to(dtype=dtype)
        ),
        "fir_kernel": _finite_tensor(
            fir_kernel,
            "fir_kernel",
            minimum_dimensions=2,
            maximum_dimensions=2,
        ).to(dtype=dtype),
        "fir_bias": (
            None
            if fir_bias is None
            else _finite_tensor(
                fir_bias,
                "fir_bias",
                minimum_dimensions=1,
                maximum_dimensions=1,
            ).to(dtype=dtype)
        ),
        "writer_weight": _finite_tensor(
            writer_weight,
            "writer_weight",
            minimum_dimensions=2,
            maximum_dimensions=2,
        ).to(dtype=dtype),
        "writer_bias": (
            None
            if writer_bias is None
            else _finite_tensor(
                writer_bias,
                "writer_bias",
                minimum_dimensions=1,
                maximum_dimensions=1,
            ).to(dtype=dtype)
        ),
    }
    candidate_state_sha256 = _state_sha256(
        state,
        "target_dtype_candidate",
    )
    ordered_cases = _canonical_unique_cases(cases, "candidate target-dtype replay")
    evaluation_cases = []
    for case in ordered_cases:
        (
            child,
            desired_context,
            desired_write,
            prefix_mask,
            suffix_mask,
        ) = _projection_case_values(case, required_split=split)
        replay = replay_sampled_cp_volterra(
            child,
            in_proj_weight=state["in_proj_weight"],
            in_proj_bias=state["in_proj_bias"],
            fir_kernel=state["fir_kernel"],
            fir_bias=state["fir_bias"],
            writer_weight=state["writer_weight"],
            writer_bias=state["writer_bias"],
            target_dtype=policy.target_dtype,
            valid_prefix_mask=prefix_mask,
        )
        prediction = replay.write_history[:, case.suffix_start :, :].contiguous()
        arithmetic_body = dict(replay.arithmetic_report)
        arithmetic_body.pop("report_sha256", None)
        arithmetic_body.update(
            {
                "projection_case_authentication_sha256": (case.authentication_sha256),
                "evaluated_suffix_start": case.suffix_start,
                "evaluated_prediction_sha256": _tensor_sha256(
                    prediction,
                    f"{case.case_id}.evaluated_prediction",
                ),
                "evaluated_context_prediction_sha256": _tensor_sha256(
                    replay.context_history,
                    f"{case.case_id}.evaluated_context_prediction",
                ),
            }
        )
        evaluation_arithmetic = _seal_report(arithmetic_body)
        evaluation_cases.append(
            Lfm2CpcfV4EvaluationCase(
                case_id=case.case_id,
                split=case.split,
                prediction=prediction,
                target=desired_write,
                valid_rows=suffix_mask,
                candidate_state_sha256=candidate_state_sha256,
                projection_case_authentication_sha256=(case.authentication_sha256),
                context_prediction=replay.context_history,
                context_target=desired_context,
                valid_context_rows=prefix_mask,
                arithmetic_report=evaluation_arithmetic,
            )
        )
    return evaluate_target_dtype_predictions(
        evaluation_cases,
        target_dtype=policy.target_dtype,
        required_split=split,
        policy_sha256=policy.authentication_sha256,
    )


def rank_short_conv_v4_selection_candidates(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    policy_sha256: str,
) -> Mapping[str, Any]:
    """Rank target-dtype reports using selection evidence and nothing else."""

    if not candidates:
        raise Lfm2CpcfShortConvV4Error(
            "selection ranking requires at least one candidate."
        )
    policy_digest = _sha256_text(policy_sha256, "policy_sha256")
    normalized = []
    evidence_contract = None
    for candidate_id, report in sorted(candidates.items()):
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise Lfm2CpcfShortConvV4Error(
                "selection candidate ids must be nonempty text."
            )
        if not isinstance(report, Mapping):
            raise Lfm2CpcfShortConvV4Error(
                "selection candidate reports must be mappings."
            )
        if report.get("kind") != "target_dtype_evaluation":
            raise Lfm2CpcfShortConvV4Error(
                "selection candidates require target-dtype evaluation reports."
            )
        if report.get("split") != "selection":
            raise Lfm2CpcfShortConvV4Error(
                "candidate ranking accepts selection reports only."
            )
        if report.get("policy_sha256") != policy_digest:
            raise Lfm2CpcfShortConvV4Error(
                "selection report policy authentication differs."
            )
        body = dict(report)
        report_sha = body.pop("report_sha256", None)
        if report_sha != _sha256(body):
            raise Lfm2CpcfShortConvV4Error("selection report authentication failed.")
        case_reports = tuple(report.get("cases", ()))
        if (
            report.get("context_evidence_present") is not True
            or not case_reports
            or any(
                not isinstance(value, Mapping)
                or not isinstance(
                    value.get("projection_case_authentication_sha256"),
                    str,
                )
                for value in case_reports
            )
        ):
            raise Lfm2CpcfShortConvV4Error(
                "selection candidates require full authenticated projection and "
                "context-state evidence."
            )
        for value in case_reports:
            _sha256_text(
                value.get("projection_case_authentication_sha256"),
                "projection_case_authentication_sha256",
            )
        contract = (
            report.get("target_dtype"),
            report.get("evidence_sha256"),
            tuple(
                (
                    value.get("case_id"),
                    value.get("rows"),
                    value.get("width"),
                    value.get("target_sha256"),
                    value.get("valid_rows_sha256"),
                    value.get("projection_case_authentication_sha256"),
                    value.get("context_rows"),
                    value.get("context_width"),
                    value.get("context_target_sha256"),
                    value.get("valid_context_rows_sha256"),
                )
                for value in case_reports
            ),
        )
        if evidence_contract is None:
            evidence_contract = contract
        elif evidence_contract != contract:
            raise Lfm2CpcfShortConvV4Error(
                "selection candidates do not share identical evidence and dtype."
            )
        candidate_state_sha256 = _sha256_text(
            report.get("candidate_state_sha256"),
            "candidate_state_sha256",
        )
        if (
            report.get("actual_parameter_and_intermediate_target_dtype_replay")
            is not True
        ):
            raise Lfm2CpcfShortConvV4Error(
                "selection candidates require actual parameter-and-intermediate "
                "target-dtype replay."
            )
        score = (
            float(report["maximum_case_relative_rmse"]),
            float(report["maximum_case_context_relative_rmse"]),
            -float(report["minimum_case_cosine_similarity"]),
            -float(report["minimum_case_context_cosine_similarity"]),
            float(report["mean_case_relative_rmse"]),
            float(report["mean_case_context_relative_rmse"]),
            -float(report["mean_case_cosine_similarity"]),
            -float(report["mean_case_context_cosine_similarity"]),
            float(report["mean_case_target_dtype_drift_relative_rmse"]),
            float(report["mean_case_context_target_dtype_drift_relative_rmse"]),
            float(report["maximum_case_absolute_error"]),
            float(report["maximum_case_context_absolute_error"]),
            candidate_id,
        )
        if not all(math.isfinite(value) for value in score[:-1]):
            raise Lfm2CpcfShortConvV4Error("selection ranking metrics must be finite.")
        normalized.append(
            {
                "candidate_id": candidate_id,
                "score": score,
                "evaluation_report_sha256": report_sha,
                "candidate_state_sha256": candidate_state_sha256,
                "actual_target_dtype_replay": report.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True,
            }
        )
    normalized.sort(key=lambda value: value["score"])
    ranking = [
        {
            "rank": index + 1,
            "candidate_id": value["candidate_id"],
            "selection_score": [
                float(item) if not isinstance(item, str) else item
                for item in value["score"]
            ],
            "evaluation_report_sha256": value["evaluation_report_sha256"],
            "candidate_state_sha256": value["candidate_state_sha256"],
            "actual_target_dtype_replay": value["actual_target_dtype_replay"],
        }
        for index, value in enumerate(normalized)
    ]
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "selection_only_candidate_ranking",
        "policy_sha256": policy_digest,
        "ranking": ranking,
        "selected_candidate_id": ranking[0]["candidate_id"],
        "fit_metrics_consumed": 0,
        "validation_metrics_consumed": 0,
        "selection_reports_consumed": len(ranking),
        "evidence_sha256": evidence_contract[1],
        "identical_candidate_evidence_contract": True,
        "full_projection_evidence_bound": True,
        "context_state_evidence_ranked": True,
        "worst_case_metrics_ranked_before_means": True,
        "all_candidates_used_actual_target_dtype_replay": all(
            value["actual_target_dtype_replay"] for value in normalized
        ),
        "validation_isolation": True,
    }
    return _seal_report(body)


def select_ridge_alpha_on_selection(
    *,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
    build_selection_evaluation: Callable[[float], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Evaluate the authenticated alpha grid and rank selection receipts."""

    if not isinstance(policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    reports = {
        f"ridge-alpha-{index:02d}-{alpha:.17g}": (build_selection_evaluation(alpha))
        for index, alpha in enumerate(policy.ridge_alphas)
    }
    ranking = rank_short_conv_v4_selection_candidates(
        reports,
        policy_sha256=policy.authentication_sha256,
    )
    selected_id = str(ranking["selected_candidate_id"])
    selected_index = int(selected_id.split("-", 2)[2].split("-", 1)[0])
    body = dict(ranking)
    body.pop("report_sha256", None)
    body.update(
        {
            "kind": "authenticated_ridge_alpha_selection",
            "complete_authenticated_grid": True,
            "ridge_alphas": list(policy.ridge_alphas),
            "selected_alpha": policy.ridge_alphas[selected_index],
        }
    )
    return _seal_report(body)


__all__ = [
    "LFM2_CPCF_SHORT_CONV_V4_SCHEMA",
    "Lfm2CpcfShortConvV4Error",
    "Lfm2CpcfShortConvV4SolverPolicy",
    "Lfm2CpcfV4BasisCase",
    "Lfm2CpcfV4BoundedStep",
    "Lfm2CpcfV4ChannelBasis",
    "Lfm2CpcfV4EvaluationCase",
    "Lfm2CpcfV4ProjectionCase",
    "Lfm2CpcfV4RidgeCase",
    "Lfm2CpcfV4RidgeResult",
    "Lfm2CpcfV4TransportedPriors",
    "Lfm2CpcfV4VariableProjectionResult",
    "Lfm2CpcfV4VolterraReplay",
    "affine_penalty_mask",
    "bound_prior_correction",
    "build_signed_endpoint_aware_channel_basis",
    "case_balanced_trace_scaled_prior_ridge",
    "causal_depthwise_context",
    "collapse_endpoint_context_sufficient_statistics",
    "evaluate_candidate_state_target_dtype",
    "evaluate_target_dtype_predictions",
    "fir_penalty_mask",
    "fixed_variable_projection_schedule",
    "linearized_bcx_affine_design",
    "rank_short_conv_v4_selection_candidates",
    "replay_sampled_cp_volterra",
    "run_fixed_sampled_cp_variable_projection",
    "sampled_cp_volterra_objective",
    "select_ridge_alpha_on_selection",
    "solve_bounded_affine_projection",
    "solve_bounded_fir_projection",
    "solve_bounded_linearized_bcx_projection",
    "solve_bounded_writer_projection",
    "transport_parent_short_conv_priors",
]
