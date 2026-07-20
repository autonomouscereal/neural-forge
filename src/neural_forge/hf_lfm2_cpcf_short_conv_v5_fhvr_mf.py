"""Future-Hankel Volterra realization with bounded matrix-free solves.

This module is an optional fit-only numerical producer for the existing LFM2
short-convolution v5 frontier seam.  It does not install a model, inspect
selection or validation evidence, or execute an optimizer.  The implementation
has four algebraic stages:

* derive a future-observable parent injection basis from streamed, case-balanced
  causal sufficient statistics;
* realize the child B/x product with fixed-schedule conditional ridge solves;
* realize C and every FIR lag with the same streamed conditional algebra; and
* solve one dense target-native writer from fit suffixes.

No row-by-row kernel, N-by-N matrix, or per-channel feature Gram is
materialized.  Only the six ordinary dense short-conv tensors survive the
compile.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    Lfm2CpcfShortConvNumericalFrontierRequest,
    Lfm2CpcfShortConvNumericalRawCandidate,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_residual_subspace_bcx import (
    automatic_residual_subspace_rank,
    causal_horizon_balanced_row_weights,
)


LFM2_CPCF_SHORT_CONV_V5_FHVR_MF_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-fhvr-mf-v1"
)
FHVR_MF_CANDIDATE_ID = "v5-fhvr-mf-future-state-realization"
FHVR_MF_CANDIDATE_KIND = (
    "fit_only_future_hankel_volterra_matrix_free_realization"
)
_IMPLEMENTATION = "neural_forge.hf_lfm2_cpcf_short_conv_v5_fhvr_mf"
_ZERO_EXECUTION = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "generation_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "autograd_graphs": 0,
}


class Lfm2CpcfFhvrMfError(ValueError):
    """FHVR-MF input, numerical, or authentication failure."""


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise Lfm2CpcfFhvrMfError("FHVR-MF requires PyTorch.") from exc


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["report_sha256"] = _sha256(value)
    return value


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfFhvrMfError(f"{name} must be a materialized dense tensor.")
    detached = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "name": name,
        "dtype": str(detached.dtype).removeprefix("torch."),
        "shape": [int(size) for size in detached.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfFhvrMfError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfFhvrMfError(f"{name} must be finite.") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise Lfm2CpcfFhvrMfError(f"{name} is outside its supported range.")
    return result


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        raise Lfm2CpcfFhvrMfError(f"{name} must be a supported positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class Lfm2CpcfFhvrMfPolicy:
    """Authenticated fixed numerical schedule for FHVR-MF."""

    condition_cap: float = 1_000.0
    pcg_iterations: int = 16
    factor_sweeps: int = 2
    row_chunk_size: int = 2_048
    channel_chunk_size: int = 64
    behavior_rank_cap: int = 64
    rank_relative_tolerance: float = 1e-7
    absolute_ridge_floor: float = 1e-8
    tail_tau_floor: float = 4.0
    backend: str = "auto"
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        cap = _finite_float(self.condition_cap, "condition_cap", minimum=1.0)
        if cap not in {1_000.0, 10_000.0}:
            raise Lfm2CpcfFhvrMfError(
                "condition_cap must be one of the authenticated v5 caps."
            )
        object.__setattr__(self, "condition_cap", cap)
        for name, maximum in (
            ("pcg_iterations", 256),
            ("factor_sweeps", 8),
            ("row_chunk_size", None),
            ("channel_chunk_size", None),
            ("behavior_rank_cap", None),
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum=maximum),
            )
        for name in (
            "rank_relative_tolerance",
            "absolute_ridge_floor",
            "tail_tau_floor",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=0.0),
            )
        if (
            self.rank_relative_tolerance == 0.0
            or self.absolute_ridge_floor == 0.0
            or self.tail_tau_floor == 0.0
        ):
            raise Lfm2CpcfFhvrMfError(
                "rank tolerance, ridge floor, and tail tau must be positive."
            )
        backend = str(self.backend).strip().lower()
        if backend not in {"auto", "cpu", "cuda"}:
            raise Lfm2CpcfFhvrMfError("backend must be auto, cpu, or cuda.")
        object.__setattr__(self, "backend", backend)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfFhvrMfError("policy authentication_sha256 is invalid.")
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_FHVR_MF_SCHEMA,
            "kind": "fhvr_mf_fixed_numerical_policy",
            "condition_cap": self.condition_cap,
            "pcg_iterations": self.pcg_iterations,
            "factor_sweeps": self.factor_sweeps,
            "row_chunk_size": self.row_chunk_size,
            "channel_chunk_size": self.channel_chunk_size,
            "behavior_rank_cap": self.behavior_rank_cap,
            "rank_relative_tolerance": self.rank_relative_tolerance,
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "tail_tau_floor": self.tail_tau_floor,
            "backend": self.backend,
            "fixed_iteration_schedule": True,
            "early_stopping": False,
            "selection_inputs_accessible": False,
            "validation_inputs_accessible": False,
            "execution": {
                **_ZERO_EXECUTION,
                "closed_form_iterative_linear_solves": True,
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
class Lfm2CpcfFhvrMfConditionalResult:
    """One fixed-PCG conditional factor solve plus its allocation audit."""

    weight: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            not isinstance(self.weight, torch.Tensor)
            or self.weight.layout != torch.strided
            or self.weight.ndim != 2
            or not bool(torch.isfinite(self.weight).all().item())
        ):
            raise Lfm2CpcfFhvrMfError(
                "conditional result weight must be one finite dense matrix."
            )
        body = dict(self.diagnostics)
        supplied = body.pop("report_sha256", None)
        if supplied != _sha256(body):
            raise Lfm2CpcfFhvrMfError(
                "conditional result diagnostics are not canonically sealed."
            )
        object.__setattr__(self, "weight", self.weight.detach().clone().contiguous())
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )

    def verify_authentication(self) -> bool:
        body = dict(self.diagnostics)
        supplied = body.pop("report_sha256", None)
        return bool(
            supplied == _sha256(body)
            and self.diagnostics.get("solution_weight_sha256")
            == _tensor_sha256(self.weight, "conditional.solution")
        )


@dataclass(slots=True)
class _WorkspaceAudit:
    row_cap: int
    channel_cap: int
    maximum_stream_rows: int = 0
    maximum_channel_chunk: int = 0
    maximum_temporary_elements: int = 0
    streamed_row_blocks: int = 0

    def rows(self, rows: int, columns: int) -> None:
        if rows > self.row_cap:
            raise Lfm2CpcfFhvrMfError(
                "A streamed row temporary exceeded the authenticated row cap."
            )
        self.maximum_stream_rows = max(self.maximum_stream_rows, rows)
        self.maximum_temporary_elements = max(
            self.maximum_temporary_elements,
            rows * columns,
        )
        self.streamed_row_blocks += 1

    def channels(self, channels: int, feature_width: int) -> None:
        if channels > self.channel_cap:
            raise Lfm2CpcfFhvrMfError(
                "A channel solve exceeded the authenticated channel cap."
            )
        self.maximum_channel_chunk = max(self.maximum_channel_chunk, channels)
        self.maximum_temporary_elements = max(
            self.maximum_temporary_elements,
            channels * feature_width,
        )

    def report(self) -> dict[str, Any]:
        return {
            "row_chunk_cap": self.row_cap,
            "channel_chunk_cap": self.channel_cap,
            "maximum_stream_rows": self.maximum_stream_rows,
            "maximum_channel_chunk": self.maximum_channel_chunk,
            "maximum_temporary_elements": self.maximum_temporary_elements,
            "streamed_row_blocks": self.streamed_row_blocks,
            "row_kernel_materialized": False,
            "n_by_n_matrix_materialized": False,
            "normal_matrix_materialized": False,
            "per_channel_feature_gram_materialized": False,
            "temporary_row_bound_independent_of_total_fit_rows": True,
        }


@dataclass(frozen=True, slots=True)
class _EvidenceCase:
    case_id: str
    case_authentication_sha256: str
    features: Any
    parent_input: Any
    parent_injection: Any
    desired_context: Any
    desired_write: Any
    prefix_weight: Any
    suffix_weight: Any
    valid_mask: Any


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    case_id: str
    case_authentication_sha256: str
    features: Any
    desired_state: Any
    desired_context: Any
    desired_write: Any
    prefix_weight: Any
    suffix_weight: Any
    valid_mask: Any


def _resolve_device(policy: Lfm2CpcfFhvrMfPolicy) -> tuple[Any, Any, str]:
    torch = _torch()
    cuda = bool(torch.cuda.is_available())
    if policy.backend == "cuda" and not cuda:
        raise Lfm2CpcfFhvrMfError("backend='cuda' is unavailable.")
    use_cuda = policy.backend == "cuda" or (policy.backend == "auto" and cuda)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    return device, dtype, str(device)


@contextmanager
def _deterministic_compute_scope(device: Any):
    torch = _torch()
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = bool(
        getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
    )
    matmul = getattr(torch.backends.cuda, "matmul", None)
    cudnn = getattr(torch.backends, "cudnn", None)
    old_matmul_tf32 = getattr(matmul, "allow_tf32", None)
    old_cudnn_tf32 = getattr(cudnn, "allow_tf32", None)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if device.type == "cuda":
            if matmul is not None and old_matmul_tf32 is not None:
                matmul.allow_tf32 = False
            if cudnn is not None and old_cudnn_tf32 is not None:
                cudnn.allow_tf32 = False
        yield
    finally:
        if matmul is not None and old_matmul_tf32 is not None:
            matmul.allow_tf32 = old_matmul_tf32
        if cudnn is not None and old_cudnn_tf32 is not None:
            cudnn.allow_tf32 = old_cudnn_tf32
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)


def _chunks(rows: int, cap: int) -> Iterator[slice]:
    for start in range(0, rows, cap):
        yield slice(start, min(rows, start + cap))


def _trace_normalized(value: Any) -> tuple[Any, float]:
    torch = _torch()
    symmetric = 0.5 * (value + value.T)
    trace = float(torch.trace(symmetric).item())
    if trace <= float(torch.finfo(value.dtype).eps):
        return torch.zeros_like(symmetric), trace
    return symmetric / trace, trace


def _canonicalize_columns(value: Any) -> Any:
    torch = _torch()
    columns = int(value.shape[1])
    pivots = value.abs().argmax(dim=0)
    pivot_values = value[pivots, torch.arange(columns, device=value.device)]
    signs = torch.where(
        pivot_values < 0.0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return (value * signs).contiguous()


def _top_eigenvectors(value: Any, rank: int) -> tuple[Any, Any]:
    torch = _torch()
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    order = torch.arange(
        int(eigenvalues.shape[0]) - 1,
        int(eigenvalues.shape[0]) - rank - 1,
        -1,
        device=value.device,
    )
    basis = _canonicalize_columns(eigenvectors.index_select(1, order))
    return basis, eigenvalues.index_select(0, order).contiguous()


def _conditioned_dense_solution(
    gram: Any,
    right: Any,
    prior: Any,
    *,
    condition_cap: float,
    ridge_floor: float,
    rank_tolerance: float,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    symmetric = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp(min=0.0)
    maximum = float(eigenvalues.max().item())
    minimum = float(eigenvalues.min().item())
    ridge = max(
        ridge_floor,
        0.0
        if maximum == 0.0
        else (maximum - condition_cap * minimum) / (condition_cap - 1.0),
    )
    residual_right = right - symmetric @ prior
    inverse = 1.0 / (eigenvalues + ridge)
    delta = eigenvectors @ (
        inverse.unsqueeze(-1) * (eigenvectors.T @ residual_right)
    )
    solution = (prior + delta).contiguous()
    threshold = maximum * rank_tolerance
    condition = 0.0 if maximum == 0.0 else (maximum + ridge) / (minimum + ridge)
    return solution, {
        "maximum_eigenvalue": maximum,
        "minimum_eigenvalue": minimum,
        "effective_ridge": ridge,
        "regularized_condition": condition,
        "condition_cap": condition_cap,
        "condition_cap_passed": condition <= condition_cap * (1.0 + 1e-6),
        "effective_rank": int((eigenvalues > threshold).sum().item()),
    }


def _validate_conditional_inputs(
    features: Any,
    multiplier: Any,
    target: Any,
    row_weight: Any,
    prior: Any,
) -> tuple[int, int, int]:
    torch = _torch()
    values = (features, multiplier, target, row_weight, prior)
    if any(
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
        for value in values
    ):
        raise Lfm2CpcfFhvrMfError(
            "conditional solve inputs must be finite dense tensors."
        )
    if (
        features.ndim != 2
        or multiplier.ndim != 2
        or target.ndim != 2
        or row_weight.ndim != 1
        or prior.ndim != 2
    ):
        raise Lfm2CpcfFhvrMfError("conditional solve tensor ranks are invalid.")
    rows, width = map(int, features.shape)
    hidden = int(multiplier.shape[1])
    if (
        rows <= 0
        or width <= 0
        or hidden <= 0
        or tuple(multiplier.shape) != (rows, hidden)
        or tuple(target.shape) != (rows, hidden)
        or tuple(row_weight.shape) != (rows,)
        or tuple(prior.shape) != (hidden, width)
        or bool((row_weight < 0.0).any().item())
        or float(row_weight.sum().item()) <= 0.0
    ):
        raise Lfm2CpcfFhvrMfError("conditional solve geometry is inconsistent.")
    return rows, width, hidden


def _conditional_solve(
    *,
    feature_width: int,
    hidden: int,
    prior: Any,
    chunk_factory: Callable[[slice], Iterator[tuple[Any, Any, Any, Any]]],
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    result = prior.clone()
    ridge_values: list[float] = []
    relative_residual_maxima = [0.0] * (policy.pcg_iterations + 1)
    breakdown_channels = 0
    total_rows = 0
    for channel_start in range(0, hidden, policy.channel_chunk_size):
        channel_stop = min(hidden, channel_start + policy.channel_chunk_size)
        channel = slice(channel_start, channel_stop)
        count = channel_stop - channel_start
        audit.channels(count, feature_width)
        prior_chunk = prior[channel]
        rhs = torch.zeros_like(prior_chunk)
        diagonal = torch.zeros_like(prior_chunk)
        upper = torch.zeros((count,), device=prior.device, dtype=prior.dtype)
        observed_rows = 0
        for features, multiplier, target, weights in chunk_factory(channel):
            rows = int(features.shape[0])
            audit.rows(rows, max(feature_width, count))
            observed_rows += rows
            base = features @ prior_chunk.T
            residual = target - multiplier * base
            coefficient = weights.unsqueeze(-1) * multiplier
            rhs = rhs + (coefficient * residual).T @ features
            squared_coefficient = weights.unsqueeze(-1) * multiplier.square()
            diagonal = diagonal + squared_coefficient.T @ features.square()
            upper = upper + (
                squared_coefficient
                * features.square().sum(dim=1, keepdim=True)
            ).sum(dim=0)
        total_rows = max(total_rows, observed_rows)
        ridge = torch.clamp(
            upper / (policy.condition_cap - 1.0),
            min=policy.absolute_ridge_floor,
        )
        ridge_values.extend(float(value) for value in ridge.tolist())
        preconditioner = (diagonal + ridge.unsqueeze(-1)).clamp(
            min=policy.absolute_ridge_floor
        )

        def normal(value: Any) -> Any:
            normal_value = ridge.unsqueeze(-1) * value
            for features, multiplier, _target, weights in chunk_factory(channel):
                rows = int(features.shape[0])
                audit.rows(rows, max(feature_width, count))
                effect = features @ value.T
                coefficient = (
                    weights.unsqueeze(-1) * multiplier.square() * effect
                )
                normal_value = normal_value + coefficient.T @ features
            return normal_value

        solution = torch.zeros_like(rhs)
        residual = rhs.clone()
        preconditioned = residual / preconditioner
        direction = preconditioned.clone()
        residual_preconditioned = (residual * preconditioned).sum(dim=1)
        initial_norm = torch.linalg.vector_norm(residual, dim=1)
        relative_residual_maxima[0] = max(
            relative_residual_maxima[0],
            0.0 if not bool((initial_norm > 0.0).any().item()) else 1.0,
        )
        active = initial_norm > 0.0
        tiny = torch.finfo(prior.dtype).tiny
        for iteration in range(policy.pcg_iterations):
            normal_direction = normal(direction)
            denominator = (direction * normal_direction).sum(dim=1)
            valid = active & torch.isfinite(denominator) & (denominator > tiny)
            alpha = torch.where(
                valid,
                residual_preconditioned / denominator.clamp(min=tiny),
                torch.zeros_like(denominator),
            )
            solution = solution + alpha.unsqueeze(-1) * direction
            residual = residual - alpha.unsqueeze(-1) * normal_direction
            next_preconditioned = residual / preconditioner
            next_residual_preconditioned = (
                residual * next_preconditioned
            ).sum(dim=1)
            beta = torch.where(
                valid & (residual_preconditioned.abs() > tiny),
                next_residual_preconditioned
                / residual_preconditioned.abs().clamp(min=tiny)
                * residual_preconditioned.sign(),
                torch.zeros_like(residual_preconditioned),
            )
            direction = next_preconditioned + beta.unsqueeze(-1) * direction
            active = valid
            residual_preconditioned = next_residual_preconditioned
            current_norm = torch.linalg.vector_norm(residual, dim=1)
            relative = torch.where(
                initial_norm > 0.0,
                current_norm / initial_norm.clamp(min=tiny),
                torch.zeros_like(current_norm),
            )
            relative_residual_maxima[iteration + 1] = max(
                relative_residual_maxima[iteration + 1],
                float(relative.max().item()),
            )
        breakdown_channels += int((~active & (initial_norm > 0.0)).sum().item())
        result[channel] = prior_chunk + solution
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfFhvrMfError("fixed PCG produced NaN or infinity.")
    report = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_FHVR_MF_SCHEMA,
        "kind": "bounded_matrix_free_conditional_factor_solve",
        "feature_width": feature_width,
        "output_channels": hidden,
        "fit_rows_visited_per_normal_pass": total_rows,
        "condition_cap": policy.condition_cap,
        "effective_ridge_minimum": min(ridge_values),
        "effective_ridge_maximum": max(ridge_values),
        "effective_ridge_by_channel": ridge_values,
        "pcg_iterations": policy.pcg_iterations,
        "fixed_iteration_schedule": True,
        "early_stopping": False,
        "maximum_relative_normal_residual_by_iteration": (
            relative_residual_maxima
        ),
        "breakdown_channel_count": breakdown_channels,
        "solution_weight_sha256": _tensor_sha256(
            result,
            "conditional.solution",
        ),
        "allocation_audit": audit.report(),
        "execution": {
            **_ZERO_EXECUTION,
            "closed_form_iterative_linear_solve": True,
        },
    }
    return result, _seal_report(report)


def solve_lfm2_cpcf_fhvr_mf_conditional_factor(
    features: Any,
    multiplier: Any,
    target: Any,
    row_weight: Any,
    prior: Any,
    *,
    policy: Lfm2CpcfFhvrMfPolicy | None = None,
) -> Lfm2CpcfFhvrMfConditionalResult:
    """Solve one bilinear factor conditionally without an N-by-N object."""

    solve_policy = Lfm2CpcfFhvrMfPolicy() if policy is None else policy
    if (
        not isinstance(solve_policy, Lfm2CpcfFhvrMfPolicy)
        or not solve_policy.verify_authentication()
    ):
        raise Lfm2CpcfFhvrMfError("conditional solve policy is invalid.")
    rows, width, hidden = _validate_conditional_inputs(
        features,
        multiplier,
        target,
        row_weight,
        prior,
    )
    if len({value.device for value in (features, multiplier, target, row_weight, prior)}) != 1:
        raise Lfm2CpcfFhvrMfError("conditional solve tensors must share a device.")
    if len({value.dtype for value in (features, multiplier, target, row_weight, prior)}) != 1:
        raise Lfm2CpcfFhvrMfError("conditional solve tensors must share a dtype.")
    audit = _WorkspaceAudit(
        row_cap=solve_policy.row_chunk_size,
        channel_cap=solve_policy.channel_chunk_size,
    )

    def chunks(channel: slice) -> Iterator[tuple[Any, Any, Any, Any]]:
        for row_slice in _chunks(rows, solve_policy.row_chunk_size):
            yield (
                features[row_slice],
                multiplier[row_slice, channel],
                target[row_slice, channel],
                row_weight[row_slice],
            )

    with _deterministic_compute_scope(features.device):
        weight, diagnostics = _conditional_solve(
            feature_width=width,
            hidden=hidden,
            prior=prior,
            chunk_factory=chunks,
            policy=solve_policy,
            audit=audit,
        )
    return Lfm2CpcfFhvrMfConditionalResult(
        weight=weight,
        diagnostics=diagnostics,
    )


def _prepare_evidence(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    context_encoder: Any,
    device: Any,
    dtype: Any,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[_EvidenceCase, ...]:
    torch = _torch()
    include_bias = request.target_templates["in_proj_bias"] is not None
    encoder = context_encoder.detach().to(device=device, dtype=dtype)
    case_count = len(request.cases)
    result = []
    for case in request.cases:
        mask = case.valid_prefix_mask.detach().to(device=device, dtype=torch.bool)
        child = case.actual_child_operator_input_history.detach().to(
            device=device,
            dtype=dtype,
        )
        child = child * mask.unsqueeze(-1).to(dtype)
        if include_bias:
            features = torch.cat(
                (
                    child,
                    torch.ones(
                        (*child.shape[:2], 1),
                        device=device,
                        dtype=dtype,
                    ),
                ),
                dim=-1,
            )
        else:
            features = child
        parent_input = case.parent_operator_input_history.detach().to(
            device=device,
            dtype=dtype,
        )
        parent_b = case.parent_b_history.detach().to(device=device, dtype=dtype)
        parent_x = case.parent_x_history.detach().to(device=device, dtype=dtype)
        parent_injection = (parent_b * parent_x) * mask.unsqueeze(-1).to(dtype)
        parent_context = request.parent_contexts[case.case_id].detach().to(
            device=device,
            dtype=dtype,
        )
        desired_context = (parent_context @ encoder) * mask.unsqueeze(-1).to(dtype)
        desired_write = torch.zeros(
            (*child.shape[:2], int(encoder.shape[1])),
            device=device,
            dtype=dtype,
        )
        suffix_value = (
            case.parent_post_operator_residual_target_space_suffix.detach().to(
                device=device,
                dtype=dtype,
            )
            - case.actual_child_pre_operator_residual_suffix.detach().to(
                device=device,
                dtype=dtype,
            )
        )
        desired_write[:, case.suffix_start :, :] = suffix_value
        prefix_weight = (
            causal_horizon_balanced_row_weights(
                mask,
                dtype=dtype,
            )
            / case_count
        )
        suffix_mask = torch.zeros_like(mask)
        suffix_mask[:, case.suffix_start :] = mask[:, case.suffix_start :]
        suffix_weight = (
            causal_horizon_balanced_row_weights(
                suffix_mask,
                dtype=dtype,
            )
            / case_count
        )
        audit.rows(
            min(int(features.numel() // features.shape[-1]), policy.row_chunk_size),
            int(features.shape[-1]),
        )
        result.append(
            _EvidenceCase(
                case_id=case.case_id,
                case_authentication_sha256=case.authentication_sha256,
                features=features.contiguous(),
                parent_input=parent_input.contiguous(),
                parent_injection=parent_injection.contiguous(),
                desired_context=desired_context.contiguous(),
                desired_write=desired_write.contiguous(),
                prefix_weight=prefix_weight.contiguous(),
                suffix_weight=suffix_weight.contiguous(),
                valid_mask=mask.contiguous(),
            )
        )
    return tuple(result)


def _streamed_dense_ridge(
    rows: Sequence[tuple[Any, Any, Any]],
    *,
    prior: Any,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    feature_width, output_width = map(int, prior.shape)
    gram = torch.zeros(
        (feature_width, feature_width),
        device=prior.device,
        dtype=prior.dtype,
    )
    right = torch.zeros(
        (feature_width, output_width),
        device=prior.device,
        dtype=prior.dtype,
    )
    total_rows = 0
    for design, target, weight in rows:
        flat_design = design.reshape(-1, feature_width)
        flat_target = target.reshape(-1, output_width)
        flat_weight = weight.reshape(-1)
        for row_slice in _chunks(int(flat_design.shape[0]), policy.row_chunk_size):
            design_chunk = flat_design[row_slice]
            target_chunk = flat_target[row_slice]
            weight_chunk = flat_weight[row_slice]
            audit.rows(int(design_chunk.shape[0]), max(feature_width, output_width))
            weighted = weight_chunk.unsqueeze(-1) * design_chunk
            gram = gram + design_chunk.T @ weighted
            right = right + weighted.T @ target_chunk
            total_rows += int(design_chunk.shape[0])
    solution, solve_report = _conditioned_dense_solution(
        gram,
        right,
        prior,
        condition_cap=policy.condition_cap,
        ridge_floor=policy.absolute_ridge_floor,
        rank_tolerance=policy.rank_relative_tolerance,
    )
    return solution, {
        **solve_report,
        "fit_rows_visited": total_rows,
        "feature_width": feature_width,
        "output_width": output_width,
    }


def _resize_fir_rows(source_fir: Any, target_length: int) -> Any:
    torch = _torch()
    source_length = int(source_fir.shape[1])
    if source_length == target_length:
        return source_fir.clone().contiguous()
    if target_length == 1:
        return source_fir.mean(dim=1, keepdim=True).contiguous()
    if source_length == 1:
        return source_fir.expand(-1, target_length).clone().contiguous()
    return (
        torch.nn.functional.interpolate(
            source_fir.unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=True,
        )
        .squeeze(0)
        .contiguous()
    )


def _initial_state(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    evidence: Sequence[_EvidenceCase],
    *,
    context_encoder: Any,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[dict[str, Any | None], dict[str, Any]]:
    torch = _torch()
    source = request.source_tensors
    hidden = int(request.report["target_hidden_size"])
    source_hidden = int(request.report["source_hidden_size"])
    feature_width = int(evidence[0].features.shape[-1])
    include_in_bias = request.target_templates["in_proj_bias"] is not None
    lift_prior = torch.zeros(
        (feature_width, source_hidden),
        device=evidence[0].features.device,
        dtype=evidence[0].features.dtype,
    )
    lift, lift_report = _streamed_dense_ridge(
        tuple(
            (
                case.features,
                case.parent_input,
                case.prefix_weight,
            )
            for case in evidence
        ),
        prior=lift_prior,
        policy=policy,
        audit=audit,
    )
    encoder = context_encoder.detach().to(
        device=lift.device,
        dtype=lift.dtype,
    )
    source_weight = source["source_in_proj_weight"].detach().to(
        device=lift.device,
        dtype=lift.dtype,
    ).reshape(3, source_hidden, source_hidden)
    source_bias = (
        torch.zeros(
            (3, source_hidden),
            device=lift.device,
            dtype=lift.dtype,
        )
        if source["source_in_proj_bias"] is None
        else source["source_in_proj_bias"].detach().to(
            device=lift.device,
            dtype=lift.dtype,
        ).reshape(3, source_hidden)
    )
    role_augmented = []
    for role in range(3):
        transported = (lift @ source_weight[role].T @ encoder).T
        if include_in_bias:
            transported[:, -1] = transported[:, -1] + source_bias[role] @ encoder
        role_augmented.append(transported.contiguous())
    in_proj_weight = torch.cat(
        tuple(role[:, :hidden] for role in role_augmented),
        dim=0,
    ).contiguous()
    in_proj_bias = (
        torch.cat(tuple(role[:, -1] for role in role_augmented)).contiguous()
        if include_in_bias
        else None
    )
    source_fir = source["source_fir_kernel"].detach().to(
        device=lift.device,
        dtype=lift.dtype,
    )
    resized_fir = _resize_fir_rows(
        source_fir,
        int(request.report["target_fir_length"]),
    )
    mixing = encoder.square()
    mixing = mixing / mixing.sum(dim=0, keepdim=True).clamp(
        min=torch.finfo(mixing.dtype).eps
    )
    fir_kernel = (mixing.T @ resized_fir).contiguous()
    fir_bias = None
    if request.target_templates["fir_bias"] is not None:
        source_fir_bias = (
            torch.zeros(
                (source_hidden,),
                device=lift.device,
                dtype=lift.dtype,
            )
            if source["source_fir_bias"] is None
            else source["source_fir_bias"].detach().to(
                device=lift.device,
                dtype=lift.dtype,
            )
        )
        fir_bias = (mixing.T @ source_fir_bias).contiguous()

    include_writer_bias = request.target_templates["writer_bias"] is not None
    writer_rows = []
    for case in evidence:
        design = case.desired_context
        if include_writer_bias:
            design = torch.cat(
                (
                    design,
                    torch.ones(
                        (*design.shape[:2], 1),
                        device=design.device,
                        dtype=design.dtype,
                    ),
                ),
                dim=-1,
            )
        writer_rows.append((design, case.desired_write, case.suffix_weight))
    writer_prior = torch.zeros(
        (hidden + int(include_writer_bias), hidden),
        device=lift.device,
        dtype=lift.dtype,
    )
    writer_augmented, writer_report = _streamed_dense_ridge(
        tuple(writer_rows),
        prior=writer_prior,
        policy=policy,
        audit=audit,
    )
    writer_weight = writer_augmented[:hidden].T.contiguous()
    writer_bias = (
        writer_augmented[-1].contiguous() if include_writer_bias else None
    )
    return {
        "in_proj_weight": in_proj_weight,
        "in_proj_bias": in_proj_bias,
        "fir_kernel": fir_kernel,
        "fir_bias": fir_bias,
        "writer_weight": writer_weight,
        "writer_bias": writer_bias,
    }, {
        "input_lift": lift_report,
        "writer_initialization": writer_report,
        "initial_state_uses_all_source_channels": bool(
            (mixing.abs().sum(dim=1) > 0.0).all().item()
        ),
    }


def _behavior_basis(
    evidence: Sequence[_EvidenceCase],
    *,
    writer_weight: Any,
    hidden: int,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    device = writer_weight.device
    dtype = writer_weight.dtype
    covariance = torch.zeros((hidden, hidden), device=device, dtype=dtype)
    endpoint_cross = torch.zeros_like(covariance)
    for case in evidence:
        context = case.desired_context.reshape(-1, hidden)
        write = case.desired_write.reshape(-1, hidden)
        prefix = case.prefix_weight.reshape(-1)
        suffix = case.suffix_weight.reshape(-1)
        for row_slice in _chunks(int(context.shape[0]), policy.row_chunk_size):
            context_chunk = context[row_slice]
            write_chunk = write[row_slice]
            prefix_chunk = prefix[row_slice]
            suffix_chunk = suffix[row_slice]
            audit.rows(int(context_chunk.shape[0]), hidden)
            covariance = covariance + context_chunk.T @ (
                prefix_chunk.unsqueeze(-1) * context_chunk
            )
            endpoint_cross = endpoint_cross + context_chunk.T @ (
                suffix_chunk.unsqueeze(-1) * write_chunk
            )
    components_with_trace = (
        _trace_normalized(covariance),
        _trace_normalized(endpoint_cross @ endpoint_cross.T),
        _trace_normalized(writer_weight.T @ writer_weight),
    )
    active = [value for value, trace in components_with_trace if trace > 0.0]
    aggregate = sum(active, torch.zeros_like(covariance)) / max(len(active), 1)
    rank = automatic_residual_subspace_rank(
        hidden,
        maximum_rank=min(hidden, policy.behavior_rank_cap),
    )
    basis, spectrum = _top_eigenvectors(aggregate, rank)
    return basis, {
        "behavior_rank": rank,
        "rank_rule": (
            "full_h_when_h_le_32_else_next_power_of_two_ceil_sqrt_h_capped"
        ),
        "component_traces": {
            "desired_context_covariance": components_with_trace[0][1],
            "fit_suffix_endpoint_cross_psd": components_with_trace[1][1],
            "initial_writer_gram": components_with_trace[2][1],
        },
        "active_component_count": len(active),
        "leading_spectrum": [float(value) for value in spectrum.tolist()],
        "basis_sha256": _tensor_sha256(basis, "fhvr.behavior_basis"),
    }


def _future_state_encoder(
    evidence: Sequence[_EvidenceCase],
    *,
    context_encoder: Any,
    behavior_basis: Any,
    source_hidden: int,
    hidden: int,
    fir_length: int,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    device = behavior_basis.device
    dtype = behavior_basis.dtype
    behavior_rank = int(behavior_basis.shape[1])
    future_width = (fir_length + 1) * behavior_rank
    c_zz = torch.zeros((source_hidden, source_hidden), device=device, dtype=dtype)
    c_zf = torch.zeros((source_hidden, future_width), device=device, dtype=dtype)
    c_ff = torch.zeros((future_width, future_width), device=device, dtype=dtype)
    c_zd = torch.zeros((source_hidden, hidden), device=device, dtype=dtype)
    total_valid_rows = 0
    for case in evidence:
        batch, tokens, _ = map(int, case.desired_context.shape)
        for batch_index in range(batch):
            desired = case.desired_context[batch_index]
            injection = case.parent_injection[batch_index]
            weight = case.prefix_weight[batch_index]
            mask = case.valid_mask[batch_index]
            valid_rows = int(mask.sum().item())
            total_valid_rows += valid_rows
            tau = max(policy.tail_tau_floor, math.sqrt(max(valid_rows, 1)))
            rho = math.exp(-1.0 / tau)
            tail_next = torch.zeros(
                (behavior_rank,),
                device=device,
                dtype=dtype,
            )
            stop = tokens
            while stop > 0:
                start = max(0, stop - policy.row_chunk_size)
                count = stop - start
                audit.rows(count, max(source_hidden, future_width, hidden))
                tail = torch.zeros(
                    (count, behavior_rank),
                    device=device,
                    dtype=dtype,
                )
                for time_index in range(stop - 1, start - 1, -1):
                    future_index = time_index + fir_length
                    if future_index < tokens and bool(mask[future_index].item()):
                        future_value = desired[future_index] @ behavior_basis
                    else:
                        future_value = torch.zeros_like(tail_next)
                    tail_next = (1.0 - rho) * future_value + rho * tail_next
                    tail[time_index - start] = tail_next
                blocks = []
                for lag in range(fir_length):
                    shifted = torch.zeros(
                        (count, behavior_rank),
                        device=device,
                        dtype=dtype,
                    )
                    source_start = start + lag
                    source_stop = min(stop + lag, tokens)
                    available = max(0, source_stop - source_start)
                    if available:
                        projected = (
                            desired[source_start:source_stop] @ behavior_basis
                        )
                        future_mask = mask[source_start:source_stop]
                        shifted[:available] = projected * future_mask.unsqueeze(-1).to(
                            dtype
                        )
                    blocks.append(shifted)
                blocks.append(tail)
                future = torch.cat(tuple(blocks), dim=-1)
                current_injection = injection[start:stop]
                current_desired = desired[start:stop]
                current_weight = weight[start:stop]
                weighted_injection = current_weight.unsqueeze(-1) * current_injection
                c_zz = c_zz + current_injection.T @ weighted_injection
                c_zf = c_zf + weighted_injection.T @ future
                c_ff = c_ff + future.T @ (
                    current_weight.unsqueeze(-1) * future
                )
                c_zd = c_zd + weighted_injection.T @ current_desired
                stop = start
    inverse_ff, ff_report = _conditioned_dense_solution(
        c_ff,
        torch.eye(future_width, device=device, dtype=dtype),
        torch.zeros_like(c_ff),
        condition_cap=policy.condition_cap,
        ridge_floor=policy.absolute_ridge_floor,
        rank_tolerance=policy.rank_relative_tolerance,
    )
    observability = c_zf @ inverse_ff @ c_zf.T
    encoder = context_encoder.detach().to(device=device, dtype=dtype)
    components_with_trace = (
        _trace_normalized(c_zz),
        _trace_normalized(observability),
        _trace_normalized(encoder @ encoder.T),
    )
    active = [value for value, trace in components_with_trace if trace > 0.0]
    aggregate = sum(active, torch.zeros_like(c_zz)) / max(len(active), 1)
    source_basis, spectrum = _top_eigenvectors(aggregate, hidden)
    left, _singular, right_t = torch.linalg.svd(
        source_basis.T @ c_zd,
        full_matrices=False,
    )
    aligned = source_basis @ left @ right_t
    state_energy = (aligned * (c_zz @ aligned)).sum(dim=0).clamp(
        min=torch.finfo(dtype).eps
    )
    context_covariance = sum(
        (
            case.desired_context.reshape(-1, hidden).T
            @ (
                case.prefix_weight.reshape(-1, 1)
                * case.desired_context.reshape(-1, hidden)
            )
            for case in evidence
        ),
        torch.zeros((hidden, hidden), device=device, dtype=dtype),
    )
    context_energy = context_covariance.diagonal().clamp(
        min=torch.finfo(dtype).eps
    )
    aligned = (
        aligned * (context_energy / state_energy).sqrt().unsqueeze(0)
    ).contiguous()
    return aligned, {
        "future_feature_width": future_width,
        "exact_future_lags": fir_length,
        "long_tail_includes_every_later_cached_horizon": True,
        "tail_tau_formula": "max(tail_tau_floor,sqrt(valid_rows_per_trajectory))",
        "valid_fit_rows": total_valid_rows,
        "future_gram_solve": ff_report,
        "component_traces": {
            "parent_injection_covariance": components_with_trace[0][1],
            "future_hankel_observability": components_with_trace[1][1],
            "audited_context_encoder_span": components_with_trace[2][1],
        },
        "active_component_count": len(active),
        "leading_state_spectrum": [float(value) for value in spectrum.tolist()],
        "state_encoder_sha256": _tensor_sha256(
            aligned,
            "fhvr.state_encoder",
        ),
        "all_fit_cases_consumed": True,
        "all_valid_prefix_rows_consumed": True,
    }


def _prepare_realization_cases(
    evidence: Sequence[_EvidenceCase],
    state_encoder: Any,
) -> tuple[_PreparedCase, ...]:
    return tuple(
        _PreparedCase(
            case_id=case.case_id,
            case_authentication_sha256=case.case_authentication_sha256,
            features=case.features,
            desired_state=(case.parent_injection @ state_encoder).contiguous(),
            desired_context=case.desired_context,
            desired_write=case.desired_write,
            prefix_weight=case.prefix_weight,
            suffix_weight=case.suffix_weight,
            valid_mask=case.valid_mask,
        )
        for case in evidence
    )


def _split_augmented_roles(
    state: Mapping[str, Any | None],
    hidden: int,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    weight = state["in_proj_weight"].reshape(3, hidden, hidden)
    if state["in_proj_bias"] is None:
        roles = weight
    else:
        bias = state["in_proj_bias"].reshape(3, hidden, 1)
        roles = torch.cat((weight, bias), dim=-1)
    return tuple(role.clone().contiguous() for role in roles)  # type: ignore[return-value]


def _factor_solve(
    cases: Sequence[_PreparedCase],
    *,
    other: Any,
    prior: Any,
    target_name: str,
    direct_multiplier: Sequence[Any] | None,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[Any, dict[str, Any]]:
    hidden, feature_width = map(int, prior.shape)

    def chunks(channel: slice) -> Iterator[tuple[Any, Any, Any, Any]]:
        for case_index, case in enumerate(cases):
            features = case.features.reshape(-1, feature_width)
            target = getattr(case, target_name).reshape(-1, hidden)
            weights = case.prefix_weight.reshape(-1)
            direct = (
                None
                if direct_multiplier is None
                else direct_multiplier[case_index].reshape(-1, hidden)
            )
            for row_slice in _chunks(int(features.shape[0]), policy.row_chunk_size):
                feature_chunk = features[row_slice]
                multiplier = (
                    feature_chunk @ other[channel].T
                    if direct is None
                    else direct[row_slice, channel]
                )
                yield (
                    feature_chunk,
                    multiplier,
                    target[row_slice, channel],
                    weights[row_slice],
                )

    return _conditional_solve(
        feature_width=feature_width,
        hidden=hidden,
        prior=prior,
        chunk_factory=chunks,
        policy=policy,
        audit=audit,
    )


def _response_rms(
    cases: Sequence[_PreparedCase],
    weight: Any,
) -> Any:
    torch = _torch()
    hidden = int(weight.shape[0])
    energy = torch.zeros((hidden,), device=weight.device, dtype=weight.dtype)
    for case in cases:
        response = case.features @ weight.T
        energy = energy + (
            case.prefix_weight.unsqueeze(-1) * response.square()
        ).sum(dim=(0, 1))
    return energy.clamp(min=torch.finfo(weight.dtype).eps).sqrt()


def _balance_pair(
    cases: Sequence[_PreparedCase],
    left: Any,
    right: Any,
) -> tuple[Any, Any]:
    left_rms = _response_rms(cases, left)
    right_rms = _response_rms(cases, right)
    scale = (right_rms / left_rms).sqrt()
    return (
        (left * scale.unsqueeze(-1)).contiguous(),
        (right / scale.unsqueeze(-1)).contiguous(),
    )


def _causal_filter(value: Any, kernel: Any, bias: Any | None) -> Any:
    torch = _torch()
    batch, sequence, hidden = map(int, value.shape)
    fir_length = int(kernel.shape[1])
    padded = torch.nn.functional.pad(
        value.transpose(1, 2),
        (fir_length - 1, 0),
    )
    filtered = torch.nn.functional.conv1d(
        padded,
        kernel.reshape(hidden, 1, fir_length),
        bias=bias,
        groups=hidden,
    )
    if tuple(filtered.shape) != (batch, hidden, sequence):
        raise Lfm2CpcfFhvrMfError("causal filter changed trajectory geometry.")
    return filtered.transpose(1, 2).contiguous()


def _filtered_histories(
    cases: Sequence[_PreparedCase],
    *,
    b_weight: Any,
    x_weight: Any,
    fir_kernel: Any,
    fir_bias: Any | None,
) -> tuple[Any, ...]:
    result = []
    for case in cases:
        b_value = case.features @ b_weight.T
        x_value = case.features @ x_weight.T
        result.append(
            _causal_filter(
                b_value * x_value,
                fir_kernel,
                fir_bias,
            )
        )
    return tuple(result)


def _solve_fir(
    cases: Sequence[_PreparedCase],
    *,
    b_weight: Any,
    x_weight: Any,
    c_weight: Any,
    prior_kernel: Any,
    prior_bias: Any | None,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[Any, Any | None, dict[str, Any]]:
    torch = _torch()
    hidden, fir_length = map(int, prior_kernel.shape)
    include_bias = prior_bias is not None
    coordinate_width = fir_length + int(include_bias)
    gram = torch.zeros(
        (hidden, coordinate_width, coordinate_width),
        device=prior_kernel.device,
        dtype=prior_kernel.dtype,
    )
    right = torch.zeros(
        (hidden, coordinate_width),
        device=prior_kernel.device,
        dtype=prior_kernel.dtype,
    )
    for case in cases:
        batch, tokens, _ = map(int, case.features.shape)
        b_value = case.features @ b_weight.T
        x_value = case.features @ x_weight.T
        c_value = case.features @ c_weight.T
        gated = b_value * x_value
        for batch_index in range(batch):
            for row_slice in _chunks(tokens, policy.row_chunk_size):
                start, stop = row_slice.start, row_slice.stop
                assert start is not None and stop is not None
                count = stop - start
                audit.rows(count, hidden * coordinate_width)
                lag = torch.zeros(
                    (count, hidden, fir_length),
                    device=prior_kernel.device,
                    dtype=prior_kernel.dtype,
                )
                positions = torch.arange(
                    start,
                    stop,
                    device=prior_kernel.device,
                )
                for kernel_index in range(fir_length):
                    delay = fir_length - 1 - kernel_index
                    source_positions = positions - delay
                    valid = source_positions >= 0
                    if bool(valid.any().item()):
                        lag[valid, :, kernel_index] = gated[
                            batch_index,
                            source_positions[valid],
                        ]
                if include_bias:
                    lag = torch.cat(
                        (
                            lag,
                            torch.ones(
                                (count, hidden, 1),
                                device=lag.device,
                                dtype=lag.dtype,
                            ),
                        ),
                        dim=-1,
                    )
                c_chunk = c_value[batch_index, start:stop]
                design = c_chunk.unsqueeze(-1) * lag
                weight = case.prefix_weight[batch_index, start:stop]
                target = case.desired_context[batch_index, start:stop]
                gram = gram + torch.einsum(
                    "r,rhq,rhp->hqp",
                    weight,
                    design,
                    design,
                )
                right = right + torch.einsum(
                    "r,rhq,rh->hq",
                    weight,
                    design,
                    target,
                )
    prior = (
        prior_kernel
        if prior_bias is None
        else torch.cat((prior_kernel, prior_bias.unsqueeze(-1)), dim=1)
    )
    solution = torch.empty_like(prior)
    ridges = []
    conditions = []
    for channel in range(hidden):
        value, report = _conditioned_dense_solution(
            gram[channel],
            right[channel].unsqueeze(-1),
            prior[channel].unsqueeze(-1),
            condition_cap=policy.condition_cap,
            ridge_floor=policy.absolute_ridge_floor,
            rank_tolerance=policy.rank_relative_tolerance,
        )
        solution[channel] = value[:, 0]
        ridges.append(report["effective_ridge"])
        conditions.append(report["regularized_condition"])
    return (
        solution[:, :fir_length].contiguous(),
        solution[:, -1].contiguous() if include_bias else None,
        {
            "coordinate_width": coordinate_width,
            "per_channel_small_gram_shape": [
                hidden,
                coordinate_width,
                coordinate_width,
            ],
            "effective_ridge_minimum": min(ridges),
            "effective_ridge_maximum": max(ridges),
            "regularized_condition_maximum": max(conditions),
            "condition_cap": policy.condition_cap,
            "n_by_n_matrix_materialized": False,
        },
    )


def _balance_context_fir(
    cases: Sequence[_PreparedCase],
    *,
    b_weight: Any,
    x_weight: Any,
    c_weight: Any,
    fir_kernel: Any,
    fir_bias: Any | None,
) -> tuple[Any, Any, Any | None]:
    torch = _torch()
    hidden = int(c_weight.shape[0])
    c_energy = torch.zeros((hidden,), device=c_weight.device, dtype=c_weight.dtype)
    f_energy = torch.zeros_like(c_energy)
    for case, filtered in zip(
        cases,
        _filtered_histories(
            cases,
            b_weight=b_weight,
            x_weight=x_weight,
            fir_kernel=fir_kernel,
            fir_bias=fir_bias,
        ),
    ):
        c_value = case.features @ c_weight.T
        weight = case.prefix_weight.unsqueeze(-1)
        c_energy = c_energy + (weight * c_value.square()).sum(dim=(0, 1))
        f_energy = f_energy + (weight * filtered.square()).sum(dim=(0, 1))
    epsilon = torch.finfo(c_weight.dtype).eps
    scale = (
        f_energy.clamp(min=epsilon).sqrt()
        / c_energy.clamp(min=epsilon).sqrt()
    ).sqrt()
    return (
        (c_weight * scale.unsqueeze(-1)).contiguous(),
        (fir_kernel / scale.unsqueeze(-1)).contiguous(),
        None
        if fir_bias is None
        else (fir_bias / scale).contiguous(),
    )


def _final_writer(
    cases: Sequence[_PreparedCase],
    *,
    b_weight: Any,
    c_weight: Any,
    x_weight: Any,
    fir_kernel: Any,
    fir_bias: Any | None,
    prior_weight: Any,
    prior_bias: Any | None,
    policy: Lfm2CpcfFhvrMfPolicy,
    audit: _WorkspaceAudit,
) -> tuple[Any, Any | None, dict[str, Any]]:
    torch = _torch()
    hidden = int(prior_weight.shape[0])
    include_bias = prior_bias is not None
    rows = []
    for case, filtered in zip(
        cases,
        _filtered_histories(
            cases,
            b_weight=b_weight,
            x_weight=x_weight,
            fir_kernel=fir_kernel,
            fir_bias=fir_bias,
        ),
    ):
        context = (case.features @ c_weight.T) * filtered
        design = context
        if include_bias:
            design = torch.cat(
                (
                    design,
                    torch.ones(
                        (*design.shape[:2], 1),
                        device=design.device,
                        dtype=design.dtype,
                    ),
                ),
                dim=-1,
            )
        rows.append((design, case.desired_write, case.suffix_weight))
    prior = prior_weight.T
    if include_bias:
        prior = torch.cat((prior, prior_bias.unsqueeze(0)), dim=0)
    solution, report = _streamed_dense_ridge(
        tuple(rows),
        prior=prior,
        policy=policy,
        audit=audit,
    )
    return (
        solution[:hidden].T.contiguous(),
        solution[-1].contiguous() if include_bias else None,
        report,
    )


def _compact_solve_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": report["kind"],
        "feature_width": report["feature_width"],
        "output_channels": report["output_channels"],
        "fit_rows_visited_per_normal_pass": report[
            "fit_rows_visited_per_normal_pass"
        ],
        "condition_cap": report["condition_cap"],
        "effective_ridge_minimum": report["effective_ridge_minimum"],
        "effective_ridge_maximum": report["effective_ridge_maximum"],
        "pcg_iterations": report["pcg_iterations"],
        "maximum_final_relative_normal_residual": report[
            "maximum_relative_normal_residual_by_iteration"
        ][-1],
        "breakdown_channel_count": report["breakdown_channel_count"],
        "allocation_audit": report["allocation_audit"],
        "report_sha256": report["report_sha256"],
    }


def build_lfm2_cpcf_fhvr_mf_candidate(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    context_encoder: Any,
    policy: Lfm2CpcfFhvrMfPolicy | None = None,
    candidate_id: str = FHVR_MF_CANDIDATE_ID,
) -> Lfm2CpcfShortConvNumericalRawCandidate:
    """Build one fit-only dense FHVR-MF raw candidate.

    ``request`` is the existing numerical-frontier request and therefore
    contains fit cases only.  ``context_encoder`` is the adapter's already
    audited source-to-target context encoder.  Selection and validation are not
    accepted by this API.
    """

    torch = _torch()
    solve_policy = Lfm2CpcfFhvrMfPolicy() if policy is None else policy
    if (
        not isinstance(request, Lfm2CpcfShortConvNumericalFrontierRequest)
        or not request.verify_authentication()
    ):
        raise Lfm2CpcfFhvrMfError(
            "FHVR-MF requires an authenticated numerical-frontier request."
        )
    if (
        not isinstance(solve_policy, Lfm2CpcfFhvrMfPolicy)
        or not solve_policy.verify_authentication()
    ):
        raise Lfm2CpcfFhvrMfError("FHVR-MF policy failed authentication.")
    if any(case.split != "fit" for case in request.cases):
        raise Lfm2CpcfFhvrMfError("FHVR-MF accepts fit cases only.")
    source_hidden = int(request.report["source_hidden_size"])
    hidden = int(request.report["target_hidden_size"])
    if (
        not isinstance(context_encoder, torch.Tensor)
        or context_encoder.layout != torch.strided
        or tuple(context_encoder.shape) != (source_hidden, hidden)
        or not bool(torch.isfinite(context_encoder).all().item())
    ):
        raise Lfm2CpcfFhvrMfError(
            "context_encoder must be one finite [source_hidden,target_hidden] tensor."
        )
    device, dtype, device_name = _resolve_device(solve_policy)
    audit = _WorkspaceAudit(
        row_cap=solve_policy.row_chunk_size,
        channel_cap=solve_policy.channel_chunk_size,
    )
    with _deterministic_compute_scope(device):
        encoder = context_encoder.detach().to(device=device, dtype=dtype)
        evidence = _prepare_evidence(
            request,
            context_encoder=encoder,
            device=device,
            dtype=dtype,
            policy=solve_policy,
            audit=audit,
        )
        initial_state, initial_report = _initial_state(
            request,
            evidence,
            context_encoder=encoder,
            policy=solve_policy,
            audit=audit,
        )
        behavior_basis, behavior_report = _behavior_basis(
            evidence,
            writer_weight=initial_state["writer_weight"],
            hidden=hidden,
            policy=solve_policy,
            audit=audit,
        )
        state_encoder, state_report = _future_state_encoder(
            evidence,
            context_encoder=encoder,
            behavior_basis=behavior_basis,
            source_hidden=source_hidden,
            hidden=hidden,
            fir_length=int(request.report["target_fir_length"]),
            policy=solve_policy,
            audit=audit,
        )
        cases = _prepare_realization_cases(evidence, state_encoder)
        b_weight, c_weight, x_weight = _split_augmented_roles(
            initial_state,
            hidden,
        )
        factor_reports = []
        for sweep in range(solve_policy.factor_sweeps):
            b_weight, b_report = _factor_solve(
                cases,
                other=x_weight,
                prior=b_weight,
                target_name="desired_state",
                direct_multiplier=None,
                policy=solve_policy,
                audit=audit,
            )
            x_weight, x_report = _factor_solve(
                cases,
                other=b_weight,
                prior=x_weight,
                target_name="desired_state",
                direct_multiplier=None,
                policy=solve_policy,
                audit=audit,
            )
            b_weight, x_weight = _balance_pair(cases, b_weight, x_weight)
            factor_reports.append(
                {
                    "sweep": sweep,
                    "b": _compact_solve_report(b_report),
                    "x": _compact_solve_report(x_report),
                    "reciprocal_bx_rms_balance": True,
                }
            )
        fir_kernel = initial_state["fir_kernel"].clone()
        fir_bias = (
            None
            if initial_state["fir_bias"] is None
            else initial_state["fir_bias"].clone()
        )
        context_reports = []
        for sweep in range(solve_policy.factor_sweeps):
            filtered = _filtered_histories(
                cases,
                b_weight=b_weight,
                x_weight=x_weight,
                fir_kernel=fir_kernel,
                fir_bias=fir_bias,
            )
            c_weight, c_report = _factor_solve(
                cases,
                other=c_weight,
                prior=c_weight,
                target_name="desired_context",
                direct_multiplier=filtered,
                policy=solve_policy,
                audit=audit,
            )
            fir_kernel, fir_bias, fir_report = _solve_fir(
                cases,
                b_weight=b_weight,
                x_weight=x_weight,
                c_weight=c_weight,
                prior_kernel=fir_kernel,
                prior_bias=fir_bias,
                policy=solve_policy,
                audit=audit,
            )
            c_weight, fir_kernel, fir_bias = _balance_context_fir(
                cases,
                b_weight=b_weight,
                x_weight=x_weight,
                c_weight=c_weight,
                fir_kernel=fir_kernel,
                fir_bias=fir_bias,
            )
            context_reports.append(
                {
                    "sweep": sweep,
                    "c": _compact_solve_report(c_report),
                    "fir": fir_report,
                    "reciprocal_c_fir_rms_balance": True,
                }
            )
        writer_weight, writer_bias, writer_report = _final_writer(
            cases,
            b_weight=b_weight,
            c_weight=c_weight,
            x_weight=x_weight,
            fir_kernel=fir_kernel,
            fir_bias=fir_bias,
            prior_weight=initial_state["writer_weight"],
            prior_bias=initial_state["writer_bias"],
            policy=solve_policy,
            audit=audit,
        )
        in_proj_weight = torch.cat(
            (
                b_weight[:, :hidden],
                c_weight[:, :hidden],
                x_weight[:, :hidden],
            ),
            dim=0,
        ).contiguous()
        in_proj_bias = (
            None
            if request.target_templates["in_proj_bias"] is None
            else torch.cat(
                (
                    b_weight[:, -1],
                    c_weight[:, -1],
                    x_weight[:, -1],
                )
            ).contiguous()
        )
        final_state = {
            "in_proj_weight": in_proj_weight,
            "in_proj_bias": in_proj_bias,
            "fir_kernel": fir_kernel,
            "fir_bias": fir_bias,
            "writer_weight": writer_weight,
            "writer_bias": writer_bias,
        }
        for role, value in final_state.items():
            template = request.target_templates[role]
            if (value is None) != (template is None) or (
                value is not None and tuple(value.shape) != tuple(template.shape)
            ):
                raise Lfm2CpcfFhvrMfError(
                    f"FHVR-MF produced invalid target topology for {role}."
                )
            if value is not None and not bool(torch.isfinite(value).all().item()):
                raise Lfm2CpcfFhvrMfError(
                    f"FHVR-MF produced non-finite {role}."
                )
        provenance = {
            "method": "future_hankel_volterra_matrix_free_realization",
            "implementation": _IMPLEMENTATION,
            "schema": LFM2_CPCF_SHORT_CONV_V5_FHVR_MF_SCHEMA,
            "policy": solve_policy.as_dict(),
            "request_sha256": request.request_sha256,
            "context_encoder_sha256": _tensor_sha256(
                encoder,
                "fhvr.context_encoder",
            ),
            "fit_case_authentication_sha256": [
                case.case_authentication_sha256 for case in cases
            ],
            "initialization": initial_report,
            "behavior_basis": behavior_report,
            "future_state_identification": state_report,
            "bx_factor_solves": factor_reports,
            "context_fir_solves": context_reports,
            "writer_solve": writer_report,
            "allocation_audit": audit.report(),
            "compute_device": device_name,
            "compute_dtype": str(dtype).removeprefix("torch."),
            "target_native_dense_weights": True,
            "full_causal_fit_histories_used": True,
            "all_future_horizons_used_by_exact_lags_plus_long_tail": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
            "fit_only": True,
            "execution": {
                **_ZERO_EXECUTION,
                "closed_form_iterative_linear_solves": True,
            },
        }
    cpu_state = {
        role: (
            None
            if value is None
            else value.detach().to(device="cpu").contiguous()
        )
        for role, value in final_state.items()
    }
    return Lfm2CpcfShortConvNumericalRawCandidate(
        request_sha256=request.request_sha256,
        candidate_id=str(candidate_id).strip(),
        candidate_kind=FHVR_MF_CANDIDATE_KIND,
        in_proj_weight=cpu_state["in_proj_weight"],
        in_proj_bias=cpu_state["in_proj_bias"],
        fir_kernel=cpu_state["fir_kernel"],
        fir_bias=cpu_state["fir_bias"],
        writer_weight=cpu_state["writer_weight"],
        writer_bias=cpu_state["writer_bias"],
        method_provenance=provenance,
    )


__all__ = [
    "FHVR_MF_CANDIDATE_ID",
    "FHVR_MF_CANDIDATE_KIND",
    "LFM2_CPCF_SHORT_CONV_V5_FHVR_MF_SCHEMA",
    "Lfm2CpcfFhvrMfConditionalResult",
    "Lfm2CpcfFhvrMfError",
    "Lfm2CpcfFhvrMfPolicy",
    "build_lfm2_cpcf_fhvr_mf_candidate",
    "solve_lfm2_cpcf_fhvr_mf_conditional_factor",
]
