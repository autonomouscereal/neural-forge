"""Scalable, globally coupled B/C/x residual folding for LFM2 CPCF v5.

The legacy v5 numerical adapter solves one channel at a time by materializing
an ``[rows, 3 * (hidden + bias)]`` Jacobian and a dense Gram matrix.  At real
LFM2 geometry that is both expensive and mathematically incomplete: the
writer couples every context channel into every residual-write coordinate.

This additive module implements the same first-order B/C/x correction in the
full dense parameter space without materializing the design or Gram:

* all ``3 * hidden * (hidden + bias)`` physical correction coordinates remain;
* one global implicit operator includes the complete writer-column coupling;
* context and endpoint residuals use equal case mass and a causal-horizon
  mixture that does not let a long prefill erase the decode tail;
* the normal equations are solved by a fixed deterministic PCG schedule with
  a channel-energy diagonal preconditioner;
* a rigorous operator-norm upper bound sets a conservative condition-cap
  ridge; and
* parameter-space and function-space trust limits jointly scale the resulting
  correction around the dense transported prior.

This is numerical folding, not training.  The implementation never constructs
an autograd graph, calls backward, invokes an optimizer, prunes parameters, or
quantizes parameters.  It is intentionally not wired into the v5 adapter yet;
the public API and small dense legacy oracle allow isolated validation first.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfV4ProjectionCase,
    Lfm2CpcfV4VolterraReplay,
    linearized_bcx_affine_design,
)


LFM2_CPCF_SHORT_CONV_V5_MATRIX_FREE_BCX_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-matrix-free-bcx-v1"
)
_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_cpcf_short_conv_v5_matrix_free_bcx."
    "global-implicit-endpoint-context-pcg.v1"
)
_ZERO_EXECUTION = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "generation_calls": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2CpcfShortConvV5MatrixFreeBcxError(ValueError):
    """The matrix-free B/C/x problem or policy is inconsistent."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for matrix-free LFM2 CPCF B/C/x folding."
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
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "Matrix-free B/C/x metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _sha256(detached)})


def verify_lfm2_cpcf_short_conv_v5_matrix_free_bcx_report(
    report: Mapping[str, Any],
) -> bool:
    """Verify one canonical matrix-free B/C/x receipt."""

    if not isinstance(report, Mapping):
        return False
    try:
        body = dict(report)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _sha256(body)
    except (TypeError, ValueError):
        return False


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            f"{name} must be a materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(f"{name} must be a finite number.")
    return result


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = _finite_float(value, name)
    if result < 0.0 or (not allow_zero and result == 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            f"{name} must be a finite {qualifier} number."
        )
    return result


def _dense_floating_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            f"{name} must be a finite nonempty dense real rank-{dimensions} tensor."
        )
    return value.detach()


def _mask_tensor(value: Any, name: str, *, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or tuple(value.shape) != shape
    ):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            f"{name} must be a dense rank-2 mask with shape {shape}."
        )
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous()


def _causal_horizon_row_mass_with_report(
    valid_rows: Any,
    *,
    uniform_fraction: float,
    tail_tau_floor: float,
    dtype: Any | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Return normalized row mass and its deterministic horizon receipt."""

    torch = _torch()
    if (
        not isinstance(valid_rows, torch.Tensor)
        or valid_rows.layout != torch.strided
        or valid_rows.device.type == "meta"
        or valid_rows.ndim not in {1, 2}
        or valid_rows.numel() == 0
    ):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "valid_rows must be a nonempty dense rank-1 or rank-2 mask."
        )
    mixture = _positive_float(
        uniform_fraction,
        "uniform_fraction",
        allow_zero=True,
    )
    if mixture > 1.0:
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "uniform_fraction must be in [0, 1]."
        )
    tau_floor = _positive_float(tail_tau_floor, "tail_tau_floor")
    mask = valid_rows.detach().to(dtype=torch.bool)
    original_rank = mask.ndim
    if original_rank == 1:
        mask = mask.unsqueeze(0)
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "causal-horizon weighting requires at least one valid row."
        )
    result_dtype = torch.float64 if dtype is None else dtype
    uniform = mask.to(dtype=result_dtype) / valid_count
    tail = torch.zeros_like(uniform)
    trajectory_reports = []
    for trajectory_index in range(int(mask.shape[0])):
        positions = torch.nonzero(
            mask[trajectory_index],
            as_tuple=False,
        ).reshape(-1)
        count = int(positions.numel())
        if count == 0:
            trajectory_reports.append(
                {
                    "trajectory_index": trajectory_index,
                    "valid_rows": 0,
                    "tau": None,
                    "last_valid_position": None,
                }
            )
            continue
        tau = max(tau_floor, math.sqrt(count))
        last_position = int(positions[-1].item())
        distances = (last_position - positions.to(dtype=result_dtype)).clamp(min=0.0)
        tail[trajectory_index, positions] = torch.exp(-distances / tau)
        trajectory_reports.append(
            {
                "trajectory_index": trajectory_index,
                "valid_rows": count,
                "tau": tau,
                "last_valid_position": last_position,
            }
        )
    tail_sum = tail.sum()
    if float(tail_sum.item()) == 0.0:
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "causal-horizon tail mass unexpectedly vanished."
        )
    tail = tail / tail_sum
    mass = mixture * uniform + (1.0 - mixture) * tail
    mass = mass * mask.to(dtype=result_dtype)
    mass = mass / mass.sum()
    if original_rank == 1:
        mass = mass.squeeze(0)
    report = MappingProxyType(
        {
            "kind": "causal_horizon_row_mass",
            "valid_rows": valid_count,
            "uniform_fraction": mixture,
            "exponential_tail_fraction": 1.0 - mixture,
            "tail_tau_formula": "max(tail_tau_floor,sqrt(valid_rows_per_trajectory))",
            "tail_tau_floor": tau_floor,
            "trajectory_inventory": trajectory_reports,
            "total_mass": float(mass.sum().item()),
            "suffix_start_used_as_decode_boundary": False,
        }
    )
    return mass.contiguous(), report


def causal_horizon_row_mass(
    valid_rows: Any,
    *,
    uniform_fraction: float = 0.5,
    tail_tau_floor: float = 4.0,
    dtype: Any | None = None,
) -> Any:
    """Weight a causal trajectory without allowing a long prefix to dominate.

    Half of the default mass is uniform over all valid rows.  The other half is
    an exponentially decaying tail whose time constant is
    ``max(4, sqrt(valid rows))``.  The returned mass always sums to one across
    the complete supplied case.  ``suffix_start`` is deliberately absent from
    this API because it is not a reliable prefill/decode boundary in cached
    CPCF evidence.
    """

    mass, _report = _causal_horizon_row_mass_with_report(
        valid_rows,
        uniform_fraction=uniform_fraction,
        tail_tau_floor=tail_tau_floor,
        dtype=dtype,
    )
    return mass


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5MatrixFreeBcxPolicy:
    """Authenticated policy for the fixed matrix-free solve."""

    condition_cap: float = 10_000.0
    pcg_iterations: int = 16
    context_objective_weight: float = 0.05
    uniform_horizon_fraction: float = 0.5
    tail_tau_floor: float = 4.0
    parameter_trust_radius: float = 0.25
    effect_trust_radius: float = 1.0
    parameter_reference_floor: float = 1e-6
    effect_reference_floor: float = 1e-6
    absolute_ridge_floor: float = 1e-12
    backend: str = "auto"
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        cap = _positive_float(self.condition_cap, "condition_cap")
        if cap not in {1_000.0, 10_000.0}:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "condition_cap must be 1000 or 10000."
            )
        if (
            isinstance(self.pcg_iterations, bool)
            or not isinstance(self.pcg_iterations, int)
            or self.pcg_iterations <= 0
            or self.pcg_iterations > 256
        ):
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "pcg_iterations must be an integer in [1, 256]."
            )
        context = _positive_float(
            self.context_objective_weight,
            "context_objective_weight",
            allow_zero=True,
        )
        uniform = _positive_float(
            self.uniform_horizon_fraction,
            "uniform_horizon_fraction",
            allow_zero=True,
        )
        if context > 0.25:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "context_objective_weight must be in [0, 0.25]."
            )
        if uniform > 1.0:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "uniform_horizon_fraction must be in [0, 1]."
            )
        backend = str(self.backend).strip().lower()
        if backend not in {"auto", "cpu", "cuda"}:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "backend must be auto, cpu, or cuda."
            )
        object.__setattr__(self, "condition_cap", cap)
        object.__setattr__(self, "context_objective_weight", context)
        object.__setattr__(self, "uniform_horizon_fraction", uniform)
        object.__setattr__(
            self,
            "tail_tau_floor",
            _positive_float(self.tail_tau_floor, "tail_tau_floor"),
        )
        for name in (
            "parameter_trust_radius",
            "effect_trust_radius",
            "parameter_reference_floor",
            "effect_reference_floor",
            "absolute_ridge_floor",
        ):
            object.__setattr__(
                self,
                name,
                _positive_float(getattr(self, name), name),
            )
        object.__setattr__(self, "backend", backend)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "matrix-free policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_MATRIX_FREE_BCX_SCHEMA,
            "kind": "matrix_free_bcx_policy",
            "condition_cap": self.condition_cap,
            "pcg_iterations": self.pcg_iterations,
            "context_objective_weight": self.context_objective_weight,
            "uniform_horizon_fraction": self.uniform_horizon_fraction,
            "tail_tau_floor": self.tail_tau_floor,
            "parameter_trust_radius": self.parameter_trust_radius,
            "effect_trust_radius": self.effect_trust_radius,
            "parameter_reference_floor": self.parameter_reference_floor,
            "effect_reference_floor": self.effect_reference_floor,
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "backend": self.backend,
            "fixed_iteration_schedule": True,
            "early_stopping": False,
            "global_writer_coupling": True,
            "physical_coordinate_topology": "3_by_hidden_by_hidden_plus_bias",
            "execution": {
                **_ZERO_EXECUTION,
                "autograd_graphs": 0,
                "closed_form_iterative_linear_solve": True,
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
class Lfm2CpcfV5MatrixFreeBcxResult:
    """Updated dense input projection plus the solved correction and receipt."""

    in_proj_weight: Any
    in_proj_bias: Any | None
    raw_correction: Any
    accepted_correction: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        for name in (
            "in_proj_weight",
            "raw_correction",
            "accepted_correction",
        ):
            value = _dense_floating_tensor(
                getattr(self, name),
                name,
                dimensions=2 if name == "in_proj_weight" else 3,
            )
            object.__setattr__(
                self,
                name,
                value.to(device="cpu").contiguous().clone(),
            )
        if self.in_proj_bias is not None:
            bias = _dense_floating_tensor(
                self.in_proj_bias,
                "in_proj_bias",
                dimensions=1,
            )
            object.__setattr__(
                self,
                "in_proj_bias",
                bias.to(device="cpu").contiguous().clone(),
            )
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )
        if self.raw_correction.dtype != torch.float64:
            object.__setattr__(
                self,
                "raw_correction",
                self.raw_correction.to(dtype=torch.float64),
            )
        if self.accepted_correction.dtype != torch.float64:
            object.__setattr__(
                self,
                "accepted_correction",
                self.accepted_correction.to(dtype=torch.float64),
            )

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_cpcf_short_conv_v5_matrix_free_bcx_report(self.diagnostics)
            and self.diagnostics.get("updated_in_proj_weight_sha256")
            == _tensor_sha256(self.in_proj_weight, "result.in_proj_weight")
            and self.diagnostics.get("updated_in_proj_bias_sha256")
            == (
                None
                if self.in_proj_bias is None
                else _tensor_sha256(self.in_proj_bias, "result.in_proj_bias")
            )
            and self.diagnostics.get("raw_correction_sha256")
            == _tensor_sha256(self.raw_correction, "result.raw_correction")
            and self.diagnostics.get("accepted_correction_sha256")
            == _tensor_sha256(
                self.accepted_correction,
                "result.accepted_correction",
            )
        )


@dataclass(slots=True)
class _PackedBcxProblem:
    features: Any
    b_history: Any
    c_history: Any
    x_history: Any
    base_filtered: Any
    endpoint_residual: Any
    context_residual: Any
    endpoint_row_mass: Any
    context_row_mass: Any
    writer_weight: Any
    fir_kernel: Any
    prior: Any
    case_inventory: tuple[Mapping[str, Any], ...]
    hidden: int
    feature_width: int
    output_width: int
    fit_bias: bool
    device: Any
    dtype: Any


def _resolve_device_and_dtype(policy: Lfm2CpcfV5MatrixFreeBcxPolicy) -> tuple[Any, Any]:
    torch = _torch()
    cuda_available = bool(torch.cuda.is_available())
    if policy.backend == "cuda" and not cuda_available:
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "backend='cuda' was requested but CUDA is unavailable."
        )
    use_cuda = policy.backend == "cuda" or (policy.backend == "auto" and cuda_available)
    return (
        torch.device("cuda") if use_cuda else torch.device("cpu"),
        torch.float32 if use_cuda else torch.float64,
    )


def _causal_shift(value: Any, delay: int) -> Any:
    torch = _torch()
    if delay == 0:
        return value
    result = torch.zeros_like(value)
    result[:, delay:] = value[:, :-delay]
    return result


def _causal_fir(value: Any, kernel: Any) -> Any:
    torch = _torch()
    result = torch.zeros_like(value)
    fir_length = int(kernel.shape[1])
    for kernel_index in range(fir_length):
        delay = fir_length - 1 - kernel_index
        coefficient = kernel[:, kernel_index].reshape(1, 1, -1)
        if delay == 0:
            result = result + value * coefficient
        else:
            updated = torch.zeros_like(value)
            updated[:, delay:] = value[:, :-delay] * coefficient
            result = result + updated
    return result


def _causal_fir_adjoint(value: Any, kernel: Any) -> Any:
    torch = _torch()
    result = torch.zeros_like(value)
    fir_length = int(kernel.shape[1])
    for kernel_index in range(fir_length):
        delay = fir_length - 1 - kernel_index
        coefficient = kernel[:, kernel_index].reshape(1, 1, -1)
        if delay == 0:
            result = result + value * coefficient
        else:
            updated = torch.zeros_like(value)
            updated[:, :-delay] = value[:, delay:] * coefficient
            result = result + updated
    return result


def _pack_prior(in_proj_weight: Any, in_proj_bias: Any | None) -> Any:
    torch = _torch()
    hidden = int(in_proj_weight.shape[1])
    fit_bias = in_proj_bias is not None
    feature_width = hidden + int(fit_bias)
    prior = torch.zeros(
        (3, hidden, feature_width),
        device=in_proj_weight.device,
        dtype=in_proj_weight.dtype,
    )
    prior[..., :hidden] = in_proj_weight.reshape(3, hidden, hidden)
    if in_proj_bias is not None:
        prior[..., -1] = in_proj_bias.reshape(3, hidden)
    return prior


def _canonical_case_replays(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: Sequence[Lfm2CpcfV4VolterraReplay],
) -> tuple[tuple[Lfm2CpcfV4ProjectionCase, Lfm2CpcfV4VolterraReplay], ...]:
    case_values = tuple(cases)
    replay_values = tuple(replays)
    if not case_values or len(case_values) != len(replay_values):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "cases and replays must be nonempty and have identical lengths."
        )
    pairs = []
    for case, replay in zip(case_values, replay_values):
        if not isinstance(case, Lfm2CpcfV4ProjectionCase):
            raise TypeError("cases must contain Lfm2CpcfV4ProjectionCase values.")
        if not isinstance(replay, Lfm2CpcfV4VolterraReplay):
            raise TypeError("replays must contain Lfm2CpcfV4VolterraReplay values.")
        if case.split != "fit":
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "matrix-free B/C/x fitting accepts fit cases only."
            )
        pairs.append((case, replay))
    pairs.sort(key=lambda value: value[0].case_id)
    if len({case.case_id for case, _replay in pairs}) != len(pairs):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "matrix-free B/C/x case ids must be unique."
        )
    return tuple(pairs)


def _build_packed_problem(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: Sequence[Lfm2CpcfV4VolterraReplay],
    *,
    in_proj_weight: Any,
    in_proj_bias: Any | None,
    fir_kernel: Any,
    fir_bias: Any | None,
    writer_weight: Any,
    policy: Lfm2CpcfV5MatrixFreeBcxPolicy,
) -> _PackedBcxProblem:
    torch = _torch()
    pairs = _canonical_case_replays(cases, replays)
    weight = _dense_floating_tensor(
        in_proj_weight,
        "in_proj_weight",
        dimensions=2,
    )
    hidden = int(weight.shape[1])
    if tuple(weight.shape) != (3 * hidden, hidden):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "in_proj_weight must have shape [3H,H]."
        )
    bias = None
    if in_proj_bias is not None:
        bias = _dense_floating_tensor(
            in_proj_bias,
            "in_proj_bias",
            dimensions=1,
        )
        if tuple(bias.shape) != (3 * hidden,):
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                "in_proj_bias must have shape [3H]."
            )
    kernel = _dense_floating_tensor(
        fir_kernel,
        "fir_kernel",
        dimensions=2,
    )
    if int(kernel.shape[0]) != hidden:
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError("fir_kernel must have H rows.")
    convolution_bias = None
    if fir_bias is not None:
        convolution_bias = _dense_floating_tensor(
            fir_bias,
            "fir_bias",
            dimensions=1,
        )
        if tuple(convolution_bias.shape) != (hidden,):
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError("fir_bias must have shape [H].")
    writer = _dense_floating_tensor(
        writer_weight,
        "writer_weight",
        dimensions=2,
    )
    if int(writer.shape[1]) != hidden:
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "writer_weight must have H input columns."
        )
    output_width = int(writer.shape[0])
    device, dtype = _resolve_device_and_dtype(policy)
    feature_width = hidden + int(bias is not None)
    total_trajectories = 0
    maximum_length = 0
    geometry = []
    for case, replay in pairs:
        child = _dense_floating_tensor(
            case.child_input_history,
            f"{case.case_id}.child_input_history",
            dimensions=3,
        )
        batch, sequence, case_hidden = map(int, child.shape)
        if case_hidden != hidden:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                f"{case.case_id} child hidden width differs from the target."
            )
        expected_history_shape = (batch, sequence, hidden)
        for role in ("b_history", "c_history", "x_history", "context_history"):
            value = _dense_floating_tensor(
                getattr(replay, role),
                f"{case.case_id}.{role}",
                dimensions=3,
            )
            if tuple(value.shape) != expected_history_shape:
                raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                    f"{case.case_id}.{role} has inconsistent geometry."
                )
        if tuple(replay.write_history.shape) != (batch, sequence, output_width):
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                f"{case.case_id}.write_history has inconsistent geometry."
            )
        if tuple(case.desired_context_history.shape) != expected_history_shape:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                f"{case.case_id}.desired_context_history has inconsistent geometry."
            )
        suffix_length = sequence - case.suffix_start
        if tuple(case.desired_write_suffix.shape) != (
            batch,
            suffix_length,
            output_width,
        ):
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                f"{case.case_id}.desired_write_suffix has inconsistent geometry."
            )
        prefix_mask = (
            torch.ones((batch, sequence), dtype=torch.bool)
            if case.valid_prefix_mask is None
            else _mask_tensor(
                case.valid_prefix_mask,
                f"{case.case_id}.valid_prefix_mask",
                shape=(batch, sequence),
            )
        )
        suffix_mask = (
            prefix_mask[:, case.suffix_start :]
            if case.valid_suffix_mask is None
            else _mask_tensor(
                case.valid_suffix_mask,
                f"{case.case_id}.valid_suffix_mask",
                shape=(batch, suffix_length),
            )
        )
        if int(suffix_mask.sum().item()) == 0:
            raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
                f"{case.case_id} has no valid endpoint rows."
            )
        geometry.append((case, replay, prefix_mask, suffix_mask))
        total_trajectories += batch
        maximum_length = max(maximum_length, sequence)

    packed_shape = (total_trajectories, maximum_length, hidden)
    features = torch.zeros(
        (total_trajectories, maximum_length, feature_width),
        device=device,
        dtype=dtype,
    )
    b_history = torch.zeros(packed_shape, device=device, dtype=dtype)
    c_history = torch.zeros_like(b_history)
    x_history = torch.zeros_like(b_history)
    endpoint_residual = torch.zeros(
        (total_trajectories, maximum_length, output_width),
        device=device,
        dtype=dtype,
    )
    context_residual = torch.zeros_like(b_history)
    endpoint_mass = torch.zeros(
        (total_trajectories, maximum_length),
        device=device,
        dtype=dtype,
    )
    context_mass = torch.zeros_like(endpoint_mass)
    case_inventory = []
    trajectory_offset = 0
    case_scale = 1.0 / len(geometry)
    for case, replay, prefix_mask, suffix_mask in geometry:
        batch, sequence, _hidden = map(int, case.child_input_history.shape)
        span = slice(trajectory_offset, trajectory_offset + batch)
        child_value = case.child_input_history.detach().to(
            device=device,
            dtype=dtype,
        )
        features[span, :sequence, :hidden] = child_value
        if bias is not None:
            features[span, :sequence, -1] = 1.0
        for destination, source in (
            (b_history, replay.b_history),
            (c_history, replay.c_history),
            (x_history, replay.x_history),
        ):
            destination[span, :sequence] = source.detach().to(
                device=device,
                dtype=dtype,
            )
        context_residual[span, :sequence] = case.desired_context_history.detach().to(
            device=device, dtype=dtype
        ) - replay.context_history.detach().to(device=device, dtype=dtype)
        suffix = slice(case.suffix_start, sequence)
        endpoint_residual[span, suffix] = case.desired_write_suffix.detach().to(
            device=device, dtype=dtype
        ) - replay.write_history[:, case.suffix_start :].detach().to(
            device=device, dtype=dtype
        )
        full_endpoint_mask = torch.zeros(
            (batch, sequence),
            dtype=torch.bool,
        )
        full_endpoint_mask[:, case.suffix_start :] = suffix_mask
        endpoint_case_mass, endpoint_report = _causal_horizon_row_mass_with_report(
            full_endpoint_mask,
            uniform_fraction=policy.uniform_horizon_fraction,
            tail_tau_floor=policy.tail_tau_floor,
            dtype=torch.float64,
        )
        context_case_mass, context_report = _causal_horizon_row_mass_with_report(
            prefix_mask,
            uniform_fraction=policy.uniform_horizon_fraction,
            tail_tau_floor=policy.tail_tau_floor,
            dtype=torch.float64,
        )
        endpoint_mass[span, :sequence] = (
            endpoint_case_mass.to(device=device, dtype=dtype) * case_scale
        )
        context_mass[span, :sequence] = (
            context_case_mass.to(device=device, dtype=dtype) * case_scale
        )
        case_inventory.append(
            MappingProxyType(
                {
                    "case_id": case.case_id,
                    "projection_case_authentication_sha256": (
                        case.authentication_sha256
                    ),
                    "replay_report_sha256": replay.arithmetic_report.get(
                        "report_sha256"
                    ),
                    "batch": batch,
                    "sequence": sequence,
                    "suffix_start": case.suffix_start,
                    "endpoint_row_mass": dict(endpoint_report),
                    "context_row_mass": dict(context_report),
                    "case_total_mass": case_scale,
                }
            )
        )
        trajectory_offset += batch
    kernel_value = kernel.to(device=device, dtype=dtype).contiguous()
    base_filtered = _causal_fir(b_history * x_history, kernel_value)
    if convolution_bias is not None:
        base_filtered = base_filtered + convolution_bias.to(
            device=device,
            dtype=dtype,
        ).reshape(1, 1, hidden)
    prior = _pack_prior(
        weight.to(device=device, dtype=dtype),
        None if bias is None else bias.to(device=device, dtype=dtype),
    )
    return _PackedBcxProblem(
        features=features,
        b_history=b_history,
        c_history=c_history,
        x_history=x_history,
        base_filtered=base_filtered,
        endpoint_residual=endpoint_residual,
        context_residual=context_residual,
        endpoint_row_mass=endpoint_mass,
        context_row_mass=context_mass,
        writer_weight=writer.to(device=device, dtype=dtype).contiguous(),
        fir_kernel=kernel_value,
        prior=prior,
        case_inventory=tuple(case_inventory),
        hidden=hidden,
        feature_width=feature_width,
        output_width=output_width,
        fit_bias=bias is not None,
        device=device,
        dtype=dtype,
    )


def _context_jvp(problem: _PackedBcxProblem, correction: Any) -> Any:
    features_2d = problem.features.reshape(-1, problem.feature_width)
    affine = tuple(
        (features_2d @ correction[role].transpose(0, 1)).reshape(
            *problem.features.shape[:2], problem.hidden
        )
        for role in range(3)
    )
    delta_b, delta_c, delta_x = affine
    filtered_delta = _causal_fir(
        delta_b * problem.x_history + problem.b_history * delta_x,
        problem.fir_kernel,
    )
    return (
        problem.c_history * filtered_delta + delta_c * problem.base_filtered
    ).contiguous()


def _context_vjp(problem: _PackedBcxProblem, context_cotangent: Any) -> Any:
    filtered_cotangent = _causal_fir_adjoint(
        context_cotangent * problem.c_history,
        problem.fir_kernel,
    )
    signal_b = filtered_cotangent * problem.x_history
    signal_x = filtered_cotangent * problem.b_history
    signal_c = context_cotangent * problem.base_filtered
    features_2d = problem.features.reshape(-1, problem.feature_width)
    return (
        _torch()
        .stack(
            tuple(
                signal.reshape(-1, problem.hidden).transpose(0, 1) @ features_2d
                for signal in (signal_b, signal_c, signal_x)
            ),
            dim=0,
        )
        .contiguous()
    )


def _rhs(problem: _PackedBcxProblem, context_weight: float) -> Any:
    endpoint_cotangent = (
        (problem.endpoint_residual @ problem.writer_weight)
        * problem.endpoint_row_mass.unsqueeze(-1)
        / problem.output_width
    )
    context_cotangent = (
        context_weight
        * problem.context_residual
        * problem.context_row_mass.unsqueeze(-1)
        / problem.hidden
    )
    return _context_vjp(problem, endpoint_cotangent + context_cotangent)


def _normal_without_ridge(
    problem: _PackedBcxProblem,
    correction: Any,
    *,
    context_weight: float,
) -> Any:
    context_effect = _context_jvp(problem, correction)
    endpoint_effect = context_effect @ problem.writer_weight.transpose(0, 1)
    endpoint_cotangent = (
        (endpoint_effect @ problem.writer_weight)
        * problem.endpoint_row_mass.unsqueeze(-1)
        / problem.output_width
    )
    context_cotangent = (
        context_weight
        * context_effect
        * problem.context_row_mass.unsqueeze(-1)
        / problem.hidden
    )
    return _context_vjp(problem, endpoint_cotangent + context_cotangent)


def _role_squared_column_sum(
    problem: _PackedBcxProblem,
    *,
    multiplier: Any,
) -> Any:
    """Exact per-row/channel Frobenius mass of a B- or x-role Jacobian."""

    torch = _torch()
    fir_length = int(problem.fir_kernel.shape[1])
    feature_shifts = []
    multiplier_shifts = []
    for kernel_index in range(fir_length):
        delay = fir_length - 1 - kernel_index
        feature_shifts.append(_causal_shift(problem.features, delay))
        multiplier_shifts.append(_causal_shift(multiplier, delay))
    result = torch.zeros_like(problem.b_history)
    for left in range(fir_length):
        left_kernel = problem.fir_kernel[:, left].reshape(1, 1, -1)
        for right in range(left, fir_length):
            right_kernel = problem.fir_kernel[:, right].reshape(1, 1, -1)
            feature_inner = (feature_shifts[left] * feature_shifts[right]).sum(
                dim=-1, keepdim=True
            )
            term = (
                feature_inner
                * multiplier_shifts[left]
                * multiplier_shifts[right]
                * left_kernel
                * right_kernel
            )
            result = result + term * (1.0 if left == right else 2.0)
    return (result * problem.c_history.square()).clamp(min=0.0)


@dataclass(frozen=True, slots=True)
class _Conditioning:
    ridge: float
    operator_upper_bound: float
    regularized_condition_upper_bound: float
    preconditioner_diagonal: Any
    role_channel_frobenius: Any
    writer_spectral_squared_upper_bound: float


def _conditioning(
    problem: _PackedBcxProblem,
    policy: Lfm2CpcfV5MatrixFreeBcxPolicy,
) -> _Conditioning:
    torch = _torch()
    b_squared = _role_squared_column_sum(
        problem,
        multiplier=problem.x_history,
    )
    x_squared = _role_squared_column_sum(
        problem,
        multiplier=problem.b_history,
    )
    feature_squared_norm = problem.features.square().sum(dim=-1, keepdim=True)
    c_squared = feature_squared_norm * problem.base_filtered.square()
    role_squared = torch.stack((b_squared, c_squared, x_squared), dim=0)
    endpoint_channel_frobenius = (
        role_squared * problem.endpoint_row_mass.unsqueeze(0).unsqueeze(-1)
    ).sum(dim=(0, 1, 2))
    context_channel_frobenius = (
        role_squared * problem.context_row_mass.unsqueeze(0).unsqueeze(-1)
    ).sum(dim=(0, 1, 2))
    writer_absolute = problem.writer_weight.abs()
    writer_norm_one = float(writer_absolute.sum(dim=0).max().item())
    writer_norm_infinity = float(writer_absolute.sum(dim=1).max().item())
    writer_spectral_squared_upper = writer_norm_one * writer_norm_infinity
    endpoint_upper = (
        writer_spectral_squared_upper
        * float(endpoint_channel_frobenius.max().item())
        / problem.output_width
    )
    context_upper = (
        policy.context_objective_weight
        * float(context_channel_frobenius.max().item())
        / problem.hidden
    )
    operator_upper = max(0.0, endpoint_upper + context_upper)
    ridge = max(
        policy.absolute_ridge_floor,
        operator_upper / (policy.condition_cap - 1.0),
    )
    condition_upper = (operator_upper + ridge) / ridge

    writer_column_squared_norm = problem.writer_weight.square().sum(dim=0)
    coefficient = (
        problem.endpoint_row_mass.unsqueeze(-1)
        * writer_column_squared_norm.reshape(1, 1, -1)
        / problem.output_width
        + policy.context_objective_weight
        * problem.context_row_mass.unsqueeze(-1)
        / problem.hidden
    )
    role_channel_frobenius = (role_squared * coefficient.unsqueeze(0)).sum(dim=(1, 2))
    channel_average_diagonal = role_channel_frobenius / problem.feature_width
    preconditioner = (
        channel_average_diagonal.unsqueeze(-1)
        .expand(3, problem.hidden, problem.feature_width)
        .clone()
        + ridge
    )
    preconditioner = preconditioner.clamp(min=ridge)
    return _Conditioning(
        ridge=ridge,
        operator_upper_bound=operator_upper,
        regularized_condition_upper_bound=condition_upper,
        preconditioner_diagonal=preconditioner,
        role_channel_frobenius=role_channel_frobenius,
        writer_spectral_squared_upper_bound=writer_spectral_squared_upper,
    )


def _dot(left: Any, right: Any) -> Any:
    return (left.reshape(-1) * right.reshape(-1)).sum()


@dataclass(frozen=True, slots=True)
class _PcgSolve:
    correction: Any
    residual_norms: tuple[float, ...]
    relative_residual_norms: tuple[float, ...]
    breakdown_iterations: tuple[int, ...]


def _fixed_pcg(
    problem: _PackedBcxProblem,
    *,
    rhs: Any,
    conditioning: _Conditioning,
    policy: Lfm2CpcfV5MatrixFreeBcxPolicy,
) -> _PcgSolve:
    torch = _torch()

    def normal(value: Any) -> Any:
        return (
            _normal_without_ridge(
                problem,
                value,
                context_weight=policy.context_objective_weight,
            )
            + conditioning.ridge * value
        )

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    inverse_diagonal = conditioning.preconditioner_diagonal.reciprocal()
    preconditioned = residual * inverse_diagonal
    direction = preconditioned.clone()
    residual_preconditioned = _dot(residual, preconditioned)
    initial_norm = float(torch.linalg.vector_norm(residual).item())
    denominator_floor = torch.finfo(problem.dtype).tiny
    residual_norms = [initial_norm]
    relative_residuals = [0.0 if initial_norm == 0.0 else 1.0]
    breakdowns = []
    active = initial_norm != 0.0
    for iteration in range(policy.pcg_iterations):
        if active:
            normal_direction = normal(direction)
            denominator = _dot(direction, normal_direction)
            denominator_value = float(denominator.item())
            if (
                not math.isfinite(denominator_value)
                or denominator_value <= denominator_floor
            ):
                active = False
                breakdowns.append(iteration)
            else:
                alpha = residual_preconditioned / denominator
                solution = solution + alpha * direction
                residual = residual - alpha * normal_direction
                next_preconditioned = residual * inverse_diagonal
                next_residual_preconditioned = _dot(
                    residual,
                    next_preconditioned,
                )
                previous_value = float(residual_preconditioned.item())
                if (
                    not math.isfinite(previous_value)
                    or abs(previous_value) <= denominator_floor
                ):
                    beta = torch.zeros(
                        (),
                        device=problem.device,
                        dtype=problem.dtype,
                    )
                else:
                    beta = next_residual_preconditioned / residual_preconditioned
                direction = next_preconditioned + beta * direction
                preconditioned = next_preconditioned
                residual_preconditioned = next_residual_preconditioned
        norm = float(torch.linalg.vector_norm(residual).item())
        residual_norms.append(norm)
        relative_residuals.append(0.0 if initial_norm == 0.0 else norm / initial_norm)
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "matrix-free PCG produced NaN or infinity."
        )
    return _PcgSolve(
        correction=solution,
        residual_norms=tuple(residual_norms),
        relative_residual_norms=tuple(relative_residuals),
        breakdown_iterations=tuple(breakdowns),
    )


def _weighted_effect_rms(
    problem: _PackedBcxProblem,
    context_effect: Any,
    *,
    context_weight: float,
) -> float:
    endpoint_effect = context_effect @ problem.writer_weight.transpose(0, 1)
    squared = (
        endpoint_effect.square().mean(dim=-1) * problem.endpoint_row_mass
    ).sum() + context_weight * (
        context_effect.square().mean(dim=-1) * problem.context_row_mass
    ).sum()
    return math.sqrt(max(0.0, float(squared.item())))


def _weighted_target_rms(
    problem: _PackedBcxProblem,
    *,
    context_weight: float,
) -> float:
    squared = (
        problem.endpoint_residual.square().mean(dim=-1) * problem.endpoint_row_mass
    ).sum() + context_weight * (
        problem.context_residual.square().mean(dim=-1) * problem.context_row_mass
    ).sum()
    return math.sqrt(max(0.0, float(squared.item())))


def _weighted_residual_objective(
    problem: _PackedBcxProblem,
    context_effect: Any | None,
    *,
    context_weight: float,
) -> float:
    if context_effect is None:
        endpoint_residual = problem.endpoint_residual
        context_residual = problem.context_residual
    else:
        endpoint_residual = (
            problem.endpoint_residual
            - context_effect @ problem.writer_weight.transpose(0, 1)
        )
        context_residual = problem.context_residual - context_effect
    return float(
        (endpoint_residual.square().mean(dim=-1) * problem.endpoint_row_mass)
        .sum()
        .item()
        + context_weight
        * (context_residual.square().mean(dim=-1) * problem.context_row_mass)
        .sum()
        .item()
    )


def solve_lfm2_cpcf_v5_matrix_free_bcx(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: Sequence[Lfm2CpcfV4VolterraReplay],
    *,
    in_proj_weight: Any,
    in_proj_bias: Any | None,
    fir_kernel: Any,
    fir_bias: Any | None,
    writer_weight: Any,
    policy: Lfm2CpcfV5MatrixFreeBcxPolicy | None = None,
) -> Lfm2CpcfV5MatrixFreeBcxResult:
    """Solve one globally coupled residual-aware dense B/C/x fold.

    The parent/replay evidence is consumed once to construct a packed causal
    operator.  PCG calls only implicit J/J-transpose products thereafter.
    """

    torch = _torch()
    solve_policy = policy or Lfm2CpcfV5MatrixFreeBcxPolicy()
    if not isinstance(solve_policy, Lfm2CpcfV5MatrixFreeBcxPolicy):
        raise TypeError("policy must be Lfm2CpcfV5MatrixFreeBcxPolicy.")
    if not solve_policy.verify_authentication():
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "matrix-free B/C/x policy failed authentication."
        )
    input_weight = _dense_floating_tensor(
        in_proj_weight,
        "in_proj_weight",
        dimensions=2,
    )
    input_bias = (
        None
        if in_proj_bias is None
        else _dense_floating_tensor(
            in_proj_bias,
            "in_proj_bias",
            dimensions=1,
        )
    )
    problem = _build_packed_problem(
        cases,
        replays,
        in_proj_weight=input_weight,
        in_proj_bias=input_bias,
        fir_kernel=fir_kernel,
        fir_bias=fir_bias,
        writer_weight=writer_weight,
        policy=solve_policy,
    )
    conditioning = _conditioning(problem, solve_policy)
    right = _rhs(problem, solve_policy.context_objective_weight)
    pcg = _fixed_pcg(
        problem,
        rhs=right,
        conditioning=conditioning,
        policy=solve_policy,
    )
    raw_effect = _context_jvp(problem, pcg.correction)
    raw_parameter_rms = float(pcg.correction.square().mean().sqrt().item())
    prior_parameter_rms = float(problem.prior.square().mean().sqrt().item())
    parameter_reference = max(
        prior_parameter_rms,
        solve_policy.parameter_reference_floor,
    )
    parameter_limit = solve_policy.parameter_trust_radius * parameter_reference
    raw_effect_rms = _weighted_effect_rms(
        problem,
        raw_effect,
        context_weight=solve_policy.context_objective_weight,
    )
    target_effect_rms = _weighted_target_rms(
        problem,
        context_weight=solve_policy.context_objective_weight,
    )
    effect_reference = max(
        target_effect_rms,
        solve_policy.effect_reference_floor,
    )
    effect_limit = solve_policy.effect_trust_radius * effect_reference
    parameter_scale = (
        1.0
        if raw_parameter_rms == 0.0
        else min(1.0, parameter_limit / raw_parameter_rms)
    )
    effect_scale = (
        1.0 if raw_effect_rms == 0.0 else min(1.0, effect_limit / raw_effect_rms)
    )
    joint_scale = min(parameter_scale, effect_scale)
    accepted = pcg.correction * joint_scale
    accepted_effect = raw_effect * joint_scale
    updated = problem.prior + accepted

    hidden = problem.hidden
    updated_weight = (
        updated[..., :hidden]
        .reshape(3 * hidden, hidden)
        .detach()
        .to(device="cpu", dtype=input_weight.dtype)
        .contiguous()
    )
    updated_bias = None
    if input_bias is not None:
        updated_bias = (
            updated[..., -1]
            .reshape(3 * hidden)
            .detach()
            .to(device="cpu", dtype=input_bias.dtype)
            .contiguous()
        )
    raw_correction = (
        pcg.correction.detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )
    accepted_correction = (
        accepted.detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )
    before_objective = _weighted_residual_objective(
        problem,
        None,
        context_weight=solve_policy.context_objective_weight,
    )
    after_objective = _weighted_residual_objective(
        problem,
        accepted_effect,
        context_weight=solve_policy.context_objective_weight,
    )
    input_state_tensors = {
        "in_proj_weight": _tensor_sha256(
            input_weight,
            "input_state.in_proj_weight",
        ),
        "fir_kernel": _tensor_sha256(
            fir_kernel,
            "input_state.fir_kernel",
        ),
        "writer_weight": _tensor_sha256(
            writer_weight,
            "input_state.writer_weight",
        ),
    }
    if input_bias is not None:
        input_state_tensors["in_proj_bias"] = _tensor_sha256(
            input_bias,
            "input_state.in_proj_bias",
        )
    if fir_bias is not None:
        input_state_tensors["fir_bias"] = _tensor_sha256(
            fir_bias,
            "input_state.fir_bias",
        )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_MATRIX_FREE_BCX_SCHEMA,
        "kind": "global_matrix_free_bcx_result",
        "implementation": _IMPLEMENTATION,
        "policy_sha256": solve_policy.authentication_sha256,
        "case_count": len(problem.case_inventory),
        "case_inventory": [dict(value) for value in problem.case_inventory],
        "packed_trajectory_count": int(problem.features.shape[0]),
        "packed_maximum_sequence_length": int(problem.features.shape[1]),
        "hidden_size": hidden,
        "feature_width": problem.feature_width,
        "physical_bias_coordinates_present": problem.fit_bias,
        "physical_correction_coordinate_count": int(problem.prior.numel()),
        "expected_physical_correction_coordinate_count": (
            3 * hidden * (hidden + int(problem.fit_bias))
        ),
        "all_physical_correction_coordinates_preserved": (
            int(problem.prior.numel()) == 3 * hidden * (hidden + int(problem.fit_bias))
        ),
        "dense_transported_prior_preserved": True,
        "input_state_tensor_sha256": input_state_tensors,
        "input_state_sha256": _sha256(input_state_tensors),
        "writer_output_width": problem.output_width,
        "global_writer_channel_coupling": True,
        "endpoint_operator": "delta_context_at_full_writer_transpose",
        "context_operator": "smaller_phase_balanced_context_residual",
        "row_mass_contract": {
            "equal_case_total_mass": True,
            "uniform_fraction": solve_policy.uniform_horizon_fraction,
            "exponential_tail_fraction": (1.0 - solve_policy.uniform_horizon_fraction),
            "tail_tau_formula": ("max(tail_tau_floor,sqrt(valid_rows_per_trajectory))"),
            "tail_tau_floor": solve_policy.tail_tau_floor,
            "suffix_start_used_as_decode_boundary": False,
        },
        "solve": {
            "method": "fixed_schedule_diagonally_preconditioned_conjugate_gradient",
            "backend": problem.device.type,
            "compute_dtype": _dtype_name(problem.features),
            "fixed_iteration_count": solve_policy.pcg_iterations,
            "early_stopping": False,
            "residual_norms": list(pcg.residual_norms),
            "relative_residual_norms": list(pcg.relative_residual_norms),
            "breakdown_iterations": list(pcg.breakdown_iterations),
            "normal_design_materialized": False,
            "gram_materialized": False,
            "autograd_used": False,
            "preconditioner": "role_channel_energy_diagonal",
            "preconditioner_minimum": float(
                conditioning.preconditioner_diagonal.min().item()
            ),
            "preconditioner_maximum": float(
                conditioning.preconditioner_diagonal.max().item()
            ),
        },
        "conditioning": {
            "condition_cap": solve_policy.condition_cap,
            "operator_spectral_squared_upper_bound": (
                conditioning.operator_upper_bound
            ),
            "writer_spectral_squared_upper_bound": (
                conditioning.writer_spectral_squared_upper_bound
            ),
            "writer_bound_formula": "matrix_one_norm_times_matrix_infinity_norm",
            "channel_jacobian_bound": (
                "maximum_weighted_per_channel_frobenius_squared"
            ),
            "effective_ridge": conditioning.ridge,
            "ridge_formula": "operator_upper_bound/(condition_cap-1)",
            "regularized_condition_upper_bound": (
                conditioning.regularized_condition_upper_bound
            ),
            "condition_cap_passed": (
                conditioning.regularized_condition_upper_bound
                <= solve_policy.condition_cap * (1.0 + 1e-12)
            ),
            "role_channel_frobenius_sha256": _tensor_sha256(
                conditioning.role_channel_frobenius,
                "conditioning.role_channel_frobenius",
            ),
        },
        "parameter_space_trust": {
            "reference_rms": parameter_reference,
            "relative_radius": solve_policy.parameter_trust_radius,
            "absolute_limit_rms": parameter_limit,
            "raw_correction_rms": raw_parameter_rms,
            "individual_scale": parameter_scale,
            "accepted_correction_rms": float(accepted.square().mean().sqrt().item()),
        },
        "function_space_trust": {
            "reference_rms": effect_reference,
            "relative_radius": solve_policy.effect_trust_radius,
            "absolute_limit_rms": effect_limit,
            "raw_correction_effect_rms": raw_effect_rms,
            "individual_scale": effect_scale,
            "accepted_correction_effect_rms": _weighted_effect_rms(
                problem,
                accepted_effect,
                context_weight=solve_policy.context_objective_weight,
            ),
        },
        "joint_trust_scale": joint_scale,
        "weighted_fit_objective_before": before_objective,
        "weighted_fit_objective_after": after_objective,
        "weighted_fit_objective_improved": after_objective <= before_objective,
        "raw_correction_sha256": _tensor_sha256(
            raw_correction,
            "result.raw_correction",
        ),
        "accepted_correction_sha256": _tensor_sha256(
            accepted_correction,
            "result.accepted_correction",
        ),
        "updated_in_proj_weight_sha256": _tensor_sha256(
            updated_weight,
            "result.in_proj_weight",
        ),
        "updated_in_proj_bias_sha256": (
            None
            if updated_bias is None
            else _tensor_sha256(updated_bias, "result.in_proj_bias")
        ),
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "execution": {
            **_ZERO_EXECUTION,
            "autograd_graphs": 0,
            "implicit_jvp_calls": solve_policy.pcg_iterations + 1,
            "implicit_vjp_calls": solve_policy.pcg_iterations + 1,
            "fixed_iterative_linear_solve": True,
        },
    }
    if body["conditioning"]["condition_cap_passed"] is not True:
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "conservative ridge failed its declared condition cap."
        )
    return Lfm2CpcfV5MatrixFreeBcxResult(
        in_proj_weight=updated_weight,
        in_proj_bias=updated_bias,
        raw_correction=raw_correction,
        accepted_correction=accepted_correction,
        diagnostics=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5MatrixFreeBcxOracle:
    """Small exact dense design constructed from the legacy channel Jacobian."""

    design: Any
    target: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "design",
            _dense_floating_tensor(
                self.design,
                "oracle.design",
                dimensions=2,
            )
            .to(device="cpu", dtype=_torch().float64)
            .contiguous()
            .clone(),
        )
        object.__setattr__(
            self,
            "target",
            _dense_floating_tensor(
                self.target,
                "oracle.target",
                dimensions=1,
            )
            .to(device="cpu", dtype=_torch().float64)
            .contiguous()
            .clone(),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_cpcf_short_conv_v5_matrix_free_bcx_report(self.report)
            and self.report.get("design_sha256")
            == _tensor_sha256(self.design, "oracle.design")
            and self.report.get("target_sha256")
            == _tensor_sha256(self.target, "oracle.target")
        )


def materialize_lfm2_cpcf_v5_bcx_legacy_oracle(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: Sequence[Lfm2CpcfV4VolterraReplay],
    *,
    in_proj_weight: Any,
    in_proj_bias: Any | None,
    fir_kernel: Any,
    fir_bias: Any | None,
    writer_weight: Any,
    context_objective_weight: float = 0.05,
    uniform_horizon_fraction: float = 0.5,
    tail_tau_floor: float = 4.0,
    maximum_parameters: int = 4096,
) -> Lfm2CpcfV5MatrixFreeBcxOracle:
    """Materialize the exact global legacy Jacobian for a small oracle only."""

    torch = _torch()
    pairs = _canonical_case_replays(cases, replays)
    weight = _dense_floating_tensor(
        in_proj_weight,
        "in_proj_weight",
        dimensions=2,
    ).to(device="cpu", dtype=torch.float64)
    hidden = int(weight.shape[1])
    fit_bias = in_proj_bias is not None
    feature_width = hidden + int(fit_bias)
    parameters = 3 * hidden * feature_width
    if (
        isinstance(maximum_parameters, bool)
        or not isinstance(maximum_parameters, int)
        or maximum_parameters <= 0
        or parameters > maximum_parameters
    ):
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "legacy oracle parameter count exceeds maximum_parameters."
        )
    context_weight = _positive_float(
        context_objective_weight,
        "context_objective_weight",
        allow_zero=True,
    )
    writer = _dense_floating_tensor(
        writer_weight,
        "writer_weight",
        dimensions=2,
    ).to(device="cpu", dtype=torch.float64)
    output_width = int(writer.shape[0])
    case_scale = 1.0 / len(pairs)
    design_blocks = []
    target_blocks = []
    case_inventory = []
    for case, replay in pairs:
        batch, sequence, _case_hidden = map(
            int,
            case.child_input_history.shape,
        )
        prefix_mask = (
            torch.ones((batch, sequence), dtype=torch.bool)
            if case.valid_prefix_mask is None
            else case.valid_prefix_mask.to(device="cpu", dtype=torch.bool)
        )
        suffix_mask = (
            prefix_mask[:, case.suffix_start :]
            if case.valid_suffix_mask is None
            else case.valid_suffix_mask.to(device="cpu", dtype=torch.bool)
        )
        endpoint_mask = torch.zeros_like(prefix_mask)
        endpoint_mask[:, case.suffix_start :] = suffix_mask
        endpoint_mass = (
            causal_horizon_row_mass(
                endpoint_mask,
                uniform_fraction=uniform_horizon_fraction,
                tail_tau_floor=tail_tau_floor,
                dtype=torch.float64,
            )
            * case_scale
        )
        context_mass = (
            causal_horizon_row_mass(
                prefix_mask,
                uniform_fraction=uniform_horizon_fraction,
                tail_tau_floor=tail_tau_floor,
                dtype=torch.float64,
            )
            * case_scale
        )
        channel_designs = torch.stack(
            tuple(
                linearized_bcx_affine_design(
                    case.child_input_history,
                    replay.b_history,
                    replay.c_history,
                    replay.x_history,
                    fir_kernel,
                    fir_bias,
                    channel=channel,
                    fit_bias=fit_bias,
                )
                for channel in range(hidden)
            ),
            dim=2,
        )
        context_residual = case.desired_context_history.to(
            dtype=torch.float64
        ) - replay.context_history.to(dtype=torch.float64)
        endpoint_residual = torch.zeros(
            (batch, sequence, output_width),
            dtype=torch.float64,
        )
        endpoint_residual[:, case.suffix_start :] = case.desired_write_suffix.to(
            dtype=torch.float64
        ) - replay.write_history[:, case.suffix_start :].to(dtype=torch.float64)
        for batch_index in range(batch):
            for time_index in range(sequence):
                context_design = torch.zeros(
                    (hidden, parameters),
                    dtype=torch.float64,
                )
                for channel in range(hidden):
                    local = channel_designs[
                        batch_index,
                        time_index,
                        channel,
                    ]
                    for role in range(3):
                        start = role * hidden * feature_width + channel * feature_width
                        context_design[
                            channel,
                            start : start + feature_width,
                        ] = local[role * feature_width : (role + 1) * feature_width]
                endpoint_weight = float(endpoint_mass[batch_index, time_index].item())
                if endpoint_weight > 0.0:
                    scale = math.sqrt(endpoint_weight / output_width)
                    design_blocks.append(scale * (writer @ context_design))
                    target_blocks.append(
                        scale * endpoint_residual[batch_index, time_index]
                    )
                context_row_weight = float(context_mass[batch_index, time_index].item())
                if context_weight > 0.0 and context_row_weight > 0.0:
                    scale = math.sqrt(context_weight * context_row_weight / hidden)
                    design_blocks.append(scale * context_design)
                    target_blocks.append(
                        scale * context_residual[batch_index, time_index]
                    )
        case_inventory.append(
            {
                "case_id": case.case_id,
                "batch": batch,
                "sequence": sequence,
                "suffix_start": case.suffix_start,
            }
        )
    design = torch.cat(design_blocks, dim=0).contiguous()
    target = torch.cat(target_blocks, dim=0).contiguous()
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_MATRIX_FREE_BCX_SCHEMA,
        "kind": "small_exact_legacy_global_bcx_oracle",
        "case_inventory": case_inventory,
        "physical_coordinate_count": parameters,
        "design_rows": int(design.shape[0]),
        "design_columns": int(design.shape[1]),
        "coordinate_order": "role_then_channel_then_affine_feature",
        "legacy_per_channel_design_function": (
            "hf_lfm2_cpcf_short_conv_v4.linearized_bcx_affine_design"
        ),
        "global_writer_channel_coupling": True,
        "context_objective_weight": context_weight,
        "uniform_horizon_fraction": uniform_horizon_fraction,
        "tail_tau_floor": tail_tau_floor,
        "design_sha256": _tensor_sha256(design, "oracle.design"),
        "target_sha256": _tensor_sha256(target, "oracle.target"),
        "production_use": False,
        "small_case_oracle_only": True,
        "execution": {
            **_ZERO_EXECUTION,
            "dense_design_materialized_for_oracle": True,
            "dense_gram_materialized": False,
        },
    }
    return Lfm2CpcfV5MatrixFreeBcxOracle(
        design=design,
        target=target,
        report=_seal_report(body),
    )


def compare_lfm2_cpcf_v5_matrix_free_bcx_to_legacy_oracle(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: Sequence[Lfm2CpcfV4VolterraReplay],
    *,
    in_proj_weight: Any,
    in_proj_bias: Any | None,
    fir_kernel: Any,
    fir_bias: Any | None,
    writer_weight: Any,
    policy: Lfm2CpcfV5MatrixFreeBcxPolicy | None = None,
    maximum_parameters: int = 4096,
) -> Mapping[str, Any]:
    """Compare fixed PCG with an exact dense legacy solve on a small problem."""

    torch = _torch()
    solve_policy = policy or Lfm2CpcfV5MatrixFreeBcxPolicy(
        backend="cpu",
        pcg_iterations=64,
    )
    if solve_policy.backend != "cpu":
        raise Lfm2CpcfShortConvV5MatrixFreeBcxError(
            "exact legacy comparison requires backend='cpu'."
        )
    result = solve_lfm2_cpcf_v5_matrix_free_bcx(
        cases,
        replays,
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        fir_kernel=fir_kernel,
        fir_bias=fir_bias,
        writer_weight=writer_weight,
        policy=solve_policy,
    )
    oracle = materialize_lfm2_cpcf_v5_bcx_legacy_oracle(
        cases,
        replays,
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        fir_kernel=fir_kernel,
        fir_bias=fir_bias,
        writer_weight=writer_weight,
        context_objective_weight=solve_policy.context_objective_weight,
        uniform_horizon_fraction=solve_policy.uniform_horizon_fraction,
        tail_tau_floor=solve_policy.tail_tau_floor,
        maximum_parameters=maximum_parameters,
    )
    ridge = float(result.diagnostics["conditioning"]["effective_ridge"])
    gram = oracle.design.T @ oracle.design
    right = oracle.design.T @ oracle.target
    exact = torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        right,
    )
    actual = result.raw_correction.reshape(-1)
    difference = actual - exact
    relative = float(torch.linalg.vector_norm(difference).item()) / max(
        float(torch.linalg.vector_norm(exact).item()),
        torch.finfo(torch.float64).eps,
    )
    exact_residual = gram @ exact + ridge * exact - right
    actual_residual = gram @ actual + ridge * actual - right
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_MATRIX_FREE_BCX_SCHEMA,
        "kind": "matrix_free_to_exact_legacy_oracle_comparison",
        "policy_sha256": solve_policy.authentication_sha256,
        "matrix_free_result_report_sha256": (result.diagnostics["report_sha256"]),
        "legacy_oracle_report_sha256": oracle.report["report_sha256"],
        "physical_coordinate_count": int(exact.numel()),
        "effective_ridge": ridge,
        "maximum_absolute_correction_error": float(difference.abs().max().item()),
        "relative_correction_error": relative,
        "exact_normal_residual_norm": float(
            torch.linalg.vector_norm(exact_residual).item()
        ),
        "matrix_free_normal_residual_norm": float(
            torch.linalg.vector_norm(actual_residual).item()
        ),
        "raw_matrix_free_correction_sha256": _tensor_sha256(
            result.raw_correction,
            "comparison.raw_matrix_free_correction",
        ),
        "exact_legacy_correction_sha256": _tensor_sha256(
            exact,
            "comparison.exact_legacy_correction",
        ),
        "fit_only": True,
        "execution": {
            **_ZERO_EXECUTION,
            "dense_design_and_gram_used_for_small_oracle_only": True,
            "production_solver_remained_matrix_free": True,
        },
    }
    return _seal_report(body)


__all__ = [
    "LFM2_CPCF_SHORT_CONV_V5_MATRIX_FREE_BCX_SCHEMA",
    "Lfm2CpcfShortConvV5MatrixFreeBcxError",
    "Lfm2CpcfV5MatrixFreeBcxOracle",
    "Lfm2CpcfV5MatrixFreeBcxPolicy",
    "Lfm2CpcfV5MatrixFreeBcxResult",
    "causal_horizon_row_mass",
    "compare_lfm2_cpcf_v5_matrix_free_bcx_to_legacy_oracle",
    "materialize_lfm2_cpcf_v5_bcx_legacy_oracle",
    "solve_lfm2_cpcf_v5_matrix_free_bcx",
    "verify_lfm2_cpcf_short_conv_v5_matrix_free_bcx_report",
]
