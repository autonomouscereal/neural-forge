"""Scalable endpoint-aware B/C/x residual-subspace folding for CPCF v5.

The original v5 numerical adapter solves a separate ``3H``-wide dense ridge
problem for every target channel.  At production width this repeatedly forms
and diagonalizes thousands of large FP64 systems.  This module keeps the
transported dense input projection as an additive prior and restricts only its
*correction* to a dynamically derived child-feature subspace.

The correction basis combines four fit-only signals:

* case-balanced child activation covariance;
* current context-residual cross covariance;
* current suffix residual-write cross covariance through the writer; and
* the transported dense prior row space.

The physical bias coordinate is never projected.  Endpoint residual writes are
the primary objective, with a smaller context residual term.  Because writer
columns couple context channels, channels are updated in deterministic
coordinate order and the residual write is refreshed after every accepted
coordinate.  Channel designs and Grams are still produced in GPU-sized
microbatches.

This is a closed-form folding primitive.  It has no gradients, optimizers,
training steps, pruning, quantization, sparsity, or parameter sharing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfV4ProjectionCase,
    Lfm2CpcfV4VolterraReplay,
)
from .hf_lfm2_cpcf_short_conv_v5 import Lfm2CpcfV5RidgePolicy


LFM2_CPCF_V5_RESIDUAL_SUBSPACE_BCX_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-residual-subspace-bcx"
)
_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_cpcf_short_conv_v5_residual_subspace_bcx."
    "solve_lfm2_cpcf_v5_residual_subspace_bcx.v1"
)
_REQUIRED_STATE_ROLES = (
    "in_proj_weight",
    "in_proj_bias",
    "fir_kernel",
    "fir_bias",
    "writer_weight",
    "writer_bias",
)
_ZERO_EXECUTION = {
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "sparse_parameters_created": 0,
    "shared_parameters_created": 0,
}


class Lfm2CpcfV5ResidualSubspaceBcxError(ValueError):
    """Residual-subspace BCX evidence or geometry is inconsistent."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for residual-subspace CPCF folding."
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
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "Residual-subspace metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> dict[str, Any]:
    detached = json.loads(_canonical_json(dict(body)).decode("ascii"))
    return {**detached, "report_sha256": _sha256(detached)}


def _verify_report(report: Mapping[str, Any]) -> bool:
    try:
        body = dict(report)
        supplied = str(body.pop("report_sha256")).strip().lower()
        return supplied == _sha256(body)
    except (KeyError, TypeError, ValueError):
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
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"{name} must be a materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(
        tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def _finite_dense(
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
    ):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"{name} must be a nonempty dense real rank-{dimensions} tensor."
        )
    result = value.detach()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"{name} contains NaN or infinity."
        )
    return result


def _positive_float(
    value: Any,
    name: str,
    *,
    allow_zero: bool = False,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"{name} must be finite numeric input."
        ) from exc
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"{name} must be finite and {qualifier}."
        )
    return result


def automatic_residual_subspace_rank(
    hidden_size: int,
    *,
    maximum_rank: int = 64,
    full_rank_hidden_threshold: int = 32,
) -> int:
    """Return the deterministic dynamic correction rank.

    Small targets retain an exact full-rank correction.  Larger targets use
    the next power of two above ``sqrt(H)``, capped at 64 by default.
    """

    for value, name in (
        (hidden_size, "hidden_size"),
        (maximum_rank, "maximum_rank"),
        (full_rank_hidden_threshold, "full_rank_hidden_threshold"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                f"{name} must be a positive integer."
            )
    if hidden_size <= full_rank_hidden_threshold:
        return hidden_size
    root = max(1, int(math.ceil(math.sqrt(hidden_size))))
    power = 1 << (root - 1).bit_length()
    return min(hidden_size, maximum_rank, power)


def causal_horizon_balanced_row_weights(
    valid_rows: Any,
    *,
    dtype: Any | None = None,
    uniform_mass: float = 0.5,
    minimum_tail_tau: float = 4.0,
) -> Any:
    """Balance a causal trajectory without unavailable phase labels.

    CPCF projection cases collapse prefill and all cached decode phases into a
    single causal history.  Exact phase boundaries therefore cannot be
    recovered here.  This deterministic proxy gives half of each case's mass
    uniformly to all valid rows and half through an exponentially tail-weighted
    distribution.  The dynamic time constant is
    ``max(minimum_tail_tau, sqrt(valid_row_count))``.

    The returned weights always sum to one, so callers obtain case balance by
    multiplying each case by ``1 / case_count``.
    """

    torch = _torch()
    if (
        not isinstance(valid_rows, torch.Tensor)
        or valid_rows.layout != torch.strided
        or valid_rows.device.type == "meta"
        or valid_rows.ndim != 2
        or valid_rows.numel() == 0
    ):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "valid_rows must be a nonempty dense rank-2 mask."
        )
    mass = _positive_float(
        uniform_mass,
        "uniform_mass",
        allow_zero=True,
    )
    if mass > 1.0:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "uniform_mass must be between zero and one."
        )
    minimum_tau = _positive_float(minimum_tail_tau, "minimum_tail_tau")
    mask = valid_rows.detach().to(dtype=torch.bool)
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "valid_rows must contain at least one valid row."
        )
    result_dtype = dtype
    if result_dtype is None:
        result_dtype = (
            valid_rows.dtype
            if torch.is_floating_point(valid_rows)
            else torch.float64
        )
    if not torch.empty((), dtype=result_dtype).is_floating_point():
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "causal row weights require a floating-point dtype."
        )
    mask_value = mask.to(dtype=result_dtype)
    uniform = mask_value / float(valid_count)
    sequence = int(mask.shape[1])
    positions = torch.arange(
        sequence,
        device=mask.device,
        dtype=result_dtype,
    ).reshape(1, sequence)
    masked_positions = torch.where(
        mask,
        positions.expand(int(mask.shape[0]), -1),
        torch.full(
            tuple(mask.shape),
            -1.0,
            device=mask.device,
            dtype=result_dtype,
        ),
    )
    last = masked_positions.max(dim=1).values.reshape(-1, 1)
    tau = max(minimum_tau, math.sqrt(valid_count))
    tail = torch.exp((positions - last) / tau) * mask_value
    tail = tail / tail.sum()
    result = mass * uniform + (1.0 - mass) * tail
    return result.contiguous()


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5ResidualSubspaceBcxResult:
    """Dense updated BCX projection and its authenticated solver receipts."""

    in_proj_weight: Any
    in_proj_bias: Any | None
    receipt_sha256: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        weight = _finite_dense(
            self.in_proj_weight,
            "result.in_proj_weight",
            dimensions=2,
        ).to(device="cpu").contiguous().clone()
        bias = None
        if self.in_proj_bias is not None:
            bias = (
                _finite_dense(
                    self.in_proj_bias,
                    "result.in_proj_bias",
                    dimensions=1,
                )
                .to(device="cpu")
                .contiguous()
                .clone()
            )
        receipts = tuple(str(value).strip().lower() for value in self.receipt_sha256)
        if not receipts or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in receipts
        ):
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                "receipt_sha256 must contain lowercase SHA-256 digests."
            )
        object.__setattr__(self, "in_proj_weight", weight)
        object.__setattr__(self, "in_proj_bias", bias)
        object.__setattr__(self, "receipt_sha256", receipts)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )

    def verify_authentication(self) -> bool:
        report = self.diagnostics
        channel_reports = report.get("channel_reports", ())
        nested_receipts = tuple(
            str(value.get("report_sha256", "")) for value in channel_reports
        )
        expected_receipts = (
            str(report.get("basis_report", {}).get("report_sha256", "")),
            *nested_receipts,
            str(report.get("report_sha256", "")),
        )
        return (
            _verify_report(report)
            and all(_verify_report(value) for value in channel_reports)
            and _verify_report(report.get("basis_report", {}))
            and self.receipt_sha256 == expected_receipts
            and report.get("solution_weight_sha256")
            == _tensor_sha256(self.in_proj_weight, "result.in_proj_weight")
            and (
                (
                    self.in_proj_bias is None
                    and report.get("solution_bias_sha256") is None
                )
                or (
                    self.in_proj_bias is not None
                    and report.get("solution_bias_sha256")
                    == _tensor_sha256(self.in_proj_bias, "result.in_proj_bias")
                )
            )
        )


@dataclass(slots=True)
class _PackedEvidence:
    child: Any
    b_history: Any
    c_history: Any
    x_history: Any
    context: Any
    context_residual: Any
    write_residual: Any
    prefix_weights: Any
    suffix_weights: Any
    prefix_mask: Any
    case_reports: tuple[dict[str, Any], ...]
    case_authentication_sha256: tuple[str, ...]


def _resolve_device(
    requested: str | None,
) -> tuple[Any, Any, str]:
    torch = _torch()
    name = "auto" if requested is None else str(requested).strip().lower()
    if name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(name)
        except (TypeError, RuntimeError) as exc:
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                "compute_device must be auto, cpu, or an available CUDA device."
            ) from exc
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"Requested CUDA device {device} is unavailable."
        )
    if device.type not in {"cpu", "cuda"}:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "Residual-subspace BCX supports CPU or CUDA compute only."
        )
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    return device, dtype, str(device)


@contextmanager
def _deterministic_compute_scope(device: Any):
    """Disable approximate CUDA matmul while restoring global flags."""

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


def _state_tensor(
    state: Mapping[str, Any | None],
    role: str,
    *,
    dimensions: int,
    optional: bool = False,
) -> Any | None:
    if role not in state:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"state is missing required role {role!r}."
        )
    value = state[role]
    if value is None:
        if optional:
            return None
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"state role {role!r} cannot be absent."
        )
    return _finite_dense(value, f"state.{role}", dimensions=dimensions)


def _ordered_case_replays(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: (
        Sequence[Lfm2CpcfV4VolterraReplay]
        | Mapping[str, Lfm2CpcfV4VolterraReplay]
    ),
) -> tuple[tuple[Lfm2CpcfV4ProjectionCase, Lfm2CpcfV4VolterraReplay], ...]:
    values = tuple(cases)
    if not values or any(
        not isinstance(case, Lfm2CpcfV4ProjectionCase) for case in values
    ):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "cases must contain projection-case values."
        )
    if len({case.case_id for case in values}) != len(values):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "projection case ids must be unique."
        )
    if any(case.split != "fit" or not case.verify_authentication() for case in values):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "Residual-subspace BCX accepts authenticated FIT cases only."
        )
    if isinstance(replays, Mapping):
        if set(replays) != {case.case_id for case in values}:
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                "replay mapping must exactly cover projection case ids."
            )
        pairs = tuple((case, replays[case.case_id]) for case in values)
    else:
        replay_values = tuple(replays)
        if len(replay_values) != len(values):
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                "cases and replay sequence lengths differ."
            )
        pairs = tuple(zip(values, replay_values))
    if any(
        not isinstance(replay, Lfm2CpcfV4VolterraReplay)
        or not replay.verify_authentication()
        for _case, replay in pairs
    ):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "replays must be authenticated Volterra replay values."
        )
    return tuple(sorted(pairs, key=lambda value: value[0].case_id))


def _pack_evidence(
    pairs: Sequence[
        tuple[Lfm2CpcfV4ProjectionCase, Lfm2CpcfV4VolterraReplay]
    ],
    *,
    hidden_size: int,
    output_width: int,
    device: Any,
    dtype: Any,
) -> _PackedEvidence:
    torch = _torch()
    trajectory_count = sum(
        int(case.child_input_history.shape[0]) for case, _replay in pairs
    )
    maximum_tokens = max(
        int(case.child_input_history.shape[1]) for case, _replay in pairs
    )
    shape = (trajectory_count, maximum_tokens, hidden_size)
    child = torch.zeros(shape, device=device, dtype=dtype)
    b_history = torch.zeros_like(child)
    c_history = torch.zeros_like(child)
    x_history = torch.zeros_like(child)
    context = torch.zeros_like(child)
    context_residual = torch.zeros_like(child)
    write_residual = torch.zeros(
        (trajectory_count, maximum_tokens, output_width),
        device=device,
        dtype=dtype,
    )
    prefix_weights = torch.zeros(
        (trajectory_count, maximum_tokens),
        device=device,
        dtype=dtype,
    )
    suffix_weights = torch.zeros_like(prefix_weights)
    prefix_mask = torch.zeros(
        (trajectory_count, maximum_tokens),
        device=device,
        dtype=torch.bool,
    )
    case_count = len(pairs)
    case_reports = []
    cursor = 0
    for case, replay in pairs:
        batch, tokens, child_width = map(int, case.child_input_history.shape)
        if child_width != hidden_size:
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                f"{case.case_id} child width differs from in_proj width."
            )
        if (
            tuple(replay.b_history.shape) != (batch, tokens, hidden_size)
            or tuple(replay.c_history.shape) != (batch, tokens, hidden_size)
            or tuple(replay.x_history.shape) != (batch, tokens, hidden_size)
            or tuple(replay.context_history.shape) != (batch, tokens, hidden_size)
            or tuple(case.desired_context_history.shape)
            != (batch, tokens, hidden_size)
            or tuple(replay.write_history.shape)
            != (batch, tokens, output_width)
        ):
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                f"{case.case_id} replay geometry differs from its projection case."
            )
        suffix_tokens = tokens - case.suffix_start
        if tuple(case.desired_write_suffix.shape) != (
            batch,
            suffix_tokens,
            output_width,
        ):
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                f"{case.case_id} desired write suffix has the wrong geometry."
            )
        prefix = case.valid_prefix_mask
        if prefix is None:
            prefix = torch.ones((batch, tokens), dtype=torch.bool)
        suffix = case.valid_suffix_mask
        if suffix is None:
            suffix = prefix[:, case.suffix_start :]
        if tuple(prefix.shape) != (batch, tokens) or tuple(suffix.shape) != (
            batch,
            suffix_tokens,
        ):
            raise Lfm2CpcfV5ResidualSubspaceBcxError(
                f"{case.case_id} validity masks have the wrong geometry."
            )
        full_suffix = torch.zeros((batch, tokens), dtype=torch.bool)
        full_suffix[:, case.suffix_start :] = suffix.to(device="cpu")
        prefix_case_weights = causal_horizon_balanced_row_weights(
            prefix.to(device=device),
            dtype=dtype,
        ) / case_count
        suffix_case_weights = causal_horizon_balanced_row_weights(
            full_suffix.to(device=device),
            dtype=dtype,
        ) / case_count
        selection = slice(cursor, cursor + batch)

        def copy(value: Any) -> Any:
            return value.detach().to(device=device, dtype=dtype)

        child[selection, :tokens] = copy(case.child_input_history)
        b_history[selection, :tokens] = copy(replay.b_history)
        c_history[selection, :tokens] = copy(replay.c_history)
        x_history[selection, :tokens] = copy(replay.x_history)
        context[selection, :tokens] = copy(replay.context_history)
        context_residual[selection, :tokens] = (
            copy(case.desired_context_history) - copy(replay.context_history)
        )
        write_residual[selection, case.suffix_start : tokens] = (
            copy(case.desired_write_suffix)
            - copy(replay.write_history[:, case.suffix_start :, :])
        )
        prefix_weights[selection, :tokens] = prefix_case_weights
        suffix_weights[selection, :tokens] = suffix_case_weights
        prefix_mask[selection, :tokens] = prefix.to(
            device=device,
            dtype=torch.bool,
        )
        valid_prefix_rows = int(prefix.sum().item())
        valid_suffix_rows = int(suffix.sum().item())
        case_reports.append(
            {
                "case_id": case.case_id,
                "case_authentication_sha256": case.authentication_sha256,
                "replay_report_sha256": replay.arithmetic_report["report_sha256"],
                "batch_size": batch,
                "causal_tokens": tokens,
                "suffix_start": case.suffix_start,
                "valid_prefix_rows": valid_prefix_rows,
                "valid_suffix_rows": valid_suffix_rows,
                "case_total_prefix_mass": float(
                    prefix_case_weights.sum().item()
                ),
                "case_total_suffix_mass": float(
                    suffix_case_weights.sum().item()
                ),
                "uniform_mass": 0.5,
                "tail_mass": 0.5,
                "prefix_tail_tau": max(4.0, math.sqrt(valid_prefix_rows)),
                "suffix_tail_tau": max(4.0, math.sqrt(valid_suffix_rows)),
            }
        )
        cursor += batch
    return _PackedEvidence(
        child=child,
        b_history=b_history,
        c_history=c_history,
        x_history=x_history,
        context=context,
        context_residual=context_residual,
        write_residual=write_residual,
        prefix_weights=prefix_weights,
        suffix_weights=suffix_weights,
        prefix_mask=prefix_mask,
        case_reports=tuple(case_reports),
        case_authentication_sha256=tuple(
            case.authentication_sha256 for case, _replay in pairs
        ),
    )


def _trace_normalized(value: Any) -> tuple[Any, float]:
    torch = _torch()
    symmetric = 0.5 * (value + value.T)
    trace = float(torch.trace(symmetric).item())
    if trace <= float(torch.finfo(value.dtype).eps):
        return torch.zeros_like(symmetric), trace
    return symmetric / trace, trace


def _canonicalize_basis_signs(basis: Any) -> Any:
    torch = _torch()
    columns = int(basis.shape[1])
    pivot_rows = basis.abs().argmax(dim=0)
    pivot_values = basis[pivot_rows, torch.arange(columns, device=basis.device)]
    signs = torch.where(
        pivot_values < 0.0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return (basis * signs).contiguous()


def _derive_basis(
    packed: _PackedEvidence,
    *,
    prior_weight: Any,
    endpoint_projection: Any,
    rank: int,
    full_rank_threshold: int,
    compute_device: str,
    compute_dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    hidden = int(packed.child.shape[-1])
    child = packed.child * packed.prefix_mask.unsqueeze(-1).to(compute_dtype)
    covariance = torch.einsum(
        "mth,mt,mtk->hk",
        child,
        packed.prefix_weights,
        child,
    )
    context_cross = torch.einsum(
        "mth,mt,mtk->hk",
        child,
        packed.prefix_weights,
        packed.context_residual,
    )
    endpoint_cross = torch.einsum(
        "mth,mt,mtk->hk",
        child,
        packed.suffix_weights,
        endpoint_projection,
    )
    prior_gram = prior_weight.T @ prior_weight / int(prior_weight.shape[0])
    child_component, child_trace = _trace_normalized(covariance)
    context_component, context_trace = _trace_normalized(
        context_cross @ context_cross.T
    )
    endpoint_component, endpoint_trace = _trace_normalized(
        endpoint_cross @ endpoint_cross.T
    )
    prior_component, prior_trace = _trace_normalized(prior_gram)
    components = (
        child_component,
        context_component,
        endpoint_component,
        prior_component,
    )
    active = [
        component
        for component, trace in zip(
            components,
            (child_trace, context_trace, endpoint_trace, prior_trace),
        )
        if trace > float(torch.finfo(compute_dtype).eps)
    ]
    aggregate = sum(active, torch.zeros_like(child_component)) / max(len(active), 1)
    aggregate = 0.5 * (aggregate + aggregate.T)
    full_rank = rank == hidden and hidden <= full_rank_threshold
    spectrum = None
    if full_rank:
        basis = torch.eye(hidden, device=packed.child.device, dtype=compute_dtype)
    else:
        spectrum, vectors = torch.linalg.eigh(aggregate)
        order = torch.arange(
            hidden - 1,
            hidden - rank - 1,
            -1,
            device=packed.child.device,
        )
        basis = vectors.index_select(1, order)
        basis = _canonicalize_basis_signs(basis)
    identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
    orthogonality_error = float((basis.T @ basis - identity).abs().max().item())
    body = {
        "schema": LFM2_CPCF_V5_RESIDUAL_SUBSPACE_BCX_SCHEMA,
        "kind": "dynamic_residual_correction_basis",
        "implementation": _IMPLEMENTATION,
        "hidden_size": hidden,
        "correction_rank": rank,
        "full_rank_small_hidden_fallback": full_rank,
        "rank_rule": (
            "full_H_when_H_le_32_else_next_power_of_two_ceil_sqrt_H_capped_64"
        ),
        "basis_sources": [
            "case_balanced_child_activation_covariance",
            "current_context_residual_cross_covariance",
            "current_suffix_residual_write_cross_covariance_through_writer",
            "transported_dense_prior_row_space",
        ],
        "component_raw_traces": {
            "child_activation_covariance": child_trace,
            "context_residual_cross_covariance_psd": context_trace,
            "endpoint_residual_cross_covariance_psd": endpoint_trace,
            "transported_prior_row_gram": prior_trace,
        },
        "active_component_count": len(active),
        "equal_trace_normalized_component_weighting": True,
        "aggregate_basis_operator_sha256": _tensor_sha256(
            aggregate,
            "basis.aggregate_operator",
        ),
        "basis_sha256": _tensor_sha256(basis, "basis.value"),
        "basis_spectrum": (
            None
            if spectrum is None
            else [float(value) for value in spectrum.flip(0)[:rank].tolist()]
        ),
        "canonical_basis_signs": True,
        "maximum_orthogonality_error": orthogonality_error,
        "compute_device": compute_device,
        "compute_dtype": _dtype_name(torch.empty((), dtype=compute_dtype)),
        "fit_only": True,
        "case_authentication_sha256": list(
            packed.case_authentication_sha256
        ),
        "execution": {
            **_ZERO_EXECUTION,
            "closed_form_covariance_eigendecomposition": not full_rank,
        },
    }
    return basis, _seal_report(body)


def _lag_windows(value: Any, fir_length: int) -> Any:
    torch = _torch()
    return (
        torch.nn.functional.pad(
            value.transpose(1, 2),
            (fir_length - 1, 0),
        )
        .unfold(dimension=2, size=fir_length, step=1)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def _reduced_channel_design(
    *,
    feature_windows: Any,
    features: Any,
    b_history: Any,
    c_history: Any,
    x_history: Any,
    kernel: Any,
    conv_bias: Any | None,
) -> Any:
    """Build all reduced BCX Jacobians in one channel microbatch."""

    b_windows = _lag_windows(b_history, int(kernel.shape[1]))
    x_windows = _lag_windows(x_history, int(kernel.shape[1]))
    weighted_b = torch_einsum(
        "mtlq,mtlc,cl->mtcq",
        feature_windows,
        x_windows,
        kernel,
    ) * c_history.unsqueeze(-1)
    weighted_x = torch_einsum(
        "mtlq,mtlc,cl->mtcq",
        feature_windows,
        b_windows,
        kernel,
    ) * c_history.unsqueeze(-1)
    filtered = torch_einsum(
        "mtlc,mtlc,cl->mtc",
        x_windows,
        b_windows,
        kernel,
    )
    if conv_bias is not None:
        filtered = filtered + conv_bias.reshape(1, 1, -1)
    weighted_c = features.unsqueeze(2) * filtered.unsqueeze(-1)
    return torch_cat((weighted_b, weighted_c, weighted_x), dim=-1).contiguous()


def torch_einsum(equation: str, *operands: Any) -> Any:
    """Tiny indirection kept patchable by focused performance tests."""

    return _torch().einsum(equation, *operands)


def torch_cat(values: Sequence[Any], *, dim: int) -> Any:
    return _torch().cat(tuple(values), dim=dim)


def _condition_cap(
    cap_id: str,
    ridge_policy: Lfm2CpcfV5RidgePolicy,
) -> float:
    if not isinstance(ridge_policy, Lfm2CpcfV5RidgePolicy):
        raise TypeError("ridge_policy must be Lfm2CpcfV5RidgePolicy.")
    if not ridge_policy.verify_authentication():
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "ridge_policy failed authentication."
        )
    candidates = {
        f"kappa-{int(value):05d}": float(value)
        for value in ridge_policy.condition_caps
    }
    normalized = str(cap_id).strip().lower()
    if normalized not in candidates:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            f"cap_id must be one of {tuple(candidates)}."
        )
    return candidates[normalized]


def _solve_reduced_coordinate(
    *,
    gram: Any,
    right: Any,
    condition_cap: float,
    rank_tolerance: float,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    diagonal = gram.diagonal().clamp(min=0.0).sqrt()
    maximum_column = float(diagonal.max().item())
    column_threshold = maximum_column * rank_tolerance
    active = diagonal > column_threshold
    scale = torch.where(active, diagonal, torch.ones_like(diagonal))
    equilibrated = gram / scale.unsqueeze(0) / scale.unsqueeze(1)
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(equilibrated)
    eigenvalues = eigenvalues.clamp(min=0.0)
    maximum = float(eigenvalues.max().item())
    minimum = float(eigenvalues.min().item())
    ridge = (
        0.0
        if maximum == 0.0
        else max(
            0.0,
            (maximum - condition_cap * minimum) / (condition_cap - 1.0),
        )
    )
    identifiable = eigenvalues > maximum * rank_tolerance
    inverse = torch.zeros_like(eigenvalues)
    inverse[identifiable] = 1.0 / (eigenvalues[identifiable] + ridge)
    eigen_right = eigenvectors.T @ (right / scale)
    delta_equilibrated = eigenvectors @ (inverse * eigen_right)
    delta = delta_equilibrated / scale
    delta[~active] = 0.0
    regularized_condition = (
        0.0
        if maximum == 0.0
        else (maximum + ridge) / (minimum + ridge)
    )
    return delta, {
        "maximum_eigenvalue": maximum,
        "minimum_eigenvalue": minimum,
        "effective_ridge": ridge,
        "regularized_condition": regularized_condition,
        "condition_cap": condition_cap,
        "condition_cap_passed": (
            regularized_condition <= condition_cap * (1.0 + 8.0 * rank_tolerance)
        ),
        "effective_rank": int(identifiable.sum().item()),
        "inactive_reduced_coordinates": int((~active).sum().item()),
        "equilibrated_gram_sha256": _tensor_sha256(
            equilibrated,
            "coordinate.equilibrated_gram",
        ),
        "eigenvalues_sha256": _tensor_sha256(
            eigenvalues,
            "coordinate.eigenvalues",
        ),
    }


def _rms(value: Any) -> float:
    return float(value.square().mean().sqrt().item())


def _trust_scale(limit: float, observed: float) -> float:
    return 1.0 if observed == 0.0 else min(1.0, limit / observed)


def solve_lfm2_cpcf_v5_residual_subspace_bcx(
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    replays: (
        Sequence[Lfm2CpcfV4VolterraReplay]
        | Mapping[str, Lfm2CpcfV4VolterraReplay]
    ),
    state: Mapping[str, Any | None],
    cap_id: str,
    ridge_policy: Lfm2CpcfV5RidgePolicy,
    context_objective_weight: float = 0.05,
    compute_device: str | None = "auto",
    channel_chunk_size: int = 32,
) -> Lfm2CpcfV5ResidualSubspaceBcxResult:
    """Fold one endpoint-aware BCX correction into a dense transported prior."""

    torch = _torch()
    context_weight = _positive_float(
        context_objective_weight,
        "context_objective_weight",
        allow_zero=True,
    )
    if context_weight > 0.1:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "context_objective_weight must be between 0.0 and 0.1."
        )
    if (
        isinstance(channel_chunk_size, bool)
        or not isinstance(channel_chunk_size, int)
        or channel_chunk_size <= 0
    ):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "channel_chunk_size must be a positive integer."
        )
    condition_cap = _condition_cap(cap_id, ridge_policy)
    pairs = _ordered_case_replays(cases, replays)
    in_weight = _state_tensor(
        state,
        "in_proj_weight",
        dimensions=2,
    )
    in_bias = _state_tensor(
        state,
        "in_proj_bias",
        dimensions=1,
        optional=True,
    )
    fir_kernel = _state_tensor(
        state,
        "fir_kernel",
        dimensions=2,
    )
    fir_bias = _state_tensor(
        state,
        "fir_bias",
        dimensions=1,
        optional=True,
    )
    writer_weight = _state_tensor(
        state,
        "writer_weight",
        dimensions=2,
    )
    _state_tensor(
        state,
        "writer_bias",
        dimensions=1,
        optional=True,
    )
    hidden = int(in_weight.shape[1])
    if tuple(in_weight.shape) != (3 * hidden, hidden):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "in_proj_weight must have shape [3H,H]."
        )
    fit_physical_bias = in_bias is not None
    if in_bias is not None and tuple(in_bias.shape) != (3 * hidden,):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "in_proj_bias must have shape [3H]."
        )
    if int(fir_kernel.shape[0]) != hidden:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "fir_kernel channel width differs from in_proj."
        )
    if fir_bias is not None and tuple(fir_bias.shape) != (hidden,):
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "fir_bias channel width differs from in_proj."
        )
    if int(writer_weight.shape[1]) != hidden:
        raise Lfm2CpcfV5ResidualSubspaceBcxError(
            "writer_weight input width differs from in_proj."
        )
    output_width = int(writer_weight.shape[0])
    device, compute_dtype, device_name = _resolve_device(compute_device)
    rank = automatic_residual_subspace_rank(hidden)
    reduced_role_width = rank + int(fit_physical_bias)
    reduced_coordinate_width = 3 * reduced_role_width
    effective_rank_tolerance = max(
        ridge_policy.rank_relative_tolerance,
        64.0 * float(torch.finfo(compute_dtype).eps),
    )
    prior_weight = in_weight.detach().to(
        device=device,
        dtype=compute_dtype,
    ).contiguous()
    prior_bias = (
        None
        if in_bias is None
        else in_bias.detach().to(
            device=device,
            dtype=compute_dtype,
        ).contiguous()
    )
    kernel = fir_kernel.detach().to(
        device=device,
        dtype=compute_dtype,
    ).contiguous()
    convolution_bias = (
        None
        if fir_bias is None
        else fir_bias.detach().to(
            device=device,
            dtype=compute_dtype,
        ).contiguous()
    )
    writer = writer_weight.detach().to(
        device=device,
        dtype=compute_dtype,
    ).contiguous()
    with _deterministic_compute_scope(device):
        packed = _pack_evidence(
            pairs,
            hidden_size=hidden,
            output_width=output_width,
            device=device,
            dtype=compute_dtype,
        )
        writer_gram = writer.T @ writer / output_width
        endpoint_projection = packed.write_residual @ writer / output_width
        endpoint_residual_energy = float(
            (
                packed.suffix_weights
                * packed.write_residual.square().mean(dim=-1)
            )
            .sum()
            .item()
        )
        basis, basis_report = _derive_basis(
            packed,
            prior_weight=prior_weight,
            endpoint_projection=endpoint_projection,
            rank=rank,
            full_rank_threshold=32,
            compute_device=device_name,
            compute_dtype=compute_dtype,
        )
        packed.write_residual = torch.empty(
            (0,),
            device=device,
            dtype=compute_dtype,
        )
        masked_child = packed.child * packed.prefix_mask.unsqueeze(-1).to(
            compute_dtype
        )
        features = masked_child @ basis
        if fit_physical_bias:
            features = torch.cat(
                (
                    features,
                    torch.ones(
                        (*features.shape[:-1], 1),
                        device=device,
                        dtype=compute_dtype,
                    ),
                ),
                dim=-1,
            )
        feature_windows = _lag_windows(features, int(kernel.shape[1]))
        solution_weight = prior_weight.clone()
        solution_bias = None if prior_bias is None else prior_bias.clone()
        channel_reports = []
        coordinate_objective_before = 0.0
        coordinate_objective_after = 0.0
        context_coefficient = context_weight / hidden
        channel_indices = torch.arange(hidden, device=device)
        role_prior = torch.stack(
            tuple(
                prior_weight[role * hidden + channel_indices]
                for role in range(3)
            ),
            dim=1,
        )
        role_prior_bias = (
            None
            if prior_bias is None
            else torch.stack(
                tuple(
                    prior_bias[role * hidden + channel_indices]
                    for role in range(3)
                ),
                dim=1,
            )
        )
        for chunk_start in range(0, hidden, channel_chunk_size):
            chunk_stop = min(hidden, chunk_start + channel_chunk_size)
            chunk = slice(chunk_start, chunk_stop)
            design = _reduced_channel_design(
                feature_windows=feature_windows,
                features=features,
                b_history=packed.b_history[..., chunk],
                c_history=packed.c_history[..., chunk],
                x_history=packed.x_history[..., chunk],
                kernel=kernel[chunk],
                conv_bias=(
                    None if convolution_bias is None else convolution_bias[chunk]
                ),
            )
            writer_chunk = writer[:, chunk]
            endpoint_coefficient = (
                writer_chunk.square().sum(dim=0) / output_width
            )
            row_coefficient = (
                packed.suffix_weights.unsqueeze(-1)
                * endpoint_coefficient.reshape(1, 1, -1)
                + packed.prefix_weights.unsqueeze(-1) * context_coefficient
            )
            weighted_design = design * row_coefficient.clamp(min=0.0).sqrt().unsqueeze(
                -1
            )
            grams = torch.einsum(
                "mtcd,mtce->cde",
                weighted_design,
                weighted_design,
            )
            grams = 0.5 * (grams + grams.transpose(-2, -1))
            for local_index, channel in enumerate(range(chunk_start, chunk_stop)):
                channel_design = design[..., local_index, :]
                writer_column = writer[:, channel]
                endpoint_right_signal = endpoint_projection[..., channel]
                right = torch.einsum(
                    "mtd,mt->d",
                    channel_design,
                    packed.suffix_weights * endpoint_right_signal,
                )
                if context_coefficient > 0.0:
                    right = right + context_coefficient * torch.einsum(
                        "mtd,mt->d",
                        channel_design,
                        packed.prefix_weights
                        * packed.context_residual[..., channel],
                    )
                raw_delta, solve_report = _solve_reduced_coordinate(
                    gram=grams[local_index],
                    right=right,
                    condition_cap=condition_cap,
                    rank_tolerance=effective_rank_tolerance,
                )
                role_delta = raw_delta.reshape(3, reduced_role_width)
                physical_weight_delta = role_delta[:, :rank] @ basis.T
                physical_bias_delta = (
                    None
                    if not fit_physical_bias
                    else role_delta[:, -1]
                )
                parameter_parts = [physical_weight_delta.reshape(-1)]
                prior_parts = [role_prior[channel].reshape(-1)]
                if physical_bias_delta is not None:
                    assert role_prior_bias is not None
                    parameter_parts.append(physical_bias_delta)
                    prior_parts.append(role_prior_bias[channel])
                parameter_delta = torch.cat(parameter_parts)
                prior_parameters = torch.cat(prior_parts)
                raw_parameter_rms = _rms(parameter_delta)
                parameter_reference_rms = max(
                    _rms(prior_parameters),
                    ridge_policy.parameter_reference_floor,
                )
                parameter_limit = (
                    ridge_policy.parameter_trust_radius
                    * parameter_reference_rms
                )
                parameter_scale = _trust_scale(
                    parameter_limit,
                    raw_parameter_rms,
                )
                raw_effect_energy = float(
                    (
                        raw_delta
                        @ grams[local_index]
                        @ raw_delta
                    )
                    .clamp(min=0.0)
                    .item()
                )
                raw_effect_rms = math.sqrt(raw_effect_energy)
                context_residual_energy = (
                    context_coefficient
                    * float(
                        (
                            packed.prefix_weights
                            * packed.context_residual[..., channel].square()
                        )
                        .sum()
                        .item()
                    )
                )
                current_context = packed.context[..., channel]
                prior_effect_energy = float(
                    (
                        packed.suffix_weights
                        * current_context.square()
                        * endpoint_coefficient[local_index]
                    )
                    .sum()
                    .item()
                ) + context_coefficient * float(
                    (
                        packed.prefix_weights
                        * current_context.square()
                    )
                    .sum()
                    .item()
                )
                effect_reference_rms = max(
                    math.sqrt(max(prior_effect_energy, 0.0)),
                    math.sqrt(
                        max(
                            endpoint_residual_energy + context_residual_energy,
                            0.0,
                        )
                    ),
                    ridge_policy.effect_reference_floor,
                )
                effect_limit = (
                    ridge_policy.effect_trust_radius * effect_reference_rms
                )
                effect_scale = _trust_scale(effect_limit, raw_effect_rms)
                accepted_scale = min(1.0, parameter_scale, effect_scale)
                accepted_delta = raw_delta * accepted_scale
                accepted_weight_delta = physical_weight_delta * accepted_scale
                accepted_bias_delta = (
                    None
                    if physical_bias_delta is None
                    else physical_bias_delta * accepted_scale
                )
                objective_before = (
                    endpoint_residual_energy + context_residual_energy
                )
                context_delta = channel_design @ accepted_delta
                endpoint_linear = float(
                    (
                        packed.suffix_weights
                        * context_delta
                        * endpoint_projection[..., channel]
                    )
                    .sum()
                    .item()
                )
                endpoint_quadratic = float(
                    (
                        packed.suffix_weights
                        * context_delta.square()
                    )
                    .sum()
                    .item()
                ) * float(writer_gram[channel, channel].item())
                proposed_endpoint_residual_energy = max(
                    0.0,
                    endpoint_residual_energy
                    - 2.0 * endpoint_linear
                    + endpoint_quadratic,
                )
                proposed_context_residual = (
                    packed.context_residual[..., channel] - context_delta
                )
                objective_after = proposed_endpoint_residual_energy + (
                    context_coefficient
                    * float(
                        (
                            packed.prefix_weights
                            * proposed_context_residual.square()
                        )
                        .sum()
                        .item()
                    )
                )
                tolerance = (
                    256.0
                    * float(torch.finfo(compute_dtype).eps)
                    * max(1.0, objective_before)
                )
                coordinate_reverted = objective_after > objective_before + tolerance
                if coordinate_reverted:
                    accepted_scale = 0.0
                    accepted_delta = torch.zeros_like(raw_delta)
                    accepted_weight_delta = torch.zeros_like(
                        physical_weight_delta
                    )
                    accepted_bias_delta = (
                        None
                        if physical_bias_delta is None
                        else torch.zeros_like(physical_bias_delta)
                    )
                    context_delta = torch.zeros_like(context_delta)
                    objective_after = objective_before
                else:
                    endpoint_projection = (
                        endpoint_projection
                        - context_delta.unsqueeze(-1)
                        * writer_gram[channel].reshape(1, 1, -1)
                    )
                    endpoint_residual_energy = (
                        proposed_endpoint_residual_energy
                    )
                    packed.context_residual[..., channel] = (
                        proposed_context_residual
                    )
                    packed.context[..., channel] = (
                        packed.context[..., channel] + context_delta
                    )
                for role in range(3):
                    row = role * hidden + channel
                    solution_weight[row] = (
                        solution_weight[row] + accepted_weight_delta[role]
                    )
                    if solution_bias is not None:
                        assert accepted_bias_delta is not None
                        solution_bias[row] = (
                            solution_bias[row] + accepted_bias_delta[role]
                        )
                coordinate_objective_before += objective_before
                coordinate_objective_after += objective_after
                channel_body = {
                    "schema": LFM2_CPCF_V5_RESIDUAL_SUBSPACE_BCX_SCHEMA,
                    "kind": "residual_subspace_bcx_coordinate_receipt",
                    "channel": channel,
                    "coordinate_order": channel,
                    "reduced_coordinate_width": reduced_coordinate_width,
                    "endpoint_residual_aware": True,
                    "writer_column_sha256": _tensor_sha256(
                        writer_column,
                        f"channel.{channel}.writer_column",
                    ),
                    "objective_before": objective_before,
                    "objective_after": objective_after,
                    "objective_not_worse": (
                        objective_after <= objective_before + tolerance
                    ),
                    "coordinate_reverted": coordinate_reverted,
                    "raw_reduced_delta_sha256": _tensor_sha256(
                        raw_delta,
                        f"channel.{channel}.raw_reduced_delta",
                    ),
                    "accepted_reduced_delta_sha256": _tensor_sha256(
                        accepted_delta,
                        f"channel.{channel}.accepted_reduced_delta",
                    ),
                    "accepted_weight_delta_sha256": _tensor_sha256(
                        accepted_weight_delta,
                        f"channel.{channel}.accepted_weight_delta",
                    ),
                    "accepted_bias_delta_sha256": (
                        None
                        if accepted_bias_delta is None
                        else _tensor_sha256(
                            accepted_bias_delta,
                            f"channel.{channel}.accepted_bias_delta",
                        )
                    ),
                    "parameter_space_trust": {
                        "reference_rms": parameter_reference_rms,
                        "absolute_limit_rms": parameter_limit,
                        "raw_correction_rms": raw_parameter_rms,
                        "individual_scale": parameter_scale,
                    },
                    "function_space_trust": {
                        "reference_rms": effect_reference_rms,
                        "absolute_limit_rms": effect_limit,
                        "raw_correction_rms": raw_effect_rms,
                        "individual_scale": effect_scale,
                    },
                    "joint_trust_scale": accepted_scale,
                    **solve_report,
                    "execution": {
                        **_ZERO_EXECUTION,
                        "closed_form_condition_capped_eigendecomposition": True,
                    },
                }
                if not solve_report["condition_cap_passed"]:
                    raise Lfm2CpcfV5ResidualSubspaceBcxError(
                        f"Channel {channel} failed its condition cap."
                    )
                channel_reports.append(_seal_report(channel_body))

        output_weight = solution_weight.to(
            device="cpu",
            dtype=in_weight.dtype,
        ).contiguous()
        output_bias = (
            None
            if solution_bias is None
            else solution_bias.to(
                device="cpu",
                dtype=in_bias.dtype,
            ).contiguous()
        )
    body = {
        "schema": LFM2_CPCF_V5_RESIDUAL_SUBSPACE_BCX_SCHEMA,
        "kind": "scalable_endpoint_aware_residual_subspace_bcx",
        "implementation": _IMPLEMENTATION,
        "cap_id": str(cap_id).strip().lower(),
        "condition_cap": condition_cap,
        "ridge_policy_sha256": ridge_policy.authentication_sha256,
        "requested_rank_relative_tolerance": (
            ridge_policy.rank_relative_tolerance
        ),
        "effective_compute_rank_relative_tolerance": (
            effective_rank_tolerance
        ),
        "hidden_size": hidden,
        "output_width": output_width,
        "correction_rank": rank,
        "physical_bias_coordinate_exact_and_unprojected": fit_physical_bias,
        "reduced_role_width": reduced_role_width,
        "reduced_coordinate_width_per_channel": reduced_coordinate_width,
        "dense_prior_preserved_additively": True,
        "endpoint_objective_primary": True,
        "context_objective_weight": context_weight,
        "context_coefficient_per_channel": context_coefficient,
        "writer_cross_channel_contract": (
            "deterministic_channel_coordinate_updates_with_"
            "writer_projected_residual_sufficient_statistics_refresh"
        ),
        "independent_simultaneous_endpoint_solve_claimed": False,
        "channel_chunk_size": channel_chunk_size,
        "channel_coordinate_count": hidden,
        "coordinate_revert_count": sum(
            int(value["coordinate_reverted"]) for value in channel_reports
        ),
        "sum_coordinate_objective_before": coordinate_objective_before,
        "sum_coordinate_objective_after": coordinate_objective_after,
        "causal_row_balance": {
            "exact_phase_boundaries_available": False,
            "proxy": (
                "per_case_50_percent_uniform_plus_50_percent_"
                "exponentially_tail_weighted"
            ),
            "tail_tau": "max(4,sqrt(valid_rows))",
            "case_balanced": True,
            "row_duplication_dominance_reduced": True,
        },
        "case_reports": list(packed.case_reports),
        "basis_report": basis_report,
        "channel_reports": channel_reports,
        "prior_weight_sha256": _tensor_sha256(
            in_weight,
            "prior.in_proj_weight",
        ),
        "prior_bias_sha256": (
            None
            if in_bias is None
            else _tensor_sha256(in_bias, "prior.in_proj_bias")
        ),
        "solution_weight_sha256": _tensor_sha256(
            output_weight,
            "result.in_proj_weight",
        ),
        "solution_bias_sha256": (
            None
            if output_bias is None
            else _tensor_sha256(output_bias, "result.in_proj_bias")
        ),
        "compute_device": device_name,
        "compute_dtype": _dtype_name(
            torch.empty((), dtype=compute_dtype)
        ),
        "cuda_fp32_when_available": compute_device in {None, "auto"}
        and device.type == "cuda",
        "cpu_fp64_fallback": device.type == "cpu",
        "same_backend_deterministic": True,
        "cross_backend_bit_identity_claimed": False,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "execution": {
            **_ZERO_EXECUTION,
            "closed_form_only": True,
            "gradients_enabled": False,
        },
    }
    diagnostics = _seal_report(body)
    receipts = (
        basis_report["report_sha256"],
        *(value["report_sha256"] for value in channel_reports),
        diagnostics["report_sha256"],
    )
    return Lfm2CpcfV5ResidualSubspaceBcxResult(
        in_proj_weight=output_weight,
        in_proj_bias=output_bias,
        receipt_sha256=receipts,
        diagnostics=diagnostics,
    )
