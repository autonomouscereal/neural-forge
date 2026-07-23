"""Analytic FP64 linearization of one materialized LFM2 macrocell.

The global residual-closure solver varies only residual writers.  At each
writer site its parameterization is ``delta_weight = U @ C`` where ``U`` is a
shared hidden-space basis and ``C`` is one site-local coefficient block.  This
module snapshots the nonlinear producer path once, then applies exact
continuous FP64 JVPs and VJPs without constructing a dense Jacobian.

The continuous RMSNorm shadow intentionally omits runtime FP32/BF16 rounding.
It uses the same equation as the public analytical RMSNorm primitive in
``hf_lfm2_cpcf_terminal_action_residual``.  It is a tangent surrogate for
folding, not a claim that floating-point rounding has a byte-exact derivative.

Attention follows the eager LFM2 Q/K RMSNorm, duplicated-half RoPE, intact GQA
repeat, masked softmax, value/context, and output-writer path.  Short
convolution follows the canonical ``B * X -> causal depthwise FIR -> C * V``
path.  Both are composed with the second pre-norm SwiGLU residual branch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Literal

from neural_forge.hf_lfm2_cached_qk_score_residual_fold import (
    apply_lfm2_finite_radius_rope,
    apply_lfm2_finite_radius_rope_transpose,
)
from neural_forge.hf_lfm2_global_residual_closure import (
    Lfm2GlobalResidualBoundaryBundle,
    Lfm2GlobalResidualLinearizationKernel,
    Lfm2GlobalResidualWriterTopology,
    adjoint_lfm2_global_residual_writer_block,
    apply_lfm2_global_residual_writer_block,
    fingerprint_lfm2_global_residual_tensor,
)


class Lfm2GlobalLinearizationError(ValueError):
    """A primal snapshot or tangent/cotangent has invalid geometry."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for analytic LFM2 macrocell linearization."
        ) from exc
    return torch


def _owned_fp64(
    value: Any,
    name: str,
    *,
    dimensions: int | None = None,
    device: Any | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or (dimensions is not None and value.ndim != dimensions)
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        rank = "" if dimensions is None else f" rank-{dimensions}"
        raise Lfm2GlobalLinearizationError(
            f"{name} must be a finite nonempty dense real{rank} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if device is not None and result.device != device:
        raise Lfm2GlobalLinearizationError(
            f"{name} archive is on {result.device}, expected {device}."
        )
    return result


def _owned_bool(
    value: Any,
    name: str,
    *,
    dimensions: int,
    device: Any,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.bool
        or value.ndim != dimensions
        or value.numel() == 0
    ):
        raise Lfm2GlobalLinearizationError(
            f"{name} must be one nonempty dense rank-{dimensions} boolean tensor "
            f"on {device}."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.device != device:
        raise Lfm2GlobalLinearizationError(
            f"{name} archive is on {result.device}, expected {device}."
        )
    return result


def _owned_padding_mask(value: Any, name: str, *, device: Any) -> Any:
    torch = _torch()
    supported_integer_dtypes = {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or (
            not torch.is_floating_point(value)
            and value.dtype not in supported_integer_dtypes
        )
        or (
            torch.is_floating_point(value)
            and not bool(torch.isfinite(value.detach()).all().item())
        )
    ):
        raise Lfm2GlobalLinearizationError(
            f"{name} must be one nonempty dense rank-two real mask."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if result.device != device:
        raise Lfm2GlobalLinearizationError(
            f"{name} archive is on {result.device}, expected {device}."
        )
    return result


def _short_conv_padding_active(value: Any | None) -> bool:
    # Mirrors apply_mask_to_padding_states in the eager LFM2 replay.
    return value is not None and int(value.shape[0]) > 1 and int(value.shape[1]) > 1


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2GlobalLinearizationError(f"{name} must be finite and positive.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2GlobalLinearizationError(f"{name} must be finite and positive.")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2GlobalLinearizationError(f"{name} must be a positive integer.")
    return value


def _plain_fp64(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    device: Any,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device != device
        or value.dtype != torch.float64
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        or (shape is not None and tuple(value.shape) != shape)
    ):
        expected = "" if shape is None else f" with shape {shape!r}"
        raise Lfm2GlobalLinearizationError(
            f"{name} must be a detached finite dense FP64 tensor on {device}{expected}."
        )
    return value


def _linear(inputs: Any, weight: Any, bias: Any | None = None) -> Any:
    result = inputs @ weight.transpose(0, 1)
    if bias is not None:
        result = result + bias
    return result


def lfm2_continuous_fp64_rmsnorm(
    rows: Any,
    gain: Any,
    epsilon: float,
) -> Any:
    """Continuous all-FP64 RMSNorm primal used by this linearization.

    The equation is the rank-agnostic form of the existing public
    ``rmsnorm_jvp`` primal in ``hf_lfm2_cpcf_terminal_action_residual``.
    """

    scale = (rows.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    return (rows / scale) * gain


def lfm2_continuous_fp64_rmsnorm_jvp(
    rows: Any,
    tangent: Any,
    gain: Any,
    epsilon: float,
) -> Any:
    """Exact continuous FP64 RMSNorm Jacobian-vector product."""

    width = int(rows.shape[-1])
    scale = (rows.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    radial = (rows * tangent).sum(dim=-1, keepdim=True)
    return gain * (tangent / scale - rows * radial / (width * scale.pow(3)))


def lfm2_continuous_fp64_rmsnorm_vjp(
    rows: Any,
    cotangent: Any,
    gain: Any,
    epsilon: float,
) -> Any:
    """Exact transpose of the continuous FP64 RMSNorm Jacobian."""

    width = int(rows.shape[-1])
    scale = (rows.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    gained = cotangent * gain
    radial = (rows * gained).sum(dim=-1, keepdim=True)
    return gained / scale - rows * radial / (width * scale.pow(3))


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _repeat_gqa(value: Any, repeats: int) -> Any:
    """Repeat intact KV heads as in ``hf_lfm2_cpcf_macrocell_replay``."""

    if repeats == 1:
        return value
    batch, heads, sequence, head_dim = map(int, value.shape)
    return (
        value[:, :, None, :, :]
        .expand(batch, heads, repeats, sequence, head_dim)
        .reshape(batch, heads * repeats, sequence, head_dim)
    )


def _sum_gqa_repeats(value: Any, key_value_heads: int, repeats: int) -> Any:
    if repeats == 1:
        return value
    batch, _, sequence, head_dim = map(int, value.shape)
    return value.reshape(
        batch,
        key_value_heads,
        repeats,
        sequence,
        head_dim,
    ).sum(dim=2)


def _masked_softmax(scores: Any, mask: Any) -> Any:
    """FP64 causal softmax copied from the cached Q/K residual replay."""

    torch = _torch()
    masked = scores.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked, dim=-1, dtype=torch.float64)
    return torch.where(mask, probabilities, torch.zeros_like(probabilities))


def _masked_softmax_jvp(probabilities: Any, tangent: Any, mask: Any) -> Any:
    centered = tangent - (probabilities * tangent).sum(dim=-1, keepdim=True)
    return _torch().where(
        mask,
        probabilities * centered,
        _torch().zeros_like(centered),
    )


def _masked_softmax_vjp(probabilities: Any, cotangent: Any, mask: Any) -> Any:
    # The rowwise softmax Jacobian is symmetric on the unmasked coordinates.
    return _masked_softmax_jvp(probabilities, cotangent, mask)


def _causal_fir(value: Any, stored_kernel: Any) -> Any:
    """Canonical stored-index FIR copied from the CPCF matrix-free BCX path."""

    torch = _torch()
    result = torch.zeros_like(value)
    fir_length = int(stored_kernel.shape[1])
    for kernel_index in range(fir_length):
        delay = fir_length - 1 - kernel_index
        coefficient = stored_kernel[:, kernel_index].reshape(1, 1, -1)
        if delay == 0:
            result = result + value * coefficient
        else:
            shifted = torch.zeros_like(value)
            shifted[:, delay:] = value[:, :-delay] * coefficient
            result = result + shifted
    return result


def _causal_fir_vjp(value: Any, stored_kernel: Any) -> Any:
    """Exact transpose of :func:`_causal_fir`."""

    torch = _torch()
    result = torch.zeros_like(value)
    fir_length = int(stored_kernel.shape[1])
    for kernel_index in range(fir_length):
        delay = fir_length - 1 - kernel_index
        coefficient = stored_kernel[:, kernel_index].reshape(1, 1, -1)
        if delay == 0:
            result = result + value * coefficient
        else:
            shifted = torch.zeros_like(value)
            shifted[:, :-delay] = value[:, delay:] * coefficient
            result = result + shifted
    return result


def _silu_derivative(value: Any) -> Any:
    sigmoid = _torch().sigmoid(value)
    return sigmoid * (1.0 + value * (1.0 - sigmoid))


def _validate_output_basis(value: Any, hidden: int, device: Any) -> Any:
    basis = _plain_fp64(value, "output_basis", device=device)
    if basis.ndim != 2 or int(basis.shape[0]) != hidden or int(basis.shape[1]) <= 0:
        raise Lfm2GlobalLinearizationError(
            "output_basis must have shape [hidden, positive_rank]."
        )
    return basis


def _validate_coefficient(
    value: Any,
    *,
    rank: int,
    input_width: int,
    device: Any,
    name: str,
) -> Any:
    return _plain_fp64(
        value,
        name,
        shape=(rank, input_width),
        device=device,
    )


def _state_descriptor(value: Any) -> Any:
    """Return a canonical, byte-bound descriptor for frozen tensor state."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "sha256": fingerprint_lfm2_global_residual_tensor(value),
        }
    if is_dataclass(value):
        return {
            item.name: _state_descriptor(getattr(value, item.name))
            for item in fields(value)
            if item.name != "authentication_sha256"
        }
    if isinstance(value, tuple):
        return [_state_descriptor(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Lfm2GlobalLinearizationError(
                "Linearization state contains a nonfinite scalar."
            )
        return value
    raise Lfm2GlobalLinearizationError(
        f"Unsupported authenticated state value {type(value).__name__}."
    )


def _state_sha256(value: Any) -> str:
    encoded = json.dumps(
        _state_descriptor(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Lfm2AttentionParameters:
    """Frozen FP64 parameters and causal geometry for one attention branch."""

    operator_norm_gain: Any
    operator_norm_epsilon: float
    query_weight: Any
    key_weight: Any
    value_weight: Any
    output_weight: Any
    query_norm_gain: Any
    key_norm_gain: Any
    query_key_norm_epsilon: float
    rope_cosine: Any
    rope_sine: Any
    causal_mask: Any
    query_heads: int
    key_value_heads: int
    head_dim: int
    scaling: float | None = None

    def __post_init__(self) -> None:
        query_heads = _positive_int(self.query_heads, "query_heads")
        key_value_heads = _positive_int(self.key_value_heads, "key_value_heads")
        head_dim = _positive_int(self.head_dim, "head_dim")
        if head_dim % 2 or query_heads % key_value_heads:
            raise Lfm2GlobalLinearizationError(
                "Attention requires an even head_dim and intact GQA groups."
            )
        query_weight = _owned_fp64(
            self.query_weight,
            "query_weight",
            dimensions=2,
        )
        device = query_weight.device
        values = {
            "operator_norm_gain": _owned_fp64(
                self.operator_norm_gain,
                "operator_norm_gain",
                dimensions=1,
                device=device,
            ),
            "key_weight": _owned_fp64(
                self.key_weight,
                "key_weight",
                dimensions=2,
                device=device,
            ),
            "value_weight": _owned_fp64(
                self.value_weight,
                "value_weight",
                dimensions=2,
                device=device,
            ),
            "output_weight": _owned_fp64(
                self.output_weight,
                "output_weight",
                dimensions=2,
                device=device,
            ),
            "query_norm_gain": _owned_fp64(
                self.query_norm_gain,
                "query_norm_gain",
                dimensions=1,
                device=device,
            ),
            "key_norm_gain": _owned_fp64(
                self.key_norm_gain,
                "key_norm_gain",
                dimensions=1,
                device=device,
            ),
            "rope_cosine": _owned_fp64(
                self.rope_cosine,
                "rope_cosine",
                dimensions=3,
                device=device,
            ),
            "rope_sine": _owned_fp64(
                self.rope_sine,
                "rope_sine",
                dimensions=3,
                device=device,
            ),
        }
        hidden = int(query_weight.shape[1])
        if (
            tuple(query_weight.shape) != (query_heads * head_dim, hidden)
            or tuple(values["key_weight"].shape) != (key_value_heads * head_dim, hidden)
            or tuple(values["value_weight"].shape)
            != (key_value_heads * head_dim, hidden)
            or tuple(values["output_weight"].shape) != (hidden, query_heads * head_dim)
            or tuple(values["operator_norm_gain"].shape) != (hidden,)
            or tuple(values["query_norm_gain"].shape) != (head_dim,)
            or tuple(values["key_norm_gain"].shape) != (head_dim,)
            or tuple(values["rope_cosine"].shape) != tuple(values["rope_sine"].shape)
            or int(values["rope_cosine"].shape[-1]) != head_dim
        ):
            raise Lfm2GlobalLinearizationError(
                "Attention weights, norms, or RoPE terms have incompatible shapes."
            )
        half = head_dim // 2
        if not all(
            bool(_torch().equal(value[..., :half], value[..., half:]))
            for value in (values["rope_cosine"], values["rope_sine"])
        ):
            raise Lfm2GlobalLinearizationError(
                "RoPE cosine/sine must use duplicated-half LFM2 coordinates."
            )
        mask = _owned_bool(
            self.causal_mask,
            "causal_mask",
            dimensions=4,
            device=device,
        )
        scale = (
            head_dim**-0.5
            if self.scaling is None
            else _positive_float(
                self.scaling,
                "scaling",
            )
        )
        object.__setattr__(self, "query_weight", query_weight)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "causal_mask", mask)
        object.__setattr__(
            self,
            "operator_norm_epsilon",
            _positive_float(self.operator_norm_epsilon, "operator_norm_epsilon"),
        )
        object.__setattr__(
            self,
            "query_key_norm_epsilon",
            _positive_float(
                self.query_key_norm_epsilon,
                "query_key_norm_epsilon",
            ),
        )
        object.__setattr__(self, "scaling", scale)

    @property
    def hidden_size(self) -> int:
        return int(self.query_weight.shape[1])

    @property
    def writer_input_width(self) -> int:
        return self.query_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class Lfm2ShortConvParameters:
    """Frozen FP64 parameters for one canonical causal BCX branch."""

    operator_norm_gain: Any
    operator_norm_epsilon: float
    input_weight: Any
    stored_fir_kernel: Any
    output_weight: Any
    input_bias: Any | None = None
    fir_bias: Any | None = None
    output_bias: Any | None = None
    padding_mask: Any | None = None

    def __post_init__(self) -> None:
        input_weight = _owned_fp64(
            self.input_weight,
            "input_weight",
            dimensions=2,
        )
        device = input_weight.device
        hidden = int(input_weight.shape[1])
        if tuple(input_weight.shape) != (3 * hidden, hidden):
            raise Lfm2GlobalLinearizationError(
                "Short-conv input_weight must have shape [3*hidden, hidden]."
            )
        norm_gain = _owned_fp64(
            self.operator_norm_gain,
            "operator_norm_gain",
            dimensions=1,
            device=device,
        )
        fir = _owned_fp64(
            self.stored_fir_kernel,
            "stored_fir_kernel",
            device=device,
        )
        if fir.ndim == 3 and int(fir.shape[1]) == 1:
            fir = fir[:, 0, :].contiguous()
        output_weight = _owned_fp64(
            self.output_weight,
            "output_weight",
            dimensions=2,
            device=device,
        )
        if (
            tuple(norm_gain.shape) != (hidden,)
            or fir.ndim != 2
            or int(fir.shape[0]) != hidden
            or int(fir.shape[1]) <= 0
            or tuple(output_weight.shape) != (hidden, hidden)
        ):
            raise Lfm2GlobalLinearizationError(
                "Short-conv norm, FIR, or output writer geometry is invalid."
            )

        def optional_vector(value: Any, name: str) -> Any | None:
            if value is None:
                return None
            result = _owned_fp64(value, name, dimensions=1, device=device)
            expected = 3 * hidden if name == "input_bias" else hidden
            if tuple(result.shape) != (expected,):
                raise Lfm2GlobalLinearizationError(
                    f"{name} has shape {tuple(result.shape)}, expected ({expected},)."
                )
            return result

        padding = None
        if self.padding_mask is not None:
            padding = _owned_padding_mask(
                self.padding_mask,
                "padding_mask",
                device=device,
            )
        object.__setattr__(self, "operator_norm_gain", norm_gain)
        object.__setattr__(self, "input_weight", input_weight)
        object.__setattr__(self, "stored_fir_kernel", fir)
        object.__setattr__(self, "output_weight", output_weight)
        object.__setattr__(
            self,
            "operator_norm_epsilon",
            _positive_float(self.operator_norm_epsilon, "operator_norm_epsilon"),
        )
        object.__setattr__(
            self, "input_bias", optional_vector(self.input_bias, "input_bias")
        )
        object.__setattr__(self, "fir_bias", optional_vector(self.fir_bias, "fir_bias"))
        object.__setattr__(
            self, "output_bias", optional_vector(self.output_bias, "output_bias")
        )
        object.__setattr__(self, "padding_mask", padding)

    @property
    def hidden_size(self) -> int:
        return int(self.output_weight.shape[0])

    @property
    def writer_input_width(self) -> int:
        return self.hidden_size


@dataclass(frozen=True, slots=True)
class Lfm2MlpParameters:
    """Frozen FP64 parameters for the second pre-norm SwiGLU branch."""

    norm_gain: Any
    norm_epsilon: float
    gate_weight: Any
    up_weight: Any
    down_weight: Any

    def __post_init__(self) -> None:
        gate = _owned_fp64(self.gate_weight, "gate_weight", dimensions=2)
        device = gate.device
        up = _owned_fp64(
            self.up_weight,
            "up_weight",
            dimensions=2,
            device=device,
        )
        down = _owned_fp64(
            self.down_weight,
            "down_weight",
            dimensions=2,
            device=device,
        )
        gain = _owned_fp64(
            self.norm_gain,
            "norm_gain",
            dimensions=1,
            device=device,
        )
        intermediate, hidden = map(int, gate.shape)
        if (
            tuple(up.shape) != (intermediate, hidden)
            or tuple(down.shape) != (hidden, intermediate)
            or tuple(gain.shape) != (hidden,)
        ):
            raise Lfm2GlobalLinearizationError(
                "SwiGLU gate/up/down weights or RMSNorm gain are incompatible."
            )
        object.__setattr__(self, "norm_gain", gain)
        object.__setattr__(self, "gate_weight", gate)
        object.__setattr__(self, "up_weight", up)
        object.__setattr__(self, "down_weight", down)
        object.__setattr__(
            self,
            "norm_epsilon",
            _positive_float(self.norm_epsilon, "norm_epsilon"),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.gate_weight.shape[1])

    @property
    def writer_input_width(self) -> int:
        return int(self.gate_weight.shape[0])


@dataclass(frozen=True, slots=True)
class Lfm2TerminalNormParameters:
    """Frozen shared final RMSNorm parameters for a terminal hidden boundary."""

    gain: Any
    epsilon: float

    def __post_init__(self) -> None:
        gain = _owned_fp64(self.gain, "terminal_norm_gain", dimensions=1)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(
            self,
            "epsilon",
            _positive_float(self.epsilon, "terminal_norm_epsilon"),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.gain.shape[0])


@dataclass(frozen=True, slots=True)
class Lfm2AttentionPrimalSnapshot:
    residual: Any
    parameters: Lfm2AttentionParameters
    normalized_input: Any
    query_linear: Any
    key_linear: Any
    value: Any
    query_pre_rope: Any
    key_pre_rope: Any
    query: Any
    key: Any
    repeated_key: Any
    repeated_value: Any
    probabilities: Any
    context: Any
    writer_input: Any
    delta: Any
    output: Any


@dataclass(frozen=True, slots=True)
class Lfm2ShortConvPrimalSnapshot:
    residual: Any
    parameters: Lfm2ShortConvParameters
    normalized_input: Any
    masked_input: Any
    b_values: Any
    c_values: Any
    x_values: Any
    bx_values: Any
    filtered: Any
    writer_input: Any
    delta: Any
    output: Any


@dataclass(frozen=True, slots=True)
class Lfm2MlpPrimalSnapshot:
    residual: Any
    parameters: Lfm2MlpParameters
    normalized_input: Any
    gate: Any
    up: Any
    activation: Any
    writer_input: Any
    delta: Any
    output: Any


@dataclass(frozen=True, slots=True)
class Lfm2TerminalNormPrimalSnapshot:
    """Detached FP64 input/output pair for the optional final RMSNorm."""

    residual: Any
    parameters: Lfm2TerminalNormParameters
    output: Any


def capture_lfm2_terminal_norm_primal(
    residual: Any,
    parameters: Lfm2TerminalNormParameters,
) -> Lfm2TerminalNormPrimalSnapshot:
    if not isinstance(parameters, Lfm2TerminalNormParameters):
        raise TypeError("parameters must be Lfm2TerminalNormParameters.")
    source = _owned_fp64(residual, "terminal_residual", dimensions=3)
    if (
        source.device != parameters.gain.device
        or int(source.shape[-1]) != parameters.hidden_size
    ):
        raise Lfm2GlobalLinearizationError(
            "Terminal residual and norm gain differ in device or hidden width."
        )
    with _torch().no_grad():
        output = lfm2_continuous_fp64_rmsnorm(
            source,
            parameters.gain,
            parameters.epsilon,
        )
    return Lfm2TerminalNormPrimalSnapshot(
        residual=source,
        parameters=parameters,
        output=output,
    )


def lfm2_terminal_norm_jvp(
    snapshot: Lfm2TerminalNormPrimalSnapshot,
    incoming_residual_tangent: Any,
) -> Any:
    incoming = _plain_fp64(
        incoming_residual_tangent,
        "terminal_incoming_residual_tangent",
        shape=tuple(snapshot.residual.shape),
        device=snapshot.residual.device,
    )
    return lfm2_continuous_fp64_rmsnorm_jvp(
        snapshot.residual,
        incoming,
        snapshot.parameters.gain,
        snapshot.parameters.epsilon,
    )


def lfm2_terminal_norm_vjp(
    snapshot: Lfm2TerminalNormPrimalSnapshot,
    output_cotangent: Any,
) -> Any:
    cotangent = _plain_fp64(
        output_cotangent,
        "terminal_output_cotangent",
        shape=tuple(snapshot.output.shape),
        device=snapshot.output.device,
    )
    return lfm2_continuous_fp64_rmsnorm_vjp(
        snapshot.residual,
        cotangent,
        snapshot.parameters.gain,
        snapshot.parameters.epsilon,
    )


def capture_lfm2_attention_primal(
    residual: Any,
    parameters: Lfm2AttentionParameters,
) -> Lfm2AttentionPrimalSnapshot:
    """Capture one independently owned, detached attention residual primal."""

    if not isinstance(parameters, Lfm2AttentionParameters):
        raise TypeError("parameters must be Lfm2AttentionParameters.")
    hidden = parameters.hidden_size
    source = _owned_fp64(residual, "residual", dimensions=3)
    if (
        source.device != parameters.query_weight.device
        or int(source.shape[-1]) != hidden
    ):
        raise Lfm2GlobalLinearizationError(
            "Attention residual and parameters differ in device or hidden width."
        )
    batch, sequence, _ = map(int, source.shape)
    rope_shape = tuple(parameters.rope_cosine.shape)
    if rope_shape not in {
        (1, sequence, parameters.head_dim),
        (batch, sequence, parameters.head_dim),
    }:
        raise Lfm2GlobalLinearizationError(
            "Attention RoPE terms do not match the residual batch/sequence."
        )
    score_shape = (batch, parameters.query_heads, sequence, sequence)
    try:
        mask = parameters.causal_mask.expand(score_shape)
    except RuntimeError as exc:
        raise Lfm2GlobalLinearizationError(
            "causal_mask cannot broadcast to attention scores."
        ) from exc
    if bool((~mask.any(dim=-1)).any().item()):
        raise Lfm2GlobalLinearizationError(
            "Every attention query row must expose at least one causal key."
        )
    with _torch().no_grad():
        normalized = lfm2_continuous_fp64_rmsnorm(
            source,
            parameters.operator_norm_gain,
            parameters.operator_norm_epsilon,
        )
        q_linear = _linear(normalized, parameters.query_weight).reshape(
            batch,
            sequence,
            parameters.query_heads,
            parameters.head_dim,
        )
        k_linear = _linear(normalized, parameters.key_weight).reshape(
            batch,
            sequence,
            parameters.key_value_heads,
            parameters.head_dim,
        )
        value = (
            _linear(normalized, parameters.value_weight)
            .reshape(
                batch,
                sequence,
                parameters.key_value_heads,
                parameters.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )
        q_pre = (
            lfm2_continuous_fp64_rmsnorm(
                q_linear,
                parameters.query_norm_gain,
                parameters.query_key_norm_epsilon,
            )
            .transpose(1, 2)
            .contiguous()
        )
        k_pre = (
            lfm2_continuous_fp64_rmsnorm(
                k_linear,
                parameters.key_norm_gain,
                parameters.query_key_norm_epsilon,
            )
            .transpose(1, 2)
            .contiguous()
        )
        query = apply_lfm2_finite_radius_rope(
            q_pre,
            parameters.rope_cosine,
            parameters.rope_sine,
        )
        key = apply_lfm2_finite_radius_rope(
            k_pre,
            parameters.rope_cosine,
            parameters.rope_sine,
        )
        repeats = parameters.query_heads // parameters.key_value_heads
        repeated_key = _repeat_gqa(key, repeats)
        repeated_value = _repeat_gqa(value, repeats)
        scores = (query @ repeated_key.transpose(-1, -2)) * parameters.scaling
        probabilities = _masked_softmax(scores, mask)
        context = probabilities @ repeated_value
        writer_input = (
            context.transpose(1, 2)
            .contiguous()
            .reshape(
                batch,
                sequence,
                parameters.writer_input_width,
            )
        )
        delta = _linear(writer_input, parameters.output_weight)
        output = source + delta
    return Lfm2AttentionPrimalSnapshot(
        residual=source,
        parameters=parameters,
        normalized_input=normalized,
        query_linear=q_linear,
        key_linear=k_linear,
        value=value,
        query_pre_rope=q_pre,
        key_pre_rope=k_pre,
        query=query,
        key=key,
        repeated_key=repeated_key,
        repeated_value=repeated_value,
        probabilities=probabilities,
        context=context,
        writer_input=writer_input,
        delta=delta,
        output=output,
    )


def capture_lfm2_short_conv_primal(
    residual: Any,
    parameters: Lfm2ShortConvParameters,
) -> Lfm2ShortConvPrimalSnapshot:
    """Capture one independently owned, detached causal BCX residual primal."""

    if not isinstance(parameters, Lfm2ShortConvParameters):
        raise TypeError("parameters must be Lfm2ShortConvParameters.")
    hidden = parameters.hidden_size
    source = _owned_fp64(residual, "residual", dimensions=3)
    if (
        source.device != parameters.input_weight.device
        or int(source.shape[-1]) != hidden
    ):
        raise Lfm2GlobalLinearizationError(
            "Short-conv residual and parameters differ in device or hidden width."
        )
    if parameters.padding_mask is not None and tuple(
        parameters.padding_mask.shape
    ) != tuple(source.shape[:2]):
        raise Lfm2GlobalLinearizationError(
            "Short-conv padding_mask must match residual batch/sequence."
        )
    with _torch().no_grad():
        normalized = lfm2_continuous_fp64_rmsnorm(
            source,
            parameters.operator_norm_gain,
            parameters.operator_norm_epsilon,
        )
        masked = normalized
        if _short_conv_padding_active(parameters.padding_mask):
            masked = normalized * parameters.padding_mask.unsqueeze(-1)
        projected = _linear(masked, parameters.input_weight, parameters.input_bias)
        b_values, c_values, x_values = projected.chunk(3, dim=-1)
        bx_values = b_values * x_values
        filtered = _causal_fir(bx_values, parameters.stored_fir_kernel)
        if parameters.fir_bias is not None:
            filtered = filtered + parameters.fir_bias
        writer_input = c_values * filtered
        delta = _linear(
            writer_input,
            parameters.output_weight,
            parameters.output_bias,
        )
        output = source + delta
    return Lfm2ShortConvPrimalSnapshot(
        residual=source,
        parameters=parameters,
        normalized_input=normalized,
        masked_input=masked,
        b_values=b_values,
        c_values=c_values,
        x_values=x_values,
        bx_values=bx_values,
        filtered=filtered,
        writer_input=writer_input,
        delta=delta,
        output=output,
    )


def capture_lfm2_mlp_primal(
    residual: Any,
    parameters: Lfm2MlpParameters,
) -> Lfm2MlpPrimalSnapshot:
    """Capture one independently owned, detached SwiGLU residual primal."""

    if not isinstance(parameters, Lfm2MlpParameters):
        raise TypeError("parameters must be Lfm2MlpParameters.")
    source = _owned_fp64(residual, "residual", dimensions=3)
    if (
        source.device != parameters.gate_weight.device
        or int(source.shape[-1]) != parameters.hidden_size
    ):
        raise Lfm2GlobalLinearizationError(
            "MLP residual and parameters differ in device or hidden width."
        )
    with _torch().no_grad():
        normalized = lfm2_continuous_fp64_rmsnorm(
            source,
            parameters.norm_gain,
            parameters.norm_epsilon,
        )
        gate = _linear(normalized, parameters.gate_weight)
        up = _linear(normalized, parameters.up_weight)
        activation = _torch().nn.functional.silu(gate)
        writer_input = activation * up
        delta = _linear(writer_input, parameters.down_weight)
        output = source + delta
    return Lfm2MlpPrimalSnapshot(
        residual=source,
        parameters=parameters,
        normalized_input=normalized,
        gate=gate,
        up=up,
        activation=activation,
        writer_input=writer_input,
        delta=delta,
        output=output,
    )


@dataclass(frozen=True, slots=True)
class Lfm2ResidualSegmentTangent:
    delta: Any
    output: Any


@dataclass(frozen=True, slots=True)
class Lfm2ResidualSegmentCotangent:
    incoming_residual: Any
    writer_coefficients: Any


def lfm2_attention_segment_jvp(
    snapshot: Lfm2AttentionPrimalSnapshot,
    incoming_residual_tangent: Any,
    writer_coefficients: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentTangent:
    """Coupled incoming-residual/writer JVP through an attention branch."""

    p = snapshot.parameters
    incoming = _plain_fp64(
        incoming_residual_tangent,
        "incoming_residual_tangent",
        shape=tuple(snapshot.residual.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    coefficients = _validate_coefficient(
        writer_coefficients,
        rank=int(basis.shape[1]),
        input_width=p.writer_input_width,
        device=snapshot.residual.device,
        name="attention_writer_coefficients",
    )
    batch, sequence, _ = map(int, snapshot.residual.shape)
    with _torch().no_grad():
        d_normalized = lfm2_continuous_fp64_rmsnorm_jvp(
            snapshot.residual,
            incoming,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
        dq_linear = _linear(d_normalized, p.query_weight).reshape(
            batch, sequence, p.query_heads, p.head_dim
        )
        dk_linear = _linear(d_normalized, p.key_weight).reshape(
            batch, sequence, p.key_value_heads, p.head_dim
        )
        d_value = (
            _linear(d_normalized, p.value_weight)
            .reshape(batch, sequence, p.key_value_heads, p.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        dq_pre = (
            lfm2_continuous_fp64_rmsnorm_jvp(
                snapshot.query_linear,
                dq_linear,
                p.query_norm_gain,
                p.query_key_norm_epsilon,
            )
            .transpose(1, 2)
            .contiguous()
        )
        dk_pre = (
            lfm2_continuous_fp64_rmsnorm_jvp(
                snapshot.key_linear,
                dk_linear,
                p.key_norm_gain,
                p.query_key_norm_epsilon,
            )
            .transpose(1, 2)
            .contiguous()
        )
        dq = apply_lfm2_finite_radius_rope(dq_pre, p.rope_cosine, p.rope_sine)
        dk = apply_lfm2_finite_radius_rope(dk_pre, p.rope_cosine, p.rope_sine)
        repeats = p.query_heads // p.key_value_heads
        drepeated_k = _repeat_gqa(dk, repeats)
        drepeated_v = _repeat_gqa(d_value, repeats)
        dscores = (
            dq @ snapshot.repeated_key.transpose(-1, -2)
            + snapshot.query @ drepeated_k.transpose(-1, -2)
        ) * p.scaling
        mask = p.causal_mask.expand_as(snapshot.probabilities)
        dprobabilities = _masked_softmax_jvp(
            snapshot.probabilities,
            dscores,
            mask,
        )
        dcontext = (
            dprobabilities @ snapshot.repeated_value
            + snapshot.probabilities @ drepeated_v
        )
        dwriter_input = (
            dcontext.transpose(1, 2)
            .contiguous()
            .reshape(batch, sequence, p.writer_input_width)
        )
        ddelta = _linear(dwriter_input, p.output_weight)
        ddelta = ddelta + apply_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            coefficients,
        )
        output = incoming + ddelta
    return Lfm2ResidualSegmentTangent(delta=ddelta, output=output)


def lfm2_attention_segment_vjp(
    snapshot: Lfm2AttentionPrimalSnapshot,
    output_cotangent: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentCotangent:
    """Exact transpose of :func:`lfm2_attention_segment_jvp`."""

    p = snapshot.parameters
    cotangent = _plain_fp64(
        output_cotangent,
        "output_cotangent",
        shape=tuple(snapshot.output.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    batch, sequence, _ = map(int, snapshot.residual.shape)
    with _torch().no_grad():
        coefficient_cotangent = adjoint_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            cotangent,
        )
        writer_input_cotangent = cotangent @ p.output_weight
        context_cotangent = (
            writer_input_cotangent.reshape(batch, sequence, p.query_heads, p.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        probability_cotangent = context_cotangent @ snapshot.repeated_value.transpose(
            -1, -2
        )
        repeated_value_cotangent = (
            snapshot.probabilities.transpose(-1, -2) @ context_cotangent
        )
        mask = p.causal_mask.expand_as(snapshot.probabilities)
        score_cotangent = (
            _masked_softmax_vjp(
                snapshot.probabilities,
                probability_cotangent,
                mask,
            )
            * p.scaling
        )
        query_cotangent = score_cotangent @ snapshot.repeated_key
        repeated_key_cotangent = score_cotangent.transpose(-1, -2) @ snapshot.query
        repeats = p.query_heads // p.key_value_heads
        key_cotangent = _sum_gqa_repeats(
            repeated_key_cotangent,
            p.key_value_heads,
            repeats,
        )
        value_cotangent = _sum_gqa_repeats(
            repeated_value_cotangent,
            p.key_value_heads,
            repeats,
        )
        q_pre_cotangent = (
            apply_lfm2_finite_radius_rope_transpose(
                query_cotangent,
                p.rope_cosine,
                p.rope_sine,
            )
            .transpose(1, 2)
            .contiguous()
        )
        k_pre_cotangent = (
            apply_lfm2_finite_radius_rope_transpose(
                key_cotangent,
                p.rope_cosine,
                p.rope_sine,
            )
            .transpose(1, 2)
            .contiguous()
        )
        q_linear_cotangent = lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.query_linear,
            q_pre_cotangent,
            p.query_norm_gain,
            p.query_key_norm_epsilon,
        ).reshape(batch, sequence, -1)
        k_linear_cotangent = lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.key_linear,
            k_pre_cotangent,
            p.key_norm_gain,
            p.query_key_norm_epsilon,
        ).reshape(batch, sequence, -1)
        value_linear_cotangent = (
            value_cotangent.transpose(1, 2)
            .contiguous()
            .reshape(
                batch,
                sequence,
                -1,
            )
        )
        normalized_cotangent = (
            q_linear_cotangent @ p.query_weight
            + k_linear_cotangent @ p.key_weight
            + value_linear_cotangent @ p.value_weight
        )
        residual_cotangent = cotangent + lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.residual,
            normalized_cotangent,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
    return Lfm2ResidualSegmentCotangent(
        incoming_residual=residual_cotangent,
        writer_coefficients=coefficient_cotangent,
    )


def lfm2_short_conv_segment_jvp(
    snapshot: Lfm2ShortConvPrimalSnapshot,
    incoming_residual_tangent: Any,
    writer_coefficients: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentTangent:
    """Coupled incoming-residual/writer JVP through a causal BCX branch."""

    p = snapshot.parameters
    incoming = _plain_fp64(
        incoming_residual_tangent,
        "incoming_residual_tangent",
        shape=tuple(snapshot.residual.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    coefficients = _validate_coefficient(
        writer_coefficients,
        rank=int(basis.shape[1]),
        input_width=p.writer_input_width,
        device=snapshot.residual.device,
        name="short_conv_writer_coefficients",
    )
    with _torch().no_grad():
        d_normalized = lfm2_continuous_fp64_rmsnorm_jvp(
            snapshot.residual,
            incoming,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
        d_masked = d_normalized
        if _short_conv_padding_active(p.padding_mask):
            d_masked = d_normalized * p.padding_mask.unsqueeze(-1)
        d_projected = _linear(d_masked, p.input_weight)
        db, dc, dx = d_projected.chunk(3, dim=-1)
        dbx = db * snapshot.x_values + snapshot.b_values * dx
        dfiltered = _causal_fir(dbx, p.stored_fir_kernel)
        dwriter_input = dc * snapshot.filtered + snapshot.c_values * dfiltered
        ddelta = _linear(dwriter_input, p.output_weight)
        ddelta = ddelta + apply_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            coefficients,
        )
        output = incoming + ddelta
    return Lfm2ResidualSegmentTangent(delta=ddelta, output=output)


def lfm2_short_conv_segment_vjp(
    snapshot: Lfm2ShortConvPrimalSnapshot,
    output_cotangent: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentCotangent:
    """Exact transpose of :func:`lfm2_short_conv_segment_jvp`."""

    p = snapshot.parameters
    cotangent = _plain_fp64(
        output_cotangent,
        "output_cotangent",
        shape=tuple(snapshot.output.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    with _torch().no_grad():
        coefficient_cotangent = adjoint_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            cotangent,
        )
        writer_input_cotangent = cotangent @ p.output_weight
        c_cotangent = writer_input_cotangent * snapshot.filtered
        filtered_cotangent = writer_input_cotangent * snapshot.c_values
        bx_cotangent = _causal_fir_vjp(filtered_cotangent, p.stored_fir_kernel)
        b_cotangent = bx_cotangent * snapshot.x_values
        x_cotangent = bx_cotangent * snapshot.b_values
        projected_cotangent = _torch().cat(
            (b_cotangent, c_cotangent, x_cotangent),
            dim=-1,
        )
        masked_cotangent = projected_cotangent @ p.input_weight
        normalized_cotangent = masked_cotangent
        if _short_conv_padding_active(p.padding_mask):
            normalized_cotangent = masked_cotangent * p.padding_mask.unsqueeze(-1)
        residual_cotangent = cotangent + lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.residual,
            normalized_cotangent,
            p.operator_norm_gain,
            p.operator_norm_epsilon,
        )
    return Lfm2ResidualSegmentCotangent(
        incoming_residual=residual_cotangent,
        writer_coefficients=coefficient_cotangent,
    )


def lfm2_mlp_segment_jvp(
    snapshot: Lfm2MlpPrimalSnapshot,
    incoming_residual_tangent: Any,
    writer_coefficients: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentTangent:
    """Coupled incoming-residual/writer JVP through a SwiGLU branch."""

    p = snapshot.parameters
    incoming = _plain_fp64(
        incoming_residual_tangent,
        "incoming_residual_tangent",
        shape=tuple(snapshot.residual.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    coefficients = _validate_coefficient(
        writer_coefficients,
        rank=int(basis.shape[1]),
        input_width=p.writer_input_width,
        device=snapshot.residual.device,
        name="mlp_writer_coefficients",
    )
    with _torch().no_grad():
        d_normalized = lfm2_continuous_fp64_rmsnorm_jvp(
            snapshot.residual,
            incoming,
            p.norm_gain,
            p.norm_epsilon,
        )
        dgate = _linear(d_normalized, p.gate_weight)
        dup = _linear(d_normalized, p.up_weight)
        dactivation = _silu_derivative(snapshot.gate) * dgate
        dwriter_input = dactivation * snapshot.up + snapshot.activation * dup
        ddelta = _linear(dwriter_input, p.down_weight)
        ddelta = ddelta + apply_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            coefficients,
        )
        output = incoming + ddelta
    return Lfm2ResidualSegmentTangent(delta=ddelta, output=output)


def lfm2_mlp_segment_vjp(
    snapshot: Lfm2MlpPrimalSnapshot,
    output_cotangent: Any,
    output_basis: Any,
) -> Lfm2ResidualSegmentCotangent:
    """Exact transpose of :func:`lfm2_mlp_segment_jvp`."""

    p = snapshot.parameters
    cotangent = _plain_fp64(
        output_cotangent,
        "output_cotangent",
        shape=tuple(snapshot.output.shape),
        device=snapshot.residual.device,
    )
    basis = _validate_output_basis(
        output_basis, p.hidden_size, snapshot.residual.device
    )
    with _torch().no_grad():
        coefficient_cotangent = adjoint_lfm2_global_residual_writer_block(
            snapshot.writer_input,
            basis,
            cotangent,
        )
        writer_input_cotangent = cotangent @ p.down_weight
        gate_cotangent = (
            writer_input_cotangent * snapshot.up * _silu_derivative(snapshot.gate)
        )
        up_cotangent = writer_input_cotangent * snapshot.activation
        normalized_cotangent = (
            gate_cotangent @ p.gate_weight + up_cotangent @ p.up_weight
        )
        residual_cotangent = cotangent + lfm2_continuous_fp64_rmsnorm_vjp(
            snapshot.residual,
            normalized_cotangent,
            p.norm_gain,
            p.norm_epsilon,
        )
    return Lfm2ResidualSegmentCotangent(
        incoming_residual=residual_cotangent,
        writer_coefficients=coefficient_cotangent,
    )


Lfm2OperatorPrimalSnapshot = Lfm2AttentionPrimalSnapshot | Lfm2ShortConvPrimalSnapshot


@dataclass(frozen=True, slots=True)
class Lfm2MacrocellTangent:
    post_operator_residual: Any
    post_layer_residual: Any

    @property
    def boundaries(self) -> tuple[Any, Any]:
        return (self.post_operator_residual, self.post_layer_residual)


@dataclass(frozen=True, slots=True)
class Lfm2MacrocellCotangent:
    incoming_residual: Any
    coefficient_blocks: tuple[Any, Any]


@dataclass(frozen=True, slots=True)
class Lfm2LinearizationNormBounds:
    """Conservative Frobenius-induced squared-norm upper bounds."""

    operator_squared_norm_upper_bound: float
    block_squared_norm_upper_bounds: tuple[float, ...]
    site_order: tuple[str, ...]
    boundary_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Lfm2WholeStackConditioningScales:
    """Useful unweighted local-direct scales, not certified global bounds.

    The values are in unweighted boundary-coordinate units.  A closure that
    uses coordinate weights ``W`` must multiply them by its authenticated
    maximum coordinate weight (for the current core,
    ``max(row_mass)/(boundary_count*hidden)``) before using them for ridge or
    preconditioning.  Keeping that conversion outside the primal snapshot is
    necessary because boundary masses are not primal state.
    """

    operator_squared_scale: float
    block_squared_scales: tuple[float, ...]
    site_order: tuple[str, ...]
    method: str = "local_direct_two_boundary_energy_without_downstream_product"


@dataclass(frozen=True, slots=True)
class Lfm2MacrocellLinearization:
    """One exact local callback block for the global closure operator.

    ``jvp`` and ``vjp`` match the closure core's callback signature.  The
    coupled variants additionally expose propagation of an incoming residual
    tangent/cotangent, which a whole-stack composer uses between macrocells.
    """

    operator: Lfm2OperatorPrimalSnapshot
    mlp: Lfm2MlpPrimalSnapshot
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.operator,
            (Lfm2AttentionPrimalSnapshot, Lfm2ShortConvPrimalSnapshot),
        ) or not isinstance(self.mlp, Lfm2MlpPrimalSnapshot):
            raise TypeError("operator and mlp must be typed primal snapshots.")
        if (
            self.operator.output.device != self.mlp.residual.device
            or tuple(self.operator.output.shape) != tuple(self.mlp.residual.shape)
            or not bool(_torch().equal(self.operator.output, self.mlp.residual))
        ):
            raise Lfm2GlobalLinearizationError(
                "MLP primal must be captured from this operator's exact output."
            )
        if any(
            value.device.type != "cpu"
            for value in (
                self.operator.residual,
                self.operator.writer_input,
                self.operator.output,
                self.mlp.residual,
                self.mlp.writer_input,
                self.mlp.output,
            )
        ):
            raise Lfm2GlobalLinearizationError(
                "Macrocell primal archives must remain CPU resident."
            )
        object.__setattr__(
            self,
            "authentication_sha256",
            self.recompute_authentication_sha256(),
        )

    @property
    def state_storage_device(self) -> str:
        return "cpu"

    def recompute_authentication_sha256(self) -> str:
        """Recompute the digest of every scalar and cloned primal tensor."""

        return _state_sha256((self.operator, self.mlp))

    def verify_authentication(self) -> bool:
        """Return whether the complete frozen primal archive is unchanged."""

        return self.authentication_sha256 == self.recompute_authentication_sha256()

    def _require_authenticated(self) -> None:
        if not self.verify_authentication():
            raise Lfm2GlobalLinearizationError(
                "Macrocell primal state changed after authentication."
            )

    @property
    def operator_kind(self) -> Literal["full_attention", "short_conv"]:
        if isinstance(self.operator, Lfm2AttentionPrimalSnapshot):
            return "full_attention"
        return "short_conv"

    @property
    def site_order(self) -> tuple[str, str]:
        return (f"{self.operator_kind}.output_writer", "mlp.down_writer")

    @property
    def boundary_order(self) -> tuple[str, str]:
        return ("post_operator_residual", "post_layer_residual")

    @property
    def input_widths(self) -> tuple[int, int]:
        return (
            self.operator.parameters.writer_input_width,
            self.mlp.parameters.writer_input_width,
        )

    def _operator_jvp(
        self,
        incoming: Any,
        coefficients: Any,
        basis: Any,
    ) -> Lfm2ResidualSegmentTangent:
        if isinstance(self.operator, Lfm2AttentionPrimalSnapshot):
            return lfm2_attention_segment_jvp(
                self.operator,
                incoming,
                coefficients,
                basis,
            )
        return lfm2_short_conv_segment_jvp(
            self.operator,
            incoming,
            coefficients,
            basis,
        )

    def _operator_vjp(
        self,
        cotangent: Any,
        basis: Any,
    ) -> Lfm2ResidualSegmentCotangent:
        if isinstance(self.operator, Lfm2AttentionPrimalSnapshot):
            return lfm2_attention_segment_vjp(self.operator, cotangent, basis)
        return lfm2_short_conv_segment_vjp(self.operator, cotangent, basis)

    def coupled_jvp(
        self,
        incoming_residual_tangent: Any,
        coefficient_blocks: tuple[Any, Any],
        output_basis: Any,
    ) -> Lfm2MacrocellTangent:
        """Apply incoming propagation and both direct writer perturbations."""

        self._require_authenticated()
        if (
            not isinstance(coefficient_blocks, (tuple, list))
            or len(coefficient_blocks) != 2
        ):
            raise Lfm2GlobalLinearizationError(
                "coefficient_blocks must follow the two-entry site order."
            )
        operator_tangent = self._operator_jvp(
            incoming_residual_tangent,
            coefficient_blocks[0],
            output_basis,
        )
        mlp_tangent = lfm2_mlp_segment_jvp(
            self.mlp,
            operator_tangent.output,
            coefficient_blocks[1],
            output_basis,
        )
        result = Lfm2MacrocellTangent(
            post_operator_residual=operator_tangent.output,
            post_layer_residual=mlp_tangent.output,
        )
        self._require_authenticated()
        return result

    def coupled_vjp(
        self,
        boundary_cotangents: tuple[Any, Any],
        output_basis: Any,
    ) -> Lfm2MacrocellCotangent:
        """Transpose both boundary observations into input and site blocks."""

        self._require_authenticated()
        if (
            not isinstance(boundary_cotangents, (tuple, list))
            or len(boundary_cotangents) != 2
        ):
            raise Lfm2GlobalLinearizationError(
                "boundary_cotangents must follow the two-entry boundary order."
            )
        post_operator_cotangent = _plain_fp64(
            boundary_cotangents[0],
            "post_operator_boundary_cotangent",
            shape=tuple(self.operator.output.shape),
            device=self.operator.output.device,
        )
        post_layer_cotangent = _plain_fp64(
            boundary_cotangents[1],
            "post_layer_boundary_cotangent",
            shape=tuple(self.mlp.output.shape),
            device=self.operator.output.device,
        )
        mlp_cotangent = lfm2_mlp_segment_vjp(
            self.mlp,
            post_layer_cotangent,
            output_basis,
        )
        operator_cotangent = self._operator_vjp(
            post_operator_cotangent + mlp_cotangent.incoming_residual,
            output_basis,
        )
        result = Lfm2MacrocellCotangent(
            incoming_residual=operator_cotangent.incoming_residual,
            coefficient_blocks=(
                operator_cotangent.writer_coefficients,
                mlp_cotangent.writer_coefficients,
            ),
        )
        self._require_authenticated()
        return result

    def jvp(
        self,
        coefficient_blocks: tuple[Any, Any],
        output_basis: Any,
    ) -> tuple[Any, Any]:
        """Closure-core callback: writer blocks to local boundary effects."""

        zero = _torch().zeros_like(self.operator.residual)
        return self.coupled_jvp(
            zero,
            coefficient_blocks,
            output_basis,
        ).boundaries

    def vjp(
        self,
        boundary_cotangents: tuple[Any, Any],
        output_basis: Any,
    ) -> tuple[Any, Any]:
        """Closure-core callback: local boundary cotangents to writer blocks."""

        return self.coupled_vjp(
            boundary_cotangents,
            output_basis,
        ).coefficient_blocks

    def residual_jvp(
        self,
        incoming_residual_tangent: Any,
        output_basis: Any,
    ) -> tuple[Any, Any]:
        """Propagate only an incoming tangent through the frozen macrocell."""

        basis = _validate_output_basis(
            output_basis,
            int(self.operator.residual.shape[-1]),
            self.operator.residual.device,
        )
        coefficients = tuple(
            _torch().zeros(
                (int(basis.shape[1]), width),
                dtype=_torch().float64,
                device=basis.device,
            )
            for width in self.input_widths
        )
        return self.coupled_jvp(
            incoming_residual_tangent,
            coefficients,
            basis,
        ).boundaries

    def residual_vjp(
        self,
        boundary_cotangents: tuple[Any, Any],
        output_basis: Any,
    ) -> Any:
        """Propagate only local boundary cotangents to the incoming residual."""

        return self.coupled_vjp(
            boundary_cotangents,
            output_basis,
        ).incoming_residual

    def squared_norm_upper_bounds(
        self,
        output_basis: Any,
    ) -> Lfm2LinearizationNormBounds:
        """Return safe per-site and combined callback squared-norm bounds."""

        self._require_authenticated()
        torch = _torch()
        basis = _validate_output_basis(
            output_basis,
            int(self.operator.residual.shape[-1]),
            self.operator.residual.device,
        )

        def frobenius(value: Any) -> float:
            return float(torch.linalg.vector_norm(value.reshape(-1)).item())

        p = self.mlp.parameters
        norm_lipschitz = float(p.norm_gain.abs().max().item()) / math.sqrt(
            p.norm_epsilon
        )
        producer_lipschitz = float(
            (self.mlp.up * _silu_derivative(self.mlp.gate)).abs().max().item()
        ) * frobenius(p.gate_weight) + float(
            self.mlp.activation.abs().max().item()
        ) * frobenius(p.up_weight)
        mlp_delta_lipschitz = (
            frobenius(p.down_weight) * producer_lipschitz * norm_lipschitz
        )
        mlp_residual_lipschitz = 1.0 + mlp_delta_lipschitz
        basis_norm = frobenius(basis)
        operator_direct = frobenius(self.operator.writer_input) * basis_norm
        mlp_direct = frobenius(self.mlp.writer_input) * basis_norm
        operator_block = operator_direct**2 * (1.0 + mlp_residual_lipschitz**2)
        mlp_block = mlp_direct**2
        result = Lfm2LinearizationNormBounds(
            operator_squared_norm_upper_bound=operator_block + mlp_block,
            block_squared_norm_upper_bounds=(operator_block, mlp_block),
            site_order=self.site_order,
            boundary_order=self.boundary_order,
        )
        self._require_authenticated()
        return result


def _scalar_tensor(value: int | float, *, integer: bool = False) -> Any:
    torch = _torch()
    dtype = torch.int64 if integer else torch.float64
    return torch.tensor(value, dtype=dtype)


def _macrocell_tensor_state(
    linearization: Lfm2MacrocellLinearization,
) -> dict[str, Any]:
    """Flatten one authenticated macrocell into tensor-only CPU state."""

    operator = linearization.operator
    mlp = linearization.mlp
    state: dict[str, Any] = {
        "operator.kind": _scalar_tensor(
            int(isinstance(operator, Lfm2ShortConvPrimalSnapshot)),
            integer=True,
        ),
        "operator.residual": operator.residual,
        "operator.output": operator.output,
        "operator.normalized_input": operator.normalized_input,
        "operator.norm_gain": operator.parameters.operator_norm_gain,
        "operator.norm_epsilon": _scalar_tensor(
            operator.parameters.operator_norm_epsilon
        ),
        "operator.writer_input": operator.writer_input,
        "mlp.residual": mlp.residual,
        "mlp.output": mlp.output,
        "mlp.normalized_input": mlp.normalized_input,
        "mlp.norm_gain": mlp.parameters.norm_gain,
        "mlp.norm_epsilon": _scalar_tensor(mlp.parameters.norm_epsilon),
        "mlp.gate_weight": mlp.parameters.gate_weight,
        "mlp.up_weight": mlp.parameters.up_weight,
        "mlp.down_weight": mlp.parameters.down_weight,
        "mlp.gate": mlp.gate,
        "mlp.up": mlp.up,
        "mlp.activation": mlp.activation,
        "mlp.writer_input": mlp.writer_input,
    }
    if isinstance(operator, Lfm2AttentionPrimalSnapshot):
        p = operator.parameters
        state.update(
            {
                "attention.query_weight": p.query_weight,
                "attention.key_weight": p.key_weight,
                "attention.value_weight": p.value_weight,
                "attention.output_weight": p.output_weight,
                "attention.query_norm_gain": p.query_norm_gain,
                "attention.key_norm_gain": p.key_norm_gain,
                "attention.query_key_norm_epsilon": _scalar_tensor(
                    p.query_key_norm_epsilon
                ),
                "attention.rope_cosine": p.rope_cosine,
                "attention.rope_sine": p.rope_sine,
                "attention.causal_mask": p.causal_mask,
                "attention.query_heads": _scalar_tensor(
                    p.query_heads,
                    integer=True,
                ),
                "attention.key_value_heads": _scalar_tensor(
                    p.key_value_heads,
                    integer=True,
                ),
                "attention.head_dim": _scalar_tensor(p.head_dim, integer=True),
                "attention.scaling": _scalar_tensor(float(p.scaling)),
                "attention.query_linear": operator.query_linear,
                "attention.key_linear": operator.key_linear,
                "attention.value": operator.value,
                "attention.query": operator.query,
                "attention.repeated_key": operator.repeated_key,
                "attention.repeated_value": operator.repeated_value,
                "attention.probabilities": operator.probabilities,
            }
        )
    else:
        p = operator.parameters
        state.update(
            {
                "short_conv.input_weight": p.input_weight,
                "short_conv.stored_fir_kernel": p.stored_fir_kernel,
                "short_conv.output_weight": p.output_weight,
                "short_conv.input_bias_present": _scalar_tensor(
                    int(p.input_bias is not None),
                    integer=True,
                ),
                "short_conv.fir_bias_present": _scalar_tensor(
                    int(p.fir_bias is not None),
                    integer=True,
                ),
                "short_conv.output_bias_present": _scalar_tensor(
                    int(p.output_bias is not None),
                    integer=True,
                ),
                "short_conv.padding_present": _scalar_tensor(
                    int(p.padding_mask is not None),
                    integer=True,
                ),
                "short_conv.b": operator.b_values,
                "short_conv.c": operator.c_values,
                "short_conv.x": operator.x_values,
                "short_conv.filtered": operator.filtered,
            }
        )
        if p.padding_mask is not None:
            state["short_conv.padding_mask"] = p.padding_mask
        if p.input_bias is not None:
            state["short_conv.input_bias"] = p.input_bias
        if p.fir_bias is not None:
            state["short_conv.fir_bias"] = p.fir_bias
        if p.output_bias is not None:
            state["short_conv.output_bias"] = p.output_bias
    return state


_STACK_PARAMETER_ALIASES = frozenset(
    {
        "operator.kind",
        "operator.norm_gain",
        "operator.norm_epsilon",
        "attention.query_weight",
        "attention.key_weight",
        "attention.value_weight",
        "attention.output_weight",
        "attention.query_norm_gain",
        "attention.key_norm_gain",
        "attention.query_key_norm_epsilon",
        "attention.query_heads",
        "attention.key_value_heads",
        "attention.head_dim",
        "attention.scaling",
        "short_conv.input_weight",
        "short_conv.stored_fir_kernel",
        "short_conv.output_weight",
        "short_conv.input_bias_present",
        "short_conv.fir_bias_present",
        "short_conv.output_bias_present",
        "short_conv.input_bias",
        "short_conv.fir_bias",
        "short_conv.output_bias",
        "mlp.norm_gain",
        "mlp.norm_epsilon",
        "mlp.gate_weight",
        "mlp.up_weight",
        "mlp.down_weight",
    }
)


def _macrocell_parameter_state(
    linearization: Lfm2MacrocellLinearization,
) -> dict[str, Any]:
    return {
        name: value
        for name, value in _macrocell_tensor_state(linearization).items()
        if name in _STACK_PARAMETER_ALIASES
    }


def _tensor_mapping_sha256(values: Mapping[str, Any]) -> str:
    body = {
        name: fingerprint_lfm2_global_residual_tensor(value)
        for name, value in sorted(values.items())
    }
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Lfm2StackPrimalBatch:
    """One variable-length target batch with donor-aligned boundary rows.

    Consecutive layer primals may differ by the child runtime's BF16/FP32
    storage round-trip.  The continuous folding surrogate treats that cast as
    a straight-through identity while authenticating every independently
    replayed primal.  ``selected_target_positions`` maps donor-bearing rows
    into the flattened full target sequence; ``canonical_boundary_row_indices``
    maps those same selected rows into the concatenated donor archive.
    """

    macrocells: tuple[Lfm2MacrocellLinearization, ...]
    selected_target_positions: Any
    canonical_boundary_row_indices: Any
    terminal: Lfm2TerminalNormPrimalSnapshot | None = None
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        macrocells = tuple(self.macrocells)
        if not macrocells or not all(
            isinstance(value, Lfm2MacrocellLinearization)
            and value.verify_authentication()
            for value in macrocells
        ):
            raise Lfm2GlobalLinearizationError(
                "A stack batch must contain one or more authenticated macrocells."
            )
        reference_shape = tuple(macrocells[0].operator.residual.shape)
        hidden = int(reference_shape[-1])
        for macrocell in macrocells:
            if (
                tuple(macrocell.operator.residual.shape) != reference_shape
                or int(macrocell.operator.residual.shape[-1]) != hidden
            ):
                raise Lfm2GlobalLinearizationError(
                    "Stack batch macrocells disagree on full target geometry."
                )
        selected = self.selected_target_positions
        canonical = self.canonical_boundary_row_indices
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if any(
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.dtype not in integer_dtypes
            or value.ndim != 1
            for value in (selected, canonical)
        ) or int(selected.numel()) != int(canonical.numel()):
            raise Lfm2GlobalLinearizationError(
                "Selected target positions and canonical boundary rows must be "
                "equal-length dense integer vectors."
            )
        archived_selected = (
            selected.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()
        )
        archived_canonical = (
            canonical.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()
        )
        full_target_rows = math.prod(reference_shape[:-1])
        if (
            int(archived_selected.numel()) == 0
            or bool((archived_selected < 0).any().item())
            or bool((archived_selected >= full_target_rows).any().item())
            or int(torch.unique(archived_selected).numel())
            != int(archived_selected.numel())
            or bool((archived_canonical < 0).any().item())
            or int(torch.unique(archived_canonical).numel())
            != int(archived_canonical.numel())
        ):
            raise Lfm2GlobalLinearizationError(
                "Target selections must be in range and both row maps must be unique."
            )
        terminal = self.terminal
        if terminal is not None:
            if (
                not isinstance(terminal, Lfm2TerminalNormPrimalSnapshot)
                or tuple(terminal.residual.shape) != reference_shape
            ):
                raise Lfm2GlobalLinearizationError(
                    "Terminal norm primal must use the full target geometry."
                )
        object.__setattr__(self, "macrocells", macrocells)
        object.__setattr__(self, "selected_target_positions", archived_selected)
        object.__setattr__(
            self,
            "canonical_boundary_row_indices",
            archived_canonical,
        )
        object.__setattr__(
            self,
            "authentication_sha256",
            self.recompute_authentication_sha256(),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.macrocells[0].operator.residual.shape[-1])

    @property
    def row_count(self) -> int:
        return int(self.canonical_boundary_row_indices.numel())

    @property
    def full_target_row_count(self) -> int:
        shape = self.macrocells[0].operator.residual.shape
        return math.prod(shape[:-1])

    def recompute_authentication_sha256(self) -> str:
        return _state_sha256(
            (
                self.macrocells,
                self.selected_target_positions,
                self.canonical_boundary_row_indices,
                self.terminal,
                "inter_layer_runtime_cast_straight_through_identity",
            )
        )

    def verify_authentication(self) -> bool:
        return (
            all(value.verify_authentication() for value in self.macrocells)
            and self.authentication_sha256 == self.recompute_authentication_sha256()
        )


@dataclass(frozen=True, slots=True)
class Lfm2WholeStackPrimalSnapshot:
    """Authenticated ragged-batch primal bank with shared layer parameters."""

    batches: tuple[Lfm2StackPrimalBatch, ...]
    layer_parameter_sha256: tuple[str, ...] = field(init=False)
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        batches = tuple(self.batches)
        if not batches or not all(
            isinstance(value, Lfm2StackPrimalBatch) and value.verify_authentication()
            for value in batches
        ):
            raise Lfm2GlobalLinearizationError(
                "batches must contain authenticated LFM2 stack primals."
            )
        hidden = batches[0].hidden_size
        layer_count = len(batches[0].macrocells)
        terminal_present = batches[0].terminal is not None
        if any(
            batch.hidden_size != hidden
            or len(batch.macrocells) != layer_count
            or (batch.terminal is not None) != terminal_present
            for batch in batches
        ):
            raise Lfm2GlobalLinearizationError(
                "Stack batches disagree on layer count, hidden width, or terminal "
                "boundary."
            )
        parameter_digests: list[str] = []
        for layer_index in range(layer_count):
            reference = _macrocell_parameter_state(batches[0].macrocells[layer_index])
            digest = _tensor_mapping_sha256(reference)
            for batch in batches[1:]:
                candidate = _macrocell_parameter_state(batch.macrocells[layer_index])
                if set(candidate) != set(reference) or any(
                    not bool(torch.equal(reference[name], candidate[name]))
                    for name in reference
                ):
                    raise Lfm2GlobalLinearizationError(
                        f"Layer {layer_index} parameter state differs across batches."
                    )
            parameter_digests.append(digest)
        if terminal_present:
            reference_terminal = batches[0].terminal.parameters
            for batch in batches[1:]:
                candidate = batch.terminal.parameters
                if candidate.epsilon != reference_terminal.epsilon or not bool(
                    torch.equal(candidate.gain, reference_terminal.gain)
                ):
                    raise Lfm2GlobalLinearizationError(
                        "Terminal norm parameters differ across batches."
                    )
        all_indices = torch.cat(
            [batch.canonical_boundary_row_indices for batch in batches]
        )
        total_rows = int(all_indices.numel())
        if not bool(
            torch.equal(
                torch.sort(all_indices).values,
                torch.arange(total_rows, dtype=torch.int64),
            )
        ):
            raise Lfm2GlobalLinearizationError(
                "Donor-aligned target row maps must partition every canonical row."
            )
        object.__setattr__(self, "batches", batches)
        object.__setattr__(
            self,
            "layer_parameter_sha256",
            tuple(parameter_digests),
        )
        object.__setattr__(
            self,
            "authentication_sha256",
            self.recompute_authentication_sha256(),
        )

    @property
    def hidden_size(self) -> int:
        return self.batches[0].hidden_size

    @property
    def row_count(self) -> int:
        return sum(batch.row_count for batch in self.batches)

    @property
    def layer_count(self) -> int:
        return len(self.batches[0].macrocells)

    @property
    def terminal_present(self) -> bool:
        return self.batches[0].terminal is not None

    def recompute_authentication_sha256(self) -> str:
        return _state_sha256((self.batches, self.layer_parameter_sha256))

    def verify_authentication(self) -> bool:
        return (
            all(batch.verify_authentication() for batch in self.batches)
            and self.authentication_sha256 == self.recompute_authentication_sha256()
        )

    def conditioning_scales(
        self,
        output_basis: Any,
    ) -> Lfm2WholeStackConditioningScales:
        """Return deterministic local-direct scales for numerical conditioning.

        These values intentionally omit products through later residual blocks;
        they are useful ridge/preconditioner scales, not publication claims or
        certified norms.  They are also unweighted; the closure adapter must
        multiply them by its authenticated maximum coordinate weight.  Use
        :meth:`squared_norm_upper_bounds` when a true (usually much more
        conservative) unweighted global upper bound is required.
        """

        return _whole_stack_norm_evidence(self, output_basis)[0]

    def squared_norm_upper_bounds(
        self,
        output_basis: Any,
    ) -> Lfm2LinearizationNormBounds:
        """Return conservative bounds including every downstream boundary."""

        return _whole_stack_norm_evidence(self, output_basis)[1]


def _matrix_spectral_norm_upper(value: Any) -> float:
    """Cheap induced 1/inf upper bound on a (possibly batched) 2-norm."""

    absolute = value.abs()
    column_norm = absolute.sum(dim=-2).amax(dim=-1)
    row_norm = absolute.sum(dim=-1).amax(dim=-1)
    return float((column_norm * row_norm).sqrt().amax().item())


def _rmsnorm_linearization_norm_upper(
    rows: Any,
    gain: Any,
    epsilon: float,
) -> float:
    inverse_rms = (rows.square().mean(dim=-1) + epsilon).rsqrt().amax()
    return float((gain.abs().amax() * inverse_rms).item())


def _terminal_transfer_norm_upper(
    snapshot: Lfm2TerminalNormPrimalSnapshot | None,
) -> float:
    if snapshot is None:
        return 0.0
    return _rmsnorm_linearization_norm_upper(
        snapshot.residual,
        snapshot.parameters.gain,
        snapshot.parameters.epsilon,
    )


def _macrocell_norm_factors(
    macrocell: Lfm2MacrocellLinearization,
    output_basis: Any,
) -> tuple[float, float, float, float]:
    """Return operator/MLP transfer and direct-writer norm factors."""

    operator = macrocell.operator
    basis_norm = _matrix_spectral_norm_upper(output_basis)
    operator_direct = basis_norm * _matrix_spectral_norm_upper(
        operator.writer_input.reshape(-1, operator.parameters.writer_input_width)
    )
    operator_norm = _rmsnorm_linearization_norm_upper(
        operator.residual,
        operator.parameters.operator_norm_gain,
        operator.parameters.operator_norm_epsilon,
    )
    if isinstance(operator, Lfm2AttentionPrimalSnapshot):
        p = operator.parameters
        repeats = p.query_heads // p.key_value_heads
        query_norm = _rmsnorm_linearization_norm_upper(
            operator.query_linear,
            p.query_norm_gain,
            p.query_key_norm_epsilon,
        )
        key_norm = _rmsnorm_linearization_norm_upper(
            operator.key_linear,
            p.key_norm_gain,
            p.query_key_norm_epsilon,
        )
        half = p.head_dim // 2
        rope_gain = float(
            (p.rope_cosine[..., :half].square() + p.rope_sine[..., :half].square())
            .sqrt()
            .amax()
            .item()
        )
        query_lipschitz = (
            operator_norm
            * _matrix_spectral_norm_upper(p.query_weight)
            * query_norm
            * rope_gain
        )
        repeated_key_lipschitz = (
            operator_norm
            * _matrix_spectral_norm_upper(p.key_weight)
            * key_norm
            * rope_gain
            * math.sqrt(repeats)
        )
        repeated_value_lipschitz = (
            operator_norm
            * _matrix_spectral_norm_upper(p.value_weight)
            * math.sqrt(repeats)
        )
        score_lipschitz = p.scaling * (
            _matrix_spectral_norm_upper(operator.repeated_key) * query_lipschitz
            + _matrix_spectral_norm_upper(operator.query) * repeated_key_lipschitz
        )
        softmax_lipschitz = float(
            (2.0 * operator.probabilities * (1.0 - operator.probabilities))
            .amax()
            .item()
        )
        context_lipschitz = (
            _matrix_spectral_norm_upper(operator.repeated_value)
            * softmax_lipschitz
            * score_lipschitz
            + _matrix_spectral_norm_upper(operator.probabilities)
            * repeated_value_lipschitz
        )
        operator_transfer = 1.0 + (
            _matrix_spectral_norm_upper(p.output_weight) * context_lipschitz
        )
    else:
        p = operator.parameters
        padding_gain = 1.0
        if p.padding_mask is not None:
            padding_gain = float(p.padding_mask.abs().amax().item())
        projected_lipschitz = (
            operator_norm * padding_gain * _matrix_spectral_norm_upper(p.input_weight)
        )
        fir_lipschitz = float(p.stored_fir_kernel.abs().sum(dim=-1).amax().item())
        writer_input_lipschitz = projected_lipschitz * (
            float(operator.filtered.abs().amax().item())
            + float(operator.c_values.abs().amax().item())
            * fir_lipschitz
            * (
                float(operator.x_values.abs().amax().item())
                + float(operator.b_values.abs().amax().item())
            )
        )
        operator_transfer = 1.0 + (
            _matrix_spectral_norm_upper(p.output_weight) * writer_input_lipschitz
        )
    mlp = macrocell.mlp
    mp = mlp.parameters
    mlp_norm = _rmsnorm_linearization_norm_upper(
        mlp.residual,
        mp.norm_gain,
        mp.norm_epsilon,
    )
    producer_lipschitz = float(
        (mlp.up * _silu_derivative(mlp.gate)).abs().amax().item()
    ) * _matrix_spectral_norm_upper(mp.gate_weight) + float(
        mlp.activation.abs().amax().item()
    ) * _matrix_spectral_norm_upper(mp.up_weight)
    mlp_transfer = 1.0 + (
        _matrix_spectral_norm_upper(mp.down_weight) * producer_lipschitz * mlp_norm
    )
    mlp_direct = basis_norm * _matrix_spectral_norm_upper(
        mlp.writer_input.reshape(-1, mp.writer_input_width)
    )
    factors = (
        operator_transfer,
        mlp_transfer,
        operator_direct,
        mlp_direct,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in factors):
        raise Lfm2GlobalLinearizationError(
            "Whole-stack analytic norm evidence is non-finite."
        )
    return factors


def _whole_stack_norm_evidence(
    snapshot: Lfm2WholeStackPrimalSnapshot,
    output_basis: Any,
) -> tuple[Lfm2WholeStackConditioningScales, Lfm2LinearizationNormBounds]:
    """Build separate useful local scales and certified global bounds."""

    if not snapshot.verify_authentication():
        raise Lfm2GlobalLinearizationError(
            "Whole-stack primal state changed before norm analysis."
        )
    basis = _validate_output_basis(
        output_basis,
        snapshot.hidden_size,
        snapshot.batches[0].macrocells[0].operator.residual.device,
    )
    per_batch_factors = tuple(
        tuple(
            _macrocell_norm_factors(macrocell, basis) for macrocell in batch.macrocells
        )
        for batch in snapshot.batches
    )
    conditioning_blocks = [0.0] * 28
    certified_blocks = [0.0] * 28
    for batch, layer_factors in zip(
        snapshot.batches,
        per_batch_factors,
        strict=True,
    ):
        terminal_transfer = _terminal_transfer_norm_upper(batch.terminal)
        for layer_index, (_, mlp_transfer, operator_direct, mlp_direct) in enumerate(
            layer_factors
        ):
            operator_site = 2 * layer_index
            mlp_site = operator_site + 1
            conditioning_blocks[operator_site] += operator_direct**2 * (
                1.0 + mlp_transfer**2
            )
            conditioning_blocks[mlp_site] += mlp_direct**2
            for site_index, amplitude in (
                (operator_site, operator_direct),
                (mlp_site, mlp_direct),
            ):
                energy = 0.0
                if site_index == operator_site:
                    energy += amplitude**2
                    amplitude *= mlp_transfer
                energy += amplitude**2
                for later_layer in range(layer_index + 1, 14):
                    later_operator, later_mlp, _, _ = layer_factors[later_layer]
                    amplitude *= later_operator
                    energy += amplitude**2
                    amplitude *= later_mlp
                    energy += amplitude**2
                if snapshot.terminal_present:
                    energy += (amplitude * terminal_transfer) ** 2
                certified_blocks[site_index] += energy
    conditioning_tuple = tuple(conditioning_blocks)
    certified_tuple = tuple(certified_blocks)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (*conditioning_tuple, *certified_tuple)
    ):
        raise Lfm2GlobalLinearizationError(
            "Whole-stack downstream norm bound overflowed; it must not be clipped "
            "or mislabeled as a useful conditioning scale."
        )
    site_order = tuple(
        f"layer.{layer_index}."
        + ("operator.output_writer" if offset == 0 else "mlp.down_writer")
        for layer_index in range(14)
        for offset in range(2)
    )
    boundary_order = tuple(
        kind
        for _ in range(14)
        for kind in ("post_operator_residual", "post_layer_residual")
    ) + (("terminal_observable",) if snapshot.terminal_present else ())
    conditioning = Lfm2WholeStackConditioningScales(
        operator_squared_scale=math.fsum(conditioning_tuple),
        block_squared_scales=conditioning_tuple,
        site_order=site_order,
    )
    certified = Lfm2LinearizationNormBounds(
        operator_squared_norm_upper_bound=math.fsum(certified_tuple),
        block_squared_norm_upper_bounds=certified_tuple,
        site_order=site_order,
        boundary_order=boundary_order,
    )
    if not snapshot.verify_authentication():
        raise Lfm2GlobalLinearizationError(
            "Whole-stack primal state changed during norm analysis."
        )
    return conditioning, certified


def _macrocell_views(state: Any) -> tuple[Any, Any]:
    """Build ephemeral tensor-only views over authenticated base-owned state."""

    hidden = int(state["operator.residual"].shape[-1])
    operator_kind = int(state["operator.kind"].item())
    if operator_kind == 0:
        query_heads = int(state["attention.query_heads"].item())
        key_value_heads = int(state["attention.key_value_heads"].item())
        head_dim = int(state["attention.head_dim"].item())
        parameters = SimpleNamespace(
            hidden_size=hidden,
            writer_input_width=query_heads * head_dim,
            operator_norm_gain=state["operator.norm_gain"],
            operator_norm_epsilon=float(state["operator.norm_epsilon"].item()),
            query_weight=state["attention.query_weight"],
            key_weight=state["attention.key_weight"],
            value_weight=state["attention.value_weight"],
            output_weight=state["attention.output_weight"],
            query_norm_gain=state["attention.query_norm_gain"],
            key_norm_gain=state["attention.key_norm_gain"],
            query_key_norm_epsilon=float(
                state["attention.query_key_norm_epsilon"].item()
            ),
            rope_cosine=state["attention.rope_cosine"],
            rope_sine=state["attention.rope_sine"],
            causal_mask=state["attention.causal_mask"],
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
            scaling=float(state["attention.scaling"].item()),
        )
        operator = SimpleNamespace(
            residual=state["operator.residual"],
            output=state["operator.output"],
            normalized_input=state["operator.normalized_input"],
            parameters=parameters,
            query_linear=state["attention.query_linear"],
            key_linear=state["attention.key_linear"],
            value=state["attention.value"],
            query=state["attention.query"],
            repeated_key=state["attention.repeated_key"],
            repeated_value=state["attention.repeated_value"],
            probabilities=state["attention.probabilities"],
            writer_input=state["operator.writer_input"],
        )
    elif operator_kind == 1:
        padding = (
            state["short_conv.padding_mask"]
            if int(state["short_conv.padding_present"].item())
            else None
        )
        parameters = SimpleNamespace(
            hidden_size=hidden,
            writer_input_width=hidden,
            operator_norm_gain=state["operator.norm_gain"],
            operator_norm_epsilon=float(state["operator.norm_epsilon"].item()),
            input_weight=state["short_conv.input_weight"],
            stored_fir_kernel=state["short_conv.stored_fir_kernel"],
            output_weight=state["short_conv.output_weight"],
            padding_mask=padding,
        )
        operator = SimpleNamespace(
            residual=state["operator.residual"],
            output=state["operator.output"],
            normalized_input=state["operator.normalized_input"],
            parameters=parameters,
            b_values=state["short_conv.b"],
            c_values=state["short_conv.c"],
            x_values=state["short_conv.x"],
            filtered=state["short_conv.filtered"],
            writer_input=state["operator.writer_input"],
        )
    else:  # pragma: no cover - state authentication fixes this at construction
        raise Lfm2GlobalLinearizationError("Unknown tensor-only operator kind.")
    mlp_parameters = SimpleNamespace(
        hidden_size=hidden,
        writer_input_width=int(state["mlp.gate_weight"].shape[0]),
        norm_gain=state["mlp.norm_gain"],
        norm_epsilon=float(state["mlp.norm_epsilon"].item()),
        gate_weight=state["mlp.gate_weight"],
        up_weight=state["mlp.up_weight"],
        down_weight=state["mlp.down_weight"],
    )
    mlp = SimpleNamespace(
        residual=state["mlp.residual"],
        output=state["mlp.output"],
        normalized_input=state["mlp.normalized_input"],
        parameters=mlp_parameters,
        gate=state["mlp.gate"],
        up=state["mlp.up"],
        activation=state["mlp.activation"],
        writer_input=state["mlp.writer_input"],
    )
    return operator, mlp


def _tensor_only_coupled_jvp(
    state: Any,
    coefficient_blocks: tuple[Any, Any],
    output_basis: Any,
) -> Lfm2MacrocellTangent:
    operator, mlp = _macrocell_views(state)
    incoming = _torch().zeros_like(operator.residual)
    if int(state["operator.kind"].item()) == 0:
        operator_tangent = lfm2_attention_segment_jvp(
            operator,
            incoming,
            coefficient_blocks[0],
            output_basis,
        )
    else:
        operator_tangent = lfm2_short_conv_segment_jvp(
            operator,
            incoming,
            coefficient_blocks[0],
            output_basis,
        )
    mlp_tangent = lfm2_mlp_segment_jvp(
        mlp,
        operator_tangent.output,
        coefficient_blocks[1],
        output_basis,
    )
    return Lfm2MacrocellTangent(
        post_operator_residual=operator_tangent.output,
        post_layer_residual=mlp_tangent.output,
    )


def _tensor_only_coupled_vjp(
    state: Any,
    boundary_cotangents: tuple[Any, Any],
    output_basis: Any,
) -> Lfm2MacrocellCotangent:
    operator, mlp = _macrocell_views(state)
    mlp_cotangent = lfm2_mlp_segment_vjp(
        mlp,
        boundary_cotangents[1],
        output_basis,
    )
    post_operator_cotangent = boundary_cotangents[0] + mlp_cotangent.incoming_residual
    if int(state["operator.kind"].item()) == 0:
        operator_cotangent = lfm2_attention_segment_vjp(
            operator,
            post_operator_cotangent,
            output_basis,
        )
    else:
        operator_cotangent = lfm2_short_conv_segment_vjp(
            operator,
            post_operator_cotangent,
            output_basis,
        )
    return Lfm2MacrocellCotangent(
        incoming_residual=operator_cotangent.incoming_residual,
        coefficient_blocks=(
            operator_cotangent.writer_coefficients,
            mlp_cotangent.writer_coefficients,
        ),
    )


class Lfm2MacrocellLinearizationKernel(Lfm2GlobalResidualLinearizationKernel):
    """Authenticated CPU-resident, row-streamed two-boundary kernel.

    The base class owns the only public execution path and guards the complete
    cloned tensor archive before and after every chunk.  This subclass stores
    no model or Python closure state; its two implementations consume only the
    authenticated tensor mapping supplied by the base.
    """

    __slots__ = ()

    def __init__(
        self,
        topology: Lfm2GlobalResidualWriterTopology,
        boundaries: Lfm2GlobalResidualBoundaryBundle,
        linearization: Lfm2MacrocellLinearization,
        *,
        linearization_receipt: Any,
    ) -> None:
        if not isinstance(linearization, Lfm2MacrocellLinearization) or not (
            linearization.verify_authentication()
        ):
            raise Lfm2GlobalLinearizationError(
                "linearization must be an authenticated macrocell primal."
            )
        if not isinstance(topology, Lfm2GlobalResidualWriterTopology) or not (
            topology.verify_authentication()
        ):
            raise Lfm2GlobalLinearizationError(
                "topology must be an authenticated writer topology."
            )
        if not isinstance(boundaries, Lfm2GlobalResidualBoundaryBundle) or not (
            boundaries.verify_authentication()
        ):
            raise Lfm2GlobalLinearizationError(
                "boundaries must be an authenticated boundary bundle."
            )
        expected_writer_kind = (
            "attention_output"
            if linearization.operator_kind == "full_attention"
            else "short_conv_output"
        )
        if (
            topology.hidden_size != int(linearization.operator.residual.shape[-1])
            or len(topology.sites) != 2
            or tuple(site.input_width for site in topology.sites)
            != linearization.input_widths
            or tuple(site.writer_kind for site in topology.sites)
            != (expected_writer_kind, "mlp_down")
        ):
            raise Lfm2GlobalLinearizationError(
                "Topology does not describe this macrocell's two writer sites."
            )
        if (
            len(boundaries.boundaries) != 2
            or tuple(boundary.boundary_kind for boundary in boundaries.boundaries)
            != linearization.boundary_order
            or tuple(tuple(boundary.defect.shape) for boundary in boundaries.boundaries)
            != (
                tuple(linearization.operator.output.shape),
                tuple(linearization.mlp.output.shape),
            )
        ):
            raise Lfm2GlobalLinearizationError(
                "Boundary bundle does not describe this macrocell's two outputs."
            )
        state = _macrocell_tensor_state(linearization)
        state_sha256 = {
            name: fingerprint_lfm2_global_residual_tensor(value)
            for name, value in state.items()
        }
        super().__init__(
            topology,
            boundaries,
            tensor_state=state,
            tensor_state_sha256=state_sha256,
            linearization_receipt=linearization_receipt,
            transpose_contract_body={
                "kernel": "lfm2_continuous_fp64_macrocell_v1",
                "boundary_order": list(linearization.boundary_order),
                "site_order": list(linearization.site_order),
                "attention_transpose": (
                    "RMSNorm.T -> RoPE.T -> GQA-sum -> masked-softmax.T -> "
                    "QKV-projection.T"
                ),
                "short_conv_transpose": (
                    "BCX-products.T -> causal-FIR.T -> input-projection.T"
                ),
                "mlp_transpose": ("down-writer.T -> SwiGLU-product.T -> RMSNorm.T"),
                "residual_additions_transposed": True,
                "writer_block_transpose": ("U.T@output_cotangent.T@producer_rows"),
            },
        )

    def _jvp_chunk_tensor_only(
        self,
        state: Any,
        coefficient_blocks: tuple[Any, ...],
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> Any:
        torch = _torch()
        device = output_basis.device
        basis = output_basis.to(device="cpu", dtype=torch.float64).contiguous()
        coefficients = tuple(
            value.to(device="cpu", dtype=torch.float64).contiguous()
            for value in coefficient_blocks
        )
        with torch.no_grad():
            effects = _tensor_only_coupled_jvp(state, coefficients, basis).boundaries
            if boundary_index not in {0, 1}:
                raise Lfm2GlobalLinearizationError(
                    "boundary_index is outside this macrocell."
                )
            flat = effects[boundary_index].reshape(
                -1,
                int(effects[boundary_index].shape[-1]),
            )
            if row_start < 0 or row_stop > int(flat.shape[0]) or row_start >= row_stop:
                raise Lfm2GlobalLinearizationError(
                    "Requested JVP row chunk is outside the boundary."
                )
            return flat[row_start:row_stop].to(device=device).contiguous()

    def _vjp_chunk_tensor_only(
        self,
        state: Any,
        boundary_cotangent: Any,
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> tuple[Any, ...]:
        torch = _torch()
        device = output_basis.device
        if boundary_index not in {0, 1}:
            raise Lfm2GlobalLinearizationError(
                "boundary_index is outside this macrocell."
            )
        basis = output_basis.to(device="cpu", dtype=torch.float64).contiguous()
        boundary_shape = tuple(
            state["operator.output" if boundary_index == 0 else "mlp.output"].shape
        )
        zero_cotangents = [
            torch.zeros_like(state["operator.output"]),
            torch.zeros_like(state["mlp.output"]),
        ]
        flat = zero_cotangents[boundary_index].reshape(-1, boundary_shape[-1])
        if row_start < 0 or row_stop > int(flat.shape[0]) or row_start >= row_stop:
            raise Lfm2GlobalLinearizationError(
                "Requested VJP row chunk is outside the boundary."
            )
        if tuple(boundary_cotangent.shape) != (
            row_stop - row_start,
            boundary_shape[-1],
        ):
            raise Lfm2GlobalLinearizationError(
                "Boundary cotangent chunk has incorrect geometry."
            )
        flat[row_start:row_stop] = boundary_cotangent.to(
            device="cpu",
            dtype=torch.float64,
        )
        with torch.no_grad():
            result = _tensor_only_coupled_vjp(
                state,
                (zero_cotangents[0], zero_cotangents[1]),
                basis,
            ).coefficient_blocks
        return tuple(value.to(device=device).contiguous() for value in result)


def _whole_stack_tensor_state(
    snapshot: Lfm2WholeStackPrimalSnapshot,
) -> dict[str, Any]:
    """Pack shared layer parameters once and ragged primal batches separately."""

    state: dict[str, Any] = {
        "stack.layer_count": _scalar_tensor(snapshot.layer_count, integer=True),
        "stack.batch_count": _scalar_tensor(len(snapshot.batches), integer=True),
        "stack.row_count": _scalar_tensor(snapshot.row_count, integer=True),
        "stack.hidden_size": _scalar_tensor(snapshot.hidden_size, integer=True),
        "stack.terminal_present": _scalar_tensor(
            int(snapshot.terminal_present),
            integer=True,
        ),
    }
    for layer_index in range(snapshot.layer_count):
        local = _macrocell_tensor_state(snapshot.batches[0].macrocells[layer_index])
        for alias, value in local.items():
            if alias in _STACK_PARAMETER_ALIASES:
                state[f"layer.{layer_index:03d}.{alias}"] = value
    if snapshot.terminal_present:
        terminal = snapshot.batches[0].terminal
        state["terminal.norm_gain"] = terminal.parameters.gain
        state["terminal.norm_epsilon"] = _scalar_tensor(terminal.parameters.epsilon)
    for batch_index, batch in enumerate(snapshot.batches):
        batch_prefix = f"batch.{batch_index:04d}"
        state[f"{batch_prefix}.selected_target_positions"] = (
            batch.selected_target_positions
        )
        state[f"{batch_prefix}.canonical_boundary_row_indices"] = (
            batch.canonical_boundary_row_indices
        )
        for layer_index, macrocell in enumerate(batch.macrocells):
            local = _macrocell_tensor_state(macrocell)
            for alias, value in local.items():
                if alias not in _STACK_PARAMETER_ALIASES:
                    state[f"{batch_prefix}.layer.{layer_index:03d}.{alias}"] = value
        if batch.terminal is not None:
            state[f"{batch_prefix}.terminal.residual"] = batch.terminal.residual
            state[f"{batch_prefix}.terminal.output"] = batch.terminal.output
    return state


def _prefixed_state(
    state: Mapping[str, Any],
    prefix: str,
    *,
    device: Any,
) -> dict[str, Any]:
    result = {}
    for name, value in state.items():
        if name.startswith(prefix):
            alias = name[len(prefix) :]
            result[alias] = value.to(device=device).contiguous()
    return result


def _stack_local_state(
    state: Mapping[str, Any],
    *,
    batch_index: int,
    layer_index: int,
    layer_parameters: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    batch_prefix = f"batch.{batch_index:04d}.layer.{layer_index:03d}."
    result = dict(layer_parameters)
    result.update(_prefixed_state(state, batch_prefix, device=device))
    return result


def _validate_stack_bundle_inputs(
    state: Mapping[str, Any],
    coefficient_blocks: Sequence[Any],
    output_basis: Any,
) -> tuple[tuple[Any, ...], int, int, int, bool]:
    layer_count = int(state["stack.layer_count"].item())
    batch_count = int(state["stack.batch_count"].item())
    row_count = int(state["stack.row_count"].item())
    hidden = int(state["stack.hidden_size"].item())
    terminal_present = bool(int(state["stack.terminal_present"].item()))
    coefficients = tuple(coefficient_blocks)
    if (
        layer_count != 14
        or len(coefficients) != 2 * layer_count
        or output_basis.ndim != 2
        or int(output_basis.shape[0]) != hidden
    ):
        raise Lfm2GlobalLinearizationError(
            "Whole-stack coefficients, basis, or authenticated geometry is invalid."
        )
    return coefficients, batch_count, row_count, hidden, terminal_present


def _workspace_bytes(values: Mapping[str, Any], *extra: Any) -> int:
    total = sum(
        int(value.numel()) * int(value.element_size()) for value in values.values()
    )
    total += sum(int(value.numel()) * int(value.element_size()) for value in extra)
    return total


def _stack_jvp_bundle_tensor_only(
    state: Mapping[str, Any],
    coefficient_blocks: Sequence[Any],
    output_basis: Any,
    *,
    maximum_workspace_bytes: int,
) -> tuple[Any, ...]:
    """One layer-major forward sweep over every ragged primal batch."""

    torch = _torch()
    coefficients, batch_count, row_count, hidden, terminal_present = (
        _validate_stack_bundle_inputs(state, coefficient_blocks, output_basis)
    )
    device = output_basis.device
    boundary_count = 28 + int(terminal_present)
    effects = tuple(
        torch.zeros((row_count, hidden), dtype=torch.float64, device="cpu")
        for _ in range(boundary_count)
    )
    incoming = [
        torch.zeros_like(state[f"batch.{batch_index:04d}.layer.000.operator.residual"])
        for batch_index in range(batch_count)
    ]
    with torch.no_grad():
        for layer_index in range(14):
            parameter_prefix = f"layer.{layer_index:03d}."
            layer_parameters = _prefixed_state(
                state,
                parameter_prefix,
                device=device,
            )
            for batch_index in range(batch_count):
                local = _stack_local_state(
                    state,
                    batch_index=batch_index,
                    layer_index=layer_index,
                    layer_parameters=layer_parameters,
                    device=device,
                )
                staged_incoming = incoming[batch_index].to(device=device).contiguous()
                if (
                    _workspace_bytes(
                        local,
                        staged_incoming,
                        output_basis,
                        coefficients[2 * layer_index],
                        coefficients[2 * layer_index + 1],
                    )
                    > maximum_workspace_bytes
                ):
                    raise Lfm2GlobalLinearizationError(
                        "A whole-stack JVP batch exceeds maximum_workspace_bytes."
                    )
                operator, mlp = _macrocell_views(local)
                if int(local["operator.kind"].item()) == 0:
                    operator_tangent = lfm2_attention_segment_jvp(
                        operator,
                        staged_incoming,
                        coefficients[2 * layer_index],
                        output_basis,
                    )
                else:
                    operator_tangent = lfm2_short_conv_segment_jvp(
                        operator,
                        staged_incoming,
                        coefficients[2 * layer_index],
                        output_basis,
                    )
                mlp_tangent = lfm2_mlp_segment_jvp(
                    mlp,
                    operator_tangent.output,
                    coefficients[2 * layer_index + 1],
                    output_basis,
                )
                selected_positions = state[
                    f"batch.{batch_index:04d}.selected_target_positions"
                ]
                canonical_indices = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                effects[2 * layer_index].index_copy_(
                    0,
                    canonical_indices,
                    operator_tangent.output.detach()
                    .to(device="cpu")
                    .reshape(-1, hidden)
                    .index_select(0, selected_positions),
                )
                effects[2 * layer_index + 1].index_copy_(
                    0,
                    canonical_indices,
                    mlp_tangent.output.detach()
                    .to(device="cpu")
                    .reshape(-1, hidden)
                    .index_select(0, selected_positions),
                )
                incoming[batch_index] = mlp_tangent.output.detach().to(device="cpu")
        if terminal_present:
            gain = state["terminal.norm_gain"].to(device=device).contiguous()
            epsilon = float(state["terminal.norm_epsilon"].item())
            for batch_index in range(batch_count):
                residual = state[f"batch.{batch_index:04d}.terminal.residual"].to(
                    device=device
                )
                tangent = incoming[batch_index].to(device=device)
                terminal_effect = lfm2_continuous_fp64_rmsnorm_jvp(
                    residual,
                    tangent,
                    gain,
                    epsilon,
                )
                selected_positions = state[
                    f"batch.{batch_index:04d}.selected_target_positions"
                ]
                canonical_indices = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                effects[-1].index_copy_(
                    0,
                    canonical_indices,
                    terminal_effect.detach()
                    .to(device="cpu")
                    .reshape(-1, hidden)
                    .index_select(0, selected_positions),
                )
    return tuple(value.contiguous() for value in effects)


def _scatter_scaled_batch_cotangent(
    cotangent: Any,
    scale: Any | None,
    selected_target_positions: Any,
    canonical_boundary_row_indices: Any,
    *,
    batch_shape: tuple[int, ...],
    device: Any,
) -> Any:
    hidden = int(batch_shape[-1])
    selected = cotangent.index_select(0, canonical_boundary_row_indices)
    if scale is not None:
        selected = selected * scale.index_select(
            0,
            canonical_boundary_row_indices,
        ).unsqueeze(-1)
    full = _torch().zeros(
        (math.prod(batch_shape[:-1]), hidden),
        dtype=_torch().float64,
        device=device,
    )
    full.index_copy_(
        0,
        selected_target_positions.to(device=device),
        selected.to(device=device),
    )
    return full.reshape(batch_shape).contiguous()


def _stack_vjp_bundle_tensor_only(
    state: Mapping[str, Any],
    boundary_cotangents_cpu: Sequence[Any],
    output_basis: Any,
    *,
    boundary_row_scales_cpu: Sequence[Any] | None,
    maximum_workspace_bytes: int,
) -> tuple[Any, ...]:
    """One reverse layer sweep with exact direct-boundary injections."""

    torch = _torch()
    layer_count = int(state["stack.layer_count"].item())
    batch_count = int(state["stack.batch_count"].item())
    row_count = int(state["stack.row_count"].item())
    hidden = int(state["stack.hidden_size"].item())
    terminal_present = bool(int(state["stack.terminal_present"].item()))
    if (
        layer_count != 14
        or output_basis.ndim != 2
        or int(output_basis.shape[0]) != hidden
    ):
        raise Lfm2GlobalLinearizationError(
            "Whole-stack basis or authenticated geometry is invalid."
        )
    cotangents = tuple(boundary_cotangents_cpu)
    expected_boundaries = 28 + int(terminal_present)
    if len(cotangents) != expected_boundaries or any(
        tuple(value.shape) != (row_count, hidden) for value in cotangents
    ):
        raise Lfm2GlobalLinearizationError(
            "Whole-stack VJP cotangent bundle has incorrect geometry."
        )
    scales = None if boundary_row_scales_cpu is None else tuple(boundary_row_scales_cpu)
    if scales is not None and (
        len(scales) != expected_boundaries
        or any(tuple(value.shape) != (row_count,) for value in scales)
    ):
        raise Lfm2GlobalLinearizationError(
            "Whole-stack VJP row-scale bundle has incorrect geometry."
        )
    device = output_basis.device
    coefficient_cotangents: list[Any] = []
    for layer_index in range(14):
        local_parameter_state = _prefixed_state(
            state,
            f"layer.{layer_index:03d}.",
            device=device,
        )
        operator_width = int(
            (
                local_parameter_state.get("attention.output_weight")
                if int(local_parameter_state["operator.kind"].item()) == 0
                else local_parameter_state["short_conv.output_weight"]
            ).shape[1]
        )
        mlp_width = int(local_parameter_state["mlp.down_weight"].shape[1])
        coefficient_cotangents.extend(
            (
                torch.zeros(
                    (int(output_basis.shape[1]), operator_width),
                    dtype=torch.float64,
                    device=device,
                ),
                torch.zeros(
                    (int(output_basis.shape[1]), mlp_width),
                    dtype=torch.float64,
                    device=device,
                ),
            )
        )
    downstream = [
        torch.zeros_like(state[f"batch.{batch_index:04d}.layer.013.mlp.output"])
        for batch_index in range(batch_count)
    ]
    with torch.no_grad():
        if terminal_present:
            gain = state["terminal.norm_gain"].to(device=device).contiguous()
            epsilon = float(state["terminal.norm_epsilon"].item())
            for batch_index in range(batch_count):
                selected_positions = state[
                    f"batch.{batch_index:04d}.selected_target_positions"
                ]
                canonical_indices = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                residual = state[f"batch.{batch_index:04d}.terminal.residual"]
                terminal_cotangent = _scatter_scaled_batch_cotangent(
                    cotangents[-1],
                    None if scales is None else scales[-1],
                    selected_positions,
                    canonical_indices,
                    batch_shape=tuple(residual.shape),
                    device=device,
                )
                downstream[batch_index] = lfm2_continuous_fp64_rmsnorm_vjp(
                    residual.to(device=device),
                    terminal_cotangent,
                    gain,
                    epsilon,
                ).to(device="cpu")
        for layer_index in range(13, -1, -1):
            layer_parameters = _prefixed_state(
                state,
                f"layer.{layer_index:03d}.",
                device=device,
            )
            for batch_index in range(batch_count):
                local = _stack_local_state(
                    state,
                    batch_index=batch_index,
                    layer_index=layer_index,
                    layer_parameters=layer_parameters,
                    device=device,
                )
                if (
                    _workspace_bytes(
                        local,
                        downstream[batch_index],
                        output_basis,
                    )
                    > maximum_workspace_bytes
                ):
                    raise Lfm2GlobalLinearizationError(
                        "A whole-stack VJP batch exceeds maximum_workspace_bytes."
                    )
                operator, mlp = _macrocell_views(local)
                selected_positions = state[
                    f"batch.{batch_index:04d}.selected_target_positions"
                ]
                canonical_indices = state[
                    f"batch.{batch_index:04d}.canonical_boundary_row_indices"
                ]
                operator_cotangent = _scatter_scaled_batch_cotangent(
                    cotangents[2 * layer_index],
                    None if scales is None else scales[2 * layer_index],
                    selected_positions,
                    canonical_indices,
                    batch_shape=tuple(operator.output.shape),
                    device=device,
                )
                layer_cotangent = _scatter_scaled_batch_cotangent(
                    cotangents[2 * layer_index + 1],
                    None if scales is None else scales[2 * layer_index + 1],
                    selected_positions,
                    canonical_indices,
                    batch_shape=tuple(mlp.output.shape),
                    device=device,
                ) + downstream[batch_index].to(device=device)
                mlp_result = lfm2_mlp_segment_vjp(
                    mlp,
                    layer_cotangent,
                    output_basis,
                )
                combined_operator_cotangent = (
                    operator_cotangent + mlp_result.incoming_residual
                )
                if int(local["operator.kind"].item()) == 0:
                    operator_result = lfm2_attention_segment_vjp(
                        operator,
                        combined_operator_cotangent,
                        output_basis,
                    )
                else:
                    operator_result = lfm2_short_conv_segment_vjp(
                        operator,
                        combined_operator_cotangent,
                        output_basis,
                    )
                coefficient_cotangents[2 * layer_index].add_(
                    operator_result.writer_coefficients
                )
                coefficient_cotangents[2 * layer_index + 1].add_(
                    mlp_result.writer_coefficients
                )
                downstream[batch_index] = operator_result.incoming_residual.to(
                    device="cpu"
                )
    return tuple(value.contiguous() for value in coefficient_cotangents)


class Lfm2WholeStackLinearizationKernel(Lfm2GlobalResidualLinearizationKernel):
    """Authenticated 14-layer/28-writer ragged-batch bundle kernel."""

    __slots__ = ()

    def __init__(
        self,
        topology: Lfm2GlobalResidualWriterTopology,
        boundaries: Lfm2GlobalResidualBoundaryBundle,
        snapshot: Lfm2WholeStackPrimalSnapshot,
        *,
        linearization_receipt: Any,
    ) -> None:
        if not isinstance(snapshot, Lfm2WholeStackPrimalSnapshot) or not (
            snapshot.verify_authentication()
        ):
            raise Lfm2GlobalLinearizationError(
                "snapshot must be an authenticated whole-stack primal bank."
            )
        if (
            not topology.verify_authentication()
            or not boundaries.verify_authentication()
        ):
            raise Lfm2GlobalLinearizationError(
                "Whole-stack topology and boundaries must be authenticated."
            )
        if topology.hidden_size != snapshot.hidden_size or len(topology.sites) != 28:
            raise Lfm2GlobalLinearizationError(
                "Whole-stack topology must contain exactly 28 LFM2 writer sites."
            )
        for layer_index in range(14):
            macrocell = snapshot.batches[0].macrocells[layer_index]
            expected_kind = (
                "attention_output"
                if macrocell.operator_kind == "full_attention"
                else "short_conv_output"
            )
            operator_site = topology.sites[2 * layer_index]
            mlp_site = topology.sites[2 * layer_index + 1]
            if (
                operator_site.layer_index != layer_index
                or mlp_site.layer_index != layer_index
                or operator_site.writer_kind != expected_kind
                or mlp_site.writer_kind != "mlp_down"
                or (operator_site.input_width, mlp_site.input_width)
                != macrocell.input_widths
            ):
                raise Lfm2GlobalLinearizationError(
                    f"Writer topology differs from layer {layer_index} primals."
                )
        expected_boundary_count = 28 + int(snapshot.terminal_present)
        if len(boundaries.boundaries) != expected_boundary_count:
            raise Lfm2GlobalLinearizationError(
                "Boundary bundle has the wrong residual/terminal inventory."
            )
        for layer_index in range(14):
            pair = boundaries.boundaries[2 * layer_index : 2 * layer_index + 2]
            if tuple(value.layer_index for value in pair) != (
                layer_index,
                layer_index,
            ) or tuple(value.boundary_kind for value in pair) != (
                "post_operator_residual",
                "post_layer_residual",
            ):
                raise Lfm2GlobalLinearizationError(
                    f"Boundary order differs at layer {layer_index}."
                )
        if snapshot.terminal_present and (
            boundaries.boundaries[-1].boundary_kind != "terminal_observable"
            or boundaries.boundaries[-1].layer_index != 13
        ):
            raise Lfm2GlobalLinearizationError(
                "The optional terminal boundary must follow layer 13."
            )
        if any(
            boundary.defect.numel() // snapshot.hidden_size != snapshot.row_count
            or int(boundary.defect.shape[-1]) != snapshot.hidden_size
            for boundary in boundaries.boundaries
        ):
            raise Lfm2GlobalLinearizationError(
                "Every whole-stack boundary must use the donor-aligned row bank."
            )
        state = _whole_stack_tensor_state(snapshot)
        state_sha256 = {
            name: fingerprint_lfm2_global_residual_tensor(value)
            for name, value in state.items()
        }
        super().__init__(
            topology,
            boundaries,
            tensor_state=state,
            tensor_state_sha256=state_sha256,
            linearization_receipt=linearization_receipt,
            transpose_contract_body={
                "kernel": "lfm2_continuous_fp64_whole_stack_v1",
                "layers": 14,
                "writer_sites": 28,
                "residual_boundaries": 28,
                "terminal_hidden_observable": snapshot.terminal_present,
                "token_bridge_row_mapping": {
                    "full_target_to_donor_rows": "selected_target_positions",
                    "donor_rows_to_archive": "canonical_boundary_row_indices",
                },
                "parameter_state_storage": "once_per_layer",
                "primal_state_storage": "separate_per_variable_length_batch",
                "inter_layer_runtime_cast": "straight_through_identity",
                "forward": "incoming_tangent_composed_layer_0_through_13",
                "transpose": "boundary_plus_downstream_cotangent_layer_13_to_0",
                "terminal_transpose": "final_rmsnorm_vjp"
                if snapshot.terminal_present
                else None,
                "bundle_fast_path": True,
                "mutable_bundle_cache": False,
            },
        )

    def _jvp_bundle_tensor_only(
        self,
        state: Mapping[str, Any],
        coefficient_blocks: tuple[Any, ...],
        output_basis: Any,
        *,
        maximum_workspace_bytes: int,
    ) -> tuple[Any, ...]:
        return _stack_jvp_bundle_tensor_only(
            state,
            coefficient_blocks,
            output_basis,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )

    def _vjp_bundle_tensor_only(
        self,
        state: Mapping[str, Any],
        boundary_cotangents_cpu: tuple[Any, ...],
        output_basis: Any,
        *,
        boundary_row_scales_cpu: tuple[Any, ...] | None,
        maximum_workspace_bytes: int,
    ) -> tuple[Any, ...]:
        return _stack_vjp_bundle_tensor_only(
            state,
            boundary_cotangents_cpu,
            output_basis,
            boundary_row_scales_cpu=boundary_row_scales_cpu,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )

    def _jvp_chunk_tensor_only(
        self,
        state: Mapping[str, Any],
        coefficient_blocks: tuple[Any, ...],
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> Any:
        bundle = _stack_jvp_bundle_tensor_only(
            state,
            coefficient_blocks,
            output_basis,
            maximum_workspace_bytes=2**63 - 1,
        )
        return bundle[boundary_index][row_start:row_stop].to(device=output_basis.device)

    def _vjp_chunk_tensor_only(
        self,
        state: Mapping[str, Any],
        boundary_cotangent: Any,
        output_basis: Any,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ) -> tuple[Any, ...]:
        row_count = int(state["stack.row_count"].item())
        hidden = int(state["stack.hidden_size"].item())
        boundary_count = 28 + int(state["stack.terminal_present"].item())
        cotangents = [
            _torch().zeros(
                (row_count, hidden),
                dtype=_torch().float64,
                device="cpu",
            )
            for _ in range(boundary_count)
        ]
        cotangents[boundary_index][row_start:row_stop].copy_(
            boundary_cotangent.to(device="cpu")
        )
        return _stack_vjp_bundle_tensor_only(
            state,
            tuple(cotangents),
            output_basis,
            boundary_row_scales_cpu=None,
            maximum_workspace_bytes=2**63 - 1,
        )


def capture_lfm2_macrocell_linearization(
    residual: Any,
    operator_parameters: Lfm2AttentionParameters | Lfm2ShortConvParameters,
    mlp_parameters: Lfm2MlpParameters,
) -> Lfm2MacrocellLinearization:
    """Capture a complete operator-plus-SwiGLU macrocell primal."""

    if isinstance(operator_parameters, Lfm2AttentionParameters):
        operator = capture_lfm2_attention_primal(residual, operator_parameters)
    elif isinstance(operator_parameters, Lfm2ShortConvParameters):
        operator = capture_lfm2_short_conv_primal(residual, operator_parameters)
    else:
        raise TypeError(
            "operator_parameters must describe LFM2 attention or short-conv."
        )
    mlp = capture_lfm2_mlp_primal(operator.output, mlp_parameters)
    return Lfm2MacrocellLinearization(operator=operator, mlp=mlp)


__all__ = [
    "Lfm2AttentionParameters",
    "Lfm2AttentionPrimalSnapshot",
    "Lfm2GlobalLinearizationError",
    "Lfm2LinearizationNormBounds",
    "Lfm2MacrocellCotangent",
    "Lfm2MacrocellLinearization",
    "Lfm2MacrocellTangent",
    "Lfm2MlpParameters",
    "Lfm2MlpPrimalSnapshot",
    "Lfm2ResidualSegmentCotangent",
    "Lfm2ResidualSegmentTangent",
    "Lfm2ShortConvParameters",
    "Lfm2ShortConvPrimalSnapshot",
    "Lfm2StackPrimalBatch",
    "Lfm2TerminalNormParameters",
    "Lfm2TerminalNormPrimalSnapshot",
    "Lfm2WholeStackLinearizationKernel",
    "Lfm2WholeStackConditioningScales",
    "Lfm2WholeStackPrimalSnapshot",
    "capture_lfm2_attention_primal",
    "capture_lfm2_macrocell_linearization",
    "capture_lfm2_mlp_primal",
    "capture_lfm2_short_conv_primal",
    "capture_lfm2_terminal_norm_primal",
    "lfm2_attention_segment_jvp",
    "lfm2_attention_segment_vjp",
    "lfm2_continuous_fp64_rmsnorm",
    "lfm2_continuous_fp64_rmsnorm_jvp",
    "lfm2_continuous_fp64_rmsnorm_vjp",
    "lfm2_mlp_segment_jvp",
    "lfm2_mlp_segment_vjp",
    "lfm2_short_conv_segment_jvp",
    "lfm2_short_conv_segment_vjp",
    "lfm2_terminal_norm_jvp",
    "lfm2_terminal_norm_vjp",
]
