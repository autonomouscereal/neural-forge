"""Target-native, read-only replay of one installed LFM2 macrocell.

This module is the bounded child-side replay primitive needed by the CPCF
compiler.  It consumes caller-supplied *actual child* prefill residual states
and reads one already-installed target ``Lfm2DecoderLayer``.  It never accepts
a parent model, cache, tokenizer, optimizer, or training callback.

The replay follows the eager Transformers LFM2 algebra explicitly instead of
calling the layer or its submodule ``forward`` methods.  That makes the
operation hook-free and exposes the exact two residual writes:

``residual -> operator RMSNorm -> attention/short-conv -> residual add``
``         -> FFN RMSNorm      -> SwiGLU              -> residual add``

Attention requires caller-supplied target-compatible RoPE cosine/sine tensors
and an additive full-prefix causal mask.  Short convolution is always replayed
as a causal full-prefix FIR, without a recurrent cache.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


LFM2_CPCF_MACROCELL_REPLAY_SCHEMA = "neural-forge-lfm2-cpcf-macrocell-replay-v1"


class Lfm2CpcfMacrocellReplayError(ValueError):
    """An installed target layer cannot be replayed under the strict contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF macrocell replay."
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
        raise Lfm2CpcfMacrocellReplayError(
            "Replay metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Only tensors can be byte-authenticated.")
    detached = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(detached),
        "shape": [int(size) for size in detached.shape],
        "layout": str(detached.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(detached.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _dense_floating_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
    finite: bool = True,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise Lfm2CpcfMacrocellReplayError(
            f"{name} must be a nonempty dense real floating rank-{dimensions} tensor."
        )
    if finite and not bool(torch.isfinite(value.detach()).all().item()):
        raise Lfm2CpcfMacrocellReplayError(f"{name} contains NaN or infinity.")
    return value


def _module_modes(layer: Any) -> tuple[tuple[str, bool], ...]:
    named_modules = getattr(layer, "named_modules", None)
    if not callable(named_modules):
        raise Lfm2CpcfMacrocellReplayError(
            "target_layer must be a torch module with named_modules()."
        )
    return tuple((str(name), bool(module.training)) for name, module in named_modules())


@dataclass(frozen=True, slots=True)
class _ModuleSnapshot:
    state_sha256: str
    tensor_count: int
    parameter_count: int
    identities: tuple[tuple[Any, ...], ...]
    modes: tuple[tuple[str, bool], ...]


def _module_snapshot(layer: Any) -> _ModuleSnapshot:
    torch = _torch()
    named_parameters = getattr(layer, "named_parameters", None)
    named_buffers = getattr(layer, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        raise Lfm2CpcfMacrocellReplayError(
            "target_layer must expose named parameters and buffers."
        )

    rows: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    parameter_count = 0
    iterators = (
        ("parameter", named_parameters(recurse=True, remove_duplicate=False)),
        ("buffer", named_buffers(recurse=True, remove_duplicate=False)),
    )
    for kind, iterator in iterators:
        for name, value in iterator:
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.device.type == "meta"
            ):
                raise Lfm2CpcfMacrocellReplayError(
                    f"Target {kind} {name!r} is not a materialized dense tensor."
                )
            if torch.is_floating_point(value) and not bool(
                torch.isfinite(value.detach()).all().item()
            ):
                raise Lfm2CpcfMacrocellReplayError(
                    f"Target {kind} {name!r} contains NaN or infinity."
                )
            grad_sha256 = None
            if kind == "parameter":
                parameter_count += int(value.numel())
                if value.grad is not None:
                    grad_sha256 = _tensor_sha256(value.grad)
            tensor_sha256 = _tensor_sha256(value)
            rows.append(
                {
                    "kind": kind,
                    "name": str(name),
                    "shape": [int(size) for size in value.shape],
                    "dtype": _dtype_name(value),
                    "device": str(value.device),
                    "requires_grad": bool(value.requires_grad),
                    "tensor_sha256": tensor_sha256,
                    "gradient_sha256": grad_sha256,
                }
            )
            identities.append(
                (
                    kind,
                    str(name),
                    id(value),
                    int(value.data_ptr()),
                    tuple(int(size) for size in value.shape),
                    _dtype_name(value),
                    str(value.device),
                    bool(value.requires_grad),
                    grad_sha256,
                    tensor_sha256,
                )
            )
    body = {
        "schema": LFM2_CPCF_MACROCELL_REPLAY_SCHEMA,
        "scope": "target_layer_state",
        "tensors": rows,
    }
    return _ModuleSnapshot(
        state_sha256=_sha(body),
        tensor_count=len(rows),
        parameter_count=parameter_count,
        identities=tuple(identities),
        modes=_module_modes(layer),
    )


def _weight(module: Any, name: str, *, dimensions: int) -> Any:
    value = getattr(module, "weight", None)
    return _dense_floating_tensor(
        value,
        f"{name}.weight",
        dimensions=dimensions,
    )


def _bias(module: Any, name: str, *, size: int, allowed: bool) -> Any | None:
    torch = _torch()
    value = getattr(module, "bias", None)
    if value is None:
        return None
    if not allowed:
        raise Lfm2CpcfMacrocellReplayError(
            f"{name}.bias is outside the LFM2 target contract."
        )
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or tuple(value.shape) != (size,)
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise Lfm2CpcfMacrocellReplayError(
            f"{name}.bias must be a finite dense vector of length {size}."
        )
    return value


def _same_runtime(
    reference: Any,
    values: Mapping[str, Any],
) -> None:
    for name, value in values.items():
        if value.device != reference.device:
            raise Lfm2CpcfMacrocellReplayError(
                f"{name} is on {value.device}, not target input device "
                f"{reference.device}."
            )
        if value.dtype != reference.dtype:
            raise Lfm2CpcfMacrocellReplayError(
                f"{name} has dtype {value.dtype}, not target input dtype "
                f"{reference.dtype}."
            )


def _linear(
    module: Any,
    inputs: Any,
    name: str,
    *,
    expected_output: int,
    bias_allowed: bool,
) -> Any:
    torch = _torch()
    weight = _weight(module, name, dimensions=2)
    if tuple(weight.shape) != (expected_output, int(inputs.shape[-1])):
        raise Lfm2CpcfMacrocellReplayError(
            f"{name}.weight has shape {tuple(weight.shape)}, expected "
            f"({expected_output}, {int(inputs.shape[-1])})."
        )
    bias = _bias(
        module,
        name,
        size=expected_output,
        allowed=bias_allowed,
    )
    _same_runtime(
        inputs,
        {
            f"{name}.weight": weight,
            **({} if bias is None else {f"{name}.bias": bias}),
        },
    )
    return torch.nn.functional.linear(inputs, weight, bias)


def _rms_norm(module: Any, inputs: Any, name: str) -> Any:
    torch = _torch()
    weight = _weight(module, name, dimensions=1)
    width = int(inputs.shape[-1])
    if tuple(weight.shape) != (width,):
        raise Lfm2CpcfMacrocellReplayError(
            f"{name}.weight has shape {tuple(weight.shape)}, expected ({width},)."
        )
    _same_runtime(inputs, {f"{name}.weight": weight})
    epsilon = getattr(module, "variance_epsilon", None)
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) < 0.0
    ):
        raise Lfm2CpcfMacrocellReplayError(
            f"{name}.variance_epsilon must be finite and nonnegative."
        )

    input_dtype = inputs.dtype
    normalized = inputs.to(torch.float32)
    variance = normalized.pow(2).mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + float(epsilon))
    return weight * normalized.to(input_dtype)


def _rotate_half(value: Any) -> Any:
    torch = _torch()
    midpoint = int(value.shape[-1]) // 2
    first = value[..., :midpoint]
    second = value[..., midpoint:]
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(value: Any, repeats: int) -> Any:
    batch, heads, sequence, head_dim = (
        int(value.shape[0]),
        int(value.shape[1]),
        int(value.shape[2]),
        int(value.shape[3]),
    )
    if repeats == 1:
        return value
    expanded = value[:, :, None, :, :].expand(
        batch,
        heads,
        repeats,
        sequence,
        head_dim,
    )
    return expanded.reshape(batch, heads * repeats, sequence, head_dim)


def _validate_position_embeddings(
    position_embeddings: Any,
    residual: Any,
    *,
    head_dim: int,
) -> tuple[Any, Any]:
    if (
        not isinstance(position_embeddings, (tuple, list))
        or len(position_embeddings) != 2
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "Attention replay requires a (cos, sin) position_embeddings pair."
        )
    cos = _dense_floating_tensor(
        position_embeddings[0],
        "position_embeddings.cos",
        dimensions=3,
    )
    sin = _dense_floating_tensor(
        position_embeddings[1],
        "position_embeddings.sin",
        dimensions=3,
    )
    batch = int(residual.shape[0])
    sequence = int(residual.shape[1])
    allowed = {(batch, sequence, head_dim), (1, sequence, head_dim)}
    if tuple(cos.shape) not in allowed or tuple(sin.shape) != tuple(cos.shape):
        raise Lfm2CpcfMacrocellReplayError(
            "RoPE cos/sin must have equal shape [batch|1, sequence, head_dim]."
        )
    _same_runtime(
        residual,
        {
            "position_embeddings.cos": cos,
            "position_embeddings.sin": sin,
        },
    )
    return cos, sin


def _validate_additive_mask(
    value: Any,
    residual: Any,
    *,
    query_heads: int,
) -> Any:
    torch = _torch()
    mask = _dense_floating_tensor(
        value,
        "additive_causal_mask",
        dimensions=4,
        finite=False,
    )
    if mask.device != residual.device:
        raise Lfm2CpcfMacrocellReplayError(
            "additive_causal_mask is not on the target input device."
        )
    if bool(torch.isnan(mask.detach()).any().item()) or bool(
        torch.isposinf(mask.detach()).any().item()
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "additive_causal_mask contains NaN or positive infinity."
        )
    batch = int(residual.shape[0])
    sequence = int(residual.shape[1])
    target_shape = (batch, query_heads, sequence, sequence)
    try:
        broadcast = torch.broadcast_shapes(tuple(mask.shape), target_shape)
    except RuntimeError as exc:
        raise Lfm2CpcfMacrocellReplayError(
            "additive_causal_mask cannot broadcast to target attention scores."
        ) from exc
    if tuple(broadcast) != target_shape:
        raise Lfm2CpcfMacrocellReplayError(
            "additive_causal_mask broadcasts outside the target score shape."
        )
    expanded = mask.expand(target_shape)
    if sequence > 1:
        future = torch.triu(
            torch.ones(
                (sequence, sequence),
                dtype=torch.bool,
                device=mask.device,
            ),
            diagonal=1,
        )
        if not bool((expanded[..., future] < 0).all().item()):
            raise Lfm2CpcfMacrocellReplayError(
                "additive_causal_mask does not mask every future position."
            )
    if bool((expanded > 0).any().item()):
        raise Lfm2CpcfMacrocellReplayError(
            "additive_causal_mask contains a positive additive score."
        )
    return mask


def _attention_replay(
    layer: Any,
    operator_input: Any,
    *,
    position_embeddings: Any,
    additive_causal_mask: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch = _torch()
    attention = getattr(layer, "self_attn", None)
    if attention is None or getattr(layer, "conv", None) is not None:
        raise Lfm2CpcfMacrocellReplayError(
            "Attention target must expose self_attn and no conv operator."
        )
    hidden = int(operator_input.shape[-1])
    head_dim = getattr(attention, "head_dim", None)
    if (
        isinstance(head_dim, bool)
        or not isinstance(head_dim, int)
        or head_dim <= 0
        or head_dim % 2
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "Target attention head_dim must be a positive even integer."
        )

    q_weight = _weight(attention.q_proj, "self_attn.q_proj", dimensions=2)
    k_weight = _weight(attention.k_proj, "self_attn.k_proj", dimensions=2)
    v_weight = _weight(attention.v_proj, "self_attn.v_proj", dimensions=2)
    if (
        int(q_weight.shape[1]) != hidden
        or int(k_weight.shape[1]) != hidden
        or int(v_weight.shape[1]) != hidden
        or int(q_weight.shape[0]) % head_dim
        or int(k_weight.shape[0]) % head_dim
        or tuple(v_weight.shape) != tuple(k_weight.shape)
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "Target attention Q/K/V tensors do not form an LFM2 topology."
        )
    query_heads = int(q_weight.shape[0]) // head_dim
    key_value_heads = int(k_weight.shape[0]) // head_dim
    if query_heads % key_value_heads:
        raise Lfm2CpcfMacrocellReplayError(
            "Target query heads do not divide into GQA KV groups."
        )
    groups = query_heads // key_value_heads
    declared_groups = getattr(attention, "num_key_value_groups", groups)
    if declared_groups != groups:
        raise Lfm2CpcfMacrocellReplayError(
            "Target num_key_value_groups differs from Q/K/V tensor topology."
        )
    scaling = getattr(attention, "scaling", None)
    expected_scaling = head_dim**-0.5
    if (
        isinstance(scaling, bool)
        or not isinstance(scaling, (int, float))
        or not math.isclose(
            float(scaling),
            expected_scaling,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "Target attention scaling differs from head_dim**-0.5."
        )
    out_weight = _weight(
        attention.out_proj,
        "self_attn.out_proj",
        dimensions=2,
    )
    if tuple(out_weight.shape) != (hidden, query_heads * head_dim):
        raise Lfm2CpcfMacrocellReplayError(
            "Target attention output projection has an incompatible shape."
        )
    for module, name, size in (
        (attention.q_proj, "self_attn.q_proj", query_heads * head_dim),
        (attention.k_proj, "self_attn.k_proj", key_value_heads * head_dim),
        (attention.v_proj, "self_attn.v_proj", key_value_heads * head_dim),
        (attention.out_proj, "self_attn.out_proj", hidden),
    ):
        _bias(module, name, size=size, allowed=False)

    cos, sin = _validate_position_embeddings(
        position_embeddings,
        operator_input,
        head_dim=head_dim,
    )
    mask = _validate_additive_mask(
        additive_causal_mask,
        operator_input,
        query_heads=query_heads,
    )
    batch, sequence = (
        int(operator_input.shape[0]),
        int(operator_input.shape[1]),
    )
    query_linear = _linear(
        attention.q_proj,
        operator_input,
        "self_attn.q_proj",
        expected_output=query_heads * head_dim,
        bias_allowed=False,
    ).view(batch, sequence, query_heads, head_dim)
    key_linear = _linear(
        attention.k_proj,
        operator_input,
        "self_attn.k_proj",
        expected_output=key_value_heads * head_dim,
        bias_allowed=False,
    ).view(batch, sequence, key_value_heads, head_dim)
    value = _linear(
        attention.v_proj,
        operator_input,
        "self_attn.v_proj",
        expected_output=key_value_heads * head_dim,
        bias_allowed=False,
    ).view(batch, sequence, key_value_heads, head_dim)

    query_pre_rope = _rms_norm(
        attention.q_layernorm,
        query_linear,
        "self_attn.q_layernorm",
    ).transpose(1, 2)
    key_pre_rope = _rms_norm(
        attention.k_layernorm,
        key_linear,
        "self_attn.k_layernorm",
    ).transpose(1, 2)
    value = value.transpose(1, 2)
    cos_heads = cos.unsqueeze(1)
    sin_heads = sin.unsqueeze(1)
    query = (query_pre_rope * cos_heads) + (_rotate_half(query_pre_rope) * sin_heads)
    key = (key_pre_rope * cos_heads) + (_rotate_half(key_pre_rope) * sin_heads)
    repeated_key = _repeat_kv(key, groups)
    repeated_value = _repeat_kv(value, groups)
    scores = torch.matmul(query, repeated_key.transpose(2, 3))
    scores = scores * float(scaling)
    scores = scores + mask
    probabilities = torch.nn.functional.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    context = torch.matmul(probabilities, repeated_value)
    concatenated = context.transpose(1, 2).contiguous()
    concatenated = concatenated.reshape(batch, sequence, -1).contiguous()
    output = _linear(
        attention.out_proj,
        concatenated,
        "self_attn.out_proj",
        expected_output=hidden,
        bias_allowed=False,
    )
    details = {
        "query_linear": query_linear,
        "key_linear": key_linear,
        "query_normalized_pre_rope": query_pre_rope,
        "key_normalized_pre_rope": key_pre_rope,
        "query": query,
        "key": key,
        "value": value,
        "scores": scores,
        "probabilities": probabilities,
        "context": context,
        "concatenated_context": concatenated,
    }
    topology = {
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "query_heads_per_kv_group": groups,
        "head_dim": head_dim,
        "attention_scaling": float(scaling),
        "rope_shape": [int(size) for size in cos.shape],
        "additive_mask_shape": [int(size) for size in mask.shape],
        "softmax_accumulation_dtype": "float32",
        "softmax_output_dtype": _dtype_name(probabilities),
    }
    return output, details, topology


def _validate_padding_mask(value: Any, residual: Any) -> Any | None:
    torch = _torch()
    if value is None:
        return None
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or tuple(value.shape) != tuple(residual.shape[:2])
        or value.device != residual.device
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "padding_mask must be a dense [batch, sequence] tensor on the "
            "target input device."
        )
    if torch.is_floating_point(value):
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise Lfm2CpcfMacrocellReplayError("padding_mask contains NaN or infinity.")
    elif value.dtype != torch.bool and value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "padding_mask must be boolean, integer, or floating."
        )
    return value


def _short_conv_replay(
    layer: Any,
    operator_input: Any,
    *,
    padding_mask: Any | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch = _torch()
    short_conv = getattr(layer, "conv", None)
    if short_conv is None or getattr(layer, "self_attn", None) is not None:
        raise Lfm2CpcfMacrocellReplayError(
            "Short-conv target must expose conv and no self_attn operator."
        )
    hidden = int(operator_input.shape[-1])
    sequence = int(operator_input.shape[1])
    mask = _validate_padding_mask(padding_mask, operator_input)

    convolution = getattr(short_conv, "conv", None)
    if convolution is None:
        raise Lfm2CpcfMacrocellReplayError(
            "Target short-conv operator lacks its depthwise FIR module."
        )
    fir_weight = _weight(
        convolution,
        "conv.conv",
        dimensions=3,
    )
    if (
        int(fir_weight.shape[0]) != hidden
        or int(fir_weight.shape[1]) != 1
        or int(fir_weight.shape[2]) <= 0
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "Target FIR weight must have shape [hidden, 1, positive_length]."
        )
    fir_length = int(fir_weight.shape[2])
    if (
        getattr(convolution, "groups", None) != hidden
        or tuple(getattr(convolution, "stride", ())) != (1,)
        or tuple(getattr(convolution, "dilation", ())) != (1,)
        or tuple(getattr(convolution, "padding", ())) != (fir_length - 1,)
        or getattr(short_conv, "L_cache", fir_length) != fir_length
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "Target FIR is not the standard causal depthwise LFM2 geometry."
        )
    declared_bias = getattr(short_conv, "bias", None)
    if not isinstance(declared_bias, bool):
        raise Lfm2CpcfMacrocellReplayError(
            "Target short-conv bias declaration must be boolean."
        )
    fir_bias = _bias(
        convolution,
        "conv.conv",
        size=hidden,
        allowed=declared_bias,
    )
    if declared_bias != (fir_bias is not None):
        raise Lfm2CpcfMacrocellReplayError(
            "Target FIR bias presence differs from the operator declaration."
        )
    _same_runtime(
        operator_input,
        {
            "conv.conv.weight": fir_weight,
            **({} if fir_bias is None else {"conv.conv.bias": fir_bias}),
        },
    )

    masked_input = operator_input
    # This deliberately matches Transformers LFM2's
    # apply_mask_to_padding_states contract, including its batch/length guard.
    if mask is not None and int(mask.shape[1]) > 1 and int(mask.shape[0]) > 1:
        masked_input = (operator_input * mask[:, :, None]).to(operator_input.dtype)

    projected = _linear(
        short_conv.in_proj,
        masked_input,
        "conv.in_proj",
        expected_output=3 * hidden,
        bias_allowed=declared_bias,
    ).transpose(-1, -2)
    b_values, c_values, x_values = projected.chunk(3, dim=-2)
    bx_values = b_values * x_values
    filtered = torch.nn.functional.conv1d(
        bx_values,
        fir_weight,
        fir_bias,
        stride=1,
        padding=fir_length - 1,
        dilation=1,
        groups=hidden,
    )[..., :sequence]
    context = c_values * filtered
    output_input = context.transpose(-1, -2).contiguous()
    output = _linear(
        short_conv.out_proj,
        output_input,
        "conv.out_proj",
        expected_output=hidden,
        bias_allowed=declared_bias,
    )
    details = {
        "masked_input": masked_input,
        "b": b_values,
        "c": c_values,
        "x": x_values,
        "bx": bx_values,
        "filtered_bx": filtered,
        "context": context,
        "output_input": output_input,
    }
    topology = {
        "channels": hidden,
        "fir_length": fir_length,
        "fir_bias": fir_bias is not None,
        "input_projection_bias": (
            getattr(short_conv.in_proj, "bias", None) is not None
        ),
        "output_projection_bias": (
            getattr(short_conv.out_proj, "bias", None) is not None
        ),
        "padding_mask_supplied": mask is not None,
        "packed_sequence_reset_used": False,
        "recurrent_cache_used": False,
        "causal_full_prefix_truncation": True,
    }
    return output, details, topology


def _mlp_replay(
    layer: Any,
    mlp_input: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch = _torch()
    mlp = getattr(layer, "feed_forward", None)
    if mlp is None:
        raise Lfm2CpcfMacrocellReplayError(
            "Target layer lacks its feed_forward SwiGLU module."
        )
    hidden = int(mlp_input.shape[-1])
    w1 = _weight(mlp.w1, "feed_forward.w1", dimensions=2)
    w3 = _weight(mlp.w3, "feed_forward.w3", dimensions=2)
    w2 = _weight(mlp.w2, "feed_forward.w2", dimensions=2)
    if (
        int(w1.shape[1]) != hidden
        or tuple(w3.shape) != tuple(w1.shape)
        or tuple(w2.shape) != (hidden, int(w1.shape[0]))
    ):
        raise Lfm2CpcfMacrocellReplayError(
            "Target feed-forward tensors do not form an LFM2 SwiGLU."
        )
    intermediate = int(w1.shape[0])
    gate = _linear(
        mlp.w1,
        mlp_input,
        "feed_forward.w1",
        expected_output=intermediate,
        bias_allowed=False,
    )
    up = _linear(
        mlp.w3,
        mlp_input,
        "feed_forward.w3",
        expected_output=intermediate,
        bias_allowed=False,
    )
    post_swiglu = torch.nn.functional.silu(gate) * up
    output = _linear(
        mlp.w2,
        post_swiglu,
        "feed_forward.w2",
        expected_output=hidden,
        bias_allowed=False,
    )
    details = {
        "gate": gate,
        "up": up,
        "post_swiglu": post_swiglu,
    }
    topology = {
        "intermediate_size": intermediate,
        "activation": "silu",
        "gated_product": True,
    }
    return output, details, topology


def _maximum_absolute_error(left: Any, right: Any) -> float:
    torch = _torch()
    if tuple(left.shape) != tuple(right.shape):
        return math.inf
    if left.numel() == 0:
        return 0.0
    return float(
        (left.detach().to(torch.float64) - right.detach().to(torch.float64))
        .abs()
        .max()
        .item()
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfMacrocellReplay:
    """Detached target-native endpoints from one installed LFM2 layer."""

    operator_input: Any
    operator_delta: Any
    post_operator_residual: Any
    mlp_input: Any
    mlp_delta: Any
    post_layer_residual: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        tensors = (
            self.operator_input,
            self.operator_delta,
            self.post_operator_residual,
            self.mlp_input,
            self.mlp_delta,
            self.post_layer_residual,
        )
        torch = _torch()
        if not all(
            isinstance(value, torch.Tensor)
            and value.ndim == 3
            and not value.requires_grad
            and value.grad_fn is None
            for value in tensors
        ):
            raise Lfm2CpcfMacrocellReplayError(
                "Replay endpoints must be detached rank-three tensors."
            )
        shape = tuple(tensors[0].shape)
        if any(tuple(value.shape) != shape for value in tensors[1:]):
            raise Lfm2CpcfMacrocellReplayError("Replay endpoint shapes differ.")
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def replay_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def replay_lfm2_cpcf_macrocell(
    target_layer: Any,
    prefill_residual_states: Any,
    *,
    position_embeddings: tuple[Any, Any] | None = None,
    additive_causal_mask: Any | None = None,
    padding_mask: Any | None = None,
) -> Lfm2CpcfMacrocellReplay:
    """Replay one installed LFM2 layer without model/cache/training operations.

    Args:
        target_layer:
            An already-materialized and installed Transformers-compatible
            ``Lfm2DecoderLayer``.
        prefill_residual_states:
            Actual current-child residual states shaped ``[batch, sequence,
            hidden]``.  The tensor's device and floating dtype define the
            required target runtime.
        position_embeddings:
            Exact target RoPE ``(cos, sin)`` tensors.  Required only for an
            attention layer.
        additive_causal_mask:
            Exact additive full-prefix causal attention mask.  Required only
            for an attention layer.
        padding_mask:
            Optional two-dimensional LFM2 short-conv padding mask.  It is
            invalid for attention replay.
    """

    torch = _torch()
    residual = _dense_floating_tensor(
        prefill_residual_states,
        "prefill_residual_states",
        dimensions=3,
    )
    if int(residual.shape[-1]) <= 0:
        raise Lfm2CpcfMacrocellReplayError("Target residual width must be positive.")
    is_attention = getattr(target_layer, "is_attention_layer", None)
    if not isinstance(is_attention, bool):
        raise Lfm2CpcfMacrocellReplayError(
            "target_layer.is_attention_layer must be boolean."
        )
    if is_attention:
        if position_embeddings is None or additive_causal_mask is None:
            raise Lfm2CpcfMacrocellReplayError(
                "Attention replay requires target RoPE and additive mask."
            )
        if padding_mask is not None:
            raise Lfm2CpcfMacrocellReplayError(
                "padding_mask is only valid for short-conv replay."
            )
        operator_kind = "full_attention"
    else:
        if position_embeddings is not None or additive_causal_mask is not None:
            raise Lfm2CpcfMacrocellReplayError(
                "RoPE/additive attention inputs are invalid for short-conv."
            )
        operator_kind = "short_conv"

    entry_grad_enabled = bool(torch.is_grad_enabled())
    input_sha256_before = _tensor_sha256(residual)
    before = _module_snapshot(target_layer)

    with torch.no_grad():
        execution_grad_enabled = bool(torch.is_grad_enabled())
        operator_input = _rms_norm(
            target_layer.operator_norm,
            residual,
            "operator_norm",
        )
        if is_attention:
            operator_delta, operator_details, operator_topology = _attention_replay(
                target_layer,
                operator_input,
                position_embeddings=position_embeddings,
                additive_causal_mask=additive_causal_mask,
            )
        else:
            operator_delta, operator_details, operator_topology = _short_conv_replay(
                target_layer,
                operator_input,
                padding_mask=padding_mask,
            )
        post_operator = residual + operator_delta
        mlp_input = _rms_norm(
            target_layer.ffn_norm,
            post_operator,
            "ffn_norm",
        )
        mlp_delta, mlp_details, mlp_topology = _mlp_replay(
            target_layer,
            mlp_input,
        )
        post_layer = post_operator + mlp_delta

        operator_equation = residual + operator_delta
        mlp_equation = post_operator + mlp_delta
        operator_exact = bool(torch.equal(post_operator, operator_equation))
        mlp_exact = bool(torch.equal(post_layer, mlp_equation))
        endpoint_values = {
            "operator_input": operator_input,
            "operator_delta": operator_delta,
            "post_operator_residual": post_operator,
            "mlp_input": mlp_input,
            "mlp_delta": mlp_delta,
            "post_layer_residual": post_layer,
        }
        if not all(
            bool(torch.isfinite(value.detach()).all().item())
            for value in endpoint_values.values()
        ):
            raise Lfm2CpcfMacrocellReplayError(
                "Target-native replay produced a non-finite endpoint."
            )
        detached_endpoints = {
            name: value.detach().clone() for name, value in endpoint_values.items()
        }
        operator_detail_hashes = {
            name: _tensor_sha256(value)
            for name, value in sorted(operator_details.items())
        }
        mlp_detail_hashes = {
            name: _tensor_sha256(value) for name, value in sorted(mlp_details.items())
        }

    after = _module_snapshot(target_layer)
    input_sha256_after = _tensor_sha256(residual)
    if before.identities != after.identities:
        raise Lfm2CpcfMacrocellReplayError(
            "Target layer tensors, gradients, or bytes changed during replay."
        )
    if before.modes != after.modes:
        raise Lfm2CpcfMacrocellReplayError(
            "Target layer training modes changed during replay."
        )
    if before.state_sha256 != after.state_sha256:
        raise Lfm2CpcfMacrocellReplayError(
            "Target layer state digest changed during replay."
        )
    if input_sha256_before != input_sha256_after:
        raise Lfm2CpcfMacrocellReplayError(
            "Caller-supplied residual states changed during replay."
        )
    if not operator_exact or not mlp_exact:
        raise Lfm2CpcfMacrocellReplayError(
            "Replay failed an exact residual-add equation."
        )

    endpoint_hashes = {
        name: _tensor_sha256(value)
        for name, value in sorted(detached_endpoints.items())
    }
    body = {
        "schema": LFM2_CPCF_MACROCELL_REPLAY_SCHEMA,
        "operator_kind": operator_kind,
        "batch_size": int(residual.shape[0]),
        "sequence_length": int(residual.shape[1]),
        "hidden_size": int(residual.shape[2]),
        "runtime_dtype": _dtype_name(residual),
        "runtime_device": str(residual.device),
        "operator_topology": operator_topology,
        "mlp_topology": mlp_topology,
        "residual_equations": {
            "post_operator_equals_prefill_plus_operator_delta": operator_exact,
            "post_operator_maximum_absolute_error": (
                _maximum_absolute_error(post_operator, operator_equation)
            ),
            "post_layer_equals_post_operator_plus_mlp_delta": mlp_exact,
            "post_layer_maximum_absolute_error": _maximum_absolute_error(
                post_layer,
                mlp_equation,
            ),
        },
        "endpoint_sha256": endpoint_hashes,
        "operator_detail_sha256": operator_detail_hashes,
        "mlp_detail_sha256": mlp_detail_hashes,
        "input_sha256_before": input_sha256_before,
        "input_sha256_after": input_sha256_after,
        "target_state_sha256_before": before.state_sha256,
        "target_state_sha256_after": after.state_sha256,
        "target_tensor_count": before.tensor_count,
        "target_parameter_count": before.parameter_count,
        "target_state_bytes_preserved": True,
        "target_tensor_identities_preserved": True,
        "target_gradients_preserved": True,
        "target_training_modes_preserved": True,
        "returned_tensors_detached": True,
        "execution": {
            "entry_grad_enabled": entry_grad_enabled,
            "grad_enabled_inside_replay": execution_grad_enabled,
            "parent_forward_count": 0,
            "child_model_forward_count": 0,
            "target_layer_forward_count": 0,
            "cache_generation_count": 0,
            "cache_update_count": 0,
            "training_step_count": 0,
            "backward_call_count": 0,
            "optimizer_step_count": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
        "scope": "installed_blank_target_macrocell_actual_child_prefill",
    }
    report = {**body, "report_sha256": _sha(body)}
    return Lfm2CpcfMacrocellReplay(
        operator_input=detached_endpoints["operator_input"],
        operator_delta=detached_endpoints["operator_delta"],
        post_operator_residual=detached_endpoints["post_operator_residual"],
        mlp_input=detached_endpoints["mlp_input"],
        mlp_delta=detached_endpoints["mlp_delta"],
        post_layer_residual=detached_endpoints["post_layer_residual"],
        report=report,
    )


__all__ = [
    "LFM2_CPCF_MACROCELL_REPLAY_SCHEMA",
    "Lfm2CpcfMacrocellReplay",
    "Lfm2CpcfMacrocellReplayError",
    "replay_lfm2_cpcf_macrocell",
]
