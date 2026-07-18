"""Sequential target-native CPCF compilation for one materialized LFM2 layer.

This module is the bridge between sealed parent circuit evidence and the
already-created dense blank child.  It deliberately compiles one layer in two
causal stages:

1. replay the current child full prefixes and compile/install the attention or
   short-convolution operator against those *actual* child coordinates;
2. replay the installed operator, then compile/install SwiGLU against the
   resulting actual post-operator residual and normalized MLP input.

Only ``fit`` phases enter attention and MLP closed-form solvers.  The
suffix-aware short-convolution compiler receives validation cases solely for
its post-materialization evaluator; its own sealed report proves that those
cases do not enter any fit.  Selection phases never enter a layer solver.

No model forward method, generation, cache update, backward pass, optimizer,
training, pruning, quantization, or fallback path is used here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfLayerReplayBank,
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
    replay_lfm2_cpcf_phase_bank,
)
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CompiledCpcfShortConvFold,
    Lfm2CpcfShortConvFoldCase,
    Lfm2CpcfShortConvGatePolicy,
    compile_lfm2_cpcf_short_conv_fold,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
    Lfm2CompiledStructuralAttentionFold,
    Lfm2CompiledStructuralMlpFold,
    compile_lfm2_structural_residual_attention_fold,
    compile_lfm2_structural_residual_mlp_fold,
    install_lfm2_structural_attention_fold,
    install_lfm2_structural_mlp_fold,
)


LFM2_CPCF_LAYER_COMPILER_SCHEMA = "neural-forge-lfm2-cpcf-layer-compiler-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPLITS = ("fit", "validation")


class Lfm2CpcfLayerCompilerError(ValueError):
    """One target layer cannot be compiled from the supplied sealed evidence."""


class Lfm2CpcfLayerInstallError(RuntimeError):
    """A target layer installation failed and was rolled back."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for LFM2 CPCF compilation.") from exc
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
        raise Lfm2CpcfLayerCompilerError(
            "Layer-compiler metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be nonnegative.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfLayerCompilerError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be finite and nonnegative.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Only tensors can be content authenticated.")
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bundle_sha256(values: Mapping[str, Any]) -> str:
    return _sha(
        {
            name: _tensor_sha256(value)
            for name, value in sorted(values.items())
        }
    )


def _short_conv_state_sha256(values: Mapping[str, Any]) -> str:
    """Match the suffix-aware short-conv compiler's sealed state digest."""

    torch = _torch()
    inventory = {}
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        descriptor = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "layout": str(torch.strided),
        }
        digest = hashlib.sha256(_canonical_json(descriptor))
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
        inventory[name] = digest.hexdigest()
    return _sha(inventory)


def _model_core(model: Any, name: str) -> Any:
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if core is None or layers is None:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must expose an LFM2-compatible model.layers sequence."
        )
    model_type = getattr(getattr(model, "config", None), "model_type", "lfm2")
    if model_type != "lfm2":
        raise Lfm2CpcfLayerCompilerError(f"{name} must have model_type='lfm2'.")
    return core


def _layer(model: Any, index: int, name: str) -> Any:
    layers = tuple(_model_core(model, name).layers)
    if index >= len(layers):
        raise Lfm2CpcfLayerCompilerError(
            f"layer_index is outside {name}.model.layers."
        )
    return layers[index]


def _module_tensors(module: Any) -> dict[str, Any]:
    torch = _torch()
    state = getattr(module, "state_dict", None)
    if not callable(state):
        raise Lfm2CpcfLayerCompilerError(
            "Source and target layers must expose state_dict()."
        )
    values = state(keep_vars=True)
    if not isinstance(values, Mapping) or not values:
        raise Lfm2CpcfLayerCompilerError("Layer state_dict() is empty or invalid.")
    result = {
        str(name): value
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    if len(result) != len(values):
        raise Lfm2CpcfLayerCompilerError(
            "Layer state_dict() contains a non-tensor value."
        )
    return result


@dataclass(frozen=True, slots=True)
class _LayerSnapshot:
    tensors: Mapping[str, Any]
    tensor_ids: Mapping[str, int]
    tensor_shapes: Mapping[str, tuple[int, ...]]
    tensor_dtypes: Mapping[str, Any]
    tensor_bytes_sha256: str
    parameter_gradients: Mapping[str, tuple[bool, int | None, str | None]]
    modes: tuple[tuple[str, bool], ...]


def _layer_snapshot(layer: Any) -> _LayerSnapshot:
    tensors = _module_tensors(layer)
    named_parameters = dict(layer.named_parameters())
    gradients = {}
    for name, parameter in named_parameters.items():
        gradient = parameter.grad
        gradients[name] = (
            bool(parameter.requires_grad),
            None if gradient is None else id(gradient),
            None if gradient is None else _tensor_sha256(gradient),
        )
    modes = tuple(
        (name, bool(module.training))
        for name, module in layer.named_modules()
    )
    return _LayerSnapshot(
        tensors=MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in tensors.items()
            }
        ),
        tensor_ids=MappingProxyType({name: id(value) for name, value in tensors.items()}),
        tensor_shapes=MappingProxyType(
            {name: tuple(value.shape) for name, value in tensors.items()}
        ),
        tensor_dtypes=MappingProxyType(
            {name: value.dtype for name, value in tensors.items()}
        ),
        tensor_bytes_sha256=_tensor_bundle_sha256(tensors),
        parameter_gradients=MappingProxyType(gradients),
        modes=modes,
    )


def _restore_layer(layer: Any, snapshot: _LayerSnapshot) -> None:
    torch = _torch()
    current = _module_tensors(layer)
    if set(current) != set(snapshot.tensors):
        raise Lfm2CpcfLayerInstallError(
            "Cannot roll back a target layer whose state keys changed."
        )
    with torch.no_grad():
        for name, tensor in current.items():
            if (
                id(tensor) != snapshot.tensor_ids[name]
                or tuple(tensor.shape) != snapshot.tensor_shapes[name]
                or tensor.dtype != snapshot.tensor_dtypes[name]
            ):
                raise Lfm2CpcfLayerInstallError(
                    "Cannot roll back a target layer whose tensor identity, "
                    "shape, or dtype changed."
                )
            tensor.copy_(
                snapshot.tensors[name].to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
            )


def _assert_snapshot_identity(layer: Any, snapshot: _LayerSnapshot) -> None:
    current = _module_tensors(layer)
    if (
        set(current) != set(snapshot.tensor_ids)
        or any(id(current[name]) != snapshot.tensor_ids[name] for name in current)
        or any(
            tuple(current[name].shape) != snapshot.tensor_shapes[name]
            for name in current
        )
        or any(current[name].dtype != snapshot.tensor_dtypes[name] for name in current)
    ):
        raise Lfm2CpcfLayerInstallError(
            "Target layer tensor identity, shape, or dtype changed during install."
        )
    parameter_gradients = {}
    for name, parameter in layer.named_parameters():
        gradient = parameter.grad
        parameter_gradients[name] = (
            bool(parameter.requires_grad),
            None if gradient is None else id(gradient),
            None if gradient is None else _tensor_sha256(gradient),
        )
    if parameter_gradients != dict(snapshot.parameter_gradients):
        raise Lfm2CpcfLayerInstallError(
            "Target layer gradient state changed during install."
        )
    modes = tuple(
        (name, bool(module.training))
        for name, module in layer.named_modules()
    )
    if modes != snapshot.modes:
        raise Lfm2CpcfLayerInstallError(
            "Target layer training modes changed during install."
        )


def _evidence_tensors(
    phases: Sequence[Lfm2CpcfTargetPhase],
    layer_index: int,
) -> dict[str, Any]:
    result = {}
    for phase in phases:
        evidence = phase.parent_evidence.layers[layer_index]
        for name, value in evidence.tensors.items():
            result[
                f"{phase.batch_index}:{phase.phase}:layer.{layer_index}:{name}"
            ] = value
    return result


def _phase_evidence(
    phase: Lfm2CpcfTargetPhase,
    layer_index: int,
    operator_kind: str,
) -> Mapping[str, Any]:
    try:
        evidence = phase.parent_evidence.layers[layer_index]
    except IndexError as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"Phase {phase.identity!r} lacks layer {layer_index} evidence."
        ) from exc
    if evidence.layer_index != layer_index or evidence.operator_kind != operator_kind:
        raise Lfm2CpcfLayerCompilerError(
            f"Phase {phase.identity!r} has the wrong layer evidence."
        )
    return evidence.tensors


def _finite_tensor(
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
        or torch.is_complex(value)
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be a nonempty dense floating rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfLayerCompilerError(f"{name} contains NaN or infinity.")
    return result


def _weight(module: Any, name: str, *, dimensions: int) -> Any:
    return _finite_tensor(
        getattr(module, "weight", None),
        f"{name}.weight",
        dimensions=dimensions,
    )


def _variance_epsilon(module: Any, name: str) -> float:
    value = getattr(module, "variance_epsilon", None)
    epsilon = _nonnegative_float(value, f"{name}.variance_epsilon")
    if epsilon == 0.0:
        raise Lfm2CpcfLayerCompilerError(
            f"{name}.variance_epsilon must be positive."
        )
    return epsilon


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope_terms(
    model: Any,
    position_ids: Any,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    core = _model_core(model, "model")
    rotary = getattr(core, "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", None)
    if (
        not isinstance(inv_freq, torch.Tensor)
        or inv_freq.ndim != 1
        or inv_freq.numel() == 0
        or isinstance(scaling, bool)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 rotary embedding lacks static inv_freq/attention_scaling."
        )
    try:
        scale = float(scaling)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE attention scaling is invalid."
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE attention scaling must be positive and finite."
        )
    positions = position_ids.detach().to(device="cpu", dtype=torch.float32)
    frequencies = positions.unsqueeze(-1) * inv_freq.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return (
        (angles.cos() * scale).to(dtype=dtype).contiguous(),
        (angles.sin() * scale).to(dtype=dtype).contiguous(),
    )


def invert_lfm2_rope(
    rotated: Any,
    cosine: Any,
    sine: Any,
) -> Any:
    """Invert duplicated-half LFM2 RoPE, including a finite non-unit radius."""

    torch = _torch()
    value = _finite_tensor(rotated, "rotated", dimensions=4)
    cos = _finite_tensor(cosine, "cosine", dimensions=3)
    sin = _finite_tensor(sine, "sine", dimensions=3)
    if (
        int(value.shape[-1]) % 2
        or tuple(cos.shape)
        != (int(value.shape[0]), int(value.shape[2]), int(value.shape[3]))
        or tuple(sin.shape) != tuple(cos.shape)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Rotated rows and RoPE terms have incompatible shapes."
        )
    half = int(cos.shape[-1]) // 2
    if not torch.equal(cos[..., :half], cos[..., half:]) or not torch.equal(
        sin[..., :half],
        sin[..., half:],
    ):
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE terms do not have duplicated-half pair structure."
        )
    radius = cos.square() + sin.square()
    floor = torch.finfo(torch.float32).tiny
    if bool((radius.to(torch.float32) <= floor).any().item()):
        raise Lfm2CpcfLayerCompilerError("Cannot invert a zero-radius RoPE row.")
    numerator = (
        value * cos.unsqueeze(1)
        - _rotate_half(value) * sin.unsqueeze(1)
    )
    result = numerator / radius.unsqueeze(1)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfLayerCompilerError("Inverse RoPE produced a non-finite row.")
    return result.to(dtype=value.dtype).contiguous()


def _runtime_rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    torch = _torch()
    input_dtype = value.dtype
    direction = value.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon)
    return weight.to(dtype=input_dtype) * direction.to(dtype=input_dtype)


def _rmsnorm_preimage(
    normalized: Any,
    weight: Any,
    *,
    epsilon: float,
) -> Any:
    """Choose a deterministic finite-precision preimage of an RMSNorm row."""

    torch = _torch()
    values = _finite_tensor(normalized, "normalized", dimensions=4)
    norm_weight = _finite_tensor(weight, "weight", dimensions=1)
    width = int(values.shape[-1])
    if tuple(norm_weight.shape) != (width,):
        raise Lfm2CpcfLayerCompilerError(
            "RMSNorm weight width differs from normalized rows."
        )
    weight64 = norm_weight.to(torch.float64)
    if bool((weight64.abs() <= torch.finfo(torch.float64).tiny).any().item()):
        raise Lfm2CpcfLayerCompilerError(
            "Cannot recover an RMSNorm preimage through a zero learned weight."
        )
    rows = values.reshape(-1, width)
    direction = rows.to(torch.float64) / weight64.reshape(1, width)
    mean_square = direction.square().mean(dim=-1, keepdim=True)
    denominator = 1.0 - mean_square
    ideal = torch.sqrt(
        torch.as_tensor(epsilon, dtype=torch.float64)
        / denominator.clamp_min(torch.finfo(torch.float64).eps)
    )
    ideal = torch.where(
        denominator > torch.finfo(torch.float32).eps,
        ideal,
        torch.full_like(ideal, 128.0),
    ).clamp(min=2.0**-8, max=2.0**16)
    scales = torch.stack(
        (
            ideal,
            torch.ones_like(ideal),
            torch.full_like(ideal, 16.0),
            torch.full_like(ideal, 128.0),
            torch.full_like(ideal, 1024.0),
        ),
        dim=0,
    )
    candidates = (
        direction.unsqueeze(0) * scales
    ).to(dtype=values.dtype)
    predictions = _runtime_rms_norm(
        candidates,
        norm_weight.reshape(1, 1, width),
        epsilon,
    )
    targets = rows.unsqueeze(0)
    errors = (
        predictions.to(torch.float64) - targets.to(torch.float64)
    ).square().mean(dim=-1)
    choice = errors.argmin(dim=0)
    row_indices = torch.arange(int(rows.shape[0]))
    selected = candidates[choice, row_indices, :]
    return selected.reshape_as(values).contiguous()


def recover_lfm2_full_key_linear_history(
    *,
    parent_key_history: Any,
    parent_key_norm_weight: Any,
    full_cosine: Any,
    full_sine: Any,
    norm_epsilon: float,
    cached_current_key_linear: Any | None = None,
    cached_current_normalized_pre_rope: Any | None = None,
) -> Any:
    """Recover full-prefix pre-norm K rows from a sealed post-RoPE history.

    Decode traces contain exact post-RoPE K history but only current-step
    projection rows.  RoPE is inverted analytically and RMSNorm is assigned a
    deterministic finite-precision preimage.  Exact cached current rows then
    replace the reconstructed suffix.
    """

    torch = _torch()
    epsilon = _nonnegative_float(norm_epsilon, "norm_epsilon")
    if epsilon == 0.0:
        raise Lfm2CpcfLayerCompilerError("norm_epsilon must be positive.")
    history = _finite_tensor(
        parent_key_history,
        "parent_key_history",
        dimensions=4,
    )
    normalized = invert_lfm2_rope(history, full_cosine, full_sine)
    if cached_current_normalized_pre_rope is not None:
        cached_normalized = _finite_tensor(
            cached_current_normalized_pre_rope,
            "cached_current_normalized_pre_rope",
            dimensions=4,
        )
        current = int(cached_normalized.shape[2])
        if (
            tuple(cached_normalized.shape[:2]) != tuple(normalized.shape[:2])
            or int(cached_normalized.shape[3]) != int(normalized.shape[3])
            or current > int(normalized.shape[2])
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Cached current normalized K rows do not align to K history."
            )
        tolerance = (
            4.0 * torch.finfo(history.dtype).eps
            if history.dtype in {torch.float16, torch.bfloat16}
            else 2e-5
        )
        if not bool(
            torch.allclose(
                normalized[:, :, -current:, :].to(torch.float32),
                cached_normalized.to(torch.float32),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Inverse RoPE disagrees with cached current pre-RoPE K rows."
            )
    linear = _rmsnorm_preimage(
        normalized,
        parent_key_norm_weight,
        epsilon=epsilon,
    )
    if cached_current_key_linear is not None:
        cached_linear = _finite_tensor(
            cached_current_key_linear,
            "cached_current_key_linear",
            dimensions=4,
        )
        current = int(cached_linear.shape[2])
        if (
            tuple(cached_linear.shape[:2]) != tuple(linear.shape[:2])
            or int(cached_linear.shape[3]) != int(linear.shape[3])
            or current > int(linear.shape[2])
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Cached current linear K rows do not align to K history."
            )
        linear = linear.clone()
        linear[:, :, -current:, :] = cached_linear.to(dtype=linear.dtype)
    reconstructed_normalized = _runtime_rms_norm(
        linear,
        _finite_tensor(
            parent_key_norm_weight,
            "parent_key_norm_weight",
            dimensions=1,
        ).reshape(1, 1, 1, -1),
        epsilon,
    )
    reconstructed = (
        reconstructed_normalized * full_cosine.unsqueeze(1)
        + _rotate_half(reconstructed_normalized) * full_sine.unsqueeze(1)
    )
    tolerance = (
        8.0 * torch.finfo(history.dtype).eps
        if history.dtype in {torch.float16, torch.bfloat16}
        else 5e-5
    )
    if not bool(
        torch.allclose(
            reconstructed.to(torch.float32),
            history.to(torch.float32),
            rtol=tolerance,
            atol=tolerance,
        )
    ):
        maximum = float(
            (reconstructed.to(torch.float32) - history.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        raise Lfm2CpcfLayerCompilerError(
            "Recovered full K-linear history does not reproduce the sealed "
            f"post-RoPE history (maximum absolute error {maximum:.6g})."
        )
    return linear


def _same_rope(
    source_model: Any,
    target_model: Any,
    phase: Lfm2CpcfTargetPhase,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    source_cos, source_sin = _rope_terms(
        source_model,
        phase.position_ids,
        dtype=dtype,
    )
    target_cos, target_sin = _rope_terms(
        target_model,
        phase.position_ids,
        dtype=dtype,
    )
    tolerance = (
        2.0 * torch.finfo(dtype).eps
        if dtype in {torch.float16, torch.bfloat16}
        else 1e-6
    )
    if not (
        torch.allclose(source_cos, target_cos, rtol=tolerance, atol=tolerance)
        and torch.allclose(source_sin, target_sin, rtol=tolerance, atol=tolerance)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Source and target LFM2 RoPE differ; this preserved-head-dimension "
            "compiler requires an explicit rotary bridge."
        )
    return source_cos, source_sin


def _copy_attention_norms(source_attention: Any, target_attention: Any) -> Mapping[str, Any]:
    torch = _torch()
    source = {
        "q_layernorm.weight": _weight(
            source_attention.q_layernorm,
            "source_attention.q_layernorm",
            dimensions=1,
        ),
        "k_layernorm.weight": _weight(
            source_attention.k_layernorm,
            "source_attention.k_layernorm",
            dimensions=1,
        ),
    }
    target = {
        "q_layernorm.weight": getattr(
            getattr(target_attention, "q_layernorm", None),
            "weight",
            None,
        ),
        "k_layernorm.weight": getattr(
            getattr(target_attention, "k_layernorm", None),
            "weight",
            None,
        ),
    }
    if any(not isinstance(value, torch.Tensor) for value in target.values()):
        raise Lfm2CpcfLayerCompilerError(
            "Target attention lacks Q/K RMSNorm weights."
        )
    if any(
        tuple(target[name].shape) != tuple(source[name].shape)
        for name in source
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Source/target attention norm widths differ; route this layer "
            "through the attention-dimension compiler."
        )
    identities = {name: id(value) for name, value in target.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in target.items()
    }
    with torch.no_grad():
        for name, value in target.items():
            value.copy_(
                source[name].to(device=value.device, dtype=value.dtype)
            )
    body = {
        "scope": "attention_qk_norm_transport",
        "weights": {
            name: _tensor_sha256(value) for name, value in sorted(target.items())
        },
        "parameter_identities_preserved": all(
            id(target[name]) == identities[name] for name in target
        ),
        "gradient_state_preserved": all(
            (
                bool(target[name].requires_grad),
                None if target[name].grad is None else id(target[name].grad),
                None
                if target[name].grad is None
                else _tensor_sha256(target[name].grad),
            )
            == gradients[name]
            for name in target
        ),
        "source_derived": True,
        "training_steps": 0,
    }
    if not body["parameter_identities_preserved"] or not body[
        "gradient_state_preserved"
    ]:
        raise Lfm2CpcfLayerInstallError(
            "Attention Q/K norm transport changed parameter or gradient identity."
        )
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _attention_key_mask(additive_mask: Any) -> Any:
    torch = _torch()
    value = _finite_tensor(additive_mask, "attention.additive_mask", dimensions=4)
    if bool((value > 0).any().item()):
        raise Lfm2CpcfLayerCompilerError(
            "Parent additive attention mask contains a positive score."
        )
    allowed = value == 0
    if int(allowed.shape[1]) > 1:
        first = allowed[:, :1, :, :]
        if not bool(torch.equal(allowed, first.expand_as(allowed))):
            raise Lfm2CpcfLayerCompilerError(
                "Head-specific parent masks cannot be reduced to one GQA mask."
            )
    return allowed[:, 0, :, :].contiguous()


def _attention_cases(
    *,
    source_model: Any,
    target_model: Any,
    source_layer: Any,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    preinstall_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> tuple[Lfm2CachedAttentionFoldCase, ...]:
    """Assemble one complete causal attention trajectory per fit record.

    The sealed prefill contains every raw K-linear row.  Each decode phase
    contains the one newly executed K-linear suffix plus the complete
    post-RoPE K/V history.  Concatenating those exact raw rows is both more
    faithful and more stable than assigning an artificial RMSNorm preimage to
    every historical key.  It also avoids fitting the same prompt prefix once
    per decode phase.
    """

    torch = _torch()
    source_attention = getattr(source_layer, "self_attn", None)
    if source_attention is None:
        raise Lfm2CpcfLayerCompilerError(
            "Source attention layer lacks self_attn."
        )
    q_epsilon = _variance_epsilon(
        source_attention.q_layernorm,
        "source_attention.q_layernorm",
    )
    k_epsilon = _variance_epsilon(
        source_attention.k_layernorm,
        "source_attention.k_layernorm",
    )
    if not math.isclose(q_epsilon, k_epsilon, rel_tol=0.0, abs_tol=0.0):
        raise Lfm2CpcfLayerCompilerError(
            "Source Q/K normalization epsilon differs."
        )
    grouped: dict[
        tuple[int, str],
        list[Lfm2CpcfTargetPhase],
    ] = {}
    for phase in input_bank.phases:
        if phase.split == "fit":
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)

    result = []
    for (batch_index, record_id), trajectory in grouped.items():
        query_linear_parts = []
        query_parts = []
        context_parts = []
        endpoint_parts = []
        key_linear_history = None
        key_history = None
        value_history = None
        key_mask_history = None
        previous_phase: Lfm2CpcfTargetPhase | None = None
        previous_child_input = None
        previous_child_state = None
        previous_input_ids = None
        previous_attention_mask = None
        previous_position_ids = None
        final_phase: Lfm2CpcfTargetPhase | None = None
        final_child_input = None
        final_child_state = None
        final_cos = None
        final_sin = None

        for phase_index, phase in enumerate(trajectory):
            expected_phase = (
                "prefill"
                if phase_index == 0
                else f"decode.{phase_index - 1:03d}"
            )
            if phase.phase != expected_phase:
                raise Lfm2CpcfLayerCompilerError(
                    f"Attention trajectory {batch_index}:{record_id} skips "
                    f"or reorders causal phase {expected_phase!r}."
                )
            if previous_phase is not None and (
                phase.split != previous_phase.split
                or phase.full_prefix_tokens
                != (
                    previous_phase.full_prefix_tokens
                    + phase.sampled_suffix_tokens
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Attention trajectory {batch_index}:{record_id} changed "
                    "split or target prefix geometry."
                )

            tensors = _phase_evidence(
                phase,
                layer_index,
                "full_attention",
            )
            replay = preinstall_replay.replays[phase.identity]
            parent_q_linear = _finite_tensor(
                tensors["attention.q_linear"],
                f"phase {phase.identity!r} attention.q_linear",
                dimensions=4,
            )
            parent_k_linear = _finite_tensor(
                tensors["attention.k_linear"],
                f"phase {phase.identity!r} attention.k_linear",
                dimensions=4,
            )
            parent_query = _finite_tensor(
                tensors["attention.q"],
                f"phase {phase.identity!r} attention.q",
                dimensions=4,
            )
            parent_context = _finite_tensor(
                tensors["attention.context"],
                f"phase {phase.identity!r} attention.context",
                dimensions=4,
            )
            current_key = _finite_tensor(
                tensors["attention.k"],
                f"phase {phase.identity!r} attention.k",
                dimensions=4,
            )
            current_value = _finite_tensor(
                tensors["attention.v"],
                f"phase {phase.identity!r} attention.v",
                dimensions=4,
            )
            observed_key_history = _finite_tensor(
                tensors["attention.key_history"],
                f"phase {phase.identity!r} attention.key_history",
                dimensions=4,
            )
            observed_value_history = _finite_tensor(
                tensors["attention.value_history"],
                f"phase {phase.identity!r} attention.value_history",
                dimensions=4,
            )
            current = phase.sampled_suffix_tokens
            if (
                int(parent_q_linear.shape[2]) != current
                or int(parent_k_linear.shape[2]) != current
                or int(parent_query.shape[2]) != current
                or int(parent_context.shape[2]) != current
                or int(current_key.shape[2]) != current
                or int(current_value.shape[2]) != current
                or int(observed_key_history.shape[2])
                != phase.full_prefix_tokens
                or tuple(observed_value_history.shape)
                != tuple(observed_key_history.shape)
                or tuple(parent_q_linear.shape) != tuple(parent_query.shape)
                or tuple(parent_k_linear.shape) != tuple(current_key.shape)
                or tuple(current_value.shape) != tuple(current_key.shape)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} attention rows do not describe "
                    "one complete history plus one aligned terminal suffix."
                )

            full_cos, full_sin = _same_rope(
                source_model,
                target_model,
                phase,
                dtype=parent_q_linear.dtype,
            )
            cached_cos = _finite_tensor(
                tensors["attention.rope_cos"],
                f"phase {phase.identity!r} attention.rope_cos",
                dimensions=3,
            )
            cached_sin = _finite_tensor(
                tensors["attention.rope_sin"],
                f"phase {phase.identity!r} attention.rope_sin",
                dimensions=3,
            )
            if (
                tuple(cached_cos.shape)
                != (
                    phase.batch_size,
                    current,
                    int(parent_q_linear.shape[3]),
                )
                or tuple(cached_sin.shape) != tuple(cached_cos.shape)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} cached RoPE is not its exact "
                    "terminal query/key suffix."
                )
            tolerance = (
                4.0 * torch.finfo(parent_q_linear.dtype).eps
                if parent_q_linear.dtype
                in {torch.float16, torch.bfloat16}
                else 2e-5
            )
            if not (
                torch.allclose(
                    full_cos[:, -current:, :].to(torch.float32),
                    cached_cos.to(torch.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )
                and torch.allclose(
                    full_sin[:, -current:, :].to(torch.float32),
                    cached_sin.to(torch.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Static source RoPE differs from sealed parent cache "
                    f"evidence for phase {phase.identity!r}."
                )
            if not (
                torch.equal(
                    observed_key_history[:, :, -current:, :],
                    current_key,
                )
                and torch.equal(
                    observed_value_history[:, :, -current:, :],
                    current_value,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} current K/V rows are not the "
                    "exact terminal suffix of their sealed histories."
                )

            observed_mask = _attention_key_mask(
                tensors["attention.additive_mask"]
            )
            if tuple(observed_mask.shape) != (
                phase.batch_size,
                current,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} attention mask does not cover "
                    "its terminal queries over the complete key history."
                )
            child_input = _finite_tensor(
                replay.operator_input,
                f"phase {phase.identity!r} actual child operator input",
                dimensions=3,
            )
            child_state = _finite_tensor(
                input_bank.state(phase),
                f"phase {phase.identity!r} actual child pre-operator state",
                dimensions=3,
            )
            if (
                tuple(child_input.shape[:2])
                != (phase.batch_size, phase.full_prefix_tokens)
                or tuple(child_state.shape[:2])
                != (phase.batch_size, phase.full_prefix_tokens)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} actual child states do not "
                    "cover their complete target prefix."
                )

            if previous_phase is None:
                if current != phase.full_prefix_tokens:
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} must seal the complete "
                        "prefill Q/K/V trajectory, not a sampled tail."
                    )
                key_linear_history = parent_k_linear
                key_history = observed_key_history
                value_history = observed_value_history
                key_mask_history = observed_mask
            else:
                previous_tokens = previous_phase.full_prefix_tokens
                if not (
                    _floating_causal_prefix_matches(
                        previous_child_input,
                        child_input,
                    )
                    and _floating_causal_prefix_matches(
                        previous_child_state,
                        child_state,
                    )
                    and _integer_causal_prefix_matches(
                        previous_input_ids,
                        phase.input_ids,
                    )
                    and _integer_causal_prefix_matches(
                        previous_attention_mask,
                        phase.attention_mask,
                    )
                    and _integer_causal_prefix_matches(
                        previous_position_ids,
                        phase.position_ids,
                    )
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} changed an authenticated "
                        "causal child prefix."
                    )
                if (
                    key_linear_history is None
                    or key_history is None
                    or value_history is None
                    or key_mask_history is None
                    or int(key_linear_history.shape[2]) != previous_tokens
                    or not torch.equal(
                        observed_key_history[:, :, :previous_tokens, :],
                        key_history,
                    )
                    or not torch.equal(
                        observed_value_history[:, :, :previous_tokens, :],
                        value_history,
                    )
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} changed its exact sealed "
                        "parent K/V causal prefix."
                    )
                key_linear_history = torch.cat(
                    (key_linear_history, parent_k_linear),
                    dim=2,
                ).contiguous()
                key_history = observed_key_history
                value_history = observed_value_history
                padded_previous_mask = torch.nn.functional.pad(
                    key_mask_history,
                    (0, current),
                    value=False,
                )
                key_mask_history = torch.cat(
                    (padded_previous_mask, observed_mask),
                    dim=1,
                ).contiguous()

            endpoint = chart.post_operator.encode_shaped(
                tensors["residual.post_operator"]
            )
            if (
                int(endpoint.shape[1]) != current
                or int(key_linear_history.shape[2])
                != phase.full_prefix_tokens
                or tuple(key_mask_history.shape)
                != (
                    phase.batch_size,
                    phase.full_prefix_tokens,
                    phase.full_prefix_tokens,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} did not extend one contiguous "
                    "attention trajectory."
                )
            query_linear_parts.append(parent_q_linear)
            query_parts.append(parent_query)
            context_parts.append(parent_context)
            endpoint_parts.append(endpoint)
            previous_phase = phase
            previous_child_input = child_input
            previous_child_state = child_state
            previous_input_ids = phase.input_ids
            previous_attention_mask = phase.attention_mask
            previous_position_ids = phase.position_ids
            final_phase = phase
            final_child_input = child_input
            final_child_state = child_state
            final_cos = full_cos
            final_sin = full_sin

        if (
            final_phase is None
            or final_child_input is None
            or final_child_state is None
            or final_cos is None
            or final_sin is None
            or key_linear_history is None
            or key_history is None
            or value_history is None
            or key_mask_history is None
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Attention trajectory did not produce one complete causal case."
            )
        parent_query_linear = torch.cat(
            query_linear_parts,
            dim=2,
        ).contiguous()
        parent_query = torch.cat(query_parts, dim=2).contiguous()
        parent_context = torch.cat(context_parts, dim=2).contiguous()
        parent_endpoint = torch.cat(endpoint_parts, dim=1).contiguous()
        if not (
            int(parent_query_linear.shape[2])
            == int(parent_query.shape[2])
            == int(parent_context.shape[2])
            == int(parent_endpoint.shape[1])
            == final_phase.full_prefix_tokens
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Attention trajectory {batch_index}:{record_id} did not "
                "cover every causal query row exactly once."
            )
        result.append(
            Lfm2CachedAttentionFoldCase(
                target_query_input=final_child_input,
                target_key_value_input=final_child_input,
                parent_query_linear=parent_query_linear,
                parent_key_linear=key_linear_history,
                parent_value=value_history,
                parent_query=parent_query,
                parent_key=key_history,
                parent_context=parent_context,
                parent_post_attention_residual_target_space=parent_endpoint,
                target_pre_attention_residual=final_child_state,
                query_cosine=final_cos,
                query_sine=final_sin,
                key_cosine=final_cos,
                key_sine=final_sin,
                key_mask=key_mask_history,
            )
        )
    if not result:
        raise Lfm2CpcfLayerCompilerError(
            "Attention compilation requires at least one fit trajectory."
        )
    return tuple(result)


_SHORT_CONV_HISTORY_FIELDS = (
    "short_conv.b_history",
    "short_conv.c_history",
    "short_conv.x_history",
)


def _extend_short_conv_parent_histories(
    phase: Lfm2CpcfTargetPhase,
    tensors: Mapping[str, Any],
    *,
    layer_index: int,
    histories: dict[tuple[int, str], Any],
) -> dict[str, Any]:
    """Rebuild full causal B/C/x histories from cached decode suffixes.

    The strict tokenizer trace adapter maps one target row to one donor row
    without changing batch or sequence geometry.  A prefill therefore seals
    the complete prompt history, while each cached decode phase contributes
    only its newly executed terminal suffix.  The short-convolution solver
    needs their authenticated concatenation so its FIR warm state covers the
    same complete prefix replayed through the child.
    """

    torch = _torch()
    result = {}
    previous_fields = {
        field: histories.get((phase.batch_index, field))
        for field in _SHORT_CONV_HISTORY_FIELDS
    }
    for field in _SHORT_CONV_HISTORY_FIELDS:
        current = _finite_tensor(
            tensors[field],
            f"phase {phase.identity!r} layer {layer_index} {field}",
            dimensions=3,
        )
        if int(current.shape[0]) != phase.batch_size:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} {field} changed its batch boundary."
            )
        key = (phase.batch_index, field)
        if phase.phase == "prefill":
            if key in histories or int(current.shape[1]) != phase.full_prefix_tokens:
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} is not one complete "
                    "causal prefill history."
                )
            history = current
        else:
            previous = histories.get(key)
            previous_tokens = (
                phase.full_prefix_tokens - phase.sampled_suffix_tokens
            )
            if (
                previous is None
                or tuple(previous.shape[:1]) != tuple(current.shape[:1])
                or int(previous.shape[1]) != previous_tokens
                or int(previous.shape[2]) != int(current.shape[2])
                or previous.dtype != current.dtype
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} is not a causal "
                    "continuation of its authenticated parent history."
                )
            current_tokens = int(current.shape[1])
            if current_tokens == phase.sampled_suffix_tokens:
                history = torch.cat((previous, current), dim=1).contiguous()
            elif current_tokens == phase.full_prefix_tokens:
                if not torch.equal(current[:, :previous_tokens, :], previous):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} full {field} changed its "
                        "authenticated causal prefix."
                    )
                history = current
            else:
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} contains neither one "
                    "terminal suffix nor one complete causal history."
                )
        if int(history.shape[1]) != phase.full_prefix_tokens:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} {field} did not reconstruct its "
                "complete target-aligned prefix."
            )
        histories[key] = history
        result[field] = history
    if phase.phase != "prefill":
        previous_b = previous_fields["short_conv.b_history"]
        previous_x = previous_fields["short_conv.x_history"]
        if previous_b is None or previous_x is None:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} has no authenticated short-conv "
                "warm-state prefix."
            )
        pre_bx_state = _finite_tensor(
            tensors["short_conv.pre_bx_state"],
            (
                f"phase {phase.identity!r} layer {layer_index} "
                "short_conv.pre_bx_state"
            ),
            dimensions=3,
        )
        fir_length = int(pre_bx_state.shape[-1])
        previous_bx = (previous_b * previous_x).transpose(1, 2).contiguous()
        if int(previous_bx.shape[-1]) < fir_length:
            previous_bx = torch.nn.functional.pad(
                previous_bx,
                (fir_length - int(previous_bx.shape[-1]), 0),
            )
        expected_pre_state = previous_bx[..., -fir_length:].contiguous()
        if (
            tuple(pre_bx_state.shape) != tuple(expected_pre_state.shape)
            or pre_bx_state.dtype != expected_pre_state.dtype
            or not torch.equal(pre_bx_state, expected_pre_state)
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} short-conv warm state is not the "
                "exact authenticated tail of its reconstructed causal prefix."
            )
    return result


def _floating_causal_prefix_matches(previous: Any, current: Any) -> bool:
    torch = _torch()
    if (
        not isinstance(previous, torch.Tensor)
        or not isinstance(current, torch.Tensor)
        or previous.ndim != current.ndim
        or previous.ndim < 2
        or previous.dtype != current.dtype
        or tuple(previous.shape[:1]) != tuple(current.shape[:1])
        or tuple(previous.shape[2:]) != tuple(current.shape[2:])
        or int(previous.shape[1]) > int(current.shape[1])
    ):
        return False
    tolerance = 8.0 * torch.finfo(previous.dtype).eps
    return bool(
        torch.allclose(
            current[:, : int(previous.shape[1]), ...].to(torch.float64),
            previous.to(torch.float64),
            rtol=tolerance,
            atol=tolerance,
        )
    )


def _integer_causal_prefix_matches(previous: Any, current: Any) -> bool:
    torch = _torch()
    return bool(
        isinstance(previous, torch.Tensor)
        and isinstance(current, torch.Tensor)
        and previous.ndim == current.ndim
        and previous.ndim >= 2
        and previous.dtype == current.dtype
        and tuple(previous.shape[:1]) == tuple(current.shape[:1])
        and tuple(previous.shape[2:]) == tuple(current.shape[2:])
        and int(previous.shape[1]) <= int(current.shape[1])
        and torch.equal(
            current[:, : int(previous.shape[1]), ...],
            previous,
        )
    )


def _short_conv_cases(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    preinstall_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> tuple[Lfm2CpcfShortConvFoldCase, ...]:
    grouped: dict[tuple[int, str], list[Lfm2CpcfTargetPhase]] = {}
    for phase in input_bank.phases:
        if phase.split in _SPLITS:
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)

    result = []
    for (_batch_index, _record_id), trajectory in grouped.items():
        histories: dict[tuple[int, str], Any] = {}
        parent_histories: dict[str, Any] | None = None
        endpoint_parts = []
        endpoint_start: int | None = None
        endpoint_end: int | None = None
        previous_phase: Lfm2CpcfTargetPhase | None = None
        previous_child_input = None
        previous_child_state = None
        previous_input_ids = None
        previous_attention_mask = None
        previous_position_ids = None
        final_phase: Lfm2CpcfTargetPhase | None = None
        final_child_input = None
        final_child_state = None
        for phase_index, phase in enumerate(trajectory):
            expected_phase = (
                "prefill"
                if phase_index == 0
                else f"decode.{phase_index - 1:03d}"
            )
            if phase.phase != expected_phase:
                raise Lfm2CpcfLayerCompilerError(
                    f"Short-conv trajectory {phase.batch_index}:"
                    f"{phase.record_id} skips or reorders causal phase "
                    f"{expected_phase!r}."
                )
            if previous_phase is not None and (
                phase.split != previous_phase.split
                or phase.full_prefix_tokens
                != (
                    previous_phase.full_prefix_tokens
                    + phase.sampled_suffix_tokens
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Short-conv trajectory {phase.batch_index}:"
                    f"{phase.record_id} changed split or target prefix "
                    "geometry."
                )
            tensors = _phase_evidence(phase, layer_index, "short_conv")
            parent_histories = _extend_short_conv_parent_histories(
                phase,
                tensors,
                layer_index=layer_index,
                histories=histories,
            )
            replay = preinstall_replay.replays[phase.identity]
            child_input = _finite_tensor(
                replay.operator_input,
                f"phase {phase.identity!r} actual child operator input",
                dimensions=3,
            )
            child_state = _finite_tensor(
                input_bank.state(phase),
                f"phase {phase.identity!r} actual child pre-operator state",
                dimensions=3,
            )
            if (
                tuple(child_input.shape[:2])
                != (phase.batch_size, phase.full_prefix_tokens)
                or tuple(child_state.shape[:2])
                != (phase.batch_size, phase.full_prefix_tokens)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} actual child states do not "
                    "cover their complete target prefix."
                )
            if previous_phase is not None and not (
                _floating_causal_prefix_matches(
                    previous_child_input,
                    child_input,
                )
                and _floating_causal_prefix_matches(
                    previous_child_state,
                    child_state,
                )
                and _integer_causal_prefix_matches(
                    previous_input_ids,
                    phase.input_ids,
                )
                and _integer_causal_prefix_matches(
                    previous_attention_mask,
                    phase.attention_mask,
                )
                and _integer_causal_prefix_matches(
                    previous_position_ids,
                    phase.position_ids,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} changed an authenticated causal "
                    "child prefix."
                )
            endpoint = chart.post_operator.encode_shaped(
                tensors["residual.post_operator"]
            )
            current_start = (
                phase.full_prefix_tokens - phase.sampled_suffix_tokens
            )
            if (
                int(endpoint.shape[1]) != phase.sampled_suffix_tokens
                or (endpoint_end is not None and current_start != endpoint_end)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} residual endpoint is not the "
                    "next contiguous causal suffix."
                )
            if endpoint_start is None:
                endpoint_start = current_start
            endpoint_end = current_start + int(endpoint.shape[1])
            endpoint_parts.append(endpoint)
            previous_phase = phase
            previous_child_input = child_input
            previous_child_state = child_state
            previous_input_ids = phase.input_ids
            previous_attention_mask = phase.attention_mask
            previous_position_ids = phase.position_ids
            final_phase = phase
            final_child_input = child_input
            final_child_state = child_state
        if (
            final_phase is None
            or final_child_input is None
            or final_child_state is None
            or parent_histories is None
            or endpoint_start is None
            or endpoint_end != final_phase.full_prefix_tokens
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Short-conv trajectory did not produce one complete causal case."
            )
        endpoint = _torch().cat(endpoint_parts, dim=1).contiguous()
        actual_pre = final_child_state[:, endpoint_start:, :].contiguous()
        result.append(
            Lfm2CpcfShortConvFoldCase(
                case_id=(
                    f"{final_phase.batch_index}:{final_phase.record_id}:trajectory:"
                    f"layer.{layer_index}"
                ),
                split=final_phase.split,
                suffix_start=endpoint_start,
                actual_child_operator_input_history=final_child_input,
                parent_b_history=parent_histories["short_conv.b_history"],
                parent_c_history=parent_histories["short_conv.c_history"],
                parent_x_history=parent_histories["short_conv.x_history"],
                parent_post_operator_residual_target_space_suffix=endpoint,
                actual_child_pre_operator_residual_suffix=actual_pre,
                valid_prefix_mask=final_phase.attention_mask,
            )
        )
    if not any(value.split == "fit" for value in result) or not any(
        value.split == "validation" for value in result
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv compilation requires fit and validation phases."
        )
    return tuple(result)


def _mlp_cases(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    operator_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> tuple[Lfm2CachedMlpFoldCase, ...]:
    result = []
    for phase in input_bank.phases:
        if phase.split != "fit":
            continue
        tensors = _phase_evidence(phase, layer_index, chart.operator_kind)
        replay = operator_replay.replays[phase.identity]
        target_input = phase.suffix(replay.mlp_input)
        target_pre = phase.suffix(replay.post_operator_residual)
        parent_endpoint = chart.post_layer.encode_shaped(
            tensors["residual.post_layer"]
        )
        result.append(
            Lfm2CachedMlpFoldCase(
                target_input=target_input.reshape(-1, int(target_input.shape[-1])),
                parent_gate=tensors["mlp.gate"].reshape(
                    -1,
                    int(tensors["mlp.gate"].shape[-1]),
                ),
                parent_up=tensors["mlp.up"].reshape(
                    -1,
                    int(tensors["mlp.up"].shape[-1]),
                ),
                parent_post_swiglu=tensors["mlp.post_swiglu"].reshape(
                    -1,
                    int(tensors["mlp.post_swiglu"].shape[-1]),
                ),
                parent_post_mlp_residual_target_space=parent_endpoint.reshape(
                    -1,
                    int(parent_endpoint.shape[-1]),
                ),
                target_pre_mlp_residual=target_pre.reshape(
                    -1,
                    int(target_pre.shape[-1]),
                ),
            )
        )
    if not result:
        raise Lfm2CpcfLayerCompilerError(
            "SwiGLU compilation requires at least one fit phase."
        )
    return tuple(result)


def _install_short_conv(
    target_short_conv: Any,
    compiled: Lfm2CompiledCpcfShortConvFold,
) -> Mapping[str, Any]:
    torch = _torch()
    if not isinstance(compiled, Lfm2CompiledCpcfShortConvFold):
        raise TypeError("compiled must be Lfm2CompiledCpcfShortConvFold.")
    available = {
        "in_proj.weight": getattr(
            getattr(target_short_conv, "in_proj", None),
            "weight",
            None,
        ),
        "in_proj.bias": getattr(
            getattr(target_short_conv, "in_proj", None),
            "bias",
            None,
        ),
        "conv.weight": getattr(
            getattr(target_short_conv, "conv", None),
            "weight",
            None,
        ),
        "conv.bias": getattr(
            getattr(target_short_conv, "conv", None),
            "bias",
            None,
        ),
        "out_proj.weight": getattr(
            getattr(target_short_conv, "out_proj", None),
            "weight",
            None,
        ),
        "out_proj.bias": getattr(
            getattr(target_short_conv, "out_proj", None),
            "bias",
            None,
        ),
    }
    expected = compiled.state_dict()
    if set(expected) != {
        name for name, value in available.items() if isinstance(value, torch.Tensor)
    }:
        raise Lfm2CpcfLayerInstallError(
            "Target short-conv bias topology differs from compiled state."
        )
    for name, expected_value in expected.items():
        observed = available[name]
        if (
            not isinstance(observed, torch.Tensor)
            or tuple(observed.shape) != tuple(expected_value.shape)
            or observed.dtype != expected_value.dtype
        ):
            raise Lfm2CpcfLayerInstallError(
                f"Target short-conv {name} differs in shape or dtype."
            )
    identities = {name: id(available[name]) for name in expected}
    gradients = {
        name: (
            bool(available[name].requires_grad),
            None if available[name].grad is None else id(available[name].grad),
            None
            if available[name].grad is None
            else _tensor_sha256(available[name].grad),
        )
        for name in expected
    }
    backup = {
        name: available[name].detach().to(device="cpu").clone()
        for name in expected
    }
    try:
        with torch.no_grad():
            for name, expected_value in expected.items():
                available[name].copy_(
                    expected_value.to(
                        device=available[name].device,
                        dtype=available[name].dtype,
                    )
                )
        installed_sha = _short_conv_state_sha256(
            {name: available[name] for name in expected}
        )
        body = {
            "scope": "suffix_aware_cpcf_short_conv_install",
            "parameter_identities_preserved": all(
                id(available[name]) == identities[name] for name in expected
            ),
            "parameter_shapes_preserved": True,
            "gradient_state_preserved": all(
                (
                    bool(available[name].requires_grad),
                    None
                    if available[name].grad is None
                    else id(available[name].grad),
                    None
                    if available[name].grad is None
                    else _tensor_sha256(available[name].grad),
                )
                == gradients[name]
                for name in expected
            ),
            "compiled_state_sha256": compiled.report["compiled_state_sha256"],
            "installed_state_sha256": installed_sha,
            "target_logical_parameters": sum(
                int(available[name].numel()) for name in expected
            ),
            "training_steps": 0,
        }
        if (
            not body["parameter_identities_preserved"]
            or not body["gradient_state_preserved"]
            or body["compiled_state_sha256"] != installed_sha
        ):
            raise Lfm2CpcfLayerInstallError(
                "Installed short-conv state failed identity or content checks."
            )
        return MappingProxyType({**body, "report_sha256": _sha(body)})
    except Exception:
        with torch.no_grad():
            for name, backup_value in backup.items():
                available[name].copy_(
                    backup_value.to(
                        device=available[name].device,
                        dtype=available[name].dtype,
                    )
                )
        raise


def _metrics(prediction: Any, target: Any) -> Mapping[str, Any]:
    torch = _torch()
    left = prediction.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = target.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if tuple(left.shape) != tuple(right.shape) or left.numel() == 0:
        raise Lfm2CpcfLayerCompilerError(
            "Endpoint metric tensors are empty or misaligned."
        )
    difference = left - right
    rmse = torch.sqrt(torch.mean(difference.square()))
    target_scale = torch.sqrt(torch.mean(right.square())).clamp_min(
        torch.finfo(torch.float64).eps
    )
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    cosine = (
        1.0
        if float(left_norm.item()) == 0.0 and float(right_norm.item()) == 0.0
        else (
            0.0
            if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0
            else float(torch.dot(left, right).div(left_norm * right_norm).item())
        )
    )
    return MappingProxyType(
        {
            "rows": int(prediction.numel() // prediction.shape[-1]),
            "width": int(prediction.shape[-1]),
            "rmse": float(rmse.item()),
            "relative_rmse": float((rmse / target_scale).item()),
            "cosine_similarity": cosine,
            "maximum_absolute_error": float(difference.abs().max().item()),
            "prediction_sha256": _tensor_sha256(prediction),
            "target_sha256": _tensor_sha256(target),
        }
    )


def _split_endpoint_metrics(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    operator_replay: Lfm2CpcfLayerReplayBank,
    final_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> Mapping[str, Any]:
    torch = _torch()
    result = {}
    for split in _SPLITS:
        operator_predictions = []
        operator_targets = []
        layer_predictions = []
        layer_targets = []
        phase_seals = []
        for phase in input_bank.phases:
            if phase.split != split:
                continue
            tensors = _phase_evidence(phase, layer_index, chart.operator_kind)
            operator_predictions.append(
                phase.suffix(
                    operator_replay.replays[
                        phase.identity
                    ].post_operator_residual
                ).to(device="cpu")
            )
            operator_targets.append(
                chart.post_operator.encode_shaped(
                    tensors["residual.post_operator"]
                ).to(device="cpu")
            )
            layer_predictions.append(
                phase.suffix(
                    final_replay.replays[phase.identity].post_layer_residual
                ).to(device="cpu")
            )
            layer_targets.append(
                chart.post_layer.encode_shaped(
                    tensors["residual.post_layer"]
                ).to(device="cpu")
            )
            phase_seals.append(phase.phase_sha256)
        if not operator_predictions:
            raise Lfm2CpcfLayerCompilerError(
                f"Layer evaluation requires at least one {split} phase."
            )
        operator_prediction = torch.cat(
            [
                value.reshape(-1, int(value.shape[-1]))
                for value in operator_predictions
            ],
            dim=0,
        )
        operator_target = torch.cat(
            [
                value.reshape(-1, int(value.shape[-1]))
                for value in operator_targets
            ],
            dim=0,
        )
        layer_prediction = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in layer_predictions],
            dim=0,
        )
        layer_target = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in layer_targets],
            dim=0,
        )
        body = {
            "split": split,
            "phase_count": len(phase_seals),
            "phase_sha256": phase_seals,
            "post_operator": dict(
                _metrics(operator_prediction, operator_target)
            ),
            "post_layer": dict(_metrics(layer_prediction, layer_target)),
        }
        result[split] = {**body, "report_sha256": _sha(body)}
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCompiledLayerResult:
    """One installed layer and its actual-child next-boundary state bank."""

    layer_index: int
    operator_kind: str
    input_bank_sha256: str
    operator_artifact: Any
    mlp_artifact: Lfm2CompiledStructuralMlpFold
    operator_install_report: Mapping[str, Any]
    mlp_install_report: Mapping[str, Any]
    preinstall_replay: Lfm2CpcfLayerReplayBank
    operator_replay: Lfm2CpcfLayerReplayBank
    final_replay: Lfm2CpcfLayerReplayBank
    split_metrics: Mapping[str, Any]
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        index = _nonnegative_int(self.layer_index, "layer_index")
        if self.operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfLayerCompilerError("operator_kind is not canonical LFM2.")
        _digest(self.input_bank_sha256, "input_bank_sha256")
        if self.operator_kind == "full_attention":
            if not isinstance(
                self.operator_artifact,
                Lfm2CompiledStructuralAttentionFold,
            ):
                raise TypeError(
                    "Attention result has the wrong operator artifact."
                )
        elif not isinstance(
            self.operator_artifact,
            Lfm2CompiledCpcfShortConvFold,
        ):
            raise TypeError("Short-conv result has the wrong operator artifact.")
        if not isinstance(
            self.mlp_artifact,
            Lfm2CompiledStructuralMlpFold,
        ):
            raise TypeError("mlp_artifact has the wrong type.")
        if (
            self.preinstall_replay.layer_index != index
            or self.operator_replay.layer_index != index
            or self.final_replay.layer_index != index
            or self.final_replay.post_layer_bank.layer_index != index + 1
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Layer replay sequence is incomplete or out of order."
            )
        if set(self.split_metrics) != set(_SPLITS):
            raise Lfm2CpcfLayerCompilerError(
                "split_metrics must contain fit and validation."
            )
        result_sha = _digest(self.result_sha256, "result_sha256")
        report = dict(self.report)
        if report.pop("result_sha256", None) != result_sha or _sha(report) != result_sha:
            raise Lfm2CpcfLayerCompilerError(
                "Layer compiler result seal is invalid."
            )
        object.__setattr__(
            self,
            "operator_install_report",
            MappingProxyType(dict(self.operator_install_report)),
        )
        object.__setattr__(
            self,
            "mlp_install_report",
            MappingProxyType(dict(self.mlp_install_report)),
        )
        object.__setattr__(
            self,
            "split_metrics",
            MappingProxyType(dict(self.split_metrics)),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "result_sha256": result_sha}),
        )

    @property
    def next_bank(self) -> Lfm2CpcfPhaseStateBank:
        return self.final_replay.post_layer_bank

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def compile_and_install_lfm2_cpcf_layer(
    source_model: Any,
    target_model: Any,
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
    ridge: float = 1e-8,
    short_conv_gate_policy: Lfm2CpcfShortConvGatePolicy | None = None,
) -> Lfm2CpcfCompiledLayerResult:
    """Compile, install, and replay one next sequential LFM2 target layer."""

    index = _nonnegative_int(layer_index, "layer_index")
    ridge_value = _nonnegative_float(ridge, "ridge")
    if not isinstance(input_bank, Lfm2CpcfPhaseStateBank):
        raise TypeError("input_bank must be Lfm2CpcfPhaseStateBank.")
    if not isinstance(boundary_charts, Lfm2CpcfLayerBoundaryCharts):
        raise TypeError(
            "boundary_charts must be Lfm2CpcfLayerBoundaryCharts."
        )
    if input_bank.layer_index != index:
        raise Lfm2CpcfLayerCompilerError(
            "Input state bank is not the next sequential target layer."
        )
    if boundary_charts.layer_index != index:
        raise Lfm2CpcfLayerCompilerError(
            "Boundary charts do not describe the requested layer."
        )
    if input_bank.hidden_size != boundary_charts.target_width:
        raise Lfm2CpcfLayerCompilerError(
            "Actual child state width differs from boundary-chart width."
        )
    split_counts = {
        split: sum(phase.split == split for phase in input_bank.phases)
        for split in ("selection", *_SPLITS)
    }
    if split_counts["fit"] == 0 or split_counts["validation"] == 0:
        raise Lfm2CpcfLayerCompilerError(
            "Layer compilation requires nonempty fit and validation phases."
        )
    source_layer = _layer(source_model, index, "source_model")
    target_layer = _layer(target_model, index, "target_model")
    source_is_attention = bool(
        getattr(source_layer, "is_attention_layer", False)
    )
    target_is_attention = bool(
        getattr(target_layer, "is_attention_layer", False)
    )
    operator_kind = "full_attention" if source_is_attention else "short_conv"
    if (
        source_is_attention != target_is_attention
        or operator_kind != boundary_charts.operator_kind
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Source, target, and boundary chart operator kinds differ."
        )
    for phase in input_bank.phases:
        evidence = phase.parent_evidence.layers[index]
        if (
            evidence.source_width != boundary_charts.source_width
            or evidence.operator_kind != operator_kind
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} evidence differs from layer charts."
            )

    source_before = _layer_snapshot(source_layer)
    target_before = _layer_snapshot(target_layer)
    evidence_values = _evidence_tensors(input_bank.phases, index)
    evidence_before = _tensor_bundle_sha256(evidence_values)
    input_before = _tensor_bundle_sha256(
        {
            f"{phase.batch_index}:{phase.phase}": input_bank.states[phase.identity]
            for phase in input_bank.phases
        }
    )
    chart_seal_before = boundary_charts.layer_charts_sha256

    try:
        preinstall_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=index,
            input_bank=input_bank,
        )
        norm_report: Mapping[str, Any] | None = None
        if operator_kind == "full_attention":
            source_attention = source_layer.self_attn
            target_attention = target_layer.self_attn
            source_q_norm = _weight(
                source_attention.q_layernorm,
                "source_attention.q_layernorm",
                dimensions=1,
            )
            source_k_norm = _weight(
                source_attention.k_layernorm,
                "source_attention.k_layernorm",
                dimensions=1,
            )
            source_q_epsilon = _variance_epsilon(
                source_attention.q_layernorm,
                "source_attention.q_layernorm",
            )
            source_k_epsilon = _variance_epsilon(
                source_attention.k_layernorm,
                "source_attention.k_layernorm",
            )
            target_q_epsilon = _variance_epsilon(
                target_attention.q_layernorm,
                "target_attention.q_layernorm",
            )
            target_k_epsilon = _variance_epsilon(
                target_attention.k_layernorm,
                "target_attention.k_layernorm",
            )
            if not (
                source_q_epsilon
                == source_k_epsilon
                == target_q_epsilon
                == target_k_epsilon
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Source/target Q/K RMSNorm epsilon differs."
                )
            cases = _attention_cases(
                source_model=source_model,
                target_model=target_model,
                source_layer=source_layer,
                layer_index=index,
                input_bank=input_bank,
                preinstall_replay=preinstall_replay,
                chart=boundary_charts,
            )
            operator_artifact = (
                compile_lfm2_structural_residual_attention_fold(
                    source_query_weight=source_attention.q_proj.weight,
                    source_key_weight=source_attention.k_proj.weight,
                    source_value_weight=source_attention.v_proj.weight,
                    source_output_weight=source_attention.out_proj.weight,
                    target_query_weight_template=target_attention.q_proj.weight,
                    target_key_weight_template=target_attention.k_proj.weight,
                    target_value_weight_template=target_attention.v_proj.weight,
                    target_output_weight_template=target_attention.out_proj.weight,
                    cases=cases,
                    head_dim=int(source_attention.head_dim),
                    query_norm_weight=source_q_norm,
                    key_norm_weight=source_k_norm,
                    residual_assignment_sha256=(
                        boundary_charts.post_operator.report[
                            "report_sha256"
                        ]
                    ),
                    norm_epsilon=source_q_epsilon,
                    ridge=ridge_value,
                    max_prior_delta_ratio=None,
                )
            )
            norm_report = _copy_attention_norms(
                source_attention,
                target_attention,
            )
            raw_operator_install = install_lfm2_structural_attention_fold(
                target_attention,
                operator_artifact,
            ).as_dict()
            operator_install_report = MappingProxyType(
                {
                    "operator": raw_operator_install,
                    "attention_norms": dict(norm_report),
                }
            )
        else:
            source_conv = source_layer.conv
            target_conv = target_layer.conv
            cases = _short_conv_cases(
                layer_index=index,
                input_bank=input_bank,
                preinstall_replay=preinstall_replay,
                chart=boundary_charts,
            )
            policy = (
                Lfm2CpcfShortConvGatePolicy(enforce=False)
                if short_conv_gate_policy is None
                else short_conv_gate_policy
            )
            operator_artifact = compile_lfm2_cpcf_short_conv_fold(
                source_fir_weight=source_conv.conv.weight,
                source_fir_bias=source_conv.conv.bias,
                target_in_proj_weight_template=target_conv.in_proj.weight,
                target_in_proj_bias_template=target_conv.in_proj.bias,
                target_conv_weight_template=target_conv.conv.weight,
                target_conv_bias_template=target_conv.conv.bias,
                target_out_proj_weight_template=target_conv.out_proj.weight,
                target_out_proj_bias_template=target_conv.out_proj.bias,
                cases=cases,
                ridge=ridge_value,
                gate_policy=policy,
            )
            operator_install_report = _install_short_conv(
                target_conv,
                operator_artifact,
            )

        operator_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=index,
            input_bank=input_bank,
        )
        source_mlp = source_layer.feed_forward
        target_mlp = target_layer.feed_forward
        mlp_cases = _mlp_cases(
            layer_index=index,
            input_bank=input_bank,
            operator_replay=operator_replay,
            chart=boundary_charts,
        )
        mlp_artifact = compile_lfm2_structural_residual_mlp_fold(
            source_gate_weight=source_mlp.w1.weight,
            source_up_weight=source_mlp.w3.weight,
            source_down_weight=source_mlp.w2.weight,
            target_gate_weight_template=target_mlp.w1.weight,
            target_up_weight_template=target_mlp.w3.weight,
            target_down_weight_template=target_mlp.w2.weight,
            cases=mlp_cases,
            residual_assignment_sha256=(
                boundary_charts.post_layer.report["report_sha256"]
            ),
            ridge=ridge_value,
            max_prior_delta_ratio=None,
        )
        mlp_install_report = install_lfm2_structural_mlp_fold(
            target_mlp,
            mlp_artifact,
        ).as_dict()
        final_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=index,
            input_bank=input_bank,
        )
        split_metrics = _split_endpoint_metrics(
            layer_index=index,
            input_bank=input_bank,
            operator_replay=operator_replay,
            final_replay=final_replay,
            chart=boundary_charts,
        )

        source_after = _layer_snapshot(source_layer)
        evidence_after = _tensor_bundle_sha256(evidence_values)
        input_after = _tensor_bundle_sha256(
            {
                f"{phase.batch_index}:{phase.phase}": input_bank.states[
                    phase.identity
                ]
                for phase in input_bank.phases
            }
        )
        _assert_snapshot_identity(target_layer, target_before)
        if (
            source_after.tensor_bytes_sha256
            != source_before.tensor_bytes_sha256
            or source_after.tensor_ids != source_before.tensor_ids
            or source_after.parameter_gradients
            != source_before.parameter_gradients
            or source_after.modes != source_before.modes
            or evidence_after != evidence_before
            or input_after != input_before
            or boundary_charts.layer_charts_sha256 != chart_seal_before
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Source, evidence, charts, or actual input bank changed during "
                "closed-form layer compilation."
            )
        body = {
            "schema": LFM2_CPCF_LAYER_COMPILER_SCHEMA,
            "layer_index": index,
            "operator_kind": operator_kind,
            "source_width": boundary_charts.source_width,
            "target_width": boundary_charts.target_width,
            "input_bank_sha256": input_bank.state_bank_sha256,
            "next_bank_sha256": final_replay.post_layer_bank.state_bank_sha256,
            "boundary_charts_sha256": boundary_charts.layer_charts_sha256,
            "preinstall_replay_sha256": preinstall_replay.report[
                "report_sha256"
            ],
            "operator_replay_sha256": operator_replay.report[
                "report_sha256"
            ],
            "final_replay_sha256": final_replay.report["report_sha256"],
            "operator_artifact": operator_artifact.as_dict(),
            "mlp_artifact": mlp_artifact.as_dict(),
            "operator_install_report": dict(operator_install_report),
            "mlp_install_report": dict(mlp_install_report),
            "split_metrics": {
                split: dict(split_metrics[split]) for split in _SPLITS
            },
            "split_phase_counts": split_counts,
            "solver_evidence_policy": {
                "operator_case_count": len(cases),
                "selection_used_for_operator_fit": False,
                "selection_used_for_mlp_fit": False,
                "validation_used_for_attention_fit": False,
                "validation_used_for_mlp_fit": False,
                "validation_used_for_short_conv_fit": False,
                "attention_fit_case_granularity": (
                    "one_complete_causal_trajectory_per_fit_record"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_key_linear_history": (
                    "exact_prefill_plus_ordered_decode_suffixes"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_inverse_rope_preimage_used": (
                    False
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_repeated_prefix_fit_rows": (
                    False
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "short_conv_validation_role": (
                    "post_materialization_heldout_evaluator"
                    if operator_kind == "short_conv"
                    else "not_applicable"
                ),
            },
            "actual_child_state_policy": {
                "operator_fit_input": (
                    "causally_stitched_full_prefix_preinstall_target_native_replay"
                ),
                "mlp_fit_input": (
                    "suffix_of_full_prefix_installed_operator_replay"
                ),
                "next_layer_input": (
                    "complete_prefill_plus_ordered_decode_suffix_layer_replay"
                ),
                "recomputed_historical_prefix_propagated": False,
                "independent_parent_projection_substitution": False,
            },
            "audit": {
                "source_state_sha256_before": (
                    source_before.tensor_bytes_sha256
                ),
                "source_state_sha256_after": (
                    source_after.tensor_bytes_sha256
                ),
                "source_unchanged": True,
                "evidence_sha256_before": evidence_before,
                "evidence_sha256_after": evidence_after,
                "evidence_unchanged": True,
                "input_bank_sha256_before": input_before,
                "input_bank_sha256_after": input_after,
                "input_bank_unchanged": True,
                "chart_seal_unchanged": True,
                "target_tensor_identities_preserved": True,
                "target_gradient_state_preserved": True,
                "target_training_modes_preserved": True,
                "layer_transaction_rollback_available": True,
            },
            "execution": {
                "closed_form_compilers": True,
                "parent_model_forward_calls": 0,
                "target_model_forward_calls": 0,
                "target_layer_forward_calls": 0,
                "generation_calls": 0,
                "cache_generation_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "training_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "fallback_paths": 0,
            },
        }
        result_sha = _sha(body)
        return Lfm2CpcfCompiledLayerResult(
            layer_index=index,
            operator_kind=operator_kind,
            input_bank_sha256=input_bank.state_bank_sha256,
            operator_artifact=operator_artifact,
            mlp_artifact=mlp_artifact,
            operator_install_report=operator_install_report,
            mlp_install_report=mlp_install_report,
            preinstall_replay=preinstall_replay,
            operator_replay=operator_replay,
            final_replay=final_replay,
            split_metrics=split_metrics,
            report={**body, "result_sha256": result_sha},
            result_sha256=result_sha,
        )
    except Exception:
        _restore_layer(target_layer, target_before)
        raise


__all__ = [
    "LFM2_CPCF_LAYER_COMPILER_SCHEMA",
    "Lfm2CpcfCompiledLayerResult",
    "Lfm2CpcfLayerCompilerError",
    "Lfm2CpcfLayerInstallError",
    "compile_and_install_lfm2_cpcf_layer",
    "invert_lfm2_rope",
    "recover_lfm2_full_key_linear_history",
]
