"""Fold-only bridge absorption for diagnostic LFM2 MLP and short-conv modules.

The progressive component diagnostic normally evaluates a compact operator as
``D(compact(E(x)))`` inside the parent residual stream.  For the LFM2 MLP and
short convolution, both bridges are adjacent to linear projections and can be
absorbed without crossing a nonlinear operation::

    MLP:   W1* = W1 @ E.T, W3* = W3 @ E.T, W2* = D.T @ W2
    Conv:  Win* = Win @ E.T, Wout* = D.T @ Wout

The MLP is an ordinary parent-width ``Lfm2MLP`` with the compact intermediate
width.  The convolution has parent-width residual I/O and compact internal
channels, so it needs an explicitly described bottleneck module.  Its forward
path mirrors the stock Transformers LFM2 slow/fast/cache implementation.

This module is additive diagnostic infrastructure.  It performs tensor-only
closed-form products and never trains, prunes, quantizes, shares parameters,
captures traces, constructs a parent model, or accesses a network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_FUSED_BODY_INPUT_SCHEMA = "neural-forge-lfm2-fused-body-input-v1"
LFM2_FUSED_BODY_CANDIDATE_SCHEMA = "neural-forge-lfm2-fused-body-candidate-v1"
LFM2_FUSED_BODY_DESCRIPTOR_SCHEMA = (
    "neural-forge-lfm2-fused-body-reconstruction-descriptor-v1"
)
LFM2_FUSED_BODY_RECONSTRUCTION_SCHEMA = "neural-forge-lfm2-fused-body-reconstruction-v1"

Lfm2FusedBodyKind = Literal["mlp", "short_conv"]

_MLP_STATE_KEYS = ("w1.weight", "w2.weight", "w3.weight")
_SHORT_CONV_STATE_KEYS = (
    "conv.weight",
    "in_proj.weight",
    "out_proj.weight",
)
_MLP_INPUT_NAMES = (
    "compact.w1.weight",
    "compact.w2.weight",
    "compact.w3.weight",
    "bridge.input_encoder",
    "bridge.output_decoder",
)
_SHORT_CONV_INPUT_NAMES = (
    "compact.conv.weight",
    "compact.in_proj.weight",
    "compact.out_proj.weight",
    "bridge.input_encoder",
    "bridge.output_decoder",
)
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "compact_model_forward_calls": 0,
        "model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "new_parent_trace_rows": 0,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "runtime_parameter_aliases_created": 0,
        "runtime_parameter_sharing_used": False,
        "network_access": False,
    }
)


class Lfm2FusedBodyDiagnosticError(RuntimeError):
    """A fused-body input, product, descriptor, state, or runtime is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError("Fused LFM2 body diagnostics require PyTorch.") from exc
    return torch


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2FusedBodyDiagnosticError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2FusedBodyDiagnosticError(f"{label} must be one positive integer.")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2FusedBodyDiagnosticError(f"{label} must be one nonnegative integer.")
    return value


def _dtype_name(dtype: Any) -> str:
    torch = _torch()
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise Lfm2FusedBodyDiagnosticError(
        "Fused-body inputs support float32 parameters and float32/float64 bridges only."
    )


def _validated_tensor(value: Any, *, name: str, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not value.dtype.is_floating_point
        or bool(getattr(value, "is_quantized", False))
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2FusedBodyDiagnosticError(
            f"{name} must be one finite dense detached floating rank-{ndim} tensor."
        )
    return value.detach()


def _plain_tensor(value: Any, *, name: str, ndim: int, clone: bool) -> Any:
    tensor = _validated_tensor(value, name=name, ndim=ndim)
    result = tensor.to(device="cpu").contiguous()
    return result.clone() if clone else result


def _assert_no_storage_aliases(
    tensors: Mapping[str, Any],
    *,
    label: str,
) -> None:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, tensor in tensors.items():
        identity = (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(name)
    aliases = [sorted(names) for names in groups.values() if len(names) > 1]
    if aliases:
        raise Lfm2FusedBodyDiagnosticError(
            f"{label} contains shared or aliased tensor storage: {aliases!r}."
        )


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(size) for size in tensor.shape],
        "dtype": _dtype_name(tensor.dtype),
        "sha256": lfm2_authenticated_tensor_sha256(tensor),
    }


def _state_keys(kind: Lfm2FusedBodyKind) -> tuple[str, ...]:
    if kind == "mlp":
        return _MLP_STATE_KEYS
    if kind == "short_conv":
        return _SHORT_CONV_STATE_KEYS
    raise Lfm2FusedBodyDiagnosticError(f"Unsupported fused-body kind: {kind!r}.")


def _validated_topology(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2FusedBodyDiagnosticError("Fused-body topology must be one mapping.")
    topology = dict(value)
    kind = topology.get("kind")
    if kind == "mlp":
        expected = {
            "kind",
            "parent_hidden_size",
            "parent_intermediate_size",
            "compact_hidden_size",
            "compact_intermediate_size",
            "parameter_dtype",
            "bias",
            "block_auto_adjust_ff_dim",
        }
        if set(topology) != expected:
            raise Lfm2FusedBodyDiagnosticError(
                "Fused MLP topology fields are not canonical."
            )
        for name in (
            "parent_hidden_size",
            "parent_intermediate_size",
            "compact_hidden_size",
            "compact_intermediate_size",
        ):
            _positive_int(topology[name], label=f"topology {name}")
        if (
            topology["compact_hidden_size"] >= topology["parent_hidden_size"]
            or topology["compact_intermediate_size"]
            >= topology["parent_intermediate_size"]
            or topology["parameter_dtype"] != "float32"
            or topology["bias"] is not False
            or topology["block_auto_adjust_ff_dim"] is not False
        ):
            raise Lfm2FusedBodyDiagnosticError(
                "Fused MLP topology is not reduced, FP32, fixed-width, and bias-free."
            )
    elif kind == "short_conv":
        expected = {
            "kind",
            "layer_index",
            "parent_hidden_size",
            "compact_hidden_size",
            "kernel_size",
            "parameter_dtype",
            "conv_bias",
            "layer_type",
        }
        if set(topology) != expected:
            raise Lfm2FusedBodyDiagnosticError(
                "Fused short-conv topology fields are not canonical."
            )
        _nonnegative_int(topology["layer_index"], label="topology layer_index")
        for name in ("parent_hidden_size", "compact_hidden_size", "kernel_size"):
            _positive_int(topology[name], label=f"topology {name}")
        if (
            topology["compact_hidden_size"] >= topology["parent_hidden_size"]
            or topology["parameter_dtype"] != "float32"
            or topology["conv_bias"] is not False
            or topology["layer_type"] != "conv"
        ):
            raise Lfm2FusedBodyDiagnosticError(
                "Fused short-conv topology is not reduced, FP32, bias-free conv."
            )
    else:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body topology names an unsupported component kind."
        )
    return MappingProxyType(topology)


def _state_sha256(kind: Lfm2FusedBodyKind, state: Mapping[str, Any]) -> str:
    keys = _state_keys(kind)
    if tuple(sorted(state)) != tuple(sorted(keys)):
        raise Lfm2FusedBodyDiagnosticError(
            f"{kind} state does not contain its exact canonical tensor namespace."
        )
    return canonical_sha256(
        [_tensor_record(name, state[name]) for name in sorted(state)]
    )


def _stored_tensor_elements(tensors: Mapping[str, Any]) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    total = 0
    for tensor in tensors.values():
        identity = (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        if identity in seen:
            continue
        seen.add(identity)
        size = int(tensor.untyped_storage().nbytes())
        element_size = int(tensor.element_size())
        if size % element_size:
            raise Lfm2FusedBodyDiagnosticError(
                "Tensor storage byte count is not divisible by its element size."
            )
        total += size // element_size
    return total


def _module_alias_group_count(module: Any) -> int:
    try:
        parameters = module.named_parameters(remove_duplicate=False)
    except TypeError as exc:  # pragma: no cover - requires supported PyTorch
        raise Lfm2FusedBodyDiagnosticError(
            "Alias authentication requires named_parameters(remove_duplicate=False)."
        ) from exc
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, parameter in parameters:
        identity = (
            str(parameter.device),
            str(parameter.dtype),
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(str(name))
    return sum(len(names) > 1 for names in groups.values())


def _runtime_bridge_cast(tensor: Any, *, name: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    source = _plain_tensor(tensor, name=name, ndim=2, clone=True)
    if source.dtype not in {torch.float32, torch.float64}:
        raise Lfm2FusedBodyDiagnosticError(
            f"{name} must be float32 or float64 before its authenticated runtime cast."
        )
    runtime = source.to(dtype=torch.float32).contiguous().clone()
    body = {
        "source": _tensor_record(name, source),
        "runtime_fp32": _tensor_record(f"{name}.runtime_fp32", runtime),
        "cast_performed": source.dtype != torch.float32,
        "cast_device": "cpu",
        "cast_rounding_authority": "torch-float32-materialization",
        "runtime_values_are_product_authority": True,
    }
    return runtime, {**body, "lineage_sha256": canonical_sha256(body)}


def _fp64_product_from_fp32(
    left: Any,
    right: Any,
    *,
    product_name: str,
    formula: str,
    source_names: tuple[str, str],
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise Lfm2FusedBodyDiagnosticError(
            f"{product_name} operands must be the authenticated runtime FP32 values."
        )
    fp64 = (
        left.to(device="cpu", dtype=torch.float64)
        @ right.to(device="cpu", dtype=torch.float64)
    ).contiguous()
    if not bool(torch.isfinite(fp64).all().item()):
        raise Lfm2FusedBodyDiagnosticError(
            f"{product_name} produced non-finite FP64 values."
        )
    product = fp64.to(dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(product).all().item()):
        raise Lfm2FusedBodyDiagnosticError(
            f"{product_name} overflowed during FP32 materialization."
        )
    body = {
        "product_name": product_name,
        "formula": formula,
        "source_names": list(source_names),
        "left_runtime_fp32_sha256": lfm2_authenticated_tensor_sha256(left),
        "right_runtime_fp32_sha256": lfm2_authenticated_tensor_sha256(right),
        "fp64_product": _tensor_record(f"{product_name}.fp64", fp64),
        "materialized_fp32_product": _tensor_record(product_name, product),
        "solve_device": "cpu",
        "solve_dtype": "float64",
        "materialization_dtype": "float32",
    }
    return product, {**body, "lineage_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2FusedMlpDiagnosticInput:
    """Authenticated compact MLP weights and residual bridge factors."""

    parent_hidden_size: int
    parent_intermediate_size: int
    compact_hidden_size: int
    compact_intermediate_size: int
    compact_w1_weight: Any = field(repr=False, compare=False)
    compact_w2_weight: Any = field(repr=False, compare=False)
    compact_w3_weight: Any = field(repr=False, compare=False)
    input_encoder: Any = field(repr=False, compare=False)
    output_decoder: Any = field(repr=False, compare=False)
    compact_w1_bias: Any | None = field(default=None, repr=False, compare=False)
    compact_w2_bias: Any | None = field(default=None, repr=False, compare=False)
    compact_w3_bias: Any | None = field(default=None, repr=False, compare=False)

    def named_tensors(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "compact.w1.weight": self.compact_w1_weight,
                "compact.w2.weight": self.compact_w2_weight,
                "compact.w3.weight": self.compact_w3_weight,
                "bridge.input_encoder": self.input_encoder,
                "bridge.output_decoder": self.output_decoder,
            }
        )


@dataclass(frozen=True, slots=True)
class Lfm2FusedShortConvDiagnosticInput:
    """Authenticated compact short-conv weights and residual bridge factors."""

    layer_index: int
    parent_hidden_size: int
    compact_hidden_size: int
    kernel_size: int
    compact_conv_weight: Any = field(repr=False, compare=False)
    compact_in_proj_weight: Any = field(repr=False, compare=False)
    compact_out_proj_weight: Any = field(repr=False, compare=False)
    input_encoder: Any = field(repr=False, compare=False)
    output_decoder: Any = field(repr=False, compare=False)
    compact_conv_bias: Any | None = field(default=None, repr=False, compare=False)
    compact_in_proj_bias: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    compact_out_proj_bias: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def named_tensors(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "compact.conv.weight": self.compact_conv_weight,
                "compact.in_proj.weight": self.compact_in_proj_weight,
                "compact.out_proj.weight": self.compact_out_proj_weight,
                "bridge.input_encoder": self.input_encoder,
                "bridge.output_decoder": self.output_decoder,
            }
        )


@dataclass(frozen=True, slots=True)
class Lfm2FusedBodyDiagnosticCandidate:
    """One parent-I/O fused body component with complete lineage receipts."""

    kind: Lfm2FusedBodyKind
    tensors: Mapping[str, Any] = field(repr=False, compare=False)
    receipt: Mapping[str, Any]
    reconstruction_descriptor: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                name: tensor.detach().to(device="cpu").contiguous().clone()
                for name, tensor in self.tensors.items()
            }
        )


@dataclass(frozen=True, slots=True)
class Lfm2FusedBodyReconstruction:
    """A freshly reconstructed fused module and its authentication receipt."""

    module: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


def _normalized_mlp_input(
    source: Lfm2FusedMlpDiagnosticInput,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch = _torch()
    if not isinstance(source, Lfm2FusedMlpDiagnosticInput):
        raise TypeError("source must be Lfm2FusedMlpDiagnosticInput.")
    for name in (
        "parent_hidden_size",
        "parent_intermediate_size",
        "compact_hidden_size",
        "compact_intermediate_size",
    ):
        _positive_int(getattr(source, name), label=name)
    if any(
        bias is not None
        for bias in (
            source.compact_w1_bias,
            source.compact_w2_bias,
            source.compact_w3_bias,
        )
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "LFM2 fused MLP diagnostics require the stock bias-free topology."
        )
    parent_hidden = source.parent_hidden_size
    parent_intermediate = source.parent_intermediate_size
    compact_hidden = source.compact_hidden_size
    compact_intermediate = source.compact_intermediate_size
    if compact_hidden >= parent_hidden or compact_intermediate >= parent_intermediate:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused MLP geometry must reduce both residual and intermediate widths."
        )
    raw = dict(source.named_tensors())
    if tuple(raw) != _MLP_INPUT_NAMES:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused MLP input tensor namespace is not canonical."
        )
    raw_checked = {
        name: _validated_tensor(tensor, name=name, ndim=2)
        for name, tensor in raw.items()
    }
    _assert_no_storage_aliases(raw_checked, label="fused MLP inputs")
    checked = {
        name: tensor.to(device="cpu").contiguous().clone()
        for name, tensor in raw_checked.items()
    }
    expected_shapes = {
        "compact.w1.weight": (compact_intermediate, compact_hidden),
        "compact.w2.weight": (compact_hidden, compact_intermediate),
        "compact.w3.weight": (compact_intermediate, compact_hidden),
        "bridge.input_encoder": (parent_hidden, compact_hidden),
        "bridge.output_decoder": (compact_hidden, parent_hidden),
    }
    if any(
        tuple(checked[name].shape) != shape for name, shape in expected_shapes.items()
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused MLP source tensors do not match the declared geometry."
        )
    if any(
        checked[name].dtype != torch.float32
        for name in (
            "compact.w1.weight",
            "compact.w2.weight",
            "compact.w3.weight",
        )
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Compact MLP parameters must be actual runtime FP32 tensors."
        )
    encoder, encoder_cast = _runtime_bridge_cast(
        checked["bridge.input_encoder"],
        name="bridge.input_encoder",
    )
    decoder, decoder_cast = _runtime_bridge_cast(
        checked["bridge.output_decoder"],
        name="bridge.output_decoder",
    )
    topology = {
        "kind": "mlp",
        "parent_hidden_size": parent_hidden,
        "parent_intermediate_size": parent_intermediate,
        "compact_hidden_size": compact_hidden,
        "compact_intermediate_size": compact_intermediate,
        "parameter_dtype": "float32",
        "bias": False,
        "block_auto_adjust_ff_dim": False,
    }
    return (
        checked,
        topology,
        {
            "encoder": encoder,
            "decoder": decoder,
            "encoder_cast": encoder_cast,
            "decoder_cast": decoder_cast,
        },
    )


def _normalized_short_conv_input(
    source: Lfm2FusedShortConvDiagnosticInput,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch = _torch()
    if not isinstance(source, Lfm2FusedShortConvDiagnosticInput):
        raise TypeError("source must be Lfm2FusedShortConvDiagnosticInput.")
    _nonnegative_int(source.layer_index, label="layer_index")
    for name in ("parent_hidden_size", "compact_hidden_size", "kernel_size"):
        _positive_int(getattr(source, name), label=name)
    if any(
        bias is not None
        for bias in (
            source.compact_conv_bias,
            source.compact_in_proj_bias,
            source.compact_out_proj_bias,
        )
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "LFM2 fused short-conv diagnostics require conv_bias=False everywhere."
        )
    parent_hidden = source.parent_hidden_size
    compact_hidden = source.compact_hidden_size
    kernel_size = source.kernel_size
    if compact_hidden >= parent_hidden:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused short-conv geometry must reduce its internal channel width."
        )
    raw = dict(source.named_tensors())
    if tuple(raw) != _SHORT_CONV_INPUT_NAMES:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused short-conv input tensor namespace is not canonical."
        )
    raw_checked = {
        name: _validated_tensor(
            tensor,
            name=name,
            ndim=3 if name.endswith("conv.weight") else 2,
        )
        for name, tensor in raw.items()
    }
    _assert_no_storage_aliases(raw_checked, label="fused short-conv inputs")
    checked = {
        name: tensor.to(device="cpu").contiguous().clone()
        for name, tensor in raw_checked.items()
    }
    expected_shapes = {
        "compact.conv.weight": (compact_hidden, 1, kernel_size),
        "compact.in_proj.weight": (3 * compact_hidden, compact_hidden),
        "compact.out_proj.weight": (compact_hidden, compact_hidden),
        "bridge.input_encoder": (parent_hidden, compact_hidden),
        "bridge.output_decoder": (compact_hidden, parent_hidden),
    }
    if any(
        tuple(checked[name].shape) != shape for name, shape in expected_shapes.items()
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused short-conv source tensors do not match the declared geometry."
        )
    if any(
        checked[name].dtype != torch.float32
        for name in (
            "compact.conv.weight",
            "compact.in_proj.weight",
            "compact.out_proj.weight",
        )
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Compact short-conv parameters must be actual runtime FP32 tensors."
        )
    encoder, encoder_cast = _runtime_bridge_cast(
        checked["bridge.input_encoder"],
        name="bridge.input_encoder",
    )
    decoder, decoder_cast = _runtime_bridge_cast(
        checked["bridge.output_decoder"],
        name="bridge.output_decoder",
    )
    topology = {
        "kind": "short_conv",
        "layer_index": source.layer_index,
        "parent_hidden_size": parent_hidden,
        "compact_hidden_size": compact_hidden,
        "kernel_size": kernel_size,
        "parameter_dtype": "float32",
        "conv_bias": False,
        "layer_type": "conv",
    }
    return (
        checked,
        topology,
        {
            "encoder": encoder,
            "decoder": decoder,
            "encoder_cast": encoder_cast,
            "decoder_cast": decoder_cast,
        },
    )


def _normalized_input(
    source: Lfm2FusedMlpDiagnosticInput | Lfm2FusedShortConvDiagnosticInput,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(source, Lfm2FusedMlpDiagnosticInput):
        return _normalized_mlp_input(source)
    if isinstance(source, Lfm2FusedShortConvDiagnosticInput):
        return _normalized_short_conv_input(source)
    raise TypeError(
        "source must be Lfm2FusedMlpDiagnosticInput or "
        "Lfm2FusedShortConvDiagnosticInput."
    )


def fingerprint_lfm2_fused_body_diagnostic_input(
    source: Lfm2FusedMlpDiagnosticInput | Lfm2FusedShortConvDiagnosticInput,
) -> Mapping[str, Any]:
    """Fingerprint source tensors and the exact FP32 bridge cast lineage."""

    tensors, topology, bridges = _normalized_input(source)
    body = {
        "schema": LFM2_FUSED_BODY_INPUT_SCHEMA,
        "status": "fused-body-inputs-authenticated",
        "kind": topology["kind"],
        "topology": topology,
        "tensor_records": {
            name: _tensor_record(name, tensor) for name, tensor in tensors.items()
        },
        "runtime_bridge_cast_lineage": {
            "input_encoder": bridges["encoder_cast"],
            "output_decoder": bridges["decoder_cast"],
        },
        "input_storage_alias_group_count": 0,
        "input_parameters_quantized": 0,
        "input_parameters_sparse": 0,
    }
    return MappingProxyType({**body, "input_bundle_sha256": canonical_sha256(body)})


def fingerprint_lfm2_fused_mlp_diagnostic_input(
    source: Lfm2FusedMlpDiagnosticInput,
) -> Mapping[str, Any]:
    return fingerprint_lfm2_fused_body_diagnostic_input(source)


def fingerprint_lfm2_fused_short_conv_diagnostic_input(
    source: Lfm2FusedShortConvDiagnosticInput,
) -> Mapping[str, Any]:
    return fingerprint_lfm2_fused_body_diagnostic_input(source)


def _authenticated_input(
    source: Lfm2FusedMlpDiagnosticInput | Lfm2FusedShortConvDiagnosticInput,
    *,
    expected_input_bundle_sha256: str,
    expected_tensor_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    expected_bundle = _digest(
        expected_input_bundle_sha256,
        label="expected input bundle",
    )
    if not isinstance(expected_tensor_sha256, Mapping):
        raise Lfm2FusedBodyDiagnosticError(
            "expected_tensor_sha256 must be one complete mapping."
        )
    tensors, topology, bridges = _normalized_input(source)
    if tuple(sorted(expected_tensor_sha256)) != tuple(sorted(tensors)):
        raise Lfm2FusedBodyDiagnosticError(
            "Expected tensor pins do not cover the exact fused-body input namespace."
        )
    expected_pins = {
        name: _digest(value, label=f"expected {name}")
        for name, value in expected_tensor_sha256.items()
    }
    authentication = fingerprint_lfm2_fused_body_diagnostic_input(source)
    observed_pins = {
        name: record["sha256"]
        for name, record in authentication["tensor_records"].items()
    }
    if (
        authentication["input_bundle_sha256"] != expected_bundle
        or observed_pins != expected_pins
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body inputs differ from their independent pins."
        )
    return tensors, topology, bridges, authentication


def lfm2_fused_mlp_parameter_accounting(
    *,
    parent_hidden_size: int,
    parent_intermediate_size: int,
    compact_hidden_size: int,
    compact_intermediate_size: int,
) -> Mapping[str, Any]:
    """Return exact bias-free physical counts for one MLP replacement."""

    parent_hidden = _positive_int(parent_hidden_size, label="parent_hidden_size")
    parent_intermediate = _positive_int(
        parent_intermediate_size,
        label="parent_intermediate_size",
    )
    compact_hidden = _positive_int(compact_hidden_size, label="compact_hidden_size")
    compact_intermediate = _positive_int(
        compact_intermediate_size,
        label="compact_intermediate_size",
    )
    parent = 3 * parent_hidden * parent_intermediate
    compact = 3 * compact_hidden * compact_intermediate
    bridges = 2 * parent_hidden * compact_hidden
    wrapper = compact + bridges
    fused = 3 * parent_hidden * compact_intermediate
    body = {
        "original_parent_component_parameter_elements": parent,
        "compact_operator_parameter_elements": compact,
        "current_bridged_wrapper_parameter_elements": wrapper,
        "current_input_bridge_parameter_elements": parent_hidden * compact_hidden,
        "current_output_bridge_parameter_elements": parent_hidden * compact_hidden,
        "fused_candidate_parameter_elements": fused,
        "fused_candidate_stored_tensor_elements": fused,
        "fused_candidate_parameter_bytes_fp32": 4 * fused,
        "physical_parameter_reduction_from_parent": parent - fused,
        "physical_parameter_reduction_from_current_wrapper": wrapper - fused,
        "runtime_input_bridge_parameter_elements": 0,
        "runtime_output_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
        "runtime_shared_parameter_elements": 0,
        "runtime_parameter_alias_group_count": 0,
        "parameter_reduction_attributed_to_sharing": False,
    }
    if parent - fused <= 0 or wrapper - fused <= 0:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused MLP is not physically smaller than both comparison components."
        )
    return MappingProxyType(body)


def lfm2_fused_short_conv_parameter_accounting(
    *,
    parent_hidden_size: int,
    compact_hidden_size: int,
    kernel_size: int,
) -> Mapping[str, Any]:
    """Return exact bias-free physical counts for one bottleneck short conv."""

    parent_hidden = _positive_int(parent_hidden_size, label="parent_hidden_size")
    compact_hidden = _positive_int(compact_hidden_size, label="compact_hidden_size")
    kernel = _positive_int(kernel_size, label="kernel_size")
    parent = 4 * parent_hidden * parent_hidden + parent_hidden * kernel
    compact = 4 * compact_hidden * compact_hidden + compact_hidden * kernel
    bridges = 2 * parent_hidden * compact_hidden
    wrapper = compact + bridges
    fused = 4 * parent_hidden * compact_hidden + compact_hidden * kernel
    body = {
        "original_parent_component_parameter_elements": parent,
        "compact_operator_parameter_elements": compact,
        "current_bridged_wrapper_parameter_elements": wrapper,
        "current_input_bridge_parameter_elements": parent_hidden * compact_hidden,
        "current_output_bridge_parameter_elements": parent_hidden * compact_hidden,
        "fused_candidate_parameter_elements": fused,
        "fused_candidate_stored_tensor_elements": fused,
        "fused_candidate_parameter_bytes_fp32": 4 * fused,
        "physical_parameter_reduction_from_parent": parent - fused,
        "physical_parameter_reduction_from_current_wrapper": wrapper - fused,
        "runtime_input_bridge_parameter_elements": 0,
        "runtime_output_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
        "runtime_shared_parameter_elements": 0,
        "runtime_parameter_alias_group_count": 0,
        "parameter_reduction_attributed_to_sharing": False,
    }
    if parent - fused <= 0 or wrapper - fused <= 0:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused short conv is not physically smaller than both comparison components."
        )
    return MappingProxyType(body)


def _physical_accounting(topology: Mapping[str, Any]) -> Mapping[str, Any]:
    topology = _validated_topology(topology)
    if topology["kind"] == "mlp":
        return lfm2_fused_mlp_parameter_accounting(
            parent_hidden_size=topology["parent_hidden_size"],
            parent_intermediate_size=topology["parent_intermediate_size"],
            compact_hidden_size=topology["compact_hidden_size"],
            compact_intermediate_size=topology["compact_intermediate_size"],
        )
    return lfm2_fused_short_conv_parameter_accounting(
        parent_hidden_size=topology["parent_hidden_size"],
        compact_hidden_size=topology["compact_hidden_size"],
        kernel_size=topology["kernel_size"],
    )


def _build_candidate(
    *,
    kind: Lfm2FusedBodyKind,
    state: Mapping[str, Any],
    topology: Mapping[str, Any],
    authentication: Mapping[str, Any],
    product_lineage: Mapping[str, Any],
) -> Lfm2FusedBodyDiagnosticCandidate:
    torch = _torch()
    topology = _validated_topology(topology)
    keys = _state_keys(kind)
    if tuple(sorted(state)) != tuple(sorted(keys)):
        raise Lfm2FusedBodyDiagnosticError(
            "Candidate builder received an incomplete fused-body state."
        )
    normalized = {
        name: _plain_tensor(
            state[name],
            name=name,
            ndim=3 if name == "conv.weight" else 2,
            clone=True,
        )
        for name in keys
    }
    if any(tensor.dtype != torch.float32 for tensor in normalized.values()):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body candidate tensors must materialize as FP32."
        )
    _assert_no_storage_aliases(normalized, label=f"fused {kind} candidate")
    state_sha = _state_sha256(kind, normalized)
    physical = dict(_physical_accounting(topology))
    observed_elements = sum(int(tensor.numel()) for tensor in normalized.values())
    stored_elements = _stored_tensor_elements(normalized)
    if (
        observed_elements != physical["fused_candidate_parameter_elements"]
        or stored_elements != physical["fused_candidate_stored_tensor_elements"]
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body state storage does not match its physical accounting."
        )
    stock_module = kind == "mlp"
    descriptor_body = {
        "schema": LFM2_FUSED_BODY_DESCRIPTOR_SCHEMA,
        "status": "production-safe-fold-only-fused-body-descriptor",
        "family_adapter": "lfm2-concrete-runtime-fused-body-v1",
        "kind": kind,
        "module_class": (
            "transformers.models.lfm2.modeling_lfm2.Lfm2MLP"
            if stock_module
            else "neural_forge.hf_lfm2_fused_body_diagnostic.Lfm2BottleneckShortConv"
        ),
        "ordinary_stock_lfm2_module": stock_module,
        "explicit_mixed_width_bottleneck_module": not stock_module,
        "topology": dict(topology),
        "state_keys": list(keys),
        "candidate_state_sha256": state_sha,
        "parameter_dtype": "float32",
        "input_bridge_module_present": False,
        "output_bridge_module_present": False,
        "bridge_factors_retained_as_runtime_parameters": False,
        "runtime_sidecar_parameter_elements": 0,
        "runtime_parameter_alias_group_count": 0,
        "runtime_parameter_sharing_used": False,
        "cache_contract": (
            None
            if kind == "mlp"
            else {
                "cache_kind": "transformers-DynamicCache-only",
                "static_or_compileable_cache_supported": False,
                "per_layer_internal_channel_width": topology["compact_hidden_size"],
                "kernel_width": topology["kernel_size"],
                "fresh_cache_required_per_candidate": True,
            }
        ),
    }
    descriptor = {
        **descriptor_body,
        "descriptor_sha256": canonical_sha256(descriptor_body),
    }
    receipt_body = {
        "schema": LFM2_FUSED_BODY_CANDIDATE_SCHEMA,
        "status": "NON-FINAL-FOLD-ONLY-FUSED-BODY-DIAGNOSTIC",
        "kind": kind,
        "topology": dict(topology),
        "input_bundle_sha256": authentication["input_bundle_sha256"],
        "source_tensor_sha256": {
            name: record["sha256"]
            for name, record in authentication["tensor_records"].items()
        },
        "runtime_bridge_cast_lineage": authentication["runtime_bridge_cast_lineage"],
        "product_lineage": dict(product_lineage),
        "candidate_state_sha256": state_sha,
        "reconstruction_descriptor_sha256": descriptor["descriptor_sha256"],
        "physical_parameter_accounting": physical,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        "interpretation": {
            "parent_residual_input_output_width_retained": True,
            "compact_internal_width_retained": True,
            "bridge_modules_eliminated_by_closed_form_absorption": True,
            "numeric_metrics_have_semantic_decision_authority": False,
            "whole_decoded_behavior_requires_manual_review": True,
            "standalone_exact4x_child_claimed": False,
            "does_not_modify_main_exact4x_compiler": True,
        },
    }
    receipt = {
        **receipt_body,
        "result_sha256": canonical_sha256(receipt_body),
    }
    return Lfm2FusedBodyDiagnosticCandidate(
        kind=kind,
        tensors=MappingProxyType(normalized),
        receipt=MappingProxyType(receipt),
        reconstruction_descriptor=MappingProxyType(descriptor),
    )


def compile_lfm2_fused_mlp_diagnostic(
    source: Lfm2FusedMlpDiagnosticInput,
    *,
    expected_input_bundle_sha256: str,
    expected_tensor_sha256: Mapping[str, str],
) -> Lfm2FusedBodyDiagnosticCandidate:
    """Absorb E/D into one compact SwiGLU MLP without a model forward."""

    tensors, topology, bridges, authentication = _authenticated_input(
        source,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_tensor_sha256=expected_tensor_sha256,
    )
    encoder = bridges["encoder"]
    decoder = bridges["decoder"]
    w1, w1_lineage = _fp64_product_from_fp32(
        tensors["compact.w1.weight"],
        encoder.T,
        product_name="w1.weight",
        formula="compact.w1.weight @ runtime_fp32(input_encoder).T",
        source_names=("compact.w1.weight", "bridge.input_encoder.runtime_fp32"),
    )
    w3, w3_lineage = _fp64_product_from_fp32(
        tensors["compact.w3.weight"],
        encoder.T,
        product_name="w3.weight",
        formula="compact.w3.weight @ runtime_fp32(input_encoder).T",
        source_names=("compact.w3.weight", "bridge.input_encoder.runtime_fp32"),
    )
    w2, w2_lineage = _fp64_product_from_fp32(
        decoder.T,
        tensors["compact.w2.weight"],
        product_name="w2.weight",
        formula="runtime_fp32(output_decoder).T @ compact.w2.weight",
        source_names=("bridge.output_decoder.runtime_fp32", "compact.w2.weight"),
    )
    return _build_candidate(
        kind="mlp",
        state={"w1.weight": w1, "w2.weight": w2, "w3.weight": w3},
        topology=topology,
        authentication=authentication,
        product_lineage={
            "w1.weight": w1_lineage,
            "w2.weight": w2_lineage,
            "w3.weight": w3_lineage,
        },
    )


def compile_lfm2_fused_short_conv_diagnostic(
    source: Lfm2FusedShortConvDiagnosticInput,
    *,
    expected_input_bundle_sha256: str,
    expected_tensor_sha256: Mapping[str, str],
) -> Lfm2FusedBodyDiagnosticCandidate:
    """Absorb E/D around a compact-channel LFM2 short convolution."""

    tensors, topology, bridges, authentication = _authenticated_input(
        source,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_tensor_sha256=expected_tensor_sha256,
    )
    in_proj, in_lineage = _fp64_product_from_fp32(
        tensors["compact.in_proj.weight"],
        bridges["encoder"].T,
        product_name="in_proj.weight",
        formula="compact.in_proj.weight @ runtime_fp32(input_encoder).T",
        source_names=(
            "compact.in_proj.weight",
            "bridge.input_encoder.runtime_fp32",
        ),
    )
    out_proj, out_lineage = _fp64_product_from_fp32(
        bridges["decoder"].T,
        tensors["compact.out_proj.weight"],
        product_name="out_proj.weight",
        formula="runtime_fp32(output_decoder).T @ compact.out_proj.weight",
        source_names=(
            "bridge.output_decoder.runtime_fp32",
            "compact.out_proj.weight",
        ),
    )
    conv = tensors["compact.conv.weight"].contiguous().clone()
    conv_body = {
        "product_name": "conv.weight",
        "formula": "identity-copy(compact.conv.weight)",
        "source_names": ["compact.conv.weight"],
        "source_sha256": lfm2_authenticated_tensor_sha256(
            tensors["compact.conv.weight"]
        ),
        "materialized_fp32_product": _tensor_record("conv.weight", conv),
        "solve_device": "cpu",
        "solve_dtype": "identity-fp32",
        "materialization_dtype": "float32",
    }
    conv_lineage = {**conv_body, "lineage_sha256": canonical_sha256(conv_body)}
    return _build_candidate(
        kind="short_conv",
        state={
            "conv.weight": conv,
            "in_proj.weight": in_proj,
            "out_proj.weight": out_proj,
        },
        topology=topology,
        authentication=authentication,
        product_lineage={
            "conv.weight": conv_lineage,
            "in_proj.weight": in_lineage,
            "out_proj.weight": out_lineage,
        },
    )


def authenticate_lfm2_fused_body_reconstruction_descriptor(
    descriptor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate one independently supplied fused-body descriptor."""

    if not isinstance(descriptor, Mapping):
        raise Lfm2FusedBodyDiagnosticError(
            "Reconstruction descriptor must be one mapping."
        )
    value = dict(descriptor)
    supplied = _digest(value.pop("descriptor_sha256", None), label="descriptor")
    if canonical_sha256(value) != supplied:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body reconstruction descriptor seal is invalid."
        )
    kind = value.get("kind")
    if (
        value.get("schema") != LFM2_FUSED_BODY_DESCRIPTOR_SCHEMA
        or value.get("status") != "production-safe-fold-only-fused-body-descriptor"
        or value.get("family_adapter") != "lfm2-concrete-runtime-fused-body-v1"
        or kind not in {"mlp", "short_conv"}
        or value.get("state_keys") != list(_state_keys(kind))
        or value.get("parameter_dtype") != "float32"
        or value.get("input_bridge_module_present") is not False
        or value.get("output_bridge_module_present") is not False
        or value.get("bridge_factors_retained_as_runtime_parameters") is not False
        or value.get("runtime_sidecar_parameter_elements") != 0
        or value.get("runtime_parameter_alias_group_count") != 0
        or value.get("runtime_parameter_sharing_used") is not False
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Descriptor does not encode the supported bridge-free fused-body seam."
        )
    _digest(value.get("candidate_state_sha256"), label="descriptor candidate state")
    topology = _validated_topology(value.get("topology"))
    if topology.get("kind") != kind:
        raise Lfm2FusedBodyDiagnosticError(
            "Descriptor topology names a different kind."
        )
    if kind == "mlp":
        expected_topology = {
            "kind",
            "parent_hidden_size",
            "parent_intermediate_size",
            "compact_hidden_size",
            "compact_intermediate_size",
            "parameter_dtype",
            "bias",
            "block_auto_adjust_ff_dim",
        }
        if (
            set(topology) != expected_topology
            or value.get("ordinary_stock_lfm2_module") is not True
            or value.get("explicit_mixed_width_bottleneck_module") is not False
            or value.get("cache_contract") is not None
            or value.get("module_class")
            != "transformers.models.lfm2.modeling_lfm2.Lfm2MLP"
        ):
            raise Lfm2FusedBodyDiagnosticError(
                "MLP reconstruction descriptor is not canonical."
            )
    else:
        expected_topology = {
            "kind",
            "layer_index",
            "parent_hidden_size",
            "compact_hidden_size",
            "kernel_size",
            "parameter_dtype",
            "conv_bias",
            "layer_type",
        }
        cache = value.get("cache_contract")
        if (
            set(topology) != expected_topology
            or value.get("ordinary_stock_lfm2_module") is not False
            or value.get("explicit_mixed_width_bottleneck_module") is not True
            or not isinstance(cache, Mapping)
            or value.get("module_class")
            != "neural_forge.hf_lfm2_fused_body_diagnostic.Lfm2BottleneckShortConv"
            or cache.get("cache_kind") != "transformers-DynamicCache-only"
            or cache.get("static_or_compileable_cache_supported") is not False
            or cache.get("fresh_cache_required_per_candidate") is not True
            or cache.get("per_layer_internal_channel_width")
            != topology.get("compact_hidden_size")
            or cache.get("kernel_width") != topology.get("kernel_size")
        ):
            raise Lfm2FusedBodyDiagnosticError(
                "Short-conv reconstruction/cache descriptor is not canonical."
            )
    return MappingProxyType({**value, "descriptor_sha256": supplied})


def authenticate_lfm2_fused_body_candidate(
    candidate: Lfm2FusedBodyDiagnosticCandidate,
) -> Mapping[str, Any]:
    """Authenticate candidate tensors, lineage, descriptor, and exact counts."""

    torch = _torch()
    if not isinstance(candidate, Lfm2FusedBodyDiagnosticCandidate):
        raise TypeError("candidate must be Lfm2FusedBodyDiagnosticCandidate.")
    receipt = dict(candidate.receipt)
    supplied = _digest(receipt.pop("result_sha256", None), label="candidate result")
    if (
        canonical_sha256(receipt) != supplied
        or receipt.get("schema") != LFM2_FUSED_BODY_CANDIDATE_SCHEMA
        or receipt.get("status") != "NON-FINAL-FOLD-ONLY-FUSED-BODY-DIAGNOSTIC"
        or receipt.get("kind") != candidate.kind
        or receipt.get("operation_audit") != dict(_ZERO_OPERATION_AUDIT)
    ):
        raise Lfm2FusedBodyDiagnosticError("Fused-body candidate receipt is invalid.")
    descriptor = authenticate_lfm2_fused_body_reconstruction_descriptor(
        candidate.reconstruction_descriptor
    )
    state = dict(candidate.state_dict())
    keys = _state_keys(candidate.kind)
    if tuple(sorted(state)) != tuple(sorted(keys)):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body candidate state namespace is invalid."
        )
    for name, tensor in state.items():
        _validated_tensor(
            tensor,
            name=name,
            ndim=3 if name == "conv.weight" else 2,
        )
        if tensor.dtype != torch.float32:
            raise Lfm2FusedBodyDiagnosticError(
                "Fused-body candidate state is not FP32."
            )
    _assert_no_storage_aliases(state, label="fused-body candidate state")
    state_sha = _state_sha256(candidate.kind, state)
    topology = _validated_topology(receipt.get("topology"))
    physical = receipt.get("physical_parameter_accounting")
    if not isinstance(physical, Mapping):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body candidate lacks topology or physical accounting."
        )
    expected_physical = dict(_physical_accounting(topology))
    observed_elements = sum(int(tensor.numel()) for tensor in state.values())
    _digest(receipt.get("input_bundle_sha256"), label="candidate input bundle")
    source_hashes = receipt.get("source_tensor_sha256")
    expected_source_names = (
        _MLP_INPUT_NAMES if candidate.kind == "mlp" else _SHORT_CONV_INPUT_NAMES
    )
    if not isinstance(source_hashes, Mapping) or tuple(sorted(source_hashes)) != tuple(
        sorted(expected_source_names)
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body candidate source hashes are incomplete."
        )
    for name, digest in source_hashes.items():
        _digest(digest, label=f"candidate source {name}")
    bridge_lineage = receipt.get("runtime_bridge_cast_lineage")
    if not isinstance(bridge_lineage, Mapping) or set(bridge_lineage) != {
        "input_encoder",
        "output_decoder",
    }:
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body runtime bridge cast lineage is incomplete."
        )
    for name, raw_row in bridge_lineage.items():
        if not isinstance(raw_row, Mapping):
            raise Lfm2FusedBodyDiagnosticError(
                f"Fused-body {name} bridge cast lineage is malformed."
            )
        row = dict(raw_row)
        seal = _digest(row.pop("lineage_sha256", None), label=f"{name} lineage")
        if (
            canonical_sha256(row) != seal
            or row.get("runtime_fp32", {}).get("dtype") != "float32"
            or row.get("runtime_values_are_product_authority") is not True
        ):
            raise Lfm2FusedBodyDiagnosticError(
                f"Fused-body {name} bridge cast lineage seal is invalid."
            )
    product_lineage = receipt.get("product_lineage")
    if not isinstance(product_lineage, Mapping) or set(product_lineage) != set(keys):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body product lineage does not cover every state tensor."
        )
    for name in keys:
        raw_row = product_lineage[name]
        if not isinstance(raw_row, Mapping):
            raise Lfm2FusedBodyDiagnosticError(
                f"Fused-body product lineage is invalid for {name!r}."
            )
        row = dict(raw_row)
        lineage_seal = _digest(
            row.pop("lineage_sha256", None),
            label=f"{name} product lineage",
        )
        if (
            canonical_sha256(row) != lineage_seal
            or row.get("product_name") != name
            or row.get("materialized_fp32_product", {}).get("sha256")
            != lfm2_authenticated_tensor_sha256(state[name])
        ):
            raise Lfm2FusedBodyDiagnosticError(
                f"Fused-body product lineage is invalid for {name!r}."
            )
    if (
        state_sha != receipt.get("candidate_state_sha256")
        or state_sha != descriptor.get("candidate_state_sha256")
        or descriptor.get("descriptor_sha256")
        != receipt.get("reconstruction_descriptor_sha256")
        or descriptor.get("topology") != topology
        or dict(physical) != expected_physical
        or observed_elements != physical["fused_candidate_parameter_elements"]
        or _stored_tensor_elements(state)
        != physical["fused_candidate_stored_tensor_elements"]
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body candidate state, descriptor, and counts are not bound together."
        )
    return MappingProxyType({**receipt, "result_sha256": supplied})


_BOTTLENECK_SHORT_CONV_RUNTIME_CLASS: type[Any] | None = None


def _bottleneck_short_conv_runtime_class() -> type[Any]:
    global _BOTTLENECK_SHORT_CONV_RUNTIME_CLASS
    if _BOTTLENECK_SHORT_CONV_RUNTIME_CLASS is not None:
        return _BOTTLENECK_SHORT_CONV_RUNTIME_CLASS
    torch = _torch()

    class BottleneckShortConv(torch.nn.Module):
        """Parent-I/O, compact-channel LFM2 short convolution."""

        def __init__(
            self,
            *,
            parent_hidden_size: int,
            compact_hidden_size: int,
            kernel_size: int,
            layer_index: int,
            device: Any = "cpu",
        ) -> None:
            super().__init__()
            parent_hidden = _positive_int(
                parent_hidden_size,
                label="parent_hidden_size",
            )
            compact_hidden = _positive_int(
                compact_hidden_size,
                label="compact_hidden_size",
            )
            kernel = _positive_int(kernel_size, label="kernel_size")
            layer = _nonnegative_int(layer_index, label="layer_index")
            if compact_hidden >= parent_hidden:
                raise Lfm2FusedBodyDiagnosticError(
                    "Bottleneck short conv requires fewer internal than external channels."
                )
            resolved_device = torch.device(device)
            if resolved_device.type == "meta":
                raise Lfm2FusedBodyDiagnosticError(
                    "Bottleneck short conv cannot be constructed on the meta device."
                )
            self.external_hidden_size = parent_hidden
            self.internal_hidden_size = compact_hidden
            self.L_cache = kernel
            self.layer_idx = layer
            self.layer_type = "conv"
            self.bias = False
            self.conv = torch.nn.Conv1d(
                compact_hidden,
                compact_hidden,
                kernel,
                groups=compact_hidden,
                bias=False,
                padding=kernel - 1,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.in_proj = torch.nn.Linear(
                parent_hidden,
                3 * compact_hidden,
                bias=False,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.out_proj = torch.nn.Linear(
                compact_hidden,
                parent_hidden,
                bias=False,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.requires_grad_(False)
            self.eval()

        def _validate_hidden(self, hidden_states: Any) -> None:
            parameter = self.in_proj.weight
            if (
                not isinstance(hidden_states, torch.Tensor)
                or hidden_states.layout != torch.strided
                or hidden_states.device.type == "meta"
                or hidden_states.ndim != 3
                or int(hidden_states.shape[-1]) != self.external_hidden_size
                or hidden_states.dtype != torch.float32
                or hidden_states.device != parameter.device
                or hidden_states.requires_grad
                or not bool(torch.isfinite(hidden_states).all().item())
            ):
                raise Lfm2FusedBodyDiagnosticError(
                    "Bottleneck short-conv hidden states must be finite detached "
                    "FP32 [batch, sequence, parent_hidden] values on the module device."
                )

        def _validate_auxiliary(
            self,
            hidden_states: Any,
            attention_mask: Any,
            seq_idx: Any,
        ) -> None:
            batch, sequence = map(int, hidden_states.shape[:2])
            if attention_mask is not None:
                if (
                    not isinstance(attention_mask, torch.Tensor)
                    or attention_mask.device != hidden_states.device
                    or attention_mask.ndim != 2
                    or tuple(attention_mask.shape) != (batch, sequence)
                    or (
                        attention_mask.dtype.is_floating_point
                        and not bool(torch.isfinite(attention_mask).all().item())
                    )
                ):
                    raise Lfm2FusedBodyDiagnosticError(
                        "Short-conv attention_mask must be finite [batch, sequence] "
                        "values on the hidden-state device."
                    )
            if seq_idx is not None:
                if (
                    not isinstance(seq_idx, torch.Tensor)
                    or seq_idx.device != hidden_states.device
                    or seq_idx.dtype not in {torch.int32, torch.int64}
                    or seq_idx.ndim != 2
                    or tuple(seq_idx.shape) != (batch, sequence)
                ):
                    raise Lfm2FusedBodyDiagnosticError(
                        "Short-conv seq_idx must be int32/int64 [batch, sequence] "
                        "values on the hidden-state device."
                    )

        def _validate_cache(self, past_key_values: Any) -> None:
            if past_key_values is None:
                return
            try:
                from transformers.cache_utils import (
                    DynamicCache,
                    LinearAttentionCacheLayerMixin,
                )
            except ImportError as exc:  # pragma: no cover - required runtime
                raise Lfm2FusedBodyDiagnosticError(
                    "Transformers DynamicCache is required for cached short conv."
                ) from exc
            if not isinstance(past_key_values, DynamicCache) or bool(
                getattr(past_key_values, "is_compileable", False)
            ):
                raise Lfm2FusedBodyDiagnosticError(
                    "Bottleneck short conv rejects static, compileable, or non-Dynamic caches."
                )
            layers = getattr(past_key_values, "layers", None)
            if not isinstance(layers, list) or self.layer_idx >= len(layers):
                raise Lfm2FusedBodyDiagnosticError(
                    "Dynamic cache does not contain the bottleneck conv layer slot."
                )
            layer = layers[self.layer_idx]
            if not isinstance(layer, LinearAttentionCacheLayerMixin):
                raise Lfm2FusedBodyDiagnosticError(
                    "Dynamic cache layer slot is not a linear-attention/conv cache."
                )
            if bool(getattr(layer, "is_conv_states_initialized", False)):
                state = getattr(layer, "conv_states", None)
                parameter = self.conv.weight
                if (
                    not isinstance(state, torch.Tensor)
                    or state.ndim != 3
                    or int(state.shape[1]) != self.internal_hidden_size
                    or int(state.shape[-1]) != self.L_cache
                    or state.dtype != torch.float32
                    or state.device != parameter.device
                    or not bool(torch.isfinite(state).all().item())
                ):
                    raise Lfm2FusedBodyDiagnosticError(
                        "Existing conv cache is not finite FP32 "
                        "[batch, compact_hidden, kernel] state on the module device."
                    )

        def cuda_kernels_forward(
            self,
            x: Any,
            past_key_values: Any = None,
            attention_mask: Any = None,
            seq_idx: Any = None,
        ) -> Any:
            from transformers.models.lfm2 import modeling_lfm2 as runtime

            x = runtime.apply_mask_to_padding_states(x, attention_mask)
            BCx = self.in_proj(x).transpose(-1, -2)
            B, C, x = BCx.chunk(3, dim=-2)
            Bx = B * x
            conv_weights = self.conv.weight.view(
                self.conv.weight.size(0),
                self.conv.weight.size(2),
            )
            if past_key_values is not None and past_key_values.has_previous_state(
                self.layer_idx
            ):
                conv_out = runtime.causal_conv1d_update(
                    Bx.squeeze(-1),
                    past_key_values.layers[self.layer_idx].conv_states,
                    conv_weights,
                    None,
                    None,
                )
                conv_out = conv_out.unsqueeze(-1)
            else:
                if past_key_values is not None:
                    conv_state = torch.nn.functional.pad(
                        Bx,
                        (self.L_cache - Bx.shape[-1], 0),
                    )
                    past_key_values.update_conv_state(conv_state, self.layer_idx)
                conv_out = runtime.causal_conv1d_fn(
                    Bx,
                    conv_weights,
                    None,
                    activation=None,
                    seq_idx=seq_idx,
                )
            y = C * conv_out
            return self.out_proj(y.transpose(-1, -2).contiguous())

        def slow_forward(
            self,
            x: Any,
            past_key_values: Any = None,
            attention_mask: Any = None,
            seq_idx: Any = None,
        ) -> Any:
            from transformers.models.lfm2 import modeling_lfm2 as runtime

            seqlen = x.shape[1]
            x = runtime.apply_mask_to_padding_states(x, attention_mask)
            BCx = self.in_proj(x).transpose(-1, -2)
            B, C, x = BCx.chunk(3, dim=-2)
            Bx = B * x
            if past_key_values is not None and past_key_values.has_previous_state(
                self.layer_idx
            ):
                conv_state = past_key_values.update_conv_state(Bx, self.layer_idx)
                conv_out = torch.sum(
                    conv_state.to(Bx.device) * self.conv.weight[:, 0, :],
                    dim=-1,
                ).unsqueeze(-1)
            elif seq_idx is not None and x.shape[0] == 1:
                if past_key_values is not None:
                    conv_state = torch.nn.functional.pad(
                        Bx,
                        (self.L_cache - Bx.shape[-1], 0),
                    )
                    past_key_values.update_conv_state(conv_state, self.layer_idx)
                si = seq_idx[0]
                change = (si[1:] != si[:-1]).nonzero(as_tuple=True)[0] + 1
                bounds = torch.cat(
                    [
                        change.new_zeros(1),
                        change,
                        change.new_full((1,), si.numel()),
                    ]
                ).tolist()
                parts = []
                for index in range(len(bounds) - 1):
                    start, stop = bounds[index], bounds[index + 1]
                    if stop > start:
                        parts.append(
                            self.conv(Bx[:, :, start:stop])[..., : stop - start]
                        )
                conv_out = torch.cat(parts, dim=-1)
            else:
                if past_key_values is not None:
                    conv_state = torch.nn.functional.pad(
                        Bx,
                        (self.L_cache - Bx.shape[-1], 0),
                    )
                    past_key_values.update_conv_state(conv_state, self.layer_idx)
                conv_out = self.conv(Bx)[..., :seqlen]
            y = C * conv_out
            return self.out_proj(y.transpose(-1, -2).contiguous())

        def forward(
            self,
            hidden_states: Any,
            past_key_values: Any = None,
            attention_mask: Any = None,
            seq_idx: Any = None,
        ) -> Any:
            from transformers.models.lfm2 import modeling_lfm2 as runtime

            self._validate_hidden(hidden_states)
            self._validate_auxiliary(hidden_states, attention_mask, seq_idx)
            self._validate_cache(past_key_values)
            if (
                runtime.is_fast_path_available
                and "cuda" in hidden_states.device.type
                and not runtime.is_torchdynamo_compiling()
            ):
                result = self.cuda_kernels_forward(
                    hidden_states,
                    past_key_values,
                    attention_mask,
                    seq_idx,
                )
            else:
                result = self.slow_forward(
                    hidden_states,
                    past_key_values,
                    attention_mask,
                    seq_idx,
                )
            self._validate_cache(past_key_values)
            if (
                result.dtype != torch.float32
                or tuple(result.shape)
                != (
                    int(hidden_states.shape[0]),
                    int(hidden_states.shape[1]),
                    self.external_hidden_size,
                )
                or not bool(torch.isfinite(result).all().item())
            ):
                raise Lfm2FusedBodyDiagnosticError(
                    "Bottleneck short conv produced invalid parent-width FP32 output."
                )
            return result

    BottleneckShortConv.__name__ = "Lfm2BottleneckShortConvRuntime"
    BottleneckShortConv.__qualname__ = "Lfm2BottleneckShortConvRuntime"
    _BOTTLENECK_SHORT_CONV_RUNTIME_CLASS = BottleneckShortConv
    return BottleneckShortConv


class Lfm2BottleneckShortConv:
    """Lazy public constructor for the concrete optional-PyTorch runtime class."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return _bottleneck_short_conv_runtime_class()(*args, **kwargs)


def _parent_config_dict(parent_config: Any) -> dict[str, Any]:
    to_dict = getattr(parent_config, "to_dict", None)
    if not callable(to_dict):
        raise Lfm2FusedBodyDiagnosticError(
            "Reconstruction requires one concrete LFM2 parent config."
        )
    result = to_dict()
    if not isinstance(result, Mapping):
        raise Lfm2FusedBodyDiagnosticError(
            "Parent LFM2 config did not serialize to one mapping."
        )
    return dict(result)


def reconstruct_lfm2_fused_body_candidate(
    *,
    candidate: Lfm2FusedBodyDiagnosticCandidate,
    parent_config: Any,
    expected_candidate_result_sha256: str,
    expected_descriptor_sha256: str,
    device: Any = "cpu",
) -> Lfm2FusedBodyReconstruction:
    """Freshly reconstruct and strictly load one fused MLP or short conv."""

    torch = _torch()
    receipt = authenticate_lfm2_fused_body_candidate(candidate)
    descriptor = authenticate_lfm2_fused_body_reconstruction_descriptor(
        candidate.reconstruction_descriptor
    )
    if receipt["result_sha256"] != _digest(
        expected_candidate_result_sha256,
        label="expected candidate result",
    ) or descriptor["descriptor_sha256"] != _digest(
        expected_descriptor_sha256,
        label="expected descriptor",
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Candidate or descriptor differs from its external reconstruction pin."
        )
    config_value = _parent_config_dict(parent_config)
    topology = descriptor["topology"]
    parent_hidden = int(topology["parent_hidden_size"])
    if int(getattr(parent_config, "hidden_size", -1)) != parent_hidden:
        raise Lfm2FusedBodyDiagnosticError(
            "Parent config hidden width differs from the fused-body descriptor."
        )
    requested_device = torch.device(device)
    if requested_device.type == "meta":
        raise Lfm2FusedBodyDiagnosticError(
            "Fused-body reconstruction cannot target the meta device."
        )
    if candidate.kind == "mlp":
        if (
            int(getattr(parent_config, "intermediate_size", -1))
            != int(topology["parent_intermediate_size"])
            or getattr(parent_config, "block_auto_adjust_ff_dim", None) is not False
            or topology.get("bias") is not False
            or topology.get("block_auto_adjust_ff_dim") is not False
        ):
            raise Lfm2FusedBodyDiagnosticError(
                "Parent config does not match the bias-free fixed-width MLP seam."
            )
        try:
            from transformers import Lfm2Config
            from transformers.models.lfm2.modeling_lfm2 import Lfm2MLP
        except ImportError as exc:  # pragma: no cover - required reconstruction
            raise Lfm2FusedBodyDiagnosticError(
                "Concrete Transformers LFM2 classes are required for MLP reconstruction."
            ) from exc
        config_value.update(
            {
                "hidden_size": parent_hidden,
                "intermediate_size": int(topology["compact_intermediate_size"]),
                "block_auto_adjust_ff_dim": False,
            }
        )
        module = Lfm2MLP(Lfm2Config(**config_value))
    else:
        layer_index = int(topology["layer_index"])
        layer_types = getattr(parent_config, "layer_types", None)
        if (
            getattr(parent_config, "conv_bias", None) is not False
            or int(getattr(parent_config, "conv_L_cache", -1))
            != int(topology["kernel_size"])
            or not isinstance(layer_types, (list, tuple))
            or layer_index >= len(layer_types)
            or layer_types[layer_index] != "conv"
        ):
            raise Lfm2FusedBodyDiagnosticError(
                "Parent config does not match the bias-free short-conv layer seam."
            )
        module = Lfm2BottleneckShortConv(
            parent_hidden_size=parent_hidden,
            compact_hidden_size=int(topology["compact_hidden_size"]),
            kernel_size=int(topology["kernel_size"]),
            layer_index=layer_index,
            device=requested_device,
        )
    module.to(device=requested_device, dtype=torch.float32)
    state = {
        name: tensor.to(device=requested_device, dtype=torch.float32)
        for name, tensor in candidate.state_dict().items()
    }
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Lfm2FusedBodyDiagnosticError(
            "Fresh fused-body module rejected its complete authenticated state."
        )
    module.requires_grad_(False)
    module.eval()
    runtime_state = {
        name: _plain_tensor(
            tensor,
            name=name,
            ndim=3 if name == "conv.weight" else 2,
            clone=False,
        )
        for name, tensor in module.state_dict().items()
    }
    _assert_no_storage_aliases(runtime_state, label="reconstructed fused-body module")
    alias_groups = _module_alias_group_count(module)
    parameter_elements = sum(
        int(parameter.numel()) for parameter in module.parameters()
    )
    stored_elements = _stored_tensor_elements(runtime_state)
    expected_elements = receipt["physical_parameter_accounting"][
        "fused_candidate_parameter_elements"
    ]
    state_sha = _state_sha256(candidate.kind, runtime_state)
    if (
        alias_groups != 0
        or parameter_elements != expected_elements
        or stored_elements != expected_elements
        or state_sha != receipt["candidate_state_sha256"]
        or any(parameter.requires_grad for parameter in module.parameters())
        or hasattr(module, "input_bridge")
        or hasattr(module, "output_bridge")
    ):
        raise Lfm2FusedBodyDiagnosticError(
            "Fresh fused-body reconstruction differs from its authenticated state/count."
        )
    body = {
        "schema": LFM2_FUSED_BODY_RECONSTRUCTION_SCHEMA,
        "status": "fused-body-module-freshly-reconstructed",
        "kind": candidate.kind,
        "candidate_result_sha256": receipt["result_sha256"],
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "candidate_state_sha256": receipt["candidate_state_sha256"],
        "reconstructed_state_sha256": state_sha,
        "parameter_elements": parameter_elements,
        "stored_tensor_elements": stored_elements,
        "runtime_parameter_alias_group_count": alias_groups,
        "input_bridge_module_present": False,
        "output_bridge_module_present": False,
        "runtime_sidecar_parameter_elements": 0,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    reconstruction_receipt = {**body, "result_sha256": canonical_sha256(body)}
    return Lfm2FusedBodyReconstruction(
        module=module,
        receipt=MappingProxyType(reconstruction_receipt),
    )


__all__ = [
    "LFM2_FUSED_BODY_CANDIDATE_SCHEMA",
    "LFM2_FUSED_BODY_DESCRIPTOR_SCHEMA",
    "LFM2_FUSED_BODY_INPUT_SCHEMA",
    "LFM2_FUSED_BODY_RECONSTRUCTION_SCHEMA",
    "Lfm2BottleneckShortConv",
    "Lfm2FusedBodyDiagnosticCandidate",
    "Lfm2FusedBodyDiagnosticError",
    "Lfm2FusedBodyReconstruction",
    "Lfm2FusedMlpDiagnosticInput",
    "Lfm2FusedShortConvDiagnosticInput",
    "authenticate_lfm2_fused_body_candidate",
    "authenticate_lfm2_fused_body_reconstruction_descriptor",
    "compile_lfm2_fused_mlp_diagnostic",
    "compile_lfm2_fused_short_conv_diagnostic",
    "fingerprint_lfm2_fused_body_diagnostic_input",
    "fingerprint_lfm2_fused_mlp_diagnostic_input",
    "fingerprint_lfm2_fused_short_conv_diagnostic_input",
    "lfm2_fused_mlp_parameter_accounting",
    "lfm2_fused_short_conv_parameter_accounting",
    "reconstruct_lfm2_fused_body_candidate",
]
