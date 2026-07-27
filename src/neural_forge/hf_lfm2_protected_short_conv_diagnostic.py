"""Protected-channel, fold-only diagnostics for an LFM2 short convolution.

An LFM2 short convolution is not a dense linear operator internally.  Each
parent channel owns an inseparable ``(B, x, C, depthwise kernel)`` path.  A
dense merge of those channels therefore does not preserve the operator's
algebra.  This module keeps every parent internal channel intact and folds a
residual map only at the two linear seams.

For parent width ``H``, child width ``h``, encoder ``E[H,h]``, decoder
``D[h,H]``, and ``P = E @ D``, the diagnostic candidates are::

    parent-input:   Win*  = Win @ P.T
    parent-output:  Wout* = P.T @ Wout
    parent-both:    both substitutions above
    standalone:     Win*  = Win @ D.T, Wout* = E.T @ Wout

The three parent-shell candidates retain parent residual I/O.  The standalone
candidate has external child width ``h`` and internal parent width ``H``.  In
all four cases the parent depthwise convolution is copied exactly, and every
closed-form product is evaluated in FP64 before one explicit FP32
materialization.  There are no runtime bridges, sidecars, aliases, training,
pruning, quantization, sparsity, or parameter sharing.
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


LFM2_PROTECTED_SHORT_CONV_INPUT_SCHEMA = (
    "neural-forge-lfm2-protected-short-conv-input-v1"
)
LFM2_PROTECTED_SHORT_CONV_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-protected-short-conv-candidate-v1"
)
LFM2_PROTECTED_SHORT_CONV_DESCRIPTOR_SCHEMA = (
    "neural-forge-lfm2-protected-short-conv-descriptor-v1"
)
LFM2_PROTECTED_SHORT_CONV_RECONSTRUCTION_SCHEMA = (
    "neural-forge-lfm2-protected-short-conv-reconstruction-v1"
)

Lfm2ProtectedShortConvVariant = Literal[
    "parent-input-roundtrip",
    "parent-output-roundtrip",
    "parent-both-roundtrip",
    "standalone-child-stem",
]

_VARIANTS = (
    "parent-input-roundtrip",
    "parent-output-roundtrip",
    "parent-both-roundtrip",
    "standalone-child-stem",
)
_SOURCE_NAMES = (
    "parent.conv.weight",
    "parent.in_proj.weight",
    "parent.out_proj.weight",
    "bridge.input_encoder",
    "bridge.output_decoder",
)
_STATE_KEYS = ("conv.weight", "in_proj.weight", "out_proj.weight")
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "candidate_model_forward_calls": 0,
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
        "internal_channels_averaged": 0,
        "internal_channels_removed": 0,
        "runtime_parameter_aliases_created": 0,
        "runtime_parameter_sharing_used": False,
        "network_access": False,
    }
)


class Lfm2ProtectedShortConvDiagnosticError(RuntimeError):
    """A protected short-conv source, candidate, or runtime is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError(
            "Protected LFM2 short-conv diagnostics require PyTorch."
        ) from exc
    return torch


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{label} must be one positive integer."
        )
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{label} must be one nonnegative integer."
        )
    return value


def _variant(value: Any) -> Lfm2ProtectedShortConvVariant:
    if value not in _VARIANTS:
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"Unsupported protected short-conv variant: {value!r}."
        )
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _validated_fp32_tensor(value: Any, *, name: str, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype != torch.float32
        or bool(getattr(value, "is_quantized", False))
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{name} must be one finite dense detached non-quantized FP32 "
            f"rank-{ndim} tensor."
        )
    return value.detach()


def _cpu_clone(value: Any, *, name: str, ndim: int) -> Any:
    return (
        _validated_fp32_tensor(value, name=name, ndim=ndim)
        .to(device="cpu")
        .contiguous()
        .clone()
    )


def _storage_identity(tensor: Any) -> tuple[str, str, int, int]:
    return (
        str(tensor.device),
        str(tensor.dtype),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor.untyped_storage().nbytes()),
    )


def _assert_no_storage_aliases(tensors: Mapping[str, Any], *, label: str) -> None:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, tensor in tensors.items():
        groups.setdefault(_storage_identity(tensor), []).append(name)
    aliases = [sorted(names) for names in groups.values() if len(names) > 1]
    if aliases:
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{label} contains shared or aliased tensor storage: {aliases!r}."
        )


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(size) for size in tensor.shape],
        "dtype": "float32",
        "sha256": lfm2_authenticated_tensor_sha256(tensor),
    }


def _state_sha256(state: Mapping[str, Any]) -> str:
    if tuple(sorted(state)) != tuple(sorted(_STATE_KEYS)):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv state namespace is incomplete."
        )
    return canonical_sha256(
        [_tensor_record(name, state[name]) for name in sorted(state)]
    )


def _stored_tensor_elements(tensors: Mapping[str, Any]) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    total = 0
    for tensor in tensors.values():
        identity = _storage_identity(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        storage_bytes = int(tensor.untyped_storage().nbytes())
        element_bytes = int(tensor.element_size())
        if storage_bytes % element_bytes:
            raise Lfm2ProtectedShortConvDiagnosticError(
                "Tensor storage bytes are not divisible by element width."
            )
        total += storage_bytes // element_bytes
    return total


def _module_alias_group_count(module: Any) -> int:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    try:
        parameters = module.named_parameters(remove_duplicate=False)
    except TypeError as exc:  # pragma: no cover - supported PyTorch is required
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Runtime alias authentication requires remove_duplicate=False."
        ) from exc
    for name, tensor in parameters:
        groups.setdefault(_storage_identity(tensor), []).append(str(name))
    return sum(len(names) > 1 for names in groups.values())


def _sealed_identity_copy(
    tensor: Any,
    *,
    source_name: str,
    product_name: str,
) -> tuple[Any, Mapping[str, Any]]:
    product = tensor.to(device="cpu").contiguous().clone()
    body = {
        "product_name": product_name,
        "formula": f"identity-copy({source_name})",
        "source_names": [source_name],
        "source_tensor_sha256": lfm2_authenticated_tensor_sha256(tensor),
        "materialized_fp32_product": _tensor_record(product_name, product),
        "solve_device": "cpu",
        "solve_dtype": "identity-fp32",
        "materialization_dtype": "float32",
    }
    return product, MappingProxyType({**body, "lineage_sha256": canonical_sha256(body)})


def _sealed_fp64_product(
    left: Any,
    right: Any,
    *,
    left_name: str,
    right_name: str,
    product_name: str,
    formula: str,
) -> tuple[Any, Mapping[str, Any]]:
    torch = _torch()
    left = _validated_fp32_tensor(left, name=left_name, ndim=2)
    right = _validated_fp32_tensor(right, name=right_name, ndim=2)
    if int(left.shape[1]) != int(right.shape[0]):
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{product_name} operands do not have compatible matrix geometry."
        )
    fp64 = (
        left.to(device="cpu", dtype=torch.float64)
        @ right.to(device="cpu", dtype=torch.float64)
    ).contiguous()
    if not bool(torch.isfinite(fp64).all().item()):
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{product_name} produced non-finite FP64 values."
        )
    product = fp64.to(dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(product).all().item()):
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{product_name} overflowed during FP32 materialization."
        )
    body = {
        "product_name": product_name,
        "formula": formula,
        "source_names": [left_name, right_name],
        "left_runtime_fp32_sha256": lfm2_authenticated_tensor_sha256(left),
        "right_runtime_fp32_sha256": lfm2_authenticated_tensor_sha256(right),
        "fp64_product": {
            "name": f"{product_name}.fp64",
            "shape": [int(size) for size in fp64.shape],
            "dtype": "float64",
            "sha256": lfm2_authenticated_tensor_sha256(fp64),
        },
        "materialized_fp32_product": _tensor_record(product_name, product),
        "solve_device": "cpu",
        "solve_dtype": "float64",
        "materialization_dtype": "float32",
    }
    return product, MappingProxyType({**body, "lineage_sha256": canonical_sha256(body)})


@dataclass(frozen=True, slots=True)
class Lfm2ProtectedShortConvInput:
    """Detached parent short-conv tensors and one child residual bridge pair."""

    layer_index: int
    parent_hidden_size: int
    child_hidden_size: int
    kernel_size: int
    parent_conv_weight: Any = field(repr=False, compare=False)
    parent_in_proj_weight: Any = field(repr=False, compare=False)
    parent_out_proj_weight: Any = field(repr=False, compare=False)
    input_encoder: Any = field(repr=False, compare=False)
    output_decoder: Any = field(repr=False, compare=False)
    parent_conv_bias: Any | None = field(default=None, repr=False, compare=False)
    parent_in_proj_bias: Any | None = field(default=None, repr=False, compare=False)
    parent_out_proj_bias: Any | None = field(default=None, repr=False, compare=False)

    def named_tensors(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "parent.conv.weight": self.parent_conv_weight,
                "parent.in_proj.weight": self.parent_in_proj_weight,
                "parent.out_proj.weight": self.parent_out_proj_weight,
                "bridge.input_encoder": self.input_encoder,
                "bridge.output_decoder": self.output_decoder,
            }
        )


@dataclass(frozen=True, slots=True)
class Lfm2ProtectedShortConvCandidate:
    """One bridge-free protected-channel candidate with sealed lineage."""

    variant: Lfm2ProtectedShortConvVariant
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
class Lfm2ProtectedShortConvReconstruction:
    """One freshly reconstructed protected short-conv runtime."""

    module: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


def _normalized_source(
    source: Lfm2ProtectedShortConvInput,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(source, Lfm2ProtectedShortConvInput):
        raise TypeError("source must be Lfm2ProtectedShortConvInput.")
    _nonnegative_int(source.layer_index, label="layer_index")
    parent = _positive_int(source.parent_hidden_size, label="parent_hidden_size")
    child = _positive_int(source.child_hidden_size, label="child_hidden_size")
    kernel = _positive_int(source.kernel_size, label="kernel_size")
    if child >= parent:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv diagnostics require child_hidden_size < "
            "parent_hidden_size."
        )
    if any(
        bias is not None
        for bias in (
            source.parent_conv_bias,
            source.parent_in_proj_bias,
            source.parent_out_proj_bias,
        )
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected LFM2 short-conv diagnostics require conv_bias=False."
        )
    raw = dict(source.named_tensors())
    if tuple(raw) != _SOURCE_NAMES:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv source namespace is not canonical."
        )
    checked = {
        name: _validated_fp32_tensor(
            tensor,
            name=name,
            ndim=3 if name == "parent.conv.weight" else 2,
        )
        for name, tensor in raw.items()
    }
    _assert_no_storage_aliases(checked, label="protected short-conv source")
    tensors = {
        name: tensor.to(device="cpu").contiguous().clone()
        for name, tensor in checked.items()
    }
    expected_shapes = {
        "parent.conv.weight": (parent, 1, kernel),
        "parent.in_proj.weight": (3 * parent, parent),
        "parent.out_proj.weight": (parent, parent),
        "bridge.input_encoder": (parent, child),
        "bridge.output_decoder": (child, parent),
    }
    if any(
        tuple(tensors[name].shape) != shape for name, shape in expected_shapes.items()
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv source tensors do not match declared geometry."
        )
    topology = {
        "layer_index": source.layer_index,
        "parent_hidden_size": parent,
        "child_hidden_size": child,
        "kernel_size": kernel,
        "parameter_dtype": "float32",
        "conv_bias": False,
        "layer_type": "conv",
        "parent_internal_channels_protected": parent,
    }
    return tensors, topology


def fingerprint_lfm2_protected_short_conv_input(
    source: Lfm2ProtectedShortConvInput,
) -> Mapping[str, Any]:
    """Authenticate exact FP32 parent weights and bridge factors."""

    tensors, topology = _normalized_source(source)
    body = {
        "schema": LFM2_PROTECTED_SHORT_CONV_INPUT_SCHEMA,
        "status": "protected-short-conv-inputs-authenticated",
        "topology": topology,
        "tensor_records": {
            name: _tensor_record(name, tensor) for name, tensor in tensors.items()
        },
        "input_storage_alias_group_count": 0,
        "input_parameters_quantized": 0,
        "input_parameters_sparse": 0,
        "input_parameters_requiring_grad": 0,
    }
    return MappingProxyType({**body, "input_bundle_sha256": canonical_sha256(body)})


def _authenticated_source(
    source: Lfm2ProtectedShortConvInput,
    *,
    expected_input_bundle_sha256: str,
    expected_tensor_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    expected_bundle = _digest(
        expected_input_bundle_sha256,
        label="expected protected input bundle",
    )
    if not isinstance(expected_tensor_sha256, Mapping) or tuple(
        sorted(expected_tensor_sha256)
    ) != tuple(sorted(_SOURCE_NAMES)):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Expected tensor pins must cover the exact protected source namespace."
        )
    expected_pins = {
        name: _digest(value, label=f"expected {name}")
        for name, value in expected_tensor_sha256.items()
    }
    tensors, topology = _normalized_source(source)
    authentication = fingerprint_lfm2_protected_short_conv_input(source)
    observed_pins = {
        name: row["sha256"] for name, row in authentication["tensor_records"].items()
    }
    if (
        authentication["input_bundle_sha256"] != expected_bundle
        or observed_pins != expected_pins
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv source differs from its independent pins."
        )
    return tensors, topology, authentication


def lfm2_protected_short_conv_parameter_accounting(
    *,
    parent_hidden_size: int,
    child_hidden_size: int,
    kernel_size: int,
    variant: Lfm2ProtectedShortConvVariant,
) -> Mapping[str, Any]:
    """Return exact dense physical counts for one protected candidate."""

    parent = _positive_int(parent_hidden_size, label="parent_hidden_size")
    child = _positive_int(child_hidden_size, label="child_hidden_size")
    kernel = _positive_int(kernel_size, label="kernel_size")
    variant = _variant(variant)
    if child >= parent:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected accounting requires child_hidden_size < parent_hidden_size."
        )
    standalone = variant == "standalone-child-stem"
    external = child if standalone else parent
    internal = parent
    original = 4 * parent * parent + kernel * parent
    candidate = 4 * external * internal + kernel * internal
    body = {
        "variant": variant,
        "original_parent_component_parameter_elements": original,
        "candidate_external_hidden_size": external,
        "candidate_internal_hidden_size": internal,
        "candidate_in_proj_parameter_elements": 3 * internal * external,
        "candidate_depthwise_conv_parameter_elements": kernel * internal,
        "candidate_out_proj_parameter_elements": external * internal,
        "candidate_parameter_elements": candidate,
        "candidate_stored_tensor_elements": candidate,
        "candidate_parameter_bytes_fp32": 4 * candidate,
        "physical_parameter_reduction_from_parent": original - candidate,
        "source_bridge_factor_elements_not_retained": 2 * parent * child,
        "runtime_input_bridge_parameter_elements": 0,
        "runtime_output_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
        "runtime_shared_parameter_elements": 0,
        "runtime_parameter_alias_group_count": 0,
        "parameter_reduction_attributed_to_sharing": False,
        "protected_parent_internal_channel_count": parent,
        "removed_parent_internal_channel_count": 0,
    }
    if standalone and original - candidate <= 0:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Standalone protected stem did not physically reduce its parent component."
        )
    if not standalone and original != candidate:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Parent-shell protected candidate unexpectedly changed physical size."
        )
    return MappingProxyType(body)


def _candidate_topology(
    source_topology: Mapping[str, Any],
    variant: Lfm2ProtectedShortConvVariant,
) -> Mapping[str, Any]:
    variant = _variant(variant)
    parent = int(source_topology["parent_hidden_size"])
    child = int(source_topology["child_hidden_size"])
    standalone = variant == "standalone-child-stem"
    return MappingProxyType(
        {
            **dict(source_topology),
            "variant": variant,
            "external_hidden_size": child if standalone else parent,
            "internal_hidden_size": parent,
            "parent_residual_shell_retained": not standalone,
            "standalone_child_residual_interface": standalone,
        }
    )


def _validate_candidate_topology(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv topology must be one mapping."
        )
    topology = dict(value)
    expected = {
        "layer_index",
        "parent_hidden_size",
        "child_hidden_size",
        "kernel_size",
        "parameter_dtype",
        "conv_bias",
        "layer_type",
        "parent_internal_channels_protected",
        "variant",
        "external_hidden_size",
        "internal_hidden_size",
        "parent_residual_shell_retained",
        "standalone_child_residual_interface",
    }
    if set(topology) != expected:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv topology fields are not canonical."
        )
    _nonnegative_int(topology["layer_index"], label="topology layer_index")
    parent = _positive_int(
        topology["parent_hidden_size"], label="topology parent_hidden_size"
    )
    child = _positive_int(
        topology["child_hidden_size"], label="topology child_hidden_size"
    )
    _positive_int(topology["kernel_size"], label="topology kernel_size")
    variant = _variant(topology["variant"])
    standalone = variant == "standalone-child-stem"
    if (
        child >= parent
        or topology["parameter_dtype"] != "float32"
        or topology["conv_bias"] is not False
        or topology["layer_type"] != "conv"
        or topology["parent_internal_channels_protected"] != parent
        or topology["internal_hidden_size"] != parent
        or topology["external_hidden_size"] != (child if standalone else parent)
        or topology["parent_residual_shell_retained"] is standalone
        or topology["standalone_child_residual_interface"] is not standalone
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv topology violates its parent-channel contract."
        )
    return MappingProxyType(topology)


def _build_candidate(
    *,
    variant: Lfm2ProtectedShortConvVariant,
    state: Mapping[str, Any],
    topology: Mapping[str, Any],
    authentication: Mapping[str, Any],
    residual_roundtrip_lineage: Mapping[str, Any],
    product_lineage: Mapping[str, Mapping[str, Any]],
) -> Lfm2ProtectedShortConvCandidate:
    topology = _validate_candidate_topology(topology)
    if tuple(sorted(state)) != tuple(sorted(_STATE_KEYS)):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate builder received incomplete state."
        )
    normalized = {
        name: _cpu_clone(
            state[name],
            name=name,
            ndim=3 if name == "conv.weight" else 2,
        )
        for name in _STATE_KEYS
    }
    _assert_no_storage_aliases(normalized, label="protected short-conv candidate")
    external = int(topology["external_hidden_size"])
    internal = int(topology["internal_hidden_size"])
    kernel = int(topology["kernel_size"])
    expected_shapes = {
        "conv.weight": (internal, 1, kernel),
        "in_proj.weight": (3 * internal, external),
        "out_proj.weight": (external, internal),
    }
    if any(
        tuple(normalized[name].shape) != shape
        for name, shape in expected_shapes.items()
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate state does not match its external/internal geometry."
        )
    state_sha = _state_sha256(normalized)
    accounting = dict(
        lfm2_protected_short_conv_parameter_accounting(
            parent_hidden_size=int(topology["parent_hidden_size"]),
            child_hidden_size=int(topology["child_hidden_size"]),
            kernel_size=kernel,
            variant=variant,
        )
    )
    observed = sum(int(tensor.numel()) for tensor in normalized.values())
    stored = _stored_tensor_elements(normalized)
    if (
        observed != accounting["candidate_parameter_elements"]
        or stored != accounting["candidate_stored_tensor_elements"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate state differs from dense physical accounting."
        )
    descriptor_body = {
        "schema": LFM2_PROTECTED_SHORT_CONV_DESCRIPTOR_SCHEMA,
        "status": "fold-only-protected-short-conv-descriptor",
        "family_adapter": "lfm2-protected-parent-channel-short-conv-v1",
        "variant": variant,
        "module_class": (
            "neural_forge.hf_lfm2_protected_short_conv_diagnostic."
            "Lfm2ProtectedShortConvRuntime"
        ),
        "topology": dict(topology),
        "state_keys": list(_STATE_KEYS),
        "candidate_state_sha256": state_sha,
        "parameter_dtype": "float32",
        "input_bridge_module_present": False,
        "output_bridge_module_present": False,
        "bridge_factors_retained_as_runtime_parameters": False,
        "runtime_sidecar_parameter_elements": 0,
        "runtime_parameter_alias_group_count": 0,
        "runtime_parameter_sharing_used": False,
        "cache_contract": {
            "cache_kind": "transformers-DynamicCache-only",
            "static_or_compileable_cache_supported": False,
            "per_layer_internal_channel_width": internal,
            "external_hidden_width": external,
            "kernel_width": kernel,
            "fresh_cache_required_per_candidate": True,
        },
    }
    descriptor = {
        **descriptor_body,
        "descriptor_sha256": canonical_sha256(descriptor_body),
    }
    receipt_body = {
        "schema": LFM2_PROTECTED_SHORT_CONV_CANDIDATE_SCHEMA,
        "status": "NON-FINAL-FOLD-ONLY-PROTECTED-SHORT-CONV-DIAGNOSTIC",
        "variant": variant,
        "topology": dict(topology),
        "input_bundle_sha256": authentication["input_bundle_sha256"],
        "source_tensor_sha256": {
            name: row["sha256"]
            for name, row in authentication["tensor_records"].items()
        },
        "residual_roundtrip_lineage": dict(residual_roundtrip_lineage),
        "product_lineage": {name: dict(product_lineage[name]) for name in _STATE_KEYS},
        "candidate_state_sha256": state_sha,
        "reconstruction_descriptor_sha256": descriptor["descriptor_sha256"],
        "physical_parameter_accounting": accounting,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        "interpretation": {
            "parent_channelwise_multiplicative_paths_protected": True,
            "parent_depthwise_kernel_copied_exactly": True,
            "dense_internal_channel_merging_performed": False,
            "bridge_modules_eliminated_by_linear_seam_absorption": True,
            "standalone_exact4x_child_claimed": False,
            "numeric_metrics_have_semantic_decision_authority": False,
            "whole_decoded_behavior_requires_manual_review": True,
            "does_not_modify_main_exact4x_compiler": True,
        },
    }
    receipt = {**receipt_body, "result_sha256": canonical_sha256(receipt_body)}
    return Lfm2ProtectedShortConvCandidate(
        variant=variant,
        tensors=MappingProxyType(normalized),
        receipt=MappingProxyType(receipt),
        reconstruction_descriptor=MappingProxyType(descriptor),
    )


def compile_lfm2_protected_short_conv_diagnostic(
    source: Lfm2ProtectedShortConvInput,
    *,
    variant: Lfm2ProtectedShortConvVariant,
    expected_input_bundle_sha256: str,
    expected_tensor_sha256: Mapping[str, str],
) -> Lfm2ProtectedShortConvCandidate:
    """Compile one protected parent-shell or standalone child-stem candidate."""

    variant = _variant(variant)
    tensors, source_topology, authentication = _authenticated_source(
        source,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_tensor_sha256=expected_tensor_sha256,
    )
    encoder = tensors["bridge.input_encoder"]
    decoder = tensors["bridge.output_decoder"]
    roundtrip, roundtrip_lineage = _sealed_fp64_product(
        encoder,
        decoder,
        left_name="bridge.input_encoder",
        right_name="bridge.output_decoder",
        product_name="residual_roundtrip.weight",
        formula="bridge.input_encoder @ bridge.output_decoder",
    )
    parent_in = tensors["parent.in_proj.weight"]
    parent_out = tensors["parent.out_proj.weight"]
    if variant in {"parent-input-roundtrip", "parent-both-roundtrip"}:
        in_proj, in_lineage = _sealed_fp64_product(
            parent_in,
            roundtrip.T.contiguous(),
            left_name="parent.in_proj.weight",
            right_name="residual_roundtrip.weight.T",
            product_name="in_proj.weight",
            formula="parent.in_proj.weight @ (input_encoder @ output_decoder).T",
        )
    elif variant == "standalone-child-stem":
        in_proj, in_lineage = _sealed_fp64_product(
            parent_in,
            decoder.T.contiguous(),
            left_name="parent.in_proj.weight",
            right_name="bridge.output_decoder.T",
            product_name="in_proj.weight",
            formula="parent.in_proj.weight @ bridge.output_decoder.T",
        )
    else:
        in_proj, in_lineage = _sealed_identity_copy(
            parent_in,
            source_name="parent.in_proj.weight",
            product_name="in_proj.weight",
        )
    if variant in {"parent-output-roundtrip", "parent-both-roundtrip"}:
        out_proj, out_lineage = _sealed_fp64_product(
            roundtrip.T.contiguous(),
            parent_out,
            left_name="residual_roundtrip.weight.T",
            right_name="parent.out_proj.weight",
            product_name="out_proj.weight",
            formula="(input_encoder @ output_decoder).T @ parent.out_proj.weight",
        )
    elif variant == "standalone-child-stem":
        out_proj, out_lineage = _sealed_fp64_product(
            encoder.T.contiguous(),
            parent_out,
            left_name="bridge.input_encoder.T",
            right_name="parent.out_proj.weight",
            product_name="out_proj.weight",
            formula="bridge.input_encoder.T @ parent.out_proj.weight",
        )
    else:
        out_proj, out_lineage = _sealed_identity_copy(
            parent_out,
            source_name="parent.out_proj.weight",
            product_name="out_proj.weight",
        )
    conv, conv_lineage = _sealed_identity_copy(
        tensors["parent.conv.weight"],
        source_name="parent.conv.weight",
        product_name="conv.weight",
    )
    return _build_candidate(
        variant=variant,
        state={
            "conv.weight": conv,
            "in_proj.weight": in_proj,
            "out_proj.weight": out_proj,
        },
        topology=_candidate_topology(source_topology, variant),
        authentication=authentication,
        residual_roundtrip_lineage=roundtrip_lineage,
        product_lineage={
            "conv.weight": conv_lineage,
            "in_proj.weight": in_lineage,
            "out_proj.weight": out_lineage,
        },
    )


def authenticate_lfm2_protected_short_conv_descriptor(
    descriptor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate a protected short-conv reconstruction descriptor."""

    if not isinstance(descriptor, Mapping):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected reconstruction descriptor must be one mapping."
        )
    value = dict(descriptor)
    supplied = _digest(value.pop("descriptor_sha256", None), label="descriptor")
    if canonical_sha256(value) != supplied:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected reconstruction descriptor seal is invalid."
        )
    topology = _validate_candidate_topology(value.get("topology"))
    cache = value.get("cache_contract")
    if (
        value.get("schema") != LFM2_PROTECTED_SHORT_CONV_DESCRIPTOR_SCHEMA
        or value.get("status") != "fold-only-protected-short-conv-descriptor"
        or value.get("family_adapter") != "lfm2-protected-parent-channel-short-conv-v1"
        or value.get("variant") != topology["variant"]
        or value.get("module_class")
        != (
            "neural_forge.hf_lfm2_protected_short_conv_diagnostic."
            "Lfm2ProtectedShortConvRuntime"
        )
        or value.get("state_keys") != list(_STATE_KEYS)
        or value.get("parameter_dtype") != "float32"
        or value.get("input_bridge_module_present") is not False
        or value.get("output_bridge_module_present") is not False
        or value.get("bridge_factors_retained_as_runtime_parameters") is not False
        or value.get("runtime_sidecar_parameter_elements") != 0
        or value.get("runtime_parameter_alias_group_count") != 0
        or value.get("runtime_parameter_sharing_used") is not False
        or not isinstance(cache, Mapping)
        or cache.get("cache_kind") != "transformers-DynamicCache-only"
        or cache.get("static_or_compileable_cache_supported") is not False
        or cache.get("per_layer_internal_channel_width")
        != topology["internal_hidden_size"]
        or cache.get("external_hidden_width") != topology["external_hidden_size"]
        or cache.get("kernel_width") != topology["kernel_size"]
        or cache.get("fresh_cache_required_per_candidate") is not True
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected reconstruction descriptor is not canonical."
        )
    _digest(value.get("candidate_state_sha256"), label="descriptor candidate state")
    return MappingProxyType({**value, "descriptor_sha256": supplied})


def _authenticate_lineage(
    raw: Any,
    *,
    label: str,
    expected_product_name: str,
    expected_tensor: Any | None = None,
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise Lfm2ProtectedShortConvDiagnosticError(f"{label} is malformed.")
    row = dict(raw)
    supplied = _digest(row.pop("lineage_sha256", None), label=f"{label} seal")
    materialized = row.get("materialized_fp32_product")
    if (
        canonical_sha256(row) != supplied
        or row.get("product_name") != expected_product_name
        or not isinstance(materialized, Mapping)
        or materialized.get("dtype") != "float32"
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(f"{label} seal is invalid.")
    if expected_tensor is not None and materialized.get(
        "sha256"
    ) != lfm2_authenticated_tensor_sha256(expected_tensor):
        raise Lfm2ProtectedShortConvDiagnosticError(
            f"{label} does not bind its materialized tensor."
        )
    return MappingProxyType({**row, "lineage_sha256": supplied})


def authenticate_lfm2_protected_short_conv_candidate(
    candidate: Lfm2ProtectedShortConvCandidate,
) -> Mapping[str, Any]:
    """Authenticate candidate state, lineage, descriptor, and dense counts."""

    if not isinstance(candidate, Lfm2ProtectedShortConvCandidate):
        raise TypeError("candidate must be Lfm2ProtectedShortConvCandidate.")
    variant = _variant(candidate.variant)
    receipt = dict(candidate.receipt)
    supplied = _digest(receipt.pop("result_sha256", None), label="candidate result")
    if (
        canonical_sha256(receipt) != supplied
        or receipt.get("schema") != LFM2_PROTECTED_SHORT_CONV_CANDIDATE_SCHEMA
        or receipt.get("status")
        != "NON-FINAL-FOLD-ONLY-PROTECTED-SHORT-CONV-DIAGNOSTIC"
        or receipt.get("variant") != variant
        or receipt.get("operation_audit") != dict(_ZERO_OPERATION_AUDIT)
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected short-conv candidate receipt is invalid."
        )
    descriptor = authenticate_lfm2_protected_short_conv_descriptor(
        candidate.reconstruction_descriptor
    )
    topology = _validate_candidate_topology(receipt.get("topology"))
    if topology["variant"] != variant:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Candidate topology names a different variant."
        )
    state = dict(candidate.tensors)
    if tuple(sorted(state)) != tuple(sorted(_STATE_KEYS)):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate state namespace is invalid."
        )
    for name, tensor in state.items():
        _validated_fp32_tensor(
            tensor,
            name=name,
            ndim=3 if name == "conv.weight" else 2,
        )
    _assert_no_storage_aliases(state, label="protected candidate state")
    state_sha = _state_sha256(state)
    _digest(receipt.get("input_bundle_sha256"), label="candidate input bundle")
    source_hashes = receipt.get("source_tensor_sha256")
    if not isinstance(source_hashes, Mapping) or tuple(sorted(source_hashes)) != tuple(
        sorted(_SOURCE_NAMES)
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate source hashes are incomplete."
        )
    for name, value in source_hashes.items():
        _digest(value, label=f"candidate source {name}")
    _authenticate_lineage(
        receipt.get("residual_roundtrip_lineage"),
        label="residual roundtrip lineage",
        expected_product_name="residual_roundtrip.weight",
    )
    product_lineage = receipt.get("product_lineage")
    if not isinstance(product_lineage, Mapping) or set(product_lineage) != set(
        _STATE_KEYS
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected product lineage does not cover every state tensor."
        )
    for name in _STATE_KEYS:
        _authenticate_lineage(
            product_lineage[name],
            label=f"{name} product lineage",
            expected_product_name=name,
            expected_tensor=state[name],
        )
    accounting = receipt.get("physical_parameter_accounting")
    if not isinstance(accounting, Mapping):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate lacks physical accounting."
        )
    expected_accounting = dict(
        lfm2_protected_short_conv_parameter_accounting(
            parent_hidden_size=int(topology["parent_hidden_size"]),
            child_hidden_size=int(topology["child_hidden_size"]),
            kernel_size=int(topology["kernel_size"]),
            variant=variant,
        )
    )
    observed = sum(int(tensor.numel()) for tensor in state.values())
    if (
        state_sha != receipt.get("candidate_state_sha256")
        or state_sha != descriptor.get("candidate_state_sha256")
        or descriptor.get("descriptor_sha256")
        != receipt.get("reconstruction_descriptor_sha256")
        or descriptor.get("topology") != topology
        or dict(accounting) != expected_accounting
        or observed != accounting["candidate_parameter_elements"]
        or _stored_tensor_elements(state)
        != accounting["candidate_stored_tensor_elements"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate state, descriptor, lineage, and counts are not bound."
        )
    return MappingProxyType({**receipt, "result_sha256": supplied})


_PROTECTED_SHORT_CONV_RUNTIME_CLASS: type[Any] | None = None


def _protected_short_conv_runtime_class() -> type[Any]:
    global _PROTECTED_SHORT_CONV_RUNTIME_CLASS
    if _PROTECTED_SHORT_CONV_RUNTIME_CLASS is not None:
        return _PROTECTED_SHORT_CONV_RUNTIME_CLASS
    torch = _torch()

    class ProtectedShortConvRuntime(torch.nn.Module):
        """LFM2 short conv with independently declared external/internal widths."""

        def __init__(
            self,
            *,
            external_hidden_size: int,
            internal_hidden_size: int,
            kernel_size: int,
            layer_index: int,
            device: Any = "cpu",
        ) -> None:
            super().__init__()
            external = _positive_int(external_hidden_size, label="external_hidden_size")
            internal = _positive_int(internal_hidden_size, label="internal_hidden_size")
            kernel = _positive_int(kernel_size, label="kernel_size")
            layer = _nonnegative_int(layer_index, label="layer_index")
            resolved_device = torch.device(device)
            if resolved_device.type == "meta":
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Protected short conv cannot be constructed on the meta device."
                )
            self.external_hidden_size = external
            self.internal_hidden_size = internal
            self.L_cache = kernel
            self.layer_idx = layer
            self.layer_type = "conv"
            self.bias = False
            self.conv = torch.nn.Conv1d(
                internal,
                internal,
                kernel,
                groups=internal,
                bias=False,
                padding=kernel - 1,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.in_proj = torch.nn.Linear(
                external,
                3 * internal,
                bias=False,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.out_proj = torch.nn.Linear(
                internal,
                external,
                bias=False,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.requires_grad_(False)
            self.eval()

        def _validate_hidden(self, hidden_states: Any) -> None:
            if (
                not isinstance(hidden_states, torch.Tensor)
                or hidden_states.layout != torch.strided
                or hidden_states.device.type == "meta"
                or hidden_states.ndim != 3
                or int(hidden_states.shape[-1]) != self.external_hidden_size
                or hidden_states.dtype != torch.float32
                or hidden_states.device != self.in_proj.weight.device
                or hidden_states.requires_grad
                or not bool(torch.isfinite(hidden_states).all().item())
            ):
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Protected short-conv hidden states must be finite detached FP32 "
                    "[batch, sequence, external_hidden] values on the module device."
                )

        def _validate_auxiliary(
            self,
            hidden_states: Any,
            attention_mask: Any,
            seq_idx: Any,
        ) -> None:
            batch, sequence = map(int, hidden_states.shape[:2])
            if attention_mask is not None and (
                not isinstance(attention_mask, torch.Tensor)
                or attention_mask.device != hidden_states.device
                or attention_mask.ndim != 2
                or tuple(attention_mask.shape) != (batch, sequence)
                or (
                    attention_mask.dtype.is_floating_point
                    and not bool(torch.isfinite(attention_mask).all().item())
                )
            ):
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Protected short-conv attention_mask must be finite "
                    "[batch, sequence] values on the hidden-state device."
                )
            if seq_idx is not None and (
                not isinstance(seq_idx, torch.Tensor)
                or seq_idx.device != hidden_states.device
                or seq_idx.dtype not in {torch.int32, torch.int64}
                or seq_idx.ndim != 2
                or tuple(seq_idx.shape) != (batch, sequence)
            ):
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Protected short-conv seq_idx must be int32/int64 "
                    "[batch, sequence] values on the hidden-state device."
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
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Transformers DynamicCache is required for cached short conv."
                ) from exc
            if not isinstance(past_key_values, DynamicCache) or bool(
                getattr(past_key_values, "is_compileable", False)
            ):
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Protected short conv rejects static, compileable, or "
                    "non-Dynamic caches."
                )
            layers = getattr(past_key_values, "layers", None)
            if not isinstance(layers, list) or self.layer_idx >= len(layers):
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Dynamic cache lacks the protected conv layer slot."
                )
            layer = layers[self.layer_idx]
            if not isinstance(layer, LinearAttentionCacheLayerMixin):
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Dynamic cache layer is not a linear-attention/conv cache."
                )
            if bool(getattr(layer, "is_conv_states_initialized", False)):
                state = getattr(layer, "conv_states", None)
                if (
                    not isinstance(state, torch.Tensor)
                    or state.ndim != 3
                    or int(state.shape[1]) != self.internal_hidden_size
                    or int(state.shape[-1]) != self.L_cache
                    or state.dtype != torch.float32
                    or state.device != self.conv.weight.device
                    or not bool(torch.isfinite(state).all().item())
                ):
                    raise Lfm2ProtectedShortConvDiagnosticError(
                        "Existing conv cache is not finite FP32 "
                        "[batch, internal_hidden, kernel] state on the module device."
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
                self.conv.weight.size(0), self.conv.weight.size(2)
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
                ).unsqueeze(-1)
            else:
                if past_key_values is not None:
                    conv_state = torch.nn.functional.pad(
                        Bx, (self.L_cache - Bx.shape[-1], 0)
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

            seqlen = int(x.shape[1])
            x = runtime.apply_mask_to_padding_states(x, attention_mask)
            BCx = self.in_proj(x).transpose(-1, -2)
            B, C, x = BCx.chunk(3, dim=-2)
            Bx = B * x
            if past_key_values is not None and past_key_values.has_previous_state(
                self.layer_idx
            ):
                conv_state = past_key_values.update_conv_state(Bx, self.layer_idx)
                conv_out = torch.sum(
                    conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1
                ).unsqueeze(-1)
            elif seq_idx is not None and x.shape[0] == 1:
                if past_key_values is not None:
                    conv_state = torch.nn.functional.pad(
                        Bx, (self.L_cache - Bx.shape[-1], 0)
                    )
                    past_key_values.update_conv_state(conv_state, self.layer_idx)
                sequence_ids = seq_idx[0]
                change = (sequence_ids[1:] != sequence_ids[:-1]).nonzero(as_tuple=True)[
                    0
                ] + 1
                bounds = torch.cat(
                    [
                        change.new_zeros(1),
                        change,
                        change.new_full((1,), sequence_ids.numel()),
                    ]
                ).tolist()
                parts = [
                    self.conv(Bx[:, :, start:stop])[..., : stop - start]
                    for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
                    if stop > start
                ]
                conv_out = torch.cat(parts, dim=-1)
            else:
                if past_key_values is not None:
                    conv_state = torch.nn.functional.pad(
                        Bx, (self.L_cache - Bx.shape[-1], 0)
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
                and hidden_states.device.type == "cuda"
                and not runtime.is_torchdynamo_compiling()
            ):
                result = self.cuda_kernels_forward(
                    hidden_states, past_key_values, attention_mask, seq_idx
                )
            else:
                result = self.slow_forward(
                    hidden_states, past_key_values, attention_mask, seq_idx
                )
            self._validate_cache(past_key_values)
            expected_shape = (
                int(hidden_states.shape[0]),
                int(hidden_states.shape[1]),
                self.external_hidden_size,
            )
            if (
                result.dtype != torch.float32
                or tuple(result.shape) != expected_shape
                or not bool(torch.isfinite(result).all().item())
            ):
                raise Lfm2ProtectedShortConvDiagnosticError(
                    "Protected short conv produced invalid external-width FP32 output."
                )
            return result

    ProtectedShortConvRuntime.__name__ = "Lfm2ProtectedShortConvRuntime"
    ProtectedShortConvRuntime.__qualname__ = "Lfm2ProtectedShortConvRuntime"
    _PROTECTED_SHORT_CONV_RUNTIME_CLASS = ProtectedShortConvRuntime
    return ProtectedShortConvRuntime


class Lfm2ProtectedShortConvRuntime:
    """Lazy public constructor for the optional-PyTorch runtime class."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return _protected_short_conv_runtime_class()(*args, **kwargs)


def reconstruct_lfm2_protected_short_conv_candidate(
    *,
    candidate: Lfm2ProtectedShortConvCandidate,
    expected_candidate_result_sha256: str,
    expected_descriptor_sha256: str,
    device: Any = "cpu",
) -> Lfm2ProtectedShortConvReconstruction:
    """Freshly construct and strictly load one protected runtime module."""

    torch = _torch()
    receipt = authenticate_lfm2_protected_short_conv_candidate(candidate)
    descriptor = authenticate_lfm2_protected_short_conv_descriptor(
        candidate.reconstruction_descriptor
    )
    if receipt["result_sha256"] != _digest(
        expected_candidate_result_sha256, label="expected candidate result"
    ) or descriptor["descriptor_sha256"] != _digest(
        expected_descriptor_sha256, label="expected descriptor"
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected candidate or descriptor differs from its external pin."
        )
    requested_device = torch.device(device)
    if requested_device.type == "meta":
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Protected reconstruction cannot target the meta device."
        )
    topology = descriptor["topology"]
    module = Lfm2ProtectedShortConvRuntime(
        external_hidden_size=int(topology["external_hidden_size"]),
        internal_hidden_size=int(topology["internal_hidden_size"]),
        kernel_size=int(topology["kernel_size"]),
        layer_index=int(topology["layer_index"]),
        device=requested_device,
    )
    state = {
        name: tensor.to(device=requested_device, dtype=torch.float32)
        for name, tensor in candidate.state_dict().items()
    }
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Fresh protected runtime rejected its complete authenticated state."
        )
    module.requires_grad_(False)
    module.eval()
    runtime_state = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in module.state_dict().items()
    }
    _assert_no_storage_aliases(runtime_state, label="reconstructed protected runtime")
    state_sha = _state_sha256(runtime_state)
    alias_groups = _module_alias_group_count(module)
    parameter_elements = sum(int(value.numel()) for value in module.parameters())
    stored_elements = _stored_tensor_elements(runtime_state)
    expected_elements = receipt["physical_parameter_accounting"][
        "candidate_parameter_elements"
    ]
    if (
        state_sha != receipt["candidate_state_sha256"]
        or parameter_elements != expected_elements
        or stored_elements != expected_elements
        or alias_groups != 0
        or any(parameter.requires_grad for parameter in module.parameters())
        or hasattr(module, "input_bridge")
        or hasattr(module, "output_bridge")
    ):
        raise Lfm2ProtectedShortConvDiagnosticError(
            "Fresh protected runtime differs from authenticated state/count."
        )
    body = {
        "schema": LFM2_PROTECTED_SHORT_CONV_RECONSTRUCTION_SCHEMA,
        "status": "protected-short-conv-freshly-reconstructed",
        "variant": candidate.variant,
        "candidate_result_sha256": receipt["result_sha256"],
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "candidate_state_sha256": receipt["candidate_state_sha256"],
        "reconstructed_state_sha256": state_sha,
        "external_hidden_size": int(topology["external_hidden_size"]),
        "internal_hidden_size": int(topology["internal_hidden_size"]),
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
    return Lfm2ProtectedShortConvReconstruction(
        module=module,
        receipt=MappingProxyType(reconstruction_receipt),
    )


__all__ = [
    "LFM2_PROTECTED_SHORT_CONV_CANDIDATE_SCHEMA",
    "LFM2_PROTECTED_SHORT_CONV_DESCRIPTOR_SCHEMA",
    "LFM2_PROTECTED_SHORT_CONV_INPUT_SCHEMA",
    "LFM2_PROTECTED_SHORT_CONV_RECONSTRUCTION_SCHEMA",
    "Lfm2ProtectedShortConvCandidate",
    "Lfm2ProtectedShortConvDiagnosticError",
    "Lfm2ProtectedShortConvInput",
    "Lfm2ProtectedShortConvReconstruction",
    "Lfm2ProtectedShortConvRuntime",
    "Lfm2ProtectedShortConvVariant",
    "authenticate_lfm2_protected_short_conv_candidate",
    "authenticate_lfm2_protected_short_conv_descriptor",
    "compile_lfm2_protected_short_conv_diagnostic",
    "fingerprint_lfm2_protected_short_conv_input",
    "lfm2_protected_short_conv_parameter_accounting",
    "reconstruct_lfm2_protected_short_conv_candidate",
]
