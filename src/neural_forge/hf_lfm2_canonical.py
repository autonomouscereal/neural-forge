"""Fail-closed LFM2 short-convolution inspection and channel transplant plans.

Transformers LFM2 short-convolution blocks are not attention blocks.  Their
``in_proj`` output is ordered ``B, C, x`` and implements a gated depthwise FIR:

``r_t = B_t * x_t``
``v_t = sum_lag k_lag * r_(t-lag)``
``delta_t = W_o(C_t * v_t)``

The helpers below recognize the concrete Transformers LFM2 module topology,
keep every occurrence of a channel coupled during surgery, and refuse layouts
whose semantics cannot be established structurally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"
_LFM2_DYNAMIC_MODULE = "neural_forge.hf_lfm2_dynamic_blank_materializer"
_BCX_ORDER = ("b", "c", "x")
_SCHEMA = "neural-forge-lfm2-layer-contract-v1"
_HETEROGENEOUS_SCHEMA = "neural-forge-lfm2-heterogeneous-layer-contract-v1"
_PLAN_SCHEMA = "neural-forge-lfm2-short-conv-channel-plan-v1"


class Lfm2CanonicalError(ValueError):
    """Base error for unsupported or inconsistent LFM2 canonical surgery."""


class Lfm2UnsupportedLayoutError(Lfm2CanonicalError):
    """Raised when an object cannot be proven to use Transformers LFM2 semantics."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for LFM2 canonical surgery.") from exc
    return torch


@dataclass(frozen=True, slots=True)
class Lfm2AttentionContract:
    """Serializable shape contract for a Transformers LFM2 attention operator."""

    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    query_projection_shape: tuple[int, int]
    key_projection_shape: tuple[int, int]
    value_projection_shape: tuple[int, int]
    output_projection_shape: tuple[int, int]
    qk_rmsnorm: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "full_attention",
            "hidden_size": self.hidden_size,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "projection_roles": ["q", "k", "v"],
            "query_projection_shape": list(self.query_projection_shape),
            "key_projection_shape": list(self.key_projection_shape),
            "value_projection_shape": list(self.value_projection_shape),
            "output_projection_shape": list(self.output_projection_shape),
            "qk_rmsnorm": self.qk_rmsnorm,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ShortConvContract:
    """Serializable contract for the gated depthwise FIR in LFM2."""

    hidden_size: int
    fir_length: int
    bias: bool
    input_projection_shape: tuple[int, int]
    depthwise_kernel_shape: tuple[int, int, int]
    output_projection_shape: tuple[int, int]
    groups: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "short_conv",
            "hidden_size": self.hidden_size,
            "fir_length": self.fir_length,
            "bias": self.bias,
            "input_projection_shape": list(self.input_projection_shape),
            "input_projection_chunk_order": list(_BCX_ORDER),
            "depthwise_kernel_shape": list(self.depthwise_kernel_shape),
            "output_projection_shape": list(self.output_projection_shape),
            "groups": self.groups,
            "semantics": {
                "project": "(B_t, C_t, x_t) = chunk_3(in_proj(h_t))",
                "gate": "r_t = B_t * x_t",
                "fir": "v_t = conv_bias + sum_lag k_lag * r_(t-lag)",
                "output": "delta_t = out_proj(C_t * v_t)",
                "stored_weight_to_lag": "k_lag = conv.weight[:, 0, L-1-lag]",
            },
            "not_attention_qkv": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousShortConvContract:
    """Exact external/internal geometry of one materialized ShortConv."""

    external_hidden_size: int
    internal_channel_width: int
    fir_length: int
    bias: bool
    input_projection_shape: tuple[int, int]
    depthwise_kernel_shape: tuple[int, int, int]
    output_projection_shapes: tuple[tuple[int, int], ...]
    writer_topology: Literal["dense", "factorized"]
    writer_rank: int | None
    groups: int

    @property
    def hidden_size(self) -> int:
        """Expose the constant residual-stream width for generic consumers."""

        return self.external_hidden_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "short_conv",
            "external_hidden_size": self.external_hidden_size,
            "internal_channel_width": self.internal_channel_width,
            "fir_length": self.fir_length,
            "bias": self.bias,
            "input_projection_shape": list(self.input_projection_shape),
            "input_projection_chunk_order": list(_BCX_ORDER),
            "depthwise_kernel_shape": list(self.depthwise_kernel_shape),
            "output_projection_shapes": [
                list(shape) for shape in self.output_projection_shapes
            ],
            "writer_topology": self.writer_topology,
            "writer_rank": self.writer_rank,
            "groups": self.groups,
            "semantics": {
                "project": "(B_t, C_t, x_t) = chunk_3(in_proj(h_t))",
                "gate": "r_t = B_t * x_t",
                "fir": "v_t = conv_bias + sum_lag k_lag * r_(t-lag)",
                "output": "delta_t = out_proj(C_t * v_t)",
                "stored_weight_to_lag": "k_lag = conv.weight[:, 0, L-1-lag]",
            },
            "not_attention_qkv": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LayerContract:
    """Validated, JSON-serializable contract for one LFM2 decoder layer."""

    layer_index: int
    kind: Literal["full_attention", "short_conv"]
    hidden_size: int
    ffn_intermediate_size: int
    operator: Lfm2AttentionContract | Lfm2ShortConvContract

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "implementation": _LFM2_MODULE,
            "model_type": "lfm2",
            "layer_index": self.layer_index,
            "kind": self.kind,
            "hidden_size": self.hidden_size,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "operator": self.operator.as_dict(),
            "structurally_validated": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousLayerContract:
    """Authenticated contract for one sealed heterogeneous dense layer."""

    layer_index: int
    kind: Literal["full_attention", "short_conv"]
    hidden_size: int
    ffn_intermediate_size: int
    selection_evidence_sha256: str
    physical_layout_descriptor_sha256: str
    operator: Lfm2AttentionContract | Lfm2HeterogeneousShortConvContract

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _HETEROGENEOUS_SCHEMA,
            "implementation": _LFM2_DYNAMIC_MODULE,
            "model_type": "lfm2",
            "layer_index": self.layer_index,
            "kind": self.kind,
            "hidden_size": self.hidden_size,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "selection_evidence_sha256": self.selection_evidence_sha256,
            "physical_layout_descriptor_sha256": (
                self.physical_layout_descriptor_sha256
            ),
            "operator": self.operator.as_dict(),
            "structurally_validated": True,
            "sealed_heterogeneous_physical_layout": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ShortConvParameters:
    """Tensor bundle using the exact Transformers LFM2 short-conv layout."""

    in_proj_weight: Any
    in_proj_bias: Any | None
    conv_weight: Any
    conv_bias: Any | None
    out_proj_weight: Any
    out_proj_bias: Any | None

    @property
    def hidden_size(self) -> int:
        return int(self.out_proj_weight.shape[0])

    @property
    def fir_length(self) -> int:
        return int(self.conv_weight.shape[2])

    @property
    def bias(self) -> bool:
        return self.in_proj_bias is not None

    def as_state_dict(self) -> dict[str, Any]:
        result = {
            "in_proj.weight": self.in_proj_weight,
            "conv.weight": self.conv_weight,
            "out_proj.weight": self.out_proj_weight,
        }
        if self.in_proj_bias is not None:
            result["in_proj.bias"] = self.in_proj_bias
            result["conv.bias"] = self.conv_bias
            result["out_proj.bias"] = self.out_proj_bias
        return result


@dataclass(frozen=True, slots=True)
class Lfm2TransplantExactness:
    """Honest algebraic classification for a channel transplant."""

    level: Literal["algebraically_exact", "approximate"]
    scope: str
    reason: str
    full_coordinate_permutation: bool
    selected_input_subnetwork_closed: bool
    selected_output_subnetwork_closed: bool
    channels_merged: bool
    topology_preserved: bool

    @property
    def algebraically_exact(self) -> bool:
        return self.level == "algebraically_exact"

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "algebraically_exact": self.algebraically_exact,
            "scope": self.scope,
            "reason": self.reason,
            "proof_obligations": {
                "full_coordinate_permutation": self.full_coordinate_permutation,
                "selected_input_subnetwork_closed": (
                    self.selected_input_subnetwork_closed
                ),
                "selected_output_subnetwork_closed": (
                    self.selected_output_subnetwork_closed
                ),
                "channels_merged": self.channels_merged,
                "topology_preserved": self.topology_preserved,
            },
        }


@dataclass(frozen=True, slots=True)
class Lfm2ChannelPlanInvariants:
    selected_channels_unique: bool
    selected_channels_in_range: bool
    target_groups_nonempty: bool
    medoids_self_assigned: bool
    bcx_kernel_output_indices_coupled: bool
    every_source_channel_accounted_for: bool
    producer_chunk_order_bcx: bool
    zero_gradient: bool

    @property
    def all_hold(self) -> bool:
        return all(
            (
                self.selected_channels_unique,
                self.selected_channels_in_range,
                self.target_groups_nonempty,
                self.medoids_self_assigned,
                self.bcx_kernel_output_indices_coupled,
                self.every_source_channel_accounted_for,
                self.producer_chunk_order_bcx,
                self.zero_gradient,
            )
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "selected_channels_unique": self.selected_channels_unique,
            "selected_channels_in_range": self.selected_channels_in_range,
            "target_groups_nonempty": self.target_groups_nonempty,
            "medoids_self_assigned": self.medoids_self_assigned,
            "bcx_kernel_output_indices_coupled": (
                self.bcx_kernel_output_indices_coupled
            ),
            "every_source_channel_accounted_for": (
                self.every_source_channel_accounted_for
            ),
            "producer_chunk_order_bcx": self.producer_chunk_order_bcx,
            "zero_gradient": self.zero_gradient,
            "all_hold": self.all_hold,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ShortConvChannelPlan:
    """Deterministic hard channel map shared by every short-conv tensor role."""

    source_contract: Lfm2ShortConvContract
    target_hidden_size: int
    mode: Literal["subset", "merge"]
    selected_source_channels: tuple[int, ...]
    source_to_target: tuple[int, ...]
    target_groups: tuple[tuple[int, ...], ...]
    selection_strategy: str
    exactness: Lfm2TransplantExactness
    invariants: Lfm2ChannelPlanInvariants
    source_parameters_sha256: str
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _PLAN_SCHEMA,
            "method": "deterministic-coupled-lfm2-short-conv-channel-transplant",
            "source_contract": self.source_contract.as_dict(),
            "source_hidden_size": self.source_contract.hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "mode": self.mode,
            "selected_source_channels": list(self.selected_source_channels),
            "source_to_target": list(self.source_to_target),
            "target_groups": [list(group) for group in self.target_groups],
            "selection_strategy": self.selection_strategy,
            "coupled_tensor_roles": [
                "hidden_coordinate",
                "in_proj.B.row",
                "in_proj.C.row",
                "in_proj.x.row",
                "in_proj.B/C/x.column",
                "conv.kernel.row",
                "out_proj.row",
                "out_proj.column",
            ],
            "projection_chunk_order": list(_BCX_ORDER),
            "not_attention_qkv": True,
            "exactness": self.exactness.as_dict(),
            "invariants": self.invariants.as_dict(),
            "source_parameters_sha256": self.source_parameters_sha256,
            "fingerprint": self.fingerprint,
            "training_steps": 0,
            "gradients_used": False,
        }

    def transplant(
        self,
        source: Lfm2ShortConvParameters | Any,
    ) -> Lfm2ShortConvParameters:
        """Create target tensors without mutating the validated source module."""

        parameters = _coerce_short_conv_parameters(source)
        _validate_parameter_bundle(parameters)
        if _parameters_sha256(parameters) != self.source_parameters_sha256:
            raise Lfm2CanonicalError(
                "The source tensors differ from the tensors used to build this plan."
            )
        if self.mode == "subset":
            return _subset_parameters(parameters, self.selected_source_channels)
        return _merge_parameters(parameters, self.source_to_target, self.target_hidden_size)


def inspect_lfm2_layer(layer: Any) -> Lfm2LayerContract:
    """Inspect one concrete Transformers LFM2 decoder layer, failing closed."""

    _require_supported_class(layer, "Lfm2DecoderLayer")
    is_attention = getattr(layer, "is_attention_layer", None)
    if not isinstance(is_attention, bool):
        raise Lfm2UnsupportedLayoutError(
            "Lfm2DecoderLayer.is_attention_layer must be an explicit bool."
        )
    has_attention = hasattr(layer, "self_attn")
    has_conv = hasattr(layer, "conv")
    if has_attention == has_conv or has_attention != is_attention:
        raise Lfm2UnsupportedLayoutError(
            "LFM2 operator attributes do not match is_attention_layer."
        )

    operator = layer.self_attn if is_attention else layer.conv
    expected_class = "Lfm2Attention" if is_attention else "Lfm2ShortConv"
    _require_supported_class(operator, expected_class)
    config = getattr(operator, "config", None)
    if getattr(config, "model_type", None) != "lfm2":
        raise Lfm2UnsupportedLayoutError("Operator config.model_type must be lfm2.")
    layer_index = getattr(operator, "layer_idx", None)
    if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
        raise Lfm2UnsupportedLayoutError("Operator layer_idx must be non-negative.")
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, Sequence) or layer_index >= len(layer_types):
        raise Lfm2UnsupportedLayoutError(
            "LFM2 config.layer_types does not contain this layer index."
        )
    expected_type = "full_attention" if is_attention else "conv"
    if layer_types[layer_index] != expected_type:
        raise Lfm2UnsupportedLayoutError(
            "LFM2 config.layer_types disagrees with the instantiated operator."
        )
    hidden_size = _positive_int(getattr(config, "hidden_size", None), "hidden_size")
    ffn_intermediate = _validate_layer_shell(layer, hidden_size)
    if is_attention:
        operator_contract: Lfm2AttentionContract | Lfm2ShortConvContract
        operator_contract = _inspect_attention(operator, hidden_size)
        kind: Literal["full_attention", "short_conv"] = "full_attention"
    else:
        operator_contract = _inspect_short_conv(operator, hidden_size)
        kind = "short_conv"
    return Lfm2LayerContract(
        layer_index=layer_index,
        kind=kind,
        hidden_size=hidden_size,
        ffn_intermediate_size=ffn_intermediate,
        operator=operator_contract,
    )


def inspect_lfm2_heterogeneous_layer(
    layer: Any,
    descriptor: Any,
) -> Lfm2HeterogeneousLayerContract:
    """Inspect one already-authenticated heterogeneous dense LFM2 layer.

    This is intentionally a separate path from :func:`inspect_lfm2_layer`.
    The native inspector above continues to require the exact Transformers
    classes and uniform native tensor geometry.  This path accepts only the
    exact dynamic runtime classes and a canonical sealed physical descriptor;
    it never infers an unsealed geometry from live tensor shapes.
    """

    from .hf_lfm2_dynamic_physical_layout import (
        Lfm2HeterogeneousPhysicalLayoutDescriptor,
        restore_lfm2_heterogeneous_physical_layout,
    )

    layout = restore_lfm2_heterogeneous_physical_layout(descriptor)
    if type(layout) is not Lfm2HeterogeneousPhysicalLayoutDescriptor:
        raise Lfm2UnsupportedLayoutError(
            "A heterogeneous layer requires the exact sealed descriptor type."
        )
    _require_supported_class(layer, "Lfm2DecoderLayer")
    is_attention = getattr(layer, "is_attention_layer", None)
    if not isinstance(is_attention, bool):
        raise Lfm2UnsupportedLayoutError(
            "Lfm2DecoderLayer.is_attention_layer must be an explicit bool."
        )
    has_attention = hasattr(layer, "self_attn")
    has_conv = hasattr(layer, "conv")
    if has_attention == has_conv or has_attention != is_attention:
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous LFM2 operator attributes disagree with the layer tag."
        )

    operator = layer.self_attn if is_attention else layer.conv
    expected_class = (
        "Lfm2DynamicAttention"
        if is_attention
        else "Lfm2DynamicWidthShortConv"
    )
    _require_dynamic_supported_class(operator, expected_class)
    config = getattr(operator, "config", None)
    if getattr(config, "model_type", None) != "lfm2":
        raise Lfm2UnsupportedLayoutError("Operator config.model_type must be lfm2.")
    if (
        getattr(config, "neural_forge_dynamic_physical_layout_sha256", None)
        != layout.descriptor_sha256
    ):
        raise Lfm2UnsupportedLayoutError(
            "Operator config is not bound to the supplied heterogeneous layout."
        )
    layer_index = getattr(operator, "layer_idx", None)
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < layout.layer_count
    ):
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous operator layer_idx is outside the sealed layout."
        )
    physical = layout.layers[layer_index]
    expected_type = "full_attention" if is_attention else "conv"
    if physical.layer_type != expected_type:
        raise Lfm2UnsupportedLayoutError(
            "Live heterogeneous operator disagrees with its sealed layer type."
        )
    configured_types = getattr(config, "layer_types", None)
    if (
        not isinstance(configured_types, Sequence)
        or tuple(configured_types)
        != tuple(value.layer_type for value in layout.layers)
    ):
        raise Lfm2UnsupportedLayoutError(
            "LFM2 config.layer_types disagrees with the sealed physical layout."
        )
    hidden_size = _positive_int(getattr(config, "hidden_size", None), "hidden_size")
    if hidden_size != layout.hidden_size:
        raise Lfm2UnsupportedLayoutError(
            "LFM2 residual width disagrees with the sealed physical layout."
        )
    ffn_intermediate = _validate_heterogeneous_layer_shell(
        layer,
        hidden_size=hidden_size,
        expected_intermediate=physical.mlp_intermediate_size,
    )
    if is_attention:
        operator_contract: (
            Lfm2AttentionContract | Lfm2HeterogeneousShortConvContract
        ) = _inspect_heterogeneous_attention(
            operator,
            hidden_size=hidden_size,
            physical=physical,
            rope_theta=layout.rope_theta,
            layer_count=layout.layer_count,
        )
        kind: Literal["full_attention", "short_conv"] = "full_attention"
    else:
        operator_contract = _inspect_heterogeneous_short_conv(
            operator,
            hidden_size=hidden_size,
            physical=physical,
            fir_length=layout.conv_kernel_size,
            layer_count=layout.layer_count,
        )
        kind = "short_conv"
    return Lfm2HeterogeneousLayerContract(
        layer_index=layer_index,
        kind=kind,
        hidden_size=hidden_size,
        ffn_intermediate_size=ffn_intermediate,
        selection_evidence_sha256=physical.selection_evidence_sha256,
        physical_layout_descriptor_sha256=layout.descriptor_sha256,
        operator=operator_contract,
    )


def extract_lfm2_short_conv_parameters(layer: Any) -> Lfm2ShortConvParameters:
    """Extract tensors only after proving the layer is an LFM2 short-conv block."""

    contract = inspect_lfm2_layer(layer)
    if contract.kind != "short_conv":
        raise Lfm2UnsupportedLayoutError(
            "Cannot extract short-conv tensors from an LFM2 attention layer."
        )
    operator = layer.conv
    result = Lfm2ShortConvParameters(
        in_proj_weight=operator.in_proj.weight,
        in_proj_bias=operator.in_proj.bias,
        conv_weight=operator.conv.weight,
        conv_bias=operator.conv.bias,
        out_proj_weight=operator.out_proj.weight,
        out_proj_bias=operator.out_proj.bias,
    )
    _validate_parameter_bundle(result)
    return result


def build_lfm2_short_conv_channel_plan(
    source_layer: Any,
    *,
    target_hidden_size: int,
    mode: Literal["subset", "merge"] = "subset",
    selected_channels: Sequence[int] | None = None,
) -> Lfm2ShortConvChannelPlan:
    """Build a deterministic, zero-gradient coupled channel surgery plan."""

    if mode not in {"subset", "merge"}:
        raise Lfm2CanonicalError("mode must be subset or merge.")
    layer_contract = inspect_lfm2_layer(source_layer)
    if layer_contract.kind != "short_conv":
        raise Lfm2UnsupportedLayoutError(
            "A short-conv channel plan requires an LFM2 short-conv layer."
        )
    contract = layer_contract.operator
    if not isinstance(contract, Lfm2ShortConvContract):  # pragma: no cover
        raise Lfm2UnsupportedLayoutError("Internal short-conv contract mismatch.")
    source = extract_lfm2_short_conv_parameters(source_layer)
    source_hidden = contract.hidden_size
    target_hidden = _positive_int(target_hidden_size, "target_hidden_size")
    if target_hidden > source_hidden:
        raise Lfm2CanonicalError(
            "This transplant primitive cannot invent additional FIR channels."
        )

    if selected_channels is None:
        selected = _select_channels(source, target_hidden)
        selection_strategy = "descending-coupled-energy-stable-source-index"
    else:
        selected = _validate_selected_channels(
            selected_channels,
            source_hidden,
            target_hidden,
        )
        selection_strategy = "explicit-validated-target-to-source-coordinate-map"

    if mode == "subset":
        source_to_target = [-1] * source_hidden
        for target, source_index in enumerate(selected):
            source_to_target[source_index] = target
        groups = tuple((source_index,) for source_index in selected)
    else:
        source_to_target = list(_assign_channels(source, selected))
        groups = tuple(
            tuple(
                source_index
                for source_index, target in enumerate(source_to_target)
                if target == target_index
            )
            for target_index in range(target_hidden)
        )
    exactness = _classify_exactness(source, mode, selected, tuple(groups))
    invariants = Lfm2ChannelPlanInvariants(
        selected_channels_unique=len(set(selected)) == len(selected),
        selected_channels_in_range=all(0 <= index < source_hidden for index in selected),
        target_groups_nonempty=all(bool(group) for group in groups),
        medoids_self_assigned=all(
            source_to_target[source_index] == target
            for target, source_index in enumerate(selected)
        ),
        bcx_kernel_output_indices_coupled=True,
        every_source_channel_accounted_for=(
            all(target == -1 or 0 <= target < target_hidden for target in source_to_target)
            if mode == "subset"
            else all(0 <= target < target_hidden for target in source_to_target)
        ),
        producer_chunk_order_bcx=True,
        zero_gradient=True,
    )
    if not invariants.all_hold:
        raise Lfm2CanonicalError("The coupled channel plan failed its invariants.")
    source_hash = _parameters_sha256(source)
    fingerprint_payload = {
        "schema": _PLAN_SCHEMA,
        "source_contract": contract.as_dict(),
        "target_hidden_size": target_hidden,
        "mode": mode,
        "selected_source_channels": list(selected),
        "source_to_target": source_to_target,
        "target_groups": [list(group) for group in groups],
        "selection_strategy": selection_strategy,
        "exactness": exactness.as_dict(),
        "source_parameters_sha256": source_hash,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return Lfm2ShortConvChannelPlan(
        source_contract=contract,
        target_hidden_size=target_hidden,
        mode=mode,
        selected_source_channels=selected,
        source_to_target=tuple(source_to_target),
        target_groups=tuple(groups),
        selection_strategy=selection_strategy,
        exactness=exactness,
        invariants=invariants,
        source_parameters_sha256=source_hash,
        fingerprint=fingerprint,
    )


def install_lfm2_short_conv_parameters(
    target_layer: Any,
    parameters: Lfm2ShortConvParameters,
) -> None:
    """Install a transplant into a structurally validated target layer."""

    torch = _require_torch()
    contract = inspect_lfm2_layer(target_layer)
    if contract.kind != "short_conv":
        raise Lfm2UnsupportedLayoutError("The target layer is not short-conv.")
    operator_contract = contract.operator
    if not isinstance(operator_contract, Lfm2ShortConvContract):  # pragma: no cover
        raise Lfm2UnsupportedLayoutError("Internal short-conv contract mismatch.")
    _validate_parameter_bundle(parameters)
    if parameters.hidden_size != operator_contract.hidden_size:
        raise Lfm2CanonicalError("Target hidden size does not match the transplant.")
    if parameters.fir_length != operator_contract.fir_length:
        raise Lfm2CanonicalError("Target FIR length does not match the transplant.")
    if parameters.bias != operator_contract.bias:
        raise Lfm2CanonicalError("Target bias topology does not match the transplant.")
    target = target_layer.conv
    state = parameters.as_state_dict()
    expected = target.state_dict()
    if set(state) != set(expected):
        raise Lfm2CanonicalError("Target state-dict topology does not match the transplant.")
    with torch.no_grad():
        for name, tensor in state.items():
            expected[name].copy_(tensor.to(device=expected[name].device, dtype=expected[name].dtype))


def lfm2_short_conv_reference(
    parameters: Lfm2ShortConvParameters,
    hidden_states: Any,
) -> Any:
    """Evaluate the validated slow-path FIR semantics without a cache."""

    torch = _require_torch()
    _validate_parameter_bundle(parameters)
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise Lfm2CanonicalError("hidden_states must have shape [batch, sequence, hidden].")
    if int(hidden_states.shape[2]) != parameters.hidden_size:
        raise Lfm2CanonicalError("hidden_states width does not match the parameter bundle.")
    if not hidden_states.is_floating_point() or not _is_finite(hidden_states):
        raise Lfm2CanonicalError("hidden_states must be finite floating-point values.")
    projected = hidden_states.matmul(parameters.in_proj_weight.transpose(0, 1))
    if parameters.in_proj_bias is not None:
        projected = projected + parameters.in_proj_bias
    b_values, c_values, x_values = projected.chunk(3, dim=-1)
    gated = b_values * x_values
    filtered = torch.zeros_like(gated)
    length = parameters.fir_length
    stored = parameters.conv_weight[:, 0, :]
    for lag in range(length):
        lag_weight = stored[:, length - 1 - lag]
        if lag == 0:
            filtered = filtered + gated * lag_weight
        elif lag < int(gated.shape[1]):
            filtered[:, lag:, :] = (
                filtered[:, lag:, :] + gated[:, :-lag, :] * lag_weight
            )
    if parameters.conv_bias is not None:
        filtered = filtered + parameters.conv_bias
    output = (c_values * filtered).matmul(parameters.out_proj_weight.transpose(0, 1))
    if parameters.out_proj_bias is not None:
        output = output + parameters.out_proj_bias
    return output


def _inspect_attention(operator: Any, hidden_size: int) -> Lfm2AttentionContract:
    torch = _require_torch()
    config = operator.config
    query_heads = _positive_int(
        getattr(config, "num_attention_heads", None), "num_attention_heads"
    )
    key_value_heads = _positive_int(
        getattr(config, "num_key_value_heads", None), "num_key_value_heads"
    )
    head_dim = _positive_int(getattr(operator, "head_dim", None), "head_dim")
    if query_heads % key_value_heads:
        raise Lfm2UnsupportedLayoutError(
            "LFM2 query heads must be divisible by key/value heads."
        )
    q_shape = _linear_shape(operator, "q_proj", bias=False, torch=torch)
    k_shape = _linear_shape(operator, "k_proj", bias=False, torch=torch)
    v_shape = _linear_shape(operator, "v_proj", bias=False, torch=torch)
    out_shape = _linear_shape(operator, "out_proj", bias=False, torch=torch)
    expected = (
        (query_heads * head_dim, hidden_size),
        (key_value_heads * head_dim, hidden_size),
        (key_value_heads * head_dim, hidden_size),
        (hidden_size, query_heads * head_dim),
    )
    if (q_shape, k_shape, v_shape, out_shape) != expected:
        raise Lfm2UnsupportedLayoutError(
            "LFM2 attention projection shapes do not match its head topology."
        )
    for name in ("q_layernorm", "k_layernorm"):
        norm = getattr(operator, name, None)
        _require_supported_class(norm, "Lfm2RMSNorm")
        _require_vector(getattr(norm, "weight", None), head_dim, f"{name}.weight")
    return Lfm2AttentionContract(
        hidden_size=hidden_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        query_projection_shape=q_shape,
        key_projection_shape=k_shape,
        value_projection_shape=v_shape,
        output_projection_shape=out_shape,
        qk_rmsnorm=True,
    )


def _inspect_short_conv(operator: Any, hidden_size: int) -> Lfm2ShortConvContract:
    torch = _require_torch()
    config = operator.config
    configured_length = _positive_int(
        getattr(config, "conv_L_cache", None), "conv_L_cache"
    )
    if getattr(operator, "L_cache", None) != configured_length:
        raise Lfm2UnsupportedLayoutError("Lfm2ShortConv.L_cache disagrees with config.")
    configured_bias = getattr(config, "conv_bias", None)
    if not isinstance(configured_bias, bool) or getattr(operator, "bias", None) is not configured_bias:
        raise Lfm2UnsupportedLayoutError("LFM2 short-conv bias metadata is inconsistent.")
    in_shape = _linear_shape(operator, "in_proj", bias=configured_bias, torch=torch)
    out_shape = _linear_shape(operator, "out_proj", bias=configured_bias, torch=torch)
    if in_shape != (3 * hidden_size, hidden_size):
        raise Lfm2UnsupportedLayoutError("LFM2 in_proj must have shape [3H, H].")
    if out_shape != (hidden_size, hidden_size):
        raise Lfm2UnsupportedLayoutError("LFM2 out_proj must have shape [H, H].")
    convolution = getattr(operator, "conv", None)
    weight = getattr(convolution, "weight", None)
    _require_tensor(weight, "conv.weight", torch)
    kernel_shape = tuple(int(value) for value in weight.shape)
    if kernel_shape != (hidden_size, 1, configured_length):
        raise Lfm2UnsupportedLayoutError(
            "LFM2 short convolution weight must have shape [H, 1, L]."
        )
    if (
        getattr(convolution, "in_channels", None) != hidden_size
        or getattr(convolution, "out_channels", None) != hidden_size
        or getattr(convolution, "groups", None) != hidden_size
        or tuple(getattr(convolution, "kernel_size", ())) != (configured_length,)
        or tuple(getattr(convolution, "stride", ())) != (1,)
        or tuple(getattr(convolution, "dilation", ())) != (1,)
        or tuple(getattr(convolution, "padding", ())) != (configured_length - 1,)
        or getattr(convolution, "padding_mode", None) != "zeros"
    ):
        raise Lfm2UnsupportedLayoutError(
            "LFM2 short convolution must be a causal depthwise Conv1d with padding L-1."
        )
    _validate_bias(getattr(convolution, "bias", None), configured_bias, hidden_size, "conv.bias")
    return Lfm2ShortConvContract(
        hidden_size=hidden_size,
        fir_length=configured_length,
        bias=configured_bias,
        input_projection_shape=in_shape,
        depthwise_kernel_shape=kernel_shape,
        output_projection_shape=out_shape,
        groups=hidden_size,
    )


def _inspect_heterogeneous_attention(
    operator: Any,
    *,
    hidden_size: int,
    physical: Any,
    rope_theta: float,
    layer_count: int,
) -> Lfm2AttentionContract:
    torch = _require_torch()
    query_heads = _positive_int(physical.query_heads, "query_heads")
    key_value_heads = _positive_int(
        physical.key_value_heads,
        "key_value_heads",
    )
    head_dim = _positive_int(physical.head_dim, "head_dim")
    if query_heads % key_value_heads:
        raise Lfm2UnsupportedLayoutError(
            "LFM2 query heads must be divisible by key/value heads."
        )
    if (
        getattr(operator, "query_heads", None) != query_heads
        or getattr(operator, "key_value_heads", None) != key_value_heads
        or getattr(operator, "head_dim", None) != head_dim
        or getattr(operator, "num_key_value_groups", None)
        != query_heads // key_value_heads
        or getattr(operator, "expected_layer_count", None) != layer_count
        or getattr(operator, "is_causal", None) is not True
        or type(getattr(operator, "default_rope_theta", None)) is not float
        or getattr(operator, "default_rope_theta", None) != rope_theta
    ):
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous attention metadata disagrees with its sealed geometry."
        )
    scaling = getattr(operator, "scaling", None)
    if type(scaling) not in {int, float} or float(scaling) != head_dim**-0.5:
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous attention scaling disagrees with head_dim."
        )
    q_shape = _linear_shape(operator, "q_proj", bias=False, torch=torch)
    k_shape = _linear_shape(operator, "k_proj", bias=False, torch=torch)
    v_shape = _linear_shape(operator, "v_proj", bias=False, torch=torch)
    out_shape = _linear_shape(operator, "out_proj", bias=False, torch=torch)
    expected = (
        (query_heads * head_dim, hidden_size),
        (key_value_heads * head_dim, hidden_size),
        (key_value_heads * head_dim, hidden_size),
        (hidden_size, query_heads * head_dim),
    )
    if (q_shape, k_shape, v_shape, out_shape) != expected:
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous attention projections disagree with sealed geometry."
        )
    for name in ("q_layernorm", "k_layernorm"):
        norm = getattr(operator, name, None)
        _require_supported_class(norm, "Lfm2RMSNorm")
        _require_vector(getattr(norm, "weight", None), head_dim, f"{name}.weight")
    return Lfm2AttentionContract(
        hidden_size=hidden_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        query_projection_shape=q_shape,
        key_projection_shape=k_shape,
        value_projection_shape=v_shape,
        output_projection_shape=out_shape,
        qk_rmsnorm=True,
    )


def _inspect_heterogeneous_short_conv(
    operator: Any,
    *,
    hidden_size: int,
    physical: Any,
    fir_length: int,
    layer_count: int,
) -> Lfm2HeterogeneousShortConvContract:
    torch = _require_torch()
    internal = _positive_int(
        physical.short_conv_internal_width,
        "short_conv_internal_width",
    )
    rank = physical.short_conv_writer_rank
    if rank is not None:
        rank = _positive_int(rank, "short_conv_writer_rank")
    kernel_values = tuple(
        getattr(operator, name)
        for name in ("L_cache", "conv_kernel_size")
        if hasattr(operator, name)
    )
    if (
        getattr(operator, "bias", None) is not False
        or getattr(operator, "layer_type", None) != "conv"
        or getattr(operator, "internal_channel_width", None) != internal
        or getattr(operator, "external_hidden_size", None) != hidden_size
        or getattr(operator, "writer_rank", None) != rank
        or getattr(operator, "expected_layer_count", None) != layer_count
        or not kernel_values
        or any(value != fir_length for value in kernel_values)
    ):
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous ShortConv metadata disagrees with sealed geometry."
        )
    in_shape = _linear_shape(operator, "in_proj", bias=False, torch=torch)
    if in_shape != (3 * internal, hidden_size):
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous ShortConv input projection must have shape [3C, H]."
        )
    convolution = getattr(operator, "conv", None)
    weight = getattr(convolution, "weight", None)
    _require_tensor(weight, "conv.weight", torch)
    kernel_shape = tuple(int(value) for value in weight.shape)
    if kernel_shape != (internal, 1, fir_length):
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous ShortConv FIR must have shape [C, 1, L]."
        )
    if (
        type(convolution) is not torch.nn.Conv1d
        or convolution.in_channels != internal
        or convolution.out_channels != internal
        or convolution.groups != internal
        or tuple(convolution.kernel_size) != (fir_length,)
        or tuple(convolution.stride) != (1,)
        or tuple(convolution.dilation) != (1,)
        or tuple(convolution.padding) != (fir_length - 1,)
        or convolution.padding_mode != "zeros"
        or convolution.bias is not None
    ):
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous ShortConv must be a causal depthwise FIR over C."
        )
    writer = getattr(operator, "out_proj", None)
    if rank is None:
        if type(writer) is not torch.nn.Linear:
            raise Lfm2UnsupportedLayoutError(
                "Dense heterogeneous ShortConv writer must be one Linear."
            )
        output_shapes = (
            _linear_module_shape(writer, bias=False, name="out_proj", torch=torch),
        )
        if output_shapes != ((hidden_size, internal),):
            raise Lfm2UnsupportedLayoutError(
                "Dense heterogeneous ShortConv writer must have shape [H, C]."
            )
        writer_topology: Literal["dense", "factorized"] = "dense"
    else:
        if type(writer) is not torch.nn.Sequential or len(writer) != 2:
            raise Lfm2UnsupportedLayoutError(
                "Factorized ShortConv writer must be exactly two Linear modules."
            )
        if any(type(module) is not torch.nn.Linear for module in writer):
            raise Lfm2UnsupportedLayoutError(
                "Factorized ShortConv writer contains a non-Linear module."
            )
        output_shapes = tuple(
            _linear_module_shape(
                module,
                bias=False,
                name=f"out_proj.{index}",
                torch=torch,
            )
            for index, module in enumerate(writer)
        )
        if output_shapes != ((rank, internal), (hidden_size, rank)):
            raise Lfm2UnsupportedLayoutError(
                "Factorized ShortConv writer must have shapes [R, C] and [H, R]."
            )
        writer_topology = "factorized"
    return Lfm2HeterogeneousShortConvContract(
        external_hidden_size=hidden_size,
        internal_channel_width=internal,
        fir_length=fir_length,
        bias=False,
        input_projection_shape=in_shape,
        depthwise_kernel_shape=kernel_shape,
        output_projection_shapes=output_shapes,
        writer_topology=writer_topology,
        writer_rank=rank,
        groups=internal,
    )


def _validate_heterogeneous_layer_shell(
    layer: Any,
    *,
    hidden_size: int,
    expected_intermediate: int,
) -> int:
    for name in ("operator_norm", "ffn_norm"):
        norm = getattr(layer, name, None)
        _require_supported_class(norm, "Lfm2RMSNorm")
        _require_vector(getattr(norm, "weight", None), hidden_size, f"{name}.weight")
    feed_forward = getattr(layer, "feed_forward", None)
    _require_dynamic_supported_class(feed_forward, "Lfm2DynamicWidthMLP")
    torch = _require_torch()
    w1 = _linear_shape(feed_forward, "w1", bias=False, torch=torch)
    w3 = _linear_shape(feed_forward, "w3", bias=False, torch=torch)
    w2 = _linear_shape(feed_forward, "w2", bias=False, torch=torch)
    expected = _positive_int(expected_intermediate, "mlp_intermediate_size")
    if (
        w1 != (expected, hidden_size)
        or w3 != (expected, hidden_size)
        or w2 != (hidden_size, expected)
    ):
        raise Lfm2UnsupportedLayoutError(
            "Heterogeneous feed-forward tensors disagree with sealed geometry."
        )
    return expected


def _validate_layer_shell(layer: Any, hidden_size: int) -> int:
    for name in ("operator_norm", "ffn_norm"):
        norm = getattr(layer, name, None)
        _require_supported_class(norm, "Lfm2RMSNorm")
        _require_vector(getattr(norm, "weight", None), hidden_size, f"{name}.weight")
    feed_forward = getattr(layer, "feed_forward", None)
    _require_supported_class(feed_forward, "Lfm2MLP")
    torch = _require_torch()
    w1 = _linear_shape(feed_forward, "w1", bias=False, torch=torch)
    w3 = _linear_shape(feed_forward, "w3", bias=False, torch=torch)
    w2 = _linear_shape(feed_forward, "w2", bias=False, torch=torch)
    if w1 != w3 or w1[1] != hidden_size or w2 != (hidden_size, w1[0]):
        raise Lfm2UnsupportedLayoutError("LFM2 feed-forward tensor layout is unsupported.")
    return w1[0]


def _coerce_short_conv_parameters(source: Any) -> Lfm2ShortConvParameters:
    if isinstance(source, Lfm2ShortConvParameters):
        return source
    return extract_lfm2_short_conv_parameters(source)


def _validate_parameter_bundle(parameters: Lfm2ShortConvParameters) -> None:
    torch = _require_torch()
    for name in (
        "in_proj_weight",
        "conv_weight",
        "out_proj_weight",
    ):
        _require_tensor(getattr(parameters, name), name, torch)
    hidden_size = parameters.hidden_size
    if tuple(parameters.in_proj_weight.shape) != (3 * hidden_size, hidden_size):
        raise Lfm2CanonicalError("in_proj_weight must have shape [3H, H].")
    if tuple(parameters.out_proj_weight.shape) != (hidden_size, hidden_size):
        raise Lfm2CanonicalError("out_proj_weight must have shape [H, H].")
    if (
        parameters.conv_weight.ndim != 3
        or tuple(parameters.conv_weight.shape[:2]) != (hidden_size, 1)
        or int(parameters.conv_weight.shape[2]) <= 0
    ):
        raise Lfm2CanonicalError("conv_weight must have shape [H, 1, L].")
    bias_values = (
        parameters.in_proj_bias,
        parameters.conv_bias,
        parameters.out_proj_bias,
    )
    if any(value is None for value in bias_values) and not all(
        value is None for value in bias_values
    ):
        raise Lfm2CanonicalError("All LFM2 short-conv biases must be present or absent together.")
    if parameters.bias:
        _require_vector(parameters.in_proj_bias, 3 * hidden_size, "in_proj_bias")
        _require_vector(parameters.conv_bias, hidden_size, "conv_bias")
        _require_vector(parameters.out_proj_bias, hidden_size, "out_proj_bias")


def _select_channels(
    parameters: Lfm2ShortConvParameters,
    target_hidden_size: int,
) -> tuple[int, ...]:
    torch = _require_torch()
    values = _cpu_float64(parameters)
    b_weight, c_weight, x_weight = values.in_proj_weight.chunk(3, dim=0)
    score = torch.zeros(parameters.hidden_size, dtype=torch.float64)
    for weight in (b_weight, c_weight, x_weight, values.out_proj_weight):
        score += weight.square().sum(dim=0) + weight.square().sum(dim=1)
    score += values.conv_weight[:, 0, :].square().sum(dim=1)
    if values.in_proj_bias is not None:
        b_bias, c_bias, x_bias = values.in_proj_bias.chunk(3)
        score += b_bias.square() + c_bias.square() + x_bias.square()
        score += values.conv_bias.square() + values.out_proj_bias.square()
    return tuple(
        sorted(
            range(parameters.hidden_size),
            key=lambda index: (-float(score[index]), index),
        )[:target_hidden_size]
    )


def _assign_channels(
    parameters: Lfm2ShortConvParameters,
    selected: tuple[int, ...],
) -> tuple[int, ...]:
    torch = _require_torch()
    values = _cpu_float64(parameters)
    b_weight, c_weight, x_weight = values.in_proj_weight.chunk(3, dim=0)
    modalities = [
        b_weight,
        c_weight,
        x_weight,
        b_weight.transpose(0, 1),
        c_weight.transpose(0, 1),
        x_weight.transpose(0, 1),
        values.conv_weight[:, 0, :],
        values.out_proj_weight,
        values.out_proj_weight.transpose(0, 1),
    ]
    features = torch.cat([_normalize_rows(value) for value in modalities], dim=1)
    if values.in_proj_bias is not None:
        b_bias, c_bias, x_bias = values.in_proj_bias.chunk(3)
        features = torch.cat(
            (
                features,
                b_bias[:, None],
                c_bias[:, None],
                x_bias[:, None],
                values.conv_bias[:, None],
                values.out_proj_bias[:, None],
            ),
            dim=1,
        )
    features = _normalize_rows(features)
    medoids = features[list(selected)]
    similarity = features.matmul(medoids.transpose(0, 1))
    assignment = similarity.argmax(dim=1).tolist()
    for target, source_index in enumerate(selected):
        assignment[source_index] = target
    return tuple(int(value) for value in assignment)


def _classify_exactness(
    parameters: Lfm2ShortConvParameters,
    mode: str,
    selected: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
) -> Lfm2TransplantExactness:
    torch = _require_torch()
    source_hidden = parameters.hidden_size
    full_permutation = len(selected) == source_hidden and set(selected) == set(
        range(source_hidden)
    )
    channels_merged = any(len(group) > 1 for group in groups)
    if full_permutation and not channels_merged:
        return Lfm2TransplantExactness(
            level="algebraically_exact",
            scope="full-operator-coordinate-permutation",
            reason=(
                "Every hidden/FIR coordinate is copied once under one shared permutation."
            ),
            full_coordinate_permutation=True,
            selected_input_subnetwork_closed=True,
            selected_output_subnetwork_closed=True,
            channels_merged=False,
            topology_preserved=True,
        )
    selected_set = set(selected)
    dropped = tuple(index for index in range(source_hidden) if index not in selected_set)
    input_closed = False
    output_closed = False
    if dropped:
        selected_index = torch.tensor(selected, dtype=torch.long, device=parameters.in_proj_weight.device)
        dropped_index = torch.tensor(dropped, dtype=torch.long, device=parameters.in_proj_weight.device)
        producer_blocks = parameters.in_proj_weight.chunk(3, dim=0)
        input_closed = all(
            _is_exact_zero(
                block.index_select(0, selected_index).index_select(1, dropped_index)
            )
            for block in producer_blocks
        )
        output_closed = _is_exact_zero(
            parameters.out_proj_weight.index_select(0, selected_index).index_select(
                1, dropped_index
            )
        )
    if mode == "subset" and dropped and input_closed and output_closed:
        return Lfm2TransplantExactness(
            level="algebraically_exact",
            scope="closed-selected-coordinate-subnetwork",
            reason=(
                "Selected producer rows ignore dropped inputs and selected outputs ignore "
                "dropped FIR channels; the projected subnetwork is exactly closed."
            ),
            full_coordinate_permutation=False,
            selected_input_subnetwork_closed=True,
            selected_output_subnetwork_closed=True,
            channels_merged=False,
            topology_preserved=True,
        )
    reason = (
        "Hard channel merging crosses multiplicative B*x and C*v nonlinearities."
        if channels_merged
        else "Dropped coordinates are not an algebraically closed LFM2 subnetwork."
    )
    return Lfm2TransplantExactness(
        level="approximate",
        scope="none",
        reason=reason,
        full_coordinate_permutation=full_permutation,
        selected_input_subnetwork_closed=input_closed,
        selected_output_subnetwork_closed=output_closed,
        channels_merged=channels_merged,
        topology_preserved=True,
    )


def _subset_parameters(
    parameters: Lfm2ShortConvParameters,
    selected: tuple[int, ...],
) -> Lfm2ShortConvParameters:
    torch = _require_torch()
    index = torch.tensor(
        selected,
        dtype=torch.long,
        device=parameters.in_proj_weight.device,
    )
    producer_blocks = parameters.in_proj_weight.chunk(3, dim=0)
    in_weight = torch.cat(
        tuple(block.index_select(0, index).index_select(1, index) for block in producer_blocks),
        dim=0,
    ).contiguous()
    in_bias = None
    conv_bias = None
    out_bias = None
    if parameters.in_proj_bias is not None:
        in_bias = torch.cat(
            tuple(block.index_select(0, index) for block in parameters.in_proj_bias.chunk(3)),
            dim=0,
        ).contiguous()
        conv_bias = parameters.conv_bias.index_select(0, index).contiguous()
        out_bias = parameters.out_proj_bias.index_select(0, index).contiguous()
    return Lfm2ShortConvParameters(
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        conv_weight=parameters.conv_weight.index_select(0, index).contiguous(),
        conv_bias=conv_bias,
        out_proj_weight=(
            parameters.out_proj_weight.index_select(0, index)
            .index_select(1, index)
            .contiguous()
        ),
        out_proj_bias=out_bias,
    )


def _merge_parameters(
    parameters: Lfm2ShortConvParameters,
    assignment: tuple[int, ...],
    target_hidden_size: int,
) -> Lfm2ShortConvParameters:
    torch = _require_torch()
    device = parameters.in_proj_weight.device
    dtype = parameters.in_proj_weight.dtype
    source_hidden = parameters.hidden_size
    lift = torch.zeros(
        (source_hidden, target_hidden_size),
        dtype=dtype,
        device=device,
    )
    source_index = torch.arange(source_hidden, device=device)
    target_index = torch.tensor(assignment, dtype=torch.long, device=device)
    lift[source_index, target_index] = 1
    sizes = lift.sum(dim=0)
    if bool((sizes <= 0).any()):  # pragma: no cover - invariant guard
        raise Lfm2CanonicalError("Every merge target must own a source channel.")
    reduce = lift.transpose(0, 1) / sizes[:, None]

    def contract_matrix(matrix: Any) -> Any:
        return reduce.matmul(matrix).matmul(lift).contiguous()

    in_weight = torch.cat(
        tuple(contract_matrix(block) for block in parameters.in_proj_weight.chunk(3)),
        dim=0,
    ).contiguous()
    kernel = reduce.matmul(parameters.conv_weight[:, 0, :]).unsqueeze(1).contiguous()
    in_bias = None
    conv_bias = None
    out_bias = None
    if parameters.in_proj_bias is not None:
        in_bias = torch.cat(
            tuple(reduce.matmul(block) for block in parameters.in_proj_bias.chunk(3)),
            dim=0,
        ).contiguous()
        conv_bias = reduce.matmul(parameters.conv_bias).contiguous()
        out_bias = reduce.matmul(parameters.out_proj_bias).contiguous()
    return Lfm2ShortConvParameters(
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        conv_weight=kernel,
        conv_bias=conv_bias,
        out_proj_weight=contract_matrix(parameters.out_proj_weight),
        out_proj_bias=out_bias,
    )


def _cpu_float64(parameters: Lfm2ShortConvParameters) -> Lfm2ShortConvParameters:
    def convert(value: Any | None) -> Any | None:
        if value is None:
            return None
        return value.detach().to(device="cpu", dtype=_require_torch().float64)

    return Lfm2ShortConvParameters(
        in_proj_weight=convert(parameters.in_proj_weight),
        in_proj_bias=convert(parameters.in_proj_bias),
        conv_weight=convert(parameters.conv_weight),
        conv_bias=convert(parameters.conv_bias),
        out_proj_weight=convert(parameters.out_proj_weight),
        out_proj_bias=convert(parameters.out_proj_bias),
    )


def _normalize_rows(values: Any) -> Any:
    denominator = values.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-30)
    return values / denominator


def _parameters_sha256(parameters: Lfm2ShortConvParameters) -> str:
    digest = hashlib.sha256()
    for name in (
        "in_proj_weight",
        "in_proj_bias",
        "conv_weight",
        "conv_bias",
        "out_proj_weight",
        "out_proj_bias",
    ):
        digest.update(name.encode("ascii"))
        value = getattr(parameters, name)
        if value is None:
            digest.update(b"<none>")
            continue
        cpu = value.detach().to(device="cpu").contiguous()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(json.dumps(list(cpu.shape)).encode("ascii"))
        digest.update(cpu.view(_require_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_selected_channels(
    values: Sequence[int],
    source_hidden_size: int,
    target_hidden_size: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Lfm2CanonicalError("selected_channels must be an integer sequence.")
    result = tuple(values)
    if len(result) != target_hidden_size:
        raise Lfm2CanonicalError(
            "selected_channels length must equal target_hidden_size."
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise Lfm2CanonicalError("selected_channels must contain only integers.")
    if len(set(result)) != len(result):
        raise Lfm2CanonicalError("selected_channels must be unique.")
    if any(value < 0 or value >= source_hidden_size for value in result):
        raise Lfm2CanonicalError("selected_channels contains an out-of-range index.")
    return result


def _require_supported_class(value: Any, expected_name: str) -> None:
    value_type = type(value)
    if value_type.__name__ != expected_name or value_type.__module__ != _LFM2_MODULE:
        actual = f"{value_type.__module__}.{value_type.__name__}"
        raise Lfm2UnsupportedLayoutError(
            f"Expected {_LFM2_MODULE}.{expected_name}, received {actual}."
        )


def _require_dynamic_supported_class(value: Any, expected_name: str) -> None:
    value_type = type(value)
    if (
        value_type.__name__ != expected_name
        or value_type.__module__ != _LFM2_DYNAMIC_MODULE
    ):
        actual = f"{value_type.__module__}.{value_type.__name__}"
        raise Lfm2UnsupportedLayoutError(
            f"Expected {_LFM2_DYNAMIC_MODULE}.{expected_name}, received {actual}."
        )


def _linear_shape(module: Any, name: str, *, bias: bool, torch: Any) -> tuple[int, int]:
    linear = getattr(module, name, None)
    weight = getattr(linear, "weight", None)
    _require_tensor(weight, f"{name}.weight", torch)
    if weight.ndim != 2:
        raise Lfm2UnsupportedLayoutError(f"{name}.weight must be a matrix.")
    shape = tuple(int(value) for value in weight.shape)
    if getattr(linear, "out_features", None) != shape[0] or getattr(
        linear, "in_features", None
    ) != shape[1]:
        raise Lfm2UnsupportedLayoutError(f"{name} metadata disagrees with its weight.")
    _validate_bias(getattr(linear, "bias", None), bias, shape[0], f"{name}.bias")
    return shape


def _linear_module_shape(
    linear: Any,
    *,
    bias: bool,
    name: str,
    torch: Any,
) -> tuple[int, int]:
    if type(linear) is not torch.nn.Linear:
        raise Lfm2UnsupportedLayoutError(f"{name} must be one exact dense Linear.")
    weight = getattr(linear, "weight", None)
    _require_tensor(weight, f"{name}.weight", torch)
    if weight.ndim != 2:
        raise Lfm2UnsupportedLayoutError(f"{name}.weight must be a matrix.")
    shape = tuple(int(value) for value in weight.shape)
    if getattr(linear, "out_features", None) != shape[0] or getattr(
        linear, "in_features", None
    ) != shape[1]:
        raise Lfm2UnsupportedLayoutError(f"{name} metadata disagrees with its weight.")
    _validate_bias(getattr(linear, "bias", None), bias, shape[0], f"{name}.bias")
    return shape


def _validate_bias(value: Any, expected: bool, size: int, name: str) -> None:
    if expected:
        _require_vector(value, size, name)
    elif value is not None:
        raise Lfm2UnsupportedLayoutError(f"{name} must be absent.")


def _require_vector(value: Any, size: int, name: str) -> None:
    torch = _require_torch()
    _require_tensor(value, name, torch)
    if tuple(value.shape) != (size,):
        raise Lfm2UnsupportedLayoutError(f"{name} must have shape [{size}].")


def _require_tensor(value: Any, name: str, torch: Any) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise Lfm2UnsupportedLayoutError(f"{name} must be a floating-point tensor.")
    if value.device.type == "meta":
        raise Lfm2UnsupportedLayoutError(f"{name} cannot be validated on the meta device.")
    if not _is_finite(value):
        raise Lfm2UnsupportedLayoutError(f"{name} contains non-finite values.")


def _is_finite(value: Any) -> bool:
    torch = _require_torch()
    return bool(torch.isfinite(value.detach()).all().item())


def _is_exact_zero(value: Any) -> bool:
    torch = _require_torch()
    return int(torch.count_nonzero(value.detach()).item()) == 0


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2UnsupportedLayoutError(f"{name} must be a positive integer.")
    return value
