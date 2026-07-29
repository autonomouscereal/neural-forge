"""Authenticated physical layouts for nonuniform dense LFM2 children.

This module describes *physical* blank-child topology.  It does not select a
fold, allocate widths, load a checkpoint, or retain a parent-side runtime.  A
descriptor names every layer's SwiGLU width and, for short-convolution layers,
the independently materialized internal channel width and cache geometry.

The accounting is deliberately limited to dense, untied FP32 parameters.  A
descriptor or materialized module using aliases, sharing, quantization, sparse
storage, pruning masks, bridge parameters, or sidecar parameters fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal


LFM2_DYNAMIC_PHYSICAL_LAYOUT_SCHEMA = "neural-forge-lfm2-dynamic-physical-layout-v2"
LFM2_DYNAMIC_PHYSICAL_AUDIT_SCHEMA = "neural-forge-lfm2-dynamic-physical-audit-v2"
LFM2_RAGGED_MLP_RUNTIME_SCHEMA = "neural-forge-lfm2-ragged-swiglu-runtime-v1"
LFM2_HETEROGENEOUS_PHYSICAL_LAYOUT_SCHEMA = (
    "neural-forge-lfm2-heterogeneous-physical-layout-v1"
)

Lfm2PhysicalLayerType = Literal["conv", "full_attention"]

_SAFE_MATERIALIZATION_CONTRACT = {
    "parameter_dtype": "float32",
    "dense_strided_parameters_only": True,
    "token_embedding_readout_tied": False,
    "runtime_parameter_alias_group_count": 0,
    "cross_layer_parameter_sharing": False,
    "quantization_kind": "none",
    "sparse_parameter_storage": False,
    "pruning_masks_present": False,
    "projection_bias": False,
    "conv_bias": False,
    "runtime_bridge_parameter_elements": 0,
    "runtime_sidecar_parameter_elements": 0,
}


class Lfm2DynamicPhysicalLayoutError(ValueError):
    """The requested layout or its physical materialization is not admissible."""


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2DynamicPhysicalLayoutError(f"{label} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2DynamicPhysicalLayoutError(f"{label} must be a non-negative integer.")
    return value


def _positive_finite_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            f"{label} must be one positive finite number."
        )
    return float(value)


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise Lfm2DynamicPhysicalLayoutError(f"{label} must be one boolean.")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Lfm2DynamicPhysicalLayoutError(f"{label} must be one nonempty string.")
    return value


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
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical-layout provenance must be finite canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise Lfm2DynamicPhysicalLayoutError(f"{label} must be lowercase SHA-256.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise Lfm2DynamicPhysicalLayoutError(
            f"{label} must be lowercase SHA-256."
        ) from exc
    return value


@dataclass(frozen=True, slots=True)
class Lfm2ShortConvCacheContract:
    """Cache geometry forced by one protected internal short-conv width."""

    layer_index: int
    external_hidden_size: int
    internal_channel_width: int
    kernel_width: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_kind": "transformers-DynamicCache-only",
            "layer_index": self.layer_index,
            "external_hidden_size": self.external_hidden_size,
            "internal_channel_width": self.internal_channel_width,
            "kernel_width": self.kernel_width,
            "cache_state_shape": [
                "batch",
                self.internal_channel_width,
                self.kernel_width,
            ],
            "static_or_compileable_cache_supported": False,
            "fresh_cache_required_per_materialization": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LayerPhysicalLayout:
    """Physical widths for one decoder layer."""

    layer_index: int
    layer_type: Lfm2PhysicalLayerType
    mlp_intermediate_size: int
    short_conv_internal_width: int | None
    cache_contract: Lfm2ShortConvCacheContract | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_type": self.layer_type,
            "mlp_intermediate_size": self.mlp_intermediate_size,
            "short_conv_internal_width": self.short_conv_internal_width,
            "cache_contract": (
                None if self.cache_contract is None else self.cache_contract.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class Lfm2DenseFp32Accounting:
    """Exact, category-complete dense FP32 parameter accounting."""

    token_embedding_parameter_elements: int
    readout_parameter_elements: int
    residual_norm_parameter_elements: int
    attention_projection_parameter_elements: int
    attention_qk_norm_parameter_elements: int
    short_conv_parameter_elements: int
    mlp_parameter_elements: int
    body_parameter_elements: int
    total_parameter_elements: int
    total_parameter_bytes_fp32: int
    parent_parameter_elements: int
    requested_dense_parameter_budget: int
    budget_headroom_parameter_elements: int
    layer_parameter_elements: tuple[Mapping[str, Any], ...]

    @property
    def exact_compaction_ratio(self) -> float:
        return self.parent_parameter_elements / self.total_parameter_elements

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_embedding_parameter_elements": (
                self.token_embedding_parameter_elements
            ),
            "readout_parameter_elements": self.readout_parameter_elements,
            "residual_norm_parameter_elements": self.residual_norm_parameter_elements,
            "attention_projection_parameter_elements": (
                self.attention_projection_parameter_elements
            ),
            "attention_qk_norm_parameter_elements": (
                self.attention_qk_norm_parameter_elements
            ),
            "attention_parameter_elements": (
                self.attention_projection_parameter_elements
                + self.attention_qk_norm_parameter_elements
            ),
            "short_conv_parameter_elements": self.short_conv_parameter_elements,
            "mlp_parameter_elements": self.mlp_parameter_elements,
            "body_parameter_elements": self.body_parameter_elements,
            "total_parameter_elements": self.total_parameter_elements,
            "total_parameter_bytes_fp32": self.total_parameter_bytes_fp32,
            "parent_parameter_elements": self.parent_parameter_elements,
            "requested_dense_parameter_budget": self.requested_dense_parameter_budget,
            "budget_headroom_parameter_elements": (
                self.budget_headroom_parameter_elements
            ),
            "compaction_ratio_exact_fraction": {
                "numerator": self.parent_parameter_elements,
                "denominator": self.total_parameter_elements,
            },
            "layer_parameter_elements": [
                dict(row) for row in self.layer_parameter_elements
            ],
        }


@dataclass(frozen=True, slots=True)
class Lfm2DynamicPhysicalLayoutDescriptor:
    """Sealed dynamic LFM2 physical layout."""

    hidden_size: int
    vocab_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    conv_kernel_size: int
    rope_theta: float
    norm_eps: float
    max_position_embeddings: int
    use_cache: bool
    attention_implementation: str
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    block_auto_adjust_ff_dim: bool
    layers: tuple[Lfm2LayerPhysicalLayout, ...]
    accounting: Lfm2DenseFp32Accounting
    descriptor_sha256: str

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_DYNAMIC_PHYSICAL_LAYOUT_SCHEMA,
            "status": "authenticated-dense-fp32-blank-child-layout",
            "model_type": "lfm2",
            "family_adapter": "lfm2-dynamic-nonuniform-physical-layout-v2",
            "topology": {
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
                "head_dim": self.head_dim,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "conv_kernel_size": self.conv_kernel_size,
                "num_hidden_layers": self.layer_count,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": self.rope_theta,
                },
            },
            "runtime_semantics": {
                "norm_eps": self.norm_eps,
                "max_position_embeddings": self.max_position_embeddings,
                "use_cache": self.use_cache,
                "attention_implementation": self.attention_implementation,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "block_auto_adjust_ff_dim": self.block_auto_adjust_ff_dim,
            },
            "layers": [layer.as_dict() for layer in self.layers],
            "materialization_contract": dict(_SAFE_MATERIALIZATION_CONTRACT),
            "parameter_accounting": self.accounting.as_dict(),
            "allocation_policy_embedded": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body_dict(), "descriptor_sha256": self.descriptor_sha256}


def _parameter_accounting(
    *,
    hidden_size: int,
    vocab_size: int,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    layers: tuple[Lfm2LayerPhysicalLayout, ...],
    parent_parameter_elements: int,
    requested_dense_parameter_budget: int,
) -> Lfm2DenseFp32Accounting:
    embedding = vocab_size * hidden_size
    readout = vocab_size * hidden_size
    residual_norms = (2 * len(layers) + 1) * hidden_size
    query_width = num_attention_heads * head_dim
    key_value_width = num_key_value_heads * head_dim
    attention_projection = 0
    attention_qk_norm = 0
    short_conv = 0
    mlp = 0
    layer_rows: list[Mapping[str, Any]] = []
    for layer in layers:
        layer_mlp = 3 * hidden_size * layer.mlp_intermediate_size
        layer_norms = 2 * hidden_size
        if layer.layer_type == "full_attention":
            operator_projection = 2 * hidden_size * (query_width + key_value_width)
            operator_norm = 2 * head_dim
            operator = operator_projection + operator_norm
            attention_projection += operator_projection
            attention_qk_norm += operator_norm
        else:
            assert layer.short_conv_internal_width is not None
            internal = layer.short_conv_internal_width
            operator_projection = 4 * hidden_size * internal
            operator_norm = 0
            operator = (
                operator_projection + internal * layer.cache_contract.kernel_width
            )
            short_conv += operator
        mlp += layer_mlp
        layer_rows.append(
            MappingProxyType(
                {
                    "layer_index": layer.layer_index,
                    "layer_type": layer.layer_type,
                    "residual_norm_parameter_elements": layer_norms,
                    "operator_parameter_elements": operator,
                    "mlp_parameter_elements": layer_mlp,
                    "total_parameter_elements": layer_norms + operator + layer_mlp,
                }
            )
        )
    body = residual_norms + attention_projection + attention_qk_norm + short_conv + mlp
    total = embedding + readout + body
    return Lfm2DenseFp32Accounting(
        token_embedding_parameter_elements=embedding,
        readout_parameter_elements=readout,
        residual_norm_parameter_elements=residual_norms,
        attention_projection_parameter_elements=attention_projection,
        attention_qk_norm_parameter_elements=attention_qk_norm,
        short_conv_parameter_elements=short_conv,
        mlp_parameter_elements=mlp,
        body_parameter_elements=body,
        total_parameter_elements=total,
        total_parameter_bytes_fp32=4 * total,
        parent_parameter_elements=parent_parameter_elements,
        requested_dense_parameter_budget=requested_dense_parameter_budget,
        budget_headroom_parameter_elements=requested_dense_parameter_budget - total,
        layer_parameter_elements=tuple(layer_rows),
    )


def build_lfm2_dynamic_physical_layout(
    *,
    hidden_size: int,
    vocab_size: int,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    conv_kernel_size: int,
    rope_theta: float,
    norm_eps: float,
    max_position_embeddings: int,
    use_cache: bool,
    attention_implementation: str,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    block_auto_adjust_ff_dim: bool,
    layer_types: Sequence[Lfm2PhysicalLayerType],
    mlp_intermediate_sizes: Sequence[int],
    short_conv_internal_widths: Sequence[int | None],
    parent_parameter_elements: int,
    requested_dense_parameter_budget: int,
) -> Lfm2DynamicPhysicalLayoutDescriptor:
    """Build and seal an arbitrary per-layer dense FP32 LFM2 layout.

    Width selection is intentionally external.  This function authenticates a
    supplied allocation; it is not an allocation policy.
    """

    hidden = _positive_int(hidden_size, label="hidden_size")
    vocab = _positive_int(vocab_size, label="vocab_size")
    head = _positive_int(head_dim, label="head_dim")
    if head < 2 or head % 2 != 0:
        raise Lfm2DynamicPhysicalLayoutError(
            "head_dim must be one positive even default-RoPE width."
        )
    query_heads = _positive_int(num_attention_heads, label="num_attention_heads")
    kv_heads = _positive_int(num_key_value_heads, label="num_key_value_heads")
    kernel = _positive_int(conv_kernel_size, label="conv_kernel_size")
    theta = _positive_finite_float(rope_theta, label="rope_theta")
    epsilon = _positive_finite_float(norm_eps, label="norm_eps")
    max_positions = _positive_int(
        max_position_embeddings,
        label="max_position_embeddings",
    )
    cache_enabled = _strict_bool(use_cache, label="use_cache")
    attention_backend = _nonempty_string(
        attention_implementation,
        label="attention_implementation",
    )
    if attention_backend != "sdpa":
        raise Lfm2DynamicPhysicalLayoutError(
            "Dynamic LFM2 physical-layout v2 currently requires reload-stable SDPA."
        )
    bos = _nonnegative_int(bos_token_id, label="bos_token_id")
    eos = _nonnegative_int(eos_token_id, label="eos_token_id")
    pad = _nonnegative_int(pad_token_id, label="pad_token_id")
    if any(token_id >= vocab for token_id in (bos, eos, pad)):
        raise Lfm2DynamicPhysicalLayoutError(
            "LFM2 token IDs must be smaller than vocab_size."
        )
    auto_adjust = _strict_bool(
        block_auto_adjust_ff_dim,
        label="block_auto_adjust_ff_dim",
    )
    if auto_adjust:
        raise Lfm2DynamicPhysicalLayoutError(
            "Dynamic per-layer MLP widths require block_auto_adjust_ff_dim=False."
        )
    parent = _positive_int(parent_parameter_elements, label="parent_parameter_elements")
    budget = _positive_int(
        requested_dense_parameter_budget,
        label="requested_dense_parameter_budget",
    )
    kinds = tuple(layer_types)
    mlp_widths = tuple(mlp_intermediate_sizes)
    conv_widths = tuple(short_conv_internal_widths)
    if not kinds or len(kinds) != len(mlp_widths) or len(kinds) != len(conv_widths):
        raise Lfm2DynamicPhysicalLayoutError(
            "Layer types, MLP widths, and short-conv widths must be nonempty and aligned."
        )
    if query_heads % kv_heads != 0:
        raise Lfm2DynamicPhysicalLayoutError(
            "LFM2 query heads must be divisible by key/value heads."
        )
    layers: list[Lfm2LayerPhysicalLayout] = []
    for index, (kind, mlp_width, conv_width) in enumerate(
        zip(kinds, mlp_widths, conv_widths, strict=True)
    ):
        if kind not in {"conv", "full_attention"}:
            raise Lfm2DynamicPhysicalLayoutError(
                f"Layer {index} has unsupported type {kind!r}."
            )
        intermediate = _positive_int(
            mlp_width, label=f"layers[{index}].mlp_intermediate_size"
        )
        if kind == "conv":
            internal = _positive_int(
                conv_width,
                label=f"layers[{index}].short_conv_internal_width",
            )
            cache = Lfm2ShortConvCacheContract(
                layer_index=index,
                external_hidden_size=hidden,
                internal_channel_width=internal,
                kernel_width=kernel,
            )
        else:
            if conv_width is not None:
                raise Lfm2DynamicPhysicalLayoutError(
                    f"Attention layer {index} cannot declare short-conv channels."
                )
            internal = None
            cache = None
        layers.append(
            Lfm2LayerPhysicalLayout(
                layer_index=index,
                layer_type=kind,
                mlp_intermediate_size=intermediate,
                short_conv_internal_width=internal,
                cache_contract=cache,
            )
        )
    layer_tuple = tuple(layers)
    accounting = _parameter_accounting(
        hidden_size=hidden,
        vocab_size=vocab,
        head_dim=head,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        layers=layer_tuple,
        parent_parameter_elements=parent,
        requested_dense_parameter_budget=budget,
    )
    if accounting.total_parameter_elements > budget:
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target exceeds the requested dense parameter budget: "
            f"{accounting.total_parameter_elements} > {budget}."
        )
    if accounting.total_parameter_elements >= parent:
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target must contain fewer dense parameters than its parent."
        )
    provisional = Lfm2DynamicPhysicalLayoutDescriptor(
        hidden_size=hidden,
        vocab_size=vocab,
        head_dim=head,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        conv_kernel_size=kernel,
        rope_theta=theta,
        norm_eps=epsilon,
        max_position_embeddings=max_positions,
        use_cache=cache_enabled,
        attention_implementation=attention_backend,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
        block_auto_adjust_ff_dim=auto_adjust,
        layers=layer_tuple,
        accounting=accounting,
        descriptor_sha256="",
    )
    seal = _canonical_sha256(provisional._body_dict())
    return Lfm2DynamicPhysicalLayoutDescriptor(
        hidden_size=hidden,
        vocab_size=vocab,
        head_dim=head,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        conv_kernel_size=kernel,
        rope_theta=theta,
        norm_eps=epsilon,
        max_position_embeddings=max_positions,
        use_cache=cache_enabled,
        attention_implementation=attention_backend,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
        block_auto_adjust_ff_dim=auto_adjust,
        layers=layer_tuple,
        accounting=accounting,
        descriptor_sha256=seal,
    )


def authenticate_lfm2_dynamic_physical_layout(
    value: Lfm2DynamicPhysicalLayoutDescriptor | Mapping[str, Any],
    *,
    expected_descriptor_sha256: str | None = None,
) -> Lfm2DynamicPhysicalLayoutDescriptor:
    """Rebuild and authenticate a serialized physical layout."""

    raw = (
        value.as_dict()
        if isinstance(value, Lfm2DynamicPhysicalLayoutDescriptor)
        else value
    )
    if not isinstance(raw, Mapping):
        raise Lfm2DynamicPhysicalLayoutError("Physical layout must be one mapping.")
    serialized = dict(raw)
    expected_top_level = {
        "schema",
        "status",
        "model_type",
        "family_adapter",
        "topology",
        "runtime_semantics",
        "layers",
        "materialization_contract",
        "parameter_accounting",
        "allocation_policy_embedded",
        "descriptor_sha256",
    }
    if set(serialized) != expected_top_level:
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical-layout fields are not canonical."
        )
    supplied = _sha256(serialized.pop("descriptor_sha256"), label="descriptor seal")
    if _canonical_sha256(serialized) != supplied:
        raise Lfm2DynamicPhysicalLayoutError("Physical-layout seal is invalid.")
    if expected_descriptor_sha256 is not None and supplied != _sha256(
        expected_descriptor_sha256, label="expected descriptor seal"
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical layout differs from its externally supplied seal."
        )
    if (
        serialized.get("schema") != LFM2_DYNAMIC_PHYSICAL_LAYOUT_SCHEMA
        or serialized.get("status") != "authenticated-dense-fp32-blank-child-layout"
        or serialized.get("model_type") != "lfm2"
        or serialized.get("family_adapter")
        != "lfm2-dynamic-nonuniform-physical-layout-v2"
        or serialized.get("materialization_contract") != _SAFE_MATERIALIZATION_CONTRACT
        or serialized.get("allocation_policy_embedded") is not False
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical layout violates the dense, unshared FP32 contract."
        )
    topology = serialized.get("topology")
    runtime_semantics = serialized.get("runtime_semantics")
    layer_rows = serialized.get("layers")
    accounting = serialized.get("parameter_accounting")
    if (
        not isinstance(topology, Mapping)
        or set(topology)
        != {
            "hidden_size",
            "vocab_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "conv_kernel_size",
            "num_hidden_layers",
            "rope_parameters",
        }
        or not isinstance(runtime_semantics, Mapping)
        or set(runtime_semantics)
        != {
            "norm_eps",
            "max_position_embeddings",
            "use_cache",
            "attention_implementation",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "block_auto_adjust_ff_dim",
        }
        or not isinstance(layer_rows, list)
        or not isinstance(accounting, Mapping)
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical layout topology, layers, or accounting are malformed."
        )
    kinds: list[Lfm2PhysicalLayerType] = []
    mlp_widths: list[int] = []
    conv_widths: list[int | None] = []
    for index, row in enumerate(layer_rows):
        if not isinstance(row, Mapping) or set(row) != {
            "layer_index",
            "layer_type",
            "mlp_intermediate_size",
            "short_conv_internal_width",
            "cache_contract",
        }:
            raise Lfm2DynamicPhysicalLayoutError(
                f"Physical layer {index} is malformed."
            )
        if row.get("layer_index") != index:
            raise Lfm2DynamicPhysicalLayoutError("Physical layers are not contiguous.")
        kind = row.get("layer_type")
        if kind not in {"conv", "full_attention"}:
            raise Lfm2DynamicPhysicalLayoutError(
                f"Physical layer {index} has an unsupported type."
            )
        kinds.append(kind)
        mlp_widths.append(row.get("mlp_intermediate_size"))
        conv_widths.append(row.get("short_conv_internal_width"))
    rebuilt = build_lfm2_dynamic_physical_layout(
        hidden_size=topology.get("hidden_size"),
        vocab_size=topology.get("vocab_size"),
        head_dim=topology.get("head_dim"),
        num_attention_heads=topology.get("num_attention_heads"),
        num_key_value_heads=topology.get("num_key_value_heads"),
        conv_kernel_size=topology.get("conv_kernel_size"),
        rope_theta=(topology.get("rope_parameters") or {}).get("rope_theta")
        if isinstance(topology.get("rope_parameters"), Mapping)
        and topology.get("rope_parameters", {}).get("rope_type") == "default"
        else None,
        norm_eps=runtime_semantics.get("norm_eps"),
        max_position_embeddings=runtime_semantics.get("max_position_embeddings"),
        use_cache=runtime_semantics.get("use_cache"),
        attention_implementation=runtime_semantics.get("attention_implementation"),
        bos_token_id=runtime_semantics.get("bos_token_id"),
        eos_token_id=runtime_semantics.get("eos_token_id"),
        pad_token_id=runtime_semantics.get("pad_token_id"),
        block_auto_adjust_ff_dim=runtime_semantics.get("block_auto_adjust_ff_dim"),
        layer_types=kinds,
        mlp_intermediate_sizes=mlp_widths,
        short_conv_internal_widths=conv_widths,
        parent_parameter_elements=accounting.get("parent_parameter_elements"),
        requested_dense_parameter_budget=accounting.get(
            "requested_dense_parameter_budget"
        ),
    )
    if rebuilt.as_dict() != {**serialized, "descriptor_sha256": supplied}:
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical layout is sealed but not a canonical reconstruction."
        )
    return rebuilt


def build_canonical_lfm2_exact4x_protected_stem_proposal() -> (
    Lfm2DynamicPhysicalLayoutDescriptor
):
    """Return the current arithmetic proposal, not a compiler allocation policy."""

    layer_types: tuple[Lfm2PhysicalLayerType, ...] = (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )
    return build_lfm2_dynamic_physical_layout(
        hidden_size=776,
        vocab_size=4096,
        head_dim=64,
        num_attention_heads=10,
        num_key_value_heads=5,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        layer_types=layer_types,
        mlp_intermediate_sizes=(
            700,
            *(674 for _ in range(6)),
            *(675 for _ in range(7)),
        ),
        short_conv_internal_widths=tuple(
            1024 if index == 0 else (776 if kind == "conv" else None)
            for index, kind in enumerate(layer_types)
        ),
        parent_parameter_elements=229_693_184,
        requested_dense_parameter_budget=229_693_184 // 4,
    )


def lfm2_expected_physical_state_shapes(
    descriptor: Lfm2DynamicPhysicalLayoutDescriptor | Mapping[str, Any],
) -> Mapping[str, tuple[int, ...]]:
    """Return the complete same-key physical state namespace and shapes."""

    layout = authenticate_lfm2_dynamic_physical_layout(descriptor)
    hidden = layout.hidden_size
    query_width = layout.num_attention_heads * layout.head_dim
    kv_width = layout.num_key_value_heads * layout.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (layout.vocab_size, hidden),
        "model.embedding_norm.weight": (hidden,),
        "lm_head.weight": (layout.vocab_size, hidden),
    }
    for layer in layout.layers:
        prefix = f"model.layers.{layer.layer_index}."
        shapes[prefix + "operator_norm.weight"] = (hidden,)
        shapes[prefix + "ffn_norm.weight"] = (hidden,)
        intermediate = layer.mlp_intermediate_size
        shapes[prefix + "feed_forward.w1.weight"] = (intermediate, hidden)
        shapes[prefix + "feed_forward.w3.weight"] = (intermediate, hidden)
        shapes[prefix + "feed_forward.w2.weight"] = (hidden, intermediate)
        if layer.layer_type == "conv":
            assert layer.short_conv_internal_width is not None
            internal = layer.short_conv_internal_width
            shapes[prefix + "conv.in_proj.weight"] = (3 * internal, hidden)
            shapes[prefix + "conv.conv.weight"] = (
                internal,
                1,
                layout.conv_kernel_size,
            )
            shapes[prefix + "conv.out_proj.weight"] = (hidden, internal)
        else:
            shapes[prefix + "self_attn.q_proj.weight"] = (query_width, hidden)
            shapes[prefix + "self_attn.k_proj.weight"] = (kv_width, hidden)
            shapes[prefix + "self_attn.v_proj.weight"] = (kv_width, hidden)
            shapes[prefix + "self_attn.out_proj.weight"] = (hidden, query_width)
            shapes[prefix + "self_attn.q_layernorm.weight"] = (layout.head_dim,)
            shapes[prefix + "self_attn.k_layernorm.weight"] = (layout.head_dim,)
    if sum(_shape_elements(shape) for shape in shapes.values()) != (
        layout.accounting.total_parameter_elements
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Expected physical namespace disagrees with parameter accounting."
        )
    return MappingProxyType(shapes)


def _shape_elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def authenticate_lfm2_physical_materialization(
    model: Any,
    descriptor: Lfm2DynamicPhysicalLayoutDescriptor | Mapping[str, Any],
    *,
    expected_descriptor_sha256: str,
    expected_runtime_root_class: type[Any],
    expected_module_type_names: Mapping[str, str],
) -> Mapping[str, Any]:
    """Fail closed unless a module is the exact dense, unshared FP32 layout."""

    layout = authenticate_lfm2_dynamic_physical_layout(
        descriptor,
        expected_descriptor_sha256=expected_descriptor_sha256,
    )
    expected = lfm2_expected_physical_state_shapes(layout)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for physical materialization audit."
        ) from exc
    if (
        not isinstance(expected_runtime_root_class, type)
        or not issubclass(expected_runtime_root_class, torch.nn.Module)
        or type(model) is not expected_runtime_root_class
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target differs from its externally pinned exact runtime class."
        )
    if not isinstance(expected_module_type_names, Mapping):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target requires one externally pinned module-type topology."
        )
    try:
        modules = tuple(model.named_modules(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old optional PyTorch
        modules = tuple(model.named_modules())
    module_names = tuple(name for name, _ in modules)
    supplied_module_types = dict(expected_module_type_names)
    if (
        len(module_names) != len(set(module_names))
        or set(supplied_module_types) != set(module_names)
        or any(
            not isinstance(value, str) or not value
            for value in supplied_module_types.values()
        )
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target module-type topology pin is malformed."
        )

    def type_name(value: Any) -> str:
        return f"{type(value).__module__}.{type(value).__qualname__}"

    observed_module_types = {name: type_name(module) for name, module in modules}
    if observed_module_types != supplied_module_types:
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target module graph differs from its external exact-type pin."
        )
    if any(
        type(module.training) is not bool or module.training for _, module in modules
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target and every registered module must be in eval mode."
        )
    if bool(getattr(model, "is_gradient_checkpointing", False)) or any(
        bool(getattr(module, "gradient_checkpointing", False)) for _, module in modules
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target cannot enable gradient checkpointing."
        )

    def contains_unregistered_runtime_state(value: Any, seen: set[int]) -> bool:
        if isinstance(value, (torch.Tensor, torch.nn.Module)) or callable(value):
            return True
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen:
                return False
            seen.add(identity)
            return any(
                contains_unregistered_runtime_state(key, seen)
                or contains_unregistered_runtime_state(item, seen)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            identity = id(value)
            if identity in seen:
                return False
            seen.add(identity)
            return any(
                contains_unregistered_runtime_state(item, seen) for item in value
            )
        try:
            attributes = vars(value)
        except TypeError:
            return False
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        return any(
            contains_unregistered_runtime_state(item, seen)
            for item in attributes.values()
        )

    registry_names = {"_modules", "_parameters", "_buffers"}
    for module_path, module in modules:
        for attribute_name, value in vars(module).items():
            if attribute_name in registry_names:
                continue
            if attribute_name == "_hidden_kernels":
                kernel_rows = value if isinstance(value, Mapping) else {}
                rotary = kernel_rows.get("apply_rotary_pos_emb")
                if (
                    set(kernel_rows) != {"apply_rotary_pos_emb"}
                    or getattr(rotary, "__module__", None)
                    != "transformers.models.lfm2.modeling_lfm2"
                    or getattr(rotary, "__qualname__", None) != "apply_rotary_pos_emb"
                ):
                    raise Lfm2DynamicPhysicalLayoutError(
                        f"Physical module {module_path!r} has an unsealed runtime kernel."
                    )
                continue
            if "hook" in attribute_name.lower() and value:
                raise Lfm2DynamicPhysicalLayoutError(
                    f"Physical module {module_path!r} has an executable runtime hook."
                )
            if contains_unregistered_runtime_state(value, set()):
                raise Lfm2DynamicPhysicalLayoutError(
                    f"Physical module {module_path!r} hides unregistered runtime state."
                )
    try:
        named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old optional PyTorch
        named = tuple(model.named_parameters())
    names = tuple(name for name, _ in named)
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target has missing, extra, or duplicate parameter names."
        )
    object_ids: set[int] = set()
    storage_ids: set[tuple[str, int]] = set()
    for name, parameter in named:
        if (
            type(parameter) is not torch.nn.Parameter
            or parameter.device.type == "meta"
            or parameter.dtype != torch.float32
            or parameter.layout != torch.strided
            or bool(getattr(parameter, "is_quantized", False))
            or tuple(parameter.shape) != expected[name]
            or not parameter.is_contiguous()
            or int(parameter.storage_offset()) != 0
            or parameter.requires_grad
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                f"Physical parameter {name} is not its exact dense FP32 shape."
            )
        if id(parameter) in object_ids:
            raise Lfm2DynamicPhysicalLayoutError(
                "Physical target uses parameter aliases or sharing."
            )
        object_ids.add(id(parameter))
        storage = parameter.untyped_storage()
        if int(storage.nbytes()) != int(parameter.numel() * parameter.element_size()):
            raise Lfm2DynamicPhysicalLayoutError(
                f"Physical parameter {name} does not own one exact dense storage."
            )
        storage_identity = (str(parameter.device), int(storage.data_ptr()))
        if storage_identity in storage_ids:
            raise Lfm2DynamicPhysicalLayoutError(
                "Physical target uses shared or aliased parameter storage."
            )
        storage_ids.add(storage_identity)
    try:
        buffers = tuple(model.named_buffers(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old optional PyTorch
        buffers = tuple(model.named_buffers())
    expected_buffer_names = {
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    }
    buffer_names = tuple(name for name, _ in buffers)
    if len(buffer_names) != len(set(buffer_names)) or set(buffer_names) != (
        expected_buffer_names
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target contains a missing, extra, aliased, or sidecar buffer."
        )
    expected_inv_freq_cpu = 1.0 / (
        layout.rope_theta
        ** (
            torch.arange(0, layout.head_dim, 2, dtype=torch.int64).to(
                dtype=torch.float32
            )
            / layout.head_dim
        )
    )
    buffer_object_ids: set[int] = set()
    buffer_storage_ids: set[tuple[str, int]] = set()
    authenticated_buffers: dict[str, Any] = {}
    for name, buffer in buffers:
        expected_inv_freq_device = 1.0 / (
            layout.rope_theta
            ** (
                torch.arange(
                    0,
                    layout.head_dim,
                    2,
                    dtype=torch.int64,
                    device=buffer.device,
                ).to(dtype=torch.float32)
                / layout.head_dim
            )
        )
        if (
            type(buffer) is not torch.Tensor
            or buffer.device.type == "meta"
            or buffer.dtype != torch.float32
            or buffer.layout != torch.strided
            or bool(getattr(buffer, "is_quantized", False))
            or tuple(buffer.shape) != (layout.head_dim // 2,)
            or not buffer.is_contiguous()
            or int(buffer.storage_offset()) != 0
            or buffer.requires_grad
            or not bool(torch.isfinite(buffer).all().item())
            or not (
                torch.equal(buffer.detach().cpu(), expected_inv_freq_cpu)
                or torch.equal(buffer.detach(), expected_inv_freq_device)
            )
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                f"Intrinsic rotary buffer {name} violates its sealed default-RoPE contract."
            )
        storage = buffer.untyped_storage()
        if int(storage.nbytes()) != int(buffer.numel() * buffer.element_size()):
            raise Lfm2DynamicPhysicalLayoutError(
                f"Intrinsic rotary buffer {name} does not own exact dense storage."
            )
        storage_identity = (str(buffer.device), int(storage.data_ptr()))
        if (
            id(buffer) in buffer_object_ids
            or storage_identity in buffer_storage_ids
            or storage_identity in storage_ids
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                "Physical target shares intrinsic rotary-buffer storage."
            )
        buffer_object_ids.add(id(buffer))
        buffer_storage_ids.add(storage_identity)
        authenticated_buffers[name] = buffer
    if not torch.equal(
        authenticated_buffers["model.rotary_emb.inv_freq"],
        authenticated_buffers["model.rotary_emb.original_inv_freq"],
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Intrinsic rotary buffers differ despite the sealed default-RoPE contract."
        )
    state = model.state_dict(keep_vars=True)
    if set(state) != set(expected) or any(
        tuple(value.shape) != expected[name] for name, value in state.items()
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target contains persistent buffers, masks, or sidecar state."
        )
    observed = sum(int(parameter.numel()) for _, parameter in named)
    if observed != layout.accounting.total_parameter_elements:
        raise Lfm2DynamicPhysicalLayoutError(
            "Physical target count differs from authenticated accounting."
        )
    body = {
        "schema": LFM2_DYNAMIC_PHYSICAL_AUDIT_SCHEMA,
        "status": "dense-fp32-physical-layout-authenticated",
        "descriptor_sha256": layout.descriptor_sha256,
        "runtime_root_class": type_name(model),
        "module_type_topology": observed_module_types,
        "module_type_topology_sha256": _canonical_sha256(observed_module_types),
        "registered_module_count": len(modules),
        "all_registered_modules_eval": True,
        "gradient_checkpointing_enabled": False,
        "parameter_state_key_count": len(named),
        "all_parameters_frozen": True,
        "unique_parameter_object_count": len(object_ids),
        "unique_parameter_storage_count": len(storage_ids),
        "intrinsic_nonpersistent_buffer_count": len(buffers),
        "intrinsic_nonpersistent_buffer_elements": sum(
            int(buffer.numel()) for _, buffer in buffers
        ),
        "intrinsic_nonpersistent_buffer_storage_count": len(buffer_storage_ids),
        "parameter_elements": observed,
        "parameter_bytes_fp32": 4 * observed,
        "runtime_parameter_alias_group_count": 0,
        "cross_layer_parameter_sharing": False,
        "parameters_quantized": 0,
        "sparse_parameter_storage": False,
        "pruning_masks_present": False,
        "runtime_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
    }
    return MappingProxyType({**body, "result_sha256": _canonical_sha256(body)})


_RAGGED_MLP_RUNTIME_CLASS: type[Any] | None = None


def _ragged_mlp_runtime_class() -> type[Any]:
    global _RAGGED_MLP_RUNTIME_CLASS
    if _RAGGED_MLP_RUNTIME_CLASS is not None:
        return _RAGGED_MLP_RUNTIME_CLASS
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for the ragged LFM2 MLP.") from exc

    class RaggedLfm2SwiGLURuntime(torch.nn.Module):
        """Bias-free LFM2 SwiGLU with native ``w1/w3/w2`` state keys."""

        def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            *,
            device: Any = "cpu",
        ) -> None:
            super().__init__()
            hidden = _positive_int(hidden_size, label="hidden_size")
            intermediate = _positive_int(intermediate_size, label="intermediate_size")
            resolved_device = torch.device(device)
            if resolved_device.type == "meta":
                raise Lfm2DynamicPhysicalLayoutError(
                    "Ragged LFM2 MLP cannot be constructed on the meta device."
                )
            self.hidden_size = hidden
            self.intermediate_size = intermediate
            self.runtime_schema = LFM2_RAGGED_MLP_RUNTIME_SCHEMA
            self.w1 = torch.nn.Linear(
                hidden,
                intermediate,
                bias=False,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.w3 = torch.nn.Linear(
                hidden,
                intermediate,
                bias=False,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.w2 = torch.nn.Linear(
                intermediate,
                hidden,
                bias=False,
                device=resolved_device,
                dtype=torch.float32,
            )
            self.requires_grad_(False)
            self.eval()

        def forward(self, hidden_states: Any) -> Any:
            if (
                not isinstance(hidden_states, torch.Tensor)
                or hidden_states.device.type == "meta"
                or hidden_states.layout != torch.strided
                or hidden_states.dtype != torch.float32
                or hidden_states.ndim < 2
                or int(hidden_states.shape[-1]) != self.hidden_size
                or hidden_states.device != self.w1.weight.device
                or hidden_states.requires_grad
                or not bool(torch.isfinite(hidden_states).all().item())
            ):
                raise Lfm2DynamicPhysicalLayoutError(
                    "Ragged LFM2 MLP input must be detached finite dense FP32 "
                    "state with the declared hidden width on the module device."
                )
            gate = torch.nn.functional.silu(self.w1(hidden_states))
            return self.w2(gate * self.w3(hidden_states))

    RaggedLfm2SwiGLURuntime.__name__ = "Lfm2RaggedSwiGLURuntime"
    RaggedLfm2SwiGLURuntime.__qualname__ = "Lfm2RaggedSwiGLURuntime"
    _RAGGED_MLP_RUNTIME_CLASS = RaggedLfm2SwiGLURuntime
    return RaggedLfm2SwiGLURuntime


class Lfm2RaggedSwiGLURuntime:
    """Lazy public constructor for the optional-PyTorch ragged MLP runtime."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return _ragged_mlp_runtime_class()(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousLayerPhysicalLayout:
    """One evidence-selected layer in a constant-residual-width LFM2 child.

    The record is deliberately a complete tagged union.  Short-convolution
    layers carry their internal ``C`` width and may replace the dense ``C -> H``
    writer with two dense matrices of rank ``r``.  Attention layers carry their
    own GQA and head-width geometry.  Every layer independently carries the
    evidence digest that selected its physical geometry.
    """

    layer_index: int
    layer_type: Lfm2PhysicalLayerType
    mlp_intermediate_size: int
    selection_evidence_sha256: str
    short_conv_internal_width: int | None = None
    short_conv_writer_rank: int | None = None
    query_heads: int | None = None
    key_value_heads: int | None = None
    head_dim: int | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.layer_index, label="layer_index")
        _positive_int(
            self.mlp_intermediate_size,
            label=f"layers[{self.layer_index}].mlp_intermediate_size",
        )
        _sha256(
            self.selection_evidence_sha256,
            label=f"layers[{self.layer_index}].selection_evidence_sha256",
        )
        if self.layer_type == "conv":
            _positive_int(
                self.short_conv_internal_width,
                label=f"layers[{self.layer_index}].short_conv_internal_width",
            )
            if self.short_conv_writer_rank is not None:
                rank = _positive_int(
                    self.short_conv_writer_rank,
                    label=f"layers[{self.layer_index}].short_conv_writer_rank",
                )
                if rank > int(self.short_conv_internal_width):
                    raise Lfm2DynamicPhysicalLayoutError(
                        "A factorized short-convolution writer rank cannot exceed "
                        "its internal channel width."
                    )
            if any(
                value is not None
                for value in (self.query_heads, self.key_value_heads, self.head_dim)
            ):
                raise Lfm2DynamicPhysicalLayoutError(
                    "Short-convolution layers cannot declare attention geometry."
                )
            return
        if self.layer_type != "full_attention":
            raise Lfm2DynamicPhysicalLayoutError(
                f"Layer {self.layer_index} has unsupported type {self.layer_type!r}."
            )
        if (
            self.short_conv_internal_width is not None
            or self.short_conv_writer_rank is not None
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                "Attention layers cannot declare short-convolution geometry."
            )
        query = _positive_int(
            self.query_heads,
            label=f"layers[{self.layer_index}].query_heads",
        )
        key_value = _positive_int(
            self.key_value_heads,
            label=f"layers[{self.layer_index}].key_value_heads",
        )
        width = _positive_int(
            self.head_dim,
            label=f"layers[{self.layer_index}].head_dim",
        )
        if query % key_value != 0:
            raise Lfm2DynamicPhysicalLayoutError(
                "Every LFM2 query-head count must be divisible by its "
                "key/value-head count."
            )
        if width % 2 != 0:
            raise Lfm2DynamicPhysicalLayoutError(
                "Every LFM2 attention head width must be even for default RoPE."
            )

    @classmethod
    def short_conv(
        cls,
        *,
        layer_index: int,
        mlp_intermediate_size: int,
        internal_width: int,
        selection_evidence_sha256: str,
        writer_rank: int | None = None,
    ) -> Lfm2HeterogeneousLayerPhysicalLayout:
        return cls(
            layer_index=layer_index,
            layer_type="conv",
            mlp_intermediate_size=mlp_intermediate_size,
            selection_evidence_sha256=selection_evidence_sha256,
            short_conv_internal_width=internal_width,
            short_conv_writer_rank=writer_rank,
        )

    @classmethod
    def attention(
        cls,
        *,
        layer_index: int,
        mlp_intermediate_size: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
        selection_evidence_sha256: str,
    ) -> Lfm2HeterogeneousLayerPhysicalLayout:
        return cls(
            layer_index=layer_index,
            layer_type="full_attention",
            mlp_intermediate_size=mlp_intermediate_size,
            selection_evidence_sha256=selection_evidence_sha256,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_type": self.layer_type,
            "mlp_intermediate_size": self.mlp_intermediate_size,
            "selection_evidence_sha256": self.selection_evidence_sha256,
            "short_conv_internal_width": self.short_conv_internal_width,
            "short_conv_writer_rank": self.short_conv_writer_rank,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
        }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousLayerParameterAccounting:
    """Exact unique-parameter inventory for one heterogeneous layer."""

    layer_index: int
    layer_type: Lfm2PhysicalLayerType
    residual_norm_parameter_elements: int
    mlp_parameter_elements: int
    attention_projection_parameter_elements: int
    attention_qk_norm_parameter_elements: int
    short_conv_input_projection_parameter_elements: int
    short_conv_fir_parameter_elements: int
    short_conv_dense_writer_parameter_elements: int
    short_conv_factor_writer_parameter_elements: int
    total_parameter_elements: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_type": self.layer_type,
            "residual_norm_parameter_elements": (
                self.residual_norm_parameter_elements
            ),
            "mlp_parameter_elements": self.mlp_parameter_elements,
            "attention_projection_parameter_elements": (
                self.attention_projection_parameter_elements
            ),
            "attention_qk_norm_parameter_elements": (
                self.attention_qk_norm_parameter_elements
            ),
            "short_conv_input_projection_parameter_elements": (
                self.short_conv_input_projection_parameter_elements
            ),
            "short_conv_fir_parameter_elements": (
                self.short_conv_fir_parameter_elements
            ),
            "short_conv_dense_writer_parameter_elements": (
                self.short_conv_dense_writer_parameter_elements
            ),
            "short_conv_factor_writer_parameter_elements": (
                self.short_conv_factor_writer_parameter_elements
            ),
            "total_parameter_elements": self.total_parameter_elements,
        }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousParameterAccounting:
    """Category-complete unique-parameter accounting for an untied child."""

    input_embedding_parameter_elements: int
    output_readout_parameter_elements: int
    untied_vocab_interface_parameter_elements: int
    per_layer_residual_norm_parameter_elements: int
    final_norm_parameter_elements: int
    norm_parameter_elements: int
    mlp_parameter_elements: int
    attention_projection_parameter_elements: int
    attention_qk_norm_parameter_elements: int
    short_conv_input_projection_parameter_elements: int
    short_conv_fir_parameter_elements: int
    short_conv_dense_writer_parameter_elements: int
    short_conv_factor_writer_parameter_elements: int
    body_parameter_elements: int
    unique_parameter_elements: int
    logical_parameter_elements: int
    parameter_bytes_fp32: int
    layers: tuple[Lfm2HeterogeneousLayerParameterAccounting, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_embedding_parameter_elements": (
                self.input_embedding_parameter_elements
            ),
            "output_readout_parameter_elements": (
                self.output_readout_parameter_elements
            ),
            "untied_vocab_interface_parameter_elements": (
                self.untied_vocab_interface_parameter_elements
            ),
            "per_layer_residual_norm_parameter_elements": (
                self.per_layer_residual_norm_parameter_elements
            ),
            "final_norm_parameter_elements": self.final_norm_parameter_elements,
            "norm_parameter_elements": self.norm_parameter_elements,
            "mlp_parameter_elements": self.mlp_parameter_elements,
            "attention_projection_parameter_elements": (
                self.attention_projection_parameter_elements
            ),
            "attention_qk_norm_parameter_elements": (
                self.attention_qk_norm_parameter_elements
            ),
            "short_conv_input_projection_parameter_elements": (
                self.short_conv_input_projection_parameter_elements
            ),
            "short_conv_fir_parameter_elements": (
                self.short_conv_fir_parameter_elements
            ),
            "short_conv_dense_writer_parameter_elements": (
                self.short_conv_dense_writer_parameter_elements
            ),
            "short_conv_factor_writer_parameter_elements": (
                self.short_conv_factor_writer_parameter_elements
            ),
            "short_conv_parameter_elements": (
                self.short_conv_input_projection_parameter_elements
                + self.short_conv_fir_parameter_elements
                + self.short_conv_dense_writer_parameter_elements
                + self.short_conv_factor_writer_parameter_elements
            ),
            "attention_parameter_elements": (
                self.attention_projection_parameter_elements
                + self.attention_qk_norm_parameter_elements
            ),
            "body_parameter_elements": self.body_parameter_elements,
            "unique_parameter_elements": self.unique_parameter_elements,
            "logical_parameter_elements": self.logical_parameter_elements,
            "parameter_bytes_fp32": self.parameter_bytes_fp32,
            "layers": [layer.as_dict() for layer in self.layers],
        }


def _heterogeneous_parameter_accounting(
    *,
    hidden_size: int,
    vocab_size: int,
    conv_kernel_size: int,
    layers: tuple[Lfm2HeterogeneousLayerPhysicalLayout, ...],
) -> Lfm2HeterogeneousParameterAccounting:
    input_embedding = vocab_size * hidden_size
    output_readout = vocab_size * hidden_size
    layer_norms = 2 * len(layers) * hidden_size
    final_norm = hidden_size
    mlp_total = 0
    attention_projection_total = 0
    attention_qk_norm_total = 0
    short_conv_input_total = 0
    short_conv_fir_total = 0
    short_conv_dense_writer_total = 0
    short_conv_factor_writer_total = 0
    rows: list[Lfm2HeterogeneousLayerParameterAccounting] = []
    for layer in layers:
        mlp = 3 * hidden_size * layer.mlp_intermediate_size
        attention_projection = 0
        attention_qk_norm = 0
        short_conv_input = 0
        short_conv_fir = 0
        short_conv_dense_writer = 0
        short_conv_factor_writer = 0
        if layer.layer_type == "full_attention":
            assert layer.query_heads is not None
            assert layer.key_value_heads is not None
            assert layer.head_dim is not None
            attention_projection = (
                2
                * hidden_size
                * layer.head_dim
                * (layer.query_heads + layer.key_value_heads)
            )
            attention_qk_norm = 2 * layer.head_dim
        else:
            assert layer.short_conv_internal_width is not None
            internal = layer.short_conv_internal_width
            short_conv_input = 3 * internal * hidden_size
            short_conv_fir = internal * conv_kernel_size
            if layer.short_conv_writer_rank is None:
                short_conv_dense_writer = internal * hidden_size
            else:
                short_conv_factor_writer = layer.short_conv_writer_rank * (
                    internal + hidden_size
                )
        total = (
            2 * hidden_size
            + mlp
            + attention_projection
            + attention_qk_norm
            + short_conv_input
            + short_conv_fir
            + short_conv_dense_writer
            + short_conv_factor_writer
        )
        rows.append(
            Lfm2HeterogeneousLayerParameterAccounting(
                layer_index=layer.layer_index,
                layer_type=layer.layer_type,
                residual_norm_parameter_elements=2 * hidden_size,
                mlp_parameter_elements=mlp,
                attention_projection_parameter_elements=attention_projection,
                attention_qk_norm_parameter_elements=attention_qk_norm,
                short_conv_input_projection_parameter_elements=short_conv_input,
                short_conv_fir_parameter_elements=short_conv_fir,
                short_conv_dense_writer_parameter_elements=(
                    short_conv_dense_writer
                ),
                short_conv_factor_writer_parameter_elements=(
                    short_conv_factor_writer
                ),
                total_parameter_elements=total,
            )
        )
        mlp_total += mlp
        attention_projection_total += attention_projection
        attention_qk_norm_total += attention_qk_norm
        short_conv_input_total += short_conv_input
        short_conv_fir_total += short_conv_fir
        short_conv_dense_writer_total += short_conv_dense_writer
        short_conv_factor_writer_total += short_conv_factor_writer
    norms = layer_norms + final_norm
    body = (
        norms
        + mlp_total
        + attention_projection_total
        + attention_qk_norm_total
        + short_conv_input_total
        + short_conv_fir_total
        + short_conv_dense_writer_total
        + short_conv_factor_writer_total
    )
    interface = input_embedding + output_readout
    unique = interface + body
    return Lfm2HeterogeneousParameterAccounting(
        input_embedding_parameter_elements=input_embedding,
        output_readout_parameter_elements=output_readout,
        untied_vocab_interface_parameter_elements=interface,
        per_layer_residual_norm_parameter_elements=layer_norms,
        final_norm_parameter_elements=final_norm,
        norm_parameter_elements=norms,
        mlp_parameter_elements=mlp_total,
        attention_projection_parameter_elements=attention_projection_total,
        attention_qk_norm_parameter_elements=attention_qk_norm_total,
        short_conv_input_projection_parameter_elements=short_conv_input_total,
        short_conv_fir_parameter_elements=short_conv_fir_total,
        short_conv_dense_writer_parameter_elements=(short_conv_dense_writer_total),
        short_conv_factor_writer_parameter_elements=(short_conv_factor_writer_total),
        body_parameter_elements=body,
        unique_parameter_elements=unique,
        logical_parameter_elements=unique,
        parameter_bytes_fp32=4 * unique,
        layers=tuple(rows),
    )


def _heterogeneous_evidence_binding(
    *,
    source_topology_sha256: str,
    allocation_evidence_sha256: str,
    token_bridge_evidence_sha256: str,
    layers: tuple[Lfm2HeterogeneousLayerPhysicalLayout, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema": "neural-forge-lfm2-heterogeneous-layout-evidence-binding-v1",
            "source_topology_sha256": source_topology_sha256,
            "allocation_evidence_sha256": allocation_evidence_sha256,
            "token_bridge_evidence_sha256": token_bridge_evidence_sha256,
            # A selection receipt authenticates a physical choice, not merely
            # an opaque per-layer evidence digest.  Binding the complete
            # ordered records prevents a count-preserving geometry
            # substitution from reusing a stale allocation receipt.
            "ordered_layer_physical_layouts": [
                layer.as_dict() for layer in layers
            ],
        }
    )


def _heterogeneous_descriptor_body(
    *,
    hidden_size: int,
    vocab_size: int,
    conv_kernel_size: int,
    rope_theta: float,
    norm_eps: float,
    max_position_embeddings: int,
    use_cache: bool,
    attention_implementation: str,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    block_auto_adjust_ff_dim: bool,
    layers: tuple[Lfm2HeterogeneousLayerPhysicalLayout, ...],
    source_topology_sha256: str,
    allocation_evidence_sha256: str,
    token_bridge_evidence_sha256: str,
    evidence_binding_sha256: str,
    accounting: Lfm2HeterogeneousParameterAccounting,
) -> dict[str, Any]:
    return {
        "schema": LFM2_HETEROGENEOUS_PHYSICAL_LAYOUT_SCHEMA,
        "status": "authenticated-heterogeneous-dense-fp32-blank-child-layout",
        "model_type": "lfm2",
        "topology": {
            "constant_external_hidden_size": hidden_size,
            "conv_kernel_size": conv_kernel_size,
            "num_hidden_layers": len(layers),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": rope_theta,
            },
        },
        "runtime_semantics": {
            "norm_eps": norm_eps,
            "max_position_embeddings": max_position_embeddings,
            "use_cache": use_cache,
            "attention_implementation": attention_implementation,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "block_auto_adjust_ff_dim": block_auto_adjust_ff_dim,
        },
        "vocab_interface": {
            "vocab_size": vocab_size,
            "input_embedding_shape": [vocab_size, hidden_size],
            "output_readout_shape": [vocab_size, hidden_size],
            "token_embedding_readout_tied": False,
        },
        "layers": [layer.as_dict() for layer in layers],
        "evidence_binding": {
            "source_topology_sha256": source_topology_sha256,
            "allocation_evidence_sha256": allocation_evidence_sha256,
            "token_bridge_evidence_sha256": token_bridge_evidence_sha256,
            "ordered_layer_selection_evidence_sha256": [
                layer.selection_evidence_sha256 for layer in layers
            ],
            "evidence_binding_sha256": evidence_binding_sha256,
        },
        "materialization_contract": dict(_SAFE_MATERIALIZATION_CONTRACT),
        "parameter_accounting": accounting.as_dict(),
    }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousPhysicalLayoutDescriptor:
    """Canonical sealed descriptor for one fully heterogeneous LFM2 layout."""

    hidden_size: int
    vocab_size: int
    conv_kernel_size: int
    rope_theta: float
    norm_eps: float
    max_position_embeddings: int
    use_cache: bool
    attention_implementation: str
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    block_auto_adjust_ff_dim: bool
    layers: tuple[Lfm2HeterogeneousLayerPhysicalLayout, ...]
    source_topology_sha256: str
    allocation_evidence_sha256: str
    token_bridge_evidence_sha256: str
    evidence_binding_sha256: str
    accounting: Lfm2HeterogeneousParameterAccounting
    descriptor_sha256: str

    def __post_init__(self) -> None:
        hidden = _positive_int(self.hidden_size, label="hidden_size")
        vocab = _positive_int(self.vocab_size, label="vocab_size")
        _positive_int(self.conv_kernel_size, label="conv_kernel_size")
        _positive_finite_float(self.rope_theta, label="rope_theta")
        _positive_finite_float(self.norm_eps, label="norm_eps")
        _positive_int(
            self.max_position_embeddings,
            label="max_position_embeddings",
        )
        _strict_bool(self.use_cache, label="use_cache")
        attention_backend = _nonempty_string(
            self.attention_implementation,
            label="attention_implementation",
        )
        if attention_backend != "sdpa":
            raise Lfm2DynamicPhysicalLayoutError(
                "Heterogeneous LFM2 layouts require reload-stable SDPA."
            )
        special_token_ids = (
            _nonnegative_int(self.bos_token_id, label="bos_token_id"),
            _nonnegative_int(self.eos_token_id, label="eos_token_id"),
            _nonnegative_int(self.pad_token_id, label="pad_token_id"),
        )
        if any(token_id >= vocab for token_id in special_token_ids):
            raise Lfm2DynamicPhysicalLayoutError(
                "Heterogeneous LFM2 token IDs must be smaller than vocab_size."
            )
        if _strict_bool(
            self.block_auto_adjust_ff_dim,
            label="block_auto_adjust_ff_dim",
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                "Per-layer heterogeneous MLP widths require "
                "block_auto_adjust_ff_dim=False."
            )
        if type(self.layers) is not tuple or not self.layers or any(
            type(layer) is not Lfm2HeterogeneousLayerPhysicalLayout
            for layer in self.layers
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                "A heterogeneous layout requires typed physical layers."
            )
        if tuple(layer.layer_index for layer in self.layers) != tuple(
            range(len(self.layers))
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                "Heterogeneous physical layers must be contiguous and ordered."
            )
        for layer in self.layers:
            if (
                layer.short_conv_writer_rank is not None
                and layer.short_conv_writer_rank
                > min(hidden, int(layer.short_conv_internal_width))
            ):
                raise Lfm2DynamicPhysicalLayoutError(
                    "A factorized short-convolution writer rank cannot exceed "
                    "either dense matrix dimension."
                )
        source = _sha256(
            self.source_topology_sha256,
            label="source_topology_sha256",
        )
        allocation = _sha256(
            self.allocation_evidence_sha256,
            label="allocation_evidence_sha256",
        )
        bridge = _sha256(
            self.token_bridge_evidence_sha256,
            label="token_bridge_evidence_sha256",
        )
        expected_binding = _heterogeneous_evidence_binding(
            source_topology_sha256=source,
            allocation_evidence_sha256=allocation,
            token_bridge_evidence_sha256=bridge,
            layers=self.layers,
        )
        if _sha256(
            self.evidence_binding_sha256,
            label="evidence_binding_sha256",
        ) != expected_binding:
            raise Lfm2DynamicPhysicalLayoutError(
                "Heterogeneous layout evidence binding is invalid."
            )
        expected_accounting = _heterogeneous_parameter_accounting(
            hidden_size=hidden,
            vocab_size=self.vocab_size,
            conv_kernel_size=self.conv_kernel_size,
            layers=self.layers,
        )
        if (
            type(self.accounting) is not Lfm2HeterogeneousParameterAccounting
            or self.accounting != expected_accounting
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                "Heterogeneous physical-layout accounting is invalid."
            )
        if _sha256(
            self.descriptor_sha256,
            label="descriptor_sha256",
        ) != _canonical_sha256(self._body_dict()):
            raise Lfm2DynamicPhysicalLayoutError(
                "Heterogeneous physical-layout seal is invalid."
            )

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def _body_dict(self) -> dict[str, Any]:
        return _heterogeneous_descriptor_body(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            conv_kernel_size=self.conv_kernel_size,
            rope_theta=self.rope_theta,
            norm_eps=self.norm_eps,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=self.use_cache,
            attention_implementation=self.attention_implementation,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            block_auto_adjust_ff_dim=self.block_auto_adjust_ff_dim,
            layers=self.layers,
            source_topology_sha256=self.source_topology_sha256,
            allocation_evidence_sha256=self.allocation_evidence_sha256,
            token_bridge_evidence_sha256=self.token_bridge_evidence_sha256,
            evidence_binding_sha256=self.evidence_binding_sha256,
            accounting=self.accounting,
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._body_dict(), "descriptor_sha256": self.descriptor_sha256}

    def serialize(self) -> bytes:
        """Return the one canonical ASCII JSON encoding of this descriptor."""

        return _canonical_json(self.as_dict())

    @classmethod
    def restore(
        cls,
        value: Lfm2HeterogeneousPhysicalLayoutDescriptor
        | Mapping[str, Any]
        | str
        | bytes,
        *,
        expected_descriptor_sha256: str | None = None,
    ) -> Lfm2HeterogeneousPhysicalLayoutDescriptor:
        return restore_lfm2_heterogeneous_physical_layout(
            value,
            expected_descriptor_sha256=expected_descriptor_sha256,
        )


def build_lfm2_heterogeneous_physical_layout(
    *,
    hidden_size: int,
    vocab_size: int,
    conv_kernel_size: int,
    rope_theta: float,
    norm_eps: float,
    max_position_embeddings: int,
    use_cache: bool,
    attention_implementation: str,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    block_auto_adjust_ff_dim: bool,
    layers: Sequence[Lfm2HeterogeneousLayerPhysicalLayout],
    source_topology_sha256: str,
    allocation_evidence_sha256: str,
    token_bridge_evidence_sha256: str,
    expected_unique_parameter_count: int | None = None,
) -> Lfm2HeterogeneousPhysicalLayoutDescriptor:
    """Validate, account, evidence-bind, and seal one arbitrary layer schedule."""

    hidden = _positive_int(hidden_size, label="hidden_size")
    vocab = _positive_int(vocab_size, label="vocab_size")
    kernel = _positive_int(conv_kernel_size, label="conv_kernel_size")
    theta = _positive_finite_float(rope_theta, label="rope_theta")
    epsilon = _positive_finite_float(norm_eps, label="norm_eps")
    max_positions = _positive_int(
        max_position_embeddings,
        label="max_position_embeddings",
    )
    cache_enabled = _strict_bool(use_cache, label="use_cache")
    attention_backend = _nonempty_string(
        attention_implementation,
        label="attention_implementation",
    )
    if attention_backend != "sdpa":
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous LFM2 layouts require reload-stable SDPA."
        )
    bos = _nonnegative_int(bos_token_id, label="bos_token_id")
    eos = _nonnegative_int(eos_token_id, label="eos_token_id")
    pad = _nonnegative_int(pad_token_id, label="pad_token_id")
    if any(token_id >= vocab for token_id in (bos, eos, pad)):
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous LFM2 token IDs must be smaller than vocab_size."
        )
    auto_adjust = _strict_bool(
        block_auto_adjust_ff_dim,
        label="block_auto_adjust_ff_dim",
    )
    if auto_adjust:
        raise Lfm2DynamicPhysicalLayoutError(
            "Per-layer heterogeneous MLP widths require "
            "block_auto_adjust_ff_dim=False."
        )
    layer_tuple = tuple(layers)
    if not layer_tuple or any(
        type(layer) is not Lfm2HeterogeneousLayerPhysicalLayout
        for layer in layer_tuple
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "A heterogeneous layout requires typed physical layers."
        )
    if tuple(layer.layer_index for layer in layer_tuple) != tuple(
        range(len(layer_tuple))
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous physical layers must be contiguous and ordered."
        )
    for layer in layer_tuple:
        if (
            layer.short_conv_writer_rank is not None
            and layer.short_conv_writer_rank
            > min(hidden, int(layer.short_conv_internal_width))
        ):
            raise Lfm2DynamicPhysicalLayoutError(
                "A factorized short-convolution writer rank cannot exceed "
                "either dense matrix dimension."
            )
    source = _sha256(source_topology_sha256, label="source_topology_sha256")
    allocation = _sha256(
        allocation_evidence_sha256,
        label="allocation_evidence_sha256",
    )
    bridge = _sha256(
        token_bridge_evidence_sha256,
        label="token_bridge_evidence_sha256",
    )
    accounting = _heterogeneous_parameter_accounting(
        hidden_size=hidden,
        vocab_size=vocab,
        conv_kernel_size=kernel,
        layers=layer_tuple,
    )
    if expected_unique_parameter_count is not None and (
        _positive_int(
            expected_unique_parameter_count,
            label="expected_unique_parameter_count",
        )
        != accounting.unique_parameter_elements
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous layout does not match the expected unique-parameter "
            "count."
        )
    binding = _heterogeneous_evidence_binding(
        source_topology_sha256=source,
        allocation_evidence_sha256=allocation,
        token_bridge_evidence_sha256=bridge,
        layers=layer_tuple,
    )
    body = _heterogeneous_descriptor_body(
        hidden_size=hidden,
        vocab_size=vocab,
        conv_kernel_size=kernel,
        rope_theta=theta,
        norm_eps=epsilon,
        max_position_embeddings=max_positions,
        use_cache=cache_enabled,
        attention_implementation=attention_backend,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
        block_auto_adjust_ff_dim=auto_adjust,
        layers=layer_tuple,
        source_topology_sha256=source,
        allocation_evidence_sha256=allocation,
        token_bridge_evidence_sha256=bridge,
        evidence_binding_sha256=binding,
        accounting=accounting,
    )
    return Lfm2HeterogeneousPhysicalLayoutDescriptor(
        hidden_size=hidden,
        vocab_size=vocab,
        conv_kernel_size=kernel,
        rope_theta=theta,
        norm_eps=epsilon,
        max_position_embeddings=max_positions,
        use_cache=cache_enabled,
        attention_implementation=attention_backend,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
        block_auto_adjust_ff_dim=auto_adjust,
        layers=layer_tuple,
        source_topology_sha256=source,
        allocation_evidence_sha256=allocation,
        token_bridge_evidence_sha256=bridge,
        evidence_binding_sha256=binding,
        accounting=accounting,
        descriptor_sha256=_canonical_sha256(body),
    )


def lfm2_expected_heterogeneous_physical_state_shapes(
    descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor | Mapping[str, Any],
) -> Mapping[str, tuple[int, ...]]:
    """Return the exact parameter namespace for a heterogeneous LFM2 child."""

    layout = restore_lfm2_heterogeneous_physical_layout(descriptor)
    hidden = layout.hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (layout.vocab_size, hidden),
        "model.embedding_norm.weight": (hidden,),
        "lm_head.weight": (layout.vocab_size, hidden),
    }
    for layer in layout.layers:
        prefix = f"model.layers.{layer.layer_index}."
        shapes[prefix + "operator_norm.weight"] = (hidden,)
        shapes[prefix + "ffn_norm.weight"] = (hidden,)
        intermediate = layer.mlp_intermediate_size
        shapes[prefix + "feed_forward.w1.weight"] = (intermediate, hidden)
        shapes[prefix + "feed_forward.w3.weight"] = (intermediate, hidden)
        shapes[prefix + "feed_forward.w2.weight"] = (hidden, intermediate)
        if layer.layer_type == "conv":
            assert layer.short_conv_internal_width is not None
            internal = layer.short_conv_internal_width
            shapes[prefix + "conv.in_proj.weight"] = (3 * internal, hidden)
            shapes[prefix + "conv.conv.weight"] = (
                internal,
                1,
                layout.conv_kernel_size,
            )
            if layer.short_conv_writer_rank is None:
                shapes[prefix + "conv.out_proj.weight"] = (hidden, internal)
            else:
                rank = layer.short_conv_writer_rank
                shapes[prefix + "conv.out_proj.0.weight"] = (rank, internal)
                shapes[prefix + "conv.out_proj.1.weight"] = (hidden, rank)
        else:
            assert layer.query_heads is not None
            assert layer.key_value_heads is not None
            assert layer.head_dim is not None
            query_width = layer.query_heads * layer.head_dim
            key_value_width = layer.key_value_heads * layer.head_dim
            shapes[prefix + "self_attn.q_proj.weight"] = (query_width, hidden)
            shapes[prefix + "self_attn.k_proj.weight"] = (key_value_width, hidden)
            shapes[prefix + "self_attn.v_proj.weight"] = (key_value_width, hidden)
            shapes[prefix + "self_attn.out_proj.weight"] = (hidden, query_width)
            shapes[prefix + "self_attn.q_layernorm.weight"] = (layer.head_dim,)
            shapes[prefix + "self_attn.k_layernorm.weight"] = (layer.head_dim,)
    if sum(_shape_elements(shape) for shape in shapes.values()) != (
        layout.accounting.unique_parameter_elements
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous physical namespace disagrees with parameter accounting."
        )
    return MappingProxyType(shapes)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Lfm2DynamicPhysicalLayoutError(
                f"Serialized heterogeneous layout repeats field {key!r}."
            )
        value[key] = item
    return value


def restore_lfm2_heterogeneous_physical_layout(
    value: Lfm2HeterogeneousPhysicalLayoutDescriptor
    | Mapping[str, Any]
    | str
    | bytes,
    *,
    expected_descriptor_sha256: str | None = None,
) -> Lfm2HeterogeneousPhysicalLayoutDescriptor:
    """Restore only a canonical, sealed, independently reproducible layout."""

    canonical_input: bytes | None = None
    if isinstance(value, Lfm2HeterogeneousPhysicalLayoutDescriptor):
        raw: Any = value.as_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    elif isinstance(value, (str, bytes)):
        try:
            canonical_input = value.encode("ascii") if isinstance(value, str) else value
            raw = json.loads(
                canonical_input.decode("ascii"),
                object_pairs_hook=_strict_json_object,
            )
        except Lfm2DynamicPhysicalLayoutError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Lfm2DynamicPhysicalLayoutError(
                "Serialized heterogeneous layout is not valid ASCII JSON."
            ) from exc
    else:
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous physical layout must be a descriptor, mapping, or JSON."
        )
    if not isinstance(raw, Mapping):
        raise Lfm2DynamicPhysicalLayoutError(
            "Serialized heterogeneous layout must decode to one mapping."
        )
    serialized = dict(raw)
    if canonical_input is not None and _canonical_json(serialized) != canonical_input:
        raise Lfm2DynamicPhysicalLayoutError(
            "Serialized heterogeneous layout is not in canonical JSON form."
        )
    if set(serialized) != {
        "schema",
        "status",
        "model_type",
        "topology",
        "runtime_semantics",
        "vocab_interface",
        "layers",
        "evidence_binding",
        "materialization_contract",
        "parameter_accounting",
        "descriptor_sha256",
    }:
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous physical-layout fields are not canonical."
        )
    supplied = _sha256(
        serialized.pop("descriptor_sha256"),
        label="descriptor_sha256",
    )
    if _canonical_sha256(serialized) != supplied:
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous physical-layout seal is invalid."
        )
    if expected_descriptor_sha256 is not None and supplied != _sha256(
        expected_descriptor_sha256,
        label="expected_descriptor_sha256",
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous physical layout differs from its external seal."
        )
    topology = serialized.get("topology")
    runtime_semantics = serialized.get("runtime_semantics")
    interface = serialized.get("vocab_interface")
    evidence = serialized.get("evidence_binding")
    layer_rows = serialized.get("layers")
    materialization_contract = serialized.get("materialization_contract")
    if materialization_contract != _SAFE_MATERIALIZATION_CONTRACT:
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous materialization contract is unsafe."
        )
    if (
        serialized.get("schema") != LFM2_HETEROGENEOUS_PHYSICAL_LAYOUT_SCHEMA
        or serialized.get("status")
        != "authenticated-heterogeneous-dense-fp32-blank-child-layout"
        or serialized.get("model_type") != "lfm2"
        or not isinstance(topology, Mapping)
        or set(topology)
        != {
            "constant_external_hidden_size",
            "conv_kernel_size",
            "num_hidden_layers",
            "rope_parameters",
        }
        or not isinstance(runtime_semantics, Mapping)
        or set(runtime_semantics)
        != {
            "norm_eps",
            "max_position_embeddings",
            "use_cache",
            "attention_implementation",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "block_auto_adjust_ff_dim",
        }
        or not isinstance(interface, Mapping)
        or set(interface)
        != {
            "vocab_size",
            "input_embedding_shape",
            "output_readout_shape",
            "token_embedding_readout_tied",
        }
        or interface.get("token_embedding_readout_tied") is not False
        or not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "source_topology_sha256",
            "allocation_evidence_sha256",
            "token_bridge_evidence_sha256",
            "ordered_layer_selection_evidence_sha256",
            "evidence_binding_sha256",
        }
        or not isinstance(layer_rows, list)
        or not isinstance(serialized.get("parameter_accounting"), Mapping)
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous physical-layout topology or evidence is malformed."
        )
    layers: list[Lfm2HeterogeneousLayerPhysicalLayout] = []
    expected_layer_fields = {
        "layer_index",
        "layer_type",
        "mlp_intermediate_size",
        "selection_evidence_sha256",
        "short_conv_internal_width",
        "short_conv_writer_rank",
        "query_heads",
        "key_value_heads",
        "head_dim",
    }
    for row in layer_rows:
        if not isinstance(row, Mapping) or set(row) != expected_layer_fields:
            raise Lfm2DynamicPhysicalLayoutError(
                "A heterogeneous physical layer is malformed."
            )
        layers.append(
            Lfm2HeterogeneousLayerPhysicalLayout(
                layer_index=row.get("layer_index"),
                layer_type=row.get("layer_type"),
                mlp_intermediate_size=row.get("mlp_intermediate_size"),
                selection_evidence_sha256=row.get("selection_evidence_sha256"),
                short_conv_internal_width=row.get("short_conv_internal_width"),
                short_conv_writer_rank=row.get("short_conv_writer_rank"),
                query_heads=row.get("query_heads"),
                key_value_heads=row.get("key_value_heads"),
                head_dim=row.get("head_dim"),
            )
        )
    restored_layers = tuple(layers)
    expected_evidence_binding = _heterogeneous_evidence_binding(
        source_topology_sha256=_sha256(
            evidence.get("source_topology_sha256"),
            label="source_topology_sha256",
        ),
        allocation_evidence_sha256=_sha256(
            evidence.get("allocation_evidence_sha256"),
            label="allocation_evidence_sha256",
        ),
        token_bridge_evidence_sha256=_sha256(
            evidence.get("token_bridge_evidence_sha256"),
            label="token_bridge_evidence_sha256",
        ),
        layers=restored_layers,
    )
    if evidence.get("evidence_binding_sha256") != expected_evidence_binding:
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous layout evidence binding is invalid."
        )
    accounting = serialized["parameter_accounting"]
    rope = topology.get("rope_parameters")
    if (
        not isinstance(rope, Mapping)
        or set(rope) != {"rope_type", "rope_theta"}
        or rope.get("rope_type") != "default"
    ):
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous layout default-RoPE semantics are malformed."
        )
    rebuilt = build_lfm2_heterogeneous_physical_layout(
        hidden_size=topology.get("constant_external_hidden_size"),
        vocab_size=interface.get("vocab_size"),
        conv_kernel_size=topology.get("conv_kernel_size"),
        rope_theta=rope.get("rope_theta"),
        norm_eps=runtime_semantics.get("norm_eps"),
        max_position_embeddings=runtime_semantics.get(
            "max_position_embeddings"
        ),
        use_cache=runtime_semantics.get("use_cache"),
        attention_implementation=runtime_semantics.get(
            "attention_implementation"
        ),
        bos_token_id=runtime_semantics.get("bos_token_id"),
        eos_token_id=runtime_semantics.get("eos_token_id"),
        pad_token_id=runtime_semantics.get("pad_token_id"),
        block_auto_adjust_ff_dim=runtime_semantics.get(
            "block_auto_adjust_ff_dim"
        ),
        layers=restored_layers,
        source_topology_sha256=evidence.get("source_topology_sha256"),
        allocation_evidence_sha256=evidence.get("allocation_evidence_sha256"),
        token_bridge_evidence_sha256=evidence.get(
            "token_bridge_evidence_sha256"
        ),
        expected_unique_parameter_count=accounting.get(
            "unique_parameter_elements"
        ),
    )
    supplied_mapping = {**serialized, "descriptor_sha256": supplied}
    if rebuilt.as_dict() != supplied_mapping:
        raise Lfm2DynamicPhysicalLayoutError(
            "Heterogeneous layout is sealed but not a canonical reconstruction."
        )
    return rebuilt


__all__ = [
    "LFM2_DYNAMIC_PHYSICAL_AUDIT_SCHEMA",
    "LFM2_DYNAMIC_PHYSICAL_LAYOUT_SCHEMA",
    "LFM2_HETEROGENEOUS_PHYSICAL_LAYOUT_SCHEMA",
    "LFM2_RAGGED_MLP_RUNTIME_SCHEMA",
    "Lfm2DenseFp32Accounting",
    "Lfm2DynamicPhysicalLayoutDescriptor",
    "Lfm2DynamicPhysicalLayoutError",
    "Lfm2HeterogeneousLayerParameterAccounting",
    "Lfm2HeterogeneousLayerPhysicalLayout",
    "Lfm2HeterogeneousParameterAccounting",
    "Lfm2HeterogeneousPhysicalLayoutDescriptor",
    "Lfm2LayerPhysicalLayout",
    "Lfm2PhysicalLayerType",
    "Lfm2RaggedSwiGLURuntime",
    "Lfm2ShortConvCacheContract",
    "authenticate_lfm2_dynamic_physical_layout",
    "authenticate_lfm2_physical_materialization",
    "build_canonical_lfm2_exact4x_protected_stem_proposal",
    "build_lfm2_dynamic_physical_layout",
    "build_lfm2_heterogeneous_physical_layout",
    "lfm2_expected_heterogeneous_physical_state_shapes",
    "lfm2_expected_physical_state_shapes",
    "restore_lfm2_heterogeneous_physical_layout",
]
