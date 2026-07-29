"""Pure exact selector for heterogeneous physical LFM2 child layouts.

The selector is the bounded policy seam between inspected source circuits and
the physical-layout/materialization code.  It performs no tensor work and no
learning.  Callers provide authenticated physical alternatives for every
source layer; the architecture-discovered exact solver chooses one alternative
per layer under the requested whole-model parameter count.

Layer groups and their causal order come only from :class:`ModelCircuitGraph`.
Paths and numeric layer identities are evidence, never policy selectors.  The
resulting allocation decision is sealed before the physical descriptor is
built, avoiding a descriptor/receipt hash cycle.  A final receipt then binds
that allocation decision to the descriptor seal.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    ModelCircuitGraph,
    canonical_json_bytes,
    canonical_sha256,
)
from .dynamic_fold_budget import (
    ArchitectureDiscoveredBudgetProblem,
    ArchitectureDiscoveredBudgetSelection,
    DiscoveredComponentChoiceGroup,
    DiscoveredComponentTopologyOption,
    DynamicFoldBudgetError,
    DynamicFoldBudgetUnattainable,
    RetainedParameterAllocation,
    residual_attention_component_protection_rules,
    solve_architecture_discovered_budget,
)
from .hf_lfm2_dynamic_physical_layout import (
    Lfm2HeterogeneousLayerPhysicalLayout,
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
    build_lfm2_heterogeneous_physical_layout,
    restore_lfm2_heterogeneous_physical_layout,
)


LFM2_HETEROGENEOUS_LAYER_OPTION_SCHEMA = (
    "neural-forge-lfm2-heterogeneous-layer-option-v1"
)
LFM2_HETEROGENEOUS_TARGET_PINS_SCHEMA = "neural-forge-lfm2-heterogeneous-target-pins-v1"
LFM2_HETEROGENEOUS_ALLOCATION_DECISION_SCHEMA = (
    "neural-forge-lfm2-heterogeneous-allocation-decision-v1"
)
LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-heterogeneous-selection-receipt-v1"
)

Lfm2SelectableLayerType = Literal["conv", "full_attention"]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2HeterogeneousLayoutSelectionError(ValueError):
    """The graph, evidence, target contract, or option set is invalid."""


class Lfm2HeterogeneousLayoutUnattainable(Lfm2HeterogeneousLayoutSelectionError):
    """No supplied physical layout exactly reaches the requested count."""


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2HeterogeneousLayoutSelectionError(
            f"{label} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2HeterogeneousLayoutSelectionError(
            f"{label} must be a non-negative integer."
        )
    return value


def _positive_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise Lfm2HeterogeneousLayoutSelectionError(
            f"{label} must be finite and positive."
        )
    return float(value)


def _risk(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Lfm2HeterogeneousLayoutSelectionError(
            "structural_risk must be finite and non-negative."
        )
    return float(value)


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Lfm2HeterogeneousLayoutSelectionError(f"{label} must be boolean.")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2HeterogeneousLayoutSelectionError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2HeterogeneousLayoutSelectionError(
            "Selection evidence is not canonical JSON."
        ) from exc


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Lfm2HeterogeneousLayoutSelectionError(
                f"Serialized selection evidence repeats field {key!r}."
            )
        result[key] = value
    return result


def _mapping(value: Mapping[str, Any] | str | bytes, label: str) -> dict[str, Any]:
    canonical_input: bytes | None = None
    if isinstance(value, Mapping):
        result = dict(value)
    elif isinstance(value, (str, bytes)):
        canonical_input = value.encode("ascii") if isinstance(value, str) else value
        try:
            result = json.loads(
                canonical_input.decode("ascii"),
                object_pairs_hook=_strict_json_object,
            )
        except Lfm2HeterogeneousLayoutSelectionError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Lfm2HeterogeneousLayoutSelectionError(
                f"{label} is not valid ASCII JSON."
            ) from exc
        if not isinstance(result, dict):
            raise Lfm2HeterogeneousLayoutSelectionError(
                f"{label} must decode to one mapping."
            )
        if _canonical_bytes(result) != canonical_input:
            raise Lfm2HeterogeneousLayoutSelectionError(
                f"{label} is not canonical JSON."
            )
    else:  # pragma: no cover - narrowed by public type annotations
        raise Lfm2HeterogeneousLayoutSelectionError(
            f"{label} must be a mapping or canonical JSON."
        )
    return result


def _node_attributes(node: CircuitNode) -> dict[str, Any]:
    return dict(node.attributes)


@dataclass(frozen=True, slots=True)
class Lfm2DiscoveredSourceLayer:
    """One causal source-layer group discovered without path policy."""

    ordinal: int
    graph_layer_index: int
    layer_type: Lfm2SelectableLayerType
    source_node_ids: tuple[str, ...]
    parameterized_node_ids: tuple[str, ...]
    operator_node_id: str
    mlp_node_id: str
    first_node_order: int
    last_node_order: int
    source_layer_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "graph_layer_index": self.graph_layer_index,
            "layer_type": self.layer_type,
            "source_node_ids": list(self.source_node_ids),
            "parameterized_node_ids": list(self.parameterized_node_ids),
            "operator_node_id": self.operator_node_id,
            "mlp_node_id": self.mlp_node_id,
            "first_node_order": self.first_node_order,
            "last_node_order": self.last_node_order,
            "source_layer_sha256": self.source_layer_sha256,
        }


def _discover_lfm2_source_layers(
    source_graph: ModelCircuitGraph,
) -> tuple[Lfm2DiscoveredSourceLayer, ...]:
    if not isinstance(source_graph, ModelCircuitGraph):
        raise Lfm2HeterogeneousLayoutSelectionError(
            "source_graph must be an inspected ModelCircuitGraph."
        )
    if source_graph.family != "lfm2":
        raise Lfm2HeterogeneousLayoutSelectionError(
            "The heterogeneous LFM2 selector requires an inspected LFM2 graph."
        )
    layer_indices = sorted(
        {
            node.layer_index
            for node in source_graph.nodes
            if node.layer_index is not None
        }
    )
    if not layer_indices:
        raise Lfm2HeterogeneousLayoutSelectionError(
            "The inspected LFM2 graph contains no source layers."
        )
    discovered_rows: list[tuple[int, int, tuple[CircuitNode, ...]]] = []
    for graph_index in layer_indices:
        assert graph_index is not None
        nodes = tuple(
            sorted(
                source_graph.layer_nodes(graph_index),
                key=lambda node: (node.order, node.node_id),
            )
        )
        if not nodes:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "An inspected LFM2 layer is empty."
            )
        discovered_rows.append((nodes[0].order, graph_index, nodes))
    discovered_rows.sort(key=lambda row: (row[0], row[1]))
    if any(
        left[2][-1].order >= right[2][0].order
        for left, right in zip(discovered_rows, discovered_rows[1:])
    ):
        raise Lfm2HeterogeneousLayoutSelectionError(
            "Inspected LFM2 layer node orders overlap or interleave."
        )

    result = []
    allowed_parameterized_kinds = {
        CircuitKind.NORM.value,
        CircuitKind.ATTENTION.value,
        CircuitKind.SHORT_CONV.value,
        CircuitKind.GATED_MLP.value,
    }
    for ordinal, (_first, graph_index, nodes) in enumerate(discovered_rows):
        attention = tuple(
            node for node in nodes if node.kind == CircuitKind.ATTENTION.value
        )
        convolution = tuple(
            node for node in nodes if node.kind == CircuitKind.SHORT_CONV.value
        )
        mlps = tuple(node for node in nodes if node.kind == CircuitKind.GATED_MLP.value)
        norms = tuple(node for node in nodes if node.kind == CircuitKind.NORM.value)
        parameterized = tuple(node for node in nodes if node.parameters)
        if bool(attention) == bool(convolution) or len(attention + convolution) != 1:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Every discovered LFM2 source layer must contain exactly one "
                "attention or short-convolution operator."
            )
        if len(mlps) != 1 or len(norms) != 2:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Every discovered LFM2 source layer must contain one gated MLP "
                "and two residual norms."
            )
        unsupported = tuple(
            node.node_id
            for node in parameterized
            if node.kind not in allowed_parameterized_kinds
        )
        if unsupported:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "The physical LFM2 selector cannot represent parameterized "
                f"source layer nodes {unsupported}."
            )
        operator = (attention or convolution)[0]
        layer_type: Lfm2SelectableLayerType = "full_attention" if attention else "conv"
        source_node_ids = tuple(node.node_id for node in nodes)
        parameterized_node_ids = tuple(node.node_id for node in parameterized)
        body = {
            "schema": "neural-forge-lfm2-discovered-source-layer-v1",
            "source_graph_fingerprint": source_graph.fingerprint,
            "causal_ordinal": ordinal,
            "graph_layer_index": graph_index,
            "layer_type": layer_type,
            "ordered_nodes": [
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "order": node.order,
                    "parameter_paths": list(node.parameter_paths),
                }
                for node in nodes
            ],
        }
        result.append(
            Lfm2DiscoveredSourceLayer(
                ordinal=ordinal,
                graph_layer_index=graph_index,
                layer_type=layer_type,
                source_node_ids=source_node_ids,
                parameterized_node_ids=parameterized_node_ids,
                operator_node_id=operator.node_id,
                mlp_node_id=mlps[0].node_id,
                first_node_order=nodes[0].order,
                last_node_order=nodes[-1].order,
                source_layer_sha256=canonical_sha256(body),
            )
        )
    return tuple(result)


def discover_lfm2_source_layers(
    source_graph: ModelCircuitGraph,
) -> tuple[Lfm2DiscoveredSourceLayer, ...]:
    """Return the deterministic option-generation keys for an inspected graph."""

    return _discover_lfm2_source_layers(source_graph)


def _layer_option_body(
    *,
    source_layer_sha256: str,
    layer_type: Lfm2SelectableLayerType,
    mlp_intermediate_size: int,
    selection_evidence_sha256: str,
    structural_risk: float,
    short_conv_internal_width: int | None,
    short_conv_writer_rank: int | None,
    query_heads: int | None,
    key_value_heads: int | None,
    head_dim: int | None,
) -> dict[str, Any]:
    return {
        "schema": LFM2_HETEROGENEOUS_LAYER_OPTION_SCHEMA,
        "source_layer_sha256": source_layer_sha256,
        "layer_type": layer_type,
        "mlp_intermediate_size": mlp_intermediate_size,
        "selection_evidence_sha256": selection_evidence_sha256,
        "structural_risk": structural_risk,
        "short_conv_internal_width": short_conv_internal_width,
        "short_conv_writer_rank": short_conv_writer_rank,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
    }


@dataclass(frozen=True, slots=True)
class Lfm2AuthenticatedLayerPhysicalOption:
    """One sealed physical alternative for one discovered source layer."""

    source_layer_sha256: str
    layer_type: Lfm2SelectableLayerType
    mlp_intermediate_size: int
    selection_evidence_sha256: str
    structural_risk: float
    short_conv_internal_width: int | None
    short_conv_writer_rank: int | None
    query_heads: int | None
    key_value_heads: int | None
    head_dim: int | None
    option_sha256: str

    def __post_init__(self) -> None:
        source = _sha256(self.source_layer_sha256, "source_layer_sha256")
        intermediate = _positive_int(
            self.mlp_intermediate_size,
            "mlp_intermediate_size",
        )
        evidence = _sha256(
            self.selection_evidence_sha256,
            "selection_evidence_sha256",
        )
        risk = _risk(self.structural_risk)
        if self.layer_type == "conv":
            internal = _positive_int(
                self.short_conv_internal_width,
                "short_conv_internal_width",
            )
            rank = self.short_conv_writer_rank
            if rank is not None:
                rank = _positive_int(rank, "short_conv_writer_rank")
                if rank > internal:
                    raise Lfm2HeterogeneousLayoutSelectionError(
                        "A factor writer rank cannot exceed its internal width."
                    )
            if any(
                value is not None
                for value in (self.query_heads, self.key_value_heads, self.head_dim)
            ):
                raise Lfm2HeterogeneousLayoutSelectionError(
                    "A short-convolution option cannot carry attention geometry."
                )
        elif self.layer_type == "full_attention":
            if (
                self.short_conv_internal_width is not None
                or self.short_conv_writer_rank is not None
            ):
                raise Lfm2HeterogeneousLayoutSelectionError(
                    "An attention option cannot carry short-convolution geometry."
                )
            query = _positive_int(self.query_heads, "query_heads")
            key_value = _positive_int(self.key_value_heads, "key_value_heads")
            dimension = _positive_int(self.head_dim, "head_dim")
            if query % key_value:
                raise Lfm2HeterogeneousLayoutSelectionError(
                    "query_heads must be divisible by key_value_heads."
                )
            if dimension % 2:
                raise Lfm2HeterogeneousLayoutSelectionError(
                    "Default RoPE requires an even head_dim."
                )
        else:
            raise Lfm2HeterogeneousLayoutSelectionError(
                f"Unsupported LFM2 physical layer type {self.layer_type!r}."
            )
        body = _layer_option_body(
            source_layer_sha256=source,
            layer_type=self.layer_type,
            mlp_intermediate_size=intermediate,
            selection_evidence_sha256=evidence,
            structural_risk=risk,
            short_conv_internal_width=self.short_conv_internal_width,
            short_conv_writer_rank=self.short_conv_writer_rank,
            query_heads=self.query_heads,
            key_value_heads=self.key_value_heads,
            head_dim=self.head_dim,
        )
        if _sha256(self.option_sha256, "option_sha256") != canonical_sha256(body):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "LFM2 layer physical-option seal is invalid."
            )

    def _body_dict(self) -> dict[str, Any]:
        return _layer_option_body(
            source_layer_sha256=self.source_layer_sha256,
            layer_type=self.layer_type,
            mlp_intermediate_size=self.mlp_intermediate_size,
            selection_evidence_sha256=self.selection_evidence_sha256,
            structural_risk=self.structural_risk,
            short_conv_internal_width=self.short_conv_internal_width,
            short_conv_writer_rank=self.short_conv_writer_rank,
            query_heads=self.query_heads,
            key_value_heads=self.key_value_heads,
            head_dim=self.head_dim,
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._body_dict(), "option_sha256": self.option_sha256}

    def serialize(self) -> bytes:
        return _canonical_bytes(self.as_dict())

    @classmethod
    def restore(
        cls,
        value: Lfm2AuthenticatedLayerPhysicalOption | Mapping[str, Any] | str | bytes,
    ) -> Lfm2AuthenticatedLayerPhysicalOption:
        if isinstance(value, cls):
            return cls(
                **{
                    key: item
                    for key, item in value.as_dict().items()
                    if key != "schema"
                }
            )
        raw = _mapping(value, "LFM2 layer physical option")
        expected = set(
            _layer_option_body(
                source_layer_sha256="0" * 64,
                layer_type="conv",
                mlp_intermediate_size=1,
                selection_evidence_sha256="0" * 64,
                structural_risk=0.0,
                short_conv_internal_width=1,
                short_conv_writer_rank=None,
                query_heads=None,
                key_value_heads=None,
                head_dim=None,
            )
        ) | {"option_sha256"}
        if set(raw) != expected or raw.pop("schema", None) != (
            LFM2_HETEROGENEOUS_LAYER_OPTION_SCHEMA
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "LFM2 layer physical-option fields are not canonical."
            )
        return cls(**raw)


def _build_layer_option(**values: Any) -> Lfm2AuthenticatedLayerPhysicalOption:
    body = _layer_option_body(**values)
    return Lfm2AuthenticatedLayerPhysicalOption(
        **{key: value for key, value in body.items() if key != "schema"},
        option_sha256=canonical_sha256(body),
    )


def build_lfm2_short_conv_layer_option(
    *,
    source_layer_sha256: str,
    mlp_intermediate_size: int,
    internal_width: int,
    selection_evidence_sha256: str,
    structural_risk: float = 0.0,
    writer_rank: int | None = None,
) -> Lfm2AuthenticatedLayerPhysicalOption:
    """Seal a dense- or factor-writer short-convolution alternative."""

    return _build_layer_option(
        source_layer_sha256=_sha256(
            source_layer_sha256,
            "source_layer_sha256",
        ),
        layer_type="conv",
        mlp_intermediate_size=_positive_int(
            mlp_intermediate_size,
            "mlp_intermediate_size",
        ),
        selection_evidence_sha256=_sha256(
            selection_evidence_sha256,
            "selection_evidence_sha256",
        ),
        structural_risk=_risk(structural_risk),
        short_conv_internal_width=_positive_int(internal_width, "internal_width"),
        short_conv_writer_rank=writer_rank,
        query_heads=None,
        key_value_heads=None,
        head_dim=None,
    )


def build_lfm2_attention_layer_option(
    *,
    source_layer_sha256: str,
    mlp_intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    selection_evidence_sha256: str,
    structural_risk: float = 0.0,
) -> Lfm2AuthenticatedLayerPhysicalOption:
    """Seal one residual-attention-aware GQA geometry alternative."""

    return _build_layer_option(
        source_layer_sha256=_sha256(
            source_layer_sha256,
            "source_layer_sha256",
        ),
        layer_type="full_attention",
        mlp_intermediate_size=_positive_int(
            mlp_intermediate_size,
            "mlp_intermediate_size",
        ),
        selection_evidence_sha256=_sha256(
            selection_evidence_sha256,
            "selection_evidence_sha256",
        ),
        structural_risk=_risk(structural_risk),
        short_conv_internal_width=None,
        short_conv_writer_rank=None,
        query_heads=_positive_int(query_heads, "query_heads"),
        key_value_heads=_positive_int(key_value_heads, "key_value_heads"),
        head_dim=_positive_int(head_dim, "head_dim"),
    )


def _target_pins_body(
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
    target_vocab_evidence_sha256: str,
    runtime_evidence_sha256: str,
    token_bridge_evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": LFM2_HETEROGENEOUS_TARGET_PINS_SCHEMA,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "conv_kernel_size": conv_kernel_size,
        "runtime_semantics": {
            "rope_theta": rope_theta,
            "norm_eps": norm_eps,
            "max_position_embeddings": max_position_embeddings,
            "use_cache": use_cache,
            "attention_implementation": attention_implementation,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "block_auto_adjust_ff_dim": block_auto_adjust_ff_dim,
        },
        "evidence": {
            "target_vocab_evidence_sha256": target_vocab_evidence_sha256,
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "token_bridge_evidence_sha256": token_bridge_evidence_sha256,
        },
    }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousTargetPins:
    """Authenticated target vocabulary, runtime, and bridge contract."""

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
    target_vocab_evidence_sha256: str
    runtime_evidence_sha256: str
    token_bridge_evidence_sha256: str
    pins_sha256: str

    def __post_init__(self) -> None:
        hidden = _positive_int(self.hidden_size, "hidden_size")
        vocab = _positive_int(self.vocab_size, "vocab_size")
        kernel = _positive_int(self.conv_kernel_size, "conv_kernel_size")
        theta = _positive_float(self.rope_theta, "rope_theta")
        epsilon = _positive_float(self.norm_eps, "norm_eps")
        max_positions = _positive_int(
            self.max_position_embeddings,
            "max_position_embeddings",
        )
        cache = _strict_bool(self.use_cache, "use_cache")
        if self.attention_implementation != "sdpa":
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous LFM2 target pins require SDPA."
            )
        bos = _nonnegative_int(self.bos_token_id, "bos_token_id")
        eos = _nonnegative_int(self.eos_token_id, "eos_token_id")
        pad = _nonnegative_int(self.pad_token_id, "pad_token_id")
        if any(value >= vocab for value in (bos, eos, pad)):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Target special-token IDs must fit the target vocabulary."
            )
        auto_adjust = _strict_bool(
            self.block_auto_adjust_ff_dim,
            "block_auto_adjust_ff_dim",
        )
        if auto_adjust:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous MLP widths require block_auto_adjust_ff_dim=False."
            )
        vocab_evidence = _sha256(
            self.target_vocab_evidence_sha256,
            "target_vocab_evidence_sha256",
        )
        runtime_evidence = _sha256(
            self.runtime_evidence_sha256,
            "runtime_evidence_sha256",
        )
        bridge_evidence = _sha256(
            self.token_bridge_evidence_sha256,
            "token_bridge_evidence_sha256",
        )
        body = _target_pins_body(
            hidden_size=hidden,
            vocab_size=vocab,
            conv_kernel_size=kernel,
            rope_theta=theta,
            norm_eps=epsilon,
            max_position_embeddings=max_positions,
            use_cache=cache,
            attention_implementation=self.attention_implementation,
            bos_token_id=bos,
            eos_token_id=eos,
            pad_token_id=pad,
            block_auto_adjust_ff_dim=auto_adjust,
            target_vocab_evidence_sha256=vocab_evidence,
            runtime_evidence_sha256=runtime_evidence,
            token_bridge_evidence_sha256=bridge_evidence,
        )
        if _sha256(self.pins_sha256, "pins_sha256") != canonical_sha256(body):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous target-pins seal is invalid."
            )

    def _body_dict(self) -> dict[str, Any]:
        return _target_pins_body(
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
            target_vocab_evidence_sha256=self.target_vocab_evidence_sha256,
            runtime_evidence_sha256=self.runtime_evidence_sha256,
            token_bridge_evidence_sha256=self.token_bridge_evidence_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._body_dict(), "pins_sha256": self.pins_sha256}

    def serialize(self) -> bytes:
        return _canonical_bytes(self.as_dict())

    @classmethod
    def restore(
        cls,
        value: Lfm2HeterogeneousTargetPins | Mapping[str, Any] | str | bytes,
    ) -> Lfm2HeterogeneousTargetPins:
        if isinstance(value, cls):
            return cls.restore(value.as_dict())
        raw = _mapping(value, "LFM2 heterogeneous target pins")
        if (
            set(raw)
            != {
                "schema",
                "hidden_size",
                "vocab_size",
                "conv_kernel_size",
                "runtime_semantics",
                "evidence",
                "pins_sha256",
            }
            or raw.pop("schema", None) != LFM2_HETEROGENEOUS_TARGET_PINS_SCHEMA
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous target-pins fields are not canonical."
            )
        runtime = raw.pop("runtime_semantics", None)
        evidence = raw.pop("evidence", None)
        if not isinstance(runtime, Mapping) or set(runtime) != {
            "rope_theta",
            "norm_eps",
            "max_position_embeddings",
            "use_cache",
            "attention_implementation",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "block_auto_adjust_ff_dim",
        }:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous target runtime fields are not canonical."
            )
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "target_vocab_evidence_sha256",
            "runtime_evidence_sha256",
            "token_bridge_evidence_sha256",
        }:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous target evidence fields are not canonical."
            )
        return cls(**raw, **dict(runtime), **dict(evidence))


def build_lfm2_heterogeneous_target_pins(
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
    target_vocab_evidence_sha256: str,
    runtime_evidence_sha256: str,
    token_bridge_evidence_sha256: str,
) -> Lfm2HeterogeneousTargetPins:
    body = _target_pins_body(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        conv_kernel_size=conv_kernel_size,
        rope_theta=rope_theta,
        norm_eps=norm_eps,
        max_position_embeddings=max_position_embeddings,
        use_cache=use_cache,
        attention_implementation=attention_implementation,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        block_auto_adjust_ff_dim=block_auto_adjust_ff_dim,
        target_vocab_evidence_sha256=target_vocab_evidence_sha256,
        runtime_evidence_sha256=runtime_evidence_sha256,
        token_bridge_evidence_sha256=token_bridge_evidence_sha256,
    )
    return Lfm2HeterogeneousTargetPins(
        **{
            key: value
            for key, value in body.items()
            if key not in {"schema", "runtime_semantics", "evidence"}
        },
        **body["runtime_semantics"],
        **body["evidence"],
        pins_sha256=canonical_sha256(body),
    )


def _role_allocations(
    node: CircuitNode,
    target_capacity: int,
) -> tuple[RetainedParameterAllocation, ...]:
    """Distribute component capacity over inspected roles without path policy."""

    counts: dict[str, int] = {}
    for parameter in node.parameters:
        counts[parameter.role] = counts.get(parameter.role, 0) + parameter.numel
    source_total = sum(counts.values())
    if not source_total:
        return ()
    retained_total = min(source_total, target_capacity)
    floors = {
        role: count * retained_total // source_total for role, count in counts.items()
    }
    remainder = retained_total - sum(floors.values())
    priority = sorted(
        counts,
        key=lambda role: (
            -(counts[role] * retained_total % source_total),
            role,
        ),
    )
    for role in priority[:remainder]:
        floors[role] += 1
    return tuple(
        RetainedParameterAllocation(node.node_id, role, floors[role])
        for role in sorted(floors)
    )


def _component_parameter_counts(
    option: Lfm2AuthenticatedLayerPhysicalOption,
    pins: Lfm2HeterogeneousTargetPins,
) -> tuple[int, int, int]:
    hidden = pins.hidden_size
    norms = 2 * hidden
    mlp = 3 * hidden * option.mlp_intermediate_size
    if option.layer_type == "conv":
        assert option.short_conv_internal_width is not None
        internal = option.short_conv_internal_width
        operator = 3 * internal * hidden + internal * pins.conv_kernel_size
        if option.short_conv_writer_rank is None:
            operator += hidden * internal
        else:
            operator += option.short_conv_writer_rank * (hidden + internal)
    else:
        assert option.query_heads is not None
        assert option.key_value_heads is not None
        assert option.head_dim is not None
        operator = (
            2 * hidden * option.head_dim * (option.query_heads + option.key_value_heads)
            + 2 * option.head_dim
        )
    return norms, mlp, operator


def _option_geometry_key(
    option: Lfm2AuthenticatedLayerPhysicalOption,
) -> tuple[Any, ...]:
    return (
        option.mlp_intermediate_size,
        option.short_conv_internal_width,
        option.short_conv_writer_rank,
        option.query_heads,
        option.key_value_heads,
        option.head_dim,
    )


def _allocation_decision_body(
    *,
    source_graph_fingerprint: str,
    source_structural_topology_fingerprint: str,
    source_layer_order_sha256: str,
    target_pins_sha256: str,
    target_vocab_evidence_sha256: str,
    runtime_evidence_sha256: str,
    token_bridge_evidence_sha256: str,
    exact_target_unique_parameters: int,
    budget_problem_fingerprint: str,
    budget_selection_fingerprint: str,
    exact_topology_problem_fingerprint: str,
    exact_topology_selection_fingerprint: str,
    selected_layer_option_sha256: tuple[str, ...],
    selected_layer_evidence_sha256: tuple[str, ...],
    total_predicted_risk: float,
) -> dict[str, Any]:
    return {
        "schema": LFM2_HETEROGENEOUS_ALLOCATION_DECISION_SCHEMA,
        "source_graph_fingerprint": source_graph_fingerprint,
        "source_structural_topology_fingerprint": (
            source_structural_topology_fingerprint
        ),
        "source_layer_order_sha256": source_layer_order_sha256,
        "target_pins_sha256": target_pins_sha256,
        "target_vocab_evidence_sha256": target_vocab_evidence_sha256,
        "runtime_evidence_sha256": runtime_evidence_sha256,
        "token_bridge_evidence_sha256": token_bridge_evidence_sha256,
        "exact_target_unique_parameters": exact_target_unique_parameters,
        "budget_problem_fingerprint": budget_problem_fingerprint,
        "budget_selection_fingerprint": budget_selection_fingerprint,
        "exact_topology_problem_fingerprint": exact_topology_problem_fingerprint,
        "exact_topology_selection_fingerprint": exact_topology_selection_fingerprint,
        "selected_layer_option_sha256": list(selected_layer_option_sha256),
        "selected_layer_evidence_sha256": list(selected_layer_evidence_sha256),
        "total_predicted_risk": total_predicted_risk,
        "selection_policy": {
            "one_option_per_discovered_source_layer": True,
            "source_order_derived_from_inspected_graph": True,
            "residual_attention_component_protection": True,
            "risk_is_behavioral_hard_gate": False,
            "exact_parameter_count_required": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
        },
    }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousAllocationDecisionCore:
    """Sealed selection core constructed before the descriptor exists."""

    source_graph_fingerprint: str
    source_structural_topology_fingerprint: str
    source_layer_order_sha256: str
    target_pins_sha256: str
    target_vocab_evidence_sha256: str
    runtime_evidence_sha256: str
    token_bridge_evidence_sha256: str
    exact_target_unique_parameters: int
    budget_problem_fingerprint: str
    budget_selection_fingerprint: str
    exact_topology_problem_fingerprint: str
    exact_topology_selection_fingerprint: str
    selected_layer_option_sha256: tuple[str, ...]
    selected_layer_evidence_sha256: tuple[str, ...]
    total_predicted_risk: float
    allocation_decision_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_graph_fingerprint",
            "source_structural_topology_fingerprint",
            "source_layer_order_sha256",
            "target_pins_sha256",
            "target_vocab_evidence_sha256",
            "runtime_evidence_sha256",
            "token_bridge_evidence_sha256",
            "budget_problem_fingerprint",
            "budget_selection_fingerprint",
            "exact_topology_problem_fingerprint",
            "exact_topology_selection_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        _positive_int(
            self.exact_target_unique_parameters,
            "exact_target_unique_parameters",
        )
        if (
            type(self.selected_layer_option_sha256) is not tuple
            or not self.selected_layer_option_sha256
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.selected_layer_option_sha256
            )
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "A decision core needs ordered selected layer-option digests."
            )
        if (
            type(self.selected_layer_evidence_sha256) is not tuple
            or len(self.selected_layer_evidence_sha256)
            != len(self.selected_layer_option_sha256)
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.selected_layer_evidence_sha256
            )
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "A decision core needs one evidence digest per selected layer."
            )
        risk = _risk(self.total_predicted_risk)
        body = _allocation_decision_body(
            source_graph_fingerprint=self.source_graph_fingerprint,
            source_structural_topology_fingerprint=(
                self.source_structural_topology_fingerprint
            ),
            source_layer_order_sha256=self.source_layer_order_sha256,
            target_pins_sha256=self.target_pins_sha256,
            target_vocab_evidence_sha256=self.target_vocab_evidence_sha256,
            runtime_evidence_sha256=self.runtime_evidence_sha256,
            token_bridge_evidence_sha256=self.token_bridge_evidence_sha256,
            exact_target_unique_parameters=self.exact_target_unique_parameters,
            budget_problem_fingerprint=self.budget_problem_fingerprint,
            budget_selection_fingerprint=self.budget_selection_fingerprint,
            exact_topology_problem_fingerprint=(
                self.exact_topology_problem_fingerprint
            ),
            exact_topology_selection_fingerprint=(
                self.exact_topology_selection_fingerprint
            ),
            selected_layer_option_sha256=self.selected_layer_option_sha256,
            selected_layer_evidence_sha256=self.selected_layer_evidence_sha256,
            total_predicted_risk=risk,
        )
        if _sha256(
            self.allocation_decision_sha256,
            "allocation_decision_sha256",
        ) != canonical_sha256(body):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous allocation-decision seal is invalid."
            )

    def _body_dict(self) -> dict[str, Any]:
        return _allocation_decision_body(
            source_graph_fingerprint=self.source_graph_fingerprint,
            source_structural_topology_fingerprint=(
                self.source_structural_topology_fingerprint
            ),
            source_layer_order_sha256=self.source_layer_order_sha256,
            target_pins_sha256=self.target_pins_sha256,
            target_vocab_evidence_sha256=self.target_vocab_evidence_sha256,
            runtime_evidence_sha256=self.runtime_evidence_sha256,
            token_bridge_evidence_sha256=self.token_bridge_evidence_sha256,
            exact_target_unique_parameters=self.exact_target_unique_parameters,
            budget_problem_fingerprint=self.budget_problem_fingerprint,
            budget_selection_fingerprint=self.budget_selection_fingerprint,
            exact_topology_problem_fingerprint=(
                self.exact_topology_problem_fingerprint
            ),
            exact_topology_selection_fingerprint=(
                self.exact_topology_selection_fingerprint
            ),
            selected_layer_option_sha256=self.selected_layer_option_sha256,
            selected_layer_evidence_sha256=self.selected_layer_evidence_sha256,
            total_predicted_risk=self.total_predicted_risk,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._body_dict(),
            "allocation_decision_sha256": self.allocation_decision_sha256,
        }

    @classmethod
    def restore(
        cls,
        value: Lfm2HeterogeneousAllocationDecisionCore
        | Mapping[str, Any]
        | str
        | bytes,
    ) -> Lfm2HeterogeneousAllocationDecisionCore:
        if isinstance(value, cls):
            return cls.restore(value.as_dict())
        raw = _mapping(value, "LFM2 heterogeneous allocation decision")
        if set(raw) != {
            "schema",
            "source_graph_fingerprint",
            "source_structural_topology_fingerprint",
            "source_layer_order_sha256",
            "target_pins_sha256",
            "target_vocab_evidence_sha256",
            "runtime_evidence_sha256",
            "token_bridge_evidence_sha256",
            "exact_target_unique_parameters",
            "budget_problem_fingerprint",
            "budget_selection_fingerprint",
            "exact_topology_problem_fingerprint",
            "exact_topology_selection_fingerprint",
            "selected_layer_option_sha256",
            "selected_layer_evidence_sha256",
            "total_predicted_risk",
            "selection_policy",
            "allocation_decision_sha256",
        } or raw.pop("schema", None) != (LFM2_HETEROGENEOUS_ALLOCATION_DECISION_SCHEMA):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous allocation-decision fields are not canonical."
            )
        expected_policy = _allocation_decision_body(
            source_graph_fingerprint="0" * 64,
            source_structural_topology_fingerprint="0" * 64,
            source_layer_order_sha256="0" * 64,
            target_pins_sha256="0" * 64,
            target_vocab_evidence_sha256="0" * 64,
            runtime_evidence_sha256="0" * 64,
            token_bridge_evidence_sha256="0" * 64,
            exact_target_unique_parameters=1,
            budget_problem_fingerprint="0" * 64,
            budget_selection_fingerprint="0" * 64,
            exact_topology_problem_fingerprint="0" * 64,
            exact_topology_selection_fingerprint="0" * 64,
            selected_layer_option_sha256=("0" * 64,),
            selected_layer_evidence_sha256=("0" * 64,),
            total_predicted_risk=0.0,
        )["selection_policy"]
        if raw.pop("selection_policy", None) != expected_policy:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous allocation-decision policy is not canonical."
            )
        options = raw.pop("selected_layer_option_sha256", None)
        evidence = raw.pop("selected_layer_evidence_sha256", None)
        if not isinstance(options, list) or not isinstance(evidence, list):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Allocation-decision layer selections must be JSON arrays."
            )
        return cls(
            **raw,
            selected_layer_option_sha256=tuple(options),
            selected_layer_evidence_sha256=tuple(evidence),
        )


def _selection_receipt_body(
    *,
    allocation_decision: Lfm2HeterogeneousAllocationDecisionCore,
    descriptor_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA,
        "allocation_decision": allocation_decision.as_dict(),
        "descriptor_sha256": descriptor_sha256,
        "exact_unique_parameter_count": (
            allocation_decision.exact_target_unique_parameters
        ),
        "allocation_precedes_descriptor": True,
    }


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousLayoutSelectionReceipt:
    """Final receipt binding the sealed allocation core to a descriptor."""

    allocation_decision: Lfm2HeterogeneousAllocationDecisionCore
    descriptor_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.allocation_decision) is not (
            Lfm2HeterogeneousAllocationDecisionCore
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "selection receipt requires a typed allocation decision."
            )
        descriptor = _sha256(self.descriptor_sha256, "descriptor_sha256")
        body = _selection_receipt_body(
            allocation_decision=self.allocation_decision,
            descriptor_sha256=descriptor,
        )
        if _sha256(self.receipt_sha256, "receipt_sha256") != canonical_sha256(body):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous layout-selection receipt seal is invalid."
            )

    def _body_dict(self) -> dict[str, Any]:
        return _selection_receipt_body(
            allocation_decision=self.allocation_decision,
            descriptor_sha256=self.descriptor_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._body_dict(), "receipt_sha256": self.receipt_sha256}

    def serialize(self) -> bytes:
        return _canonical_bytes(self.as_dict())

    @classmethod
    def restore(
        cls,
        value: Lfm2HeterogeneousLayoutSelectionReceipt
        | Mapping[str, Any]
        | str
        | bytes,
    ) -> Lfm2HeterogeneousLayoutSelectionReceipt:
        if isinstance(value, cls):
            return cls.restore(value.as_dict())
        raw = _mapping(value, "LFM2 heterogeneous layout-selection receipt")
        if set(raw) != {
            "schema",
            "allocation_decision",
            "descriptor_sha256",
            "exact_unique_parameter_count",
            "allocation_precedes_descriptor",
            "receipt_sha256",
        } or raw.pop("schema", None) != (LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Heterogeneous layout-selection receipt fields are not canonical."
            )
        decision = Lfm2HeterogeneousAllocationDecisionCore.restore(
            raw.pop("allocation_decision", None)
        )
        if raw.pop("allocation_precedes_descriptor", None) is not True:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Selection receipt does not prove allocation-before-descriptor."
            )
        if raw.pop("exact_unique_parameter_count", None) != (
            decision.exact_target_unique_parameters
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Selection receipt count differs from its allocation decision."
            )
        return cls(allocation_decision=decision, **raw)


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousLayoutSelection:
    """In-memory pure selector result and its authenticated public artifacts."""

    descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor
    receipt: Lfm2HeterogeneousLayoutSelectionReceipt
    budget_problem: ArchitectureDiscoveredBudgetProblem
    budget_selection: ArchitectureDiscoveredBudgetSelection

    def __post_init__(self) -> None:
        descriptor = restore_lfm2_heterogeneous_physical_layout(
            self.descriptor,
            expected_descriptor_sha256=self.receipt.descriptor_sha256,
        )
        decision = self.receipt.allocation_decision
        if descriptor.allocation_evidence_sha256 != (
            decision.allocation_decision_sha256
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Descriptor does not bind the selected allocation decision."
            )
        if descriptor.accounting.unique_parameter_elements != (
            decision.exact_target_unique_parameters
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Descriptor count differs from the allocation decision."
            )
        if self.budget_problem.fingerprint != decision.budget_problem_fingerprint:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Budget problem differs from the allocation decision."
            )
        if self.budget_selection.fingerprint != (decision.budget_selection_fingerprint):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "Budget selection differs from the allocation decision."
            )


def _boundary_nodes(source_graph: ModelCircuitGraph) -> tuple[CircuitNode, ...]:
    nodes = tuple(
        node
        for node in source_graph.nodes
        if node.layer_index is None and node.parameters
    )
    kinds = [node.kind for node in nodes]
    if (
        kinds.count(CircuitKind.TOKEN_EMBEDDING.value) != 1
        or kinds.count(CircuitKind.OUTPUT_HEAD.value) != 1
        or kinds.count(CircuitKind.NORM.value) != 1
        or any(
            kind
            not in {
                CircuitKind.TOKEN_EMBEDDING.value,
                CircuitKind.OUTPUT_HEAD.value,
                CircuitKind.NORM.value,
            }
            for kind in kinds
        )
    ):
        raise Lfm2HeterogeneousLayoutSelectionError(
            "The physical selector requires one inspected token embedding, "
            "output head, and final norm boundary."
        )
    return tuple(sorted(nodes, key=lambda node: (node.order, node.node_id)))


def _selected_physical_layer(
    ordinal: int,
    option: Lfm2AuthenticatedLayerPhysicalOption,
) -> Lfm2HeterogeneousLayerPhysicalLayout:
    if option.layer_type == "conv":
        assert option.short_conv_internal_width is not None
        return Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=ordinal,
            mlp_intermediate_size=option.mlp_intermediate_size,
            internal_width=option.short_conv_internal_width,
            writer_rank=option.short_conv_writer_rank,
            selection_evidence_sha256=option.option_sha256,
        )
    assert option.query_heads is not None
    assert option.key_value_heads is not None
    assert option.head_dim is not None
    return Lfm2HeterogeneousLayerPhysicalLayout.attention(
        layer_index=ordinal,
        mlp_intermediate_size=option.mlp_intermediate_size,
        query_heads=option.query_heads,
        key_value_heads=option.key_value_heads,
        head_dim=option.head_dim,
        selection_evidence_sha256=option.option_sha256,
    )


def select_lfm2_heterogeneous_physical_layout(
    *,
    source_graph: ModelCircuitGraph,
    layer_options: Sequence[
        Lfm2AuthenticatedLayerPhysicalOption | Mapping[str, Any] | str | bytes
    ],
    target_pins: Lfm2HeterogeneousTargetPins | Mapping[str, Any] | str | bytes,
    exact_target_unique_parameters: int,
    max_states: int = 100_000,
) -> Lfm2HeterogeneousLayoutSelection:
    """Select and authenticate one aggressive whole-model physical layout.

    Risk only orders exact feasible layouts.  The selector deliberately sets no
    behavioral retention floor; coherent held-out evaluation remains a later
    manual/semantic decision.
    """

    layers = _discover_lfm2_source_layers(source_graph)
    pins = Lfm2HeterogeneousTargetPins.restore(target_pins)
    target = _positive_int(
        exact_target_unique_parameters,
        "exact_target_unique_parameters",
    )
    state_ceiling = _positive_int(max_states, "max_states")
    restored_options = tuple(
        Lfm2AuthenticatedLayerPhysicalOption.restore(value) for value in layer_options
    )
    if not restored_options:
        raise Lfm2HeterogeneousLayoutSelectionError(
            "At least one authenticated physical option is required."
        )
    if len({option.option_sha256 for option in restored_options}) != len(
        restored_options
    ):
        raise Lfm2HeterogeneousLayoutSelectionError(
            "The physical option set contains a duplicate option."
        )
    discovered_by_sha = {layer.source_layer_sha256: layer for layer in layers}
    unknown = sorted(
        {
            option.source_layer_sha256
            for option in restored_options
            if option.source_layer_sha256 not in discovered_by_sha
        }
    )
    if unknown:
        raise Lfm2HeterogeneousLayoutSelectionError(
            f"Physical options bind unknown source layers: {unknown}."
        )
    options_by_layer: dict[str, list[Lfm2AuthenticatedLayerPhysicalOption]] = {
        layer.source_layer_sha256: [] for layer in layers
    }
    for option in restored_options:
        source_layer = discovered_by_sha[option.source_layer_sha256]
        if option.layer_type != source_layer.layer_type:
            raise Lfm2HeterogeneousLayoutSelectionError(
                "A physical option has the wrong operator kind for its source layer."
            )
        if (
            option.short_conv_writer_rank is not None
            and option.short_conv_writer_rank
            > min(pins.hidden_size, int(option.short_conv_internal_width))
        ):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "A factor writer rank exceeds a target dense matrix dimension."
            )
        options_by_layer[option.source_layer_sha256].append(option)
    incomplete = tuple(
        layer.ordinal
        for layer in layers
        if not options_by_layer[layer.source_layer_sha256]
    )
    if incomplete:
        raise Lfm2HeterogeneousLayoutSelectionError(
            f"Physical options are incomplete for source ordinals {incomplete}."
        )
    for layer in layers:
        candidates = options_by_layer[layer.source_layer_sha256]
        geometry_keys = [_option_geometry_key(option) for option in candidates]
        if len(set(geometry_keys)) != len(geometry_keys):
            raise Lfm2HeterogeneousLayoutSelectionError(
                "A source layer repeats an equivalent physical geometry."
            )
        candidates.sort(key=lambda option: option.option_sha256)

    boundary = _boundary_nodes(source_graph)
    boundary_group_id = "00000000.boundary"
    boundary_count = 2 * pins.vocab_size * pins.hidden_size + pins.hidden_size
    boundary_retained = []
    for node in boundary:
        if node.kind in {
            CircuitKind.TOKEN_EMBEDDING.value,
            CircuitKind.OUTPUT_HEAD.value,
        }:
            capacity = pins.vocab_size * pins.hidden_size
        else:
            capacity = pins.hidden_size
        boundary_retained.extend(_role_allocations(node, capacity))
    groups: list[DiscoveredComponentChoiceGroup] = [
        DiscoveredComponentChoiceGroup(
            boundary_group_id,
            (
                DiscoveredComponentTopologyOption(
                    option_id=f"{boundary_group_id}/fixed",
                    group_id=boundary_group_id,
                    source_node_ids=tuple(node.node_id for node in boundary),
                    local_unique_parameters=boundary_count,
                    local_serialized_bytes=4 * boundary_count,
                    retained_parameters=tuple(boundary_retained),
                    structural_risk=0.0,
                    metadata={"lfm2_selection.kind": "fixed_untied_vocab_boundary"},
                ),
            ),
        )
    ]
    solver_option_lookup: dict[str, Lfm2AuthenticatedLayerPhysicalOption] = {}
    node_by_id = {node.node_id: node for node in source_graph.nodes}
    for layer in layers:
        group_id = f"{layer.ordinal + 1:08d}.layer.{layer.source_layer_sha256}"
        converted = []
        layer_nodes = tuple(
            node_by_id[node_id] for node_id in layer.parameterized_node_ids
        )
        norms = tuple(
            node for node in layer_nodes if node.kind == CircuitKind.NORM.value
        )
        mlp = node_by_id[layer.mlp_node_id]
        operator = node_by_id[layer.operator_node_id]
        for option in options_by_layer[layer.source_layer_sha256]:
            norm_count, mlp_count, operator_count = _component_parameter_counts(
                option,
                pins,
            )
            retained = []
            for norm in norms:
                retained.extend(_role_allocations(norm, pins.hidden_size))
            retained.extend(_role_allocations(mlp, mlp_count))
            retained.extend(_role_allocations(operator, operator_count))
            local_count = norm_count + mlp_count + operator_count
            option_id = f"{group_id}/option/{option.option_sha256}"
            solver_option_lookup[option_id] = option
            converted.append(
                DiscoveredComponentTopologyOption(
                    option_id=option_id,
                    group_id=group_id,
                    source_node_ids=layer.parameterized_node_ids,
                    local_unique_parameters=local_count,
                    local_serialized_bytes=4 * local_count,
                    retained_parameters=tuple(retained),
                    structural_risk=option.structural_risk,
                    metadata={
                        "lfm2_selection.layer_type": option.layer_type,
                        "lfm2_selection.option_sha256": option.option_sha256,
                        "lfm2_selection.evidence_sha256": (
                            option.selection_evidence_sha256
                        ),
                    },
                )
            )
        groups.append(DiscoveredComponentChoiceGroup(group_id, tuple(converted)))

    problem = ArchitectureDiscoveredBudgetProblem(
        source_graph=source_graph,
        exact_target_unique_parameters=target,
        groups=tuple(groups),
        protection_rules=residual_attention_component_protection_rules(),
        unclassified_loss_weight=0.0,
        bottleneck_risk_weight=1.0,
        max_states=state_ceiling,
    )
    try:
        selection = solve_architecture_discovered_budget(problem)
    except DynamicFoldBudgetUnattainable as exc:
        raise Lfm2HeterogeneousLayoutUnattainable(str(exc)) from exc
    except DynamicFoldBudgetError as exc:
        raise Lfm2HeterogeneousLayoutSelectionError(str(exc)) from exc

    selected_ids = tuple(
        option_id
        for option_id in selection.selected_option_ids
        if option_id in solver_option_lookup
    )
    if len(selected_ids) != len(layers):
        raise Lfm2HeterogeneousLayoutSelectionError(
            "Exact solver did not select one physical option per source layer."
        )
    selected_options = tuple(solver_option_lookup[value] for value in selected_ids)
    if tuple(option.source_layer_sha256 for option in selected_options) != tuple(
        layer.source_layer_sha256 for layer in layers
    ):
        raise Lfm2HeterogeneousLayoutSelectionError(
            "Exact solver selection order differs from inspected causal order."
        )

    layer_order_sha256 = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-source-layer-order-v1",
            "source_graph_fingerprint": source_graph.fingerprint,
            "layers": [layer.as_dict() for layer in layers],
        }
    )
    decision_body = _allocation_decision_body(
        source_graph_fingerprint=source_graph.fingerprint,
        source_structural_topology_fingerprint=(
            source_graph.structural_topology_fingerprint
        ),
        source_layer_order_sha256=layer_order_sha256,
        target_pins_sha256=pins.pins_sha256,
        target_vocab_evidence_sha256=pins.target_vocab_evidence_sha256,
        runtime_evidence_sha256=pins.runtime_evidence_sha256,
        token_bridge_evidence_sha256=pins.token_bridge_evidence_sha256,
        exact_target_unique_parameters=target,
        budget_problem_fingerprint=problem.fingerprint,
        budget_selection_fingerprint=selection.fingerprint,
        exact_topology_problem_fingerprint=(
            selection.topology_selection.problem_fingerprint
        ),
        exact_topology_selection_fingerprint=(selection.topology_selection.fingerprint),
        selected_layer_option_sha256=tuple(
            option.option_sha256 for option in selected_options
        ),
        selected_layer_evidence_sha256=tuple(
            option.selection_evidence_sha256 for option in selected_options
        ),
        total_predicted_risk=selection.total_predicted_risk,
    )
    decision = Lfm2HeterogeneousAllocationDecisionCore(
        **{
            key: value
            for key, value in decision_body.items()
            if key
            not in {
                "schema",
                "selection_policy",
                "selected_layer_option_sha256",
                "selected_layer_evidence_sha256",
            }
        },
        selected_layer_option_sha256=tuple(
            decision_body["selected_layer_option_sha256"]
        ),
        selected_layer_evidence_sha256=tuple(
            decision_body["selected_layer_evidence_sha256"]
        ),
        allocation_decision_sha256=canonical_sha256(decision_body),
    )
    physical_layers = tuple(
        _selected_physical_layer(index, option)
        for index, option in enumerate(selected_options)
    )
    descriptor = build_lfm2_heterogeneous_physical_layout(
        hidden_size=pins.hidden_size,
        vocab_size=pins.vocab_size,
        conv_kernel_size=pins.conv_kernel_size,
        rope_theta=pins.rope_theta,
        norm_eps=pins.norm_eps,
        max_position_embeddings=pins.max_position_embeddings,
        use_cache=pins.use_cache,
        attention_implementation=pins.attention_implementation,
        bos_token_id=pins.bos_token_id,
        eos_token_id=pins.eos_token_id,
        pad_token_id=pins.pad_token_id,
        block_auto_adjust_ff_dim=pins.block_auto_adjust_ff_dim,
        layers=physical_layers,
        source_topology_sha256=source_graph.fingerprint,
        allocation_evidence_sha256=decision.allocation_decision_sha256,
        token_bridge_evidence_sha256=pins.token_bridge_evidence_sha256,
        expected_unique_parameter_count=target,
    )
    receipt_body = _selection_receipt_body(
        allocation_decision=decision,
        descriptor_sha256=descriptor.descriptor_sha256,
    )
    receipt = Lfm2HeterogeneousLayoutSelectionReceipt(
        allocation_decision=decision,
        descriptor_sha256=descriptor.descriptor_sha256,
        receipt_sha256=canonical_sha256(receipt_body),
    )
    return Lfm2HeterogeneousLayoutSelection(
        descriptor=descriptor,
        receipt=receipt,
        budget_problem=problem,
        budget_selection=selection,
    )


__all__ = [
    "LFM2_HETEROGENEOUS_ALLOCATION_DECISION_SCHEMA",
    "LFM2_HETEROGENEOUS_LAYER_OPTION_SCHEMA",
    "LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA",
    "LFM2_HETEROGENEOUS_TARGET_PINS_SCHEMA",
    "Lfm2AuthenticatedLayerPhysicalOption",
    "Lfm2DiscoveredSourceLayer",
    "Lfm2HeterogeneousAllocationDecisionCore",
    "Lfm2HeterogeneousLayoutSelection",
    "Lfm2HeterogeneousLayoutSelectionError",
    "Lfm2HeterogeneousLayoutSelectionReceipt",
    "Lfm2HeterogeneousLayoutUnattainable",
    "Lfm2HeterogeneousTargetPins",
    "build_lfm2_attention_layer_option",
    "build_lfm2_heterogeneous_target_pins",
    "build_lfm2_short_conv_layer_option",
    "discover_lfm2_source_layers",
    "select_lfm2_heterogeneous_physical_layout",
]
