"""LFM2 bridge for the model-agnostic dynamic folding compiler.

This module deliberately adapts existing implementations instead of copying
their mathematics:

* :mod:`dynamic_lfm2_targets` owns progressive exact-budget target planning;
* :mod:`hf_lfm2_closed_loop_fold` owns coupled Q/K/V/O, SwiGLU, short-conv,
  endpoint-fit, installation, commit, and byte-exact rollback behavior; and
* :mod:`dynamic_fold_executor` owns transactional orchestration and auditing.

The bridge keeps parent traces outside the execution loop.  Its evidence
builder receives only an immutable parent-cache payload plus the current child
state, so it can refresh on-policy child designs without regenerating parent
traces.  Whole-output semantic evaluation remains an injected final evaluator;
the admission evaluator here proves only transaction-local invariants.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Protocol

from .dynamic_circuit_ir import (
    CircuitKind,
    JsonScalar,
    ModelCircuitGraph,
    canonical_sha256,
)
from .dynamic_fold_compiler import DynamicFoldCompilation, DynamicFoldingCompiler
from .dynamic_fold_budget import (
    DynamicFoldBudgetUnattainable,
    ExactTopologyBudgetProblem,
    TopologyChoiceGroup,
    TopologyOption,
    solve_exact_topology_budget,
)
from .dynamic_fold_executor import (
    DynamicFoldExecutionError,
    ExactCopyOperator,
    FoldDecision,
    FoldEvaluationContext,
    FoldEvaluator,
    FoldEvidence,
    FoldExecutionContext,
    FoldInitializer,
    FoldOperatorRegistry,
    FoldTransaction,
    fingerprint_model_state,
)
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_fold_registry import (
    ClosureLevel,
    ExperimentLedger,
    ExperimentSignature,
    LedgerDecision,
    LedgerRecord,
    RecordStatus,
)
from .dynamic_lfm2_targets import (
    Lfm2ProgressiveStage,
    Lfm2ProgressiveTargetPlan,
    Lfm2Topology,
    plan_lfm2_progressive_targets,
)
from .hf_lfm2_closed_loop_fold import (
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2ClosedLoopFoldTransaction,
    build_lfm2_closed_loop_fold_plan,
    install_lfm2_closed_loop_fold_plan,
)
from .hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_CONFIG_KEYS,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    StrictTokenizerBridgeArtifact,
    strict_tokenizer_bridge_configuration,
)
from .dynamic_model_inspector import inspect_model_circuits
from .dynamic_parameter_frontier import (
    COMPONENT_ALLOCATION_POLICY,
    COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    DenseComponentAllocationReceipt,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_KIND,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY,
    component_allocation_topology_fingerprint,
)
from .dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetRequest,
)
from .hf_lfm2_dynamic_blank_materializer import (
    LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA,
    extract_lfm2_dynamic_physical_layout_from_config,
    materialize_lfm2_dynamic_blank_child,
)
from .hf_lfm2_dynamic_physical_layout import (
    LFM2_DYNAMIC_PHYSICAL_AUDIT_SCHEMA,
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
    restore_lfm2_heterogeneous_physical_layout,
)
from .hf_lfm2_heterogeneous_layout_selection import (
    Lfm2HeterogeneousLayoutSelectionReceipt,
)


ADAPTER_SCHEMA = "neural-forge-dynamic-lfm2-adapter-v1"
TARGET_SELECTION_METHOD = "lfm2_progressive_exact_budget"
LAYER_METHOD = "lfm2_closed_loop_layer"
HYBRID_MODEL_METHOD = "lfm2_hybrid_residual_attention_model"
PRODUCER_FIRST_MODEL_METHOD = "lfm2_producer_first_residual_attention_model"
_DENSE_BUDGET_MODEL_METHOD = "lfm2_dense_budget_residual_attention_model"
_DENSE_BUDGET_CATALOG_MARKER = "_neural_forge_dense_topology_catalog"
_DENSE_BUDGET_CONFIGURATION_KEYS = frozenset(
    {
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "layer_types",
        "conv_L_cache",
        "block_auto_adjust_ff_dim",
        "tie_word_embeddings",
        "_neural_forge_ordered_source_spans",
        "_neural_forge_target_layer_types",
        _DENSE_BUDGET_CATALOG_MARKER,
        "_neural_forge_tokenizer_bridge_required",
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
        COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    }
)
_DENSE_BUDGET_UNTIED_CONFIGURATION_KEYS = (
    _DENSE_BUDGET_CONFIGURATION_KEYS
    | {LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY}
)
_DENSE_BUDGET_HETEROGENEOUS_CONFIGURATION_KEYS = frozenset(
    {
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY,
    }
)


class DynamicLfm2AdapterError(RuntimeError):
    """The LFM2 target or execution bridge cannot prove its contract."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicLfm2AdapterError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicLfm2AdapterError(f"{name} must be a positive integer.")
    return value


def _config_value(config: Any, names: Sequence[str], label: str) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    raise DynamicLfm2AdapterError(f"LFM2 config does not expose {label}.")


@dataclass(frozen=True, slots=True)
class Lfm2InspectedSource:
    """Topology and accounting proven jointly by config and circuit graph."""

    topology: Lfm2Topology
    hidden_size: int
    head_dim: int
    layer_types: tuple[str, ...]
    unique_parameters: int
    graph_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology.as_dict(),
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "layer_types": list(self.layer_types),
            "unique_parameters": self.unique_parameters,
            "graph_fingerprint": self.graph_fingerprint,
        }


def inspect_lfm2_progressive_source(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
) -> Lfm2InspectedSource:
    """Derive progressive-target inputs without hard-coded model dimensions."""

    if not isinstance(parent_graph, ModelCircuitGraph) or parent_graph.family != "lfm2":
        raise DynamicLfm2AdapterError("parent_graph must be an inspected LFM2 graph.")
    config = getattr(parent_model, "config", None)
    if config is None or getattr(config, "model_type", None) != "lfm2":
        raise DynamicLfm2AdapterError("parent_model must expose an LFM2 config.")
    hidden_size = _positive_int(
        int(_config_value(config, ("hidden_size",), "hidden_size")), "hidden_size"
    )
    query_heads = _positive_int(
        int(_config_value(config, ("num_attention_heads", "num_heads"), "query heads")),
        "query_heads",
    )
    key_value_heads = _positive_int(
        int(_config_value(config, ("num_key_value_heads",), "key/value heads")),
        "key_value_heads",
    )
    head_dim_value = getattr(config, "head_dim", None)
    head_dim = (
        hidden_size // query_heads
        if head_dim_value is None
        else _positive_int(int(head_dim_value), "head_dim")
    )
    layer_types = tuple(
        str(value) for value in _config_value(config, ("layer_types",), "layer_types")
    )
    if len(layer_types) != len(parent_graph.layer_indices):
        raise DynamicLfm2AdapterError(
            "LFM2 config layer_types and inspected graph depth disagree."
        )
    mlp_widths = {
        int(dict(node.attributes)["intermediate_size"])
        for node in parent_graph.nodes
        if node.kind == CircuitKind.GATED_MLP.value
    }
    if len(mlp_widths) != 1:
        raise DynamicLfm2AdapterError(
            "The current progressive LFM2 adapter requires uniform MLP width."
        )
    intermediate_size = next(iter(mlp_widths))
    configured_intermediate = int(
        _config_value(
            config,
            ("intermediate_size", "block_ff_dim"),
            "intermediate_size",
        )
    )
    if configured_intermediate != intermediate_size and not bool(
        getattr(config, "block_auto_adjust_ff_dim", False)
    ):
        raise DynamicLfm2AdapterError(
            "Configured and inspected LFM2 intermediate sizes disagree."
        )
    attention_nodes = tuple(
        node for node in parent_graph.nodes if node.kind == CircuitKind.ATTENTION.value
    )
    for node in attention_nodes:
        attributes = dict(node.attributes)
        observed = (
            int(attributes["query_heads"]),
            int(attributes["key_value_heads"]),
            int(attributes["head_dim"]),
        )
        if observed != (query_heads, key_value_heads, head_dim):
            raise DynamicLfm2AdapterError(
                "Configured and inspected LFM2 attention topology disagree."
            )
    graph_layer_types = []
    for layer_index in parent_graph.layer_indices:
        kinds = {
            node.kind
            for node in parent_graph.layer_nodes(layer_index)
            if node.kind
            in {
                CircuitKind.ATTENTION.value,
                CircuitKind.SHORT_CONV.value,
            }
        }
        if len(kinds) != 1:
            raise DynamicLfm2AdapterError(
                "Each inspected LFM2 layer must expose one canonical operator."
            )
        graph_layer_types.append(
            "full_attention"
            if next(iter(kinds)) == CircuitKind.ATTENTION.value
            else "conv"
        )
    if layer_types != tuple(graph_layer_types):
        raise DynamicLfm2AdapterError(
            "Configured layer_types ordering differs from the inspected graph."
        )
    return Lfm2InspectedSource(
        topology=Lfm2Topology(query_heads, key_value_heads, intermediate_size),
        hidden_size=hidden_size,
        head_dim=head_dim,
        layer_types=layer_types,
        unique_parameters=parent_graph.unique_parameter_count,
        graph_fingerprint=parent_graph.fingerprint,
    )


def derive_lfm2_target_ladder(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    *,
    terminal_topology: Lfm2Topology,
    initial_mlp_step: int = 1,
    maximum_mlp_step: int = 64,
) -> Lfm2ProgressiveTargetPlan:
    """Build the smallest-change-first ladder from inspected source facts."""

    source = inspect_lfm2_progressive_source(parent_model, parent_graph)
    return plan_lfm2_progressive_targets(
        source_unique_parameters=source.unique_parameters,
        hidden_size=source.hidden_size,
        head_dim=source.head_dim,
        layer_types=source.layer_types,
        source=source.topology,
        target=terminal_topology,
        initial_mlp_step=initial_mlp_step,
        maximum_mlp_step=maximum_mlp_step,
    )


def _target_selection_signature(
    parent_graph: ModelCircuitGraph,
    parent_state_fingerprint: str,
    plan: Lfm2ProgressiveTargetPlan,
    stage: Lfm2ProgressiveStage,
    evidence_fingerprint: str,
) -> ExperimentSignature:
    if parent_graph.tokenizer_fingerprint is None:
        raise DynamicLfm2AdapterError(
            "LFM2 target selection requires a tokenizer fingerprint."
        )
    target_identity = {
        "schema": ADAPTER_SCHEMA,
        "kind": "unmaterialized_lfm2_exact_budget_target",
        "source_graph_fingerprint": parent_graph.fingerprint,
        "stage": stage.as_dict(),
    }
    child_graph_fingerprint = canonical_sha256(target_identity)
    child_state_fingerprint = canonical_sha256(
        {
            "kind": "preconstruction_blank_identity",
            "child_graph_fingerprint": child_graph_fingerprint,
            "exact_target_unique_parameters": stage.unique_parameters,
        }
    )
    source_units = tuple(
        node.node_id for node in parent_graph.nodes if node.layer_index is not None
    )
    target_unit = (
        "lfm2:"
        f"q{stage.topology.query_heads}:"
        f"kv{stage.topology.key_value_heads}:"
        f"mlp{stage.topology.intermediate_size}:"
        f"p{stage.unique_parameters}"
    )
    return ExperimentSignature(
        method_id=TARGET_SELECTION_METHOD,
        method_version=1,
        family_id="lfm2_progressive_target_selection",
        parent_graph_fingerprints=(parent_graph.fingerprint,),
        parent_state_fingerprints=(
            (
                parent_graph.model_name,
                _digest(
                    parent_state_fingerprint,
                    "parent_state_fingerprint",
                ),
            ),
        ),
        child_graph_fingerprint=child_graph_fingerprint,
        child_state_fingerprint=child_state_fingerprint,
        evidence_fingerprint=_digest(evidence_fingerprint, "evidence_fingerprint"),
        source_unit_ids=source_units,
        target_unit_ids=(target_unit,),
        topology_transition=(
            f"q{plan.source.query_heads}_kv{plan.source.key_value_heads}_"
            f"mlp{plan.source.intermediate_size}->"
            f"q{stage.topology.query_heads}_kv{stage.topology.key_value_heads}_"
            f"mlp{stage.topology.intermediate_size}"
        ),
        selection_policy="first_unclosed_smallest_representable_exact_budget",
        fit_objective="coupled_lfm2_closed_loop_residual_attention",
        promotion_gate="automatic_behavior_then_final_manual_review",
        zero_gradient=True,
        configuration=(
            ("stage_index", stage.stage_index),
            ("changed_circuit", stage.changed_circuit),
            ("exact_target_unique_parameters", stage.unique_parameters),
            ("cumulative_parameters_removed", stage.cumulative_parameters_removed),
            ("query_heads", stage.topology.query_heads),
            ("key_value_heads", stage.topology.key_value_heads),
            ("intermediate_size", stage.topology.intermediate_size),
            ("target_plan_fingerprint", plan.fingerprint),
            ("parent_name", parent_graph.model_name),
            ("tokenizer_fingerprint", parent_graph.tokenizer_fingerprint),
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2SelectedTarget:
    """The first progressive exact-budget stage not closed by the ledger."""

    plan: Lfm2ProgressiveTargetPlan
    stage: Lfm2ProgressiveStage
    signature: ExperimentSignature
    ledger_decision: LedgerDecision

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.plan, Lfm2ProgressiveTargetPlan):
            raise DynamicLfm2AdapterError("Selected target plan has the wrong type.")
        if not isinstance(self.stage, Lfm2ProgressiveStage):
            raise DynamicLfm2AdapterError("Selected target stage has the wrong type.")
        if not isinstance(self.signature, ExperimentSignature):
            raise DynamicLfm2AdapterError(
                "Selected target signature has the wrong type."
            )
        if not isinstance(self.ledger_decision, LedgerDecision):
            raise DynamicLfm2AdapterError(
                "Selected target ledger decision has the wrong type."
            )
        if (
            self.stage.stage_index >= len(self.plan.stages)
            or self.plan.stages[self.stage.stage_index] != self.stage
        ):
            raise DynamicLfm2AdapterError(
                "Selected stage is not the exact indexed stage in its plan."
            )
        if not self.ledger_decision.allowed:
            raise DynamicLfm2AdapterError(
                "A selected target must have an allowed ledger decision."
            )
        if (
            self.signature.method_id != TARGET_SELECTION_METHOD
            or self.signature.method_version != 1
            or self.signature.family_id != "lfm2_progressive_target_selection"
            or not self.signature.zero_gradient
        ):
            raise DynamicLfm2AdapterError(
                "Selected target signature has the wrong method contract."
            )
        configuration = dict(self.signature.configuration)
        expected_configuration = {
            "stage_index": self.stage.stage_index,
            "changed_circuit": self.stage.changed_circuit,
            "exact_target_unique_parameters": self.stage.unique_parameters,
            "cumulative_parameters_removed": (self.stage.cumulative_parameters_removed),
            "query_heads": self.stage.topology.query_heads,
            "key_value_heads": self.stage.topology.key_value_heads,
            "intermediate_size": self.stage.topology.intermediate_size,
            "target_plan_fingerprint": self.plan.fingerprint,
        }
        for name, expected in expected_configuration.items():
            if configuration.get(name) != expected:
                raise DynamicLfm2AdapterError(
                    f"Selected target signature {name} does not match its stage."
                )
        parent_name = configuration.get("parent_name")
        tokenizer_fingerprint = configuration.get("tokenizer_fingerprint")
        if not isinstance(parent_name, str) or not parent_name.strip():
            raise DynamicLfm2AdapterError(
                "Selected target signature lacks a parent name."
            )
        _digest(tokenizer_fingerprint, "selected tokenizer_fingerprint")
        expected_configuration.update(
            {
                "parent_name": parent_name,
                "tokenizer_fingerprint": tokenizer_fingerprint,
            }
        )
        if configuration != expected_configuration:
            raise DynamicLfm2AdapterError(
                "Selected target signature configuration is not exact."
            )
        expected_transition = (
            f"q{self.plan.source.query_heads}_kv"
            f"{self.plan.source.key_value_heads}_"
            f"mlp{self.plan.source.intermediate_size}->"
            f"q{self.stage.topology.query_heads}_kv"
            f"{self.stage.topology.key_value_heads}_"
            f"mlp{self.stage.topology.intermediate_size}"
        )
        if (
            self.signature.topology_transition != expected_transition
            or self.signature.selection_policy
            != "first_unclosed_smallest_representable_exact_budget"
            or self.signature.fit_objective
            != "coupled_lfm2_closed_loop_residual_attention"
            or self.signature.promotion_gate
            != "automatic_behavior_then_final_manual_review"
        ):
            raise DynamicLfm2AdapterError(
                "Selected target signature policy contract is inconsistent."
            )
        if len(self.signature.parent_graph_fingerprints) != 1:
            raise DynamicLfm2AdapterError(
                "Selected LFM2 target must bind exactly one parent graph."
            )
        parent_states = dict(self.signature.parent_state_fingerprints)
        if set(parent_states) != {parent_name}:
            raise DynamicLfm2AdapterError(
                "Selected target parent state name differs from its binding."
            )
        target_identity = {
            "schema": ADAPTER_SCHEMA,
            "kind": "unmaterialized_lfm2_exact_budget_target",
            "source_graph_fingerprint": (self.signature.parent_graph_fingerprints[0]),
            "stage": self.stage.as_dict(),
        }
        expected_child_graph = canonical_sha256(target_identity)
        expected_child_state = canonical_sha256(
            {
                "kind": "preconstruction_blank_identity",
                "child_graph_fingerprint": expected_child_graph,
                "exact_target_unique_parameters": self.stage.unique_parameters,
            }
        )
        if (
            self.signature.child_graph_fingerprint != expected_child_graph
            or self.signature.child_state_fingerprint != expected_child_state
        ):
            raise DynamicLfm2AdapterError(
                "Selected target signature child identity is inconsistent."
            )
        expected_target_unit = (
            "lfm2:"
            f"q{self.stage.topology.query_heads}:"
            f"kv{self.stage.topology.key_value_heads}:"
            f"mlp{self.stage.topology.intermediate_size}:"
            f"p{self.stage.unique_parameters}"
        )
        if self.signature.target_unit_ids != (expected_target_unit,):
            raise DynamicLfm2AdapterError(
                "Selected target unit ID differs from its exact stage."
            )

    @property
    def exact_target_unique_parameters(self) -> int:
        return self.stage.unique_parameters

    @property
    def parent_name(self) -> str:
        return str(dict(self.signature.configuration)["parent_name"])

    @property
    def tokenizer_fingerprint(self) -> str:
        return _digest(
            dict(self.signature.configuration)["tokenizer_fingerprint"],
            "selected tokenizer_fingerprint",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "plan_fingerprint": self.plan.fingerprint,
            "stage": self.stage.as_dict(),
            "signature": self.signature.as_dict(),
            "ledger_decision": {
                "allowed": self.ledger_decision.allowed,
                "reason": self.ledger_decision.reason,
                "blocking_record_sha256": self.ledger_decision.blocking_record_sha256,
                "closure_level": self.ledger_decision.closure_level,
            },
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "parent_name": self.parent_name,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }


def encode_lfm2_layer_types(layer_types: Sequence[str]) -> str:
    """Encode an ordered LFM2 layer topology into a request-safe scalar."""

    values = tuple(str(value).strip() for value in layer_types)
    if not values or any(value not in {"conv", "full_attention"} for value in values):
        raise DynamicLfm2AdapterError(
            "LFM2 layer types must be a nonempty ordered sequence of "
            "'conv' or 'full_attention'."
        )
    return ",".join(values)


def decode_lfm2_layer_types(value: Any) -> tuple[str, ...]:
    """Decode and validate an ordered request/config layer topology."""

    if not isinstance(value, str) or not value.strip():
        raise DynamicLfm2AdapterError(
            "target_layer_types must be a nonempty comma-separated string."
        )
    result = tuple(part.strip() for part in value.split(","))
    encode_lfm2_layer_types(result)
    return result


def encode_lfm2_source_spans(spans: Sequence[Sequence[int]]) -> str:
    """Encode target-index-ordered source spans without losing their order."""

    normalized = tuple(tuple(span) for span in spans)
    if not normalized:
        raise DynamicLfm2AdapterError("LFM2 source spans must not be empty.")
    for span in normalized:
        if (
            not span
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in span
            )
            or tuple(range(span[0], span[-1] + 1)) != span
        ):
            raise DynamicLfm2AdapterError(
                "Every LFM2 source span must be a nonempty contiguous sequence "
                "of non-negative layer indices."
            )
    return ";".join(",".join(str(index) for index in span) for span in normalized)


def decode_lfm2_source_spans(value: Any) -> tuple[tuple[int, ...], ...]:
    """Decode the canonical ordered source-span request scalar."""

    if not isinstance(value, str) or not value.strip():
        raise DynamicLfm2AdapterError(
            "ordered_source_spans must be a nonempty semicolon-separated string."
        )
    try:
        result = tuple(
            tuple(int(part.strip()) for part in span.split(","))
            for span in value.split(";")
        )
    except ValueError as exc:
        raise DynamicLfm2AdapterError(
            "ordered_source_spans contains a non-integer layer index."
        ) from exc
    encode_lfm2_source_spans(result)
    return result


def _validate_lfm2_hybrid_layout(
    *,
    source_layer_types: Sequence[str],
    target_layer_types: Sequence[str],
    ordered_source_spans: Sequence[Sequence[int]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, ...], ...]]:
    source = decode_lfm2_layer_types(encode_lfm2_layer_types(source_layer_types))
    target = decode_lfm2_layer_types(encode_lfm2_layer_types(target_layer_types))
    spans = decode_lfm2_source_spans(
        encode_lfm2_source_spans(ordered_source_spans)
    )
    if len(spans) != len(target):
        raise DynamicLfm2AdapterError(
            "LFM2 source-span count must equal the target layer depth."
        )
    if len(target) >= len(source):
        raise DynamicLfm2AdapterError(
            "The hybrid model fold requires a strictly shallower LFM2 child."
        )
    flattened = tuple(index for span in spans for index in span)
    if flattened != tuple(range(len(source))):
        raise DynamicLfm2AdapterError(
            "Ordered LFM2 source spans must partition every parent layer "
            "exactly once in ascending order."
        )
    return source, target, spans


def _hybrid_target_signature(
    *,
    parent_graph: ModelCircuitGraph,
    parent_state_fingerprint: str,
    source_layer_types: tuple[str, ...],
    target_layer_types: tuple[str, ...],
    ordered_source_spans: tuple[tuple[int, ...], ...],
    topology: Lfm2Topology,
    exact_target_unique_parameters: int,
    evidence_fingerprint: str,
) -> ExperimentSignature:
    if parent_graph.tokenizer_fingerprint is None:
        raise DynamicLfm2AdapterError(
            "Hybrid LFM2 target selection requires a tokenizer fingerprint."
        )
    configuration = {
        "parent_name": parent_graph.model_name,
        "tokenizer_fingerprint": parent_graph.tokenizer_fingerprint,
        "source_depth": len(source_layer_types),
        "target_depth": len(target_layer_types),
        "source_layer_types": encode_lfm2_layer_types(source_layer_types),
        "target_layer_types": encode_lfm2_layer_types(target_layer_types),
        "ordered_source_spans": encode_lfm2_source_spans(ordered_source_spans),
        "query_heads": topology.query_heads,
        "key_value_heads": topology.key_value_heads,
        "intermediate_size": topology.intermediate_size,
        "exact_target_unique_parameters": exact_target_unique_parameters,
    }
    child_graph_fingerprint = canonical_sha256(
        {
            "schema": ADAPTER_SCHEMA,
            "kind": "unmaterialized_lfm2_hybrid_model_target",
            "source_graph_fingerprint": parent_graph.fingerprint,
            "configuration": configuration,
        }
    )
    child_state_fingerprint = canonical_sha256(
        {
            "kind": "preconstruction_blank_identity",
            "child_graph_fingerprint": child_graph_fingerprint,
            "exact_target_unique_parameters": exact_target_unique_parameters,
        }
    )
    return ExperimentSignature(
        method_id=HYBRID_MODEL_METHOD,
        method_version=1,
        family_id="lfm2_hybrid_residual_attention_model",
        parent_graph_fingerprints=(parent_graph.fingerprint,),
        parent_state_fingerprints=(
            (
                parent_graph.model_name,
                _digest(parent_state_fingerprint, "parent_state_fingerprint"),
            ),
        ),
        child_graph_fingerprint=child_graph_fingerprint,
        child_state_fingerprint=child_state_fingerprint,
        evidence_fingerprint=_digest(evidence_fingerprint, "evidence_fingerprint"),
        source_unit_ids=tuple(
            node.node_id for node in parent_graph.nodes if node.layer_index is not None
        ),
        target_unit_ids=tuple(
            f"layer.{index:03d}" for index in range(len(target_layer_types))
        ),
        topology_transition=(
            f"depth{len(source_layer_types)}->"
            f"depth{len(target_layer_types)}_q{topology.query_heads}_"
            f"kv{topology.key_value_heads}_mlp{topology.intermediate_size}"
        ),
        selection_policy="explicit_ordered_source_span_model_fold",
        fit_objective=(
            "joint_depth_attention_swiglu_ordered_residual_endpoint_compaction"
        ),
        promotion_gate="cached_whole_output_behavior_then_fresh_reload",
        zero_gradient=True,
        configuration=tuple(configuration.items()),
    )


@dataclass(frozen=True, slots=True)
class Lfm2HybridSelectedTarget:
    """Authenticated opt-in whole-model LFM2 depth/topology target."""

    topology: Lfm2Topology
    source_layer_types: tuple[str, ...]
    target_layer_types: tuple[str, ...]
    ordered_source_spans: tuple[tuple[int, ...], ...]
    exact_target_unique_parameters: int
    signature: ExperimentSignature
    ledger_decision: LedgerDecision

    def __post_init__(self) -> None:
        source, target, spans = _validate_lfm2_hybrid_layout(
            source_layer_types=self.source_layer_types,
            target_layer_types=self.target_layer_types,
            ordered_source_spans=self.ordered_source_spans,
        )
        object.__setattr__(self, "source_layer_types", source)
        object.__setattr__(self, "target_layer_types", target)
        object.__setattr__(self, "ordered_source_spans", spans)
        object.__setattr__(
            self,
            "exact_target_unique_parameters",
            _positive_int(
                self.exact_target_unique_parameters,
                "exact_target_unique_parameters",
            ),
        )
        if not isinstance(self.topology, Lfm2Topology):
            raise DynamicLfm2AdapterError("topology must be Lfm2Topology.")
        if not isinstance(self.signature, ExperimentSignature):
            raise DynamicLfm2AdapterError(
                "Hybrid selected target signature has the wrong type."
            )
        if not isinstance(self.ledger_decision, LedgerDecision):
            raise DynamicLfm2AdapterError(
                "Hybrid selected target ledger decision has the wrong type."
            )
        if not self.ledger_decision.allowed:
            raise DynamicLfm2AdapterError(
                "A selected hybrid target must have an allowed ledger decision."
            )
        configuration = dict(self.signature.configuration)
        expected = {
            "parent_name": self.parent_name,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "source_depth": len(source),
            "target_depth": len(target),
            "source_layer_types": encode_lfm2_layer_types(source),
            "target_layer_types": encode_lfm2_layer_types(target),
            "ordered_source_spans": encode_lfm2_source_spans(spans),
            "query_heads": self.topology.query_heads,
            "key_value_heads": self.topology.key_value_heads,
            "intermediate_size": self.topology.intermediate_size,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
        }
        if (
            self.signature.method_id != HYBRID_MODEL_METHOD
            or self.signature.method_version != 1
            or self.signature.family_id
            != "lfm2_hybrid_residual_attention_model"
            or not self.signature.zero_gradient
            or configuration != expected
        ):
            raise DynamicLfm2AdapterError(
                "Hybrid selected target signature differs from its topology."
            )

    @property
    def parent_name(self) -> str:
        value = dict(self.signature.configuration).get("parent_name")
        if not isinstance(value, str) or not value:
            raise DynamicLfm2AdapterError(
                "Hybrid selected target lacks its parent binding."
            )
        return value

    @property
    def tokenizer_fingerprint(self) -> str:
        return _digest(
            dict(self.signature.configuration).get("tokenizer_fingerprint"),
            "hybrid tokenizer_fingerprint",
        )

    @property
    def target_depth(self) -> int:
        return len(self.target_layer_types)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "kind": "lfm2_hybrid_selected_target",
            "topology": self.topology.as_dict(),
            "source_layer_types": list(self.source_layer_types),
            "target_layer_types": list(self.target_layer_types),
            "ordered_source_spans": [
                list(span) for span in self.ordered_source_spans
            ],
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "signature": self.signature.as_dict(),
            "ledger_decision": {
                "allowed": self.ledger_decision.allowed,
                "reason": self.ledger_decision.reason,
                "blocking_record_sha256": (
                    self.ledger_decision.blocking_record_sha256
                ),
                "closure_level": self.ledger_decision.closure_level,
            },
        }


def _heterogeneous_scaffold_geometry(
    descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor,
) -> tuple[int, int, int, int]:
    """Return the deterministic native-only I/Q/KV/D construction scaffold."""

    first_intermediate = descriptor.layers[0].mlp_intermediate_size
    for layer in descriptor.layers:
        if layer.layer_type == "full_attention":
            assert layer.query_heads is not None
            assert layer.key_value_heads is not None
            assert layer.head_dim is not None
            return (
                first_intermediate,
                layer.query_heads,
                layer.key_value_heads,
                layer.head_dim,
            )
    return (first_intermediate, 1, 1, 2)


def _restore_heterogeneous_target_artifacts(
    configuration: Mapping[str, JsonScalar],
) -> tuple[
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
    Lfm2HeterogeneousLayoutSelectionReceipt,
] | None:
    """Restore the all-or-nothing physical descriptor/selection seal bundle."""

    present = _DENSE_BUDGET_HETEROGENEOUS_CONFIGURATION_KEYS.intersection(
        configuration
    )
    if not present:
        return None
    if present != _DENSE_BUDGET_HETEROGENEOUS_CONFIGURATION_KEYS:
        raise DynamicLfm2AdapterError(
            "Dense budget heterogeneous physical-layout markers are incomplete."
        )
    descriptor_json = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY
    ]
    descriptor_sha256 = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY
    ]
    selection_json = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY
    ]
    selection_sha256 = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY
    ]
    if not all(
        isinstance(value, str)
        for value in (
            descriptor_json,
            descriptor_sha256,
            selection_json,
            selection_sha256,
        )
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget heterogeneous markers must be ASCII JSON/digest strings."
        )
    try:
        descriptor = restore_lfm2_heterogeneous_physical_layout(
            descriptor_json,
            expected_descriptor_sha256=_digest(
                descriptor_sha256,
                "heterogeneous descriptor SHA-256",
            ),
        )
        selection_receipt = Lfm2HeterogeneousLayoutSelectionReceipt.restore(
            selection_json
        )
    except (TypeError, ValueError) as exc:
        raise DynamicLfm2AdapterError(
            "Dense budget heterogeneous physical-layout artifacts failed "
            "authentication."
        ) from exc
    decision = selection_receipt.allocation_decision
    if (
        descriptor.serialize().decode("ascii") != descriptor_json
        or selection_receipt.serialize().decode("ascii") != selection_json
        or selection_receipt.receipt_sha256
        != _digest(selection_sha256, "heterogeneous selection receipt SHA-256")
        or selection_receipt.descriptor_sha256 != descriptor.descriptor_sha256
        or descriptor.source_topology_sha256
        != decision.source_graph_fingerprint
        or descriptor.allocation_evidence_sha256
        != decision.allocation_decision_sha256
        or descriptor.token_bridge_evidence_sha256
        != decision.token_bridge_evidence_sha256
        or descriptor.accounting.unique_parameter_elements
        != decision.exact_target_unique_parameters
        or tuple(
            layer.selection_evidence_sha256 for layer in descriptor.layers
        )
        != decision.selected_layer_option_sha256
        or len(decision.selected_layer_option_sha256) != descriptor.layer_count
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget heterogeneous descriptor/selection/source/count seals "
            "differ."
        )
    return descriptor, selection_receipt


@dataclass(frozen=True, slots=True)
class Lfm2DenseBudgetedSelectedTarget:
    """Authenticated width/depth topology emitted by the dense budget catalog.

    This payload is deliberately a target-materialization contract, not a
    claim that the existing same-width LFM2 fold operator can execute it.
    """

    source_parent: str
    source_graph_fingerprint: str
    source_state_fingerprint: str
    parent_tokenizer_semantics_sha256: str
    child_tokenizer_semantics_sha256: str
    tokenizer_bridge_artifact_sha256: str | None
    tokenizer_transport_profile_sha256: str | None
    catalog_id: str
    component_allocation_policy: str
    component_allocation_receipt_sha256: str
    component_allocation_topology_sha256: str
    hidden_size: int
    intermediate_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    vocabulary_size: int
    fir_length: int
    layer_types: tuple[str, ...]
    ordered_source_spans: tuple[tuple[int, ...], ...]
    tie_word_embeddings: bool
    exact_target_unique_parameters: int
    exact_target_logical_parameters: int
    configuration: tuple[tuple[str, JsonScalar], ...]
    heterogeneous_layout_descriptor: (
        Lfm2HeterogeneousPhysicalLayoutDescriptor | None
    ) = None
    heterogeneous_layout_selection_receipt: (
        Lfm2HeterogeneousLayoutSelectionReceipt | None
    ) = None

    def __post_init__(self) -> None:
        for name in ("source_parent", "catalog_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DynamicLfm2AdapterError(
                    f"Dense budget target {name} must be nonempty."
                )
        for name in (
            "source_graph_fingerprint",
            "source_state_fingerprint",
            "parent_tokenizer_semantics_sha256",
            "child_tokenizer_semantics_sha256",
            "component_allocation_receipt_sha256",
            "component_allocation_topology_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        if self.component_allocation_policy != COMPONENT_ALLOCATION_POLICY:
            raise DynamicLfm2AdapterError(
                "Dense budget target has an unknown component-allocation "
                "policy."
            )
        bridge_values = (
            self.tokenizer_bridge_artifact_sha256,
            self.tokenizer_transport_profile_sha256,
        )
        if any(value is None for value in bridge_values) != all(
            value is None for value in bridge_values
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget tokenizer bridge artifact/profile seals must be "
                "supplied together."
            )
        for name in (
            "tokenizer_bridge_artifact_sha256",
            "tokenizer_transport_profile_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _digest(value, name))
        if self.tokenizer_bridge_required == (
            self.parent_tokenizer_semantics_sha256
            == self.child_tokenizer_semantics_sha256
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget tokenizer semantics and bridge declaration "
                "disagree."
            )
        for name in (
            "hidden_size",
            "intermediate_size",
            "query_heads",
            "key_value_heads",
            "head_dim",
            "vocabulary_size",
            "fir_length",
            "exact_target_unique_parameters",
            "exact_target_logical_parameters",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        if self.query_heads % self.key_value_heads:
            raise DynamicLfm2AdapterError(
                "Dense budget target query heads must be divisible by KV heads."
            )
        if not isinstance(self.tie_word_embeddings, bool):
            raise DynamicLfm2AdapterError(
                "Dense budget target tie_word_embeddings must be boolean."
            )
        layer_types = decode_lfm2_layer_types(
            encode_lfm2_layer_types(self.layer_types)
        )
        spans = decode_lfm2_source_spans(
            encode_lfm2_source_spans(self.ordered_source_spans)
        )
        if len(layer_types) != len(spans):
            raise DynamicLfm2AdapterError(
                "Dense budget target needs one source span per target layer."
            )
        flattened = tuple(index for span in spans for index in span)
        if flattened != tuple(range(len(flattened))):
            raise DynamicLfm2AdapterError(
                "Dense budget source spans must cover the parent once in order."
            )
        object.__setattr__(self, "layer_types", layer_types)
        object.__setattr__(self, "ordered_source_spans", spans)
        configuration = tuple(sorted(self.configuration))
        if len({name for name, _value in configuration}) != len(configuration):
            raise DynamicLfm2AdapterError(
                "Dense budget target configuration keys must be unique."
            )
        object.__setattr__(self, "configuration", configuration)
        configuration_values = dict(configuration)
        configured_heterogeneous = _restore_heterogeneous_target_artifacts(
            configuration_values
        )
        supplied_heterogeneous = (
            self.heterogeneous_layout_descriptor,
            self.heterogeneous_layout_selection_receipt,
        )
        if any(value is None for value in supplied_heterogeneous) != all(
            value is None for value in supplied_heterogeneous
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget heterogeneous descriptor and selection receipt "
                "must be supplied together."
            )
        if configured_heterogeneous is None:
            if any(value is not None for value in supplied_heterogeneous):
                raise DynamicLfm2AdapterError(
                    "Dense budget heterogeneous typed artifacts have no "
                    "configuration seal bundle."
                )
        else:
            if any(value is None for value in supplied_heterogeneous):
                raise DynamicLfm2AdapterError(
                    "Dense budget heterogeneous configuration requires its "
                    "restored typed artifacts."
                )
            try:
                supplied_descriptor = restore_lfm2_heterogeneous_physical_layout(
                    self.heterogeneous_layout_descriptor,
                    expected_descriptor_sha256=(
                        configured_heterogeneous[0].descriptor_sha256
                    ),
                )
                supplied_selection = (
                    Lfm2HeterogeneousLayoutSelectionReceipt.restore(
                        self.heterogeneous_layout_selection_receipt
                    )
                )
            except (TypeError, ValueError) as exc:
                raise DynamicLfm2AdapterError(
                    "Dense budget heterogeneous typed artifacts failed "
                    "authentication."
                ) from exc
            if (
                supplied_descriptor != configured_heterogeneous[0]
                or supplied_selection != configured_heterogeneous[1]
            ):
                raise DynamicLfm2AdapterError(
                    "Dense budget heterogeneous typed artifacts differ from "
                    "their configuration seals."
                )
            object.__setattr__(
                self,
                "heterogeneous_layout_descriptor",
                supplied_descriptor,
            )
            object.__setattr__(
                self,
                "heterogeneous_layout_selection_receipt",
                supplied_selection,
            )
        if (
            configured_heterogeneous is None
            and self.query_heads * self.head_dim > self.hidden_size
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget target query width exceeds its residual width."
            )
        token_interface_marker = configuration_values.get(
            LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY
        )
        if token_interface_marker is None:
            token_interface_mode = LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
        elif (
            token_interface_marker
            == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        ):
            token_interface_mode = LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        else:
            raise DynamicLfm2AdapterError(
                "Dense budget target token-interface mode marker must be "
                "absent (preserve) or explicitly untied."
            )
        if (
            token_interface_mode == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
            and self.tie_word_embeddings
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget untied token-interface marker requires distinct "
                "input and output parameters."
            )
        expected_topology_configuration: dict[str, JsonScalar] = {
            "model_type": "lfm2",
            "vocab_size": self.vocabulary_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": len(self.layer_types),
            "num_attention_heads": self.query_heads,
            "num_key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "layer_types": encode_lfm2_layer_types(self.layer_types),
            "conv_L_cache": self.fir_length,
            "block_auto_adjust_ff_dim": False,
            "tie_word_embeddings": self.tie_word_embeddings,
            "_neural_forge_ordered_source_spans": (
                encode_lfm2_source_spans(self.ordered_source_spans)
            ),
            "_neural_forge_target_layer_types": (
                encode_lfm2_layer_types(self.layer_types)
            ),
            _DENSE_BUDGET_CATALOG_MARKER: self.catalog_id,
            COMPONENT_ALLOCATION_POLICY_CONFIG_KEY: (
                self.component_allocation_policy
            ),
            COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY: (
                self.component_allocation_receipt_sha256
            ),
            COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY: (
                self.component_allocation_topology_sha256
            ),
        }
        if any(
            configuration_values.get(name) != value
            for name, value in expected_topology_configuration.items()
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget target fields differ from their authenticated "
                "catalog configuration."
            )
        marker = configuration_values.get(
            "_neural_forge_tokenizer_bridge_required"
        )
        if marker is not self.tokenizer_bridge_required:
            raise DynamicLfm2AdapterError(
                "Dense budget tokenizer bridge marker differs from selected "
                "target metadata."
            )
        if self.tokenizer_bridge_required:
            expected_bridge = {
                TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY: (
                    self.tokenizer_bridge_artifact_sha256
                ),
                TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY: (
                    self.tokenizer_transport_profile_sha256
                ),
                PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                    self.parent_tokenizer_semantics_sha256
                ),
                CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                    self.child_tokenizer_semantics_sha256
                ),
            }
            if any(
                configuration_values.get(name) != value
                for name, value in expected_bridge.items()
            ):
                raise DynamicLfm2AdapterError(
                    "Dense budget tokenizer bridge configuration seals differ "
                    "from selected target metadata."
                )
        elif TOKENIZER_BRIDGE_CONFIG_KEYS.intersection(configuration_values):
            raise DynamicLfm2AdapterError(
                "Unbridged dense target contains tokenizer bridge seals."
            )
        native_tied = (
            self.vocabulary_size * self.hidden_size
            if self.tie_word_embeddings
            else 0
        )
        if (
            self.exact_target_logical_parameters
            - self.exact_target_unique_parameters
            != native_tied
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget target logical/unique counts disagree with its "
                "native token-interface tie."
            )
        descriptor = self.heterogeneous_layout_descriptor
        selection_receipt = self.heterogeneous_layout_selection_receipt
        if descriptor is not None:
            assert selection_receipt is not None
            scaffold = _heterogeneous_scaffold_geometry(descriptor)
            decision = selection_receipt.allocation_decision
            if (
                token_interface_mode != LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
                or self.tie_word_embeddings
                or descriptor.hidden_size != self.hidden_size
                or descriptor.vocab_size != self.vocabulary_size
                or descriptor.conv_kernel_size != self.fir_length
                or tuple(layer.layer_type for layer in descriptor.layers)
                != self.layer_types
                or scaffold
                != (
                    self.intermediate_size,
                    self.query_heads,
                    self.key_value_heads,
                    self.head_dim,
                )
                or descriptor.source_topology_sha256
                != self.source_graph_fingerprint
                or decision.source_graph_fingerprint
                != self.source_graph_fingerprint
                or decision.exact_target_unique_parameters
                != self.exact_target_unique_parameters
                or descriptor.accounting.unique_parameter_elements
                != self.exact_target_unique_parameters
                or descriptor.accounting.logical_parameter_elements
                != self.exact_target_logical_parameters
            ):
                raise DynamicLfm2AdapterError(
                    "Dense budget heterogeneous target fields differ from its "
                    "sealed per-layer physical layout."
                )
            return
        token_matrix = self.vocabulary_size * self.hidden_size
        body_parameters = self.hidden_size
        for layer_type in self.layer_types:
            body_parameters += 2 * self.hidden_size
            if layer_type == "conv":
                body_parameters += (
                    4 * self.hidden_size * self.hidden_size
                    + self.fir_length * self.hidden_size
                )
            else:
                body_parameters += (
                    2
                    * self.hidden_size
                    * self.head_dim
                    * (self.query_heads + self.key_value_heads)
                    + 2 * self.head_dim
                )
            body_parameters += (
                3 * self.hidden_size * self.intermediate_size
            )
        reproduced_unique = body_parameters + token_matrix * (
            1 if self.tie_word_embeddings else 2
        )
        reproduced_logical = body_parameters + 2 * token_matrix
        if (
            reproduced_unique != self.exact_target_unique_parameters
            or reproduced_logical != self.exact_target_logical_parameters
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget target exact counts do not reproduce from its "
                "authenticated topology."
            )

    @property
    def topology(self) -> Lfm2Topology:
        return Lfm2Topology(
            self.query_heads,
            self.key_value_heads,
            self.intermediate_size,
        )

    @property
    def target_depth(self) -> int:
        return len(self.layer_types)

    @property
    def tokenizer_fingerprint(self) -> str:
        """Backward-compatible child-tokenizer graph identity."""

        return self.child_tokenizer_semantics_sha256

    @property
    def tokenizer_bridge_required(self) -> bool:
        return self.tokenizer_bridge_artifact_sha256 is not None

    @property
    def heterogeneous_layout(self) -> bool:
        return self.heterogeneous_layout_descriptor is not None

    @property
    def target_token_interface_mode(self) -> str:
        return dict(self.configuration).get(
            LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
            LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE,
        )

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": ADAPTER_SCHEMA,
            "kind": "lfm2_dense_budgeted_selected_target",
            "source_parent": self.source_parent,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "source_state_fingerprint": self.source_state_fingerprint,
            "parent_tokenizer_semantics_sha256": (
                self.parent_tokenizer_semantics_sha256
            ),
            "child_tokenizer_semantics_sha256": (
                self.child_tokenizer_semantics_sha256
            ),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tokenizer_bridge_required": self.tokenizer_bridge_required,
            "tokenizer_bridge_artifact_sha256": (
                self.tokenizer_bridge_artifact_sha256
            ),
            "tokenizer_transport_profile_sha256": (
                self.tokenizer_transport_profile_sha256
            ),
            "catalog_id": self.catalog_id,
            "component_allocation_policy": self.component_allocation_policy,
            "component_allocation_receipt_sha256": (
                self.component_allocation_receipt_sha256
            ),
            "component_allocation_topology_sha256": (
                self.component_allocation_topology_sha256
            ),
            "component_allocation_verified_by_target_adapter": True,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "vocabulary_size": self.vocabulary_size,
            "fir_length": self.fir_length,
            "layer_types": list(self.layer_types),
            "ordered_source_spans": [
                list(span) for span in self.ordered_source_spans
            ],
            "tie_word_embeddings": self.tie_word_embeddings,
            "exact_target_unique_parameters": (
                self.exact_target_unique_parameters
            ),
            "exact_target_logical_parameters": (
                self.exact_target_logical_parameters
            ),
            "configuration": dict(self.configuration),
            "dense_parameters": True,
            "quantized": False,
            "pruned": False,
            "trained": False,
        }
        if self.heterogeneous_layout:
            assert self.heterogeneous_layout_descriptor is not None
            assert self.heterogeneous_layout_selection_receipt is not None
            value.update(
                {
                    "physical_layout_kind": LFM2_HETEROGENEOUS_LAYOUT_KIND,
                    "heterogeneous_layout_descriptor": (
                        self.heterogeneous_layout_descriptor.as_dict()
                    ),
                    "heterogeneous_layout_selection_receipt": (
                        self.heterogeneous_layout_selection_receipt.as_dict()
                    ),
                    "heterogeneous_per_layer_widths": True,
                    "fp32_physical_parameters": True,
                }
            )
        return value


Lfm2SelectedTargetLike = Lfm2SelectedTarget | Lfm2HybridSelectedTarget


def _validate_selected_target_against_parent(
    selected_target: Lfm2SelectedTarget,
    parent_graph: ModelCircuitGraph,
    parent_state_fingerprint: str,
) -> None:
    selected_target.validate()
    expected = _target_selection_signature(
        parent_graph,
        parent_state_fingerprint,
        selected_target.plan,
        selected_target.stage,
        selected_target.signature.evidence_fingerprint,
    )
    if expected != selected_target.signature:
        raise DynamicLfm2AdapterError(
            "Selected target signature differs from the inspected parent source."
        )


def validate_lfm2_selected_child(
    selected_target: Lfm2SelectedTarget,
    child_model: Any,
    child_graph: ModelCircuitGraph,
) -> Lfm2InspectedSource:
    """Prove that a materialized child is exactly the selected topology/budget."""

    if not isinstance(selected_target, Lfm2SelectedTarget):
        raise DynamicLfm2AdapterError("selected_target must be Lfm2SelectedTarget.")
    inspected = inspect_lfm2_progressive_source(child_model, child_graph)
    expected_stage = selected_target.stage
    mismatches = []
    if inspected.topology != expected_stage.topology:
        mismatches.append(f"topology {inspected.topology} != {expected_stage.topology}")
    if inspected.unique_parameters != expected_stage.unique_parameters:
        mismatches.append(
            "unique parameters "
            f"{inspected.unique_parameters} != {expected_stage.unique_parameters}"
        )
    if inspected.hidden_size != selected_target.plan.hidden_size:
        mismatches.append("hidden_size differs from the selected target plan")
    if inspected.head_dim != selected_target.plan.head_dim:
        mismatches.append("head_dim differs from the selected target plan")
    if (
        len(inspected.layer_types) != selected_target.plan.layer_count
        or inspected.layer_types.count("full_attention")
        != selected_target.plan.attention_layer_count
    ):
        mismatches.append("hybrid layer topology differs from the target plan")
    if mismatches:
        raise DynamicLfm2AdapterError(
            "Materialized child does not match the selected LFM2 target: "
            + "; ".join(mismatches)
        )
    return inspected


def _validate_hybrid_target_against_parent(
    selected_target: Lfm2HybridSelectedTarget,
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    parent_state_fingerprint: str,
) -> None:
    source = inspect_lfm2_progressive_source(parent_model, parent_graph)
    expected = _hybrid_target_signature(
        parent_graph=parent_graph,
        parent_state_fingerprint=parent_state_fingerprint,
        source_layer_types=source.layer_types,
        target_layer_types=selected_target.target_layer_types,
        ordered_source_spans=selected_target.ordered_source_spans,
        topology=selected_target.topology,
        exact_target_unique_parameters=(
            selected_target.exact_target_unique_parameters
        ),
        evidence_fingerprint=selected_target.signature.evidence_fingerprint,
    )
    if expected != selected_target.signature:
        raise DynamicLfm2AdapterError(
            "Hybrid selected target differs from the inspected parent source."
        )


def validate_lfm2_hybrid_child(
    selected_target: Lfm2HybridSelectedTarget,
    child_model: Any,
    child_graph: ModelCircuitGraph,
) -> Lfm2InspectedSource:
    """Prove depth, operator ordering, topology, and exact target budget."""

    if not isinstance(selected_target, Lfm2HybridSelectedTarget):
        raise DynamicLfm2AdapterError(
            "selected_target must be Lfm2HybridSelectedTarget."
        )
    inspected = inspect_lfm2_progressive_source(child_model, child_graph)
    mismatches = []
    if inspected.topology != selected_target.topology:
        mismatches.append(
            f"topology {inspected.topology} != {selected_target.topology}"
        )
    if inspected.layer_types != selected_target.target_layer_types:
        mismatches.append("ordered layer_types differ")
    if inspected.unique_parameters != selected_target.exact_target_unique_parameters:
        mismatches.append(
            "unique parameters "
            f"{inspected.unique_parameters} != "
            f"{selected_target.exact_target_unique_parameters}"
        )
    if len(inspected.layer_types) != selected_target.target_depth:
        mismatches.append("target depth differs")
    if mismatches:
        raise DynamicLfm2AdapterError(
            "Materialized child does not match the selected hybrid LFM2 "
            "target: " + "; ".join(mismatches)
        )
    return inspected


def _dense_budget_positive_int(
    configuration: Mapping[str, JsonScalar],
    name: str,
) -> int:
    value = configuration.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicLfm2AdapterError(
            f"Dense budget configuration {name} must be a positive integer."
        )
    return value


def _select_lfm2_dense_budgeted_target(
    request: DynamicTargetRequest,
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None,
) -> Lfm2DenseBudgetedSelectedTarget:
    """Authenticate one catalog configuration against its actual source."""

    configuration = dict(request.configuration)
    heterogeneous_artifacts = _restore_heterogeneous_target_artifacts(
        configuration
    )
    token_interface_marker_present = (
        LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY in configuration
    )
    token_interface_mode = configuration.get(
        LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
        LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE,
    )
    if token_interface_marker_present and (
        token_interface_mode != LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget target token-interface mode marker must explicitly "
            "request untied."
        )
    bridge_marker = configuration.get(
        "_neural_forge_tokenizer_bridge_required"
    )
    if not isinstance(bridge_marker, bool):
        raise DynamicLfm2AdapterError(
            "Dense budget tokenizer bridge marker must be boolean."
        )
    expected_configuration_keys = (
        _DENSE_BUDGET_UNTIED_CONFIGURATION_KEYS
        if token_interface_mode == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        else _DENSE_BUDGET_CONFIGURATION_KEYS
    )
    if bridge_marker:
        expected_configuration_keys = (
            expected_configuration_keys | TOKENIZER_BRIDGE_CONFIG_KEYS
        )
    if heterogeneous_artifacts is not None:
        if token_interface_mode != LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED:
            raise DynamicLfm2AdapterError(
                "Heterogeneous dense LFM2 targets require an explicitly untied "
                "token interface."
            )
        expected_configuration_keys = (
            expected_configuration_keys
            | _DENSE_BUDGET_HETEROGENEOUS_CONFIGURATION_KEYS
        )
    if set(configuration) != expected_configuration_keys:
        missing = sorted(expected_configuration_keys - set(configuration))
        extra = sorted(set(configuration) - expected_configuration_keys)
        raise DynamicLfm2AdapterError(
            "Dense budget LFM2 configuration is not the exact catalog schema "
            f"(missing={missing}, extra={extra})."
        )
    if request.fold_method_id not in {
        None,
        _DENSE_BUDGET_MODEL_METHOD,
    }:
        raise DynamicLfm2AdapterError(
            "The dense width/depth budget target only accepts its registered "
            "CPCF residual-attention fold method."
        )
    component_allocation_policy = configuration[
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY
    ]
    if component_allocation_policy != COMPONENT_ALLOCATION_POLICY:
        raise DynamicLfm2AdapterError(
            "Dense budget component-allocation policy is not registered."
        )
    component_allocation_receipt_sha256 = _digest(
        configuration[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY],
        "component_allocation_receipt_sha256",
    )
    component_allocation_topology_sha256 = _digest(
        configuration[COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY],
        "component_allocation_topology_sha256",
    )
    component_allocation_receipt = request.component_allocation_receipt
    if not isinstance(
        component_allocation_receipt,
        DenseComponentAllocationReceipt,
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget materialization requires its authenticated "
            "component-allocation receipt object."
        )
    if (
        component_allocation_receipt.policy != component_allocation_policy
        or component_allocation_receipt.fingerprint
        != component_allocation_receipt_sha256
        or component_allocation_receipt.topology_configuration_sha256
        != component_allocation_topology_sha256
        or component_allocation_receipt.source_graph_fingerprint
        != parent_graph.fingerprint
        or component_allocation_receipt.source_unique_parameters
        != parent_graph.unique_parameter_count
        or component_allocation_receipt.source_logical_parameters
        != parent_graph.logical_parameter_count
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget component-allocation receipt differs from its "
            "selected source or configuration seals."
        )
    if heterogeneous_artifacts is None:
        if component_allocation_receipt.physical_layout_kind is not None:
            raise DynamicLfm2AdapterError(
                "Dense budget heterogeneous component receipt has no physical "
                "layout configuration."
            )
    else:
        descriptor, selection_receipt = heterogeneous_artifacts
        if (
            component_allocation_receipt.physical_layout_kind
            != LFM2_HETEROGENEOUS_LAYOUT_KIND
            or component_allocation_receipt.physical_layout_descriptor_sha256
            != descriptor.descriptor_sha256
            or component_allocation_receipt.physical_layout_selection_receipt_sha256
            != selection_receipt.receipt_sha256
            or component_allocation_receipt.physical_layout_source_topology_sha256
            != descriptor.source_topology_sha256
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget heterogeneous component receipt differs from "
                "its descriptor/selection/source seals."
            )
    if configuration["model_type"] != "lfm2":
        raise DynamicLfm2AdapterError(
            "Dense budget target model_type must be lfm2."
        )
    catalog_id = configuration[_DENSE_BUDGET_CATALOG_MARKER]
    if not isinstance(catalog_id, str) or not catalog_id.strip():
        raise DynamicLfm2AdapterError(
            "Dense budget target lacks its topology-catalog provenance."
        )
    if configuration["block_auto_adjust_ff_dim"] is not False:
        raise DynamicLfm2AdapterError(
            "Dense budget target must disable automatic FFN width adjustment."
        )
    tie_word_embeddings = configuration["tie_word_embeddings"]
    if not isinstance(tie_word_embeddings, bool):
        raise DynamicLfm2AdapterError(
            "Dense budget tie_word_embeddings must be boolean."
        )
    if parent_graph.tokenizer_fingerprint is None:
        raise DynamicLfm2AdapterError(
            "Dense budget materialization requires a source tokenizer "
            "fingerprint."
        )
    bridge = tokenizer_bridge_artifact
    if bridge_marker:
        if not isinstance(bridge, StrictTokenizerBridgeArtifact):
            raise DynamicLfm2AdapterError(
                "Dense tokenizer-bridge materialization requires the explicit "
                "verified StrictTokenizerBridgeArtifact."
            )
        try:
            bridge.verify_files()
        except ValueError as exc:
            raise DynamicLfm2AdapterError(
                "Dense tokenizer bridge artifact failed verification."
            ) from exc
        expected_bridge_configuration = strict_tokenizer_bridge_configuration(
            bridge
        )
        if any(
            configuration.get(name) != value
            for name, value in expected_bridge_configuration.items()
        ):
            raise DynamicLfm2AdapterError(
                "Dense target configuration differs from its tokenizer bridge "
                "artifact seals."
            )
        if (
            parent_graph.tokenizer_fingerprint
            != bridge.donor_tokenizer_semantics_sha256
            or request.tokenizer_fingerprint
            != bridge.target_tokenizer_semantics_sha256
        ):
            raise DynamicLfm2AdapterError(
                "Dense tokenizer bridge parent/child semantics differ from "
                "the inspected source or requested child."
            )
    elif (
        request.tokenizer_fingerprint is None
        or request.tokenizer_fingerprint != parent_graph.tokenizer_fingerprint
    ):
        raise DynamicLfm2AdapterError(
            "Unbridged dense child tokenizer fingerprint must exactly match "
            "its selected source."
        )
    source = inspect_lfm2_progressive_source(parent_model, parent_graph)
    if heterogeneous_artifacts is not None:
        descriptor, selection_receipt = heterogeneous_artifacts
        decision = selection_receipt.allocation_decision
        if (
            descriptor.source_topology_sha256 != parent_graph.fingerprint
            or decision.source_graph_fingerprint != parent_graph.fingerprint
            or decision.source_structural_topology_fingerprint
            != parent_graph.structural_topology_fingerprint
            or decision.exact_target_unique_parameters
            != request.exact_unique_parameters
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget heterogeneous allocation is not bound to the "
                "loaded source graph/structure/exact request."
            )
    parent_config = getattr(parent_model, "config", None)
    if parent_config is None:
        raise DynamicLfm2AdapterError(
            "Dense budget materialization requires a loaded local LFM2 config."
        )
    source_vocabulary = _positive_int(
        int(_config_value(parent_config, ("vocab_size",), "vocabulary size")),
        "source vocabulary_size",
    )
    source_fir = _positive_int(
        int(_config_value(parent_config, ("conv_L_cache",), "conv FIR length")),
        "source conv_L_cache",
    )
    source_tied = bool(getattr(parent_config, "tie_word_embeddings", False))

    vocabulary_size = _dense_budget_positive_int(configuration, "vocab_size")
    hidden_size = _dense_budget_positive_int(configuration, "hidden_size")
    intermediate_size = _dense_budget_positive_int(
        configuration,
        "intermediate_size",
    )
    target_depth = _dense_budget_positive_int(
        configuration,
        "num_hidden_layers",
    )
    query_heads = _dense_budget_positive_int(
        configuration,
        "num_attention_heads",
    )
    key_value_heads = _dense_budget_positive_int(
        configuration,
        "num_key_value_heads",
    )
    head_dim = _dense_budget_positive_int(configuration, "head_dim")
    fir_length = _dense_budget_positive_int(configuration, "conv_L_cache")
    layer_types = decode_lfm2_layer_types(configuration["layer_types"])
    marker_layer_types = decode_lfm2_layer_types(
        configuration["_neural_forge_target_layer_types"]
    )
    spans = decode_lfm2_source_spans(
        configuration["_neural_forge_ordered_source_spans"]
    )
    if (
        layer_types != marker_layer_types
        or len(layer_types) != target_depth
        or len(spans) != target_depth
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget target depth/layer markers/source spans disagree."
        )
    if heterogeneous_artifacts is not None:
        descriptor, _selection_receipt = heterogeneous_artifacts
        if (
            descriptor.hidden_size != hidden_size
            or descriptor.vocab_size != vocabulary_size
            or descriptor.conv_kernel_size != fir_length
            or descriptor.layer_count != target_depth
            or tuple(layer.layer_type for layer in descriptor.layers)
            != layer_types
            or _heterogeneous_scaffold_geometry(descriptor)
            != (
                intermediate_size,
                query_heads,
                key_value_heads,
                head_dim,
            )
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget heterogeneous scalar scaffold/schedule differs "
                "from its authoritative per-layer descriptor."
            )
    flattened = tuple(index for span in spans for index in span)
    if flattened != tuple(range(len(source.layer_types))):
        raise DynamicLfm2AdapterError(
            "Dense budget source spans do not cover the inspected parent "
            "exactly once in causal order."
        )
    for target_type, span in zip(layer_types, spans, strict=True):
        if target_type not in {source.layer_types[index] for index in span}:
            raise DynamicLfm2AdapterError(
                "Dense budget target operator lacks a compatible source anchor "
                "inside its assigned span."
            )
    if (
        token_interface_mode == LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
        and tie_word_embeddings != source_tied
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget preserve mode changed its native token-interface tie."
        )
    if (
        token_interface_mode == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        and tie_word_embeddings
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget untied mode requires distinct input and output "
            "parameters."
        )
    if bridge_marker:
        assert bridge is not None
        profile = bridge.transport_profile
        if (
            profile.donor_row_capacity != source_vocabulary
            or profile.target_row_capacity != vocabulary_size
        ):
            raise DynamicLfm2AdapterError(
                "Dense tokenizer bridge row capacities differ from the source "
                "and target vocabulary operators."
            )
    elif vocabulary_size != source_vocabulary:
        raise DynamicLfm2AdapterError(
            "Dense budget target changed tokenizer vocabulary without a bridge."
        )
    if fir_length != source_fir:
        raise DynamicLfm2AdapterError(
            "Dense budget target FIR length differs from its source catalog."
        )
    if hidden_size > source.hidden_size or target_depth > len(source.layer_types):
        raise DynamicLfm2AdapterError(
            "Dense compression target exceeds one or more inspected source "
            "topology axes."
        )
    if heterogeneous_artifacts is None:
        if (
            intermediate_size > source.topology.intermediate_size
            or query_heads > source.topology.query_heads
            or key_value_heads > source.topology.key_value_heads
            or head_dim > source.head_dim
        ):
            raise DynamicLfm2AdapterError(
                "Dense compression target exceeds one or more inspected source "
                "topology axes."
            )
        if (
            query_heads % key_value_heads
            or query_heads * head_dim > hidden_size
            or head_dim % 2
        ):
            raise DynamicLfm2AdapterError(
                "Dense budget attention geometry is not instantiable by LFM2."
            )

        token_matrix = vocabulary_size * hidden_size
        unique_parameters = token_matrix * (1 if tie_word_embeddings else 2)
        logical_parameters = 2 * token_matrix
        body_parameters = hidden_size
        for layer_type in layer_types:
            body_parameters += 2 * hidden_size
            if layer_type == "conv":
                body_parameters += (
                    4 * hidden_size * hidden_size + fir_length * hidden_size
                )
            else:
                body_parameters += (
                    2
                    * hidden_size
                    * head_dim
                    * (query_heads + key_value_heads)
                    + 2 * head_dim
                )
            body_parameters += 3 * hidden_size * intermediate_size
        unique_parameters += body_parameters
        logical_parameters += body_parameters
    else:
        descriptor, _selection_receipt = heterogeneous_artifacts
        for physical in descriptor.layers:
            if physical.mlp_intermediate_size > source.topology.intermediate_size:
                raise DynamicLfm2AdapterError(
                    "Heterogeneous target MLP width exceeds its inspected source."
                )
            if physical.layer_type == "conv":
                assert physical.short_conv_internal_width is not None
                if physical.short_conv_internal_width > source.hidden_size:
                    raise DynamicLfm2AdapterError(
                        "Heterogeneous target ShortConv width exceeds its "
                        "inspected source."
                    )
            else:
                assert physical.query_heads is not None
                assert physical.key_value_heads is not None
                assert physical.head_dim is not None
                if (
                    physical.query_heads > source.topology.query_heads
                    or physical.key_value_heads > source.topology.key_value_heads
                    or physical.head_dim > source.head_dim
                ):
                    raise DynamicLfm2AdapterError(
                        "Heterogeneous target attention geometry exceeds its "
                        "inspected source or residual width."
                    )
        unique_parameters = descriptor.accounting.unique_parameter_elements
        logical_parameters = descriptor.accounting.logical_parameter_elements
    if unique_parameters != request.exact_unique_parameters:
        raise DynamicLfm2AdapterError(
            "Dense budget request count does not reproduce from its concrete "
            f"LFM2 configuration ({request.exact_unique_parameters} != "
            f"{unique_parameters})."
        )
    if (
        component_allocation_receipt.target_unique_parameters != unique_parameters
        or component_allocation_receipt.target_logical_parameters != logical_parameters
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget component-allocation receipt differs from the "
            "reproduced target parameter counts."
        )
    if unique_parameters >= parent_graph.unique_parameter_count:
        raise DynamicLfm2AdapterError(
            "Dense budget target does not compact the selected source."
        )
    if component_allocation_topology_sha256 != (
        component_allocation_topology_fingerprint(configuration)
    ):
        raise DynamicLfm2AdapterError(
            "Dense budget component-allocation topology seal differs from "
            "the materializable global configuration."
        )
    return Lfm2DenseBudgetedSelectedTarget(
        source_parent=parent_graph.model_name,
        source_graph_fingerprint=parent_graph.fingerprint,
        source_state_fingerprint=fingerprint_model_state(parent_model),
        parent_tokenizer_semantics_sha256=(
            parent_graph.tokenizer_fingerprint
        ),
        child_tokenizer_semantics_sha256=(
            request.tokenizer_fingerprint
        ),
        tokenizer_bridge_artifact_sha256=(
            None if bridge is None or not bridge_marker else bridge.artifact_sha256
        ),
        tokenizer_transport_profile_sha256=(
            None if bridge is None or not bridge_marker else bridge.profile_fingerprint
        ),
        catalog_id=catalog_id,
        component_allocation_policy=component_allocation_policy,
        component_allocation_receipt_sha256=(
            component_allocation_receipt_sha256
        ),
        component_allocation_topology_sha256=(
            component_allocation_topology_sha256
        ),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        vocabulary_size=vocabulary_size,
        fir_length=fir_length,
        layer_types=layer_types,
        ordered_source_spans=spans,
        tie_word_embeddings=tie_word_embeddings,
        exact_target_unique_parameters=unique_parameters,
        exact_target_logical_parameters=logical_parameters,
        configuration=request.configuration,
        heterogeneous_layout_descriptor=(
            None if heterogeneous_artifacts is None else heterogeneous_artifacts[0]
        ),
        heterogeneous_layout_selection_receipt=(
            None if heterogeneous_artifacts is None else heterogeneous_artifacts[1]
        ),
    )


def _validate_heterogeneous_materialization_receipt(
    selected_target: Lfm2DenseBudgetedSelectedTarget,
    receipt: Mapping[str, Any],
) -> None:
    descriptor = selected_target.heterogeneous_layout_descriptor
    if descriptor is None:
        raise DynamicLfm2AdapterError(
            "A heterogeneous physical receipt requires a selected descriptor."
        )
    audit_value = receipt.get("physical_materialization_audit")
    if not isinstance(audit_value, Mapping):
        raise DynamicLfm2AdapterError(
            "Heterogeneous blank materialization returned no physical audit."
        )
    audit = dict(audit_value)
    expected_count = selected_target.exact_target_unique_parameters
    if (
        receipt.get("schema") != LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA
        or receipt.get("status")
        != "authenticated-dense-fp32-blank-child-materialization"
        or receipt.get("descriptor_sha256") != descriptor.descriptor_sha256
        or receipt.get("parameter_elements") != expected_count
        or audit.get("schema") != LFM2_DYNAMIC_PHYSICAL_AUDIT_SCHEMA
        or audit.get("status") != "dense-fp32-physical-layout-authenticated"
        or audit.get("descriptor_sha256") != descriptor.descriptor_sha256
        or audit.get("parameter_elements") != expected_count
        or audit.get("parameter_bytes_fp32") != 4 * expected_count
        or audit.get("runtime_parameter_alias_group_count") != 0
        or audit.get("cross_layer_parameter_sharing") is not False
        or audit.get("parameters_quantized") != 0
        or audit.get("sparse_parameter_storage") is not False
        or audit.get("pruning_masks_present") is not False
        or audit.get("runtime_bridge_parameter_elements") != 0
        or audit.get("runtime_sidecar_parameter_elements") != 0
        or audit.get("all_parameters_frozen") is not True
        or audit.get("unique_parameter_object_count")
        != audit.get("parameter_state_key_count")
        or audit.get("unique_parameter_storage_count")
        != audit.get("parameter_state_key_count")
    ):
        raise DynamicLfm2AdapterError(
            "Heterogeneous blank materialization receipt does not prove the "
            "selected exact dense FP32/no-alias/no-sidecar contract."
        )


def _validate_lfm2_heterogeneous_dense_child(
    selected_target: Lfm2DenseBudgetedSelectedTarget,
    child_model: Any,
    child_graph: ModelCircuitGraph,
) -> Lfm2InspectedSource:
    descriptor = selected_target.heterogeneous_layout_descriptor
    if descriptor is None:  # pragma: no cover - private dispatch contract
        raise DynamicLfm2AdapterError(
            "Heterogeneous child validation requires a selected descriptor."
        )
    if child_graph.family != "lfm2":
        raise DynamicLfm2AdapterError(
            "Heterogeneous child graph must be an inspected LFM2 graph."
        )
    config = getattr(child_model, "config", None)
    try:
        recovered = extract_lfm2_dynamic_physical_layout_from_config(config)
    except (TypeError, ValueError) as exc:
        raise DynamicLfm2AdapterError(
            "Heterogeneous child config has no authenticated physical layout."
        ) from exc
    if recovered != descriptor:
        raise DynamicLfm2AdapterError(
            "Heterogeneous child config descriptor differs from selection."
        )
    expected_config_markers = _dense_budgeted_lfm2_config_markers(
        selected_target
    )
    possible_config_markers = {
        _DENSE_BUDGET_CATALOG_MARKER,
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
        COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
        "_neural_forge_dense_ordered_source_spans",
        "_neural_forge_dense_target_layer_types",
        "_neural_forge_tokenizer_bridge_required",
        LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
        *TOKENIZER_BRIDGE_CONFIG_KEYS,
    }
    if config is None or any(
        getattr(config, name, None) != value
        for name, value in expected_config_markers.items()
    ) or any(
        getattr(config, name, None) is not None
        for name in possible_config_markers - set(expected_config_markers)
    ):
        raise DynamicLfm2AdapterError(
            "Heterogeneous child config dense-budget provenance differs "
            "from selection."
        )
    metadata = dict(child_graph.metadata)
    expected_metadata = {
        "physical_layout": "heterogeneous_dense_fp32",
        "physical_layout_descriptor_sha256": descriptor.descriptor_sha256,
        "runtime_parameter_alias_group_count": 0,
        "cross_layer_parameter_sharing": False,
        "runtime_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
        "parameters_quantized": 0,
        "sparse_parameter_storage": False,
        "pruning_masks_present": False,
        "fold_method_id": _DENSE_BUDGET_MODEL_METHOD,
        "dense_topology_catalog": selected_target.catalog_id,
        "ordered_source_spans": encode_lfm2_source_spans(
            selected_target.ordered_source_spans
        ),
        "target_layer_types": encode_lfm2_layer_types(
            selected_target.layer_types
        ),
        "tokenizer_bridge_required": (
            selected_target.tokenizer_bridge_required
        ),
    }
    if selected_target.tokenizer_bridge_required:
        expected_metadata.update(
            {
                "tokenizer_bridge_artifact_sha256": (
                    selected_target.tokenizer_bridge_artifact_sha256
                ),
                "tokenizer_transport_profile_sha256": (
                    selected_target.tokenizer_transport_profile_sha256
                ),
                "parent_tokenizer_semantics_sha256": (
                    selected_target.parent_tokenizer_semantics_sha256
                ),
                "child_tokenizer_semantics_sha256": (
                    selected_target.child_tokenizer_semantics_sha256
                ),
            }
        )
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise DynamicLfm2AdapterError(
            "Heterogeneous inspected graph does not prove its sealed dense "
            "physical storage and whole-model fold contract."
        )
    if (
        child_graph.unique_parameter_count
        != descriptor.accounting.unique_parameter_elements
        or child_graph.logical_parameter_count
        != descriptor.accounting.logical_parameter_elements
        or child_graph.unique_parameter_bytes
        != descriptor.accounting.parameter_bytes_fp32
        or child_graph.tokenizer_fingerprint
        != selected_target.child_tokenizer_semantics_sha256
        or child_graph.layer_indices != tuple(range(descriptor.layer_count))
    ):
        raise DynamicLfm2AdapterError(
            "Heterogeneous inspected graph differs from descriptor accounting, "
            "token interface, or layer schedule."
        )
    for physical in descriptor.layers:
        mlp = child_graph.node(f"layer.{physical.layer_index:03d}.mlp")
        operator = child_graph.node(f"layer.{physical.layer_index:03d}.operator")
        mlp_attributes = dict(mlp.attributes)
        operator_attributes = dict(operator.attributes)
        if (
            mlp.kind != CircuitKind.GATED_MLP.value
            or mlp_attributes.get("intermediate_size")
            != physical.mlp_intermediate_size
            or operator_attributes.get("sealed_heterogeneous_physical_geometry")
            is not True
            or operator_attributes.get("selection_evidence_sha256")
            != physical.selection_evidence_sha256
            or operator_attributes.get("external_hidden_size")
            != descriptor.hidden_size
        ):
            raise DynamicLfm2AdapterError(
                f"Heterogeneous child layer {physical.layer_index} MLP/operator "
                "geometry differs from its sealed selection."
            )
        if physical.layer_type == "conv":
            writer_rank = physical.short_conv_writer_rank
            writer_topology = (
                "factorized" if writer_rank is not None else "dense"
            )
            if (
                operator.kind != CircuitKind.SHORT_CONV.value
                or operator_attributes.get("internal_channel_width")
                != physical.short_conv_internal_width
                or operator_attributes.get("writer_topology") != writer_topology
                or operator_attributes.get("writer_rank") != writer_rank
                or operator_attributes.get("writer_factor_count")
                != (2 if writer_rank is not None else 1)
            ):
                raise DynamicLfm2AdapterError(
                    f"Heterogeneous child layer {physical.layer_index} "
                    "ShortConv geometry differs from its descriptor."
                )
        elif (
            operator.kind != CircuitKind.ATTENTION.value
            or operator_attributes.get("query_heads") != physical.query_heads
            or operator_attributes.get("key_value_heads")
            != physical.key_value_heads
            or operator_attributes.get("head_dim") != physical.head_dim
        ):
            raise DynamicLfm2AdapterError(
                f"Heterogeneous child layer {physical.layer_index} attention "
                "geometry differs from its descriptor."
            )
    return Lfm2InspectedSource(
        topology=selected_target.topology,
        hidden_size=selected_target.hidden_size,
        head_dim=selected_target.head_dim,
        layer_types=selected_target.layer_types,
        unique_parameters=child_graph.unique_parameter_count,
        graph_fingerprint=child_graph.fingerprint,
    )


def validate_lfm2_dense_budgeted_child(
    selected_target: Lfm2DenseBudgetedSelectedTarget,
    child_model: Any,
    child_graph: ModelCircuitGraph,
) -> Lfm2InspectedSource:
    """Prove a catalog topology became the exact dense native LFM2 graph."""

    if not isinstance(selected_target, Lfm2DenseBudgetedSelectedTarget):
        raise DynamicLfm2AdapterError(
            "selected_target must be Lfm2DenseBudgetedSelectedTarget."
        )
    if selected_target.heterogeneous_layout:
        return _validate_lfm2_heterogeneous_dense_child(
            selected_target,
            child_model,
            child_graph,
        )
    inspected = inspect_lfm2_progressive_source(child_model, child_graph)
    config = getattr(child_model, "config", None)
    input_embeddings = getattr(child_model, "get_input_embeddings", lambda: None)()
    output_embeddings = getattr(child_model, "get_output_embeddings", lambda: None)()
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    native_tie = (
        input_weight is not None
        and output_weight is not None
        and input_weight is output_weight
    )
    mismatches = []
    if inspected.hidden_size != selected_target.hidden_size:
        mismatches.append("hidden_size differs")
    if inspected.topology != selected_target.topology:
        mismatches.append("attention/FFN topology differs")
    if inspected.head_dim != selected_target.head_dim:
        mismatches.append("head_dim differs")
    if inspected.layer_types != selected_target.layer_types:
        mismatches.append("ordered layer_types differ")
    if inspected.unique_parameters != selected_target.exact_target_unique_parameters:
        mismatches.append("unique parameter count differs")
    if (
        child_graph.logical_parameter_count
        != selected_target.exact_target_logical_parameters
    ):
        mismatches.append("logical parameter count differs")
    if native_tie != selected_target.tie_word_embeddings:
        mismatches.append("native token-interface tie differs")
    if config is None or bool(
        getattr(config, "tie_word_embeddings", not native_tie)
    ) != selected_target.tie_word_embeddings:
        mismatches.append("configured token-interface tie differs")
    configured_token_interface_mode = (
        None
        if config is None
        else getattr(
            config,
            LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
            None,
        )
    )
    expected_token_interface_marker = (
        None
        if selected_target.target_token_interface_mode
        == LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
        else LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
    )
    if configured_token_interface_mode != expected_token_interface_marker:
        mismatches.append("target token-interface mode marker differs")
    if config is None or int(getattr(config, "vocab_size", -1)) != (
        selected_target.vocabulary_size
    ):
        mismatches.append("vocabulary size differs")
    if config is None or int(getattr(config, "conv_L_cache", -1)) != (
        selected_target.fir_length
    ):
        mismatches.append("FIR length differs")
    if config is None or getattr(
        config,
        _DENSE_BUDGET_CATALOG_MARKER,
        None,
    ) != selected_target.catalog_id:
        mismatches.append("catalog provenance marker differs")
    if config is None or getattr(
        config,
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
        None,
    ) != selected_target.component_allocation_policy:
        mismatches.append("component-allocation policy marker differs")
    if config is None or getattr(
        config,
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
        None,
    ) != selected_target.component_allocation_receipt_sha256:
        mismatches.append("component-allocation receipt marker differs")
    if config is None or getattr(
        config,
        COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
        None,
    ) != selected_target.component_allocation_topology_sha256:
        mismatches.append("component-allocation topology marker differs")
    if config is None or getattr(
        config,
        "_neural_forge_dense_ordered_source_spans",
        None,
    ) != encode_lfm2_source_spans(selected_target.ordered_source_spans):
        mismatches.append("dense source-span provenance differs")
    if config is None or getattr(
        config,
        "_neural_forge_dense_target_layer_types",
        None,
    ) != encode_lfm2_layer_types(selected_target.layer_types):
        mismatches.append("dense target-layer provenance differs")
    if child_graph.tokenizer_fingerprint != (
        selected_target.child_tokenizer_semantics_sha256
    ):
        mismatches.append("child tokenizer semantics differ")
    if config is None or getattr(
        config,
        "_neural_forge_tokenizer_bridge_required",
        None,
    ) is not selected_target.tokenizer_bridge_required:
        mismatches.append("tokenizer bridge marker differs")
    if selected_target.tokenizer_bridge_required:
        expected_bridge = {
            TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY: (
                selected_target.tokenizer_bridge_artifact_sha256
            ),
            TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY: (
                selected_target.tokenizer_transport_profile_sha256
            ),
            PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                selected_target.parent_tokenizer_semantics_sha256
            ),
            CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                selected_target.child_tokenizer_semantics_sha256
            ),
        }
        if config is None or any(
            getattr(config, name, None) != value
            for name, value in expected_bridge.items()
        ):
            mismatches.append("tokenizer bridge seal bundle differs")
    if mismatches:
        raise DynamicLfm2AdapterError(
            "Materialized child does not match the selected dense LFM2 budget "
            "target: " + "; ".join(mismatches)
        )
    return inspected


def select_first_open_lfm2_target(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    *,
    terminal_topology: Lfm2Topology,
    evidence_fingerprint: str,
    ledger: ExperimentLedger | None = None,
    initial_mlp_step: int = 1,
    maximum_mlp_step: int = 64,
) -> Lfm2SelectedTarget:
    """Skip previously closed stages and return the first exact open budget."""

    if parent_graph.tokenizer_fingerprint is None:
        raise DynamicLfm2AdapterError(
            "LFM2 target selection requires a tokenizer fingerprint."
        )
    selected_ledger = ledger or ExperimentLedger()
    plan = derive_lfm2_target_ladder(
        parent_model,
        parent_graph,
        terminal_topology=terminal_topology,
        initial_mlp_step=initial_mlp_step,
        maximum_mlp_step=maximum_mlp_step,
    )
    parent_state_fingerprint = fingerprint_model_state(parent_model)
    for stage in plan.stages:
        signature = _target_selection_signature(
            parent_graph,
            parent_state_fingerprint,
            plan,
            stage,
            evidence_fingerprint,
        )
        decision = selected_ledger.decision(signature)
        if decision.allowed:
            return Lfm2SelectedTarget(plan, stage, signature, decision)
    raise DynamicLfm2AdapterError(
        "Every progressive LFM2 exact-budget stage is already closed."
    )


def select_exact_lfm2_target(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    *,
    terminal_topology: Lfm2Topology,
    exact_target_unique_parameters: int,
    evidence_fingerprint: str,
    ledger: ExperimentLedger | None = None,
    initial_mlp_step: int = 1,
    maximum_mlp_step: int = 64,
) -> Lfm2SelectedTarget:
    """Select one legal LFM2 topology through the generic exact-budget solver."""

    exact_target_unique_parameters = _positive_int(
        exact_target_unique_parameters,
        "exact_target_unique_parameters",
    )
    selected_ledger = ledger or ExperimentLedger()
    plan = derive_lfm2_target_ladder(
        parent_model,
        parent_graph,
        terminal_topology=terminal_topology,
        initial_mlp_step=initial_mlp_step,
        maximum_mlp_step=maximum_mlp_step,
    )
    options = tuple(
        TopologyOption(
            option_id=f"lfm2.stage.{stage.stage_index:04d}",
            group_id="lfm2.exact_target",
            local_unique_parameters=stage.unique_parameters,
            local_serialized_bytes=stage.unique_parameters * 2,
            predicted_risk=float(stage.stage_index),
            metadata=(
                ("stage_index", stage.stage_index),
                ("changed_circuit", stage.changed_circuit),
            ),
        )
        for stage in plan.stages
    )
    if not options:
        raise DynamicLfm2AdapterError(
            "The requested terminal topology creates no compact stage."
        )
    problem = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=exact_target_unique_parameters,
        groups=(TopologyChoiceGroup("lfm2.exact_target", options),),
        max_states=max(1, len(options)),
    )
    try:
        selection = solve_exact_topology_budget(problem)
    except DynamicFoldBudgetUnattainable as exc:
        raise DynamicLfm2AdapterError(str(exc)) from exc
    option_id = selection.selected_option_ids[0]
    stage_index = int(option_id.rsplit(".", 1)[1])
    stage = plan.stages[stage_index]
    parent_state_fingerprint = fingerprint_model_state(parent_model)
    signature = _target_selection_signature(
        parent_graph,
        parent_state_fingerprint,
        plan,
        stage,
        evidence_fingerprint,
    )
    decision = selected_ledger.decision(signature)
    if not decision.allowed:
        raise DynamicLfm2AdapterError(
            "The exact requested LFM2 target is already closed: " + decision.reason
        )
    return Lfm2SelectedTarget(plan, stage, signature, decision)


def select_lfm2_hybrid_target(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    *,
    topology: Lfm2Topology,
    target_layer_types: Sequence[str],
    ordered_source_spans: Sequence[Sequence[int]],
    exact_target_unique_parameters: int,
    evidence_fingerprint: str,
    ledger: ExperimentLedger | None = None,
) -> Lfm2HybridSelectedTarget:
    """Select one explicit shallower model target as a single atomic lane."""

    source = inspect_lfm2_progressive_source(parent_model, parent_graph)
    source_types, target_types, spans = _validate_lfm2_hybrid_layout(
        source_layer_types=source.layer_types,
        target_layer_types=target_layer_types,
        ordered_source_spans=ordered_source_spans,
    )
    exact_target_unique_parameters = _positive_int(
        exact_target_unique_parameters,
        "exact_target_unique_parameters",
    )
    if exact_target_unique_parameters >= source.unique_parameters:
        raise DynamicLfm2AdapterError(
            "The hybrid target must remove at least one unique parameter."
        )
    signature = _hybrid_target_signature(
        parent_graph=parent_graph,
        parent_state_fingerprint=fingerprint_model_state(parent_model),
        source_layer_types=source_types,
        target_layer_types=target_types,
        ordered_source_spans=spans,
        topology=topology,
        exact_target_unique_parameters=exact_target_unique_parameters,
        evidence_fingerprint=evidence_fingerprint,
    )
    selected_ledger = ledger if ledger is not None else ExperimentLedger()
    decision = selected_ledger.decision(signature)
    if not decision.allowed:
        raise DynamicLfm2AdapterError(
            "The exact requested hybrid LFM2 target is already closed: "
            + decision.reason
        )
    return Lfm2HybridSelectedTarget(
        topology=topology,
        source_layer_types=source_types,
        target_layer_types=target_types,
        ordered_source_spans=spans,
        exact_target_unique_parameters=exact_target_unique_parameters,
        signature=signature,
        ledger_decision=decision,
    )


@dataclass(frozen=True, slots=True)
class Lfm2EvidenceBuildRequest:
    """Restricted request that exposes cached parent data, never parent models."""

    action: FoldAction
    layer_index: int
    child_model: Any
    child_graph: ModelCircuitGraph
    child_state_fingerprint: str
    parent_cache_fingerprint: str
    parent_cache_payload: Any


@dataclass(frozen=True, slots=True)
class Lfm2PreparedEvidence:
    evidence: Lfm2ClosedLoopFoldEvidence
    child_forward_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, Lfm2ClosedLoopFoldEvidence):
            raise DynamicLfm2AdapterError(
                "prepared evidence must be Lfm2ClosedLoopFoldEvidence."
            )
        if (
            isinstance(self.child_forward_count, bool)
            or not isinstance(self.child_forward_count, int)
            or self.child_forward_count < 0
        ):
            raise DynamicLfm2AdapterError(
                "child_forward_count must be a non-negative integer."
            )


class Lfm2EvidenceBuilder(Protocol):
    def __call__(self, request: Lfm2EvidenceBuildRequest) -> Lfm2PreparedEvidence: ...


@dataclass(frozen=True, slots=True)
class Lfm2ModelEvidenceBuildRequest:
    """Restricted whole-model evidence request backed by the parent cache."""

    action: FoldAction
    child_model: Any
    child_graph: ModelCircuitGraph
    child_state_fingerprint: str
    selected_target: Lfm2HybridSelectedTarget
    parent_cache_fingerprint: str
    parent_cache_payload: Any


@dataclass(frozen=True, slots=True)
class Lfm2PreparedModelEvidence:
    """A generic model-fold evidence payload plus measured child forwards."""

    evidence: Any
    producer_state_fingerprint: str
    child_feature_state_fingerprint: str
    producer_plan_fingerprint: str
    child_forward_count: int = 0

    def __post_init__(self) -> None:
        try:
            from .hf_lfm2_model_fold import Lfm2ModelFoldEvidence
        except ImportError as exc:  # pragma: no cover - optional during import
            raise DynamicLfm2AdapterError(
                "The LFM2 model-fold primitive is unavailable."
            ) from exc
        if not isinstance(self.evidence, Lfm2ModelFoldEvidence):
            raise DynamicLfm2AdapterError(
                "prepared model evidence must be Lfm2ModelFoldEvidence."
            )
        producer_state = _digest(
            self.producer_state_fingerprint,
            "producer_state_fingerprint",
        )
        feature_state = _digest(
            self.child_feature_state_fingerprint,
            "child_feature_state_fingerprint",
        )
        object.__setattr__(
            self,
            "producer_state_fingerprint",
            producer_state,
        )
        object.__setattr__(
            self,
            "child_feature_state_fingerprint",
            feature_state,
        )
        object.__setattr__(
            self,
            "producer_plan_fingerprint",
            _digest(
                self.producer_plan_fingerprint,
                "producer_plan_fingerprint",
            ),
        )
        if producer_state != feature_state:
            raise DynamicLfm2AdapterError(
                "Hybrid child features were not captured from the exact "
                "producer-installed child state."
            )
        if (
            isinstance(self.child_forward_count, bool)
            or not isinstance(self.child_forward_count, int)
            or self.child_forward_count < 0
        ):
            raise DynamicLfm2AdapterError(
                "child_forward_count must be a non-negative integer."
            )


@dataclass(frozen=True, slots=True)
class Lfm2PreparedHybridProducerEvidence:
    """Sealed parent-only hybrid evidence; the blank has not been executed."""

    evidence: Any
    child_forward_count: int = 0

    def __post_init__(self) -> None:
        try:
            from .hf_lfm2_producer_first_model_fold import (
                Lfm2ProducerFirstEvidence,
            )
        except ImportError as exc:  # pragma: no cover - optional at import time
            raise DynamicLfm2AdapterError(
                "The producer-first LFM2 primitive is unavailable."
            ) from exc
        if not isinstance(self.evidence, Lfm2ProducerFirstEvidence):
            raise DynamicLfm2AdapterError(
                "prepared hybrid producer evidence has the wrong type."
            )
        if self.child_forward_count != 0:
            raise DynamicLfm2AdapterError(
                "Preparing hybrid producer evidence must not execute the "
                "compiler-zero child."
            )


class Lfm2ModelEvidenceBuilder(Protocol):
    def __call__(
        self,
        request: Lfm2ModelEvidenceBuildRequest,
    ) -> Lfm2PreparedHybridProducerEvidence: ...


@dataclass(frozen=True, slots=True)
class Lfm2ProducerFirstEvidenceBuildRequest:
    """Full-depth producer planning request; the live child is still blank."""

    action: FoldAction
    child_model: Any
    child_graph: ModelCircuitGraph
    blank_state_fingerprint: str
    selected_target: Lfm2SelectedTarget
    parent_cache_fingerprint: str
    parent_cache_payload: Any


@dataclass(frozen=True, slots=True)
class Lfm2PreparedProducerFirstEvidence:
    """Parent-only model evidence; no blank-child feature is materialized."""

    evidence: Any
    child_forward_count: int = 0

    def __post_init__(self) -> None:
        try:
            from .hf_lfm2_producer_first_model_fold import (
                Lfm2ProducerFirstEvidence,
            )
        except ImportError as exc:  # pragma: no cover - optional at import time
            raise DynamicLfm2AdapterError(
                "The producer-first LFM2 primitive is unavailable."
            ) from exc
        if not isinstance(self.evidence, Lfm2ProducerFirstEvidence):
            raise DynamicLfm2AdapterError(
                "prepared producer-first evidence has the wrong type."
            )
        if self.child_forward_count != 0:
            raise DynamicLfm2AdapterError(
                "Preparing producer-first evidence must not execute the blank child."
            )


class Lfm2ProducerFirstEvidenceBuilder(Protocol):
    def __call__(
        self,
        request: Lfm2ProducerFirstEvidenceBuildRequest,
    ) -> Lfm2PreparedProducerFirstEvidence: ...


def _evidence_value_identity(value: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise DynamicLfm2AdapterError(
            "PyTorch is required to fingerprint LFM2 evidence."
        ) from exc
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta" or value.layout != torch.strided:
            raise DynamicLfm2AdapterError(
                "LFM2 evidence tensors must be materialized and dense."
            )
        tensor = value.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
            "sha256": digest.hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _evidence_value_identity(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _evidence_value_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_evidence_value_identity(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise DynamicLfm2AdapterError(
        f"Unsupported LFM2 evidence value type: {type(value)!r}."
    )


def fingerprint_lfm2_fold_evidence(
    evidence: Lfm2ClosedLoopFoldEvidence,
) -> str:
    """Hash the actual detached evidence tensors, not a caller assertion."""

    if not isinstance(evidence, Lfm2ClosedLoopFoldEvidence):
        raise DynamicLfm2AdapterError("evidence must be Lfm2ClosedLoopFoldEvidence.")
    return canonical_sha256(
        {
            "schema": ADAPTER_SCHEMA,
            "kind": "lfm2_closed_loop_evidence_payload",
            "payload": _evidence_value_identity(evidence),
        }
    )


def fingerprint_lfm2_model_fold_evidence(evidence: Any) -> str:
    """Hash the typed whole-model evidence without trusting a cache label."""

    try:
        from .hf_lfm2_model_fold import Lfm2ModelFoldEvidence
    except ImportError as exc:  # pragma: no cover - optional during import
        raise DynamicLfm2AdapterError(
            "The LFM2 model-fold primitive is unavailable."
        ) from exc
    if not isinstance(evidence, Lfm2ModelFoldEvidence):
        raise DynamicLfm2AdapterError("evidence must be Lfm2ModelFoldEvidence.")
    return canonical_sha256(
        {
            "schema": ADAPTER_SCHEMA,
            "kind": "lfm2_hybrid_model_evidence_payload",
            "payload": _evidence_value_identity(evidence),
        }
    )


def fingerprint_lfm2_producer_first_evidence(evidence: Any) -> str:
    """Hash the sealed parent-only producer-first request."""

    try:
        from .hf_lfm2_producer_first_model_fold import Lfm2ProducerFirstEvidence
    except ImportError as exc:  # pragma: no cover - optional at import time
        raise DynamicLfm2AdapterError(
            "The producer-first LFM2 primitive is unavailable."
        ) from exc
    if not isinstance(evidence, Lfm2ProducerFirstEvidence):
        raise DynamicLfm2AdapterError(
            "evidence must be Lfm2ProducerFirstEvidence."
        )
    return evidence.fingerprint


class Lfm2SealedProducerFirstEvidence:
    """Executor-safe verified seal around rows plus an opaque capture callback.

    ``FoldEvidence`` canonically fingerprints ordinary dataclasses field by
    field.  A live capture callback and a trace-row cache object are deliberate
    registered-provider trust boundaries, so this non-dataclass wrapper exposes
    the executor's explicit ``verify``/``fingerprint`` cache-seal contract.
    """

    __slots__ = ("_evidence", "_fingerprint")

    def __init__(self, evidence: Any) -> None:
        try:
            from .hf_lfm2_producer_first_model_fold import (
                Lfm2ProducerFirstEvidence,
            )
        except ImportError as exc:  # pragma: no cover - optional at import time
            raise DynamicLfm2AdapterError(
                "The producer-first LFM2 primitive is unavailable."
            ) from exc
        if not isinstance(evidence, Lfm2ProducerFirstEvidence):
            raise DynamicLfm2AdapterError(
                "sealed producer-first payload has the wrong evidence type."
            )
        evidence.verify()
        self._evidence = evidence
        self._fingerprint = fingerprint_lfm2_producer_first_evidence(evidence)

    @property
    def evidence(self) -> Any:
        return self._evidence

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def verify(self) -> "Lfm2SealedProducerFirstEvidence":
        self._evidence.verify()
        observed = fingerprint_lfm2_producer_first_evidence(self._evidence)
        if observed != self._fingerprint:
            raise DynamicLfm2AdapterError(
                "Sealed producer-first evidence changed after binding."
            )
        return self


def _default_cache_integrity(payload: Any) -> str:
    verify = getattr(payload, "verify", None)
    cache_key = getattr(payload, "cache_key", None)
    if callable(verify) and cache_key is not None:
        verify()
        return _digest(str(cache_key), "verified parent cache key")
    if isinstance(payload, Mapping):
        try:
            return canonical_sha256(dict(payload))
        except (TypeError, ValueError) as exc:
            raise DynamicLfm2AdapterError(
                "Mapping cache payload is not canonical JSON; provide "
                "parent_cache_integrity_check."
            ) from exc
    raise DynamicLfm2AdapterError(
        "Parent cache payload must expose verify()/cache_key or provide "
        "parent_cache_integrity_check."
    )


def _target_layer_index(action: FoldAction, graph: ModelCircuitGraph) -> int:
    indices = {graph.node(node_id).layer_index for node_id in action.target_node_ids}
    if None in indices or len(indices) != 1:
        raise DynamicLfm2AdapterError(
            "An LFM2 layer action must own exactly one complete target layer."
        )
    return int(next(iter(indices)))


class Lfm2CachedEvidenceProvider:
    """On-policy child evidence backed by one immutable parent trace cache."""

    def __init__(
        self,
        *,
        parent_cache_fingerprint: str,
        parent_cache_payload: Any,
        builder: Lfm2EvidenceBuilder,
        selected_target: Lfm2SelectedTargetLike,
        model_builder: Lfm2ModelEvidenceBuilder | None = None,
        producer_first_builder: Lfm2ProducerFirstEvidenceBuilder | None = None,
        blank_state_fingerprint: str | None = None,
        parent_cache_integrity_check: Callable[[Any], str] | None = None,
    ) -> None:
        self.parent_cache_fingerprint = _digest(
            parent_cache_fingerprint, "parent_cache_fingerprint"
        )
        if not callable(builder):
            raise DynamicLfm2AdapterError("evidence builder must be callable.")
        if not isinstance(
            selected_target,
            (Lfm2SelectedTarget, Lfm2HybridSelectedTarget),
        ):
            raise DynamicLfm2AdapterError(
                "selected_target has an unsupported LFM2 target type."
            )
        if model_builder is not None and not callable(model_builder):
            raise DynamicLfm2AdapterError(
                "model_builder must be callable or None."
            )
        if producer_first_builder is not None and not callable(
            producer_first_builder
        ):
            raise DynamicLfm2AdapterError(
                "producer_first_builder must be callable or None."
            )
        self.parent_cache_payload = parent_cache_payload
        self.builder = builder
        self.selected_target = selected_target
        self.model_builder = model_builder
        self.producer_first_builder = producer_first_builder
        self.blank_state_fingerprint = (
            None
            if blank_state_fingerprint is None
            else _digest(blank_state_fingerprint, "blank_state_fingerprint")
        )
        if (
            (
                isinstance(selected_target, Lfm2HybridSelectedTarget)
                or producer_first_builder is not None
            )
            and self.blank_state_fingerprint is None
        ):
            raise DynamicLfm2AdapterError(
                "Hybrid/producer-first evidence requires the compiler-"
                "authenticated blank state fingerprint."
            )
        self.parent_cache_integrity_check = (
            _default_cache_integrity
            if parent_cache_integrity_check is None
            else parent_cache_integrity_check
        )
        if not callable(self.parent_cache_integrity_check):
            raise DynamicLfm2AdapterError(
                "parent_cache_integrity_check must be callable."
            )
        self._verify_parent_cache()
        self._cache: dict[
            tuple[str, str],
            tuple[
                Lfm2PreparedEvidence
                | Lfm2PreparedModelEvidence
                | Lfm2PreparedHybridProducerEvidence
                | Lfm2PreparedProducerFirstEvidence,
                str,
            ],
        ] = {}
        self._parent_only_cache: dict[
            str,
            tuple[
                Lfm2PreparedHybridProducerEvidence
                | Lfm2PreparedProducerFirstEvidence,
                str,
            ],
        ] = {}
        self._model_level_action_methods: dict[str, str] = {}
        self.prepare_count = 0
        self.build_count = 0
        self.refresh_count = 0
        self.restore_count = 0

    def _verify_parent_cache(self) -> None:
        observed = _digest(
            self.parent_cache_integrity_check(self.parent_cache_payload),
            "parent cache integrity result",
        )
        if observed != self.parent_cache_fingerprint:
            raise DynamicLfm2AdapterError(
                "Parent trace cache integrity differs from its bound fingerprint."
            )

    @property
    def base_fingerprint(self) -> str:
        return self.parent_cache_fingerprint

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        self.prepare_count += 1
        self._verify_parent_cache()
        if action.method_id not in {
            LAYER_METHOD,
            HYBRID_MODEL_METHOD,
            PRODUCER_FIRST_MODEL_METHOD,
        }:
            return FoldEvidence(
                self.parent_cache_fingerprint,
                payload=None,
                channels=action.evidence_requirements,
                parent_forward_count=0,
                child_forward_count=0,
            )
        state = context.child_state_fingerprint
        key = (action.action_id, state)
        cached = self._cache.get(key)
        if (
            cached is None
            and action.method_id
            in {HYBRID_MODEL_METHOD, PRODUCER_FIRST_MODEL_METHOD}
            and state != self.blank_state_fingerprint
        ):
            cached = self._parent_only_cache.get(action.action_id)
            if cached is None:
                raise DynamicLfm2AdapterError(
                    "Final parent-only evidence was requested before the "
                    "compiler-zero evidence binding was authenticated."
                )
            self._cache[key] = cached
        if cached is None:
            self._verify_parent_cache()
            if action.method_id == LAYER_METHOD:
                layer_index = _target_layer_index(action, context.child_graph)
                prepared = self.builder(
                    Lfm2EvidenceBuildRequest(
                        action=action,
                        layer_index=layer_index,
                        child_model=context.child_model,
                        child_graph=context.child_graph,
                        child_state_fingerprint=state,
                        parent_cache_fingerprint=self.parent_cache_fingerprint,
                        parent_cache_payload=self.parent_cache_payload,
                    )
                )
                if not isinstance(prepared, Lfm2PreparedEvidence):
                    raise DynamicLfm2AdapterError(
                        "evidence builder must return Lfm2PreparedEvidence."
                    )
                payload_fingerprint = fingerprint_lfm2_fold_evidence(
                    prepared.evidence
                )
                evidence_kind = "bound_lfm2_layer_evidence"
                binding: dict[str, Any] = {"layer_index": layer_index}
            elif action.method_id == HYBRID_MODEL_METHOD:
                if not isinstance(
                    self.selected_target,
                    Lfm2HybridSelectedTarget,
                ):
                    raise DynamicLfm2AdapterError(
                        "A hybrid action is bound to a full-depth target."
                    )
                if self.model_builder is None:
                    raise DynamicLfm2AdapterError(
                        "A hybrid LFM2 target requires model_evidence_builder."
                    )
                if self.blank_state_fingerprint is None or state != (
                    self.blank_state_fingerprint
                ):
                    raise DynamicLfm2AdapterError(
                        "Hybrid parent-only evidence must be prepared exactly "
                        "once from the authenticated compiler-zero state."
                    )
                prepared = self.model_builder(
                    Lfm2ModelEvidenceBuildRequest(
                        action=action,
                        child_model=context.child_model,
                        child_graph=context.child_graph,
                        child_state_fingerprint=state,
                        selected_target=self.selected_target,
                        parent_cache_fingerprint=self.parent_cache_fingerprint,
                        parent_cache_payload=self.parent_cache_payload,
                    )
                )
                if not isinstance(
                    prepared,
                    Lfm2PreparedHybridProducerEvidence,
                ):
                    raise DynamicLfm2AdapterError(
                        "hybrid model evidence builder must return "
                        "Lfm2PreparedHybridProducerEvidence."
                    )
                payload_fingerprint = fingerprint_lfm2_producer_first_evidence(
                    prepared.evidence
                )
                evidence_kind = "bound_lfm2_hybrid_parent_only_evidence"
                binding = {
                    "ordered_source_spans": encode_lfm2_source_spans(
                        self.selected_target.ordered_source_spans
                    ),
                    "target_layer_types": encode_lfm2_layer_types(
                        self.selected_target.target_layer_types
                    ),
                    "blank_state_fingerprint": state,
                    "blank_child_features_used": False,
                    "capture_builder_id": (
                        prepared.evidence.capture_builder_id
                    ),
                }
            else:
                if not isinstance(self.selected_target, Lfm2SelectedTarget):
                    raise DynamicLfm2AdapterError(
                        "Producer-first execution requires a full-depth target."
                    )
                if self.producer_first_builder is None:
                    raise DynamicLfm2AdapterError(
                        "Producer-first execution requires its parent-only "
                        "evidence builder."
                    )
                if self.blank_state_fingerprint is None or state != (
                    self.blank_state_fingerprint
                ):
                    raise DynamicLfm2AdapterError(
                        "Producer-first evidence must be prepared exactly once "
                        "from the authenticated compiler-zero state."
                    )
                prepared = self.producer_first_builder(
                    Lfm2ProducerFirstEvidenceBuildRequest(
                        action=action,
                        child_model=context.child_model,
                        child_graph=context.child_graph,
                        blank_state_fingerprint=state,
                        selected_target=self.selected_target,
                        parent_cache_fingerprint=self.parent_cache_fingerprint,
                        parent_cache_payload=self.parent_cache_payload,
                    )
                )
                if not isinstance(prepared, Lfm2PreparedProducerFirstEvidence):
                    raise DynamicLfm2AdapterError(
                        "producer-first builder must return "
                        "Lfm2PreparedProducerFirstEvidence."
                    )
                payload_fingerprint = fingerprint_lfm2_producer_first_evidence(
                    prepared.evidence
                )
                evidence_kind = "bound_lfm2_producer_first_model_evidence"
                binding = {
                    "blank_state_fingerprint": state,
                    "blank_child_features_used": False,
                    "capture_builder_id": (
                        prepared.evidence.capture_builder_id
                    ),
                }
            if context.child_state_fingerprint != state:
                raise DynamicLfm2AdapterError(
                    "LFM2 evidence builder mutated the child model."
                )
            self._verify_parent_cache()
            bound_fingerprint = canonical_sha256(
                {
                    "schema": ADAPTER_SCHEMA,
                    "kind": evidence_kind,
                    "parent_cache_fingerprint": self.parent_cache_fingerprint,
                    "action_sha256": action.action_sha256,
                    "child_state_fingerprint": state,
                    "payload_fingerprint": payload_fingerprint,
                    **binding,
                }
            )
            cached = (prepared, bound_fingerprint)
            self._cache[key] = cached
            if isinstance(
                prepared,
                (
                    Lfm2PreparedHybridProducerEvidence,
                    Lfm2PreparedProducerFirstEvidence,
                ),
            ):
                self._parent_only_cache[action.action_id] = (
                    prepared,
                    bound_fingerprint,
                )
                known_method = self._model_level_action_methods.get(action.action_id)
                if known_method is not None and known_method != action.method_id:
                    raise DynamicLfm2AdapterError(
                        "A model-level evidence action ID changed fold methods."
                    )
                self._model_level_action_methods[action.action_id] = action.method_id
            self.build_count += 1
        prepared, bound_fingerprint = cached
        payload: Any = prepared.evidence
        if isinstance(
            prepared,
            (
                Lfm2PreparedHybridProducerEvidence,
                Lfm2PreparedProducerFirstEvidence,
            ),
        ):
            payload = Lfm2SealedProducerFirstEvidence(payload)
        return FoldEvidence(
            bound_fingerprint,
            payload=payload,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=prepared.child_forward_count,
            root_fingerprint=self.parent_cache_fingerprint,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        # The cache key includes the exact child state, so the next dependent
        # action automatically builds fresh child-side evidence while retaining
        # the same immutable parent trace payload.
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        initial_state_fingerprint: str,
        accepted_action_ids: tuple[str, ...],
    ) -> None:
        expected = _digest(
            initial_state_fingerprint,
            "initial_state_fingerprint",
        )
        if context.child_state_fingerprint != expected:
            raise DynamicLfm2AdapterError(
                "Evidence restore observed a child that was not globally restored."
            )
        self._verify_parent_cache()
        marker_name = "_neural_forge_producer_plan_fingerprint"
        marker_present = hasattr(context.child_model, marker_name)
        selected_model_method = (
            HYBRID_MODEL_METHOD
            if isinstance(self.selected_target, Lfm2HybridSelectedTarget)
            else (
                PRODUCER_FIRST_MODEL_METHOD
                if self.producer_first_builder is not None
                else None
            )
        )
        if marker_present and selected_model_method is None:
            raise DynamicLfm2AdapterError(
                "Producer-plan rollback metadata is not owned by the "
                "selected LFM2 layer-fold method."
            )
        if selected_model_method is not None and (
            marker_present or accepted_action_ids
        ):
            if (
                self.blank_state_fingerprint is None
                or expected != self.blank_state_fingerprint
            ):
                raise DynamicLfm2AdapterError(
                    "Model-level evidence restore is not bound to the "
                    "authenticated compiler-zero blank."
                )
            unknown_action_ids = tuple(
                action_id
                for action_id in accepted_action_ids
                if (
                    action_id not in self._parent_only_cache
                    or self._model_level_action_methods.get(action_id)
                    != selected_model_method
                )
            )
            if unknown_action_ids:
                raise DynamicLfm2AdapterError(
                    "Model-level evidence restore received provisional action "
                    "IDs that were not authenticated by its parent-only cache: "
                    + ", ".join(sorted(unknown_action_ids))
                )
            if accepted_action_ids and not marker_present:
                raise DynamicLfm2AdapterError(
                    "An authenticated model-level provisional action lost its "
                    "owned producer-plan marker before global restore."
                )
            if marker_present:
                delattr(context.child_model, marker_name)
                if hasattr(context.child_model, marker_name):
                    raise DynamicLfm2AdapterError(
                        "Model-level evidence restore could not remove its "
                        "owned producer-plan marker."
                    )
        # All child-side evidence is state-bound.  Clearing it prevents a
        # later retry from retaining entries built along the rejected rollout.
        self._cache.clear()
        self._parent_only_cache.clear()
        self._model_level_action_methods.clear()
        self.restore_count += 1


def _decoder_layers(model: Any) -> tuple[Any, ...]:
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", None)
    try:
        result = tuple(layers)
    except TypeError as exc:
        raise DynamicLfm2AdapterError(
            "Could not resolve the LFM2 decoder layer container."
        ) from exc
    if not result:
        raise DynamicLfm2AdapterError("The LFM2 decoder has no layers.")
    return result


def _module_path(model: Any, target: Any) -> str:
    for path, module in model.named_modules():
        if module is target:
            if not path:
                break
            return path
    raise DynamicLfm2AdapterError("Could not bind an LFM2 layer to its model path.")


def _layer_parameter_paths(model: Any, layer: Any) -> tuple[str, ...]:
    prefix = _module_path(model, layer)
    return tuple(
        sorted(
            f"{prefix}.{name}" if name else prefix
            for name, _parameter in layer.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
    )


class Lfm2DynamicLayerTransaction:
    """Whole-model fingerprint wrapper around the proven layer transaction."""

    method_id = LAYER_METHOD

    def __init__(
        self,
        context: FoldExecutionContext,
        transaction: Lfm2ClosedLoopFoldTransaction,
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
    ) -> None:
        self.context = context
        self.transaction = transaction
        self.plan = transaction.plan
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = candidate_state_fingerprint

    @property
    def active(self) -> bool:
        return self.transaction.active

    def commit(self) -> Mapping[str, Any]:
        report = self.transaction.commit().as_dict()
        final = self.context.child_state_fingerprint
        if final != self.candidate_state_fingerprint:
            raise DynamicFoldExecutionError(
                "Committed LFM2 layer changed outside its candidate state."
            )
        return {
            **report,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": final,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
        }

    def rollback(self) -> Mapping[str, Any]:
        report = self.transaction.rollback().as_dict()
        final = self.context.child_state_fingerprint
        if final != self.incumbent_state_fingerprint:
            raise DynamicFoldExecutionError(
                "LFM2 adapter rollback did not restore the whole child model."
            )
        return {
            **report,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": final,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
        }


class Lfm2ClosedLoopLayerOperator:
    """Generic FoldOperator backed by the existing closed-loop LFM2 algorithm."""

    method_id = LAYER_METHOD

    def __init__(
        self,
        config: Lfm2ClosedLoopFoldConfig | None = None,
    ) -> None:
        self.config = config or Lfm2ClosedLoopFoldConfig()
        if not isinstance(self.config, Lfm2ClosedLoopFoldConfig):
            raise DynamicLfm2AdapterError(
                "config must be Lfm2ClosedLoopFoldConfig or None."
            )

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        if action.method_id != self.method_id:
            raise DynamicLfm2AdapterError("LFM2 operator received a different method.")
        if len(action.sources) != 1:
            raise DynamicLfm2AdapterError(
                "The current LFM2 layer operator accepts one parent."
            )
        if action.compression_delta <= 0:
            raise DynamicLfm2AdapterError(
                "LFM2 closed-loop execution requires a real layer compaction."
            )
        if not isinstance(evidence.payload, Lfm2ClosedLoopFoldEvidence):
            raise DynamicLfm2AdapterError(
                "LFM2 layer evidence payload has the wrong type."
            )
        source_ref = action.sources[0]
        parent_graph = context.parent_graphs[source_ref.parent_name]
        source_indices = {
            parent_graph.node(node_id).layer_index for node_id in source_ref.node_ids
        }
        if None in source_indices or len(source_indices) != 1:
            raise DynamicLfm2AdapterError(
                "An LFM2 source reference must own exactly one complete layer."
            )
        source_index = int(next(iter(source_indices)))
        target_index = _target_layer_index(action, context.child_graph)
        parent_model = context.parent_models[source_ref.parent_name]
        source_layers = _decoder_layers(parent_model)
        target_layers = _decoder_layers(context.child_model)
        if source_index >= len(source_layers) or target_index >= len(target_layers):
            raise DynamicLfm2AdapterError("LFM2 layer index is out of range.")
        source_layer = source_layers[source_index]
        target_layer = target_layers[target_index]
        expected_source_paths = _layer_parameter_paths(parent_model, source_layer)
        declared_source_paths = tuple(
            sorted(
                parameter.path
                for node_id in source_ref.node_ids
                for parameter in parent_graph.node(node_id).parameters
            )
        )
        if expected_source_paths != declared_source_paths:
            raise DynamicLfm2AdapterError(
                "LFM2 source action does not own every source layer parameter "
                "exactly once."
            )
        expected_target_paths = _layer_parameter_paths(
            context.child_model, target_layer
        )
        if expected_target_paths != tuple(sorted(action.target_parameter_paths)):
            raise DynamicLfm2AdapterError(
                "LFM2 action does not own every target layer parameter exactly once."
            )
        incumbent = context.child_state_fingerprint
        plan = build_lfm2_closed_loop_fold_plan(
            source_layer,
            target_layer,
            evidence.payload,
            config=self.config,
        )
        transaction = install_lfm2_closed_loop_fold_plan(target_layer, plan)
        try:
            candidate = context.child_state_fingerprint
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise
        return Lfm2DynamicLayerTransaction(
            context,
            transaction,
            incumbent,
            candidate,
        )


def _transaction_report_dict(value: Any) -> dict[str, Any]:
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        report = as_dict()
    elif isinstance(value, Mapping):
        report = dict(value)
    else:
        raise DynamicLfm2AdapterError(
            "LFM2 model transaction did not return a report mapping."
        )
    if not isinstance(report, Mapping):
        raise DynamicLfm2AdapterError(
            "LFM2 model transaction report serialization is not a mapping."
        )
    return dict(report)


class Lfm2DynamicModelTransaction:
    """Whole-state authentication wrapper around the generic model primitive."""

    def __init__(
        self,
        context: FoldExecutionContext,
        transaction: Any,
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
        *,
        method_id: str = HYBRID_MODEL_METHOD,
    ) -> None:
        if method_id not in {HYBRID_MODEL_METHOD, PRODUCER_FIRST_MODEL_METHOD}:
            raise DynamicLfm2AdapterError(
                "Unsupported LFM2 model transaction method."
            )
        self.method_id = method_id
        self.context = context
        self.transaction = transaction
        self.plan = transaction.plan
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = candidate_state_fingerprint

    @property
    def active(self) -> bool:
        return bool(self.transaction.active)

    def commit(self) -> Mapping[str, Any]:
        report = _transaction_report_dict(self.transaction.commit())
        final = self.context.child_state_fingerprint
        if final != self.candidate_state_fingerprint:
            raise DynamicFoldExecutionError(
                "Committed LFM2 model fold changed outside its candidate state."
            )
        return {
            **report,
            "model_level_action": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": final,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
        }

    def rollback(self) -> Mapping[str, Any]:
        report = _transaction_report_dict(self.transaction.rollback())
        final = self.context.child_state_fingerprint
        if final != self.incumbent_state_fingerprint:
            raise DynamicFoldExecutionError(
                "LFM2 model-fold rollback did not restore the whole child model."
            )
        return {
            **report,
            "model_level_action": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": final,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
        }


class Lfm2HybridModelOperator:
    """One generic whole-model depth+attention+FFN folding transaction."""

    method_id = HYBRID_MODEL_METHOD

    def __init__(
        self,
        selected_target: Lfm2HybridSelectedTarget,
        config: Any | None = None,
    ) -> None:
        if not isinstance(selected_target, Lfm2HybridSelectedTarget):
            raise DynamicLfm2AdapterError(
                "Hybrid model operator requires Lfm2HybridSelectedTarget."
            )
        self.selected_target = selected_target
        self.config = config

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        try:
            from .hf_lfm2_hybrid_producer_first_model_fold import (
                install_lfm2_hybrid_producer_first_model_fold,
            )
            from .hf_lfm2_model_fold import (
                Lfm2ModelFoldSpan,
            )
            from .hf_lfm2_producer_first_model_fold import (
                Lfm2ProducerFirstEvidence,
            )
        except ImportError as exc:  # pragma: no cover - optional during import
            raise DynamicLfm2AdapterError(
                "The generic LFM2 model-fold primitive is unavailable."
            ) from exc
        if action.method_id != self.method_id:
            raise DynamicLfm2AdapterError(
                "Hybrid LFM2 operator received a different method."
            )
        if len(action.sources) != 1 or action.compression_delta <= 0:
            raise DynamicLfm2AdapterError(
                "Hybrid LFM2 execution requires one parent and real compaction."
            )
        if not isinstance(
            evidence.payload,
            Lfm2SealedProducerFirstEvidence,
        ):
            raise DynamicLfm2AdapterError(
                "Hybrid LFM2 parent-only evidence payload has the wrong type."
            )
        evidence.payload.verify()
        producer_evidence = evidence.payload.evidence
        if not isinstance(producer_evidence, Lfm2ProducerFirstEvidence):
            raise DynamicLfm2AdapterError(
                "Hybrid LFM2 sealed payload lost its typed evidence."
            )
        expected_configuration = {
            "ordered_source_spans": encode_lfm2_source_spans(
                self.selected_target.ordered_source_spans
            ),
            "target_layer_types": encode_lfm2_layer_types(
                self.selected_target.target_layer_types
            ),
            "target_depth": self.selected_target.target_depth,
        }
        action_configuration = dict(action.configuration)
        if any(
            action_configuration.get(name) != expected
            for name, expected in expected_configuration.items()
        ):
            raise DynamicLfm2AdapterError(
                "Hybrid LFM2 action topology differs from its selected target."
            )
        source_ref = action.sources[0]
        parent_graph = context.parent_graphs[source_ref.parent_name]
        if set(source_ref.node_ids) != {
            node.node_id for node in parent_graph.nodes
        } or set(action.target_node_ids) != {
            node.node_id for node in context.child_graph.nodes
        }:
            raise DynamicLfm2AdapterError(
                "Hybrid LFM2 action must own every source and target circuit."
            )
        expected_target_paths = {
            name
            for name, _parameter in context.child_model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        }
        if set(action.target_parameter_paths) != expected_target_paths:
            raise DynamicLfm2AdapterError(
                "Hybrid LFM2 action does not own every target parameter."
            )
        parent_model = context.parent_models[source_ref.parent_name]
        spans = tuple(
            Lfm2ModelFoldSpan(
                source_layer_indices=span,
                target_layer_index=target_index,
            )
            for target_index, span in enumerate(
                self.selected_target.ordered_source_spans
            )
        )
        incumbent = context.child_state_fingerprint
        transaction = install_lfm2_hybrid_producer_first_model_fold(
            parent_model,
            context.child_model,
            spans,
            producer_evidence,
            config=self.config,
        )
        try:
            candidate = context.child_state_fingerprint
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise
        if candidate == incumbent:
            if transaction.active:
                transaction.rollback()
            raise DynamicLfm2AdapterError(
                "Hybrid model primitive left the blank child unchanged."
            )
        return Lfm2DynamicModelTransaction(
            context,
            transaction,
            incumbent,
            candidate,
            method_id=HYBRID_MODEL_METHOD,
        )


class Lfm2ProducerFirstModelOperator:
    """One full-depth producer-first residual-attention transaction."""

    method_id = PRODUCER_FIRST_MODEL_METHOD

    def __init__(self, config: Any | None = None) -> None:
        try:
            from .hf_lfm2_producer_first_model_fold import (
                Lfm2ProducerFirstFoldConfig,
            )
        except ImportError as exc:  # pragma: no cover - optional at import time
            raise DynamicLfm2AdapterError(
                "The producer-first LFM2 primitive is unavailable."
            ) from exc
        self.config = config or Lfm2ProducerFirstFoldConfig()
        if not isinstance(self.config, Lfm2ProducerFirstFoldConfig):
            raise DynamicLfm2AdapterError(
                "producer-first config has the wrong type."
            )

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        from .hf_lfm2_producer_first_model_fold import (
            Lfm2ProducerFirstEvidence,
            install_lfm2_producer_first_model_fold,
        )

        if action.method_id != self.method_id:
            raise DynamicLfm2AdapterError(
                "Producer-first operator received a different method."
            )
        if len(action.sources) != 1 or action.compression_delta <= 0:
            raise DynamicLfm2AdapterError(
                "Producer-first execution requires one parent and real "
                "full-depth compaction."
            )
        if not isinstance(
            evidence.payload,
            Lfm2SealedProducerFirstEvidence,
        ):
            raise DynamicLfm2AdapterError(
                "Producer-first model evidence payload has the wrong type."
            )
        evidence.payload.verify()
        producer_evidence = evidence.payload.evidence
        if not isinstance(producer_evidence, Lfm2ProducerFirstEvidence):
            raise DynamicLfm2AdapterError(
                "Producer-first sealed payload lost its typed evidence."
            )
        source_ref = action.sources[0]
        parent_graph = context.parent_graphs[source_ref.parent_name]
        if set(source_ref.node_ids) != {
            node.node_id for node in parent_graph.nodes
        } or set(action.target_node_ids) != {
            node.node_id for node in context.child_graph.nodes
        }:
            raise DynamicLfm2AdapterError(
                "Producer-first action must own every source and target circuit."
            )
        if len(parent_graph.layer_indices) != len(context.child_graph.layer_indices):
            raise DynamicLfm2AdapterError(
                "Producer-first folding explicitly rejects depth spans."
            )
        expected_target_paths = {
            name
            for name, _parameter in context.child_model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        }
        if set(action.target_parameter_paths) != expected_target_paths:
            raise DynamicLfm2AdapterError(
                "Producer-first action does not own every target parameter."
            )
        incumbent = context.child_state_fingerprint
        transaction = install_lfm2_producer_first_model_fold(
            context.parent_models[source_ref.parent_name],
            context.child_model,
            producer_evidence,
            config=self.config,
        )
        try:
            candidate = context.child_state_fingerprint
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise
        if candidate == incumbent:
            if transaction.active:
                transaction.rollback()
            raise DynamicLfm2AdapterError(
                "Producer-first model primitive left the blank unchanged."
            )
        return Lfm2DynamicModelTransaction(
            context,
            transaction,
            incumbent,
            candidate,
            method_id=PRODUCER_FIRST_MODEL_METHOD,
        )


def _layer_operator_sequence(graph: ModelCircuitGraph) -> tuple[str, ...]:
    result = []
    for layer_index in graph.layer_indices:
        operators = {
            node.kind
            for node in graph.layer_nodes(layer_index)
            if node.kind
            in {
                CircuitKind.ATTENTION.value,
                CircuitKind.SHORT_CONV.value,
            }
        }
        if len(operators) != 1:
            raise DynamicLfm2AdapterError(
                "Each LFM2 layer must expose exactly one canonical operator."
            )
        result.append(next(iter(operators)))
    return tuple(result)


class Lfm2FoldOnlyInitializer:
    """Non-mutating bootstrap proof for cached, zero-gradient execution."""

    def __init__(
        self,
        parent_cache_fingerprint: str,
        selected_target: Lfm2SelectedTargetLike,
    ) -> None:
        self.parent_cache_fingerprint = _digest(
            parent_cache_fingerprint, "parent_cache_fingerprint"
        )
        if not isinstance(
            selected_target,
            (Lfm2SelectedTarget, Lfm2HybridSelectedTarget),
        ):
            raise DynamicLfm2AdapterError(
                "selected_target has an unsupported LFM2 target type."
            )
        self.selected_target = selected_target

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        if isinstance(self.selected_target, Lfm2SelectedTarget):
            self.selected_target.validate()
        unsupported = sorted(
            {
                action.method_id
                for action in plan.actions
                if action.method_id
                not in {
                    "copy_exact",
                    LAYER_METHOD,
                    HYBRID_MODEL_METHOD,
                    PRODUCER_FIRST_MODEL_METHOD,
                }
            }
        )
        if unsupported:
            raise DynamicLfm2AdapterError(
                "LFM2 bridge cannot initialize methods: " + ", ".join(unsupported)
            )
        if any(not action.signature.zero_gradient for action in plan.actions):
            raise DynamicLfm2AdapterError(
                "An LFM2 bridge action is not declared zero-gradient."
            )
        if plan.evidence_fingerprint != self.parent_cache_fingerprint:
            raise DynamicLfm2AdapterError(
                "The plan is not bound to this immutable parent trace cache."
            )
        if (
            self.selected_target.signature.evidence_fingerprint
            != self.parent_cache_fingerprint
        ):
            raise DynamicLfm2AdapterError(
                "Selected target is not bound to this parent trace cache."
            )
        if (
            plan.exact_target_unique_parameters
            != self.selected_target.exact_target_unique_parameters
        ):
            raise DynamicLfm2AdapterError(
                "Concrete compiler budget differs from the selected LFM2 stage."
            )
        if len(context.parent_graphs) != 1:
            raise DynamicLfm2AdapterError(
                "The current LFM2 bridge accepts exactly one parent graph."
            )
        parent_graph = next(iter(context.parent_graphs.values()))
        parent_state = dict(plan.parent_state_fingerprints)[parent_graph.model_name]
        if isinstance(self.selected_target, Lfm2SelectedTarget):
            _validate_selected_target_against_parent(
                self.selected_target,
                parent_graph,
                parent_state,
            )
        else:
            _validate_hybrid_target_against_parent(
                self.selected_target,
                context.parent_models[parent_graph.model_name],
                parent_graph,
                parent_state,
            )
        if (parent_graph.fingerprint,) != (
            self.selected_target.signature.parent_graph_fingerprints
        ):
            raise DynamicLfm2AdapterError(
                "Execution parent graph differs from target selection."
            )
        if tuple(plan.parent_state_fingerprints) != tuple(
            self.selected_target.signature.parent_state_fingerprints
        ):
            raise DynamicLfm2AdapterError(
                "Execution parent state differs from target selection."
            )
        if isinstance(self.selected_target, Lfm2SelectedTarget):
            validate_lfm2_selected_child(
                self.selected_target,
                context.child_model,
                context.child_graph,
            )
            if _layer_operator_sequence(parent_graph) != _layer_operator_sequence(
                context.child_graph
            ):
                raise DynamicLfm2AdapterError(
                    "Parent and selected child hybrid layer ordering differ."
                )
            producer_actions = tuple(
                action
                for action in plan.actions
                if action.method_id == PRODUCER_FIRST_MODEL_METHOD
            )
            if producer_actions:
                if (
                    len(plan.actions) != 1
                    or len(producer_actions) != 1
                    or producer_actions[0].compression_delta <= 0
                    or set(producer_actions[0].target_node_ids)
                    != {node.node_id for node in context.child_graph.nodes}
                    or len(parent_graph.layer_indices)
                    != len(context.child_graph.layer_indices)
                ):
                    raise DynamicLfm2AdapterError(
                        "Producer-first LFM2 must compile to one positive-"
                        "compaction full-depth whole-model action."
                    )
                complete_target_coverage = True
            else:
                lfm_actions = tuple(
                    action
                    for action in plan.actions
                    if action.method_id == LAYER_METHOD
                )
                if not lfm_actions or any(
                    action.compression_delta <= 0 for action in lfm_actions
                ):
                    raise DynamicLfm2AdapterError(
                        "Selected LFM2 plan does not contain real layer compaction."
                    )
                complete_target_coverage = False
        else:
            validate_lfm2_hybrid_child(
                self.selected_target,
                context.child_model,
                context.child_graph,
            )
            hybrid_actions = tuple(
                action
                for action in plan.actions
                if action.method_id == HYBRID_MODEL_METHOD
            )
            if (
                len(plan.actions) != 1
                or len(hybrid_actions) != 1
                or hybrid_actions[0].compression_delta <= 0
                or set(hybrid_actions[0].target_node_ids)
                != {node.node_id for node in context.child_graph.nodes}
            ):
                raise DynamicLfm2AdapterError(
                    "Hybrid LFM2 target must compile to one positive-compaction "
                    "whole-model action."
                )
            complete_target_coverage = True
        before = context.child_state_fingerprint
        after = context.child_state_fingerprint
        if before != after:
            raise DynamicLfm2AdapterError("LFM2 initializer mutated the child.")
        return {
            "kind": "lfm2_fold_only_cached_parent",
            "child_state_fingerprint": after,
            "complete_target_coverage": complete_target_coverage,
            "parent_cache_fingerprint": self.parent_cache_fingerprint,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


class Lfm2FoldAdmissionEvaluator:
    """Local invariant gate; this is not a whole-output coherence judgment."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        if context.action.method_id == "copy_exact":
            return FoldDecision.accept(
                "Exact global interface copy satisfies its algebraic contract.",
                {"local_gate_only": True},
            )
        transaction = context.transaction
        if isinstance(transaction, Lfm2DynamicModelTransaction):
            plan = transaction.plan
            plan_reporter = getattr(plan, "as_dict", None)
            report = plan_reporter() if callable(plan_reporter) else {}
            zero_gradient = bool(
                report.get(
                    "zero_gradient",
                    getattr(plan, "zero_gradient", False),
                )
            )
            seed_erasure = bool(
                report.get(
                    "seed_erasure_by_construction",
                    getattr(plan, "seed_erasure_by_construction", False),
                )
            )
            metrics: dict[str, JsonScalar] = {
                "local_gate_only": True,
                "model_level_action": True,
                "zero_gradient": zero_gradient,
                "seed_erasure_by_construction": seed_erasure,
            }
            if not zero_gradient:
                return FoldDecision.reject(
                    "Hybrid LFM2 model plan is not zero-gradient.",
                    metrics,
                )
            if not seed_erasure:
                return FoldDecision.reject(
                    "Hybrid LFM2 model plan does not erase the blank child.",
                    metrics,
                )
            return FoldDecision.accept(
                "Hybrid LFM2 model transaction passed its local model-level "
                "fold invariants; whole-output behavior remains mandatory.",
                metrics,
            )
        if not isinstance(transaction, Lfm2DynamicLayerTransaction):
            return FoldDecision.reject("LFM2 action lacks the typed transaction.")
        plan = transaction.plan
        metrics: dict[str, JsonScalar] = {
            "local_gate_only": True,
            "seed_erasure_by_construction": plan.seed_erasure_by_construction,
            "short_conv_preserved_exactly": plan.short_conv_preserved_exactly,
            "mlp_endpoint_relative_rmse": (
                plan.mlp.endpoint_fit.gate.metrics.relative_rmse
            ),
            "mlp_endpoint_cosine": (
                plan.mlp.endpoint_fit.gate.metrics.mean_cosine_similarity
            ),
        }
        if plan.attention is not None:
            metrics.update(
                {
                    "attention_endpoint_relative_rmse": (
                        plan.attention.endpoint_fit.gate.metrics.relative_rmse
                    ),
                    "attention_endpoint_cosine": (
                        plan.attention.endpoint_fit.gate.metrics.mean_cosine_similarity
                    ),
                }
            )
        if not plan.zero_gradient:
            return FoldDecision.reject("LFM2 plan is not zero-gradient.", metrics)
        if not plan.seed_erasure_by_construction:
            return FoldDecision.reject(
                "LFM2 plan does not erase the blank initialization.", metrics
            )
        if not plan.mlp.endpoint_fit.accepted:
            return FoldDecision.reject("LFM2 MLP endpoint gate rejected.", metrics)
        if plan.attention is not None and not plan.attention.endpoint_fit.accepted:
            return FoldDecision.reject(
                "LFM2 attention endpoint gate rejected.", metrics
            )
        if not plan.short_conv_preserved_exactly:
            return FoldDecision.reject(
                "LFM2 short-conv exactness contract failed.", metrics
            )
        return FoldDecision.accept(
            "Closed-loop LFM2 layer passed local coupled-circuit invariants; "
            "whole-output behavior still requires the injected final gate.",
            metrics,
        )


@dataclass(frozen=True, slots=True)
class Lfm2DynamicCompilerBridge:
    """Components ready to pass directly to DynamicFoldingCompiler.compile."""

    operators: FoldOperatorRegistry
    evidence_provider: Lfm2CachedEvidenceProvider
    initializer: FoldInitializer
    evaluator: FoldEvaluator
    final_evaluator: FoldEvaluator
    selected_target: Lfm2SelectedTargetLike

    def __post_init__(self) -> None:
        if not isinstance(
            self.selected_target,
            (Lfm2SelectedTarget, Lfm2HybridSelectedTarget),
        ):
            raise DynamicLfm2AdapterError(
                "selected_target has an unsupported LFM2 target type."
            )
        if self.final_evaluator is None or not callable(
            getattr(self.final_evaluator, "evaluate", None)
        ):
            raise DynamicLfm2AdapterError(
                "A whole-output final evaluator is required by the production bridge."
            )

    @property
    def evidence_fingerprint(self) -> str:
        return self.evidence_provider.base_fingerprint

    @property
    def requires_external_semantic_final_gate(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "operator_method_ids": list(self.operators.method_ids),
            "evidence_fingerprint": self.evidence_fingerprint,
            "parent_trace_cache_reused": True,
            "selected_target": self.selected_target.as_dict(),
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "requires_external_semantic_final_gate": (
                self.requires_external_semantic_final_gate
            ),
        }


def build_lfm2_dynamic_compiler_bridge(
    *,
    parent_cache_fingerprint: str,
    parent_cache_payload: Any,
    evidence_builder: Lfm2EvidenceBuilder,
    selected_target: Lfm2SelectedTargetLike,
    final_evaluator: FoldEvaluator,
    fold_config: Lfm2ClosedLoopFoldConfig | None = None,
    model_evidence_builder: Lfm2ModelEvidenceBuilder | None = None,
    model_fold_config: Any | None = None,
    producer_first_evidence_builder: (
        Lfm2ProducerFirstEvidenceBuilder | None
    ) = None,
    producer_first_fold_config: Any | None = None,
    blank_state_fingerprint: str | None = None,
    parent_cache_integrity_check: Callable[[Any], str] | None = None,
) -> Lfm2DynamicCompilerBridge:
    """Register the selected full-depth or whole-model LFM2 operator."""

    if not isinstance(
        selected_target,
        (Lfm2SelectedTarget, Lfm2HybridSelectedTarget),
    ):
        raise DynamicLfm2AdapterError(
            "selected_target has an unsupported LFM2 target type."
        )
    if selected_target.signature.evidence_fingerprint != _digest(
        parent_cache_fingerprint, "parent_cache_fingerprint"
    ):
        raise DynamicLfm2AdapterError(
            "Selected target and parent cache fingerprints differ."
        )
    provider = Lfm2CachedEvidenceProvider(
        parent_cache_fingerprint=parent_cache_fingerprint,
        parent_cache_payload=parent_cache_payload,
        builder=evidence_builder,
        selected_target=selected_target,
        model_builder=model_evidence_builder,
        producer_first_builder=producer_first_evidence_builder,
        blank_state_fingerprint=blank_state_fingerprint,
        parent_cache_integrity_check=parent_cache_integrity_check,
    )
    if isinstance(selected_target, Lfm2HybridSelectedTarget):
        selected_operator = Lfm2HybridModelOperator(
            selected_target,
            model_fold_config,
        )
    elif producer_first_evidence_builder is not None:
        selected_operator = Lfm2ProducerFirstModelOperator(
            producer_first_fold_config
        )
    else:
        selected_operator = Lfm2ClosedLoopLayerOperator(fold_config)
    return Lfm2DynamicCompilerBridge(
        operators=FoldOperatorRegistry(
            (
                ExactCopyOperator(),
                selected_operator,
            )
        ),
        evidence_provider=provider,
        initializer=Lfm2FoldOnlyInitializer(
            parent_cache_fingerprint,
            selected_target,
        ),
        evaluator=Lfm2FoldAdmissionEvaluator(),
        final_evaluator=final_evaluator,
        selected_target=selected_target,
    )


@dataclass(frozen=True, slots=True)
class Lfm2SelectedTargetRun:
    """A generic compiler result plus its run-level target closure record."""

    compilation: DynamicFoldCompilation
    selection_record: LedgerRecord

    @property
    def complete(self) -> bool:
        return self.compilation.complete

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "compilation": self.compilation.as_dict(),
            "selection_record": self.selection_record.as_dict(),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class Lfm2MaterializedChild:
    """A locally instantiated exact-budget blank selected by the compiler."""

    model: Any
    graph: ModelCircuitGraph
    seed: int
    blank_state_fingerprint: str
    config_fingerprint: str
    exact_unique_parameters: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "seed": self.seed,
            "blank_state_fingerprint": self.blank_state_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "graph_fingerprint": self.graph.fingerprint,
            "exact_unique_parameters": self.exact_unique_parameters,
            "local_parent_class_instantiation": True,
            "hub_loader_used": False,
        }


def build_selected_lfm2_config(
    parent_model: Any,
    selected_target: Lfm2SelectedTarget,
    *,
    fold_method_id: str | None = None,
) -> Any:
    """Clone the loaded local parent config into the selected exact topology."""

    selected_target.validate()
    expected_parent_states = dict(selected_target.signature.parent_state_fingerprints)
    if len(expected_parent_states) != 1 or (
        fingerprint_model_state(parent_model) not in expected_parent_states.values()
    ):
        raise DynamicLfm2AdapterError(
            "Loaded parent checkpoint differs from target selection."
        )
    config = getattr(parent_model, "config", None)
    if config is None or getattr(config, "model_type", None) != "lfm2":
        raise DynamicLfm2AdapterError(
            "Selected LFM2 materialization requires a loaded LFM2 parent."
        )
    source = selected_target.plan.source
    observed = Lfm2Topology(
        int(_config_value(config, ("num_attention_heads", "num_heads"), "query heads")),
        int(_config_value(config, ("num_key_value_heads",), "key/value heads")),
        int(
            _config_value(
                config,
                ("intermediate_size", "block_ff_dim"),
                "intermediate_size",
            )
        ),
    )
    if observed != source:
        raise DynamicLfm2AdapterError(
            "Loaded parent config differs from the selected source topology."
        )
    result = copy.deepcopy(config)
    topology = selected_target.stage.topology
    result.num_attention_heads = topology.query_heads
    result.num_key_value_heads = topology.key_value_heads
    result.num_heads = topology.query_heads
    result.head_dim = selected_target.plan.head_dim
    result.intermediate_size = topology.intermediate_size
    result.block_ff_dim = topology.intermediate_size
    result.block_auto_adjust_ff_dim = False
    for marker in (
        "_neural_forge_fold_method_id",
        "_neural_forge_ordered_source_spans",
        "_neural_forge_target_layer_types",
    ):
        if hasattr(result, marker):
            delattr(result, marker)
    if fold_method_id is not None:
        if fold_method_id != PRODUCER_FIRST_MODEL_METHOD:
            raise DynamicLfm2AdapterError(
                "The full-depth LFM2 target received an unknown model method."
            )
        result._neural_forge_fold_method_id = PRODUCER_FIRST_MODEL_METHOD
    payload = result.to_dict()
    required = {
        "num_attention_heads": topology.query_heads,
        "num_key_value_heads": topology.key_value_heads,
        "num_heads": topology.query_heads,
        "head_dim": selected_target.plan.head_dim,
        "intermediate_size": topology.intermediate_size,
        "block_ff_dim": topology.intermediate_size,
        "block_auto_adjust_ff_dim": False,
    }
    if fold_method_id is not None:
        required["_neural_forge_fold_method_id"] = PRODUCER_FIRST_MODEL_METHOD
    if any(payload.get(name) != value for name, value in required.items()):
        raise DynamicLfm2AdapterError(
            "Serialized LFM2 target config dropped a required topology alias."
        )
    return result


def materialize_selected_lfm2_child(
    parent_model: Any,
    selected_target: Lfm2SelectedTarget,
    *,
    child_name: str,
    seed: int,
    device: Any | None = None,
    dtype: Any | None = None,
    fold_method_id: str | None = None,
) -> Lfm2MaterializedChild:
    """Instantiate a genuine local blank with the selected exact budget."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise DynamicLfm2AdapterError(
            "PyTorch is required to materialize an LFM2 child."
        ) from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DynamicLfm2AdapterError("Blank child seed must be non-negative.")
    parameters = tuple(parent_model.parameters())
    if not parameters:
        raise DynamicLfm2AdapterError("Loaded parent has no parameters.")
    parent_devices = {value.device for value in parameters}
    parent_dtypes = {value.dtype for value in parameters}
    if device is None:
        if len(parent_devices) != 1:
            raise DynamicLfm2AdapterError(
                "A sharded parent requires an explicit child device."
            )
        device = next(iter(parent_devices))
    else:
        device = torch.device(device)
    if dtype is None:
        if len(parent_dtypes) != 1:
            raise DynamicLfm2AdapterError(
                "A mixed-dtype parent requires an explicit child dtype."
            )
        dtype = next(iter(parent_dtypes))
    target_config = build_selected_lfm2_config(
        parent_model,
        selected_target,
        fold_method_id=fold_method_id,
    )
    cuda_devices: list[int] = []
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        cuda_devices = [index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        child = type(parent_model)(copy.deepcopy(target_config))
    child.to(device=device, dtype=dtype)
    child.eval()
    graph = inspect_model_circuits(
        child,
        model_name=child_name,
        tokenizer_fingerprint=selected_target.tokenizer_fingerprint,
    )
    validate_lfm2_selected_child(selected_target, child, graph)
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise DynamicLfm2AdapterError(
            "Fresh blank LFM2 child unexpectedly contains gradients."
        )
    return Lfm2MaterializedChild(
        model=child,
        graph=graph,
        seed=seed,
        blank_state_fingerprint=fingerprint_model_state(child),
        config_fingerprint=canonical_sha256(target_config.to_dict()),
        exact_unique_parameters=graph.unique_parameter_count,
    )


def build_hybrid_lfm2_config(
    parent_model: Any,
    selected_target: Lfm2HybridSelectedTarget,
) -> Any:
    """Clone a local LFM2 config into the explicit shallower hybrid topology."""

    if not isinstance(selected_target, Lfm2HybridSelectedTarget):
        raise DynamicLfm2AdapterError(
            "selected_target must be Lfm2HybridSelectedTarget."
        )
    parent_state = fingerprint_model_state(parent_model)
    if dict(selected_target.signature.parent_state_fingerprints) != {
        selected_target.parent_name: parent_state
    }:
        raise DynamicLfm2AdapterError(
            "Loaded parent checkpoint differs from hybrid target selection."
        )
    config = getattr(parent_model, "config", None)
    if config is None or getattr(config, "model_type", None) != "lfm2":
        raise DynamicLfm2AdapterError(
            "Hybrid LFM2 materialization requires a loaded local LFM2 parent."
        )
    result = copy.deepcopy(config)
    topology = selected_target.topology
    result.num_hidden_layers = selected_target.target_depth
    if hasattr(result, "num_layers"):
        result.num_layers = selected_target.target_depth
    result.layer_types = list(selected_target.target_layer_types)
    result.num_attention_heads = topology.query_heads
    result.num_key_value_heads = topology.key_value_heads
    result.num_heads = topology.query_heads
    source_hidden_size = _positive_int(
        int(_config_value(config, ("hidden_size",), "hidden_size")),
        "hidden_size",
    )
    source_query_heads = _positive_int(
        int(
            _config_value(
                config,
                ("num_attention_heads", "num_heads"),
                "query heads",
            )
        ),
        "query_heads",
    )
    source_head_dim_value = getattr(config, "head_dim", None)
    source_head_dim = (
        source_hidden_size // source_query_heads
        if source_head_dim_value is None
        else _positive_int(int(source_head_dim_value), "head_dim")
    )
    # LFM2 attention falls back to hidden_size // target_query_heads whenever
    # head_dim is absent.  That silently changed a 16Q/64D parent into a
    # 14Q/73D child and invalidated both the circuit topology and exact budget.
    # Preserve the parent's per-head coordinate width explicitly.
    result.head_dim = source_head_dim
    result.intermediate_size = topology.intermediate_size
    result.block_ff_dim = topology.intermediate_size
    result.block_auto_adjust_ff_dim = False
    result._neural_forge_fold_method_id = HYBRID_MODEL_METHOD
    result._neural_forge_ordered_source_spans = encode_lfm2_source_spans(
        selected_target.ordered_source_spans
    )
    result._neural_forge_target_layer_types = encode_lfm2_layer_types(
        selected_target.target_layer_types
    )
    payload = result.to_dict()
    required = {
        "num_hidden_layers": selected_target.target_depth,
        "layer_types": list(selected_target.target_layer_types),
        "num_attention_heads": topology.query_heads,
        "num_key_value_heads": topology.key_value_heads,
        "num_heads": topology.query_heads,
        "head_dim": source_head_dim,
        "intermediate_size": topology.intermediate_size,
        "block_ff_dim": topology.intermediate_size,
        "block_auto_adjust_ff_dim": False,
        "_neural_forge_fold_method_id": HYBRID_MODEL_METHOD,
        "_neural_forge_ordered_source_spans": encode_lfm2_source_spans(
            selected_target.ordered_source_spans
        ),
        "_neural_forge_target_layer_types": encode_lfm2_layer_types(
            selected_target.target_layer_types
        ),
    }
    if any(payload.get(name) != value for name, value in required.items()):
        raise DynamicLfm2AdapterError(
            "Serialized hybrid LFM2 config dropped a required topology field."
        )
    return result


def materialize_hybrid_lfm2_child(
    parent_model: Any,
    selected_target: Lfm2HybridSelectedTarget,
    *,
    child_name: str,
    seed: int,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Lfm2MaterializedChild:
    """Instantiate and independently inspect the explicit shallower blank."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise DynamicLfm2AdapterError(
            "PyTorch is required to materialize an LFM2 child."
        ) from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DynamicLfm2AdapterError("Blank child seed must be non-negative.")
    parameters = tuple(parent_model.parameters())
    if not parameters:
        raise DynamicLfm2AdapterError("Loaded parent has no parameters.")
    parent_devices = {value.device for value in parameters}
    parent_dtypes = {value.dtype for value in parameters}
    if device is None:
        if len(parent_devices) != 1:
            raise DynamicLfm2AdapterError(
                "A sharded parent requires an explicit child device."
            )
        device = next(iter(parent_devices))
    else:
        device = torch.device(device)
    if dtype is None:
        if len(parent_dtypes) != 1:
            raise DynamicLfm2AdapterError(
                "A mixed-dtype parent requires an explicit child dtype."
            )
        dtype = next(iter(parent_dtypes))
    target_config = build_hybrid_lfm2_config(parent_model, selected_target)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        cuda_devices = [index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        child = type(parent_model)(copy.deepcopy(target_config))
    child.to(device=device, dtype=dtype)
    child.eval()
    graph = inspect_model_circuits(
        child,
        model_name=child_name,
        tokenizer_fingerprint=selected_target.tokenizer_fingerprint,
    )
    validate_lfm2_hybrid_child(selected_target, child, graph)
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise DynamicLfm2AdapterError(
            "Fresh hybrid LFM2 child unexpectedly contains gradients."
        )
    return Lfm2MaterializedChild(
        model=child,
        graph=graph,
        seed=seed,
        blank_state_fingerprint=fingerprint_model_state(child),
        config_fingerprint=canonical_sha256(target_config.to_dict()),
        exact_unique_parameters=graph.unique_parameter_count,
    )


def _dense_budgeted_lfm2_config_markers(
    selected_target: Lfm2DenseBudgetedSelectedTarget,
) -> dict[str, str | bool]:
    """Return the complete authenticated dense-target provenance bundle."""

    markers: dict[str, str | bool] = {
        _DENSE_BUDGET_CATALOG_MARKER: selected_target.catalog_id,
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY: (
            selected_target.component_allocation_policy
        ),
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY: (
            selected_target.component_allocation_receipt_sha256
        ),
        COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY: (
            selected_target.component_allocation_topology_sha256
        ),
        "_neural_forge_dense_ordered_source_spans": encode_lfm2_source_spans(
            selected_target.ordered_source_spans
        ),
        "_neural_forge_dense_target_layer_types": encode_lfm2_layer_types(
            selected_target.layer_types
        ),
        "_neural_forge_tokenizer_bridge_required": (
            selected_target.tokenizer_bridge_required
        ),
    }
    if (
        selected_target.target_token_interface_mode
        == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
    ):
        markers[LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY] = (
            LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        )
    if selected_target.tokenizer_bridge_required:
        bridge_markers = {
            TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY: (
                selected_target.tokenizer_bridge_artifact_sha256
            ),
            TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY: (
                selected_target.tokenizer_transport_profile_sha256
            ),
            PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                selected_target.parent_tokenizer_semantics_sha256
            ),
            CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY: (
                selected_target.child_tokenizer_semantics_sha256
            ),
        }
        markers.update(
            {
                name: _digest(value, name)
                for name, value in bridge_markers.items()
            }
        )
    return markers


def _install_dense_budgeted_lfm2_config_markers(
    config: Any,
    selected_target: Lfm2DenseBudgetedSelectedTarget,
) -> dict[str, str | bool]:
    """Install the authenticated dense-target provenance bundle once."""

    for marker in (
        "_neural_forge_fold_method_id",
        "_neural_forge_ordered_source_spans",
        "_neural_forge_target_layer_types",
    ):
        if hasattr(config, marker):
            delattr(config, marker)
    markers = _dense_budgeted_lfm2_config_markers(selected_target)
    possible_markers = {
        LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
        *TOKENIZER_BRIDGE_CONFIG_KEYS,
    }
    for name in possible_markers - set(markers):
        if hasattr(config, name):
            delattr(config, name)
    for name, value in markers.items():
        setattr(config, name, value)
    return markers


def build_dense_budgeted_lfm2_config(
    parent_model: Any,
    selected_target: Lfm2DenseBudgetedSelectedTarget,
) -> Any:
    """Clone the loaded local config into a catalog-selected dense topology."""

    if not isinstance(selected_target, Lfm2DenseBudgetedSelectedTarget):
        raise DynamicLfm2AdapterError(
            "selected_target must be Lfm2DenseBudgetedSelectedTarget."
        )
    if fingerprint_model_state(parent_model) != (
        selected_target.source_state_fingerprint
    ):
        raise DynamicLfm2AdapterError(
            "Loaded parent checkpoint changed after dense target selection."
        )
    config = getattr(parent_model, "config", None)
    if config is None or getattr(config, "model_type", None) != "lfm2":
        raise DynamicLfm2AdapterError(
            "Dense budget materialization requires a loaded local LFM2 parent."
        )
    source_tied = bool(getattr(config, "tie_word_embeddings", False))
    if (
        selected_target.target_token_interface_mode
        == LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
        and selected_target.tie_word_embeddings != source_tied
    ):
        raise DynamicLfm2AdapterError(
            "Preserve-mode dense target changed the loaded source token tie."
        )
    if (
        selected_target.target_token_interface_mode
        == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        and selected_target.tie_word_embeddings
    ):
        raise DynamicLfm2AdapterError(
            "Untied dense target cannot materialize a tied token interface."
        )
    descriptor = selected_target.heterogeneous_layout_descriptor
    if descriptor is not None:
        intermediate, query_heads, key_value_heads, head_dim = (
            _heterogeneous_scaffold_geometry(descriptor)
        )
        config_class = type(config)
        try:
            result = config_class(
                architectures=["Lfm2ForCausalLM"],
                dtype="float32",
                vocab_size=descriptor.vocab_size,
                hidden_size=descriptor.hidden_size,
                intermediate_size=intermediate,
                num_hidden_layers=descriptor.layer_count,
                num_attention_heads=query_heads,
                num_key_value_heads=key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=descriptor.max_position_embeddings,
                initializer_range=float(
                    getattr(config, "initializer_range", 0.02)
                ),
                norm_eps=descriptor.norm_eps,
                use_cache=descriptor.use_cache,
                pad_token_id=descriptor.pad_token_id,
                bos_token_id=descriptor.bos_token_id,
                eos_token_id=descriptor.eos_token_id,
                tie_word_embeddings=False,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": descriptor.rope_theta,
                },
                conv_bias=False,
                conv_L_cache=descriptor.conv_kernel_size,
                block_multiple_of=int(
                    getattr(config, "block_multiple_of", 256)
                ),
                block_ffn_dim_multiplier=float(
                    getattr(config, "block_ffn_dim_multiplier", 1.0)
                ),
                block_auto_adjust_ff_dim=False,
                layer_types=[layer.layer_type for layer in descriptor.layers],
            )
        except Exception as exc:
            raise DynamicLfm2AdapterError(
                "Could not construct a local native LFM2 scaffold from the "
                "heterogeneous descriptor."
            ) from exc
        result._name_or_path = ""
        result._attn_implementation = descriptor.attention_implementation
        result.head_dim = head_dim
        dense_markers = _install_dense_budgeted_lfm2_config_markers(
            result,
            selected_target,
        )
        payload = result.to_dict()
        required = {
            "model_type": "lfm2",
            "vocab_size": descriptor.vocab_size,
            "hidden_size": descriptor.hidden_size,
            "intermediate_size": intermediate,
            "num_hidden_layers": descriptor.layer_count,
            "num_attention_heads": query_heads,
            "num_key_value_heads": key_value_heads,
            "head_dim": head_dim,
            "max_position_embeddings": descriptor.max_position_embeddings,
            "norm_eps": descriptor.norm_eps,
            "use_cache": descriptor.use_cache,
            "pad_token_id": descriptor.pad_token_id,
            "bos_token_id": descriptor.bos_token_id,
            "eos_token_id": descriptor.eos_token_id,
            "tie_word_embeddings": False,
            "conv_bias": False,
            "conv_L_cache": descriptor.conv_kernel_size,
            "block_auto_adjust_ff_dim": False,
            "layer_types": [layer.layer_type for layer in descriptor.layers],
            "_name_or_path": "",
            **dense_markers,
        }
        if (
            any(payload.get(name) != value for name, value in required.items())
            or result._attn_implementation
            != descriptor.attention_implementation
            or payload.get("rope_parameters")
            != {
                "rope_type": "default",
                "rope_theta": descriptor.rope_theta,
            }
            or payload.get("dtype") not in (None, "float32")
        ):
            raise DynamicLfm2AdapterError(
                "Serialized heterogeneous LFM2 scaffold dropped or changed a "
                "descriptor runtime pin."
            )
        return result
    result = copy.deepcopy(config)
    result.vocab_size = selected_target.vocabulary_size
    result.hidden_size = selected_target.hidden_size
    result.intermediate_size = selected_target.intermediate_size
    result.block_ff_dim = selected_target.intermediate_size
    result.num_hidden_layers = selected_target.target_depth
    if hasattr(result, "num_layers"):
        result.num_layers = selected_target.target_depth
    result.layer_types = list(selected_target.layer_types)
    result.num_attention_heads = selected_target.query_heads
    result.num_key_value_heads = selected_target.key_value_heads
    result.num_heads = selected_target.query_heads
    result.head_dim = selected_target.head_dim
    result.conv_L_cache = selected_target.fir_length
    result.block_auto_adjust_ff_dim = False
    result.tie_word_embeddings = selected_target.tie_word_embeddings
    dense_markers = _install_dense_budgeted_lfm2_config_markers(
        result,
        selected_target,
    )
    payload = result.to_dict()
    required = {
        "model_type": "lfm2",
        "vocab_size": selected_target.vocabulary_size,
        "hidden_size": selected_target.hidden_size,
        "intermediate_size": selected_target.intermediate_size,
        "block_ff_dim": selected_target.intermediate_size,
        "num_hidden_layers": selected_target.target_depth,
        "layer_types": list(selected_target.layer_types),
        "num_attention_heads": selected_target.query_heads,
        "num_key_value_heads": selected_target.key_value_heads,
        "num_heads": selected_target.query_heads,
        "head_dim": selected_target.head_dim,
        "conv_L_cache": selected_target.fir_length,
        "block_auto_adjust_ff_dim": False,
        "tie_word_embeddings": selected_target.tie_word_embeddings,
        **dense_markers,
    }
    if any(payload.get(name) != value for name, value in required.items()):
        raise DynamicLfm2AdapterError(
            "Serialized dense budget LFM2 config dropped or changed a "
            "catalog-selected topology field."
        )
    return result


def materialize_dense_budgeted_lfm2_child(
    parent_model: Any,
    selected_target: Lfm2DenseBudgetedSelectedTarget,
    *,
    child_name: str,
    seed: int,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Lfm2MaterializedChild:
    """Instantiate and inspect a genuine dense width/depth-reduced blank."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise DynamicLfm2AdapterError(
            "PyTorch is required to materialize an LFM2 child."
        ) from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DynamicLfm2AdapterError("Blank child seed must be non-negative.")
    parameters = tuple(parent_model.parameters())
    if not parameters:
        raise DynamicLfm2AdapterError("Loaded parent has no parameters.")
    parent_devices = {value.device for value in parameters}
    parent_dtypes = {value.dtype for value in parameters}
    heterogeneous_descriptor = selected_target.heterogeneous_layout_descriptor
    if device is None:
        if len(parent_devices) != 1:
            raise DynamicLfm2AdapterError(
                "A sharded parent requires an explicit child device."
            )
        device = next(iter(parent_devices))
    else:
        device = torch.device(device)
    if heterogeneous_descriptor is not None:
        if dtype is None:
            dtype = torch.float32
        elif dtype != torch.float32:
            raise DynamicLfm2AdapterError(
                "Heterogeneous dense LFM2 physical layouts require FP32 "
                "parameters."
            )
    elif dtype is None:
        if len(parent_dtypes) != 1:
            raise DynamicLfm2AdapterError(
                "A mixed-dtype parent requires an explicit child dtype."
            )
        dtype = next(iter(parent_dtypes))
    target_config = build_dense_budgeted_lfm2_config(
        parent_model,
        selected_target,
    )
    cuda_devices: list[int] = []
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        cuda_devices = [index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        child = type(parent_model)(copy.deepcopy(target_config))
    if heterogeneous_descriptor is not None:
        try:
            physical_receipt = materialize_lfm2_dynamic_blank_child(
                child,
                child.config,
                heterogeneous_descriptor,
                expected_descriptor_sha256=(
                    heterogeneous_descriptor.descriptor_sha256
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DynamicLfm2AdapterError(
                "Could not physically materialize the selected heterogeneous "
                "LFM2 blank."
            ) from exc
        _validate_heterogeneous_materialization_receipt(
            selected_target,
            physical_receipt,
        )
        target_config = child.config
    child.to(device=device, dtype=dtype)
    child.eval()
    graph = inspect_model_circuits(
        child,
        model_name=child_name,
        tokenizer_fingerprint=(
            selected_target.child_tokenizer_semantics_sha256
        ),
    )
    validate_lfm2_dense_budgeted_child(selected_target, child, graph)
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise DynamicLfm2AdapterError(
            "Fresh dense budget LFM2 child unexpectedly contains gradients."
        )
    return Lfm2MaterializedChild(
        model=child,
        graph=graph,
        seed=seed,
        blank_state_fingerprint=fingerprint_model_state(child),
        config_fingerprint=canonical_sha256(target_config.to_dict()),
        exact_unique_parameters=graph.unique_parameter_count,
    )


class Lfm2TargetSynthesisAdapter:
    """Generic target registry adapter backed by the proven LFM2 planner."""

    adapter_id = "neural_forge.lfm2.progressive_exact"
    version = 1
    priority = 100
    target_families = ("lfm2",)
    target_modalities = ("text",)

    def __init__(
        self,
        *,
        terminal_topology: Lfm2Topology | None,
        evidence_fingerprint: str,
        ledger: ExperimentLedger | None = None,
        initial_mlp_step: int = 1,
        maximum_mlp_step: int = 64,
        device: Any | None = None,
        dtype: Any | None = None,
        tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None,
    ) -> None:
        if terminal_topology is not None and not isinstance(
            terminal_topology,
            Lfm2Topology,
        ):
            raise DynamicLfm2AdapterError(
                "terminal_topology must be Lfm2Topology or None."
            )
        self.terminal_topology = terminal_topology
        self.evidence_fingerprint = _digest(
            evidence_fingerprint,
            "evidence_fingerprint",
        )
        self.ledger = ledger if ledger is not None else ExperimentLedger()
        self.initial_mlp_step = _positive_int(
            initial_mlp_step,
            "initial_mlp_step",
        )
        self.maximum_mlp_step = _positive_int(
            maximum_mlp_step,
            "maximum_mlp_step",
        )
        if self.maximum_mlp_step < self.initial_mlp_step:
            raise DynamicLfm2AdapterError(
                "maximum_mlp_step cannot be smaller than initial_mlp_step."
            )
        self.device = device
        self.dtype = dtype
        if tokenizer_bridge_artifact is not None:
            if not isinstance(
                tokenizer_bridge_artifact,
                StrictTokenizerBridgeArtifact,
            ):
                raise DynamicLfm2AdapterError(
                    "tokenizer_bridge_artifact must be a "
                    "StrictTokenizerBridgeArtifact."
                )
            try:
                tokenizer_bridge_artifact.verify_files()
            except ValueError as exc:
                raise DynamicLfm2AdapterError(
                    "tokenizer_bridge_artifact failed verification."
                ) from exc
        self.tokenizer_bridge_artifact = tokenizer_bridge_artifact

    def supports(
        self,
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> bool:
        configuration = dict(request.configuration)
        dense_budget_request = _DENSE_BUDGET_CATALOG_MARKER in configuration
        bridge_required = (
            configuration.get(
                "_neural_forge_tokenizer_bridge_required"
            )
            is True
        )
        if bridge_required:
            bridge = self.tokenizer_bridge_artifact
            if not isinstance(bridge, StrictTokenizerBridgeArtifact):
                return False
            try:
                bridge.verify_files()
            except ValueError:
                return False
            if any(
                configuration.get(name) != value
                for name, value in strict_tokenizer_bridge_configuration(
                    bridge
                ).items()
            ):
                return False
        return (
            request.target_family in {"auto", "lfm2"}
            and request.target_modalities == ("text",)
            and (
                self.terminal_topology is not None
                or (
                    dense_budget_request
                    and request.fold_method_id
                    in {None, _DENSE_BUDGET_MODEL_METHOD}
                )
            )
            and request.fold_method_id
            in {
                None,
                LAYER_METHOD,
                HYBRID_MODEL_METHOD,
                PRODUCER_FIRST_MODEL_METHOD,
                _DENSE_BUDGET_MODEL_METHOD,
            }
            and any(
                graph.family == "lfm2" and graph.tokenizer_fingerprint is not None
                for graph in parent_graphs.values()
            )
        )

    @staticmethod
    def _source_name(
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> str:
        configured = dict(request.configuration).get("source_parent")
        if configured is not None:
            if not isinstance(configured, str) or configured not in parent_graphs:
                raise DynamicLfm2AdapterError(
                    "Configured source_parent is not a loaded parent."
                )
            if parent_graphs[configured].family != "lfm2":
                raise DynamicLfm2AdapterError(
                    "Configured source_parent is not an LFM2 graph."
                )
            return configured
        candidates = [
            (name, graph)
            for name, graph in parent_graphs.items()
            if graph.family == "lfm2"
        ]
        if not candidates:
            raise DynamicLfm2AdapterError(
                "No loaded LFM2 parent can define the target topology."
            )
        candidates.sort(
            key=lambda item: (
                -item[1].unique_parameter_count,
                item[0],
            )
        )
        return candidates[0][0]

    def materialize(
        self,
        request: DynamicTargetRequest,
        parent_models: Mapping[str, Any],
        parent_graphs: Mapping[str, ModelCircuitGraph],
        _inspectors: Any,
    ) -> AdapterMaterialization:
        source_name = self._source_name(request, parent_graphs)
        parent_model = parent_models[source_name]
        parent_graph = parent_graphs[source_name]
        bridge_required = (
            dict(request.configuration).get(
                "_neural_forge_tokenizer_bridge_required"
            )
            is True
        )
        if (
            not bridge_required
            and request.tokenizer_fingerprint is not None
            and request.tokenizer_fingerprint
            != parent_graph.tokenizer_fingerprint
        ):
            raise DynamicLfm2AdapterError(
                "Requested child tokenizer differs from the selected LFM2 "
                "source interface."
            )
        if _DENSE_BUDGET_CATALOG_MARKER in dict(request.configuration):
            selected_budgeted = _select_lfm2_dense_budgeted_target(
                request,
                parent_model,
                parent_graph,
                self.tokenizer_bridge_artifact,
            )
            materialized_budgeted = materialize_dense_budgeted_lfm2_child(
                parent_model,
                selected_budgeted,
                child_name=request.child_name,
                seed=request.blank_seed,
                device=self.device,
                dtype=self.dtype,
            )
            return AdapterMaterialization(
                model=materialized_budgeted.model,
                graph=materialized_budgeted.graph,
                topology_fingerprint=canonical_sha256(
                    {
                        "adapter": self.adapter_id,
                        "selected_target": selected_budgeted.as_dict(),
                        "materialized_config": (
                            materialized_budgeted.config_fingerprint
                        ),
                    }
                ),
                report={
                    "source_parent": source_name,
                    "dense_budget_catalog": selected_budgeted.catalog_id,
                    "component_allocation_policy": (
                        selected_budgeted.component_allocation_policy
                    ),
                    "component_allocation_receipt_sha256": (
                        selected_budgeted.component_allocation_receipt_sha256
                    ),
                    "component_allocation_topology_sha256": (
                        selected_budgeted.component_allocation_topology_sha256
                    ),
                    "component_allocation_verified_by_target_adapter": True,
                    "target_depth": selected_budgeted.target_depth,
                    "target_layer_types": encode_lfm2_layer_types(
                        selected_budgeted.layer_types
                    ),
                    "ordered_source_spans": encode_lfm2_source_spans(
                        selected_budgeted.ordered_source_spans
                    ),
                    "target_hidden_size": selected_budgeted.hidden_size,
                    "target_intermediate_size": (
                        selected_budgeted.intermediate_size
                    ),
                    "target_query_heads": selected_budgeted.query_heads,
                    "target_key_value_heads": (
                        selected_budgeted.key_value_heads
                    ),
                    "target_head_dim": selected_budgeted.head_dim,
                    "target_logical_parameters": (
                        selected_budgeted.exact_target_logical_parameters
                    ),
                    "target_unique_parameters": (
                        selected_budgeted.exact_target_unique_parameters
                    ),
                    "native_token_interface_tied": (
                        selected_budgeted.tie_word_embeddings
                    ),
                    **(
                        {
                            "physical_layout_kind": (
                                LFM2_HETEROGENEOUS_LAYOUT_KIND
                            ),
                            "physical_layout_descriptor_sha256": (
                                selected_budgeted
                                .heterogeneous_layout_descriptor
                                .descriptor_sha256
                            ),
                            "physical_layout_selection_receipt_sha256": (
                                selected_budgeted
                                .heterogeneous_layout_selection_receipt
                                .receipt_sha256
                            ),
                            "heterogeneous_per_layer_widths": True,
                            "fp32_physical_parameters": True,
                        }
                        if selected_budgeted.heterogeneous_layout
                        else {}
                    ),
                    **(
                        {
                            "tokenizer_bridge_required": True,
                            "parent_tokenizer_semantics_sha256": (
                                selected_budgeted.parent_tokenizer_semantics_sha256
                            ),
                            "child_tokenizer_semantics_sha256": (
                                selected_budgeted.child_tokenizer_semantics_sha256
                            ),
                            "tokenizer_bridge_artifact_sha256": (
                                selected_budgeted.tokenizer_bridge_artifact_sha256
                            ),
                            "tokenizer_transport_profile_sha256": (
                                selected_budgeted.tokenizer_transport_profile_sha256
                            ),
                        }
                        if selected_budgeted.tokenizer_bridge_required
                        else {}
                    ),
                    "blank_state_fingerprint": (
                        materialized_budgeted.blank_state_fingerprint
                    ),
                    "local_parent_class_instantiation": True,
                    "hub_loader_used": False,
                    "dense_parameters": True,
                    "quantization_used": False,
                    "pruning_used": False,
                    "training_used": False,
                },
                adapter_payload=selected_budgeted,
            )
        if request.fold_method_id == HYBRID_MODEL_METHOD:
            if self.terminal_topology is None:
                raise DynamicLfm2AdapterError(
                    "Hybrid target materialization requires terminal_topology."
                )
            configuration = dict(request.configuration)
            target_layer_types = decode_lfm2_layer_types(
                configuration.get("target_layer_types")
            )
            ordered_source_spans = decode_lfm2_source_spans(
                configuration.get("ordered_source_spans")
            )
            target_depth = configuration.get("target_depth")
            if (
                isinstance(target_depth, bool)
                or not isinstance(target_depth, int)
                or target_depth != len(target_layer_types)
            ):
                raise DynamicLfm2AdapterError(
                    "Hybrid target_depth must equal target_layer_types length."
                )
            selected_hybrid = select_lfm2_hybrid_target(
                parent_model,
                parent_graph,
                topology=self.terminal_topology,
                target_layer_types=target_layer_types,
                ordered_source_spans=ordered_source_spans,
                exact_target_unique_parameters=request.exact_unique_parameters,
                evidence_fingerprint=self.evidence_fingerprint,
                ledger=self.ledger,
            )
            materialized_hybrid = materialize_hybrid_lfm2_child(
                parent_model,
                selected_hybrid,
                child_name=request.child_name,
                seed=request.blank_seed,
                device=self.device,
                dtype=self.dtype,
            )
            return AdapterMaterialization(
                model=materialized_hybrid.model,
                graph=materialized_hybrid.graph,
                topology_fingerprint=canonical_sha256(
                    {
                        "adapter": self.adapter_id,
                        "selected_target": selected_hybrid.as_dict(),
                        "materialized_config": (
                            materialized_hybrid.config_fingerprint
                        ),
                    }
                ),
                report={
                    "source_parent": source_name,
                    "fold_method_id": HYBRID_MODEL_METHOD,
                    "target_depth": selected_hybrid.target_depth,
                    "target_layer_types": encode_lfm2_layer_types(
                        selected_hybrid.target_layer_types
                    ),
                    "ordered_source_spans": encode_lfm2_source_spans(
                        selected_hybrid.ordered_source_spans
                    ),
                    "blank_state_fingerprint": (
                        materialized_hybrid.blank_state_fingerprint
                    ),
                    "local_parent_class_instantiation": True,
                    "hub_loader_used": False,
                },
                adapter_payload=selected_hybrid,
            )
        if self.terminal_topology is None:
            raise DynamicLfm2AdapterError(
                "Non-budgeted target materialization requires terminal_topology."
            )
        selected = select_exact_lfm2_target(
            parent_model,
            parent_graph,
            terminal_topology=self.terminal_topology,
            exact_target_unique_parameters=request.exact_unique_parameters,
            evidence_fingerprint=self.evidence_fingerprint,
            ledger=self.ledger,
            initial_mlp_step=self.initial_mlp_step,
            maximum_mlp_step=self.maximum_mlp_step,
        )
        materialized = materialize_selected_lfm2_child(
            parent_model,
            selected,
            child_name=request.child_name,
            seed=request.blank_seed,
            device=self.device,
            dtype=self.dtype,
            fold_method_id=(
                PRODUCER_FIRST_MODEL_METHOD
                if request.fold_method_id == PRODUCER_FIRST_MODEL_METHOD
                else None
            ),
        )
        return AdapterMaterialization(
            model=materialized.model,
            graph=materialized.graph,
            topology_fingerprint=canonical_sha256(
                {
                    "adapter": self.adapter_id,
                    "selected_target": selected.as_dict(),
                    "materialized_config": materialized.config_fingerprint,
                }
            ),
            report={
                "source_parent": source_name,
                **(
                    {"fold_method_id": PRODUCER_FIRST_MODEL_METHOD}
                    if request.fold_method_id == PRODUCER_FIRST_MODEL_METHOD
                    else {}
                ),
                "selected_stage_index": selected.stage.stage_index,
                "selected_target_plan": selected.plan.fingerprint,
                "blank_state_fingerprint": (materialized.blank_state_fingerprint),
                "local_parent_class_instantiation": True,
                "hub_loader_used": False,
            },
            adapter_payload=selected,
        )


def materialize_and_execute_selected_lfm2_target(
    *,
    parent_model: Any,
    parent_name: str,
    child_name: str,
    bridge: Lfm2DynamicCompilerBridge,
    ledger: ExperimentLedger,
    tokenizer_fingerprint: str,
    blank_seed: int,
    device: Any | None = None,
    dtype: Any | None = None,
    stop_on_rejection: bool = True,
) -> tuple[Lfm2MaterializedChild, Lfm2SelectedTargetRun]:
    """Materialize the selected exact child, then compile it automatically."""

    materialized = materialize_selected_lfm2_child(
        parent_model,
        bridge.selected_target,
        child_name=child_name,
        seed=blank_seed,
        device=device,
        dtype=dtype,
    )
    run = execute_selected_lfm2_target(
        parent_model=parent_model,
        parent_name=parent_name,
        child_model=materialized.model,
        child_name=child_name,
        bridge=bridge,
        ledger=ledger,
        parent_tokenizer_fingerprint=tokenizer_fingerprint,
        child_tokenizer_fingerprint=tokenizer_fingerprint,
        stop_on_rejection=stop_on_rejection,
    )
    return materialized, run


def execute_selected_lfm2_target(
    *,
    parent_model: Any,
    parent_name: str,
    child_model: Any,
    child_name: str,
    bridge: Lfm2DynamicCompilerBridge,
    ledger: ExperimentLedger,
    parent_tokenizer_fingerprint: str,
    child_tokenizer_fingerprint: str,
    stop_on_rejection: bool = True,
) -> Lfm2SelectedTargetRun:
    """Compile one selected stage, then append its run-level closure record.

    The generic executor owns model rollback.  The caller remains responsible
    for persisting the in-memory ledger; model execution and external ledger
    persistence are not represented as one cross-resource atomic transaction.
    """

    if not isinstance(bridge, Lfm2DynamicCompilerBridge):
        raise DynamicLfm2AdapterError("bridge must be Lfm2DynamicCompilerBridge.")
    if not isinstance(ledger, ExperimentLedger):
        raise DynamicLfm2AdapterError("ledger must be ExperimentLedger.")
    selected = bridge.selected_target
    selected.validate()
    if parent_name != selected.parent_name:
        raise DynamicLfm2AdapterError(
            "Loaded parent name differs from the selected target binding."
        )
    expected_parent_states = dict(selected.signature.parent_state_fingerprints)
    live_parent_state = fingerprint_model_state(parent_model)
    if expected_parent_states != {parent_name: live_parent_state}:
        raise DynamicLfm2AdapterError(
            "Loaded parent checkpoint differs from the selected target binding."
        )
    current_decision = ledger.decision(selected.signature)
    if not current_decision.allowed:
        raise DynamicLfm2AdapterError(
            "The selected exact-budget target is already closed."
        )
    parent_tokenizer_fingerprint = _digest(
        parent_tokenizer_fingerprint,
        "parent_tokenizer_fingerprint",
    )
    child_tokenizer_fingerprint = _digest(
        child_tokenizer_fingerprint,
        "child_tokenizer_fingerprint",
    )
    if parent_tokenizer_fingerprint != child_tokenizer_fingerprint:
        raise DynamicLfm2AdapterError(
            "The current LFM2 bridge requires an identical tokenizer interface."
        )
    if parent_tokenizer_fingerprint != selected.tokenizer_fingerprint:
        raise DynamicLfm2AdapterError(
            "Loaded tokenizer differs from the selected target binding."
        )
    parent_graph = inspect_model_circuits(
        parent_model,
        model_name=parent_name,
        tokenizer_fingerprint=parent_tokenizer_fingerprint,
    )
    _validate_selected_target_against_parent(
        selected,
        parent_graph,
        live_parent_state,
    )
    if (parent_graph.fingerprint,) != selected.signature.parent_graph_fingerprints:
        raise DynamicLfm2AdapterError(
            "Loaded parent graph differs from the selected target source."
        )
    child_graph = inspect_model_circuits(
        child_model,
        model_name=child_name,
        tokenizer_fingerprint=child_tokenizer_fingerprint,
    )
    validate_lfm2_selected_child(selected, child_model, child_graph)
    if _layer_operator_sequence(parent_graph) != _layer_operator_sequence(child_graph):
        raise DynamicLfm2AdapterError(
            "Loaded child changes the selected LFM2 hybrid layer ordering."
        )
    compiler = DynamicFoldingCompiler(
        operators=bridge.operators,
        ledger=ledger,
    )
    compilation = compiler.compile(
        parent_models={parent_name: parent_model},
        child_model=child_model,
        child_name=child_name,
        evidence_provider=bridge.evidence_provider,
        evaluator=bridge.evaluator,
        evidence_fingerprint=bridge.evidence_fingerprint,
        parent_tokenizer_fingerprints={parent_name: parent_tokenizer_fingerprint},
        child_tokenizer_fingerprint=child_tokenizer_fingerprint,
        exact_target_unique_parameters=selected.exact_target_unique_parameters,
        initializer=bridge.initializer,
        final_evaluator=bridge.final_evaluator,
        stop_on_rejection=stop_on_rejection,
    )
    if tuple(compilation.plan.parent_state_fingerprints) != tuple(
        selected.signature.parent_state_fingerprints
    ):
        raise DynamicLfm2AdapterError(
            "Compiled run parent state differs from target selection."
        )
    if compilation.run.status == "complete":
        record_status = RecordStatus.ACCEPTED
        reason_class = "whole_output_behavior_accept"
        assert compilation.run.final_decision is not None
        reason = compilation.run.final_decision.reason or "accepted"
    elif compilation.run.status in {"rejected", "final_gate_rejected"}:
        record_status = RecordStatus.REJECTED
        reason_class = "whole_output_behavior_reject"
        if compilation.run.final_decision is not None:
            reason = compilation.run.final_decision.reason or "final gate rejected"
        elif compilation.run.attempts:
            reason = compilation.run.attempts[-1].decision.reason or "rejected"
        else:
            reason = "rejected before an executable fold attempt"
    else:
        raise DynamicLfm2AdapterError(
            "Non-terminal compiler status did not close the selected target: "
            + compilation.run.status
        )
    selection_record = LedgerRecord(
        signature=selected.signature,
        closure_level=ClosureLevel.EXACT,
        status=record_status,
        reason_class=reason_class,
        reason=reason,
        metrics=(
            ("run_status", compilation.run.status),
            ("run_complete", compilation.run.complete),
            ("attempt_count", len(compilation.run.attempts)),
            (
                "accepted_action_count",
                len(compilation.run.accepted_action_ids),
            ),
            (
                "exact_target_unique_parameters",
                selected.exact_target_unique_parameters,
            ),
            ("final_behavior_gate_present", True),
        ),
        previous_record_sha256=(
            None if not ledger.records else ledger.records[-1].record_sha256
        ),
    )
    ledger.append(selection_record)
    return Lfm2SelectedTargetRun(compilation, selection_record)


def assert_lfm2_fold_only_runtime(model: Any) -> Mapping[str, Any]:
    """Inspect current gradient state without inventing historical counters."""

    gradients = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    return MappingProxyType(
        {
            "child_state_fingerprint": fingerprint_model_state(model),
            "gradient_tensor_count": len(gradients),
            "gradient_tensors_absent": len(gradients) == 0,
        }
    )


__all__ = [
    "ADAPTER_SCHEMA",
    "DynamicLfm2AdapterError",
    "HYBRID_MODEL_METHOD",
    "LAYER_METHOD",
    "PRODUCER_FIRST_MODEL_METHOD",
    "Lfm2CachedEvidenceProvider",
    "Lfm2ClosedLoopLayerOperator",
    "Lfm2DynamicCompilerBridge",
    "Lfm2DynamicLayerTransaction",
    "Lfm2DynamicModelTransaction",
    "Lfm2DenseBudgetedSelectedTarget",
    "Lfm2EvidenceBuildRequest",
    "Lfm2EvidenceBuilder",
    "Lfm2FoldAdmissionEvaluator",
    "Lfm2FoldOnlyInitializer",
    "Lfm2HybridModelOperator",
    "Lfm2HybridSelectedTarget",
    "Lfm2InspectedSource",
    "Lfm2MaterializedChild",
    "Lfm2ModelEvidenceBuildRequest",
    "Lfm2ModelEvidenceBuilder",
    "Lfm2PreparedEvidence",
    "Lfm2PreparedHybridProducerEvidence",
    "Lfm2PreparedModelEvidence",
    "Lfm2PreparedProducerFirstEvidence",
    "Lfm2ProducerFirstEvidenceBuildRequest",
    "Lfm2ProducerFirstEvidenceBuilder",
    "Lfm2ProducerFirstModelOperator",
    "Lfm2SelectedTarget",
    "Lfm2SelectedTargetLike",
    "Lfm2SealedProducerFirstEvidence",
    "Lfm2SelectedTargetRun",
    "Lfm2TargetSynthesisAdapter",
    "TARGET_SELECTION_METHOD",
    "assert_lfm2_fold_only_runtime",
    "build_dense_budgeted_lfm2_config",
    "build_hybrid_lfm2_config",
    "build_selected_lfm2_config",
    "build_lfm2_dynamic_compiler_bridge",
    "decode_lfm2_layer_types",
    "decode_lfm2_source_spans",
    "derive_lfm2_target_ladder",
    "encode_lfm2_layer_types",
    "encode_lfm2_source_spans",
    "execute_selected_lfm2_target",
    "fingerprint_lfm2_fold_evidence",
    "fingerprint_lfm2_model_fold_evidence",
    "fingerprint_lfm2_producer_first_evidence",
    "inspect_lfm2_progressive_source",
    "materialize_and_execute_selected_lfm2_target",
    "materialize_dense_budgeted_lfm2_child",
    "materialize_hybrid_lfm2_child",
    "materialize_selected_lfm2_child",
    "select_first_open_lfm2_target",
    "select_exact_lfm2_target",
    "select_lfm2_hybrid_target",
    "validate_lfm2_hybrid_child",
    "validate_lfm2_dense_budgeted_child",
    "validate_lfm2_selected_child",
]
