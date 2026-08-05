"""Automatic causal-circuit alignment and fold-plan generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dynamic_circuit_ir import (
    MODALITY_AUDIO,
    MODALITY_POINTS3D,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    CircuitKind,
    CircuitNode,
    EdgeKind,
    JsonScalar,
    ModelCircuitGraph,
    PortKind,
    canonical_sha256,
)
from .dynamic_fold_registry import (
    ExperimentLedger,
    ExperimentSignature,
    FoldMethodSpec,
    FoldMethodRegistry,
    LedgerDecision,
    MethodStatus,
    default_method_registry,
)


PLAN_SCHEMA = "neural-forge-dynamic-fold-plan-v1"

CAUSAL_ATTENTION_BRANCH_METHOD = "causal_gqa_rope_residual_attention_layer"
CAUSAL_GATED_MLP_BRANCH_METHOD = "causal_swiglu_residual_mlp_layer"
# Both branches must be enabled together: a layer whose attention half has no
# owning transaction cannot be closed by folding its MLP half alone.
CAUSAL_LAYER_BRANCH_METHODS = (
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_GATED_MLP_BRANCH_METHOD,
)
#: The same coupled kernel under a NoPE policy: windowed local attention whose
#: RoPE transport is the identity rather than a rotation, and a dense gated MLP
#: whose gate and up projections share one fused tensor.  These are separate
#: method ids, not a relaxation of the RoPE pair, because the operator that
#: executes them is a different executable with a different proven envelope.
CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD = "causal_windowed_nope_attention_layer"
CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD = "causal_fused_gate_up_mlp_layer"
NATIVE_CAUSAL_LAYER_BRANCH_METHODS = (
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
    CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
)
#: The two remaining chimera sequence-mixer families that have a fold kernel.
#: The names are the design document's own
#: (``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 4); the gap
#: capability names the planner reports while they are *missing* are different
#: strings on purpose, because a capability is what an implementer must build
#: and a method id is what an operator executes.
CAUSAL_KDA_RECURRENT_BRANCH_METHOD = "causal_kda_recurrent_layer"
#: One method id covers both ``csa`` and ``hca``.  They are the same
#: ``SparseLatentAttention`` module at two strides and one stride-parameterized
#: kernel executes both, so a second id would claim a second executable that
#: does not exist.  The stride is a contract constant of the crossing: it is
#: sealed in the action configuration and the receipt, and changing it fails
#: closed as a kind translation rather than folding.
CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD = "causal_summary_attention_layer"
#: The last two chimera families, named by
#: ``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`` sections 1 and 2.
#: One method id covers both gated-MLA query variants: the dense and the
#: compressed query path are the same ``GatedMultiHeadLatentAttention`` class
#: under one config constant, so the latent size is a sealed contract shape of
#: the crossing rather than a second executable.
CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD = "causal_latent_kv_attention_layer"
#: The Stable LatentMoE channel mixer: one composite operator making one
#: ordered residual write, so one method id rather than a router method plus an
#: expert method.
CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD = (
    "causal_stable_latent_moe_channel_mixer_layer"
)
#: Every native-trunk branch method, for a caller that wants the whole chimera
#: sequence-mixer coverage rather than only the local/dense pair.
NATIVE_CHIMERA_LAYER_BRANCH_METHODS = (
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
    CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
    CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
    CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
    CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
    CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
)
#: Global interface circuits whose only difference is the residual coordinate
#: system a contraction chart already covers.
NATIVE_RESIDUAL_CHART_INTERFACE_METHOD = "native_residual_chart_interface"
_CAUSAL_DECODER_INSPECTOR = "hf-causal-decoder"
#: Inspectors whose graphs describe a causal decoder trunk that carries order
#: without any rotary position encoding.  Membership is not a courtesy: a graph
#: in this set must still satisfy every branch role contract node by node, and
#: the RoPE lane stays reserved for graphs that really do rotate.
_NATIVE_DECODER_INSPECTORS = frozenset({"chimera-omni-native-decoder"})
#: Attribute keys that only a rotary implementation can honour.  A NoPE fold
#: refuses any node that declares one instead of pretending the rotation is
#: absent.
_ROPE_SPECIFIC_ATTRIBUTES = (
    "rope_theta",
    "rope_type",
    "rope_scaling",
    "rope_variants",
    "partial_rotary_factor",
    "rotary_dim",
    "rotary_emb_base",
)
#: The mixer/channel-mixer kernel families the IR knows how to name but this
#: repository cannot yet execute.  The planner reports the missing capability
#: by name rather than collapsing every uncovered layer into one generic gap.
_NATIVE_KERNEL_FAMILY_GAPS = {
    "bounded_gated_delta_recurrence": (
        "bounded_gated_delta_recurrence_layer_operator"
    ),
    "compressed_summary_attention": "compressed_summary_attention_layer_operator",
    "compressed_latent_attention": (
        "compressed_latent_kv_attention_layer_operator"
    ),
    "stable_latent_moe_channel_mixer": (
        "stable_latent_moe_channel_mixer_operator"
    ),
}
#: Ordered-branch labels for the two causal branches.  ``operator`` and
#: ``feed_forward`` are the native-trunk spellings of ``attention`` and ``mlp``.
_ATTENTION_BRANCH_LABELS = ("attention", "operator")
_MLP_BRANCH_LABELS = ("mlp", "feed_forward")
_NATIVE_AUDIO_FAMILY = "native_audio_chat"
_NATIVE_AUDIO_MODALITIES = frozenset({MODALITY_AUDIO, MODALITY_TEXT})
_NATIVE_VIDEO_FAMILY = "native_video_chat"
_NATIVE_VIDEO_MODALITIES = frozenset({MODALITY_VIDEO, MODALITY_TEXT})
_NATIVE_POINTS3D_FAMILY = "native_points3d_chat"
# The modality *set* is unchanged by the optional emitted-centroid head: that
# head adds a second ``points3d`` interface, in the output direction, not a
# fourth modality.  Direction is checked separately below.
_NATIVE_POINTS3D_MODALITIES = frozenset({MODALITY_POINTS3D, MODALITY_TEXT})


class DynamicFoldPlanningError(ValueError):
    """The requested target cannot yet be covered by registered fold operators."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicFoldPlanningError(f"{name} must be a nonempty string.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicFoldPlanningError(f"{name} must be a positive integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicFoldPlanningError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _is_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _attributes(
    values: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    if values is None:
        return ()
    items = values.items() if isinstance(values, Mapping) else values
    result = sorted((str(key), value) for key, value in items)
    if len({key for key, _ in result}) != len(result):
        raise DynamicFoldPlanningError("Action configuration keys repeat.")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SourceCircuitRef:
    parent_name: str
    node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_name", _text(self.parent_name, "parent_name"))
        node_ids = tuple(
            sorted({_text(value, "source node_id") for value in self.node_ids})
        )
        if not node_ids:
            raise DynamicFoldPlanningError(
                "A source circuit reference cannot be empty."
            )
        object.__setattr__(self, "node_ids", node_ids)

    def as_dict(self) -> dict[str, Any]:
        return {"parent_name": self.parent_name, "node_ids": list(self.node_ids)}


@dataclass(frozen=True, slots=True)
class FoldAction:
    """One atomic complete-circuit transaction selected without layer surgery."""

    action_id: str
    sequence_index: int
    method_id: str
    sources: tuple[SourceCircuitRef, ...]
    target_node_ids: tuple[str, ...]
    target_parameter_paths: tuple[str, ...]
    target_storage_groups: tuple[str, ...]
    dependencies: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    source_logical_parameters: int
    target_logical_parameters: int
    configuration: tuple[tuple[str, JsonScalar], ...]
    signature: ExperimentSignature
    ledger_decision: LedgerDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        object.__setattr__(
            self,
            "sequence_index",
            _positive_int(self.sequence_index + 1, "sequence_index") - 1,
        )
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        sources = tuple(sorted(self.sources, key=lambda value: value.parent_name))
        if not sources or any(
            not isinstance(value, SourceCircuitRef) for value in sources
        ):
            raise DynamicFoldPlanningError("Fold action sources are invalid.")
        object.__setattr__(self, "sources", sources)
        for name in (
            "target_node_ids",
            "target_parameter_paths",
            "target_storage_groups",
            "dependencies",
            "evidence_requirements",
        ):
            object.__setattr__(
                self,
                name,
                tuple(sorted({_text(value, name) for value in getattr(self, name)})),
            )
        if not self.target_node_ids:
            raise DynamicFoldPlanningError("Fold action target nodes cannot be empty.")
        for name in ("source_logical_parameters", "target_logical_parameters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DynamicFoldPlanningError(f"{name} must be non-negative.")
        object.__setattr__(self, "configuration", _attributes(self.configuration))
        if not isinstance(self.signature, ExperimentSignature):
            raise DynamicFoldPlanningError("Fold action signature is invalid.")
        if not isinstance(self.ledger_decision, LedgerDecision):
            raise DynamicFoldPlanningError("Fold action ledger decision is invalid.")

    @property
    def compression_delta(self) -> int:
        return self.source_logical_parameters - self.target_logical_parameters

    @property
    def action_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "sequence_index": self.sequence_index,
            "method_id": self.method_id,
            "sources": [value.as_dict() for value in self.sources],
            "target_node_ids": list(self.target_node_ids),
            "target_parameter_paths": list(self.target_parameter_paths),
            "target_storage_groups": list(self.target_storage_groups),
            "dependencies": list(self.dependencies),
            "evidence_requirements": list(self.evidence_requirements),
            "source_logical_parameters": self.source_logical_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "compression_delta": self.compression_delta,
            "configuration": dict(self.configuration),
            "signature": self.signature.as_dict(),
            "ledger_decision": {
                "allowed": self.ledger_decision.allowed,
                "reason": self.ledger_decision.reason,
                "blocking_record_sha256": (self.ledger_decision.blocking_record_sha256),
                "closure_level": self.ledger_decision.closure_level,
            },
        }


def validate_fold_action_evidence_channels(
    action: FoldAction,
    available_channels: Sequence[str],
    *,
    allow_additional: bool = False,
) -> tuple[str, ...]:
    """Validate an evidence manifest against one action's exact method contract.

    The executor cannot infer semantic channel names from an arbitrary payload.
    Evidence providers can call this hook before constructing ``FoldEvidence``;
    production adapters should expose the returned canonical manifest in their
    own immutable evidence receipt.
    """

    if not isinstance(action, FoldAction):
        raise DynamicFoldPlanningError("action must be FoldAction.")
    if isinstance(available_channels, (str, bytes)):
        raise DynamicFoldPlanningError(
            "available_channels must be a sequence of channel names."
        )
    channels = tuple(
        sorted(_text(value, "evidence channel") for value in available_channels)
    )
    if len(set(channels)) != len(channels):
        raise DynamicFoldPlanningError(
            "Available evidence channel names must be unique."
        )
    required = action.evidence_requirements
    missing = sorted(set(required) - set(channels))
    unexpected = sorted(set(channels) - set(required))
    if missing or (unexpected and not allow_additional):
        raise DynamicFoldPlanningError(
            "Evidence channels do not satisfy the fold action contract: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return channels


@dataclass(frozen=True, slots=True)
class FoldPlanGap:
    target_node_ids: tuple[str, ...]
    reason: str
    required_capability: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_node_ids": list(self.target_node_ids),
            "reason": self.reason,
            "required_capability": self.required_capability,
        }


@dataclass(frozen=True, slots=True)
class DynamicFoldPlan:
    parent_graph_fingerprints: tuple[str, ...]
    parent_state_fingerprints: tuple[tuple[str, str], ...]
    child_graph_fingerprint: str
    child_state_fingerprint: str
    evidence_fingerprint: str
    exact_target_unique_parameters: int
    actions: tuple[FoldAction, ...]
    gaps: tuple[FoldPlanGap, ...]
    registry_fingerprint: str
    ledger_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_graph_fingerprints",
            tuple(
                sorted(
                    _digest(value, "parent graph")
                    for value in self.parent_graph_fingerprints
                )
            ),
        )
        if not self.parent_graph_fingerprints:
            raise DynamicFoldPlanningError("A plan requires at least one parent.")
        parent_states = tuple(
            sorted(
                (
                    _text(name, "parent state name"),
                    _digest(digest, "parent state fingerprint"),
                )
                for name, digest in self.parent_state_fingerprints
            )
        )
        if not parent_states or len({name for name, _value in parent_states}) != len(
            parent_states
        ):
            raise DynamicFoldPlanningError(
                "Plan parent state fingerprints must be uniquely named."
            )
        object.__setattr__(self, "parent_state_fingerprints", parent_states)
        for name in (
            "child_graph_fingerprint",
            "child_state_fingerprint",
            "evidence_fingerprint",
            "registry_fingerprint",
            "ledger_fingerprint",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "exact_target_unique_parameters",
            _positive_int(
                self.exact_target_unique_parameters,
                "exact_target_unique_parameters",
            ),
        )
        actions = tuple(sorted(self.actions, key=lambda value: value.sequence_index))
        if len({value.action_id for value in actions}) != len(actions):
            raise DynamicFoldPlanningError("Fold action IDs must be unique.")
        object.__setattr__(self, "actions", actions)
        target_paths = [
            path for action in actions for path in action.target_parameter_paths
        ]
        if len(set(target_paths)) != len(target_paths):
            raise DynamicFoldPlanningError(
                "A child parameter path is owned by multiple fold actions."
            )
        target_storage_groups = [
            group for action in actions for group in action.target_storage_groups
        ]
        if len(set(target_storage_groups)) != len(target_storage_groups):
            raise DynamicFoldPlanningError(
                "A child learned-storage group is owned by multiple fold actions."
            )
        object.__setattr__(self, "gaps", tuple(self.gaps))

    @property
    def ready(self) -> bool:
        return not self.gaps and all(
            action.ledger_decision.allowed for action in self.actions
        )

    @property
    def expected_logical_compression_delta(self) -> int:
        return sum(value.compression_delta for value in self.actions)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": PLAN_SCHEMA,
            "parent_graph_fingerprints": list(self.parent_graph_fingerprints),
            "parent_state_fingerprints": dict(self.parent_state_fingerprints),
            "child_graph_fingerprint": self.child_graph_fingerprint,
            "child_state_fingerprint": self.child_state_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "actions": [action.as_dict() for action in self.actions],
            "gaps": [gap.as_dict() for gap in self.gaps],
            "registry_fingerprint": self.registry_fingerprint,
            "ledger_fingerprint": self.ledger_fingerprint,
            "complete_target_parameter_coverage": not self.gaps,
            "known_closed_methods_filtered": True,
            "ready": self.ready,
            "expected_logical_compression_delta": (
                self.expected_logical_compression_delta
            ),
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def _node_semantic_contract(node: CircuitNode) -> dict[str, Any]:
    return {
        "kind": node.kind,
        "inputs": [value.as_dict() for value in node.inputs],
        "outputs": [value.as_dict() for value in node.outputs],
        "parameters": [
            {
                "role": value.role,
                "shape": list(value.shape),
                "dtype": value.dtype,
                "trainable": value.trainable,
            }
            for value in sorted(node.parameters, key=lambda item: item.role)
        ],
        "attributes": dict(node.attributes),
    }


def _residual_width_substituted_contract(
    node: CircuitNode,
    substitution: tuple[int, int],
) -> dict[str, Any]:
    """Restate one source node contract at the child's residual width.

    Only *port* dimensions are rewritten, and only for a node that owns no
    learned tensor.  A node with parameters keeps its real shapes, so a norm
    diagonal or an embedding matrix can never pass a width crossing off as an
    exact copy; a pure wiring node such as the RoPE operator, whose only
    difference is the residual width the chart set already covers, can.
    """

    source_width, target_width = substitution
    contract = _node_semantic_contract(node)
    if contract["parameters"]:
        return contract

    def rewrite(ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **port,
                "dimensions": [
                    target_width if value == source_width else value
                    for value in port["dimensions"]
                ],
            }
            for port in ports
        ]

    return {
        **contract,
        "inputs": rewrite(contract["inputs"]),
        "outputs": rewrite(contract["outputs"]),
    }


#: The interface classes one global residual chart can carry, and what the
#: registered operator actually does with each of them.
_RESIDUAL_ROW_TABLE = "residual_row_table"
_RESIDUAL_PRODUCER_PROJECTION = "residual_producer_projection"
_RESIDUAL_CONSUMER_HEAD = "residual_consumer_head"
_RESIDUAL_NORM_GAIN = "residual_norm_gain"
_DEPTH_SOURCE_SELECTOR = "depth_source_selector"


def _residual_width_restated_contract(
    node: CircuitNode,
    substitution: tuple[int, int],
) -> dict[str, Any]:
    """Restate one node contract, including its tensors, at the child width.

    This is deliberately weaker than
    :func:`_residual_width_substituted_contract`, which refuses to touch a node
    that owns a tensor because an exact copy must stay exact.  A chart-projected
    interface is *not* an exact copy: it is a different tensor, compiled by a
    registered operator, and the only claim this contract makes is that the
    child circuit is the parent circuit restated at the contracted residual
    width.  Every shape the operator installs is re-derived from the live child
    parameter, so this comparison decides plannability, never correctness.
    """

    source_width, target_width = substitution
    contract = _node_semantic_contract(node)

    def rewrite_dimensions(values: Sequence[Any]) -> list[Any]:
        return [target_width if value == source_width else value for value in values]

    return {
        **contract,
        "inputs": [
            {**port, "dimensions": rewrite_dimensions(port["dimensions"])}
            for port in contract["inputs"]
        ],
        "outputs": [
            {**port, "dimensions": rewrite_dimensions(port["dimensions"])}
            for port in contract["outputs"]
        ],
        "parameters": [
            {**parameter, "shape": rewrite_dimensions(parameter["shape"])}
            for parameter in contract["parameters"]
        ],
        "attributes": {
            key: target_width
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value == source_width
            else value
            for key, value in contract["attributes"].items()
        },
    }


def _native_chart_interface_class(
    node: CircuitNode,
    width: int,
) -> str | None:
    """Classify one global interface circuit for chart-projected folding."""

    parameters = tuple(node.parameters)
    attributes = _node_attributes(node)
    if node.kind == CircuitKind.RESIDUAL_ADD.value:
        if not _depth_source_selector_contract(node) or len(parameters) != 2:
            return None
        shapes = sorted(
            tuple(int(size) for size in value.shape) for value in parameters
        )
        roles = " ".join(sorted(_semantic_name(value.role) for value in parameters))
        if (
            shapes != [(width,), (width,)]
            or "norm" not in roles
            or "query" not in roles
        ):
            return None
        return _DEPTH_SOURCE_SELECTOR
    if len(parameters) != 1:
        return None
    shape = tuple(int(size) for size in parameters[0].shape)
    if node.kind == CircuitKind.NORM.value:
        if (
            shape != (width,)
            or attributes.get("norm_kind") != "rms_norm"
            or attributes.get("weight_kind") != "direct_scale"
        ):
            return None
        return _RESIDUAL_NORM_GAIN
    if len(shape) != 2 or shape[0] == shape[1]:
        # A square interface tensor cannot be told apart from its transpose by
        # shape alone, so it is refused rather than guessed at.
        return None
    if node.kind in {
        CircuitKind.OUTPUT_HEAD.value,
        CircuitKind.MODALITY_OUTPUT.value,
    }:
        return _RESIDUAL_CONSUMER_HEAD if shape[1] == width else None
    if node.kind == CircuitKind.TOKEN_EMBEDDING.value:
        return _RESIDUAL_ROW_TABLE if shape[1] == width else None
    if node.kind in {
        CircuitKind.MODALITY_INPUT.value,
        CircuitKind.POSITION_ENCODING.value,
    }:
        if shape[1] == width:
            return _RESIDUAL_ROW_TABLE
        if shape[0] == width:
            return _RESIDUAL_PRODUCER_PROJECTION
    return None


def _native_chart_interface_classes(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> tuple[str, ...] | None:
    width = dict(graph.metadata).get("hidden_size")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        return None
    classes = []
    for node in nodes:
        value = _native_chart_interface_class(node, width)
        if value is None:
            return None
        classes.append(value)
    if not classes:
        return None
    return tuple(sorted(set(classes)))


def _native_chart_interface_group_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    return _native_chart_interface_classes(graph, nodes) is not None


def _native_chart_interface_crossing(
    parent: ModelCircuitGraph,
    source_nodes: Sequence[CircuitNode],
    child: ModelCircuitGraph,
    target_nodes: Sequence[CircuitNode],
    substitution: tuple[int, int],
) -> tuple[str, ...] | None:
    """Prove one global interface group is the parent's, restated narrower."""

    classes = _native_chart_interface_classes(child, target_nodes)
    if classes is None or _native_chart_interface_classes(parent, source_nodes) is None:
        return None
    source_by_id = {value.node_id: value for value in source_nodes}
    target_by_id = {value.node_id: value for value in target_nodes}
    if set(source_by_id) != set(target_by_id) or len(source_by_id) != len(
        tuple(source_nodes)
    ):
        return None
    for node_id, source in source_by_id.items():
        if _residual_width_restated_contract(
            source, substitution
        ) != _node_semantic_contract(target_by_id[node_id]):
            return None
    return classes


def _native_kernel_family_gaps(
    nodes: Sequence[CircuitNode],
) -> tuple[tuple[str, str], ...]:
    """Name every declared mixer kernel family this repository cannot execute.

    The IR already states what each operator needs
    (``fold_kernel_family_required``).  Reporting that name is the difference
    between "no operator covers this layer" and "the bounded gated delta
    recurrence has no kernel yet", and only the second one tells the next
    implementer what to build.
    """

    found: dict[str, list[str]] = {}
    for node in nodes:
        family = _node_attributes(node).get("fold_kernel_family_required")
        if not isinstance(family, str):
            continue
        capability = _NATIVE_KERNEL_FAMILY_GAPS.get(family)
        if capability is None:
            continue
        found.setdefault(family, []).append(node.node_id)
    return tuple(
        (
            _NATIVE_KERNEL_FAMILY_GAPS[family],
            f"No registered fold operator implements the {family!r} kernel "
            "family required by " + ", ".join(sorted(found[family])) + ".",
        )
        for family in sorted(found)
    )


def _same_parameter_contract(
    source_graph: ModelCircuitGraph,
    source_nodes: Sequence[CircuitNode],
    target_graph: ModelCircuitGraph,
    target_nodes: Sequence[CircuitNode],
    *,
    residual_width_substitution: tuple[int, int] | None = None,
) -> bool:
    if len(source_nodes) != len(target_nodes):
        return False
    source_by_id = {value.node_id: value for value in source_nodes}
    target_by_id = {value.node_id: value for value in target_nodes}
    if set(source_by_id) != set(target_by_id):
        return False

    def source_contract(node: CircuitNode) -> dict[str, Any]:
        if residual_width_substitution is None:
            return _node_semantic_contract(node)
        return _residual_width_substituted_contract(
            node,
            residual_width_substitution,
        )

    if any(
        source_contract(source_by_id[node_id])
        != _node_semantic_contract(target_by_id[node_id])
        for node_id in source_by_id
    ):
        return False
    node_ids = set(source_by_id)

    def induced_edges(graph: ModelCircuitGraph) -> tuple[dict[str, Any], ...]:
        return tuple(
            edge.as_dict()
            for edge in graph.edges
            if edge.source_node in node_ids and edge.target_node in node_ids
        )

    return induced_edges(source_graph) == induced_edges(target_graph)


def _source_span_indices(
    source_depth: int,
    target_depth: int,
    target_index: int,
) -> tuple[int, ...]:
    start = target_index * source_depth // target_depth
    stop = (target_index + 1) * source_depth // target_depth
    if stop <= start:
        stop = start + 1
    return tuple(range(start, min(stop, source_depth)))


def _nodes_for_span(
    graph: ModelCircuitGraph,
    indices: Sequence[int],
) -> tuple[CircuitNode, ...]:
    wanted = set(indices)
    return tuple(value for value in graph.nodes if value.layer_index in wanted)


def _global_match(
    parent: ModelCircuitGraph,
    target: CircuitNode,
) -> tuple[CircuitNode, ...]:
    exact_id = [value for value in parent.nodes if value.node_id == target.node_id]
    if exact_id:
        return tuple(exact_id)
    return tuple(
        value
        for value in parent.nodes
        if value.layer_index is None and value.kind == target.kind
    )


def _storage_groups(nodes: Sequence[CircuitNode]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {parameter.storage_group for node in nodes for parameter in node.parameters}
        )
    )


def _global_alias_bundle(
    child: ModelCircuitGraph,
    target: CircuitNode,
) -> tuple[CircuitNode, ...]:
    """Return all global nodes connected by learned-storage aliasing."""

    selected = {target.node_id}
    groups = set(_storage_groups((target,)))
    changed = True
    while changed and groups:
        changed = False
        for node in child.nodes:
            if node.layer_index is not None or node.node_id in selected:
                continue
            node_groups = set(_storage_groups((node,)))
            if not groups.intersection(node_groups):
                continue
            selected.add(node.node_id)
            groups.update(node_groups)
            changed = True
    return tuple(value for value in child.nodes if value.node_id in selected)


def _global_token_interface_bundle(
    child: ModelCircuitGraph,
) -> tuple[CircuitNode, ...]:
    """Return the complete learned token interface, including storage aliases.

    A tokenizer bridge is one coupled semantic transaction even when the target
    intentionally owns independent input-embedding and output-head parameters.
    Storage aliasing alone therefore cannot discover the complete bridge target.
    """

    interface_kinds = {
        CircuitKind.TOKEN_EMBEDDING.value,
        CircuitKind.OUTPUT_HEAD.value,
    }
    interface_nodes = tuple(
        node
        for node in child.nodes
        if node.layer_index is None and node.kind in interface_kinds
    )
    if {node.kind for node in interface_nodes} != interface_kinds:
        return ()
    selected = {
        alias.node_id
        for node in interface_nodes
        for alias in _global_alias_bundle(child, node)
    }
    return tuple(node for node in child.nodes if node.node_id in selected)


def _cross_width_terminal_interface_bundle(
    child: ModelCircuitGraph,
) -> tuple[CircuitNode, ...]:
    """Return the token interface plus the final norm it shares a chart with.

    Boundary 0 and the terminal boundary are the two interface-locked residual
    boundaries.  Under a width crossing the embedding writes boundary 0, and
    the final norm plus the output head read the terminal boundary; one chart
    pair fixes all three tensors, so they are one transaction.  The bundle is
    returned only when the child owns exactly one global norm and it sits after
    every layer, so nothing else can be swept in silently.
    """

    interface = _global_token_interface_bundle(child)
    if not interface:
        return ()
    layer_orders = [
        value.order for value in child.nodes if value.layer_index is not None
    ]
    if not layer_orders:
        return ()
    last_layer_order = max(layer_orders)
    norms = tuple(
        value
        for value in child.nodes
        if value.layer_index is None and value.kind == CircuitKind.NORM.value
    )
    if len(norms) != 1 or norms[0].order < last_layer_order:
        return ()
    selected = {value.node_id for value in interface} | {norms[0].node_id}
    return tuple(value for value in child.nodes if value.node_id in selected)


def _semantic_name(value: Any) -> str:
    text = str(value).strip().lower()
    return "_".join(
        part
        for part in "".join(
            character if character.isalnum() else " " for character in text
        ).split()
        if part
    )


def _node_attributes(node: CircuitNode) -> dict[str, JsonScalar]:
    return dict(node.attributes)


def _nodes_declare_semantic(
    nodes: Sequence[CircuitNode],
    semantic: str,
) -> bool:
    wanted = _semantic_name(semantic)
    for node in nodes:
        if _semantic_name(node.kind) == wanted:
            return True
        for key, value in node.attributes:
            key_name = _semantic_name(key)
            value_name = _semantic_name(value)
            if key_name == wanted and value not in {False, None, ""}:
                return True
            if key_name in {
                "circuit_kind",
                "coupled_role",
                "coupled_roles",
                "semantic_kind",
                "semantic_kinds",
                "semantic_role",
                "semantic_roles",
            } and (
                value_name == wanted
                or wanted in value_name.split("_")
                or wanted in value_name
            ):
                return True
        if any(
            wanted == _semantic_name(parameter.role)
            or wanted in _semantic_name(parameter.role)
            for parameter in node.parameters
        ):
            return True
    return False


def _complete_layer_bundle(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
    *,
    minimum_layers: int = 1,
) -> bool:
    indices = {node.layer_index for node in nodes if node.layer_index is not None}
    if len(indices) < minimum_layers or any(node.layer_index is None for node in nodes):
        return False
    expected = {node.node_id for index in indices for node in graph.layer_nodes(index)}
    return expected == {node.node_id for node in nodes}


def _complete_model_bundle(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    return {node.node_id for node in nodes} == {node.node_id for node in graph.nodes}


def _has_edge(
    graph: ModelCircuitGraph,
    sources: Sequence[CircuitNode],
    targets: Sequence[CircuitNode],
) -> bool:
    source_ids = {node.node_id for node in sources}
    target_ids = {node.node_id for node in targets}
    return any(
        edge.source_node in source_ids and edge.target_node in target_ids
        for edge in graph.edges
    )


def _ordered_residual_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
    *,
    require_attention: bool = False,
) -> bool:
    operator_kinds = (
        {CircuitKind.ATTENTION.value}
        if require_attention
        else {
            CircuitKind.ATTENTION.value,
            CircuitKind.SHORT_CONV.value,
            CircuitKind.RECURRENT_MIXER.value,
        }
    )
    operators = tuple(node for node in nodes if node.kind in operator_kinds)
    mlps = tuple(
        node
        for node in nodes
        if node.kind
        in {
            CircuitKind.GATED_MLP.value,
            CircuitKind.DENSE_MLP.value,
        }
    )
    operator_residuals = tuple(
        node
        for node in nodes
        if node.kind == CircuitKind.RESIDUAL_ADD.value
        and _node_attributes(node).get("ordered_branch")
        in ({"attention"} if require_attention else {"attention", "operator"})
    )
    mlp_residuals = tuple(
        node
        for node in nodes
        if node.kind == CircuitKind.RESIDUAL_ADD.value
        and _node_attributes(node).get("ordered_branch") == "mlp"
    )
    if not operators or not mlps or not operator_residuals or not mlp_residuals:
        return False
    return (
        max(node.order for node in operators)
        < min(node.order for node in operator_residuals)
        < min(node.order for node in mlps)
        < min(node.order for node in mlp_residuals)
        and _has_edge(graph, operators, operator_residuals)
        and _has_edge(graph, mlps, mlp_residuals)
        and _has_edge(graph, operator_residuals, mlp_residuals)
    )


def _depth_source_selector_contract(node: CircuitNode) -> bool:
    """Prove one learned depth-source selector on a block-attnres trunk.

    A ``block_attnres`` branch reads a learned softmax mixture of checkpointed
    depth sources instead of the running residual, so its bundle carries a
    second ``RESIDUAL_ADD`` node that owns the pseudo-query and the depth-source
    normalization.  Admitting that node is not a relaxation of the branch
    contract: it is still exactly one normalization, one operator and one
    ordered residual write, plus the selector that names the branch input.
    """

    attributes = _node_attributes(node)
    count = attributes.get("depth_source_count")
    return bool(
        node.kind == CircuitKind.RESIDUAL_ADD.value
        and attributes.get("residual_mode") == "block_attnres"
        and attributes.get("learned_pseudo_query") is True
        and attributes.get("selection")
        == "learned_softmax_over_rms_normalized_depth_sources"
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    )


def _causal_branch_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
    *,
    branch: str,
) -> bool:
    """Prove one ordered pre-norm -> operator -> residual-write causal branch.

    This is the family-neutral unit the promoted causal kernels install: one
    input normalization, one complete coupled operator, and the one ordered
    residual write that operator owns, optionally preceded by the learned depth
    selector that chooses which residual the normalization reads.  Nothing here
    inspects a model family.

    The sequence-mixing branch admits one ``ATTENTION`` node *or* one
    ``RECURRENT_MIXER`` node.  That is not a relaxation: a bounded gated
    delta-rule recurrence is exactly one complete coupled operator making one
    ordered residual write, and demanding split Q/K/V/O of it would demand the
    softmax projections it does not have.  The two operator classes are proven
    by different contracts and select different executables.
    """

    nodes = tuple(nodes)
    norms = tuple(node for node in nodes if node.kind == CircuitKind.NORM.value)
    residuals = tuple(
        node for node in nodes if node.kind == CircuitKind.RESIDUAL_ADD.value
    )
    if branch == "attention":
        labels = _ATTENTION_BRANCH_LABELS
        attention_operators = tuple(
            node for node in nodes if node.kind == CircuitKind.ATTENTION.value
        )
        recurrent_operators = tuple(
            node for node in nodes if node.kind == CircuitKind.RECURRENT_MIXER.value
        )
        operators = attention_operators + recurrent_operators
        operator_proven = (
            (
                _projection_role_contract(
                    attention_operators,
                    require_q=True,
                    require_k=True,
                    require_v=True,
                    require_o=True,
                )
                # A compressed-latent-KV operator has no split K/V projection at
                # all: its keys and values are read out of one shared latent, so
                # it is proven by its own contract rather than by relaxing the
                # split-projection one.
                or _compressed_latent_kv_attention_contract(attention_operators)
            )
            if attention_operators
            else _bounded_gated_delta_recurrence_contract(graph, recurrent_operators)
        )
    elif branch == "mlp":
        labels = _MLP_BRANCH_LABELS
        operators = tuple(
            node for node in nodes if node.kind == CircuitKind.GATED_MLP.value
        )
        operator_proven = (
            _gated_mlp_contract(operators, require_swiglu=True)
            or _fused_gate_up_mlp_contract(operators)
            or _stable_latent_moe_channel_mixer_contract(operators)
        )
    else:
        return False
    writes = tuple(
        node
        for node in residuals
        if _node_attributes(node).get("ordered_branch") in labels
    )
    write_ids = {node.node_id for node in writes}
    selectors = tuple(node for node in residuals if node.node_id not in write_ids)
    if (
        len(nodes) != 3 + len(selectors)
        or len(norms) != 1
        or len(operators) != 1
        or len(writes) != 1
        or any(not _depth_source_selector_contract(node) for node in selectors)
    ):
        return False
    return bool(
        operator_proven
        and all(node.order < norms[0].order for node in selectors)
        and norms[0].order < operators[0].order < writes[0].order
        and _has_edge(graph, norms, operators)
        and _has_edge(graph, operators, writes)
    )


def _causal_layer_branch_split(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> tuple[tuple[CircuitNode, ...], tuple[CircuitNode, ...]] | None:
    """Split one complete layer bundle into its two ordered causal branches."""

    ordered = tuple(sorted(nodes, key=lambda value: value.order))
    boundaries = tuple(
        node
        for node in ordered
        if node.kind == CircuitKind.RESIDUAL_ADD.value
        and _node_attributes(node).get("ordered_branch") in _ATTENTION_BRANCH_LABELS
    )
    if not _complete_layer_bundle(graph, ordered) or len(boundaries) != 1:
        return None
    boundary = boundaries[0].order
    attention_branch = tuple(node for node in ordered if node.order <= boundary)
    mlp_branch = tuple(node for node in ordered if node.order > boundary)
    if not _causal_branch_contract(
        graph, attention_branch, branch="attention"
    ) or not _causal_branch_contract(graph, mlp_branch, branch="mlp"):
        return None
    return attention_branch, mlp_branch


def _causal_operator_attributes(
    nodes: Sequence[CircuitNode],
    kind: str,
) -> dict[str, JsonScalar]:
    operators = tuple(node for node in nodes if node.kind == kind)
    return _node_attributes(operators[0]) if len(operators) == 1 else {}


def _sequence_mixer_attributes(
    nodes: Sequence[CircuitNode],
) -> dict[str, JsonScalar]:
    """Return the attributes of the branch's one sequence-mixing operator.

    The branch contract has already proven there is exactly one, and that it is
    an attention node or a recurrent mixer; this reads whichever it is so the
    width and policy checks do not have to know first.
    """

    operators = tuple(
        node
        for node in nodes
        if node.kind
        in {CircuitKind.ATTENTION.value, CircuitKind.RECURRENT_MIXER.value}
    )
    return _node_attributes(operators[0]) if len(operators) == 1 else {}


def _residual_widths(
    parent: ModelCircuitGraph,
    child: ModelCircuitGraph,
) -> tuple[int, int] | None:
    """Return both declared residual widths, or ``None`` if either is unknown."""

    source = dict(parent.metadata).get("hidden_size")
    target = dict(child.metadata).get("hidden_size")
    if (
        isinstance(source, bool)
        or not isinstance(source, int)
        or isinstance(target, bool)
        or not isinstance(target, int)
        or source <= 0
        or target <= 0
    ):
        return None
    return source, target


def _cross_width_boundary_chart_crossing(
    parent: ModelCircuitGraph,
    child: ModelCircuitGraph,
) -> bool:
    """Prove one contraction-only residual-width crossing charts can cover.

    The promoted branch kernels are width-agnostic, so the only thing a width
    change breaks is the coordinate system every residual write lands in.  A
    sealed per-boundary chart set repairs exactly that, so the planner may emit
    the same branch methods for this crossing — flagged
    ``boundary_chart_required`` — and the operators fail closed unless an
    authenticated chart set is supplied.  Width *expansion* is out of the v1
    envelope: the balanced basis is a contraction construction.
    """

    widths = _residual_widths(parent, child)
    if widths is None:
        return False
    source, target = widths
    return bool(
        target < source
        and _causal_layer_position_transport(
            dict(parent.metadata),
            dict(child.metadata),
        )
        is not None
    )


def _causal_layer_position_transport(
    parent_metadata: Mapping[str, Any],
    child_metadata: Mapping[str, Any],
) -> str | None:
    """Name the position transport both graphs can honestly share, or refuse.

    ``rope`` is the promoted lane: two ``hf-causal-decoder`` graphs that both
    rotate, where the kernel transports queries and keys through a real
    rotation.  ``nope`` is the second lane: two graphs from the *same* native
    decoder inspector that both declare they carry order without any rotary
    encoding, where the RoPE transport degenerates to the identity.  A mixed
    pair has no honest transport at all -- there is no evidence that says what
    a rotated parent's key would have been at an unrotated child's position --
    so it keeps the planner's explicit operator gap.
    """

    parent_inspector = parent_metadata.get("inspector")
    child_inspector = child_metadata.get("inspector")
    parent_policy = parent_metadata.get("position_policy")
    child_policy = child_metadata.get("position_policy")
    if (
        parent_inspector == _CAUSAL_DECODER_INSPECTOR
        and child_inspector == _CAUSAL_DECODER_INSPECTOR
        and parent_policy == "rope"
        and child_policy == "rope"
    ):
        return "rope"
    if (
        parent_inspector == child_inspector
        and parent_inspector in _NATIVE_DECODER_INSPECTORS
        and parent_policy == "nope"
        and child_policy == "nope"
    ):
        return "nope"
    return None


def _declares_rope_requirement(attributes: Mapping[str, JsonScalar]) -> bool:
    """True when a node needs something only a rotary implementation provides."""

    if any(
        attributes.get(name) not in (None, "", False)
        for name in _ROPE_SPECIFIC_ATTRIBUTES
    ):
        return True
    for name in ("position_policy", "position_encoding"):
        value = attributes.get(name)
        if isinstance(value, str) and _semantic_name(value) not in {"nope", ""}:
            return True
    return False


def _windowed_nope_attention_contract(nodes: Sequence[CircuitNode]) -> bool:
    """Prove one windowed NoPE grouped-query attention operator.

    The design contract is ``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md``
    section 1.3: local attention is the deployed GQA operator class minus the
    rotation plus a band mask, so the same coupled kernel runs it under two
    policy fields.  Everything that makes that true is checked here -- split
    Q/K/V/O, no projection bias, an even head dimension the kernel can pair,
    a declared NoPE policy on the node itself, and no rotary requirement
    anywhere -- and anything else is refused rather than approximated.
    """

    operators = tuple(
        node for node in nodes if node.kind == CircuitKind.ATTENTION.value
    )
    if len(operators) != 1:
        return False
    attributes = _node_attributes(operators[0])
    window = attributes.get("sliding_window")
    head_dim = attributes.get("head_dim")
    return bool(
        attributes.get("position_policy") == "nope"
        and not _declares_rope_requirement(attributes)
        and attributes.get("projection_style") == "split_qkv"
        and attributes.get("projection_bias") is False
        and attributes.get("causal") is True
        and attributes.get("fold_kernel_family_required")
        == "causal_sliding_window_split_qkv_attention"
        and isinstance(head_dim, int)
        and not isinstance(head_dim, bool)
        and head_dim % 2 == 0
        and (
            window is None
            or (
                isinstance(window, int)
                and not isinstance(window, bool)
                and window > 0
            )
        )
        and _projection_role_contract(
            operators,
            require_q=True,
            require_k=True,
            require_v=True,
            require_o=True,
        )
    )


def _positive_attribute(attributes: Mapping[str, JsonScalar], name: str) -> bool:
    value = attributes.get(name)
    return bool(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


def _compressed_summary_attention_contract(nodes: Sequence[CircuitNode]) -> bool:
    """Prove one composite local-window + compressed-summary attention operator.

    The design contract is
    ``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 1.2: ``csa``
    and ``hca`` are one ``SparseLatentAttention`` module that computes a
    windowed grouped-query sub-branch and a strided full-width summary
    sub-branch from the *same* normalized rows and writes their sum into the
    residual stream exactly once.  This contract therefore demands a single
    composite node that says so -- two sub-branches, one residual write -- and
    refuses a node that still describes the pair as two independent operators.
    Both the window and the stride are contract constants of the crossing; the
    equality check between parent and child lives in the layer contract, and
    the kernel refuses a change to either as a kind translation.
    """

    operators = tuple(
        node for node in nodes if node.kind == CircuitKind.ATTENTION.value
    )
    if len(operators) != 1:
        return False
    attributes = _node_attributes(operators[0])
    window = attributes.get("local_window")
    return bool(
        attributes.get("position_policy") == "nope"
        and not _declares_rope_requirement(attributes)
        and attributes.get("composite_operator") is True
        and attributes.get("composite_residual_writes") == 1
        and attributes.get("composite_sub_branches")
        == "local_window,compressed_summary"
        and attributes.get("sub_branch_combination")
        == "sum_of_local_window_and_compressed_summary"
        and attributes.get("summary_reduction")
        == "mean_over_stride_with_leading_zero_summary"
        and attributes.get("summary_projection_style") == "split_qkv_full_width"
        and attributes.get("projection_style")
        == "split_qkv_local_window_plus_full_width_compressed_summary"
        and attributes.get("projection_bias") is False
        and attributes.get("causal") is True
        and attributes.get("fold_kernel_family_required")
        == "compressed_summary_attention"
        and attributes.get("mixer_family") in {"csa", "hca"}
        and _positive_attribute(attributes, "summary_stride")
        and _positive_attribute(attributes, "head_dim")
        and _positive_attribute(attributes, "query_heads")
        and _positive_attribute(attributes, "key_value_heads")
        # The local sub-branch really is banded; an unbanded summary attention
        # is a different operator and the kernel has no envelope for it.
        and isinstance(window, int)
        and not isinstance(window, bool)
        and window > 0
        and _projection_role_contract(
            operators,
            require_q=True,
            require_k=True,
            require_v=True,
            require_o=True,
        )
    )


def _bounded_gated_delta_recurrence_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    """Prove one bounded gated delta-rule recurrence operator.

    The design contract is
    ``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` sections 1.1 and 4:
    a KDA branch is one ``RECURRENT_MIXER`` node carrying an explicit
    ``[batch, heads, key_dim, value_dim]`` state across a recurrent ``STATE``
    edge, with the three internal depthwise short convolutions, the low-rank
    decay map and the headwise output norm as *internal structure* of that one
    operator rather than as separate circuits.  Every attribute the kernel
    reads is required here by name, so a node that does not declare its
    geometry keeps the planner's gap instead of reaching a kernel that would
    then have to guess it.
    """

    operators = tuple(
        node for node in nodes if node.kind == CircuitKind.RECURRENT_MIXER.value
    )
    if len(operators) != 1:
        return False
    attributes = _node_attributes(operators[0])
    lower_bound = attributes.get("gate_lower_bound")
    roles = _parameter_roles(operators, {CircuitKind.RECURRENT_MIXER.value})
    return bool(
        attributes.get("position_policy") == "nope"
        and not _declares_rope_requirement(attributes)
        and attributes.get("operator_type") == "kimi_delta_attention"
        and attributes.get("recurrence") == "bounded_gated_delta_rule"
        and attributes.get("softmax_free") is True
        and attributes.get("not_attention_qkv") is True
        and attributes.get("internal_headwise_output_norm") is True
        and attributes.get("state_layout") == "batch_heads_key_value"
        and attributes.get("internal_short_conv_operators") == 3
        and attributes.get("internal_short_conv_paths") == "q_conv,k_conv,v_conv"
        and attributes.get("fold_kernel_family_required")
        == "bounded_gated_delta_recurrence"
        and all(
            _positive_attribute(attributes, name)
            for name in (
                "heads",
                "key_head_dim",
                "value_head_dim",
                "decay_rank",
                "state_elements",
                "internal_short_conv_kernel_size",
            )
        )
        # The retention floor is ``exp(gate_lower_bound)``; a non-negative bound
        # would put the floor at or above one, which is not a decay at all.
        and isinstance(lower_bound, (int, float))
        and not isinstance(lower_bound, bool)
        and float(lower_bound) < 0.0
        and _roles_contain(roles, ("q_proj",))
        and _roles_contain(roles, ("k_proj",))
        and _roles_contain(roles, ("v_proj",))
        and _roles_contain(roles, ("out_weight",))
        and _roles_contain(roles, ("decay_down",))
        and _roles_contain(roles, ("decay_up",))
        and _roles_contain(roles, ("decay_bias",))
        and _roles_contain(roles, ("decay_log_scale",))
        and _roles_contain(roles, ("write_strength",))
        and _roles_contain(roles, ("output_gate",))
        and _roles_contain(roles, ("output_norm",))
        and _roles_contain(roles, ("q_conv",))
        and _roles_contain(roles, ("k_conv",))
        and _roles_contain(roles, ("v_conv",))
        and _stateful_operator_contract(
            graph,
            operators,
            operator_kind=CircuitKind.RECURRENT_MIXER.value,
            port_kind=PortKind.RECURRENT_STATE.value,
            edge_kind=EdgeKind.STATE.value,
        )
    )


def _compressed_latent_kv_attention_contract(nodes: Sequence[CircuitNode]) -> bool:
    """Prove one gated compressed-latent-KV attention operator.

    The design contract is
    ``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`` section 1: a
    ``gated_mla`` branch is one ``ATTENTION`` node whose keys and values are
    read out of one shared low-rank latent, whose queries are optionally read
    out of a second one, and whose context is gated channel-wise before a
    single writer.  It has no split K/V projection, so it is proven here rather
    than by relaxing ``_projection_role_contract``.

    ``extra_head_dim`` is admitted at zero: the shared key block is optional in
    the module (``layers.py:290-296``) and its *width* is a contract constant
    checked between the two sides, not a positivity requirement.  Everything
    else the kernel reads is required by name, so a node that does not declare
    its geometry keeps the planner's gap instead of reaching a kernel that
    would then have to guess it.
    """

    operators = tuple(
        node for node in nodes if node.kind == CircuitKind.ATTENTION.value
    )
    if len(operators) != 1:
        return False
    attributes = _node_attributes(operators[0])
    extra = attributes.get("extra_head_dim")
    query_latent = attributes.get("query_latent_size")
    roles = _parameter_roles(operators, {CircuitKind.ATTENTION.value})
    return bool(
        attributes.get("position_policy") == "nope"
        and not _declares_rope_requirement(attributes)
        and attributes.get("projection_style") == "compressed_latent_kv"
        and attributes.get("operator_type") == "gated_multi_head_latent_attention"
        and attributes.get("layer_type") == "full_attention"
        and attributes.get("sliding_window") in (None, 0)
        and attributes.get("causal") is True
        and attributes.get("projection_bias") is False
        and attributes.get("full_rank_output_gate") is True
        and attributes.get("fold_kernel_family_required")
        == "compressed_latent_attention"
        and all(
            _positive_attribute(attributes, name)
            for name in (
                "query_heads",
                "key_value_heads",
                "head_dim",
                "key_head_dim",
                "value_head_dim",
                "kv_latent_size",
            )
        )
        and isinstance(extra, int)
        and not isinstance(extra, bool)
        and extra >= 0
        and attributes.get("head_dim")
        == int(attributes.get("key_head_dim", 0)) + extra
        and isinstance(query_latent, int)
        and not isinstance(query_latent, bool)
        and query_latent >= 0
        and _roles_contain(roles, ("kv_down",))
        and _roles_contain(roles, ("kv_norm",))
        and _roles_contain(roles, ("kv_up",))
        and _roles_contain(roles, ("gate",))
        and _roles_contain(roles, ("out_weight",))
        and (
            _roles_contain(roles, ("q_down",))
            and _roles_contain(roles, ("q_norm",))
            and _roles_contain(roles, ("q_up",))
            if query_latent
            else _roles_contain(roles, ("q_proj",))
        )
    )


def _stable_latent_moe_channel_mixer_contract(nodes: Sequence[CircuitNode]) -> bool:
    """Prove one composite Stable LatentMoE channel mixer.

    The design contract is
    ``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`` section 2: a
    top-``k`` routed set of latent experts and one or more full-width shared
    experts computed from the *same* normalized rows, summed into the residual
    stream exactly once.  This contract therefore demands a single composite
    node that says so, and refuses a node that still describes the pair as
    independent operators joined by residual adds the module never performs.
    """

    mlps = tuple(node for node in nodes if node.kind == CircuitKind.GATED_MLP.value)
    if len(mlps) != 1:
        return False
    attributes = _node_attributes(mlps[0])
    roles = _parameter_roles(mlps, {CircuitKind.GATED_MLP.value})
    return bool(
        attributes.get("channel_mixer") == "stable_latent_moe"
        and attributes.get("composite_operator") is True
        and attributes.get("composite_residual_writes") == 1
        and attributes.get("composite_sub_branches")
        == "top_k_routed_latent_experts,full_width_shared_experts"
        and attributes.get("sub_branch_combination")
        == "sum_of_routed_latent_projection_and_full_width_shared_experts"
        and attributes.get("router_score_activation") == "sigmoid"
        and attributes.get("routed_weight_normalization")
        == "selected_score_sum_clamped"
        and attributes.get("router_bias_is_non_gradient_buffer") is True
        and attributes.get("fused_gate_up_projection") is True
        and attributes.get("gate_up_split") == "chunk_two_on_last_dimension"
        and attributes.get("projection_layout") == "fused_gate_up_then_down"
        and attributes.get("projection_bias") is False
        and attributes.get("fold_kernel_family_required")
        == "stable_latent_moe_channel_mixer"
        and _semantic_name(attributes.get("activation", ""))
        in {"silu", "swish", "situ"}
        and all(
            _positive_attribute(attributes, name)
            for name in (
                "expert_count",
                "experts_per_token",
                "shared_expert_count",
                "expert_latent_size",
                "routed_intermediate_size",
                "intermediate_size",
            )
        )
        and int(attributes.get("experts_per_token", 0))
        <= int(attributes.get("expert_count", 0))
        and _roles_contain(roles, ("router_weight",))
        and _roles_contain(roles, ("latent_down",))
        and _roles_contain(roles, ("latent_up",))
        and _roles_contain(roles, ("routed_norm",))
        and _roles_contain(roles, ("routed_experts",))
        and _roles_contain(roles, ("shared_experts",))
    )


def _fused_gate_up_mlp_contract(nodes: Sequence[CircuitNode]) -> bool:
    """Prove one dense SiLU-gated MLP whose gate and up share one tensor.

    ``CausalGatedMLPBranchOperator`` matches ``gate_proj``/``up_proj``/
    ``down_proj``; a fused ``[2 * intermediate, hidden]`` layout is a different
    tensor set for the same function, which is why the coverage document
    records it as a mismatch rather than a near miss.  The split point is a
    declared attribute, not an assumption: only a node that says it chunks the
    fused output in two on the last dimension qualifies.
    """

    mlps = tuple(node for node in nodes if node.kind == CircuitKind.GATED_MLP.value)
    if len(mlps) != 1:
        return False
    roles = _parameter_roles(mlps, {CircuitKind.GATED_MLP.value})
    attributes = _node_attributes(mlps[0])
    intermediate = attributes.get("intermediate_size")
    return bool(
        _roles_contain(roles, ("up_weight",))
        and _roles_contain(roles, ("down_weight",))
        and not _roles_contain(roles, ("gate_proj", "gate_weight"))
        and attributes.get("fused_gate_up_projection") is True
        and attributes.get("gate_up_split") == "chunk_two_on_last_dimension"
        and attributes.get("projection_layout") == "fused_gate_up_then_down"
        and attributes.get("channel_mixer") == "dense"
        and attributes.get("projection_bias") is False
        and _semantic_name(attributes.get("activation", "")) in {"silu", "swish"}
        and isinstance(intermediate, int)
        and not isinstance(intermediate, bool)
        and intermediate > 0
    )


#: The declared operator widths each native sequence-mixer family may contract.
#: Every one of them is a *coordinate* the registered kernel can cut; anything
#: not listed here (a window, a stride, a short-conv kernel size, a gate floor)
#: is a contract constant checked for equality instead.
_NATIVE_OPERATOR_CONTRACTION_WIDTHS: Mapping[str, tuple[str, ...]] = {
    "causal_sliding_window_split_qkv_attention": (
        "query_heads",
        "key_value_heads",
        "head_dim",
    ),
    "compressed_summary_attention": (
        "query_heads",
        "key_value_heads",
        "head_dim",
    ),
    "bounded_gated_delta_recurrence": (
        "heads",
        "key_head_dim",
        "value_head_dim",
        "decay_rank",
    ),
    # ``query_latent_size`` is deliberately absent: it may legitimately be zero
    # on both sides (the dense-query variant), which the shared positive-width
    # loop cannot express, so it is checked in the family arm below.
    "compressed_latent_attention": (
        "query_heads",
        "key_head_dim",
        "value_head_dim",
        "kv_latent_size",
    ),
}
#: Attributes that must be *equal* on both sides of a native sequence-mixer
#: crossing, per family.  Each one changes which inputs the operator can see or
#: what its own arithmetic is, so changing it is a kind translation and not a
#: fold coordinate (design section 3).
_NATIVE_OPERATOR_CONTRACT_CONSTANTS: Mapping[str, tuple[str, ...]] = {
    "causal_sliding_window_split_qkv_attention": ("sliding_window",),
    "compressed_summary_attention": (
        "local_window",
        "summary_stride",
        "mixer_family",
    ),
    "bounded_gated_delta_recurrence": (
        "internal_short_conv_kernel_size",
        "gate_lower_bound",
    ),
    "compressed_latent_attention": (
        "extra_head_dim",
        "layer_type",
        "sliding_window",
    ),
}
#: The same two tables for the *channel*-mixing branch.  ``intermediate_size``
#: is already checked for every family by the shared width rule, so only the
#: widths a channel-mixer family adds appear here.
_NATIVE_CHANNEL_MIXER_CONTRACTION_WIDTHS: Mapping[str, tuple[str, ...]] = {
    "stable_latent_moe_channel_mixer": (
        "expert_latent_size",
        "routed_intermediate_size",
    ),
}
_NATIVE_CHANNEL_MIXER_CONTRACT_CONSTANTS: Mapping[str, tuple[str, ...]] = {
    "stable_latent_moe_channel_mixer": (
        "expert_count",
        "experts_per_token",
        "shared_expert_count",
        "activation",
        "situ_gate_beta",
        "situ_up_beta",
        "router_score_activation",
        "routed_weight_normalization",
    ),
}


def _native_sequence_mixer_branch(
    source_nodes: Sequence[CircuitNode],
    target_nodes: Sequence[CircuitNode],
    source_attributes: Mapping[str, JsonScalar],
    target_attributes: Mapping[str, JsonScalar],
) -> tuple[str, str] | None:
    """Name the branch method one NoPE sequence-mixer crossing may install.

    Four families, four separately proven contracts, four method ids -- and a
    fifth outcome, ``None``, for everything else.  ``csa`` and ``hca`` share one
    method id because they share one stride-parameterized executable; the
    stride itself is a contract constant checked below, so the two strides can
    never be folded into each other by accident.  The two gated-MLA query
    variants share one for the same reason, and their shape agreement is
    checked here because a zero latent size cannot travel through the shared
    positive-width rule.
    """

    family = source_attributes.get("fold_kernel_family_required")
    contracts = {
        "causal_sliding_window_split_qkv_attention": (
            _windowed_nope_attention_contract,
            CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
            "cached_parent_endpoint_coupled_windowed_nope_gqa_qk_softmax_vo",
        ),
        "compressed_summary_attention": (
            _compressed_summary_attention_contract,
            CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
            (
                "cached_parent_endpoint_coupled_local_window_plus_compressed_"
                "summary_sub_branch_write"
            ),
        ),
        "compressed_latent_attention": (
            _compressed_latent_kv_attention_contract,
            CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
            (
                "cached_parent_endpoint_coupled_compressed_latent_kv_softmax_"
                "gated_write"
            ),
        ),
    }
    if family in contracts:
        contract, method_id, fit_objective = contracts[family]
        if not contract(source_nodes) or not contract(target_nodes):
            return None
        if family == "compressed_latent_attention":
            source_latent = int(source_attributes.get("query_latent_size", 0))
            target_latent = int(target_attributes.get("query_latent_size", 0))
            if bool(source_latent) != bool(target_latent) or (
                target_latent > source_latent
            ):
                # Both sides compress the query or neither does; when they do,
                # the latent is a contraction coordinate like any other width.
                return None
    elif family == "bounded_gated_delta_recurrence":
        method_id = CAUSAL_KDA_RECURRENT_BRANCH_METHOD
        fit_objective = (
            "cached_parent_endpoint_coupled_bounded_gated_delta_recurrence_write"
        )
        # The branch contract already proved this arm through
        # ``_bounded_gated_delta_recurrence_contract`` on both sides.
    else:
        return None
    for name in _NATIVE_OPERATOR_CONTRACT_CONSTANTS[family]:
        if source_attributes.get(name) != target_attributes.get(name):
            # A band, a stride, a short-conv kernel or a gate floor is a
            # contract constant: no evidence anywhere says what the parent
            # would have computed under a different one.
            return None
    return method_id, fit_objective


def _native_channel_mixer_branch(
    source_nodes: Sequence[CircuitNode],
    target_nodes: Sequence[CircuitNode],
    source_attributes: Mapping[str, JsonScalar],
    target_attributes: Mapping[str, JsonScalar],
) -> tuple[str, str] | None:
    """Name the branch method one composite channel-mixer crossing may install.

    The dense fused-gate/up and split-SwiGLU cases are decided by the caller;
    this arm exists for channel mixers that are *composite* -- one operator
    whose internal structure is a routed sub-branch plus a shared one -- and it
    returns ``None`` for everything else so the crossing keeps its gap.
    """

    family = source_attributes.get("fold_kernel_family_required")
    if family != "stable_latent_moe_channel_mixer":
        return None
    if not _stable_latent_moe_channel_mixer_contract(
        source_nodes
    ) or not _stable_latent_moe_channel_mixer_contract(target_nodes):
        return None
    for name in _NATIVE_CHANNEL_MIXER_CONTRACT_CONSTANTS[family]:
        if source_attributes.get(name) != target_attributes.get(name):
            # An expert count, an experts-per-token, a shared-expert count or
            # an activation is a contract constant: no evidence anywhere says
            # which rows a different router would have produced, or what a
            # different nonlinearity would have computed on them.
            return None
    return (
        CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
        (
            "cached_parent_endpoint_coupled_top_k_routed_latent_experts_plus_"
            "shared_expert_write"
        ),
    )


def _family_neutral_causal_layer_contract(
    parent: ModelCircuitGraph,
    source_nodes: Sequence[CircuitNode],
    child: ModelCircuitGraph,
    target_nodes: Sequence[CircuitNode],
) -> tuple[
    tuple[str, tuple[CircuitNode, ...], tuple[CircuitNode, ...], str],
    ...,
] | None:
    """Prove one contraction-only causal layer transfer without family equality.

    Both sides must be inspectable causal decoders whose layer decomposes into
    the two ordered causal branches, share one position transport, share the
    residual coordinate width the promoted kernels fit in (or an authenticated
    chart crossing), and describe a contraction rather than an expansion.
    Anything else returns ``None`` so the planner keeps its explicit operator
    gap.

    Two transports are proven, and they select *different executables*: the
    RoPE lane installs the coupled GQA/RoPE kernel, the NoPE lane installs the
    windowed-NoPE policy variant whose rotation is the identity.  A relaxation
    that let a NoPE graph reach the RoPE operator would plan a transaction the
    operator could not run, which is the failure this contract exists to
    prevent.
    """

    source = _causal_layer_branch_split(parent, source_nodes)
    target = _causal_layer_branch_split(child, target_nodes)
    if source is None or target is None:
        return None
    parent_metadata = dict(parent.metadata)
    child_metadata = dict(child.metadata)
    transport = _causal_layer_position_transport(parent_metadata, child_metadata)
    if transport is None:
        return None
    hidden = parent_metadata.get("hidden_size")
    if isinstance(hidden, bool) or not isinstance(hidden, int):
        return None
    if child_metadata.get("hidden_size") != hidden and not (
        _cross_width_boundary_chart_crossing(parent, child)
    ):
        # The promoted kernels fit in one shared residual coordinate system.
        # A residual-width crossing is executable only when a sealed
        # per-boundary chart set re-coordinates every residual write; anything
        # else keeps the planner's explicit operator gap.
        return None
    source_attention = _sequence_mixer_attributes(source[0])
    target_attention = _sequence_mixer_attributes(target[0])
    source_mlp = _causal_operator_attributes(source[1], CircuitKind.GATED_MLP.value)
    target_mlp = _causal_operator_attributes(target[1], CircuitKind.GATED_MLP.value)
    if transport == "rope":
        operator_widths: tuple[str, ...] = ("query_heads", "key_value_heads", "head_dim")
    else:
        operator_widths = _NATIVE_OPERATOR_CONTRACTION_WIDTHS.get(
            source_attention.get("fold_kernel_family_required"), ()
        )
        if not operator_widths or source_attention.get(
            "fold_kernel_family_required"
        ) != target_attention.get("fold_kernel_family_required"):
            # Either the family has no registered branch operator at all, or
            # the two sides declare different families -- and no registered
            # operator crosses between two sequence-mixer classes in one
            # transaction, so the crossing keeps its gap.
            return None
    channel_widths = _NATIVE_CHANNEL_MIXER_CONTRACTION_WIDTHS.get(
        source_mlp.get("fold_kernel_family_required"), ()
    )
    widths = (
        tuple((name, source_attention, target_attention) for name in operator_widths)
        + (("intermediate_size", source_mlp, target_mlp),)
        + tuple((name, source_mlp, target_mlp) for name in channel_widths)
    )
    for name, source_values, target_values in widths:
        source_width = source_values.get(name)
        target_width = target_values.get(name)
        if (
            not isinstance(source_width, int)
            or isinstance(source_width, bool)
            or not isinstance(target_width, int)
            or isinstance(target_width, bool)
            or target_width <= 0
            or target_width > source_width
        ):
            return None
    if transport == "rope":
        if (
            source_attention.get("sliding_window") is not None
            or target_attention.get("sliding_window") is not None
        ):
            # The promoted RoPE kernel builds a full causal mask and a real
            # rotation; a declared window is a different operator class.
            return None
        attention_branch = (
            CAUSAL_ATTENTION_BRANCH_METHOD,
            "cached_parent_endpoint_coupled_gqa_rope_qk_softmax_vo",
        )
    else:
        attention_branch = _native_sequence_mixer_branch(
            source[0], target[0], source_attention, target_attention
        )
        if attention_branch is None:
            return None
    channel_branch = _native_channel_mixer_branch(
        source[1], target[1], source_mlp, target_mlp
    )
    source_fused = _fused_gate_up_mlp_contract(source[1])
    target_fused = _fused_gate_up_mlp_contract(target[1])
    if channel_branch is not None:
        # A composite channel mixer is decided before the dense cases: a MoE
        # parent against a dense child (or the reverse) fails one of the two
        # contracts, falls through, and keeps its gap, because no registered
        # operator crosses between two channel-mixer classes in one transaction.
        mlp_branch = channel_branch
    elif source_fused != target_fused:
        # One side stores gate/up fused and the other stores them split.  Both
        # layouts are foldable, but no registered operator crosses between them
        # in one transaction, so the crossing keeps its gap.
        return None
    elif source_fused:
        mlp_branch = (
            CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
            "cached_parent_endpoint_coupled_fused_gate_up_swiglu_residual_write",
        )
    elif _gated_mlp_contract(
        source[1], require_swiglu=True
    ) and _gated_mlp_contract(target[1], require_swiglu=True):
        mlp_branch = (
            CAUSAL_GATED_MLP_BRANCH_METHOD,
            "cached_parent_endpoint_coupled_swiglu_residual_write",
        )
    else:
        return None
    return (
        (attention_branch[0], source[0], target[0], attention_branch[1]),
        (mlp_branch[0], source[1], target[1], mlp_branch[1]),
    )


def _lfm2_layer_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    kinds = {node.kind for node in nodes}
    operator_count = sum(
        node.kind
        in {
            CircuitKind.ATTENTION.value,
            CircuitKind.SHORT_CONV.value,
        }
        for node in nodes
    )
    return (
        graph.family == "lfm2"
        and _complete_layer_bundle(graph, nodes)
        and operator_count == 1
        and CircuitKind.GATED_MLP.value in kinds
        and sum(node.kind == CircuitKind.RESIDUAL_ADD.value for node in nodes) >= 2
        and sum(node.kind == CircuitKind.NORM.value for node in nodes) >= 2
        and _ordered_residual_contract(graph, nodes)
    )


def _neo_single_neuron_swiglu_contract(
    source_graph: ModelCircuitGraph,
    source_nodes: Sequence[CircuitNode],
    target_graph: ModelCircuitGraph,
    target_nodes: Sequence[CircuitNode],
) -> bool:
    if (
        source_graph.family != "neo_chat"
        or target_graph.family != "neo_chat"
        or not _residual_layer_contract(source_graph, source_nodes)
        or not _residual_layer_contract(target_graph, target_nodes)
    ):
        return False
    source_by_id = {node.node_id: node for node in source_nodes}
    target_by_id = {node.node_id: node for node in target_nodes}
    if set(source_by_id) != set(target_by_id):
        return False
    source_mlps = tuple(
        node for node in source_nodes if node.kind == CircuitKind.GATED_MLP.value
    )
    target_mlps = tuple(
        node for node in target_nodes if node.kind == CircuitKind.GATED_MLP.value
    )
    if len(source_mlps) != 1 or len(target_mlps) != 1:
        return False
    source_intermediate = _node_attributes(source_mlps[0]).get("intermediate_size")
    target_intermediate = _node_attributes(target_mlps[0]).get("intermediate_size")
    if (
        not isinstance(source_intermediate, int)
        or not isinstance(target_intermediate, int)
        or source_intermediate != target_intermediate + 1
    ):
        return False
    for node_id, source in source_by_id.items():
        target = target_by_id[node_id]
        if (
            source.kind != target.kind
            or source.layer_index != target.layer_index
            or source.inputs != target.inputs
            or source.outputs != target.outputs
        ):
            return False
        if source.kind != CircuitKind.GATED_MLP.value:
            if not _same_parameter_contract(
                source_graph,
                (source,),
                target_graph,
                (target,),
            ):
                return False
            continue
        source_parameters = {value.role: value for value in source.parameters}
        target_parameters = {value.role: value for value in target.parameters}
        if set(source_parameters) != set(target_parameters):
            return False
        changed = 0
        for role, source_parameter in source_parameters.items():
            target_parameter = target_parameters[role]
            if (
                source_parameter.dtype != target_parameter.dtype
                or source_parameter.trainable != target_parameter.trainable
            ):
                return False
            source_shape = source_parameter.shape
            target_shape = target_parameter.shape
            if source_shape == target_shape:
                continue
            semantic = _semantic_name(role)
            if (
                ("gate_proj" in semantic or "up_proj" in semantic)
                and len(source_shape) == 2
                and len(target_shape) == 2
                and source_shape[0] == target_shape[0] + 1
                and source_shape[1] == target_shape[1]
            ) or (
                "down_proj" in semantic
                and len(source_shape) == 2
                and len(target_shape) == 2
                and source_shape[0] == target_shape[0]
                and source_shape[1] == target_shape[1] + 1
            ):
                changed += 1
                continue
            return False
        if changed != 3:
            return False
    return True


def _neo_single_neuron_model_contract(
    source: ModelCircuitGraph,
    target: ModelCircuitGraph,
) -> bool:
    if (
        source.family != "neo_chat"
        or target.family != "neo_chat"
        or source.layer_indices != target.layer_indices
        or not source.layer_indices
        or source.tokenizer_fingerprint != target.tokenizer_fingerprint
        or {interface.modality for interface in source.interfaces} != {"image", "text"}
        or {interface.modality for interface in target.interfaces} != {"image", "text"}
    ):
        return False
    if any(
        not _neo_single_neuron_swiglu_contract(
            source,
            source.layer_nodes(index),
            target,
            target.layer_nodes(index),
        )
        for index in source.layer_indices
    ):
        return False
    source_globals = {
        node.node_id: node for node in source.nodes if node.layer_index is None
    }
    target_globals = {
        node.node_id: node for node in target.nodes if node.layer_index is None
    }
    if set(source_globals) != set(target_globals):
        return False
    for node_id, source_node in source_globals.items():
        target_node = target_globals[node_id]
        if (
            source_node.kind != target_node.kind
            or source_node.inputs != target_node.inputs
            or source_node.outputs != target_node.outputs
            or source_node.attributes != target_node.attributes
        ):
            return False
        if source_node.parameters or target_node.parameters:
            if not _same_parameter_contract(
                source,
                (source_node,),
                target,
                (target_node,),
            ):
                return False
    return True


def _single_neuron_swiglu_layer_contract(
    source_graph: ModelCircuitGraph,
    source_nodes: Sequence[CircuitNode],
    target_graph: ModelCircuitGraph,
    target_nodes: Sequence[CircuitNode],
    *,
    family: str,
) -> bool:
    """Prove one layer differs only by one removed coupled SwiGLU neuron.

    This is the family-parameterized form of the validated NEO layer contract.
    Everything except the single gate/up/down width step must be identical:
    same node identities, same ports, same non-MLP parameter semantics.
    """

    if (
        source_graph.family != family
        or target_graph.family != family
        or not _residual_layer_contract(source_graph, source_nodes)
        or not _residual_layer_contract(target_graph, target_nodes)
    ):
        return False
    source_by_id = {node.node_id: node for node in source_nodes}
    target_by_id = {node.node_id: node for node in target_nodes}
    if set(source_by_id) != set(target_by_id):
        return False
    source_mlps = tuple(
        node for node in source_nodes if node.kind == CircuitKind.GATED_MLP.value
    )
    target_mlps = tuple(
        node for node in target_nodes if node.kind == CircuitKind.GATED_MLP.value
    )
    if len(source_mlps) != 1 or len(target_mlps) != 1:
        return False
    source_intermediate = _node_attributes(source_mlps[0]).get("intermediate_size")
    target_intermediate = _node_attributes(target_mlps[0]).get("intermediate_size")
    if (
        not isinstance(source_intermediate, int)
        or not isinstance(target_intermediate, int)
        or source_intermediate != target_intermediate + 1
    ):
        return False
    for node_id, source in source_by_id.items():
        target = target_by_id[node_id]
        if (
            source.kind != target.kind
            or source.layer_index != target.layer_index
            or source.inputs != target.inputs
            or source.outputs != target.outputs
        ):
            return False
        if source.kind != CircuitKind.GATED_MLP.value:
            if not _same_parameter_contract(
                source_graph,
                (source,),
                target_graph,
                (target,),
            ):
                return False
            continue
        source_parameters = {value.role: value for value in source.parameters}
        target_parameters = {value.role: value for value in target.parameters}
        if set(source_parameters) != set(target_parameters):
            return False
        changed = 0
        for role, source_parameter in source_parameters.items():
            target_parameter = target_parameters[role]
            if (
                source_parameter.dtype != target_parameter.dtype
                or source_parameter.trainable != target_parameter.trainable
            ):
                return False
            source_shape = source_parameter.shape
            target_shape = target_parameter.shape
            if source_shape == target_shape:
                continue
            semantic = _semantic_name(role)
            if (
                ("gate_proj" in semantic or "up_proj" in semantic)
                and len(source_shape) == 2
                and len(target_shape) == 2
                and source_shape[0] == target_shape[0] + 1
                and source_shape[1] == target_shape[1]
            ) or (
                "down_proj" in semantic
                and len(source_shape) == 2
                and len(target_shape) == 2
                and source_shape[0] == target_shape[0]
                and source_shape[1] == target_shape[1] + 1
            ):
                changed += 1
                continue
            return False
        if changed != 3:
            return False
    return True


def _native_audio_model_contract(
    source: ModelCircuitGraph,
    target: ModelCircuitGraph,
) -> bool:
    """Prove one native audio child differs only by one SwiGLU neuron per layer.

    The tokenizer-fingerprint equality is deliberately identical to the NEO
    contract: folding never touches vocabulary, so a lane that would add audio
    rows to the text vocabulary is rejected here rather than repaired later.
    """

    if (
        source.family != _NATIVE_AUDIO_FAMILY
        or target.family != _NATIVE_AUDIO_FAMILY
        or source.layer_indices != target.layer_indices
        or not source.layer_indices
        or source.tokenizer_fingerprint != target.tokenizer_fingerprint
        or {interface.modality for interface in source.interfaces}
        != _NATIVE_AUDIO_MODALITIES
        or {interface.modality for interface in target.interfaces}
        != _NATIVE_AUDIO_MODALITIES
    ):
        return False
    if any(
        not _single_neuron_swiglu_layer_contract(
            source,
            source.layer_nodes(index),
            target,
            target.layer_nodes(index),
            family=_NATIVE_AUDIO_FAMILY,
        )
        for index in source.layer_indices
    ):
        return False
    source_globals = {
        node.node_id: node for node in source.nodes if node.layer_index is None
    }
    target_globals = {
        node.node_id: node for node in target.nodes if node.layer_index is None
    }
    if set(source_globals) != set(target_globals):
        return False
    for node_id, source_node in source_globals.items():
        target_node = target_globals[node_id]
        if (
            source_node.kind != target_node.kind
            or source_node.inputs != target_node.inputs
            or source_node.outputs != target_node.outputs
            or source_node.attributes != target_node.attributes
        ):
            return False
        if source_node.parameters or target_node.parameters:
            if not _same_parameter_contract(
                source,
                (source_node,),
                target,
                (target_node,),
            ):
                return False
    return True


def _native_video_model_contract(
    source: ModelCircuitGraph,
    target: ModelCircuitGraph,
) -> bool:
    """Prove one native video child differs only by one SwiGLU neuron per layer.

    Structurally identical to :func:`_native_audio_model_contract`; only the
    family and the required interface modality set differ.  The
    tokenizer-fingerprint equality is deliberately unchanged: folding never
    touches vocabulary, so a lane that would add video rows to the text
    vocabulary is rejected here rather than repaired later.
    """

    if (
        source.family != _NATIVE_VIDEO_FAMILY
        or target.family != _NATIVE_VIDEO_FAMILY
        or source.layer_indices != target.layer_indices
        or not source.layer_indices
        or source.tokenizer_fingerprint != target.tokenizer_fingerprint
        or {interface.modality for interface in source.interfaces}
        != _NATIVE_VIDEO_MODALITIES
        or {interface.modality for interface in target.interfaces}
        != _NATIVE_VIDEO_MODALITIES
    ):
        return False
    if any(
        not _single_neuron_swiglu_layer_contract(
            source,
            source.layer_nodes(index),
            target,
            target.layer_nodes(index),
            family=_NATIVE_VIDEO_FAMILY,
        )
        for index in source.layer_indices
    ):
        return False
    source_globals = {
        node.node_id: node for node in source.nodes if node.layer_index is None
    }
    target_globals = {
        node.node_id: node for node in target.nodes if node.layer_index is None
    }
    if set(source_globals) != set(target_globals):
        return False
    for node_id, source_node in source_globals.items():
        target_node = target_globals[node_id]
        if (
            source_node.kind != target_node.kind
            or source_node.inputs != target_node.inputs
            or source_node.outputs != target_node.outputs
            or source_node.attributes != target_node.attributes
        ):
            return False
        if source_node.parameters or target_node.parameters:
            if not _same_parameter_contract(
                source,
                (source_node,),
                target,
                (target_node,),
            ):
                return False
    return True


def _native_points3d_model_contract(
    source: ModelCircuitGraph,
    target: ModelCircuitGraph,
) -> bool:
    """Prove one native points3d child differs only by one SwiGLU neuron per layer.

    Structurally identical to :func:`_native_video_model_contract`; only the
    family and the required interface modality set differ, plus one extra
    check the earlier lanes could not need.  A points3d model may own an
    emitted-centroid ``MODALITY_OUTPUT`` interface, and folding must never
    change whether a child emits media or which media it emits, so the full
    ``(modality, direction, representation)`` interface identity is compared
    rather than the modality set alone.  The tokenizer-fingerprint equality is
    deliberately unchanged: folding never touches vocabulary, so a lane that
    would add geometry rows to the text vocabulary is rejected here rather than
    repaired later.
    """

    if (
        source.family != _NATIVE_POINTS3D_FAMILY
        or target.family != _NATIVE_POINTS3D_FAMILY
        or source.layer_indices != target.layer_indices
        or not source.layer_indices
        or source.tokenizer_fingerprint != target.tokenizer_fingerprint
        or {interface.modality for interface in source.interfaces}
        != _NATIVE_POINTS3D_MODALITIES
        or {interface.modality for interface in target.interfaces}
        != _NATIVE_POINTS3D_MODALITIES
        or source.interfaces != target.interfaces
    ):
        return False
    if any(
        not _single_neuron_swiglu_layer_contract(
            source,
            source.layer_nodes(index),
            target,
            target.layer_nodes(index),
            family=_NATIVE_POINTS3D_FAMILY,
        )
        for index in source.layer_indices
    ):
        return False
    source_globals = {
        node.node_id: node for node in source.nodes if node.layer_index is None
    }
    target_globals = {
        node.node_id: node for node in target.nodes if node.layer_index is None
    }
    if set(source_globals) != set(target_globals):
        return False
    for node_id, source_node in source_globals.items():
        target_node = target_globals[node_id]
        if (
            source_node.kind != target_node.kind
            or source_node.inputs != target_node.inputs
            or source_node.outputs != target_node.outputs
            or source_node.attributes != target_node.attributes
        ):
            return False
        if source_node.parameters or target_node.parameters:
            if not _same_parameter_contract(
                source,
                (source_node,),
                target,
                (target_node,),
            ):
                return False
    return True


def _qwen3_compensated_model_contract(
    source: ModelCircuitGraph,
    target: ModelCircuitGraph,
) -> bool:
    """Prove one full-depth Qwen3 differs only by coupled MLP row removal."""

    if (
        source.family != "qwen3"
        or target.family != "qwen3"
        or source.layer_indices != target.layer_indices
        or not source.layer_indices
        or source.tokenizer_fingerprint is None
        or source.tokenizer_fingerprint != target.tokenizer_fingerprint
        or dict(target.metadata).get("compensated_ragged") is not True
        or source.interfaces != target.interfaces
    ):
        return False
    source_nodes = {node.node_id: node for node in source.nodes}
    target_nodes = {node.node_id: node for node in target.nodes}
    if set(source_nodes) != set(target_nodes):
        return False
    changed_layers: list[tuple[int, int, int]] = []
    for node_id, source_node in source_nodes.items():
        target_node = target_nodes[node_id]
        if (
            source_node.kind != target_node.kind
            or source_node.order != target_node.order
            or source_node.path != target_node.path
            or source_node.layer_index != target_node.layer_index
            or source_node.inputs != target_node.inputs
            or source_node.outputs != target_node.outputs
        ):
            return False
        source_attributes = dict(source_node.attributes)
        target_attributes = dict(target_node.attributes)
        source_parameters = {
            parameter.role: parameter for parameter in source_node.parameters
        }
        target_parameters = {
            parameter.role: parameter for parameter in target_node.parameters
        }
        if set(source_parameters) != set(target_parameters):
            return False
        if source_node.kind != CircuitKind.GATED_MLP.value:
            if source_attributes != target_attributes or any(
                source_parameters[role].shape != target_parameters[role].shape
                or source_parameters[role].dtype != target_parameters[role].dtype
                or source_parameters[role].trainable
                != target_parameters[role].trainable
                for role in source_parameters
            ):
                return False
            continue
        source_width = source_attributes.get("intermediate_size")
        target_width = target_attributes.get("intermediate_size")
        if (
            not isinstance(source_width, int)
            or not isinstance(target_width, int)
            or target_width <= 0
            or target_width > source_width
        ):
            return False
        expected_attributes = dict(source_attributes)
        expected_attributes["intermediate_size"] = target_width
        if target_attributes != expected_attributes:
            return False
        if target_width == source_width:
            if any(
                source_parameters[role].shape != target_parameters[role].shape
                or source_parameters[role].dtype != target_parameters[role].dtype
                or source_parameters[role].trainable
                != target_parameters[role].trainable
                for role in source_parameters
            ):
                return False
            continue
        hidden: int | None = None
        for role, source_parameter in source_parameters.items():
            target_parameter = target_parameters[role]
            if (
                source_parameter.dtype != target_parameter.dtype
                or source_parameter.trainable != target_parameter.trainable
            ):
                return False
            semantic = _semantic_name(role)
            source_shape = source_parameter.shape
            target_shape = target_parameter.shape
            if "gate_proj" in semantic or "up_proj" in semantic:
                if (
                    len(source_shape) != 2
                    or source_shape[0] != source_width
                    or target_shape != (target_width, source_shape[1])
                ):
                    return False
                hidden = int(source_shape[1])
            elif "down_proj" in semantic:
                if (
                    len(source_shape) != 2
                    or source_shape[1] != source_width
                    or target_shape != (source_shape[0], target_width)
                ):
                    return False
                hidden = int(source_shape[0])
            elif source_shape != target_shape:
                return False
        if hidden is None or target_node.layer_index is None:
            return False
        changed_layers.append(
            (target_node.layer_index, source_width - target_width, hidden)
        )
    if len(changed_layers) != 1:
        return False
    _layer_index, removed, hidden = changed_layers[0]
    expected_delta = 3 * removed * hidden
    return (
        source.unique_parameter_count - target.unique_parameter_count == expected_delta
        and source.logical_parameter_count - target.logical_parameter_count
        == expected_delta
        and all(
            _residual_layer_contract(source, source.layer_nodes(index))
            and _residual_layer_contract(target, target.layer_nodes(index))
            for index in source.layer_indices
        )
    )


def _residual_layer_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    return _complete_layer_bundle(graph, nodes) and _ordered_residual_contract(
        graph, nodes
    )


def _ordered_layer_span_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    indices = sorted(
        {node.layer_index for node in nodes if node.layer_index is not None}
    )
    return (
        len(indices) >= 2
        and _complete_layer_bundle(graph, nodes, minimum_layers=2)
        and all(
            _residual_layer_contract(graph, graph.layer_nodes(index))
            for index in indices
        )
    )


def _bundle_kind_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
    requirement: str,
) -> bool:
    kinds = {node.kind for node in nodes}
    if requirement == "any":
        return bool(nodes)
    if requirement == "attention_or_short_conv":
        return bool(
            kinds
            & {
                CircuitKind.ATTENTION.value,
                CircuitKind.SHORT_CONV.value,
            }
        )
    if requirement == "residual":
        return CircuitKind.RESIDUAL_ADD.value in kinds
    if requirement == "causal_windowed_nope_attention_branch":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _windowed_nope_attention_contract(nodes)
    if requirement == "causal_summary_attention_branch":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _compressed_summary_attention_contract(nodes)
    if requirement == "causal_kda_recurrent_branch":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _bounded_gated_delta_recurrence_contract(graph, nodes)
    if requirement == "causal_latent_kv_attention_branch":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _compressed_latent_kv_attention_contract(nodes)
    if requirement == "causal_fused_gate_up_mlp_branch":
        return _causal_branch_contract(
            graph, nodes, branch="mlp"
        ) and _fused_gate_up_mlp_contract(nodes)
    if requirement == "causal_stable_latent_moe_channel_mixer_branch":
        return _causal_branch_contract(
            graph, nodes, branch="mlp"
        ) and _stable_latent_moe_channel_mixer_contract(nodes)
    if requirement == "native_residual_chart_interface_group":
        return _native_chart_interface_group_contract(graph, nodes)
    if requirement == "lfm2_layer":
        return _lfm2_layer_contract(graph, nodes)
    if requirement in {
        "lfm2_hybrid_model",
        "lfm2_producer_first_model",
        "lfm2_dense_budget_model",
    }:
        return (
            graph.family == "lfm2"
            and _complete_model_bundle(graph, nodes)
            and bool(graph.layer_indices)
            and dict(graph.metadata).get("hybrid_attention_short_conv") is True
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    if requirement == "residual_layer":
        return _residual_layer_contract(graph, nodes)
    if requirement == "ordered_layer_span":
        return _ordered_layer_span_contract(graph, nodes)
    if requirement == "gated_mlp_neuron":
        return CircuitKind.GATED_MLP.value in kinds
    if requirement == "attention_group":
        return CircuitKind.ATTENTION.value in kinds
    if requirement == "qwen3_dense_model":
        return (
            graph.family == "qwen3"
            and _complete_model_bundle(graph, nodes)
            and bool(graph.layer_indices)
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    if requirement == "qwen3_codebook_model":
        return (
            graph.family == "qwen3_codebook"
            and _complete_model_bundle(graph, nodes)
            and bool(graph.layer_indices)
            and any(node.kind == "attention_codebook_registry" for node in nodes)
            and any(node.kind == "neuron_codebook_registry" for node in nodes)
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    if requirement == "qwen3_compensated_model":
        return (
            graph.family == "qwen3"
            and dict(graph.metadata).get("compensated_ragged") is True
            and _complete_model_bundle(graph, nodes)
            and bool(graph.layer_indices)
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    if requirement == "neo_native_vision_model":
        return (
            graph.family == "neo_chat"
            and _complete_model_bundle(graph, nodes)
            and {interface.modality for interface in graph.interfaces}
            == {"image", "text"}
            and bool(graph.layer_indices)
            and dict(graph.metadata).get("native_temporal_height_width_attention")
            is True
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    if requirement == "native_audio_model":
        metadata = dict(graph.metadata)
        return (
            graph.family == _NATIVE_AUDIO_FAMILY
            and _complete_model_bundle(graph, nodes)
            and {interface.modality for interface in graph.interfaces}
            == _NATIVE_AUDIO_MODALITIES
            and bool(graph.layer_indices)
            and metadata.get("native_pcm16_frame_encoder") is True
            and metadata.get("external_frozen_audio_encoder") is False
            and bool(graph.nodes_of_kind(CircuitKind.MODALITY_INPUT))
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    if requirement == "native_video_model":
        metadata = dict(graph.metadata)
        return (
            graph.family == _NATIVE_VIDEO_FAMILY
            and _complete_model_bundle(graph, nodes)
            and {interface.modality for interface in graph.interfaces}
            == _NATIVE_VIDEO_MODALITIES
            and bool(graph.layer_indices)
            and metadata.get("native_rgb24_tubelet_encoder") is True
            and metadata.get("external_frozen_video_encoder") is False
            and bool(graph.nodes_of_kind(CircuitKind.MODALITY_INPUT))
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    if requirement == "native_points3d_model":
        metadata = dict(graph.metadata)
        # ``emits_quantized_centroid`` is read but not required: both the
        # input-only and the emitting shape are valid points3d models.  What is
        # required is that the declaration and the graph agree, so a model that
        # claims to emit without a MODALITY_OUTPUT node -- or that carries one
        # without declaring it -- is not plannable.
        emits = metadata.get("emits_quantized_centroid")
        return (
            graph.family == _NATIVE_POINTS3D_FAMILY
            and _complete_model_bundle(graph, nodes)
            and {interface.modality for interface in graph.interfaces}
            == _NATIVE_POINTS3D_MODALITIES
            and bool(graph.layer_indices)
            and metadata.get("native_quantized_point_set_encoder") is True
            and metadata.get("external_frozen_points3d_encoder") is False
            and bool(graph.nodes_of_kind(CircuitKind.MODALITY_INPUT))
            and isinstance(emits, bool)
            and bool(graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT)) == emits
            and all(
                _residual_layer_contract(graph, graph.layer_nodes(index))
                for index in graph.layer_indices
            )
        )
    return _nodes_declare_semantic(nodes, requirement)


def _parameter_roles(
    nodes: Sequence[CircuitNode],
    kinds: set[str],
) -> tuple[str, ...]:
    return tuple(
        _semantic_name(parameter.role)
        for node in nodes
        if node.kind in kinds
        for parameter in node.parameters
    )


def _roles_contain(
    roles: Sequence[str],
    alternatives: Sequence[str],
) -> bool:
    return any(
        _semantic_name(alternative) in role
        for role in roles
        for alternative in alternatives
    )


def _projection_role_contract(
    nodes: Sequence[CircuitNode],
    *,
    require_q: bool,
    require_k: bool,
    require_v: bool,
    require_o: bool,
) -> bool:
    roles = _parameter_roles(nodes, {CircuitKind.ATTENTION.value})
    requirements = (
        (require_q, ("q_proj", "query_proj")),
        (require_k, ("k_proj", "key_proj")),
        (require_v, ("v_proj", "value_proj")),
        # ``out_weight`` is the native-trunk spelling of the output projection
        # (chimera's ``mixer.out``); no other inspector emits that role.
        (require_o, ("o_proj", "out_proj", "output_proj", "out_weight")),
    )
    return bool(roles) and all(
        not required or _roles_contain(roles, alternatives)
        for required, alternatives in requirements
    )


def _stateful_operator_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
    *,
    operator_kind: str,
    port_kind: str,
    edge_kind: str,
) -> bool:
    operators = tuple(node for node in nodes if node.kind == operator_kind)
    for operator in operators:
        has_input = any(
            port.kind == port_kind and port.stateful for port in operator.inputs
        )
        has_output = any(
            port.kind == port_kind and port.stateful for port in operator.outputs
        )
        has_edge = any(
            edge.source_node == operator.node_id
            and edge.target_node == operator.node_id
            and edge.kind == edge_kind
            and edge.recurrent
            for edge in graph.edges
        )
        if has_input and has_output and has_edge:
            return True
    return False


def _gated_mlp_contract(
    nodes: Sequence[CircuitNode],
    *,
    require_swiglu: bool,
) -> bool:
    mlps = tuple(node for node in nodes if node.kind == CircuitKind.GATED_MLP.value)
    roles = _parameter_roles(nodes, {CircuitKind.GATED_MLP.value})
    projections = (
        _roles_contain(roles, ("gate_proj", "w1_weight"))
        and _roles_contain(roles, ("up_proj", "w3_weight"))
        and _roles_contain(roles, ("down_proj", "w2_weight"))
    )
    if not projections or not mlps:
        return False
    if not require_swiglu:
        return True
    return any(
        _semantic_name(_node_attributes(node).get("activation", ""))
        in {"silu", "swish"}
        or "post_swiglu"
        in _semantic_name(_node_attributes(node).get("coupled_roles", ""))
        for node in mlps
    )


def _short_conv_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    operators = tuple(
        node for node in nodes if node.kind == CircuitKind.SHORT_CONV.value
    )
    roles = _parameter_roles(nodes, {CircuitKind.SHORT_CONV.value})
    return bool(operators) and (
        _roles_contain(roles, ("in_proj",))
        and _roles_contain(roles, ("conv_weight",))
        and _roles_contain(roles, ("out_proj", "output_proj"))
        and any(
            _node_attributes(node).get("input_chunk_order") == "b_c_x"
            and isinstance(_node_attributes(node).get("fir_length"), int)
            for node in operators
        )
        and _stateful_operator_contract(
            graph,
            nodes,
            operator_kind=CircuitKind.SHORT_CONV.value,
            port_kind=PortKind.RECURRENT_STATE.value,
            edge_kind=EdgeKind.STATE.value,
        )
    )


def _token_interface_tying_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    embeddings = tuple(
        node for node in nodes if node.kind == CircuitKind.TOKEN_EMBEDDING.value
    )
    heads = tuple(node for node in nodes if node.kind == CircuitKind.OUTPUT_HEAD.value)
    embedding_groups = set(_storage_groups(embeddings))
    head_groups = set(_storage_groups(heads))
    return (
        graph.tokenizer_fingerprint is not None
        and bool(embeddings)
        and bool(heads)
        and bool(embedding_groups & head_groups)
    )


def _token_interface_contract(
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool:
    """Prove one complete tied or explicitly untied learned token interface."""

    embeddings = tuple(
        node for node in nodes if node.kind == CircuitKind.TOKEN_EMBEDDING.value
    )
    heads = tuple(node for node in nodes if node.kind == CircuitKind.OUTPUT_HEAD.value)
    embedding_groups = set(_storage_groups(embeddings))
    head_groups = set(_storage_groups(heads))
    tied_storage = bool(embedding_groups & head_groups)
    tied_claim = (
        _node_attributes(heads[0]).get("tied_to_embedding")
        if len(heads) == 1
        else None
    )
    return (
        graph.tokenizer_fingerprint is not None
        and len(embeddings) == 1
        and len(heads) == 1
        and bool(embedding_groups)
        and bool(head_groups)
        and type(tied_claim) is bool
        and tied_claim is tied_storage
    )


def _coupled_role_contract(
    role: str,
    graph: ModelCircuitGraph,
    nodes: Sequence[CircuitNode],
) -> bool | None:
    if _nodes_declare_semantic(nodes, role):
        return True
    if role == "q_k_v_o_cache":
        if not any(node.kind == CircuitKind.ATTENTION.value for node in nodes):
            return None
        return _projection_role_contract(
            nodes,
            require_q=True,
            require_k=True,
            require_v=True,
            require_o=True,
        ) and _stateful_operator_contract(
            graph,
            nodes,
            operator_kind=CircuitKind.ATTENTION.value,
            port_kind=PortKind.CACHE.value,
            edge_kind=EdgeKind.CACHE.value,
        )
    if role == "b_c_x_fir_state_output":
        if not any(node.kind == CircuitKind.SHORT_CONV.value for node in nodes):
            return None
        return _short_conv_contract(graph, nodes)
    if role == "gate_up_post_swiglu_down":
        return _gated_mlp_contract(nodes, require_swiglu=True)
    if role == "gate_up_down":
        return _gated_mlp_contract(nodes, require_swiglu=False)
    if role == "ordered_residual":
        return _ordered_residual_contract(graph, nodes)
    if role == "all_layers_ordered_residual":
        return bool(graph.layer_indices) and all(
            _ordered_residual_contract(graph, graph.layer_nodes(index))
            for index in graph.layer_indices
        )
    if role == "attention_mlp_residual_order":
        return _ordered_residual_contract(
            graph,
            nodes,
            require_attention=True,
        )
    if role == "q_k":
        return _projection_role_contract(
            nodes,
            require_q=True,
            require_k=True,
            require_v=False,
            require_o=False,
        )
    if role == "v_o":
        return _projection_role_contract(
            nodes,
            require_q=False,
            require_k=False,
            require_v=True,
            require_o=True,
        )
    if role == "q_k_v_o":
        return _projection_role_contract(
            nodes,
            require_q=True,
            require_k=True,
            require_v=True,
            require_o=True,
        )
    if role == "gqa_group":
        attention = tuple(
            node for node in nodes if node.kind == CircuitKind.ATTENTION.value
        )
        return bool(attention) and all(
            isinstance(_node_attributes(node).get("query_heads"), int)
            and isinstance(_node_attributes(node).get("key_value_heads"), int)
            and int(_node_attributes(node)["key_value_heads"]) > 0
            and int(_node_attributes(node)["query_heads"])
            % int(_node_attributes(node)["key_value_heads"])
            == 0
            for node in attention
        )
    if role == "cache":
        return _stateful_operator_contract(
            graph,
            nodes,
            operator_kind=CircuitKind.ATTENTION.value,
            port_kind=PortKind.CACHE.value,
            edge_kind=EdgeKind.CACHE.value,
        )
    if role == "input_norm_q_k_v_o_residual_write":
        return _causal_branch_contract(graph, nodes, branch="attention")
    if role == "input_norm_gate_up_post_swiglu_down_residual_write":
        return _causal_branch_contract(graph, nodes, branch="mlp")
    if role == "input_norm_windowed_nope_q_k_v_o_residual_write":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _windowed_nope_attention_contract(nodes)
    if role == "input_norm_local_window_plus_compressed_summary_residual_write":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _compressed_summary_attention_contract(nodes)
    if role == "local_window_and_compressed_summary_sub_branches":
        return _compressed_summary_attention_contract(nodes)
    if role == "input_norm_bounded_gated_delta_recurrence_residual_write":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _bounded_gated_delta_recurrence_contract(graph, nodes)
    if role == "bounded_gated_delta_recurrence_state":
        return _stateful_operator_contract(
            graph,
            nodes,
            operator_kind=CircuitKind.RECURRENT_MIXER.value,
            port_kind=PortKind.RECURRENT_STATE.value,
            edge_kind=EdgeKind.STATE.value,
        )
    if role == "input_norm_fused_gate_up_swiglu_down_residual_write":
        return _causal_branch_contract(
            graph, nodes, branch="mlp"
        ) and _fused_gate_up_mlp_contract(nodes)
    if role == "fused_gate_up_down":
        return _fused_gate_up_mlp_contract(nodes)
    if role == "input_norm_compressed_latent_kv_softmax_gated_residual_write":
        return _causal_branch_contract(
            graph, nodes, branch="attention"
        ) and _compressed_latent_kv_attention_contract(nodes)
    if role == "shared_latent_key_value_readout":
        return _compressed_latent_kv_attention_contract(nodes)
    if role == (
        "input_norm_top_k_routed_latent_experts_plus_shared_experts_residual_write"
    ):
        return _causal_branch_contract(
            graph, nodes, branch="mlp"
        ) and _stable_latent_moe_channel_mixer_contract(nodes)
    if role in {
        "top_k_routed_latent_and_full_width_shared_sub_branches",
        "sparse_expert_row_budget",
    }:
        return _stable_latent_moe_channel_mixer_contract(nodes)
    if role == "nope_position_transport":
        operators = tuple(
            node
            for node in nodes
            if node.kind
            in {
                CircuitKind.ATTENTION.value,
                CircuitKind.RECURRENT_MIXER.value,
                CircuitKind.SHORT_CONV.value,
            }
        )
        return bool(operators) and all(
            _node_attributes(node).get("position_policy") == "nope"
            and not _declares_rope_requirement(_node_attributes(node))
            for node in operators
        )
    if role == "learned_depth_source_selection":
        selectors = tuple(
            node
            for node in nodes
            if node.kind == CircuitKind.RESIDUAL_ADD.value
            and _node_attributes(node).get("residual_mode") == "block_attnres"
        )
        return bool(selectors) and all(
            _depth_source_selector_contract(node) for node in selectors
        )
    if role == "residual_chart_projected_interface":
        return _native_chart_interface_group_contract(graph, nodes)
    if role == "embedding_tokenizer_output_head_tying":
        return _token_interface_tying_contract(graph, nodes)
    if role == "embedding_tokenizer_output_head_interface":
        return _token_interface_contract(graph, nodes)
    return False


def validate_fold_method_contract(
    method: FoldMethodSpec,
    *,
    source_groups: Sequence[tuple[ModelCircuitGraph, Sequence[CircuitNode]]],
    target_graph: ModelCircuitGraph,
    target_nodes: Sequence[CircuitNode],
    residual_width_substitution: tuple[int, int] | None = None,
) -> None:
    """Fail closed unless actual source/target bundles satisfy a method spec.

    ``residual_width_substitution`` is supplied only for an authenticated
    cross-width boundary-chart crossing, and only ever relaxes *port* widths on
    parameter-free wiring nodes (see
    :func:`_residual_width_substituted_contract`).
    """

    if not isinstance(method, FoldMethodSpec):
        raise DynamicFoldPlanningError("method must be FoldMethodSpec.")
    groups = tuple((graph, tuple(nodes)) for graph, nodes in source_groups)
    targets = tuple(target_nodes)
    if not groups or any(not nodes for _graph, nodes in groups):
        raise DynamicFoldPlanningError(
            f"Method {method.method_id!r} requires nonempty source bundles."
        )
    if not targets:
        raise DynamicFoldPlanningError(
            f"Method {method.method_id!r} requires a nonempty target bundle."
        )
    if not method.evidence_channels:
        raise DynamicFoldPlanningError(
            f"Method {method.method_id!r} has no evidence-channel contract."
        )

    for graph, nodes in groups:
        for requirement in method.source_kinds:
            if not _bundle_kind_contract(graph, nodes, requirement):
                raise DynamicFoldPlanningError(
                    f"Method {method.method_id!r} source kind contract "
                    f"{requirement!r} is not satisfied by "
                    f"{graph.model_name!r}."
                )

    for requirement in method.target_kinds:
        if requirement == "same_semantics_same_shape":
            satisfied = all(
                _same_parameter_contract(
                    graph,
                    nodes,
                    target_graph,
                    targets,
                    residual_width_substitution=residual_width_substitution,
                )
                for graph, nodes in groups
            )
        else:
            satisfied = _bundle_kind_contract(
                target_graph,
                targets,
                requirement,
            )
        if not satisfied:
            raise DynamicFoldPlanningError(
                f"Method {method.method_id!r} target kind contract "
                f"{requirement!r} is not satisfied."
            )

    bundles = (*groups, (target_graph, targets))
    roles = set(method.coupled_roles)
    operator_alternatives = {
        "q_k_v_o_cache",
        "b_c_x_fir_state_output",
    }
    declared_alternatives = roles & operator_alternatives
    if len(declared_alternatives) > 1:
        for graph, nodes in bundles:
            results = tuple(
                _coupled_role_contract(role, graph, nodes)
                for role in sorted(declared_alternatives)
            )
            applicable = tuple(value for value in results if value is not None)
            if not applicable or not any(applicable) or not all(applicable):
                raise DynamicFoldPlanningError(
                    f"Method {method.method_id!r} operator-role alternative "
                    f"is not satisfied by {graph.model_name!r}."
                )
        roles -= declared_alternatives

    for role in sorted(roles):
        if role == "all_target_tensor_roles":
            satisfied = all(
                _same_parameter_contract(
                    graph,
                    nodes,
                    target_graph,
                    targets,
                    residual_width_substitution=residual_width_substitution,
                )
                for graph, nodes in groups
            )
            if not satisfied:
                raise DynamicFoldPlanningError(
                    f"Method {method.method_id!r} does not cover every target "
                    "tensor role with matching source semantics."
                )
            continue
        for graph, nodes in bundles:
            if _coupled_role_contract(role, graph, nodes) is not True:
                raise DynamicFoldPlanningError(
                    f"Method {method.method_id!r} coupled role {role!r} "
                    f"is not proven for {graph.model_name!r}."
                )


def _lfm2_ordered_layer_types(graph: ModelCircuitGraph) -> tuple[str, ...]:
    result = []
    for index in graph.layer_indices:
        operators = {
            node.kind
            for node in graph.layer_nodes(index)
            if node.kind
            in {
                CircuitKind.ATTENTION.value,
                CircuitKind.SHORT_CONV.value,
            }
        }
        if len(operators) != 1:
            return ()
        result.append(
            "full_attention"
            if next(iter(operators)) == CircuitKind.ATTENTION.value
            else "conv"
        )
    return tuple(result)


def _lfm2_hybrid_model_contract(
    parent: ModelCircuitGraph,
    child: ModelCircuitGraph,
) -> bool:
    """Recognize only real shallower LFM2 model targets."""

    parent_types = _lfm2_ordered_layer_types(parent)
    child_types = _lfm2_ordered_layer_types(child)
    child_metadata = dict(child.metadata)
    encoded_types = ",".join(child_types)
    spans_value = child_metadata.get("ordered_source_spans")
    if not isinstance(spans_value, str):
        return False
    try:
        spans = tuple(
            tuple(int(part) for part in span.split(","))
            for span in spans_value.split(";")
        )
    except ValueError:
        return False
    flattened = tuple(index for span in spans for index in span)
    return bool(
        parent.family == child.family == "lfm2"
        and parent_types
        and child_types
        and len(child_types) < len(parent_types)
        and parent.unique_parameter_count > child.unique_parameter_count
        and dict(parent.metadata).get("hidden_size") == child_metadata.get("hidden_size")
        and dict(parent.metadata).get("hybrid_attention_short_conv") is True
        and child_metadata.get("hybrid_attention_short_conv") is True
        and child_metadata.get("fold_method_id")
        == "lfm2_hybrid_residual_attention_model"
        and child_metadata.get("target_layer_types") == encoded_types
        and len(spans) == len(child_types)
        and flattened == tuple(range(len(parent_types)))
        and parent.tokenizer_fingerprint is not None
        and parent.tokenizer_fingerprint == child.tokenizer_fingerprint
    )


def _lfm2_producer_first_model_contract(
    parent: ModelCircuitGraph,
    child: ModelCircuitGraph,
) -> bool:
    """Recognize only full-depth LFM2 width/GQA targets with an explicit marker."""

    parent_types = _lfm2_ordered_layer_types(parent)
    child_types = _lfm2_ordered_layer_types(child)
    parent_metadata = dict(parent.metadata)
    child_metadata = dict(child.metadata)
    return bool(
        parent.family == child.family == "lfm2"
        and parent_types
        and parent_types == child_types
        and len(parent.layer_indices) == len(child.layer_indices)
        and parent.unique_parameter_count > child.unique_parameter_count
        and parent_metadata.get("hidden_size") == child_metadata.get("hidden_size")
        and parent_metadata.get("hybrid_attention_short_conv") is True
        and child_metadata.get("hybrid_attention_short_conv") is True
        and child_metadata.get("fold_method_id")
        == "lfm2_producer_first_residual_attention_model"
        and "ordered_source_spans" not in child_metadata
        and "target_layer_types" not in child_metadata
        and parent.tokenizer_fingerprint is not None
        and parent.tokenizer_fingerprint == child.tokenizer_fingerprint
    )


def _lfm2_dense_budget_model_contract(
    parent: ModelCircuitGraph,
    child: ModelCircuitGraph,
) -> bool:
    """Recognize an executable same-depth dense width/GQA LFM2 target."""

    parent_types = _lfm2_ordered_layer_types(parent)
    child_types = _lfm2_ordered_layer_types(child)
    parent_metadata = dict(parent.metadata)
    child_metadata = dict(child.metadata)
    spans_value = child_metadata.get("ordered_source_spans")
    if not isinstance(spans_value, str):
        return False
    try:
        spans = tuple(
            tuple(int(part) for part in span.split(","))
            for span in spans_value.split(";")
        )
    except ValueError:
        return False
    expected_spans = tuple((index,) for index in range(len(parent_types)))
    bridge_required = child_metadata.get("tokenizer_bridge_required")
    if bridge_required is False:
        tokenizer_contract = bool(
            parent.tokenizer_fingerprint is not None
            and parent.tokenizer_fingerprint == child.tokenizer_fingerprint
        )
    elif bridge_required is True:
        tokenizer_contract = bool(
            parent.tokenizer_fingerprint is not None
            and child.tokenizer_fingerprint is not None
            and parent.tokenizer_fingerprint
            != child.tokenizer_fingerprint
            and child_metadata.get(
                "parent_tokenizer_semantics_sha256"
            )
            == parent.tokenizer_fingerprint
            and child_metadata.get(
                "child_tokenizer_semantics_sha256"
            )
            == child.tokenizer_fingerprint
            and _is_digest(
                child_metadata.get(
                    "tokenizer_bridge_artifact_sha256"
                )
            )
            and _is_digest(
                child_metadata.get(
                    "tokenizer_transport_profile_sha256"
                )
            )
        )
    else:
        tokenizer_contract = False
    return bool(
        parent.family == child.family == "lfm2"
        and parent_types
        and parent_types == child_types
        and len(parent.layer_indices) == len(child.layer_indices)
        and spans == expected_spans
        and parent.unique_parameter_count > child.unique_parameter_count
        and isinstance(parent_metadata.get("hidden_size"), int)
        and isinstance(child_metadata.get("hidden_size"), int)
        and int(parent_metadata["hidden_size"])
        > int(child_metadata["hidden_size"])
        and parent_metadata.get("hybrid_attention_short_conv") is True
        and child_metadata.get("hybrid_attention_short_conv") is True
        and child_metadata.get("fold_method_id")
        == "lfm2_dense_budget_residual_attention_model"
        and isinstance(child_metadata.get("dense_topology_catalog"), str)
        and bool(child_metadata["dense_topology_catalog"])
        and child_metadata.get("target_layer_types")
        == ",".join(child_types)
        and tokenizer_contract
    )


class DynamicFoldPlanner:
    """Compile structural graphs into automatic complete-circuit actions."""

    def __init__(
        self,
        *,
        methods: FoldMethodRegistry | None = None,
        ledger: ExperimentLedger | None = None,
        enabled_experimental_method_ids: Sequence[str] = (),
    ) -> None:
        self.methods = methods or default_method_registry()
        self.ledger = ledger or ExperimentLedger()
        if isinstance(enabled_experimental_method_ids, (str, bytes)):
            raise DynamicFoldPlanningError(
                "enabled_experimental_method_ids must be a sequence of ids."
            )
        enabled = tuple(
            sorted(
                {
                    _text(value, "enabled experimental method id")
                    for value in enabled_experimental_method_ids
                }
            )
        )
        for method_id in enabled:
            status = self.methods.get(method_id).status
            if status != MethodStatus.EXPERIMENTAL.value:
                raise DynamicFoldPlanningError(
                    f"Method {method_id!r} is {status}; only experimental "
                    "methods are enabled by explicit opt-in."
                )
        self.enabled_experimental_method_ids = enabled

    def _action(
        self,
        *,
        index: int,
        method_id: str,
        parent_graphs: Sequence[ModelCircuitGraph],
        parent_state_fingerprints: Mapping[str, str],
        source_groups: Sequence[tuple[ModelCircuitGraph, Sequence[CircuitNode]]],
        target_nodes: Sequence[CircuitNode],
        child_graph: ModelCircuitGraph,
        child_state_fingerprint: str,
        evidence_fingerprint: str,
        dependencies: tuple[str, ...],
        configuration: Mapping[str, JsonScalar],
        residual_width_substitution: tuple[int, int] | None = None,
    ) -> FoldAction:
        method = self.methods.require_selectable(method_id)
        if not method.standalone:
            raise DynamicFoldPlanningError(
                f"Method {method_id!r} cannot produce a standalone child."
            )
        validate_fold_method_contract(
            method,
            source_groups=source_groups,
            target_graph=child_graph,
            target_nodes=target_nodes,
            residual_width_substitution=residual_width_substitution,
        )
        sources = tuple(
            SourceCircuitRef(
                parent_name=graph.model_name,
                node_ids=tuple(value.node_id for value in nodes),
            )
            for graph, nodes in source_groups
        )
        target_ids = tuple(value.node_id for value in target_nodes)
        transition = (
            "+".join(
                sorted(
                    {value.kind for _graph, nodes in source_groups for value in nodes}
                )
            )
            + "->"
            + "+".join(sorted({value.kind for value in target_nodes}))
        )
        signature = ExperimentSignature(
            method_id=method.method_id,
            method_version=method.version,
            family_id=method.family_id,
            parent_graph_fingerprints=tuple(
                value.fingerprint for value in parent_graphs
            ),
            parent_state_fingerprints=tuple(sorted(parent_state_fingerprints.items())),
            child_graph_fingerprint=child_graph.fingerprint,
            child_state_fingerprint=child_state_fingerprint,
            evidence_fingerprint=evidence_fingerprint,
            source_unit_ids=tuple(
                f"{graph.model_name}:{node.node_id}"
                for graph, nodes in source_groups
                for node in nodes
            ),
            target_unit_ids=target_ids,
            topology_transition=transition,
            selection_policy=str(
                configuration.get("selection_policy", "automatic_semantic_alignment")
            ),
            fit_objective=str(
                configuration.get(
                    "fit_objective",
                    "closed_loop_parent_relative_residual_endpoint",
                )
            ),
            promotion_gate=str(
                configuration.get(
                    "promotion_gate",
                    "automatic_behavior_then_final_manual_review",
                )
            ),
            zero_gradient=True,
            configuration=tuple(configuration.items()),
        )
        decision = self.ledger.decision(signature)
        action_id = f"action-{index:04d}-{signature.candidate_sha256[:12]}"
        action = FoldAction(
            action_id=action_id,
            sequence_index=index,
            method_id=method_id,
            sources=sources,
            target_node_ids=target_ids,
            target_parameter_paths=tuple(
                path for value in target_nodes for path in value.parameter_paths
            ),
            target_storage_groups=_storage_groups(target_nodes),
            dependencies=dependencies,
            evidence_requirements=method.evidence_channels,
            source_logical_parameters=sum(
                value.logical_parameter_count
                for _graph, nodes in source_groups
                for value in nodes
            ),
            target_logical_parameters=sum(
                value.logical_parameter_count for value in target_nodes
            ),
            configuration=tuple(configuration.items()),
            signature=signature,
            ledger_decision=decision,
        )
        if action.evidence_requirements != method.evidence_channels:
            raise DynamicFoldPlanningError(
                f"Method {method_id!r} evidence channels were not bound "
                "exactly into the fold action."
            )
        return action

    def plan(
        self,
        parents: Sequence[ModelCircuitGraph],
        child: ModelCircuitGraph,
        *,
        child_state_fingerprint: str,
        evidence_fingerprint: str,
        parent_state_fingerprints: Mapping[str, str],
        exact_target_unique_parameters: int | None = None,
    ) -> DynamicFoldPlan:
        parents = tuple(parents)
        if not parents:
            raise DynamicFoldPlanningError("At least one parent graph is required.")
        if len({value.model_name for value in parents}) != len(parents):
            raise DynamicFoldPlanningError("Parent model names must be unique.")
        parent_state_fingerprints = {
            _text(name, "parent state name"): _digest(
                digest, "parent state fingerprint"
            )
            for name, digest in parent_state_fingerprints.items()
        }
        parent_names = {value.model_name for value in parents}
        if set(parent_state_fingerprints) != parent_names:
            raise DynamicFoldPlanningError(
                "Parent state fingerprint names must match parent graph names."
            )
        child_state_fingerprint = _digest(
            child_state_fingerprint, "child_state_fingerprint"
        )
        evidence_fingerprint = _digest(evidence_fingerprint, "evidence_fingerprint")
        budget = (
            child.unique_parameter_count
            if exact_target_unique_parameters is None
            else _positive_int(
                exact_target_unique_parameters,
                "exact_target_unique_parameters",
            )
        )
        if budget != child.unique_parameter_count:
            raise DynamicFoldPlanningError(
                "The concrete child graph does not match the requested exact budget."
            )
        if (
            len(parents) == 1
            and _lfm2_dense_budget_model_contract(parents[0], child)
        ):
            parent = parents[0]
            parent_types = _lfm2_ordered_layer_types(parent)
            child_metadata = dict(child.metadata)
            action = self._action(
                index=0,
                method_id="lfm2_dense_budget_residual_attention_model",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "catalog_id": str(
                        child_metadata["dense_topology_catalog"]
                    ),
                    "source_depth": len(parent_types),
                    "target_depth": len(parent_types),
                    "source_spans": str(
                        child_metadata["ordered_source_spans"]
                    ),
                    "target_layer_types": str(
                        child_metadata["target_layer_types"]
                    ),
                    "target_hidden_size": int(
                        child_metadata["hidden_size"]
                    ),
                    "target_intermediate_size": int(
                        next(
                            dict(node.attributes)["intermediate_size"]
                            for node in child.nodes
                            if node.kind == CircuitKind.GATED_MLP.value
                        )
                    ),
                    "target_query_heads": int(
                        next(
                            dict(node.attributes)["query_heads"]
                            for node in child.nodes
                            if node.kind == CircuitKind.ATTENTION.value
                        )
                    ),
                    "target_key_value_heads": int(
                        next(
                            dict(node.attributes)["key_value_heads"]
                            for node in child.nodes
                            if node.kind == CircuitKind.ATTENTION.value
                        )
                    ),
                    "target_head_dim": int(
                        next(
                            dict(node.attributes)["head_dim"]
                            for node in child.nodes
                            if node.kind == CircuitKind.ATTENTION.value
                        )
                    ),
                    **(
                        {
                            "tokenizer_bridge_required": True,
                            "tokenizer_bridge_artifact_sha256": str(
                                child_metadata[
                                    "tokenizer_bridge_artifact_sha256"
                                ]
                            ),
                            "tokenizer_transport_profile_sha256": str(
                                child_metadata[
                                    "tokenizer_transport_profile_sha256"
                                ]
                            ),
                            "parent_tokenizer_semantics_sha256": str(
                                child_metadata[
                                    "parent_tokenizer_semantics_sha256"
                                ]
                            ),
                            "child_tokenizer_semantics_sha256": str(
                                child_metadata[
                                    "child_tokenizer_semantics_sha256"
                                ]
                            ),
                        }
                        if child_metadata[
                            "tokenizer_bridge_required"
                        ]
                        else {}
                    ),
                    "fit_objective": (
                        "complete_weight_residual_basis_then_cached_child_"
                        "closed_loop_attention_swiglu_output"
                    ),
                    "selection_policy": (
                        "automatic_dense_budget_catalog_same_depth_executable"
                    ),
                    "promotion_gate": (
                        "cached_whole_output_behavior_then_fresh_reload"
                    ),
                    "blank_child_features_used": False,
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if (
            len(parents) == 1
            and _lfm2_producer_first_model_contract(parents[0], child)
        ):
            parent = parents[0]
            layer_types = _lfm2_ordered_layer_types(parent)
            action = self._action(
                index=0,
                method_id="lfm2_producer_first_residual_attention_model",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "source_depth": len(layer_types),
                    "target_depth": len(layer_types),
                    "ordered_layer_types": ",".join(layer_types),
                    "fit_objective": (
                        "producer_first_sequential_residual_attention_feedback"
                    ),
                    "selection_policy": (
                        "sealed_parent_medoid_producers_before_child_capture"
                    ),
                    "promotion_gate": (
                        "cached_whole_output_behavior_then_fresh_reload"
                    ),
                    "blank_child_features_used": False,
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if len(parents) == 1 and _lfm2_hybrid_model_contract(parents[0], child):
            parent = parents[0]
            source_types = _lfm2_ordered_layer_types(parent)
            target_types = _lfm2_ordered_layer_types(child)
            spans = tuple(
                tuple(int(part) for part in span.split(","))
                for span in str(
                    dict(child.metadata)["ordered_source_spans"]
                ).split(";")
            )
            action = self._action(
                index=0,
                method_id="lfm2_hybrid_residual_attention_model",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "source_depth": len(source_types),
                    "target_depth": len(target_types),
                    "source_layer_types": ",".join(source_types),
                    "target_layer_types": ",".join(target_types),
                    "ordered_source_spans": ";".join(
                        ",".join(str(index) for index in span) for span in spans
                    ),
                    "fit_objective": (
                        "joint_depth_attention_swiglu_ordered_residual_"
                        "endpoint_compaction"
                    ),
                    "selection_policy": (
                        "explicit_ordered_source_span_model_fold"
                    ),
                    "promotion_gate": (
                        "cached_whole_output_behavior_then_fresh_reload"
                    ),
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if (
            len(parents) == 1
            and parents[0].family == "qwen3"
            and child.family == "qwen3_codebook"
        ):
            parent = parents[0]
            attention_owner = child.node("storage.attention_codebook")
            neuron_owner = child.node("storage.neuron_codebook")
            attention_attributes = dict(attention_owner.attributes)
            neuron_attributes = dict(neuron_owner.attributes)
            action = self._action(
                index=0,
                method_id="qwen3_global_codebook_fold",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "attention_prototypes_per_group": int(
                        attention_attributes["prototypes_per_group"]
                    ),
                    "neuron_prototypes_per_neuron": (
                        neuron_attributes["prototypes_per_neuron"]
                    ),
                    "fit_objective": (
                        "global_coupled_qkvo_and_swiglu_medoid_installation"
                    ),
                    "selection_policy": (
                        "single_atomic_qwen3_model_level_codebook_action"
                    ),
                    "promotion_gate": ("cached_parent_whole_output_then_fresh_reload"),
                    "target_token_row_fingerprint": dict(child.metadata)[
                        "token_row_fingerprint"
                    ],
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if len(parents) == 1 and _neo_single_neuron_model_contract(parents[0], child):
            parent = parents[0]
            action = self._action(
                index=0,
                method_id="neo_native_vision_swiglu_model",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "fit_objective": (
                        "image_conditioned_single_swiglu_neuron_merge_per_layer_"
                        "with_exact_native_spatial_attention_copy"
                    ),
                    "selection_policy": (
                        "sealed_calibration_trace_low_importance_anchor_regression"
                    ),
                    "promotion_gate": (
                        "cached_parent_visual_behavior_then_fresh_reload"
                    ),
                    "layer_count": len(child.layer_indices),
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if len(parents) == 1 and _native_audio_model_contract(parents[0], child):
            parent = parents[0]
            action = self._action(
                index=0,
                method_id="native_audio_swiglu_model",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "fit_objective": (
                        "audio_conditioned_single_swiglu_neuron_merge_per_layer_"
                        "with_exact_native_pcm16_front_end_copy"
                    ),
                    "selection_policy": (
                        "sealed_calibration_trace_low_importance_anchor_regression"
                    ),
                    "promotion_gate": (
                        "cached_parent_audio_behavior_then_fresh_reload"
                    ),
                    "layer_count": len(child.layer_indices),
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if len(parents) == 1 and _native_video_model_contract(parents[0], child):
            parent = parents[0]
            action = self._action(
                index=0,
                method_id="native_video_swiglu_model",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "fit_objective": (
                        "video_conditioned_single_swiglu_neuron_merge_per_layer_"
                        "with_exact_native_rgb24_front_end_copy"
                    ),
                    "selection_policy": (
                        "sealed_calibration_trace_low_importance_anchor_regression"
                    ),
                    "promotion_gate": (
                        "cached_parent_video_behavior_then_fresh_reload"
                    ),
                    "layer_count": len(child.layer_indices),
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if len(parents) == 1 and _native_points3d_model_contract(parents[0], child):
            parent = parents[0]
            emits = bool(dict(child.metadata).get("emits_quantized_centroid", False))
            action = self._action(
                index=0,
                method_id="native_points3d_swiglu_model",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "fit_objective": (
                        "points3d_conditioned_single_swiglu_neuron_merge_per_"
                        "layer_with_exact_native_point_set_front_end_copy"
                    ),
                    "selection_policy": (
                        "sealed_calibration_trace_low_importance_anchor_regression"
                    ),
                    "promotion_gate": (
                        "cached_parent_points3d_emission_and_behavior_then_"
                        "fresh_reload"
                        if emits
                        else "cached_parent_points3d_behavior_then_fresh_reload"
                    ),
                    "emits_modality_output": emits,
                    "layer_count": len(child.layer_indices),
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        if len(parents) == 1 and _qwen3_compensated_model_contract(parents[0], child):
            parent = parents[0]
            child_metadata = dict(child.metadata)
            compact_layers = str(child_metadata["compact_layer_indices"])
            action = self._action(
                index=0,
                method_id="qwen3_compensated_transaction_replay",
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=((parent, parent.nodes),),
                target_nodes=child.nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=(),
                configuration={
                    "fit_objective": (
                        "authenticated_multi_anchor_down_column_compensation"
                    ),
                    "selection_policy": ("cached_accepted_transaction_sequence_replay"),
                    "promotion_gate": ("cached_parent_whole_output_then_fresh_reload"),
                    "compact_layer_indices": compact_layers,
                    "layer_widths_sha256": str(child_metadata["layer_widths_sha256"]),
                    "zero_gradient": True,
                },
            )
            return DynamicFoldPlan(
                parent_graph_fingerprints=(parent.fingerprint,),
                parent_state_fingerprints=tuple(
                    sorted(parent_state_fingerprints.items())
                ),
                child_graph_fingerprint=child.fingerprint,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                exact_target_unique_parameters=budget,
                actions=(action,),
                gaps=(),
                registry_fingerprint=self.methods.fingerprint,
                ledger_fingerprint=self.ledger.fingerprint,
            )
        actions: list[FoldAction] = []
        gaps: list[FoldPlanGap] = []
        covered: set[str] = set()
        gapped: set[str] = set()
        covered_nodes: set[str] = set()
        gapped_nodes: set[str] = set()
        dependency: tuple[str, ...] = ()

        target_layers = child.layer_indices
        layer_orders = [
            value.order for value in child.nodes if value.layer_index is not None
        ]
        first_layer_order = min(layer_orders) if layer_orders else 0
        pre_layer_globals = tuple(
            value
            for value in child.nodes
            if value.layer_index is None and value.order < first_layer_order
        )
        post_layer_globals = tuple(
            value
            for value in child.nodes
            if value.layer_index is None and value.order >= first_layer_order
        )

        cross_width_globals = bool(
            len(parents) == 1
            and _cross_width_boundary_chart_crossing(parents[0], child)
        )
        cross_width_substitution = (
            _residual_widths(parents[0], child) if cross_width_globals else None
        )

        def add_global_targets(targets: Sequence[CircuitNode]) -> None:
            nonlocal dependency
            for target in targets:
                if target.node_id in covered_nodes or target.node_id in gapped_nodes:
                    continue
                target_bundle = _global_alias_bundle(child, target)
                target_is_token_interface = target.kind in {
                    CircuitKind.TOKEN_EMBEDDING.value,
                    CircuitKind.OUTPUT_HEAD.value,
                }
                tokenizer_bridge_required = bool(
                    target_is_token_interface
                    and child.tokenizer_fingerprint is not None
                    and all(
                        parent.tokenizer_fingerprint is not None
                        for parent in parents
                    )
                    and any(
                        parent.tokenizer_fingerprint != child.tokenizer_fingerprint
                        for parent in parents
                    )
                )
                if tokenizer_bridge_required:
                    bridge_bundle = _global_token_interface_bundle(child)
                    if bridge_bundle:
                        target_bundle = bridge_bundle
                chart_interface_bundle = ()
                if cross_width_globals and target_is_token_interface:
                    # Both interface-locked residual boundaries change width at
                    # once: the embedding writes boundary 0 and the final norm
                    # reads the terminal boundary before the head.  They are one
                    # transaction because one chart pair fixes both, and an
                    # exact copy is impossible at either.
                    chart_interface_bundle = _cross_width_terminal_interface_bundle(
                        child
                    )
                    if chart_interface_bundle:
                        target_bundle = chart_interface_bundle
                target_node_ids = {value.node_id for value in target_bundle}
                target_paths = {
                    path for node in target_bundle for path in node.parameter_paths
                }
                if target_node_ids <= covered_nodes:
                    continue
                source_groups = []
                for parent in parents:
                    matched = tuple(
                        source
                        for bundle_node in target_bundle
                        for source in _global_match(parent, bundle_node)
                    )
                    if len(matched) != len(target_bundle):
                        source_groups = []
                        break
                    source_groups.append((parent, matched))
                if not source_groups:
                    if target_paths:
                        gaps.append(
                            FoldPlanGap(
                                tuple(value.node_id for value in target_bundle),
                                "No parent exposes the required global interface circuit.",
                                "global_interface_alignment",
                            )
                        )
                        gapped.update(target_paths)
                        gapped_nodes.update(target_node_ids)
                    continue
                token_interface = any(
                    value.kind
                    in {
                        CircuitKind.TOKEN_EMBEDDING.value,
                        CircuitKind.OUTPUT_HEAD.value,
                    }
                    for value in target_bundle
                )
                if token_interface and (
                    child.tokenizer_fingerprint is None
                    or any(parent.tokenizer_fingerprint is None for parent in parents)
                ):
                    gaps.append(
                        FoldPlanGap(
                            tuple(value.node_id for value in target_bundle),
                            "Tokenizer identity is unknown; exact token-interface "
                            "copy or bridging cannot be proven.",
                            "tokenizer_fingerprint",
                        )
                    )
                    gapped.update(target_paths)
                    gapped_nodes.update(target_node_ids)
                    continue
                substitution: tuple[int, int] | None = None
                boundary_chart_required = False
                if token_interface and any(
                    parent.tokenizer_fingerprint != child.tokenizer_fingerprint
                    for parent in parents
                ):
                    method_id = "token_interface_bridge"
                    fit_objective = "byte_span_and_semantic_token_interface"
                    boundary_chart_required = bool(chart_interface_bundle)
                elif token_interface and chart_interface_bundle:
                    method_id = "token_interface_bridge"
                    fit_objective = "chart_projected_token_head_interface"
                    boundary_chart_required = True
                elif len(parents) == 1 and _same_parameter_contract(
                    parents[0],
                    source_groups[0][1],
                    child,
                    target_bundle,
                ):
                    method_id = "copy_exact"
                    fit_objective = "exact_role_copy"
                elif (
                    cross_width_substitution is not None
                    and not target_paths
                    and _same_parameter_contract(
                        parents[0],
                        source_groups[0][1],
                        child,
                        target_bundle,
                        residual_width_substitution=cross_width_substitution,
                    )
                ):
                    # A parameter-free global whose only difference is the
                    # residual width the chart set already re-coordinates.
                    # There is no tensor to copy, so nothing is claimed here
                    # beyond structural wiring equality.
                    method_id = "copy_exact"
                    fit_objective = "exact_role_copy"
                    substitution = cross_width_substitution
                    boundary_chart_required = True
                elif cross_width_substitution is not None and (
                    interface_classes := _native_chart_interface_crossing(
                        parents[0],
                        source_groups[0][1],
                        child,
                        target_bundle,
                        cross_width_substitution,
                    )
                ):
                    # A learned global interface circuit that is the parent's,
                    # restated at the contracted residual width.  The tensor
                    # itself is compiled by a registered chart operator, so
                    # nothing here claims an exact copy.
                    method_id = NATIVE_RESIDUAL_CHART_INTERFACE_METHOD
                    fit_objective = "chart_projected_" + "_and_".join(
                        interface_classes
                    )
                    boundary_chart_required = True
                else:
                    gaps.append(
                        FoldPlanGap(
                            tuple(value.node_id for value in target_bundle),
                            "Global circuit semantics differ and no bridge is "
                            "registered.",
                            (
                                "global_interface_projection"
                                if target_paths
                                else "global_structural_semantic_alignment"
                            ),
                        )
                    )
                    gapped.update(target_paths)
                    gapped_nodes.update(target_node_ids)
                    continue
                action = self._action(
                    index=len(actions),
                    method_id=method_id,
                    parent_graphs=parents,
                    parent_state_fingerprints=parent_state_fingerprints,
                    source_groups=source_groups,
                    target_nodes=target_bundle,
                    child_graph=child,
                    child_state_fingerprint=child_state_fingerprint,
                    evidence_fingerprint=evidence_fingerprint,
                    dependencies=dependency,
                    configuration={
                        "fit_objective": fit_objective,
                        "selection_policy": (
                            "automatic_global_semantic_role_alignment"
                        ),
                        "promotion_gate": (
                            "automatic_behavior_then_final_manual_review"
                        ),
                        **(
                            {"boundary_chart_required": True}
                            if boundary_chart_required
                            else {}
                        ),
                    },
                    residual_width_substitution=substitution,
                )
                actions.append(action)
                covered.update(action.target_parameter_paths)
                covered_nodes.update(action.target_node_ids)
                dependency = (action.action_id,)

        add_global_targets(pre_layer_globals)
        for target_index in target_layers:
            target_nodes = child.layer_nodes(target_index)
            source_groups = []
            spans = []
            for parent in parents:
                span = _source_span_indices(
                    len(parent.layer_indices),
                    len(target_layers),
                    target_index,
                )
                spans.append(span)
                source_groups.append((parent, _nodes_for_span(parent, span)))
            if any(not nodes for _graph, nodes in source_groups):
                gaps.append(
                    FoldPlanGap(
                        tuple(value.node_id for value in target_nodes),
                        "A parent depth span has no executable circuit nodes.",
                        "depth_alignment",
                    )
                )
                gapped.update(
                    path for value in target_nodes for path in value.parameter_paths
                )
                gapped_nodes.update(value.node_id for value in target_nodes)
                continue

            method_id: str | None = None
            configuration: dict[str, JsonScalar] = {
                "target_layer_index": target_index,
                "source_spans": ";".join(
                    f"{graph.model_name}:{','.join(str(i) for i in span)}"
                    for (graph, _nodes), span in zip(source_groups, spans, strict=True)
                ),
                "selection_policy": "automatic_complete_circuit_alignment",
                "promotion_gate": "automatic_behavior_then_final_manual_review",
            }
            if (
                method_id is None
                and len(parents) == 1
                and parents[0].family == child.family == "lfm2"
                and len(spans[0]) == 1
            ):
                source_kinds = {value.kind for value in source_groups[0][1]}
                target_kinds = {value.kind for value in target_nodes}
                source_operator = source_kinds & {
                    CircuitKind.ATTENTION.value,
                    CircuitKind.SHORT_CONV.value,
                }
                target_operator = target_kinds & {
                    CircuitKind.ATTENTION.value,
                    CircuitKind.SHORT_CONV.value,
                }
                if source_operator == target_operator:
                    method_id = "lfm2_closed_loop_layer"
                    configuration["fit_objective"] = (
                        "qkvo_or_bcx_plus_swiglu_ordered_residual_closed_loop"
                    )
            if method_id is None and len(spans[0]) > 1:
                method_id = "ordered_residual_span_compose"
                configuration["fit_objective"] = (
                    "ordered_parent_span_to_child_residual_transition"
                )
            if (
                method_id is None
                and len(parents) == 1
                and _same_parameter_contract(
                    parents[0],
                    source_groups[0][1],
                    child,
                    target_nodes,
                )
            ):
                method_id = "copy_exact"
                configuration["fit_objective"] = "exact_role_copy"
            causal_branches = None
            cross_width_crossing = False
            if method_id is None and len(parents) == 1 and len(spans[0]) == 1:
                causal_branches = _family_neutral_causal_layer_contract(
                    parents[0],
                    source_groups[0][1],
                    child,
                    target_nodes,
                )
                if causal_branches is not None and not all(
                    branch_method_id in self.enabled_experimental_method_ids
                    for branch_method_id, _source, _target, _fit in causal_branches
                ):
                    # Both branches of a layer are opted into together: a layer
                    # whose attention half has no owning transaction cannot be
                    # closed by folding its MLP half alone.
                    causal_branches = None
                cross_width_crossing = causal_branches is not None and bool(
                    _cross_width_boundary_chart_crossing(parents[0], child)
                )
            if causal_branches is not None:
                for (
                    branch_method_id,
                    source_branch,
                    target_branch,
                    fit_objective,
                ) in causal_branches:
                    action = self._action(
                        index=len(actions),
                        method_id=branch_method_id,
                        parent_graphs=parents,
                        parent_state_fingerprints=parent_state_fingerprints,
                        source_groups=((parents[0], source_branch),),
                        target_nodes=target_branch,
                        child_graph=child,
                        child_state_fingerprint=child_state_fingerprint,
                        evidence_fingerprint=evidence_fingerprint,
                        dependencies=dependency,
                        configuration={
                            **configuration,
                            "source_layer_index": spans[0][0],
                            "source_family": parents[0].family,
                            "target_family": child.family,
                            "fit_objective": fit_objective,
                            "selection_policy": (
                                "explicit_family_neutral_causal_branch_operator"
                            ),
                            "zero_gradient": True,
                            **(
                                {"boundary_chart_required": True}
                                if cross_width_crossing
                                else {}
                            ),
                        },
                    )
                    actions.append(action)
                    covered.update(action.target_parameter_paths)
                    covered_nodes.update(action.target_node_ids)
                    dependency = (action.action_id,)
                continue
            if method_id is None:
                node_ids = tuple(value.node_id for value in target_nodes)
                named = _native_kernel_family_gaps(target_nodes)
                if not named:
                    named = _native_kernel_family_gaps(
                        tuple(
                            value
                            for _graph, nodes in source_groups
                            for value in nodes
                        )
                    )
                if not named:
                    named = (
                        (
                            "cross_family_closed_loop_layer_operator",
                            "No registered complete-layer operator covers this "
                            "source/target topology.",
                        ),
                    )
                for capability, reason in named:
                    gaps.append(FoldPlanGap(node_ids, reason, capability))
                gapped.update(
                    path for value in target_nodes for path in value.parameter_paths
                )
                gapped_nodes.update(value.node_id for value in target_nodes)
                continue
            if not self.methods.get(method_id).standalone:
                gaps.append(
                    FoldPlanGap(
                        tuple(value.node_id for value in target_nodes),
                        f"Method {method_id!r} requires a non-standalone sidecar.",
                        "standalone_depth_operator",
                    )
                )
                gapped.update(
                    path for value in target_nodes for path in value.parameter_paths
                )
                gapped_nodes.update(value.node_id for value in target_nodes)
                continue
            action = self._action(
                index=len(actions),
                method_id=method_id,
                parent_graphs=parents,
                parent_state_fingerprints=parent_state_fingerprints,
                source_groups=source_groups,
                target_nodes=target_nodes,
                child_graph=child,
                child_state_fingerprint=child_state_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                dependencies=dependency,
                configuration=configuration,
            )
            actions.append(action)
            covered.update(action.target_parameter_paths)
            covered_nodes.update(action.target_node_ids)
            dependency = (action.action_id,)

        add_global_targets(post_layer_globals)

        missing_paths = sorted(set(child.parameter_paths) - covered - gapped)
        if missing_paths:
            gaps.append(
                FoldPlanGap(
                    tuple(
                        value.node_id
                        for value in child.nodes
                        if set(value.parameter_paths) & set(missing_paths)
                    ),
                    "Target parameters remain without an owning fold transaction: "
                    + ", ".join(missing_paths[:16]),
                    "complete_target_tensor_ownership",
                )
            )
        missing_nodes = sorted(
            set(value.node_id for value in child.nodes) - covered_nodes - gapped_nodes
        )
        if missing_nodes:
            gaps.append(
                FoldPlanGap(
                    tuple(missing_nodes),
                    "Target causal circuits remain without an owning fold transaction.",
                    "complete_target_circuit_ownership",
                )
            )
        return DynamicFoldPlan(
            parent_graph_fingerprints=tuple(value.fingerprint for value in parents),
            parent_state_fingerprints=tuple(sorted(parent_state_fingerprints.items())),
            child_graph_fingerprint=child.fingerprint,
            child_state_fingerprint=child_state_fingerprint,
            evidence_fingerprint=evidence_fingerprint,
            exact_target_unique_parameters=budget,
            actions=tuple(actions),
            gaps=tuple(gaps),
            registry_fingerprint=self.methods.fingerprint,
            ledger_fingerprint=self.ledger.fingerprint,
        )


def plan_dynamic_fold(
    parents: Sequence[ModelCircuitGraph],
    child: ModelCircuitGraph,
    *,
    child_state_fingerprint: str,
    evidence_fingerprint: str,
    parent_state_fingerprints: Mapping[str, str],
    exact_target_unique_parameters: int | None = None,
    methods: FoldMethodRegistry | None = None,
    ledger: ExperimentLedger | None = None,
    enabled_experimental_method_ids: Sequence[str] = (),
) -> DynamicFoldPlan:
    return DynamicFoldPlanner(
        methods=methods,
        ledger=ledger,
        enabled_experimental_method_ids=enabled_experimental_method_ids,
    ).plan(
        parents,
        child,
        child_state_fingerprint=child_state_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        parent_state_fingerprints=parent_state_fingerprints,
        exact_target_unique_parameters=exact_target_unique_parameters,
    )


__all__ = [
    "CAUSAL_ATTENTION_BRANCH_METHOD",
    "CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD",
    "CAUSAL_GATED_MLP_BRANCH_METHOD",
    "CAUSAL_KDA_RECURRENT_BRANCH_METHOD",
    "CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD",
    "CAUSAL_LAYER_BRANCH_METHODS",
    "CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD",
    "CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD",
    "CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD",
    "NATIVE_CAUSAL_LAYER_BRANCH_METHODS",
    "NATIVE_CHIMERA_LAYER_BRANCH_METHODS",
    "NATIVE_RESIDUAL_CHART_INTERFACE_METHOD",
    "DynamicFoldPlan",
    "DynamicFoldPlanner",
    "DynamicFoldPlanningError",
    "FoldAction",
    "FoldPlanGap",
    "PLAN_SCHEMA",
    "SourceCircuitRef",
    "plan_dynamic_fold",
    "validate_fold_action_evidence_channels",
    "validate_fold_method_contract",
]
