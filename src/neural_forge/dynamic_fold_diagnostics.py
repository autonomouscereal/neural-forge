"""Component diagnostics inside one complete exact-budget child fold.

The historical LFM2 ablation code installs compact operators in a parent
shell.  That is useful localization evidence, but it is not a standalone
compacted child.  This module describes the stricter experiment: discover
components from the common circuit IR, keep one exact compact-child topology,
fold every component, and substitute an alternate fold strategy only for the
selected focus.  A focus is never permission to leave the rest blank or to
publish a partial/hybrid model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .dynamic_circuit_ir import (
    CircuitKind,
    JsonScalar,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_json_bytes,
    canonical_sha256,
)
from .dynamic_fold_executor import (
    DynamicFoldRun,
    FoldOperatorRegistry,
)
from .dynamic_fold_planner import (
    DynamicFoldPlan,
    FoldAction,
    validate_fold_method_contract,
)
from .dynamic_fold_registry import (
    ExperimentLedger,
    FoldMethodRegistry,
)

SCHEMA = "neural-forge-whole-child-component-diagnostic-plan-v1"
BINDING_SCHEMA = "neural-forge-whole-child-component-diagnostic-binding-v1"
TRANSACTION_REPORT_BINDING_KEY = "whole_child_component_diagnostic_binding"
DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY = "whole_child_component_diagnostic_dispatch_json"
DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY = (
    "whole_child_component_diagnostic_base_action_id"
)
WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY = (
    "complete_whole_child_component_dispatch_v1"
)

_FORBIDDEN_CONFIGURATION_TOKENS = frozenset(
    {
        "adam",
        "adamw",
        "autograd",
        "backprop",
        "backward",
        "finetune",
        "gradient",
        "lora",
        "optimizer",
        "prune",
        "pruning",
        "quantization",
        "quantize",
        "sgd",
        "share",
        "shared",
        "sharing",
        "sidecar",
        "sparse",
        "train",
        "training",
    }
)


class WholeChildComponentDiagnosticError(ValueError):
    """The diagnostic no longer describes a complete compact child fold."""


def _forbidden_words(value: Any) -> tuple[str, ...]:
    words = {token for token in re.split(r"[^a-z0-9]+", str(value).casefold()) if token}
    return tuple(sorted(words & _FORBIDDEN_CONFIGURATION_TOKENS))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WholeChildComponentDiagnosticError(f"{label} must be nonempty.")
    return value.strip()


def _digest(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise WholeChildComponentDiagnosticError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return result


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(sorted(_text(value, label) for value in values))
    if len(result) != len(set(result)):
        raise WholeChildComponentDiagnosticError(f"{label} values must be unique.")
    return result


def _config(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
    *,
    enforce_fold_only: bool = False,
) -> tuple[tuple[str, JsonScalar], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    result = []
    for key, item in items:
        key = _text(key, "configuration key")
        if not isinstance(item, (str, int, float, bool, type(None))) or (
            isinstance(item, float) and not math.isfinite(item)
        ):
            raise WholeChildComponentDiagnosticError(
                f"Configuration {key!r} must be a finite JSON scalar."
            )
        if enforce_fold_only:
            forbidden = _forbidden_words(f"{key} {item}")
            safe_zero_gradient_declaration = (
                key.casefold() == "zero_gradient" and item is True
            )
            if forbidden and not safe_zero_gradient_declaration:
                raise WholeChildComponentDiagnosticError(
                    "Diagnostic strategy configuration requests a forbidden "
                    "non-fold mechanism: " + ", ".join(forbidden) + "."
                )
        result.append((key, item))
    result.sort()
    if len(result) != len({key for key, _item in result}):
        raise WholeChildComponentDiagnosticError("Configuration keys repeat.")
    return tuple(result)


def _tokens(role: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", role.casefold()))


def _projection_match(
    role: str,
    words: frozenset[str],
    fragments: tuple[str, ...],
) -> bool:
    lowered = role.casefold()
    return bool(_tokens(role).intersection(words)) or any(
        fragment in lowered for fragment in fragments
    )


def classify_diagnostic_parameter_semantic(
    circuit_kind: CircuitKind | str,
    parameter: ParameterSpec,
) -> str:
    """Classify a parameter from canonical IR kind/role, never module paths.

    Ambiguous fused attention tensors stay coupled; they are not falsely split
    into Q/K/V/O controls.
    """

    if not isinstance(parameter, ParameterSpec):
        raise WholeChildComponentDiagnosticError(
            "parameter must be an inspected ParameterSpec."
        )
    kind = (
        circuit_kind.value
        if isinstance(circuit_kind, CircuitKind)
        else _text(circuit_kind, "circuit_kind")
    )
    role = parameter.role
    lowered = role.casefold()
    if kind == CircuitKind.ATTENTION.value:
        if any(
            part in lowered
            for part in ("q_layernorm", "k_layernorm", "q_norm", "k_norm")
        ):
            return "attention_normalization"
        if any(
            part in lowered
            for part in ("qkv", "query_key_value", "c_attn", "in_proj_weight")
        ):
            return "attention_coupled_projection"
        candidates = []
        if _projection_match(
            role, frozenset({"q", "query", "wq"}), ("q_proj", "query_proj")
        ):
            candidates.append("attention_query")
        if _projection_match(
            role, frozenset({"k", "key", "wk"}), ("k_proj", "key_proj")
        ):
            candidates.append("attention_key")
        if _projection_match(
            role, frozenset({"v", "value", "wv"}), ("v_proj", "value_proj")
        ):
            candidates.append("attention_value")
        if _projection_match(
            role,
            frozenset({"o", "out", "output", "wo"}),
            ("o_proj", "out_proj", "output_proj", "c_proj"),
        ):
            candidates.append("attention_output")
        return candidates[0] if len(candidates) == 1 else "attention_coupled_projection"
    if kind in {CircuitKind.GATED_MLP.value, CircuitKind.DENSE_MLP.value}:
        writer = _projection_match(
            role,
            frozenset({"down", "out", "output", "writer", "w2", "wo", "fc2"}),
            ("down_proj", "out_proj", "output_proj"),
        )
        producer = _projection_match(
            role,
            frozenset({"gate", "up", "in", "input", "w1", "w3", "fc1"}),
            ("gate_proj", "up_proj", "in_proj", "input_proj"),
        )
        if writer != producer:
            return "mlp_writer" if writer else "mlp_producer"
        return "mlp_coupled_projection"
    by_kind = {
        CircuitKind.SHORT_CONV.value: "state_mixer",
        CircuitKind.RECURRENT_MIXER.value: "state_mixer",
        CircuitKind.NORM.value: "normalization",
        CircuitKind.TOKEN_EMBEDDING.value: "token_embedding",
        CircuitKind.OUTPUT_HEAD.value: "output_head",
        CircuitKind.POSITION_ENCODING.value: "position_encoding",
        CircuitKind.MODALITY_INPUT.value: "modality_input",
        CircuitKind.MODALITY_OUTPUT.value: "modality_output",
        CircuitKind.INTERFACE_PROJECTION.value: "interface_projection",
    }
    return by_kind.get(kind, f"other_{kind}")


def _structural_semantic(kind: str) -> str:
    if kind == CircuitKind.RESIDUAL_ADD.value:
        return "residual_routing"
    if kind == CircuitKind.POSITION_ENCODING.value:
        return "position_encoding"
    return f"structural_{kind}"


@dataclass(frozen=True, slots=True)
class DiagnosticFoldStrategy:
    """Registry- and implementation-bound executable fold strategy."""

    method_id: str
    method_version: int
    family_id: str
    method_spec_sha256: str
    operator_type: str
    operator_source_sha256: str
    operator_prepare_implementation_fingerprint: str
    operator_resource_fingerprint: str | None = None
    configuration: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        if (
            isinstance(self.method_version, bool)
            or not isinstance(self.method_version, int)
            or self.method_version <= 0
        ):
            raise WholeChildComponentDiagnosticError("method_version must be positive.")
        object.__setattr__(self, "family_id", _text(self.family_id, "family_id"))
        object.__setattr__(
            self,
            "method_spec_sha256",
            _digest(self.method_spec_sha256, "method_spec_sha256"),
        )
        object.__setattr__(
            self, "operator_type", _text(self.operator_type, "operator_type")
        )
        object.__setattr__(
            self,
            "operator_source_sha256",
            _digest(self.operator_source_sha256, "operator_source_sha256"),
        )
        object.__setattr__(
            self,
            "operator_prepare_implementation_fingerprint",
            _digest(
                self.operator_prepare_implementation_fingerprint,
                "operator_prepare_implementation_fingerprint",
            ),
        )
        if self.operator_resource_fingerprint is not None:
            object.__setattr__(
                self,
                "operator_resource_fingerprint",
                _digest(
                    self.operator_resource_fingerprint,
                    "operator_resource_fingerprint",
                ),
            )
        object.__setattr__(
            self,
            "configuration",
            _config(self.configuration, enforce_fold_only=True),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "family_id": self.family_id,
            "method_spec_sha256": self.method_spec_sha256,
            "operator_type": self.operator_type,
            "operator_source_sha256": self.operator_source_sha256,
            "operator_prepare_implementation_fingerprint": (
                self.operator_prepare_implementation_fingerprint
            ),
            "operator_resource_fingerprint": self.operator_resource_fingerprint,
            "operator_prepare_static_audit_binding_required_at_compile": True,
            "operator_capability": WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY,
            "configuration": dict(self.configuration),
        }


def _registered_strategy(
    method_id: str,
    configuration: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
) -> DiagnosticFoldStrategy:
    if not isinstance(methods, FoldMethodRegistry) or not isinstance(
        operators, FoldOperatorRegistry
    ):
        raise WholeChildComponentDiagnosticError(
            "Executable diagnostics require method and operator registries."
        )
    try:
        method = methods.require_selectable(
            _text(method_id, "method_id"), require_zero_gradient=True
        )
        if not method.standalone:
            raise WholeChildComponentDiagnosticError(
                f"Method {method.method_id!r} is not a standalone child fold."
            )
        forbidden_identity = _forbidden_words(f"{method.method_id} {method.family_id}")
        if forbidden_identity:
            raise WholeChildComponentDiagnosticError(
                f"Method {method.method_id!r} has a forbidden non-fold identity: "
                + ", ".join(forbidden_identity)
                + "."
            )
        operator, _prepare, binding = operators.authenticated_prepare(method.method_id)
    except (ValueError, RuntimeError) as exc:
        raise WholeChildComponentDiagnosticError(str(exc)) from exc
    if (
        getattr(operator, "whole_child_component_diagnostic_capability", None)
        != WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY
    ):
        raise WholeChildComponentDiagnosticError(
            f"Operator {method.method_id!r} does not declare the executable "
            "whole-child component-dispatch capability."
        )
    row = next(
        (
            value
            for value in operators.implementation_manifest
            if value["method_id"] == method.method_id
        ),
        None,
    )
    if row is None or not binding.static_zero_training_audit_passed:
        raise WholeChildComponentDiagnosticError(
            f"Operator {method.method_id!r} has no authenticated clean prepare."
        )
    return DiagnosticFoldStrategy(
        method_id=method.method_id,
        method_version=method.version,
        family_id=method.family_id,
        method_spec_sha256=method.fingerprint,
        operator_type=str(row["implementation_type"]),
        operator_source_sha256=str(row["source_sha256"]),
        operator_prepare_implementation_fingerprint=str(
            row["prepare_implementation_fingerprint"]
        ),
        operator_resource_fingerprint=row["resource_fingerprint"],
        configuration=_config(configuration, enforce_fold_only=True),
    )


@dataclass(frozen=True, slots=True)
class DiagnosticComponent:
    component_id: str
    semantics: tuple[str, ...]
    node_ids: tuple[str, ...]
    parameter_paths: tuple[str, ...]
    storage_groups: tuple[str, ...]
    layer_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "semantics": list(self.semantics),
            "node_ids": list(self.node_ids),
            "parameter_paths": list(self.parameter_paths),
            "storage_groups": list(self.storage_groups),
            "layer_indices": list(self.layer_indices),
            "structural_only": not self.parameter_paths,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticScope:
    scope_id: str
    scope_kind: str
    component_ids: tuple[str, ...]
    semantics: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "scope_kind": self.scope_kind,
            "component_ids": list(self.component_ids),
            "semantics": list(self.semantics),
        }


@dataclass(frozen=True, slots=True)
class WholeChildDiagnosticVariant:
    variant_id: str
    focus_scope_id: str | None
    component_strategy_fingerprints: tuple[tuple[str, str], ...]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "focus_scope_id": self.focus_scope_id,
            "component_strategy_fingerprints": dict(
                self.component_strategy_fingerprints
            ),
            "complete_whole_child_fold_required": True,
            "focus_is_strategy_substitution_not_partial_fold": True,
        }


@dataclass(frozen=True, slots=True)
class WholeChildComponentDiagnosticPlan:
    parent_graph_fingerprints: tuple[str, ...]
    parent_unique_parameters: int
    target_graph_fingerprint: str
    target_topology_fingerprint: str
    base_fold_plan_fingerprint: str
    blank_child_state_fingerprint: str
    exact_target_unique_parameters: int
    target_parameter_paths_sha256: str
    target_storage_groups_sha256: str
    include_layer_scopes: bool
    baseline_strategy: DiagnosticFoldStrategy
    focus_strategy: DiagnosticFoldStrategy
    components: tuple[DiagnosticComponent, ...]
    scopes: tuple[DiagnosticScope, ...]
    variants: tuple[WholeChildDiagnosticVariant, ...]

    @property
    def compression_ratio(self) -> float:
        return self.parent_unique_parameters / self.exact_target_unique_parameters

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def variant(self, variant_id: str) -> WholeChildDiagnosticVariant:
        for value in self.variants:
            if value.variant_id == variant_id:
                return value
        raise WholeChildComponentDiagnosticError(
            f"Unknown diagnostic variant {variant_id!r}."
        )

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "parent_graph_fingerprints": list(self.parent_graph_fingerprints),
            "parent_unique_parameters": self.parent_unique_parameters,
            "target_graph_fingerprint": self.target_graph_fingerprint,
            "target_topology_fingerprint": self.target_topology_fingerprint,
            "base_fold_plan_fingerprint": self.base_fold_plan_fingerprint,
            "blank_child_state_fingerprint": self.blank_child_state_fingerprint,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "source_to_target_parameter_ratio": self.compression_ratio,
            "target_parameter_paths_sha256": self.target_parameter_paths_sha256,
            "target_storage_groups_sha256": self.target_storage_groups_sha256,
            "include_layer_scopes": self.include_layer_scopes,
            "baseline_strategy": self.baseline_strategy.as_dict(),
            "focus_strategy": self.focus_strategy.as_dict(),
            "components": [value.as_dict() for value in self.components],
            "scopes": [value.as_dict() for value in self.scopes],
            "variants": [value.as_dict() for value in self.variants],
            "complete_whole_model_target_required": True,
            "same_exact_target_for_every_variant": True,
            "component_focus_is_not_partial_compaction": True,
            "parent_shell_hybrid_forbidden": True,
            "fold_only_execution_claim": (
                "unproven_until_a_completed_executor_run_is_validated"
            ),
            "required_execution_attestation": {
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "autograd_grad_calls": 0,
                "pruning_operations": 0,
                "quantization_operations": 0,
                "observed_parent_module_forward_calls": 0,
                "observed_parent_root_forward_calls": 0,
            },
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class WholeChildDiagnosticBinding:
    plan_sha256: str
    variant_id: str
    variant_sha256: str
    target_graph_fingerprint: str
    base_fold_plan_fingerprint: str
    blank_child_state_fingerprint: str
    exact_target_unique_parameters: int
    focus_scope_id: str | None
    assignments_sha256: str

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BINDING_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "variant_id": self.variant_id,
            "variant_sha256": self.variant_sha256,
            "target_graph_fingerprint": self.target_graph_fingerprint,
            "base_fold_plan_fingerprint": self.base_fold_plan_fingerprint,
            "blank_child_state_fingerprint": self.blank_child_state_fingerprint,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "focus_scope_id": self.focus_scope_id,
            "assignments_sha256": self.assignments_sha256,
            "complete_whole_child_fold_required": True,
        }


@dataclass(slots=True)
class _Fragment:
    semantics: set[str]
    nodes: set[str]
    paths: set[str]
    groups: set[str]
    layers: set[int]


def _discover_components(graph: ModelCircuitGraph) -> tuple[DiagnosticComponent, ...]:
    fragments: list[_Fragment] = []
    for node in graph.nodes:
        if not node.parameters:
            fragments.append(
                _Fragment(
                    {_structural_semantic(node.kind)},
                    {node.node_id},
                    set(),
                    set(),
                    set() if node.layer_index is None else {node.layer_index},
                )
            )
            continue
        by_semantic: dict[str, list[ParameterSpec]] = {}
        for parameter in node.parameters:
            semantic = classify_diagnostic_parameter_semantic(node.kind, parameter)
            by_semantic.setdefault(semantic, []).append(parameter)
        for semantic, parameters in by_semantic.items():
            fragments.append(
                _Fragment(
                    {semantic},
                    {node.node_id},
                    {value.path for value in parameters},
                    {value.storage_group for value in parameters},
                    set() if node.layer_index is None else {node.layer_index},
                )
            )

    # Storage aliases (notably tied embedding/head weights) are indivisible.
    parents = list(range(len(fragments)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    owner: dict[str, int] = {}
    for index, fragment in enumerate(fragments):
        for group in fragment.groups:
            union(owner.setdefault(group, index), index)
    merged: dict[int, _Fragment] = {}
    for index, fragment in enumerate(fragments):
        current = merged.setdefault(
            find(index), _Fragment(set(), set(), set(), set(), set())
        )
        current.semantics.update(fragment.semantics)
        current.nodes.update(fragment.nodes)
        current.paths.update(fragment.paths)
        current.groups.update(fragment.groups)
        current.layers.update(fragment.layers)

    components = []
    for value in merged.values():
        descriptor = {
            "semantics": sorted(value.semantics),
            "nodes": sorted(value.nodes),
            "paths": sorted(value.paths),
            "groups": sorted(value.groups),
        }
        label = "_and_".join(sorted(value.semantics))
        components.append(
            DiagnosticComponent(
                component_id=f"component.{label}.{canonical_sha256(descriptor)[:12]}",
                semantics=tuple(sorted(value.semantics)),
                node_ids=tuple(sorted(value.nodes)),
                parameter_paths=tuple(sorted(value.paths)),
                storage_groups=tuple(sorted(value.groups)),
                layer_indices=tuple(sorted(value.layers)),
            )
        )
    return tuple(sorted(components, key=lambda value: value.component_id))


_SUBSYSTEMS = {
    "attention": frozenset(
        {
            "attention_query",
            "attention_key",
            "attention_value",
            "attention_output",
            "attention_normalization",
            "attention_coupled_projection",
        }
    ),
    "state_mixer": frozenset({"state_mixer"}),
    "mlp": frozenset({"mlp_producer", "mlp_writer", "mlp_coupled_projection"}),
    "interfaces": frozenset(
        {
            "token_embedding",
            "output_head",
            "modality_input",
            "modality_output",
            "interface_projection",
        }
    ),
    "residual_and_norm": frozenset(
        {"normalization", "residual_routing", "position_encoding"}
    ),
}


def _scope(
    kind: str, label: str, values: Sequence[DiagnosticComponent]
) -> DiagnosticScope:
    component_ids = tuple(sorted(value.component_id for value in values))
    return DiagnosticScope(
        scope_id=f"scope.{kind}.{label}.{canonical_sha256(component_ids)[:12]}",
        scope_kind=kind,
        component_ids=component_ids,
        semantics=tuple(
            sorted({semantic for value in values for semantic in value.semantics})
        ),
    )


def _discover_scopes(
    components: Sequence[DiagnosticComponent],
    include_layer_scopes: bool,
) -> tuple[DiagnosticScope, ...]:
    # Parameterless residual/route nodes stay in the complete graph inventory,
    # but cannot honestly receive a distinct learned-state strategy.  An
    # edge-aware residual method may expose parameterized interface components;
    # until then, no parameterless "residual focus" variant is emitted.
    components = tuple(value for value in components if value.parameter_paths)
    proposed = []
    for semantics in sorted({value.semantics for value in components}):
        proposed.append(
            _scope(
                "semantic",
                "_and_".join(semantics),
                [value for value in components if value.semantics == semantics],
            )
        )
    for label, semantics in _SUBSYSTEMS.items():
        selected = [value for value in components if set(value.semantics) & semantics]
        if selected:
            proposed.append(_scope("subsystem", label, selected))
    if include_layer_scopes:
        for layer in sorted(
            {layer for value in components for layer in value.layer_indices}
        ):
            selected = [value for value in components if layer in value.layer_indices]
            proposed.append(_scope("layer", str(layer), selected))
            for label, semantics in _SUBSYSTEMS.items():
                subset = [
                    value for value in selected if set(value.semantics) & semantics
                ]
                if subset:
                    proposed.append(
                        _scope("layer_subsystem", f"{layer}_{label}", subset)
                    )
    unique: dict[tuple[str, ...], DiagnosticScope] = {}
    for value in proposed:
        unique.setdefault(value.component_ids, value)
    return tuple(sorted(unique.values(), key=lambda value: value.scope_id))


def _validate_base(
    parent_graphs: Sequence[ModelCircuitGraph],
    child: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
) -> None:
    parents = tuple(parent_graphs)
    if not parents or not isinstance(child, ModelCircuitGraph):
        raise WholeChildComponentDiagnosticError(
            "Diagnostics need inspected parent and child graphs."
        )
    if not isinstance(fold_plan, DynamicFoldPlan) or not fold_plan.ready:
        raise WholeChildComponentDiagnosticError("The base fold plan is not ready.")
    if (
        tuple(sorted(value.fingerprint for value in parents))
        != fold_plan.parent_graph_fingerprints
    ):
        raise WholeChildComponentDiagnosticError(
            "Parent graphs differ from the fold plan."
        )
    if (
        child.fingerprint != fold_plan.child_graph_fingerprint
        or child.unique_parameter_count != fold_plan.exact_target_unique_parameters
    ):
        raise WholeChildComponentDiagnosticError(
            "The child graph differs from the plan's exact target."
        )
    if (
        sum(value.unique_parameter_count for value in parents)
        <= child.unique_parameter_count
    ):
        raise WholeChildComponentDiagnosticError(
            "Diagnostics require a real positively compacted child."
        )
    planned_nodes = Counter(
        item for action in fold_plan.actions for item in action.target_node_ids
    )
    planned_paths = Counter(
        item for action in fold_plan.actions for item in action.target_parameter_paths
    )
    planned_groups = Counter(
        item for action in fold_plan.actions for item in action.target_storage_groups
    )
    target_groups = {
        parameter.storage_group for node in child.nodes for parameter in node.parameters
    }
    if (
        planned_nodes != Counter(value.node_id for value in child.nodes)
        or planned_paths != Counter(child.parameter_paths)
        or planned_groups != Counter({value: 1 for value in target_groups})
    ):
        raise WholeChildComponentDiagnosticError(
            "The base plan does not own every target node, parameter path, and "
            "learned-storage group exactly once."
        )
    if any(not action.signature.zero_gradient for action in fold_plan.actions):
        raise WholeChildComponentDiagnosticError(
            "The base plan contains a non-fold-only action."
        )
    for action in fold_plan.actions:
        if action.method_id != action.signature.method_id:
            raise WholeChildComponentDiagnosticError(
                "A base action method differs from its experiment signature."
            )
        if action.configuration != action.signature.configuration:
            raise WholeChildComponentDiagnosticError(
                "A base action configuration differs from its signature."
            )
        _config(action.configuration, enforce_fold_only=True)


def _expected_variants(
    components: Sequence[DiagnosticComponent],
    scopes: Sequence[DiagnosticScope],
    baseline: DiagnosticFoldStrategy,
    focus: DiagnosticFoldStrategy,
) -> tuple[WholeChildDiagnosticVariant, ...]:
    ids = tuple(value.component_id for value in components)
    result = [
        WholeChildDiagnosticVariant(
            "baseline.complete_whole_child",
            None,
            tuple((value, baseline.fingerprint) for value in ids),
        )
    ]
    for scope in scopes:
        focused = set(scope.component_ids)
        result.append(
            WholeChildDiagnosticVariant(
                f"focus.{scope.scope_id}",
                scope.scope_id,
                tuple(
                    (
                        value,
                        focus.fingerprint if value in focused else baseline.fingerprint,
                    )
                    for value in ids
                ),
            )
        )
    return tuple(sorted(result, key=lambda value: value.variant_id))


def build_whole_child_component_diagnostic_plan(
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
    *,
    baseline_method_id: str,
    focus_method_id: str,
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
    baseline_configuration: Mapping[str, JsonScalar]
    | Sequence[tuple[str, JsonScalar]] = (),
    focus_configuration: Mapping[str, JsonScalar]
    | Sequence[tuple[str, JsonScalar]] = (),
    include_layer_scopes: bool = False,
) -> WholeChildComponentDiagnosticPlan:
    """Build executable architecture-derived variants for one complete child."""

    _validate_base(parent_graphs, child_graph, fold_plan)
    baseline_strategy = _registered_strategy(
        baseline_method_id,
        baseline_configuration,
        methods,
        operators,
    )
    focus_strategy = _registered_strategy(
        focus_method_id,
        focus_configuration,
        methods,
        operators,
    )
    if baseline_strategy.fingerprint == focus_strategy.fingerprint:
        raise WholeChildComponentDiagnosticError("Strategies must be distinct.")
    if not isinstance(include_layer_scopes, bool):
        raise WholeChildComponentDiagnosticError(
            "include_layer_scopes must be boolean."
        )
    components = _discover_components(child_graph)
    paths = [path for value in components for path in value.parameter_paths]
    groups = [group for value in components for group in value.storage_groups]
    if (
        Counter(paths) != Counter(child_graph.parameter_paths)
        or Counter(groups)
        != Counter(
            {
                parameter.storage_group: 1
                for node in child_graph.nodes
                for parameter in node.parameters
            }
        )
        or {node for value in components for node in value.node_ids}
        != {node.node_id for node in child_graph.nodes}
    ):
        raise WholeChildComponentDiagnosticError(
            "Architecture discovery did not partition the complete child."
        )
    scopes = _discover_scopes(components, include_layer_scopes)
    result = WholeChildComponentDiagnosticPlan(
        parent_graph_fingerprints=tuple(
            sorted(value.fingerprint for value in parent_graphs)
        ),
        parent_unique_parameters=sum(
            value.unique_parameter_count for value in parent_graphs
        ),
        target_graph_fingerprint=child_graph.fingerprint,
        target_topology_fingerprint=child_graph.structural_topology_fingerprint,
        base_fold_plan_fingerprint=fold_plan.fingerprint,
        blank_child_state_fingerprint=fold_plan.child_state_fingerprint,
        exact_target_unique_parameters=child_graph.unique_parameter_count,
        target_parameter_paths_sha256=canonical_sha256(
            sorted(child_graph.parameter_paths)
        ),
        target_storage_groups_sha256=canonical_sha256(sorted(groups)),
        include_layer_scopes=include_layer_scopes,
        baseline_strategy=baseline_strategy,
        focus_strategy=focus_strategy,
        components=components,
        scopes=scopes,
        variants=_expected_variants(
            components, scopes, baseline_strategy, focus_strategy
        ),
    )
    return validate_whole_child_component_diagnostic_plan(
        result, parent_graphs, child_graph, fold_plan
    )


def validate_whole_child_component_diagnostic_plan(
    diagnostic: WholeChildComponentDiagnosticPlan,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
) -> WholeChildComponentDiagnosticPlan:
    """Re-derive the component partition and reject stale/tampered plans."""

    if not isinstance(diagnostic, WholeChildComponentDiagnosticPlan):
        raise WholeChildComponentDiagnosticError("diagnostic has the wrong type.")
    _validate_base(parent_graphs, child_graph, fold_plan)
    components = _discover_components(child_graph)
    scopes = _discover_scopes(components, diagnostic.include_layer_scopes)
    groups = [group for value in components for group in value.storage_groups]
    expected = {
        "parent_graph_fingerprints": tuple(
            sorted(value.fingerprint for value in parent_graphs)
        ),
        "parent_unique_parameters": sum(
            value.unique_parameter_count for value in parent_graphs
        ),
        "target_graph_fingerprint": child_graph.fingerprint,
        "target_topology_fingerprint": child_graph.structural_topology_fingerprint,
        "base_fold_plan_fingerprint": fold_plan.fingerprint,
        "blank_child_state_fingerprint": fold_plan.child_state_fingerprint,
        "exact_target_unique_parameters": child_graph.unique_parameter_count,
        "target_parameter_paths_sha256": canonical_sha256(
            sorted(child_graph.parameter_paths)
        ),
        "target_storage_groups_sha256": canonical_sha256(sorted(groups)),
        "components": components,
        "scopes": scopes,
        "variants": _expected_variants(
            components,
            scopes,
            diagnostic.baseline_strategy,
            diagnostic.focus_strategy,
        ),
    }
    for name, value in expected.items():
        if getattr(diagnostic, name) != value:
            raise WholeChildComponentDiagnosticError(
                f"Diagnostic {name} differs from authenticated compiler inputs."
            )
    return diagnostic


def _validate_live_strategy(
    strategy: DiagnosticFoldStrategy,
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
) -> None:
    observed = _registered_strategy(
        strategy.method_id,
        strategy.configuration,
        methods,
        operators,
    )
    if observed != strategy:
        raise WholeChildComponentDiagnosticError(
            f"Strategy {strategy.method_id!r} differs from the live authenticated "
            "method/operator registries."
        )


def _focus_components(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant: WholeChildDiagnosticVariant,
) -> tuple[DiagnosticComponent, ...]:
    if variant.focus_scope_id is None:
        return ()
    scope = next(
        value for value in diagnostic.scopes if value.scope_id == variant.focus_scope_id
    )
    component_ids = set(scope.component_ids)
    return tuple(
        value for value in diagnostic.components if value.component_id in component_ids
    )


def _action_strategy(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant: WholeChildDiagnosticVariant,
    action: FoldAction,
) -> DiagnosticFoldStrategy:
    focused = _focus_components(diagnostic, variant)
    focus_paths = {path for component in focused for path in component.parameter_paths}
    focus_groups = {
        group for component in focused for group in component.storage_groups
    }
    if focus_paths.intersection(action.target_parameter_paths) or (
        focus_groups.intersection(action.target_storage_groups)
    ):
        return diagnostic.focus_strategy
    return diagnostic.baseline_strategy


def _diagnostic_dispatch(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant: WholeChildDiagnosticVariant,
    binding: WholeChildDiagnosticBinding,
) -> dict[str, Any]:
    assignments = dict(variant.component_strategy_fingerprints)
    return {
        "schema": "neural-forge-whole-child-component-dispatch-v1",
        "binding": binding.as_dict(),
        "assignments": assignments,
        "strategies": {
            diagnostic.baseline_strategy.fingerprint: (
                diagnostic.baseline_strategy.as_dict()
            ),
            diagnostic.focus_strategy.fingerprint: diagnostic.focus_strategy.as_dict(),
        },
        "components": [
            {
                **component.as_dict(),
                "assigned_strategy_fingerprint": assignments[component.component_id],
            }
            for component in diagnostic.components
        ],
        "complete_whole_child_fold_required": True,
        "exact_target_unique_parameters": (diagnostic.exact_target_unique_parameters),
    }


def _source_groups_for_action(
    action: FoldAction,
    parent_graphs: Sequence[ModelCircuitGraph],
) -> tuple[tuple[ModelCircuitGraph, tuple[Any, ...]], ...]:
    by_name = {value.model_name: value for value in parent_graphs}
    if len(by_name) != len(tuple(parent_graphs)):
        raise WholeChildComponentDiagnosticError(
            "Parent graph model names must be unique."
        )
    result = []
    for source in action.sources:
        try:
            graph = by_name[source.parent_name]
        except KeyError as exc:
            raise WholeChildComponentDiagnosticError(
                f"Unknown source parent {source.parent_name!r}."
            ) from exc
        nodes = {value.node_id: value for value in graph.nodes}
        try:
            selected = tuple(nodes[value] for value in source.node_ids)
        except KeyError as exc:
            raise WholeChildComponentDiagnosticError(
                "A fold action names an unknown source node."
            ) from exc
        result.append((graph, selected))
    return tuple(result)


def compile_whole_child_component_diagnostic_variant(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    base_fold_plan: DynamicFoldPlan,
    *,
    methods: FoldMethodRegistry,
    operators: FoldOperatorRegistry,
    ledger: ExperimentLedger,
) -> DynamicFoldPlan:
    """Compile one diagnostic into the actual executor plan it will run."""

    validate_whole_child_component_diagnostic_plan(
        diagnostic, parent_graphs, child_graph, base_fold_plan
    )
    if not isinstance(ledger, ExperimentLedger):
        raise WholeChildComponentDiagnosticError("ledger must be ExperimentLedger.")
    _validate_live_strategy(diagnostic.baseline_strategy, methods, operators)
    _validate_live_strategy(diagnostic.focus_strategy, methods, operators)
    variant = diagnostic.variant(_text(variant_id, "variant_id"))
    binding = bind_whole_child_component_diagnostic_variant(
        diagnostic,
        variant.variant_id,
        parent_graphs,
        child_graph,
        base_fold_plan,
    )
    dispatch_json = canonical_json_bytes(
        _diagnostic_dispatch(diagnostic, variant, binding)
    ).decode("ascii")
    action_ids: dict[str, str] = {}
    action_strategies: dict[str, DiagnosticFoldStrategy] = {}
    for action in base_fold_plan.actions:
        strategy = _action_strategy(diagnostic, variant, action)
        action_strategies[action.action_id] = strategy
        action_ids[action.action_id] = (
            "whole-child-diagnostic."
            + canonical_sha256(
                {
                    "binding": binding.fingerprint,
                    "base_action": action.action_id,
                    "method_id": strategy.method_id,
                }
            )[:20]
        )

    target_nodes = {value.node_id: value for value in child_graph.nodes}
    actions = []
    for action in base_fold_plan.actions:
        strategy = action_strategies[action.action_id]
        method = methods.require_selectable(
            strategy.method_id, require_zero_gradient=True
        )
        if method.fingerprint != strategy.method_spec_sha256:
            raise WholeChildComponentDiagnosticError(
                f"Method contract for {method.method_id!r} changed before compile."
            )
        reserved = {
            DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY,
            DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY,
            "whole_child_component_diagnostic_selected_strategy_sha256",
        }
        base_configuration = dict(action.configuration)
        collision = reserved.intersection(base_configuration)
        if collision:
            raise WholeChildComponentDiagnosticError(
                "Base action uses compiler-reserved diagnostic keys: "
                + ", ".join(sorted(collision))
                + "."
            )
        base_configuration.update(
            {
                DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY: action.action_id,
                DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY: dispatch_json,
                "whole_child_component_diagnostic_selected_strategy_sha256": (
                    strategy.fingerprint
                ),
            }
        )
        configuration = _config(base_configuration)
        signature = replace(
            action.signature,
            method_id=method.method_id,
            method_version=method.version,
            family_id=method.family_id,
            selection_policy=(
                "architecture_driven_complete_whole_child_component_dispatch"
            ),
            fit_objective="complete_whole_child_component_fold",
            zero_gradient=True,
            configuration=configuration,
        )
        compiled = replace(
            action,
            action_id=action_ids[action.action_id],
            method_id=method.method_id,
            dependencies=tuple(action_ids[value] for value in action.dependencies),
            evidence_requirements=method.evidence_channels,
            configuration=configuration,
            signature=signature,
            ledger_decision=ledger.decision(signature),
        )
        try:
            selected_targets = tuple(
                target_nodes[value] for value in compiled.target_node_ids
            )
        except KeyError as exc:
            raise WholeChildComponentDiagnosticError(
                "A compiled action names an unknown target node."
            ) from exc
        validate_fold_method_contract(
            method,
            source_groups=_source_groups_for_action(compiled, parent_graphs),
            target_graph=child_graph,
            target_nodes=selected_targets,
        )
        actions.append(compiled)

    result = DynamicFoldPlan(
        parent_graph_fingerprints=base_fold_plan.parent_graph_fingerprints,
        parent_state_fingerprints=base_fold_plan.parent_state_fingerprints,
        child_graph_fingerprint=base_fold_plan.child_graph_fingerprint,
        child_state_fingerprint=base_fold_plan.child_state_fingerprint,
        evidence_fingerprint=base_fold_plan.evidence_fingerprint,
        exact_target_unique_parameters=base_fold_plan.exact_target_unique_parameters,
        actions=tuple(actions),
        gaps=(),
        registry_fingerprint=methods.fingerprint,
        ledger_fingerprint=ledger.fingerprint,
    )
    if not result.ready:
        raise WholeChildComponentDiagnosticError(
            "The compiled diagnostic is blocked by the experiment ledger."
        )
    if (
        Counter(value for action in result.actions for value in action.target_node_ids)
        != Counter(value.node_id for value in child_graph.nodes)
        or Counter(
            value
            for action in result.actions
            for value in action.target_parameter_paths
        )
        != Counter(child_graph.parameter_paths)
        or Counter(
            value for action in result.actions for value in action.target_storage_groups
        )
        != Counter(
            {
                parameter.storage_group: 1
                for node in child_graph.nodes
                for parameter in node.parameters
            }
        )
    ):
        raise WholeChildComponentDiagnosticError(
            "Compilation changed exact whole-child ownership."
        )
    return result


def bind_whole_child_component_diagnostic_variant(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    fold_plan: DynamicFoldPlan,
) -> WholeChildDiagnosticBinding:
    """Return the exact provenance an operator must seal in its transaction."""

    validate_whole_child_component_diagnostic_plan(
        diagnostic, parent_graphs, child_graph, fold_plan
    )
    variant = diagnostic.variant(_text(variant_id, "variant_id"))
    return WholeChildDiagnosticBinding(
        plan_sha256=diagnostic.fingerprint,
        variant_id=variant.variant_id,
        variant_sha256=variant.fingerprint,
        target_graph_fingerprint=child_graph.fingerprint,
        base_fold_plan_fingerprint=fold_plan.fingerprint,
        blank_child_state_fingerprint=fold_plan.child_state_fingerprint,
        exact_target_unique_parameters=child_graph.unique_parameter_count,
        focus_scope_id=variant.focus_scope_id,
        assignments_sha256=canonical_sha256(
            dict(variant.component_strategy_fingerprints)
        ),
    )


def validate_whole_child_diagnostic_transaction_report(
    report: Mapping[str, Any],
    binding: WholeChildDiagnosticBinding,
) -> None:
    """Fail closed unless an operator reports the exact selected binding."""

    if (
        not isinstance(report, Mapping)
        or report.get(TRANSACTION_REPORT_BINDING_KEY) != binding.as_dict()
    ):
        raise WholeChildComponentDiagnosticError(
            "Fold transaction did not seal the whole-child diagnostic binding."
        )


def validate_compiled_whole_child_component_diagnostic_variant(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    compiled_plan: DynamicFoldPlan,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    base_fold_plan: DynamicFoldPlan,
) -> WholeChildDiagnosticBinding:
    """Authenticate compiler-owned dispatch without trusting plan metadata."""

    validate_whole_child_component_diagnostic_plan(
        diagnostic, parent_graphs, child_graph, base_fold_plan
    )
    if not isinstance(compiled_plan, DynamicFoldPlan) or not compiled_plan.ready:
        raise WholeChildComponentDiagnosticError(
            "The compiled diagnostic plan is not executable."
        )
    variant = diagnostic.variant(_text(variant_id, "variant_id"))
    binding = bind_whole_child_component_diagnostic_variant(
        diagnostic,
        variant.variant_id,
        parent_graphs,
        child_graph,
        base_fold_plan,
    )
    dispatch_json = canonical_json_bytes(
        _diagnostic_dispatch(diagnostic, variant, binding)
    ).decode("ascii")
    if (
        compiled_plan.parent_graph_fingerprints
        != base_fold_plan.parent_graph_fingerprints
        or compiled_plan.parent_state_fingerprints
        != base_fold_plan.parent_state_fingerprints
        or compiled_plan.child_graph_fingerprint != child_graph.fingerprint
        or compiled_plan.child_state_fingerprint
        != diagnostic.blank_child_state_fingerprint
        or compiled_plan.evidence_fingerprint != base_fold_plan.evidence_fingerprint
        or compiled_plan.exact_target_unique_parameters
        != diagnostic.exact_target_unique_parameters
    ):
        raise WholeChildComponentDiagnosticError(
            "The compiled variant changed the authenticated parents, blank child, "
            "evidence root, topology, or exact budget."
        )
    by_base: dict[str, FoldAction] = {}
    for action in compiled_plan.actions:
        configuration = dict(action.configuration)
        base_id = configuration.get(DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY)
        if not isinstance(base_id, str) or base_id in by_base:
            raise WholeChildComponentDiagnosticError(
                "Compiled actions do not map one-to-one onto base actions."
            )
        by_base[base_id] = action
    if set(by_base) != {value.action_id for value in base_fold_plan.actions}:
        raise WholeChildComponentDiagnosticError(
            "Compiled actions do not cover each base action exactly once."
        )
    for base_action in base_fold_plan.actions:
        action = by_base[base_action.action_id]
        strategy = _action_strategy(diagnostic, variant, base_action)
        configuration = dict(action.configuration)
        if (
            action.method_id != strategy.method_id
            or action.signature.method_id != strategy.method_id
            or action.signature.method_version != strategy.method_version
            or action.signature.family_id != strategy.family_id
            or action.signature.zero_gradient is not True
            or action.configuration != action.signature.configuration
            or configuration.get(DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY) != dispatch_json
            or configuration.get(
                "whole_child_component_diagnostic_selected_strategy_sha256"
            )
            != strategy.fingerprint
        ):
            raise WholeChildComponentDiagnosticError(
                "A compiled action does not dispatch its authenticated strategy."
            )
        for key, value in base_action.configuration:
            if configuration.get(key) != value:
                raise WholeChildComponentDiagnosticError(
                    "Compilation changed base fold configuration."
                )
        for name in (
            "sources",
            "target_node_ids",
            "target_parameter_paths",
            "target_storage_groups",
            "source_logical_parameters",
            "target_logical_parameters",
        ):
            if getattr(action, name) != getattr(base_action, name):
                raise WholeChildComponentDiagnosticError(
                    f"Compilation changed base action {name}."
                )
        expected_dependencies = tuple(
            sorted(by_base[value].action_id for value in base_action.dependencies)
        )
        if action.dependencies != expected_dependencies:
            raise WholeChildComponentDiagnosticError(
                "Compilation changed base action dependencies."
            )
    if (
        Counter(
            value
            for action in compiled_plan.actions
            for value in action.target_node_ids
        )
        != Counter(value.node_id for value in child_graph.nodes)
        or Counter(
            value
            for action in compiled_plan.actions
            for value in action.target_parameter_paths
        )
        != Counter(child_graph.parameter_paths)
        or Counter(
            value
            for action in compiled_plan.actions
            for value in action.target_storage_groups
        )
        != Counter(
            {
                parameter.storage_group: 1
                for node in child_graph.nodes
                for parameter in node.parameters
            }
        )
    ):
        raise WholeChildComponentDiagnosticError(
            "Compiled ownership is not exact for the complete child."
        )
    return binding


def validate_executed_whole_child_component_diagnostic_run(
    diagnostic: WholeChildComponentDiagnosticPlan,
    variant_id: str,
    compiled_plan: DynamicFoldPlan,
    run: DynamicFoldRun,
    parent_graphs: Sequence[ModelCircuitGraph],
    child_graph: ModelCircuitGraph,
    base_fold_plan: DynamicFoldPlan,
) -> DynamicFoldRun:
    """Promote fold-only claims only from an observed completed executor run."""

    binding = validate_compiled_whole_child_component_diagnostic_variant(
        diagnostic,
        variant_id,
        compiled_plan,
        parent_graphs,
        child_graph,
        base_fold_plan,
    )
    if not isinstance(run, DynamicFoldRun):
        raise WholeChildComponentDiagnosticError("run must be DynamicFoldRun.")
    if not run.complete or run.plan_fingerprint != compiled_plan.fingerprint:
        raise WholeChildComponentDiagnosticError(
            "The executor did not complete the exact compiled diagnostic plan."
        )
    attempts = {value.action_id: value for value in run.attempts}
    if (
        len(attempts) != len(run.attempts)
        or set(attempts) != {value.action_id for value in compiled_plan.actions}
        or set(run.accepted_action_ids) != set(attempts)
    ):
        raise WholeChildComponentDiagnosticError(
            "The run did not accept each compiled action exactly once."
        )
    for action in compiled_plan.actions:
        attempt = attempts[action.action_id]
        base_id = dict(action.configuration)[DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY]
        base_action = next(
            value for value in base_fold_plan.actions if value.action_id == base_id
        )
        strategy = _action_strategy(
            diagnostic,
            diagnostic.variant(variant_id),
            base_action,
        )
        if attempt.method_id != action.method_id or not attempt.accepted:
            raise WholeChildComponentDiagnosticError(
                "An executor attempt used the wrong registered method."
            )
        if (
            attempt.operator_prepare_implementation_fingerprint
            != strategy.operator_prepare_implementation_fingerprint
        ):
            raise WholeChildComponentDiagnosticError(
                "The executor used a different operator implementation than the "
                "authenticated diagnostic strategy."
            )
        if (
            not attempt.operator_prepare_static_zero_training_audit_passed
            or not attempt.transaction_static_zero_training_audit_passed
        ):
            raise WholeChildComponentDiagnosticError(
                "Executed fold callables failed source-visible static audit."
            )
        validate_whole_child_diagnostic_transaction_report(
            attempt.transaction_report,
            binding,
        )
    counters = {
        "optimizer_constructions": run.optimizer_constructions,
        "optimizer_steps": run.optimizer_steps,
        "backward_calls": run.backward_calls,
        "autograd_grad_calls": run.autograd_grad_calls,
        "pruning_operations": run.pruning_operations,
        "quantization_operations": run.quantization_operations,
        "observed_parent_module_forward_calls": (
            run.observed_parent_module_forward_calls
        ),
        "observed_parent_root_forward_calls": run.observed_parent_root_forward_calls,
    }
    nonzero = {key: value for key, value in counters.items() if value != 0}
    if nonzero:
        raise WholeChildComponentDiagnosticError(
            "Observed execution violated fold-only/cached-parent requirements: "
            + ", ".join(f"{key}={value}" for key, value in sorted(nonzero.items()))
            + "."
        )
    if not run.executed_fold_static_zero_training_audit_passed:
        raise WholeChildComponentDiagnosticError(
            "The executed fold implementation manifest is not statically clean."
        )
    if Counter(run.completed_target_parameter_paths) != Counter(
        child_graph.parameter_paths
    ):
        raise WholeChildComponentDiagnosticError(
            "The run did not complete every exact child parameter path once."
        )
    if run.final_state_fingerprint == run.initial_state_fingerprint:
        raise WholeChildComponentDiagnosticError(
            "The completed diagnostic installed no learned-state delta."
        )
    return run


__all__ = [
    "DIAGNOSTIC_BASE_ACTION_CONFIGURATION_KEY",
    "DIAGNOSTIC_DISPATCH_CONFIGURATION_KEY",
    "TRANSACTION_REPORT_BINDING_KEY",
    "WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY",
    "DiagnosticComponent",
    "DiagnosticFoldStrategy",
    "DiagnosticScope",
    "WholeChildComponentDiagnosticError",
    "WholeChildComponentDiagnosticPlan",
    "WholeChildDiagnosticBinding",
    "WholeChildDiagnosticVariant",
    "bind_whole_child_component_diagnostic_variant",
    "build_whole_child_component_diagnostic_plan",
    "classify_diagnostic_parameter_semantic",
    "compile_whole_child_component_diagnostic_variant",
    "validate_compiled_whole_child_component_diagnostic_variant",
    "validate_executed_whole_child_component_diagnostic_run",
    "validate_whole_child_component_diagnostic_plan",
    "validate_whole_child_diagnostic_transaction_report",
]
