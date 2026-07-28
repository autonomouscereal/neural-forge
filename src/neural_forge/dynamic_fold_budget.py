"""Exact family-neutral topology budgeting for the folding compiler.

This module extracts the bounded exact-search contract proven in the Qwen
allocator into an architecture-neutral form.  A topology adapter supplies
legal implementation options for every mandatory child role.  The solver
chooses exactly one option per role, counts shared learned resources once,
honors dependencies and conflicts, and refuses approximation when its
immutable state ceiling is exceeded.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dynamic_circuit_ir import (
    CircuitKind,
    JsonScalar,
    ModelCircuitGraph,
    canonical_sha256,
)


SCHEMA = "neural-forge-dynamic-topology-budget-v1"
DISCOVERED_COMPONENT_SCHEMA = (
    "neural-forge-architecture-discovered-component-budget-v1"
)
COMPONENT_RETENTION_RISK_POLICY = (
    "semantic_role_fractional_loss_geometric_balance_v1"
)

SEMANTIC_STATE_MIXER_INTERNAL = "state_mixer_internal"
SEMANTIC_ATTENTION_QUERY_KEY = "attention_query_key"
SEMANTIC_ATTENTION_WRITER = "attention_writer"
SEMANTIC_MLP_PRODUCER = "mlp_producer"
SEMANTIC_MLP_WRITER = "mlp_writer"
SEMANTIC_NORMALIZATION = "normalization"
SEMANTIC_INTERFACE = "interface"
SEMANTIC_OTHER = "other"

_COMPONENT_SEMANTICS = frozenset(
    {
        SEMANTIC_STATE_MIXER_INTERNAL,
        SEMANTIC_ATTENTION_QUERY_KEY,
        SEMANTIC_ATTENTION_WRITER,
        SEMANTIC_MLP_PRODUCER,
        SEMANTIC_MLP_WRITER,
        SEMANTIC_NORMALIZATION,
        SEMANTIC_INTERFACE,
        SEMANTIC_OTHER,
    }
)
_INTERNAL_COMPONENT_SEMANTICS = (
    SEMANTIC_STATE_MIXER_INTERNAL,
    SEMANTIC_ATTENTION_QUERY_KEY,
    SEMANTIC_ATTENTION_WRITER,
    SEMANTIC_MLP_PRODUCER,
    SEMANTIC_MLP_WRITER,
    SEMANTIC_NORMALIZATION,
)


class DynamicFoldBudgetError(ValueError):
    """The topology budget problem or one of its options is invalid."""


class DynamicFoldBudgetUnattainable(DynamicFoldBudgetError):
    """No legal exact topology exists within the bounded search."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicFoldBudgetError(f"{name} must be a nonempty string.")
    return value.strip()


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicFoldBudgetError(f"{name} must be a non-negative integer.")
    return value


def _positive(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if result == 0:
        raise DynamicFoldBudgetError(f"{name} must be positive.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise DynamicFoldBudgetError(f"{name} must be finite and non-negative.")
    return float(value)


def _strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(sorted({_text(value, name) for value in values}))
    return result


def _metadata(
    values: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    if values is None:
        return ()
    items = values.items() if isinstance(values, Mapping) else values
    result = []
    for key, value in items:
        key = _text(key, "metadata key")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise DynamicFoldBudgetError(f"metadata.{key} must be a JSON scalar.")
        if isinstance(value, float) and not math.isfinite(value):
            raise DynamicFoldBudgetError(f"metadata.{key} must be finite.")
        result.append((key, value))
    result.sort(key=lambda item: item[0])
    if len({key for key, _value in result}) != len(result):
        raise DynamicFoldBudgetError("Metadata keys must be unique.")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SharedTopologyResource:
    resource_id: str
    unique_parameters: int
    serialized_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_id", _text(self.resource_id, "resource_id"))
        object.__setattr__(
            self,
            "unique_parameters",
            _nonnegative(self.unique_parameters, "unique_parameters"),
        )
        object.__setattr__(
            self,
            "serialized_bytes",
            _nonnegative(self.serialized_bytes, "serialized_bytes"),
        )
        if self.unique_parameters == 0 and self.serialized_bytes == 0:
            raise DynamicFoldBudgetError(
                "A shared resource must contribute parameters or bytes."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "unique_parameters": self.unique_parameters,
            "serialized_bytes": self.serialized_bytes,
        }


@dataclass(frozen=True, slots=True)
class TopologyOption:
    """One legal implementation of one mandatory child topology role."""

    option_id: str
    group_id: str
    local_unique_parameters: int
    local_serialized_bytes: int
    shared_resource_ids: tuple[str, ...] = ()
    dependency_option_ids: tuple[str, ...] = ()
    conflict_keys: tuple[str, ...] = ()
    predicted_risk: float = 0.0
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", _text(self.option_id, "option_id"))
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id"))
        for name in (
            "local_unique_parameters",
            "local_serialized_bytes",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "shared_resource_ids",
            "dependency_option_ids",
            "conflict_keys",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if self.option_id in self.dependency_option_ids:
            raise DynamicFoldBudgetError("An option cannot depend on itself.")
        object.__setattr__(
            self,
            "predicted_risk",
            _finite_nonnegative(self.predicted_risk, "predicted_risk"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "group_id": self.group_id,
            "local_unique_parameters": self.local_unique_parameters,
            "local_serialized_bytes": self.local_serialized_bytes,
            "shared_resource_ids": list(self.shared_resource_ids),
            "dependency_option_ids": list(self.dependency_option_ids),
            "conflict_keys": list(self.conflict_keys),
            "predicted_risk": self.predicted_risk,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TopologyChoiceGroup:
    group_id: str
    options: tuple[TopologyOption, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id"))
        options = tuple(sorted(self.options, key=lambda value: value.option_id))
        if not options or any(
            not isinstance(value, TopologyOption) for value in options
        ):
            raise DynamicFoldBudgetError(
                "A topology choice group needs at least one option."
            )
        if any(value.group_id != self.group_id for value in options):
            raise DynamicFoldBudgetError("Every option must name its containing group.")
        if len({value.option_id for value in options}) != len(options):
            raise DynamicFoldBudgetError("Option IDs must be unique in a group.")
        object.__setattr__(self, "options", options)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "options": [value.as_dict() for value in self.options],
        }


@dataclass(frozen=True, slots=True)
class ExactTopologyBudgetProblem:
    base_unique_parameters: int
    base_serialized_bytes: int
    exact_target_unique_parameters: int
    groups: tuple[TopologyChoiceGroup, ...]
    shared_resources: tuple[SharedTopologyResource, ...] = ()
    max_states: int = 100_000

    def __post_init__(self) -> None:
        for name in ("base_unique_parameters", "base_serialized_bytes"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "exact_target_unique_parameters",
            _positive(
                self.exact_target_unique_parameters,
                "exact_target_unique_parameters",
            ),
        )
        object.__setattr__(self, "max_states", _positive(self.max_states, "max_states"))
        groups = tuple(sorted(self.groups, key=lambda value: value.group_id))
        if not groups or any(
            not isinstance(value, TopologyChoiceGroup) for value in groups
        ):
            raise DynamicFoldBudgetError(
                "An exact topology problem needs choice groups."
            )
        if len({value.group_id for value in groups}) != len(groups):
            raise DynamicFoldBudgetError("Topology group IDs must be unique.")
        object.__setattr__(self, "groups", groups)
        resources = tuple(
            sorted(self.shared_resources, key=lambda value: value.resource_id)
        )
        if any(not isinstance(value, SharedTopologyResource) for value in resources):
            raise DynamicFoldBudgetError(
                "shared_resources must contain SharedTopologyResource objects."
            )
        if len({value.resource_id for value in resources}) != len(resources):
            raise DynamicFoldBudgetError("Shared resource IDs must be unique.")
        object.__setattr__(self, "shared_resources", resources)

        options = tuple(option for group in groups for option in group.options)
        option_ids = {value.option_id for value in options}
        if len(option_ids) != len(options):
            raise DynamicFoldBudgetError("Topology option IDs must be globally unique.")
        resource_ids = {value.resource_id for value in resources}
        for option in options:
            unknown_resources = set(option.shared_resource_ids) - resource_ids
            unknown_dependencies = set(option.dependency_option_ids) - option_ids
            if unknown_resources:
                raise DynamicFoldBudgetError(
                    f"Option {option.option_id!r} references unknown resources: "
                    f"{sorted(unknown_resources)}."
                )
            if unknown_dependencies:
                raise DynamicFoldBudgetError(
                    f"Option {option.option_id!r} references unknown dependencies: "
                    f"{sorted(unknown_dependencies)}."
                )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "base_unique_parameters": self.base_unique_parameters,
            "base_serialized_bytes": self.base_serialized_bytes,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "groups": [value.as_dict() for value in self.groups],
            "shared_resources": [value.as_dict() for value in self.shared_resources],
            "max_states": self.max_states,
        }


@dataclass(frozen=True, slots=True)
class ExactTopologyBudgetSelection:
    problem_fingerprint: str
    selected_option_ids: tuple[str, ...]
    shared_resource_ids: tuple[str, ...]
    exact_unique_parameters: int
    serialized_bytes: int
    total_predicted_risk: float
    states_considered: int
    peak_retained_states: int

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "problem_fingerprint": self.problem_fingerprint,
            "selected_option_ids": list(self.selected_option_ids),
            "shared_resource_ids": list(self.shared_resource_ids),
            "exact_unique_parameters": self.exact_unique_parameters,
            "serialized_bytes": self.serialized_bytes,
            "total_predicted_risk": self.total_predicted_risk,
            "states_considered": self.states_considered,
            "peak_retained_states": self.peak_retained_states,
            "exact_budget_met": True,
            "one_option_per_mandatory_group": True,
            "shared_resources_counted_once": True,
        }


@dataclass(frozen=True, slots=True)
class _State:
    selected_option_ids: tuple[str, ...]
    shared_resource_ids: frozenset[str]
    conflict_keys: frozenset[str]
    unique_parameters: int
    serialized_bytes: int
    risk: float


def solve_exact_topology_budget(
    problem: ExactTopologyBudgetProblem,
) -> ExactTopologyBudgetSelection:
    """Solve the exact multi-choice topology budget or fail closed."""

    if not isinstance(problem, ExactTopologyBudgetProblem):
        raise DynamicFoldBudgetError("problem must be ExactTopologyBudgetProblem.")
    resource_by_id = {value.resource_id: value for value in problem.shared_resources}
    option_group_index = {
        option.option_id: index
        for index, group in enumerate(problem.groups)
        for option in group.options
    }
    states = [
        _State(
            (),
            frozenset(),
            frozenset(),
            problem.base_unique_parameters,
            problem.base_serialized_bytes,
            0.0,
        )
    ]
    considered = 1
    peak = 1
    for group_index, group in enumerate(problem.groups):
        next_states: list[_State] = []
        for state in states:
            selected = frozenset(state.selected_option_ids)
            for option in group.options:
                considered += 1
                if state.conflict_keys.intersection(option.conflict_keys):
                    continue
                impossible_dependency = any(
                    option_group_index[dependency] < group_index
                    and dependency not in selected
                    for dependency in option.dependency_option_ids
                )
                if impossible_dependency:
                    continue
                new_resources = (
                    frozenset(option.shared_resource_ids) - state.shared_resource_ids
                )
                unique_parameters = (
                    state.unique_parameters
                    + option.local_unique_parameters
                    + sum(
                        resource_by_id[value].unique_parameters
                        for value in new_resources
                    )
                )
                if unique_parameters > problem.exact_target_unique_parameters:
                    continue
                serialized_bytes = (
                    state.serialized_bytes
                    + option.local_serialized_bytes
                    + sum(
                        resource_by_id[value].serialized_bytes
                        for value in new_resources
                    )
                )
                next_states.append(
                    _State(
                        state.selected_option_ids + (option.option_id,),
                        state.shared_resource_ids | new_resources,
                        state.conflict_keys | frozenset(option.conflict_keys),
                        unique_parameters,
                        serialized_bytes,
                        state.risk + option.predicted_risk,
                    )
                )
        if not next_states:
            raise DynamicFoldBudgetUnattainable(
                f"No legal topology remains after mandatory group {group.group_id!r}."
            )
        next_states.sort(
            key=lambda value: (
                value.unique_parameters,
                value.risk,
                value.serialized_bytes,
                value.selected_option_ids,
            )
        )
        if len(next_states) > problem.max_states:
            raise DynamicFoldBudgetUnattainable(
                "Exact topology search exceeded its immutable retained-state "
                "ceiling; refusing an approximate budget claim."
            )
        states = next_states
        peak = max(peak, len(states))

    viable = []
    for state in states:
        selected = frozenset(state.selected_option_ids)
        if state.unique_parameters != problem.exact_target_unique_parameters:
            continue
        if any(
            dependency not in selected
            for option_id in state.selected_option_ids
            for dependency in next(
                option
                for group in problem.groups
                for option in group.options
                if option.option_id == option_id
            ).dependency_option_ids
        ):
            continue
        viable.append(state)
    if not viable:
        nearest = min(
            states,
            key=lambda value: (
                abs(value.unique_parameters - problem.exact_target_unique_parameters),
                value.risk,
                value.serialized_bytes,
                value.selected_option_ids,
            ),
        )
        raise DynamicFoldBudgetUnattainable(
            "No legal topology exactly reaches "
            f"{problem.exact_target_unique_parameters} unique parameters; "
            f"nearest retained state has {nearest.unique_parameters}."
        )
    selected = min(
        viable,
        key=lambda value: (
            value.risk,
            value.serialized_bytes,
            value.selected_option_ids,
        ),
    )
    return ExactTopologyBudgetSelection(
        problem_fingerprint=problem.fingerprint,
        selected_option_ids=selected.selected_option_ids,
        shared_resource_ids=tuple(sorted(selected.shared_resource_ids)),
        exact_unique_parameters=selected.unique_parameters,
        serialized_bytes=selected.serialized_bytes,
        total_predicted_risk=selected.risk,
        states_considered=considered,
        peak_retained_states=peak,
    )


def _component_semantic(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if result not in _COMPONENT_SEMANTICS:
        raise DynamicFoldBudgetError(
            f"{name} must be one of {tuple(sorted(_COMPONENT_SEMANTICS))}."
        )
    return result


def _ppm(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if result > 1_000_000:
        raise DynamicFoldBudgetError(f"{name} cannot exceed 1,000,000.")
    return result


def _role_tokens(role: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.split(r"[^a-z0-9]+", role.casefold())
        if token
    )


def classify_component_parameter_semantic(
    circuit_kind: CircuitKind | str,
    parameter_role: str,
) -> str:
    """Classify an inspected IR role without using paths or layer numbers.

    Inspectors already reduce framework-specific modules to canonical circuit
    kinds and functional parameter roles.  The allocator deliberately works
    from those two facts only; a renamed module path or reordered layer cannot
    change the classification.
    """

    kind = circuit_kind.value if isinstance(circuit_kind, CircuitKind) else _text(
        circuit_kind,
        "circuit_kind",
    )
    role = _text(parameter_role, "parameter_role")
    tokens = _role_tokens(role)
    compact = "".join(sorted(tokens))
    if kind in {
        CircuitKind.SHORT_CONV.value,
        CircuitKind.RECURRENT_MIXER.value,
    }:
        return SEMANTIC_STATE_MIXER_INTERNAL
    if kind == CircuitKind.ATTENTION.value:
        query_key_tokens = {
            "q",
            "k",
            "query",
            "key",
            "qproj",
            "kproj",
            "wq",
            "wk",
        }
        if (
            tokens.intersection(query_key_tokens)
            or "q_proj" in role.casefold()
            or "k_proj" in role.casefold()
            or "queryproj" in compact
            or "keyproj" in compact
        ):
            return SEMANTIC_ATTENTION_QUERY_KEY
        return SEMANTIC_ATTENTION_WRITER
    if kind in {
        CircuitKind.GATED_MLP.value,
        CircuitKind.DENSE_MLP.value,
    }:
        writer_tokens = {
            "down",
            "downproj",
            "out",
            "output",
            "writer",
            "w2",
            "wo",
        }
        if (
            tokens.intersection(writer_tokens)
            or "down_proj" in role.casefold()
            or "outputproj" in compact
        ):
            return SEMANTIC_MLP_WRITER
        return SEMANTIC_MLP_PRODUCER
    if kind == CircuitKind.NORM.value:
        return SEMANTIC_NORMALIZATION
    if kind in {
        CircuitKind.TOKEN_EMBEDDING.value,
        CircuitKind.OUTPUT_HEAD.value,
        CircuitKind.MODALITY_INPUT.value,
        CircuitKind.MODALITY_OUTPUT.value,
        CircuitKind.INTERFACE_PROJECTION.value,
    }:
        return SEMANTIC_INTERFACE
    return SEMANTIC_OTHER


@dataclass(frozen=True, slots=True)
class ComponentProtectionRule:
    """Evidence-informed loss/floor policy over discovered IR semantics."""

    rule_id: str
    circuit_kinds: tuple[str, ...] = ()
    parameter_semantics: tuple[str, ...] = ()
    loss_weight: float = 1.0
    minimum_retention_ppm: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        kinds = tuple(
            sorted(
                {
                    value.value if isinstance(value, CircuitKind) else _text(
                        value,
                        "circuit_kind",
                    )
                    for value in self.circuit_kinds
                }
            )
        )
        semantics = tuple(
            sorted(
                {
                    _component_semantic(value, "parameter_semantic")
                    for value in self.parameter_semantics
                }
            )
        )
        if not kinds and not semantics:
            raise DynamicFoldBudgetError(
                "A component protection rule needs a circuit-kind or semantic "
                "selector."
            )
        object.__setattr__(self, "circuit_kinds", kinds)
        object.__setattr__(self, "parameter_semantics", semantics)
        object.__setattr__(
            self,
            "loss_weight",
            _finite_nonnegative(self.loss_weight, "loss_weight"),
        )
        object.__setattr__(
            self,
            "minimum_retention_ppm",
            _ppm(self.minimum_retention_ppm, "minimum_retention_ppm"),
        )

    def matches(self, circuit_kind: str, parameter_semantic: str) -> bool:
        return (
            (not self.circuit_kinds or circuit_kind in self.circuit_kinds)
            and (
                not self.parameter_semantics
                or parameter_semantic in self.parameter_semantics
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "circuit_kinds": list(self.circuit_kinds),
            "parameter_semantics": list(self.parameter_semantics),
            "loss_weight": self.loss_weight,
            "minimum_retention_ppm": self.minimum_retention_ppm,
        }


def residual_attention_component_protection_rules(
) -> tuple[ComponentProtectionRule, ...]:
    """Default ordering learned from the current fold-only evidence frontier.

    The numbers only prioritize topology allocation.  They are not behavioral
    scores and do not gate publication.  Short-conv/recurrent internals and
    attention Q/K coordinates are expensive to damage; MLP producers/writers
    and attention V/O writers remain available to absorb a tighter global
    budget.  No layer index or family-specific module path is named.
    """

    return (
        ComponentProtectionRule(
            "protect.state_mixer_internal",
            parameter_semantics=(SEMANTIC_STATE_MIXER_INTERNAL,),
            loss_weight=24.0,
        ),
        ComponentProtectionRule(
            "protect.attention_query_key",
            parameter_semantics=(SEMANTIC_ATTENTION_QUERY_KEY,),
            loss_weight=12.0,
        ),
        ComponentProtectionRule(
            "protect.normalization",
            parameter_semantics=(SEMANTIC_NORMALIZATION,),
            loss_weight=12.0,
        ),
        ComponentProtectionRule(
            "protect.model_interfaces",
            parameter_semantics=(SEMANTIC_INTERFACE,),
            loss_weight=6.0,
        ),
        ComponentProtectionRule(
            "compress.attention_writer",
            parameter_semantics=(SEMANTIC_ATTENTION_WRITER,),
            loss_weight=2.0,
        ),
        ComponentProtectionRule(
            "compress.mlp_producer",
            parameter_semantics=(SEMANTIC_MLP_PRODUCER,),
            loss_weight=1.0,
        ),
        ComponentProtectionRule(
            "compress.mlp_writer",
            parameter_semantics=(SEMANTIC_MLP_WRITER,),
            loss_weight=1.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class RetainedParameterAllocation:
    """Retained target capacity for one discovered source node/role."""

    node_id: str
    parameter_role: str
    retained_parameters: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "parameter_role",
            _text(self.parameter_role, "parameter_role"),
        )
        object.__setattr__(
            self,
            "retained_parameters",
            _nonnegative(self.retained_parameters, "retained_parameters"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parameter_role": self.parameter_role,
            "retained_parameters": self.retained_parameters,
        }


@dataclass(frozen=True, slots=True)
class DiscoveredComponentTopologyOption:
    """One adapter-emitted target shape for discovered source components."""

    option_id: str
    group_id: str
    source_node_ids: tuple[str, ...]
    local_unique_parameters: int
    local_serialized_bytes: int
    retained_parameters: tuple[RetainedParameterAllocation, ...] = ()
    shared_resource_ids: tuple[str, ...] = ()
    dependency_option_ids: tuple[str, ...] = ()
    conflict_keys: tuple[str, ...] = ()
    structural_risk: float = 0.0
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", _text(self.option_id, "option_id"))
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id"))
        nodes = _strings(self.source_node_ids, "source_node_id")
        if not nodes:
            raise DynamicFoldBudgetError(
                "A discovered component option needs source_node_ids."
            )
        object.__setattr__(self, "source_node_ids", nodes)
        for name in ("local_unique_parameters", "local_serialized_bytes"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        allocations = tuple(
            sorted(
                self.retained_parameters,
                key=lambda value: (value.node_id, value.parameter_role),
            )
        )
        if any(
            not isinstance(value, RetainedParameterAllocation)
            for value in allocations
        ):
            raise DynamicFoldBudgetError(
                "retained_parameters must contain RetainedParameterAllocation "
                "objects."
            )
        keys = tuple(
            (value.node_id, value.parameter_role) for value in allocations
        )
        if len(set(keys)) != len(keys):
            raise DynamicFoldBudgetError(
                "A component option repeats a retained node/role allocation."
            )
        if set(value.node_id for value in allocations) - set(nodes):
            raise DynamicFoldBudgetError(
                "A retained allocation references a node outside its option."
            )
        object.__setattr__(self, "retained_parameters", allocations)
        for name in (
            "shared_resource_ids",
            "dependency_option_ids",
            "conflict_keys",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if self.option_id in self.dependency_option_ids:
            raise DynamicFoldBudgetError("An option cannot depend on itself.")
        object.__setattr__(
            self,
            "structural_risk",
            _finite_nonnegative(self.structural_risk, "structural_risk"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "group_id": self.group_id,
            "source_node_ids": list(self.source_node_ids),
            "local_unique_parameters": self.local_unique_parameters,
            "local_serialized_bytes": self.local_serialized_bytes,
            "retained_parameters": [
                value.as_dict() for value in self.retained_parameters
            ],
            "shared_resource_ids": list(self.shared_resource_ids),
            "dependency_option_ids": list(self.dependency_option_ids),
            "conflict_keys": list(self.conflict_keys),
            "structural_risk": self.structural_risk,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DiscoveredComponentChoiceGroup:
    group_id: str
    options: tuple[DiscoveredComponentTopologyOption, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id"))
        options = tuple(sorted(self.options, key=lambda value: value.option_id))
        if not options or any(
            not isinstance(value, DiscoveredComponentTopologyOption)
            for value in options
        ):
            raise DynamicFoldBudgetError(
                "A discovered component group needs topology options."
            )
        if any(value.group_id != self.group_id for value in options):
            raise DynamicFoldBudgetError(
                "Every discovered option must name its containing group."
            )
        node_sets = {value.source_node_ids for value in options}
        if len(node_sets) != 1:
            raise DynamicFoldBudgetError(
                "Alternatives in a component group must cover identical source "
                "nodes."
            )
        if len({value.option_id for value in options}) != len(options):
            raise DynamicFoldBudgetError("Option IDs must be unique in a group.")
        object.__setattr__(self, "options", options)

    @property
    def source_node_ids(self) -> tuple[str, ...]:
        return self.options[0].source_node_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "source_node_ids": list(self.source_node_ids),
            "options": [value.as_dict() for value in self.options],
        }


@dataclass(frozen=True, slots=True)
class ArchitectureDiscoveredBudgetProblem:
    """Whole-model exact budget over groups built from an inspected graph."""

    source_graph: ModelCircuitGraph
    exact_target_unique_parameters: int
    groups: tuple[DiscoveredComponentChoiceGroup, ...]
    shared_resources: tuple[SharedTopologyResource, ...] = ()
    base_unique_parameters: int = 0
    base_serialized_bytes: int = 0
    protection_rules: tuple[ComponentProtectionRule, ...] = ()
    unclassified_loss_weight: float = 1.0
    bottleneck_risk_weight: float = 0.0
    max_states: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.source_graph, ModelCircuitGraph):
            raise DynamicFoldBudgetError(
                "source_graph must be an inspected ModelCircuitGraph."
            )
        target = _positive(
            self.exact_target_unique_parameters,
            "exact_target_unique_parameters",
        )
        if target >= self.source_graph.unique_parameter_count:
            raise DynamicFoldBudgetError(
                "A discovered component budget must compact the source graph."
            )
        object.__setattr__(self, "exact_target_unique_parameters", target)
        for name in ("base_unique_parameters", "base_serialized_bytes"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(self, "max_states", _positive(self.max_states, "max_states"))
        object.__setattr__(
            self,
            "unclassified_loss_weight",
            _finite_nonnegative(
                self.unclassified_loss_weight,
                "unclassified_loss_weight",
            ),
        )
        object.__setattr__(
            self,
            "bottleneck_risk_weight",
            _finite_nonnegative(
                self.bottleneck_risk_weight,
                "bottleneck_risk_weight",
            ),
        )
        groups = tuple(sorted(self.groups, key=lambda value: value.group_id))
        if not groups or any(
            not isinstance(value, DiscoveredComponentChoiceGroup)
            for value in groups
        ):
            raise DynamicFoldBudgetError(
                "An architecture-discovered budget needs component groups."
            )
        if len({value.group_id for value in groups}) != len(groups):
            raise DynamicFoldBudgetError("Component group IDs must be unique.")
        object.__setattr__(self, "groups", groups)
        option_ids = [
            option.option_id for group in groups for option in group.options
        ]
        if len(set(option_ids)) != len(option_ids):
            raise DynamicFoldBudgetError(
                "Discovered topology option IDs must be globally unique."
            )
        graph_nodes = {value.node_id: value for value in self.source_graph.nodes}
        parameterized_nodes = {
            value.node_id for value in self.source_graph.nodes if value.parameters
        }
        covered = [node_id for group in groups for node_id in group.source_node_ids]
        unknown = set(covered) - set(graph_nodes)
        if unknown:
            raise DynamicFoldBudgetError(
                f"Component groups reference unknown source nodes: {sorted(unknown)}."
            )
        repeated = sorted(
            node_id for node_id in set(covered) if covered.count(node_id) > 1
        )
        if repeated:
            raise DynamicFoldBudgetError(
                f"Source nodes appear in multiple component groups: {repeated}."
            )
        missing = parameterized_nodes - set(covered)
        if missing:
            raise DynamicFoldBudgetError(
                "Architecture-discovered allocation must cover every "
                f"parameterized source node; missing {sorted(missing)}."
            )
        role_counts = _source_role_counts(self.source_graph)
        for group in groups:
            group_keys = {
                key for key in role_counts if key[0] in group.source_node_ids
            }
            for option in group.options:
                for allocation in option.retained_parameters:
                    key = (allocation.node_id, allocation.parameter_role)
                    if key not in group_keys:
                        raise DynamicFoldBudgetError(
                            f"Option {option.option_id!r} retains unknown source "
                            f"role {key!r}."
                        )
                    if allocation.retained_parameters > role_counts[key]:
                        raise DynamicFoldBudgetError(
                            f"Option {option.option_id!r} retains more than source "
                            f"role {key!r}."
                        )
        resources = tuple(
            sorted(self.shared_resources, key=lambda value: value.resource_id)
        )
        if any(not isinstance(value, SharedTopologyResource) for value in resources):
            raise DynamicFoldBudgetError(
                "shared_resources must contain SharedTopologyResource objects."
            )
        if len({value.resource_id for value in resources}) != len(resources):
            raise DynamicFoldBudgetError("Shared resource IDs must be unique.")
        object.__setattr__(self, "shared_resources", resources)
        rules = tuple(self.protection_rules)
        if not rules:
            rules = residual_attention_component_protection_rules()
        if any(not isinstance(value, ComponentProtectionRule) for value in rules):
            raise DynamicFoldBudgetError(
                "protection_rules must contain ComponentProtectionRule objects."
            )
        if len({value.rule_id for value in rules}) != len(rules):
            raise DynamicFoldBudgetError("Protection rule IDs must be unique.")
        object.__setattr__(self, "protection_rules", rules)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DISCOVERED_COMPONENT_SCHEMA,
            "source_graph_fingerprint": self.source_graph.fingerprint,
            "component_retention_risk_policy": (
                COMPONENT_RETENTION_RISK_POLICY
            ),
            "source_unique_parameters": self.source_graph.unique_parameter_count,
            "source_logical_parameters": self.source_graph.logical_parameter_count,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "base_unique_parameters": self.base_unique_parameters,
            "base_serialized_bytes": self.base_serialized_bytes,
            "groups": [value.as_dict() for value in self.groups],
            "shared_resources": [value.as_dict() for value in self.shared_resources],
            "protection_rules": [value.as_dict() for value in self.protection_rules],
            "unclassified_loss_weight": self.unclassified_loss_weight,
            "bottleneck_risk_weight": self.bottleneck_risk_weight,
            "max_states": self.max_states,
            "complete_parameterized_node_coverage": True,
        }


@dataclass(frozen=True, slots=True)
class ComponentRetentionAudit:
    scope_id: str
    source_parameters: int
    retained_parameters: int
    retention_ppm: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "source_parameters": self.source_parameters,
            "retained_parameters": self.retained_parameters,
            "retention_ppm": self.retention_ppm,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureDiscoveredBudgetSelection:
    problem_fingerprint: str
    source_graph_fingerprint: str
    topology_selection: ExactTopologyBudgetSelection
    rule_retention: tuple[ComponentRetentionAudit, ...]
    semantic_retention: tuple[ComponentRetentionAudit, ...]
    discovered_group_kinds: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def selected_option_ids(self) -> tuple[str, ...]:
        return self.topology_selection.selected_option_ids

    @property
    def exact_unique_parameters(self) -> int:
        return self.topology_selection.exact_unique_parameters

    @property
    def serialized_bytes(self) -> int:
        return self.topology_selection.serialized_bytes

    @property
    def total_predicted_risk(self) -> float:
        return self.topology_selection.total_predicted_risk

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DISCOVERED_COMPONENT_SCHEMA,
            "kind": "architecture_discovered_budget_selection",
            "component_retention_risk_policy": (
                COMPONENT_RETENTION_RISK_POLICY
            ),
            "problem_fingerprint": self.problem_fingerprint,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "topology_selection": self.topology_selection.as_dict(),
            "rule_retention": [value.as_dict() for value in self.rule_retention],
            "semantic_retention": [
                value.as_dict() for value in self.semantic_retention
            ],
            "discovered_group_kinds": {
                group_id: list(kinds)
                for group_id, kinds in self.discovered_group_kinds
            },
            "complete_parameterized_node_coverage": True,
            "exact_global_parameter_budget_met": True,
            "architecture_discovery_uses_layer_ids": False,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
        }


def _source_role_counts(
    graph: ModelCircuitGraph,
) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for node in graph.nodes:
        for parameter in node.parameters:
            key = (node.node_id, parameter.role)
            result[key] = result.get(key, 0) + parameter.numel
    return result


def _source_role_descriptors(
    graph: ModelCircuitGraph,
) -> dict[tuple[str, str], tuple[str, str, int]]:
    nodes = {value.node_id: value for value in graph.nodes}
    counts = _source_role_counts(graph)
    return {
        key: (
            nodes[key[0]].kind,
            classify_component_parameter_semantic(nodes[key[0]].kind, key[1]),
            count,
        )
        for key, count in counts.items()
    }


def _option_retained_parameters(
    option: DiscoveredComponentTopologyOption,
    descriptors: Mapping[tuple[str, str], tuple[str, str, int]],
) -> dict[tuple[str, str], int]:
    group_keys = tuple(
        key for key in descriptors if key[0] in option.source_node_ids
    )
    source_total = sum(descriptors[key][2] for key in group_keys)
    if source_total <= 0:
        return {}
    explicit = {
        (value.node_id, value.parameter_role): value.retained_parameters
        for value in option.retained_parameters
    }
    numerator = min(option.local_unique_parameters, source_total)
    return {
        key: min(
            descriptors[key][2],
            explicit.get(
                key,
                descriptors[key][2] * numerator // source_total,
            ),
        )
        for key in group_keys
    }


def _retention_ppm(source: int, retained: int) -> int:
    return (
        1_000_000
        if source == 0
        else min(1_000_000, retained * 1_000_000 // source)
    )


def _rule_totals(
    rule: ComponentProtectionRule,
    descriptors: Mapping[tuple[str, str], tuple[str, str, int]],
    retained: Mapping[tuple[str, str], int],
) -> tuple[int, int]:
    keys = tuple(
        key
        for key, (kind, semantic, _count) in descriptors.items()
        if rule.matches(kind, semantic)
    )
    return (
        sum(descriptors[key][2] for key in keys),
        sum(retained.get(key, 0) for key in keys),
    )


def _effective_component_protection_rules(
    rules: Sequence[ComponentProtectionRule],
) -> tuple[ComponentProtectionRule, ...]:
    """Ignore exact policy duplicates while preserving intentional weights.

    A caller may express one multi-semantic rule as several single-semantic
    rules, and the normalized risk remains identical because every rule is
    split into semantic scopes below.  Repeating an otherwise identical rule
    under another ID is not additional evidence, so it must not silently
    amplify that scope's weight.
    """

    effective: list[ComponentProtectionRule] = []
    seen: set[tuple[Any, ...]] = set()
    for rule in rules:
        signature = (
            rule.circuit_kinds,
            rule.parameter_semantics,
            rule.loss_weight,
            rule.minimum_retention_ppm,
        )
        if signature in seen:
            continue
        seen.add(signature)
        effective.append(rule)
    return tuple(effective)


@dataclass(frozen=True, slots=True)
class _ComponentLossScope:
    semantic: str
    weight: float
    source_parameters: int
    rule: ComponentProtectionRule | None = None


@dataclass(frozen=True, slots=True)
class _PreparedComponentRiskPolicy:
    loss_scopes: tuple[_ComponentLossScope, ...]
    total_loss_scope_weight: float
    internal_source_parameters: tuple[tuple[str, int], ...]

    @property
    def internal_semantic_count(self) -> int:
        return len(self.internal_source_parameters)


def _prepare_component_risk_policy(
    problem: ArchitectureDiscoveredBudgetProblem,
    descriptors: Mapping[tuple[str, str], tuple[str, str, int]],
) -> _PreparedComponentRiskPolicy:
    semantics = tuple(sorted({value[1] for value in descriptors.values()}))
    scopes: list[_ComponentLossScope] = []
    if problem.unclassified_loss_weight:
        for semantic in semantics:
            source = sum(
                count
                for _kind, role_semantic, count in descriptors.values()
                if role_semantic == semantic
            )
            if source:
                scopes.append(
                    _ComponentLossScope(
                        semantic=semantic,
                        weight=problem.unclassified_loss_weight,
                        source_parameters=source,
                    )
                )
    for rule in _effective_component_protection_rules(
        problem.protection_rules
    ):
        if not rule.loss_weight:
            continue
        for semantic in semantics:
            source = sum(
                count
                for kind, role_semantic, count in descriptors.values()
                if role_semantic == semantic
                and rule.matches(kind, role_semantic)
            )
            if source:
                scopes.append(
                    _ComponentLossScope(
                        semantic=semantic,
                        weight=rule.loss_weight,
                        source_parameters=source,
                        rule=rule,
                    )
                )
    internal_sources = tuple(
        (
            semantic,
            sum(
                count
                for _kind, role_semantic, count in descriptors.values()
                if role_semantic == semantic
            ),
        )
        for semantic in _INTERNAL_COMPONENT_SEMANTICS
        if any(
            role_semantic == semantic
            for _kind, role_semantic, _count in descriptors.values()
        )
    )
    return _PreparedComponentRiskPolicy(
        loss_scopes=tuple(scopes),
        total_loss_scope_weight=math.fsum(
            scope.weight for scope in scopes
        ),
        internal_source_parameters=internal_sources,
    )


def _normalized_component_retention_risk(
    *,
    structural_risk: float,
    weighted_loss_scopes: Sequence[tuple[float, int, int, int]],
    total_loss_scope_weight: float,
    internal_retention_scopes: Sequence[tuple[int, int, int]],
    internal_semantic_count: int,
    bottleneck_risk_weight: float,
) -> float | None:
    """Score fractional semantic loss and within-role geometric balance.

    Every weighted-loss tuple is ``(weight, global source, scoped source,
    scoped retained)``.  Dividing by the global source makes the selected
    option risks additive across disjoint component groups while remaining
    invariant when a role's source and retained counts are uniformly scaled.

    The bottleneck term is the source-share-weighted mean negative log
    retention over internal semantic roles.  It is invariant to uniform
    scaling within each role, but deliberately not invariant when only one
    subcomponent is enlarged: that changes its share of the semantic role.
    A zero-retained scope uses a one-parameter/global-role-source floor.  It
    therefore remains legal for layer deletion, fusion, or cross-architecture
    transfer, but carries a large finite and explicitly auditable penalty.
    Only configured minimum-retention rules reject an option.
    """

    weighted_loss_numerator = math.fsum(
        weight * (scoped_source - scoped_retained) / global_source
        for weight, global_source, scoped_source, scoped_retained in (
            weighted_loss_scopes
        )
        if weight and global_source and scoped_source
    )
    weighted_loss = (
        0.0
        if total_loss_scope_weight == 0.0
        else weighted_loss_numerator / total_loss_scope_weight
    )
    bottleneck_risk = 0.0
    if bottleneck_risk_weight and internal_semantic_count:
        barrier_terms: list[float] = []
        for global_source, scoped_source, scoped_retained in (
            internal_retention_scopes
        ):
            if not scoped_source:
                continue
            retention = max(
                scoped_retained / scoped_source,
                1.0 / global_source,
            )
            barrier_terms.append(
                (scoped_source / global_source) * -math.log(retention)
            )
        bottleneck_risk = (
            bottleneck_risk_weight
            * math.fsum(barrier_terms)
            / internal_semantic_count
        )
    return structural_risk + weighted_loss + bottleneck_risk


def _component_option_risk(
    problem: ArchitectureDiscoveredBudgetProblem,
    option: DiscoveredComponentTopologyOption,
    descriptors: Mapping[tuple[str, str], tuple[str, str, int]],
    prepared_policy: _PreparedComponentRiskPolicy,
) -> tuple[float, dict[tuple[str, str], int]] | None:
    retained = _option_retained_parameters(option, descriptors)
    scoped_descriptors = {
        key: value for key, value in descriptors.items() if key in retained
    }
    for rule in problem.protection_rules:
        source, kept = _rule_totals(rule, scoped_descriptors, retained)
        if (
            source
            and _retention_ppm(source, kept) < rule.minimum_retention_ppm
        ):
            return None
    weighted_loss_scopes = []
    for scope in prepared_policy.loss_scopes:
        keys = tuple(
            key
            for key, (kind, semantic, _count) in scoped_descriptors.items()
            if semantic == scope.semantic
            and (scope.rule is None or scope.rule.matches(kind, semantic))
        )
        scoped_source = sum(scoped_descriptors[key][2] for key in keys)
        scoped_retained = sum(retained.get(key, 0) for key in keys)
        weighted_loss_scopes.append(
            (
                scope.weight,
                scope.source_parameters,
                scoped_source,
                scoped_retained,
            )
        )
    internal_retention_scopes = []
    for semantic, global_source in (
        prepared_policy.internal_source_parameters
    ):
        keys = tuple(
            key
            for key, (_kind, role_semantic, _count) in (
                scoped_descriptors.items()
            )
            if role_semantic == semantic
        )
        internal_retention_scopes.append(
            (
                global_source,
                sum(scoped_descriptors[key][2] for key in keys),
                sum(retained.get(key, 0) for key in keys),
            )
        )
    risk = _normalized_component_retention_risk(
        structural_risk=option.structural_risk,
        weighted_loss_scopes=weighted_loss_scopes,
        total_loss_scope_weight=(
            prepared_policy.total_loss_scope_weight
        ),
        internal_retention_scopes=internal_retention_scopes,
        internal_semantic_count=prepared_policy.internal_semantic_count,
        bottleneck_risk_weight=problem.bottleneck_risk_weight,
    )
    if risk is None:
        return None
    return risk, retained


def solve_architecture_discovered_budget(
    problem: ArchitectureDiscoveredBudgetProblem,
) -> ArchitectureDiscoveredBudgetSelection:
    """Allocate one exact whole-model budget using discovered component risk."""

    if not isinstance(problem, ArchitectureDiscoveredBudgetProblem):
        raise DynamicFoldBudgetError(
            "problem must be ArchitectureDiscoveredBudgetProblem."
        )
    descriptors = _source_role_descriptors(problem.source_graph)
    prepared_risk_policy = _prepare_component_risk_policy(
        problem,
        descriptors,
    )
    converted_groups: list[TopologyChoiceGroup] = []
    retained_by_option: dict[str, dict[tuple[str, str], int]] = {}
    source_nodes = {value.node_id: value for value in problem.source_graph.nodes}
    discovered_kinds: list[tuple[str, tuple[str, ...]]] = []
    for group in problem.groups:
        kinds = tuple(
            sorted({source_nodes[node_id].kind for node_id in group.source_node_ids})
        )
        discovered_kinds.append((group.group_id, kinds))
        converted_options: list[TopologyOption] = []
        for option in group.options:
            scored = _component_option_risk(
                problem,
                option,
                descriptors,
                prepared_risk_policy,
            )
            if scored is None:
                continue
            risk, retained = scored
            retained_by_option[option.option_id] = retained
            metadata = dict(option.metadata)
            generated = {
                "component_budget.discovered_kinds": ",".join(kinds),
                "component_budget.source_nodes_sha256": canonical_sha256(
                    list(group.source_node_ids)
                ),
                "component_budget.policy_derived_risk": risk,
            }
            overlap = set(metadata).intersection(generated)
            if overlap:
                raise DynamicFoldBudgetError(
                    f"Option {option.option_id!r} metadata overrides reserved "
                    f"component-budget keys: {sorted(overlap)}."
                )
            converted_options.append(
                TopologyOption(
                    option_id=option.option_id,
                    group_id=option.group_id,
                    local_unique_parameters=option.local_unique_parameters,
                    local_serialized_bytes=option.local_serialized_bytes,
                    shared_resource_ids=option.shared_resource_ids,
                    dependency_option_ids=option.dependency_option_ids,
                    conflict_keys=option.conflict_keys,
                    predicted_risk=risk,
                    metadata={**metadata, **generated},
                )
            )
        if not converted_options:
            raise DynamicFoldBudgetUnattainable(
                f"Every option for component group {group.group_id!r} violates "
                "its discovered retention floors."
            )
        converted_groups.append(
            TopologyChoiceGroup(group.group_id, tuple(converted_options))
        )
    exact_problem = ExactTopologyBudgetProblem(
        base_unique_parameters=problem.base_unique_parameters,
        base_serialized_bytes=problem.base_serialized_bytes,
        exact_target_unique_parameters=problem.exact_target_unique_parameters,
        groups=tuple(converted_groups),
        shared_resources=problem.shared_resources,
        max_states=problem.max_states,
    )
    topology_selection = solve_exact_topology_budget(exact_problem)
    selected_retained: dict[tuple[str, str], int] = {}
    for option_id in topology_selection.selected_option_ids:
        overlap = set(selected_retained).intersection(retained_by_option[option_id])
        if overlap:
            raise DynamicFoldBudgetError(
                "Selected component groups overlap source node/role allocations."
            )
        selected_retained.update(retained_by_option[option_id])
    rule_audits = []
    for rule in problem.protection_rules:
        source, kept = _rule_totals(rule, descriptors, selected_retained)
        rule_audits.append(
            ComponentRetentionAudit(
                scope_id=rule.rule_id,
                source_parameters=source,
                retained_parameters=kept,
                retention_ppm=_retention_ppm(source, kept),
            )
        )
    semantic_audits = []
    for semantic in sorted(_COMPONENT_SEMANTICS):
        keys = tuple(
            key for key, value in descriptors.items() if value[1] == semantic
        )
        if not keys:
            continue
        source = sum(descriptors[key][2] for key in keys)
        kept = sum(selected_retained.get(key, 0) for key in keys)
        semantic_audits.append(
            ComponentRetentionAudit(
                scope_id=semantic,
                source_parameters=source,
                retained_parameters=kept,
                retention_ppm=_retention_ppm(source, kept),
            )
        )
    return ArchitectureDiscoveredBudgetSelection(
        problem_fingerprint=problem.fingerprint,
        source_graph_fingerprint=problem.source_graph.fingerprint,
        topology_selection=topology_selection,
        rule_retention=tuple(rule_audits),
        semantic_retention=tuple(semantic_audits),
        discovered_group_kinds=tuple(discovered_kinds),
    )


__all__ = [
    "ArchitectureDiscoveredBudgetProblem",
    "ArchitectureDiscoveredBudgetSelection",
    "COMPONENT_RETENTION_RISK_POLICY",
    "ComponentProtectionRule",
    "ComponentRetentionAudit",
    "DISCOVERED_COMPONENT_SCHEMA",
    "DiscoveredComponentChoiceGroup",
    "DiscoveredComponentTopologyOption",
    "DynamicFoldBudgetError",
    "DynamicFoldBudgetUnattainable",
    "ExactTopologyBudgetProblem",
    "ExactTopologyBudgetSelection",
    "RetainedParameterAllocation",
    "SCHEMA",
    "SEMANTIC_ATTENTION_QUERY_KEY",
    "SEMANTIC_ATTENTION_WRITER",
    "SEMANTIC_INTERFACE",
    "SEMANTIC_MLP_PRODUCER",
    "SEMANTIC_MLP_WRITER",
    "SEMANTIC_NORMALIZATION",
    "SEMANTIC_OTHER",
    "SEMANTIC_STATE_MIXER_INTERNAL",
    "SharedTopologyResource",
    "TopologyChoiceGroup",
    "TopologyOption",
    "classify_component_parameter_semantic",
    "residual_attention_component_protection_rules",
    "solve_architecture_discovered_budget",
    "solve_exact_topology_budget",
]
