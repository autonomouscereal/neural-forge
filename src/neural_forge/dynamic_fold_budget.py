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
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dynamic_circuit_ir import JsonScalar, canonical_sha256


SCHEMA = "neural-forge-dynamic-topology-budget-v1"


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


__all__ = [
    "DynamicFoldBudgetError",
    "DynamicFoldBudgetUnattainable",
    "ExactTopologyBudgetProblem",
    "ExactTopologyBudgetSelection",
    "SCHEMA",
    "SharedTopologyResource",
    "TopologyChoiceGroup",
    "TopologyOption",
    "solve_exact_topology_budget",
]
