"""Family-neutral, transactional child-closed-loop fold orchestration.

The module defines orchestration rather than architecture-specific folding
mathematics.  An adapter supplies sealed parent trajectories, endpoint
bridging, child replay/capture, analytical compilers, in-place installers,
validation, state fingerprints, and exact transaction handling.

For every selected layer, the compiler:

1. captures the current child input;
2. compiles and installs the producer path;
3. recaptures the *installed child* before fitting its consumer;
4. installs the consumer and recaptures again;
5. compiles and installs the downstream path; and
6. validates on whole trajectories never exposed to fitting.

Layers always run in ascending order.  An optional model-global body-reclosure
stage runs after every selected layer and before final normalization so that
residual-coordinate repairs are visible to every terminal fit.  Final
normalization runs only after body reclosure validates.  A separately compiled
output-interface fold then runs from a pre-install fit replay, records a
distinct held-out pre-install baseline, and must survive a fresh held-out
post-install replay through the complete child.  Any error or rejected
validation rolls the entire child back to its exact pre-run state.  Parent
execution, trace regeneration, training, backward, and optimizer activity are
forbidden.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .dynamic_circuit_ir import JsonScalar, canonical_sha256
from .dynamic_fold_executor import run_fold_only_callback


RUN_SCHEMA = "neural-forge-dynamic-child-closed-loop-fold-v3"
SPLIT_SCHEMA = "neural-forge-whole-trajectory-fit-validation-split-v1"
_FORBIDDEN_COUNTERS = (
    "parent_forward_calls",
    "parent_trace_regeneration_calls",
    "training_steps",
    "backward_calls",
    "optimizer_steps",
)


class ClosedLoopFoldError(RuntimeError):
    """The adapter contract, closed-loop compile, validation, or rollback failed."""


class ClosedLoopAuditTripwire(ClosedLoopFoldError):
    """A forbidden parent-execution or training counter became nonzero."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClosedLoopFoldError(f"{name} must be a nonempty string.")
    return value.strip()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ClosedLoopFoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClosedLoopFoldError(f"{name} must be a non-negative integer.")
    return value


def _layer_index(value: Any, name: str = "layer_index") -> int:
    return _nonnegative_int(value, name)


def _metadata(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalized: list[tuple[str, JsonScalar]] = []
    for key, item in items:
        key = _text(str(key), "metadata key")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise ClosedLoopFoldError(
                f"Metadata {key!r} must be a JSON scalar."
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise ClosedLoopFoldError(f"Metadata {key!r} must be finite.")
        normalized.append((key, item))
    normalized.sort()
    if len({key for key, _item in normalized}) != len(normalized):
        raise ClosedLoopFoldError("Metadata keys must be unique.")
    return tuple(normalized)


RecordKey = tuple[str, str]


def _record_key(value: Any, name: str = "record key") -> RecordKey:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
    ):
        raise ClosedLoopFoldError(
            f"{name} must be a (trajectory_id, path_id) tuple."
        )
    return (
        _text(value[0], f"{name} trajectory_id"),
        _text(value[1], f"{name} path_id"),
    )


def _record_keys(
    records: Sequence["TrajectoryRecord"],
) -> tuple[RecordKey, ...]:
    keys = tuple(sorted(value.key for value in records))
    if len(keys) != len(set(keys)):
        raise ClosedLoopFoldError("Trajectory record keys must be unique.")
    return keys


def _checked_record_keys(
    values: Sequence[RecordKey],
    name: str,
) -> tuple[RecordKey, ...]:
    result = tuple(_record_key(value, name) for value in values)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ClosedLoopFoldError(
            f"{name} values must be sorted and unique."
        )
    if not result:
        raise ClosedLoopFoldError(f"{name} cannot be empty.")
    return result


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """One sealed path payload belonging to one indivisible trajectory."""

    trajectory_id: str
    path_id: str
    record_sha256: str
    payload: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_id", _text(self.trajectory_id, "trajectory_id")
        )
        object.__setattr__(self, "path_id", _text(self.path_id, "path_id"))
        object.__setattr__(
            self,
            "record_sha256",
            _digest(self.record_sha256, "record_sha256"),
        )

    @property
    def key(self) -> RecordKey:
        return (self.trajectory_id, self.path_id)

    def descriptor(self) -> dict[str, str]:
        return {
            "trajectory_id": self.trajectory_id,
            "path_id": self.path_id,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True, slots=True)
class TrajectorySplit:
    """Deterministic fit/validation split with trajectory-level isolation."""

    fit: tuple[TrajectoryRecord, ...]
    validation: tuple[TrajectoryRecord, ...]
    available_paths: tuple[str, ...]
    selected_paths: tuple[str, ...]
    split_seed: str
    validation_fraction: float
    split_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        fit = tuple(self.fit)
        validation = tuple(self.validation)
        available = tuple(_text(value, "available path") for value in self.available_paths)
        selected = tuple(_text(value, "selected path") for value in self.selected_paths)
        seed = _text(self.split_seed, "split_seed")
        fraction = self.validation_fraction
        if (
            not fit
            or not validation
            or any(not isinstance(value, TrajectoryRecord) for value in fit)
            or any(
                not isinstance(value, TrajectoryRecord)
                for value in validation
            )
            or available != tuple(sorted(set(available)))
            or selected != tuple(sorted(set(selected)))
            or not set(selected).issubset(available)
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0
        ):
            raise ClosedLoopFoldError(
                "Trajectory split structure or validation fraction is invalid."
            )
        fit_keys = _record_keys(fit)
        validation_keys = _record_keys(validation)
        if set(fit_keys) & set(validation_keys):
            raise ClosedLoopFoldError(
                "A trajectory record appears in both fit and validation."
            )
        fit_trajectories = {value.trajectory_id for value in fit}
        validation_trajectories = {
            value.trajectory_id for value in validation
        }
        if fit_trajectories & validation_trajectories:
            raise ClosedLoopFoldError(
                "Whole trajectories cannot leak between fit and validation."
            )
        object.__setattr__(self, "fit", tuple(sorted(fit, key=lambda value: value.key)))
        object.__setattr__(
            self,
            "validation",
            tuple(sorted(validation, key=lambda value: value.key)),
        )
        object.__setattr__(self, "available_paths", available)
        object.__setattr__(self, "selected_paths", selected)
        object.__setattr__(self, "split_seed", seed)
        object.__setattr__(self, "validation_fraction", float(fraction))
        object.__setattr__(
            self,
            "split_sha256",
            canonical_sha256(
                {
                    "schema": SPLIT_SCHEMA,
                    "fit": [
                        value.descriptor()
                        for value in sorted(fit, key=lambda item: item.key)
                    ],
                    "validation": [
                        value.descriptor()
                        for value in sorted(
                            validation, key=lambda item: item.key
                        )
                    ],
                    "available_paths": list(available),
                    "selected_paths": list(selected),
                    "split_seed": seed,
                    "validation_fraction": float(fraction),
                }
            ),
        )

    @property
    def fit_record_keys(self) -> tuple[RecordKey, ...]:
        return _record_keys(self.fit)

    @property
    def validation_record_keys(self) -> tuple[RecordKey, ...]:
        return _record_keys(self.validation)

    @property
    def fit_trajectory_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value.trajectory_id for value in self.fit}))

    @property
    def validation_trajectory_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({value.trajectory_id for value in self.validation})
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SPLIT_SCHEMA,
            "split_sha256": self.split_sha256,
            "split_seed": self.split_seed,
            "validation_fraction": self.validation_fraction,
            "available_paths": list(self.available_paths),
            "selected_paths": list(self.selected_paths),
            "default_all_paths": self.selected_paths == self.available_paths,
            "fit_trajectory_ids": list(self.fit_trajectory_ids),
            "validation_trajectory_ids": list(
                self.validation_trajectory_ids
            ),
            "fit_record_keys": [list(value) for value in self.fit_record_keys],
            "validation_record_keys": [
                list(value) for value in self.validation_record_keys
            ],
            "whole_trajectory_isolation": True,
        }


def deterministic_whole_trajectory_split(
    records: Sequence[TrajectoryRecord],
    *,
    validation_fraction: float = 0.2,
    split_seed: str = "neural-forge-closed-loop-v1",
    selected_paths: Sequence[str] | None = None,
) -> TrajectorySplit:
    """Split by trajectory identity, never by token, frame, row, or path."""

    records = tuple(records)
    if not records or any(
        not isinstance(value, TrajectoryRecord) for value in records
    ):
        raise ClosedLoopFoldError(
            "records must contain sealed TrajectoryRecord values."
        )
    if len(_record_keys(records)) != len(records):
        raise ClosedLoopFoldError("Trajectory records contain duplicate keys.")
    available_paths = tuple(sorted({value.path_id for value in records}))
    if selected_paths is None:
        selected = available_paths
    else:
        selected = tuple(
            sorted({_text(value, "selected path") for value in selected_paths})
        )
        if (
            not selected
            or len(selected) != len(tuple(selected_paths))
            or not set(selected).issubset(available_paths)
        ):
            raise ClosedLoopFoldError(
                "selected_paths must be unique available path identifiers."
            )
    filtered = tuple(
        sorted(
            (value for value in records if value.path_id in selected),
            key=lambda value: value.key,
        )
    )
    by_trajectory: dict[str, list[TrajectoryRecord]] = {}
    for record in filtered:
        by_trajectory.setdefault(record.trajectory_id, []).append(record)
    if len(by_trajectory) < 2:
        raise ClosedLoopFoldError(
            "At least two complete trajectories are required for a fit/"
            "validation split."
        )
    expected_paths = set(selected)
    for trajectory_id, values in by_trajectory.items():
        observed = {value.path_id for value in values}
        if observed != expected_paths:
            raise ClosedLoopFoldError(
                f"Trajectory {trajectory_id!r} does not cover every selected "
                "path."
            )
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ClosedLoopFoldError(
            "validation_fraction must be finite and strictly within (0, 1)."
        )
    seed = _text(split_seed, "split_seed")
    ranked = sorted(
        (
            canonical_sha256(
                {
                    "schema": SPLIT_SCHEMA,
                    "split_seed": seed,
                    "trajectory_id": trajectory_id,
                    "records": [
                        value.descriptor()
                        for value in sorted(
                            values, key=lambda item: item.key
                        )
                    ],
                }
            ),
            trajectory_id,
        )
        for trajectory_id, values in by_trajectory.items()
    )
    validation_count = int(
        math.floor(len(ranked) * float(validation_fraction) + 0.5)
    )
    validation_count = max(1, min(len(ranked) - 1, validation_count))
    validation_ids = {
        trajectory_id
        for _rank, trajectory_id in ranked[:validation_count]
    }
    fit = tuple(
        value
        for value in filtered
        if value.trajectory_id not in validation_ids
    )
    validation = tuple(
        value
        for value in filtered
        if value.trajectory_id in validation_ids
    )
    return TrajectorySplit(
        fit=fit,
        validation=validation,
        available_paths=available_paths,
        selected_paths=selected,
        split_seed=seed,
        validation_fraction=float(validation_fraction),
    )


@dataclass(frozen=True, slots=True)
class FoldScope:
    """One layer or one authenticated model-global fold scope."""

    kind: str
    layer_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "layer",
            "body_reclosure",
            "final_normalization",
            "output_interface",
        }:
            raise ClosedLoopFoldError(
                "Fold scope kind must be layer, body_reclosure, "
                "final_normalization, or output_interface."
            )
        if self.kind == "layer":
            object.__setattr__(
                self,
                "layer_index",
                _layer_index(self.layer_index),
            )
        elif self.layer_index is not None:
            raise ClosedLoopFoldError(
                "Model-global scope cannot carry a layer index."
            )

    @property
    def scope_id(self) -> str:
        if self.kind == "layer":
            return f"layer:{self.layer_index:06d}"
        return self.kind

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "layer_index": self.layer_index,
            "scope_id": self.scope_id,
        }


class CapturePoint(str, Enum):
    PRE_PRODUCER = "pre_producer"
    POST_PRODUCER = "post_producer"
    POST_CONSUMER = "post_consumer"
    POST_DOWNSTREAM = "post_downstream"
    PRE_BODY_RECLOSURE = "pre_body_reclosure"
    POST_BODY_RECLOSURE = "post_body_reclosure"
    PRE_FINAL_NORMALIZATION = "pre_final_normalization"
    POST_FINAL_NORMALIZATION = "post_final_normalization"
    PRE_OUTPUT_INTERFACE = "pre_output_interface"
    POST_OUTPUT_INTERFACE = "post_output_interface"


class CompileStage(str, Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    DOWNSTREAM = "downstream"
    BODY_RECLOSURE = "body_reclosure"
    FINAL_NORMALIZATION = "final_normalization"
    OUTPUT_INTERFACE = "output_interface"


@dataclass(frozen=True, slots=True)
class ParentEndpointBatch:
    scope: FoldScope
    record_keys: tuple[RecordKey, ...]
    batch_sha256: str
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FoldScope):
            raise ClosedLoopFoldError("Endpoint batch scope is invalid.")
        object.__setattr__(
            self,
            "record_keys",
            _checked_record_keys(self.record_keys, "endpoint record key"),
        )
        object.__setattr__(
            self,
            "batch_sha256",
            _digest(self.batch_sha256, "endpoint batch_sha256"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.as_dict(),
            "record_keys": [list(value) for value in self.record_keys],
            "batch_sha256": self.batch_sha256,
        }


@dataclass(frozen=True, slots=True)
class ChildTrajectoryCapture:
    scope: FoldScope
    point: CapturePoint
    record_keys: tuple[RecordKey, ...]
    capture_sha256: str
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FoldScope):
            raise ClosedLoopFoldError("Child capture scope is invalid.")
        if not isinstance(self.point, CapturePoint):
            raise ClosedLoopFoldError("Child capture point is invalid.")
        object.__setattr__(
            self,
            "record_keys",
            _checked_record_keys(self.record_keys, "capture record key"),
        )
        object.__setattr__(
            self,
            "capture_sha256",
            _digest(self.capture_sha256, "capture_sha256"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.as_dict(),
            "point": self.point.value,
            "record_keys": [list(value) for value in self.record_keys],
            "capture_sha256": self.capture_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompiledFoldStage:
    scope: FoldScope
    stage: CompileStage
    fit_record_keys: tuple[RecordKey, ...]
    artifact_sha256: str
    payload: Any = None
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FoldScope):
            raise ClosedLoopFoldError("Compiled stage scope is invalid.")
        if not isinstance(self.stage, CompileStage):
            raise ClosedLoopFoldError("Compiled stage kind is invalid.")
        object.__setattr__(
            self,
            "fit_record_keys",
            _checked_record_keys(
                self.fit_record_keys, "compiled fit record key"
            ),
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest(self.artifact_sha256, "artifact_sha256"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.as_dict(),
            "stage": self.stage.value,
            "fit_record_keys": [
                list(value) for value in self.fit_record_keys
            ],
            "artifact_sha256": self.artifact_sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class InstallReceipt:
    scope: FoldScope
    stage: CompileStage
    artifact_sha256: str
    accepted: bool
    state_sha256_before: str
    state_sha256_after: str
    topology_preserved: bool
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scope, FoldScope)
            or not isinstance(self.stage, CompileStage)
            or not isinstance(self.accepted, bool)
            or not isinstance(self.topology_preserved, bool)
        ):
            raise ClosedLoopFoldError("Install receipt structure is invalid.")
        for name in (
            "artifact_sha256",
            "state_sha256_before",
            "state_sha256_after",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.as_dict(),
            "stage": self.stage.value,
            "artifact_sha256": self.artifact_sha256,
            "accepted": self.accepted,
            "state_sha256_before": self.state_sha256_before,
            "state_sha256_after": self.state_sha256_after,
            "topology_preserved": self.topology_preserved,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ValidationDecision:
    scope: FoldScope
    accepted: bool
    validation_record_keys: tuple[RecordKey, ...]
    reason: str
    metrics: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FoldScope) or not isinstance(
            self.accepted, bool
        ):
            raise ClosedLoopFoldError("Validation decision structure is invalid.")
        object.__setattr__(
            self,
            "validation_record_keys",
            _checked_record_keys(
                self.validation_record_keys,
                "validation record key",
            ),
        )
        object.__setattr__(self, "reason", _text(self.reason, "validation reason"))
        object.__setattr__(self, "metrics", _metadata(self.metrics))

    @property
    def decision_sha256(self) -> str:
        return canonical_sha256(
            {
                "scope": self.scope.as_dict(),
                "accepted": self.accepted,
                "validation_record_keys": [
                    list(value) for value in self.validation_record_keys
                ],
                "reason": self.reason,
                "metrics": dict(self.metrics),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.as_dict(),
            "accepted": self.accepted,
            "validation_record_keys": [
                list(value) for value in self.validation_record_keys
            ],
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "decision_sha256": self.decision_sha256,
        }


@dataclass(frozen=True, slots=True)
class TransactionReceipt:
    action: str
    accepted: bool
    exact: bool
    state_sha256_before: str
    state_sha256_after: str
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.action not in {"commit", "rollback"}
            or not isinstance(self.accepted, bool)
            or not isinstance(self.exact, bool)
        ):
            raise ClosedLoopFoldError("Transaction receipt structure is invalid.")
        object.__setattr__(
            self,
            "state_sha256_before",
            _digest(self.state_sha256_before, "transaction state before"),
        )
        object.__setattr__(
            self,
            "state_sha256_after",
            _digest(self.state_sha256_after, "transaction state after"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "accepted": self.accepted,
            "exact": self.exact,
            "state_sha256_before": self.state_sha256_before,
            "state_sha256_after": self.state_sha256_after,
            "metadata": dict(self.metadata),
        }


class ChildStateTransaction(Protocol):
    active: bool
    state_sha256_before: str
    topology_sha256_before: str

    def commit(self) -> TransactionReceipt: ...

    def rollback(self) -> TransactionReceipt: ...


@dataclass(slots=True)
class ClosedLoopAudit:
    """Run-local allowed-operation counts and zero-tolerance tripwires."""

    parent_forward_calls: int = 0
    parent_trace_regeneration_calls: int = 0
    training_steps: int = 0
    backward_calls: int = 0
    optimizer_steps: int = 0
    optimizer_constructions: int = 0
    autograd_grad_calls: int = 0
    child_replay_capture_calls: int = 0
    parent_endpoint_bridge_calls: int = 0
    producer_compile_calls: int = 0
    consumer_fit_calls: int = 0
    downstream_compile_calls: int = 0
    body_reclosure_compile_calls: int = 0
    final_normalization_compile_calls: int = 0
    output_interface_compile_calls: int = 0
    install_calls: int = 0
    validation_calls: int = 0
    adapter_counter_checks: int = 0
    torch_fold_only_guard_passed: bool = False

    def _forbid(self, name: str, count: int = 1) -> None:
        count = _nonnegative_int(count, name)
        setattr(self, name, getattr(self, name) + count)
        if count:
            raise ClosedLoopAuditTripwire(
                f"{name} is forbidden during child-closed-loop folding."
            )

    def record_parent_forward(self, count: int = 1) -> None:
        self._forbid("parent_forward_calls", count)

    def record_parent_trace_regeneration(self, count: int = 1) -> None:
        self._forbid("parent_trace_regeneration_calls", count)

    def record_training_step(self, count: int = 1) -> None:
        self._forbid("training_steps", count)

    def record_backward(self, count: int = 1) -> None:
        self._forbid("backward_calls", count)

    def record_optimizer_step(self, count: int = 1) -> None:
        self._forbid("optimizer_steps", count)

    def verify_adapter_counters(
        self,
        counters: Mapping[str, int],
        *,
        phase: str,
    ) -> None:
        if not isinstance(counters, Mapping):
            raise ClosedLoopAuditTripwire(
                f"{phase} adapter audit counters are unavailable."
            )
        for name in _FORBIDDEN_COUNTERS:
            if name not in counters:
                raise ClosedLoopAuditTripwire(
                    f"{phase} adapter audit omits {name}."
                )
            value = _nonnegative_int(counters[name], f"{phase} {name}")
            setattr(self, name, value)
            if value:
                raise ClosedLoopAuditTripwire(
                    f"{phase} observed forbidden {name}={value}."
                )
        self.adapter_counter_checks += 1

    def absorb_torch_guard(self, report: Any) -> None:
        self.optimizer_constructions = _nonnegative_int(
            report.optimizer_constructions,
            "optimizer_constructions",
        )
        self.optimizer_steps = _nonnegative_int(
            report.optimizer_steps,
            "optimizer_steps",
        )
        self.backward_calls = _nonnegative_int(
            report.backward_calls,
            "backward_calls",
        )
        self.autograd_grad_calls = _nonnegative_int(
            report.autograd_grad_calls,
            "autograd_grad_calls",
        )
        self.torch_fold_only_guard_passed = bool(report.clean)
        self.assert_clean()

    def assert_clean(self) -> None:
        dirty = {
            name: getattr(self, name)
            for name in (
                *_FORBIDDEN_COUNTERS,
                "optimizer_constructions",
                "autograd_grad_calls",
            )
            if getattr(self, name) != 0
        }
        if dirty:
            raise ClosedLoopAuditTripwire(
                f"Forbidden fold activity was observed: {dirty}."
            )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "parent_forward_calls": self.parent_forward_calls,
            "parent_trace_regeneration_calls": (
                self.parent_trace_regeneration_calls
            ),
            "training_steps": self.training_steps,
            "backward_calls": self.backward_calls,
            "optimizer_steps": self.optimizer_steps,
            "optimizer_constructions": self.optimizer_constructions,
            "autograd_grad_calls": self.autograd_grad_calls,
            "child_replay_capture_calls": self.child_replay_capture_calls,
            "parent_endpoint_bridge_calls": (
                self.parent_endpoint_bridge_calls
            ),
            "producer_compile_calls": self.producer_compile_calls,
            "consumer_fit_calls": self.consumer_fit_calls,
            "downstream_compile_calls": self.downstream_compile_calls,
            "body_reclosure_compile_calls": (
                self.body_reclosure_compile_calls
            ),
            "final_normalization_compile_calls": (
                self.final_normalization_compile_calls
            ),
            "output_interface_compile_calls": (
                self.output_interface_compile_calls
            ),
            "install_calls": self.install_calls,
            "validation_calls": self.validation_calls,
            "adapter_counter_checks": self.adapter_counter_checks,
            "torch_fold_only_guard_passed": (
                self.torch_fold_only_guard_passed
            ),
            "zero_parent_forward_tripwire_passed": (
                self.parent_forward_calls == 0
            ),
            "zero_training_tripwire_passed": all(
                getattr(self, name) == 0
                for name in (
                    "training_steps",
                    "backward_calls",
                    "optimizer_steps",
                    "optimizer_constructions",
                    "autograd_grad_calls",
                )
            ),
        }


class ReplayCaptureAdapter(Protocol):
    def replay_and_capture_child(
        self,
        scope: FoldScope,
        point: CapturePoint,
        records: Sequence[TrajectoryRecord],
        audit: ClosedLoopAudit,
    ) -> ChildTrajectoryCapture: ...


class ParentEndpointBridge(Protocol):
    def bridge_parent_endpoints(
        self,
        scope: FoldScope,
        records: Sequence[TrajectoryRecord],
        audit: ClosedLoopAudit,
    ) -> ParentEndpointBatch: ...


class ChildClosedLoopFoldAdapter(
    ReplayCaptureAdapter,
    ParentEndpointBridge,
    Protocol,
):
    """Architecture/modality adapter consumed by the generic orchestrator."""

    def layer_indices(self, audit: ClosedLoopAudit) -> Sequence[int]: ...

    def parent_trajectory_records(
        self, audit: ClosedLoopAudit
    ) -> Sequence[TrajectoryRecord]: ...

    def child_state_fingerprint(self) -> str: ...

    def child_topology_fingerprint(self) -> str: ...

    def audit_counters(self) -> Mapping[str, int]: ...

    def begin_child_transaction(
        self, audit: ClosedLoopAudit
    ) -> ChildStateTransaction: ...

    def compile_producer(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage: ...

    def install_producer(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt: ...

    def fit_consumer(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage: ...

    def install_consumer(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt: ...

    def compile_downstream(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage: ...

    def install_downstream(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt: ...

    def validate_layer(
        self,
        scope: FoldScope,
        validation_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> ValidationDecision: ...

    def compile_body_reclosure(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage: ...

    def install_body_reclosure(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt: ...

    def validate_body_reclosure(
        self,
        scope: FoldScope,
        validation_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> ValidationDecision: ...

    def compile_final_normalization(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage: ...

    def install_final_normalization(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt: ...

    def validate_final_normalization(
        self,
        scope: FoldScope,
        validation_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> ValidationDecision: ...

    def compile_output_interface(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage: ...

    def install_output_interface(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt: ...

    def validate_output_interface(
        self,
        scope: FoldScope,
        validation_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        baseline_capture: ChildTrajectoryCapture,
        candidate_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> ValidationDecision: ...


@dataclass(frozen=True, slots=True)
class ClosedLoopCompilerConfig:
    """Deterministic selection and holdout policy for one compiler run."""

    validation_fraction: float = 0.2
    split_seed: str = "neural-forge-closed-loop-v1"
    selected_paths: tuple[str, ...] | None = None
    selected_layers: tuple[int, ...] | None = None
    fold_body_reclosure: bool = False
    fold_output_interface: bool = True

    def __post_init__(self) -> None:
        fraction = self.validation_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0
        ):
            raise ClosedLoopFoldError(
                "validation_fraction must be finite and within (0, 1)."
            )
        object.__setattr__(self, "validation_fraction", float(fraction))
        object.__setattr__(self, "split_seed", _text(self.split_seed, "split_seed"))
        if self.selected_paths is not None:
            values = tuple(
                _text(value, "selected path") for value in self.selected_paths
            )
            if not values or len(values) != len(set(values)):
                raise ClosedLoopFoldError(
                    "selected_paths must be nonempty and unique when supplied."
                )
            object.__setattr__(self, "selected_paths", tuple(sorted(values)))
        if self.selected_layers is not None:
            values = tuple(
                _layer_index(value, "selected layer")
                for value in self.selected_layers
            )
            if not values or len(values) != len(set(values)):
                raise ClosedLoopFoldError(
                    "selected_layers must be nonempty and unique when supplied."
                )
            object.__setattr__(self, "selected_layers", tuple(sorted(values)))
        if not isinstance(self.fold_body_reclosure, bool):
            raise ClosedLoopFoldError(
                "fold_body_reclosure must be boolean."
            )
        if not isinstance(self.fold_output_interface, bool):
            raise ClosedLoopFoldError(
                "fold_output_interface must be boolean."
            )


class ChildClosedLoopCompiler:
    """Execute one all-or-nothing child-closed-loop fold."""

    def __init__(
        self,
        adapter: ChildClosedLoopFoldAdapter,
        config: ClosedLoopCompilerConfig | None = None,
        *,
        progress_sink: (
            Callable[[str, Mapping[str, JsonScalar]], None] | None
        ) = None,
    ) -> None:
        self.adapter = adapter
        self.config = (
            ClosedLoopCompilerConfig() if config is None else config
        )
        if not isinstance(self.config, ClosedLoopCompilerConfig):
            raise TypeError("config must be ClosedLoopCompilerConfig.")
        if progress_sink is not None and not callable(progress_sink):
            raise TypeError("progress_sink must be callable or None.")
        self.progress_sink = progress_sink

    def _emit_progress(self, stage: str, **metadata: JsonScalar) -> None:
        if self.progress_sink is not None:
            self.progress_sink(_text(stage, "progress stage"), metadata)

    def _adapter_audit(
        self,
        audit: ClosedLoopAudit,
        phase: str,
    ) -> None:
        audit.verify_adapter_counters(
            self.adapter.audit_counters(),
            phase=phase,
        )

    def _call(
        self,
        audit: ClosedLoopAudit,
        phase: str,
        callback: Callable[[], Any],
    ) -> Any:
        self._adapter_audit(audit, phase + ":before")
        value = callback()
        self._adapter_audit(audit, phase + ":after")
        audit.assert_clean()
        return value

    def _state(self) -> str:
        return _digest(
            self.adapter.child_state_fingerprint(),
            "child state fingerprint",
        )

    def _topology(self) -> str:
        return _digest(
            self.adapter.child_topology_fingerprint(),
            "child topology fingerprint",
        )

    def _endpoint_batch(
        self,
        scope: FoldScope,
        records: Sequence[TrajectoryRecord],
        audit: ClosedLoopAudit,
        phase: str,
    ) -> ParentEndpointBatch:
        audit.parent_endpoint_bridge_calls += 1
        result = self._call(
            audit,
            phase,
            lambda: self.adapter.bridge_parent_endpoints(
                scope, records, audit
            ),
        )
        if (
            not isinstance(result, ParentEndpointBatch)
            or result.scope != scope
            or result.record_keys != _record_keys(records)
        ):
            raise ClosedLoopFoldError(
                f"{phase} endpoint bridge did not preserve scope/trajectory "
                "coverage."
            )
        return result

    def _capture(
        self,
        scope: FoldScope,
        point: CapturePoint,
        records: Sequence[TrajectoryRecord],
        audit: ClosedLoopAudit,
    ) -> ChildTrajectoryCapture:
        audit.child_replay_capture_calls += 1
        result = self._call(
            audit,
            f"{scope.scope_id}:{point.value}",
            lambda: self.adapter.replay_and_capture_child(
                scope, point, records, audit
            ),
        )
        if (
            not isinstance(result, ChildTrajectoryCapture)
            or result.scope != scope
            or result.point is not point
            or result.record_keys != _record_keys(records)
        ):
            raise ClosedLoopFoldError(
                "Child replay/capture changed scope, point, or trajectory "
                "coverage."
            )
        return result

    def _compiled(
        self,
        result: Any,
        *,
        scope: FoldScope,
        stage: CompileStage,
        split: TrajectorySplit,
    ) -> CompiledFoldStage:
        if (
            not isinstance(result, CompiledFoldStage)
            or result.scope != scope
            or result.stage is not stage
            or result.fit_record_keys != split.fit_record_keys
        ):
            raise ClosedLoopFoldError(
                f"{scope.scope_id} {stage.value} compiler consumed anything "
                "other than the deterministic fit trajectories."
            )
        return result

    def _install(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
        installer: Callable[[], InstallReceipt],
        *,
        run_topology_sha256: str,
    ) -> InstallReceipt:
        before = self._state()
        before_topology = self._topology()
        audit.install_calls += 1
        receipt = self._call(
            audit,
            f"{scope.scope_id}:install_{compiled.stage.value}",
            installer,
        )
        after = self._state()
        after_topology = self._topology()
        if (
            not isinstance(receipt, InstallReceipt)
            or receipt.scope != scope
            or receipt.stage is not compiled.stage
            or receipt.artifact_sha256 != compiled.artifact_sha256
            or not receipt.accepted
            or not receipt.topology_preserved
            or receipt.state_sha256_before != before
            or receipt.state_sha256_after != after
            or before_topology != run_topology_sha256
            or after_topology != run_topology_sha256
        ):
            raise ClosedLoopFoldError(
                f"{scope.scope_id} {compiled.stage.value} install violated "
                "artifact, state, acceptance, or topology invariants."
            )
        return receipt

    def _validate(
        self,
        decision: Any,
        *,
        scope: FoldScope,
        split: TrajectorySplit,
    ) -> ValidationDecision:
        if (
            not isinstance(decision, ValidationDecision)
            or decision.scope != scope
            or decision.validation_record_keys
            != split.validation_record_keys
        ):
            raise ClosedLoopFoldError(
                f"{scope.scope_id} validation did not use exactly the held-out "
                "whole trajectories."
            )
        if not decision.accepted:
            raise ClosedLoopFoldError(
                f"{scope.scope_id} validation rejected the folded child: "
                f"{decision.reason}"
            )
        return decision

    def _run_layer(
        self,
        layer_index: int,
        split: TrajectorySplit,
        audit: ClosedLoopAudit,
        *,
        run_topology_sha256: str,
    ) -> dict[str, Any]:
        scope = FoldScope("layer", layer_index)
        fit_endpoints = self._endpoint_batch(
            scope,
            split.fit,
            audit,
            f"{scope.scope_id}:fit_parent_endpoints",
        )
        validation_endpoints = self._endpoint_batch(
            scope,
            split.validation,
            audit,
            f"{scope.scope_id}:validation_parent_endpoints",
        )
        pre_producer = self._capture(
            scope,
            CapturePoint.PRE_PRODUCER,
            split.fit,
            audit,
        )

        audit.producer_compile_calls += 1
        producer = self._compiled(
            self._call(
                audit,
                f"{scope.scope_id}:compile_producer",
                lambda: self.adapter.compile_producer(
                    scope,
                    split.fit,
                    fit_endpoints,
                    pre_producer,
                    audit,
                ),
            ),
            scope=scope,
            stage=CompileStage.PRODUCER,
            split=split,
        )
        producer_install = self._install(
            scope,
            producer,
            audit,
            lambda: self.adapter.install_producer(scope, producer, audit),
            run_topology_sha256=run_topology_sha256,
        )

        post_producer = self._capture(
            scope,
            CapturePoint.POST_PRODUCER,
            split.fit,
            audit,
        )
        audit.consumer_fit_calls += 1
        consumer = self._compiled(
            self._call(
                audit,
                f"{scope.scope_id}:fit_consumer",
                lambda: self.adapter.fit_consumer(
                    scope,
                    split.fit,
                    fit_endpoints,
                    post_producer,
                    audit,
                ),
            ),
            scope=scope,
            stage=CompileStage.CONSUMER,
            split=split,
        )
        consumer_install = self._install(
            scope,
            consumer,
            audit,
            lambda: self.adapter.install_consumer(scope, consumer, audit),
            run_topology_sha256=run_topology_sha256,
        )

        post_consumer = self._capture(
            scope,
            CapturePoint.POST_CONSUMER,
            split.fit,
            audit,
        )
        audit.downstream_compile_calls += 1
        downstream = self._compiled(
            self._call(
                audit,
                f"{scope.scope_id}:compile_downstream",
                lambda: self.adapter.compile_downstream(
                    scope,
                    split.fit,
                    fit_endpoints,
                    post_consumer,
                    audit,
                ),
            ),
            scope=scope,
            stage=CompileStage.DOWNSTREAM,
            split=split,
        )
        downstream_install = self._install(
            scope,
            downstream,
            audit,
            lambda: self.adapter.install_downstream(
                scope, downstream, audit
            ),
            run_topology_sha256=run_topology_sha256,
        )

        post_downstream = self._capture(
            scope,
            CapturePoint.POST_DOWNSTREAM,
            split.validation,
            audit,
        )
        audit.validation_calls += 1
        validation = self._validate(
            self._call(
                audit,
                f"{scope.scope_id}:validate",
                lambda: self.adapter.validate_layer(
                    scope,
                    split.validation,
                    validation_endpoints,
                    post_downstream,
                    audit,
                ),
            ),
            scope=scope,
            split=split,
        )
        return {
            "scope": scope.as_dict(),
            "fit_parent_endpoints": fit_endpoints.as_dict(),
            "validation_parent_endpoints": validation_endpoints.as_dict(),
            "captures": {
                "pre_producer_fit": pre_producer.as_dict(),
                "post_producer_fit": post_producer.as_dict(),
                "post_consumer_fit": post_consumer.as_dict(),
                "post_downstream_validation": post_downstream.as_dict(),
            },
            "producer": {
                "compiled": producer.as_dict(),
                "install": producer_install.as_dict(),
            },
            "consumer": {
                "compiled": consumer.as_dict(),
                "install": consumer_install.as_dict(),
                "fitted_after_installed_producer_recapture": True,
            },
            "downstream": {
                "compiled": downstream.as_dict(),
                "install": downstream_install.as_dict(),
                "compiled_after_installed_consumer_recapture": True,
            },
            "validation": validation.as_dict(),
            "fit_validation_trajectory_leakage": False,
            "child_closed_loop": True,
        }

    def _run_final_normalization(
        self,
        split: TrajectorySplit,
        audit: ClosedLoopAudit,
        *,
        run_topology_sha256: str,
    ) -> dict[str, Any]:
        scope = FoldScope("final_normalization")
        fit_endpoints = self._endpoint_batch(
            scope,
            split.fit,
            audit,
            "final_normalization:fit_parent_endpoints",
        )
        validation_endpoints = self._endpoint_batch(
            scope,
            split.validation,
            audit,
            "final_normalization:validation_parent_endpoints",
        )
        pre_final = self._capture(
            scope,
            CapturePoint.PRE_FINAL_NORMALIZATION,
            split.fit,
            audit,
        )
        audit.final_normalization_compile_calls += 1
        compiled = self._compiled(
            self._call(
                audit,
                "final_normalization:compile",
                lambda: self.adapter.compile_final_normalization(
                    scope,
                    split.fit,
                    fit_endpoints,
                    pre_final,
                    audit,
                ),
            ),
            scope=scope,
            stage=CompileStage.FINAL_NORMALIZATION,
            split=split,
        )
        install = self._install(
            scope,
            compiled,
            audit,
            lambda: self.adapter.install_final_normalization(
                scope, compiled, audit
            ),
            run_topology_sha256=run_topology_sha256,
        )
        post_final = self._capture(
            scope,
            CapturePoint.POST_FINAL_NORMALIZATION,
            split.validation,
            audit,
        )
        audit.validation_calls += 1
        validation = self._validate(
            self._call(
                audit,
                "final_normalization:validate",
                lambda: self.adapter.validate_final_normalization(
                    scope,
                    split.validation,
                    validation_endpoints,
                    post_final,
                    audit,
                ),
            ),
            scope=scope,
            split=split,
        )
        return {
            "scope": scope.as_dict(),
            "fit_parent_endpoints": fit_endpoints.as_dict(),
            "validation_parent_endpoints": validation_endpoints.as_dict(),
            "captures": {
                "pre_final_normalization_fit": pre_final.as_dict(),
                "post_final_normalization_validation": post_final.as_dict(),
            },
            "compiled": compiled.as_dict(),
            "install": install.as_dict(),
            "validation": validation.as_dict(),
            "runs_after_all_ascending_layers": True,
        }

    def _run_body_reclosure(
        self,
        split: TrajectorySplit,
        audit: ClosedLoopAudit,
        *,
        run_topology_sha256: str,
    ) -> dict[str, Any]:
        """Compile one whole-body residual repair before terminal fitting."""

        scope = FoldScope("body_reclosure")
        fit_endpoints = self._endpoint_batch(
            scope,
            split.fit,
            audit,
            "body_reclosure:fit_parent_endpoints",
        )
        validation_endpoints = self._endpoint_batch(
            scope,
            split.validation,
            audit,
            "body_reclosure:validation_parent_endpoints",
        )
        pre_body = self._capture(
            scope,
            CapturePoint.PRE_BODY_RECLOSURE,
            split.fit,
            audit,
        )
        audit.body_reclosure_compile_calls += 1
        compiled = self._compiled(
            self._call(
                audit,
                "body_reclosure:compile",
                lambda: self.adapter.compile_body_reclosure(
                    scope,
                    split.fit,
                    fit_endpoints,
                    pre_body,
                    audit,
                ),
            ),
            scope=scope,
            stage=CompileStage.BODY_RECLOSURE,
            split=split,
        )
        install = self._install(
            scope,
            compiled,
            audit,
            lambda: self.adapter.install_body_reclosure(
                scope, compiled, audit
            ),
            run_topology_sha256=run_topology_sha256,
        )
        post_body = self._capture(
            scope,
            CapturePoint.POST_BODY_RECLOSURE,
            split.validation,
            audit,
        )
        audit.validation_calls += 1
        validation = self._validate(
            self._call(
                audit,
                "body_reclosure:validate",
                lambda: self.adapter.validate_body_reclosure(
                    scope,
                    split.validation,
                    validation_endpoints,
                    post_body,
                    audit,
                ),
            ),
            scope=scope,
            split=split,
        )
        return {
            "scope": scope.as_dict(),
            "mode": "compiled_body_reclosure",
            "fold_enabled": True,
            "fit_parent_endpoints": fit_endpoints.as_dict(),
            "validation_parent_endpoints": validation_endpoints.as_dict(),
            "captures": {
                "pre_body_reclosure_fit": pre_body.as_dict(),
                "post_body_reclosure_validation": post_body.as_dict(),
            },
            "compiled": compiled.as_dict(),
            "install": install.as_dict(),
            "validation": validation.as_dict(),
            "runs_after_all_ascending_layers": True,
            "runs_before_final_normalization": True,
            "fit_validation_trajectory_leakage": False,
        }

    def _disabled_body_reclosure_report(
        self,
        *,
        state_sha256: str,
        topology_sha256: str,
    ) -> dict[str, Any]:
        body = {
            "schema": (
                "neural-forge-dynamic-body-reclosure-explicit-identity-v1"
            ),
            "scope": FoldScope("body_reclosure").as_dict(),
            "mode": "explicit_identity_preservation",
            "fold_enabled": False,
            "compiled": None,
            "install": None,
            "validation": None,
            "child_state_sha256_before": state_sha256,
            "child_state_sha256_after": state_sha256,
            "child_topology_sha256": topology_sha256,
            "child_parameter_mutation_requested": False,
            "runs_after_all_ascending_layers": True,
            "runs_before_final_normalization": True,
            "identity_is_not_a_behavioral_acceptance": True,
        }
        return {**body, "report_sha256": canonical_sha256(body)}

    def _run_output_interface(
        self,
        split: TrajectorySplit,
        audit: ClosedLoopAudit,
        *,
        run_topology_sha256: str,
    ) -> dict[str, Any]:
        """Compile the terminal interface and validate its tied feedback live."""

        scope = FoldScope("output_interface")
        fit_endpoints = self._endpoint_batch(
            scope,
            split.fit,
            audit,
            "output_interface:fit_parent_endpoints",
        )
        validation_endpoints = self._endpoint_batch(
            scope,
            split.validation,
            audit,
            "output_interface:validation_parent_endpoints",
        )
        pre_fit = self._capture(
            scope,
            CapturePoint.PRE_OUTPUT_INTERFACE,
            split.fit,
            audit,
        )
        pre_validation = self._capture(
            scope,
            CapturePoint.PRE_OUTPUT_INTERFACE,
            split.validation,
            audit,
        )
        audit.output_interface_compile_calls += 1
        compiled = self._compiled(
            self._call(
                audit,
                "output_interface:compile",
                lambda: self.adapter.compile_output_interface(
                    scope,
                    split.fit,
                    fit_endpoints,
                    pre_fit,
                    audit,
                ),
            ),
            scope=scope,
            stage=CompileStage.OUTPUT_INTERFACE,
            split=split,
        )
        install = self._install(
            scope,
            compiled,
            audit,
            lambda: self.adapter.install_output_interface(
                scope, compiled, audit
            ),
            run_topology_sha256=run_topology_sha256,
        )
        post_validation = self._capture(
            scope,
            CapturePoint.POST_OUTPUT_INTERFACE,
            split.validation,
            audit,
        )
        audit.validation_calls += 1
        validation = self._validate(
            self._call(
                audit,
                "output_interface:validate",
                lambda: self.adapter.validate_output_interface(
                    scope,
                    split.validation,
                    validation_endpoints,
                    pre_validation,
                    post_validation,
                    audit,
                ),
            ),
            scope=scope,
            split=split,
        )
        return {
            "scope": scope.as_dict(),
            "fit_parent_endpoints": fit_endpoints.as_dict(),
            "validation_parent_endpoints": validation_endpoints.as_dict(),
            "captures": {
                "pre_output_interface_fit": pre_fit.as_dict(),
                "pre_output_interface_validation_baseline": (
                    pre_validation.as_dict()
                ),
                "post_output_interface_validation_live_child": (
                    post_validation.as_dict()
                ),
            },
            "compiled": compiled.as_dict(),
            "install": install.as_dict(),
            "validation": validation.as_dict(),
            "runs_after_final_normalization": True,
            "pre_install_heldout_baseline_captured": True,
            "post_install_full_live_child_replayed": True,
            "teacher_forced_terminal_acceptance_is_necessary_not_sufficient": (
                True
            ),
            "external_whole_output_behavior_gate_required_for_promotion": True,
            "promotion_authority": (
                "external_whole_output_semantic_behavior_gate"
            ),
        }

    def _rollback_exact(
        self,
        transaction: ChildStateTransaction,
        *,
        state_before: str,
        topology_before: str,
    ) -> TransactionReceipt:
        if not bool(getattr(transaction, "active", False)):
            raise ClosedLoopFoldError(
                "Failed closed-loop run left no active transaction to roll back."
            )
        receipt = transaction.rollback()
        state_after = self._state()
        topology_after = self._topology()
        if (
            not isinstance(receipt, TransactionReceipt)
            or receipt.action != "rollback"
            or receipt.accepted
            or not receipt.exact
            or receipt.state_sha256_before != state_before
            or receipt.state_sha256_after != state_before
            or state_after != state_before
            or topology_after != topology_before
            or bool(getattr(transaction, "active", True))
        ):
            raise ClosedLoopFoldError(
                "Closed-loop rollback was not exact or did not deactivate."
            )
        return receipt

    def _execute_guarded(self, audit: ClosedLoopAudit) -> dict[str, Any]:
        self._adapter_audit(audit, "run:entry")
        state_before = self._state()
        topology_before = self._topology()
        transaction = self._call(
            audit,
            "transaction:begin",
            lambda: self.adapter.begin_child_transaction(audit),
        )
        try:
            if (
                not bool(getattr(transaction, "active", False))
                or _digest(
                    getattr(transaction, "state_sha256_before", None),
                    "transaction state_sha256_before",
                )
                != state_before
                or _digest(
                    getattr(transaction, "topology_sha256_before", None),
                    "transaction topology_sha256_before",
                )
                != topology_before
            ):
                raise ClosedLoopFoldError(
                    "Child transaction did not bind the exact initial "
                    "state/topology."
                )
            raw_layers = self._call(
                audit,
                "layers:discover",
                lambda: self.adapter.layer_indices(audit),
            )
            raw_layers = tuple(raw_layers)
            available_layers = tuple(
                sorted(
                    {
                        _layer_index(value, "available layer")
                        for value in raw_layers
                    }
                )
            )
            if not available_layers or len(available_layers) != len(
                raw_layers
            ):
                raise ClosedLoopFoldError(
                    "Adapter layers must be nonempty and unique."
                )
            if self.config.selected_layers is None:
                selected_layers = available_layers
            else:
                selected_layers = self.config.selected_layers
                if not set(selected_layers).issubset(available_layers):
                    raise ClosedLoopFoldError(
                        "selected_layers contains an unavailable layer."
                    )

            records = tuple(
                self._call(
                    audit,
                    "parent_trajectories:load_sealed",
                    lambda: self.adapter.parent_trajectory_records(audit),
                )
            )
            split = deterministic_whole_trajectory_split(
                records,
                validation_fraction=self.config.validation_fraction,
                split_seed=self.config.split_seed,
                selected_paths=self.config.selected_paths,
            )
            self._emit_progress(
                "child_closed_loop_compiler_started",
                selected_layer_count=len(selected_layers),
                body_reclosure_enabled=self.config.fold_body_reclosure,
                output_interface_enabled=self.config.fold_output_interface,
            )
            layer_report_values = []
            for layer_ordinal, layer_index in enumerate(selected_layers):
                self._emit_progress(
                    "child_closed_loop_layer_started",
                    layer_index=layer_index,
                    layer_ordinal=layer_ordinal,
                    selected_layer_count=len(selected_layers),
                )
                layer_report_values.append(
                    self._run_layer(
                        layer_index,
                        split,
                        audit,
                        run_topology_sha256=topology_before,
                    )
                )
                self._emit_progress(
                    "child_closed_loop_layer_completed",
                    layer_index=layer_index,
                    layer_ordinal=layer_ordinal,
                    selected_layer_count=len(selected_layers),
                )
            layer_reports = tuple(layer_report_values)
            if self.config.fold_body_reclosure:
                self._emit_progress(
                    "child_closed_loop_body_reclosure_started",
                    selected_layer_count=len(selected_layers),
                )
                body_reclosure_report = self._run_body_reclosure(
                    split,
                    audit,
                    run_topology_sha256=topology_before,
                )
                self._emit_progress(
                    "child_closed_loop_body_reclosure_completed",
                    selected_layer_count=len(selected_layers),
                )
            else:
                self._emit_progress(
                    "child_closed_loop_body_reclosure_skipped",
                    selected_layer_count=len(selected_layers),
                )
                body_reclosure_report = (
                    self._disabled_body_reclosure_report(
                        state_sha256=self._state(),
                        topology_sha256=self._topology(),
                    )
                )
            self._emit_progress(
                "child_closed_loop_final_normalization_started",
                selected_layer_count=len(selected_layers),
            )
            final_report = self._run_final_normalization(
                split,
                audit,
                run_topology_sha256=topology_before,
            )
            self._emit_progress(
                "child_closed_loop_final_normalization_completed",
                selected_layer_count=len(selected_layers),
            )
            if self.config.fold_output_interface:
                self._emit_progress(
                    "child_closed_loop_output_interface_started",
                    selected_layer_count=len(selected_layers),
                )
                output_interface_report = self._run_output_interface(
                    split,
                    audit,
                    run_topology_sha256=topology_before,
                )
                self._emit_progress(
                    "child_closed_loop_output_interface_completed",
                    selected_layer_count=len(selected_layers),
                )
            else:
                self._emit_progress(
                    "child_closed_loop_output_interface_skipped",
                    selected_layer_count=len(selected_layers),
                )
                output_interface_report = {
                    "scope": FoldScope("output_interface").as_dict(),
                    "mode": "explicit_identity_preservation",
                    "fold_enabled": False,
                    "compiled": None,
                    "install": None,
                    "validation": None,
                    "runs_after_final_normalization": True,
                    "output_parameter_mutation_requested": False,
                    "identity_is_not_a_behavioral_acceptance": True,
                    "external_whole_output_behavior_gate_required_for_promotion": (
                        True
                    ),
                    "promotion_authority": (
                        "external_whole_output_semantic_behavior_gate"
                    ),
                }
            candidate_state = self._state()
            if self._topology() != topology_before:
                raise ClosedLoopFoldError(
                    "Closed-loop compilation changed child topology."
                )
            audit.assert_clean()
            self._emit_progress(
                "child_closed_loop_compiler_candidate_ready",
                selected_layer_count=len(selected_layers),
                body_reclosure_enabled=self.config.fold_body_reclosure,
                output_interface_enabled=self.config.fold_output_interface,
                candidate_state_sha256=candidate_state,
            )
            commit = self._call(
                audit,
                "transaction:commit",
                transaction.commit,
            )
            committed_state = self._state()
            if (
                not isinstance(commit, TransactionReceipt)
                or commit.action != "commit"
                or not commit.accepted
                or not commit.exact
                or commit.state_sha256_before != state_before
                or commit.state_sha256_after != committed_state
                or committed_state != candidate_state
                or bool(getattr(transaction, "active", True))
                or self._topology() != topology_before
            ):
                raise ClosedLoopFoldError(
                    "Child transaction commit violated state/topology invariants."
                )
            return {
                "schema": RUN_SCHEMA,
                "trajectory_split": split.as_dict(),
                "available_layers": list(available_layers),
                "selected_layers": list(selected_layers),
                "default_all_layers": selected_layers == available_layers,
                "layers_executed_in_ascending_order": True,
                "layer_reports": list(layer_reports),
                "body_reclosure": body_reclosure_report,
                "body_reclosure_fold_enabled": (
                    self.config.fold_body_reclosure
                ),
                "final_normalization": final_report,
                "output_interface": output_interface_report,
                "output_interface_fold_enabled": (
                    self.config.fold_output_interface
                ),
                "child_state_sha256_before": state_before,
                "child_state_sha256_after": committed_state,
                "child_topology_sha256": topology_before,
                "child_topology_preserved": True,
                "transaction": commit.as_dict(),
                "transaction_committed": True,
                "transaction_rolls_back_precommit_failure": True,
                "fallible_progress_callbacks_after_commit": 0,
                "parent_trajectory_source": "sealed-cache-adapter",
                "parent_forward_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "teacher_forced_terminal_acceptance_is_necessary_not_sufficient": (
                    True
                ),
                "external_whole_output_behavior_gate_required_for_promotion": (
                    True
                ),
                "promotion_authority": (
                    "external_whole_output_semantic_behavior_gate"
                ),
            }
        except BaseException:
            if bool(getattr(transaction, "active", False)):
                self._rollback_exact(
                    transaction,
                    state_before=state_before,
                    topology_before=topology_before,
                )
            elif (
                self._state() != state_before
                or self._topology() != topology_before
            ):
                raise ClosedLoopFoldError(
                    "An inactive failed transaction left mutated child state; "
                    "exact rollback is impossible."
                )
            raise

    def run(self) -> dict[str, Any]:
        """Compile, validate, and atomically commit one closed-loop child."""

        audit = ClosedLoopAudit()
        report, guard = run_fold_only_callback(
            lambda: self._execute_guarded(audit),
            phase="dynamic_child_closed_loop_fold",
        )
        audit.absorb_torch_guard(guard)
        report["audit"] = audit.as_dict()
        report["run_sha256"] = canonical_sha256(report)
        return report


__all__ = [
    "CapturePoint",
    "ChildClosedLoopCompiler",
    "ChildClosedLoopFoldAdapter",
    "ChildStateTransaction",
    "ChildTrajectoryCapture",
    "ClosedLoopAudit",
    "ClosedLoopAuditTripwire",
    "ClosedLoopCompilerConfig",
    "ClosedLoopFoldError",
    "CompileStage",
    "CompiledFoldStage",
    "FoldScope",
    "InstallReceipt",
    "ParentEndpointBatch",
    "ParentEndpointBridge",
    "ReplayCaptureAdapter",
    "RUN_SCHEMA",
    "SPLIT_SCHEMA",
    "TrajectoryRecord",
    "TrajectorySplit",
    "TransactionReceipt",
    "ValidationDecision",
    "deterministic_whole_trajectory_split",
]
