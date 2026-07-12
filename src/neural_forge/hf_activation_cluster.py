"""Deterministic real-medoid plans for nonlinear attention and MLP circuits.

The plans in this module are structural, zero-gradient objects.  Producers are
never averaged: a target producer is copied from a real source head or neuron.
Every source circuit is assigned once to a target medoid, and only consumer
directions are aggregated.  Consumer scalars are fitted in float64 from the
activation immediately consumed by the linear consumer (attention context or
post-SwiGLU activity), not by dividing one nonlinear factor by another.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class ActivationClusterError(ValueError):
    """Base class for activation-cluster planning failures."""


class ActivationClusterShapeError(ActivationClusterError):
    """Raised when trace topology or tensor shapes are not trustworthy."""


class ActivationClusterFiniteError(ActivationClusterError):
    """Raised when an input or computed coefficient is non-finite."""


class ActivationClusterInvariantError(ActivationClusterError):
    """Raised when a plan cannot prove its structural invariants."""


_GLOBAL_CLUSTER_MAX_SOURCES = 256
_GLOBAL_CLUSTER_MAX_SOURCE_TARGET_PAIRS = 16_384
_LOCALITY_DIRECTION_DIMENSIONS = 4
_MORTON_CODE_BITS = 60
_GLOBAL_CLUSTER_STRATEGY = (
    "deterministic-farthest-first-global-assignment-real-medoid-refinement"
)
_SCALABLE_CLUSTER_STRATEGY = (
    "deterministic-normalized-projection-morton-balanced-exact-local-medoid"
)
_IDENTITY_CLUSTER_STRATEGY = "deterministic-identity-real-medoid"
_CLUSTER_MEDOID_SELECTION = "minimum-within-cluster-squared-distance-real-source"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for activation-cluster planning."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActivationClusterShapeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ActivationClusterShapeError(
            f"{name} must be a finite non-negative number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ActivationClusterShapeError(
            f"{name} must be a finite non-negative number."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ActivationClusterShapeError(
            f"{name} must be a finite non-negative number."
        )
    return result


def _all_true(values: tuple[bool, ...]) -> bool:
    return all(value is True for value in values)


@dataclass(frozen=True, slots=True)
class GQAActivationClusterInvariants:
    """Proof obligations for a hierarchical GQA medoid plan."""

    source_groups_assigned_exactly_once: bool
    source_query_heads_assigned_exactly_once: bool
    nonempty_target_groups: bool
    nonempty_target_query_heads: bool
    real_source_group_medoids: bool
    real_source_query_head_medoids: bool
    medoids_self_assigned: bool
    gqa_group_offsets_preserved: bool
    rope_head_dimension_preserved: bool
    finite_consumer_coefficients: bool
    no_producer_averaging: bool
    zero_gradient: bool

    @property
    def all_hold(self) -> bool:
        return _all_true(
            (
                self.source_groups_assigned_exactly_once,
                self.source_query_heads_assigned_exactly_once,
                self.nonempty_target_groups,
                self.nonempty_target_query_heads,
                self.real_source_group_medoids,
                self.real_source_query_head_medoids,
                self.medoids_self_assigned,
                self.gqa_group_offsets_preserved,
                self.rope_head_dimension_preserved,
                self.finite_consumer_coefficients,
                self.no_producer_averaging,
                self.zero_gradient,
            )
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "source_groups_assigned_exactly_once": (
                self.source_groups_assigned_exactly_once
            ),
            "source_query_heads_assigned_exactly_once": (
                self.source_query_heads_assigned_exactly_once
            ),
            "nonempty_target_groups": self.nonempty_target_groups,
            "nonempty_target_query_heads": self.nonempty_target_query_heads,
            "real_source_group_medoids": self.real_source_group_medoids,
            "real_source_query_head_medoids": (self.real_source_query_head_medoids),
            "medoids_self_assigned": self.medoids_self_assigned,
            "gqa_group_offsets_preserved": self.gqa_group_offsets_preserved,
            "rope_head_dimension_preserved": self.rope_head_dimension_preserved,
            "finite_consumer_coefficients": self.finite_consumer_coefficients,
            "no_producer_averaging": self.no_producer_averaging,
            "zero_gradient": self.zero_gradient,
            "all_hold": self.all_hold,
        }


@dataclass(frozen=True, slots=True)
class GatedMLPActivationClusterInvariants:
    """Proof obligations for a coupled gated-MLP medoid plan."""

    source_neurons_assigned_exactly_once: bool
    nonempty_target_neurons: bool
    real_source_neuron_medoids: bool
    medoids_self_assigned: bool
    shared_gate_up_down_assignment: bool
    finite_consumer_coefficients: bool
    coefficients_fit_from_post_swiglu: bool
    no_gate_value_division: bool
    no_producer_averaging: bool
    zero_gradient: bool

    @property
    def all_hold(self) -> bool:
        return _all_true(
            (
                self.source_neurons_assigned_exactly_once,
                self.nonempty_target_neurons,
                self.real_source_neuron_medoids,
                self.medoids_self_assigned,
                self.shared_gate_up_down_assignment,
                self.finite_consumer_coefficients,
                self.coefficients_fit_from_post_swiglu,
                self.no_gate_value_division,
                self.no_producer_averaging,
                self.zero_gradient,
            )
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "source_neurons_assigned_exactly_once": (
                self.source_neurons_assigned_exactly_once
            ),
            "nonempty_target_neurons": self.nonempty_target_neurons,
            "real_source_neuron_medoids": self.real_source_neuron_medoids,
            "medoids_self_assigned": self.medoids_self_assigned,
            "shared_gate_up_down_assignment": (self.shared_gate_up_down_assignment),
            "finite_consumer_coefficients": self.finite_consumer_coefficients,
            "coefficients_fit_from_post_swiglu": (
                self.coefficients_fit_from_post_swiglu
            ),
            "no_gate_value_division": self.no_gate_value_division,
            "no_producer_averaging": self.no_producer_averaging,
            "zero_gradient": self.zero_gradient,
            "all_hold": self.all_hold,
        }


@dataclass(frozen=True, slots=True)
class GQAActivationClusterReport:
    """Immutable topology, clustering, fit, and invariant evidence for GQA."""

    source_query_heads: int
    source_key_value_heads: int
    target_query_heads: int
    target_key_value_heads: int
    source_query_heads_per_group: int
    target_query_heads_per_group: int
    head_dim: int
    group_medoids: tuple[int, ...]
    query_offset_medoids: tuple[int, ...]
    query_head_medoids: tuple[int, ...]
    group_assignment: tuple[int, ...]
    query_offset_assignment: tuple[int, ...]
    query_head_assignment: tuple[int, ...]
    group_sizes: tuple[int, ...]
    query_offset_group_sizes: tuple[int, ...]
    query_head_group_sizes: tuple[int, ...]
    context_consumer_coefficients: tuple[float, ...]
    group_medoid_candidate_indices: tuple[int, ...]
    group_cluster_strategy: str
    query_offset_cluster_strategy: str
    group_cluster_balanced_partition: bool
    query_offset_cluster_balanced_partition: bool
    group_cluster_locality_dimensions: int
    query_offset_cluster_locality_dimensions: int
    group_cluster_iterations: int
    query_offset_cluster_iterations: int
    group_cluster_converged: bool
    query_offset_cluster_converged: bool
    group_cluster_squared_error: float
    query_offset_cluster_squared_error: float
    context_fit_relative_error: float
    context_fit_effective_ridge: float
    near_zero_context_medoid_count: int
    invariants: GQAActivationClusterInvariants

    @property
    def full_source_coverage(self) -> bool:
        return (
            self.invariants.source_groups_assigned_exactly_once
            and self.invariants.source_query_heads_assigned_exactly_once
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "deterministic-hierarchical-gqa-real-medoid",
            "source_query_heads": self.source_query_heads,
            "source_key_value_heads": self.source_key_value_heads,
            "target_query_heads": self.target_query_heads,
            "target_key_value_heads": self.target_key_value_heads,
            "source_query_heads_per_group": (self.source_query_heads_per_group),
            "target_query_heads_per_group": (self.target_query_heads_per_group),
            "head_dim": self.head_dim,
            "source_head_dim": self.head_dim,
            "target_head_dim": self.head_dim,
            "group_medoids": list(self.group_medoids),
            "query_offset_medoids": list(self.query_offset_medoids),
            "query_head_medoids": list(self.query_head_medoids),
            "group_assignment": list(self.group_assignment),
            "query_offset_assignment": list(self.query_offset_assignment),
            "query_head_assignment": list(self.query_head_assignment),
            "group_sizes": list(self.group_sizes),
            "query_offset_group_sizes": list(self.query_offset_group_sizes),
            "query_head_group_sizes": list(self.query_head_group_sizes),
            "context_consumer_coefficients": list(self.context_consumer_coefficients),
            "group_medoid_candidate_indices": list(self.group_medoid_candidate_indices),
            "group_cluster_strategy": self.group_cluster_strategy,
            "query_offset_cluster_strategy": self.query_offset_cluster_strategy,
            "group_cluster_balanced_partition": (self.group_cluster_balanced_partition),
            "query_offset_cluster_balanced_partition": (
                self.query_offset_cluster_balanced_partition
            ),
            "group_cluster_locality_dimensions": (
                self.group_cluster_locality_dimensions
            ),
            "query_offset_cluster_locality_dimensions": (
                self.query_offset_cluster_locality_dimensions
            ),
            "cluster_medoid_selection": _CLUSTER_MEDOID_SELECTION,
            "group_cluster_iterations": self.group_cluster_iterations,
            "query_offset_cluster_iterations": (self.query_offset_cluster_iterations),
            "group_cluster_converged": self.group_cluster_converged,
            "query_offset_cluster_converged": (self.query_offset_cluster_converged),
            "group_cluster_squared_error": self.group_cluster_squared_error,
            "query_offset_cluster_squared_error": (
                self.query_offset_cluster_squared_error
            ),
            "context_fit_relative_error": self.context_fit_relative_error,
            "context_fit_effective_ridge": self.context_fit_effective_ridge,
            "near_zero_context_medoid_count": (self.near_zero_context_medoid_count),
            "producer_selection": "copy-real-source-head-block",
            "producer_aggregation": "none",
            "consumer_aggregation": "post-context-least-squares-index-add",
            "full_source_coverage": self.full_source_coverage,
            "full_source_group_coverage": (
                self.invariants.source_groups_assigned_exactly_once
            ),
            "full_source_query_head_coverage": (
                self.invariants.source_query_heads_assigned_exactly_once
            ),
            "invariants": self.invariants.as_dict(),
            "deterministic": True,
            "zero_gradient": True,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class GatedMLPActivationClusterReport:
    """Immutable topology, clustering, fit, and invariant evidence for an MLP."""

    source_neurons: int
    target_neurons: int
    observation_rows: int
    medoid_source_neurons: tuple[int, ...]
    source_to_target: tuple[int, ...]
    group_sizes: tuple[int, ...]
    consumer_coefficients: tuple[float, ...]
    cluster_strategy: str
    cluster_balanced_partition: bool
    cluster_locality_dimensions: int
    cluster_iterations: int
    cluster_converged: bool
    cluster_squared_error: float
    post_swiglu_fit_relative_error: float
    post_swiglu_fit_effective_ridge: float
    near_zero_post_swiglu_medoid_count: int
    maximum_absolute_consumer_coefficient: float
    negative_consumer_coefficient_count: int
    invariants: GatedMLPActivationClusterInvariants

    @property
    def full_source_coverage(self) -> bool:
        return self.invariants.source_neurons_assigned_exactly_once

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "deterministic-coupled-gated-mlp-real-medoid",
            "source_neurons": self.source_neurons,
            "target_neurons": self.target_neurons,
            "observation_rows": self.observation_rows,
            "medoid_source_neurons": list(self.medoid_source_neurons),
            "source_to_target": list(self.source_to_target),
            "group_sizes": list(self.group_sizes),
            "consumer_coefficients": list(self.consumer_coefficients),
            "cluster_strategy": self.cluster_strategy,
            "cluster_balanced_partition": self.cluster_balanced_partition,
            "cluster_locality_dimensions": self.cluster_locality_dimensions,
            "cluster_medoid_selection": _CLUSTER_MEDOID_SELECTION,
            "cluster_iterations": self.cluster_iterations,
            "cluster_converged": self.cluster_converged,
            "cluster_squared_error": self.cluster_squared_error,
            "post_swiglu_fit_relative_error": (self.post_swiglu_fit_relative_error),
            "post_swiglu_fit_effective_ridge": (self.post_swiglu_fit_effective_ridge),
            "near_zero_post_swiglu_medoid_count": (
                self.near_zero_post_swiglu_medoid_count
            ),
            "maximum_absolute_consumer_coefficient": (
                self.maximum_absolute_consumer_coefficient
            ),
            "negative_consumer_coefficient_count": (
                self.negative_consumer_coefficient_count
            ),
            "signature_modalities": [
                "gate",
                "up",
                "post_swiglu",
                "contribution",
            ],
            "producer_selection": "copy-real-source-neuron-row",
            "producer_aggregation": "none",
            "consumer_aggregation": "post-swiglu-least-squares-index-add",
            "consumer_coefficient_gate_division": False,
            "full_source_coverage": self.full_source_coverage,
            "invariants": self.invariants.as_dict(),
            "deterministic": True,
            "zero_gradient": True,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _tuple_counts(assignment: tuple[int, ...], target_count: int) -> tuple[int, ...]:
    counts = [0] * target_count
    for target in assignment:
        if not 0 <= target < target_count:
            raise ActivationClusterInvariantError(
                "An assignment contains an out-of-range target."
            )
        counts[target] += 1
    return tuple(counts)


def _validate_real_medoids(
    medoids: tuple[int, ...],
    assignment: tuple[int, ...],
    source_count: int,
    target_count: int,
    name: str,
) -> None:
    if len(medoids) != target_count or len(set(medoids)) != target_count:
        raise ActivationClusterInvariantError(
            f"{name} must contain one unique source index per target."
        )
    if any(not 0 <= medoid < source_count for medoid in medoids):
        raise ActivationClusterInvariantError(
            f"{name} contains an out-of-range source index."
        )
    if len(assignment) != source_count:
        raise ActivationClusterInvariantError(
            f"{name} assignment does not cover every source exactly once."
        )
    counts = _tuple_counts(assignment, target_count)
    if any(count <= 0 for count in counts):
        raise ActivationClusterInvariantError(f"{name} leaves a target cluster empty.")
    if any(assignment[source] != target for target, source in enumerate(medoids)):
        raise ActivationClusterInvariantError(
            f"{name} medoids must be assigned to their own targets."
        )


@dataclass(frozen=True, slots=True)
class GQAActivationClusterPlan:
    """A full-coverage GQA plan with real source producers and LS consumers."""

    source_query_heads: int
    source_key_value_heads: int
    target_query_heads: int
    target_key_value_heads: int
    head_dim: int
    group_medoids: tuple[int, ...]
    query_offset_medoids: tuple[int, ...]
    query_head_medoids: tuple[int, ...]
    group_assignment: tuple[int, ...]
    query_offset_assignment: tuple[int, ...]
    query_head_assignment: tuple[int, ...]
    context_consumer_coefficients: tuple[float, ...]
    report: GQAActivationClusterReport

    def __post_init__(self) -> None:
        source_q = _positive_int(self.source_query_heads, "source_query_heads")
        source_kv = _positive_int(self.source_key_value_heads, "source_key_value_heads")
        target_q = _positive_int(self.target_query_heads, "target_query_heads")
        target_kv = _positive_int(self.target_key_value_heads, "target_key_value_heads")
        head_dim = _positive_int(self.head_dim, "head_dim")
        if head_dim % 2:
            raise ActivationClusterInvariantError(
                "A GQA plan must preserve an even RoPE head dimension."
            )
        if source_q % source_kv or target_q % target_kv:
            raise ActivationClusterInvariantError(
                "Query heads must divide exactly into GQA groups."
            )
        source_offsets = source_q // source_kv
        target_offsets = target_q // target_kv
        _validate_real_medoids(
            self.group_medoids,
            self.group_assignment,
            source_kv,
            target_kv,
            "group_medoids",
        )
        _validate_real_medoids(
            self.query_offset_medoids,
            self.query_offset_assignment,
            source_offsets,
            target_offsets,
            "query_offset_medoids",
        )
        expected_assignment = tuple(
            self.group_assignment[group] * target_offsets
            + self.query_offset_assignment[offset]
            for group in range(source_kv)
            for offset in range(source_offsets)
        )
        expected_medoids = tuple(
            self.group_medoids[group] * source_offsets
            + self.query_offset_medoids[offset]
            for group in range(target_kv)
            for offset in range(target_offsets)
        )
        if self.query_head_assignment != expected_assignment:
            raise ActivationClusterInvariantError(
                "Query-head assignment does not preserve GQA group offsets."
            )
        if self.query_head_medoids != expected_medoids:
            raise ActivationClusterInvariantError(
                "Query-head medoids are not real Cartesian source heads."
            )
        _validate_real_medoids(
            self.query_head_medoids,
            self.query_head_assignment,
            source_q,
            target_q,
            "query_head_medoids",
        )
        if len(self.context_consumer_coefficients) != source_q or not all(
            math.isfinite(value) for value in self.context_consumer_coefficients
        ):
            raise ActivationClusterInvariantError(
                "Context consumer coefficients must be finite and cover every head."
            )
        if not isinstance(self.report, GQAActivationClusterReport):
            raise ActivationClusterInvariantError(
                "report must be a GQAActivationClusterReport."
            )
        if (
            self.report.group_assignment != self.group_assignment
            or self.report.query_head_assignment != self.query_head_assignment
            or self.report.group_medoids != self.group_medoids
            or self.report.query_head_medoids != self.query_head_medoids
            or not self.report.invariants.all_hold
        ):
            raise ActivationClusterInvariantError(
                "The GQA report does not prove the supplied plan."
            )

    @property
    def source_query_heads_per_group(self) -> int:
        return self.source_query_heads // self.source_key_value_heads

    @property
    def target_query_heads_per_group(self) -> int:
        return self.target_query_heads // self.target_key_value_heads

    @property
    def source_head_dim(self) -> int:
        return self.head_dim

    @property
    def target_head_dim(self) -> int:
        return self.head_dim

    @property
    def source_group_to_target(self) -> tuple[int, ...]:
        return self.group_assignment

    @property
    def source_query_head_to_target(self) -> tuple[int, ...]:
        return self.query_head_assignment

    @property
    def target_group_source_indices(self) -> tuple[int, ...]:
        return self.group_medoids

    @property
    def target_query_source_indices(self) -> tuple[int, ...]:
        return self.query_head_medoids

    def select_query_producer(self, source_tensor: Any, *, axis: int = 0) -> Any:
        """Copy complete query-head blocks from the selected source medoids."""

        return _select_unit_blocks(
            source_tensor,
            source_units=self.source_query_heads,
            unit_width=self.head_dim,
            medoids=self.query_head_medoids,
            axis=axis,
            name="query producer",
        )

    def select_key_producer(self, source_tensor: Any, *, axis: int = 0) -> Any:
        """Copy complete key-head blocks from selected source GQA groups."""

        return _select_unit_blocks(
            source_tensor,
            source_units=self.source_key_value_heads,
            unit_width=self.head_dim,
            medoids=self.group_medoids,
            axis=axis,
            name="key producer",
        )

    def select_value_producer(self, source_tensor: Any, *, axis: int = 0) -> Any:
        """Copy complete value-head blocks from selected source GQA groups."""

        return _select_unit_blocks(
            source_tensor,
            source_units=self.source_key_value_heads,
            unit_width=self.head_dim,
            medoids=self.group_medoids,
            axis=axis,
            name="value producer",
        )

    def aggregate_output_consumer(self, source_tensor: Any, *, axis: int = -1) -> Any:
        """Aggregate source output-consumer head blocks into target head blocks."""

        return _aggregate_unit_blocks(
            source_tensor,
            source_units=self.source_query_heads,
            target_units=self.target_query_heads,
            unit_width=self.head_dim,
            assignment=self.query_head_assignment,
            coefficients=self.context_consumer_coefficients,
            axis=axis,
            name="attention output consumer",
        )

    fold_query_producer = select_query_producer
    fold_key_producer = select_key_producer
    fold_value_producer = select_value_producer
    fold_output_consumer = aggregate_output_consumer


@dataclass(frozen=True, slots=True)
class GatedMLPActivationClusterPlan:
    """A shared gate/up/down neuron plan with real source producer medoids."""

    source_neurons: int
    target_neurons: int
    medoid_source_neurons: tuple[int, ...]
    source_to_target: tuple[int, ...]
    consumer_coefficients: tuple[float, ...]
    report: GatedMLPActivationClusterReport

    def __post_init__(self) -> None:
        source = _positive_int(self.source_neurons, "source_neurons")
        target = _positive_int(self.target_neurons, "target_neurons")
        if target > source:
            raise ActivationClusterInvariantError(
                "An MLP activation-cluster plan cannot expand neurons."
            )
        _validate_real_medoids(
            self.medoid_source_neurons,
            self.source_to_target,
            source,
            target,
            "medoid_source_neurons",
        )
        if len(self.consumer_coefficients) != source or not all(
            math.isfinite(value) for value in self.consumer_coefficients
        ):
            raise ActivationClusterInvariantError(
                "Consumer coefficients must be finite and cover every neuron."
            )
        if not isinstance(self.report, GatedMLPActivationClusterReport):
            raise ActivationClusterInvariantError(
                "report must be a GatedMLPActivationClusterReport."
            )
        if (
            self.report.medoid_source_neurons != self.medoid_source_neurons
            or self.report.source_to_target != self.source_to_target
            or self.report.consumer_coefficients != self.consumer_coefficients
            or not self.report.invariants.all_hold
        ):
            raise ActivationClusterInvariantError(
                "The gated-MLP report does not prove the supplied plan."
            )

    @property
    def assignment(self) -> tuple[int, ...]:
        return self.source_to_target

    @property
    def medoids(self) -> tuple[int, ...]:
        return self.medoid_source_neurons

    def select_gate_producer(self, source_tensor: Any, *, axis: int = 0) -> Any:
        """Copy gate producer rows from real source-neuron medoids."""

        return _select_unit_blocks(
            source_tensor,
            source_units=self.source_neurons,
            unit_width=1,
            medoids=self.medoid_source_neurons,
            axis=axis,
            name="gate producer",
        )

    def select_up_producer(self, source_tensor: Any, *, axis: int = 0) -> Any:
        """Copy up producer rows from the same real source-neuron medoids."""

        return _select_unit_blocks(
            source_tensor,
            source_units=self.source_neurons,
            unit_width=1,
            medoids=self.medoid_source_neurons,
            axis=axis,
            name="up producer",
        )

    def aggregate_down_consumer(self, source_tensor: Any, *, axis: int = -1) -> Any:
        """Aggregate down-consumer columns using post-SwiGLU LS scalars."""

        return _aggregate_unit_blocks(
            source_tensor,
            source_units=self.source_neurons,
            target_units=self.target_neurons,
            unit_width=1,
            assignment=self.source_to_target,
            coefficients=self.consumer_coefficients,
            axis=axis,
            name="down consumer",
        )

    fold_gate_producer = select_gate_producer
    fold_up_producer = select_up_producer
    fold_down_consumer = aggregate_down_consumer


def _real_floating_tensor(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise ActivationClusterShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise ActivationClusterShapeError(
            f"{name} must be a materialized dense strided tensor."
        )
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise ActivationClusterShapeError(
            f"{name} must have a real floating-point dtype."
        )
    if value.ndim <= 0 or any(int(size) <= 0 for size in value.shape):
        raise ActivationClusterShapeError(f"{name} must have nonempty dimensions.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ActivationClusterFiniteError(f"{name} contains NaN or infinity.")
    return result


def _signature_matrix(
    torch: Any,
    value: Any,
    name: str,
    *,
    unit_count: int,
) -> Any:
    result = _real_floating_tensor(torch, value, name)
    if result.ndim < 2 or int(result.shape[0]) != unit_count:
        raise ActivationClusterShapeError(
            f"{name} must be head-major with first dimension {unit_count}."
        )
    return result.reshape(unit_count, -1).contiguous()


def _aligned_mlp_rows(
    torch: Any,
    gate_rows: Any,
    up_rows: Any,
    post_swiglu_rows: Any,
    contribution_rows: Any,
    source_neurons: int | None,
) -> tuple[Any, Any, Any, Any, int, int]:
    gate = _real_floating_tensor(torch, gate_rows, "gate_rows")
    up = _real_floating_tensor(torch, up_rows, "up_rows")
    post = _real_floating_tensor(torch, post_swiglu_rows, "post_swiglu_rows")
    if gate.ndim != 2 or up.ndim != 2 or post.ndim != 2:
        raise ActivationClusterShapeError(
            "gate_rows, up_rows, and post_swiglu_rows must be two-dimensional."
        )
    if gate.shape != up.shape or gate.shape != post.shape:
        raise ActivationClusterShapeError(
            "Gate, up, and post-SwiGLU rows must have identical shapes."
        )
    rows, inferred_neurons = (int(gate.shape[0]), int(gate.shape[1]))
    if source_neurons is not None:
        source_neurons = _positive_int(source_neurons, "source_neurons")
        if source_neurons != inferred_neurons:
            raise ActivationClusterShapeError(
                "source_neurons does not match the aligned MLP rows."
            )
    contribution = _real_floating_tensor(torch, contribution_rows, "contribution_rows")
    if (
        contribution.ndim < 2
        or int(contribution.shape[0]) != rows
        or int(contribution.shape[1]) != inferred_neurons
    ):
        raise ActivationClusterShapeError(
            "contribution_rows must begin with [observation_rows, source_neurons]."
        )
    contribution_signatures = contribution.movedim(1, 0).reshape(inferred_neurons, -1)
    return (
        gate,
        up,
        post,
        contribution_signatures.contiguous(),
        rows,
        inferred_neurons,
    )


def _combine_modalities(torch: Any, matrices: tuple[Any, ...]) -> Any:
    if not matrices:
        raise ActivationClusterShapeError(
            "At least one activation-signature modality is required."
        )
    unit_count = int(matrices[0].shape[0])
    parts = []
    for matrix in matrices:
        if matrix.ndim != 2 or int(matrix.shape[0]) != unit_count:
            raise ActivationClusterShapeError(
                "Activation-signature modalities must have the same unit count."
            )
        width = int(matrix.shape[1])
        rms = torch.sqrt(matrix.square().mean())
        scale = float(rms.item())
        if scale == 0.0:
            scale = 1.0
        parts.append(matrix / (scale * math.sqrt(width)))
    result = torch.cat(parts, dim=1).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ActivationClusterFiniteError(
            "Activation-signature normalization produced non-finite values."
        )
    return result


@dataclass(frozen=True, slots=True)
class _ClusterResult:
    medoids: Any
    assignment: Any
    group_sizes: Any
    strategy: str
    balanced_partition: bool
    locality_dimensions: int
    iterations: int
    converged: bool
    squared_error: float


def _squared_distances(torch: Any, left: Any, right: Any) -> Any:
    distances = (
        left.square().sum(dim=1, keepdim=True)
        + right.square().sum(dim=1).unsqueeze(0)
        - 2.0 * left.matmul(right.T)
    )
    return distances.clamp_min_(0.0)


def _nearest_assignment(
    torch: Any,
    features: Any,
    medoids: Any,
    chunk_size: int,
) -> Any:
    result = torch.empty(int(features.shape[0]), dtype=torch.long)
    selected = features.index_select(0, medoids)
    for start in range(0, int(features.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(features.shape[0]))
        distances = _squared_distances(torch, features[start:stop], selected)
        result[start:stop] = torch.argmin(distances, dim=1)
    result[medoids] = torch.arange(int(medoids.numel()), dtype=torch.long)
    return result


def _cluster_squared_error(
    torch: Any,
    features: Any,
    medoids: Any,
    assignment: Any,
    chunk_size: int,
) -> float:
    selected = features.index_select(0, medoids)
    row_squared_errors = torch.empty(int(features.shape[0]), dtype=torch.float64)
    for start in range(0, int(features.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(features.shape[0]))
        targets = selected.index_select(0, assignment[start:stop])
        residual = features[start:stop] - targets
        row_squared_errors[start:stop] = residual.square().sum(dim=1)
    squared_error = float(row_squared_errors.sum().item())
    if not math.isfinite(squared_error):
        raise ActivationClusterFiniteError(
            "Medoid clustering produced a non-finite objective."
        )
    return squared_error


def _normalized_locality_projection(torch: Any, features: Any) -> Any:
    """Project rows deterministically in O(source_count * feature_dim)."""

    source_count, feature_dim = map(int, features.shape)
    direction_dimensions = min(_LOCALITY_DIRECTION_DIMENSIONS, feature_dim)
    row_norm = torch.linalg.vector_norm(features, dim=1)
    safe_row_norm = row_norm.clamp_min(torch.finfo(torch.float64).tiny)
    projection = torch.empty(
        (source_count, direction_dimensions + 1), dtype=torch.float64
    )
    prime = 2_147_483_647
    for dimension in range(direction_dimensions):
        column_indices = torch.arange(
            dimension, feature_dim, direction_dimensions, dtype=torch.long
        )
        hashes = torch.remainder(
            column_indices * 1_326_451_463 + (dimension + 1) * 374_761_393,
            prime,
        )
        hashes = torch.bitwise_xor(hashes, hashes >> 13)
        signs = torch.where(
            hashes < prime // 2,
            torch.tensor(-1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        direction = features[:, dimension::direction_dimensions].matmul(signs)
        projection[:, dimension] = (
            direction / math.sqrt(int(column_indices.numel())) / safe_row_norm
        )
    projection[:, -1] = torch.log1p(row_norm)

    projection -= projection.mean(dim=0, keepdim=True)
    scale = torch.sqrt(projection.square().mean(dim=0, keepdim=True))
    projection /= torch.where(scale > 0.0, scale, torch.ones_like(scale))
    if not bool(torch.isfinite(projection).all().item()):
        raise ActivationClusterFiniteError(
            "Locality projection produced non-finite values."
        )
    return projection.contiguous()


def _morton_locality_order(torch: Any, projection: Any) -> Any:
    """Return a deterministic locality order from rank-normalized coordinates."""

    source_count, dimensions = map(int, projection.shape)
    bits_per_dimension = max(1, _MORTON_CODE_BITS // dimensions)
    quantization_levels = (1 << bits_per_dimension) - 1
    rank_scale = max(source_count - 1, 1)
    codes = torch.zeros(source_count, dtype=torch.long)
    source_ranks = torch.arange(source_count, dtype=torch.long)
    for dimension in range(dimensions):
        coordinate_order = torch.argsort(projection[:, dimension], stable=True)
        ranks = torch.empty(source_count, dtype=torch.long)
        ranks[coordinate_order] = source_ranks
        quantized = torch.div(
            ranks * quantization_levels,
            rank_scale,
            rounding_mode="floor",
        )
        for bit in range(bits_per_dimension):
            shift = bit * dimensions + dimension
            codes |= ((quantized >> bit) & 1) << shift
    return torch.argsort(codes, stable=True)


def _balanced_locality_medoid_cluster(
    torch: Any,
    features: Any,
    target_count: int,
    *,
    chunk_size: int,
) -> _ClusterResult:
    source_count = int(features.shape[0])
    projection = _normalized_locality_projection(torch, features)
    locality_order = _morton_locality_order(torch, projection)

    base_size, remainder = divmod(source_count, target_count)
    group_sizes = torch.full((target_count,), base_size, dtype=torch.long)
    group_sizes[:remainder] += 1
    ordered_assignment = torch.repeat_interleave(
        torch.arange(target_count, dtype=torch.long), group_sizes
    )
    assignment = torch.empty(source_count, dtype=torch.long)
    assignment[locality_order] = ordered_assignment

    medoids = torch.empty(target_count, dtype=torch.long)
    start = 0
    for target, size_value in enumerate(group_sizes.tolist()):
        stop = start + int(size_value)
        members = torch.sort(locality_order[start:stop]).values
        member_rows = features.index_select(0, members)
        centroid = member_rows.mean(dim=0, keepdim=True)
        # This exactly minimizes pairwise squared distance over real members.
        distances = _squared_distances(torch, member_rows, centroid).squeeze(1)
        medoids[target] = members[int(torch.argmin(distances).item())]
        start = stop

    order = torch.argsort(medoids, stable=True)
    old_to_new = torch.empty(target_count, dtype=torch.long)
    old_to_new[order] = torch.arange(target_count, dtype=torch.long)
    medoids = medoids.index_select(0, order)
    assignment = old_to_new.index_select(0, assignment)
    group_sizes = group_sizes.index_select(0, order)
    assignment[medoids] = torch.arange(target_count, dtype=torch.long)
    squared_error = _cluster_squared_error(
        torch, features, medoids, assignment, chunk_size
    )
    return _ClusterResult(
        medoids=medoids.contiguous(),
        assignment=assignment.contiguous(),
        group_sizes=group_sizes.contiguous(),
        strategy=_SCALABLE_CLUSTER_STRATEGY,
        balanced_partition=True,
        locality_dimensions=int(projection.shape[1]),
        iterations=1,
        converged=True,
        squared_error=squared_error,
    )


def _restricted_real_medoid_cluster(
    torch: Any,
    features: Any,
    target_count: int,
    candidate_indices: Any,
    *,
    max_iterations: int,
    chunk_size: int,
) -> _ClusterResult:
    source_count = int(features.shape[0])
    candidates = candidate_indices.detach().to(device="cpu", dtype=torch.long)
    if candidates.ndim != 1 or int(candidates.numel()) < target_count:
        raise ActivationClusterShapeError(
            "Restricted medoid candidates must contain at least target_count rows."
        )
    if int(torch.unique(candidates).numel()) != int(candidates.numel()):
        raise ActivationClusterShapeError(
            "Restricted medoid candidate indices must be unique."
        )
    if bool((candidates < 0).any()) or bool((candidates >= source_count).any()):
        raise ActivationClusterShapeError(
            "A restricted medoid candidate index is outside the source rows."
        )
    candidate_result = _deterministic_medoid_cluster(
        torch,
        features.index_select(0, candidates),
        target_count,
        max_iterations=max_iterations,
        chunk_size=chunk_size,
    )
    medoids = candidates.index_select(0, candidate_result.medoids)
    assignment = _nearest_assignment(torch, features, medoids, chunk_size)
    assignment[medoids] = torch.arange(target_count, dtype=torch.long)
    group_sizes = torch.bincount(assignment, minlength=target_count)
    if bool((group_sizes <= 0).any()):
        raise ActivationClusterInvariantError(
            "Restricted medoid assignment left a target cluster empty."
        )
    squared_error = _cluster_squared_error(
        torch,
        features,
        medoids,
        assignment,
        chunk_size,
    )
    return _ClusterResult(
        medoids=medoids.contiguous(),
        assignment=assignment.contiguous(),
        group_sizes=group_sizes.contiguous(),
        strategy=f"restricted-candidates:{candidate_result.strategy}",
        balanced_partition=False,
        locality_dimensions=candidate_result.locality_dimensions,
        iterations=candidate_result.iterations,
        converged=candidate_result.converged,
        squared_error=squared_error,
    )


def _deterministic_medoid_cluster(
    torch: Any,
    features: Any,
    target_count: int,
    *,
    max_iterations: int,
    chunk_size: int,
) -> _ClusterResult:
    source_count = int(features.shape[0])
    if target_count > source_count:
        raise ActivationClusterShapeError(
            "target_count cannot exceed the source unit count."
        )
    if target_count == source_count:
        identity = torch.arange(source_count, dtype=torch.long)
        return _ClusterResult(
            medoids=identity,
            assignment=identity.clone(),
            group_sizes=torch.ones(source_count, dtype=torch.long),
            strategy=_IDENTITY_CLUSTER_STRATEGY,
            balanced_partition=True,
            locality_dimensions=0,
            iterations=0,
            converged=True,
            squared_error=0.0,
        )

    use_global_refinement = (
        source_count <= _GLOBAL_CLUSTER_MAX_SOURCES
        and source_count * target_count <= _GLOBAL_CLUSTER_MAX_SOURCE_TARGET_PAIRS
    )
    if not use_global_refinement:
        return _balanced_locality_medoid_cluster(
            torch,
            features,
            target_count,
            chunk_size=chunk_size,
        )

    centroid = features.mean(dim=0, keepdim=True)
    first_distances = _squared_distances(torch, features, centroid).squeeze(1)
    selected_indices = [int(torch.argmax(first_distances).item())]
    selected_mask = torch.zeros(source_count, dtype=torch.bool)
    selected_mask[selected_indices[0]] = True
    minimum_distance = _squared_distances(
        torch, features, features[selected_indices]
    ).squeeze(1)
    for _ in range(1, target_count):
        scores = torch.where(
            selected_mask,
            torch.full_like(minimum_distance, -1.0),
            minimum_distance,
        )
        selected_index = int(torch.argmax(scores).item())
        selected_indices.append(selected_index)
        selected_mask[selected_index] = True
        distance = _squared_distances(
            torch,
            features,
            features[selected_index : selected_index + 1],
        ).squeeze(1)
        minimum_distance = torch.minimum(minimum_distance, distance)

    medoids = torch.tensor(selected_indices, dtype=torch.long)
    seen: set[tuple[int, ...]] = set()
    converged = False
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        state = tuple(int(value) for value in medoids.tolist())
        seen.add(state)
        assignment = _nearest_assignment(torch, features, medoids, chunk_size)
        updated = torch.empty_like(medoids)
        for target in range(target_count):
            members = torch.nonzero(assignment == target, as_tuple=False).flatten()
            if int(members.numel()) == 0:  # protected medoids make this unreachable
                raise ActivationClusterInvariantError(
                    "Medoid refinement produced an empty target cluster."
                )
            member_rows = features.index_select(0, members)
            member_centroid = member_rows.mean(dim=0, keepdim=True)
            distances = _squared_distances(torch, member_rows, member_centroid).squeeze(
                1
            )
            updated[target] = members[int(torch.argmin(distances).item())]
        updated_state = tuple(int(value) for value in updated.tolist())
        if updated_state == state:
            converged = True
            break
        if updated_state in seen:
            medoids = torch.tensor(min(state, updated_state), dtype=torch.long)
            break
        medoids = updated

    assignment = _nearest_assignment(torch, features, medoids, chunk_size)
    order = torch.argsort(medoids, stable=True)
    old_to_new = torch.empty(target_count, dtype=torch.long)
    old_to_new[order] = torch.arange(target_count, dtype=torch.long)
    medoids = medoids.index_select(0, order)
    assignment = old_to_new.index_select(0, assignment)
    assignment[medoids] = torch.arange(target_count, dtype=torch.long)
    group_sizes = torch.bincount(assignment, minlength=target_count)
    if bool((group_sizes <= 0).any().item()):
        raise ActivationClusterInvariantError(
            "Deterministic clustering left a target medoid empty."
        )
    squared_error = _cluster_squared_error(
        torch, features, medoids, assignment, chunk_size
    )
    return _ClusterResult(
        medoids=medoids.contiguous(),
        assignment=assignment.contiguous(),
        group_sizes=group_sizes.contiguous(),
        strategy=_GLOBAL_CLUSTER_STRATEGY,
        balanced_partition=False,
        locality_dimensions=0,
        iterations=iterations,
        converged=converged,
        squared_error=squared_error,
    )


@dataclass(frozen=True, slots=True)
class _CoefficientFit:
    coefficients: Any
    relative_error: float
    effective_ridge: float
    near_zero_medoid_count: int
    maximum_absolute_coefficient: float
    negative_coefficient_count: int


def _least_squares_consumer_coefficients(
    torch: Any,
    signal_signatures: Any,
    medoids: Any,
    assignment: Any,
    *,
    ridge: float,
) -> _CoefficientFit:
    selected = signal_signatures.index_select(0, medoids)
    basis = selected.index_select(0, assignment)
    selected_energy = selected.square().sum(dim=1)
    basis_energy = selected_energy.index_select(0, assignment)
    trace_scale = float(selected_energy.mean().item())
    tiny = torch.finfo(torch.float64).tiny
    effective_ridge = ridge * max(trace_scale, tiny)
    numerical_floor = max(
        trace_scale
        * torch.finfo(torch.float64).eps
        * max(32, int(signal_signatures.shape[1])),
        tiny,
    )
    dot = (signal_signatures * basis).sum(dim=1)
    stable = basis_energy > numerical_floor
    denominator = basis_energy + effective_ridge
    denominator = torch.where(
        stable,
        denominator,
        denominator.clamp_min(numerical_floor),
    )
    coefficients = dot / denominator

    exact_match = (signal_signatures == basis).all(dim=1)
    coefficients[exact_match] = 1.0
    coefficients[medoids] = 1.0
    if not bool(torch.isfinite(coefficients).all().item()):
        raise ActivationClusterFiniteError(
            "Least-squares consumer coefficients are non-finite."
        )
    reconstruction = coefficients.unsqueeze(1) * basis
    residual_norm = float(
        torch.linalg.vector_norm(signal_signatures - reconstruction).item()
    )
    signal_norm = float(torch.linalg.vector_norm(signal_signatures).item())
    relative_error = residual_norm / max(signal_norm, torch.finfo(torch.float64).eps)
    maximum = float(coefficients.abs().max().item())
    if not math.isfinite(relative_error) or not math.isfinite(maximum):
        raise ActivationClusterFiniteError(
            "Least-squares coefficient diagnostics are non-finite."
        )
    return _CoefficientFit(
        coefficients=coefficients.contiguous(),
        relative_error=relative_error,
        effective_ridge=effective_ridge,
        near_zero_medoid_count=int((selected_energy <= numerical_floor).sum().item()),
        maximum_absolute_coefficient=maximum,
        negative_coefficient_count=int((coefficients < 0.0).sum().item()),
    )


def _resolve_preserved_head_dim(
    *,
    head_dim: int | None,
    source_head_dim: int | None,
    target_head_dim: int | None,
) -> int:
    supplied = [
        value
        for value in (head_dim, source_head_dim, target_head_dim)
        if value is not None
    ]
    if not supplied:
        raise ActivationClusterShapeError(
            "head_dim or preserved source/target head dimensions are required."
        )
    normalized = tuple(_positive_int(value, "head_dim") for value in supplied)
    if len(set(normalized)) != 1:
        raise ActivationClusterShapeError(
            "Source and target RoPE head dimensions must be identical."
        )
    result = normalized[0]
    if result % 2:
        raise ActivationClusterShapeError(
            "RoPE-preserving activation clusters require an even head dimension."
        )
    return result


def _leading_unit_count(torch: Any, value: Any, name: str) -> int:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ActivationClusterShapeError(
            f"{name} must be a head-major tensor with at least two dimensions."
        )
    return _positive_int(int(value.shape[0]), f"{name} first dimension")


def build_gqa_activation_cluster_plan(
    query_signatures: Any,
    key_signatures: Any,
    value_signatures: Any,
    context_signatures: Any,
    contribution_signatures: Any,
    *,
    target_query_heads: int,
    target_key_value_heads: int,
    head_dim: int | None = None,
    source_query_heads: int | None = None,
    source_key_value_heads: int | None = None,
    source_head_dim: int | None = None,
    target_head_dim: int | None = None,
    group_medoid_candidates: Sequence[int] | None = None,
    ridge: float = 1e-8,
    max_iterations: int = 16,
    distance_chunk_size: int = 512,
) -> GQAActivationClusterPlan:
    """Build a hierarchical real-medoid plan for split-QKV GQA attention.

    Each signature is head-major: its first dimension is the query-head count
    for query/context/contribution, or the KV-head count for key/value.  Any
    remaining dimensions are flattened into that head's activation signature.
    Source GQA groups and source query offsets are clustered independently;
    their Cartesian product assigns every source query head exactly once while
    preserving group-major offsets.  ``source_head_dim`` and
    ``target_head_dim``, when supplied, must be equal.
    """

    torch = _require_torch()
    target_query_heads = _positive_int(target_query_heads, "target_query_heads")
    target_key_value_heads = _positive_int(
        target_key_value_heads, "target_key_value_heads"
    )
    max_iterations = _positive_int(max_iterations, "max_iterations")
    distance_chunk_size = _positive_int(distance_chunk_size, "distance_chunk_size")
    ridge = _nonnegative_float(ridge, "ridge")
    preserved_head_dim = _resolve_preserved_head_dim(
        head_dim=head_dim,
        source_head_dim=source_head_dim,
        target_head_dim=target_head_dim,
    )

    inferred_query_heads = _leading_unit_count(
        torch, query_signatures, "query_signatures"
    )
    inferred_key_value_heads = _leading_unit_count(
        torch, key_signatures, "key_signatures"
    )
    if source_query_heads is None:
        source_query_heads = inferred_query_heads
    else:
        source_query_heads = _positive_int(source_query_heads, "source_query_heads")
        if source_query_heads != inferred_query_heads:
            raise ActivationClusterShapeError(
                "source_query_heads does not match query_signatures."
            )
    if source_key_value_heads is None:
        source_key_value_heads = inferred_key_value_heads
    else:
        source_key_value_heads = _positive_int(
            source_key_value_heads, "source_key_value_heads"
        )
        if source_key_value_heads != inferred_key_value_heads:
            raise ActivationClusterShapeError(
                "source_key_value_heads does not match key_signatures."
            )
    if source_query_heads % source_key_value_heads:
        raise ActivationClusterShapeError(
            "Source query heads must divide exactly into source GQA groups."
        )
    if target_query_heads % target_key_value_heads:
        raise ActivationClusterShapeError(
            "Target query heads must divide exactly into target GQA groups."
        )
    if target_key_value_heads > source_key_value_heads:
        raise ActivationClusterShapeError(
            "Target GQA group count cannot exceed the source group count."
        )
    source_offsets = source_query_heads // source_key_value_heads
    target_offsets = target_query_heads // target_key_value_heads
    if target_offsets > source_offsets:
        raise ActivationClusterShapeError(
            "Target query-head offsets per GQA group cannot exceed the source."
        )

    with torch.no_grad():
        query = _signature_matrix(
            torch,
            query_signatures,
            "query_signatures",
            unit_count=source_query_heads,
        )
        key = _signature_matrix(
            torch,
            key_signatures,
            "key_signatures",
            unit_count=source_key_value_heads,
        )
        value = _signature_matrix(
            torch,
            value_signatures,
            "value_signatures",
            unit_count=source_key_value_heads,
        )
        context = _signature_matrix(
            torch,
            context_signatures,
            "context_signatures",
            unit_count=source_query_heads,
        )
        contribution = _signature_matrix(
            torch,
            contribution_signatures,
            "contribution_signatures",
            unit_count=source_query_heads,
        )

        grouped_query = query.reshape(source_key_value_heads, source_offsets, -1)
        grouped_context = context.reshape(source_key_value_heads, source_offsets, -1)
        grouped_contribution = contribution.reshape(
            source_key_value_heads, source_offsets, -1
        )
        group_features = _combine_modalities(
            torch,
            (
                grouped_query.reshape(source_key_value_heads, -1),
                key,
                value,
                grouped_context.reshape(source_key_value_heads, -1),
                grouped_contribution.reshape(source_key_value_heads, -1),
            ),
        )
        offset_features = _combine_modalities(
            torch,
            (
                grouped_query.permute(1, 0, 2).reshape(source_offsets, -1),
                grouped_context.permute(1, 0, 2).reshape(source_offsets, -1),
                grouped_contribution.permute(1, 0, 2).reshape(source_offsets, -1),
            ),
        )
        if group_medoid_candidates is None:
            group_candidate_indices = torch.arange(
                source_key_value_heads,
                dtype=torch.long,
            )
            group_clusters = _deterministic_medoid_cluster(
                torch,
                group_features,
                target_key_value_heads,
                max_iterations=max_iterations,
                chunk_size=distance_chunk_size,
            )
        else:
            try:
                group_candidate_indices = torch.tensor(
                    tuple(group_medoid_candidates),
                    dtype=torch.long,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ActivationClusterShapeError(
                    "group_medoid_candidates must be integer source-group indices."
                ) from exc
            group_clusters = _restricted_real_medoid_cluster(
                torch,
                group_features,
                target_key_value_heads,
                group_candidate_indices,
                max_iterations=max_iterations,
                chunk_size=distance_chunk_size,
            )
        offset_clusters = _deterministic_medoid_cluster(
            torch,
            offset_features,
            target_offsets,
            max_iterations=max_iterations,
            chunk_size=distance_chunk_size,
        )
        query_assignment = (
            group_clusters.assignment[:, None] * target_offsets
            + offset_clusters.assignment[None, :]
        ).reshape(-1)
        query_medoids = (
            group_clusters.medoids[:, None] * source_offsets
            + offset_clusters.medoids[None, :]
        ).reshape(-1)
        coefficient_fit = _least_squares_consumer_coefficients(
            torch,
            context,
            query_medoids,
            query_assignment,
            ridge=ridge,
        )
        query_group_sizes = torch.bincount(
            query_assignment, minlength=target_query_heads
        )

    group_medoids_tuple = tuple(int(value) for value in group_clusters.medoids)
    offset_medoids_tuple = tuple(int(value) for value in offset_clusters.medoids)
    query_medoids_tuple = tuple(int(value) for value in query_medoids)
    group_assignment_tuple = tuple(int(value) for value in group_clusters.assignment)
    offset_assignment_tuple = tuple(int(value) for value in offset_clusters.assignment)
    query_assignment_tuple = tuple(int(value) for value in query_assignment)
    coefficient_tuple = tuple(float(value) for value in coefficient_fit.coefficients)
    group_sizes = tuple(int(value) for value in group_clusters.group_sizes)
    offset_group_sizes = tuple(int(value) for value in offset_clusters.group_sizes)
    query_group_sizes_tuple = tuple(int(value) for value in query_group_sizes)
    medoids_self_assigned = all(
        group_assignment_tuple[source] == target
        for target, source in enumerate(group_medoids_tuple)
    ) and all(
        query_assignment_tuple[source] == target
        for target, source in enumerate(query_medoids_tuple)
    )
    invariants = GQAActivationClusterInvariants(
        source_groups_assigned_exactly_once=(
            len(group_assignment_tuple) == source_key_value_heads
        ),
        source_query_heads_assigned_exactly_once=(
            len(query_assignment_tuple) == source_query_heads
        ),
        nonempty_target_groups=all(size > 0 for size in group_sizes),
        nonempty_target_query_heads=all(size > 0 for size in query_group_sizes_tuple),
        real_source_group_medoids=(
            len(set(group_medoids_tuple)) == target_key_value_heads
            and all(
                0 <= value < source_key_value_heads for value in group_medoids_tuple
            )
        ),
        real_source_query_head_medoids=(
            len(set(query_medoids_tuple)) == target_query_heads
            and all(0 <= value < source_query_heads for value in query_medoids_tuple)
        ),
        medoids_self_assigned=medoids_self_assigned,
        gqa_group_offsets_preserved=True,
        rope_head_dimension_preserved=True,
        finite_consumer_coefficients=all(
            math.isfinite(value) for value in coefficient_tuple
        ),
        no_producer_averaging=True,
        zero_gradient=True,
    )
    if not invariants.all_hold:
        raise ActivationClusterInvariantError(
            "The computed GQA activation-cluster invariants do not all hold."
        )
    report = GQAActivationClusterReport(
        source_query_heads=source_query_heads,
        source_key_value_heads=source_key_value_heads,
        target_query_heads=target_query_heads,
        target_key_value_heads=target_key_value_heads,
        source_query_heads_per_group=source_offsets,
        target_query_heads_per_group=target_offsets,
        head_dim=preserved_head_dim,
        group_medoids=group_medoids_tuple,
        query_offset_medoids=offset_medoids_tuple,
        query_head_medoids=query_medoids_tuple,
        group_assignment=group_assignment_tuple,
        query_offset_assignment=offset_assignment_tuple,
        query_head_assignment=query_assignment_tuple,
        group_sizes=group_sizes,
        query_offset_group_sizes=offset_group_sizes,
        query_head_group_sizes=query_group_sizes_tuple,
        context_consumer_coefficients=coefficient_tuple,
        group_medoid_candidate_indices=tuple(
            int(value) for value in group_candidate_indices
        ),
        group_cluster_strategy=group_clusters.strategy,
        query_offset_cluster_strategy=offset_clusters.strategy,
        group_cluster_balanced_partition=group_clusters.balanced_partition,
        query_offset_cluster_balanced_partition=(offset_clusters.balanced_partition),
        group_cluster_locality_dimensions=group_clusters.locality_dimensions,
        query_offset_cluster_locality_dimensions=(offset_clusters.locality_dimensions),
        group_cluster_iterations=group_clusters.iterations,
        query_offset_cluster_iterations=offset_clusters.iterations,
        group_cluster_converged=group_clusters.converged,
        query_offset_cluster_converged=offset_clusters.converged,
        group_cluster_squared_error=group_clusters.squared_error,
        query_offset_cluster_squared_error=offset_clusters.squared_error,
        context_fit_relative_error=coefficient_fit.relative_error,
        context_fit_effective_ridge=coefficient_fit.effective_ridge,
        near_zero_context_medoid_count=(coefficient_fit.near_zero_medoid_count),
        invariants=invariants,
    )
    return GQAActivationClusterPlan(
        source_query_heads=source_query_heads,
        source_key_value_heads=source_key_value_heads,
        target_query_heads=target_query_heads,
        target_key_value_heads=target_key_value_heads,
        head_dim=preserved_head_dim,
        group_medoids=group_medoids_tuple,
        query_offset_medoids=offset_medoids_tuple,
        query_head_medoids=query_medoids_tuple,
        group_assignment=group_assignment_tuple,
        query_offset_assignment=offset_assignment_tuple,
        query_head_assignment=query_assignment_tuple,
        context_consumer_coefficients=coefficient_tuple,
        report=report,
    )


def build_gated_mlp_activation_cluster_plan(
    gate_rows: Any,
    up_rows: Any,
    post_swiglu_rows: Any,
    contribution_rows: Any,
    target_neurons: int,
    *,
    source_neurons: int | None = None,
    ridge: float = 1e-8,
    max_iterations: int = 16,
    distance_chunk_size: int = 512,
) -> GatedMLPActivationClusterPlan:
    """Build one real-medoid assignment shared by gate, up, and down.

    Gate, up, and post-SwiGLU tensors have shape ``[rows, source_neurons]``.
    Contribution rows have shape ``[rows, source_neurons, ...]`` (a rank-two
    scalar contribution matrix is also valid).  Clustering uses all four
    modalities.  Consumer coefficients solve each source post-SwiGLU trace
    against its selected medoid trace and never divide by gate values.
    """

    torch = _require_torch()
    target_neurons = _positive_int(target_neurons, "target_neurons")
    max_iterations = _positive_int(max_iterations, "max_iterations")
    distance_chunk_size = _positive_int(distance_chunk_size, "distance_chunk_size")
    ridge = _nonnegative_float(ridge, "ridge")
    with torch.no_grad():
        (
            gate,
            up,
            post,
            contribution_signatures,
            observation_rows,
            inferred_neurons,
        ) = _aligned_mlp_rows(
            torch,
            gate_rows,
            up_rows,
            post_swiglu_rows,
            contribution_rows,
            source_neurons,
        )
        if target_neurons > inferred_neurons:
            raise ActivationClusterShapeError(
                "target_neurons cannot exceed source_neurons."
            )
        gate_signatures = gate.T.contiguous()
        up_signatures = up.T.contiguous()
        post_signatures = post.T.contiguous()
        features = _combine_modalities(
            torch,
            (
                gate_signatures,
                up_signatures,
                post_signatures,
                contribution_signatures,
            ),
        )
        clusters = _deterministic_medoid_cluster(
            torch,
            features,
            target_neurons,
            max_iterations=max_iterations,
            chunk_size=distance_chunk_size,
        )
        coefficient_fit = _least_squares_consumer_coefficients(
            torch,
            post_signatures,
            clusters.medoids,
            clusters.assignment,
            ridge=ridge,
        )

    medoid_tuple = tuple(int(value) for value in clusters.medoids)
    assignment_tuple = tuple(int(value) for value in clusters.assignment)
    group_sizes = tuple(int(value) for value in clusters.group_sizes)
    coefficient_tuple = tuple(float(value) for value in coefficient_fit.coefficients)
    medoids_self_assigned = all(
        assignment_tuple[source] == target for target, source in enumerate(medoid_tuple)
    )
    invariants = GatedMLPActivationClusterInvariants(
        source_neurons_assigned_exactly_once=(
            len(assignment_tuple) == inferred_neurons
        ),
        nonempty_target_neurons=all(size > 0 for size in group_sizes),
        real_source_neuron_medoids=(
            len(set(medoid_tuple)) == target_neurons
            and all(0 <= value < inferred_neurons for value in medoid_tuple)
        ),
        medoids_self_assigned=medoids_self_assigned,
        shared_gate_up_down_assignment=True,
        finite_consumer_coefficients=all(
            math.isfinite(value) for value in coefficient_tuple
        ),
        coefficients_fit_from_post_swiglu=True,
        no_gate_value_division=True,
        no_producer_averaging=True,
        zero_gradient=True,
    )
    if not invariants.all_hold:
        raise ActivationClusterInvariantError(
            "The computed gated-MLP activation-cluster invariants do not all hold."
        )
    report = GatedMLPActivationClusterReport(
        source_neurons=inferred_neurons,
        target_neurons=target_neurons,
        observation_rows=observation_rows,
        medoid_source_neurons=medoid_tuple,
        source_to_target=assignment_tuple,
        group_sizes=group_sizes,
        consumer_coefficients=coefficient_tuple,
        cluster_strategy=clusters.strategy,
        cluster_balanced_partition=clusters.balanced_partition,
        cluster_locality_dimensions=clusters.locality_dimensions,
        cluster_iterations=clusters.iterations,
        cluster_converged=clusters.converged,
        cluster_squared_error=clusters.squared_error,
        post_swiglu_fit_relative_error=coefficient_fit.relative_error,
        post_swiglu_fit_effective_ridge=coefficient_fit.effective_ridge,
        near_zero_post_swiglu_medoid_count=(coefficient_fit.near_zero_medoid_count),
        maximum_absolute_consumer_coefficient=(
            coefficient_fit.maximum_absolute_coefficient
        ),
        negative_consumer_coefficient_count=(
            coefficient_fit.negative_coefficient_count
        ),
        invariants=invariants,
    )
    return GatedMLPActivationClusterPlan(
        source_neurons=inferred_neurons,
        target_neurons=target_neurons,
        medoid_source_neurons=medoid_tuple,
        source_to_target=assignment_tuple,
        consumer_coefficients=coefficient_tuple,
        report=report,
    )


def _validated_axis(ndim: int, axis: Any, name: str) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise ActivationClusterShapeError(f"{name} axis must be an integer.")
    normalized = axis + ndim if axis < 0 else axis
    if not 0 <= normalized < ndim:
        raise ActivationClusterShapeError(
            f"{name} axis {axis} is outside a rank-{ndim} tensor."
        )
    return normalized


def _runtime_tensor(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise ActivationClusterShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise ActivationClusterShapeError(
            f"{name} must be a materialized dense strided tensor."
        )
    if value.ndim <= 0 or not torch.is_floating_point(value) or torch.is_complex(value):
        raise ActivationClusterShapeError(
            f"{name} must be a non-scalar real floating-point tensor."
        )
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise ActivationClusterFiniteError(f"{name} contains NaN or infinity.")
    return value.detach()


def _select_unit_blocks(
    source_tensor: Any,
    *,
    source_units: int,
    unit_width: int,
    medoids: tuple[int, ...],
    axis: int,
    name: str,
) -> Any:
    torch = _require_torch()
    with torch.no_grad():
        source = _runtime_tensor(torch, source_tensor, name)
        normalized_axis = _validated_axis(source.ndim, axis, name)
        expected = source_units * unit_width
        if int(source.shape[normalized_axis]) != expected:
            raise ActivationClusterShapeError(
                f"{name} axis must have width {expected}."
            )
        moved = source.movedim(normalized_axis, 0)
        blocks = moved.reshape(source_units, unit_width, *moved.shape[1:])
        indices = torch.tensor(medoids, dtype=torch.long, device=source.device)
        selected = blocks.index_select(0, indices)
        flattened = selected.reshape(len(medoids) * unit_width, *moved.shape[1:])
        result = flattened.movedim(0, normalized_axis).contiguous().clone()
        if not bool(torch.isfinite(result).all().item()):
            raise ActivationClusterFiniteError(
                f"{name} selection produced NaN or infinity."
            )
        return result


def _aggregate_unit_blocks(
    source_tensor: Any,
    *,
    source_units: int,
    target_units: int,
    unit_width: int,
    assignment: tuple[int, ...],
    coefficients: tuple[float, ...],
    axis: int,
    name: str,
) -> Any:
    torch = _require_torch()
    with torch.no_grad():
        source = _runtime_tensor(torch, source_tensor, name)
        normalized_axis = _validated_axis(source.ndim, axis, name)
        expected = source_units * unit_width
        if int(source.shape[normalized_axis]) != expected:
            raise ActivationClusterShapeError(
                f"{name} axis must have width {expected}."
            )
        moved = source.movedim(normalized_axis, 0)
        blocks = moved.reshape(source_units, unit_width, *moved.shape[1:])
        mapping = torch.tensor(assignment, dtype=torch.long, device=source.device)
        values = torch.tensor(coefficients, dtype=source.dtype, device=source.device)
        coefficient_shape = (source_units,) + (1,) * (blocks.ndim - 1)
        weighted = blocks * values.reshape(coefficient_shape)
        aggregate = source.new_zeros((target_units, unit_width, *moved.shape[1:]))
        aggregate.index_add_(0, mapping, weighted)
        flattened = aggregate.reshape(target_units * unit_width, *moved.shape[1:])
        result = flattened.movedim(0, normalized_axis).contiguous()
        if not bool(torch.isfinite(result).all().item()):
            raise ActivationClusterFiniteError(
                f"{name} aggregation produced NaN or infinity."
            )
        return result


plan_gqa_activation_clusters = build_gqa_activation_cluster_plan
build_gqa_medoid_plan = build_gqa_activation_cluster_plan
plan_gated_mlp_activation_clusters = build_gated_mlp_activation_cluster_plan
build_coupled_gated_mlp_activation_cluster_plan = (
    build_gated_mlp_activation_cluster_plan
)


__all__ = [
    "ActivationClusterError",
    "ActivationClusterFiniteError",
    "ActivationClusterInvariantError",
    "ActivationClusterShapeError",
    "GQAActivationClusterInvariants",
    "GQAActivationClusterPlan",
    "GQAActivationClusterReport",
    "GatedMLPActivationClusterInvariants",
    "GatedMLPActivationClusterPlan",
    "GatedMLPActivationClusterReport",
    "build_coupled_gated_mlp_activation_cluster_plan",
    "build_gated_mlp_activation_cluster_plan",
    "build_gqa_activation_cluster_plan",
    "build_gqa_medoid_plan",
    "plan_gated_mlp_activation_clusters",
    "plan_gqa_activation_clusters",
]
