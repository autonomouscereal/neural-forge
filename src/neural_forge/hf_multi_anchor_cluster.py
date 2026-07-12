"""Sparse multi-anchor real-medoid folding for gated MLP circuits.

This module extends the one-anchor plan in :mod:`hf_activation_cluster` while
preserving its most important structural rule: gate and up producers are real
source neurons and are copied, never averaged.  Each source post-SwiGLU path is
instead represented by a small ridge fit over target-medoid activations.  The
resulting consumer transport is stored in CSR-like tuples and applied with
``index_add_``; a dense source-by-target coefficient matrix is never built.

Planning is deterministic, CPU/float64, and zero-gradient.  Explicit limits
bound similarity work, projection solves, sparse-plan size, and peak fitting
workspace before the first fit is attempted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .hf_activation_cluster import (
    ActivationClusterError,
    ActivationClusterFiniteError,
    ActivationClusterInvariantError,
    ActivationClusterShapeError,
    GatedMLPActivationClusterReport,
    build_gated_mlp_activation_cluster_plan,
)


_HARD_MAX_ANCHORS_PER_SOURCE = 32
_HARD_MAX_CANDIDATE_TARGETS = 64
_CANDIDATE_POOL_STRATEGY = (
    "primary-plus-deterministic-local-and-coprime-scattered-targets"
)


class MultiAnchorClusterError(ActivationClusterError):
    """Base class for sparse multi-anchor planning failures."""


class MultiAnchorClusterShapeError(
    MultiAnchorClusterError, ActivationClusterShapeError
):
    """Raised when shapes, limits, or topology are not trustworthy."""


class MultiAnchorClusterFiniteError(
    MultiAnchorClusterError, ActivationClusterFiniteError
):
    """Raised when an input, coefficient, or diagnostic is non-finite."""


class MultiAnchorClusterInvariantError(
    MultiAnchorClusterError, ActivationClusterInvariantError
):
    """Raised when a sparse plan cannot prove its invariants."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for sparse multi-anchor planning."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MultiAnchorClusterShapeError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise MultiAnchorClusterShapeError(
            f"{name} must be a finite positive number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MultiAnchorClusterShapeError(
            f"{name} must be a finite positive number."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise MultiAnchorClusterShapeError(
            f"{name} must be a finite positive number."
        )
    return result


def _optional_provenance(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MultiAnchorClusterShapeError(
            f"{name} must be a nonempty string when supplied."
        )
    return value.strip()


def _planning_tensor(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise MultiAnchorClusterShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise MultiAnchorClusterShapeError(
            f"{name} must be a materialized dense strided tensor."
        )
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise MultiAnchorClusterShapeError(
            f"{name} must have a real floating-point dtype."
        )
    if value.ndim <= 0 or any(int(size) <= 0 for size in value.shape):
        raise MultiAnchorClusterShapeError(f"{name} must have nonempty dimensions.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise MultiAnchorClusterFiniteError(f"{name} contains NaN or infinity.")
    return result


def _aligned_activation_rows(
    torch: Any,
    gate_rows: Any,
    up_rows: Any,
    post_swiglu_rows: Any,
    source_neurons: int | None,
) -> tuple[Any, Any, Any, int, int]:
    gate = _planning_tensor(torch, gate_rows, "gate_rows")
    up = _planning_tensor(torch, up_rows, "up_rows")
    post = _planning_tensor(torch, post_swiglu_rows, "post_swiglu_rows")
    if gate.ndim != 2 or up.ndim != 2 or post.ndim != 2:
        raise MultiAnchorClusterShapeError(
            "gate_rows, up_rows, and post_swiglu_rows must be two-dimensional."
        )
    if gate.shape != up.shape or gate.shape != post.shape:
        raise MultiAnchorClusterShapeError(
            "Gate, up, and post-SwiGLU rows must have identical shapes."
        )
    rows, inferred_source = map(int, gate.shape)
    if source_neurons is not None:
        expected_source = _positive_int(source_neurons, "source_neurons")
        if expected_source != inferred_source:
            raise MultiAnchorClusterShapeError(
                "source_neurons does not match the aligned MLP rows."
            )
    return gate, up, post, rows, inferred_source


def _normalized_axis(ndim: int, axis: Any, name: str) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise MultiAnchorClusterShapeError(f"{name} axis must be an integer.")
    normalized = axis + ndim if axis < 0 else axis
    if not 0 <= normalized < ndim:
        raise MultiAnchorClusterShapeError(
            f"{name} axis {axis} is outside a rank-{ndim} tensor."
        )
    return normalized


def _consumer_source_major(
    torch: Any,
    value: Any,
    *,
    source_neurons: int,
    axis: int,
    name: str,
) -> Any:
    consumer = _planning_tensor(torch, value, name)
    normalized = _normalized_axis(consumer.ndim, axis, name)
    if int(consumer.shape[normalized]) != source_neurons:
        raise MultiAnchorClusterShapeError(
            f"{name} source axis must have width {source_neurons}."
        )
    return consumer.movedim(normalized, 0).reshape(source_neurons, -1).contiguous()


def _runtime_tensor(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise MultiAnchorClusterShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise MultiAnchorClusterShapeError(
            f"{name} must be a materialized dense strided tensor."
        )
    if value.ndim <= 0 or not torch.is_floating_point(value) or torch.is_complex(value):
        raise MultiAnchorClusterShapeError(
            f"{name} must be a non-scalar real floating-point tensor."
        )
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise MultiAnchorClusterFiniteError(f"{name} contains NaN or infinity.")
    return detached


@dataclass(frozen=True, slots=True)
class MultiAnchorReconstructionDiagnostics:
    """Calibration or held-out reconstruction measurements."""

    observation_rows: int
    post_swiglu_relative_l2: float
    post_swiglu_root_mean_square_error: float
    mean_source_relative_l2: float
    maximum_source_relative_l2: float
    zero_signal_source_count: int
    consumer_output_relative_l2: float | None
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_rows": self.observation_rows,
            "post_swiglu_relative_l2": self.post_swiglu_relative_l2,
            "post_swiglu_root_mean_square_error": (
                self.post_swiglu_root_mean_square_error
            ),
            "mean_source_relative_l2": self.mean_source_relative_l2,
            "maximum_source_relative_l2": self.maximum_source_relative_l2,
            "zero_signal_source_count": self.zero_signal_source_count,
            "consumer_output_relative_l2": self.consumer_output_relative_l2,
            "finite": self.finite,
        }


@dataclass(frozen=True, slots=True)
class GatedMLPMultiAnchorInvariants:
    """Proof obligations for a sparse multi-anchor gated-MLP plan."""

    all_source_paths_covered: bool
    nonempty_source_anchor_sets: bool
    anchor_count_bounded: bool
    candidate_pool_bounded: bool
    primary_anchor_included: bool
    real_source_neuron_medoids: bool
    medoids_self_map_exactly: bool
    shared_gate_up_producers: bool
    finite_consumer_coefficients: bool
    finite_reconstruction_diagnostics: bool
    no_producer_averaging: bool
    sparse_consumer_index_add: bool
    no_dense_source_target_matrix: bool
    hard_work_bounds_respected: bool
    hard_memory_bounds_respected: bool
    zero_gradient: bool
    deterministic: bool

    @property
    def all_hold(self) -> bool:
        return all(self.as_dict(include_all=False).values())

    def as_dict(self, *, include_all: bool = True) -> dict[str, bool]:
        result = {
            "all_source_paths_covered": self.all_source_paths_covered,
            "nonempty_source_anchor_sets": self.nonempty_source_anchor_sets,
            "anchor_count_bounded": self.anchor_count_bounded,
            "candidate_pool_bounded": self.candidate_pool_bounded,
            "primary_anchor_included": self.primary_anchor_included,
            "real_source_neuron_medoids": self.real_source_neuron_medoids,
            "medoids_self_map_exactly": self.medoids_self_map_exactly,
            "shared_gate_up_producers": self.shared_gate_up_producers,
            "finite_consumer_coefficients": self.finite_consumer_coefficients,
            "finite_reconstruction_diagnostics": (
                self.finite_reconstruction_diagnostics
            ),
            "no_producer_averaging": self.no_producer_averaging,
            "sparse_consumer_index_add": self.sparse_consumer_index_add,
            "no_dense_source_target_matrix": self.no_dense_source_target_matrix,
            "hard_work_bounds_respected": self.hard_work_bounds_respected,
            "hard_memory_bounds_respected": self.hard_memory_bounds_respected,
            "zero_gradient": self.zero_gradient,
            "deterministic": self.deterministic,
        }
        if include_all:
            result["all_hold"] = all(result.values())
        return result


@dataclass(frozen=True, slots=True)
class GatedMLPMultiAnchorReport:
    """Immutable fit, work-bound, and structural evidence."""

    source_neurons: int
    target_neurons: int
    observation_rows: int
    anchors_per_source: int
    max_candidate_targets: int
    candidate_pool_size: int
    candidate_pool_strategy: str
    medoid_source_neurons: tuple[int, ...]
    source_to_target: tuple[int, ...]
    source_anchor_offsets: tuple[int, ...]
    anchor_target_indices: tuple[int, ...]
    anchor_coefficients: tuple[float, ...]
    anchor_counts: tuple[int, ...]
    coefficient_l1_per_source: tuple[float, ...]
    total_sparse_entries: int
    total_coefficient_l1: float
    mean_coefficient_l1: float
    maximum_coefficient_l1: float
    effective_ridge: float
    calibration_diagnostics: MultiAnchorReconstructionDiagnostics
    heldout_diagnostics: MultiAnchorReconstructionDiagnostics | None
    calibration_provenance: str | None
    heldout_provenance: str | None
    similarity_products: int
    similarity_multiply_adds: int
    worst_case_similarity_products: int
    projection_fit_count: int
    projection_solve_cubic_work: int
    peak_similarity_elements: int
    peak_workspace_elements: int
    max_similarity_products: int
    max_projection_fits: int
    max_sparse_entries: int
    max_workspace_elements: int
    similarity_chunk_size: int
    cluster_distance_chunk_size: int
    materialized_full_source_target_matrix: bool
    medoid_cluster_report: GatedMLPActivationClusterReport
    invariants: GatedMLPMultiAnchorInvariants

    @property
    def full_source_coverage(self) -> bool:
        return self.invariants.all_source_paths_covered

    @property
    def post_swiglu_fit_relative_error(self) -> float:
        return self.calibration_diagnostics.post_swiglu_relative_l2

    @property
    def heldout_post_swiglu_fit_relative_error(self) -> float | None:
        if self.heldout_diagnostics is None:
            return None
        return self.heldout_diagnostics.post_swiglu_relative_l2

    @property
    def consumer_weighted_fit_relative_error(self) -> float | None:
        return self.calibration_diagnostics.consumer_output_relative_l2

    @property
    def heldout_ready(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "deterministic-sparse-multi-anchor-gated-mlp-real-medoid",
            "source_neurons": self.source_neurons,
            "target_neurons": self.target_neurons,
            "observation_rows": self.observation_rows,
            "anchors_per_source": self.anchors_per_source,
            "max_candidate_targets": self.max_candidate_targets,
            "candidate_pool_size": self.candidate_pool_size,
            "candidate_pool_strategy": self.candidate_pool_strategy,
            "medoid_source_neurons": list(self.medoid_source_neurons),
            "source_to_target": list(self.source_to_target),
            "source_anchor_offsets": list(self.source_anchor_offsets),
            "anchor_target_indices": list(self.anchor_target_indices),
            "anchor_coefficients": list(self.anchor_coefficients),
            "consumer_coefficients": list(self.anchor_coefficients),
            "anchor_counts": list(self.anchor_counts),
            "coefficient_l1_per_source": list(self.coefficient_l1_per_source),
            "total_sparse_entries": self.total_sparse_entries,
            "total_coefficient_l1": self.total_coefficient_l1,
            "mean_coefficient_l1": self.mean_coefficient_l1,
            "maximum_coefficient_l1": self.maximum_coefficient_l1,
            "effective_ridge": self.effective_ridge,
            "post_swiglu_fit_relative_error": (
                self.post_swiglu_fit_relative_error
            ),
            "consumer_weighted_fit_relative_error": (
                self.consumer_weighted_fit_relative_error
            ),
            "heldout_post_swiglu_fit_relative_error": (
                self.heldout_post_swiglu_fit_relative_error
            ),
            "calibration_diagnostics": self.calibration_diagnostics.as_dict(),
            "heldout_diagnostics": (
                None
                if self.heldout_diagnostics is None
                else self.heldout_diagnostics.as_dict()
            ),
            "calibration_provenance": self.calibration_provenance,
            "heldout_provenance": self.heldout_provenance,
            "heldout_ready": self.heldout_ready,
            "similarity_products": self.similarity_products,
            "similarity_multiply_adds": self.similarity_multiply_adds,
            "worst_case_similarity_products": (
                self.worst_case_similarity_products
            ),
            "projection_fit_count": self.projection_fit_count,
            "projection_solve_cubic_work": self.projection_solve_cubic_work,
            "peak_similarity_elements": self.peak_similarity_elements,
            "peak_workspace_elements": self.peak_workspace_elements,
            "max_similarity_products": self.max_similarity_products,
            "max_projection_fits": self.max_projection_fits,
            "max_sparse_entries": self.max_sparse_entries,
            "max_workspace_elements": self.max_workspace_elements,
            "similarity_chunk_size": self.similarity_chunk_size,
            "cluster_distance_chunk_size": self.cluster_distance_chunk_size,
            "materialized_full_source_target_matrix": (
                self.materialized_full_source_target_matrix
            ),
            "producer_selection": "copy-real-source-neuron-row",
            "producer_aggregation": "none",
            "consumer_aggregation": "sparse-post-swiglu-ridge-index-add",
            "full_source_coverage": self.full_source_coverage,
            "medoid_cluster_report": self.medoid_cluster_report.as_dict(),
            "invariants": self.invariants.as_dict(),
            "deterministic": True,
            "zero_gradient": True,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class GatedMLPMultiAnchorPlan:
    """Sparse real-medoid plan compatible with the blank-bootstrap MLP API."""

    source_neurons: int
    target_neurons: int
    anchors_per_source: int
    medoid_source_neurons: tuple[int, ...]
    source_to_target: tuple[int, ...]
    source_anchor_offsets: tuple[int, ...]
    anchor_target_indices: tuple[int, ...]
    anchor_coefficients: tuple[float, ...]
    report: GatedMLPMultiAnchorReport

    def __post_init__(self) -> None:
        source = _positive_int(self.source_neurons, "source_neurons")
        target = _positive_int(self.target_neurons, "target_neurons")
        anchor_limit = _positive_int(
            self.anchors_per_source, "anchors_per_source"
        )
        if target > source:
            raise MultiAnchorClusterInvariantError(
                "A multi-anchor plan cannot expand neurons."
            )
        if len(self.medoid_source_neurons) != target or len(
            set(self.medoid_source_neurons)
        ) != target:
            raise MultiAnchorClusterInvariantError(
                "The plan must contain one unique real source medoid per target."
            )
        if any(not 0 <= index < source for index in self.medoid_source_neurons):
            raise MultiAnchorClusterInvariantError(
                "A medoid source index is outside the source width."
            )
        if len(self.source_to_target) != source or any(
            not 0 <= index < target for index in self.source_to_target
        ):
            raise MultiAnchorClusterInvariantError(
                "source_to_target must cover every source with valid targets."
            )
        offsets = self.source_anchor_offsets
        if len(offsets) != source + 1 or offsets[0] != 0:
            raise MultiAnchorClusterInvariantError(
                "source_anchor_offsets must be a complete zero-based CSR offset."
            )
        if any(left >= right for left, right in zip(offsets, offsets[1:])):
            raise MultiAnchorClusterInvariantError(
                "Every source must own a nonempty sparse anchor span."
            )
        edge_count = offsets[-1]
        if edge_count != len(self.anchor_target_indices) or edge_count != len(
            self.anchor_coefficients
        ):
            raise MultiAnchorClusterInvariantError(
                "Sparse targets, coefficients, and offsets must agree."
            )
        if any(
            not 0 <= index < target for index in self.anchor_target_indices
        ) or not all(math.isfinite(value) for value in self.anchor_coefficients):
            raise MultiAnchorClusterInvariantError(
                "Sparse anchor targets and coefficients must be finite and valid."
            )
        for source_index in range(source):
            start, stop = offsets[source_index : source_index + 2]
            if stop - start > min(anchor_limit, target):
                raise MultiAnchorClusterInvariantError(
                    "A source sparse span exceeds anchors_per_source."
                )
            targets = self.anchor_target_indices[start:stop]
            if len(set(targets)) != len(targets):
                raise MultiAnchorClusterInvariantError(
                    "A source sparse span contains duplicate target anchors."
                )
        for target_index, source_index in enumerate(self.medoid_source_neurons):
            start, stop = offsets[source_index : source_index + 2]
            if (
                stop - start != 1
                or self.anchor_target_indices[start] != target_index
                or self.anchor_coefficients[start] != 1.0
            ):
                raise MultiAnchorClusterInvariantError(
                    "Every real medoid must self-map exactly with coefficient one."
                )
        if not isinstance(self.report, GatedMLPMultiAnchorReport):
            raise MultiAnchorClusterInvariantError(
                "report must be a GatedMLPMultiAnchorReport."
            )
        if (
            self.report.medoid_source_neurons != self.medoid_source_neurons
            or self.report.source_to_target != self.source_to_target
            or self.report.source_anchor_offsets != self.source_anchor_offsets
            or self.report.anchor_target_indices != self.anchor_target_indices
            or self.report.anchor_coefficients != self.anchor_coefficients
            or not self.report.invariants.all_hold
        ):
            raise MultiAnchorClusterInvariantError(
                "The report does not prove the supplied sparse plan."
            )

    @property
    def assignment(self) -> tuple[int, ...]:
        return self.source_to_target

    @property
    def medoids(self) -> tuple[int, ...]:
        return self.medoid_source_neurons

    @property
    def consumer_coefficients(self) -> tuple[float, ...]:
        return self.anchor_coefficients

    @property
    def full_source_coverage(self) -> bool:
        return self.report.full_source_coverage

    def anchor_targets_for_source(self, source_index: int) -> tuple[int, ...]:
        source = self._validated_source_index(source_index)
        start, stop = self.source_anchor_offsets[source : source + 2]
        return self.anchor_target_indices[start:stop]

    def coefficients_for_source(self, source_index: int) -> tuple[float, ...]:
        source = self._validated_source_index(source_index)
        start, stop = self.source_anchor_offsets[source : source + 2]
        return self.anchor_coefficients[start:stop]

    def candidate_targets_for_source(self, source_index: int) -> tuple[int, ...]:
        """Reconstruct the bounded deterministic pool used during fitting."""

        source = self._validated_source_index(source_index)
        medoid_lookup = {
            medoid: target
            for target, medoid in enumerate(self.medoid_source_neurons)
        }
        if source in medoid_lookup:
            return (medoid_lookup[source],)
        return _candidate_target_pool(
            source,
            self.source_to_target[source],
            self.target_neurons,
            self.report.max_candidate_targets,
        )

    def _validated_source_index(self, source_index: Any) -> int:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < self.source_neurons
        ):
            raise MultiAnchorClusterShapeError(
                "source_index is outside the source neuron range."
            )
        return source_index

    def _select_producer(self, source_tensor: Any, *, axis: int, name: str) -> Any:
        torch = _require_torch()
        with torch.no_grad():
            source = _runtime_tensor(torch, source_tensor, name)
            normalized = _normalized_axis(source.ndim, axis, name)
            if int(source.shape[normalized]) != self.source_neurons:
                raise MultiAnchorClusterShapeError(
                    f"{name} axis must have width {self.source_neurons}."
                )
            indices = torch.tensor(
                self.medoid_source_neurons,
                dtype=torch.long,
                device=source.device,
            )
            return source.index_select(normalized, indices).contiguous().clone()

    def select_gate_producer(self, source_tensor: Any, *, axis: int = 0) -> Any:
        """Copy gate producer rows from real source-neuron medoids."""

        return self._select_producer(source_tensor, axis=axis, name="gate producer")

    def select_up_producer(self, source_tensor: Any, *, axis: int = 0) -> Any:
        """Copy up producer rows from the same real source-neuron medoids."""

        return self._select_producer(source_tensor, axis=axis, name="up producer")

    def aggregate_down_consumer(self, source_tensor: Any, *, axis: int = -1) -> Any:
        """Aggregate every source consumer through sparse ``index_add_`` edges."""

        torch = _require_torch()
        with torch.no_grad():
            source = _runtime_tensor(torch, source_tensor, "down consumer")
            normalized = _normalized_axis(source.ndim, axis, "down consumer")
            if int(source.shape[normalized]) != self.source_neurons:
                raise MultiAnchorClusterShapeError(
                    "down consumer axis must have width "
                    f"{self.source_neurons}."
                )
            moved = source.movedim(normalized, 0)
            edge_sources = torch.repeat_interleave(
                torch.arange(self.source_neurons, device=source.device),
                torch.tensor(
                    [
                        self.source_anchor_offsets[index + 1]
                        - self.source_anchor_offsets[index]
                        for index in range(self.source_neurons)
                    ],
                    dtype=torch.long,
                    device=source.device,
                ),
            )
            targets = torch.tensor(
                self.anchor_target_indices,
                dtype=torch.long,
                device=source.device,
            )
            coefficients = torch.tensor(
                self.anchor_coefficients,
                dtype=source.dtype,
                device=source.device,
            )
            selected = moved.index_select(0, edge_sources)
            coefficient_shape = (len(self.anchor_coefficients),) + (1,) * (
                selected.ndim - 1
            )
            weighted = selected * coefficients.reshape(coefficient_shape)
            result = source.new_zeros((self.target_neurons, *moved.shape[1:]))
            result.index_add_(0, targets, weighted)
            result = result.movedim(0, normalized).contiguous()
            if not bool(torch.isfinite(result).all().item()):
                raise MultiAnchorClusterFiniteError(
                    "Sparse down-consumer aggregation produced NaN or infinity."
                )
            return result

    def evaluate_post_swiglu_reconstruction(
        self,
        post_swiglu_rows: Any,
        *,
        consumer_directions: Any | None = None,
        consumer_axis: int = -1,
    ) -> MultiAnchorReconstructionDiagnostics:
        """Evaluate the frozen sparse fit on calibration or untouched rows."""

        torch = _require_torch()
        post = _planning_tensor(torch, post_swiglu_rows, "post_swiglu_rows")
        if post.ndim != 2 or int(post.shape[1]) != self.source_neurons:
            raise MultiAnchorClusterShapeError(
                "post_swiglu_rows must have shape [rows, source_neurons]."
            )
        consumer = None
        if consumer_directions is not None:
            consumer = _consumer_source_major(
                torch,
                consumer_directions,
                source_neurons=self.source_neurons,
                axis=consumer_axis,
                name="consumer_directions",
            )
        return _reconstruction_diagnostics(
            torch,
            post,
            medoids=self.medoid_source_neurons,
            offsets=self.source_anchor_offsets,
            anchor_targets=self.anchor_target_indices,
            coefficients=self.anchor_coefficients,
            consumer_source_major=consumer,
        )

    reconstruction_diagnostics = evaluate_post_swiglu_reconstruction
    fold_gate_producer = select_gate_producer
    fold_up_producer = select_up_producer
    fold_down_consumer = aggregate_down_consumer


@dataclass(frozen=True, slots=True)
class _SparseSourceFit:
    targets: tuple[int, ...]
    coefficients: tuple[float, ...]
    projection_fits: int
    similarity_products: int
    solve_cubic_work: int


def _candidate_target_pool(
    source: int,
    primary_target: int,
    target_count: int,
    max_candidate_targets: int,
) -> tuple[int, ...]:
    """Return a bounded pool without inspecting every target activation."""

    pool_size = min(max_candidate_targets, target_count)
    result = [primary_target]
    seen = {primary_target}
    local_slots = (pool_size - 1) // 2
    radius = 1
    while len(result) - 1 < local_slots and radius < target_count:
        for candidate in (
            (primary_target - radius) % target_count,
            (primary_target + radius) % target_count,
        ):
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
                if len(result) - 1 >= local_slots:
                    break
        radius += 1

    seed = (
        (source + 1) * 2_654_435_761
        + (primary_target + 1) * 2_246_822_519
    ) % target_count
    stride = (
        (source + 1) * 1_103_515_245
        + (primary_target + 1) * 12_345
        + 1
    ) % target_count
    if stride == 0:
        stride = 1
    while math.gcd(stride, target_count) != 1:
        stride = (stride + 1) % target_count
        if stride == 0:
            stride = 1
    for offset in range(target_count):
        candidate = (seed + offset * stride) % target_count
        if candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
            if len(result) == pool_size:
                break
    if len(result) != pool_size or result[0] != primary_target:
        raise MultiAnchorClusterInvariantError(
            "The deterministic candidate pool is incomplete or lost its primary."
        )
    return tuple(result)


def _ridge_coefficients(
    torch: Any,
    design: Any,
    signal: Any,
    *,
    effective_ridge: float,
) -> Any:
    anchor_count = int(design.shape[1])
    gram = design.T.matmul(design)
    regularized = gram + torch.eye(anchor_count, dtype=torch.float64) * (
        effective_ridge
    )
    try:
        coefficients = torch.linalg.solve(regularized, design.T.matmul(signal))
    except RuntimeError as exc:
        raise MultiAnchorClusterFiniteError(
            "A bounded ridge projection could not be solved."
        ) from exc
    if not bool(torch.isfinite(coefficients).all().item()):
        raise MultiAnchorClusterFiniteError(
            "A ridge projection produced NaN or infinity."
        )
    return coefficients


def _best_residual_anchor(
    torch: Any,
    medoid_signatures: Any,
    residual: Any,
    selected: set[int],
    *,
    candidate_targets: tuple[int, ...],
    similarity_chunk_size: int,
) -> int | None:
    tiny = torch.finfo(torch.float64).tiny
    best_score = -1.0
    best_target: int | None = None
    for start in range(0, len(candidate_targets), similarity_chunk_size):
        stop = min(start + similarity_chunk_size, len(candidate_targets))
        target_chunk = candidate_targets[start:stop]
        target_indices = torch.tensor(target_chunk, dtype=torch.long)
        chunk = medoid_signatures.index_select(0, target_indices)
        norms = torch.linalg.vector_norm(chunk, dim=1).clamp_min(tiny)
        scores = chunk.matmul(residual).abs() / norms
        for local_index, target in enumerate(target_chunk):
            if target in selected:
                scores[local_index] = -1.0
        local_score, local_index = torch.max(scores, dim=0)
        score = float(local_score.item())
        target = target_chunk[int(local_index.item())]
        if score > best_score:
            best_score = score
            best_target = target
    numerical_floor = (
        torch.finfo(torch.float64).eps
        * max(1.0, float(torch.linalg.vector_norm(residual).item()))
        * 64.0
    )
    if best_target is None or best_score <= numerical_floor:
        return None
    return best_target


def _fit_source_path(
    torch: Any,
    signal: Any,
    medoid_signatures: Any,
    *,
    primary_target: int,
    candidate_targets: tuple[int, ...],
    anchors_per_source: int,
    effective_ridge: float,
    max_coefficient_l1: float,
    similarity_chunk_size: int,
) -> _SparseSourceFit:
    selected = [primary_target]
    selected_set = {primary_target}
    projection_fits = 0
    similarity_products = 0
    solve_work = 0
    signal_energy = float(signal.square().sum().item())
    accepted_coefficients: Any | None = None
    accepted_residual: Any | None = None
    accepted_residual_energy = math.inf

    for anchor_count in range(1, anchors_per_source + 1):
        indices = torch.tensor(selected, dtype=torch.long)
        design = medoid_signatures.index_select(0, indices).T.contiguous()
        if anchor_count == 1 and bool(
            torch.equal(signal, design[:, 0])
        ):
            coefficients = torch.ones(1, dtype=torch.float64)
        else:
            coefficients = _ridge_coefficients(
                torch,
                design,
                signal,
                effective_ridge=effective_ridge,
            )
            projection_fits += 1
            solve_work += anchor_count**3
        coefficient_l1 = float(coefficients.abs().sum().item())
        if not math.isfinite(coefficient_l1):
            raise MultiAnchorClusterFiniteError(
                "A source coefficient L1 diagnostic is non-finite."
            )
        if coefficient_l1 > max_coefficient_l1:
            if accepted_coefficients is None:
                raise MultiAnchorClusterInvariantError(
                    "No coefficient-L1-bounded fit exists for a source path."
                )
            selected.pop()
            break
        residual = signal - design.matmul(coefficients)
        residual_energy = float(residual.square().sum().item())
        improvement_floor = (
            torch.finfo(torch.float64).eps * max(1.0, signal_energy) * 64.0
        )
        if (
            accepted_coefficients is not None
            and residual_energy >= accepted_residual_energy - improvement_floor
        ):
            selected.pop()
            break
        accepted_coefficients = coefficients
        accepted_residual = residual
        accepted_residual_energy = residual_energy
        if anchor_count >= anchors_per_source:
            break
        if residual_energy <= improvement_floor:
            break
        candidate = _best_residual_anchor(
            torch,
            medoid_signatures,
            residual,
            selected_set,
            candidate_targets=candidate_targets,
            similarity_chunk_size=similarity_chunk_size,
        )
        similarity_products += len(candidate_targets)
        if candidate is None:
            break
        selected.append(candidate)
        selected_set.add(candidate)

    if accepted_coefficients is None or accepted_residual is None:
        raise MultiAnchorClusterInvariantError(
            "Every source path must have at least one bounded sparse fit."
        )
    coefficient_tuple = tuple(
        float(value) for value in accepted_coefficients.tolist()
    )
    accepted_count = len(coefficient_tuple)
    return _SparseSourceFit(
        targets=tuple(selected[:accepted_count]),
        coefficients=coefficient_tuple,
        projection_fits=projection_fits,
        similarity_products=similarity_products,
        solve_cubic_work=solve_work,
    )


def _edge_source_indices(offsets: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        source
        for source, (start, stop) in enumerate(zip(offsets, offsets[1:]))
        for _ in range(stop - start)
    )


def _reconstruction_diagnostics(
    torch: Any,
    post: Any,
    *,
    medoids: tuple[int, ...],
    offsets: tuple[int, ...],
    anchor_targets: tuple[int, ...],
    coefficients: tuple[float, ...],
    consumer_source_major: Any | None,
) -> MultiAnchorReconstructionDiagnostics:
    source_neurons = int(post.shape[1])
    medoid_index = torch.tensor(medoids, dtype=torch.long)
    medoid_rows = post.index_select(1, medoid_index)
    residual_square_sum = 0.0
    source_square_sum = float(post.square().sum().item())
    source_relative_errors: list[float] = []
    zero_signal_count = 0
    epsilon = torch.finfo(torch.float64).eps
    for source in range(source_neurons):
        start, stop = offsets[source : source + 2]
        targets = torch.tensor(anchor_targets[start:stop], dtype=torch.long)
        values = torch.tensor(coefficients[start:stop], dtype=torch.float64)
        predicted = medoid_rows.index_select(1, targets).matmul(values)
        signal = post[:, source]
        residual_norm = float(torch.linalg.vector_norm(signal - predicted).item())
        signal_norm = float(torch.linalg.vector_norm(signal).item())
        if signal_norm <= epsilon:
            zero_signal_count += 1
        source_relative_errors.append(residual_norm / max(signal_norm, epsilon))
        residual_square_sum += residual_norm * residual_norm
    relative_l2 = math.sqrt(residual_square_sum) / max(
        math.sqrt(source_square_sum), epsilon
    )
    root_mean_square_error = math.sqrt(
        residual_square_sum / int(post.numel())
    )
    consumer_relative: float | None = None
    if consumer_source_major is not None:
        if int(consumer_source_major.shape[0]) != source_neurons:
            raise MultiAnchorClusterShapeError(
                "consumer directions do not cover every source neuron."
            )
        edge_sources = torch.tensor(
            _edge_source_indices(offsets), dtype=torch.long
        )
        edge_targets = torch.tensor(anchor_targets, dtype=torch.long)
        edge_values = torch.tensor(coefficients, dtype=torch.float64)
        edge_consumer = consumer_source_major.index_select(0, edge_sources)
        weighted = edge_consumer * edge_values.unsqueeze(1)
        target_consumer = torch.zeros(
            (len(medoids), int(consumer_source_major.shape[1])),
            dtype=torch.float64,
        )
        target_consumer.index_add_(0, edge_targets, weighted)
        source_output = post.matmul(consumer_source_major)
        folded_output = medoid_rows.matmul(target_consumer)
        consumer_relative = float(
            (
                torch.linalg.vector_norm(source_output - folded_output)
                / torch.linalg.vector_norm(source_output).clamp_min(epsilon)
            ).item()
        )
    diagnostics = MultiAnchorReconstructionDiagnostics(
        observation_rows=int(post.shape[0]),
        post_swiglu_relative_l2=relative_l2,
        post_swiglu_root_mean_square_error=root_mean_square_error,
        mean_source_relative_l2=(
            sum(source_relative_errors) / source_neurons
        ),
        maximum_source_relative_l2=max(source_relative_errors),
        zero_signal_source_count=zero_signal_count,
        consumer_output_relative_l2=consumer_relative,
        finite=True,
    )
    numeric_values = (
        diagnostics.post_swiglu_relative_l2,
        diagnostics.post_swiglu_root_mean_square_error,
        diagnostics.mean_source_relative_l2,
        diagnostics.maximum_source_relative_l2,
    )
    if not all(math.isfinite(value) for value in numeric_values) or (
        consumer_relative is not None and not math.isfinite(consumer_relative)
    ):
        raise MultiAnchorClusterFiniteError(
            "Reconstruction diagnostics contain NaN or infinity."
        )
    return diagnostics


def build_gated_mlp_multi_anchor_cluster_plan(
    gate_rows: Any,
    up_rows: Any,
    post_swiglu_rows: Any,
    contribution_rows: Any | None = None,
    target_neurons: int | None = None,
    *,
    consumer_directions: Any | None = None,
    consumer_axis: int = -1,
    source_neurons: int | None = None,
    anchors_per_source: int = 4,
    max_candidate_targets: int = 64,
    ridge: float = 1e-8,
    max_coefficient_l1: float = 64.0,
    max_iterations: int = 16,
    distance_chunk_size: int = 512,
    similarity_chunk_size: int = 512,
    max_similarity_products: int = 64_000_000,
    max_projection_fits: int = 1_000_000,
    max_sparse_entries: int = 1_000_000,
    max_workspace_elements: int = 16_000_000,
    heldout_post_swiglu_rows: Any | None = None,
    validation_post_swiglu_rows: Any | None = None,
    calibration_provenance: str | None = None,
    heldout_provenance: str | None = None,
) -> GatedMLPMultiAnchorPlan:
    """Build a bounded sparse multi-anchor gated-MLP folding plan.

    ``contribution_rows`` follows the one-anchor API and begins with
    ``[observation_rows, source_neurons]``.  Alternatively, callers may pass a
    static down/consumer tensor through ``consumer_directions`` and identify
    its source-neuron axis with ``consumer_axis``; consumer-norm-weighted
    contribution rows are then derived without dividing by gate values.

    Every target producer is copied from a real source medoid.  Every source
    owns at least one CSR edge to a target, each medoid owns exactly its one
    self-edge, and non-medoid paths use at most ``anchors_per_source`` targets.
    Additional anchors are searched only inside a deterministic pool of at
    most ``max_candidate_targets`` (hard-capped at 64), with the primary
    cluster target always included first.
    """

    torch = _require_torch()
    if target_neurons is None:
        raise MultiAnchorClusterShapeError(
            "target_neurons must be supplied as a positive integer."
        )
    target = _positive_int(target_neurons, "target_neurons")
    anchor_limit = _positive_int(anchors_per_source, "anchors_per_source")
    if anchor_limit > _HARD_MAX_ANCHORS_PER_SOURCE:
        raise MultiAnchorClusterShapeError(
            "anchors_per_source exceeds the hard cap of "
            f"{_HARD_MAX_ANCHORS_PER_SOURCE}."
        )
    candidate_limit = _positive_int(
        max_candidate_targets, "max_candidate_targets"
    )
    if candidate_limit > _HARD_MAX_CANDIDATE_TARGETS:
        raise MultiAnchorClusterShapeError(
            "max_candidate_targets exceeds the hard cap of "
            f"{_HARD_MAX_CANDIDATE_TARGETS}."
        )
    ridge_value = _positive_float(ridge, "ridge")
    coefficient_l1_limit = _positive_float(
        max_coefficient_l1, "max_coefficient_l1"
    )
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    distance_chunk = _positive_int(distance_chunk_size, "distance_chunk_size")
    similarity_chunk = _positive_int(
        similarity_chunk_size, "similarity_chunk_size"
    )
    similarity_limit = _positive_int(
        max_similarity_products, "max_similarity_products"
    )
    projection_limit = _positive_int(
        max_projection_fits, "max_projection_fits"
    )
    sparse_limit = _positive_int(max_sparse_entries, "max_sparse_entries")
    workspace_limit = _positive_int(
        max_workspace_elements, "max_workspace_elements"
    )
    calibration_source = _optional_provenance(
        calibration_provenance, "calibration_provenance"
    )
    heldout_source = _optional_provenance(
        heldout_provenance, "heldout_provenance"
    )
    if (
        heldout_post_swiglu_rows is not None
        and validation_post_swiglu_rows is not None
    ):
        raise MultiAnchorClusterShapeError(
            "Supply only one of heldout_post_swiglu_rows and "
            "validation_post_swiglu_rows."
        )
    heldout_rows = (
        heldout_post_swiglu_rows
        if heldout_post_swiglu_rows is not None
        else validation_post_swiglu_rows
    )

    with torch.no_grad():
        gate, up, post, observation_rows, inferred_source = (
            _aligned_activation_rows(
                torch,
                gate_rows,
                up_rows,
                post_swiglu_rows,
                source_neurons,
            )
        )
        if target > inferred_source:
            raise MultiAnchorClusterShapeError(
                "target_neurons cannot exceed source_neurons."
            )
        candidate_pool_size = min(candidate_limit, target)
        effective_anchor_limit = min(
            anchor_limit, target, candidate_pool_size
        )
        nonmedoid_count = inferred_source - target
        worst_similarity_products = (
            nonmedoid_count
            * max(0, effective_anchor_limit - 1)
            * candidate_pool_size
        )
        worst_projection_fits = nonmedoid_count * effective_anchor_limit
        worst_sparse_entries = target + nonmedoid_count * effective_anchor_limit
        if worst_similarity_products > similarity_limit:
            raise MultiAnchorClusterShapeError(
                "Multi-anchor similarity work exceeds max_similarity_products."
            )
        if worst_projection_fits > projection_limit:
            raise MultiAnchorClusterShapeError(
                "Multi-anchor projection work exceeds max_projection_fits."
            )
        if worst_sparse_entries > sparse_limit:
            raise MultiAnchorClusterShapeError(
                "Sparse plan size exceeds max_sparse_entries."
            )

        consumer = None
        if consumer_directions is not None:
            consumer = _consumer_source_major(
                torch,
                consumer_directions,
                source_neurons=inferred_source,
                axis=consumer_axis,
                name="consumer_directions",
            )
        if contribution_rows is None:
            if consumer is None:
                raise MultiAnchorClusterShapeError(
                    "Supply contribution_rows or consumer_directions."
                )
            direction_norm = torch.linalg.vector_norm(consumer, dim=1)
            contribution = post * direction_norm.unsqueeze(0)
        else:
            contribution = _planning_tensor(
                torch, contribution_rows, "contribution_rows"
            )
            if (
                contribution.ndim < 2
                or int(contribution.shape[0]) != observation_rows
                or int(contribution.shape[1]) != inferred_source
            ):
                raise MultiAnchorClusterShapeError(
                    "contribution_rows must begin with "
                    "[observation_rows, source_neurons]."
                )

        cluster_chunk_cap = max(1, workspace_limit // target)
        cluster_chunk = min(distance_chunk, inferred_source, cluster_chunk_cap)
        if inferred_source > 1 and target < inferred_source:
            cluster_chunk = min(cluster_chunk, inferred_source - 1)
        peak_cluster_distance_elements = cluster_chunk * target
        peak_similarity_elements = min(similarity_chunk, candidate_pool_size)
        peak_fit_workspace = (
            target * observation_rows
            + observation_rows * effective_anchor_limit
            + effective_anchor_limit * effective_anchor_limit
            + peak_similarity_elements
            + peak_cluster_distance_elements
        )
        if peak_fit_workspace > workspace_limit:
            raise MultiAnchorClusterShapeError(
                "Multi-anchor fitting memory exceeds max_workspace_elements."
            )

        medoid_plan = build_gated_mlp_activation_cluster_plan(
            gate,
            up,
            post,
            contribution,
            target_neurons=target,
            source_neurons=inferred_source,
            ridge=ridge_value,
            max_iterations=iteration_limit,
            distance_chunk_size=cluster_chunk,
        )
        post_signatures = post.T.contiguous()
        medoid_tensor = torch.tensor(
            medoid_plan.medoid_source_neurons, dtype=torch.long
        )
        medoid_signatures = post_signatures.index_select(0, medoid_tensor)
        mean_medoid_energy = float(
            medoid_signatures.square().sum(dim=1).mean().item()
        )
        effective_ridge = max(
            ridge_value * max(mean_medoid_energy, torch.finfo(torch.float64).tiny),
            torch.finfo(torch.float64).tiny,
        )
        medoid_lookup = {
            source: target_index
            for target_index, source in enumerate(
                medoid_plan.medoid_source_neurons
            )
        }
        offsets = [0]
        anchor_targets: list[int] = []
        anchor_coefficients: list[float] = []
        projection_fit_count = 0
        similarity_products = 0
        solve_cubic_work = 0
        for source in range(inferred_source):
            if source in medoid_lookup:
                fit = _SparseSourceFit(
                    targets=(medoid_lookup[source],),
                    coefficients=(1.0,),
                    projection_fits=0,
                    similarity_products=0,
                    solve_cubic_work=0,
                )
            else:
                candidate_targets = _candidate_target_pool(
                    source,
                    medoid_plan.source_to_target[source],
                    target,
                    candidate_limit,
                )
                fit = _fit_source_path(
                    torch,
                    post_signatures[source],
                    medoid_signatures,
                    primary_target=medoid_plan.source_to_target[source],
                    candidate_targets=candidate_targets,
                    anchors_per_source=effective_anchor_limit,
                    effective_ridge=effective_ridge,
                    max_coefficient_l1=coefficient_l1_limit,
                    similarity_chunk_size=similarity_chunk,
                )
            anchor_targets.extend(fit.targets)
            anchor_coefficients.extend(fit.coefficients)
            offsets.append(len(anchor_targets))
            projection_fit_count += fit.projection_fits
            similarity_products += fit.similarity_products
            solve_cubic_work += fit.solve_cubic_work

        offset_tuple = tuple(offsets)
        target_tuple = tuple(anchor_targets)
        coefficient_tuple = tuple(anchor_coefficients)
        medoid_tuple = tuple(medoid_plan.medoid_source_neurons)
        assignment_tuple = tuple(medoid_plan.source_to_target)
        calibration = _reconstruction_diagnostics(
            torch,
            post,
            medoids=medoid_tuple,
            offsets=offset_tuple,
            anchor_targets=target_tuple,
            coefficients=coefficient_tuple,
            consumer_source_major=consumer,
        )
        heldout = None
        if heldout_rows is not None:
            heldout_post = _planning_tensor(
                torch, heldout_rows, "heldout_post_swiglu_rows"
            )
            if heldout_post.ndim != 2 or int(heldout_post.shape[1]) != inferred_source:
                raise MultiAnchorClusterShapeError(
                    "heldout_post_swiglu_rows must have shape "
                    "[rows, source_neurons]."
                )
            heldout = _reconstruction_diagnostics(
                torch,
                heldout_post,
                medoids=medoid_tuple,
                offsets=offset_tuple,
                anchor_targets=target_tuple,
                coefficients=coefficient_tuple,
                consumer_source_major=consumer,
            )

    anchor_counts = tuple(
        stop - start for start, stop in zip(offset_tuple, offset_tuple[1:])
    )
    coefficient_l1 = tuple(
        sum(abs(value) for value in coefficient_tuple[start:stop])
        for start, stop in zip(offset_tuple, offset_tuple[1:])
    )
    total_l1 = sum(coefficient_l1)
    finite_diagnostics = calibration.finite and (
        heldout is None or heldout.finite
    )
    medoids_self_map = all(
        offset_tuple[source + 1] - offset_tuple[source] == 1
        and target_tuple[offset_tuple[source]] == target_index
        and coefficient_tuple[offset_tuple[source]] == 1.0
        for target_index, source in enumerate(medoid_tuple)
    )
    invariants = GatedMLPMultiAnchorInvariants(
        all_source_paths_covered=(
            len(anchor_counts) == inferred_source
            and sum(anchor_counts) == len(target_tuple)
        ),
        nonempty_source_anchor_sets=all(count >= 1 for count in anchor_counts),
        anchor_count_bounded=all(
            count <= effective_anchor_limit for count in anchor_counts
        ),
        candidate_pool_bounded=(
            candidate_pool_size <= candidate_limit
            and candidate_limit <= _HARD_MAX_CANDIDATE_TARGETS
        ),
        primary_anchor_included=all(
            target_tuple[offset_tuple[source]] == assignment_tuple[source]
            for source in range(inferred_source)
        ),
        real_source_neuron_medoids=(
            len(set(medoid_tuple)) == target
            and all(0 <= source < inferred_source for source in medoid_tuple)
        ),
        medoids_self_map_exactly=medoids_self_map,
        shared_gate_up_producers=True,
        finite_consumer_coefficients=all(
            math.isfinite(value) for value in coefficient_tuple
        ),
        finite_reconstruction_diagnostics=finite_diagnostics,
        no_producer_averaging=True,
        sparse_consumer_index_add=True,
        no_dense_source_target_matrix=True,
        hard_work_bounds_respected=(
            similarity_products <= similarity_limit
            and projection_fit_count <= projection_limit
        ),
        hard_memory_bounds_respected=(
            len(target_tuple) <= sparse_limit
            and peak_fit_workspace <= workspace_limit
        ),
        zero_gradient=True,
        deterministic=True,
    )
    if not invariants.all_hold:
        raise MultiAnchorClusterInvariantError(
            "The computed sparse multi-anchor invariants do not all hold."
        )
    report = GatedMLPMultiAnchorReport(
        source_neurons=inferred_source,
        target_neurons=target,
        observation_rows=observation_rows,
        anchors_per_source=anchor_limit,
        max_candidate_targets=candidate_limit,
        candidate_pool_size=candidate_pool_size,
        candidate_pool_strategy=_CANDIDATE_POOL_STRATEGY,
        medoid_source_neurons=medoid_tuple,
        source_to_target=assignment_tuple,
        source_anchor_offsets=offset_tuple,
        anchor_target_indices=target_tuple,
        anchor_coefficients=coefficient_tuple,
        anchor_counts=anchor_counts,
        coefficient_l1_per_source=coefficient_l1,
        total_sparse_entries=len(target_tuple),
        total_coefficient_l1=total_l1,
        mean_coefficient_l1=total_l1 / inferred_source,
        maximum_coefficient_l1=max(coefficient_l1),
        effective_ridge=effective_ridge,
        calibration_diagnostics=calibration,
        heldout_diagnostics=heldout,
        calibration_provenance=calibration_source,
        heldout_provenance=heldout_source,
        similarity_products=similarity_products,
        similarity_multiply_adds=similarity_products * observation_rows,
        worst_case_similarity_products=worst_similarity_products,
        projection_fit_count=projection_fit_count,
        projection_solve_cubic_work=solve_cubic_work,
        peak_similarity_elements=peak_similarity_elements,
        peak_workspace_elements=peak_fit_workspace,
        max_similarity_products=similarity_limit,
        max_projection_fits=projection_limit,
        max_sparse_entries=sparse_limit,
        max_workspace_elements=workspace_limit,
        similarity_chunk_size=similarity_chunk,
        cluster_distance_chunk_size=cluster_chunk,
        materialized_full_source_target_matrix=False,
        medoid_cluster_report=medoid_plan.report,
        invariants=invariants,
    )
    return GatedMLPMultiAnchorPlan(
        source_neurons=inferred_source,
        target_neurons=target,
        anchors_per_source=anchor_limit,
        medoid_source_neurons=medoid_tuple,
        source_to_target=assignment_tuple,
        source_anchor_offsets=offset_tuple,
        anchor_target_indices=target_tuple,
        anchor_coefficients=coefficient_tuple,
        report=report,
    )


build_gated_mlp_multi_anchor_plan = build_gated_mlp_multi_anchor_cluster_plan
plan_gated_mlp_multi_anchor_clusters = build_gated_mlp_multi_anchor_cluster_plan


__all__ = [
    "GatedMLPMultiAnchorInvariants",
    "GatedMLPMultiAnchorPlan",
    "GatedMLPMultiAnchorReport",
    "MultiAnchorClusterError",
    "MultiAnchorClusterFiniteError",
    "MultiAnchorClusterInvariantError",
    "MultiAnchorClusterShapeError",
    "MultiAnchorReconstructionDiagnostics",
    "build_gated_mlp_multi_anchor_cluster_plan",
    "build_gated_mlp_multi_anchor_plan",
    "plan_gated_mlp_multi_anchor_clusters",
]
