"""No-gradient activation alignment and MLP parameter sharing.

This module implements the alignment, averaging, and tying direction from
"Merging Feed-Forward Sublayers for Compressed Transformers" (2025) for an
extreme-parameter-budget intermediate representation.  Inputs are explicit
tensor bundles and already-collected pre-down activations; this module never
runs a model, constructs an optimizer, or invokes backward.

The permutation convention is ``permutation[anchor_neuron] = source_neuron``.
Accordingly, all producer rows (and producer biases) and the consumer/down
columns are indexed by the same permutation before bundles are averaged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_EXACT_ASSIGNMENT_LIMIT = 256
MAX_EXACT_ASSIGNMENT_LIMIT = 512
DEFAULT_CORRELATION_CHUNK_SIZE = 256
_TIE_BREAK = "lowest source-neuron index"


class ParameterSharingError(ValueError):
    """Base class for fail-closed parameter-sharing validation failures."""


class ShapeValidationError(ParameterSharingError):
    """A tensor, sequence, or module parameter has an invalid shape."""


class TopologyValidationError(ParameterSharingError):
    """An MLP bundle or module layout has an invalid topology."""


class FiniteValidationError(ParameterSharingError):
    """An input, intermediate, or result contains a non-finite value."""


class AssignmentCollisionError(ParameterSharingError):
    """A proposed assignment maps multiple anchor neurons to one source."""


class IncompleteAssignmentError(ParameterSharingError):
    """A proposed assignment does not cover both neuron sets exactly once."""


class ModuleSharingError(ParameterSharingError):
    """PyTorch modules cannot be tied or expanded under the requested layout."""


class IncompleteSharingError(ModuleSharingError):
    """A module sequence is only partially tied for at least one parameter."""


@dataclass(frozen=True, slots=True)
class MLPTensorBundle:
    """Explicit dense or gated MLP tensors in PyTorch linear orientation.

    Dense topology supplies ``input_weight`` and ``down_weight``.  Gated
    topology supplies ``gate_weight``, ``up_weight``, and ``down_weight``.
    Producer weights have shape ``[intermediate, input]`` and the down weight
    has shape ``[output, intermediate]``.  Biases are independently optional.
    """

    input_weight: Any | None = None
    down_weight: Any | None = None
    gate_weight: Any | None = None
    up_weight: Any | None = None
    input_bias: Any | None = None
    down_bias: Any | None = None
    gate_bias: Any | None = None
    up_bias: Any | None = None

    @property
    def topology(self) -> str:
        """Return ``dense`` or ``gated``, rejecting mixed/incomplete bundles."""

        dense = self.input_weight is not None
        gate = self.gate_weight is not None
        up = self.up_weight is not None
        if dense and not gate and not up:
            return "dense"
        if not dense and gate and up:
            return "gated"
        raise TopologyValidationError(
            "A bundle must contain either input/down weights or "
            "gate/up/down weights, without mixing topologies."
        )


@dataclass(frozen=True, slots=True)
class AssignmentDiagnostics:
    """Determinism, bounded-work, objective, and approximation evidence."""

    solver: str
    neuron_count: int
    certified_optimal: bool
    approximate: bool
    deterministic: bool
    tie_break: str
    correlation_values_computed: int
    correlation_chunks: int
    chunk_size: int
    assignment_steps: int
    assignment_step_limit: int
    objective: float
    relaxed_row_max_upper_bound: float
    relaxed_upper_bound_gap: float
    suboptimality_upper_bound: float
    mean_assigned_correlation: float
    minimum_assigned_correlation: float
    maximum_assigned_correlation: float
    complete: bool
    collision_count: int
    finite: bool


@dataclass(frozen=True, slots=True)
class NeuronAssignmentResult:
    """One source neuron for every anchor neuron."""

    permutation: Any
    assigned_correlations: Any
    correlation_matrix: Any | None
    diagnostics: AssignmentDiagnostics


@dataclass(frozen=True, slots=True)
class ParameterCountReport:
    """Logical execution size and physical unique storage, reported separately."""

    topology: str
    execution_depth: int
    dense_equivalent_parameters: int
    unique_parameters: int
    parameters_saved: int
    dense_equivalent_parameter_objects: int
    unique_parameter_objects: int
    unique_parameter_sets: int
    fully_shared: bool
    fully_expanded: bool

    @property
    def dense_equivalent_parameter_count(self) -> int:
        """Descriptive alias for the logical parameter count."""

        return self.dense_equivalent_parameters

    @property
    def unique_parameter_count(self) -> int:
        """Descriptive alias for the physical unique parameter count."""

        return self.unique_parameters


@dataclass(frozen=True, slots=True)
class MLPMergeResult:
    """Aligned bundles, their weighted mean, and planned sharing evidence."""

    merged_bundle: MLPTensorBundle
    aligned_bundles: tuple[MLPTensorBundle, ...]
    assignments: tuple[NeuronAssignmentResult, ...]
    permutations: tuple[Any, ...]
    normalized_weights: Any
    anchor_index: int
    parameter_counts: ParameterCountReport

    @property
    def execution_depth(self) -> int:
        """Number of MLP executions retained after parameter tying."""

        return self.parameter_counts.execution_depth


@dataclass(frozen=True, slots=True)
class MLPModuleLayout:
    """Dotted parameter paths connecting a module to an explicit MLP bundle."""

    input_weight: str | None = None
    down_weight: str | None = None
    gate_weight: str | None = None
    up_weight: str | None = None
    input_bias: str | None = None
    down_bias: str | None = None
    gate_bias: str | None = None
    up_bias: str | None = None

    @classmethod
    def dense(
        cls,
        input_module: str,
        down_module: str,
        *,
        input_bias: bool = True,
        down_bias: bool = True,
    ) -> MLPModuleLayout:
        """Build a layout for two conventional modules with weight/bias fields."""

        return cls(
            input_weight=f"{input_module}.weight",
            down_weight=f"{down_module}.weight",
            input_bias=f"{input_module}.bias" if input_bias else None,
            down_bias=f"{down_module}.bias" if down_bias else None,
        )

    @classmethod
    def gated(
        cls,
        gate_module: str,
        up_module: str,
        down_module: str,
        *,
        gate_bias: bool = True,
        up_bias: bool = True,
        down_bias: bool = True,
    ) -> MLPModuleLayout:
        """Build a layout for conventional gate/up/down projection modules."""

        return cls(
            gate_weight=f"{gate_module}.weight",
            up_weight=f"{up_module}.weight",
            down_weight=f"{down_module}.weight",
            gate_bias=f"{gate_module}.bias" if gate_bias else None,
            up_bias=f"{up_module}.bias" if up_bias else None,
            down_bias=f"{down_module}.bias" if down_bias else None,
        )

    @property
    def topology(self) -> str:
        """Return the strict topology encoded by the weight paths."""

        dense = self.input_weight is not None
        gate = self.gate_weight is not None
        up = self.up_weight is not None
        if self.down_weight is None:
            raise TopologyValidationError("A module layout requires down_weight.")
        if dense and not gate and not up:
            return "dense"
        if not dense and gate and up:
            return "gated"
        raise TopologyValidationError(
            "A module layout must describe input/down or gate/up/down weights."
        )


@dataclass(frozen=True, slots=True)
class _ValidatedBundle:
    topology: str
    tensors: dict[str, Any]
    intermediate_width: int
    input_width: int
    output_width: int


@dataclass(frozen=True, slots=True)
class _ParameterBinding:
    owner: Any
    name: str
    parameter: Any


_ALL_ROLES = (
    "input_weight",
    "gate_weight",
    "up_weight",
    "down_weight",
    "input_bias",
    "gate_bias",
    "up_bias",
    "down_bias",
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without HF extra
        raise RuntimeError(
            "PyTorch is required for neural_forge.hf_parameter_sharing."
        ) from exc
    return torch


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShapeValidationError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ShapeValidationError(f"{name} cannot exceed {maximum}.")
    return value


def _nonnegative_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShapeValidationError(f"{name} must be a non-negative integer.")
    if maximum is not None and value > maximum:
        raise ShapeValidationError(f"{name} cannot exceed {maximum}.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ShapeValidationError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ShapeValidationError(f"{name} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ShapeValidationError(f"{name} must be finite and non-negative.")
    return result


def _as_float64_tensor(
    torch,
    value: Any,
    name: str,
    *,
    dimensions: int,
):
    if not isinstance(value, torch.Tensor):
        raise ShapeValidationError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise ShapeValidationError(f"{name} must be a dense strided tensor.")
    if value.ndim != dimensions:
        raise ShapeValidationError(
            f"{name} must be {dimensions}-dimensional, got {value.ndim}."
        )
    if any(int(size) <= 0 for size in value.shape):
        raise ShapeValidationError(f"{name} cannot have an empty dimension.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise ShapeValidationError(f"{name} must have a real floating-point dtype.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise FiniteValidationError(f"{name} contains NaN or infinity.")
    return result


def _bundle_roles(topology: str, tensors: Mapping[str, Any]) -> tuple[str, ...]:
    producer_weights = (
        ("input_weight",) if topology == "dense" else ("gate_weight", "up_weight")
    )
    producer_biases = (
        ("input_bias",) if topology == "dense" else ("gate_bias", "up_bias")
    )
    ordered = (*producer_weights, "down_weight", *producer_biases, "down_bias")
    return tuple(role for role in ordered if role in tensors)


def _validate_bundle(torch, bundle: Any, name: str) -> _ValidatedBundle:
    if not isinstance(bundle, MLPTensorBundle):
        raise TopologyValidationError(f"{name} must be an MLPTensorBundle.")
    topology = bundle.topology
    if bundle.down_weight is None:
        raise TopologyValidationError(f"{name}.down_weight is required.")
    if bundle.input_weight is None and bundle.input_bias is not None:
        raise TopologyValidationError(
            f"{name}.input_bias cannot exist without input_weight."
        )
    if bundle.gate_weight is None and bundle.gate_bias is not None:
        raise TopologyValidationError(
            f"{name}.gate_bias cannot exist without gate_weight."
        )
    if bundle.up_weight is None and bundle.up_bias is not None:
        raise TopologyValidationError(f"{name}.up_bias cannot exist without up_weight.")

    tensors: dict[str, Any] = {}
    for role in _ALL_ROLES:
        value = getattr(bundle, role)
        if value is None:
            continue
        dimensions = 2 if role.endswith("weight") else 1
        tensors[role] = _as_float64_tensor(
            torch,
            value,
            f"{name}.{role}",
            dimensions=dimensions,
        )

    if topology == "dense":
        producer = tensors["input_weight"]
        producer_roles = ("input_weight",)
        producer_bias_roles = ("input_bias",)
    else:
        gate = tensors["gate_weight"]
        up = tensors["up_weight"]
        if tuple(gate.shape) != tuple(up.shape):
            raise ShapeValidationError(
                f"{name}.gate_weight and {name}.up_weight must have equal shapes."
            )
        producer = gate
        producer_roles = ("gate_weight", "up_weight")
        producer_bias_roles = ("gate_bias", "up_bias")

    intermediate_width = int(producer.shape[0])
    input_width = int(producer.shape[1])
    down = tensors["down_weight"]
    output_width = int(down.shape[0])
    if int(down.shape[1]) != intermediate_width:
        raise ShapeValidationError(
            f"{name}.down_weight column count must equal intermediate width "
            f"{intermediate_width}."
        )
    for role in producer_roles:
        if tuple(tensors[role].shape) != (intermediate_width, input_width):
            raise ShapeValidationError(
                f"{name}.{role} must have shape ({intermediate_width}, {input_width})."
            )
    for role in producer_bias_roles:
        if role in tensors and tuple(tensors[role].shape) != (intermediate_width,):
            raise ShapeValidationError(
                f"{name}.{role} must have shape ({intermediate_width},)."
            )
    if "down_bias" in tensors and tuple(tensors["down_bias"].shape) != (output_width,):
        raise ShapeValidationError(
            f"{name}.down_bias must have shape ({output_width},)."
        )
    return _ValidatedBundle(
        topology=topology,
        tensors=tensors,
        intermediate_width=intermediate_width,
        input_width=input_width,
        output_width=output_width,
    )


def _bundle_from_tensors(
    topology: str,
    tensors: Mapping[str, Any],
) -> MLPTensorBundle:
    values = {role: tensors.get(role) for role in _ALL_ROLES}
    if topology == "dense":
        values["gate_weight"] = None
        values["up_weight"] = None
        values["gate_bias"] = None
        values["up_bias"] = None
    else:
        values["input_weight"] = None
        values["input_bias"] = None
    return MLPTensorBundle(**values)


def _canonical_bundle(validated: _ValidatedBundle) -> MLPTensorBundle:
    return _bundle_from_tensors(
        validated.topology,
        {role: tensor.detach().clone() for role, tensor in validated.tensors.items()},
    )


def _as_signature(torch, value: Any, name: str):
    result = _as_float64_tensor(torch, value, name, dimensions=2)
    if int(result.shape[0]) < 2:
        raise ShapeValidationError(f"{name} must contain at least two sample rows.")
    return result


def _standardize_signature(
    torch,
    value: Any,
    name: str,
    *,
    variance_floor: float,
):
    rows = _as_signature(torch, value, name)
    centered = rows - rows.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    if not bool(torch.isfinite(variance).all()):
        raise FiniteValidationError(f"{name} variance is non-finite.")
    degenerate = torch.nonzero(variance <= variance_floor, as_tuple=False).flatten()
    if int(degenerate.numel()) > 0:
        preview = degenerate[:8].tolist()
        raise ShapeValidationError(
            f"{name} has constant or near-constant neuron columns at indices "
            f"{preview} under variance_floor={variance_floor:.6g}."
        )
    standardized = centered / torch.sqrt(variance).unsqueeze(0)
    if not bool(torch.isfinite(standardized).all()):
        raise FiniteValidationError(f"{name} standardization is non-finite.")
    return standardized.detach().contiguous()


def _standardized_pair(
    torch,
    anchor_signatures: Any,
    source_signatures: Any,
    *,
    variance_floor: float,
):
    anchor = _standardize_signature(
        torch,
        anchor_signatures,
        "anchor_signatures",
        variance_floor=variance_floor,
    )
    source = _standardize_signature(
        torch,
        source_signatures,
        "source_signatures",
        variance_floor=variance_floor,
    )
    if tuple(anchor.shape) != tuple(source.shape):
        raise ShapeValidationError(
            "Anchor and source signatures must have the same sample count and width."
        )
    return anchor, source


def _correlation_block(torch, anchor: Any, source: Any, start: int, end: int):
    block = anchor[:, start:end].T @ source
    block = block / int(anchor.shape[0])
    if not bool(torch.isfinite(block).all()):
        raise FiniteValidationError("Standardized cross-correlation is non-finite.")
    return block.clamp(min=-1.0, max=1.0).contiguous()


def standardized_cross_correlation(
    anchor_signatures: Any,
    source_signatures: Any,
    *,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
):
    """Compute detached neuron-by-neuron Pearson cross-correlation.

    Rows are shared exemplar observations and columns are intermediate neurons.
    The result uses anchor neurons as rows and source neurons as columns.
    """

    torch = _require_torch()
    with torch.no_grad():
        chunk_size = _positive_int(chunk_size, "chunk_size")
        variance_floor = _nonnegative_float(variance_floor, "variance_floor")
        anchor, source = _standardized_pair(
            torch,
            anchor_signatures,
            source_signatures,
            variance_floor=variance_floor,
        )
        width = int(anchor.shape[1])
        result = torch.empty((width, width), dtype=torch.float64)
        for start in range(0, width, chunk_size):
            end = min(start + chunk_size, width)
            result[start:end] = _correlation_block(
                torch,
                anchor,
                source,
                start,
                end,
            )
        if not bool(torch.isfinite(result).all()):
            raise FiniteValidationError(
                "Standardized cross-correlation result is non-finite."
            )
        return result.detach().contiguous()


compute_standardized_cross_correlation = standardized_cross_correlation


def _as_correlation_matrix(torch, value: Any, name: str):
    matrix = _as_float64_tensor(torch, value, name, dimensions=2)
    if int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ShapeValidationError(f"{name} must be square for one-to-one assignment.")
    return matrix


def _validate_permutation(torch, permutation: Any, width: int, name: str):
    if isinstance(permutation, torch.Tensor):
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if permutation.layout != torch.strided or permutation.ndim != 1:
            raise IncompleteAssignmentError(f"{name} must be one-dimensional.")
        if permutation.dtype not in integer_dtypes:
            raise IncompleteAssignmentError(f"{name} must have an integer dtype.")
        result = permutation.detach().to(device="cpu", dtype=torch.long).contiguous()
    elif isinstance(permutation, Sequence) and not isinstance(
        permutation, (str, bytes)
    ):
        values = list(permutation)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise IncompleteAssignmentError(
                f"{name} must contain only integer neuron indices."
            )
        result = torch.tensor(values, dtype=torch.long)
    else:
        raise IncompleteAssignmentError(
            f"{name} must be a one-dimensional integer tensor or sequence."
        )
    if int(result.numel()) != width:
        raise IncompleteAssignmentError(
            f"{name} must contain exactly {width} assignments."
        )
    if bool(((result < 0) | (result >= width)).any()):
        raise IncompleteAssignmentError(
            f"{name} contains an out-of-range source-neuron index."
        )
    counts = torch.bincount(result, minlength=width)
    collisions = torch.nonzero(counts > 1, as_tuple=False).flatten()
    if int(collisions.numel()) > 0:
        raise AssignmentCollisionError(
            f"{name} contains source-neuron collisions at indices "
            f"{collisions[:8].tolist()}."
        )
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if int(missing.numel()) > 0:
        raise IncompleteAssignmentError(
            f"{name} omits source-neuron indices {missing[:8].tolist()}."
        )
    return result


def _hungarian_maximize(scores: Any) -> tuple[list[int], int, int]:
    """Exact deterministic square assignment with a structural width bound."""

    values = scores.tolist()
    width = len(values)
    maximum = max(max(row) for row in values)
    costs = [[maximum - score for score in row] for row in values]
    row_potential = [0.0] * (width + 1)
    column_potential = [0.0] * (width + 1)
    matched_row = [0] * (width + 1)
    predecessor = [0] * (width + 1)
    steps = 0
    step_limit = width * width

    for row in range(1, width + 1):
        matched_row[0] = row
        minimum = [math.inf] * (width + 1)
        used = [False] * (width + 1)
        column = 0
        while True:
            steps += 1
            if steps > step_limit:
                raise IncompleteAssignmentError(
                    "Bounded Hungarian assignment exceeded its structural step limit."
                )
            used[column] = True
            active_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, width + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            if not math.isfinite(delta) or next_column == 0:
                raise IncompleteAssignmentError(
                    "Hungarian assignment could not find a finite augmenting column."
                )
            for candidate in range(width + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = [-1] * width
    for column in range(1, width + 1):
        row = matched_row[column]
        if row <= 0 or assignment[row - 1] != -1:
            raise IncompleteAssignmentError(
                "Hungarian assignment produced an incomplete row mapping."
            )
        assignment[row - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise IncompleteAssignmentError(
            "Hungarian assignment left at least one anchor neuron unmatched."
        )
    return assignment, steps, step_limit


def _diagnostics(
    torch,
    *,
    solver: str,
    permutation: Any,
    assigned: Any,
    row_upper_values: Any,
    certified_optimal: bool,
    correlation_values_computed: int,
    correlation_chunks: int,
    chunk_size: int,
    assignment_steps: int,
    assignment_step_limit: int,
) -> AssignmentDiagnostics:
    width = int(permutation.numel())
    checked = _validate_permutation(torch, permutation, width, "permutation")
    if not torch.equal(checked, permutation):
        raise IncompleteAssignmentError(
            "Assignment changed while validating its integer permutation."
        )
    if tuple(assigned.shape) != (width,) or tuple(row_upper_values.shape) != (width,):
        raise IncompleteAssignmentError(
            "Assignment diagnostics require one score per anchor neuron."
        )
    if not bool(torch.isfinite(assigned).all()) or not bool(
        torch.isfinite(row_upper_values).all()
    ):
        raise FiniteValidationError("Assignment diagnostics contain non-finite scores.")
    objective = float(assigned.sum().item())
    relaxed_upper = float(row_upper_values.sum().item())
    relaxed_gap = max(0.0, relaxed_upper - objective)
    suboptimality_upper_bound = 0.0 if certified_optimal else relaxed_gap
    scalar_values = (
        objective,
        relaxed_upper,
        relaxed_gap,
        suboptimality_upper_bound,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise FiniteValidationError("Assignment objective diagnostics are non-finite.")
    return AssignmentDiagnostics(
        solver=solver,
        neuron_count=width,
        certified_optimal=certified_optimal,
        approximate=not certified_optimal,
        deterministic=True,
        tie_break=_TIE_BREAK,
        correlation_values_computed=correlation_values_computed,
        correlation_chunks=correlation_chunks,
        chunk_size=chunk_size,
        assignment_steps=assignment_steps,
        assignment_step_limit=assignment_step_limit,
        objective=objective,
        relaxed_row_max_upper_bound=relaxed_upper,
        relaxed_upper_bound_gap=relaxed_gap,
        suboptimality_upper_bound=suboptimality_upper_bound,
        mean_assigned_correlation=float(assigned.mean().item()),
        minimum_assigned_correlation=float(assigned.min().item()),
        maximum_assigned_correlation=float(assigned.max().item()),
        complete=True,
        collision_count=0,
        finite=True,
    )


def _greedy_from_blocks(
    torch,
    *,
    width: int,
    chunk_size: int,
    block_at: Any,
) -> tuple[Any, Any, Any, int]:
    available = torch.ones(width, dtype=torch.bool)
    permutation = torch.full((width,), -1, dtype=torch.long)
    assigned = torch.empty(width, dtype=torch.float64)
    row_upper = torch.empty(width, dtype=torch.float64)
    chunks = 0
    for start in range(0, width, chunk_size):
        end = min(start + chunk_size, width)
        block = block_at(start, end)
        if tuple(block.shape) != (end - start, width):
            raise IncompleteAssignmentError(
                "Correlation chunk has an unexpected shape during assignment."
            )
        if not bool(torch.isfinite(block).all()):
            raise FiniteValidationError(
                "Correlation chunk is non-finite during assignment."
            )
        row_upper[start:end] = block.max(dim=1).values
        chunks += 1
        for local_row in range(end - start):
            anchor_row = start + local_row
            if not bool(available.any()):
                raise IncompleteAssignmentError(
                    "Greedy assignment exhausted source neurons before completion."
                )
            masked = block[local_row].masked_fill(~available, -math.inf)
            source_column = int(torch.argmax(masked).item())
            score = masked[source_column]
            if not bool(torch.isfinite(score)):
                raise IncompleteAssignmentError(
                    "Greedy assignment could not find an available finite source."
                )
            permutation[anchor_row] = source_column
            assigned[anchor_row] = score
            available[source_column] = False
    if bool(available.any()) or bool((permutation < 0).any()):
        raise IncompleteAssignmentError(
            "Greedy assignment did not consume both neuron sets completely."
        )
    _validate_permutation(torch, permutation, width, "permutation")
    return permutation, assigned, row_upper, chunks


def solve_neuron_assignment(
    correlation_matrix: Any,
    *,
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
) -> NeuronAssignmentResult:
    """Maximize a square correlation assignment without SciPy.

    Widths at or below ``exact_assignment_limit`` use a deterministic exact
    Hungarian solver.  Larger widths use bounded row-order chunked greedy
    assignment and report a certified additive suboptimality upper bound from
    the sum of unconstrained row maxima.
    """

    torch = _require_torch()
    with torch.no_grad():
        exact_assignment_limit = _nonnegative_int(
            exact_assignment_limit,
            "exact_assignment_limit",
            maximum=MAX_EXACT_ASSIGNMENT_LIMIT,
        )
        chunk_size = _positive_int(chunk_size, "chunk_size")
        scores = _as_correlation_matrix(
            torch,
            correlation_matrix,
            "correlation_matrix",
        )
        width = int(scores.shape[0])
        row_upper = scores.max(dim=1).values
        if width <= exact_assignment_limit:
            values, steps, step_limit = _hungarian_maximize(scores)
            permutation = _validate_permutation(
                torch,
                values,
                width,
                "permutation",
            )
            rows = torch.arange(width, dtype=torch.long)
            assigned = scores[rows, permutation]
            diagnostics = _diagnostics(
                torch,
                solver="bounded_exact_hungarian",
                permutation=permutation,
                assigned=assigned,
                row_upper_values=row_upper,
                certified_optimal=True,
                correlation_values_computed=width * width,
                correlation_chunks=1,
                chunk_size=width,
                assignment_steps=steps,
                assignment_step_limit=step_limit,
            )
        else:
            permutation, assigned, row_upper, chunks = _greedy_from_blocks(
                torch,
                width=width,
                chunk_size=chunk_size,
                block_at=lambda start, end: scores[start:end],
            )
            diagnostics = _diagnostics(
                torch,
                solver="row_order_chunked_greedy",
                permutation=permutation,
                assigned=assigned,
                row_upper_values=row_upper,
                certified_optimal=False,
                correlation_values_computed=width * width,
                correlation_chunks=chunks,
                chunk_size=chunk_size,
                assignment_steps=width,
                assignment_step_limit=width,
            )
        return NeuronAssignmentResult(
            permutation=permutation.detach().contiguous(),
            assigned_correlations=assigned.detach().contiguous(),
            correlation_matrix=scores.detach().contiguous(),
            diagnostics=diagnostics,
        )


def _assignment_from_standardized(
    torch,
    anchor: Any,
    source: Any,
    *,
    exact_assignment_limit: int,
    chunk_size: int,
) -> NeuronAssignmentResult:
    width = int(anchor.shape[1])
    if width <= exact_assignment_limit:
        matrix = torch.empty((width, width), dtype=torch.float64)
        chunks = 0
        for start in range(0, width, chunk_size):
            end = min(start + chunk_size, width)
            matrix[start:end] = _correlation_block(
                torch,
                anchor,
                source,
                start,
                end,
            )
            chunks += 1
        values, steps, step_limit = _hungarian_maximize(matrix)
        permutation = _validate_permutation(torch, values, width, "permutation")
        rows = torch.arange(width, dtype=torch.long)
        assigned = matrix[rows, permutation]
        row_upper = matrix.max(dim=1).values
        diagnostics = _diagnostics(
            torch,
            solver="bounded_exact_hungarian",
            permutation=permutation,
            assigned=assigned,
            row_upper_values=row_upper,
            certified_optimal=True,
            correlation_values_computed=width * width,
            correlation_chunks=chunks,
            chunk_size=chunk_size,
            assignment_steps=steps,
            assignment_step_limit=step_limit,
        )
        return NeuronAssignmentResult(
            permutation=permutation.detach().contiguous(),
            assigned_correlations=assigned.detach().contiguous(),
            correlation_matrix=matrix.detach().contiguous(),
            diagnostics=diagnostics,
        )

    permutation, assigned, row_upper, chunks = _greedy_from_blocks(
        torch,
        width=width,
        chunk_size=chunk_size,
        block_at=lambda start, end: _correlation_block(
            torch,
            anchor,
            source,
            start,
            end,
        ),
    )
    diagnostics = _diagnostics(
        torch,
        solver="row_order_chunked_greedy",
        permutation=permutation,
        assigned=assigned,
        row_upper_values=row_upper,
        certified_optimal=False,
        correlation_values_computed=width * width,
        correlation_chunks=chunks,
        chunk_size=chunk_size,
        assignment_steps=width,
        assignment_step_limit=width,
    )
    return NeuronAssignmentResult(
        permutation=permutation.detach().contiguous(),
        assigned_correlations=assigned.detach().contiguous(),
        correlation_matrix=None,
        diagnostics=diagnostics,
    )


def derive_neuron_permutation(
    anchor_signatures: Any,
    source_signatures: Any,
    *,
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
) -> NeuronAssignmentResult:
    """Standardize signatures and derive a deterministic anchor-to-source map."""

    torch = _require_torch()
    with torch.no_grad():
        exact_assignment_limit = _nonnegative_int(
            exact_assignment_limit,
            "exact_assignment_limit",
            maximum=MAX_EXACT_ASSIGNMENT_LIMIT,
        )
        chunk_size = _positive_int(chunk_size, "chunk_size")
        variance_floor = _nonnegative_float(variance_floor, "variance_floor")
        anchor, source = _standardized_pair(
            torch,
            anchor_signatures,
            source_signatures,
            variance_floor=variance_floor,
        )
        return _assignment_from_standardized(
            torch,
            anchor,
            source,
            exact_assignment_limit=exact_assignment_limit,
            chunk_size=chunk_size,
        )


find_neuron_permutation = derive_neuron_permutation


def permute_mlp_bundle(
    bundle: MLPTensorBundle,
    permutation: Any,
) -> MLPTensorBundle:
    """Align producer rows/biases and down columns with one permutation."""

    torch = _require_torch()
    with torch.no_grad():
        validated = _validate_bundle(torch, bundle, "bundle")
        indices = _validate_permutation(
            torch,
            permutation,
            validated.intermediate_width,
            "permutation",
        )
        transformed: dict[str, Any] = {}
        for role, tensor in validated.tensors.items():
            if role == "down_weight":
                value = tensor.index_select(1, indices)
            elif role == "down_bias":
                value = tensor.clone()
            else:
                value = tensor.index_select(0, indices)
            if not bool(torch.isfinite(value).all()):
                raise FiniteValidationError(
                    f"Permuting {role} produced a non-finite tensor."
                )
            transformed[role] = value.detach().contiguous()
        return _bundle_from_tensors(validated.topology, transformed)


align_mlp_bundle = permute_mlp_bundle


def _normalized_merge_weights(torch, weights: Any, count: int):
    if weights is None:
        result = torch.ones(count, dtype=torch.float64)
    elif isinstance(weights, torch.Tensor):
        if weights.layout != torch.strided or weights.ndim != 1:
            raise ShapeValidationError("weights must be one-dimensional.")
        if not torch.is_floating_point(weights) or torch.is_complex(weights):
            raise ShapeValidationError("weights must have a real floating dtype.")
        result = weights.detach().to(device="cpu", dtype=torch.float64).contiguous()
    elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        values = list(weights)
        if any(isinstance(value, bool) for value in values):
            raise ShapeValidationError("weights cannot contain booleans.")
        try:
            result = torch.tensor(values, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ShapeValidationError("weights must contain real numbers.") from exc
    else:
        raise ShapeValidationError("weights must be a tensor or numeric sequence.")
    if tuple(result.shape) != (count,):
        raise ShapeValidationError(f"weights must have shape ({count},).")
    if not bool(torch.isfinite(result).all()):
        raise FiniteValidationError("weights contain NaN or infinity.")
    if bool((result < 0.0).any()):
        raise ShapeValidationError("weights must be non-negative.")
    total = float(result.sum().item())
    if not math.isfinite(total) or total <= 0.0:
        raise ShapeValidationError("weights must have a finite positive total.")
    normalized = result / total
    if not bool(torch.isfinite(normalized).all()):
        raise FiniteValidationError("Normalized merge weights are non-finite.")
    return normalized.detach().contiguous()


def _validated_bundle_sequence(torch, bundles: Any, name: str):
    if not isinstance(bundles, Sequence) or isinstance(bundles, (str, bytes)):
        raise TopologyValidationError(f"{name} must be a sequence of bundles.")
    items = tuple(bundles)
    if not items:
        raise TopologyValidationError(f"{name} cannot be empty.")
    validated = tuple(
        _validate_bundle(torch, bundle, f"{name}[{index}]")
        for index, bundle in enumerate(items)
    )
    anchor = validated[0]
    anchor_roles = _bundle_roles(anchor.topology, anchor.tensors)
    for index, current in enumerate(validated[1:], start=1):
        if current.topology != anchor.topology:
            raise TopologyValidationError(
                f"{name}[{index}] topology does not match {name}[0]."
            )
        if _bundle_roles(current.topology, current.tensors) != anchor_roles:
            raise TopologyValidationError(
                f"{name}[{index}] optional-bias topology does not match {name}[0]."
            )
        for role in anchor_roles:
            if tuple(current.tensors[role].shape) != tuple(anchor.tensors[role].shape):
                raise ShapeValidationError(
                    f"{name}[{index}].{role} shape does not match {name}[0].{role}."
                )
    return validated


def _merge_validated_bundles(
    torch,
    bundles: Sequence[_ValidatedBundle],
    weights: Any,
):
    normalized = _normalized_merge_weights(torch, weights, len(bundles))
    roles = _bundle_roles(bundles[0].topology, bundles[0].tensors)
    merged: dict[str, Any] = {}
    for role in roles:
        accumulator = torch.zeros_like(bundles[0].tensors[role])
        for weight, bundle in zip(normalized, bundles):
            accumulator.add_(bundle.tensors[role], alpha=float(weight.item()))
        if not bool(torch.isfinite(accumulator).all()):
            raise FiniteValidationError(
                f"Weighted merge produced non-finite values for {role}."
            )
        merged[role] = accumulator.detach().contiguous()
    return _bundle_from_tensors(bundles[0].topology, merged), normalized


def merge_aligned_mlp_bundles(
    bundles: Sequence[MLPTensorBundle],
    *,
    weights: Any = None,
) -> MLPTensorBundle:
    """Merge already-aligned, shape-identical bundles by weighted mean."""

    torch = _require_torch()
    with torch.no_grad():
        validated = _validated_bundle_sequence(torch, bundles, "bundles")
        merged, _ = _merge_validated_bundles(torch, validated, weights)
        return merged


weighted_merge_mlp_bundles = merge_aligned_mlp_bundles


def _planned_parameter_counts(
    bundle: _ValidatedBundle,
    execution_depth: int,
) -> ParameterCountReport:
    roles = _bundle_roles(bundle.topology, bundle.tensors)
    one_bundle = sum(int(bundle.tensors[role].numel()) for role in roles)
    dense_equivalent = one_bundle * execution_depth
    return ParameterCountReport(
        topology=bundle.topology,
        execution_depth=execution_depth,
        dense_equivalent_parameters=dense_equivalent,
        unique_parameters=one_bundle,
        parameters_saved=dense_equivalent - one_bundle,
        dense_equivalent_parameter_objects=len(roles) * execution_depth,
        unique_parameter_objects=len(roles),
        unique_parameter_sets=1,
        fully_shared=True,
        fully_expanded=execution_depth == 1,
    )


def align_and_merge_mlp_bundles(
    bundles: Sequence[MLPTensorBundle],
    activation_signatures: Sequence[Any],
    *,
    anchor_index: int = 0,
    weights: Any = None,
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
) -> MLPMergeResult:
    """Align same-width MLPs to one anchor and compute their weighted mean."""

    torch = _require_torch()
    with torch.no_grad():
        validated = _validated_bundle_sequence(torch, bundles, "bundles")
        depth = len(validated)
        if depth < 2:
            raise TopologyValidationError(
                "At least two MLP bundles are required for parameter sharing."
            )
        if not isinstance(activation_signatures, Sequence) or isinstance(
            activation_signatures, (str, bytes)
        ):
            raise ShapeValidationError(
                "activation_signatures must be a sequence with one tensor per layer."
            )
        signatures = tuple(activation_signatures)
        if len(signatures) != depth:
            raise ShapeValidationError(
                "activation_signatures count must equal the MLP bundle count."
            )
        if (
            isinstance(anchor_index, bool)
            or not isinstance(anchor_index, int)
            or anchor_index < 0
            or anchor_index >= depth
        ):
            raise ShapeValidationError(
                f"anchor_index must be an integer in [0, {depth - 1}]."
            )
        exact_assignment_limit = _nonnegative_int(
            exact_assignment_limit,
            "exact_assignment_limit",
            maximum=MAX_EXACT_ASSIGNMENT_LIMIT,
        )
        chunk_size = _positive_int(chunk_size, "chunk_size")
        variance_floor = _nonnegative_float(variance_floor, "variance_floor")

        standardized = tuple(
            _standardize_signature(
                torch,
                signature,
                f"activation_signatures[{index}]",
                variance_floor=variance_floor,
            )
            for index, signature in enumerate(signatures)
        )
        expected_shape = (
            int(standardized[anchor_index].shape[0]),
            validated[anchor_index].intermediate_width,
        )
        for index, signature in enumerate(standardized):
            if tuple(signature.shape) != expected_shape:
                raise ShapeValidationError(
                    f"activation_signatures[{index}] must have shape {expected_shape}."
                )

        assignments: list[NeuronAssignmentResult] = []
        aligned: list[MLPTensorBundle] = []
        permutations: list[Any] = []
        width = expected_shape[1]
        for index, (bundle, signature) in enumerate(zip(validated, standardized)):
            if index == anchor_index:
                permutation = torch.arange(width, dtype=torch.long)
                assigned = torch.ones(width, dtype=torch.float64)
                diagnostics = _diagnostics(
                    torch,
                    solver="anchor_identity",
                    permutation=permutation,
                    assigned=assigned,
                    row_upper_values=assigned,
                    certified_optimal=True,
                    correlation_values_computed=0,
                    correlation_chunks=0,
                    chunk_size=chunk_size,
                    assignment_steps=0,
                    assignment_step_limit=0,
                )
                assignment = NeuronAssignmentResult(
                    permutation=permutation,
                    assigned_correlations=assigned,
                    correlation_matrix=None,
                    diagnostics=diagnostics,
                )
            else:
                assignment = _assignment_from_standardized(
                    torch,
                    standardized[anchor_index],
                    signature,
                    exact_assignment_limit=exact_assignment_limit,
                    chunk_size=chunk_size,
                )
            canonical = _canonical_bundle(bundle)
            aligned_bundle = permute_mlp_bundle(
                canonical,
                assignment.permutation,
            )
            assignments.append(assignment)
            permutations.append(assignment.permutation.detach().clone())
            aligned.append(aligned_bundle)

        aligned_validated = _validated_bundle_sequence(
            torch,
            aligned,
            "aligned_bundles",
        )
        merged, normalized = _merge_validated_bundles(
            torch,
            aligned_validated,
            weights,
        )
        merged_validated = _validate_bundle(torch, merged, "merged_bundle")
        return MLPMergeResult(
            merged_bundle=merged,
            aligned_bundles=tuple(aligned),
            assignments=tuple(assignments),
            permutations=tuple(permutations),
            normalized_weights=normalized,
            anchor_index=anchor_index,
            parameter_counts=_planned_parameter_counts(merged_validated, depth),
        )


activation_align_and_merge_mlp_bundles = align_and_merge_mlp_bundles


def _coerce_layout(layout: Any) -> MLPModuleLayout:
    if isinstance(layout, MLPModuleLayout):
        result = layout
    elif isinstance(layout, Mapping):
        unknown = set(layout) - set(_ALL_ROLES)
        if unknown:
            raise TopologyValidationError(
                f"Unknown module-layout roles: {sorted(unknown)}."
            )
        try:
            result = MLPModuleLayout(**dict(layout))
        except TypeError as exc:
            raise TopologyValidationError("Invalid module-layout mapping.") from exc
    else:
        raise TopologyValidationError(
            "layout must be an MLPModuleLayout or role-to-path mapping."
        )
    topology = result.topology
    paths: dict[str, str] = {}
    for role in _ALL_ROLES:
        path = getattr(result, role)
        if path is None:
            continue
        if (
            not isinstance(path, str)
            or not path
            or path.startswith(".")
            or path.endswith(".")
        ):
            raise TopologyValidationError(
                f"layout.{role} must be a non-empty dotted parameter path."
            )
        if path in paths:
            raise TopologyValidationError(
                f"layout.{role} collides with layout.{paths[path]} at {path!r}."
            )
        paths[path] = role
    if topology == "dense" and (
        result.gate_bias is not None or result.up_bias is not None
    ):
        raise TopologyValidationError("Dense layouts cannot include gate/up biases.")
    if topology == "gated" and result.input_bias is not None:
        raise TopologyValidationError("Gated layouts cannot include input_bias.")
    return result


def _layout_roles(layout: MLPModuleLayout) -> tuple[str, ...]:
    values = {
        role: getattr(layout, role)
        for role in _ALL_ROLES
        if getattr(layout, role) is not None
    }
    return _bundle_roles(layout.topology, values)


def _resolve_parameter(torch, module: Any, path: str, label: str) -> _ParameterBinding:
    if not isinstance(module, torch.nn.Module):
        raise ModuleSharingError(f"{label} must be a torch.nn.Module.")
    parent_path, separator, name = path.rpartition(".")
    owner = module
    if separator:
        try:
            owner = module.get_submodule(parent_path)
        except (AttributeError, KeyError) as exc:
            raise ModuleSharingError(
                f"{label} has no submodule for parameter path {path!r}."
            ) from exc
    if not isinstance(owner, torch.nn.Module):
        raise ModuleSharingError(f"Owner of {label}.{path} is not a torch.nn.Module.")
    if name not in owner._parameters or owner._parameters[name] is None:
        raise ModuleSharingError(f"{label}.{path} is not a registered Parameter.")
    parameter = owner._parameters[name]
    if not isinstance(parameter, torch.nn.Parameter):
        raise ModuleSharingError(f"{label}.{path} is not a torch.nn.Parameter.")
    if parameter.layout != torch.strided or parameter.ndim not in (1, 2):
        raise ShapeValidationError(
            f"{label}.{path} must be a dense rank-1 or rank-2 Parameter."
        )
    if any(int(size) <= 0 for size in parameter.shape):
        raise ShapeValidationError(f"{label}.{path} cannot have an empty dimension.")
    if not torch.is_floating_point(parameter) or torch.is_complex(parameter):
        raise ShapeValidationError(
            f"{label}.{path} must have a real floating-point dtype."
        )
    if not bool(torch.isfinite(parameter.detach()).all()):
        raise FiniteValidationError(f"{label}.{path} contains NaN or infinity.")
    return _ParameterBinding(owner=owner, name=name, parameter=parameter)


def _module_sequence(modules: Any) -> tuple[Any, ...]:
    if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)):
        raise ModuleSharingError(
            "modules must be a sequence of torch.nn.Module values."
        )
    items = tuple(modules)
    if not items:
        raise ModuleSharingError("modules cannot be empty.")
    identities = [id(module) for module in items]
    if len(set(identities)) != len(identities):
        raise ModuleSharingError(
            "modules contains the same module object more than once."
        )
    return items


def _collect_bindings(
    torch,
    modules: Any,
    layout: Any,
    *,
    reject_gradients: bool,
):
    items = _module_sequence(modules)
    resolved_layout = _coerce_layout(layout)
    roles = _layout_roles(resolved_layout)
    bindings: list[dict[str, _ParameterBinding]] = []
    object_roles: dict[int, str] = {}
    storage_objects: dict[tuple[str, int], int] = {}
    for module_index, module in enumerate(items):
        current: dict[str, _ParameterBinding] = {}
        for role in roles:
            path = getattr(resolved_layout, role)
            binding = _resolve_parameter(
                torch,
                module,
                path,
                f"modules[{module_index}]",
            )
            parameter_id = id(binding.parameter)
            prior_role = object_roles.get(parameter_id)
            if prior_role is not None and prior_role != role:
                raise ModuleSharingError(
                    f"Parameter-object collision between roles {prior_role} and {role}."
                )
            object_roles[parameter_id] = role
            storage_key = (
                str(binding.parameter.device),
                int(binding.parameter.untyped_storage().data_ptr()),
            )
            prior_object = storage_objects.get(storage_key)
            if prior_object is not None and prior_object != parameter_id:
                raise ModuleSharingError(
                    "Distinct Parameter objects unexpectedly share tensor storage."
                )
            storage_objects[storage_key] = parameter_id
            if reject_gradients and binding.parameter.grad is not None:
                raise ModuleSharingError(
                    f"modules[{module_index}].{path} has an existing gradient."
                )
            current[role] = binding
        bindings.append(current)

    anchor = bindings[0]
    for module_index, current in enumerate(bindings[1:], start=1):
        for role in roles:
            left = anchor[role].parameter
            right = current[role].parameter
            if tuple(left.shape) != tuple(right.shape):
                raise ShapeValidationError(
                    f"modules[{module_index}] {role} shape differs from modules[0]."
                )
            if left.dtype != right.dtype or left.device != right.device:
                raise ModuleSharingError(
                    f"All {role} parameters must have the same dtype and device."
                )
            if left.requires_grad != right.requires_grad:
                raise ModuleSharingError(
                    f"All {role} parameters must agree on requires_grad."
                )
    return items, resolved_layout, roles, tuple(bindings)


def _sharing_report_from_bindings(
    layout: MLPModuleLayout,
    roles: Sequence[str],
    bindings: Sequence[Mapping[str, _ParameterBinding]],
) -> ParameterCountReport:
    dense_equivalent = sum(
        int(binding[role].parameter.numel()) for binding in bindings for role in roles
    )
    unique: dict[int, Any] = {}
    for binding in bindings:
        for role in roles:
            parameter = binding[role].parameter
            unique[id(parameter)] = parameter
    unique_parameters = sum(int(parameter.numel()) for parameter in unique.values())
    sets = {
        tuple(id(binding[role].parameter) for role in roles) for binding in bindings
    }
    role_unique_counts = [
        len({id(binding[role].parameter) for binding in bindings}) for role in roles
    ]
    depth = len(bindings)
    return ParameterCountReport(
        topology=layout.topology,
        execution_depth=depth,
        dense_equivalent_parameters=dense_equivalent,
        unique_parameters=unique_parameters,
        parameters_saved=dense_equivalent - unique_parameters,
        dense_equivalent_parameter_objects=len(roles) * depth,
        unique_parameter_objects=len(unique),
        unique_parameter_sets=len(sets),
        fully_shared=all(count == 1 for count in role_unique_counts),
        fully_expanded=all(count == depth for count in role_unique_counts),
    )


def parameter_sharing_report(
    modules: Sequence[Any],
    layout: MLPModuleLayout | Mapping[str, str | None],
) -> ParameterCountReport:
    """Inspect logical and unique MLP parameters without changing modules."""

    torch = _require_torch()
    with torch.no_grad():
        _, resolved_layout, roles, bindings = _collect_bindings(
            torch,
            modules,
            layout,
            reject_gradients=False,
        )
        return _sharing_report_from_bindings(resolved_layout, roles, bindings)


report_parameter_sharing = parameter_sharing_report


def extract_mlp_bundle(
    module: Any,
    layout: MLPModuleLayout | Mapping[str, str | None],
) -> MLPTensorBundle:
    """Extract one module's layout as a detached CPU float64 tensor bundle."""

    torch = _require_torch()
    with torch.no_grad():
        _, resolved_layout, roles, bindings = _collect_bindings(
            torch,
            (module,),
            layout,
            reject_gradients=False,
        )
        tensors = {role: bindings[0][role].parameter for role in roles}
        bundle = _bundle_from_tensors(resolved_layout.topology, tensors)
        return _canonical_bundle(_validate_bundle(torch, bundle, "module_bundle"))


def _replace_parameters(replacements: Sequence[tuple[_ParameterBinding, Any]]) -> None:
    changed: list[tuple[_ParameterBinding, Any]] = []
    try:
        for binding, replacement in replacements:
            old = binding.owner._parameters[binding.name]
            binding.owner._parameters[binding.name] = replacement
            changed.append((binding, old))
    except Exception:
        for binding, old in reversed(changed):
            binding.owner._parameters[binding.name] = old
        raise


def _restore_parameters(replacements: Sequence[tuple[_ParameterBinding, Any]]) -> None:
    for binding, old in reversed(replacements):
        binding.owner._parameters[binding.name] = old


def tie_mlp_modules(
    modules: Sequence[Any],
    shared_bundle: MLPTensorBundle,
    layout: MLPModuleLayout | Mapping[str, str | None],
) -> ParameterCountReport:
    """Tie all laid-out module parameters to one fresh shared Parameter per role."""

    torch = _require_torch()
    with torch.no_grad():
        validated_bundle = _validate_bundle(torch, shared_bundle, "shared_bundle")
        items, resolved_layout, roles, bindings = _collect_bindings(
            torch,
            modules,
            layout,
            reject_gradients=True,
        )
        if len(items) < 2:
            raise ModuleSharingError("At least two modules are required for tying.")
        bundle_roles = _bundle_roles(
            validated_bundle.topology,
            validated_bundle.tensors,
        )
        if resolved_layout.topology != validated_bundle.topology:
            raise TopologyValidationError(
                "Module layout topology does not match shared_bundle topology."
            )
        if roles != bundle_roles:
            raise TopologyValidationError(
                "Module layout roles, including optional biases, must exactly match "
                "shared_bundle."
            )
        depth = len(bindings)
        for role in roles:
            unique_count = len({id(binding[role].parameter) for binding in bindings})
            if unique_count not in (1, depth):
                raise IncompleteSharingError(
                    f"Role {role} has partial pre-existing sharing across modules."
                )
            expected_shape = tuple(validated_bundle.tensors[role].shape)
            actual_shape = tuple(bindings[0][role].parameter.shape)
            if actual_shape != expected_shape:
                raise ShapeValidationError(
                    f"Module role {role} has shape {actual_shape}, expected "
                    f"{expected_shape} from shared_bundle."
                )

        shared_parameters: dict[str, Any] = {}
        for role in roles:
            anchor_parameter = bindings[0][role].parameter
            value = validated_bundle.tensors[role].to(
                device=anchor_parameter.device,
                dtype=anchor_parameter.dtype,
            )
            if not bool(torch.isfinite(value).all()):
                raise FiniteValidationError(
                    f"Converting shared_bundle.{role} to the module dtype is non-finite."
                )
            shared_parameters[role] = torch.nn.Parameter(
                value.detach().clone(),
                requires_grad=anchor_parameter.requires_grad,
            )

        replacements = [
            (binding[role], shared_parameters[role])
            for binding in bindings
            for role in roles
        ]
        rollback = [
            (binding, binding.owner._parameters[binding.name])
            for binding, _ in replacements
        ]
        try:
            _replace_parameters(replacements)
            report = parameter_sharing_report(items, resolved_layout)
            if not report.fully_shared:
                raise IncompleteSharingError(
                    "Parameter identity verification failed after tying."
                )
            expected_unique = sum(
                int(validated_bundle.tensors[role].numel()) for role in roles
            )
            if report.unique_parameters != expected_unique:
                raise ModuleSharingError(
                    "Unique parameter count differs from the shared bundle count."
                )
            return report
        except Exception:
            _restore_parameters(rollback)
            raise


def expand_mlp_modules(
    modules: Sequence[Any],
    layout: MLPModuleLayout | Mapping[str, str | None],
) -> ParameterCountReport:
    """Expand a fully tied MLP sequence into independent equal Parameter clones."""

    torch = _require_torch()
    with torch.no_grad():
        items, resolved_layout, roles, bindings = _collect_bindings(
            torch,
            modules,
            layout,
            reject_gradients=True,
        )
        if len(items) < 2:
            raise ModuleSharingError("At least two modules are required for expansion.")
        for role in roles:
            identities = {id(binding[role].parameter) for binding in bindings}
            if len(identities) != 1:
                raise IncompleteSharingError(
                    f"Role {role} must be fully shared before expansion."
                )

        replacements: list[tuple[_ParameterBinding, Any]] = []
        for binding in bindings:
            for role in roles:
                source = binding[role].parameter
                clone = torch.nn.Parameter(
                    source.detach().clone(),
                    requires_grad=source.requires_grad,
                )
                replacements.append((binding[role], clone))
        rollback = [
            (binding, binding.owner._parameters[binding.name])
            for binding, _ in replacements
        ]
        try:
            _replace_parameters(replacements)
            report = parameter_sharing_report(items, resolved_layout)
            if not report.fully_expanded:
                raise IncompleteSharingError(
                    "Parameter identity verification failed after expansion."
                )
            if report.unique_parameters != report.dense_equivalent_parameters:
                raise ModuleSharingError(
                    "Expanded unique count does not equal dense-equivalent count."
                )
            return report
        except Exception:
            _restore_parameters(rollback)
            raise


untie_mlp_modules = expand_mlp_modules
expand_untie_mlp_modules = expand_mlp_modules


__all__ = [
    "AssignmentCollisionError",
    "AssignmentDiagnostics",
    "DEFAULT_CORRELATION_CHUNK_SIZE",
    "DEFAULT_EXACT_ASSIGNMENT_LIMIT",
    "FiniteValidationError",
    "IncompleteAssignmentError",
    "IncompleteSharingError",
    "MAX_EXACT_ASSIGNMENT_LIMIT",
    "MLPMergeResult",
    "MLPModuleLayout",
    "MLPTensorBundle",
    "ModuleSharingError",
    "NeuronAssignmentResult",
    "ParameterCountReport",
    "ParameterSharingError",
    "ShapeValidationError",
    "TopologyValidationError",
    "activation_align_and_merge_mlp_bundles",
    "align_and_merge_mlp_bundles",
    "align_mlp_bundle",
    "compute_standardized_cross_correlation",
    "derive_neuron_permutation",
    "expand_mlp_modules",
    "expand_untie_mlp_modules",
    "extract_mlp_bundle",
    "find_neuron_permutation",
    "merge_aligned_mlp_bundles",
    "parameter_sharing_report",
    "permute_mlp_bundle",
    "report_parameter_sharing",
    "solve_neuron_assignment",
    "standardized_cross_correlation",
    "tie_mlp_modules",
    "untie_mlp_modules",
    "weighted_merge_mlp_bundles",
]
