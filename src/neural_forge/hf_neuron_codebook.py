"""Topology-preserving global codebooks for gated MLP neurons.

The primitive in this module compresses storage, not execution topology.  Every
logical layer still executes its full intermediate width.  A logical neuron
selects one global real-neuron prototype, and the same selection is used for
the gate row, up row, and corresponding down-projection column.

Fitting is deterministic, runs without autograd, and keeps only bounded
feature and distance blocks.  Runtime parameters live in one registered
``NeuronCodebookRegistry``; layer views have no parameters or buffers and keep
a normal strong Python reference to that registry.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_WEIGHT_SKETCH_SIZE = 64
DEFAULT_ACTIVATION_SKETCH_SIZE = 64
DEFAULT_POINT_CHUNK_SIZE = 1024
DEFAULT_CENTER_CHUNK_SIZE = 64
DEFAULT_MEDOID_CANDIDATE_LIMIT = 64
DEFAULT_MEDOID_ITERATIONS = 2

_ACTIVATION_CODES = {
    "identity": 0,
    "silu": 1,
    "gelu": 2,
    "gelu_tanh": 3,
    "relu": 4,
}
_ACTIVATION_NAMES = {code: name for name, code in _ACTIVATION_CODES.items()}
_TIE_BREAK = "lowest global neuron index"
_GLOBAL_MEDOIDS_STRATEGY = "global_medoids"
_ALIGNED_INDEX_STRATEGY = "aligned_index"
_ASSIGNMENT_STRATEGIES = (
    _GLOBAL_MEDOIDS_STRATEGY,
    _ALIGNED_INDEX_STRATEGY,
)


class NeuronCodebookError(ValueError):
    """Base class for neuron-codebook validation and fitting failures."""


class NeuronCodebookShapeError(NeuronCodebookError):
    """A source bundle, activation sketch, or runtime input has bad shape."""


class NeuronCodebookFiniteError(NeuronCodebookError):
    """A source, sketch, scale, or runtime parameter is non-finite."""


class NeuronCodebookStateError(NeuronCodebookError):
    """A tensor-only state dictionary cannot reconstruct a registry."""


@dataclass(frozen=True, slots=True)
class GatedMLPBundle:
    """One gated MLP in conventional PyTorch linear orientation.

    Producer weights have shape ``[intermediate, input]``.  The down weight has
    shape ``[output, intermediate]``.  Gate, up, and down biases are optional,
    but their presence must be consistent across all fitted layers.
    """

    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    gate_bias: Tensor | None = None
    up_bias: Tensor | None = None
    down_bias: Tensor | None = None

    @classmethod
    def from_linears(
        cls,
        gate: nn.Linear,
        up: nn.Linear,
        down: nn.Linear,
    ) -> GatedMLPBundle:
        """Create a non-copying tensor bundle from three linear projections."""

        if not all(isinstance(module, nn.Linear) for module in (gate, up, down)):
            raise NeuronCodebookShapeError(
                "gate, up, and down must all be torch.nn.Linear modules."
            )
        return cls(
            gate_weight=gate.weight,
            up_weight=up.weight,
            down_weight=down.weight,
            gate_bias=gate.bias,
            up_bias=up.bias,
            down_bias=down.bias,
        )

    def forward(self, inputs: Tensor, *, activation: str = "silu") -> Tensor:
        """Execute this bundle with a named, state-serializable activation."""

        gate = F.linear(inputs, self.gate_weight, self.gate_bias)
        up = F.linear(inputs, self.up_weight, self.up_bias)
        hidden = _apply_activation(gate, _normalize_activation(activation)) * up
        return F.linear(hidden, self.down_weight, self.down_bias)


@dataclass(frozen=True, slots=True)
class NeuronCodebookParameterReport:
    """Logical dense execution and unique learned storage accounting."""

    logical_layers: int
    input_width: int
    intermediate_width: int
    output_width: int
    prototype_count: int
    logical_neuron_executions: int
    logical_dense_parameters: int
    unique_learned_parameters: int
    prototype_parameters: int
    scale_parameters: int
    retained_layer_bias_parameters: int
    parameters_saved: int
    assignment_index_bytes: int
    prototype_source_index_bytes: int
    index_buffer_bytes: int

    @property
    def logical_dense_parameter_count(self) -> int:
        return self.logical_dense_parameters

    @property
    def logical_parameter_count(self) -> int:
        return self.logical_dense_parameters

    @property
    def unique_learned_parameter_count(self) -> int:
        return self.unique_learned_parameters

    @property
    def unique_parameter_count(self) -> int:
        return self.unique_learned_parameters

    @property
    def index_bytes(self) -> int:
        return self.index_buffer_bytes

    @property
    def compression_ratio(self) -> float:
        """Legacy child-to-dense parameter fraction."""

        return self.unique_learned_parameters / self.logical_dense_parameters

    @property
    def child_fraction_of_dense(self) -> float:
        return self.compression_ratio

    @property
    def source_to_child_ratio(self) -> float:
        return self.logical_dense_parameters / self.unique_learned_parameters

    def as_dict(self) -> dict[str, int | float]:
        return {
            "logical_layers": self.logical_layers,
            "input_width": self.input_width,
            "intermediate_width": self.intermediate_width,
            "output_width": self.output_width,
            "prototype_count": self.prototype_count,
            "logical_neuron_executions": self.logical_neuron_executions,
            "logical_dense_parameters": self.logical_dense_parameters,
            "logical_dense_parameter_count": self.logical_dense_parameters,
            "unique_learned_parameters": self.unique_learned_parameters,
            "unique_learned_parameter_count": self.unique_learned_parameters,
            "prototype_parameters": self.prototype_parameters,
            "scale_parameters": self.scale_parameters,
            "retained_layer_bias_parameters": self.retained_layer_bias_parameters,
            "parameters_saved": self.parameters_saved,
            "assignment_index_bytes": self.assignment_index_bytes,
            "prototype_source_index_bytes": self.prototype_source_index_bytes,
            "index_buffer_bytes": self.index_buffer_bytes,
            "compression_ratio": self.compression_ratio,
            "child_fraction_of_dense": self.child_fraction_of_dense,
            "source_to_child_ratio": self.source_to_child_ratio,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class NeuronCodebookDiagnostics:
    """Determinism, real-medoid, sketch, and bounded-work evidence."""

    algorithm: str
    assignment_strategy: str
    logical_layers: int
    intermediate_width: int
    logical_neurons: int
    prototype_count: int
    deterministic: bool
    tie_break: str
    real_medoid_prototypes: bool
    used_activation_sketches: bool
    requested_weight_sketch_size: int
    weight_sketch_dimensions: int
    requested_activation_sketch_size: int
    activation_sketch_dimensions: int
    combined_sketch_dimensions: int
    sketch_elements: int
    point_chunk_size: int
    center_chunk_size: int
    medoid_candidate_limit: int
    max_iterations: int
    iterations_run: int
    distance_values_computed: int
    exact_distance_work_bound: int
    exact_distance_coordinate_work_bound: int
    distance_work_bound_formula: str
    max_distance_block_elements: int
    distance_block_element_limit: int
    full_pairwise_matrix_elements: int
    materialized_full_pairwise_matrix: bool
    used_n_by_k_assignment_search: bool
    permutations_provided: bool
    permutation_proof: bool
    permutation_proof_method: str
    permutation_rows_verified: int
    aligned_group_size: int
    aligned_subgroup_factor: int
    bounded_work: bool
    fitting_requires_grad: bool
    squared_sketch_error: float
    maximum_assigned_squared_distance: float

    @property
    def full_pairwise_matrix_materialized(self) -> bool:
        return self.materialized_full_pairwise_matrix

    @property
    def no_full_pairwise_matrix(self) -> bool:
        return not self.materialized_full_pairwise_matrix

    @property
    def strategy(self) -> str:
        return self.assignment_strategy

    @property
    def permutations_bijective(self) -> bool:
        return self.permutation_proof

    @property
    def distance_work_bound(self) -> int:
        return self.exact_distance_work_bound

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(slots=True)
class _WorkTracker:
    distance_values: int = 0
    maximum_block_elements: int = 0

    def add(self, rows: int, columns: int) -> None:
        elements = rows * columns
        self.distance_values += elements
        self.maximum_block_elements = max(self.maximum_block_elements, elements)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NeuronCodebookShapeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NeuronCodebookShapeError(f"{name} must be a non-negative integer.")
    return value


def _normalize_assignment_strategy(value: str) -> str:
    if not isinstance(value, str):
        raise NeuronCodebookShapeError("assignment_strategy must be a string.")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in _ASSIGNMENT_STRATEGIES:
        raise NeuronCodebookShapeError(
            f"assignment_strategy must be one of {_ASSIGNMENT_STRATEGIES!r}."
        )
    return normalized


def _normalize_activation(activation: str) -> str:
    if not isinstance(activation, str):
        raise NeuronCodebookShapeError(
            "activation must be a named activation, not a callable."
        )
    normalized = activation.strip().lower().replace("-", "_")
    aliases = {
        "linear": "identity",
        "swish": "silu",
        "gelu_new": "gelu_tanh",
        "gelu_pytorch_tanh": "gelu_tanh",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _ACTIVATION_CODES:
        raise NeuronCodebookShapeError(
            f"activation must be one of {tuple(_ACTIVATION_CODES)!r}."
        )
    return normalized


def _apply_activation(inputs: Tensor, activation: str) -> Tensor:
    if activation == "identity":
        return inputs
    if activation == "silu":
        return F.silu(inputs)
    if activation == "gelu":
        return F.gelu(inputs, approximate="none")
    if activation == "gelu_tanh":
        return F.gelu(inputs, approximate="tanh")
    if activation == "relu":
        return F.relu(inputs)
    raise NeuronCodebookStateError(f"Unknown stored activation {activation!r}.")


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _projection_parts(value: Any, name: str) -> tuple[Tensor, Tensor | None]:
    if isinstance(value, nn.Linear):
        return value.weight, value.bias
    weight = getattr(value, "weight", None)
    bias = getattr(value, "bias", None)
    if isinstance(weight, Tensor):
        return weight, bias
    raise NeuronCodebookShapeError(
        f"{name} must be an nn.Linear or expose tensor weight/bias attributes."
    )


def _coerce_bundle(value: Any, layer_index: int) -> GatedMLPBundle:
    if isinstance(value, GatedMLPBundle):
        return value

    direct_names = ("gate_weight", "up_weight", "down_weight")
    if isinstance(value, Mapping) and all(name in value for name in direct_names):
        return GatedMLPBundle(
            gate_weight=value["gate_weight"],
            up_weight=value["up_weight"],
            down_weight=value["down_weight"],
            gate_bias=value.get("gate_bias"),
            up_bias=value.get("up_bias"),
            down_bias=value.get("down_bias"),
        )
    if all(hasattr(value, name) for name in direct_names):
        return GatedMLPBundle(
            gate_weight=value.gate_weight,
            up_weight=value.up_weight,
            down_weight=value.down_weight,
            gate_bias=getattr(value, "gate_bias", None),
            up_bias=getattr(value, "up_bias", None),
            down_bias=getattr(value, "down_bias", None),
        )

    projections: tuple[Any, Any, Any] | None = None
    if isinstance(value, Mapping):
        gate = _mapping_value(value, "gate_proj", "gate")
        up = _mapping_value(value, "up_proj", "up")
        down = _mapping_value(value, "down_proj", "down")
        if gate is not None and up is not None and down is not None:
            projections = (gate, up, down)
    elif all(hasattr(value, name) for name in ("gate_proj", "up_proj", "down_proj")):
        projections = (value.gate_proj, value.up_proj, value.down_proj)
    elif all(hasattr(value, name) for name in ("gate", "up", "down")):
        projections = (value.gate, value.up, value.down)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 3:
            projections = (value[0], value[1], value[2])

    if projections is None:
        raise NeuronCodebookShapeError(
            f"bundles[{layer_index}] is not a recognized gated MLP bundle."
        )
    gate_weight, gate_bias = _projection_parts(
        projections[0], f"bundles[{layer_index}].gate"
    )
    up_weight, up_bias = _projection_parts(projections[1], f"bundles[{layer_index}].up")
    down_weight, down_bias = _projection_parts(
        projections[2], f"bundles[{layer_index}].down"
    )
    return GatedMLPBundle(
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        gate_bias=gate_bias,
        up_bias=up_bias,
        down_bias=down_bias,
    )


def _validate_tensor(value: Any, name: str, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise NeuronCodebookShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise NeuronCodebookShapeError(f"{name} must be a dense non-meta tensor.")
    if value.ndim != dimensions or any(int(size) <= 0 for size in value.shape):
        raise NeuronCodebookShapeError(
            f"{name} must be a non-empty {dimensions}-dimensional tensor."
        )
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise NeuronCodebookShapeError(f"{name} must have a real floating-point dtype.")
    if not bool(torch.isfinite(value.detach()).all()):
        raise NeuronCodebookFiniteError(f"{name} contains NaN or infinity.")
    return value


def _validated_bundles(values: Any) -> tuple[GatedMLPBundle, ...]:
    if isinstance(values, (str, bytes, Mapping, Tensor)):
        raise NeuronCodebookShapeError("bundles must be an iterable across depth.")
    try:
        source_values = tuple(values)
    except TypeError as exc:
        raise NeuronCodebookShapeError(
            "bundles must be an iterable across depth."
        ) from exc
    if len(source_values) == 0:
        raise NeuronCodebookShapeError("bundles must contain at least one layer.")

    bundles = tuple(
        _coerce_bundle(value, index) for index, value in enumerate(source_values)
    )
    reference: tuple[Any, ...] | None = None
    for index, bundle in enumerate(bundles):
        gate = _validate_tensor(bundle.gate_weight, f"bundles[{index}].gate_weight", 2)
        up = _validate_tensor(bundle.up_weight, f"bundles[{index}].up_weight", 2)
        down = _validate_tensor(bundle.down_weight, f"bundles[{index}].down_weight", 2)
        if tuple(gate.shape) != tuple(up.shape):
            raise NeuronCodebookShapeError(
                f"bundles[{index}] gate and up weights must have equal shape."
            )
        intermediate, input_width = map(int, gate.shape)
        output_width = int(down.shape[0])
        if int(down.shape[1]) != intermediate:
            raise NeuronCodebookShapeError(
                f"bundles[{index}] down columns must equal intermediate width."
            )
        for role, bias, width in (
            ("gate_bias", bundle.gate_bias, intermediate),
            ("up_bias", bundle.up_bias, intermediate),
            ("down_bias", bundle.down_bias, output_width),
        ):
            if bias is not None:
                checked = _validate_tensor(bias, f"bundles[{index}].{role}", 1)
                if tuple(checked.shape) != (width,):
                    raise NeuronCodebookShapeError(
                        f"bundles[{index}].{role} must have shape ({width},)."
                    )
        tensor_values = tuple(
            tensor
            for tensor in (
                gate,
                up,
                down,
                bundle.gate_bias,
                bundle.up_bias,
                bundle.down_bias,
            )
            if tensor is not None
        )
        if any(tensor.dtype != gate.dtype for tensor in tensor_values):
            raise NeuronCodebookShapeError(
                f"bundles[{index}] tensors must have one common dtype."
            )
        if any(tensor.device != gate.device for tensor in tensor_values):
            raise NeuronCodebookShapeError(
                f"bundles[{index}] tensors must have one common device."
            )
        topology = (
            intermediate,
            input_width,
            output_width,
            bundle.gate_bias is not None,
            bundle.up_bias is not None,
            bundle.down_bias is not None,
            gate.dtype,
            gate.device,
        )
        if reference is None:
            reference = topology
        elif topology != reference:
            raise NeuronCodebookShapeError(
                "All gated MLP bundles must have the same shape, bias topology, "
                "dtype, and device across depth."
            )
    return bundles


def _permutation_row(value: Any, layer_index: int, intermediate: int) -> Tensor:
    name = f"neuron_permutations[{layer_index}]"
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if isinstance(value, Tensor):
        if value.layout != torch.strided or value.device.type == "meta":
            raise NeuronCodebookShapeError(f"{name} must be a dense non-meta tensor.")
        if value.ndim != 1 or tuple(value.shape) != (intermediate,):
            raise NeuronCodebookShapeError(f"{name} must have shape ({intermediate},).")
        if value.dtype not in integer_dtypes:
            raise NeuronCodebookShapeError(f"{name} must use an integer dtype.")
        return value.detach().to("cpu", torch.int64)

    if isinstance(value, (str, bytes, Mapping)):
        raise NeuronCodebookShapeError(
            f"{name} must be an integer sequence of length {intermediate}."
        )
    try:
        entries = tuple(value)
    except TypeError as exc:
        raise NeuronCodebookShapeError(
            f"{name} must be an integer sequence of length {intermediate}."
        ) from exc
    if len(entries) != intermediate:
        raise NeuronCodebookShapeError(f"{name} must have shape ({intermediate},).")
    converted: list[int] = []
    for neuron_index, entry in enumerate(entries):
        if isinstance(entry, bool):
            raise NeuronCodebookShapeError(
                f"{name}[{neuron_index}] must be an integer neuron index."
            )
        try:
            converted.append(operator.index(entry))
        except TypeError as exc:
            raise NeuronCodebookShapeError(
                f"{name}[{neuron_index}] must be an integer neuron index."
            ) from exc
    return torch.tensor(converted, dtype=torch.int64)


def _validated_neuron_permutations(
    value: Any,
    logical_layers: int,
    intermediate: int,
) -> tuple[Tensor, bool]:
    """Return aligned-index-to-physical-index rows and validation evidence."""

    provided = value is not None
    if value is None:
        permutations = torch.arange(intermediate, dtype=torch.int64).repeat(
            logical_layers, 1
        )
    elif isinstance(value, Tensor):
        if value.layout != torch.strided or value.device.type == "meta":
            raise NeuronCodebookShapeError(
                "neuron_permutations must be a dense non-meta tensor."
            )
        if value.ndim != 2 or tuple(value.shape) != (
            logical_layers,
            intermediate,
        ):
            raise NeuronCodebookShapeError(
                "neuron_permutations must have shape "
                f"[{logical_layers}, {intermediate}]."
            )
        permutations = torch.stack(
            [
                _permutation_row(value[index], index, intermediate)
                for index in range(logical_layers)
            ]
        )
    else:
        if isinstance(value, (str, bytes, Mapping)):
            raise NeuronCodebookShapeError(
                "neuron_permutations must be a tensor or iterable across depth."
            )
        try:
            rows = tuple(value)
        except TypeError as exc:
            raise NeuronCodebookShapeError(
                "neuron_permutations must be a tensor or iterable across depth."
            ) from exc
        if len(rows) != logical_layers:
            raise NeuronCodebookShapeError(
                "neuron_permutations must contain one row per logical layer."
            )
        permutations = torch.stack(
            [
                _permutation_row(row, index, intermediate)
                for index, row in enumerate(rows)
            ]
        )

    expected = torch.arange(intermediate, dtype=torch.int64)
    for layer_index, row in enumerate(permutations):
        if not torch.equal(torch.sort(row).values, expected):
            raise NeuronCodebookShapeError(
                f"neuron_permutations[{layer_index}] must be a bijection of "
                f"range({intermediate})."
            )
    return permutations.contiguous(), provided


def _even_indices(length: int, count: int) -> Tensor:
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    if count >= length:
        return torch.arange(length, dtype=torch.long)
    numerators = (2 * torch.arange(count, dtype=torch.long) + 1) * length
    return torch.div(numerators, 2 * count, rounding_mode="floor")


def _allocate_role_dimensions(widths: Sequence[int], total: int) -> tuple[int, ...]:
    effective = min(sum(widths), total)
    counts = [0] * len(widths)
    if effective >= len(widths):
        counts = [1] * len(widths)
        remaining = effective - len(widths)
    else:
        for index in range(effective):
            counts[index] = 1
        remaining = 0
    while remaining:
        candidates = [
            (width / (counts[index] + 1), -index, index)
            for index, width in enumerate(widths)
            if counts[index] < width
        ]
        if not candidates:  # pragma: no cover - effective already caps this
            break
        selected = max(candidates)[2]
        counts[selected] += 1
        remaining -= 1
    return tuple(counts)


def _sample_producer_role(
    weight: Tensor,
    bias: Tensor | None,
    count: int,
) -> Tensor:
    if count == 0:
        return torch.empty(
            (int(weight.shape[0]), 0),
            dtype=torch.float64,
            device="cpu",
        )
    width = int(weight.shape[1]) + int(bias is not None)
    indices = _even_indices(width, count)
    result = torch.empty(
        (int(weight.shape[0]), count),
        dtype=torch.float64,
        device="cpu",
    )
    weight_mask = indices < int(weight.shape[1])
    weight_positions = torch.nonzero(weight_mask, as_tuple=False).flatten()
    if int(weight_positions.numel()):
        columns = indices.index_select(0, weight_positions).to(weight.device)
        sampled = weight.detach().index_select(1, columns).to("cpu", torch.float64)
        result[:, weight_positions] = sampled
    bias_positions = torch.nonzero(~weight_mask, as_tuple=False).flatten()
    if int(bias_positions.numel()):
        if bias is None:  # pragma: no cover - index construction prevents this
            raise NeuronCodebookShapeError("A sampled producer bias is missing.")
        result[:, bias_positions] = bias.detach().to("cpu", torch.float64).unsqueeze(1)
    return result


def _weight_sketch(
    bundles: Sequence[GatedMLPBundle],
    requested_size: int,
) -> Tensor:
    intermediate = int(bundles[0].gate_weight.shape[0])
    gate_width = int(bundles[0].gate_weight.shape[1]) + int(
        bundles[0].gate_bias is not None
    )
    up_width = int(bundles[0].up_weight.shape[1]) + int(bundles[0].up_bias is not None)
    down_width = int(bundles[0].down_weight.shape[0])
    role_counts = _allocate_role_dimensions(
        (gate_width, up_width, down_width), requested_size
    )
    effective = sum(role_counts)
    result = torch.empty(
        (len(bundles) * intermediate, effective),
        dtype=torch.float64,
        device="cpu",
    )
    for layer_index, bundle in enumerate(bundles):
        parts = [
            _sample_producer_role(bundle.gate_weight, bundle.gate_bias, role_counts[0]),
            _sample_producer_role(bundle.up_weight, bundle.up_bias, role_counts[1]),
        ]
        down_indices = _even_indices(down_width, role_counts[2]).to(
            bundle.down_weight.device
        )
        sampled_down = (
            bundle.down_weight.detach()
            .index_select(0, down_indices)
            .transpose(0, 1)
            .to("cpu", torch.float64)
        )
        parts.append(sampled_down)
        start = layer_index * intermediate
        result[start : start + intermediate] = torch.cat(parts, dim=1)
    return result


def _activation_sequence(
    activations: Any,
    logical_layers: int,
) -> tuple[Tensor, ...] | None:
    if activations is None:
        return None
    if isinstance(activations, Tensor):
        if activations.ndim < 2 or int(activations.shape[0]) != logical_layers:
            raise NeuronCodebookShapeError(
                "A stacked activation tensor must have logical depth first."
            )
        return tuple(activations[index] for index in range(logical_layers))
    if not isinstance(activations, Sequence) or isinstance(activations, (str, bytes)):
        raise NeuronCodebookShapeError(
            "activation_sketches must be a tensor or sequence across depth."
        )
    if len(activations) != logical_layers:
        raise NeuronCodebookShapeError(
            "activation_sketches must contain one tensor per logical layer."
        )
    return tuple(activations)


def _activation_sketch(
    activations: tuple[Tensor, ...] | None,
    intermediate: int,
    requested_size: int,
) -> Tensor | None:
    if activations is None or requested_size == 0:
        return None
    flattened: list[Tensor] = []
    row_counts: list[int] = []
    for index, value in enumerate(activations):
        if not isinstance(value, Tensor) or value.ndim < 1:
            raise NeuronCodebookShapeError(
                f"activation_sketches[{index}] must be a non-scalar tensor."
            )
        if int(value.shape[-1]) != intermediate:
            raise NeuronCodebookShapeError(
                f"activation_sketches[{index}] must end in width {intermediate}."
            )
        if not torch.is_floating_point(value) or torch.is_complex(value):
            raise NeuronCodebookShapeError(
                f"activation_sketches[{index}] must be real floating point."
            )
        rows = value.detach().reshape(-1, intermediate)
        if int(rows.shape[0]) == 0:
            raise NeuronCodebookShapeError(
                f"activation_sketches[{index}] cannot be empty."
            )
        if not bool(torch.isfinite(rows).all()):
            raise NeuronCodebookFiniteError(
                f"activation_sketches[{index}] contains NaN or infinity."
            )
        flattened.append(rows)
        row_counts.append(int(rows.shape[0]))
    dimensions = min(requested_size, min(row_counts))
    result = torch.empty(
        (len(flattened) * intermediate, dimensions),
        dtype=torch.float64,
        device="cpu",
    )
    for layer_index, rows in enumerate(flattened):
        indices = _even_indices(int(rows.shape[0]), dimensions).to(rows.device)
        sampled = rows.index_select(0, indices).transpose(0, 1).to("cpu", torch.float64)
        start = layer_index * intermediate
        result[start : start + intermediate] = sampled
    return result


def _standardized_features(weight: Tensor, activation: Tensor | None) -> Tensor:
    blocks = [weight] if activation is None else [weight, activation]
    standardized: list[Tensor] = []
    for block in blocks:
        centered = block - block.mean(dim=0, keepdim=True)
        scale = torch.sqrt(centered.square().mean(dim=0, keepdim=True))
        usable = scale > torch.finfo(torch.float64).eps
        normalized = torch.where(usable, centered / scale.clamp_min(1e-30), 0.0)
        normalized = normalized / math.sqrt(max(1, int(block.shape[1])))
        standardized.append(normalized)
    result = torch.cat(standardized, dim=1).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise NeuronCodebookFiniteError(
            "Standardizing activation/weight sketches produced non-finite values."
        )
    return result


def _squared_distances(points: Tensor, centers: Tensor) -> Tensor:
    distances = points.square().sum(dim=1, keepdim=True)
    distances = distances + centers.square().sum(dim=1).unsqueeze(0)
    distances.addmm_(points, centers.transpose(0, 1), beta=1.0, alpha=-2.0)
    return distances.clamp_min_(0.0)


def _closest_to_centroid(
    features: Tensor,
    point_chunk_size: int,
    tracker: _WorkTracker,
) -> int:
    centroid = features.mean(dim=0, keepdim=True)
    best_distance = math.inf
    best_index = 0
    for start in range(0, int(features.shape[0]), point_chunk_size):
        end = min(int(features.shape[0]), start + point_chunk_size)
        distances = _squared_distances(features[start:end], centroid).flatten()
        tracker.add(end - start, 1)
        local_distance, local_index = torch.min(distances, dim=0)
        candidate_distance = float(local_distance.item())
        candidate_index = start + int(local_index.item())
        if candidate_distance < best_distance:
            best_distance = candidate_distance
            best_index = candidate_index
    return best_index


def _initial_medoids(
    features: Tensor,
    prototype_count: int,
    point_chunk_size: int,
    tracker: _WorkTracker,
) -> Tensor:
    point_count = int(features.shape[0])
    first = _closest_to_centroid(features, point_chunk_size, tracker)
    selected = torch.zeros(point_count, dtype=torch.bool)
    selected[first] = True
    medoids = [first]
    nearest = torch.full((point_count,), math.inf, dtype=torch.float64)

    while len(medoids) < prototype_count:
        center = features[medoids[-1] : medoids[-1] + 1]
        for start in range(0, point_count, point_chunk_size):
            end = min(point_count, start + point_chunk_size)
            distances = _squared_distances(features[start:end], center).flatten()
            tracker.add(end - start, 1)
            nearest[start:end] = torch.minimum(nearest[start:end], distances)
        candidates = nearest.clone()
        candidates[selected] = -1.0
        next_medoid = int(torch.argmax(candidates).item())
        medoids.append(next_medoid)
        selected[next_medoid] = True
    return torch.tensor(medoids, dtype=torch.long)


def _assign_points(
    features: Tensor,
    medoids: Tensor,
    point_chunk_size: int,
    center_chunk_size: int,
    tracker: _WorkTracker,
) -> tuple[Tensor, Tensor]:
    point_count = int(features.shape[0])
    prototype_count = int(medoids.numel())
    assignments = torch.zeros(point_count, dtype=torch.long)
    assigned_distances = torch.full((point_count,), math.inf, dtype=torch.float64)
    centers = features.index_select(0, medoids)
    for point_start in range(0, point_count, point_chunk_size):
        point_end = min(point_count, point_start + point_chunk_size)
        points = features[point_start:point_end]
        local_best = assigned_distances[point_start:point_end]
        local_assignment = assignments[point_start:point_end]
        for center_start in range(0, prototype_count, center_chunk_size):
            center_end = min(prototype_count, center_start + center_chunk_size)
            distances = _squared_distances(points, centers[center_start:center_end])
            tracker.add(point_end - point_start, center_end - center_start)
            chunk_best, chunk_indices = torch.min(distances, dim=1)
            improve = chunk_best < local_best
            local_best[improve] = chunk_best[improve]
            local_assignment[improve] = center_start + chunk_indices[improve]
        assigned_distances[point_start:point_end] = local_best
        assignments[point_start:point_end] = local_assignment
    for prototype_index, point_index in enumerate(medoids.tolist()):
        assignments[point_index] = prototype_index
        assigned_distances[point_index] = 0.0
    return assignments, assigned_distances


def _bounded_candidates(
    members: Tensor,
    current: int,
    limit: int,
) -> Tensor:
    if int(members.numel()) <= limit:
        return members
    positions = _even_indices(int(members.numel()), limit)
    candidates = members.index_select(0, positions).clone()
    if not bool((candidates == current).any()):
        candidates[-1] = current
    return torch.sort(candidates).values


def _refine_medoids(
    features: Tensor,
    medoids: Tensor,
    assignments: Tensor,
    point_chunk_size: int,
    center_chunk_size: int,
    candidate_limit: int,
    tracker: _WorkTracker,
) -> Tensor:
    refined = medoids.clone()
    for prototype_index, current in enumerate(medoids.tolist()):
        members = torch.nonzero(
            assignments == prototype_index, as_tuple=False
        ).flatten()
        if int(members.numel()) == 0:  # pragma: no cover - medoids own themselves
            continue
        candidates = _bounded_candidates(members, current, candidate_limit)
        costs = torch.zeros(int(candidates.numel()), dtype=torch.float64)
        for point_start in range(0, int(members.numel()), point_chunk_size):
            point_indices = members[point_start : point_start + point_chunk_size]
            points = features.index_select(0, point_indices)
            for center_start in range(0, int(candidates.numel()), center_chunk_size):
                center_end = min(
                    int(candidates.numel()), center_start + center_chunk_size
                )
                center_indices = candidates[center_start:center_end]
                centers = features.index_select(0, center_indices)
                distances = _squared_distances(points, centers)
                tracker.add(int(points.shape[0]), int(centers.shape[0]))
                costs[center_start:center_end] += distances.sum(dim=0)
        best_cost = torch.min(costs)
        tied = candidates[costs == best_cost]
        refined[prototype_index] = torch.min(tied)
    return refined


def _fit_medoids(
    features: Tensor,
    prototype_count: int,
    point_chunk_size: int,
    center_chunk_size: int,
    candidate_limit: int,
    max_iterations: int,
) -> tuple[Tensor, Tensor, Tensor, _WorkTracker, int]:
    point_count = int(features.shape[0])
    tracker = _WorkTracker()
    if prototype_count == point_count:
        identity = torch.arange(point_count, dtype=torch.long)
        return identity, identity.clone(), torch.zeros(point_count), tracker, 0

    medoids = _initial_medoids(features, prototype_count, point_chunk_size, tracker)
    iterations_run = 0
    for _ in range(max_iterations):
        assignments, _ = _assign_points(
            features,
            medoids,
            point_chunk_size,
            center_chunk_size,
            tracker,
        )
        refined = _refine_medoids(
            features,
            medoids,
            assignments,
            point_chunk_size,
            center_chunk_size,
            candidate_limit,
            tracker,
        )
        iterations_run += 1
        if torch.equal(refined, medoids):
            break
        medoids = refined
    assignments, distances = _assign_points(
        features,
        medoids,
        point_chunk_size,
        center_chunk_size,
        tracker,
    )
    return medoids, assignments, distances, tracker, iterations_run


def _fit_aligned_index_medoids(
    features: Tensor,
    permutations: Tensor,
    point_chunk_size: int,
    center_chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, _WorkTracker, int]:
    """Fit one exact real medoid inside each aligned depth-only group."""

    logical_layers, intermediate = map(int, permutations.shape)
    logical_neurons = logical_layers * intermediate
    if int(features.shape[0]) != logical_neurons:  # pragma: no cover - internal guard
        raise NeuronCodebookShapeError(
            "Aligned permutations and feature rows describe different neurons."
        )

    tracker = _WorkTracker()
    medoids = torch.empty(intermediate, dtype=torch.long)
    assignments = torch.empty(logical_neurons, dtype=torch.long)
    assigned_distances = torch.empty(logical_neurons, dtype=torch.float64)
    layer_offsets = torch.arange(logical_layers, dtype=torch.long) * intermediate

    for aligned_index in range(intermediate):
        members = layer_offsets + permutations[:, aligned_index]
        group_features = features.index_select(0, members)
        group_distances = torch.empty(
            (logical_layers, logical_layers),
            dtype=torch.float64,
        )
        for point_start in range(0, logical_layers, point_chunk_size):
            point_end = min(logical_layers, point_start + point_chunk_size)
            points = group_features[point_start:point_end]
            for center_start in range(0, logical_layers, center_chunk_size):
                center_end = min(logical_layers, center_start + center_chunk_size)
                centers = group_features[center_start:center_end]
                block = _squared_distances(points, centers)
                tracker.add(point_end - point_start, center_end - center_start)
                group_distances[
                    point_start:point_end,
                    center_start:center_end,
                ] = block

        costs = group_distances.sum(dim=0)
        best_cost = torch.min(costs)
        tied_positions = torch.nonzero(costs == best_cost, as_tuple=False).flatten()
        tied_global_indices = members.index_select(0, tied_positions)
        selected_tie = int(torch.argmin(tied_global_indices).item())
        medoid_position = int(tied_positions[selected_tie].item())
        medoid_global_index = int(members[medoid_position].item())

        medoids[aligned_index] = medoid_global_index
        assignments[members] = aligned_index
        assigned_distances[members] = group_distances[:, medoid_position]
        assigned_distances[medoid_global_index] = 0.0

    return medoids, assignments, assigned_distances, tracker, 1


def _source_neuron(
    bundles: Sequence[GatedMLPBundle],
    global_index: int,
) -> tuple[GatedMLPBundle, int]:
    intermediate = int(bundles[0].gate_weight.shape[0])
    layer_index, neuron_index = divmod(global_index, intermediate)
    return bundles[layer_index], neuron_index


def _gather_prototypes(
    bundles: Sequence[GatedMLPBundle],
    medoids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    gate_rows: list[Tensor] = []
    up_rows: list[Tensor] = []
    down_columns: list[Tensor] = []
    gate_biases: list[Tensor] = []
    up_biases: list[Tensor] = []
    for global_index in medoids.tolist():
        bundle, neuron_index = _source_neuron(bundles, global_index)
        gate_rows.append(bundle.gate_weight[neuron_index].detach().clone())
        up_rows.append(bundle.up_weight[neuron_index].detach().clone())
        down_columns.append(bundle.down_weight[:, neuron_index].detach().clone())
        if bundle.gate_bias is not None:
            gate_biases.append(bundle.gate_bias[neuron_index].detach().clone())
        if bundle.up_bias is not None:
            up_biases.append(bundle.up_bias[neuron_index].detach().clone())
    gate_bias = torch.stack(gate_biases) if gate_biases else None
    up_bias = torch.stack(up_biases) if up_biases else None
    return (
        torch.stack(gate_rows),
        torch.stack(up_rows),
        torch.stack(down_columns, dim=1),
        gate_bias,
        up_bias,
    )


def _least_squares_scale(source: Tensor, prototype: Tensor) -> float:
    source64 = source.detach().reshape(-1).to("cpu", torch.float64)
    prototype64 = prototype.detach().reshape(-1).to("cpu", torch.float64)
    denominator = float(torch.dot(prototype64, prototype64).item())
    if denominator <= torch.finfo(torch.float64).tiny:
        return 1.0 if bool(torch.equal(source64, prototype64)) else 0.0
    return float(torch.dot(source64, prototype64).item()) / denominator


def _producer_vector(weight: Tensor, bias: Tensor | None, neuron: int) -> Tensor:
    row = weight[neuron]
    if bias is None:
        return row
    return torch.cat((row, bias[neuron].reshape(1)))


def _fitted_scales(
    bundles: Sequence[GatedMLPBundle],
    medoids: Tensor,
    assignments: Tensor,
) -> Tensor:
    logical_layers = len(bundles)
    intermediate = int(bundles[0].gate_weight.shape[0])
    scales = torch.empty(
        (logical_layers, intermediate, 3),
        dtype=torch.float64,
        device="cpu",
    )
    for global_index, prototype_index in enumerate(assignments.tolist()):
        layer_index, neuron_index = divmod(global_index, intermediate)
        source = bundles[layer_index]
        medoid_global = int(medoids[prototype_index].item())
        prototype, prototype_neuron = _source_neuron(bundles, medoid_global)
        if global_index == medoid_global:
            role_scales = (1.0, 1.0, 1.0)
        else:
            role_scales = (
                _least_squares_scale(
                    _producer_vector(
                        source.gate_weight, source.gate_bias, neuron_index
                    ),
                    _producer_vector(
                        prototype.gate_weight,
                        prototype.gate_bias,
                        prototype_neuron,
                    ),
                ),
                _least_squares_scale(
                    _producer_vector(source.up_weight, source.up_bias, neuron_index),
                    _producer_vector(
                        prototype.up_weight,
                        prototype.up_bias,
                        prototype_neuron,
                    ),
                ),
                _least_squares_scale(
                    source.down_weight[:, neuron_index],
                    prototype.down_weight[:, prototype_neuron],
                ),
            )
        scales[layer_index, neuron_index] = torch.tensor(role_scales)
    reference = bundles[0].gate_weight
    result = scales.to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(result).all()):
        raise NeuronCodebookFiniteError("Fitted neuron scales are non-finite.")
    return result


def _validated_scales(
    value: Tensor,
    logical_layers: int,
    intermediate: int,
    reference: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor) or not torch.is_floating_point(value):
        raise NeuronCodebookShapeError("per_neuron_scales must be floating point.")
    if tuple(value.shape) == (logical_layers, intermediate):
        value = value.unsqueeze(-1).expand(-1, -1, 3)
    if tuple(value.shape) != (logical_layers, intermediate, 3):
        raise NeuronCodebookShapeError(
            "per_neuron_scales must have shape [layers, intermediate] or "
            "[layers, intermediate, 3]."
        )
    result = value.detach().to(device=reference.device, dtype=reference.dtype).clone()
    if not bool(torch.isfinite(result).all()):
        raise NeuronCodebookFiniteError("per_neuron_scales contains NaN or infinity.")
    return result


class NeuronCodebookRegistry(nn.Module):
    """The sole registered owner of codebook parameters and index buffers."""

    def __init__(
        self,
        prototype_gate_weight: Tensor,
        prototype_up_weight: Tensor,
        prototype_down_weight: Tensor,
        assignments: Tensor,
        prototype_source_indices: Tensor,
        *,
        prototype_gate_bias: Tensor | None = None,
        prototype_up_bias: Tensor | None = None,
        layer_down_bias: Tensor | None = None,
        neuron_scales: Tensor | None = None,
        activation: str = "silu",
        diagnostics: NeuronCodebookDiagnostics | None = None,
    ) -> None:
        super().__init__()
        activation = _normalize_activation(activation)
        gate = _validate_tensor(prototype_gate_weight, "prototype_gate_weight", 2)
        up = _validate_tensor(prototype_up_weight, "prototype_up_weight", 2)
        down = _validate_tensor(prototype_down_weight, "prototype_down_weight", 2)
        if tuple(gate.shape) != tuple(up.shape):
            raise NeuronCodebookShapeError(
                "Prototype gate and up weights must have equal shape."
            )
        prototype_count, input_width = map(int, gate.shape)
        output_width = int(down.shape[0])
        if int(down.shape[1]) != prototype_count:
            raise NeuronCodebookShapeError(
                "Prototype down columns must equal prototype count."
            )
        if not isinstance(assignments, Tensor) or assignments.ndim != 2:
            raise NeuronCodebookShapeError(
                "assignments must have shape [layers, intermediate]."
            )
        logical_layers, intermediate = map(int, assignments.shape)
        if logical_layers <= 0 or intermediate <= 0:
            raise NeuronCodebookShapeError("assignments cannot have an empty axis.")
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if assignments.dtype not in integer_dtypes:
            raise NeuronCodebookShapeError("assignments must use an integer dtype.")
        assignment64 = assignments.detach().to("cpu", torch.int64)
        if int(assignment64.min()) < 0 or int(assignment64.max()) >= prototype_count:
            raise NeuronCodebookShapeError(
                "assignments contains an out-of-range prototype index."
            )
        if (
            not isinstance(prototype_source_indices, Tensor)
            or prototype_source_indices.ndim != 1
            or int(prototype_source_indices.numel()) != prototype_count
            or prototype_source_indices.dtype not in integer_dtypes
        ):
            raise NeuronCodebookShapeError(
                "prototype_source_indices must contain one integer per prototype."
            )
        source64 = prototype_source_indices.detach().to("cpu", torch.int64)
        logical_neurons = logical_layers * intermediate
        if int(source64.min()) < 0 or int(source64.max()) >= logical_neurons:
            raise NeuronCodebookShapeError(
                "prototype_source_indices contains an out-of-range logical neuron."
            )
        if int(torch.unique(source64).numel()) != prototype_count:
            raise NeuronCodebookShapeError(
                "prototype_source_indices must identify distinct real neurons."
            )

        tensors = [gate, up, down]
        for name, value, expected in (
            ("prototype_gate_bias", prototype_gate_bias, (prototype_count,)),
            ("prototype_up_bias", prototype_up_bias, (prototype_count,)),
            ("layer_down_bias", layer_down_bias, (logical_layers, output_width)),
            ("neuron_scales", neuron_scales, (logical_layers, intermediate, 3)),
        ):
            if value is None:
                continue
            checked = _validate_tensor(value, name, len(expected))
            if tuple(checked.shape) != expected:
                raise NeuronCodebookShapeError(f"{name} must have shape {expected!r}.")
            tensors.append(checked)
        if any(value.dtype != gate.dtype for value in tensors):
            raise NeuronCodebookShapeError("All learned tensors need one dtype.")
        if any(value.device != gate.device for value in tensors):
            raise NeuronCodebookShapeError("All learned tensors need one device.")

        self.prototype_gate_weight = nn.Parameter(gate.detach().clone())
        self.prototype_up_weight = nn.Parameter(up.detach().clone())
        self.prototype_down_weight = nn.Parameter(down.detach().clone())
        self.prototype_gate_bias = (
            nn.Parameter(prototype_gate_bias.detach().clone())
            if prototype_gate_bias is not None
            else None
        )
        self.prototype_up_bias = (
            nn.Parameter(prototype_up_bias.detach().clone())
            if prototype_up_bias is not None
            else None
        )
        self.layer_down_bias = (
            nn.Parameter(layer_down_bias.detach().clone())
            if layer_down_bias is not None
            else None
        )
        self.neuron_scales = (
            nn.Parameter(neuron_scales.detach().clone())
            if neuron_scales is not None
            else None
        )
        self.register_buffer("assignments", assignment64.to(torch.int32))
        self.register_buffer("prototype_source_indices", source64.to(torch.int32))
        self.register_buffer(
            "_activation_code",
            torch.tensor(_ACTIVATION_CODES[activation], dtype=torch.uint8),
        )
        self.diagnostics = diagnostics
        self.input_width = input_width
        self.output_width = output_width

    @property
    def logical_layers(self) -> int:
        return int(self.assignments.shape[0])

    @property
    def intermediate_width(self) -> int:
        return int(self.assignments.shape[1])

    @property
    def prototype_count(self) -> int:
        return int(self.prototype_gate_weight.shape[0])

    @property
    def activation(self) -> str:
        code = int(self._activation_code.item())
        try:
            return _ACTIVATION_NAMES[code]
        except KeyError as exc:
            raise NeuronCodebookStateError(
                f"State contains unknown activation code {code}."
            ) from exc

    @property
    def has_producer_bias(self) -> bool:
        return (
            self.prototype_gate_bias is not None or self.prototype_up_bias is not None
        )

    @property
    def has_gate_bias(self) -> bool:
        return self.prototype_gate_bias is not None

    @property
    def has_up_bias(self) -> bool:
        return self.prototype_up_bias is not None

    @property
    def has_down_bias(self) -> bool:
        return self.layer_down_bias is not None

    @property
    def has_scales(self) -> bool:
        return self.neuron_scales is not None

    @property
    def gate_scales(self) -> Tensor | None:
        if self.neuron_scales is None:
            return None
        return self.neuron_scales[..., 0]

    @property
    def up_scales(self) -> Tensor | None:
        if self.neuron_scales is None:
            return None
        return self.neuron_scales[..., 1]

    @property
    def down_scales(self) -> Tensor | None:
        if self.neuron_scales is None:
            return None
        return self.neuron_scales[..., 2]

    @property
    def gate_prototypes(self) -> Tensor:
        return self.prototype_gate_weight

    @property
    def up_prototypes(self) -> Tensor:
        return self.prototype_up_weight

    @property
    def down_prototypes(self) -> Tensor:
        return self.prototype_down_weight

    @property
    def layer_assignments(self) -> Tensor:
        return self.assignments

    @property
    def scales(self) -> Tensor | None:
        return self.neuron_scales

    @property
    def report(self) -> NeuronCodebookParameterReport:
        logical_layers = self.logical_layers
        intermediate = self.intermediate_width
        producer_biases = intermediate * (
            int(self.prototype_gate_bias is not None)
            + int(self.prototype_up_bias is not None)
        )
        down_biases = self.output_width if self.has_down_bias else 0
        per_layer = (
            2 * intermediate * self.input_width
            + self.output_width * intermediate
            + producer_biases
            + down_biases
        )
        logical_dense = logical_layers * per_layer
        unique_learned = sum(parameter.numel() for parameter in self.parameters())
        scale_parameters = (
            int(self.neuron_scales.numel()) if self.neuron_scales is not None else 0
        )
        retained_bias = (
            int(self.layer_down_bias.numel()) if self.layer_down_bias is not None else 0
        )
        prototype_parameters = unique_learned - scale_parameters - retained_bias
        assignment_bytes = self.assignments.numel() * self.assignments.element_size()
        source_bytes = (
            self.prototype_source_indices.numel()
            * self.prototype_source_indices.element_size()
        )
        return NeuronCodebookParameterReport(
            logical_layers=logical_layers,
            input_width=self.input_width,
            intermediate_width=intermediate,
            output_width=self.output_width,
            prototype_count=self.prototype_count,
            logical_neuron_executions=logical_layers * intermediate,
            logical_dense_parameters=logical_dense,
            unique_learned_parameters=unique_learned,
            prototype_parameters=prototype_parameters,
            scale_parameters=scale_parameters,
            retained_layer_bias_parameters=retained_bias,
            parameters_saved=logical_dense - unique_learned,
            assignment_index_bytes=assignment_bytes,
            prototype_source_index_bytes=source_bytes,
            index_buffer_bytes=assignment_bytes + source_bytes,
        )

    def _layer_index(self, layer_index: int) -> int:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.logical_layers
        ):
            raise NeuronCodebookShapeError(
                "layer_index must identify a logical codebook layer."
            )
        return layer_index

    def materialize_layer(self, layer_index: int) -> GatedMLPBundle:
        """Gather one logical layer with one coupled assignment vector."""

        layer_index = self._layer_index(layer_index)
        indices = self.assignments[layer_index].to(
            device=self.prototype_gate_weight.device, dtype=torch.long
        )
        gate_weight = self.prototype_gate_weight.index_select(0, indices)
        up_weight = self.prototype_up_weight.index_select(0, indices)
        down_weight = self.prototype_down_weight.index_select(1, indices)
        gate_bias = (
            self.prototype_gate_bias.index_select(0, indices)
            if self.prototype_gate_bias is not None
            else None
        )
        up_bias = (
            self.prototype_up_bias.index_select(0, indices)
            if self.prototype_up_bias is not None
            else None
        )
        if self.neuron_scales is not None:
            scales = self.neuron_scales[layer_index]
            gate_weight = gate_weight * scales[:, 0].unsqueeze(1)
            up_weight = up_weight * scales[:, 1].unsqueeze(1)
            down_weight = down_weight * scales[:, 2].unsqueeze(0)
            if gate_bias is not None:
                gate_bias = gate_bias * scales[:, 0]
            if up_bias is not None:
                up_bias = up_bias * scales[:, 1]
        down_bias = (
            self.layer_down_bias[layer_index]
            if self.layer_down_bias is not None
            else None
        )
        return GatedMLPBundle(
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            gate_bias=gate_bias,
            up_bias=up_bias,
            down_bias=down_bias,
        )

    def forward(self, inputs: Tensor, layer_index: int) -> Tensor:
        if not isinstance(inputs, Tensor) or inputs.ndim == 0:
            raise NeuronCodebookShapeError("inputs must be a non-scalar tensor.")
        if int(inputs.shape[-1]) != self.input_width:
            raise NeuronCodebookShapeError(
                f"inputs must end in width {self.input_width}."
            )
        bundle = self.materialize_layer(layer_index)
        return bundle.forward(inputs, activation=self.activation)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
        *,
        prefix: str = "",
    ) -> NeuronCodebookRegistry:
        """Reconstruct solely from ordinary tensors, without opaque metadata."""

        if not isinstance(state_dict, Mapping):
            raise NeuronCodebookStateError("state_dict must be a tensor mapping.")
        if not prefix and "registry.prototype_gate_weight" in state_dict:
            prefix = "registry"
        if prefix and not prefix.endswith("."):
            prefix = f"{prefix}."
        required = (
            "prototype_gate_weight",
            "prototype_up_weight",
            "prototype_down_weight",
            "assignments",
            "prototype_source_indices",
            "_activation_code",
        )
        missing = [name for name in required if f"{prefix}{name}" not in state_dict]
        if missing:
            raise NeuronCodebookStateError(
                f"state_dict is missing required tensors {missing!r}."
            )
        code_tensor = state_dict[f"{prefix}_activation_code"]
        if not isinstance(code_tensor, Tensor) or code_tensor.numel() != 1:
            raise NeuronCodebookStateError(
                "_activation_code must be one tensor scalar."
            )
        code = int(code_tensor.detach().to("cpu").item())
        if code not in _ACTIVATION_NAMES:
            raise NeuronCodebookStateError(
                f"state_dict has unknown activation code {code}."
            )

        def optional(name: str) -> Tensor | None:
            return state_dict.get(f"{prefix}{name}")

        result = cls(
            state_dict[f"{prefix}prototype_gate_weight"],
            state_dict[f"{prefix}prototype_up_weight"],
            state_dict[f"{prefix}prototype_down_weight"],
            state_dict[f"{prefix}assignments"],
            state_dict[f"{prefix}prototype_source_indices"],
            prototype_gate_bias=optional("prototype_gate_bias"),
            prototype_up_bias=optional("prototype_up_bias"),
            layer_down_bias=optional("layer_down_bias"),
            neuron_scales=optional("neuron_scales"),
            activation=_ACTIVATION_NAMES[code],
        )
        local_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        incompatible = result.load_state_dict(local_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise NeuronCodebookStateError(
                "state_dict did not exactly match the reconstructed registry."
            )
        return result


class NeuronCodebookLayerView(nn.Module):
    """Parameterless runtime view of one logical layer.

    ``object.__setattr__`` deliberately keeps the registry as an ordinary strong
    reference.  It is not registered a second time beneath every view.
    """

    def __init__(self, registry: NeuronCodebookRegistry, layer_index: int) -> None:
        super().__init__()
        if not isinstance(registry, NeuronCodebookRegistry):
            raise NeuronCodebookShapeError("registry must be a NeuronCodebookRegistry.")
        registry._layer_index(layer_index)
        object.__setattr__(self, "_registry", registry)
        self.layer_index = layer_index
        self.in_features = registry.input_width
        self.out_features = registry.output_width
        self.intermediate_features = registry.intermediate_width
        self.train(registry.training)

    @property
    def registry(self) -> NeuronCodebookRegistry:
        return self._registry

    def forward(self, inputs: Tensor) -> Tensor:
        return self.registry(inputs, self.layer_index)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"intermediate_features={self.intermediate_features}, "
            f"out_features={self.out_features}, layer_index={self.layer_index}"
        )


class GlobalNeuronCodebook(nn.Module):
    """One strongly owned registry plus all logical parameterless layer views."""

    def __init__(self, registry: NeuronCodebookRegistry) -> None:
        super().__init__()
        if not isinstance(registry, NeuronCodebookRegistry):
            raise NeuronCodebookShapeError("registry must be a NeuronCodebookRegistry.")
        self.registry = registry
        self.layers = nn.ModuleList(
            NeuronCodebookLayerView(registry, index)
            for index in range(registry.logical_layers)
        )

    @property
    def layer_views(self) -> nn.ModuleList:
        return self.layers

    @property
    def logical_layers(self) -> int:
        return self.registry.logical_layers

    @property
    def report(self) -> NeuronCodebookParameterReport:
        return self.registry.report

    @property
    def diagnostics(self) -> NeuronCodebookDiagnostics | None:
        return self.registry.diagnostics

    def __len__(self) -> int:
        return self.logical_layers

    def __getitem__(self, layer_index: int) -> NeuronCodebookLayerView:
        return self.layers[layer_index]

    def forward(self, inputs: Tensor, layer_index: int) -> Tensor:
        return self.registry(inputs, layer_index)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
    ) -> GlobalNeuronCodebook:
        """Rebuild a complete runtime from its ordinary tensor state mapping."""

        prefix = "registry" if "registry.prototype_gate_weight" in state_dict else ""
        registry = NeuronCodebookRegistry.from_state_dict(
            state_dict,
            prefix=prefix,
        )
        result = cls(registry)
        if prefix:
            incompatible = result.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise NeuronCodebookStateError(
                    "state_dict did not exactly match the reconstructed codebook."
                )
        return result


def _resolve_prototype_count(
    prototype_count: int | None,
    codebook_size: int | None,
    num_prototypes: int | None,
    logical_neurons: int,
    *,
    default_count: int | None = None,
) -> int:
    values = [
        value
        for value in (prototype_count, codebook_size, num_prototypes)
        if value is not None
    ]
    if len(values) > 1:
        raise NeuronCodebookShapeError(
            "Specify only one of prototype_count, codebook_size, or num_prototypes."
        )
    result = (
        (logical_neurons if default_count is None else default_count)
        if not values
        else _positive_int(values[0], "prototype_count")
    )
    if result > logical_neurons:
        raise NeuronCodebookShapeError(
            f"prototype_count cannot exceed {logical_neurons} logical neurons."
        )
    return result


def _resolve_scale_request(
    use_scales: bool,
    per_neuron_scales: bool | Tensor | None,
) -> bool | Tensor:
    if not isinstance(use_scales, bool):
        raise NeuronCodebookShapeError("use_scales must be boolean.")
    if per_neuron_scales is None:
        return use_scales
    if use_scales:
        raise NeuronCodebookShapeError(
            "Specify use_scales or per_neuron_scales, not both."
        )
    if not isinstance(per_neuron_scales, (bool, Tensor)):
        raise NeuronCodebookShapeError(
            "per_neuron_scales must be boolean or a scale tensor."
        )
    return per_neuron_scales


def fit_global_neuron_codebook(
    bundles: Sequence[Any],
    prototype_count: int | None = None,
    *,
    codebook_size: int | None = None,
    num_prototypes: int | None = None,
    activation_sketches: Tensor | Sequence[Tensor] | None = None,
    activations: Tensor | Sequence[Tensor] | None = None,
    activation: str = "silu",
    use_scales: bool = False,
    per_neuron_scales: bool | Tensor | None = None,
    weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
    activation_sketch_size: int = DEFAULT_ACTIVATION_SKETCH_SIZE,
    point_chunk_size: int = DEFAULT_POINT_CHUNK_SIZE,
    center_chunk_size: int = DEFAULT_CENTER_CHUNK_SIZE,
    medoid_candidate_limit: int = DEFAULT_MEDOID_CANDIDATE_LIMIT,
    max_iterations: int = DEFAULT_MEDOID_ITERATIONS,
    assignment_strategy: str = _GLOBAL_MEDOIDS_STRATEGY,
    neuron_permutations: Tensor | Sequence[Tensor | Sequence[int]] | None = None,
) -> GlobalNeuronCodebook:
    """Fit one global real-neuron codebook without building an ``N x N`` matrix.

    ``activation_sketches`` are optional pre-down neuron activations with
    intermediate width on the final axis.  ``per_neuron_scales=True`` fits
    independent gate/up/down least-squares scales for every logical neuron;
    callers may instead provide a ``[layers, intermediate]`` shared-role scale
    tensor or a ``[layers, intermediate, 3]`` role-wise tensor.

    The default ``global_medoids`` strategy preserves unrestricted global
    clustering.  ``aligned_index`` creates exactly one real medoid per
    intermediate index.  For that strategy, ``neuron_permutations[layer,
    aligned_index]`` identifies the physical gate/up row and coupled down
    column occupying the aligned slot.  Every row must be a bijection of
    ``range(intermediate)``; omitted permutations are identity rows.  Aligned
    fitting evaluates exactly ``intermediate * layers**2`` sketch distances,
    uses no logical-neuron by prototype assignment search, and fixes the
    deterministic subgroup factor at one.
    """

    if activation_sketches is not None and activations is not None:
        raise NeuronCodebookShapeError(
            "Specify activation_sketches or activations, not both."
        )
    activation_values = (
        activation_sketches if activation_sketches is not None else activations
    )
    weight_size = _positive_int(weight_sketch_size, "weight_sketch_size")
    activation_size = _nonnegative_int(activation_sketch_size, "activation_sketch_size")
    point_chunk = _positive_int(point_chunk_size, "point_chunk_size")
    center_chunk = _positive_int(center_chunk_size, "center_chunk_size")
    candidate_limit = _positive_int(medoid_candidate_limit, "medoid_candidate_limit")
    iterations = _nonnegative_int(max_iterations, "max_iterations")
    activation_name = _normalize_activation(activation)
    strategy = _normalize_assignment_strategy(assignment_strategy)
    if strategy != _ALIGNED_INDEX_STRATEGY and neuron_permutations is not None:
        raise NeuronCodebookShapeError(
            "neuron_permutations are only valid with "
            "assignment_strategy='aligned_index'."
        )
    scale_request = _resolve_scale_request(use_scales, per_neuron_scales)

    with torch.no_grad():
        sources = _validated_bundles(bundles)
        logical_layers = len(sources)
        intermediate = int(sources[0].gate_weight.shape[0])
        logical_neurons = logical_layers * intermediate
        prototypes = _resolve_prototype_count(
            prototype_count,
            codebook_size,
            num_prototypes,
            logical_neurons,
            default_count=(
                intermediate if strategy == _ALIGNED_INDEX_STRATEGY else None
            ),
        )
        if strategy == _ALIGNED_INDEX_STRATEGY and prototypes != intermediate:
            raise NeuronCodebookShapeError(
                "assignment_strategy='aligned_index' requires exactly one "
                f"prototype per aligned index ({intermediate} prototypes)."
            )
        if strategy == _ALIGNED_INDEX_STRATEGY:
            aligned_permutations, permutations_provided = (
                _validated_neuron_permutations(
                    neuron_permutations,
                    logical_layers,
                    intermediate,
                )
            )
        else:
            aligned_permutations = None
            permutations_provided = False
        activation_sequence = _activation_sequence(activation_values, logical_layers)
        weight_features = _weight_sketch(sources, weight_size)
        activation_features = _activation_sketch(
            activation_sequence,
            intermediate,
            activation_size,
        )
        features = _standardized_features(weight_features, activation_features)
        if aligned_permutations is not None:
            medoids, flat_assignments, distances, tracker, iterations_run = (
                _fit_aligned_index_medoids(
                    features,
                    aligned_permutations,
                    point_chunk,
                    center_chunk,
                )
            )
            exact_distance_work_bound = intermediate * logical_layers * logical_layers
            if tracker.distance_values != exact_distance_work_bound:
                raise NeuronCodebookStateError(
                    "Aligned-index fitting violated its exact distance-work bound."
                )
            algorithm = "deterministic_aligned_index_exact_group_medoids"
            distance_work_bound_formula = "intermediate_width * logical_layers**2"
            used_n_by_k_assignment_search = False
            permutation_proof = True
            permutation_proof_method = "sorted_rows_equal_arange"
            permutation_rows_verified = logical_layers
            aligned_group_size = logical_layers
            aligned_subgroup_factor = 1
        else:
            medoids, flat_assignments, distances, tracker, iterations_run = (
                _fit_medoids(
                    features,
                    prototypes,
                    point_chunk,
                    center_chunk,
                    candidate_limit,
                    iterations,
                )
            )
            exact_distance_work_bound = tracker.distance_values
            algorithm = "deterministic_chunked_farthest_first_bounded_pam"
            distance_work_bound_formula = "measured_global_medoids_distance_values"
            used_n_by_k_assignment_search = prototypes != logical_neurons
            permutation_proof = False
            permutation_proof_method = "not_applicable"
            permutation_rows_verified = 0
            aligned_group_size = 0
            aligned_subgroup_factor = 0
        (
            prototype_gate_weight,
            prototype_up_weight,
            prototype_down_weight,
            prototype_gate_bias,
            prototype_up_bias,
        ) = _gather_prototypes(sources, medoids)
        layer_down_bias = (
            torch.stack([source.down_bias.detach().clone() for source in sources])
            if sources[0].down_bias is not None
            else None
        )
        if isinstance(scale_request, Tensor):
            neuron_scales = _validated_scales(
                scale_request,
                logical_layers,
                intermediate,
                sources[0].gate_weight,
            )
        elif scale_request:
            neuron_scales = _fitted_scales(sources, medoids, flat_assignments)
        else:
            neuron_scales = None

        diagnostics = NeuronCodebookDiagnostics(
            algorithm=algorithm,
            assignment_strategy=strategy,
            logical_layers=logical_layers,
            intermediate_width=intermediate,
            logical_neurons=logical_neurons,
            prototype_count=prototypes,
            deterministic=True,
            tie_break=_TIE_BREAK,
            real_medoid_prototypes=True,
            used_activation_sketches=activation_features is not None,
            requested_weight_sketch_size=weight_size,
            weight_sketch_dimensions=int(weight_features.shape[1]),
            requested_activation_sketch_size=activation_size,
            activation_sketch_dimensions=(
                int(activation_features.shape[1])
                if activation_features is not None
                else 0
            ),
            combined_sketch_dimensions=int(features.shape[1]),
            sketch_elements=int(features.numel()),
            point_chunk_size=point_chunk,
            center_chunk_size=center_chunk,
            medoid_candidate_limit=candidate_limit,
            max_iterations=iterations,
            iterations_run=iterations_run,
            distance_values_computed=tracker.distance_values,
            exact_distance_work_bound=exact_distance_work_bound,
            exact_distance_coordinate_work_bound=(
                exact_distance_work_bound * int(features.shape[1])
            ),
            distance_work_bound_formula=distance_work_bound_formula,
            max_distance_block_elements=tracker.maximum_block_elements,
            distance_block_element_limit=point_chunk * center_chunk,
            full_pairwise_matrix_elements=logical_neurons * logical_neurons,
            materialized_full_pairwise_matrix=False,
            used_n_by_k_assignment_search=used_n_by_k_assignment_search,
            permutations_provided=permutations_provided,
            permutation_proof=permutation_proof,
            permutation_proof_method=permutation_proof_method,
            permutation_rows_verified=permutation_rows_verified,
            aligned_group_size=aligned_group_size,
            aligned_subgroup_factor=aligned_subgroup_factor,
            bounded_work=(
                tracker.maximum_block_elements <= point_chunk * center_chunk
                and tracker.distance_values <= exact_distance_work_bound
            ),
            fitting_requires_grad=False,
            squared_sketch_error=float(distances.sum().item()),
            maximum_assigned_squared_distance=float(distances.max().item()),
        )
        registry = NeuronCodebookRegistry(
            prototype_gate_weight,
            prototype_up_weight,
            prototype_down_weight,
            flat_assignments.reshape(logical_layers, intermediate),
            medoids,
            prototype_gate_bias=prototype_gate_bias,
            prototype_up_bias=prototype_up_bias,
            layer_down_bias=layer_down_bias,
            neuron_scales=neuron_scales,
            activation=activation_name,
            diagnostics=diagnostics,
        )
    return GlobalNeuronCodebook(registry)


fit_neuron_codebook = fit_global_neuron_codebook
build_global_neuron_codebook = fit_global_neuron_codebook
TopologyPreservingNeuronCodebook = GlobalNeuronCodebook
ParameterlessNeuronCodebookLayer = NeuronCodebookLayerView


__all__ = [
    "DEFAULT_ACTIVATION_SKETCH_SIZE",
    "DEFAULT_CENTER_CHUNK_SIZE",
    "DEFAULT_MEDOID_CANDIDATE_LIMIT",
    "DEFAULT_MEDOID_ITERATIONS",
    "DEFAULT_POINT_CHUNK_SIZE",
    "DEFAULT_WEIGHT_SKETCH_SIZE",
    "GatedMLPBundle",
    "GlobalNeuronCodebook",
    "NeuronCodebookDiagnostics",
    "NeuronCodebookError",
    "NeuronCodebookFiniteError",
    "NeuronCodebookLayerView",
    "NeuronCodebookParameterReport",
    "NeuronCodebookRegistry",
    "NeuronCodebookShapeError",
    "NeuronCodebookStateError",
    "ParameterlessNeuronCodebookLayer",
    "TopologyPreservingNeuronCodebook",
    "build_global_neuron_codebook",
    "fit_global_neuron_codebook",
    "fit_neuron_codebook",
]
