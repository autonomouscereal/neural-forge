"""Deterministic cross-layer permutation inference for HF codebooks.

The inferred rows map canonical slots to physical source slots.  They can be
passed directly as ``neuron_permutations`` to ``hf_neuron_codebook`` or as
``group_permutations`` to ``hf_attention_codebook``.  SwiGLU inference never
forms an intermediate-width by intermediate-width distance matrix: compact
coupled weight/activation sketches are ordered by a coarse Morton signature,
then refined in bounded local assignment blocks.  Qwen GQA groups are small,
so exact per-layer assignment is used up to an explicit limit.

Fitting is raw Torch, deterministic, and entirely inside ``torch.no_grad``.
All returned permutation and inverse tensors live on CPU and are proven
bijective before a result is returned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from torch import Tensor


DEFAULT_WEIGHT_SKETCH_SIZE = 64
DEFAULT_ACTIVATION_SKETCH_SIZE = 32
DEFAULT_SIGNATURE_DIMENSIONS = 8
DEFAULT_SIGNATURE_BITS = 6
DEFAULT_LOCAL_BLOCK_SIZE = 32
DEFAULT_REFINEMENT_PASSES = 2
DEFAULT_MAX_EXACT_GQA_GROUPS = 64

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_TIE_BREAK = (
    "identity on objective ties; stable Morton signature order; "
    "lowest canonical then physical index in assignment ties"
)


class CodebookAlignmentError(ValueError):
    """Base class for alignment validation and inference failures."""


class CodebookAlignmentShapeError(CodebookAlignmentError):
    """A bundle, activation sketch, topology, or option has invalid shape."""


class CodebookAlignmentFiniteError(CodebookAlignmentError):
    """A weight, activation sketch, or derived objective is non-finite."""


class CodebookAlignmentStateError(CodebookAlignmentError):
    """An internal bijection, inverse, or work-bound proof failed."""


@dataclass(frozen=True, slots=True)
class PermutationAlignmentDiagnostics:
    """Objective, proof, determinism, and bounded-work evidence."""

    topology: str
    algorithm: str
    assignment_solver: str
    logical_layers: int
    units_per_layer: int
    reference_layer: int
    deterministic: bool
    tie_break: str
    fitting_requires_grad: bool
    used_weight_sketches: bool
    used_activation_sketches: bool
    requested_weight_sketch_size: int
    weight_sketch_dimensions: int
    requested_activation_sketch_size: int
    activation_sketch_dimensions: int
    combined_sketch_dimensions: int
    sketch_elements: int
    coarse_signature_dimensions: int
    coarse_bits_per_dimension: int
    local_block_size: int
    refinement_passes: int
    local_assignment_blocks: int
    accepted_refinement_blocks: int
    exact_small_group_assignment: bool
    identity_baseline_selected_layers: int
    pairwise_distance_values: int
    distance_feature_operations: int
    distance_work_bound: int
    distance_work_bound_formula: str
    maximum_pairwise_block_elements: int
    pairwise_block_element_limit: int
    full_pairwise_matrix_elements: int
    materialized_full_pairwise_matrix: bool
    bounded_work: bool
    identity_objective: float
    coarse_objective: float
    aligned_objective: float
    objective_improvement: float
    objective_improvement_fraction: float
    objective_ratio: float
    per_layer_identity_objective: tuple[float, ...]
    per_layer_coarse_objective: tuple[float, ...]
    per_layer_aligned_objective: tuple[float, ...]
    permutations_bijective: bool
    inverses_verified: bool
    permutation_rows_verified: int
    permutation_proof_method: str

    @property
    def objective(self) -> float:
        return self.aligned_objective

    @property
    def final_objective(self) -> float:
        return self.aligned_objective

    @property
    def inferred_objective(self) -> float:
        return self.aligned_objective

    @property
    def identity_baseline_objective(self) -> float:
        return self.identity_objective

    @property
    def objective_delta(self) -> float:
        return self.identity_objective - self.aligned_objective

    @property
    def objective_better_than_or_equal_to_identity(self) -> bool:
        return self.aligned_objective <= self.identity_objective

    @property
    def zero_gradient(self) -> bool:
        return not self.fitting_requires_grad

    @property
    def zero_grad(self) -> bool:
        return self.zero_gradient

    @property
    def permutation_proof(self) -> bool:
        return self.permutations_bijective and self.inverses_verified

    @property
    def inverse_proof(self) -> bool:
        return self.inverses_verified

    @property
    def no_full_pairwise_matrix(self) -> bool:
        return not self.materialized_full_pairwise_matrix

    @property
    def maximum_assignment_matrix_elements(self) -> int:
        return self.maximum_pairwise_block_elements

    @property
    def assignment_matrix_element_limit(self) -> int:
        return self.pairwise_block_element_limit

    @property
    def bounded_memory(self) -> bool:
        return self.bounded_work

    @property
    def distance_values_computed(self) -> int:
        return self.pairwise_distance_values

    @property
    def exact_distance_work_bound(self) -> int:
        return self.distance_work_bound

    @property
    def exact_distance_coordinate_work_bound(self) -> int:
        return self.distance_work_bound * self.combined_sketch_dimensions

    @property
    def work_complexity(self) -> str:
        if self.exact_small_group_assignment:
            return "O(layers * groups^2 * sketch_dim)"
        return "O(layers * width * log(width) + layers * width * local_block)"

    def as_dict(self) -> dict[str, Any]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result.update(
            {
                "objective": self.objective,
                "final_objective": self.final_objective,
                "inferred_objective": self.inferred_objective,
                "identity_baseline_objective": self.identity_baseline_objective,
                "objective_delta": self.objective_delta,
                "objective_better_than_or_equal_to_identity": (
                    self.objective_better_than_or_equal_to_identity
                ),
                "zero_gradient": self.zero_gradient,
                "zero_grad": self.zero_grad,
                "permutation_proof": self.permutation_proof,
                "inverse_proof": self.inverse_proof,
                "no_full_pairwise_matrix": self.no_full_pairwise_matrix,
                "maximum_assignment_matrix_elements": (
                    self.maximum_assignment_matrix_elements
                ),
                "assignment_matrix_element_limit": (
                    self.assignment_matrix_element_limit
                ),
                "bounded_memory": self.bounded_memory,
                "distance_values_computed": self.distance_values_computed,
                "exact_distance_work_bound": self.exact_distance_work_bound,
                "exact_distance_coordinate_work_bound": (
                    self.exact_distance_coordinate_work_bound
                ),
                "work_complexity": self.work_complexity,
            }
        )
        return result

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class PermutationAlignmentResult:
    """Canonical-to-physical rows, verified inverses, and diagnostics."""

    permutations: Tensor
    inverse_permutations: Tensor
    diagnostics: PermutationAlignmentDiagnostics

    @property
    def canonical_to_physical(self) -> Tensor:
        return self.permutations

    @property
    def physical_to_canonical(self) -> Tensor:
        return self.inverse_permutations

    @property
    def inverse(self) -> Tensor:
        return self.inverse_permutations

    @property
    def neuron_permutations(self) -> Tensor:
        return self.permutations

    @property
    def group_permutations(self) -> Tensor:
        return self.permutations

    @property
    def kv_group_permutations(self) -> Tensor:
        return self.permutations

    @property
    def permutation_proof(self) -> bool:
        return self.diagnostics.permutation_proof

    @property
    def shape(self) -> torch.Size:
        return self.permutations.shape

    def __len__(self) -> int:
        return len(self.permutations)

    def __getitem__(self, index: Any) -> Tensor:
        return self.permutations[index]

    def codebook_kwargs(self) -> dict[str, Tensor]:
        key = (
            "neuron_permutations"
            if self.diagnostics.topology == "coupled_swiglu_neurons"
            else "group_permutations"
        )
        return {key: self.permutations}


CodebookAlignmentDiagnostics = PermutationAlignmentDiagnostics
CodebookAlignmentResult = PermutationAlignmentResult
CodebookPermutationAlignment = PermutationAlignmentResult


@dataclass(frozen=True, slots=True)
class _SwiGLUBundle:
    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    gate_bias: Tensor | None
    up_bias: Tensor | None
    down_bias: Tensor | None


@dataclass(frozen=True, slots=True)
class _GQABundle:
    q_weight: Tensor
    k_weight: Tensor
    v_weight: Tensor
    o_weight: Tensor
    q_bias: Tensor | None
    k_bias: Tensor | None
    v_bias: Tensor | None
    o_bias: Tensor | None
    query_heads: int | None
    key_value_heads: int | None
    head_dim: int | None


@dataclass(frozen=True, slots=True)
class _GQATopology:
    query_heads: int
    key_value_heads: int
    head_dim: int
    input_width: int
    output_width: int

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.key_value_heads * self.head_dim

    @property
    def query_rows_per_group(self) -> int:
        return (self.query_heads // self.key_value_heads) * self.head_dim


@dataclass(slots=True)
class _WorkTracker:
    distance_values: int = 0
    maximum_matrix_elements: int = 0
    assignment_blocks: int = 0
    accepted_blocks: int = 0

    def add_matrix(self, rows: int, columns: int) -> None:
        elements = rows * columns
        self.distance_values += elements
        self.maximum_matrix_elements = max(self.maximum_matrix_elements, elements)
        self.assignment_blocks += 1


@dataclass(frozen=True, slots=True)
class _InferenceState:
    permutations: Tensor
    identity_objectives: tuple[float, ...]
    coarse_objectives: tuple[float, ...]
    aligned_objectives: tuple[float, ...]
    identity_selected_layers: int
    tracker: _WorkTracker
    distance_work_bound: int
    distance_work_bound_formula: str
    materialized_full_pairwise_matrix: bool
    exact_small_group_assignment: bool
    algorithm: str
    assignment_solver: str


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise CodebookAlignmentShapeError(f"{name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise CodebookAlignmentShapeError(f"{name} must be a non-negative integer.")
    return int(value)


def _optional_index(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _validate_float_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int | None = None,
    minimum_dimensions: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise CodebookAlignmentShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise CodebookAlignmentShapeError(f"{name} must be a dense non-meta tensor.")
    if dimensions is not None and value.ndim != dimensions:
        raise CodebookAlignmentShapeError(f"{name} must be {dimensions}-dimensional.")
    if minimum_dimensions is not None and value.ndim < minimum_dimensions:
        raise CodebookAlignmentShapeError(
            f"{name} must have at least {minimum_dimensions} dimensions."
        )
    if value.ndim == 0 or any(int(size) <= 0 for size in value.shape):
        raise CodebookAlignmentShapeError(f"{name} must be non-scalar and non-empty.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise CodebookAlignmentShapeError(
            f"{name} must have a real floating-point dtype."
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise CodebookAlignmentFiniteError(f"{name} contains NaN or infinity.")
    return value


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _attribute_value(value: Any, *names: str) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _projection_parts(value: Any, name: str) -> tuple[Tensor, Tensor | None]:
    if isinstance(value, Tensor):
        return value, None
    if isinstance(value, Mapping):
        weight = value.get("weight")
        bias = value.get("bias")
    else:
        weight = getattr(value, "weight", None)
        bias = getattr(value, "bias", None)
    if not isinstance(weight, Tensor) or (
        bias is not None and not isinstance(bias, Tensor)
    ):
        raise CodebookAlignmentShapeError(
            f"{name} must expose tensor weight and optional tensor bias values."
        )
    return weight, bias


def _source_sequence(values: Any, name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, Mapping, Tensor)):
        raise CodebookAlignmentShapeError(f"{name} must be an iterable across depth.")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise CodebookAlignmentShapeError(
            f"{name} must be an iterable across depth."
        ) from exc
    if not result:
        raise CodebookAlignmentShapeError(f"{name} must contain at least one layer.")
    return result


def _coerce_swiglu_bundle(value: Any, layer_index: int) -> _SwiGLUBundle:
    direct_names = ("gate_weight", "up_weight", "down_weight")
    if isinstance(value, Mapping) and all(name in value for name in direct_names):
        return _SwiGLUBundle(
            gate_weight=value["gate_weight"],
            up_weight=value["up_weight"],
            down_weight=value["down_weight"],
            gate_bias=value.get("gate_bias"),
            up_bias=value.get("up_bias"),
            down_bias=value.get("down_bias"),
        )
    if all(hasattr(value, name) for name in direct_names):
        return _SwiGLUBundle(
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
    else:
        gate = _attribute_value(value, "gate_proj", "gate")
        up = _attribute_value(value, "up_proj", "up")
        down = _attribute_value(value, "down_proj", "down")
        if gate is not None and up is not None and down is not None:
            projections = (gate, up, down)
    if (
        projections is None
        and isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
    ):
        if len(value) == 3:
            projections = (value[0], value[1], value[2])
    if projections is None:
        raise CodebookAlignmentShapeError(
            f"bundles[{layer_index}] is not a recognized coupled SwiGLU bundle."
        )
    gate_weight, gate_bias = _projection_parts(
        projections[0], f"bundles[{layer_index}].gate"
    )
    up_weight, up_bias = _projection_parts(projections[1], f"bundles[{layer_index}].up")
    down_weight, down_bias = _projection_parts(
        projections[2], f"bundles[{layer_index}].down"
    )
    return _SwiGLUBundle(
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        gate_bias=gate_bias,
        up_bias=up_bias,
        down_bias=down_bias,
    )


def _validated_swiglu_bundles(values: Any) -> tuple[_SwiGLUBundle, ...]:
    raw = _source_sequence(values, "bundles")
    bundles = tuple(
        _coerce_swiglu_bundle(value, index) for index, value in enumerate(raw)
    )
    reference: tuple[Any, ...] | None = None
    for layer_index, bundle in enumerate(bundles):
        gate = _validate_float_tensor(
            bundle.gate_weight,
            f"bundles[{layer_index}].gate_weight",
            dimensions=2,
        )
        up = _validate_float_tensor(
            bundle.up_weight,
            f"bundles[{layer_index}].up_weight",
            dimensions=2,
        )
        down = _validate_float_tensor(
            bundle.down_weight,
            f"bundles[{layer_index}].down_weight",
            dimensions=2,
        )
        if tuple(gate.shape) != tuple(up.shape):
            raise CodebookAlignmentShapeError(
                f"bundles[{layer_index}] gate and up weights must have equal shape."
            )
        intermediate, input_width = map(int, gate.shape)
        output_width = int(down.shape[0])
        if int(down.shape[1]) != intermediate:
            raise CodebookAlignmentShapeError(
                f"bundles[{layer_index}] down columns must equal intermediate width."
            )
        tensors = [gate, up, down]
        for role, bias, width in (
            ("gate_bias", bundle.gate_bias, intermediate),
            ("up_bias", bundle.up_bias, intermediate),
            ("down_bias", bundle.down_bias, output_width),
        ):
            if bias is not None:
                checked = _validate_float_tensor(
                    bias,
                    f"bundles[{layer_index}].{role}",
                    dimensions=1,
                )
                if tuple(checked.shape) != (width,):
                    raise CodebookAlignmentShapeError(
                        f"bundles[{layer_index}].{role} must have shape ({width},)."
                    )
                tensors.append(checked)
        if any(tensor.dtype != gate.dtype for tensor in tensors):
            raise CodebookAlignmentShapeError(
                f"bundles[{layer_index}] tensors must have one common dtype."
            )
        if any(tensor.device != gate.device for tensor in tensors):
            raise CodebookAlignmentShapeError(
                f"bundles[{layer_index}] tensors must have one common device."
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
            raise CodebookAlignmentShapeError(
                "All SwiGLU bundles must share shape, bias topology, dtype, and device."
            )
    return bundles


def _coerce_gqa_bundle(value: Any, layer_index: int) -> _GQABundle:
    direct_roles = {
        "q": ("q_weight", "query_weight"),
        "k": ("k_weight", "key_weight"),
        "v": ("v_weight", "value_weight"),
        "o": ("o_weight", "output_weight"),
    }
    direct_weights: dict[str, Any] = {}
    direct_biases: dict[str, Any] = {}
    for role, names in direct_roles.items():
        if isinstance(value, Mapping):
            direct_weights[role] = _mapping_value(value, *names)
            direct_biases[role] = _mapping_value(
                value, f"{role}_bias", f"{names[1].split('_')[0]}_bias"
            )
        else:
            direct_weights[role] = _attribute_value(value, *names)
            direct_biases[role] = _attribute_value(
                value, f"{role}_bias", f"{names[1].split('_')[0]}_bias"
            )

    if all(isinstance(direct_weights[role], Tensor) for role in direct_roles):
        weights = direct_weights
        biases = direct_biases
    else:
        projections: dict[str, Any] = {}
        for role in direct_roles:
            names = (f"{role}_proj", role)
            projections[role] = (
                _mapping_value(value, *names)
                if isinstance(value, Mapping)
                else _attribute_value(value, *names)
            )
        projections_complete = all(
            projections[role] is not None for role in direct_roles
        )
        if (
            not projections_complete
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            if len(value) == 4:
                projections = dict(zip(("q", "k", "v", "o"), value, strict=True))
        if not all(projections[role] is not None for role in direct_roles):
            raise CodebookAlignmentShapeError(
                f"bundles[{layer_index}] is not a recognized split-QKV GQA bundle."
            )
        weights = {}
        biases = {}
        for role, projection in projections.items():
            weights[role], biases[role] = _projection_parts(
                projection, f"bundles[{layer_index}].{role}_proj"
            )

    def metadata(*names: str) -> int | None:
        raw = (
            _mapping_value(value, *names)
            if isinstance(value, Mapping)
            else _attribute_value(value, *names)
        )
        return _optional_index(raw, f"bundles[{layer_index}].{names[0]}")

    return _GQABundle(
        q_weight=weights["q"],
        k_weight=weights["k"],
        v_weight=weights["v"],
        o_weight=weights["o"],
        q_bias=biases["q"],
        k_bias=biases["k"],
        v_bias=biases["v"],
        o_bias=biases["o"],
        query_heads=metadata(
            "query_heads", "num_query_heads", "num_attention_heads", "num_heads"
        ),
        key_value_heads=metadata(
            "key_value_heads", "num_key_value_heads", "num_kv_heads"
        ),
        head_dim=metadata("head_dim"),
    )


def _one_topology_value(name: str, *values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    if present and any(value != present[0] for value in present[1:]):
        raise CodebookAlignmentShapeError(
            f"Conflicting values were supplied for {name}."
        )
    return present[0] if present else None


def _resolved_gqa_topology(
    bundles: tuple[_GQABundle, ...],
    *,
    query_heads: int | None,
    key_value_heads: int | None,
    head_dim: int | None,
) -> _GQATopology:
    query = _one_topology_value(
        "query_heads", query_heads, *(bundle.query_heads for bundle in bundles)
    )
    groups = _one_topology_value(
        "key_value_heads",
        key_value_heads,
        *(bundle.key_value_heads for bundle in bundles),
    )
    dimension = _one_topology_value(
        "head_dim", head_dim, *(bundle.head_dim for bundle in bundles)
    )
    q_rows = int(bundles[0].q_weight.shape[0])
    k_rows = int(bundles[0].k_weight.shape[0])
    if dimension is None:
        if groups is not None and k_rows % groups == 0:
            dimension = k_rows // groups
        elif query is not None and q_rows % query == 0:
            dimension = q_rows // query
        else:
            raise CodebookAlignmentShapeError(
                "GQA head_dim cannot be inferred; supply head_dim or head counts."
            )
    dimension = _positive_int(dimension, "head_dim")
    if groups is None:
        if k_rows % dimension:
            raise CodebookAlignmentShapeError(
                "K projection width is not divisible by head_dim."
            )
        groups = k_rows // dimension
    if query is None:
        if q_rows % dimension:
            raise CodebookAlignmentShapeError(
                "Q projection width is not divisible by head_dim."
            )
        query = q_rows // dimension
    query = _positive_int(query, "query_heads")
    groups = _positive_int(groups, "key_value_heads")
    if query % groups:
        raise CodebookAlignmentShapeError(
            "query_heads must be divisible by key_value_heads for coupled GQA groups."
        )
    return _GQATopology(
        query_heads=query,
        key_value_heads=groups,
        head_dim=dimension,
        input_width=int(bundles[0].q_weight.shape[1]),
        output_width=int(bundles[0].o_weight.shape[0]),
    )


def _validated_gqa_bundles(
    values: Any,
    *,
    query_heads: int | None,
    key_value_heads: int | None,
    head_dim: int | None,
) -> tuple[tuple[_GQABundle, ...], _GQATopology]:
    raw = _source_sequence(values, "bundles")
    bundles = tuple(_coerce_gqa_bundle(value, index) for index, value in enumerate(raw))
    for layer_index, bundle in enumerate(bundles):
        for role in ("q", "k", "v", "o"):
            _validate_float_tensor(
                getattr(bundle, f"{role}_weight"),
                f"bundles[{layer_index}].{role}_weight",
                dimensions=2,
            )
    topology = _resolved_gqa_topology(
        bundles,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
    )
    expected_weights = {
        "q_weight": (topology.query_width, topology.input_width),
        "k_weight": (topology.key_value_width, topology.input_width),
        "v_weight": (topology.key_value_width, topology.input_width),
        "o_weight": (topology.output_width, topology.query_width),
    }
    expected_biases = {
        "q_bias": topology.query_width,
        "k_bias": topology.key_value_width,
        "v_bias": topology.key_value_width,
        "o_bias": topology.output_width,
    }
    reference_presence = tuple(
        getattr(bundles[0], name) is not None for name in expected_biases
    )
    reference_dtype = bundles[0].q_weight.dtype
    reference_device = bundles[0].q_weight.device
    for layer_index, bundle in enumerate(bundles):
        tensors: list[Tensor] = []
        for name, expected_shape in expected_weights.items():
            tensor = getattr(bundle, name)
            if tuple(tensor.shape) != expected_shape:
                raise CodebookAlignmentShapeError(
                    f"bundles[{layer_index}].{name} must have shape {expected_shape}."
                )
            tensors.append(tensor)
        presence = tuple(getattr(bundle, name) is not None for name in expected_biases)
        if presence != reference_presence:
            raise CodebookAlignmentShapeError(
                "Q/K/V/O bias presence must be consistent across depth."
            )
        for name, width in expected_biases.items():
            bias = getattr(bundle, name)
            if bias is not None:
                checked = _validate_float_tensor(
                    bias,
                    f"bundles[{layer_index}].{name}",
                    dimensions=1,
                )
                if tuple(checked.shape) != (width,):
                    raise CodebookAlignmentShapeError(
                        f"bundles[{layer_index}].{name} must have shape ({width},)."
                    )
                tensors.append(checked)
        if any(tensor.dtype != reference_dtype for tensor in tensors):
            raise CodebookAlignmentShapeError(
                "All GQA projection tensors must have one common dtype."
            )
        if any(tensor.device != reference_device for tensor in tensors):
            raise CodebookAlignmentShapeError(
                "All GQA projection tensors must have one common device."
            )
    return bundles, topology


def _even_indices(length: int, count: int) -> Tensor:
    if count <= 0:
        return torch.empty(0, dtype=torch.int64)
    if count >= length:
        return torch.arange(length, dtype=torch.int64)
    positions = torch.arange(count, dtype=torch.int64)
    return ((2 * positions + 1) * length) // (2 * count)


def _allocate_dimensions(widths: Sequence[int], total: int) -> tuple[int, ...]:
    target = min(sum(widths), total)
    allocation = [0] * len(widths)
    active = [index for index, width in enumerate(widths) if width > 0]
    for index in active:
        if target == 0:
            break
        allocation[index] = 1
        target -= 1
    while target:
        candidates = [index for index in active if allocation[index] < widths[index]]
        selected = max(
            candidates,
            key=lambda index: (widths[index] / (allocation[index] + 1), -index),
        )
        allocation[selected] += 1
        target -= 1
    return tuple(allocation)


def _take_flat_positions(value: Tensor, positions: Tensor) -> Tensor:
    if int(positions.numel()) == 0:
        return torch.empty(0, dtype=value.dtype, device=value.device)
    positions = positions.to(value.device)
    if value.is_contiguous():
        return value.reshape(-1).index_select(0, positions)
    remainder = positions
    coordinates: list[Tensor] = []
    for size in reversed(value.shape):
        coordinates.append(torch.remainder(remainder, int(size)))
        remainder = torch.div(remainder, int(size), rounding_mode="floor")
    return value[tuple(reversed(coordinates))]


def _sample_flat_parts(parts: Sequence[Tensor], count: int) -> Tensor:
    if count == 0:
        return torch.empty(0, dtype=torch.float64)
    lengths = [int(part.numel()) for part in parts]
    positions = _even_indices(sum(lengths), count)
    result = torch.empty(int(positions.numel()), dtype=torch.float64)
    offset = 0
    for part, length in zip(parts, lengths, strict=True):
        mask = (positions >= offset) & (positions < offset + length)
        output_positions = torch.nonzero(mask, as_tuple=False).flatten()
        if int(output_positions.numel()):
            local = positions.index_select(0, output_positions) - offset
            sampled = _take_flat_positions(part.detach(), local).to(
                "cpu", torch.float64
            )
            result[output_positions] = sampled
        offset += length
    return result


def _swiglu_weight_sketch(
    bundles: tuple[_SwiGLUBundle, ...],
    requested_size: int,
) -> Tensor:
    layers = len(bundles)
    intermediate, input_width = map(int, bundles[0].gate_weight.shape)
    output_width = int(bundles[0].down_weight.shape[0])
    widths = (
        input_width + int(bundles[0].gate_bias is not None),
        input_width + int(bundles[0].up_bias is not None),
        output_width,
    )
    allocation = _allocate_dimensions(widths, requested_size)
    dimensions = sum(allocation)
    result = torch.empty((layers, intermediate, dimensions), dtype=torch.float64)

    def producer(weight: Tensor, bias: Tensor | None, count: int) -> Tensor:
        if count == 0:
            return torch.empty((intermediate, 0), dtype=torch.float64)
        positions = _even_indices(int(weight.shape[1]) + int(bias is not None), count)
        sampled = torch.empty((intermediate, count), dtype=torch.float64)
        weight_mask = positions < int(weight.shape[1])
        weight_outputs = torch.nonzero(weight_mask, as_tuple=False).flatten()
        if int(weight_outputs.numel()):
            columns = positions.index_select(0, weight_outputs).to(weight.device)
            sampled[:, weight_outputs] = (
                weight.detach().index_select(1, columns).to("cpu", torch.float64)
            )
        bias_outputs = torch.nonzero(~weight_mask, as_tuple=False).flatten()
        if int(bias_outputs.numel()):
            if bias is None:  # pragma: no cover - positions make this unreachable
                raise CodebookAlignmentStateError("A sampled producer bias is missing.")
            sampled[:, bias_outputs] = (
                bias.detach().to("cpu", torch.float64).unsqueeze(1)
            )
        return sampled

    for layer_index, bundle in enumerate(bundles):
        down_rows = _even_indices(output_width, allocation[2]).to(
            bundle.down_weight.device
        )
        sampled_down = (
            bundle.down_weight.detach()
            .index_select(0, down_rows)
            .transpose(0, 1)
            .to("cpu", torch.float64)
        )
        result[layer_index] = torch.cat(
            (
                producer(bundle.gate_weight, bundle.gate_bias, allocation[0]),
                producer(bundle.up_weight, bundle.up_bias, allocation[1]),
                sampled_down,
            ),
            dim=1,
        )
    return result


def _activation_layers(
    value: Tensor | Sequence[Tensor] | None,
    logical_layers: int,
    name: str,
) -> tuple[Tensor, ...] | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.ndim < 2 or int(value.shape[0]) != logical_layers:
            raise CodebookAlignmentShapeError(
                f"A stacked {name} tensor must have logical depth first."
            )
        return tuple(value[index] for index in range(logical_layers))
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise CodebookAlignmentShapeError(
            f"{name} must be a tensor or sequence across depth."
        )
    if len(value) != logical_layers:
        raise CodebookAlignmentShapeError(
            f"{name} must contain one tensor per logical layer."
        )
    return tuple(value)


def _swiglu_activation_sketch(
    activations: tuple[Tensor, ...] | None,
    intermediate: int,
    requested_size: int,
) -> Tensor | None:
    if activations is None or requested_size == 0:
        return None
    flattened: list[Tensor] = []
    row_counts: list[int] = []
    for layer_index, value in enumerate(activations):
        checked = _validate_float_tensor(
            value,
            f"activation_sketches[{layer_index}]",
            minimum_dimensions=1,
        )
        if int(checked.shape[-1]) != intermediate:
            raise CodebookAlignmentShapeError(
                f"activation_sketches[{layer_index}] must end in width {intermediate}."
            )
        rows = checked.detach().reshape(-1, intermediate)
        flattened.append(rows)
        row_counts.append(int(rows.shape[0]))
    dimensions = min(requested_size, min(row_counts))
    result = torch.empty(
        (len(flattened), intermediate, dimensions), dtype=torch.float64
    )
    for layer_index, rows in enumerate(flattened):
        positions = _even_indices(int(rows.shape[0]), dimensions).to(rows.device)
        result[layer_index] = (
            rows.index_select(0, positions).transpose(0, 1).to("cpu", torch.float64)
        )
    return result


def _gqa_weight_sketch(
    bundles: tuple[_GQABundle, ...],
    topology: _GQATopology,
    requested_size: int,
) -> Tensor:
    groups = topology.key_value_heads
    q_rows = topology.query_rows_per_group
    head_dim = topology.head_dim
    widths = (
        q_rows * topology.input_width + q_rows * int(bundles[0].q_bias is not None),
        head_dim * topology.input_width + head_dim * int(bundles[0].k_bias is not None),
        head_dim * topology.input_width + head_dim * int(bundles[0].v_bias is not None),
        topology.output_width * q_rows,
    )
    allocation = _allocate_dimensions(widths, requested_size)
    dimensions = sum(allocation)
    result = torch.empty((len(bundles), groups, dimensions), dtype=torch.float64)
    for layer_index, bundle in enumerate(bundles):
        q_weight = bundle.q_weight.reshape(groups, q_rows, topology.input_width)
        k_weight = bundle.k_weight.reshape(groups, head_dim, topology.input_width)
        v_weight = bundle.v_weight.reshape(groups, head_dim, topology.input_width)
        o_weight = bundle.o_weight.reshape(topology.output_width, groups, q_rows)
        q_bias = (
            bundle.q_bias.reshape(groups, q_rows) if bundle.q_bias is not None else None
        )
        k_bias = (
            bundle.k_bias.reshape(groups, head_dim)
            if bundle.k_bias is not None
            else None
        )
        v_bias = (
            bundle.v_bias.reshape(groups, head_dim)
            if bundle.v_bias is not None
            else None
        )
        for group_index in range(groups):
            q_parts = [q_weight[group_index]]
            k_parts = [k_weight[group_index]]
            v_parts = [v_weight[group_index]]
            if q_bias is not None:
                q_parts.append(q_bias[group_index])
            if k_bias is not None:
                k_parts.append(k_bias[group_index])
            if v_bias is not None:
                v_parts.append(v_bias[group_index])
            pieces = (
                _sample_flat_parts(q_parts, allocation[0]),
                _sample_flat_parts(k_parts, allocation[1]),
                _sample_flat_parts(v_parts, allocation[2]),
                _sample_flat_parts([o_weight[:, group_index, :]], allocation[3]),
            )
            result[layer_index, group_index] = torch.cat(pieces)
    return result


def _gqa_activation_width(value: Tensor, topology: _GQATopology) -> int:
    groups = topology.key_value_heads
    if int(value.shape[-1]) == groups:
        return int(value.numel()) // groups
    if value.ndim >= 2 and int(value.shape[-2]) == groups:
        return int(value.numel()) // groups
    if int(value.shape[-1]) in (topology.query_width, topology.key_value_width):
        return int(value.numel()) // groups
    raise CodebookAlignmentShapeError(
        "GQA activation sketches must have groups on the last or penultimate axis, "
        "or a packed Q/K/V projection width on the last axis."
    )


def _sample_gqa_activation(
    value: Tensor,
    topology: _GQATopology,
    count: int,
) -> Tensor:
    groups = topology.key_value_heads
    last = int(value.shape[-1])
    positions = _even_indices(_gqa_activation_width(value, topology), count)
    if last == groups:
        rows = value.detach().reshape(-1, groups)
        return (
            rows.index_select(0, positions.to(rows.device))
            .transpose(0, 1)
            .to("cpu", torch.float64)
        )
    if value.ndim >= 2 and int(value.shape[-2]) == groups:
        rows = value.detach().reshape(-1, groups, last)
    elif last in (topology.query_width, topology.key_value_width):
        grouped_width = last // groups
        rows = value.detach().reshape(-1, groups, grouped_width)
    else:
        rows = value.detach().reshape(-1, groups, last)
    return torch.stack(
        [
            _take_flat_positions(rows[:, group_index, :], positions).to(
                "cpu", torch.float64
            )
            for group_index in range(groups)
        ]
    )


def _gqa_activation_sketch(
    activations: tuple[Tensor, ...] | None,
    topology: _GQATopology,
    requested_size: int,
) -> Tensor | None:
    if activations is None or requested_size == 0:
        return None
    checked_values: list[Tensor] = []
    widths: list[int] = []
    for layer_index, value in enumerate(activations):
        checked = _validate_float_tensor(
            value,
            f"activation_sketches[{layer_index}]",
            minimum_dimensions=1,
        )
        checked_values.append(checked)
        widths.append(_gqa_activation_width(checked, topology))
    dimensions = min(requested_size, min(widths))
    return torch.stack(
        [
            _sample_gqa_activation(value, topology, dimensions)
            for value in checked_values
        ]
    )


def _standardized_features(weight: Tensor, activation: Tensor | None) -> Tensor:
    blocks = [weight] if activation is None else [weight, activation]
    normalized_blocks: list[Tensor] = []
    for block in blocks:
        if int(block.shape[-1]) == 0:
            continue
        flattened = block.reshape(-1, int(block.shape[-1]))
        centered = block - flattened.mean(dim=0).reshape(1, 1, -1)
        rms = flattened.sub(flattened.mean(dim=0)).square().mean(dim=0).sqrt()
        floor = torch.finfo(torch.float64).eps
        normalized = torch.where(
            rms.reshape(1, 1, -1) > floor,
            centered / rms.clamp_min(floor).reshape(1, 1, -1),
            torch.zeros_like(centered),
        )
        normalized_blocks.append(normalized / math.sqrt(int(block.shape[-1])))
    if not normalized_blocks:
        raise CodebookAlignmentShapeError(
            "At least one non-empty weight or activation sketch is required."
        )
    result = torch.cat(normalized_blocks, dim=-1).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise CodebookAlignmentFiniteError(
            "Standardizing activation/weight sketches produced non-finite values."
        )
    return result


def _squared_distances(points: Tensor, centers: Tensor) -> Tensor:
    distances = points.square().sum(dim=1, keepdim=True)
    distances = distances + centers.square().sum(dim=1).unsqueeze(0)
    distances.addmm_(points, centers.transpose(0, 1), beta=1.0, alpha=-2.0)
    distances.clamp_min_(0.0)
    if not bool(torch.isfinite(distances).all()):
        raise CodebookAlignmentFiniteError("A sketch-distance block is non-finite.")
    return distances


def _layer_objective(reference: Tensor, target: Tensor, permutation: Tensor) -> float:
    aligned = target.index_select(0, permutation)
    value = float((reference - aligned).square().sum().item())
    if not math.isfinite(value):
        raise CodebookAlignmentFiniteError("An alignment objective is non-finite.")
    return value


def _morton_order(
    features: Tensor,
    signature_dimensions: int,
    bits_per_dimension: int,
) -> tuple[Tensor, int]:
    feature_dimensions = int(features.shape[1])
    dimensions = min(
        signature_dimensions,
        feature_dimensions,
        max(1, 62 // bits_per_dimension),
    )
    positions = _even_indices(feature_dimensions, dimensions)
    selected = features.index_select(1, positions)
    maximum_bin = (1 << bits_per_dimension) - 1
    clipped = selected.clamp(-4.0, 4.0)
    quantized = torch.round((clipped + 4.0) * (maximum_bin / 8.0)).to(torch.int64)
    codes = torch.zeros(int(features.shape[0]), dtype=torch.int64)
    for bit in range(bits_per_dimension):
        for dimension in range(dimensions):
            source_bit = torch.bitwise_and(
                torch.bitwise_right_shift(quantized[:, dimension], bit), 1
            )
            codes |= torch.bitwise_left_shift(source_bit, bit * dimensions + dimension)

    coordinate = torch.arange(feature_dimensions, dtype=torch.int64)
    coefficients = (
        torch.remainder(coordinate * 104729 + 8191, 65521).to(torch.float64) / 32760.5
        - 1.0
    )
    secondary = features.mv(coefficients)
    norms = features.square().sum(dim=1)
    order = torch.argsort(secondary, stable=True)
    order = order.index_select(0, torch.argsort(norms[order], stable=True))
    order = order.index_select(0, torch.argsort(codes[order], stable=True))
    return order, dimensions


def _coarse_permutation(reference_order: Tensor, target_order: Tensor) -> Tensor:
    result = torch.empty_like(reference_order)
    result[reference_order] = target_order
    return result


def _deterministic_greedy_assignment(costs: Tensor) -> Tensor:
    size = int(costs.shape[0])
    if tuple(costs.shape) != (size, size):
        raise CodebookAlignmentStateError("Local assignment costs must be square.")
    remaining = costs.clone()
    assignment = torch.empty(size, dtype=torch.int64)
    for _ in range(size):
        flat_index = int(torch.argmin(remaining).item())
        row, column = divmod(flat_index, size)
        assignment[row] = column
        remaining[row, :] = math.inf
        remaining[:, column] = math.inf
    return assignment


def _hungarian_assignment(costs: Tensor) -> Tensor:
    """Return a deterministic exact square assignment using raw Torch."""

    size = int(costs.shape[0])
    if tuple(costs.shape) != (size, size):
        raise CodebookAlignmentStateError("Hungarian costs must be square.")
    u = torch.zeros(size + 1, dtype=torch.float64)
    v = torch.zeros(size + 1, dtype=torch.float64)
    p = torch.zeros(size + 1, dtype=torch.int64)
    way = torch.zeros(size + 1, dtype=torch.int64)
    for row in range(1, size + 1):
        p[0] = row
        minimum = torch.full((size + 1,), math.inf, dtype=torch.float64)
        used = torch.zeros(size + 1, dtype=torch.bool)
        column = 0
        while True:
            used[column] = True
            active_row = int(p[column].item())
            reduced = costs[active_row - 1] - u[active_row] - v[1:]
            unused = ~used[1:]
            improve = unused & (reduced < minimum[1:])
            minimum[1:][improve] = reduced[improve]
            way[1:][improve] = column
            candidates = torch.where(unused, minimum[1:], math.inf)
            delta, zero_based_column = torch.min(candidates, dim=0)
            if not bool(torch.isfinite(delta)):
                raise CodebookAlignmentStateError(
                    "Hungarian assignment became singular."
                )
            used_columns = torch.nonzero(used, as_tuple=False).flatten()
            u[p.index_select(0, used_columns)] += delta
            v[used_columns] -= delta
            minimum[1:][unused] -= delta
            column = int(zero_based_column.item()) + 1
            if int(p[column].item()) == 0:
                break
        while True:
            previous = int(way[column].item())
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = torch.empty(size, dtype=torch.int64)
    for column in range(1, size + 1):
        assignment[int(p[column].item()) - 1] = column - 1
    return assignment


def _partition_bounds(
    length: int, block_size: int, offset: int
) -> tuple[tuple[int, int], ...]:
    boundaries = [0]
    if 0 < offset < length:
        boundaries.append(offset)
    cursor = offset
    while cursor < length:
        cursor = min(length, cursor + block_size)
        if cursor > boundaries[-1]:
            boundaries.append(cursor)
    if boundaries[-1] != length:
        boundaries.append(length)
    return tuple(
        (start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start
    )


def _refine_permutation(
    reference: Tensor,
    target: Tensor,
    reference_order: Tensor,
    initial: Tensor,
    *,
    block_size: int,
    refinement_passes: int,
    tracker: _WorkTracker,
) -> Tensor:
    permutation = initial.clone()
    width = int(reference.shape[0])
    block_size = min(block_size, width)
    if refinement_passes == 0 or block_size <= 1:
        return permutation
    offsets = (0,) if block_size == 1 else (0, block_size // 2)
    for _ in range(refinement_passes):
        for offset in offsets:
            for start, end in _partition_bounds(width, block_size, offset):
                if end - start <= 1:
                    continue
                canonical = torch.sort(reference_order[start:end]).values
                physical = torch.sort(permutation.index_select(0, canonical)).values
                costs = _squared_distances(
                    reference.index_select(0, canonical),
                    target.index_select(0, physical),
                )
                tracker.add_matrix(end - start, end - start)
                assignment = _deterministic_greedy_assignment(costs)
                proposed = physical.index_select(0, assignment)
                current = permutation.index_select(0, canonical)
                row_indices = torch.arange(end - start, dtype=torch.int64)
                proposed_cost = float(costs[row_indices, assignment].sum().item())
                current_columns = torch.searchsorted(physical, current)
                current_cost = float(costs[row_indices, current_columns].sum().item())
                tolerance = 1e-12 * max(1.0, abs(current_cost), abs(proposed_cost))
                if proposed_cost < current_cost - tolerance:
                    permutation[canonical] = proposed
                    tracker.accepted_blocks += 1
    return permutation


def _identity_is_no_worse(identity: float, candidate: float) -> bool:
    tolerance = 1e-12 * max(1.0, abs(identity), abs(candidate))
    return identity <= candidate + tolerance


def _infer_with_sorted_signatures(
    features: Tensor,
    *,
    reference_layer: int,
    signature_dimensions: int,
    signature_bits: int,
    local_block_size: int,
    refinement_passes: int,
) -> tuple[_InferenceState, int]:
    layers, width, _ = map(int, features.shape)
    identity = torch.arange(width, dtype=torch.int64)
    permutations = identity.repeat(layers, 1)
    orders: list[Tensor] = []
    actual_signature_dimensions = 0
    for layer_index in range(layers):
        order, actual_dimensions = _morton_order(
            features[layer_index], signature_dimensions, signature_bits
        )
        orders.append(order)
        actual_signature_dimensions = actual_dimensions

    tracker = _WorkTracker()
    identity_objectives: list[float] = []
    coarse_objectives: list[float] = []
    aligned_objectives: list[float] = []
    identity_selected = 0
    reference = features[reference_layer]
    reference_order = orders[reference_layer]
    for layer_index in range(layers):
        target = features[layer_index]
        identity_objective = _layer_objective(reference, target, identity)
        if layer_index == reference_layer:
            coarse = identity.clone()
            refined = identity.clone()
        else:
            coarse = _coarse_permutation(reference_order, orders[layer_index])
            refined = _refine_permutation(
                reference,
                target,
                reference_order,
                coarse,
                block_size=local_block_size,
                refinement_passes=refinement_passes,
                tracker=tracker,
            )
        coarse_objective = _layer_objective(reference, target, coarse)
        refined_objective = _layer_objective(reference, target, refined)
        if layer_index != reference_layer and _identity_is_no_worse(
            identity_objective, refined_objective
        ):
            if not torch.equal(refined, identity):
                identity_selected += 1
            refined = identity.clone()
            refined_objective = identity_objective
        permutations[layer_index] = refined
        identity_objectives.append(identity_objective)
        coarse_objectives.append(coarse_objective)
        aligned_objectives.append(refined_objective)

    partitions_per_pass = 1 if min(local_block_size, width) <= 1 else 2
    work_bound = (
        max(0, layers - 1)
        * refinement_passes
        * partitions_per_pass
        * width
        * min(local_block_size, width)
    )
    if tracker.distance_values > work_bound:
        raise CodebookAlignmentStateError(
            "Local refinement exceeded its declared distance-work bound."
        )
    return (
        _InferenceState(
            permutations=permutations,
            identity_objectives=tuple(identity_objectives),
            coarse_objectives=tuple(coarse_objectives),
            aligned_objectives=tuple(aligned_objectives),
            identity_selected_layers=identity_selected,
            tracker=tracker,
            distance_work_bound=work_bound,
            distance_work_bound_formula=(
                "(layers - 1) * refinement_passes * partitions_per_pass * "
                "width * local_block_size"
            ),
            materialized_full_pairwise_matrix=False,
            exact_small_group_assignment=False,
            algorithm="deterministic_morton_sorted_bounded_local_assignment",
            assignment_solver="bounded_greedy_local_bijection",
        ),
        actual_signature_dimensions,
    )


def _infer_exact_small_groups(
    features: Tensor,
    *,
    reference_layer: int,
) -> _InferenceState:
    layers, groups, _ = map(int, features.shape)
    identity = torch.arange(groups, dtype=torch.int64)
    permutations = identity.repeat(layers, 1)
    reference = features[reference_layer]
    tracker = _WorkTracker()
    identity_objectives: list[float] = []
    coarse_objectives: list[float] = []
    aligned_objectives: list[float] = []
    identity_selected = 0
    for layer_index in range(layers):
        target = features[layer_index]
        identity_objective = _layer_objective(reference, target, identity)
        if layer_index == reference_layer:
            assignment = identity.clone()
            assigned_objective = 0.0
        else:
            costs = _squared_distances(reference, target)
            tracker.add_matrix(groups, groups)
            assignment = _hungarian_assignment(costs)
            rows = torch.arange(groups, dtype=torch.int64)
            assigned_objective = float(costs[rows, assignment].sum().item())
            if _identity_is_no_worse(identity_objective, assigned_objective):
                if not torch.equal(assignment, identity):
                    identity_selected += 1
                assignment = identity.clone()
                assigned_objective = identity_objective
        assigned_objective = _layer_objective(reference, target, assignment)
        permutations[layer_index] = assignment
        identity_objectives.append(identity_objective)
        coarse_objectives.append(assigned_objective)
        aligned_objectives.append(assigned_objective)
    work_bound = max(0, layers - 1) * groups * groups
    if tracker.distance_values != work_bound:
        raise CodebookAlignmentStateError(
            "Exact GQA assignment violated its distance-work proof."
        )
    return _InferenceState(
        permutations=permutations,
        identity_objectives=tuple(identity_objectives),
        coarse_objectives=tuple(coarse_objectives),
        aligned_objectives=tuple(aligned_objectives),
        identity_selected_layers=identity_selected,
        tracker=tracker,
        distance_work_bound=work_bound,
        distance_work_bound_formula="(layers - 1) * key_value_groups**2",
        materialized_full_pairwise_matrix=True,
        exact_small_group_assignment=True,
        algorithm="deterministic_exact_small_gqa_hungarian",
        assignment_solver="exact_hungarian_lowest_index_ties",
    )


def _validated_reference_layer(value: Any, logical_layers: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CodebookAlignmentShapeError("reference_layer must be an integer.")
    result = int(value)
    if result < 0 or result >= logical_layers:
        raise CodebookAlignmentShapeError(
            f"reference_layer must be in range({logical_layers})."
        )
    return result


def _validated_permutation_matrix(value: Any, name: str) -> tuple[Tensor, bool]:
    squeeze = False
    if isinstance(value, Tensor):
        if (
            value.layout != torch.strided
            or value.device.type == "meta"
            or value.dtype not in _INTEGER_DTYPES
        ):
            raise CodebookAlignmentShapeError(f"{name} must be a dense integer tensor.")
        result = value.detach().to("cpu", torch.int64)
    else:
        if isinstance(value, (str, bytes, Mapping)):
            raise CodebookAlignmentShapeError(f"{name} must be an integer matrix.")
        try:
            rows = tuple(value)
        except TypeError as exc:
            raise CodebookAlignmentShapeError(
                f"{name} must be an integer matrix."
            ) from exc
        if rows and all(
            isinstance(entry, Integral) and not isinstance(entry, bool)
            for entry in rows
        ):
            result = torch.tensor(rows, dtype=torch.int64)
        else:
            normalized: list[tuple[int, ...]] = []
            for row in rows:
                if isinstance(row, Tensor):
                    if row.ndim != 1 or row.dtype not in _INTEGER_DTYPES:
                        raise CodebookAlignmentShapeError(
                            f"Every {name} row must be an integer vector."
                        )
                    entries = tuple(int(entry) for entry in row.detach().to("cpu"))
                else:
                    try:
                        raw_entries = tuple(row)
                    except TypeError as exc:
                        raise CodebookAlignmentShapeError(
                            f"Every {name} row must be an integer vector."
                        ) from exc
                    if any(
                        isinstance(entry, bool) or not isinstance(entry, Integral)
                        for entry in raw_entries
                    ):
                        raise CodebookAlignmentShapeError(
                            f"Every {name} value must be an integer."
                        )
                    entries = tuple(int(entry) for entry in raw_entries)
                normalized.append(entries)
            try:
                result = torch.tensor(normalized, dtype=torch.int64)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CodebookAlignmentShapeError(
                    f"{name} must be a rectangular integer matrix."
                ) from exc
    if result.ndim == 1:
        result = result.unsqueeze(0)
        squeeze = True
    if result.ndim != 2 or any(int(size) <= 0 for size in result.shape):
        raise CodebookAlignmentShapeError(f"{name} must be a non-empty integer matrix.")
    return result.contiguous(), squeeze


def invert_permutations(permutations: Any) -> Tensor:
    """Validate canonical-to-physical rows and return physical-to-canonical rows."""

    matrix, squeeze = _validated_permutation_matrix(permutations, "permutations")
    layers, width = map(int, matrix.shape)
    expected = torch.arange(width, dtype=torch.int64)
    inverse = torch.empty_like(matrix)
    for layer_index, row in enumerate(matrix):
        if not torch.equal(torch.sort(row).values, expected):
            raise CodebookAlignmentShapeError(
                f"permutations[{layer_index}] must be a bijection of range({width})."
            )
        inverse[layer_index, row] = expected
    canonical_round_trip = inverse.gather(1, matrix)
    physical_round_trip = matrix.gather(1, inverse)
    proof = expected.repeat(layers, 1)
    if not torch.equal(canonical_round_trip, proof) or not torch.equal(
        physical_round_trip, proof
    ):
        raise CodebookAlignmentStateError(
            "Permutation inverse round-trip proof failed."
        )
    return inverse[0] if squeeze else inverse


def verify_permutation_bijections(permutations: Any) -> bool:
    """Fail closed on malformed rows and return ``True`` after both proofs pass."""

    invert_permutations(permutations)
    return True


def _result_from_state(
    state: _InferenceState,
    features: Tensor,
    *,
    topology: str,
    reference_layer: int,
    requested_weight_size: int,
    weight_dimensions: int,
    requested_activation_size: int,
    activation_dimensions: int,
    signature_dimensions: int,
    signature_bits: int,
    local_block_size: int,
    refinement_passes: int,
) -> PermutationAlignmentResult:
    permutations = state.permutations.to("cpu", torch.int64).contiguous()
    inverse = invert_permutations(permutations)
    layers, units, dimensions = map(int, features.shape)
    identity_objective = math.fsum(state.identity_objectives)
    coarse_objective = math.fsum(state.coarse_objectives)
    aligned_objective = math.fsum(state.aligned_objectives)
    improvement = max(0.0, identity_objective - aligned_objective)
    if identity_objective > 0.0:
        objective_ratio = aligned_objective / identity_objective
        improvement_fraction = improvement / identity_objective
    else:
        objective_ratio = 1.0 if aligned_objective == 0.0 else math.inf
        improvement_fraction = 0.0
    block_limit = (
        units * units
        if state.exact_small_group_assignment
        else min(local_block_size, units) ** 2
    )
    bounded = (
        state.tracker.distance_values <= state.distance_work_bound
        and state.tracker.maximum_matrix_elements <= block_limit
        and (
            state.exact_small_group_assignment
            or not state.materialized_full_pairwise_matrix
        )
    )
    diagnostics = PermutationAlignmentDiagnostics(
        topology=topology,
        algorithm=state.algorithm,
        assignment_solver=state.assignment_solver,
        logical_layers=layers,
        units_per_layer=units,
        reference_layer=reference_layer,
        deterministic=True,
        tie_break=_TIE_BREAK,
        fitting_requires_grad=False,
        used_weight_sketches=weight_dimensions > 0,
        used_activation_sketches=activation_dimensions > 0,
        requested_weight_sketch_size=requested_weight_size,
        weight_sketch_dimensions=weight_dimensions,
        requested_activation_sketch_size=requested_activation_size,
        activation_sketch_dimensions=activation_dimensions,
        combined_sketch_dimensions=dimensions,
        sketch_elements=int(features.numel()),
        coarse_signature_dimensions=signature_dimensions,
        coarse_bits_per_dimension=signature_bits,
        local_block_size=min(local_block_size, units),
        refinement_passes=refinement_passes,
        local_assignment_blocks=state.tracker.assignment_blocks,
        accepted_refinement_blocks=state.tracker.accepted_blocks,
        exact_small_group_assignment=state.exact_small_group_assignment,
        identity_baseline_selected_layers=state.identity_selected_layers,
        pairwise_distance_values=state.tracker.distance_values,
        distance_feature_operations=state.tracker.distance_values * dimensions,
        distance_work_bound=state.distance_work_bound,
        distance_work_bound_formula=state.distance_work_bound_formula,
        maximum_pairwise_block_elements=state.tracker.maximum_matrix_elements,
        pairwise_block_element_limit=block_limit,
        full_pairwise_matrix_elements=units * units,
        materialized_full_pairwise_matrix=state.materialized_full_pairwise_matrix,
        bounded_work=bounded,
        identity_objective=identity_objective,
        coarse_objective=coarse_objective,
        aligned_objective=aligned_objective,
        objective_improvement=improvement,
        objective_improvement_fraction=improvement_fraction,
        objective_ratio=objective_ratio,
        per_layer_identity_objective=state.identity_objectives,
        per_layer_coarse_objective=state.coarse_objectives,
        per_layer_aligned_objective=state.aligned_objectives,
        permutations_bijective=True,
        inverses_verified=True,
        permutation_rows_verified=layers,
        permutation_proof_method=(
            "sorted_rows_equal_arange_and_bidirectional_gather_round_trip"
        ),
    )
    if not diagnostics.objective_better_than_or_equal_to_identity:
        raise CodebookAlignmentStateError(
            "Alignment objective is worse than the retained identity baseline."
        )
    if not diagnostics.bounded_work:
        raise CodebookAlignmentStateError("Alignment exceeded its bounded-work proof.")
    return PermutationAlignmentResult(permutations, inverse, diagnostics)


def _resolved_alias_size(
    primary: Any,
    alias: Any,
    *,
    default: int,
    name: str,
) -> int:
    if alias is None:
        return _positive_int(primary, name)
    alias_value = _positive_int(alias, name)
    primary_value = _positive_int(primary, name)
    if primary_value != default and primary_value != alias_value:
        raise CodebookAlignmentShapeError(
            f"Conflicting values were supplied for {name}."
        )
    return alias_value


def _resolved_alias_nonnegative_size(
    primary: Any,
    alias: Any,
    *,
    default: int,
    name: str,
) -> int:
    if alias is None:
        return _nonnegative_int(primary, name)
    alias_value = _nonnegative_int(alias, name)
    primary_value = _nonnegative_int(primary, name)
    if primary_value != default and primary_value != alias_value:
        raise CodebookAlignmentShapeError(
            f"Conflicting values were supplied for {name}."
        )
    return alias_value


def infer_swiglu_neuron_permutations(
    bundles: Sequence[Any],
    *,
    activation_sketches: Tensor | Sequence[Tensor] | None = None,
    activations: Tensor | Sequence[Tensor] | None = None,
    reference_layer: int = 0,
    weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
    activation_sketch_size: int = DEFAULT_ACTIVATION_SKETCH_SIZE,
    coarse_signature_dimensions: int = DEFAULT_SIGNATURE_DIMENSIONS,
    coarse_bits_per_dimension: int = DEFAULT_SIGNATURE_BITS,
    local_block_size: int = DEFAULT_LOCAL_BLOCK_SIZE,
    local_window_size: int | None = None,
    refinement_passes: int = DEFAULT_REFINEMENT_PASSES,
) -> PermutationAlignmentResult:
    """Infer coupled gate/up-row and down-column permutations across depth.

    A result row maps a canonical neuron index in ``reference_layer`` to the
    physical neuron index in that row's layer.  At most
    ``local_block_size**2`` distances are materialized, so Qwen3-4B's 9728
    intermediate neurons never create a 9728 by 9728 matrix.
    """

    if activation_sketches is not None and activations is not None:
        raise CodebookAlignmentShapeError(
            "Specify activation_sketches or activations, not both."
        )
    weight_size = _nonnegative_int(weight_sketch_size, "weight_sketch_size")
    activation_size = _nonnegative_int(activation_sketch_size, "activation_sketch_size")
    signature_dimensions = _positive_int(
        coarse_signature_dimensions, "coarse_signature_dimensions"
    )
    signature_bits = _positive_int(
        coarse_bits_per_dimension, "coarse_bits_per_dimension"
    )
    if signature_bits > 15:
        raise CodebookAlignmentShapeError("coarse_bits_per_dimension cannot exceed 15.")
    block_size = _resolved_alias_size(
        local_block_size,
        local_window_size,
        default=DEFAULT_LOCAL_BLOCK_SIZE,
        name="local_block_size",
    )
    passes = _nonnegative_int(refinement_passes, "refinement_passes")
    activation_values = (
        activation_sketches if activation_sketches is not None else activations
    )
    with torch.no_grad():
        sources = _validated_swiglu_bundles(bundles)
        layers = len(sources)
        intermediate = int(sources[0].gate_weight.shape[0])
        reference = _validated_reference_layer(reference_layer, layers)
        activation_layers = _activation_layers(
            activation_values, layers, "activation_sketches"
        )
        if weight_size == 0 and (activation_layers is None or activation_size == 0):
            raise CodebookAlignmentShapeError(
                "At least one weight or activation sketch dimension is required."
            )
        weight = _swiglu_weight_sketch(sources, weight_size)
        activation = _swiglu_activation_sketch(
            activation_layers, intermediate, activation_size
        )
        features = _standardized_features(weight, activation)
        state, actual_signature_dimensions = _infer_with_sorted_signatures(
            features,
            reference_layer=reference,
            signature_dimensions=signature_dimensions,
            signature_bits=signature_bits,
            local_block_size=block_size,
            refinement_passes=passes,
        )
        return _result_from_state(
            state,
            features,
            topology="coupled_swiglu_neurons",
            reference_layer=reference,
            requested_weight_size=weight_size,
            weight_dimensions=int(weight.shape[-1]),
            requested_activation_size=activation_size,
            activation_dimensions=(
                int(activation.shape[-1]) if activation is not None else 0
            ),
            signature_dimensions=actual_signature_dimensions,
            signature_bits=signature_bits,
            local_block_size=block_size,
            refinement_passes=passes,
        )


def infer_qwen_gqa_group_permutations(
    bundles: Sequence[Any],
    *,
    query_heads: int | None = None,
    num_query_heads: int | None = None,
    num_attention_heads: int | None = None,
    num_heads: int | None = None,
    key_value_heads: int | None = None,
    num_key_value_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    activation_sketches: Tensor | Sequence[Tensor] | None = None,
    activations: Tensor | Sequence[Tensor] | None = None,
    reference_layer: int = 0,
    weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
    sketch_dim: int | None = None,
    activation_sketch_size: int = DEFAULT_ACTIVATION_SKETCH_SIZE,
    max_exact_groups: int = DEFAULT_MAX_EXACT_GQA_GROUPS,
    coarse_signature_dimensions: int = DEFAULT_SIGNATURE_DIMENSIONS,
    coarse_bits_per_dimension: int = DEFAULT_SIGNATURE_BITS,
    local_block_size: int = DEFAULT_LOCAL_BLOCK_SIZE,
    local_window_size: int | None = None,
    refinement_passes: int = DEFAULT_REFINEMENT_PASSES,
) -> PermutationAlignmentResult:
    """Infer canonical-to-physical permutations of coupled Q/K/V/O GQA groups."""

    if activation_sketches is not None and activations is not None:
        raise CodebookAlignmentShapeError(
            "Specify activation_sketches or activations, not both."
        )
    resolved_query = _one_topology_value(
        "query_heads",
        _optional_index(query_heads, "query_heads"),
        _optional_index(num_query_heads, "num_query_heads"),
        _optional_index(num_attention_heads, "num_attention_heads"),
        _optional_index(num_heads, "num_heads"),
    )
    resolved_groups = _one_topology_value(
        "key_value_heads",
        _optional_index(key_value_heads, "key_value_heads"),
        _optional_index(num_key_value_heads, "num_key_value_heads"),
        _optional_index(num_kv_heads, "num_kv_heads"),
    )
    resolved_dim = _optional_index(head_dim, "head_dim")
    weight_size = _resolved_alias_nonnegative_size(
        weight_sketch_size,
        sketch_dim,
        default=DEFAULT_WEIGHT_SKETCH_SIZE,
        name="weight_sketch_size",
    )
    activation_size = _nonnegative_int(activation_sketch_size, "activation_sketch_size")
    exact_limit = _positive_int(max_exact_groups, "max_exact_groups")
    signature_dimensions = _positive_int(
        coarse_signature_dimensions, "coarse_signature_dimensions"
    )
    signature_bits = _positive_int(
        coarse_bits_per_dimension, "coarse_bits_per_dimension"
    )
    if signature_bits > 15:
        raise CodebookAlignmentShapeError("coarse_bits_per_dimension cannot exceed 15.")
    block_size = _resolved_alias_size(
        local_block_size,
        local_window_size,
        default=DEFAULT_LOCAL_BLOCK_SIZE,
        name="local_block_size",
    )
    passes = _nonnegative_int(refinement_passes, "refinement_passes")
    activation_values = (
        activation_sketches if activation_sketches is not None else activations
    )
    with torch.no_grad():
        sources, topology = _validated_gqa_bundles(
            bundles,
            query_heads=resolved_query,
            key_value_heads=resolved_groups,
            head_dim=resolved_dim,
        )
        layers = len(sources)
        reference = _validated_reference_layer(reference_layer, layers)
        activation_layers = _activation_layers(
            activation_values, layers, "activation_sketches"
        )
        if weight_size == 0 and (activation_layers is None or activation_size == 0):
            raise CodebookAlignmentShapeError(
                "At least one weight or activation sketch dimension is required."
            )
        weight = _gqa_weight_sketch(sources, topology, weight_size)
        activation = _gqa_activation_sketch(
            activation_layers, topology, activation_size
        )
        features = _standardized_features(weight, activation)
        if topology.key_value_heads <= exact_limit:
            state = _infer_exact_small_groups(features, reference_layer=reference)
            actual_signature_dimensions = 0
            result_block_size = topology.key_value_heads
            result_passes = 0
        else:
            state, actual_signature_dimensions = _infer_with_sorted_signatures(
                features,
                reference_layer=reference,
                signature_dimensions=signature_dimensions,
                signature_bits=signature_bits,
                local_block_size=block_size,
                refinement_passes=passes,
            )
            result_block_size = block_size
            result_passes = passes
        return _result_from_state(
            state,
            features,
            topology="qwen_gqa_groups",
            reference_layer=reference,
            requested_weight_size=weight_size,
            weight_dimensions=int(weight.shape[-1]),
            requested_activation_size=activation_size,
            activation_dimensions=(
                int(activation.shape[-1]) if activation is not None else 0
            ),
            signature_dimensions=actual_signature_dimensions,
            signature_bits=signature_bits,
            local_block_size=result_block_size,
            refinement_passes=result_passes,
        )


infer_swiglu_permutations = infer_swiglu_neuron_permutations
infer_neuron_permutations = infer_swiglu_neuron_permutations
infer_swiglu_codebook_permutations = infer_swiglu_neuron_permutations
infer_gqa_group_permutations = infer_qwen_gqa_group_permutations
infer_attention_group_permutations = infer_qwen_gqa_group_permutations
infer_qwen_gqa_permutations = infer_qwen_gqa_group_permutations
infer_gqa_codebook_permutations = infer_qwen_gqa_group_permutations


__all__ = [
    "CodebookAlignmentDiagnostics",
    "CodebookAlignmentError",
    "CodebookAlignmentFiniteError",
    "CodebookAlignmentResult",
    "CodebookAlignmentShapeError",
    "CodebookAlignmentStateError",
    "CodebookPermutationAlignment",
    "DEFAULT_ACTIVATION_SKETCH_SIZE",
    "DEFAULT_LOCAL_BLOCK_SIZE",
    "DEFAULT_MAX_EXACT_GQA_GROUPS",
    "DEFAULT_REFINEMENT_PASSES",
    "DEFAULT_SIGNATURE_BITS",
    "DEFAULT_SIGNATURE_DIMENSIONS",
    "DEFAULT_WEIGHT_SKETCH_SIZE",
    "PermutationAlignmentDiagnostics",
    "PermutationAlignmentResult",
    "infer_attention_group_permutations",
    "infer_gqa_codebook_permutations",
    "infer_gqa_group_permutations",
    "infer_neuron_permutations",
    "infer_qwen_gqa_group_permutations",
    "infer_qwen_gqa_permutations",
    "infer_swiglu_codebook_permutations",
    "infer_swiglu_neuron_permutations",
    "infer_swiglu_permutations",
    "invert_permutations",
    "verify_permutation_bijections",
]
