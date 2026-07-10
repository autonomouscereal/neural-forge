"""Topology-preserving depth codebooks for split-QKV GQA projections.

This module compresses projection storage, not attention execution topology.
Every logical layer reconstructs every query head and KV group. A prototype
is always one real, coupled GQA group: its query rows, key rows, value rows,
and matching output-projection columns come from the same source layer/group.

Fitting operates independently at each aligned KV-group position. Optional
per-layer permutations map canonical groups to physical source groups, so the
only pairwise work is across depth: O(groups * layers**2 * sketch_dim).
Runtime parameters and buffers have one registered owner. Layer and linear
views are parameterless and keep ordinary strong references to that registry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_SKETCH_DIM = 64
_ROLE_NAMES = ("q", "k", "v", "o")
_ROLE_INDEX = {name: index for index, name in enumerate(_ROLE_NAMES)}
_TIE_BREAK = "lowest source layer, then lowest canonical group"


class AttentionCodebookError(ValueError):
    """Base class for attention-codebook validation and fitting failures."""


class AttentionCodebookShapeError(AttentionCodebookError):
    """A projection bundle, topology, index, or runtime input has bad shape."""


class AttentionCodebookFiniteError(AttentionCodebookError):
    """A source projection, sketch, scale, or runtime parameter is non-finite."""


class AttentionCodebookStateError(AttentionCodebookError):
    """A tensor-only state mapping cannot reconstruct an attention registry."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttentionCodebookShapeError(f"{name} must be a positive integer.")
    return value


def _one_value(name: str, *values: Any) -> Any:
    present = [value for value in values if value is not None]
    if len(present) > 1 and any(value != present[0] for value in present[1:]):
        raise AttentionCodebookShapeError(
            f"Conflicting values were supplied for {name}."
        )
    return present[0] if present else None


def _validate_float_tensor(value: Any, name: str, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise AttentionCodebookShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise AttentionCodebookShapeError(f"{name} must be a dense non-meta tensor.")
    if value.ndim != dimensions or any(int(size) <= 0 for size in value.shape):
        raise AttentionCodebookShapeError(
            f"{name} must be a non-empty {dimensions}-dimensional tensor."
        )
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise AttentionCodebookShapeError(
            f"{name} must have a real floating-point dtype."
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise AttentionCodebookFiniteError(f"{name} contains NaN or infinity.")
    return value


def _projection_parts(value: Any, name: str) -> tuple[Tensor, Tensor | None]:
    weight = getattr(value, "weight", None)
    bias = getattr(value, "bias", None)
    if isinstance(weight, Tensor) and (bias is None or isinstance(bias, Tensor)):
        return weight, bias
    raise AttentionCodebookShapeError(
        f"{name} must expose tensor weight and optional tensor bias attributes."
    )


@dataclass(frozen=True, slots=True)
class GQAAttentionBundle:
    """One split-QKV attention projection bundle in PyTorch orientation.

    Q/K/V weights are [projection_width, input_width] and the output weight is
    [output_width, query_projection_width]. The class only exposes linear
    projections; it deliberately does not implement attention.
    """

    q_weight: Tensor
    k_weight: Tensor
    v_weight: Tensor
    o_weight: Tensor
    q_bias: Tensor | None = None
    k_bias: Tensor | None = None
    v_bias: Tensor | None = None
    o_bias: Tensor | None = None
    query_heads: int | None = None
    key_value_heads: int | None = None
    head_dim: int | None = None

    @classmethod
    def from_linears(
        cls,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        *,
        query_heads: int | None = None,
        num_query_heads: int | None = None,
        num_attention_heads: int | None = None,
        num_heads: int | None = None,
        key_value_heads: int | None = None,
        num_key_value_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
    ) -> GQAAttentionBundle:
        """Create a non-copying bundle from four linear-compatible modules."""

        q_weight, q_bias = _projection_parts(q_proj, "q_proj")
        k_weight, k_bias = _projection_parts(k_proj, "k_proj")
        v_weight, v_bias = _projection_parts(v_proj, "v_proj")
        o_weight, o_bias = _projection_parts(o_proj, "o_proj")
        return cls(
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            o_weight=o_weight,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            o_bias=o_bias,
            query_heads=_one_value(
                "query_heads",
                query_heads,
                num_query_heads,
                num_attention_heads,
                num_heads,
            ),
            key_value_heads=_one_value(
                "key_value_heads",
                key_value_heads,
                num_key_value_heads,
                num_kv_heads,
            ),
            head_dim=head_dim,
        )

    @property
    def query_weight(self) -> Tensor:
        return self.q_weight

    @property
    def key_weight(self) -> Tensor:
        return self.k_weight

    @property
    def value_weight(self) -> Tensor:
        return self.v_weight

    @property
    def output_weight(self) -> Tensor:
        return self.o_weight

    @property
    def num_attention_heads(self) -> int | None:
        return self.query_heads

    @property
    def num_key_value_heads(self) -> int | None:
        return self.key_value_heads

    def project_qkv(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the three input projections without attention scoring."""

        return (
            F.linear(inputs, self.q_weight, self.q_bias),
            F.linear(inputs, self.k_weight, self.k_bias),
            F.linear(inputs, self.v_weight, self.v_bias),
        )

    def project_output(self, contexts: Tensor) -> Tensor:
        """Apply the output projection to flattened query-head contexts."""

        return F.linear(contexts, self.o_weight, self.o_bias)


SplitQKVAttentionBundle = GQAAttentionBundle
AttentionProjectionBundle = GQAAttentionBundle


@dataclass(frozen=True, slots=True)
class AttentionCodebookParameterReport:
    """Logical dense execution and unique learned storage accounting."""

    logical_layers: int
    input_width: int
    output_width: int
    query_heads: int
    key_value_heads: int
    query_heads_per_group: int
    head_dim: int
    prototypes_per_group: int
    prototype_count: int
    logical_group_executions: int
    logical_query_head_executions: int
    logical_key_value_head_executions: int
    logical_dense_parameters: int
    unique_learned_parameters: int
    prototype_parameters: int
    scale_parameters: int
    retained_layer_bias_parameters: int
    parameters_saved: int
    assignment_index_bytes: int
    prototype_source_index_bytes: int
    permutation_index_bytes: int
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
    def logical_gqa_group_executions(self) -> int:
        return self.logical_group_executions

    @property
    def logical_attention_head_executions(self) -> int:
        return self.logical_query_head_executions

    @property
    def logical_key_head_executions(self) -> int:
        return self.logical_key_value_head_executions

    @property
    def logical_value_head_executions(self) -> int:
        return self.logical_key_value_head_executions

    @property
    def logical_output_head_executions(self) -> int:
        return self.logical_query_head_executions

    @property
    def logical_projection_head_executions(self) -> int:
        return (
            self.logical_query_head_executions
            + self.logical_key_head_executions
            + self.logical_value_head_executions
            + self.logical_output_head_executions
        )

    @property
    def index_bytes(self) -> int:
        return self.index_buffer_bytes

    @property
    def compression_ratio(self) -> float:
        return self.unique_learned_parameters / self.logical_dense_parameters

    def as_dict(self) -> dict[str, int | float]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result.update(
            {
                "logical_dense_parameter_count": self.logical_dense_parameters,
                "logical_parameter_count": self.logical_dense_parameters,
                "unique_learned_parameter_count": self.unique_learned_parameters,
                "unique_parameter_count": self.unique_learned_parameters,
                "logical_gqa_group_executions": self.logical_group_executions,
                "logical_attention_head_executions": (
                    self.logical_query_head_executions
                ),
                "logical_key_head_executions": self.logical_key_head_executions,
                "logical_value_head_executions": self.logical_value_head_executions,
                "logical_output_head_executions": (self.logical_output_head_executions),
                "logical_projection_head_executions": (
                    self.logical_projection_head_executions
                ),
                "index_bytes": self.index_buffer_bytes,
                "compression_ratio": self.compression_ratio,
            }
        )
        return result

    def __getitem__(self, key: str) -> int | float:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class AttentionCodebookDiagnostics:
    """Determinism, alignment, real-medoid, and bounded-work evidence."""

    algorithm: str
    logical_layers: int
    key_value_groups: int
    logical_groups: int
    prototypes_per_group: int
    prototype_count: int
    deterministic: bool
    tie_break: str
    real_coupled_group_medoids: bool
    used_group_permutations: bool
    permutations_provided: bool
    permutations_bijective: bool
    requested_sketch_dim: int
    sketch_dimensions: int
    sketch_elements: int
    pairwise_distance_values: int
    distance_feature_operations: int
    distance_work_bound: int
    maximum_pairwise_block_elements: int
    full_global_pairwise_matrix_elements: int
    materialized_global_pairwise_matrix: bool
    bounded_work: bool
    fitting_requires_grad: bool
    squared_sketch_error: float
    maximum_assigned_squared_distance: float

    @property
    def real_medoid_prototypes(self) -> bool:
        return self.real_coupled_group_medoids

    @property
    def exact_distance_work_bound(self) -> int:
        return self.key_value_groups * self.logical_layers**2

    @property
    def exact_distance_coordinate_work_bound(self) -> int:
        return self.exact_distance_work_bound * self.sketch_dimensions

    @property
    def distance_values_computed(self) -> int:
        return self.pairwise_distance_values

    @property
    def combined_sketch_dimensions(self) -> int:
        return self.sketch_dimensions

    @property
    def distance_work_bound_formula(self) -> str:
        return "key_value_groups * logical_layers**2 * sketch_dimensions"

    @property
    def permutation_proof(self) -> bool:
        return self.permutations_bijective

    @property
    def permutation_proof_method(self) -> str:
        return "sorted_rows_equal_arange"

    @property
    def permutation_rows_verified(self) -> int:
        return self.logical_layers

    @property
    def aligned_group_size(self) -> int:
        return self.logical_layers

    @property
    def work_complexity(self) -> str:
        return "O(groups * layers^2 * sketch_dim)"

    @property
    def no_global_pairwise_matrix(self) -> bool:
        return not self.materialized_global_pairwise_matrix

    def as_dict(self) -> dict[str, Any]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result.update(
            {
                "real_medoid_prototypes": self.real_medoid_prototypes,
                "exact_distance_work_bound": self.exact_distance_work_bound,
                "exact_distance_coordinate_work_bound": (
                    self.exact_distance_coordinate_work_bound
                ),
                "distance_values_computed": self.distance_values_computed,
                "combined_sketch_dimensions": self.combined_sketch_dimensions,
                "distance_work_bound_formula": self.distance_work_bound_formula,
                "permutation_proof": self.permutation_proof,
                "permutation_proof_method": self.permutation_proof_method,
                "permutation_rows_verified": self.permutation_rows_verified,
                "aligned_group_size": self.aligned_group_size,
                "work_complexity": self.work_complexity,
                "no_global_pairwise_matrix": self.no_global_pairwise_matrix,
            }
        )
        return result

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class _Topology:
    query_heads: int
    key_value_heads: int
    head_dim: int
    input_width: int
    output_width: int

    @property
    def query_heads_per_group(self) -> int:
        return self.query_heads // self.key_value_heads

    @property
    def query_rows_per_group(self) -> int:
        return self.query_heads_per_group * self.head_dim

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.key_value_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class _GroupBlocks:
    q_weight: Tensor
    k_weight: Tensor
    v_weight: Tensor
    o_weight: Tensor
    q_bias: Tensor | None
    k_bias: Tensor | None
    v_bias: Tensor | None


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _metadata_value(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        return _mapping_value(value, *names)
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _coerce_bundle(value: Any, layer_index: int) -> GQAAttentionBundle:
    if isinstance(value, GQAAttentionBundle):
        return value

    query_heads = _metadata_value(
        value,
        "query_heads",
        "num_query_heads",
        "num_attention_heads",
        "num_heads",
    )
    key_value_heads = _metadata_value(
        value,
        "key_value_heads",
        "num_key_value_heads",
        "num_kv_heads",
    )
    head_dim = _metadata_value(value, "head_dim")

    direct_roles = {
        "q": ("q_weight", "query_weight"),
        "k": ("k_weight", "key_weight"),
        "v": ("v_weight", "value_weight"),
        "o": ("o_weight", "output_weight"),
    }
    direct: dict[str, Any] = {}
    for role, names in direct_roles.items():
        direct[role] = _metadata_value(value, *names)
    if all(isinstance(direct[role], Tensor) for role in _ROLE_NAMES):
        return GQAAttentionBundle(
            q_weight=direct["q"],
            k_weight=direct["k"],
            v_weight=direct["v"],
            o_weight=direct["o"],
            q_bias=_metadata_value(value, "q_bias", "query_bias"),
            k_bias=_metadata_value(value, "k_bias", "key_bias"),
            v_bias=_metadata_value(value, "v_bias", "value_bias"),
            o_bias=_metadata_value(value, "o_bias", "output_bias"),
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
        )

    projections: tuple[Any, Any, Any, Any] | None = None
    if isinstance(value, Mapping):
        q_proj = _mapping_value(value, "q_proj", "query_proj", "query")
        k_proj = _mapping_value(value, "k_proj", "key_proj", "key")
        v_proj = _mapping_value(value, "v_proj", "value_proj", "value")
        o_proj = _mapping_value(
            value,
            "o_proj",
            "out_proj",
            "output_proj",
            "output",
        )
        if all(item is not None for item in (q_proj, k_proj, v_proj, o_proj)):
            projections = (q_proj, k_proj, v_proj, o_proj)
    elif all(hasattr(value, name) for name in ("q_proj", "k_proj", "v_proj")):
        output = getattr(value, "o_proj", getattr(value, "out_proj", None))
        if output is not None:
            projections = (value.q_proj, value.k_proj, value.v_proj, output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 4:
            projections = (value[0], value[1], value[2], value[3])

    if projections is None:
        raise AttentionCodebookShapeError(
            f"bundles[{layer_index}] is not a recognized split-QKV bundle."
        )
    q_weight, q_bias = _projection_parts(
        projections[0], f"bundles[{layer_index}].q_proj"
    )
    k_weight, k_bias = _projection_parts(
        projections[1], f"bundles[{layer_index}].k_proj"
    )
    v_weight, v_bias = _projection_parts(
        projections[2], f"bundles[{layer_index}].v_proj"
    )
    o_weight, o_bias = _projection_parts(
        projections[3], f"bundles[{layer_index}].o_proj"
    )
    return GQAAttentionBundle(
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        o_weight=o_weight,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        o_bias=o_bias,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
    )


def _resolved_topology(
    bundles: tuple[GQAAttentionBundle, ...],
    *,
    query_heads: int | None,
    key_value_heads: int | None,
    head_dim: int | None,
) -> _Topology:
    q_width = int(bundles[0].q_weight.shape[0])
    kv_width = int(bundles[0].k_weight.shape[0])
    bundle_query = {
        int(bundle.query_heads) for bundle in bundles if bundle.query_heads is not None
    }
    bundle_kv = {
        int(bundle.key_value_heads)
        for bundle in bundles
        if bundle.key_value_heads is not None
    }
    bundle_dims = {
        int(bundle.head_dim) for bundle in bundles if bundle.head_dim is not None
    }
    if len(bundle_query) > 1 or len(bundle_kv) > 1 or len(bundle_dims) > 1:
        raise AttentionCodebookShapeError(
            "All bundle topology metadata must agree across depth."
        )
    resolved_query = _one_value(
        "query_heads", query_heads, next(iter(bundle_query), None)
    )
    resolved_kv = _one_value(
        "key_value_heads", key_value_heads, next(iter(bundle_kv), None)
    )
    resolved_dim = _one_value("head_dim", head_dim, next(iter(bundle_dims), None))
    if resolved_dim is not None:
        resolved_dim = _positive_int(resolved_dim, "head_dim")
        if resolved_query is None and q_width % resolved_dim == 0:
            resolved_query = q_width // resolved_dim
        if resolved_kv is None and kv_width % resolved_dim == 0:
            resolved_kv = kv_width // resolved_dim
    if resolved_query is not None:
        resolved_query = _positive_int(resolved_query, "query_heads")
        if resolved_dim is None and q_width % resolved_query == 0:
            resolved_dim = q_width // resolved_query
    if resolved_kv is not None:
        resolved_kv = _positive_int(resolved_kv, "key_value_heads")
        if resolved_dim is None and kv_width % resolved_kv == 0:
            resolved_dim = kv_width // resolved_kv
    if resolved_query is None or resolved_kv is None or resolved_dim is None:
        raise AttentionCodebookShapeError(
            "query_heads, key_value_heads, and head_dim must be supplied or "
            "inferable from bundle metadata."
        )
    resolved_query = _positive_int(resolved_query, "query_heads")
    resolved_kv = _positive_int(resolved_kv, "key_value_heads")
    resolved_dim = _positive_int(resolved_dim, "head_dim")
    if resolved_query % resolved_kv != 0:
        raise AttentionCodebookShapeError(
            "query_heads must divide exactly into key_value_heads groups."
        )
    return _Topology(
        query_heads=resolved_query,
        key_value_heads=resolved_kv,
        head_dim=resolved_dim,
        input_width=int(bundles[0].q_weight.shape[1]),
        output_width=int(bundles[0].o_weight.shape[0]),
    )


def _validated_bundles(
    values: Any,
    *,
    query_heads: int | None,
    key_value_heads: int | None,
    head_dim: int | None,
) -> tuple[tuple[GQAAttentionBundle, ...], _Topology]:
    if isinstance(values, (str, bytes, Mapping, Tensor)):
        raise AttentionCodebookShapeError("bundles must be an iterable across depth.")
    try:
        source_values = tuple(values)
    except TypeError as exc:
        raise AttentionCodebookShapeError(
            "bundles must be an iterable across depth."
        ) from exc
    if not source_values:
        raise AttentionCodebookShapeError("bundles must contain at least one layer.")
    bundles = tuple(
        _coerce_bundle(value, index) for index, value in enumerate(source_values)
    )
    for index, bundle in enumerate(bundles):
        for role in _ROLE_NAMES:
            _validate_float_tensor(
                getattr(bundle, f"{role}_weight"),
                f"bundles[{index}].{role}_weight",
                2,
            )
    topology = _resolved_topology(
        bundles,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
    )
    expected_shapes = {
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
    for index, bundle in enumerate(bundles):
        for name, shape in expected_shapes.items():
            tensor = getattr(bundle, name)
            if tuple(tensor.shape) != shape:
                raise AttentionCodebookShapeError(
                    f"bundles[{index}].{name} must have shape {shape!r}."
                )
        presence = tuple(getattr(bundle, name) is not None for name in expected_biases)
        if presence != reference_presence:
            raise AttentionCodebookShapeError(
                "Q/K/V/O bias presence must be consistent across depth."
            )
        tensors = [getattr(bundle, name) for name in expected_shapes]
        for name, width in expected_biases.items():
            bias = getattr(bundle, name)
            if bias is not None:
                checked = _validate_float_tensor(bias, f"bundles[{index}].{name}", 1)
                if tuple(checked.shape) != (width,):
                    raise AttentionCodebookShapeError(
                        f"bundles[{index}].{name} must have shape ({width},)."
                    )
                tensors.append(checked)
        if any(tensor.dtype != reference_dtype for tensor in tensors):
            raise AttentionCodebookShapeError(
                "All fitted projection tensors must have one common dtype."
            )
        if any(tensor.device != reference_device for tensor in tensors):
            raise AttentionCodebookShapeError(
                "All fitted projection tensors must have one common device."
            )
    return bundles, topology


def _group_blocks(
    bundle: GQAAttentionBundle,
    physical_group: int,
    topology: _Topology,
) -> _GroupBlocks:
    groups = topology.key_value_heads
    q_rows = topology.query_rows_per_group
    head_dim = topology.head_dim
    q_weight = bundle.q_weight.reshape(groups, q_rows, topology.input_width)[
        physical_group
    ]
    k_weight = bundle.k_weight.reshape(groups, head_dim, topology.input_width)[
        physical_group
    ]
    v_weight = bundle.v_weight.reshape(groups, head_dim, topology.input_width)[
        physical_group
    ]
    o_weight = bundle.o_weight.reshape(topology.output_width, groups, q_rows)[
        :, physical_group, :
    ]
    q_bias = (
        bundle.q_bias.reshape(groups, q_rows)[physical_group]
        if bundle.q_bias is not None
        else None
    )
    k_bias = (
        bundle.k_bias.reshape(groups, head_dim)[physical_group]
        if bundle.k_bias is not None
        else None
    )
    v_bias = (
        bundle.v_bias.reshape(groups, head_dim)[physical_group]
        if bundle.v_bias is not None
        else None
    )
    return _GroupBlocks(
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        o_weight=o_weight,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
    )


def _role_vectors(blocks: _GroupBlocks) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    def producer(weight: Tensor, bias: Tensor | None) -> Tensor:
        values = [weight.reshape(-1)]
        if bias is not None:
            values.append(bias.reshape(-1))
        return torch.cat(values)

    return (
        producer(blocks.q_weight, blocks.q_bias),
        producer(blocks.k_weight, blocks.k_bias),
        producer(blocks.v_weight, blocks.v_bias),
        blocks.o_weight.reshape(-1),
    )


def _allocate_role_dimensions(widths: Sequence[int], total: int) -> tuple[int, ...]:
    target = min(sum(widths), total)
    allocation = [0 for _ in widths]
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
            key=lambda index: (
                widths[index] / (allocation[index] + 1),
                -index,
            ),
        )
        allocation[selected] += 1
        target -= 1
    return tuple(allocation)


def _sample_vector(vector: Tensor, count: int) -> Tensor:
    if count == 0:
        return torch.empty(0, dtype=torch.float64)
    length = int(vector.numel())
    if count >= length:
        sampled = vector.detach().reshape(-1)
    else:
        positions = torch.arange(count, device=vector.device, dtype=torch.int64)
        indices = ((2 * positions + 1) * length) // (2 * count)
        sampled = vector.detach().reshape(-1).index_select(0, indices)
    return sampled.to(device="cpu", dtype=torch.float64) / math.sqrt(count)


def _feature_sketch(
    bundles: tuple[GQAAttentionBundle, ...],
    topology: _Topology,
    permutations: Tensor,
    sketch_dim: int,
) -> Tensor:
    first = _group_blocks(bundles[0], int(permutations[0, 0]), topology)
    widths = tuple(int(vector.numel()) for vector in _role_vectors(first))
    allocation = _allocate_role_dimensions(widths, sketch_dim)
    actual_dim = sum(allocation)
    features = torch.empty(
        len(bundles),
        topology.key_value_heads,
        actual_dim,
        dtype=torch.float64,
    )
    for layer_index, bundle in enumerate(bundles):
        for group_index in range(topology.key_value_heads):
            physical = int(permutations[layer_index, group_index])
            vectors = _role_vectors(_group_blocks(bundle, physical, topology))
            sampled = [
                _sample_vector(vector, count)
                for vector, count in zip(vectors, allocation, strict=True)
            ]
            features[layer_index, group_index] = torch.cat(sampled)
    centered = features - features.mean(dim=0, keepdim=True)
    rms = centered.square().mean(dim=0, keepdim=True).sqrt()
    floor = torch.finfo(features.dtype).eps
    normalized = torch.where(rms > floor, centered / rms.clamp_min(floor), centered)
    if not bool(torch.isfinite(normalized).all()):
        raise AttentionCodebookFiniteError("The aligned group sketch is non-finite.")
    return normalized


def _validated_permutations(
    value: Tensor | Sequence[Sequence[int]] | None,
    logical_layers: int,
    groups: int,
) -> Tensor:
    if value is None:
        return torch.arange(groups, dtype=torch.int64).repeat(logical_layers, 1)
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if isinstance(value, Tensor):
        if (
            value.layout != torch.strided
            or value.device.type == "meta"
            or value.dtype not in integer_dtypes
        ):
            raise AttentionCodebookShapeError(
                "group_permutations must be a dense integer tensor."
            )
        result = value.detach().to(device="cpu", dtype=torch.int64)
    else:
        if isinstance(value, (str, bytes, Mapping)):
            raise AttentionCodebookShapeError(
                "group_permutations must be an integer matrix."
            )
        try:
            rows = tuple(value)
        except TypeError as exc:
            raise AttentionCodebookShapeError(
                "group_permutations must be an integer matrix."
            ) from exc
        normalized: list[tuple[int, ...]] = []
        for row in rows:
            if isinstance(row, Tensor):
                if (
                    row.layout != torch.strided
                    or row.device.type == "meta"
                    or row.ndim != 1
                    or row.dtype not in integer_dtypes
                ):
                    raise AttentionCodebookShapeError(
                        "Every group_permutations row must be an integer vector."
                    )
                row_values = tuple(int(item) for item in row.detach().to("cpu"))
            else:
                if isinstance(row, (str, bytes, Mapping)):
                    raise AttentionCodebookShapeError(
                        "Every group_permutations row must be an integer vector."
                    )
                try:
                    raw_values = tuple(row)
                except TypeError as exc:
                    raise AttentionCodebookShapeError(
                        "Every group_permutations row must be an integer vector."
                    ) from exc
                if any(
                    isinstance(item, bool) or not isinstance(item, Integral)
                    for item in raw_values
                ):
                    raise AttentionCodebookShapeError(
                        "Every group_permutations value must be an integer."
                    )
                row_values = tuple(int(item) for item in raw_values)
            normalized.append(row_values)
        try:
            result = torch.tensor(normalized, dtype=torch.int64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise AttentionCodebookShapeError(
                "group_permutations must be a rectangular integer matrix."
            ) from exc
    if tuple(result.shape) != (logical_layers, groups):
        raise AttentionCodebookShapeError(
            "group_permutations must have shape [layers, key_value_heads]."
        )
    expected = torch.arange(groups, dtype=torch.int64)
    if any(not torch.equal(torch.sort(row).values, expected) for row in result):
        raise AttentionCodebookShapeError(
            "Every group_permutations row must be a bijection."
        )
    return result


def _fit_aligned_medoids(
    features: Tensor,
    prototypes_per_group: int,
) -> tuple[Tensor, Tensor, Tensor, int, int]:
    layers, groups, sketch_dim = map(int, features.shape)
    if prototypes_per_group == layers:
        medoids = torch.arange(layers, dtype=torch.int64).repeat(groups, 1)
        assignments = torch.arange(layers, dtype=torch.int64).repeat(groups, 1).t()
        distances = torch.zeros((layers, groups), dtype=torch.float64)
        return medoids, assignments, distances, 0, 0
    medoids = torch.empty((groups, prototypes_per_group), dtype=torch.int64)
    assignments = torch.empty((layers, groups), dtype=torch.int64)
    assigned_distances = torch.empty((layers, groups), dtype=torch.float64)
    distance_values = 0
    max_block = 0
    for group_index in range(groups):
        points = features[:, group_index]
        delta = points[:, None, :] - points[None, :, :]
        pairwise = delta.square().sum(dim=-1)
        distance_values += layers * layers
        max_block = max(max_block, layers * layers)
        selected = [int(torch.argmin(pairwise.sum(dim=1)).item())]
        minimum = pairwise[:, selected[0]].clone()
        while len(selected) < prototypes_per_group:
            scores = minimum.clone()
            scores[selected] = -1.0
            next_index = int(torch.argmax(scores).item())
            selected.append(next_index)
            minimum = torch.minimum(minimum, pairwise[:, next_index])
        selected_tensor = torch.tensor(selected, dtype=torch.int64)
        distances = pairwise.index_select(1, selected_tensor)
        group_assignments = torch.argmin(distances, dim=1)
        group_distances = distances.gather(1, group_assignments.unsqueeze(1)).squeeze(1)
        for slot, source_layer in enumerate(selected):
            group_assignments[source_layer] = slot
            group_distances[source_layer] = 0.0
        medoids[group_index] = selected_tensor
        assignments[:, group_index] = group_assignments
        assigned_distances[:, group_index] = group_distances
    feature_operations = distance_values * sketch_dim
    return (
        medoids,
        assignments,
        assigned_distances,
        feature_operations,
        max_block,
    )


def _gather_prototypes(
    bundles: tuple[GQAAttentionBundle, ...],
    topology: _Topology,
    permutations: Tensor,
    medoid_layers: Tensor,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor,
]:
    q_groups: list[Tensor] = []
    k_groups: list[Tensor] = []
    v_groups: list[Tensor] = []
    o_groups: list[Tensor] = []
    qb_groups: list[Tensor] = []
    kb_groups: list[Tensor] = []
    vb_groups: list[Tensor] = []
    source_groups: list[Tensor] = []
    for group_index in range(topology.key_value_heads):
        q_slots: list[Tensor] = []
        k_slots: list[Tensor] = []
        v_slots: list[Tensor] = []
        o_slots: list[Tensor] = []
        qb_slots: list[Tensor] = []
        kb_slots: list[Tensor] = []
        vb_slots: list[Tensor] = []
        source_slots: list[int] = []
        for layer_tensor in medoid_layers[group_index]:
            layer_index = int(layer_tensor)
            physical = int(permutations[layer_index, group_index])
            blocks = _group_blocks(bundles[layer_index], physical, topology)
            q_slots.append(blocks.q_weight.detach())
            k_slots.append(blocks.k_weight.detach())
            v_slots.append(blocks.v_weight.detach())
            o_slots.append(blocks.o_weight.detach())
            if blocks.q_bias is not None:
                qb_slots.append(blocks.q_bias.detach())
            if blocks.k_bias is not None:
                kb_slots.append(blocks.k_bias.detach())
            if blocks.v_bias is not None:
                vb_slots.append(blocks.v_bias.detach())
            source_slots.append(layer_index * topology.key_value_heads + physical)
        q_groups.append(torch.stack(q_slots))
        k_groups.append(torch.stack(k_slots))
        v_groups.append(torch.stack(v_slots))
        o_groups.append(torch.stack(o_slots))
        if qb_slots:
            qb_groups.append(torch.stack(qb_slots))
        if kb_slots:
            kb_groups.append(torch.stack(kb_slots))
        if vb_slots:
            vb_groups.append(torch.stack(vb_slots))
        source_groups.append(torch.tensor(source_slots, dtype=torch.int64))
    return (
        torch.stack(q_groups),
        torch.stack(k_groups),
        torch.stack(v_groups),
        torch.stack(o_groups),
        torch.stack(qb_groups) if qb_groups else None,
        torch.stack(kb_groups) if kb_groups else None,
        torch.stack(vb_groups) if vb_groups else None,
        torch.stack(source_groups),
    )


def _least_squares_scale(source: Tensor, prototype: Tensor) -> float:
    source64 = source.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    prototype64 = prototype.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    denominator = float(torch.dot(prototype64, prototype64).item())
    if denominator <= torch.finfo(torch.float64).tiny:
        return 1.0
    value = float(torch.dot(source64, prototype64).item()) / denominator
    return value if math.isfinite(value) else 1.0


def _fitted_role_scales(
    bundles: tuple[GQAAttentionBundle, ...],
    topology: _Topology,
    permutations: Tensor,
    assignments: Tensor,
    prototype_blocks: tuple[Tensor, Tensor, Tensor, Tensor],
    prototype_biases: tuple[Tensor | None, Tensor | None, Tensor | None],
) -> Tensor:
    scales = torch.empty(
        len(bundles),
        topology.key_value_heads,
        len(_ROLE_NAMES),
        dtype=bundles[0].q_weight.dtype,
        device=bundles[0].q_weight.device,
    )
    for layer_index, bundle in enumerate(bundles):
        for group_index in range(topology.key_value_heads):
            physical = int(permutations[layer_index, group_index])
            slot = int(assignments[layer_index, group_index])
            source = _group_blocks(bundle, physical, topology)
            source_vectors = _role_vectors(source)
            prototype_vectors: list[Tensor] = []
            for role_index, weight in enumerate(prototype_blocks):
                selected_weight = weight[group_index, slot]
                if role_index < 3 and prototype_biases[role_index] is not None:
                    selected_bias = prototype_biases[role_index][group_index, slot]
                    prototype_vectors.append(
                        torch.cat(
                            (selected_weight.reshape(-1), selected_bias.reshape(-1))
                        )
                    )
                else:
                    prototype_vectors.append(selected_weight.reshape(-1))
            for role_index, (source_vector, prototype_vector) in enumerate(
                zip(source_vectors, prototype_vectors, strict=True)
            ):
                scales[layer_index, group_index, role_index] = _least_squares_scale(
                    source_vector, prototype_vector
                )
    if not bool(torch.isfinite(scales).all()):
        raise AttentionCodebookFiniteError("Fitted role scales are non-finite.")
    return scales


def _validated_role_scales(
    value: Tensor | Mapping[str, Tensor],
    logical_layers: int,
    groups: int,
    reference: Tensor,
) -> Tensor:
    if isinstance(value, Mapping):
        if set(value) != set(_ROLE_NAMES):
            raise AttentionCodebookShapeError(
                "A role-scale mapping must contain exactly q, k, v, and o."
            )
        try:
            value = torch.stack(
                [torch.as_tensor(value[role]) for role in _ROLE_NAMES],
                dim=-1,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise AttentionCodebookShapeError(
                "Role-scale mapping values must be compatible tensors."
            ) from exc
    if not isinstance(value, Tensor) or not torch.is_floating_point(value):
        raise AttentionCodebookShapeError("role_scales must be floating point.")
    if tuple(value.shape) == (logical_layers, groups):
        value = value.unsqueeze(-1).expand(-1, -1, len(_ROLE_NAMES))
    expected = (logical_layers, groups, len(_ROLE_NAMES))
    if tuple(value.shape) != expected:
        raise AttentionCodebookShapeError(
            "role_scales must have shape [layers, groups] or [layers, groups, 4]."
        )
    result = (
        value.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone()
    )
    if not bool(torch.isfinite(result).all()):
        raise AttentionCodebookFiniteError("role_scales contains NaN or infinity.")
    return result


class AttentionCodebookRegistry(nn.Module):
    """The sole registered owner of codebook parameters and index buffers."""

    def __init__(
        self,
        prototype_q_weight: Tensor,
        prototype_k_weight: Tensor,
        prototype_v_weight: Tensor,
        prototype_o_weight: Tensor,
        assignments: Tensor,
        prototype_source_indices: Tensor,
        group_permutations: Tensor | None = None,
        *,
        prototype_q_bias: Tensor | None = None,
        prototype_k_bias: Tensor | None = None,
        prototype_v_bias: Tensor | None = None,
        layer_o_bias: Tensor | None = None,
        role_scales: Tensor | None = None,
        diagnostics: AttentionCodebookDiagnostics | None = None,
    ) -> None:
        super().__init__()
        q_weight = _validate_float_tensor(prototype_q_weight, "prototype_q_weight", 4)
        k_weight = _validate_float_tensor(prototype_k_weight, "prototype_k_weight", 4)
        v_weight = _validate_float_tensor(prototype_v_weight, "prototype_v_weight", 4)
        o_weight = _validate_float_tensor(prototype_o_weight, "prototype_o_weight", 4)
        groups, prototypes, q_rows, input_width = map(int, q_weight.shape)
        if tuple(k_weight.shape[:2]) != (groups, prototypes):
            raise AttentionCodebookShapeError(
                "Prototype K groups and depth slots must match Q."
            )
        if tuple(v_weight.shape) != tuple(k_weight.shape):
            raise AttentionCodebookShapeError(
                "Prototype K and V weights must have equal shape."
            )
        head_dim = int(k_weight.shape[2])
        if int(k_weight.shape[3]) != input_width:
            raise AttentionCodebookShapeError(
                "Prototype Q/K/V input widths must match."
            )
        if tuple(o_weight.shape[:2]) != (groups, prototypes):
            raise AttentionCodebookShapeError(
                "Prototype O groups and depth slots must match Q."
            )
        output_width = int(o_weight.shape[2])
        if int(o_weight.shape[3]) != q_rows:
            raise AttentionCodebookShapeError(
                "Prototype O columns per group must match Q rows per group."
            )
        if q_rows % head_dim != 0:
            raise AttentionCodebookShapeError(
                "Prototype Q rows per group must be divisible by head_dim."
            )
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if (
            not isinstance(assignments, Tensor)
            or assignments.ndim != 2
            or assignments.dtype not in integer_dtypes
        ):
            raise AttentionCodebookShapeError(
                "assignments must be a [layers, groups] integer tensor."
            )
        logical_layers, assignment_groups = map(int, assignments.shape)
        if logical_layers <= 0 or assignment_groups != groups:
            raise AttentionCodebookShapeError(
                "assignments must have shape [positive layers, prototype groups]."
            )
        assignment64 = assignments.detach().to(device="cpu", dtype=torch.int64)
        if int(assignment64.min()) < 0 or int(assignment64.max()) >= prototypes:
            raise AttentionCodebookShapeError(
                "assignments contains an out-of-range depth prototype slot."
            )
        if group_permutations is not None and (
            not isinstance(group_permutations, Tensor)
            or group_permutations.dtype not in integer_dtypes
            or tuple(group_permutations.shape) != (logical_layers, groups)
        ):
            raise AttentionCodebookShapeError(
                "group_permutations must be a [layers, groups] integer tensor."
            )
        permutation64 = _validated_permutations(
            group_permutations, logical_layers, groups
        )
        if (
            not isinstance(prototype_source_indices, Tensor)
            or prototype_source_indices.dtype not in integer_dtypes
            or tuple(prototype_source_indices.shape) != (groups, prototypes)
        ):
            raise AttentionCodebookShapeError(
                "prototype_source_indices must have shape [groups, depth slots]."
            )
        source64 = prototype_source_indices.detach().to(device="cpu", dtype=torch.int64)
        logical_groups = logical_layers * groups
        if int(source64.min()) < 0 or int(source64.max()) >= logical_groups:
            raise AttentionCodebookShapeError(
                "prototype_source_indices contains an out-of-range source group."
            )
        if int(torch.unique(source64).numel()) != groups * prototypes:
            raise AttentionCodebookShapeError(
                "Every prototype must identify a distinct real source group."
            )
        for group_index in range(groups):
            for source_index in source64[group_index]:
                layer_index, physical_group = divmod(int(source_index), groups)
                if int(permutation64[layer_index, group_index]) != physical_group:
                    raise AttentionCodebookShapeError(
                        "Prototype provenance disagrees with group_permutations."
                    )
        learned = [q_weight, k_weight, v_weight, o_weight]
        optional_shapes = (
            ("prototype_q_bias", prototype_q_bias, (groups, prototypes, q_rows)),
            (
                "prototype_k_bias",
                prototype_k_bias,
                (groups, prototypes, head_dim),
            ),
            (
                "prototype_v_bias",
                prototype_v_bias,
                (groups, prototypes, head_dim),
            ),
            ("layer_o_bias", layer_o_bias, (logical_layers, output_width)),
            (
                "role_scales",
                role_scales,
                (logical_layers, groups, len(_ROLE_NAMES)),
            ),
        )
        for name, value, expected in optional_shapes:
            if value is None:
                continue
            checked = _validate_float_tensor(value, name, len(expected))
            if tuple(checked.shape) != expected:
                raise AttentionCodebookShapeError(
                    f"{name} must have shape {expected!r}."
                )
            learned.append(checked)
        if any(value.dtype != q_weight.dtype for value in learned):
            raise AttentionCodebookShapeError("All learned tensors need one dtype.")
        if any(value.device != q_weight.device for value in learned):
            raise AttentionCodebookShapeError("All learned tensors need one device.")
        self.prototype_q_weight = nn.Parameter(q_weight.detach().clone())
        self.prototype_k_weight = nn.Parameter(k_weight.detach().clone())
        self.prototype_v_weight = nn.Parameter(v_weight.detach().clone())
        self.prototype_o_weight = nn.Parameter(o_weight.detach().clone())
        self.prototype_q_bias = (
            nn.Parameter(prototype_q_bias.detach().clone())
            if prototype_q_bias is not None
            else None
        )
        self.prototype_k_bias = (
            nn.Parameter(prototype_k_bias.detach().clone())
            if prototype_k_bias is not None
            else None
        )
        self.prototype_v_bias = (
            nn.Parameter(prototype_v_bias.detach().clone())
            if prototype_v_bias is not None
            else None
        )
        self.layer_o_bias = (
            nn.Parameter(layer_o_bias.detach().clone())
            if layer_o_bias is not None
            else None
        )
        self.role_scales = (
            nn.Parameter(role_scales.detach().clone())
            if role_scales is not None
            else None
        )
        self.register_buffer("assignments", assignment64.to(torch.int32))
        self.register_buffer("prototype_source_indices", source64.to(torch.int32))
        self.register_buffer("group_permutations", permutation64.to(torch.int32))
        self.diagnostics = diagnostics
        self.input_width = input_width
        self.output_width = output_width
        self.head_dim = head_dim

    @property
    def logical_layers(self) -> int:
        return int(self.assignments.shape[0])

    @property
    def key_value_heads(self) -> int:
        return int(self.prototype_q_weight.shape[0])

    @property
    def num_key_value_heads(self) -> int:
        return self.key_value_heads

    @property
    def prototypes_per_group(self) -> int:
        return int(self.prototype_q_weight.shape[1])

    @property
    def prototype_count(self) -> int:
        return self.key_value_heads * self.prototypes_per_group

    @property
    def query_rows_per_group(self) -> int:
        return int(self.prototype_q_weight.shape[2])

    @property
    def query_heads_per_group(self) -> int:
        return self.query_rows_per_group // self.head_dim

    @property
    def query_heads(self) -> int:
        return self.key_value_heads * self.query_heads_per_group

    @property
    def num_attention_heads(self) -> int:
        return self.query_heads

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.key_value_heads * self.head_dim

    @property
    def has_scales(self) -> bool:
        return self.role_scales is not None

    @property
    def q_scales(self) -> Tensor | None:
        return None if self.role_scales is None else self.role_scales[..., 0]

    @property
    def k_scales(self) -> Tensor | None:
        return None if self.role_scales is None else self.role_scales[..., 1]

    @property
    def v_scales(self) -> Tensor | None:
        return None if self.role_scales is None else self.role_scales[..., 2]

    @property
    def o_scales(self) -> Tensor | None:
        return None if self.role_scales is None else self.role_scales[..., 3]

    @property
    def layer_assignments(self) -> Tensor:
        return self.assignments

    @property
    def kv_group_permutations(self) -> Tensor:
        return self.group_permutations

    @property
    def report(self) -> AttentionCodebookParameterReport:
        query_width = self.query_width
        kv_width = self.key_value_width
        producer_biases = (
            query_width * int(self.prototype_q_bias is not None)
            + kv_width * int(self.prototype_k_bias is not None)
            + kv_width * int(self.prototype_v_bias is not None)
        )
        output_biases = self.output_width if self.layer_o_bias is not None else 0
        per_layer = (
            query_width * self.input_width
            + 2 * kv_width * self.input_width
            + self.output_width * query_width
            + producer_biases
            + output_biases
        )
        logical_dense = self.logical_layers * per_layer
        unique_learned = sum(int(parameter.numel()) for parameter in self.parameters())
        scale_parameters = (
            int(self.role_scales.numel()) if self.role_scales is not None else 0
        )
        retained_bias = (
            int(self.layer_o_bias.numel()) if self.layer_o_bias is not None else 0
        )
        prototype_parameters = unique_learned - scale_parameters - retained_bias
        assignment_bytes = (
            int(self.assignments.numel()) * self.assignments.element_size()
        )
        source_bytes = (
            int(self.prototype_source_indices.numel())
            * self.prototype_source_indices.element_size()
        )
        permutation_bytes = (
            int(self.group_permutations.numel())
            * self.group_permutations.element_size()
        )
        return AttentionCodebookParameterReport(
            logical_layers=self.logical_layers,
            input_width=self.input_width,
            output_width=self.output_width,
            query_heads=self.query_heads,
            key_value_heads=self.key_value_heads,
            query_heads_per_group=self.query_heads_per_group,
            head_dim=self.head_dim,
            prototypes_per_group=self.prototypes_per_group,
            prototype_count=self.prototype_count,
            logical_group_executions=self.logical_layers * self.key_value_heads,
            logical_query_head_executions=self.logical_layers * self.query_heads,
            logical_key_value_head_executions=(
                self.logical_layers * self.key_value_heads
            ),
            logical_dense_parameters=logical_dense,
            unique_learned_parameters=unique_learned,
            prototype_parameters=prototype_parameters,
            scale_parameters=scale_parameters,
            retained_layer_bias_parameters=retained_bias,
            parameters_saved=logical_dense - unique_learned,
            assignment_index_bytes=assignment_bytes,
            prototype_source_index_bytes=source_bytes,
            permutation_index_bytes=permutation_bytes,
            index_buffer_bytes=(assignment_bytes + source_bytes + permutation_bytes),
        )

    def _layer_index(self, layer_index: int) -> int:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.logical_layers
        ):
            raise AttentionCodebookShapeError(
                "layer_index must identify a logical codebook layer."
            )
        return layer_index

    def _role(self, role: str) -> str:
        aliases = {
            "query": "q",
            "query_proj": "q",
            "q_proj": "q",
            "key": "k",
            "key_proj": "k",
            "k_proj": "k",
            "value": "v",
            "value_proj": "v",
            "v_proj": "v",
            "out": "o",
            "output": "o",
            "out_proj": "o",
            "output_proj": "o",
            "o_proj": "o",
        }
        if not isinstance(role, str):
            raise AttentionCodebookShapeError("projection role must be a string.")
        normalized = aliases.get(role.strip().lower(), role.strip().lower())
        if normalized not in _ROLE_INDEX:
            raise AttentionCodebookShapeError("projection role must be q, k, v, or o.")
        return normalized

    def materialize_projection(
        self,
        layer_index: int,
        role: str,
    ) -> tuple[Tensor, Tensor | None]:
        """Reconstruct one differentiable Q/K/V/O linear projection."""

        layer_index = self._layer_index(layer_index)
        role = self._role(role)
        device = self.prototype_q_weight.device
        groups = torch.arange(self.key_value_heads, device=device)
        slots = self.assignments[layer_index].to(device=device, dtype=torch.long)
        weight = getattr(self, f"prototype_{role}_weight")[groups, slots]
        bias_parameter = getattr(self, f"prototype_{role}_bias", None)
        bias = bias_parameter[groups, slots] if bias_parameter is not None else None
        if self.role_scales is not None:
            scales = self.role_scales[layer_index, :, _ROLE_INDEX[role]]
            weight = weight * scales[:, None, None]
            if bias is not None:
                bias = bias * scales[:, None]
        physical_to_canonical = torch.argsort(
            self.group_permutations[layer_index].to(
                device=device,
                dtype=torch.long,
            )
        )
        weight = weight.index_select(0, physical_to_canonical)
        if bias is not None:
            bias = bias.index_select(0, physical_to_canonical)
        if role == "o":
            weight = weight.permute(1, 0, 2).reshape(
                self.output_width, self.query_width
            )
            bias = (
                self.layer_o_bias[layer_index]
                if self.layer_o_bias is not None
                else None
            )
        elif role == "q":
            weight = weight.reshape(self.query_width, self.input_width)
            if bias is not None:
                bias = bias.reshape(self.query_width)
        else:
            weight = weight.reshape(self.key_value_width, self.input_width)
            if bias is not None:
                bias = bias.reshape(self.key_value_width)
        return weight, bias

    def materialize_layer(self, layer_index: int) -> GQAAttentionBundle:
        """Gather all four projections with one coupled group assignment map."""

        q_weight, q_bias = self.materialize_projection(layer_index, "q")
        k_weight, k_bias = self.materialize_projection(layer_index, "k")
        v_weight, v_bias = self.materialize_projection(layer_index, "v")
        o_weight, o_bias = self.materialize_projection(layer_index, "o")
        return GQAAttentionBundle(
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            o_weight=o_weight,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            o_bias=o_bias,
            query_heads=self.query_heads,
            key_value_heads=self.key_value_heads,
            head_dim=self.head_dim,
        )

    reconstruct_layer = materialize_layer

    def project_qkv(
        self, inputs: Tensor, layer_index: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.materialize_layer(layer_index).project_qkv(inputs)

    def project_output(self, contexts: Tensor, layer_index: int) -> Tensor:
        return self.materialize_layer(layer_index).project_output(contexts)

    def forward(
        self, inputs: Tensor, layer_index: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply Q/K/V only; attention scoring is intentionally out of scope."""

        return self.project_qkv(inputs, layer_index)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
        *,
        prefix: str = "",
    ) -> AttentionCodebookRegistry:
        """Reconstruct solely from ordinary state tensors."""

        if not isinstance(state_dict, Mapping):
            raise AttentionCodebookStateError("state_dict must be a tensor mapping.")
        if not prefix and "registry.prototype_q_weight" in state_dict:
            prefix = "registry"
        if prefix and not prefix.endswith("."):
            prefix = f"{prefix}."
        required = (
            "prototype_q_weight",
            "prototype_k_weight",
            "prototype_v_weight",
            "prototype_o_weight",
            "assignments",
            "prototype_source_indices",
            "group_permutations",
        )
        missing = [name for name in required if f"{prefix}{name}" not in state_dict]
        if missing:
            raise AttentionCodebookStateError(
                f"state_dict is missing required tensors {missing!r}."
            )

        def optional(name: str) -> Tensor | None:
            return state_dict.get(f"{prefix}{name}")

        result = cls(
            state_dict[f"{prefix}prototype_q_weight"],
            state_dict[f"{prefix}prototype_k_weight"],
            state_dict[f"{prefix}prototype_v_weight"],
            state_dict[f"{prefix}prototype_o_weight"],
            state_dict[f"{prefix}assignments"],
            state_dict[f"{prefix}prototype_source_indices"],
            state_dict[f"{prefix}group_permutations"],
            prototype_q_bias=optional("prototype_q_bias"),
            prototype_k_bias=optional("prototype_k_bias"),
            prototype_v_bias=optional("prototype_v_bias"),
            layer_o_bias=optional("layer_o_bias"),
            role_scales=optional("role_scales"),
        )
        local_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        incompatible = result.load_state_dict(local_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AttentionCodebookStateError(
                "state_dict did not exactly match the reconstructed registry."
            )
        return result


class AttentionCodebookProjectionView(nn.Module):
    """Parameterless linear-compatible view into one reconstructed projection."""

    def __init__(
        self,
        registry: AttentionCodebookRegistry,
        layer_index: int,
        role: str,
    ) -> None:
        super().__init__()
        if not isinstance(registry, AttentionCodebookRegistry):
            raise AttentionCodebookShapeError(
                "registry must be an AttentionCodebookRegistry."
            )
        registry._layer_index(layer_index)
        role = registry._role(role)
        object.__setattr__(self, "_registry", registry)
        self.layer_index = layer_index
        self.role = role
        if role == "o":
            self.in_features = registry.query_width
            self.out_features = registry.output_width
        else:
            self.in_features = registry.input_width
            self.out_features = (
                registry.query_width if role == "q" else registry.key_value_width
            )
        self.train(registry.training)

    @property
    def registry(self) -> AttentionCodebookRegistry:
        return self._registry

    @property
    def weight(self) -> Tensor:
        return self.registry.materialize_projection(self.layer_index, self.role)[0]

    @property
    def bias(self) -> Tensor | None:
        return self.registry.materialize_projection(self.layer_index, self.role)[1]

    def forward(self, inputs: Tensor) -> Tensor:
        weight, bias = self.registry.materialize_projection(self.layer_index, self.role)
        return F.linear(inputs, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, role={self.role!r}, "
            f"layer_index={self.layer_index}"
        )


class AttentionCodebookLayerView(nn.Module):
    """Parameterless Q/K/V/O module views for one logical attention layer."""

    def __init__(
        self,
        registry: AttentionCodebookRegistry,
        layer_index: int,
    ) -> None:
        super().__init__()
        if not isinstance(registry, AttentionCodebookRegistry):
            raise AttentionCodebookShapeError(
                "registry must be an AttentionCodebookRegistry."
            )
        registry._layer_index(layer_index)
        object.__setattr__(self, "_registry", registry)
        self.layer_index = layer_index
        self.q_proj = AttentionCodebookProjectionView(registry, layer_index, "q")
        self.k_proj = AttentionCodebookProjectionView(registry, layer_index, "k")
        self.v_proj = AttentionCodebookProjectionView(registry, layer_index, "v")
        self.o_proj = AttentionCodebookProjectionView(registry, layer_index, "o")
        self.num_heads = registry.query_heads
        self.num_attention_heads = registry.query_heads
        self.num_key_value_heads = registry.key_value_heads
        self.head_dim = registry.head_dim
        self.train(registry.training)

    @property
    def registry(self) -> AttentionCodebookRegistry:
        return self._registry

    @property
    def out_proj(self) -> AttentionCodebookProjectionView:
        return self.o_proj

    def materialize(self) -> GQAAttentionBundle:
        return self.registry.materialize_layer(self.layer_index)

    def project_qkv(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)

    def project_output(self, contexts: Tensor) -> Tensor:
        return self.o_proj(contexts)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply Q/K/V only; callers retain their existing attention kernel."""

        return self.project_qkv(inputs)

    def extra_repr(self) -> str:
        return (
            f"query_heads={self.num_attention_heads}, "
            f"key_value_heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, layer_index={self.layer_index}"
        )


class GlobalAttentionCodebook(nn.Module):
    """One strongly registered owner plus all logical parameterless views."""

    def __init__(self, registry: AttentionCodebookRegistry) -> None:
        super().__init__()
        if not isinstance(registry, AttentionCodebookRegistry):
            raise AttentionCodebookShapeError(
                "registry must be an AttentionCodebookRegistry."
            )
        self.registry = registry
        self.layers = nn.ModuleList(
            AttentionCodebookLayerView(registry, index)
            for index in range(registry.logical_layers)
        )

    @property
    def layer_views(self) -> nn.ModuleList:
        return self.layers

    @property
    def logical_layers(self) -> int:
        return self.registry.logical_layers

    @property
    def report(self) -> AttentionCodebookParameterReport:
        return self.registry.report

    @property
    def diagnostics(self) -> AttentionCodebookDiagnostics | None:
        return self.registry.diagnostics

    def __len__(self) -> int:
        return self.logical_layers

    def __getitem__(self, layer_index: int) -> AttentionCodebookLayerView:
        return self.layers[layer_index]

    def forward(
        self, inputs: Tensor, layer_index: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.registry(inputs, layer_index)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
    ) -> GlobalAttentionCodebook:
        """Rebuild a complete runtime from an ordinary tensor state mapping."""

        prefix = "registry" if "registry.prototype_q_weight" in state_dict else ""
        registry = AttentionCodebookRegistry.from_state_dict(state_dict, prefix=prefix)
        result = cls(registry)
        if prefix:
            incompatible = result.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise AttentionCodebookStateError(
                    "state_dict did not exactly match the reconstructed codebook."
                )
        return result


def _resolve_prototypes_per_group(
    *,
    logical_layers: int,
    groups: int,
    prototype_count: int | None,
    codebook_size: int | None,
    num_prototypes: int | None,
    prototypes_per_group: int | None,
    depth_prototype_count: int | None,
) -> int:
    total = _one_value(
        "prototype_count", prototype_count, codebook_size, num_prototypes
    )
    per_group = _one_value(
        "prototypes_per_group", prototypes_per_group, depth_prototype_count
    )
    if total is not None and per_group is not None:
        raise AttentionCodebookShapeError(
            "Specify a total prototype count or prototypes_per_group, not both."
        )
    if total is None and per_group is None:
        return logical_layers
    if per_group is not None:
        result = _positive_int(per_group, "prototypes_per_group")
    else:
        total = _positive_int(total, "prototype_count")
        if total % groups != 0:
            raise AttentionCodebookShapeError(
                "prototype_count must be divisible by key_value_heads so every "
                "aligned group receives equal depth capacity."
            )
        result = total // groups
    if result > logical_layers:
        raise AttentionCodebookShapeError(
            "prototypes_per_group cannot exceed logical layer count."
        )
    return result


def _resolve_scale_request(
    *,
    use_scales: bool,
    per_group_scales: bool | Tensor | Mapping[str, Tensor] | None,
    role_scales: Tensor | Mapping[str, Tensor] | None,
    per_layer_group_scales: Tensor | Mapping[str, Tensor] | None,
) -> bool | Tensor | Mapping[str, Tensor]:
    if not isinstance(use_scales, bool):
        raise AttentionCodebookShapeError("use_scales must be boolean.")
    supplied = [
        value
        for value in (per_group_scales, role_scales, per_layer_group_scales)
        if value is not None
    ]
    if len(supplied) > 1 or (use_scales and supplied):
        raise AttentionCodebookShapeError("Specify only one scale request or tensor.")
    if not supplied:
        return use_scales
    result = supplied[0]
    if not isinstance(result, (bool, Tensor, Mapping)):
        raise AttentionCodebookShapeError(
            "Per-group scales must be boolean, a tensor, or a role mapping."
        )
    return result


def fit_global_attention_codebook(
    bundles: Sequence[Any],
    prototype_count: int | None = None,
    *,
    codebook_size: int | None = None,
    num_prototypes: int | None = None,
    prototypes_per_group: int | None = None,
    depth_prototype_count: int | None = None,
    query_heads: int | None = None,
    num_query_heads: int | None = None,
    num_attention_heads: int | None = None,
    num_heads: int | None = None,
    key_value_heads: int | None = None,
    num_key_value_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    group_permutations: Tensor | Sequence[Sequence[int]] | None = None,
    kv_group_permutations: Tensor | Sequence[Sequence[int]] | None = None,
    per_layer_group_permutations: Tensor | Sequence[Sequence[int]] | None = None,
    use_scales: bool = False,
    per_group_scales: bool | Tensor | Mapping[str, Tensor] | None = None,
    role_scales: Tensor | Mapping[str, Tensor] | None = None,
    per_layer_group_scales: Tensor | Mapping[str, Tensor] | None = None,
    sketch_dim: int = DEFAULT_SKETCH_DIM,
    weight_sketch_size: int | None = None,
) -> GlobalAttentionCodebook:
    """Fit real coupled GQA-group medoids independently across aligned depth.

    A permutation row maps canonical group indices to physical source groups.
    It must be bijective, and all four coupled projection slices are gathered
    through the same map. prototype_count is the total across groups;
    prototypes_per_group is the unambiguous depth capacity per group.
    """

    resolved_query_heads = _one_value(
        "query_heads",
        query_heads,
        num_query_heads,
        num_attention_heads,
        num_heads,
    )
    resolved_kv_heads = _one_value(
        "key_value_heads",
        key_value_heads,
        num_key_value_heads,
        num_kv_heads,
    )
    permutation_values = [
        value
        for value in (
            group_permutations,
            kv_group_permutations,
            per_layer_group_permutations,
        )
        if value is not None
    ]
    if len(permutation_values) > 1:
        raise AttentionCodebookShapeError(
            "Specify only one per-layer group permutation matrix."
        )
    requested_sketch = _one_value(
        "sketch_dim",
        sketch_dim if weight_sketch_size is None else None,
        weight_sketch_size,
    )
    requested_sketch = _positive_int(requested_sketch, "sketch_dim")
    scale_request = _resolve_scale_request(
        use_scales=use_scales,
        per_group_scales=per_group_scales,
        role_scales=role_scales,
        per_layer_group_scales=per_layer_group_scales,
    )
    with torch.no_grad():
        sources, topology = _validated_bundles(
            bundles,
            query_heads=resolved_query_heads,
            key_value_heads=resolved_kv_heads,
            head_dim=head_dim,
        )
        logical_layers = len(sources)
        groups = topology.key_value_heads
        per_group = _resolve_prototypes_per_group(
            logical_layers=logical_layers,
            groups=groups,
            prototype_count=prototype_count,
            codebook_size=codebook_size,
            num_prototypes=num_prototypes,
            prototypes_per_group=prototypes_per_group,
            depth_prototype_count=depth_prototype_count,
        )
        permutations = _validated_permutations(
            permutation_values[0] if permutation_values else None,
            logical_layers,
            groups,
        )
        features = _feature_sketch(sources, topology, permutations, requested_sketch)
        (
            medoid_layers,
            assignments,
            assigned_distances,
            feature_operations,
            max_pairwise_block,
        ) = _fit_aligned_medoids(features, per_group)
        (
            prototype_q_weight,
            prototype_k_weight,
            prototype_v_weight,
            prototype_o_weight,
            prototype_q_bias,
            prototype_k_bias,
            prototype_v_bias,
            prototype_sources,
        ) = _gather_prototypes(sources, topology, permutations, medoid_layers)
        layer_o_bias = (
            torch.stack([source.o_bias.detach().clone() for source in sources])
            if sources[0].o_bias is not None
            else None
        )
        if isinstance(scale_request, (Tensor, Mapping)):
            fitted_scales = _validated_role_scales(
                scale_request,
                logical_layers,
                groups,
                sources[0].q_weight,
            )
        elif scale_request:
            fitted_scales = _fitted_role_scales(
                sources,
                topology,
                permutations,
                assignments,
                (
                    prototype_q_weight,
                    prototype_k_weight,
                    prototype_v_weight,
                    prototype_o_weight,
                ),
                (
                    prototype_q_bias,
                    prototype_k_bias,
                    prototype_v_bias,
                ),
            )
        else:
            fitted_scales = None
        actual_sketch_dim = int(features.shape[2])
        pairwise_values = (
            0
            if per_group == logical_layers
            else groups * logical_layers * logical_layers
        )
        work_bound = groups * logical_layers * logical_layers * actual_sketch_dim
        identity = torch.arange(groups, dtype=torch.int64).repeat(logical_layers, 1)
        diagnostics = AttentionCodebookDiagnostics(
            algorithm="aligned_group_farthest_first_real_medoid",
            logical_layers=logical_layers,
            key_value_groups=groups,
            logical_groups=logical_layers * groups,
            prototypes_per_group=per_group,
            prototype_count=groups * per_group,
            deterministic=True,
            tie_break=_TIE_BREAK,
            real_coupled_group_medoids=True,
            used_group_permutations=not torch.equal(permutations, identity),
            permutations_provided=bool(permutation_values),
            permutations_bijective=True,
            requested_sketch_dim=requested_sketch,
            sketch_dimensions=actual_sketch_dim,
            sketch_elements=int(features.numel()),
            pairwise_distance_values=pairwise_values,
            distance_feature_operations=feature_operations,
            distance_work_bound=work_bound,
            maximum_pairwise_block_elements=max_pairwise_block,
            full_global_pairwise_matrix_elements=(logical_layers * groups) ** 2,
            materialized_global_pairwise_matrix=False,
            bounded_work=feature_operations <= work_bound,
            fitting_requires_grad=False,
            squared_sketch_error=float(assigned_distances.sum().item()),
            maximum_assigned_squared_distance=float(assigned_distances.max().item()),
        )
        registry = AttentionCodebookRegistry(
            prototype_q_weight,
            prototype_k_weight,
            prototype_v_weight,
            prototype_o_weight,
            assignments,
            prototype_sources,
            permutations,
            prototype_q_bias=prototype_q_bias,
            prototype_k_bias=prototype_k_bias,
            prototype_v_bias=prototype_v_bias,
            layer_o_bias=layer_o_bias,
            role_scales=fitted_scales,
            diagnostics=diagnostics,
        )
    return GlobalAttentionCodebook(registry)


fit_attention_codebook = fit_global_attention_codebook
build_global_attention_codebook = fit_global_attention_codebook
build_attention_head_codebook = fit_global_attention_codebook
TopologyPreservingAttentionCodebook = GlobalAttentionCodebook
ParameterlessAttentionCodebookLayer = AttentionCodebookLayerView
AttentionHeadCodebookRegistry = AttentionCodebookRegistry
AttentionHeadCodebookLayerView = AttentionCodebookLayerView
GlobalAttentionHeadCodebook = GlobalAttentionCodebook


__all__ = [
    "DEFAULT_SKETCH_DIM",
    "AttentionCodebookDiagnostics",
    "AttentionCodebookError",
    "AttentionCodebookFiniteError",
    "AttentionCodebookLayerView",
    "AttentionCodebookParameterReport",
    "AttentionCodebookProjectionView",
    "AttentionCodebookRegistry",
    "AttentionCodebookShapeError",
    "AttentionCodebookStateError",
    "AttentionHeadCodebookLayerView",
    "AttentionHeadCodebookRegistry",
    "AttentionProjectionBundle",
    "GQAAttentionBundle",
    "GlobalAttentionCodebook",
    "GlobalAttentionHeadCodebook",
    "ParameterlessAttentionCodebookLayer",
    "SplitQKVAttentionBundle",
    "TopologyPreservingAttentionCodebook",
    "build_attention_head_codebook",
    "build_global_attention_codebook",
    "fit_attention_codebook",
    "fit_global_attention_codebook",
]
