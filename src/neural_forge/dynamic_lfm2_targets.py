"""Exact legal target planning for the LFM2 folding adapter.

The original LFM experiment changed every attention layer by a full GQA group
and every MLP by 32 neurons before the first closed-loop transaction could be
evaluated.  This planner creates the smallest representable stages first:
attention remains exact while MLP width is reduced geometrically, then GQA
groups are removed one legal group at a time.  It also publishes an immutable
10/20/30/50 percent hybrid frontier that jointly allocates depth, GQA groups,
and MLP width while assigning every source layer to one ordered target span.
The stretch 75 percent budget is represented separately as an explicitly
composite hybrid-plus-factorized-vocabulary target because the unchanged dense
token interface alone is larger than the complete 75 percent child budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .dynamic_circuit_ir import canonical_sha256


SCHEMA = "neural-forge-dynamic-lfm2-progressive-targets-v1"
HYBRID_SCHEMA = "neural-forge-dynamic-lfm2-hybrid-target-v1"
HYBRID_FACTORIZED_VOCAB_SCHEMA = (
    "neural-forge-dynamic-lfm2-hybrid-factorized-vocab-target-v1"
)

LFM2_230M_SOURCE_UNIQUE_PARAMETERS = 229_693_184
LFM2_230M_BASE_UNIQUE_PARAMETERS = 67_109_888
LFM2_230M_HIDDEN_SIZE = 1_024
LFM2_230M_HEAD_DIM = 64
LFM2_230M_VOCAB_SIZE = 65_536
LFM2_230M_DENSE_TIED_VOCAB_UNIQUE_PARAMETERS = (
    LFM2_230M_VOCAB_SIZE * LFM2_230M_HIDDEN_SIZE
)
LFM2_230M_FACTORIZED_TIED_VOCAB_PARAMETERS_PER_RANK = (
    LFM2_230M_VOCAB_SIZE + LFM2_230M_HIDDEN_SIZE
)
LFM2_230M_CONV_FIXED_UNIQUE_PARAMETERS = 4_199_424
LFM2_230M_ATTENTION_FIXED_UNIQUE_PARAMETERS = 2_176
LFM2_230M_SOURCE_LAYER_TYPES = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)


class DynamicLfm2TargetError(ValueError):
    pass


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicLfm2TargetError(f"{name} must be a positive integer.")
    return value


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicLfm2TargetError(f"{name} must be a non-negative integer.")
    return value


@dataclass(frozen=True, slots=True)
class Lfm2Topology:
    query_heads: int
    key_value_heads: int
    intermediate_size: int

    def __post_init__(self) -> None:
        for name in ("query_heads", "key_value_heads", "intermediate_size"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.query_heads % self.key_value_heads:
            raise DynamicLfm2TargetError(
                "query_heads must be divisible by key_value_heads."
            )

    @property
    def queries_per_kv_group(self) -> int:
        return self.query_heads // self.key_value_heads

    def as_dict(self) -> dict[str, int]:
        return {
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "intermediate_size": self.intermediate_size,
            "queries_per_kv_group": self.queries_per_kv_group,
        }


def _layer_types(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if not result or any(
        value not in {"full_attention", "conv"} for value in result
    ):
        raise DynamicLfm2TargetError(
            f"{name} must contain only full_attention or conv."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2OrderedSourceSpan:
    """One target layer's complete, contiguous source-layer responsibility."""

    target_layer_index: int
    source_layer_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_layer_index",
            _nonnegative(self.target_layer_index, "target_layer_index"),
        )
        source_indices = tuple(self.source_layer_indices)
        if not source_indices:
            raise DynamicLfm2TargetError(
                "source_layer_indices must contain at least one source layer."
            )
        for source_index in source_indices:
            _nonnegative(source_index, "source_layer_index")
        expected = tuple(
            range(source_indices[0], source_indices[0] + len(source_indices))
        )
        if source_indices != expected:
            raise DynamicLfm2TargetError(
                "Each source span must be ordered, contiguous, and duplicate-free."
            )
        object.__setattr__(self, "source_layer_indices", source_indices)

    @property
    def source_start(self) -> int:
        return self.source_layer_indices[0]

    @property
    def source_stop(self) -> int:
        return self.source_layer_indices[-1] + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_layer_index": self.target_layer_index,
            "source_layer_indices": list(self.source_layer_indices),
            "source_start": self.source_start,
            "source_stop": self.source_stop,
        }


def lfm2_230m_hybrid_parameter_count(
    *,
    layer_types: Sequence[str],
    topology: Lfm2Topology,
) -> int:
    """Return the exact unique learned parameters for a legal 230M hybrid."""

    if not isinstance(topology, Lfm2Topology):
        raise DynamicLfm2TargetError("topology must be Lfm2Topology.")
    target_layer_types = _layer_types(layer_types, "layer_types")
    source = Lfm2Topology(16, 8, 2_560)
    if topology.query_heads > source.query_heads:
        raise DynamicLfm2TargetError("Target query heads exceed the source.")
    if topology.key_value_heads > source.key_value_heads:
        raise DynamicLfm2TargetError("Target key/value heads exceed the source.")
    if topology.intermediate_size > source.intermediate_size:
        raise DynamicLfm2TargetError("Target MLP width exceeds the source.")
    if topology.queries_per_kv_group != source.queries_per_kv_group:
        raise DynamicLfm2TargetError(
            "The LFM2 230M hybrid frontier preserves the source GQA ratio."
        )

    layer_count = len(target_layer_types)
    convolution_count = target_layer_types.count("conv")
    attention_count = target_layer_types.count("full_attention")
    mlp_parameters = (
        layer_count
        * 3
        * LFM2_230M_HIDDEN_SIZE
        * topology.intermediate_size
    )
    attention_projection_parameters = (
        attention_count
        * 2
        * LFM2_230M_HIDDEN_SIZE
        * LFM2_230M_HEAD_DIM
        * (topology.query_heads + topology.key_value_heads)
    )
    return (
        LFM2_230M_BASE_UNIQUE_PARAMETERS
        + mlp_parameters
        + convolution_count * LFM2_230M_CONV_FIXED_UNIQUE_PARAMETERS
        + attention_count * LFM2_230M_ATTENTION_FIXED_UNIQUE_PARAMETERS
        + attention_projection_parameters
    )


def lfm2_230m_hybrid_factorized_vocab_parameter_count(
    *,
    layer_types: Sequence[str],
    topology: Lfm2Topology,
    vocabulary_rank: int,
) -> int:
    """Count a legal hybrid whose tied dense token matrix is physically factored."""

    rank = _positive(vocabulary_rank, "vocabulary_rank")
    if rank > LFM2_230M_HIDDEN_SIZE:
        raise DynamicLfm2TargetError(
            "vocabulary_rank cannot exceed the source hidden size."
        )
    dense = lfm2_230m_hybrid_parameter_count(
        layer_types=layer_types,
        topology=topology,
    )
    return (
        dense
        - LFM2_230M_DENSE_TIED_VOCAB_UNIQUE_PARAMETERS
        + rank * LFM2_230M_FACTORIZED_TIED_VOCAB_PARAMETERS_PER_RANK
    )


@dataclass(frozen=True, slots=True)
class Lfm2HybridTarget:
    """An exact, executable-shape target with total ordered source coverage."""

    requested_reduction_basis_points: int
    topology: Lfm2Topology
    layer_types: tuple[str, ...]
    unique_parameters: int
    source_spans: tuple[Lfm2OrderedSourceSpan, ...]
    source_layer_count: int = len(LFM2_230M_SOURCE_LAYER_TYPES)
    source_unique_parameters: int = LFM2_230M_SOURCE_UNIQUE_PARAMETERS

    def __post_init__(self) -> None:
        requested_reduction = _positive(
            self.requested_reduction_basis_points,
            "requested_reduction_basis_points",
        )
        if requested_reduction >= 10_000:
            raise DynamicLfm2TargetError(
                "requested_reduction_basis_points must be below 10,000."
            )
        if not isinstance(self.topology, Lfm2Topology):
            raise DynamicLfm2TargetError("topology must be Lfm2Topology.")
        target_layer_types = _layer_types(self.layer_types, "layer_types")
        object.__setattr__(self, "layer_types", target_layer_types)
        object.__setattr__(
            self,
            "unique_parameters",
            _positive(self.unique_parameters, "unique_parameters"),
        )
        object.__setattr__(
            self,
            "source_layer_count",
            _positive(self.source_layer_count, "source_layer_count"),
        )
        object.__setattr__(
            self,
            "source_unique_parameters",
            _positive(self.source_unique_parameters, "source_unique_parameters"),
        )
        if self.source_layer_count != len(LFM2_230M_SOURCE_LAYER_TYPES):
            raise DynamicLfm2TargetError(
                "The exact LFM2 230M frontier requires 14 source layers."
            )
        if self.source_unique_parameters != LFM2_230M_SOURCE_UNIQUE_PARAMETERS:
            raise DynamicLfm2TargetError(
                "The exact LFM2 230M frontier requires its verified source count."
            )

        spans = tuple(self.source_spans)
        if any(not isinstance(span, Lfm2OrderedSourceSpan) for span in spans):
            raise DynamicLfm2TargetError(
                "source_spans must contain only Lfm2OrderedSourceSpan values."
            )
        object.__setattr__(self, "source_spans", spans)
        if len(spans) != len(target_layer_types):
            raise DynamicLfm2TargetError(
                "There must be exactly one source span per target layer."
            )
        target_indices = tuple(span.target_layer_index for span in spans)
        if target_indices != tuple(range(len(target_layer_types))):
            raise DynamicLfm2TargetError(
                "Source spans must follow target layers in exact order."
            )
        covered_source_indices = tuple(
            source_index
            for span in spans
            for source_index in span.source_layer_indices
        )
        if covered_source_indices != tuple(range(self.source_layer_count)):
            raise DynamicLfm2TargetError(
                "Source spans must provide complete ordered source coverage."
            )
        for target_type, span in zip(target_layer_types, spans, strict=True):
            source_types = {
                LFM2_230M_SOURCE_LAYER_TYPES[source_index]
                for source_index in span.source_layer_indices
            }
            if target_type not in source_types:
                raise DynamicLfm2TargetError(
                    "Every target layer must have a source operator-compatible "
                    "anchor inside its assigned span."
                )

        expected_parameters = lfm2_230m_hybrid_parameter_count(
            layer_types=target_layer_types,
            topology=self.topology,
        )
        if self.unique_parameters != expected_parameters:
            raise DynamicLfm2TargetError(
                "Hybrid target unique parameter accounting is inconsistent: "
                f"expected {expected_parameters}, received {self.unique_parameters}."
            )
        if self.unique_parameters >= self.source_unique_parameters:
            raise DynamicLfm2TargetError(
                "A hybrid compression target must remove learned parameters."
            )
        exact_reduction_numerator = self.parameters_removed * 10_000
        requested_reduction_numerator = (
            self.requested_reduction_basis_points * self.source_unique_parameters
        )
        if (
            abs(exact_reduction_numerator - requested_reduction_numerator)
            > self.source_unique_parameters
        ):
            raise DynamicLfm2TargetError(
                "The exact target must be within one basis point of its "
                "requested reduction."
            )

    @property
    def layer_count(self) -> int:
        return len(self.layer_types)

    @property
    def source_layer_types(self) -> tuple[str, ...]:
        return LFM2_230M_SOURCE_LAYER_TYPES

    @property
    def attention_layer_count(self) -> int:
        return self.layer_types.count("full_attention")

    @property
    def convolution_layer_count(self) -> int:
        return self.layer_types.count("conv")

    @property
    def parameters_removed(self) -> int:
        return self.source_unique_parameters - self.unique_parameters

    @property
    def achieved_reduction(self) -> float:
        return self.parameters_removed / self.source_unique_parameters

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HYBRID_SCHEMA,
            "requested_reduction_basis_points": (
                self.requested_reduction_basis_points
            ),
            "topology": self.topology.as_dict(),
            "layer_types": list(self.layer_types),
            "layer_count": self.layer_count,
            "attention_layer_count": self.attention_layer_count,
            "convolution_layer_count": self.convolution_layer_count,
            "source_layer_count": self.source_layer_count,
            "source_layer_types": list(self.source_layer_types),
            "source_unique_parameters": self.source_unique_parameters,
            "unique_parameters": self.unique_parameters,
            "parameters_removed": self.parameters_removed,
            "exact_reduction_fraction": {
                "numerator": self.parameters_removed,
                "denominator": self.source_unique_parameters,
            },
            "source_spans": [span.as_dict() for span in self.source_spans],
            "complete_ordered_source_coverage": True,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def _ordered_source_spans(
    *source_layer_indices: tuple[int, ...],
) -> tuple[Lfm2OrderedSourceSpan, ...]:
    return tuple(
        Lfm2OrderedSourceSpan(
            target_layer_index=target_layer_index,
            source_layer_indices=indices,
        )
        for target_layer_index, indices in enumerate(source_layer_indices)
    )


def _hybrid_target(
    *,
    requested_reduction_basis_points: int,
    query_heads: int,
    key_value_heads: int,
    intermediate_size: int,
    layer_types: tuple[str, ...],
    unique_parameters: int,
    source_spans: tuple[Lfm2OrderedSourceSpan, ...],
) -> Lfm2HybridTarget:
    return Lfm2HybridTarget(
        requested_reduction_basis_points=requested_reduction_basis_points,
        topology=Lfm2Topology(
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            intermediate_size=intermediate_size,
        ),
        layer_types=layer_types,
        unique_parameters=unique_parameters,
        source_spans=source_spans,
    )


LFM2_230M_HYBRID_TARGET_FRONTIER = (
    _hybrid_target(
        requested_reduction_basis_points=1_000,
        query_heads=14,
        key_value_heads=7,
        intermediate_size=2_346,
        layer_types=(
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
        unique_parameters=206_723_840,
        source_spans=_ordered_source_spans(
            (0, 1),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
        ),
    ),
    _hybrid_target(
        requested_reduction_basis_points=2_000,
        query_heads=14,
        key_value_heads=7,
        intermediate_size=2_299,
        layer_types=(
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
        unique_parameters=183_767_680,
        source_spans=_ordered_source_spans(
            (0, 1),
            (2,),
            (3,),
            (4,),
            (5, 6),
            (7, 8),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
        ),
    ),
    _hybrid_target(
        requested_reduction_basis_points=3_000,
        query_heads=14,
        key_value_heads=7,
        intermediate_size=2_230,
        layer_types=(
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
        unique_parameters=160_780_800,
        source_spans=_ordered_source_spans(
            (0, 1),
            (2,),
            (3,),
            (4,),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13,),
        ),
    ),
    _hybrid_target(
        requested_reduction_basis_points=5_000,
        query_heads=14,
        key_value_heads=7,
        intermediate_size=1_458,
        layer_types=(
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
        ),
        unique_parameters=114_846_080,
        source_spans=_ordered_source_spans(
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9, 10),
            (11, 12, 13),
        ),
    ),
)


def lfm2_230m_hybrid_target_frontier() -> tuple[Lfm2HybridTarget, ...]:
    """Return the immutable, increasing 10/20/30/50 percent target frontier."""

    return LFM2_230M_HYBRID_TARGET_FRONTIER


def solve_lfm2_230m_hybrid_structural_target(
    *,
    layer_types: Sequence[str],
    source_spans: Sequence[Lfm2OrderedSourceSpan],
    requested_reduction_basis_points: int = 5_000,
) -> Lfm2HybridTarget:
    """Jointly solve legal GQA groups and FFN width under one exact ceiling.

    The primary objective retains the largest representable parameter count
    below the requested child budget.  Equal-count solutions retain more
    attention heads, leaving the causal evaluator—not an arbitrary uniform
    shrink rule—to decide whether a lower-head/wider-FFN tradeoff is better.
    """

    requested = _positive(
        requested_reduction_basis_points,
        "requested_reduction_basis_points",
    )
    if requested >= 10_000:
        raise DynamicLfm2TargetError(
            "requested_reduction_basis_points must be below 10,000."
        )
    target_layer_types = _layer_types(layer_types, "layer_types")
    spans = tuple(source_spans)
    target_ceiling = (
        LFM2_230M_SOURCE_UNIQUE_PARAMETERS * (10_000 - requested)
    ) // 10_000
    mlp_parameters_per_width = (
        len(target_layer_types) * 3 * LFM2_230M_HIDDEN_SIZE
    )
    choices: list[tuple[int, int, int, int]] = []
    for key_value_heads in range(1, 9):
        query_heads = key_value_heads * 2
        unit_topology = Lfm2Topology(
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            intermediate_size=1,
        )
        fixed_parameters = (
            lfm2_230m_hybrid_parameter_count(
                layer_types=target_layer_types,
                topology=unit_topology,
            )
            - mlp_parameters_per_width
        )
        intermediate_size = min(
            2_560,
            (target_ceiling - fixed_parameters)
            // mlp_parameters_per_width,
        )
        if intermediate_size < 1:
            continue
        topology = Lfm2Topology(
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            intermediate_size=intermediate_size,
        )
        unique_parameters = lfm2_230m_hybrid_parameter_count(
            layer_types=target_layer_types,
            topology=topology,
        )
        if unique_parameters > target_ceiling:
            continue
        reduction_error = abs(
            (
                LFM2_230M_SOURCE_UNIQUE_PARAMETERS - unique_parameters
            )
            * 10_000
            - requested * LFM2_230M_SOURCE_UNIQUE_PARAMETERS
        )
        if reduction_error > LFM2_230M_SOURCE_UNIQUE_PARAMETERS:
            continue
        choices.append(
            (
                target_ceiling - unique_parameters,
                -query_heads,
                intermediate_size,
                unique_parameters,
            )
        )
    if not choices:
        raise DynamicLfm2TargetError(
            "No legal GQA/FFN topology reaches the requested structural "
            "budget within one basis point."
        )
    headroom, negative_query_heads, intermediate_size, unique_parameters = min(
        choices
    )
    del headroom
    query_heads = -negative_query_heads
    return _hybrid_target(
        requested_reduction_basis_points=requested,
        query_heads=query_heads,
        key_value_heads=query_heads // 2,
        intermediate_size=intermediate_size,
        layer_types=target_layer_types,
        unique_parameters=unique_parameters,
        source_spans=spans,
    )


_LFM2_230M_HALF_SIZE_DEPTH_LAYOUTS = (
    (
        (
            "conv",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
        _ordered_source_spans(
            (0, 1),
            (2, 3),
            (4,),
            (5, 6),
            (7, 8),
            (9,),
            (10, 11),
            (12, 13),
        ),
    ),
    (
        (
            "conv",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
        _ordered_source_spans(
            (0,),
            (1,),
            (2, 3),
            (4, 5),
            (6,),
            (7,),
            (8,),
            (9, 10),
            (11, 12),
            (13,),
        ),
    ),
    (
        (
            "conv",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
        _ordered_source_spans(
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5, 6),
            (7, 8),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
        ),
    ),
    (
        LFM2_230M_SOURCE_LAYER_TYPES,
        _ordered_source_spans(
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
        ),
    ),
)


LFM2_230M_HALF_SIZE_STRUCTURAL_FRONTIER = tuple(
    solve_lfm2_230m_hybrid_structural_target(
        layer_types=layer_types,
        source_spans=source_spans,
    )
    for layer_types, source_spans in _LFM2_230M_HALF_SIZE_DEPTH_LAYOUTS
)


def lfm2_230m_half_size_structural_frontier(
) -> tuple[Lfm2HybridTarget, ...]:
    """Return the depth 8/10/12/14 joint GQA/FFN 50-percent frontier."""

    return LFM2_230M_HALF_SIZE_STRUCTURAL_FRONTIER


def lfm2_230m_half_size_structural_support_report(
) -> tuple[dict[str, Any], ...]:
    """Describe the already implemented physical route for every frontier point."""

    target_ceiling = LFM2_230M_SOURCE_UNIQUE_PARAMETERS // 2
    reports = []
    for target in LFM2_230M_HALF_SIZE_STRUCTURAL_FRONTIER:
        shallower = target.layer_count < len(LFM2_230M_SOURCE_LAYER_TYPES)
        reports.append(
            {
                "target_fingerprint": target.fingerprint,
                "depth": target.layer_count,
                "topology": target.topology.as_dict(),
                "unique_parameters": target.unique_parameters,
                "parameters_removed": target.parameters_removed,
                "budget_headroom": target_ceiling - target.unique_parameters,
                "target_selector": (
                    "dynamic_lfm2_adapter.select_lfm2_hybrid_target"
                    if shallower
                    else "dynamic_lfm2_adapter.select_exact_lfm2_target"
                ),
                "blank_materializer": (
                    "dynamic_lfm2_adapter.materialize_hybrid_lfm2_child"
                    if shallower
                    else "dynamic_lfm2_adapter.materialize_selected_lfm2_child"
                ),
                "producer_first_installer": (
                    "hf_lfm2_hybrid_producer_first_model_fold."
                    "install_lfm2_hybrid_producer_first_model_fold"
                    if shallower
                    else "hf_lfm2_producer_first_model_fold."
                    "install_lfm2_producer_first_model_fold"
                ),
                "capture_contract": (
                    "hf_lfm2_producer_first_model_fold."
                    "Lfm2ProducerFirstEvidence.capture_child"
                ),
                "residual_attention_installer": (
                    "hf_lfm2_model_fold.install_lfm2_model_fold_plan"
                ),
                "physical_route_implemented": True,
                "causal_intelligence_retention_proven": False,
            }
        )
    return tuple(reports)


@dataclass(frozen=True, slots=True)
class Lfm2HybridFactorizedVocabularyTarget:
    """One mechanically exact hybrid-plus-token-interface stretch budget.

    This is target/accounting evidence only.  It deliberately does not promote
    raw matrix SVD, sparse residual rows, or any other vocabulary fold as an
    intelligence-retention winner.
    """

    requested_reduction_basis_points: int
    dense_hybrid_target: Lfm2HybridTarget
    vocabulary_rank: int
    unique_parameters: int
    budget_headroom: int

    def __post_init__(self) -> None:
        requested = _positive(
            self.requested_reduction_basis_points,
            "requested_reduction_basis_points",
        )
        if requested >= 10_000:
            raise DynamicLfm2TargetError(
                "requested_reduction_basis_points must be below 10,000."
            )
        if not isinstance(self.dense_hybrid_target, Lfm2HybridTarget):
            raise DynamicLfm2TargetError(
                "dense_hybrid_target must be an Lfm2HybridTarget."
            )
        rank = _positive(self.vocabulary_rank, "vocabulary_rank")
        if rank > LFM2_230M_HIDDEN_SIZE:
            raise DynamicLfm2TargetError(
                "vocabulary_rank cannot exceed the source hidden size."
            )
        unique = _positive(self.unique_parameters, "unique_parameters")
        headroom = _nonnegative(self.budget_headroom, "budget_headroom")
        expected = lfm2_230m_hybrid_factorized_vocab_parameter_count(
            layer_types=self.dense_hybrid_target.layer_types,
            topology=self.dense_hybrid_target.topology,
            vocabulary_rank=rank,
        )
        requested_target = (
            LFM2_230M_SOURCE_UNIQUE_PARAMETERS * (10_000 - requested)
        ) // 10_000
        if unique != expected:
            raise DynamicLfm2TargetError(
                "Composite hybrid vocabulary parameter accounting is "
                "inconsistent."
            )
        if unique > requested_target:
            raise DynamicLfm2TargetError(
                "Composite hybrid vocabulary target exceeds its requested "
                "parameter ceiling."
            )
        if headroom != requested_target - unique:
            raise DynamicLfm2TargetError(
                "Composite hybrid vocabulary budget headroom is inconsistent."
            )
        object.__setattr__(self, "vocabulary_rank", rank)
        object.__setattr__(self, "unique_parameters", unique)
        object.__setattr__(self, "budget_headroom", headroom)

    @property
    def requested_target_unique_parameters(self) -> int:
        return (
            LFM2_230M_SOURCE_UNIQUE_PARAMETERS
            * (10_000 - self.requested_reduction_basis_points)
        ) // 10_000

    @property
    def parameters_removed(self) -> int:
        return LFM2_230M_SOURCE_UNIQUE_PARAMETERS - self.unique_parameters

    @property
    def achieved_reduction(self) -> float:
        return self.parameters_removed / LFM2_230M_SOURCE_UNIQUE_PARAMETERS

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HYBRID_FACTORIZED_VOCAB_SCHEMA,
            "requested_reduction_basis_points": (
                self.requested_reduction_basis_points
            ),
            "requested_target_unique_parameters": (
                self.requested_target_unique_parameters
            ),
            "unique_parameters": self.unique_parameters,
            "parameters_removed": self.parameters_removed,
            "achieved_reduction": self.achieved_reduction,
            "budget_headroom": self.budget_headroom,
            "dense_hybrid_target": self.dense_hybrid_target.as_dict(),
            "vocabulary": {
                "vocab_size": LFM2_230M_VOCAB_SIZE,
                "hidden_size": LFM2_230M_HIDDEN_SIZE,
                "rank": self.vocabulary_rank,
                "dense_unique_parameters": (
                    LFM2_230M_DENSE_TIED_VOCAB_UNIQUE_PARAMETERS
                ),
                "factor_unique_parameters": (
                    self.vocabulary_rank
                    * LFM2_230M_FACTORIZED_TIED_VOCAB_PARAMETERS_PER_RANK
                ),
                "dense_matrix_retained": False,
                "embedding_and_head_factors_physically_shared": True,
            },
            "mechanical_budget_only": True,
            "intelligence_retention_proven": False,
            "joint_causal_allocation_required": True,
            "composite_installer_required": True,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def solve_lfm2_230m_hybrid_factorized_vocab_target(
    dense_hybrid_target: Lfm2HybridTarget,
    *,
    requested_reduction_basis_points: int,
) -> Lfm2HybridFactorizedVocabularyTarget:
    """Fit the largest positive shared vocabulary rank under a total budget."""

    if not isinstance(dense_hybrid_target, Lfm2HybridTarget):
        raise DynamicLfm2TargetError(
            "dense_hybrid_target must be an Lfm2HybridTarget."
        )
    requested = _positive(
        requested_reduction_basis_points,
        "requested_reduction_basis_points",
    )
    if requested >= 10_000:
        raise DynamicLfm2TargetError(
            "requested_reduction_basis_points must be below 10,000."
        )
    requested_target = (
        LFM2_230M_SOURCE_UNIQUE_PARAMETERS * (10_000 - requested)
    ) // 10_000
    fixed_non_vocabulary = (
        dense_hybrid_target.unique_parameters
        - LFM2_230M_DENSE_TIED_VOCAB_UNIQUE_PARAMETERS
    )
    rank = min(
        LFM2_230M_HIDDEN_SIZE,
        (requested_target - fixed_non_vocabulary)
        // LFM2_230M_FACTORIZED_TIED_VOCAB_PARAMETERS_PER_RANK,
    )
    if rank < 1:
        raise DynamicLfm2TargetError(
            "The requested composite target cannot retain a positive tied "
            "vocabulary rank with this hybrid body."
        )
    unique = lfm2_230m_hybrid_factorized_vocab_parameter_count(
        layer_types=dense_hybrid_target.layer_types,
        topology=dense_hybrid_target.topology,
        vocabulary_rank=rank,
    )
    return Lfm2HybridFactorizedVocabularyTarget(
        requested_reduction_basis_points=requested,
        dense_hybrid_target=dense_hybrid_target,
        vocabulary_rank=rank,
        unique_parameters=unique,
        budget_headroom=requested_target - unique,
    )


LFM2_230M_HYBRID_FACTORIZED_VOCAB_TARGET_FRONTIER = (
    solve_lfm2_230m_hybrid_factorized_vocab_target(
        LFM2_230M_HYBRID_TARGET_FRONTIER[-1],
        requested_reduction_basis_points=7_500,
    ),
)


def lfm2_230m_hybrid_factorized_vocab_target_frontier(
) -> tuple[Lfm2HybridFactorizedVocabularyTarget, ...]:
    """Return the explicit mechanically feasible 75 percent stretch target."""

    return LFM2_230M_HYBRID_FACTORIZED_VOCAB_TARGET_FRONTIER


@dataclass(frozen=True, slots=True)
class Lfm2ProgressiveStage:
    stage_index: int
    topology: Lfm2Topology
    changed_circuit: str
    previous_unique_parameters: int
    unique_parameters: int
    incremental_parameters_removed: int
    cumulative_parameters_removed: int
    parent_stage_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage_index", _nonnegative(self.stage_index, "stage_index")
        )
        if not isinstance(self.topology, Lfm2Topology):
            raise DynamicLfm2TargetError("topology must be Lfm2Topology.")
        if self.changed_circuit not in {"mlp", "attention"}:
            raise DynamicLfm2TargetError(
                "changed_circuit must be 'mlp' or 'attention'."
            )
        for name in (
            "previous_unique_parameters",
            "unique_parameters",
            "incremental_parameters_removed",
            "cumulative_parameters_removed",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.unique_parameters >= self.previous_unique_parameters:
            raise DynamicLfm2TargetError(
                "Each progressive stage must remove learned parameters."
            )
        if (
            self.previous_unique_parameters - self.unique_parameters
            != self.incremental_parameters_removed
        ):
            raise DynamicLfm2TargetError(
                "Stage incremental parameter accounting is inconsistent."
            )
        if (
            not isinstance(self.parent_stage_sha256, str)
            or len(self.parent_stage_sha256) != 64
        ):
            raise DynamicLfm2TargetError(
                "parent_stage_sha256 must be a SHA-256 digest."
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "topology": self.topology.as_dict(),
            "changed_circuit": self.changed_circuit,
            "previous_unique_parameters": self.previous_unique_parameters,
            "unique_parameters": self.unique_parameters,
            "incremental_parameters_removed": (self.incremental_parameters_removed),
            "cumulative_parameters_removed": self.cumulative_parameters_removed,
            "parent_stage_sha256": self.parent_stage_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ProgressiveTargetPlan:
    source: Lfm2Topology
    target: Lfm2Topology
    source_unique_parameters: int
    hidden_size: int
    head_dim: int
    layer_count: int
    attention_layer_count: int
    stages: tuple[Lfm2ProgressiveStage, ...]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "source": self.source.as_dict(),
            "target": self.target.as_dict(),
            "source_unique_parameters": self.source_unique_parameters,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "layer_count": self.layer_count,
            "attention_layer_count": self.attention_layer_count,
            "stages": [stage.as_dict() for stage in self.stages],
            "smallest_representable_stage_first": True,
            "attention_held_exact_during_initial_mlp_stages": True,
            "gqa_groups_removed_one_at_a_time": True,
            "final_unique_parameters": (
                self.source_unique_parameters
                if not self.stages
                else self.stages[-1].unique_parameters
            ),
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def _parameter_count(
    *,
    source_unique_parameters: int,
    source: Lfm2Topology,
    topology: Lfm2Topology,
    hidden_size: int,
    head_dim: int,
    layer_count: int,
    attention_layer_count: int,
) -> int:
    mlp_removed = (
        (source.intermediate_size - topology.intermediate_size)
        * 3
        * hidden_size
        * layer_count
    )
    query_heads_removed = source.query_heads - topology.query_heads
    key_value_heads_removed = source.key_value_heads - topology.key_value_heads
    attention_removed = (
        2
        * hidden_size
        * head_dim
        * (query_heads_removed + key_value_heads_removed)
        * attention_layer_count
    )
    result = source_unique_parameters - mlp_removed - attention_removed
    if result <= 0:
        raise DynamicLfm2TargetError(
            "Topology accounting removes at least the whole source model."
        )
    return result


def plan_lfm2_progressive_targets(
    *,
    source_unique_parameters: int,
    hidden_size: int,
    head_dim: int,
    layer_types: Sequence[str],
    source: Lfm2Topology,
    target: Lfm2Topology,
    initial_mlp_step: int = 1,
    maximum_mlp_step: int = 64,
) -> Lfm2ProgressiveTargetPlan:
    """Return a deterministic smallest-change-first topology ladder."""

    source_unique_parameters = _positive(
        source_unique_parameters, "source_unique_parameters"
    )
    hidden_size = _positive(hidden_size, "hidden_size")
    head_dim = _positive(head_dim, "head_dim")
    initial_mlp_step = _positive(initial_mlp_step, "initial_mlp_step")
    maximum_mlp_step = _positive(maximum_mlp_step, "maximum_mlp_step")
    layer_types = tuple(str(value).strip() for value in layer_types)
    if not layer_types or any(
        value not in {"full_attention", "conv"} for value in layer_types
    ):
        raise DynamicLfm2TargetError(
            "layer_types must contain only full_attention or conv."
        )
    if target.query_heads > source.query_heads:
        raise DynamicLfm2TargetError("Target query heads exceed the source.")
    if target.key_value_heads > source.key_value_heads:
        raise DynamicLfm2TargetError("Target key/value heads exceed the source.")
    if target.intermediate_size > source.intermediate_size:
        raise DynamicLfm2TargetError("Target MLP width exceeds the source.")
    if target.queries_per_kv_group != source.queries_per_kv_group:
        raise DynamicLfm2TargetError(
            "The current LFM operator requires a stable GQA grouping ratio."
        )
    layer_count = len(layer_types)
    attention_count = layer_types.count("full_attention")
    stages: list[Lfm2ProgressiveStage] = []
    current = source
    previous_parameters = source_unique_parameters
    source_identity = canonical_sha256(
        {
            "source": source.as_dict(),
            "source_unique_parameters": source_unique_parameters,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "layer_types": list(layer_types),
        }
    )
    parent_identity = source_identity

    total_mlp_reduction = source.intermediate_size - target.intermediate_size
    cumulative_mlp_reduction = min(initial_mlp_step, total_mlp_reduction)
    while current.intermediate_size > target.intermediate_size:
        next_topology = Lfm2Topology(
            current.query_heads,
            current.key_value_heads,
            source.intermediate_size - cumulative_mlp_reduction,
        )
        parameters = _parameter_count(
            source_unique_parameters=source_unique_parameters,
            source=source,
            topology=next_topology,
            hidden_size=hidden_size,
            head_dim=head_dim,
            layer_count=layer_count,
            attention_layer_count=attention_count,
        )
        stage = Lfm2ProgressiveStage(
            stage_index=len(stages),
            topology=next_topology,
            changed_circuit="mlp",
            previous_unique_parameters=previous_parameters,
            unique_parameters=parameters,
            incremental_parameters_removed=previous_parameters - parameters,
            cumulative_parameters_removed=source_unique_parameters - parameters,
            parent_stage_sha256=parent_identity,
        )
        stages.append(stage)
        current = next_topology
        previous_parameters = parameters
        parent_identity = stage.fingerprint
        if cumulative_mlp_reduction >= total_mlp_reduction:
            break
        if cumulative_mlp_reduction < maximum_mlp_step:
            cumulative_mlp_reduction = min(
                total_mlp_reduction,
                max(
                    cumulative_mlp_reduction + 1,
                    min(maximum_mlp_step, cumulative_mlp_reduction * 2),
                ),
            )
        else:
            cumulative_mlp_reduction = total_mlp_reduction

    group_ratio = source.queries_per_kv_group
    while current.key_value_heads > target.key_value_heads:
        next_topology = Lfm2Topology(
            current.query_heads - group_ratio,
            current.key_value_heads - 1,
            current.intermediate_size,
        )
        parameters = _parameter_count(
            source_unique_parameters=source_unique_parameters,
            source=source,
            topology=next_topology,
            hidden_size=hidden_size,
            head_dim=head_dim,
            layer_count=layer_count,
            attention_layer_count=attention_count,
        )
        stage = Lfm2ProgressiveStage(
            stage_index=len(stages),
            topology=next_topology,
            changed_circuit="attention",
            previous_unique_parameters=previous_parameters,
            unique_parameters=parameters,
            incremental_parameters_removed=previous_parameters - parameters,
            cumulative_parameters_removed=source_unique_parameters - parameters,
            parent_stage_sha256=parent_identity,
        )
        stages.append(stage)
        current = next_topology
        previous_parameters = parameters
        parent_identity = stage.fingerprint

    if current != target:
        raise DynamicLfm2TargetError(
            "Progressive construction did not reach the requested topology."
        )
    return Lfm2ProgressiveTargetPlan(
        source=source,
        target=target,
        source_unique_parameters=source_unique_parameters,
        hidden_size=hidden_size,
        head_dim=head_dim,
        layer_count=layer_count,
        attention_layer_count=attention_count,
        stages=tuple(stages),
    )


__all__ = [
    "DynamicLfm2TargetError",
    "HYBRID_FACTORIZED_VOCAB_SCHEMA",
    "HYBRID_SCHEMA",
    "LFM2_230M_ATTENTION_FIXED_UNIQUE_PARAMETERS",
    "LFM2_230M_BASE_UNIQUE_PARAMETERS",
    "LFM2_230M_CONV_FIXED_UNIQUE_PARAMETERS",
    "LFM2_230M_DENSE_TIED_VOCAB_UNIQUE_PARAMETERS",
    "LFM2_230M_FACTORIZED_TIED_VOCAB_PARAMETERS_PER_RANK",
    "LFM2_230M_HEAD_DIM",
    "LFM2_230M_HIDDEN_SIZE",
    "LFM2_230M_HYBRID_FACTORIZED_VOCAB_TARGET_FRONTIER",
    "LFM2_230M_HYBRID_TARGET_FRONTIER",
    "LFM2_230M_HALF_SIZE_STRUCTURAL_FRONTIER",
    "LFM2_230M_SOURCE_LAYER_TYPES",
    "LFM2_230M_SOURCE_UNIQUE_PARAMETERS",
    "LFM2_230M_VOCAB_SIZE",
    "Lfm2HybridFactorizedVocabularyTarget",
    "Lfm2HybridTarget",
    "Lfm2OrderedSourceSpan",
    "Lfm2ProgressiveStage",
    "Lfm2ProgressiveTargetPlan",
    "Lfm2Topology",
    "SCHEMA",
    "lfm2_230m_hybrid_factorized_vocab_parameter_count",
    "lfm2_230m_hybrid_factorized_vocab_target_frontier",
    "lfm2_230m_hybrid_parameter_count",
    "lfm2_230m_hybrid_target_frontier",
    "lfm2_230m_half_size_structural_frontier",
    "lfm2_230m_half_size_structural_support_report",
    "plan_lfm2_progressive_targets",
    "solve_lfm2_230m_hybrid_factorized_vocab_target",
    "solve_lfm2_230m_hybrid_structural_target",
]
