from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_attention_codebook import (  # noqa: E402
    GQAAttentionBundle,
    fit_global_attention_codebook,
)
from neural_forge.hf_codebook_alignment import (  # noqa: E402
    CodebookAlignmentFiniteError,
    CodebookAlignmentShapeError,
    infer_qwen_gqa_group_permutations,
    infer_swiglu_neuron_permutations,
    invert_permutations,
    verify_permutation_bijections,
)
from neural_forge.hf_neuron_codebook import (  # noqa: E402
    GatedMLPBundle,
    fit_global_neuron_codebook,
)


def _leaf(value):
    return value.detach().clone().requires_grad_()


def _swiglu_source(
    intermediate: int,
    input_width: int,
    output_width: int,
    *,
    seed: int,
    bias: bool = True,
) -> GatedMLPBundle:
    generator = torch.Generator().manual_seed(seed)
    return GatedMLPBundle(
        gate_weight=torch.randn(
            intermediate, input_width, generator=generator, dtype=torch.float64
        ),
        up_weight=torch.randn(
            intermediate, input_width, generator=generator, dtype=torch.float64
        ),
        down_weight=torch.randn(
            output_width, intermediate, generator=generator, dtype=torch.float64
        ),
        gate_bias=(
            torch.randn(intermediate, generator=generator, dtype=torch.float64)
            if bias
            else None
        ),
        up_bias=(
            torch.randn(intermediate, generator=generator, dtype=torch.float64)
            if bias
            else None
        ),
        down_bias=(
            torch.randn(output_width, generator=generator, dtype=torch.float64)
            if bias
            else None
        ),
    )


def _permute_swiglu(
    source: GatedMLPBundle,
    physical_order,
    *,
    requires_grad: bool = False,
) -> GatedMLPBundle:
    order = torch.as_tensor(physical_order, dtype=torch.int64)

    def copied(value, dimension):
        if value is None:
            return None
        result = value.index_select(dimension, order).detach().clone()
        return result.requires_grad_() if requires_grad else result

    down_bias = (
        source.down_bias.detach().clone() if source.down_bias is not None else None
    )
    if down_bias is not None and requires_grad:
        down_bias.requires_grad_()
    return GatedMLPBundle(
        gate_weight=copied(source.gate_weight, 0),
        up_weight=copied(source.up_weight, 0),
        down_weight=copied(source.down_weight, 1),
        gate_bias=copied(source.gate_bias, 0),
        up_bias=copied(source.up_bias, 0),
        down_bias=down_bias,
    )


def _bundle_tensors(bundle):
    return tuple(
        value
        for value in (
            bundle.gate_weight,
            bundle.up_weight,
            bundle.down_weight,
            bundle.gate_bias,
            bundle.up_bias,
            bundle.down_bias,
        )
        if value is not None
    )


def test_swiglu_recovers_known_coupled_permutations_without_gradients() -> None:
    layers = 4
    intermediate = 47
    generator = torch.Generator().manual_seed(410)
    source = _swiglu_source(intermediate, 9, 7, seed=411)
    physical_orders = [torch.arange(intermediate)] + [
        torch.randperm(intermediate, generator=generator) for _ in range(layers - 1)
    ]
    bundles = [
        _permute_swiglu(source, order, requires_grad=True) for order in physical_orders
    ]
    base_activations = torch.randn(
        23,
        intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    activations = [
        _leaf(base_activations.index_select(1, order)) for order in physical_orders
    ]
    expected = torch.stack([torch.argsort(order) for order in physical_orders])

    kwargs = {
        "activation_sketches": activations,
        "weight_sketch_size": 21,
        "activation_sketch_size": 13,
        "local_block_size": 8,
        "refinement_passes": 2,
    }
    first = infer_swiglu_neuron_permutations(bundles, **kwargs)
    second = infer_swiglu_neuron_permutations(bundles, **kwargs)
    codebook = fit_global_neuron_codebook(
        bundles,
        assignment_strategy="aligned_index",
        activation_sketches=activations,
        weight_sketch_size=9,
        activation_sketch_size=7,
        **first.codebook_kwargs(),
    )

    assert torch.equal(first.permutations, expected)
    assert torch.equal(first.inverse_permutations, torch.stack(physical_orders))
    assert torch.equal(first.permutations, second.permutations)
    assert torch.equal(first.inverse_permutations, second.inverse_permutations)
    assert first.diagnostics == second.diagnostics
    assert first.codebook_kwargs().keys() == {"neuron_permutations"}
    assert verify_permutation_bijections(first.permutations) is True
    assert first.permutation_proof is True
    assert first.diagnostics.permutations_bijective is True
    assert first.diagnostics.inverses_verified is True
    assert first.diagnostics.permutation_rows_verified == layers
    assert first.diagnostics.aligned_objective == 0.0
    assert first.diagnostics.identity_objective > 0.0
    assert first.diagnostics.objective_improvement > 0.0
    assert first.diagnostics.objective_ratio == 0.0
    assert first.diagnostics.fitting_requires_grad is False
    assert torch.equal(
        codebook.registry.assignments, torch.stack(physical_orders).to(torch.int32)
    )
    assert codebook.diagnostics is not None
    assert codebook.diagnostics.permutation_proof is True
    assert first.permutations.requires_grad is False
    assert first.inverse_permutations.requires_grad is False
    assert all(
        tensor.grad is None for bundle in bundles for tensor in _bundle_tensors(bundle)
    )
    assert all(value.grad is None for value in activations)


def _gqa_source(
    *,
    query_heads: int,
    groups: int,
    head_dim: int,
    input_width: int,
    output_width: int,
    seed: int,
) -> GQAAttentionBundle:
    generator = torch.Generator().manual_seed(seed)
    query_width = query_heads * head_dim
    key_value_width = groups * head_dim
    return GQAAttentionBundle(
        q_weight=torch.randn(
            query_width, input_width, generator=generator, dtype=torch.float64
        ),
        k_weight=torch.randn(
            key_value_width, input_width, generator=generator, dtype=torch.float64
        ),
        v_weight=torch.randn(
            key_value_width, input_width, generator=generator, dtype=torch.float64
        ),
        o_weight=torch.randn(
            output_width, query_width, generator=generator, dtype=torch.float64
        ),
        q_bias=torch.randn(query_width, generator=generator, dtype=torch.float64),
        k_bias=torch.randn(key_value_width, generator=generator, dtype=torch.float64),
        v_bias=torch.randn(key_value_width, generator=generator, dtype=torch.float64),
        o_bias=torch.randn(output_width, generator=generator, dtype=torch.float64),
        query_heads=query_heads,
        key_value_heads=groups,
        head_dim=head_dim,
    )


def _permute_gqa(
    source: GQAAttentionBundle,
    physical_order,
) -> GQAAttentionBundle:
    order = torch.as_tensor(physical_order, dtype=torch.int64)
    groups = int(order.numel())
    head_dim = int(source.head_dim)
    query_heads = int(source.query_heads)
    query_rows = (query_heads // groups) * head_dim
    input_width = int(source.q_weight.shape[1])
    output_width = int(source.o_weight.shape[0])

    def producer(weight, rows):
        return (
            weight.reshape(groups, rows, input_width)
            .index_select(0, order)
            .reshape_as(weight)
            .detach()
            .clone()
        )

    def bias(value, rows):
        return (
            value.reshape(groups, rows)
            .index_select(0, order)
            .reshape_as(value)
            .detach()
            .clone()
        )

    return GQAAttentionBundle(
        q_weight=producer(source.q_weight, query_rows),
        k_weight=producer(source.k_weight, head_dim),
        v_weight=producer(source.v_weight, head_dim),
        o_weight=(
            source.o_weight.reshape(output_width, groups, query_rows)
            .index_select(1, order)
            .reshape_as(source.o_weight)
            .detach()
            .clone()
        ),
        q_bias=bias(source.q_bias, query_rows),
        k_bias=bias(source.k_bias, head_dim),
        v_bias=bias(source.v_bias, head_dim),
        o_bias=source.o_bias.detach().clone(),
        query_heads=query_heads,
        key_value_heads=groups,
        head_dim=head_dim,
    )


def test_qwen_gqa_recovers_known_coupled_group_permutations_exactly() -> None:
    layers = 5
    groups = 4
    generator = torch.Generator().manual_seed(420)
    source = _gqa_source(
        query_heads=12,
        groups=groups,
        head_dim=3,
        input_width=7,
        output_width=8,
        seed=421,
    )
    physical_orders = [torch.arange(groups)] + [
        torch.randperm(groups, generator=generator) for _ in range(layers - 1)
    ]
    bundles = [_permute_gqa(source, order) for order in physical_orders]
    base_activations = torch.randn(17, groups, generator=generator, dtype=torch.float64)
    activations = [
        base_activations.index_select(1, order).clone() for order in physical_orders
    ]
    expected = torch.stack([torch.argsort(order) for order in physical_orders])

    result = infer_qwen_gqa_group_permutations(
        bundles,
        activation_sketches=activations,
        weight_sketch_size=29,
        activation_sketch_size=11,
    )
    activation_only = infer_qwen_gqa_group_permutations(
        bundles,
        activation_sketches=activations,
        weight_sketch_size=0,
        activation_sketch_size=11,
    )
    codebook = fit_global_attention_codebook(
        bundles,
        prototypes_per_group=1,
        sketch_dim=13,
        **result.codebook_kwargs(),
    )

    assert torch.equal(result.group_permutations, expected)
    assert torch.equal(activation_only.group_permutations, expected)
    assert activation_only.diagnostics.used_weight_sketches is False
    assert activation_only.diagnostics.used_activation_sketches is True
    assert torch.equal(codebook.registry.group_permutations, expected.to(torch.int32))
    assert codebook.diagnostics is not None
    assert codebook.diagnostics.permutations_bijective is True
    assert torch.equal(result.inverse_permutations, torch.stack(physical_orders))
    assert result.codebook_kwargs().keys() == {"group_permutations"}
    diagnostics = result.diagnostics
    assert diagnostics.topology == "qwen_gqa_groups"
    assert diagnostics.exact_small_group_assignment is True
    assert diagnostics.assignment_solver == "exact_hungarian_lowest_index_ties"
    assert diagnostics.pairwise_distance_values == (layers - 1) * groups**2
    assert diagnostics.distance_work_bound == (layers - 1) * groups**2
    assert diagnostics.maximum_pairwise_block_elements == groups**2
    assert diagnostics.pairwise_block_element_limit == groups**2
    assert diagnostics.materialized_full_pairwise_matrix is True
    assert diagnostics.bounded_work is True
    assert diagnostics.aligned_objective == 0.0
    assert diagnostics.identity_objective > 0.0
    assert diagnostics.permutation_proof is True
    assert diagnostics.zero_gradient is True
    assert diagnostics.inferred_objective == diagnostics.aligned_objective
    assert diagnostics.objective_delta == diagnostics.objective_improvement


def test_swiglu_qwen3_4b_width_never_materializes_dense_pairwise_matrix() -> None:
    intermediate = 9728
    generator = torch.Generator().manual_seed(430)
    source = _swiglu_source(intermediate, 3, 2, seed=431, bias=False)
    physical_order = torch.randperm(intermediate, generator=generator)
    bundles = [source, _permute_swiglu(source, physical_order)]

    result = infer_swiglu_neuron_permutations(
        bundles,
        weight_sketch_size=9,
        activation_sketch_size=0,
        local_block_size=16,
        refinement_passes=1,
    )

    assert torch.equal(result.permutations[1], torch.argsort(physical_order))
    diagnostics = result.diagnostics
    assert diagnostics.units_per_layer == intermediate
    assert diagnostics.full_pairwise_matrix_elements == intermediate**2
    assert diagnostics.materialized_full_pairwise_matrix is False
    assert diagnostics.no_full_pairwise_matrix is True
    assert diagnostics.maximum_pairwise_block_elements <= 16**2
    assert diagnostics.pairwise_block_element_limit == 16**2
    assert diagnostics.pairwise_distance_values <= diagnostics.distance_work_bound
    assert diagnostics.distance_work_bound == 2 * intermediate * 16
    assert diagnostics.weight_sketch_dimensions == 8
    assert diagnostics.sketch_elements == (
        2 * intermediate * diagnostics.combined_sketch_dimensions
    )
    assert diagnostics.bounded_work is True


def test_reference_layer_and_inverse_round_trips_use_canonical_to_physical_rows() -> (
    None
):
    source = _swiglu_source(7, 4, 5, seed=440, bias=False)
    orders = [
        torch.tensor([2, 0, 6, 4, 1, 5, 3]),
        torch.tensor([4, 3, 1, 6, 0, 2, 5]),
        torch.tensor([1, 5, 3, 0, 6, 4, 2]),
    ]
    bundles = [_permute_swiglu(source, order) for order in orders]

    result = infer_swiglu_neuron_permutations(
        bundles,
        reference_layer=1,
        weight_sketch_size=12,
        activation_sketch_size=0,
        local_block_size=4,
    )

    reference_physical_to_canonical = orders[1]
    expected = torch.stack(
        [
            torch.argsort(order).index_select(0, reference_physical_to_canonical)
            for order in orders
        ]
    )
    identity = torch.arange(7, dtype=torch.int64)
    assert torch.equal(result.permutations, expected)
    assert torch.equal(result.permutations[1], identity)
    assert torch.equal(
        result.inverse_permutations.gather(1, result.permutations),
        identity.repeat(3, 1),
    )
    assert torch.equal(
        result.permutations.gather(1, result.inverse_permutations),
        identity.repeat(3, 1),
    )


@pytest.mark.parametrize(
    "permutations",
    [
        torch.tensor([[0, 1, 1]], dtype=torch.int64),
        torch.tensor([[0, 1, 3]], dtype=torch.int64),
        torch.tensor([[0.0, 1.0, 2.0]]),
        [[0, 1, False]],
    ],
    ids=("duplicate", "out-of-range", "floating", "boolean"),
)
def test_inverse_validation_fails_closed_for_non_bijections(permutations) -> None:
    with pytest.raises(CodebookAlignmentShapeError):
        invert_permutations(permutations)


def test_shape_and_finite_validation_fail_before_assignment() -> None:
    source = _swiglu_source(6, 4, 5, seed=450)
    malformed = GatedMLPBundle(
        gate_weight=source.gate_weight,
        up_weight=source.up_weight,
        down_weight=source.down_weight[:, :-1],
    )
    with pytest.raises(CodebookAlignmentShapeError, match="down columns"):
        infer_swiglu_neuron_permutations([source, malformed])

    bad_activations = torch.randn(2, 8, 6, dtype=torch.float64)
    bad_activations[1, 3, 2] = torch.nan
    with pytest.raises(CodebookAlignmentFiniteError, match="NaN or infinity"):
        infer_swiglu_neuron_permutations(
            [source, source],
            activation_sketches=bad_activations,
        )
