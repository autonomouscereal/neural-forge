from __future__ import annotations

import gc

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_neuron_codebook import (  # noqa: E402
    GatedMLPBundle,
    GlobalNeuronCodebook,
    NeuronCodebookLayerView,
    NeuronCodebookRegistry,
    NeuronCodebookShapeError,
    fit_global_neuron_codebook,
)


def _bundles(
    *,
    layers: int = 3,
    input_width: int = 5,
    intermediate_width: int = 4,
    output_width: int = 5,
    bias: bool = True,
    seed: int = 100,
):
    generator = torch.Generator().manual_seed(seed)
    bundles = []
    linears = []
    for _ in range(layers):
        gate = torch.nn.Linear(
            input_width,
            intermediate_width,
            bias=bias,
            dtype=torch.float64,
        )
        up = torch.nn.Linear(
            input_width,
            intermediate_width,
            bias=bias,
            dtype=torch.float64,
        )
        down = torch.nn.Linear(
            intermediate_width,
            output_width,
            bias=bias,
            dtype=torch.float64,
        )
        with torch.no_grad():
            for parameter in (*gate.parameters(), *up.parameters(), *down.parameters()):
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                )
        bundles.append(GatedMLPBundle.from_linears(gate, up, down))
        linears.append((gate, up, down))
    return bundles, linears


def _activation_sketches(
    layers: int,
    rows: int,
    intermediate_width: int,
    *,
    seed: int,
    requires_grad: bool = False,
):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(
            rows,
            intermediate_width,
            generator=generator,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
        for _ in range(layers)
    ]


def _permuted_bundle(bundle: GatedMLPBundle, physical_order):
    row_indices = physical_order.to(bundle.gate_weight.device)
    return GatedMLPBundle(
        gate_weight=bundle.gate_weight.detach().index_select(0, row_indices).clone(),
        up_weight=bundle.up_weight.detach().index_select(0, row_indices).clone(),
        down_weight=bundle.down_weight.detach().index_select(1, row_indices).clone(),
        gate_bias=(
            bundle.gate_bias.detach().index_select(0, row_indices).clone()
            if bundle.gate_bias is not None
            else None
        ),
        up_bias=(
            bundle.up_bias.detach().index_select(0, row_indices).clone()
            if bundle.up_bias is not None
            else None
        ),
        down_bias=(
            bundle.down_bias.detach().clone() if bundle.down_bias is not None else None
        ),
    )


@pytest.mark.parametrize("bias", [False, True])
def test_full_prototype_codebook_is_exact_and_keeps_every_logical_neuron(bias) -> None:
    bundles, _ = _bundles(layers=3, bias=bias, seed=110)
    logical_neurons = 3 * 4
    codebook = fit_global_neuron_codebook(
        bundles,
        prototype_count=logical_neurons,
        weight_sketch_size=7,
        point_chunk_size=3,
        center_chunk_size=2,
    )
    inputs = torch.randn(6, 5, dtype=torch.float64)

    for layer_index, source in enumerate(bundles):
        torch.testing.assert_close(
            codebook[layer_index](inputs),
            source.forward(inputs),
            atol=0.0,
            rtol=0.0,
        )

    expected_indices = torch.arange(logical_neurons, dtype=torch.int32).reshape(3, 4)
    assert torch.equal(codebook.registry.assignments, expected_indices)
    assert torch.equal(
        codebook.registry.prototype_source_indices,
        torch.arange(logical_neurons, dtype=torch.int32),
    )
    assert codebook.report.logical_neuron_executions == logical_neurons
    assert codebook.report.unique_learned_parameters == (
        codebook.report.logical_dense_parameters
    )
    assert codebook.report.child_fraction_of_dense == pytest.approx(
        codebook.report.unique_learned_parameters
        / codebook.report.logical_dense_parameters
    )
    assert codebook.report.source_to_child_ratio == pytest.approx(
        codebook.report.logical_dense_parameters
        / codebook.report.unique_learned_parameters
    )
    assert codebook.report.index_buffer_bytes > 0
    assert codebook.diagnostics is not None
    assert codebook.diagnostics.distance_values_computed == 0
    assert all(isinstance(view, NeuronCodebookLayerView) for view in codebook.layers)
    assert all(list(view.parameters()) == [] for view in codebook.layers)
    assert all(view.registry is codebook.registry for view in codebook.layers)


def test_compressed_codebook_runs_all_layers_and_has_normal_inference_gradients() -> (
    None
):
    bundles, linears = _bundles(layers=4, seed=120)
    activations = _activation_sketches(
        4,
        9,
        4,
        seed=121,
        requires_grad=True,
    )
    codebook = fit_global_neuron_codebook(
        bundles,
        prototype_count=3,
        activation_sketches=activations,
        use_scales=True,
        weight_sketch_size=9,
        activation_sketch_size=5,
        point_chunk_size=4,
        center_chunk_size=2,
        medoid_candidate_limit=3,
    )

    assert all(
        parameter.grad is None
        for modules in linears
        for module in modules
        for parameter in module.parameters()
    )
    assert all(value.grad is None for value in activations)
    assert codebook.report.logical_neuron_executions == 16
    assert codebook.report.prototype_count == 3
    assert codebook.report.scale_parameters == 4 * 4 * 3
    assert codebook.report.unique_learned_parameters < (
        codebook.report.logical_dense_parameters
    )
    assert codebook.report.index_buffer_bytes == (
        codebook.report.assignment_index_bytes
        + codebook.report.prototype_source_index_bytes
    )

    inputs = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    outputs = [view(inputs) for view in codebook.layers]
    assert len(outputs) == 4
    assert all(output.shape == (2, 3, 5) for output in outputs)
    assert all(bool(torch.isfinite(output).all()) for output in outputs)
    sum(output.square().mean() for output in outputs).backward()

    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())
    assert all(parameter.grad is not None for parameter in codebook.parameters())
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in codebook.parameters()
    )
    assert all(
        parameter.grad is None
        for modules in linears
        for module in modules
        for parameter in module.parameters()
    )
    assert all(value.grad is None for value in activations)


def test_real_medoid_triplets_and_runtime_assignments_stay_coupled() -> None:
    bundles, _ = _bundles(layers=3, seed=130)
    codebook = fit_global_neuron_codebook(
        bundles,
        prototype_count=4,
        weight_sketch_size=11,
        point_chunk_size=3,
        center_chunk_size=2,
        medoid_candidate_limit=3,
    )
    registry = codebook.registry
    intermediate = registry.intermediate_width

    for prototype_index, global_index in enumerate(
        registry.prototype_source_indices.tolist()
    ):
        layer_index, neuron_index = divmod(global_index, intermediate)
        source = bundles[layer_index]
        torch.testing.assert_close(
            registry.prototype_gate_weight[prototype_index],
            source.gate_weight[neuron_index],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            registry.prototype_up_weight[prototype_index],
            source.up_weight[neuron_index],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            registry.prototype_down_weight[:, prototype_index],
            source.down_weight[:, neuron_index],
            atol=0.0,
            rtol=0.0,
        )

    for layer_index in range(registry.logical_layers):
        materialized = registry.materialize_layer(layer_index)
        for neuron_index, prototype_index in enumerate(
            registry.assignments[layer_index].tolist()
        ):
            torch.testing.assert_close(
                materialized.gate_weight[neuron_index],
                registry.prototype_gate_weight[prototype_index],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                materialized.up_weight[neuron_index],
                registry.prototype_up_weight[prototype_index],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                materialized.down_weight[:, neuron_index],
                registry.prototype_down_weight[:, prototype_index],
                atol=0.0,
                rtol=0.0,
            )


def test_fitting_is_deterministic_and_reports_bounded_sketch_distance_work() -> None:
    bundles, _ = _bundles(
        layers=4,
        intermediate_width=7,
        output_width=6,
        seed=140,
    )
    activations = _activation_sketches(4, 13, 7, seed=141)
    kwargs = {
        "prototype_count": 5,
        "activation_sketches": activations,
        "weight_sketch_size": 8,
        "activation_sketch_size": 6,
        "point_chunk_size": 3,
        "center_chunk_size": 2,
        "medoid_candidate_limit": 3,
        "max_iterations": 2,
    }

    first = fit_global_neuron_codebook(bundles, **kwargs)
    second = fit_global_neuron_codebook(bundles, **kwargs)

    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
    assert first.diagnostics == second.diagnostics
    diagnostics = first.diagnostics
    assert diagnostics is not None
    assert diagnostics.assignment_strategy == "global_medoids"
    assert diagnostics.deterministic is True
    assert diagnostics.real_medoid_prototypes is True
    assert diagnostics.materialized_full_pairwise_matrix is False
    assert diagnostics.no_full_pairwise_matrix is True
    assert diagnostics.bounded_work is True
    assert diagnostics.fitting_requires_grad is False
    assert diagnostics.max_distance_block_elements <= 3 * 2
    assert diagnostics.distance_block_element_limit == 3 * 2
    assert diagnostics.full_pairwise_matrix_elements == 28 * 28
    assert diagnostics.weight_sketch_dimensions <= 8
    assert diagnostics.activation_sketch_dimensions <= 6
    assert diagnostics.sketch_elements == (
        diagnostics.logical_neurons * diagnostics.combined_sketch_dimensions
    )


def test_aligned_index_deterministically_selects_one_real_medoid_per_index() -> None:
    source_bundles, _ = _bundles(layers=1, intermediate_width=5, seed=145)
    bundles = source_bundles * 4
    kwargs = {
        "assignment_strategy": "aligned_index",
        "prototypes_per_neuron": 1,
        "weight_sketch_size": 9,
        "point_chunk_size": 2,
        "center_chunk_size": 3,
    }

    first = fit_global_neuron_codebook(bundles, **kwargs)
    second = fit_global_neuron_codebook(bundles, **kwargs)

    expected_assignments = torch.arange(5, dtype=torch.int32).repeat(4, 1)
    expected_medoids = torch.arange(5, dtype=torch.int32)
    assert torch.equal(first.registry.assignments, expected_assignments)
    assert torch.equal(first.registry.prototype_source_indices, expected_medoids)
    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
    assert first.diagnostics == second.diagnostics

    source = bundles[0]
    torch.testing.assert_close(
        first.registry.prototype_gate_weight,
        source.gate_weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first.registry.prototype_up_weight,
        source.up_weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first.registry.prototype_down_weight,
        source.down_weight,
        atol=0.0,
        rtol=0.0,
    )

    diagnostics = first.diagnostics
    assert diagnostics is not None
    assert diagnostics.assignment_strategy == "aligned_index"
    assert diagnostics.prototype_count == 5
    assert diagnostics.prototypes_per_neuron == 1
    assert first.report.prototypes_per_neuron == 1
    assert diagnostics.tie_break == "lowest global neuron index"
    assert diagnostics.permutations_provided is False
    assert diagnostics.permutation_proof is True
    assert diagnostics.permutations_bijective is True
    assert diagnostics.permutation_rows_verified == 4
    assert diagnostics.aligned_group_size == 4
    assert diagnostics.aligned_subgroup_factor == 1
    assert diagnostics.used_n_by_k_assignment_search is False
    assert diagnostics.distance_values_computed == 5 * 4**2
    assert diagnostics.exact_distance_work_bound == 5 * 4**2


def test_aligned_index_intermediate_depth_slots_are_real_and_deterministic() -> None:
    layers = 5
    intermediate = 4
    prototypes_per_neuron = 3
    input_width = 5
    output_width = 5
    bundles, _ = _bundles(
        layers=layers,
        input_width=input_width,
        intermediate_width=intermediate,
        output_width=output_width,
        seed=1451,
    )
    kwargs = {
        "assignment_strategy": "aligned_index",
        "prototypes_per_neuron": prototypes_per_neuron,
        "weight_sketch_size": 9,
        "point_chunk_size": 2,
        "center_chunk_size": 3,
    }

    first = fit_global_neuron_codebook(bundles, **kwargs)
    second = fit_global_neuron_codebook(bundles, **kwargs)
    registry = first.registry
    prototype_count = intermediate * prototypes_per_neuron

    assert registry.prototype_count == prototype_count
    assert registry.prototypes_per_neuron == prototypes_per_neuron
    assert first.report.prototypes_per_neuron == prototypes_per_neuron
    assert first.diagnostics is not None
    assert first.diagnostics.prototypes_per_neuron == prototypes_per_neuron
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
    assert first.diagnostics == second.diagnostics

    flat_assignments = registry.assignments.flatten()
    assert tuple(registry.assignments.shape) == (layers, intermediate)
    assert int(registry.assignments.min()) == 0
    assert int(registry.assignments.max()) < prototype_count
    assert int(torch.unique(registry.prototype_source_indices).numel()) == (
        prototype_count
    )
    for canonical_neuron in range(intermediate):
        start = canonical_neuron * prototypes_per_neuron
        end = start + prototypes_per_neuron
        assignments = registry.assignments[:, canonical_neuron]
        assert bool(((assignments >= start) & (assignments < end)).all())
        for prototype_index in range(start, end):
            source_index = int(registry.prototype_source_indices[prototype_index])
            source_layer, source_neuron = divmod(source_index, intermediate)
            assert source_neuron == canonical_neuron
            assert int(flat_assignments[source_index]) == prototype_index
            torch.testing.assert_close(
                registry.prototype_gate_weight[prototype_index],
                bundles[source_layer].gate_weight[source_neuron],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                registry.prototype_up_weight[prototype_index],
                bundles[source_layer].up_weight[source_neuron],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                registry.prototype_down_weight[:, prototype_index],
                bundles[source_layer].down_weight[:, source_neuron],
                rtol=0.0,
                atol=0.0,
            )

    expected_prototype_parameters = prototype_count * (
        2 * input_width + output_width + 2
    )
    expected_retained_biases = layers * output_width
    expected_logical_parameters = layers * (
        intermediate * (2 * input_width + output_width + 2) + output_width
    )
    assert first.report.prototype_parameters == expected_prototype_parameters
    assert first.report.retained_layer_bias_parameters == expected_retained_biases
    assert first.report.unique_learned_parameters == (
        expected_prototype_parameters + expected_retained_biases
    )
    assert first.report.logical_dense_parameters == expected_logical_parameters
    assert first.report.assignment_index_bytes == layers * intermediate * 4
    assert first.report.prototype_source_index_bytes == prototype_count * 4
    assert first.report.index_buffer_bytes == (
        first.report.assignment_index_bytes
        + first.report.prototype_source_index_bytes
    )


@pytest.mark.parametrize("prototypes_per_neuron", [1, 2])
def test_aligned_index_permutations_are_equivalent_to_physical_reordering(
    prototypes_per_neuron,
) -> None:
    layers = 4
    intermediate = 5
    bundles, _ = _bundles(
        layers=layers,
        intermediate_width=intermediate,
        seed=146,
    )
    activations = _activation_sketches(layers, 8, intermediate, seed=147)
    physical_orders = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [2, 0, 4, 1, 3],
            [4, 3, 2, 1, 0],
            [1, 3, 0, 4, 2],
        ],
        dtype=torch.int64,
    )
    aligned_to_physical = torch.argsort(physical_orders, dim=1)
    permuted_bundles = [
        _permuted_bundle(bundle, physical_orders[layer_index])
        for layer_index, bundle in enumerate(bundles)
    ]
    permuted_activations = [
        values.index_select(-1, physical_orders[layer_index])
        for layer_index, values in enumerate(activations)
    ]
    kwargs = {
        "assignment_strategy": "aligned_index",
        "prototypes_per_neuron": prototypes_per_neuron,
        "weight_sketch_size": 10,
        "activation_sketch_size": 6,
        "point_chunk_size": 2,
        "center_chunk_size": 3,
    }

    canonical = fit_global_neuron_codebook(
        bundles,
        activation_sketches=activations,
        **kwargs,
    )
    permuted = fit_global_neuron_codebook(
        permuted_bundles,
        activation_sketches=permuted_activations,
        neuron_permutations=aligned_to_physical,
        **kwargs,
    )

    expected_assignments = torch.stack(
        [
            canonical.registry.assignments[layer_index].index_select(0, order)
            for layer_index, order in enumerate(physical_orders)
        ]
    )
    assert torch.equal(permuted.registry.assignments, expected_assignments)
    for name in (
        "prototype_gate_weight",
        "prototype_up_weight",
        "prototype_down_weight",
        "prototype_gate_bias",
        "prototype_up_bias",
        "layer_down_bias",
    ):
        torch.testing.assert_close(
            getattr(permuted.registry, name),
            getattr(canonical.registry, name),
            atol=0.0,
            rtol=0.0,
        )

    inputs = torch.randn(3, 5, dtype=torch.float64)
    for layer_index, order in enumerate(physical_orders):
        canonical_layer = canonical.registry.materialize_layer(layer_index)
        permuted_layer = permuted.registry.materialize_layer(layer_index)
        torch.testing.assert_close(
            permuted_layer.gate_weight,
            canonical_layer.gate_weight.index_select(0, order),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            permuted_layer.up_weight,
            canonical_layer.up_weight.index_select(0, order),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            permuted_layer.down_weight,
            canonical_layer.down_weight.index_select(1, order),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            permuted[layer_index](inputs),
            canonical[layer_index](inputs),
        )

    diagnostics = permuted.diagnostics
    assert diagnostics is not None
    assert diagnostics.permutations_provided is True
    assert diagnostics.permutation_proof is True
    assert diagnostics.permutation_proof_method == "sorted_rows_equal_arange"


def test_aligned_full_depth_reconstructs_every_permuted_source_and_reloads() -> None:
    layers = 4
    intermediate = 5
    bundles, _ = _bundles(
        layers=layers,
        intermediate_width=intermediate,
        seed=1471,
    )
    physical_orders = torch.tensor(
        [
            [3, 0, 4, 1, 2],
            [1, 4, 2, 0, 3],
            [4, 3, 0, 2, 1],
            [2, 1, 3, 4, 0],
        ],
        dtype=torch.int64,
    )
    canonical_to_physical = torch.argsort(physical_orders, dim=1)
    permuted_bundles = [
        _permuted_bundle(bundle, physical_orders[layer_index])
        for layer_index, bundle in enumerate(bundles)
    ]
    codebook = fit_global_neuron_codebook(
        permuted_bundles,
        assignment_strategy="aligned_index",
        prototypes_per_neuron=layers,
        neuron_permutations=canonical_to_physical,
        use_scales=False,
        weight_sketch_size=8,
    )
    registry = codebook.registry

    expected_assignments = torch.stack(
        [
            (
                torch.arange(intermediate, dtype=torch.int32) * layers
                + layer_index
            ).index_select(0, order)
            for layer_index, order in enumerate(physical_orders)
        ]
    )
    expected_sources = torch.tensor(
        [
            layer_index * intermediate
            + int(canonical_to_physical[layer_index, canonical_neuron])
            for canonical_neuron in range(intermediate)
            for layer_index in range(layers)
        ],
        dtype=torch.int32,
    )
    assert torch.equal(registry.assignments, expected_assignments)
    assert torch.equal(registry.prototype_source_indices, expected_sources)
    assert registry.neuron_scales is None
    assert registry.prototypes_per_neuron == layers
    assert codebook.report.prototypes_per_neuron == layers
    assert codebook.report.prototype_count == layers * intermediate
    assert codebook.report.unique_learned_parameters == (
        codebook.report.logical_dense_parameters
    )
    assert codebook.report.assignment_index_bytes == layers * intermediate * 4
    assert codebook.report.prototype_source_index_bytes == layers * intermediate * 4
    assert codebook.report.index_buffer_bytes == layers * intermediate * 8
    assert codebook.diagnostics is not None
    assert codebook.diagnostics.distance_values_computed == 0
    assert codebook.diagnostics.squared_sketch_error == 0.0

    inputs = torch.randn(3, 5, dtype=torch.float64)
    for layer_index, source in enumerate(permuted_bundles):
        materialized = registry.materialize_layer(layer_index)
        for name in (
            "gate_weight",
            "up_weight",
            "down_weight",
            "gate_bias",
            "up_bias",
            "down_bias",
        ):
            torch.testing.assert_close(
                getattr(materialized, name),
                getattr(source, name),
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            codebook[layer_index](inputs),
            source.forward(inputs),
            rtol=0.0,
            atol=0.0,
        )

    state = {
        name: value.detach().clone() for name, value in codebook.state_dict().items()
    }
    restored = GlobalNeuronCodebook.from_state_dict(state)
    assert restored.report == codebook.report
    assert restored.registry.prototypes_per_neuron == layers
    assert restored.state_dict().keys() == state.keys()
    for name, value in state.items():
        assert torch.equal(restored.state_dict()[name], value)
    for layer_index, source in enumerate(permuted_bundles):
        restored_layer = restored.registry.materialize_layer(layer_index)
        torch.testing.assert_close(
            restored_layer.gate_weight,
            source.gate_weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            restored_layer.up_weight,
            source.up_weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            restored_layer.down_weight,
            source.down_weight,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "permutations",
    [
        torch.arange(8, dtype=torch.int64).reshape(2, 4),
        torch.tensor(
            [[0, 1, 2, 3], [0, 1, 1, 3], [0, 1, 2, 3]],
            dtype=torch.int64,
        ),
        torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 3]],
            dtype=torch.int64,
        ),
        torch.arange(12, dtype=torch.float64).reshape(3, 4),
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, False]],
    ],
    ids=("shape", "duplicate", "out-of-range", "dtype", "boolean"),
)
def test_aligned_index_rejects_malformed_permutations(permutations) -> None:
    bundles, _ = _bundles(layers=3, intermediate_width=4, seed=148)

    with pytest.raises(NeuronCodebookShapeError, match="neuron_permutations"):
        fit_global_neuron_codebook(
            bundles,
            assignment_strategy="aligned_index",
            neuron_permutations=permutations,
        )


def test_aligned_index_rejects_non_aligned_prototype_count() -> None:
    bundles, _ = _bundles(layers=3, intermediate_width=4, seed=149)

    with pytest.raises(NeuronCodebookShapeError, match="divisible by intermediate"):
        fit_global_neuron_codebook(
            bundles,
            prototype_count=3,
            assignment_strategy="aligned_index",
        )


@pytest.mark.parametrize("prototypes_per_neuron", [False, 0, 4])
def test_aligned_index_rejects_invalid_prototypes_per_neuron(
    prototypes_per_neuron,
) -> None:
    bundles, _ = _bundles(layers=3, intermediate_width=4, seed=1491)

    with pytest.raises(NeuronCodebookShapeError, match="prototypes_per_neuron"):
        fit_global_neuron_codebook(
            bundles,
            prototypes_per_neuron=prototypes_per_neuron,
            assignment_strategy="aligned_index",
        )


def test_aligned_index_distance_work_scales_with_width_and_squared_depth() -> None:
    intermediate = 9
    observed_work = {}
    for layers in (2, 4, 6):
        bundles, _ = _bundles(
            layers=layers,
            intermediate_width=intermediate,
            seed=150 + layers,
        )
        codebook = fit_global_neuron_codebook(
            bundles,
            assignment_strategy="aligned_index",
            weight_sketch_size=7,
            point_chunk_size=2,
            center_chunk_size=3,
        )
        diagnostics = codebook.diagnostics
        assert diagnostics is not None
        expected_work = intermediate * layers**2
        observed_work[layers] = diagnostics.distance_values_computed

        assert codebook.report.prototype_count == intermediate
        assert codebook.report.prototypes_per_neuron == 1
        assert diagnostics.distance_values_computed == expected_work
        assert diagnostics.exact_distance_work_bound == expected_work
        assert diagnostics.distance_work_bound == expected_work
        assert diagnostics.exact_distance_coordinate_work_bound == (
            expected_work * diagnostics.combined_sketch_dimensions
        )
        assert diagnostics.distance_work_bound_formula == (
            "intermediate_width * logical_layers**2"
        )
        assert diagnostics.max_distance_block_elements <= 2 * 3
        assert diagnostics.bounded_work is True

    assert observed_work[4] == 4 * observed_work[2]
    assert observed_work[6] == 9 * observed_work[2]


def test_tensor_state_roundtrip_is_exact_and_views_keep_registry_alive() -> None:
    bundles, _ = _bundles(layers=3, seed=150)
    supplied_scales = torch.linspace(
        0.75,
        1.25,
        steps=3 * 4,
        dtype=torch.float64,
    ).reshape(3, 4)
    codebook = fit_global_neuron_codebook(
        bundles,
        prototype_count=3,
        per_neuron_scales=supplied_scales,
        weight_sketch_size=10,
        point_chunk_size=3,
        center_chunk_size=2,
    )
    state = {
        key: value.detach().clone() for key, value in codebook.state_dict().items()
    }

    assert state
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert not any("extra_state" in key for key in state)
    restored = GlobalNeuronCodebook.from_state_dict(state)
    direct_registry = NeuronCodebookRegistry.from_state_dict(
        restored.registry.state_dict()
    )
    inputs = torch.randn(4, 5, dtype=torch.float64)

    assert restored.report == codebook.report
    assert restored.state_dict().keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(restored.state_dict()[key], value)
    for layer_index in range(3):
        torch.testing.assert_close(
            restored[layer_index](inputs),
            codebook[layer_index](inputs),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            direct_registry(inputs, layer_index),
            codebook[layer_index](inputs),
            atol=0.0,
            rtol=0.0,
        )
        assert not restored[layer_index].state_dict()

    view = restored[1]
    registry = restored.registry
    del direct_registry
    del restored
    gc.collect()
    assert view.registry is registry
    assert bool(torch.isfinite(view(inputs)).all())
