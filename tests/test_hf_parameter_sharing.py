from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_parameter_sharing import (  # noqa: E402
    AssignmentCollisionError,
    FiniteValidationError,
    IncompleteAssignmentError,
    MLPModuleLayout,
    MLPTensorBundle,
    ModuleSharingError,
    ShapeValidationError,
    TopologyValidationError,
    align_and_merge_mlp_bundles,
    derive_neuron_permutation,
    expand_mlp_modules,
    merge_aligned_mlp_bundles,
    parameter_sharing_report,
    permute_mlp_bundle,
    solve_neuron_assignment,
    standardized_cross_correlation,
    tie_mlp_modules,
)


def _randn(*shape: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


def _dense_bundle(seed: int = 1) -> MLPTensorBundle:
    return MLPTensorBundle(
        input_weight=_randn(4, 3, seed=seed),
        down_weight=_randn(2, 4, seed=seed + 1),
        input_bias=_randn(4, seed=seed + 2),
        down_bias=_randn(2, seed=seed + 3),
    )


def _gated_bundle(seed: int = 10) -> MLPTensorBundle:
    return MLPTensorBundle(
        gate_weight=_randn(4, 3, seed=seed),
        up_weight=_randn(4, 3, seed=seed + 1),
        down_weight=_randn(2, 4, seed=seed + 2),
        gate_bias=_randn(4, seed=seed + 3),
        up_bias=_randn(4, seed=seed + 4),
        down_bias=_randn(2, seed=seed + 5),
    )


def _bundle_tensors(bundle: MLPTensorBundle):
    return {
        name: value
        for name in (
            "input_weight",
            "gate_weight",
            "up_weight",
            "down_weight",
            "input_bias",
            "gate_bias",
            "up_bias",
            "down_bias",
        )
        if (value := getattr(bundle, name)) is not None
    }


def _assert_bundles_close(
    actual: MLPTensorBundle,
    expected: MLPTensorBundle,
) -> None:
    assert actual.topology == expected.topology
    assert _bundle_tensors(actual).keys() == _bundle_tensors(expected).keys()
    for role, actual_tensor in _bundle_tensors(actual).items():
        torch.testing.assert_close(
            actual_tensor,
            _bundle_tensors(expected)[role],
            atol=1e-12,
            rtol=1e-12,
        )


def _source_for_anchor_permutation(anchor, permutation):
    inverse = torch.argsort(permutation)
    return anchor.index_select(1, inverse)


def _dense_forward(bundle: MLPTensorBundle, inputs):
    hidden = torch.nn.functional.gelu(
        torch.nn.functional.linear(
            inputs,
            bundle.input_weight,
            bundle.input_bias,
        )
    )
    return torch.nn.functional.linear(
        hidden,
        bundle.down_weight,
        bundle.down_bias,
    )


def _gated_forward(bundle: MLPTensorBundle, inputs):
    gate = torch.nn.functional.silu(
        torch.nn.functional.linear(
            inputs,
            bundle.gate_weight,
            bundle.gate_bias,
        )
    )
    up = torch.nn.functional.linear(
        inputs,
        bundle.up_weight,
        bundle.up_bias,
    )
    return torch.nn.functional.linear(
        gate * up,
        bundle.down_weight,
        bundle.down_bias,
    )


def test_known_synthetic_permutation_recovery_and_correlation() -> None:
    anchor = _randn(128, 6, seed=100)
    expected = torch.tensor([4, 1, 5, 0, 3, 2], dtype=torch.long)
    source = _source_for_anchor_permutation(anchor, expected)

    correlation = standardized_cross_correlation(anchor, source, chunk_size=2)
    result = derive_neuron_permutation(
        anchor,
        source,
        exact_assignment_limit=6,
        chunk_size=2,
    )

    assert torch.equal(result.permutation, expected)
    torch.testing.assert_close(
        correlation[torch.arange(6), expected],
        torch.ones(6, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert result.diagnostics.solver == "bounded_exact_hungarian"
    assert result.diagnostics.certified_optimal is True
    assert result.diagnostics.complete is True
    assert result.diagnostics.collision_count == 0


def test_dense_and_gated_permutations_preserve_functions() -> None:
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inputs = _randn(11, 3, seed=110)
    dense = _dense_bundle(111)
    gated = _gated_bundle(121)

    aligned_dense = permute_mlp_bundle(dense, permutation)
    aligned_gated = permute_mlp_bundle(gated, permutation)

    torch.testing.assert_close(
        _dense_forward(aligned_dense, inputs),
        _dense_forward(dense, inputs),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        _gated_forward(aligned_gated, inputs),
        _gated_forward(gated, inputs),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.equal(
        aligned_dense.input_weight,
        dense.input_weight.index_select(0, permutation),
    )
    assert torch.equal(
        aligned_gated.up_weight,
        gated.up_weight.index_select(0, permutation),
    )
    assert torch.equal(
        aligned_gated.down_weight,
        gated.down_weight.index_select(1, permutation),
    )


def test_activation_alignment_and_weighted_merge() -> None:
    anchor_bundle = _dense_bundle(130)
    desired_aligned = MLPTensorBundle(
        **{
            role: tensor + float(index + 1)
            for index, (role, tensor) in enumerate(
                _bundle_tensors(anchor_bundle).items()
            )
        }
    )
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inverse = torch.argsort(permutation)
    unaligned = permute_mlp_bundle(desired_aligned, inverse)
    anchor_signatures = _randn(96, 4, seed=140)
    source_signatures = _source_for_anchor_permutation(
        anchor_signatures,
        permutation,
    )

    result = align_and_merge_mlp_bundles(
        [anchor_bundle, unaligned],
        [anchor_signatures, source_signatures],
        weights=[1.0, 3.0],
        chunk_size=2,
    )
    expected = merge_aligned_mlp_bundles(
        [anchor_bundle, desired_aligned],
        weights=[1.0, 3.0],
    )

    assert torch.equal(result.permutations[0], torch.arange(4))
    assert torch.equal(result.permutations[1], permutation)
    _assert_bundles_close(result.aligned_bundles[1], desired_aligned)
    _assert_bundles_close(result.merged_bundle, expected)
    torch.testing.assert_close(
        result.normalized_weights,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
    )
    assert result.execution_depth == 2
    assert result.parameter_counts.dense_equivalent_parameters == 52
    assert result.parameter_counts.unique_parameters == 26
    assert result.parameter_counts.parameters_saved == 26


def test_deterministic_bounded_greedy_reports_approximation_gap() -> None:
    scores = torch.tensor(
        [[0.90, 0.80], [0.85, 0.10]],
        dtype=torch.float64,
    )

    exact = solve_neuron_assignment(scores, exact_assignment_limit=2)
    first = solve_neuron_assignment(
        scores,
        exact_assignment_limit=0,
        chunk_size=1,
    )
    second = solve_neuron_assignment(
        scores.clone(),
        exact_assignment_limit=0,
        chunk_size=1,
    )

    assert torch.equal(exact.permutation, torch.tensor([1, 0]))
    assert exact.diagnostics.objective == pytest.approx(1.65)
    assert torch.equal(first.permutation, torch.tensor([0, 1]))
    assert torch.equal(first.permutation, second.permutation)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics.solver == "row_order_chunked_greedy"
    assert first.diagnostics.approximate is True
    assert first.diagnostics.assignment_steps == 2
    assert first.diagnostics.assignment_step_limit == 2
    assert first.diagnostics.suboptimality_upper_bound == pytest.approx(0.75)


class _DenseMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Linear(3, 4, bias=True)
        self.down = torch.nn.Linear(4, 2, bias=True)

    def forward(self, inputs):
        return self.down(torch.nn.functional.gelu(self.input(inputs)))


def _dense_layout() -> MLPModuleLayout:
    return MLPModuleLayout.dense("input", "down")


def _module_parameters(module: _DenseMLP):
    return {
        "input_weight": module.input.weight,
        "down_weight": module.down.weight,
        "input_bias": module.input.bias,
        "down_bias": module.down.bias,
    }


def test_actual_parameter_identity_sharing_and_unique_counts() -> None:
    modules = [_DenseMLP() for _ in range(3)]
    bundle = _dense_bundle(150)

    report = tie_mlp_modules(modules, bundle, _dense_layout())

    for role in _module_parameters(modules[0]):
        parameters = [_module_parameters(module)[role] for module in modules]
        assert all(parameter is parameters[0] for parameter in parameters)
        assert parameters[0].grad is None
        torch.testing.assert_close(
            parameters[0].detach(),
            _bundle_tensors(bundle)[role].to(dtype=parameters[0].dtype),
        )
    assert report.execution_depth == 3
    assert report.dense_equivalent_parameters == 78
    assert report.unique_parameters == 26
    assert report.dense_equivalent_parameter_objects == 12
    assert report.unique_parameter_objects == 4
    assert report.unique_parameter_sets == 1
    assert report.fully_shared is True
    assert parameter_sharing_report(modules, _dense_layout()) == report


def test_expand_is_inverse_storage_operation_with_independent_equal_clones() -> None:
    modules = [_DenseMLP() for _ in range(3)]
    tie_mlp_modules(modules, _dense_bundle(160), _dense_layout())
    shared_values = {
        role: parameter.detach().clone()
        for role, parameter in _module_parameters(modules[0]).items()
    }

    report = expand_mlp_modules(modules, _dense_layout())

    for role in shared_values:
        parameters = [_module_parameters(module)[role] for module in modules]
        assert len({id(parameter) for parameter in parameters}) == 3
        assert all(parameter.grad is None for parameter in parameters)
        for parameter in parameters:
            torch.testing.assert_close(parameter.detach(), shared_values[role])
    with torch.no_grad():
        modules[0].input.weight[0, 0].add_(1.0)
    assert not torch.equal(modules[0].input.weight, modules[1].input.weight)
    assert torch.equal(modules[1].input.weight, modules[2].input.weight)
    assert report.execution_depth == 3
    assert report.unique_parameters == 78
    assert report.unique_parameters == report.dense_equivalent_parameters
    assert report.unique_parameter_sets == 3
    assert report.fully_expanded is True


def test_malformed_topology_shapes_nonfinite_and_assignments_fail_closed() -> None:
    mixed = MLPTensorBundle(
        input_weight=torch.ones((4, 3)),
        gate_weight=torch.ones((4, 3)),
        up_weight=torch.ones((4, 3)),
        down_weight=torch.ones((2, 4)),
    )
    with pytest.raises(TopologyValidationError, match="without mixing"):
        permute_mlp_bundle(mixed, torch.arange(4))

    malformed = MLPTensorBundle(
        input_weight=torch.ones((4, 3)),
        down_weight=torch.ones((2, 5)),
    )
    with pytest.raises(ShapeValidationError, match="column count"):
        permute_mlp_bundle(malformed, torch.arange(4))

    with pytest.raises(AssignmentCollisionError, match="collisions"):
        permute_mlp_bundle(_dense_bundle(), [0, 0, 2, 3])
    with pytest.raises(IncompleteAssignmentError, match="exactly 4"):
        permute_mlp_bundle(_dense_bundle(), [0, 1, 2])
    with pytest.raises(ShapeValidationError, match="square"):
        solve_neuron_assignment(torch.ones((2, 3), dtype=torch.float64))

    nonfinite = _randn(12, 4, seed=170)
    nonfinite[0, 0] = torch.nan
    with pytest.raises(FiniteValidationError, match="NaN or infinity"):
        standardized_cross_correlation(nonfinite, _randn(12, 4, seed=171))
    constant = torch.ones((12, 4), dtype=torch.float64)
    with pytest.raises(ShapeValidationError, match="constant or near-constant"):
        standardized_cross_correlation(constant, constant)


def test_bundle_bias_mismatch_and_module_collisions_fail_closed() -> None:
    without_bias = MLPTensorBundle(
        input_weight=_randn(4, 3, seed=180),
        down_weight=_randn(2, 4, seed=181),
    )
    with pytest.raises(TopologyValidationError, match="optional-bias topology"):
        merge_aligned_mlp_bundles([_dense_bundle(182), without_bias])

    module = _DenseMLP()
    with pytest.raises(ModuleSharingError, match="same module object"):
        tie_mlp_modules([module, module], _dense_bundle(183), _dense_layout())

    modules = [_DenseMLP(), _DenseMLP()]
    bad_layout = MLPModuleLayout(
        input_weight="input.weight",
        down_weight="input.weight",
        input_bias="input.bias",
        down_bias="down.bias",
    )
    with pytest.raises(TopologyValidationError, match="collides"):
        tie_mlp_modules(modules, _dense_bundle(184), bad_layout)


def test_all_alignment_outputs_are_detached_and_inputs_receive_zero_gradients() -> None:
    first = _dense_bundle(190)
    second = _dense_bundle(200)
    for tensor in (*_bundle_tensors(first).values(), *_bundle_tensors(second).values()):
        tensor.requires_grad_()
    anchor = _randn(64, 4, seed=210).requires_grad_()
    permutation = torch.tensor([1, 3, 0, 2], dtype=torch.long)
    source = _source_for_anchor_permutation(anchor.detach(), permutation)
    source.requires_grad_()

    result = align_and_merge_mlp_bundles(
        [first, second],
        [anchor, source],
        weights=torch.tensor([2.0, 1.0], requires_grad=True),
        chunk_size=2,
    )

    outputs = [
        *(_bundle_tensors(result.merged_bundle).values()),
        *(
            tensor
            for bundle in result.aligned_bundles
            for tensor in _bundle_tensors(bundle).values()
        ),
        result.normalized_weights,
        *(assignment.permutation for assignment in result.assignments),
        *(assignment.assigned_correlations for assignment in result.assignments),
    ]
    outputs.extend(
        assignment.correlation_matrix
        for assignment in result.assignments
        if assignment.correlation_matrix is not None
    )
    assert all(output.requires_grad is False for output in outputs)
    assert all(output.grad_fn is None for output in outputs)
    assert anchor.grad is None
    assert source.grad is None
    assert all(
        tensor.grad is None
        for tensor in (
            *_bundle_tensors(first).values(),
            *_bundle_tensors(second).values(),
        )
    )
