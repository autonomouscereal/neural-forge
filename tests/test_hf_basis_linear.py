from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_basis_linear as basis_linear_module  # noqa: E402
from neural_forge.hf_basis_linear import (  # noqa: E402
    DepthBasisBudgetError,
    DepthBasisFiniteError,
    DepthBasisLinear,
    DepthBasisLinearError,
    DepthBasisRankError,
    DepthBasisShapeError,
    factorize_linears,
    materialize_linears,
    stack_linear_operators,
)


def _linears(
    count: int = 4,
    *,
    in_features: int = 3,
    out_features: int = 2,
    bias: bool = True,
    dtype=torch.float64,
    seed: int = 1,
):
    generator = torch.Generator().manual_seed(seed)
    modules = []
    for _ in range(count):
        module = torch.nn.Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
        )
        with torch.no_grad():
            module.weight.copy_(
                torch.randn(
                    out_features,
                    in_features,
                    generator=generator,
                    dtype=dtype,
                )
            )
            if module.bias is not None:
                module.bias.copy_(
                    torch.randn(
                        out_features,
                        generator=generator,
                        dtype=dtype,
                    )
                )
        modules.append(module)
    return modules


@pytest.mark.parametrize("bias", [False, True])
def test_full_layer_rank_exactly_reconstructs_forwards_and_parameters(bias) -> None:
    linears = _linears(bias=bias, seed=10)
    representation = factorize_linears(linears, feature_chunk_size=2)
    inputs = torch.randn(7, 3, dtype=torch.float64)

    assert representation.rank == 4
    assert representation.report is not None
    assert representation.report.selection == "full_rank"
    assert representation.report.full_rank_reconstruction is True
    for index, linear in enumerate(linears):
        weight, actual_bias = representation.materialize_weight(index)
        torch.testing.assert_close(weight, linear.weight, atol=1e-12, rtol=1e-12)
        if bias:
            torch.testing.assert_close(
                actual_bias,
                linear.bias,
                atol=1e-12,
                rtol=1e-12,
            )
        else:
            assert actual_bias is None
        torch.testing.assert_close(
            representation(inputs, index),
            linear(inputs),
            atol=1e-12,
            rtol=1e-12,
        )


def test_low_rank_error_energy_and_parameter_budget_are_bounded() -> None:
    linears = _linears(count=3, in_features=4, out_features=3, seed=20)
    rank_one = factorize_linears(linears, rank=1)
    energy_fit = factorize_linears(linears, energy=0.8)
    feature_count = 3 * 4 + 3
    budget = 2 * (feature_count + 3)
    budget_fit = factorize_linears(linears, parameter_budget=budget)
    source = stack_linear_operators(linears)

    singular_values = torch.linalg.svdvals(source)
    expected_error = float(singular_values[1:].square().sum().item())
    assert rank_one.report is not None
    assert rank_one.report.squared_reconstruction_error == pytest.approx(
        expected_error,
        rel=1e-12,
        abs=1e-12,
    )
    assert 0.0 <= rank_one.report.retained_energy_fraction <= 1.0
    assert energy_fit.report is not None
    assert energy_fit.report.retained_energy_fraction >= 0.8 - 1e-12
    assert budget_fit.rank == 2
    assert budget_fit.report is not None
    assert budget_fit.report.unique_parameter_count == budget
    assert budget_fit.report.budget_headroom == 0


def test_decomposition_and_sign_canonicalization_are_deterministic() -> None:
    linears = _linears(seed=30)

    first = factorize_linears(linears, rank=3)
    second = factorize_linears(linears, rank=3)

    assert torch.equal(first.basis_weight, second.basis_weight)
    assert torch.equal(first.basis_bias, second.basis_bias)
    assert torch.equal(first.coefficients, second.coefficients)
    assert first.report == second.report
    flattened = torch.cat(
        [first.basis_weight.flatten(1), first.basis_bias],
        dim=1,
    )
    pivots = torch.argmax(torch.abs(flattened), dim=1)
    assert torch.all(flattened[torch.arange(first.rank), pivots] >= 0.0)


def test_unique_and_logical_parameter_reports_include_coefficients() -> None:
    linears = _linears(count=5, in_features=3, out_features=2, seed=40)
    representation = factorize_linears(linears, rank=2)
    report = representation.report

    assert report is not None
    assert report.logical_parameter_count == 5 * (2 * 3 + 2)
    assert report.basis_parameters == 2 * (2 * 3 + 2)
    assert report.coefficient_parameters == 5 * 2
    assert report.unique_parameter_count == sum(
        parameter.numel() for parameter in representation.parameters()
    )
    assert report.as_dict()["unique_parameter_count"] == report.unique_parameters


def test_runtime_parameters_receive_gradients_but_sources_do_not() -> None:
    linears = _linears(seed=50)
    for linear in linears:
        linear.weight.requires_grad_(True)
        if linear.bias is not None:
            linear.bias.requires_grad_(True)
    representation = factorize_linears(linears, rank=2)
    inputs = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)

    loss = sum(representation(inputs, index).square().mean() for index in range(4))
    loss.backward()

    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in representation.parameters())
    assert all(linear.weight.grad is None for linear in linears)
    assert all(linear.bias.grad is None for linear in linears)


def test_optional_tied_coefficient_rows_share_storage_and_execution() -> None:
    sources = _linears(count=2, seed=60)
    linears = [sources[0], sources[0], sources[1], sources[1]]
    representation = factorize_linears(
        linears,
        tied_coefficient_rows=[7, 7, 3, 3],
        feature_chunk_size=2,
    )
    inputs = torch.randn(5, 3, dtype=torch.float64)

    assert representation.coefficient_rows == 2
    assert representation.layer_to_coefficient.tolist() == [0, 0, 1, 1]
    assert representation.report is not None
    assert representation.report.coefficient_parameters == 4
    for index, source in enumerate(linears):
        torch.testing.assert_close(
            representation(inputs, index),
            source(inputs),
            atol=1e-12,
            rtol=1e-12,
        )


def test_tied_coefficient_rows_reconstruct_the_group_mean() -> None:
    linears = _linears(count=2, in_features=2, out_features=2, seed=65)
    representation = factorize_linears(
        linears,
        tied_coefficient_rows=[0, 0],
        feature_chunk_size=1,
    )
    expected_weight = (linears[0].weight + linears[1].weight) * 0.5
    expected_bias = (linears[0].bias + linears[1].bias) * 0.5

    for layer_index in range(2):
        weight, bias = representation.materialize_weight(layer_index)
        torch.testing.assert_close(weight, expected_weight, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(bias, expected_bias, atol=1e-12, rtol=1e-12)
    assert representation.report is not None
    assert representation.report.full_rank_reconstruction is False


def test_materialization_returns_independent_trainable_linears() -> None:
    representation = factorize_linears(_linears(seed=70))
    materialized = materialize_linears(representation)
    inputs = torch.randn(3, 3, dtype=torch.float64)

    for index, linear in enumerate(materialized):
        torch.testing.assert_close(
            linear(inputs),
            representation(inputs, index),
            atol=1e-12,
            rtol=1e-12,
        )
        assert linear.weight.requires_grad is True
    assert len({id(linear.weight) for linear in materialized}) == 4
    assert len({id(linear.bias) for linear in materialized}) == 4
    before = materialized[1].weight.detach().clone()
    with torch.no_grad():
        materialized[0].weight.add_(1.0)
        representation.basis_weight.add_(1.0)
    assert torch.equal(materialized[1].weight, before)


def test_float32_dtype_is_preserved() -> None:
    linears = _linears(count=2, bias=False, dtype=torch.float32, seed=80)
    representation = factorize_linears(linears)
    materialized = representation.materialize()

    assert representation.basis_weight.dtype == torch.float32
    assert representation.coefficients.dtype == torch.float32
    assert representation.basis_bias is None
    assert all(linear.weight.dtype == torch.float32 for linear in materialized)


def test_factorization_never_calls_dense_stack_helper(monkeypatch) -> None:
    linears = _linears(count=3, in_features=5, out_features=4, seed=85)

    def fail_dense_stack(*_args, **_kwargs):
        raise AssertionError("production fitting materialized the dense stack")

    monkeypatch.setattr(
        basis_linear_module,
        "stack_linear_operators",
        fail_dense_stack,
    )
    representation = factorize_linears(linears, feature_chunk_size=3)
    inputs = torch.randn(6, 5, dtype=torch.float64)

    for index, linear in enumerate(linears):
        torch.testing.assert_close(
            representation(inputs, index),
            linear(inputs),
            atol=1e-11,
            rtol=1e-11,
        )
    report = representation.report
    assert report is not None
    assert report.algorithm == "two_pass_chunked_layer_gram_eigh"
    assert report.feature_chunk_size == 3
    assert report.feature_chunks == 8
    assert report.gram_elements == 9
    assert report.fitting_passes == 2
    assert report.materialized_operator_stack is False
    assert report.materialized_full_right_singular_matrix is False
    assert report.estimated_peak_float64_bytes == (
        report.estimated_peak_float64_elements * 8
    )


def test_feature_chunk_size_is_invariant_within_float64_tolerance() -> None:
    linears = _linears(count=4, in_features=5, out_features=4, seed=86)

    one = factorize_linears(linears, rank=3, feature_chunk_size=1)
    uneven = factorize_linears(linears, rank=3, feature_chunk_size=7)
    whole = factorize_linears(linears, rank=3, feature_chunk_size=10_000)

    for candidate in (uneven, whole):
        torch.testing.assert_close(
            candidate.coefficients,
            one.coefficients,
            atol=2e-11,
            rtol=2e-11,
        )
        torch.testing.assert_close(
            candidate.basis_weight,
            one.basis_weight,
            atol=2e-11,
            rtol=2e-11,
        )
        torch.testing.assert_close(
            candidate.basis_bias,
            one.basis_bias,
            atol=2e-11,
            rtol=2e-11,
        )
        assert candidate.report is not None
        assert one.report is not None
        assert candidate.report.squared_reconstruction_error == pytest.approx(
            one.report.squared_reconstruction_error,
            rel=2e-12,
            abs=2e-12,
        )
    assert whole.report is not None
    assert whole.report.requested_feature_chunk_size == 10_000
    assert whole.report.feature_chunk_size == whole.report.feature_count


def test_zero_and_exactly_degenerate_gram_spectra_are_deterministic() -> None:
    zero_linears = _linears(count=3, in_features=2, out_features=2, seed=87)
    with torch.no_grad():
        for linear in zero_linears:
            linear.weight.zero_()
            linear.bias.zero_()
    zero = factorize_linears(zero_linears, feature_chunk_size=2)

    assert zero.report is not None
    assert zero.report.numerical_rank == 0
    assert zero.report.full_rank_reconstruction is True
    assert torch.count_nonzero(zero.coefficients) == 0
    assert torch.count_nonzero(zero.basis_weight) == 0
    assert torch.count_nonzero(zero.basis_bias) == 0

    degenerate = _linears(count=2, in_features=1, out_features=1, seed=88)
    with torch.no_grad():
        degenerate[0].weight.fill_(1.0)
        degenerate[0].bias.zero_()
        degenerate[1].weight.zero_()
        degenerate[1].bias.fill_(1.0)
    first = factorize_linears(degenerate, rank=1, feature_chunk_size=1)
    second = factorize_linears(degenerate, rank=1, feature_chunk_size=2)

    assert first.report is not None
    assert first.report.degenerate_eigenspaces >= 1
    torch.testing.assert_close(first.coefficients, second.coefficients)
    torch.testing.assert_close(first.basis_weight, second.basis_weight)
    torch.testing.assert_close(first.basis_bias, second.basis_bias)


def test_malformed_sources_selectors_and_runtime_inputs_fail_closed() -> None:
    with pytest.raises(DepthBasisShapeError, match="at least one"):
        factorize_linears([])
    with pytest.raises(DepthBasisShapeError, match="Every operator"):
        factorize_linears([torch.nn.ReLU()])

    mixed_shape = _linears(count=1) + _linears(
        count=1,
        in_features=4,
        seed=90,
    )
    with pytest.raises(DepthBasisShapeError, match="same canonical"):
        factorize_linears(mixed_shape)
    mixed_bias = _linears(count=1, bias=True) + _linears(
        count=1,
        bias=False,
        seed=91,
    )
    with pytest.raises(DepthBasisShapeError, match="optional-bias"):
        factorize_linears(mixed_bias)

    nonfinite = _linears(count=1)
    with torch.no_grad():
        nonfinite[0].weight[0, 0] = torch.nan
    with pytest.raises(DepthBasisFiniteError, match="NaN or infinity"):
        factorize_linears(nonfinite)

    valid = _linears(count=2)
    with pytest.raises(DepthBasisLinearError, match="at most one"):
        factorize_linears(valid, rank=1, energy=0.9)
    with pytest.raises(DepthBasisRankError, match="cannot exceed"):
        factorize_linears(valid, rank=3)
    with pytest.raises(DepthBasisRankError, match="interval"):
        factorize_linears(valid, energy=0.0)
    with pytest.raises(DepthBasisBudgetError, match="at least"):
        factorize_linears(valid, parameter_budget=1)
    with pytest.raises(DepthBasisShapeError, match="feature_chunk_size"):
        factorize_linears(valid, feature_chunk_size=0)
    with pytest.raises(DepthBasisShapeError, match="one value"):
        factorize_linears(valid, tied_coefficient_rows=[0])

    representation = factorize_linears(valid)
    with pytest.raises(DepthBasisShapeError, match="layer_index"):
        representation(torch.ones(2, 3, dtype=torch.float64), 2)
    with pytest.raises(DepthBasisShapeError, match="last dimension"):
        representation(torch.ones(2, 4, dtype=torch.float64), 0)


def test_direct_factor_validation_rejects_bad_shapes_and_nonfinite_values() -> None:
    with pytest.raises(DepthBasisShapeError, match="columns"):
        DepthBasisLinear(
            torch.ones(2, 3, 4),
            torch.ones(3, 3),
        )
    bad = torch.ones(2, 3, 4)
    bad[0, 0, 0] = torch.inf
    with pytest.raises(DepthBasisFiniteError, match="NaN or infinity"):
        DepthBasisLinear(bad, torch.ones(3, 2))
    with pytest.raises(DepthBasisShapeError, match="Every stored"):
        DepthBasisLinear(
            torch.ones(2, 3, 4),
            torch.ones(2, 2),
            layer_to_coefficient=[0, 0],
        )
