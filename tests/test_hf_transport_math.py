from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_transport_math import (  # noqa: E402
    FiniteValidationError,
    PathConsistencyError,
    RankValidationError,
    ShapeValidationError,
    SinkhornConvergenceError,
    build_parent_residual_basis,
    check_path_composition,
    decode_coordinates,
    encode_coordinates,
    transport_linear_operator,
    uniform_entropic_sinkhorn,
)


def _rows(row_count: int, width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        row_count,
        width,
        generator=generator,
        dtype=torch.float64,
    )


def _coordinates(source_dim: int, target_dim: int, seed: int):
    return build_parent_residual_basis(
        _rows(source_dim + 7, source_dim, seed),
        target_dim,
    )


def test_square_orthogonal_transport_is_exact_coordinate_equivalence() -> None:
    input_coordinates = _coordinates(4, 4, 11)
    output_coordinates = _coordinates(3, 3, 12)
    source_weight = _rows(3, 4, 13)
    source_bias = _rows(1, 3, 14).squeeze(0)
    target_inputs = _rows(9, 4, 15)

    transported = transport_linear_operator(
        source_weight,
        input_coordinates,
        output_coordinates,
        source_bias=source_bias,
    )

    source_inputs = decode_coordinates(target_inputs, input_coordinates)
    source_outputs = source_inputs @ source_weight.T + source_bias
    expected_target_outputs = encode_coordinates(
        source_outputs,
        output_coordinates,
    )
    actual_target_outputs = target_inputs @ transported.weight.T + transported.bias

    torch.testing.assert_close(
        actual_target_outputs,
        expected_target_outputs,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        decode_coordinates(actual_target_outputs, output_coordinates),
        source_outputs,
        atol=1e-12,
        rtol=1e-12,
    )
    assert input_coordinates.diagnostics.target_identity_max_abs_error < 1e-12
    assert input_coordinates.diagnostics.decode_pseudoinverse_max_abs_error < 1e-12


def test_rectangular_transport_decodes_to_source_space_projection() -> None:
    input_coordinates = _coordinates(5, 3, 21)
    output_coordinates = _coordinates(6, 2, 22)
    source_weight = _rows(6, 5, 23)
    source_bias = _rows(1, 6, 24).squeeze(0)
    target_inputs = _rows(8, 3, 25)

    transported = transport_linear_operator(
        source_weight,
        input_coordinates,
        output_coordinates,
        source_bias=source_bias,
    )
    source_inputs = decode_coordinates(target_inputs, input_coordinates)
    source_outputs = source_inputs @ source_weight.T + source_bias
    target_outputs = target_inputs @ transported.weight.T + transported.bias
    decoded_outputs = decode_coordinates(target_outputs, output_coordinates)
    projected_outputs = source_outputs @ output_coordinates.source_projector

    torch.testing.assert_close(
        decoded_outputs,
        projected_outputs,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.linalg.matrix_rank(output_coordinates.source_projector).item() == 2
    assert not torch.allclose(decoded_outputs, source_outputs, atol=1e-8, rtol=1e-8)


def test_two_linear_path_reuses_boundary_map_and_composes_projection() -> None:
    input_coordinates = _coordinates(4, 3, 31)
    boundary_coordinates = _coordinates(5, 3, 32)
    output_coordinates = _coordinates(3, 2, 33)
    producer_weight = _rows(5, 4, 34)
    producer_bias = _rows(1, 5, 35).squeeze(0)
    consumer_weight = _rows(3, 5, 36)
    consumer_bias = _rows(1, 3, 37).squeeze(0)

    producer = transport_linear_operator(
        producer_weight,
        input_coordinates,
        boundary_coordinates,
        source_bias=producer_bias,
    )
    consumer = transport_linear_operator(
        consumer_weight,
        boundary_coordinates,
        output_coordinates,
        source_bias=consumer_bias,
    )

    diagnostics = check_path_composition(
        producer,
        consumer,
        producer_source_weight=producer_weight,
        consumer_source_weight=consumer_weight,
        producer_source_bias=producer_bias,
        consumer_source_bias=consumer_bias,
        atol=1e-12,
        rtol=1e-12,
    )

    target_inputs = _rows(7, 3, 38)
    actual = (
        target_inputs @ producer.weight.T + producer.bias
    ) @ consumer.weight.T + consumer.bias
    source_inputs = decode_coordinates(target_inputs, input_coordinates)
    source_mid = source_inputs @ producer_weight.T + producer_bias
    projected_mid = source_mid @ boundary_coordinates.source_projector
    source_outputs = projected_mid @ consumer_weight.T + consumer_bias
    expected = encode_coordinates(source_outputs, output_coordinates)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert diagnostics.shared_boundary_map is True
    assert diagnostics.boundary_projector_rank == 3
    assert diagnostics.composed_weight_max_abs_error < 1e-12
    assert diagnostics.passed is True


def test_path_check_rejects_independently_fitted_boundary_coordinates() -> None:
    input_coordinates = _coordinates(4, 3, 41)
    producer_boundary = _coordinates(5, 3, 42)
    consumer_boundary = _coordinates(5, 3, 43)
    output_coordinates = _coordinates(3, 2, 44)
    producer_weight = _rows(5, 4, 45)
    consumer_weight = _rows(3, 5, 46)
    producer = transport_linear_operator(
        producer_weight,
        input_coordinates,
        producer_boundary,
    )
    consumer = transport_linear_operator(
        consumer_weight,
        consumer_boundary,
        output_coordinates,
    )

    with pytest.raises(PathConsistencyError, match="one shared boundary"):
        check_path_composition(
            producer,
            consumer,
            producer_source_weight=producer_weight,
            consumer_source_weight=consumer_weight,
        )


def test_parent_basis_is_bitwise_deterministic_with_canonical_signs() -> None:
    activations = _rows(30, 7, 51)

    first = build_parent_residual_basis(activations, 5)
    second = build_parent_residual_basis(activations.clone(), 5)

    assert first.decode.dtype == torch.float64
    assert first.decode.device.type == "cpu"
    assert torch.equal(first.decode, second.decode)
    assert torch.equal(first.encode, second.encode)
    pivot_rows = torch.argmax(torch.abs(first.decode), dim=0)
    columns = torch.arange(first.target_dim)
    assert bool((first.decode[pivot_rows, columns] > 0.0).all())


def test_log_sinkhorn_has_uniform_marginals_and_finite_mass() -> None:
    source = _rows(5, 4, 61)
    target = _rows(3, 4, 62)

    result = uniform_entropic_sinkhorn(
        source,
        target,
        epsilon=0.4,
        max_iterations=2_000,
        tolerance=1e-11,
    )

    torch.testing.assert_close(
        result.plan.sum(dim=1),
        torch.full((5,), 0.2, dtype=torch.float64),
        atol=1e-11,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.plan.sum(dim=0),
        torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
        atol=1e-11,
        rtol=0.0,
    )
    assert result.diagnostics.converged is True
    assert result.diagnostics.finite is True
    assert result.diagnostics.iterations <= 2_000
    assert result.diagnostics.mass_error < 1e-12


def test_log_sinkhorn_stays_finite_when_naive_kernel_underflows() -> None:
    source = torch.tensor([[0.0], [1_000.0], [2_000.0]], dtype=torch.float64)
    target = torch.tensor([[0.1], [1_000.1], [2_000.1]], dtype=torch.float64)

    result = uniform_entropic_sinkhorn(
        source,
        target,
        epsilon=0.01,
        max_iterations=100,
        tolerance=1e-12,
    )

    assert bool(torch.isfinite(result.plan).all())
    assert result.diagnostics.cost_max > 1_000.0
    assert result.diagnostics.total_mass == pytest.approx(1.0, abs=1e-12)


def test_malformed_nan_and_rank_deficient_inputs_fail_closed() -> None:
    with pytest.raises(ShapeValidationError, match="two-dimensional"):
        build_parent_residual_basis(torch.ones(5, dtype=torch.float64), 1)

    nan_rows = torch.eye(4, dtype=torch.float64)
    nan_rows[0, 0] = torch.nan
    with pytest.raises(FiniteValidationError, match="NaN or infinity"):
        build_parent_residual_basis(nan_rows, 2)

    rank_one = torch.arange(8, dtype=torch.float64).unsqueeze(1).repeat(1, 3)
    with pytest.raises(RankValidationError, match="effective rank"):
        build_parent_residual_basis(rank_one, 2)

    with pytest.raises(FiniteValidationError, match="rank tolerance"):
        build_parent_residual_basis(_rows(8, 3, 70), 2, rank_rtol=1e308)

    coordinates = _coordinates(4, 3, 71)
    with pytest.raises(ShapeValidationError, match="source_weight must have shape"):
        transport_linear_operator(
            torch.ones((3, 4), dtype=torch.float64),
            coordinates,
            coordinates,
        )

    with pytest.raises(ShapeValidationError, match="same feature width"):
        uniform_entropic_sinkhorn(
            torch.ones((2, 3), dtype=torch.float64),
            torch.ones((2, 4), dtype=torch.float64),
        )


def test_sinkhorn_nonconvergence_reports_bounded_failure() -> None:
    source = torch.tensor([[0.0], [0.2], [2.0]], dtype=torch.float64)
    target = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

    with pytest.raises(SinkhornConvergenceError) as caught:
        uniform_entropic_sinkhorn(
            source,
            target,
            epsilon=0.2,
            max_iterations=1,
            tolerance=1e-16,
        )

    assert caught.value.diagnostics.iterations == 1
    assert caught.value.diagnostics.max_iterations == 1
    assert caught.value.diagnostics.converged is False
    assert caught.value.diagnostics.finite is True


def test_all_outputs_are_detached_and_inputs_accumulate_no_gradients() -> None:
    activations = _rows(12, 4, 81).requires_grad_()
    source_weight = _rows(3, 4, 82).requires_grad_()
    source_bias = _rows(1, 3, 83).squeeze(0).requires_grad_()
    source_signatures = _rows(4, 3, 84).requires_grad_()
    target_signatures = _rows(3, 3, 85).requires_grad_()

    input_coordinates = build_parent_residual_basis(activations, 3)
    output_coordinates = _coordinates(3, 2, 86)
    transported = transport_linear_operator(
        source_weight,
        input_coordinates,
        output_coordinates,
        source_bias=source_bias,
    )
    sinkhorn = uniform_entropic_sinkhorn(
        source_signatures,
        target_signatures,
        epsilon=0.5,
        max_iterations=1_000,
        tolerance=1e-10,
    )

    outputs = (
        input_coordinates.decode,
        input_coordinates.encode,
        input_coordinates.source_projector,
        input_coordinates.singular_values,
        transported.weight,
        transported.bias,
        sinkhorn.plan,
        sinkhorn.cost_matrix,
    )
    assert all(value.requires_grad is False for value in outputs)
    assert all(value.grad_fn is None for value in outputs)
    assert activations.grad is None
    assert source_weight.grad is None
    assert source_bias.grad is None
    assert source_signatures.grad is None
    assert target_signatures.grad is None
