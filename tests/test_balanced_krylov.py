from __future__ import annotations

import json

import numpy as np
import pytest

from neural_forge.balanced_krylov import (
    BalancedKrylovConditionError,
    BalancedKrylovFiniteError,
    BalancedKrylovRankError,
    BalancedKrylovShapeError,
    build_balanced_krylov_basis,
    canonical_array_sha256,
    linear_rollout,
    multi_step_rollout_error,
    multi_step_rollout_error_report,
    reconstruction_error,
    reduced_linear_rollout,
    relative_reconstruction_error,
    rollout_error_report,
)


def _output_critical_system():
    transition = np.diag(np.array([0.25, 0.97], dtype=np.float64))
    output = np.array([[0.0, 1.0]], dtype=np.float64)

    # The first state owns over 99.99% of static activation energy.  The
    # second state is low variance but is the only state visible at the output.
    activations = np.array(
        [
            [12.0, 0.08],
            [12.0, -0.08],
            [-12.0, 0.08],
            [-12.0, -0.08],
        ],
        dtype=np.float64,
    )
    forward_responses = []
    current = activations.copy()
    for _ in range(4):
        current = current @ transition.T
        forward_responses.append(current.copy())
    dual_responses = np.vstack(
        [output @ np.linalg.matrix_power(transition, step) for step in range(8)]
    )
    return transition, output, activations, forward_responses, dual_responses


def test_balancing_preserves_low_variance_output_critical_mode() -> None:
    (
        transition,
        output,
        activations,
        forward_responses,
        dual_responses,
    ) = _output_critical_system()

    activation_pca = build_balanced_krylov_basis(
        activations,
        rank=1,
        forward_responses=forward_responses,
    )
    balanced = build_balanced_krylov_basis(
        activations,
        rank=1,
        forward_jvp_responses=forward_responses,
        output_sensitivities=dual_responses,
    )

    assert activation_pca.report.method == "controllability_svd_fallback"
    assert activation_pca.report.captured_controllability_energy_fraction > 0.9999
    assert abs(activation_pca.decoder[0, 0]) > 0.999
    assert balanced.report.method == "balanced_pod"
    assert balanced.report.fallback_used is False
    assert abs(balanced.decoder[1, 0]) > abs(balanced.decoder[0, 0])

    held_out = np.array(
        [[3.0, 0.5], [-4.0, -0.3], [1.5, 0.8]],
        dtype=np.float64,
    )
    pca_error = multi_step_rollout_error(
        transition,
        held_out,
        activation_pca,
        steps=12,
        output_matrix=output,
    )
    balanced_report = multi_step_rollout_error_report(
        transition,
        held_out,
        balanced,
        steps=12,
        output_matrix=output,
    )

    assert pca_error > 0.99
    assert balanced_report.relative_l2_error < 1e-12
    assert balanced_report.relative_l2_error < pca_error * 1e-6
    assert max(balanced_report.per_step_relative_l2_error) < 1e-12


def test_basis_is_biorthogonal_deterministic_hashed_and_zero_optimizer() -> None:
    generator = np.random.default_rng(20260710)
    activations = generator.normal(size=(64, 5)).astype(np.float64)
    responses = [
        generator.normal(size=(23, 5)).astype(np.float64),
        generator.normal(size=(17, 5)).astype(np.float64),
    ]
    sensitivities = generator.normal(size=(31, 5)).astype(np.float64)

    first = build_balanced_krylov_basis(
        activations,
        rank=3,
        krylov_responses=responses,
        dual_sensitivities=sensitivities,
    )
    second = build_balanced_krylov_basis(
        activations.copy(),
        rank=3,
        krylov_responses=[value.copy() for value in responses],
        dual_sensitivities=sensitivities.copy(),
    )

    np.testing.assert_array_equal(first.encoder, second.encoder)
    np.testing.assert_array_equal(first.decoder, second.decoder)
    np.testing.assert_array_equal(first.mean, second.mean)
    assert first.report == second.report
    np.testing.assert_allclose(
        first.encoder @ first.decoder,
        np.eye(3),
        atol=1e-12,
        rtol=0.0,
    )
    assert all(
        first.decoder[pivot, index] >= 0.0
        for index, pivot in enumerate(first.report.canonical_pivot_indices)
    )
    assert all(value >= 0.0 for value in first.report.canonical_pivot_values)
    assert first.encoder.flags.writeable is False
    assert first.decoder.flags.writeable is False
    assert first.mean.flags.writeable is False

    report = first.report
    assert report.finite is True
    assert report.shape_checks_passed is True
    assert report.deterministic is True
    assert report.backward_used is False
    assert report.backward_operations == 0
    assert report.optimizer_used is False
    assert report.optimizer_steps == 0
    assert report.zero_gradient is True
    assert report.no_training is True
    assert report.biorthogonality_max_abs_error < 1e-12
    assert report.projection_idempotence_max_abs_error < 1e-12
    assert report.basis_sha256.startswith("sha256:")
    assert len(report.basis_sha256) == 71
    assert canonical_array_sha256(first.encoder) == report.encoder_sha256
    assert json.dumps(report.as_dict(), allow_nan=False)
    assert report.as_dict()["execution"] == {
        "deterministic": True,
        "backward_used": False,
        "backward_operations": 0,
        "optimizer_used": False,
        "optimizer_steps": 0,
        "zero_gradient": True,
        "no_training": True,
    }


def test_rank_deficient_hankel_uses_safe_joint_svd_fallback() -> None:
    activations = np.vstack((np.eye(3), -np.eye(3))).astype(np.float64)
    sensitivities = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    basis = build_balanced_krylov_basis(
        activations,
        rank=2,
        output_sensitivities=sensitivities,
        center=False,
    )

    assert basis.report.method == "joint_sketch_svd_fallback"
    assert basis.report.fallback_used is True
    assert "effective rank 1" in basis.report.fallback_reason
    assert basis.report.hankel_rank == 1
    np.testing.assert_allclose(basis.encoder, basis.decoder.T, atol=1e-15)
    np.testing.assert_allclose(basis.encoder @ basis.decoder, np.eye(2))
    assert np.isfinite(basis.encoder).all()
    assert np.isfinite(basis.decoder).all()

    with pytest.raises(BalancedKrylovConditionError, match="fallback is disabled"):
        build_balanced_krylov_basis(
            activations,
            rank=2,
            output_sensitivities=sensitivities,
            center=False,
            allow_svd_fallback=False,
        )


def test_degenerate_svd_subspace_has_canonical_coordinate_order() -> None:
    activations = np.eye(4, dtype=np.float64)

    first = build_balanced_krylov_basis(
        activations,
        rank=2,
        center=False,
    )
    second = build_balanced_krylov_basis(
        activations,
        rank=2,
        center=False,
    )

    expected = np.eye(4, dtype=np.float64)[:, :2]
    np.testing.assert_array_equal(first.decoder, expected)
    np.testing.assert_array_equal(first.decoder, second.decoder)
    assert first.report.canonical_pivot_indices == (0, 1)


def test_reconstruction_and_rollout_helpers_are_consistent() -> None:
    activations = np.array(
        [[-2.0, 1.0], [2.0, -1.0], [1.0, 2.0], [-1.0, -2.0]],
        dtype=np.float64,
    )
    basis = build_balanced_krylov_basis(activations, rank=2)
    transition = np.array([[0.8, 0.1], [-0.2, 0.7]], dtype=np.float64)
    initial = np.array([[1.5, -0.5], [-0.2, 0.8]], dtype=np.float64)

    assert relative_reconstruction_error(initial, basis) < 1e-12
    assert reconstruction_error(initial, basis, relative=False) < 1e-12
    reference = linear_rollout(transition, initial, 6)
    approximate = reduced_linear_rollout(transition, initial, basis, 6)
    np.testing.assert_allclose(approximate, reference, atol=1e-12, rtol=1e-12)
    report = rollout_error_report(reference, approximate)
    assert report.transition_count == 6
    assert report.evaluated_step_count == 6
    assert report.relative_l2_error < 1e-12
    assert report.finite is True


def test_torch_inputs_are_detached_without_gradients_or_mutation() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(19)
    activations = torch.randn(
        24,
        4,
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )
    sensitivities = torch.randn(
        12,
        4,
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )
    activation_copy = activations.detach().clone()
    sensitivity_copy = sensitivities.detach().clone()

    basis = build_balanced_krylov_basis(
        activations,
        rank=2,
        output_sensitivities=sensitivities,
    )

    assert isinstance(basis.encoder, np.ndarray)
    assert activations.grad is None
    assert sensitivities.grad is None
    torch.testing.assert_close(activations.detach(), activation_copy)
    torch.testing.assert_close(sensitivities.detach(), sensitivity_copy)
    assert basis.report.backward_used is False
    assert basis.report.optimizer_used is False


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_fail_closed(nonfinite: float) -> None:
    activations = np.eye(3, dtype=np.float64)
    activations[0, 1] = nonfinite

    with pytest.raises(BalancedKrylovFiniteError, match="NaN or infinity"):
        build_balanced_krylov_basis(activations, rank=1)


@pytest.mark.parametrize(
    ("activations", "kwargs", "error", "message"),
    [
        (
            np.ones(3, dtype=np.float64),
            {},
            BalancedKrylovShapeError,
            "two-dimensional",
        ),
        (
            np.ones((3, 2), dtype=np.int64),
            {},
            BalancedKrylovShapeError,
            "floating-point",
        ),
        (
            np.ones((3, 2), dtype=np.float64),
            {"rank": 0},
            BalancedKrylovShapeError,
            "positive integer",
        ),
        (
            np.ones((3, 2), dtype=np.float64),
            {"rank": 3},
            BalancedKrylovShapeError,
            "cannot exceed 2",
        ),
        (
            np.eye(2, dtype=np.float64),
            {"output_sensitivities": np.ones((2, 3), dtype=np.float64)},
            BalancedKrylovShapeError,
            "does not match",
        ),
        (
            np.eye(2, dtype=np.float64),
            {"max_condition_number": 0.5},
            BalancedKrylovShapeError,
            "at least one",
        ),
    ],
)
def test_shape_and_control_validation_fails_closed(
    activations,
    kwargs,
    error,
    message: str,
) -> None:
    rank = kwargs.pop("rank", 1)
    with pytest.raises(error, match=message):
        build_balanced_krylov_basis(activations, rank=rank, **kwargs)


def test_rank_and_rollout_shape_errors_fail_closed() -> None:
    rank_one_rows = np.ones((6, 3), dtype=np.float64)
    with pytest.raises(BalancedKrylovRankError, match="effective rank 0"):
        build_balanced_krylov_basis(rank_one_rows, rank=1)

    with pytest.raises(BalancedKrylovShapeError, match="must be square"):
        linear_rollout(
            np.ones((2, 3), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            2,
        )
    with pytest.raises(BalancedKrylovShapeError, match="same shape"):
        rollout_error_report(
            np.ones((3, 1), dtype=np.float64),
            np.ones((4, 1), dtype=np.float64),
        )
    with pytest.raises(BalancedKrylovShapeError, match="one transition"):
        rollout_error_report(
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
        )


def test_hashes_change_with_snapshot_content_and_preserve_group_boundaries() -> None:
    first_rows = np.eye(3, dtype=np.float64)
    second_rows = first_rows.copy()
    second_rows[0, 0] += 1e-6

    first = build_balanced_krylov_basis(first_rows, rank=2, center=False)
    second = build_balanced_krylov_basis(second_rows, rank=2, center=False)
    split = build_balanced_krylov_basis(
        [first_rows[:1], first_rows[1:]],
        rank=2,
        center=False,
    )

    assert first.report.activation_sha256 != second.report.activation_sha256
    assert first.report.provenance_sha256 != second.report.provenance_sha256
    assert first.report.activation_sha256 != split.report.activation_sha256
    assert canonical_array_sha256(first_rows) == canonical_array_sha256(
        first_rows.copy()
    )
