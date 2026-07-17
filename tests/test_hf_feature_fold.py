from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from neural_forge import hf_feature_fold as feature_fold  # noqa: E402
from neural_forge.hf_feature_fold import (  # noqa: E402
    FeatureFoldFiniteError,
    FeatureFoldMap,
    FeatureFoldMemoryError,
    FeatureFoldRankError,
    FeatureFoldShapeError,
    build_coupled_feature_fold_map,
    build_feature_fold_map,
)


def _rows(rows: int, width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, width, generator=generator, dtype=torch.float64)


def test_sparse_apply_decode_and_weight_folds_equal_tiny_dense_map() -> None:
    calibration = _rows(31, 9, 10)
    fold = build_feature_fold_map(
        calibration,
        4,
        chunk_size=3,
        sketch_size=8,
    )
    dense = fold.materialize_dense(max_elements=100)
    source_rows = _rows(14, 9, 11)
    target_rows = _rows(14, 4, 12)
    producer = _rows(9, 5, 13)
    consumer = _rows(6, 9, 14)

    torch.testing.assert_close(
        fold.encode(source_rows),
        source_rows @ dense,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        fold.decode(target_rows),
        target_rows @ dense.T,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        fold.fold_producer_weight(producer),
        dense.T @ producer,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        fold.fold_consumer_weight(consumer),
        consumer @ dense,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dense.T @ dense,
        torch.eye(4, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_build_has_full_coverage_nonempty_balanced_groups_and_report() -> None:
    first = _rows(27, 103, 20)
    second = _rows(19, 103, 21) + 0.4 * first.mean(dim=0)
    fold = build_coupled_feature_fold_map(
        [first, second],
        17,
        chunk_size=11,
        sketch_size=7,
    )
    report = fold.report

    assert report is not None
    assert fold.assignment.shape == (103,)
    assert fold.coefficients.shape == (103,)
    assert torch.equal(
        torch.sort(torch.unique(fold.assignment)).values,
        torch.arange(17),
    )
    assert report.full_source_coverage is True
    assert report.zero_dropped_channels is True
    assert report.source_channels_covered == 103
    assert sum(report.group_sizes) == 103
    assert min(report.group_sizes) > 0
    assert max(report.group_sizes) - min(report.group_sizes) <= 1
    assert len(report.group_loads) == 17
    assert report.calibration_matrix_count == 2
    assert report.calibration_rows == (27, 19)
    assert report.signature_width == 14
    assert report.full_target_rank is True
    assert report.columns_normalized is True
    assert report.deterministic is True
    assert report.dense_materialized is False
    assert 0.0 <= report.sketch_retained_energy_fraction <= 1.0
    assert 0.0 <= report.sketch_relative_reconstruction_error <= 1.0
    assert 0.0 <= report.exact_retained_energy_fraction <= 1.0
    assert 0.0 <= report.exact_relative_reconstruction_error <= 1.0
    assert len(report.exact_matrix_input_energies) == 2
    assert len(report.exact_matrix_projected_energies) == 2
    assert len(report.exact_matrix_retained_energy_fractions) == 2
    assert len(report.exact_matrix_relative_reconstruction_errors) == 2
    assert report.as_dict()["memory_bounds"][
        "caller_calibration_storage_excluded"
    ] is True
    assert report.as_dict()["exact_projection"][
        "retained_energy_fraction"
    ] == report.exact_retained_energy_fraction


def test_build_is_bitwise_deterministic_for_coupled_inputs() -> None:
    matrices = [_rows(23, 71, 30), _rows(13, 71, 31)]

    first = build_feature_fold_map(
        matrices,
        9,
        chunk_size=8,
        sketch_size=6,
    )
    second = build_feature_fold_map(
        [matrix.clone() for matrix in matrices],
        9,
        chunk_size=8,
        sketch_size=6,
    )

    assert torch.equal(first.assignment, second.assignment)
    assert torch.equal(first.coefficients, second.coefficients)
    assert first.report == second.report


def test_ten_thousand_source_channels_stay_sparse_and_chunk_bounded() -> None:
    source_width = 10_013
    target_width = 256
    chunk_size = 97
    calibration = _rows(9, source_width, 40)

    fold = build_feature_fold_map(
        calibration,
        target_width,
        chunk_size=chunk_size,
        sketch_size=5,
    )
    report = fold.report

    assert report is not None
    assert fold.stored_elements == 2 * source_width
    assert report.map_stored_elements == 2 * source_width
    assert report.dense_map_elements_avoided == source_width * target_width
    assert report.peak_similarity_chunk_elements == chunk_size * target_width
    assert report.peak_signature_chunk_elements == chunk_size * 5
    assert report.peak_similarity_chunk_elements < source_width * target_width
    assert sum(report.group_sizes) == source_width
    assert max(report.group_sizes) - min(report.group_sizes) <= 1
    assert report.dropped_channels == 0
    with pytest.raises(FeatureFoldMemoryError, match="exceeding"):
        fold.materialize_dense()


def test_apply_supports_row_major_batches_and_finite_autograd() -> None:
    fold = build_feature_fold_map(_rows(18, 12, 50), 5, chunk_size=4)
    inputs = _rows(24, 12, 51).reshape(2, 3, 4, 12).requires_grad_()

    output = fold.apply(inputs)
    loss = output.square().mean() + fold.decode(output).abs().mean()
    loss.backward()

    assert output.shape == (2, 3, 4, 5)
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())
    assert bool((inputs.grad != 0.0).any())
    assert fold.assignment.requires_grad is False
    assert fold.coefficients.requires_grad is False


def test_decode_is_pseudoinverse_projection() -> None:
    fold = build_feature_fold_map(_rows(25, 11, 60), 4)
    source = _rows(7, 11, 61)
    target = _rows(7, 4, 62)

    torch.testing.assert_close(
        fold.encode(fold.decode(target)),
        target,
        atol=1e-12,
        rtol=1e-12,
    )
    projected = fold.decode(fold.encode(source))
    torch.testing.assert_close(
        fold.decode(fold.encode(projected)),
        projected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_zero_energy_channels_are_merged_with_nonzero_finite_coefficients() -> None:
    calibration = torch.zeros((8, 29), dtype=torch.float64)
    fold = build_feature_fold_map(calibration, 6, chunk_size=7)

    assert bool(torch.isfinite(fold.coefficients).all())
    assert bool((fold.coefficients != 0.0).all())
    assert fold.report is not None
    assert fold.report.dropped_channels == 0
    assert fold.report.sketch_retained_energy_fraction == 1.0
    assert fold.report.sketch_relative_reconstruction_error == 0.0
    assert fold.report.exact_retained_energy_fraction == 1.0
    assert fold.report.exact_relative_reconstruction_error == 0.0


def test_countsketch_hashes_do_not_force_periodic_same_sign_collisions() -> None:
    sketch_size = 64
    buckets, signs = feature_fold._deterministic_countsketch_hashes(
        torch,
        4 * sketch_size,
        sketch_size,
        0,
    )
    earlier = slice(0, 3 * sketch_size)
    later = slice(sketch_size, 4 * sketch_size)

    # The previous bucket/sign arithmetic made both tensors exactly periodic:
    # every row r and r + sketch_size had the same bucket and the same sign.
    assert not torch.equal(buckets[earlier], buckets[later])
    assert not torch.equal(signs[earlier], signs[later])
    same_bucket_and_sign = (
        (buckets[earlier] == buckets[later])
        & (signs[earlier] == signs[later])
    )
    assert int(same_bucket_and_sign.sum().item()) < sketch_size


def test_exact_projection_metrics_equal_explicit_dense_calculation() -> None:
    matrices = [_rows(29, 13, 70), _rows(17, 13, 71)]
    fold = build_feature_fold_map(
        matrices,
        6,
        chunk_size=4,
        sketch_size=9,
    )
    dense = fold.materialize_dense(max_elements=1_000)
    input_energies = tuple(
        float(matrix.square().sum().item()) for matrix in matrices
    )
    projected_energies = tuple(
        float((matrix @ dense).square().sum().item()) for matrix in matrices
    )
    total_input = sum(input_energies)
    total_projected = sum(projected_energies)
    retained = total_projected / total_input
    error = math.sqrt(1.0 - retained)
    report = fold.report

    assert report is not None
    assert report.exact_input_energy == pytest.approx(total_input, abs=1e-10)
    assert report.exact_projected_energy == pytest.approx(
        total_projected, abs=1e-10
    )
    assert report.exact_retained_energy_fraction == pytest.approx(
        retained, abs=1e-12
    )
    assert report.exact_relative_reconstruction_error == pytest.approx(
        error, abs=1e-12
    )
    assert report.exact_matrix_input_energies == pytest.approx(
        input_energies, abs=1e-10
    )
    assert report.exact_matrix_projected_energies == pytest.approx(
        projected_energies, abs=1e-10
    )
    for index, input_energy in enumerate(input_energies):
        matrix_retained = projected_energies[index] / input_energy
        assert report.exact_matrix_retained_energy_fractions[
            index
        ] == pytest.approx(matrix_retained, abs=1e-12)
        assert report.exact_matrix_relative_reconstruction_errors[
            index
        ] == pytest.approx(math.sqrt(1.0 - matrix_retained), abs=1e-12)


def test_sign_allowed_groups_use_exact_leading_eigenvectors() -> None:
    matrices = [_rows(23, 12, 80), _rows(19, 12, 81)]
    fold = build_feature_fold_map(
        matrices,
        5,
        chunk_size=4,
        sketch_size=7,
        allow_sign_flip=True,
    )

    for target in range(fold.target_width):
        indices = torch.nonzero(
            fold.assignment == target, as_tuple=False
        ).reshape(-1)
        group_rows = torch.cat(
            tuple(matrix.index_select(1, indices) for matrix in matrices),
            dim=0,
        )
        gram = group_rows.T @ group_rows
        eigenvalues = torch.linalg.eigvalsh(gram)
        coefficient = fold.coefficients.index_select(0, indices)
        rayleigh = coefficient @ gram @ coefficient
        torch.testing.assert_close(
            rayleigh,
            eigenvalues[-1],
            atol=1e-10,
            rtol=1e-10,
        )
        pivot = int(torch.argmax(coefficient.abs()).item())
        assert float(coefficient[pivot].item()) > 0.0


def test_nonlinear_fold_can_prohibit_sign_flips() -> None:
    calibration = _rows(64, 12, 61)
    calibration[:, 6:] = -calibration[:, :6]

    fold = build_feature_fold_map(
        calibration,
        6,
        allow_sign_flip=False,
    )

    assert bool((fold.coefficients > 0.0).all())
    assert fold.report.sign_flips_allowed is False
    assert fold.report.negative_coefficient_count == 0


def test_shape_rank_and_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(FeatureFoldShapeError, match="same source width"):
        build_feature_fold_map(
            [torch.ones(3, 4), torch.ones(2, 5)],
            2,
        )
    with pytest.raises(FeatureFoldShapeError, match="cannot exceed"):
        build_feature_fold_map(torch.ones(3, 4), 5)

    nonfinite = torch.ones(3, 4)
    nonfinite[0, 1] = torch.nan
    with pytest.raises(FeatureFoldFiniteError, match="NaN or infinity"):
        build_feature_fold_map(nonfinite, 2)

    with pytest.raises(FeatureFoldRankError, match="nonempty"):
        FeatureFoldMap(
            torch.tensor([0, 0, 1]),
            torch.ones(3),
            3,
        )
    with pytest.raises(FeatureFoldRankError, match="nonzero coefficient"):
        FeatureFoldMap(
            torch.tensor([0, 1]),
            torch.tensor([1.0, 0.0]),
            2,
        )

    fold = build_feature_fold_map(torch.ones(3, 4), 2)
    bad_runtime = torch.ones(2, 4)
    bad_runtime[0, 0] = torch.inf
    with pytest.raises(FeatureFoldFiniteError, match="NaN or infinity"):
        fold.apply(bad_runtime)
    with pytest.raises(FeatureFoldShapeError, match="last dimension"):
        fold.decode(torch.ones(2, 3))
