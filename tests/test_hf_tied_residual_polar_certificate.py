from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_tied_residual_congruence import (  # noqa: E402
    TiedResidualCongruenceError,
    _source_bound_polar_certificate,
)


def _polar_factor(value):
    left, _singular, right_h = torch.linalg.svd(value, full_matrices=False)
    return (left @ right_h).contiguous()


def test_independent_resvd_last_bit_drift_still_proves_the_polar_relation():
    generator = torch.Generator().manual_seed(7)
    encoder = torch.randn((128, 96), generator=generator, dtype=torch.float64)
    original_threads = int(torch.get_num_threads())
    try:
        torch.set_num_threads(1)
        first_basis = _polar_factor(encoder)
        torch.set_num_threads(max(2, original_threads))
        second_basis = _polar_factor(encoder)
    finally:
        torch.set_num_threads(original_threads)

    first, _first_stretch = _source_bound_polar_certificate(
        first_basis,
        encoder,
    )
    second, _second_stretch = _source_bound_polar_certificate(
        second_basis,
        encoder,
    )

    assert first["source_bound_unique_full_column_polar_factor_authenticated"] is True
    assert second["source_bound_unique_full_column_polar_factor_authenticated"] is True
    assert first["fresh_svd_reexecution_required"] is False
    assert second["fresh_svd_reexecution_required"] is False
    # MKL normally takes a different last-bit path after this thread transition.
    # The assertion is deliberately about mathematical acceptance, not requiring
    # a particular BLAS implementation to exhibit the drift.
    assert float((first_basis - second_basis).abs().max().item()) < 1e-12


def test_nontrivial_orthonormal_rotation_cannot_forge_the_polar_basis():
    generator = torch.Generator().manual_seed(31)
    encoder = torch.randn((32, 8), generator=generator, dtype=torch.float64)
    basis = _polar_factor(encoder)
    rotation = torch.eye(8, dtype=torch.float64)
    angle = 0.01
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation[0, 0] = cosine
    rotation[0, 1] = -sine
    rotation[1, 0] = sine
    rotation[1, 1] = cosine
    forged_basis = (basis @ rotation).contiguous()

    torch.testing.assert_close(
        forged_basis.T @ forged_basis,
        torch.eye(8, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(
        TiedResidualCongruenceError,
        match="source-bound full-column polar factor",
    ):
        _source_bound_polar_certificate(forged_basis, encoder)


def test_symmetric_indefinite_right_stretch_is_not_a_polar_certificate():
    generator = torch.Generator().manual_seed(47)
    raw = torch.randn((16, 4), generator=generator, dtype=torch.float64)
    basis, _remainder = torch.linalg.qr(raw, mode="reduced")
    indefinite = torch.diag(
        torch.tensor((3.0, 2.0, 1.0, -0.5), dtype=torch.float64)
    )
    encoder = (basis @ indefinite).contiguous()

    with pytest.raises(
        TiedResidualCongruenceError,
        match="source-bound full-column polar factor",
    ):
        _source_bound_polar_certificate(basis, encoder)


def test_numerically_rank_deficient_right_stretch_is_rejected():
    generator = torch.Generator().manual_seed(61)
    raw = torch.randn((24, 8), generator=generator, dtype=torch.float64)
    basis, _remainder = torch.linalg.qr(raw, mode="reduced")
    nearly_singular = torch.diag(
        torch.tensor(
            (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 1e-16),
            dtype=torch.float64,
        )
    )
    encoder = (basis @ nearly_singular).contiguous()

    with pytest.raises(
        TiedResidualCongruenceError,
        match="source-bound full-column polar factor",
    ):
        _source_bound_polar_certificate(basis, encoder)
