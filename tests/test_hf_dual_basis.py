from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_dual_basis import (  # noqa: E402
    build_dual_aware_residual_basis,
    evaluate_dual_basis_oracle,
)
from neural_forge.hf_transport_math import (  # noqa: E402
    CoordinateMap,
    FiniteValidationError,
    RankValidationError,
    ShapeValidationError,
)


def _rows(row_count: int, width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        row_count,
        width,
        generator=generator,
        dtype=torch.float64,
    )


def _basis_inputs(width: int = 6):
    return (
        _rows(30, width, 101),
        _rows(19, width, 102),
        _rows(13, width, 103),
    )


def test_dual_basis_is_cpu_float64_orthonormal_and_fully_reported() -> None:
    residual, embeddings, margins = _basis_inputs()
    margin_weights = torch.linspace(0.25, 2.0, margins.shape[0])

    coordinates, report = build_dual_aware_residual_basis(
        residual.float().requires_grad_(),
        embeddings.float().requires_grad_(),
        margins.float().requires_grad_(),
        4,
        margin_weights=margin_weights.float(),
        alpha=2.0,
        beta=3.0,
        gamma=5.0,
    )

    assert isinstance(coordinates, CoordinateMap)
    assert coordinates.decode.device.type == "cpu"
    assert coordinates.decode.dtype == torch.float64
    assert coordinates.decode.requires_grad is False
    torch.testing.assert_close(
        coordinates.decode.T @ coordinates.decode,
        torch.eye(4, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        coordinates.source_projector,
        coordinates.decode @ coordinates.decode.T,
        atol=1e-12,
        rtol=1e-12,
    )
    assert report["covariance_weights"] == {
        "alpha": 2.0,
        "beta": 3.0,
        "gamma": 5.0,
    }
    assert report["row_counts"] == {
        "residual": 30,
        "embedding": 19,
        "margin": 13,
    }
    assert report["margin_weights"]["provided"] is True
    assert report["combined_covariance_trace"] == pytest.approx(10.0)
    assert all(
        value == pytest.approx(1.0)
        for value in report["normalized_covariance_traces"].values()
    )
    assert report["orthonormality_max_abs_error"] < 1e-12
    assert report["gradient_free"] is True
    assert report["optimizer_used"] is False


def test_dual_basis_is_bitwise_deterministic_for_degenerate_spectrum() -> None:
    identity = torch.eye(4, dtype=torch.float64)
    symmetric_axes = torch.cat((identity, -identity), dim=0)

    first, first_report = build_dual_aware_residual_basis(
        symmetric_axes,
        symmetric_axes,
        symmetric_axes,
        3,
    )
    second, second_report = build_dual_aware_residual_basis(
        symmetric_axes.clone(),
        symmetric_axes.clone(),
        symmetric_axes.clone(),
        3,
    )

    assert torch.equal(first.decode, second.decode)
    assert torch.equal(first.encode, second.encode)
    assert torch.equal(first.source_projector, second.source_projector)
    assert first_report == second_report
    assert first_report["degenerate_selected_eigenspace_count"] == 1
    torch.testing.assert_close(
        first.decode,
        torch.eye(4, dtype=torch.float64)[:, :3],
        atol=1e-12,
        rtol=0.0,
    )
    pivots = torch.argmax(torch.abs(first.decode), dim=0)
    columns = torch.arange(first.target_dim)
    assert bool((first.decode[pivots, columns] > 0.0).all())


def test_gamma_and_per_margin_weights_move_the_selected_direction() -> None:
    e0, e1, e2 = torch.eye(3, dtype=torch.float64)
    residual = torch.stack((e0, -e0, 2.0 * e0, -2.0 * e0))
    embeddings = torch.stack((e0, -e0))
    margins = torch.stack((e1, e2))

    primal, _ = build_dual_aware_residual_basis(
        residual,
        embeddings,
        margins,
        1,
        margin_weights=torch.tensor([100.0, 1.0]),
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
    )
    margin_first, _ = build_dual_aware_residual_basis(
        residual,
        embeddings,
        margins,
        1,
        margin_weights=torch.tensor([100.0, 1.0]),
        alpha=0.1,
        beta=0.1,
        gamma=10.0,
    )
    margin_second, _ = build_dual_aware_residual_basis(
        residual,
        embeddings,
        margins,
        1,
        margin_weights=torch.tensor([1.0, 100.0]),
        alpha=0.1,
        beta=0.1,
        gamma=10.0,
    )

    assert torch.argmax(torch.abs(primal.decode[:, 0])).item() == 0
    assert torch.argmax(torch.abs(margin_first.decode[:, 0])).item() == 1
    assert torch.argmax(torch.abs(margin_second.decode[:, 0])).item() == 2


def test_no_model_oracle_is_exact_for_a_full_coordinate_basis() -> None:
    residual, embeddings, margins = _basis_inputs(width=4)
    coordinates, _ = build_dual_aware_residual_basis(
        residual,
        embeddings,
        margins,
        4,
    )
    hidden = _rows(11, 4, 201)
    head = _rows(9, 4, 202)
    bias = _rows(1, 9, 203).squeeze(0)
    teacher_logits = hidden @ head.T + bias

    report = evaluate_dual_basis_oracle(
        coordinates,
        hidden.requires_grad_(),
        head.requires_grad_(),
        teacher_logits.requires_grad_(),
        head_bias=bias,
    )

    assert report["top1_agreement"] == 1.0
    assert report["top5_agreement"] == 1.0
    assert report["top5_exact_agreement"] == 1.0
    assert report["teacher_top1_in_candidate_top5"] == 1.0
    assert report["teacher_margin_sign_retention"] == 1.0
    assert report["kl"] == pytest.approx(0.0, abs=1e-14)
    assert report["coordinate_logit_error"] == pytest.approx(0.0, abs=1e-14)
    assert report["gradient_free"] is True
    assert report["optimizer_used"] is False


def test_oracle_aligns_full_or_preselected_vocabulary_rows() -> None:
    hidden = _rows(7, 3, 301)
    head = _rows(10, 3, 302)
    teacher = hidden @ head.T
    basis = torch.eye(3, dtype=torch.float64)[:, :2]
    indices = torch.tensor([1, 3, 4, 7, 9])

    from_full = evaluate_dual_basis_oracle(
        basis,
        hidden,
        head,
        teacher,
        vocab_indices=indices,
    )
    from_selected = evaluate_dual_basis_oracle(
        basis,
        hidden,
        head.index_select(0, indices),
        teacher,
        selected_vocab_indices=indices,
    )

    assert from_full == from_selected
    assert from_full["vocabulary_scope"] == "selected"
    assert from_full["vocab_row_count"] == 5
    assert from_full["top5_effective_k"] == 5


def test_builder_and_oracle_validation_fail_closed() -> None:
    residual, embeddings, margins = _basis_inputs(width=3)
    bad_residual = residual.clone()
    bad_residual[0, 0] = torch.nan
    with pytest.raises(FiniteValidationError, match="NaN or infinity"):
        build_dual_aware_residual_basis(
            bad_residual,
            embeddings,
            margins,
            2,
        )
    with pytest.raises(ShapeValidationError, match="width"):
        build_dual_aware_residual_basis(
            residual,
            _rows(4, 4, 401),
            margins,
            2,
        )
    with pytest.raises(ShapeValidationError, match="At least one"):
        build_dual_aware_residual_basis(
            residual,
            embeddings,
            margins,
            2,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
        )
    with pytest.raises(ShapeValidationError, match="negatives"):
        build_dual_aware_residual_basis(
            residual,
            embeddings,
            margins,
            2,
            margin_weights=torch.tensor([-1.0] + [1.0] * 12),
        )

    one_axis = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    with pytest.raises(RankValidationError, match="effective rank"):
        build_dual_aware_residual_basis(one_axis, one_axis, one_axis, 2)

    hidden = _rows(4, 3, 402)
    head = _rows(6, 3, 403)
    teacher = hidden @ head.T
    with pytest.raises(RankValidationError, match="orthonormal"):
        evaluate_dual_basis_oracle(
            torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
            hidden,
            head,
            teacher,
        )
    with pytest.raises(ShapeValidationError, match="unique"):
        evaluate_dual_basis_oracle(
            torch.eye(3, dtype=torch.float64),
            hidden,
            head,
            teacher,
            vocab_indices=[1, 1],
        )
