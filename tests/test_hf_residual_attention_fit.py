from __future__ import annotations

import numpy as np
import pytest

from neural_forge.hf_residual_attention_fit import (
    ResidualAttentionFitError,
    ResidualAttentionFitLimits,
    attention_weighted_value_error,
    fit_attention_output_ridge,
    fit_fixed_attention_value_ridge,
    fit_residual_branch_scale,
    select_score_space_qk_medoid,
    softmax_fisher_score_error,
)


def test_score_space_error_ignores_softmax_constant_shift_but_detects_route_flip() -> None:
    reference = np.array([[3.0, 1.0, -2.0], [0.5, 0.0, -0.5]])
    shifted = reference + np.array([[20.0], [-7.0]])
    flipped = reference[:, ::-1]

    invisible = softmax_fisher_score_error(reference, shifted)
    visible = softmax_fisher_score_error(reference, flipped)

    assert invisible.mean == pytest.approx(0.0, abs=1e-12)
    assert invisible.worst == pytest.approx(0.0, abs=1e-12)
    assert visible.mean > 0.0
    assert visible.worst > 0.0
    assert "modulo-constant" in invisible.method


def test_value_error_is_measured_after_attention_readout() -> None:
    reference = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    # Large token-space errors cancel under equal attention.
    folded = reference + np.array([[[8.0, -3.0], [-8.0, 3.0]]])
    equal_attention = np.array([[[0.5, 0.5]]])
    focused_attention = np.array([[[1.0, 0.0]]])

    cancelled = attention_weighted_value_error(
        reference, folded, equal_attention
    )
    visible = attention_weighted_value_error(
        reference, folded, focused_attention
    )

    assert cancelled.worst == pytest.approx(0.0, abs=1e-12)
    assert visible.worst > 1.0
    assert cancelled.method == "fixed-attention-value-readout-error"


def test_residual_branch_scale_recovers_delta_magnitude_without_gradient() -> None:
    target = np.array([[2.0, -4.0], [6.0, -8.0]])
    candidate = target / 2.0

    fit = fit_residual_branch_scale(
        candidate,
        target,
        ridge_to_one=0.0,
        maximum_scale=3.0,
    )

    assert fit.scale == pytest.approx(2.0)
    assert fit.after_relative_rmse == pytest.approx(0.0)
    assert fit.after_relative_rmse < fit.before_relative_rmse
    assert fit.zero_gradient
    assert fit.as_dict()["training_used"] is False


def test_attention_output_ridge_recovers_residual_writer_and_is_deterministic() -> None:
    rng = np.random.default_rng(17)
    context = rng.normal(size=(64, 3))
    expected = np.array([[1.5, -0.25, 0.75], [-0.5, 2.0, 0.125]])
    target = context @ expected.T
    anchor = np.zeros_like(expected)

    first = fit_attention_output_ridge(
        context,
        target,
        anchor,
        ridge=1e-10,
    )
    second = fit_attention_output_ridge(
        context,
        target,
        anchor,
        ridge=1e-10,
    )

    np.testing.assert_allclose(first.weight, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert not first.weight.flags.writeable
    assert first.report.after_relative_rmse < 1e-10
    assert first.report.after_relative_rmse < first.report.before_relative_rmse
    assert first.report.fitted_sha256 == second.report.fitted_sha256
    assert first.report.zero_gradient


def test_weighted_value_ridge_uses_same_bounded_closed_form_contract() -> None:
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = np.array([[2.0, -1.0]])
    target = design @ expected.T
    anchor = np.zeros_like(expected)

    fit = fit_fixed_attention_value_ridge(
        design,
        target,
        anchor,
        ridge=1e-8,
        row_weights=np.array([1.0, 3.0, 2.0]),
    )

    np.testing.assert_allclose(fit.weight, expected, atol=1e-7, rtol=1e-7)
    assert fit.report.weighted
    assert fit.report.method == "fixed-qk-attention-readout-value-ridge"


def test_score_space_medoid_uses_real_candidate_and_lowest_index_ties() -> None:
    base = np.array([[2.0, 0.0, -1.0], [1.0, 0.0, -0.5]])
    traces = np.stack(
        (
            base,
            base + 100.0,  # Same softmax geometry as candidate zero.
            base[:, ::-1],
        )
    )

    medoid = select_score_space_qk_medoid(traces)

    assert medoid.medoid_index == 0
    assert medoid.pairwise_costs[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert medoid.pairwise_costs[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert medoid.total_costs[0] == pytest.approx(medoid.total_costs[1])
    assert not medoid.pairwise_costs.flags.writeable
    assert medoid.as_dict()["real_medoid"] is True


def test_ridge_rejects_shape_and_workspace_lies() -> None:
    context = np.ones((4, 3))
    target = np.ones((4, 2))
    anchor = np.ones((2, 3))

    with pytest.raises(ResidualAttentionFitError, match="anchor_weight"):
        fit_attention_output_ridge(context, target, np.ones((3, 2)))
    with pytest.raises(ResidualAttentionFitError, match="must be positive"):
        fit_attention_output_ridge(context, target, anchor, ridge=0.0)
    with pytest.raises(ResidualAttentionFitError, match="bounds"):
        fit_attention_output_ridge(
            context,
            target,
            anchor,
            limits=ResidualAttentionFitLimits(max_rows=3),
        )
    with pytest.raises(ResidualAttentionFitError, match="row_weights"):
        fit_attention_output_ridge(
            context,
            target,
            anchor,
            row_weights=np.zeros(4),
        )


def test_torch_evidence_is_copied_without_autograd() -> None:
    torch = pytest.importorskip("torch")
    context = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor([[2.0], [-1.0]], dtype=torch.float64)
    anchor = torch.zeros((1, 2), dtype=torch.float64)

    fit = fit_attention_output_ridge(context, target, anchor, ridge=1e-8)

    assert context.grad is None
    assert fit.report.optimizer_steps == 0
    assert fit.report.backward_calls == 0
