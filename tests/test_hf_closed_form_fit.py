from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_closed_form_fit import (  # noqa: E402
    ClosedFormFitError,
    fit_diagonal_norm,
    fit_prior_preserving_ridge_linear,
    fit_ridge_linear,
    gate_fit,
    normalize_rows,
)


def _rows(rows: int, width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, width, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize("fit_bias", [False, True])
def test_ridge_recovers_known_linear_map(fit_bias: bool) -> None:
    inputs = _rows(128, 6, 1)
    weight = _rows(4, 6, 2)
    bias = _rows(1, 4, 3).squeeze(0) if fit_bias else torch.zeros(4)
    targets = inputs @ weight.T + bias

    result = fit_ridge_linear(inputs, targets, ridge=0.0, fit_bias=fit_bias)

    torch.testing.assert_close(result.weight, weight, atol=1e-10, rtol=1e-10)
    if fit_bias:
        torch.testing.assert_close(result.bias, bias, atol=1e-10, rtol=1e-10)
    else:
        assert result.bias is None
    assert result.train_metrics.relative_rmse < 1e-10
    assert result.train_metrics.mean_cosine_similarity > 1.0 - 1e-12


def test_ridge_reports_independent_validation_metrics() -> None:
    train = _rows(96, 5, 10)
    validation = _rows(32, 5, 11)
    weight = _rows(3, 5, 12)
    bias = _rows(1, 3, 13).squeeze(0)

    result = fit_ridge_linear(
        train,
        train @ weight.T + bias,
        ridge=1e-8,
        validation_input_rows=validation,
        validation_target_rows=validation @ weight.T + bias,
    )

    assert result.validation_metrics is not None
    assert result.validation_metrics.rows == 32
    assert result.validation_metrics.relative_rmse < 1e-8


def test_prior_preserving_ridge_is_scale_invariant_and_bounded() -> None:
    inputs = _rows(128, 6, 14)
    targets = _rows(128, 4, 15)
    prior_weight = _rows(4, 6, 16)

    first = fit_prior_preserving_ridge_linear(
        inputs,
        targets,
        prior_weight,
        ridge=0.1,
        fit_bias=False,
    )
    scaled = fit_prior_preserving_ridge_linear(
        inputs * 7.0,
        targets * 7.0,
        prior_weight,
        ridge=0.1,
        fit_bias=False,
    )

    torch.testing.assert_close(first.weight, scaled.weight, atol=1e-10, rtol=1e-10)
    assert first.bias is None and scaled.bias is None
    assert first.effective_ridge > 0.0
    assert first.trace_scale > 0.0
    assert first.prior_delta_norm > 0.0
    assert first.train_metrics.finite is True


def test_strong_prior_ridge_stays_closer_to_prior_than_unregularized_fit() -> None:
    inputs = _rows(96, 5, 18)
    targets = _rows(96, 3, 19)
    prior = _rows(3, 5, 20)

    weak = fit_prior_preserving_ridge_linear(
        inputs,
        targets,
        prior,
        ridge=0.0,
        fit_bias=False,
    )
    strong = fit_prior_preserving_ridge_linear(
        inputs,
        targets,
        prior,
        ridge=100.0,
        fit_bias=False,
    )

    assert strong.prior_delta_norm < weak.prior_delta_norm
    assert torch.isfinite(strong.weight).all()


def test_prior_trust_radius_caps_collapsed_feature_update() -> None:
    inputs = _rows(128, 6, 21) * 1e-6
    targets = _rows(128, 4, 22)
    prior = _rows(4, 6, 23)

    result = fit_prior_preserving_ridge_linear(
        inputs,
        targets,
        prior,
        ridge=1e-3,
        fit_bias=False,
        max_prior_delta_ratio=0.25,
    )

    assert result.clipped is True
    assert 0.0 < result.update_scale < 1.0
    assert result.prior_norm > 0.0
    assert result.prior_delta_norm <= result.prior_norm * 0.25 * (1.0 + 1e-12)
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.weight - prior),
        torch.tensor(result.prior_delta_norm, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_wide_ridge_dual_matches_primal_solution() -> None:
    inputs = _rows(9, 24, 24)
    targets = _rows(9, 5, 25)
    ridge = 0.125

    result = fit_ridge_linear(
        inputs,
        targets,
        ridge=ridge,
        fit_bias=False,
    )
    expected = torch.linalg.solve(
        inputs.T @ inputs
        + ridge * torch.eye(inputs.shape[1], dtype=torch.float64),
        inputs.T @ targets,
    ).T

    torch.testing.assert_close(result.weight, expected, atol=1e-10, rtol=1e-10)
    assert result.bias is None
    assert result.train_metrics.finite is True


def test_wide_prior_preserving_dual_matches_primal_solution() -> None:
    inputs = _rows(10, 22, 26)
    targets = _rows(10, 4, 27)
    prior = _rows(4, 22, 28)
    ridge = 0.3

    result = fit_prior_preserving_ridge_linear(
        inputs,
        targets,
        prior,
        ridge=ridge,
        fit_bias=False,
    )
    trace_scale = float((inputs.T @ inputs).diagonal().sum()) / inputs.shape[1]
    effective_ridge = ridge * trace_scale
    expected = torch.linalg.solve(
        inputs.T @ inputs
        + effective_ridge * torch.eye(inputs.shape[1], dtype=torch.float64),
        inputs.T @ targets + effective_ridge * prior.T,
    ).T

    torch.testing.assert_close(result.weight, expected, atol=1e-10, rtol=1e-10)
    assert result.bias is None
    assert result.effective_ridge == pytest.approx(effective_ridge)


def test_layer_norm_affine_fit_recovers_weight_and_bias() -> None:
    inputs = _rows(128, 7, 20)
    normalized = normalize_rows(inputs, kind="layer_norm", epsilon=1e-5)
    weight = torch.linspace(0.5, 1.5, 7, dtype=torch.float64)
    bias = torch.linspace(-0.2, 0.2, 7, dtype=torch.float64)
    targets = normalized * weight + bias

    result = fit_diagonal_norm(
        inputs,
        targets,
        kind="layer_norm",
        epsilon=1e-5,
        has_bias=True,
        ridge=0.0,
    )

    torch.testing.assert_close(result.weight, weight, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result.bias, bias, atol=1e-10, rtol=1e-10)
    assert result.train_metrics.relative_rmse < 1e-10


def test_rms_norm_affine_fit_recovers_weight_without_bias() -> None:
    inputs = _rows(128, 7, 30)
    normalized = normalize_rows(inputs, kind="rms_norm", epsilon=1e-6)
    weight = torch.linspace(0.5, 1.5, 7, dtype=torch.float64)
    targets = normalized * weight

    result = fit_diagonal_norm(
        inputs,
        targets,
        kind="rms_norm",
        epsilon=1e-6,
        has_bias=False,
        ridge=0.0,
    )

    torch.testing.assert_close(result.weight, weight, atol=1e-10, rtol=1e-10)
    assert result.bias is None
    assert result.train_metrics.relative_rmse < 1e-10


def test_fit_gate_reports_both_failure_signals() -> None:
    inputs = _rows(32, 4, 40)
    targets = _rows(32, 3, 41)
    result = fit_ridge_linear(inputs, targets, ridge=10.0)

    gate = gate_fit(
        result.train_metrics,
        max_relative_rmse=0.01,
        min_mean_cosine_similarity=0.99,
    )

    assert gate.accepted is False
    assert len(gate.reasons) == 2


def test_malformed_nonfinite_and_invalid_norm_cases_fail_closed() -> None:
    with pytest.raises(ClosedFormFitError, match="same row count"):
        fit_ridge_linear(_rows(3, 2, 50), _rows(4, 2, 51))
    bad = _rows(3, 2, 52)
    bad[0, 0] = torch.nan
    with pytest.raises(ClosedFormFitError, match="NaN or infinity"):
        fit_ridge_linear(bad, _rows(3, 2, 53))
    with pytest.raises(ClosedFormFitError, match="RMSNorm does not support"):
        fit_diagonal_norm(
            _rows(4, 2, 54),
            _rows(4, 2, 55),
            kind="rms_norm",
            epsilon=1e-6,
            has_bias=True,
        )
    with pytest.raises(ClosedFormFitError, match="greater than zero"):
        normalize_rows(_rows(4, 2, 56), kind="layer_norm", epsilon=0.0)


def test_all_fit_outputs_are_detached_and_inputs_receive_no_gradients() -> None:
    inputs = _rows(32, 4, 60).requires_grad_()
    targets = _rows(32, 3, 61).requires_grad_()
    result = fit_ridge_linear(inputs, targets)
    norm_targets = _rows(32, 4, 62).requires_grad_()
    norm = fit_diagonal_norm(
        inputs,
        norm_targets,
        kind="rms_norm",
        epsilon=1e-6,
        has_bias=False,
    )

    outputs = (result.weight, result.bias, norm.weight)
    assert all(
        output is not None and output.requires_grad is False for output in outputs
    )
    assert all(output.grad_fn is None for output in outputs if output is not None)
    assert inputs.grad is None
    assert targets.grad is None
    assert norm_targets.grad is None
