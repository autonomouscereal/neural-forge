"""Closed-form solver-math extensions for the LFM2 folding compiler.

Three additive mitigations are covered here, all fold-only and gradient-free:

* an errors-in-variables Tikhonov term that makes a fitted layer robust to
  *additional* upstream drift the child will see at generation time,
* externally supplied per-row fit weights derived from cached vocabulary
  decision margins, and
* gate pre-activation operating-point statistics plus an opt-in two-moment
  rescale of a fitted gate projection.

Every default path is asserted to reproduce the pre-existing formula exactly,
because five other lanes compile against these modules concurrently.
"""

from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_structural_residual_attention_fold as fold_module  # noqa: E402
from neural_forge.hf_lfm2_balanced_circuit_basis import (  # noqa: E402
    lfm2_vocabulary_margin_vulnerability,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedMlpFoldCase,
    Lfm2StructuralResidualAttentionFoldError,
    compile_lfm2_real_medoid_structural_residual_mlp_fold,
    compile_lfm2_structural_residual_mlp_fold,
    lfm2_estimate_input_error_covariance,
    lfm2_gate_operating_point_correction,
    lfm2_gate_operating_point_report,
    lfm2_gate_preactivation_statistics,
    lfm2_vocabulary_margin_row_weights,
)


class _ToyMlp(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)


def _legacy_fit_linear(
    input_rows: torch.Tensor,
    target_rows: torch.Tensor,
    *,
    ridge: float,
    prior: torch.Tensor | None,
    max_prior_delta_ratio: float | None,
) -> torch.Tensor:
    """The pre-extension closed form, transcribed here as the reference.

    Keeping an independent copy in the test is the only way to prove the
    default path is bit-identical: comparing the module against itself would
    pass no matter what the extension changed.
    """

    x = input_rows.reshape(-1, int(input_rows.shape[-1]))
    y = target_rows.reshape(-1, int(target_rows.shape[-1]))
    if prior is None:
        prior_rows = torch.zeros(
            (int(y.shape[1]), int(x.shape[1])), dtype=torch.float64
        )
    else:
        prior_rows = prior.to(dtype=torch.float64)
    residual = y if prior is None else y - x @ prior_rows.T
    if ridge == 0.0:
        solution = torch.linalg.lstsq(x, residual, driver="gelsd").solution
    elif int(x.shape[1]) > int(x.shape[0]):
        system = x @ x.T + ridge * torch.eye(int(x.shape[0]), dtype=torch.float64)
        solution = x.T @ torch.linalg.solve(system, residual)
    else:
        system = x.T @ x + ridge * torch.eye(int(x.shape[1]), dtype=torch.float64)
        solution = torch.linalg.solve(system, x.T @ residual)
    correction = solution.T.contiguous()
    if (
        prior is not None
        and max_prior_delta_ratio is not None
        and float(torch.linalg.vector_norm(correction).item()) > 0.0
    ):
        allowed = max_prior_delta_ratio * max(
            float(torch.linalg.vector_norm(prior_rows).item()),
            torch.finfo(torch.float64).tiny,
        )
        update_scale = min(
            1.0, allowed / float(torch.linalg.vector_norm(correction).item())
        )
        if update_scale < 1.0:
            correction = correction * update_scale
    return prior_rows + correction


# ---------------------------------------------------------------------------
# 1. Exact reproduction of the pre-extension closed form.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "width", "outputs", "ridge", "use_prior", "delta_ratio"),
    [
        (12, 5, 3, 0.0, False, None),
        (12, 5, 3, 1e-3, False, None),
        (12, 5, 3, 1e-3, True, None),
        (12, 5, 3, 1e-3, True, 0.25),
        (4, 9, 2, 1e-2, True, 1.0),
        (4, 9, 2, 0.0, True, None),
    ],
)
def test_default_fit_linear_reproduces_the_pre_extension_formula_exactly(
    rows: int,
    width: int,
    outputs: int,
    ridge: float,
    use_prior: bool,
    delta_ratio: float | None,
) -> None:
    generator = torch.Generator().manual_seed(20260731)
    x = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    y = torch.randn(rows, outputs, generator=generator, dtype=torch.float64)
    prior = (
        torch.randn(outputs, width, generator=generator, dtype=torch.float64)
        if use_prior
        else None
    )
    weight, report = fold_module._fit_linear(
        x,
        y,
        role="reproduction",
        ridge=ridge,
        prior=prior,
        max_prior_delta_ratio=delta_ratio,
    )
    expected = _legacy_fit_linear(
        x,
        y,
        ridge=ridge,
        prior=prior,
        max_prior_delta_ratio=delta_ratio,
    )
    assert torch.equal(weight, expected)
    assert report.input_error_covariance_applied is False
    assert report.as_dict() == {
        "role": "reproduction",
        "rows": rows,
        "input_width": width,
        "output_width": outputs,
        "underdetermined": rows < width,
        "prior_provided": use_prior,
        "prior_norm": report.prior_norm,
        "correction_norm": report.correction_norm,
        "update_scale": report.update_scale,
        "objective": "source_derived_prior_plus_closed_form_residual_correction",
    }


def test_zero_scale_covariance_matches_the_plain_primal_ridge_system() -> None:
    generator = torch.Generator().manual_seed(4242)
    x = torch.randn(16, 4, generator=generator, dtype=torch.float64)
    y = torch.randn(16, 2, generator=generator, dtype=torch.float64)
    covariance = torch.eye(4, dtype=torch.float64)
    weight, report = fold_module._fit_linear(
        x,
        y,
        role="zero_scale",
        ridge=1e-3,
        prior=None,
        max_prior_delta_ratio=None,
        input_error_covariance=covariance,
        input_error_scale=0.0,
    )
    expected = _legacy_fit_linear(
        x, y, ridge=1e-3, prior=None, max_prior_delta_ratio=None
    )
    assert torch.allclose(weight, expected, rtol=0.0, atol=1e-12)
    assert report.input_error_covariance_applied is True
    assert report.as_dict()["noise_aware_ridge"]["input_error_scale"] == 0.0


# ---------------------------------------------------------------------------
# 2. Noise-aware ridge: hand-computed system and errors-in-variables recovery.
# ---------------------------------------------------------------------------


def test_noise_aware_normal_equations_match_a_hand_computed_small_case() -> None:
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    y = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    prior = torch.tensor([[0.5, -0.25], [1.0, 0.75]], dtype=torch.float64)
    covariance = torch.tensor([[0.25, 0.1], [0.1, 4.0]], dtype=torch.float64)
    ridge = 0.5
    scale = 2.0

    # (X.T X + ridge*I + scale*Sigma) S = X.T (Y - X P.T) - scale * Sigma P.T
    residual = y - x @ prior.T
    system = (
        x.T @ x + ridge * torch.eye(2, dtype=torch.float64) + scale * covariance
    )
    right = x.T @ residual - scale * (covariance @ prior.T)
    expected = prior + torch.linalg.solve(system, right).T

    weight, report = fold_module._fit_linear(
        x,
        y,
        role="hand_computed",
        ridge=ridge,
        prior=prior,
        max_prior_delta_ratio=None,
        input_error_covariance=covariance,
        input_error_scale=scale,
    )
    assert torch.allclose(weight, expected, rtol=0.0, atol=1e-13)
    noise_aware = report.as_dict()["noise_aware_ridge"]
    assert noise_aware["input_error_scale"] == scale
    assert noise_aware["input_error_covariance_trace"] == pytest.approx(4.25)
    assert noise_aware["penalized_operator"] == "full_prior_plus_correction_weight"


def test_noise_aware_ridge_penalizes_the_full_operator_not_the_correction() -> None:
    # With a prior whose response lives in the noisy direction, the noise-aware
    # solve must move the *total* operator away from that direction.  A term
    # that only shrank the correction could not do this: it would leave the
    # prior's exposure untouched.
    # Input coordinate one is never observed by the fit rows, so the plain
    # solve has no evidence about it and leaves the prior's exposure intact.
    x = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    y = x @ torch.tensor([[1.0, 0.0]], dtype=torch.float64).T
    prior = torch.tensor([[0.0, 3.0]], dtype=torch.float64)
    covariance = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    plain, _ = fold_module._fit_linear(
        x, y, role="plain", ridge=0.0, prior=prior, max_prior_delta_ratio=None
    )
    robust, _ = fold_module._fit_linear(
        x,
        y,
        role="robust",
        ridge=0.0,
        prior=prior,
        max_prior_delta_ratio=None,
        input_error_covariance=covariance,
        input_error_scale=8.0,
    )
    assert float(plain[0, 1]) == pytest.approx(3.0)
    # X.T X has no energy in coordinate one, so the whole system there is
    # 8*Sigma and the right-hand side is -8*Sigma@prior.T: the correction is
    # exactly -3 and the installed operator's exposure collapses to zero.
    assert float(robust[0, 1]) == pytest.approx(0.0, abs=1e-12)
    assert float(robust[0, 0]) == pytest.approx(1.0)
    assert abs(float(robust[0, 1])) < abs(float(plain[0, 1]))


def test_noise_aware_ridge_beats_plain_ridge_under_real_input_drift() -> None:
    generator = torch.Generator().manual_seed(1071)
    rows = 96
    width = 6
    outputs = 3

    # An ill-conditioned design: the last two input coordinates carry little
    # fit energy, which is exactly where a plain ridge is happy to place large
    # weights and where generation-time drift then hurts most.
    design = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    design[:, 4] *= 0.05
    design[:, 5] *= 0.05
    truth = torch.randn(outputs, width, generator=generator, dtype=torch.float64)
    targets = design @ truth.T

    drift_scale = torch.tensor(
        [0.02, 0.02, 0.02, 0.02, 0.6, 0.6], dtype=torch.float64
    )
    paired_clean = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    paired_drift = (
        torch.randn(rows, width, generator=generator, dtype=torch.float64)
        * drift_scale
    )
    covariance, covariance_report = lfm2_estimate_input_error_covariance(
        paired_clean,
        paired_clean + paired_drift,
        name="upstream",
    )
    assert covariance_report["rank"] == width
    assert covariance_report["rank_deficient"] is False

    plain, _ = fold_module._fit_linear(
        design,
        targets,
        role="plain",
        ridge=1e-6,
        prior=None,
        max_prior_delta_ratio=None,
    )
    robust, robust_report = fold_module._fit_linear(
        design,
        targets,
        role="robust",
        ridge=1e-6,
        prior=None,
        max_prior_delta_ratio=None,
        input_error_covariance=covariance,
        input_error_scale=float(rows),
    )
    assert robust_report.input_error_covariance_applied is True

    # Held-out generation-time evaluation: the child receives drifted inputs
    # but is judged against the clean parent action.
    evaluation = torch.randn(512, width, generator=generator, dtype=torch.float64)
    evaluation[:, 4] *= 0.05
    evaluation[:, 5] *= 0.05
    clean_action = evaluation @ truth.T

    def drifted_error(weight: torch.Tensor) -> float:
        total = 0.0
        for _ in range(8):
            noise = (
                torch.randn(
                    evaluation.shape, generator=generator, dtype=torch.float64
                )
                * drift_scale
            )
            predicted = (evaluation + noise) @ weight.T
            total += float((predicted - clean_action).square().mean().item())
        return total / 8.0

    plain_error = drifted_error(plain)
    robust_error = drifted_error(robust)
    assert robust_error < plain_error

    # Teacher-forced error on the fit rows moves the other way.  That is the
    # documented trade and the reason this term is opt-in.
    plain_fit = float((design @ plain.T - targets).square().mean().item())
    robust_fit = float((design @ robust.T - targets).square().mean().item())
    assert plain_fit <= robust_fit


def test_input_error_covariance_estimator_is_exact_and_guards_rank() -> None:
    clean = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    perturbed = clean + torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]], dtype=torch.float64
    )
    covariance, report = lfm2_estimate_input_error_covariance(clean, perturbed)
    expected = torch.tensor([[2.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(covariance, expected, rtol=0.0, atol=1e-13)
    assert torch.equal(covariance, covariance.T)
    assert report["rank"] == 1
    assert report["rank_deficient"] is True
    assert report["centered"] is False
    assert report["trace"] == pytest.approx(2.0)

    centered, centered_report = lfm2_estimate_input_error_covariance(
        clean, perturbed, center=True
    )
    # The uncentered second moment keeps the systematic 2/3 bias; centering
    # drops it, which is why the centered trace is strictly smaller.
    assert centered_report["trace"] < report["trace"]
    assert float(centered[0, 0]) == pytest.approx(2.0 - (2.0 / 3.0) ** 2)

    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        lfm2_estimate_input_error_covariance(clean, perturbed, require_full_rank=True)


def test_noise_aware_ridge_rejects_malformed_covariance_and_lonely_scale() -> None:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    y = torch.randn(8, 2, generator=generator, dtype=torch.float64)
    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        fold_module._fit_linear(
            x,
            y,
            role="lonely_scale",
            ridge=0.0,
            prior=None,
            max_prior_delta_ratio=None,
            input_error_scale=2.0,
        )
    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        fold_module._fit_linear(
            x,
            y,
            role="wrong_width",
            ridge=0.0,
            prior=None,
            max_prior_delta_ratio=None,
            input_error_covariance=torch.eye(4, dtype=torch.float64),
        )
    asymmetric = torch.eye(3, dtype=torch.float64)
    asymmetric[0, 1] = 1.0
    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        fold_module._fit_linear(
            x,
            y,
            role="asymmetric",
            ridge=0.0,
            prior=None,
            max_prior_delta_ratio=None,
            input_error_covariance=asymmetric,
        )
    indefinite = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        fold_module._fit_linear(
            x,
            y,
            role="indefinite",
            ridge=0.0,
            prior=None,
            max_prior_delta_ratio=None,
            input_error_covariance=indefinite,
        )


def test_affine_fit_leaves_the_constant_bias_column_unpenalized() -> None:
    generator = torch.Generator().manual_seed(31337)
    x = torch.randn(20, 3, generator=generator, dtype=torch.float64)
    y = torch.randn(20, 2, generator=generator, dtype=torch.float64)
    covariance = torch.diag(torch.tensor([0.5, 0.25, 2.0], dtype=torch.float64))
    weight, bias, _report = fold_module._fit_affine(
        x,
        y,
        role="affine",
        ridge=1e-4,
        weight_prior=None,
        bias_prior=None,
        max_prior_delta_ratio=None,
        fit_bias=True,
        input_error_covariance=covariance,
        input_error_scale=3.0,
    )
    augmented = torch.cat((x, torch.ones((20, 1), dtype=torch.float64)), dim=1)
    padded = torch.zeros((4, 4), dtype=torch.float64)
    padded[:3, :3] = covariance
    system = (
        augmented.T @ augmented
        + 1e-4 * torch.eye(4, dtype=torch.float64)
        + 3.0 * padded
    )
    expected = torch.linalg.solve(system, augmented.T @ y).T
    assert torch.allclose(weight, expected[:, :-1], rtol=0.0, atol=1e-12)
    assert torch.allclose(bias, expected[:, -1], rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Margin-derived row weighting.
# ---------------------------------------------------------------------------


def test_margin_row_weights_match_the_balanced_basis_vulnerability_formula() -> None:
    margins = torch.tensor(
        [[0.05, 0.40, 1.20], [3.00, 4.00, 5.00], [0.90, 1.10, 2.00]],
        dtype=torch.float64,
    )
    logits = torch.tensor(
        [[6.0, 5.95, 5.6, 4.8], [9.0, 6.0, 5.0, 4.0], [7.0, 6.1, 5.9, 5.0]],
        dtype=torch.float64,
    )
    weights, report = lfm2_vocabulary_margin_row_weights(
        margins,
        logit_values=logits,
        weight_floor=1e-6,
        weight_ceiling=1e6,
    )
    vulnerability, basis_report = lfm2_vocabulary_margin_vulnerability(
        margins, logit_values=logits
    )
    assert report["margin_floor"] == pytest.approx(basis_report["margin_floor"])
    assert report["margin_scale"] == pytest.approx(basis_report["margin_scale"])
    assert report["value_scale"] == pytest.approx(basis_report["value_scale"])

    per_row = vulnerability.sum(dim=1)
    expected = per_row / per_row.mean()
    assert torch.allclose(weights, expected, rtol=0.0, atol=1e-13)
    assert float(weights.mean().item()) == pytest.approx(1.0)
    # The nearly tied first row outweighs the comfortably decided second row.
    assert float(weights[0]) > float(weights[1])


def test_margin_row_weight_clamp_bounds_the_emphasis_ratio() -> None:
    margins = torch.tensor([[1e-9], [1.0], [2.0], [3.0]], dtype=torch.float64)
    unclamped, _ = lfm2_vocabulary_margin_row_weights(
        margins, weight_floor=1e-9, weight_ceiling=1e9
    )
    clamped, report = lfm2_vocabulary_margin_row_weights(
        margins, weight_floor=0.5, weight_ceiling=2.0
    )
    ratio = float(clamped.max().item()) / float(clamped.min().item())
    assert ratio <= 2.0 / 0.5 + 1e-9
    assert ratio < float(unclamped.max().item()) / float(unclamped.min().item())
    assert report["rows_clamped_at_ceiling"] >= 1
    assert float(clamped.mean().item()) == pytest.approx(1.0)


def test_margin_row_weights_move_the_fit_toward_low_margin_rows() -> None:
    # Two groups of rows demand contradictory slopes on the same input.  The
    # unweighted fit splits the difference; weighting by inverse margin has to
    # pull it toward the group whose downstream decision is nearly tied.
    design = torch.ones((4, 1), dtype=torch.float64)
    targets = torch.tensor([[0.0], [0.0], [10.0], [10.0]], dtype=torch.float64)
    margins = torch.tensor([[4.0], [4.0], [0.02], [0.02]], dtype=torch.float64)
    weights, _ = lfm2_vocabulary_margin_row_weights(
        margins, weight_floor=1e-6, weight_ceiling=1e6
    )
    sqrt_weights = torch.sqrt(weights).unsqueeze(1)

    uniform, _ = fold_module._fit_linear(
        design,
        targets,
        role="uniform",
        ridge=0.0,
        prior=None,
        max_prior_delta_ratio=None,
    )
    weighted, _ = fold_module._fit_linear(
        design * sqrt_weights,
        targets * sqrt_weights,
        role="margin_weighted",
        ridge=0.0,
        prior=None,
        max_prior_delta_ratio=None,
    )
    assert float(uniform[0, 0]) == pytest.approx(5.0)
    assert float(weighted[0, 0]) > 9.0


def test_constant_row_profile_reproduces_the_uniform_compile_byte_for_byte() -> None:
    common, _case = _real_medoid_fixture()
    baseline = compile_lfm2_real_medoid_structural_residual_mlp_fold(**common)
    rows = int(common["cases"][0].target_input.shape[0])
    constant = compile_lfm2_real_medoid_structural_residual_mlp_fold(
        **common,
        case_row_sample_weights=[[1.0] * rows],
    )
    scaled = compile_lfm2_real_medoid_structural_residual_mlp_fold(
        **common,
        case_row_sample_weights=[[7.5] * rows],
    )
    assert constant.compiled_state_sha256 == baseline.compiled_state_sha256
    assert scaled.compiled_state_sha256 == baseline.compiled_state_sha256
    assert "fit_row_weight_profile" not in baseline.as_dict()
    assert (
        baseline.as_dict()["fit_weighting"]
        == "sqrt_case_weight_repeated_per_case_row"
    )
    weighted_receipt = constant.as_dict()
    assert (
        weighted_receipt["fit_weighting"]
        == "sqrt_case_weight_times_unit_mean_caller_supplied_row_profile"
    )
    assert weighted_receipt["fit_row_weight_profile"]["rows"] == rows


def test_nonconstant_row_profile_changes_the_compiled_state() -> None:
    common, _case = _real_medoid_fixture()
    rows = int(common["cases"][0].target_input.shape[0])
    baseline = compile_lfm2_real_medoid_structural_residual_mlp_fold(**common)
    emphasis = [1.0] * rows
    emphasis[0] = 3.0
    weighted = compile_lfm2_real_medoid_structural_residual_mlp_fold(
        **common,
        case_row_sample_weights=[emphasis],
    )
    assert weighted.compiled_state_sha256 != baseline.compiled_state_sha256
    assert weighted.fit_row_weight_maximum > weighted.fit_row_weight_minimum

    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        compile_lfm2_real_medoid_structural_residual_mlp_fold(
            **common,
            case_row_sample_weights=[[1.0] * (rows + 1)],
        )
    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        compile_lfm2_real_medoid_structural_residual_mlp_fold(
            **common,
            case_row_sample_weights=[[0.0] * rows],
        )


def test_semantic_margin_row_weight_profiles_preserve_case_mass() -> None:
    adapter = pytest.importorskip(
        "neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter"
    )
    uniform = adapter._semantic_case_row_weights(
        (1.0, 1.0, 1.0, 1.0),
        case_weight=0.5,
        row_count=4,
        batch_index=0,
    )
    assert uniform == pytest.approx((0.125,) * 4)
    skewed = adapter._semantic_case_row_weights(
        (3.0, 1.0, 1.0, 1.0),
        case_weight=0.5,
        row_count=4,
        batch_index=0,
    )
    assert math.fsum(skewed) == pytest.approx(0.5)
    assert skewed[0] > skewed[1]
    with pytest.raises(adapter.Lfm2SemanticLayerStackRefoldError):
        adapter._semantic_case_row_weights(
            (1.0, 1.0),
            case_weight=0.5,
            row_count=4,
            batch_index=0,
        )
    with pytest.raises(adapter.Lfm2SemanticLayerStackRefoldError):
        adapter._semantic_case_row_weights(
            (1.0, 0.0, 1.0, 1.0),
            case_weight=0.5,
            row_count=4,
            batch_index=0,
        )


# ---------------------------------------------------------------------------
# 4. Gate operating point.
# ---------------------------------------------------------------------------


def test_gate_preactivation_statistics_are_exact_on_a_hand_built_block() -> None:
    rows = torch.tensor(
        [[-9.0, 0.0], [-1.0, 2.0], [1.0, 4.0], [9.0, 6.0]],
        dtype=torch.float64,
    )
    statistics = lfm2_gate_preactivation_statistics(
        rows, label="hand", saturation_thresholds=(5.0,)
    )
    assert statistics.rows == 4
    assert statistics.channels == 2
    assert torch.allclose(
        statistics.channel_mean,
        torch.tensor([0.0, 3.0], dtype=torch.float64),
    )
    assert torch.allclose(
        statistics.channel_variance,
        torch.tensor([41.0, 5.0], dtype=torch.float64),
    )
    # One of four rows is below -5 on channel zero; one is above +5 there, and
    # channel one has one row above +5 and none below -5.
    assert torch.allclose(
        statistics.negative_saturation_fraction[0],
        torch.tensor([0.25, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        statistics.positive_saturation_fraction[0],
        torch.tensor([0.25, 0.25], dtype=torch.float64),
    )
    body = statistics.as_dict()
    assert body["saturation"][0]["threshold"] == 5.0
    assert body["saturation"][0]["mean_fraction_below_negative_threshold"] == 0.125


def test_gate_correction_matches_both_moments_exactly_on_synthetic_rows() -> None:
    generator = torch.Generator().manual_seed(88991)
    rows, width, channels = 64, 5, 7
    design = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    # The all-ones vector must lie in the design's row space for the shift to
    # be a pure constant; a genuine RMSNorm output has no such coordinate,
    # which is exactly what the reported residual is for.
    design[:, 0] = 1.0
    fitted = torch.randn(channels, width, generator=generator, dtype=torch.float64)
    targets = (
        torch.randn(rows, channels, generator=generator, dtype=torch.float64) * 2.5
        + torch.tensor([0.0, 1.0, -2.0, 3.0, 0.5, -0.5, 4.0], dtype=torch.float64)
    )
    correction = lfm2_gate_operating_point_correction(
        design_rows=design,
        fitted_gate_weight=fitted,
        parent_gate_targets=targets,
        saturation_thresholds=(2.0, 4.0),
    )
    diagnostics = correction.as_dict()
    assert diagnostics["constant_direction"]["exact_constant_response"] is True
    assert diagnostics["constant_direction"]["relative_residual"] < 1e-14
    assert diagnostics["moment_match"][
        "maximum_absolute_channel_mean_error"
    ] < 1e-10
    assert diagnostics["moment_match"][
        "maximum_absolute_channel_standard_deviation_error"
    ] < 1e-10

    corrected_rows = design @ correction.corrected_gate_weight.T
    parent = lfm2_gate_preactivation_statistics(targets, label="parent")
    child = lfm2_gate_preactivation_statistics(corrected_rows, label="child")
    assert torch.allclose(
        child.channel_mean, parent.channel_mean, rtol=0.0, atol=1e-10
    )
    assert torch.allclose(
        child.channel_standard_deviation,
        parent.channel_standard_deviation,
        rtol=0.0,
        atol=1e-10,
    )
    assert torch.equal(correction.uncorrected_gate_weight, fitted)
    assert not torch.equal(correction.corrected_gate_weight, fitted)
    assert diagnostics["requires_behavioral_validation_before_production_use"] is True


def test_scale_only_gate_correction_matches_std_exactly_without_a_constant() -> (
    None
):
    # A generic RMSNorm-shaped design has no constant coordinate, so the shared
    # constant direction cannot carry a mean shift.  Scale-only mode drops the
    # shift and matches the standard deviation exactly instead of matching
    # neither moment well.
    generator = torch.Generator().manual_seed(5150)
    design = torch.randn(40, 6, generator=generator, dtype=torch.float64)
    fitted = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    targets = (
        torch.randn(40, 5, generator=generator, dtype=torch.float64) * 3.0 + 2.0
    )
    affine = lfm2_gate_operating_point_correction(
        design_rows=design,
        fitted_gate_weight=fitted,
        parent_gate_targets=targets,
    )
    scale_only = lfm2_gate_operating_point_correction(
        design_rows=design,
        fitted_gate_weight=fitted,
        parent_gate_targets=targets,
        match_channel_mean=False,
    )
    affine_body = affine.as_dict()
    scale_body = scale_only.as_dict()
    assert affine_body["constant_direction"]["exact_constant_response"] is False
    assert affine_body["constant_direction"]["relative_residual"] > 0.1
    assert scale_body["channel_mean_matched"] is False
    assert scale_body["constant_direction"]["construction"] == (
        "not_solved_scale_only_correction"
    )
    assert torch.count_nonzero(scale_only.channel_shift) == 0
    assert torch.count_nonzero(scale_only.constant_direction) == 0

    # Standard deviation matches exactly in both modes; the mean does not, and
    # the affine mode is not better at it than doing nothing would be.
    assert scale_body["moment_match"][
        "maximum_absolute_channel_standard_deviation_error"
    ] < 1e-10
    corrected = design @ scale_only.corrected_gate_weight.T
    parent = lfm2_gate_preactivation_statistics(targets, label="parent")
    child = lfm2_gate_preactivation_statistics(corrected, label="child")
    assert torch.allclose(
        child.channel_standard_deviation,
        parent.channel_standard_deviation,
        rtol=0.0,
        atol=1e-10,
    )
    assert torch.equal(
        scale_only.corrected_gate_weight,
        (scale_only.channel_scale.unsqueeze(1) * fitted).contiguous(),
    )


def test_gate_correction_holds_degenerate_channels_at_unit_scale() -> None:
    design = torch.tensor(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
        dtype=torch.float64,
    )
    # Channel zero reads only the constant coordinate, so its child
    # pre-activation has exactly zero spread and no finite rescale exists.
    fitted = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    targets = torch.tensor(
        [[1.0, 0.0], [2.0, 4.0], [3.0, 8.0], [4.0, 12.0]],
        dtype=torch.float64,
    )
    correction = lfm2_gate_operating_point_correction(
        design_rows=design,
        fitted_gate_weight=fitted,
        parent_gate_targets=targets,
        maximum_channel_scale=8.0,
    )
    diagnostics = correction.as_dict()
    assert diagnostics["channel_scale"]["degenerate_channels_held_at_unit_scale"] == 1
    assert float(correction.channel_scale[0]) == 1.0
    assert float(correction.channel_scale[1]) == pytest.approx(4.0)
    assert diagnostics["channel_scale"]["maximum"] <= 8.0


def test_gate_diagnostics_do_not_change_the_compiled_state_but_correction_does() -> (
    None
):
    common, _case = _prepared_mlp_style_fixture()
    baseline = compile_lfm2_structural_residual_mlp_fold(**common)
    assert "gate_operating_point" not in baseline.as_dict()
    assert baseline.gate_operating_point is None
    assert baseline.uncorrected_gate_weight is None

    observed = compile_lfm2_structural_residual_mlp_fold(
        **common,
        gate_operating_point_diagnostics=True,
    )
    assert observed.compiled_state_sha256 == baseline.compiled_state_sha256
    receipt = observed.as_dict()["gate_operating_point"]
    assert receipt["decision_role"] == "diagnostic_only"
    assert receipt["correction_applied"] is False
    assert receipt["drift"]["maximum_absolute_channel_mean_drift"] >= 0.0

    corrected = compile_lfm2_structural_residual_mlp_fold(
        **common,
        gate_operating_point_diagnostics=True,
        gate_operating_point_correction=True,
    )
    assert corrected.compiled_state_sha256 != baseline.compiled_state_sha256
    corrected_receipt = corrected.as_dict()["gate_operating_point"]
    assert corrected_receipt["correction_applied"] is True
    assert corrected_receipt["applied_automatically"] is False
    assert corrected.uncorrected_gate_weight is not None
    assert torch.allclose(
        corrected.uncorrected_gate_weight.to(torch.float64),
        baseline.gate_weight.to(torch.float64),
        rtol=0.0,
        atol=1e-12,
    )

    scale_only = compile_lfm2_structural_residual_mlp_fold(
        **common,
        gate_operating_point_diagnostics=True,
        gate_operating_point_correction=True,
        gate_operating_point_match_channel_mean=False,
    )
    assert scale_only.compiled_state_sha256 != corrected.compiled_state_sha256
    assert scale_only.as_dict()["gate_operating_point"]["channel_mean_matched"] is (
        False
    )

    with pytest.raises(Lfm2StructuralResidualAttentionFoldError):
        compile_lfm2_structural_residual_mlp_fold(
            **common,
            gate_operating_point_correction=True,
        )


def test_gate_operating_point_report_is_read_only() -> None:
    generator = torch.Generator().manual_seed(717)
    design = torch.randn(24, 4, generator=generator, dtype=torch.float64)
    fitted = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    targets = torch.randn(24, 3, generator=generator, dtype=torch.float64) * 5.0
    before = fold_module.lfm2_authenticated_tensor_sha256(fitted)
    report = lfm2_gate_operating_point_report(
        design_rows=design,
        fitted_gate_weight=fitted,
        parent_gate_targets=targets,
    )
    assert fold_module.lfm2_authenticated_tensor_sha256(fitted) == before
    assert report["correction_applied"] is False
    assert report["decision_role"] == "diagnostic_only"
    assert report["parent"]["channels"] == 3
    assert report["child"]["rows"] == 24
    assert report["drift"]["maximum_absolute_channel_standard_deviation_drift"] > 0.0


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _real_medoid_fixture():
    generator = torch.Generator().manual_seed(6021)
    source_gate = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    source_up = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    source_down = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    target_input = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    parent_gate = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    parent_up = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    target_residual = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    endpoint = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    case = Lfm2CachedMlpFoldCase(
        target_input=target_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=target_residual,
    )
    target = _ToyMlp(hidden_size=2, intermediate_size=2, dtype=torch.float64)
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25], [-0.75, 0.125]],
        dtype=torch.float64,
    )
    common = {
        "source_gate_weight": source_gate,
        "source_up_weight": source_up,
        "source_down_weight": source_down,
        "target_gate_weight_template": target.w1.weight,
        "target_up_weight_template": target.w3.weight,
        "target_down_weight_template": target.w2.weight,
        "residual_chart": chart,
        "residual_chart_sha256": fold_module.lfm2_authenticated_tensor_sha256(chart),
        "cases": (case,),
        "anchors_per_source": 2,
        "max_candidate_targets": 2,
        "ridge": 1e-6,
        "cluster_ridge": 1e-8,
    }
    return common, case


def _prepared_mlp_style_fixture():
    generator = torch.Generator().manual_seed(9917)
    source_gate = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    source_up = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    source_down = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    target_input = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    parent_gate = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    parent_up = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    target_residual = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    parent_endpoint = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    case = Lfm2CachedMlpFoldCase(
        target_input=target_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=parent_endpoint,
        target_pre_mlp_residual=target_residual,
    )
    target = _ToyMlp(hidden_size=4, intermediate_size=2, dtype=torch.float64)
    common = {
        "source_gate_weight": source_gate,
        "source_up_weight": source_up,
        "source_down_weight": source_down,
        "target_gate_weight_template": target.w1.weight,
        "target_up_weight_template": target.w3.weight,
        "target_down_weight_template": target.w2.weight,
        "cases": (case,),
        "residual_assignment_sha256": "9" * 64,
        "ridge": 1e-4,
        "max_prior_delta_ratio": None,
        "cluster_ridge": 1e-8,
        "cluster_max_iterations": 16,
        "cluster_distance_chunk_size": 512,
    }
    return common, case
