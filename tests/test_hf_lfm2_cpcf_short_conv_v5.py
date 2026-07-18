from __future__ import annotations

import copy
import hashlib
import json

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (  # noqa: E402
    Lfm2CpcfShortConvV5Error,
    Lfm2CpcfV5AcceptedPathState,
    Lfm2CpcfV5FirBasisCase,
    Lfm2CpcfV5FirBasisPolicy,
    Lfm2CpcfV5RidgeCase,
    Lfm2CpcfV5RidgePolicy,
    Lfm2CpcfV5SelectionPolicy,
    build_fir_commuting_balanced_all_channel_basis,
    condition_capped_equilibrated_prior_residual_ridge_frontier,
    fixed_v5_accepted_edge_interpolation_fractions,
    interpolate_every_lfm2_cpcf_v5_accepted_edge,
    rescore_sealed_v4_short_conv_selection_frontier,
    verify_lfm2_cpcf_short_conv_v5_report,
)


def _sha256(value) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _seal(value: dict) -> dict:
    return {**value, "report_sha256": _sha256(value)}


def _selection_report(
    candidate_number: int,
    *,
    endpoint_max: float,
    endpoint_min_cos: float,
    endpoint_mean: float,
    endpoint_mean_cos: float,
    context_max: float,
    context_min_cos: float,
    context_mean: float,
    context_mean_cos: float,
    split: str = "selection",
) -> dict:
    return _seal(
        {
            "schema": "neural-forge-lfm2-cpcf-short-conv-v4",
            "kind": "target_dtype_evaluation",
            "policy_sha256": "1" * 64,
            "split": split,
            "target_dtype": "bfloat16",
            "case_balanced": True,
            "case_count": 1,
            "cases": [
                {
                    "case_id": "selection-trajectory-000",
                    "rows": 8,
                    "width": 2,
                    "target_sha256": "2" * 64,
                    "valid_rows_sha256": None,
                    "projection_case_authentication_sha256": "3" * 64,
                    "context_rows": 12,
                    "context_width": 2,
                    "context_target_sha256": "4" * 64,
                    "valid_context_rows_sha256": None,
                }
            ],
            "evidence_sha256": "5" * 64,
            "candidate_state_sha256": f"{candidate_number:064x}",
            "maximum_case_relative_rmse": endpoint_max,
            "minimum_case_cosine_similarity": endpoint_min_cos,
            "mean_case_relative_rmse": endpoint_mean,
            "mean_case_cosine_similarity": endpoint_mean_cos,
            "maximum_case_context_relative_rmse": context_max,
            "minimum_case_context_cosine_similarity": context_min_cos,
            "mean_case_context_relative_rmse": context_mean,
            "mean_case_context_cosine_similarity": context_mean_cos,
            "context_evidence_present": True,
            "actual_parameter_and_intermediate_target_dtype_replay": True,
            "selection_used_for_parameter_fitting": False,
            "validation_used_for_parameter_fitting": False,
        }
    )


def _layer0_like_selection_frontier() -> dict[str, dict]:
    return {
        "ridge-alpha-00-1e-08--anchor-00-prior": _selection_report(
            1,
            endpoint_max=0.999,
            endpoint_min_cos=0.10,
            endpoint_mean=0.985,
            endpoint_mean_cos=0.13,
            context_max=0.9990,
            context_min_cos=0.050,
            context_mean=0.9800,
            context_mean_cos=0.080,
        ),
        "ridge-alpha-00-1e-08--anchor-04-pass-01-writer": _selection_report(
            2,
            endpoint_max=0.979,
            endpoint_min_cos=0.17,
            endpoint_mean=0.960,
            endpoint_mean_cos=0.20,
            context_max=0.9995,
            context_min_cos=0.048,
            context_mean=0.9810,
            context_mean_cos=0.078,
        ),
        "ridge-alpha-01-0.0001--anchor-04-pass-01-writer": _selection_report(
            3,
            endpoint_max=0.974019,
            endpoint_min_cos=0.18,
            endpoint_mean=0.955,
            endpoint_mean_cos=0.22,
            context_max=0.999646,
            context_min_cos=0.047,
            context_mean=0.9812,
            context_mean_cos=0.077,
        ),
        "ridge-alpha-01-0.0001--trust-05-0.5": _selection_report(
            4,
            endpoint_max=0.983921,
            endpoint_min_cos=0.16,
            endpoint_mean=0.968,
            endpoint_mean_cos=0.19,
            context_max=0.9994,
            context_min_cos=0.049,
            context_mean=0.9807,
            context_mean_cos=0.079,
        ),
        "ridge-alpha-01-0.0001--anchor-06-pass-01-bcx": _selection_report(
            5,
            endpoint_max=0.927514,
            endpoint_min_cos=0.25,
            endpoint_mean=0.90,
            endpoint_mean_cos=0.30,
            context_max=1.001542,
            context_min_cos=0.04,
            context_mean=0.985,
            context_mean_cos=0.07,
        ),
        "ridge-alpha-01-0.0001--context-exchange-violator": _selection_report(
            6,
            endpoint_max=0.90,
            endpoint_min_cos=0.30,
            endpoint_mean=0.88,
            endpoint_mean_cos=0.32,
            context_max=0.999,
            context_min_cos=0.0,
            context_mean=0.990,
            context_mean_cos=0.0,
        ),
    }


def test_v5_rescorer_selects_layer0_alpha_1e4_pass1_writer() -> None:
    report = rescore_sealed_v4_short_conv_selection_frontier(
        _layer0_like_selection_frontier()
    )

    assert verify_lfm2_cpcf_short_conv_v5_report(report)
    assert report["selected_candidate_id"] == (
        "ridge-alpha-01-0.0001--anchor-04-pass-01-writer"
    )
    assert [item["candidate_id"] for item in report["ranking"][:3]] == [
        "ridge-alpha-01-0.0001--anchor-04-pass-01-writer",
        "ridge-alpha-00-1e-08--anchor-04-pass-01-writer",
        "ridge-alpha-01-0.0001--trust-05-0.5",
    ]
    assert report["candidate_count"] == 6
    assert report["feasible_candidate_count"] == 4
    assert report["minimum_gate_slack_ranked"] is False
    assert report["validation_reports_consumed"] == 0
    assert report["validation_accessible"] is False
    selected = next(
        value
        for value in report["candidate_inventory"]
        if value["candidate_id"] == report["selected_candidate_id"]
    )
    assert all(selected["absolute_gate_checks"].values())
    assert all(
        value["passed"]
        for value in selected["prior_relative_context_exchange"].values()
    )
    maximum_exchange = selected["prior_relative_context_exchange"][
        "maximum_relative_rmse"
    ]
    assert maximum_exchange["context_damage"] == pytest.approx(0.000646)
    assert maximum_exchange["allowed_context_damage"] == pytest.approx(
        0.05 * (0.999 - 0.974019)
    )
    assert selected["lexicographic_key"] == [
        0.974019,
        -0.18,
        0.955,
        -0.22,
        0.999646,
        -0.047,
        0.9812,
        -0.077,
        "ridge-alpha-01-0.0001--anchor-04-pass-01-writer",
    ]
    final_bcx = next(
        value
        for value in report["candidate_inventory"]
        if value["candidate_id"].endswith("pass-01-bcx")
    )
    assert final_bcx["absolute_gate_checks"]["context_maximum_relative_rmse"] is False
    exchange_violator = next(
        value
        for value in report["candidate_inventory"]
        if value["candidate_id"].endswith("context-exchange-violator")
    )
    assert exchange_violator["passed"] is False
    assert any(
        name.startswith("context_exchange_")
        for name in exchange_violator["failed_checks"]
    )


def test_v5_rescorer_rejects_validation_tamper_and_zero_feasible() -> None:
    validation = _layer0_like_selection_frontier()
    candidate_id = next(iter(validation))
    validation[candidate_id] = _selection_report(
        1,
        endpoint_max=0.9,
        endpoint_min_cos=0.1,
        endpoint_mean=0.8,
        endpoint_mean_cos=0.2,
        context_max=0.9,
        context_min_cos=0.1,
        context_mean=0.8,
        context_mean_cos=0.2,
        split="validation",
    )
    with pytest.raises(Lfm2CpcfShortConvV5Error, match="selection reports"):
        rescore_sealed_v4_short_conv_selection_frontier(validation)

    tampered = _layer0_like_selection_frontier()
    tampered_id = next(iter(tampered))
    tampered[tampered_id]["maximum_case_relative_rmse"] = 0.1
    with pytest.raises(Lfm2CpcfShortConvV5Error, match="canonically sealed"):
        rescore_sealed_v4_short_conv_selection_frontier(tampered)

    no_feasible = _layer0_like_selection_frontier()
    for index, (name, report) in enumerate(no_feasible.items(), start=10):
        body = dict(report)
        body.pop("report_sha256")
        body["maximum_case_relative_rmse"] = 1.1
        body["candidate_state_sha256"] = f"{index:064x}"
        no_feasible[name] = _seal(body)
    with pytest.raises(Lfm2CpcfShortConvV5Error, match="zero candidates"):
        rescore_sealed_v4_short_conv_selection_frontier(no_feasible)


def test_v5_selection_policy_is_frozen() -> None:
    policy = Lfm2CpcfV5SelectionPolicy()
    assert policy.verify_authentication()
    assert policy.as_dict()["context_exchange_beta"] == 0.05
    with pytest.raises(Lfm2CpcfShortConvV5Error, match="frozen"):
        Lfm2CpcfV5SelectionPolicy(context_exchange_beta=0.051)
    with pytest.raises(Lfm2CpcfShortConvV5Error, match="frozen"):
        Lfm2CpcfV5SelectionPolicy(endpoint_maximum_relative_rmse=1.1)


def test_equilibrated_ridge_caps_condition_and_preserves_null_prior() -> None:
    sample = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    design = torch.stack(
        (
            sample * 1e6,
            sample + 1e-7 * sample.square(),
            torch.zeros_like(sample),
        ),
        dim=1,
    )
    prior_weight = torch.tensor([[0.0], [0.0], [7.25]], dtype=torch.float64)
    prior_bias = torch.tensor([-0.5], dtype=torch.float64)
    target = 2e-6 * design[:, :1] - 0.75 * design[:, 1:2] + 1.75
    cases = (
        Lfm2CpcfV5RidgeCase(
            "fit-a",
            "fit",
            design[:61],
            target[:61],
        ),
        Lfm2CpcfV5RidgeCase(
            "fit-b",
            "fit",
            design[61:],
            target[61:],
        ),
    )
    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=1e8,
        effect_trust_radius=1e8,
    )

    frontier = condition_capped_equilibrated_prior_residual_ridge_frontier(
        cases,
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        policy=policy,
    )

    assert tuple(frontier) == ("kappa-01000", "kappa-10000")
    for name, result in frontier.items():
        assert result.verify_authentication()
        receipt = result.diagnostics
        assert receipt["condition_cap"] in (1_000.0, 10_000.0)
        assert receipt["regularized_condition"] <= receipt["condition_cap"] * (
            1.0 + 1e-12
        )
        assert receipt["condition_cap_passed"] is True
        assert receipt["column_equilibrated"] is True
        assert receipt["symmetric_factorization"] == "torch.linalg.eigh"
        assert receipt["inactive_column_indices"] == [2]
        assert receipt["nullspace_preserves_prior"] is True
        assert result.weight[2, 0].item() == prior_weight[2, 0].item()
        prediction = design @ result.weight + result.bias
        assert torch.sqrt(torch.mean((prediction - target).square())).item() < 0.2
        assert name.endswith(str(int(receipt["condition_cap"])).zfill(5))


def test_equilibrated_ridge_separates_intercept_and_trust_units() -> None:
    design = torch.linspace(-2.0, 2.0, 81, dtype=torch.float64).reshape(-1, 1)
    prior_weight = torch.tensor([[2.0]], dtype=torch.float64)
    prior_bias = torch.tensor([-3.0], dtype=torch.float64)
    target = 2.0 * design + 7.0
    exact = condition_capped_equilibrated_prior_residual_ridge_frontier(
        (Lfm2CpcfV5RidgeCase("fit", "fit", design, target),),
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        policy=Lfm2CpcfV5RidgePolicy(
            parameter_trust_radius=1e8,
            effect_trust_radius=1e8,
        ),
    )["kappa-01000"]
    assert exact.weight.item() == pytest.approx(2.0, abs=1e-12)
    assert exact.bias.item() == pytest.approx(7.0, abs=1e-12)
    assert exact.diagnostics["fit_intercept"] is True
    assert exact.diagnostics["bias_contract"] == "separate_unpenalized_intercept"
    assert exact.diagnostics["separate_unpenalized_intercept"] is True
    assert exact.diagnostics["centering_applied"] is True

    bounded_target = 50.0 * design + 100.0
    bounded = condition_capped_equilibrated_prior_residual_ridge_frontier(
        (Lfm2CpcfV5RidgeCase("fit", "fit", design, bounded_target),),
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        policy=Lfm2CpcfV5RidgePolicy(
            parameter_trust_radius=0.1,
            effect_trust_radius=0.2,
        ),
    )["kappa-01000"]
    parameter = bounded.diagnostics["parameter_space_trust"]
    effect = bounded.diagnostics["function_space_trust"]
    assert parameter["units"] == "coefficient_parameter_units"
    assert effect["units"] == "predicted_effect_units"
    assert parameter["passed"] is True
    assert effect["passed"] is True
    assert parameter["accepted_correction_rms"] <= parameter["absolute_limit_rms"] * (
        1.0 + 1e-12
    )
    assert effect["accepted_correction_rms"] <= effect["absolute_limit_rms"] * (
        1.0 + 1e-12
    )
    assert bounded.diagnostics["joint_trust_is_intersection"] is True
    assert bounded.diagnostics["parameter_trust_reference_uses_function_units"] is False
    assert bounded.diagnostics["function_trust_reference_uses_parameter_units"] is False


def test_equilibrated_ridge_no_intercept_uses_encoded_bias_coordinate() -> None:
    sample = torch.linspace(-2.0, 2.0, 81, dtype=torch.float64)
    design = torch.stack((sample, torch.ones_like(sample)), dim=1)
    prior_weight = torch.zeros((2, 1), dtype=torch.float64)
    target = (2.0 * sample + 3.0).reshape(-1, 1)

    result = condition_capped_equilibrated_prior_residual_ridge_frontier(
        (Lfm2CpcfV5RidgeCase("fit", "fit", design, target),),
        prior_weight=prior_weight,
        prior_bias=None,
        fit_intercept=False,
        policy=Lfm2CpcfV5RidgePolicy(
            parameter_trust_radius=1e12,
            effect_trust_radius=1e12,
        ),
    )["kappa-01000"]

    assert result.verify_authentication()
    assert result.bias is None
    assert torch.allclose(
        result.weight[:, 0],
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    receipt = result.diagnostics
    assert receipt["fit_intercept"] is False
    assert receipt["bias_contract"] == "absent_frozen_zero"
    assert receipt["separate_unpenalized_intercept"] is False
    assert receipt["bias_value_is_absent"] is True
    assert receipt["bias_effect_is_frozen_zero"] is True
    assert receipt["centering_applied"] is False
    assert receipt["no_intercept_uses_uncentered_system"] is True
    assert receipt["design_mean_sha256"] is None
    assert receipt["prior_residual_mean_sha256"] is None
    assert receipt["centered_case_balanced_design_sha256"] is None
    assert receipt["uncentered_case_balanced_design_sha256"] is not None
    assert receipt["prior_bias_sha256"] is None
    assert receipt["raw_bias_delta_sha256"] is None
    assert receipt["solution_bias_sha256"] is None

    with pytest.raises(Lfm2CpcfShortConvV5Error, match="prior_bias=None"):
        condition_capped_equilibrated_prior_residual_ridge_frontier(
            (Lfm2CpcfV5RidgeCase("fit", "fit", design, target),),
            prior_weight=prior_weight,
            prior_bias=torch.zeros(1, dtype=torch.float64),
            fit_intercept=False,
        )


def test_equilibrated_ridge_rejects_nonfit_evidence() -> None:
    with pytest.raises(Lfm2CpcfShortConvV5Error, match="fit evidence only"):
        Lfm2CpcfV5RidgeCase(
            "selection",
            "selection",
            torch.ones((3, 1)),
            torch.ones((3, 1)),
        )


def test_equilibrated_ridge_is_column_scale_and_row_duplication_invariant() -> None:
    generator = torch.Generator().manual_seed(20260718)
    design_a = torch.randn((37, 3), generator=generator, dtype=torch.float64)
    design_b = torch.randn((11, 3), generator=generator, dtype=torch.float64)
    coefficient = torch.tensor(
        [[1.25, -0.3], [0.2, 0.8], [-0.6, 0.4]],
        dtype=torch.float64,
    )
    bias = torch.tensor([0.75, -1.5], dtype=torch.float64)
    target_a = design_a @ coefficient + bias
    target_b = design_b @ coefficient + bias
    prior_weight = 0.8 * coefficient
    prior_bias = 0.8 * bias
    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=1e12,
        effect_trust_radius=1e12,
    )

    def solve(first_design, first_target, second_design, second_target):
        return condition_capped_equilibrated_prior_residual_ridge_frontier(
            (
                Lfm2CpcfV5RidgeCase(
                    "fit-a",
                    "fit",
                    first_design,
                    first_target,
                ),
                Lfm2CpcfV5RidgeCase(
                    "fit-b",
                    "fit",
                    second_design,
                    second_target,
                ),
            ),
            prior_weight=prior_weight,
            prior_bias=prior_bias,
            policy=policy,
        )["kappa-10000"]

    base = solve(design_a, target_a, design_b, target_b)
    duplicated = solve(
        design_a.repeat((5, 1)),
        target_a.repeat((5, 1)),
        design_b.repeat((3, 1)),
        target_b.repeat((3, 1)),
    )
    assert torch.allclose(base.weight, duplicated.weight, atol=2e-12, rtol=2e-12)
    assert torch.allclose(base.bias, duplicated.bias, atol=2e-12, rtol=2e-12)

    scales = torch.tensor([1e5, 1e-3, 17.0], dtype=torch.float64)
    scaled = condition_capped_equilibrated_prior_residual_ridge_frontier(
        (
            Lfm2CpcfV5RidgeCase(
                "fit-a",
                "fit",
                design_a * scales,
                target_a,
            ),
            Lfm2CpcfV5RidgeCase(
                "fit-b",
                "fit",
                design_b * scales,
                target_b,
            ),
        ),
        prior_weight=prior_weight / scales.unsqueeze(-1),
        prior_bias=prior_bias,
        policy=policy,
    )["kappa-10000"]
    assert torch.allclose(
        base.weight,
        scaled.weight * scales.unsqueeze(-1),
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.allclose(base.bias, scaled.bias, atol=2e-10, rtol=2e-10)


def _basis_cases() -> tuple[Lfm2CpcfV5FirBasisCase, ...]:
    source = torch.arange(8, dtype=torch.float64)
    return (
        Lfm2CpcfV5FirBasisCase(
            "fit-a",
            "fit",
            torch.stack((source, source.square()), dim=1),
            torch.stack((torch.sin(source), torch.cos(source)), dim=1),
        ),
        Lfm2CpcfV5FirBasisCase(
            "fit-b",
            "fit",
            torch.stack((source * 0.5, (source - 2.0).square()), dim=1),
            torch.stack((source.remainder(3.0), source / 8.0), dim=1),
        ),
    )


def test_fir_commuting_basis_balances_and_uses_every_channel() -> None:
    fir = torch.tensor(
        [
            [1.00, 0.20, -0.10],
            [1.01, 0.19, -0.11],
            [0.99, 0.21, -0.09],
            [-0.50, 0.70, 0.20],
            [-0.49, 0.69, 0.21],
            [0.10, -0.80, 0.90],
            [0.11, -0.81, 0.89],
            [0.09, -0.79, 0.91],
        ],
        dtype=torch.float64,
    )
    first = build_fir_commuting_balanced_all_channel_basis(
        _basis_cases(),
        source_fir_kernel=fir,
        target_channels=3,
        policy=Lfm2CpcfV5FirBasisPolicy(
            maximum_projected_fir_offdiagonal_relative_frobenius=0.0,
            maximum_per_lag_commutator_relative_frobenius=1.0,
        ),
    )
    second = build_fir_commuting_balanced_all_channel_basis(
        tuple(reversed(_basis_cases())),
        source_fir_kernel=fir,
        target_channels=3,
        policy=Lfm2CpcfV5FirBasisPolicy(
            maximum_projected_fir_offdiagonal_relative_frobenius=0.0,
            maximum_per_lag_commutator_relative_frobenius=1.0,
        ),
    )

    assert first.verify_authentication()
    assert second.verify_authentication()
    assert torch.equal(first.encoder, second.encoder)
    assert first.source_to_target_group == second.source_to_target_group
    assert sorted(first.source_to_target_group) == [0, 0, 0, 1, 1, 1, 2, 2]
    assert torch.all((first.encoder != 0).sum(dim=1) == 1)
    assert torch.allclose(
        first.decoder @ first.encoder,
        torch.eye(3, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )
    assert first.report["source_channels_consumed_count"] == 8
    assert first.report["source_channels_dropped"] == 0
    assert first.report["source_channels_pruned"] == 0
    assert first.report["full_source_numeric_participation"] is True
    assert first.report["projected_fir_offdiagonal_relative_frobenius"] == 0.0
    assert first.report["projected_fir_offdiagonal_gate_passed"] is True
    assert (
        first.report["projected_offdiagonal_is_not_sufficient_for_commuting_claim"]
        is True
    )
    assert first.report["maximum_per_lag_commutator_relative_frobenius"] == (
        pytest.approx(0.987049763115577)
    )
    assert (
        first.report["maximum_per_lag_commutator_relative_frobenius_gate_passed"]
        is True
    )
    assert first.report["commuting_claim_uses_actual_intertwining_residual"] is True
    assert first.report["depthwise_fir_representable_in_target_basis"] is True
    assert first.report["exact_dual"] is True


def test_fir_commutator_gate_passes_exact_groups_and_rejects_distortion() -> None:
    descriptors = torch.tensor(
        [[0.0], [0.0], [10.0], [10.0]],
        dtype=torch.float64,
    )
    cases = (
        Lfm2CpcfV5FirBasisCase(
            "fit",
            "fit",
            descriptors,
            descriptors,
        ),
    )
    strict = Lfm2CpcfV5FirBasisPolicy(
        maximum_projected_fir_offdiagonal_relative_frobenius=0.0,
        maximum_per_lag_commutator_relative_frobenius=2e-16,
    )
    exact = build_fir_commuting_balanced_all_channel_basis(
        cases,
        source_fir_kernel=torch.tensor(
            [[1.0, -0.5], [1.0, -0.5], [2.0, 0.25], [2.0, 0.25]],
            dtype=torch.float64,
        ),
        target_channels=2,
        policy=strict,
    )
    assert max(exact.report["per_lag_commutator_relative_frobenius"]) <= 2e-16
    assert exact.report["maximum_per_lag_commutator_relative_frobenius"] <= 2e-16
    assert (
        exact.report["maximum_per_lag_commutator_relative_frobenius_gate_passed"]
        is True
    )
    assert exact.report["depthwise_fir_representable_in_target_basis"] is True

    with pytest.raises(
        Lfm2CpcfShortConvV5Error,
        match="commutator residual",
    ):
        build_fir_commuting_balanced_all_channel_basis(
            cases,
            source_fir_kernel=torch.tensor(
                [[1.0, -0.5], [1.2, -0.4], [2.0, 0.25], [2.2, 0.35]],
                dtype=torch.float64,
            ),
            target_channels=2,
            policy=strict,
        )


def test_fir_commuting_basis_rejects_selection_descriptors() -> None:
    with pytest.raises(Lfm2CpcfShortConvV5Error, match="fit evidence only"):
        Lfm2CpcfV5FirBasisCase(
            "selection",
            "selection",
            torch.ones((4, 2)),
            torch.ones((4, 2)),
        )


def test_interpolation_covers_every_accepted_edge() -> None:
    first = Lfm2CpcfV5AcceptedPathState(
        stage_id="prior",
        state={
            "weight": torch.zeros((2, 2), dtype=torch.float64),
            "bias": torch.zeros(2, dtype=torch.float64),
        },
        acceptance_report_sha256="a" * 64,
    )
    second = Lfm2CpcfV5AcceptedPathState(
        stage_id="pass-00-writer",
        state={
            "weight": torch.ones((2, 2), dtype=torch.float64),
            "bias": torch.ones(2, dtype=torch.float64),
        },
        acceptance_report_sha256="b" * 64,
        predecessor_authentication_sha256=first.authentication_sha256,
    )
    third = Lfm2CpcfV5AcceptedPathState(
        stage_id="pass-00-fir",
        state={
            "weight": torch.full((2, 2), 3.0, dtype=torch.float64),
            "bias": torch.full((2,), 3.0, dtype=torch.float64),
        },
        acceptance_report_sha256="c" * 64,
        predecessor_authentication_sha256=second.authentication_sha256,
    )

    frontier = interpolate_every_lfm2_cpcf_v5_accepted_edge((first, second, third))

    fractions = fixed_v5_accepted_edge_interpolation_fractions()
    assert frontier.verify_authentication()
    assert frontier.report["accepted_edge_count"] == 2
    assert frontier.report["covered_accepted_edge_count"] == 2
    assert frontier.report["skipped_accepted_edge_count"] == 0
    assert frontier.report["per_accepted_edge_invariant"] is True
    assert len(frontier.candidates) == 2 * len(fractions)
    assert torch.allclose(
        frontier.candidates[0].state["weight"],
        torch.full((2, 2), fractions[0], dtype=torch.float64),
    )
    assert torch.allclose(
        frontier.candidates[len(fractions)].state["weight"],
        torch.full(
            (2, 2),
            1.0 + 2.0 * fractions[0],
            dtype=torch.float64,
        ),
    )

    skipped = copy.copy(third)
    object.__setattr__(
        skipped,
        "predecessor_authentication_sha256",
        first.authentication_sha256,
    )
    with pytest.raises(Lfm2CpcfShortConvV5Error):
        interpolate_every_lfm2_cpcf_v5_accepted_edge((first, second, skipped))
