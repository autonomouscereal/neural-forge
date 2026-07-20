from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge import hf_lfm2_cpcf_short_conv_v5 as v5_module  # noqa: E402
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (  # noqa: E402
    Lfm2CpcfShortConvV5Error,
    Lfm2CpcfV5AcceptedPathState,
    Lfm2CpcfV5AllChannelFirRidgeCase,
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
    solve_condition_capped_batched_all_channel_fir_ridge_candidate,
    solve_condition_capped_equilibrated_prior_residual_ridge_candidate,
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


def test_equilibrated_ridge_frontier_shares_one_system_with_exact_parity(
    monkeypatch,
) -> None:
    generator = torch.Generator().manual_seed(20260718)
    design_a = torch.randn((31, 4), generator=generator, dtype=torch.float64)
    design_b = torch.randn((17, 4), generator=generator, dtype=torch.float64)
    design_a[:, 3] = 0.0
    design_b[:, 3] = 0.0
    coefficient = torch.tensor(
        [
            [1.25, -0.5],
            [0.3, 0.75],
            [-0.8, 0.2],
            [4.0, -3.0],
        ],
        dtype=torch.float64,
    )
    bias = torch.tensor([0.5, -1.25], dtype=torch.float64)
    cases = (
        Lfm2CpcfV5RidgeCase(
            "fit-a",
            "fit",
            design_a,
            design_a @ coefficient + bias,
        ),
        Lfm2CpcfV5RidgeCase(
            "fit-b",
            "fit",
            design_b,
            design_b @ coefficient + bias,
        ),
    )
    prior_weight = torch.tensor(
        [
            [0.5, -0.25],
            [0.1, 0.4],
            [-0.2, 0.05],
            [7.0, -9.0],
        ],
        dtype=torch.float64,
    )
    prior_bias = torch.tensor([-0.5, 0.25], dtype=torch.float64)
    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=0.75,
        effect_trust_radius=0.5,
    )
    rows = v5_module._canonical_ridge_rows(cases)
    expected = {
        f"kappa-{int(cap):05d}": v5_module._solve_equilibrated_candidate(
            rows=rows,
            prior_weight=prior_weight,
            prior_bias=prior_bias,
            fit_intercept=True,
            condition_cap=cap,
            policy=policy,
        )
        for cap in policy.condition_caps
    }
    prepare_calls = []
    original_prepare = v5_module._prepare_equilibrated_ridge_system

    def counted_prepare(**kwargs):
        prepare_calls.append(kwargs)
        return original_prepare(**kwargs)

    monkeypatch.setattr(
        v5_module,
        "_prepare_equilibrated_ridge_system",
        counted_prepare,
    )
    actual = condition_capped_equilibrated_prior_residual_ridge_frontier(
        cases,
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        policy=policy,
    )

    assert len(prepare_calls) == 1
    assert tuple(actual) == tuple(expected) == ("kappa-01000", "kappa-10000")
    for candidate_id in expected:
        expected_result = expected[candidate_id]
        actual_result = actual[candidate_id]
        assert torch.equal(actual_result.weight, expected_result.weight)
        assert torch.equal(actual_result.bias, expected_result.bias)
        assert dict(actual_result.diagnostics) == dict(expected_result.diagnostics)
        assert actual_result.diagnostics["condition_cap_passed"] is True
        assert actual_result.diagnostics["nullspace_preserves_prior"] is True
        assert actual_result.verify_authentication()


@pytest.mark.parametrize("condition_cap", [1_000.0, 10_000.0])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_single_condition_cap_is_byte_identical_to_frontier_candidate(
    monkeypatch,
    condition_cap,
    fit_intercept,
) -> None:
    generator = torch.Generator().manual_seed(20260719)
    design_a = torch.randn((41, 5), generator=generator, dtype=torch.float64)
    design_b = torch.randn((23, 5), generator=generator, dtype=torch.float64)
    coefficient = torch.randn((5, 2), generator=generator, dtype=torch.float64)
    target_bias = torch.tensor([0.75, -1.25], dtype=torch.float64)
    cases = (
        Lfm2CpcfV5RidgeCase(
            "fit-a",
            "fit",
            design_a,
            design_a @ coefficient + target_bias,
        ),
        Lfm2CpcfV5RidgeCase(
            "fit-b",
            "fit",
            design_b,
            design_b @ coefficient + target_bias,
        ),
    )
    prior_weight = 0.6 * coefficient
    prior_bias = (
        0.6 * target_bias
        if fit_intercept
        else None
    )
    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=0.75,
        effect_trust_radius=0.5,
    )
    candidate_id = f"kappa-{int(condition_cap):05d}"
    expected = condition_capped_equilibrated_prior_residual_ridge_frontier(
        cases,
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        fit_intercept=fit_intercept,
        policy=policy,
    )[candidate_id]
    prepare_calls = []
    solve_calls = []
    original_prepare = v5_module._prepare_equilibrated_ridge_system
    original_solve = v5_module._solve_equilibrated_candidate

    def counted_prepare(**kwargs):
        prepare_calls.append(kwargs)
        return original_prepare(**kwargs)

    def counted_solve(**kwargs):
        solve_calls.append(kwargs)
        return original_solve(**kwargs)

    monkeypatch.setattr(
        v5_module,
        "_prepare_equilibrated_ridge_system",
        counted_prepare,
    )
    monkeypatch.setattr(
        v5_module,
        "_solve_equilibrated_candidate",
        counted_solve,
    )
    actual = solve_condition_capped_equilibrated_prior_residual_ridge_candidate(
        cases,
        condition_cap=condition_cap,
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        fit_intercept=fit_intercept,
        policy=policy,
    )

    assert len(prepare_calls) == 1
    assert len(solve_calls) == 1
    assert solve_calls[0]["condition_cap"] == condition_cap
    assert solve_calls[0]["shared_system"] is not None
    assert torch.equal(actual.weight, expected.weight)
    if expected.bias is None:
        assert actual.bias is None
    else:
        assert torch.equal(actual.bias, expected.bias)
    assert dict(actual.diagnostics) == dict(expected.diagnostics)
    assert (
        actual.diagnostics["report_sha256"]
        == expected.diagnostics["report_sha256"]
    )
    assert actual.verify_authentication()


def test_single_condition_cap_rejects_cap_outside_authenticated_policy() -> None:
    design = torch.ones((4, 1), dtype=torch.float64)
    case = Lfm2CpcfV5RidgeCase("fit", "fit", design, 2.0 * design)

    with pytest.raises(
        Lfm2CpcfShortConvV5Error,
        match="authenticated by the ridge policy",
    ):
        solve_condition_capped_equilibrated_prior_residual_ridge_candidate(
            (case,),
            condition_cap=2_000.0,
            prior_weight=torch.zeros((1, 1), dtype=torch.float64),
            prior_bias=None,
            fit_intercept=False,
        )


def _all_channel_fir_problem(
    *,
    channel_count: int,
    fir_length: int,
    physical_bias: bool,
    seed: int = 20260719,
):
    generator = torch.Generator().manual_seed(seed)
    feature_width = fir_length + int(physical_bias)
    prior = (
        torch.randn(
            (channel_count, feature_width),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    )
    desired = (
        prior
        + torch.randn(
            (channel_count, feature_width),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.35
    )
    cases = []
    for case_index, row_count in enumerate((37, 23, 11)):
        design = torch.randn(
            (channel_count, row_count, feature_width),
            generator=generator,
            dtype=torch.float64,
        )
        target = torch.bmm(
            design,
            desired.unsqueeze(-1),
        ).squeeze(-1)
        target = target + 0.01 * torch.randn(
            target.shape,
            generator=generator,
            dtype=torch.float64,
        )
        mask = torch.ones(row_count, dtype=torch.bool)
        mask[(case_index + 1) :: (case_index + 5)] = False
        cases.append(
            Lfm2CpcfV5AllChannelFirRidgeCase(
                case_id=f"fit-{case_index:02d}",
                split="fit",
                design=design,
                target=target,
                valid_rows=mask,
            )
        )
    return tuple(cases), prior


def _scalar_all_channel_fir_oracle(
    cases,
    prior,
    *,
    condition_cap,
    policy,
):
    results = []
    for channel in range(int(prior.shape[0])):
        scalar_cases = tuple(
            Lfm2CpcfV5RidgeCase(
                case_id=case.case_id,
                split="fit",
                design=case.design[channel],
                target=case.target[channel].unsqueeze(-1),
                valid_rows=case.valid_rows,
            )
            for case in cases
        )
        results.append(
            solve_condition_capped_equilibrated_prior_residual_ridge_candidate(
                scalar_cases,
                condition_cap=condition_cap,
                prior_weight=prior[channel].unsqueeze(-1),
                prior_bias=None,
                fit_intercept=False,
                policy=policy,
            ).weight[:, 0]
        )
    return torch.stack(results)


@pytest.mark.parametrize("condition_cap", [1_000.0, 10_000.0])
@pytest.mark.parametrize("physical_bias", [False, True])
def test_batched_fir_matches_scalar_for_both_caps_bias_and_masks(
    condition_cap,
    physical_bias,
) -> None:
    cases, prior = _all_channel_fir_problem(
        channel_count=17,
        fir_length=3,
        physical_bias=physical_bias,
    )
    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=0.7,
        effect_trust_radius=0.5,
    )

    actual = solve_condition_capped_batched_all_channel_fir_ridge_candidate(
        tuple(reversed(cases)),
        condition_cap=condition_cap,
        prior_parameters=prior,
        physical_bias_coordinate=physical_bias,
        policy=policy,
    )
    expected = _scalar_all_channel_fir_oracle(
        cases,
        prior,
        condition_cap=condition_cap,
        policy=policy,
    )

    assert actual.verify_authentication()
    assert torch.allclose(actual.parameters, expected, atol=5e-12, rtol=5e-12)
    assert (
        float((actual.parameters - expected).abs().max().item())
        < 5e-12
    )
    report = actual.diagnostics
    assert report["case_balanced"] is True
    assert report["mask_scope"] == "case_shared_all_channels"
    assert report["channel_count"] == 17
    assert report["fir_length"] == 3
    assert report["physical_bias_coordinate_present"] is physical_bias
    assert report["condition_cap"] == condition_cap
    assert report["condition_cap_passed"] is True
    assert report["parameter_space_trust"]["passed"] is True
    assert report["function_space_trust"]["passed"] is True
    assert report["scalar_oracle_policy"]["deterministic_indices"] == [
        0,
        8,
        16,
    ]
    assert report["scalar_oracle_policy"]["oracle_results_installed"] is True
    assert report["execution_device"] == "cpu"
    assert report["execution_dtype"] == "float64"


def test_batched_fir_degenerate_threshold_and_tight_trust_fallback() -> None:
    generator = torch.Generator().manual_seed(20260720)
    channel_count = 7
    row_count = 41
    feature_width = 4
    design = torch.randn(
        (channel_count, row_count, feature_width),
        generator=generator,
        dtype=torch.float64,
    )
    design[0] = 0.0
    design[1, :, 3] = design[1, :, 0] * 1e-12
    design[2, :, 2] = design[2, :, 1]
    prior = torch.randn(
        (channel_count, feature_width),
        generator=generator,
        dtype=torch.float64,
    )
    target = torch.randn(
        (channel_count, row_count),
        generator=generator,
        dtype=torch.float64,
    ) * 100.0
    mask = torch.ones(row_count, dtype=torch.bool)
    mask[::7] = False
    case = Lfm2CpcfV5AllChannelFirRidgeCase(
        "fit-degenerate",
        "fit",
        design,
        target,
        mask,
    )
    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=0.01,
        effect_trust_radius=0.02,
    )

    actual = solve_condition_capped_batched_all_channel_fir_ridge_candidate(
        (case,),
        condition_cap=1_000.0,
        prior_parameters=prior,
        physical_bias_coordinate=True,
        policy=policy,
    )
    expected = _scalar_all_channel_fir_oracle(
        (case,),
        prior,
        condition_cap=1_000.0,
        policy=policy,
    )

    assert actual.verify_authentication()
    assert torch.allclose(actual.parameters, expected, atol=5e-11, rtol=5e-11)
    ambiguous = actual.diagnostics["scalar_oracle_policy"][
        "threshold_ambiguous_indices"
    ]
    assert 0 in ambiguous
    assert set((0, channel_count // 2, channel_count - 1)) <= set(
        actual.diagnostics["scalar_oracle_policy"]["oracle_channel_indices"]
    )
    assert torch.equal(actual.parameters[0], prior[0])
    assert actual.diagnostics["effective_rank_minimum"] == 0
    assert actual.diagnostics["nullspace_preserves_prior"] is True
    assert actual.diagnostics["parameter_space_trust"]["passed"] is True
    assert actual.diagnostics["function_space_trust"]["passed"] is True


def test_batched_fir_receipt_and_solution_tamper_fail_authentication() -> None:
    cases, prior = _all_channel_fir_problem(
        channel_count=9,
        fir_length=3,
        physical_bias=True,
    )
    result = solve_condition_capped_batched_all_channel_fir_ridge_candidate(
        cases,
        condition_cap=10_000.0,
        prior_parameters=prior,
        physical_bias_coordinate=True,
    )

    assert result.verify_authentication()
    tampered_report = dict(result.diagnostics)
    tampered_report["channel_count"] += 1
    assert not replace(result, diagnostics=tampered_report).verify_authentication()
    tampered_parameters = result.parameters.clone()
    tampered_parameters[1, 1] += 1e-4
    assert not replace(
        result,
        parameters=tampered_parameters,
    ).verify_authentication()


def test_batched_fir_production_shape_shadow_preserves_target_dtype_decisions() -> None:
    cases, prior = _all_channel_fir_problem(
        channel_count=64,
        fir_length=3,
        physical_bias=True,
        seed=20260721,
    )
    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=0.8,
        effect_trust_radius=0.6,
    )
    batched = solve_condition_capped_batched_all_channel_fir_ridge_candidate(
        cases,
        condition_cap=1_000.0,
        prior_parameters=prior,
        physical_bias_coordinate=True,
        policy=policy,
    )
    scalar = _scalar_all_channel_fir_oracle(
        cases,
        prior,
        condition_cap=1_000.0,
        policy=policy,
    )

    assert torch.allclose(batched.parameters, scalar, atol=5e-12, rtol=5e-12)
    assert torch.equal(
        batched.parameters.to(dtype=torch.float32),
        scalar.to(dtype=torch.float32),
    )
    assert torch.equal(
        batched.parameters.to(dtype=torch.bfloat16),
        scalar.to(dtype=torch.bfloat16),
    )


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


def _legacy_balanced_farthest_first(
    descriptors: torch.Tensor,
    *,
    target_channels: int,
) -> tuple[list[int], list[list[int]], list[int]]:
    source_channels = int(descriptors.shape[0])
    centroid = descriptors.mean(dim=0)
    first_seed = min(
        range(source_channels),
        key=lambda index: (
            -float((descriptors[index] - centroid).square().sum().item()),
            index,
        ),
    )
    seeds = [first_seed]
    while len(seeds) < target_channels:
        remaining = [index for index in range(source_channels) if index not in seeds]
        seeds.append(
            min(
                remaining,
                key=lambda index: (
                    -min(
                        float(
                            (descriptors[index] - descriptors[seed])
                            .square()
                            .sum()
                            .item()
                        )
                        for seed in seeds
                    ),
                    index,
                ),
            )
        )
    minimum_group_size, larger_groups = divmod(
        source_channels,
        target_channels,
    )
    capacities = [
        minimum_group_size + int(group < larger_groups)
        for group in range(target_channels)
    ]
    groups = [[seed] for seed in seeds]
    assignment = [-1] * source_channels
    for group, seed in enumerate(seeds):
        assignment[seed] = group
    remaining = [index for index in range(source_channels) if assignment[index] < 0]
    remaining.sort(
        key=lambda index: (
            -min(
                float((descriptors[index] - descriptors[seed]).square().sum().item())
                for seed in seeds
            ),
            index,
        )
    )
    for source in remaining:
        available = [
            group
            for group in range(target_channels)
            if len(groups[group]) < capacities[group]
        ]
        group = min(
            available,
            key=lambda index: (
                float(
                    (descriptors[source] - descriptors[seeds[index]])
                    .square()
                    .sum()
                    .item()
                ),
                index,
            ),
        )
        groups[group].append(source)
        assignment[source] = group
    return seeds, groups, assignment


def test_seed_distance_table_matches_legacy_partition_and_columns() -> None:
    generator = torch.Generator().manual_seed(20260719)
    blocks = (
        torch.randn((17, 3), generator=generator, dtype=torch.float64),
        torch.randn((17, 11), generator=generator, dtype=torch.float64),
        torch.randn((17, 29), generator=generator, dtype=torch.float64),
    )
    descriptors = torch.cat(blocks, dim=1).contiguous()
    expected = _legacy_balanced_farthest_first(
        descriptors,
        target_channels=9,
    )
    actual = v5_module._balanced_farthest_first_with_seed_distance_table(
        descriptors,
        target_channels=9,
    )
    assert actual[:3] == expected
    seed_distance_table = actual[4]
    for group, seed in enumerate(actual[0]):
        direct = torch.stack(
            tuple(
                (descriptors[source] - descriptors[seed]).square().sum()
                for source in range(17)
            )
        )
        assert torch.equal(seed_distance_table[:, group], direct)


def test_seed_distance_table_preserves_lowest_index_ties() -> None:
    descriptors = torch.zeros((8, 6), dtype=torch.float64)
    seeds, groups, assignment, capacity_lookup_count, seed_distance_table = (
        v5_module._balanced_farthest_first_with_seed_distance_table(
            descriptors,
            target_channels=3,
        )
    )
    assert seeds == [0, 1, 2]
    assert groups == [[0, 3, 4], [1, 5, 6], [2, 7]]
    assert assignment == [0, 1, 2, 0, 0, 1, 1, 2]
    assert capacity_lookup_count == 11
    assert torch.count_nonzero(seed_distance_table).item() == 0


def test_seed_distance_table_preserves_equilibrated_near_duplicate_partition() -> None:
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(
        (8, 16_384),
        generator=generator,
        dtype=torch.float64,
    )
    raw[1].copy_(raw[0])
    raw[1, 0] += 1e-12
    descriptors = v5_module._equilibrate_descriptor_block(raw)

    expected = _legacy_balanced_farthest_first(
        descriptors,
        target_channels=2,
    )
    actual = v5_module._balanced_farthest_first_with_seed_distance_table(
        descriptors,
        target_channels=2,
    )

    assert actual[:3] == expected
    assert actual[0] == [7, 2]
    assert actual[2] == [0, 1, 1, 1, 0, 0, 1, 0]


def test_direct_seed_distances_preserve_exact_nonzero_duplicates() -> None:
    generator = torch.Generator().manual_seed(19)
    raw = torch.randn(
        (8, 4_096),
        generator=generator,
        dtype=torch.float64,
    )
    raw[1].copy_(raw[0])
    descriptors = v5_module._equilibrate_descriptor_block(raw)

    distances = v5_module._direct_seed_squared_distances(
        descriptors,
        seed_index=0,
    )
    scalar_legacy = torch.stack(
        tuple(
            (descriptors[source] - descriptors[0]).square().sum()
            for source in range(len(descriptors))
        )
    )

    assert torch.equal(distances, scalar_legacy)
    assert distances[0].item() == 0.0
    assert distances[1].item() == 0.0
    assert torch.all(distances >= 0.0)


def test_direct_seed_distance_chunks_preserve_partition_with_fp64_tolerance() -> None:
    generator = torch.Generator().manual_seed(20_260_719)
    descriptors = torch.randn(
        (129, 4_096),
        generator=generator,
        dtype=torch.float64,
    )
    target_channels = 8

    expected = _legacy_balanced_farthest_first(
        descriptors,
        target_channels=target_channels,
    )
    actual = v5_module._balanced_farthest_first_with_seed_distance_table(
        descriptors,
        target_channels=target_channels,
    )

    assert actual[:3] == expected
    seed_distances = actual[4]
    for group, seed in enumerate(actual[0]):
        scalar = torch.stack(
            tuple(
                (descriptors[source] - descriptors[seed]).square().sum()
                for source in range(len(descriptors))
            )
        )
        torch.testing.assert_close(
            seed_distances[:, group],
            scalar,
            rtol=8.0 * torch.finfo(torch.float64).eps,
            atol=0.0,
        )


def test_direct_seed_distances_preserve_permuted_equal_norm_lowest_tie() -> None:
    width = 16_384
    generator = torch.Generator().manual_seed(20_260_719)
    signs = torch.where(
        torch.rand(width, generator=generator) > 0.5,
        1.0,
        -1.0,
    ).to(dtype=torch.float64)
    first = signs[torch.randperm(width, generator=generator)]
    second = signs[torch.randperm(width, generator=generator)]
    descriptors = torch.stack(
        (
            torch.zeros(width, dtype=torch.float64),
            first,
            second,
        )
    )

    distances = v5_module._direct_seed_squared_distances(
        descriptors,
        seed_index=0,
    )
    scalar_legacy = torch.stack(
        tuple(
            (descriptors[source] - descriptors[0]).square().sum()
            for source in range(len(descriptors))
        )
    )

    assert torch.equal(distances, scalar_legacy)
    assert distances.tolist() == [0.0, float(width), float(width)]
    assert int(torch.argmax(distances).item()) == 1


def test_production_basis_distance_complexity_receipt() -> None:
    source_channels = 1024
    target_channels = 776
    fit_case_count = 154
    descriptor_width = 4 + fit_case_count * (10 + 2 * target_channels)
    remaining = source_channels - target_channels
    receipt = v5_module._seed_distance_table_complexity_receipt(
        source_channels=source_channels,
        target_channels=target_channels,
        descriptor_width=descriptor_width,
        descriptor_block_count=1 + 2 * fit_case_count,
        capacity_assignment_distance_lookups=(remaining * (remaining + 1) // 2),
    )
    assert receipt["legacy_seed_vector_reduction_count"] == 152_454_900
    assert receipt["legacy_total_vector_reduction_count"] == 152_678_224
    assert receipt["seed_distance_table_entries"] == 794_624
    assert receipt["seed_distance_table_bytes"] == 6_356_992
    assert receipt["maximum_direct_seed_temporary_bytes"] == 246_325_248
    assert receipt["direct_seed_column_count"] == 776
    assert receipt["direct_seed_chunk_reduction_count"] == 6_208
    assert receipt["direct_seed_vector_reduction_count"] == 794_624
    assert receipt["distance_row_chunk_size"] == 128
    assert receipt["maximum_rows_per_distance_chunk"] == 128
    assert receipt["distance_chunks_per_seed"] == 8
    assert receipt["direct_seed_temporary_reused_across_rows"] is True
    assert receipt["rowwise_direct_squared_euclidean_objective_preserved"] is True
    assert receipt["scalar_legacy_reduction_semantics_preserved"] is False
    assert (
        receipt["finite_precision_reduction_association"]
        == "fixed_128_row_batched_reduction"
    )
    assert receipt["full_source_by_descriptor_temporary_materialized"] is False
    assert receipt["descriptor_width"] == 240_552
    assert (
        receipt["legacy_descriptor_coordinate_reductions"]
        > 190 * receipt["direct_seed_descriptor_coordinate_reductions"]
    )
    assert receipt["gram_identity_used"] is False
    assert receipt["negative_distance_clamping_used"] is False
    assert receipt["folding_objective_changed"] is False


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
    distance_receipt = first.report["seed_distance_table"]
    assert distance_receipt["seed_distance_table_entries"] == 24
    assert distance_receipt["seed_distance_table_bytes"] == 192
    assert distance_receipt["direct_seed_column_count"] == 3
    assert distance_receipt["direct_seed_vector_reduction_count"] == 24
    assert distance_receipt["gram_identity_used"] is False
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
