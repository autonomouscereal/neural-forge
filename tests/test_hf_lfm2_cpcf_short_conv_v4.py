from __future__ import annotations

from typing import Any

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_short_conv_v4 as short_conv_v4_module  # noqa: E402
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (  # noqa: E402
    LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
    Lfm2CpcfShortConvV4Error,
    Lfm2CpcfShortConvV4SolverPolicy,
    Lfm2CpcfV4BasisCase,
    Lfm2CpcfV4EvaluationCase,
    Lfm2CpcfV4ProjectionCase,
    Lfm2CpcfV4RidgeCase,
    affine_penalty_mask,
    build_signed_endpoint_aware_channel_basis,
    case_balanced_trace_scaled_prior_ridge,
    collapse_endpoint_context_sufficient_statistics,
    evaluate_candidate_state_target_dtype,
    evaluate_target_dtype_predictions,
    fir_penalty_mask,
    fixed_variable_projection_schedule,
    rank_short_conv_v4_selection_candidates,
    replay_sampled_cp_volterra,
    run_fixed_sampled_cp_variable_projection,
    sampled_cp_volterra_objective,
    transport_parent_short_conv_priors,
)


def _policy(**overrides: Any) -> Lfm2CpcfShortConvV4SolverPolicy:
    values = {
        "ridge_alphas": (1e-8, 1e-6, 1e-4),
        "predictor_ridge_alpha": 1e-10,
        "rank_relative_tolerance": 1e-10,
        "participation_relative_tolerance": 1e-12,
        "variable_projection_passes": 2,
        "writer_trust_radius": 8.0,
        "fir_trust_radius": 8.0,
        "bcx_trust_radius": 1.0,
        "correction_rms_floor": 1e-8,
        "target_dtype": "bfloat16",
    }
    values.update(overrides)
    return Lfm2CpcfShortConvV4SolverPolicy(**values)


def _ridge_cases(
    design_a: Any,
    target_a: Any,
    design_b: Any,
    target_b: Any,
) -> tuple[Lfm2CpcfV4RidgeCase, ...]:
    return (
        Lfm2CpcfV4RidgeCase("trajectory-a", "fit", design_a, target_a),
        Lfm2CpcfV4RidgeCase("trajectory-b", "fit", design_b, target_b),
    )


def test_authenticated_policy_and_case_balanced_ridge_invariances() -> None:
    policy = _policy()
    assert policy.verify_authentication()
    assert policy.as_dict()["schema"] == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
    with pytest.raises(
        Lfm2CpcfShortConvV4Error,
        match="authentication_sha256",
    ):
        Lfm2CpcfShortConvV4SolverPolicy(authentication_sha256="0" * 64)
    with pytest.raises(
        Lfm2CpcfShortConvV4Error,
        match="context_objective_weight",
    ):
        _policy(context_objective_weight=0.1000001)

    generator = torch.Generator().manual_seed(1101)
    design_a = torch.randn((23, 3), generator=generator, dtype=torch.float64)
    design_b = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    coefficient = torch.tensor(
        [[1.2, -0.4], [0.3, 0.8], [-0.7, 0.2]],
        dtype=torch.float64,
    )
    target_a = design_a @ coefficient
    target_b = design_b @ coefficient
    prior = torch.randn((3, 2), generator=generator, dtype=torch.float64)
    base = case_balanced_trace_scaled_prior_ridge(
        _ridge_cases(design_a, target_a, design_b, target_b),
        prior=prior,
        alpha=0.07,
        penalty_mask=torch.ones(3, dtype=torch.float64),
    )
    scale = 91.0
    scaled = case_balanced_trace_scaled_prior_ridge(
        _ridge_cases(
            scale * design_a,
            scale * target_a,
            scale * design_b,
            scale * target_b,
        ),
        prior=prior,
        alpha=0.07,
        penalty_mask=torch.ones(3, dtype=torch.float64),
    )
    duplicated = case_balanced_trace_scaled_prior_ridge(
        _ridge_cases(
            design_a.repeat((5, 1)),
            target_a.repeat((5, 1)),
            design_b.repeat((3, 1)),
            target_b.repeat((3, 1)),
        ),
        prior=prior,
        alpha=0.07,
        penalty_mask=torch.ones(3, dtype=torch.float64),
    )
    assert torch.allclose(base.solution, scaled.solution, atol=2e-11, rtol=2e-11)
    assert torch.allclose(
        base.solution,
        duplicated.solution,
        atol=2e-11,
        rtol=2e-11,
    )
    assert base.diagnostics["case_balanced"] is True
    assert base.diagnostics["row_duplication_invariant_objective"] is True
    assert base.diagnostics["effective_ridge"] > 0.0


def test_affine_and_fir_biases_are_unpenalized() -> None:
    x_value = torch.linspace(-2.0, 2.0, 51, dtype=torch.float64).reshape(-1, 1)
    design = torch.cat((x_value, torch.ones_like(x_value)), dim=1)
    target = 2.0 * x_value + 7.0
    cases = (Lfm2CpcfV4RidgeCase("bias-fit", "fit", design, target),)
    first = case_balanced_trace_scaled_prior_ridge(
        cases,
        prior=torch.tensor([[2.0], [-1000.0]], dtype=torch.float64),
        alpha=1e8,
        penalty_mask=affine_penalty_mask(1, fit_bias=True),
    )
    second = case_balanced_trace_scaled_prior_ridge(
        cases,
        prior=torch.tensor([[2.0], [1000.0]], dtype=torch.float64),
        alpha=1e8,
        penalty_mask=affine_penalty_mask(1, fit_bias=True),
    )
    assert first.solution[0, 0].item() == pytest.approx(2.0, abs=1e-11)
    assert first.solution[1, 0].item() == pytest.approx(7.0, abs=1e-10)
    assert torch.allclose(first.solution, second.solution, atol=1e-10, rtol=0.0)
    assert first.diagnostics["unpenalized_columns"] == [1]
    assert affine_penalty_mask(3, fit_bias=True).tolist() == [1.0, 1.0, 1.0, 0.0]
    assert fir_penalty_mask(3, fit_bias=True).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_rank_deficient_basis_is_deterministic_and_completed() -> None:
    sample = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)
    child = torch.stack((sample, sample), dim=1)
    mixing = torch.tensor(
        [
            [1.0, -0.8, 0.6, -0.4, 0.3, -0.2, 0.1, -0.05],
            [1.0, -0.8, 0.6, -0.4, 0.3, -0.2, 0.1, -0.05],
        ],
        dtype=torch.float64,
    )
    parent = child @ mixing
    endpoint = sample.reshape(-1, 1)
    case = Lfm2CpcfV4BasisCase(
        "rank-one-fit",
        "fit",
        child,
        parent,
        endpoint,
    )
    first = build_signed_endpoint_aware_channel_basis(
        (case,),
        target_channels=2,
        policy=_policy(),
    )
    second = build_signed_endpoint_aware_channel_basis(
        (case,),
        target_channels=2,
        policy=_policy(),
    )
    assert torch.equal(first.encoder, second.encoder)
    assert torch.equal(first.decoder, second.decoder)
    assert torch.allclose(
        first.decoder @ first.encoder,
        torch.eye(2, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert first.report["balanced_endpoint_effective_rank"] == 1
    assert first.report["endpoint_basis_columns"] == 1
    assert len(first.report["rank_completion_columns"]) == 1
    assert first.report["validation_cases_consumed"] == 0


def test_signed_endpoint_basis_preserves_cancellation_positive_fold_loses() -> None:
    angle = torch.arange(128, dtype=torch.float64) * (2.0 * torch.pi / 128)
    signal = torch.sin(angle)
    orthogonal = torch.cos(angle)
    child = torch.stack((signal, orthogonal), dim=1)
    parent = torch.stack(
        (signal, -signal, orthogonal, -orthogonal),
        dim=1,
    )
    endpoint = signal.reshape(-1, 1)
    basis = build_signed_endpoint_aware_channel_basis(
        (
            Lfm2CpcfV4BasisCase(
                "signed-cancellation",
                "fit",
                child,
                parent,
                endpoint,
            ),
        ),
        target_channels=1,
        policy=_policy(),
    )
    encoded = parent @ basis.encoder
    encoded_writer = torch.linalg.lstsq(encoded, endpoint).solution
    signed_prediction = encoded @ encoded_writer
    positive_barycenter = parent.mean(dim=1, keepdim=True)
    assert basis.encoder[0, 0] * basis.encoder[1, 0] < 0.0
    assert (
        torch.sqrt(torch.mean((signed_prediction - endpoint).square())).item() < 1e-10
    )
    assert (
        torch.sqrt(torch.mean((positive_barycenter - endpoint).square())).item() > 0.6
    )
    assert basis.report["canonical_svd_signs"] is True
    assert basis.report["balanced_endpoint_cross_covariance_singular_values"][0] > 0.0


def test_eight_to_one_heterogeneous_basis_and_transported_parent_priors() -> None:
    generator = torch.Generator().manual_seed(2201)
    child = torch.randn((192, 2), generator=generator, dtype=torch.float64)
    magnitudes = torch.logspace(-1, 1, 16, dtype=torch.float64)
    signs = torch.where(
        torch.arange(16) % 2 == 0,
        torch.ones(16, dtype=torch.float64),
        -torch.ones(16, dtype=torch.float64),
    )
    mixing = torch.stack(
        (
            magnitudes * signs,
            torch.flip(magnitudes, dims=(0,)) * torch.roll(signs, 1),
        ),
        dim=0,
    )
    parent = child @ mixing
    endpoint = child @ torch.tensor(
        [[1.2, -0.3], [0.4, 0.9]],
        dtype=torch.float64,
    )
    basis = build_signed_endpoint_aware_channel_basis(
        (
            Lfm2CpcfV4BasisCase(
                "heterogeneous-16-to-2",
                "fit",
                child,
                parent,
                endpoint,
            ),
        ),
        target_channels=2,
        policy=_policy(),
    )
    assert basis.report["hidden_width_compression_ratio"] == 8.0
    assert basis.report["full_source_coverage"] is True
    assert basis.report["all_source_channels_consumed"] == list(range(16))
    assert basis.report["full_source_numeric_participation"] is True
    assert len(basis.report["source_participation"]) == 16
    assert len(basis.report["predictability_controllability_spectrum"]) == 16
    assert len(basis.report["endpoint_observability_spectrum"]) == 16

    parent_in = torch.randn((48, 16), generator=generator, dtype=torch.float64)
    parent_in_bias = torch.randn((48,), generator=generator, dtype=torch.float64)
    parent_fir = torch.randn((16, 3), generator=generator, dtype=torch.float64)
    parent_fir_bias = torch.randn((16,), generator=generator, dtype=torch.float64)
    parent_out = torch.randn((16, 16), generator=generator, dtype=torch.float64)
    parent_out_bias = torch.randn((16,), generator=generator, dtype=torch.float64)
    priors = transport_parent_short_conv_priors(
        basis=basis,
        source_in_proj_weight=parent_in,
        source_in_proj_bias=parent_in_bias,
        source_fir_kernel=parent_fir,
        source_fir_bias=parent_fir_bias,
        source_out_proj_weight=parent_out,
        source_out_proj_bias=parent_out_bias,
        target_fir_length=2,
    )
    assert tuple(priors.in_proj_weight.shape) == (6, 2)
    assert tuple(priors.in_proj_bias.shape) == (6,)
    assert tuple(priors.fir_kernel.shape) == (2, 2)
    assert tuple(priors.fir_bias.shape) == (2,)
    assert tuple(priors.out_proj_weight.shape) == (2, 2)
    assert tuple(priors.out_proj_bias.shape) == (2,)
    assert priors.report["all_parent_tensors_consumed"] is True
    assert priors.report["source_channels_dropped"] == 0
    assert priors.report["source_channels_pruned"] == 0
    assert priors.report["source_channels_quantized"] == 0


def test_target_bfloat16_diagnostics_and_selection_only_ranking() -> None:
    policy = _policy()
    generator = torch.Generator().manual_seed(2901)
    child = torch.randn((1, 6, 2), generator=generator, dtype=torch.float64)
    in_weight = torch.randn((6, 2), generator=generator, dtype=torch.float64) * 0.2
    in_bias = torch.randn((6,), generator=generator, dtype=torch.float64) * 0.05
    kernel = torch.randn((2, 3), generator=generator, dtype=torch.float64) * 0.2
    fir_bias = torch.randn((2,), generator=generator, dtype=torch.float64) * 0.03
    writer = torch.randn((2, 2), generator=generator, dtype=torch.float64) * 0.2
    writer_bias = torch.randn((2,), generator=generator, dtype=torch.float64) * 0.02
    target_replay = replay_sampled_cp_volterra(
        child,
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        fir_kernel=kernel,
        fir_bias=fir_bias,
        writer_weight=writer,
        writer_bias=writer_bias,
        target_dtype="bfloat16",
    )
    case = Lfm2CpcfV4ProjectionCase(
        case_id="selection-shared-evidence",
        split="selection",
        child_input_history=child,
        desired_context_history=target_replay.context_history.to(torch.float64),
        desired_write_suffix=target_replay.write_history[:, 2:, :].to(torch.float64),
        suffix_start=2,
    )
    good = evaluate_candidate_state_target_dtype(
        cases=(case,),
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        fir_kernel=kernel,
        fir_bias=fir_bias,
        writer_weight=writer,
        writer_bias=writer_bias,
        required_split="selection",
        policy=policy,
    )
    bad = evaluate_candidate_state_target_dtype(
        cases=(case,),
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        fir_kernel=kernel,
        fir_bias=fir_bias,
        writer_weight=writer + 0.5,
        writer_bias=writer_bias,
        required_split="selection",
        policy=policy,
    )
    assert good["target_dtype"] == "bfloat16"
    assert good["actual_parameter_and_intermediate_target_dtype_replay"] is True
    assert good["final_output_materialization_only"] is False
    assert (
        good["cases"][0]["actual_parameter_and_intermediate_target_dtype_replay"]
        is True
    )
    ranking = rank_short_conv_v4_selection_candidates(
        {"good": good, "bad": bad},
        policy_sha256=policy.authentication_sha256,
    )
    assert ranking["selected_candidate_id"] == "good"
    assert ranking["identical_candidate_evidence_contract"] is True
    assert ranking["all_candidates_used_actual_target_dtype_replay"] is True
    assert ranking["validation_metrics_consumed"] == 0
    assert ranking["validation_isolation"] is True


def test_validation_isolation_is_enforced_at_every_helper_boundary() -> None:
    policy = _policy()
    values = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    validation_basis_case = Lfm2CpcfV4BasisCase(
        "never-fit-validation",
        "validation",
        values,
        torch.cat((values, values), dim=1),
        values,
    )
    with pytest.raises(Lfm2CpcfShortConvV4Error, match="fit cases only"):
        build_signed_endpoint_aware_channel_basis(
            (validation_basis_case,),
            target_channels=2,
            policy=policy,
        )
    with pytest.raises(Lfm2CpcfShortConvV4Error, match="requires 'fit' cases"):
        case_balanced_trace_scaled_prior_ridge(
            (
                Lfm2CpcfV4RidgeCase(
                    "validation-ridge",
                    "validation",
                    values,
                    values[:, :1],
                ),
            ),
            prior=torch.zeros((2, 1), dtype=torch.float64),
            alpha=1e-6,
            penalty_mask=torch.ones(2, dtype=torch.float64),
            required_split="fit",
        )
    validation_report = evaluate_target_dtype_predictions(
        (
            Lfm2CpcfV4EvaluationCase(
                "validation-evaluation",
                "validation",
                values,
                values,
            ),
        ),
        target_dtype="bfloat16",
        required_split="validation",
        policy_sha256=policy.authentication_sha256,
    )
    with pytest.raises(
        Lfm2CpcfShortConvV4Error,
        match="selection reports only",
    ):
        rank_short_conv_v4_selection_candidates(
            {"illegal": validation_report},
            policy_sha256=policy.authentication_sha256,
        )


def _explicit_volterra_replay(
    child: Any,
    in_weight: Any,
    in_bias: Any,
    kernel: Any,
    fir_bias: Any,
    writer: Any,
    writer_bias: Any,
) -> tuple[Any, Any]:
    projection = child @ in_weight.T + in_bias
    b_value, c_value, x_value = projection.chunk(3, dim=-1)
    batch, sequence, hidden = map(int, b_value.shape)
    context = torch.zeros((batch, sequence, hidden), dtype=torch.float64)
    for batch_index in range(batch):
        for token in range(sequence):
            for channel in range(hidden):
                filtered = fir_bias[channel]
                for lag in range(int(kernel.shape[1])):
                    source_token = token - (int(kernel.shape[1]) - 1 - lag)
                    if source_token >= 0:
                        filtered = filtered + (
                            kernel[channel, lag]
                            * b_value[batch_index, source_token, channel]
                            * x_value[batch_index, source_token, channel]
                        )
                context[batch_index, token, channel] = (
                    c_value[batch_index, token, channel] * filtered
                )
    return context, context @ writer.T + writer_bias


def test_sampled_cp_volterra_replay_and_fixed_bounded_factor_sweeps() -> None:
    generator = torch.Generator().manual_seed(3301)
    child = torch.randn((1, 7, 2), generator=generator, dtype=torch.float64)
    in_weight = torch.randn((6, 2), generator=generator, dtype=torch.float64) * 0.3
    in_bias = torch.randn((6,), generator=generator, dtype=torch.float64) * 0.1
    kernel = torch.tensor(
        [[0.4, -0.2, 0.6], [-0.1, 0.3, 0.2]],
        dtype=torch.float64,
    )
    fir_bias = torch.tensor([0.05, -0.07], dtype=torch.float64)
    writer = torch.tensor(
        [[1.0, -0.4], [0.2, 0.8]],
        dtype=torch.float64,
    )
    writer_bias = torch.tensor([0.03, -0.02], dtype=torch.float64)
    explicit_context, explicit_write = _explicit_volterra_replay(
        child,
        in_weight,
        in_bias,
        kernel,
        fir_bias,
        writer,
        writer_bias,
    )
    replay = replay_sampled_cp_volterra(
        child,
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        fir_kernel=kernel,
        fir_bias=fir_bias,
        writer_weight=writer,
        writer_bias=writer_bias,
    )
    assert torch.allclose(replay.context_history, explicit_context, atol=1e-12)
    assert torch.allclose(replay.write_history, explicit_write, atol=1e-12)

    suffix_start = 3
    case = Lfm2CpcfV4ProjectionCase(
        case_id="exact-sampled-cp",
        split="fit",
        child_input_history=child,
        desired_context_history=explicit_context,
        desired_write_suffix=explicit_write[:, suffix_start:, :],
        suffix_start=suffix_start,
    )
    policy = _policy(variable_projection_passes=2)
    result = run_fixed_sampled_cp_variable_projection(
        cases=(case,),
        initial_in_proj_weight=in_weight,
        initial_in_proj_bias=in_bias,
        initial_fir_kernel=kernel,
        initial_fir_bias=fir_bias,
        initial_writer_weight=writer,
        initial_writer_bias=writer_bias,
        alpha=1e-8,
        policy=policy,
    )
    assert result.report["passes"] == 2
    assert result.report["coupled_cp_rank"] == 2
    assert result.report["huge_volterra_tensor_materialized"] is False
    assert result.report["operator_objective"].endswith("lagged_cubic_coupled_cp")
    assert len(result.report["schedule"]) == 6
    assert (
        fixed_variable_projection_schedule(policy)
        == (
            "sampled_cp_writer_factor",
            "sampled_cp_fir_factor",
            "sampled_cp_linearized_bcx_factors",
        )
        * 2
    )
    assert result.report["final_objective"]["mean_case_relative_rmse"] < 1e-9
    execution = result.report["execution"]
    assert execution["backward_calls"] == 0
    assert execution["optimizer_steps"] == 0
    assert execution["training_steps"] == 0
    assert execution["parameters_pruned"] == 0
    assert execution["parameters_quantized"] == 0


def test_collapsed_endpoint_statistics_exactly_match_expanded_output_axis() -> None:
    generator = torch.Generator().manual_seed(4401)
    prior = torch.randn((5, 1), generator=generator, dtype=torch.float64)
    expanded_cases = []
    collapsed_cases = []
    for index, (rows, outputs) in enumerate(((11, 7), (5, 7))):
        design = torch.randn(
            (1, rows, 5),
            generator=generator,
            dtype=torch.float64,
        )
        residual = torch.randn(
            (1, rows, outputs),
            generator=generator,
            dtype=torch.float64,
        )
        writer = torch.randn((outputs,), generator=generator, dtype=torch.float64)
        mask = torch.ones((1, rows), dtype=torch.bool)
        mask[:, index + 1] = False
        collapsed, receipt = collapse_endpoint_context_sufficient_statistics(
            case_id=f"case-{index}",
            split="fit",
            context_design=design,
            endpoint_residual=residual,
            writer_column=writer,
            valid_rows=mask,
        )
        endpoint_design = design.unsqueeze(2) * writer.reshape(1, 1, -1, 1)
        expanded_cases.append(
            Lfm2CpcfV4RidgeCase(
                case_id=f"case-{index}",
                split="fit",
                design=endpoint_design.reshape(-1, 5),
                target=residual.reshape(-1, 1),
                valid_rows=mask.unsqueeze(-1).expand(-1, -1, outputs).reshape(-1),
            )
        )
        collapsed_cases.append(collapsed)
        assert receipt["exact_per_case_mean_gram_and_right_preserved"] is True
        assert receipt["expanded_rows_avoided"] == int(mask.sum()) * (outputs - 1)
    expanded = case_balanced_trace_scaled_prior_ridge(
        expanded_cases,
        prior=prior,
        alpha=0.03,
        penalty_mask=torch.ones(5, dtype=torch.float64),
    )
    collapsed = case_balanced_trace_scaled_prior_ridge(
        collapsed_cases,
        prior=prior,
        alpha=0.03,
        penalty_mask=torch.ones(5, dtype=torch.float64),
    )
    assert torch.allclose(expanded.solution, collapsed.solution, atol=2e-11, rtol=2e-11)
    assert expanded.diagnostics["gram_trace"] == pytest.approx(
        collapsed.diagnostics["gram_trace"],
        rel=2e-12,
    )

    zero_case, zero_receipt = collapse_endpoint_context_sufficient_statistics(
        case_id="zero-writer",
        split="fit",
        context_design=torch.randn((1, 4, 3), generator=generator),
        endpoint_residual=torch.randn((1, 4, 6), generator=generator),
        writer_column=torch.zeros(6),
        valid_rows=torch.ones((1, 4), dtype=torch.bool),
    )
    assert zero_receipt["zero_information_noop"] is True
    assert torch.count_nonzero(zero_case.design) == 0
    assert torch.count_nonzero(zero_case.target) == 0


def test_collapsed_statistics_scale_to_256_target_128_trajectories() -> None:
    generator = torch.Generator().manual_seed(4451)
    target_hidden = 128
    output_width = 128
    design_width = 3 * (target_hidden + 1)
    writer = torch.randn((output_width,), generator=generator, dtype=torch.float64)
    cases = []
    expanded_rows_avoided = 0
    for index in range(256):
        collapsed, receipt = collapse_endpoint_context_sufficient_statistics(
            case_id=f"trajectory-{index:03d}",
            split="fit",
            context_design=torch.randn(
                (1, 1, design_width),
                generator=generator,
                dtype=torch.float64,
            ),
            endpoint_residual=torch.randn(
                (1, 1, output_width),
                generator=generator,
                dtype=torch.float64,
            ),
            writer_column=writer,
            valid_rows=torch.ones((1, 1), dtype=torch.bool),
        )
        cases.append(collapsed)
        expanded_rows_avoided += int(receipt["expanded_rows_avoided"])
        assert tuple(collapsed.design.shape) == (1, design_width)
    result = case_balanced_trace_scaled_prior_ridge(
        cases,
        prior=torch.zeros((design_width, 1), dtype=torch.float64),
        alpha=1e-4,
        penalty_mask=torch.ones(design_width, dtype=torch.float64),
    )
    assert tuple(result.solution.shape) == (design_width, 1)
    assert torch.isfinite(result.solution).all()
    assert result.diagnostics["case_count"] == 256
    assert expanded_rows_avoided == 256 * (output_width - 1)


def test_ridge_penalized_trace_duplicate_ids_and_mutation_fail_closed() -> None:
    x_value = torch.linspace(-1e-4, 1e-4, 17, dtype=torch.float64).reshape(-1, 1)
    design = torch.cat((x_value, torch.ones_like(x_value)), dim=1)
    case = Lfm2CpcfV4RidgeCase("authenticated", "fit", design, 3.0 * x_value)
    result = case_balanced_trace_scaled_prior_ridge(
        (case,),
        prior=torch.zeros((2, 1), dtype=torch.float64),
        alpha=0.1,
        penalty_mask=affine_penalty_mask(1, fit_bias=True),
    )
    assert result.diagnostics["gram_trace"] == pytest.approx(
        float(x_value.square().mean()),
    )
    assert result.diagnostics["total_gram_trace"] == pytest.approx(
        1.0 + float(x_value.square().mean()),
    )
    duplicate = Lfm2CpcfV4RidgeCase("authenticated", "fit", design, 3.0 * x_value)
    with pytest.raises(Lfm2CpcfShortConvV4Error, match="unique"):
        case_balanced_trace_scaled_prior_ridge(
            (case, duplicate),
            prior=torch.zeros((2, 1), dtype=torch.float64),
            alpha=0.1,
            penalty_mask=affine_penalty_mask(1, fit_bias=True),
        )
    case.design.add_(1.0)
    with pytest.raises(Lfm2CpcfShortConvV4Error, match="changed"):
        case_balanced_trace_scaled_prior_ridge(
            (case,),
            prior=torch.zeros((2, 1), dtype=torch.float64),
            alpha=0.1,
            penalty_mask=affine_penalty_mask(1, fit_bias=True),
        )


def test_affine_input_lift_bias_transport_is_algebraically_oriented() -> None:
    generator = torch.Generator().manual_seed(5501)
    child = torch.randn((96, 2), generator=generator, dtype=torch.float64)
    parent = child @ torch.randn((2, 3), generator=generator, dtype=torch.float64)
    endpoint = child @ torch.randn((2, 2), generator=generator, dtype=torch.float64)
    basis = build_signed_endpoint_aware_channel_basis(
        (Lfm2CpcfV4BasisCase("transport-fit", "fit", child, parent, endpoint),),
        target_channels=2,
        policy=_policy(),
    )
    source_in = torch.randn((9, 3), generator=generator, dtype=torch.float64)
    source_bias = torch.randn((9,), generator=generator, dtype=torch.float64)
    source_fir = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    source_out = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    lift = torch.randn((2, 3), generator=generator, dtype=torch.float64)
    lift_bias = torch.randn((3,), generator=generator, dtype=torch.float64)
    priors = transport_parent_short_conv_priors(
        basis=basis,
        source_in_proj_weight=source_in,
        source_in_proj_bias=source_bias,
        source_fir_kernel=source_fir,
        source_out_proj_weight=source_out,
        target_fir_length=3,
        input_lift=lift,
        input_lift_bias=lift_bias,
    )
    hidden = torch.randn((31, 2), generator=generator, dtype=torch.float64)
    source_hidden = hidden @ lift + lift_bias
    direct_roles = []
    prior_roles = []
    for role_index, role in enumerate(source_in.reshape(3, 3, 3)):
        direct_roles.append(
            (source_hidden @ role.T + source_bias.reshape(3, 3)[role_index])
            @ basis.encoder
        )
        target_role = priors.in_proj_weight.reshape(3, 2, 2)[role_index]
        target_bias = priors.in_proj_bias.reshape(3, 2)[role_index]
        prior_roles.append(hidden @ target_role.T + target_bias)
    assert torch.allclose(torch.cat(direct_roles, dim=1), torch.cat(prior_roles, dim=1))
    expected_fir = []
    for lag in range(3):
        projected = basis.decoder @ (source_fir[:, lag].reshape(-1, 1) * basis.encoder)
        expected_fir.append(torch.diag(projected))
    assert torch.allclose(priors.fir_kernel, torch.stack(expected_fir, dim=1))
    assert (
        priors.report["fir_diagonal_projection_is_linear_surrogate_for_cubic_operator"]
        is True
    )
    assert priors.verify_authentication()


def test_basis_numeric_participation_and_mutation_are_honest() -> None:
    sample = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64)
    child = sample.reshape(-1, 1)
    parent = torch.cat(
        (sample.reshape(-1, 1), torch.zeros((32, 3), dtype=torch.float64)),
        dim=1,
    )
    basis = build_signed_endpoint_aware_channel_basis(
        (Lfm2CpcfV4BasisCase("sparse-source", "fit", child, parent, child),),
        target_channels=1,
        policy=_policy(),
    )
    assert basis.report["numeric_participating_source_indices"] == [0]
    assert basis.report["full_source_coverage"] is False
    assert basis.report["source_channels_dropped"] == 3
    assert basis.report["execution"]["deterministic_svd"] is False
    assert basis.report["degenerate_svd_subspace_rotation_canonicalized"] is False
    basis.encoder.zero_()
    with pytest.raises(Lfm2CpcfShortConvV4Error, match="changed"):
        transport_parent_short_conv_priors(
            basis=basis,
            source_in_proj_weight=torch.eye(4, dtype=torch.float64).repeat(3, 1),
            source_fir_kernel=torch.ones((4, 2), dtype=torch.float64),
            source_out_proj_weight=torch.eye(4, dtype=torch.float64),
            target_fir_length=2,
        )


def test_actual_bfloat16_intermediates_differ_from_final_output_cast() -> None:
    generator = torch.Generator().manual_seed(6601)
    child = torch.randn((2, 19, 3), generator=generator, dtype=torch.float64) * 3.0
    in_weight = torch.randn((9, 3), generator=generator, dtype=torch.float64) * 2.0
    in_bias = torch.randn((9,), generator=generator, dtype=torch.float64)
    kernel = torch.randn((3, 5), generator=generator, dtype=torch.float64)
    fir_bias = torch.randn((3,), generator=generator, dtype=torch.float64)
    writer = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    writer_bias = torch.randn((3,), generator=generator, dtype=torch.float64)
    fp64 = replay_sampled_cp_volterra(
        child,
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        fir_kernel=kernel,
        fir_bias=fir_bias,
        writer_weight=writer,
        writer_bias=writer_bias,
    )
    actual = replay_sampled_cp_volterra(
        child,
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        fir_kernel=kernel,
        fir_bias=fir_bias,
        writer_weight=writer,
        writer_bias=writer_bias,
        target_dtype="bfloat16",
    )
    final_cast = fp64.write_history.to(torch.bfloat16)
    assert not torch.equal(actual.write_history, final_cast)
    assert actual.arithmetic_report["intermediate_arithmetic_dtype"] == "bfloat16"
    assert actual.arithmetic_report["final_output_cast_only"] is False
    assert actual.verify_authentication()


def test_selection_enforces_worst_case_and_identical_evidence() -> None:
    policy = _policy()
    cases = tuple(
        Lfm2CpcfV4ProjectionCase(
            case_id=f"shared-{index}",
            split="selection",
            child_input_history=torch.tensor([[[float(index)]]], dtype=torch.float64),
            desired_context_history=torch.ones((1, 1, 1), dtype=torch.float64),
            desired_write_suffix=torch.ones((1, 1, 1), dtype=torch.float64),
            suffix_start=0,
        )
        for index in range(2)
    )

    def report(*, c_weight: float, c_bias: float) -> Any:
        return evaluate_candidate_state_target_dtype(
            cases=cases,
            in_proj_weight=torch.tensor(
                [[0.0], [c_weight], [0.0]],
                dtype=torch.float64,
            ),
            in_proj_bias=torch.tensor(
                [1.0, c_bias, 1.0],
                dtype=torch.float64,
            ),
            fir_kernel=torch.ones((1, 1), dtype=torch.float64),
            fir_bias=None,
            writer_weight=torch.ones((1, 1), dtype=torch.float64),
            writer_bias=None,
            required_split="selection",
            policy=policy,
        )

    spiky = report(c_weight=2.0, c_bias=1.0)
    balanced = report(c_weight=0.0, c_bias=2.0)
    ranking = rank_short_conv_v4_selection_candidates(
        {"spiky": spiky, "balanced": balanced},
        policy_sha256=policy.authentication_sha256,
    )
    assert ranking["selected_candidate_id"] == "balanced"
    mismatched_cases = (
        Lfm2CpcfV4ProjectionCase(
            "different-target",
            "selection",
            torch.zeros((1, 1, 1), dtype=torch.float64),
            torch.ones((1, 1, 1), dtype=torch.float64),
            torch.full((1, 1, 1), 2.0, dtype=torch.float64),
            0,
        ),
    )
    mismatched = evaluate_candidate_state_target_dtype(
        cases=mismatched_cases,
        in_proj_weight=torch.zeros((3, 1), dtype=torch.float64),
        in_proj_bias=torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64),
        fir_kernel=torch.ones((1, 1), dtype=torch.float64),
        fir_bias=None,
        writer_weight=torch.ones((1, 1), dtype=torch.float64),
        writer_bias=None,
        required_split="selection",
        policy=policy,
    )
    with pytest.raises(Lfm2CpcfShortConvV4Error, match="identical evidence"):
        rank_short_conv_v4_selection_candidates(
            {"balanced": balanced, "different": mismatched},
            policy_sha256=policy.authentication_sha256,
        )


def test_prefix_masks_match_lfm2_mask_before_biased_projection() -> None:
    child = torch.tensor(
        [[[1000.0], [1.0], [2.0], [3.0]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[False, True, True, True]])
    parameters = {
        "in_proj_weight": torch.ones((3, 1), dtype=torch.float64),
        "in_proj_bias": torch.ones((3,), dtype=torch.float64),
        "fir_kernel": torch.ones((1, 2), dtype=torch.float64),
        "fir_bias": None,
        "writer_weight": torch.ones((1, 1), dtype=torch.float64),
        "writer_bias": None,
    }
    masked = replay_sampled_cp_volterra(
        child,
        **parameters,
        valid_prefix_mask=mask,
    )
    manually_zeroed = replay_sampled_cp_volterra(
        child * mask.unsqueeze(-1),
        **parameters,
        valid_prefix_mask=mask,
    )
    assert torch.equal(masked.write_history, manually_zeroed.write_history)
    projected = (child * mask.unsqueeze(-1)) @ parameters["in_proj_weight"].T
    projected = projected + parameters["in_proj_bias"]
    expected_b, expected_c, expected_x = projected.chunk(3, dim=-1)
    expected_context = short_conv_v4_module.causal_depthwise_context(
        expected_b,
        expected_c,
        expected_x,
        parameters["fir_kernel"],
        parameters["fir_bias"],
    )
    assert torch.equal(masked.b_history, expected_b)
    assert torch.equal(masked.c_history, expected_c)
    assert torch.equal(masked.x_history, expected_x)
    assert torch.equal(masked.context_history, expected_context)
    assert masked.b_history[0, 0, 0].item() == pytest.approx(1.0)
    assert (
        masked.arithmetic_report["invalid_projected_history_zeroed_after_affine"]
        is False
    )
    assert (
        masked.arithmetic_report["biased_projection_matches_lfm2_mask_before_linear"]
        is True
    )


def test_context_objective_is_live_and_overshoot_phase_reverts(
    monkeypatch: Any,
) -> None:
    generator = torch.Generator().manual_seed(7701)
    child = torch.randn((1, 6, 2), generator=generator, dtype=torch.float64)
    in_weight = torch.randn((6, 2), generator=generator, dtype=torch.float64) * 0.2
    in_bias = torch.zeros(6, dtype=torch.float64)
    kernel = torch.randn((2, 2), generator=generator, dtype=torch.float64) * 0.2
    fir_bias = torch.zeros(2, dtype=torch.float64)
    writer = torch.eye(2, dtype=torch.float64)
    writer_bias = torch.zeros(2, dtype=torch.float64)
    replay = replay_sampled_cp_volterra(
        child,
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        fir_kernel=kernel,
        fir_bias=fir_bias,
        writer_weight=writer,
        writer_bias=writer_bias,
    )
    case = Lfm2CpcfV4ProjectionCase(
        case_id="context-live",
        split="fit",
        child_input_history=child,
        desired_context_history=replay.context_history + 0.25,
        desired_write_suffix=replay.write_history[:, 2:, :],
        suffix_start=2,
    )
    no_context = sampled_cp_volterra_objective(
        (case,),
        (replay,),
        context_objective_weight=0.0,
    )
    with_context = sampled_cp_volterra_objective(
        (case,),
        (replay,),
        context_objective_weight=0.05,
    )
    assert no_context["combined_fit_objective"] == pytest.approx(0.0)
    assert with_context["mean_case_context_relative_rmse"] > 0.0
    assert with_context["combined_fit_objective"] > 0.0

    original = short_conv_v4_module.solve_bounded_linearized_bcx_projection

    def overshoot(**kwargs: Any) -> Any:
        weight, bias, report = original(**kwargs)
        return weight * 1e308, None if bias is None else bias * 1e308, report

    monkeypatch.setattr(
        short_conv_v4_module,
        "solve_bounded_linearized_bcx_projection",
        overshoot,
    )
    result = run_fixed_sampled_cp_variable_projection(
        cases=(case,),
        initial_in_proj_weight=in_weight,
        initial_in_proj_bias=in_bias,
        initial_fir_kernel=kernel,
        initial_fir_bias=fir_bias,
        initial_writer_weight=writer,
        initial_writer_bias=writer_bias,
        alpha=1e-6,
        policy=_policy(variable_projection_passes=2),
    )
    assert all(not sweep["bcx_phase_accepted"] for sweep in result.report["sweeps"])
    assert result.report["final_objective_not_worse_than_initial"] is True
    assert (
        result.report["final_objective"]["combined_fit_objective"]
        <= result.report["initial_objective"]["combined_fit_objective"] + 1e-12
    )
    assert result.verify_authentication()
    result.in_proj_weight.add_(1.0)
    assert not result.verify_authentication()


def test_selection_binds_full_projection_evidence_not_only_targets() -> None:
    policy = _policy()

    def evaluation(child_value: float) -> tuple[Any, str]:
        case = Lfm2CpcfV4ProjectionCase(
            "shared-id",
            "selection",
            torch.tensor([[[child_value]]], dtype=torch.float64),
            torch.ones((1, 1, 1), dtype=torch.float64),
            torch.ones((1, 1, 1), dtype=torch.float64),
            0,
        )
        report = evaluate_candidate_state_target_dtype(
            cases=(case,),
            in_proj_weight=torch.ones((3, 1), dtype=torch.float64),
            in_proj_bias=None,
            fir_kernel=torch.ones((1, 1), dtype=torch.float64),
            fir_bias=None,
            writer_weight=torch.ones((1, 1), dtype=torch.float64),
            writer_bias=None,
            required_split="selection",
            policy=policy,
        )
        return report, case.authentication_sha256

    first, first_projection_sha = evaluation(0.0)
    second, second_projection_sha = evaluation(2.0)
    assert first_projection_sha != second_projection_sha
    assert first["evidence_sha256"] != second["evidence_sha256"]
    with pytest.raises(Lfm2CpcfShortConvV4Error, match="identical evidence"):
        rank_short_conv_v4_selection_candidates(
            {"first": first, "second": second},
            policy_sha256=policy.authentication_sha256,
        )


def test_selection_scores_context_state_and_true_worst_case_before_mean() -> None:
    policy = _policy()
    shared = Lfm2CpcfV4ProjectionCase(
        "context-shared",
        "selection",
        torch.ones((1, 1, 1), dtype=torch.float64),
        torch.ones((1, 1, 1), dtype=torch.float64),
        torch.ones((1, 1, 1), dtype=torch.float64),
        0,
    )

    def context_candidate(c_bias: float, writer: float) -> Any:
        return evaluate_candidate_state_target_dtype(
            cases=(shared,),
            in_proj_weight=torch.zeros((3, 1), dtype=torch.float64),
            in_proj_bias=torch.tensor([1.0, c_bias, 1.0], dtype=torch.float64),
            fir_kernel=torch.ones((1, 1), dtype=torch.float64),
            fir_bias=None,
            writer_weight=torch.tensor([[writer]], dtype=torch.float64),
            writer_bias=None,
            required_split="selection",
            policy=policy,
        )

    bad_context = context_candidate(2.0, 0.5)
    good_context = context_candidate(1.0, 1.0)
    assert bad_context["mean_case_relative_rmse"] == pytest.approx(0.0)
    assert good_context["mean_case_relative_rmse"] == pytest.approx(0.0)
    assert bad_context["maximum_case_context_relative_rmse"] > 0.9
    context_ranking = rank_short_conv_v4_selection_candidates(
        {"a-bad-context": bad_context, "z-good-context": good_context},
        policy_sha256=policy.authentication_sha256,
    )
    assert context_ranking["selected_candidate_id"] == "z-good-context"
    assert context_ranking["context_state_evidence_ranked"] is True

    worst_cases = tuple(
        Lfm2CpcfV4ProjectionCase(
            f"worst-{index:02d}",
            "selection",
            torch.tensor([[[1.0 if index == 9 else 0.0]]], dtype=torch.float64),
            torch.ones((1, 1, 1), dtype=torch.float64),
            torch.ones((1, 1, 1), dtype=torch.float64),
            0,
        )
        for index in range(10)
    )

    def affine_context(c_weight: float, c_bias: float) -> Any:
        return evaluate_candidate_state_target_dtype(
            cases=worst_cases,
            in_proj_weight=torch.tensor(
                [[0.0], [c_weight], [0.0]],
                dtype=torch.float64,
            ),
            in_proj_bias=torch.tensor([1.0, c_bias, 1.0], dtype=torch.float64),
            fir_kernel=torch.ones((1, 1), dtype=torch.float64),
            fir_bias=None,
            writer_weight=torch.ones((1, 1), dtype=torch.float64),
            writer_bias=None,
            required_split="selection",
            policy=policy,
        )

    lower_mean_catastrophe = affine_context(3.0, 1.0)
    uniformly_safe = affine_context(0.0, 1.5)
    assert (
        lower_mean_catastrophe["mean_case_relative_rmse"]
        < uniformly_safe["mean_case_relative_rmse"]
    )
    assert (
        lower_mean_catastrophe["maximum_case_relative_rmse"]
        > uniformly_safe["maximum_case_relative_rmse"]
    )
    worst_ranking = rank_short_conv_v4_selection_candidates(
        {
            "lower-mean-catastrophe": lower_mean_catastrophe,
            "uniformly-safe": uniformly_safe,
        },
        policy_sha256=policy.authentication_sha256,
    )
    assert worst_ranking["selected_candidate_id"] == "uniformly-safe"
    assert worst_ranking["worst_case_metrics_ranked_before_means"] is True


def test_channel_factor_solves_cache_full_replays_and_use_exact_width_one_updates(
    monkeypatch: Any,
) -> None:
    generator = torch.Generator().manual_seed(8801)
    hidden = 4
    sequence = 6
    case_count = 2
    in_weight = (
        torch.randn(
            (3 * hidden, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.1
    )
    in_bias = torch.zeros(3 * hidden, dtype=torch.float64)
    kernel = torch.randn((hidden, 3), generator=generator, dtype=torch.float64) * 0.1
    fir_bias = torch.zeros(hidden, dtype=torch.float64)
    writer = torch.eye(hidden, dtype=torch.float64)
    writer_bias = torch.zeros(hidden, dtype=torch.float64)
    cases = []
    current_bcx = []
    for index in range(case_count):
        child = torch.randn(
            (1, sequence, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        replay = replay_sampled_cp_volterra(
            child,
            in_proj_weight=in_weight,
            in_proj_bias=in_bias,
            fir_kernel=kernel,
            fir_bias=fir_bias,
            writer_weight=writer,
            writer_bias=writer_bias,
        )
        current_bcx.append((replay.b_history, replay.c_history, replay.x_history))
        cases.append(
            Lfm2CpcfV4ProjectionCase(
                f"cache-{index}",
                "fit",
                child,
                replay.context_history + 0.01,
                replay.write_history[:, 2:, :] + 0.01,
                2,
            )
        )
    policy = _policy(variable_projection_passes=2)
    original_replay = short_conv_v4_module.replay_sampled_cp_volterra
    replay_calls = 0

    def counted_replay(*args: Any, **kwargs: Any) -> Any:
        nonlocal replay_calls
        replay_calls += 1
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(
        short_conv_v4_module,
        "replay_sampled_cp_volterra",
        counted_replay,
    )
    _weight, _bias, bcx_report = (
        short_conv_v4_module.solve_bounded_linearized_bcx_projection(
            cases=tuple(cases),
            kernel=kernel,
            conv_bias=fir_bias,
            writer_weight=writer,
            writer_bias=writer_bias,
            current_in_proj_weight=in_weight,
            current_in_proj_bias=in_bias,
            alpha=1e-6,
            policy=policy,
        )
    )
    assert replay_calls == case_count
    assert bcx_report["full_operator_replays_per_channel"] == 0
    assert bcx_report["cached_full_operator_replays"] == case_count

    original_context = short_conv_v4_module.causal_depthwise_context
    context_widths = []

    def counted_context(*args: Any, **kwargs: Any) -> Any:
        context_widths.append(int(args[0].shape[-1]))
        return original_context(*args, **kwargs)

    monkeypatch.setattr(
        short_conv_v4_module,
        "causal_depthwise_context",
        counted_context,
    )
    _kernel, _fir_bias, fir_report = short_conv_v4_module.solve_bounded_fir_projection(
        cases=tuple(cases),
        current_bcx=tuple(current_bcx),
        current_kernel=kernel,
        current_bias=fir_bias,
        writer_weight=writer,
        writer_bias=writer_bias,
        alpha=1e-6,
        policy=policy,
    )
    assert context_widths.count(hidden) == case_count
    assert context_widths.count(1) == 0
    assert fir_report["full_operator_replays_per_channel"] == 0
    assert fir_report["cached_full_context_replays"] == case_count
    assert fir_report["single_channel_state_updates"] == hidden
