from __future__ import annotations

import torch

from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
    causal_depthwise_context,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    Lfm2CpcfShortConvNumericalFrontierRequest,
    Lfm2CpcfShortConvV4FoldCase,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_fhvr_mf import (
    FHVR_MF_CANDIDATE_KIND,
    Lfm2CpcfFhvrMfPolicy,
    build_lfm2_cpcf_fhvr_mf_candidate,
    solve_lfm2_cpcf_fhvr_mf_conditional_factor,
)


def _request() -> tuple[Lfm2CpcfShortConvNumericalFrontierRequest, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260719)
    source_hidden = 4
    target_hidden = 2
    target_fir_length = 2
    source_in_weight = (
        torch.randn(
            (3 * source_hidden, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    )
    source_in_bias = (
        torch.randn(
            (3 * source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.03
    )
    source_fir = torch.tensor(
        [
            [0.8, -0.2, 0.1],
            [0.7, -0.1, 0.2],
            [0.6, 0.1, -0.1],
            [0.5, 0.2, -0.2],
        ],
        dtype=torch.float64,
    )
    source_fir_bias = (
        torch.randn(
            (source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.02
    )
    source_writer = (
        torch.randn(
            (source_hidden, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    )
    source_writer_bias = (
        torch.randn(
            (source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.01
    )
    context_encoder = torch.randn(
        (source_hidden, target_hidden),
        generator=generator,
        dtype=torch.float64,
    )
    child_to_parent = torch.randn(
        (target_hidden, source_hidden),
        generator=generator,
        dtype=torch.float64,
    )
    cases = []
    contexts = {}
    for case_index, case_id in enumerate(("fit-a", "fit-b")):
        batch = 1
        tokens = 8
        suffix_start = 3
        child = torch.randn(
            (batch, tokens, target_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        parent_input = child @ child_to_parent + 0.01 * case_index
        projection = parent_input @ source_in_weight.T + source_in_bias
        parent_b, parent_c, parent_x = projection.chunk(3, dim=-1)
        parent_context = causal_depthwise_context(
            parent_b,
            parent_c,
            parent_x,
            source_fir,
            source_fir_bias,
        )
        parent_write = (
            parent_context @ source_writer.T + source_writer_bias
        ) @ context_encoder
        actual_pre = (
            torch.randn(
                (batch, tokens - suffix_start, target_hidden),
                generator=generator,
                dtype=torch.float64,
            )
            * 0.02
        )
        mask = torch.ones((batch, tokens), dtype=torch.bool)
        if case_index:
            mask[:, -1] = False
        cases.append(
            Lfm2CpcfShortConvV4FoldCase(
                case_id=case_id,
                split="fit",
                suffix_start=suffix_start,
                actual_child_operator_input_history=child,
                parent_operator_input_history=parent_input,
                parent_b_history=parent_b,
                parent_c_history=parent_c,
                parent_x_history=parent_x,
                parent_post_operator_residual_target_space_suffix=(
                    actual_pre + parent_write[:, suffix_start:]
                ),
                actual_child_pre_operator_residual_suffix=actual_pre,
                valid_prefix_mask=mask,
            )
        )
        contexts[case_id] = parent_context
    request = Lfm2CpcfShortConvNumericalFrontierRequest(
        source_tensors={
            "source_in_proj_weight": source_in_weight,
            "source_in_proj_bias": source_in_bias,
            "source_fir_kernel": source_fir,
            "source_fir_bias": source_fir_bias,
            "source_out_proj_weight": source_writer,
            "source_out_proj_bias": source_writer_bias,
            "output_encoder": context_encoder,
        },
        cases=tuple(cases),
        parent_contexts=contexts,
        target_templates={
            "in_proj_weight": torch.zeros(
                (3 * target_hidden, target_hidden),
                dtype=torch.float64,
            ),
            "in_proj_bias": torch.zeros(
                (3 * target_hidden,),
                dtype=torch.float64,
            ),
            "fir_kernel": torch.zeros(
                (target_hidden, target_fir_length),
                dtype=torch.float64,
            ),
            "fir_bias": torch.zeros(
                (target_hidden,),
                dtype=torch.float64,
            ),
            "writer_weight": torch.zeros(
                (target_hidden, target_hidden),
                dtype=torch.float64,
            ),
            "writer_bias": torch.zeros(
                (target_hidden,),
                dtype=torch.float64,
            ),
        },
        policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-6,),
            variable_projection_passes=2,
            target_dtype="float64",
        ),
    )
    return request, context_encoder


def test_conditional_matrix_free_solve_matches_explicit_oracle() -> None:
    generator = torch.Generator().manual_seed(41)
    rows = 37
    width = 5
    channels = 4
    features = torch.randn(
        (rows, width),
        generator=generator,
        dtype=torch.float64,
    )
    multiplier = torch.randn(
        (rows, channels),
        generator=generator,
        dtype=torch.float64,
    )
    target = torch.randn(
        (rows, channels),
        generator=generator,
        dtype=torch.float64,
    )
    row_weight = torch.rand(
        (rows,),
        generator=generator,
        dtype=torch.float64,
    )
    row_weight = row_weight / row_weight.sum()
    prior = torch.randn(
        (channels, width),
        generator=generator,
        dtype=torch.float64,
    )
    policy = Lfm2CpcfFhvrMfPolicy(
        pcg_iterations=64,
        factor_sweeps=1,
        row_chunk_size=7,
        channel_chunk_size=2,
        absolute_ridge_floor=1e-12,
        backend="cpu",
    )
    result = solve_lfm2_cpcf_fhvr_mf_conditional_factor(
        features,
        multiplier,
        target,
        row_weight,
        prior,
        policy=policy,
    )
    ridge = result.diagnostics["effective_ridge_by_channel"]
    oracle = torch.empty_like(prior)
    identity = torch.eye(width, dtype=torch.float64)
    for channel in range(channels):
        design = multiplier[:, channel].unsqueeze(-1) * features
        gram = design.T @ (row_weight.unsqueeze(-1) * design)
        right = design.T @ (row_weight * target[:, channel])
        oracle[channel] = torch.linalg.solve(
            gram + float(ridge[channel]) * identity,
            right + float(ridge[channel]) * prior[channel],
        )
    assert result.verify_authentication()
    assert torch.allclose(result.weight, oracle, atol=1e-9, rtol=1e-9)
    assert result.diagnostics["allocation_audit"]["maximum_stream_rows"] == 7
    assert result.diagnostics["allocation_audit"]["n_by_n_matrix_materialized"] is False


def test_conditional_solve_profiles_no_n_by_n_matmul_and_obeys_row_cap() -> None:
    generator = torch.Generator().manual_seed(73)
    rows = 257
    width = 9
    channels = 7
    features = torch.randn(
        (rows, width),
        generator=generator,
        dtype=torch.float64,
    )
    multiplier = torch.randn(
        (rows, channels),
        generator=generator,
        dtype=torch.float64,
    )
    target = torch.randn(
        (rows, channels),
        generator=generator,
        dtype=torch.float64,
    )
    weights = torch.ones((rows,), dtype=torch.float64) / rows
    prior = torch.zeros((channels, width), dtype=torch.float64)
    policy = Lfm2CpcfFhvrMfPolicy(
        pcg_iterations=8,
        factor_sweeps=1,
        row_chunk_size=13,
        channel_chunk_size=3,
        backend="cpu",
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as profile:
        result = solve_lfm2_cpcf_fhvr_mf_conditional_factor(
            features,
            multiplier,
            target,
            weights,
            prior,
            policy=policy,
        )
    forbidden = []
    for event in profile.events():
        if event.name not in {"aten::mm", "aten::matmul"}:
            continue
        shapes = event.input_shapes
        if (
            len(shapes) >= 2
            and len(shapes[0]) == 2
            and len(shapes[1]) == 2
            and shapes[0][0] == rows
            and shapes[1][1] == rows
        ):
            forbidden.append(shapes)
    allocation = result.diagnostics["allocation_audit"]
    assert forbidden == []
    assert allocation["maximum_stream_rows"] == policy.row_chunk_size
    assert allocation["maximum_channel_chunk"] == policy.channel_chunk_size
    assert allocation["n_by_n_matrix_materialized"] is False
    assert allocation["normal_matrix_materialized"] is False
    assert allocation["per_channel_feature_gram_materialized"] is False


def test_full_fhvr_mf_candidate_is_authenticated_and_repeatable() -> None:
    request, context_encoder = _request()
    policy = Lfm2CpcfFhvrMfPolicy(
        pcg_iterations=16,
        factor_sweeps=1,
        row_chunk_size=3,
        channel_chunk_size=1,
        behavior_rank_cap=2,
        absolute_ridge_floor=1e-10,
        backend="cpu",
    )
    first = build_lfm2_cpcf_fhvr_mf_candidate(
        request,
        context_encoder=context_encoder,
        policy=policy,
    )
    second = build_lfm2_cpcf_fhvr_mf_candidate(
        request,
        context_encoder=context_encoder,
        policy=policy,
    )
    assert request.verify_authentication()
    assert first.verify_authentication()
    assert second.verify_authentication()
    assert first.candidate_kind == FHVR_MF_CANDIDATE_KIND
    assert first.state_sha256 == second.state_sha256
    assert first.report["report_sha256"] == second.report["report_sha256"]
    assert tuple(first.in_proj_weight.shape) == (6, 2)
    assert tuple(first.in_proj_bias.shape) == (6,)
    assert tuple(first.fir_kernel.shape) == (2, 2)
    assert tuple(first.fir_bias.shape) == (2,)
    assert tuple(first.writer_weight.shape) == (2, 2)
    assert tuple(first.writer_bias.shape) == (2,)
    provenance = dict(first.method_provenance)
    assert provenance["fit_only"] is True
    assert provenance["selection_cases_consumed"] == 0
    assert provenance["validation_cases_consumed"] == 0
    assert provenance["target_native_dense_weights"] is True
    assert provenance["allocation_audit"]["n_by_n_matrix_materialized"] is False
    assert provenance["execution"]["training_steps"] == 0
    assert provenance["execution"]["backward_calls"] == 0
    assert provenance["execution"]["optimizer_steps"] == 0
