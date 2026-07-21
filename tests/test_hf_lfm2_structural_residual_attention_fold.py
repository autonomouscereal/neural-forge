from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_activation_cluster as activation_cluster  # noqa: E402
import neural_forge.hf_lfm2_structural_residual_attention_fold as fold_module  # noqa: E402
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2AttentionScorePreservationThresholds,
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
    Lfm2CachedNormalizedInputMlpFoldCase,
    Lfm2CachedShortConvFoldCase,
    Lfm2PreparedStructuralMlpFoldBasis,
    Lfm2SparseResidualAssignment,
    Lfm2StructuralResidualAttentionFoldError,
    cached_attention_case_from_lfm2_trace,
    cached_attention_cases_from_lfm2_trace_phases,
    cached_mlp_case_from_lfm2_trace,
    cached_short_conv_case_from_lfm2_trace,
    compose_lfm2_structural_layer_fold,
    compile_lfm2_normalized_input_real_medoid_structural_mlp_fold,
    compile_lfm2_real_medoid_structural_residual_mlp_fold,
    compile_lfm2_structural_attention_producer_fold,
    compile_lfm2_structural_residual_attention_fold,
    compile_lfm2_structural_residual_mlp_fold,
    compile_lfm2_structural_residual_short_conv_fold,
    evaluate_lfm2_compiled_attention_producer_score_preservation,
    evaluate_lfm2_structural_attention_score_preservation,
    install_lfm2_structural_attention_fold,
    install_lfm2_structural_layer_fold,
    install_lfm2_structural_short_conv_fold,
    lfm2_normalized_input_mlp_fit_evidence_sha256,
    prepare_lfm2_structural_residual_mlp_fold_basis,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    open_lfm2_circuit_trace_cache,
)


def test_attention_score_preservation_is_masked_shift_invariant_and_canonical():
    reference = torch.tensor(
        [[[[3.0, 2.0, 1.0, 0.0], [4.0, 2.0, 1.0, -20.0], [3.0, 2.0, 1.0, 0.0]]]],
        dtype=torch.float64,
    )
    candidate = reference.clone()
    candidate[0, 0, 0] = torch.tensor(
        [30.0, -900.0, 700.0, -800.0],
        dtype=torch.float64,
    )
    candidate[0, 0, 1, :3] += 7.0
    candidate[0, 0, 1, 3] = 1000.0
    candidate[0, 0, 2] -= 11.0
    key_mask = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ]
    )
    empty_reference = torch.tensor(
        [[[[1.0, 0.0]]]],
        dtype=torch.float64,
    )
    empty_candidate = torch.tensor(
        [[[[900.0, -900.0]]]],
        dtype=torch.float64,
    )
    empty_mask = torch.zeros((1, 1, 2), dtype=torch.bool)
    originals = tuple(
        value.clone()
        for value in (
            reference,
            candidate,
            key_mask,
            empty_reference,
            empty_candidate,
            empty_mask,
        )
    )
    thresholds = Lfm2AttentionScorePreservationThresholds(
        maximum_softmax_fisher_error=1e-12,
        maximum_js_divergence=1e-12,
        minimum_top1_key_agreement_fraction=1.0,
        minimum_top_k_pair_order_agreement_fraction=1.0,
        minimum_winner_margin_sign_agreement_fraction=1.0,
        minimum_relative_margin_floor_retention_fraction=1.0,
    )

    first = evaluate_lfm2_structural_attention_score_preservation(
        (reference, empty_reference),
        (candidate, empty_candidate),
        key_masks=(key_mask, empty_mask),
        top_k=4,
        relative_margin_floor=1.0,
        thresholds=thresholds,
    )
    second = evaluate_lfm2_structural_attention_score_preservation(
        (reference, empty_reference),
        (candidate, empty_candidate),
        key_masks=(key_mask, empty_mask),
        top_k=4,
        relative_margin_floor=1.0,
        thresholds=thresholds,
    )

    assert first.accepted
    assert first.case_count == 2
    assert first.total_row_count == 4
    assert first.eligible_row_count == 2
    assert first.excluded_singleton_row_count == 1
    assert first.excluded_empty_row_count == 1
    assert first.softmax_fisher_error_worst == pytest.approx(0.0, abs=1e-12)
    assert first.js_divergence_worst == pytest.approx(0.0, abs=1e-12)
    assert first.top1_key_agreement_fraction == 1.0
    assert first.top_k_pair_order_agreement_fraction == 1.0
    assert first.winner_margin_sign_agreement_fraction == 1.0
    assert first.relative_margin_floor_retention_fraction == 1.0
    assert first.as_dict() == second.as_dict()
    assert first.report_sha256 == second.report_sha256
    assert first.as_dict()["score_tensors_only"] is True
    assert first.as_dict()["numeric_weight_matching_used"] is False
    assert first.as_dict()["parent_forward_calls"] == 0
    assert first.as_dict()["training_steps"] == 0
    for value, original in zip(
        (
            reference,
            candidate,
            key_mask,
            empty_reference,
            empty_candidate,
            empty_mask,
        ),
        originals,
        strict=True,
    ):
        assert torch.equal(value, original)


def test_attention_score_preservation_rejects_top_k_pair_order_flip():
    reference = torch.tensor(
        [[[[4.0, 3.0, 2.0, 0.0]]]],
        dtype=torch.float64,
    )
    candidate = torch.tensor(
        [[[[4.0, 2.0, 3.0, 0.0]]]],
        dtype=torch.float64,
    )

    report = evaluate_lfm2_structural_attention_score_preservation(
        reference,
        candidate,
        top_k=3,
        thresholds=Lfm2AttentionScorePreservationThresholds(
            minimum_top1_key_agreement_fraction=1.0,
            minimum_top_k_pair_order_agreement_fraction=1.0,
        ),
    )

    assert report.top1_key_agreement_fraction == 1.0
    assert report.top_k_pair_order_agreement_fraction == pytest.approx(2.0 / 3.0)
    assert not report.accepted
    assert dict(report.checks)["minimum_top_k_pair_order_agreement_fraction"] is False


def test_attention_score_preservation_rejects_relative_margin_collapse():
    reference = torch.tensor(
        [[[[4.0, 2.0, 0.0]]]],
        dtype=torch.float64,
    )
    candidate = torch.tensor(
        [[[[1.0, 0.9, 0.8]]]],
        dtype=torch.float64,
    )

    report = evaluate_lfm2_structural_attention_score_preservation(
        reference,
        candidate,
        top_k=3,
        relative_margin_floor=0.5,
        thresholds=Lfm2AttentionScorePreservationThresholds(
            minimum_top1_key_agreement_fraction=1.0,
            minimum_top_k_pair_order_agreement_fraction=1.0,
            minimum_winner_margin_sign_agreement_fraction=1.0,
            minimum_relative_margin_floor_retention_fraction=1.0,
        ),
    )

    assert report.top1_key_agreement_fraction == 1.0
    assert report.top_k_pair_order_agreement_fraction == 1.0
    assert report.winner_margin_sign_agreement_fraction == 1.0
    assert report.relative_margin_floor_retention_fraction == 0.0
    assert not report.accepted


def test_attention_score_preservation_requires_two_valid_keys():
    reference = torch.zeros((1, 1, 2, 3), dtype=torch.float64)
    candidate = torch.ones_like(reference)
    key_mask = torch.tensor([[[True, False, False], [False, False, False]]])

    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="no row with two valid keys",
    ):
        evaluate_lfm2_structural_attention_score_preservation(
            reference,
            candidate,
            key_masks=key_mask,
        )


class _ToyAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        output_size: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(
            hidden_size, query_heads * head_dim, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(
            hidden_size, key_value_heads * head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            hidden_size, key_value_heads * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(
            query_heads * head_dim, output_size, bias=False, dtype=dtype
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


class _ToyShortConv(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        fir_length: int,
        bias: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, dtype=dtype)
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            fir_length,
            groups=hidden_size,
            bias=bias,
            padding=fir_length - 1,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)


def _source_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(101)
    return {
        "query": torch.randn(8, 8, generator=generator, dtype=torch.float64),
        "key": torch.randn(4, 8, generator=generator, dtype=torch.float64),
        "value": torch.randn(4, 8, generator=generator, dtype=torch.float64),
        "output": torch.randn(8, 8, generator=generator, dtype=torch.float64),
    }


def _duplicate_head_oracle() -> tuple[
    dict[str, torch.Tensor],
    Lfm2CachedAttentionFoldCase,
]:
    # Four source Q heads (two GQA offsets in each of two KV groups) collapse
    # into two target Q heads and one target KV head.  The second source GQA
    # group is an exact functional duplicate of the first, so a full-source
    # merge has an exact smaller representation.
    source = _source_weights()
    target_input = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    target_q_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    # The key rows have zero sequence mean, so centering the score rows does
    # not change this exact rank-two attention oracle.
    target_k_weight = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    target_v_weight = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    target_o_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    query_linear_target = (target_input @ target_q_weight.T).reshape(1, 4, 2, 2)
    query_linear_target = query_linear_target.permute(0, 2, 1, 3)
    key_linear_target = (target_input @ target_k_weight.T).reshape(1, 4, 1, 2)
    key_linear_target = key_linear_target.permute(0, 2, 1, 3)
    value_target = (target_input @ target_v_weight.T).reshape(1, 4, 1, 2)
    value_target = value_target.permute(0, 2, 1, 3)

    parent_query = torch.cat(
        [query_linear_target[:, 0:1], query_linear_target[:, 1:2]] * 2,
        dim=1,
    )
    parent_key = key_linear_target.repeat(1, 2, 1, 1)
    parent_value = value_target.repeat(1, 2, 1, 1)
    repeated_parent_key = parent_key.repeat_interleave(2, dim=1)
    parent_scores = (parent_query @ repeated_parent_key.transpose(-2, -1)) / (2.0**0.5)
    parent_probability = torch.softmax(parent_scores, dim=-1)
    parent_context = parent_probability @ parent_value.repeat_interleave(2, dim=1)

    target_scores = (
        query_linear_target
        @ key_linear_target.repeat_interleave(2, dim=1).transpose(-2, -1)
    ) / (2.0**0.5)
    target_probability = torch.softmax(target_scores, dim=-1)
    target_context = target_probability @ value_target.repeat_interleave(2, dim=1)
    flat_context = target_context.permute(0, 2, 1, 3).reshape(1, 4, 4)
    branch_delta = flat_context @ target_o_weight.T
    residual = torch.tensor([[[0.25, -0.5, 0.75, -1.0]]], dtype=torch.float64).repeat(
        1, 4, 1
    )
    endpoint = residual + branch_delta
    case = Lfm2CachedAttentionFoldCase(
        target_query_input=target_input,
        target_key_value_input=target_input,
        parent_query_linear=parent_query,
        parent_key_linear=parent_key,
        parent_value=parent_value,
        parent_query=parent_query,
        parent_key=parent_key,
        parent_context=parent_context,
        parent_post_attention_residual_target_space=endpoint,
        target_pre_attention_residual=residual,
    )
    return source, case


def _compile(
    source: dict[str, torch.Tensor],
    case: Lfm2CachedAttentionFoldCase,
    target: _ToyAttention,
):
    return compile_lfm2_structural_residual_attention_fold(
        source_query_weight=source["query"],
        source_key_weight=source["key"],
        source_value_weight=source["value"],
        source_output_weight=source["output"],
        target_query_weight_template=target.q_proj.weight,
        target_key_weight_template=target.k_proj.weight,
        target_value_weight_template=target.v_proj.weight,
        target_output_weight_template=target.out_proj.weight,
        cases=(case,),
        head_dim=2,
        residual_assignment_sha256="a" * 64,
        ridge=0.0,
        cluster_ridge=0.0,
    )


def test_exact_duplicate_attention_oracle_is_full_coverage_and_structural() -> None:
    source, case = _duplicate_head_oracle()
    source_before = {name: value.clone() for name, value in source.items()}
    torch.manual_seed(201)
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )

    compiled = _compile(source, case, target)

    assert compiled.audit.all_hold is True
    assert compiled.cluster_plan.report.full_source_coverage is True
    assert sorted(
        source_head for group in compiled.query_groups for source_head in group
    ) == [0, 1, 2, 3]
    assert sorted(
        source_head for group in compiled.key_value_groups for source_head in group
    ) == [0, 1]
    assert compiled.as_dict()["full_source_assignment_fraction"] == 1.0
    assert compiled.as_dict()["source_groups_dropped"] == 0
    assert compiled.as_dict()["query_row_weighting_mode"] == "uniform_default"
    assert compiled.as_dict()["key_value_row_weighting_mode"] == "uniform_default"
    assert compiled.as_dict()["equal_causal_phase_mass_proven"] is False
    assert "equal_causal_phase_mass" not in compiled.as_dict()["score_objective"]
    assert len(compiled.as_dict()["query_row_sqrt_weight_bundle_sha256"]) == 64
    assert compiled.centered_score_relative_rmse < 1e-10
    assert compiled.residual_endpoint_relative_rmse < 1e-10

    expected_shapes = {
        "q_proj.weight": (4, 4),
        "k_proj.weight": (2, 4),
        "v_proj.weight": (2, 4),
        "out_proj.weight": (4, 4),
    }
    state = compiled.state_dict()
    assert {
        name: tuple(value.shape) for name, value in state.items()
    } == expected_shapes
    source_numels = {
        "q_proj.weight": source["query"].numel(),
        "k_proj.weight": source["key"].numel(),
        "v_proj.weight": source["value"].numel(),
        "out_proj.weight": source["output"].numel(),
    }
    assert all(state[name].numel() < source_numels[name] for name in state)
    assert compiled.source_logical_parameters == sum(source_numels.values())
    assert compiled.target_logical_parameters == sum(
        value.numel() for value in state.values()
    )
    assert compiled.target_logical_parameters == 48
    assert compiled.removed_logical_parameters == 144
    assert compiled.parameter_reduction_fraction == pytest.approx(0.75)
    for name, value in source.items():
        assert torch.equal(value, source_before[name])


def test_attention_fold_uses_directional_contributions_and_weighted_qkvo_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, case = _duplicate_head_oracle()
    query_sqrt_weights = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0]],
        dtype=torch.float64,
    )
    key_value_sqrt_weights = torch.tensor(
        [[2.0, 1.5, 1.0, 0.5]],
        dtype=torch.float64,
    )
    contribution = torch.zeros((1, 4, 4, 4), dtype=torch.float64)
    contribution[:, 0, :, 0] = 1.0
    contribution[:, 1, :, 1] = 1.0
    contribution[:, 2, :, 0] = 1.0
    contribution[:, 3, :, 1] = 1.0
    weighted_case = replace(
        case,
        parent_contribution=contribution,
        query_row_sqrt_weights=query_sqrt_weights,
        key_value_row_sqrt_weights=key_value_sqrt_weights,
    )
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    observed: dict[str, object] = {"fits": {}}
    original_cluster = fold_module.build_gqa_activation_cluster_plan
    original_fit = fold_module._fit_linear

    def cluster_spy(*args, **kwargs):
        observed["contribution_signatures"] = args[4].clone()
        return original_cluster(*args, **kwargs)

    def fit_spy(input_rows, target_rows, *, role, **kwargs):
        observed["fits"][role] = (
            input_rows.clone(),
            target_rows.clone(),
        )
        return original_fit(
            input_rows,
            target_rows,
            role=role,
            **kwargs,
        )

    monkeypatch.setattr(
        fold_module,
        "build_gqa_activation_cluster_plan",
        cluster_spy,
    )
    monkeypatch.setattr(fold_module, "_fit_linear", fit_spy)

    compiled = _compile(source, weighted_case, target)

    expected_contribution_signature = (
        (contribution * query_sqrt_weights[:, None, :, None])
        .permute(1, 0, 2, 3)
        .reshape(4, -1)
    )
    actual_contribution_signature = observed["contribution_signatures"]
    assert torch.equal(
        actual_contribution_signature,
        expected_contribution_signature,
    )
    assert int(actual_contribution_signature.shape[1]) == 16
    assert not torch.equal(
        actual_contribution_signature[0],
        actual_contribution_signature[1],
    )
    assert torch.equal(
        torch.linalg.vector_norm(contribution[0], dim=-1),
        torch.ones((4, 4), dtype=torch.float64),
    )

    fits = observed["fits"]
    expected_query_input = case.target_query_input.reshape(
        -1, 4
    ) * query_sqrt_weights.reshape(-1, 1)
    expected_key_value_input = case.target_key_value_input.reshape(
        -1, 4
    ) * key_value_sqrt_weights.reshape(-1, 1)
    expected_output_target = (
        case.parent_post_attention_residual_target_space
        - case.target_pre_attention_residual
    ).reshape(-1, 4) * query_sqrt_weights.reshape(-1, 1)
    assert torch.equal(fits["query"][0], expected_query_input)
    assert torch.equal(fits["key"][0], expected_key_value_input)
    assert torch.equal(fits["value"][0], expected_key_value_input)
    assert torch.allclose(
        fits["output"][1],
        expected_output_target,
        rtol=1e-12,
        atol=1e-12,
    )
    report = compiled.as_dict()
    assert report["query_row_weighting_mode"] == (
        "caller_supplied_positive_sqrt_row_weights"
    )
    assert report["key_value_row_weighting_mode"] == (
        "caller_supplied_positive_sqrt_row_weights"
    )
    assert report["equal_causal_phase_mass_proven"] is False
    assert report["causal_phase_weighting"] == (
        "authenticated_cpcf_caller_report_required"
    )
    assert (
        report["query_row_sqrt_weight_bundle_sha256"]
        != report["key_value_row_sqrt_weight_bundle_sha256"]
    )


def test_structural_attention_allows_preserved_heads_when_hidden_width_shrinks() -> (
    None
):
    source, case = _duplicate_head_oracle()
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=4,
        key_value_heads=2,
        head_dim=2,
        dtype=torch.float64,
    )

    compiled = compile_lfm2_structural_residual_attention_fold(
        source_query_weight=source["query"],
        source_key_weight=source["key"],
        source_value_weight=source["value"],
        source_output_weight=source["output"],
        target_query_weight_template=target.q_proj.weight,
        target_key_weight_template=target.k_proj.weight,
        target_value_weight_template=target.v_proj.weight,
        target_output_weight_template=target.out_proj.weight,
        cases=(case,),
        head_dim=2,
        output_weight_prior=torch.zeros(4, 8, dtype=torch.float64),
        residual_assignment_sha256="a" * 64,
        ridge=0.0,
        max_prior_delta_ratio=None,
        cluster_ridge=0.0,
    )

    assert compiled.topology.source_query_heads == 4
    assert compiled.topology.target_query_heads == 4
    assert compiled.topology.source_key_value_heads == 2
    assert compiled.topology.target_key_value_heads == 2
    assert compiled.topology.source_hidden_size == 8
    assert compiled.topology.target_hidden_size == 4
    assert compiled.cluster_plan.report.full_source_coverage is True
    assert compiled.audit.source_query_heads_dropped == 0
    assert compiled.audit.source_key_value_heads_dropped == 0


def test_compiled_attention_producer_heldout_score_oracle_passes_and_rejects() -> None:
    source, case = _duplicate_head_oracle()
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    producer = compile_lfm2_structural_attention_producer_fold(
        source_query_weight=source["query"],
        source_key_weight=source["key"],
        source_value_weight=source["value"],
        source_output_weight=source["output"],
        target_query_weight_template=target.q_proj.weight,
        target_key_weight_template=target.k_proj.weight,
        target_value_weight_template=target.v_proj.weight,
        target_output_weight_template=target.out_proj.weight,
        cases=(case,),
        head_dim=2,
        residual_assignment_sha256="b" * 64,
        ridge=0.0,
        cluster_ridge=0.0,
    )
    thresholds = Lfm2AttentionScorePreservationThresholds(
        maximum_softmax_fisher_error=1e-10,
        maximum_js_divergence=1e-10,
        minimum_top1_key_agreement_fraction=1.0,
        minimum_top_k_pair_order_agreement_fraction=1.0,
        minimum_winner_margin_sign_agreement_fraction=1.0,
        minimum_relative_margin_floor_retention_fraction=1.0,
    )

    exact = evaluate_lfm2_compiled_attention_producer_score_preservation(
        producer,
        (case,),
        top_k=4,
        relative_margin_floor=0.99,
        thresholds=thresholds,
    )
    collapsed = evaluate_lfm2_compiled_attention_producer_score_preservation(
        replace(
            producer,
            query_weight=torch.zeros_like(producer.query_weight),
        ),
        (case,),
        top_k=4,
        relative_margin_floor=0.99,
        thresholds=thresholds,
    )

    assert exact.accepted
    assert exact.top1_key_agreement_fraction == 1.0
    assert exact.top_k_pair_order_agreement_fraction == 1.0
    assert exact.relative_margin_floor_retention_fraction == 1.0
    assert not collapsed.accepted
    assert collapsed.softmax_fisher_error_worst > 0.0
    assert collapsed.relative_margin_floor_retention_fraction == 0.0
    assert collapsed.as_dict()["model_forward_calls"] == 0
    assert collapsed.as_dict()["parent_forward_calls"] == 0
    assert collapsed.as_dict()["numeric_weight_matching_used"] is False


def test_compile_erases_blank_seed_and_install_preserves_parameter_identity() -> None:
    source, case = _duplicate_head_oracle()
    torch.manual_seed(301)
    first_target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    torch.manual_seed(999)
    second_target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first_target.parameters(), second_target.parameters())
    )

    first = _compile(source, case, first_target)
    second = _compile(source, case, second_target)

    assert first.target_template_sha256 == second.target_template_sha256
    assert first.compiled_state_sha256 == second.compiled_state_sha256
    for name in first.state_dict():
        assert torch.equal(first.state_dict()[name], second.state_dict()[name])

    parameter_ids = tuple(id(value) for value in first_target.parameters())
    parameter_shapes = tuple(tuple(value.shape) for value in first_target.parameters())
    for parameter in first_target.parameters():
        parameter.grad = torch.zeros_like(parameter)
    gradient_ids = tuple(id(value.grad) for value in first_target.parameters())
    report = install_lfm2_structural_attention_fold(first_target, first)

    assert report.accepted is True
    assert report.target_logical_parameters == first.target_logical_parameters
    assert tuple(id(value) for value in first_target.parameters()) == parameter_ids
    assert tuple(tuple(value.shape) for value in first_target.parameters()) == (
        parameter_shapes
    )
    assert tuple(id(value.grad) for value in first_target.parameters()) == gradient_ids
    installed = {
        "q_proj.weight": first_target.q_proj.weight.detach(),
        "k_proj.weight": first_target.k_proj.weight.detach(),
        "v_proj.weight": first_target.v_proj.weight.detach(),
        "out_proj.weight": first_target.out_proj.weight.detach(),
    }
    for name, value in first.state_dict().items():
        assert torch.equal(installed[name], value)


def test_bfloat16_target_is_dense_and_all_four_projections_are_smaller() -> None:
    source, case = _duplicate_head_oracle()
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.bfloat16,
    )

    compiled = _compile(source, case, target)

    assert compiled.audit.every_projection_structurally_smaller is True
    assert compiled.audit.target_shapes_exact is True
    for value in compiled.state_dict().values():
        assert value.dtype == torch.bfloat16
        assert value.layout == torch.strided
        assert value.is_floating_point()


def _runtime_lfm2_head_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    direction = value.float()
    direction = direction * torch.rsqrt(
        direction.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return weight * direction.to(dtype=value.dtype)


def _runtime_bfloat16_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return value * cosine.unsqueeze(1) + fold_module._rotate_half(
        value
    ) * sine.unsqueeze(1)


def test_bfloat16_rope_recovery_is_quantization_aware_and_qk_cross_checked() -> None:
    generator = torch.Generator().manual_seed(2026071701)
    epsilon = 1e-5
    query_linear = torch.randn(1, 4, 5, 4, generator=generator, dtype=torch.bfloat16)
    key_linear = torch.randn(1, 2, 5, 4, generator=generator, dtype=torch.bfloat16)
    query_weight = torch.tensor(
        [0.75, 1.125, 0.875, 1.25],
        dtype=torch.bfloat16,
    )
    key_weight = torch.tensor(
        [1.25, 0.875, 1.125, 0.75],
        dtype=torch.bfloat16,
    )
    angles = torch.tensor(
        [[0.0, 0.15, 0.30, 0.45, 0.60]],
        dtype=torch.float32,
    ).unsqueeze(-1)
    cosine_half = torch.cat((angles.cos(), (angles * 0.5).cos()), dim=-1)
    sine_half = torch.cat((angles.sin(), (angles * 0.5).sin()), dim=-1)
    cosine = torch.cat((cosine_half, cosine_half), dim=-1).to(torch.bfloat16)
    sine = torch.cat((sine_half, sine_half), dim=-1).to(torch.bfloat16)
    normalized_query_runtime = _runtime_lfm2_head_norm(
        query_linear,
        query_weight,
        epsilon,
    )
    normalized_key_runtime = _runtime_lfm2_head_norm(
        key_linear,
        key_weight,
        epsilon,
    )
    rotated_query = _runtime_bfloat16_rope(
        normalized_query_runtime,
        cosine,
        sine,
    )
    rotated_key = _runtime_bfloat16_rope(
        normalized_key_runtime,
        cosine,
        sine,
    )
    normalized_query = fold_module._cached_lfm2_rms_norm(
        query_linear,
        query_weight,
        epsilon,
    )
    normalized_key = fold_module._cached_lfm2_rms_norm(
        key_linear,
        key_weight,
        epsilon,
    )

    query_cosine, query_sine = fold_module._infer_rope_from_cached_rows(
        normalized_query,
        rotated_query.to(torch.float64),
        "synthetic.query",
        evidence_dtype=torch.bfloat16,
    )
    key_cosine, key_sine = fold_module._infer_rope_from_cached_rows(
        normalized_key,
        rotated_key.to(torch.float64),
        "synthetic.key",
        evidence_dtype=torch.bfloat16,
    )
    fold_module._validate_shared_qk_rope(
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        "synthetic",
        evidence_dtype=torch.bfloat16,
    )

    shifted_cosine = torch.roll(cosine, shifts=1, dims=1)
    shifted_sine = torch.roll(sine, shifts=1, dims=1)
    shifted_key = _runtime_bfloat16_rope(
        normalized_key_runtime,
        shifted_cosine,
        shifted_sine,
    )
    wrong_key_cosine, wrong_key_sine = fold_module._infer_rope_from_cached_rows(
        normalized_key,
        shifted_key.to(torch.float64),
        "synthetic.shifted_key",
        evidence_dtype=torch.bfloat16,
    )
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="same legal causal positions",
    ):
        fold_module._validate_shared_qk_rope(
            query_cosine,
            query_sine,
            wrong_key_cosine,
            wrong_key_sine,
            "synthetic.shifted",
            evidence_dtype=torch.bfloat16,
        )


def test_bfloat16_rope_reconstruction_separates_replay_and_geometry_bounds() -> None:
    epsilon = float(torch.finfo(torch.bfloat16).eps)
    unit_roundoff = epsilon / 2.0
    gamma_six = (6.0 * unit_roundoff) / (1.0 - 6.0 * unit_roundoff)
    assert (
        fold_module._rope_low_precision_reconstruction_error_bound(torch.bfloat16)
        == gamma_six
    )
    assert fold_module._rope_low_precision_reconstruction_error_bound(
        torch.float16
    ) == pytest.approx(
        (3.0 * float(torch.finfo(torch.float16).eps))
        / (1.0 - 3.0 * float(torch.finfo(torch.float16).eps))
    )
    assert (
        fold_module._rope_low_precision_reconstruction_error_bound(torch.float32)
        is None
    )
    assert (
        fold_module._rope_low_precision_geometry_error_bound(torch.bfloat16)
        == 2.0 * epsilon
    )
    normalized = torch.tensor(
        [[[[4.09375, 0.2451171875]]]],
        dtype=torch.float64,
    )
    cosine = torch.full(
        (1, 1, 2),
        -0.70703125,
        dtype=torch.float64,
    )
    sine = cosine.clone()
    legal_composed_route = torch.tensor(
        [[[[-2.671875, -3.015625]]]],
        dtype=torch.float64,
    )
    fold_module._validate_rope_reconstruction(
        normalized,
        legal_composed_route,
        cosine,
        sine,
        "synthetic.composed_route",
        evidence_dtype=torch.bfloat16,
    )
    illegal_route = legal_composed_route.clone()
    illegal_route[..., 0] = -2.625
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="one shared legal RoPE transform",
    ):
        fold_module._validate_rope_reconstruction(
            normalized,
            illegal_route,
            cosine,
            sine,
            "synthetic.illegal_route",
            evidence_dtype=torch.bfloat16,
        )
    query_cosine = torch.ones((1, 1, 2), dtype=torch.float64)
    query_sine = torch.zeros((1, 1, 2), dtype=torch.float64)
    key_cosine = torch.full(
        (1, 1, 2),
        float(torch.cos(torch.tensor(0.02)).item()),
        dtype=torch.float64,
    )
    key_sine = torch.full(
        (1, 1, 2),
        float(torch.sin(torch.tensor(0.02)).item()),
        dtype=torch.float64,
    )
    assert 2.0 * epsilon < gamma_six
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="same legal causal positions",
    ):
        fold_module._validate_shared_qk_rope(
            query_cosine,
            query_sine,
            key_cosine,
            key_sine,
            "synthetic.geometry_remains_tight",
            evidence_dtype=torch.bfloat16,
        )


def test_full_prefix_phase_assembler_never_concatenates_overlapping_phases() -> None:
    generator = torch.Generator().manual_seed(2026071702)
    tensors: dict[str, torch.Tensor] = {}
    for phase, length in (("prefill", 3), ("decode.000", 2)):
        prefix = f"{phase}.layer.002."
        query = torch.randn(1, 2, length, 2, generator=generator, dtype=torch.float64)
        key = torch.randn(1, 1, length, 2, generator=generator, dtype=torch.float64)
        tensors.update(
            {
                prefix + "operator.input": torch.randn(
                    1, length, 4, generator=generator, dtype=torch.float64
                ),
                prefix + "attention.q_linear": query.clone(),
                prefix + "attention.k_linear": key.clone(),
                prefix + "attention.q": query,
                prefix + "attention.k": key,
                prefix + "attention.v": key.clone(),
                prefix + "attention.context": query.clone(),
                prefix + "residual.pre": torch.randn(
                    1, length, 4, generator=generator, dtype=torch.float64
                ),
                prefix + "residual.post_operator": torch.randn(
                    1, length, 4, generator=generator, dtype=torch.float64
                ),
            }
        )

    cases = cached_attention_cases_from_lfm2_trace_phases(
        tensors,
        phases=("prefill", "decode.000"),
        layer_index=2,
        decode_trace_mode="full_prefix_canonical",
    )

    assert [case.parent_key.shape[2] for case in cases] == [3, 2]
    assert [case.target_key_value_input.shape[1] for case in cases] == [3, 2]
    assert cases[1].key_mask.tolist() == [[[True, False], [True, True]]]
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="missing.*key/value cache",
    ):
        cached_attention_cases_from_lfm2_trace_phases(
            tensors,
            phases=("prefill", "decode.000"),
            layer_index=2,
            decode_trace_mode="cached_verified",
        )


def test_direct_full_prefix_decode_adapter_uses_a_causal_mask() -> None:
    generator = torch.Generator().manual_seed(2026071703)
    prefix = "decode.000.layer.002."
    query = torch.randn(1, 2, 3, 2, generator=generator, dtype=torch.float64)
    key = torch.randn(1, 1, 3, 2, generator=generator, dtype=torch.float64)
    tensors = {
        prefix + "operator.input": torch.randn(
            1, 3, 4, generator=generator, dtype=torch.float64
        ),
        prefix + "attention.q_linear": query.clone(),
        prefix + "attention.k_linear": key.clone(),
        prefix + "attention.q": query,
        prefix + "attention.k": key,
        prefix + "attention.v": key.clone(),
        prefix + "attention.context": query.clone(),
        prefix + "residual.pre": torch.randn(
            1, 3, 4, generator=generator, dtype=torch.float64
        ),
        prefix + "residual.post_operator": torch.randn(
            1, 3, 4, generator=generator, dtype=torch.float64
        ),
    }

    case = cached_attention_case_from_lfm2_trace(
        tensors,
        phase="decode.000",
        layer_index=2,
        decode_trace_mode="full_prefix_canonical",
    )

    assert case.key_mask.tolist() == [
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    ]


@pytest.mark.skipif(
    not os.environ.get("NEURAL_FORGE_REAL_LFM2_CACHE")
    or not os.environ.get("NEURAL_FORGE_REAL_LFM2_MODEL"),
    reason="real sealed LFM2 cache/model paths were not supplied",
)
def test_real_bfloat16_cache_compiles_q_and_k_without_parent_forward() -> None:
    transformers = pytest.importorskip("transformers")
    cache_path = Path(os.environ["NEURAL_FORGE_REAL_LFM2_CACHE"])
    model_path = Path(os.environ["NEURAL_FORGE_REAL_LFM2_MODEL"])
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    def forbidden_forward(*_args, **_kwargs):
        raise AssertionError("real-cache compile probe executed the parent")

    model.forward = forbidden_forward
    layer = model.model.layers[2].self_attn
    batch = cache.load_batch(0)
    case = cached_attention_case_from_lfm2_trace(
        batch.tensors,
        phase="prefill",
        layer_index=2,
        decode_trace_mode="cached_verified",
    )
    target_query = torch.empty(
        (8 * int(layer.head_dim), 1024),
        dtype=torch.bfloat16,
    )
    target_key = torch.empty(
        (4 * int(layer.head_dim), 1024),
        dtype=torch.bfloat16,
    )
    target_value = torch.empty_like(target_key)
    target_output = torch.empty(
        (1024, 8 * int(layer.head_dim)),
        dtype=torch.bfloat16,
    )
    compiled = compile_lfm2_structural_attention_producer_fold(
        source_query_weight=layer.q_proj.weight,
        source_key_weight=layer.k_proj.weight,
        source_value_weight=layer.v_proj.weight,
        source_output_weight=layer.out_proj.weight,
        target_query_weight_template=target_query,
        target_key_weight_template=target_key,
        target_value_weight_template=target_value,
        target_output_weight_template=target_output,
        cases=(case,),
        head_dim=int(layer.head_dim),
        query_norm_weight=layer.q_layernorm.weight,
        key_norm_weight=layer.k_layernorm.weight,
        query_weight_prior=layer.q_proj.weight[: target_query.shape[0]],
        key_weight_prior=layer.k_proj.weight[: target_key.shape[0]],
        value_weight_prior=layer.v_proj.weight[: target_value.shape[0]],
        output_weight_prior=layer.out_proj.weight[:, : target_output.shape[1]],
        residual_assignment_sha256="f" * 64,
        norm_epsilon=float(layer.q_layernorm.variance_epsilon),
        ridge=1e-6,
        max_prior_delta_ratio=0.0,
    )

    assert compiled.full_source_coverage is True
    assert torch.isfinite(compiled.query_weight.float()).all()
    assert torch.isfinite(compiled.key_weight.float()).all()


def test_trace_adapter_uses_residual_endpoint_not_isolated_attention_delta() -> None:
    prefix = "prefill.layer.003."
    query_linear = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    key_linear = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    tensors = {
        prefix + "operator.input": torch.zeros(1, 2, 4, dtype=torch.float64),
        prefix + "attention.q_linear": query_linear,
        prefix + "attention.k_linear": key_linear,
        prefix + "attention.v": key_linear.clone(),
        prefix + "attention.q": query_linear.clone(),
        prefix + "attention.k": key_linear.clone(),
        prefix + "attention.context": query_linear.clone(),
        prefix + "attention.o": torch.full((1, 2, 4), 7.0, dtype=torch.float64),
        prefix + "residual.pre": torch.full((1, 2, 4), 11.0, dtype=torch.float64),
        prefix + "residual.post_operator": torch.full(
            (1, 2, 4), 18.0, dtype=torch.float64
        ),
    }

    case = cached_attention_case_from_lfm2_trace(
        tensors, phase="prefill", layer_index=3
    )

    assert torch.equal(
        case.parent_post_attention_residual_target_space,
        tensors[prefix + "residual.post_operator"],
    )
    assert torch.equal(
        case.target_pre_attention_residual,
        tensors[prefix + "residual.pre"],
    )
    assert not torch.equal(
        case.parent_post_attention_residual_target_space,
        tensors[prefix + "attention.o"],
    )
    assert torch.equal(
        case.parent_post_attention_residual_target_space
        - case.target_pre_attention_residual,
        tensors[prefix + "attention.o"],
    )


def test_sealed_phase_assembler_reuses_prefill_and_decode_kv_history() -> None:
    generator = torch.Generator().manual_seed(661)
    tensors: dict[str, torch.Tensor] = {}
    key_history = []
    value_history = []
    for phase, length in (("prefill", 2), ("decode.000", 1)):
        prefix = f"{phase}.layer.001."
        query = torch.randn(1, 2, length, 2, generator=generator, dtype=torch.float64)
        key = torch.randn(1, 1, length, 2, generator=generator, dtype=torch.float64)
        value = torch.randn(1, 1, length, 2, generator=generator, dtype=torch.float64)
        key_history.append(key)
        value_history.append(value)
        tensors[prefix + "operator.input"] = torch.randn(
            1, length, 4, generator=generator, dtype=torch.float64
        )
        tensors[prefix + "attention.q_linear"] = query.clone()
        tensors[prefix + "attention.k_linear"] = key.clone()
        tensors[prefix + "attention.q"] = query
        tensors[prefix + "attention.k"] = key
        tensors[prefix + "attention.v"] = value
        tensors[prefix + "attention.context"] = torch.randn(
            1, 2, length, 2, generator=generator, dtype=torch.float64
        )
        tensors[prefix + "residual.pre"] = torch.randn(
            1, length, 4, generator=generator, dtype=torch.float64
        )
        tensors[prefix + "residual.post_operator"] = torch.randn(
            1, length, 4, generator=generator, dtype=torch.float64
        )
        if phase.startswith("decode"):
            tensors[prefix + "attention.key_cache"] = torch.cat(key_history, dim=2)
            tensors[prefix + "attention.value_cache"] = torch.cat(value_history, dim=2)

    cases = cached_attention_cases_from_lfm2_trace_phases(
        tensors,
        phases=("prefill", "decode.000"),
        layer_index=1,
    )

    assert len(cases) == 2
    assert cases[0].parent_key.shape[2] == 2
    assert cases[1].parent_key.shape[2] == 3
    assert cases[1].parent_key_linear.shape[2] == 3
    assert cases[1].target_key_value_input.shape[1] == 3
    assert cases[0].key_mask.tolist() == [[[True, False], [True, True]]]
    assert bool(cases[1].key_mask.all().item()) is True


def test_sealed_phase_assembler_keeps_real_rolling_cache_alignment() -> None:
    generator = torch.Generator().manual_seed(20260717)
    tensors: dict[str, torch.Tensor] = {}
    prefill_length = 8
    head_dim = 4
    prefill_input = torch.randn(
        1, prefill_length, 8, generator=generator, dtype=torch.float64
    )
    prefill_q = torch.randn(
        1, 2, prefill_length, head_dim, generator=generator, dtype=torch.float64
    )
    prefill_k = torch.randn(
        1, 1, prefill_length, head_dim, generator=generator, dtype=torch.float64
    )
    prefill_v = torch.randn(
        1, 1, prefill_length, head_dim, generator=generator, dtype=torch.float64
    )
    decode_input = torch.randn(1, 1, 8, generator=generator, dtype=torch.float64)
    decode_q = torch.randn(1, 2, 1, head_dim, generator=generator, dtype=torch.float64)
    decode_k = torch.randn(1, 1, 1, head_dim, generator=generator, dtype=torch.float64)
    decode_v = torch.randn(1, 1, 1, head_dim, generator=generator, dtype=torch.float64)
    for phase, operator_input, query, key, value in (
        ("prefill", prefill_input, prefill_q, prefill_k, prefill_v),
        ("decode.000", decode_input, decode_q, decode_k, decode_v),
    ):
        prefix = f"{phase}.layer.003."
        tensors.update(
            {
                prefix + "operator.input": operator_input,
                prefix + "attention.q_linear": query.clone(),
                prefix + "attention.k_linear": key.clone(),
                prefix + "attention.q": query,
                prefix + "attention.k": key,
                prefix + "attention.v": value,
                prefix + "attention.context": torch.randn(
                    1,
                    2,
                    query.shape[2],
                    head_dim,
                    generator=generator,
                    dtype=torch.float64,
                ),
                prefix + "residual.pre": torch.randn(
                    1,
                    query.shape[2],
                    8,
                    generator=generator,
                    dtype=torch.float64,
                ),
                prefix + "residual.post_operator": torch.randn(
                    1,
                    query.shape[2],
                    8,
                    generator=generator,
                    dtype=torch.float64,
                ),
            }
        )
    rolling_key = torch.cat((prefill_k, decode_k), dim=2)[:, :, -8:, :]
    rolling_value = torch.cat((prefill_v, decode_v), dim=2)[:, :, -8:, :]
    tensors["decode.000.layer.003.attention.key_cache"] = rolling_key.clone()
    tensors["decode.000.layer.003.attention.value_cache"] = rolling_value.clone()

    cases = cached_attention_cases_from_lfm2_trace_phases(
        tensors,
        phases=("prefill", "decode.000"),
        layer_index=3,
    )

    assert len(cases) == 2
    assert cases[1].parent_key.shape[2] == 8
    assert cases[1].parent_value.shape[2] == 8
    assert cases[1].parent_key_linear.shape[2] == 8
    assert cases[1].target_key_value_input.shape[1] == 8
    assert torch.equal(cases[1].parent_key, rolling_key)
    assert torch.equal(cases[1].parent_value, rolling_value)
    assert torch.equal(
        cases[1].target_key_value_input[:, 0],
        prefill_input[:, 1],
    )


def test_mlp_trace_adapter_flattens_cached_batch_sequence_rows() -> None:
    prefix = "prefill.layer.002."
    tensors = {
        prefix + "mlp.input": torch.zeros(1, 3, 4, dtype=torch.float64),
        prefix + "mlp.gate": torch.ones(1, 3, 6, dtype=torch.float64),
        prefix + "mlp.up": torch.full((1, 3, 6), 2.0, dtype=torch.float64),
        prefix + "mlp.post_swiglu": torch.full((1, 3, 6), 3.0, dtype=torch.float64),
        prefix + "residual.post_operator": torch.full(
            (1, 3, 4), 5.0, dtype=torch.float64
        ),
        prefix + "residual.post_layer": torch.full((1, 3, 4), 8.0, dtype=torch.float64),
    }

    case = cached_mlp_case_from_lfm2_trace(tensors, phase="prefill", layer_index=2)

    assert case.target_input.shape == (3, 4)
    assert case.parent_gate.shape == (3, 6)
    assert case.parent_post_mlp_residual_target_space.shape == (3, 4)
    assert torch.equal(
        case.parent_post_mlp_residual_target_space - case.target_pre_mlp_residual,
        torch.full((3, 4), 3.0, dtype=torch.float64),
    )


def test_rejects_source_output_projection_with_non_residual_output_width() -> None:
    source, case = _duplicate_head_oracle()
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    bad_output = source["output"][:7]

    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="valid GQA topology",
    ):
        compile_lfm2_structural_residual_attention_fold(
            source_query_weight=source["query"],
            source_key_weight=source["key"],
            source_value_weight=source["value"],
            source_output_weight=bad_output,
            target_query_weight_template=target.q_proj.weight,
            target_key_weight_template=target.k_proj.weight,
            target_value_weight_template=target.v_proj.weight,
            target_output_weight_template=target.out_proj.weight,
            cases=(case,),
            head_dim=2,
            residual_assignment_sha256="a" * 64,
        )


def test_underdetermined_cached_fit_requires_and_preserves_source_prior_nullspace() -> (
    None
):
    source, case = _duplicate_head_oracle()
    padded_input = torch.nn.functional.pad(case.target_query_input, (0, 2))
    case = replace(
        case,
        target_query_input=padded_input,
        target_key_value_input=padded_input,
    )
    target = _ToyAttention(
        hidden_size=6,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    common = dict(
        source_query_weight=source["query"],
        source_key_weight=source["key"],
        source_value_weight=source["value"],
        source_output_weight=source["output"],
        target_query_weight_template=target.q_proj.weight,
        target_key_weight_template=target.k_proj.weight,
        target_value_weight_template=target.v_proj.weight,
        target_output_weight_template=target.out_proj.weight,
        cases=(case,),
        head_dim=2,
        residual_assignment_sha256="a" * 64,
        ridge=0.0,
        cluster_ridge=0.0,
    )

    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="source-derived target-shaped prior is required",
    ):
        compile_lfm2_structural_residual_attention_fold(**common)

    generator = torch.Generator().manual_seed(777)
    q_prior = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    k_prior = torch.randn(2, 6, generator=generator, dtype=torch.float64)
    v_prior = torch.randn(2, 6, generator=generator, dtype=torch.float64)
    compiled = compile_lfm2_structural_residual_attention_fold(
        **common,
        query_weight_prior=q_prior,
        key_weight_prior=k_prior,
        value_weight_prior=v_prior,
        max_prior_delta_ratio=None,
    )

    # The cache has no signal in the last two input coordinates.  Residual
    # fitting around the source-derived prior leaves that nullspace planted,
    # instead of silently resetting it to zero.
    assert torch.equal(compiled.query_weight[:, 4:], q_prior[:, 4:])
    assert torch.equal(compiled.key_weight[:, 4:], k_prior[:, 4:])
    assert torch.equal(compiled.value_weight[:, 4:], v_prior[:, 4:])
    reports = {value.role: value for value in compiled.prior_correction_reports}
    assert reports["query"].underdetermined is True
    assert reports["query"].prior_provided is True
    assert reports["key"].underdetermined is True
    assert reports["value"].underdetermined is True
    assert compiled.source_derived_prior_sha256 is not None


def test_exact_duplicate_swiglu_oracle_uses_every_neuron_and_installs_atomically() -> (
    None
):
    generator = torch.Generator().manual_seed(880)
    source_gate_weight = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    source_up_weight = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    source_down_weight = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    target_input = torch.eye(4, dtype=torch.float64)
    desired_gate = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    desired_up = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    gate_two = target_input @ desired_gate.T
    up_two = target_input @ desired_up.T
    parent_gate = torch.stack(
        (gate_two[:, 0], gate_two[:, 0], gate_two[:, 1], gate_two[:, 1]),
        dim=1,
    )
    parent_up = torch.stack(
        (up_two[:, 0], up_two[:, 0], up_two[:, 1], up_two[:, 1]),
        dim=1,
    )
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    desired_down = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
        dtype=torch.float64,
    )
    target_post = torch.nn.functional.silu(gate_two) * up_two
    residual = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float64).repeat(
        4, 1
    )
    endpoint = residual + target_post @ desired_down.T
    case = Lfm2CachedMlpFoldCase(
        target_input=target_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=residual,
    )
    target = _ToyMlp(hidden_size=4, intermediate_size=2, dtype=torch.float64)
    parameter_ids = tuple(id(value) for value in target.parameters())
    parameter_shapes = tuple(tuple(value.shape) for value in target.parameters())

    compiled = compile_lfm2_structural_residual_mlp_fold(
        source_gate_weight=source_gate_weight,
        source_up_weight=source_up_weight,
        source_down_weight=source_down_weight,
        target_gate_weight_template=target.w1.weight,
        target_up_weight_template=target.w3.weight,
        target_down_weight_template=target.w2.weight,
        cases=(case,),
        residual_assignment_sha256="a" * 64,
        ridge=0.0,
        cluster_ridge=0.0,
    )

    assert compiled.full_source_coverage is True
    assert compiled.as_dict()["source_neurons_dropped"] == 0
    assert sorted(neuron for group in compiled.neuron_groups for neuron in group) == [
        0,
        1,
        2,
        3,
    ]
    assert compiled.residual_endpoint_relative_rmse < 1e-10
    assert compiled.source_logical_parameters == 96
    assert compiled.target_logical_parameters == 24
    assert compiled.removed_logical_parameters == 72
    assert compiled.parameter_reduction_fraction == pytest.approx(0.75)
    assert {
        name: tuple(value.shape) for name, value in compiled.state_dict().items()
    } == {
        "w1.weight": (2, 4),
        "w3.weight": (2, 4),
        "w2.weight": (4, 2),
    }

    attention_source, attention_case = _duplicate_head_oracle()
    attention_target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    attention_compiled = _compile(attention_source, attention_case, attention_target)
    target_layer = nn.Module()
    target_layer.self_attn = attention_target
    target_layer.feed_forward = target
    all_parameter_ids = tuple(id(value) for value in target_layer.parameters())
    all_parameter_shapes = tuple(
        tuple(value.shape) for value in target_layer.parameters()
    )
    layer_compiled = compose_lfm2_structural_layer_fold(attention_compiled, compiled)
    report = install_lfm2_structural_layer_fold(target_layer, layer_compiled)
    assert report.accepted is True
    assert report.target_logical_parameters == 72
    assert tuple(id(value) for value in target_layer.parameters()) == all_parameter_ids
    assert tuple(tuple(value.shape) for value in target_layer.parameters()) == (
        all_parameter_shapes
    )
    assert tuple(id(value) for value in target.parameters()) == parameter_ids
    assert tuple(tuple(value.shape) for value in target.parameters()) == (
        parameter_shapes
    )
    assert torch.equal(target.w1.weight, compiled.gate_weight)
    assert torch.equal(target.w3.weight, compiled.up_weight)
    assert torch.equal(target.w2.weight, compiled.down_weight)


def _prepared_mlp_fixture():
    generator = torch.Generator().manual_seed(9917)
    source_hidden = 8
    target_hidden = 4
    source_intermediate = 4
    target_intermediate = 2
    rows = 8
    source_gate = torch.randn(
        source_intermediate,
        source_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    source_up = torch.randn(
        source_intermediate,
        source_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    source_down = torch.randn(
        source_hidden,
        source_intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    target_input = torch.randn(
        rows,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    parent_gate = torch.randn(
        rows,
        source_intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    parent_up = torch.randn(
        rows,
        source_intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    target_residual = torch.randn(
        rows,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    parent_endpoint = torch.randn(
        rows,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    case = Lfm2CachedMlpFoldCase(
        target_input=target_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=parent_endpoint,
        target_pre_mlp_residual=target_residual,
    )
    target = _ToyMlp(
        hidden_size=target_hidden,
        intermediate_size=target_intermediate,
        dtype=torch.bfloat16,
    )
    common = {
        "source_gate_weight": source_gate,
        "source_up_weight": source_up,
        "source_down_weight": source_down,
        "target_gate_weight_template": target.w1.weight,
        "target_up_weight_template": target.w3.weight,
        "target_down_weight_template": target.w2.weight,
        "residual_assignment_sha256": "9" * 64,
        "cluster_ridge": 1e-8,
        "cluster_max_iterations": 16,
        "cluster_distance_chunk_size": 512,
    }
    return common, case


def test_preallocated_mlp_features_preserve_reports_and_compiled_state_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, case = _prepared_mlp_fixture()
    preallocated = prepare_lfm2_structural_residual_mlp_fold_basis(
        **common,
        cases=(case,),
    )
    preallocated_compiled = compile_lfm2_structural_residual_mlp_fold(
        **common,
        cases=(case,),
        ridge=1e-4,
        max_prior_delta_ratio=None,
        prepared_basis=preallocated,
    )

    def legacy_combine_modalities(torch_module, matrices):
        parts = []
        for matrix in matrices:
            width = int(matrix.shape[1])
            rms = torch_module.sqrt(matrix.square().mean())
            scale = float(rms.item())
            if scale == 0.0:
                scale = 1.0
            parts.append(matrix / (scale * math.sqrt(width)))
        return torch_module.cat(parts, dim=1).contiguous()

    monkeypatch.setattr(
        activation_cluster,
        "_combine_modalities",
        legacy_combine_modalities,
    )
    legacy = prepare_lfm2_structural_residual_mlp_fold_basis(
        **common,
        cases=(case,),
    )
    legacy_compiled = compile_lfm2_structural_residual_mlp_fold(
        **common,
        cases=(case,),
        ridge=1e-4,
        max_prior_delta_ratio=None,
        prepared_basis=legacy,
    )

    assert preallocated.as_dict() == legacy.as_dict()
    assert preallocated.preparation_sha256 == legacy.preparation_sha256
    assert preallocated.prepared_tensor_sha256 == legacy.prepared_tensor_sha256
    assert preallocated_compiled.as_dict() == legacy_compiled.as_dict()
    assert (
        preallocated_compiled.compiled_state_sha256
        == legacy_compiled.compiled_state_sha256
    )
    for name, value in preallocated_compiled.state_dict().items():
        assert torch.equal(value, legacy_compiled.state_dict()[name])


def test_prepared_mlp_basis_matches_uncached_bf16_state_and_remains_candidate_specific() -> (
    None
):
    common, case = _prepared_mlp_fixture()
    prepared = prepare_lfm2_structural_residual_mlp_fold_basis(
        **common,
        cases=(case,),
    )
    assert isinstance(prepared, Lfm2PreparedStructuralMlpFoldBasis)
    assert prepared.verify_authentication() is True
    assert prepared.report["preparation_calls"] == 1
    assert prepared.report["candidate_compiles_performed_during_preparation"] == 0
    assert prepared.report["combined_cluster_feature_matrix_retained"] is False

    uncached = compile_lfm2_structural_residual_mlp_fold(
        **common,
        cases=(case,),
        ridge=1e-4,
        max_prior_delta_ratio=None,
    )
    cached = compile_lfm2_structural_residual_mlp_fold(
        **common,
        cases=(case,),
        ridge=1e-4,
        max_prior_delta_ratio=None,
        prepared_basis=prepared,
    )
    assert uncached.compiled_state_sha256 == cached.compiled_state_sha256
    assert uncached.neuron_groups == cached.neuron_groups
    assert uncached.neuron_barycentric_weights == cached.neuron_barycentric_weights
    assert (
        uncached.residual_endpoint_relative_rmse
        == cached.residual_endpoint_relative_rmse
    )
    assert uncached.prepared_basis_sha256 == prepared.preparation_sha256
    assert cached.prepared_basis_sha256 == prepared.preparation_sha256
    assert uncached.prepared_basis_reused is False
    assert cached.prepared_basis_reused is True
    for name in uncached.state_dict():
        assert uncached.state_dict()[name].dtype == torch.bfloat16
        assert torch.equal(
            uncached.state_dict()[name],
            cached.state_dict()[name],
        )

    candidate_case = replace(
        case,
        target_input=case.target_input.roll(1, dims=0),
        target_pre_mlp_residual=case.target_pre_mlp_residual + 0.5,
    )
    candidate = compile_lfm2_structural_residual_mlp_fold(
        **common,
        cases=(candidate_case,),
        ridge=1e-4,
        max_prior_delta_ratio=None,
        prepared_basis=prepared,
    )
    assert candidate.prepared_basis_sha256 == prepared.preparation_sha256
    assert candidate.compiled_state_sha256 != cached.compiled_state_sha256


def test_prepared_mlp_basis_fails_closed_on_parent_source_or_template_mismatch() -> (
    None
):
    common, case = _prepared_mlp_fixture()
    prepared = prepare_lfm2_structural_residual_mlp_fold_basis(
        **common,
        cases=(case,),
    )
    tampered_parent = replace(
        case,
        parent_gate=case.parent_gate.clone(),
    )
    tampered_parent.parent_gate[0, 0] += 1.0
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="parent evidence changed",
    ):
        compile_lfm2_structural_residual_mlp_fold(
            **common,
            cases=(tampered_parent,),
            ridge=1e-4,
            prepared_basis=prepared,
        )
    tampered_source = dict(common)
    tampered_source["source_gate_weight"] = common["source_gate_weight"].clone()
    tampered_source["source_gate_weight"][0, 0] += 1.0
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="source state changed",
    ):
        compile_lfm2_structural_residual_mlp_fold(
            **tampered_source,
            cases=(case,),
            ridge=1e-4,
            prepared_basis=prepared,
        )
    tampered_template = dict(common)
    tampered_template["target_gate_weight_template"] = common[
        "target_gate_weight_template"
    ].to(torch.float32)
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="target template changed",
    ):
        compile_lfm2_structural_residual_mlp_fold(
            **tampered_template,
            cases=(case,),
            ridge=1e-4,
            prepared_basis=prepared,
        )


def test_no_prior_linear_fit_bypasses_zero_prior_without_changing_bytes() -> None:
    generator = torch.Generator().manual_seed(812)
    inputs = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    targets = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    without_prior, no_prior_report = fold_module._fit_linear(
        inputs,
        targets,
        role="without_prior",
        ridge=1e-4,
        prior=None,
        max_prior_delta_ratio=None,
    )
    with_zero_prior, zero_prior_report = fold_module._fit_linear(
        inputs,
        targets,
        role="with_zero_prior",
        ridge=1e-4,
        prior=torch.zeros(3, 4, dtype=torch.float64),
        max_prior_delta_ratio=None,
    )
    assert torch.equal(without_prior, with_zero_prior)
    assert no_prior_report.prior_provided is False
    assert zero_prior_report.prior_provided is True


def test_structural_swiglu_allows_preserved_neuron_count_when_hidden_width_shrinks() -> (
    None
):
    generator = torch.Generator().manual_seed(1880)
    source_hidden = 8
    target_hidden = 4
    intermediate = 4
    rows = 4
    source_gate_weight = torch.randn(
        intermediate,
        source_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    source_up_weight = torch.randn(
        intermediate,
        source_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    source_down_weight = torch.randn(
        source_hidden,
        intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    target_input = torch.eye(rows, target_hidden, dtype=torch.float64)
    parent_gate = torch.randn(
        rows,
        intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    parent_up = torch.randn(
        rows,
        intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    target_residual = torch.randn(
        rows,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    desired_down = torch.randn(
        target_hidden,
        intermediate,
        generator=generator,
        dtype=torch.float64,
    )
    endpoint = target_residual + parent_post @ desired_down.T
    target = _ToyMlp(
        hidden_size=target_hidden,
        intermediate_size=intermediate,
        dtype=torch.float64,
    )

    compiled = compile_lfm2_structural_residual_mlp_fold(
        source_gate_weight=source_gate_weight,
        source_up_weight=source_up_weight,
        source_down_weight=source_down_weight,
        target_gate_weight_template=target.w1.weight,
        target_up_weight_template=target.w3.weight,
        target_down_weight_template=target.w2.weight,
        cases=(
            Lfm2CachedMlpFoldCase(
                target_input=target_input,
                parent_gate=parent_gate,
                parent_up=parent_up,
                parent_post_swiglu=parent_post,
                parent_post_mlp_residual_target_space=endpoint,
                target_pre_mlp_residual=target_residual,
            ),
        ),
        residual_assignment_sha256="b" * 64,
        ridge=0.0,
        cluster_ridge=0.0,
    )

    assert compiled.source_intermediate_size == intermediate
    assert compiled.target_intermediate_size == intermediate
    assert compiled.source_hidden_size == source_hidden
    assert compiled.target_hidden_size == target_hidden
    assert compiled.full_source_coverage is True
    assert compiled.removed_logical_parameters == 48
    assert compiled.parameter_reduction_fraction == pytest.approx(0.5)
    assert compiled.residual_endpoint_relative_rmse < 1e-10


def _real_medoid_chart_mlp_fixture(*, dtype=torch.float64):
    source_gate = torch.tensor(
        [[1.125, 7.25], [2.375, -3.5], [-1.75, 4.125], [0.625, -6.75]],
        dtype=torch.float64,
    )
    source_up = torch.tensor(
        [[0.375, -2.25], [-0.875, 5.5], [2.25, 3.125], [-1.5, -4.25]],
        dtype=torch.float64,
    )
    source_down = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -3.0]],
        dtype=torch.float64,
    )
    target_input = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    parent_gate = target_input @ source_gate.T
    parent_up = target_input @ source_up.T
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    contribution = parent_post.abs() + torch.tensor(
        [[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64
    )
    target_residual = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    endpoint = target_residual + torch.tensor([[0.75, -0.35]], dtype=torch.float64)
    case = Lfm2CachedMlpFoldCase(
        target_input=target_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=target_residual,
        parent_contribution=contribution,
    )
    target = _ToyMlp(hidden_size=2, intermediate_size=2, dtype=dtype)
    chart = torch.eye(2, dtype=torch.float64)
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
        "ridge": 0.0,
        "cluster_ridge": 1e-8,
    }
    return common, case


def test_real_medoid_chart_fold_plants_coupled_producers_and_fit_nullspace() -> None:
    common, case = _real_medoid_chart_mlp_fixture()
    compiled = compile_lfm2_real_medoid_structural_residual_mlp_fold(**common)
    medoids = torch.tensor(
        compiled.cluster_plan.medoid_source_neurons, dtype=torch.long
    )
    expected_gate = common["source_gate_weight"].index_select(0, medoids)
    expected_up = common["source_up_weight"].index_select(0, medoids)

    # The single FIT row observes only input coordinate zero.  The chart prior
    # therefore supplies coordinate one, and gate/up remain the same coupled
    # real source pair instead of two independently averaged producers.
    assert torch.equal(compiled.gate_weight, expected_gate)
    assert torch.equal(compiled.up_weight, expected_up)
    assert torch.equal(compiled.gate_weight[:, 1], expected_gate[:, 1])
    assert torch.equal(compiled.up_weight[:, 1], expected_up[:, 1])
    reports = {value.role: value for value in compiled.prior_correction_reports}
    assert reports["mlp_real_medoid_gate"].underdetermined is True
    assert reports["mlp_real_medoid_gate"].prior_provided is True
    assert reports["mlp_real_medoid_up"].underdetermined is True

    for target_index, group in enumerate(compiled.neuron_groups):
        if len(group) <= 1:
            continue
        group_indices = torch.tensor(group, dtype=torch.long)
        independent_gate_mean = (
            common["source_gate_weight"].index_select(0, group_indices).mean(dim=0)
        )
        independent_up_mean = (
            common["source_up_weight"].index_select(0, group_indices).mean(dim=0)
        )
        assert not (
            torch.equal(compiled.gate_weight[target_index], independent_gate_mean)
            and torch.equal(compiled.up_weight[target_index], independent_up_mean)
        )
        break
    else:  # pragma: no cover - impossible for four sources and two targets
        raise AssertionError("Expected at least one multi-source primary group.")

    medoid_post = case.parent_post_swiglu.index_select(1, medoids)
    actual_post = torch.nn.functional.silu(
        case.target_input @ compiled.gate_weight.T
    ) * (case.target_input @ compiled.up_weight.T)
    assert torch.equal(actual_post, medoid_post)
    receipt = compiled.as_dict()
    assert receipt["producer_averaging"] is False
    assert receipt["producer_plan"] == "sparse_multi_anchor_real_medoid"
    assert receipt["source_neurons_dropped"] == 0
    assert receipt["full_source_coverage"] is True
    assert receipt["plan_evidence_scope"] == "supplied_fit_cases_only"
    assert receipt["heldout_rows_used_for_plan"] == 0
    assert receipt["training_steps"] == 0


def test_real_medoid_chart_fold_authenticates_chart_and_every_source_consumer() -> None:
    common, _ = _real_medoid_chart_mlp_fixture()
    compiled = compile_lfm2_real_medoid_structural_residual_mlp_fold(**common)
    expected_prior = common["residual_chart"].T @ (
        compiled.cluster_plan.aggregate_down_consumer(
            common["source_down_weight"], axis=1
        )
    )
    assert compiled.consumer_prior_sha256 == fold_module._tensor_sha256(expected_prior)
    assert sorted(
        source for group in compiled.neuron_groups for source in group
    ) == list(range(common["source_down_weight"].shape[1]))
    assert len(compiled.cluster_plan.source_anchor_offsets) == (
        common["source_down_weight"].shape[1] + 1
    )

    # Chart orientation is explicitly source-hidden by target-hidden.  A
    # non-square valid chart compiles, while its authenticated transpose does
    # not silently receive the opposite interpretation.
    cross_hidden = dict(common)
    cross_hidden["source_gate_weight"] = torch.column_stack(
        (common["source_gate_weight"], torch.tensor([0.2, -0.3, 0.4, -0.5]))
    )
    cross_hidden["source_up_weight"] = torch.column_stack(
        (common["source_up_weight"], torch.tensor([-0.6, 0.7, -0.8, 0.9]))
    )
    cross_hidden["source_down_weight"] = torch.row_stack(
        (common["source_down_weight"], torch.tensor([0.1, 0.2, -0.4, 0.8]))
    )
    cross_hidden_chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.25, -0.5]], dtype=torch.float64
    )
    cross_hidden["residual_chart"] = cross_hidden_chart
    cross_hidden["residual_chart_sha256"] = (
        fold_module.lfm2_authenticated_tensor_sha256(cross_hidden_chart)
    )
    cross_hidden_compiled = compile_lfm2_real_medoid_structural_residual_mlp_fold(
        **cross_hidden
    )
    assert cross_hidden_compiled.source_hidden_size == 3
    assert cross_hidden_compiled.target_hidden_size == 2
    wrong_orientation = dict(cross_hidden)
    wrong_orientation["residual_chart"] = cross_hidden_chart.T.contiguous()
    wrong_orientation["residual_chart_sha256"] = (
        fold_module.lfm2_authenticated_tensor_sha256(
            wrong_orientation["residual_chart"]
        )
    )
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match=r"\[source_hidden, target_hidden\]",
    ):
        compile_lfm2_real_medoid_structural_residual_mlp_fold(**wrong_orientation)

    nonmedoid = next(
        source
        for source in range(common["source_down_weight"].shape[1])
        if source not in compiled.cluster_plan.medoid_source_neurons
    )
    start, stop = compiled.cluster_plan.source_anchor_offsets[nonmedoid : nonmedoid + 2]
    assert any(
        value != 0.0 for value in compiled.cluster_plan.anchor_coefficients[start:stop]
    )
    changed = dict(common)
    changed_down = common["source_down_weight"].clone()
    changed_down[:, nonmedoid] += torch.tensor([0.75, -0.5])
    changed["source_down_weight"] = changed_down
    changed_compiled = compile_lfm2_real_medoid_structural_residual_mlp_fold(**changed)
    assert changed_compiled.consumer_prior_sha256 != compiled.consumer_prior_sha256

    corrupt_chart = dict(common)
    corrupt_chart["residual_chart"] = common["residual_chart"].clone()
    corrupt_chart["residual_chart"][0, 0] += 0.25
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="authentication failed",
    ):
        compile_lfm2_real_medoid_structural_residual_mlp_fold(**corrupt_chart)


def test_real_medoid_chart_w2_uses_materialized_bf16_producer_features() -> None:
    common, case = _real_medoid_chart_mlp_fixture(dtype=torch.bfloat16)
    first = compile_lfm2_real_medoid_structural_residual_mlp_fold(**common)
    second = compile_lfm2_real_medoid_structural_residual_mlp_fold(**common)
    assert first.compiled_state_sha256 == second.compiled_state_sha256
    assert first.as_dict()["endpoint_feature_source"] == (
        "materialized_target_dtype_w1_w3_cast_back_to_fp64"
    )
    assert first.gate_weight.dtype == torch.bfloat16
    assert first.up_weight.dtype == torch.bfloat16
    materialized_post = torch.nn.functional.silu(
        case.target_input @ first.gate_weight.to(torch.float64).T
    ) * (case.target_input @ first.up_weight.to(torch.float64).T)
    endpoint_target = (
        case.parent_post_mlp_residual_target_space - case.target_pre_mlp_residual
    )
    down_prior = common["residual_chart"].T @ (
        first.cluster_plan.aggregate_down_consumer(common["source_down_weight"], axis=1)
    )
    expected_down, _ = fold_module._fit_linear(
        materialized_post,
        endpoint_target,
        role="oracle_materialized_down",
        ridge=0.0,
        prior=down_prior,
        max_prior_delta_ratio=None,
    )
    assert torch.equal(first.down_weight, expected_down.to(torch.bfloat16))


def test_real_medoid_chart_applies_authenticated_fit_case_weights() -> None:
    common, first_case = _real_medoid_chart_mlp_fixture()
    second_gate = first_case.parent_gate + torch.tensor(
        [[1.0, -0.5, 0.75, -1.25]], dtype=torch.float64
    )
    second_up = first_case.parent_up + torch.tensor(
        [[-0.25, 0.75, -0.5, 1.0]], dtype=torch.float64
    )
    second_case = replace(
        first_case,
        parent_gate=second_gate,
        parent_up=second_up,
        parent_post_swiglu=torch.nn.functional.silu(second_gate) * second_up,
        parent_post_mlp_residual_target_space=(
            first_case.parent_post_mlp_residual_target_space
            + torch.tensor([[0.4, -0.2]], dtype=torch.float64)
        ),
    )
    weighted_common = dict(common)
    weighted_common["cases"] = (first_case, second_case)
    uniform = compile_lfm2_real_medoid_structural_residual_mlp_fold(
        **weighted_common,
        case_sample_weights=(1.0, 1.0),
    )
    imbalanced = compile_lfm2_real_medoid_structural_residual_mlp_fold(
        **weighted_common,
        case_sample_weights=(100.0, 1.0),
    )
    assert uniform.compiled_state_sha256 != imbalanced.compiled_state_sha256
    receipt = imbalanced.as_dict()
    assert receipt["fit_weighting"] == ("sqrt_case_weight_repeated_per_case_row")
    assert receipt["fit_case_weight_sum"] == 101.0
    assert receipt["fit_row_weight_sum"] == 101.0
    assert receipt["fit_row_weight_minimum"] == 1.0
    assert receipt["fit_row_weight_maximum"] == 100.0
    assert receipt["fit_row_weights_sha256"] == (
        fold_module.lfm2_authenticated_tensor_sha256(
            torch.tensor([100.0, 1.0], dtype=torch.float64)
        )
    )
    assert receipt["heldout_rows_used_for_plan"] == 0
    assert receipt["heldout_rows_used_by_solver"] == 0
    assert receipt["multi_anchor_plan"]["calibration_provenance"] == (
        "cached_fit_rows_only"
    )
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="positive finite",
    ):
        compile_lfm2_real_medoid_structural_residual_mlp_fold(
            **weighted_common,
            case_sample_weights=(1.0, 0.0),
        )


def _normalized_input_real_medoid_fixture(*, dtype=torch.float64):
    parent_gamma = torch.tensor([10.0, 0.1, 5.0], dtype=torch.float64)
    target_gamma = torch.tensor([0.2, 4.0], dtype=torch.float64)
    normalized_polar = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    residual_chart = (
        parent_gamma.unsqueeze(1)
        * normalized_polar
        * target_gamma.reciprocal().unsqueeze(0)
    ).contiguous()
    source_gate = torch.tensor(
        [
            [1.0, 0.25, -0.5],
            [-0.75, 1.5, 0.4],
            [0.3, -2.0, 1.25],
            [1.75, 0.6, -0.8],
        ],
        dtype=torch.float64,
    )
    source_up = torch.tensor(
        [
            [0.5, -1.0, 0.75],
            [1.25, 0.4, -0.6],
            [-0.2, 1.1, 1.5],
            [0.9, -0.7, 0.3],
        ],
        dtype=torch.float64,
    )
    source_down = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [0.4, 1.25, -0.8, 0.2],
            [-0.3, 0.6, 1.1, -1.0],
        ],
        dtype=torch.float64,
    )
    target_input = torch.tensor([[1.0, 0.0], [-2.0, 0.0]], dtype=torch.float64)
    parent_input = target_input @ normalized_polar.T
    parent_gate = parent_input @ source_gate.T
    parent_up = parent_input @ source_up.T
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    target_residual = torch.tensor([[0.1, -0.2], [0.3, 0.15]], dtype=torch.float64)
    parent_delta = parent_post @ source_down.T
    endpoint = target_residual + parent_delta @ residual_chart
    case = Lfm2CachedNormalizedInputMlpFoldCase(
        target_input=target_input,
        parent_normalized_input=parent_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=target_residual,
        parent_contribution=parent_post.abs() + 0.125,
    )
    target = _ToyMlp(hidden_size=2, intermediate_size=2, dtype=dtype)
    common = {
        "source_gate_weight": source_gate,
        "source_up_weight": source_up,
        "source_down_weight": source_down,
        "target_gate_weight_template": target.w1.weight,
        "target_up_weight_template": target.w3.weight,
        "target_down_weight_template": target.w2.weight,
        "residual_chart": residual_chart,
        "residual_chart_sha256": fold_module.lfm2_authenticated_tensor_sha256(
            residual_chart
        ),
        "parent_ffn_norm_weight": parent_gamma,
        "parent_ffn_norm_weight_sha256": (
            fold_module.lfm2_authenticated_tensor_sha256(parent_gamma)
        ),
        "target_ffn_norm_weight": target_gamma,
        "target_ffn_norm_weight_sha256": (
            fold_module.lfm2_authenticated_tensor_sha256(target_gamma)
        ),
        "cases": (case,),
        "anchors_per_source": 2,
        "max_candidate_targets": 2,
        "normalized_decoder_ridge": 1e-8,
        "ridge": 0.0,
        "cluster_ridge": 1e-8,
    }
    return common, case, normalized_polar


def test_normalized_input_real_medoid_bridge_generalizes_raw_scale_mismatch() -> None:
    common, case, normalized_polar = _normalized_input_real_medoid_fixture()
    compiled = compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(**common)

    # FIT sees only child coordinate zero.  The gamma-aware polar completion
    # preserves unseen coordinate one, whereas a raw chart pseudoinverse is
    # wrong by forty-fold under this deliberately severe normalization scale.
    heldout_target_input = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    expected_parent_input = heldout_target_input @ normalized_polar.T
    normalized_prediction = heldout_target_input @ compiled.normalized_input_decoder
    raw_prediction = heldout_target_input @ torch.linalg.pinv(common["residual_chart"])
    assert torch.allclose(normalized_prediction, expected_parent_input, atol=1e-12)
    assert torch.linalg.vector_norm(raw_prediction - expected_parent_input) > 30.0

    medoids = torch.tensor(
        compiled.cluster_plan.medoid_source_neurons, dtype=torch.long
    )
    selected_gate = common["source_gate_weight"].index_select(0, medoids)
    selected_up = common["source_up_weight"].index_select(0, medoids)
    assert torch.allclose(
        compiled.gate_weight,
        selected_gate @ compiled.normalized_input_decoder.T,
        atol=1e-12,
    )
    assert torch.allclose(
        compiled.up_weight,
        selected_up @ compiled.normalized_input_decoder.T,
        atol=1e-12,
    )
    actual_fit_post = torch.nn.functional.silu(
        case.target_input @ compiled.gate_weight.T
    ) * (case.target_input @ compiled.up_weight.T)
    expected_fit_post = case.parent_post_swiglu.index_select(1, medoids)
    assert torch.allclose(actual_fit_post, expected_fit_post, atol=1e-12)

    receipt = compiled.as_dict()
    producer = receipt["producer_bridge"]
    consumer = receipt["consumer_bridge"]
    assert producer["role"] == "normalized_mlp_input_target_to_parent"
    assert consumer["role"] == "raw_residual_source_to_target"
    assert consumer["orthonormal_columns"] is False
    assert consumer["decoder_method"] == ("single_authenticated_thin_svd_pseudoinverse")
    assert consumer["residual_chart_svd_calls"] == 1
    assert consumer["residual_chart_pseudoinverse_calls"] == 0
    assert receipt["producer_consumer_bridge_fields_separate"] is True
    assert producer["fit_rank"] == 1
    assert producer["decoder_rank"] == 2
    assert producer["positive_global_gain"] == pytest.approx(1.0)
    assert producer["ridge_policy_value"] == pytest.approx(
        common["normalized_decoder_ridge"]
    )
    assert producer["supported_relative_cap_rho"] == 1.0
    assert producer["correction_rowspace_leakage"] < 1e-12
    assert producer["null_completion_error"] < 1e-12
    assert producer["null_completion_operator_norm"] <= (
        producer["null_completion_operator_norm_bound"] + 1e-12
    )
    assert receipt["producer_per_neuron_interpolation"] is False
    assert receipt["producer_shared_decoder"] is True
    assert receipt["normalized_chart_sha256"] == producer["normalized_chart_sha256"]
    assert receipt["normalized_input_decoder_sha256"] == producer["decoder_sha256"]
    assert receipt["consumer_prior_sha256"] == consumer["consumer_prior_sha256"]
    assert receipt["fit_evidence_sha256"] == (
        lfm2_normalized_input_mlp_fit_evidence_sha256(
            common["cases"],
            (1.0,),
        )
    )
    assert receipt["source_neurons_dropped"] == 0
    assert receipt["heldout_rows_used_by_solver"] == 0
    assert receipt["model_forward_calls"] == 0
    assert receipt["backward_calls"] == 0
    assert receipt["optimizer_steps"] == 0
    assert receipt["training_steps"] == 0


def test_normalized_input_decoder_preserves_mixed_gamma_anisotropy() -> None:
    parent_gamma = torch.tensor([12.0, 0.08, 3.5], dtype=torch.float64)
    target_gamma = torch.tensor([0.25, 5.0], dtype=torch.float64)
    seed = torch.tensor(
        [[1.0, 0.4], [0.7, -1.2], [0.3, 0.9]],
        dtype=torch.float64,
    )
    residual_chart = torch.linalg.qr(seed, mode="reduced").Q.contiguous()
    normalized_chart = (
        residual_chart
        * parent_gamma.reciprocal().unsqueeze(1)
        * target_gamma.unsqueeze(0)
    )
    expected_decoder = (
        target_gamma.reciprocal().unsqueeze(1)
        * residual_chart.T
        * parent_gamma.unsqueeze(0)
    )
    source_gate = torch.tensor(
        [
            [1.0, 0.25, -0.5],
            [-0.75, 1.5, 0.4],
            [0.3, -2.0, 1.25],
            [1.75, 0.6, -0.8],
        ],
        dtype=torch.float64,
    )
    source_up = torch.tensor(
        [
            [0.5, -1.0, 0.75],
            [1.25, 0.4, -0.6],
            [-0.2, 1.1, 1.5],
            [0.9, -0.7, 0.3],
        ],
        dtype=torch.float64,
    )
    source_down = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [0.4, 1.25, -0.8, 0.2],
            [-0.3, 0.6, 1.1, -1.0],
        ],
        dtype=torch.float64,
    )
    target_input = torch.tensor([[1.0, 0.0], [-1.5, 0.0]], dtype=torch.float64)
    parent_input = target_input @ expected_decoder
    parent_gate = parent_input @ source_gate.T
    parent_up = parent_input @ source_up.T
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    target_residual = torch.tensor([[0.1, -0.2], [0.3, 0.15]], dtype=torch.float64)
    endpoint = target_residual + (parent_post @ source_down.T) @ residual_chart
    case = Lfm2CachedNormalizedInputMlpFoldCase(
        target_input=target_input,
        parent_normalized_input=parent_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=target_residual,
        parent_contribution=parent_post.abs() + 0.125,
    )
    target = _ToyMlp(hidden_size=2, intermediate_size=2, dtype=torch.float64)
    compiled = compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(
        source_gate_weight=source_gate,
        source_up_weight=source_up,
        source_down_weight=source_down,
        target_gate_weight_template=target.w1.weight,
        target_up_weight_template=target.w3.weight,
        target_down_weight_template=target.w2.weight,
        residual_chart=residual_chart,
        residual_chart_sha256=fold_module.lfm2_authenticated_tensor_sha256(
            residual_chart
        ),
        parent_ffn_norm_weight=parent_gamma,
        parent_ffn_norm_weight_sha256=(
            fold_module.lfm2_authenticated_tensor_sha256(parent_gamma)
        ),
        target_ffn_norm_weight=target_gamma,
        target_ffn_norm_weight_sha256=(
            fold_module.lfm2_authenticated_tensor_sha256(target_gamma)
        ),
        cases=(case,),
        anchors_per_source=2,
        max_candidate_targets=2,
        normalized_decoder_ridge=1e-8,
        ridge=0.0,
        cluster_ridge=1e-8,
    )

    # FIT identifies only child coordinate zero.  The exact gamma-conjugated
    # structural inverse must therefore carry unseen coordinate one; a polar
    # decoder has discarded that anisotropy and cannot do so.
    heldout_target_input = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    expected_heldout = heldout_target_input @ expected_decoder
    actual_heldout = heldout_target_input @ compiled.normalized_input_decoder
    polar_u, _, polar_vh = torch.linalg.svd(normalized_chart, full_matrices=False)
    polar_decoder = (polar_u @ polar_vh).T
    polar_fit = target_input @ polar_decoder
    polar_gain = float(
        (polar_fit * parent_input).sum().item() / polar_fit.square().sum().item()
    )
    polar_heldout = heldout_target_input @ (polar_decoder * polar_gain)
    torch.testing.assert_close(actual_heldout, expected_heldout, rtol=1e-10, atol=1e-10)
    assert torch.linalg.vector_norm(polar_heldout - expected_heldout) > 1.0
    producer = compiled.as_dict()["producer_bridge"]
    consumer = compiled.as_dict()["consumer_bridge"]
    assert producer["structural_decoder_prior_rank"] == 2
    assert producer["structural_decoder_prior_operator_norm"] > 0.0
    assert producer["structural_decoder_left_inverse_error"] < 1e-10
    assert "structural_decoder_prior_sha256" in producer
    assert "thin_polar_diagnostic_sha256" in producer
    assert consumer["orthonormal_columns"] is True
    assert consumer["orthogonality_maximum_absolute_error"] < 1e-12
    assert consumer["decoder_method"] == ("authenticated_orthonormal_chart_transpose")
    assert consumer["residual_chart_svd_calls"] == 0
    assert consumer["residual_chart_pseudoinverse_calls"] == 0


def test_normalized_input_decoder_caps_supported_correction_and_authenticates_gamma():
    common, case, normalized_polar = _normalized_input_real_medoid_fixture()
    changed_parent_input = case.parent_normalized_input.clone()
    changed_parent_input[:, 2] = 8.0 * case.target_input[:, 0]
    changed_gate = changed_parent_input @ common["source_gate_weight"].T
    changed_up = changed_parent_input @ common["source_up_weight"].T
    changed_post = torch.nn.functional.silu(changed_gate) * changed_up
    changed_case = replace(
        case,
        parent_normalized_input=changed_parent_input,
        parent_gate=changed_gate,
        parent_up=changed_up,
        parent_post_swiglu=changed_post,
        parent_post_mlp_residual_target_space=(
            case.target_pre_mlp_residual
            + (changed_post @ common["source_down_weight"].T) @ common["residual_chart"]
        ),
        parent_contribution=changed_post.abs() + 0.125,
    )
    changed = dict(common)
    changed["cases"] = (changed_case,)
    compiled = compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(**changed)
    polar_decoder = normalized_polar.T * compiled.normalized_input_global_gain
    correction = compiled.normalized_input_decoder - polar_decoder
    supported_projector = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.linalg.vector_norm(correction) <= (
        torch.linalg.vector_norm(supported_projector @ polar_decoder) + 1e-12
    )
    assert compiled.normalized_input_correction_scale < 1.0
    assert (
        torch.linalg.vector_norm(
            (torch.eye(2, dtype=torch.float64) - supported_projector) @ correction
        )
        < 1e-12
    )

    corrupt = dict(common)
    corrupt["parent_ffn_norm_weight"] = common["parent_ffn_norm_weight"] + 0.25
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="parent_ffn_norm_weight tensor authentication failed",
    ):
        compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(**corrupt)


def test_normalized_input_w2_solve_uses_materialized_bf16_producers() -> None:
    common, case, _ = _normalized_input_real_medoid_fixture(dtype=torch.bfloat16)
    compiled = compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(**common)
    assert compiled.gate_weight.dtype == torch.bfloat16
    assert compiled.up_weight.dtype == torch.bfloat16
    assert compiled.down_weight.dtype == torch.bfloat16
    materialized_post = torch.nn.functional.silu(
        case.target_input @ compiled.gate_weight.to(torch.float64).T
    ) * (case.target_input @ compiled.up_weight.to(torch.float64).T)
    endpoint_target = (
        case.parent_post_mlp_residual_target_space - case.target_pre_mlp_residual
    )
    down_prior = common["residual_chart"].T @ (
        compiled.cluster_plan.aggregate_down_consumer(
            common["source_down_weight"], axis=1
        )
    )
    expected_down, _ = fold_module._fit_linear(
        materialized_post,
        endpoint_target,
        role="oracle_normalized_bridge_materialized_down",
        ridge=0.0,
        prior=down_prior,
        max_prior_delta_ratio=None,
    )
    assert torch.equal(compiled.down_weight, expected_down.to(torch.bfloat16))


def test_short_conv_fold_uses_all_channels_actual_fir_and_endpoint_prior() -> None:
    generator = torch.Generator().manual_seed(1441)
    target_input = torch.randn(1, 7, 2, generator=generator, dtype=torch.float64)
    desired_in_weight = torch.tensor(
        [
            [1.0, 0.2],
            [-0.3, 0.8],
            [0.4, -0.7],
            [0.9, 0.1],
            [-0.6, 0.5],
            [0.25, 1.1],
        ],
        dtype=torch.float64,
    )
    desired_in_bias = torch.tensor(
        [0.1, -0.2, 0.05, 0.15, -0.1, 0.2],
        dtype=torch.float64,
    )
    desired_kernel = torch.tensor([[[0.25, 0.75]], [[-0.5, 0.4]]], dtype=torch.float64)
    desired_conv_bias = torch.tensor([0.125, -0.075], dtype=torch.float64)
    desired_out_weight = torch.tensor([[1.2, -0.4], [0.3, 0.9]], dtype=torch.float64)
    desired_out_bias = torch.tensor([0.05, -0.1], dtype=torch.float64)

    projected = target_input @ desired_in_weight.T + desired_in_bias
    target_b, target_c, target_x = projected.chunk(3, dim=-1)
    target_bx = (target_b * target_x).transpose(1, 2)
    filtered = torch.nn.functional.conv1d(
        torch.nn.functional.pad(target_bx, (1, 0)),
        desired_kernel,
        bias=desired_conv_bias,
        groups=2,
    )
    target_context = target_c * filtered.transpose(1, 2)
    desired_delta = target_context @ desired_out_weight.T + desired_out_bias
    residual = torch.randn(1, 7, 2, generator=generator, dtype=torch.float64)
    endpoint = residual + desired_delta

    def duplicate_channels(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                value[..., 0],
                value[..., 0],
                value[..., 1],
                value[..., 1],
            ),
            dim=-1,
        )

    parent_b = duplicate_channels(target_b)
    parent_c = duplicate_channels(target_c)
    parent_x = duplicate_channels(target_x)
    parent_contribution = duplicate_channels(target_context)
    split = 4
    cases = (
        Lfm2CachedShortConvFoldCase(
            target_input=target_input[:, :split],
            parent_b=parent_b[:, :split],
            parent_c=parent_c[:, :split],
            parent_x=parent_x[:, :split],
            parent_post_operator_residual_target_space=endpoint[:, :split],
            target_pre_operator_residual=residual[:, :split],
            reset_state=True,
            parent_contribution=parent_contribution[:, :split],
        ),
        Lfm2CachedShortConvFoldCase(
            target_input=target_input[:, split:],
            parent_b=parent_b[:, split:],
            parent_c=parent_c[:, split:],
            parent_x=parent_x[:, split:],
            parent_post_operator_residual_target_space=endpoint[:, split:],
            target_pre_operator_residual=residual[:, split:],
            reset_state=False,
            parent_contribution=parent_contribution[:, split:],
        ),
    )
    source_values = {
        "in_proj.weight": torch.randn(12, 4, generator=generator, dtype=torch.float64),
        "in_proj.bias": torch.randn(12, generator=generator, dtype=torch.float64),
        "conv.weight": torch.stack(
            (
                desired_kernel[0],
                desired_kernel[0],
                desired_kernel[1],
                desired_kernel[1],
            )
        ),
        "conv.bias": torch.tensor(
            [
                desired_conv_bias[0],
                desired_conv_bias[0],
                desired_conv_bias[1],
                desired_conv_bias[1],
            ],
            dtype=torch.float64,
        ),
        "out_proj.weight": torch.randn(4, 4, generator=generator, dtype=torch.float64),
        "out_proj.bias": torch.randn(4, generator=generator, dtype=torch.float64),
    }
    source_before = {name: value.clone() for name, value in source_values.items()}
    target = _ToyShortConv(hidden_size=2, fir_length=2, bias=True, dtype=torch.float64)
    assignment = Lfm2SparseResidualAssignment(
        target_width=2,
        source_to_target=(0, 0, 1, 1),
        coefficients=(1.0, 1.0, 1.0, 1.0),
    )

    compiled = compile_lfm2_structural_residual_short_conv_fold(
        source_in_proj_weight=source_values["in_proj.weight"],
        source_in_proj_bias=source_values["in_proj.bias"],
        source_conv_weight=source_values["conv.weight"],
        source_conv_bias=source_values["conv.bias"],
        source_out_proj_weight=source_values["out_proj.weight"],
        source_out_proj_bias=source_values["out_proj.bias"],
        target_in_proj_weight_template=target.in_proj.weight,
        target_in_proj_bias_template=target.in_proj.bias,
        target_conv_weight_template=target.conv.weight,
        target_conv_bias_template=target.conv.bias,
        target_out_proj_weight_template=target.out_proj.weight,
        target_out_proj_bias_template=target.out_proj.bias,
        cases=cases,
        residual_assignment=assignment,
        ridge=0.0,
        max_prior_delta_ratio=None,
    )

    assert compiled.full_source_coverage is True
    assert compiled.residual_assignment_sha256 == assignment.assignment_sha256
    assert compiled.as_dict()["source_channels_dropped"] == 0
    assert sorted(source for group in compiled.channel_groups for source in group) == [
        0,
        1,
        2,
        3,
    ]
    assert compiled.residual_endpoint_relative_rmse < 1e-10
    assert compiled.source_logical_parameters == 92
    assert compiled.target_logical_parameters == 30
    assert compiled.removed_logical_parameters == 62
    assert compiled.parameter_reduction_fraction == pytest.approx(62 / 92)
    assert {
        name: tuple(value.shape) for name, value in compiled.state_dict().items()
    } == {
        "in_proj.weight": (6, 2),
        "conv.weight": (2, 1, 2),
        "out_proj.weight": (2, 2),
        "in_proj.bias": (6,),
        "conv.bias": (2,),
        "out_proj.bias": (2,),
    }
    for name, value in source_values.items():
        assert torch.equal(value, source_before[name])

    parameter_ids = tuple(id(value) for value in target.parameters())
    parameter_shapes = tuple(tuple(value.shape) for value in target.parameters())
    for parameter in target.parameters():
        parameter.grad = torch.zeros_like(parameter)
    gradient_ids = tuple(id(value.grad) for value in target.parameters())
    report = install_lfm2_structural_short_conv_fold(target, compiled)
    assert report.accepted is True
    assert tuple(id(value) for value in target.parameters()) == parameter_ids
    assert tuple(tuple(value.shape) for value in target.parameters()) == (
        parameter_shapes
    )
    assert tuple(id(value.grad) for value in target.parameters()) == gradient_ids

    installed_projection = target.in_proj(target_input)
    installed_b, installed_c, installed_x = installed_projection.chunk(3, dim=-1)
    installed_filtered = torch.nn.functional.conv1d(
        torch.nn.functional.pad((installed_b * installed_x).transpose(1, 2), (1, 0)),
        target.conv.weight,
        bias=target.conv.bias,
        groups=2,
    )
    installed_delta = target.out_proj(installed_c * installed_filtered.transpose(1, 2))
    torch.testing.assert_close(
        residual + installed_delta,
        endpoint,
        rtol=1e-9,
        atol=1e-10,
    )


def test_short_conv_trace_adapter_preserves_endpoint_and_state_order() -> None:
    tensors: dict[str, torch.Tensor] = {}
    for phase, sequence in (("prefill", 3), ("decode.000", 1)):
        prefix = f"{phase}.layer.004."
        tensors[prefix + "operator.input"] = torch.zeros(
            1, sequence, 2, dtype=torch.float64
        )
        for role, fill in (
            ("short_conv.b", 1.0),
            ("short_conv.c", 2.0),
            ("short_conv.x", 3.0),
            ("short_conv.context", 4.0),
        ):
            tensors[prefix + role] = torch.full(
                (1, sequence, 4), fill, dtype=torch.float64
            )
        tensors[prefix + "residual.pre"] = torch.full(
            (1, sequence, 2), 5.0, dtype=torch.float64
        )
        tensors[prefix + "residual.post_operator"] = torch.full(
            (1, sequence, 2), 9.0, dtype=torch.float64
        )

    prefill = cached_short_conv_case_from_lfm2_trace(
        tensors, phase="prefill", layer_index=4
    )
    decode = cached_short_conv_case_from_lfm2_trace(
        tensors, phase="decode.000", layer_index=4
    )

    assert prefill.reset_state is True
    assert decode.reset_state is False
    assert torch.equal(
        prefill.parent_post_operator_residual_target_space
        - prefill.target_pre_operator_residual,
        torch.full((1, 3, 2), 4.0, dtype=torch.float64),
    )
    assert torch.equal(
        prefill.parent_contribution,
        tensors["prefill.layer.004.short_conv.context"],
    )
