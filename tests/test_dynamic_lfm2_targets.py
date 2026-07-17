from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from neural_forge.dynamic_lfm2_targets import (
    DynamicLfm2TargetError,
    LFM2_230M_BASE_UNIQUE_PARAMETERS,
    LFM2_230M_DENSE_TIED_VOCAB_UNIQUE_PARAMETERS,
    LFM2_230M_FACTORIZED_TIED_VOCAB_PARAMETERS_PER_RANK,
    LFM2_230M_HYBRID_FACTORIZED_VOCAB_TARGET_FRONTIER,
    LFM2_230M_HYBRID_TARGET_FRONTIER,
    LFM2_230M_HALF_SIZE_STRUCTURAL_FRONTIER,
    LFM2_230M_SOURCE_LAYER_TYPES,
    LFM2_230M_SOURCE_UNIQUE_PARAMETERS,
    Lfm2HybridFactorizedVocabularyTarget,
    Lfm2OrderedSourceSpan,
    Lfm2Topology,
    lfm2_230m_hybrid_factorized_vocab_parameter_count,
    lfm2_230m_hybrid_factorized_vocab_target_frontier,
    lfm2_230m_hybrid_parameter_count,
    lfm2_230m_hybrid_target_frontier,
    lfm2_230m_half_size_structural_frontier,
    lfm2_230m_half_size_structural_support_report,
    plan_lfm2_progressive_targets,
    solve_lfm2_230m_hybrid_factorized_vocab_target,
    solve_lfm2_230m_hybrid_structural_target,
)


def test_lfm_plan_starts_with_one_mlp_channel_and_delays_attention() -> None:
    plan = plan_lfm2_progressive_targets(
        source_unique_parameters=229_693_184,
        hidden_size=1024,
        head_dim=64,
        layer_types=(
            "conv",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
        source=Lfm2Topology(16, 8, 2560),
        target=Lfm2Topology(14, 7, 2528),
    )

    assert plan.stages[0].topology == Lfm2Topology(16, 8, 2559)
    assert plan.stages[0].changed_circuit == "mlp"
    assert plan.stages[0].incremental_parameters_removed == 43_008
    assert [
        2560 - stage.topology.intermediate_size
        for stage in plan.stages
        if stage.changed_circuit == "mlp"
    ] == [1, 2, 4, 8, 16, 32]
    assert all(
        stage.topology.query_heads == 16
        for stage in plan.stages
        if stage.changed_circuit == "mlp"
    )
    assert plan.stages[-1].topology == Lfm2Topology(14, 7, 2528)
    assert plan.stages[-1].changed_circuit == "attention"
    assert plan.stages[-1].unique_parameters == 225_957_632
    assert plan.as_dict()["smallest_representable_stage_first"] is True


def test_lfm_plan_rejects_changed_gqa_ratio() -> None:
    with pytest.raises(DynamicLfm2TargetError, match="stable GQA"):
        plan_lfm2_progressive_targets(
            source_unique_parameters=100_000,
            hidden_size=16,
            head_dim=4,
            layer_types=("full_attention",),
            source=Lfm2Topology(4, 2, 24),
            target=Lfm2Topology(3, 1, 20),
        )


def test_lfm_identity_target_has_no_stages() -> None:
    source = Lfm2Topology(4, 2, 24)
    plan = plan_lfm2_progressive_targets(
        source_unique_parameters=100_000,
        hidden_size=16,
        head_dim=4,
        layer_types=("full_attention", "conv"),
        source=source,
        target=source,
    )

    assert plan.stages == ()
    assert plan.as_dict()["final_unique_parameters"] == 100_000


def test_lfm_230m_hybrid_frontier_has_exact_verified_targets() -> None:
    expected = (
        (1_000, 13, 7, 6, 14, 7, 2_346, 206_723_840),
        (2_000, 11, 6, 5, 14, 7, 2_299, 183_767_680),
        (3_000, 9, 5, 4, 14, 7, 2_230, 160_780_800),
        (5_000, 6, 3, 3, 14, 7, 1_458, 114_846_080),
    )
    frontier = lfm2_230m_hybrid_target_frontier()

    assert frontier is LFM2_230M_HYBRID_TARGET_FRONTIER
    assert tuple(
        (
            target.requested_reduction_basis_points,
            target.layer_count,
            target.convolution_layer_count,
            target.attention_layer_count,
            target.topology.query_heads,
            target.topology.key_value_heads,
            target.topology.intermediate_size,
            target.unique_parameters,
        )
        for target in frontier
    ) == expected
    assert tuple(target.parameters_removed for target in frontier) == (
        22_969_344,
        45_925_504,
        68_912_384,
        114_847_104,
    )
    assert all(
        target.unique_parameters
        == lfm2_230m_hybrid_parameter_count(
            layer_types=target.layer_types,
            topology=target.topology,
        )
        for target in frontier
    )


def test_lfm_230m_hybrid_frontier_covers_every_source_layer_once_in_order() -> None:
    for target in lfm2_230m_hybrid_target_frontier():
        assert tuple(
            span.target_layer_index for span in target.source_spans
        ) == tuple(range(target.layer_count))
        assert tuple(
            source_index
            for span in target.source_spans
            for source_index in span.source_layer_indices
        ) == tuple(range(len(LFM2_230M_SOURCE_LAYER_TYPES)))
        assert target.as_dict()["complete_ordered_source_coverage"] is True


def test_lfm_230m_ten_percent_target_has_minimal_conv_span_composition() -> None:
    target = lfm2_230m_hybrid_target_frontier()[0]

    assert target.layer_types == (
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )
    assert target.source_spans[0] == Lfm2OrderedSourceSpan(0, (0, 1))
    assert tuple(
        span.source_layer_indices for span in target.source_spans[1:]
    ) == tuple((source_index,) for source_index in range(2, 14))


def test_lfm_230m_hybrid_parameter_accounting_reconstructs_parent() -> None:
    assert (
        lfm2_230m_hybrid_parameter_count(
            layer_types=LFM2_230M_SOURCE_LAYER_TYPES,
            topology=Lfm2Topology(16, 8, 2_560),
        )
        == LFM2_230M_SOURCE_UNIQUE_PARAMETERS
    )


def test_lfm_230m_half_size_frontier_solves_depth_heads_and_ffn_jointly() -> None:
    frontier = lfm2_230m_half_size_structural_frontier()

    assert frontier is LFM2_230M_HALF_SIZE_STRUCTURAL_FRONTIER
    assert tuple(
        (
            target.layer_count,
            target.convolution_layer_count,
            target.attention_layer_count,
            target.topology.query_heads,
            target.topology.key_value_heads,
            target.topology.intermediate_size,
            target.unique_parameters,
            target.parameters_removed,
        )
        for target in frontier
    ) == (
        (8, 5, 3, 16, 8, 703, 114_827_648, 114_865_536),
        (10, 6, 4, 14, 7, 375, 114_845_184, 114_848_000),
        (12, 7, 5, 12, 6, 177, 114_838_144, 114_855_040),
        (14, 8, 6, 8, 4, 109, 114_843_392, 114_849_792),
    )
    assert tuple(
        LFM2_230M_SOURCE_UNIQUE_PARAMETERS // 2 - target.unique_parameters
        for target in frontier
    ) == (18_944, 1_408, 8_448, 3_200)
    assert all(target.achieved_reduction >= 0.50 for target in frontier)


def test_lfm_230m_half_size_frontier_preserves_operator_ratio_and_anchors() -> None:
    for target in lfm2_230m_half_size_structural_frontier():
        expected_convolution_count = round(
            target.layer_count
            * LFM2_230M_SOURCE_LAYER_TYPES.count("conv")
            / len(LFM2_230M_SOURCE_LAYER_TYPES)
        )
        assert target.convolution_layer_count == expected_convolution_count
        assert tuple(
            source_index
            for span in target.source_spans
            for source_index in span.source_layer_indices
        ) == tuple(range(14))
        assert all(
            target_type
            in {
                LFM2_230M_SOURCE_LAYER_TYPES[source_index]
                for source_index in span.source_layer_indices
            }
            for target_type, span in zip(
                target.layer_types,
                target.source_spans,
                strict=True,
            )
        )


def test_lfm_230m_half_size_solver_prefers_count_then_attention() -> None:
    expected = LFM2_230M_HALF_SIZE_STRUCTURAL_FRONTIER[1]
    solved = solve_lfm2_230m_hybrid_structural_target(
        layer_types=expected.layer_types,
        source_spans=expected.source_spans,
    )

    assert solved == expected
    assert solved.topology == Lfm2Topology(14, 7, 375)
    # Q4/KV2/FFN631 has the same exact count.  The solver deliberately keeps
    # the higher-attention member for subsequent causal evaluation.
    assert (
        lfm2_230m_hybrid_parameter_count(
            layer_types=solved.layer_types,
            topology=Lfm2Topology(4, 2, 631),
        )
        == solved.unique_parameters
    )


def test_lfm_230m_half_size_frontier_has_existing_physical_routes() -> None:
    reports = lfm2_230m_half_size_structural_support_report()

    assert tuple(report["depth"] for report in reports) == (8, 10, 12, 14)
    assert all(report["physical_route_implemented"] is True for report in reports)
    assert all(
        report["causal_intelligence_retention_proven"] is False
        for report in reports
    )
    assert all(
        "hybrid_producer_first" in report["producer_first_installer"]
        for report in reports[:3]
    )
    assert (
        reports[-1]["producer_first_installer"]
        == "hf_lfm2_producer_first_model_fold."
        "install_lfm2_producer_first_model_fold"
    )


def test_lfm_230m_dense_interface_alone_exceeds_75_percent_child_budget() -> None:
    target_unique_parameters = LFM2_230M_SOURCE_UNIQUE_PARAMETERS // 4

    assert LFM2_230M_BASE_UNIQUE_PARAMETERS > target_unique_parameters
    assert (
        LFM2_230M_DENSE_TIED_VOCAB_UNIQUE_PARAMETERS
        == 65_536 * 1_024
    )


def test_lfm_230m_factorized_vocab_makes_75_percent_mechanically_feasible() -> None:
    frontier = lfm2_230m_hybrid_factorized_vocab_target_frontier()

    assert frontier is LFM2_230M_HYBRID_FACTORIZED_VOCAB_TARGET_FRONTIER
    assert len(frontier) == 1
    target = frontier[0]
    assert isinstance(target, Lfm2HybridFactorizedVocabularyTarget)
    assert target.requested_reduction_basis_points == 7_500
    assert target.dense_hybrid_target is LFM2_230M_HYBRID_TARGET_FRONTIER[-1]
    assert target.vocabulary_rank == 145
    assert target.unique_parameters == 57_388_416
    assert target.parameters_removed == 172_304_768
    assert target.budget_headroom == 34_880
    assert target.achieved_reduction >= 0.75
    report = target.as_dict()
    assert report["mechanical_budget_only"] is True
    assert report["intelligence_retention_proven"] is False
    assert report["joint_causal_allocation_required"] is True
    assert report["vocabulary"]["dense_matrix_retained"] is False


def test_lfm_230m_factorized_vocab_solver_uses_largest_rank_under_budget() -> None:
    dense_target = LFM2_230M_HYBRID_TARGET_FRONTIER[-1]
    target = solve_lfm2_230m_hybrid_factorized_vocab_target(
        dense_target,
        requested_reduction_basis_points=7_500,
    )
    next_rank_count = lfm2_230m_hybrid_factorized_vocab_parameter_count(
        layer_types=dense_target.layer_types,
        topology=dense_target.topology,
        vocabulary_rank=target.vocabulary_rank + 1,
    )

    assert (
        next_rank_count
        == target.unique_parameters
        + LFM2_230M_FACTORIZED_TIED_VOCAB_PARAMETERS_PER_RANK
    )
    assert next_rank_count > target.requested_target_unique_parameters


def test_lfm_230m_factorized_vocab_solver_rejects_impossible_body_budget() -> None:
    with pytest.raises(
        DynamicLfm2TargetError,
        match="positive tied vocabulary rank",
    ):
        solve_lfm2_230m_hybrid_factorized_vocab_target(
            LFM2_230M_HYBRID_TARGET_FRONTIER[-1],
            requested_reduction_basis_points=9_000,
        )


def test_lfm_hybrid_target_is_immutable_and_rejects_inexact_accounting() -> None:
    target = lfm2_230m_hybrid_target_frontier()[0]

    with pytest.raises(FrozenInstanceError):
        target.unique_parameters = 1  # type: ignore[misc]
    with pytest.raises(
        DynamicLfm2TargetError,
        match="unique parameter accounting is inconsistent",
    ):
        replace(target, unique_parameters=target.unique_parameters - 1)


def test_lfm_hybrid_target_rejects_incomplete_ordered_source_coverage() -> None:
    target = lfm2_230m_hybrid_target_frontier()[0]
    missing_source_layer = (
        Lfm2OrderedSourceSpan(0, (0,)),
        *target.source_spans[1:],
    )

    with pytest.raises(
        DynamicLfm2TargetError,
        match="complete ordered source coverage",
    ):
        replace(target, source_spans=missing_source_layer)


def test_lfm_ordered_source_span_rejects_noncontiguous_or_duplicate_layers() -> None:
    with pytest.raises(DynamicLfm2TargetError, match="ordered, contiguous"):
        Lfm2OrderedSourceSpan(0, (0, 2))
    with pytest.raises(DynamicLfm2TargetError, match="ordered, contiguous"):
        Lfm2OrderedSourceSpan(0, (0, 0))
