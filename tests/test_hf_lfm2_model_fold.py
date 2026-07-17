from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.hf_lfm2_closed_loop_fold import (
    Lfm2AttentionFoldEvidence,
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2MLPFoldEvidence,
    build_lfm2_closed_loop_fold_plan,
)
from neural_forge.hf_lfm2_model_fold import (
    MODEL_FOLD_METHOD,
    Lfm2ModelFoldConfig,
    Lfm2ModelFoldEvidenceError,
    Lfm2ModelFoldTopologyError,
    assemble_lfm2_model_fold_evidence,
    build_leading_pair_lfm2_span_plan,
    build_lfm2_model_fold_plan,
    fingerprint_lfm2_model_feature_state,
    install_lfm2_model_fold_plan,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(
    seed: int,
    *,
    layer_types: tuple[str, ...],
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(layer_types),
        num_attention_heads=query_heads,
        num_key_value_heads=key_value_heads,
        head_dim=2,
        layer_types=list(layer_types),
        conv_L_cache=3,
        conv_bias=True,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    return transformers.Lfm2ForCausalLM(config).eval()


def _mlp_evidence(
    *,
    seed: int,
    parent_post: torch.Tensor,
    source_neurons: int = 6,
    target_neurons: int = 4,
) -> Lfm2MLPFoldEvidence:
    generator = torch.Generator().manual_seed(seed)
    selection_rows = 18
    fit_rows, hidden_size = parent_post.shape
    gate = torch.randn(
        selection_rows,
        source_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    up = torch.randn(
        selection_rows,
        source_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    child_post = torch.randn(
        fit_rows,
        target_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    desired_down = torch.randn(
        hidden_size,
        target_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    child_pre = parent_post - child_post @ desired_down.T
    return Lfm2MLPFoldEvidence(
        parent_gate_rows=gate,
        parent_up_rows=up,
        parent_post_swiglu_rows=torch.nn.functional.silu(gate) * up,
        child_post_swiglu_rows=child_post,
        parent_branch_output_rows=parent_post,
        child_pre_branch_residual_rows=child_pre,
    )


def _attention_evidence(
    *,
    seed: int,
    fit_rows: int,
    hidden_size: int = 8,
    source_query_heads: int = 4,
    source_key_value_heads: int = 2,
    target_query_heads: int = 2,
    head_dim: int = 2,
) -> Lfm2AttentionFoldEvidence:
    generator = torch.Generator().manual_seed(seed)
    selection_rows = 20
    query = torch.randn(
        selection_rows,
        source_query_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    key = torch.randn(
        selection_rows,
        source_key_value_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    value = torch.randn(
        selection_rows,
        source_key_value_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    context = torch.randn(
        selection_rows,
        source_query_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    child_context = torch.randn(
        fit_rows,
        target_query_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    desired_output = torch.randn(
        hidden_size,
        target_query_heads * head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    child_pre = torch.randn(
        fit_rows,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    return Lfm2AttentionFoldEvidence(
        parent_query_rows=query,
        parent_key_rows=key,
        parent_value_rows=value,
        parent_context_rows=context,
        child_context_rows=child_context,
        parent_branch_output_rows=(
            child_pre + child_context.reshape(fit_rows, -1) @ desired_output.T
        ),
        child_pre_branch_residual_rows=child_pre,
    )


def _seed_producer_shell(
    parent,
    child,
    layer_evidence: dict[int, Lfm2ClosedLoopFoldEvidence],
    layer_config: Lfm2ClosedLoopFoldConfig,
) -> None:
    bindings = (
        (parent.model.layers[1], child.model.layers[0], layer_evidence[1]),
        (parent.model.layers[2], child.model.layers[1], layer_evidence[2]),
    )
    with torch.no_grad():
        for source, target, evidence in bindings:
            plan = build_lfm2_closed_loop_fold_plan(
                source,
                target,
                evidence,
                config=layer_config,
            )
            candidate = plan.candidate_state_dict()
            keys = {
                "feed_forward.w1.weight",
                "feed_forward.w3.weight",
            }
            if plan.attention is not None:
                keys.update(
                    {
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    }
                )
            else:
                keys.update(name for name in candidate if name.startswith("conv."))
            target_state = target.state_dict()
            for name in keys:
                target_state[name].copy_(candidate[name])


def _fixture():
    parent = _model(
        10,
        layer_types=("conv", "conv", "full_attention"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    )
    child = _model(
        20,
        layer_types=("conv", "full_attention"),
        intermediate_size=4,
        query_heads=2,
        key_value_heads=1,
    )
    rows = 20
    generator = torch.Generator().manual_seed(30)
    pre_0 = torch.zeros((rows, 8), dtype=torch.float64)
    post_0 = pre_0.clone()
    post_1 = torch.randn(
        rows,
        8,
        generator=generator,
        dtype=torch.float64,
    )
    pre_1 = post_0
    pre_2 = post_1
    post_2 = pre_2 + torch.randn(
        rows,
        8,
        generator=generator,
        dtype=torch.float64,
    )
    layer_evidence = {
        0: Lfm2ClosedLoopFoldEvidence(
            mlp=_mlp_evidence(seed=40, parent_post=post_0)
        ),
        1: Lfm2ClosedLoopFoldEvidence(
            mlp=_mlp_evidence(seed=41, parent_post=post_1)
        ),
        2: Lfm2ClosedLoopFoldEvidence(
            attention=_attention_evidence(
                seed=42,
                fit_rows=rows,
            ),
            mlp=_mlp_evidence(seed=43, parent_post=post_2),
        ),
    }
    spans = build_leading_pair_lfm2_span_plan(3)
    layer_config = Lfm2ClosedLoopFoldConfig(
        endpoint_ridge=0.0,
        max_endpoint_relative_rmse=1e-8,
        min_endpoint_cosine_similarity=0.99999999,
    )
    _seed_producer_shell(parent, child, layer_evidence, layer_config)
    evidence = assemble_lfm2_model_fold_evidence(
        cache_fingerprint="a" * 64,
        child_feature_state_sha256=fingerprint_lfm2_model_feature_state(child),
        spans=spans,
        layer_fold_evidence=layer_evidence,
        parent_pre_residual_rows={0: pre_0, 1: pre_1, 2: pre_2},
        parent_post_residual_rows={0: post_0, 1: post_1, 2: post_2},
    )
    config = Lfm2ModelFoldConfig(
        layer_config=layer_config,
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1e9,
    )
    return parent, child, spans, evidence, config


def _state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_state_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_leading_pair_plan_matches_the_14_to_13_target() -> None:
    spans = build_leading_pair_lfm2_span_plan()

    assert len(spans) == 13
    assert spans[0].source_layer_indices == (0, 1)
    assert spans[0].target_layer_index == 0
    assert [
        (span.source_layer_indices, span.target_layer_index)
        for span in spans[1:]
    ] == [((index,), index - 1) for index in range(2, 14)]


def test_hybrid_plan_fits_full_span_endpoint_and_installs_one_full_model() -> None:
    parent, child, spans, evidence, config = _fixture()
    parent_before = _state(parent)
    child_before = _state(child)

    plan = build_lfm2_model_fold_plan(
        parent,
        child,
        spans,
        evidence,
        config=config,
    )

    _assert_state_equal(parent_before, _state(parent))
    _assert_state_equal(child_before, _state(child))
    assert plan.span_plans[0].anchor_source_layer_index == 1
    assert plan.span_plans[0].span.source_layer_indices == (0, 1)
    assert plan.span_plans[0].exact_composition_claimed is False
    assert plan.span_plans[0].endpoint_correction_bounded is True
    assert (
        plan.span_plans[0].layer_plan.mlp.endpoint_fit.solver
        == "bounded-survivor-anchor-residual-ridge"
    )
    assert plan.span_plans[0].layer_plan.target_contract.ffn_intermediate_size == 4
    assert plan.span_plans[1].layer_plan.target_contract.ffn_intermediate_size == 4
    assert plan.span_plans[1].layer_plan.attention is not None
    assert plan.span_plans[1].layer_plan.attention.source_query_heads == 4
    assert plan.span_plans[1].layer_plan.attention.target_query_heads == 2
    assert plan.parameters_removed > 0
    report = plan.as_dict()
    assert report["method_id"] == MODEL_FOLD_METHOD
    assert report["source_layer_span_coverage_complete"] is True
    assert report["complete_target_state_coverage"] is True
    assert report["atomic_model_level_transaction"] is True
    assert report["child_feature_state_authentication"][
        "matched_live_child_during_planning"
    ] is True
    assert report["child_feature_state_authentication"][
        "all_target_layer_producer_shells_match_selected_plans"
    ] is True
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    depth = report["span_plans"][0]["depth_composition"]
    assert depth["exact_composition_claimed"] is False
    assert depth["fit_domain"] == (
        "cached-span-input-to-parent-last-layer-post-residual"
    )

    transaction = install_lfm2_model_fold_plan(child, plan)
    candidate = dict(plan.candidate_state_dict())
    _assert_state_equal(candidate, _state(child))
    committed = transaction.commit()
    assert committed.accepted is True
    assert committed.rollback_exact is None
    assert committed.zero_gradient is True
    assert committed.as_dict()["parameter_gradients_written"] is False

    parent_state = _state(parent)
    for name in plan.copied_non_layer_state_keys:
        assert torch.equal(candidate[name], parent_state[name]), name


def test_whole_model_rollback_restores_bytes_parameter_and_gradient_objects() -> None:
    parent, child, spans, evidence, config = _fixture()
    for parameter in child.parameters():
        parameter.grad = torch.randn_like(parameter)
    before = _state(child)
    parameter_ids = tuple(id(parameter) for parameter in child.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in child.parameters())
    gradient_values = tuple(
        parameter.grad.detach().clone() for parameter in child.parameters()
    )
    plan = build_lfm2_model_fold_plan(
        parent,
        child,
        spans,
        evidence,
        config=config,
    )

    transaction = install_lfm2_model_fold_plan(child, plan)
    assert any(
        not torch.equal(before[name], child.state_dict()[name].detach().cpu())
        for name in before
    )
    report = transaction.rollback()

    _assert_state_equal(before, _state(child))
    assert report.accepted is False
    assert report.rollback_exact is True
    assert tuple(id(parameter) for parameter in child.parameters()) == parameter_ids
    assert tuple(id(parameter.grad) for parameter in child.parameters()) == gradient_ids
    for parameter, expected in zip(
        child.parameters(),
        gradient_values,
        strict=True,
    ):
        assert torch.equal(parameter.grad, expected)
    assert report.as_dict()["optimizer_steps"] == 0
    assert report.as_dict()["backward_calls"] == 0


def test_blank_child_designs_are_rejected_by_producer_shell_authentication() -> None:
    parent = _model(
        80,
        layer_types=("conv", "conv"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    )
    child = _model(
        81,
        layer_types=("conv",),
        intermediate_size=4,
        query_heads=2,
        key_value_heads=1,
    )
    with torch.no_grad():
        for parameter in child.parameters():
            parameter.zero_()
    child_before = _state(child)
    rows = 16
    selection_rows = 18
    generator = torch.Generator().manual_seed(82)
    zero_residual = torch.zeros((rows, 8), dtype=torch.float64)
    blank_design = torch.zeros((rows, 4), dtype=torch.float64)

    def blank_layer_evidence() -> Lfm2ClosedLoopFoldEvidence:
        gate = torch.randn(
            selection_rows,
            6,
            generator=generator,
            dtype=torch.float64,
        )
        up = torch.randn(
            selection_rows,
            6,
            generator=generator,
            dtype=torch.float64,
        )
        return Lfm2ClosedLoopFoldEvidence(
            mlp=Lfm2MLPFoldEvidence(
                parent_gate_rows=gate,
                parent_up_rows=up,
                parent_post_swiglu_rows=torch.nn.functional.silu(gate) * up,
                child_post_swiglu_rows=blank_design,
                parent_branch_output_rows=zero_residual,
                child_pre_branch_residual_rows=zero_residual,
            )
        )

    spans = build_leading_pair_lfm2_span_plan(2)
    evidence = assemble_lfm2_model_fold_evidence(
        cache_fingerprint="b" * 64,
        child_feature_state_sha256=fingerprint_lfm2_model_feature_state(child),
        spans=spans,
        layer_fold_evidence={
            0: blank_layer_evidence(),
            1: blank_layer_evidence(),
        },
        parent_pre_residual_rows={
            0: zero_residual,
            1: zero_residual,
        },
        parent_post_residual_rows={
            0: zero_residual,
            1: zero_residual,
        },
    )
    config = Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=0.0,
            max_endpoint_relative_rmse=1e9,
            min_endpoint_cosine_similarity=-1.0,
        ),
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1.0,
    )

    with pytest.raises(
        Lfm2ModelFoldEvidenceError,
        match="zero/random blank",
    ):
        build_lfm2_model_fold_plan(
            parent,
            child,
            spans,
            evidence,
            config=config,
        )

    _assert_state_equal(child_before, _state(child))


def test_span_and_cache_evidence_fail_closed_without_model_mutation() -> None:
    parent, child, spans, evidence, config = _fixture()
    parent_before = _state(parent)
    child_before = _state(child)

    stale = replace(evidence, child_feature_state_sha256="c" * 64)
    with pytest.raises(
        Lfm2ModelFoldEvidenceError,
        match="child_feature_state_sha256",
    ):
        build_lfm2_model_fold_plan(
            parent,
            child,
            spans,
            stale,
            config=config,
        )

    with pytest.raises(Lfm2ModelFoldTopologyError, match="one source span"):
        build_lfm2_model_fold_plan(
            parent,
            child,
            spans[:-1],
            evidence,
            config=config,
        )
    _assert_state_equal(parent_before, _state(parent))
    _assert_state_equal(child_before, _state(child))

    broken_pre = dict(
        (index, layer.parent_pre_residual_rows)
        for span in evidence.spans
        for index, layer in (
            (layer.source_layer_index, layer) for layer in span.layers
        )
    )
    broken_pre[1] = broken_pre[1] + 1.0
    broken = assemble_lfm2_model_fold_evidence(
        cache_fingerprint="a" * 64,
        child_feature_state_sha256=fingerprint_lfm2_model_feature_state(child),
        spans=spans,
        layer_fold_evidence={
            layer.source_layer_index: layer.fold_evidence
            for span in evidence.spans
            for layer in span.layers
        },
        parent_pre_residual_rows=broken_pre,
        parent_post_residual_rows={
            layer.source_layer_index: layer.parent_post_residual_rows
            for span in evidence.spans
            for layer in span.layers
        },
    )
    with pytest.raises(Lfm2ModelFoldEvidenceError, match="span chain"):
        build_lfm2_model_fold_plan(
            parent,
            child,
            spans,
            broken,
            config=config,
        )
    _assert_state_equal(parent_before, _state(parent))
    _assert_state_equal(child_before, _state(child))
