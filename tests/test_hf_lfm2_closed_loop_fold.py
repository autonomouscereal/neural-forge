from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from neural_forge.hf_activation_cluster import (  # noqa: E402
    build_gated_mlp_activation_cluster_plan,
)
from neural_forge.hf_lfm2_closed_loop_fold import (  # noqa: E402
    Lfm2AttentionFoldEvidence,
    Lfm2ClosedLoopEvidenceError,
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2ClosedLoopTopologyError,
    Lfm2MLPFoldEvidence,
    build_lfm2_closed_loop_fold_plan,
    install_lfm2_closed_loop_fold_plan,
    snapshot_lfm2_layer,
)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float64))


class Lfm2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=torch.float64,
        )
        self.w3 = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=torch.float64,
        )
        self.w2 = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=torch.float64,
        )


class Lfm2ShortConv(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            config.conv_L_cache,
            groups=hidden_size,
            bias=config.conv_bias,
            padding=config.conv_L_cache - 1,
            dtype=torch.float64,
        )
        self.in_proj = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=config.conv_bias,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=config.conv_bias,
            dtype=torch.float64,
        )


class Lfm2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.float64,
        )
        self.q_layernorm = Lfm2RMSNorm(self.head_dim)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim)


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int = 0):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)


for _class in (
    Lfm2RMSNorm,
    Lfm2MLP,
    Lfm2ShortConv,
    Lfm2Attention,
    Lfm2DecoderLayer,
):
    _class.__module__ = _LFM2_MODULE


def _config(
    *,
    kind: str,
    hidden_size: int = 8,
    intermediate_size: int = 6,
    query_heads: int = 16,
    key_value_heads: int = 8,
    head_dim: int = 2,
    conv_bias: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="lfm2",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        conv_L_cache=3,
        conv_bias=conv_bias,
        layer_types=[kind],
        num_attention_heads=query_heads,
        num_key_value_heads=key_value_heads,
        num_heads=query_heads,
        head_dim=head_dim,
        block_ff_dim=intermediate_size,
        block_auto_adjust_ff_dim=False,
    )


def _layer(seed: int, **config_values) -> Lfm2DecoderLayer:
    torch.manual_seed(seed)
    return Lfm2DecoderLayer(_config(**config_values))


def _state_clone(layer: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in layer.state_dict().items()
    }


def _assert_state_equal(
    left: dict[str, torch.Tensor] | nn.Module,
    right: dict[str, torch.Tensor] | nn.Module,
) -> None:
    left_state = _state_clone(left) if isinstance(left, nn.Module) else left
    right_state = _state_clone(right) if isinstance(right, nn.Module) else right
    assert set(left_state) == set(right_state)
    for name in left_state:
        assert torch.equal(left_state[name], right_state[name]), name


def _mlp_evidence(
    *,
    source_neurons: int,
    target_neurons: int,
    hidden_size: int,
    seed: int,
    fit_rows: int = 48,
) -> tuple[Lfm2MLPFoldEvidence, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    selection_rows = 32
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
    post = torch.nn.functional.silu(gate) * up
    contribution = torch.randn(
        selection_rows,
        source_neurons,
        hidden_size,
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
    residual = torch.randn(
        fit_rows,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    endpoint = child_post @ desired_down.T
    validation_post = torch.randn(
        fit_rows // 2,
        target_neurons,
        generator=generator,
        dtype=torch.float64,
    )
    validation_residual = torch.randn(
        fit_rows // 2,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    return (
        Lfm2MLPFoldEvidence(
            parent_gate_rows=gate,
            parent_up_rows=up,
            parent_post_swiglu_rows=post,
            parent_contribution_rows=contribution,
            child_post_swiglu_rows=child_post,
            parent_branch_output_rows=residual + endpoint,
            child_pre_branch_residual_rows=residual,
            validation_child_post_swiglu_rows=validation_post,
            validation_parent_branch_output_rows=(
                validation_residual + validation_post @ desired_down.T
            ),
            validation_child_pre_branch_residual_rows=validation_residual,
        ),
        desired_down,
    )


def _attention_evidence(
    *,
    source_query_heads: int,
    source_key_value_heads: int,
    target_query_heads: int,
    head_dim: int,
    hidden_size: int,
    seed: int,
) -> tuple[Lfm2AttentionFoldEvidence, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    selection_rows = 36
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
    contribution = torch.randn(
        selection_rows,
        source_query_heads,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    fit_rows = target_query_heads * head_dim + 20
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
    residual = torch.randn(
        fit_rows,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    endpoint = child_context.reshape(fit_rows, -1) @ desired_output.T
    validation_context = torch.randn(
        24,
        target_query_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    validation_residual = torch.randn(
        24,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
    )
    return (
        Lfm2AttentionFoldEvidence(
            parent_query_rows=query,
            parent_key_rows=key,
            parent_value_rows=value,
            parent_context_rows=context,
            parent_contribution_rows=contribution,
            child_context_rows=child_context,
            parent_branch_output_rows=residual + endpoint,
            child_pre_branch_residual_rows=residual,
            validation_child_context_rows=validation_context,
            validation_parent_branch_output_rows=(
                validation_residual
                + validation_context.reshape(24, -1) @ desired_output.T
            ),
            validation_child_pre_branch_residual_rows=validation_residual,
        ),
        desired_output,
    )


def _attention_transaction_fixture(seed: int = 100):
    source = _layer(
        seed,
        kind="full_attention",
        intermediate_size=6,
        query_heads=16,
        key_value_heads=8,
        head_dim=2,
    )
    target = _layer(
        seed + 1,
        kind="full_attention",
        intermediate_size=4,
        query_heads=14,
        key_value_heads=7,
        head_dim=2,
    )
    attention, desired_output = _attention_evidence(
        source_query_heads=16,
        source_key_value_heads=8,
        target_query_heads=14,
        head_dim=2,
        hidden_size=8,
        seed=seed + 2,
    )
    mlp, desired_down = _mlp_evidence(
        source_neurons=6,
        target_neurons=4,
        hidden_size=8,
        seed=seed + 3,
    )
    evidence = Lfm2ClosedLoopFoldEvidence(mlp=mlp, attention=attention)
    config = Lfm2ClosedLoopFoldConfig(
        endpoint_ridge=0.0,
        max_endpoint_relative_rmse=1e-9,
        min_endpoint_cosine_similarity=0.999999999,
    )
    return source, target, evidence, config, desired_output, desired_down


def test_real_16q8kv_to_14q7kv_transaction_is_coupled_and_endpoint_fitted() -> None:
    (
        source,
        target,
        evidence,
        config,
        desired_output,
        desired_down,
    ) = _attention_transaction_fixture()
    source_before = _state_clone(source)
    target_before = _state_clone(target)

    plan = build_lfm2_closed_loop_fold_plan(
        source,
        target,
        evidence,
        config=config,
    )

    _assert_state_equal(source_before, source)
    _assert_state_equal(target_before, target)
    assert plan.attention is not None
    attention = plan.attention
    assert attention.source_query_heads == 16
    assert attention.source_key_value_heads == 8
    assert attention.target_query_heads == 14
    assert attention.target_key_value_heads == 7
    assert attention.cluster_plan.head_dim == 2
    assert attention.cluster_plan.source_query_heads_per_group == 2
    assert attention.cluster_plan.target_query_heads_per_group == 2
    assert attention.cluster_plan.report.full_source_coverage is True
    assert sum(attention.cluster_plan.report.group_sizes) == 8
    assert max(attention.cluster_plan.report.group_sizes) == 2
    assert attention.endpoint_fit.solver == "full-primal-residual-ridge"
    assert attention.endpoint_fit.correction_rank_bound == 28
    assert plan.mlp.cluster_plan.report.full_source_coverage is True
    assert plan.mlp.endpoint_fit.solver == "full-primal-residual-ridge"
    assert plan.zero_gradient is True
    assert plan.seed_erasure_by_construction is True
    assert plan.as_dict()["optimizer_steps"] == 0
    assert plan.as_dict()["backward_calls"] == 0

    source_query = source.self_attn.q_proj.weight.reshape(16, 2, 8)
    folded_query = attention.query_weight.reshape(14, 2, 8)
    for target_head, source_head in enumerate(
        attention.cluster_plan.query_head_medoids
    ):
        assert torch.equal(folded_query[target_head], source_query[source_head])
    source_key = source.self_attn.k_proj.weight.reshape(8, 2, 8)
    folded_key = attention.key_weight.reshape(7, 2, 8)
    for target_group, source_group in enumerate(attention.cluster_plan.group_medoids):
        assert torch.equal(folded_key[target_group], source_key[source_group])
    torch.testing.assert_close(
        attention.output_weight,
        desired_output,
        atol=1e-9,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        plan.mlp.down_weight,
        desired_down,
        atol=1e-9,
        rtol=1e-9,
    )

    transaction = install_lfm2_closed_loop_fold_plan(target, plan)
    candidate = plan.candidate_state_dict()
    for name, value in target.state_dict().items():
        assert torch.equal(value.detach().cpu(), candidate[name]), name
    report = transaction.commit()
    assert report.accepted is True
    assert report.rollback_exact is None
    assert report.zero_gradient is True
    assert report.as_dict()["optimizer_steps"] == 0
    assert report.as_dict()["parameter_gradients_written"] is False


def test_endpoint_correction_can_be_bounded_to_the_cluster_prior() -> None:
    source, target, evidence, _config, _attention, _down = (
        _attention_transaction_fixture(seed=120)
    )
    plan = build_lfm2_closed_loop_fold_plan(
        source,
        target,
        evidence,
        config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=0.0,
            max_endpoint_relative_rmse=1e9,
            min_endpoint_cosine_similarity=-1.0,
            max_endpoint_prior_delta_ratio=0.0,
        ),
    )

    assert plan.attention is not None
    for endpoint_fit in (
        plan.attention.endpoint_fit,
        plan.mlp.endpoint_fit,
    ):
        assert endpoint_fit.solver == "cluster-prior-without-endpoint-solve"
        assert endpoint_fit.correction_rank_bound == 0
        assert endpoint_fit.correction_anchor_indices == ()
        endpoint = endpoint_fit.fit
        assert endpoint.clipped is True
        assert endpoint.update_scale == 0.0
        assert endpoint.prior_delta_norm == 0.0


def test_short_conv_is_exact_and_large_mlp_uses_bounded_anchor_correction() -> None:
    source = _layer(
        201,
        kind="conv",
        hidden_size=6,
        intermediate_size=14,
        conv_bias=True,
    )
    target = _layer(
        202,
        kind="conv",
        hidden_size=6,
        intermediate_size=12,
        conv_bias=True,
    )
    generator = torch.Generator().manual_seed(203)
    selection_rows = 24
    gate = torch.randn(selection_rows, 14, generator=generator, dtype=torch.float64)
    up = torch.randn(selection_rows, 14, generator=generator, dtype=torch.float64)
    post = torch.nn.functional.silu(gate) * up
    contribution = torch.randn(
        selection_rows, 14, 6, generator=generator, dtype=torch.float64
    )
    external_cluster = build_gated_mlp_activation_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=12,
        source_neurons=14,
    )
    prior = external_cluster.aggregate_down_consumer(
        source.feed_forward.w2.weight,
        axis=1,
    ).to(torch.float64)
    fit_rows = 20
    child_post = torch.zeros((fit_rows, 12), dtype=torch.float64)
    child_post[:, :3] = torch.randn(
        fit_rows, 3, generator=generator, dtype=torch.float64
    )
    correction = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    endpoint = child_post @ prior.T + child_post[:, :3] @ correction.T
    residual = torch.randn(fit_rows, 6, generator=generator, dtype=torch.float64)
    evidence = Lfm2ClosedLoopFoldEvidence(
        mlp=Lfm2MLPFoldEvidence(
            parent_gate_rows=gate,
            parent_up_rows=up,
            parent_post_swiglu_rows=post,
            parent_contribution_rows=contribution,
            child_post_swiglu_rows=child_post,
            parent_branch_output_rows=residual + endpoint,
            child_pre_branch_residual_rows=residual,
        )
    )
    config = Lfm2ClosedLoopFoldConfig(
        endpoint_ridge=0.0,
        max_primal_features=4,
        max_dual_rows=8,
        max_correction_anchors=3,
        max_endpoint_relative_rmse=1e-9,
        min_endpoint_cosine_similarity=0.999999999,
    )
    plan = build_lfm2_closed_loop_fold_plan(
        source,
        target,
        evidence,
        config=config,
    )

    assert plan.attention is None
    assert plan.short_conv_plan is not None
    assert plan.short_conv_plan.exactness.algebraically_exact is True
    assert plan.short_conv_preserved_exactly is True
    assert plan.mlp.endpoint_fit.solver == ("bounded-survivor-anchor-residual-ridge")
    assert plan.mlp.endpoint_fit.correction_rank_bound == 3
    assert plan.mlp.endpoint_fit.correction_anchor_indices == (0, 1, 2)

    transaction = install_lfm2_closed_loop_fold_plan(target, plan)
    for name, source_value in source.state_dict().items():
        if name.startswith("conv.") or name in {
            "operator_norm.weight",
            "ffn_norm.weight",
        }:
            assert torch.equal(
                target.state_dict()[name].detach().cpu(),
                source_value.detach().cpu(),
            ), name
    report = transaction.commit()
    assert report.short_conv_preserved_exactly is True
    assert report.gradient_state_preserved is True


def test_large_width_with_fewer_rows_uses_dual_not_feature_gram() -> None:
    source = _layer(
        301,
        kind="conv",
        hidden_size=5,
        intermediate_size=14,
    )
    target = _layer(
        302,
        kind="conv",
        hidden_size=5,
        intermediate_size=12,
    )
    generator = torch.Generator().manual_seed(303)
    gate = torch.randn(18, 14, generator=generator, dtype=torch.float64)
    up = torch.randn(18, 14, generator=generator, dtype=torch.float64)
    post = torch.nn.functional.silu(gate) * up
    child_post = torch.randn(8, 12, generator=generator, dtype=torch.float64)
    desired = torch.randn(5, 12, generator=generator, dtype=torch.float64)
    residual = torch.randn(8, 5, generator=generator, dtype=torch.float64)
    evidence = Lfm2ClosedLoopFoldEvidence(
        mlp=Lfm2MLPFoldEvidence(
            parent_gate_rows=gate,
            parent_up_rows=up,
            parent_post_swiglu_rows=post,
            child_post_swiglu_rows=child_post,
            parent_branch_output_rows=residual + child_post @ desired.T,
            child_pre_branch_residual_rows=residual,
        )
    )
    plan = build_lfm2_closed_loop_fold_plan(
        source,
        target,
        evidence,
        config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=0.0,
            max_primal_features=4,
            max_dual_rows=16,
            max_endpoint_relative_rmse=1e-8,
            min_endpoint_cosine_similarity=0.99999999,
        ),
    )

    assert plan.mlp.endpoint_fit.solver == "full-dual-residual-ridge"
    assert plan.mlp.endpoint_fit.correction_rank_bound == 8
    assert plan.mlp.endpoint_fit.correction_anchor_indices == ()


def test_seed_erasure_installs_identical_state_into_independent_blanks() -> None:
    source, planning_target, evidence, config, _output, _down = (
        _attention_transaction_fixture(400)
    )
    plan = build_lfm2_closed_loop_fold_plan(
        source,
        planning_target,
        evidence,
        config=config,
    )
    blank_a = _layer(
        411,
        kind="full_attention",
        intermediate_size=4,
        query_heads=14,
        key_value_heads=7,
        head_dim=2,
    )
    blank_b = _layer(
        947,
        kind="full_attention",
        intermediate_size=4,
        query_heads=14,
        key_value_heads=7,
        head_dim=2,
    )
    assert any(
        not torch.equal(blank_a.state_dict()[name], blank_b.state_dict()[name])
        for name in blank_a.state_dict()
    )

    report_a = install_lfm2_closed_loop_fold_plan(blank_a, plan).commit()
    report_b = install_lfm2_closed_loop_fold_plan(blank_b, plan).commit()

    _assert_state_equal(blank_a, blank_b)
    assert report_a.final_state_sha256 == plan.candidate_state_sha256
    assert report_b.final_state_sha256 == plan.candidate_state_sha256
    assert report_a.seed_erasure_by_construction is True
    assert report_b.seed_erasure_by_construction is True


def test_rejected_transaction_restores_exact_state_and_gradient_objects() -> None:
    source, target, evidence, config, _output, _down = _attention_transaction_fixture(
        500
    )
    plan = build_lfm2_closed_loop_fold_plan(
        source,
        target,
        evidence,
        config=config,
    )
    before = _state_clone(target)
    for parameter in target.parameters():
        parameter.grad = torch.randn_like(parameter)
    gradient_ids = tuple(id(parameter.grad) for parameter in target.parameters())
    gradient_values = tuple(
        parameter.grad.detach().clone() for parameter in target.parameters()
    )
    parameter_ids = tuple(id(parameter) for parameter in target.parameters())
    snapshot = snapshot_lfm2_layer(target)

    transaction = install_lfm2_closed_loop_fold_plan(target, plan)
    assert any(
        not torch.equal(target.state_dict()[name].detach().cpu(), before[name])
        for name in before
    )
    report = transaction.rollback()

    _assert_state_equal(before, target)
    assert report.accepted is False
    assert report.rollback_exact is True
    assert tuple(id(parameter) for parameter in target.parameters()) == parameter_ids
    assert (
        tuple(id(parameter.grad) for parameter in target.parameters()) == gradient_ids
    )
    for parameter, expected in zip(target.parameters(), gradient_values, strict=True):
        assert torch.equal(parameter.grad, expected)
    assert snapshot.state_sha256 == report.final_state_sha256
    assert report.as_dict()["backward_calls"] == 0


def test_topology_and_evidence_fail_closed_without_mutation() -> None:
    source, target, evidence, config, _output, _down = _attention_transaction_fixture(
        600
    )
    wrong_head_dim = _layer(
        604,
        kind="full_attention",
        intermediate_size=4,
        query_heads=14,
        key_value_heads=7,
        head_dim=4,
    )
    wrong_before = _state_clone(wrong_head_dim)
    with pytest.raises(Lfm2ClosedLoopTopologyError, match="head_dim"):
        build_lfm2_closed_loop_fold_plan(
            source,
            wrong_head_dim,
            evidence,
            config=config,
        )
    _assert_state_equal(wrong_before, wrong_head_dim)

    missing_attention = Lfm2ClosedLoopFoldEvidence(mlp=evidence.mlp)
    target_before = _state_clone(target)
    with pytest.raises(Lfm2ClosedLoopEvidenceError, match="required"):
        build_lfm2_closed_loop_fold_plan(
            source,
            target,
            missing_attention,
            config=config,
        )
    _assert_state_equal(target_before, target)

    conv_target = _layer(
        605,
        kind="conv",
        hidden_size=8,
        intermediate_size=4,
    )
    with pytest.raises(Lfm2ClosedLoopTopologyError, match="same operator kind"):
        build_lfm2_closed_loop_fold_plan(
            source,
            conv_target,
            Lfm2ClosedLoopFoldEvidence(mlp=evidence.mlp),
            config=config,
        )
