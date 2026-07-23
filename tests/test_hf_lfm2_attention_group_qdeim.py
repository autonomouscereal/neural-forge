from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_attention_group_qdeim import (  # noqa: E402
    LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC,
    Lfm2AttentionGroupQdeimError,
    Lfm2AttentionGroupQdeimInsufficientRankError,
    Lfm2AttentionGroupQdeimSnapshotBlock,
    compile_lfm2_attention_group_qdeim,
    lfm2_attention_group_qdeim_context_evidence_sha256,
    lfm2_attention_group_qdeim_snapshot_evidence_sha256,
    lfm2_attention_group_qdeim_source_state_sha256,
    lfm2_attention_group_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


def _contexts_from_group_features(
    features: torch.Tensor,
    *,
    rows: int,
    groups: int,
    queries_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    return (
        features.reshape(rows, queries_per_group, head_dim, groups)
        .permute(0, 3, 1, 2)
        .reshape(rows, groups * queries_per_group, head_dim)
        .contiguous()
    )


def _small_case(*, rank_deficient: bool = False) -> dict[str, object]:
    source_groups = 5
    target_groups = 3
    q_per_group = 2
    head_dim = 2
    source_q_heads = source_groups * q_per_group
    target_q_heads = target_groups * q_per_group
    source_hidden = 4
    target_hidden = 3
    rows = 4

    source_q = (
        torch.arange(source_q_heads * head_dim * source_hidden, dtype=torch.float64)
        .reshape(source_q_heads * head_dim, source_hidden)
        .div(8.0)
    )
    source_k = (
        torch.arange(source_groups * head_dim * source_hidden, dtype=torch.float64)
        .reshape(source_groups * head_dim, source_hidden)
        .add(3.0)
        .div(16.0)
    )
    source_v = source_k.mul(-0.5).add(1.0)
    source_o = (
        torch.arange(source_hidden * source_q_heads * head_dim, dtype=torch.float64)
        .reshape(source_hidden, source_q_heads * head_dim)
        .sub(7.0)
        .div(32.0)
    )
    decoder = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, -0.5],
        ],
        dtype=torch.float64,
    )
    chart = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.0, -0.5, 1.0],
        ],
        dtype=torch.float64,
    )

    if rank_deficient:
        common = torch.arange(rows * q_per_group * head_dim, dtype=torch.float64)
        group_features = common.unsqueeze(1).repeat(1, source_groups)
    else:
        latent = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, -1.0, 0.0],
                [0.0, 2.0, -1.0],
                [-1.0, 0.0, 2.0],
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
                [2.0, 3.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=torch.float64,
        )
        mixing = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, 1.0, 0.5, 1.0],
            ],
            dtype=torch.float64,
        )
        group_features = latent @ mixing
    contexts = _contexts_from_group_features(
        group_features,
        rows=rows,
        groups=source_groups,
        queries_per_group=q_per_group,
        head_dim=head_dim,
    )
    weights = (1.0, 2.0, 0.5, 3.0)
    templates = {
        "target_query_weight_template": torch.empty(
            (target_q_heads * head_dim, target_hidden), dtype=torch.bfloat16
        ),
        "target_key_weight_template": torch.empty(
            (target_groups * head_dim, target_hidden), dtype=torch.bfloat16
        ),
        "target_value_weight_template": torch.empty(
            (target_groups * head_dim, target_hidden), dtype=torch.bfloat16
        ),
        "target_out_weight_template": torch.empty(
            (target_hidden, target_q_heads * head_dim), dtype=torch.bfloat16
        ),
    }
    return {
        "source_query_weight": source_q,
        "source_key_weight": source_k,
        "source_value_weight": source_v,
        "source_out_weight": source_o,
        "expected_source_state_sha256": (
            lfm2_attention_group_qdeim_source_state_sha256(
                source_q, source_k, source_v, source_o
            )
        ),
        **templates,
        "expected_target_template_sha256": (
            lfm2_attention_group_qdeim_target_template_sha256(
                templates["target_query_weight_template"],
                templates["target_key_weight_template"],
                templates["target_value_weight_template"],
                templates["target_out_weight_template"],
            )
        ),
        "source_query_head_count": source_q_heads,
        "source_key_value_head_count": source_groups,
        "target_query_head_count": target_q_heads,
        "target_key_value_head_count": target_groups,
        "normalized_input_decoder": decoder,
        "expected_normalized_input_decoder_sha256": (
            lfm2_authenticated_tensor_sha256(decoder)
        ),
        "residual_output_chart": chart,
        "expected_residual_output_chart_sha256": (
            lfm2_authenticated_tensor_sha256(chart)
        ),
        "parent_post_softmax_context_rows": contexts,
        "row_sample_weights": weights,
        "expected_context_evidence_sha256": (
            lfm2_attention_group_qdeim_context_evidence_sha256(contexts, weights)
        ),
    }


def test_exact_block_low_rank_recovery_identity_and_o_orientation() -> None:
    arguments = _small_case()
    source_copies = {
        name: arguments[name].clone()
        for name in (
            "source_query_weight",
            "source_key_weight",
            "source_value_weight",
            "source_out_weight",
        )
    }
    result = compile_lfm2_attention_group_qdeim(**arguments)

    selected = result.selected_source_key_value_groups
    interpolation = result.group_interpolation
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert torch.allclose(
        interpolation[:, list(selected)],
        torch.eye(3, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    contexts = arguments["parent_post_softmax_context_rows"]
    grouped = contexts.reshape(4, 5, 2, 2).reshape(4, 5, 4)
    selected_flat = grouped[:, list(selected), :].reshape(4, 12)
    parent_flat = grouped.reshape(4, 20)
    reconstructed = selected_flat @ result.context_reconstruction
    assert torch.allclose(reconstructed, parent_flat, atol=2e-12, rtol=2e-12)

    chart = arguments["residual_output_chart"].to(torch.float64)
    source_o = arguments["source_out_weight"].to(torch.float64)
    expected_o = (chart.T @ source_o @ result.context_reconstruction.T).to(
        torch.bfloat16
    )
    assert torch.equal(result.out_weight, expected_o)
    expected_parent_target_output = parent_flat @ (chart.T @ source_o).T
    child_target_output = selected_flat @ result.out_weight.to(torch.float64).T
    assert torch.allclose(
        child_target_output,
        expected_parent_target_output,
        atol=0.35,
        rtol=0.01,
    )

    for name, before in source_copies.items():
        assert torch.equal(arguments[name], before)
    assert result.report["selection"]["qkv_producer_averaging_performed"] is False
    assert result.report["materialization"]["dense_all_parent_output_columns_used"]
    assert (
        result.report["semantic_policy"]["numeric_threshold_auto_veto_permitted"]
        is False
    )
    assert (
        result.report["semantic_policy"]["binary_manual_acceptance_gate_required"]
        is False
    )
    assert (
        result.report["semantic_policy"]["retention_targets_are_objectives_not_cutoffs"]
        is True
    )


def test_qkv_are_intact_selected_rows_through_decoder_not_averages() -> None:
    arguments = _small_case()
    result = compile_lfm2_attention_group_qdeim(**arguments)
    head_dim = 2
    q_per_group = 2
    selected_groups = result.selected_source_key_value_groups
    selected_q_heads = tuple(
        group * q_per_group + offset
        for group in selected_groups
        for offset in range(q_per_group)
    )
    selected_q_rows = [
        head * head_dim + coordinate
        for head in selected_q_heads
        for coordinate in range(head_dim)
    ]
    selected_kv_rows = [
        group * head_dim + coordinate
        for group in selected_groups
        for coordinate in range(head_dim)
    ]
    decoder = arguments["normalized_input_decoder"].to(torch.float64)
    expected_q = (
        arguments["source_query_weight"][selected_q_rows].to(torch.float64) @ decoder.T
    ).to(torch.bfloat16)
    expected_k = (
        arguments["source_key_weight"][selected_kv_rows].to(torch.float64) @ decoder.T
    ).to(torch.bfloat16)
    expected_v = (
        arguments["source_value_weight"][selected_kv_rows].to(torch.float64) @ decoder.T
    ).to(torch.bfloat16)
    assert result.selected_source_query_heads == selected_q_heads
    assert torch.equal(result.query_weight, expected_q)
    assert torch.equal(result.key_weight, expected_k)
    assert torch.equal(result.value_weight, expected_v)


def test_candidate_is_deterministic_dense_bf16_and_standalone() -> None:
    arguments = _small_case()
    first = compile_lfm2_attention_group_qdeim(**arguments)
    second = compile_lfm2_attention_group_qdeim(**arguments)
    assert first.selected_source_key_value_groups == (
        second.selected_source_key_value_groups
    )
    assert first.report["result_sha256"] == second.report["result_sha256"]
    assert (
        first.report["materialization"]["candidate_state_sha256"]
        == (second.report["materialization"]["candidate_state_sha256"])
    )
    for first_tensor, second_tensor in zip(
        (first.query_weight, first.key_weight, first.value_weight, first.out_weight),
        (
            second.query_weight,
            second.key_weight,
            second.value_weight,
            second.out_weight,
        ),
        strict=True,
    ):
        assert first_tensor.dtype == torch.bfloat16
        assert first_tensor.layout == torch.strided
        assert not first_tensor.requires_grad
        assert first_tensor.grad_fn is None
        assert torch.isfinite(first_tensor).all()
        assert torch.equal(first_tensor, second_tensor)
    state = first.state_dict()
    state["q_proj.weight"].zero_()
    assert torch.count_nonzero(first.query_weight).item() > 0


def test_compute_device_is_explicit_receipted_and_returns_cpu_candidate() -> None:
    arguments = _small_case()

    result = compile_lfm2_attention_group_qdeim(
        **arguments,
        compute_device="cpu",
    )

    execution = result.report["compute_execution"]
    execution_body = dict(execution)
    receipt_sha256 = execution_body.pop("receipt_sha256")
    assert execution["requested_device"] == "cpu"
    assert execution["resolved_device"] == "cpu"
    assert execution["backend"] == "cpu"
    assert execution["authenticated_source_archive_device"] == "cpu"
    assert execution["materialized_candidate_device"] == "cpu"
    assert execution["silent_device_fallback_performed"] is False
    assert receipt_sha256 == canonical_sha256(execution_body)
    assert all(value.device.type == "cpu" for value in result.state_dict().values())

    for invalid in ("cuda", "cuda:00", "cuda:\u0660"):
        with pytest.raises(
            Lfm2AttentionGroupQdeimError,
            match="explicitly indexed",
        ):
            compile_lfm2_attention_group_qdeim(
                **arguments,
                compute_device=invalid,
            )
    unavailable = f"cuda:{torch.cuda.device_count()}"
    with pytest.raises(Lfm2AttentionGroupQdeimError, match="unavailable"):
        compile_lfm2_attention_group_qdeim(
            **arguments,
            compute_device=unavailable,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_stages_fp64_qdeim_and_returns_standalone_cpu_bf16() -> None:
    result = compile_lfm2_attention_group_qdeim(
        **_small_case(),
        compute_device="cuda:0",
    )

    execution = result.report["compute_execution"]
    assert execution["requested_device"] == "cuda:0"
    assert execution["resolved_device"] == "cuda:0"
    assert execution["backend"] == "cuda"
    assert execution["active_layer_staged_to_compute_device"] is True
    assert execution["gpu_is_ephemeral_dense_algebra_workspace"] is True
    assert execution["compute_dtype"] == "float64"
    assert all(
        value.device.type == "cpu" and value.dtype == torch.bfloat16
        for value in result.state_dict().values()
    )


def test_authenticated_rank_deficit_has_no_silent_completion() -> None:
    arguments = _small_case(rank_deficient=True)
    with pytest.raises(Lfm2AttentionGroupQdeimInsufficientRankError) as captured:
        compile_lfm2_attention_group_qdeim(**arguments)
    report = captured.value.report
    assert report["status"] == "insufficient-authenticated-context-group-rank"
    assert report["observed_context_group_rank"] == 1
    assert report["minimum_context_group_rank"] == 3
    assert report["selected_source_key_value_groups"] == []
    assert report["random_rank_completion_permitted"] is False
    assert report["heldout_rank_completion_permitted"] is False
    assert report["silent_pseudoinverse_completion_permitted"] is False
    assert report["source_state_immutable"] is True
    assert report["compute_execution"]["requested_device"] == "cpu"
    assert report["compute_execution"]["resolved_device"] == "cpu"
    assert len(report["result_sha256"]) == 64


def test_authentication_fails_closed_and_snapshot_blocks_are_separate() -> None:
    arguments = _small_case()
    wrong_source = dict(arguments)
    wrong_source["expected_source_state_sha256"] = "0" * 64
    with pytest.raises(Lfm2AttentionGroupQdeimError, match="Source Q/K/V/O"):
        compile_lfm2_attention_group_qdeim(**wrong_source)

    snapshot_context = arguments["parent_post_softmax_context_rows"].mul(0.25)
    block = Lfm2AttentionGroupQdeimSnapshotBlock(
        block_id="analytic-tangent-0",
        snapshot_kind="analytic_child_lifted_parent_attention_tangent",
        parent_post_softmax_context_rows=snapshot_context,
        row_sample_weights=(1.0, 1.0, 2.0, 2.0),
        generator_receipt_sha256="a" * 64,
        block_weight=0.5,
    )
    with pytest.raises(Lfm2AttentionGroupQdeimError, match="expected evidence"):
        compile_lfm2_attention_group_qdeim(
            **arguments,
            functional_snapshot_blocks=(block,),
        )
    snapshot_digest = lfm2_attention_group_qdeim_snapshot_evidence_sha256((block,))
    with pytest.raises(Lfm2AttentionGroupQdeimError, match="authentication failed"):
        compile_lfm2_attention_group_qdeim(
            **arguments,
            functional_snapshot_blocks=(block,),
            expected_functional_snapshot_evidence_sha256="b" * 64,
        )
    result = compile_lfm2_attention_group_qdeim(
        **arguments,
        functional_snapshot_blocks=(block,),
        expected_functional_snapshot_evidence_sha256=snapshot_digest,
    )
    evidence = result.report["evidence"]
    assert evidence["structural_tangent_snapshot_block_count"] == 1
    assert evidence["heldout_rows_used"] is False
    assert evidence["snapshot_blocks"][0]["generator_receipt_sha256"] == "a" * 64


def test_residual_output_metric_covariance_factors_and_energy_balances_before_pod() -> (
    None
):
    arguments = _small_case()
    contexts = arguments["parent_post_softmax_context_rows"]
    blocks = (
        Lfm2AttentionGroupQdeimSnapshotBlock(
            block_id="joint-sampled-token-qkv",
            snapshot_kind="analytic_child_lifted_parent_attention_tangent",
            parent_post_softmax_context_rows=contexts.mul(0.25),
            row_sample_weights=(1.0, 1.0, 2.0, 2.0),
            generator_receipt_sha256="a" * 64,
        ),
        Lfm2AttentionGroupQdeimSnapshotBlock(
            block_id="older-prefix-kv",
            snapshot_kind="analytic_child_lifted_parent_attention_tangent",
            parent_post_softmax_context_rows=torch.roll(contexts, shifts=1, dims=1).mul(
                0.125
            ),
            row_sample_weights=(2.0, 1.0, 1.0, 2.0),
            generator_receipt_sha256="b" * 64,
        ),
    )
    result = compile_lfm2_attention_group_qdeim(
        **arguments,
        functional_snapshot_blocks=blocks,
        expected_functional_snapshot_evidence_sha256=(
            lfm2_attention_group_qdeim_snapshot_evidence_sha256(blocks)
        ),
        functional_pod_group_metric=(
            LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC
        ),
    )

    evidence = result.report["evidence"]
    receipt = evidence["functional_pod_receipt"]
    empirical = receipt["empirical_covariance_factor"]
    tangent = receipt["causal_tangent_balanced_covariance_factor"]
    assert evidence["raw_fit_context_used_directly_for_group_selection"] is False
    assert receipt["empirical_post_softmax_context_mapped_through_source_o"]
    assert receipt[
        "empirical_post_softmax_context_mapped_through_residual_output_chart"
    ]
    assert receipt["empirical_to_aggregate_tangent_energy_balance_applied"]
    assert empirical["covariance_equality_proved"]
    assert tangent["covariance_equality_proved"]
    assert tangent["covariance_trace"] == pytest.approx(
        empirical["covariance_trace"],
        rel=2e-12,
        abs=2e-12,
    )
    assert receipt["covariance_factoring_changes_group_pod_gram_matrix"] is False
    assert (
        result.report["authentication"]["functional_pod_receipt_sha256"]
        == (receipt["result_sha256"])
    )


def test_residual_output_metric_requires_both_authenticated_causal_families() -> None:
    arguments = _small_case()
    contexts = arguments["parent_post_softmax_context_rows"]
    block = Lfm2AttentionGroupQdeimSnapshotBlock(
        block_id="joint-sampled-token-qkv",
        snapshot_kind="analytic_child_lifted_parent_attention_tangent",
        parent_post_softmax_context_rows=contexts,
        row_sample_weights=(1.0, 1.0, 1.0, 1.0),
        generator_receipt_sha256="a" * 64,
    )
    with pytest.raises(
        Lfm2AttentionGroupQdeimError,
        match="exactly the separately authenticated",
    ):
        compile_lfm2_attention_group_qdeim(
            **arguments,
            functional_snapshot_blocks=(block,),
            expected_functional_snapshot_evidence_sha256=(
                lfm2_attention_group_qdeim_snapshot_evidence_sha256((block,))
            ),
            functional_pod_group_metric=(
                LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC
            ),
        )


def test_real_lfm2_16_8_to_10_5_head_shapes_are_derived_not_hardcoded() -> None:
    torch.manual_seed(29)
    source_q_heads = 16
    source_kv_heads = 8
    target_q_heads = 10
    target_kv_heads = 5
    head_dim = 64
    source_hidden = 7
    target_hidden = 5
    source_q = torch.randn(source_q_heads * head_dim, source_hidden)
    source_k = torch.randn(source_kv_heads * head_dim, source_hidden)
    source_v = torch.randn(source_kv_heads * head_dim, source_hidden)
    source_o = torch.randn(source_hidden, source_q_heads * head_dim)
    decoder = torch.randn(target_hidden, source_hidden)
    chart = torch.randn(source_hidden, target_hidden)
    contexts = torch.randn(2, source_q_heads, head_dim)
    weights = (1.0, 2.0)
    templates = (
        torch.empty(target_q_heads * head_dim, target_hidden, dtype=torch.bfloat16),
        torch.empty(target_kv_heads * head_dim, target_hidden, dtype=torch.bfloat16),
        torch.empty(target_kv_heads * head_dim, target_hidden, dtype=torch.bfloat16),
        torch.empty(target_hidden, target_q_heads * head_dim, dtype=torch.bfloat16),
    )
    result = compile_lfm2_attention_group_qdeim(
        source_query_weight=source_q,
        source_key_weight=source_k,
        source_value_weight=source_v,
        source_out_weight=source_o,
        expected_source_state_sha256=(
            lfm2_attention_group_qdeim_source_state_sha256(
                source_q, source_k, source_v, source_o
            )
        ),
        target_query_weight_template=templates[0],
        target_key_weight_template=templates[1],
        target_value_weight_template=templates[2],
        target_out_weight_template=templates[3],
        expected_target_template_sha256=(
            lfm2_attention_group_qdeim_target_template_sha256(*templates)
        ),
        source_query_head_count=source_q_heads,
        source_key_value_head_count=source_kv_heads,
        target_query_head_count=target_q_heads,
        target_key_value_head_count=target_kv_heads,
        normalized_input_decoder=decoder,
        expected_normalized_input_decoder_sha256=(
            lfm2_authenticated_tensor_sha256(decoder)
        ),
        residual_output_chart=chart,
        expected_residual_output_chart_sha256=(lfm2_authenticated_tensor_sha256(chart)),
        parent_post_softmax_context_rows=contexts,
        row_sample_weights=weights,
        expected_context_evidence_sha256=(
            lfm2_attention_group_qdeim_context_evidence_sha256(contexts, weights)
        ),
    )
    assert result.query_weight.shape == (10 * 64, target_hidden)
    assert result.key_weight.shape == (5 * 64, target_hidden)
    assert result.value_weight.shape == (5 * 64, target_hidden)
    assert result.out_weight.shape == (target_hidden, 10 * 64)
    assert result.report["topology"]["head_dim"] == 64
    assert result.report["topology"]["queries_per_key_value"] == 2
