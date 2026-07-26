from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_attention_group_qdeim import (  # noqa: E402
    LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE,
    LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE,
    LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC,
    Lfm2AttentionGroupQdeimError,
    Lfm2AttentionGroupQdeimInsufficientRankError,
    Lfm2AttentionGroupQdeimSnapshotBlock,
    Lfm2AttentionGroupQdeimSourceLocalOutputFitCase,
    compile_lfm2_attention_group_qdeim,
    compile_lfm2_attention_group_qdeim_source_local_output_residual,
    lfm2_attention_group_qdeim_candidate_state_sha256,
    lfm2_attention_group_qdeim_context_evidence_sha256,
    lfm2_attention_group_qdeim_snapshot_evidence_sha256,
    lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256,
    lfm2_attention_group_qdeim_source_state_sha256,
    lfm2_attention_group_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)
import neural_forge.hf_lfm2_attention_group_qdeim as qdeim_module  # noqa: E402, I001


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


def _small_case(
    *,
    rank_deficient: bool = False,
    target_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, object]:
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
            (target_q_heads * head_dim, target_hidden), dtype=target_dtype
        ),
        "target_key_weight_template": torch.empty(
            (target_groups * head_dim, target_hidden), dtype=target_dtype
        ),
        "target_value_weight_template": torch.empty(
            (target_groups * head_dim, target_hidden), dtype=target_dtype
        ),
        "target_out_weight_template": torch.empty(
            (target_hidden, target_q_heads * head_dim), dtype=target_dtype
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


def _source_local_output_fit_case(
    structural_candidate: object,
    *,
    case_id: str = "fit-0",
    context_rows: torch.Tensor | None = None,
    action_scale: float = 1.0,
) -> Lfm2AttentionGroupQdeimSourceLocalOutputFitCase:
    rows = 6
    source_hidden, target_hidden = structural_candidate.residual_output_chart.shape
    context_width = structural_candidate.out_weight.shape[1]
    generator = torch.Generator().manual_seed(20260726)
    parent_pre = torch.randn(
        rows,
        source_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    parent_action = (
        torch.randn(
            rows,
            source_hidden,
            generator=generator,
            dtype=torch.float64,
        )
        * action_scale
    )
    child_pre = torch.randn(
        rows,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    if context_rows is None:
        context_rows = torch.randn(
            rows,
            context_width,
            generator=generator,
            dtype=torch.float64,
        )
    parent_post = parent_pre + parent_action
    return Lfm2AttentionGroupQdeimSourceLocalOutputFitCase(
        case_id=case_id,
        parent_pre_operator_residual=parent_pre,
        parent_post_operator_residual=parent_post,
        live_child_pre_operator_residual=child_pre,
        live_child_pre_o_context=context_rows,
        row_sample_weights=(1.0, 0.5, 2.0, 1.5, 0.75, 3.0),
        parent_trajectory_receipt_sha256=canonical_sha256(
            {"kind": "parent-trajectory", "case_id": case_id}
        ),
        live_child_replay_receipt_sha256=canonical_sha256(
            {"kind": "live-child-replay", "case_id": case_id}
        ),
        aligned_row_identity_sha256=canonical_sha256(
            {"kind": "aligned-row-identity", "case_id": case_id}
        ),
    )


def _source_local_output_arguments(
    structural_candidate: object,
    fit_cases: tuple[Lfm2AttentionGroupQdeimSourceLocalOutputFitCase, ...],
    *,
    ridge: float = 1.0e-6,
) -> dict[str, object]:
    return {
        "structural_candidate": structural_candidate,
        "expected_structural_candidate_state_sha256": (
            lfm2_attention_group_qdeim_candidate_state_sha256(
                structural_candidate.query_weight,
                structural_candidate.key_weight,
                structural_candidate.value_weight,
                structural_candidate.out_weight,
            )
        ),
        "expected_structural_receipt_sha256": structural_candidate.report[
            "result_sha256"
        ],
        "expected_residual_output_chart_sha256": (
            lfm2_authenticated_tensor_sha256(structural_candidate.residual_output_chart)
        ),
        "fit_cases": fit_cases,
        "expected_fit_evidence_sha256": (
            lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256(
                fit_cases
            )
        ),
        "ridge": ridge,
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


def test_float32_templates_publish_direct_float32_candidate() -> None:
    arguments = _small_case(target_dtype=torch.float32)
    result = compile_lfm2_attention_group_qdeim(**arguments)

    decoder = arguments["normalized_input_decoder"].to(torch.float64)
    selected_groups = result.selected_source_key_value_groups
    selected_query_heads = result.selected_source_query_heads
    selected_query_rows = [
        head * 2 + coordinate
        for head in selected_query_heads
        for coordinate in range(2)
    ]
    selected_kv_rows = [
        group * 2 + coordinate for group in selected_groups for coordinate in range(2)
    ]
    expected_query = (
        arguments["source_query_weight"][selected_query_rows].to(torch.float64)
        @ decoder.T
    ).to(torch.float32)
    expected_key = (
        arguments["source_key_weight"][selected_kv_rows].to(torch.float64) @ decoder.T
    ).to(torch.float32)

    assert result.report["materialization"]["candidate_dtype"] == "float32"
    assert all(
        value.dtype == torch.float32 and value.device.type == "cpu"
        for value in result.state_dict().values()
    )
    assert torch.equal(result.query_weight, expected_query)
    assert torch.equal(result.key_weight, expected_key)


def test_attention_templates_reject_mixed_target_dtypes() -> None:
    arguments = _small_case(target_dtype=torch.float32)
    arguments["target_key_weight_template"] = arguments[
        "target_key_weight_template"
    ].to(torch.bfloat16)
    arguments["expected_target_template_sha256"] = "0" * 64

    with pytest.raises(
        Lfm2AttentionGroupQdeimError,
        match="must share one supported dtype",
    ):
        compile_lfm2_attention_group_qdeim(**arguments)


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
    legacy_snapshot_digest = canonical_sha256(
        {
            "schema": (
                "neural-forge-lfm2-attention-group-qdeim-functional-snapshots-v1"
            ),
            "heldout_rows_permitted": False,
            "blocks": [
                {
                    "block_id": "analytic-tangent-0",
                    "snapshot_kind": ("analytic_child_lifted_parent_attention_tangent"),
                    "context_rows_sha256": lfm2_authenticated_tensor_sha256(
                        snapshot_context
                    ),
                    "row_sample_weights": [1.0, 1.0, 2.0, 2.0],
                    "block_weight": 0.5,
                    "generator_receipt_sha256": "a" * 64,
                }
            ],
        }
    )
    assert snapshot_digest == legacy_snapshot_digest
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


@pytest.mark.parametrize(
    ("factor", "required_positive_diagonal_count"),
    (
        (torch.diag(torch.tensor([1.0, 1e-13**0.5], dtype=torch.float64)), 2),
        (torch.zeros((3, 2), dtype=torch.float64), 0),
        (
            torch.diag(torch.tensor([1.0, 1e-10, 1e-20], dtype=torch.float64)),
            3,
        ),
    ),
    ids=("diag-one-and-one-e-minus-thirteen", "zero", "ill-conditioned"),
)
def test_direct_feature_gram_never_eigentruncates_tiny_or_zero_modes(
    factor: torch.Tensor,
    required_positive_diagonal_count: int,
) -> None:
    gram, receipt = qdeim_module._direct_feature_gram_receipt(
        factor,
        factor_id="test-direct-factor",
    )

    expected = factor.T @ factor
    assert torch.equal(gram, expected)
    assert int((torch.diag(gram) > 0.0).sum().item()) == (
        required_positive_diagonal_count
    )
    assert receipt["factor_sha256"] == lfm2_authenticated_tensor_sha256(factor)
    assert receipt["gram_sha256"] == lfm2_authenticated_tensor_sha256(expected)
    assert receipt["covariance_eigendecomposition_performed"] is False
    assert receipt["covariance_refactor_performed"] is False
    assert receipt["positive_finite_feature_modes_removed"] == 0


def test_residual_output_metric_preserves_direct_features_and_energy_balances_before_pod() -> (
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
            perturbation_role=(LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE),
            perturbation_receipt_sha256="1" * 64,
        ),
        Lfm2AttentionGroupQdeimSnapshotBlock(
            block_id="older-prefix-kv",
            snapshot_kind="analytic_child_lifted_parent_attention_tangent",
            parent_post_softmax_context_rows=torch.roll(contexts, shifts=1, dims=1).mul(
                0.125
            ),
            row_sample_weights=(2.0, 1.0, 1.0, 2.0),
            generator_receipt_sha256="a" * 64,
            perturbation_role=(
                LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE
            ),
            perturbation_receipt_sha256="2" * 64,
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
    empirical = receipt["empirical_direct_feature_gram"]
    tangent = receipt["causal_tangent_balanced_direct_feature_gram"]
    assert evidence["raw_fit_context_used_directly_for_group_selection"] is False
    assert receipt["empirical_post_softmax_context_mapped_through_source_o"]
    assert receipt[
        "empirical_post_softmax_context_mapped_through_residual_output_chart"
    ]
    assert receipt["empirical_to_aggregate_tangent_energy_balance_applied"]
    assert empirical["covariance_eigendecomposition_performed"] is False
    assert tangent["covariance_refactor_performed"] is False
    assert empirical["positive_finite_feature_modes_removed"] == 0
    assert tangent["gram_trace"] == pytest.approx(
        empirical["gram_trace"],
        rel=2e-12,
        abs=2e-12,
    )
    assert receipt["combined_gram_within_fp64_roundoff"]
    assert receipt["covariance_eigendecomposition_or_refactor_performed"] is False
    assert receipt["positive_finite_feature_modes_removed"] == 0
    assert {
        value["perturbation_role"] for value in receipt["causal_tangent_blocks"]
    } == {
        LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE,
        LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE,
    }
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
        perturbation_role=(LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE),
        perturbation_receipt_sha256="1" * 64,
    )
    with pytest.raises(
        Lfm2AttentionGroupQdeimError,
        match="exact distinct typed",
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

    duplicate = Lfm2AttentionGroupQdeimSnapshotBlock(
        block_id="different-free-form-id",
        snapshot_kind="analytic_child_lifted_parent_attention_tangent",
        parent_post_softmax_context_rows=contexts.mul(0.5),
        row_sample_weights=(1.0, 1.0, 1.0, 1.0),
        generator_receipt_sha256="a" * 64,
        perturbation_role=(LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE),
        perturbation_receipt_sha256="2" * 64,
    )
    duplicate_blocks = (block, duplicate)
    with pytest.raises(Lfm2AttentionGroupQdeimError, match="exact distinct typed"):
        compile_lfm2_attention_group_qdeim(
            **arguments,
            functional_snapshot_blocks=duplicate_blocks,
            expected_functional_snapshot_evidence_sha256=(
                lfm2_attention_group_qdeim_snapshot_evidence_sha256(duplicate_blocks)
            ),
            functional_pod_group_metric=(
                LFM2_ATTENTION_GROUP_QDEIM_RESIDUAL_OUTPUT_BALANCED_METRIC
            ),
        )


def test_snapshot_evidence_binds_typed_role_to_its_reduction_receipt() -> None:
    contexts = _small_case()["parent_post_softmax_context_rows"]

    def blocks(joint_receipt: str, prefix_receipt: str):
        return (
            Lfm2AttentionGroupQdeimSnapshotBlock(
                block_id="opaque-a",
                snapshot_kind="analytic_child_lifted_parent_attention_tangent",
                parent_post_softmax_context_rows=contexts,
                row_sample_weights=(1.0, 1.0, 1.0, 1.0),
                generator_receipt_sha256="a" * 64,
                perturbation_role=(
                    LFM2_ATTENTION_GROUP_QDEIM_JOINT_QKV_PERTURBATION_ROLE
                ),
                perturbation_receipt_sha256=joint_receipt,
            ),
            Lfm2AttentionGroupQdeimSnapshotBlock(
                block_id="opaque-b",
                snapshot_kind="analytic_child_lifted_parent_attention_tangent",
                parent_post_softmax_context_rows=contexts.mul(0.5),
                row_sample_weights=(1.0, 1.0, 1.0, 1.0),
                generator_receipt_sha256="a" * 64,
                perturbation_role=(
                    LFM2_ATTENTION_GROUP_QDEIM_OLDER_PREFIX_KV_PERTURBATION_ROLE
                ),
                perturbation_receipt_sha256=prefix_receipt,
            ),
        )

    bound = blocks("1" * 64, "2" * 64)
    swapped = blocks("2" * 64, "1" * 64)
    assert lfm2_attention_group_qdeim_snapshot_evidence_sha256(bound) != (
        lfm2_attention_group_qdeim_snapshot_evidence_sha256(swapped)
    )

    invalid = Lfm2AttentionGroupQdeimSnapshotBlock(
        block_id="opaque-invalid",
        snapshot_kind="analytic_child_lifted_parent_attention_tangent",
        parent_post_softmax_context_rows=contexts,
        row_sample_weights=(1.0, 1.0, 1.0, 1.0),
        generator_receipt_sha256="a" * 64,
        perturbation_role="free_form_id_is_not_a_role",
        perturbation_receipt_sha256="1" * 64,
    )
    with pytest.raises(
        Lfm2AttentionGroupQdeimError, match="typed causal-attention role"
    ):
        lfm2_attention_group_qdeim_snapshot_evidence_sha256((invalid,))


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


def test_source_local_o_is_invariant_to_parent_offset_and_child_pre() -> None:
    structural_arguments = _small_case(target_dtype=torch.float32)
    structural = compile_lfm2_attention_group_qdeim(**structural_arguments)
    generator = torch.Generator().manual_seed(97)
    context_rows = torch.randn(
        6,
        structural.out_weight.shape[1],
        generator=generator,
        dtype=torch.float64,
    )
    base_case = _source_local_output_fit_case(
        structural,
        context_rows=context_rows,
    )
    parent_offset_case = replace(
        base_case,
        parent_pre_operator_residual=(base_case.parent_pre_operator_residual + 8.0),
        parent_post_operator_residual=(base_case.parent_post_operator_residual + 8.0),
    )
    child_pre_case = replace(
        base_case,
        live_child_pre_operator_residual=(
            base_case.live_child_pre_operator_residual
            + torch.arange(18, dtype=torch.float64).reshape(6, 3) * 1.0e4
        ),
    )

    base = compile_lfm2_attention_group_qdeim_source_local_output_residual(
        **_source_local_output_arguments(structural, (base_case,))
    )
    parent_offset = compile_lfm2_attention_group_qdeim_source_local_output_residual(
        **_source_local_output_arguments(structural, (parent_offset_case,))
    )
    child_pre = compile_lfm2_attention_group_qdeim_source_local_output_residual(
        **_source_local_output_arguments(structural, (child_pre_case,))
    )

    for name, base_tensor in base.state_dict().items():
        assert torch.equal(parent_offset.state_dict()[name], base_tensor)
        assert torch.equal(child_pre.state_dict()[name], base_tensor)
    assert torch.allclose(
        parent_offset.analytic_out_correction_fp64,
        base.analytic_out_correction_fp64,
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    assert torch.equal(
        child_pre.analytic_out_correction_fp64,
        base.analytic_out_correction_fp64,
    )
    assert (
        base.report["target"]["live_child_pre_operator_residual_used_for_correction"]
        is False
    )
    assert (
        base.report["target"][
            "absolute_or_cumulative_endpoint_target_used_for_correction"
        ]
        is False
    )
    assert (
        base.report["target"]["common_parent_pre_post_offset_changes_local_target"]
        is False
    )
    assert (
        base.report["target"][
            "arbitrary_child_pre_residual_perturbation_changes_candidate"
        ]
        is False
    )
    assert (
        base.report["authentication"]["fit_evidence_sha256_before"]
        != parent_offset.report["authentication"]["fit_evidence_sha256_before"]
    )
    assert (
        base.report["authentication"]["fit_evidence_sha256_before"]
        != child_pre.report["authentication"]["fit_evidence_sha256_before"]
    )
    assert (
        base.report["diagnostics"]["corrected_endpoint_relative_rmse_advisory"]
        != child_pre.report["diagnostics"]["corrected_endpoint_relative_rmse_advisory"]
    )


@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
def test_source_local_o_truncates_dtype_rank_bounds_delta_and_preserves_nullspace(
    target_dtype: torch.dtype,
) -> None:
    structural = compile_lfm2_attention_group_qdeim(
        **_small_case(target_dtype=target_dtype)
    )
    context_rows = torch.zeros(
        6,
        structural.out_weight.shape[1],
        dtype=torch.float64,
    )
    context_rows[0, 0] = 2.0
    context_rows[1, 1] = 1.0
    context_rows[2, 2] = 1.0e-2
    context_rows[3, 3] = 1.0e-5
    fit_case = _source_local_output_fit_case(
        structural,
        context_rows=context_rows,
        action_scale=1.0e6,
    )
    structural_state_before = structural.state_dict()
    result = compile_lfm2_attention_group_qdeim_source_local_output_residual(
        **_source_local_output_arguments(structural, (fit_case,), ridge=0.0)
    )
    solver = result.report["solver"]

    expected_dtype_rank = 3 if target_dtype == torch.float32 else 2
    assert solver["fit_design_machine_rank"] == 4
    assert solver["fit_design_rank"] == expected_dtype_rank
    assert solver["fit_design_dtype_truncated_rank"] == 4 - expected_dtype_rank
    assert solver["ridge_expansion_steps"] > 0
    assert solver["ridge_bisection_steps"] == 64
    assert solver["analytic_correction_delta_frobenius_norm"] <= (
        solver["source_relative_trust_radius"]
        + solver["analytic_unsupported_component_tolerance"]
    )

    delta_t = result.analytic_out_correction_fp64
    supported_basis = result.fit_supported_context_basis_fp64
    unsupported = delta_t - supported_basis @ (supported_basis.T @ delta_t)
    assert (
        torch.linalg.vector_norm(unsupported).item()
        <= solver["analytic_unsupported_component_tolerance"]
    )
    assert torch.count_nonzero(delta_t[expected_dtype_rank:]).item() == 0

    published_delta_t = (
        result.out_weight_source_local_corrected.to(torch.float64)
        - result.out_weight_structural_prior.to(torch.float64)
    ).T
    published_unsupported = published_delta_t - supported_basis @ (
        supported_basis.T @ published_delta_t
    )
    assert torch.linalg.vector_norm(published_unsupported).item() == pytest.approx(
        solver["target_dtype_published_unsupported_leakage_frobenius_norm"],
        abs=1.0e-15,
    )
    assert solver["unsupported_prior_directions_preserved_analytically"] is True
    assert solver["whole_prior_plus_delta_scaled"] is False
    assert solver["delta_only_trust_bound"] is True
    assert (
        result.report["materialization"][
            "prior_plus_delta_whole_writer_scaling_performed"
        ]
        is False
    )
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
        assert torch.equal(result.state_dict()[name], structural_state_before[name])
        assert torch.equal(structural.state_dict()[name], structural_state_before[name])
    assert torch.equal(
        structural.state_dict()["out_proj.weight"],
        structural_state_before["out_proj.weight"],
    )
    authentication = result.report["authentication"]
    for key in (
        "structural_candidate_state_immutable",
        "structural_receipt_immutable",
        "stored_residual_output_chart_immutable",
        "fit_evidence_immutable",
    ):
        assert authentication[key] is True
    assert (
        authentication["structural_candidate_state_sha256_before"]
        == authentication["structural_candidate_state_sha256_after"]
    )
    assert (
        authentication["structural_receipt_sha256_before"]
        == authentication["structural_receipt_sha256_after"]
    )
    assert (
        authentication["stored_residual_output_chart_sha256_before"]
        == authentication["stored_residual_output_chart_sha256_after"]
    )
    assert (
        authentication["fit_evidence_sha256_before"]
        == authentication["fit_evidence_sha256_after"]
    )


def test_source_local_o_authentication_fails_closed_and_cannot_repeat() -> None:
    structural = compile_lfm2_attention_group_qdeim(
        **_small_case(target_dtype=torch.float32)
    )
    fit_case = _source_local_output_fit_case(structural)
    arguments = _source_local_output_arguments(structural, (fit_case,))

    wrong_fit = dict(arguments)
    wrong_fit["expected_fit_evidence_sha256"] = "0" * 64
    with pytest.raises(
        Lfm2AttentionGroupQdeimError,
        match="FIT evidence authentication failed",
    ):
        compile_lfm2_attention_group_qdeim_source_local_output_residual(**wrong_fit)

    tampered_structural = replace(
        structural,
        out_weight=structural.out_weight.add(1.0),
    )
    with pytest.raises(
        Lfm2AttentionGroupQdeimError,
        match="do not match their sealed receipt",
    ):
        compile_lfm2_attention_group_qdeim_source_local_output_residual(
            **_source_local_output_arguments(tampered_structural, (fit_case,))
        )

    result = compile_lfm2_attention_group_qdeim_source_local_output_residual(
        **arguments
    )
    report = result.as_dict()
    receipt_sha256 = report.pop("result_sha256")
    assert canonical_sha256(report) == receipt_sha256
    assert result.report["authentication"][
        "source_local_corrected_candidate_state_sha256"
    ] == lfm2_attention_group_qdeim_candidate_state_sha256(
        result.query_weight,
        result.key_weight,
        result.value_weight,
        result.out_weight_source_local_corrected,
    )
    prohibited = result.report["prohibited_operations"]
    assert prohibited["training_steps"] == 0
    assert prohibited["backward_calls"] == 0
    assert prohibited["optimizer_steps"] == 0
    assert prohibited["numeric_parent_child_weight_equality_required"] is False
    repeated = dict(arguments)
    repeated["structural_candidate"] = result
    with pytest.raises(
        Lfm2AttentionGroupQdeimError,
        match="pristine compiled intact-GQA",
    ):
        compile_lfm2_attention_group_qdeim_source_local_output_residual(**repeated)

    duplicate_case = replace(fit_case)
    with pytest.raises(
        Lfm2AttentionGroupQdeimError,
        match="case IDs must be unique",
    ):
        lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256(
            (fit_case, duplicate_case)
        )
