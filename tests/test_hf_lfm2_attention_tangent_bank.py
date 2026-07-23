from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_attention_tangent_bank import (  # noqa: E402
    Lfm2AttentionTangentAuthenticationError,
    Lfm2AttentionTangentFitCase,
    build_lfm2_attention_tangent_bank,
    lfm2_attention_causal_jvp,
    lfm2_attention_tangent_case_evidence_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)
import neural_forge.hf_lfm2_attention_tangent_bank as tangent_module  # noqa: E402, I001


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    midpoint = int(value.shape[-1]) // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def _rope(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return value * cos + _rotate_half(value) * sin


def _norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return (
        value
        * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
        * weight
    )


def _fixture() -> tuple[dict[str, object], Lfm2AttentionTangentFitCase]:
    dtype = torch.float64
    epsilon = 1e-5
    query_heads = 4
    key_value_heads = 2
    head_dim = 2
    source_q = torch.tensor(
        [
            [0.7, -0.2, 0.1],
            [0.3, 0.5, -0.4],
            [-0.6, 0.2, 0.8],
            [0.1, -0.9, 0.4],
            [0.5, 0.4, -0.3],
            [-0.2, 0.6, 0.9],
            [0.8, -0.7, 0.2],
            [0.4, 0.3, 0.6],
        ],
        dtype=dtype,
    )
    source_k = torch.tensor(
        [
            [0.6, 0.1, -0.4],
            [-0.3, 0.8, 0.2],
            [0.5, -0.6, 0.7],
            [0.2, 0.4, -0.9],
        ],
        dtype=dtype,
    )
    source_v = torch.tensor(
        [
            [0.4, -0.5, 0.3],
            [0.7, 0.2, -0.1],
            [-0.2, 0.9, 0.5],
            [0.6, -0.3, 0.8],
        ],
        dtype=dtype,
    )
    source_o = torch.tensor(
        [
            [0.3, -0.2, 0.4, 0.1, -0.5, 0.6, 0.2, -0.1],
            [-0.4, 0.7, 0.2, -0.3, 0.1, 0.5, -0.6, 0.8],
            [0.6, 0.1, -0.5, 0.4, 0.3, -0.2, 0.7, 0.2],
        ],
        dtype=dtype,
    )
    residual_chart = torch.tensor(
        [[1.0, -0.2], [0.3, 0.8], [-0.4, 0.5]],
        dtype=dtype,
    )
    decoder = torch.tensor(
        [[1.0, -0.25, 0.5], [0.2, 0.9, -0.4]],
        dtype=dtype,
    )
    query_norm = torch.tensor([1.125, 0.875], dtype=dtype)
    key_norm = torch.tensor([0.75, 1.25], dtype=dtype)
    q_linear = torch.tensor(
        [
            [
                [[0.8, -0.4], [0.3, 0.9]],
                [[-0.5, 0.7], [1.1, -0.2]],
                [[0.6, 0.2], [-0.8, 0.5]],
                [[-0.3, -0.9], [0.4, 0.7]],
            ]
        ],
        dtype=dtype,
    )
    k_linear = torch.tensor(
        [
            [
                [[0.7, -0.6], [0.2, 0.8]],
                [[-0.4, 0.9], [1.0, 0.3]],
            ]
        ],
        dtype=dtype,
    )
    q_pre = _norm(q_linear, query_norm, epsilon)
    k_pre = _norm(k_linear, key_norm, epsilon)
    angle = torch.tensor([[[0.2, 0.2], [0.65, 0.65]]], dtype=dtype)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    query = _rope(
        q_pre.permute(0, 2, 1, 3),
        cos.unsqueeze(2),
        sin.unsqueeze(2),
    ).permute(0, 2, 1, 3)
    key_suffix = _rope(
        k_pre.permute(0, 2, 1, 3),
        cos.unsqueeze(2),
        sin.unsqueeze(2),
    ).permute(0, 2, 1, 3)
    key_linear_prefix = torch.tensor(
        [[[[0.35, -0.15]], [[-0.2, 0.45]]]],
        dtype=dtype,
    )
    key_pre_prefix = _norm(key_linear_prefix, key_norm, epsilon)
    prefix_angle = torch.tensor([[[-0.25, -0.25]]], dtype=dtype)
    prefix_cos = torch.cos(prefix_angle)
    prefix_sin = torch.sin(prefix_angle)
    key_prefix = _rope(
        key_pre_prefix.permute(0, 2, 1, 3),
        prefix_cos.unsqueeze(2),
        prefix_sin.unsqueeze(2),
    ).permute(0, 2, 1, 3)
    key_linear_history = torch.cat((key_linear_prefix, k_linear), dim=2)
    key_pre_history = torch.cat((key_pre_prefix, k_pre), dim=2)
    key_cos_history = torch.cat((prefix_cos, cos), dim=1)
    key_sin_history = torch.cat((prefix_sin, sin), dim=1)
    value_history = torch.tensor(
        [
            [
                [[0.1, 0.5], [0.8, -0.2], [0.4, 0.7]],
                [[-0.3, 0.6], [0.9, 0.1], [-0.5, 0.8]],
            ]
        ],
        dtype=dtype,
    )
    key_history = torch.cat((key_prefix, key_suffix), dim=2)
    repeated_key = key_history.repeat_interleave(2, dim=1)
    additive_mask = torch.zeros((1, 1, 2, 3), dtype=dtype)
    additive_mask[:, :, 0, 2] = -torch.inf
    qk_scores = query @ repeated_key.transpose(2, 3) / (head_dim**0.5)
    probabilities = torch.softmax(qk_scores + additive_mask, dim=-1)
    source_receipt = canonical_sha256(
        {"schema": "test-attention-tangent-source-v1", "phase": "prefill"}
    )
    case = Lfm2AttentionTangentFitCase(
        case_id="batch-0:prefill",
        phase_weight=1.0,
        query_linear_suffix=q_linear,
        key_linear_suffix=k_linear,
        query_normalized_pre_rope=q_pre,
        key_normalized_pre_rope=k_pre,
        rope_cos=cos,
        rope_sin=sin,
        query_rotary_suffix=query,
        key_linear_history=key_linear_history,
        key_normalized_pre_rope_history=key_pre_history,
        key_rope_cos_history=key_cos_history,
        key_rope_sin_history=key_sin_history,
        key_history=key_history,
        value_history=value_history,
        qk_scores=qk_scores,
        probabilities=probabilities,
        additive_mask=additive_mask,
        source_receipt_sha256=source_receipt,
    )
    case_sha = lfm2_attention_tangent_case_evidence_sha256(case)
    arguments = {
        "site_id": "test-layer-0",
        "source_query_weight": source_q,
        "expected_source_query_weight_sha256": lfm2_authenticated_tensor_sha256(
            source_q
        ),
        "source_key_weight": source_k,
        "expected_source_key_weight_sha256": lfm2_authenticated_tensor_sha256(source_k),
        "source_value_weight": source_v,
        "expected_source_value_weight_sha256": lfm2_authenticated_tensor_sha256(
            source_v
        ),
        "source_out_weight": source_o,
        "expected_source_out_weight_sha256": lfm2_authenticated_tensor_sha256(source_o),
        "source_query_norm_weight": query_norm,
        "expected_source_query_norm_weight_sha256": (
            lfm2_authenticated_tensor_sha256(query_norm)
        ),
        "source_key_norm_weight": key_norm,
        "expected_source_key_norm_weight_sha256": (
            lfm2_authenticated_tensor_sha256(key_norm)
        ),
        "normalized_input_decoder": decoder,
        "expected_normalized_input_decoder_sha256": (
            lfm2_authenticated_tensor_sha256(decoder)
        ),
        "residual_output_chart": residual_chart,
        "expected_residual_output_chart_sha256": (
            lfm2_authenticated_tensor_sha256(residual_chart)
        ),
        "norm_epsilon": epsilon,
        "source_query_head_count": query_heads,
        "source_key_value_head_count": key_value_heads,
        "fit_cases": (case,),
        "expected_fit_case_evidence_sha256": canonical_sha256(
            {
                "schema": "neural-forge-lfm2-attention-tangent-fit-bank-v1",
                "case_sha256": [case_sha],
            }
        ),
        "direction_chunk_size": 1,
    }
    return arguments, case


def _snapshot_covariance(
    context: torch.Tensor,
    *,
    groups: int,
) -> torch.Tensor:
    rows, query_heads, head_dim = map(int, context.shape)
    q_per_group = query_heads // groups
    features = (
        context.reshape(rows, groups, q_per_group, head_dim)
        .permute(0, 2, 3, 1)
        .reshape(rows * q_per_group * head_dim, groups)
    )
    return features.T @ features


def test_causal_attention_jvp_matches_central_finite_difference() -> None:
    torch.manual_seed(503)
    dtype = torch.float64
    query = torch.randn(1, 3, 2, 4, dtype=dtype)
    key = torch.randn(1, 3, 5, 4, dtype=dtype)
    value = torch.randn(1, 3, 5, 4, dtype=dtype)
    delta_query = torch.randn_like(query)
    delta_key = torch.randn_like(key)
    delta_value = torch.randn_like(value)
    probabilities = torch.softmax(
        query @ key.transpose(2, 3) / 2.0,
        dim=-1,
    )
    _delta_score, _delta_probability, delta_context = lfm2_attention_causal_jvp(
        query=query,
        key_history=key,
        value_history=value,
        probabilities=probabilities,
        delta_query=delta_query,
        delta_key_history=delta_key,
        delta_value_history=delta_value,
    )

    def context(step: float) -> torch.Tensor:
        q = query + step * delta_query
        k = key + step * delta_key
        v = value + step * delta_value
        probability = torch.softmax(q @ k.transpose(2, 3) / 2.0, dim=-1)
        return probability @ v

    step = 1e-6
    finite_difference = (context(step) - context(-step)) / (2.0 * step)
    torch.testing.assert_close(delta_context, finite_difference, rtol=2e-7, atol=2e-8)


def test_tangent_bank_reduction_preserves_both_group_covariances_and_is_deterministic() -> (
    None
):
    arguments, _case = _fixture()
    first = build_lfm2_attention_tangent_bank(**arguments)
    second = build_lfm2_attention_tangent_bank(**arguments)
    assert first.bank_sha256 == second.bank_sha256
    assert torch.equal(first.query_local.context_rows, second.query_local.context_rows)
    assert torch.equal(
        first.sampled_suffix_history_key_value.context_rows,
        second.sampled_suffix_history_key_value.context_rows,
    )
    assert torch.equal(
        first.joint_sampled_token_query_key_value.context_rows,
        second.joint_sampled_token_query_key_value.context_rows,
    )
    assert torch.equal(
        first.older_prefix_key_value.context_rows,
        second.older_prefix_key_value.context_rows,
    )
    for snapshot in (
        first.query_local,
        first.sampled_suffix_history_key_value,
        first.joint_sampled_token_query_key_value,
        first.older_prefix_key_value,
    ):
        reduced = _snapshot_covariance(snapshot.context_rows, groups=2)
        torch.testing.assert_close(reduced, snapshot.covariance, rtol=1e-12, atol=1e-12)
        receipt = snapshot.reduction_receipt
        assert receipt["covariance_gram_within_fp64_roundoff"] is True
        assert receipt["positive_finite_eigenmodes_removed"] == 0
        if receipt["covariance_reduction_changes_qdeim_group_pod_ftf"]:
            assert receipt["covariance_reduction_change_is_within_fp64_roundoff"]
    report = first.report
    assert report["geometry"]["authenticated_child_decoder_direction_count"] == 2
    assert report["query_local_and_history_key_value_evidence_separately_labelled"]
    assert report["cross_phase_older_key_anchor_stitch_implemented"] is True
    assert report["complete_key_value_history_tangent_coverage"] is True
    assert report["unsupported_historical_prefix_key_position_count"] == 0
    assert report["cross_phase_stitched_historical_prefix_key_position_count"] == 1
    assert (
        report["precise_missing_cached_evidence_field_for_older_history_key_tangent"]
        is None
    )
    assert (
        report["precise_missing_authenticated_transform_for_older_history_key_tangent"]
        is None
    )
    assert report["complete_value_history_tangent_coverage"]
    assert report["functional_pod_uses_joint_sampled_token_qkv_cross_terms"]
    assert report["functional_pod_double_counts_partial_q_or_kv_covariance"] is False
    assert report["source_o_and_residual_chart_applied_before_group_covariance"]
    assert report[
        "partial_diagnostic_family_scope_is_explicit_and_not_promoted_to_"
        "functional_complete"
    ]
    assert report["heldout_tensor_payloads_used"] == 0
    assert report["operation_audit"]["training_steps"] == 0
    assert report["operation_audit"]["autograd_grad_calls"] == 0


@pytest.mark.parametrize(
    ("covariance", "expected_factor_rows", "expected_positive_modes"),
    (
        (torch.diag(torch.tensor([1.0, 1e-13], dtype=torch.float64)), 2, 2),
        (torch.zeros((2, 2), dtype=torch.float64), 1, 0),
        (
            torch.diag(torch.tensor([1.0, 1e-20, 1e-40], dtype=torch.float64)),
            3,
            3,
        ),
    ),
    ids=("diag-one-and-one-e-minus-thirteen", "zero", "ill-conditioned"),
)
def test_covariance_snapshot_preserves_every_positive_finite_mode(
    covariance: torch.Tensor,
    expected_factor_rows: int,
    expected_positive_modes: int,
) -> None:
    groups = int(covariance.shape[0])
    snapshot = tangent_module._covariance_snapshot(
        covariance,
        block_id="tiny-mode-test",
        perturbation_kind="tiny_mode_test",
        query_heads=groups,
        key_value_heads=groups,
        head_dim=1,
        original_tangent_context_row_count=1,
        original_weight_sum=1.0,
    )
    reduced = _snapshot_covariance(snapshot.context_rows, groups=groups)

    assert int(snapshot.context_rows.shape[0]) == expected_factor_rows
    assert int((torch.linalg.eigvalsh(reduced) > 0.0).sum().item()) == (
        expected_positive_modes
    )
    torch.testing.assert_close(reduced, covariance, rtol=1e-14, atol=1e-300)
    receipt = snapshot.reduction_receipt
    assert receipt["positive_finite_eigenmodes_removed"] == 0
    assert receipt["covariance_gram_within_fp64_roundoff"]
    assert receipt["negative_roundoff_eigenvalues_clamped_to_zero"] == 0


def test_vectorized_chunks_preserve_batch_two_covariances() -> None:
    arguments, case = _fixture()

    def repeated(value: torch.Tensor) -> torch.Tensor:
        return value.repeat(2, *([1] * (value.ndim - 1))).contiguous()

    batched = Lfm2AttentionTangentFitCase(
        case_id="batch-two:prefill",
        phase_weight=case.phase_weight,
        query_linear_suffix=repeated(case.query_linear_suffix),
        key_linear_suffix=repeated(case.key_linear_suffix),
        query_normalized_pre_rope=repeated(case.query_normalized_pre_rope),
        key_normalized_pre_rope=repeated(case.key_normalized_pre_rope),
        rope_cos=repeated(case.rope_cos),
        rope_sin=repeated(case.rope_sin),
        query_rotary_suffix=repeated(case.query_rotary_suffix),
        key_linear_history=repeated(case.key_linear_history),
        key_normalized_pre_rope_history=repeated(case.key_normalized_pre_rope_history),
        key_rope_cos_history=repeated(case.key_rope_cos_history),
        key_rope_sin_history=repeated(case.key_rope_sin_history),
        key_history=repeated(case.key_history),
        value_history=repeated(case.value_history),
        qk_scores=repeated(case.qk_scores),
        probabilities=repeated(case.probabilities),
        additive_mask=repeated(case.additive_mask),
        source_receipt_sha256=case.source_receipt_sha256,
    )
    case_sha = lfm2_attention_tangent_case_evidence_sha256(batched)
    arguments["fit_cases"] = (batched,)
    arguments["expected_fit_case_evidence_sha256"] = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-attention-tangent-fit-bank-v1",
            "case_sha256": [case_sha],
        }
    )
    arguments["direction_chunk_size"] = 1
    arguments["history_position_chunk_size"] = 1
    arguments["older_history_position_chunk_size"] = 1
    scalar_chunks = build_lfm2_attention_tangent_bank(**arguments)
    arguments["direction_chunk_size"] = 2
    arguments["history_position_chunk_size"] = 2
    arguments["older_history_position_chunk_size"] = 2
    full_chunks = build_lfm2_attention_tangent_bank(**arguments)
    for left, right in (
        (scalar_chunks.query_local, full_chunks.query_local),
        (
            scalar_chunks.sampled_suffix_history_key_value,
            full_chunks.sampled_suffix_history_key_value,
        ),
        (
            scalar_chunks.joint_sampled_token_query_key_value,
            full_chunks.joint_sampled_token_query_key_value,
        ),
        (
            scalar_chunks.older_prefix_key_value,
            full_chunks.older_prefix_key_value,
        ),
    ):
        torch.testing.assert_close(
            left.covariance,
            right.covariance,
            rtol=1e-12,
            atol=1e-12,
        )


def test_query_local_and_sampled_suffix_covariances_match_finite_difference() -> None:
    arguments, case = _fixture()
    bank = build_lfm2_attention_tangent_bank(**arguments)
    source_q = arguments["source_query_weight"]
    source_k = arguments["source_key_weight"]
    source_v = arguments["source_value_weight"]
    source_o = arguments["source_out_weight"]
    residual_chart = arguments["residual_output_chart"]
    decoder = arguments["normalized_input_decoder"]
    query_norm = arguments["source_query_norm_weight"]
    key_norm = arguments["source_key_norm_weight"]
    epsilon = float(arguments["norm_epsilon"])
    query_heads = int(arguments["source_query_head_count"])
    key_value_heads = int(arguments["source_key_value_head_count"])
    head_dim = int(source_q.shape[0]) // query_heads
    q_per_group = query_heads // key_value_heads
    child_hidden = int(decoder.shape[0])
    q_directions = (decoder @ source_q.T).reshape(
        child_hidden,
        query_heads,
        head_dim,
    )
    k_directions = (decoder @ source_k.T).reshape(
        child_hidden,
        key_value_heads,
        head_dim,
    )
    v_directions = (decoder @ source_v.T).reshape(
        child_hidden,
        key_value_heads,
        head_dim,
    )
    query_rows = int(case.query_linear_suffix.shape[2])
    suffix_rows = int(case.key_linear_suffix.shape[2])
    history_rows = int(case.key_history.shape[2])
    group_width = q_per_group * head_dim
    group_residual_maps = (
        (residual_chart.T @ source_o)
        .reshape(-1, key_value_heads, group_width)
        .permute(1, 2, 0)
    )

    def context(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        repeated_key = key.repeat_interleave(q_per_group, dim=1)
        repeated_value = value.repeat_interleave(q_per_group, dim=1)
        probability = torch.softmax(
            query @ repeated_key.transpose(2, 3) / (head_dim**0.5) + case.additive_mask,
            dim=-1,
        )
        return probability @ repeated_value

    def features(rows: torch.Tensor) -> torch.Tensor:
        grouped = rows.reshape(-1, key_value_heads, group_width)
        mapped = torch.einsum("rgw,gwt->rgt", grouped, group_residual_maps)
        return mapped.permute(0, 2, 1).reshape(-1, key_value_heads)

    step = 1e-6
    query_covariance = torch.zeros(
        (key_value_heads, key_value_heads),
        dtype=torch.float64,
    )
    query_row_weight = 1.0 / (query_rows * child_hidden)
    for query_index in range(query_rows):
        for direction_index in range(child_hidden):
            outputs = []
            for sign in (-1.0, 1.0):
                q_linear = case.query_linear_suffix.clone()
                q_linear[:, :, query_index, :] += (
                    sign * step * q_directions[direction_index]
                )
                q_pre = _norm(q_linear, query_norm, epsilon)
                query = _rope(
                    q_pre.permute(0, 2, 1, 3),
                    case.rope_cos.unsqueeze(2),
                    case.rope_sin.unsqueeze(2),
                ).permute(0, 2, 1, 3)
                outputs.append(
                    context(query, case.key_history, case.value_history)[
                        :, :, query_index, :
                    ]
                )
            tangent = (outputs[1] - outputs[0]) / (2.0 * step)
            group_features = features(tangent)
            query_covariance += query_row_weight * (group_features.T @ group_features)

    history_covariance = torch.zeros_like(query_covariance)
    joint_covariance = torch.zeros_like(query_covariance)
    history_row_weight = 1.0 / (suffix_rows * child_hidden * query_rows)
    for suffix_index in range(suffix_rows):
        history_index = history_rows - suffix_rows + suffix_index
        for direction_index in range(child_hidden):
            outputs = []
            joint_outputs = []
            for sign in (-1.0, 1.0):
                k_linear = case.key_linear_suffix.clone()
                k_linear[:, :, suffix_index, :] += (
                    sign * step * k_directions[direction_index]
                )
                k_pre = _norm(k_linear, key_norm, epsilon)
                key_suffix = _rope(
                    k_pre.permute(0, 2, 1, 3),
                    case.rope_cos.unsqueeze(2),
                    case.rope_sin.unsqueeze(2),
                ).permute(0, 2, 1, 3)
                key = case.key_history.clone()
                key[:, :, history_index, :] = key_suffix[:, :, suffix_index, :]
                value = case.value_history.clone()
                value[:, :, history_index, :] += (
                    sign * step * v_directions[direction_index]
                )
                outputs.append(context(case.query_rotary_suffix, key, value))
                q_linear = case.query_linear_suffix.clone()
                q_linear[:, :, suffix_index, :] += (
                    sign * step * q_directions[direction_index]
                )
                q_pre = _norm(q_linear, query_norm, epsilon)
                joint_query = _rope(
                    q_pre.permute(0, 2, 1, 3),
                    case.rope_cos.unsqueeze(2),
                    case.rope_sin.unsqueeze(2),
                ).permute(0, 2, 1, 3)
                joint_outputs.append(context(joint_query, key, value))
            tangent = (outputs[1] - outputs[0]) / (2.0 * step)
            rows = tangent.permute(0, 2, 1, 3).reshape(
                -1,
                query_heads,
                head_dim,
            )
            group_features = features(rows)
            history_covariance += history_row_weight * (
                group_features.T @ group_features
            )
            joint_tangent = (joint_outputs[1] - joint_outputs[0]) / (2.0 * step)
            joint_rows = joint_tangent.permute(0, 2, 1, 3).reshape(
                -1,
                query_heads,
                head_dim,
            )
            joint_features = features(joint_rows)
            joint_covariance += history_row_weight * (joint_features.T @ joint_features)

    older_key_value_covariance = torch.zeros_like(query_covariance)
    older_prefix_rows = history_rows - suffix_rows
    older_value_row_weight = 1.0 / (older_prefix_rows * child_hidden * query_rows)
    for history_index in range(older_prefix_rows):
        for direction_index in range(child_hidden):
            outputs = []
            for sign in (-1.0, 1.0):
                key_linear_history = case.key_linear_history.clone()
                key_linear_history[:, :, history_index, :] += (
                    sign * step * k_directions[direction_index]
                )
                key_pre_history = _norm(
                    key_linear_history,
                    key_norm,
                    epsilon,
                )
                key = _rope(
                    key_pre_history.permute(0, 2, 1, 3),
                    case.key_rope_cos_history.unsqueeze(2),
                    case.key_rope_sin_history.unsqueeze(2),
                ).permute(0, 2, 1, 3)
                value = case.value_history.clone()
                value[:, :, history_index, :] += (
                    sign * step * v_directions[direction_index]
                )
                outputs.append(context(case.query_rotary_suffix, key, value))
            tangent = (outputs[1] - outputs[0]) / (2.0 * step)
            rows = tangent.permute(0, 2, 1, 3).reshape(
                -1,
                query_heads,
                head_dim,
            )
            group_features = features(rows)
            older_key_value_covariance += older_value_row_weight * (
                group_features.T @ group_features
            )

    torch.testing.assert_close(
        bank.query_local.covariance,
        query_covariance,
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        bank.sampled_suffix_history_key_value.covariance,
        history_covariance,
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        bank.joint_sampled_token_query_key_value.covariance,
        joint_covariance,
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        bank.older_prefix_key_value.covariance,
        older_key_value_covariance,
        rtol=2e-6,
        atol=2e-8,
    )


def test_tangent_bank_rejects_source_and_case_tamper() -> None:
    arguments, case = _fixture()
    arguments["expected_source_query_weight_sha256"] = "0" * 64
    with pytest.raises(
        Lfm2AttentionTangentAuthenticationError,
        match="source_query_weight differs",
    ):
        build_lfm2_attention_tangent_bank(**arguments)

    arguments, _case = _fixture()
    tampered = Lfm2AttentionTangentFitCase(
        case_id=case.case_id,
        phase_weight=case.phase_weight,
        query_linear_suffix=case.query_linear_suffix + 0.125,
        key_linear_suffix=case.key_linear_suffix,
        query_normalized_pre_rope=case.query_normalized_pre_rope,
        key_normalized_pre_rope=case.key_normalized_pre_rope,
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        query_rotary_suffix=case.query_rotary_suffix,
        key_linear_history=case.key_linear_history,
        key_normalized_pre_rope_history=case.key_normalized_pre_rope_history,
        key_rope_cos_history=case.key_rope_cos_history,
        key_rope_sin_history=case.key_rope_sin_history,
        key_history=case.key_history,
        value_history=case.value_history,
        qk_scores=case.qk_scores,
        probabilities=case.probabilities,
        additive_mask=case.additive_mask,
        source_receipt_sha256=case.source_receipt_sha256,
    )
    arguments["fit_cases"] = (tampered,)
    with pytest.raises(
        Lfm2AttentionTangentAuthenticationError,
        match="FIT attention tangent cases differ",
    ):
        build_lfm2_attention_tangent_bank(**arguments)
