from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_attention_group_qdeim import (  # noqa: E402
    Lfm2AttentionGroupQdeimError,
    Lfm2AttentionGroupQdeimSpanSource,
    assemble_lfm2_attention_group_qdeim_span_sources,
    compile_lfm2_attention_group_span_qdeim,
    lfm2_attention_group_qdeim_span_context_evidence_sha256,
    lfm2_attention_group_qdeim_span_source_state_sha256,
    lfm2_attention_group_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_model_fold import Lfm2ModelFoldSpan  # noqa: E402
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


def _span_case(source_count: int) -> dict[str, Any]:
    source_indices = tuple(range(7, 7 + source_count))
    span = Lfm2ModelFoldSpan(source_indices, 2)
    source_groups = 2
    target_groups = source_count
    queries_per_group = 2
    head_dim = 2
    rows = 4
    source_hidden = 3
    target_hidden = 2
    feature_rows = rows * queries_per_group * head_dim

    coordinate = torch.linspace(0.2, 1.7, feature_rows, dtype=torch.float64)
    latent = torch.stack(
        [coordinate.pow(power + 1) for power in range(source_count)],
        dim=1,
    )
    mixing = torch.zeros(
        (source_count, source_count * source_groups),
        dtype=torch.float64,
    )
    for source_offset in range(source_count):
        mixing[source_offset, 2 * source_offset] = 1.0
        mixing[source_offset, 2 * source_offset + 1] = 1.0
        mixing[(source_offset + 1) % source_count, 2 * source_offset + 1] = 0.2
    union_features = latent @ mixing
    alignment_sha256 = canonical_sha256(
        {"aligned_prompt_token_rows": list(range(rows))}
    )

    sources_by_layer: dict[int, Lfm2AttentionGroupQdeimSpanSource] = {}
    raw: dict[int, dict[str, torch.Tensor]] = {}
    for source_offset, layer_index in enumerate(source_indices):
        source_q_heads = source_groups * queries_per_group
        base = 1.0 + 3.0 * source_offset
        query = (
            torch.arange(
                source_q_heads * head_dim * source_hidden,
                dtype=torch.float64,
            )
            .reshape(source_q_heads * head_dim, source_hidden)
            .add(base)
            .div(13.0)
        )
        key = (
            torch.arange(
                source_groups * head_dim * source_hidden,
                dtype=torch.float64,
            )
            .reshape(source_groups * head_dim, source_hidden)
            .add(2.0 * base)
            .div(17.0)
        )
        value = key.mul(-0.4).add(base)
        out = (
            torch.arange(
                source_hidden * source_q_heads * head_dim,
                dtype=torch.float64,
            )
            .reshape(source_hidden, source_q_heads * head_dim)
            .add(4.0 * base)
            .div(19.0)
        )
        decoder = torch.tensor(
            [
                [1.0, 0.0, 0.1 * (source_offset + 1)],
                [0.0, 1.0, -0.05 * (source_offset + 1)],
            ],
            dtype=torch.float64,
        )
        chart = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.15 * (source_offset + 1), -0.1],
            ],
            dtype=torch.float64,
        )
        query_norm = torch.tensor(
            [1.0 + source_offset, 1.5 + source_offset],
            dtype=torch.bfloat16,
        )
        key_norm = torch.tensor(
            [2.0 + source_offset, 2.5 + source_offset],
            dtype=torch.bfloat16,
        )
        local_features = union_features[
            :,
            source_offset * source_groups : (source_offset + 1) * source_groups,
        ].contiguous()
        contexts = _contexts_from_group_features(
            local_features,
            rows=rows,
            groups=source_groups,
            queries_per_group=queries_per_group,
            head_dim=head_dim,
        )
        source_state_sha256 = lfm2_attention_group_qdeim_span_source_state_sha256(
            query,
            key,
            value,
            out,
            query_norm,
            key_norm,
        )
        sources_by_layer[layer_index] = Lfm2AttentionGroupQdeimSpanSource(
            source_layer_index=layer_index,
            source_query_weight=query,
            source_key_weight=key,
            source_value_weight=value,
            source_out_weight=out,
            expected_source_state_sha256=source_state_sha256,
            source_query_head_count=source_q_heads,
            source_key_value_head_count=source_groups,
            cached_parent_post_softmax_context_rows=contexts,
            expected_cached_parent_post_softmax_context_sha256=(
                lfm2_authenticated_tensor_sha256(contexts)
            ),
            cache_receipt_sha256=canonical_sha256(
                {"sealed_cache_receipt_for_layer": layer_index}
            ),
            aligned_row_identity_sha256=alignment_sha256,
            normalized_input_decoder=decoder,
            expected_normalized_input_decoder_sha256=(
                lfm2_authenticated_tensor_sha256(decoder)
            ),
            residual_output_chart=chart,
            expected_residual_output_chart_sha256=(
                lfm2_authenticated_tensor_sha256(chart)
            ),
            source_query_norm_weight=query_norm,
            source_key_norm_weight=key_norm,
        )
        raw[layer_index] = {
            "query": query,
            "key": key,
            "value": value,
            "out": out,
            "decoder": decoder,
            "chart": chart,
            "query_norm": query_norm,
            "key_norm": key_norm,
            "contexts": contexts,
            "features": local_features,
        }

    sources = assemble_lfm2_attention_group_qdeim_span_sources(
        span=span,
        attention_source_layer_indices=source_indices,
        sources_by_layer=sources_by_layer,
    )
    row_weights = (1.0, 0.5, 2.0, 1.5)
    templates = (
        torch.empty(
            target_groups * queries_per_group * head_dim,
            target_hidden,
            dtype=torch.bfloat16,
        ),
        torch.empty(
            target_groups * head_dim,
            target_hidden,
            dtype=torch.bfloat16,
        ),
        torch.empty(
            target_groups * head_dim,
            target_hidden,
            dtype=torch.bfloat16,
        ),
        torch.empty(
            target_hidden,
            target_groups * queries_per_group * head_dim,
            dtype=torch.bfloat16,
        ),
    )
    return {
        "span": span,
        "sources": sources,
        "expected_context_evidence_sha256": (
            lfm2_attention_group_qdeim_span_context_evidence_sha256(
                span=span,
                sources=sources,
                row_sample_weights=row_weights,
            )
        ),
        "target_query_weight_template": templates[0],
        "target_key_weight_template": templates[1],
        "target_value_weight_template": templates[2],
        "target_out_weight_template": templates[3],
        "expected_target_template_sha256": (
            lfm2_attention_group_qdeim_target_template_sha256(*templates)
        ),
        "target_query_head_count": target_groups * queries_per_group,
        "target_key_value_head_count": target_groups,
        "row_sample_weights": row_weights,
        "_raw": raw,
        "_union_features": union_features,
        "_source_groups": source_groups,
        "_queries_per_group": queries_per_group,
        "_head_dim": head_dim,
        "_rows": rows,
    }


def _compile_and_prove(source_count: int) -> None:
    case = _span_case(source_count)
    raw = case.pop("_raw")
    union_features = case.pop("_union_features")
    source_groups = case.pop("_source_groups")
    queries_per_group = case.pop("_queries_per_group")
    head_dim = case.pop("_head_dim")
    rows = case.pop("_rows")
    result = compile_lfm2_attention_group_span_qdeim(**case)

    selected_pairs = result.selected_source_key_value_groups
    assert len(selected_pairs) == source_count
    assert {layer_index for layer_index, _group in selected_pairs} == set(
        case["span"].source_layer_indices
    )
    assert all(
        count == 1 for count in result.report["selected_group_counts_by_layer"].values()
    )

    selected_global = []
    layer_to_offset = {
        layer_index: offset
        for offset, layer_index in enumerate(case["span"].source_layer_indices)
    }
    for layer_index, local_group in selected_pairs:
        selected_global.append(
            layer_to_offset[layer_index] * source_groups + local_group
        )
    selected_features = union_features[:, selected_global]
    reconstructed_features = selected_features @ result.group_interpolation
    assert torch.allclose(
        reconstructed_features,
        union_features,
        atol=2e-11,
        rtol=2e-11,
    )
    assert torch.allclose(
        result.group_interpolation[:, selected_global],
        torch.eye(source_count, dtype=torch.float64),
        atol=2e-11,
        rtol=2e-11,
    )

    union_flat = torch.cat(
        [
            raw[layer_index]["contexts"]
            .reshape(rows, source_groups, queries_per_group, head_dim)
            .reshape(rows, source_groups * queries_per_group * head_dim)
            for layer_index in case["span"].source_layer_indices
        ],
        dim=1,
    )
    selected_flat = torch.cat(
        [
            raw[layer_index]["contexts"].reshape(
                rows, source_groups, queries_per_group * head_dim
            )[:, local_group]
            for layer_index, local_group in selected_pairs
        ],
        dim=1,
    )
    assert torch.allclose(
        selected_flat @ result.context_reconstruction,
        union_flat,
        atol=2e-11,
        rtol=2e-11,
    )

    expected_q = []
    expected_k = []
    expected_v = []
    expected_q_norm = []
    expected_k_norm = []
    for layer_index, local_group in selected_pairs:
        source = raw[layer_index]
        local_heads = tuple(
            local_group * queries_per_group + offset
            for offset in range(queries_per_group)
        )
        query_rows = [
            head * head_dim + coordinate
            for head in local_heads
            for coordinate in range(head_dim)
        ]
        key_value_rows = [
            local_group * head_dim + coordinate for coordinate in range(head_dim)
        ]
        expected_q.append(source["query"][query_rows] @ source["decoder"].T)
        expected_k.append(source["key"][key_value_rows] @ source["decoder"].T)
        expected_v.append(source["value"][key_value_rows] @ source["decoder"].T)
        expected_q_norm.append(
            source["query_norm"].unsqueeze(0).repeat(queries_per_group, 1)
        )
        expected_k_norm.append(source["key_norm"].unsqueeze(0))
    assert torch.equal(result.query_weight, torch.cat(expected_q).to(torch.bfloat16))
    assert torch.equal(result.key_weight, torch.cat(expected_k).to(torch.bfloat16))
    assert torch.equal(result.value_weight, torch.cat(expected_v).to(torch.bfloat16))
    assert torch.equal(
        result.selected_query_norm_weight_rows,
        torch.cat(expected_q_norm).to(torch.bfloat16),
    )
    assert torch.equal(
        result.selected_key_norm_weight_rows,
        torch.cat(expected_k_norm).to(torch.bfloat16),
    )

    group_width = queries_per_group * head_dim
    expected_o = torch.zeros_like(result.out_weight, dtype=torch.float64)
    expected_parent_output = torch.zeros(
        (rows, result.out_weight.shape[0]),
        dtype=torch.float64,
    )
    for source_offset, layer_index in enumerate(case["span"].source_layer_indices):
        source = raw[layer_index]
        interpolation_slice = result.group_interpolation[
            :,
            source_offset * source_groups : (source_offset + 1) * source_groups,
        ]
        transport = torch.kron(
            interpolation_slice,
            torch.eye(group_width, dtype=torch.float64),
        )
        expected_o.add_(source["chart"].T @ source["out"] @ transport.T)
        source_flat = source["contexts"].reshape(
            rows,
            source_groups * group_width,
        )
        expected_parent_output.add_(source_flat @ (source["chart"].T @ source["out"]).T)
    assert torch.equal(result.out_weight, expected_o.to(torch.bfloat16))
    child_output = selected_flat @ result.out_weight.to(torch.float64).T
    assert torch.allclose(
        child_output,
        expected_parent_output,
        atol=0.25,
        rtol=0.02,
    )

    writer_contributions = result.report["writer_transport"]["source_contributions"]
    assert len(writer_contributions) == source_count
    assert all(
        item["writer_contribution_frobenius_norm"] > 0.0
        and item["complete_source_output_projection_consumed"] is True
        for item in writer_contributions
    )
    assert (
        result.report["qk_norm_lineage"][
            "heterogeneous_selected_norms_require_architecture_aware_wiring"
        ]
        is True
    )
    assert (
        result.report["remaining_integration"][
            "upstream_cache_receipt_content_verification_required"
        ]
        is True
    )
    assert (
        result.report["remaining_integration"][
            "production_semantic_stack_wiring_required"
        ]
        is True
    )
    assert (
        result.report["remaining_integration"][
            "behavior_or_intelligence_retention_claimed_here"
        ]
        is False
    )
    assert result.report["training_steps"] == 0
    assert result.report["parameters_pruned"] == 0
    assert result.report["parameters_quantized"] == 0
    for tensor in result.state_dict().values():
        assert tensor.dtype == torch.bfloat16
        assert tensor.layout == torch.strided
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_exact_synthetic_two_attention_layers_fold_to_one_dense_child() -> None:
    _compile_and_prove(2)


def test_exact_synthetic_three_attention_layers_fold_to_one_dense_child() -> None:
    _compile_and_prove(3)


def test_semantic_stack_seam_rejects_missing_attention_payload() -> None:
    case = _span_case(2)
    sources = case["sources"]
    with pytest.raises(Lfm2AttentionGroupQdeimError, match="exactly cover"):
        assemble_lfm2_attention_group_qdeim_span_sources(
            span=case["span"],
            attention_source_layer_indices=case["span"].source_layer_indices,
            sources_by_layer={sources[0].source_layer_index: sources[0]},
        )


def test_semantic_stack_seam_accepts_attention_subset_inside_mixed_span() -> None:
    case = _span_case(2)
    sources = case["sources"]
    mixed_span = Lfm2ModelFoldSpan((6, 7, 8), 2)
    assembled = assemble_lfm2_attention_group_qdeim_span_sources(
        span=mixed_span,
        attention_source_layer_indices=(7, 8),
        sources_by_layer={source.source_layer_index: source for source in sources},
    )
    assert assembled == sources


def test_span_context_authentication_fails_closed() -> None:
    case = _span_case(2)
    case.pop("_raw")
    case.pop("_union_features")
    case.pop("_source_groups")
    case.pop("_queries_per_group")
    case.pop("_head_dim")
    case.pop("_rows")
    case["expected_context_evidence_sha256"] = "0" * 64
    with pytest.raises(Lfm2AttentionGroupQdeimError, match="evidence changed"):
        compile_lfm2_attention_group_span_qdeim(**case)


@pytest.mark.parametrize(
    "component",
    ("source_state", "norm_state", "decoder", "chart"),
)
def test_external_context_root_rejects_paired_tensor_hash_substitution(
    component: str,
) -> None:
    case = _span_case(2)
    old_root = case["expected_context_evidence_sha256"]
    source = case["sources"][0]
    if component == "source_state":
        source.source_query_weight[0, 0] += 0.5
        source = replace(
            source,
            expected_source_state_sha256=(
                lfm2_attention_group_qdeim_span_source_state_sha256(
                    source.source_query_weight,
                    source.source_key_weight,
                    source.source_value_weight,
                    source.source_out_weight,
                    source.source_query_norm_weight,
                    source.source_key_norm_weight,
                )
            ),
        )
    elif component == "norm_state":
        source.source_query_norm_weight[0] += 1.0
        source = replace(
            source,
            expected_source_state_sha256=(
                lfm2_attention_group_qdeim_span_source_state_sha256(
                    source.source_query_weight,
                    source.source_key_weight,
                    source.source_value_weight,
                    source.source_out_weight,
                    source.source_query_norm_weight,
                    source.source_key_norm_weight,
                )
            ),
        )
    elif component == "decoder":
        source.normalized_input_decoder[0, 0] += 0.5
        source = replace(
            source,
            expected_normalized_input_decoder_sha256=(
                lfm2_authenticated_tensor_sha256(source.normalized_input_decoder)
            ),
        )
    else:
        source.residual_output_chart[0, 0] += 0.5
        source = replace(
            source,
            expected_residual_output_chart_sha256=(
                lfm2_authenticated_tensor_sha256(source.residual_output_chart)
            ),
        )
    case["sources"] = (source, case["sources"][1])
    arguments = {key: value for key, value in case.items() if not key.startswith("_")}
    arguments["expected_context_evidence_sha256"] = old_root

    with pytest.raises(Lfm2AttentionGroupQdeimError, match="evidence changed"):
        compile_lfm2_attention_group_span_qdeim(**arguments)
