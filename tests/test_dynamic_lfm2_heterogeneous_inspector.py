from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_circuit_ir import CircuitKind  # noqa: E402
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    DynamicModelInspectionError,
    inspect_model_circuits,
)
from neural_forge.hf_lfm2_dynamic_blank_materializer import (  # noqa: E402
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_heterogeneous_physical_layout,
)


def _descriptor():
    layers = (
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=0,
            mlp_intermediate_size=5,
            internal_width=10,
            writer_rank=3,
            selection_evidence_sha256="0" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=1,
            mlp_intermediate_size=6,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            selection_evidence_sha256="1" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=2,
            mlp_intermediate_size=7,
            query_heads=4,
            key_value_heads=2,
            head_dim=2,
            selection_evidence_sha256="2" * 64,
        ),
    )
    return build_lfm2_heterogeneous_physical_layout(
        hidden_size=8,
        vocab_size=16,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        layers=layers,
        source_topology_sha256="a" * 64,
        allocation_evidence_sha256="b" * 64,
        token_bridge_evidence_sha256="c" * 64,
    )


def _config():
    return transformers.Lfm2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=9,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _materialized():
    config = _config()
    model = transformers.Lfm2ForCausalLM(config)
    descriptor = _descriptor()
    materialize_lfm2_dynamic_blank_child(
        model,
        config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    return model, descriptor


def test_inspects_sealed_heterogeneous_blank_with_exact_physical_geometry() -> None:
    model, descriptor = _materialized()

    graph = inspect_model_circuits(
        model,
        model_name="heterogeneous-blank",
        tokenizer_fingerprint="d" * 64,
    )

    assert graph.family == "lfm2"
    assert graph.layer_indices == (0, 1, 2)
    assert graph.logical_parameter_count == (
        descriptor.accounting.logical_parameter_elements
    )
    assert graph.unique_parameter_count == (
        descriptor.accounting.unique_parameter_elements
    )
    assert graph.logical_parameter_count == graph.unique_parameter_count
    assert graph.unique_parameter_bytes == descriptor.accounting.parameter_bytes_fp32
    assert set(graph.parameter_paths) == {
        name
        for name, _ in model.named_parameters(recurse=True, remove_duplicate=False)
    }
    parameter_specs = tuple(
        parameter for node in graph.nodes for parameter in node.parameters
    )
    assert len({parameter.storage_group for parameter in parameter_specs}) == len(
        parameter_specs
    )
    assert all(not parameter.trainable for parameter in parameter_specs)

    short_conv = graph.node("layer.000.operator")
    assert short_conv.kind == CircuitKind.SHORT_CONV.value
    assert dict(short_conv.attributes) == {
        "external_hidden_size": 8,
        "fir_length": 3,
        "groups": 10,
        "input_chunk_order": "b_c_x",
        "input_projection_rows": 30,
        "internal_channel_width": 10,
        "not_attention_qkv": True,
        "operator_type": "short_conv",
        "sealed_heterogeneous_physical_geometry": True,
        "selection_evidence_sha256": "0" * 64,
        "writer_factor_count": 2,
        "writer_rank": 3,
        "writer_topology": "factorized",
    }
    assert {parameter.role: parameter.shape for parameter in short_conv.parameters} == {
        "operator.conv_weight": (10, 1, 3),
        "operator.in_proj_weight": (30, 8),
        "operator.out_proj_0_weight": (3, 10),
        "operator.out_proj_1_weight": (8, 3),
    }

    attention = graph.node("layer.002.operator")
    assert dict(attention.attributes) == {
        "external_hidden_size": 8,
        "grouped_query_attention": True,
        "head_dim": 2,
        "key_value_heads": 2,
        "key_value_projection_width": 4,
        "operator_type": "full_attention",
        "qk_rmsnorm": True,
        "query_heads": 4,
        "query_projection_width": 8,
        "sealed_heterogeneous_physical_geometry": True,
        "selection_evidence_sha256": "2" * 64,
    }
    assert {parameter.role: parameter.shape for parameter in attention.parameters} == {
        "operator.k_layernorm_weight": (2,),
        "operator.k_proj_weight": (4, 8),
        "operator.out_proj_weight": (8, 8),
        "operator.q_layernorm_weight": (2,),
        "operator.q_proj_weight": (8, 8),
        "operator.v_proj_weight": (4, 8),
    }
    mlp = graph.node("layer.002.mlp")
    assert dict(mlp.attributes)["intermediate_size"] == 7
    assert {parameter.role: parameter.shape for parameter in mlp.parameters} == {
        "mlp.w1_weight": (7, 8),
        "mlp.w2_weight": (8, 7),
        "mlp.w3_weight": (7, 8),
    }
    metadata = dict(graph.metadata)
    assert metadata["physical_layout"] == "heterogeneous_dense_fp32"
    assert metadata["physical_layout_descriptor_sha256"] == (
        descriptor.descriptor_sha256
    )
    assert metadata["runtime_parameter_alias_group_count"] == 0
    assert metadata["cross_layer_parameter_sharing"] is False
    assert metadata["runtime_bridge_parameter_elements"] == 0
    assert metadata["runtime_sidecar_parameter_elements"] == 0
    assert metadata["parameters_quantized"] == 0
    assert metadata["sparse_parameter_storage"] is False
    assert metadata["pruning_masks_present"] is False


def test_heterogeneous_inspector_rejects_config_and_live_storage_tampering() -> None:
    model, _ = _materialized()
    raw = copy.deepcopy(model.config.neural_forge_dynamic_physical_layout)
    raw["layers"][0]["short_conv_internal_width"] += 1
    model.config.neural_forge_dynamic_physical_layout = raw
    with pytest.raises(
        DynamicModelInspectionError,
        match="config authentication failed",
    ):
        inspect_model_circuits(model, model_name="tampered-config")

    model, _ = _materialized()
    model.model.layers[2].self_attn.q_proj.weight = (
        model.model.layers[1].self_attn.q_proj.weight
    )
    with pytest.raises(
        DynamicModelInspectionError,
        match="live physical materialization authentication failed",
    ):
        inspect_model_circuits(model, model_name="aliased-live-storage")


def test_sealed_config_is_not_materialized_as_an_inspection_side_effect() -> None:
    materialized, _ = _materialized()
    config = transformers.Lfm2Config(**materialized.config.to_dict())
    fresh = transformers.Lfm2ForCausalLM(config)
    original_mlp_type = type(fresh.model.layers[0].feed_forward)

    with pytest.raises(
        DynamicModelInspectionError,
        match="not an already-materialized runtime",
    ):
        inspect_model_circuits(fresh, model_name="sealed-but-native")

    assert type(fresh.model.layers[0].feed_forward) is original_mlp_type
    assert not hasattr(fresh, "_neural_forge_dynamic_blank_runtime_sha256")


def test_native_lfm2_inspection_retains_uniform_canonical_contract() -> None:
    model = transformers.Lfm2ForCausalLM(_config()).eval()

    graph = inspect_model_circuits(model, model_name="native-lfm2")

    assert graph.logical_parameter_count == sum(
        parameter.numel()
        for _, parameter in model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )
    assert "physical_layout" not in dict(graph.metadata)
    assert dict(graph.node("layer.000.operator").attributes) == {
        "fir_length": 3,
        "groups": 8,
        "input_chunk_order": "b_c_x",
        "not_attention_qkv": True,
        "operator_type": "short_conv",
    }
    assert dict(graph.node("layer.002.operator").attributes) == {
        "grouped_query_attention": True,
        "head_dim": 4,
        "key_value_heads": 1,
        "operator_type": "full_attention",
        "qk_rmsnorm": True,
        "query_heads": 2,
    }
