from __future__ import annotations

import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from neural_forge.hf_lfm2_dynamic_blank_materializer import (  # noqa: E402
    LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA,
    Lfm2DynamicBlankMaterializationError,
    extract_lfm2_dynamic_physical_layout_from_config,
    lfm2_factorized_short_conv_evidence_state_key_mapping,
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2DynamicPhysicalLayoutError,
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_dynamic_physical_layout,
    build_lfm2_heterogeneous_physical_layout,
    lfm2_expected_heterogeneous_physical_state_shapes,
    lfm2_expected_physical_state_shapes,
)


def _descriptor():
    return build_lfm2_dynamic_physical_layout(
        hidden_size=8,
        vocab_size=16,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
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
        layer_types=("conv", "full_attention"),
        mlp_intermediate_sizes=(5, 6),
        short_conv_internal_widths=(10, None),
        parent_parameter_elements=2_000,
        requested_dense_parameter_budget=1_200,
    )


def _attention_only_descriptor():
    return build_lfm2_dynamic_physical_layout(
        hidden_size=8,
        vocab_size=16,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
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
        layer_types=("full_attention", "full_attention"),
        mlp_intermediate_sizes=(5, 6),
        short_conv_internal_widths=(None, None),
        parent_parameter_elements=2_000,
        requested_dense_parameter_budget=1_200,
    )


def _config(*, conv_kernel: int = 3, tied: bool = False):
    return transformers.Lfm2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=9,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=conv_kernel,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=tied,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _attention_only_config():
    return transformers.Lfm2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=9,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention", "full_attention"],
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


def _heterogeneous_descriptor():
    digest = "a" * 64
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
        source_topology_sha256=digest,
        allocation_evidence_sha256="b" * 64,
        token_bridge_evidence_sha256="c" * 64,
    )


def _heterogeneous_config():
    # The first attention layer is the deterministic native-construction
    # scaffold.  Materialization replaces both attention layers with their
    # descriptor-local geometries before any folded state is loaded.
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


def _model(config=None):
    return transformers.Lfm2ForCausalLM(config or _config())


def _materialize(model, descriptor=None):
    layout = descriptor or _descriptor()
    receipt = materialize_lfm2_dynamic_blank_child(
        model,
        model.config,
        layout,
        expected_descriptor_sha256=layout.descriptor_sha256,
    )
    return layout, receipt


def test_materializes_native_namespace_nonuniform_widths_and_dynamic_cache() -> None:
    torch.manual_seed(3901)
    model = _model()
    descriptor, receipt = _materialize(model)

    assert tuple(model.model.layers[0].feed_forward.w1.weight.shape) == (5, 8)
    assert tuple(model.model.layers[1].feed_forward.w2.weight.shape) == (8, 6)
    assert tuple(model.model.layers[0].conv.in_proj.weight.shape) == (30, 8)
    assert tuple(model.model.layers[0].conv.conv.weight.shape) == (10, 1, 3)
    assert tuple(model.model.layers[0].conv.out_proj.weight.shape) == (8, 10)
    assert type(model.model.layers[0].conv).__name__ == "Lfm2DynamicWidthShortConv"
    assert set(model.state_dict()) == set(
        lfm2_expected_physical_state_shapes(descriptor)
    )
    assert len(model.state_dict()) == 22
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_110
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.model.embed_tokens.weight is not model.lm_head.weight
    assert receipt["parameter_elements"] == 1_110
    assert receipt["parameter_state_key_count"] == 22
    assert len(receipt["receipt_sha256"]) == 64
    assert len(receipt["config_binding"]["normalized_config_sha256"]) == 64
    assert receipt["origin_authentication"] == {
        "caller_must_supply_fresh_local_native_blank": True,
        "empty_name_or_path_required": True,
        "origin_proven_by_materialization_receipt": False,
    }
    assert receipt["config_binding"]["runtime_semantics"] == {
        "norm_eps": 1e-5,
        "max_position_embeddings": 128_000,
        "use_cache": True,
        "attention_implementation": "sdpa",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "block_auto_adjust_ff_dim": False,
        "descriptor_sha256": descriptor.descriptor_sha256,
    }
    assert (
        receipt["physical_materialization_audit"][
            "intrinsic_nonpersistent_buffer_count"
        ]
        == 2
    )
    assert receipt["fresh_reload_contract"] == {
        "source": "caller-supplied-local-config-and-state-only",
        "hub_loader_allowed": False,
        "instantiate_native_lfm2_before_load": True,
        "extract_embedded_descriptor": True,
        "materialize_before_state_load": True,
        "strict_state_load_required": True,
        "fresh_dynamic_cache_required": True,
        "runtime_parent_or_sidecar_required": False,
    }
    assert model.config.neural_forge_dynamic_blank_config_schema == (
        LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA
    )
    assert extract_lfm2_dynamic_physical_layout_from_config(model.config) == descriptor

    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        cached = model(prefix, use_cache=True)
        continued = model(
            torch.tensor([[4]], dtype=torch.long),
            past_key_values=cached.past_key_values,
            use_cache=True,
        )
        full = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long), use_cache=False)
    assert tuple(cached.past_key_values.layers[0].conv_states.shape) == (1, 10, 3)
    assert tuple(continued.logits.shape) == (1, 1, 16)
    torch.testing.assert_close(
        continued.logits[:, -1],
        full.logits[:, -1],
        rtol=2e-5,
        atol=2e-6,
    )

    repeated = materialize_lfm2_dynamic_blank_child(
        model,
        model.config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    assert repeated == receipt

def test_direct_local_config_and_safetensors_roundtrip_is_strictly_equivalent(
    tmp_path: Path,
) -> None:
    torch.manual_seed(815)
    source = _model()
    descriptor, source_receipt = _materialize(source)
    source.eval()
    inputs = torch.tensor([[2, 5, 7, 3]], dtype=torch.long)
    with torch.inference_mode():
        expected_logits = source(inputs, use_cache=False).logits

    checkpoint = tmp_path / "child"
    source.save_pretrained(checkpoint, safe_serialization=True)
    payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    fresh_config = transformers.Lfm2Config(**payload)
    recovered = extract_lfm2_dynamic_physical_layout_from_config(fresh_config)
    assert recovered == descriptor

    fresh = transformers.Lfm2ForCausalLM(fresh_config)
    fresh_receipt = materialize_lfm2_dynamic_blank_child(
        fresh,
        fresh_config,
        recovered,
        expected_descriptor_sha256=recovered.descriptor_sha256,
    )
    state = safetensors_torch.load_file(str(checkpoint / "model.safetensors"))
    result = fresh.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    fresh.eval()
    with torch.inference_mode():
        actual_logits = fresh(inputs, use_cache=False).logits

    assert fresh_receipt == source_receipt
    assert tuple(fresh.state_dict()) == tuple(source.state_dict())
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)


def test_config_namespace_and_runtime_type_drift_fail_closed() -> None:
    descriptor = _descriptor()

    wrong_kernel = _model(_config(conv_kernel=4))
    with pytest.raises(
        Lfm2DynamicBlankMaterializationError,
        match="conv_L_cache",
    ):
        _materialize(wrong_kernel, descriptor)

    tied = _model(_config(tied=True))
    with pytest.raises(
        Lfm2DynamicBlankMaterializationError,
        match="untied",
    ):
        _materialize(tied, descriptor)

    sidecar = _model()
    sidecar.register_parameter(
        "unaccounted_sidecar",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32)),
    )
    with pytest.raises(
        Lfm2DynamicBlankMaterializationError,
        match="sidecar state keys",
    ):
        _materialize(sidecar, descriptor)

    materialized = _model()
    _materialize(materialized, descriptor)

    class LinearImpostor(torch.nn.Linear):
        pass

    impostor = LinearImpostor(8, 5, bias=False, dtype=torch.float32)
    impostor.weight = torch.nn.Parameter(
        materialized.model.layers[0].feed_forward.w1.weight.detach().clone(),
        requires_grad=False,
    )
    impostor.eval()
    materialized.model.layers[0].feed_forward.w1 = impostor
    with pytest.raises(
        Lfm2DynamicPhysicalLayoutError,
        match="module graph|linear geometry",
    ):
        _materialize(materialized, descriptor)


def test_stale_native_width_cache_is_rejected_after_materialization() -> None:
    model = _model()
    with torch.inference_mode():
        stale = model(torch.tensor([[1, 2, 3]], dtype=torch.long), use_cache=True)
    assert tuple(stale.past_key_values.layers[0].conv_states.shape) == (1, 8, 3)

    descriptor, _ = _materialize(model)
    assert descriptor.layers[0].short_conv_internal_width == 10
    with (
        pytest.raises(
            Lfm2DynamicBlankMaterializationError,
            match="freshly created|cache contract",
        ),
        torch.inference_mode(),
    ):
        model(
            torch.tensor([[4]], dtype=torch.long),
            past_key_values=stale.past_key_values,
            use_cache=True,
        )


@pytest.mark.parametrize(
    "mutation",
    ("training", "gradient", "route", "conv_kernel", "conv_bias", "norm_eps"),
)
def test_idempotent_materialization_rejects_runtime_semantic_drift(
    mutation: str,
) -> None:
    model = _model()
    descriptor, _ = _materialize(model)
    if mutation == "training":
        model.train()
    elif mutation == "gradient":
        model.lm_head.weight.requires_grad_(True)
    elif mutation == "route":
        model.model.layers[0].is_attention_layer = True
    elif mutation == "conv_kernel":
        conv = model.model.layers[0].conv
        attribute = "L_cache" if hasattr(conv, "L_cache") else "conv_kernel_size"
        setattr(conv, attribute, descriptor.conv_kernel_size + 1)
    elif mutation == "conv_bias":
        model.model.layers[0].conv.bias = True
    else:
        model.model.embedding_norm.variance_epsilon = 1e-4

    with pytest.raises(Lfm2DynamicPhysicalLayoutError):
        materialize_lfm2_dynamic_blank_child(
            model,
            model.config,
            descriptor,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )


def test_hidden_config_runtime_state_and_semantic_config_drift_fail_closed() -> None:
    hidden = _model()
    hidden.config.hidden_parent_tensor = torch.ones(4, dtype=torch.float32)
    with pytest.raises(Lfm2DynamicPhysicalLayoutError):
        _materialize(hidden)

    semantic = _model()
    descriptor, _ = _materialize(semantic)
    semantic.config.norm_eps = 1e-4
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="norm_eps"):
        materialize_lfm2_dynamic_blank_child(
            semantic,
            semantic.config,
            descriptor,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("is_causal", True),
        ("num_kv_shared_layers", 2),
        ("hidden_parent_weights", ["parent-layer-0", "parent-layer-1"]),
    ),
)
def test_json_safe_unsealed_config_fields_fail_closed(field: str, value) -> None:
    model = _model()
    setattr(model.config, field, value)
    with pytest.raises(
        Lfm2DynamicBlankMaterializationError,
        match="namespace contains missing or unsealed fields",
    ):
        _materialize(model)


@pytest.mark.parametrize("mutation", ("embedding", "generation"))
def test_idempotent_materialization_rejects_embedding_and_generation_drift(
    mutation: str,
) -> None:
    model = _model()
    descriptor, _ = _materialize(model)
    if mutation == "embedding":
        model.model.embed_tokens.max_norm = 1.0
        pattern = "embedding runtime geometry"
    else:
        model.generation_config.eos_token_id = 5
        pattern = "generation config drifted"
    with pytest.raises(Lfm2DynamicBlankMaterializationError, match=pattern):
        materialize_lfm2_dynamic_blank_child(
            model,
            model.config,
            descriptor,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )


def test_same_shape_cache_from_another_materialization_is_rejected() -> None:
    first = _model()
    second = _model()
    descriptor, _ = _materialize(first)
    _materialize(second, descriptor)
    with torch.inference_mode():
        cached = first(torch.tensor([[1, 2, 3]], dtype=torch.long), use_cache=True)
        with pytest.raises(
            Lfm2DynamicBlankMaterializationError,
            match="different physical materialization",
        ):
            second(
                torch.tensor([[4]], dtype=torch.long),
                past_key_values=cached.past_key_values,
                use_cache=True,
            )


def test_attention_cache_tensor_subclasses_fail_closed_before_decode() -> None:
    model = _model()
    _materialize(model)

    class EvilTensor(torch.Tensor):
        pass

    with torch.inference_mode():
        cached = model(torch.tensor([[1, 2, 3]], dtype=torch.long), use_cache=True)
        attention_cache = cached.past_key_values.layers[1]
        attention_cache.keys = torch.Tensor._make_subclass(
            EvilTensor,
            attention_cache.keys.detach().clone(),
            False,
        )
        with pytest.raises(
            Lfm2DynamicBlankMaterializationError,
            match="noncanonical tensor",
        ):
            model(
                torch.tensor([[4]], dtype=torch.long),
                past_key_values=cached.past_key_values,
                use_cache=True,
            )


def test_attention_only_cache_from_another_materialization_is_rejected() -> None:
    first = _model(_attention_only_config())
    second = _model(_attention_only_config())
    descriptor = _attention_only_descriptor()
    _materialize(first, descriptor)
    _materialize(second, descriptor)
    with torch.inference_mode():
        cached = first(torch.tensor([[1, 2, 3]], dtype=torch.long), use_cache=True)
        with pytest.raises(
            Lfm2DynamicBlankMaterializationError,
            match="different physical materialization",
        ):
            second(
                torch.tensor([[4]], dtype=torch.long),
                past_key_values=cached.past_key_values,
                use_cache=True,
            )


def test_cache_tensor_subclass_is_rejected_before_shortconv_use() -> None:
    class EvilTensor(torch.Tensor):
        pass

    model = _model()
    _materialize(model)
    with torch.inference_mode():
        cached = model(torch.tensor([[1, 2, 3]], dtype=torch.long), use_cache=True)
        layer = cached.past_key_values.layers[0]
        original = layer.conv_states.detach().clone()
        layer.conv_states = torch.Tensor._make_subclass(EvilTensor, original, False)
        with pytest.raises(
            Lfm2DynamicBlankMaterializationError,
            match="physical cache contract",
        ):
            model(
                torch.tensor([[4]], dtype=torch.long),
                past_key_values=cached.past_key_values,
                use_cache=True,
            )


def test_materializes_fully_heterogeneous_geometry_and_native_cache() -> None:
    torch.manual_seed(2107)
    descriptor = _heterogeneous_descriptor()
    model = _model(_heterogeneous_config())
    _, receipt = _materialize(model, descriptor)
    expected = lfm2_expected_heterogeneous_physical_state_shapes(descriptor)

    conv = model.model.layers[0].conv
    assert tuple(conv.in_proj.weight.shape) == (30, 8)
    assert type(conv.out_proj) is torch.nn.Sequential
    assert tuple(conv.out_proj[0].weight.shape) == (3, 10)
    assert tuple(conv.out_proj[1].weight.shape) == (8, 3)
    first_attention = model.model.layers[1].self_attn
    second_attention = model.model.layers[2].self_attn
    assert tuple(first_attention.q_proj.weight.shape) == (8, 8)
    assert tuple(first_attention.k_proj.weight.shape) == (4, 8)
    assert tuple(second_attention.q_proj.weight.shape) == (8, 8)
    assert tuple(second_attention.k_proj.weight.shape) == (4, 8)
    assert tuple(first_attention.q_layernorm.weight.shape) == (4,)
    assert tuple(second_attention.q_layernorm.weight.shape) == (2,)
    assert first_attention.num_key_value_groups == 2
    assert second_attention.num_key_value_groups == 2
    assert set(model.state_dict()) == set(expected)
    assert all(tuple(model.state_dict()[key].shape) == shape for key, shape in expected.items())
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        descriptor.accounting.unique_parameter_elements
    )
    assert receipt["parameter_elements"] == (
        descriptor.accounting.unique_parameter_elements
    )
    evidence_mapping = lfm2_factorized_short_conv_evidence_state_key_mapping(
        descriptor
    )
    assert dict(evidence_mapping) == {
        "model.layers.0.conv.out_encoder.weight": (
            "model.layers.0.conv.out_proj.0.weight"
        ),
        "model.layers.0.conv.out_decoder.weight": (
            "model.layers.0.conv.out_proj.1.weight"
        ),
    }
    assert receipt["factorized_short_conv_evidence_install"][
        "source_to_physical_state_key"
    ] == dict(evidence_mapping)
    assert receipt["factorized_short_conv_evidence_install"][
        "persistent_runtime_evidence_aliases"
    ] is False
    assert len(
        receipt["factorized_short_conv_evidence_install"]["mapping_sha256"]
    ) == 64

    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        cached = model(prefix, use_cache=True)
        continued = model(
            torch.tensor([[4]], dtype=torch.long),
            past_key_values=cached.past_key_values,
            use_cache=True,
        )
        full = model(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            use_cache=False,
        )
    assert tuple(cached.past_key_values.layers[0].conv_states.shape) == (1, 10, 3)
    assert tuple(cached.past_key_values.layers[1].keys.shape) == (1, 1, 4, 4)
    assert tuple(cached.past_key_values.layers[2].keys.shape) == (1, 2, 4, 2)
    torch.testing.assert_close(
        continued.logits[:, -1],
        full.logits[:, -1],
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_heterogeneous_cpu_rotary_reauthenticates_after_cuda_move() -> None:
    """A CPU-built default-RoPE buffer remains authentic after a CUDA move."""

    descriptor = build_lfm2_heterogeneous_physical_layout(
        hidden_size=64,
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
        layers=(
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
                query_heads=1,
                key_value_heads=1,
                head_dim=64,
                selection_evidence_sha256="1" * 64,
            ),
        ),
        source_topology_sha256="a" * 64,
        allocation_evidence_sha256="b" * 64,
        token_bridge_evidence_sha256="c" * 64,
    )
    config = transformers.Lfm2Config(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=9,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["conv", "full_attention"],
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
    model = transformers.Lfm2ForCausalLM(config)
    _, receipt = _materialize(model, descriptor)

    expected_cpu = 1.0 / (
        descriptor.rope_theta
        ** (
            torch.arange(0, 64, 2, dtype=torch.int64).to(dtype=torch.float32)
            / 64
        )
    )
    assert torch.equal(model.model.rotary_emb.inv_freq, expected_cpu)

    model.to(device="cuda")
    moved = model.model.rotary_emb.inv_freq
    expected_cuda = 1.0 / (
        descriptor.rope_theta
        ** (
            torch.arange(0, 64, 2, dtype=torch.int64, device="cuda").to(
                dtype=torch.float32
            )
            / 64
        )
    )
    assert torch.equal(moved.cpu(), expected_cpu)
    assert not torch.equal(moved, expected_cuda)

    repeated = materialize_lfm2_dynamic_blank_child(
        model,
        model.config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    assert repeated == receipt

    with torch.no_grad():
        moved[0] = torch.nextafter(moved[0], torch.zeros_like(moved[0]))
    with pytest.raises(
        Lfm2DynamicBlankMaterializationError,
        match="rotary buffer",
    ):
        materialize_lfm2_dynamic_blank_child(
            model,
            model.config,
            descriptor,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )


def test_heterogeneous_direct_local_safetensors_roundtrip_is_exact(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1709)
    descriptor = _heterogeneous_descriptor()
    source = _model(_heterogeneous_config())
    _, source_receipt = _materialize(source, descriptor)
    inputs = torch.tensor([[2, 5, 7, 3]], dtype=torch.long)
    with torch.inference_mode():
        expected_logits = source(inputs, use_cache=False).logits

    checkpoint = tmp_path / "heterogeneous-child"
    source.save_pretrained(checkpoint, safe_serialization=True)
    payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    fresh_config = transformers.Lfm2Config(**payload)
    recovered = extract_lfm2_dynamic_physical_layout_from_config(fresh_config)
    assert recovered == descriptor

    fresh = transformers.Lfm2ForCausalLM(fresh_config)
    fresh_receipt = materialize_lfm2_dynamic_blank_child(
        fresh,
        fresh_config,
        recovered,
        expected_descriptor_sha256=recovered.descriptor_sha256,
    )
    state = safetensors_torch.load_file(str(checkpoint / "model.safetensors"))
    result = fresh.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    with torch.inference_mode():
        actual_logits = fresh(inputs, use_cache=False).logits

    assert fresh_receipt == source_receipt
    assert tuple(fresh.state_dict()) == tuple(source.state_dict())
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)


def test_heterogeneous_factor_writer_and_descriptor_tampering_fail_closed() -> None:
    descriptor = _heterogeneous_descriptor()
    model = _model(_heterogeneous_config())
    _materialize(model, descriptor)
    model.model.layers[0].conv.out_proj[0] = torch.nn.Linear(
        10,
        4,
        bias=False,
        dtype=torch.float32,
    )
    model.model.layers[0].conv.out_proj[0].requires_grad_(False)
    model.model.layers[0].conv.out_proj[0].eval()
    with pytest.raises(Lfm2DynamicPhysicalLayoutError):
        materialize_lfm2_dynamic_blank_child(
            model,
            model.config,
            descriptor,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )

    fresh_config = _heterogeneous_config()
    setattr(
        fresh_config,
        "neural_forge_dynamic_blank_config_schema",
        LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA,
    )
    tampered = descriptor.as_dict()
    tampered["layers"][1]["head_dim"] = 6
    setattr(fresh_config, "neural_forge_dynamic_physical_layout", tampered)
    setattr(
        fresh_config,
        "neural_forge_dynamic_physical_layout_sha256",
        descriptor.descriptor_sha256,
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError):
        extract_lfm2_dynamic_physical_layout_from_config(fresh_config)


def test_heterogeneous_runtime_matches_native_when_physical_geometry_matches() -> None:
    layers = (
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=0,
            mlp_intermediate_size=9,
            internal_width=8,
            selection_evidence_sha256="3" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=1,
            mlp_intermediate_size=9,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            selection_evidence_sha256="4" * 64,
        ),
    )
    descriptor = build_lfm2_heterogeneous_physical_layout(
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
        source_topology_sha256="d" * 64,
        allocation_evidence_sha256="e" * 64,
        token_bridge_evidence_sha256="f" * 64,
    )
    torch.manual_seed(8181)
    reference = _model()
    target = _model()
    state = {
        name: value.detach().clone() for name, value in reference.state_dict().items()
    }
    _materialize(target, descriptor)
    result = target.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    reference.requires_grad_(False).eval()
    inputs = torch.tensor([[1, 5, 9, 2]], dtype=torch.long)
    with torch.inference_mode():
        expected = reference(inputs, use_cache=False).logits
        actual = target(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
