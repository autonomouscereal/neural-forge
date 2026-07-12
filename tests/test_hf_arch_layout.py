from __future__ import annotations

from types import SimpleNamespace

import pytest

from neural_forge.hf_arch_layout import (
    UnsupportedDecoderLayoutError,
    build_decoder_layout,
    resolve_module,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_gpt2(*, tied: bool = True):
    config = transformers.GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=4,
        n_inner=24,
        activation_function="gelu_new",
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=tied,
    )
    return transformers.GPT2LMHeadModel(config)


def _tiny_llama(*, tied: bool = False):
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=tied,
    )
    return transformers.LlamaForCausalLM(config)


@pytest.mark.parametrize("tied", [False, True])
def test_gpt2_layout_is_complete_and_explicit(tied: bool) -> None:
    model = _tiny_gpt2(tied=tied)

    layout = build_decoder_layout(model)

    assert layout.family == "gpt2"
    assert layout.model_type == "gpt2"
    assert layout.hidden_size == 16
    assert layout.intermediate_size == 24
    assert layout.layer_count == 2
    assert layout.tie_word_embeddings is tied
    assert layout.embeddings.token_path == "transformer.wte"
    assert layout.embeddings.position_path == "transformer.wpe"
    assert layout.position.kind == "learned_absolute"
    assert layout.position.max_positions == 32
    assert layout.final_norm.kind == "layer_norm"
    assert layout.final_norm.placement == "final"

    block = layout.blocks[0]
    assert block.path == "transformer.h.0"
    assert block.attention_input_norm.path == "transformer.h.0.ln_1"
    assert block.attention_input_norm.kind == "layer_norm"
    assert block.attention.projection_style == "fused_qkv"
    assert block.attention.fused_qkv_path == "transformer.h.0.attn.c_attn"
    assert block.attention.query_path is None
    assert block.attention.query_heads == 4
    assert block.attention.key_value_heads == 4
    assert block.attention.head_dim == 4
    assert block.attention.grouped_query_attention is False
    assert block.attention.projection_weight_layout == "input_output"
    assert block.mlp.kind == "dense"
    assert block.mlp.input_projection_path == "transformer.h.0.mlp.c_fc"
    assert block.mlp.gate_projection_path is None
    assert block.mlp.up_projection_path is None
    assert block.mlp.down_projection_path == "transformer.h.0.mlp.c_proj"
    assert block.mlp.activation_name == "gelu_new"
    assert block.mlp.activation_module_type == "NewGELUActivation"
    assert block.residual.kind == "serial_pre_norm"
    assert block.residual.mlp_residual_source == "post_attention_residual"

    required = dict(layout.iter_operator_paths())
    assert required["block.1.attention.qkv"] == "transformer.h.1.attn.c_attn"
    assert required["block.1.mlp.activation"] == "transformer.h.1.mlp.act"
    for path in layout.required_operator_paths:
        assert resolve_module(model, path, family=layout.family) is not None


@pytest.mark.parametrize("tied", [False, True])
def test_llama_layout_captures_rope_gqa_and_gated_mlp(tied: bool) -> None:
    model = _tiny_llama(tied=tied)

    layout = build_decoder_layout(model)

    assert layout.family == "llama"
    assert layout.hidden_size == 16
    assert layout.intermediate_size == 24
    assert layout.layer_count == 2
    assert layout.tie_word_embeddings is tied
    assert layout.embeddings.token_path == "model.embed_tokens"
    assert layout.embeddings.position_path is None
    assert layout.position.kind == "rope"
    assert layout.position.operator_paths == ("model.rotary_emb",)
    assert layout.position.max_positions == 32
    assert layout.position.rotary_dim == 4
    assert layout.position.rope_theta == 10_000.0
    assert layout.position.rope_type == "default"
    assert layout.final_norm.kind == "rms_norm"

    block = layout.blocks[0]
    assert block.attention_input_norm.path == "model.layers.0.input_layernorm"
    assert block.mlp_input_norm.path == "model.layers.0.post_attention_layernorm"
    assert block.attention.projection_style == "split_qkv"
    assert block.attention.fused_qkv_path is None
    assert block.attention.query_path == "model.layers.0.self_attn.q_proj"
    assert block.attention.key_path == "model.layers.0.self_attn.k_proj"
    assert block.attention.value_path == "model.layers.0.self_attn.v_proj"
    assert block.attention.output_path == "model.layers.0.self_attn.o_proj"
    assert block.attention.query_heads == 4
    assert block.attention.key_value_heads == 2
    assert block.attention.grouped_query_attention is True
    assert block.attention.query_width == 16
    assert block.attention.key_value_width == 8
    assert block.attention.projection_weight_layout == "output_input"
    assert block.mlp.kind == "gated"
    assert block.mlp.input_projection_path is None
    assert block.mlp.gate_projection_path == "model.layers.0.mlp.gate_proj"
    assert block.mlp.up_projection_path == "model.layers.0.mlp.up_proj"
    assert block.mlp.down_projection_path == "model.layers.0.mlp.down_proj"
    assert block.mlp.activation_name == "silu"
    assert block.mlp.activation_module_type == "SiLUActivation"


def test_unsupported_model_type_fails_closed() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="bert"))

    with pytest.raises(UnsupportedDecoderLayoutError, match="supported model types"):
        build_decoder_layout(model)


def test_recognized_gpt2_missing_required_projection_fails_with_exact_path() -> None:
    model = _tiny_gpt2()
    del model.transformer.h[0].attn.c_attn

    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match=r"transformer\.h\.0\.attn\.c_attn.*missing",
    ):
        build_decoder_layout(model)


def test_recognized_llama_wrong_projection_shape_fails_closed() -> None:
    model = _tiny_llama()
    model.model.layers[0].self_attn.k_proj = torch.nn.Linear(16, 7, bias=False)

    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match=r"model\.layers\.0\.self_attn\.k_proj.*shape must be \(8, 16\)",
    ):
        build_decoder_layout(model)


def test_configured_activation_must_match_actual_operator() -> None:
    model = _tiny_gpt2()
    model.transformer.h[0].mlp.act = torch.nn.ReLU()

    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match=r"activation operator .* is ReLU, but configuration requires 'gelu_new'",
    ):
        build_decoder_layout(model)


def test_configured_and_observed_weight_tying_must_agree() -> None:
    model = _tiny_llama(tied=False)
    model.config.tie_word_embeddings = True

    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match="configured and observed input/output weight tying disagree",
    ):
        build_decoder_layout(model)


def test_gpt2_cross_attention_is_rejected_in_first_slice() -> None:
    model = _tiny_gpt2()
    model.config.add_cross_attention = True

    with pytest.raises(UnsupportedDecoderLayoutError, match="cross-attention"):
        build_decoder_layout(model)
