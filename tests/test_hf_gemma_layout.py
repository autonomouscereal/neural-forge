from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not all(
    hasattr(transformers, name)
    for name in (
        "Gemma2Config",
        "Gemma2ForCausalLM",
        "Gemma3TextConfig",
        "Gemma3ForCausalLM",
    )
):
    pytest.skip(
        "Transformers build has no current Gemma2/Gemma3 text support",
        allow_module_level=True,
    )

from neural_forge.hf_arch_layout import (  # noqa: E402
    UnsupportedDecoderLayoutError,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_operator_adapter import (  # noqa: E402
    read_attention_operators,
    read_mlp_operators,
    write_attention_operators,
    write_mlp_operators,
)
from neural_forge.hf_targeted_trace import (  # noqa: E402
    collect_hf_targeted_trace,
)
from neural_forge.hf_transport_manifest import (  # noqa: E402
    build_decoder_transport_manifest,
)
from neural_forge.hf_transport_trace import (  # noqa: E402
    collect_hf_transport_trace,
)


def _tiny_gemma(
    family: str,
    *,
    tied: bool = False,
    layers: int | None = None,
    **overrides,
):
    values = {
        "vocab_size": 48,
        "hidden_size": 12,
        "intermediate_size": 20,
        "num_hidden_layers": layers or 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "sliding_window": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_word_embeddings": tied,
    }
    values.update(overrides)
    if family == "gemma2":
        config = transformers.Gemma2Config(**values)
        return transformers.Gemma2ForCausalLM(config)
    if family == "gemma3":
        config = transformers.Gemma3TextConfig(**values)
        return transformers.Gemma3ForCausalLM(config)
    raise AssertionError(f"unknown test family {family!r}")


@pytest.mark.parametrize("family", ["gemma2", "gemma3"])
@pytest.mark.parametrize("tied", [False, True])
def test_gemma_layout_captures_exact_decoder_contract(
    family: str, tied: bool
) -> None:
    layer_count = 2 if family == "gemma2" else 6
    model = _tiny_gemma(family, tied=tied, layers=layer_count)

    layout = build_decoder_layout(model)

    assert layout.family == family
    assert layout.model_type == ("gemma2" if family == "gemma2" else "gemma3_text")
    assert layout.model_class == (
        "Gemma2ForCausalLM" if family == "gemma2" else "Gemma3ForCausalLM"
    )
    assert layout.layer_count == layer_count
    assert layout.tie_word_embeddings is tied
    assert layout.embeddings.token_path == "model.embed_tokens"
    assert layout.embeddings.token_scale == pytest.approx(math.sqrt(12))
    assert (
        layout.embeddings.token_scale_buffer_name
        == "model.embed_tokens.embed_scale"
    )
    assert layout.position.kind == "rope"
    assert layout.position.operator_paths == ("model.rotary_emb",)
    assert layout.position.rotary_dim == 4
    assert layout.position.rope_type == "default"
    if family == "gemma2":
        assert layout.position.rope_variants == (("all", 10_000.0, "default"),)
        assert layout.final_logit_softcap == 30.0
        assert [block.attention.layer_type for block in layout.blocks] == [
            "sliding_attention",
            "full_attention",
        ]
    else:
        assert set(layout.position.rope_variants) == {
            ("sliding_attention", 10_000.0, "default"),
            ("full_attention", 1_000_000.0, "default"),
        }
        assert layout.final_logit_softcap is None
        assert layout.blocks[5].attention.layer_type == "full_attention"

    block = layout.blocks[0]
    assert block.residual.kind == "serial_pre_post_norm"
    assert block.attention.projection_style == "split_qkv"
    assert block.attention.query_heads == 4
    assert block.attention.key_value_heads == 2
    assert block.attention.query_width == 16
    assert block.attention.key_value_width == 8
    assert block.attention.sliding_window == 8
    assert block.attention.scaling == 1 / 16
    assert block.attention_input_norm.weight_kind == "unit_offset"
    assert block.attention_output_norm is not None
    assert block.attention_output_norm.path.endswith("post_attention_layernorm")
    assert block.attention_output_norm.weight_kind == "unit_offset"
    assert block.mlp_input_norm.path.endswith("pre_feedforward_layernorm")
    assert block.mlp_output_norm is not None
    assert block.mlp_output_norm.path.endswith("post_feedforward_layernorm")
    assert block.mlp.kind == "gated"
    assert block.mlp.activation_name == "gelu_pytorch_tanh"
    if family == "gemma2":
        assert block.attention.query_norm is None
        assert block.attention.key_norm is None
        assert block.attention.logit_softcap == 50.0
    else:
        assert block.attention.query_norm is not None
        assert block.attention.query_norm.weight_kind == "unit_offset"
        assert block.attention.key_norm is not None
        assert block.attention.key_norm.weight_kind == "unit_offset"
        assert block.attention.logit_softcap is None

    required = dict(layout.iter_operator_paths())
    assert required["block.0.attention.output_norm"].endswith(
        "post_attention_layernorm"
    )
    assert required["block.0.mlp.output_norm"].endswith(
        "post_feedforward_layernorm"
    )
    for path in layout.required_operator_paths:
        assert resolve_module(model, path, family=family) is not None


@pytest.mark.parametrize("family", ["gemma2", "gemma3"])
def test_gemma_requires_exact_wrapper_norms_and_complete_state(family: str) -> None:
    impostor = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=48,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    impostor.config.model_type = "gemma2" if family == "gemma2" else "gemma3_text"
    expected_name = "Gemma2ForCausalLM" if family == "gemma2" else "Gemma3ForCausalLM"
    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match=rf"model must be exactly .*{expected_name}",
    ):
        build_decoder_layout(impostor)

    model = _tiny_gemma(family)
    model.model.layers[0].post_feedforward_layernorm = torch.nn.Identity()
    expected_norm = "Gemma2RMSNorm" if family == "gemma2" else "Gemma3RMSNorm"
    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match=rf"post_feedforward_layernorm.*{expected_norm}",
    ):
        build_decoder_layout(model)

    model = _tiny_gemma(family)
    model.register_buffer("unexpected_runtime_state", torch.ones(1))
    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match="explicit operator paths do not claim exactly all buffers",
    ):
        build_decoder_layout(model)


@pytest.mark.parametrize("family", ["gemma2", "gemma3"])
def test_gemma_rejects_noncausal_or_malformed_layer_policy(family: str) -> None:
    model = _tiny_gemma(family)
    model.config.use_bidirectional_attention = True
    with pytest.raises(UnsupportedDecoderLayoutError, match="bidirectional attention"):
        build_decoder_layout(model)

    model = _tiny_gemma(family)
    model.config.layer_types = ["unsupported"] * model.config.num_hidden_layers
    with pytest.raises(UnsupportedDecoderLayoutError, match="layer_types must name"):
        build_decoder_layout(model)


@pytest.mark.parametrize("family", ["gemma2", "gemma3"])
def test_gemma_canonical_operator_roundtrip_preserves_all_state(family: str) -> None:
    model = _tiny_gemma(family)
    block = build_decoder_layout(model).blocks[0]
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    before_buffers = {
        name: value.detach().clone()
        for name, value in model.named_buffers(remove_duplicate=False)
    }

    attention = read_attention_operators(model, block.attention)
    mlp = read_mlp_operators(model, block)
    assert attention.query.weight.shape == (16, 12)
    assert attention.key.weight.shape == (8, 12)
    assert attention.value.weight.shape == (8, 12)
    assert attention.output.weight.shape == (12, 16)
    assert mlp.gate is not None and mlp.gate.weight.shape == (20, 12)
    assert mlp.up is not None and mlp.up.weight.shape == (20, 12)
    assert mlp.down.weight.shape == (12, 20)

    write_attention_operators(model, block.attention, attention)
    write_mlp_operators(model, block, mlp)

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    for name, value in model.named_buffers(remove_duplicate=False):
        assert torch.equal(value, before_buffers[name])


@pytest.mark.parametrize("family", ["gemma2", "gemma3"])
def test_gemma_full_and_targeted_traces_validate_post_norm_residuals(
    family: str,
) -> None:
    torch.manual_seed(713)
    model = _tiny_gemma(family)
    input_ids = torch.tensor([[1, 3, 5, 7], [1, 4, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    trace = collect_hf_transport_trace(model, input_ids, attention_mask)
    targeted = collect_hf_targeted_trace(
        model,
        input_ids,
        attention_mask,
        block_index=0,
        capture_logits=True,
    )

    assert trace.family == family
    assert trace.valid_token_count == 6
    assert set(trace.hook_counts.values()) == {1}
    for index, block in enumerate(trace.blocks):
        torch.testing.assert_close(
            block.attention_delta,
            model.model.layers[index].post_attention_layernorm(
                block.attention_output_projection_output
            ),
        )
        torch.testing.assert_close(
            block.mlp_delta,
            model.model.layers[index].post_feedforward_layernorm(
                block.down_projection_output
            ),
        )
        torch.testing.assert_close(
            block.post_attention_residual,
            block.block_input + block.attention_delta,
        )
        torch.testing.assert_close(
            block.block_output,
            block.post_attention_residual + block.mlp_delta,
        )

    torch.testing.assert_close(
        targeted.block.attention_delta,
        model.model.layers[0].post_attention_layernorm(
            targeted.block.attention_output_projection_output
        ),
    )
    torch.testing.assert_close(
        targeted.block.mlp_delta,
        model.model.layers[0].post_feedforward_layernorm(
            targeted.block.down_projection_output
        ),
    )
    assert targeted.final_interface is not None
    raw_logits = model.lm_head(targeted.final_interface.final_norm_output)
    softcap = targeted.layout.final_logit_softcap
    expected_logits = (
        raw_logits if softcap is None else torch.tanh(raw_logits / softcap) * softcap
    )
    torch.testing.assert_close(targeted.final_interface.logits, expected_logits)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("family", ["gemma2", "gemma3"])
def test_gemma_manifest_claims_every_parameter_and_buffer_explicitly(
    family: str,
) -> None:
    manifest = build_decoder_transport_manifest(_tiny_gemma(family))
    roles = manifest.plan.roles_by_name

    assert manifest.unclaimed_parameter_tensors == ()
    assert manifest.unclaimed_buffer_tensors == ()
    assert set(manifest.parameter_tensors) <= set(manifest.claimed_tensors)
    assert set(manifest.buffer_tensors) <= set(manifest.claimed_tensors)
    assert roles["embeddings.token"].target_tensors == (
        "model.model.embed_tokens.weight",
    )
    assert roles["embeddings.token_scale"].target_tensors == (
        "model.model.embed_tokens.embed_scale",
    )
    assert roles["block.0.attention.output_norm"].target_tensors == (
        "model.model.layers.0.post_attention_layernorm.weight",
    )
    assert roles["block.0.mlp.output_norm"].target_tensors == (
        "model.model.layers.0.post_feedforward_layernorm.weight",
    )
    assert "runtime.derived_buffers" not in roles
