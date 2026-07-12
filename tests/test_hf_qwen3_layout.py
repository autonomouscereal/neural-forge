from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

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
from neural_forge.hf_transport_manifest import (  # noqa: E402
    build_decoder_transport_manifest,
)
from neural_forge.hf_transport_trace import (  # noqa: E402
    TransportTraceError,
    collect_hf_transport_trace,
)


def _tiny_qwen3(*, tied: bool = False, **overrides):
    values = {
        "vocab_size": 48,
        "hidden_size": 12,
        "intermediate_size": 20,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_word_embeddings": tied,
    }
    values.update(overrides)
    return transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**values))


@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_layout_is_exact_and_captures_real_dense_topology(tied: bool) -> None:
    model = _tiny_qwen3(tied=tied)

    layout = build_decoder_layout(model)

    assert layout.family == "qwen3"
    assert layout.model_type == "qwen3"
    assert layout.model_class == "Qwen3ForCausalLM"
    assert layout.hidden_size == 12
    assert layout.intermediate_size == 20
    assert layout.layer_count == 2
    assert layout.tie_word_embeddings is tied
    assert layout.position.kind == "rope"
    assert layout.position.operator_paths == ("model.rotary_emb",)
    assert layout.position.rotary_dim == 4
    assert layout.position.rope_theta == 10_000.0
    assert layout.position.rope_type == "default"

    block = layout.blocks[0]
    attention = block.attention
    assert attention.projection_style == "split_qkv"
    assert attention.query_heads == 4
    assert attention.key_value_heads == 2
    assert attention.grouped_query_attention is True
    assert attention.query_width == 16
    assert attention.key_value_width == 8
    assert attention.hidden_size == 12
    assert attention.query_norm is not None
    assert attention.query_norm.path == "model.layers.0.self_attn.q_norm"
    assert attention.query_norm.placement == "query_head"
    assert attention.query_norm.epsilon == model.config.rms_norm_eps
    assert attention.key_norm is not None
    assert attention.key_norm.path == "model.layers.0.self_attn.k_norm"
    assert attention.key_norm.placement == "key_head"
    assert block.attention_input_norm.kind == "rms_norm"
    assert block.mlp.kind == "gated"
    assert block.mlp.activation_name == "silu"
    assert block.mlp.gate_projection_path == "model.layers.0.mlp.gate_proj"
    assert block.mlp.up_projection_path == "model.layers.0.mlp.up_proj"
    assert block.mlp.down_projection_path == "model.layers.0.mlp.down_proj"

    required = dict(layout.iter_operator_paths())
    assert required["block.1.attention.q_norm"] == (
        "model.layers.1.self_attn.q_norm"
    )
    assert required["block.1.attention.k_norm"] == (
        "model.layers.1.self_attn.k_norm"
    )
    for path in layout.required_operator_paths:
        assert resolve_module(model, path, family=layout.family) is not None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_key_value_heads": 1}, "grouped-query attention"),
        ({"num_key_value_heads": 4}, "grouped-query attention"),
        ({"hidden_act": "relu"}, "hidden_act='silu'"),
        ({"attention_bias": True}, "bias-free attention"),
        (
            {"use_sliding_window": True, "max_window_layers": 0},
            "sliding-window Qwen3 variants",
        ),
    ],
)
def test_qwen3_rejects_incompatible_config_variants(
    overrides: dict[str, object], message: str
) -> None:
    model = _tiny_qwen3(**overrides)

    with pytest.raises(UnsupportedDecoderLayoutError, match=message):
        build_decoder_layout(model)


def test_qwen3_requires_exact_hugging_face_wrapper_and_head_norms() -> None:
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
    impostor.config.model_type = "qwen3"
    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match=r"model must be exactly .*Qwen3ForCausalLM",
    ):
        build_decoder_layout(impostor)

    model = _tiny_qwen3()
    model.model.layers[0].self_attn.q_norm = torch.nn.Identity()
    with pytest.raises(
        UnsupportedDecoderLayoutError,
        match=r"q_norm.*must be exactly .*Qwen3RMSNorm",
    ):
        build_decoder_layout(model)


def test_qwen3_trace_validates_gqa_widths_swiglu_and_residuals() -> None:
    torch.manual_seed(303)
    model = _tiny_qwen3()
    input_ids = torch.tensor([[1, 3, 5, 7], [1, 4, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    trace = collect_hf_transport_trace(model, input_ids, attention_mask)

    assert trace.family == "qwen3"
    assert trace.valid_token_count == 6
    assert set(trace.hook_counts.values()) == {1}
    for block in trace.blocks:
        assert block.fused_qkv_projection_output is None
        assert block.query_projection_output.shape == (2, 4, 16)
        assert block.key_projection_output.shape == (2, 4, 8)
        assert block.value_projection_output.shape == (2, 4, 8)
        assert block.attention_output_projection_input.shape == (2, 4, 16)
        assert block.mlp_gate_projection_output.shape == (2, 4, 20)
        assert block.mlp_up_projection_output.shape == (2, 4, 20)
        torch.testing.assert_close(
            block.down_projection_input,
            block.activation_output * block.mlp_up_projection_output,
        )
        torch.testing.assert_close(
            block.post_attention_residual,
            block.block_input + block.attention_delta,
        )
        torch.testing.assert_close(
            block.block_output,
            block.post_attention_residual + block.mlp_delta,
        )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_qwen3_trace_rejects_malformed_runtime_projection_width() -> None:
    model = _tiny_qwen3()
    input_ids = torch.tensor([[1, 3, 5]], dtype=torch.long)

    def truncate_query(_module, _args, output):
        return output[..., :-1]

    handle = model.model.layers[0].self_attn.q_proj.register_forward_hook(
        truncate_query
    )
    try:
        with pytest.raises(
            TransportTraceError,
            match=r"block\.0\.attention\.q.*expected \[batch, sequence, 16\]",
        ):
            collect_hf_transport_trace(model, input_ids)
    finally:
        handle.remove()


def test_qwen3_canonical_operator_roundtrip_preserves_all_state() -> None:
    model = _tiny_qwen3()
    block = build_decoder_layout(model).blocks[0]
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
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


def test_qwen3_manifest_claims_every_parameter_and_buffer() -> None:
    manifest = build_decoder_transport_manifest(_tiny_qwen3())
    roles = manifest.plan.roles_by_name

    assert manifest.unclaimed_parameter_tensors == ()
    assert manifest.unclaimed_buffer_tensors == ()
    assert set(manifest.parameter_tensors) <= set(manifest.claimed_tensors)
    assert set(manifest.buffer_tensors) <= set(manifest.claimed_tensors)
    assert roles["block.0.attention.query_norm"].target_tensors == (
        "model.model.layers.0.self_attn.q_norm.weight",
    )
    assert roles["block.0.attention.key_norm"].target_tensors == (
        "model.model.layers.0.self_attn.k_norm.weight",
    )
    assert set(roles["position.rope"].target_tensors) == set(
        manifest.buffer_tensors
    )
