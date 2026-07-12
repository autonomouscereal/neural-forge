from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from neural_forge.hf_transport_trace import (
    DecoderBlockTrace,
    TransportTraceError,
    collect_hf_transport_trace,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_gpt2():
    torch.manual_seed(101)
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
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    return transformers.GPT2LMHeadModel(config)


def _tiny_llama():
    torch.manual_seed(202)
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
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    return transformers.LlamaForCausalLM(config)


def _batch():
    input_ids = torch.tensor(
        [
            [1, 5, 7, 9, 11],
            [1, 4, 6, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    return input_ids, attention_mask


def _assert_residual_identities(block: DecoderBlockTrace) -> None:
    torch.testing.assert_close(
        block.post_attention_residual,
        block.block_input + block.attention_delta,
    )
    torch.testing.assert_close(
        block.block_output,
        block.post_attention_residual + block.mlp_delta,
    )
    torch.testing.assert_close(
        block.block_output,
        block.block_input + block.attention_delta + block.mlp_delta,
    )
    torch.testing.assert_close(
        block.attention_delta,
        block.attention_output_projection_output,
    )
    torch.testing.assert_close(block.mlp_delta, block.down_projection_output)


def _module_modes(model) -> tuple[bool, ...]:
    return tuple(bool(module.training) for module in model.modules())


def _hook_inventory(model) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            len(module._forward_pre_hooks),
            len(module._forward_hooks),
            len(module._backward_hooks),
        )
        for module in model.modules()
    )


def _assert_no_parameter_gradients(model) -> None:
    assert all(parameter.grad is None for parameter in model.parameters())


def _assert_same_trace(left, right) -> None:
    assert left.family == right.family
    assert dict(left.hook_counts) == dict(right.hook_counts)
    assert torch.equal(left.valid_token_mask, right.valid_token_mask)
    assert torch.equal(left.final_norm_output, right.final_norm_output)
    assert torch.equal(left.logits, right.logits)
    for left_block, right_block in zip(left.blocks, right.blocks, strict=True):
        assert left_block.index == right_block.index
        for field in fields(DecoderBlockTrace):
            if field.name == "index":
                continue
            left_value = getattr(left_block, field.name)
            right_value = getattr(right_block, field.name)
            if left_value is None or right_value is None:
                assert left_value is right_value
            else:
                assert torch.equal(left_value, right_value), field.name


def test_gpt2_trace_splits_fused_qkv_and_preserves_residuals_and_mask() -> None:
    model = _tiny_gpt2()
    input_ids, attention_mask = _batch()
    valid_mask = attention_mask.bool()
    valid_mask[0, 1] = False
    model.train()
    model.transformer.h[0].attn.attn_dropout.eval()
    original_modes = _module_modes(model)
    original_hooks = _hook_inventory(model)

    trace = collect_hf_transport_trace(
        model,
        {"input_ids": input_ids, "attention_mask": attention_mask},
        valid_token_mask=valid_mask,
    )

    assert trace.family == "gpt2"
    assert trace.valid_token_mask.dtype == torch.bool
    assert trace.valid_token_mask.shape == (2, 5)
    assert torch.equal(trace.valid_token_mask, valid_mask)
    assert trace.valid_token_count == 7
    assert len(trace.blocks) == 2
    assert set(trace.hook_counts.values()) == {1}
    assert _module_modes(model) == original_modes
    assert _hook_inventory(model) == original_hooks
    _assert_no_parameter_gradients(model)

    for block in trace.blocks:
        assert block.block_input.shape == (2, 5, 16)
        assert block.pre_attention_norm_output.shape == (2, 5, 16)
        assert block.fused_qkv_projection_output.shape == (2, 5, 48)
        assert block.query_projection_output.shape == (2, 5, 16)
        assert block.key_projection_output.shape == (2, 5, 16)
        assert block.value_projection_output.shape == (2, 5, 16)
        torch.testing.assert_close(
            block.fused_qkv_projection_output,
            torch.cat(
                (
                    block.query_projection_output,
                    block.key_projection_output,
                    block.value_projection_output,
                ),
                dim=-1,
            ),
        )
        assert block.attention_output_projection_input.shape == (2, 5, 16)
        assert block.mlp_input_projection_output.shape == (2, 5, 24)
        assert block.mlp_gate_projection_output is None
        assert block.mlp_up_projection_output is None
        assert block.activation_output.shape == (2, 5, 24)
        torch.testing.assert_close(block.down_projection_input, block.activation_output)
        _assert_residual_identities(block)

    assert trace.final_norm_output.shape == (2, 5, 16)
    assert trace.logits.shape == (2, 5, 64)
    flattened = trace.flattened_valid_rows()
    assert flattened.source_batch_sequence_shape == (2, 5)
    assert flattened.valid_token_count == 7
    assert flattened.blocks[0].block_input.shape == (7, 16)
    assert flattened.logits.shape == (7, 64)
    expected_rows = trace.blocks[0].block_input.reshape(-1, 16)[valid_mask.reshape(-1)]
    assert torch.equal(flattened.blocks[0].block_input, expected_rows)


def test_llama_trace_preserves_gqa_and_gated_mlp_shapes() -> None:
    model = _tiny_llama()
    input_ids, attention_mask = _batch()

    trace = collect_hf_transport_trace(model, input_ids, attention_mask)

    assert trace.family == "llama"
    assert trace.valid_token_count == 8
    assert trace.layout.blocks[0].attention.grouped_query_attention is True
    assert set(trace.hook_counts.values()) == {1}
    _assert_no_parameter_gradients(model)
    for block in trace.blocks:
        assert block.fused_qkv_projection_output is None
        assert block.query_projection_output.shape == (2, 5, 16)
        assert block.key_projection_output.shape == (2, 5, 8)
        assert block.value_projection_output.shape == (2, 5, 8)
        assert block.attention_output_projection_input.shape == (2, 5, 16)
        assert block.mlp_input_projection_output is None
        assert block.mlp_gate_projection_output.shape == (2, 5, 24)
        assert block.mlp_up_projection_output.shape == (2, 5, 24)
        assert block.activation_output.shape == (2, 5, 24)
        torch.testing.assert_close(
            block.down_projection_input,
            block.activation_output * block.mlp_up_projection_output,
        )
        _assert_residual_identities(block)


@pytest.mark.parametrize("model_factory", [_tiny_gpt2, _tiny_llama])
def test_repeated_traces_are_bitwise_deterministic_and_gradient_free(
    model_factory,
) -> None:
    model = model_factory()
    input_ids, attention_mask = _batch()

    first = collect_hf_transport_trace(model, input_ids, attention_mask)
    second = collect_hf_transport_trace(model, input_ids, attention_mask)

    _assert_same_trace(first, second)
    _assert_no_parameter_gradients(model)


def test_trace_rejects_checkpointing_and_cache_inputs() -> None:
    model = _tiny_llama()
    input_ids, attention_mask = _batch()

    with pytest.raises(ValueError, match="use_cache must be False"):
        collect_hf_transport_trace(model, input_ids, attention_mask, use_cache=True)
    with pytest.raises(ValueError, match="Cached decoding input 'past_key_values'"):
        collect_hf_transport_trace(
            model,
            input_ids,
            attention_mask,
            past_key_values=object(),
        )

    model.gradient_checkpointing_enable()
    with pytest.raises(TransportTraceError, match="Gradient checkpointing"):
        collect_hf_transport_trace(model, input_ids, attention_mask)


def test_trace_rejects_cached_or_malformed_model_outputs_and_cleans_up() -> None:
    model = _tiny_gpt2()
    input_ids, attention_mask = _batch()
    model.train()
    original_modes = _module_modes(model)
    original_hooks = _hook_inventory(model)

    def add_cache(_module, _args, output):
        return SimpleNamespace(logits=output.logits, past_key_values=object())

    handle = model.register_forward_hook(add_cache)
    hooks_with_injection = _hook_inventory(model)
    try:
        with pytest.raises(TransportTraceError, match="returned a cache"):
            collect_hf_transport_trace(model, input_ids, attention_mask)
        assert _module_modes(model) == original_modes
        assert _hook_inventory(model) == hooks_with_injection
    finally:
        handle.remove()
    assert _hook_inventory(model) == original_hooks

    projection = model.transformer.h[0].attn.c_attn

    def malformed_projection(_module, _args, output):
        return (output,)

    handle = projection.register_forward_hook(malformed_projection)
    hooks_with_injection = _hook_inventory(model)
    try:
        with pytest.raises(TransportTraceError, match="Malformed output"):
            collect_hf_transport_trace(model, input_ids, attention_mask)
        assert _module_modes(model) == original_modes
        assert _hook_inventory(model) == hooks_with_injection
    finally:
        handle.remove()
    assert _hook_inventory(model) == original_hooks


def test_trace_restores_exact_modes_and_hooks_after_injected_exception() -> None:
    model = _tiny_llama()
    input_ids, attention_mask = _batch()
    model.train()
    model.model.layers[0].mlp.eval()
    model.model.layers[1].self_attn.q_proj.eval()
    original_modes = _module_modes(model)
    original_hooks = _hook_inventory(model)

    def fail(_module, _args, _output):
        raise RuntimeError("injected trace failure")

    handle = model.model.layers[1].self_attn.o_proj.register_forward_hook(fail)
    hooks_with_injection = _hook_inventory(model)
    try:
        with pytest.raises(RuntimeError, match="injected trace failure"):
            collect_hf_transport_trace(model, input_ids, attention_mask)
        assert _module_modes(model) == original_modes
        assert _hook_inventory(model) == hooks_with_injection
        _assert_no_parameter_gradients(model)
    finally:
        handle.remove()
    assert _hook_inventory(model) == original_hooks
