from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from neural_forge.hf_targeted_trace import collect_hf_targeted_trace
from neural_forge.hf_trace_pooling import pool_hf_transport_trace
from neural_forge.hf_transport_trace import (
    DecoderBlockTrace,
    TransportTraceError,
    collect_hf_transport_trace,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_gpt2(*, layers: int = 3):
    torch.manual_seed(101)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=48,
            n_positions=32,
            n_embd=16,
            n_layer=layers,
            n_head=4,
            n_inner=24,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _tiny_llama(*, layers: int = 3):
    torch.manual_seed(202)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=layers,
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
    )


def _tiny_qwen3(*, layers: int = 3):
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(303)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=48,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _batch():
    input_ids = torch.tensor([[1, 5, 7, 9, 11], [1, 4, 6, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.long)
    valid_mask = attention_mask.bool()
    valid_mask[0, 1] = False
    return input_ids, attention_mask, valid_mask


def _assert_block_equal(left: DecoderBlockTrace, right: DecoderBlockTrace) -> None:
    assert left.index == right.index
    for field in fields(DecoderBlockTrace):
        if field.name == "index":
            continue
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if left_value is None or right_value is None:
            assert left_value is right_value, field.name
        else:
            assert torch.equal(left_value, right_value), field.name


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


@pytest.mark.parametrize(
    ("factory", "family"),
    [
        (_tiny_gpt2, "gpt2"),
        (_tiny_llama, "llama"),
        (_tiny_qwen3, "qwen3"),
    ],
)
def test_selected_first_middle_last_match_full_trace_field_for_field(
    factory, family: str
) -> None:
    model = factory()
    input_ids, attention_mask, valid_mask = _batch()
    reference = collect_hf_transport_trace(
        model,
        input_ids,
        attention_mask,
        valid_token_mask=valid_mask,
    )

    for index in (0, 1, 2):
        targeted = collect_hf_targeted_trace(
            model,
            input_ids,
            attention_mask,
            block_index=index,
            valid_token_mask=valid_mask,
        )

        assert targeted.family == family
        assert targeted.blocks == (targeted.block,)
        assert targeted.mode == "sequence"
        assert torch.equal(targeted.valid_token_mask, valid_mask)
        assert targeted.valid_token_count == 7
        assert targeted.final_interface is None
        assert targeted.final_norm_output is None
        assert targeted.logits is None
        assert set(targeted.hook_counts.values()) == {1}
        assert targeted.captured_byte_count == sum(
            item.byte_count for item in targeted.capture_inventory.values()
        )
        assert all(
            name.startswith(f"block.{index}.") for name in targeted.capture_inventory
        )
        assert not any(
            name.startswith(f"block.{other}.")
            for name in targeted.capture_inventory
            for other in range(3)
            if other != index
        )
        _assert_block_equal(targeted.block, reference.blocks[index])

    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("factory", [_tiny_gpt2, _tiny_llama, _tiny_qwen3])
def test_final_norm_and_head_interface_matches_full_trace(factory) -> None:
    model = factory()
    input_ids, attention_mask, valid_mask = _batch()
    reference = collect_hf_transport_trace(
        model,
        input_ids,
        attention_mask,
        valid_token_mask=valid_mask,
    )
    targeted = collect_hf_targeted_trace(
        model,
        input_ids,
        attention_mask,
        block_index=2,
        valid_token_mask=valid_mask,
        capture_final_norm=True,
        capture_logits=True,
    )

    interface = targeted.final_interface
    assert interface is not None
    assert torch.equal(interface.final_norm_input, reference.blocks[-1].block_output)
    assert torch.equal(interface.final_norm_output, reference.final_norm_output)
    assert torch.equal(interface.lm_head_input, reference.final_norm_output)
    assert torch.equal(interface.logits, reference.logits)
    assert torch.equal(targeted.final_norm_output, reference.final_norm_output)
    assert torch.equal(targeted.logits, reference.logits)
    assert {"final_norm.input", "final_norm.output", "model.logits"}.issubset(
        targeted.capture_inventory
    )


@pytest.mark.parametrize("mode", ["flatten_valid", "mean", "last_valid"])
def test_reduction_modes_match_existing_trace_reductions(mode: str) -> None:
    model = _tiny_llama()
    input_ids, attention_mask, valid_mask = _batch()
    reference = collect_hf_transport_trace(
        model,
        input_ids,
        attention_mask,
        valid_token_mask=valid_mask,
    )
    targeted = collect_hf_targeted_trace(
        model,
        input_ids,
        attention_mask,
        block_index=1,
        valid_token_mask=valid_mask,
        mode=mode,
        capture_logits=True,
    )

    if mode == "flatten_valid":
        expected_block = reference.blocks[1].flatten_valid(valid_mask)
        expected_final = reference.flatten_valid_rows(reference.final_norm_output)
        expected_logits = reference.flatten_valid_rows(reference.logits)
        expected_rows = 7
    else:
        pooled = pool_hf_transport_trace(
            reference,
            pooling=mode,
            logits_policy=mode,
        )
        expected_block = pooled.blocks[1]
        expected_final = pooled.final_norm_output
        expected_logits = pooled.logits
        expected_rows = 2

    _assert_block_equal(targeted.block, expected_block)
    assert torch.equal(targeted.final_norm_output, expected_final)
    assert torch.equal(targeted.logits, expected_logits)
    assert targeted.block.block_input.shape == (expected_rows, 16)


def test_capture_bytes_and_inventory_do_not_scale_with_layer_count() -> None:
    input_ids, attention_mask, _valid_mask = _batch()
    shallow = _tiny_gpt2(layers=2)
    deep = _tiny_gpt2(layers=6)

    shallow_calls = 0
    deep_calls = 0

    def count_shallow(_module, _args, _output):
        nonlocal shallow_calls
        shallow_calls += 1

    def count_deep(_module, _args, _output):
        nonlocal deep_calls
        deep_calls += 1

    shallow_handle = shallow.register_forward_hook(count_shallow)
    deep_handle = deep.register_forward_hook(count_deep)
    try:
        shallow_trace = collect_hf_targeted_trace(
            shallow,
            input_ids,
            attention_mask,
            block_index=0,
        )
        deep_trace = collect_hf_targeted_trace(
            deep,
            input_ids,
            attention_mask,
            block_index=0,
        )
    finally:
        shallow_handle.remove()
        deep_handle.remove()

    assert shallow_calls == deep_calls == 1
    assert shallow_trace.captured_byte_count == deep_trace.captured_byte_count
    assert shallow_trace.capture_inventory == deep_trace.capture_inventory
    assert shallow_trace.captured_tensor_count == deep_trace.captured_tensor_count
    assert not any(name.startswith("block.1.") for name in deep_trace.capture_inventory)


def test_modes_hooks_and_gradients_are_restored_after_success_and_failure() -> None:
    model = _tiny_llama()
    input_ids, attention_mask, _valid_mask = _batch()
    model.train()
    model.model.layers[0].mlp.eval()
    model.model.layers[1].self_attn.q_proj.eval()
    original_modes = _module_modes(model)
    original_hooks = _hook_inventory(model)

    collect_hf_targeted_trace(
        model,
        input_ids,
        attention_mask,
        block_index=1,
    )
    assert _module_modes(model) == original_modes
    assert _hook_inventory(model) == original_hooks
    assert all(parameter.grad is None for parameter in model.parameters())

    def malformed_projection(_module, _args, output):
        return (output,)

    handle = model.model.layers[1].self_attn.q_proj.register_forward_hook(
        malformed_projection
    )
    injected_hooks = _hook_inventory(model)
    try:
        with pytest.raises(TransportTraceError, match="Malformed output"):
            collect_hf_targeted_trace(
                model,
                input_ids,
                attention_mask,
                block_index=1,
            )
        assert _module_modes(model) == original_modes
        assert _hook_inventory(model) == injected_hooks
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        handle.remove()
    assert _hook_inventory(model) == original_hooks


def test_fail_closed_inputs_masks_cache_outputs_and_indices() -> None:
    model = _tiny_gpt2()
    input_ids, attention_mask, valid_mask = _batch()

    with pytest.raises(IndexError, match="block_index"):
        collect_hf_targeted_trace(model, input_ids, attention_mask, block_index=3)
    with pytest.raises(TypeError, match="block_index"):
        collect_hf_targeted_trace(model, input_ids, attention_mask, block_index=True)
    with pytest.raises(ValueError, match="use_cache must be False"):
        collect_hf_targeted_trace(
            model,
            input_ids,
            attention_mask,
            block_index=0,
            use_cache=True,
        )
    with pytest.raises(ValueError, match="cannot select attention-masked"):
        invalid_mask = valid_mask.clone()
        invalid_mask[1, 4] = True
        collect_hf_targeted_trace(
            model,
            input_ids,
            attention_mask,
            block_index=0,
            valid_token_mask=invalid_mask,
        )

    model.train()
    original_modes = _module_modes(model)
    original_hooks = _hook_inventory(model)

    def add_cache(_module, _args, output):
        return SimpleNamespace(logits=output.logits, past_key_values=object())

    handle = model.register_forward_hook(add_cache)
    injected_hooks = _hook_inventory(model)
    try:
        with pytest.raises(TransportTraceError, match="returned a cache"):
            collect_hf_targeted_trace(
                model,
                input_ids,
                attention_mask,
                block_index=0,
            )
        assert _module_modes(model) == original_modes
        assert _hook_inventory(model) == injected_hooks
    finally:
        handle.remove()
    assert _hook_inventory(model) == original_hooks


def test_prompt_pooling_rejects_empty_valid_rows_before_forward() -> None:
    model = _tiny_llama()
    input_ids, attention_mask, _valid_mask = _batch()
    valid_mask = attention_mask.bool()
    valid_mask[1] = False
    calls = 0

    def count(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = model.register_forward_hook(count)
    try:
        with pytest.raises(ValueError, match="at least one valid token per sequence"):
            collect_hf_targeted_trace(
                model,
                input_ids,
                attention_mask,
                block_index=0,
                valid_token_mask=valid_mask,
                mode="mean",
            )
    finally:
        handle.remove()
    assert calls == 0
