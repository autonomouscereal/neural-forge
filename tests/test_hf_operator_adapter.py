from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_arch_layout import build_decoder_layout  # noqa: E402
from neural_forge.hf_operator_adapter import (  # noqa: E402
    AttentionOperators,
    LinearOperator,
    OperatorAdapterError,
    read_attention_operators,
    read_mlp_operators,
    write_attention_operators,
    write_mlp_operators,
)


def _gpt2():
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=12,
            n_layer=1,
            n_head=3,
            n_inner=20,
            tie_word_embeddings=False,
        )
    )


def _llama():
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=16,
            tie_word_embeddings=False,
        )
    )


@pytest.mark.parametrize("model", [_gpt2(), _llama()])
def test_attention_and_mlp_roundtrip_preserve_every_state_tensor(model) -> None:
    layout = build_decoder_layout(model)
    block = layout.blocks[0]
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    attention = read_attention_operators(model, block.attention)
    mlp = read_mlp_operators(model, block)
    write_attention_operators(model, block.attention, attention)
    write_mlp_operators(model, block, mlp)

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_gpt2_fused_qkv_is_split_and_reassembled_in_canonical_order() -> None:
    model = _gpt2()
    layout = build_decoder_layout(model).blocks[0].attention
    fused = model.transformer.h[0].attn.c_attn
    with torch.no_grad():
        fused.weight.copy_(
            torch.arange(fused.weight.numel(), dtype=fused.weight.dtype).reshape_as(
                fused.weight
            )
        )
        fused.bias.copy_(torch.arange(fused.bias.numel(), dtype=fused.bias.dtype))

    operators = read_attention_operators(model, layout)

    canonical = fused.weight.detach().to(torch.float64).T
    assert torch.equal(operators.query.weight, canonical[:12])
    assert torch.equal(operators.key.weight, canonical[12:24])
    assert torch.equal(operators.value.weight, canonical[24:36])
    assert torch.equal(operators.query.bias, fused.bias[:12].to(torch.float64))
    write_attention_operators(model, layout, operators)
    assert torch.equal(fused.weight, canonical.T.to(fused.weight.dtype))


def test_llama_gqa_canonical_shapes_are_explicit() -> None:
    model = _llama()
    block = build_decoder_layout(model).blocks[0]

    attention = read_attention_operators(model, block.attention)
    mlp = read_mlp_operators(model, block)

    assert attention.query.weight.shape == (12, 12)
    assert attention.key.weight.shape == (4, 12)
    assert attention.value.weight.shape == (4, 12)
    assert attention.output.weight.shape == (12, 12)
    assert mlp.gate is not None and mlp.gate.weight.shape == (20, 12)
    assert mlp.up is not None and mlp.up.weight.shape == (20, 12)
    assert mlp.down.weight.shape == (12, 20)


def test_write_rejects_shape_and_bias_contract_mismatches() -> None:
    model = _llama()
    layout = build_decoder_layout(model).blocks[0].attention
    operators = read_attention_operators(model, layout)
    malformed = AttentionOperators(
        query=LinearOperator(
            operators.query.path,
            torch.ones((11, 12)),
            None,
        ),
        key=operators.key,
        value=operators.value,
        output=operators.output,
    )

    with pytest.raises(OperatorAdapterError, match="query weight must have shape"):
        write_attention_operators(model, layout, malformed)

    biased = AttentionOperators(
        query=LinearOperator(
            operators.query.path,
            operators.query.weight,
            torch.zeros(12),
        ),
        key=operators.key,
        value=operators.value,
        output=operators.output,
    )
    with pytest.raises(OperatorAdapterError, match="bias presence"):
        write_attention_operators(model, layout, biased)


def test_adapter_outputs_are_detached_and_writes_create_no_gradients() -> None:
    model = _gpt2()
    block = build_decoder_layout(model).blocks[0]
    attention = read_attention_operators(model, block.attention)
    mlp = read_mlp_operators(model, block)

    values = [
        attention.query.weight,
        attention.key.weight,
        attention.value.weight,
        attention.output.weight,
        mlp.input.weight if mlp.input is not None else mlp.gate.weight,
        mlp.down.weight,
    ]
    assert all(
        value.requires_grad is False and value.grad_fn is None for value in values
    )
    write_attention_operators(model, block.attention, attention)
    write_mlp_operators(model, block, mlp)
    assert all(parameter.grad is None for parameter in model.parameters())
