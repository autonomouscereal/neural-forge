from __future__ import annotations

from dataclasses import fields, replace
from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_arch_layout import build_decoder_layout  # noqa: E402
from neural_forge.hf_trace_pooling import (  # noqa: E402
    TracePoolingError,
    pool_hf_transport_trace,
    pool_row_aligned_hf_transport_traces,
)
from neural_forge.hf_transport_trace import (  # noqa: E402
    DecoderBlockTrace,
    HFTransportTrace,
)


def _tiny_gpt2():
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=11,
            n_positions=16,
            n_embd=6,
            n_layer=1,
            n_head=2,
            n_inner=8,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def _tiny_llama():
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=13,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            hidden_act="silu",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def _values(batch: int, sequence: int, width: int, offset: int):
    values = torch.arange(
        batch * sequence * width,
        dtype=torch.float64,
    ).reshape(batch, sequence, width)
    return values + float(offset * 1000)


def _synthetic_trace(model, mask, *, offset: int) -> HFTransportTrace:
    layout = build_decoder_layout(model)
    batch, sequence = (int(size) for size in mask.shape)
    blocks = []
    for block_layout in layout.blocks:
        hidden = layout.hidden_size
        intermediate = layout.intermediate_size
        query = block_layout.attention.query_width
        key_value = block_layout.attention.key_value_width
        next_offset = offset * 100

        def tensor(width: int):
            nonlocal next_offset
            next_offset += 1
            return _values(batch, sequence, width, next_offset)

        blocks.append(
            DecoderBlockTrace(
                index=block_layout.index,
                block_input=tensor(hidden),
                pre_attention_norm_output=tensor(hidden),
                fused_qkv_projection_output=(
                    tensor(query + 2 * key_value)
                    if block_layout.attention.fused_qkv_path is not None
                    else None
                ),
                query_projection_output=tensor(query),
                key_projection_output=tensor(key_value),
                value_projection_output=tensor(key_value),
                attention_output_projection_input=tensor(query),
                attention_output_projection_output=tensor(hidden),
                attention_delta=tensor(hidden),
                post_attention_residual=tensor(hidden),
                pre_mlp_norm_output=tensor(hidden),
                mlp_input_projection_output=(
                    tensor(intermediate)
                    if block_layout.mlp.input_projection_path is not None
                    else None
                ),
                mlp_gate_projection_output=(
                    tensor(intermediate)
                    if block_layout.mlp.gate_projection_path is not None
                    else None
                ),
                mlp_up_projection_output=(
                    tensor(intermediate)
                    if block_layout.mlp.up_projection_path is not None
                    else None
                ),
                activation_output=tensor(intermediate),
                down_projection_input=tensor(intermediate),
                down_projection_output=tensor(hidden),
                mlp_delta=tensor(hidden),
                block_output=tensor(hidden),
            )
        )
    return HFTransportTrace(
        layout=layout,
        valid_token_mask=mask,
        blocks=tuple(blocks),
        final_norm_output=_values(batch, sequence, layout.hidden_size, offset + 1),
        logits=_values(
            batch,
            sequence,
            layout.embeddings.vocab_size,
            offset + 2,
        ),
        hook_counts=MappingProxyType({}),
    )


def _expected_mean(value, mask):
    return torch.stack(
        [value[row, mask[row]].mean(dim=0) for row in range(mask.shape[0])]
    )


def _expected_last(value, mask):
    indices = [mask[row].nonzero().flatten()[-1] for row in range(mask.shape[0])]
    return torch.stack([value[row, index] for row, index in enumerate(indices)])


def test_mean_pooling_preserves_gpt2_block_semantics_and_both_logit_anchors() -> None:
    mask = torch.tensor([[True, True, False, False], [False, True, False, True]])
    trace = _synthetic_trace(_tiny_gpt2(), mask, offset=1)

    pooled = pool_hf_transport_trace(
        trace,
        pooling="mean",
        logits_policy="both",
    )

    assert pooled.source_batch_sequence_shape == (2, 4)
    assert pooled.batch_count == 2
    assert torch.equal(pooled.valid_token_counts, torch.tensor([2, 2]))
    assert torch.equal(pooled.source_valid_token_mask, mask)
    assert torch.equal(
        pooled.final_norm_output,
        _expected_mean(trace.final_norm_output, mask),
    )
    for field in fields(DecoderBlockTrace):
        if field.name == "index":
            continue
        original = getattr(trace.blocks[0], field.name)
        actual = getattr(pooled.blocks[0], field.name)
        if original is None:
            assert actual is None
        else:
            assert actual.shape == (2, original.shape[-1])
            assert torch.equal(actual, _expected_mean(original, mask))
            assert actual.requires_grad is False
    assert pooled.logits is None
    assert torch.equal(pooled.mean_logits, _expected_mean(trace.logits, mask))
    assert torch.equal(
        pooled.last_valid_logits,
        _expected_last(trace.logits, mask),
    )


def test_last_valid_pooling_uses_greatest_selected_llama_position() -> None:
    mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, True, True, False, True],
        ]
    )
    trace = _synthetic_trace(_tiny_llama(), mask, offset=2)

    pooled = pool_hf_transport_trace(
        trace,
        pooling="last_valid",
        logits_policy="last_valid",
    )

    assert torch.equal(
        pooled.blocks[0].block_output,
        _expected_last(trace.blocks[0].block_output, mask),
    )
    assert torch.equal(
        pooled.final_norm_output,
        _expected_last(trace.final_norm_output, mask),
    )
    assert pooled.logits is pooled.last_valid_logits
    assert torch.equal(pooled.logits, _expected_last(trace.logits, mask))
    assert pooled.mean_logits is None


def test_row_alignment_allows_different_families_and_sequence_lengths() -> None:
    source_mask = torch.tensor([[True, True, False], [False, True, True]])
    target_mask = torch.tensor(
        [
            [False, True, True, False, True],
            [True, False, False, True, False],
        ]
    )
    source = _synthetic_trace(_tiny_gpt2(), source_mask, offset=3)
    target = _synthetic_trace(_tiny_llama(), target_mask, offset=4)

    aligned = pool_row_aligned_hf_transport_traces(
        source,
        target,
        pooling="mean",
        logits_policy="mean",
    )

    assert aligned.batch_count == 2
    assert aligned.source.source_batch_sequence_shape == (2, 3)
    assert aligned.target.source_batch_sequence_shape == (2, 5)
    assert aligned.source.final_norm_output.shape == (2, 6)
    assert aligned.target.final_norm_output.shape == (2, 8)
    assert aligned.source.logits.shape == (2, 11)
    assert aligned.target.logits.shape == (2, 13)
    assert torch.equal(aligned.source.logits, aligned.source.mean_logits)
    assert torch.equal(aligned.target.logits, aligned.target.mean_logits)


def test_none_logit_policy_does_not_pool_or_expose_logits() -> None:
    mask = torch.tensor([[True, False, True]])
    trace = _synthetic_trace(_tiny_gpt2(), mask, offset=5)

    pooled = pool_hf_transport_trace(
        trace,
        pooling="mean",
        logits_policy="none",
    )

    assert pooled.logits is None
    assert pooled.last_valid_logits is None
    assert pooled.mean_logits is None


@pytest.mark.parametrize(
    ("pooling", "logits_policy", "message"),
    [
        ("sum", "none", "pooling"),
        ("mean", "automatic", "logits_policy"),
    ],
)
def test_pooling_rejects_unknown_policies(pooling, logits_policy, message) -> None:
    trace = _synthetic_trace(
        _tiny_gpt2(),
        torch.tensor([[True, True]]),
        offset=6,
    )

    with pytest.raises(TracePoolingError, match=message):
        pool_hf_transport_trace(
            trace,
            pooling=pooling,
            logits_policy=logits_policy,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda trace: replace(trace, valid_token_mask=torch.ones(1, 3)), "boolean"),
        (
            lambda trace: replace(
                trace,
                valid_token_mask=torch.tensor([[True, False, False], [False] * 3]),
            ),
            "Every sequence",
        ),
        (
            lambda trace: replace(
                trace,
                final_norm_output=trace.final_norm_output[:, :-1],
            ),
            "final_norm_output must have shape",
        ),
        (
            lambda trace: replace(
                trace,
                logits=trace.logits.clone().index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor([float("nan")], dtype=trace.logits.dtype),
                ),
            ),
            "logits contains NaN",
        ),
        (
            lambda trace: replace(
                trace,
                final_norm_output=trace.final_norm_output.clone().requires_grad_(),
            ),
            "gradient graph",
        ),
    ],
)
def test_pooling_strictly_rejects_invalid_masks_shapes_and_values(
    mutation,
    message,
) -> None:
    trace = _synthetic_trace(
        _tiny_gpt2(),
        torch.tensor([[True, True, False], [True, False, True]]),
        offset=7,
    )

    with pytest.raises(TracePoolingError, match=message):
        pool_hf_transport_trace(
            mutation(trace),
            pooling="mean",
            logits_policy="none",
        )


def test_row_alignment_rejects_different_batch_counts() -> None:
    source = _synthetic_trace(
        _tiny_gpt2(),
        torch.tensor([[True, True], [True, False]]),
        offset=8,
    )
    target = _synthetic_trace(
        _tiny_llama(),
        torch.tensor([[True, False, True]]),
        offset=9,
    )

    with pytest.raises(TracePoolingError, match="same batch count"):
        pool_row_aligned_hf_transport_traces(
            source,
            target,
            pooling="last_valid",
            logits_policy="none",
        )
