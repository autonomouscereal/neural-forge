from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from neural_forge.hf_lfm2_cached_second_order_attention_solver import (
    verify_lfm2_attention_bridge,
    verify_lfm2_cached_second_order_evidence,
)
from neural_forge.hf_lfm2_crevf_trace_evidence import (
    Lfm2CrevfTraceEvidenceError,
    build_lfm2_causal_gqa_attention_bridge,
    build_lfm2_tail_anchored_second_order_evidence,
)


def _assert_parameter_free(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    assert not isinstance(value, nn.Parameter)
    if isinstance(value, torch.Tensor):
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _assert_parameter_free(getattr(value, field.name), seen)


def test_tail_evidence_uses_source_regularizer_and_repeated_real_parent_tail():
    generator = torch.Generator().manual_seed(31)
    weight = torch.randn(7, 5, generator=generator).to(torch.bfloat16)
    child_inputs = torch.randn(2, 6, 5, generator=generator)
    child_outputs = child_inputs @ weight.float().T
    parent_tail_inputs = child_inputs[:, -2:, :] + 0.3
    parent_tail_outputs = parent_tail_inputs @ (weight.float() + 0.1).T
    evidence, audit = build_lfm2_tail_anchored_second_order_evidence(
        weight,
        site_id="model.layers.3.feed_forward.w2.weight",
        tensor_role="residual.writer",
        child_inputs=child_inputs,
        child_outputs=child_outputs,
        cached_parent_tail_inputs=parent_tail_inputs,
        cached_parent_tail_outputs=parent_tail_outputs,
        anchor_repeat=3,
    )
    verify_lfm2_cached_second_order_evidence(evidence)
    assert evidence.sample_count == 2 * 6 + 2 * 2 * 2
    assert audit.source_regularization_rows == 2 * 4
    assert audit.cached_parent_anchor_rows == 2 * 2 * 3
    assert audit.residual_output_sensitivities
    assert evidence.output_sensitivities is not None
    assert torch.equal(
        evidence.output_sensitivities,
        evidence.parent_outputs - evidence.child_outputs,
    )
    assert audit.parent_model_calls == 0
    assert audit.child_model_calls == 0
    assert audit.training_steps == 0
    assert torch.equal(
        evidence.parent_inputs[-8:],
        parent_tail_inputs.reshape(-1, 5).to(torch.float64).repeat(2, 1),
    )
    assert torch.equal(
        evidence.parent_outputs[-8:],
        parent_tail_outputs.reshape(-1, 7).to(torch.float64).repeat(2, 1),
    )
    _assert_parameter_free((evidence, audit))


def test_tail_evidence_can_disable_output_row_coupling_explicitly():
    weight = torch.eye(4)
    inputs = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    outputs = inputs @ weight.T
    evidence, audit = build_lfm2_tail_anchored_second_order_evidence(
        weight,
        site_id="model.layers.0.feed_forward.w1.weight",
        tensor_role="ffn.producer",
        child_inputs=inputs,
        child_outputs=outputs,
        cached_parent_tail_inputs=inputs[:, -2:],
        cached_parent_tail_outputs=outputs[:, -2:],
        residual_output_sensitivities=False,
    )
    assert evidence.output_sensitivities is None
    assert not audit.residual_output_sensitivities


def test_tail_evidence_rejects_unaligned_or_too_long_parent_tail():
    weight = torch.zeros(3, 4)
    with pytest.raises(Lfm2CrevfTraceEvidenceError, match="longer"):
        build_lfm2_tail_anchored_second_order_evidence(
            weight,
            site_id="x",
            tensor_role="other",
            child_inputs=torch.zeros(1, 2, 4),
            child_outputs=torch.zeros(1, 2, 3),
            cached_parent_tail_inputs=torch.zeros(1, 3, 4),
            cached_parent_tail_outputs=torch.zeros(1, 3, 3),
        )
    with pytest.raises(Lfm2CrevfTraceEvidenceError, match="geometry"):
        build_lfm2_tail_anchored_second_order_evidence(
            weight,
            site_id="x",
            tensor_role="other",
            child_inputs=torch.zeros(1, 2, 4),
            child_outputs=torch.zeros(1, 2, 3),
            cached_parent_tail_inputs=torch.zeros(2, 1, 4),
            cached_parent_tail_outputs=torch.zeros(2, 1, 3),
        )


def _gqa_fixture():
    generator = torch.Generator().manual_seed(43)
    batch = 2
    query_heads = 4
    key_value_heads = 2
    tokens = 4
    head_dim = 3
    query = torch.randn(
        batch,
        query_heads,
        tokens,
        head_dim,
        generator=generator,
    )
    key = torch.randn(
        batch,
        key_value_heads,
        tokens,
        head_dim,
        generator=generator,
    )
    value = torch.randn(
        batch,
        key_value_heads,
        tokens,
        head_dim,
        generator=generator,
    )
    repeats = query_heads // key_value_heads
    repeated_key = key.repeat_interleave(repeats, dim=1)
    repeated_value = value.repeat_interleave(repeats, dim=1)
    selected_query = query[:, :, -2:, :]
    scores = selected_query @ repeated_key.transpose(-1, -2) / (head_dim**0.5)
    positions = torch.arange(tokens)
    allowed = positions[None, :] <= positions[-2:, None]
    scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1)
    context = probabilities @ repeated_value
    return query, key, value, context


def test_causal_gqa_bridge_is_nonidentity_masked_and_reproduces_context():
    query, key, value, context = _gqa_fixture()
    bridge, audit = build_lfm2_causal_gqa_attention_bridge(
        site_id="model.layers.1.self_attn.v_proj.weight",
        child_query=query,
        child_key=key,
        child_value=value,
        child_context=context,
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        query_tokens=2,
    )
    verify_lfm2_attention_bridge(bridge)
    assert bridge.child_attention_probabilities.shape == (2, 4, 2, 4)
    assert bridge.query_heads == 4
    assert bridge.key_value_heads == 2
    assert audit.maximum_context_error < 1e-6
    assert audit.masked_probability_maximum == 0.0
    assert not torch.equal(
        bridge.child_attention_probabilities[..., -2:],
        torch.eye(2, dtype=torch.float64).expand(2, 4, 2, 2),
    )
    assert audit.parent_model_calls == 0
    assert audit.child_model_calls == 0
    _assert_parameter_free((bridge, audit))


def test_attention_bridge_fails_if_probabilities_cannot_explain_context():
    query, key, value, context = _gqa_fixture()
    corrupted = context.clone()
    corrupted[:, :, 0, :] += 1.0
    with pytest.raises(Lfm2CrevfTraceEvidenceError, match="do not reproduce"):
        build_lfm2_causal_gqa_attention_bridge(
            site_id="attention.v",
            child_query=query,
            child_key=key,
            child_value=value,
            child_context=corrupted,
            query_tokens=2,
        )


def test_attention_bridge_rejects_invalid_gqa_and_mask_geometry():
    query, key, value, context = _gqa_fixture()
    with pytest.raises(Lfm2CrevfTraceEvidenceError, match="GQA"):
        build_lfm2_causal_gqa_attention_bridge(
            site_id="attention.v",
            child_query=query[:, :3],
            child_key=key,
            child_value=value,
            child_context=context[:, :3],
            query_tokens=2,
        )
    with pytest.raises(Lfm2CrevfTraceEvidenceError, match="attention_mask"):
        build_lfm2_causal_gqa_attention_bridge(
            site_id="attention.v",
            child_query=query,
            child_key=key,
            child_value=value,
            child_context=context,
            attention_mask=torch.ones(1, 4),
            query_tokens=2,
        )
