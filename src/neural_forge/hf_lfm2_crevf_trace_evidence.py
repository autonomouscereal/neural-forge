"""Turn cached LFM2 traces into second-order CREVF evidence.

The parent cache deliberately stores only a bounded tail of each causal path.
The child-only refresh may store a complete prefill.  This module combines
those two sources without calling either model:

* complete child rows regularize a folded operator toward the source operator;
* the aligned cached parent tail replaces that synthetic target with the real
  parent input/output path; and
* fixed child Q/K traces produce the actual causal GQA probability map used to
  propagate V quantization error into the O projection.

The resulting evidence is immutable, CPU-resident, and accepted directly by
``hf_lfm2_cached_second_order_attention_solver``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .hf_lfm2_cached_second_order_attention_solver import (
    Lfm2CachedSecondOrderEvidence,
    Lfm2SealedAttentionBridge,
    seal_lfm2_attention_bridge,
    seal_lfm2_cached_second_order_evidence,
    verify_lfm2_attention_bridge,
    verify_lfm2_cached_second_order_evidence,
)


class Lfm2CrevfTraceEvidenceError(ValueError):
    """Cached parent/child paths cannot form authenticated CREVF evidence."""


@dataclass(frozen=True, slots=True)
class Lfm2CrevfTailEvidenceAudit:
    site_id: str
    batch_size: int
    sequence_tokens: int
    parent_tail_tokens: int
    source_regularization_rows: int
    cached_parent_anchor_rows: int
    anchor_repeat: int
    total_solver_rows: int
    residual_output_sensitivities: bool
    source_teacher_calls: int
    parent_model_calls: int
    child_model_calls: int
    training_steps: int
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2CrevfAttentionBridgeAudit:
    site_id: str
    batch_size: int
    query_tokens: int
    key_tokens: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    maximum_context_error: float
    mean_context_error: float
    masked_probability_maximum: float
    parent_model_calls: int
    child_model_calls: int
    training_steps: int
    bridge_sha256: str


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CrevfTraceEvidenceError(
            "Trace evidence metadata must be finite canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _plain_tensor(
    value: Any,
    name: str,
    *,
    rank: int,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or isinstance(value, nn.Parameter)
        or value.requires_grad
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.ndim != rank
        or value.numel() == 0
        or value.dtype.is_complex
    ):
        raise Lfm2CrevfTraceEvidenceError(
            f"{name} must be a nonempty plain rank-{rank} dense Tensor."
        )
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise Lfm2CrevfTraceEvidenceError(f"{name} contains NaN or infinity.")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CrevfTraceEvidenceError(f"{name} must be a positive integer.")
    return value


def build_lfm2_tail_anchored_second_order_evidence(
    source_weight: Tensor,
    *,
    site_id: str,
    tensor_role: str,
    child_inputs: Tensor,
    child_outputs: Tensor,
    cached_parent_tail_inputs: Tensor,
    cached_parent_tail_outputs: Tensor,
    anchor_repeat: int = 4,
    residual_output_sensitivities: bool = True,
) -> tuple[Lfm2CachedSecondOrderEvidence, Lfm2CrevfTailEvidenceAudit]:
    """Blend a complete child prefill with the real cached parent tail.

    The unanchored child rows use ``child_inputs @ source_weight.T`` as a
    zero-forward teacher target.  The last ``T`` rows of each sequence are
    replaced by the aligned cached parent inputs and outputs, then repeated to
    give the real parent path explicit leverage over the regularizer.
    """

    if not isinstance(site_id, str) or not site_id.strip():
        raise Lfm2CrevfTraceEvidenceError("site_id must be nonempty text.")
    if not isinstance(tensor_role, str) or not tensor_role.strip():
        raise Lfm2CrevfTraceEvidenceError("tensor_role must be nonempty text.")
    repeat = _positive_int(anchor_repeat, "anchor_repeat")
    weight = _plain_tensor(source_weight, "source_weight", rank=2)
    child_x = _plain_tensor(child_inputs, "child_inputs", rank=3)
    child_y = _plain_tensor(child_outputs, "child_outputs", rank=3)
    parent_x = _plain_tensor(
        cached_parent_tail_inputs,
        "cached_parent_tail_inputs",
        rank=3,
    )
    parent_y = _plain_tensor(
        cached_parent_tail_outputs,
        "cached_parent_tail_outputs",
        rank=3,
    )
    batch, sequence, input_features = child_x.shape
    if child_y.shape[:2] != (batch, sequence):
        raise Lfm2CrevfTraceEvidenceError(
            "Child input and output paths are not batch/sequence aligned."
        )
    output_features = int(child_y.shape[2])
    if weight.shape != (output_features, input_features):
        raise Lfm2CrevfTraceEvidenceError(
            "source_weight geometry differs from the child operator path."
        )
    if (
        parent_x.shape[0] != batch
        or parent_y.shape[0] != batch
        or parent_x.shape[1] != parent_y.shape[1]
        or parent_x.shape[2] != input_features
        or parent_y.shape[2] != output_features
    ):
        raise Lfm2CrevfTraceEvidenceError(
            "Cached parent input/output tails have incompatible geometry."
        )
    tail = int(parent_x.shape[1])
    if tail > sequence:
        raise Lfm2CrevfTraceEvidenceError(
            "Cached parent tail is longer than the refreshed child sequence."
        )

    child_x_rows = child_x.reshape(batch * sequence, input_features)
    child_y_rows = child_y.reshape(batch * sequence, output_features)
    parent_x_full = child_x.clone()
    parent_y_full = torch.matmul(child_x, weight.T)
    parent_x_full[:, -tail:, :] = parent_x
    parent_y_full[:, -tail:, :] = parent_y
    parent_x_rows = parent_x_full.reshape(batch * sequence, input_features)
    parent_y_rows = parent_y_full.reshape(batch * sequence, output_features)

    if repeat > 1:
        repeated_parent_x = parent_x.reshape(batch * tail, input_features).repeat(
            repeat - 1,
            1,
        )
        repeated_parent_y = parent_y.reshape(batch * tail, output_features).repeat(
            repeat - 1,
            1,
        )
        repeated_child_x = (
            child_x[:, -tail:, :]
            .reshape(
                batch * tail,
                input_features,
            )
            .repeat(repeat - 1, 1)
        )
        repeated_child_y = (
            child_y[:, -tail:, :]
            .reshape(
                batch * tail,
                output_features,
            )
            .repeat(repeat - 1, 1)
        )
        parent_x_rows = torch.cat((parent_x_rows, repeated_parent_x), dim=0)
        parent_y_rows = torch.cat((parent_y_rows, repeated_parent_y), dim=0)
        child_x_rows = torch.cat((child_x_rows, repeated_child_x), dim=0)
        child_y_rows = torch.cat((child_y_rows, repeated_child_y), dim=0)

    if not isinstance(residual_output_sensitivities, bool):
        raise Lfm2CrevfTraceEvidenceError(
            "residual_output_sensitivities must be boolean."
        )
    sensitivities = (
        (parent_y_rows - child_y_rows).contiguous()
        if residual_output_sensitivities
        else None
    )
    evidence = seal_lfm2_cached_second_order_evidence(
        site_id=site_id.strip(),
        tensor_role=tensor_role.strip(),
        parent_inputs=parent_x_rows,
        child_inputs=child_x_rows,
        parent_outputs=parent_y_rows,
        child_outputs=child_y_rows,
        output_sensitivities=sensitivities,
    )
    verify_lfm2_cached_second_order_evidence(evidence)
    anchor_rows = batch * tail * repeat
    audit = Lfm2CrevfTailEvidenceAudit(
        site_id=site_id.strip(),
        batch_size=batch,
        sequence_tokens=sequence,
        parent_tail_tokens=tail,
        source_regularization_rows=batch * max(0, sequence - tail),
        cached_parent_anchor_rows=anchor_rows,
        anchor_repeat=repeat,
        total_solver_rows=evidence.sample_count,
        residual_output_sensitivities=residual_output_sensitivities,
        source_teacher_calls=1,
        parent_model_calls=0,
        child_model_calls=0,
        training_steps=0,
        evidence_sha256=evidence.evidence_sha256,
    )
    return evidence, audit


def build_lfm2_causal_gqa_attention_bridge(
    *,
    site_id: str,
    child_query: Tensor,
    child_key: Tensor,
    child_value: Tensor,
    child_context: Tensor,
    attention_mask: Tensor | None = None,
    query_tokens: int | None = None,
    scaling: float | None = None,
    context_rtol: float = 5e-3,
    context_atol: float = 5e-4,
) -> tuple[Lfm2SealedAttentionBridge, Lfm2CrevfAttentionBridgeAudit]:
    """Reconstruct and seal the actual fixed-A causal GQA bridge.

    Q/K/V/context use the collector's canonical shapes:
    ``[batch, heads, sequence, head_dim]``.  ``query_tokens`` may select a
    suffix of query/context rows while all key/value tokens remain available.
    """

    if not isinstance(site_id, str) or not site_id.strip():
        raise Lfm2CrevfTraceEvidenceError("site_id must be nonempty text.")
    query = _plain_tensor(child_query, "child_query", rank=4)
    key = _plain_tensor(child_key, "child_key", rank=4)
    value = _plain_tensor(child_value, "child_value", rank=4)
    context = _plain_tensor(child_context, "child_context", rank=4)
    batch, query_heads, full_query_tokens, head_dim = query.shape
    key_batch, key_value_heads, key_tokens, key_head_dim = key.shape
    if (
        key_batch != batch
        or value.shape != (batch, key_value_heads, key_tokens, head_dim)
        or key_head_dim != head_dim
        or query_heads % key_value_heads
    ):
        raise Lfm2CrevfTraceEvidenceError(
            "Q/K/V tensors do not form valid causal GQA geometry."
        )
    selected_queries = (
        full_query_tokens
        if query_tokens is None
        else _positive_int(query_tokens, "query_tokens")
    )
    if selected_queries > full_query_tokens or selected_queries > key_tokens:
        raise Lfm2CrevfTraceEvidenceError(
            "query_tokens exceeds the available causal Q/K sequence."
        )
    if context.shape != (batch, query_heads, selected_queries, head_dim):
        raise Lfm2CrevfTraceEvidenceError(
            "child_context must contain exactly the selected query suffix."
        )
    scale = 1.0 / math.sqrt(head_dim) if scaling is None else float(scaling)
    if not math.isfinite(scale) or scale <= 0.0:
        raise Lfm2CrevfTraceEvidenceError("scaling must be finite and positive.")
    if (
        not math.isfinite(float(context_rtol))
        or not math.isfinite(float(context_atol))
        or context_rtol < 0.0
        or context_atol < 0.0
    ):
        raise Lfm2CrevfTraceEvidenceError(
            "Context tolerances must be finite and nonnegative."
        )

    query_suffix = query[:, :, -selected_queries:, :]
    repeats = query_heads // key_value_heads
    repeated_key = key.repeat_interleave(repeats, dim=1)
    repeated_value = value.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query_suffix, repeated_key.transpose(-1, -2)) * scale
    query_positions = torch.arange(
        full_query_tokens - selected_queries,
        full_query_tokens,
        dtype=torch.long,
    )
    key_positions = torch.arange(key_tokens, dtype=torch.long)
    allowed = key_positions[None, :] <= query_positions[:, None]
    allowed = allowed.reshape(1, 1, selected_queries, key_tokens).expand(
        batch,
        query_heads,
        selected_queries,
        key_tokens,
    )
    if attention_mask is not None:
        raw_mask = _plain_tensor(attention_mask, "attention_mask", rank=2)
        if raw_mask.shape != (batch, key_tokens):
            raise Lfm2CrevfTraceEvidenceError(
                "attention_mask must have [batch, key_tokens] geometry."
            )
        allowed = allowed & raw_mask.to(torch.bool)[:, None, None, :]
    if not bool(allowed.any(dim=-1).all()):
        raise Lfm2CrevfTraceEvidenceError(
            "At least one selected query has no allowed attention key."
        )
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(torch.float64)
    reproduced = torch.matmul(probabilities, repeated_value)
    difference = (reproduced - context).abs()
    maximum_error = float(difference.amax())
    mean_error = float(difference.mean())
    if not torch.allclose(
        reproduced,
        context,
        rtol=float(context_rtol),
        atol=float(context_atol),
    ):
        raise Lfm2CrevfTraceEvidenceError(
            "Reconstructed causal GQA probabilities do not reproduce the "
            f"captured child context (maximum error {maximum_error:.8g})."
        )
    masked_values = probabilities.masked_select(~allowed)
    masked_maximum = (
        0.0 if masked_values.numel() == 0 else float(masked_values.abs().amax())
    )
    trace_config_sha256 = _canonical_sha256(
        {
            "schema": "neural-forge.lfm2.crevf-fixed-causal-gqa.v1",
            "causal": True,
            "batch_size": batch,
            "full_query_tokens": full_query_tokens,
            "query_tokens": selected_queries,
            "key_tokens": key_tokens,
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "head_dim": head_dim,
            "scaling": scale,
            "softmax_dtype": "float32",
            "context_rtol": float(context_rtol),
            "context_atol": float(context_atol),
        }
    )
    bridge = seal_lfm2_attention_bridge(
        site_id=site_id.strip(),
        child_attention_probabilities=probabilities,
        batch_size=batch,
        query_tokens=selected_queries,
        key_tokens=key_tokens,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        trace_attention_config_sha256=trace_config_sha256,
        attention_mask=allowed,
    )
    verify_lfm2_attention_bridge(bridge)
    audit = Lfm2CrevfAttentionBridgeAudit(
        site_id=site_id.strip(),
        batch_size=batch,
        query_tokens=selected_queries,
        key_tokens=key_tokens,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        maximum_context_error=maximum_error,
        mean_context_error=mean_error,
        masked_probability_maximum=masked_maximum,
        parent_model_calls=0,
        child_model_calls=0,
        training_steps=0,
        bridge_sha256=bridge.bridge_sha256,
    )
    return bridge, audit


__all__ = [
    "Lfm2CrevfAttentionBridgeAudit",
    "Lfm2CrevfTailEvidenceAudit",
    "Lfm2CrevfTraceEvidenceError",
    "build_lfm2_causal_gqa_attention_bridge",
    "build_lfm2_tail_anchored_second_order_evidence",
]
