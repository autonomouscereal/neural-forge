"""Deterministic prompt-row pooling for Hugging Face transport traces."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

from neural_forge.hf_arch_layout import DecoderLayout
from neural_forge.hf_transport_trace import DecoderBlockTrace, HFTransportTrace


PoolingMode = Literal["mean", "last_valid"]
LogitsPoolingPolicy = Literal["none", "mean", "last_valid", "both"]


class TracePoolingError(ValueError):
    """Raised when a trace cannot be reduced to trustworthy prompt rows."""


@dataclass(frozen=True, slots=True)
class PooledTransportTrace:
    """One deterministic row per input sequence, suitable for closed-form fits.

    ``logits`` is populated only when ``logits_policy`` selects one logit
    representation. For ``both``, callers must use the two explicitly named
    fields so the fit target cannot be selected accidentally.
    """

    layout: DecoderLayout
    source_batch_sequence_shape: tuple[int, int]
    source_valid_token_mask: Any
    valid_token_counts: Any
    batch_count: int
    pooling: PoolingMode
    logits_policy: LogitsPoolingPolicy
    blocks: tuple[DecoderBlockTrace, ...]
    final_norm_output: Any
    logits: Any | None
    last_valid_logits: Any | None
    mean_logits: Any | None


@dataclass(frozen=True, slots=True)
class RowAlignedPooledTraces:
    """Source and target prompt anchors with matching batch-row semantics."""

    source: PooledTransportTrace
    target: PooledTransportTrace
    batch_count: int
    pooling: PoolingMode
    logits_policy: LogitsPoolingPolicy


_OPTIONAL_BLOCK_FIELDS = frozenset(
    {
        "fused_qkv_projection_output",
        "mlp_input_projection_output",
        "mlp_gate_projection_output",
        "mlp_up_projection_output",
    }
)
_RESIDUAL_BLOCK_FIELDS = frozenset(
    {
        "block_input",
        "pre_attention_norm_output",
        "attention_output_projection_output",
        "attention_delta",
        "post_attention_residual",
        "pre_mlp_norm_output",
        "down_projection_output",
        "mlp_delta",
        "block_output",
    }
)
_INTERMEDIATE_BLOCK_FIELDS = frozenset(
    {
        "mlp_input_projection_output",
        "mlp_gate_projection_output",
        "mlp_up_projection_output",
        "activation_output",
        "down_projection_input",
    }
)


def pool_hf_transport_trace(
    trace: HFTransportTrace,
    *,
    pooling: PoolingMode,
    logits_policy: LogitsPoolingPolicy,
) -> PooledTransportTrace:
    """Pool every trace tensor to one row per sequence without autograd.

    Valid-token means ignore all unselected positions. ``last_valid`` means the
    greatest selected sequence index, so left padding and non-contiguous masks
    have unambiguous behavior.
    """

    torch = _torch()
    pooling = _validate_pooling(pooling)
    logits_policy = _validate_logits_policy(logits_policy)
    mask, counts = _validate_mask(torch, trace)
    _validate_trace_structure(torch, trace, mask)

    with torch.no_grad():
        pooled_blocks = tuple(
            _pool_block(torch, trace, block, mask, pooling) for block in trace.blocks
        )
        final_norm = _pool_tensor(
            torch,
            trace.final_norm_output,
            mask,
            pooling,
            "final_norm_output",
        )
        last_valid_logits = None
        mean_logits = None
        if logits_policy in {"last_valid", "both"}:
            last_valid_logits = _pool_tensor(
                torch,
                trace.logits,
                mask,
                "last_valid",
                "logits",
            )
        if logits_policy in {"mean", "both"}:
            mean_logits = _pool_tensor(
                torch,
                trace.logits,
                mask,
                "mean",
                "logits",
            )

    selected_logits = None
    if logits_policy == "last_valid":
        selected_logits = last_valid_logits
    elif logits_policy == "mean":
        selected_logits = mean_logits

    return PooledTransportTrace(
        layout=trace.layout,
        source_batch_sequence_shape=(int(mask.shape[0]), int(mask.shape[1])),
        source_valid_token_mask=mask.detach().clone(),
        valid_token_counts=counts.detach().clone(),
        batch_count=int(mask.shape[0]),
        pooling=pooling,
        logits_policy=logits_policy,
        blocks=pooled_blocks,
        final_norm_output=final_norm,
        logits=selected_logits,
        last_valid_logits=last_valid_logits,
        mean_logits=mean_logits,
    )


def pool_row_aligned_hf_transport_traces(
    source_trace: HFTransportTrace,
    target_trace: HFTransportTrace,
    *,
    pooling: PoolingMode,
    logits_policy: LogitsPoolingPolicy,
) -> RowAlignedPooledTraces:
    """Pool tokenizer-independent traces and require one shared prompt count."""

    source = pool_hf_transport_trace(
        source_trace,
        pooling=pooling,
        logits_policy=logits_policy,
    )
    target = pool_hf_transport_trace(
        target_trace,
        pooling=pooling,
        logits_policy=logits_policy,
    )
    if source.batch_count != target.batch_count:
        raise TracePoolingError(
            "Source and target traces must have the same batch count for "
            f"row-aligned pooling; got {source.batch_count} and {target.batch_count}."
        )
    return RowAlignedPooledTraces(
        source=source,
        target=target,
        batch_count=source.batch_count,
        pooling=source.pooling,
        logits_policy=source.logits_policy,
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without HF extras
        raise RuntimeError(
            "Hugging Face trace pooling requires the 'hf' optional dependencies."
        ) from exc
    return torch


def _validate_pooling(value: Any) -> PoolingMode:
    if value not in {"mean", "last_valid"}:
        raise TracePoolingError("pooling must be 'mean' or 'last_valid'.")
    return value


def _validate_logits_policy(value: Any) -> LogitsPoolingPolicy:
    if value not in {"none", "mean", "last_valid", "both"}:
        raise TracePoolingError(
            "logits_policy must be 'none', 'mean', 'last_valid', or 'both'."
        )
    return value


def _validate_mask(torch: Any, trace: Any) -> tuple[Any, Any]:
    if not isinstance(trace, HFTransportTrace):
        raise TracePoolingError("trace must be an HFTransportTrace.")
    mask = trace.valid_token_mask
    if not torch.is_tensor(mask):
        raise TracePoolingError("valid_token_mask must be a torch.Tensor.")
    if mask.layout != torch.strided or mask.dtype != torch.bool or mask.ndim != 2:
        raise TracePoolingError(
            "valid_token_mask must be a dense rank-2 boolean tensor."
        )
    if int(mask.shape[0]) <= 0 or int(mask.shape[1]) <= 0:
        raise TracePoolingError("valid_token_mask cannot have an empty dimension.")
    counts = mask.sum(dim=1)
    empty_rows = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if empty_rows:
        raise TracePoolingError(
            "Every sequence must select at least one valid token; empty batch rows: "
            + ", ".join(str(int(row)) for row in empty_rows)
            + "."
        )
    return mask.detach(), counts.detach().to(device="cpu", dtype=torch.int64)


def _validate_trace_structure(torch: Any, trace: HFTransportTrace, mask: Any) -> None:
    layout = trace.layout
    if not isinstance(layout, DecoderLayout):
        raise TracePoolingError("trace.layout must be a DecoderLayout.")
    if not isinstance(trace.blocks, tuple):
        raise TracePoolingError("trace.blocks must be a tuple.")
    if len(trace.blocks) != layout.layer_count or not trace.blocks:
        raise TracePoolingError(
            "trace.blocks must contain exactly one block per decoder layer."
        )

    for block, block_layout in zip(trace.blocks, layout.blocks):
        if not isinstance(block, DecoderBlockTrace):
            raise TracePoolingError("Every trace block must be a DecoderBlockTrace.")
        if block.index != block_layout.index:
            raise TracePoolingError(
                f"Trace block index {block.index} does not match layout block "
                f"index {block_layout.index}."
            )
        expected_widths = _block_widths(layout, block_layout)
        expected_presence = {
            "fused_qkv_projection_output": block_layout.attention.fused_qkv_path
            is not None,
            "mlp_input_projection_output": block_layout.mlp.input_projection_path
            is not None,
            "mlp_gate_projection_output": block_layout.mlp.gate_projection_path
            is not None,
            "mlp_up_projection_output": block_layout.mlp.up_projection_path is not None,
        }
        for field in fields(DecoderBlockTrace):
            if field.name == "index":
                continue
            value = getattr(block, field.name)
            if field.name in _OPTIONAL_BLOCK_FIELDS:
                required = expected_presence[field.name]
                if required != (value is not None):
                    state = "present" if required else "absent"
                    raise TracePoolingError(
                        f"block.{block.index}.{field.name} must be {state} for "
                        f"the {layout.family} layout."
                    )
                if value is None:
                    continue
            _validate_tensor(
                torch,
                value,
                mask,
                expected_widths[field.name],
                f"block.{block.index}.{field.name}",
            )

    _validate_tensor(
        torch,
        trace.final_norm_output,
        mask,
        layout.hidden_size,
        "final_norm_output",
    )
    _validate_tensor(
        torch,
        trace.logits,
        mask,
        layout.embeddings.vocab_size,
        "logits",
    )


def _block_widths(layout: DecoderLayout, block_layout: Any) -> dict[str, int]:
    hidden = layout.hidden_size
    intermediate = layout.intermediate_size
    query = block_layout.attention.query_width
    key_value = block_layout.attention.key_value_width
    widths = {name: hidden for name in _RESIDUAL_BLOCK_FIELDS}
    widths.update({name: intermediate for name in _INTERMEDIATE_BLOCK_FIELDS})
    widths.update(
        {
            "fused_qkv_projection_output": query + 2 * key_value,
            "query_projection_output": query,
            "key_projection_output": key_value,
            "value_projection_output": key_value,
            "attention_output_projection_input": query,
        }
    )
    return widths


def _validate_tensor(
    torch: Any,
    value: Any,
    mask: Any,
    expected_width: int,
    name: str,
) -> None:
    if not torch.is_tensor(value):
        raise TracePoolingError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.ndim != 3:
        raise TracePoolingError(f"{name} must be a dense rank-3 tensor.")
    expected_shape = (int(mask.shape[0]), int(mask.shape[1]), int(expected_width))
    if tuple(int(size) for size in value.shape) != expected_shape:
        raise TracePoolingError(
            f"{name} must have shape {expected_shape}, got {tuple(value.shape)}."
        )
    if not torch.is_floating_point(value):
        raise TracePoolingError(f"{name} must have a floating-point dtype.")
    if value.requires_grad or value.grad_fn is not None:
        raise TracePoolingError(f"{name} must not retain a gradient graph.")
    if not bool(torch.isfinite(value).all().item()):
        raise TracePoolingError(f"{name} contains NaN or infinity.")


def _pool_block(
    torch: Any,
    trace: HFTransportTrace,
    block: DecoderBlockTrace,
    mask: Any,
    pooling: PoolingMode,
) -> DecoderBlockTrace:
    del trace
    values: dict[str, Any] = {}
    for field in fields(DecoderBlockTrace):
        if field.name == "index":
            continue
        value = getattr(block, field.name)
        values[field.name] = (
            None
            if value is None
            else _pool_tensor(
                torch,
                value,
                mask,
                pooling,
                f"block.{block.index}.{field.name}",
            )
        )
    return DecoderBlockTrace(index=block.index, **values)


def _pool_tensor(
    torch: Any,
    value: Any,
    mask: Any,
    pooling: PoolingMode,
    name: str,
) -> Any:
    detached = value.detach()
    device_mask = mask.to(device=detached.device)
    if pooling == "last_valid":
        positions = torch.arange(detached.shape[1], device=detached.device)
        positions = positions.unsqueeze(0).expand(detached.shape[0], -1)
        indices = positions.masked_fill(~device_mask, -1).max(dim=1).values
        batch_indices = torch.arange(detached.shape[0], device=detached.device)
        pooled = detached[batch_indices, indices]
    else:
        rows = [
            detached[index, device_mask[index]].mean(dim=0)
            for index in range(detached.shape[0])
        ]
        pooled = torch.stack(rows, dim=0)
    result = pooled.detach().contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise TracePoolingError(f"Pooling {name} produced NaN or infinity.")
    return result


# Short aliases for callers already scoped to Hugging Face transport traces.
pool_transport_trace = pool_hf_transport_trace
pool_row_aligned_transport_traces = pool_row_aligned_hf_transport_traces


__all__ = [
    "LogitsPoolingPolicy",
    "PooledTransportTrace",
    "PoolingMode",
    "RowAlignedPooledTraces",
    "TracePoolingError",
    "pool_hf_transport_trace",
    "pool_row_aligned_hf_transport_traces",
    "pool_row_aligned_transport_traces",
    "pool_transport_trace",
]
