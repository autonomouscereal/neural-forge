"""Fail-closed embedding and language-head transport from token alignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from neural_forge.hf_token_align import (
    AlignmentClass,
    TokenizerAlignmentReport,
    TransportAlignmentThresholds,
    require_alignment_for_transport,
)
from neural_forge.hf_transport_math import CoordinateMap, encode_coordinates


SpanReduction = Literal["mean", "sum"]


class TokenTransportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TokenTransportResult:
    embedding_weight: Any
    lm_head_weight: Any
    lm_head_bias: Any | None
    report: dict[str, Any]


def transport_vocabulary_rows(
    source_embedding_weight: Any,
    source_lm_head_weight: Any,
    alignment: TokenizerAlignmentReport,
    embedding_coordinates: CoordinateMap,
    lm_head_coordinates: CoordinateMap,
    *,
    source_lm_head_bias: Any | None = None,
    thresholds: TransportAlignmentThresholds | None = None,
    embedding_span_reduction: SpanReduction = "mean",
    allow_lm_head_spans: bool = False,
    lm_head_span_reduction: SpanReduction = "mean",
) -> TokenTransportResult:
    """Transport target vocabulary rows with separate embedding/head semantics.

    Embedding spans may be reduced explicitly. A source token sequence is not a
    single categorical next-token event, so language-head spans fail closed by
    default and must instead be fitted from aligned calibration likelihoods.
    """

    torch = _require_torch()
    with torch.no_grad():
        if not isinstance(alignment, TokenizerAlignmentReport):
            raise TokenTransportError("alignment must be a TokenizerAlignmentReport.")
        gate = require_alignment_for_transport(alignment, thresholds)
        embedding_reduction = _validate_reduction(
            embedding_span_reduction,
            "embedding_span_reduction",
        )
        head_reduction = _validate_reduction(
            lm_head_span_reduction,
            "lm_head_span_reduction",
        )
        if not isinstance(allow_lm_head_spans, bool):
            raise TokenTransportError("allow_lm_head_spans must be boolean.")
        embedding = _matrix(
            torch,
            source_embedding_weight,
            "source_embedding_weight",
        )
        head = _matrix(torch, source_lm_head_weight, "source_lm_head_weight")
        if embedding.shape[0] != alignment.donor_vocab_size:
            raise TokenTransportError(
                "Source embedding rows do not match donor tokenizer size."
            )
        if head.shape[0] != alignment.donor_vocab_size:
            raise TokenTransportError(
                "Source language-head rows do not match donor tokenizer size."
            )
        if embedding.shape[1] != embedding_coordinates.source_dim:
            raise TokenTransportError(
                "Source embedding width does not match embedding coordinates."
            )
        if head.shape[1] != lm_head_coordinates.source_dim:
            raise TokenTransportError(
                "Source language-head width does not match head coordinates."
            )
        bias = None
        if source_lm_head_bias is not None:
            bias = _vector(
                torch,
                source_lm_head_bias,
                "source_lm_head_bias",
                alignment.donor_vocab_size,
            )

        embedding_rows = []
        head_rows = []
        head_bias_rows = []
        exact_count = 0
        span_count = 0
        embedding_span_lengths = []
        head_span_lengths = []
        for item in alignment.alignments:
            if item.classification is AlignmentClass.UNALIGNABLE:
                raise TokenTransportError(
                    f"Target token {item.target_token_id} is unalignable after gate."
                )
            if len(item.candidates) != 1:
                raise TokenTransportError(
                    f"Target token {item.target_token_id} has ambiguous donor candidates."
                )
            token_ids = item.candidates[0].donor_token_ids
            indices = torch.tensor(token_ids, dtype=torch.long)
            source_embedding_rows = embedding.index_select(0, indices)
            source_head_rows = head.index_select(0, indices)
            if item.classification is AlignmentClass.EXACT_TOKEN:
                if len(token_ids) != 1:
                    raise TokenTransportError(
                        "Exact-token alignment must contain exactly one donor token."
                    )
                exact_count += 1
                embedding_row = source_embedding_rows[0]
                head_row = source_head_rows[0]
                bias_row = None if bias is None else bias[token_ids[0]]
            else:
                span_count += 1
                embedding_span_lengths.append(len(token_ids))
                embedding_row = _reduce_rows(
                    source_embedding_rows,
                    embedding_reduction,
                )
                if not allow_lm_head_spans:
                    raise TokenTransportError(
                        "A UTF-8 donor token span cannot initialize one LM-head row; "
                        "use a calibration likelihood fit or explicitly allow an "
                        "approximate head reduction."
                    )
                head_span_lengths.append(len(token_ids))
                head_row = _reduce_rows(source_head_rows, head_reduction)
                bias_row = (
                    None
                    if bias is None
                    else _reduce_rows(bias.index_select(0, indices), head_reduction)
                )
            embedding_rows.append(embedding_row)
            head_rows.append(head_row)
            if bias_row is not None:
                head_bias_rows.append(bias_row)

        stacked_embedding = torch.stack(embedding_rows, dim=0)
        stacked_head = torch.stack(head_rows, dim=0)
        transported_embedding = encode_coordinates(
            stacked_embedding,
            embedding_coordinates,
        )
        transported_head = encode_coordinates(stacked_head, lm_head_coordinates)
        transported_bias = (
            None
            if bias is None
            else torch.stack(head_bias_rows, dim=0).detach().contiguous()
        )
        expected_embedding_shape = (
            alignment.target_vocab_size,
            embedding_coordinates.target_dim,
        )
        expected_head_shape = (
            alignment.target_vocab_size,
            lm_head_coordinates.target_dim,
        )
        if tuple(transported_embedding.shape) != expected_embedding_shape:
            raise TokenTransportError("Transported embedding shape is incomplete.")
        if tuple(transported_head.shape) != expected_head_shape:
            raise TokenTransportError("Transported language-head shape is incomplete.")
        if transported_bias is not None and tuple(transported_bias.shape) != (
            alignment.target_vocab_size,
        ):
            raise TokenTransportError("Transported language-head bias is incomplete.")
        report = {
            "operation": "transport_vocabulary_rows",
            "alignment_fingerprint": alignment.fingerprint(),
            "alignment_gate": gate.to_dict(),
            "donor_vocab_size": alignment.donor_vocab_size,
            "target_vocab_size": alignment.target_vocab_size,
            "exact_token_rows": exact_count,
            "utf8_span_rows": span_count,
            "embedding_span_reduction": embedding_reduction,
            "embedding_span_lengths": embedding_span_lengths,
            "lm_head_spans_allowed": allow_lm_head_spans,
            "lm_head_span_reduction": (head_reduction if allow_lm_head_spans else None),
            "lm_head_span_lengths": head_span_lengths,
            "embedding_shape": list(transported_embedding.shape),
            "lm_head_shape": list(transported_head.shape),
            "lm_head_bias_present": transported_bias is not None,
            "random_rows": 0,
            "zero_gradient": True,
        }
        return TokenTransportResult(
            embedding_weight=transported_embedding.detach(),
            lm_head_weight=transported_head.detach(),
            lm_head_bias=transported_bias,
            report=report,
        )


def _reduce_rows(rows: Any, reduction: SpanReduction) -> Any:
    if reduction == "mean":
        return rows.mean(dim=0)
    if reduction == "sum":
        return rows.sum(dim=0)
    raise AssertionError(f"Unsupported reduction: {reduction}")


def _validate_reduction(value: Any, name: str) -> SpanReduction:
    if value not in {"mean", "sum"}:
        raise TokenTransportError(f"{name} must be mean or sum.")
    return value


def _matrix(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TokenTransportError(f"{name} must be a two-dimensional torch tensor.")
    if not torch.is_floating_point(value):
        raise TokenTransportError(f"{name} must have a floating-point dtype.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise TokenTransportError(f"{name} contains NaN or infinity.")
    return result


def _vector(torch: Any, value: Any, name: str, length: int) -> Any:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (length,):
        raise TokenTransportError(f"{name} must have shape ({length},).")
    if not torch.is_floating_point(value):
        raise TokenTransportError(f"{name} must have a floating-point dtype.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise TokenTransportError(f"{name} contains NaN or infinity.")
    return result


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for token transport.") from exc
    return torch


__all__ = [
    "TokenTransportError",
    "TokenTransportResult",
    "transport_vocabulary_rows",
]
