"""Tokenizer-neutral token-trace alignment over exact UTF-8 byte spans.

Hugging Face fast-tokenizer offsets are normally Unicode-codepoint offsets,
not UTF-8 byte offsets.  This module validates each parent trace against its
raw prompts, converts offsets to byte spans, builds one shared atomic segment
inventory, and overlap-pools token states into those rows.  It performs no
learned projection and never runs a model or optimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from neural_forge.hf_canonical_kernel import (
    CanonicalDepthCover,
    CanonicalParentTrace,
    CanonicalRowIdentity,
)


OffsetUnit = Literal["character", "utf8_byte"]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ByteSpanTraceError(ValueError):
    """Raised when token traces cannot be proven byte-aligned."""


@dataclass(frozen=True, slots=True)
class ByteSpanTraceInput:
    """One parent's full-token trace and tokenizer provenance.

    ``hidden_states`` accepts either one ``[batch, sequence, width]`` tensor or
    an ordered sequence of such tensors (for example every layer boundary).
    ``logits`` is optional and follows the same three-dimensional shape rule.
    Offsets default to Hugging Face's Unicode-codepoint convention.
    """

    parent_name: str
    raw_prompts: Sequence[str]
    tokenizer: Any
    input_ids: Any
    offset_mapping: Any
    attention_mask: Any
    hidden_states: Any
    logits: Any | None = None
    offset_unit: OffsetUnit = "character"
    tokenizer_fingerprint: str | None = None
    cache_fingerprint: str | None = None


ParentByteSpanTrace = ByteSpanTraceInput


@dataclass(frozen=True, slots=True)
class CanonicalByteSegment:
    """One shared row backed by a contiguous raw-prompt byte segment."""

    row_id: str
    prompt_index: int
    start_byte: int
    end_byte: int
    utf8_hex: str
    segment_sha256: str
    raw_prompt_sha256: str

    @property
    def byte_length(self) -> int:
        return self.end_byte - self.start_byte

    def report(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "prompt_index": self.prompt_index,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "byte_length": self.byte_length,
            "utf8_hex": self.utf8_hex,
            "segment_sha256": self.segment_sha256,
            "raw_prompt_sha256": self.raw_prompt_sha256,
        }


@dataclass(frozen=True, slots=True)
class AlignedByteSpanTrace:
    """A parent's token tensors pooled into the canonical segment rows."""

    parent_name: str
    row_identity: CanonicalRowIdentity
    aligned_hidden_states: tuple[Any, ...]
    aligned_logits: Any | None
    tokenizer_fingerprint: str
    cache_fingerprint: str
    active_token_count: int
    zero_width_token_count: int
    trimmed_whitespace_byte_count: int
    synthetic_prefix_space_count: int
    tensor_sha256: tuple[str, ...]
    fingerprint: str

    @property
    def hidden_states(self) -> Any:
        """Return one matrix directly, or all layer matrices as a tuple."""

        if len(self.aligned_hidden_states) == 1:
            return self.aligned_hidden_states[0]
        return self.aligned_hidden_states

    @property
    def logits(self) -> Any | None:
        return self.aligned_logits

    def report(self) -> dict[str, Any]:
        return {
            "parent_name": self.parent_name,
            "row_identity": self.row_identity.report(),
            "hidden_state_matrix_count": len(self.aligned_hidden_states),
            "has_logits": self.aligned_logits is not None,
            "active_token_count": self.active_token_count,
            "zero_width_token_count": self.zero_width_token_count,
            "trimmed_whitespace_byte_count": self.trimmed_whitespace_byte_count,
            "synthetic_prefix_space_count": self.synthetic_prefix_space_count,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "tensor_sha256": list(self.tensor_sha256),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ByteSpanTraceAlignment:
    """Shared byte-row inventory and every aligned parent trace."""

    split_name: str
    raw_prompts: tuple[str, ...]
    segments: tuple[CanonicalByteSegment, ...]
    parents: tuple[AlignedByteSpanTrace, ...]
    corpus_sha256: str
    alignment_fingerprint: str
    omitted_whitespace_byte_count: int
    fingerprint: str

    @property
    def row_count(self) -> int:
        return len(self.segments)

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(segment.row_id for segment in self.segments)

    def parent(self, name: str) -> AlignedByteSpanTrace:
        for parent in self.parents:
            if parent.parent_name == name:
                return parent
        raise KeyError(name)

    def report(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-byte-span-trace-v1",
            "split_name": self.split_name,
            "prompt_count": len(self.raw_prompts),
            "row_count": self.row_count,
            "parent_count": len(self.parents),
            "corpus_sha256": self.corpus_sha256,
            "alignment_fingerprint": self.alignment_fingerprint,
            "omitted_whitespace_byte_count": self.omitted_whitespace_byte_count,
            "segments": [segment.report() for segment in self.segments],
            "parents": [parent.report() for parent in self.parents],
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class _TokenSpan:
    prompt_index: int
    token_index: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class _ValidatedParent:
    name: str
    prompts: tuple[str, ...]
    prompt_bytes: tuple[bytes, ...]
    tokenizer: Any
    input_ids: Any
    attention_mask: Any
    hidden_states: tuple[Any, ...]
    logits: Any | None
    spans: tuple[tuple[_TokenSpan, ...], ...]
    tokenizer_fingerprint: str
    cache_fingerprint: str
    active_token_count: int
    zero_width_token_count: int
    trimmed_whitespace_byte_count: int
    synthetic_prefix_space_count: int


def align_byte_span_traces(
    raw_prompts: Sequence[str],
    parents: Sequence[ByteSpanTraceInput],
    *,
    split_name: str = "calibration",
) -> ByteSpanTraceAlignment:
    """Align heterogeneous tokenizer traces into deterministic byte rows.

    All inputs are detached and overlap pooling is performed on CPU in
    ``float64``.  Parent order does not affect rows, identities, or the final
    fingerprint.
    """

    torch = _require_torch()
    prompts, prompt_bytes = _validate_prompts(raw_prompts)
    split = _nonempty_text(split_name, "split_name")
    supplied = tuple(parents)
    if not supplied:
        raise ByteSpanTraceError("At least one parent trace is required.")
    if any(not isinstance(parent, ByteSpanTraceInput) for parent in supplied):
        raise ByteSpanTraceError("parents must contain ByteSpanTraceInput values.")
    names = tuple(_nonempty_text(parent.parent_name, "parent_name") for parent in supplied)
    if len(set(names)) != len(names):
        raise ByteSpanTraceError("Parent names must be unique.")

    validated = tuple(
        sorted(
            (
                _validate_parent(torch, parent, prompts, prompt_bytes)
                for parent in supplied
            ),
            key=lambda parent: parent.name,
        )
    )
    segments, omitted_whitespace_bytes = _build_segment_inventory(
        prompts,
        prompt_bytes,
        validated,
        split,
    )
    if not segments:
        raise ByteSpanTraceError(
            "The prompt inventory produced no commonly covered byte segments."
        )

    row_ids = tuple(segment.row_id for segment in segments)
    prompt_hashes = tuple(segment.raw_prompt_sha256 for segment in segments)
    corpus_sha256 = _corpus_fingerprint(prompt_bytes)
    aligned_parents = tuple(
        _align_parent(torch, parent, segments, split, row_ids, prompt_hashes, corpus_sha256)
        for parent in validated
    )
    alignment_fingerprints = {
        parent.row_identity.alignment_fingerprint for parent in aligned_parents
    }
    if len(alignment_fingerprints) != 1:
        raise ByteSpanTraceError("Canonical row identities unexpectedly diverged.")
    alignment_fingerprint = next(iter(alignment_fingerprints))
    fingerprint_payload = {
        "schema": "neural-forge-byte-span-trace-v1",
        "split_name": split,
        "corpus_sha256": corpus_sha256,
        "alignment_fingerprint": alignment_fingerprint,
        "segments": [segment.report() for segment in segments],
        "parents": [
            {
                "parent_name": parent.parent_name,
                "fingerprint": parent.fingerprint,
            }
            for parent in aligned_parents
        ],
        "omitted_whitespace_byte_count": omitted_whitespace_bytes,
    }
    return ByteSpanTraceAlignment(
        split_name=split,
        raw_prompts=prompts,
        segments=segments,
        parents=aligned_parents,
        corpus_sha256=corpus_sha256,
        alignment_fingerprint=alignment_fingerprint,
        omitted_whitespace_byte_count=omitted_whitespace_bytes,
        fingerprint=_fingerprint(fingerprint_payload),
    )


def build_byte_span_trace_alignment(
    raw_prompts: Sequence[str],
    parents: Sequence[ByteSpanTraceInput],
    *,
    split_name: str = "calibration",
) -> ByteSpanTraceAlignment:
    """Named constructor alias for :func:`align_byte_span_traces`."""

    return align_byte_span_traces(raw_prompts, parents, split_name=split_name)


def align_full_token_traces_by_byte_span(
    raw_prompts: Sequence[str],
    parents: Sequence[ByteSpanTraceInput],
    *,
    split_name: str = "calibration",
) -> ByteSpanTraceAlignment:
    """Descriptive alias for :func:`align_byte_span_traces`."""

    return align_byte_span_traces(raw_prompts, parents, split_name=split_name)


def canonical_parent_trace_from_byte_alignment(
    alignment: ByteSpanTraceAlignment,
    parent_name: str,
    *,
    depth_cover: CanonicalDepthCover,
    weight: float = 1.0,
) -> CanonicalParentTrace:
    """Convert one aligned parent directly into the canonical kernel contract."""

    if not isinstance(alignment, ByteSpanTraceAlignment):
        raise ByteSpanTraceError("alignment must be a ByteSpanTraceAlignment.")
    if not isinstance(depth_cover, CanonicalDepthCover):
        raise ByteSpanTraceError("depth_cover must be a CanonicalDepthCover.")
    try:
        parent = alignment.parent(_nonempty_text(parent_name, "parent_name"))
    except KeyError as exc:
        raise ByteSpanTraceError(f"Unknown aligned parent {parent_name!r}.") from exc
    if len(parent.aligned_hidden_states) != depth_cover.source_layer_count + 1:
        raise ByteSpanTraceError(
            "depth_cover source layer count does not match aligned hidden boundaries."
        )
    try:
        normalized_weight = float(weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ByteSpanTraceError("weight must be finite and positive.") from exc
    if not math.isfinite(normalized_weight) or normalized_weight <= 0.0:
        raise ByteSpanTraceError("weight must be finite and positive.")
    return CanonicalParentTrace(
        name=parent.parent_name,
        layer_states=parent.aligned_hidden_states,
        row_identity=parent.row_identity,
        depth_cover=depth_cover,
        weight=normalized_weight,
    )


def _validate_parent(
    torch: Any,
    value: ByteSpanTraceInput,
    prompts: tuple[str, ...],
    prompt_bytes: tuple[bytes, ...],
) -> _ValidatedParent:
    name = _nonempty_text(value.parent_name, "parent_name")
    parent_prompts, parent_prompt_bytes = _validate_prompts(value.raw_prompts)
    if parent_prompts != prompts or parent_prompt_bytes != prompt_bytes:
        raise ByteSpanTraceError(
            f"Parent {name!r} raw prompts do not match the ordered shared prompts."
        )
    if value.offset_unit not in {"character", "utf8_byte"}:
        raise ByteSpanTraceError(
            f"Parent {name!r} offset_unit must be character or utf8_byte."
        )
    if value.tokenizer is None or not callable(getattr(value.tokenizer, "decode", None)):
        raise ByteSpanTraceError(f"Parent {name!r} tokenizer must provide decode().")

    input_ids = _integer_tensor(torch, value.input_ids, f"{name}.input_ids", rank=2)
    mask = _binary_mask(torch, value.attention_mask, f"{name}.attention_mask")
    offsets = _integer_tensor(
        torch,
        value.offset_mapping,
        f"{name}.offset_mapping",
        rank=3,
    )
    if int(offsets.shape[-1]) != 2:
        raise ByteSpanTraceError(f"{name}.offset_mapping must end in a size-two axis.")
    batch_shape = tuple(int(part) for part in input_ids.shape)
    if tuple(int(part) for part in mask.shape) != batch_shape:
        raise ByteSpanTraceError(f"{name} attention-mask shape does not match input IDs.")
    if tuple(int(part) for part in offsets.shape[:2]) != batch_shape:
        raise ByteSpanTraceError(f"{name} offset shape does not match input IDs.")
    if batch_shape[0] != len(prompts):
        raise ByteSpanTraceError(f"{name} batch size does not match raw prompts.")
    if batch_shape[1] <= 0:
        raise ByteSpanTraceError(f"{name} sequence length must be positive.")
    if bool((input_ids < 0).any()):
        raise ByteSpanTraceError(f"{name} input IDs must be nonnegative.")

    hidden_states = _hidden_tensors(torch, value.hidden_states, name, batch_shape)
    logits = None
    if value.logits is not None:
        logits = _trace_tensor(torch, value.logits, f"{name}.logits", batch_shape)

    spans_by_prompt: list[tuple[_TokenSpan, ...]] = []
    active_token_count = 0
    zero_width_token_count = 0
    trimmed_whitespace_byte_count = 0
    synthetic_prefix_space_count = 0
    for prompt_index, (prompt, raw_bytes) in enumerate(zip(prompts, prompt_bytes, strict=True)):
        char_boundaries = _character_byte_boundaries(prompt)
        spans: list[_TokenSpan] = []
        previous_end = 0
        valid_ids: list[int] = []
        for token_index in range(batch_shape[1]):
            masked = not bool(mask[prompt_index, token_index].item())
            start = int(offsets[prompt_index, token_index, 0].item())
            end = int(offsets[prompt_index, token_index, 1].item())
            if start < 0 or end < 0 or end < start:
                raise ByteSpanTraceError(
                    f"{name} prompt {prompt_index} token {token_index} has an invalid offset."
                )
            if masked:
                if start != end:
                    raise ByteSpanTraceError(
                        f"{name} prompt {prompt_index} padding token {token_index} "
                        "has a nonzero-width offset."
                    )
                continue
            valid_ids.append(int(input_ids[prompt_index, token_index].item()))
            active_token_count += 1
            if start == end:
                limit = len(prompt) if value.offset_unit == "character" else len(raw_bytes)
                if start > limit:
                    raise ByteSpanTraceError(
                        f"{name} prompt {prompt_index} zero-width token offset is out of range."
                    )
                zero_width_token_count += 1
                continue
            start_byte, end_byte = _to_byte_span(
                start,
                end,
                value.offset_unit,
                char_boundaries,
                len(raw_bytes),
                name,
                prompt_index,
                token_index,
            )
            if start_byte < previous_end:
                raise ByteSpanTraceError(
                    f"{name} prompt {prompt_index} offsets overlap or are nonmonotone "
                    f"at token {token_index}."
                )
            previous_end = end_byte
            spans.append(
                _TokenSpan(
                    prompt_index=prompt_index,
                    token_index=token_index,
                    start_byte=start_byte,
                    end_byte=end_byte,
                )
            )

        synthetic_prefix_space_count += _validate_decode_round_trip(
            value.tokenizer,
            valid_ids,
            prompt,
            name,
            prompt_index,
        )
        covered = _covered_byte_mask(len(raw_bytes), spans)
        missing_non_whitespace = _missing_non_whitespace_ranges(
            prompt,
            char_boundaries,
            covered,
        )
        if missing_non_whitespace:
            raise ByteSpanTraceError(
                f"{name} prompt {prompt_index} leaves non-whitespace UTF-8 bytes "
                f"uncovered: {missing_non_whitespace}."
            )
        trimmed_whitespace_byte_count += sum(not present for present in covered)
        spans_by_prompt.append(tuple(spans))

    tokenizer_fingerprint = _tokenizer_fingerprint(value)
    cache_fingerprint = _cache_fingerprint(
        torch,
        value,
        prompts,
        input_ids,
        offsets,
        mask,
        hidden_states,
        logits,
        tokenizer_fingerprint,
    )
    return _ValidatedParent(
        name=name,
        prompts=prompts,
        prompt_bytes=prompt_bytes,
        tokenizer=value.tokenizer,
        input_ids=input_ids,
        attention_mask=mask,
        hidden_states=hidden_states,
        logits=logits,
        spans=tuple(spans_by_prompt),
        tokenizer_fingerprint=tokenizer_fingerprint,
        cache_fingerprint=cache_fingerprint,
        active_token_count=active_token_count,
        zero_width_token_count=zero_width_token_count,
        trimmed_whitespace_byte_count=trimmed_whitespace_byte_count,
        synthetic_prefix_space_count=synthetic_prefix_space_count,
    )


def _build_segment_inventory(
    prompts: tuple[str, ...],
    prompt_bytes: tuple[bytes, ...],
    parents: tuple[_ValidatedParent, ...],
    split_name: str,
) -> tuple[tuple[CanonicalByteSegment, ...], int]:
    segments: list[CanonicalByteSegment] = []
    omitted_whitespace_bytes = 0
    for prompt_index, (prompt, raw_bytes) in enumerate(zip(prompts, prompt_bytes, strict=True)):
        boundaries = {0, len(raw_bytes)}
        for parent in parents:
            for span in parent.spans[prompt_index]:
                boundaries.add(span.start_byte)
                boundaries.add(span.end_byte)
        ordered = sorted(boundaries)
        char_boundaries = set(_character_byte_boundaries(prompt))
        if any(boundary not in char_boundaries for boundary in ordered):
            raise ByteSpanTraceError(
                f"Prompt {prompt_index} contains a segment boundary inside a UTF-8 codepoint."
            )
        prompt_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        for start, end in zip(ordered, ordered[1:]):
            if end <= start:
                continue
            segment_bytes = raw_bytes[start:end]
            segment_text = segment_bytes.decode("utf-8", errors="strict")
            covered_by_all = all(
                _span_coverage(parent.spans[prompt_index], start, end) == end - start
                for parent in parents
            )
            if not covered_by_all:
                if segment_text.isspace():
                    omitted_whitespace_bytes += end - start
                    continue
                raise ByteSpanTraceError(
                    f"Prompt {prompt_index} byte segment [{start}, {end}) is not "
                    "covered by every parent."
                )
            segment_sha256 = hashlib.sha256(segment_bytes).hexdigest()
            row_id = (
                f"{split_name}:prompt-{prompt_index:08d}:"
                f"bytes-{start:08d}-{end:08d}:{segment_sha256[:16]}"
            )
            segments.append(
                CanonicalByteSegment(
                    row_id=row_id,
                    prompt_index=prompt_index,
                    start_byte=start,
                    end_byte=end,
                    utf8_hex=segment_bytes.hex(),
                    segment_sha256=segment_sha256,
                    raw_prompt_sha256=prompt_sha256,
                )
            )
    return tuple(segments), omitted_whitespace_bytes


def _align_parent(
    torch: Any,
    parent: _ValidatedParent,
    segments: tuple[CanonicalByteSegment, ...],
    split_name: str,
    row_ids: tuple[str, ...],
    prompt_hashes: tuple[str, ...],
    corpus_sha256: str,
) -> AlignedByteSpanTrace:
    aligned_hidden = tuple(
        _pool_matrix(torch, matrix, parent.spans, segments)
        for matrix in parent.hidden_states
    )
    aligned_logits = (
        None
        if parent.logits is None
        else _pool_matrix(torch, parent.logits, parent.spans, segments)
    )
    row_identity = CanonicalRowIdentity(
        split_name=split_name,
        pooling="token_rows",
        row_ids=row_ids,
        raw_prompt_sha256=prompt_hashes,
        corpus_sha256=corpus_sha256,
        tokenizer_fingerprint=parent.tokenizer_fingerprint,
        cache_fingerprint=parent.cache_fingerprint,
    )
    tensors = aligned_hidden + (() if aligned_logits is None else (aligned_logits,))
    tensor_hashes = tuple(_tensor_sha256(torch, tensor) for tensor in tensors)
    payload = {
        "schema": "neural-forge-aligned-byte-span-parent-v1",
        "parent_name": parent.name,
        "row_identity": row_identity.report(),
        "tensor_sha256": list(tensor_hashes),
        "active_token_count": parent.active_token_count,
        "zero_width_token_count": parent.zero_width_token_count,
        "trimmed_whitespace_byte_count": parent.trimmed_whitespace_byte_count,
        "synthetic_prefix_space_count": parent.synthetic_prefix_space_count,
    }
    return AlignedByteSpanTrace(
        parent_name=parent.name,
        row_identity=row_identity,
        aligned_hidden_states=aligned_hidden,
        aligned_logits=aligned_logits,
        tokenizer_fingerprint=parent.tokenizer_fingerprint,
        cache_fingerprint=parent.cache_fingerprint,
        active_token_count=parent.active_token_count,
        zero_width_token_count=parent.zero_width_token_count,
        trimmed_whitespace_byte_count=parent.trimmed_whitespace_byte_count,
        synthetic_prefix_space_count=parent.synthetic_prefix_space_count,
        tensor_sha256=tensor_hashes,
        fingerprint=_fingerprint(payload),
    )


def _pool_matrix(
    torch: Any,
    matrix: Any,
    spans_by_prompt: tuple[tuple[_TokenSpan, ...], ...],
    segments: tuple[CanonicalByteSegment, ...],
) -> Any:
    source = matrix.detach().to(device="cpu", dtype=torch.float64)
    rows = []
    for segment in segments:
        weighted = torch.zeros(int(source.shape[-1]), dtype=torch.float64)
        denominator = 0
        for span in spans_by_prompt[segment.prompt_index]:
            overlap = min(segment.end_byte, span.end_byte) - max(
                segment.start_byte,
                span.start_byte,
            )
            if overlap <= 0:
                continue
            weighted.add_(source[segment.prompt_index, span.token_index], alpha=overlap)
            denominator += overlap
        if denominator != segment.byte_length:
            raise ByteSpanTraceError(
                f"Parent trace does not exactly cover canonical row {segment.row_id!r}."
            )
        rows.append(weighted.div(float(denominator)))
    return torch.stack(rows, dim=0).contiguous()


def _hidden_tensors(
    torch: Any,
    value: Any,
    parent_name: str,
    batch_shape: tuple[int, int],
) -> tuple[Any, ...]:
    if _is_tensor(torch, value):
        supplied = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        supplied = tuple(value)
    else:
        raise ByteSpanTraceError(
            f"{parent_name}.hidden_states must be a tensor or tensor sequence."
        )
    if not supplied:
        raise ByteSpanTraceError(f"{parent_name}.hidden_states cannot be empty.")
    return tuple(
        _trace_tensor(torch, tensor, f"{parent_name}.hidden_states[{index}]", batch_shape)
        for index, tensor in enumerate(supplied)
    )


def _trace_tensor(
    torch: Any,
    value: Any,
    name: str,
    batch_shape: tuple[int, int],
) -> Any:
    if not _is_tensor(torch, value):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ByteSpanTraceError(f"{name} is not tensor-like.") from exc
    if value.ndim != 3 or tuple(int(part) for part in value.shape[:2]) != batch_shape:
        raise ByteSpanTraceError(
            f"{name} must have shape [batch, sequence, width] matching input IDs."
        )
    if int(value.shape[-1]) <= 0 or not (value.is_floating_point() or value.is_complex()):
        raise ByteSpanTraceError(f"{name} must have a positive floating-point width.")
    if value.is_complex():
        raise ByteSpanTraceError(f"{name} complex tensors are unsupported.")
    if not bool(torch.isfinite(value).all()):
        raise ByteSpanTraceError(f"{name} contains non-finite values.")
    return value.detach()


def _integer_tensor(torch: Any, value: Any, name: str, *, rank: int) -> Any:
    try:
        tensor = value.detach().to(device="cpu") if _is_tensor(torch, value) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ByteSpanTraceError(f"{name} is not tensor-like.") from exc
    if tensor.ndim != rank:
        raise ByteSpanTraceError(f"{name} must have rank {rank}.")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise ByteSpanTraceError(f"{name} must contain integers.")
    return tensor.to(dtype=torch.int64).contiguous()


def _binary_mask(torch: Any, value: Any, name: str) -> Any:
    try:
        tensor = value.detach().to(device="cpu") if _is_tensor(torch, value) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ByteSpanTraceError(f"{name} is not tensor-like.") from exc
    if tensor.ndim != 2:
        raise ByteSpanTraceError(f"{name} must have rank 2.")
    if tensor.is_complex() or not bool(((tensor == 0) | (tensor == 1)).all()):
        raise ByteSpanTraceError(f"{name} must contain only zero and one.")
    return tensor.to(dtype=torch.bool).contiguous()


def _to_byte_span(
    start: int,
    end: int,
    unit: OffsetUnit,
    char_boundaries: tuple[int, ...],
    byte_length: int,
    parent_name: str,
    prompt_index: int,
    token_index: int,
) -> tuple[int, int]:
    if unit == "character":
        if end >= len(char_boundaries):
            raise ByteSpanTraceError(
                f"{parent_name} prompt {prompt_index} token {token_index} character "
                "offset is out of range."
            )
        return char_boundaries[start], char_boundaries[end]
    if end > byte_length:
        raise ByteSpanTraceError(
            f"{parent_name} prompt {prompt_index} token {token_index} byte offset "
            "is out of range."
        )
    boundaries = set(char_boundaries)
    if start not in boundaries or end not in boundaries:
        raise ByteSpanTraceError(
            f"{parent_name} prompt {prompt_index} token {token_index} splits a UTF-8 codepoint."
        )
    return start, end


def _validate_decode_round_trip(
    tokenizer: Any,
    valid_ids: list[int],
    prompt: str,
    parent_name: str,
    prompt_index: int,
) -> int:
    if not valid_ids:
        if prompt:
            raise ByteSpanTraceError(
                f"{parent_name} prompt {prompt_index} has no valid token IDs."
            )
        return 0
    decoded = _decode_ids(tokenizer, valid_ids)
    try:
        decoded_bytes = decoded.encode("utf-8", errors="strict")
        prompt_bytes = prompt.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ByteSpanTraceError(
            f"{parent_name} prompt {prompt_index} decode is not valid UTF-8."
        ) from exc
    if decoded_bytes == prompt_bytes:
        return 0
    if (
        _is_byte_level(tokenizer)
        and not prompt.startswith(tuple(_WHITESPACE_PREFIXES))
        and decoded_bytes == b" " + prompt_bytes
    ):
        return 1
    raise ByteSpanTraceError(
        f"{parent_name} prompt {prompt_index} tokenizer decode is lossy or does not "
        "round-trip the exact raw UTF-8 bytes."
    )


_WHITESPACE_PREFIXES = (" ", "\t", "\n", "\r", "\v", "\f")


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    attempts = (
        {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
        {"skip_special_tokens": True},
    )
    last_type_error: TypeError | None = None
    for kwargs in attempts:
        try:
            decoded = tokenizer.decode(token_ids, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue
        except Exception as exc:
            raise ByteSpanTraceError(
                f"Tokenizer decode failed with {type(exc).__name__}."
            ) from exc
        if not isinstance(decoded, str):
            raise ByteSpanTraceError("Tokenizer decode must return text.")
        return decoded
    special_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "all_special_ids", ()) or ())
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    }
    filtered = [token_id for token_id in token_ids if token_id not in special_ids]
    try:
        decoded = tokenizer.decode(filtered)
    except Exception as exc:
        error = type(last_type_error).__name__ if last_type_error else type(exc).__name__
        raise ByteSpanTraceError(f"Tokenizer decode failed with {error}.") from exc
    if not isinstance(decoded, str):
        raise ByteSpanTraceError("Tokenizer decode must return text.")
    return decoded


def _is_byte_level(tokenizer: Any) -> bool:
    if bool(getattr(tokenizer, "is_byte_level", False)):
        return True
    candidates = [
        getattr(tokenizer, "backend_tokenizer", None),
        getattr(tokenizer, "_tokenizer", None),
        getattr(tokenizer, "pre_tokenizer", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        serializer = getattr(candidate, "to_str", None)
        try:
            text = serializer() if callable(serializer) else repr(candidate)
        except Exception:
            continue
        if "ByteLevel" in text or "bytelevel" in text.lower():
            return True
    return "bytelevel" in type(tokenizer).__name__.lower()


def _validate_prompts(values: Sequence[str]) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ByteSpanTraceError("raw_prompts must be an ordered text sequence.")
    prompts = tuple(values)
    if not prompts:
        raise ByteSpanTraceError("raw_prompts cannot be empty.")
    encoded = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise ByteSpanTraceError(f"raw_prompts[{index}] must be text.")
        try:
            encoded.append(prompt.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise ByteSpanTraceError(
                f"raw_prompts[{index}] is not valid UTF-8 text."
            ) from exc
    return prompts, tuple(encoded)


def _character_byte_boundaries(prompt: str) -> tuple[int, ...]:
    boundaries = [0]
    size = 0
    for character in prompt:
        size += len(character.encode("utf-8", errors="strict"))
        boundaries.append(size)
    return tuple(boundaries)


def _covered_byte_mask(length: int, spans: Sequence[_TokenSpan]) -> list[bool]:
    covered = [False] * length
    for span in spans:
        for index in range(span.start_byte, span.end_byte):
            if covered[index]:
                raise ByteSpanTraceError("Validated token byte spans unexpectedly overlap.")
            covered[index] = True
    return covered


def _missing_non_whitespace_ranges(
    prompt: str,
    char_boundaries: tuple[int, ...],
    covered: list[bool],
) -> tuple[tuple[int, int], ...]:
    missing = []
    for char_index, character in enumerate(prompt):
        if character.isspace():
            continue
        start = char_boundaries[char_index]
        end = char_boundaries[char_index + 1]
        if not all(covered[start:end]):
            missing.append((start, end))
    return tuple(missing)


def _span_coverage(spans: Sequence[_TokenSpan], start: int, end: int) -> int:
    return sum(
        max(0, min(end, span.end_byte) - max(start, span.start_byte))
        for span in spans
    )


def _tokenizer_fingerprint(value: ByteSpanTraceInput) -> str:
    if value.tokenizer_fingerprint is not None:
        return _sha256(value.tokenizer_fingerprint, "tokenizer_fingerprint")
    tokenizer = value.tokenizer
    payload: dict[str, Any] = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "all_special_ids": list(getattr(tokenizer, "all_special_ids", ()) or ()),
        "is_byte_level": _is_byte_level(tokenizer),
    }
    backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(
        tokenizer,
        "_tokenizer",
        None,
    )
    serializer = getattr(backend, "to_str", None)
    if callable(serializer):
        try:
            payload["backend"] = serializer()
        except Exception:
            pass
    vocabulary = getattr(tokenizer, "get_vocab", None)
    if callable(vocabulary):
        try:
            vocab = vocabulary()
        except Exception:
            vocab = None
        if isinstance(vocab, dict):
            payload["vocabulary"] = sorted(
                (str(token), int(token_id)) for token, token_id in vocab.items()
            )
    custom = getattr(tokenizer, "fingerprint_payload", None)
    if custom is not None:
        payload["custom"] = _jsonable(custom)
    return _fingerprint(payload)


def _cache_fingerprint(
    torch: Any,
    value: ByteSpanTraceInput,
    prompts: tuple[str, ...],
    input_ids: Any,
    offsets: Any,
    mask: Any,
    hidden_states: tuple[Any, ...],
    logits: Any | None,
    tokenizer_fingerprint: str,
) -> str:
    if value.cache_fingerprint is not None:
        return _sha256(value.cache_fingerprint, "cache_fingerprint")
    payload = {
        "schema": "neural-forge-byte-span-trace-cache-v1",
        "parent_name": value.parent_name,
        "raw_prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
        ],
        "offset_unit": value.offset_unit,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "input_ids_sha256": _tensor_sha256(torch, input_ids),
        "offset_mapping_sha256": _tensor_sha256(torch, offsets),
        "attention_mask_sha256": _tensor_sha256(torch, mask),
        "hidden_states_sha256": [
            _tensor_sha256(torch, tensor) for tensor in hidden_states
        ],
        "logits_sha256": None if logits is None else _tensor_sha256(torch, logits),
    }
    return _fingerprint(payload)


def _corpus_fingerprint(prompt_bytes: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-byte-span-corpus-v1\0")
    for raw in prompt_bytes:
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _tensor_sha256(torch: Any, value: Any) -> str:
    tensor = value.detach().contiguous().to(device="cpu")
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    digest.update(raw)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ByteSpanTraceError("Fingerprint payload contains a non-finite float.")
        return value
    if isinstance(value, bytes):
        return {"utf8_hex": value.hex()}
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ByteSpanTraceError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ByteSpanTraceError(f"{name} must be nonempty text.")
    return value.strip()


def _is_tensor(torch: Any, value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError("Byte-span trace alignment requires PyTorch.") from exc
    return torch


__all__ = [
    "AlignedByteSpanTrace",
    "ByteSpanTraceAlignment",
    "ByteSpanTraceError",
    "ByteSpanTraceInput",
    "CanonicalByteSegment",
    "OffsetUnit",
    "ParentByteSpanTrace",
    "align_byte_span_traces",
    "align_full_token_traces_by_byte_span",
    "build_byte_span_trace_alignment",
    "canonical_parent_trace_from_byte_alignment",
]
