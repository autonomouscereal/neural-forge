from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_byte_span_trace import (  # noqa: E402
    ByteSpanTraceError,
    ByteSpanTraceInput,
    align_byte_span_traces,
    canonical_parent_trace_from_byte_alignment,
)
from neural_forge.hf_canonical_kernel import CanonicalDepthCover  # noqa: E402


class FakeTokenizer:
    def __init__(
        self,
        decoded: dict[tuple[int, ...], str],
        *,
        name: str,
        byte_level: bool = False,
        special_ids: tuple[int, ...] = (),
    ) -> None:
        self._decoded = decoded
        self.name_or_path = name
        self.is_byte_level = byte_level
        self.all_special_ids = list(special_ids)
        self.fingerprint_payload = {"name": name, "decoded": sorted(decoded.items())}

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        ids = tuple(
            token_id
            for token_id in token_ids
            if not skip_special_tokens or token_id not in self.all_special_ids
        )
        return self._decoded[ids]


def _semantic_tensor(prompts, offsets, widths, *, sequence_length):
    result = torch.zeros((len(prompts), sequence_length, widths), dtype=torch.float64)
    for batch_index, prompt in enumerate(prompts):
        boundaries = [0]
        for character in prompt:
            boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
        for token_index, (start, end) in enumerate(offsets[batch_index]):
            if end <= start:
                continue
            raw = prompt.encode("utf-8")[boundaries[start] : boundaries[end]]
            # Every character in a test token carries one semantic signal.  The
            # two tokenizations differ only in how equal-signal bytes are grouped.
            semantic_bytes = bytes(byte for byte in raw if not chr(byte).isspace())
            signal = (
                float(sum(semantic_bytes) / len(semantic_bytes))
                if semantic_bytes
                else 0.0
            )
            result[batch_index, token_index] = torch.tensor(
                [signal, signal * 2, -signal][:widths],
                dtype=torch.float64,
            )
    return result


def _parents():
    prompts = ("aa bb", "éé! 東京")
    # Parent A groups repeated semantic units. Parent B splits them. Whitespace
    # is included by A and trimmed from B to exercise ByteLevel offsets.
    offsets_a = (
        ((0, 0), (0, 2), (2, 5), (0, 0), (0, 0)),
        ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6)),
    )
    offsets_b = (
        ((0, 1), (1, 2), (3, 5), (0, 0), (0, 0)),
        ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6)),
    )
    ids_a = torch.tensor(((99, 1, 2, 0, 0), (3, 4, 5, 6, 7)))
    ids_b = torch.tensor(((10, 11, 12, 0, 0), (13, 14, 15, 16, 17)))
    mask_a = torch.tensor(((1, 1, 1, 0, 0), (1, 1, 1, 1, 1)))
    mask_b = torch.tensor(((1, 1, 1, 0, 0), (1, 1, 1, 1, 1)))
    tokenizer_a = FakeTokenizer(
        {(1, 2): "aa bb", (3, 4, 5, 6, 7): "éé! 東京"},
        name="grouped-bytelevel",
        byte_level=True,
        special_ids=(99,),
    )
    tokenizer_b = FakeTokenizer(
        {(10, 11, 12): "aa bb", (13, 14, 15, 16, 17): "éé! 東京"},
        name="split-tokenizer",
    )
    hidden_a = _semantic_tensor(prompts, offsets_a, 3, sequence_length=5)
    hidden_b = _semantic_tensor(prompts, offsets_b, 3, sequence_length=5)
    # Repeated letters/codepoints have equal byte sums, so union-boundary rows
    # recover equal semantic signals despite deliberately different segmentation.
    return prompts, (
        ByteSpanTraceInput(
            parent_name="parent-a",
            raw_prompts=prompts,
            tokenizer=tokenizer_a,
            input_ids=ids_a,
            offset_mapping=torch.tensor(offsets_a),
            attention_mask=mask_a,
            hidden_states=(hidden_a, hidden_a + 1),
            logits=hidden_a * 0.5,
        ),
        ByteSpanTraceInput(
            parent_name="parent-b",
            raw_prompts=prompts,
            tokenizer=tokenizer_b,
            input_ids=ids_b,
            offset_mapping=torch.tensor(offsets_b),
            attention_mask=mask_b,
            hidden_states=(hidden_b, hidden_b + 1),
            logits=hidden_b * 0.5,
        ),
    )


def test_different_segmentations_align_ascii_and_multibyte_unicode() -> None:
    prompts, parents = _parents()
    alignment = align_byte_span_traces(prompts, parents, split_name="heldout")

    assert alignment.row_count == 8
    assert b"".join(bytes.fromhex(row.utf8_hex) for row in alignment.segments) == (
        b"aabb" + "éé!東京".encode("utf-8")
    )
    left = alignment.parent("parent-a")
    right = alignment.parent("parent-b")
    assert len(left.aligned_hidden_states) == 2
    assert torch.equal(left.aligned_hidden_states[0], right.aligned_hidden_states[0])
    assert torch.equal(left.aligned_hidden_states[1], right.aligned_hidden_states[1])
    assert torch.equal(left.aligned_logits, right.aligned_logits)
    assert left.row_identity.alignment_fingerprint == right.row_identity.alignment_fingerprint
    assert left.row_identity.pooling == "token_rows"
    assert alignment.omitted_whitespace_byte_count == 2


def test_alignment_is_parent_order_invariant_and_fingerprinted() -> None:
    prompts, parents = _parents()
    first = align_byte_span_traces(prompts, parents)
    second = align_byte_span_traces(prompts, tuple(reversed(parents)))

    assert first.fingerprint == second.fingerprint
    assert first.alignment_fingerprint == second.alignment_fingerprint
    assert first.row_ids == second.row_ids
    assert [parent.parent_name for parent in first.parents] == ["parent-a", "parent-b"]
    assert len(first.fingerprint) == 64
    assert first.report()["fingerprint"] == first.fingerprint


def test_alignment_converts_directly_to_canonical_parent_trace() -> None:
    prompts, parents = _parents()
    alignment = align_byte_span_traces(prompts, parents)
    trace = canonical_parent_trace_from_byte_alignment(
        alignment,
        "parent-a",
        depth_cover=CanonicalDepthCover((0, 1)),
        weight=0.4,
    )

    assert trace.name == "parent-a"
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            trace.layer_states,
            alignment.parent("parent-a").aligned_hidden_states,
            strict=True,
        )
    )
    assert trace.row_identity.alignment_fingerprint == alignment.alignment_fingerprint
    assert trace.depth_cover.boundary_indices == (0, 1)
    assert trace.weight == 0.4


def test_bytelevel_synthetic_leading_space_is_accepted_but_recorded() -> None:
    prompt = ("hello",)
    tokenizer = FakeTokenizer(
        {(1,): " hello"},
        name="prefix-space",
        byte_level=True,
    )
    trace = ByteSpanTraceInput(
        parent_name="bytelevel",
        raw_prompts=prompt,
        tokenizer=tokenizer,
        input_ids=torch.tensor([[1]]),
        offset_mapping=torch.tensor([[[0, 5]]]),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        hidden_states=torch.ones((1, 1, 2)),
    )

    aligned = align_byte_span_traces(prompt, (trace,)).parents[0]

    assert aligned.synthetic_prefix_space_count == 1
    assert tuple(aligned.hidden_states.shape) == (1, 2)


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        (((0, 2), (1, 3)), "overlap or are nonmonotone"),
        (((1, 2), (0, 1)), "overlap or are nonmonotone"),
        (((0, 1), (2, 3)), "non-whitespace UTF-8 bytes uncovered"),
    ],
)
def test_malformed_or_incomplete_offsets_are_rejected(offsets, message) -> None:
    prompt = ("abc",)
    trace = ByteSpanTraceInput(
        parent_name="bad-offsets",
        raw_prompts=prompt,
        tokenizer=FakeTokenizer({(1, 2): "abc"}, name="bad"),
        input_ids=torch.tensor([[1, 2]]),
        offset_mapping=torch.tensor([offsets]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        hidden_states=torch.ones((1, 2, 2)),
    )

    with pytest.raises(ByteSpanTraceError, match=message):
        align_byte_span_traces(prompt, (trace,))


def test_lossy_decode_and_prompt_mismatch_are_rejected() -> None:
    prompt = ("café",)
    base = dict(
        parent_name="lossy",
        raw_prompts=prompt,
        tokenizer=FakeTokenizer({(1,): "cafe"}, name="lossy"),
        input_ids=torch.tensor([[1]]),
        offset_mapping=torch.tensor([[[0, 4]]]),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        hidden_states=torch.ones((1, 1, 2)),
    )
    with pytest.raises(ByteSpanTraceError, match="decode is lossy"):
        align_byte_span_traces(prompt, (ByteSpanTraceInput(**base),))

    mismatch = ByteSpanTraceInput(**{**base, "raw_prompts": ("other",)})
    with pytest.raises(ByteSpanTraceError, match="do not match"):
        align_byte_span_traces(prompt, (mismatch,))


def test_zero_width_special_and_padding_tokens_are_ignored() -> None:
    prompt = ("ok",)
    tokenizer = FakeTokenizer(
        {(1,): "ok"},
        name="specials",
        special_ids=(99,),
    )
    trace = ByteSpanTraceInput(
        parent_name="special-parent",
        raw_prompts=prompt,
        tokenizer=tokenizer,
        input_ids=torch.tensor([[99, 1, 0]]),
        offset_mapping=torch.tensor([[[0, 0], [0, 2], [0, 0]]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
        hidden_states=torch.tensor([[[999.0], [7.0], [-999.0]]]),
    )

    aligned = align_byte_span_traces(prompt, (trace,)).parents[0]

    assert aligned.zero_width_token_count == 1
    assert aligned.active_token_count == 2
    assert aligned.hidden_states.tolist() == [[7.0]]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attention_mask", torch.ones((1, 3)), "attention-mask shape"),
        ("hidden_states", torch.ones((1, 3, 2)), "must have shape"),
        ("logits", torch.ones((2, 2, 4)), "must have shape"),
    ],
)
def test_hidden_mask_and_logit_shape_mismatches_fail_closed(field, value, message) -> None:
    prompt = ("ok",)
    kwargs = {
        "parent_name": "shape",
        "raw_prompts": prompt,
        "tokenizer": FakeTokenizer({(1, 2): "ok"}, name="shape"),
        "input_ids": torch.tensor([[1, 2]]),
        "offset_mapping": torch.tensor([[[0, 1], [1, 2]]]),
        "attention_mask": torch.ones((1, 2)),
        "hidden_states": torch.ones((1, 2, 2)),
        "logits": torch.ones((1, 2, 4)),
    }
    kwargs[field] = value

    with pytest.raises(ByteSpanTraceError, match=message):
        align_byte_span_traces(prompt, (ByteSpanTraceInput(**kwargs),))


def test_utf8_byte_offsets_cannot_split_a_multibyte_codepoint() -> None:
    prompt = ("é",)
    trace = ByteSpanTraceInput(
        parent_name="split-codepoint",
        raw_prompts=prompt,
        tokenizer=FakeTokenizer({(1,): "é"}, name="bytes"),
        input_ids=torch.tensor([[1]]),
        offset_mapping=torch.tensor([[[0, 1]]]),
        attention_mask=torch.ones((1, 1)),
        hidden_states=torch.ones((1, 1, 2)),
        offset_unit="utf8_byte",
    )

    with pytest.raises(ByteSpanTraceError, match="splits a UTF-8 codepoint"):
        align_byte_span_traces(prompt, (trace,))


def test_explicit_canonical_hashes_are_preserved() -> None:
    prompt = ("x",)
    tokenizer_hash = hashlib.sha256(b"tokenizer").hexdigest()
    cache_hash = hashlib.sha256(b"cache").hexdigest()
    trace = ByteSpanTraceInput(
        parent_name="hashed",
        raw_prompts=prompt,
        tokenizer=FakeTokenizer({(1,): "x"}, name="hash"),
        input_ids=torch.tensor([[1]]),
        offset_mapping=torch.tensor([[[0, 1]]]),
        attention_mask=torch.ones((1, 1)),
        hidden_states=torch.ones((1, 1, 1)),
        tokenizer_fingerprint=tokenizer_hash,
        cache_fingerprint=cache_hash,
    )

    aligned = align_byte_span_traces(prompt, (trace,)).parents[0]

    assert aligned.row_identity.tokenizer_fingerprint == tokenizer_hash
    assert aligned.row_identity.cache_fingerprint == cache_hash
