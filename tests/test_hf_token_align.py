from __future__ import annotations

import json
from typing import Any

import pytest

from neural_forge.hf_token_align import AlignmentClass
from neural_forge.hf_token_align import AlignmentRejectedError
from neural_forge.hf_token_align import StrictTokenTransportRejectedError
from neural_forge.hf_token_align import TransportAlignmentThresholds
from neural_forge.hf_token_align import build_strict_token_transport_profile
from neural_forge.hf_token_align import build_token_alignment
from neural_forge.hf_token_align import evaluate_alignment_for_transport
from neural_forge.hf_token_align import require_alignment_for_transport


class FakeTokenizer:
    def __init__(
        self,
        surfaces: list[str | Exception | object],
        *,
        encodings: dict[str, list[int]] | None = None,
        decode_overrides: dict[tuple[int, ...], str] | None = None,
        special_ids: tuple[int, ...] = (),
        vocab_ids: tuple[int, ...] | None = None,
    ) -> None:
        self.surfaces = list(surfaces)
        self.encodings = dict(encodings or {})
        self.decode_overrides = dict(decode_overrides or {})
        self.all_special_ids = list(special_ids)
        self.vocab_ids = tuple(
            range(len(self.surfaces)) if vocab_ids is None else vocab_ids
        )

    def __len__(self) -> int:
        return len(self.surfaces)

    def decode(self, token_ids: list[int], **_: Any) -> str:
        ids = tuple(int(token_id) for token_id in token_ids)
        if ids in self.decode_overrides:
            return self.decode_overrides[ids]
        pieces = []
        for token_id in ids:
            surface = self.surfaces[token_id]
            if isinstance(surface, Exception):
                raise surface
            if not isinstance(surface, str):
                return surface  # type: ignore[return-value]
            pieces.append(surface)
        return "".join(pieces)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text not in self.encodings:
            raise KeyError(text)
        return list(self.encodings[text])

    def get_vocab(self) -> dict[str, int]:
        return {
            f"token-{position}": token_id
            for position, token_id in enumerate(self.vocab_ids)
        }


def _bytelevel_bpe_tokenizer() -> Any:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: token_id for token_id, token in enumerate(alphabet)}
    merges = [("a", "b"), ("ab", "c"), ("x", "y")]
    for token in ("ab", "abc", "xy"):
        vocabulary[token] = len(vocabulary)

    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab=vocabulary, merges=merges)
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def _pruned_bytelevel_descendant(donor: Any) -> tuple[Any, tuple[int, ...]]:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    data = json.loads(donor.backend_tokenizer.to_str())
    removed_id = donor.get_vocab()["xy"]
    selected_ids = tuple(
        token_id for token_id in range(len(donor)) if token_id != removed_id
    )
    id_map = {donor_id: target_id for target_id, donor_id in enumerate(selected_ids)}
    model = data["model"]
    model["vocab"] = {
        token: id_map[token_id]
        for token, token_id in model["vocab"].items()
        if token_id in id_map
    }
    retained_tokens = set(model["vocab"])
    model["merges"] = [
        merge
        for merge in model["merges"]
        if merge[0] in retained_tokens
        and merge[1] in retained_tokens
        and merge[0] + merge[1] in retained_tokens
    ]
    data["added_tokens"] = [
        {**item, "id": id_map[item["id"]]}
        for item in data["added_tokens"]
        if item["id"] in id_map
    ]

    backend = tokenizers.Tokenizer.from_str(
        json.dumps(data, ensure_ascii=False, sort_keys=True)
    )
    target = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        **donor.special_tokens_map,
    )
    return target, selected_ids


def _bytelevel_semantic_variant(donor: Any, component: str) -> Any:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    data = json.loads(donor.backend_tokenizer.to_str())
    if component == "model":
        data["model"]["dropout"] = 0.1
    elif component == "pre_tokenizer":
        data["pre_tokenizer"]["add_prefix_space"] = True
    elif component == "decoder":
        data["decoder"]["trim_offsets"] = False
    elif component == "normalizer":
        data["normalizer"] = {"type": "NFC"}
    else:  # pragma: no cover - guarded by test parameters.
        raise AssertionError(component)

    backend = tokenizers.Tokenizer.from_str(
        json.dumps(data, ensure_ascii=False, sort_keys=True)
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        **donor.special_tokens_map,
    )


def test_exact_alignment_preserves_leading_and_trailing_whitespace() -> None:
    donor = FakeTokenizer(["hello", " hello", "hello "])
    target = FakeTokenizer([" hello", "hello ", "hello"])

    report = build_token_alignment(donor, target)

    assert [item.target_text for item in report.alignments] == [
        " hello",
        "hello ",
        "hello",
    ]
    assert [item.classification for item in report.alignments] == [
        AlignmentClass.EXACT_TOKEN,
        AlignmentClass.EXACT_TOKEN,
        AlignmentClass.EXACT_TOKEN,
    ]
    assert [item.candidates[0].donor_token_ids for item in report.alignments] == [
        (1,),
        (2,),
        (0,),
    ]
    assert report.alignments[0].target_utf8_hex == b" hello".hex()


def test_multibyte_utf8_span_round_trips_exact_bytes() -> None:
    donor = FakeTokenizer(
        ["caf", "é", "!"],
        encodings={"café!": [0, 1, 2]},
    )
    target = FakeTokenizer(["café!"])

    alignment = build_token_alignment(donor, target).alignments[0]

    assert alignment.classification is AlignmentClass.UTF8_BYTE_SPAN
    assert alignment.target_utf8_hex == "café!".encode("utf-8").hex()
    assert [candidate.donor_token_ids for candidate in alignment.candidates] == [
        (0, 1, 2)
    ]
    assert alignment.candidates[0].origins == (
        "decoded-utf8-segmentation",
        "donor-encode-roundtrip",
    )


def test_context_dependent_utf8_byte_fallback_uses_full_sequence_roundtrip() -> None:
    donor = FakeTokenizer(
        ["\ufffd", "\ufffd"],
        encodings={"é": [0, 1]},
        decode_overrides={(0, 1): "é"},
    )
    target = FakeTokenizer(["é"])

    report = build_token_alignment(donor, target)
    alignment = report.alignments[0]

    assert alignment.classification is AlignmentClass.UTF8_BYTE_SPAN
    assert alignment.candidates[0].donor_token_ids == (0, 1)
    assert alignment.candidates[0].origins == ("donor-encode-roundtrip",)
    assert [issue.reason for issue in report.donor_surface_issues] == [
        "replacement-character-is-not-byte-faithful",
        "replacement-character-is-not-byte-faithful",
    ]


def test_duplicate_decoded_bytes_record_every_exact_candidate() -> None:
    donor = FakeTokenizer([" same", " same", "other"])
    target = FakeTokenizer([" same"])

    report = build_token_alignment(donor, target)
    alignment = report.alignments[0]

    assert alignment.classification is AlignmentClass.EXACT_TOKEN
    assert [candidate.donor_token_ids for candidate in alignment.candidates] == [
        (0,),
        (1,),
    ]
    assert alignment.is_ambiguous is True
    assert report.donor_collision_surface_count == 1
    assert report.donor_collision_token_count == 2
    assert report.coverage.ambiguous_count == 1
    assert report.coverage.unique_coverage_fraction == 0.0


def test_invalid_and_unrepresentable_tokens_are_explicitly_unalignable() -> None:
    donor = FakeTokenizer(["ok", ValueError("bad donor"), "", "\ufffd"])
    target = FakeTokenizer(
        ["ok", "\ud800", "", "missing", ValueError("bad target"), object()]
    )

    report = build_token_alignment(donor, target)

    assert report.alignments[0].classification is AlignmentClass.EXACT_TOKEN
    assert [item.classification for item in report.alignments[1:]] == [
        AlignmentClass.UNALIGNABLE,
        AlignmentClass.UNALIGNABLE,
        AlignmentClass.UNALIGNABLE,
        AlignmentClass.UNALIGNABLE,
        AlignmentClass.UNALIGNABLE,
    ]
    assert [item.reason for item in report.alignments[1:]] == [
        "not-valid-utf8",
        "empty-decoded-text",
        "no-byte-faithful-donor-representation",
        "decode-error:ValueError",
        "decoded-value-is-not-text",
    ]
    assert [
        (issue.token_id, issue.reason) for issue in report.donor_surface_issues
    ] == [
        (1, "decode-error:ValueError"),
        (2, "empty-decoded-text"),
        (3, "replacement-character-is-not-byte-faithful"),
    ]


def test_reports_and_fingerprints_are_deterministic() -> None:
    donor = FakeTokenizer(
        ["x", "x", "a", "b"],
        encodings={"ab": [2, 3]},
    )
    target = FakeTokenizer(["x", "ab"])

    first = build_token_alignment(donor, target)
    second = build_token_alignment(donor, target)

    assert first.to_dict() == second.to_dict()
    assert first.fingerprint() == second.fingerprint()
    assert json.dumps(first.to_dict(), sort_keys=True, ensure_ascii=True) == json.dumps(
        second.to_dict(), sort_keys=True, ensure_ascii=True
    )


def test_fail_closed_gate_rejects_low_coverage_and_ambiguity() -> None:
    low_coverage = build_token_alignment(
        FakeTokenizer(["a"]), FakeTokenizer(["a", "b"])
    )

    gate = evaluate_alignment_for_transport(low_coverage)

    assert gate.accepted is False
    assert gate.failures == (
        "coverage_fraction=0.500000 is below required 0.950000",
        "unique_coverage_fraction=0.500000 is below required 0.950000",
        "unalignable_fraction=0.500000 exceeds allowed 0.050000",
    )
    with pytest.raises(AlignmentRejectedError) as caught:
        require_alignment_for_transport(low_coverage)
    assert caught.value.result == gate

    ambiguous = build_token_alignment(FakeTokenizer(["a", "a"]), FakeTokenizer(["a"]))
    with pytest.raises(AlignmentRejectedError, match="ambiguity_fraction"):
        require_alignment_for_transport(ambiguous)


def test_explicit_relaxed_thresholds_can_accept_known_partial_alignment() -> None:
    report = build_token_alignment(FakeTokenizer(["a"]), FakeTokenizer(["a", "b"]))
    thresholds = TransportAlignmentThresholds(
        min_coverage_fraction=0.5,
        min_unique_coverage_fraction=0.5,
        max_unalignable_fraction=0.5,
    )

    result = require_alignment_for_transport(report, thresholds)

    assert result.accepted is True
    assert result.failures == ()


def test_truncated_span_collision_search_is_reported_and_rejected() -> None:
    donor = FakeTokenizer(["a", "a"], encodings={"aa": [0, 0]})
    target = FakeTokenizer(["aa"])

    report = build_token_alignment(donor, target, max_span_candidates=2)
    alignment = report.alignments[0]

    assert alignment.classification is AlignmentClass.UTF8_BYTE_SPAN
    assert alignment.candidate_search_truncated is True
    assert len(alignment.candidates) == 2
    assert report.coverage.truncated_search_count == 1
    with pytest.raises(AlignmentRejectedError, match="candidate search was truncated"):
        require_alignment_for_transport(
            report,
            TransportAlignmentThresholds(
                min_coverage_fraction=1.0,
                min_unique_coverage_fraction=0.0,
                max_unalignable_fraction=0.0,
                max_ambiguity_fraction=1.0,
            ),
        )


def test_special_tokens_are_unalignable_unless_explicitly_included() -> None:
    donor = FakeTokenizer(["<eos>", "word"], special_ids=(0,))
    target = FakeTokenizer(["<eos>", "word"], special_ids=(0,))

    excluded = build_token_alignment(donor, target)
    included = build_token_alignment(donor, target, include_special_tokens=True)

    assert excluded.alignments[0].classification is AlignmentClass.UNALIGNABLE
    assert excluded.alignments[0].reason == "special-token-excluded"
    assert included.alignments[0].classification is AlignmentClass.EXACT_TOKEN
    assert included.alignments[0].candidates[0].donor_token_ids == (0,)


def test_strict_threshold_profile_requires_complete_unique_alignment() -> None:
    complete = build_token_alignment(
        FakeTokenizer(["a", "b"]), FakeTokenizer(["a", "b"])
    )

    result = require_alignment_for_transport(
        complete,
        TransportAlignmentThresholds.strict(),
    )

    assert result.accepted is True


def test_strict_profile_maps_reordered_smaller_vocab_and_padded_donor() -> None:
    donor = FakeTokenizer(["zero", "one", "two", "three"])
    target = FakeTokenizer(["three", "one"])

    profile = build_strict_token_transport_profile(
        donor,
        target,
        donor_row_capacity=8,
        target_row_capacity=2,
    )

    assert profile.target_to_donor_row_indices == (3, 1)
    assert profile.donor_vocab_size == 4
    assert profile.donor_row_capacity == 8
    assert [item.resolution for item in profile.row_mappings] == [
        "exact-token",
        "exact-token",
    ]


def test_strict_profile_maps_special_tokens_by_semantic_role() -> None:
    donor = FakeTokenizer(
        ["<s>", "</s>", "<pad>", "<unk>", "word"],
        special_ids=(0, 1, 2, 3),
    )
    donor.bos_token_id = 0
    donor.eos_token_id = 1
    donor.pad_token_id = 2
    donor.unk_token_id = 3
    target = FakeTokenizer(
        ["word", "<target-unk>", "<target-bos>", "<target-pad>", "<target-eos>"],
        special_ids=(1, 2, 3, 4),
    )
    target.bos_token_id = 2
    target.eos_token_id = 4
    target.pad_token_id = 3
    target.unk_token_id = 1

    profile = build_strict_token_transport_profile(donor, target)

    assert profile.target_to_donor_row_indices == (4, 3, 0, 2, 1)
    assert [item.special_token_roles for item in profile.row_mappings] == [
        (),
        ("UNK",),
        ("BOS",),
        ("PAD",),
        ("EOS",),
    ]


def test_strict_profile_rejects_target_id_holes_and_capacity_mismatch() -> None:
    donor = FakeTokenizer(["a", "b", "c"])
    target = FakeTokenizer(["a", "b", "c"], vocab_ids=(0, 2))

    with pytest.raises(StrictTokenTransportRejectedError) as caught:
        build_strict_token_transport_profile(
            donor,
            target,
            target_row_capacity=4,
        )

    assert caught.value.failures == (
        "target row capacity 4 must equal tokenizer vocabulary size 3",
        "target vocabulary has token ID holes: [1]",
    )


def test_strict_profile_rejects_spans_ambiguity_and_unalignable_rows() -> None:
    span_donor = FakeTokenizer(["a", "b"], encodings={"ab": [0, 1]})
    with pytest.raises(
        StrictTokenTransportRejectedError,
        match="resolves only to a donor token span",
    ):
        build_strict_token_transport_profile(span_donor, FakeTokenizer(["ab"]))

    with pytest.raises(StrictTokenTransportRejectedError, match="ambiguous"):
        build_strict_token_transport_profile(
            FakeTokenizer(["same", "same"]),
            FakeTokenizer(["same"]),
        )

    with pytest.raises(StrictTokenTransportRejectedError, match="unalignable"):
        build_strict_token_transport_profile(
            FakeTokenizer(["present"]),
            FakeTokenizer(["missing"]),
        )


def test_strict_profile_and_mapping_fingerprints_are_deterministic() -> None:
    donor = FakeTokenizer(["b", "a", "c"])
    target = FakeTokenizer(["a", "c"])

    first = build_strict_token_transport_profile(donor, target)
    second = build_strict_token_transport_profile(donor, target)

    assert first.target_to_donor_row_indices == (1, 2)
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint() == second.fingerprint()


def test_strict_profile_uses_direct_identity_for_raw_bytes_and_padding() -> None:
    tokenizer = _bytelevel_bpe_tokenizer()
    raw_byte_id = next(
        token_id
        for token_id in range(len(tokenizer))
        if "\ufffd"
        in tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
    )
    row_capacity = len(tokenizer) + 3

    profile = build_strict_token_transport_profile(
        tokenizer,
        tokenizer,
        donor_row_capacity=row_capacity,
        target_row_capacity=row_capacity,
    )
    repeated = build_strict_token_transport_profile(
        tokenizer,
        tokenizer,
        donor_row_capacity=row_capacity,
        target_row_capacity=row_capacity,
    )

    assert profile.target_to_donor_row_indices == tuple(range(row_capacity))
    assert profile.target_vocab_size == len(tokenizer)
    assert profile.target_row_capacity == row_capacity
    assert profile.alignment_resolution == "direct-tokenizer-identity"
    assert profile.row_mappings[raw_byte_id].resolution == ("direct-tokenizer-identity")
    assert profile.row_mappings[-1].target_row_index == row_capacity - 1
    assert profile.to_dict()["alignment_resolution"] == ("direct-tokenizer-identity")
    assert profile.to_dict() == repeated.to_dict()
    assert profile.fingerprint() == repeated.fingerprint()


def test_strict_profile_maps_reindexed_bytelevel_bpe_descendant_by_token() -> None:
    donor = _bytelevel_bpe_tokenizer()
    target, selected_donor_ids = _pruned_bytelevel_descendant(donor)
    raw_donor_id = next(
        token_id
        for token_id in selected_donor_ids
        if "\ufffd"
        in donor.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
    )
    raw_target_id = selected_donor_ids.index(raw_donor_id)
    assert "\ufffd" in target.decode(
        [raw_target_id],
        clean_up_tokenization_spaces=False,
    )

    profile = build_strict_token_transport_profile(
        donor,
        target,
        donor_row_capacity=len(donor) + 5,
        target_row_capacity=len(target),
    )

    assert target is not donor
    assert profile.target_to_donor_row_indices == selected_donor_ids
    assert profile.alignment_resolution == "bytelevel-bpe-token-string"
    assert profile.row_mappings[raw_target_id].resolution == (
        "bytelevel-bpe-token-string"
    )
    assert profile.to_dict()["alignment_resolution"] == ("bytelevel-bpe-token-string")
    assert len(profile.alignment_fingerprint) == 64


@pytest.mark.parametrize(
    ("component", "failure"),
    (
        ("model", "ByteLevel-BPE model semantics differ"),
        ("pre_tokenizer", "ByteLevel-BPE pre-tokenizer semantics differ"),
        ("decoder", "ByteLevel-BPE decoder semantics differ"),
        ("normalizer", "ByteLevel-BPE normalizer semantics differ"),
    ),
)
def test_strict_profile_rejects_bytelevel_backend_semantic_changes(
    component: str,
    failure: str,
) -> None:
    donor = _bytelevel_bpe_tokenizer()
    target = _bytelevel_semantic_variant(donor, component)

    with pytest.raises(StrictTokenTransportRejectedError) as caught:
        build_strict_token_transport_profile(donor, target)

    assert caught.value.failures == (failure,)
