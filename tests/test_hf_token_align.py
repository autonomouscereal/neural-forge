from __future__ import annotations

import json
from typing import Any

import pytest

from neural_forge.hf_token_align import AlignmentClass
from neural_forge.hf_token_align import AlignmentRejectedError
from neural_forge.hf_token_align import TransportAlignmentThresholds
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
    ) -> None:
        self.surfaces = list(surfaces)
        self.encodings = dict(encodings or {})
        self.decode_overrides = dict(decode_overrides or {})
        self.all_special_ids = list(special_ids)

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
