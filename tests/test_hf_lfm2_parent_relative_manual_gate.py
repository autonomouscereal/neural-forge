from __future__ import annotations

import copy

import pytest

from neural_forge import hf_lfm2_parent_relative_manual_gate as gate


def _source_packet() -> dict:
    return {
        "packet_sha256": "1" * 64,
        "baseline_document_sha256": "2" * 64,
        "candidate_outputs_sha256": "3" * 64,
        "policy_sha256": "4" * 64,
        "cases": [
            {
                "case_id": "manual_00",
                "prompt": "Help",
                "reference_response": "",
                "candidate_response": "",
            },
            {
                "case_id": "manual_02",
                "prompt": "Explain",
                "reference_response": "Inherited misconception",
                "candidate_response": "Inherited misconception",
            },
            {
                "case_id": "manual_05",
                "prompt": "Diagnose",
                "reference_response": "Two questions and a next step",
                "candidate_response": "Irrelevant template",
            },
        ],
    }


def _ratings(*, repaired_manual_05: bool) -> list[dict]:
    return [
        {
            "case_id": "manual_00",
            "parent": {
                "correctness": 0,
                "coherence": 0,
                "instruction_following": 0,
                "capability_retention": 0,
            },
            "candidate": {
                "correctness": 0,
                "coherence": 0,
                "instruction_following": 0,
                "capability_retention": 0,
            },
            "reason": "Both are blank; no parent ability was demonstrated.",
        },
        {
            "case_id": "manual_02",
            "parent": {
                "correctness": 1,
                "coherence": 2,
                "instruction_following": 2,
                "capability_retention": 4,
            },
            "candidate": {
                "correctness": 1,
                "coherence": 2,
                "instruction_following": 2,
                "capability_retention": 4,
            },
            "reason": "The child exactly preserves the parent's limited answer.",
        },
        {
            "case_id": "manual_05",
            "parent": {
                "correctness": 4,
                "coherence": 4,
                "instruction_following": 4,
                "capability_retention": 4,
            },
            "candidate": (
                {
                    "correctness": 3,
                    "coherence": 3,
                    "instruction_following": 3,
                    "capability_retention": 3,
                }
                if repaired_manual_05
                else {
                    "correctness": 0,
                    "coherence": 1,
                    "instruction_following": 0,
                    "capability_retention": 1,
                }
            ),
            "reason": (
                "The child keeps the demonstrated diagnostic behavior."
                if repaired_manual_05
                else "The child loses prompt binding and diagnostic behavior."
            ),
        },
    ]


def _packet(monkeypatch) -> dict:
    source = _source_packet()
    monkeypatch.setattr(
        gate,
        "load_manual_review_packet",
        lambda value, expected_packet_sha256: value,
    )
    return gate.build_parent_relative_manual_packet(source)


def test_parent_absent_abilities_are_neutralized_but_clear_regression_rejects(
    monkeypatch,
) -> None:
    packet = _packet(monkeypatch)
    ratings = gate.bind_parent_relative_manual_ratings(
        packet,
        _ratings(repaired_manual_05=False),
        reviewer_id="paired-reviewer",
    )
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    assert report["passed"] is False
    assert report["status"] == "rejected"
    assert report["parent_demonstrated_dimension_count"] == 5
    assert report["retained_dimension_count"] == 1
    assert report["neutralized_parent_absent_dimension_count"] == 7
    assert report["clear_regression_count"] == 4
    assert all(value.startswith("manual_05:") for value in report["clear_regressions"])
    manual_00 = report["case_reports"][0]
    assert manual_00["parent_demonstrated_dimensions"] == []
    assert len(manual_00["neutralized_parent_absent_dimensions"]) == 4
    assert report["absolute_diagnostics"]["promotion_eligible"] is False


def test_ninety_percent_parent_demonstrated_retention_accepts(
    monkeypatch,
) -> None:
    packet = _packet(monkeypatch)
    ratings = gate.bind_parent_relative_manual_ratings(
        packet,
        _ratings(repaired_manual_05=True),
        reviewer_id="paired-reviewer",
    )
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    assert report["passed"] is True
    assert report["status"] == "accepted"
    assert report["retention"] == 1.0
    assert report["clear_regressions"] == []
    assert report["contract"]["abilities_absent_in_parent_neutralized"] is True
    assert report["contract"]["exact_wording_comparison_used"] is False


def test_packet_and_rating_tampering_fail_closed(monkeypatch) -> None:
    packet = _packet(monkeypatch)
    ratings = gate.bind_parent_relative_manual_ratings(
        packet,
        _ratings(repaired_manual_05=True),
        reviewer_id="paired-reviewer",
    )
    tampered = copy.deepcopy(ratings)
    tampered["ratings"][0]["candidate"]["coherence"] = 4
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="seals or binding",
    ):
        gate.evaluate_parent_relative_manual_retention(packet, tampered)
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="Exact wording",
    ):
        gate.bind_parent_relative_manual_ratings(
            packet,
            _ratings(repaired_manual_05=True),
            reviewer_id="paired-reviewer",
            used_exact_wording_comparison=True,
        )
