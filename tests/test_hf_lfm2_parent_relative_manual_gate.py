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


def _dense_packet(monkeypatch) -> dict:
    source = {
        "packet_sha256": "5" * 64,
        "baseline_document_sha256": "6" * 64,
        "candidate_outputs_sha256": "7" * 64,
        "policy_sha256": "8" * 64,
        "cases": [
            {
                "case_id": f"manual_{index:02d}",
                "prompt": f"Prompt {index}",
                "reference_response": f"Strong parent response {index}",
                "candidate_response": f"Useful child response {index}",
            }
            for index in range(6)
        ],
    }
    monkeypatch.setattr(
        gate,
        "load_manual_review_packet",
        lambda value, expected_packet_sha256: value,
    )
    return gate.build_parent_relative_manual_packet(source)


def _dense_ratings() -> list[dict]:
    return [
        {
            "case_id": f"manual_{index:02d}",
            "parent": {dimension: 4 for dimension in gate.DIMENSIONS},
            "candidate": {dimension: 4 for dimension in gate.DIMENSIONS},
            "reason": "Both responses remain semantically useful and coherent.",
        }
        for index in range(6)
    ]


def _bind(packet: dict, ratings: list[dict]) -> dict:
    return gate.bind_parent_relative_manual_ratings(
        packet,
        ratings,
        reviewer_id="paired-reviewer",
    )


def _qualitative_reviews(packet: dict) -> list[dict]:
    rows = []
    for index, value in enumerate(packet["cases"]):
        parent_completion = value.get("parent_completion", {})
        candidate_completion = value.get("candidate_completion", {})
        rows.append(
            {
                "case_id": value["case_id"],
                "verdict": (
                    "parent_not_demonstrated"
                    if value["case_id"] == "manual_00"
                    else "different_but_equivalent"
                ),
                "failure_modes": [],
                "reason": (
                    f"Case observation {index} compares the complete semantic content "
                    "of both responses for this prompt."
                ),
                "full_parent_and_candidate_responses_reviewed": True,
                "criterion_observations": {
                    "correctness": (
                        f"Correctness observation {index} checks the response claims "
                        "against the requested task."
                    ),
                    "coherence": (
                        f"Coherence observation {index} checks whether the complete "
                        "answer remains logically understandable."
                    ),
                    "usefulness": (
                        f"Usefulness observation {index} checks whether the answer "
                        "provides actionable semantic value."
                    ),
                    "instruction_following": (
                        f"Instruction observation {index} checks each requested action "
                        "against the produced response."
                    ),
                    "retained_capability_and_intelligence": (
                        f"Capability observation {index} compares the reasoning shown "
                        "by parent and child responses."
                    ),
                },
                "truncation_consideration": {
                    "parent_unfinished_at_generation_limit": parent_completion.get(
                        "unfinished_at_generation_limit", False
                    ),
                    "candidate_unfinished_at_generation_limit": (
                        candidate_completion.get(
                            "unfinished_at_generation_limit",
                            False,
                        )
                    ),
                    "assessment": "none",
                    "observation": (
                        f"Truncation observation {index} explicitly considers whether "
                        "either complete response ended at its token limit."
                    ),
                },
            }
        )
    return rows


def test_parent_absent_abilities_are_neutralized_and_severe_regression_is_flagged(
    monkeypatch,
) -> None:
    packet = _packet(monkeypatch)
    ratings = _bind(packet, _ratings(repaired_manual_05=False))
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    assert report["status"] == "diagnostic_complete"
    assert report["diagnostic_retention_reference_met"] is False
    assert report["parent_demonstrated_dimension_count"] == 5
    assert report["candidate_demonstrated_dimension_count"] == 1
    assert report["parent_demonstrated_point_count"] == 20
    assert report["capped_candidate_retained_point_count"] == 6
    assert report["lost_parent_demonstrated_point_count"] == 14
    assert report["retention"] == pytest.approx(0.30)
    assert report["neutralized_parent_absent_dimension_count"] == 7
    assert report["severe_regression_count"] == 4
    assert all(value.startswith("manual_05:") for value in report["severe_regressions"])
    manual_00 = report["case_reports"][0]
    assert manual_00["parent_demonstrated_dimensions"] == []
    assert len(manual_00["neutralized_parent_absent_dimensions"]) == 4
    assert report["absolute_diagnostics"]["promotion_eligible"] is False
    assert packet["schema"].endswith("-v2")
    assert packet["rubric"]["scale"]["anchor_3"].startswith("substantially")
    assert packet["rubric"]["automatic_accept_reject_permitted"] is False
    assert report["contract"]["automatic_accept_reject_authority"] is False
    assert report["contract"]["publication_blocked_by_diagnostics"] is False


def test_graded_ninety_to_ninety_nine_percent_retention_is_recorded(
    monkeypatch,
) -> None:
    packet = _dense_packet(monkeypatch)
    rows = _dense_ratings()
    for index, row in enumerate(rows):
        row["candidate"][gate.DIMENSIONS[index % len(gate.DIMENSIONS)]] = 3
    ratings = _bind(packet, rows)
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    assert report["status"] == "diagnostic_complete"
    assert report["diagnostic_retention_reference_met"] is True
    assert report["parent_demonstrated_point_count"] == 96
    assert report["capped_candidate_retained_point_count"] == 90
    assert report["retention"] == pytest.approx(0.9375)
    assert report["severe_regressions"] == []
    assert report["contract"]["abilities_absent_in_parent_neutralized"] is True
    assert report["contract"]["retention_uses_capped_score_points"] is True
    assert report["contract"]["exact_wording_comparison_used"] is False


def test_graded_retention_below_ninety_percent_remains_publishable_diagnostics(
    monkeypatch,
) -> None:
    packet = _dense_packet(monkeypatch)
    rows = _dense_ratings()
    losses_remaining = 10
    for row in rows:
        for dimension in gate.DIMENSIONS:
            if losses_remaining:
                row["candidate"][dimension] = 3
                losses_remaining -= 1
    ratings = _bind(packet, rows)
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    assert report["retention"] == pytest.approx(86 / 96)
    assert report["severe_regression_count"] == 0
    assert report["diagnostic_retention_reference_met"] is False
    assert report["status"] == "diagnostic_complete"


def test_severe_drop_is_flagged_without_vetoing_diagnostic_publication(
    monkeypatch,
) -> None:
    packet = _dense_packet(monkeypatch)
    rows = _dense_ratings()
    rows[0]["candidate"]["coherence"] = 2
    ratings = _bind(packet, rows)
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    assert report["retention"] == pytest.approx(94 / 96)
    assert report["severe_regressions"] == ["manual_00:coherence"]
    assert report["diagnostic_retention_reference_met"] is True
    assert report["status"] == "diagnostic_complete"
    assert report["contract"]["publication_blocked_by_diagnostics"] is False


def test_candidate_improvement_cannot_compensate_for_a_loss(monkeypatch) -> None:
    packet = _dense_packet(monkeypatch)
    rows = _dense_ratings()
    rows[0]["parent"]["coherence"] = 3
    rows[0]["candidate"]["coherence"] = 4
    rows[0]["candidate"]["correctness"] = 3
    ratings = _bind(packet, rows)
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    assert report["parent_demonstrated_point_count"] == 95
    assert report["capped_candidate_retained_point_count"] == 94
    assert report["retention"] == pytest.approx(94 / 95)
    assert report["absolute_diagnostics"]["candidate_points"] == 95
    assert report["diagnostic_retention_reference_met"] is True


def test_v1_packet_and_ratings_are_rejected(monkeypatch) -> None:
    packet = _dense_packet(monkeypatch)
    v1_packet = copy.deepcopy(packet)
    v1_packet["schema"] = "neural-forge-lfm2-parent-relative-manual-packet-v1"
    packet_body = dict(v1_packet)
    packet_body.pop("packet_sha256")
    v1_packet["packet_sha256"] = gate.canonical_sha256(packet_body)
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="packet seal or rubric",
    ):
        _bind(v1_packet, _dense_ratings())

    ratings = _bind(packet, _dense_ratings())
    v1_ratings = copy.deepcopy(ratings)
    v1_ratings["schema"] = "neural-forge-lfm2-parent-relative-manual-ratings-v1"
    ratings_body = dict(v1_ratings)
    ratings_body.pop("ratings_sha256")
    v1_ratings["ratings_sha256"] = gate.canonical_sha256(ratings_body)
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="seals or binding",
    ):
        gate.evaluate_parent_relative_manual_retention(packet, v1_ratings)


def test_packet_and_rating_tampering_fail_closed(monkeypatch) -> None:
    packet = _packet(monkeypatch)
    ratings = _bind(packet, _ratings(repaired_manual_05=True))
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


def test_qualitative_case_verdicts_and_suite_conclusion_are_authoritative(
    monkeypatch,
) -> None:
    packet = _packet(monkeypatch)
    rows = _ratings(repaired_manual_05=True)
    qualitative = _qualitative_reviews(packet)
    ratings = gate.bind_parent_relative_manual_ratings(
        packet,
        rows,
        reviewer_id="holistic-reviewer",
        qualitative_case_reviews=qualitative,
        holistic_suite_conclusion={
            "conclusion": "credible_intelligent_compaction",
            "reason": "The reviewed responses remain coherent and useful.",
            "strengths": ["coherence"],
            "limitations": ["parent did not demonstrate every ability"],
            "material_regression_case_ids": [],
        },
    )
    report = gate.evaluate_parent_relative_manual_retention(packet, ratings)
    holistic = report["holistic_review"]
    assert holistic["status"] == "complete"
    assert holistic["qualitative_judgment_is_authoritative"] is True
    assert holistic["suite_conclusion"]["conclusion"] == (
        gate.PROMOTION_ELIGIBLE_SUITE_CONCLUSION
    )
    assert report["contract"]["numeric_scores_and_thresholds_are_diagnostic_only"]

    stale = copy.deepcopy(ratings)
    stale.pop("qualitative_judgment_is_authoritative")
    core = dict(stale)
    core.pop("ratings_sha256")
    stale["ratings_sha256"] = gate.canonical_sha256(core)
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="qualitative-authority contract",
    ):
        gate.evaluate_parent_relative_manual_retention(packet, stale)


def test_qualitative_review_rejects_missing_criterion_and_duplicate_boilerplate(
    monkeypatch,
) -> None:
    packet = _packet(monkeypatch)
    rows = _ratings(repaired_manual_05=True)
    qualitative = _qualitative_reviews(packet)
    missing = copy.deepcopy(qualitative)
    missing[0]["criterion_observations"].pop("usefulness")
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="criterion coverage",
    ):
        gate.bind_parent_relative_manual_ratings(
            packet,
            rows,
            reviewer_id="holistic-reviewer",
            qualitative_case_reviews=missing,
            holistic_suite_conclusion={
                "conclusion": "credible_intelligent_compaction",
                "reason": "The suite was reviewed despite malformed case coverage.",
                "strengths": ["coherence"],
                "limitations": [],
                "material_regression_case_ids": [],
            },
        )

    duplicated = copy.deepcopy(qualitative)
    duplicated[1]["criterion_observations"]["correctness"] = duplicated[0][
        "criterion_observations"
    ]["correctness"]
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="duplicate boilerplate",
    ):
        gate.bind_parent_relative_manual_ratings(
            packet,
            rows,
            reviewer_id="holistic-reviewer",
            qualitative_case_reviews=duplicated,
            holistic_suite_conclusion={
                "conclusion": "credible_intelligent_compaction",
                "reason": "The suite was reviewed despite duplicated observations.",
                "strengths": ["coherence"],
                "limitations": [],
                "material_regression_case_ids": [],
            },
        )


def test_all_case_completion_metadata_is_sealed_and_truncation_review_is_bound(
    monkeypatch,
) -> None:
    source = _source_packet()
    monkeypatch.setattr(
        gate,
        "load_manual_review_packet",
        lambda value, expected_packet_sha256: value,
    )
    all_case_packet = gate.build_all_case_parent_relative_manual_packet(
        source,
        [
            {
                "case_id": value["case_id"],
                "prompt": value["prompt"],
                "parent_response": value["reference_response"],
                "candidate_response": value["candidate_response"],
                "generation_limit": 8,
                "parent_completion": {
                    "generated_token_count": 4,
                    "eos_observed": True,
                    "unfinished_at_generation_limit": False,
                },
                "candidate_completion": {
                    "generated_token_count": 8 if index == 2 else 4,
                    "eos_observed": index != 2,
                    "unfinished_at_generation_limit": index == 2,
                },
            }
            for index, value in enumerate(source["cases"])
        ],
        manifest_sha256="9" * 64,
        manifest_document_sha256="a" * 64,
    )
    assert all_case_packet["completion_metadata_rows_sha256"]
    assert all_case_packet["rubric"]["completion_metadata_is_diagnostic_context_only"]
    assert (
        all_case_packet["rubric"][
            "completion_metadata_automatic_accept_reject_permitted"
        ]
        is False
    )

    qualitative = _qualitative_reviews(all_case_packet)
    ratings = gate.bind_parent_relative_manual_ratings(
        all_case_packet,
        _ratings(repaired_manual_05=True),
        reviewer_id="completion-reviewer",
        qualitative_case_reviews=qualitative,
        holistic_suite_conclusion={
            "conclusion": "promising_but_incomplete",
            "reason": "One response reached its limit and was assessed manually.",
            "strengths": ["coherence"],
            "limitations": ["one saturated continuation"],
            "material_regression_case_ids": [],
        },
    )
    report = gate.evaluate_parent_relative_manual_retention(all_case_packet, ratings)
    assert report["holistic_review"]["status"] == "complete"
    assert (
        report["holistic_review"]["qualitative_case_reviews"][2][
            "truncation_consideration"
        ]["candidate_unfinished_at_generation_limit"]
        is True
    )

    mismatched = _qualitative_reviews(all_case_packet)
    mismatched[2]["truncation_consideration"][
        "candidate_unfinished_at_generation_limit"
    ] = False
    with pytest.raises(
        gate.Lfm2ParentRelativeManualGateError,
        match="differs from sealed response completion metadata",
    ):
        gate.bind_parent_relative_manual_ratings(
            all_case_packet,
            _ratings(repaired_manual_05=True),
            reviewer_id="completion-reviewer",
            qualitative_case_reviews=mismatched,
            holistic_suite_conclusion={
                "conclusion": "promising_but_incomplete",
                "reason": "One response reached its limit and was assessed manually.",
                "strengths": ["coherence"],
                "limitations": ["one saturated continuation"],
                "material_regression_case_ids": [],
            },
        )
