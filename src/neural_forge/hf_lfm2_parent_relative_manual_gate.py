"""Parent-relative semantic review for LFM2 open-ended holdout cases.

The generic intelligence gate historically rates only the candidate against
an absolute rubric.  That can misclassify a faithful child as a regression
when the cached parent itself is blank, mistaken, or unable to follow the
instruction.  This module keeps the same sealed parent/candidate packet but
requires the reviewer to score both responses on the same four dimensions.

Only dimensions demonstrated by the parent (score >= 3) enter retention.
Abilities absent in the parent are reported as absolute diagnostics and are
not charged to the child.  Retention is the candidate's score-point total,
capped independently at the parent's score on every eligible dimension,
divided by the parent's eligible score-point total.  Candidate improvements
therefore cannot compensate for losses elsewhere.  The 90% reference and
severe two-or-more-point drops are diagnostic signals for holistic review;
they never accept, reject, certify, or block publication.  Exact wording,
token identity, and sentence structure are forbidden criteria.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_intelligence_gate import load_manual_review_packet


PACKET_SCHEMA = "neural-forge-lfm2-parent-relative-manual-packet-v2"
RATINGS_SCHEMA = "neural-forge-lfm2-parent-relative-manual-ratings-v2"
REPORT_SCHEMA = "neural-forge-lfm2-parent-relative-manual-report-v2"
DIMENSIONS = (
    "correctness",
    "coherence",
    "instruction_following",
    "capability_retention",
)
MINIMUM_DEMONSTRATED_SCORE = 3
MINIMUM_RETENTION = 0.90
SEVERE_REGRESSION_MINIMUM_DROP = 2


class Lfm2ParentRelativeManualGateError(ValueError):
    """The packet, ratings, or parent-relative scoring contract is invalid."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ParentRelativeManualGateError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _score(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 4:
        raise Lfm2ParentRelativeManualGateError(f"{name} must be an integer in [0, 4].")
    return value


def build_parent_relative_manual_packet(
    manual_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a paired-rating contract to an existing sealed review packet."""

    source = load_manual_review_packet(
        manual_packet,
        expected_packet_sha256=str(manual_packet.get("packet_sha256", "")),
    )
    cases = source.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Lfm2ParentRelativeManualGateError(
            "The source manual packet has no review cases."
        )
    body = {
        "schema": PACKET_SCHEMA,
        "source_manual_packet_sha256": source["packet_sha256"],
        "baseline_document_sha256": source["baseline_document_sha256"],
        "candidate_outputs_sha256": source["candidate_outputs_sha256"],
        "policy_sha256": source["policy_sha256"],
        "rubric": {
            "dimensions": list(DIMENSIONS),
            "scale": {
                "minimum": 0,
                "maximum": 4,
                "anchor_0": "absent, wrong, or incoherent",
                "anchor_1": "mostly unusable or severely impaired",
                "anchor_2": "partly useful but materially impaired",
                "anchor_3": "substantially useful and coherent with limited defects",
                "anchor_4": "fully useful and coherent",
            },
            "paired_scoring_required": True,
            "same_rubric_for_parent_and_candidate": True,
            "parent_demonstrated_minimum_score": MINIMUM_DEMONSTRATED_SCORE,
            "diagnostic_retention_reference": MINIMUM_RETENTION,
            "retention_rule": (
                "sum(min(candidate, parent)) / sum(parent) over dimensions "
                "where parent >= 3"
            ),
            "candidate_improvements_capped_at_parent_score": True,
            "severe_regression_minimum_score_drop": (SEVERE_REGRESSION_MINIMUM_DROP),
            "severe_regression_rule": ("parent >= 3 and parent - candidate >= 2"),
            "one_point_losses_reduce_retention_without_automatic_veto": True,
            "numeric_goal_and_severe_flags_are_diagnostic_only": True,
            "automatic_accept_reject_permitted": False,
            "publication_blocked_by_diagnostics": False,
            "abilities_absent_in_parent_neutralized": True,
            "absolute_scores_are_diagnostics_only": True,
            "exact_wording_comparison_permitted": False,
        },
        "cases": [
            {
                "case_id": str(value["case_id"]),
                "prompt": str(value["prompt"]),
                "parent_response": str(value["reference_response"]),
                "candidate_response": str(value["candidate_response"]),
            }
            for value in cases
        ],
    }
    return {**body, "packet_sha256": canonical_sha256(body)}


def bind_parent_relative_manual_ratings(
    packet: Mapping[str, Any],
    ratings: Sequence[Mapping[str, Any]],
    *,
    reviewer_id: str,
    used_exact_wording_comparison: bool = False,
) -> dict[str, Any]:
    """Validate and seal paired parent/candidate rubric scores."""

    if not isinstance(packet, Mapping):
        raise Lfm2ParentRelativeManualGateError("packet must be a mapping.")
    packet_body = dict(packet)
    packet_sha = packet_body.pop("packet_sha256", None)
    if (
        packet_body.get("schema") != PACKET_SCHEMA
        or _digest(packet_sha, "packet_sha256") != canonical_sha256(packet_body)
        or packet_body.get("rubric", {}).get("exact_wording_comparison_permitted")
        is not False
    ):
        raise Lfm2ParentRelativeManualGateError(
            "Parent-relative manual packet seal or rubric is invalid."
        )
    if used_exact_wording_comparison is not False:
        raise Lfm2ParentRelativeManualGateError(
            "Exact wording cannot be used for parent-relative ratings."
        )
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise Lfm2ParentRelativeManualGateError("reviewer_id must be nonempty.")
    expected_ids = tuple(str(value["case_id"]) for value in packet_body["cases"])
    normalized = []
    for row in ratings:
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "parent",
            "candidate",
            "reason",
        }:
            raise Lfm2ParentRelativeManualGateError(
                "Paired manual rating fields differ from the schema."
            )
        case_id = str(row["case_id"])
        reason = row["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise Lfm2ParentRelativeManualGateError(
                "Each paired manual rating needs a concrete reason."
            )
        values = {"case_id": case_id, "reason": reason.strip()}
        for side in ("parent", "candidate"):
            source = row[side]
            if not isinstance(source, Mapping) or set(source) != set(DIMENSIONS):
                raise Lfm2ParentRelativeManualGateError(
                    f"{side} ratings must contain all four dimensions."
                )
            values[side] = {
                dimension: _score(
                    source[dimension],
                    f"{case_id}.{side}.{dimension}",
                )
                for dimension in DIMENSIONS
            }
        normalized.append(values)
    actual_ids = tuple(value["case_id"] for value in normalized)
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise Lfm2ParentRelativeManualGateError(
            "Paired ratings must cover every packet case exactly once."
        )
    normalized.sort(key=lambda value: expected_ids.index(value["case_id"]))
    body = {
        "schema": RATINGS_SCHEMA,
        "packet_sha256": packet_sha,
        "reviewer_id": reviewer_id.strip(),
        "exact_wording_comparison_used": False,
        "ratings": normalized,
    }
    return {**body, "ratings_sha256": canonical_sha256(body)}


def evaluate_parent_relative_manual_retention(
    packet: Mapping[str, Any],
    ratings: Mapping[str, Any],
    *,
    diagnostic_retention_reference: float = MINIMUM_RETENTION,
) -> dict[str, Any]:
    """Publish capped score-point and severe-drop diagnostics without gating."""

    if (
        isinstance(diagnostic_retention_reference, bool)
        or not isinstance(diagnostic_retention_reference, (int, float))
        or not 0.0 <= float(diagnostic_retention_reference) <= 1.0
    ):
        raise Lfm2ParentRelativeManualGateError(
            "diagnostic_retention_reference must be in [0, 1]."
        )
    packet_body = dict(packet)
    packet_sha = packet_body.pop("packet_sha256", None)
    ratings_body = dict(ratings)
    ratings_sha = ratings_body.pop("ratings_sha256", None)
    if (
        packet_body.get("schema") != PACKET_SCHEMA
        or _digest(packet_sha, "packet_sha256") != canonical_sha256(packet_body)
        or ratings_body.get("schema") != RATINGS_SCHEMA
        or ratings_body.get("packet_sha256") != packet_sha
        or ratings_body.get("exact_wording_comparison_used") is not False
        or _digest(ratings_sha, "ratings_sha256") != canonical_sha256(ratings_body)
    ):
        raise Lfm2ParentRelativeManualGateError(
            "Paired packet/ratings seals or binding are invalid."
        )
    case_ids = tuple(str(value["case_id"]) for value in packet_body["cases"])
    rows = ratings_body.get("ratings")
    if (
        not isinstance(rows, list)
        or tuple(str(value.get("case_id")) for value in rows) != case_ids
    ):
        raise Lfm2ParentRelativeManualGateError(
            "Paired ratings order or coverage differs from the packet."
        )
    eligible = []
    candidate_demonstrated = []
    fully_preserved = []
    neutralized = []
    severe_regressions = []
    case_reports = []
    absolute_parent_points = 0
    absolute_candidate_points = 0
    parent_demonstrated_points = 0
    capped_candidate_retained_points = 0
    for row in rows:
        case_eligible = []
        case_candidate_demonstrated = []
        case_fully_preserved = []
        case_neutralized = []
        case_severe_regressions = []
        case_parent_points = 0
        case_capped_candidate_points = 0
        for dimension in DIMENSIONS:
            parent_score = _score(
                row["parent"][dimension],
                f"{row['case_id']}.parent.{dimension}",
            )
            candidate_score = _score(
                row["candidate"][dimension],
                f"{row['case_id']}.candidate.{dimension}",
            )
            absolute_parent_points += parent_score
            absolute_candidate_points += candidate_score
            key = f"{row['case_id']}:{dimension}"
            if parent_score < MINIMUM_DEMONSTRATED_SCORE:
                neutralized.append(key)
                case_neutralized.append(dimension)
                continue
            eligible.append(key)
            case_eligible.append(dimension)
            parent_demonstrated_points += parent_score
            case_parent_points += parent_score
            capped_candidate_score = min(candidate_score, parent_score)
            capped_candidate_retained_points += capped_candidate_score
            case_capped_candidate_points += capped_candidate_score
            if candidate_score >= MINIMUM_DEMONSTRATED_SCORE:
                candidate_demonstrated.append(key)
                case_candidate_demonstrated.append(dimension)
            if candidate_score >= parent_score:
                fully_preserved.append(key)
                case_fully_preserved.append(dimension)
            if parent_score - candidate_score >= SEVERE_REGRESSION_MINIMUM_DROP:
                severe_regressions.append(key)
                case_severe_regressions.append(dimension)
        case_reports.append(
            {
                "case_id": row["case_id"],
                "parent_demonstrated_dimensions": case_eligible,
                "candidate_demonstrated_dimensions": case_candidate_demonstrated,
                "fully_preserved_dimensions": case_fully_preserved,
                "neutralized_parent_absent_dimensions": case_neutralized,
                "parent_demonstrated_points": case_parent_points,
                "capped_candidate_retained_points": case_capped_candidate_points,
                "lost_parent_demonstrated_points": (
                    case_parent_points - case_capped_candidate_points
                ),
                "severe_regression_dimensions": case_severe_regressions,
                "severe_regression": bool(case_severe_regressions),
                "reason": row["reason"],
            }
        )
    if not eligible:
        raise Lfm2ParentRelativeManualGateError(
            "The cached parent demonstrated no manually scorable dimensions."
        )
    retention = capped_candidate_retained_points / parent_demonstrated_points
    diagnostic_goal_met = retention >= float(diagnostic_retention_reference)
    body = {
        "schema": REPORT_SCHEMA,
        "packet_sha256": packet_sha,
        "ratings_sha256": ratings_sha,
        "reviewer_id": ratings_body["reviewer_id"],
        "status": "diagnostic_complete",
        "diagnostic_retention_reference": float(diagnostic_retention_reference),
        "diagnostic_retention_reference_met": diagnostic_goal_met,
        "parent_demonstrated_dimension_count": len(eligible),
        "candidate_demonstrated_dimension_count": len(candidate_demonstrated),
        "fully_preserved_dimension_count": len(fully_preserved),
        "parent_demonstrated_point_count": parent_demonstrated_points,
        "capped_candidate_retained_point_count": capped_candidate_retained_points,
        "lost_parent_demonstrated_point_count": (
            parent_demonstrated_points - capped_candidate_retained_points
        ),
        "retention": retention,
        "neutralized_parent_absent_dimension_count": len(neutralized),
        "severe_regression_count": len(severe_regressions),
        "severe_regressions": severe_regressions,
        "case_reports": case_reports,
        "absolute_diagnostics": {
            "parent_points": absolute_parent_points,
            "candidate_points": absolute_candidate_points,
            "available_points": len(rows) * len(DIMENSIONS) * 4,
            "promotion_eligible": False,
        },
        "contract": {
            "parent_and_candidate_scored_on_same_rubric": True,
            "abilities_absent_in_parent_neutralized": True,
            "retention_uses_capped_score_points": True,
            "candidate_improvements_cannot_compensate_losses": True,
            "severe_regression_minimum_score_drop": (SEVERE_REGRESSION_MINIMUM_DROP),
            "severe_regressions_flagged_for_holistic_review": True,
            "numeric_goal_and_severe_flags_are_diagnostic_only": True,
            "automatic_accept_reject_authority": False,
            "publication_blocked_by_diagnostics": False,
            "holistic_whole_response_judgment_required": True,
            "exact_wording_comparison_used": False,
            "cached_outputs_only": True,
            "parent_model_loaded": False,
            "candidate_model_loaded": False,
            "forward_calls": 0,
            "generation_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
    }
    return {**body, "report_sha256": canonical_sha256(body)}


__all__ = [
    "DIMENSIONS",
    "MINIMUM_DEMONSTRATED_SCORE",
    "MINIMUM_RETENTION",
    "PACKET_SCHEMA",
    "RATINGS_SCHEMA",
    "REPORT_SCHEMA",
    "SEVERE_REGRESSION_MINIMUM_DROP",
    "Lfm2ParentRelativeManualGateError",
    "bind_parent_relative_manual_ratings",
    "build_parent_relative_manual_packet",
    "evaluate_parent_relative_manual_retention",
]
