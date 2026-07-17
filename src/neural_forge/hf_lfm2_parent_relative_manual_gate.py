"""Parent-relative semantic review for LFM2 open-ended holdout cases.

The generic intelligence gate historically rates only the candidate against
an absolute rubric.  That can misclassify a faithful child as a regression
when the cached parent itself is blank, mistaken, or unable to follow the
instruction.  This module keeps the same sealed parent/candidate packet but
requires the reviewer to score both responses on the same four dimensions.

Only dimensions demonstrated by the parent (score >= 3) enter retention.
Abilities absent in the parent are reported as absolute diagnostics and are
not charged to the child.  Promotion requires at least 90% of demonstrated
dimensions to remain demonstrated and rejects any clear demonstrated
capability regression, even when the aggregate rate would otherwise pass.
Exact wording, token identity, and sentence structure are forbidden criteria.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_intelligence_gate import load_manual_review_packet


PACKET_SCHEMA = "neural-forge-lfm2-parent-relative-manual-packet-v1"
RATINGS_SCHEMA = "neural-forge-lfm2-parent-relative-manual-ratings-v1"
REPORT_SCHEMA = "neural-forge-lfm2-parent-relative-manual-report-v1"
DIMENSIONS = (
    "correctness",
    "coherence",
    "instruction_following",
    "capability_retention",
)
MINIMUM_DEMONSTRATED_SCORE = 3
MINIMUM_RETENTION = 0.90


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
            "scale": {"minimum": 0, "maximum": 4},
            "paired_scoring_required": True,
            "same_rubric_for_parent_and_candidate": True,
            "parent_demonstrated_minimum_score": MINIMUM_DEMONSTRATED_SCORE,
            "minimum_demonstrated_capability_retention": MINIMUM_RETENTION,
            "clear_regression_rule": (
                "candidate < 3 on any dimension where parent >= 3"
            ),
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
    minimum_retention: float = MINIMUM_RETENTION,
) -> dict[str, Any]:
    """Score only parent-demonstrated dimensions and reject regressions."""

    if (
        isinstance(minimum_retention, bool)
        or not isinstance(minimum_retention, (int, float))
        or not 0.0 <= float(minimum_retention) <= 1.0
    ):
        raise Lfm2ParentRelativeManualGateError("minimum_retention must be in [0, 1].")
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
    retained = []
    neutralized = []
    regressions = []
    case_reports = []
    absolute_parent_points = 0
    absolute_candidate_points = 0
    for row in rows:
        case_eligible = []
        case_retained = []
        case_neutralized = []
        case_regressions = []
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
            if candidate_score >= MINIMUM_DEMONSTRATED_SCORE:
                retained.append(key)
                case_retained.append(dimension)
            else:
                regressions.append(key)
                case_regressions.append(dimension)
        case_reports.append(
            {
                "case_id": row["case_id"],
                "parent_demonstrated_dimensions": case_eligible,
                "retained_dimensions": case_retained,
                "neutralized_parent_absent_dimensions": case_neutralized,
                "clear_regression_dimensions": case_regressions,
                "clear_regression": bool(case_regressions),
                "reason": row["reason"],
            }
        )
    if not eligible:
        raise Lfm2ParentRelativeManualGateError(
            "The cached parent demonstrated no manually scorable dimensions."
        )
    retention = len(retained) / len(eligible)
    passed = retention >= float(minimum_retention) and not regressions
    body = {
        "schema": REPORT_SCHEMA,
        "packet_sha256": packet_sha,
        "ratings_sha256": ratings_sha,
        "reviewer_id": ratings_body["reviewer_id"],
        "status": "accepted" if passed else "rejected",
        "passed": passed,
        "minimum_retention": float(minimum_retention),
        "parent_demonstrated_dimension_count": len(eligible),
        "retained_dimension_count": len(retained),
        "retention": retention,
        "neutralized_parent_absent_dimension_count": len(neutralized),
        "clear_regression_count": len(regressions),
        "clear_regressions": regressions,
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
            "clear_regressions_rejected": True,
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
    "Lfm2ParentRelativeManualGateError",
    "bind_parent_relative_manual_ratings",
    "build_parent_relative_manual_packet",
    "evaluate_parent_relative_manual_retention",
]
