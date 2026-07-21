"""Parent-relative semantic review for LFM2 whole-response holdout cases.

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
token identity, and sentence structure are forbidden criteria.  Optional
categorical case verdicts and a suite conclusion carry the qualitative
judgment; numeric scores never carry accept/reject authority.

An exact all-case packet also seals each response's generated-token count, EOS
observation, and unfinished-at-limit flag.  Those completion facts inform the
reviewer but never decide the outcome.  An authoritative qualitative row must
record distinct case-specific observations for correctness, coherence,
usefulness, instruction following, and retained capability/intelligence, plus
an explicit consideration of both responses' truncation state.
"""

from __future__ import annotations

import re
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
QUALITATIVE_CASE_VERDICTS = (
    "retained_or_improved",
    "different_but_equivalent",
    "partially_retained",
    "material_regression",
    "parent_not_demonstrated",
    "inconclusive",
)
HOLISTIC_SUITE_CONCLUSIONS = (
    "credible_intelligent_compaction",
    "promising_but_incomplete",
    "not_yet_coherent_or_intelligent",
    "inconclusive",
)
PROMOTION_ELIGIBLE_SUITE_CONCLUSION = "credible_intelligent_compaction"
QUALITATIVE_FAILURE_MODES = (
    "incorrect",
    "incoherent",
    "instruction_miss",
    "capability_loss",
    "hallucination",
    "unsafe",
    "inappropriate_refusal",
    "repetition",
    "premature_termination",
    "context_loss",
    "code_failure",
    "arithmetic_failure",
    "other",
)
QUALITATIVE_CRITERIA = (
    "correctness",
    "coherence",
    "usefulness",
    "instruction_following",
    "retained_capability_and_intelligence",
)
TRUNCATION_ASSESSMENTS = (
    "none",
    "limited",
    "material",
    "inconclusive",
)
_MINIMUM_SUBSTANTIVE_OBSERVATION_CHARACTERS = 24
_MINIMUM_SUBSTANTIVE_OBSERVATION_WORDS = 4


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


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2ParentRelativeManualGateError(f"{name} must be nonempty text.")
    return value.strip()


def _text_list(
    value: Any,
    name: str,
    *,
    allowed: Sequence[str] | None = None,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2ParentRelativeManualGateError(f"{name} must be a list of text.")
    normalized = [_text(item, name) for item in value]
    if len(normalized) != len(set(normalized)):
        raise Lfm2ParentRelativeManualGateError(f"{name} contains duplicates.")
    if allowed is not None and not set(normalized).issubset(allowed):
        raise Lfm2ParentRelativeManualGateError(
            f"{name} contains an unsupported value."
        )
    return normalized


def normalize_parent_relative_completion_metadata(
    value: Any,
    *,
    generation_limit: int,
    name: str,
) -> dict[str, Any]:
    """Authenticate non-authoritative completion context for one response."""

    expected_fields = {
        "generated_token_count",
        "eos_observed",
        "unfinished_at_generation_limit",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise Lfm2ParentRelativeManualGateError(
            f"{name} completion metadata fields differ from the schema."
        )
    count = value["generated_token_count"]
    eos_observed = value["eos_observed"]
    unfinished = value["unfinished_at_generation_limit"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or count > generation_limit
        or not isinstance(eos_observed, bool)
        or not isinstance(unfinished, bool)
        or (eos_observed and count == 0)
        or unfinished != (count == generation_limit and not eos_observed)
    ):
        raise Lfm2ParentRelativeManualGateError(
            f"{name} completion metadata is internally inconsistent."
        )
    return {
        "generated_token_count": count,
        "eos_observed": eos_observed,
        "unfinished_at_generation_limit": unfinished,
    }


def _substantive_observation(value: Any, name: str) -> str:
    normalized = _text(value, name)
    words = re.findall(r"\b[\w'’-]+\b", normalized, flags=re.UNICODE)
    if (
        len(normalized) < _MINIMUM_SUBSTANTIVE_OBSERVATION_CHARACTERS
        or len(words) < _MINIMUM_SUBSTANTIVE_OBSERVATION_WORDS
    ):
        raise Lfm2ParentRelativeManualGateError(
            f"{name} must be a substantive case-specific observation."
        )
    return normalized


def _narrative_key(value: str, case_id: str) -> str:
    """Normalize narratives so changing only a case ID remains boilerplate."""

    normalized = " ".join(value.casefold().split())
    return re.sub(re.escape(case_id.casefold()), "<case>", normalized)


def _normalize_holistic_review(
    packet_cases: Sequence[Mapping[str, Any]],
    qualitative_case_reviews: Any,
    holistic_suite_conclusion: Any,
) -> dict[str, Any] | None:
    """Validate an optional categorical review without deriving a scalar verdict."""

    if qualitative_case_reviews is None and holistic_suite_conclusion is None:
        return None
    if qualitative_case_reviews is None or holistic_suite_conclusion is None:
        raise Lfm2ParentRelativeManualGateError(
            "Qualitative case reviews and the holistic suite conclusion must be "
            "provided together."
        )
    if isinstance(qualitative_case_reviews, (str, bytes)) or not isinstance(
        qualitative_case_reviews, Sequence
    ):
        raise Lfm2ParentRelativeManualGateError(
            "qualitative_case_reviews must be a sequence."
        )
    expected_ids = tuple(str(value["case_id"]) for value in packet_cases)
    packet_by_id = {str(value["case_id"]): value for value in packet_cases}
    normalized_reviews = []
    for row in qualitative_case_reviews:
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "verdict",
            "failure_modes",
            "reason",
            "full_parent_and_candidate_responses_reviewed",
            "criterion_observations",
            "truncation_consideration",
        }:
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative case-review fields differ from the schema."
            )
        case_id = _text(row["case_id"], "qualitative case_id")
        packet_case = packet_by_id.get(case_id)
        if packet_case is None:
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative review names a case outside the review packet."
            )
        verdict = _text(row["verdict"], "qualitative verdict")
        if verdict not in QUALITATIVE_CASE_VERDICTS:
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative case verdict is unsupported."
            )
        if row["full_parent_and_candidate_responses_reviewed"] is not True:
            raise Lfm2ParentRelativeManualGateError(
                "Each qualitative review must attest that both full responses "
                "were reviewed."
            )
        criterion_source = row["criterion_observations"]
        if not isinstance(criterion_source, Mapping) or set(criterion_source) != set(
            QUALITATIVE_CRITERIA
        ):
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative criterion coverage differs from the required schema."
            )
        criterion_observations = {
            criterion: _substantive_observation(
                criterion_source[criterion],
                f"{case_id}.{criterion}",
            )
            for criterion in QUALITATIVE_CRITERIA
        }
        truncation_source = row["truncation_consideration"]
        if not isinstance(truncation_source, Mapping) or set(truncation_source) != {
            "parent_unfinished_at_generation_limit",
            "candidate_unfinished_at_generation_limit",
            "assessment",
            "observation",
        }:
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative truncation consideration differs from the schema."
            )
        parent_unfinished = truncation_source["parent_unfinished_at_generation_limit"]
        candidate_unfinished = truncation_source[
            "candidate_unfinished_at_generation_limit"
        ]
        if not isinstance(parent_unfinished, bool) or not isinstance(
            candidate_unfinished, bool
        ):
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative truncation flags must be booleans."
            )
        assessment = _text(
            truncation_source["assessment"],
            "qualitative truncation assessment",
        )
        if assessment not in TRUNCATION_ASSESSMENTS:
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative truncation assessment is unsupported."
            )
        packet_parent_completion = packet_case.get("parent_completion")
        packet_candidate_completion = packet_case.get("candidate_completion")
        if (packet_parent_completion is None) != (packet_candidate_completion is None):
            raise Lfm2ParentRelativeManualGateError(
                "Review packet has only one side of its completion metadata."
            )
        if packet_parent_completion is not None and (
            not isinstance(packet_parent_completion, Mapping)
            or not isinstance(packet_candidate_completion, Mapping)
            or parent_unfinished
            is not packet_parent_completion.get("unfinished_at_generation_limit")
            or candidate_unfinished
            is not packet_candidate_completion.get("unfinished_at_generation_limit")
        ):
            raise Lfm2ParentRelativeManualGateError(
                "Qualitative truncation consideration differs from sealed response "
                "completion metadata."
            )
        normalized_reviews.append(
            {
                "case_id": case_id,
                "verdict": verdict,
                "failure_modes": _text_list(
                    row["failure_modes"],
                    "qualitative failure_modes",
                    allowed=QUALITATIVE_FAILURE_MODES,
                ),
                "reason": _substantive_observation(
                    row["reason"],
                    f"{case_id}.qualitative reason",
                ),
                "full_parent_and_candidate_responses_reviewed": True,
                "criterion_observations": criterion_observations,
                "truncation_consideration": {
                    "parent_unfinished_at_generation_limit": parent_unfinished,
                    "candidate_unfinished_at_generation_limit": candidate_unfinished,
                    "assessment": assessment,
                    "observation": _substantive_observation(
                        truncation_source["observation"],
                        f"{case_id}.truncation observation",
                    ),
                },
            }
        )
    actual_ids = tuple(value["case_id"] for value in normalized_reviews)
    if actual_ids != expected_ids or len(set(actual_ids)) != len(expected_ids):
        raise Lfm2ParentRelativeManualGateError(
            "Qualitative reviews must cover every packet case exactly once in order."
        )
    narrative_fields: dict[str, set[str]] = {
        "reason": set(),
        "truncation_consideration": set(),
        **{criterion: set() for criterion in QUALITATIVE_CRITERIA},
    }
    for row in normalized_reviews:
        case_id = row["case_id"]
        case_narratives = {
            "reason": row["reason"],
            "truncation_consideration": row["truncation_consideration"]["observation"],
            **row["criterion_observations"],
        }
        local_keys = {
            name: _narrative_key(value, case_id)
            for name, value in case_narratives.items()
        }
        if len(set(local_keys.values())) != len(local_keys):
            raise Lfm2ParentRelativeManualGateError(
                f"Qualitative observations for {case_id} are duplicated rather "
                "than criterion-specific."
            )
        for name, key in local_keys.items():
            if key in narrative_fields[name]:
                raise Lfm2ParentRelativeManualGateError(
                    f"Qualitative {name} observations contain duplicate boilerplate "
                    "across cases."
                )
            narrative_fields[name].add(key)
    if not isinstance(holistic_suite_conclusion, Mapping) or set(
        holistic_suite_conclusion
    ) != {
        "conclusion",
        "reason",
        "strengths",
        "limitations",
        "material_regression_case_ids",
    }:
        raise Lfm2ParentRelativeManualGateError(
            "Holistic suite-conclusion fields differ from the schema."
        )
    conclusion = _text(
        holistic_suite_conclusion["conclusion"],
        "holistic suite conclusion",
    )
    if conclusion not in HOLISTIC_SUITE_CONCLUSIONS:
        raise Lfm2ParentRelativeManualGateError(
            "Holistic suite conclusion is unsupported."
        )
    material_ids = _text_list(
        holistic_suite_conclusion["material_regression_case_ids"],
        "material_regression_case_ids",
    )
    expected_material_ids = [
        value["case_id"]
        for value in normalized_reviews
        if value["verdict"] == "material_regression"
    ]
    if material_ids != expected_material_ids:
        raise Lfm2ParentRelativeManualGateError(
            "Suite material-regression IDs must exactly match the qualitative "
            "case verdicts in packet order."
        )
    suite = {
        "conclusion": conclusion,
        "reason": _text(
            holistic_suite_conclusion["reason"],
            "holistic suite reason",
        ),
        "strengths": _text_list(
            holistic_suite_conclusion["strengths"],
            "holistic suite strengths",
        ),
        "limitations": _text_list(
            holistic_suite_conclusion["limitations"],
            "holistic suite limitations",
        ),
        "material_regression_case_ids": material_ids,
    }
    return {
        "status": "complete",
        "qualitative_case_reviews": normalized_reviews,
        "suite_conclusion": suite,
        "qualitative_judgment_is_authoritative": True,
        "numeric_scores_and_thresholds_are_diagnostic_only": True,
        "automatic_scalar_accept_reject_permitted": False,
    }


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
            "qualitative_criteria": list(QUALITATIVE_CRITERIA),
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


def build_all_case_parent_relative_manual_packet(
    manual_packet: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    manifest_document_sha256: str,
) -> dict[str, Any]:
    """Expand the legacy manual subset into one sealed all-case review packet."""

    if isinstance(case_rows, (str, bytes)) or not isinstance(case_rows, Sequence):
        raise Lfm2ParentRelativeManualGateError("case_rows must be a sequence.")
    normalized_rows = []
    for row in case_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "prompt",
            "parent_response",
            "candidate_response",
            "generation_limit",
            "parent_completion",
            "candidate_completion",
        }:
            raise Lfm2ParentRelativeManualGateError(
                "All-case review row fields differ from the schema."
            )
        limit = row["generation_limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise Lfm2ParentRelativeManualGateError(
                "All-case generation_limit must be a positive integer."
            )
        normalized_rows.append(
            {
                "case_id": _text(row["case_id"], "all-case case_id"),
                "prompt": str(row["prompt"]),
                "parent_response": str(row["parent_response"]),
                "candidate_response": str(row["candidate_response"]),
                "generation_limit": limit,
                "parent_completion": normalize_parent_relative_completion_metadata(
                    row["parent_completion"],
                    generation_limit=limit,
                    name=f"{row['case_id']}.parent",
                ),
                "candidate_completion": normalize_parent_relative_completion_metadata(
                    row["candidate_completion"],
                    generation_limit=limit,
                    name=f"{row['case_id']}.candidate",
                ),
            }
        )
    case_ids = [value["case_id"] for value in normalized_rows]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise Lfm2ParentRelativeManualGateError(
            "All-case review rows are empty or contain duplicate case IDs."
        )
    generic_packet = build_parent_relative_manual_packet(manual_packet)
    generic_body = dict(generic_packet)
    generic_body.pop("packet_sha256")
    rubric = dict(generic_body["rubric"])
    rubric.update(
        {
            "review_every_manifest_case": True,
            "one_reason_per_case_required": True,
            "full_parent_and_candidate_response_review_required": True,
            "per_case_generation_limit_bound": True,
            "parent_and_candidate_completion_metadata_bound": True,
            "completion_metadata_is_diagnostic_context_only": True,
            "completion_metadata_automatic_accept_reject_permitted": False,
            "explicit_truncation_consideration_required": True,
            "criterion_specific_observations_required": True,
            "duplicate_boilerplate_observations_permitted": False,
            "scalar_scores_are_diagnostic_only": True,
            "automatic_accept_reject_permitted": False,
            "qualitative_case_verdicts": list(QUALITATIVE_CASE_VERDICTS),
            "qualitative_failure_modes": list(QUALITATIVE_FAILURE_MODES),
            "holistic_suite_conclusions": list(HOLISTIC_SUITE_CONCLUSIONS),
            "qualitative_case_verdicts_are_authoritative": True,
            "holistic_suite_conclusion_is_authoritative": True,
        }
    )
    limit_rows = [
        {
            "case_id": value["case_id"],
            "generation_limit": value["generation_limit"],
        }
        for value in normalized_rows
    ]
    completion_rows = [
        {
            "case_id": value["case_id"],
            "parent_completion": value["parent_completion"],
            "candidate_completion": value["candidate_completion"],
        }
        for value in normalized_rows
    ]
    body = {
        **generic_body,
        "review_scope": "exact_authenticated_manifest_all_cases",
        "all_case_manual_review_mode": True,
        "manifest_sha256": _digest(manifest_sha256, "manifest_sha256"),
        "manifest_document_sha256": _digest(
            manifest_document_sha256,
            "manifest_document_sha256",
        ),
        "case_count": len(normalized_rows),
        "case_ids_sha256": canonical_sha256(case_ids),
        "generation_limit_rows_sha256": canonical_sha256(limit_rows),
        "completion_metadata_rows_sha256": canonical_sha256(completion_rows),
        "legacy_generic_manual_subset_case_count": len(manual_packet["cases"]),
        "legacy_generic_manual_subset_is_diagnostic_only": True,
        "rubric": rubric,
        "cases": normalized_rows,
    }
    return {**body, "packet_sha256": canonical_sha256(body)}


def bind_parent_relative_manual_ratings(
    packet: Mapping[str, Any],
    ratings: Sequence[Mapping[str, Any]],
    *,
    reviewer_id: str,
    used_exact_wording_comparison: bool = False,
    qualitative_case_reviews: Sequence[Mapping[str, Any]] | None = None,
    holistic_suite_conclusion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal paired diagnostics and an optional authoritative qualitative review."""

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
    holistic_review = _normalize_holistic_review(
        packet_body["cases"],
        qualitative_case_reviews,
        holistic_suite_conclusion,
    )
    body = {
        "schema": RATINGS_SCHEMA,
        "packet_sha256": packet_sha,
        "reviewer_id": reviewer_id.strip(),
        "exact_wording_comparison_used": False,
        "ratings": normalized,
        **(
            {
                "qualitative_case_reviews": holistic_review["qualitative_case_reviews"],
                "holistic_suite_conclusion": holistic_review["suite_conclusion"],
                "qualitative_judgment_is_authoritative": True,
                "numeric_scores_and_thresholds_are_diagnostic_only": True,
                "automatic_scalar_accept_reject_permitted": False,
            }
            if holistic_review is not None
            else {}
        ),
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
    holistic_review = _normalize_holistic_review(
        packet_body["cases"],
        ratings_body.get("qualitative_case_reviews"),
        ratings_body.get("holistic_suite_conclusion"),
    )
    if holistic_review is not None and (
        ratings_body.get("qualitative_judgment_is_authoritative") is not True
        or ratings_body.get("numeric_scores_and_thresholds_are_diagnostic_only")
        is not True
        or ratings_body.get("automatic_scalar_accept_reject_permitted") is not False
    ):
        raise Lfm2ParentRelativeManualGateError(
            "Holistic ratings lost their qualitative-authority contract."
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
    retention = (
        None
        if not parent_demonstrated_points
        else capped_candidate_retained_points / parent_demonstrated_points
    )
    diagnostic_goal_met = (
        None
        if retention is None
        else retention >= float(diagnostic_retention_reference)
    )
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
        "holistic_review": (
            holistic_review
            if holistic_review is not None
            else {
                "status": "not_provided",
                "qualitative_judgment_is_authoritative": False,
                "numeric_scores_and_thresholds_are_diagnostic_only": True,
                "automatic_scalar_accept_reject_permitted": False,
            }
        ),
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
            "qualitative_review_present": holistic_review is not None,
            "qualitative_judgment_is_authoritative": holistic_review is not None,
            "numeric_scores_and_thresholds_are_diagnostic_only": True,
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
    "HOLISTIC_SUITE_CONCLUSIONS",
    "PACKET_SCHEMA",
    "PROMOTION_ELIGIBLE_SUITE_CONCLUSION",
    "QUALITATIVE_CRITERIA",
    "QUALITATIVE_CASE_VERDICTS",
    "QUALITATIVE_FAILURE_MODES",
    "TRUNCATION_ASSESSMENTS",
    "RATINGS_SCHEMA",
    "REPORT_SCHEMA",
    "SEVERE_REGRESSION_MINIMUM_DROP",
    "Lfm2ParentRelativeManualGateError",
    "bind_parent_relative_manual_ratings",
    "build_all_case_parent_relative_manual_packet",
    "build_parent_relative_manual_packet",
    "evaluate_parent_relative_manual_retention",
    "normalize_parent_relative_completion_metadata",
]
