"""Deterministic, training-free long-horizon anchors for LFM2 folding.

The policy in this module selects *causal predecessor* trace rows.  It never
uses a gradient, optimizer, backward pass, or candidate model output.  The
caller supplies a sealed set of parent token trajectories and an ordered,
token-authenticated description of the circuit-trace paths.

The output-token alignment is intentionally strict:

* the final sampled ``prefill`` row predicts parent output token ``0``;
* ``decode.t`` predicts parent output token ``t + 1``.

Rows that cannot prove that alignment are rejected instead of guessed.
Semantic tags are derived from caller-supplied per-token decoded text, never
from vocabulary IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CASE_ENVELOPE_SCHEMA = "neural-forge-lfm2-long-horizon-case-envelope-v1"
ANCHOR_PLAN_SCHEMA = "neural-forge-lfm2-long-horizon-anchor-plan-v1"
ANCHOR_TAGGING_SCHEMA = "neural-forge-lfm2-long-horizon-row-tags-v1"
ANCHOR_POLICY_SCHEMA = "neural-forge-lfm2-long-horizon-anchor-policy-v1"
COMPOSITE_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-long-horizon-composite-evidence-v1"
)

ANCHOR_TAGS = (
    "termination",
    "code_delimiter",
    "code_operator",
    "arithmetic_answer",
    "instruction",
    "rare_symbol",
    "general_coverage",
)
SPECIALIZED_ANCHOR_TAGS = ANCHOR_TAGS[:-1]
EVIDENCE_SPLITS = ("selection", "fit", "validation")

_QUOTAS = {
    "selection": {
        "termination": 24,
        "code_delimiter": 24,
        "code_operator": 16,
        "arithmetic_answer": 16,
        "instruction": 24,
        "rare_symbol": 24,
        "general_coverage": 128,
    },
    "fit": {
        "termination": 64,
        "code_delimiter": 64,
        "code_operator": 48,
        "arithmetic_answer": 32,
        "instruction": 48,
        "rare_symbol": 48,
        "general_coverage": 208,
    },
    "validation": {
        "termination": 16,
        "code_delimiter": 16,
        "code_operator": 12,
        "arithmetic_answer": 8,
        "instruction": 12,
        "rare_symbol": 12,
        "general_coverage": 52,
    },
}

ANCHOR_QUOTAS = {
    split: dict(quotas)
    for split, quotas in _QUOTAS.items()
}

_CASE_SPLIT_COUNTS = {
    "selection": 3,
    "fit": 4,
    "validation": 1,
}
_CASE_SPLIT_SALT = "neural-forge-lfm2-long-horizon-case-split-v1"
_SHA256_LENGTH = 64
_TRACE_PATH_PATTERN = re.compile(
    r"^batch\.(?P<batch>\d{6})/"
    r"(?P<phase>prefill|decode\.(?P<step>\d{3}))/"
    r"row\.(?P<row>\d{4})$"
)
_DECODE_PHASE_PATTERN = re.compile(r"^decode\.(\d{3})$")
_CODE_DELIMITER_PATTERN = re.compile(r"`{3,}|~{3,}|[()[\]{}:;,'\"]")
_CODE_OPERATOR_PATTERN = re.compile(
    r"\*\*=|//=|<<=|>>=|"
    r"==|!=|<=|>=|->|=>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|"
    r"\*\*|//|<<|>>|:=|"
    r"[+\-*/%=<>!&|^~]"
)
_INSTRUCTION_START_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:[-*]\s+|\d+[.)]\s+)?"
    r"(?:return|use|write|calculate|compute|explain|list|create|implement|"
    r"print|set|check|call|raise|add|remove|replace|sort|open|close|run|"
    r"give|provide|summarize|compare|convert|do|make)\b"
)
_EOS_TEXT_MARKERS = frozenset(
    {
        "<eos>",
        "</s>",
        "<|eos|>",
        "<|endoftext|>",
        "<|im_end|>",
        "[eos]",
    }
)

_ALLOWED_CASE_FIELDS = frozenset(
    {
        "case_id",
        "source_split",
        "domain",
        "parent_output_ids",
        "parent_output_token_texts",
        "parent_output_text",
        "arithmetic_answer_correct",
        "arithmetic_answer_token_indices",
        "termination_token_indices",
        "termination_confirmed",
        "instruction_start_token_indices",
    }
)
_REQUIRED_CASE_FIELDS = frozenset(
    {
        "case_id",
        "source_split",
        "domain",
        "parent_output_ids",
        "parent_output_token_texts",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "source_identity_sha256",
        "tokenizer_fingerprint",
        "eos_token_ids",
        "cases",
        "envelope_sha256",
    }
)
_TRACE_DESCRIPTOR_FIELDS = frozenset(
    {
        "path",
        "case_id",
        "phase",
        "row_index",
        "predicted_output_index",
        "predicted_token_id",
    }
)


class Lfm2LongHorizonAnchorError(RuntimeError):
    """Long-horizon evidence is incomplete, ambiguous, or unauthenticated."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2LongHorizonAnchorError(
            "Long-horizon evidence must be canonical JSON data."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2LongHorizonAnchorError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2LongHorizonAnchorError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _optional_index_tuple(
    value: Any,
    *,
    name: str,
    token_count: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise Lfm2LongHorizonAnchorError(f"{name} must be a JSON array.")
    indices = tuple(
        _strict_nonnegative_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if indices != tuple(sorted(set(indices))):
        raise Lfm2LongHorizonAnchorError(
            f"{name} must contain sorted unique indices."
        )
    if any(index >= token_count for index in indices):
        raise Lfm2LongHorizonAnchorError(
            f"{name} contains an out-of-range token index."
        )
    return indices


def _policy_descriptor() -> dict[str, Any]:
    return {
        "schema": ANCHOR_POLICY_SCHEMA,
        "output_alignment": {
            "final_prefill_predicts_output_index": 0,
            "decode_t_predicts_output_index": "t+1",
        },
        "eligible_source_split": "heldout_recovery",
        "excluded_source_split": "heldout_manual",
        "domain_case_counts": {
            "math": dict(_CASE_SPLIT_COUNTS),
            "code": dict(_CASE_SPLIT_COUNTS),
        },
        "case_split": {
            "method": "ascending-sha256",
            "salt": _CASE_SPLIT_SALT,
            "case_disjoint": True,
        },
        "quotas": {
            split: dict(_QUOTAS[split])
            for split in EVIDENCE_SPLITS
        },
        "allocation": {
            "unique_first": True,
            "selection_fit_cycle": True,
            "validation_cycle": False,
            "unavailable_specialized_quota_spills_to": "general_coverage",
            "repeated_rows_encoded_as_integer_weights": True,
        },
        "semantic_source": "caller-supplied-per-token-decoded-text",
        "gradients_used": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


ANCHOR_POLICY = _policy_descriptor()
ANCHOR_POLICY_IDENTITY = _json_sha256(ANCHOR_POLICY)


@dataclass(frozen=True, slots=True)
class _Case:
    case_id: str
    source_split: str
    domain: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    output_text: str
    arithmetic_answer_correct: bool | None
    arithmetic_answer_indices: tuple[int, ...] | None
    termination_indices: tuple[int, ...]
    termination_confirmed: bool
    instruction_indices: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class _TracePath:
    index: int
    path: str
    batch_index: int
    case_id: str
    phase: str
    row_index: int
    predicted_output_index: int | None
    predicted_token_id: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "case_id": self.case_id,
            "phase": self.phase,
            "row_index": self.row_index,
            "predicted_output_index": self.predicted_output_index,
            "predicted_token_id": self.predicted_token_id,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonTaggedRow:
    """One authenticated predecessor path and all applicable anchor tags."""

    path_index: int
    path: str
    case_id: str
    evidence_split: str
    phase: str
    predicted_output_index: int
    predicted_token_id: int
    tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path_index": self.path_index,
            "path": self.path,
            "case_id": self.case_id,
            "evidence_split": self.evidence_split,
            "phase": self.phase,
            "predicted_output_index": self.predicted_output_index,
            "predicted_token_id": self.predicted_token_id,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonTagExclusion:
    """A deliberate reason that a semantic tag was not trusted."""

    case_id: str
    tag: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "tag": self.tag,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonTagging:
    """Authenticated row tags plus case-split and identity evidence."""

    envelope_sha256: str
    policy_identity: str
    ordered_path_sha256: str
    composite_evidence_identity: str
    case_splits: tuple[tuple[str, str], ...]
    excluded_case_ids: tuple[str, ...]
    rows: tuple[Lfm2LongHorizonTaggedRow, ...]
    exclusions: tuple[Lfm2LongHorizonTagExclusion, ...]
    tagging_sha256: str

    @property
    def case_split_map(self) -> dict[str, str]:
        return dict(self.case_splits)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": ANCHOR_TAGGING_SCHEMA,
            "envelope_sha256": self.envelope_sha256,
            "policy_identity": self.policy_identity,
            "ordered_path_sha256": self.ordered_path_sha256,
            "composite_evidence_identity": self.composite_evidence_identity,
            "case_splits": [
                {"case_id": case_id, "split": split}
                for case_id, split in self.case_splits
            ],
            "excluded_case_ids": list(self.excluded_case_ids),
            "rows": [row.as_dict() for row in self.rows],
            "exclusions": [
                exclusion.as_dict() for exclusion in self.exclusions
            ],
        }
        return {**body, "tagging_sha256": self.tagging_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonQuotaFulfillment:
    """How one requested category quota was allocated or spilled."""

    tag: str
    requested_quota: int
    spill_received: int
    effective_quota: int
    candidate_unique_count: int
    allocated_occurrences: int
    allocated_unique_count: int
    cycled_occurrences: int
    spilled_to_general: int
    expanded_path_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "requested_quota": self.requested_quota,
            "spill_received": self.spill_received,
            "effective_quota": self.effective_quota,
            "candidate_unique_count": self.candidate_unique_count,
            "allocated_occurrences": self.allocated_occurrences,
            "allocated_unique_count": self.allocated_unique_count,
            "cycled_occurrences": self.cycled_occurrences,
            "spilled_to_general": self.spilled_to_general,
            "quota_accounted": (
                self.allocated_occurrences + self.spilled_to_general
                == self.effective_quota
            ),
            "expanded_path_sha256": self.expanded_path_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonMultiplicity:
    """An ordered path index and its integer least-squares row weight."""

    path_index: int
    path: str
    integer_weight: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path_index": self.path_index,
            "path": self.path,
            "integer_weight": self.integer_weight,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonSplitPlan:
    """One case-disjoint selection, fit, or validation allocation."""

    split: str
    case_ids: tuple[str, ...]
    indices: tuple[int, ...]
    integer_weights: tuple[int, ...]
    expanded_indices: tuple[int, ...]
    multiplicities: tuple[Lfm2LongHorizonMultiplicity, ...]
    quota_fulfillment: tuple[Lfm2LongHorizonQuotaFulfillment, ...]
    total_weight: int
    unique_count: int
    repeated_occurrence_count: int
    unique_path_sha256: str
    expanded_path_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "case_ids": list(self.case_ids),
            "indices": list(self.indices),
            "integer_weights": list(self.integer_weights),
            "expanded_indices": list(self.expanded_indices),
            "multiplicities": [
                multiplicity.as_dict() for multiplicity in self.multiplicities
            ],
            "quota_fulfillment": [
                fulfillment.as_dict()
                for fulfillment in self.quota_fulfillment
            ],
            "total_weight": self.total_weight,
            "unique_count": self.unique_count,
            "repeated_occurrence_count": self.repeated_occurrence_count,
            "unique_path_sha256": self.unique_path_sha256,
            "expanded_path_sha256": self.expanded_path_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonAnchorPlan:
    """Complete zero-training anchor plan for all three evidence splits."""

    envelope_sha256: str
    policy_identity: str
    ordered_path_sha256: str
    composite_evidence_identity: str
    tagging_sha256: str
    excluded_case_ids: tuple[str, ...]
    selection: Lfm2LongHorizonSplitPlan
    fit: Lfm2LongHorizonSplitPlan
    validation: Lfm2LongHorizonSplitPlan
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": ANCHOR_PLAN_SCHEMA,
            "envelope_sha256": self.envelope_sha256,
            "policy_identity": self.policy_identity,
            "ordered_path_sha256": self.ordered_path_sha256,
            "composite_evidence_identity": self.composite_evidence_identity,
            "tagging_sha256": self.tagging_sha256,
            "excluded_case_ids": list(self.excluded_case_ids),
            "selection": self.selection.as_dict(),
            "fit": self.fit.as_dict(),
            "validation": self.validation.as_dict(),
            "zero_training": {
                "gradients_used": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
            },
        }
        return {**body, "plan_sha256": self.plan_sha256}


def seal_lfm2_long_horizon_case_envelope(
    cases: Sequence[Mapping[str, Any]],
    *,
    source_identity_sha256: str,
    tokenizer_fingerprint: str,
    eos_token_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Build the exact sealed input envelope accepted by this policy."""

    source_identity = _require_digest(
        source_identity_sha256,
        "source_identity_sha256",
    )
    tokenizer_identity = _require_digest(
        tokenizer_fingerprint,
        "tokenizer_fingerprint",
    )
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise Lfm2LongHorizonAnchorError("cases must be an ordered sequence.")
    copied_cases = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise Lfm2LongHorizonAnchorError(
                f"Case {index} must be a mapping."
            )
        copied_cases.append(json.loads(_canonical_json(dict(case))))
    eos = tuple(
        _strict_nonnegative_int(value, f"eos_token_ids[{index}]")
        for index, value in enumerate(eos_token_ids)
    )
    if eos != tuple(sorted(set(eos))):
        raise Lfm2LongHorizonAnchorError(
            "eos_token_ids must be sorted and unique."
        )
    body = {
        "schema": CASE_ENVELOPE_SCHEMA,
        "source_identity_sha256": source_identity,
        "tokenizer_fingerprint": tokenizer_identity,
        "eos_token_ids": list(eos),
        "cases": copied_cases,
    }
    envelope = {**body, "envelope_sha256": _json_sha256(body)}
    _authenticate_case_envelope(envelope)
    return envelope


def _authenticate_case_envelope(
    value: Mapping[str, Any],
) -> tuple[tuple[_Case, ...], tuple[int, ...], str]:
    if not isinstance(value, Mapping):
        raise Lfm2LongHorizonAnchorError(
            "Long-horizon case envelope must be a mapping."
        )
    if set(value) != _ENVELOPE_FIELDS:
        raise Lfm2LongHorizonAnchorError(
            "Long-horizon case envelope fields are incomplete or ambiguous."
        )
    if value.get("schema") != CASE_ENVELOPE_SCHEMA:
        raise Lfm2LongHorizonAnchorError(
            "Long-horizon case envelope schema is unsupported."
        )
    _require_digest(value.get("source_identity_sha256"), "source_identity_sha256")
    _require_digest(value.get("tokenizer_fingerprint"), "tokenizer_fingerprint")
    claimed = _require_digest(value.get("envelope_sha256"), "envelope_sha256")
    body = dict(value)
    body.pop("envelope_sha256")
    if _json_sha256(body) != claimed:
        raise Lfm2LongHorizonAnchorError(
            "Long-horizon case envelope identity does not match its contents."
        )
    eos_value = value.get("eos_token_ids")
    if not isinstance(eos_value, list):
        raise Lfm2LongHorizonAnchorError("eos_token_ids must be a JSON array.")
    eos = tuple(
        _strict_nonnegative_int(item, f"eos_token_ids[{index}]")
        for index, item in enumerate(eos_value)
    )
    if eos != tuple(sorted(set(eos))):
        raise Lfm2LongHorizonAnchorError(
            "eos_token_ids must be sorted and unique."
        )
    rows = value.get("cases")
    if not isinstance(rows, list) or not rows:
        raise Lfm2LongHorizonAnchorError(
            "Long-horizon case envelope must contain cases."
        )
    result = []
    seen = set()
    for row_index, row in enumerate(rows):
        result.append(_normalize_case(row, row_index=row_index, seen=seen))
    recovery_by_domain = {
        domain: [
            case
            for case in result
            if case.source_split == "heldout_recovery" and case.domain == domain
        ]
        for domain in ("math", "code")
    }
    for domain, domain_cases in recovery_by_domain.items():
        if len(domain_cases) != sum(_CASE_SPLIT_COUNTS.values()):
            raise Lfm2LongHorizonAnchorError(
                f"Expected exactly eight heldout_recovery {domain} cases."
            )
    unsupported = [
        case.case_id
        for case in result
        if case.source_split not in {"heldout_recovery", "heldout_manual"}
        or (
            case.source_split == "heldout_recovery"
            and case.domain not in {"math", "code"}
        )
    ]
    if unsupported:
        raise Lfm2LongHorizonAnchorError(
            "Envelope contains unsupported evidence cases: "
            + ", ".join(unsupported)
        )
    return tuple(result), eos, claimed


def _normalize_case(
    row: Any,
    *,
    row_index: int,
    seen: set[str],
) -> _Case:
    if not isinstance(row, Mapping):
        raise Lfm2LongHorizonAnchorError(
            f"Case {row_index} must be a mapping."
        )
    fields = set(row)
    if not _REQUIRED_CASE_FIELDS.issubset(fields) or not fields.issubset(
        _ALLOWED_CASE_FIELDS
    ):
        raise Lfm2LongHorizonAnchorError(
            f"Case {row_index} fields are incomplete or ambiguous."
        )
    case_id = row.get("case_id")
    source_split = row.get("source_split")
    domain = row.get("domain")
    if (
        not isinstance(case_id, str)
        or not case_id.strip()
        or case_id in seen
    ):
        raise Lfm2LongHorizonAnchorError(
            f"Case {row_index} has an invalid or duplicate case_id."
        )
    if not isinstance(source_split, str) or not source_split:
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} has an invalid source_split."
        )
    if not isinstance(domain, str) or not domain:
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} has an invalid domain."
        )
    seen.add(case_id)
    ids_value = row.get("parent_output_ids")
    texts_value = row.get("parent_output_token_texts")
    if not isinstance(ids_value, list) or not isinstance(texts_value, list):
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} token IDs and token texts must be arrays."
        )
    token_ids = tuple(
        _strict_nonnegative_int(value, f"{case_id}.parent_output_ids[{index}]")
        for index, value in enumerate(ids_value)
    )
    if any(not isinstance(value, str) for value in texts_value):
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} token texts must all be strings."
        )
    token_texts = tuple(texts_value)
    if len(token_ids) != len(token_texts):
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} token IDs and token texts are not aligned."
        )
    if source_split == "heldout_recovery" and not token_ids:
        raise Lfm2LongHorizonAnchorError(
            f"Recovery case {case_id!r} has no parent output tokens."
        )
    output_text = "".join(token_texts)
    supplied_text = row.get("parent_output_text")
    if supplied_text is not None and (
        not isinstance(supplied_text, str) or supplied_text != output_text
    ):
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} token texts do not reproduce parent_output_text."
        )
    arithmetic_correct = row.get("arithmetic_answer_correct")
    if arithmetic_correct is not None and not isinstance(
        arithmetic_correct,
        bool,
    ):
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} arithmetic correctness must be bool or null."
        )
    arithmetic_indices = _optional_index_tuple(
        row.get("arithmetic_answer_token_indices"),
        name=f"{case_id}.arithmetic_answer_token_indices",
        token_count=len(token_ids),
    )
    if domain != "math" and (
        "arithmetic_answer_correct" in fields
        or "arithmetic_answer_token_indices" in fields
    ):
        raise Lfm2LongHorizonAnchorError(
            f"Non-math case {case_id!r} cannot claim an arithmetic answer."
        )
    if arithmetic_correct is False and arithmetic_indices:
        raise Lfm2LongHorizonAnchorError(
            f"Known arithmetic mismatch {case_id!r} cannot claim answer indices."
        )
    termination_indices = _optional_index_tuple(
        row.get("termination_token_indices"),
        name=f"{case_id}.termination_token_indices",
        token_count=len(token_ids),
    )
    termination_confirmed = row.get("termination_confirmed", False)
    if not isinstance(termination_confirmed, bool):
        raise Lfm2LongHorizonAnchorError(
            f"Case {case_id!r} termination_confirmed must be bool."
        )
    instruction_indices = _optional_index_tuple(
        row.get("instruction_start_token_indices"),
        name=f"{case_id}.instruction_start_token_indices",
        token_count=len(token_ids),
    )
    return _Case(
        case_id=case_id,
        source_split=source_split,
        domain=domain,
        token_ids=token_ids,
        token_texts=token_texts,
        output_text=output_text,
        arithmetic_answer_correct=arithmetic_correct,
        arithmetic_answer_indices=arithmetic_indices,
        termination_indices=termination_indices or (),
        termination_confirmed=termination_confirmed,
        instruction_indices=instruction_indices,
    )


def _case_split_assignments(cases: Sequence[_Case]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for domain in ("math", "code"):
        domain_cases = [
            case
            for case in cases
            if case.source_split == "heldout_recovery" and case.domain == domain
        ]
        ordered = sorted(
            domain_cases,
            key=lambda case: (
                hashlib.sha256(
                    (
                        f"{_CASE_SPLIT_SALT}\0{domain}\0{case.case_id}"
                    ).encode("utf-8")
                ).hexdigest(),
                case.case_id,
            ),
        )
        cursor = 0
        for split in EVIDENCE_SPLITS:
            count = _CASE_SPLIT_COUNTS[split]
            for case in ordered[cursor : cursor + count]:
                assignments[case.case_id] = split
            cursor += count
    if len(assignments) != 16:
        raise Lfm2LongHorizonAnchorError(
            "Case split did not assign exactly sixteen recovery cases."
        )
    return assignments


def _normalize_trace_paths(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    cases: Sequence[_Case],
) -> tuple[tuple[_TracePath, ...], str]:
    if isinstance(descriptors, (str, bytes)) or not isinstance(
        descriptors,
        Sequence,
    ):
        raise Lfm2LongHorizonAnchorError(
            "trace_path_descriptors must be an ordered sequence."
        )
    case_map = {case.case_id: case for case in cases}
    paths = []
    seen_paths = set()
    batch_cases: dict[int, str] = {}
    case_batches: dict[str, int] = {}
    previous_key: tuple[int, int, int] | None = None
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping) or set(
            descriptor
        ) != _TRACE_DESCRIPTOR_FIELDS:
            raise Lfm2LongHorizonAnchorError(
                f"Trace descriptor {index} fields are incomplete or ambiguous."
            )
        path = descriptor.get("path")
        case_id = descriptor.get("case_id")
        phase = descriptor.get("phase")
        if not isinstance(path, str) or not isinstance(case_id, str):
            raise Lfm2LongHorizonAnchorError(
                f"Trace descriptor {index} path and case_id must be strings."
            )
        if case_id not in case_map:
            raise Lfm2LongHorizonAnchorError(
                f"Trace descriptor {index} references an unknown case."
            )
        if path in seen_paths:
            raise Lfm2LongHorizonAnchorError(
                f"Trace descriptor path {path!r} is duplicated."
            )
        match = _TRACE_PATH_PATTERN.fullmatch(path)
        if match is None or phase != match.group("phase"):
            raise Lfm2LongHorizonAnchorError(
                f"Trace descriptor {index} path/phase mapping is invalid."
            )
        batch_index = int(match.group("batch"))
        row_index = _strict_nonnegative_int(
            descriptor.get("row_index"),
            f"trace_path_descriptors[{index}].row_index",
        )
        if row_index != int(match.group("row")):
            raise Lfm2LongHorizonAnchorError(
                f"Trace descriptor {index} path/row mapping is invalid."
            )
        if phase != "prefill" and row_index != 0:
            raise Lfm2LongHorizonAnchorError(
                f"Cached decode descriptor {index} must have row_index zero."
            )
        predicted_index_value = descriptor.get("predicted_output_index")
        predicted_token_value = descriptor.get("predicted_token_id")
        if predicted_index_value is None or predicted_token_value is None:
            if predicted_index_value is not None or predicted_token_value is not None:
                raise Lfm2LongHorizonAnchorError(
                    f"Trace descriptor {index} has a partial token mapping."
                )
            predicted_index = None
            predicted_token = None
        else:
            predicted_index = _strict_nonnegative_int(
                predicted_index_value,
                f"trace_path_descriptors[{index}].predicted_output_index",
            )
            predicted_token = _strict_nonnegative_int(
                predicted_token_value,
                f"trace_path_descriptors[{index}].predicted_token_id",
            )
        prior_case = batch_cases.setdefault(batch_index, case_id)
        if prior_case != case_id:
            raise Lfm2LongHorizonAnchorError(
                f"Trace batch {batch_index} maps to multiple cases."
            )
        prior_batch = case_batches.setdefault(case_id, batch_index)
        if prior_batch != batch_index:
            raise Lfm2LongHorizonAnchorError(
                f"Case {case_id!r} maps to multiple trace batches."
            )
        phase_order = (
            0
            if phase == "prefill"
            else 1 + int(_DECODE_PHASE_PATTERN.fullmatch(phase).group(1))
        )
        ordering_key = (batch_index, phase_order, row_index)
        if previous_key is not None and ordering_key <= previous_key:
            raise Lfm2LongHorizonAnchorError(
                "Trace descriptors are not in strict circuit-path order."
            )
        previous_key = ordering_key
        paths.append(
            _TracePath(
                index=index,
                path=path,
                batch_index=batch_index,
                case_id=case_id,
                phase=phase,
                row_index=row_index,
                predicted_output_index=predicted_index,
                predicted_token_id=predicted_token,
            )
        )
        seen_paths.add(path)
    _validate_trace_alignment(tuple(paths), cases=cases)
    normalized = [path.as_dict() for path in paths]
    return tuple(paths), _json_sha256(normalized)


def _validate_trace_alignment(
    paths: tuple[_TracePath, ...],
    *,
    cases: Sequence[_Case],
) -> None:
    by_case: dict[str, list[_TracePath]] = {}
    for path in paths:
        by_case.setdefault(path.case_id, []).append(path)
    for case in cases:
        case_paths = by_case.get(case.case_id, [])
        if case.source_split == "heldout_manual" and not case_paths:
            continue
        if case.source_split == "heldout_recovery" and not case_paths:
            raise Lfm2LongHorizonAnchorError(
                f"Recovery case {case.case_id!r} has no trace paths."
            )
        prefill = [path for path in case_paths if path.phase == "prefill"]
        if case.token_ids and not prefill:
            raise Lfm2LongHorizonAnchorError(
                f"Case {case.case_id!r} has no prefill predecessor."
            )
        final_prefill_row = max(
            (path.row_index for path in prefill),
            default=None,
        )
        mapped: dict[int, _TracePath] = {}
        for path in case_paths:
            predicted_index = path.predicted_output_index
            if path.phase == "prefill":
                if path.row_index == final_prefill_row:
                    if predicted_index != 0:
                        raise Lfm2LongHorizonAnchorError(
                            f"Final prefill for {case.case_id!r} must predict token 0."
                        )
                elif predicted_index is not None:
                    raise Lfm2LongHorizonAnchorError(
                        f"Non-final prefill for {case.case_id!r} claims an output token."
                    )
            else:
                step_match = _DECODE_PHASE_PATTERN.fullmatch(path.phase)
                if step_match is None:  # pragma: no cover - normalized above
                    raise Lfm2LongHorizonAnchorError("Invalid decode phase.")
                expected_index = int(step_match.group(1)) + 1
                if expected_index < len(case.token_ids):
                    if predicted_index != expected_index:
                        raise Lfm2LongHorizonAnchorError(
                            f"{path.phase} for {case.case_id!r} must predict "
                            f"output token {expected_index}."
                        )
                elif predicted_index is not None:
                    raise Lfm2LongHorizonAnchorError(
                        f"{path.phase} for {case.case_id!r} predicts beyond "
                        "the sealed parent output."
                    )
            if predicted_index is None:
                continue
            if predicted_index >= len(case.token_ids):
                raise Lfm2LongHorizonAnchorError(
                    f"Trace path for {case.case_id!r} predicts an out-of-range token."
                )
            if path.predicted_token_id != case.token_ids[predicted_index]:
                raise Lfm2LongHorizonAnchorError(
                    f"Trace token ID differs from sealed case {case.case_id!r}."
                )
            if predicted_index in mapped:
                raise Lfm2LongHorizonAnchorError(
                    f"Case {case.case_id!r} has ambiguous predecessor rows for "
                    f"output token {predicted_index}."
                )
            mapped[predicted_index] = path
        if case.source_split == "heldout_recovery" and set(mapped) != set(
            range(len(case.token_ids))
        ):
            raise Lfm2LongHorizonAnchorError(
                f"Recovery case {case.case_id!r} does not map every output token "
                "to exactly one causal predecessor."
            )


def _token_character_ranges(case: _Case) -> tuple[tuple[int, int], ...]:
    cursor = 0
    result = []
    for text in case.token_texts:
        end = cursor + len(text)
        result.append((cursor, end))
        cursor = end
    return tuple(result)


def _indices_overlapping_spans(
    ranges: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
) -> set[int]:
    result = set()
    for token_index, (token_start, token_end) in enumerate(ranges):
        if token_start == token_end:
            continue
        if any(
            token_start < span_end and span_start < token_end
            for span_start, span_end in spans
        ):
            result.add(token_index)
    return result


def _pattern_token_indices(case: _Case, pattern: re.Pattern[str]) -> set[int]:
    spans = [match.span() for match in pattern.finditer(case.output_text)]
    return _indices_overlapping_spans(_token_character_ranges(case), spans)


def _has_symbol(text: str) -> bool:
    return any(
        not character.isspace()
        and (
            ord(character) > 127
            or
            unicodedata.category(character).startswith(("P", "S"))
            or not character.isalnum()
        )
        for character in text
    )


def _semantic_indices(
    cases: Sequence[_Case],
    *,
    eos_token_ids: Sequence[int],
) -> tuple[dict[str, dict[str, set[int]]], tuple[Lfm2LongHorizonTagExclusion, ...]]:
    eligible = [
        case for case in cases if case.source_split == "heldout_recovery"
    ]
    token_frequency = Counter(
        token_id for case in eligible for token_id in case.token_ids
    )
    eos_ids = set(eos_token_ids)
    result: dict[str, dict[str, set[int]]] = {}
    exclusions = []
    for case in eligible:
        tags = {tag: set() for tag in SPECIALIZED_ANCHOR_TAGS}
        tags["termination"].update(case.termination_indices)
        tags["termination"].update(
            index
            for index, (token_id, text) in enumerate(
                zip(case.token_ids, case.token_texts)
            )
            if token_id in eos_ids or text.strip().lower() in _EOS_TEXT_MARKERS
        )
        if case.termination_confirmed and case.token_ids:
            tags["termination"].add(len(case.token_ids) - 1)
        if case.domain == "code":
            tags["code_delimiter"] = _pattern_token_indices(
                case,
                _CODE_DELIMITER_PATTERN,
            )
            tags["code_operator"] = _pattern_token_indices(
                case,
                _CODE_OPERATOR_PATTERN,
            )
        if case.domain == "math":
            if case.arithmetic_answer_correct is True:
                if case.arithmetic_answer_indices:
                    tags["arithmetic_answer"].update(
                        case.arithmetic_answer_indices
                    )
                else:
                    exclusions.append(
                        Lfm2LongHorizonTagExclusion(
                            case_id=case.case_id,
                            tag="arithmetic_answer",
                            reason="correct-answer-claimed-without-exact-token-indices",
                        )
                    )
            elif case.arithmetic_answer_correct is False:
                exclusions.append(
                    Lfm2LongHorizonTagExclusion(
                        case_id=case.case_id,
                        tag="arithmetic_answer",
                        reason="known-arithmetic-mismatch",
                    )
                )
            else:
                exclusions.append(
                    Lfm2LongHorizonTagExclusion(
                        case_id=case.case_id,
                        tag="arithmetic_answer",
                        reason="arithmetic-correctness-not-attested",
                    )
                )
        if case.instruction_indices is not None:
            tags["instruction"].update(case.instruction_indices)
        else:
            tags["instruction"] = _pattern_token_indices(
                case,
                _INSTRUCTION_START_PATTERN,
            )
        tags["rare_symbol"].update(
            index
            for index, (token_id, text) in enumerate(
                zip(case.token_ids, case.token_texts)
            )
            if token_frequency[token_id] == 1 and _has_symbol(text)
        )
        result[case.case_id] = tags
    return result, tuple(exclusions)


def lfm2_long_horizon_composite_evidence_identity(
    *,
    envelope_sha256: str,
    ordered_path_sha256: str,
    policy_identity: str = ANCHOR_POLICY_IDENTITY,
) -> str:
    """Bind the sealed cases, exact path order, and immutable policy."""

    body = {
        "schema": COMPOSITE_EVIDENCE_SCHEMA,
        "envelope_sha256": _require_digest(
            envelope_sha256,
            "envelope_sha256",
        ),
        "ordered_path_sha256": _require_digest(
            ordered_path_sha256,
            "ordered_path_sha256",
        ),
        "policy_identity": _require_digest(
            policy_identity,
            "policy_identity",
        ),
    }
    return _json_sha256(body)


def tag_lfm2_long_horizon_rows(
    case_envelope: Mapping[str, Any],
    trace_path_descriptors: Sequence[Mapping[str, Any]],
    *,
    expected_envelope_sha256: str | None = None,
    expected_ordered_path_sha256: str | None = None,
    expected_policy_identity: str = ANCHOR_POLICY_IDENTITY,
) -> Lfm2LongHorizonTagging:
    """Authenticate and tag exact causal predecessors without model execution."""

    cases, eos_ids, envelope_sha256 = _authenticate_case_envelope(case_envelope)
    if expected_envelope_sha256 is not None and envelope_sha256 != _require_digest(
        expected_envelope_sha256,
        "expected_envelope_sha256",
    ):
        raise Lfm2LongHorizonAnchorError(
            "Case envelope differs from the expected identity."
        )
    if _require_digest(
        expected_policy_identity,
        "expected_policy_identity",
    ) != ANCHOR_POLICY_IDENTITY:
        raise Lfm2LongHorizonAnchorError(
            "Requested long-horizon anchor policy is not this implementation."
        )
    paths, ordered_path_sha256 = _normalize_trace_paths(
        trace_path_descriptors,
        cases=cases,
    )
    if (
        expected_ordered_path_sha256 is not None
        and ordered_path_sha256
        != _require_digest(
            expected_ordered_path_sha256,
            "expected_ordered_path_sha256",
        )
    ):
        raise Lfm2LongHorizonAnchorError(
            "Ordered trace paths differ from the expected identity."
        )
    case_splits = _case_split_assignments(cases)
    semantic, exclusions = _semantic_indices(cases, eos_token_ids=eos_ids)
    tagged_rows = []
    for path in paths:
        output_index = path.predicted_output_index
        if path.case_id not in case_splits or output_index is None:
            continue
        tags = tuple(
            tag
            for tag in SPECIALIZED_ANCHOR_TAGS
            if output_index in semantic[path.case_id][tag]
        ) + ("general_coverage",)
        tagged_rows.append(
            Lfm2LongHorizonTaggedRow(
                path_index=path.index,
                path=path.path,
                case_id=path.case_id,
                evidence_split=case_splits[path.case_id],
                phase=path.phase,
                predicted_output_index=output_index,
                predicted_token_id=int(path.predicted_token_id),
                tags=tags,
            )
        )
    excluded = tuple(
        case.case_id
        for case in cases
        if case.source_split == "heldout_manual"
    )
    composite = lfm2_long_horizon_composite_evidence_identity(
        envelope_sha256=envelope_sha256,
        ordered_path_sha256=ordered_path_sha256,
    )
    case_split_pairs = tuple(
        sorted(case_splits.items(), key=lambda item: item[0])
    )
    body = {
        "schema": ANCHOR_TAGGING_SCHEMA,
        "envelope_sha256": envelope_sha256,
        "policy_identity": ANCHOR_POLICY_IDENTITY,
        "ordered_path_sha256": ordered_path_sha256,
        "composite_evidence_identity": composite,
        "case_splits": [
            {"case_id": case_id, "split": split}
            for case_id, split in case_split_pairs
        ],
        "excluded_case_ids": list(excluded),
        "rows": [row.as_dict() for row in tagged_rows],
        "exclusions": [exclusion.as_dict() for exclusion in exclusions],
    }
    return Lfm2LongHorizonTagging(
        envelope_sha256=envelope_sha256,
        policy_identity=ANCHOR_POLICY_IDENTITY,
        ordered_path_sha256=ordered_path_sha256,
        composite_evidence_identity=composite,
        case_splits=case_split_pairs,
        excluded_case_ids=excluded,
        rows=tuple(tagged_rows),
        exclusions=exclusions,
        tagging_sha256=_json_sha256(body),
    )


def _candidate_order(
    rows: Sequence[Lfm2LongHorizonTaggedRow],
    *,
    case_lengths: Mapping[str, int],
) -> list[Lfm2LongHorizonTaggedRow]:
    return sorted(
        rows,
        key=lambda row: (
            -(
                (row.predicted_output_index + 1) * 1_000_000
                // case_lengths[row.case_id]
            ),
            hashlib.sha256(
                (
                    f"{_CASE_SPLIT_SALT}\0{row.case_id}"
                ).encode("utf-8")
            ).hexdigest(),
            row.predicted_output_index,
            row.path_index,
        ),
    )


def _allocate_split(
    split: str,
    *,
    rows: Sequence[Lfm2LongHorizonTaggedRow],
    all_paths: Mapping[int, str],
    case_lengths: Mapping[str, int],
) -> Lfm2LongHorizonSplitPlan:
    split_rows = [row for row in rows if row.evidence_split == split]
    if not split_rows:
        raise Lfm2LongHorizonAnchorError(
            f"Evidence split {split!r} has no tagged rows."
        )
    case_ids = tuple(sorted({row.case_id for row in split_rows}))
    quotas = _QUOTAS[split]
    expanded: list[int] = []
    used: set[int] = set()
    fulfillments: list[Lfm2LongHorizonQuotaFulfillment] = []
    spill_to_general = 0

    def allocate_tag(
        tag: str,
        requested: int,
        *,
        spill_received: int = 0,
    ) -> None:
        nonlocal spill_to_general
        candidates = _candidate_order(
            [row for row in split_rows if tag in row.tags],
            case_lengths=case_lengths,
        )
        # A path appears at most once in tagged rows, but keep this explicit.
        unique_candidates = list(
            {row.path_index: row for row in candidates}.values()
        )
        target = requested + spill_received
        allocated: list[int] = []
        unseen = [
            row.path_index
            for row in unique_candidates
            if row.path_index not in used
        ]
        already_used = [
            row.path_index
            for row in unique_candidates
            if row.path_index in used
        ]
        if split == "validation":
            allocated.extend(unseen[:target])
        else:
            unique_first = unseen + already_used
            allocated.extend(unique_first[:target])
            remaining = target - len(allocated)
            if remaining > 0 and unique_candidates:
                cycle = [row.path_index for row in unique_candidates]
                allocated.extend(
                    cycle[index % len(cycle)] for index in range(remaining)
                )
        shortage = target - len(allocated)
        if tag == "general_coverage" and shortage:
            raise Lfm2LongHorizonAnchorError(
                f"{split} lacks {shortage} unique general rows and repetition "
                "is forbidden by its evidence policy."
            )
        if tag != "general_coverage":
            spill_to_general += shortage
        seen_for_multiplicity = set(used)
        cycled = 0
        for path_index in allocated:
            if path_index in seen_for_multiplicity:
                cycled += 1
            seen_for_multiplicity.add(path_index)
        expanded.extend(allocated)
        used.update(allocated)
        fulfillments.append(
            Lfm2LongHorizonQuotaFulfillment(
                tag=tag,
                requested_quota=requested,
                spill_received=spill_received,
                effective_quota=target,
                candidate_unique_count=len(unique_candidates),
                allocated_occurrences=len(allocated),
                allocated_unique_count=len(set(allocated)),
                cycled_occurrences=cycled,
                spilled_to_general=shortage,
                expanded_path_sha256=_json_sha256(
                    [all_paths[index] for index in allocated]
                ),
            )
        )

    for tag in SPECIALIZED_ANCHOR_TAGS:
        allocate_tag(tag, quotas[tag])
    allocate_tag(
        "general_coverage",
        quotas["general_coverage"],
        spill_received=spill_to_general,
    )
    expected_total = sum(quotas.values())
    if len(expanded) != expected_total:
        raise Lfm2LongHorizonAnchorError(
            f"{split} anchor allocation has {len(expanded)} occurrences; "
            f"expected exactly {expected_total}."
        )
    counts = Counter(expanded)
    unique_indices = tuple(dict.fromkeys(expanded))
    weights = tuple(counts[index] for index in unique_indices)
    if any(
        isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
        for weight in weights
    ):
        raise Lfm2LongHorizonAnchorError(
            f"{split} produced a nonpositive integer row weight."
        )
    multiplicities = tuple(
        Lfm2LongHorizonMultiplicity(
            path_index=index,
            path=all_paths[index],
            integer_weight=counts[index],
        )
        for index in unique_indices
    )
    if split == "validation" and any(weight != 1 for weight in weights):
        raise Lfm2LongHorizonAnchorError(
            "Validation anchor allocation repeated a trace row."
        )
    return Lfm2LongHorizonSplitPlan(
        split=split,
        case_ids=case_ids,
        indices=unique_indices,
        integer_weights=weights,
        expanded_indices=tuple(expanded),
        multiplicities=multiplicities,
        quota_fulfillment=tuple(fulfillments),
        total_weight=sum(weights),
        unique_count=len(unique_indices),
        repeated_occurrence_count=len(expanded) - len(unique_indices),
        unique_path_sha256=_json_sha256(
            [all_paths[index] for index in unique_indices]
        ),
        expanded_path_sha256=_json_sha256(
            [all_paths[index] for index in expanded]
        ),
    )


def build_lfm2_long_horizon_anchor_plan(
    case_envelope: Mapping[str, Any],
    trace_path_descriptors: Sequence[Mapping[str, Any]],
    *,
    expected_envelope_sha256: str | None = None,
    expected_ordered_path_sha256: str | None = None,
    expected_policy_identity: str = ANCHOR_POLICY_IDENTITY,
) -> Lfm2LongHorizonAnchorPlan:
    """Build exact 256/512/128 weighted anchors without training."""

    tagging = tag_lfm2_long_horizon_rows(
        case_envelope,
        trace_path_descriptors,
        expected_envelope_sha256=expected_envelope_sha256,
        expected_ordered_path_sha256=expected_ordered_path_sha256,
        expected_policy_identity=expected_policy_identity,
    )
    cases, _eos, _envelope_sha256 = _authenticate_case_envelope(case_envelope)
    case_lengths = {
        case.case_id: len(case.token_ids)
        for case in cases
        if case.source_split == "heldout_recovery"
    }
    all_paths = {
        index: str(descriptor["path"])
        for index, descriptor in enumerate(trace_path_descriptors)
    }
    split_plans = {
        split: _allocate_split(
            split,
            rows=tagging.rows,
            all_paths=all_paths,
            case_lengths=case_lengths,
        )
        for split in EVIDENCE_SPLITS
    }
    case_sets = {
        split: set(plan.case_ids) for split, plan in split_plans.items()
    }
    if any(
        case_sets[left].intersection(case_sets[right])
        for left_index, left in enumerate(EVIDENCE_SPLITS)
        for right in EVIDENCE_SPLITS[left_index + 1 :]
    ):
        raise Lfm2LongHorizonAnchorError(
            "Selection, fit, and validation source cases overlap."
        )
    body = {
        "schema": ANCHOR_PLAN_SCHEMA,
        "envelope_sha256": tagging.envelope_sha256,
        "policy_identity": tagging.policy_identity,
        "ordered_path_sha256": tagging.ordered_path_sha256,
        "composite_evidence_identity": tagging.composite_evidence_identity,
        "tagging_sha256": tagging.tagging_sha256,
        "excluded_case_ids": list(tagging.excluded_case_ids),
        "selection": split_plans["selection"].as_dict(),
        "fit": split_plans["fit"].as_dict(),
        "validation": split_plans["validation"].as_dict(),
        "zero_training": {
            "gradients_used": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
    }
    return Lfm2LongHorizonAnchorPlan(
        envelope_sha256=tagging.envelope_sha256,
        policy_identity=tagging.policy_identity,
        ordered_path_sha256=tagging.ordered_path_sha256,
        composite_evidence_identity=tagging.composite_evidence_identity,
        tagging_sha256=tagging.tagging_sha256,
        excluded_case_ids=tagging.excluded_case_ids,
        selection=split_plans["selection"],
        fit=split_plans["fit"],
        validation=split_plans["validation"],
        plan_sha256=_json_sha256(body),
    )
