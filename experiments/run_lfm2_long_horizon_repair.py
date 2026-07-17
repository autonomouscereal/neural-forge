#!/usr/bin/env python3
"""Retry the four rejected LFM2 folds with long-horizon cached evidence.

This runner is intentionally transductive only during *development*: the
sixteen recovery math/code cases supply sealed parent forced trajectories.
The twelve manual cases are excluded from every cache, anchor, fit, and ridge
choice and are the sole promotion gate.  A full 28-case evaluation is emitted
only as a diagnostic after the manual-only gate accepts.

No parent response is generated.  The sealed envelope must already contain
exact prompt token IDs and parent output IDs.  Per-output-token decoded pieces
are derived locally from the tokenizer bound by the envelope; no prompt is
retokenized and no parent response is regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from neural_forge.hf_lfm2_long_horizon_anchors import (
    ANCHOR_POLICY_IDENTITY,
    Lfm2LongHorizonAnchorError,
    Lfm2LongHorizonAnchorPlan,
    Lfm2LongHorizonTagging,
    build_lfm2_long_horizon_anchor_plan,
    seal_lfm2_long_horizon_case_envelope,
    tag_lfm2_long_horizon_rows,
)


RUN_SCHEMA = "neural-forge-lfm2-long-horizon-repair-run-v1"
DATA_CONTRACT_SCHEMA = "neural-forge-lfm2-long-horizon-data-contract-v1"
SOURCE_BINDING_SCHEMA = "neural-forge-lfm2-long-horizon-source-binding-v1"
PROFILE_SCHEMA = "neural-forge-lfm2-long-horizon-evidence-profile-v1"
TRANCHE_SCHEMA = "neural-forge-lfm2-long-horizon-atomic-tranche-v1"
CANDIDATE_SELECTION_SCHEMA = (
    "neural-forge-lfm2-long-horizon-candidate-selection-v1"
)
ROW_FALLBACK_POLICY_SCHEMA = (
    "neural-forge-lfm2-long-horizon-row-disjoint-fallback-policy-v1"
)
ROW_FALLBACK_PLAN_SCHEMA = (
    "neural-forge-lfm2-long-horizon-row-disjoint-fallback-plan-v1"
)

WHOLE_OUTPUT_ENVELOPE_SCHEMA = "neural-forge-lfm2-whole-output-envelope-v1"
EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_CHILD_UNIQUE_PARAMETERS = 206_726_912
EXPECTED_COMPRESSION_FRACTION = (
    EXPECTED_PARENT_UNIQUE_PARAMETERS - EXPECTED_CHILD_UNIQUE_PARAMETERS
) / EXPECTED_PARENT_UNIQUE_PARAMETERS
REJECTED_LAYER_ORDER = (6, 8, 11, 13)
RIDGE_VALUES = (1e-8, 1e-6, 1e-4)
PROFILE_ORDER = ("balanced", "termination_heavy")
FIXED_CANDIDATE_INDICES = tuple(
    range(len(PROFILE_ORDER) * len(RIDGE_VALUES))
)
MAX_DECODE_STEPS = 64
EXPECTED_CASE_COUNT = 28
EXPECTED_RECOVERY_CASE_COUNT = 16
EXPECTED_MANUAL_CASE_COUNT = 12
EXPECTED_DOMAIN_CASE_COUNT = 8
TERMINATION_BOOST = {
    "selection": 24,
    "fit": 64,
    "validation": 0,
}
_SHA256_LENGTH = 64
_PATH_PATTERN = re.compile(
    r"^batch\.(?P<batch>\d{6})/"
    r"(?P<phase>prefill|decode\.(?P<step>\d{3}))/"
    r"row\.(?P<row>\d{4})$"
)
_VALIDATION_SHORTAGE_PATTERN = re.compile(
    r"^validation lacks (?P<shortage>[1-9]\d*) unique general rows and "
    r"repetition is forbidden by its evidence policy\.$"
)


class Lfm2LongHorizonRepairError(RuntimeError):
    """A repair precondition, transaction, or identity failed closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2LongHorizonRepairError(
            "Repair evidence must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2LongHorizonRepairError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _strict_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise Lfm2LongHorizonRepairError(
            f"{name} must be an integer in the allowed range."
        )
    return value


def _candidate_indices(
    value: Sequence[int],
    *,
    name: str = "candidate_indices",
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2LongHorizonRepairError(
            f"{name} must be a nonempty sorted unique integer sequence."
        )
    result = tuple(value)
    if (
        not result
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in FIXED_CANDIDATE_INDICES
            for index in result
        )
        or result != tuple(sorted(set(result)))
    ):
        raise Lfm2LongHorizonRepairError(
            f"{name} must be a nonempty sorted unique subset of 0..5."
        )
    return result


def parse_candidate_indices(value: str) -> tuple[int, ...]:
    """Parse the canonical CLI form ``0,2,5`` without reordering input."""

    if not isinstance(value, str) or re.fullmatch(
        r"[0-5](?:,[0-5])*",
        value,
    ) is None:
        raise argparse.ArgumentTypeError(
            "candidate indices must be a comma list using only 0..5"
        )
    parsed = tuple(int(part) for part in value.split(","))
    try:
        return _candidate_indices(parsed)
    except Lfm2LongHorizonRepairError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _candidate_request(indices: Sequence[int]) -> dict[str, Any]:
    requested = _candidate_indices(indices)
    body = {
        "schema": CANDIDATE_SELECTION_SCHEMA,
        "requested_candidate_indices": list(requested),
        "fixed_candidate_indices": list(FIXED_CANDIDATE_INDICES),
        "requested_candidate_count": len(requested),
        "covers_all_six_fixed_candidates": (
            requested == FIXED_CANDIDATE_INDICES
        ),
        "result_scope": (
            "all_six_fixed_candidates"
            if requested == FIXED_CANDIDATE_INDICES
            else "requested_candidate_subset_only"
        ),
    }
    return {**body, "request_sha256": _json_sha256(body)}


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2LongHorizonRepairError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2LongHorizonRepairError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise Lfm2LongHorizonRepairError(f"{name} must be finite.")
    return result


def _safe_local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2LongHorizonRepairError(f"{name} must be a local file.")
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2LongHorizonRepairError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


def _safe_local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2LongHorizonRepairError(f"{name} must be a local directory.")
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2LongHorizonRepairError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _case_domain(case_id: str) -> str | None:
    if case_id.startswith("recovery_math_"):
        return "math"
    if case_id.startswith("recovery_code_"):
        return "code"
    if case_id.startswith("manual_"):
        return "manual"
    return None


def _case_source_split(case_id: str) -> str | None:
    domain = _case_domain(case_id)
    if domain in {"math", "code"}:
        return "heldout"
    if domain == "manual":
        return "heldout_manual"
    return None


@dataclass(frozen=True, slots=True)
class DataContractBlocker:
    """One exact reason horizon activation development cannot start."""

    code: str
    case_id: str | None
    field: str | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "case_id": self.case_id,
            "field": self.field,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SealedEnvelopeAudit:
    """Verified original envelope plus a non-generative data readiness audit."""

    source_path: Path
    source_file_sha256: str
    envelope_sha256: str
    parent_cache_key: str
    parent_state_fingerprint: str
    tokenizer_fingerprint: str
    eos_token_ids: tuple[int, ...]
    generation_limit: int
    rows: tuple[Mapping[str, Any], ...]
    recovery_rows: tuple[Mapping[str, Any], ...]
    manual_rows: tuple[Mapping[str, Any], ...]
    blockers: tuple[DataContractBlocker, ...]
    split_identity_sha256: str

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def recovery_case_ids(self) -> tuple[str, ...]:
        return tuple(str(row["id"]) for row in self.recovery_rows)

    @property
    def manual_case_ids(self) -> tuple[str, ...]:
        return tuple(str(row["id"]) for row in self.manual_rows)

    def as_dict(self) -> dict[str, Any]:
        missing_counts = Counter(
            blocker.field
            for blocker in self.blockers
            if blocker.field is not None
        )
        return {
            "schema": DATA_CONTRACT_SCHEMA,
            "ready": self.ready,
            "source_path": str(self.source_path),
            "source_file_sha256": self.source_file_sha256,
            "envelope_sha256": self.envelope_sha256,
            "parent_cache_key": self.parent_cache_key,
            "parent_state_fingerprint": self.parent_state_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "eos_token_ids": list(self.eos_token_ids),
            "generation_limit": self.generation_limit,
            "case_count": len(self.rows),
            "recovery_case_count": len(self.recovery_rows),
            "manual_case_count": len(self.manual_rows),
            "recovery_case_ids": list(self.recovery_case_ids),
            "manual_case_ids": list(self.manual_case_ids),
            "activation_development_case_ids": list(self.recovery_case_ids),
            "promotion_only_case_ids": list(self.manual_case_ids),
            "case_sets_disjoint": not bool(
                set(self.recovery_case_ids).intersection(self.manual_case_ids)
            ),
            "split_identity_sha256": self.split_identity_sha256,
            "blockers": [blocker.as_dict() for blocker in self.blockers],
            "missing_field_counts": {
                str(field): count
                for field, count in sorted(missing_counts.items())
            },
            "existing_parent_output_ids_reused_as_forced_decode_ids": True,
            "parent_response_generation_required": False,
            "parent_response_generation_allowed": False,
            "required_enrichment": {
                "input_ids": (
                    "Exact sealed prompt IDs, including the same special tokens "
                    "used for the original parent run."
                ),
                "parent_output_token_texts": (
                    "Derived locally, one piece per existing parent_output_id, "
                    "using the tokenizer fingerprint bound by the envelope."
                ),
                "arithmetic_answer_correct": (
                    "Derived from sealed expected_response and parent_output_text."
                ),
                "arithmetic_answer_token_indices": (
                    "Derived by exact character-span overlap with locally decoded "
                    "token pieces."
                ),
                "termination_evidence": (
                    "Derived from tokenizer EOS IDs or accepted from exact sealed "
                    "termination indices."
                ),
            },
        }


def _canonical_envelope(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if _file_sha256(path) != _digest(expected_sha256, "expected_sha256"):
        raise Lfm2LongHorizonRepairError(
            "Sealed whole-output envelope file SHA-256 differs from the CLI binding."
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2LongHorizonRepairError(
            "Sealed whole-output envelope is not canonical ASCII JSON."
        ) from exc
    if not isinstance(value, Mapping) or _canonical_json(value) + b"\n" != raw:
        raise Lfm2LongHorizonRepairError(
            "Sealed whole-output envelope encoding is not canonical."
        )
    body = dict(value)
    seal = body.pop("envelope_sha256", None)
    if _digest(seal, "envelope_sha256") != _json_sha256(body):
        raise Lfm2LongHorizonRepairError(
            "Whole-output envelope seal does not match its contents."
        )
    if body.get("schema") != WHOLE_OUTPUT_ENVELOPE_SCHEMA:
        raise Lfm2LongHorizonRepairError(
            "Whole-output envelope schema is unsupported."
        )
    return value


def _token_ids(
    value: Any,
    *,
    maximum: int,
) -> tuple[int, ...] | None:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > maximum
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in value
        )
    ):
        return None
    return tuple(value)


def _sorted_token_indices(
    value: Any,
    *,
    token_count: int,
) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= token_count
        for index in value
    ):
        return None
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        return None
    return result


def _attention_mask(
    value: Any,
    *,
    token_count: int,
) -> tuple[int, ...] | None:
    if (
        not isinstance(value, list)
        or len(value) != token_count
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item not in {0, 1}
            for item in value
        )
        or not any(value)
    ):
        return None
    return tuple(value)


def _decode_token_pieces(
    tokenizer: Any,
    token_ids: Sequence[int],
) -> tuple[str, ...]:
    """Decode exact prefix deltas so concatenation preserves tokenizer output."""

    previous = ""
    pieces = []
    for end in range(1, len(token_ids) + 1):
        current = tokenizer.decode(
            list(token_ids[:end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(current, str) or not current.startswith(previous):
            raise Lfm2LongHorizonRepairError(
                "Local tokenizer decoding is not prefix-stable for the sealed "
                "parent output IDs."
            )
        pieces.append(current[len(previous) :])
        previous = current
    return tuple(pieces)


def _tokenizer_eos_ids(tokenizer: Any) -> tuple[int, ...]:
    values = getattr(tokenizer, "eos_token_id", None)
    if values is None:
        values = []
    elif isinstance(values, int) and not isinstance(values, bool):
        values = [values]
    if (
        not isinstance(values, (list, tuple))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        )
    ):
        raise Lfm2LongHorizonRepairError(
            "Local tokenizer exposes an invalid eos_token_id."
        )
    return tuple(sorted(set(values)))


def _exact_span_token_indices(
    token_texts: Sequence[str],
    spans: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    result = []
    cursor = 0
    for index, text in enumerate(token_texts):
        end = cursor + len(text)
        if cursor != end and any(
            cursor < span_end and span_start < end
            for span_start, span_end in spans
        ):
            result.append(index)
        cursor = end
    return tuple(result)


def _derive_arithmetic_evidence(
    row: Mapping[str, Any],
    token_texts: Sequence[str],
) -> tuple[bool, tuple[int, ...]]:
    expected = row.get("expected_response")
    if not isinstance(expected, str) or not expected.strip():
        return False, ()
    text = "".join(token_texts)
    escaped = re.escape(expected.strip())
    matches = list(
        re.finditer(
            rf"(?<![0-9A-Za-z_]){escaped}(?![0-9A-Za-z_])",
            text,
        )
    )
    spans = tuple(match.span() for match in matches)
    return bool(spans), _exact_span_token_indices(token_texts, spans)


def audit_sealed_horizon_envelope(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    maximum_prompt_tokens: int = 512,
    token_text_decoder: (
        Callable[[Sequence[int]], Sequence[str]] | None
    ) = None,
    tokenizer_eos_token_ids: Sequence[int] = (),
    require_token_texts: bool = True,
) -> SealedEnvelopeAudit:
    """Verify the 28-case seal and report missing tokenized development data."""

    source = _safe_local_file(path, "cached_parent_outputs")
    envelope = _canonical_envelope(source, expected_sha256)
    for field in (
        "parent_cache_key",
        "parent_state_fingerprint",
        "tokenizer_fingerprint",
    ):
        _digest(envelope.get(field), field)
    generation = envelope.get("generation")
    if not isinstance(generation, Mapping):
        raise Lfm2LongHorizonRepairError(
            "Whole-output envelope has no generation contract."
        )
    generation_limit = _strict_int(
        generation.get("max_new_tokens"),
        "generation.max_new_tokens",
        minimum=1,
        maximum=MAX_DECODE_STEPS,
    )
    if generation_limit != MAX_DECODE_STEPS:
        raise Lfm2LongHorizonRepairError(
            "Long-horizon repair requires the sealed 64-token generation limit."
        )
    if envelope.get("parent_outputs_generated_once") is not True:
        raise Lfm2LongHorizonRepairError(
            "Envelope does not attest one-time parent output generation."
        )
    if envelope.get("parent_forward_passes_after_seal") != 0:
        raise Lfm2LongHorizonRepairError(
            "Envelope does not attest zero parent forwards after sealing."
        )
    rows_value = envelope.get("rows")
    if not isinstance(rows_value, list) or len(rows_value) != EXPECTED_CASE_COUNT:
        raise Lfm2LongHorizonRepairError(
            "Whole-output envelope must contain exactly 28 cases."
        )
    rows = []
    recovery = []
    manual = []
    seen = set()
    blockers: list[DataContractBlocker] = []
    for index, row in enumerate(rows_value):
        if not isinstance(row, Mapping):
            raise Lfm2LongHorizonRepairError(
                f"Whole-output row {index} is not a mapping."
            )
        case_id = row.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
        ):
            raise Lfm2LongHorizonRepairError(
                f"Whole-output row {index} has an invalid or duplicate id."
            )
        seen.add(case_id)
        domain = _case_domain(case_id)
        source_split = _case_source_split(case_id)
        if domain is None or source_split is None:
            blockers.append(
                DataContractBlocker(
                    code="unsupported_case_identity",
                    case_id=case_id,
                    field="id",
                    detail=(
                        "Case IDs must use recovery_math_, recovery_code_, or "
                        "manual_ prefixes."
                    ),
                )
            )
        declared_split = row.get("source_split", row.get("split"))
        if declared_split is not None and declared_split != source_split:
            blockers.append(
                DataContractBlocker(
                    code="source_split_conflict",
                    case_id=case_id,
                    field="source_split",
                    detail=(
                        f"Declared {declared_split!r}; deterministic identity "
                        f"classification is {source_split!r}."
                    ),
                )
            )
        copied = json.loads(_canonical_json(dict(row)))
        rows.append(copied)
        if source_split == "heldout":
            recovery.append(copied)
        elif source_split == "heldout_manual":
            manual.append(copied)
    domain_counts = Counter(_case_domain(str(row["id"])) for row in recovery)
    if (
        len(recovery) != EXPECTED_RECOVERY_CASE_COUNT
        or domain_counts != {"math": 8, "code": 8}
    ):
        blockers.append(
            DataContractBlocker(
                code="recovery_case_partition_invalid",
                case_id=None,
                field="id",
                detail=(
                    "Activation development requires exactly eight recovery_math "
                    "and eight recovery_code cases."
                ),
            )
        )
    if len(manual) != EXPECTED_MANUAL_CASE_COUNT:
        blockers.append(
            DataContractBlocker(
                code="manual_case_partition_invalid",
                case_id=None,
                field="id",
                detail="Promotion requires exactly twelve manual_ cases.",
            )
        )
    eos_value = envelope.get("eos_token_ids", [])
    if (
        not isinstance(eos_value, list)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in eos_value
        )
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in tokenizer_eos_token_ids
        )
    ):
        raise Lfm2LongHorizonRepairError(
            "Envelope or local tokenizer EOS token IDs are invalid."
        )
    eos_ids = set(eos_value).union(tokenizer_eos_token_ids)
    eos_markers = {
        "<eos>",
        "</s>",
        "<|eos|>",
        "<|endoftext|>",
        "<|im_end|>",
        "[eos]",
    }
    for row in recovery:
        case_id = str(row["id"])
        domain = str(_case_domain(case_id))
        prompt_value = row.get("input_ids", row.get("prompt_input_ids"))
        prompt_ids = _token_ids(
            prompt_value,
            maximum=maximum_prompt_tokens,
        )
        if prompt_ids is None or len(prompt_ids) < 3:
            blockers.append(
                DataContractBlocker(
                    code="missing_exact_prompt_token_ids",
                    case_id=case_id,
                    field="input_ids",
                    detail=(
                        "Need a flat sequence of 3.."
                        f"{maximum_prompt_tokens} exact prompt token IDs; raw prompt "
                        "text will not be silently retokenized."
                    ),
                )
            )
        else:
            row["input_ids"] = list(prompt_ids)
            row["prompt_input_ids"] = list(prompt_ids)
            attention = _attention_mask(
                row.get("attention_mask"),
                token_count=len(prompt_ids),
            )
            if attention is None:
                blockers.append(
                    DataContractBlocker(
                        code="missing_exact_attention_mask",
                        case_id=case_id,
                        field="attention_mask",
                        detail=(
                            "Need a sealed binary attention_mask aligned exactly "
                            "with input_ids."
                        ),
                    )
                )
            else:
                row["attention_mask"] = list(attention)
        output_ids = _token_ids(
            row.get("parent_output_ids"),
            maximum=MAX_DECODE_STEPS,
        )
        if output_ids is None:
            blockers.append(
                DataContractBlocker(
                    code="missing_existing_parent_output_ids",
                    case_id=case_id,
                    field="parent_output_ids",
                    detail=(
                        "Need 1..64 existing parent output token IDs for forced "
                        "decode; parent text will not be regenerated."
                    ),
                )
            )
            continue
        forced_decode_ids = _token_ids(
            row.get("forced_decode_ids"),
            maximum=MAX_DECODE_STEPS,
        )
        if forced_decode_ids != output_ids:
            blockers.append(
                DataContractBlocker(
                    code="forced_decode_ids_mismatch",
                    case_id=case_id,
                    field="forced_decode_ids",
                    detail=(
                        "forced_decode_ids must already exist and exactly equal "
                        "the sealed parent_output_ids."
                    ),
                )
            )
        token_texts = row.get("parent_output_token_texts")
        if (
            not isinstance(token_texts, list)
            or len(token_texts) != len(output_ids)
            or any(not isinstance(value, str) for value in token_texts)
        ):
            if token_text_decoder is not None:
                token_texts = list(token_text_decoder(output_ids))
                if (
                    len(token_texts) != len(output_ids)
                    or any(not isinstance(value, str) for value in token_texts)
                ):
                    raise Lfm2LongHorizonRepairError(
                        "Local token decoder did not return one text piece per "
                        "sealed parent output ID."
                    )
                row["parent_output_token_texts"] = token_texts
            elif require_token_texts:
                blockers.append(
                    DataContractBlocker(
                        code="missing_aligned_parent_output_token_texts",
                        case_id=case_id,
                        field="parent_output_token_texts",
                        detail=(
                            "Need the bound local tokenizer to decode exactly one "
                            "piece per existing parent_output_id."
                        ),
                    )
                )
                token_texts = []
            else:
                token_texts = []
        if domain == "math":
            correct = row.get("arithmetic_answer_correct")
            if (
                not isinstance(correct, bool)
                and token_text_decoder is not None
                and token_texts
            ):
                correct, derived_indices = _derive_arithmetic_evidence(
                    row,
                    token_texts,
                )
                row["arithmetic_answer_correct"] = correct
                row["arithmetic_answer_token_indices"] = list(derived_indices)
            if not isinstance(correct, bool):
                if require_token_texts:
                    blockers.append(
                        DataContractBlocker(
                            code="missing_arithmetic_correctness_attestation",
                            case_id=case_id,
                            field="arithmetic_answer_correct",
                            detail=(
                                "Need sealed expected_response plus the bound local "
                                "tokenizer to derive arithmetic evidence."
                            ),
                        )
                    )
            elif correct:
                indices = _sorted_token_indices(
                    row.get("arithmetic_answer_token_indices"),
                    token_count=len(output_ids),
                )
                if not indices:
                    blockers.append(
                        DataContractBlocker(
                            code="missing_exact_arithmetic_answer_span",
                            case_id=case_id,
                            field="arithmetic_answer_token_indices",
                            detail=(
                                "Parent-correct arithmetic cases need sorted exact "
                                "answer-token indices."
                            ),
                        )
                    )
        termination_indices = _sorted_token_indices(
            row.get("termination_token_indices"),
            token_count=len(output_ids),
        )
        if termination_indices is None:
            derived_termination = tuple(
                index
                for index, token_id in enumerate(output_ids)
                if token_id in eos_ids
            )
            if derived_termination:
                termination_indices = derived_termination
                row["termination_token_indices"] = list(derived_termination)
                row["termination_confirmed"] = True
        termination_by_id = any(token in eos_ids for token in output_ids)
        termination_by_text = any(
            text.strip().lower() in eos_markers for text in token_texts
        )
        if require_token_texts and (
            not termination_indices
            and row.get("termination_confirmed") is not True
            and not termination_by_id
            and not termination_by_text
        ):
            blockers.append(
                DataContractBlocker(
                    code="missing_termination_anchor",
                    case_id=case_id,
                    field="termination_token_indices",
                    detail=(
                        "Termination-heavy repair needs an exact termination "
                        "index, EOS ID, decoded EOS marker, or confirmed final token."
                    ),
                )
            )
    split_body = {
        "classification": {
            "recovery_math_*": "heldout/activation-development",
            "recovery_code_*": "heldout/activation-development",
            "manual_*": "heldout_manual/promotion-only",
        },
        "recovery_case_ids": [str(row["id"]) for row in recovery],
        "manual_case_ids": [str(row["id"]) for row in manual],
    }
    return SealedEnvelopeAudit(
        source_path=source,
        source_file_sha256=_file_sha256(source),
        envelope_sha256=_digest(
            envelope.get("envelope_sha256"),
            "envelope_sha256",
        ),
        parent_cache_key=_digest(
            envelope.get("parent_cache_key"),
            "parent_cache_key",
        ),
        parent_state_fingerprint=_digest(
            envelope.get("parent_state_fingerprint"),
            "parent_state_fingerprint",
        ),
        tokenizer_fingerprint=_digest(
            envelope.get("tokenizer_fingerprint"),
            "tokenizer_fingerprint",
        ),
        eos_token_ids=tuple(sorted(eos_ids)),
        generation_limit=generation_limit,
        rows=tuple(rows),
        recovery_rows=tuple(recovery),
        manual_rows=tuple(manual),
        blockers=tuple(blockers),
        split_identity_sha256=_json_sha256(split_body),
    )


@dataclass(frozen=True, slots=True)
class SourceChildBinding:
    """Authenticated identity of the existing 9.9987% child."""

    report_path: Path
    report_file_sha256: str
    report_payload_sha256: str
    source_child_state_sha256: str
    parent_cache_key: str
    parent_unique_parameters: int
    child_unique_parameters: int
    compression_fraction: float
    rejected_layers: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_BINDING_SCHEMA,
            "report_path": str(self.report_path),
            "report_file_sha256": self.report_file_sha256,
            "report_payload_sha256": self.report_payload_sha256,
            "source_child_state_sha256": self.source_child_state_sha256,
            "parent_cache_key": self.parent_cache_key,
            "parent_unique_parameters": self.parent_unique_parameters,
            "child_unique_parameters": self.child_unique_parameters,
            "compression_fraction": self.compression_fraction,
            "compression_percent": self.compression_fraction * 100.0,
            "rejected_layers": list(self.rejected_layers),
            "exact_target_authenticated": True,
        }


def _nested_value(value: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    return None


def load_source_child_binding(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> SourceChildBinding:
    """Authenticate the exact source report and its rejected-layer frontier."""

    source = _safe_local_file(path, "source_child_report")
    file_sha = _file_sha256(source)
    expected = _digest(expected_sha256, "source_child_report_sha256")
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2LongHorizonRepairError(
            "Source-child report is not valid JSON."
        ) from exc
    if not isinstance(report, Mapping):
        raise Lfm2LongHorizonRepairError(
            "Source-child report must contain an object."
        )
    claimed_report_sha = report.get("report_sha256")
    if claimed_report_sha is not None:
        claimed_report_sha = _digest(
            claimed_report_sha,
            "source report report_sha256",
        )
        report_body = dict(report)
        report_body.pop("report_sha256")
        if _json_sha256(report_body) != claimed_report_sha:
            raise Lfm2LongHorizonRepairError(
                "Source-child report self-seal does not match its contents."
            )
    if expected not in {file_sha, claimed_report_sha}:
        raise Lfm2LongHorizonRepairError(
            "Source-child report SHA-256 differs from both its file and "
            "authenticated payload identities."
        )
    rejected_value = report.get("rejected_layers")
    if (
        not isinstance(rejected_value, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in rejected_value
        )
    ):
        raise Lfm2LongHorizonRepairError(
            "Source-child report does not declare rejected_layers."
        )
    rejected = tuple(rejected_value)
    if rejected != REJECTED_LAYER_ORDER:
        raise Lfm2LongHorizonRepairError(
            "Source child is not the expected layer-6/8/11/13 rejection frontier."
        )
    parent_count = _nested_value(
        report,
        (
            ("accounting", "observed_parent_unique_parameters"),
            ("accounting", "source_unique_parameters"),
            ("accounting", "parent_unique_parameters"),
        ),
    )
    child_count = _nested_value(
        report,
        (
            ("accounting", "observed_target_unique_parameters"),
            ("accounting", "target_unique_parameters"),
            ("final_checkpoint", "unique_parameters"),
        ),
    )
    fraction = _nested_value(
        report,
        (
            ("accounting", "fraction_removed"),
            ("candidate", "compression_fraction"),
        ),
    )
    state_sha = _nested_value(
        report,
        (
            ("final_checkpoint", "state_sha256"),
            ("current_checkpoint", "state_sha256"),
        ),
    )
    parent_cache_key = _nested_value(
        report,
        (
            ("cache", "cache_key"),
            ("parent", "cache", "cache_key"),
            ("parent_cache_key",),
        ),
    )
    parent_count = _strict_int(
        parent_count,
        "source report parent unique parameters",
        minimum=1,
    )
    child_count = _strict_int(
        child_count,
        "source report child unique parameters",
        minimum=1,
    )
    fraction = _finite_float(fraction, "source report compression fraction")
    state_sha = _digest(state_sha, "source child state_sha256")
    parent_cache_key = _digest(parent_cache_key, "source report parent_cache_key")
    if (
        parent_count != EXPECTED_PARENT_UNIQUE_PARAMETERS
        or child_count != EXPECTED_CHILD_UNIQUE_PARAMETERS
        or not math.isclose(
            fraction,
            EXPECTED_COMPRESSION_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise Lfm2LongHorizonRepairError(
            "Source report is not the exact 9.9987% / 206,726,912-parameter child."
        )
    return SourceChildBinding(
        report_path=source,
        report_file_sha256=file_sha,
        report_payload_sha256=(
            claimed_report_sha
            if claimed_report_sha is not None
            else _json_sha256(report)
        ),
        source_child_state_sha256=state_sha,
        parent_cache_key=parent_cache_key,
        parent_unique_parameters=parent_count,
        child_unique_parameters=child_count,
        compression_fraction=fraction,
        rejected_layers=rejected,
    )


def build_horizon_batch_records(
    audit: SealedEnvelopeAudit,
) -> tuple[dict[str, list[int]], ...]:
    """Return exact non-tensor cache inputs using existing output IDs verbatim."""

    if not isinstance(audit, SealedEnvelopeAudit) or not audit.ready:
        raise Lfm2LongHorizonRepairError(
            "A ready sealed envelope audit is required to build horizon batches."
        )
    result = []
    for row in audit.recovery_rows:
        prompt = _token_ids(row.get("input_ids"), maximum=512)
        output = _token_ids(
            row.get("parent_output_ids"),
            maximum=MAX_DECODE_STEPS,
        )
        attention = (
            None
            if prompt is None
            else _attention_mask(
                row.get("attention_mask"),
                token_count=len(prompt),
            )
        )
        forced = _token_ids(
            row.get("forced_decode_ids"),
            maximum=MAX_DECODE_STEPS,
        )
        if (
            prompt is None
            or output is None
            or attention is None
            or forced != output
        ):  # pragma: no cover - audited above
            raise Lfm2LongHorizonRepairError(
                "Audited horizon token IDs changed before batch construction."
            )
        result.append(
            {
                "input_ids": list(prompt),
                "attention_mask": list(attention),
                "forced_decode_ids": list(forced),
            }
        )
    return tuple(result)


def build_anchor_case_envelope(
    audit: SealedEnvelopeAudit,
) -> dict[str, Any]:
    """Adapt the verified 28-case envelope to the pure anchor-policy schema."""

    if not audit.ready:
        raise Lfm2LongHorizonRepairError(
            "Cannot build anchors from a blocked envelope."
        )
    cases = []
    for row in audit.rows:
        case_id = str(row["id"])
        domain = _case_domain(case_id)
        split = _case_source_split(case_id)
        if domain is None or split is None:  # pragma: no cover - audited above
            raise Lfm2LongHorizonRepairError(
                f"Unsupported case identity {case_id!r}."
            )
        is_recovery = split == "heldout"
        value: dict[str, Any] = {
            "case_id": case_id,
            "source_split": (
                "heldout_recovery" if is_recovery else "heldout_manual"
            ),
            "domain": domain,
            # Manual trajectories are deliberately redacted from activation
            # evidence. Their IDs remain bound by the original envelope but
            # never enter the anchor policy.
            "parent_output_ids": (
                list(row.get("parent_output_ids", [])) if is_recovery else []
            ),
            "parent_output_token_texts": (
                list(row.get("parent_output_token_texts", []))
                if is_recovery
                else []
            ),
        }
        for field in (
            "arithmetic_answer_correct",
            "arithmetic_answer_token_indices",
            "termination_token_indices",
            "termination_confirmed",
            "instruction_start_token_indices",
        ):
            if field in row:
                value[field] = row[field]
        cases.append(value)
    return seal_lfm2_long_horizon_case_envelope(
        cases,
        source_identity_sha256=audit.envelope_sha256,
        tokenizer_fingerprint=audit.tokenizer_fingerprint,
        eos_token_ids=audit.eos_token_ids,
    )


def synthetic_trace_path_descriptors(
    audit: SealedEnvelopeAudit,
) -> tuple[dict[str, Any], ...]:
    """Preflight anchor feasibility without loading or executing a model."""

    if not audit.ready:
        raise Lfm2LongHorizonRepairError(
            "Cannot synthesize path descriptors for blocked evidence."
        )
    result = []
    for batch_index, row in enumerate(audit.recovery_rows):
        case_id = str(row["id"])
        output_ids = tuple(int(value) for value in row["parent_output_ids"])
        result.append(
            {
                "path": f"batch.{batch_index:06d}/prefill/row.0000",
                "case_id": case_id,
                "phase": "prefill",
                "row_index": 0,
                "predicted_output_index": 0,
                "predicted_token_id": output_ids[0],
            }
        )
        for output_index in range(1, len(output_ids)):
            step = output_index - 1
            result.append(
                {
                    "path": (
                        f"batch.{batch_index:06d}/decode.{step:03d}/row.0000"
                    ),
                    "case_id": case_id,
                    "phase": f"decode.{step:03d}",
                    "row_index": 0,
                    "predicted_output_index": output_index,
                    "predicted_token_id": output_ids[output_index],
                }
            )
    return tuple(result)


def trace_path_descriptors_from_paths(
    paths: Sequence[str],
    audit: SealedEnvelopeAudit,
) -> tuple[dict[str, Any], ...]:
    """Bind actual ordered cache paths to exact sealed output-token targets."""

    if not audit.ready:
        raise Lfm2LongHorizonRepairError(
            "Cannot bind trace paths from a blocked envelope."
        )
    grouped: dict[int, list[tuple[int, str, str, int, int | None]]] = {}
    for path_index, path in enumerate(paths):
        if not isinstance(path, str):
            raise Lfm2LongHorizonRepairError("Trace paths must be strings.")
        match = _PATH_PATTERN.fullmatch(path)
        if match is None:
            raise Lfm2LongHorizonRepairError(
                f"Unsupported horizon trace path {path!r}."
            )
        batch = int(match.group("batch"))
        phase = str(match.group("phase"))
        row = int(match.group("row"))
        step = None if match.group("step") is None else int(match.group("step"))
        grouped.setdefault(batch, []).append((path_index, path, phase, row, step))
    if set(grouped) != set(range(len(audit.recovery_rows))):
        raise Lfm2LongHorizonRepairError(
            "Horizon trace batches do not exactly match the 16 recovery cases."
        )
    result: list[dict[str, Any] | None] = [None] * len(paths)
    for batch_index, row in enumerate(audit.recovery_rows):
        output_ids = tuple(int(value) for value in row["parent_output_ids"])
        case_id = str(row["id"])
        batch_paths = grouped[batch_index]
        prefill_rows = [
            row_index
            for _index, _path, phase, row_index, _step in batch_paths
            if phase == "prefill"
        ]
        if not prefill_rows:
            raise Lfm2LongHorizonRepairError(
                f"Trace batch {batch_index} has no prefill rows."
            )
        final_prefill = max(prefill_rows)
        for path_index, path, phase, row_index, step in batch_paths:
            predicted_index: int | None = None
            predicted_token: int | None = None
            if phase == "prefill" and row_index == final_prefill:
                predicted_index = 0
                predicted_token = output_ids[0]
            elif step is not None and step + 1 < len(output_ids):
                predicted_index = step + 1
                predicted_token = output_ids[predicted_index]
            result[path_index] = {
                "path": path,
                "case_id": case_id,
                "phase": phase,
                "row_index": row_index,
                "predicted_output_index": predicted_index,
                "predicted_token_id": predicted_token,
            }
    if any(value is None for value in result):  # pragma: no cover - exhaustive
        raise Lfm2LongHorizonRepairError(
            "Trace descriptor construction left unmapped paths."
        )
    return tuple(value for value in result if value is not None)


ROW_FALLBACK_POLICY = {
    "schema": ROW_FALLBACK_POLICY_SCHEMA,
    "trigger": {
        "strict_policy_authenticated_first": True,
        "condition": "strict-case-disjoint-validation-has-fewer-than-128-rows",
        "other_anchor_errors_fail_closed": True,
    },
    "eligible_source_split": "heldout_recovery",
    "excluded_source_split": "heldout_manual",
    "allocation": {
        "case_disjoint": False,
        "row_disjoint": True,
        "all_sixteen_recovery_cases_seed_each_split": True,
        "strata": ["domain", "primary_semantic_tag", "case_id"],
        "selection_weight": 256,
        "fit_weight": 512,
        "validation_weight": 128,
        "validation_rows_unique": True,
        "selection_fit_integer_cycling_after_unique_partition": True,
    },
    "promotion_set": "twelve-heldout-manual-cases-only",
    "zero_training": {
        "gradients_used": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
    },
}
ROW_FALLBACK_POLICY_IDENTITY = _json_sha256(ROW_FALLBACK_POLICY)
_PRIMARY_TAG_ORDER = (
    "termination",
    "arithmetic_answer",
    "code_delimiter",
    "code_operator",
    "instruction",
    "rare_symbol",
    "general_coverage",
)


@dataclass(frozen=True, slots=True)
class RowDisjointFallbackSplit:
    """One unique row partition plus its post-partition integer expansion."""

    split: str
    case_ids: tuple[str, ...]
    indices: tuple[int, ...]
    expanded_indices: tuple[int, ...]
    termination_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        counts = Counter(self.expanded_indices)
        return {
            "split": self.split,
            "case_ids": list(self.case_ids),
            "indices": list(self.indices),
            "integer_weights": [counts[index] for index in self.indices],
            "expanded_indices": list(self.expanded_indices),
            "termination_indices": list(self.termination_indices),
            "total_weight": len(self.expanded_indices),
            "unique_count": len(self.indices),
            "repeated_occurrence_count": (
                len(self.expanded_indices) - len(set(self.expanded_indices))
            ),
            "validation_unique": (
                self.split != "validation"
                or len(self.expanded_indices)
                == len(set(self.expanded_indices))
            ),
        }


@dataclass(frozen=True, slots=True)
class RowDisjointFallbackPlan:
    """Authenticated fallback used only when strict validation is too short."""

    envelope_sha256: str
    ordered_path_sha256: str
    composite_evidence_identity: str
    tagging_sha256: str
    excluded_case_ids: tuple[str, ...]
    strict_error: str
    strict_validation_unique_count: int
    selection: RowDisjointFallbackSplit
    fit: RowDisjointFallbackSplit
    validation: RowDisjointFallbackSplit
    plan_sha256: str

    @property
    def policy_identity(self) -> str:
        return ROW_FALLBACK_POLICY_IDENTITY

    @property
    def termination_indices_by_split(
        self,
    ) -> dict[str, tuple[int, ...]]:
        return {
            "selection": self.selection.termination_indices,
            "fit": self.fit.termination_indices,
            "validation": self.validation.termination_indices,
        }

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": ROW_FALLBACK_PLAN_SCHEMA,
            "policy": dict(ROW_FALLBACK_POLICY),
            "policy_identity": ROW_FALLBACK_POLICY_IDENTITY,
            "envelope_sha256": self.envelope_sha256,
            "ordered_path_sha256": self.ordered_path_sha256,
            "composite_evidence_identity": self.composite_evidence_identity,
            "tagging_sha256": self.tagging_sha256,
            "excluded_case_ids": list(self.excluded_case_ids),
            "fallback_trigger": {
                "strict_error": self.strict_error,
                "strict_validation_unique_count": (
                    self.strict_validation_unique_count
                ),
                "required_validation_unique_count": 128,
                "authenticated_insufficient_unique_validation": True,
            },
            "case_disjoint": False,
            "row_disjoint": True,
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


RepairAnchorPlan = Lfm2LongHorizonAnchorPlan | RowDisjointFallbackPlan


def _primary_anchor_tag(tags: Sequence[str]) -> str:
    tag_set = set(tags)
    return next(
        (tag for tag in _PRIMARY_TAG_ORDER if tag in tag_set),
        "general_coverage",
    )


def _stratified_row_order(
    rows: Sequence[Any],
    *,
    salt: str,
) -> tuple[Any, ...]:
    """Round-robin deterministic domain/tag/case buckets."""

    buckets: dict[tuple[str, str, str], list[Any]] = {}
    for row in rows:
        domain = str(_case_domain(row.case_id))
        key = (domain, _primary_anchor_tag(row.tags), row.case_id)
        buckets.setdefault(key, []).append(row)
    for key, values in buckets.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{salt}\0{key}\0{row.path}".encode()
            ).hexdigest()
        )
    keys = sorted(
        buckets,
        key=lambda key: hashlib.sha256(
            f"{salt}\0{key}".encode()
        ).hexdigest(),
    )
    result = []
    cursor = 0
    while keys:
        key = keys[cursor % len(keys)]
        result.append(buckets[key].pop(0))
        if not buckets[key]:
            keys.remove(key)
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor += 1
    return tuple(result)


def _cycle_to_weight(
    rows: Sequence[Any],
    *,
    weight: int,
) -> tuple[int, ...]:
    if not rows:
        raise Lfm2LongHorizonRepairError(
            "Row-disjoint fallback produced an empty weighted split."
        )
    return tuple(rows[index % len(rows)].path_index for index in range(weight))


def _row_disjoint_fallback_plan(
    tagging: Lfm2LongHorizonTagging,
    *,
    strict_error: str,
    strict_validation_unique_count: int,
) -> RowDisjointFallbackPlan:
    rows = tuple(tagging.rows)
    if (
        len({row.path_index for row in rows}) != len(rows)
        or any(_case_source_split(row.case_id) != "heldout" for row in rows)
        or len({row.case_id for row in rows}) != EXPECTED_RECOVERY_CASE_COUNT
    ):
        raise Lfm2LongHorizonRepairError(
            "Authenticated fallback rows contain duplicates, manual leakage, "
            "or an incomplete recovery-case set."
        )
    by_case: dict[str, list[Any]] = {}
    for row in rows:
        by_case.setdefault(row.case_id, []).append(row)
    splits: dict[str, list[Any]] = {
        "selection": [],
        "fit": [],
        "validation": [],
    }
    assigned: set[int] = set()
    case_order = sorted(
        by_case,
        key=lambda case_id: hashlib.sha256(
            f"{ROW_FALLBACK_POLICY_IDENTITY}\0{case_id}".encode()
        ).hexdigest(),
    )
    split_order = ("selection", "fit", "validation")
    for case_index, case_id in enumerate(case_order):
        available = _stratified_row_order(
            by_case[case_id],
            salt=f"{ROW_FALLBACK_POLICY_IDENTITY}\0seed\0{case_id}",
        )
        termination = [
            row for row in available if "termination" in row.tags
        ]
        termination_split = split_order[case_index % len(split_order)]
        for split in (
            termination_split,
            *(value for value in split_order if value != termination_split),
        ):
            candidates = [
                row
                for row in available
                if row.path_index not in assigned
                and (
                    split != termination_split
                    or not termination
                    or row.path_index == termination[0].path_index
                )
            ]
            if not candidates:
                raise Lfm2LongHorizonRepairError(
                    f"Recovery case {case_id!r} cannot seed all fallback splits."
                )
            chosen = candidates[0]
            splits[split].append(chosen)
            assigned.add(chosen.path_index)

    remaining = [row for row in rows if row.path_index not in assigned]
    validation_needed = 128 - len(splits["validation"])
    validation_order = _stratified_row_order(
        remaining,
        salt=f"{ROW_FALLBACK_POLICY_IDENTITY}\0validation",
    )
    if validation_needed < 0 or len(validation_order) < validation_needed:
        raise Lfm2LongHorizonRepairError(
            "Row-disjoint fallback cannot allocate 128 unique validation paths."
        )
    selected_validation = validation_order[:validation_needed]
    splits["validation"].extend(selected_validation)
    assigned.update(row.path_index for row in selected_validation)

    remaining = [row for row in rows if row.path_index not in assigned]
    available_after_validation = (
        len(splits["selection"]) + len(splits["fit"]) + len(remaining)
    )
    selection_target = min(
        256,
        max(
            len(splits["selection"]),
            available_after_validation // 3,
        ),
    )
    fit_target = min(
        512,
        available_after_validation - selection_target,
    )
    if (
        fit_target < len(splits["fit"])
        or selection_target + fit_target > available_after_validation
    ):
        raise Lfm2LongHorizonRepairError(
            "Row-disjoint fallback cannot retain nonempty selection and fit pools."
        )
    allocation_order = _stratified_row_order(
        remaining,
        salt=f"{ROW_FALLBACK_POLICY_IDENTITY}\0selection-fit",
    )
    selection_needed = selection_target - len(splits["selection"])
    fit_needed = fit_target - len(splits["fit"])
    for row in allocation_order:
        if selection_needed <= 0 and fit_needed <= 0:
            break
        if selection_needed > 0 and (
            fit_needed <= 0
            or selection_needed / selection_target
            >= fit_needed / fit_target
        ):
            splits["selection"].append(row)
            selection_needed -= 1
        else:
            splits["fit"].append(row)
            fit_needed -= 1
        assigned.add(row.path_index)
    if selection_needed or fit_needed:
        raise Lfm2LongHorizonRepairError(
            "Row-disjoint fallback allocation ended before its exact pool targets."
        )

    unique_sets = {
        split: {row.path_index for row in values}
        for split, values in splits.items()
    }
    if (
        len(unique_sets["validation"]) != 128
        or any(
            unique_sets[left].intersection(unique_sets[right])
            for left, right in (
                ("selection", "fit"),
                ("selection", "validation"),
                ("fit", "validation"),
            )
        )
        or any(
            {row.case_id for row in values} != set(case_order)
            for values in splits.values()
        )
    ):
        raise Lfm2LongHorizonRepairError(
            "Fallback split audit failed row disjointness, validation size, "
            "or all-case stratification."
        )

    def split_plan(split: str, weight: int) -> RowDisjointFallbackSplit:
        ordered = _stratified_row_order(
            splits[split],
            salt=f"{ROW_FALLBACK_POLICY_IDENTITY}\0final\0{split}",
        )
        indices = tuple(row.path_index for row in ordered)
        expanded = (
            indices
            if split == "validation"
            else _cycle_to_weight(ordered, weight=weight)
        )
        termination = tuple(
            row.path_index for row in ordered if "termination" in row.tags
        )
        return RowDisjointFallbackSplit(
            split=split,
            case_ids=tuple(sorted({row.case_id for row in ordered})),
            indices=indices,
            expanded_indices=expanded,
            termination_indices=termination,
        )

    selection = split_plan("selection", 256)
    fit = split_plan("fit", 512)
    validation = split_plan("validation", 128)
    composite = _json_sha256(
        {
            "schema": ROW_FALLBACK_PLAN_SCHEMA,
            "envelope_sha256": tagging.envelope_sha256,
            "ordered_path_sha256": tagging.ordered_path_sha256,
            "tagging_sha256": tagging.tagging_sha256,
            "policy_identity": ROW_FALLBACK_POLICY_IDENTITY,
        }
    )
    body = {
        "schema": ROW_FALLBACK_PLAN_SCHEMA,
        "policy": dict(ROW_FALLBACK_POLICY),
        "policy_identity": ROW_FALLBACK_POLICY_IDENTITY,
        "envelope_sha256": tagging.envelope_sha256,
        "ordered_path_sha256": tagging.ordered_path_sha256,
        "composite_evidence_identity": composite,
        "tagging_sha256": tagging.tagging_sha256,
        "excluded_case_ids": list(tagging.excluded_case_ids),
        "fallback_trigger": {
            "strict_error": strict_error,
            "strict_validation_unique_count": strict_validation_unique_count,
            "required_validation_unique_count": 128,
            "authenticated_insufficient_unique_validation": True,
        },
        "case_disjoint": False,
        "row_disjoint": True,
        "selection": selection.as_dict(),
        "fit": fit.as_dict(),
        "validation": validation.as_dict(),
        "zero_training": {
            "gradients_used": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
    }
    return RowDisjointFallbackPlan(
        envelope_sha256=tagging.envelope_sha256,
        ordered_path_sha256=tagging.ordered_path_sha256,
        composite_evidence_identity=composite,
        tagging_sha256=tagging.tagging_sha256,
        excluded_case_ids=tagging.excluded_case_ids,
        strict_error=strict_error,
        strict_validation_unique_count=strict_validation_unique_count,
        selection=selection,
        fit=fit,
        validation=validation,
        plan_sha256=_json_sha256(body),
    )


def build_repair_anchor_plan(
    case_envelope: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
    *,
    tagging: Lfm2LongHorizonTagging | None = None,
) -> RepairAnchorPlan:
    """Prefer the pure policy; fall back only on authenticated row shortage."""

    authenticated = (
        tag_lfm2_long_horizon_rows(case_envelope, descriptors)
        if tagging is None
        else tagging
    )
    try:
        return build_lfm2_long_horizon_anchor_plan(
            case_envelope,
            descriptors,
        )
    except Lfm2LongHorizonAnchorError as exc:
        message = str(exc)
        match = _VALIDATION_SHORTAGE_PATTERN.fullmatch(message)
        strict_validation_rows = {
            row.path_index
            for row in authenticated.rows
            if row.evidence_split == "validation"
        }
        shortage = 128 - len(strict_validation_rows)
        if (
            match is None
            or shortage <= 0
            or int(match.group("shortage")) != shortage
        ):
            raise
        return _row_disjoint_fallback_plan(
            authenticated,
            strict_error=message,
            strict_validation_unique_count=len(strict_validation_rows),
        )


@dataclass(frozen=True, slots=True)
class EvidenceProfile:
    """Expanded weighted row indices for one repair profile."""

    name: str
    selection_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    selection_termination_occurrences: int
    fit_termination_occurrences: int
    profile_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "name": self.name,
            "selection_indices": list(self.selection_indices),
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "selection_weight": len(self.selection_indices),
            "fit_weight": len(self.fit_indices),
            "validation_weight": len(self.validation_indices),
            "selection_termination_occurrences": (
                self.selection_termination_occurrences
            ),
            "fit_termination_occurrences": self.fit_termination_occurrences,
            "validation_repeats": (
                len(self.validation_indices) - len(set(self.validation_indices))
            ),
            "profile_sha256": self.profile_sha256,
        }


def _termination_indices(
    tagging: Lfm2LongHorizonTagging,
    split: str,
) -> tuple[int, ...]:
    return tuple(
        row.path_index
        for row in tagging.rows
        if row.evidence_split == split and "termination" in row.tags
    )


def _boost_termination(
    base: Sequence[int],
    termination: Sequence[int],
    *,
    extra_occurrences: int,
) -> tuple[int, ...]:
    values = list(int(value) for value in base)
    termination_order = tuple(dict.fromkeys(int(value) for value in termination))
    if extra_occurrences == 0:
        return tuple(values)
    if not termination_order:
        raise Lfm2LongHorizonRepairError(
            "Termination-heavy profile has no authenticated termination paths."
        )
    termination_set = set(termination_order)
    removable_positions = [
        index
        for index, path_index in enumerate(values)
        if path_index not in termination_set
    ]
    if len(removable_positions) < extra_occurrences:
        raise Lfm2LongHorizonRepairError(
            "Termination boost would erase all non-termination horizon coverage."
        )
    remove = set(removable_positions[-extra_occurrences:])
    kept = [value for index, value in enumerate(values) if index not in remove]
    kept.extend(
        termination_order[index % len(termination_order)]
        for index in range(extra_occurrences)
    )
    if len(kept) != len(values):
        raise Lfm2LongHorizonRepairError(
            "Termination reweighting changed the exact row budget."
        )
    return tuple(kept)


def build_evidence_profiles(
    plan: RepairAnchorPlan,
    tagging: Lfm2LongHorizonTagging,
) -> tuple[EvidenceProfile, ...]:
    """Build the balanced and stronger termination-weighted retry profiles."""

    if plan.tagging_sha256 != tagging.tagging_sha256:
        raise Lfm2LongHorizonRepairError(
            "Anchor plan and row tags have different evidence identities."
        )
    balanced_values = {
        "selection": tuple(plan.selection.expanded_indices),
        "fit": tuple(plan.fit.expanded_indices),
        "validation": tuple(plan.validation.expanded_indices),
    }
    fallback_termination = (
        plan.termination_indices_by_split
        if isinstance(plan, RowDisjointFallbackPlan)
        else None
    )
    termination_values = {
        split: (
            fallback_termination[split]
            if fallback_termination is not None
            else _termination_indices(tagging, split)
        )
        for split in ("selection", "fit", "validation")
    }
    profile_values = {
        "balanced": balanced_values,
        "termination_heavy": {
            split: _boost_termination(
                balanced_values[split],
                termination_values[split],
                extra_occurrences=TERMINATION_BOOST[split],
            )
            for split in ("selection", "fit", "validation")
        },
    }
    result = []
    for name in PROFILE_ORDER:
        values = profile_values[name]
        selection_termination = set(termination_values["selection"])
        fit_termination = set(termination_values["fit"])
        body = {
            "schema": PROFILE_SCHEMA,
            "name": name,
            "selection_indices": list(values["selection"]),
            "fit_indices": list(values["fit"]),
            "validation_indices": list(values["validation"]),
            "selection_termination_occurrences": sum(
                value in selection_termination for value in values["selection"]
            ),
            "fit_termination_occurrences": sum(
                value in fit_termination for value in values["fit"]
            ),
            "anchor_plan_sha256": plan.plan_sha256,
            "tagging_sha256": tagging.tagging_sha256,
            "validation_repeated": (
                len(values["validation"]) != len(set(values["validation"]))
            ),
        }
        if body["validation_repeated"]:
            raise Lfm2LongHorizonRepairError(
                "Evidence profile repeated a validation path."
            )
        result.append(
            EvidenceProfile(
                name=name,
                selection_indices=values["selection"],
                fit_indices=values["fit"],
                validation_indices=values["validation"],
                selection_termination_occurrences=int(
                    body["selection_termination_occurrences"]
                ),
                fit_termination_occurrences=int(
                    body["fit_termination_occurrences"]
                ),
                profile_sha256=_json_sha256(body),
            )
        )
    if (
        result[1].selection_termination_occurrences
        <= result[0].selection_termination_occurrences
        or result[1].fit_termination_occurrences
        <= result[0].fit_termination_occurrences
    ):
        raise Lfm2LongHorizonRepairError(
            "Termination-heavy profile did not increase termination weight."
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RetryCandidate:
    candidate_index: int
    profile_name: str
    profile_sha256: str
    endpoint_ridge: float
    layer_order: tuple[int, ...]
    candidate_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "profile_name": self.profile_name,
            "profile_sha256": self.profile_sha256,
            "endpoint_ridge": self.endpoint_ridge,
            "layer_order": list(self.layer_order),
            "candidate_sha256": self.candidate_sha256,
        }


def build_retry_grid(
    profiles: Sequence[EvidenceProfile],
) -> tuple[RetryCandidate, ...]:
    """Return the fixed two-profile × three-ridge candidate grid."""

    by_name = {profile.name: profile for profile in profiles}
    if set(by_name) != set(PROFILE_ORDER):
        raise Lfm2LongHorizonRepairError(
            "Retry grid requires balanced and termination_heavy profiles."
        )
    result = []
    for profile_name in PROFILE_ORDER:
        profile = by_name[profile_name]
        for ridge in RIDGE_VALUES:
            body = {
                "profile_name": profile_name,
                "profile_sha256": profile.profile_sha256,
                "endpoint_ridge": ridge,
                "layer_order": list(REJECTED_LAYER_ORDER),
                "zero_training": True,
            }
            result.append(
                RetryCandidate(
                    candidate_index=len(result),
                    profile_name=profile_name,
                    profile_sha256=profile.profile_sha256,
                    endpoint_ridge=ridge,
                    layer_order=REJECTED_LAYER_ORDER,
                    candidate_sha256=_json_sha256(body),
                )
            )
    return tuple(result)


def filter_retry_grid(
    fixed_grid: Sequence[RetryCandidate],
    candidate_indices: Sequence[int],
) -> tuple[RetryCandidate, ...]:
    """Authenticate all six fixed arms, then select without renumbering."""

    grid = tuple(fixed_grid)
    expected_pairs = tuple(
        (profile_name, ridge)
        for profile_name in PROFILE_ORDER
        for ridge in RIDGE_VALUES
    )
    if (
        len(grid) != len(FIXED_CANDIDATE_INDICES)
        or tuple(candidate.candidate_index for candidate in grid)
        != FIXED_CANDIDATE_INDICES
        or tuple(
            (candidate.profile_name, candidate.endpoint_ridge)
            for candidate in grid
        )
        != expected_pairs
    ):
        raise Lfm2LongHorizonRepairError(
            "Candidate filtering requires the complete authenticated fixed grid."
        )
    for candidate in grid:
        body = {
            "profile_name": candidate.profile_name,
            "profile_sha256": candidate.profile_sha256,
            "endpoint_ridge": candidate.endpoint_ridge,
            "layer_order": list(candidate.layer_order),
            "zero_training": True,
        }
        if (
            candidate.layer_order != REJECTED_LAYER_ORDER
            or _json_sha256(body) != candidate.candidate_sha256
        ):
            raise Lfm2LongHorizonRepairError(
                "A fixed-grid candidate identity failed authentication."
            )
    requested = _candidate_indices(candidate_indices)
    return tuple(grid[index] for index in requested)


def candidate_selection_report(
    fixed_grid: Sequence[RetryCandidate],
    selected_grid: Sequence[RetryCandidate],
    candidate_indices: Sequence[int],
) -> dict[str, Any]:
    """Seal the authenticated six-arm grid and this run's bounded subset."""

    authenticated = filter_retry_grid(fixed_grid, candidate_indices)
    selected = tuple(selected_grid)
    if selected != authenticated:
        raise Lfm2LongHorizonRepairError(
            "Selected retry candidates differ from the authenticated request."
        )
    request = _candidate_request(candidate_indices)
    body = {
        **request,
        "complete_fixed_grid_sha256": _json_sha256(
            [candidate.as_dict() for candidate in fixed_grid]
        ),
        "selected_candidates_sha256": _json_sha256(
            [candidate.as_dict() for candidate in selected]
        ),
        "selected_candidate_indices": [
            candidate.candidate_index for candidate in selected
        ],
        "selected_candidate_count": len(selected),
        "candidate_indices_preserved": True,
        "candidate_identities_preserved": True,
        "filter_applied_after_fixed_grid_authentication": True,
    }
    return {**body, "selection_sha256": _json_sha256(body)}


@dataclass(frozen=True, slots=True)
class AtomicTrancheResult:
    """Outcome of a four-layer all-or-nothing attempt."""

    accepted: bool
    rolled_back: bool
    rollback_exact: bool
    baseline_model_sha256: str
    result_model_sha256: str
    baseline_layer_sha256: tuple[tuple[int, str], ...]
    result_layer_sha256: tuple[tuple[int, str], ...]
    layer_reports: tuple[Mapping[str, Any], ...]
    gate: Mapping[str, Any]
    error: Mapping[str, Any] | None
    tranche_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TRANCHE_SCHEMA,
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "rollback_exact": self.rollback_exact,
            "baseline_model_sha256": self.baseline_model_sha256,
            "result_model_sha256": self.result_model_sha256,
            "baseline_layer_sha256": [
                {"layer_index": index, "state_sha256": digest}
                for index, digest in self.baseline_layer_sha256
            ],
            "result_layer_sha256": [
                {"layer_index": index, "state_sha256": digest}
                for index, digest in self.result_layer_sha256
            ],
            "layer_reports": [dict(report) for report in self.layer_reports],
            "gate": dict(self.gate),
            "error": None if self.error is None else dict(self.error),
            "tranche_sha256": self.tranche_sha256,
        }


def run_atomic_tranche(
    layer_indices: Sequence[int],
    *,
    snapshot_layer: Callable[[int], Any],
    restore_layer: Callable[[int, Any], bool],
    layer_state_sha256: Callable[[int], str],
    model_state_sha256: Callable[[], str],
    mutate_layer: Callable[[int], Mapping[str, Any]],
    promotion_gate: Callable[[], Mapping[str, Any]],
) -> AtomicTrancheResult:
    """Mutate a layer sequence and exactly restore the whole tranche on failure."""

    normalized = tuple(
        _strict_int(index, f"layer_indices[{position}]", maximum=13)
        for position, index in enumerate(layer_indices)
    )
    if normalized != REJECTED_LAYER_ORDER:
        raise Lfm2LongHorizonRepairError(
            "Atomic tranche must target rejected layers 6, 8, 11, 13 in order."
        )
    snapshots = {index: snapshot_layer(index) for index in normalized}
    baseline_layers = tuple(
        (index, _digest(layer_state_sha256(index), "layer state SHA-256"))
        for index in normalized
    )
    baseline_model = _digest(model_state_sha256(), "model state SHA-256")
    reports = []
    gate: Mapping[str, Any] = {
        "accepted": False,
        "reason": "promotion gate was not reached",
    }
    error = None
    try:
        for index in normalized:
            report = mutate_layer(index)
            if not isinstance(report, Mapping):
                raise Lfm2LongHorizonRepairError(
                    f"Layer {index} mutation did not return a report."
                )
            reports.append(dict(report))
        gate = promotion_gate()
        if not isinstance(gate, Mapping) or not isinstance(
            gate.get("accepted"),
            bool,
        ):
            raise Lfm2LongHorizonRepairError(
                "Promotion gate must return a mapping with bool accepted."
            )
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        gate = {
            "accepted": False,
            "reason": "mutation or promotion gate raised",
        }
    accepted = bool(gate["accepted"]) and error is None
    rolled_back = not accepted
    if rolled_back:
        restore_failures = [
            index
            for index in normalized
            if not bool(restore_layer(index, snapshots[index]))
        ]
        if restore_failures:
            raise Lfm2LongHorizonRepairError(
                f"Whole-tranche rollback failed for layers {restore_failures}."
            )
    result_layers = tuple(
        (index, _digest(layer_state_sha256(index), "layer state SHA-256"))
        for index in normalized
    )
    result_model = _digest(model_state_sha256(), "model state SHA-256")
    rollback_exact = bool(
        accepted
        or (
            result_layers == baseline_layers
            and result_model == baseline_model
        )
    )
    if rolled_back and not rollback_exact:
        raise Lfm2LongHorizonRepairError(
            "Whole-tranche rollback did not restore exact layer and model state."
        )
    body = {
        "schema": TRANCHE_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_exact": rollback_exact,
        "baseline_model_sha256": baseline_model,
        "result_model_sha256": result_model,
        "baseline_layer_sha256": list(baseline_layers),
        "result_layer_sha256": list(result_layers),
        "layer_reports": reports,
        "gate": dict(gate),
        "error": error,
    }
    return AtomicTrancheResult(
        accepted=accepted,
        rolled_back=rolled_back,
        rollback_exact=rollback_exact,
        baseline_model_sha256=baseline_model,
        result_model_sha256=result_model,
        baseline_layer_sha256=baseline_layers,
        result_layer_sha256=result_layers,
        layer_reports=tuple(reports),
        gate=dict(gate),
        error=error,
        tranche_sha256=_json_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class RepairRunConfig:
    parent_model: Path
    source_child: Path
    source_child_report: Path
    source_child_report_sha256: str
    cached_parent_outputs: Path
    cached_parent_outputs_sha256: str
    tokenizer_directory: Path
    parent_horizon_cache_root: Path
    child_horizon_cache_root: Path
    work_root: Path
    checkpoint_destination: Path
    report_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    data_contract_only: bool = False
    candidate_indices: tuple[int, ...] = FIXED_CANDIDATE_INDICES
    sample_last_tokens: int = 8
    max_capture_bytes: int = 512 * 1024 * 1024
    capture_logit_rtol: float = 0.20
    capture_logit_atol: float = 1.0
    endpoint_refinement_passes: int = 2
    max_endpoint_relative_rmse: float = 1.5
    min_endpoint_cosine_similarity: float = -0.5

    def __post_init__(self) -> None:
        for name in (
            "parent_model",
            "source_child",
            "tokenizer_directory",
        ):
            object.__setattr__(
                self,
                name,
                _safe_local_directory(getattr(self, name), name),
            )
        for name in ("source_child_report", "cached_parent_outputs"):
            object.__setattr__(
                self,
                name,
                _safe_local_file(getattr(self, name), name),
            )
        for name in (
            "parent_horizon_cache_root",
            "child_horizon_cache_root",
            "work_root",
            "checkpoint_destination",
            "report_path",
        ):
            object.__setattr__(
                self,
                name,
                Path(getattr(self, name)).expanduser().absolute(),
            )
        object.__setattr__(
            self,
            "source_child_report_sha256",
            _digest(
                self.source_child_report_sha256,
                "source_child_report_sha256",
            ),
        )
        object.__setattr__(
            self,
            "cached_parent_outputs_sha256",
            _digest(
                self.cached_parent_outputs_sha256,
                "cached_parent_outputs_sha256",
            ),
        )
        if not isinstance(self.data_contract_only, bool):
            raise Lfm2LongHorizonRepairError(
                "data_contract_only must be bool."
            )
        object.__setattr__(
            self,
            "candidate_indices",
            _candidate_indices(self.candidate_indices),
        )
        for name in (
            "sample_last_tokens",
            "max_capture_bytes",
            "endpoint_refinement_passes",
        ):
            object.__setattr__(
                self,
                name,
                _strict_int(getattr(self, name), name, minimum=1),
            )
        for name in (
            "capture_logit_rtol",
            "capture_logit_atol",
            "max_endpoint_relative_rmse",
            "min_endpoint_cosine_similarity",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name),
            )
        if self.capture_logit_rtol < 0.0 or self.capture_logit_atol < 0.0:
            raise Lfm2LongHorizonRepairError(
                "Capture tolerances must be nonnegative."
            )
        if self.max_endpoint_relative_rmse < 0.0 or not (
            -1.0 <= self.min_endpoint_cosine_similarity <= 1.0
        ):
            raise Lfm2LongHorizonRepairError(
                "Endpoint fit thresholds are invalid."
            )
        if self.checkpoint_destination.exists():
            raise Lfm2LongHorizonRepairError(
                "checkpoint_destination already exists; refusing replacement."
            )
        if self.report_path.exists():
            raise Lfm2LongHorizonRepairError(
                "report_path already exists; refusing replacement."
            )
        if self.checkpoint_destination == self.report_path:
            raise Lfm2LongHorizonRepairError(
                "checkpoint_destination and report_path must differ."
            )
        protected = (
            self.parent_model,
            self.source_child,
            self.tokenizer_directory,
        )
        for output in (
            self.parent_horizon_cache_root,
            self.child_horizon_cache_root,
            self.work_root,
            self.checkpoint_destination,
            self.report_path,
        ):
            if any(_is_within(output, root) for root in protected):
                raise Lfm2LongHorizonRepairError(
                    "Repair outputs must be outside all source directories."
                )
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2LongHorizonRepairError("device must be nonempty.")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise Lfm2LongHorizonRepairError("dtype must be nonempty.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_model": str(self.parent_model),
            "source_child": str(self.source_child),
            "source_child_report": str(self.source_child_report),
            "source_child_report_sha256": self.source_child_report_sha256,
            "cached_parent_outputs": str(self.cached_parent_outputs),
            "cached_parent_outputs_sha256": (
                self.cached_parent_outputs_sha256
            ),
            "tokenizer_directory": str(self.tokenizer_directory),
            "parent_horizon_cache_root": str(self.parent_horizon_cache_root),
            "child_horizon_cache_root": str(self.child_horizon_cache_root),
            "work_root": str(self.work_root),
            "checkpoint_destination": str(self.checkpoint_destination),
            "report_path": str(self.report_path),
            "device": self.device,
            "dtype": self.dtype,
            "data_contract_only": self.data_contract_only,
            "candidate_selection": _candidate_request(
                self.candidate_indices
            ),
            "sample_last_tokens": self.sample_last_tokens,
            "max_decode_steps": MAX_DECODE_STEPS,
            "max_capture_bytes": self.max_capture_bytes,
            "capture_logit_rtol": self.capture_logit_rtol,
            "capture_logit_atol": self.capture_logit_atol,
            "endpoint_refinement_passes": self.endpoint_refinement_passes,
            "max_endpoint_relative_rmse": (
                self.max_endpoint_relative_rmse
            ),
            "min_endpoint_cosine_similarity": (
                self.min_endpoint_cosine_similarity
            ),
            "profiles": list(PROFILE_ORDER),
            "ridge_values": list(RIDGE_VALUES),
            "rejected_layer_order": list(REJECTED_LAYER_ORDER),
            "promotion_case_policy": (
                "twelve-heldout-manual-cases-only"
            ),
            "full_28_case_role": "transductive-diagnostic-only",
            "training_allowed": False,
            "parent_response_generation_allowed": False,
        }


def build_data_contract_report(
    config: RepairRunConfig,
    *,
    envelope: SealedEnvelopeAudit,
    source: SourceChildBinding,
) -> dict[str, Any]:
    """Build the terminal preflight report without loading a model."""

    if source.parent_cache_key != envelope.parent_cache_key:
        raise Lfm2LongHorizonRepairError(
            "Source-child report and original envelope bind different parent caches."
        )
    status = (
        "data_contract_ready"
        if envelope.ready
        else "data_contract_blocked"
    )
    candidate_request = _candidate_request(config.candidate_indices)
    body = {
        "schema": RUN_SCHEMA,
        "status": status,
        "config": config.as_dict(),
        "candidate_selection": candidate_request,
        "source_child": source.as_dict(),
        "sealed_envelope": envelope.as_dict(),
        "identities": {
            "source_child_report_sha256": source.report_file_sha256,
            "original_envelope_file_sha256": envelope.source_file_sha256,
            "original_envelope_sha256": envelope.envelope_sha256,
            "original_parent_cache_key": envelope.parent_cache_key,
            "original_parent_state_fingerprint": (
                envelope.parent_state_fingerprint
            ),
            "tokenizer_fingerprint": envelope.tokenizer_fingerprint,
            "preflight_binding_sha256": _json_sha256(
                {
                    "source_child_report_sha256": source.report_file_sha256,
                    "source_child_state_sha256": (
                        source.source_child_state_sha256
                    ),
                    "source_report_parent_cache_key": source.parent_cache_key,
                    "envelope_file_sha256": envelope.source_file_sha256,
                    "envelope_sha256": envelope.envelope_sha256,
                    "parent_cache_key": envelope.parent_cache_key,
                    "parent_state_fingerprint": (
                        envelope.parent_state_fingerprint
                    ),
                    "tokenizer_fingerprint": envelope.tokenizer_fingerprint,
                    "split_identity_sha256": envelope.split_identity_sha256,
                    "anchor_policy_identity": ANCHOR_POLICY_IDENTITY,
                    "candidate_request_sha256": (
                        candidate_request["request_sha256"]
                    ),
                }
            ),
        },
        "execution": {
            "models_loaded": False,
            "parent_forward_calls": 0,
            "parent_response_generation_calls": 0,
            "child_forward_calls": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "checkpoint_published": False,
        },
        "next_action": (
            "Create a newly sealed, locally tokenized enrichment of the same "
            "28 cached rows containing every blocker field. Reuse the existing "
            "parent_output_ids exactly; do not execute or regenerate the parent."
            if not envelope.ready
            else (
                "Rerun without --data-contract-only to build/reuse the parent "
                "horizon cache and execute "
                + (
                    "all six fixed candidates."
                    if config.candidate_indices == FIXED_CANDIDATE_INDICES
                    else "only the requested candidate subset."
                )
            )
        ),
    }
    return {**body, "report_sha256": _json_sha256(body)}


def _runtime_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from experiments import run_lfm2_dynamic_catalog as catalog
        from experiments import run_lfm2_residual_attention_fold as residual
    except ImportError:  # Direct execution from experiments/.
        import torch
        import run_lfm2_dynamic_catalog as catalog
        import run_lfm2_residual_attention_fold as residual
    from neural_forge.dynamic_lfm2_behavior import (
        Lfm2CachedWholeOutputBaseline,
        Lfm2WholeOutputBehaviorEvaluator,
        load_lfm2_cached_whole_output_baseline,
    )
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        Lfm2CircuitTraceCaptureConfig,
        build_lfm2_circuit_trace_cache,
        fingerprint_lfm2_parent,
    )

    behavior = {
        "baseline_type": Lfm2CachedWholeOutputBaseline,
        "evaluator_type": Lfm2WholeOutputBehaviorEvaluator,
        "load_baseline": load_lfm2_cached_whole_output_baseline,
    }
    cache = {
        "capture_config_type": Lfm2CircuitTraceCaptureConfig,
        "build": build_lfm2_circuit_trace_cache,
        "fingerprint": fingerprint_lfm2_parent,
    }
    return torch, residual, catalog, (behavior, cache)


def _subset_baseline(
    baseline: Any,
    case_ids: Sequence[str],
    *,
    baseline_type: Any,
) -> Any:
    selected_ids = tuple(case_ids)
    selected = set(selected_ids)
    cases = tuple(
        case for case in baseline.cases if case.case_id in selected
    )
    contracts = tuple(
        row for row in baseline.contracts if row[0] in selected
    )
    if (
        len(cases) != len(selected_ids)
        or len(contracts) != len(selected_ids)
        or {case.case_id for case in cases} != selected
    ):
        raise Lfm2LongHorizonRepairError(
            "Could not construct an exact case-subset behavior baseline."
        )
    return baseline_type(
        source_path=baseline.source_path,
        source_sha256=baseline.source_sha256,
        envelope_sha256=baseline.envelope_sha256,
        parent_cache_key=baseline.parent_cache_key,
        parent_state_fingerprint=baseline.parent_state_fingerprint,
        tokenizer_fingerprint=baseline.tokenizer_fingerprint,
        cases=cases,
        contracts=contracts,
        generation_limit=baseline.generation_limit,
    )


def _behavior_evaluator(
    baseline: Any,
    tokenizer: Any,
    *,
    evaluator_type: Any,
) -> Any:
    return evaluator_type(
        baseline=baseline,
        tokenizer=tokenizer,
        maximum_prompt_tokens=512,
        fragment_minimum_characters=12,
        maximum_new_empty_responses=1,
        maximum_new_fragment_responses=1,
        minimum_parent_nonempty_retention=0.90,
        minimum_task_score_retention=0.90,
        maximum_parent_task_score_regressions=1,
        maximum_new_generation_limit_saturations=1,
        maximum_repeated_ngram_share_regressions=1,
        maximum_repeated_bigram_share_delta=0.10,
    )


def _behavior_event(
    evaluator: Any,
    model: Any,
    *,
    phase: str,
    decision_role: str,
) -> dict[str, Any]:
    decision = evaluator.evaluate_model(model, phase=phase)
    event = dict(evaluator.events[-1])
    return {
        "accepted": bool(decision.accepted),
        "decision": decision.as_dict(),
        "event": event,
        "event_sha256": _json_sha256(event),
        "case_count": len(evaluator.baseline.cases),
        "decision_role": decision_role,
        "exact_token_match_used_for_decision": False,
        "parent_forward_calls": 0,
    }


def _tensorize_batches(torch: Any, records: Sequence[Mapping[str, Any]]) -> tuple:
    result = []
    for row in records:
        result.append(
            {
                "input_ids": torch.tensor(
                    [row["input_ids"]],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    [row["attention_mask"]],
                    dtype=torch.long,
                ),
                "forced_decode_ids": torch.tensor(
                    [row["forced_decode_ids"]],
                    dtype=torch.long,
                ),
            }
        )
    return tuple(result)


def _directory_keys(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        child.name
        for child in path.iterdir()
        if child.is_dir() and not child.is_symlink()
    }


def _repair_one_layer(
    *,
    residual: Any,
    parent: Any,
    child: Any,
    layer_index: int,
    parent_rows: Any,
    child_cache: Any,
    child_rows: Any,
    calibration_batches: Sequence[Mapping[str, Any]],
    child_cache_root: Path,
    capture_config: Any,
    profile: EvidenceProfile,
    endpoint_ridge: float,
    endpoint_refinement_passes: int,
    max_endpoint_relative_rmse: float,
    min_endpoint_cosine_similarity: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """Install one long-horizon closed-loop layer candidate without gating it."""

    from neural_forge.hf_lfm2_closed_loop_fold import (
        Lfm2ClosedLoopFoldConfig,
        build_lfm2_closed_loop_fold_plan,
        install_lfm2_closed_loop_fold_plan,
        restore_lfm2_layer_snapshot,
        snapshot_lfm2_layer,
    )

    source_layer = residual._lfm2_layers(parent)[layer_index]
    target_layer = residual._lfm2_layers(child)[layer_index]
    baseline = snapshot_lfm2_layer(target_layer)
    fold_config = Lfm2ClosedLoopFoldConfig(
        endpoint_ridge=endpoint_ridge,
        cluster_ridge=1e-8,
        max_endpoint_relative_rmse=max_endpoint_relative_rmse,
        min_endpoint_cosine_similarity=min_endpoint_cosine_similarity,
    )
    producer = residual.derive_layer_producer_plan(
        source_layer,
        target_layer,
        parent_rows,
        layer_index,
        fold_config=fold_config,
        selection_indices=profile.selection_indices,
        fit_indices=profile.fit_indices,
        validation_indices=profile.validation_indices,
    )
    working_cache = child_cache
    working_rows = child_rows
    refinements = []
    final_evidence = None
    try:
        for pass_index in range(endpoint_refinement_passes):
            evidence, split = residual.build_cached_layer_evidence(
                parent_rows,
                working_rows,
                layer_index=layer_index,
                selection_indices=profile.selection_indices,
                fit_indices=profile.fit_indices,
                validation_indices=profile.validation_indices,
            )
            if pass_index + 1 == endpoint_refinement_passes:
                final_evidence = evidence
                break
            provisional = build_lfm2_closed_loop_fold_plan(
                source_layer,
                target_layer,
                evidence,
                config=fold_config,
            )
            residual.assert_closed_loop_preserves_producer_plan(
                provisional,
                producer,
            )
            residual._copy_plan_state(target_layer, provisional)
            working_cache, working_rows = residual.build_child_trace_cache(
                child_cache_root,
                child,
                calibration_batches,
                capture_config=capture_config,
            )
            refinements.append(
                {
                    "pass_index": pass_index,
                    "plan": provisional.as_dict(),
                    "child_cache_key": working_cache.cache_key,
                    "child_trace_rows_sha256": (
                        working_rows.trace_rows_sha256
                    ),
                }
            )
        if final_evidence is None:
            raise Lfm2LongHorizonRepairError(
                "Endpoint refinement produced no final evidence."
            )
        if not restore_lfm2_layer_snapshot(target_layer, baseline):
            raise Lfm2LongHorizonRepairError(
                f"Could not restore layer {layer_index} before final planning."
            )
        final_plan = build_lfm2_closed_loop_fold_plan(
            source_layer,
            target_layer,
            final_evidence,
            config=fold_config,
        )
        residual.assert_closed_loop_preserves_producer_plan(
            final_plan,
            producer,
        )
        transaction = install_lfm2_closed_loop_fold_plan(
            target_layer,
            final_plan,
        )
        transaction_report = transaction.commit().as_dict()
        result_cache, result_rows = residual.build_child_trace_cache(
            child_cache_root,
            child,
            calibration_batches,
            capture_config=capture_config,
        )
        return result_cache, result_rows, {
            "layer_index": layer_index,
            "profile": profile.name,
            "profile_sha256": profile.profile_sha256,
            "endpoint_ridge": endpoint_ridge,
            "producer_plan": producer.as_dict(),
            "refinements": refinements,
            "final_plan": final_plan.as_dict(),
            "transaction": transaction_report,
            "result_child_cache_key": result_cache.cache_key,
            "result_child_trace_rows_sha256": result_rows.trace_rows_sha256,
            "zero_training": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
    except Exception:
        restore_lfm2_layer_snapshot(target_layer, baseline)
        raise


def execute_repair(config: RepairRunConfig) -> dict[str, Any]:
    """Run preflight, then the six atomic candidates when evidence is complete."""

    if not isinstance(config, RepairRunConfig):
        raise TypeError("config must be RepairRunConfig.")
    source = load_source_child_binding(
        config.source_child_report,
        expected_sha256=config.source_child_report_sha256,
    )
    # First authenticate only data that must already be sealed.  Token text is
    # an interface view of the sealed IDs and is decoded locally only after the
    # tokenizer identity itself has been checked.
    envelope = audit_sealed_horizon_envelope(
        config.cached_parent_outputs,
        expected_sha256=config.cached_parent_outputs_sha256,
        require_token_texts=False,
    )
    preflight = build_data_contract_report(
        config,
        envelope=envelope,
        source=source,
    )
    if not envelope.ready:
        _atomic_write_json(config.report_path, preflight)
        return preflight

    torch, residual, catalog, runtime = _runtime_modules()
    behavior_runtime, cache_runtime = runtime
    tokenizer_fingerprint = catalog.fingerprint_local_tokenizer(
        config.tokenizer_directory
    )
    if tokenizer_fingerprint != envelope.tokenizer_fingerprint:
        raise Lfm2LongHorizonRepairError(
            "Local tokenizer differs from the sealed whole-output envelope."
        )
    tokenizer = residual.load_local_tokenizer(config.tokenizer_directory)
    eos_ids = _tokenizer_eos_ids(tokenizer)
    envelope = audit_sealed_horizon_envelope(
        config.cached_parent_outputs,
        expected_sha256=config.cached_parent_outputs_sha256,
        token_text_decoder=lambda ids: _decode_token_pieces(tokenizer, ids),
        tokenizer_eos_token_ids=eos_ids,
    )
    preflight = build_data_contract_report(
        config,
        envelope=envelope,
        source=source,
    )
    if not envelope.ready or config.data_contract_only:
        _atomic_write_json(config.report_path, preflight)
        return preflight

    # Prove the exact anchor policy is feasible before either model is loaded.
    anchor_envelope = build_anchor_case_envelope(envelope)
    synthetic_paths = synthetic_trace_path_descriptors(envelope)
    try:
        build_repair_anchor_plan(
            anchor_envelope,
            synthetic_paths,
        )
    except Exception as exc:
        blocked = dict(preflight)
        blocked.pop("report_sha256", None)
        blocked["status"] = "data_contract_blocked"
        blocked["sealed_envelope"] = dict(blocked["sealed_envelope"])
        blockers = list(blocked["sealed_envelope"]["blockers"])
        blockers.append(
            {
                "code": "anchor_quota_preflight_failed",
                "case_id": None,
                "field": "parent_output_ids",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        blocked["sealed_envelope"]["ready"] = False
        blocked["sealed_envelope"]["blockers"] = blockers
        blocked["next_action"] = (
            "Provide enough exact case-disjoint output-token predecessors to "
            "satisfy the 256/512/128 anchor policy without repeating validation."
        )
        blocked["report_sha256"] = _json_sha256(blocked)
        _atomic_write_json(config.report_path, blocked)
        return blocked

    config.parent_horizon_cache_root.mkdir(parents=True, exist_ok=True)
    config.child_horizon_cache_root.mkdir(parents=True, exist_ok=True)
    config.work_root.mkdir(parents=True, exist_ok=True)
    config.checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    dtype = residual._torch_dtype(config.dtype)
    parent = residual.load_local_lfm2_model(
        config.parent_model,
        device=config.device,
        dtype=dtype,
    )
    child = residual.load_local_lfm2_model(
        config.source_child,
        device=config.device,
        dtype=dtype,
    )
    if residual.model_state_sha256(child) != source.source_child_state_sha256:
        raise Lfm2LongHorizonRepairError(
            "Loaded source child state differs from its authenticated report."
        )
    if residual.unique_parameter_count(child) != EXPECTED_CHILD_UNIQUE_PARAMETERS:
        raise Lfm2LongHorizonRepairError(
            "Loaded source child does not have 206,726,912 unique parameters."
        )
    source_gradient_audit = residual._gradient_audit(child)
    if not source_gradient_audit["zero_gradient"]:
        raise Lfm2LongHorizonRepairError(
            "Loaded source child already contains parameter gradients."
        )
    parent_fingerprint = cache_runtime["fingerprint"](parent).identity_sha256
    if parent_fingerprint != envelope.parent_state_fingerprint:
        raise Lfm2LongHorizonRepairError(
            "Loaded parent differs from the sealed whole-output envelope."
        )
    batch_records = build_horizon_batch_records(envelope)
    batches = _tensorize_batches(torch, batch_records)
    capture_config = cache_runtime["capture_config_type"](
        sample_last_tokens=config.sample_last_tokens,
        max_decode_steps=MAX_DECODE_STEPS,
        max_capture_bytes=config.max_capture_bytes,
        logit_rtol=config.capture_logit_rtol,
        logit_atol=config.capture_logit_atol,
        require_top1_agreement=False,
    )
    parent_cache_before = _directory_keys(config.parent_horizon_cache_root)
    parent_cache = cache_runtime["build"](
        config.parent_horizon_cache_root,
        parent,
        batches,
        config=capture_config,
        input_device=residual._model_device(parent),
    )
    parent_cache_reused = parent_cache.cache_key in parent_cache_before
    parent_rows = residual.extract_trace_rows(parent_cache)
    actual_descriptors = trace_path_descriptors_from_paths(
        parent_rows.paths,
        envelope,
    )
    tagging = tag_lfm2_long_horizon_rows(
        anchor_envelope,
        actual_descriptors,
    )
    anchor_plan = build_repair_anchor_plan(
        anchor_envelope,
        actual_descriptors,
        tagging=tagging,
    )
    profiles = build_evidence_profiles(anchor_plan, tagging)
    profile_map = {profile.name: profile for profile in profiles}
    fixed_grid = build_retry_grid(profiles)
    grid = filter_retry_grid(fixed_grid, config.candidate_indices)
    candidate_selection = candidate_selection_report(
        fixed_grid,
        grid,
        config.candidate_indices,
    )
    baseline = behavior_runtime["load_baseline"](
        config.cached_parent_outputs,
        expected_sha256=config.cached_parent_outputs_sha256,
        expected_parent_cache_key=envelope.parent_cache_key,
        expected_parent_state_fingerprint=envelope.parent_state_fingerprint,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_generation_limit=MAX_DECODE_STEPS,
        fragment_minimum_characters=12,
    )
    manual_baseline = _subset_baseline(
        baseline,
        envelope.manual_case_ids,
        baseline_type=behavior_runtime["baseline_type"],
    )
    manual_evaluator = _behavior_evaluator(
        manual_baseline,
        tokenizer,
        evaluator_type=behavior_runtime["evaluator_type"],
    )
    full_evaluator = _behavior_evaluator(
        baseline,
        tokenizer,
        evaluator_type=behavior_runtime["evaluator_type"],
    )
    base_child_cache, base_child_rows = residual.build_child_trace_cache(
        config.child_horizon_cache_root,
        child,
        batches,
        capture_config=capture_config,
    )
    attempts = []
    accepted_stage: Path | None = None
    accepted_reload = None
    accepted_diagnostic = None
    accepted_candidate = None

    for candidate in grid:
        profile = profile_map[candidate.profile_name]
        working_cache = base_child_cache
        working_rows = base_child_rows
        staged: Path | None = None
        reload_holder: dict[str, Any] = {}
        layers = residual._lfm2_layers(child)

        def mutate(layer_index: int) -> Mapping[str, Any]:
            nonlocal working_cache, working_rows
            working_cache, working_rows, report = _repair_one_layer(
                residual=residual,
                parent=parent,
                child=child,
                layer_index=layer_index,
                parent_rows=parent_rows,
                child_cache=working_cache,
                child_rows=working_rows,
                calibration_batches=batches,
                child_cache_root=config.child_horizon_cache_root,
                capture_config=capture_config,
                profile=profile,
                endpoint_ridge=candidate.endpoint_ridge,
                endpoint_refinement_passes=config.endpoint_refinement_passes,
                max_endpoint_relative_rmse=(
                    config.max_endpoint_relative_rmse
                ),
                min_endpoint_cosine_similarity=(
                    config.min_endpoint_cosine_similarity
                ),
            )
            return report

        def gate() -> Mapping[str, Any]:
            nonlocal staged
            gradient_audit = residual._gradient_audit(child)
            if not gradient_audit["zero_gradient"]:
                raise Lfm2LongHorizonRepairError(
                    "Repair tranche wrote parameter gradients."
                )
            in_memory = _behavior_event(
                manual_evaluator,
                child,
                phase=f"candidate_{candidate.candidate_index}_in_memory_manual",
                decision_role="binding-promotion",
            )
            if not in_memory["accepted"]:
                return {
                    "accepted": False,
                    "stage": "in_memory_manual_only",
                    "in_memory_manual": in_memory,
                    "gradient_audit": gradient_audit,
                }
            staged = (
                config.checkpoint_destination.parent
                / (
                    f".{config.checkpoint_destination.name}."
                    f"candidate-{candidate.candidate_index}-{uuid.uuid4().hex}"
                )
            )
            checkpoint = residual.atomic_save_local_checkpoint(
                child,
                staged,
                output_root=config.checkpoint_destination.parent,
            )
            reloaded = residual.load_local_lfm2_model(
                staged,
                device=config.device,
                dtype=dtype,
            )
            if (
                residual.model_state_sha256(reloaded)
                != residual.model_state_sha256(child)
                or residual.unique_parameter_count(reloaded)
                != EXPECTED_CHILD_UNIQUE_PARAMETERS
            ):
                raise Lfm2LongHorizonRepairError(
                    "Fresh repair checkpoint reload changed state or size."
                )
            reload_manual = _behavior_event(
                manual_evaluator,
                reloaded,
                phase=f"candidate_{candidate.candidate_index}_reload_manual",
                decision_role="binding-promotion",
            )
            if not reload_manual["accepted"]:
                shutil.rmtree(staged)
                staged = None
                return {
                    "accepted": False,
                    "stage": "fresh_reload_manual_only",
                    "in_memory_manual": in_memory,
                    "fresh_reload_manual": reload_manual,
                    "gradient_audit": gradient_audit,
                }
            diagnostic = _behavior_event(
                full_evaluator,
                reloaded,
                phase=f"candidate_{candidate.candidate_index}_full_28_diagnostic",
                decision_role="transductive-diagnostic-only",
            )
            reload_holder.update(
                {
                    "model": reloaded,
                    "checkpoint": checkpoint,
                    "manual": reload_manual,
                    "diagnostic": diagnostic,
                }
            )
            return {
                "accepted": True,
                "stage": "fresh_reload_manual_only",
                "in_memory_manual": in_memory,
                "fresh_reload_manual": reload_manual,
                "full_28_diagnostic": diagnostic,
                "full_28_diagnostic_used_for_decision": False,
                "gradient_audit": gradient_audit,
            }

        result = run_atomic_tranche(
            candidate.layer_order,
            snapshot_layer=lambda index: residual.snapshot_lfm2_layer(
                layers[index]
            ),
            restore_layer=lambda index, snapshot: (
                residual.restore_lfm2_layer_snapshot(
                    layers[index],
                    snapshot,
                )
            ),
            layer_state_sha256=lambda index: residual.snapshot_lfm2_layer(
                layers[index]
            ).state_sha256,
            model_state_sha256=lambda: residual.model_state_sha256(child),
            mutate_layer=mutate,
            promotion_gate=gate,
        )
        if not result.accepted and staged is not None and staged.exists():
            shutil.rmtree(staged)
            staged = None
        attempts.append(
            {
                "candidate": candidate.as_dict(),
                "tranche": result.as_dict(),
            }
        )
        if result.accepted:
            if staged is None or not staged.is_dir() or not reload_holder:
                raise Lfm2LongHorizonRepairError(
                    "Accepted tranche has no authenticated staged reload."
                )
            accepted_stage = staged
            accepted_reload = reload_holder
            accepted_diagnostic = reload_holder["diagnostic"]
            accepted_candidate = candidate
            break
        # The all-or-nothing rollback restored this exact cached child state.
        working_cache = base_child_cache
        working_rows = base_child_rows

    report_body = {
        "schema": RUN_SCHEMA,
        "status": "accepted" if accepted_stage is not None else "rejected",
        "result_scope": candidate_selection["result_scope"],
        "config": config.as_dict(),
        "candidate_selection": candidate_selection,
        "source_child": source.as_dict(),
        "sealed_envelope": envelope.as_dict(),
        "identities": {
            "original_envelope_sha256": envelope.envelope_sha256,
            "original_parent_cache_key": envelope.parent_cache_key,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "source_child_report_sha256": source.report_file_sha256,
            "source_child_state_sha256": source.source_child_state_sha256,
            "parent_horizon_cache_key": parent_cache.cache_key,
            "parent_horizon_corpus_fingerprint": (
                parent_cache.corpus_fingerprint
            ),
            "parent_horizon_trace_rows_sha256": (
                parent_rows.trace_rows_sha256
            ),
            "pure_anchor_policy_identity": ANCHOR_POLICY_IDENTITY,
            "active_anchor_policy_identity": anchor_plan.policy_identity,
            "anchor_plan_sha256": anchor_plan.plan_sha256,
            "tagging_sha256": tagging.tagging_sha256,
            "composite_evidence_identity": (
                anchor_plan.composite_evidence_identity
            ),
            "candidate_request_sha256": (
                candidate_selection["request_sha256"]
            ),
            "candidate_selection_sha256": (
                candidate_selection["selection_sha256"]
            ),
            "complete_fixed_grid_sha256": (
                candidate_selection["complete_fixed_grid_sha256"]
            ),
        },
        "parent_horizon_cache": {
            "path": str(parent_cache.path),
            "cache_key": parent_cache.cache_key,
            "reused": parent_cache_reused,
            "new_parent_horizon_forwards": (
                0
                if parent_cache_reused
                else sum(
                    2 + 2 * len(row["forced_decode_ids"])
                    for row in batch_records
                )
            ),
            "parent_response_generation_calls": 0,
            "forced_decode_ids_source": (
                "existing-sealed-parent_output_ids"
            ),
            "maximum_decode_steps": MAX_DECODE_STEPS,
        },
        "anchor_plan": anchor_plan.as_dict(),
        "profiles": [profile.as_dict() for profile in profiles],
        "fixed_retry_grid": [
            candidate.as_dict() for candidate in fixed_grid
        ],
        "retry_grid": [candidate.as_dict() for candidate in grid],
        "attempts": attempts,
        "promotion": {
            "binding_case_ids": list(envelope.manual_case_ids),
            "binding_case_count": len(envelope.manual_case_ids),
            "development_case_ids_excluded": list(
                envelope.recovery_case_ids
            ),
            "minimum_semantic_and_nonempty_retention": 0.90,
            "maximum_case_regressions": 1,
            "maximum_new_saturations": 1,
            "maximum_repetition_regressions": 1,
            "full_28_used_for_decision": False,
            "full_28_diagnostic": accepted_diagnostic,
        },
        "training": {
            "training_steps": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "source_child_gradient_audit": source_gradient_audit,
            "final_in_memory_gradient_audit": residual._gradient_audit(child),
        },
        "checkpoint_published": accepted_stage is not None,
        "accepted_candidate": (
            None
            if accepted_candidate is None
            else accepted_candidate.as_dict()
        ),
    }
    if accepted_stage is not None:
        if config.checkpoint_destination.exists():
            raise Lfm2LongHorizonRepairError(
                "Checkpoint destination appeared before atomic publication."
            )
        os.rename(accepted_stage, config.checkpoint_destination)
        report_body["checkpoint"] = {
            **dict(accepted_reload["checkpoint"]),
            "path": str(config.checkpoint_destination),
            "manual_only_fresh_reload_gate": accepted_reload["manual"],
            "unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
            "compression_fraction": EXPECTED_COMPRESSION_FRACTION,
        }
    report = {**report_body, "report_sha256": _json_sha256(report_body)}
    _atomic_write_json(config.report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retry LFM2 layers 6,8,11,13 using sealed 64-step recovery "
            "trajectories and a manual-only promotion gate."
        )
    )
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--source-child", type=Path, required=True)
    parser.add_argument("--source-child-report", type=Path, required=True)
    parser.add_argument("--source-child-report-sha256", required=True)
    parser.add_argument("--cached-parent-outputs", type=Path, required=True)
    parser.add_argument("--cached-parent-outputs-sha256", required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--parent-horizon-cache-root", type=Path, required=True)
    parser.add_argument("--child-horizon-cache-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint-destination", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data-contract-only", action="store_true")
    parser.add_argument(
        "--candidate-indices",
        type=parse_candidate_indices,
        default=FIXED_CANDIDATE_INDICES,
        help=(
            "Sorted unique comma-list subset of 0..5. The default schedules "
            "all six fixed candidates."
        ),
    )
    parser.add_argument("--sample-last-tokens", type=int, default=8)
    parser.add_argument(
        "--max-capture-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--capture-logit-rtol", type=float, default=0.20)
    parser.add_argument("--capture-logit-atol", type=float, default=1.0)
    parser.add_argument("--endpoint-refinement-passes", type=int, default=2)
    parser.add_argument(
        "--max-endpoint-relative-rmse",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--min-endpoint-cosine-similarity",
        type=float,
        default=-0.5,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> RepairRunConfig:
    return RepairRunConfig(
        parent_model=args.parent_model,
        source_child=args.source_child,
        source_child_report=args.source_child_report,
        source_child_report_sha256=args.source_child_report_sha256,
        cached_parent_outputs=args.cached_parent_outputs,
        cached_parent_outputs_sha256=args.cached_parent_outputs_sha256,
        tokenizer_directory=args.tokenizer_directory,
        parent_horizon_cache_root=args.parent_horizon_cache_root,
        child_horizon_cache_root=args.child_horizon_cache_root,
        work_root=args.work_root,
        checkpoint_destination=args.checkpoint_destination,
        report_path=args.report_path,
        device=args.device,
        dtype=args.dtype,
        data_contract_only=args.data_contract_only,
        candidate_indices=args.candidate_indices,
        sample_last_tokens=args.sample_last_tokens,
        max_capture_bytes=args.max_capture_bytes,
        capture_logit_rtol=args.capture_logit_rtol,
        capture_logit_atol=args.capture_logit_atol,
        endpoint_refinement_passes=args.endpoint_refinement_passes,
        max_endpoint_relative_rmse=args.max_endpoint_relative_rmse,
        min_endpoint_cosine_similarity=(
            args.min_endpoint_cosine_similarity
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = execute_repair(_config_from_args(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"accepted", "data_contract_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
