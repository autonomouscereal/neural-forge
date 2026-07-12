"""Evidence-only learned policy for ranking measured surgery candidates.

This module deliberately has no dependency on a target checkpoint or on the
mutable adaptive surgery graph.  It learns only from canonical JSON evidence
records. A policy checkpoint contains only this policy's state, audited
metadata, and hashes; it cannot recover, load, retain, or modify a target
model. The policy ranks measured, pre-decision surgery candidates. It does not
transfer model intelligence itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "neural-forge.surgery-attempt.v1"
FEATURE_VERSION = "summary-stats.v2"
CHECKPOINT_FORMAT = "neural-forge.surgery-meta-policy.v2"
POLICY_PURPOSE = (
    "ranks measured pre-decision surgery candidates; "
    "does not transfer model intelligence"
)
MAX_BATCH_SIZE = 256
MAX_DATASET_RECORDS = 16_384
MAX_DATASET_BYTES = 256 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_FEATURE_VALUES = 64
MAX_HIDDEN_DIM = 256
MAX_SET_CONTEXT = 64
MAX_JSON_DEPTH = 24
MAX_JSON_NODES = 100_000
MAX_STRING_LENGTH = 4_096
MAX_METRICS_PER_GROUP = 128
_METRIC_GROUPS = (
    "parent",
    "incumbent",
    "candidate",
    "behavior",
    "generation",
    "choice",
    "manual",
)
_FEATURE_CHANNELS = (
    ("static", "weight"),
    ("static", "shape"),
    ("static", "role"),
    ("observed", "activation"),
    ("observed", "jvp"),
    ("observed", "dual"),
    ("observed", "output"),
)
_SUMMARY_FEATURE_CHANNELS = {
    ("static", "weight"),
    ("observed", "activation"),
    ("observed", "jvp"),
    ("observed", "dual"),
    ("observed", "output"),
}
_SUMMARY_STAT_KEYS = {
    "abs_max",
    "condition",
    "correlation",
    "cosine",
    "count",
    "effective_rank",
    "energy",
    "entropy",
    "l1",
    "l2",
    "max",
    "mean",
    "min",
    "negative_fraction",
    "overlap",
    "positive_fraction",
    "rms",
    "sensitivity",
    "std",
    "zero_fraction",
}
_RECORD_KEYS = {
    "schema_version",
    "attempt_id",
    "model",
    "features",
    "anchors",
    "action",
    "deployable_byte_savings",
    "metrics",
    "outcome",
    "projection",
    "provenance",
}
_MODEL_KEYS = {
    "family_id",
    "model_id",
    "architecture_id",
    "tokenizer_id",
    "modality",
}
_PROVENANCE_DIGEST_KEYS = {
    "source_attempt_sha256",
    "source_dataset_sha256",
    "parent_snapshot_sha256",
    "eval_contract_sha256",
    "feature_manifest_sha256",
}
_OPTIONAL_PROVENANCE_DIGEST_KEYS = {
    "source_transaction_sha256",
    "proposed_snapshot_sha256",
}
_PROVENANCE_KEYS = (
    _PROVENANCE_DIGEST_KEYS
    | _OPTIONAL_PROVENANCE_DIGEST_KEYS
    | {"feature_stage", "feature_encoding", "record_sha256"}
)
_SAFE_PRE_DECISION_METRIC_GROUPS: tuple[str, ...] = ()


class SurgeryMetaCompilerError(ValueError):
    """Base class for evidence-schema and policy failures."""


class DatasetIntegrityError(SurgeryMetaCompilerError):
    """A record is malformed, non-canonical, or has a mismatched hash."""


class TargetModelReferenceError(SurgeryMetaCompilerError):
    """A caller attempted to put a model object into evidence-only storage."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical representation used for records and reports."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_copy(
    value: Any,
    path: str = "record",
    *,
    _depth: int = 0,
    _budget: list[int] | None = None,
) -> Any:
    """Copy JSON values and reject live objects before they can be retained."""
    if _depth > MAX_JSON_DEPTH:
        raise DatasetIntegrityError(f"{path} exceeds the JSON nesting limit.")
    if _budget is None:
        _budget = [MAX_JSON_NODES]
    _budget[0] -= 1
    if _budget[0] < 0:
        raise DatasetIntegrityError(f"{path} exceeds the JSON node limit.")
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            raise DatasetIntegrityError(f"{path} exceeds the string length limit.")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetIntegrityError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DatasetIntegrityError(f"{path} has a non-string key.")
            if len(key) > MAX_STRING_LENGTH:
                raise DatasetIntegrityError(f"{path} has an overlong key.")
            copied[key] = _json_copy(
                item,
                f"{path}.{key}",
                _depth=_depth + 1,
                _budget=_budget,
            )
        return copied
    if isinstance(value, (list, tuple)):
        return [
            _json_copy(
                item,
                f"{path}[]",
                _depth=_depth + 1,
                _budget=_budget,
            )
            for item in value
        ]
    raise TargetModelReferenceError(
        f"{path} must contain JSON evidence only, not {type(value).__name__}."
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetIntegrityError(f"{name} must be an object.")
    return _json_copy(value, name)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetIntegrityError(f"{name} must be a non-empty string.")
    return value


def _numeric_values(
    value: Any,
    name: str,
    *,
    max_values: int = MAX_FEATURE_VALUES,
) -> list[float]:
    result: list[float] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, bool):
            raise DatasetIntegrityError(f"{name} must be numeric evidence.")
        if isinstance(item, (int, float)):
            number = float(item)
            if not math.isfinite(number):
                raise DatasetIntegrityError(f"{name} must be finite.")
            result.append(number)
            if len(result) > max_values:
                raise DatasetIntegrityError(
                    f"{name} exceeds the {max_values}-value summary limit."
                )
            continue
        if isinstance(item, (list, tuple)):
            pending.extend(reversed(item))
            continue
        raise DatasetIntegrityError(f"{name} must be a number or list of numbers.")
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DatasetIntegrityError(
            f"{name} keys do not match the schema; missing={missing}, extra={extra}."
        )


def _bounded_count(values: Sequence[Any], name: str = "records") -> None:
    if len(values) > MAX_DATASET_RECORDS:
        raise SurgeryMetaCompilerError(
            f"{name} exceeds the {MAX_DATASET_RECORDS}-record history limit."
        )


def _get_mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_copy(value)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _json_copy(as_dict())
    raise DatasetIntegrityError("Attempt/outcome input must be a mapping or expose as_dict().")


def _default_record() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": "",
        "model": {
            "family_id": "",
            "model_id": "",
            "architecture_id": "unknown",
            "tokenizer_id": "unknown",
            "modality": "unknown",
        },
        "features": {
            "static": {"weight": {}, "shape": [], "role": "unknown"},
            "observed": {"activation": {}, "jvp": {}, "dual": {}, "output": {}},
        },
        "anchors": {"mode": "none", "count": 0, "penalty": 0.0},
        "action": {"kind": "unknown", "source_count": 0},
        "deployable_byte_savings": 0,
        "metrics": {group: {} for group in _METRIC_GROUPS},
        "outcome": {"accepted": False, "rolled_back": True},
        "projection": {"coefficient": 0.0, "risk": 0.0},
        "provenance": {
            "feature_stage": "pre_decision",
            "feature_encoding": FEATURE_VERSION,
        },
    }


def canonicalize_attempt(record: Mapping[str, Any], *, verify_hash: bool = True) -> dict[str, Any]:
    """Validate a schema record and add or verify its canonical record hash."""
    item = _json_copy(record)
    _require_exact_keys(item, _RECORD_KEYS, "record")
    if item.get("schema_version") != SCHEMA_VERSION:
        raise DatasetIntegrityError(f"schema_version must equal {SCHEMA_VERSION!r}.")
    _nonempty_string(item.get("attempt_id"), "attempt_id")
    model = _mapping(item.get("model"), "model")
    _require_exact_keys(model, _MODEL_KEYS, "model")
    for name in _MODEL_KEYS:
        _nonempty_string(model.get(name), f"model.{name}")
    features = _mapping(item.get("features"), "features")
    _require_exact_keys(features, {"static", "observed"}, "features")
    expected_feature_keys: dict[str, set[str]] = {"static": set(), "observed": set()}
    for group, key in _FEATURE_CHANNELS:
        expected_feature_keys[group].add(key)
    for group, expected in expected_feature_keys.items():
        channel_group = _mapping(features.get(group), f"features.{group}")
        _require_exact_keys(channel_group, expected, f"features.{group}")
        features[group] = channel_group
    for group, key in _FEATURE_CHANNELS:
        channel = features[group]
        raw = channel.get(key, [])
        if key == "role":
            _nonempty_string(raw, f"features.{group}.{key}")
            continue
        if (group, key) in _SUMMARY_FEATURE_CHANNELS:
            summary = _mapping(raw, f"features.{group}.{key}")
            unknown_statistics = set(summary) - _SUMMARY_STAT_KEYS
            if unknown_statistics:
                raise DatasetIntegrityError(
                    f"features.{group}.{key} contains unsupported summary statistics: "
                    f"{sorted(unknown_statistics)}."
                )
            for statistic, value in summary.items():
                _numeric_values(
                    value,
                    f"features.{group}.{key}.{statistic}",
                    max_values=1,
                )
            channel[key] = summary
            continue
        _numeric_values(
            raw,
            f"features.{group}.{key}",
            max_values=MAX_FEATURE_VALUES,
        )
    anchors = _mapping(item.get("anchors"), "anchors")
    _require_exact_keys(anchors, {"mode", "count", "penalty"}, "anchors")
    _nonempty_string(anchors.get("mode", "none"), "anchors.mode")
    if (
        not isinstance(anchors.get("count"), int)
        or isinstance(anchors["count"], bool)
        or anchors["count"] < 0
    ):
        raise DatasetIntegrityError("anchors.count must be a non-negative integer.")
    _numeric_values(anchors.get("penalty", 0.0), "anchors.penalty", max_values=1)
    action = _mapping(item.get("action"), "action")
    _require_exact_keys(action, {"kind", "source_count"}, "action")
    _nonempty_string(action.get("kind"), "action.kind")
    if (
        not isinstance(action.get("source_count"), int)
        or isinstance(action["source_count"], bool)
        or action["source_count"] < 0
    ):
        raise DatasetIntegrityError("action.source_count must be a non-negative integer.")
    if not isinstance(item.get("deployable_byte_savings"), int) or isinstance(
        item["deployable_byte_savings"], bool
    ):
        raise DatasetIntegrityError("deployable_byte_savings must be an integer.")
    metrics = _mapping(item.get("metrics"), "metrics")
    _require_exact_keys(metrics, set(_METRIC_GROUPS), "metrics")
    for group in _METRIC_GROUPS:
        values = _mapping(metrics.get(group, {}), f"metrics.{group}")
        if len(values) > MAX_METRICS_PER_GROUP:
            raise DatasetIntegrityError(
                f"metrics.{group} exceeds the metric-count limit."
            )
        for key, value in values.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (str, int, float)) and value is not None:
                raise DatasetIntegrityError(f"metrics.{group}.{key} must be scalar.")
            if isinstance(value, float) and not math.isfinite(value):
                raise DatasetIntegrityError(f"metrics.{group}.{key} must be finite.")
        metrics[group] = values
    outcome = _mapping(item.get("outcome"), "outcome")
    _require_exact_keys(outcome, {"accepted", "rolled_back"}, "outcome")
    if not isinstance(outcome.get("accepted"), bool) or not isinstance(
        outcome.get("rolled_back"), bool
    ):
        raise DatasetIntegrityError("outcome.accepted and outcome.rolled_back must be booleans.")
    if outcome["rolled_back"] == outcome["accepted"]:
        raise DatasetIntegrityError("outcome.rolled_back must be the inverse of accepted.")
    projection = _mapping(item.get("projection"), "projection")
    _require_exact_keys(projection, {"coefficient", "risk"}, "projection")
    coefficient = _numeric_values(
        projection.get("coefficient", 0.0),
        "projection.coefficient",
        max_values=1,
    )[0]
    risk = _numeric_values(
        projection.get("risk", 0.0), "projection.risk", max_values=1
    )[0]
    if not 0.0 <= risk <= 1.0:
        raise DatasetIntegrityError("projection.risk must be between zero and one.")
    projection["coefficient"] = coefficient
    projection["risk"] = risk
    damage = metrics["parent"].get("damage")
    if (
        isinstance(damage, bool)
        or not isinstance(damage, (int, float))
        or not math.isfinite(float(damage))
        or float(damage) < 0.0
    ):
        raise DatasetIntegrityError(
            "metrics.parent.damage must be a finite, non-negative measured label."
        )
    provenance = _mapping(item.get("provenance"), "provenance")
    missing_provenance = _PROVENANCE_DIGEST_KEYS - set(provenance)
    extra_provenance = set(provenance) - _PROVENANCE_KEYS
    if missing_provenance or extra_provenance:
        raise DatasetIntegrityError(
            "provenance keys do not match the schema; "
            f"missing={sorted(missing_provenance)}, "
            f"extra={sorted(extra_provenance)}."
        )
    for key in _PROVENANCE_DIGEST_KEYS | _OPTIONAL_PROVENANCE_DIGEST_KEYS:
        if key in provenance and not _is_digest(provenance[key]):
            raise DatasetIntegrityError(f"provenance.{key} must be a SHA-256 digest.")
    if provenance.get("feature_stage") != "pre_decision":
        raise DatasetIntegrityError(
            "provenance.feature_stage must be 'pre_decision' to prevent outcome leakage."
        )
    if provenance.get("feature_encoding") != FEATURE_VERSION:
        raise DatasetIntegrityError(
            f"provenance.feature_encoding must equal {FEATURE_VERSION!r}."
        )

    item["model"] = model
    item["features"] = features
    item["anchors"] = anchors
    item["action"] = action
    item["metrics"] = metrics
    item["outcome"] = outcome
    item["projection"] = projection
    item["provenance"] = provenance
    supplied_hash = provenance.pop("record_sha256", None)
    expected_hash = canonical_sha256(item)
    if verify_hash and supplied_hash is not None and supplied_hash != expected_hash:
        raise DatasetIntegrityError("provenance.record_sha256 does not match record contents.")
    provenance["record_sha256"] = expected_hash
    return item


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> str:
    """Write integrity-protected, canonical JSONL and return its file hash."""
    payload = bytearray()
    count = 0
    for record in records:
        count += 1
        if count > MAX_DATASET_RECORDS:
            raise SurgeryMetaCompilerError(
                f"records exceeds the {MAX_DATASET_RECORDS}-record history limit."
            )
        payload.extend(canonical_json_bytes(canonicalize_attempt(record)))
        payload.extend(b"\n")
        if len(payload) > MAX_DATASET_BYTES:
            raise SurgeryMetaCompilerError(
                f"Canonical dataset exceeds the {MAX_DATASET_BYTES}-byte limit."
            )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read a canonical JSONL dataset, rejecting whitespace and hash tampering."""
    source = Path(path)
    if source.stat().st_size > MAX_DATASET_BYTES:
        raise DatasetIntegrityError(
            f"Dataset exceeds the {MAX_DATASET_BYTES}-byte input limit."
        )
    result: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="ascii").splitlines(), 1
    ):
        if not raw_line:
            continue
        if len(result) >= MAX_DATASET_RECORDS:
            raise DatasetIntegrityError(
                f"Dataset exceeds the {MAX_DATASET_RECORDS}-record history limit."
            )
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise DatasetIntegrityError(f"Invalid JSON on line {line_number}.") from error
        canonical = canonicalize_attempt(item, verify_hash=True)
        if canonical_json_bytes(canonical).decode("ascii") != raw_line:
            raise DatasetIntegrityError(f"Line {line_number} is not canonical JSON.")
        result.append(canonical)
    return tuple(result)


def bootstrap_from_attempt(
    attempt: Any,
    outcome: Any | None = None,
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy adaptive-surgery-style attempt/outcome evidence without importing it.

    ``context`` supplies family-neutral model identity and measured feature
    channels.  Attribute/mapping lookup makes this work with evolving graph
    records while ensuring that only JSON evidence crosses the boundary.
    """
    source = _as_plain_mapping(attempt)
    result = _default_record()
    supplied = _json_copy(context or {}, "context")
    allowed_context = {
        "attempt_id",
        "model",
        "features",
        "anchors",
        "action",
        "metrics",
        "projection",
        "provenance",
    }
    unexpected_context = set(supplied) - allowed_context
    if unexpected_context:
        raise DatasetIntegrityError(
            f"context contains unsupported keys: {sorted(unexpected_context)}."
        )
    for key in ("model", "features", "anchors", "action", "metrics", "projection"):
        if key in supplied:
            if not isinstance(supplied[key], Mapping):
                raise DatasetIntegrityError(f"context.{key} must be an object.")
            result[key].update(supplied[key])
    if "provenance" in supplied:
        if not isinstance(supplied["provenance"], Mapping):
            raise DatasetIntegrityError("context.provenance must be an object.")
        result["provenance"].update(supplied["provenance"])
    result["attempt_id"] = str(supplied.get("attempt_id") or source.get("attempt_sha256") or source.get("attempt_index") or "attempt")
    accepted = bool(source.get("accepted", _get_mapping_value(outcome, "accepted", False)))
    result["outcome"] = {"accepted": accepted, "rolled_back": not accepted}
    decision = source.get("decision", {})
    decision_map = _as_plain_mapping(decision) if decision else {}
    metrics = decision_map.get("metrics", {})
    if isinstance(metrics, Mapping):
        result["metrics"]["manual"].update(_json_copy(metrics, "decision.metrics"))
    actions = source.get("actions", ())
    if actions:
        first = actions[0]
        first_map = _as_plain_mapping(first) if not isinstance(first, Mapping) else _json_copy(first)
        result["action"].update(
            {
                "kind": str(first_map.get("kind", result["action"]["kind"])),
                "source_count": len(first_map.get("sources", ())),
            }
        )
        details = first_map.get("details", {})
        if isinstance(details, Mapping):
            result["metrics"]["candidate"].update(_json_copy(details, "action.details"))
    for source_key, output_key in (
        ("attempt_sha256", "source_attempt_sha256"),
        ("transaction_sha256", "source_transaction_sha256"),
        ("parent_snapshot_sha256", "parent_snapshot_sha256"),
        ("proposed_snapshot_sha256", "proposed_snapshot_sha256"),
    ):
        value = source.get(source_key)
        if _is_digest(value):
            result["provenance"][output_key] = value
    return canonicalize_attempt(result, verify_hash=False)


def family_held_out_split(
    records: Sequence[Mapping[str, Any]], held_out_family: str
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Return train/holdout partitions isolated by family, model, and eval contract."""
    held_out = _nonempty_string(held_out_family, "held_out_family")
    _bounded_count(records)
    canonical = tuple(canonicalize_attempt(record) for record in records)
    train = tuple(record for record in canonical if record["model"]["family_id"] != held_out)
    evaluation = tuple(record for record in canonical if record["model"]["family_id"] == held_out)
    if not train or not evaluation:
        raise SurgeryMetaCompilerError("Family-held-out split requires non-empty train and holdout sets.")
    attempt_ids = [record["attempt_id"] for record in canonical]
    source_attempts = [
        record["provenance"]["source_attempt_sha256"] for record in canonical
    ]
    record_hashes = [record["provenance"]["record_sha256"] for record in canonical]
    for values, name in (
        (attempt_ids, "attempt_id"),
        (source_attempts, "source_attempt_sha256"),
        (record_hashes, "record_sha256"),
    ):
        if len(values) != len(set(values)):
            raise DatasetIntegrityError(f"Duplicate {name} values are not allowed.")
    train_models = {record["model"]["model_id"] for record in train}
    held_out_models = {record["model"]["model_id"] for record in evaluation}
    model_overlap = train_models & held_out_models
    train_contracts = {
        record["provenance"]["eval_contract_sha256"] for record in train
    }
    held_out_contracts = {
        record["provenance"]["eval_contract_sha256"] for record in evaluation
    }
    contract_overlap = train_contracts & held_out_contracts
    if model_overlap or contract_overlap:
        raise DatasetIntegrityError(
            "Strict holdout leakage detected; "
            f"model_ids={sorted(model_overlap)}, "
            f"eval_contract_sha256={sorted(contract_overlap)}."
        )
    return train, evaluation


def _stable_scalar(value: Any) -> float:
    encoded = canonical_json_bytes(_json_copy(value))
    integer = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return (integer / float((1 << 63) - 1)) - 1.0


def _summary(value: Any) -> list[float]:
    if isinstance(value, str):
        return [_stable_scalar(value), 0.0, 0.0, 0.0, 1.0]
    if isinstance(value, Mapping):
        value = [value[key] for key in sorted(value)]
    values = np.asarray(_numeric_values(value, "feature"), dtype=np.float64)
    if values.size == 0:
        return [0.0] * 5
    return [
        float(values.mean()),
        float(values.std()),
        float(np.abs(values).max()),
        float(np.sqrt(np.square(values).sum())),
        float(math.log1p(values.size)),
    ]


class AttemptVectorizer:
    """Fixed-width summaries of bounded, pre-decision candidate evidence.

    Model identity, eval identity, outcomes, projection labels, and all metric
    groups are intentionally excluded. They remain in canonical records only
    for holdout enforcement, supervision, and audited reporting.
    """

    version = FEATURE_VERSION

    @property
    def dimension(self) -> int:
        return (len(_FEATURE_CHANNELS) * 5) + 20 + (len(_METRIC_GROUPS) * 3)

    def vectorize(self, record: Mapping[str, Any]) -> np.ndarray:
        item = canonicalize_attempt(record)
        values: list[float] = []
        for group, channel in _FEATURE_CHANNELS:
            values.extend(_summary(item["features"][group].get(channel, [])))
        # Four reserved identity slots remain zero so v2 cannot learn a family,
        # model, tokenizer, or eval-contract shortcut.
        values.extend([0.0] * 4)
        anchors = item["anchors"]
        action = item["action"]
        values.extend(
            [
                _stable_scalar(anchors.get("mode", "none")),
                float(_numeric_values(anchors.get("count", 0), "anchors.count")[0]),
                float(_numeric_values(anchors.get("penalty", 0.0), "anchors.penalty")[0]),
                _stable_scalar(action["kind"]),
                float(_numeric_values(action.get("source_count", 0), "action.source_count")[0]),
                math.copysign(math.log1p(abs(float(item["deployable_byte_savings"]))), float(item["deployable_byte_savings"]) or 1.0),
                0.0,
                0.0,
            ]
        )
        values.extend([0.0] * 8)
        for group in _METRIC_GROUPS:
            if group in _SAFE_PRE_DECISION_METRIC_GROUPS:
                numeric = [
                    float(value)
                    for value in item["metrics"].get(group, {}).values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                values.extend(_summary(numeric)[:3])
            else:
                values.extend([0.0] * 3)
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (self.dimension,) or not np.isfinite(array).all():
            raise DatasetIntegrityError("Feature vector construction produced invalid values.")
        return array

    def pair_features(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        _bounded_count(records)
        if not records:
            return np.empty((0, 2, self.dimension), dtype=np.float32)
        rows = np.stack([self.vectorize(record) for record in records])
        # Candidate evidence is neutral about left/right ordering.  Duplicate
        # streams are intentionally symmetric; a pair policy can later accept
        # explicit participant channels without changing its contract.
        return np.stack((rows, rows), axis=1)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised where torch absent
        raise SurgeryMetaCompilerError("PyTorch is required for policy training.") from error
    return torch


@contextmanager
def _isolated_torch_cpu_rng(torch: Any, seed: int) -> Any:
    """Temporarily seed only Torch's CPU generator and restore it exactly."""
    state = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)


def _make_policy(input_dim: int, hidden_dim: int) -> Any:
    torch = _torch()

    class CandidateEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Tanh(),
            )

        def forward(self, value: Any) -> Any:
            return self.network(value)

    class SetPairPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.candidate_encoder = CandidateEncoder()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 4, hidden_dim), torch.nn.Tanh()
            )
            self.acceptance = torch.nn.Linear(hidden_dim, 1)
            self.damage = torch.nn.Linear(hidden_dim, 1)
            self.byte_utility = torch.nn.Linear(hidden_dim, 1)
            self.projection = torch.nn.Linear(hidden_dim, 2)

        def forward(self, pair_features: Any, set_features: Any | None = None) -> dict[str, Any]:
            if pair_features.ndim != 3 or pair_features.shape[1:] != (2, input_dim):
                raise ValueError("pair_features must have shape [batch, 2, input_dim].")
            encoded = self.candidate_encoder(pair_features)
            left, right = encoded[:, 0], encoded[:, 1]
            pair = torch.cat((left + right, torch.abs(left - right), left * right), dim=-1)
            if set_features is None:
                context = (left + right) * 0.5
            else:
                if set_features.ndim != 4 or set_features.shape[2:] != (2, input_dim):
                    raise ValueError("set_features must have shape [batch, set, 2, input_dim].")
                context = self.candidate_encoder(set_features).mean(dim=(1, 2))
            hidden = self.trunk(torch.cat((pair, context), dim=-1))
            projection = self.projection(hidden)
            return {
                "acceptance_logit": self.acceptance(hidden).squeeze(-1),
                "damage": torch.nn.functional.softplus(self.damage(hidden).squeeze(-1)),
                "byte_utility": self.byte_utility(hidden).squeeze(-1),
                "projection_coefficient": projection[:, 0],
                "projection_risk_logit": projection[:, 1],
            }

    return SetPairPolicy()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 20260710
    epochs: int = 24
    batch_size: int = 32
    hidden_dim: int = 32
    learning_rate: float = 0.02

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise SurgeryMetaCompilerError("epochs must be positive.")
        if not 1 <= self.hidden_dim <= MAX_HIDDEN_DIM:
            raise SurgeryMetaCompilerError(
                f"hidden_dim must be between 1 and {MAX_HIDDEN_DIM}."
            )
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise SurgeryMetaCompilerError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise SurgeryMetaCompilerError("learning_rate must be finite and positive.")


@dataclass(frozen=True, slots=True)
class TrainingReport:
    held_out_family: str
    held_out_model_count: int
    held_out_models_sha256: str
    held_out_eval_contract_count: int
    held_out_eval_contracts_sha256: str
    train_records_sha256: str
    holdout_records_sha256: str
    train_count: int
    holdout_count: int
    epochs: int
    batch_size: int
    seed: int
    train_loss: float
    holdout_loss: float
    temperature: float
    feature_version: str
    recovery_status: str
    report_sha256: str

    def __post_init__(self) -> None:
        _nonempty_string(self.held_out_family, "report.held_out_family")
        for name in (
            "held_out_models_sha256",
            "held_out_eval_contracts_sha256",
            "train_records_sha256",
            "holdout_records_sha256",
            "report_sha256",
        ):
            if not _is_digest(getattr(self, name)):
                raise DatasetIntegrityError(f"report.{name} must be a SHA-256 digest.")
        for name in (
            "held_out_model_count",
            "held_out_eval_contract_count",
            "train_count",
            "holdout_count",
            "epochs",
            "batch_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise DatasetIntegrityError(f"report.{name} must be a positive integer.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise DatasetIntegrityError("report.seed must be an integer.")
        for name in ("train_loss", "holdout_loss", "temperature"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DatasetIntegrityError(f"report.{name} must be finite.")
        if self.temperature <= 0.0:
            raise DatasetIntegrityError("report.temperature must be positive.")
        if self.feature_version != FEATURE_VERSION:
            raise DatasetIntegrityError("report.feature_version is unsupported.")
        if self.recovery_status != (
            "not_attempted: policy ranks candidates only and does not transfer intelligence"
        ):
            raise DatasetIntegrityError("report.recovery_status is unsupported.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RankedAttempt:
    attempt_id: str
    acceptance_probability: float
    parent_relative_damage: float
    byte_utility: float
    projection_coefficient: float
    projection_risk: float
    uncertainty: float
    score: float


class SurgeryMetaPolicy:
    """Ranks measured surgery candidates; it does not transfer intelligence."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        temperature: float = 1.0,
        initialization_seed: int = 0,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise SurgeryMetaCompilerError("temperature must be finite and positive.")
        if not 1 <= hidden_dim <= MAX_HIDDEN_DIM:
            raise SurgeryMetaCompilerError(
                f"hidden_dim must be between 1 and {MAX_HIDDEN_DIM}."
            )
        self.vectorizer = AttemptVectorizer()
        if input_dim != self.vectorizer.dimension:
            raise SurgeryMetaCompilerError("input_dim does not match the current vectorizer.")
        torch = _torch()
        with _isolated_torch_cpu_rng(torch, initialization_seed):
            self.model = _make_policy(input_dim, hidden_dim)
        self.temperature = float(temperature)

    @property
    def recovery_status(self) -> str:
        return "not_attempted: policy ranks candidates only and does not transfer intelligence"

    def rank(self, records: Sequence[Mapping[str, Any]], *, max_candidates: int = MAX_BATCH_SIZE) -> tuple[RankedAttempt, ...]:
        if not 1 <= max_candidates <= MAX_BATCH_SIZE:
            raise SurgeryMetaCompilerError(f"max_candidates must be between 1 and {MAX_BATCH_SIZE}.")
        torch = _torch()
        _bounded_count(records)
        canonical = tuple(canonicalize_attempt(record) for record in records)
        if not canonical:
            return ()
        context_pairs = torch.from_numpy(
            self.vectorizer.pair_features(canonical[:MAX_SET_CONTEXT])
        )
        ranked: list[RankedAttempt] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(canonical), max_candidates):
                chunk = canonical[start : start + max_candidates]
                pairs = torch.from_numpy(self.vectorizer.pair_features(chunk))
                set_features = context_pairs.unsqueeze(0).expand(
                    len(chunk), -1, -1, -1
                )
                output = self.model(pairs, set_features)
                probability = torch.sigmoid(output["acceptance_logit"] / self.temperature).cpu().numpy()
                damage = output["damage"].cpu().numpy()
                utility = output["byte_utility"].cpu().numpy()
                coefficient = output["projection_coefficient"].cpu().numpy()
                risk = torch.sigmoid(output["projection_risk_logit"]).cpu().numpy()
                for index, record in enumerate(chunk):
                    p = float(probability[index])
                    uncertainty = float(-(p * math.log(max(p, 1e-12)) + (1.0 - p) * math.log(max(1.0 - p, 1e-12))))
                    score = p + float(utility[index]) - float(damage[index]) - float(risk[index])
                    ranked.append(RankedAttempt(record["attempt_id"], p, float(damage[index]), float(utility[index]), float(coefficient[index]), float(risk[index]), uncertainty, score))
        return tuple(sorted(ranked, key=lambda item: (-item.score, item.attempt_id)))


def _labels(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accepted = np.asarray([float(record["outcome"]["accepted"]) for record in records], dtype=np.float32)
    damage = np.asarray(
        [float(record["metrics"]["parent"]["damage"]) for record in records],
        dtype=np.float32,
    )
    utility = np.asarray([math.copysign(math.log1p(abs(float(record["deployable_byte_savings"]))), float(record["deployable_byte_savings"]) or 1.0) for record in records], dtype=np.float32)
    coefficient = np.asarray([float(record["projection"].get("coefficient", 0.0)) for record in records], dtype=np.float32)
    risk = np.asarray([float(record["projection"].get("risk", 0.0)) for record in records], dtype=np.float32)
    return accepted, damage, utility, coefficient, risk


def _loss(output: Mapping[str, Any], targets: tuple[Any, Any, Any, Any, Any], torch: Any) -> Any:
    accepted, damage, utility, coefficient, risk = targets
    return (
        torch.nn.functional.binary_cross_entropy_with_logits(output["acceptance_logit"], accepted)
        + torch.nn.functional.smooth_l1_loss(output["damage"], damage)
        + torch.nn.functional.smooth_l1_loss(output["byte_utility"], utility)
        + torch.nn.functional.smooth_l1_loss(output["projection_coefficient"], coefficient)
        + torch.nn.functional.binary_cross_entropy_with_logits(output["projection_risk_logit"], risk.clamp(0.0, 1.0))
    )


def _temperature_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.linspace(0.5, 3.0, 101)
    losses = []
    for temperature in candidates:
        scaled = np.clip(logits / temperature, -80.0, 80.0)
        probability = 1.0 / (1.0 + np.exp(-scaled))
        losses.append(float(-(labels * np.log(np.maximum(probability, 1e-12)) + (1.0 - labels) * np.log(np.maximum(1.0 - probability, 1e-12))).mean()))
    return float(candidates[int(np.argmin(losses))])


def _record_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        sorted(record["provenance"]["record_sha256"] for record in records)
    )


def _identity_set_sha256(values: Iterable[str]) -> str:
    return canonical_sha256(sorted(set(values)))


def _evaluate_policy(
    policy: SurgeryMetaPolicy,
    pair_features: np.ndarray,
    labels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    context_pairs: Any,
    *,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    torch = _torch()
    total_loss = 0.0
    logits: list[np.ndarray] = []
    policy.model.eval()
    with torch.no_grad():
        for start in range(0, len(pair_features), batch_size):
            end = min(start + batch_size, len(pair_features))
            pairs = torch.from_numpy(pair_features[start:end])
            context = context_pairs.unsqueeze(0).expand(
                end - start, -1, -1, -1
            )
            targets = tuple(torch.from_numpy(value[start:end]) for value in labels)
            output = policy.model(pairs, context)
            total_loss += float(_loss(output, targets, torch).cpu()) * (end - start)
            logits.append(output["acceptance_logit"].cpu().numpy())
    return total_loss / len(pair_features), np.concatenate(logits)


def train_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    held_out_family: str,
    config: TrainingConfig = TrainingConfig(),
) -> tuple[SurgeryMetaPolicy, TrainingReport]:
    """Train a candidate ranker without retaining records or changing global RNG state."""
    torch = _torch()
    train, holdout = family_held_out_split(records, held_out_family)
    policy = SurgeryMetaPolicy(
        input_dim=AttemptVectorizer().dimension,
        hidden_dim=config.hidden_dim,
        initialization_seed=config.seed,
    )
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)
    ordered = tuple(sorted(train, key=lambda record: record["provenance"]["record_sha256"]))
    train_pairs = policy.vectorizer.pair_features(ordered)
    train_labels = _labels(ordered)
    context_pairs = torch.from_numpy(train_pairs[:MAX_SET_CONTEXT])
    policy.model.train()
    for _epoch in range(config.epochs):
        for start in range(0, len(ordered), config.batch_size):
            end = min(start + config.batch_size, len(ordered))
            pairs = torch.from_numpy(train_pairs[start:end])
            context = context_pairs.unsqueeze(0).expand(
                end - start, -1, -1, -1
            )
            targets = tuple(torch.from_numpy(value[start:end]) for value in train_labels)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(policy.model(pairs, context), targets, torch)
            loss.backward()
            optimizer.step()
    train_loss, calibration_logits = _evaluate_policy(
        policy,
        train_pairs,
        train_labels,
        context_pairs,
        batch_size=config.batch_size,
    )
    holdout_pairs = policy.vectorizer.pair_features(holdout)
    holdout_loss, _holdout_logits = _evaluate_policy(
        policy,
        holdout_pairs,
        _labels(holdout),
        context_pairs,
        batch_size=config.batch_size,
    )
    # Fit calibration only on training families.  The held-out family remains
    # a reporting partition rather than a source of learned policy parameters.
    policy.temperature = _temperature_from_logits(calibration_logits, train_labels[0])
    held_out_models = {record["model"]["model_id"] for record in holdout}
    held_out_contracts = {
        record["provenance"]["eval_contract_sha256"] for record in holdout
    }
    report_payload = {
        "held_out_family": held_out_family,
        "held_out_model_count": len(held_out_models),
        "held_out_models_sha256": _identity_set_sha256(held_out_models),
        "held_out_eval_contract_count": len(held_out_contracts),
        "held_out_eval_contracts_sha256": _identity_set_sha256(
            held_out_contracts
        ),
        "train_records_sha256": _record_set_sha256(train),
        "holdout_records_sha256": _record_set_sha256(holdout),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "train_loss": train_loss,
        "holdout_loss": holdout_loss,
        "temperature": policy.temperature,
        "feature_version": FEATURE_VERSION,
        "recovery_status": policy.recovery_status,
    }
    report = TrainingReport(**report_payload, report_sha256=canonical_sha256(report_payload))
    return policy, report


def _expected_policy_shapes(
    input_dim: int, hidden_dim: int
) -> dict[str, tuple[int, ...]]:
    return {
        "candidate_encoder.network.0.weight": (hidden_dim, input_dim),
        "candidate_encoder.network.0.bias": (hidden_dim,),
        "candidate_encoder.network.2.weight": (hidden_dim, hidden_dim),
        "candidate_encoder.network.2.bias": (hidden_dim,),
        "trunk.0.weight": (hidden_dim, hidden_dim * 4),
        "trunk.0.bias": (hidden_dim,),
        "acceptance.weight": (1, hidden_dim),
        "acceptance.bias": (1,),
        "damage.weight": (1, hidden_dim),
        "damage.bias": (1,),
        "byte_utility.weight": (1, hidden_dim),
        "byte_utility.bias": (1,),
        "projection.weight": (2, hidden_dim),
        "projection.bias": (2,),
    }


def _validate_policy_state(
    state: Any, *, input_dim: int, hidden_dim: int
) -> dict[str, Any]:
    torch = _torch()
    if not isinstance(state, Mapping):
        raise DatasetIntegrityError("checkpoint.state_dict must be a mapping.")
    expected = _expected_policy_shapes(input_dim, hidden_dim)
    if set(state) != set(expected):
        raise DatasetIntegrityError(
            "Checkpoint contains missing, extra, or target-model tensor keys."
        )
    validated: dict[str, Any] = {}
    for name, shape in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise DatasetIntegrityError(f"checkpoint.state_dict.{name} is not a tensor.")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise DatasetIntegrityError(
                f"checkpoint.state_dict.{name} must be a CPU float32 tensor."
            )
        if tuple(tensor.shape) != shape:
            raise DatasetIntegrityError(
                f"checkpoint.state_dict.{name} has an invalid shape."
            )
        if not torch.isfinite(tensor).all().item():
            raise DatasetIntegrityError(
                f"checkpoint.state_dict.{name} contains non-finite values."
            )
        validated[name] = tensor.detach().contiguous().clone()
    return validated


def _policy_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_report(value: Any) -> TrainingReport:
    report_payload = _mapping(value, "checkpoint.report")
    expected = set(TrainingReport.__dataclass_fields__)
    _require_exact_keys(report_payload, expected, "checkpoint.report")
    supplied = report_payload.pop("report_sha256")
    if supplied != canonical_sha256(report_payload):
        raise DatasetIntegrityError("Checkpoint report hash does not match.")
    return TrainingReport(**report_payload, report_sha256=supplied)


def save_policy_checkpoint(
    policy: SurgeryMetaPolicy, report: TrainingReport, path: str | Path
) -> str:
    """Atomically save a policy-only tensor whitelist and audited metadata."""
    torch = _torch()
    if not isinstance(policy, SurgeryMetaPolicy) or not isinstance(
        report, TrainingReport
    ):
        raise SurgeryMetaCompilerError(
            "save_policy_checkpoint accepts only SurgeryMetaPolicy and TrainingReport."
        )
    if report.temperature != policy.temperature or report.feature_version != FEATURE_VERSION:
        raise DatasetIntegrityError("Policy and training report metadata do not match.")
    report_payload = report.as_dict()
    report_hash = report_payload.pop("report_sha256")
    if report_hash != canonical_sha256(report_payload):
        raise DatasetIntegrityError("Training report hash does not match report contents.")
    input_dim = policy.vectorizer.dimension
    hidden_dim = policy.model.hidden_dim
    state = _validate_policy_state(
        {
            name: value.detach().cpu()
            for name, value in policy.model.state_dict().items()
        },
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "purpose": POLICY_PURPOSE,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "temperature": policy.temperature,
        "feature_version": FEATURE_VERSION,
        "report": report.as_dict(),
        "state_dict": state,
        "state_sha256": _policy_state_sha256(state),
        "recovery_status": policy.recovery_status,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        if temporary.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise DatasetIntegrityError(
                f"Policy checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte limit."
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(destination)


def load_policy_checkpoint(
    path: str | Path,
) -> tuple[SurgeryMetaPolicy, TrainingReport, str]:
    """Safely load a policy-only checkpoint or fail closed without pickle fallback."""
    torch = _torch()
    source = Path(path)
    if not source.is_file():
        raise DatasetIntegrityError("Policy checkpoint is not a regular file.")
    if source.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise DatasetIntegrityError(
            f"Policy checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte limit."
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise DatasetIntegrityError(
            "This Torch version cannot guarantee weights-only checkpoint loading."
        ) from error
    except Exception as error:
        raise DatasetIntegrityError(
            "Policy checkpoint failed safe weights-only loading."
        ) from error
    if not isinstance(payload, Mapping):
        raise DatasetIntegrityError("Policy checkpoint payload must be a mapping.")
    expected_payload_keys = {
        "format",
        "purpose",
        "input_dim",
        "hidden_dim",
        "temperature",
        "feature_version",
        "report",
        "state_dict",
        "state_sha256",
        "recovery_status",
    }
    if set(payload) != expected_payload_keys:
        raise DatasetIntegrityError(
            "Policy checkpoint contains unsupported metadata or retained evidence."
        )
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise DatasetIntegrityError("Unsupported policy checkpoint format.")
    if payload.get("purpose") != POLICY_PURPOSE:
        raise DatasetIntegrityError("Unsupported policy checkpoint purpose.")
    if payload.get("feature_version") != FEATURE_VERSION:
        raise DatasetIntegrityError("Unsupported checkpoint feature version.")
    if payload.get("recovery_status") != (
        "not_attempted: policy ranks candidates only and does not transfer intelligence"
    ):
        raise DatasetIntegrityError("Unsupported checkpoint recovery claim.")
    input_dim = payload.get("input_dim")
    hidden_dim = payload.get("hidden_dim")
    temperature = payload.get("temperature")
    if (
        not isinstance(input_dim, int)
        or isinstance(input_dim, bool)
        or input_dim != AttemptVectorizer().dimension
    ):
        raise DatasetIntegrityError("Checkpoint input_dim is invalid.")
    if (
        not isinstance(hidden_dim, int)
        or isinstance(hidden_dim, bool)
        or not 1 <= hidden_dim <= MAX_HIDDEN_DIM
    ):
        raise DatasetIntegrityError("Checkpoint hidden_dim is invalid.")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise DatasetIntegrityError("Checkpoint temperature is invalid.")
    report = _validated_report(payload.get("report"))
    if report.temperature != float(temperature):
        raise DatasetIntegrityError("Checkpoint report and policy temperature differ.")
    state = _validate_policy_state(
        payload.get("state_dict"), input_dim=input_dim, hidden_dim=hidden_dim
    )
    if payload.get("state_sha256") != _policy_state_sha256(state):
        raise DatasetIntegrityError("Checkpoint policy-state hash does not match.")
    policy = SurgeryMetaPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        temperature=float(temperature),
    )
    policy.model.load_state_dict(state, strict=True)
    return policy, report, _file_sha256(source)


__all__ = [
    "AttemptVectorizer",
    "CHECKPOINT_FORMAT",
    "DatasetIntegrityError",
    "FEATURE_VERSION",
    "MAX_BATCH_SIZE",
    "MAX_CHECKPOINT_BYTES",
    "MAX_DATASET_BYTES",
    "MAX_DATASET_RECORDS",
    "MAX_FEATURE_VALUES",
    "MAX_HIDDEN_DIM",
    "MAX_SET_CONTEXT",
    "POLICY_PURPOSE",
    "RankedAttempt",
    "SCHEMA_VERSION",
    "SurgeryMetaCompilerError",
    "SurgeryMetaPolicy",
    "TargetModelReferenceError",
    "TrainingConfig",
    "TrainingReport",
    "bootstrap_from_attempt",
    "canonical_json_bytes",
    "canonical_sha256",
    "canonicalize_attempt",
    "family_held_out_split",
    "load_policy_checkpoint",
    "read_jsonl",
    "save_policy_checkpoint",
    "train_policy",
    "write_jsonl",
]
