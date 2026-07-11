#!/usr/bin/env python3
"""Summarize and validate Qwen codebook sweep evidence deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SUMMARY_SCHEMA = "neural-forge-qwen-codebook-sweep-summary-v1"
TARGET_MANUAL_CATEGORIES = (
    "math",
    "code",
    "reasoning",
    "commonsense",
    "model-surgery",
)
_SUITES = ("behavior", "generation", "choice")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_HASH_KEYS = (
    "parent_eval_sha256",
    "cached_parent_eval_sha256",
    "parent_eval_fingerprint",
    "cached_parent_fingerprint",
    "parent_fingerprint",
)
_OUTCOME_PRIORITY = {
    "loss": 0,
    "gain": 1,
    "retained-pass": 2,
    "shared-failure": 3,
}


class EvidenceSummaryError(ValueError):
    """The supplied evidence is incomplete, inconsistent, or corrupted."""


class _Evidence:
    __slots__ = (
        "tag",
        "input_path",
        "input_kind",
        "document",
        "run_report",
        "comparison",
        "candidate_eval",
        "parent_eval_reference",
        "base_directory",
        "parent_eval",
        "parent_eval_path",
    )

    def __init__(
        self,
        *,
        tag: str,
        input_path: Path,
        input_kind: str,
        document: dict[str, Any],
        run_report: dict[str, Any] | None,
        comparison: dict[str, Any],
        candidate_eval: dict[str, Any] | None,
        parent_eval_reference: str | None,
        base_directory: Path,
    ) -> None:
        self.tag = tag
        self.input_path = input_path
        self.input_kind = input_kind
        self.document = document
        self.run_report = run_report
        self.comparison = comparison
        self.candidate_eval = candidate_eval
        self.parent_eval_reference = parent_eval_reference
        self.base_directory = base_directory
        self.parent_eval: dict[str, Any] | None = None
        self.parent_eval_path: Path | None = None


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceSummaryError("Evidence is not finite canonical JSON.") from exc


def canonical_json(value: Any) -> str:
    """Return one canonical JSON record with a trailing newline."""

    return _canonical_bytes(value).decode("utf-8") + "\n"


def _canonical_sha256(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceSummaryError(f"Cannot hash artifact: {path}") from exc
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceSummaryError(f"Cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceSummaryError(f"{label} must be a JSON object: {path}")
    _canonical_bytes(value)
    return value


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _remote_basename(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).name


def _resolve_reference(reference: Any, base: Path, fallback: str) -> Path | None:
    candidates: list[Path] = []
    if isinstance(reference, str) and reference:
        direct = Path(reference).expanduser()
        candidates.extend((direct, base / _remote_basename(reference)))
    candidates.append(base / fallback)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _assert_same(left: Any, right: Any, label: str) -> None:
    if _canonical_bytes(left) != _canonical_bytes(right):
        raise EvidenceSummaryError(f"Mismatched {label}.")


def _synthesized_comparison(run: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _mapping(run.get("aggregate")) or {}
    comparison: dict[str, Any] = {}
    for key in ("parent", "child", "param_compression_ratio"):
        if key in aggregate:
            comparison[key] = aggregate[key]
    for suite in _SUITES:
        if isinstance(aggregate.get(suite), Mapping):
            comparison[suite] = {"overall": dict(aggregate[suite])}
    if isinstance(run.get("manual_samples"), Mapping):
        comparison["manual_samples"] = dict(run["manual_samples"])
    return comparison


def _verify_run_comparison(
    run: Mapping[str, Any], comparison: Mapping[str, Any]
) -> None:
    aggregate = _mapping(run.get("aggregate"))
    if aggregate is None:
        return
    for key in ("parent", "child", "param_compression_ratio"):
        if key in aggregate and key in comparison:
            _assert_same(aggregate[key], comparison[key], f"run/comparison {key}")
    for suite in _SUITES:
        run_suite = _mapping(aggregate.get(suite))
        comparison_suite = _mapping(comparison.get(suite))
        comparison_overall = (
            _mapping(comparison_suite.get("overall")) if comparison_suite else None
        )
        if run_suite is not None and comparison_overall is not None:
            _assert_same(
                run_suite,
                comparison_overall,
                f"run/comparison {suite} aggregate",
            )


def _tag(document: Mapping[str, Any], base: Path) -> str:
    for key in ("tag", "run_tag", "name"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if not base.name:
        raise EvidenceSummaryError("Cannot derive a tag from an input path.")
    return base.name


def _load_evidence(path_value: str | Path) -> _Evidence:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        run_path = path / "run_report.json"
        comparison_path = path / "parent_candidate_comparison.json"
        if run_path.is_file():
            path = run_path
        elif comparison_path.is_file():
            path = comparison_path
        else:
            raise EvidenceSummaryError(
                f"Input directory has no supported evidence artifact: {path}"
            )
    if path.name not in {"run_report.json", "parent_candidate_comparison.json"}:
        raise EvidenceSummaryError(
            "Inputs must be run_report.json, parent_candidate_comparison.json, "
            f"or their containing directories: {path}"
        )
    document = _load_json(path, path.name)
    base = path.parent
    run: dict[str, Any] | None
    comparison: dict[str, Any]
    if path.name == "run_report.json":
        run = document
        comparison_path = _resolve_reference(
            run.get("comparison"), base, "parent_candidate_comparison.json"
        )
        if comparison_path is not None:
            comparison = _load_json(comparison_path, "parent/candidate comparison")
            _verify_run_comparison(run, comparison)
        else:
            comparison = _synthesized_comparison(run)
        input_kind = "run_report"
    else:
        run = None
        comparison = document
        input_kind = "parent_candidate_comparison"

    embedded_candidate = _mapping(run.get("candidate_eval")) if run else None
    candidate_path = _resolve_reference(
        comparison.get("child_eval"), base, "candidate_eval.json"
    )
    on_disk_candidate = (
        _load_json(candidate_path, "candidate evaluation")
        if candidate_path is not None
        else None
    )
    if embedded_candidate is not None and on_disk_candidate is not None:
        _assert_same(
            embedded_candidate,
            on_disk_candidate,
            f"embedded/on-disk candidate evaluation for {base.name}",
        )
    candidate_eval = (
        dict(embedded_candidate)
        if embedded_candidate is not None
        else on_disk_candidate
    )
    parent_reference = comparison.get("parent_eval")
    return _Evidence(
        tag=_tag(document, base),
        input_path=path,
        input_kind=input_kind,
        document=document,
        run_report=run,
        comparison=comparison,
        candidate_eval=candidate_eval,
        parent_eval_reference=(
            str(parent_reference) if isinstance(parent_reference, str) else None
        ),
        base_directory=base,
    )


def _candidate_parent_paths(
    evidence: Sequence[_Evidence], explicit_parent: Path | None
) -> list[Path]:
    resolved_paths: list[Path] = []
    if explicit_parent is not None:
        resolved = explicit_parent.expanduser().resolve()
        if not resolved.is_file():
            raise EvidenceSummaryError(f"Parent evaluation does not exist: {resolved}")
        resolved_paths.append(resolved)
    by_basename: dict[str, list[Path]] = {}
    for item in evidence:
        for candidate in item.base_directory.glob("*.json"):
            by_basename.setdefault(candidate.name, []).append(candidate.resolve())
    for item in evidence:
        reference = item.parent_eval_reference
        if reference:
            direct = Path(reference).expanduser()
            if direct.is_file():
                resolved_paths.append(direct.resolve())
                continue
            matches = by_basename.get(_remote_basename(reference), [])
            if matches:
                resolved_paths.extend(matches)
    if not resolved_paths:
        fallback = sorted(
            {
                candidate.resolve()
                for item in evidence
                for candidate in item.base_directory.glob("*parent*.json")
                if candidate.name
                not in {"parent_candidate_comparison.json", "run_report.json"}
            }
        )
        resolved_paths.extend(fallback)
    return sorted(set(resolved_paths))


def _collect_values_for_keys(value: Any, keys: set[str]) -> list[str]:
    values: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                values.append(item)
            values.extend(_collect_values_for_keys(item, keys))
    elif isinstance(value, list):
        for item in value:
            values.extend(_collect_values_for_keys(item, keys))
    return values


def _normalize_digest(value: str, label: str) -> str:
    normalized = value.lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[7:]
    if not _HEX_DIGEST.fullmatch(normalized):
        raise EvidenceSummaryError(f"Invalid SHA-256 digest in {label}: {value!r}")
    return f"sha256:{normalized}"


def _parent_hash_declarations(item: _Evidence) -> set[str]:
    declarations = {
        _normalize_digest(value, f"{item.tag} parent fingerprint")
        for value in _collect_values_for_keys(item.document, set(_HASH_KEYS))
    }
    cached_parent = None
    if item.run_report:
        certification = _mapping(item.run_report.get("certification_contract"))
        cached_parent = _mapping(certification.get("cached_parent")) if certification else None
    if cached_parent:
        for key in ("sha256", "fingerprint", "eval_sha256"):
            value = cached_parent.get(key)
            if isinstance(value, str):
                declarations.add(
                    _normalize_digest(value, f"{item.tag} cached parent {key}")
                )
    if len(declarations) > 1:
        raise EvidenceSummaryError(
            f"Conflicting parent fingerprint declarations for {item.tag}."
        )
    return declarations


def _aggregate_eval(report: Mapping[str, Any]) -> dict[str, Any]:
    behavior = _mapping(report.get("behavior_suite")) or {}
    generation = _mapping(report.get("generation_health")) or {}
    choice = _mapping(report.get("choice_accuracy")) or {}
    perplexity = _mapping(report.get("perplexity")) or {}
    result: dict[str, Any] = {}
    fields = {
        "model": report.get("model", report.get("source_model")),
        "param_count": report.get("param_count"),
        "behavior_correct": behavior.get("correct"),
        "behavior_total": behavior.get("total"),
        "generation_correct": generation.get("correct"),
        "generation_total": generation.get("total"),
        "choice_correct": choice.get("correct"),
        "choice_total": choice.get("total"),
        "perplexity_mean": perplexity.get("mean"),
    }
    for key, value in fields.items():
        if value is not None:
            result[key] = value
    return result


def _verify_aggregate(
    declared: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    actual = _aggregate_eval(evaluation) if evaluation is not None else {}
    if declared is not None and evaluation is not None:
        aggregate_fields = {
            "model",
            "param_count",
            "behavior_correct",
            "behavior_total",
            "generation_correct",
            "generation_total",
            "choice_correct",
            "choice_total",
            "perplexity_mean",
        }
        for key, value in declared.items():
            if key in aggregate_fields and key not in actual:
                raise EvidenceSummaryError(f"Missing {label} aggregate field {key}.")
            if key in actual and actual[key] != value:
                raise EvidenceSummaryError(f"Mismatched {label} aggregate field {key}.")
    if declared is not None:
        return dict(declared)
    return actual


def _attach_parent_evaluations(
    evidence: Sequence[_Evidence], explicit_parent: Path | None
) -> tuple[str, str | None, dict[str, Any]]:
    declared_basenames = {
        _remote_basename(item.parent_eval_reference)
        for item in evidence
        if item.parent_eval_reference
    }
    if len(declared_basenames) > 1:
        raise EvidenceSummaryError(
            "Candidates reference different cached parent evaluation artifacts."
        )
    paths = _candidate_parent_paths(evidence, explicit_parent)
    loaded: list[tuple[Path, dict[str, Any], str, str]] = []
    for path in paths:
        report = _load_json(path, "cached parent evaluation")
        loaded.append((path, report, _canonical_sha256(report), _file_sha256(path)))
    canonical_fingerprints = {entry[2] for entry in loaded}
    if len(canonical_fingerprints) > 1:
        raise EvidenceSummaryError("Cached parent evaluation artifacts do not match.")

    declarations = set().union(*(_parent_hash_declarations(item) for item in evidence))
    if len(declarations) > 1:
        raise EvidenceSummaryError("Candidates declare different cached parent fingerprints.")
    declared = next(iter(declarations), None)
    if loaded:
        parent_path, parent_report, canonical_fingerprint, raw_hash = loaded[0]
        valid_file_hashes = {entry[3] for entry in loaded}
        if declared is not None and declared not in {
            canonical_fingerprint,
            *valid_file_hashes,
        }:
            raise EvidenceSummaryError(
                "Declared cached parent fingerprint does not match the parent artifact."
            )
        for item in evidence:
            item.parent_eval = parent_report
            item.parent_eval_path = parent_path
        fingerprint = canonical_fingerprint
        artifact_hash: str | None = raw_hash
    elif declared is not None:
        fingerprint = declared
        artifact_hash = None
    else:
        raise EvidenceSummaryError(
            "Cannot establish a cached parent fingerprint; provide --parent-eval or "
            "a parent fingerprint field."
        )

    parent_aggregates: list[dict[str, Any]] = []
    for item in evidence:
        declared_parent = _mapping(item.comparison.get("parent"))
        parent_aggregates.append(
            _verify_aggregate(
                declared_parent,
                item.parent_eval,
                f"{item.tag} parent",
            )
        )
    if parent_aggregates:
        first = parent_aggregates[0]
        for aggregate in parent_aggregates[1:]:
            _assert_same(first, aggregate, "cached parent aggregate")
        parent_aggregate = first
    else:
        parent_aggregate = {}
    return fingerprint, artifact_hash, parent_aggregate


def _ordered_ids(items: Any, label: str) -> list[str]:
    if not isinstance(items, list):
        return []
    result = [
        str(item["id"])
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    ]
    if len(result) != len(items) or len(result) != len(set(result)):
        raise EvidenceSummaryError(f"{label} has missing or duplicate prompt IDs.")
    return result


def _derived_contract(report: Mapping[str, Any] | None, label: str) -> dict[str, Any] | None:
    if report is None:
        return None
    suite = _mapping(report.get("eval_suite"))
    if suite is None:
        return None
    collections = {
        "choice_probes": _sequence((_mapping(report.get("choice_accuracy")) or {}).get("items")),
        "generation_prompts": _sequence(report.get("generations")),
        "behavior_probes": _sequence((_mapping(report.get("behavior_suite")) or {}).get("items")),
    }
    perplexity_items = _sequence((_mapping(report.get("perplexity")) or {}).get("items"))
    counts = {
        "text_evals": suite.get("text_eval_count"),
        "choice_probes": suite.get("choice_probe_count"),
        "generation_prompts": suite.get("generation_prompt_count"),
        "behavior_probes": suite.get("behavior_probe_count"),
    }
    observed_counts = {
        "text_evals": len(perplexity_items),
        **{name: len(items) for name, items in collections.items()},
    }
    for name, observed in observed_counts.items():
        declared = counts.get(name)
        if declared is None:
            counts[name] = observed
        elif int(declared) != observed:
            raise EvidenceSummaryError(
                f"{label} eval count mismatch for {name}: {declared} != {observed}."
            )
        else:
            counts[name] = int(declared)
    source = suite.get("source")
    if not isinstance(source, str) or not source:
        raise EvidenceSummaryError(f"{label} eval contract lacks a suite source.")
    return {
        "suite_source": source,
        "counts": counts,
        "prompt_ids": {
            name: _ordered_ids(items, f"{label} {name}")
            for name, items in collections.items()
        },
    }


def _declared_contracts(item: _Evidence) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    run = item.run_report
    if run:
        certification = _mapping(run.get("certification_contract"))
        if certification:
            candidates.append(certification.get("contract", certification))
        candidates.extend((run.get("eval_contract"), run.get("evaluation_contract")))
    candidates.extend(
        (
            item.comparison.get("eval_contract"),
            item.comparison.get("evaluation_contract"),
        )
    )
    if item.candidate_eval:
        context = _mapping(item.candidate_eval.get("certification_context"))
        if context:
            candidates.append(context.get("contract"))
    contracts = [dict(value) for value in candidates if isinstance(value, Mapping)]
    if contracts:
        first = contracts[0]
        for contract in contracts[1:]:
            _assert_same(first, contract, f"duplicated eval contract for {item.tag}")
    return contracts


def _contract_identity(contract: Mapping[str, Any], label: str) -> dict[str, Any]:
    source = contract.get("suite_source")
    counts = _mapping(contract.get("counts"))
    prompt_ids = _mapping(contract.get("prompt_ids"))
    if not isinstance(source, str) or not source or counts is None or prompt_ids is None:
        raise EvidenceSummaryError(f"{label} eval contract is incomplete.")
    normalized_counts: dict[str, int] = {}
    for name in ("text_evals", "choice_probes", "generation_prompts", "behavior_probes"):
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EvidenceSummaryError(f"{label} eval contract has invalid count {name}.")
        normalized_counts[name] = value
    normalized_ids: dict[str, list[str]] = {}
    for name in ("choice_probes", "generation_prompts", "behavior_probes"):
        values = prompt_ids.get(name)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise EvidenceSummaryError(f"{label} eval contract has invalid IDs for {name}.")
        if len(values) != normalized_counts[name] or len(values) != len(set(values)):
            raise EvidenceSummaryError(f"{label} eval contract ID count mismatch for {name}.")
        normalized_ids[name] = list(values)
    text_hash = contract.get("text_evals_sha256")
    if text_hash is not None:
        if not isinstance(text_hash, str):
            raise EvidenceSummaryError(f"{label} text eval hash is not a string.")
        _normalize_digest(text_hash, f"{label} text eval hash")
    return {
        "suite_source": source,
        "counts": normalized_counts,
        "prompt_ids": normalized_ids,
    }


def _verify_validation_record(
    record: Mapping[str, Any] | None,
    contract_identity: Mapping[str, Any],
    aggregate: Mapping[str, Any] | None,
    label: str,
) -> None:
    if record is None:
        return
    eval_counts = record.get("eval_counts")
    if eval_counts is not None:
        if not isinstance(eval_counts, Mapping):
            raise EvidenceSummaryError(f"{label} eval_counts is not an object.")
        _assert_same(eval_counts, contract_identity["counts"], f"{label} eval counts")
    prompt_ids = record.get("prompt_ids")
    if prompt_ids is not None:
        if not isinstance(prompt_ids, Mapping):
            raise EvidenceSummaryError(f"{label} prompt_ids is not an object.")
        _assert_same(prompt_ids, contract_identity["prompt_ids"], f"{label} prompt IDs")
    model = record.get("model")
    if model is not None and aggregate is not None and aggregate.get("model") is not None:
        if model != aggregate["model"]:
            raise EvidenceSummaryError(f"{label} model provenance does not match.")


def _verify_certification_records(
    item: _Evidence, contract_identity: Mapping[str, Any]
) -> None:
    parent_aggregate = _mapping(item.comparison.get("parent"))
    candidate_aggregate = _mapping(item.comparison.get("child"))
    certification = (
        _mapping(item.run_report.get("certification_contract"))
        if item.run_report
        else None
    )
    cached_parent = _mapping(certification.get("cached_parent")) if certification else None
    _verify_validation_record(
        cached_parent,
        contract_identity,
        parent_aggregate,
        f"{item.tag} cached-parent certification",
    )
    context = (
        _mapping(item.candidate_eval.get("certification_context"))
        if item.candidate_eval
        else None
    )
    validated_candidate = (
        _mapping(context.get("validated_candidate")) if context else None
    )
    _verify_validation_record(
        validated_candidate,
        contract_identity,
        candidate_aggregate,
        f"{item.tag} candidate certification",
    )


def _verify_contracts(evidence: Sequence[_Evidence]) -> tuple[dict[str, Any], str]:
    identities: list[dict[str, Any]] = []
    full_declared: list[dict[str, Any]] = []
    for item in evidence:
        contracts = _declared_contracts(item)
        declared = contracts[0] if contracts else None
        parent_derived = _derived_contract(item.parent_eval, f"{item.tag} parent")
        candidate_derived = _derived_contract(item.candidate_eval, f"{item.tag} candidate")
        if parent_derived is not None and candidate_derived is not None:
            _assert_same(parent_derived, candidate_derived, f"{item.tag} parent/candidate eval contract")
        derived = candidate_derived or parent_derived
        if declared is not None:
            identity = _contract_identity(declared, item.tag)
            if derived is not None:
                _assert_same(identity, derived, f"{item.tag} declared/observed eval contract")
            settings = declared.get("settings")
            if settings is not None and not isinstance(settings, Mapping):
                raise EvidenceSummaryError(f"{item.tag} eval settings are not an object.")
            full_declared.append(declared)
        elif derived is not None:
            identity = derived
        else:
            raise EvidenceSummaryError(
                f"Cannot establish the evaluation contract for {item.tag}."
            )
        _verify_certification_records(item, identity)
        identities.append(identity)
    first_identity = identities[0]
    for identity in identities[1:]:
        _assert_same(first_identity, identity, "cross-candidate eval contract")
    if full_declared:
        first_declared = full_declared[0]
        for contract in full_declared[1:]:
            _assert_same(first_declared, contract, "cross-candidate declared eval contract")
    return first_identity, _canonical_sha256(first_identity)


def _manifest_sources(run: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    sources: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    execution = _mapping(run.get("execution_configuration"))
    verification = _mapping(run.get("verification"))
    for container in (
        _mapping(execution.get("compact_checkpoint")) if execution else None,
        _mapping(verification.get("compact_checkpoint")) if verification else None,
        _mapping(run.get("compact_checkpoint")),
    ):
        if container and isinstance(container.get("manifest"), Mapping):
            sources.append((container, container["manifest"]))
    return sources


def _valid_manifest_digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise EvidenceSummaryError(f"Missing artifact hash for {label}.")
    return _normalize_digest(value, label)


def _checkpoint_directories(
    item: _Evidence,
    container: Mapping[str, Any],
    artifact_roots: Sequence[Path],
) -> list[Path]:
    primary_candidates = [item.base_directory / "compact_child"]
    declared = container.get("artifact_directory")
    if isinstance(declared, str) and declared:
        direct = Path(declared).expanduser()
        primary_candidates.extend(
            (direct, item.base_directory / _remote_basename(declared))
        )
    root_candidates: list[Path] = []
    for root in artifact_roots:
        resolved = root.expanduser().resolve()
        root_candidates.extend(
            (
                resolved,
                resolved / "compact_child",
                resolved / item.tag / "compact_child",
            )
        )
    unique: dict[str, Path] = {}
    for candidate in primary_candidates:
        if candidate.is_dir():
            unique[str(candidate)] = candidate
    for candidate in root_candidates:
        if candidate.is_dir() and (candidate / "manifest.json").is_file():
            unique[str(candidate)] = candidate
    return list(unique.values())


def _verify_checkpoint_manifest_contract(
    manifest: Mapping[str, Any], parameters: Mapping[str, Any], item: _Evidence
) -> None:
    state_key = manifest.get("state_key_sha256")
    if state_key is not None:
        _valid_manifest_digest(state_key, f"{item.tag} state-key hash")
    for group_name in ("assignment_hashes", "prototype_source_hashes"):
        group = manifest.get(group_name)
        if group is not None:
            if not isinstance(group, Mapping):
                raise EvidenceSummaryError(f"{item.tag} {group_name} is not an object.")
            for name, digest in group.items():
                _valid_manifest_digest(digest, f"{item.tag} {group_name}.{name}")
    report = _mapping(manifest.get("parameter_report"))
    if report:
        checks = {
            "source_unique_parameter_count": parameters.get("source_parameter_count"),
            "unique_parameter_count": parameters.get("unique_child_parameter_count"),
            "index_buffer_bytes": parameters.get("index_buffer_bytes"),
        }
        for key, expected in checks.items():
            if expected is not None and report.get(key) is not None and report[key] != expected:
                raise EvidenceSummaryError(
                    f"{item.tag} checkpoint parameter contract mismatch for {key}."
                )
    token_fingerprint = manifest.get("token_row_fingerprint")
    if token_fingerprint is not None:
        normalized_token_fingerprint = _valid_manifest_digest(
            token_fingerprint, f"{item.tag} checkpoint token-row fingerprint"
        )
        accounting = _mapping(item.run_report.get("accounting")) if item.run_report else None
        runtime = _mapping(accounting.get("runtime_report")) if accounting else None
        declared_token_fingerprint = runtime.get("token_row_fingerprint") if runtime else None
        if (
            declared_token_fingerprint is not None
            and normalized_token_fingerprint
            != _normalize_digest(
                str(declared_token_fingerprint),
                f"{item.tag} runtime token-row fingerprint",
            )
        ):
            raise EvidenceSummaryError(
                f"{item.tag} checkpoint token-row contract mismatch."
            )


def _verify_artifacts(
    item: _Evidence,
    parameters: Mapping[str, Any],
    artifact_roots: Sequence[Path],
) -> dict[str, Any]:
    if item.run_report is None:
        return {"status": "not-declared"}
    sources = _manifest_sources(item.run_report)
    if not sources:
        return {"status": "not-declared"}
    first_container, first_manifest = sources[0]
    for _, manifest in sources[1:]:
        _assert_same(first_manifest, manifest, f"duplicated checkpoint manifest for {item.tag}")
    manifest = dict(first_manifest)
    files = _mapping(manifest.get("files"))
    if files is None or not files:
        raise EvidenceSummaryError(f"{item.tag} checkpoint manifest has no file hashes.")
    normalized_files: dict[str, str] = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not name or _remote_basename(name) != name:
            raise EvidenceSummaryError(f"{item.tag} checkpoint manifest has an unsafe path.")
        normalized_files[name] = _valid_manifest_digest(
            digest, f"{item.tag} checkpoint file {name}"
        )
    _verify_checkpoint_manifest_contract(manifest, parameters, item)
    directories = _checkpoint_directories(item, first_container, artifact_roots)
    if not directories:
        return {
            "status": "declared-not-locally-available",
            "manifest_sha256": _canonical_sha256(manifest),
            "declared_file_count": len(normalized_files),
            "verified_file_count": 0,
        }
    valid_directories: list[Path] = []
    for directory in directories:
        local_manifest_path = directory / "manifest.json"
        if local_manifest_path.is_file():
            local_manifest = _load_json(local_manifest_path, "checkpoint manifest")
            _assert_same(
                manifest,
                local_manifest,
                f"embedded/on-disk checkpoint manifest for {item.tag}",
            )
        for name, expected in normalized_files.items():
            candidate = directory / name
            if not candidate.is_file():
                raise EvidenceSummaryError(
                    f"{item.tag} checkpoint artifact is missing: {candidate}"
                )
            if _file_sha256(candidate) != expected:
                raise EvidenceSummaryError(
                    f"{item.tag} checkpoint hash mismatch for {name}."
                )
        valid_directories.append(directory)
    return {
        "status": "verified",
        "manifest_sha256": _canonical_sha256(manifest),
        "declared_file_count": len(normalized_files),
        "verified_file_count": len(normalized_files),
        "verified_copy_count": len(valid_directories),
    }


def _dtype_bytes(item: _Evidence) -> tuple[str, int] | None:
    candidates: list[Any] = []
    if item.run_report:
        execution = _mapping(item.run_report.get("execution_configuration"))
        if execution:
            candidates.extend(_sequence(execution.get("parameter_dtypes")))
            candidates.append(execution.get("dtype"))
    if item.candidate_eval:
        context = _mapping(item.candidate_eval.get("certification_context"))
        runtime = _mapping(context.get("runtime")) if context else None
        if runtime:
            candidates.extend(_sequence(runtime.get("parameter_dtypes")))
            candidates.append(runtime.get("requested_dtype"))
    normalized = {
        str(value).lower().removeprefix("torch.")
        for value in candidates
        if isinstance(value, str) and value
    }
    byte_widths = {
        "float64": 8,
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int64": 8,
        "int32": 4,
        "int16": 2,
        "int8": 1,
        "uint8": 1,
    }
    recognized = {(name, byte_widths[name]) for name in normalized if name in byte_widths}
    widths = {width for _, width in recognized}
    if len(widths) != 1 or not recognized:
        return None
    preferred = sorted(name for name, _ in recognized)[0]
    return preferred, next(iter(widths))


def _number(mapping: Mapping[str, Any], key: str) -> int | float | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceSummaryError(f"Non-finite numeric evidence in {key}.")
    return value


def _parameters(item: _Evidence) -> dict[str, Any]:
    accounting = _mapping(item.run_report.get("accounting")) if item.run_report else None
    accounting_parameters = _mapping(accounting.get("parameters")) if accounting else None
    accounting_indices = _mapping(accounting.get("indices")) if accounting else None
    accounting_ratios = _mapping(accounting.get("ratios")) if accounting else None
    parent = _mapping(item.comparison.get("parent")) or {}
    child = _mapping(item.comparison.get("child")) or {}
    source = (
        _number(accounting_parameters, "source_unique")
        if accounting_parameters
        else None
    )
    unique_child = (
        _number(accounting_parameters, "runtime_unique")
        if accounting_parameters
        else None
    )
    if source is None:
        source = _number(parent, "param_count")
    if unique_child is None:
        unique_child = _number(child, "param_count")
    result: dict[str, Any] = {}
    if accounting_parameters:
        logical = _number(accounting_parameters, "source_logical")
        if logical is not None:
            result["source_logical_parameter_count"] = logical
    if source is not None:
        result["source_parameter_count"] = source
    if unique_child is not None:
        result["unique_child_parameter_count"] = unique_child
    index_bytes = (
        _number(accounting_indices, "total_bytes") if accounting_indices else None
    )
    if index_bytes is not None:
        result["index_buffer_bytes"] = index_bytes
    if source is not None and unique_child not in (None, 0):
        result["raw_compression_ratio"] = source / unique_child
    reported_ratio = (
        _number(accounting_ratios, "source_to_runtime")
        if accounting_ratios
        else None
    )
    if reported_ratio is None:
        reported_ratio = _number(item.comparison, "param_compression_ratio")
    if reported_ratio is not None:
        result["reported_compression_ratio"] = reported_ratio
        derived = result.get("raw_compression_ratio")
        if derived is not None and not math.isclose(
            float(derived), float(reported_ratio), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise EvidenceSummaryError(f"{item.tag} compression ratio is inconsistent.")
    dtype = _dtype_bytes(item)
    if (
        dtype is not None
        and source is not None
        and unique_child is not None
        and index_bytes is not None
    ):
        dtype_name, bytes_per_parameter = dtype
        source_bytes = source * bytes_per_parameter
        child_bytes = unique_child * bytes_per_parameter + index_bytes
        if child_bytes > 0:
            result.update(
                {
                    "parameter_dtype": dtype_name,
                    "bytes_per_parameter": bytes_per_parameter,
                    "adjusted_ratio": source_bytes / child_bytes,
                    "adjusted_ratio_formula": (
                        "source_parameter_count*bytes_per_parameter/"
                        "(unique_child_parameter_count*bytes_per_parameter+"
                        "index_buffer_bytes)"
                    ),
                }
            )
    return result


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "uncategorized"
    normalized = re.sub(r"[\s_]+", "-", value.strip().lower())
    aliases = {
        "common-sense": "commonsense",
        "modelsurgery": "model-surgery",
        "model-surgery": "model-surgery",
    }
    return aliases.get(normalized, normalized)


def _retention_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "total_ids": "total",
        "shared_ids": "shared",
        "parent_pass": "parent_passes",
        "child_pass": "candidate_passes",
        "retained_parent_pass": "retained_parent_passes",
        "lost_parent_pass": "losses",
        "child_only_or_parent_failed_gain": "gains",
        "parent_pass_retention_rate": "parent_pass_retention_rate",
        "shared_parent_pass_retention_rate": "shared_parent_pass_retention_rate",
    }
    result = {
        target: summary[source]
        for source, target in keys.items()
        if summary.get(source) is not None
    }
    parent_passes = result.get("parent_passes")
    candidate_passes = result.get("candidate_passes")
    if isinstance(parent_passes, (int, float)) and isinstance(
        candidate_passes, (int, float)
    ):
        result["pass_delta"] = candidate_passes - parent_passes
    return result


def _item_map(report: Mapping[str, Any] | None, suite: str) -> dict[str, dict[str, Any]]:
    if report is None:
        return {}
    if suite == "behavior":
        items = _sequence((_mapping(report.get("behavior_suite")) or {}).get("items"))
    elif suite == "choice":
        items = _sequence((_mapping(report.get("choice_accuracy")) or {}).get("items"))
    else:
        items = _sequence(report.get("generations"))
        diagnostics = {
            str(item["id"]): item
            for item in _sequence(report.get("generation_diagnostics"))
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        merged: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                continue
            combined = dict(item)
            combined.update(diagnostics.get(str(item["id"]), {}))
            merged.append(combined)
        items = merged
    return {
        str(item["id"]): dict(item)
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }


def _passed(item: Mapping[str, Any], suite: str) -> bool:
    return bool(item.get("correct" if suite == "choice" else "passed"))


def _computed_category_summaries(
    parent_eval: Mapping[str, Any] | None,
    candidate_eval: Mapping[str, Any] | None,
    suite: str,
) -> dict[str, dict[str, Any]]:
    parent_items = _item_map(parent_eval, suite)
    candidate_items = _item_map(candidate_eval, suite)
    all_ids = sorted(set(parent_items) | set(candidate_items))

    def category_for(item_id: str) -> str:
        candidate = candidate_items.get(item_id, {})
        parent = parent_items.get(item_id, {})
        return _normalize_category(candidate.get("category", parent.get("category")))

    categories = sorted(
        {category_for(item_id) for item_id in all_ids}
        - {"uncategorized"}
    )
    result: dict[str, dict[str, Any]] = {}
    for category in categories:
        ids = [item_id for item_id in all_ids if category_for(item_id) == category]
        parent_pass = {
            item_id
            for item_id in ids
            if item_id in parent_items and _passed(parent_items[item_id], suite)
        }
        candidate_pass = {
            item_id
            for item_id in ids
            if item_id in candidate_items and _passed(candidate_items[item_id], suite)
        }
        retained = parent_pass & candidate_pass
        result[category] = {
            "total": len(ids),
            "parent_passes": len(parent_pass),
            "candidate_passes": len(candidate_pass),
            "retained_parent_passes": len(retained),
            "losses": len(parent_pass - candidate_pass),
            "gains": len(candidate_pass - parent_pass),
            "pass_delta": len(candidate_pass) - len(parent_pass),
            "parent_pass_retention_rate": (
                len(retained) / len(parent_pass) if parent_pass else None
            ),
        }
    return result


def _category_metrics(item: _Evidence) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for suite in _SUITES:
        suite_comparison = _mapping(item.comparison.get(suite))
        by_category = (
            _mapping(suite_comparison.get("by_category"))
            if suite_comparison
            else None
        )
        if by_category:
            categories: dict[str, dict[str, Any]] = {}
            for raw_category, summary in sorted(by_category.items()):
                if not isinstance(summary, Mapping):
                    continue
                category = _normalize_category(raw_category)
                if category in categories:
                    raise EvidenceSummaryError(
                        f"{item.tag} has colliding normalized {suite} categories."
                    )
                categories[category] = _retention_summary(summary)
        else:
            categories = _computed_category_summaries(
                item.parent_eval, item.candidate_eval, suite
            )
        if categories:
            result[suite] = categories
    return result


def _evaluation(item: _Evidence) -> dict[str, Any]:
    parent = _verify_aggregate(
        _mapping(item.comparison.get("parent")), item.parent_eval, f"{item.tag} parent"
    )
    candidate = _verify_aggregate(
        _mapping(item.comparison.get("child")),
        item.candidate_eval,
        f"{item.tag} candidate",
    )
    suites: dict[str, Any] = {}
    prefixes = {
        "behavior": "behavior",
        "generation": "generation",
        "choice": "choice",
    }
    for suite, prefix in prefixes.items():
        parent_pass = parent.get(f"{prefix}_correct")
        parent_total = parent.get(f"{prefix}_total")
        child_pass = candidate.get(f"{prefix}_correct")
        child_total = candidate.get(f"{prefix}_total")
        values: dict[str, Any] = {}
        if parent_pass is not None or parent_total is not None:
            values["parent"] = {
                key: value
                for key, value in {"passes": parent_pass, "total": parent_total}.items()
                if value is not None
            }
        if child_pass is not None or child_total is not None:
            values["candidate"] = {
                key: value
                for key, value in {"passes": child_pass, "total": child_total}.items()
                if value is not None
            }
        if parent_pass is not None and child_pass is not None:
            values["pass_delta"] = child_pass - parent_pass
        comparison_suite = _mapping(item.comparison.get(suite))
        overall = _mapping(comparison_suite.get("overall")) if comparison_suite else None
        if overall:
            values["retention"] = _retention_summary(overall)
        if values:
            suites[suite] = values
    result: dict[str, Any] = {"suites": suites}
    parent_ppl = parent.get("perplexity_mean")
    candidate_ppl = candidate.get("perplexity_mean")
    if parent_ppl is not None or candidate_ppl is not None:
        perplexity: dict[str, Any] = {}
        if parent_ppl is not None:
            perplexity["parent_mean"] = parent_ppl
        if candidate_ppl is not None:
            perplexity["candidate_mean"] = candidate_ppl
        if parent_ppl is not None and candidate_ppl is not None:
            perplexity["delta"] = candidate_ppl - parent_ppl
            if parent_ppl != 0:
                perplexity["ratio"] = candidate_ppl / parent_ppl
        result["perplexity"] = perplexity
    logit_drift = _logit_drift(item)
    if logit_drift is not None:
        result["logit_drift"] = logit_drift
    categories = _category_metrics(item)
    if categories:
        result["categories"] = categories
    return result


def _logit_drift(item: _Evidence) -> dict[str, Any] | None:
    direct_candidates: list[tuple[str, Any]] = [
        ("comparison.logit_drift", item.comparison.get("logit_drift")),
    ]
    if item.run_report:
        direct_candidates.extend(
            (
                ("run_report.logit_drift", item.run_report.get("logit_drift")),
                (
                    "verification.parent_candidate_logit_drift",
                    (_mapping(item.run_report.get("verification")) or {}).get(
                        "parent_candidate_logit_drift"
                    ),
                ),
            )
        )
    for source, value in direct_candidates:
        if isinstance(value, Mapping) and value:
            return {"source": source, **dict(value)}
    if item.run_report:
        verification = _mapping(item.run_report.get("verification")) or {}
        noop = _mapping(verification.get("full_depth_noop"))
        if noop:
            drift_keys = (
                "maximum_absolute_logit_error",
                "mean_absolute_logit_error",
                "relative_l2_logit_error",
                "top1_match_count",
                "top1_count",
                "top1_exact",
                "logits_bitwise_exact",
                "logits_numerically_equivalent",
                "absolute_tolerance",
                "relative_tolerance",
            )
            values = {key: noop[key] for key in drift_keys if noop.get(key) is not None}
            if values:
                return {"source": "verification.full_depth_noop", **values}
    return None


def _continuation(item: Mapping[str, Any]) -> str:
    continuation = item.get("continuation")
    if isinstance(continuation, str):
        return continuation.strip()
    output = item.get("output")
    prompt = item.get("prompt")
    if isinstance(output, str):
        if isinstance(prompt, str) and output.startswith(prompt):
            return output[len(prompt) :].strip()
        return output.strip()
    prediction = item.get("prediction")
    return str(prediction) if prediction is not None else ""


def _manual_pairs_from_evals(item: _Evidence) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for suite in _SUITES:
        parent_items = _item_map(item.parent_eval, suite)
        candidate_items = _item_map(item.candidate_eval, suite)
        for item_id in sorted(set(parent_items) & set(candidate_items)):
            parent_item = parent_items[item_id]
            candidate_item = candidate_items[item_id]
            category = _normalize_category(
                candidate_item.get("category", parent_item.get("category"))
            )
            if category not in TARGET_MANUAL_CATEGORIES:
                continue
            parent_passed = _passed(parent_item, suite)
            candidate_passed = _passed(candidate_item, suite)
            if parent_passed and candidate_passed:
                outcome = "retained-pass"
            elif parent_passed:
                outcome = "loss"
            elif candidate_passed:
                outcome = "gain"
            else:
                outcome = "shared-failure"
            example: dict[str, Any] = {
                "suite": suite,
                "id": item_id,
                "category": category,
                "outcome": outcome,
                "parent_passed": parent_passed,
                "candidate_passed": candidate_passed,
                "parent_output": _continuation(parent_item),
                "candidate_output": _continuation(candidate_item),
            }
            prompt = candidate_item.get("prompt", parent_item.get("prompt"))
            if isinstance(prompt, str):
                example["prompt"] = prompt
            pairs.append(example)
    return pairs


def _manual_pairs_from_comparison(item: _Evidence) -> list[dict[str, Any]]:
    manual = _mapping(item.comparison.get("manual_samples")) or {}
    pairs: list[dict[str, Any]] = []
    for suite in ("behavior", "generation", "choice"):
        for value in _sequence(manual.get(suite)):
            if not isinstance(value, Mapping):
                continue
            category = _normalize_category(value.get("category"))
            if category not in TARGET_MANUAL_CATEGORIES:
                continue
            parent_passed = bool(value.get("parent_passed"))
            candidate_passed = bool(
                value.get("candidate_passed", value.get("child_passed"))
            )
            if parent_passed and candidate_passed:
                outcome = "retained-pass"
            elif parent_passed:
                outcome = "loss"
            elif candidate_passed:
                outcome = "gain"
            else:
                outcome = "shared-failure"
            example = {
                "suite": suite,
                "id": str(value.get("id", "")),
                "category": category,
                "outcome": outcome,
                "parent_passed": parent_passed,
                "candidate_passed": candidate_passed,
                "parent_output": str(value.get("parent_continuation", "")),
                "candidate_output": str(
                    value.get("candidate_continuation", value.get("child_continuation", ""))
                ),
            }
            if isinstance(value.get("prompt"), str):
                example["prompt"] = value["prompt"]
            pairs.append(example)
    return pairs


def _manual_examples(item: _Evidence, per_category: int) -> dict[str, list[dict[str, Any]]]:
    if per_category <= 0:
        raise EvidenceSummaryError("examples_per_category must be positive.")
    pairs = _manual_pairs_from_comparison(item) + _manual_pairs_from_evals(item)
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        deduplicated[(str(pair["suite"]), str(pair["id"]))] = pair
    result: dict[str, list[dict[str, Any]]] = {}
    for category in TARGET_MANUAL_CATEGORIES:
        category_pairs = [
            pair for pair in deduplicated.values() if pair["category"] == category
        ]
        category_pairs.sort(
            key=lambda pair: (
                _OUTCOME_PRIORITY[str(pair["outcome"])],
                str(pair["suite"]),
                str(pair["id"]),
            )
        )
        result[category] = category_pairs[:per_category]
    return result


def _script_counts(value: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for character in value:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name:
            script = "latin"
        elif "CJK" in name or "IDEOGRAPH" in name:
            script = "han"
        elif "HIRAGANA" in name or "KATAKANA" in name:
            script = "japanese"
        elif "CYRILLIC" in name:
            script = "cyrillic"
        elif "ARABIC" in name:
            script = "arabic"
        elif "HEBREW" in name:
            script = "hebrew"
        elif "DEVANAGARI" in name:
            script = "devanagari"
        elif "HANGUL" in name:
            script = "hangul"
        elif "GREEK" in name:
            script = "greek"
        else:
            script = "other"
        result[script] += 1
    return result


def _language_switch(prompt: str, output: str) -> bool:
    prompt_scripts = _script_counts(prompt)
    output_scripts = _script_counts(output)
    output_total = sum(output_scripts.values())
    if output_total < 3:
        return False
    prompt_dominant = prompt_scripts.most_common(1)[0][0] if prompt_scripts else None
    output_dominant = output_scripts.most_common(1)[0][0]
    if prompt_dominant and output_dominant != prompt_dominant:
        dominant_count = output_scripts[output_dominant]
        if dominant_count >= 3 and dominant_count / output_total >= 0.25:
            return True
    significant = [
        count for count in output_scripts.values() if count >= 3 and count / output_total >= 0.15
    ]
    return len(significant) >= 2


def _has_repetition(value: str) -> bool:
    normalized = " ".join(value.split())
    if not normalized:
        return False
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) >= 4 and Counter(lines).most_common(1)[0][1] >= 3:
        return True
    tokens = re.findall(r"[^\W_]+|[^\w\s]", normalized, flags=re.UNICODE)
    if len(tokens) >= 12 and Counter(tokens).most_common(1)[0][1] / len(tokens) >= 0.55:
        return True
    compact = re.sub(r"\s+", "", normalized)[:10000]
    for width in range(1, min(13, len(compact) // 5 + 1)):
        repeated_width = width * 5
        for start in range(0, len(compact) - repeated_width + 1):
            unit = compact[start : start + width]
            if unit and compact.startswith(unit * 5, start):
                return True
    return False


def _candidate_outputs(item: _Evidence) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    if item.candidate_eval is not None:
        for suite in ("behavior", "generation"):
            for item_id, value in sorted(_item_map(item.candidate_eval, suite).items()):
                outputs.append(
                    {
                        "key": f"{suite}:{item_id}",
                        "category": _normalize_category(value.get("category")),
                        "prompt": str(value.get("prompt", "")),
                        "output": _continuation(value),
                    }
                )
    else:
        for suite, values in (_mapping(item.comparison.get("manual_samples")) or {}).items():
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                if not isinstance(value, Mapping):
                    continue
                item_id = str(value.get("id", index))
                outputs.append(
                    {
                        "key": f"{suite}:{item_id}",
                        "category": _normalize_category(value.get("category")),
                        "prompt": str(value.get("prompt", "")),
                        "output": str(
                            value.get(
                                "candidate_continuation",
                                value.get("child_continuation", ""),
                            )
                        ).strip(),
                    }
                )
    deduplicated = {value["key"]: value for value in outputs}
    return [deduplicated[key] for key in sorted(deduplicated)]


def _heuristics(item: _Evidence) -> dict[str, Any]:
    outputs = _candidate_outputs(item)
    repetition: list[str] = []
    language: list[str] = []
    empty: list[str] = []
    by_category: dict[str, dict[str, int]] = {}
    for value in outputs:
        category = value["category"]
        bucket = by_category.setdefault(
            category,
            {"outputs": 0, "repetition": 0, "language_switch": 0, "empty_output": 0},
        )
        bucket["outputs"] += 1
        if not value["output"].strip():
            empty.append(value["key"])
            bucket["empty_output"] += 1
            continue
        if _has_repetition(value["output"]):
            repetition.append(value["key"])
            bucket["repetition"] += 1
        if _language_switch(value["prompt"], value["output"]):
            language.append(value["key"])
            bucket["language_switch"] += 1
    total = len(outputs)

    def summary(values: list[str], definition: str) -> dict[str, Any]:
        return {
            "flagged": len(values),
            "rate": len(values) / total if total else None,
            "sample_ids": values[:25],
            "definition": definition,
        }

    return {
        "outputs_analyzed": total,
        "repetition": summary(
            repetition,
            "repeated line, >=55% dominant token over >=12 tokens, or a 1-12 "
            "character unit repeated at least five times",
        ),
        "language_switch": summary(
            language,
            "output changes dominant Unicode script from the prompt or mixes at "
            "least two scripts at >=15% each",
        ),
        "empty_output": summary(empty, "continuation is empty after whitespace stripping"),
        "by_category": {key: by_category[key] for key in sorted(by_category)},
    }


def _candidate_summary(
    item: _Evidence,
    parent_fingerprint: str,
    contract_fingerprint: str,
    artifact_roots: Sequence[Path],
    examples_per_category: int,
) -> dict[str, Any]:
    parameters = _parameters(item)
    artifacts = _verify_artifacts(item, parameters, artifact_roots)
    result = {
        "tag": item.tag,
        "input": {
            "artifact": item.input_path.name,
            "kind": item.input_kind,
            "sha256": _file_sha256(item.input_path),
        },
        "parent_fingerprint": parent_fingerprint,
        "eval_contract_fingerprint": contract_fingerprint,
        "parameters": parameters,
        "evaluation": _evaluation(item),
        "output_heuristics": _heuristics(item),
        "manual_examples": _manual_examples(item, examples_per_category),
        "artifact_verification": artifacts,
    }
    contracts = _declared_contracts(item)
    if contracts:
        result["declared_eval_contract_sha256"] = _canonical_sha256(contracts[0])
    return result


def summarize_evidence(
    inputs: Sequence[str | Path],
    *,
    parent_eval: str | Path | None = None,
    artifact_roots: Sequence[str | Path] = (),
    examples_per_category: int = 2,
) -> dict[str, Any]:
    """Load, validate, and summarize one or more Qwen codebook sweep tags."""

    if not inputs:
        raise EvidenceSummaryError("At least one evidence input is required.")
    evidence = [_load_evidence(value) for value in inputs]
    tags = [item.tag for item in evidence]
    if len(tags) != len(set(tags)):
        raise EvidenceSummaryError("Evidence inputs contain duplicate tags.")
    evidence.sort(key=lambda item: item.tag)
    explicit_parent = Path(parent_eval) if parent_eval is not None else None
    parent_fingerprint, parent_artifact_sha256, parent_aggregate = (
        _attach_parent_evaluations(evidence, explicit_parent)
    )
    contract, contract_fingerprint = _verify_contracts(evidence)
    roots = tuple(Path(value) for value in artifact_roots)
    candidates = [
        _candidate_summary(
            item,
            parent_fingerprint,
            contract_fingerprint,
            roots,
            examples_per_category,
        )
        for item in evidence
    ]
    parent_summary: dict[str, Any] = {
        "fingerprint": parent_fingerprint,
        "aggregate": parent_aggregate,
        "eval_contract_fingerprint": contract_fingerprint,
        "eval_contract": contract,
    }
    if parent_artifact_sha256 is not None:
        parent_summary["artifact_sha256"] = parent_artifact_sha256
    return {
        "schema": SUMMARY_SCHEMA,
        "candidate_count": len(candidates),
        "parent": parent_summary,
        "candidates": candidates,
    }


summarize = summarize_evidence


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:.6g}"
    return "-"


def _short_text(value: Any, limit: int = 96) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_text(summary: Mapping[str, Any]) -> str:
    """Render the canonical summary as a concise deterministic text report."""

    parent = _mapping(summary.get("parent")) or {}
    lines = [
        f"Qwen codebook sweep: {summary.get('candidate_count', 0)} candidate(s)",
        f"Parent: {parent.get('fingerprint', '-')}",
        f"Eval contract: {parent.get('eval_contract_fingerprint', '-')}",
    ]
    for candidate_value in _sequence(summary.get("candidates")):
        if not isinstance(candidate_value, Mapping):
            continue
        parameters = _mapping(candidate_value.get("parameters")) or {}
        lines.append("")
        lines.append(f"[{candidate_value.get('tag', '-')}]")
        parameter_line = (
            f"params {_format_number(parameters.get('source_parameter_count'))} -> "
            f"{_format_number(parameters.get('unique_child_parameter_count'))}"
        )
        if parameters.get("raw_compression_ratio") is not None:
            parameter_line += f"; raw {parameters['raw_compression_ratio']:.6f}x"
        if parameters.get("adjusted_ratio") is not None:
            parameter_line += f"; adjusted {parameters['adjusted_ratio']:.6f}x"
        lines.append(parameter_line)
        evaluation = _mapping(candidate_value.get("evaluation")) or {}
        suites = _mapping(evaluation.get("suites")) or {}
        for suite in _SUITES:
            values = _mapping(suites.get(suite))
            if not values:
                continue
            parent_values = _mapping(values.get("parent")) or {}
            child_values = _mapping(values.get("candidate")) or {}
            retention = _mapping(values.get("retention")) or {}
            lines.append(
                f"{suite}: {_format_number(parent_values.get('passes'))}/"
                f"{_format_number(parent_values.get('total'))} -> "
                f"{_format_number(child_values.get('passes'))}/"
                f"{_format_number(child_values.get('total'))}; retained "
                f"{_format_number(retention.get('retained_parent_passes'))}, "
                f"loss { _format_number(retention.get('losses'))}, "
                f"gain {_format_number(retention.get('gains'))}"
            )
        perplexity = _mapping(evaluation.get("perplexity"))
        if perplexity:
            lines.append(
                "PPL: "
                f"{_format_number(perplexity.get('parent_mean'))} -> "
                f"{_format_number(perplexity.get('candidate_mean'))}"
            )
        drift = _mapping(evaluation.get("logit_drift"))
        if drift:
            fields = [
                f"{key}={_format_number(drift[key])}"
                for key in (
                    "maximum_absolute_logit_error",
                    "mean_absolute_logit_error",
                    "relative_l2_logit_error",
                )
                if drift.get(key) is not None
            ]
            lines.append(f"logit drift ({drift.get('source', '-')}): " + ", ".join(fields))
        categories = _mapping(evaluation.get("categories")) or {}
        for suite in _SUITES:
            suite_categories = _mapping(categories.get(suite))
            if not suite_categories:
                continue
            deltas = [
                f"{name}:{values['pass_delta']:+}"
                for name, values in sorted(suite_categories.items())
                if isinstance(values, Mapping) and values.get("pass_delta") is not None
            ]
            if deltas:
                lines.append(f"{suite} category deltas: " + ", ".join(deltas))
        heuristics = _mapping(candidate_value.get("output_heuristics")) or {}
        heuristic_parts = []
        for name in ("repetition", "language_switch", "empty_output"):
            values = _mapping(heuristics.get(name)) or {}
            heuristic_parts.append(f"{name}={values.get('flagged', 0)}")
        lines.append(
            f"heuristics over {heuristics.get('outputs_analyzed', 0)} outputs: "
            + ", ".join(heuristic_parts)
        )
        artifact_verification = _mapping(candidate_value.get("artifact_verification")) or {}
        lines.append(f"artifacts: {artifact_verification.get('status', 'not-declared')}")
        manual = _mapping(candidate_value.get("manual_examples")) or {}
        for category in TARGET_MANUAL_CATEGORIES:
            examples = _sequence(manual.get(category))
            if not examples:
                continue
            example = examples[0]
            if isinstance(example, Mapping):
                lines.append(
                    f"manual {category}/{example.get('outcome', '-')}: "
                    f"{example.get('id', '-')} -> "
                    f"{_short_text(example.get('candidate_output', ''))}"
                )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Run directories, run_report.json files, or comparison JSON files.",
    )
    parser.add_argument(
        "--parent-eval",
        type=Path,
        help="Explicit cached parent eval used by every candidate.",
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        type=Path,
        help="Additional root containing compact checkpoint artifacts.",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=2,
    )
    parser.add_argument("--format", choices=("json", "text"), default="text")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = summarize_evidence(
            args.inputs,
            parent_eval=args.parent_eval,
            artifact_roots=args.artifact_root,
            examples_per_category=args.examples_per_category,
        )
        output = canonical_json(summary) if args.format == "json" else render_text(summary)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(output)
    except EvidenceSummaryError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
