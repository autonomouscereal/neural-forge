#!/usr/bin/env python3
"""Evaluate the independent 64-case LFM2 holdout from cached outputs only.

This command never imports a model loader, executes a model, trains, or calls
the parent.  It authenticates the one-time parent envelope, a candidate output
artifact/report, and the bound local tokenizer files before delegating semantic
and task scoring to ``hf_lfm2_intelligence_gate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_retention as posthoc

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_intelligence_gate import (
    IntelligenceGatePolicy,
    build_manual_review_packet,
    evaluate_cached_intelligence,
    load_bound_manual_ratings,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    CANDIDATE_OUTPUT_SCHEMA,
    baseline_from_lfm2_intelligence_holdout_envelope,
    candidate_outputs_from_lfm2_holdout_rows,
    load_lfm2_intelligence_holdout_candidate_outputs,
    load_lfm2_intelligence_holdout_parent_envelope,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RUNNER_SCHEMA = "neural-forge-lfm2-intelligence-holdout-evaluation-v1"


class Lfm2HoldoutEvaluationError(RuntimeError):
    """Cached evidence cannot support a strict holdout decision."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2HoldoutEvaluationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _safe_input_file(value: str | os.PathLike[str], name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2HoldoutEvaluationError(f"{name} does not exist.") from exc
    if not path.is_file() or path.is_symlink():
        raise Lfm2HoldoutEvaluationError(
            f"{name} is missing or unsafe: {path}."
        )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_file(
    value: str | os.PathLike[str],
    *,
    name: str,
    expected_sha256: str,
) -> tuple[Path, Any]:
    path = _safe_input_file(value, name)
    raw = path.read_bytes()
    if _file_sha256(path) != _digest(
        expected_sha256,
        f"expected_{name.replace(' ', '_')}_sha256",
    ):
        raise Lfm2HoldoutEvaluationError(
            f"{name} fingerprint differs from its expected SHA-256."
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2HoldoutEvaluationError(
            f"{name} is not valid UTF-8 JSON."
        ) from exc
    return path, document


def _stage_new_json(
    path: Path,
    value: Mapping[str, Any],
) -> Path:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Lfm2HoldoutEvaluationError(
            f"Refusing to replace existing output {destination}."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def _publish_new_documents(
    documents: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    destinations = tuple(path.expanduser().absolute() for path, _ in documents)
    if len(set(destinations)) != len(destinations):
        raise Lfm2HoldoutEvaluationError(
            "Output destinations must be distinct."
        )
    for destination in destinations:
        if destination.exists():
            raise Lfm2HoldoutEvaluationError(
                f"Refusing to replace existing output {destination}."
            )
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, (_, document) in zip(destinations, documents):
            staged.append((destination, _stage_new_json(destination, document)))
        for destination, temporary in staged:
            os.link(temporary, destination)
            published.append(destination)
    except BaseException:
        for destination in published:
            if destination.exists():
                destination.unlink()
        raise
    finally:
        for _destination, temporary in staged:
            if temporary.exists():
                temporary.unlink()


def _raw_candidate_rows(
    document: Any,
    *,
    candidate_phase: str | None,
    manifest: Any,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    rows, selection = posthoc.extract_candidate_output_rows(
        document,
        candidate_phase=candidate_phase,
    )
    normalized: list[Mapping[str, Any]] = []
    for row in rows:
        case_id = row.get("case_id")
        spec = (
            manifest.case_map.get(case_id)
            if isinstance(case_id, str)
            else None
        )
        if spec is None:
            raise Lfm2HoldoutEvaluationError(
                "Candidate report contains a case outside the holdout."
            )
        if row.get("generation_limit") != spec.generation_limit:
            raise Lfm2HoldoutEvaluationError(
                f"Candidate {case_id} lacks its exact manifest generation limit."
            )
        topical = row.get("topical_relevance")
        if topical is None:
            topical = row.get("task_relevant")
        normalized.append(
            {
                "case_id": case_id,
                "output": row.get("output"),
                "output_token_ids": row.get("output_token_ids"),
                "generation_limit": row.get("generation_limit"),
                "topical_relevance": topical,
            }
        )
    return normalized, {
        **selection,
        "source_kind": "nested_candidate_report",
        "sealed_candidate_outputs_sha256": None,
        "candidate_state_fingerprint": None,
    }


@dataclass(frozen=True, slots=True)
class HoldoutEvaluationConfig:
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    expected_parent_envelope_sha256: str
    expected_manifest_sha256: str
    expected_parent_state_fingerprint: str
    expected_parent_model_fingerprint: str
    tokenizer_directory: Path
    expected_tokenizer_fingerprint: str
    candidate_report: Path
    expected_candidate_report_sha256: str
    report_path: Path
    manual_review_packet_path: Path
    candidate_phase: str | None = None
    expected_candidate_outputs_sha256: str | None = None
    expected_candidate_state_fingerprint: str | None = None
    manual_ratings_path: Path | None = None
    expected_manual_ratings_file_sha256: str | None = None
    expected_manual_ratings_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "expected_parent_envelope_file_sha256",
            "expected_parent_envelope_sha256",
            "expected_manifest_sha256",
            "expected_parent_state_fingerprint",
            "expected_parent_model_fingerprint",
            "expected_tokenizer_fingerprint",
            "expected_candidate_report_sha256",
        ):
            _digest(getattr(self, name), name)
        candidate_bindings = (
            self.expected_candidate_outputs_sha256,
            self.expected_candidate_state_fingerprint,
        )
        if any(value is None for value in candidate_bindings) != all(
            value is None for value in candidate_bindings
        ):
            raise Lfm2HoldoutEvaluationError(
                "Candidate output seal and state fingerprint must be supplied "
                "together."
            )
        for name, value in (
            (
                "expected_candidate_outputs_sha256",
                self.expected_candidate_outputs_sha256,
            ),
            (
                "expected_candidate_state_fingerprint",
                self.expected_candidate_state_fingerprint,
            ),
        ):
            if value is not None:
                _digest(value, name)
        manual_bindings = (
            self.manual_ratings_path,
            self.expected_manual_ratings_file_sha256,
            self.expected_manual_ratings_sha256,
        )
        if any(value is None for value in manual_bindings) != all(
            value is None for value in manual_bindings
        ):
            raise Lfm2HoldoutEvaluationError(
                "Manual ratings path, file SHA-256, and ratings seal must be "
                "supplied together."
            )
        for name, value in (
            (
                "expected_manual_ratings_file_sha256",
                self.expected_manual_ratings_file_sha256,
            ),
            (
                "expected_manual_ratings_sha256",
                self.expected_manual_ratings_sha256,
            ),
        ):
            if value is not None:
                _digest(value, name)
        report = Path(self.report_path).expanduser().absolute()
        packet = Path(self.manual_review_packet_path).expanduser().absolute()
        if report == packet:
            raise Lfm2HoldoutEvaluationError(
                "Report and manual packet destinations must differ."
            )
        for destination in (report, packet):
            if destination.exists():
                raise Lfm2HoldoutEvaluationError(
                    f"Refusing to replace existing output {destination}."
                )
        object.__setattr__(self, "report_path", report)
        object.__setattr__(self, "manual_review_packet_path", packet)


def _load_candidates(
    config: HoldoutEvaluationConfig,
    *,
    manifest: Any,
    eos_token_ids: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    _candidate_path, document = _read_json_file(
        config.candidate_report,
        name="candidate report",
        expected_sha256=config.expected_candidate_report_sha256,
    )
    is_sealed = (
        isinstance(document, Mapping)
        and document.get("schema") == CANDIDATE_OUTPUT_SCHEMA
    )
    if is_sealed:
        if (
            config.expected_candidate_outputs_sha256 is None
            or config.expected_candidate_state_fingerprint is None
        ):
            raise Lfm2HoldoutEvaluationError(
                "A sealed candidate artifact requires its expected inner seal "
                "and candidate-state fingerprint."
            )
        evidence = load_lfm2_intelligence_holdout_candidate_outputs(
            document,
            expected_candidate_outputs_sha256=(
                config.expected_candidate_outputs_sha256
            ),
            expected_manifest_sha256=config.expected_manifest_sha256,
            expected_candidate_state_fingerprint=(
                config.expected_candidate_state_fingerprint
            ),
            expected_tokenizer_fingerprint=(
                config.expected_tokenizer_fingerprint
            ),
            eos_token_ids=eos_token_ids,
        )
        return evidence.candidates, {
            "source_kind": "sealed_candidate_outputs",
            "selected_event_path": "$.rows",
            "selected_phase": None,
            "selected_event_outputs_sha256": canonical_sha256(
                [dict(row) for row in evidence.rows]
            ),
            "sealed_candidate_outputs_sha256": (
                evidence.candidate_outputs_sha256
            ),
            "candidate_state_fingerprint": (
                evidence.candidate_state_fingerprint
            ),
        }
    if (
        config.expected_candidate_outputs_sha256 is not None
        or config.expected_candidate_state_fingerprint is not None
    ):
        raise Lfm2HoldoutEvaluationError(
            "Expected sealed-candidate bindings were supplied for an unsealed "
            "candidate report."
        )
    rows, selection = _raw_candidate_rows(
        document,
        candidate_phase=config.candidate_phase,
        manifest=manifest,
    )
    candidates = candidate_outputs_from_lfm2_holdout_rows(
        manifest,
        rows,
        eos_token_ids=eos_token_ids,
    )
    return candidates, selection


def evaluate_lfm2_intelligence_holdout(
    config: HoldoutEvaluationConfig,
) -> dict[str, Any]:
    """Score only authenticated cached evidence and publish immutable reports."""

    if not isinstance(config, HoldoutEvaluationConfig):
        raise TypeError("config must be HoldoutEvaluationConfig.")
    if config.report_path.exists() or config.manual_review_packet_path.exists():
        raise Lfm2HoldoutEvaluationError(
            "An output appeared after configuration; refusing overwrite."
        )
    _parent_path, parent_document = _read_json_file(
        config.parent_envelope,
        name="parent envelope",
        expected_sha256=config.expected_parent_envelope_file_sha256,
    )
    envelope = load_lfm2_intelligence_holdout_parent_envelope(
        parent_document,
        expected_envelope_sha256=config.expected_parent_envelope_sha256,
        expected_manifest_sha256=config.expected_manifest_sha256,
        expected_parent_state_fingerprint=(
            config.expected_parent_state_fingerprint
        ),
        expected_parent_model_fingerprint=(
            config.expected_parent_model_fingerprint
        ),
        expected_tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
    )
    special_identity = posthoc.load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
    )
    eos_token_ids = special_identity["special_token_ids"]["eos"]
    if tuple(eos_token_ids) != envelope.eos_token_ids:
        raise Lfm2HoldoutEvaluationError(
            "Current bound tokenizer EOS IDs differ from the parent envelope."
        )
    candidates, candidate_selection = _load_candidates(
        config,
        manifest=envelope.manifest,
        eos_token_ids=eos_token_ids,
    )
    baseline = baseline_from_lfm2_intelligence_holdout_envelope(envelope)
    policy = IntelligenceGatePolicy()
    packet = build_manual_review_packet(baseline, candidates, policy)
    manual_ratings = None
    if config.manual_ratings_path is not None:
        _ratings_path, ratings_document = _read_json_file(
            config.manual_ratings_path,
            name="manual ratings",
            expected_sha256=str(
                config.expected_manual_ratings_file_sha256
            ),
        )
        manual_ratings = load_bound_manual_ratings(
            ratings_document,
            expected_packet_sha256=packet["packet_sha256"],
            expected_ratings_sha256=str(
                config.expected_manual_ratings_sha256
            ),
        )
    gate = evaluate_cached_intelligence(
        baseline,
        candidates,
        policy=policy,
        manual_ratings=manual_ratings,
    )
    gate_report = gate.as_dict()
    components = {
        str(item["name"]): item for item in gate_report["components"]
    }
    manual_status = str(gate_report["manual_review"]["status"])
    if manual_status == "pending":
        claim = {
            "status": "withheld_pending_manual_review",
            "certified_lower_bound": None,
            "warning": (
                "No intelligence-retention headline is valid until all six "
                "open-ended cases receive packet-bound semantic review."
            ),
        }
    elif gate.promoted and gate.headline_lower_bound is not None:
        claim = {
            "status": "certified_conservative_lower_bound",
            "certified_lower_bound": gate.headline_lower_bound,
            "warning": (
                "This is the minimum required parent-relative task, capability, "
                "behavioral, and manual retention component; it is not exact "
                "parent wording or token matching."
            ),
        }
    else:
        claim = {
            "status": "rejected",
            "certified_lower_bound": None,
            "warning": (
                "A score is not presented as intelligence retention when any "
                "required component fails."
            ),
        }
    task = components["task_correctness"]
    capability = components["parent_capability"]
    core = {
        "schema": RUNNER_SCHEMA,
        "status": "promoted" if gate.promoted else "rejected",
        "promoted": gate.promoted,
        "reasons": list(gate.reasons),
        "intelligence_retention_claim": claim,
        "diagnostic_lower_bound": gate.headline_lower_bound,
        "parent_relative_task_retention": {
            "eligible_parent_correct_count": task["eligible_count"],
            "retained_count": task["success_count"],
            "rate": task["retention"],
            "minimum_rate": policy.minimum_task_correctness_retention,
            "candidate_absolute_accuracy": task["details"][
                "candidate_absolute_accuracy"
            ],
            "parent_absolute_accuracy": task["details"][
                "parent_absolute_accuracy"
            ],
            "absolute_accuracy_retention": task["details"][
                "candidate_absolute_accuracy_retention"
            ],
            "minimum_absolute_accuracy_ratio_to_parent": (
                policy.minimum_absolute_task_correctness
            ),
            "parent_relative_not_exact_output": True,
        },
        "parent_demonstrated_capability_retention": {
            "eligible_count": capability["eligible_count"],
            "retained_count": capability["success_count"],
            "rate": capability["retention"],
            "minimum_rate": policy.minimum_parent_capability_retention,
            "contracts_are_parent_demonstrated": True,
        },
        "core_intelligence_gate": gate_report,
        "manual_review": {
            **dict(gate_report["manual_review"]),
            "packet_sha256": packet["packet_sha256"],
            "packet_case_ids": [
                str(item["case_id"]) for item in packet["cases"]
            ],
            "exact_parent_wording_criterion": False,
        },
        "policy": {
            **policy.as_dict(),
            "policy_sha256": policy.fingerprint,
            "all_required_component_floor_is_0_90": True,
            "strict_regression_limits_preserved": True,
        },
        "inputs": {
            "manifest_sha256": envelope.manifest.manifest_sha256,
            "manifest_document_sha256": envelope.manifest.document_sha256,
            "case_count": len(envelope.manifest.cases),
            "parent_envelope_file_sha256": (
                config.expected_parent_envelope_file_sha256
            ),
            "parent_envelope_sha256": envelope.envelope_sha256,
            "parent_cache_key": envelope.parent_cache_key,
            "parent_state_fingerprint": envelope.parent_state_fingerprint,
            "parent_model_fingerprint": envelope.parent_model_fingerprint,
            "tokenizer_fingerprint": envelope.tokenizer_fingerprint,
            "candidate_report_file_sha256": (
                config.expected_candidate_report_sha256
            ),
            **candidate_selection,
        },
        "bound_special_tokens": special_identity,
        "execution_audit": {
            "parent_model_loaded": False,
            "candidate_model_loaded": False,
            "parent_forward_passes": 0,
            "parent_generate_invocations": 0,
            "candidate_forward_passes": 0,
            "candidate_generate_invocations": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "network_access": False,
            "hub_loader_used": False,
            "candidate_python_exec_used": False,
            "cached_outputs_only": True,
        },
        "semantic_evaluation": {
            "exact_parent_wording_used": False,
            "exact_parent_token_matching_used": False,
            "safe_restricted_ast_for_python": True,
            "manual_packet_required_for_open_ended_claim": True,
        },
    }
    report = {**core, "report_sha256": canonical_sha256(core)}
    _publish_new_documents(
        (
            (config.manual_review_packet_path, packet),
            (config.report_path, report),
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the independent LFM2 holdout from cached parent and "
            "candidate outputs, with zero model calls."
        )
    )
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-parent-state-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--expected-parent-model-fingerprint",
        required=True,
    )
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--expected-candidate-report-sha256", required=True)
    parser.add_argument("--candidate-phase")
    parser.add_argument("--expected-candidate-outputs-sha256")
    parser.add_argument("--expected-candidate-state-fingerprint")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--manual-review-packet-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--manual-ratings-path", type=Path)
    parser.add_argument("--expected-manual-ratings-file-sha256")
    parser.add_argument("--expected-manual-ratings-sha256")
    return parser


def config_from_args(args: argparse.Namespace) -> HoldoutEvaluationConfig:
    return HoldoutEvaluationConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    report = evaluate_lfm2_intelligence_holdout(config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted": report["promoted"],
                "report_sha256": report["report_sha256"],
                "manual_review": report["manual_review"]["status"],
                "certified_lower_bound": report[
                    "intelligence_retention_claim"
                ]["certified_lower_bound"],
                "case_count": report["inputs"]["case_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
