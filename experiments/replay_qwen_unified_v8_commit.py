#!/usr/bin/env python3
"""Replay and save the already-approved unified-v8 neuron transaction.

This is deliberately not a discovery or evaluation runner.  It verifies the
immutable v8 evidence bundle, replays its seven exact real-medoid redirects on
the immutable p35 checkpoint, requires the original proposed registry and full
accounting, then saves, reloads, and smoke-tests the parent-free child.

Hash-domain note: ``dffa7190...`` is the allocator/gate candidate-state hash,
not a model-content hash.  The failed run computed model content in memory but
failed before persisting that digest.  This replay therefore binds ``dffa...``
to the original gate/transaction proof, requires the exact executable registry
and accounting, computes a fresh model-content digest, and requires reload
equality for that digest.
"""

from __future__ import annotations

import argparse
import errno
import gc
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge import hf_qwen_adaptive_surgery as neuron_surgery
from neural_forge import hf_qwen_attention_surgery as attention_surgery
from neural_forge import hf_qwen_surgery_allocator as surgery_allocator
from neural_forge.adaptive_surgery_graph import CandidateScore, MergeCandidate
from neural_forge.hf_qwen_codebook_model import (
    load_qwen3_codebook_checkpoint,
    plan_qwen3_codebook_artifact,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
from neural_forge.hf_qwen_surgery_allocator import (
    Qwen3CandidateValidationDamage,
    Qwen3MixedSurgeryBatch,
    Qwen3UnifiedSurgeryAccounting,
    Qwen3UnifiedSurgeryProposal,
)


SCHEMA = "neural-forge-qwen3-unified-v8-exact-replay-v1"
MODEL_CONTENT_SCHEMA = "neural-forge-qwen3-model-content-v1"
INITIAL_CONTENT_SHA256 = (
    "27dc554fb89f968c89c43f5c80e36e6ebfb725be2ba0da5667fa3cdcfd0b52ea"
)
GATE_CANDIDATE_STATE_SHA256 = (
    "dffa7190b43bd64554ccfc1943c173d96f9d43c00dc46d82693de7a168c88a34"
)
PARENT_ALLOCATOR_STATE_SHA256 = (
    "4a33405d1d671389aff21551b71995a20e29be37ae293e151780bf3ae2c67b34"
)
EXPECTED_NEURON_REGISTRY_SHA256 = (
    "f6d9e207d0b13b51f90852dcb761f70b4803661320d0d0b1d1843c84ff0ce763"
)
EXPECTED_CHILD_UNIQUE_PARAMETERS = 3_995_516_416
EXPECTED_DEPLOYABLE_BYTES = 7_994_478_724
EXPECTED_BATCH_SHA256 = (
    "a1a6e3a5c85339c478b2206fb844f52eb5dbc9b9727db92253efe8eaea5ed673"
)
EXPECTED_SELECTED_VALIDATED_PROPOSAL_SHA256S = (
    "15c94112a7f48cd135882521400d445b8c8ae79ebbea6738bbe06d506b4b316b",
    "4acb6ee41f62009d316347ec93cbb88d0d7587d80920815ea4cf8f768643c7c6",
    "7d771a7e6805c7ad1516790ccd834d6a8659b68506756121d884d2308cec9916",
    "80ee8523e107f9f717dc98d93f42c7770b56c7cce00fa76624ea21659b994d97",
    "c8cd3f0a2de383cc9d862fcd3d8bd2469cb50f9a267fbdfa197425764eba9f6d",
    "efacdb8d92f8a4ad5e824e317b074443911c6816f4466acf1690bf1dbe3a1520",
    "f452c7518e284ed4c9264e8ff79d1da9b71f97a8f4f8485eeb45fd51556f83e1",
)

# Exact semantic transaction requested by the accepted batch.  Nodes are
# (layer, removed neuron, survivor/medoid neuron).
V8_REDIRECTS = (
    (1, 5523, 5394),
    (1, 7446, 6035),
    (1, 7883, 7564),
    (1, 6398, 6397),
    (1, 8911, 8017),
    (1, 8431, 7120),
    (1, 9286, 8654),
)

EVIDENCE_FILE_SHA256S = {
    "attempt-000001.json": "e5dbfe4b3d6bf3b44adf9a610921db6e1618c86b3bfcd0bde06a175f35b96e01",
    "candidate-000001-dffa7190b43bd64554ccfc1943c173d96f9d43c00dc46d82693de7a168c88a34.gate.json": "ec087c4b5d80467d28f84a68647b6a110d2ce167279b8b7e5f14b38dc1f6687c",
    "candidate-000001-dffa7190b43bd64554ccfc1943c173d96f9d43c00dc46d82693de7a168c88a34.json": "500a45d675077a6a3cbd489525a3d677cb43bcd014c45276c6b6f379ee63cb3e",
    "candidate-0001-neuron-68bda91a777ef22df0f355e33dac070370bd03af5dbf95a3ea90a6e4e13404e0.json": "75a9cf1712e05366acbd512e9e36a9563122e239c2aec12f1f628ec06e61f85a",
    "candidate-0002-neuron-7dfa5216517bdfd947a3be62178bd147bdc103027619e7d61dc1e028a25a67a6.json": "9faa7dddf53e7ce33c36240e44092348058779ebb32fd9785722d6d6a478a5e2",
    "candidate-0003-neuron-45337f2766389f0ab5bbd751e72625c17fbee1fd9d7d4c7a2d65fbd19dd6148b.json": "6df4cbe95b893baf306ac687d0126fce2c8ceabe76aa179499989e460fe63d05",
    "candidate-0004-neuron-cb78157cf9e1b717868a0d7c0ed1189912194f43abaf484afaab8d2799a4598b.json": "91c905d561dc73196ba44f5fc9d204773dbf30bc2ab2c098fc0b91eb63e5f921",
    "candidate-0005-neuron-132272852a684feef866a4d76e5cbe756e9d254d89305fc84979a6565f7084ad.json": "02663c1c35e61e59e9058c45ec80a0df67513e5ad81311e8652d92ef975d683b",
    "candidate-0006-neuron-6b0375b602c0e4c97ec4a603297f216f472362f86357f0a698498d52d51bbc52.json": "6d098d72a7b4240faa809a982b7aa1a5934b1d087d90583d41a6591eba005a83",
    "candidate-0007-neuron-a53ec53df978d888bd6ddcde16ec5d25f0c3c322dee71f5a8e640fae8457afee.json": "9fa1d82db21acd24ead23fe8384ac5f853f8457cbe370939f8e3bf688f2941e1",
    "candidate-0008-neuron-5b69ff813ed02696b6422560ca1b47b66cfcee2f131406ba8d0cc8c1dd6a5b0c.json": "cee41779fef83b9ae75e0079c2cc430007a706f5137bfb3be19ae6e477d1cc6f",
    "initial_incumbent.json": "48d8716c75ff8c50e43abe95f19df994833e5660d62a35041e62bc771ef52804",
    "transactions.jsonl": "bc6b4ad92bc466ea136e1fbc99f06f76486f7f0d9c6ef79e88643ff8b4042d14",
    "validation_summary.json": "bf1b25c65d0b9f65ec0fb26feacba40ba3f35d75082058eee586994c87d15493",
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INCOMPLETE_PROMOTION_MARKER = ".neural-forge-promotion-incomplete"


class QwenUnifiedV8ReplayError(RuntimeError):
    """The exact v8 replay cannot prove an invariant."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QwenUnifiedV8ReplayError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise QwenUnifiedV8ReplayError(f"Non-finite JSON value {value!r}.")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QwenUnifiedV8ReplayError(f"Unreadable JSON artifact: {path}.") from exc
    if not isinstance(value, dict):
        raise QwenUnifiedV8ReplayError(f"JSON artifact is not an object: {path}.")
    return value


def _record_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256(payload)


def _mapping_diff(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return an exact, bounded field diff without weakening equality checks."""

    result: dict[str, dict[str, Any]] = {}
    for name in sorted(set(expected) | set(observed)):
        expected_value = expected.get(name, "<missing>")
        observed_value = observed.get(name, "<missing>")
        if expected_value != observed_value or type(expected_value) is not type(
            observed_value
        ):
            result[name] = {
                "expected": expected_value,
                "observed": observed_value,
                "expected_type": type(expected_value).__name__,
                "observed_type": type(observed_value).__name__,
            }
    return result


def _basename(value: Any) -> str:
    return PurePosixPath(str(value).replace("\\", "/")).name


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise QwenUnifiedV8ReplayError(f"{name} is not lowercase SHA-256.")
    return value


def _decode_tagged(value: Any) -> Any:
    if not isinstance(value, Mapping) or set(value) != {"type", "value"}:
        raise QwenUnifiedV8ReplayError("Candidate node ID is not canonically tagged.")
    kind = value["type"]
    raw = value["value"]
    if kind == "tuple":
        if not isinstance(raw, list):
            raise QwenUnifiedV8ReplayError("Tagged tuple payload is invalid.")
        return tuple(_decode_tagged(item) for item in raw)
    if kind == "str" and isinstance(raw, str):
        return raw
    if kind == "int" and isinstance(raw, str):
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise QwenUnifiedV8ReplayError("Tagged integer payload is invalid.") from exc
        if str(parsed) != raw:
            raise QwenUnifiedV8ReplayError("Tagged integer is not canonical.")
        return parsed
    raise QwenUnifiedV8ReplayError(f"Unsupported tagged candidate value {kind!r}.")


def _candidate_from_payload(value: Mapping[str, Any]) -> MergeCandidate:
    score_raw = value.get("score")
    if not isinstance(score_raw, Mapping):
        raise QwenUnifiedV8ReplayError("Candidate score is missing.")
    score = CandidateScore(
        **{field.name: score_raw[field.name] for field in fields(CandidateScore)}
    )
    if value.get("score_sha256") != score.score_sha256:
        raise QwenUnifiedV8ReplayError("Candidate score SHA-256 is invalid.")
    sequence_names = (
        "left_members",
        "right_members",
        "member_ids",
        "medoid_candidates_considered",
    )
    try:
        candidate = MergeCandidate(
            parent_snapshot_sha256=value["parent_snapshot_sha256"],
            evidence_sha256=value["evidence_sha256"],
            left_prototype=_decode_tagged(value["left_prototype"]),
            right_prototype=_decode_tagged(value["right_prototype"]),
            resulting_medoid=_decode_tagged(value["resulting_medoid"]),
            exact_anchor=(
                None
                if value.get("exact_anchor") is None
                else _decode_tagged(value["exact_anchor"])
            ),
            score=score,
            **{
                name: tuple(_decode_tagged(item) for item in value[name])
                for name in sequence_names
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenUnifiedV8ReplayError("Candidate payload is invalid.") from exc
    if candidate.as_dict() != dict(value):
        raise QwenUnifiedV8ReplayError("Candidate payload is not canonical.")
    return candidate


def _proposal_from_payload(
    value: Mapping[str, Any], candidate: MergeCandidate
) -> Qwen3UnifiedSurgeryProposal:
    try:
        kwargs = {
            field.name: value[field.name]
            for field in fields(Qwen3UnifiedSurgeryProposal)
            if field.name != "candidate"
        }
        protected = kwargs.get("protected_anchor")
        if protected is not None:
            kwargs["protected_anchor"] = tuple(protected)
        proposal = Qwen3UnifiedSurgeryProposal(candidate=candidate, **kwargs)
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenUnifiedV8ReplayError("Proposal payload is invalid.") from exc
    if proposal.as_dict() != dict(value):
        raise QwenUnifiedV8ReplayError("Proposal payload is not canonical.")
    return proposal


def _validation_record(value: Mapping[str, Any]) -> Qwen3CandidateValidationDamage:
    try:
        record = Qwen3CandidateValidationDamage(
            **{
                field.name: value[field.name]
                for field in fields(Qwen3CandidateValidationDamage)
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenUnifiedV8ReplayError("Validation record is invalid.") from exc
    if record.as_dict() != dict(value):
        raise QwenUnifiedV8ReplayError("Validation record is not canonical.")
    return record


def _validated_proposal(
    proposal: Qwen3UnifiedSurgeryProposal,
    record: Qwen3CandidateValidationDamage,
) -> Qwen3UnifiedSurgeryProposal:
    physical = proposal.individual_physical_bytes_saved
    return replace(
        proposal,
        measured_heldout_damage=record.marginal_damage,
        risk=record.marginal_damage,
        risk_per_individual_physical_byte=(
            record.marginal_damage / physical
            if physical > 0
            else record.marginal_damage + proposal.scale_setup_bytes + 1.0
        ),
        validation_parent_state_sha256=record.parent_state_sha256,
        validation_provenance_sha256=record.provenance_sha256,
    )


def _behavior_outputs(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "perplexity": report.get("perplexity"),
        "choice_accuracy": report.get("choice_accuracy"),
        "generation_diagnostics": report.get("generation_diagnostics"),
        "generation_health": report.get("generation_health"),
        "behavior_suite": report.get("behavior_suite"),
    }


def _behavior_score(report: Mapping[str, Any]) -> tuple[float, int]:
    correct = 0
    total = 0
    for payload, key in (
        (report.get("choice_accuracy"), "correct"),
        (report.get("generation_health"), "correct"),
        (report.get("behavior_suite"), "correct"),
    ):
        if not isinstance(payload, Mapping):
            raise QwenUnifiedV8ReplayError("Evaluation report domain is missing.")
        value, size = payload.get(key), payload.get("total")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= value <= size
        ):
            raise QwenUnifiedV8ReplayError("Evaluation count is invalid.")
        correct += value
        total += size
    if total <= 0:
        raise QwenUnifiedV8ReplayError("Evaluation report is empty.")
    return correct / total, total


@dataclass(frozen=True, slots=True)
class V8EvidenceProof:
    proposed_accounting: dict[str, Any]
    selected_proposal_sha256s: tuple[str, ...]
    selected_candidate_sha256s: tuple[str, ...]
    redirects: tuple[tuple[int, int, int], ...]
    evidence_file_sha256s: dict[str, str]


def validate_v8_evidence_bundle(directory: str | Path) -> V8EvidenceProof:
    """Cryptographically and semantically verify the flattened v8 proof bundle."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise QwenUnifiedV8ReplayError(f"Evidence directory does not exist: {root}.")
    observed: dict[str, str] = {}
    for name, expected in EVIDENCE_FILE_SHA256S.items():
        path = root / name
        if not path.is_file():
            raise QwenUnifiedV8ReplayError(f"Required evidence artifact is missing: {name}.")
        actual = _file_sha256(path)
        if actual != expected:
            raise QwenUnifiedV8ReplayError(f"Evidence SHA-256 mismatch: {name}.")
        observed[name] = actual

    initial = _read_json(root / "initial_incumbent.json")
    if (
        initial.get("schema") != "neural-forge-qwen3-initial-incumbent-v1"
        or initial.get("record_sha256") != _record_sha256(initial, "record_sha256")
        or initial.get("model_content_sha256") != INITIAL_CONTENT_SHA256
        or initial.get("model_content", {}).get("model_content_sha256")
        != INITIAL_CONTENT_SHA256
    ):
        raise QwenUnifiedV8ReplayError("Initial-incumbent integrity is invalid.")
    run_contract = initial.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise QwenUnifiedV8ReplayError("Initial incumbent lacks its run contract.")
    contract_payload = dict(run_contract)
    binding = contract_payload.pop("binding_sha256", None)
    contract = contract_payload.pop("run_contract_sha256", None)
    if binding != _sha256(contract_payload) or contract != binding:
        raise QwenUnifiedV8ReplayError("Run-contract SHA-256 is invalid.")

    attempt = _read_json(root / "attempt-000001.json")
    records_raw = attempt.get("records")
    if (
        attempt.get("schema") != "neural-forge-unified-candidate-discovery-v1"
        or not isinstance(records_raw, list)
        or attempt.get("candidate_count") != len(records_raw)
        or attempt.get("manifest_sha256") != _record_sha256(attempt, "manifest_sha256")
    ):
        raise QwenUnifiedV8ReplayError("Candidate-discovery manifest is invalid.")
    proposals: dict[str, Qwen3UnifiedSurgeryProposal] = {}
    ordered_candidates: list[str] = []
    ordered_operations: list[str] = []
    ordered_proposals: list[str] = []
    candidate_by_sha: dict[str, MergeCandidate] = {}
    for index, item in enumerate(records_raw, start=1):
        if not isinstance(item, Mapping) or item.get("index") != index:
            raise QwenUnifiedV8ReplayError("Candidate-discovery order is invalid.")
        candidate_raw = item.get("candidate")
        proposal_raw = item.get("proposal")
        if not isinstance(candidate_raw, Mapping) or not isinstance(
            proposal_raw, Mapping
        ):
            raise QwenUnifiedV8ReplayError("Candidate-discovery record is incomplete.")
        candidate = _candidate_from_payload(candidate_raw)
        proposal = _proposal_from_payload(proposal_raw, candidate)
        if (
            item.get("kind") != proposal.kind
            or item.get("candidate_sha256") != candidate.candidate_sha256
            or item.get("candidate_operation_sha256") != candidate.operation_sha256
            or item.get("proposal_sha256") != proposal.proposal_sha256
        ):
            raise QwenUnifiedV8ReplayError("Candidate exact identity is invalid.")
        proposals[proposal.proposal_sha256] = proposal
        candidate_by_sha[candidate.candidate_sha256] = candidate
        ordered_candidates.append(candidate.candidate_sha256)
        ordered_operations.append(candidate.operation_sha256)
        ordered_proposals.append(proposal.proposal_sha256)
    if (
        len(proposals) != len(records_raw)
        or len(candidate_by_sha) != len(records_raw)
        or attempt.get("ordered_exact_candidate_sequence_sha256")
        != _sha256(ordered_candidates)
        or attempt.get("ordered_semantic_operation_sequence_sha256")
        != _sha256(ordered_operations)
        or attempt.get("ordered_proposal_sequence_sha256")
        != _sha256(ordered_proposals)
        or attempt.get("semantic_operation_set_sha256")
        != _sha256(sorted(ordered_operations))
        or attempt.get("exact_identity_required_for_validation") is not True
    ):
        raise QwenUnifiedV8ReplayError("Candidate discovery sequence hashes are invalid.")

    summary = _read_json(root / "validation_summary.json")
    validation_records = summary.get("records")
    audits = summary.get("audits")
    if (
        summary.get("schema")
        != "neural-forge-independent-candidate-validation-summary-v2"
        or summary.get("summary_sha256") != _record_sha256(summary, "summary_sha256")
        or summary.get("parent_state_sha256") != PARENT_ALLOCATOR_STATE_SHA256
        or summary.get("every_probe_rolled_back_exactly") is not True
        or not isinstance(validation_records, list)
        or not isinstance(audits, list)
        or len(validation_records) != 8
        or len(audits) != 8
        or summary.get("validated_candidate_count") != 8
    ):
        raise QwenUnifiedV8ReplayError("Candidate-validation summary is invalid.")
    validated_by_sha: dict[str, Qwen3UnifiedSurgeryProposal] = {}
    validation_initial_sequence: list[str] = []
    validation_final_sequence: list[str] = []
    for record_raw, audit in zip(validation_records, audits, strict=True):
        if not isinstance(record_raw, Mapping) or not isinstance(audit, Mapping):
            raise QwenUnifiedV8ReplayError("Candidate-validation entry is invalid.")
        record = _validation_record(record_raw)
        proposal_sha = _require_sha256(audit.get("proposal_sha256"), "proposal SHA")
        proposal = proposals.get(proposal_sha)
        if proposal is None:
            raise QwenUnifiedV8ReplayError("Validated proposal was not discovered.")
        validated = _validated_proposal(proposal, record)
        report_name = _basename(audit.get("artifact"))
        report_path = root / report_name
        if (
            record.kind != proposal.kind
            or record.candidate_sha256 != proposal.candidate.candidate_sha256
            or audit.get("candidate_sha256") != record.candidate_sha256
            or audit.get("candidate_operation_sha256")
            != proposal.candidate.operation_sha256
            or audit.get("kind") != record.kind
            or audit.get("rollback_exact") is not True
            or audit.get("validated_proposal_sha256") != validated.proposal_sha256
            or report_name not in EVIDENCE_FILE_SHA256S
            or audit.get("artifact_sha256") != observed[report_name]
        ):
            raise QwenUnifiedV8ReplayError("Candidate-validation binding is invalid.")
        candidate_report = _read_json(report_path)
        score, count = _behavior_score(candidate_report)
        outputs_sha = _sha256(_behavior_outputs(candidate_report))
        candidate_error = 1.0 - score
        fingerprint = _sha256(
            {
                "role": "independent-candidate-marginal",
                "candidate_sha256": record.candidate_sha256,
                "parent_state_sha256": PARENT_ALLOCATOR_STATE_SHA256,
                "suite_sha256": summary.get("suite_sha256"),
                "outputs_sha256": outputs_sha,
            }
        )
        if (
            count != record.validation_sample_count
            or audit.get("candidate_behavioral_sample_count") != count
            or audit.get("candidate_outputs_sha256") != outputs_sha
            or audit.get("candidate_evaluation_fingerprint") != fingerprint
            or record.candidate_evaluation_fingerprint != fingerprint
            or not math.isclose(
                float(audit.get("candidate_error")),
                candidate_error,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(record.candidate_metric_value),
                candidate_error,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise QwenUnifiedV8ReplayError("Candidate evaluation proof is invalid.")
        validated_by_sha[validated.proposal_sha256] = validated
        validation_initial_sequence.append(proposal.proposal_sha256)
        validation_final_sequence.append(validated.proposal_sha256)
    if (
        summary.get("proposal_sequence_sha256")
        != _sha256(validation_initial_sequence)
        or summary.get("validated_proposal_sequence_sha256")
        != _sha256(validation_final_sequence)
    ):
        raise QwenUnifiedV8ReplayError("Validated proposal sequence hash is invalid.")

    lines = [
        line
        for line in (root / "transactions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise QwenUnifiedV8ReplayError("Transaction ledger must contain exactly one line.")
    try:
        ledger = json.loads(lines[0], object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, QwenUnifiedV8ReplayError) as exc:
        raise QwenUnifiedV8ReplayError("Transaction ledger JSON is invalid.") from exc
    if not isinstance(ledger, Mapping):
        raise QwenUnifiedV8ReplayError("Transaction ledger line is not an object.")
    batch_raw = ledger.get("batch")
    transaction = ledger.get("transaction")
    validation_binding = ledger.get("candidate_validation")
    if not isinstance(batch_raw, Mapping) or not isinstance(transaction, Mapping):
        raise QwenUnifiedV8ReplayError("Transaction ledger is incomplete.")
    if (
        not isinstance(validation_binding, Mapping)
        or validation_binding.get("summary_artifact_sha256")
        != observed["validation_summary.json"]
        or validation_binding.get("validated_candidate_count") != 8
        or validation_binding.get("validation_class")
        != "completed-independent-candidate-validation"
        or validation_binding.get("validation_fingerprint")
        != summary.get("validation_fingerprint")
    ):
        raise QwenUnifiedV8ReplayError(
            "Transaction candidate-validation binding is invalid."
        )
    selected = tuple(batch_raw.get("proposal_sha256s") or ())
    if selected != EXPECTED_SELECTED_VALIDATED_PROPOSAL_SHA256S:
        raise QwenUnifiedV8ReplayError("Accepted proposal sequence is not exact.")
    try:
        selected_proposals = tuple(validated_by_sha[item] for item in selected)
        batch = Qwen3MixedSurgeryBatch(
            proposals=selected_proposals,
            parent_state_sha256=batch_raw["parent_state_sha256"],
            requested_physical_bytes=batch_raw["requested_physical_bytes"],
            prototype_bytes_saved=batch_raw["prototype_bytes_saved"],
            scale_setup_bytes=batch_raw["scale_setup_bytes"],
            physical_bytes_saved=batch_raw["physical_bytes_saved"],
            total_risk=batch_raw["total_risk"],
            proposal_work=batch_raw["proposal_work"],
            knapsack_states_considered=batch_raw["knapsack_states_considered"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenUnifiedV8ReplayError("Accepted batch cannot be reconstructed.") from exc
    if batch.as_dict() != dict(batch_raw) or batch.batch_sha256 != EXPECTED_BATCH_SHA256:
        raise QwenUnifiedV8ReplayError("Accepted batch SHA-256 is invalid.")

    gate_name = _basename(ledger.get("gate_artifact"))
    gate = _read_json(root / gate_name)
    gate_candidate_name = _basename(gate.get("candidate_report"))
    gate_candidate = _read_json(root / gate_candidate_name)
    context = gate.get("context")
    metrics = gate.get("metrics")
    decision = transaction.get("decision")
    proposed_accounting = transaction.get("proposed_accounting")
    if (
        gate_name not in EVIDENCE_FILE_SHA256S
        or gate_candidate_name not in EVIDENCE_FILE_SHA256S
        or gate.get("candidate_report_sha256") != observed[gate_candidate_name]
        or not isinstance(context, Mapping)
        or not isinstance(metrics, Mapping)
        or not isinstance(decision, Mapping)
        or context.get("candidate_state_sha256") != GATE_CANDIDATE_STATE_SHA256
        or metrics.get("candidate_state_sha256") != GATE_CANDIDATE_STATE_SHA256
        or context.get("parent_state_sha256") != PARENT_ALLOCATOR_STATE_SHA256
        or context.get("batch_sha256") != EXPECTED_BATCH_SHA256
        or gate.get("verdict", {}).get("passed") is not True
        or decision.get("accepted") is not True
        or decision.get("metrics") != metrics
        or transaction.get("accepted") is not True
        or transaction.get("zero_recovery") is not True
        or transaction.get("optimizer_steps") != 0
        or transaction.get("backward_calls") != 0
        or transaction.get("batch_sha256") != EXPECTED_BATCH_SHA256
        or tuple(transaction.get("proposal_sha256s") or ()) != selected
        or transaction.get("committed_accounting") != proposed_accounting
        or transaction.get("parent_accounting") != initial.get("accounting")
        or not isinstance(proposed_accounting, Mapping)
    ):
        raise QwenUnifiedV8ReplayError("Accepted transaction/gate binding is invalid.")
    gate_score, gate_count = _behavior_score(gate_candidate)
    gate_outputs = _sha256(_behavior_outputs(gate_candidate))
    fingerprint_payload = {
        name: metrics.get(name)
        for name in (
            "evaluation_kind",
            "parent_state_sha256",
            "candidate_state_sha256",
            "evaluation_suite_sha256",
            "parent_outputs_sha256",
            "candidate_outputs_sha256",
            "behavioral_sample_count",
            "parent_behavior_score",
            "candidate_behavior_score",
        )
    }
    if (
        metrics.get("candidate_outputs_sha256") != gate_outputs
        or metrics.get("behavioral_sample_count") != gate_count
        or metrics.get("evaluation_fingerprint") != _sha256(fingerprint_payload)
        or not math.isclose(
            float(metrics.get("candidate_behavior_score")),
            gate_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise QwenUnifiedV8ReplayError("Accepted gate evaluation proof is invalid.")
    if (
        proposed_accounting.get("child_unique_parameters")
        != EXPECTED_CHILD_UNIQUE_PARAMETERS
        or proposed_accounting.get("deployable_bytes") != EXPECTED_DEPLOYABLE_BYTES
        or proposed_accounting.get("neuron_registry_sha256")
        != EXPECTED_NEURON_REGISTRY_SHA256
    ):
        raise QwenUnifiedV8ReplayError("Accepted proposed accounting is not exact.")

    selected_candidates = tuple(
        proposal.candidate.candidate_sha256 for proposal in selected_proposals
    )
    operation_redirects = {
        (
            proposal.candidate.resulting_medoid[2],
            next(
                node[3]
                for node in proposal.candidate.member_ids
                if node != proposal.candidate.resulting_medoid
            ),
            proposal.candidate.resulting_medoid[3],
        )
        for proposal in selected_proposals
    }
    if operation_redirects != set(V8_REDIRECTS):
        raise QwenUnifiedV8ReplayError("Accepted candidate operations changed redirects.")
    return V8EvidenceProof(
        proposed_accounting=dict(proposed_accounting),
        selected_proposal_sha256s=selected,
        selected_candidate_sha256s=selected_candidates,
        redirects=V8_REDIRECTS,
        evidence_file_sha256s=observed,
    )


@dataclass(frozen=True, slots=True)
class _ReplayAssignmentPlan:
    assignment_pairs: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]


class ExactNeuronMedoidReplay:
    """Stage exact medoid redirects and roll back unless explicitly committed."""

    def __init__(
        self,
        runtime: nn.Module,
        redirects: Sequence[tuple[int, int, int]],
    ) -> None:
        self.runtime = runtime
        self.redirects = tuple((int(a), int(b), int(c)) for a, b, c in redirects)
        self.parent = neuron_surgery._capture_installed(runtime)
        self.proposed: Any | None = None
        self._committed = False
        self._staged = False
        self._validate_redirects()

    def _validate_redirects(self) -> None:
        if not self.redirects:
            raise QwenUnifiedV8ReplayError("Replay needs at least one redirect.")
        removed = {(layer, source) for layer, source, _target in self.redirects}
        targets = {(layer, target) for layer, _source, target in self.redirects}
        if len(removed) != len(self.redirects) or removed & targets:
            raise QwenUnifiedV8ReplayError(
                "Replay redirects must have unique, non-chained removed nodes."
            )
        width = self.parent.public.intermediate_width
        active = set(self.parent.public.prototype_source_ids)
        for layer, source, target in self.redirects:
            source_node = neuron_surgery.qwen3_neuron_node_id(layer, source)
            target_node = neuron_surgery.qwen3_neuron_node_id(layer, target)
            if source == target or source_node not in active or target_node not in active:
                raise QwenUnifiedV8ReplayError(
                    f"Replay redirect is not active and distinct: {source_node}->{target_node}."
                )
            if not 0 <= source < width or not 0 <= target < width:
                raise QwenUnifiedV8ReplayError("Replay neuron index is out of range.")

    def _restore_parent(self) -> None:
        if not neuron_surgery._restore_installed(self.runtime, self.parent):
            raise QwenUnifiedV8ReplayError("Replay rollback was not exact.")
        setattr(
            self.runtime,
            surgery_allocator._REPORT_ATTRIBUTE,
            self.parent.aggregate_report,
        )
        if (
            qwen3_codebook_parameter_report(self.runtime)
            != self.parent.aggregate_report
        ):
            raise QwenUnifiedV8ReplayError(
                "Replay rollback did not restore the incumbent parameter report."
            )

    def stage(self) -> Any:
        if self._staged:
            raise QwenUnifiedV8ReplayError("Replay transaction is already staged.")
        pairs = tuple(
            (
                neuron_surgery.qwen3_neuron_node_id(layer, source),
                neuron_surgery.qwen3_neuron_node_id(layer, target),
            )
            for layer, source, target in self.redirects
        )
        try:
            with torch.inference_mode():
                compact = neuron_surgery._compacted_codebook(
                    self.runtime,
                    _ReplayAssignmentPlan(pairs),  # type: ignore[arg-type]
                    (),
                )
                neuron_surgery._install_codebook(self.runtime, compact)
                # Match the accepted unified allocator transaction exactly.  Its
                # accounting identity is captured only after this incremental
                # report synchronization, which intentionally differs from the
                # general live-report recomputation for registered-logical fields.
                surgery_allocator._sync_parameter_report(self.runtime)
            self.proposed = neuron_surgery._capture_installed(self.runtime)
            changes = neuron_surgery._assignment_changes(
                self.parent.public, self.proposed.public
            )
            observed_pairs = {
                (change.before_prototype, change.after_prototype) for change in changes
            }
            expected_pairs = set(pairs)
            if (
                self.proposed.public.prototype_count
                != self.parent.public.prototype_count - len(self.redirects)
                or observed_pairs != expected_pairs
                or any(parameter.grad is not None for parameter in self.runtime.parameters())
            ):
                raise QwenUnifiedV8ReplayError(
                    "Staged replay does not implement only the exact redirects."
                )
            self._staged = True
            return self.proposed
        except Exception:
            self._restore_parent()
            raise

    def commit(self) -> None:
        if not self._staged or self.proposed is None:
            raise QwenUnifiedV8ReplayError("Cannot commit an unstaged replay.")
        if neuron_surgery.qwen3_runtime_assignment_snapshot(self.runtime) != (
            self.proposed.public
        ):
            raise QwenUnifiedV8ReplayError("Replay state changed before commit.")
        self._committed = True

    def rollback(self) -> None:
        if not self._staged or self._committed:
            return
        self._restore_parent()
        self._staged = False

    def __enter__(self) -> ExactNeuronMedoidReplay:
        self.stage()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.rollback()


def _tensor_bytes(value: Tensor | None) -> int:
    return 0 if value is None else int(value.numel()) * int(value.element_size())


def exact_accounting(
    runtime: nn.Module,
    tokenizer_contract: Mapping[str, Any],
) -> Qwen3UnifiedSurgeryAccounting:
    # This must follow Qwen3UnifiedSurgeryAllocator._compute_accounting, not
    # merely recompute the general live codebook report.  The allocator's
    # incremental synchronization preserves its registered-logical baseline
    # while updating the compact registry fields; that exact report is part of
    # the accepted accounting SHA-256.
    report = surgery_allocator._sync_parameter_report(runtime)
    neuron = neuron_surgery.qwen3_runtime_assignment_snapshot(runtime)
    attention = attention_surgery.qwen3_attention_runtime_snapshot(runtime)
    plan = plan_qwen3_codebook_artifact(runtime, tokenizer_contract=tokenizer_contract)
    neuron_registry = runtime.neuron_codebook.registry
    attention_registry = runtime.attention_codebook.registry
    scale_bytes = _tensor_bytes(neuron_registry.neuron_scales) + _tensor_bytes(
        attention_registry.role_scales
    )
    return Qwen3UnifiedSurgeryAccounting(
        source_unique_parameters=report.source_unique_parameter_count,
        child_unique_parameters=report.unique_parameter_count,
        parameter_tensor_bytes=plan.runtime_parameter_storage_bytes,
        buffer_tensor_bytes=plan.persistent_buffer_storage_bytes,
        tensor_bytes=plan.safetensors_payload_bytes,
        neuron_registry_bytes=neuron.tensor_bytes,
        attention_registry_bytes=attention.tensor_bytes,
        index_bytes=report.index_buffer_bytes,
        scale_bytes=scale_bytes,
        metadata_bytes=plan.artifact_total_bytes - plan.safetensors_payload_bytes,
        deployable_bytes=plan.artifact_total_bytes,
        parameter_report_sha256=_sha256(report.as_dict()),
        neuron_registry_sha256=neuron.registry_sha256,
        attention_registry_sha256=attention.registry_sha256,
        logical_state_tensor_bytes=plan.logical_state_tensor_bytes,
        runtime_registered_storage_bytes=plan.runtime_registered_storage_bytes,
        artifact_payload_bytes=plan.safetensors_payload_bytes,
        artifact_header_bytes=plan.safetensors_header_bytes,
        artifact_config_bytes=plan.config_file_bytes,
        artifact_tokenizer_contract_bytes=plan.tokenizer_contract_file_bytes,
        artifact_manifest_bytes=plan.manifest_file_bytes,
        artifact_plan_sha256=plan.artifact_plan_sha256,
        artifact_tokenizer_contract_resolved=plan.tokenizer_contract_resolved,
    )


def _tensor_raw_sha256(value: Tensor) -> str:
    tensor = value.detach().contiguous().reshape(-1)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    elements = max(1, (8 * 1024**2) // max(int(value.element_size()), 1))
    for start in range(0, int(tensor.numel()), elements):
        piece = tensor[start : start + elements].to(device="cpu").contiguous()
        digest.update(piece.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_content_identity(runtime: nn.Module) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    for name, value in sorted(runtime.state_dict().items()):
        if not isinstance(value, Tensor) or value.device.type == "meta":
            raise QwenUnifiedV8ReplayError(f"Cannot content-hash state entry {name}.")
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_raw_sha256(value),
            }
        )
    report = qwen3_codebook_parameter_report(runtime).as_dict()
    config = dict(runtime.config.to_dict())
    for name in ("_name_or_path", "transformers_version"):
        config.pop(name, None)
    parent_rows = getattr(runtime, "parent_token_rows", None)
    payload = {
        "schema": MODEL_CONTENT_SCHEMA,
        "state_tensor_count": len(tensors),
        "state_tensor_manifest_sha256": _sha256(tensors),
        "parameter_report_sha256": _sha256(report),
        "executable_config_sha256": _sha256(config),
        "parent_token_rows_sha256": (
            None
            if not isinstance(parent_rows, Tensor)
            else hashlib.sha256(
                parent_rows.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        ),
    }
    payload["model_content_sha256"] = _sha256(payload)
    return payload


def _artifact_tree(
    directory: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in excluded_relative_paths:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not files:
        raise QwenUnifiedV8ReplayError("Saved artifact tree is empty.")
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "tree_sha256": _sha256(files),
        "files": files,
    }


def _cleanup_new_destination(destination: Path, requested_destination: Path) -> None:
    """Remove only the exact, newly-created publication directory."""

    if not destination.exists() and not destination.is_symlink():
        return
    try:
        resolved = destination.resolve(strict=True)
    except OSError as error:
        raise QwenUnifiedV8ReplayError(
            "Cannot resolve the failed publication destination for safe cleanup."
        ) from error
    if (
        destination != requested_destination
        or resolved != requested_destination
        or destination.is_symlink()
        or not destination.is_dir()
    ):
        raise QwenUnifiedV8ReplayError(
            "Refusing to clean a publication path other than the requested absent output."
        )
    shutil.rmtree(destination)
    if destination.exists() or destination.is_symlink():
        raise QwenUnifiedV8ReplayError(
            "Failed publication destination remained after scoped cleanup."
        )


def _publish_staged_checkpoint(
    staged: Path,
    destination: Path,
    *,
    expected_tree: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish by rename, or by a marked and hash-verified CIFS copy."""

    staged = staged.expanduser().resolve(strict=True)
    requested_destination = destination.expanduser().resolve(strict=False)
    destination = requested_destination
    if not staged.is_dir():
        raise QwenUnifiedV8ReplayError("Staged checkpoint is not a directory.")
    if destination.exists() or destination.is_symlink():
        raise QwenUnifiedV8ReplayError(
            "Checkpoint publication destination must be absent."
        )
    if destination.parent.resolve(strict=True) != destination.parent:
        raise QwenUnifiedV8ReplayError(
            "Checkpoint publication parent did not resolve exactly."
        )
    marker_source = staged / _INCOMPLETE_PROMOTION_MARKER
    if marker_source.exists() or marker_source.is_symlink():
        raise QwenUnifiedV8ReplayError(
            "Staged checkpoint collides with the incomplete-promotion marker."
        )
    staged_tree = _artifact_tree(staged)
    if expected_tree is not None and dict(expected_tree) != staged_tree:
        raise QwenUnifiedV8ReplayError(
            "Staged checkpoint changed before final publication."
        )

    created_destination = False
    try:
        try:
            os.replace(staged, destination)
            created_destination = True
            mode = "atomic-replace"
        except OSError as error:
            if not isinstance(error, PermissionError) and error.errno != errno.EXDEV:
                raise
            if destination.exists() or destination.is_symlink():
                raise QwenUnifiedV8ReplayError(
                    "Failed rename unexpectedly created the publication destination."
                ) from error
            destination.mkdir(exist_ok=False)
            created_destination = True
            marker = destination / _INCOMPLETE_PROMOTION_MARKER
            with marker.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema": "neural-forge-incomplete-checkpoint-promotion-v1",
                            "staged_tree_sha256": staged_tree["tree_sha256"],
                            "staged_total_bytes": staged_tree["total_bytes"],
                        },
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            shutil.copytree(staged, destination, dirs_exist_ok=True)
            copied_tree = _artifact_tree(
                destination,
                excluded_relative_paths=frozenset({_INCOMPLETE_PROMOTION_MARKER}),
            )
            if copied_tree != staged_tree or _artifact_tree(staged) != staged_tree:
                raise QwenUnifiedV8ReplayError(
                    "CIFS checkpoint copy differs from the staged artifact tree."
                )
            marker.unlink()
            mode = "verified-copy-fallback"

        published_tree = _artifact_tree(destination)
        if published_tree != staged_tree:
            raise QwenUnifiedV8ReplayError(
                "Published checkpoint differs from the staged artifact tree."
            )
        if (destination / _INCOMPLETE_PROMOTION_MARKER).exists():
            raise QwenUnifiedV8ReplayError(
                "Published checkpoint retained its incomplete marker."
            )
        return {
            "mode": mode,
            "atomic_directory_replace": mode == "atomic-replace",
            "verified_copy_fallback": mode == "verified-copy-fallback",
            "incomplete_marker": (
                None
                if mode == "atomic-replace"
                else _INCOMPLETE_PROMOTION_MARKER
            ),
            "incomplete_marker_removed": True,
            "staged_tree_sha256": staged_tree["tree_sha256"],
            "published_tree_sha256": published_tree["tree_sha256"],
            "file_count": published_tree["file_count"],
            "total_bytes": published_tree["total_bytes"],
        }
    except BaseException as error:
        if created_destination:
            try:
                _cleanup_new_destination(destination, requested_destination)
            except BaseException as cleanup_error:
                error.add_note(
                    "Scoped cleanup of the failed publication also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise


def _parent_free_smoke(runtime: nn.Module) -> dict[str, Any]:
    config = runtime.config
    token = next(
        (
            int(value)
            for value in (
                getattr(config, "bos_token_id", None),
                getattr(config, "eos_token_id", None),
                getattr(config, "pad_token_id", None),
                0,
            )
            if isinstance(value, int) and 0 <= value < int(config.vocab_size)
        ),
        0,
    )
    input_ids = torch.tensor([[token]], dtype=torch.long, device=next(runtime.parameters()).device)
    runtime.eval()
    with torch.inference_mode():
        first = runtime(input_ids=input_ids, use_cache=True)
        logits = first.logits
        if tuple(logits.shape) != (1, 1, int(config.vocab_size)) or not bool(
            torch.isfinite(logits).all()
        ):
            raise QwenUnifiedV8ReplayError("Reloaded child first-step smoke failed.")
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        second = runtime(
            input_ids=next_token,
            past_key_values=first.past_key_values,
            use_cache=True,
        )
        if tuple(second.logits.shape) != (1, 1, int(config.vocab_size)) or not bool(
            torch.isfinite(second.logits).all()
        ):
            raise QwenUnifiedV8ReplayError("Reloaded child cached-step smoke failed.")
    return {
        "parent_model_loaded": False,
        "first_token_id": token,
        "selected_next_token_id": int(next_token.item()),
        "first_step_finite": True,
        "cached_step_finite": True,
    }


def _checkpoint_source_identity(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise QwenUnifiedV8ReplayError("Input checkpoint manifest lacks files.")
    tree = _artifact_tree(checkpoint)
    return {
        "manifest_file_sha256": _file_sha256(manifest_path),
        "declared_files_sha256": _sha256(dict(files)),
        "artifact_tree_sha256": tree["tree_sha256"],
        "artifact_total_bytes": tree["total_bytes"],
        "schema": manifest.get("schema"),
    }


def replay_v8_commit(
    *,
    input_checkpoint: str | Path,
    evidence_directory: str | Path,
    output_checkpoint: str | Path,
    device: str = "cpu",
    reload_device: str = "cpu",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    proof = validate_v8_evidence_bundle(evidence_directory)
    source = Path(input_checkpoint).expanduser().resolve()
    destination = Path(output_checkpoint).expanduser().resolve()
    target_report = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else destination.with_name(destination.name + ".replay.json")
    )
    if not source.is_dir():
        raise QwenUnifiedV8ReplayError(f"Input checkpoint does not exist: {source}.")
    if destination.exists():
        raise QwenUnifiedV8ReplayError("Output checkpoint must not already exist.")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise QwenUnifiedV8ReplayError("Output checkpoint cannot be inside its source.")
    try:
        target_report.relative_to(destination)
    except ValueError:
        pass
    else:
        raise QwenUnifiedV8ReplayError(
            "Replay report cannot alter the canonical checkpoint directory."
        )
    if target_report.exists():
        raise QwenUnifiedV8ReplayError("Replay report path already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    target_report.parent.mkdir(parents=True, exist_ok=True)
    initial = _read_json(Path(evidence_directory).resolve() / "initial_incumbent.json")
    expected_source = initial.get("source_identity", {}).get("identity")
    actual_source = _checkpoint_source_identity(source)
    if not isinstance(expected_source, Mapping) or any(
        actual_source.get(name) != expected_source.get(name)
        for name in actual_source
    ):
        raise QwenUnifiedV8ReplayError("Input checkpoint is not the immutable p35 source.")
    tokenizer_contract = _read_json(source / "tokenizer_contract.json")
    runtime = load_qwen3_codebook_checkpoint(source, device=device)
    temporary_root: Path | None = None
    reloaded: nn.Module | None = None
    published_destination = False
    try:
        initial_content = model_content_identity(runtime)
        initial_accounting = exact_accounting(runtime, tokenizer_contract)
        if (
            initial_content["model_content_sha256"] != INITIAL_CONTENT_SHA256
            or initial_accounting.as_dict() != initial.get("accounting")
            or any(parameter.grad is not None for parameter in runtime.parameters())
        ):
            raise QwenUnifiedV8ReplayError("Loaded p35 content/accounting is not exact.")
        with ExactNeuronMedoidReplay(runtime, proof.redirects) as transaction:
            proposed_content = model_content_identity(runtime)
            proposed_accounting = exact_accounting(runtime, tokenizer_contract)
            proposed_snapshot = neuron_surgery.qwen3_runtime_assignment_snapshot(runtime)
            proposed_accounting_dict = proposed_accounting.as_dict()
            accounting_diff = _mapping_diff(
                proof.proposed_accounting, proposed_accounting_dict
            )
            registry_matches = (
                proposed_snapshot.registry_sha256
                == EXPECTED_NEURON_REGISTRY_SHA256
            )
            if not registry_matches or accounting_diff:
                replay_diagnostic = {
                    "schema": "neural-forge-qwen3-unified-v8-replay-mismatch-v1",
                    "registry": {
                        "expected_sha256": EXPECTED_NEURON_REGISTRY_SHA256,
                        "observed_sha256": proposed_snapshot.registry_sha256,
                        "matches": registry_matches,
                        "prototype_count": proposed_snapshot.prototype_count,
                        "tensor_bytes": proposed_snapshot.tensor_bytes,
                        "prototype_source_indices_sha256": _sha256(
                            list(proposed_snapshot.prototype_source_indices)
                        ),
                        "assignments_sha256": _sha256(
                            [list(row) for row in proposed_snapshot.assignments]
                        ),
                        "assignment_source_indices_sha256": _sha256(
                            [
                                list(row)
                                for row in proposed_snapshot.assignment_source_indices
                            ]
                        ),
                    },
                    "accounting_field_diff": accounting_diff,
                    "observed_model_content_sha256": proposed_content[
                        "model_content_sha256"
                    ],
                    "redirects": [list(item) for item in proof.redirects],
                }
                raise QwenUnifiedV8ReplayError(
                    "Replayed executable state differs from approved proposed "
                    "accounting: "
                    + json.dumps(
                        replay_diagnostic,
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            temporary_root = Path(
                tempfile.mkdtemp(prefix=".v8-exact-replay.", dir=destination.parent)
            )
            staged = temporary_root / "checkpoint"
            save_qwen3_codebook_checkpoint(
                runtime,
                staged,
                tokenizer_contract=tokenizer_contract,
            )
            reloaded = load_qwen3_codebook_checkpoint(staged, device=reload_device)
            reloaded_content = model_content_identity(reloaded)
            reloaded_accounting = exact_accounting(reloaded, tokenizer_contract)
            if (
                reloaded_content != proposed_content
                or reloaded_accounting != proposed_accounting
                or neuron_surgery.qwen3_runtime_assignment_snapshot(reloaded).registry_sha256
                != EXPECTED_NEURON_REGISTRY_SHA256
            ):
                raise QwenUnifiedV8ReplayError("Saved/reloaded child changed exact content.")
            smoke = _parent_free_smoke(reloaded)
            tree = _artifact_tree(staged)
            if tree["total_bytes"] != EXPECTED_DEPLOYABLE_BYTES:
                raise QwenUnifiedV8ReplayError("Saved child byte count is not exact.")
            promotion = _publish_staged_checkpoint(
                staged,
                destination,
                expected_tree=tree,
            )
            published_destination = True
            transaction.commit()
        report = {
            "schema": SCHEMA,
            "status": "saved-reloaded-parent-free-smoke-passed",
            "hash_domain_correction": {
                "dffa_is_allocator_gate_state_not_model_content": True,
                "gate_candidate_state_sha256": GATE_CANDIDATE_STATE_SHA256,
                "computed_model_content_sha256": proposed_content[
                    "model_content_sha256"
                ],
                "save_reload_model_content_equal": True,
            },
            "input_checkpoint": str(source),
            "output_checkpoint": str(destination),
            "initial_model_content": initial_content,
            "proposed_model_content": proposed_content,
            "reloaded_model_content": reloaded_content,
            "proposed_accounting": proposed_accounting.as_dict(),
            "reloaded_accounting": reloaded_accounting.as_dict(),
            "neuron_registry_sha256": EXPECTED_NEURON_REGISTRY_SHA256,
            "redirects": [
                {"layer": layer, "removed": removed, "medoid": medoid}
                for layer, removed, medoid in proof.redirects
            ],
            "selected_proposal_sha256s": list(proof.selected_proposal_sha256s),
            "selected_candidate_sha256s": list(proof.selected_candidate_sha256s),
            "evidence_file_sha256s": proof.evidence_file_sha256s,
            "checkpoint_artifact_tree": tree,
            "checkpoint_promotion": promotion,
            "parent_free_smoke": smoke,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "behavioral_evaluations_repeated": 0,
            "discovery_repeated": False,
        }
        report["report_sha256"] = _record_sha256(report, "report_sha256")
        with target_report.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    report,
                    ensure_ascii=True,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        return report
    except BaseException as error:
        if published_destination:
            try:
                _cleanup_new_destination(destination, destination)
            except BaseException as cleanup_error:
                error.add_note(
                    "Scoped cleanup after replay publication failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
    finally:
        del reloaded
        del runtime
        gc.collect()
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reload-device", default="cpu")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = replay_v8_commit(
        input_checkpoint=args.input_codebook_checkpoint,
        evidence_directory=args.evidence_directory,
        output_checkpoint=args.output,
        device=args.device,
        reload_device=args.reload_device,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
