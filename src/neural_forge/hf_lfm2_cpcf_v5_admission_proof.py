"""Standalone proof for one store-authenticated CPCF-v5 admission chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from . import hf_lfm2_cpcf_resume_artifact as resume_artifact
from . import hf_lfm2_cpcf_v5_resume as resume
from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256


LFM2_CPCF_V5_ADMISSION_CHAIN_PROOF_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-admission-chain-proof-v1"
)
_ZERO_AUDIT = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "generation_calls": 0,
    "parent_cache_generation_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "fallback_paths": 0,
}
_PROOF_FIELDS = frozenset(
    {
        "schema",
        "resume_schema",
        "terminal_document",
        "admission_document",
        "pending_document",
        "progress_document",
        "boundary_manifest",
        "boundary_reports",
        "boundary_certificate_sha256",
        "checkpoint_manifest",
        "checkpoint_completed_layers",
        "checkpoint_certificate_sha256",
        "store_artifacts_reauthenticated",
        "compiler_persisted_standalone_chain",
        "parent_or_cache_tensors_embedded",
        "operation_audit",
        "proof_sha256",
    }
)
_PENDING_FIELDS = frozenset(
    {
        "schema",
        "resume_schema",
        "coordinator_identity_sha256",
        "capability_sha256",
        "frontier_reference",
        "checkpoint_reference",
        "frontier_sha256",
        "event_report_sha256",
        "next_layer_index",
        "rss_checkpoint_expectation",
        "parent_or_cache_tensors_archived_in_frontier",
        "progress_event_persisted_and_reauthenticated",
        "operation_audit",
        "pending_sha256",
    }
)


class Lfm2CpcfV5AdmissionProofError(RuntimeError):
    """A persisted admission-chain proof is incomplete or inconsistent."""


def _plain(value: Any) -> Any:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [thaw(child) for child in item]
        return item

    try:
        return json.loads(canonical_json_bytes(thaw(value)).decode("ascii"))
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            "Admission proof content must be finite canonical JSON."
        ) from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    result = _plain(value)
    if not isinstance(result, dict):
        raise Lfm2CpcfV5AdmissionProofError(f"{name} must be a mapping.")
    return result


def _digest(value: Any, name: str) -> str:
    try:
        return resume._digest(value, name)
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(str(exc)) from exc


def _sealed_document(
    value: Any,
    *,
    fields: frozenset[str],
    schema: str,
    seal_name: str,
    name: str,
) -> dict[str, Any]:
    document = _mapping(value, name)
    body = dict(document)
    supplied = _digest(body.pop(seal_name, None), f"{name}.{seal_name}")
    if (
        set(document) != fields
        or document.get("schema") != schema
        or document.get("resume_schema") != resume.LFM2_CPCF_V5_RESUME_SCHEMA
        or canonical_sha256(body) != supplied
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            f"{name} schema, fields, or seal is invalid."
        )
    return document


def _reference(value: Any, *, kind: str) -> Any:
    try:
        return resume._reference(value, expected_kind=kind)
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            f"Admission proof {kind} reference is invalid."
        ) from exc


def _json_record_matches(
    record: Mapping[str, Any],
    value: Mapping[str, Any],
) -> bool:
    payload = canonical_json_bytes(value) + b"\n"
    return (
        record.get("format") == "canonical-json"
        and record.get("size_bytes") == len(payload)
        and record.get("sha256") == hashlib.sha256(payload).hexdigest()
    )


def _tensor_record_inventory(
    record: Mapping[str, Any],
    *,
    name: str,
) -> list[dict[str, Any]]:
    entries = record.get("tensor_entries")
    if (
        record.get("format") != "safetensors"
        or not isinstance(entries, list)
        or not entries
        or record.get("tensor_set_sha256") != canonical_sha256(entries)
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            f"{name} tensor inventory is invalid."
        )
    expected_fields = {
        "name",
        "dtype",
        "shape",
        "numel",
        "element_size",
        "tensor_sha256",
        "runtime_tensor_sha256",
    }
    names = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != expected_fields:
            raise Lfm2CpcfV5AdmissionProofError(
                f"{name} tensor entry has an invalid field set."
            )
        tensor_name = entry.get("name")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        numel = entry.get("numel")
        element_size = entry.get("element_size")
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(shape, list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            )
            or isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel < 0
            or isinstance(element_size, bool)
            or not isinstance(element_size, int)
            or element_size <= 0
        ):
            raise Lfm2CpcfV5AdmissionProofError(
                f"{name} tensor entry metadata is invalid."
            )
        expected_numel = 1
        for size in shape:
            expected_numel *= size
        if numel != expected_numel:
            raise Lfm2CpcfV5AdmissionProofError(
                f"{name} tensor entry numel differs from its shape."
            )
        _digest(entry.get("tensor_sha256"), f"{name} tensor byte SHA-256")
        _digest(
            entry.get("runtime_tensor_sha256"),
            f"{name} runtime tensor SHA-256",
        )
        names.append(tensor_name)
    if names != sorted(set(names)):
        raise Lfm2CpcfV5AdmissionProofError(
            f"{name} tensor entries are duplicated or unordered."
        )
    return entries


def _authenticate_boundary(
    *,
    manifest_value: Any,
    reports_value: Any,
    certificate_sha256: Any,
    reference: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _mapping(manifest_value, "boundary manifest")
    reports = _mapping(reports_value, "boundary reports")
    body = dict(manifest)
    supplied = _digest(
        body.pop("manifest_sha256", None),
        "boundary manifest SHA-256",
    )
    try:
        bindings = resume_artifact._boundary_bindings_from_mapping(
            manifest.get("bindings")
        )
        resume_artifact._verify_non_promotable(manifest)
        records = resume_artifact._records_by_path(
            manifest,
            resume_artifact._BOUNDARY_FILES - {"manifest.json"},
        )
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            "Boundary manifest bindings or file inventory are invalid."
        ) from exc
    report_record = records["reports.json"]
    tensor_record = records["tensors.safetensors"]
    tensor_entries = _tensor_record_inventory(
        tensor_record,
        name="boundary",
    )
    if (
        set(manifest) != resume_artifact._BOUNDARY_MANIFEST_FIELDS
        or manifest.get("schema") != resume_artifact.RESUME_ARTIFACT_SCHEMA
        or manifest.get("artifact_kind") != "boundary"
        or canonical_sha256(body) != supplied
        or reference.manifest_sha256 != supplied
        or reference.artifact_key != manifest.get("artifact_key")
        or manifest.get("key_schema") != resume_artifact.BOUNDARY_KEY_SCHEMA
        or manifest.get("binding_key_schema")
        != resume_artifact.BOUNDARY_BINDING_KEY_SCHEMA
        or manifest.get("binding_key") != bindings.binding_key
        or manifest.get("bindings") != bindings.as_dict()
        or resume_artifact._boundary_content_key(
            bindings=bindings,
            files=manifest["files"],
        )
        != reference.artifact_key
        or not _json_record_matches(report_record, reports)
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "Boundary manifest, report, reference, or content key is invalid."
        )
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    expected_certificate = resume_artifact._certificate(manifest_raw, records)
    if (
        _digest(certificate_sha256, "boundary certificate SHA-256")
        != expected_certificate
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "Boundary certificate does not reconstruct from its inventory."
        )
    inventory = reports.get("tensor_inventory")
    by_name = {
        str(entry.get("name")): entry
        for entry in tensor_entries
        if isinstance(entry, Mapping)
    }
    if (
        not isinstance(inventory, list)
        or [item.get("name") for item in inventory] != sorted(by_name)
        or len(inventory) != len(by_name)
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "Boundary report and manifest tensor inventories differ."
        )
    for row in inventory:
        entry = by_name.get(str(row.get("name")))
        if (
            entry is None
            or row.get("dtype") != str(entry.get("dtype")).removeprefix("torch.")
            or row.get("shape") != entry.get("shape")
            or row.get("numel") != entry.get("numel")
            or row.get("element_size") != entry.get("element_size")
            or row.get("byte_sha256") != entry.get("tensor_sha256")
            or row.get("runtime_tensor_sha256")
            != entry.get("runtime_tensor_sha256")
        ):
            raise Lfm2CpcfV5AdmissionProofError(
                "Boundary tensor inventory row differs from its manifest."
            )
    return manifest, reports


def _authenticate_checkpoint(
    *,
    manifest_value: Any,
    completed_value: Any,
    certificate_sha256: Any,
    reference: Any,
    boundary_reference: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _mapping(manifest_value, "checkpoint manifest")
    completed = _mapping(completed_value, "checkpoint completed layers")
    body = dict(manifest)
    supplied = _digest(
        body.pop("manifest_sha256", None),
        "checkpoint manifest SHA-256",
    )
    try:
        external, derived = resume_artifact._checkpoint_bindings_from_mapping(
            manifest.get("bindings")
        )
        resume_artifact._verify_non_promotable(manifest)
        records = resume_artifact._records_by_path(
            manifest,
            resume_artifact._CHECKPOINT_FILES - {"manifest.json"},
        )
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            "Checkpoint manifest bindings or file inventory are invalid."
        ) from exc
    prefix_entries = _tensor_record_inventory(
        records["prefix-model.safetensors"],
        name="checkpoint prefix-model",
    )
    incoming_entries = _tensor_record_inventory(
        records["incoming-state-bank.safetensors"],
        name="checkpoint incoming-state-bank",
    )
    recomputed = {
        "prefix_model_state_sha256": canonical_sha256(prefix_entries),
        "prefix_model_entry_sha256": {
            item["name"]: item["tensor_sha256"] for item in prefix_entries
        },
        "incoming_state_bank_sha256": canonical_sha256(incoming_entries),
        "incoming_state_entry_sha256": {
            item["name"]: item["tensor_sha256"] for item in incoming_entries
        },
        "completed_layers_document_sha256": canonical_sha256(completed),
    }
    try:
        resume_artifact._validate_completed_layers_document(
            completed,
            bindings=external,
            prefix_entries=prefix_entries,
            incoming_entries=incoming_entries,
        )
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            "Checkpoint completed-layer lineage is invalid."
        ) from exc
    if (
        set(manifest) != resume_artifact._CHECKPOINT_MANIFEST_FIELDS
        or manifest.get("schema") != resume_artifact.RESUME_ARTIFACT_SCHEMA
        or manifest.get("artifact_kind") != "checkpoint"
        or canonical_sha256(body) != supplied
        or reference.manifest_sha256 != supplied
        or reference.artifact_key != manifest.get("artifact_key")
        or manifest.get("key_schema") != resume_artifact.CHECKPOINT_KEY_SCHEMA
        or resume_artifact._checkpoint_content_key(
            bindings=manifest["bindings"],
            files=manifest["files"],
        )
        != reference.artifact_key
        or external.boundary != boundary_reference
        or derived != recomputed
        or any(manifest["bindings"].get(key) != value for key, value in recomputed.items())
        or not _json_record_matches(records["completed-layers.json"], completed)
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "Checkpoint manifest, reference, content, or boundary lineage is invalid."
        )
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    expected_certificate = resume_artifact._certificate(manifest_raw, records)
    if (
        _digest(certificate_sha256, "checkpoint certificate SHA-256")
        != expected_certificate
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "Checkpoint certificate does not reconstruct from its inventory."
        )
    return manifest, completed


def build_lfm2_cpcf_v5_admission_chain_proof(
    coordinator: Any,
    admission_sha256: str,
) -> Mapping[str, Any]:
    """Reopen one exact admission and persist its tensor-free document chain."""

    if type(coordinator) is not resume.Lfm2CpcfV5ResumeCoordinator:
        raise TypeError(
            "coordinator must be the exact Lfm2CpcfV5ResumeCoordinator type."
        )
    admission_key = _digest(admission_sha256, "admission SHA-256")
    try:
        state = coordinator.open_admitted(admission_key)
        pending_key = state.admission["pending_sha256"]
        terminal, admission = coordinator._open_terminal(pending_key)
        pending = coordinator._open_pending_document(pending_key)
        frontier, checkpoint, progress = (
            coordinator._reauthenticate_pending_artifacts(pending)
        )
        boundary_reference = _reference(
            pending["frontier_reference"],
            kind="boundary",
        )
        checkpoint_reference = _reference(
            pending["checkpoint_reference"],
            kind="checkpoint",
        )
        boundary = coordinator.store.final_reauthenticate_boundary(
            coordinator.store.open_boundary(boundary_reference)
        )
        checkpoint = coordinator.store.final_reauthenticate_checkpoint(
            coordinator.store.open_checkpoint(checkpoint_reference)
        )
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            "Could not reauthenticate the admitted coordinator/store chain."
        ) from exc
    if (
        state.frontier_reference != boundary_reference
        or state.checkpoint_reference != checkpoint_reference
        or state.frontier.frontier_sha256 != frontier.frontier_sha256
        or state.next_layer_index != frontier.next_layer_index
        or state.admission.get("admission_sha256") != admission_key
        or dict(admission) != dict(state.admission)
        or boundary.reference != boundary_reference
        or checkpoint.reference != checkpoint_reference
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "Reauthenticated store objects differ from the admitted state."
        )
    body = {
        "schema": LFM2_CPCF_V5_ADMISSION_CHAIN_PROOF_SCHEMA,
        "resume_schema": resume.LFM2_CPCF_V5_RESUME_SCHEMA,
        "terminal_document": _plain(terminal),
        "admission_document": _plain(admission),
        "pending_document": _plain(pending),
        "progress_document": _plain(progress),
        "boundary_manifest": _plain(boundary.manifest),
        "boundary_reports": _plain(boundary.reports),
        "boundary_certificate_sha256": boundary.certificate_sha256,
        "checkpoint_manifest": _plain(checkpoint.manifest),
        "checkpoint_completed_layers": _plain(checkpoint.completed_layers),
        "checkpoint_certificate_sha256": checkpoint.certificate_sha256,
        "store_artifacts_reauthenticated": True,
        "compiler_persisted_standalone_chain": True,
        "parent_or_cache_tensors_embedded": 0,
        "operation_audit": dict(_ZERO_AUDIT),
    }
    proof = {**body, "proof_sha256": canonical_sha256(body)}
    return authenticate_lfm2_cpcf_v5_admission_chain_proof(proof)


def authenticate_lfm2_cpcf_v5_admission_chain_proof(
    value: Any,
) -> Mapping[str, Any]:
    """Authenticate a persisted chain without the original mutable store."""

    proof = _mapping(value, "V5 admission-chain proof")
    body = dict(proof)
    supplied = _digest(
        body.pop("proof_sha256", None),
        "admission-chain proof SHA-256",
    )
    if (
        set(proof) != _PROOF_FIELDS
        or proof.get("schema") != LFM2_CPCF_V5_ADMISSION_CHAIN_PROOF_SCHEMA
        or proof.get("resume_schema") != resume.LFM2_CPCF_V5_RESUME_SCHEMA
        or canonical_sha256(body) != supplied
        or proof.get("store_artifacts_reauthenticated") is not True
        or proof.get("compiler_persisted_standalone_chain") is not True
        or proof.get("parent_or_cache_tensors_embedded") != 0
        or proof.get("operation_audit") != _ZERO_AUDIT
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "V5 admission-chain proof schema, seal, or safety claims are invalid."
        )
    terminal = _sealed_document(
        proof["terminal_document"],
        fields=resume._TERMINAL_FIELDS,
        schema=resume.LFM2_CPCF_V5_TERMINAL_SCHEMA,
        seal_name="terminal_sha256",
        name="terminal document",
    )
    admission = _sealed_document(
        proof["admission_document"],
        fields=resume._ADMISSION_FIELDS,
        schema=resume.LFM2_CPCF_V5_ADMISSION_SCHEMA,
        seal_name="admission_sha256",
        name="admission document",
    )
    pending = _sealed_document(
        proof["pending_document"],
        fields=_PENDING_FIELDS,
        schema=resume.LFM2_CPCF_V5_PENDING_SCHEMA,
        seal_name="pending_sha256",
        name="pending document",
    )
    progress = _sealed_document(
        proof["progress_document"],
        fields=resume._PROGRESS_FIELDS,
        schema=resume.LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA,
        seal_name="report_sha256",
        name="progress document",
    )
    boundary_reference = _reference(
        pending["frontier_reference"],
        kind="boundary",
    )
    checkpoint_reference = _reference(
        pending["checkpoint_reference"],
        kind="checkpoint",
    )
    frontier_sha256 = _digest(
        pending.get("frontier_sha256"),
        "pending frontier SHA-256",
    )
    event_sha256 = _digest(
        pending.get("event_report_sha256"),
        "pending event SHA-256",
    )
    admission_sha256 = _digest(
        admission.get("admission_sha256"),
        "admission SHA-256",
    )
    try:
        next_layer_index = resume._nonnegative_int(
            pending.get("next_layer_index"),
            "pending next_layer_index",
        )
        expectation = resume._rss_checkpoint_expectation(
            pending.get("rss_checkpoint_expectation"),
            next_layer_index=next_layer_index,
        )
        rss = resume._authenticate_rss_checkpoint(
            admission.get("rss_checkpoint_receipt"),
            expectation=expectation,
            passed=True,
            require_typed=False,
        )
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            "Pending RSS expectation or admitted receipt is invalid."
        ) from exc
    boundary_manifest, boundary_reports = _authenticate_boundary(
        manifest_value=proof["boundary_manifest"],
        reports_value=proof["boundary_reports"],
        certificate_sha256=proof["boundary_certificate_sha256"],
        reference=boundary_reference,
    )
    checkpoint_manifest, completed = _authenticate_checkpoint(
        manifest_value=proof["checkpoint_manifest"],
        completed_value=proof["checkpoint_completed_layers"],
        certificate_sha256=proof["checkpoint_certificate_sha256"],
        reference=checkpoint_reference,
        boundary_reference=boundary_reference,
    )
    frontier_body = dict(boundary_reports)
    supplied_frontier = _digest(
        frontier_body.pop("frontier_sha256", None),
        "boundary frontier SHA-256",
    )
    try:
        context = resume._context(boundary_reports.get("context"))
    except Exception as exc:
        raise Lfm2CpcfV5AdmissionProofError(
            "Boundary frontier context is invalid."
        ) from exc
    if (
        set(boundary_reports) != resume._FRONTIER_FIELDS
        or boundary_reports.get("schema")
        != resume.LFM2_CPCF_V5_BOUNDARY_FRONTIER_SCHEMA
        or boundary_reports.get("resume_schema")
        != resume.LFM2_CPCF_V5_RESUME_SCHEMA
        or canonical_sha256(frontier_body) != supplied_frontier
        or supplied_frontier != frontier_sha256
        or boundary_reports.get("next_layer_index") != next_layer_index
        or boundary_reports.get("operation_audit") != _ZERO_AUDIT
        or progress.get("context_sha256") != context["context_sha256"]
        or progress.get("frontier_sha256") != frontier_sha256
        or progress.get("next_layer_index") != next_layer_index
        or progress.get("report_sha256") != event_sha256
        or progress.get("pending_rss_admission") is not True
        or progress.get("rss_admission_claimed") is not False
        or progress.get("parent_or_cache_tensors_archived") != 0
        or progress.get("operation_audit") != _ZERO_AUDIT
        or pending.get("operation_audit") != _ZERO_AUDIT
        or pending.get("parent_or_cache_tensors_archived_in_frontier") != 0
        or pending.get("progress_event_persisted_and_reauthenticated") is not True
        or admission.get("pending_sha256") != pending["pending_sha256"]
        or admission.get("frontier_reference") != boundary_reference.as_dict()
        or admission.get("checkpoint_reference") != checkpoint_reference.as_dict()
        or admission.get("frontier_sha256") != frontier_sha256
        or admission.get("event_report_sha256") != event_sha256
        or admission.get("next_layer_index") != next_layer_index
        or admission.get("rss_checkpoint_expectation_sha256")
        != expectation["expectation_sha256"]
        or admission.get("rss_checkpoint_policy_binding")
        != resume._rss_checkpoint_policy_binding(expectation)
        or admission.get("rss_checkpoint_report_sha256")
        != rss["report_sha256"]
        or admission.get("pending_checkpoint_reauthenticated") is not True
        or admission.get("persisted_progress_event_recomputed") is not True
        or admission.get("original_policy_admitted") is not True
        or admission.get("readmission_used") is not False
        or admission.get("promotion_claimed") is not False
        or admission.get("operation_audit") != _ZERO_AUDIT
        or terminal.get("pending_sha256") != pending["pending_sha256"]
        or terminal.get("outcome_kind") != "admission"
        or terminal.get("outcome_sha256") != admission_sha256
        or terminal.get("outcome_filename") != f"{admission_sha256}.json"
        or terminal.get("exactly_one_terminal_outcome") is not True
        or terminal.get("operation_audit") != _ZERO_AUDIT
        or completed.get("next_layer_index") != next_layer_index
        or checkpoint_manifest["bindings"].get("boundary")
        != boundary_reference.as_dict()
        or boundary_manifest.get("artifact_key") != boundary_reference.artifact_key
    ):
        raise Lfm2CpcfV5AdmissionProofError(
            "Persisted admission, pending, progress, frontier, checkpoint, "
            "or terminal lineage differs."
        )
    return MappingProxyType(proof)


__all__ = [
    "LFM2_CPCF_V5_ADMISSION_CHAIN_PROOF_SCHEMA",
    "Lfm2CpcfV5AdmissionProofError",
    "authenticate_lfm2_cpcf_v5_admission_chain_proof",
    "build_lfm2_cpcf_v5_admission_chain_proof",
]
