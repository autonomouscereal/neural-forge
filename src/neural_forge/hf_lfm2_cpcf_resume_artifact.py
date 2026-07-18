"""Immutable, non-promotable intermediate artifacts for LFM2 CPCF.

This store deliberately keeps resumability evidence separate from accepted
children.  A boundary bundle, a prefix checkpoint, or a failure diagnostic is
useful for deterministic continuation and diagnosis, but none of those
artifacts is evidence that compaction worked.

The store has three fixed layouts::

    boundaries/<key>/
        manifest.json
        reports.json
        tensors.safetensors

    checkpoints/<key>/
        manifest.json
        prefix-model.safetensors
        incoming-state-bank.safetensors
        completed-layers.json

    failures/<key>/
        diagnostic.json

Boundary binding keys are algorithm-independent discovery identities, never
trust anchors.  Published boundary paths use a separate content key over those
bindings and the exact report/tensor payload records.  Checkpoint and
diagnostic keys likewise include their content seals.

Publication uses an exclusively created sibling staging directory followed by
an atomic no-replace directory rename.  An already-published byte-identical
artifact is reused; a different artifact at the same key is never changed.
Every open reauthenticates the exact inventory, canonical JSON, safetensors
metadata, file hashes, and every tensor's shape, dtype, and byte hash.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import socket
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256


RESUME_ARTIFACT_SCHEMA = "neural-forge-lfm2-cpcf-resume-artifact-v2"
BOUNDARY_BINDING_KEY_SCHEMA = "neural-forge-lfm2-cpcf-boundary-binding-key-v1"
BOUNDARY_KEY_SCHEMA = "neural-forge-lfm2-cpcf-boundary-content-key-v1"
CHECKPOINT_KEY_SCHEMA = "neural-forge-lfm2-cpcf-checkpoint-key-v2"
FAILURE_KEY_SCHEMA = "neural-forge-lfm2-cpcf-failure-key-v1"
TENSOR_FILE_SCHEMA = "neural-forge-lfm2-cpcf-resume-tensors-v1"
COMPLETED_LAYERS_SCHEMA = "neural-forge-lfm2-cpcf-completed-layers-v2"
TARGET_TOPOLOGY_SNAPSHOT_SCHEMA = "neural-forge-lfm2-cpcf-target-topology-snapshot-v1"
INSTALL_PROGRESS_SCHEMA = "neural-forge-lfm2-cpcf-install-progress-v1"
INCOMING_STATE_BANK_RUNTIME_SCHEMA = "neural-forge-lfm2-cpcf-target-runtime-v1"
STAGE_OWNER_SCHEMA = "neural-forge-lfm2-cpcf-stage-owner-v1"

_LAYER_RESULT_SCHEMA = "neural-forge-lfm2-cpcf-layer-compiler-v1"
_LAYER_FAILURE_SCHEMA = "neural-forge-lfm2-cpcf-layer-failure-v3"
_MODEL_FAILURE_SCHEMA = "neural-forge-lfm2-cpcf-model-failure-v3"

_LAYER_TRANSACTION_STAGES = frozenset(
    {
        "preinstall_replay",
        "operator_compile",
        "operator_install",
        "operator_replay",
        "mlp_compile",
        "mlp_install",
        "final_replay",
        "endpoint_evaluation",
        "immutability_audit",
        "result_seal",
    }
)
_MODEL_TRANSACTION_STAGES = frozenset(
    {
        "validate_request",
        "inventory_and_blank_authentication",
        "parameter_compaction_audit",
        "boundary_bundle_fit",
        "interface_install",
        "neutral_norm_install",
        "seed_incoming_bank",
        "layer_compile_install",
        "progress_checkpoint_publish",
        "final_norm_install",
        "final_audit",
        "result_seal",
    }
)

_STATUS = "non-promotable-intermediate"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_EVIDENCE_JSON_BYTES = 1024 * 1024 * 1024
_EVIDENCE_JSON_FILENAMES = frozenset(
    {
        "completed-layers.json",
        "diagnostic.json",
    }
)
_STAGE_PREFIX = ".cpcf-resume-"
_STAGE_SUFFIX = ".staging"
_STAGE_OWNER_FILE = ".owner.json"
_STAGE_KEY_FILE = ".cpcf-stage-key"
_STAGE_KEY_BYTES = 32

_BOUNDARY_FILES = frozenset({"manifest.json", "reports.json", "tensors.safetensors"})
_CHECKPOINT_FILES = frozenset(
    {
        "manifest.json",
        "prefix-model.safetensors",
        "incoming-state-bank.safetensors",
        "completed-layers.json",
    }
)
_FAILURE_FILES = frozenset({"diagnostic.json"})

_BOUNDARY_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "artifact_kind",
        "status",
        "artifact_key",
        "key_schema",
        "binding_key",
        "binding_key_schema",
        "bindings",
        "files",
        "promotion_authorized",
        "accepted_artifact",
        "behavior_capability_proven",
        "training_performed",
        "optimizer_steps",
        "manifest_sha256",
    }
)
_CHECKPOINT_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "artifact_kind",
        "status",
        "artifact_key",
        "key_schema",
        "bindings",
        "files",
        "promotion_authorized",
        "accepted_artifact",
        "behavior_capability_proven",
        "training_performed",
        "optimizer_steps",
        "manifest_sha256",
    }
)
_FAILURE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema",
        "artifact_kind",
        "status",
        "artifact_key",
        "key_schema",
        "bindings",
        "report",
        "promotion_authorized",
        "accepted_artifact",
        "compaction_success_proven",
        "behavior_success_proven",
        "success_claimed",
        "training_performed",
        "optimizer_steps",
        "diagnostic_sha256",
    }
)
_NON_PROMOTABLE_FLAGS = {
    "promotion_authorized": False,
    "accepted_artifact": False,
    "behavior_capability_proven": False,
    "training_performed": False,
    "optimizer_steps": 0,
}
_BOUNDARY_BINDING_FIELDS = frozenset(
    {
        "parent_sha256",
        "cache_sha256",
        "evidence_sha256",
        "phase_id",
        "trace_sha256",
        "bridge_sha256",
        "tokenizer_sha256",
        "source_table_sha256",
        "blank_sha256",
        "target_sha256",
        "boundary_policy_sha256",
        "solver_schema_sha256",
    }
)
_CHECKPOINT_EXTERNAL_BINDING_FIELDS = frozenset(
    {
        "boundary",
        "next_layer_index",
        "completed_fold_seals_sha256",
        "install_receipt_sha256",
        "full_policy_sha256",
        "target_topology_sha256",
        "compiler_fingerprint_sha256",
    }
)
_COMPLETED_LAYER_FIELDS = frozenset(
    {
        "schema",
        "completed_layer_indices",
        "next_layer_index",
        "completed_layer_receipts",
        "completed_layer_receipts_sha256",
        "fold_seals_sha256",
        "install_receipt_sha256",
        "installed_parameter_paths",
        "target_topology_snapshot",
        "target_topology_sha256",
        "incoming_state_layer_index",
        "incoming_state_bank_metadata",
        "incoming_state_bank_metadata_sha256",
        "incoming_state_bank_sha256",
        "incoming_state_entry_sha256",
    }
)
_REFERENCE_FIELDS = frozenset({"artifact_kind", "artifact_key", "manifest_sha256"})
_CHECKPOINT_DERIVED_BINDING_FIELDS = frozenset(
    {
        "prefix_model_state_sha256",
        "prefix_model_entry_sha256",
        "incoming_state_bank_sha256",
        "incoming_state_entry_sha256",
        "completed_layers_document_sha256",
    }
)
_FAILURE_BINDING_FIELDS = frozenset(
    {
        "invocation_sha256",
        "layer_index",
        "operator_id",
        "stage_id",
        "guard_proof_sha256",
        "parent_proof_sha256",
        "rollback_proof_sha256",
        "boundary",
        "checkpoint",
        "report_sha256",
    }
)

_TARGET_TOPOLOGY_FIELDS = frozenset(
    {
        "schema",
        "model_type",
        "layer_count",
        "tensor_metadata",
        "object_alias_groups",
        "storage_alias_groups",
        "all_named_parameters_and_buffers",
        "nonpersistent_buffers_included",
        "topology_sha256",
    }
)
_TARGET_TENSOR_METADATA_FIELDS = frozenset(
    {
        "role",
        "persistent",
        "shape",
        "dtype",
        "stride",
        "storage_offset",
        "original_device_type",
        "original_device_index",
    }
)
_INCOMING_STATE_BANK_FIELDS = frozenset(
    {
        "schema",
        "scope",
        "boundary",
        "layer_index",
        "phase_count",
        "hidden_size",
        "ordered_states",
        "actual_folded_upstream_states",
        "state_bank_sha256",
    }
)
_INCOMING_STATE_FIELDS = frozenset(
    {
        "tensor_name",
        "identity",
        "phase_sha256",
        "state_sha256",
    }
)
_COMPLETED_LAYER_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "layer_index",
        "operator_kind",
        "source_width",
        "target_width",
        "input_bank_sha256",
        "next_bank_sha256",
        "boundary_charts_sha256",
        "preinstall_replay_sha256",
        "operator_replay_sha256",
        "final_replay_sha256",
        "operator_artifact",
        "mlp_artifact",
        "operator_install_report",
        "mlp_install_report",
        "split_metrics",
        "split_phase_counts",
        "solver_evidence_policy",
        "actual_child_state_policy",
        "audit",
        "execution",
        "result_sha256",
    }
)
_LAYER_FAILURE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "layer_index",
        "operator_kind",
        "input_bank_sha256",
        "input_tensor_sha256",
        "boundary_charts_sha256",
        "boundary_charts_payload_sha256",
        "cause",
        "nested_short_conv_validation_report",
        "nested_short_conv_validation_report_sha256",
        "nested_short_conv_validation_report_canonical_bytes_sha256",
        "nested_short_conv_validation_report_preserved_byte_for_byte",
        "rollback_proof",
        "layer_result_issued",
        "promotion_claimed",
        "report_sha256",
    }
)
_MODEL_FAILURE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "active_layer_index",
        "active_operator_kind",
        "cause",
        "boundary_bundle_sha256",
        "current_incoming_bank_sha256",
        "current_incoming_layer_index",
        "completed_layer_count",
        "completed_layer_receipts",
        "failed_prefix_target_cpu_tensor_snapshot",
        "nested_layer_failure_report",
        "nested_layer_failure_report_sha256",
        "nested_layer_failure_report_canonical_bytes_sha256",
        "nested_layer_failure_report_preserved_byte_for_byte",
        "rollback_proof",
        "compiled_model_fold_issued",
        "parameter_compaction_success_receipt_issued",
        "promotion_claimed",
        "promotable_candidate",
        "failure_evidence_only",
        "report_sha256",
    }
)
_LAYER_ROLLBACK_PROOF_FIELDS = frozenset(
    {
        "target_state_sha256_before",
        "target_state_sha256_after",
        "target_full_inventory_sha256_before",
        "target_full_inventory_sha256_after",
        "target_tensor_bytes_restored",
        "target_tensor_identities_preserved",
        "target_tensor_aliases_preserved",
        "target_tensor_shapes_preserved",
        "target_tensor_dtypes_preserved",
        "target_all_named_parameters_and_buffers_restored",
        "target_parameter_and_buffer_registries_restored",
        "target_nonpersistent_buffer_membership_restored",
        "target_parameter_gradient_identity_and_bytes_restored",
        "target_parameter_gradient_storage_aliases_restored",
        "target_module_modes_restored",
        "source_state_sha256_before",
        "source_state_sha256_after",
        "source_full_inventory_sha256_before",
        "source_full_inventory_sha256_after",
        "source_state_preserved",
        "source_full_inventory_preserved",
        "evidence_sha256_before",
        "evidence_sha256_after",
        "evidence_preserved",
        "input_tensor_sha256_before",
        "input_tensor_sha256_after",
        "input_tensors_preserved",
        "boundary_charts_sha256_before",
        "boundary_charts_sha256_after",
        "boundary_charts_payload_sha256_before",
        "boundary_charts_payload_sha256_after",
        "boundary_charts_preserved",
        "exact_layer_rollback_proven",
        "proof_sha256",
    }
)
_MODEL_ROLLBACK_PROOF_FIELDS = frozenset(
    {
        "source_state_fingerprint_before",
        "source_state_fingerprint_after",
        "source_full_inventory_sha256_before",
        "source_full_inventory_sha256_after",
        "source_state_preserved",
        "source_full_inventory_preserved",
        "target_state_fingerprint_before",
        "target_state_fingerprint_after",
        "target_full_inventory_sha256_before",
        "target_full_inventory_sha256_after",
        "target_tensor_bytes_restored",
        "target_tensor_identities_preserved",
        "target_tensor_aliases_preserved",
        "target_all_named_parameters_and_buffers_restored",
        "target_parameter_and_buffer_registries_restored",
        "target_nonpersistent_buffers_restored",
        "target_parameter_gradient_identity_and_bytes_restored",
        "target_parameter_gradient_storage_aliases_restored",
        "target_module_modes_restored",
        "exact_whole_model_rollback_proven",
        "proof_sha256",
    }
)
_PREFIX_SNAPSHOT_RECEIPT_FIELDS = frozenset(
    {
        "state_fingerprint",
        "device",
        "contiguous",
        "tensor_count",
        "tensor_inventory",
        "object_alias_groups",
        "storage_alias_groups",
        "all_named_parameters_and_buffers",
        "nonpersistent_buffers_included",
        "snapshot_role",
        "snapshot_sha256",
    }
)
_PREFIX_SNAPSHOT_TENSOR_FIELDS = frozenset(
    {
        "shape",
        "dtype",
        "tensor_sha256",
        "role",
        "persistent",
        "original_stride",
        "original_storage_offset",
        "original_device_type",
        "original_device_index",
    }
)
_FALSE_SUCCESS_CLAIMS = frozenset(
    {
        "promotion_authorized",
        "accepted_artifact",
        "compaction_success_proven",
        "behavior_success_proven",
        "success_claimed",
        "compiled_model_fold_issued",
        "parameter_compaction_success_receipt_issued",
        "promotion_claimed",
        "promotable_candidate",
        "layer_result_issued",
    }
)
_ZERO_EXECUTION_FIELDS = frozenset(
    {
        "parent_model_forward_calls",
        "target_model_forward_calls",
        "target_layer_forward_calls",
        "generation_calls",
        "cache_generation_calls",
        "backward_calls",
        "optimizer_steps",
        "training_steps",
        "parameters_pruned",
        "parameters_quantized",
        "fallback_paths",
    }
)
_LAYER_PATH = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class Lfm2CpcfResumeArtifactError(RuntimeError):
    """Base failure for CPCF intermediate storage."""


class Lfm2CpcfResumeArtifactIntegrityError(Lfm2CpcfResumeArtifactError):
    """A path, manifest, document, or tensor failed authentication."""


class Lfm2CpcfResumeArtifactPublicationError(Lfm2CpcfResumeArtifactError):
    """An intermediate artifact could not be published atomically."""


class Lfm2CpcfResumeArtifactConflictError(Lfm2CpcfResumeArtifactPublicationError):
    """A different or invalid artifact already occupies a content key."""


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _safe_text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or _SAFE_TEXT.fullmatch(value) is None
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} must be a nonempty stable identifier."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} must be a non-negative integer."
        )
    return value


def _canonical_value(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} must contain finite canonical JSON values."
        ) from exc


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    normalized = _canonical_value(value, name)
    if not isinstance(normalized, dict):
        raise Lfm2CpcfResumeArtifactIntegrityError(f"{name} must be a JSON object.")
    return normalized


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(callable(is_junction) and is_junction(path))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows cannot fsync directory handles opened through os.open.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, RuntimeError) as exc:
        raise Lfm2CpcfResumeArtifactPublicationError(
            f"Could not exclusively write {path.name}."
        ) from exc


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json_bytes(_deep_thaw(value)) + b"\n"
    maximum_bytes = _maximum_json_bytes(path)
    if len(payload) > maximum_bytes:
        raise Lfm2CpcfResumeArtifactPublicationError(
            f"{path.name} canonical JSON payload is {len(payload)} bytes; "
            f"maximum allowed is {maximum_bytes} bytes."
        )
    _write_exclusive(path, payload)
    return payload


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _real_directory(path: Path, *, name: str) -> os.stat_result:
    try:
        result = path.lstat()
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} is missing or unreadable."
        ) from exc
    if not stat.S_ISDIR(result.st_mode) or _is_link_or_junction(path):
        raise Lfm2CpcfResumeArtifactIntegrityError(f"{name} must be a real directory.")
    return result


def _regular_file(path: Path, *, name: str) -> os.stat_result:
    try:
        result = path.lstat()
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} is missing or unreadable."
        ) from exc
    if (
        not stat.S_ISREG(result.st_mode)
        or _is_link_or_junction(path)
        or int(result.st_nlink) != 1
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} must be one unaliased regular file."
        )
    return result


def _read_regular_bytes(
    path: Path,
    *,
    name: str,
    maximum_bytes: int | None = None,
) -> bytes:
    """Read one pinned regular file and reject pathname/inode races.

    POSIX requires ``O_NOFOLLOW``.  Windows has no corresponding ``os.open``
    flag, so the lstat/fstat identity is checked before and after the read;
    opening a symlink follows a different inode and fails the comparison.
    """

    parent_before = _real_directory(path.parent, name=f"{name} parent")
    before = _regular_file(path, name=name)
    if maximum_bytes is not None and int(before.st_size) > maximum_bytes:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} exceeds the JSON size limit."
        )
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "This POSIX runtime cannot open artifact files without following links."
            )
        flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} could not be opened without following links."
        ) from exc
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or _stat_identity(opened) != _stat_identity(before)
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} changed identity while it was opened."
            )
        total = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum_bytes is not None and total > maximum_bytes:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    f"{name} exceeds the JSON size limit."
                )
            chunks.append(chunk)
        after_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} could not be read from its authenticated descriptor."
        ) from exc
    finally:
        os.close(descriptor)
    after = _regular_file(path, name=name)
    parent_after = _real_directory(path.parent, name=f"{name} parent")
    if (
        _stat_identity(after_descriptor) != _stat_identity(before)
        or int(after_descriptor.st_size) != int(before.st_size)
        or _stat_identity(after) != _stat_identity(before)
        or int(after.st_size) != int(before.st_size)
        or _stat_identity(parent_after) != _stat_identity(parent_before)
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} or its parent changed while it was read."
        )
    return b"".join(chunks)


def _stream_regular_sha256(
    path: Path,
    *,
    name: str,
) -> tuple[int, str]:
    """Hash one pinned regular file without retaining its payload in memory."""

    parent_before = _real_directory(path.parent, name=f"{name} parent")
    before = _regular_file(path, name=name)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "This POSIX runtime cannot hash artifact files without following links."
            )
        flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} could not be opened without following links."
        ) from exc
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or _stat_identity(opened) != _stat_identity(before)
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} changed identity while it was opened."
            )
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} could not be hashed from its authenticated descriptor."
        ) from exc
    finally:
        os.close(descriptor)
    after = _regular_file(path, name=name)
    parent_after = _real_directory(path.parent, name=f"{name} parent")
    if (
        _stat_identity(after_descriptor) != _stat_identity(before)
        or int(after_descriptor.st_size) != int(before.st_size)
        or total != int(before.st_size)
        or _stat_identity(after) != _stat_identity(before)
        or int(after.st_size) != int(before.st_size)
        or _stat_identity(parent_after) != _stat_identity(parent_before)
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} or its parent changed while it was hashed."
        )
    return total, digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return _stream_regular_sha256(path, name=path.name)[1]


def _canonical_json_from_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} is not valid canonical JSON."
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} is not a canonical JSON object."
        )
    return value


def _maximum_json_bytes(path: Path) -> int:
    if path.name in _EVIDENCE_JSON_FILENAMES:
        return _MAX_EVIDENCE_JSON_BYTES
    return _MAX_JSON_BYTES


def _read_canonical_json_with_raw(
    path: Path,
    *,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(
        path,
        name=name,
        maximum_bytes=_maximum_json_bytes(path),
    )
    return _canonical_json_from_bytes(raw, name=name), raw


def _read_canonical_json(path: Path, *, name: str) -> dict[str, Any]:
    value, _raw = _read_canonical_json_with_raw(path, name=name)
    return value


def _file_record(
    path: Path,
    *,
    tensor_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    size_bytes, file_sha256 = _stream_regular_sha256(path, name=path.name)
    record: dict[str, Any] = {
        "path": path.name,
        "size_bytes": size_bytes,
        "sha256": file_sha256,
    }
    if tensor_entries is None:
        record["format"] = "canonical-json"
    else:
        record["format"] = "safetensors"
        record["tensor_entries"] = tensor_entries
        record["tensor_set_sha256"] = canonical_sha256(tensor_entries)
    return record


def _tensor_byte_view(tensor: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "PyTorch is required for CPCF resume tensors."
        ) from exc
    value = tensor.detach().to(device="cpu").contiguous()
    if value.numel() == 0:
        return memoryview(b"")
    return memoryview(value.view(torch.uint8).numpy())


def _tensor_content_sha256(tensor: Any) -> str:
    return hashlib.sha256(_tensor_byte_view(tensor)).hexdigest()


def _runtime_tensor_sha256(tensor: Any) -> str:
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(canonical_json_bytes(descriptor))
    digest.update(_tensor_byte_view(tensor))
    return digest.hexdigest()


def _tensor_descriptor(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": [int(size) for size in tensor.shape],
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "tensor_sha256": _tensor_content_sha256(tensor),
        "runtime_tensor_sha256": _runtime_tensor_sha256(tensor),
    }


def _normalize_tensors(
    tensors: Mapping[str, Any],
    *,
    name: str,
    clone: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "PyTorch is required for CPCF resume tensors."
        ) from exc
    if not isinstance(tensors, Mapping) or not tensors:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} must contain at least one named tensor."
        )
    items = list(tensors.items())
    if any(not isinstance(key, str) for key, _source in items):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} requires string tensor names."
        )
    normalized: dict[str, Any] = {}
    for key, source in sorted(items, key=lambda item: item[0]):
        if (
            not isinstance(key, str)
            or not key
            or key.strip() != key
            or not isinstance(source, torch.Tensor)
            or source.device.type == "meta"
            or source.layout != torch.strided
            or source.is_quantized
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} requires named, materialized, dense tensors."
            )
        value = source.detach().to(device="cpu").contiguous()
        if clone:
            value = value.clone()
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} tensor {key!r} is not finite."
            )
        normalized[key] = value
    entries = [_tensor_descriptor(key, normalized[key]) for key in sorted(normalized)]
    return normalized, entries


def _save_safetensors(
    path: Path,
    tensors: Mapping[str, Any],
    *,
    artifact_kind: str,
    file_role: str,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise Lfm2CpcfResumeArtifactPublicationError(
            "safetensors is required for CPCF resume artifacts."
        ) from exc
    if _lexists(path):
        raise Lfm2CpcfResumeArtifactPublicationError(
            f"Refusing to replace staged tensor file {path.name}."
        )
    # safetensors metadata is encoded from a Rust HashMap.  Multiple metadata
    # keys can therefore produce byte-distinct headers across otherwise
    # identical saves.  One canonical contract value keeps cold/resumed
    # publications byte-identical.
    contract = canonical_json_bytes(
        {
            "schema": TENSOR_FILE_SCHEMA,
            "artifact_kind": artifact_kind,
            "file_role": file_role,
        }
    ).decode("ascii")
    metadata = {"contract": contract}
    try:
        save_file(dict(tensors), str(path), metadata=metadata)
        # Windows' _commit (used by os.fsync) rejects a read-only descriptor.
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2CpcfResumeArtifactPublicationError(
            f"Could not write {path.name}."
        ) from exc


def _safetensors_metadata(payload: bytes, *, name: str) -> dict[str, str]:
    if len(payload) < 8:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} has a truncated safetensors header."
        )
    header_size = int.from_bytes(payload[:8], byteorder="little", signed=False)
    if header_size <= 0 or header_size > len(payload) - 8:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} has an invalid safetensors header size."
        )
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} has invalid safetensors header JSON."
        ) from exc
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} safetensors metadata is invalid."
        )
    return metadata


def _load_safetensors(
    payload: bytes,
    *,
    name: str,
    artifact_kind: str,
    file_role: str,
    expected_entries: Any,
) -> dict[str, Any]:
    try:
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "safetensors is required for CPCF resume artifacts."
        ) from exc
    expected = _canonical_value(expected_entries, f"{name} tensor entries")
    if not isinstance(expected, list) or not expected:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} has no authenticated tensor entries."
        )
    try:
        metadata = _safetensors_metadata(payload, name=name)
        loaded = dict(load(payload))
        tensors = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in loaded.items()
        }
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} is not a readable safetensors file."
        ) from exc
    expected_contract = canonical_json_bytes(
        {
            "schema": TENSOR_FILE_SCHEMA,
            "artifact_kind": artifact_kind,
            "file_role": file_role,
        }
    ).decode("ascii")
    if metadata != {"contract": expected_contract}:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} safetensors metadata is invalid."
        )
    actual = [_tensor_descriptor(key, tensors[key]) for key in sorted(tensors)]
    if actual != expected:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} tensor inventory or content is invalid."
        )
    for tensor_name, value in tensors.items():
        if (value.is_floating_point() or value.is_complex()) and not bool(
            value.isfinite().all().item()
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} tensor {tensor_name!r} is not finite."
            )
    return tensors


def _descriptor_sha256(descriptor: int) -> tuple[int, str]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Could not hash an authenticated safetensors descriptor."
        ) from exc
    return total, digest.hexdigest()


def _load_safetensors_path(
    path: Path,
    *,
    name: str,
    artifact_kind: str,
    file_role: str,
    record: Mapping[str, Any],
    clone_tensors: bool,
) -> dict[str, Any]:
    """Inspect or load one safetensors file through a pinned file identity.

    Linux uses ``/proc/self/fd`` so the Rust safetensors reader opens the same
    inode that was inspected and hashed.  Windows falls back to the descriptor-
    authenticated byte reader because safetensors does not accept a Windows
    HANDLE or a Python buffer object.
    """

    expected_entries = _canonical_value(
        record.get("tensor_entries"), f"{name} tensor entries"
    )
    if not isinstance(expected_entries, list) or not expected_entries:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} has no authenticated tensor entries."
        )
    expected_size = record.get("size_bytes")
    expected_sha = _sha256(record.get("sha256"), f"{name} SHA-256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(f"{name} size is invalid.")

    if os.name != "posix" or not Path("/proc/self/fd").is_dir():
        payload = _read_regular_bytes(path, name=name)
        if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != (
            expected_sha
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} differs from its manifest record."
            )
        tensors = _load_safetensors(
            payload,
            name=name,
            artifact_kind=artifact_kind,
            file_role=file_role,
            expected_entries=expected_entries,
        )
        return tensors if clone_tensors else {}

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "safetensors is required for CPCF resume artifacts."
        ) from exc

    parent_before = _real_directory(path.parent, name=f"{name} parent")
    before = _regular_file(path, name=name)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "This POSIX runtime cannot pin safetensors without following links."
        )
    flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} could not be opened without following links."
        ) from exc
    tensors: dict[str, Any] = {}
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or _stat_identity(opened) != _stat_identity(before)
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} changed identity while it was opened."
            )
        size_before, sha_before = _descriptor_sha256(descriptor)
        if size_before != expected_size or sha_before != expected_sha:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} differs from its manifest record."
            )
        contract = canonical_json_bytes(
            {
                "schema": TENSOR_FILE_SCHEMA,
                "artifact_kind": artifact_kind,
                "file_role": file_role,
            }
        ).decode("ascii")
        pinned_path = f"/proc/self/fd/{descriptor}"
        try:
            with safe_open(
                pinned_path,
                framework="pt",
                device="cpu",
            ) as archive:
                if archive.metadata() != {"contract": contract}:
                    raise Lfm2CpcfResumeArtifactIntegrityError(
                        f"{name} safetensors metadata is invalid."
                    )
                actual_entries = []
                for tensor_name in sorted(archive.keys()):
                    value = archive.get_tensor(tensor_name)
                    if (value.is_floating_point() or value.is_complex()) and not bool(
                        value.isfinite().all().item()
                    ):
                        raise Lfm2CpcfResumeArtifactIntegrityError(
                            f"{name} tensor {tensor_name!r} is not finite."
                        )
                    actual_entries.append(_tensor_descriptor(tensor_name, value))
                    if clone_tensors:
                        tensors[tensor_name] = (
                            value.detach().to(device="cpu").contiguous().clone()
                        )
        except Lfm2CpcfResumeArtifactIntegrityError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} is not a readable safetensors file."
            ) from exc
        if actual_entries != expected_entries:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} tensor inventory or content is invalid."
            )
        size_after, sha_after = _descriptor_sha256(descriptor)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = _regular_file(path, name=name)
    parent_after = _real_directory(path.parent, name=f"{name} parent")
    if (
        size_after != expected_size
        or sha_after != expected_sha
        or _stat_identity(after_descriptor) != _stat_identity(before)
        or int(after_descriptor.st_size) != int(before.st_size)
        or _stat_identity(after) != _stat_identity(before)
        or int(after.st_size) != int(before.st_size)
        or _stat_identity(parent_after) != _stat_identity(parent_before)
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} or its parent changed while safetensors was loaded."
        )
    return tensors


@dataclass(frozen=True, slots=True)
class _DisposableTensorArchive:
    path: Path
    artifact_kind: str
    file_role: str
    _record_bytes: bytes

    @classmethod
    def from_record(
        cls,
        path: Path,
        *,
        artifact_kind: str,
        file_role: str,
        record: Mapping[str, Any],
    ) -> _DisposableTensorArchive:
        return cls(
            path=path,
            artifact_kind=artifact_kind,
            file_role=file_role,
            _record_bytes=canonical_json_bytes(_deep_thaw(record)),
        )

    def load(self) -> Mapping[str, Any]:
        record = json.loads(self._record_bytes.decode("ascii"))
        tensors = _load_safetensors_path(
            self.path,
            name=self.path.name,
            artifact_kind=self.artifact_kind,
            file_role=self.file_role,
            record=record,
            clone_tensors=True,
        )
        return MappingProxyType(tensors)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfBoundaryBindings:
    """Algorithm-independent inputs that address one boundary bundle."""

    parent_sha256: str
    cache_sha256: str
    evidence_sha256: str
    phase_id: str
    trace_sha256: str
    bridge_sha256: str
    tokenizer_sha256: str
    source_table_sha256: str
    blank_sha256: str
    target_sha256: str
    boundary_policy_sha256: str
    solver_schema_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "parent_sha256",
            "cache_sha256",
            "evidence_sha256",
            "trace_sha256",
            "bridge_sha256",
            "tokenizer_sha256",
            "source_table_sha256",
            "blank_sha256",
            "target_sha256",
            "boundary_policy_sha256",
            "solver_schema_sha256",
        ):
            _sha256(getattr(self, field_name), field_name)
        _safe_text(self.phase_id, "phase_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_sha256": self.parent_sha256,
            "cache_sha256": self.cache_sha256,
            "evidence_sha256": self.evidence_sha256,
            "phase_id": self.phase_id,
            "trace_sha256": self.trace_sha256,
            "bridge_sha256": self.bridge_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "source_table_sha256": self.source_table_sha256,
            "blank_sha256": self.blank_sha256,
            "target_sha256": self.target_sha256,
            "boundary_policy_sha256": self.boundary_policy_sha256,
            "solver_schema_sha256": self.solver_schema_sha256,
        }

    @property
    def binding_key(self) -> str:
        return canonical_sha256(
            {
                "schema": BOUNDARY_BINDING_KEY_SCHEMA,
                "bindings": self.as_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCheckpointBindings:
    """External seals required in addition to checkpoint tensor content."""

    boundary: Lfm2CpcfArtifactReference
    next_layer_index: int
    completed_fold_seals_sha256: str
    install_receipt_sha256: str
    full_policy_sha256: str
    target_topology_sha256: str
    compiler_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.boundary, Lfm2CpcfArtifactReference)
            or self.boundary.artifact_kind != "boundary"
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint boundary must be a pinned boundary reference."
            )
        for field_name in (
            "completed_fold_seals_sha256",
            "install_receipt_sha256",
            "full_policy_sha256",
            "target_topology_sha256",
            "compiler_fingerprint_sha256",
        ):
            _sha256(getattr(self, field_name), field_name)
        _nonnegative_int(self.next_layer_index, "next_layer_index")

    def as_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.as_dict(),
            "next_layer_index": self.next_layer_index,
            "completed_fold_seals_sha256": self.completed_fold_seals_sha256,
            "install_receipt_sha256": self.install_receipt_sha256,
            "full_policy_sha256": self.full_policy_sha256,
            "target_topology_sha256": self.target_topology_sha256,
            "compiler_fingerprint_sha256": self.compiler_fingerprint_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfArtifactReference:
    """Authenticated reference to a boundary or prefix checkpoint."""

    artifact_kind: Literal["boundary", "checkpoint"]
    artifact_key: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.artifact_kind not in {"boundary", "checkpoint"}:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "artifact_kind must be boundary or checkpoint."
            )
        _sha256(self.artifact_key, "artifact_key")
        _sha256(self.manifest_sha256, "manifest_sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_key": self.artifact_key,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfFailureBindings:
    """Execution and rollback evidence that addresses a failure diagnostic."""

    invocation_sha256: str
    layer_index: int | None
    operator_id: str | None
    stage_id: str
    guard_proof_sha256: str
    parent_proof_sha256: str
    rollback_proof_sha256: str
    boundary: Lfm2CpcfArtifactReference | None = None
    checkpoint: Lfm2CpcfArtifactReference | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "invocation_sha256",
            "guard_proof_sha256",
            "parent_proof_sha256",
            "rollback_proof_sha256",
        ):
            _sha256(getattr(self, field_name), field_name)
        if (self.layer_index is None) != (self.operator_id is None):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Failure layer_index and operator_id must both be present or absent."
            )
        if self.layer_index is not None:
            _nonnegative_int(self.layer_index, "layer_index")
            _safe_text(self.operator_id, "operator_id")
        _safe_text(self.stage_id, "stage_id")
        if self.boundary is not None and (
            not isinstance(self.boundary, Lfm2CpcfArtifactReference)
            or self.boundary.artifact_kind != "boundary"
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Failure boundary must be a pinned boundary reference."
            )
        if self.checkpoint is not None and (
            not isinstance(self.checkpoint, Lfm2CpcfArtifactReference)
            or self.checkpoint.artifact_kind != "checkpoint"
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Failure checkpoint must be a pinned checkpoint reference."
            )
        if self.boundary is None and self.checkpoint is None:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "A failure must reference a boundary or checkpoint."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "invocation_sha256": self.invocation_sha256,
            "layer_index": self.layer_index,
            "operator_id": self.operator_id,
            "stage_id": self.stage_id,
            "guard_proof_sha256": self.guard_proof_sha256,
            "parent_proof_sha256": self.parent_proof_sha256,
            "rollback_proof_sha256": self.rollback_proof_sha256,
            "boundary": (None if self.boundary is None else self.boundary.as_dict()),
            "checkpoint": (
                None if self.checkpoint is None else self.checkpoint.as_dict()
            ),
        }


def _reference_from_mapping(
    value: Any,
    *,
    name: str,
) -> Lfm2CpcfArtifactReference | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _REFERENCE_FIELDS:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} must be an exact CPCF artifact reference."
        )
    return Lfm2CpcfArtifactReference(
        artifact_kind=value["artifact_kind"],
        artifact_key=value["artifact_key"],
        manifest_sha256=value["manifest_sha256"],
    )


def _boundary_bindings_from_mapping(
    value: Any,
) -> Lfm2CpcfBoundaryBindings:
    if not isinstance(value, dict) or set(value) != _BOUNDARY_BINDING_FIELDS:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Boundary manifest does not have the exact v1 binding set."
        )
    return Lfm2CpcfBoundaryBindings(**value)


def _checkpoint_bindings_from_mapping(
    value: Any,
) -> tuple[Lfm2CpcfCheckpointBindings, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != (
        _CHECKPOINT_EXTERNAL_BINDING_FIELDS | _CHECKPOINT_DERIVED_BINDING_FIELDS
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Checkpoint manifest does not have the exact v1 binding set."
        )
    external = Lfm2CpcfCheckpointBindings(
        boundary=_reference_from_mapping(value["boundary"], name="checkpoint boundary"),
        next_layer_index=value["next_layer_index"],
        completed_fold_seals_sha256=value["completed_fold_seals_sha256"],
        install_receipt_sha256=value["install_receipt_sha256"],
        full_policy_sha256=value["full_policy_sha256"],
        target_topology_sha256=value["target_topology_sha256"],
        compiler_fingerprint_sha256=value["compiler_fingerprint_sha256"],
    )
    derived = {name: value[name] for name in _CHECKPOINT_DERIVED_BINDING_FIELDS}
    for name in (
        "prefix_model_state_sha256",
        "incoming_state_bank_sha256",
        "completed_layers_document_sha256",
    ):
        _sha256(derived[name], name)
    for name in (
        "prefix_model_entry_sha256",
        "incoming_state_entry_sha256",
    ):
        hashes = derived[name]
        if not isinstance(hashes, dict) or not hashes:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} must contain at least one tensor hash."
            )
        for tensor_name, digest in hashes.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    f"{name} contains an invalid tensor name."
                )
            _sha256(digest, f"{name}[{tensor_name!r}]")
    return external, derived


def _failure_bindings_from_mapping(
    value: Any,
) -> tuple[Lfm2CpcfFailureBindings, str]:
    if not isinstance(value, dict) or set(value) != _FAILURE_BINDING_FIELDS:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Failure diagnostic does not have the exact v1 binding set."
        )
    report_sha256 = _sha256(value["report_sha256"], "report_sha256")
    result = Lfm2CpcfFailureBindings(
        invocation_sha256=value["invocation_sha256"],
        layer_index=value["layer_index"],
        operator_id=value["operator_id"],
        stage_id=value["stage_id"],
        guard_proof_sha256=value["guard_proof_sha256"],
        parent_proof_sha256=value["parent_proof_sha256"],
        rollback_proof_sha256=value["rollback_proof_sha256"],
        boundary=_reference_from_mapping(value["boundary"], name="boundary reference"),
        checkpoint=_reference_from_mapping(
            value["checkpoint"], name="checkpoint reference"
        ),
    )
    return result, report_sha256


def _reject_nested_success_claims(value: Any, *, path: str = "failure report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FALSE_SUCCESS_CLAIMS and child is not False:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    f"{path}.{key} makes a success or promotion claim."
                )
            if key == "failure_evidence_only" and child is not True:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    f"{path}.failure_evidence_only must be exactly true."
                )
            _reject_nested_success_claims(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nested_success_claims(child, path=f"{path}[{index}]")


def _self_sealed_mapping(
    value: Any,
    *,
    name: str,
    seal_field: str,
) -> tuple[dict[str, Any], str]:
    normalized = _canonical_mapping(value, name)
    body = dict(normalized)
    claimed = _sha256(body.pop(seal_field, None), f"{name} {seal_field}")
    if canonical_sha256(body) != claimed:
        raise Lfm2CpcfResumeArtifactIntegrityError(f"{name} self-hash is invalid.")
    return normalized, claimed


def _canonical_nested_report(
    report: Any,
    *,
    name: str,
    claimed_sha256: Any,
    claimed_bytes_sha256: Any,
    preserved: Any,
) -> dict[str, Any] | None:
    if report is None:
        if (
            claimed_sha256 is not None
            or claimed_bytes_sha256 is not None
            or preserved is not False
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Absent {name} evidence has inconsistent seals."
            )
        return None
    nested, nested_sha = _self_sealed_mapping(
        report,
        name=name,
        seal_field="report_sha256",
    )
    expected_bytes_sha = hashlib.sha256(canonical_json_bytes(nested)).hexdigest()
    if (
        _sha256(claimed_sha256, f"{name} SHA-256") != nested_sha
        or _sha256(
            claimed_bytes_sha256,
            f"{name} canonical bytes SHA-256",
        )
        != expected_bytes_sha
        or preserved is not True
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} is not preserved byte-for-byte with exact seals."
        )
    _reject_nested_success_claims(nested, path=name)
    return nested


def _validate_failure_cause(value: Any, *, name: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"module", "type", "message"}
        or any(not isinstance(value.get(field), str) for field in value)
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} failure cause is not canonical."
        )


def _validate_exact_rollback_proof(
    proof: Mapping[str, Any],
    *,
    expected_fields: frozenset[str],
    digest_fields: Sequence[str],
    true_fields: Sequence[str],
    equal_digest_pairs: Sequence[tuple[str, str]],
    name: str,
) -> None:
    if set(proof) != expected_fields:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} does not have the exact known proof schema."
        )
    for field in digest_fields:
        _sha256(proof.get(field), f"{name} {field}")
    if any(proof.get(field) is not True for field in true_fields):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{name} does not prove every required restoration invariant."
        )
    for before, after in equal_digest_pairs:
        if proof.get(before) != proof.get(after):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{name} differs between {before} and {after}."
            )


def _validate_prefix_snapshot_receipt(value: Any) -> tuple[dict[str, Any], str]:
    snapshot, snapshot_sha = _self_sealed_mapping(
        value,
        name="failed prefix target snapshot",
        seal_field="snapshot_sha256",
    )
    inventory = snapshot.get("tensor_inventory")
    if (
        set(snapshot) != _PREFIX_SNAPSHOT_RECEIPT_FIELDS
        or snapshot.get("device") != "cpu"
        or snapshot.get("contiguous") is not True
        or snapshot.get("all_named_parameters_and_buffers") is not True
        or snapshot.get("nonpersistent_buffers_included") is not True
        or snapshot.get("snapshot_role")
        != "failed_transaction_prefix_before_whole_model_rollback"
        or not isinstance(inventory, dict)
        or not inventory
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure prefix snapshot does not have the exact receipt schema."
        )
    _sha256(
        snapshot.get("state_fingerprint"),
        "failed prefix target state fingerprint",
    )
    tensor_count = _nonnegative_int(
        snapshot.get("tensor_count"),
        "failed prefix target tensor_count",
    )
    if tensor_count == 0 or tensor_count != len(inventory):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure prefix snapshot tensor count is inconsistent."
        )
    normalized_inventory: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in sorted(inventory.items()):
        tensor_name = _safe_text(raw_name, "failed prefix tensor name")
        if (
            not isinstance(raw_record, dict)
            or set(raw_record) != _PREFIX_SNAPSHOT_TENSOR_FIELDS
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Failed prefix tensor receipt for {tensor_name!r} is invalid."
            )
        record = dict(raw_record)
        shape = record.get("shape")
        stride = record.get("original_stride")
        role = record.get("role")
        persistent = record.get("persistent")
        if (
            not isinstance(shape, list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            )
            or not isinstance(stride, list)
            or len(stride) != len(shape)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in stride
            )
            or role not in {"parameter", "buffer"}
            or type(persistent) is not bool
            or (role == "parameter" and persistent is not True)
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Failed prefix tensor geometry for {tensor_name!r} is invalid."
            )
        _safe_text(
            record.get("dtype"),
            f"failed prefix tensor dtype for {tensor_name!r}",
        )
        _sha256(
            record.get("tensor_sha256"),
            f"failed prefix tensor SHA-256 for {tensor_name!r}",
        )
        _nonnegative_int(
            record.get("original_storage_offset"),
            f"failed prefix storage offset for {tensor_name!r}",
        )
        if record.get("original_device_type") not in {
            "cpu",
            "cuda",
            "mps",
            "xpu",
        }:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Failed prefix tensor device for {tensor_name!r} is invalid."
            )
        device_index = record.get("original_device_index")
        if device_index is not None:
            _nonnegative_int(
                device_index,
                f"failed prefix device index for {tensor_name!r}",
            )
        normalized_inventory[tensor_name] = record
    names = set(normalized_inventory)
    _alias_groups(
        snapshot.get("object_alias_groups"),
        names=names,
        field_name="failed prefix object_alias_groups",
    )
    _alias_groups(
        snapshot.get("storage_alias_groups"),
        names=names,
        field_name="failed prefix storage_alias_groups",
    )
    return snapshot, snapshot_sha


def _validate_layer_failure_report(
    value: Any,
    *,
    bindings: Lfm2CpcfFailureBindings | None,
) -> dict[str, Any]:
    report, _report_sha = _self_sealed_mapping(
        value,
        name="layer failure report",
        seal_field="report_sha256",
    )
    if (
        set(report) != _LAYER_FAILURE_REPORT_FIELDS
        or report.get("schema") != _LAYER_FAILURE_SCHEMA
        or report.get("operator_kind") not in {"full_attention", "short_conv"}
        or report.get("layer_result_issued") is not False
        or report.get("promotion_claimed") is not False
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Layer failure report does not have the exact known failure schema."
        )
    layer_index = _nonnegative_int(
        report.get("layer_index"), "layer failure layer_index"
    )
    stage = _safe_text(report.get("stage"), "layer failure stage")
    if stage not in _LAYER_TRANSACTION_STAGES:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Layer failure stage is not a known transaction stage."
        )
    for name in (
        "input_bank_sha256",
        "input_tensor_sha256",
        "boundary_charts_sha256",
        "boundary_charts_payload_sha256",
    ):
        _sha256(report.get(name), f"layer failure {name}")
    _validate_failure_cause(report.get("cause"), name="layer")
    proof, proof_sha = _self_sealed_mapping(
        report.get("rollback_proof"),
        name="layer rollback proof",
        seal_field="proof_sha256",
    )
    _validate_exact_rollback_proof(
        proof,
        expected_fields=_LAYER_ROLLBACK_PROOF_FIELDS,
        digest_fields=(
            "target_state_sha256_before",
            "target_state_sha256_after",
            "target_full_inventory_sha256_before",
            "target_full_inventory_sha256_after",
            "source_state_sha256_before",
            "source_state_sha256_after",
            "source_full_inventory_sha256_before",
            "source_full_inventory_sha256_after",
            "evidence_sha256_before",
            "evidence_sha256_after",
            "input_tensor_sha256_before",
            "input_tensor_sha256_after",
            "boundary_charts_sha256_before",
            "boundary_charts_sha256_after",
            "boundary_charts_payload_sha256_before",
            "boundary_charts_payload_sha256_after",
        ),
        true_fields=(
            "target_tensor_bytes_restored",
            "target_tensor_identities_preserved",
            "target_tensor_aliases_preserved",
            "target_tensor_shapes_preserved",
            "target_tensor_dtypes_preserved",
            "target_all_named_parameters_and_buffers_restored",
            "target_parameter_and_buffer_registries_restored",
            "target_nonpersistent_buffer_membership_restored",
            "target_parameter_gradient_identity_and_bytes_restored",
            "target_parameter_gradient_storage_aliases_restored",
            "target_module_modes_restored",
            "source_state_preserved",
            "source_full_inventory_preserved",
            "evidence_preserved",
            "input_tensors_preserved",
            "boundary_charts_preserved",
            "exact_layer_rollback_proven",
        ),
        equal_digest_pairs=(
            ("target_state_sha256_before", "target_state_sha256_after"),
            (
                "target_full_inventory_sha256_before",
                "target_full_inventory_sha256_after",
            ),
            ("source_state_sha256_before", "source_state_sha256_after"),
            (
                "source_full_inventory_sha256_before",
                "source_full_inventory_sha256_after",
            ),
            ("evidence_sha256_before", "evidence_sha256_after"),
            ("input_tensor_sha256_before", "input_tensor_sha256_after"),
            (
                "boundary_charts_sha256_before",
                "boundary_charts_sha256_after",
            ),
            (
                "boundary_charts_payload_sha256_before",
                "boundary_charts_payload_sha256_after",
            ),
        ),
        name="layer rollback proof",
    )
    if (
        proof["input_tensor_sha256_before"] != report["input_tensor_sha256"]
        or proof["boundary_charts_sha256_before"] != report["boundary_charts_sha256"]
        or proof["boundary_charts_payload_sha256_before"]
        != report["boundary_charts_payload_sha256"]
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Layer rollback proof differs from the failure evidence digests."
        )
    parent_proof = _sha256(
        proof.get("source_full_inventory_sha256_before"),
        "layer parent proof",
    )
    if proof.get("source_full_inventory_sha256_after") != parent_proof:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Layer parent inventory differs across rollback."
        )
    _canonical_nested_report(
        report.get("nested_short_conv_validation_report"),
        name="nested short-conv validation report",
        claimed_sha256=report.get("nested_short_conv_validation_report_sha256"),
        claimed_bytes_sha256=report.get(
            "nested_short_conv_validation_report_canonical_bytes_sha256"
        ),
        preserved=report.get(
            "nested_short_conv_validation_report_preserved_byte_for_byte"
        ),
    )
    _reject_nested_success_claims(report)
    if bindings is not None and (
        bindings.layer_index != layer_index
        or bindings.operator_id != report["operator_kind"]
        or bindings.stage_id != stage
        or bindings.guard_proof_sha256 != report["boundary_charts_payload_sha256"]
        or bindings.parent_proof_sha256 != parent_proof
        or bindings.rollback_proof_sha256 != proof_sha
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Layer failure report differs from its bound execution context."
        )
    return report


def _validate_model_failure_report(
    value: Any,
    *,
    bindings: Lfm2CpcfFailureBindings,
) -> dict[str, Any]:
    report, _report_sha = _self_sealed_mapping(
        value,
        name="model failure report",
        seal_field="report_sha256",
    )
    if (
        set(report) != _MODEL_FAILURE_REPORT_FIELDS
        or report.get("schema") != _MODEL_FAILURE_SCHEMA
        or report.get("compiled_model_fold_issued") is not False
        or report.get("parameter_compaction_success_receipt_issued") is not False
        or report.get("promotion_claimed") is not False
        or report.get("promotable_candidate") is not False
        or report.get("failure_evidence_only") is not True
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure report does not have the exact known failure schema."
        )
    stage = _safe_text(report.get("stage"), "model failure stage")
    if stage not in _MODEL_TRANSACTION_STAGES:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure stage is not a known transaction stage."
        )
    _validate_failure_cause(report.get("cause"), name="model")
    proof, proof_sha = _self_sealed_mapping(
        report.get("rollback_proof"),
        name="model rollback proof",
        seal_field="proof_sha256",
    )
    _validate_exact_rollback_proof(
        proof,
        expected_fields=_MODEL_ROLLBACK_PROOF_FIELDS,
        digest_fields=(
            "source_state_fingerprint_before",
            "source_state_fingerprint_after",
            "source_full_inventory_sha256_before",
            "source_full_inventory_sha256_after",
            "target_state_fingerprint_before",
            "target_state_fingerprint_after",
            "target_full_inventory_sha256_before",
            "target_full_inventory_sha256_after",
        ),
        true_fields=(
            "source_state_preserved",
            "source_full_inventory_preserved",
            "target_tensor_bytes_restored",
            "target_tensor_identities_preserved",
            "target_tensor_aliases_preserved",
            "target_all_named_parameters_and_buffers_restored",
            "target_parameter_and_buffer_registries_restored",
            "target_nonpersistent_buffers_restored",
            "target_parameter_gradient_identity_and_bytes_restored",
            "target_parameter_gradient_storage_aliases_restored",
            "target_module_modes_restored",
            "exact_whole_model_rollback_proven",
        ),
        equal_digest_pairs=(
            (
                "source_state_fingerprint_before",
                "source_state_fingerprint_after",
            ),
            (
                "source_full_inventory_sha256_before",
                "source_full_inventory_sha256_after",
            ),
            (
                "target_state_fingerprint_before",
                "target_state_fingerprint_after",
            ),
            (
                "target_full_inventory_sha256_before",
                "target_full_inventory_sha256_after",
            ),
        ),
        name="model rollback proof",
    )
    parent_proof = _sha256(
        proof.get("source_state_fingerprint_before"),
        "model parent proof",
    )
    snapshot, snapshot_sha = _validate_prefix_snapshot_receipt(
        report.get("failed_prefix_target_cpu_tensor_snapshot"),
    )
    boundary_sha = report.get("boundary_bundle_sha256")
    current_bank_sha = report.get("current_incoming_bank_sha256")
    current_bank_index = report.get("current_incoming_layer_index")
    if boundary_sha is not None:
        _sha256(boundary_sha, "model failure boundary bundle SHA-256")
    if (current_bank_sha is None) != (current_bank_index is None):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure incoming bank digest and layer index must travel together."
        )
    if current_bank_sha is not None:
        _sha256(current_bank_sha, "model failure incoming bank SHA-256")
        _nonnegative_int(
            current_bank_index,
            "model failure current incoming layer index",
        )
    completed_count = _nonnegative_int(
        report.get("completed_layer_count"),
        "model failure completed_layer_count",
    )
    if completed_count:
        _sha256(current_bank_sha, "model failure incoming bank SHA-256")
    if current_bank_index is not None and current_bank_index != completed_count:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure incoming bank is not the completed prefix endpoint."
        )
    receipts = _completed_layer_receipts(
        report.get("completed_layer_receipts"),
        next_layer_index=completed_count,
        incoming_state_bank_sha256=(current_bank_sha if completed_count else "0" * 64),
    )
    if receipts != report["completed_layer_receipts"]:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure completed receipts are not canonical."
        )
    nested = _canonical_nested_report(
        report.get("nested_layer_failure_report"),
        name="nested layer failure report",
        claimed_sha256=report.get("nested_layer_failure_report_sha256"),
        claimed_bytes_sha256=report.get(
            "nested_layer_failure_report_canonical_bytes_sha256"
        ),
        preserved=report.get("nested_layer_failure_report_preserved_byte_for_byte"),
    )
    if nested is not None:
        validated_nested = _validate_layer_failure_report(nested, bindings=None)
        if (
            validated_nested["layer_index"] != report.get("active_layer_index")
            or validated_nested["operator_kind"] != report.get("active_operator_kind")
            or validated_nested["input_bank_sha256"] != current_bank_sha
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Nested layer failure differs from active model context."
            )
    active_index = report.get("active_layer_index")
    active_kind = report.get("active_operator_kind")
    if stage == "layer_compile_install":
        if (
            isinstance(active_index, bool)
            or not isinstance(active_index, int)
            or active_index < 0
            or active_kind not in {"full_attention", "short_conv"}
            or bindings.layer_index != active_index
            or bindings.operator_id != active_kind
            or report.get("current_incoming_layer_index") != active_index
            or completed_count != active_index
            or boundary_sha is None
            or current_bank_sha is None
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Active model failure context is not contiguous."
            )
    elif (
        active_index is not None
        or active_kind is not None
        or bindings.layer_index is not None
        or bindings.operator_id is not None
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Inactive model failure carries active layer context."
        )
    _reject_nested_success_claims(report)
    if (
        bindings.stage_id != stage
        or bindings.guard_proof_sha256 != snapshot_sha
        or bindings.parent_proof_sha256 != parent_proof
        or bindings.rollback_proof_sha256 != proof_sha
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Model failure report differs from its bound execution context."
        )
    return report


def _validated_known_failure_report(
    value: Any,
    *,
    bindings: Lfm2CpcfFailureBindings,
) -> dict[str, Any]:
    normalized = _canonical_mapping(value, "failure report")
    schema = normalized.get("schema")
    if schema == _LAYER_FAILURE_SCHEMA:
        return _validate_layer_failure_report(normalized, bindings=bindings)
    if schema == _MODEL_FAILURE_SCHEMA:
        return _validate_model_failure_report(normalized, bindings=bindings)
    raise Lfm2CpcfResumeArtifactIntegrityError(
        "Failure report must use an exact known LFM2 CPCF failure schema."
    )


@dataclass(frozen=True, slots=True)
class VerifiedLfm2CpcfBoundaryBundle:
    path: Path
    artifact_key: str
    manifest: Mapping[str, Any]
    reports: Mapping[str, Any]
    _tensor_archive: _DisposableTensorArchive
    certificate_sha256: str
    reused_existing: bool = False

    @property
    def reference(self) -> Lfm2CpcfArtifactReference:
        return Lfm2CpcfArtifactReference(
            artifact_kind="boundary",
            artifact_key=self.artifact_key,
            manifest_sha256=str(self.manifest["manifest_sha256"]),
        )

    @property
    def binding_key(self) -> str:
        return str(self.manifest["binding_key"])

    @property
    def tensors(self) -> Mapping[str, Any]:
        return self._tensor_archive.load()


@dataclass(frozen=True, slots=True)
class VerifiedLfm2CpcfPrefixCheckpoint:
    path: Path
    artifact_key: str
    manifest: Mapping[str, Any]
    _prefix_model_archive: _DisposableTensorArchive
    _incoming_state_bank_archive: _DisposableTensorArchive
    completed_layers: Mapping[str, Any]
    certificate_sha256: str
    reused_existing: bool = False

    @property
    def reference(self) -> Lfm2CpcfArtifactReference:
        return Lfm2CpcfArtifactReference(
            artifact_kind="checkpoint",
            artifact_key=self.artifact_key,
            manifest_sha256=str(self.manifest["manifest_sha256"]),
        )

    @property
    def prefix_model(self) -> Mapping[str, Any]:
        return self._prefix_model_archive.load()

    @property
    def incoming_state_bank(self) -> Mapping[str, Any]:
        return self._incoming_state_bank_archive.load()


@dataclass(frozen=True, slots=True)
class VerifiedLfm2CpcfFailureDiagnostic:
    path: Path
    artifact_key: str
    diagnostic: Mapping[str, Any]
    certificate_sha256: str
    reused_existing: bool = False


def _manifest_body(
    *,
    kind: Literal["boundary", "checkpoint"],
    artifact_key: str,
    key_schema: str,
    bindings: Mapping[str, Any],
    files: list[dict[str, Any]],
    binding_key: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema": RESUME_ARTIFACT_SCHEMA,
        "artifact_kind": kind,
        "status": _STATUS,
        "artifact_key": artifact_key,
        "key_schema": key_schema,
        "bindings": _canonical_mapping(bindings, "artifact bindings"),
        "files": files,
        **_NON_PROMOTABLE_FLAGS,
    }
    if kind == "boundary":
        result["binding_key"] = _sha256(binding_key, "boundary binding_key")
        result["binding_key_schema"] = BOUNDARY_BINDING_KEY_SCHEMA
    elif binding_key is not None:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Checkpoint manifests cannot carry a boundary binding key."
        )
    return result


def _seal_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _deep_thaw(body)
    return {
        **normalized,
        "manifest_sha256": canonical_sha256(normalized),
    }


def _verify_non_promotable(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("status") != _STATUS
        or manifest.get("promotion_authorized") is not False
        or manifest.get("accepted_artifact") is not False
        or manifest.get("behavior_capability_proven") is not False
        or manifest.get("training_performed") is not False
        or type(manifest.get("optimizer_steps")) is not int
        or manifest.get("optimizer_steps") != 0
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume artifact is not fixed non-promotable evidence."
        )


def _verify_inventory(
    root: Path,
    expected: frozenset[str],
) -> tuple[int, int]:
    try:
        root_stat = root.lstat()
        entries = list(os.scandir(root))
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume artifact directory is unreadable."
        ) from exc
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or _is_link_or_junction(root)
        or {entry.name for entry in entries} != expected
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume artifact inventory is unsafe or contains extra files."
        )
    for entry in entries:
        path = root / entry.name
        _regular_file(path, name=entry.name)
    after = _real_directory(root, name="resume artifact directory")
    if _stat_identity(after) != _stat_identity(root_stat):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume artifact directory changed during inventory."
        )
    return _stat_identity(root_stat)


def _records_by_path(
    manifest: Mapping[str, Any],
    expected_payloads: frozenset[str],
) -> dict[str, dict[str, Any]]:
    raw = manifest.get("files")
    if not isinstance(raw, list):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume manifest file inventory is invalid."
        )
    records: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or set(item) not in (
            {"path", "size_bytes", "sha256", "format"},
            {
                "path",
                "size_bytes",
                "sha256",
                "format",
                "tensor_entries",
                "tensor_set_sha256",
            },
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Resume manifest file record is invalid."
            )
        path = item.get("path")
        if (
            not isinstance(path, str)
            or "/" in path
            or "\\" in path
            or path in {"", ".", ".."}
            or path in records
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Resume manifest contains an unsafe or duplicate path."
            )
        _sha256(item.get("sha256"), f"{path} SHA-256")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Lfm2CpcfResumeArtifactIntegrityError(f"{path} size is invalid.")
        records[path] = item
    if set(records) != expected_payloads:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume manifest does not describe the exact payload inventory."
        )
    return records


def _verify_file_record(
    root: Path,
    record: Mapping[str, Any],
) -> bytes | None:
    path = root / str(record["path"])
    if path.suffix == ".json":
        raw = _read_regular_bytes(
            path,
            name=path.name,
            maximum_bytes=_maximum_json_bytes(path),
        )
        actual_size = len(raw)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
    else:
        actual_size, actual_sha256 = _stream_regular_sha256(
            path,
            name=path.name,
        )
        raw = None
    if actual_size != record["size_bytes"] or actual_sha256 != record["sha256"]:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{path.name} differs from its manifest record."
        )
    return raw


def _certificate(manifest_raw: bytes, records: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "files": [
                {
                    "path": name,
                    "size_bytes": records[name]["size_bytes"],
                    "sha256": records[name]["sha256"],
                }
                for name in sorted(records)
            ],
        }
    )


def _artifact_directory(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"Resume artifact does not exist: {candidate}."
        ) from exc
    if resolved != candidate or _is_link_or_junction(candidate):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume artifact path must not traverse a symlink or junction."
        )
    _real_directory(resolved, name="resume artifact directory")
    return resolved


def _authenticate_manifest(
    root: Path,
    *,
    artifact_key: str,
    kind: Literal["boundary", "checkpoint"],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    bytes,
    dict[str, bytes],
]:
    expected_files = _BOUNDARY_FILES if kind == "boundary" else _CHECKPOINT_FILES
    directory_identity = _verify_inventory(root, expected_files)
    manifest_path = root / "manifest.json"
    manifest, raw = _read_canonical_json_with_raw(manifest_path, name="manifest.json")
    expected_fields = (
        _BOUNDARY_MANIFEST_FIELDS if kind == "boundary" else _CHECKPOINT_MANIFEST_FIELDS
    )
    if (
        set(manifest) != expected_fields
        or manifest.get("schema") != RESUME_ARTIFACT_SCHEMA
        or manifest.get("artifact_kind") != kind
        or manifest.get("artifact_key") != artifact_key
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume manifest schema, kind, or key is invalid."
        )
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if canonical_sha256(body) != claimed:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume manifest self-hash is invalid."
        )
    _verify_non_promotable(manifest)
    payloads = expected_files - {"manifest.json"}
    records = _records_by_path(manifest, payloads)
    authenticated_payloads = {}
    for name in sorted(records):
        payload = _verify_file_record(root, records[name])
        if payload is not None:
            authenticated_payloads[name] = payload
    if _verify_inventory(root, expected_files) != directory_identity:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Resume artifact directory changed during authentication."
        )
    return manifest, records, raw, authenticated_payloads


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return _sha256(manifest.get("manifest_sha256"), "manifest_sha256")


def _boundary_content_key(
    *,
    bindings: Lfm2CpcfBoundaryBindings,
    files: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_sha256(
        {
            "schema": BOUNDARY_KEY_SCHEMA,
            "binding_key": bindings.binding_key,
            "bindings": bindings.as_dict(),
            "files": [_deep_thaw(item) for item in files],
        }
    )


def _checkpoint_content_key(
    *,
    bindings: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_sha256(
        {
            "schema": CHECKPOINT_KEY_SCHEMA,
            "bindings": _deep_thaw(bindings),
            "files": [_deep_thaw(item) for item in files],
        }
    )


def _entry_hashes(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(item["name"]): _sha256(
            item["tensor_sha256"],
            f"tensor entry {item['name']!r}",
        )
        for item in entries
    }


def _installed_parameter_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "installed_parameter_paths must be a canonical list."
        )
    result = []
    for index, item in enumerate(value):
        result.append(_safe_text(item, f"installed_parameter_paths[{index}]"))
    if result != sorted(set(result)):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "installed_parameter_paths must be unique and sorted."
        )
    return result


def _alias_groups(
    value: Any,
    *,
    names: set[str],
    field_name: str,
) -> list[list[str]]:
    if not isinstance(value, list):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{field_name} must be a canonical list."
        )
    groups: list[list[str]] = []
    flattened: list[str] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, list) or not group:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{field_name}[{group_index}] must be a nonempty list."
            )
        normalized = [
            _safe_text(name, f"{field_name}[{group_index}] name") for name in group
        ]
        if normalized != sorted(set(normalized)):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"{field_name} contains a noncanonical alias group."
            )
        groups.append(normalized)
        flattened.extend(normalized)
    if (
        groups != sorted(groups)
        or len(flattened) != len(set(flattened))
        or set(flattened) != names
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            f"{field_name} must cover every tensor exactly once."
        )
    return groups


def _target_topology(
    value: Any,
    *,
    prefix_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot = _canonical_mapping(value, "target topology snapshot")
    if (
        set(snapshot) != _TARGET_TOPOLOGY_FIELDS
        or snapshot.get("schema") != TARGET_TOPOLOGY_SNAPSHOT_SCHEMA
        or snapshot.get("model_type") != "lfm2"
        or snapshot.get("all_named_parameters_and_buffers") is not True
        or snapshot.get("nonpersistent_buffers_included") is not True
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Target topology snapshot does not have the exact LFM2 schema."
        )
    layer_count = _nonnegative_int(snapshot.get("layer_count"), "topology layer_count")
    if layer_count == 0:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Target topology must contain at least one layer."
        )
    raw_metadata = snapshot.get("tensor_metadata")
    if not isinstance(raw_metadata, dict) or not raw_metadata:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Target topology tensor metadata cannot be empty."
        )
    tensor_metadata: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in sorted(raw_metadata.items()):
        name = _safe_text(raw_name, "target topology tensor name")
        if (
            not isinstance(raw_record, dict)
            or set(raw_record) != _TARGET_TENSOR_METADATA_FIELDS
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Target topology metadata for {name!r} is invalid."
            )
        record = dict(raw_record)
        role = record.get("role")
        persistent = record.get("persistent")
        if (
            role not in {"parameter", "buffer"}
            or type(persistent) is not bool
            or (role == "parameter" and persistent is not True)
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Target topology tensor role for {name!r} is invalid."
            )
        shape = record.get("shape")
        stride = record.get("stride")
        if (
            not isinstance(shape, list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            )
            or not isinstance(stride, list)
            or len(stride) != len(shape)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in stride
            )
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Target topology shape or stride for {name!r} is invalid."
            )
        _safe_text(record.get("dtype"), f"topology dtype for {name!r}")
        _nonnegative_int(
            record.get("storage_offset"),
            f"topology storage offset for {name!r}",
        )
        if record.get("original_device_type") not in {
            "cpu",
            "cuda",
            "mps",
            "xpu",
        }:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                f"Target topology device for {name!r} is invalid."
            )
        device_index = record.get("original_device_index")
        if device_index is not None:
            _nonnegative_int(
                device_index,
                f"topology device index for {name!r}",
            )
        tensor_metadata[name] = record
    names = set(tensor_metadata)
    object_alias_groups = _alias_groups(
        snapshot.get("object_alias_groups"),
        names=names,
        field_name="object_alias_groups",
    )
    storage_alias_groups = _alias_groups(
        snapshot.get("storage_alias_groups"),
        names=names,
        field_name="storage_alias_groups",
    )
    body = dict(snapshot)
    claimed = _sha256(body.pop("topology_sha256", None), "target topology SHA-256")
    if canonical_sha256(body) != claimed:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Target topology snapshot self-hash is invalid."
        )
    if prefix_entries is not None:
        by_name = {str(item.get("name")): item for item in prefix_entries}
        if set(by_name) != names:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Prefix model tensors differ from the target topology snapshot."
            )
        for name, metadata in tensor_metadata.items():
            entry = by_name[name]
            if (
                entry.get("shape") != metadata["shape"]
                or str(entry.get("dtype", "")).removeprefix("torch.")
                != metadata["dtype"]
            ):
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    f"Prefix tensor {name!r} differs from target topology."
                )
        for group in object_alias_groups:
            hashes = {by_name[name]["tensor_sha256"] for name in group}
            if len(hashes) != 1:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    "Object-aliased prefix tensors do not have identical values."
                )
    return {
        **snapshot,
        "tensor_metadata": tensor_metadata,
        "object_alias_groups": object_alias_groups,
        "storage_alias_groups": storage_alias_groups,
    }


def _validate_installed_paths_against_topology(
    paths: Sequence[str],
    *,
    topology: Mapping[str, Any],
    next_layer_index: int,
) -> None:
    tensor_metadata = topology["tensor_metadata"]
    parameters = {
        name
        for name, record in tensor_metadata.items()
        if record["role"] == "parameter"
    }
    if any(path not in parameters for path in paths):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Installed parameter paths must name target topology parameters."
        )
    if next_layer_index > int(topology["layer_count"]):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Checkpoint progress exceeds the target topology layer count."
        )
    covered_layers = set()
    for path in paths:
        match = _LAYER_PATH.search(path)
        if match is None:
            continue
        layer_index = int(match.group(1))
        if layer_index >= next_layer_index:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Installed parameter paths include an uncompleted target layer."
            )
        covered_layers.add(layer_index)
    if covered_layers != set(range(next_layer_index)):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Installed parameter paths do not cover every completed target layer."
        )


def _completed_layer_receipts(
    value: Any,
    *,
    next_layer_index: int,
    incoming_state_bank_sha256: str,
) -> list[dict[str, Any]]:
    receipts = _canonical_value(value, "completed layer receipts")
    if not isinstance(receipts, list) or len(receipts) != next_layer_index:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed layer receipts must exactly match the completed prefix."
        )
    previous_next_bank: str | None = None
    result = []
    for expected_index, raw_receipt in enumerate(receipts):
        if (
            not isinstance(raw_receipt, dict)
            or set(raw_receipt) != _COMPLETED_LAYER_RECEIPT_FIELDS
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "A completed layer receipt has an unexpected field set."
            )
        receipt = dict(raw_receipt)
        claimed = _sha256(
            receipt.pop("result_sha256", None),
            f"completed layer {expected_index} result SHA-256",
        )
        if (
            canonical_sha256(receipt) != claimed
            or receipt.get("schema") != _LAYER_RESULT_SCHEMA
            or receipt.get("layer_index") != expected_index
            or receipt.get("operator_kind") not in {"full_attention", "short_conv"}
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Completed layer receipt is out of order or not self-sealed."
            )
        for name in (
            "input_bank_sha256",
            "next_bank_sha256",
            "boundary_charts_sha256",
            "preinstall_replay_sha256",
            "operator_replay_sha256",
            "final_replay_sha256",
        ):
            _sha256(receipt.get(name), f"completed layer {expected_index} {name}")
        if previous_next_bank is not None and (
            receipt["input_bank_sha256"] != previous_next_bank
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Completed layer state-bank seals are not causally contiguous."
            )
        previous_next_bank = receipt["next_bank_sha256"]
        execution = receipt.get("execution")
        if (
            not isinstance(execution, dict)
            or set(execution) != (_ZERO_EXECUTION_FIELDS | {"closed_form_compilers"})
            or execution.get("closed_form_compilers") is not True
            or any(
                type(execution.get(name)) is not int or execution.get(name) != 0
                for name in _ZERO_EXECUTION_FIELDS
            )
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Completed layer receipt does not prove zero-training execution."
            )
        for name in (
            "operator_artifact",
            "mlp_artifact",
            "operator_install_report",
            "mlp_install_report",
            "split_metrics",
            "split_phase_counts",
            "solver_evidence_policy",
            "actual_child_state_policy",
            "audit",
        ):
            if not isinstance(receipt.get(name), dict):
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    f"Completed layer receipt {name} is not canonical."
                )
        result.append({**receipt, "result_sha256": claimed})
    if result and previous_next_bank != incoming_state_bank_sha256:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "The final completed layer does not produce the archived incoming bank."
        )
    return result


def _incoming_state_bank_metadata(
    value: Any,
    *,
    next_layer_index: int,
    incoming_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = _canonical_mapping(value, "incoming state-bank metadata")
    if (
        set(metadata) != _INCOMING_STATE_BANK_FIELDS
        or metadata.get("schema") != INCOMING_STATE_BANK_RUNTIME_SCHEMA
        or metadata.get("scope") != "phase_state_bank"
        or metadata.get("actual_folded_upstream_states") is not True
        or metadata.get("layer_index") != next_layer_index
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank metadata does not have the exact runtime schema."
        )
    _safe_text(metadata.get("boundary"), "incoming state-bank boundary")
    phase_count = _nonnegative_int(
        metadata.get("phase_count"), "incoming state-bank phase_count"
    )
    hidden_size = _nonnegative_int(
        metadata.get("hidden_size"), "incoming state-bank hidden_size"
    )
    if phase_count == 0 or hidden_size == 0:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank metadata cannot be empty."
        )
    entries_by_name = {str(item.get("name")): item for item in incoming_entries}
    ordered = metadata.get("ordered_states")
    if not isinstance(ordered, list) or len(ordered) != phase_count:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank ordered state inventory is invalid."
        )
    runtime_inventory = []
    seen_names = set()
    seen_identities = set()
    dtypes = set()
    for index, raw_state in enumerate(ordered):
        if not isinstance(raw_state, dict) or set(raw_state) != _INCOMING_STATE_FIELDS:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state-bank entry has an unexpected field set."
            )
        tensor_name = _safe_text(
            raw_state.get("tensor_name"),
            f"incoming state tensor name {index}",
        )
        identity = raw_state.get("identity")
        if (
            not isinstance(identity, list)
            or len(identity) != 2
            or isinstance(identity[0], bool)
            or not isinstance(identity[0], int)
            or identity[0] < 0
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state-bank identity is invalid."
            )
        phase = _safe_text(identity[1], "incoming state-bank phase")
        identity_key = (identity[0], phase)
        if tensor_name in seen_names or identity_key in seen_identities:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state-bank names and identities must be unique."
            )
        seen_names.add(tensor_name)
        seen_identities.add(identity_key)
        try:
            tensor_entry = entries_by_name[tensor_name]
        except KeyError as exc:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state-bank metadata names an absent tensor."
            ) from exc
        shape = tensor_entry.get("shape")
        dtype = str(tensor_entry.get("dtype", ""))
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in shape
            )
            or shape[-1] != hidden_size
            or (
                "float" not in dtype.removeprefix("torch.")
                and "bfloat" not in dtype.removeprefix("torch.")
            )
            or "complex" in dtype
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state tensor shape or dtype is not a runtime residual."
            )
        dtypes.add(dtype)
        phase_sha = _sha256(
            raw_state.get("phase_sha256"),
            "incoming state phase SHA-256",
        )
        state_sha = _sha256(
            raw_state.get("state_sha256"),
            "incoming state tensor SHA-256",
        )
        if state_sha != tensor_entry.get("runtime_tensor_sha256"):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming runtime state seal differs from stored tensor bytes."
            )
        runtime_inventory.append(
            {
                "identity": [identity[0], phase],
                "phase_sha256": phase_sha,
                "state_sha256": state_sha,
            }
        )
    if seen_names != set(entries_by_name) or len(dtypes) != 1:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank metadata does not cover one uniform tensor bank."
        )
    runtime_body = {
        "schema": INCOMING_STATE_BANK_RUNTIME_SCHEMA,
        "scope": "phase_state_bank",
        "boundary": metadata["boundary"],
        "layer_index": next_layer_index,
        "hidden_size": hidden_size,
        "ordered_states": runtime_inventory,
        "actual_folded_upstream_states": True,
    }
    claimed = _sha256(
        metadata.get("state_bank_sha256"),
        "incoming state-bank SHA-256",
    )
    if canonical_sha256(runtime_body) != claimed:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank runtime seal is invalid."
        )
    return metadata


def _install_receipt_sha256(
    *,
    completed_layer_receipts_sha256: str,
    installed_parameter_paths: Sequence[str],
    target_topology_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema": INSTALL_PROGRESS_SCHEMA,
            "completed_layer_receipts_sha256": completed_layer_receipts_sha256,
            "installed_parameter_paths": list(installed_parameter_paths),
            "target_topology_sha256": target_topology_sha256,
        }
    )


def build_lfm2_cpcf_target_topology_snapshot(
    *,
    prefix_model: Mapping[str, Any],
    layer_count: int,
    snapshot_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert exact model-snapshot metadata into the canonical target topology."""

    count = _nonnegative_int(layer_count, "target topology layer_count")
    if count == 0:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Target topology must contain at least one layer."
        )
    _prefix, entries = _normalize_tensors(
        prefix_model,
        name="prefix model",
        clone=False,
    )
    metadata = _canonical_mapping(snapshot_metadata, "prefix snapshot metadata")
    expected_fields = {
        "schema",
        "all_named_parameters_and_buffers",
        "nonpersistent_buffers_included",
        "tensor_metadata",
        "object_alias_groups",
        "storage_alias_groups",
    }
    if (
        set(metadata) != expected_fields
        or metadata.get("schema") != "neural-forge-model-tensor-snapshot-metadata-v1"
        or metadata.get("all_named_parameters_and_buffers") is not True
        or metadata.get("nonpersistent_buffers_included") is not True
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Prefix snapshot metadata does not have the exact model schema."
        )
    tensor_metadata = metadata.get("tensor_metadata")
    if not isinstance(tensor_metadata, dict) or set(tensor_metadata) != {
        entry["name"] for entry in entries
    }:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Prefix snapshot metadata differs from prefix tensor names."
        )
    body = {
        "schema": TARGET_TOPOLOGY_SNAPSHOT_SCHEMA,
        "model_type": "lfm2",
        "layer_count": count,
        "tensor_metadata": tensor_metadata,
        "object_alias_groups": metadata["object_alias_groups"],
        "storage_alias_groups": metadata["storage_alias_groups"],
        "all_named_parameters_and_buffers": True,
        "nonpersistent_buffers_included": True,
    }
    result = {**body, "topology_sha256": canonical_sha256(body)}
    return _target_topology(result, prefix_entries=entries)


def build_lfm2_cpcf_installed_parameter_paths(
    *,
    target_topology_snapshot: Mapping[str, Any],
    next_layer_index: int,
    global_parameter_paths: Sequence[str] = (),
) -> list[str]:
    """Build one sorted install inventory for the completed target prefix."""

    topology = _target_topology(target_topology_snapshot)
    next_index = _nonnegative_int(next_layer_index, "next_layer_index")
    if isinstance(global_parameter_paths, (str, bytes)):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "global_parameter_paths must be a sequence of names."
        )
    globals_ = {
        _safe_text(path, "global installed parameter path")
        for path in global_parameter_paths
    }
    layer_paths = set()
    for name, metadata in topology["tensor_metadata"].items():
        if metadata["role"] != "parameter":
            continue
        match = _LAYER_PATH.search(name)
        if match is not None and int(match.group(1)) < next_index:
            layer_paths.add(name)
    result = sorted(globals_ | layer_paths)
    _validate_installed_paths_against_topology(
        result,
        topology=topology,
        next_layer_index=next_index,
    )
    return result


def build_lfm2_cpcf_checkpoint_bindings(
    *,
    boundary: Lfm2CpcfArtifactReference,
    completed_layer_receipts: Sequence[Mapping[str, Any]],
    installed_parameter_paths: Sequence[str],
    target_topology_snapshot: Mapping[str, Any],
    incoming_state_bank_metadata: Mapping[str, Any],
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
) -> Lfm2CpcfCheckpointBindings:
    """Compute every checkpoint progress digest instead of trusting caller math."""

    if (
        not isinstance(boundary, Lfm2CpcfArtifactReference)
        or boundary.artifact_kind != "boundary"
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Checkpoint bindings require a pinned boundary reference."
        )
    topology = _target_topology(target_topology_snapshot)
    incoming_metadata = _canonical_mapping(
        incoming_state_bank_metadata,
        "incoming state-bank metadata",
    )
    normalized_receipts = _canonical_value(
        completed_layer_receipts,
        "completed layer receipts",
    )
    if not isinstance(normalized_receipts, list):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "completed_layer_receipts must be a canonical sequence."
        )
    next_index = len(normalized_receipts)
    if (
        set(incoming_metadata) != _INCOMING_STATE_BANK_FIELDS
        or incoming_metadata.get("schema") != INCOMING_STATE_BANK_RUNTIME_SCHEMA
        or incoming_metadata.get("scope") != "phase_state_bank"
        or incoming_metadata.get("actual_folded_upstream_states") is not True
        or incoming_metadata.get("layer_index") != next_index
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Checkpoint bindings require exact typed incoming-bank metadata."
        )
    ordered_states = incoming_metadata.get("ordered_states")
    phase_count = _nonnegative_int(
        incoming_metadata.get("phase_count"),
        "incoming state-bank phase_count",
    )
    hidden_size = _nonnegative_int(
        incoming_metadata.get("hidden_size"),
        "incoming state-bank hidden_size",
    )
    if (
        not isinstance(ordered_states, list)
        or len(ordered_states) != phase_count
        or phase_count == 0
        or hidden_size == 0
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Checkpoint incoming-bank metadata is empty or unordered."
        )
    runtime_inventory = []
    seen_names = set()
    seen_identities = set()
    for state in ordered_states:
        if not isinstance(state, dict) or set(state) != _INCOMING_STATE_FIELDS:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint incoming-bank state metadata is invalid."
            )
        tensor_name = _safe_text(state["tensor_name"], "incoming state tensor name")
        identity = state["identity"]
        if (
            not isinstance(identity, list)
            or len(identity) != 2
            or isinstance(identity[0], bool)
            or not isinstance(identity[0], int)
            or identity[0] < 0
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint incoming-bank state identity is invalid."
            )
        phase_name = _safe_text(identity[1], "incoming state phase")
        identity_key = (identity[0], phase_name)
        if tensor_name in seen_names or identity_key in seen_identities:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint incoming-bank states are duplicated."
            )
        seen_names.add(tensor_name)
        seen_identities.add(identity_key)
        runtime_inventory.append(
            {
                "identity": [identity[0], phase_name],
                "phase_sha256": _sha256(
                    state["phase_sha256"],
                    "incoming state phase SHA-256",
                ),
                "state_sha256": _sha256(
                    state["state_sha256"],
                    "incoming state SHA-256",
                ),
            }
        )
    runtime_body = {
        "schema": INCOMING_STATE_BANK_RUNTIME_SCHEMA,
        "scope": "phase_state_bank",
        "boundary": _safe_text(
            incoming_metadata.get("boundary"),
            "incoming state-bank boundary",
        ),
        "layer_index": next_index,
        "hidden_size": hidden_size,
        "ordered_states": runtime_inventory,
        "actual_folded_upstream_states": True,
    }
    if canonical_sha256(runtime_body) != _sha256(
        incoming_metadata.get("state_bank_sha256"),
        "incoming state-bank SHA-256",
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Checkpoint incoming-bank runtime seal is invalid."
        )
    receipts = _completed_layer_receipts(
        normalized_receipts,
        next_layer_index=next_index,
        incoming_state_bank_sha256=_sha256(
            incoming_metadata.get("state_bank_sha256"),
            "incoming state-bank SHA-256",
        ),
    )
    paths = _installed_parameter_paths(list(installed_parameter_paths))
    _validate_installed_paths_against_topology(
        paths,
        topology=topology,
        next_layer_index=next_index,
    )
    receipts_sha = canonical_sha256(receipts)
    fold_seals_sha = canonical_sha256(
        [receipt["result_sha256"] for receipt in receipts]
    )
    topology_sha = topology["topology_sha256"]
    return Lfm2CpcfCheckpointBindings(
        boundary=boundary,
        next_layer_index=next_index,
        completed_fold_seals_sha256=fold_seals_sha,
        install_receipt_sha256=_install_receipt_sha256(
            completed_layer_receipts_sha256=receipts_sha,
            installed_parameter_paths=paths,
            target_topology_sha256=topology_sha,
        ),
        full_policy_sha256=_sha256(full_policy_sha256, "full_policy_sha256"),
        target_topology_sha256=topology_sha,
        compiler_fingerprint_sha256=_sha256(
            compiler_fingerprint_sha256,
            "compiler_fingerprint_sha256",
        ),
    )


def _validate_completed_layers_document(
    value: Any,
    *,
    bindings: Lfm2CpcfCheckpointBindings,
    prefix_entries: Sequence[Mapping[str, Any]] | None,
    incoming_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    document = _canonical_mapping(value, "completed layers")
    if set(document) != _COMPLETED_LAYER_FIELDS:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed-layers document does not have the exact v1 field set."
        )
    if document.get("schema") != COMPLETED_LAYERS_SCHEMA:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed-layers schema is invalid."
        )
    next_index = _nonnegative_int(
        document.get("next_layer_index"), "completed next_layer_index"
    )
    incoming_index = _nonnegative_int(
        document.get("incoming_state_layer_index"),
        "incoming_state_layer_index",
    )
    completed = document.get("completed_layer_indices")
    if (
        not isinstance(completed, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in completed
        )
        or completed != list(range(next_index))
        or next_index != bindings.next_layer_index
        or incoming_index != next_index
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed layers must be contiguous and end at the incoming state layer."
        )
    topology = _target_topology(
        document.get("target_topology_snapshot"),
        prefix_entries=prefix_entries,
    )
    topology_sha = _sha256(
        document.get("target_topology_sha256"),
        "completed target topology",
    )
    if (
        topology_sha != topology["topology_sha256"]
        or topology_sha != bindings.target_topology_sha256
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed target topology differs from checkpoint bindings."
        )
    paths = _installed_parameter_paths(document.get("installed_parameter_paths"))
    _validate_installed_paths_against_topology(
        paths,
        topology=topology,
        next_layer_index=next_index,
    )
    entry_hashes = _entry_hashes(incoming_entries)
    claimed_entry_hashes = document.get("incoming_state_entry_sha256")
    if (
        not isinstance(claimed_entry_hashes, dict)
        or claimed_entry_hashes != entry_hashes
        or _sha256(
            document.get("incoming_state_bank_sha256"),
            "completed incoming state bank",
        )
        != canonical_sha256(list(incoming_entries))
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed-layer incoming-state seals differ from stored tensors."
        )
    incoming_metadata = _incoming_state_bank_metadata(
        document.get("incoming_state_bank_metadata"),
        next_layer_index=next_index,
        incoming_entries=incoming_entries,
    )
    incoming_metadata_sha = _sha256(
        document.get("incoming_state_bank_metadata_sha256"),
        "incoming state-bank metadata SHA-256",
    )
    if incoming_metadata_sha != canonical_sha256(incoming_metadata):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank metadata seal is invalid."
        )
    receipts = _completed_layer_receipts(
        document.get("completed_layer_receipts"),
        next_layer_index=next_index,
        incoming_state_bank_sha256=incoming_metadata["state_bank_sha256"],
    )
    receipts_sha = _sha256(
        document.get("completed_layer_receipts_sha256"),
        "completed layer receipts SHA-256",
    )
    fold_seals_sha = _sha256(
        document.get("fold_seals_sha256"),
        "completed fold seals",
    )
    expected_receipts_sha = canonical_sha256(receipts)
    expected_fold_seals_sha = canonical_sha256(
        [receipt["result_sha256"] for receipt in receipts]
    )
    expected_install_sha = _install_receipt_sha256(
        completed_layer_receipts_sha256=expected_receipts_sha,
        installed_parameter_paths=paths,
        target_topology_sha256=topology_sha,
    )
    if (
        receipts_sha != expected_receipts_sha
        or fold_seals_sha != expected_fold_seals_sha
        or fold_seals_sha != bindings.completed_fold_seals_sha256
        or _sha256(
            document.get("install_receipt_sha256"),
            "completed install receipt",
        )
        != expected_install_sha
        or expected_install_sha != bindings.install_receipt_sha256
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed layer receipts, fold seals, or install receipt are invalid."
        )
    return document


def build_lfm2_cpcf_incoming_state_bank_metadata(
    *,
    boundary: str,
    layer_index: int,
    ordered_phases: Sequence[Mapping[str, Any]],
    incoming_state_bank: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the typed runtime receipt for one archived incoming state bank."""

    normalized_boundary = _safe_text(boundary, "incoming state-bank boundary")
    index = _nonnegative_int(layer_index, "incoming state-bank layer_index")
    _incoming, entries = _normalize_tensors(
        incoming_state_bank,
        name="incoming state bank",
        clone=False,
    )
    entries_by_name = {entry["name"]: entry for entry in entries}
    phases = _canonical_value(ordered_phases, "incoming state-bank phases")
    if not isinstance(phases, list) or not phases:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank phases cannot be empty."
        )
    ordered_states = []
    runtime_inventory = []
    for phase_index, phase in enumerate(phases):
        if not isinstance(phase, dict) or set(phase) != {
            "tensor_name",
            "identity",
            "phase_sha256",
        }:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state-bank phase descriptor is invalid."
            )
        tensor_name = _safe_text(
            phase["tensor_name"],
            f"incoming state tensor name {phase_index}",
        )
        if tensor_name not in entries_by_name:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state-bank phase names an absent tensor."
            )
        identity = phase["identity"]
        if (
            not isinstance(identity, list)
            or len(identity) != 2
            or isinstance(identity[0], bool)
            or not isinstance(identity[0], int)
            or identity[0] < 0
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming state-bank phase identity is invalid."
            )
        phase_name = _safe_text(identity[1], "incoming state-bank phase")
        phase_sha = _sha256(
            phase["phase_sha256"],
            "incoming state-bank phase SHA-256",
        )
        state_sha = entries_by_name[tensor_name]["runtime_tensor_sha256"]
        ordered_states.append(
            {
                "tensor_name": tensor_name,
                "identity": [identity[0], phase_name],
                "phase_sha256": phase_sha,
                "state_sha256": state_sha,
            }
        )
        runtime_inventory.append(
            {
                "identity": [identity[0], phase_name],
                "phase_sha256": phase_sha,
                "state_sha256": state_sha,
            }
        )
    if {item["tensor_name"] for item in ordered_states} != set(entries_by_name):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming phase descriptors must cover every tensor exactly once."
        )
    hidden_sizes = {int(entry["shape"][-1]) for entry in entries}
    if len(hidden_sizes) != 1:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming state-bank residual widths differ."
        )
    runtime_body = {
        "schema": INCOMING_STATE_BANK_RUNTIME_SCHEMA,
        "scope": "phase_state_bank",
        "boundary": normalized_boundary,
        "layer_index": index,
        "hidden_size": next(iter(hidden_sizes)),
        "ordered_states": runtime_inventory,
        "actual_folded_upstream_states": True,
    }
    result = {
        **runtime_body,
        "phase_count": len(ordered_states),
        "ordered_states": ordered_states,
        "state_bank_sha256": canonical_sha256(runtime_body),
    }
    return _incoming_state_bank_metadata(
        result,
        next_layer_index=index,
        incoming_entries=entries,
    )


def flatten_lfm2_cpcf_incoming_state_bank(
    state_bank: Any,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Flatten a typed live or failure-snapshot phase-state bank.

    Live banks return borrowed child-state tensors.  Failure snapshots are
    consumed only through their public ``phase_identities``,
    ``phase_references``, and ``state(identity)`` accessors; their external
    parent-evidence references are never materialized.  ``publish_checkpoint``
    reauthenticates the completed safetensors file against descriptors captured
    before the write, so a concurrent mutation fails publication instead of
    silently changing content.
    """

    boundary = _safe_text(
        getattr(state_bank, "boundary", None),
        "incoming state-bank boundary",
    )
    layer_index = _nonnegative_int(
        getattr(state_bank, "layer_index", None),
        "incoming state-bank layer_index",
    )
    phases = getattr(state_bank, "phases", None)
    states = getattr(state_bank, "states", None)
    claimed_bank_sha = _sha256(
        getattr(state_bank, "state_bank_sha256", None),
        "incoming state-bank SHA-256",
    )
    live_bank = (
        isinstance(phases, tuple) and bool(phases) and isinstance(states, Mapping)
    )
    snapshot_identities = getattr(state_bank, "phase_identities", None)
    snapshot_references = getattr(state_bank, "phase_references", None)
    snapshot_state = getattr(state_bank, "state", None)
    snapshot_bank = (
        isinstance(snapshot_identities, tuple)
        and bool(snapshot_identities)
        and isinstance(snapshot_references, tuple)
        and callable(snapshot_state)
    )
    if live_bank == snapshot_bank:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming bank must expose exactly one typed live-bank or "
            "failure-snapshot interface."
        )
    if snapshot_bank:
        verify = getattr(state_bank, "verify", None)
        if not callable(verify):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming failure snapshot must expose typed verification."
            )
        try:
            verify()
        except Exception as exc:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming failure snapshot verification failed."
            ) from exc
        if len(snapshot_references) != len(snapshot_identities):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming failure snapshot phase references are incomplete."
            )
    tensors = {}
    ordered_phases = []
    observed_identities = []
    phase_count = len(phases) if live_bank else len(snapshot_identities)
    for ordinal in range(phase_count):
        if live_bank:
            phase = phases[ordinal]
            identity = getattr(phase, "identity", None)
            phase_sha_value = getattr(phase, "phase_sha256", None)
        else:
            identity = snapshot_identities[ordinal]
            reference = snapshot_references[ordinal]
            if not isinstance(reference, Mapping):
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    "Incoming failure snapshot phase reference is invalid."
                )
            reference_identity = reference.get("identity")
            if (
                not isinstance(reference_identity, (list, tuple))
                or tuple(reference_identity) != identity
            ):
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    "Incoming failure snapshot reference identity changed."
                )
            phase_sha_value = reference.get("phase_sha256")
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or isinstance(identity[0], bool)
            or not isinstance(identity[0], int)
            or identity[0] < 0
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Incoming bank phase identity is invalid."
            )
        phase_name = _safe_text(identity[1], "incoming bank phase")
        phase_sha = _sha256(
            phase_sha_value,
            "incoming bank phase SHA-256",
        )
        tensor_name = f"state.{ordinal:08d}.{identity[0]:08d}.{phase_name}"
        if live_bank:
            try:
                tensor = states[identity]
            except KeyError as exc:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    "Incoming bank is missing a typed phase state."
                ) from exc
        else:
            try:
                tensor = snapshot_state(identity)
            except (KeyError, RuntimeError, ValueError) as exc:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    "Incoming failure snapshot is missing a typed phase state."
                ) from exc
        observed_identities.append(identity)
        tensors[tensor_name] = tensor
        ordered_phases.append(
            {
                "tensor_name": tensor_name,
                "identity": [identity[0], phase_name],
                "phase_sha256": phase_sha,
            }
        )
    if (
        len(tensors) != phase_count
        or len(set(observed_identities)) != phase_count
        or (
            live_bank
            and set(states) != {getattr(phase, "identity") for phase in phases}
        )
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Incoming bank states are missing, duplicated, or reordered."
        )
    metadata = build_lfm2_cpcf_incoming_state_bank_metadata(
        boundary=boundary,
        layer_index=layer_index,
        ordered_phases=ordered_phases,
        incoming_state_bank=tensors,
    )
    if metadata["state_bank_sha256"] != claimed_bank_sha:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Flattened incoming bank differs from its typed runtime seal."
        )
    return MappingProxyType(tensors), metadata


def build_lfm2_cpcf_completed_layers_document(
    *,
    bindings: Lfm2CpcfCheckpointBindings,
    completed_layer_receipts: Sequence[Mapping[str, Any]],
    installed_parameter_paths: Sequence[str],
    target_topology_snapshot: Mapping[str, Any],
    incoming_state_bank: Mapping[str, Any],
    incoming_state_bank_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the one exact checkpoint-progress document accepted by the store."""

    if not isinstance(bindings, Lfm2CpcfCheckpointBindings):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "bindings must be Lfm2CpcfCheckpointBindings."
        )
    if isinstance(installed_parameter_paths, (str, bytes)):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "installed_parameter_paths must be a sequence of names."
        )
    paths = sorted(
        {
            _safe_text(item, "installed parameter path")
            for item in installed_parameter_paths
        }
    )
    _incoming, entries = _normalize_tensors(
        incoming_state_bank,
        name="incoming state bank",
        clone=False,
    )
    topology = _target_topology(target_topology_snapshot)
    if topology["topology_sha256"] != bindings.target_topology_sha256:
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Target topology snapshot differs from checkpoint bindings."
        )
    _validate_installed_paths_against_topology(
        paths,
        topology=topology,
        next_layer_index=bindings.next_layer_index,
    )
    incoming_metadata = _incoming_state_bank_metadata(
        incoming_state_bank_metadata,
        next_layer_index=bindings.next_layer_index,
        incoming_entries=entries,
    )
    receipts = _completed_layer_receipts(
        completed_layer_receipts,
        next_layer_index=bindings.next_layer_index,
        incoming_state_bank_sha256=incoming_metadata["state_bank_sha256"],
    )
    receipts_sha = canonical_sha256(receipts)
    fold_seals_sha = canonical_sha256(
        [receipt["result_sha256"] for receipt in receipts]
    )
    install_sha = _install_receipt_sha256(
        completed_layer_receipts_sha256=receipts_sha,
        installed_parameter_paths=paths,
        target_topology_sha256=topology["topology_sha256"],
    )
    if (
        fold_seals_sha != bindings.completed_fold_seals_sha256
        or install_sha != bindings.install_receipt_sha256
    ):
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Completed receipts or installed paths differ from checkpoint bindings."
        )
    result = {
        "schema": COMPLETED_LAYERS_SCHEMA,
        "completed_layer_indices": list(range(bindings.next_layer_index)),
        "next_layer_index": bindings.next_layer_index,
        "completed_layer_receipts": receipts,
        "completed_layer_receipts_sha256": receipts_sha,
        "fold_seals_sha256": fold_seals_sha,
        "install_receipt_sha256": install_sha,
        "installed_parameter_paths": paths,
        "target_topology_snapshot": topology,
        "target_topology_sha256": topology["topology_sha256"],
        "incoming_state_layer_index": bindings.next_layer_index,
        "incoming_state_bank_metadata": incoming_metadata,
        "incoming_state_bank_metadata_sha256": canonical_sha256(incoming_metadata),
        "incoming_state_bank_sha256": canonical_sha256(entries),
        "incoming_state_entry_sha256": _entry_hashes(entries),
    }
    return _validate_completed_layers_document(
        result,
        bindings=bindings,
        prefix_entries=None,
        incoming_entries=entries,
    )


def _checkpoint_content_bindings(
    bindings: Lfm2CpcfCheckpointBindings,
    *,
    prefix_entries: list[dict[str, Any]],
    incoming_entries: list[dict[str, Any]],
    completed_layers: Mapping[str, Any],
) -> dict[str, Any]:
    completed = _validate_completed_layers_document(
        completed_layers,
        bindings=bindings,
        prefix_entries=prefix_entries,
        incoming_entries=incoming_entries,
    )
    prefix_hashes = {item["name"]: item["tensor_sha256"] for item in prefix_entries}
    incoming_hashes = {item["name"]: item["tensor_sha256"] for item in incoming_entries}
    return {
        **bindings.as_dict(),
        "prefix_model_state_sha256": canonical_sha256(prefix_entries),
        "prefix_model_entry_sha256": prefix_hashes,
        "incoming_state_bank_sha256": canonical_sha256(incoming_entries),
        "incoming_state_entry_sha256": incoming_hashes,
        "completed_layers_document_sha256": canonical_sha256(completed),
    }


def _safe_stage_cleanup(
    stage: Path,
    *,
    collection: Path,
    identity: tuple[int, int],
) -> None:
    if not _lexists(stage):
        return
    try:
        current = stage.lstat()
        resolved_collection = collection.resolve(strict=True)
        resolved_stage = stage.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactPublicationError(
            "Could not authenticate owned staging for cleanup."
        ) from exc
    if (
        (int(current.st_dev), int(current.st_ino)) != identity
        or not stat.S_ISDIR(current.st_mode)
        or _is_link_or_junction(stage)
        or resolved_stage.parent != resolved_collection
        or not stage.name.startswith(_STAGE_PREFIX)
        or not stage.name.endswith(_STAGE_SUFFIX)
    ):
        raise Lfm2CpcfResumeArtifactPublicationError(
            "Refusing to remove unauthenticated CPCF staging."
        )
    allowed = (
        _BOUNDARY_FILES
        | _CHECKPOINT_FILES
        | _FAILURE_FILES
        | frozenset({_STAGE_OWNER_FILE})
    )
    try:
        entries = list(os.scandir(stage))
        for entry in entries:
            if entry.name not in allowed:
                raise Lfm2CpcfResumeArtifactPublicationError(
                    "Refusing to remove staging with foreign content."
                )
            _regular_file(stage / entry.name, name=entry.name)
        for entry in entries:
            os.unlink(stage / entry.name)
        after = stage.lstat()
        if _stat_identity(after) != identity or not stat.S_ISDIR(after.st_mode):
            raise Lfm2CpcfResumeArtifactPublicationError(
                "Staging changed while its files were removed."
            )
        os.rmdir(stage)
        _fsync_directory(collection)
    except Lfm2CpcfResumeArtifactPublicationError:
        raise
    except OSError as exc:
        raise Lfm2CpcfResumeArtifactPublicationError(
            "Could not remove authenticated CPCF staging."
        ) from exc


def _process_identity(
    pid: int,
) -> tuple[Literal["live", "dead", "unknown"], str | None]:
    """Return a creation identity when the platform can prove one."""

    if pid <= 0:
        return "dead", None
    if os.name == "nt":
        from ctypes import wintypes

        if pid > 0xFFFFFFFF:
            return "dead", None

        class _FileTime(ctypes.Structure):
            _fields_ = (
                ("low", wintypes.DWORD),
                ("high", wintypes.DWORD),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            error_number = ctypes.get_last_error()
            if error_number in {87, 1168}:
                return "dead", None
            return "unknown", None
        try:
            creation = _FileTime()
            exit_time = _FileTime()
            kernel = _FileTime()
            user = _FileTime()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return "unknown", None
            creation_ticks = (int(creation.high) << 32) | int(creation.low)
            return "live", f"windows-filetime:{creation_ticks}"
        finally:
            kernel32.CloseHandle(handle)
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.parent.parent == Path("/proc") and Path("/proc").is_dir():
        try:
            raw = proc_stat.read_text(encoding="ascii")
        except FileNotFoundError:
            return "dead", None
        except OSError:
            return "unknown", None
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split() if closing >= 0 else []
        if len(fields) <= 19:
            return "unknown", None
        return "live", f"linux-start-ticks:{fields[19]}"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead", None
    except (PermissionError, OSError):
        return "unknown", None
    return "live", None


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""

    if os.name == "posix":
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
        except (AttributeError, OSError):
            renameat2 = None
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(error_number, os.strerror(error_number))
            raise OSError(error_number, os.strerror(error_number))
        raise Lfm2CpcfResumeArtifactPublicationError(
            "This POSIX runtime lacks atomic no-replace directory rename."
        )
    # Windows directory rename is atomic and refuses an existing destination.
    os.rename(source, destination)


class Lfm2CpcfResumeArtifactStore:
    """Content-addressed immutable store for CPCF continuation evidence."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        candidate = Path(root).expanduser().absolute()
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise Lfm2CpcfResumeArtifactPublicationError(
                f"Could not create resume store root {candidate}."
            ) from exc
        if _is_link_or_junction(candidate) or not candidate.is_dir():
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Resume store root must be a real directory."
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Could not resolve resume store root."
            ) from exc
        self.root = resolved
        self._root_identity = _stat_identity(
            _real_directory(self.root, name="resume store root")
        )
        self._stage_key = self._load_or_create_stage_key()

    def _authenticate_root(self) -> None:
        current = _real_directory(self.root, name="resume store root")
        if _stat_identity(current) != self._root_identity:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Resume store root changed identity."
            )

    def _load_or_create_stage_key(self) -> bytes:
        key_path = self.root / _STAGE_KEY_FILE
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        created = False
        try:
            descriptor = os.open(key_path, flags, 0o600)
            created = True
        except FileExistsError:
            descriptor = -1
        except OSError as exc:
            raise Lfm2CpcfResumeArtifactPublicationError(
                "Could not create the staging ownership key."
            ) from exc
        if created:
            payload = secrets.token_bytes(_STAGE_KEY_BYTES)
            try:
                offset = 0
                while offset < len(payload):
                    offset += os.write(descriptor, payload[offset:])
                os.fsync(descriptor)
            except OSError as exc:
                raise Lfm2CpcfResumeArtifactPublicationError(
                    "Could not persist the staging ownership key."
                ) from exc
            finally:
                os.close(descriptor)
            if os.name == "posix":
                try:
                    os.chmod(key_path, 0o600)
                except OSError as exc:
                    raise Lfm2CpcfResumeArtifactPublicationError(
                        "Could not restrict the staging ownership key."
                    ) from exc
            _fsync_directory(self.root)
            return payload
        last_error: BaseException | None = None
        for _attempt in range(25):
            try:
                payload = _read_regular_bytes(key_path, name=_STAGE_KEY_FILE)
                if len(payload) != _STAGE_KEY_BYTES:
                    raise Lfm2CpcfResumeArtifactIntegrityError(
                        "Staging ownership key has the wrong size."
                    )
                return payload
            except Lfm2CpcfResumeArtifactIntegrityError as exc:
                last_error = exc
                time.sleep(0.01)
        raise Lfm2CpcfResumeArtifactIntegrityError(
            "Staging ownership key is invalid or incomplete."
        ) from last_error

    def _collection(
        self,
        kind: Literal["boundaries", "checkpoints", "failures"],
    ) -> Path:
        self._authenticate_root()
        path = self.root / kind
        existed = _lexists(path)
        try:
            path.mkdir(exist_ok=True)
        except OSError as exc:
            raise Lfm2CpcfResumeArtifactPublicationError(
                f"Could not create {kind} collection."
            ) from exc
        if (
            _is_link_or_junction(path)
            or not path.is_dir()
            or path.resolve(strict=True).parent != self.root
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(f"{kind} collection is unsafe.")
        if not existed:
            _fsync_directory(self.root)
        return path.resolve(strict=True)

    def _stage_owner_mac(self, body: Mapping[str, Any]) -> str:
        return hmac.new(
            self._stage_key,
            canonical_json_bytes(_deep_thaw(body)),
            hashlib.sha256,
        ).hexdigest()

    def _stage_owner_document(
        self,
        *,
        collection: Path,
        stage: Path,
        owner_token: str,
    ) -> dict[str, Any]:
        status, process_identity = _process_identity(os.getpid())
        if status != "live":
            process_identity = None
        body = {
            "schema": STAGE_OWNER_SCHEMA,
            "collection": collection.name,
            "stage_name": stage.name,
            "owner_token": _sha256(owner_token, "stage owner token"),
            "pid": os.getpid(),
            "process_identity": process_identity,
            "hostname": socket.gethostname(),
            "created_unix_ns": time.time_ns(),
        }
        return {**body, "hmac_sha256": self._stage_owner_mac(body)}

    def _verify_stage_owner(
        self,
        stage: Path,
        *,
        collection: Path,
    ) -> dict[str, Any]:
        document = _read_canonical_json(
            stage / _STAGE_OWNER_FILE, name=_STAGE_OWNER_FILE
        )
        expected_fields = {
            "schema",
            "collection",
            "stage_name",
            "owner_token",
            "pid",
            "process_identity",
            "hostname",
            "created_unix_ns",
            "hmac_sha256",
        }
        if set(document) != expected_fields:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Staging owner marker has an unexpected field set."
            )
        body = dict(document)
        claimed = _sha256(body.pop("hmac_sha256", None), "stage owner HMAC")
        process_identity = body.get("process_identity")
        if (
            body.get("schema") != STAGE_OWNER_SCHEMA
            or body.get("collection") != collection.name
            or body.get("stage_name") != stage.name
            or not isinstance(body.get("hostname"), str)
            or not body["hostname"]
            or (
                process_identity is not None
                and (not isinstance(process_identity, str) or not process_identity)
            )
            or _nonnegative_int(body.get("pid"), "stage owner pid") == 0
            or _nonnegative_int(body.get("created_unix_ns"), "stage creation time") == 0
            or not hmac.compare_digest(self._stage_owner_mac(body), claimed)
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Staging owner marker is invalid."
            )
        _sha256(body.get("owner_token"), "stage owner token")
        return document

    def _stage(
        self,
        collection: Path,
    ) -> tuple[Path, tuple[int, int], str]:
        try:
            stage = Path(
                tempfile.mkdtemp(
                    prefix=_STAGE_PREFIX,
                    suffix=_STAGE_SUFFIX,
                    dir=collection,
                )
            ).absolute()
            stage_stat = stage.lstat()
        except OSError as exc:
            raise Lfm2CpcfResumeArtifactPublicationError(
                "Could not exclusively create CPCF staging."
            ) from exc
        owner_token = secrets.token_hex(32)
        identity = _stat_identity(stage_stat)
        try:
            _write_canonical_json(
                stage / _STAGE_OWNER_FILE,
                self._stage_owner_document(
                    collection=collection,
                    stage=stage,
                    owner_token=owner_token,
                ),
            )
            _fsync_directory(stage)
        except BaseException:
            _safe_stage_cleanup(stage, collection=collection, identity=identity)
            raise
        return stage, identity, owner_token

    def _remove_stage_owner(
        self,
        stage: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
        owner_token: str,
    ) -> None:
        current = _real_directory(stage, name="owned staging")
        owner = self._verify_stage_owner(stage, collection=collection)
        if _stat_identity(current) != identity or owner["owner_token"] != owner_token:
            raise Lfm2CpcfResumeArtifactPublicationError(
                "Owned staging changed before publication."
            )
        try:
            os.unlink(stage / _STAGE_OWNER_FILE)
            _fsync_directory(stage)
        except OSError as exc:
            raise Lfm2CpcfResumeArtifactPublicationError(
                "Could not clear the staging owner marker."
            ) from exc

    def cleanup_stale_stages(
        self,
        *,
        minimum_age_seconds: float,
        now_unix_ns: int | None = None,
        maximum_reported_entries: int = 4096,
    ) -> Mapping[str, Any]:
        """Remove only signed, expired stages whose creating process is dead."""

        if (
            isinstance(minimum_age_seconds, bool)
            or not isinstance(minimum_age_seconds, (int, float))
            or not math.isfinite(float(minimum_age_seconds))
            or minimum_age_seconds < 0
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "minimum_age_seconds must be non-negative."
            )
        if (
            isinstance(maximum_reported_entries, bool)
            or not isinstance(maximum_reported_entries, int)
            or maximum_reported_entries <= 0
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "maximum_reported_entries must be a positive integer."
            )
        now = (
            time.time_ns()
            if now_unix_ns is None
            else _nonnegative_int(now_unix_ns, "now_unix_ns")
        )
        removed: list[str] = []
        preserved: list[str] = []
        removed_count = 0
        preserved_count = 0
        report_truncated = False

        def record(label: str, *, was_removed: bool) -> None:
            nonlocal removed_count, preserved_count, report_truncated
            if was_removed:
                removed_count += 1
                destination = removed
            else:
                preserved_count += 1
                destination = preserved
            if len(removed) + len(preserved) < maximum_reported_entries:
                destination.append(label)
            else:
                report_truncated = True

        minimum_age_ns = int(float(minimum_age_seconds) * 1_000_000_000)
        for kind in ("boundaries", "checkpoints", "failures"):
            collection = self._collection(kind)
            with os.scandir(collection) as entries:
                for entry in entries:
                    if not (
                        entry.name.startswith(_STAGE_PREFIX)
                        and entry.name.endswith(_STAGE_SUFFIX)
                    ):
                        continue
                    stage = collection / entry.name
                    label = f"{kind}/{entry.name}"
                    try:
                        stage_stat = _real_directory(stage, name=label)
                        owner = self._verify_stage_owner(stage, collection=collection)
                        created = _nonnegative_int(
                            owner["created_unix_ns"], "stage creation time"
                        )
                        if (
                            now < created
                            or now - created < minimum_age_ns
                            or owner["hostname"] != socket.gethostname()
                        ):
                            record(label, was_removed=False)
                            continue
                        status, current_identity = _process_identity(owner["pid"])
                        owner_identity = owner["process_identity"]
                        if status == "unknown" or (
                            status == "live"
                            and (
                                owner_identity is None
                                or current_identity is None
                                or current_identity == owner_identity
                            )
                        ):
                            record(label, was_removed=False)
                            continue
                        _safe_stage_cleanup(
                            stage,
                            collection=collection,
                            identity=_stat_identity(stage_stat),
                        )
                        record(label, was_removed=True)
                    except Lfm2CpcfResumeArtifactError:
                        record(label, was_removed=False)
        return MappingProxyType(
            {
                "removed": tuple(removed),
                "preserved": tuple(preserved),
                "removed_count": removed_count,
                "preserved_count": preserved_count,
                "report_truncated": report_truncated,
            }
        )

    def _artifact_path(
        self,
        collection: Literal["boundaries", "checkpoints", "failures"],
        artifact_key: str,
    ) -> Path:
        key = _sha256(artifact_key, "artifact_key")
        return self._collection(collection) / key

    def publish_boundary(
        self,
        *,
        bindings: Lfm2CpcfBoundaryBindings,
        reports: Mapping[str, Any],
        tensors: Mapping[str, Any],
    ) -> VerifiedLfm2CpcfBoundaryBundle:
        """Publish or reuse one algorithm-independent boundary bundle."""

        if not isinstance(bindings, Lfm2CpcfBoundaryBindings):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "bindings must be Lfm2CpcfBoundaryBindings."
            )
        normalized_reports = _canonical_mapping(reports, "boundary reports")
        normalized_tensors, entries = _normalize_tensors(
            tensors,
            name="boundary tensors",
            clone=False,
        )
        collection = self._collection("boundaries")
        stage, identity, owner_token = self._stage(collection)
        try:
            reports_path = stage / "reports.json"
            tensors_path = stage / "tensors.safetensors"
            _write_canonical_json(reports_path, normalized_reports)
            _save_safetensors(
                tensors_path,
                normalized_tensors,
                artifact_kind="boundary",
                file_role="boundary-tensors",
            )
            files = [
                _file_record(reports_path),
                _file_record(tensors_path, tensor_entries=entries),
            ]
            files.sort(key=lambda item: item["path"])
            key = _boundary_content_key(bindings=bindings, files=files)
            destination = collection / key
            manifest = _seal_manifest(
                _manifest_body(
                    kind="boundary",
                    artifact_key=key,
                    key_schema=BOUNDARY_KEY_SCHEMA,
                    bindings=bindings.as_dict(),
                    files=files,
                    binding_key=bindings.binding_key,
                )
            )
            _write_canonical_json(stage / "manifest.json", manifest)
            _fsync_directory(stage)
            self._remove_stage_owner(
                stage,
                collection=collection,
                identity=identity,
                owner_token=owner_token,
            )
            staged = self._open_boundary_path(stage, key)
            return self._publish_boundary_stage(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        except BaseException:
            if _lexists(stage):
                _safe_stage_cleanup(stage, collection=collection, identity=identity)
            raise

    def open_boundary(
        self,
        reference: Lfm2CpcfArtifactReference,
        *,
        expected_bindings: Lfm2CpcfBoundaryBindings | None = None,
    ) -> VerifiedLfm2CpcfBoundaryBundle:
        if (
            not isinstance(reference, Lfm2CpcfArtifactReference)
            or reference.artifact_kind != "boundary"
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "open_boundary requires a pinned boundary reference."
            )
        key = reference.artifact_key
        if expected_bindings is not None and not isinstance(
            expected_bindings, Lfm2CpcfBoundaryBindings
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "expected_bindings must be Lfm2CpcfBoundaryBindings."
            )
        result = self._open_boundary_path(self._artifact_path("boundaries", key), key)
        if result.reference != reference:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Boundary differs from the pinned artifact reference."
            )
        if expected_bindings is not None and (
            result.binding_key != expected_bindings.binding_key
            or _deep_thaw(result.manifest["bindings"]) != expected_bindings.as_dict()
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Boundary manifest differs from expected bindings."
            )
        return result

    def discover_boundary_references(
        self,
        binding_key: str,
        *,
        expected_bindings: Lfm2CpcfBoundaryBindings | None = None,
    ) -> tuple[Lfm2CpcfArtifactReference, ...]:
        """Return every authenticated content candidate for one discovery key."""

        key = _sha256(binding_key, "boundary binding_key")
        if expected_bindings is not None and (
            not isinstance(expected_bindings, Lfm2CpcfBoundaryBindings)
            or expected_bindings.binding_key != key
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Discovery key differs from expected boundary bindings."
            )
        collection = self._collection("boundaries")
        references = []
        for entry in sorted(os.scandir(collection), key=lambda item: item.name):
            if _SHA256.fullmatch(entry.name) is None:
                continue
            candidate = self._open_boundary_path(collection / entry.name, entry.name)
            if candidate.binding_key == key:
                if (
                    expected_bindings is not None
                    and _deep_thaw(candidate.manifest["bindings"])
                    != expected_bindings.as_dict()
                ):
                    raise Lfm2CpcfResumeArtifactIntegrityError(
                        "Discovered boundary has colliding binding identity."
                    )
                references.append(candidate.reference)
        return tuple(references)

    def _open_boundary_path(
        self,
        path: Path,
        key: str,
    ) -> VerifiedLfm2CpcfBoundaryBundle:
        root = _artifact_directory(path)
        manifest, records, manifest_raw, payloads = _authenticate_manifest(
            root, artifact_key=key, kind="boundary"
        )
        bindings = _boundary_bindings_from_mapping(manifest.get("bindings"))
        if (
            manifest.get("key_schema") != BOUNDARY_KEY_SCHEMA
            or manifest.get("binding_key_schema") != BOUNDARY_BINDING_KEY_SCHEMA
            or manifest.get("binding_key") != bindings.binding_key
            or bindings.as_dict() != manifest.get("bindings")
            or _boundary_content_key(
                bindings=bindings,
                files=manifest.get("files"),
            )
            != key
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Boundary content key does not authenticate its bindings."
            )
        report_record = records["reports.json"]
        tensor_record = records["tensors.safetensors"]
        if (
            report_record.get("format") != "canonical-json"
            or tensor_record.get("format") != "safetensors"
            or tensor_record.get("tensor_set_sha256")
            != canonical_sha256(tensor_record.get("tensor_entries"))
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Boundary file formats or tensor seal are invalid."
            )
        reports = _canonical_json_from_bytes(
            payloads["reports.json"], name="reports.json"
        )
        _load_safetensors_path(
            root / "tensors.safetensors",
            name="tensors.safetensors",
            artifact_kind="boundary",
            file_role="boundary-tensors",
            record=tensor_record,
            clone_tensors=False,
        )
        (
            after_manifest,
            after_records,
            after_raw,
            after_payloads,
        ) = _authenticate_manifest(root, artifact_key=key, kind="boundary")
        if (
            canonical_json_bytes(after_manifest) != canonical_json_bytes(manifest)
            or after_records != records
            or after_raw != manifest_raw
            or after_payloads != payloads
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Boundary artifact changed while it was opened."
            )
        return VerifiedLfm2CpcfBoundaryBundle(
            path=root,
            artifact_key=key,
            manifest=_deep_freeze(manifest),
            reports=_deep_freeze(reports),
            _tensor_archive=_DisposableTensorArchive.from_record(
                root / "tensors.safetensors",
                artifact_kind="boundary",
                file_role="boundary-tensors",
                record=tensor_record,
            ),
            certificate_sha256=_certificate(manifest_raw, records),
        )

    def _publish_boundary_stage(
        self,
        stage: Path,
        destination: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
        staged: VerifiedLfm2CpcfBoundaryBundle,
    ) -> VerifiedLfm2CpcfBoundaryBundle:
        if _lexists(destination):
            return self._reuse_boundary(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        try:
            _atomic_rename_noreplace(stage, destination)
        except (FileExistsError, OSError) as exc:
            if not _lexists(destination):
                raise Lfm2CpcfResumeArtifactPublicationError(
                    "Atomic boundary publication failed."
                ) from exc
            return self._reuse_boundary(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        _fsync_directory(collection)
        result = self._open_boundary_path(destination, staged.artifact_key)
        if result.certificate_sha256 != staged.certificate_sha256:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Published boundary differs from authenticated staging."
            )
        return result

    def _reuse_boundary(
        self,
        stage: Path,
        destination: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
        staged: VerifiedLfm2CpcfBoundaryBundle,
    ) -> VerifiedLfm2CpcfBoundaryBundle:
        try:
            existing = self._open_boundary_path(destination, staged.artifact_key)
        except Lfm2CpcfResumeArtifactError as exc:
            raise Lfm2CpcfResumeArtifactConflictError(
                "Conflicting boundary path was left untouched."
            ) from exc
        if existing.certificate_sha256 != staged.certificate_sha256:
            raise Lfm2CpcfResumeArtifactConflictError(
                "Different boundary content already occupies this key."
            )
        _safe_stage_cleanup(stage, collection=collection, identity=identity)
        return replace(existing, reused_existing=True)

    def publish_checkpoint(
        self,
        *,
        bindings: Lfm2CpcfCheckpointBindings,
        prefix_model: Mapping[str, Any],
        incoming_state_bank: Mapping[str, Any],
        completed_layers: Mapping[str, Any],
    ) -> VerifiedLfm2CpcfPrefixCheckpoint:
        """Publish or reuse a fully sealed prefix-model continuation point."""

        if not isinstance(bindings, Lfm2CpcfCheckpointBindings):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "bindings must be Lfm2CpcfCheckpointBindings."
            )
        self.open_boundary(bindings.boundary)
        prefix, prefix_entries = _normalize_tensors(
            prefix_model,
            name="prefix model",
            clone=False,
        )
        incoming, incoming_entries = _normalize_tensors(
            incoming_state_bank,
            name="incoming state bank",
            clone=False,
        )
        completed = _validate_completed_layers_document(
            completed_layers,
            bindings=bindings,
            prefix_entries=prefix_entries,
            incoming_entries=incoming_entries,
        )
        content_bindings = _checkpoint_content_bindings(
            bindings,
            prefix_entries=prefix_entries,
            incoming_entries=incoming_entries,
            completed_layers=completed,
        )
        collection = self._collection("checkpoints")
        stage, identity, owner_token = self._stage(collection)
        try:
            prefix_path = stage / "prefix-model.safetensors"
            incoming_path = stage / "incoming-state-bank.safetensors"
            completed_path = stage / "completed-layers.json"
            _save_safetensors(
                prefix_path,
                prefix,
                artifact_kind="checkpoint",
                file_role="prefix-model",
            )
            _save_safetensors(
                incoming_path,
                incoming,
                artifact_kind="checkpoint",
                file_role="incoming-state-bank",
            )
            _write_canonical_json(completed_path, completed)
            files = [
                _file_record(prefix_path, tensor_entries=prefix_entries),
                _file_record(incoming_path, tensor_entries=incoming_entries),
                _file_record(completed_path),
            ]
            files.sort(key=lambda item: item["path"])
            key = _checkpoint_content_key(
                bindings=content_bindings,
                files=files,
            )
            destination = collection / key
            manifest = _seal_manifest(
                _manifest_body(
                    kind="checkpoint",
                    artifact_key=key,
                    key_schema=CHECKPOINT_KEY_SCHEMA,
                    bindings=content_bindings,
                    files=files,
                )
            )
            _write_canonical_json(stage / "manifest.json", manifest)
            _fsync_directory(stage)
            self._remove_stage_owner(
                stage,
                collection=collection,
                identity=identity,
                owner_token=owner_token,
            )
            staged = self._open_checkpoint_path(stage, key)
            return self._publish_checkpoint_stage(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        except BaseException:
            if _lexists(stage):
                _safe_stage_cleanup(stage, collection=collection, identity=identity)
            raise

    def open_checkpoint(
        self,
        reference: Lfm2CpcfArtifactReference,
    ) -> VerifiedLfm2CpcfPrefixCheckpoint:
        if (
            not isinstance(reference, Lfm2CpcfArtifactReference)
            or reference.artifact_kind != "checkpoint"
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "open_checkpoint requires a pinned checkpoint reference."
            )
        result = self._open_checkpoint_path(
            self._artifact_path("checkpoints", reference.artifact_key),
            reference.artifact_key,
        )
        if result.reference != reference:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint differs from the pinned artifact reference."
            )
        return result

    def _open_checkpoint_path(
        self,
        path: Path,
        key: str,
    ) -> VerifiedLfm2CpcfPrefixCheckpoint:
        root = _artifact_directory(path)
        manifest, records, manifest_raw, payloads = _authenticate_manifest(
            root, artifact_key=key, kind="checkpoint"
        )
        if (
            manifest.get("key_schema") != CHECKPOINT_KEY_SCHEMA
            or _checkpoint_content_key(
                bindings=manifest.get("bindings"),
                files=manifest.get("files"),
            )
            != key
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint content key does not authenticate its bindings."
            )
        prefix_record = records["prefix-model.safetensors"]
        incoming_record = records["incoming-state-bank.safetensors"]
        completed_record = records["completed-layers.json"]
        if (
            prefix_record.get("format") != "safetensors"
            or incoming_record.get("format") != "safetensors"
            or completed_record.get("format") != "canonical-json"
            or prefix_record.get("tensor_set_sha256")
            != canonical_sha256(prefix_record.get("tensor_entries"))
            or incoming_record.get("tensor_set_sha256")
            != canonical_sha256(incoming_record.get("tensor_entries"))
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint file formats or tensor seals are invalid."
            )
        _load_safetensors_path(
            root / "prefix-model.safetensors",
            name="prefix-model.safetensors",
            artifact_kind="checkpoint",
            file_role="prefix-model",
            record=prefix_record,
            clone_tensors=False,
        )
        _load_safetensors_path(
            root / "incoming-state-bank.safetensors",
            name="incoming-state-bank.safetensors",
            artifact_kind="checkpoint",
            file_role="incoming-state-bank",
            record=incoming_record,
            clone_tensors=False,
        )
        completed = _canonical_json_from_bytes(
            payloads["completed-layers.json"], name="completed-layers.json"
        )
        recomputed = {
            "prefix_model_state_sha256": canonical_sha256(
                prefix_record["tensor_entries"]
            ),
            "prefix_model_entry_sha256": {
                item["name"]: item["tensor_sha256"]
                for item in prefix_record["tensor_entries"]
            },
            "incoming_state_bank_sha256": canonical_sha256(
                incoming_record["tensor_entries"]
            ),
            "incoming_state_entry_sha256": {
                item["name"]: item["tensor_sha256"]
                for item in incoming_record["tensor_entries"]
            },
            "completed_layers_document_sha256": canonical_sha256(completed),
        }
        bindings = manifest.get("bindings")
        external, authenticated_derived = _checkpoint_bindings_from_mapping(bindings)
        if authenticated_derived != recomputed or any(
            bindings.get(name) != value for name, value in recomputed.items()
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint bindings do not match checkpoint content."
            )
        _validate_completed_layers_document(
            completed,
            bindings=external,
            prefix_entries=prefix_record["tensor_entries"],
            incoming_entries=incoming_record["tensor_entries"],
        )
        self.open_boundary(external.boundary)
        (
            after_manifest,
            after_records,
            after_raw,
            after_payloads,
        ) = _authenticate_manifest(root, artifact_key=key, kind="checkpoint")
        if (
            canonical_json_bytes(after_manifest) != canonical_json_bytes(manifest)
            or after_records != records
            or after_raw != manifest_raw
            or after_payloads != payloads
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint artifact changed while it was opened."
            )
        return VerifiedLfm2CpcfPrefixCheckpoint(
            path=root,
            artifact_key=key,
            manifest=_deep_freeze(manifest),
            _prefix_model_archive=_DisposableTensorArchive.from_record(
                root / "prefix-model.safetensors",
                artifact_kind="checkpoint",
                file_role="prefix-model",
                record=prefix_record,
            ),
            _incoming_state_bank_archive=_DisposableTensorArchive.from_record(
                root / "incoming-state-bank.safetensors",
                artifact_kind="checkpoint",
                file_role="incoming-state-bank",
                record=incoming_record,
            ),
            completed_layers=_deep_freeze(completed),
            certificate_sha256=_certificate(manifest_raw, records),
        )

    def _publish_checkpoint_stage(
        self,
        stage: Path,
        destination: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
        staged: VerifiedLfm2CpcfPrefixCheckpoint,
    ) -> VerifiedLfm2CpcfPrefixCheckpoint:
        if _lexists(destination):
            return self._reuse_checkpoint(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        try:
            _atomic_rename_noreplace(stage, destination)
        except (FileExistsError, OSError) as exc:
            if not _lexists(destination):
                raise Lfm2CpcfResumeArtifactPublicationError(
                    "Atomic checkpoint publication failed."
                ) from exc
            return self._reuse_checkpoint(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        _fsync_directory(collection)
        result = self._open_checkpoint_path(destination, staged.artifact_key)
        if result.certificate_sha256 != staged.certificate_sha256:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Published checkpoint differs from authenticated staging."
            )
        return result

    def _reuse_checkpoint(
        self,
        stage: Path,
        destination: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
        staged: VerifiedLfm2CpcfPrefixCheckpoint,
    ) -> VerifiedLfm2CpcfPrefixCheckpoint:
        try:
            existing = self._open_checkpoint_path(destination, staged.artifact_key)
        except Lfm2CpcfResumeArtifactError as exc:
            raise Lfm2CpcfResumeArtifactConflictError(
                "Conflicting checkpoint path was left untouched."
            ) from exc
        if existing.certificate_sha256 != staged.certificate_sha256:
            raise Lfm2CpcfResumeArtifactConflictError(
                "Different checkpoint content already occupies this key."
            )
        _safe_stage_cleanup(stage, collection=collection, identity=identity)
        return replace(existing, reused_existing=True)

    def _authenticate_failure_lineage(
        self,
        bindings: Lfm2CpcfFailureBindings,
    ) -> None:
        boundary = (
            None if bindings.boundary is None else self.open_boundary(bindings.boundary)
        )
        checkpoint = (
            None
            if bindings.checkpoint is None
            else self.open_checkpoint(bindings.checkpoint)
        )
        if boundary is not None and checkpoint is not None:
            checkpoint_boundary = _reference_from_mapping(
                _deep_thaw(checkpoint.manifest["bindings"]["boundary"]),
                name="checkpoint boundary",
            )
            if checkpoint_boundary != boundary.reference:
                raise Lfm2CpcfResumeArtifactIntegrityError(
                    "Failure boundary and checkpoint lineage disagree."
                )

    def publish_failure(
        self,
        *,
        bindings: Lfm2CpcfFailureBindings,
        report: Mapping[str, Any],
    ) -> VerifiedLfm2CpcfFailureDiagnostic:
        """Publish a sealed failure without making a success claim."""

        if not isinstance(bindings, Lfm2CpcfFailureBindings):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "bindings must be Lfm2CpcfFailureBindings."
            )
        self._authenticate_failure_lineage(bindings)
        normalized_report = _validated_known_failure_report(
            report,
            bindings=bindings,
        )
        key_bindings = {
            **bindings.as_dict(),
            "report_sha256": canonical_sha256(normalized_report),
        }
        key = canonical_sha256({"schema": FAILURE_KEY_SCHEMA, "bindings": key_bindings})
        body = {
            "schema": RESUME_ARTIFACT_SCHEMA,
            "artifact_kind": "failure",
            "status": _STATUS,
            "artifact_key": key,
            "key_schema": FAILURE_KEY_SCHEMA,
            "bindings": key_bindings,
            "report": normalized_report,
            "promotion_authorized": False,
            "accepted_artifact": False,
            "compaction_success_proven": False,
            "behavior_success_proven": False,
            "success_claimed": False,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        diagnostic = {
            **body,
            "diagnostic_sha256": canonical_sha256(body),
        }
        collection = self._collection("failures")
        destination = collection / key
        stage, identity, owner_token = self._stage(collection)
        try:
            _write_canonical_json(stage / "diagnostic.json", diagnostic)
            _fsync_directory(stage)
            self._remove_stage_owner(
                stage,
                collection=collection,
                identity=identity,
                owner_token=owner_token,
            )
            staged = self._open_failure_path(stage, key)
            return self._publish_failure_stage(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        except BaseException:
            if _lexists(stage):
                _safe_stage_cleanup(stage, collection=collection, identity=identity)
            raise

    def open_failure(
        self,
        artifact_key: str,
    ) -> VerifiedLfm2CpcfFailureDiagnostic:
        key = _sha256(artifact_key, "artifact_key")
        return self._open_failure_path(self._artifact_path("failures", key), key)

    def _open_failure_path(
        self,
        path: Path,
        key: str,
    ) -> VerifiedLfm2CpcfFailureDiagnostic:
        root = _artifact_directory(path)
        directory_identity = _verify_inventory(root, _FAILURE_FILES)
        diagnostic_path = root / "diagnostic.json"
        diagnostic, raw = _read_canonical_json_with_raw(
            diagnostic_path, name="diagnostic.json"
        )
        if (
            set(diagnostic) != _FAILURE_DIAGNOSTIC_FIELDS
            or diagnostic.get("schema") != RESUME_ARTIFACT_SCHEMA
            or diagnostic.get("artifact_kind") != "failure"
            or diagnostic.get("artifact_key") != key
            or diagnostic.get("key_schema") != FAILURE_KEY_SCHEMA
            or diagnostic.get("status") != _STATUS
            or diagnostic.get("promotion_authorized") is not False
            or diagnostic.get("accepted_artifact") is not False
            or diagnostic.get("compaction_success_proven") is not False
            or diagnostic.get("behavior_success_proven") is not False
            or diagnostic.get("success_claimed") is not False
            or diagnostic.get("training_performed") is not False
            or type(diagnostic.get("optimizer_steps")) is not int
            or diagnostic.get("optimizer_steps") != 0
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Failure diagnostic schema or fixed no-success flags are invalid."
            )
        body = dict(diagnostic)
        claimed = body.pop("diagnostic_sha256", None)
        if canonical_sha256(body) != claimed:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Failure diagnostic self-hash is invalid."
            )
        bindings = diagnostic.get("bindings")
        report = diagnostic.get("report")
        authenticated_bindings, report_sha256 = _failure_bindings_from_mapping(bindings)
        if (
            not isinstance(report, dict)
            or report_sha256 != canonical_sha256(report)
            or {
                **authenticated_bindings.as_dict(),
                "report_sha256": report_sha256,
            }
            != bindings
            or canonical_sha256({"schema": FAILURE_KEY_SCHEMA, "bindings": bindings})
            != key
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Failure content key or nested report seal is invalid."
            )
        _validated_known_failure_report(
            report,
            bindings=authenticated_bindings,
        )
        self._authenticate_failure_lineage(authenticated_bindings)
        after, after_raw = _read_canonical_json_with_raw(
            diagnostic_path, name="diagnostic.json"
        )
        if (
            after != diagnostic
            or after_raw != raw
            or _verify_inventory(root, _FAILURE_FILES) != directory_identity
        ):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Failure diagnostic changed while it was opened."
            )
        return VerifiedLfm2CpcfFailureDiagnostic(
            path=root,
            artifact_key=key,
            diagnostic=_deep_freeze(diagnostic),
            certificate_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def _publish_failure_stage(
        self,
        stage: Path,
        destination: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
        staged: VerifiedLfm2CpcfFailureDiagnostic,
    ) -> VerifiedLfm2CpcfFailureDiagnostic:
        if _lexists(destination):
            return self._reuse_failure(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        try:
            _atomic_rename_noreplace(stage, destination)
        except (FileExistsError, OSError) as exc:
            if not _lexists(destination):
                raise Lfm2CpcfResumeArtifactPublicationError(
                    "Atomic failure publication failed."
                ) from exc
            return self._reuse_failure(
                stage,
                destination,
                collection=collection,
                identity=identity,
                staged=staged,
            )
        _fsync_directory(collection)
        result = self._open_failure_path(destination, staged.artifact_key)
        if result.certificate_sha256 != staged.certificate_sha256:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Published failure differs from authenticated staging."
            )
        return result

    def _reuse_failure(
        self,
        stage: Path,
        destination: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
        staged: VerifiedLfm2CpcfFailureDiagnostic,
    ) -> VerifiedLfm2CpcfFailureDiagnostic:
        try:
            existing = self._open_failure_path(destination, staged.artifact_key)
        except Lfm2CpcfResumeArtifactError as exc:
            raise Lfm2CpcfResumeArtifactConflictError(
                "Conflicting failure path was left untouched."
            ) from exc
        if existing.certificate_sha256 != staged.certificate_sha256:
            raise Lfm2CpcfResumeArtifactConflictError(
                "Different failure content already occupies this key."
            )
        _safe_stage_cleanup(stage, collection=collection, identity=identity)
        return replace(existing, reused_existing=True)

    def final_reauthenticate_boundary(
        self,
        verified: VerifiedLfm2CpcfBoundaryBundle,
    ) -> VerifiedLfm2CpcfBoundaryBundle:
        """Reopen a boundary immediately before consuming disposable clones."""

        if not isinstance(verified, VerifiedLfm2CpcfBoundaryBundle):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "verified must be a verified boundary bundle."
            )
        result = self.open_boundary(verified.reference)
        if result.certificate_sha256 != verified.certificate_sha256:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Boundary certificate changed before final use."
            )
        return result

    def final_reauthenticate_checkpoint(
        self,
        verified: VerifiedLfm2CpcfPrefixCheckpoint,
    ) -> VerifiedLfm2CpcfPrefixCheckpoint:
        """Reopen a checkpoint immediately before installing tensor clones."""

        if not isinstance(verified, VerifiedLfm2CpcfPrefixCheckpoint):
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "verified must be a verified prefix checkpoint."
            )
        result = self.open_checkpoint(verified.reference)
        if result.certificate_sha256 != verified.certificate_sha256:
            raise Lfm2CpcfResumeArtifactIntegrityError(
                "Checkpoint certificate changed before final use."
            )
        return result


__all__ = [
    "BOUNDARY_BINDING_KEY_SCHEMA",
    "BOUNDARY_KEY_SCHEMA",
    "CHECKPOINT_KEY_SCHEMA",
    "COMPLETED_LAYERS_SCHEMA",
    "FAILURE_KEY_SCHEMA",
    "INCOMING_STATE_BANK_RUNTIME_SCHEMA",
    "INSTALL_PROGRESS_SCHEMA",
    "RESUME_ARTIFACT_SCHEMA",
    "TARGET_TOPOLOGY_SNAPSHOT_SCHEMA",
    "Lfm2CpcfArtifactReference",
    "Lfm2CpcfBoundaryBindings",
    "Lfm2CpcfCheckpointBindings",
    "Lfm2CpcfFailureBindings",
    "Lfm2CpcfResumeArtifactConflictError",
    "Lfm2CpcfResumeArtifactError",
    "Lfm2CpcfResumeArtifactIntegrityError",
    "Lfm2CpcfResumeArtifactPublicationError",
    "Lfm2CpcfResumeArtifactStore",
    "VerifiedLfm2CpcfBoundaryBundle",
    "VerifiedLfm2CpcfFailureDiagnostic",
    "VerifiedLfm2CpcfPrefixCheckpoint",
    "build_lfm2_cpcf_checkpoint_bindings",
    "build_lfm2_cpcf_completed_layers_document",
    "build_lfm2_cpcf_incoming_state_bank_metadata",
    "build_lfm2_cpcf_installed_parameter_paths",
    "build_lfm2_cpcf_target_topology_snapshot",
    "flatten_lfm2_cpcf_incoming_state_bank",
]
