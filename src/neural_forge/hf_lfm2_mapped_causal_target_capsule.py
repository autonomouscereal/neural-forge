"""Immutable cache-only dense target-logit evidence for LFM2 causal folding.

The ordinary causal trie deliberately observes only a bounded set of donor
rows at each target prefix.  The strict tokenizer bridge contains a second,
orthogonal source of evidence: every target vocabulary row names exactly one
donor vocabulary row.  The sealed parent cache therefore supplies one exact
direct-mapped logit anchor for every target row and every retained causal
root, without loading or executing the parent.

This module derives those anchors directly from the authenticated strict
bridge row map and publishes only the compact ``[causal_roots,
target_vocabulary]`` matrix.  It does not depend on a calibration corpus or a
trace adapter: each sealed donor-canonical cache phase is projected with one
exact ``index_select``.  Full donor logits, parent hidden states, parent
weights, and model objects are never persisted.  Publication uses a sibling
staging directory, canonical JSON, Safetensors, and an atomic no-replace
rename.  Loading requires an externally pinned reference.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_bridge_only_control import _tensor_sha256
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    _atomic_rename_noreplace,
    _fsync_directory,
    _is_link_or_junction,
    _lexists,
    _read_canonical_json_with_raw,
    _real_directory,
    _regular_file,
    _stat_identity,
    _write_canonical_json,
)
from neural_forge.hf_lfm2_interface_frontier import (
    LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
    LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
    Lfm2CausalTrieEvidence,
    Lfm2TerminalCacheEvidence,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_MAPPED_CAUSAL_TARGET_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-mapped-causal-target-evidence-v1"
)
LFM2_MAPPED_CAUSAL_TARGET_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-mapped-causal-target-capsule-manifest-v1"
)
LFM2_MAPPED_CAUSAL_TARGET_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-mapped-causal-target-capsule-receipt-v1"
)
LFM2_DIRECT_MAPPED_CAUSAL_TARGET_PROJECTION_SCHEMA = (
    "neural-forge-lfm2-direct-mapped-causal-target-projection-v1"
)

_ARTIFACT_KIND = "lfm2-cache-only-dense-direct-mapped-causal-targets"
_STATUS = "immutable-cache-only-causal-target-evidence"
_TENSOR_NAME = "mapped-causal-targets.safetensors"
_MANIFEST_NAME = "capsule-manifest.json"
_RECEIPT_NAME = "capsule-receipt.json"
_ROOT_INVENTORY = frozenset({_TENSOR_NAME, _MANIFEST_NAME, _RECEIPT_NAME})
_STAGE_PREFIX = ".lfm2-mapped-causal-target-"
_TENSOR_KEYS = (
    "root_target_logits",
    "root_terminal_row_indices",
    "target_to_donor_row_indices",
)
_LINEAGE_FIELDS = (
    "terminal_evidence_sha256",
    "causal_trie_evidence_sha256",
    "cache_key",
    "parent_identity_sha256",
    "selective_session_sha256",
    "manifest_sha256",
    "completion_sha256",
    "causal_root_rows_sha256",
    "contextual_transition_plan_sha256",
    "target_prefix_rows_sha256",
    "bridge_artifact_sha256",
    "bridge_profile_sha256",
    "donor_tokenizer_semantics_sha256",
    "target_tokenizer_semantics_sha256",
    "mapping_sha256",
)
_ZERO_MODEL_OPERATION_AUDIT = {
    "parent_model_objects_constructed": 0,
    "parent_weight_tensor_loads": 0,
    "parent_model_forward_calls": 0,
    "parent_model_generate_calls": 0,
    "child_model_objects_constructed": 0,
    "child_model_forward_calls": 0,
    "child_model_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parameters_sparsified": 0,
    "network_access": False,
}


class Lfm2MappedCausalTargetCapsuleError(RuntimeError):
    """Dense mapped target evidence failed compilation or authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise Lfm2MappedCausalTargetCapsuleError(
            "PyTorch is required for mapped causal target evidence."
        ) from exc
    return torch


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - required production dependency
        raise Lfm2MappedCausalTargetCapsuleError(
            "Safetensors is required for mapped causal target capsules."
        ) from exc
    return safe_open, save_file


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            f"{label} must be one nonzero lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2MappedCausalTargetCapsuleError(
            f"{label} must be one positive integer."
        )
    return value


def _sealed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = canonical_sha256(result)
    return result


def _verify_seal(
    value: Any,
    *,
    field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MappedCausalTargetCapsuleError(f"{label} must be a mapping.")
    result = dict(value)
    body = dict(result)
    supplied = _digest(body.pop(field, None), f"{label} {field}")
    if canonical_sha256(body) != supplied:
        raise Lfm2MappedCausalTargetCapsuleError(f"{label} seal is invalid.")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    information = _regular_file(path, name=path.name)
    return {
        "path": path.name,
        "size_bytes": int(information.st_size),
        "sha256": _file_sha256(path),
    }


def _tensor_descriptor(value: Any) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": [int(size) for size in value.shape],
        "numel": int(value.numel()),
        "size_bytes": int(value.numel() * value.element_size()),
        "sha256": _tensor_sha256(value),
    }


def _valid_tensor_descriptor(
    value: Any,
    *,
    rows: int,
    columns: int,
) -> bool:
    if (
        not isinstance(value, Mapping)
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or isinstance(columns, bool)
        or not isinstance(columns, int)
        or columns <= 0
    ):
        return False
    shape = value.get("shape")
    numel = rows * columns
    try:
        _digest(value.get("sha256"), "projection tensor SHA-256")
    except Lfm2MappedCausalTargetCapsuleError:
        return False
    return bool(
        isinstance(value.get("dtype"), str)
        and value.get("dtype")
        and shape == [rows, columns]
        and value.get("numel") == numel
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and value.get("size_bytes") > 0
        and value.get("size_bytes") % numel == 0
    )


def _canonical_root_indices(trie: Lfm2CausalTrieEvidence) -> tuple[int, ...]:
    receipt = trie.as_dict()
    transition = receipt.get("contextual_transition_plan")
    roots = (
        transition.get("root_transition_plans")
        if isinstance(transition, Mapping)
        else None
    )
    if (
        not isinstance(roots, list)
        or len(roots) != receipt.get("root_row_count")
        or any(
            not isinstance(row, Mapping)
            or row.get("root_index") != index
            or isinstance(row.get("terminal_row_index"), bool)
            or not isinstance(row.get("terminal_row_index"), int)
            for index, row in enumerate(roots)
        )
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Causal trie does not expose canonical terminal-root indices."
        )
    return tuple(int(row["terminal_row_index"]) for row in roots)


def _direct_projection_record(
    donor_logits: Any,
    target_logits: Any,
    *,
    bridge: StrictTokenizerBridgeArtifact,
    cache_key: str,
    cache_batch_index: int,
    input_sha256: str,
    phase: str,
    logits_name: str,
    mapping_sha256: str,
    terminal_batch_selection_sha256: str,
    terminal_row_start: int,
    terminal_row_end: int,
    selected_payload_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": LFM2_DIRECT_MAPPED_CAUSAL_TARGET_PROJECTION_SCHEMA,
        "cache_key": _digest(cache_key, "projection cache key"),
        "cache_batch_index": cache_batch_index,
        "input_sha256": _digest(input_sha256, "projection input SHA-256"),
        "phase": phase,
        "logits_name": logits_name,
        "source_logits": _tensor_descriptor(donor_logits),
        "target_logits": _tensor_descriptor(target_logits),
        "mapping_sha256": _digest(mapping_sha256, "projection mapping SHA-256"),
        "terminal_batch_selection_sha256": _digest(
            terminal_batch_selection_sha256,
            "terminal batch selection SHA-256",
        ),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "terminal_row_start": terminal_row_start,
        "terminal_row_end": terminal_row_end,
        "selected_payload_sha256": _digest(
            selected_payload_sha256,
            "selected payload SHA-256",
        ),
        "direct_target_to_donor_index_select": True,
        "inverse_output_plan_used": False,
        "operation_audit": dict(_ZERO_MODEL_OPERATION_AUDIT),
    }
    return {**body, "projection_sha256": canonical_sha256(body)}


def _verified_terminal_receipt(
    value: Lfm2TerminalCacheEvidence | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, Lfm2TerminalCacheEvidence):
        return value.as_dict()
    terminal = _verify_seal(
        value,
        field="evidence_sha256",
        label="terminal cache evidence",
    )
    selected_batches = terminal.get("selected_batches")
    audit = terminal.get("operation_audit")
    try:
        for name in (
            "cache_key",
            "parent_identity_sha256",
            "selective_session_sha256",
            "manifest_sha256",
            "completion_sha256",
            "hidden_rows_sha256",
            "donor_logits_sha256",
            "bridge_artifact_sha256",
            "bridge_profile_sha256",
            "donor_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
            "semantic_alignment_sha256",
            "tokenizer_live_session_sha256",
            "causal_root_rows_sha256",
        ):
            _digest(terminal.get(name), name)
        terminal_rows = _positive_int(
            terminal.get("terminal_row_count"),
            "terminal_row_count",
        )
        _positive_int(terminal.get("hidden_size"), "hidden_size")
        _positive_int(
            terminal.get("donor_vocabulary_size"),
            "donor_vocabulary_size",
        )
        _positive_int(
            terminal.get("target_vocabulary_size"),
            "target_vocabulary_size",
        )
        root_rows = _positive_int(
            terminal.get("causal_root_row_count"),
            "causal_root_row_count",
        )
    except Lfm2MappedCausalTargetCapsuleError:
        raise
    if (
        terminal.get("schema") != LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA
        or terminal.get("status")
        != "complete-authenticated-terminal-cache-selection"
        or not isinstance(selected_batches, list)
        or not selected_batches
        or terminal.get("batch_count") != len(selected_batches)
        or root_rows > terminal_rows
        or not isinstance(audit, Mapping)
        or audit.get("network_access") is not False
        or any(
            audit.get(name) != 0
            for name in (
                "training_steps",
                "optimizer_constructions",
                "optimizer_steps",
                "autograd_grad_calls",
                "backward_calls",
                "parent_forward_calls",
                "parent_generate_calls",
                "parent_trace_recaptures",
                "child_forward_calls",
                "child_generate_calls",
            )
        )
        or terminal.get("sealed_parent_cache_reused_without_recapture") is not True
        or terminal.get("all_terminal_rows_retained_in_diagnostic_tensors")
        is not True
        or terminal.get(
            "all_eligible_exact_boundary_rows_retained_for_causal_replay"
        )
        is not True
        or terminal.get("unpaired_cached_prefill_logits_excluded") is not True
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Terminal cache evidence receipt is incomplete or unsafe."
        )
    selected_rows = 0
    for batch_index, batch in enumerate(selected_batches):
        phases = batch.get("phases") if isinstance(batch, Mapping) else None
        batch_rows = batch.get("terminal_rows") if isinstance(batch, Mapping) else None
        if isinstance(batch, Mapping):
            _digest(batch.get("input_sha256"), "terminal batch input SHA-256")
            _digest(
                batch.get("selection_sha256"),
                "terminal batch selection SHA-256",
            )
        if (
            not isinstance(batch, Mapping)
            or batch.get("batch_index") != batch_index
            or isinstance(batch_rows, bool)
            or not isinstance(batch_rows, int)
            or batch_rows <= 0
            or not isinstance(phases, list)
            or not phases
        ):
            raise Lfm2MappedCausalTargetCapsuleError(
                "Terminal selected-batch receipt is invalid."
            )
        phase_rows = 0
        for phase in phases:
            rows = phase.get("rows") if isinstance(phase, Mapping) else None
            if (
                not isinstance(phase, Mapping)
                or not isinstance(phase.get("phase"), str)
                or not phase.get("phase")
                or not isinstance(phase.get("logits_name"), str)
                or not phase.get("logits_name")
                or isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows <= 0
            ):
                raise Lfm2MappedCausalTargetCapsuleError(
                    "Terminal selected phase receipt is invalid."
                )
            phase_rows += rows
        if phase_rows != batch_rows:
            raise Lfm2MappedCausalTargetCapsuleError(
                "Terminal selected phase rows do not cover their batch."
            )
        selected_rows += batch_rows
    if selected_rows != terminal_rows:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Terminal selected batches do not cover every terminal row."
        )
    return terminal


def _projection_records_match_terminal_order(
    projections: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> bool:
    selected_batches = terminal.get("selected_batches")
    if not isinstance(selected_batches, list):
        return False
    record_index = 0
    terminal_cursor = 0
    for batch_index, selected_batch in enumerate(selected_batches):
        if not isinstance(selected_batch, Mapping):
            return False
        phases = selected_batch.get("phases")
        if not isinstance(phases, list):
            return False
        for phase in phases:
            if record_index >= len(projections) or not isinstance(phase, Mapping):
                return False
            record = projections[record_index]
            rows = phase.get("rows")
            if (
                not isinstance(record, Mapping)
                or isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows <= 0
                or record.get("cache_batch_index") != batch_index
                or record.get("input_sha256")
                != selected_batch.get("input_sha256")
                or record.get("phase") != phase.get("phase")
                or record.get("logits_name") != phase.get("logits_name")
                or record.get("terminal_batch_selection_sha256")
                != selected_batch.get("selection_sha256")
                or record.get("terminal_row_start") != terminal_cursor
                or record.get("terminal_row_end") != terminal_cursor + rows
            ):
                return False
            terminal_cursor += rows
            record_index += 1
    return bool(
        record_index == len(projections)
        and terminal_cursor == terminal.get("terminal_row_count")
    )


@dataclass(frozen=True, slots=True)
class Lfm2MappedCausalTargetEvidence:
    """Exact direct-mapped target logits in canonical causal-root order."""

    root_target_logits: Any = field(repr=False, compare=False)
    root_terminal_row_indices: Any = field(repr=False, compare=False)
    target_to_donor_row_indices: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        logits = self.root_target_logits
        root_indices = self.root_terminal_row_indices
        mapping = self.target_to_donor_row_indices
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or not torch.is_floating_point(logits)
            or torch.is_complex(logits)
            or logits.layout != torch.strided
            or logits.device.type == "meta"
            or logits.requires_grad
            or not bool(torch.isfinite(logits).all())
            or not isinstance(root_indices, torch.Tensor)
            or root_indices.ndim != 1
            or root_indices.dtype != torch.int64
            or root_indices.layout != torch.strided
            or root_indices.device.type == "meta"
            or root_indices.requires_grad
            or not isinstance(mapping, torch.Tensor)
            or mapping.ndim != 1
            or mapping.dtype != torch.int64
            or mapping.layout != torch.strided
            or mapping.device.type == "meta"
            or mapping.requires_grad
            or int(logits.shape[0]) <= 0
            or int(logits.shape[1]) <= 1
            or int(root_indices.numel()) != int(logits.shape[0])
            or int(mapping.numel()) != int(logits.shape[1])
            or bool((root_indices < 0).any())
            or bool((mapping < 0).any())
            or len(set(int(value) for value in root_indices.tolist()))
            != int(root_indices.numel())
        ):
            raise Lfm2MappedCausalTargetCapsuleError(
                "Mapped causal target tensors are invalid or ambiguous."
            )
        value = _verify_seal(
            self.receipt,
            field="evidence_sha256",
            label="mapped causal target evidence",
        )
        for name in _LINEAGE_FIELDS:
            _digest(value.get(name), name)
        projections = value.get("projection_records")
        projection_valid = isinstance(projections, list) and bool(projections)
        if projection_valid:
            expected_start = 0
            for record in projections:
                if not isinstance(record, Mapping):
                    projection_valid = False
                    break
                body = dict(record)
                supplied = body.pop("projection_sha256", None)
                start = body.get("terminal_row_start")
                end = body.get("terminal_row_end")
                try:
                    _digest(supplied, "projection record seal")
                    _digest(
                        body.get("selected_payload_sha256"),
                        "projection selected payload",
                    )
                    _digest(
                        body.get("terminal_batch_selection_sha256"),
                        "projection terminal batch selection",
                    )
                    _digest(body.get("input_sha256"), "projection input")
                except Lfm2MappedCausalTargetCapsuleError:
                    projection_valid = False
                    break
                rows = (
                    end - start
                    if isinstance(start, int)
                    and not isinstance(start, bool)
                    and isinstance(end, int)
                    and not isinstance(end, bool)
                    else 0
                )
                source_descriptor = body.get("source_logits")
                target_descriptor = body.get("target_logits")
                if (
                    supplied != canonical_sha256(body)
                    or body.get("schema")
                    != LFM2_DIRECT_MAPPED_CAUSAL_TARGET_PROJECTION_SCHEMA
                    or isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or start != expected_start
                    or end <= start
                    or isinstance(body.get("cache_batch_index"), bool)
                    or not isinstance(body.get("cache_batch_index"), int)
                    or body.get("cache_batch_index") < 0
                    or not isinstance(body.get("phase"), str)
                    or not body.get("phase")
                    or not isinstance(body.get("logits_name"), str)
                    or not body.get("logits_name")
                    or body.get("cache_key") != value.get("cache_key")
                    or body.get("mapping_sha256")
                    != value.get("mapping_sha256")
                    or body.get("bridge_artifact_sha256")
                    != value.get("bridge_artifact_sha256")
                    or body.get("bridge_profile_sha256")
                    != value.get("bridge_profile_sha256")
                    or body.get("donor_tokenizer_semantics_sha256")
                    != value.get("donor_tokenizer_semantics_sha256")
                    or body.get("target_tokenizer_semantics_sha256")
                    != value.get("target_tokenizer_semantics_sha256")
                    or not _valid_tensor_descriptor(
                        source_descriptor,
                        rows=rows,
                        columns=value.get("donor_row_count"),
                    )
                    or not _valid_tensor_descriptor(
                        target_descriptor,
                        rows=rows,
                        columns=value.get("target_row_count"),
                    )
                    or source_descriptor.get("dtype")
                    != target_descriptor.get("dtype")
                    or source_descriptor.get("size_bytes")
                    * target_descriptor.get("numel")
                    != target_descriptor.get("size_bytes")
                    * source_descriptor.get("numel")
                    or body.get("direct_target_to_donor_index_select") is not True
                    or body.get("inverse_output_plan_used") is not False
                    or body.get("operation_audit") != _ZERO_MODEL_OPERATION_AUDIT
                ):
                    projection_valid = False
                    break
                expected_start = end
            projection_valid = bool(
                projection_valid
                and expected_start == value.get("terminal_row_count")
            )
        audit = value.get("operation_audit")
        if (
            value.get("schema") != LFM2_MAPPED_CAUSAL_TARGET_EVIDENCE_SCHEMA
            or value.get("status") != _STATUS
            or value.get("artifact_kind") != _ARTIFACT_KIND
            or value.get("root_row_count") != int(logits.shape[0])
            or value.get("target_row_count") != int(logits.shape[1])
            or value.get("terminal_row_count")
            <= int(root_indices.max().item())
            or value.get("donor_row_count") <= int(mapping.max().item())
            or value.get("root_target_logits_sha256") != _tensor_sha256(logits)
            or value.get("root_terminal_row_indices_sha256")
            != _tensor_sha256(root_indices)
            or value.get("target_to_donor_row_indices_sha256")
            != _tensor_sha256(mapping)
            or value.get("mapping_sha256")
            != canonical_sha256([int(item) for item in mapping.tolist()])
            or value.get("mapping_source")
            != "authenticated-strict-tokenizer-bridge-transport-profile"
            or value.get("root_terminal_row_indices")
            != [int(item) for item in root_indices.tolist()]
            or value.get("all_root_target_cells_supported") is not True
            or value.get("direct_strict_row_projection") is not True
            or value.get("inverse_output_plan_used") is not False
            or value.get("all_donor_mass_bucket_aggregation_used") is not False
            or value.get("full_cross_tokenizer_mass_equivalence_claimed") is not False
            or value.get("parent_model_or_weight_payload_persisted") is not False
            or value.get("full_donor_logits_persisted") is not False
            or not projection_valid
            or not isinstance(audit, Mapping)
            or any(audit.get(name) != expected for name, expected in _ZERO_MODEL_OPERATION_AUDIT.items())
            or audit.get("parent_cache_selective_authenticated_sessions") != 1
            or audit.get("parent_cache_selected_tensor_payloads")
            != value.get("selected_tensor_payload_count")
            or audit.get("parent_cache_selected_logit_tensors")
            != len(projections)
        ):
            raise Lfm2MappedCausalTargetCapsuleError(
                "Mapped causal target evidence receipt or tensor lineage is invalid."
            )
        object.__setattr__(
            self,
            "root_target_logits",
            logits.detach().to(device="cpu").contiguous(),
        )
        object.__setattr__(
            self,
            "root_terminal_row_indices",
            root_indices.detach().to(device="cpu").contiguous(),
        )
        object.__setattr__(
            self,
            "target_to_donor_row_indices",
            mapping.detach().to(device="cpu").contiguous(),
        )
        object.__setattr__(self, "receipt", MappingProxyType(value))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2MappedCausalTargetReference:
    """External pins required to authenticate one mapped-target capsule."""

    path: Path
    receipt_sha256: str
    receipt_file_sha256: str
    manifest_sha256: str
    tensor_file_sha256: str
    evidence_sha256: str
    root_target_logits_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().absolute())
        for name in (
            "receipt_sha256",
            "receipt_file_sha256",
            "manifest_sha256",
            "tensor_file_sha256",
            "evidence_sha256",
            "root_target_logits_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "receipt_sha256": self.receipt_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "tensor_file_sha256": self.tensor_file_sha256,
            "evidence_sha256": self.evidence_sha256,
            "root_target_logits_sha256": self.root_target_logits_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "Lfm2MappedCausalTargetReference":
        expected = {
            "path",
            "receipt_sha256",
            "receipt_file_sha256",
            "manifest_sha256",
            "tensor_file_sha256",
            "evidence_sha256",
            "root_target_logits_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Lfm2MappedCausalTargetCapsuleError(
                "Mapped causal target reference has missing or unknown fields."
            )
        path = value.get("path")
        if not isinstance(path, (str, os.PathLike)) or not os.fspath(path).strip():
            raise Lfm2MappedCausalTargetCapsuleError(
                "Mapped causal target reference path must be nonempty."
            )
        return cls(
            path=Path(path),
            **{name: value[name] for name in expected if name != "path"},
        )


@dataclass(frozen=True, slots=True)
class Lfm2MappedCausalTargetCapsule:
    """One freshly loaded and fully authenticated mapped-target capsule."""

    path: Path
    evidence: Lfm2MappedCausalTargetEvidence
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    reference: Lfm2MappedCausalTargetReference


def _new_output(value: str | os.PathLike[str]) -> Path:
    output = Path(value).expanduser().absolute()
    if _lexists(output):
        raise Lfm2MappedCausalTargetCapsuleError(
            f"Refusing to replace existing mapped causal target capsule {output}."
        )
    parent = output.parent
    _real_directory(parent, name="mapped causal target capsule parent")
    if _is_link_or_junction(parent) or parent.resolve(strict=True) != parent:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule parent cannot traverse a link."
        )
    return output


def _safe_remove_stage(
    stage: Path,
    *,
    parent: Path,
    identity: tuple[int, int],
) -> None:
    if not _lexists(stage):
        return
    current = stage.lstat()
    if (
        _stat_identity(current) != identity
        or not stat.S_ISDIR(current.st_mode)
        or _is_link_or_junction(stage)
        or stage.resolve(strict=True).parent != parent.resolve(strict=True)
        or not stage.name.startswith(_STAGE_PREFIX)
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Refusing to remove unauthenticated mapped-target staging."
        )
    shutil.rmtree(stage)
    _fsync_directory(parent)


def _root_path(value: str | os.PathLike[str]) -> tuple[Path, tuple[int, int]]:
    lexical = Path(value).expanduser().absolute()
    if _is_link_or_junction(lexical):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule cannot be a link or junction."
        )
    try:
        root = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule is missing."
        ) from exc
    if root != lexical:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule path cannot traverse a link."
        )
    root_stat = _real_directory(root, name="mapped causal target capsule")
    try:
        entries = tuple(os.scandir(root))
    except OSError as exc:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule cannot be enumerated."
        ) from exc
    if {entry.name for entry in entries} != _ROOT_INVENTORY:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule has unexpected root entries."
        )
    for name in _ROOT_INVENTORY:
        _regular_file(root / name, name=name)
    return root, _stat_identity(root_stat)


def _tensor_metadata(evidence_sha256: str) -> dict[str, str]:
    return {
        "schema": LFM2_MAPPED_CAUSAL_TARGET_EVIDENCE_SCHEMA,
        "artifact_kind": _ARTIFACT_KIND,
        "evidence_sha256": evidence_sha256,
    }


def _read_tensor_header(path: Path, evidence_sha256: str) -> dict[str, Any]:
    safe_open, _save_file = _safetensors()
    before = _stat_identity(_regular_file(path, name=_TENSOR_NAME))
    descriptors: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if handle.metadata() != _tensor_metadata(evidence_sha256):
            raise Lfm2MappedCausalTargetCapsuleError(
                "Mapped causal target Safetensors metadata is invalid."
            )
        if tuple(sorted(handle.keys())) != tuple(sorted(_TENSOR_KEYS)):
            raise Lfm2MappedCausalTargetCapsuleError(
                "Mapped causal target Safetensors keys are invalid."
            )
        for name in _TENSOR_KEYS:
            value = handle.get_tensor(name)
            descriptors[name] = _tensor_descriptor(value)
    after = _stat_identity(_regular_file(path, name=_TENSOR_NAME))
    if before != after:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target tensor file changed while authenticating."
        )
    return descriptors


def _open_capsule(reference: Lfm2MappedCausalTargetReference) -> dict[str, Any]:
    if not isinstance(reference, Lfm2MappedCausalTargetReference):
        raise TypeError("reference must be Lfm2MappedCausalTargetReference.")
    root, root_identity = _root_path(reference.path)
    receipt, receipt_raw = _read_canonical_json_with_raw(
        root / _RECEIPT_NAME,
        name=_RECEIPT_NAME,
    )
    receipt = _verify_seal(
        receipt,
        field="receipt_sha256",
        label="mapped causal target capsule receipt",
    )
    manifest, manifest_raw = _read_canonical_json_with_raw(
        root / _MANIFEST_NAME,
        name=_MANIFEST_NAME,
    )
    manifest = _verify_seal(
        manifest,
        field="manifest_sha256",
        label="mapped causal target capsule manifest",
    )
    tensor_record = manifest.get("tensor_file")
    receipt_tensor_record = receipt.get("tensor_file")
    manifest_record = receipt.get("manifest_file")
    evidence_receipt = manifest.get("evidence")
    if not isinstance(evidence_receipt, Mapping):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule has no evidence receipt."
        )
    evidence_receipt = _verify_seal(
        evidence_receipt,
        field="evidence_sha256",
        label="mapped causal target evidence",
    )
    if (
        receipt.get("schema") != LFM2_MAPPED_CAUSAL_TARGET_RECEIPT_SCHEMA
        or manifest.get("schema") != LFM2_MAPPED_CAUSAL_TARGET_MANIFEST_SCHEMA
        or receipt.get("artifact_kind") != _ARTIFACT_KIND
        or manifest.get("artifact_kind") != _ARTIFACT_KIND
        or receipt.get("status") != _STATUS
        or manifest.get("status") != _STATUS
        or hashlib.sha256(receipt_raw).hexdigest()
        != reference.receipt_file_sha256
        or receipt.get("receipt_sha256") != reference.receipt_sha256
        or manifest.get("manifest_sha256") != reference.manifest_sha256
        or receipt.get("manifest_sha256") != reference.manifest_sha256
        or receipt.get("evidence_sha256") != reference.evidence_sha256
        or manifest.get("evidence_sha256") != reference.evidence_sha256
        or evidence_receipt.get("evidence_sha256") != reference.evidence_sha256
        or evidence_receipt.get("root_target_logits_sha256")
        != reference.root_target_logits_sha256
        or not isinstance(tensor_record, Mapping)
        or tensor_record != receipt_tensor_record
        or tensor_record.get("path") != _TENSOR_NAME
        or tensor_record.get("sha256") != reference.tensor_file_sha256
        or tensor_record != _file_record(root / _TENSOR_NAME)
        or not isinstance(manifest_record, Mapping)
        or manifest_record.get("path") != _MANIFEST_NAME
        or manifest_record.get("size_bytes") != len(manifest_raw)
        or manifest_record.get("sha256")
        != hashlib.sha256(manifest_raw).hexdigest()
        or receipt.get("operation_audit") != evidence_receipt.get("operation_audit")
        or manifest.get("operation_audit") != evidence_receipt.get("operation_audit")
        or manifest.get("full_donor_logits_persisted") is not False
        or manifest.get("parent_model_or_weight_payload_persisted") is not False
        or manifest.get("inverse_output_plan_used") is not False
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule differs from its external pins."
        )
    descriptors = _read_tensor_header(
        root / _TENSOR_NAME,
        reference.evidence_sha256,
    )
    if descriptors != manifest.get("tensor_descriptors"):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target tensor descriptors changed."
        )
    descriptor_receipt_fields = {
        "root_target_logits": "root_target_logits_sha256",
        "root_terminal_row_indices": "root_terminal_row_indices_sha256",
        "target_to_donor_row_indices": (
            "target_to_donor_row_indices_sha256"
        ),
    }
    if any(
        descriptors[name].get("sha256")
        != evidence_receipt.get(receipt_field)
        for name, receipt_field in descriptor_receipt_fields.items()
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target tensor bytes differ from the evidence receipt."
        )
    current_root, current_identity = _root_path(root)
    if current_root != root or current_identity != root_identity:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target capsule changed during authentication."
        )
    return {
        "root": root,
        "root_identity": root_identity,
        "manifest": manifest,
        "receipt": receipt,
        "evidence_receipt": dict(evidence_receipt),
    }


def build_lfm2_mapped_causal_target_evidence(
    donor_terminal_logits: Any,
    *,
    terminal_evidence: Lfm2TerminalCacheEvidence | Mapping[str, Any],
    causal_trie_evidence: Lfm2CausalTrieEvidence,
    bridge: StrictTokenizerBridgeArtifact,
    selective_session_receipt: Mapping[str, Any],
    projection_records: Sequence[Mapping[str, Any]],
    selected_tensor_payload_count: int,
    selected_tensor_logical_bytes: int,
) -> Lfm2MappedCausalTargetEvidence:
    """Derive dense roots from authenticated donor-terminal cache logits.

    The public builder deliberately does not accept caller-projected root
    logits.  It first authenticates the complete donor-terminal tensor against
    the sealed terminal receipt, reproduces every strict bridge projection,
    verifies the per-phase projection descriptors, and only then selects the
    canonical causal roots.  This prevents arbitrary root bytes from being
    resealed under otherwise authentic cache lineage.
    """

    torch = _torch()
    if not isinstance(causal_trie_evidence, Lfm2CausalTrieEvidence):
        raise TypeError("causal_trie_evidence must be Lfm2CausalTrieEvidence.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_files()
    profile = bridge.transport_profile
    bridge_mapping = profile.target_to_donor_row_indices
    terminal = _verified_terminal_receipt(terminal_evidence)
    trie = causal_trie_evidence.as_dict()
    session = dict(selective_session_receipt)
    canonical_roots = _canonical_root_indices(causal_trie_evidence)
    if (
        not isinstance(donor_terminal_logits, torch.Tensor)
        or donor_terminal_logits.ndim != 2
        or not torch.is_floating_point(donor_terminal_logits)
        or torch.is_complex(donor_terminal_logits)
        or donor_terminal_logits.layout != torch.strided
        or donor_terminal_logits.device.type == "meta"
        or donor_terminal_logits.requires_grad
        or not bool(torch.isfinite(donor_terminal_logits).all())
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Donor terminal logits must be one finite detached dense matrix."
        )
    donor_logits = donor_terminal_logits.detach().to(device="cpu").contiguous()
    roots = torch.tensor(canonical_roots, dtype=torch.int64)
    mapping = torch.tensor(bridge_mapping, dtype=torch.int64)
    terminal_target_logits = donor_logits.index_select(1, mapping).contiguous()
    logits = terminal_target_logits.index_select(0, roots).contiguous()
    projection_values = [dict(value) for value in projection_records]
    projection_descriptors_match = True
    for record in projection_values:
        start = record.get("terminal_row_start")
        end = record.get("terminal_row_end")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > int(donor_logits.shape[0])
            or record.get("source_logits")
            != _tensor_descriptor(donor_logits[start:end].contiguous())
            or record.get("target_logits")
            != _tensor_descriptor(terminal_target_logits[start:end].contiguous())
        ):
            projection_descriptors_match = False
            break
    if (
        terminal.get("schema") != LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA
        or trie.get("schema") != LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA
        or trie.get("parent_terminal_evidence_sha256")
        != terminal.get("evidence_sha256")
        or tuple(donor_logits.shape)
        != (
            terminal.get("terminal_row_count"),
            profile.donor_row_capacity,
        )
        or _tensor_sha256(donor_logits) != terminal.get("donor_logits_sha256")
        or int(logits.shape[0]) != len(canonical_roots)
        or int(logits.shape[1]) != profile.target_row_capacity
        or terminal.get("target_vocabulary_size")
        != profile.target_row_capacity
        or terminal.get("donor_vocabulary_size")
        != profile.donor_row_capacity
        or terminal.get("bridge_artifact_sha256")
        != bridge.artifact_sha256
        or terminal.get("bridge_profile_sha256")
        != bridge.profile_fingerprint
        or terminal.get("donor_tokenizer_semantics_sha256")
        != bridge.donor_tokenizer_semantics_sha256
        or terminal.get("target_tokenizer_semantics_sha256")
        != bridge.target_tokenizer_semantics_sha256
        or terminal.get("cache_key") != session.get("cache_key")
        or terminal.get("selective_session_sha256")
        != session.get("session_sha256")
        or terminal.get("manifest_sha256") != session.get("manifest_sha256")
        or terminal.get("completion_sha256")
        != session.get("completion_sha256")
        or not projection_values
        or not projection_descriptors_match
        or not _projection_records_match_terminal_order(
            projection_values,
            terminal,
        )
        or selected_tensor_payload_count <= 0
        or selected_tensor_logical_bytes <= 0
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Dense mapped roots do not match terminal, trie, bridge, or session lineage."
        )
    audit = {
        **_ZERO_MODEL_OPERATION_AUDIT,
        "parent_cache_selective_authenticated_sessions": 1,
        "parent_cache_selected_tensor_payloads": int(selected_tensor_payload_count),
        "parent_cache_selected_logit_tensors": len(projection_values),
        "parent_cache_selected_logical_bytes": int(selected_tensor_logical_bytes),
    }
    body = {
        "schema": LFM2_MAPPED_CAUSAL_TARGET_EVIDENCE_SCHEMA,
        "artifact_kind": _ARTIFACT_KIND,
        "status": _STATUS,
        "terminal_evidence_sha256": terminal["evidence_sha256"],
        "causal_trie_evidence_sha256": trie["result_sha256"],
        "cache_key": terminal["cache_key"],
        "parent_identity_sha256": terminal["parent_identity_sha256"],
        "selective_session_sha256": terminal["selective_session_sha256"],
        "manifest_sha256": terminal["manifest_sha256"],
        "completion_sha256": terminal["completion_sha256"],
        "causal_root_rows_sha256": terminal["causal_root_rows_sha256"],
        "contextual_transition_plan_sha256": trie[
            "contextual_transition_plan_sha256"
        ],
        "target_prefix_rows_sha256": trie["target_prefix_rows_sha256"],
        "bridge_artifact_sha256": terminal["bridge_artifact_sha256"],
        "bridge_profile_sha256": terminal["bridge_profile_sha256"],
        "donor_tokenizer_semantics_sha256": terminal[
            "donor_tokenizer_semantics_sha256"
        ],
        "target_tokenizer_semantics_sha256": terminal[
            "target_tokenizer_semantics_sha256"
        ],
        "mapping_sha256": canonical_sha256(
            [int(item) for item in mapping.tolist()]
        ),
        "mapping_source": (
            "authenticated-strict-tokenizer-bridge-transport-profile"
        ),
        "terminal_row_count": terminal["terminal_row_count"],
        "root_row_count": int(logits.shape[0]),
        "target_row_count": int(logits.shape[1]),
        "donor_row_count": terminal["donor_vocabulary_size"],
        "root_target_logits_sha256": _tensor_sha256(logits),
        "root_terminal_row_indices_sha256": _tensor_sha256(roots),
        "target_to_donor_row_indices_sha256": _tensor_sha256(mapping),
        "root_terminal_row_indices": [int(item) for item in roots.tolist()],
        "projection_records": projection_values,
        "projection_record_count": len(projection_values),
        "selected_tensor_payload_count": int(selected_tensor_payload_count),
        "selected_tensor_logical_bytes": int(selected_tensor_logical_bytes),
        "all_root_target_cells_supported": True,
        "direct_strict_row_projection": True,
        "inverse_output_plan_used": False,
        "all_donor_mass_bucket_aggregation_used": False,
        "full_cross_tokenizer_mass_equivalence_claimed": False,
        "contextual_decomposition_trie_remains_separate_evidence": True,
        "parent_model_or_weight_payload_persisted": False,
        "full_donor_logits_persisted": False,
        "operation_audit": audit,
    }
    return Lfm2MappedCausalTargetEvidence(
        root_target_logits=logits,
        root_terminal_row_indices=roots,
        target_to_donor_row_indices=mapping,
        receipt={**body, "evidence_sha256": canonical_sha256(body)},
    )


def collect_lfm2_mapped_causal_target_evidence(
    cache: Any,
    *,
    terminal_evidence: Lfm2TerminalCacheEvidence | Mapping[str, Any],
    causal_trie_evidence: Lfm2CausalTrieEvidence,
    bridge: StrictTokenizerBridgeArtifact,
) -> Lfm2MappedCausalTargetEvidence:
    """Open one sealed cache session and collect exact mapped root logits.

    This function never accepts a parent model.  The only model-derived bytes
    read are the already-sealed per-phase logit tensors in ``cache``.
    """

    torch = _torch()
    if not isinstance(causal_trie_evidence, Lfm2CausalTrieEvidence):
        raise TypeError("causal_trie_evidence must be Lfm2CausalTrieEvidence.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_files()
    profile = bridge.transport_profile
    mapping_values = profile.target_to_donor_row_indices
    mapping_sha256 = canonical_sha256(list(mapping_values))
    mapping = torch.tensor(mapping_values, dtype=torch.int64)
    terminal = _verified_terminal_receipt(terminal_evidence)
    selected_batches = terminal.get("selected_batches")
    if (
        getattr(cache, "cache_key", None) != terminal.get("cache_key")
        or getattr(cache, "parent_fingerprint", None)
        != terminal.get("parent_identity_sha256")
        or not isinstance(selected_batches, list)
        or terminal.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or terminal.get("bridge_profile_sha256") != bridge.profile_fingerprint
        or terminal.get("donor_tokenizer_semantics_sha256")
        != bridge.donor_tokenizer_semantics_sha256
        or terminal.get("target_tokenizer_semantics_sha256")
        != bridge.target_tokenizer_semantics_sha256
        or terminal.get("donor_vocabulary_size") != profile.donor_row_capacity
        or terminal.get("target_vocabulary_size") != profile.target_row_capacity
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Sealed cache, bridge, and terminal evidence have different lineage."
        )
    source_donor_parts = []
    projection_records = []
    terminal_cursor = 0
    payload_count = 0
    logical_bytes = 0
    with cache.selective_authenticated_session() as session:
        if len(session.batch_references) != len(selected_batches):
            raise Lfm2MappedCausalTargetCapsuleError(
                "Selective cache session has a different record count."
            )
        for expected_index, selected_batch in enumerate(selected_batches):
            phases = selected_batch.get("phases")
            reference = session.batch_references[expected_index]
            if (
                selected_batch.get("batch_index") != expected_index
                or not isinstance(phases, list)
                or not phases
                or reference.index != expected_index
                or selected_batch.get("input_sha256") != reference.input_sha256
            ):
                raise Lfm2MappedCausalTargetCapsuleError(
                    "Terminal batch order differs from donor-canonical cache order."
                )
            names = tuple(sorted({str(phase.get("logits_name")) for phase in phases}))
            if any(not name or name == "None" for name in names):
                raise Lfm2MappedCausalTargetCapsuleError(
                    "Terminal phase lacks a sealed logit tensor name."
                )
            payload = session.load_selected_tensors(expected_index, names)
            payload_count += 1
            logical_bytes += int(payload.logical_bytes)
            try:
                for phase_row in phases:
                    phase = phase_row.get("phase")
                    rows = phase_row.get("rows")
                    logits_name = phase_row.get("logits_name")
                    donor_logits = payload.tensors.get(logits_name)
                    if (
                        not isinstance(phase, str)
                        or isinstance(rows, bool)
                        or not isinstance(rows, int)
                        or rows <= 0
                        or not isinstance(donor_logits, torch.Tensor)
                        or donor_logits.ndim != 3
                        or int(donor_logits.shape[0]) != 1
                        or int(donor_logits.shape[1]) != rows
                        or int(donor_logits.shape[2]) != profile.donor_row_capacity
                    ):
                        raise Lfm2MappedCausalTargetCapsuleError(
                            "Selected terminal logits do not match their phase receipt."
                        )
                    selected_terminal_rows = donor_logits.detach().reshape(
                        rows,
                        int(donor_logits.shape[-1]),
                    ).to(device="cpu").contiguous()
                    source_donor_parts.append(selected_terminal_rows)
                    if isinstance(terminal_evidence, Lfm2TerminalCacheEvidence):
                        expected_terminal_rows = terminal_evidence.donor_logits[
                            terminal_cursor : terminal_cursor + rows
                        ].contiguous()
                        if not torch.equal(
                            selected_terminal_rows,
                            expected_terminal_rows,
                        ):
                            raise Lfm2MappedCausalTargetCapsuleError(
                                "Selected cache logits differ from the sealed "
                                "terminal evidence rows."
                            )
                    projected = selected_terminal_rows.index_select(
                        1,
                        mapping,
                    ).contiguous()
                    projection_records.append(
                        _direct_projection_record(
                            selected_terminal_rows,
                            projected,
                            bridge=bridge,
                            cache_key=terminal["cache_key"],
                            cache_batch_index=expected_index,
                            input_sha256=reference.input_sha256,
                            phase=phase,
                            logits_name=logits_name,
                            mapping_sha256=mapping_sha256,
                            terminal_batch_selection_sha256=selected_batch[
                                "selection_sha256"
                            ],
                            terminal_row_start=terminal_cursor,
                            terminal_row_end=terminal_cursor + rows,
                            selected_payload_sha256=payload.selection_sha256,
                        )
                    )
                    terminal_cursor += rows
            finally:
                session.release_selected_tensors(payload)
        session.verify()
        session_receipt = session.as_dict()
    if (
        not source_donor_parts
        or terminal_cursor != terminal.get("terminal_row_count")
        or session_receipt.get("session_sha256")
        != terminal.get("selective_session_sha256")
        or session_receipt.get("manifest_sha256")
        != terminal.get("manifest_sha256")
        or session_receipt.get("completion_sha256")
        != terminal.get("completion_sha256")
    ):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped-target cache session differs from terminal evidence."
        )
    selected_donor_logits = torch.cat(source_donor_parts, dim=0).contiguous()
    if _tensor_sha256(selected_donor_logits) != terminal.get("donor_logits_sha256"):
        raise Lfm2MappedCausalTargetCapsuleError(
            "Selected cache logits differ from the terminal evidence digest."
        )
    return build_lfm2_mapped_causal_target_evidence(
        selected_donor_logits,
        terminal_evidence=terminal_evidence,
        causal_trie_evidence=causal_trie_evidence,
        bridge=bridge,
        selective_session_receipt=session_receipt,
        projection_records=projection_records,
        selected_tensor_payload_count=payload_count,
        selected_tensor_logical_bytes=logical_bytes,
    )


def publish_lfm2_mapped_causal_target_capsule(
    evidence: Lfm2MappedCausalTargetEvidence,
    *,
    output_directory: str | os.PathLike[str],
) -> Lfm2MappedCausalTargetReference:
    """Atomically publish one compact mapped causal target evidence capsule."""

    if not isinstance(evidence, Lfm2MappedCausalTargetEvidence):
        raise TypeError("evidence must be Lfm2MappedCausalTargetEvidence.")
    evidence = Lfm2MappedCausalTargetEvidence(
        root_target_logits=(
            evidence.root_target_logits.detach().cpu().clone().contiguous()
        ),
        root_terminal_row_indices=(
            evidence.root_terminal_row_indices.detach()
            .cpu()
            .clone()
            .contiguous()
        ),
        target_to_donor_row_indices=(
            evidence.target_to_donor_row_indices.detach()
            .cpu()
            .clone()
            .contiguous()
        ),
        receipt=evidence.receipt,
    )
    output = _new_output(output_directory)
    stage = Path(
        tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=str(output.parent))
    ).absolute()
    stage_identity = _stat_identity(stage.lstat())
    moved = False
    try:
        _safe_open, save_file = _safetensors()
        tensors = {
            "root_target_logits": evidence.root_target_logits,
            "root_terminal_row_indices": evidence.root_terminal_row_indices,
            "target_to_donor_row_indices": (
                evidence.target_to_donor_row_indices
            ),
        }
        tensor_path = stage / _TENSOR_NAME
        save_file(
            tensors,
            str(tensor_path),
            metadata=_tensor_metadata(evidence.receipt["evidence_sha256"]),
        )
        # Windows rejects ``fsync`` on a descriptor opened read-only.  Open the
        # freshly written immutable payload read/write solely for the durability
        # barrier; no bytes are modified here.
        with tensor_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        tensor_record = _file_record(tensor_path)
        tensor_descriptors = {
            name: _tensor_descriptor(value) for name, value in tensors.items()
        }
        manifest_body = {
            "schema": LFM2_MAPPED_CAUSAL_TARGET_MANIFEST_SCHEMA,
            "artifact_kind": _ARTIFACT_KIND,
            "status": _STATUS,
            "evidence": evidence.as_dict(),
            "evidence_sha256": evidence.receipt["evidence_sha256"],
            "tensor_file": tensor_record,
            "tensor_descriptors": tensor_descriptors,
            "root_row_count": int(evidence.root_target_logits.shape[0]),
            "target_row_count": int(evidence.root_target_logits.shape[1]),
            "root_target_logits_sha256": evidence.receipt[
                "root_target_logits_sha256"
            ],
            "full_donor_logits_persisted": False,
            "parent_model_or_weight_payload_persisted": False,
            "inverse_output_plan_used": False,
            "operation_audit": evidence.receipt["operation_audit"],
        }
        manifest = _sealed(manifest_body, "manifest_sha256")
        manifest_path = stage / _MANIFEST_NAME
        _write_canonical_json(manifest_path, manifest)
        manifest_record = _file_record(manifest_path)
        receipt_body = {
            "schema": LFM2_MAPPED_CAUSAL_TARGET_RECEIPT_SCHEMA,
            "artifact_kind": _ARTIFACT_KIND,
            "status": _STATUS,
            "evidence_sha256": evidence.receipt["evidence_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file": manifest_record,
            "tensor_file": tensor_record,
            "root_target_logits_sha256": evidence.receipt[
                "root_target_logits_sha256"
            ],
            "operation_audit": evidence.receipt["operation_audit"],
        }
        receipt = _sealed(receipt_body, "receipt_sha256")
        receipt_path = stage / _RECEIPT_NAME
        _write_canonical_json(receipt_path, receipt)
        _fsync_directory(stage)
        staged_reference = Lfm2MappedCausalTargetReference(
            path=stage,
            receipt_sha256=receipt["receipt_sha256"],
            receipt_file_sha256=_file_sha256(receipt_path),
            manifest_sha256=manifest["manifest_sha256"],
            tensor_file_sha256=tensor_record["sha256"],
            evidence_sha256=evidence.receipt["evidence_sha256"],
            root_target_logits_sha256=evidence.receipt[
                "root_target_logits_sha256"
            ],
        )
        _open_capsule(staged_reference)
        _atomic_rename_noreplace(stage, output)
        moved = True
        _fsync_directory(output.parent)
        reference = Lfm2MappedCausalTargetReference(
            path=output,
            receipt_sha256=staged_reference.receipt_sha256,
            receipt_file_sha256=staged_reference.receipt_file_sha256,
            manifest_sha256=staged_reference.manifest_sha256,
            tensor_file_sha256=staged_reference.tensor_file_sha256,
            evidence_sha256=staged_reference.evidence_sha256,
            root_target_logits_sha256=(
                staged_reference.root_target_logits_sha256
            ),
        )
        _open_capsule(reference)
        return reference
    except FileExistsError as exc:
        raise Lfm2MappedCausalTargetCapsuleError(
            f"Mapped causal target capsule destination already exists: {output}."
        ) from exc
    finally:
        if not moved and _lexists(stage):
            _safe_remove_stage(
                stage,
                parent=output.parent,
                identity=stage_identity,
            )


def collect_and_publish_lfm2_mapped_causal_target_capsule(
    cache: Any,
    *,
    terminal_evidence: Lfm2TerminalCacheEvidence | Mapping[str, Any],
    causal_trie_evidence: Lfm2CausalTrieEvidence,
    bridge: StrictTokenizerBridgeArtifact,
    output_directory: str | os.PathLike[str],
) -> Lfm2MappedCausalTargetReference:
    """Collect through one cache session, then atomically publish the result."""

    evidence = collect_lfm2_mapped_causal_target_evidence(
        cache,
        terminal_evidence=terminal_evidence,
        causal_trie_evidence=causal_trie_evidence,
        bridge=bridge,
    )
    return publish_lfm2_mapped_causal_target_capsule(
        evidence,
        output_directory=output_directory,
    )


def authenticate_lfm2_mapped_causal_target_capsule(
    reference: Lfm2MappedCausalTargetReference,
) -> Mapping[str, Any]:
    """Authenticate all capsule bytes and external pins without returning tensors."""

    opened = _open_capsule(reference)
    return MappingProxyType(
        {
            "path": opened["root"],
            "root_identity": opened["root_identity"],
            "manifest": MappingProxyType(dict(opened["manifest"])),
            "receipt": MappingProxyType(dict(opened["receipt"])),
            "evidence_receipt": MappingProxyType(
                dict(opened["evidence_receipt"])
            ),
        }
    )


def load_lfm2_mapped_causal_target_capsule(
    reference: Lfm2MappedCausalTargetReference,
) -> Lfm2MappedCausalTargetCapsule:
    """Load a fresh, externally pinned mapped target evidence tensor set."""

    opened = _open_capsule(reference)
    safe_open, _save_file = _safetensors()
    tensor_path = opened["root"] / _TENSOR_NAME
    identity = _stat_identity(_regular_file(tensor_path, name=_TENSOR_NAME))
    tensors = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        for name in _TENSOR_KEYS:
            tensors[name] = handle.get_tensor(name).detach().cpu().contiguous()
    if _stat_identity(_regular_file(tensor_path, name=_TENSOR_NAME)) != identity:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Mapped causal target tensor file changed while loading."
        )
    evidence = Lfm2MappedCausalTargetEvidence(
        root_target_logits=tensors["root_target_logits"],
        root_terminal_row_indices=tensors["root_terminal_row_indices"],
        target_to_donor_row_indices=tensors["target_to_donor_row_indices"],
        receipt=opened["evidence_receipt"],
    )
    if evidence.receipt["evidence_sha256"] != reference.evidence_sha256:
        raise Lfm2MappedCausalTargetCapsuleError(
            "Loaded mapped target evidence differs from its external pin."
        )
    return Lfm2MappedCausalTargetCapsule(
        path=opened["root"],
        evidence=evidence,
        manifest=MappingProxyType(dict(opened["manifest"])),
        receipt=MappingProxyType(dict(opened["receipt"])),
        reference=reference,
    )


__all__ = [
    "LFM2_MAPPED_CAUSAL_TARGET_EVIDENCE_SCHEMA",
    "LFM2_MAPPED_CAUSAL_TARGET_MANIFEST_SCHEMA",
    "LFM2_MAPPED_CAUSAL_TARGET_RECEIPT_SCHEMA",
    "Lfm2MappedCausalTargetCapsule",
    "Lfm2MappedCausalTargetCapsuleError",
    "Lfm2MappedCausalTargetEvidence",
    "Lfm2MappedCausalTargetReference",
    "authenticate_lfm2_mapped_causal_target_capsule",
    "build_lfm2_mapped_causal_target_evidence",
    "collect_and_publish_lfm2_mapped_causal_target_capsule",
    "collect_lfm2_mapped_causal_target_evidence",
    "load_lfm2_mapped_causal_target_capsule",
    "publish_lfm2_mapped_causal_target_capsule",
]
