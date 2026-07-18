"""Reusable whole-model LFM2 CPCF compiler.

The public operation in this module owns the complete in-memory fold after a
dense target and aligned cached phases have been created.  It fits the tied
interface and independent boundary coordinates, installs every macrocell in
strict order on actual folded-upstream states, fits the final target-coordinate
RMSNorm, and emits a sealed parameter-count receipt.

This module is intentionally independent of experiment scripts, checkpoint
paths, model hubs, and parent execution.  It can compile any representable
parameter budget.  A caller may attach a promotion ratio (for example 100x)
and optionally require it, but lower-ratio diagnostic controls remain valid
fold requests and are explicitly marked non-promotable rather than being
misrepresented as the project goal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Callable

from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_closed_form_causal_circuit_fold import (
    fit_rmsnorm_diagonal,
)
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    Lfm2CpcfBoundaryBundle,
    fit_lfm2_cpcf_boundary_bundle,
)
from neural_forge.hf_lfm2_cpcf_layer_compiler import (
    LFM2_CPCF_LAYER_COMPILER_SCHEMA,
    LFM2_CPCF_LAYER_FAILURE_SCHEMA,
    Lfm2CpcfCompiledLayerResult,
    Lfm2CpcfLayerCompilationFailure,
    compile_and_install_lfm2_cpcf_layer,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
    install_lfm2_cpcf_neutral_norms,
    install_lfm2_cpcf_tied_interface,
    seed_lfm2_cpcf_phase_states,
)
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CpcfShortConvGatePolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
)


LFM2_CPCF_MODEL_FOLD_SCHEMA = "neural-forge-lfm2-cpcf-model-fold-v1"
LFM2_CPCF_MODEL_FAILURE_SCHEMA = "neural-forge-lfm2-cpcf-model-failure-v3"
LFM2_CPCF_MODEL_PROGRESS_EVENT_SCHEMA = "neural-forge-lfm2-cpcf-model-progress-event-v1"
LFM2_CPCF_FAILURE_INCOMING_BANK_SNAPSHOT_SCHEMA = (
    "neural-forge-lfm2-cpcf-failure-incoming-bank-snapshot-v1"
)
LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA = "neural-forge-lfm2-cpcf-parameter-receipt-v1"
LFM2_CPCF_PREFIX_INSTALL_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-prefix-install-v1"
LFM2_CPCF_TARGET_TOPOLOGY_SNAPSHOT_SCHEMA = (
    "neural-forge-lfm2-cpcf-target-topology-snapshot-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_TRANSACTION_STAGES = (
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
_COMPLETED_LAYER_ZERO_EXECUTION_FIELDS = frozenset(
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
_PREFIX_INSTALL_REPORT_FIELDS = frozenset(
    {
        "schema",
        "checkpoint_reference",
        "target_topology_sha256",
        "blank_state_sha256_before",
        "installed_state_sha256_after",
        "tensor_count",
        "next_layer_index",
        "exact_inventory_verified",
        "exact_object_aliases_verified",
        "exact_storage_aliases_verified",
        "exact_shape_dtype_stride_offset_device_verified",
        "tensor_identities_preserved",
        "exact_prefix_tensor_bytes_installed",
        "loaders_used",
        "parent_evidence_tensors_copied",
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
        "promotion_claimed",
        "report_sha256",
    }
)
_MODEL_PROGRESS_EVENT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "next_layer_index",
        "full_policy_sha256",
        "boundary_bundle_sha256",
        "boundary_bundle_inventory_sha256",
        "incoming_state_bank_sha256",
        "incoming_state_snapshot_sha256",
        "completed_layer_count",
        "completed_layer_result_sha256",
        "target_prefix_snapshot_receipt",
        "target_prefix_tensor_count",
        "parent_evidence_tensors_copied",
        "target_prefix_tensors_owned_cpu",
        "incoming_child_state_tensors_owned_cpu",
        "live_target_model_exposed",
        "checkpoint_publication_claimed",
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


class Lfm2CpcfModelFoldError(ValueError):
    """The requested complete CPCF model fold is invalid."""


class Lfm2CpcfModelFoldInstallError(RuntimeError):
    """A complete model fold could not be installed honestly."""


@dataclass(frozen=True, slots=True)
class Lfm2CpcfFailureIncomingBankSnapshot:
    """Owned child states plus immutable references to external parent evidence.

    Parent evidence can be tens of gigabytes and already has cache, phase, and
    content seals.  A failure therefore owns only the mutable child residuals
    needed to inspect the failed prefix.  Parent evidence and target-token
    phases are represented by authenticated scalar references, never by live
    tensor aliases or duplicate tensor archives.
    """

    _metadata_bytes: bytes
    _state_items: tuple[tuple[tuple[int, str], Any], ...]

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def capture(
        cls,
        value: Lfm2CpcfPhaseStateBank,
    ) -> Lfm2CpcfFailureIncomingBankSnapshot:
        if not isinstance(value, Lfm2CpcfPhaseStateBank):
            raise Lfm2CpcfModelFoldError(
                "Incoming failure evidence must be a typed phase-state bank."
            )
        phase_references = []
        state_inventory = []
        state_items = []
        runtime_inventory = []
        for phase in value.phases:
            evidence = phase.parent_evidence
            identity = phase.identity
            state = (
                value.states[identity].detach().to(device="cpu").contiguous().clone()
            )
            state_sha = _tensor_sha256(state)
            state_items.append((identity, state))
            state_inventory.append(
                {
                    "identity": [identity[0], identity[1]],
                    "dtype": str(state.dtype).removeprefix("torch."),
                    "shape": [int(size) for size in state.shape],
                    "state_sha256": state_sha,
                }
            )
            runtime_inventory.append(
                {
                    "identity": [identity[0], identity[1]],
                    "phase_sha256": phase.phase_sha256,
                    "state_sha256": state_sha,
                }
            )
            phase_references.append(
                {
                    "identity": [identity[0], identity[1]],
                    "record_id": phase.record_id,
                    "split": phase.split,
                    "donor_input_sha256": phase.donor_input_sha256,
                    "target_alignment_sha256": (phase.target_alignment_sha256),
                    "phase_sha256": phase.phase_sha256,
                    "sampled_suffix_tokens": phase.sampled_suffix_tokens,
                    "input_ids_sha256": _tensor_sha256(phase.input_ids),
                    "attention_mask_sha256": _tensor_sha256(phase.attention_mask),
                    "position_ids_sha256": _tensor_sha256(phase.position_ids),
                    "parent_evidence_reference": {
                        "cache_key": evidence.cache_key,
                        "batch_index": evidence.batch_index,
                        "input_sha256": evidence.input_sha256,
                        "phase": evidence.phase,
                        "evidence_sha256": evidence.evidence_sha256,
                    },
                }
            )
        runtime_body = {
            "schema": value.as_dict()["schema"],
            "scope": "phase_state_bank",
            "boundary": value.boundary,
            "layer_index": value.layer_index,
            "hidden_size": value.hidden_size,
            "ordered_states": runtime_inventory,
            "actual_folded_upstream_states": True,
        }
        if _sha(runtime_body) != value.state_bank_sha256:
            raise Lfm2CpcfModelFoldInstallError(
                "Incoming state bank changed after its runtime seal."
            )
        body = {
            "schema": LFM2_CPCF_FAILURE_INCOMING_BANK_SNAPSHOT_SCHEMA,
            "boundary": value.boundary,
            "layer_index": value.layer_index,
            "hidden_size": value.hidden_size,
            "phase_count": len(phase_references),
            "phase_references": phase_references,
            "state_inventory": state_inventory,
            "state_bank_sha256": value.state_bank_sha256,
            "parent_evidence_storage": (
                "external_cache_key_and_content_digest_reference"
            ),
            "parent_evidence_tensors_copied": False,
            "target_phase_tensors_copied": False,
            "child_state_tensors_owned": True,
        }
        metadata = {**body, "snapshot_sha256": _sha(body)}
        return cls(
            _metadata_bytes=_canonical_json(metadata),
            _state_items=tuple(state_items),
        )

    def _metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self._metadata_bytes.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2CpcfModelFoldError(
                "Incoming failure snapshot metadata is not canonical JSON."
            ) from exc
        if _canonical_json(value) != self._metadata_bytes:
            raise Lfm2CpcfModelFoldError(
                "Incoming failure snapshot metadata is not canonical."
            )
        return value

    def verify(self) -> None:
        metadata = self._metadata()
        expected_fields = {
            "schema",
            "boundary",
            "layer_index",
            "hidden_size",
            "phase_count",
            "phase_references",
            "state_inventory",
            "state_bank_sha256",
            "parent_evidence_storage",
            "parent_evidence_tensors_copied",
            "target_phase_tensors_copied",
            "child_state_tensors_owned",
            "snapshot_sha256",
        }
        body = dict(metadata)
        claimed_snapshot_sha = body.pop("snapshot_sha256", None)
        if (
            set(metadata) != expected_fields
            or metadata.get("schema") != LFM2_CPCF_FAILURE_INCOMING_BANK_SNAPSHOT_SCHEMA
            or _sha(body) != claimed_snapshot_sha
            or metadata.get("parent_evidence_storage")
            != "external_cache_key_and_content_digest_reference"
            or metadata.get("parent_evidence_tensors_copied") is not False
            or metadata.get("target_phase_tensors_copied") is not False
            or metadata.get("child_state_tensors_owned") is not True
        ):
            raise Lfm2CpcfModelFoldError(
                "Incoming failure snapshot metadata or seal is invalid."
            )
        boundary = metadata.get("boundary")
        layer_index = metadata.get("layer_index")
        hidden_size = metadata.get("hidden_size")
        phase_count = metadata.get("phase_count")
        phase_references = metadata.get("phase_references")
        state_inventory = metadata.get("state_inventory")
        if (
            not isinstance(boundary, str)
            or not boundary
            or isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
            or isinstance(phase_count, bool)
            or not isinstance(phase_count, int)
            or phase_count <= 0
            or not isinstance(phase_references, list)
            or not isinstance(state_inventory, list)
            or len(phase_references) != phase_count
            or len(state_inventory) != phase_count
            or len(self._state_items) != phase_count
        ):
            raise Lfm2CpcfModelFoldError(
                "Incoming failure snapshot geometry is invalid."
            )
        _digest(
            metadata.get("state_bank_sha256"),
            "incoming failure snapshot state_bank_sha256",
        )
        runtime_inventory = []
        seen = set()
        torch = _torch()
        for offset, ((identity, state), phase, inventory) in enumerate(
            zip(
                self._state_items,
                phase_references,
                state_inventory,
            )
        ):
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or isinstance(identity[0], bool)
                or not isinstance(identity[0], int)
                or identity[0] < 0
                or not isinstance(identity[1], str)
                or not identity[1]
                or identity in seen
                or not isinstance(phase, dict)
                or not isinstance(inventory, dict)
                or phase.get("identity") != [identity[0], identity[1]]
                or inventory.get("identity") != [identity[0], identity[1]]
            ):
                raise Lfm2CpcfModelFoldError(
                    f"Incoming failure snapshot phase {offset} is invalid."
                )
            seen.add(identity)
            expected_phase_fields = {
                "identity",
                "record_id",
                "split",
                "donor_input_sha256",
                "target_alignment_sha256",
                "phase_sha256",
                "sampled_suffix_tokens",
                "input_ids_sha256",
                "attention_mask_sha256",
                "position_ids_sha256",
                "parent_evidence_reference",
            }
            reference = phase.get("parent_evidence_reference")
            if (
                set(phase) != expected_phase_fields
                or not isinstance(phase.get("record_id"), str)
                or not phase["record_id"]
                or not isinstance(phase.get("split"), str)
                or not phase["split"]
                or isinstance(phase.get("sampled_suffix_tokens"), bool)
                or not isinstance(phase.get("sampled_suffix_tokens"), int)
                or phase["sampled_suffix_tokens"] <= 0
                or not isinstance(reference, dict)
                or set(reference)
                != {
                    "cache_key",
                    "batch_index",
                    "input_sha256",
                    "phase",
                    "evidence_sha256",
                }
                or reference.get("batch_index") != identity[0]
                or reference.get("phase") != identity[1]
                or reference.get("input_sha256") != phase.get("donor_input_sha256")
            ):
                raise Lfm2CpcfModelFoldError(
                    "Incoming failure snapshot phase reference is invalid."
                )
            for name in (
                "donor_input_sha256",
                "target_alignment_sha256",
                "phase_sha256",
                "input_ids_sha256",
                "attention_mask_sha256",
                "position_ids_sha256",
            ):
                _digest(
                    phase.get(name),
                    f"incoming failure phase {offset} {name}",
                )
            for name in (
                "cache_key",
                "input_sha256",
                "evidence_sha256",
            ):
                _digest(
                    reference.get(name),
                    f"incoming failure parent reference {offset} {name}",
                )
            if (
                set(inventory) != {"identity", "dtype", "shape", "state_sha256"}
                or not isinstance(state, torch.Tensor)
                or state.device.type != "cpu"
                or not state.is_contiguous()
                or state.requires_grad
                or state.grad_fn is not None
                or state.ndim != 3
                or int(state.shape[-1]) != hidden_size
                or inventory.get("dtype") != str(state.dtype).removeprefix("torch.")
                or inventory.get("shape") != [int(size) for size in state.shape]
                or inventory.get("state_sha256") != _tensor_sha256(state)
            ):
                raise Lfm2CpcfModelFoldError(
                    "Incoming failure child-state snapshot is invalid."
                )
            runtime_inventory.append(
                {
                    "identity": [identity[0], identity[1]],
                    "phase_sha256": phase["phase_sha256"],
                    "state_sha256": inventory["state_sha256"],
                }
            )
        runtime_body = {
            "schema": LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
            "scope": "phase_state_bank",
            "boundary": boundary,
            "layer_index": layer_index,
            "hidden_size": hidden_size,
            "ordered_states": runtime_inventory,
            "actual_folded_upstream_states": True,
        }
        if _sha(runtime_body) != metadata["state_bank_sha256"]:
            raise Lfm2CpcfModelFoldError(
                "Incoming failure child states differ from the bank seal."
            )

    @property
    def boundary(self) -> str:
        return str(self._metadata()["boundary"])

    @property
    def layer_index(self) -> int:
        return int(self._metadata()["layer_index"])

    @property
    def hidden_size(self) -> int:
        return int(self._metadata()["hidden_size"])

    @property
    def state_bank_sha256(self) -> str:
        return str(self._metadata()["state_bank_sha256"])

    @property
    def snapshot_sha256(self) -> str:
        return str(self._metadata()["snapshot_sha256"])

    @property
    def phase_identities(self) -> tuple[tuple[int, str], ...]:
        return tuple(identity for identity, _state in self._state_items)

    @property
    def phase_references(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            _freeze_evidence_metadata(value)
            for value in self._metadata()["phase_references"]
        )

    def state(self, identity: tuple[int, str]) -> Any:
        for observed, value in self._state_items:
            if observed == identity:
                return value.clone()
        raise KeyError(identity)

    def as_dict(self) -> Mapping[str, Any]:
        return _freeze_evidence_metadata(self._metadata())


class Lfm2CpcfModelFoldFailure(Lfm2CpcfModelFoldInstallError):
    """Structured non-promotable failure with proven whole-model rollback."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        boundary_bundle: Lfm2CpcfBoundaryBundle | None,
        current_incoming_bank: (
            Lfm2CpcfPhaseStateBank | Lfm2CpcfFailureIncomingBankSnapshot | None
        ),
        completed_layer_receipts: Sequence[Mapping[str, Any]],
        prefix_target_cpu_tensor_snapshot: Mapping[str, Any],
        prefix_target_snapshot_metadata: Mapping[str, Any],
        rollback_proof: Mapping[str, Any],
        report: Mapping[str, Any],
        layer_failure_report: Mapping[str, Any] | None = None,
        layer_failure_report_bytes: bytes | None = None,
    ) -> None:
        super().__init__(message)
        if stage not in _MODEL_TRANSACTION_STAGES:
            raise Lfm2CpcfModelFoldError(
                "Model failure stage is not a strict transaction stage."
            )
        proof = dict(rollback_proof)
        proof_sha = _digest(
            proof.pop("proof_sha256", None),
            "rollback_proof.proof_sha256",
        )
        proof_with_seal = {**proof, "proof_sha256": proof_sha}
        for name in (
            "source_state_fingerprint_before",
            "source_state_fingerprint_after",
            "source_full_inventory_sha256_before",
            "source_full_inventory_sha256_after",
            "target_state_fingerprint_before",
            "target_state_fingerprint_after",
            "target_full_inventory_sha256_before",
            "target_full_inventory_sha256_after",
        ):
            _digest(proof.get(name), f"rollback_proof.{name}")
        true_proof_fields = (
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
        )
        if (
            set(proof_with_seal) != _MODEL_ROLLBACK_PROOF_FIELDS
            or _sha(proof) != proof_sha
            or any(proof.get(name) is not True for name in true_proof_fields)
            or proof.get("source_state_fingerprint_before")
            != proof.get("source_state_fingerprint_after")
            or proof.get("source_full_inventory_sha256_before")
            != proof.get("source_full_inventory_sha256_after")
            or proof.get("target_state_fingerprint_before")
            != proof.get("target_state_fingerprint_after")
            or proof.get("target_full_inventory_sha256_before")
            != proof.get("target_full_inventory_sha256_after")
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Model failure lacks an exact whole-model rollback proof."
            )
        failure_report = dict(report)
        report_sha = _digest(
            failure_report.pop("report_sha256", None),
            "failure_report.report_sha256",
        )
        report_with_seal = {
            **failure_report,
            "report_sha256": report_sha,
        }
        if (
            set(report_with_seal) != _MODEL_FAILURE_REPORT_FIELDS
            or _sha(failure_report) != report_sha
            or failure_report.get("schema") != LFM2_CPCF_MODEL_FAILURE_SCHEMA
            or failure_report.get("stage") != stage
            or failure_report.get("promotion_claimed") is not False
            or failure_report.get("compiled_model_fold_issued") is not False
            or failure_report.get("parameter_compaction_success_receipt_issued")
            is not False
            or failure_report.get("promotable_candidate") is not False
            or failure_report.get("failure_evidence_only") is not True
            or failure_report.get("rollback_proof") != proof_with_seal
        ):
            raise Lfm2CpcfModelFoldError(
                "Model failure report is invalid or claims success."
            )
        cause = failure_report.get("cause")
        if (
            not isinstance(cause, Mapping)
            or set(cause) != {"module", "type", "message"}
            or any(
                not isinstance(cause.get(name), str)
                for name in ("module", "type", "message")
            )
        ):
            raise Lfm2CpcfModelFoldError("Model failure cause is not canonical.")
        nested = layer_failure_report
        nested_bytes = layer_failure_report_bytes
        if (nested is None) != (nested_bytes is None):
            raise Lfm2CpcfModelFoldError(
                "Layer failure report and bytes must travel together."
            )
        if nested is not None and _canonical_json(dict(nested)) != nested_bytes:
            raise Lfm2CpcfModelFoldError("Nested layer failure report bytes changed.")
        if nested is not None:
            nested_body = dict(nested)
            nested_sha = _digest(
                nested_body.pop("report_sha256", None),
                "layer_failure_report.report_sha256",
            )
            if (
                _sha(nested_body) != nested_sha
                or failure_report.get("nested_layer_failure_report") != dict(nested)
                or failure_report.get("nested_layer_failure_report_sha256")
                != nested_sha
                or failure_report.get(
                    "nested_layer_failure_report_canonical_bytes_sha256"
                )
                != hashlib.sha256(nested_bytes).hexdigest()
                or failure_report.get(
                    "nested_layer_failure_report_preserved_byte_for_byte"
                )
                is not True
            ):
                raise Lfm2CpcfModelFoldError(
                    "Nested layer failure evidence is not preserved and self-sealed."
                )
        elif (
            failure_report.get("nested_layer_failure_report") is not None
            or failure_report.get("nested_layer_failure_report_sha256") is not None
            or failure_report.get("nested_layer_failure_report_canonical_bytes_sha256")
            is not None
            or failure_report.get("nested_layer_failure_report_preserved_byte_for_byte")
            is not False
        ):
            raise Lfm2CpcfModelFoldError(
                "Absent layer failure evidence is inconsistent."
            )
        expected_boundary_sha = (
            None
            if boundary_bundle is None
            else getattr(boundary_bundle, "bundle_sha256", None)
        )
        expected_incoming_sha = (
            None
            if current_incoming_bank is None
            else getattr(
                current_incoming_bank,
                "state_bank_sha256",
                None,
            )
        )
        expected_incoming_index = (
            None
            if current_incoming_bank is None
            else getattr(current_incoming_bank, "layer_index", None)
        )
        if (
            failure_report.get("boundary_bundle_sha256") != expected_boundary_sha
            or failure_report.get("current_incoming_bank_sha256")
            != expected_incoming_sha
            or failure_report.get("current_incoming_layer_index")
            != expected_incoming_index
        ):
            raise Lfm2CpcfModelFoldError(
                "Model failure boundary or incoming-bank evidence differs "
                "from its typed context."
            )
        active_index = failure_report.get("active_layer_index")
        active_kind = failure_report.get("active_operator_kind")
        if stage == "layer_compile_install":
            if (
                isinstance(active_index, bool)
                or not isinstance(active_index, int)
                or active_index < 0
                or active_kind not in {"full_attention", "short_conv"}
                or boundary_bundle is None
                or current_incoming_bank is None
                or expected_incoming_index != active_index
            ):
                raise Lfm2CpcfModelFoldError(
                    "Active layer failure context is incomplete."
                )
        elif active_index is not None or active_kind is not None:
            raise Lfm2CpcfModelFoldError(
                "Inactive model failure carries active-layer context."
            )
        torch = _torch()
        snapshot = {}
        for name, tensor in prefix_target_cpu_tensor_snapshot.items():
            if (
                not isinstance(name, str)
                or not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or not tensor.is_contiguous()
            ):
                raise Lfm2CpcfModelFoldError(
                    "Prefix target snapshot must contain named contiguous CPU tensors."
                )
            snapshot[name] = tensor
        if not snapshot:
            raise Lfm2CpcfModelFoldError("Prefix target snapshot cannot be empty.")
        receipts = tuple(
            MappingProxyType(dict(receipt)) for receipt in completed_layer_receipts
        )
        for expected_index, receipt in enumerate(receipts):
            receipt_body = dict(receipt)
            result_sha = _digest(
                receipt_body.pop("result_sha256", None),
                "completed_layer_receipt.result_sha256",
            )
            if (
                _sha(receipt_body) != result_sha
                or receipt.get("layer_index") != expected_index
            ):
                raise Lfm2CpcfModelFoldError(
                    "Completed layer failure receipt is unordered or not self-sealed."
                )
        if failure_report.get("completed_layer_count") != len(
            receipts
        ) or failure_report.get("completed_layer_receipts") != [
            dict(receipt) for receipt in receipts
        ]:
            raise Lfm2CpcfModelFoldError(
                "Model failure report does not preserve completed layer receipts."
            )
        if stage == "layer_compile_install" and len(receipts) != active_index:
            raise Lfm2CpcfModelFoldError(
                "Completed prefix length differs from the active layer."
            )
        if nested is not None:
            nested_report = dict(nested)
            if (
                nested_report.get("schema") != LFM2_CPCF_LAYER_FAILURE_SCHEMA
                or nested_report.get("layer_index") != active_index
                or nested_report.get("operator_kind") != active_kind
                or nested_report.get("input_bank_sha256") != expected_incoming_sha
            ):
                raise Lfm2CpcfModelFoldError(
                    "Nested layer failure differs from active model context."
                )
            try:
                active_charts = boundary_bundle.layers[active_index]
            except (AttributeError, IndexError, TypeError) as exc:
                raise Lfm2CpcfModelFoldError(
                    "Active boundary charts are unavailable."
                ) from exc
            if nested_report.get("boundary_charts_sha256") != getattr(
                active_charts,
                "layer_charts_sha256",
                None,
            ):
                raise Lfm2CpcfModelFoldError(
                    "Nested layer failure differs from active boundary charts."
                )
        snapshot_report = failure_report.get("failed_prefix_target_cpu_tensor_snapshot")
        if not isinstance(snapshot_report, Mapping):
            raise Lfm2CpcfModelFoldError(
                "Model failure lacks a prefix target snapshot receipt."
            )
        snapshot_metadata = _canonical_mapping_copy(
            prefix_target_snapshot_metadata,
            "prefix_target_snapshot_metadata",
        )
        observed_snapshot_report = _cpu_tensor_snapshot_receipt_from_parts(
            snapshot,
            snapshot_metadata,
        )
        if dict(snapshot_report) != dict(observed_snapshot_report):
            raise Lfm2CpcfModelFoldError(
                "Prefix target CPU tensor snapshot receipt is invalid."
            )
        self.stage = stage
        self._boundary_bundle = _clone_boundary_bundle_evidence(boundary_bundle)
        self._current_incoming_bank = _snapshot_incoming_bank_evidence(
            current_incoming_bank
        )
        self._completed_layer_receipt_bytes = tuple(
            _canonical_json(dict(receipt)) for receipt in receipts
        )
        self._prefix_target_cpu_tensor_snapshot = MappingProxyType(
            {
                name: tensor.detach().to(device="cpu").contiguous().clone()
                for name, tensor in snapshot.items()
            }
        )
        self._prefix_target_snapshot_metadata_bytes = _canonical_json(snapshot_metadata)
        self._rollback_proof_bytes = _canonical_json(proof_with_seal)
        self._layer_failure_report_bytes = nested_bytes
        self._report_bytes = _canonical_json(report_with_seal)

    @property
    def boundary_bundle(self) -> Lfm2CpcfBoundaryBundle | None:
        return _clone_boundary_bundle_evidence(self._boundary_bundle)

    @property
    def current_incoming_bank(
        self,
    ) -> Lfm2CpcfFailureIncomingBankSnapshot | None:
        if self._current_incoming_bank is not None:
            self._current_incoming_bank.verify()
        return self._current_incoming_bank

    @property
    def completed_layer_receipts(
        self,
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            json.loads(payload.decode("ascii"))
            for payload in self._completed_layer_receipt_bytes
        )

    @property
    def prefix_target_cpu_tensor_snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                name: tensor.clone()
                for name, tensor in (self._prefix_target_cpu_tensor_snapshot.items())
            }
        )

    @property
    def prefix_target_snapshot_metadata(self) -> Mapping[str, Any]:
        return json.loads(self._prefix_target_snapshot_metadata_bytes.decode("ascii"))

    @property
    def rollback_proof(self) -> Mapping[str, Any]:
        return json.loads(self._rollback_proof_bytes.decode("ascii"))

    @property
    def layer_failure_report(self) -> Mapping[str, Any] | None:
        if self._layer_failure_report_bytes is None:
            return None
        return json.loads(self._layer_failure_report_bytes.decode("ascii"))

    @property
    def layer_failure_report_bytes(self) -> bytes | None:
        return self._layer_failure_report_bytes

    @property
    def report(self) -> Mapping[str, Any]:
        return json.loads(self._report_bytes.decode("ascii"))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for LFM2 CPCF model folding.") from exc
    return torch


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
        raise Lfm2CpcfModelFoldError(
            "Model-fold metadata must be finite canonical JSON."
        ) from exc


def _canonical_mapping_copy(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfModelFoldError(f"{name} must be a canonical JSON mapping.")
    try:
        normalized = json.loads(_canonical_json(dict(value)).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfModelFoldError(
            f"{name} must be a canonical JSON mapping."
        ) from exc
    if not isinstance(normalized, dict):
        raise Lfm2CpcfModelFoldError(f"{name} must be a canonical JSON mapping.")
    return normalized


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfModelFoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfModelFoldError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfModelFoldError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfModelFoldError(f"{name} must be finite and nonnegative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfModelFoldError(f"{name} must be finite and nonnegative.")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result > 1.0:
        raise Lfm2CpcfModelFoldError(f"{name} must be within [0, 1].")
    return result


def _model_core(model: Any, name: str) -> Any:
    config = getattr(model, "config", None)
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if getattr(config, "model_type", None) != "lfm2" or core is None or layers is None:
        raise Lfm2CpcfModelFoldError(
            f"{name} must be a materialized LFM2 causal language model."
        )
    return core


def independent_parameter_inventory(model: Any) -> Mapping[str, Any]:
    """Count learned scalars by materialized storage identity.

    This deliberately does not trust a topology planner, config declaration,
    checkpoint byte size, quantization metadata, or pruning mask.
    """

    try:
        named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older Torch compatibility
        named = tuple(model.named_parameters())
    groups: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    raw_numel = 0
    for name, parameter in named:
        if parameter.device.type == "meta":
            raise Lfm2CpcfModelFoldError(f"Parameter {name!r} is not materialized.")
        raw_numel += int(parameter.numel())
        key = (
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.storage_offset()),
            int(parameter.numel()),
            int(parameter.element_size()),
        )
        group = groups.setdefault(
            key,
            {
                "numel": int(parameter.numel()),
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "names": [],
            },
        )
        group["names"].append(str(name))
    if not groups:
        raise Lfm2CpcfModelFoldError(
            "Model exposes no materialized learned parameters."
        )
    unique_numel = sum(int(group["numel"]) for group in groups.values())
    iterator_numel = sum(int(parameter.numel()) for parameter in model.parameters())
    aliases = sorted(
        (
            {
                "names": sorted(group["names"]),
                "numel": group["numel"],
                "shape": group["shape"],
                "dtype": group["dtype"],
            }
            for group in groups.values()
            if len(group["names"]) > 1
        ),
        key=lambda value: value["names"],
    )
    body = {
        "method": ("unique_materialized_parameter_storage_offset_numel_identity"),
        "unique_parameters": unique_numel,
        "raw_named_parameter_occurrences": len(named),
        "raw_named_parameter_numel": raw_numel,
        "parameter_iterator_unique_numel": iterator_numel,
        "independent_methods_agree": unique_numel == iterator_numel,
        "alias_group_count": len(aliases),
        "alias_groups": aliases,
        "checkpoint_byte_size_used": False,
        "quantization_adjustment_used": False,
        "pruning_mask_adjustment_used": False,
    }
    if not body["independent_methods_agree"]:
        raise Lfm2CpcfModelFoldError("Independent parameter-count methods disagree.")
    return MappingProxyType({**body, "inventory_sha256": _sha(body)})


def audit_lfm2_cpcf_parameter_compaction(
    source_model: Any,
    target_model: Any,
    *,
    planned_target_unique_parameters: int,
    promotion_compression_ratio: float = 1.0,
) -> Mapping[str, Any]:
    """Measure real parameter compaction without restricting valid budgets."""

    planned = _positive_int(
        planned_target_unique_parameters,
        "planned_target_unique_parameters",
    )
    promotion_floor = _nonnegative_float(
        promotion_compression_ratio,
        "promotion_compression_ratio",
    )
    if promotion_floor <= 0.0:
        raise Lfm2CpcfModelFoldError("promotion_compression_ratio must be positive.")
    source = independent_parameter_inventory(source_model)
    target = independent_parameter_inventory(target_model)
    source_count = int(source["unique_parameters"])
    target_count = int(target["unique_parameters"])
    if target_count != planned:
        raise Lfm2CpcfModelFoldError(
            "Independent target count differs from the catalog plan."
        )
    if not 0 < target_count < source_count:
        raise Lfm2CpcfModelFoldError(
            "Target topology does not compact the source parameter count."
        )
    ratio = source_count / target_count
    body = {
        "schema": LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
        "source_inventory": dict(source),
        "target_inventory": dict(target),
        "source_unique_parameters": source_count,
        "target_unique_parameters": target_count,
        "planned_target_unique_parameters": planned,
        "parameters_removed_from_topology": source_count - target_count,
        "realized_compression_ratio": ratio,
        "promotion_compression_ratio": promotion_floor,
        "promotion_ratio_met": ratio >= promotion_floor,
        "valid_lower_ratio_diagnostic_fold": ratio < promotion_floor,
        "parameter_count_compaction_only": True,
        "checkpoint_byte_ratio_used": False,
        "quantization_used": False,
        "pruning_used": False,
    }
    return MappingProxyType({**body, "receipt_sha256": _sha(body)})


def _blank_receipt(
    value: Mapping[str, Any],
    *,
    target_unique_parameters: int,
) -> Mapping[str, Any]:
    required_true = (
        "compiler_created_blank",
        "all_parameters_zero",
        "dense_materialized_parameters",
        "topology_provenance_verified",
    )
    if (
        not isinstance(value, Mapping)
        or any(value.get(name) is not True for name in required_true)
        or value.get("weights_installed") is not False
        or value.get("training_used") is not False
        or value.get("quantization_used") is not False
        or value.get("pruning_used") is not False
        or int(value.get("target_unique_parameters", -1)) != target_unique_parameters
    ):
        raise Lfm2CpcfModelFoldError(
            "Target lacks an authenticated compiler-created dense blank receipt."
        )
    report = dict(value)
    report_sha = _digest(
        report.pop("report_sha256", None),
        "blank_target_report.report_sha256",
    )
    if _sha(report) != report_sha:
        raise Lfm2CpcfModelFoldError("Blank-target report seal is invalid.")
    return MappingProxyType({**report, "report_sha256": report_sha})


def _assert_target_is_zero_blank(target_model: Any) -> None:
    torch = _torch()
    for name, parameter in target_model.named_parameters():
        if not bool(torch.count_nonzero(parameter.detach()).item() == 0):
            raise Lfm2CpcfModelFoldError(
                f"Target parameter {name!r} is not blank before compilation."
            )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCompletedLayerReceipt:
    """A complete authenticated layer result without archived replay tensors."""

    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        report = _canonical_mapping_copy(
            self.report,
            "completed_layer_receipt",
        )
        body = dict(report)
        result_sha = _digest(
            body.pop("result_sha256", None),
            "completed_layer_receipt.result_sha256",
        )
        layer_index = report.get("layer_index")
        execution = report.get("execution")
        if (
            set(report) != _COMPLETED_LAYER_RECEIPT_FIELDS
            or _sha(body) != result_sha
            or report.get("schema") != LFM2_CPCF_LAYER_COMPILER_SCHEMA
            or isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or report.get("operator_kind") not in {"full_attention", "short_conv"}
        ):
            raise Lfm2CpcfModelFoldError(
                "Completed layer receipt is not exact, ordered, or self-sealed."
            )
        for name in (
            "source_width",
            "target_width",
        ):
            _positive_int(report.get(name), f"completed_layer_receipt.{name}")
        for name in (
            "input_bank_sha256",
            "next_bank_sha256",
            "boundary_charts_sha256",
            "preinstall_replay_sha256",
            "operator_replay_sha256",
            "final_replay_sha256",
        ):
            _digest(report.get(name), f"completed_layer_receipt.{name}")
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
            if not isinstance(report.get(name), dict):
                raise Lfm2CpcfModelFoldError(
                    f"Completed layer receipt {name!r} is not canonical."
                )
        if (
            not isinstance(execution, dict)
            or set(execution)
            != (_COMPLETED_LAYER_ZERO_EXECUTION_FIELDS | {"closed_form_compilers"})
            or execution.get("closed_form_compilers") is not True
            or any(
                type(execution.get(name)) is not int or execution[name] != 0
                for name in _COMPLETED_LAYER_ZERO_EXECUTION_FIELDS
            )
        ):
            raise Lfm2CpcfModelFoldError(
                "Completed layer receipt does not prove zero-training execution."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(report),
        )

    @property
    def layer_index(self) -> int:
        return int(self.report["layer_index"])

    @property
    def operator_kind(self) -> str:
        return str(self.report["operator_kind"])

    @property
    def input_bank_sha256(self) -> str:
        return str(self.report["input_bank_sha256"])

    @property
    def next_bank_sha256(self) -> str:
        return str(self.report["next_bank_sha256"])

    @property
    def result_sha256(self) -> str:
        return str(self.report["result_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self.report)).decode("ascii"))


@dataclass(frozen=True, slots=True)
class Lfm2CpcfModelFoldResumeState:
    """Caller-restored, authenticated state for one in-memory continuation."""

    prefix_install_report: Mapping[str, Any]
    full_policy_sha256: str
    boundary_bundle: Lfm2CpcfBoundaryBundle
    incoming_state_bank: Lfm2CpcfPhaseStateBank
    completed_layer_receipts: Sequence[
        Mapping[str, Any] | Lfm2CpcfCompletedLayerReceipt
    ]

    def __post_init__(self) -> None:
        install = _canonical_mapping_copy(
            self.prefix_install_report,
            "prefix_install_report",
        )
        install_body = dict(install)
        install_sha = _digest(
            install_body.pop("report_sha256", None),
            "prefix_install_report.report_sha256",
        )
        reference = install.get("checkpoint_reference")
        next_layer = install.get("next_layer_index")
        tensor_count = install.get("tensor_count")
        if (
            set(install) != _PREFIX_INSTALL_REPORT_FIELDS
            or _sha(install_body) != install_sha
            or install.get("schema") != LFM2_CPCF_PREFIX_INSTALL_REPORT_SCHEMA
            or not isinstance(reference, dict)
            or set(reference) != {"artifact_kind", "artifact_key", "manifest_sha256"}
            or reference.get("artifact_kind") != "checkpoint"
            or isinstance(next_layer, bool)
            or not isinstance(next_layer, int)
            or next_layer < 0
            or isinstance(tensor_count, bool)
            or not isinstance(tensor_count, int)
            or tensor_count <= 0
        ):
            raise Lfm2CpcfModelFoldError(
                "Prefix install report is not the exact authenticated bridge schema."
            )
        for name in (
            "artifact_key",
            "manifest_sha256",
        ):
            _digest(reference.get(name), f"checkpoint_reference.{name}")
        for name in (
            "target_topology_sha256",
            "blank_state_sha256_before",
            "installed_state_sha256_after",
        ):
            _digest(install.get(name), f"prefix_install_report.{name}")
        for name in (
            "exact_inventory_verified",
            "exact_object_aliases_verified",
            "exact_storage_aliases_verified",
            "exact_shape_dtype_stride_offset_device_verified",
            "tensor_identities_preserved",
            "exact_prefix_tensor_bytes_installed",
        ):
            if install.get(name) is not True:
                raise Lfm2CpcfModelFoldError(
                    "Prefix install report does not prove every exact install invariant."
                )
        for name in (
            "loaders_used",
            "parent_evidence_tensors_copied",
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
        ):
            if type(install.get(name)) is not int or install[name] != 0:
                raise Lfm2CpcfModelFoldError(
                    "Prefix install report contains a prohibited execution claim."
                )
        if install.get("promotion_claimed") is not False:
            raise Lfm2CpcfModelFoldError(
                "Prefix install report cannot claim promotion."
            )
        policy_sha = _digest(
            self.full_policy_sha256,
            "full_policy_sha256",
        )
        if not isinstance(self.boundary_bundle, Lfm2CpcfBoundaryBundle):
            raise TypeError("boundary_bundle must be Lfm2CpcfBoundaryBundle.")
        if not isinstance(self.incoming_state_bank, Lfm2CpcfPhaseStateBank):
            raise TypeError("incoming_state_bank must be Lfm2CpcfPhaseStateBank.")
        raw_receipts = tuple(self.completed_layer_receipts)
        receipts = tuple(
            value
            if isinstance(value, Lfm2CpcfCompletedLayerReceipt)
            else Lfm2CpcfCompletedLayerReceipt(value)
            for value in raw_receipts
        )
        if (
            len(receipts) != next_layer
            or tuple(receipt.layer_index for receipt in receipts)
            != tuple(range(next_layer))
            or self.incoming_state_bank.layer_index != next_layer
            or next_layer >= len(self.boundary_bundle.layers)
        ):
            raise Lfm2CpcfModelFoldError(
                "Resume prefix, receipts, incoming bank, and next layer disagree "
                "or no unfinished layer remains."
            )
        boundary = _clone_evidence_tree(self.boundary_bundle)
        incoming = Lfm2CpcfPhaseStateBank(
            boundary=self.incoming_state_bank.boundary,
            layer_index=self.incoming_state_bank.layer_index,
            phases=tuple(self.incoming_state_bank.phases),
            states={
                phase.identity: self.incoming_state_bank.state(phase)
                for phase in self.incoming_state_bank.phases
            },
            state_bank_sha256=self.incoming_state_bank.state_bank_sha256,
        )
        object.__setattr__(
            self,
            "prefix_install_report",
            MappingProxyType(install),
        )
        object.__setattr__(self, "full_policy_sha256", policy_sha)
        object.__setattr__(self, "boundary_bundle", boundary)
        object.__setattr__(self, "incoming_state_bank", incoming)
        object.__setattr__(self, "completed_layer_receipts", receipts)

    @property
    def next_layer_index(self) -> int:
        return int(self.prefix_install_report["next_layer_index"])

    @property
    def checkpoint_reference(self) -> Mapping[str, Any]:
        return MappingProxyType(
            dict(self.prefix_install_report["checkpoint_reference"])
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix_install_report": json.loads(
                _canonical_json(dict(self.prefix_install_report)).decode("ascii")
            ),
            "full_policy_sha256": self.full_policy_sha256,
            "boundary_bundle_sha256": self.boundary_bundle.bundle_sha256,
            "incoming_state_bank_sha256": (self.incoming_state_bank.state_bank_sha256),
            "completed_layer_result_sha256": [
                receipt.result_sha256 for receipt in self.completed_layer_receipts
            ],
            "next_layer_index": self.next_layer_index,
        }


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ModelSnapshot:
    tensor_values: Mapping[str, Any]
    tensor_storage_anchors: Mapping[str, Any]
    tensor_identity: Mapping[str, tuple[Any, ...]]
    snapshot_metadata: Mapping[str, Any]
    parameter_gradients: Mapping[str, Any]
    module_modes: tuple[tuple[str, int, bool], ...]
    module_registries: tuple[Any, ...]
    state_fingerprint: str
    full_inventory_sha256: str
    structure_inventory_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2CpcfModelFoldProgressEvent:
    """Owned, sealed prefix evidence delivered to a synchronous observer."""

    _boundary_bundle: Lfm2CpcfBoundaryBundle
    _incoming_state_bank: Lfm2CpcfFailureIncomingBankSnapshot
    _completed_layer_receipt_bytes: tuple[bytes, ...]
    _target_prefix_cpu_tensor_snapshot: Mapping[str, Any]
    _target_prefix_snapshot_metadata_bytes: bytes
    _report_bytes: bytes

    def __post_init__(self) -> None:
        self.verify()

    @classmethod
    def capture(
        cls,
        *,
        boundary_bundle: Lfm2CpcfBoundaryBundle,
        incoming_state_bank: Lfm2CpcfPhaseStateBank,
        completed_layer_results: Sequence[
            Lfm2CpcfCompletedLayerReceipt | Lfm2CpcfCompiledLayerResult
        ],
        target_prefix_snapshot: _ModelSnapshot,
        full_policy_sha256: str,
    ) -> Lfm2CpcfModelFoldProgressEvent:
        boundary = _clone_boundary_bundle_evidence(boundary_bundle)
        if boundary is None:  # pragma: no cover - typed call boundary
            raise Lfm2CpcfModelFoldInstallError(
                "Progress event requires fitted boundary evidence."
            )
        incoming = _snapshot_incoming_bank_evidence(incoming_state_bank)
        if incoming is None:  # pragma: no cover - typed call boundary
            raise Lfm2CpcfModelFoldInstallError(
                "Progress event requires an incoming state bank."
            )
        receipts = tuple(
            Lfm2CpcfCompletedLayerReceipt(result.as_dict()).as_dict()
            for result in completed_layer_results
        )
        next_layer = incoming.layer_index
        if (
            next_layer > len(boundary.layers)
            or len(receipts) != next_layer
            or tuple(receipt["layer_index"] for receipt in receipts)
            != tuple(range(next_layer))
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Progress event prefix receipts and incoming bank are unordered."
            )
        policy_sha = _digest(
            full_policy_sha256,
            "progress_event.full_policy_sha256",
        )
        target_tensors = MappingProxyType(
            {
                name: tensor.detach().to(device="cpu").contiguous().clone()
                for name, tensor in target_prefix_snapshot.tensor_values.items()
            }
        )
        target_metadata = _snapshot_metadata_plain(target_prefix_snapshot)
        target_receipt = _cpu_tensor_snapshot_receipt_from_parts(
            target_tensors,
            target_metadata,
        )
        body = {
            "schema": LFM2_CPCF_MODEL_PROGRESS_EVENT_SCHEMA,
            "stage": "progress_checkpoint_publish",
            "next_layer_index": next_layer,
            "full_policy_sha256": policy_sha,
            "boundary_bundle_sha256": boundary.bundle_sha256,
            "boundary_bundle_inventory_sha256": (
                _boundary_bundle_inventory_sha256(boundary)
            ),
            "incoming_state_bank_sha256": incoming.state_bank_sha256,
            "incoming_state_snapshot_sha256": incoming.snapshot_sha256,
            "completed_layer_count": len(receipts),
            "completed_layer_result_sha256": [
                receipt["result_sha256"] for receipt in receipts
            ],
            "target_prefix_snapshot_receipt": dict(target_receipt),
            "target_prefix_tensor_count": len(target_tensors),
            "parent_evidence_tensors_copied": 0,
            "target_prefix_tensors_owned_cpu": True,
            "incoming_child_state_tensors_owned_cpu": True,
            "live_target_model_exposed": False,
            "checkpoint_publication_claimed": False,
        }
        report = {**body, "report_sha256": _sha(body)}
        return cls(
            _boundary_bundle=boundary,
            _incoming_state_bank=incoming,
            _completed_layer_receipt_bytes=tuple(
                _canonical_json(receipt) for receipt in receipts
            ),
            _target_prefix_cpu_tensor_snapshot=target_tensors,
            _target_prefix_snapshot_metadata_bytes=_canonical_json(target_metadata),
            _report_bytes=_canonical_json(report),
        )

    def verify(self) -> None:
        if not isinstance(self._boundary_bundle, Lfm2CpcfBoundaryBundle):
            raise Lfm2CpcfModelFoldError("Progress event boundary has the wrong type.")
        if not isinstance(
            self._incoming_state_bank,
            Lfm2CpcfFailureIncomingBankSnapshot,
        ):
            raise Lfm2CpcfModelFoldError(
                "Progress event incoming snapshot has the wrong type."
            )
        self._incoming_state_bank.verify()
        try:
            report = json.loads(self._report_bytes.decode("ascii"))
            metadata = json.loads(
                self._target_prefix_snapshot_metadata_bytes.decode("ascii")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2CpcfModelFoldError(
                "Progress event metadata is not canonical JSON."
            ) from exc
        if (
            _canonical_json(report) != self._report_bytes
            or _canonical_json(metadata) != self._target_prefix_snapshot_metadata_bytes
        ):
            raise Lfm2CpcfModelFoldError("Progress event metadata is not canonical.")
        body = dict(report)
        report_sha = _digest(
            body.pop("report_sha256", None),
            "progress_event.report_sha256",
        )
        receipts = tuple(
            Lfm2CpcfCompletedLayerReceipt(json.loads(payload.decode("ascii"))).as_dict()
            for payload in self._completed_layer_receipt_bytes
        )
        target_receipt = _cpu_tensor_snapshot_receipt_from_parts(
            self._target_prefix_cpu_tensor_snapshot,
            metadata,
        )
        next_layer = self._incoming_state_bank.layer_index
        if (
            set(report) != _MODEL_PROGRESS_EVENT_FIELDS
            or _sha(body) != report_sha
            or report.get("schema") != LFM2_CPCF_MODEL_PROGRESS_EVENT_SCHEMA
            or report.get("stage") != "progress_checkpoint_publish"
            or report.get("next_layer_index") != next_layer
            or report.get("completed_layer_count") != len(receipts)
            or len(receipts) != next_layer
            or tuple(receipt["layer_index"] for receipt in receipts)
            != tuple(range(next_layer))
            or report.get("completed_layer_result_sha256")
            != [receipt["result_sha256"] for receipt in receipts]
            or report.get("boundary_bundle_sha256")
            != self._boundary_bundle.bundle_sha256
            or report.get("boundary_bundle_inventory_sha256")
            != _boundary_bundle_inventory_sha256(self._boundary_bundle)
            or report.get("incoming_state_bank_sha256")
            != self._incoming_state_bank.state_bank_sha256
            or report.get("incoming_state_snapshot_sha256")
            != self._incoming_state_bank.snapshot_sha256
            or report.get("target_prefix_snapshot_receipt") != dict(target_receipt)
            or report.get("target_prefix_tensor_count")
            != len(self._target_prefix_cpu_tensor_snapshot)
            or report.get("parent_evidence_tensors_copied") != 0
            or report.get("target_prefix_tensors_owned_cpu") is not True
            or report.get("incoming_child_state_tensors_owned_cpu") is not True
            or report.get("live_target_model_exposed") is not False
            or report.get("checkpoint_publication_claimed") is not False
            or next_layer > len(self._boundary_bundle.layers)
        ):
            raise Lfm2CpcfModelFoldError(
                "Progress event prefix evidence or seal is invalid."
            )
        _digest(
            report.get("full_policy_sha256"),
            "progress_event.full_policy_sha256",
        )

    @property
    def stage(self) -> str:
        return "progress_checkpoint_publish"

    @property
    def next_layer_index(self) -> int:
        return self._incoming_state_bank.layer_index

    @property
    def full_policy_sha256(self) -> str:
        return str(self.report["full_policy_sha256"])

    @property
    def boundary_bundle(self) -> Lfm2CpcfBoundaryBundle:
        boundary = _clone_boundary_bundle_evidence(self._boundary_bundle)
        if boundary is None:  # pragma: no cover - internal invariant
            raise Lfm2CpcfModelFoldError("Progress event lost its boundary.")
        return boundary

    @property
    def incoming_state_bank(self) -> Lfm2CpcfFailureIncomingBankSnapshot:
        self._incoming_state_bank.verify()
        return self._incoming_state_bank

    @property
    def completed_layer_receipts(
        self,
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            json.loads(payload.decode("ascii"))
            for payload in self._completed_layer_receipt_bytes
        )

    @property
    def target_prefix_cpu_tensor_snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                name: tensor.clone()
                for name, tensor in (self._target_prefix_cpu_tensor_snapshot.items())
            }
        )

    @property
    def target_prefix_snapshot_metadata(self) -> Mapping[str, Any]:
        return json.loads(self._target_prefix_snapshot_metadata_bytes.decode("ascii"))

    @property
    def target_prefix_snapshot_receipt(self) -> Mapping[str, Any]:
        return dict(self.report["target_prefix_snapshot_receipt"])

    @property
    def report(self) -> Mapping[str, Any]:
        return json.loads(self._report_bytes.decode("ascii"))

    @property
    def report_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


Lfm2CpcfModelFoldProgressObserver = Callable[
    [Lfm2CpcfModelFoldProgressEvent],
    None,
]


@dataclass(frozen=True, slots=True)
class _ModelAuthInventory:
    state_fingerprint: str
    full_inventory_sha256: str


@dataclass(frozen=True, slots=True)
class _GradientSnapshot:
    requires_grad: bool
    gradient: Any | None
    storage_anchor: Any | None
    value: Any | None
    identity: tuple[Any, ...] | None


@dataclass(frozen=True, slots=True)
class _ModuleRegistrySnapshot:
    module: Any
    parameters: tuple[tuple[str, Any], ...]
    buffers: tuple[tuple[str, Any], ...]
    children: tuple[tuple[str, Any], ...]
    nonpersistent_buffers: frozenset[str]


@dataclass(slots=True)
class _ModelFoldProgress:
    stage: str = "validate_request"
    boundary_bundle: Lfm2CpcfBoundaryBundle | None = None
    boundary_bundle_inventory_sha256: str | None = None
    current_incoming_bank: Lfm2CpcfPhaseStateBank | None = None
    completed_layer_results: list[
        Lfm2CpcfCompletedLayerReceipt | Lfm2CpcfCompiledLayerResult
    ] = field(default_factory=list)
    active_layer_index: int | None = None
    active_operator_kind: str | None = None


def _named_values_all(
    model: Any,
    method_name: str,
) -> tuple[tuple[str, Any], ...]:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise Lfm2CpcfModelFoldInstallError(f"Model must expose {method_name}().")
    try:
        values = tuple(method(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        values = tuple(method())
    names = [str(name) for name, _value in values]
    if len(set(names)) != len(names):
        raise Lfm2CpcfModelFoldInstallError(
            f"Model {method_name}() contains duplicate names."
        )
    return tuple((str(name), value) for name, value in values)


def _tensor_identity(value: Any) -> tuple[Any, ...]:
    storage_pointer = (
        0 if value.numel() == 0 else int(value.untyped_storage().data_ptr())
    )
    return (
        id(value),
        int(value.data_ptr()),
        storage_pointer,
        int(value.storage_offset()),
        tuple(int(item) for item in value.stride()),
        tuple(int(item) for item in value.shape),
        value.dtype,
        value.device.type,
        value.device.index,
    )


def _tensor_identity_record(value: Any) -> dict[str, Any]:
    identity = _tensor_identity(value)
    return {
        "object_id": identity[0],
        "data_pointer": identity[1],
        "storage_pointer": identity[2],
        "storage_offset": identity[3],
        "stride": list(identity[4]),
        "shape": list(identity[5]),
        "dtype": str(identity[6]).removeprefix("torch."),
        "device_type": identity[7],
        "device_index": identity[8],
    }


def _module_registry_inventory(model: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "object_id": id(value),
            "parameters": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._parameters.items()
            ],
            "buffers": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._buffers.items()
            ],
            "children": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._modules.items()
            ],
            "nonpersistent_buffers": sorted(
                str(item) for item in value._non_persistent_buffers_set
            ),
        }
        for name, value in _named_values_all(model, "named_modules")
    ]


def _capture_module_registries(
    model: Any,
) -> tuple[_ModuleRegistrySnapshot, ...]:
    seen = set()
    result = []
    for _name, value in _named_values_all(model, "named_modules"):
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append(
            _ModuleRegistrySnapshot(
                module=value,
                parameters=tuple(value._parameters.items()),
                buffers=tuple(value._buffers.items()),
                children=tuple(value._modules.items()),
                nonpersistent_buffers=frozenset(value._non_persistent_buffers_set),
            )
        )
    return tuple(result)


def _restore_module_registries(
    snapshots: Sequence[_ModuleRegistrySnapshot],
) -> None:
    for snapshot in snapshots:
        module = snapshot.module
        module._parameters.clear()
        module._parameters.update(snapshot.parameters)
        module._buffers.clear()
        module._buffers.update(snapshot.buffers)
        module._modules.clear()
        module._modules.update(snapshot.children)
        module._non_persistent_buffers_set.clear()
        module._non_persistent_buffers_set.update(snapshot.nonpersistent_buffers)


def _alias_groups(
    values: Mapping[str, Any],
    *,
    storage: bool,
) -> tuple[tuple[str, ...], ...]:
    groups: dict[Any, list[str]] = {}
    for name, value in values.items():
        if storage and value.numel() != 0:
            key = (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
        elif storage:
            key = ("empty", id(value))
        else:
            key = id(value)
        groups.setdefault(key, []).append(name)
    return tuple(
        sorted(
            (tuple(sorted(names)) for names in groups.values()),
            key=lambda names: names,
        )
    )


def _model_named_tensors(
    model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    parameters = _named_values_all(model, "named_parameters")
    buffers = _named_values_all(model, "named_buffers")
    values = dict(parameters + buffers)
    if len(values) != len(parameters) + len(buffers) or not values:
        raise Lfm2CpcfModelFoldInstallError(
            "Model parameter and buffer names collide or are empty."
        )
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise Lfm2CpcfModelFoldInstallError(
            "Model parameters and buffers must be tensors."
        )
    modules = dict(_named_values_all(model, "named_modules"))
    metadata: dict[str, Any] = {}
    parameter_names = {name for name, _value in parameters}
    for name, value in values.items():
        if name in parameter_names:
            role = "parameter"
            persistent = True
        else:
            role = "buffer"
            owner_name, _separator, local_name = name.rpartition(".")
            try:
                owner = modules[owner_name]
            except KeyError as exc:
                raise Lfm2CpcfModelFoldInstallError(
                    f"Buffer owner for {name!r} is absent."
                ) from exc
            persistent = local_name not in getattr(
                owner,
                "_non_persistent_buffers_set",
                set(),
            )
        metadata[name] = {
            "role": role,
            "persistent": persistent,
            "shape": [int(size) for size in value.shape],
            "stride": [int(size) for size in value.stride()],
            "storage_offset": int(value.storage_offset()),
            "dtype": str(value.dtype).removeprefix("torch."),
            "original_device_type": value.device.type,
            "original_device_index": value.device.index,
        }
    snapshot_metadata = {
        "schema": "neural-forge-model-tensor-snapshot-metadata-v1",
        "all_named_parameters_and_buffers": True,
        "nonpersistent_buffers_included": True,
        "tensor_metadata": metadata,
        "object_alias_groups": [
            list(group) for group in _alias_groups(values, storage=False)
        ],
        "storage_alias_groups": [
            list(group) for group in _alias_groups(values, storage=True)
        ],
    }
    return values, snapshot_metadata


def _model_inventory(
    model: Any,
    *,
    include_tensor_bytes: bool,
) -> dict[str, Any]:
    values, metadata = _model_named_tensors(model)
    parameters = dict(_named_values_all(model, "named_parameters"))
    tensor_inventory = {}
    for name, value in sorted(values.items()):
        record = {
            **_tensor_identity_record(value),
            **metadata["tensor_metadata"][name],
        }
        if include_tensor_bytes:
            record["tensor_sha256"] = _tensor_sha256(value)
        tensor_inventory[name] = record
    gradients = {}
    gradient_values = {}
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad
        if gradient is not None:
            gradient_values[name] = gradient
        gradients[name] = {
            "requires_grad": bool(parameter.requires_grad),
            "present": gradient is not None,
            "identity": (
                None if gradient is None else _tensor_identity_record(gradient)
            ),
            "tensor_sha256": (None if gradient is None else _tensor_sha256(gradient)),
        }
    return {
        "tensor_inventory": tensor_inventory,
        "tensor_object_alias_groups": [
            list(group) for group in _alias_groups(values, storage=False)
        ],
        "tensor_storage_alias_groups": [
            list(group) for group in _alias_groups(values, storage=True)
        ],
        "parameter_gradients": gradients,
        "gradient_object_alias_groups": [
            list(group)
            for group in _alias_groups(
                gradient_values,
                storage=False,
            )
        ],
        "gradient_storage_alias_groups": [
            list(group)
            for group in _alias_groups(
                gradient_values,
                storage=True,
            )
        ],
        "module_registries": _module_registry_inventory(model),
        "module_modes": [
            [name, id(module), bool(module.training)]
            for name, module in _named_values_all(
                model,
                "named_modules",
            )
        ],
    }


def _snapshot_model_auth(model: Any) -> _ModelAuthInventory:
    return _ModelAuthInventory(
        state_fingerprint=fingerprint_model_state(model),
        full_inventory_sha256=_sha(_model_inventory(model, include_tensor_bytes=True)),
    )


def _authenticate_model(
    model: Any,
    snapshot: _ModelAuthInventory,
) -> None:
    if (
        fingerprint_model_state(model) != snapshot.state_fingerprint
        or _sha(_model_inventory(model, include_tensor_bytes=True))
        != snapshot.full_inventory_sha256
    ):
        raise Lfm2CpcfModelFoldInstallError(
            "Model bytes, registries, identities, aliases, gradients, "
            "buffer persistence, or modes changed."
        )


def _state_fingerprint_from_values(values: Mapping[str, Any]) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _snapshot_model(model: Any) -> _ModelSnapshot:
    state, snapshot_metadata = _model_named_tensors(model)
    tensor_values = {
        str(name): value.detach().to(device="cpu").contiguous().clone()
        for name, value in state.items()
    }
    tensor_identity = {
        str(name): _tensor_identity(value) for name, value in state.items()
    }
    tensor_storage_anchors = {
        str(name): value.detach() for name, value in state.items()
    }
    gradients = {}
    for name, parameter in _named_values_all(
        model,
        "named_parameters",
    ):
        gradient = parameter.grad
        gradients[str(name)] = _GradientSnapshot(
            requires_grad=bool(parameter.requires_grad),
            gradient=gradient,
            storage_anchor=(None if gradient is None else gradient.detach()),
            value=(
                None
                if gradient is None
                else gradient.detach().to(device="cpu").contiguous().clone()
            ),
            identity=(None if gradient is None else _tensor_identity(gradient)),
        )
    state_fingerprint = fingerprint_model_state(model)
    if _state_fingerprint_from_values(tensor_values) != state_fingerprint:
        raise Lfm2CpcfModelFoldInstallError(
            "Complete named tensor snapshot differs from live model state."
        )
    return _ModelSnapshot(
        tensor_values=MappingProxyType(tensor_values),
        tensor_storage_anchors=MappingProxyType(tensor_storage_anchors),
        tensor_identity=MappingProxyType(tensor_identity),
        snapshot_metadata=MappingProxyType(snapshot_metadata),
        parameter_gradients=MappingProxyType(gradients),
        module_modes=tuple(
            (str(name), id(module), bool(module.training))
            for name, module in _named_values_all(
                model,
                "named_modules",
            )
        ),
        module_registries=_capture_module_registries(model),
        state_fingerprint=state_fingerprint,
        full_inventory_sha256=_sha(_model_inventory(model, include_tensor_bytes=True)),
        structure_inventory_sha256=_sha(
            _model_inventory(model, include_tensor_bytes=False)
        ),
    )


def _restore_model(model: Any, snapshot: _ModelSnapshot) -> None:
    torch = _torch()
    _restore_module_registries(snapshot.module_registries)
    state, snapshot_metadata = _model_named_tensors(model)
    if set(state) != set(snapshot.tensor_values):
        raise Lfm2CpcfModelFoldInstallError(
            "Target state keys changed and cannot be transactionally restored."
        )
    if snapshot_metadata != dict(snapshot.snapshot_metadata):
        raise Lfm2CpcfModelFoldInstallError(
            "Target tensor registries could not be restored safely."
        )
    with torch.no_grad():
        for name, value in state.items():
            if _tensor_identity(value) != snapshot.tensor_identity[name]:
                value.data = snapshot.tensor_storage_anchors[name]
            if _tensor_identity(value) != snapshot.tensor_identity[name]:
                raise Lfm2CpcfModelFoldInstallError(
                    "Target tensor storage identity could not be restored."
                )
            value.copy_(
                snapshot.tensor_values[name].to(
                    device=value.device,
                    dtype=value.dtype,
                )
            )
    named_parameters = dict(_named_values_all(model, "named_parameters"))
    if set(named_parameters) != set(snapshot.parameter_gradients):
        raise Lfm2CpcfModelFoldInstallError(
            "Target Parameter set changed during the model transaction."
        )
    for name, parameter in named_parameters.items():
        gradient_snapshot = snapshot.parameter_gradients[name]
        parameter.requires_grad_(gradient_snapshot.requires_grad)
        parameter.grad = gradient_snapshot.gradient
        if gradient_snapshot.gradient is not None:
            gradient = gradient_snapshot.gradient
            with torch.no_grad():
                if _tensor_identity(gradient) != gradient_snapshot.identity:
                    gradient.data = gradient_snapshot.storage_anchor
                gradient.copy_(
                    gradient_snapshot.value.to(
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                )
    modules = dict(_named_values_all(model, "named_modules"))
    if set(modules) != {name for name, _identity, _mode in snapshot.module_modes}:
        raise Lfm2CpcfModelFoldInstallError(
            "Target module tree changed during the model transaction."
        )
    for name, identity, mode in snapshot.module_modes:
        if id(modules[name]) != identity:
            raise Lfm2CpcfModelFoldInstallError(
                "Target module identities changed during the transaction."
            )
        modules[name].training = mode
    if fingerprint_model_state(model) != snapshot.state_fingerprint:
        raise Lfm2CpcfModelFoldInstallError(
            "Target values failed exact whole-model rollback."
        )
    restored_state, restored_metadata = _model_named_tensors(model)
    if restored_metadata != dict(snapshot.snapshot_metadata):
        raise Lfm2CpcfModelFoldInstallError(
            "Target tensor roles or aliases changed during rollback."
        )
    for name, expected in snapshot.tensor_identity.items():
        value = restored_state[name]
        observed = _tensor_identity(value)
        if observed != expected:
            raise Lfm2CpcfModelFoldInstallError(
                "Target tensor identity changed during rollback."
            )
    for name, gradient_snapshot in snapshot.parameter_gradients.items():
        parameter = named_parameters[name]
        gradient = gradient_snapshot.gradient
        if (
            parameter.requires_grad != gradient_snapshot.requires_grad
            or parameter.grad is not gradient
            or (
                gradient is not None
                and (
                    _tensor_identity(gradient) != gradient_snapshot.identity
                    or _tensor_sha256(gradient)
                    != _tensor_sha256(gradient_snapshot.value)
                )
            )
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Target gradient state failed exact rollback."
            )
    if (
        tuple(
            (name, id(module), bool(module.training))
            for name, module in _named_values_all(
                model,
                "named_modules",
            )
        )
        != snapshot.module_modes
    ):
        raise Lfm2CpcfModelFoldInstallError(
            "Target training modes failed exact rollback."
        )
    if (
        _sha(_model_inventory(model, include_tensor_bytes=True))
        != snapshot.full_inventory_sha256
    ):
        raise Lfm2CpcfModelFoldInstallError(
            "Target full inventory failed exact rollback."
        )


def _prove_exact_whole_model_rollback(
    source_model: Any,
    target_model: Any,
    *,
    source_snapshot_before: _ModelAuthInventory,
    target_snapshot_before: _ModelSnapshot,
) -> Mapping[str, Any]:
    """Seal exact target restoration plus immutable-source evidence."""

    _authenticate_model(source_model, source_snapshot_before)
    source_after = fingerprint_model_state(source_model)
    source_full_inventory_after = _sha(
        _model_inventory(source_model, include_tensor_bytes=True)
    )
    target_after = fingerprint_model_state(target_model)
    target_full_inventory_after = _sha(
        _model_inventory(target_model, include_tensor_bytes=True)
    )
    if (
        source_after != source_snapshot_before.state_fingerprint
        or source_full_inventory_after != source_snapshot_before.full_inventory_sha256
        or target_after != target_snapshot_before.state_fingerprint
        or target_full_inventory_after != target_snapshot_before.full_inventory_sha256
    ):
        raise Lfm2CpcfModelFoldInstallError(
            "Source immutability or exact target rollback could not be proven."
        )
    body = {
        "source_state_fingerprint_before": (source_snapshot_before.state_fingerprint),
        "source_state_fingerprint_after": source_after,
        "source_full_inventory_sha256_before": (
            source_snapshot_before.full_inventory_sha256
        ),
        "source_full_inventory_sha256_after": (source_full_inventory_after),
        "source_state_preserved": True,
        "source_full_inventory_preserved": True,
        "target_state_fingerprint_before": (target_snapshot_before.state_fingerprint),
        "target_state_fingerprint_after": target_after,
        "target_full_inventory_sha256_before": (
            target_snapshot_before.full_inventory_sha256
        ),
        "target_full_inventory_sha256_after": (target_full_inventory_after),
        "target_tensor_bytes_restored": True,
        "target_tensor_identities_preserved": True,
        "target_tensor_aliases_preserved": True,
        "target_all_named_parameters_and_buffers_restored": True,
        "target_parameter_and_buffer_registries_restored": True,
        "target_nonpersistent_buffers_restored": True,
        "target_parameter_gradient_identity_and_bytes_restored": True,
        "target_parameter_gradient_storage_aliases_restored": True,
        "target_module_modes_restored": True,
        "exact_whole_model_rollback_proven": True,
    }
    return MappingProxyType({**body, "proof_sha256": _sha(body)})


def _snapshot_metadata_plain(snapshot: _ModelSnapshot) -> dict[str, Any]:
    return _canonical_mapping_copy(
        dict(snapshot.snapshot_metadata),
        "snapshot.snapshot_metadata",
    )


def _validated_alias_groups(
    value: Any,
    *,
    names: set[str],
    field_name: str,
) -> list[list[str]]:
    if not isinstance(value, list):
        raise Lfm2CpcfModelFoldError(f"{field_name} must be a list of alias groups.")
    groups: list[list[str]] = []
    flattened: list[str] = []
    for group in value:
        if (
            not isinstance(group, list)
            or not group
            or any(not isinstance(name, str) for name in group)
            or group != sorted(group)
            or len(set(group)) != len(group)
        ):
            raise Lfm2CpcfModelFoldError(
                f"{field_name} contains a noncanonical alias group."
            )
        groups.append(group)
        flattened.extend(group)
    if (
        groups != sorted(groups)
        or len(flattened) != len(set(flattened))
        or set(flattened) != names
    ):
        raise Lfm2CpcfModelFoldError(
            f"{field_name} does not cover every tensor exactly once."
        )
    return groups


def _cpu_tensor_snapshot_receipt_from_parts(
    tensors: Mapping[str, Any],
    snapshot_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    metadata = _canonical_mapping_copy(
        snapshot_metadata,
        "snapshot_metadata",
    )
    names = set(tensors)
    tensor_metadata = metadata.get("tensor_metadata")
    if (
        metadata.get("schema") != "neural-forge-model-tensor-snapshot-metadata-v1"
        or metadata.get("all_named_parameters_and_buffers") is not True
        or metadata.get("nonpersistent_buffers_included") is not True
        or not isinstance(tensor_metadata, dict)
        or set(tensor_metadata) != names
    ):
        raise Lfm2CpcfModelFoldError("Prefix snapshot metadata is incomplete.")
    object_alias_groups = _validated_alias_groups(
        metadata.get("object_alias_groups"),
        names=names,
        field_name="object_alias_groups",
    )
    storage_alias_groups = _validated_alias_groups(
        metadata.get("storage_alias_groups"),
        names=names,
        field_name="storage_alias_groups",
    )
    inventory = {
        name: {
            "shape": [int(size) for size in tensor.shape],
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "tensor_sha256": _tensor_sha256(tensor),
            "role": tensor_metadata[name].get("role"),
            "persistent": tensor_metadata[name].get("persistent"),
            "original_stride": tensor_metadata[name].get("stride"),
            "original_storage_offset": tensor_metadata[name].get("storage_offset"),
            "original_device_type": tensor_metadata[name].get("original_device_type"),
            "original_device_index": tensor_metadata[name].get("original_device_index"),
        }
        for name, tensor in sorted(tensors.items())
    }
    for name, record in inventory.items():
        expected = tensor_metadata[name]
        if (
            record["role"] not in {"parameter", "buffer"}
            or not isinstance(record["persistent"], bool)
            or (record["role"] == "parameter" and record["persistent"] is not True)
            or expected.get("shape") != record["shape"]
            or expected.get("dtype") != record["dtype"]
            or not isinstance(record["original_stride"], list)
            or len(record["original_stride"]) != len(record["shape"])
            or not isinstance(record["original_storage_offset"], int)
            or record["original_storage_offset"] < 0
            or record["original_device_type"]
            not in {
                "cpu",
                "cuda",
                "mps",
                "xpu",
            }
            or (
                record["original_device_index"] is not None
                and (
                    isinstance(record["original_device_index"], bool)
                    or not isinstance(
                        record["original_device_index"],
                        int,
                    )
                    or record["original_device_index"] < 0
                )
            )
        ):
            raise Lfm2CpcfModelFoldError(
                f"Prefix tensor metadata for {name!r} is invalid."
            )
    body = {
        "state_fingerprint": _state_fingerprint_from_values(tensors),
        "device": "cpu",
        "contiguous": True,
        "tensor_count": len(inventory),
        "tensor_inventory": inventory,
        "object_alias_groups": object_alias_groups,
        "storage_alias_groups": storage_alias_groups,
        "all_named_parameters_and_buffers": True,
        "nonpersistent_buffers_included": True,
        "snapshot_role": ("failed_transaction_prefix_before_whole_model_rollback"),
    }
    return MappingProxyType({**body, "snapshot_sha256": _sha(body)})


def _cpu_tensor_snapshot_receipt(
    snapshot: _ModelSnapshot,
) -> Mapping[str, Any]:
    receipt = _cpu_tensor_snapshot_receipt_from_parts(
        snapshot.tensor_values,
        _snapshot_metadata_plain(snapshot),
    )
    if receipt["state_fingerprint"] != snapshot.state_fingerprint:
        raise Lfm2CpcfModelFoldInstallError(
            "Prefix snapshot values differ from their live state fingerprint."
        )
    return receipt


def _authenticated_evidence_tree(value: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            **_tensor_identity_record(value),
            "tensor_sha256": _tensor_sha256(value),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        pairs = [
            [
                _authenticated_evidence_tree(key),
                _authenticated_evidence_tree(child),
            ]
            for key, child in value.items()
        ]
        pairs.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"kind": "mapping", "items": pairs}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_authenticated_evidence_tree(child) for child in value],
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": (f"{type(value).__module__}.{type(value).__qualname__}"),
            "fields": {
                descriptor.name: _authenticated_evidence_tree(
                    getattr(value, descriptor.name)
                )
                for descriptor in fields(value)
            },
        }
    raise Lfm2CpcfModelFoldError(
        "Fold evidence contains an unauthenticated runtime object "
        f"of type {type(value).__module__}.{type(value).__qualname__}."
    )


def _model_fold_inputs_sha256(
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
) -> str:
    return _sha(
        {
            "phases": _authenticated_evidence_tree(tuple(phases)),
            "source_tied_table": _authenticated_evidence_tree(source_tied_table),
        }
    )


def _boundary_bundle_inventory_sha256(
    boundary_bundle: Lfm2CpcfBoundaryBundle,
) -> str:
    if not isinstance(boundary_bundle, Lfm2CpcfBoundaryBundle):
        raise Lfm2CpcfModelFoldError(
            "Boundary inventory requires a typed boundary bundle."
        )
    return _sha(_authenticated_evidence_tree(boundary_bundle))


def _clone_evidence_tree(
    value: Any,
    *,
    memo: dict[int, Any] | None = None,
) -> Any:
    torch = _torch()
    active = {} if memo is None else memo
    identity = id(value)
    if identity in active:
        return active[identity]
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        active[identity] = cloned
        return cloned
    if isinstance(value, Mapping):
        cloned_mapping: dict[Any, Any] = {}
        active[identity] = cloned_mapping
        for key, child in value.items():
            cloned_mapping[_clone_evidence_tree(key, memo=active)] = (
                _clone_evidence_tree(child, memo=active)
            )
        return cloned_mapping
    if isinstance(value, tuple):
        cloned_tuple = tuple(
            _clone_evidence_tree(child, memo=active) for child in value
        )
        active[identity] = cloned_tuple
        return cloned_tuple
    if isinstance(value, list):
        cloned_list: list[Any] = []
        active[identity] = cloned_list
        cloned_list.extend(_clone_evidence_tree(child, memo=active) for child in value)
        return cloned_list
    if is_dataclass(value) and not isinstance(value, type):
        kwargs = {
            descriptor.name: _clone_evidence_tree(
                getattr(value, descriptor.name),
                memo=active,
            )
            for descriptor in fields(value)
            if descriptor.init
        }
        cloned_dataclass = type(value)(**kwargs)
        active[identity] = cloned_dataclass
        return cloned_dataclass
    return value


def _clone_boundary_bundle_evidence(
    value: Lfm2CpcfBoundaryBundle | None,
) -> Lfm2CpcfBoundaryBundle | None:
    if value is None:
        return None
    if not isinstance(value, Lfm2CpcfBoundaryBundle):
        raise Lfm2CpcfModelFoldError("Boundary evidence must be a typed bundle.")
    return _clone_evidence_tree(value)


def _target_topology_sha256(model: Any) -> tuple[str, int]:
    values, metadata = _model_named_tensors(model)
    layers = tuple(_model_core(model, "target_model").layers)
    body = {
        "schema": LFM2_CPCF_TARGET_TOPOLOGY_SNAPSHOT_SCHEMA,
        "model_type": "lfm2",
        "layer_count": len(layers),
        "tensor_metadata": metadata["tensor_metadata"],
        "object_alias_groups": metadata["object_alias_groups"],
        "storage_alias_groups": metadata["storage_alias_groups"],
        "all_named_parameters_and_buffers": True,
        "nonpersistent_buffers_included": True,
    }
    return _sha(body), len(values)


def _phase_lineage_hashes(
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> tuple[dict[str, str], dict[str, str]]:
    splits = ("selection", "fit", "validation")
    grouped = {
        split: tuple(phase for phase in phases if phase.split == split)
        for split in splits
    }
    if any(not values for values in grouped.values()):
        raise Lfm2CpcfModelFoldError(
            "Resume phase lineage requires selection, fit, and validation phases."
        )
    phase_hashes = {}
    row_hashes = {}
    for split, values in grouped.items():
        phase_hashes[split] = _sha(
            [
                {
                    "ordinal": ordinal,
                    "batch_index": phase.batch_index,
                    "record_id": phase.record_id,
                    "phase": phase.phase,
                    "phase_sha256": phase.phase_sha256,
                    "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
                }
                for ordinal, phase in enumerate(values)
            ]
        )
        rows = []
        for phase in values:
            first_suffix_position = (
                phase.full_prefix_tokens - phase.sampled_suffix_tokens
            )
            rows.extend(
                {
                    "batch_index": phase.batch_index,
                    "record_id": phase.record_id,
                    "split": phase.split,
                    "phase": phase.phase,
                    "phase_sha256": phase.phase_sha256,
                    "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
                    "batch_row_index": batch_row,
                    "sampled_suffix_index": sample_index,
                    "full_prefix_position": first_suffix_position + sample_index,
                }
                for batch_row in range(phase.batch_size)
                for sample_index in range(phase.sampled_suffix_tokens)
            )
        row_hashes[split] = _sha(rows)
    return phase_hashes, row_hashes


def _authenticate_model_fold_resume_state(
    resume_state: Lfm2CpcfModelFoldResumeState,
    *,
    target_model: Any,
    phases: Sequence[Lfm2CpcfTargetPhase],
    policy: Lfm2CpcfModelFoldPolicy,
) -> Lfm2CpcfModelFoldResumeState:
    if not isinstance(resume_state, Lfm2CpcfModelFoldResumeState):
        raise TypeError("resume_state must be Lfm2CpcfModelFoldResumeState.")
    # Reconstruct the typed boundary so caller mutation after construction
    # cannot bypass any report, tensor, receipt, or bank authentication.
    authenticated = Lfm2CpcfModelFoldResumeState(
        prefix_install_report=resume_state.prefix_install_report,
        full_policy_sha256=resume_state.full_policy_sha256,
        boundary_bundle=resume_state.boundary_bundle,
        incoming_state_bank=resume_state.incoming_state_bank,
        completed_layer_receipts=resume_state.completed_layer_receipts,
    )
    if authenticated.full_policy_sha256 != _sha(policy.as_dict()):
        raise Lfm2CpcfModelFoldError(
            "Resume checkpoint policy SHA-256 differs from the requested fold."
        )
    install = authenticated.prefix_install_report
    topology_sha, tensor_count = _target_topology_sha256(target_model)
    if (
        install["target_topology_sha256"] != topology_sha
        or install["tensor_count"] != tensor_count
        or install["installed_state_sha256_after"]
        != fingerprint_model_state(target_model)
    ):
        raise Lfm2CpcfModelFoldError(
            "Live target topology or installed prefix bytes differ from the "
            "authenticated checkpoint install."
        )
    bundle = authenticated.boundary_bundle
    if len(bundle.layers) != len(
        _model_core(target_model, "target_model").layers
    ) or bundle.target_width != int(target_model.config.hidden_size):
        raise Lfm2CpcfModelFoldError(
            "Resume boundary bundle differs from the live target topology."
        )
    phase_hashes, row_hashes = _phase_lineage_hashes(phases)
    if (
        dict(bundle.split_phase_sha256) != phase_hashes
        or dict(bundle.split_row_sha256) != row_hashes
    ):
        raise Lfm2CpcfModelFoldError(
            "Resume boundary bundle differs from authenticated phase lineage."
        )
    incoming = authenticated.incoming_state_bank
    expected_boundary = (
        "input_embeddings"
        if authenticated.next_layer_index == 0
        else f"layer.{authenticated.next_layer_index - 1:03d}.post_layer"
    )
    if (
        incoming.boundary != expected_boundary
        or incoming.hidden_size != bundle.target_width
        or len(incoming.phases) != len(phases)
        or any(
            observed.identity != expected.identity
            or observed.phase_sha256 != expected.phase_sha256
            or _authenticated_evidence_tree(observed)
            != _authenticated_evidence_tree(expected)
            for observed, expected in zip(incoming.phases, phases, strict=True)
        )
    ):
        raise Lfm2CpcfModelFoldError(
            "Resume incoming state bank differs from authenticated phases."
        )
    return authenticated


def _validate_resume_receipt_chain(
    resume_state: Lfm2CpcfModelFoldResumeState,
    *,
    initial_state_bank: Lfm2CpcfPhaseStateBank,
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> None:
    receipts = tuple(resume_state.completed_layer_receipts)
    bundle = resume_state.boundary_bundle
    expected_input_sha = initial_state_bank.state_bank_sha256
    split_counts = {
        split: sum(phase.split == split for phase in phases)
        for split in ("selection", "fit", "validation")
    }
    for expected_index, (receipt, charts) in enumerate(
        zip(receipts, bundle.layers, strict=False)
    ):
        report = receipt.report
        if (
            receipt.layer_index != expected_index
            or receipt.operator_kind != charts.operator_kind
            or receipt.input_bank_sha256 != expected_input_sha
            or report["boundary_charts_sha256"] != charts.layer_charts_sha256
            or report["source_width"] != charts.source_width
            or report["target_width"] != charts.target_width
            or report["split_phase_counts"] != split_counts
        ):
            raise Lfm2CpcfModelFoldError(
                "Resume completed layer receipts differ from boundary, phase, "
                "or causal state-bank lineage."
            )
        expected_input_sha = receipt.next_bank_sha256
    if expected_input_sha != resume_state.incoming_state_bank.state_bank_sha256:
        raise Lfm2CpcfModelFoldError(
            "Resume completed receipt chain does not end at the incoming bank."
        )


def _model_fold_resume_inputs_sha256(
    value: Lfm2CpcfModelFoldResumeState | None,
) -> str | None:
    if value is None:
        return None
    incoming = value.incoming_state_bank
    return _sha(
        {
            "prefix_install_report": dict(value.prefix_install_report),
            "full_policy_sha256": value.full_policy_sha256,
            "boundary_bundle_inventory_sha256": (
                _boundary_bundle_inventory_sha256(value.boundary_bundle)
            ),
            "incoming_state_bank": {
                "boundary": incoming.boundary,
                "layer_index": incoming.layer_index,
                "state_bank_sha256": incoming.state_bank_sha256,
                "ordered_states": [
                    {
                        "identity": [phase.batch_index, phase.phase],
                        "phase_sha256": phase.phase_sha256,
                        "state_sha256": _tensor_sha256(incoming.states[phase.identity]),
                    }
                    for phase in incoming.phases
                ],
            },
            "completed_layer_receipts": [
                receipt.as_dict() for receipt in value.completed_layer_receipts
            ],
        }
    )


def _freeze_evidence_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_evidence_metadata(child) for key, child in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_evidence_metadata(child) for child in value)
    return value


def _snapshot_incoming_bank_evidence(
    value: (Lfm2CpcfPhaseStateBank | Lfm2CpcfFailureIncomingBankSnapshot | None),
) -> Lfm2CpcfFailureIncomingBankSnapshot | None:
    if value is None:
        return None
    if isinstance(value, Lfm2CpcfFailureIncomingBankSnapshot):
        value.verify()
        return value
    return Lfm2CpcfFailureIncomingBankSnapshot.capture(value)


def _verified_layer_failure_report(
    error: Exception,
) -> tuple[Mapping[str, Any] | None, bytes | None, str | None]:
    if not isinstance(error, Lfm2CpcfLayerCompilationFailure):
        return None, None, None
    report = error.report
    report_body = dict(report)
    report_sha = _digest(
        report_body.pop("report_sha256", None),
        "layer_failure_report.report_sha256",
    )
    if _sha(report_body) != report_sha:
        raise Lfm2CpcfModelFoldInstallError(
            "Nested layer failure report is not self-sealed."
        )
    report_bytes = _canonical_json(dict(report))
    return (
        report,
        report_bytes,
        hashlib.sha256(report_bytes).hexdigest(),
    )


def _completed_layer_receipts(
    progress: _ModelFoldProgress,
) -> tuple[Mapping[str, Any], ...]:
    receipts = []
    for expected_index, result in enumerate(progress.completed_layer_results):
        if (
            not isinstance(
                result,
                (Lfm2CpcfCompletedLayerReceipt, Lfm2CpcfCompiledLayerResult),
            )
            or result.layer_index != expected_index
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Completed model prefix lacks ordered typed layer receipts."
            )
        receipt = Lfm2CpcfCompletedLayerReceipt(result.as_dict()).as_dict()
        receipts.append(MappingProxyType(dict(receipt)))
    return tuple(receipts)


def _emit_model_fold_progress(
    observer: Lfm2CpcfModelFoldProgressObserver | None,
    *,
    target_model: Any,
    boundary_bundle: Lfm2CpcfBoundaryBundle,
    state_bank: Lfm2CpcfPhaseStateBank,
    resolved_policy: Lfm2CpcfModelFoldPolicy,
    progress: _ModelFoldProgress,
) -> None:
    if observer is None:
        return
    progress.stage = "progress_checkpoint_publish"
    progress.active_layer_index = None
    progress.active_operator_kind = None
    target_snapshot = _snapshot_model(target_model)
    target_auth = _ModelAuthInventory(
        state_fingerprint=target_snapshot.state_fingerprint,
        full_inventory_sha256=target_snapshot.full_inventory_sha256,
    )
    event = Lfm2CpcfModelFoldProgressEvent.capture(
        boundary_bundle=boundary_bundle,
        incoming_state_bank=state_bank,
        completed_layer_results=progress.completed_layer_results,
        target_prefix_snapshot=target_snapshot,
        full_policy_sha256=_sha(resolved_policy.as_dict()),
    )
    observer(event)
    event.verify()
    _authenticate_model(target_model, target_auth)


def _wrap_model_fold_failure(
    error: Exception,
    *,
    progress: _ModelFoldProgress,
    failed_prefix_snapshot: _ModelSnapshot,
    rollback_proof: Mapping[str, Any],
) -> Lfm2CpcfModelFoldFailure:
    layer_report, layer_report_bytes, layer_report_bytes_sha = (
        _verified_layer_failure_report(error)
    )
    completed_receipts = _completed_layer_receipts(progress)
    snapshot_receipt = _cpu_tensor_snapshot_receipt(failed_prefix_snapshot)
    boundary_sha = (
        None
        if progress.boundary_bundle is None
        else progress.boundary_bundle.bundle_sha256
    )
    incoming_sha = (
        None
        if progress.current_incoming_bank is None
        else progress.current_incoming_bank.state_bank_sha256
    )
    incoming_index = (
        None
        if progress.current_incoming_bank is None
        else progress.current_incoming_bank.layer_index
    )
    body = {
        "schema": LFM2_CPCF_MODEL_FAILURE_SCHEMA,
        "stage": progress.stage,
        "active_layer_index": progress.active_layer_index,
        "active_operator_kind": progress.active_operator_kind,
        "cause": {
            "module": type(error).__module__,
            "type": type(error).__qualname__,
            "message": str(error),
        },
        "boundary_bundle_sha256": boundary_sha,
        "current_incoming_bank_sha256": incoming_sha,
        "current_incoming_layer_index": incoming_index,
        "completed_layer_count": len(completed_receipts),
        "completed_layer_receipts": [dict(receipt) for receipt in completed_receipts],
        "failed_prefix_target_cpu_tensor_snapshot": dict(snapshot_receipt),
        "nested_layer_failure_report": (
            None if layer_report is None else dict(layer_report)
        ),
        "nested_layer_failure_report_sha256": (
            None if layer_report is None else layer_report["report_sha256"]
        ),
        "nested_layer_failure_report_canonical_bytes_sha256": (layer_report_bytes_sha),
        "nested_layer_failure_report_preserved_byte_for_byte": (
            layer_report is not None
        ),
        "rollback_proof": dict(rollback_proof),
        "compiled_model_fold_issued": False,
        "parameter_compaction_success_receipt_issued": False,
        "promotion_claimed": False,
        "promotable_candidate": False,
        "failure_evidence_only": True,
    }
    report = {**body, "report_sha256": _sha(body)}
    message = (
        f"LFM2 CPCF model fold failed at {progress.stage} after exact "
        f"whole-model rollback: {error}"
    )
    return Lfm2CpcfModelFoldFailure(
        message,
        stage=progress.stage,
        boundary_bundle=progress.boundary_bundle,
        current_incoming_bank=progress.current_incoming_bank,
        completed_layer_receipts=completed_receipts,
        prefix_target_cpu_tensor_snapshot=(failed_prefix_snapshot.tensor_values),
        prefix_target_snapshot_metadata=(
            _snapshot_metadata_plain(failed_prefix_snapshot)
        ),
        rollback_proof=rollback_proof,
        report=report,
        layer_failure_report=layer_report,
        layer_failure_report_bytes=layer_report_bytes,
    )


def _preserved_qk_norm_weights(
    source_model: Any,
    target_model: Any,
) -> dict[int, tuple[Any, Any]]:
    source_layers = tuple(_model_core(source_model, "source_model").layers)
    target_layers = tuple(_model_core(target_model, "target_model").layers)
    if len(source_layers) != len(target_layers):
        raise Lfm2CpcfModelFoldError(
            "Current LFM2 CPCF compiler requires equal source/target depth."
        )
    result = {}
    for index, (source, target) in enumerate(zip(source_layers, target_layers)):
        if not bool(getattr(target, "is_attention_layer", False)):
            continue
        source_attention = getattr(source, "self_attn", None)
        target_attention = getattr(target, "self_attn", None)
        values = (
            getattr(
                getattr(source_attention, "q_layernorm", None),
                "weight",
                None,
            ),
            getattr(
                getattr(source_attention, "k_layernorm", None),
                "weight",
                None,
            ),
            getattr(
                getattr(target_attention, "q_layernorm", None),
                "weight",
                None,
            ),
            getattr(
                getattr(target_attention, "k_layernorm", None),
                "weight",
                None,
            ),
        )
        if any(value is None for value in values):
            raise Lfm2CpcfModelFoldError(
                f"Attention layer {index} lacks native Q/K RMSNorm gains."
            )
        source_q, source_k, target_q, target_k = values
        if tuple(source_q.shape) == tuple(target_q.shape) and tuple(
            source_k.shape
        ) == tuple(target_k.shape):
            result[index] = (
                source_q.detach().to(device="cpu"),
                source_k.detach().to(device="cpu"),
            )
    return result


def _final_norm_epsilon(target_model: Any) -> float:
    norm = getattr(
        _model_core(target_model, "target_model"),
        "embedding_norm",
        None,
    )
    for name in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(norm, name, None)
        if value is None:
            continue
        try:
            epsilon = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(epsilon) and epsilon > 0.0:
            return epsilon
    raise Lfm2CpcfModelFoldError("Target final RMSNorm exposes no positive epsilon.")


def _balanced_phase_weights(
    phases: Sequence[Lfm2CpcfTargetPhase],
    split: str,
) -> Any:
    torch = _torch()
    selected = tuple(phase for phase in phases if phase.split == split)
    if not selected:
        raise Lfm2CpcfModelFoldError(f"No {split!r} phases are available.")
    phase_mass = 1.0 / len(selected)
    pieces = []
    for phase in selected:
        rows = phase.batch_size * phase.sampled_suffix_tokens
        pieces.append(
            torch.full(
                (rows,),
                phase_mass / rows,
                dtype=torch.float64,
            )
        )
    return torch.cat(pieces).contiguous()


def _relative_rmse(prediction: Any, target: Any, weights: Any) -> float:
    numerator = (weights[:, None] * (prediction - target).square()).sum()
    denominator = (weights[:, None] * target.square()).sum().clamp_min(1e-30)
    return float((numerator / denominator).sqrt().item())


def _retained_energy(prediction: Any, target: Any, weights: Any) -> float:
    error = (weights[:, None] * (prediction - target).square()).sum()
    energy = (weights[:, None] * target.square()).sum().clamp_min(1e-30)
    return float(max(0.0, 1.0 - float((error / energy).item())))


def _fit_and_install_final_norm(
    target_model: Any,
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    final_state_bank: Lfm2CpcfPhaseStateBank,
    boundary_bundle: Lfm2CpcfBoundaryBundle,
    ridge: float,
) -> Mapping[str, Any]:
    torch = _torch()
    interface = boundary_bundle.interface
    split_rows: dict[str, tuple[Any, Any, Any]] = {}
    for split in ("fit", "validation"):
        selected = tuple(phase for phase in phases if phase.split == split)
        child = torch.cat(
            [
                final_state_bank.suffix(phase).reshape(
                    -1,
                    final_state_bank.hidden_size,
                )
                for phase in selected
            ],
            dim=0,
        ).to(device="cpu", dtype=torch.float64)
        parent = torch.cat(
            [
                phase.parent_evidence.final_norm_tensors[
                    "output.final_norm_output"
                ].reshape(-1, interface.source_width)
                for phase in selected
            ],
            dim=0,
        )
        target = interface.encode_producer_rows(parent)
        split_rows[split] = (
            child,
            target,
            _balanced_phase_weights(phases, split),
        )
    fit_child, fit_target, fit_weights = split_rows["fit"]
    fitted = fit_rmsnorm_diagonal(
        fit_child,
        fit_target,
        epsilon=_final_norm_epsilon(target_model),
        sample_weights=fit_weights,
        ridge=ridge,
    )
    parameter = getattr(
        getattr(
            _model_core(target_model, "target_model"),
            "embedding_norm",
            None,
        ),
        "weight",
        None,
    )
    if not isinstance(parameter, torch.nn.Parameter) or tuple(parameter.shape) != tuple(
        fitted.weight.shape
    ):
        raise Lfm2CpcfModelFoldError(
            "Target final RMSNorm differs from the fitted target gain."
        )
    identity = (
        id(parameter),
        int(parameter.data_ptr()),
        tuple(parameter.shape),
    )
    with torch.no_grad():
        parameter.copy_(
            fitted.weight.to(device=parameter.device, dtype=parameter.dtype)
        )
    if identity != (
        id(parameter),
        int(parameter.data_ptr()),
        tuple(parameter.shape),
    ):
        raise Lfm2CpcfModelFoldError(
            "Final RMSNorm installation replaced a target Parameter."
        )
    metrics = {}
    for split, (child, target, weights) in split_rows.items():
        prediction = fitted.predict(child)
        metrics[split] = {
            "rows": int(child.shape[0]),
            "relative_rmse": _relative_rmse(
                prediction,
                target,
                weights,
            ),
            "retained_energy_fraction": _retained_energy(
                prediction,
                target,
                weights,
            ),
        }
    body = {
        "component": "target_coordinate_final_rmsnorm",
        "fit_report_sha256": fitted.report["report_sha256"],
        "fit": metrics["fit"],
        "validation": metrics["validation"],
        "fit_validation_disjoint": True,
        "source_gain_projection_used": False,
        "parameter_identity_preserved": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


@dataclass(frozen=True, slots=True)
class Lfm2CpcfModelFoldPolicy:
    """Whole-model coordinate, gate, and promotion controls."""

    ridge: float = 1e-8
    state_reconstruction_weight: float = 0.05
    minimum_predictive_energy_fraction: float = 0.90
    minimum_validation_block_retention: float = 0.0
    enforce_validation_block_floor: bool = True
    short_conv_gate_policy: Lfm2CpcfShortConvGatePolicy | None = None
    short_conv_v4_solver_policy: Lfm2CpcfShortConvV4SolverPolicy | None = None
    promotion_compression_ratio: float = 1.0
    require_promotion_ratio: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ridge",
            _nonnegative_float(self.ridge, "ridge"),
        )
        object.__setattr__(
            self,
            "state_reconstruction_weight",
            _nonnegative_float(
                self.state_reconstruction_weight,
                "state_reconstruction_weight",
            ),
        )
        object.__setattr__(
            self,
            "minimum_predictive_energy_fraction",
            _fraction(
                self.minimum_predictive_energy_fraction,
                "minimum_predictive_energy_fraction",
            ),
        )
        object.__setattr__(
            self,
            "minimum_validation_block_retention",
            _fraction(
                self.minimum_validation_block_retention,
                "minimum_validation_block_retention",
            ),
        )
        if not isinstance(self.enforce_validation_block_floor, bool):
            raise Lfm2CpcfModelFoldError(
                "enforce_validation_block_floor must be boolean."
            )
        if self.short_conv_gate_policy is not None and not isinstance(
            self.short_conv_gate_policy,
            Lfm2CpcfShortConvGatePolicy,
        ):
            raise TypeError(
                "short_conv_gate_policy must be Lfm2CpcfShortConvGatePolicy."
            )
        if self.short_conv_v4_solver_policy is not None:
            if not isinstance(
                self.short_conv_v4_solver_policy,
                Lfm2CpcfShortConvV4SolverPolicy,
            ):
                raise TypeError(
                    "short_conv_v4_solver_policy must be "
                    "Lfm2CpcfShortConvV4SolverPolicy."
                )
            if not self.short_conv_v4_solver_policy.verify_authentication():
                raise Lfm2CpcfModelFoldError(
                    "short_conv_v4_solver_policy authentication failed."
                )
        promotion = _nonnegative_float(
            self.promotion_compression_ratio,
            "promotion_compression_ratio",
        )
        if promotion <= 0.0:
            raise Lfm2CpcfModelFoldError(
                "promotion_compression_ratio must be positive."
            )
        object.__setattr__(
            self,
            "promotion_compression_ratio",
            promotion,
        )
        if not isinstance(self.require_promotion_ratio, bool):
            raise Lfm2CpcfModelFoldError("require_promotion_ratio must be boolean.")

    def resolved_short_conv_policy(self) -> Lfm2CpcfShortConvGatePolicy:
        if self.short_conv_gate_policy is not None:
            return self.short_conv_gate_policy
        return Lfm2CpcfShortConvGatePolicy()

    def as_dict(self) -> dict[str, Any]:
        result = {
            "ridge": self.ridge,
            "state_reconstruction_weight": (self.state_reconstruction_weight),
            "minimum_predictive_energy_fraction": (
                self.minimum_predictive_energy_fraction
            ),
            "minimum_validation_block_retention": (
                self.minimum_validation_block_retention
            ),
            "enforce_validation_block_floor": (self.enforce_validation_block_floor),
            "short_conv_gate_policy": (self.resolved_short_conv_policy().as_dict()),
            "promotion_compression_ratio": (self.promotion_compression_ratio),
            "require_promotion_ratio": self.require_promotion_ratio,
        }
        if self.short_conv_v4_solver_policy is not None:
            result["short_conv_v4_solver_policy"] = (
                self.short_conv_v4_solver_policy.as_dict()
            )
        return result


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCompiledModelFold:
    """One completely installed and sequentially replayed LFM2 child."""

    boundary_bundle: Lfm2CpcfBoundaryBundle
    layer_results: tuple[
        Lfm2CpcfCompletedLayerReceipt | Lfm2CpcfCompiledLayerResult,
        ...,
    ]
    final_state_bank: Lfm2CpcfPhaseStateBank
    parameter_receipt: Mapping[str, Any]
    interface_install_report: Mapping[str, Any]
    neutral_norm_install_report: Mapping[str, Any]
    final_norm_report: Mapping[str, Any]
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_bundle, Lfm2CpcfBoundaryBundle):
            raise TypeError("boundary_bundle has the wrong type.")
        layers = tuple(self.layer_results)
        if (
            not layers
            or any(
                not isinstance(
                    result,
                    (
                        Lfm2CpcfCompletedLayerReceipt,
                        Lfm2CpcfCompiledLayerResult,
                    ),
                )
                for result in layers
            )
            or tuple(result.layer_index for result in layers)
            != tuple(range(len(layers)))
        ):
            raise Lfm2CpcfModelFoldError("layer_results are incomplete or unordered.")
        if not isinstance(
            self.final_state_bank, Lfm2CpcfPhaseStateBank
        ) or self.final_state_bank.layer_index != len(layers):
            raise Lfm2CpcfModelFoldError(
                "Final state bank does not follow every target layer."
            )
        result_sha = _digest(self.result_sha256, "result_sha256")
        report = dict(self.report)
        if (
            report.pop("result_sha256", None) != result_sha
            or _sha(report) != result_sha
        ):
            raise Lfm2CpcfModelFoldError("Compiled model-fold report seal is invalid.")
        object.__setattr__(self, "layer_results", layers)
        for name in (
            "parameter_receipt",
            "interface_install_report",
            "neutral_norm_install_report",
            "final_norm_report",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(getattr(self, name))),
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "result_sha256": result_sha}),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _compile_lfm2_cpcf_model_fold_mutating(
    source_model: Any,
    target_model: Any,
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
    blank_target_report: Mapping[str, Any],
    planned_target_unique_parameters: int,
    policy: Lfm2CpcfModelFoldPolicy | None = None,
    resume_state: Lfm2CpcfModelFoldResumeState | None = None,
    progress_observer: Lfm2CpcfModelFoldProgressObserver | None = None,
    progress: _ModelFoldProgress,
) -> Lfm2CpcfCompiledModelFold:
    """Mutating implementation guarded by the public whole-model transaction."""

    progress.stage = "validate_request"
    resolved_policy = policy or Lfm2CpcfModelFoldPolicy()
    if not isinstance(resolved_policy, Lfm2CpcfModelFoldPolicy):
        raise TypeError("policy must be Lfm2CpcfModelFoldPolicy.")
    if resume_state is not None and not isinstance(
        resume_state,
        Lfm2CpcfModelFoldResumeState,
    ):
        raise TypeError("resume_state must be Lfm2CpcfModelFoldResumeState.")
    if progress_observer is not None and not callable(progress_observer):
        raise TypeError("progress_observer must be callable.")
    ordered_phases = tuple(phases)
    if not ordered_phases or any(
        not isinstance(phase, Lfm2CpcfTargetPhase) for phase in ordered_phases
    ):
        raise Lfm2CpcfModelFoldError(
            "phases must contain aligned Lfm2CpcfTargetPhase evidence."
        )
    progress.stage = "inventory_and_blank_authentication"
    target_inventory_before = independent_parameter_inventory(target_model)
    target_count = int(target_inventory_before["unique_parameters"])
    planned = _positive_int(
        planned_target_unique_parameters,
        "planned_target_unique_parameters",
    )
    if target_count != planned:
        raise Lfm2CpcfModelFoldError(
            "Live target count differs from the planned dense blank."
        )
    blank = _blank_receipt(
        blank_target_report,
        target_unique_parameters=target_count,
    )
    if resume_state is None:
        _assert_target_is_zero_blank(target_model)
    source_layers = tuple(_model_core(source_model, "source_model").layers)
    target_layers = tuple(_model_core(target_model, "target_model").layers)
    if not source_layers or len(source_layers) != len(target_layers):
        raise Lfm2CpcfModelFoldError(
            "Source and target must expose equal nonzero LFM2 depth."
        )
    if len(ordered_phases[0].parent_evidence.layers) != len(source_layers):
        raise Lfm2CpcfModelFoldError(
            "Cached phase depth differs from the live source model."
        )
    source_state_before = fingerprint_model_state(source_model)
    progress.stage = "parameter_compaction_audit"
    parameter_receipt = audit_lfm2_cpcf_parameter_compaction(
        source_model,
        target_model,
        planned_target_unique_parameters=planned,
        promotion_compression_ratio=(resolved_policy.promotion_compression_ratio),
    )
    if (
        resolved_policy.require_promotion_ratio
        and not parameter_receipt["promotion_ratio_met"]
    ):
        raise Lfm2CpcfModelFoldError(
            "Target is a valid diagnostic fold but does not meet the caller's "
            "promotion compression ratio."
        )
    progress.stage = "boundary_bundle_fit"
    authenticated_resume: Lfm2CpcfModelFoldResumeState | None = None
    if resume_state is None:
        boundary_bundle = fit_lfm2_cpcf_boundary_bundle(
            ordered_phases,
            source_tied_table,
            target_hidden_width=int(target_model.config.hidden_size),
            target_norm_epsilon=_final_norm_epsilon(target_model),
            state_reconstruction_weight=(resolved_policy.state_reconstruction_weight),
            ridge=resolved_policy.ridge,
            minimum_predictive_energy_fraction=(
                resolved_policy.minimum_predictive_energy_fraction
            ),
            minimum_validation_block_retention=(
                resolved_policy.minimum_validation_block_retention
            ),
            enforce_validation_block_floor=(
                resolved_policy.enforce_validation_block_floor
            ),
        )
    else:
        authenticated_resume = _authenticate_model_fold_resume_state(
            resume_state,
            target_model=target_model,
            phases=ordered_phases,
            policy=resolved_policy,
        )
        boundary_bundle = authenticated_resume.boundary_bundle
    progress.boundary_bundle = boundary_bundle
    progress.boundary_bundle_inventory_sha256 = _boundary_bundle_inventory_sha256(
        boundary_bundle
    )
    initial_state_bank: Lfm2CpcfPhaseStateBank | None = None
    if authenticated_resume is not None:
        progress.stage = "seed_incoming_bank"
        initial_state_bank = seed_lfm2_cpcf_phase_states(
            target_model,
            ordered_phases,
        )
        _validate_resume_receipt_chain(
            authenticated_resume,
            initial_state_bank=initial_state_bank,
            phases=ordered_phases,
        )
        progress.completed_layer_results.extend(
            authenticated_resume.completed_layer_receipts
        )
        progress.current_incoming_bank = authenticated_resume.incoming_state_bank
    progress.stage = "interface_install"
    interface_install = install_lfm2_cpcf_tied_interface(
        target_model,
        boundary_bundle.interface,
    )
    progress.stage = "neutral_norm_install"
    neutral_norm_install = install_lfm2_cpcf_neutral_norms(
        target_model,
        attention_qk_norm_weights=_preserved_qk_norm_weights(
            source_model,
            target_model,
        ),
    )
    if authenticated_resume is None:
        progress.stage = "seed_incoming_bank"
        state_bank = seed_lfm2_cpcf_phase_states(
            target_model,
            ordered_phases,
        )
        layer_results: list[
            Lfm2CpcfCompletedLayerReceipt | Lfm2CpcfCompiledLayerResult
        ] = []
        first_layer_index = 0
    else:
        if (
            fingerprint_model_state(target_model)
            != authenticated_resume.prefix_install_report[
                "installed_state_sha256_after"
            ]
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Reinstalling the authenticated interface and neutral norms "
                "changed the installed prefix."
            )
        state_bank = authenticated_resume.incoming_state_bank
        layer_results = list(authenticated_resume.completed_layer_receipts)
        first_layer_index = authenticated_resume.next_layer_index
    progress.current_incoming_bank = state_bank
    if authenticated_resume is None:
        _emit_model_fold_progress(
            progress_observer,
            target_model=target_model,
            boundary_bundle=boundary_bundle,
            state_bank=state_bank,
            resolved_policy=resolved_policy,
            progress=progress,
        )
    short_conv_policy = resolved_policy.resolved_short_conv_policy()
    for index in range(first_layer_index, len(boundary_bundle.layers)):
        charts = boundary_bundle.layers[index]
        progress.stage = "layer_compile_install"
        progress.active_layer_index = index
        progress.active_operator_kind = charts.operator_kind
        if index != state_bank.layer_index:
            raise Lfm2CpcfModelFoldInstallError(
                "Sequential target replay attempted to skip a layer."
            )
        result = compile_and_install_lfm2_cpcf_layer(
            source_model,
            target_model,
            layer_index=index,
            input_bank=state_bank,
            boundary_charts=charts,
            ridge=resolved_policy.ridge,
            short_conv_gate_policy=short_conv_policy,
            short_conv_v4_solver_policy=(resolved_policy.short_conv_v4_solver_policy),
        )
        if result.layer_index != index:
            raise Lfm2CpcfModelFoldInstallError(
                "Layer compiler returned an out-of-order result."
            )
        layer_results.append(result)
        progress.completed_layer_results.append(result)
        state_bank = result.next_bank
        progress.current_incoming_bank = state_bank
        _emit_model_fold_progress(
            progress_observer,
            target_model=target_model,
            boundary_bundle=boundary_bundle,
            state_bank=state_bank,
            resolved_policy=resolved_policy,
            progress=progress,
        )
    if len(layer_results) != len(target_layers):
        raise Lfm2CpcfModelFoldInstallError(
            "Not every target macrocell was compiled and replayed."
        )
    progress.active_layer_index = None
    progress.active_operator_kind = None
    progress.stage = "final_norm_install"
    final_norm_report = _fit_and_install_final_norm(
        target_model,
        phases=ordered_phases,
        final_state_bank=state_bank,
        boundary_bundle=boundary_bundle,
        ridge=resolved_policy.ridge,
    )
    progress.stage = "final_audit"
    target_inventory_after = independent_parameter_inventory(target_model)
    if (
        int(target_inventory_after["unique_parameters"]) != target_count
        or fingerprint_model_state(source_model) != source_state_before
        or _boundary_bundle_inventory_sha256(boundary_bundle)
        != progress.boundary_bundle_inventory_sha256
    ):
        raise Lfm2CpcfModelFoldInstallError(
            "Model topology changed or source state was mutated during folding."
        )
    progress.stage = "result_seal"
    body = {
        "schema": LFM2_CPCF_MODEL_FOLD_SCHEMA,
        "blank_target_report_sha256": blank["report_sha256"],
        "policy": resolved_policy.as_dict(),
        "phase_count": len(ordered_phases),
        "phase_sha256": [phase.phase_sha256 for phase in ordered_phases],
        "boundary_bundle_sha256": boundary_bundle.bundle_sha256,
        "interface_install_report_sha256": interface_install["report_sha256"],
        "neutral_norm_install_report_sha256": neutral_norm_install["report_sha256"],
        "ordered_layer_result_sha256": [
            result.result_sha256 for result in layer_results
        ],
        "final_state_bank_sha256": state_bank.state_bank_sha256,
        "final_norm_report_sha256": final_norm_report["report_sha256"],
        "parameter_receipt": dict(parameter_receipt),
        "target_parameter_inventory_before": dict(target_inventory_before),
        "target_parameter_inventory_after": dict(target_inventory_after),
        "source_state_preserved": True,
        "target_parameter_count_preserved": True,
        "all_layers_compiled_in_order": True,
        "actual_folded_upstream_replay_used": True,
        "operation_audit": {
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
        },
    }
    result_sha = _sha(body)
    return Lfm2CpcfCompiledModelFold(
        boundary_bundle=boundary_bundle,
        layer_results=tuple(layer_results),
        final_state_bank=state_bank,
        parameter_receipt=parameter_receipt,
        interface_install_report=interface_install,
        neutral_norm_install_report=neutral_norm_install,
        final_norm_report=final_norm_report,
        report={**body, "result_sha256": result_sha},
        result_sha256=result_sha,
    )


def compile_lfm2_cpcf_model_fold(
    source_model: Any,
    target_model: Any,
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
    blank_target_report: Mapping[str, Any],
    planned_target_unique_parameters: int,
    policy: Lfm2CpcfModelFoldPolicy | None = None,
    resume_state: Lfm2CpcfModelFoldResumeState | None = None,
    progress_observer: Lfm2CpcfModelFoldProgressObserver | None = None,
) -> Lfm2CpcfCompiledModelFold:
    """Compile every target layer as one rollback-safe model transaction."""

    if resume_state is not None and not isinstance(
        resume_state,
        Lfm2CpcfModelFoldResumeState,
    ):
        raise TypeError("resume_state must be Lfm2CpcfModelFoldResumeState.")
    if progress_observer is not None and not callable(progress_observer):
        raise TypeError("progress_observer must be callable.")
    authenticated_phases = tuple(phases)
    fold_inputs_before = _model_fold_inputs_sha256(
        authenticated_phases,
        source_tied_table,
    )
    resume_inputs_before = _model_fold_resume_inputs_sha256(resume_state)
    source_before = _snapshot_model_auth(source_model)
    target_before = _snapshot_model(target_model)
    progress = _ModelFoldProgress()
    try:
        result = _compile_lfm2_cpcf_model_fold_mutating(
            source_model,
            target_model,
            phases=phases,
            source_tied_table=source_tied_table,
            blank_target_report=blank_target_report,
            planned_target_unique_parameters=(planned_target_unique_parameters),
            policy=policy,
            resume_state=resume_state,
            progress_observer=progress_observer,
            progress=progress,
        )
        _authenticate_model(source_model, source_before)
        if (
            _model_fold_inputs_sha256(
                authenticated_phases,
                source_tied_table,
            )
            != fold_inputs_before
            or (_model_fold_resume_inputs_sha256(resume_state) != resume_inputs_before)
            or (
                progress.boundary_bundle is not None
                and _boundary_bundle_inventory_sha256(progress.boundary_bundle)
                != progress.boundary_bundle_inventory_sha256
            )
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Successful fold changed phases, cached evidence, source "
                "table, or boundary-bundle evidence."
            )
        if (
            _sha(
                _model_inventory(
                    target_model,
                    include_tensor_bytes=False,
                )
            )
            != target_before.structure_inventory_sha256
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Successful fold changed target tensor identities, aliases, "
                "registries, gradients, buffer persistence, or modes."
            )
        return result
    except BaseException as error:
        failed_prefix_snapshot: _ModelSnapshot | None = None
        prefix_snapshot_error: BaseException | None = None
        try:
            failed_prefix_snapshot = _snapshot_model(target_model)
        except BaseException as snapshot_exc:
            prefix_snapshot_error = snapshot_exc
        try:
            _restore_model(target_model, target_before)
            rollback_proof = _prove_exact_whole_model_rollback(
                source_model,
                target_model,
                source_snapshot_before=source_before,
                target_snapshot_before=target_before,
            )
        except BaseException as rollback_exc:
            raise Lfm2CpcfModelFoldInstallError(
                "Whole-model CPCF compilation failed and exact target "
                "rollback could not be proven."
            ) from rollback_exc
        try:
            fold_inputs_preserved = (
                _model_fold_inputs_sha256(
                    authenticated_phases,
                    source_tied_table,
                )
                == fold_inputs_before
            )
            boundary_preserved = progress.boundary_bundle is None or (
                _boundary_bundle_inventory_sha256(progress.boundary_bundle)
                == progress.boundary_bundle_inventory_sha256
            )
            resume_inputs_preserved = (
                _model_fold_resume_inputs_sha256(resume_state) == resume_inputs_before
            )
        except BaseException as evidence_exc:
            raise Lfm2CpcfModelFoldInstallError(
                "Whole-model target rollback succeeded, but fold-input "
                "evidence could not be reauthenticated."
            ) from evidence_exc
        if (
            not fold_inputs_preserved
            or not resume_inputs_preserved
            or not boundary_preserved
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Whole-model target rollback succeeded, but phases, cached "
                "evidence, source table, resume state, or boundary evidence "
                "changed."
            ) from error
        if not isinstance(error, Exception):
            try:
                error.add_note(
                    "LFM2 CPCF whole-model target was exactly restored "
                    "before cancellation propagated."
                )
            except AttributeError:  # pragma: no cover - Python boundary
                pass
            raise
        if prefix_snapshot_error is not None:
            raise Lfm2CpcfModelFoldInstallError(
                "Whole-model CPCF compilation failed; exact target rollback "
                "was proven, but the failed-prefix diagnostic snapshot "
                "could not be captured."
            ) from prefix_snapshot_error
        if failed_prefix_snapshot is None:  # pragma: no cover - defensive
            raise Lfm2CpcfModelFoldInstallError(
                "Whole-model rollback succeeded without prefix evidence."
            ) from error
        failure = _wrap_model_fold_failure(
            error,
            progress=progress,
            failed_prefix_snapshot=failed_prefix_snapshot,
            rollback_proof=rollback_proof,
        )
        raise failure from error


__all__ = [
    "LFM2_CPCF_FAILURE_INCOMING_BANK_SNAPSHOT_SCHEMA",
    "LFM2_CPCF_MODEL_FOLD_SCHEMA",
    "LFM2_CPCF_MODEL_FAILURE_SCHEMA",
    "LFM2_CPCF_MODEL_PROGRESS_EVENT_SCHEMA",
    "LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA",
    "LFM2_CPCF_PREFIX_INSTALL_REPORT_SCHEMA",
    "LFM2_CPCF_TARGET_TOPOLOGY_SNAPSHOT_SCHEMA",
    "Lfm2CpcfCompletedLayerReceipt",
    "Lfm2CpcfCompiledModelFold",
    "Lfm2CpcfFailureIncomingBankSnapshot",
    "Lfm2CpcfModelFoldFailure",
    "Lfm2CpcfModelFoldError",
    "Lfm2CpcfModelFoldInstallError",
    "Lfm2CpcfModelFoldPolicy",
    "Lfm2CpcfModelFoldProgressEvent",
    "Lfm2CpcfModelFoldProgressObserver",
    "Lfm2CpcfModelFoldResumeState",
    "audit_lfm2_cpcf_parameter_compaction",
    "compile_lfm2_cpcf_model_fold",
    "independent_parameter_inventory",
]
