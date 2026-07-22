"""Authenticated, nonpromotable LFM2 semantic preterminal seam capsules.

The capsule freezes the expensive semantic layer-stack result immediately
after its attention receipts authenticate and before the fused terminal solve.
It contains a strict-local dense child checkpoint, the four typed final-layer
MLP evidence banks, and the active tied-interface coordinates.  It is resume
evidence only: it makes no capability, acceptance, or promotion claim.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_artifact import (
    canonical_standalone_fold_artifact_inventory,
    scan_standalone_fold_artifact_tree,
)
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_model_fold import independent_parameter_inventory
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    _atomic_rename_noreplace,
    _canonical_mapping,
    _deep_freeze,
    _deep_thaw,
    _file_record,
    _fsync_directory,
    _is_link_or_junction,
    _lexists,
    _load_safetensors_path,
    _normalize_tensors,
    _read_canonical_json_with_raw,
    _real_directory,
    _regular_file,
    _save_safetensors,
    _stat_identity,
    _write_canonical_json,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
    Lfm2CpcfTerminalActionEvidence,
    _tensor_sha256 as _terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
    LFM2_CPCF_V5_ARTIFACT_METADATA,
    Lfm2CpcfV5QuarantineSaver,
    Lfm2CpcfV5StrictLocalLoader,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    Lfm2SemanticLayerStackInstallation,
    _ffn_width,
    _final_w2_parameter,
    _interface_tensor_sha256,
    _lfm2_dense_producer_chain_coverage,
    _parameter_count,
    _tensor_sha256,
    _verify_sealed,
    authenticate_lfm2_semantic_layer_stack_attention_receipts,
)
from neural_forge.hf_tied_causal_interface_fold import TiedCausalInterfaceFold
from neural_forge.hf_tied_residual_congruence import (
    TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
    TiedResidualAnchorTransportWitness,
    authenticate_tied_residual_congruence_interface,
)
from neural_forge.hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics


LFM2_SEMANTIC_PRETERMINAL_CAPSULE_BINDINGS_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-bindings-v1"
)
LFM2_SEMANTIC_PRETERMINAL_CAPSULE_CONTEXT_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-context-v1"
)
LFM2_SEMANTIC_PRETERMINAL_CAPSULE_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-manifest-v1"
)
LFM2_SEMANTIC_PRETERMINAL_CAPSULE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-receipt-v1"
)

_ARTIFACT_KIND = "lfm2-semantic-preterminal-capsule"
_PAYLOAD_NAME = "preterminal-seam.safetensors"
_MANIFEST_NAME = "capsule-manifest.json"
_RECEIPT_NAME = "capsule-receipt.json"
_CHECKPOINT_NAME = "checkpoint"
_ROOT_INVENTORY = frozenset(
    {_CHECKPOINT_NAME, _PAYLOAD_NAME, _MANIFEST_NAME, _RECEIPT_NAME}
)
_SPLITS = ("fit", "fit_check", "selection", "validation")
_EVIDENCE_TENSOR_ROLES = (
    "mlp_input",
    "pre_mlp_residual",
    "parent_final_norm_output",
    "sample_weights",
)
_INTERFACE_TENSOR_NAMES = (
    "interface.target_weight",
    "interface.producer_encoder",
)
_ANCHOR_TRANSPORT_TENSOR_NAMES = (
    "anchor_source.target_weight",
    "anchor_source.producer_encoder",
    "anchor_base.target_weight",
    "anchor_source.residual_weight",
)
_DOWN_WEIGHT_PRIOR_TENSOR_NAME = "terminal_prior.final_layer_mlp_w2"
_NONPROMOTABLE_FLAGS = {
    "status": "quarantined-preterminal-resume-only",
    "terminal_complete": False,
    "promotion_authorized": False,
    "accepted_artifact": False,
    "behavior_capability_proven": False,
    "training_performed": False,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}
_ZERO_OPERATION_AUDIT = {
    "parent_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "child_model_forward_calls": 0,
    "model_generate_calls": 0,
    "trace_recaptures": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "network_access_performed": False,
    "pickle_files_written": 0,
}


class Lfm2SemanticPreterminalCapsuleError(RuntimeError):
    """A preterminal capsule could not be authenticated or published."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _optional_digest(value: Any, name: str) -> str | None:
    return None if value is None else _digest(value, name)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticPreterminalCapsuleBindings:
    """Externally rooted identities needed to interpret one seam capsule."""

    input_artifact_sha256: str
    compiler_diagnostic_sha256: str
    compiler_cache_key: str
    semantic_cache_key: str
    semantic_cache_manifest_sha256: str
    cache_parent_fingerprint: str
    bridge_artifact_sha256: str
    mapped_parent_tied_table_receipt_sha256: str
    parent_final_norm_receipt_sha256: str
    interface_rehydration_sha256: str
    tied_residual_interface_gauge_sha256: str
    cache_selective_session_sha256: str
    interface_statistics_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_artifact_sha256",
            "compiler_diagnostic_sha256",
            "compiler_cache_key",
            "semantic_cache_key",
            "semantic_cache_manifest_sha256",
            "cache_parent_fingerprint",
            "bridge_artifact_sha256",
            "mapped_parent_tied_table_receipt_sha256",
            "parent_final_norm_receipt_sha256",
            "interface_rehydration_sha256",
            "tied_residual_interface_gauge_sha256",
            "cache_selective_session_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "interface_statistics_artifact_sha256",
            _optional_digest(
                self.interface_statistics_artifact_sha256,
                "interface_statistics_artifact_sha256",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_PRETERMINAL_CAPSULE_BINDINGS_SCHEMA,
            "input_artifact_sha256": self.input_artifact_sha256,
            "compiler_diagnostic_sha256": self.compiler_diagnostic_sha256,
            "compiler_cache_key": self.compiler_cache_key,
            "semantic_cache_key": self.semantic_cache_key,
            "semantic_cache_manifest_sha256": self.semantic_cache_manifest_sha256,
            "cache_parent_fingerprint": self.cache_parent_fingerprint,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "mapped_parent_tied_table_receipt_sha256": (
                self.mapped_parent_tied_table_receipt_sha256
            ),
            "parent_final_norm_receipt_sha256": (self.parent_final_norm_receipt_sha256),
            "interface_rehydration_sha256": self.interface_rehydration_sha256,
            "tied_residual_interface_gauge_sha256": (
                self.tied_residual_interface_gauge_sha256
            ),
            "cache_selective_session_sha256": self.cache_selective_session_sha256,
            "interface_statistics_artifact_sha256": (
                self.interface_statistics_artifact_sha256
            ),
        }

    @property
    def bindings_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "Lfm2SemanticPreterminalCapsuleBindings":
        normalized = _canonical_mapping(value, "preterminal capsule bindings")
        if normalized.pop("schema", None) != (
            LFM2_SEMANTIC_PRETERMINAL_CAPSULE_BINDINGS_SCHEMA
        ):
            raise Lfm2SemanticPreterminalCapsuleError(
                "Preterminal capsule binding schema is invalid."
            )
        try:
            return cls(**normalized)
        except TypeError as exc:
            raise Lfm2SemanticPreterminalCapsuleError(
                "Preterminal capsule bindings have missing or extra fields."
            ) from exc


@dataclass(frozen=True, slots=True)
class Lfm2SemanticPreterminalCapsuleReference:
    path: Path
    receipt_sha256: str
    receipt_file_sha256: str
    manifest_sha256: str
    bindings_sha256: str
    context_binding_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "receipt_sha256",
            "receipt_file_sha256",
            "manifest_sha256",
            "bindings_sha256",
            "context_binding_sha256",
        ):
            _digest(getattr(self, name), name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "receipt_sha256": self.receipt_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "bindings_sha256": self.bindings_sha256,
            "context_binding_sha256": self.context_binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticPreterminalCapsule:
    path: Path
    target_model: Any = field(repr=False, compare=False)
    target_tokenizer: Any = field(repr=False, compare=False)
    layer_stack: Lfm2SemanticLayerStackInstallation = field(
        repr=False,
        compare=False,
    )
    interface: TiedCausalInterfaceFold = field(repr=False, compare=False)
    attention_receipt_authentication: Mapping[str, Any]
    bindings: Lfm2SemanticPreterminalCapsuleBindings
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    reference: Lfm2SemanticPreterminalCapsuleReference
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def _sealed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if field_name in body:
        raise Lfm2SemanticPreterminalCapsuleError(
            f"Capsule body already contains {field_name!r}."
        )
    normalized = _canonical_mapping(body, "capsule receipt body")
    return {**normalized, field_name: canonical_sha256(normalized)}


def _verify_seal(
    value: Mapping[str, Any],
    *,
    field_name: str,
    label: str,
) -> dict[str, Any]:
    normalized = _canonical_mapping(value, label)
    body = dict(normalized)
    supplied = body.pop(field_name, None)
    if supplied != canonical_sha256(body):
        raise Lfm2SemanticPreterminalCapsuleError(f"{label} self-seal is invalid.")
    return normalized


def _expected_payload_names(
    *,
    anchor_transport: bool = False,
    down_weight_prior: bool = False,
) -> tuple[str, ...]:
    return tuple(
        [
            f"evidence.{split}.{role}"
            for split in _SPLITS
            for role in _EVIDENCE_TENSOR_ROLES
        ]
        + list(_INTERFACE_TENSOR_NAMES)
        + (list(_ANCHOR_TRANSPORT_TENSOR_NAMES) if anchor_transport else [])
        + ([_DOWN_WEIGHT_PRIOR_TENSOR_NAME] if down_weight_prior else [])
    )


def _payload_tensors(
    layer_stack: Lfm2SemanticLayerStackInstallation,
    interface: TiedCausalInterfaceFold,
    *,
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
) -> dict[str, Any]:
    tensors = {
        f"evidence.{split}.{role}": getattr(
            layer_stack.final_layer_mlp_evidence[split],
            role,
        )
        for split in _SPLITS
        for role in _EVIDENCE_TENSOR_ROLES
    }
    tensors.update(
        {
            "interface.target_weight": interface.target_weight,
            "interface.producer_encoder": interface.producer_encoder,
        }
    )
    if anchor_transport_witness is not None:
        if not isinstance(
            anchor_transport_witness,
            TiedResidualAnchorTransportWitness,
        ):
            raise TypeError(
                "anchor_transport_witness must be a typed transport witness."
            )
        anchor_source_interface = (
            anchor_transport_witness.materialize_source_interface()
        )
        tensors.update(
            {
                "anchor_source.target_weight": anchor_source_interface.target_weight,
                "anchor_source.producer_encoder": (
                    anchor_source_interface.producer_encoder
                ),
                "anchor_base.target_weight": (
                    anchor_transport_witness.anchor_base_target_weight
                ),
                "anchor_source.residual_weight": (
                    anchor_transport_witness.anchor_source_residual_weight
                ),
            }
        )
    prior = layer_stack.final_layer_mlp_down_weight_prior
    if prior is not None:
        if (
            not isinstance(prior, Lfm2CpcfTerminalActionDownWeightPrior)
            or not prior.verify_authentication()
        ):
            raise Lfm2SemanticPreterminalCapsuleError(
                "Preterminal layer stack has an invalid final-W2 prior."
            )
        tensors[_DOWN_WEIGHT_PRIOR_TENSOR_NAME] = prior.down_weight
    return tensors


def _validate_nonpromotable(value: Mapping[str, Any], *, label: str) -> None:
    if any(
        value.get(name) != expected for name, expected in _NONPROMOTABLE_FLAGS.items()
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            f"{label} is not fixed nonpromotable resume evidence."
        )


def _root_path(value: str | os.PathLike[str]) -> tuple[Path, tuple[int, int]]:
    lexical = Path(value).expanduser().absolute()
    if _is_link_or_junction(lexical):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule root cannot be a symlink or junction."
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule root is missing."
        ) from exc
    if resolved != lexical:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule path cannot traverse a link."
        )
    root_stat = _real_directory(resolved, name="preterminal capsule root")
    try:
        entries = tuple(os.scandir(resolved))
    except OSError as exc:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule root cannot be enumerated."
        ) from exc
    if {entry.name for entry in entries} != _ROOT_INVENTORY:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule has missing or unexpected root entries."
        )
    for entry in entries:
        path = resolved / entry.name
        if entry.name == _CHECKPOINT_NAME:
            _real_directory(path, name="preterminal dense checkpoint")
        else:
            _regular_file(path, name=entry.name)
    return resolved, _stat_identity(root_stat)


def _recheck_root(root: Path, identity: tuple[int, int]) -> None:
    current, current_identity = _root_path(root)
    if current != root or current_identity != identity:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule root changed during authentication."
        )


def _new_output(value: str | os.PathLike[str]) -> Path:
    output = Path(value).expanduser().absolute()
    if _lexists(output):
        raise Lfm2SemanticPreterminalCapsuleError(
            f"Refusing to replace existing preterminal capsule {output}."
        )
    parent = output.parent
    _real_directory(parent, name="preterminal capsule parent")
    if _is_link_or_junction(parent):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule parent cannot be a link."
        )
    if parent.resolve(strict=True) != parent:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule parent path cannot traverse a link."
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
        or not stage.name.startswith(".semantic-preterminal-")
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Refusing to remove unauthenticated preterminal staging."
        )
    shutil.rmtree(stage)
    _fsync_directory(parent)


def _publish_quarantine_checkpoint(
    *,
    target_model: Any,
    target_tokenizer: Any,
    checkpoint: Path,
    context_binding_sha256: str,
    model_state_sha256: str,
    parameter_inventory_sha256: str,
    tokenizer_semantics_sha256: str,
    ffn_width: int,
) -> dict[str, Any]:
    checkpoint.mkdir()
    saver = Lfm2CpcfV5QuarantineSaver(
        target_tokenizer=target_tokenizer,
        context_binding_sha256=context_binding_sha256,
        expected_ffn_width=ffn_width,
        expected_model_state_sha256=model_state_sha256,
        expected_parameter_inventory_sha256=parameter_inventory_sha256,
        expected_tokenizer_semantics_sha256=tokenizer_semantics_sha256,
    )
    saver(target_model, checkpoint)
    metadata, _raw = _read_canonical_json_with_raw(
        checkpoint / LFM2_CPCF_V5_ARTIFACT_METADATA,
        name=LFM2_CPCF_V5_ARTIFACT_METADATA,
    )
    inventory = canonical_standalone_fold_artifact_inventory(checkpoint)
    roles = {row["path"]: row["artifact_role"] for row in inventory}
    tree = scan_standalone_fold_artifact_tree(
        checkpoint,
        expected_file_roles=roles,
    )
    return {
        "artifact_metadata": metadata,
        "artifact_inventory": list(inventory),
        "artifact_inventory_sha256": canonical_sha256(inventory),
        "artifact_tree": list(tree),
        "artifact_tree_sha256": canonical_sha256(tree),
    }


def _load_quarantine_checkpoint(
    *,
    checkpoint: Path,
    expected: Mapping[str, Any],
    context_binding_sha256: str,
    model_state_sha256: str,
    parameter_inventory_sha256: str,
    tokenizer_semantics_sha256: str,
    ffn_width: int,
    device: str,
    dtype: str,
) -> tuple[Any, Any]:
    _verify_quarantine_checkpoint_record(
        checkpoint=checkpoint,
        expected=expected,
    )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=context_binding_sha256,
        device=device,
        dtype=dtype,
        expected_ffn_width=ffn_width,
        expected_model_state_sha256=model_state_sha256,
        expected_parameter_inventory_sha256=parameter_inventory_sha256,
        expected_tokenizer_semantics_sha256=tokenizer_semantics_sha256,
    )
    model = loader(checkpoint)
    tokenizer = getattr(model, "_neural_forge_quarantine_tokenizer", None)
    if tokenizer is None:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Strict-local capsule reload omitted its tokenizer."
        )
    return model, tokenizer


def _verify_quarantine_checkpoint_record(
    *,
    checkpoint: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Rescan every dense-checkpoint byte without constructing the model."""

    metadata, _raw = _read_canonical_json_with_raw(
        checkpoint / LFM2_CPCF_V5_ARTIFACT_METADATA,
        name=LFM2_CPCF_V5_ARTIFACT_METADATA,
    )
    inventory = canonical_standalone_fold_artifact_inventory(checkpoint)
    roles = {row["path"]: row["artifact_role"] for row in inventory}
    tree = scan_standalone_fold_artifact_tree(
        checkpoint,
        expected_file_roles=roles,
    )
    if (
        metadata != expected.get("artifact_metadata")
        or list(inventory) != expected.get("artifact_inventory")
        or canonical_sha256(inventory) != expected.get("artifact_inventory_sha256")
        or list(tree) != expected.get("artifact_tree")
        or canonical_sha256(tree) != expected.get("artifact_tree_sha256")
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Dense checkpoint differs from the capsule manifest."
        )
    return {
        "artifact_metadata": metadata,
        "artifact_inventory": list(inventory),
        "artifact_inventory_sha256": canonical_sha256(inventory),
        "artifact_tree": list(tree),
        "artifact_tree_sha256": canonical_sha256(tree),
    }


def _validate_live_seam(
    *,
    target_model: Any,
    target_tokenizer: Any,
    layer_stack: Lfm2SemanticLayerStackInstallation,
    interface: TiedCausalInterfaceFold,
    attention_receipt_authentication: Mapping[str, Any],
    bindings: Lfm2SemanticPreterminalCapsuleBindings,
    mapped_parent_tied_table: Any | None = None,
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
    pinned_interface_authentication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError("PyTorch is required for preterminal capsules.") from exc
    if not isinstance(layer_stack, Lfm2SemanticLayerStackInstallation):
        raise TypeError("layer_stack must be Lfm2SemanticLayerStackInstallation.")
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    layer_receipt = _verify_sealed(
        layer_stack.receipt,
        seal_name="result_sha256",
        label="preterminal semantic layer stack",
    )
    if any(
        not layer_stack.final_layer_mlp_evidence[split].verify_authentication()
        for split in _SPLITS
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal final-layer evidence changed after authentication."
        )
    recomputed_attention = authenticate_lfm2_semantic_layer_stack_attention_receipts(
        layer_receipt
    )
    supplied_attention = _verify_sealed(
        attention_receipt_authentication,
        seal_name="result_sha256",
        label="preterminal attention receipt authentication",
    )
    if dict(recomputed_attention) != supplied_attention:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal attention receipt authentication changed."
        )
    hybrid = (
        interface.report.get("mode")
        == TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
    )
    if hybrid and not isinstance(
        anchor_transport_witness,
        TiedResidualAnchorTransportWitness,
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Hybrid preterminal seam omitted its typed anchor transport witness."
        )
    if not hybrid and anchor_transport_witness is not None:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Non-hybrid preterminal seam supplied an anchor transport witness."
        )
    if hybrid and mapped_parent_tied_table is None:
        if pinned_interface_authentication is None:
            raise Lfm2SemanticPreterminalCapsuleError(
                "Reloaded hybrid seam has no pinned exact transport authentication."
            )
        interface_authentication = _verify_sealed(
            pinned_interface_authentication,
            seal_name="result_sha256",
            label="preterminal hybrid interface authentication",
        )
        source_report = _verify_sealed(
            anchor_transport_witness.source_interface_report,
            seal_name="report_sha256",
            label="preterminal anchor source interface",
        )
        anchor_transport_receipt = layer_receipt.get(
            "anchor_residual_transport_authentication"
        )
        if (
            interface_authentication.get("interface_report_sha256")
            != interface.report.get("report_sha256")
            or interface_authentication.get(
                "anchor_residual_formula_recomputed_exact"
            )
            is not True
            or interface_authentication.get(
                "strong_base_recomputed_for_anchor_residual"
            )
            is not True
            or source_report.get("report_sha256")
            != interface.report.get("source_interface_report_sha256")
            or _interface_tensor_sha256(
                anchor_transport_witness.anchor_base_target_weight
            )
            != interface.report.get("anchor_base_target_weight_sha256")
            or _interface_tensor_sha256(
                anchor_transport_witness.anchor_source_residual_weight
            )
            != interface.report.get("anchor_source_target_residual_sha256")
            or interface.report.get("anchor_transport_witness_sha256")
            != anchor_transport_witness.witness_sha256
            or not isinstance(anchor_transport_receipt, Mapping)
            or dict(anchor_transport_receipt) != interface_authentication
            or layer_receipt.get(
                "anchor_residual_transport_authentication_sha256"
            )
            != interface_authentication["result_sha256"]
        ):
            raise Lfm2SemanticPreterminalCapsuleError(
                "Pinned hybrid transport proof or persisted algebraic inputs changed."
            )
    else:
        interface_authentication = authenticate_tied_residual_congruence_interface(
            interface,
            mapped_parent_tied_table=mapped_parent_tied_table,
            anchor_transport_witness=anchor_transport_witness,
        )
    lineage = layer_receipt.get("tied_interface_lineage")
    input_weight = getattr(target_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(target_model.get_output_embeddings(), "weight", None)
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("interface_report_sha256")
        != interface_authentication.get("interface_report_sha256")
        or lineage.get("producer_encoder_sha256")
        != _interface_tensor_sha256(interface.producer_encoder)
        or lineage.get("target_weight_sha256")
        != _interface_tensor_sha256(interface.target_weight)
        or not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or tuple(input_weight.shape) != tuple(interface.target_weight.shape)
        or not torch.equal(
            input_weight.detach().to(device="cpu"),
            interface.target_weight.to(device="cpu", dtype=input_weight.dtype),
        )
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Active tied interface differs from the layer stack or live child."
        )
    if (
        layer_receipt.get("cache_key") != bindings.semantic_cache_key
        or layer_receipt.get("cache_parent_fingerprint")
        != bindings.cache_parent_fingerprint
        or layer_receipt.get("bridge_artifact_sha256")
        != bindings.bridge_artifact_sha256
        or layer_receipt.get("cache_selective_session_sha256")
        != bindings.cache_selective_session_sha256
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Layer stack differs from the externally rooted cache/bridge bindings."
        )
    model_identity = fingerprint_lfm2_parent(target_model).identity_sha256
    if model_identity != layer_receipt.get("model_state_after_sha256"):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Live preterminal child differs from the layer-stack final state."
        )
    final_w2 = _final_w2_parameter(target_model)
    if (
        final_w2.dtype != torch.bfloat16
        or final_w2.layout != torch.strided
        or final_w2.device.type == "meta"
        or not bool(torch.isfinite(final_w2).all().item())
        or int(torch.count_nonzero(final_w2).item()) != 0
        or _tensor_sha256(final_w2) != layer_receipt.get("deferred_final_w2_sha256")
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule requires the exact authenticated zero BF16 final W2."
        )
    final_mlp_coverage = _lfm2_dense_producer_chain_coverage(
        target_model,
        layer_index=13,
        producer_kind="mlp",
        defer_down_weight=True,
    )
    if final_mlp_coverage.get(
        "all_required_roles_structurally_nonzero"
    ) is not True or final_mlp_coverage.get("deferred_roles") != [
        "feed_forward.w2.weight"
    ]:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Final-layer W1/W3 producer coverage is incomplete."
        )
    inventory = _canonical_parameter_inventory(target_model)
    inventory_sha = _digest(
        inventory.get("inventory_sha256"),
        "preterminal parameter inventory SHA-256",
    )
    parameter_count = _parameter_count(target_model)
    if parameter_count != layer_receipt.get("parameter_element_count"):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal parameter count differs from the layer-stack receipt."
        )
    result = {
        "model_identity_sha256": model_identity,
        "model_state_sha256": fingerprint_model_state(target_model),
        "parameter_inventory": inventory,
        "parameter_inventory_sha256": inventory_sha,
        "parameter_element_count": parameter_count,
        "tokenizer_semantics_sha256": fingerprint_hf_tokenizer_semantics(
            target_tokenizer
        ),
        "materialized_ffn_width": _ffn_width(target_model),
        "final_w2_parameter_name": "model.layers.13.feed_forward.w2.weight",
        "final_w2_semantic_sha256": _tensor_sha256(final_w2),
        "final_w2_terminal_action_sha256": _terminal_action_tensor_sha256(
            final_w2,
            "preterminal.final_w2",
        ),
        "final_w2_shape": [int(size) for size in final_w2.shape],
        "final_w2_dtype": "bfloat16",
        "final_w2_nonzero_element_count": 0,
        "final_w2_exact_zero": True,
        "final_mlp_producer_coverage": dict(final_mlp_coverage),
        "interface_authentication": dict(interface_authentication),
        "attention_receipt_authentication": supplied_attention,
    }
    prior = layer_stack.final_layer_mlp_down_weight_prior
    if prior is not None:
        result.update(
            {
                "final_layer_mlp_down_weight_prior": prior.as_dict(),
                "final_layer_mlp_down_weight_prior_authentication_sha256": (
                    prior.authentication_sha256
                ),
                "final_layer_mlp_down_weight_prior_tensor_sha256": (
                    _terminal_action_tensor_sha256(
                        prior.down_weight,
                        "preterminal.final_layer_mlp_down_weight_prior",
                    )
                ),
            }
        )
    return result


def _canonical_parameter_inventory(target_model: Any) -> dict[str, Any]:
    """Thaw the inventory's read-only top-level view before JSON sealing."""

    return _canonical_mapping(
        dict(independent_parameter_inventory(target_model)),
        "preterminal parameter inventory",
    )


def _context_body(
    *,
    bindings: Lfm2SemanticPreterminalCapsuleBindings,
    seam: Mapping[str, Any],
    layer_stack: Lfm2SemanticLayerStackInstallation,
    interface: TiedCausalInterfaceFold,
) -> dict[str, Any]:
    body = {
        "schema": LFM2_SEMANTIC_PRETERMINAL_CAPSULE_CONTEXT_SCHEMA,
        "bindings_sha256": bindings.bindings_sha256,
        "semantic_layer_stack_result_sha256": layer_stack.receipt["result_sha256"],
        "final_layer_mlp_evidence_sha256": {
            split: layer_stack.final_layer_mlp_evidence[split].authentication_sha256
            for split in _SPLITS
        },
        "active_interface_report_sha256": interface.report["report_sha256"],
        "interface_authentication_sha256": seam["interface_authentication"][
            "result_sha256"
        ],
        "attention_receipt_authentication_sha256": seam[
            "attention_receipt_authentication"
        ]["result_sha256"],
        "model_identity_sha256": seam["model_identity_sha256"],
        "model_state_sha256": seam["model_state_sha256"],
        "parameter_inventory_sha256": seam["parameter_inventory_sha256"],
        "parameter_element_count": seam["parameter_element_count"],
        "tokenizer_semantics_sha256": seam["tokenizer_semantics_sha256"],
        "deferred_final_w2_semantic_sha256": seam["final_w2_semantic_sha256"],
        "deferred_final_w2_terminal_action_sha256": seam[
            "final_w2_terminal_action_sha256"
        ],
        "deferred_final_w2_exact_zero": True,
        "terminal_compiler_called": False,
    }
    prior = layer_stack.final_layer_mlp_down_weight_prior
    if prior is not None:
        body.update(
            {
                "final_layer_mlp_down_weight_prior_authentication_sha256": (
                    prior.authentication_sha256
                ),
                "final_layer_mlp_down_weight_prior_tensor_sha256": (
                    _terminal_action_tensor_sha256(
                        prior.down_weight,
                        "preterminal.context.final_layer_mlp_down_weight_prior",
                    )
                ),
                "final_layer_mlp_down_weight_prior_installed_before_terminal": (
                    False
                ),
            }
        )
    return body


def publish_lfm2_semantic_preterminal_capsule(
    *,
    target_model: Any,
    target_tokenizer: Any,
    layer_stack: Lfm2SemanticLayerStackInstallation,
    interface: TiedCausalInterfaceFold,
    attention_receipt_authentication: Mapping[str, Any],
    bindings: Lfm2SemanticPreterminalCapsuleBindings,
    output_directory: str | os.PathLike[str],
    mapped_parent_tied_table: Any | None = None,
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
) -> Lfm2SemanticPreterminalCapsuleReference:
    """Atomically publish the exact preterminal seam before terminal execution."""

    if not isinstance(bindings, Lfm2SemanticPreterminalCapsuleBindings):
        raise TypeError("bindings must be Lfm2SemanticPreterminalCapsuleBindings.")
    output = _new_output(output_directory)
    seam = _validate_live_seam(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        layer_stack=layer_stack,
        interface=interface,
        attention_receipt_authentication=attention_receipt_authentication,
        bindings=bindings,
        mapped_parent_tied_table=mapped_parent_tied_table,
        anchor_transport_witness=anchor_transport_witness,
    )
    context = _context_body(
        bindings=bindings,
        seam=seam,
        layer_stack=layer_stack,
        interface=interface,
    )
    context_sha = canonical_sha256(context)
    payload, payload_entries = _normalize_tensors(
        _payload_tensors(
            layer_stack,
            interface,
            anchor_transport_witness=anchor_transport_witness,
        ),
        name="preterminal seam payload",
        clone=True,
    )
    anchor_transport = anchor_transport_witness is not None
    down_weight_prior = layer_stack.final_layer_mlp_down_weight_prior is not None
    expected_payload_names = _expected_payload_names(
        anchor_transport=anchor_transport,
        down_weight_prior=down_weight_prior,
    )
    if tuple(payload) != tuple(sorted(expected_payload_names)):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal seam payload does not have its exact typed tensor schema."
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=".semantic-preterminal-",
            dir=str(output.parent),
        )
    ).absolute()
    stage_identity = _stat_identity(_real_directory(stage, name="preterminal stage"))
    moved = False
    try:
        checkpoint = _publish_quarantine_checkpoint(
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            checkpoint=stage / _CHECKPOINT_NAME,
            context_binding_sha256=context_sha,
            model_state_sha256=seam["model_state_sha256"],
            parameter_inventory_sha256=seam["parameter_inventory_sha256"],
            tokenizer_semantics_sha256=seam["tokenizer_semantics_sha256"],
            ffn_width=seam["materialized_ffn_width"],
        )
        payload_path = stage / _PAYLOAD_NAME
        _save_safetensors(
            payload_path,
            payload,
            artifact_kind=_ARTIFACT_KIND,
            file_role="preterminal-seam",
        )
        payload_record = _file_record(
            payload_path,
            tensor_entries=payload_entries,
        )
        manifest_body = {
            "schema": LFM2_SEMANTIC_PRETERMINAL_CAPSULE_MANIFEST_SCHEMA,
            "artifact_kind": _ARTIFACT_KIND,
            **_NONPROMOTABLE_FLAGS,
            "bindings": bindings.as_dict(),
            "bindings_sha256": bindings.bindings_sha256,
            "context_binding": context,
            "context_binding_sha256": context_sha,
            "semantic_layer_stack_receipt": dict(layer_stack.receipt),
            "attention_receipt_authentication": seam[
                "attention_receipt_authentication"
            ],
            "final_layer_mlp_evidence": {
                split: layer_stack.final_layer_mlp_evidence[split].as_dict()
                for split in _SPLITS
            },
            "active_interface_report": dict(interface.report),
            "active_interface_authentication": _deep_thaw(
                seam["interface_authentication"]
            ),
            "anchor_source_interface_report": (
                None
                if anchor_transport_witness is None
                else dict(anchor_transport_witness.source_interface_report)
            ),
            "anchor_transport_witness": (
                None
                if anchor_transport_witness is None
                else dict(anchor_transport_witness.as_dict())
            ),
            "seam_payload": payload_record,
            "checkpoint": checkpoint,
            "model": {
                name: _deep_thaw(seam[name])
                for name in (
                    "model_identity_sha256",
                    "model_state_sha256",
                    "parameter_inventory",
                    "parameter_inventory_sha256",
                    "parameter_element_count",
                    "tokenizer_semantics_sha256",
                    "materialized_ffn_width",
                    "final_w2_parameter_name",
                    "final_w2_semantic_sha256",
                    "final_w2_terminal_action_sha256",
                    "final_w2_shape",
                    "final_w2_dtype",
                    "final_w2_nonzero_element_count",
                    "final_w2_exact_zero",
                    "final_mlp_producer_coverage",
                )
            },
            "payload_tensor_count": len(payload_entries),
            "payload_tensor_names": list(sorted(payload)),
            "anchor_residual_transport_inputs_persisted": anchor_transport,
            "exact_four_typed_evidence_banks": True,
            "active_interface_tensors_persisted": True,
            "dense_strict_local_checkpoint": True,
            "no_pickle_files": True,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        if down_weight_prior:
            prior = layer_stack.final_layer_mlp_down_weight_prior
            if prior is None:  # pragma: no cover - narrowed above
                raise AssertionError("Q-DEIM prior disappeared during publication.")
            manifest_body.update(
                {
                    "final_layer_mlp_down_weight_prior": prior.as_dict(),
                    "final_layer_mlp_down_weight_prior_persisted": True,
                    "final_layer_mlp_down_weight_prior_tensor_name": (
                        _DOWN_WEIGHT_PRIOR_TENSOR_NAME
                    ),
                }
            )
        manifest = _sealed(manifest_body, "manifest_sha256")
        manifest_path = stage / _MANIFEST_NAME
        manifest_raw = _write_canonical_json(manifest_path, manifest)
        manifest_record = {
            "path": _MANIFEST_NAME,
            "size_bytes": len(manifest_raw),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        receipt_body = {
            "schema": LFM2_SEMANTIC_PRETERMINAL_CAPSULE_RECEIPT_SCHEMA,
            "artifact_kind": _ARTIFACT_KIND,
            **_NONPROMOTABLE_FLAGS,
            "manifest_file": manifest_record,
            "manifest_sha256": manifest["manifest_sha256"],
            "bindings_sha256": bindings.bindings_sha256,
            "context_binding_sha256": context_sha,
            "checkpoint_artifact_sha256": checkpoint["artifact_metadata"][
                "artifact_sha256"
            ],
            "checkpoint_artifact_tree_sha256": checkpoint["artifact_tree_sha256"],
            "seam_payload_file_sha256": payload_record["sha256"],
            "semantic_layer_stack_result_sha256": layer_stack.receipt["result_sha256"],
            "attention_receipt_authentication_sha256": seam[
                "attention_receipt_authentication"
            ]["result_sha256"],
            "deferred_final_w2_exact_zero": True,
            "terminal_compiler_called_before_publication": False,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        if down_weight_prior:
            prior = layer_stack.final_layer_mlp_down_weight_prior
            if prior is None:  # pragma: no cover - narrowed above
                raise AssertionError("Q-DEIM prior disappeared during receipt seal.")
            receipt_body.update(
                {
                    "final_layer_mlp_down_weight_prior_authentication_sha256": (
                        prior.authentication_sha256
                    ),
                    "final_layer_mlp_down_weight_prior_persisted": True,
                }
            )
        receipt = _sealed(receipt_body, "receipt_sha256")
        receipt_path = stage / _RECEIPT_NAME
        receipt_raw = _write_canonical_json(receipt_path, receipt)
        _fsync_directory(stage)
        _open_outer_capsule(
            stage,
            expected_receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_bindings_sha256=bindings.bindings_sha256,
            load_payload=False,
        )
        _atomic_rename_noreplace(stage, output)
        moved = True
        _fsync_directory(output.parent)
        opened = _open_outer_capsule(
            output,
            expected_receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_bindings_sha256=bindings.bindings_sha256,
            load_payload=False,
        )
        return Lfm2SemanticPreterminalCapsuleReference(
            path=output,
            receipt_sha256=receipt["receipt_sha256"],
            receipt_file_sha256=opened["receipt_file_sha256"],
            manifest_sha256=manifest["manifest_sha256"],
            bindings_sha256=bindings.bindings_sha256,
            context_binding_sha256=context_sha,
        )
    except BaseException:
        if not moved and _lexists(stage):
            _safe_remove_stage(
                stage,
                parent=output.parent,
                identity=stage_identity,
            )
        raise


def _open_outer_capsule(
    directory: str | os.PathLike[str],
    *,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
    expected_bindings_sha256: str,
    load_payload: bool,
) -> dict[str, Any]:
    expected_file_sha = _digest(
        expected_receipt_file_sha256,
        "expected capsule receipt file SHA-256",
    )
    expected_receipt_sha = _digest(
        expected_receipt_sha256,
        "expected capsule receipt SHA-256",
    )
    expected_bindings_sha = _digest(
        expected_bindings_sha256,
        "expected capsule bindings SHA-256",
    )
    root, root_identity = _root_path(directory)
    receipt_path = root / _RECEIPT_NAME
    receipt, receipt_raw = _read_canonical_json_with_raw(
        receipt_path,
        name=_RECEIPT_NAME,
    )
    receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
    receipt = _verify_seal(
        receipt,
        field_name="receipt_sha256",
        label="preterminal capsule receipt",
    )
    _validate_nonpromotable(receipt, label="preterminal capsule receipt")
    if (
        receipt_file_sha != expected_file_sha
        or receipt.get("receipt_sha256") != expected_receipt_sha
        or receipt.get("bindings_sha256") != expected_bindings_sha
        or receipt.get("schema") != LFM2_SEMANTIC_PRETERMINAL_CAPSULE_RECEIPT_SCHEMA
        or receipt.get("artifact_kind") != _ARTIFACT_KIND
        or receipt.get("deferred_final_w2_exact_zero") is not True
        or receipt.get("terminal_compiler_called_before_publication") is not False
        or receipt.get("operation_audit") != _ZERO_OPERATION_AUDIT
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule receipt differs from its external pins."
        )
    manifest_record = receipt.get("manifest_file")
    if not isinstance(manifest_record, Mapping):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule receipt omitted its manifest record."
        )
    manifest_path = root / _MANIFEST_NAME
    manifest, manifest_raw = _read_canonical_json_with_raw(
        manifest_path,
        name=_MANIFEST_NAME,
    )
    if (
        manifest_record.get("path") != _MANIFEST_NAME
        or manifest_record.get("size_bytes") != len(manifest_raw)
        or manifest_record.get("sha256") != hashlib.sha256(manifest_raw).hexdigest()
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule manifest differs from its receipt."
        )
    manifest = _verify_seal(
        manifest,
        field_name="manifest_sha256",
        label="preterminal capsule manifest",
    )
    _validate_nonpromotable(manifest, label="preterminal capsule manifest")
    bindings = Lfm2SemanticPreterminalCapsuleBindings.from_mapping(
        manifest.get("bindings")
    )
    anchor_transport = manifest.get(
        "anchor_residual_transport_inputs_persisted"
    )
    if type(anchor_transport) is not bool:
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule omitted its anchor transport inventory mode."
        )
    prior_document = manifest.get("final_layer_mlp_down_weight_prior")
    down_weight_prior = isinstance(prior_document, Mapping)
    if down_weight_prior:
        prior_body = dict(prior_document)
        prior_authentication_sha256 = prior_body.pop(
            "authentication_sha256",
            None,
        )
        if (
            prior_authentication_sha256 != canonical_sha256(prior_body)
            or manifest.get("final_layer_mlp_down_weight_prior_persisted") is not True
            or manifest.get("final_layer_mlp_down_weight_prior_tensor_name")
            != _DOWN_WEIGHT_PRIOR_TENSOR_NAME
            or receipt.get(
                "final_layer_mlp_down_weight_prior_authentication_sha256"
            )
            != prior_authentication_sha256
            or receipt.get("final_layer_mlp_down_weight_prior_persisted") is not True
        ):
            raise Lfm2SemanticPreterminalCapsuleError(
                "Preterminal capsule W2 prior document is unauthenticated."
            )
    elif (
        manifest.get("final_layer_mlp_down_weight_prior_persisted") is not None
        or manifest.get("final_layer_mlp_down_weight_prior_tensor_name") is not None
        or receipt.get(
            "final_layer_mlp_down_weight_prior_authentication_sha256"
        )
        is not None
        or receipt.get("final_layer_mlp_down_weight_prior_persisted") is not None
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Legacy capsule has a partial final-layer W2 prior."
        )
    expected_payload_names = _expected_payload_names(
        anchor_transport=anchor_transport,
        down_weight_prior=down_weight_prior,
    )
    if (
        manifest.get("schema") != LFM2_SEMANTIC_PRETERMINAL_CAPSULE_MANIFEST_SCHEMA
        or manifest.get("artifact_kind") != _ARTIFACT_KIND
        or manifest.get("manifest_sha256") != receipt.get("manifest_sha256")
        or manifest.get("bindings_sha256") != expected_bindings_sha
        or bindings.bindings_sha256 != expected_bindings_sha
        or manifest.get("context_binding_sha256")
        != receipt.get("context_binding_sha256")
        or manifest.get("payload_tensor_count") != len(expected_payload_names)
        or manifest.get("payload_tensor_names")
        != list(sorted(expected_payload_names))
        or (
            anchor_transport
            and (
                not isinstance(
                    manifest.get("anchor_source_interface_report"),
                    Mapping,
                )
                or not isinstance(manifest.get("anchor_transport_witness"), Mapping)
            )
        )
        or (
            not anchor_transport
            and (
                manifest.get("anchor_source_interface_report") is not None
                or manifest.get("anchor_transport_witness") is not None
            )
        )
        or manifest.get("exact_four_typed_evidence_banks") is not True
        or manifest.get("active_interface_tensors_persisted") is not True
        or manifest.get("dense_strict_local_checkpoint") is not True
        or manifest.get("no_pickle_files") is not True
        or manifest.get("operation_audit") != _ZERO_OPERATION_AUDIT
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule manifest contract is invalid."
        )
    payload_record = manifest.get("seam_payload")
    if not isinstance(payload_record, Mapping):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule manifest omitted its seam payload record."
        )
    tensors = _load_safetensors_path(
        root / _PAYLOAD_NAME,
        name=_PAYLOAD_NAME,
        artifact_kind=_ARTIFACT_KIND,
        file_role="preterminal-seam",
        record=payload_record,
        clone_tensors=load_payload,
    )
    if (
        payload_record.get("path") != _PAYLOAD_NAME
        or payload_record.get("sha256") != receipt.get("seam_payload_file_sha256")
        or [entry.get("name") for entry in payload_record.get("tensor_entries", [])]
        != list(sorted(expected_payload_names))
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal seam tensor inventory is incomplete."
        )
    checkpoint = manifest.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("artifact_metadata", {}).get("artifact_sha256")
        != receipt.get("checkpoint_artifact_sha256")
        or checkpoint.get("artifact_tree_sha256")
        != receipt.get("checkpoint_artifact_tree_sha256")
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal dense checkpoint receipt is incomplete."
        )
    _verify_quarantine_checkpoint_record(
        checkpoint=root / _CHECKPOINT_NAME,
        expected=checkpoint,
    )
    _recheck_root(root, root_identity)
    return {
        "root": root,
        "root_identity": root_identity,
        "receipt": receipt,
        "receipt_file_sha256": receipt_file_sha,
        "manifest": manifest,
        "bindings": bindings,
        "tensors": tensors,
    }


def load_lfm2_semantic_preterminal_capsule(
    directory: str | os.PathLike[str],
    *,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
    expected_bindings_sha256: str,
    device: str,
    dtype: str = "bfloat16",
) -> Lfm2SemanticPreterminalCapsule:
    """Strictly load and reconstruct one externally pinned preterminal seam."""

    opened = _open_outer_capsule(
        directory,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_bindings_sha256=expected_bindings_sha256,
        load_payload=True,
    )
    root = opened["root"]
    manifest = opened["manifest"]
    receipt = opened["receipt"]
    bindings = opened["bindings"]
    model_record = manifest.get("model")
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(model_record, Mapping) or not isinstance(
        checkpoint_record, Mapping
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule omitted model or checkpoint bindings."
        )
    model, tokenizer = _load_quarantine_checkpoint(
        checkpoint=root / _CHECKPOINT_NAME,
        expected=checkpoint_record,
        context_binding_sha256=manifest["context_binding_sha256"],
        model_state_sha256=model_record["model_state_sha256"],
        parameter_inventory_sha256=model_record["parameter_inventory_sha256"],
        tokenizer_semantics_sha256=model_record["tokenizer_semantics_sha256"],
        ffn_width=model_record["materialized_ffn_width"],
        device=device,
        dtype=dtype,
    )
    tensors = opened["tensors"]
    evidence_documents = manifest.get("final_layer_mlp_evidence")
    if not isinstance(evidence_documents, Mapping) or set(evidence_documents) != set(
        _SPLITS
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Preterminal capsule omitted one of four evidence documents."
        )
    evidence: dict[str, Lfm2CpcfTerminalActionEvidence] = {}
    for split in _SPLITS:
        document = evidence_documents[split]
        if not isinstance(document, Mapping):
            raise Lfm2SemanticPreterminalCapsuleError(
                f"Preterminal {split} evidence document is invalid."
            )
        evidence[split] = Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(document.get("row_ids", ())),
            mlp_input=tensors[f"evidence.{split}.mlp_input"],
            pre_mlp_residual=tensors[f"evidence.{split}.pre_mlp_residual"],
            parent_final_norm_output=tensors[
                f"evidence.{split}.parent_final_norm_output"
            ],
            sample_weights=tensors[f"evidence.{split}.sample_weights"],
            authentication_sha256=document.get("authentication_sha256", ""),
        )
        if evidence[split].as_dict() != dict(document):
            raise Lfm2SemanticPreterminalCapsuleError(
                f"Reconstructed {split} evidence differs from its manifest."
            )
    prior_document = manifest.get("final_layer_mlp_down_weight_prior")
    down_weight_prior = None
    if prior_document is not None:
        if not isinstance(prior_document, Mapping):
            raise Lfm2SemanticPreterminalCapsuleError(
                "Persisted final-layer W2 prior document is invalid."
            )
        down_weight_prior = Lfm2CpcfTerminalActionDownWeightPrior(
            down_weight=tensors[_DOWN_WEIGHT_PRIOR_TENSOR_NAME],
            body_recipe_id=prior_document.get("body_recipe_id", ""),
            body_recipe_sha256=prior_document.get("body_recipe_sha256", ""),
            mlp_lane=prior_document.get("mlp_lane", ""),
            selected_role=prior_document.get("selected_role", ""),
            selected_candidate_sha256=prior_document.get(
                "selected_candidate_sha256",
                "",
            ),
            preparation_sha256=prior_document.get("preparation_sha256", ""),
            mlp_installation_core_sha256=prior_document.get(
                "mlp_installation_core_sha256",
                "",
            ),
            live_deferred_w2_sha256=prior_document.get(
                "live_deferred_w2_sha256",
                "",
            ),
            authentication_sha256=prior_document.get(
                "authentication_sha256",
                "",
            ),
        )
        if down_weight_prior.as_dict() != dict(prior_document):
            raise Lfm2SemanticPreterminalCapsuleError(
                "Reconstructed final-layer W2 prior differs from its manifest."
            )
    layer_stack = Lfm2SemanticLayerStackInstallation(
        receipt=manifest["semantic_layer_stack_receipt"],
        final_layer_mlp_evidence=evidence,
        final_layer_mlp_down_weight_prior=down_weight_prior,
    )
    interface = TiedCausalInterfaceFold(
        target_weight=tensors["interface.target_weight"],
        producer_encoder=tensors["interface.producer_encoder"],
        report=manifest["active_interface_report"],
    )
    anchor_transport_witness = None
    if manifest["anchor_residual_transport_inputs_persisted"]:
        witness_document = manifest["anchor_transport_witness"]
        anchor_transport_witness = TiedResidualAnchorTransportWitness(
            source_interface_report=manifest["anchor_source_interface_report"],
            source_encoder=tensors["anchor_source.producer_encoder"],
            anchor_base_target_weight=tensors["anchor_base.target_weight"],
            anchor_source_residual_weight=tensors[
                "anchor_source.residual_weight"
            ],
            witness_sha256=witness_document.get("witness_sha256", ""),
        )
        reconstructed_source = (
            anchor_transport_witness.materialize_source_interface()
        )
        if (
            dict(anchor_transport_witness.as_dict()) != dict(witness_document)
            or _interface_tensor_sha256(reconstructed_source.target_weight)
            != _interface_tensor_sha256(
                tensors["anchor_source.target_weight"]
            )
        ):
            raise Lfm2SemanticPreterminalCapsuleError(
                "Persisted anchor witness tensors do not reproduce their manifest."
            )
    attention = manifest["attention_receipt_authentication"]
    seam = _validate_live_seam(
        target_model=model,
        target_tokenizer=tokenizer,
        layer_stack=layer_stack,
        interface=interface,
        attention_receipt_authentication=attention,
        bindings=bindings,
        anchor_transport_witness=anchor_transport_witness,
        pinned_interface_authentication=manifest[
            "active_interface_authentication"
        ],
    )
    if any(
        _deep_thaw(seam[name]) != model_record.get(name)
        for name in (
            "model_identity_sha256",
            "model_state_sha256",
            "parameter_inventory",
            "parameter_inventory_sha256",
            "parameter_element_count",
            "tokenizer_semantics_sha256",
            "materialized_ffn_width",
            "final_w2_parameter_name",
            "final_w2_semantic_sha256",
            "final_w2_terminal_action_sha256",
            "final_w2_shape",
            "final_w2_dtype",
            "final_w2_nonzero_element_count",
            "final_w2_exact_zero",
            "final_mlp_producer_coverage",
        )
    ):
        raise Lfm2SemanticPreterminalCapsuleError(
            "Reloaded preterminal model differs from its topology/state binding."
        )
    reference = Lfm2SemanticPreterminalCapsuleReference(
        path=root,
        receipt_sha256=receipt["receipt_sha256"],
        receipt_file_sha256=opened["receipt_file_sha256"],
        manifest_sha256=manifest["manifest_sha256"],
        bindings_sha256=bindings.bindings_sha256,
        context_binding_sha256=manifest["context_binding_sha256"],
    )
    return Lfm2SemanticPreterminalCapsule(
        path=root,
        target_model=model,
        target_tokenizer=tokenizer,
        layer_stack=layer_stack,
        interface=interface,
        attention_receipt_authentication=_deep_freeze(dict(attention)),
        bindings=bindings,
        manifest=_deep_freeze(dict(manifest)),
        receipt=_deep_freeze(dict(receipt)),
        reference=reference,
        anchor_transport_witness=anchor_transport_witness,
    )


__all__ = [
    "LFM2_SEMANTIC_PRETERMINAL_CAPSULE_BINDINGS_SCHEMA",
    "LFM2_SEMANTIC_PRETERMINAL_CAPSULE_CONTEXT_SCHEMA",
    "LFM2_SEMANTIC_PRETERMINAL_CAPSULE_MANIFEST_SCHEMA",
    "LFM2_SEMANTIC_PRETERMINAL_CAPSULE_RECEIPT_SCHEMA",
    "Lfm2SemanticPreterminalCapsule",
    "Lfm2SemanticPreterminalCapsuleBindings",
    "Lfm2SemanticPreterminalCapsuleError",
    "Lfm2SemanticPreterminalCapsuleReference",
    "load_lfm2_semantic_preterminal_capsule",
    "publish_lfm2_semantic_preterminal_capsule",
]
