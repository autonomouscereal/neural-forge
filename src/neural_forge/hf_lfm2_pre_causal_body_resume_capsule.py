"""Immutable full-body resume capsules for the exact LFM2 semantic fold.

This boundary is deliberately narrow: it freezes one complete dense child
after the corrected final W2 and moving-gauge body finalization, but before
any causal-output evidence is collected or installed.  The artifact is a
restart point, never an accepted child or a capability claim.

Publication uses the reviewed direct-local LFM2 checkpoint I/O inside an
exclusive sibling staging directory.  The staged checkpoint is freshly
loaded and authenticated before an atomic no-replace directory rename.  A
consumer must hold the returned externally pinned reference; path-only loads
are intentionally unavailable.
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
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.dynamic_model_inspector import (
    InspectorRegistry,
    default_inspector_registry,
)
from neural_forge.dynamic_target_synthesis import (
    serialized_inference_state_inventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
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
from neural_forge.hf_lfm2_direct_local import Lfm2DirectLocalCheckpointIO
from neural_forge.hf_lfm2_interface_frontier import (
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_tokenizer_bridge import (
    strict_tokenizer_bridge_target_package_fingerprint,
)


LFM2_PRE_CAUSAL_BODY_RESUME_BINDINGS_SCHEMA = (
    "neural-forge-lfm2-pre-causal-body-resume-bindings-v1"
)
LFM2_PRE_CAUSAL_BODY_RESUME_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-pre-causal-body-resume-manifest-v1"
)
LFM2_PRE_CAUSAL_BODY_RESUME_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-pre-causal-body-resume-receipt-v1"
)

_ARTIFACT_KIND = "lfm2-post-moving-gauge-pre-causal-body-resume"
_STATUS = "quarantined-pre-causal-body-resume-only"
_CHECKPOINT_NAME = "checkpoint"
_MANIFEST_NAME = "capsule-manifest.json"
_RECEIPT_NAME = "capsule-receipt.json"
_ROOT_INVENTORY = frozenset({_CHECKPOINT_NAME, _MANIFEST_NAME, _RECEIPT_NAME})
_STAGE_PREFIX = ".lfm2-pre-causal-body-"

_MODULE_HOOK_REGISTRY_NAMES = (
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
)
_GLOBAL_HOOK_REGISTRY_NAMES = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_buffer_registration_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_pre_hooks",
    "_global_forward_pre_hooks_with_kwargs",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
)

_NONPROMOTABLE_FLAGS = {
    "status": _STATUS,
    "causal_output_head_installed": False,
    "complete_child": False,
    "promotion_authorized": False,
    "accepted_artifact": False,
    "behavior_capability_proven": False,
    "training_performed": False,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parent_sidecar_parameters": 0,
    "runtime_shared_parent_parameters": 0,
    "parent_tensor_payloads": 0,
    "parent_model_required_for_reload": False,
}

_ZERO_OPERATION_AUDIT = {
    "parent_model_objects_constructed": 0,
    "parent_weight_tensor_loads": 0,
    "parent_model_forward_calls": 0,
    "child_model_forward_calls": 0,
    "model_generate_calls": 0,
    "trace_recaptures": 0,
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "network_access_performed": False,
    "pickle_files_written": 0,
}

_DIGEST_FIELDS = (
    "parent_identity_sha256",
    "target_synthesis_fingerprint",
    "target_topology_fingerprint",
    "target_configuration_sha256",
    "target_state_inventory_sha256",
    "native_target_contract_sha256",
    "child_graph_fingerprint",
    "child_structural_topology_fingerprint",
    "bridge_artifact_sha256",
    "bridge_profile_sha256",
    "target_tokenizer_semantics_sha256",
    "target_tokenizer_package_fingerprint",
    "body_fold_realization_sha256",
    "global_residual_closure_selection_sha256",
    "structural_interface_install_sha256",
    "semantic_layer_stack_sha256",
    "structural_attention_candidate_archive_manifest_sha256",
    "postfold_reclosure_seed_archive_manifest_sha256",
    "postfold_reclosure_seed_publication_sha256",
    "deferred_final_w2_install_sha256",
    "moving_gauge_post_w2_finalization_sha256",
    "body_model_state_sha256",
    "body_runtime_state_sha256",
    "body_lfm2_identity_sha256",
    "causal_body_inventory_sha256",
    "installed_input_table_sha256",
    "installed_output_baseline_sha256",
)


class Lfm2PreCausalBodyResumeCapsuleError(RuntimeError):
    """A pre-causal body capsule failed publication or authentication."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    if value == "0" * 64:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            f"{name} cannot be an unset all-zero digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            f"{name} must be one positive integer."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2PreCausalBodyResumeCapsuleError(f"{name} must be nonempty.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Lfm2PreCausalBodyResumeBindings:
    """Externally authenticated target, bridge, body, and lineage seals."""

    child_name: str
    target_unique_parameters: int
    target_logical_parameters: int
    parent_identity_sha256: str
    target_synthesis_fingerprint: str
    target_topology_fingerprint: str
    target_configuration_sha256: str
    target_state_inventory_sha256: str
    native_target_contract_sha256: str
    child_graph_fingerprint: str
    child_structural_topology_fingerprint: str
    bridge_artifact_sha256: str
    bridge_profile_sha256: str
    target_tokenizer_semantics_sha256: str
    target_tokenizer_package_fingerprint: str
    body_fold_realization_sha256: str
    global_residual_closure_selection_sha256: str
    structural_interface_install_sha256: str
    semantic_layer_stack_sha256: str
    structural_attention_candidate_archive_manifest_sha256: str
    postfold_reclosure_seed_archive_manifest_sha256: str
    postfold_reclosure_seed_publication_sha256: str
    deferred_final_w2_install_sha256: str
    moving_gauge_post_w2_finalization_sha256: str
    body_model_state_sha256: str
    body_runtime_state_sha256: str
    body_lfm2_identity_sha256: str
    causal_body_inventory_sha256: str
    installed_input_table_sha256: str
    installed_output_baseline_sha256: str
    bindings_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_name", _text(self.child_name, "child_name"))
        unique = _positive_int(
            self.target_unique_parameters,
            "target_unique_parameters",
        )
        logical = _positive_int(
            self.target_logical_parameters,
            "target_logical_parameters",
        )
        if logical < unique:
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "target_logical_parameters cannot be below the unique count."
            )
        for name in _DIGEST_FIELDS:
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "bindings_sha256",
            canonical_sha256(self.as_dict(include_sha256=False)),
        )

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        result = {
            "schema": LFM2_PRE_CAUSAL_BODY_RESUME_BINDINGS_SCHEMA,
            "child_name": self.child_name,
            "target_unique_parameters": self.target_unique_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            **{name: getattr(self, name) for name in _DIGEST_FIELDS},
        }
        if include_sha256:
            result["bindings_sha256"] = self.bindings_sha256
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Any,
    ) -> "Lfm2PreCausalBodyResumeBindings":
        if not isinstance(value, Mapping):
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Capsule bindings must be one mapping."
            )
        expected = {
            "schema",
            "child_name",
            "target_unique_parameters",
            "target_logical_parameters",
            "bindings_sha256",
            *_DIGEST_FIELDS,
        }
        if set(value) != expected or value.get("schema") != (
            LFM2_PRE_CAUSAL_BODY_RESUME_BINDINGS_SCHEMA
        ):
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Capsule bindings have missing or unknown fields."
            )
        result = cls(
            child_name=value["child_name"],
            target_unique_parameters=value["target_unique_parameters"],
            target_logical_parameters=value["target_logical_parameters"],
            **{name: value[name] for name in _DIGEST_FIELDS},
        )
        if result.bindings_sha256 != value.get("bindings_sha256"):
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Capsule binding self-authentication failed."
            )
        return result


@dataclass(frozen=True, slots=True)
class Lfm2PreCausalBodyResumeReference:
    """External pins required to discover and authenticate one capsule."""

    path: Path
    receipt_sha256: str
    receipt_file_sha256: str
    manifest_sha256: str
    bindings_sha256: str
    checkpoint_inventory_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().absolute())
        for name in (
            "receipt_sha256",
            "receipt_file_sha256",
            "manifest_sha256",
            "bindings_sha256",
            "checkpoint_inventory_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "receipt_sha256": self.receipt_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "bindings_sha256": self.bindings_sha256,
            "checkpoint_inventory_sha256": self.checkpoint_inventory_sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Any,
    ) -> "Lfm2PreCausalBodyResumeReference":
        if not isinstance(value, Mapping):
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Capsule reference must be one mapping."
            )
        expected = {
            "path",
            "receipt_sha256",
            "receipt_file_sha256",
            "manifest_sha256",
            "bindings_sha256",
            "checkpoint_inventory_sha256",
        }
        if set(value) != expected:
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Capsule reference has missing or unknown fields."
            )
        path = value.get("path")
        if not isinstance(path, (str, os.PathLike)) or not os.fspath(path).strip():
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Capsule reference path must be nonempty."
            )
        return cls(
            path=Path(path),
            receipt_sha256=value["receipt_sha256"],
            receipt_file_sha256=value["receipt_file_sha256"],
            manifest_sha256=value["manifest_sha256"],
            bindings_sha256=value["bindings_sha256"],
            checkpoint_inventory_sha256=value[
                "checkpoint_inventory_sha256"
            ],
        )


@dataclass(frozen=True, slots=True)
class Lfm2PreCausalBodyResumeCapsule:
    """One fully authenticated fresh child at the pre-causal boundary."""

    path: Path
    model: Any = field(repr=False, compare=False)
    bindings: Lfm2PreCausalBodyResumeBindings
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    reference: Lfm2PreCausalBodyResumeReference


@dataclass(frozen=True, slots=True)
class Lfm2PreCausalBodyResumeAuthentication:
    """Authenticated capsule metadata without materializing its model."""

    path: Path
    root_identity: tuple[int, int]
    bindings: Lfm2PreCausalBodyResumeBindings
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    reference: Lfm2PreCausalBodyResumeReference


def _sealed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    result = dict(body)
    result[field_name] = canonical_sha256(result)
    return result


def _verify_seal(value: Any, *, field_name: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2PreCausalBodyResumeCapsuleError(f"{label} must be a mapping.")
    result = dict(value)
    body = dict(result)
    supplied = _digest(body.pop(field_name, None), f"{label} {field_name}")
    if canonical_sha256(body) != supplied:
        raise Lfm2PreCausalBodyResumeCapsuleError(f"{label} seal is invalid.")
    return result


def _validate_nonpromotable(value: Mapping[str, Any], *, label: str) -> None:
    if any(value.get(name) != expected for name, expected in _NONPROMOTABLE_FLAGS.items()):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            f"{label} is not fixed non-promotable resume evidence."
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    details = _regular_file(path, name=path.name)
    return {
        "path": path.name if relative_path is None else relative_path,
        "size_bytes": int(details.st_size),
        "sha256": _file_sha256(path),
    }


def _checkpoint_inventory(checkpoint: Path) -> list[dict[str, Any]]:
    """Hash every ordinary checkpoint file and reject every indirect entry."""

    root = checkpoint.expanduser().absolute()
    root_stat = _real_directory(root, name="pre-causal body checkpoint")
    root_identity = _stat_identity(root_stat)
    rows: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        directory_stat = _real_directory(directory, name="checkpoint directory")
        try:
            entries = tuple(sorted(os.scandir(directory), key=lambda item: item.name))
        except OSError as exc:
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Checkpoint directory could not be enumerated."
            ) from exc
        for entry in entries:
            path = directory / entry.name
            if _is_link_or_junction(path):
                raise Lfm2PreCausalBodyResumeCapsuleError(
                    "Checkpoint inventory cannot contain links or junctions."
                )
            details = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISDIR(details.st_mode):
                if path.resolve(strict=True).parent != directory.resolve(strict=True):
                    raise Lfm2PreCausalBodyResumeCapsuleError(
                        "Checkpoint directory escaped its artifact root."
                    )
                pending.append(path)
            elif stat.S_ISREG(details.st_mode):
                if int(getattr(details, "st_nlink", 1)) != 1:
                    raise Lfm2PreCausalBodyResumeCapsuleError(
                        "Checkpoint files cannot be hard links."
                    )
                rows.append(_file_record(path, relative_path=relative))
            else:
                raise Lfm2PreCausalBodyResumeCapsuleError(
                    "Checkpoint inventory contains a non-file entry."
                )
        if _stat_identity(
            _real_directory(directory, name="checkpoint directory")
        ) != _stat_identity(directory_stat):
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Checkpoint directory changed during authentication."
            )
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Checkpoint file inventory is empty or ambiguous."
        )
    if _stat_identity(
        _real_directory(root, name="pre-causal body checkpoint")
    ) != root_identity:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Checkpoint root changed during authentication."
        )
    return sorted(rows, key=lambda row: row["path"])


def _new_output(value: str | os.PathLike[str]) -> Path:
    output = Path(value).expanduser().absolute()
    if _lexists(output):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            f"Refusing to replace existing pre-causal body capsule {output}."
        )
    parent = output.parent
    _real_directory(parent, name="pre-causal body capsule parent")
    if _is_link_or_junction(parent) or parent.resolve(strict=True) != parent:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule parent path cannot traverse a link."
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
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Refusing to remove unauthenticated pre-causal capsule staging."
        )
    shutil.rmtree(stage)
    _fsync_directory(parent)


def _interface_tensor_sha256(value: Any) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "PyTorch is required for pre-causal body capsules."
        ) from exc
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal interface weights must be materialized dense tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _storage_identity(value: Any) -> tuple[Any, ...]:
    try:
        return (
            str(value.device),
            int(value.untyped_storage().data_ptr()),
            int(value.storage_offset()),
            tuple(int(size) for size in value.shape),
            tuple(int(size) for size in value.stride()),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Could not authenticate interface storage identity."
        ) from exc


def _assert_disjoint_models(original: Any, reloaded: Any) -> None:
    if original is reloaded:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Fresh loader returned the live child object."
        )
    original_modules = {id(module) for module in original.modules()}
    if any(id(module) in original_modules for module in reloaded.modules()):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Fresh child shares a module object with the live child."
        )
    original_tensors = tuple(original.parameters()) + tuple(original.buffers())
    original_objects = {id(value) for value in original_tensors}
    original_storages = {_storage_identity(value) for value in original_tensors}
    for value in tuple(reloaded.parameters()) + tuple(reloaded.buffers()):
        if id(value) in original_objects or _storage_identity(value) in original_storages:
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Fresh child shares tensor storage with the live child."
            )


def _checkpoint_io_bindings(checkpoint_io: Lfm2DirectLocalCheckpointIO) -> dict[str, str]:
    if not isinstance(checkpoint_io, Lfm2DirectLocalCheckpointIO):
        raise TypeError("checkpoint_io must be Lfm2DirectLocalCheckpointIO.")
    bridge = checkpoint_io.tokenizer_bridge
    bridge.verify_files()
    return {
        "saver_id": checkpoint_io.saver_id,
        "loader_id": checkpoint_io.loader_id,
        "resource_fingerprint": checkpoint_io.resource_fingerprint,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "target_tokenizer_package_fingerprint": (
            strict_tokenizer_bridge_target_package_fingerprint(bridge)
        ),
    }


def _model_binding_signature(model: Any) -> tuple[Any, ...]:
    """Capture object/storage bindings that freeze/eval is forbidden to replace."""

    return (
        tuple(
            (name, id(value), _storage_identity(value))
            for name, value in model.named_parameters()
        ),
        tuple(
            (name, id(value), _storage_identity(value))
            for name, value in model.named_buffers()
        ),
        tuple((name, id(module)) for name, module in model.named_modules()),
    )


def _runtime_policy(model: Any) -> dict[str, Any]:
    parameters = tuple(model.named_parameters())
    modules = tuple(model.named_modules())
    body = {
        "parameter_requires_grad": [
            [name, bool(parameter.requires_grad)]
            for name, parameter in parameters
        ],
        "parameter_gradients_absent": all(
            parameter.grad is None for _name, parameter in parameters
        ),
        "module_training_modes": [
            [name, bool(module.training)] for name, module in modules
        ],
    }
    return {**body, "runtime_policy_sha256": canonical_sha256(body)}


def _hook_value_is_nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return bool(value)
    return True


def _nonempty_hook_registries(
    owner: Any,
    required_names: tuple[str, ...],
    *,
    discover_instance_fields: bool,
) -> list[str]:
    discovered = (
        {
            str(name)
            for name in vars(owner)
            if "hook" in str(name).lower()
        }
        if discover_instance_fields
        else set()
    )
    names = sorted(set(required_names) | discovered)
    return [
        name
        for name in names
        if hasattr(owner, name)
        and _hook_value_is_nonempty(getattr(owner, name))
    ]


def _portable_hook_registry_audit(model: Any) -> dict[str, Any]:
    """Require a direct-local body whose behavior has no hook sidecar."""

    from torch.nn.modules import module as torch_module

    global_nonempty = _nonempty_hook_registries(
        torch_module,
        _GLOBAL_HOOK_REGISTRY_NAMES,
        discover_instance_fields=False,
    )
    module_nonempty = []
    modules = tuple(model.named_modules())
    for name, module in modules:
        nonempty = _nonempty_hook_registries(
            module,
            _MODULE_HOOK_REGISTRY_NAMES,
            discover_instance_fields=True,
        )
        if nonempty:
            module_nonempty.append(
                {
                    "module_path": str(name),
                    "registry_names": nonempty,
                }
            )
    body = {
        "global_nonempty_registry_names": global_nonempty,
        "module_nonempty_registries": module_nonempty,
        "inspected_module_count": len(modules),
        "portable_hook_registries_empty": (
            not global_nonempty and not module_nonempty
        ),
    }
    return {**body, "hook_registry_audit_sha256": canonical_sha256(body)}


def _inspect_model(
    model: Any,
    *,
    bindings: Lfm2PreCausalBodyResumeBindings,
    inspectors: InspectorRegistry | None,
    require_frozen: bool,
    expected_runtime_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "PyTorch is required for pre-causal body capsules."
        ) from exc
    if getattr(getattr(model, "config", None), "model_type", None) != "lfm2":
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule requires a concrete LFM2 child."
        )
    parameters = tuple(model.parameters())
    modules = tuple(model.modules())
    if not parameters or any(parameter.grad is not None for parameter in parameters):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body child must be materialized and gradient-free."
        )
    if require_frozen and (
        any(bool(parameter.requires_grad) for parameter in parameters)
        or any(bool(module.training) for module in modules)
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Serialized pre-causal body must be frozen and eval-only."
        )
    hook_registry_audit = _portable_hook_registry_audit(model)
    if hook_registry_audit["portable_hook_registries_empty"] is not True:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body cannot depend on live or global hook registries."
        )
    input_module = getattr(model, "get_input_embeddings", lambda: None)()
    output_module = getattr(model, "get_output_embeddings", lambda: None)()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or not isinstance(output_weight, torch.nn.Parameter)
        or input_weight is output_weight
        or _storage_identity(input_weight) == _storage_identity(output_weight)
        or bool(getattr(model.config, "tie_word_embeddings", True))
        or tuple(input_weight.shape) != tuple(output_weight.shape)
        or not bool(torch.equal(input_weight.detach(), output_weight.detach()))
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body requires distinct, value-equal untied token tables."
        )
    registry = inspectors or default_inspector_registry()
    graph = registry.inspect(
        model,
        model_name=bindings.child_name,
        tokenizer_fingerprint=bindings.target_tokenizer_semantics_sha256,
    )
    inventory = dict(serialized_inference_state_inventory(model))
    state_sha = fingerprint_model_state(model)
    runtime_sha = fingerprint_model_runtime_state(model)
    runtime_policy = _runtime_policy(model)
    identity_sha = fingerprint_lfm2_parent(model).identity_sha256
    body_sha = fingerprint_lfm2_causal_body_inventory(model)
    input_sha = _interface_tensor_sha256(input_weight)
    output_sha = _interface_tensor_sha256(output_weight)
    observed = {
        "model_state_sha256": state_sha,
        "materialization_runtime_state_sha256": runtime_sha,
        "runtime_policy": runtime_policy,
        "lfm2_identity_sha256": identity_sha,
        "causal_body_inventory_sha256": body_sha,
        "input_table_sha256": input_sha,
        "output_baseline_sha256": output_sha,
        "input_shape": [int(size) for size in input_weight.shape],
        "output_shape": [int(size) for size in output_weight.shape],
        "interface_dtype": str(input_weight.dtype),
        "input_and_output_parameters_distinct": True,
        "input_and_output_values_equal": True,
        "child_graph_fingerprint": graph.fingerprint,
        "child_structural_topology_fingerprint": (
            graph.structural_topology_fingerprint
        ),
        "target_unique_parameters": graph.unique_parameter_count,
        "target_logical_parameters": graph.logical_parameter_count,
        "serialized_inference_state_inventory": inventory,
        "serialized_inference_state_inventory_sha256": canonical_sha256(inventory),
        "parameters_frozen": not any(
            bool(parameter.requires_grad) for parameter in parameters
        ),
        "parameter_requires_grad_count": sum(
            int(bool(parameter.requires_grad)) for parameter in parameters
        ),
        "parameter_gradients_absent": True,
        "model_eval": not any(bool(module.training) for module in modules),
        "portable_hook_registries_empty": True,
        "hook_registry_audit": hook_registry_audit,
        "parent_sidecar_parameters": 0,
        "runtime_shared_parent_parameters": 0,
    }
    expected = {
        "model_state_sha256": bindings.body_model_state_sha256,
        "lfm2_identity_sha256": bindings.body_lfm2_identity_sha256,
        "causal_body_inventory_sha256": bindings.causal_body_inventory_sha256,
        "input_table_sha256": bindings.installed_input_table_sha256,
        "output_baseline_sha256": bindings.installed_output_baseline_sha256,
        "child_structural_topology_fingerprint": (
            bindings.child_structural_topology_fingerprint
        ),
        "target_unique_parameters": bindings.target_unique_parameters,
        "target_logical_parameters": bindings.target_logical_parameters,
        "serialized_inference_state_inventory_sha256": (
            bindings.target_state_inventory_sha256
        ),
    }
    if expected_runtime_sha256 is not None:
        expected["materialization_runtime_state_sha256"] = _digest(
            expected_runtime_sha256,
            "expected_runtime_sha256",
        )
    if any(observed.get(name) != value for name, value in expected.items()):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Live or freshly loaded body differs from its target/body bindings."
        )
    return graph, observed


def _freeze_for_publication(
    model: Any,
    *,
    bindings: Lfm2PreCausalBodyResumeBindings,
    inspectors: InspectorRegistry | None,
    pre_graph: Any,
    pre_model: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Install inference-only policy without changing the body coordinate."""

    binding_signature = _model_binding_signature(model)
    pre_requires_grad_count = int(pre_model["parameter_requires_grad_count"])
    pre_eval = bool(pre_model["model_eval"])
    model.requires_grad_(False)
    model.eval()
    if _model_binding_signature(model) != binding_signature:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Freeze/eval replaced a body object or tensor storage binding."
        )
    frozen_graph, frozen_model = _inspect_model(
        model,
        bindings=bindings,
        inspectors=inspectors,
        require_frozen=True,
    )
    invariant_fields = (
        "model_state_sha256",
        "lfm2_identity_sha256",
        "causal_body_inventory_sha256",
        "input_table_sha256",
        "output_baseline_sha256",
        "input_shape",
        "output_shape",
        "interface_dtype",
        "input_and_output_parameters_distinct",
        "input_and_output_values_equal",
        "child_structural_topology_fingerprint",
        "target_unique_parameters",
        "target_logical_parameters",
        "serialized_inference_state_inventory",
        "serialized_inference_state_inventory_sha256",
        "portable_hook_registries_empty",
        "hook_registry_audit",
    )
    if (
        any(pre_model[name] != frozen_model[name] for name in invariant_fields)
        or pre_graph.as_structural_topology_dict()
        != frozen_graph.as_structural_topology_dict()
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Freeze/eval changed the pre-causal body coordinate."
        )
    audit = {
        "boundary_action": (
            "authenticated-inference-policy-noop"
            if pre_requires_grad_count == 0 and pre_eval
            else "authenticated-inference-policy-transition"
        ),
        "incoming_body_runtime_state_sha256": pre_model[
            "materialization_runtime_state_sha256"
        ],
        "frozen_materialization_runtime_state_sha256": frozen_model[
            "materialization_runtime_state_sha256"
        ],
        "incoming_child_graph_fingerprint": pre_graph.fingerprint,
        "frozen_child_graph_fingerprint": frozen_graph.fingerprint,
        "compiled_child_graph_fingerprint": bindings.child_graph_fingerprint,
        "child_structural_topology_fingerprint": (
            bindings.child_structural_topology_fingerprint
        ),
        "incoming_parameter_requires_grad_count": pre_requires_grad_count,
        "frozen_parameter_requires_grad_count": 0,
        "incoming_model_eval": pre_eval,
        "frozen_model_eval": True,
        "state_preserved": True,
        "parameter_counts_preserved": True,
        "object_and_storage_bindings_preserved": True,
        "structural_topology_preserved": True,
    }
    return frozen_graph, frozen_model, audit


def _portable_model_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only the intentionally materialization-local runtime identity seal."""

    result = dict(value)
    result.pop("materialization_runtime_state_sha256", None)
    return result


def _verify_checkpoint_io_against_bindings(
    checkpoint_io: Lfm2DirectLocalCheckpointIO,
    bindings: Lfm2PreCausalBodyResumeBindings,
) -> dict[str, str]:
    values = _checkpoint_io_bindings(checkpoint_io)
    expected = {
        "bridge_artifact_sha256": bindings.bridge_artifact_sha256,
        "bridge_profile_sha256": bindings.bridge_profile_sha256,
        "target_tokenizer_semantics_sha256": (
            bindings.target_tokenizer_semantics_sha256
        ),
        "target_tokenizer_package_fingerprint": (
            bindings.target_tokenizer_package_fingerprint
        ),
    }
    if any(values[name] != value for name, value in expected.items()):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Direct-local checkpoint I/O differs from the capsule bridge bindings."
        )
    return values


def _root_path(value: str | os.PathLike[str]) -> tuple[Path, tuple[int, int]]:
    lexical = Path(value).expanduser().absolute()
    if _is_link_or_junction(lexical):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule cannot be a link or junction."
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule is missing."
        ) from exc
    if resolved != lexical:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule path cannot traverse a link."
        )
    root_stat = _real_directory(resolved, name="pre-causal body capsule")
    try:
        entries = tuple(os.scandir(resolved))
    except OSError as exc:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule cannot be enumerated."
        ) from exc
    if {entry.name for entry in entries} != _ROOT_INVENTORY:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule has missing or unexpected root entries."
        )
    for entry in entries:
        path = resolved / entry.name
        if entry.name == _CHECKPOINT_NAME:
            _real_directory(path, name="pre-causal body checkpoint")
        else:
            _regular_file(path, name=entry.name)
    return resolved, _stat_identity(root_stat)


def _open_capsule(
    reference: Lfm2PreCausalBodyResumeReference,
) -> dict[str, Any]:
    if not isinstance(reference, Lfm2PreCausalBodyResumeReference):
        raise TypeError("reference must be Lfm2PreCausalBodyResumeReference.")
    root, root_identity = _root_path(reference.path)
    receipt, receipt_raw = _read_canonical_json_with_raw(
        root / _RECEIPT_NAME,
        name=_RECEIPT_NAME,
    )
    receipt = _verify_seal(
        receipt,
        field_name="receipt_sha256",
        label="pre-causal body capsule receipt",
    )
    _validate_nonpromotable(receipt, label="pre-causal body capsule receipt")
    if (
        hashlib.sha256(receipt_raw).hexdigest() != reference.receipt_file_sha256
        or receipt.get("receipt_sha256") != reference.receipt_sha256
        or receipt.get("schema") != LFM2_PRE_CAUSAL_BODY_RESUME_RECEIPT_SCHEMA
        or receipt.get("artifact_kind") != _ARTIFACT_KIND
        or receipt.get("bindings_sha256") != reference.bindings_sha256
        or receipt.get("manifest_sha256") != reference.manifest_sha256
        or receipt.get("checkpoint_inventory_sha256")
        != reference.checkpoint_inventory_sha256
        or receipt.get("operation_audit") != _ZERO_OPERATION_AUDIT
        or receipt.get("publication_fresh_reload_materializations") != 1
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body receipt differs from its external pins."
        )
    manifest, manifest_raw = _read_canonical_json_with_raw(
        root / _MANIFEST_NAME,
        name=_MANIFEST_NAME,
    )
    manifest = _verify_seal(
        manifest,
        field_name="manifest_sha256",
        label="pre-causal body capsule manifest",
    )
    _validate_nonpromotable(manifest, label="pre-causal body capsule manifest")
    manifest_record = receipt.get("manifest_file")
    manifest_model = manifest.get("model")
    hook_registry_audit = (
        manifest_model.get("hook_registry_audit")
        if isinstance(manifest_model, Mapping)
        else None
    )
    hook_registry_body = (
        dict(hook_registry_audit)
        if isinstance(hook_registry_audit, Mapping)
        else {}
    )
    hook_registry_sha256 = hook_registry_body.pop(
        "hook_registry_audit_sha256",
        None,
    )
    if (
        manifest.get("schema") != LFM2_PRE_CAUSAL_BODY_RESUME_MANIFEST_SCHEMA
        or manifest.get("artifact_kind") != _ARTIFACT_KIND
        or manifest.get("manifest_sha256") != reference.manifest_sha256
        or manifest.get("bindings_sha256") != reference.bindings_sha256
        or not isinstance(manifest_record, Mapping)
        or manifest_record.get("path") != _MANIFEST_NAME
        or manifest_record.get("size_bytes") != len(manifest_raw)
        or manifest_record.get("sha256")
        != hashlib.sha256(manifest_raw).hexdigest()
        or manifest.get("operation_audit") != _ZERO_OPERATION_AUDIT
        or manifest.get("dense_full_body_checkpoint") is not True
        or manifest.get("input_and_output_parameters_distinct") is not True
        or manifest.get("input_and_output_values_equal") is not True
        or manifest.get("no_parent_model_or_tensor_payload") is not True
        or manifest.get("no_pickle_files") is not True
        or not isinstance(manifest_model, Mapping)
        or manifest_model.get("portable_hook_registries_empty") is not True
        or not isinstance(hook_registry_audit, Mapping)
        or hook_registry_body.get("global_nonempty_registry_names") != []
        or hook_registry_body.get("module_nonempty_registries") != []
        or hook_registry_body.get("portable_hook_registries_empty") is not True
        or canonical_sha256(hook_registry_body) != hook_registry_sha256
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule manifest contract is invalid."
        )
    bindings = Lfm2PreCausalBodyResumeBindings.from_mapping(
        manifest.get("bindings")
    )
    checkpoint_io_binding = manifest.get("checkpoint_io")
    expected_checkpoint_io_fields = {
        "saver_id",
        "loader_id",
        "resource_fingerprint",
        "bridge_artifact_sha256",
        "bridge_profile_sha256",
        "target_tokenizer_semantics_sha256",
        "target_tokenizer_package_fingerprint",
    }
    if (
        not isinstance(checkpoint_io_binding, Mapping)
        or set(checkpoint_io_binding) != expected_checkpoint_io_fields
        or not isinstance(checkpoint_io_binding.get("saver_id"), str)
        or not checkpoint_io_binding["saver_id"].strip()
        or not isinstance(checkpoint_io_binding.get("loader_id"), str)
        or not checkpoint_io_binding["loader_id"].strip()
        or checkpoint_io_binding.get("bridge_artifact_sha256")
        != bindings.bridge_artifact_sha256
        or checkpoint_io_binding.get("bridge_profile_sha256")
        != bindings.bridge_profile_sha256
        or checkpoint_io_binding.get("target_tokenizer_semantics_sha256")
        != bindings.target_tokenizer_semantics_sha256
        or checkpoint_io_binding.get("target_tokenizer_package_fingerprint")
        != bindings.target_tokenizer_package_fingerprint
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body checkpoint I/O binding is invalid."
        )
    _digest(
        checkpoint_io_binding.get("resource_fingerprint"),
        "checkpoint I/O resource_fingerprint",
    )
    if (
        bindings.bindings_sha256 != reference.bindings_sha256
        or manifest.get("body_runtime_state_sha256")
        != bindings.body_runtime_state_sha256
        or receipt.get("body_runtime_state_sha256")
        != bindings.body_runtime_state_sha256
        or manifest.get("compiled_child_graph_fingerprint")
        != bindings.child_graph_fingerprint
        or receipt.get("compiled_child_graph_fingerprint")
        != bindings.child_graph_fingerprint
        or manifest.get("child_structural_topology_fingerprint")
        != bindings.child_structural_topology_fingerprint
        or receipt.get("child_structural_topology_fingerprint")
        != bindings.child_structural_topology_fingerprint
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body bindings differ from the external pin."
        )
    inventory = _checkpoint_inventory(root / _CHECKPOINT_NAME)
    inventory_sha = canonical_sha256(inventory)
    if (
        inventory != manifest.get("checkpoint_inventory")
        or inventory_sha != manifest.get("checkpoint_inventory_sha256")
        or inventory_sha != reference.checkpoint_inventory_sha256
        or receipt.get("checkpoint_inventory") != inventory
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body checkpoint bytes differ from their receipt."
        )
    current_root, current_identity = _root_path(root)
    if current_root != root or current_identity != root_identity:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Pre-causal body capsule changed during authentication."
        )
    return {
        "root": root,
        "root_identity": root_identity,
        "receipt": receipt,
        "manifest": manifest,
        "bindings": bindings,
    }


def authenticate_lfm2_pre_causal_body_resume_capsule(
    reference: Lfm2PreCausalBodyResumeReference,
    *,
    checkpoint_io: Lfm2DirectLocalCheckpointIO | None = None,
) -> Lfm2PreCausalBodyResumeAuthentication:
    """Authenticate every capsule byte and seal without loading its model."""

    opened = _open_capsule(reference)
    bindings = opened["bindings"]
    if checkpoint_io is not None:
        io_bindings = _verify_checkpoint_io_against_bindings(
            checkpoint_io,
            bindings,
        )
        if opened["manifest"].get("checkpoint_io") != io_bindings:
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Resume checkpoint I/O differs from the publication resource."
            )
    return Lfm2PreCausalBodyResumeAuthentication(
        path=opened["root"],
        root_identity=opened["root_identity"],
        bindings=bindings,
        manifest=MappingProxyType(dict(opened["manifest"])),
        receipt=MappingProxyType(dict(opened["receipt"])),
        reference=reference,
    )


def publish_lfm2_pre_causal_body_resume_capsule(
    *,
    model: Any,
    checkpoint_io: Lfm2DirectLocalCheckpointIO,
    bindings: Lfm2PreCausalBodyResumeBindings,
    output_directory: str | os.PathLike[str],
    inspectors: InspectorRegistry | None = None,
) -> Lfm2PreCausalBodyResumeReference:
    """Atomically publish one authenticated post-body/pre-causal restart point."""

    if not isinstance(bindings, Lfm2PreCausalBodyResumeBindings):
        raise TypeError("bindings must be Lfm2PreCausalBodyResumeBindings.")
    io_bindings = _verify_checkpoint_io_against_bindings(checkpoint_io, bindings)
    incoming_graph, incoming_model = _inspect_model(
        model,
        bindings=bindings,
        inspectors=inspectors,
        require_frozen=False,
        expected_runtime_sha256=bindings.body_runtime_state_sha256,
    )
    live_graph, live_model, runtime_transition = _freeze_for_publication(
        model,
        bindings=bindings,
        inspectors=inspectors,
        pre_graph=incoming_graph,
        pre_model=incoming_model,
    )
    output = _new_output(output_directory)
    stage = Path(
        tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=str(output.parent))
    ).absolute()
    stage_identity = _stat_identity(stage.lstat())
    moved = False
    try:
        checkpoint = stage / _CHECKPOINT_NAME
        checkpoint.mkdir()
        checkpoint_io.save(model, checkpoint)
        if (
            fingerprint_model_state(model) != live_model["model_state_sha256"]
            or fingerprint_model_runtime_state(model)
            != live_model["materialization_runtime_state_sha256"]
            or any(parameter.grad is not None for parameter in model.parameters())
        ):
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Direct-local saver mutated the live pre-causal child."
            )
        checkpoint_inventory = _checkpoint_inventory(checkpoint)
        checkpoint_inventory_sha = canonical_sha256(checkpoint_inventory)
        reloaded = checkpoint_io.load(checkpoint)
        _assert_disjoint_models(model, reloaded)
        reload_graph, reload_model = _inspect_model(
            reloaded,
            bindings=bindings,
            inspectors=inspectors,
            require_frozen=True,
        )
        if (
            _portable_model_record(reload_model)
            != _portable_model_record(live_model)
            or reload_graph.as_structural_topology_dict()
            != live_graph.as_structural_topology_dict()
        ):
            raise Lfm2PreCausalBodyResumeCapsuleError(
                "Fresh direct-local reload changed the pre-causal body coordinate."
            )
        del reloaded
        manifest_body = {
            "schema": LFM2_PRE_CAUSAL_BODY_RESUME_MANIFEST_SCHEMA,
            "artifact_kind": _ARTIFACT_KIND,
            **_NONPROMOTABLE_FLAGS,
            "bindings": bindings.as_dict(),
            "bindings_sha256": bindings.bindings_sha256,
            "checkpoint_io": io_bindings,
            "checkpoint_inventory": checkpoint_inventory,
            "checkpoint_inventory_sha256": checkpoint_inventory_sha,
            "model": live_model,
            "body_runtime_state_sha256": bindings.body_runtime_state_sha256,
            "compiled_child_graph_fingerprint": bindings.child_graph_fingerprint,
            "child_structural_topology_fingerprint": (
                bindings.child_structural_topology_fingerprint
            ),
            "runtime_boundary_audit": runtime_transition,
            "dense_full_body_checkpoint": True,
            "input_and_output_parameters_distinct": True,
            "input_and_output_values_equal": True,
            "fresh_reload_verified": True,
            "publication_fresh_reload_materializations": 1,
            "no_parent_model_or_tensor_payload": True,
            "no_pickle_files": True,
            "resume_boundary": "after-moving-gauge-w2-before-causal-evidence",
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        manifest = _sealed(manifest_body, "manifest_sha256")
        manifest_path = stage / _MANIFEST_NAME
        _write_canonical_json(manifest_path, manifest)
        manifest_record = _file_record(manifest_path)
        receipt_body = {
            "schema": LFM2_PRE_CAUSAL_BODY_RESUME_RECEIPT_SCHEMA,
            "artifact_kind": _ARTIFACT_KIND,
            **_NONPROMOTABLE_FLAGS,
            "bindings_sha256": bindings.bindings_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file": manifest_record,
            "checkpoint_inventory": checkpoint_inventory,
            "checkpoint_inventory_sha256": checkpoint_inventory_sha,
            "body_model_state_sha256": bindings.body_model_state_sha256,
            "body_runtime_state_sha256": bindings.body_runtime_state_sha256,
            "body_lfm2_identity_sha256": bindings.body_lfm2_identity_sha256,
            "causal_body_inventory_sha256": bindings.causal_body_inventory_sha256,
            "child_structural_topology_fingerprint": (
                bindings.child_structural_topology_fingerprint
            ),
            "compiled_child_graph_fingerprint": bindings.child_graph_fingerprint,
            "runtime_boundary_action": runtime_transition["boundary_action"],
            "target_unique_parameters": bindings.target_unique_parameters,
            "fresh_reload_verified": True,
            "publication_fresh_reload_materializations": 1,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        receipt = _sealed(receipt_body, "receipt_sha256")
        receipt_path = stage / _RECEIPT_NAME
        _write_canonical_json(receipt_path, receipt)
        receipt_file_sha = _file_sha256(receipt_path)
        _fsync_directory(checkpoint)
        _fsync_directory(stage)
        staged_reference = Lfm2PreCausalBodyResumeReference(
            path=stage,
            receipt_sha256=receipt["receipt_sha256"],
            receipt_file_sha256=receipt_file_sha,
            manifest_sha256=manifest["manifest_sha256"],
            bindings_sha256=bindings.bindings_sha256,
            checkpoint_inventory_sha256=checkpoint_inventory_sha,
        )
        _open_capsule(staged_reference)
        _atomic_rename_noreplace(stage, output)
        moved = True
        _fsync_directory(output.parent)
        reference = Lfm2PreCausalBodyResumeReference(
            path=output,
            receipt_sha256=staged_reference.receipt_sha256,
            receipt_file_sha256=staged_reference.receipt_file_sha256,
            manifest_sha256=staged_reference.manifest_sha256,
            bindings_sha256=staged_reference.bindings_sha256,
            checkpoint_inventory_sha256=(
                staged_reference.checkpoint_inventory_sha256
            ),
        )
        _open_capsule(reference)
        return reference
    except FileExistsError as exc:
        raise Lfm2PreCausalBodyResumeCapsuleError(
            f"Pre-causal body capsule destination already exists: {output}."
        ) from exc
    finally:
        if not moved and _lexists(stage):
            _safe_remove_stage(
                stage,
                parent=output.parent,
                identity=stage_identity,
            )


def load_lfm2_pre_causal_body_resume_capsule(
    reference: Lfm2PreCausalBodyResumeReference,
    *,
    checkpoint_io: Lfm2DirectLocalCheckpointIO,
    inspectors: InspectorRegistry | None = None,
) -> Lfm2PreCausalBodyResumeCapsule:
    """Freshly load an externally pinned, parent-free pre-causal body."""

    authentication = authenticate_lfm2_pre_causal_body_resume_capsule(
        reference,
        checkpoint_io=checkpoint_io,
    )
    root = authentication.path
    root_identity = authentication.root_identity
    bindings = authentication.bindings
    manifest = authentication.manifest
    model = checkpoint_io.load(root / _CHECKPOINT_NAME)
    _graph, observed = _inspect_model(
        model,
        bindings=bindings,
        inspectors=inspectors,
        require_frozen=True,
    )
    manifest_model = manifest.get("model")
    if (
        not isinstance(manifest_model, Mapping)
        or _portable_model_record(observed)
        != _portable_model_record(manifest_model)
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Freshly loaded pre-causal body differs from its manifest."
        )
    if _checkpoint_inventory(root / _CHECKPOINT_NAME) != manifest.get(
        "checkpoint_inventory"
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Checkpoint changed while the pre-causal body was loaded."
        )
    current_root, current_identity = _root_path(root)
    if (
        current_root != root
        or current_identity != root_identity
        or fingerprint_model_state(model) != observed["model_state_sha256"]
        or fingerprint_model_runtime_state(model)
        != observed["materialization_runtime_state_sha256"]
    ):
        raise Lfm2PreCausalBodyResumeCapsuleError(
            "Capsule root changed while the pre-causal body was loaded."
        )
    return Lfm2PreCausalBodyResumeCapsule(
        path=root,
        model=model,
        bindings=bindings,
        manifest=MappingProxyType(dict(manifest)),
        receipt=MappingProxyType(dict(authentication.receipt)),
        reference=reference,
    )


__all__ = [
    "LFM2_PRE_CAUSAL_BODY_RESUME_BINDINGS_SCHEMA",
    "LFM2_PRE_CAUSAL_BODY_RESUME_MANIFEST_SCHEMA",
    "LFM2_PRE_CAUSAL_BODY_RESUME_RECEIPT_SCHEMA",
    "Lfm2PreCausalBodyResumeBindings",
    "Lfm2PreCausalBodyResumeAuthentication",
    "Lfm2PreCausalBodyResumeCapsule",
    "Lfm2PreCausalBodyResumeCapsuleError",
    "Lfm2PreCausalBodyResumeReference",
    "authenticate_lfm2_pre_causal_body_resume_capsule",
    "load_lfm2_pre_causal_body_resume_capsule",
    "publish_lfm2_pre_causal_body_resume_capsule",
]
