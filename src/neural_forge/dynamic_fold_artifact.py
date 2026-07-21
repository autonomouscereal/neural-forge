"""Atomic fresh-reload publication for dynamic folded children."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import marshal
import os
import pickletools
import re
import shutil
import struct
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import CodeType
from typing import Any

from .dynamic_circuit_ir import (
    ModelCircuitGraph,
    canonical_json_bytes,
    canonical_sha256,
)
from .dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
    DynamicBehaviorExecutionError,
    execution_contract_satisfied,
    inference_execution_contract_from_manifest,
    observe_model_execution,
)
from .dynamic_fold_executor import (
    FoldDecision,
    _FoldOnlyTorchGuard,
    _WholeModelSnapshot,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_model_inspector import InspectorRegistry, default_inspector_registry
from .dynamic_target_synthesis import (
    SCHEMA as TARGET_SYNTHESIS_SCHEMA,
    DynamicTargetSynthesisResult,
    serialized_inference_state_inventory,
    validate_registered_inference_state,
    validate_structural_runtime_buffers,
    verify_compiler_authenticated_target,
)


LEGACY_SCHEMA = "neural-forge-dynamic-fold-artifact-v2"
SCHEMA = "neural-forge-dynamic-fold-artifact-v3"
PROVENANCE_SCHEMA = "neural-forge-dynamic-fold-provenance-v1"
CALLABLE_FINGERPRINT_SCHEMA = "neural-forge-local-callable-fingerprint-v1"
_PROVENANCE_DIRECTORY = "provenance"
_PROVENANCE_FILENAMES = {
    "plan": "plan.json",
    "run": "run.json",
    "child_graph": "child_graph.json",
    "parent_graphs": "parent_graphs.json",
    "target": "target.json",
}
_FORBIDDEN_NAME_PARTS = (
    "admission",
    "adapter",
    "cache",
    "checkpoint",
    "coordinator",
    "donor",
    "optimizer",
    "parent",
    "pending",
    "resume",
    "scheduler",
    "sidecar",
    "teacher",
    "trainer",
    "training_args",
)
_FORBIDDEN_CONTENT_TOKENS = frozenset(
    {
        "admission",
        "checkpoint",
        "coordinator",
        "donor",
        "pending",
        "resume",
        "teacher",
    }
)
_CACHE_REFERENCE_QUALIFIERS = frozenset(
    {
        "artifact",
        "directory",
        "file",
        "key",
        "manifest",
        "path",
        "payload",
        "reference",
        "sidecar",
        "state",
        "store",
    }
)
_ALLOWED_MODEL_CACHE_KEYS = frozenset(
    {
        "cache_implementation",
        "return_legacy_cache",
        "use_cache",
    }
)
_JSON_LITERAL_CONTAINERS = frozenset(
    {
        "added_tokens",
        "added_tokens_decoder",
        "merges",
        "tokens",
        "vocab",
    }
)
_MAX_STRUCTURED_INVENTORY_BYTES = 64 * 1024**2
STANDALONE_FOLD_ARTIFACT_INVENTORY_SCHEMA = (
    "neural-forge-standalone-fold-artifact-inventory-v1"
)
_LFM2_STANDALONE_ARTIFACT_METADATA = "neural_forge_lfm2_artifact.json"
_LFM2_STANDALONE_ARTIFACT_SCHEMA = "neural-forge-lfm2-cpcf-v5-standalone-candidate-v1"
_SAFETENSORS_SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors\Z")
_CANONICAL_TOKENIZER_FILES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_STANDALONE_FILE_ROLES = frozenset(
    {
        "artifact_metadata",
        "generation_config",
        "model_config",
        "model_tensor",
        "model_tensor_index",
        "tokenizer_resource",
    }
)


class DynamicFoldArtifactError(RuntimeError):
    """A folded child could not be published or freshly authenticated."""


@dataclass(frozen=True, slots=True)
class DynamicFoldArtifactCertificate:
    path: str
    plan_fingerprint: str
    run_fingerprint: str
    child_graph_fingerprint: str
    child_state_fingerprint: str
    exact_unique_parameters: int
    artifact_tree_sha256: str
    artifact_file_count: int
    artifact_total_bytes: int
    fresh_reload_decision: FoldDecision
    saver_id: str
    loader_id: str
    compilation_optimizer_constructions: int
    compilation_optimizer_steps: int
    compilation_backward_calls: int
    compilation_autograd_grad_calls: int
    optimizer_constructions: int
    optimizer_steps: int
    backward_calls: int
    autograd_grad_calls: int
    certificate_sha256: str
    evidence_fingerprint: str = "0" * 64
    behavior_gate_id: str = ""
    behavior_gate_version: int = 0
    behavior_baseline_fingerprint: str = "0" * 64
    behavior_gate_manifest_fingerprint: str = "0" * 64
    behavior_gate_manifest: Mapping[str, Any] | None = None
    fresh_reload_state_inventory: Mapping[str, int | str] | None = None
    fresh_reload_module_forward_calls: int = 0
    fresh_reload_root_forward_calls: int = 0
    fresh_reload_inference_entrypoint_calls: int = 0
    preparation_guard_applied: bool = False
    preparation_optimizer_constructions: int = 0
    preparation_optimizer_steps: int = 0
    preparation_backward_calls: int = 0
    preparation_autograd_grad_calls: int = 0
    preparation_parent_module_forward_calls: int = 0
    preparation_parent_root_forward_calls: int = 0
    publication_guard_covers_save_load: bool = False
    publication_optimizer_constructions: int = 0
    publication_optimizer_steps: int = 0
    publication_backward_calls: int = 0
    publication_autograd_grad_calls: int = 0
    publication_parent_module_forward_calls: int = 0
    publication_parent_root_forward_calls: int = 0
    provenance_plan_sha256: str = "0" * 64
    provenance_run_sha256: str = "0" * 64
    provenance_child_graph_sha256: str = "0" * 64
    provenance_parent_graphs_sha256: str = "0" * 64
    provenance_parent_graphs_fingerprint: str = "0" * 64
    provenance_target_sha256: str = "0" * 64
    provenance_target_fingerprint: str = "0" * 64
    dynamic_target_provenance_present: bool = False
    provenance_tree_sha256: str = "0" * 64
    loader_path_challenge_performed: bool = False
    loader_path_challenge_exception_type: str = ""
    saver_implementation_fingerprint: str = "0" * 64
    saver_resource_fingerprint: str = "0" * 64
    saver_resource_fingerprint_declared: bool = False
    loader_implementation_fingerprint: str = "0" * 64
    loader_resource_fingerprint: str = "0" * 64
    loader_resource_fingerprint_declared: bool = False
    certificate_schema: str = SCHEMA

    def body(self) -> dict[str, Any]:
        body = {
            "schema": self.certificate_schema,
            "path": self.path,
            "plan_fingerprint": self.plan_fingerprint,
            "run_fingerprint": self.run_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "child_graph_fingerprint": self.child_graph_fingerprint,
            "child_state_fingerprint": self.child_state_fingerprint,
            "exact_unique_parameters": self.exact_unique_parameters,
            "artifact_tree_sha256": self.artifact_tree_sha256,
            "artifact_file_count": self.artifact_file_count,
            "artifact_total_bytes": self.artifact_total_bytes,
            "fresh_reload_decision": self.fresh_reload_decision.as_dict(),
            "saver_id": self.saver_id,
            "loader_id": self.loader_id,
            "behavior_gate_id": self.behavior_gate_id,
            "behavior_gate_version": self.behavior_gate_version,
            "behavior_baseline_fingerprint": (self.behavior_baseline_fingerprint),
            "behavior_gate_manifest_fingerprint": (
                self.behavior_gate_manifest_fingerprint
            ),
            "behavior_gate_manifest": (
                {}
                if self.behavior_gate_manifest is None
                else dict(self.behavior_gate_manifest)
            ),
            "fresh_reload_state_inventory": (
                {}
                if self.fresh_reload_state_inventory is None
                else dict(self.fresh_reload_state_inventory)
            ),
            "fresh_reload_module_forward_calls": (
                self.fresh_reload_module_forward_calls
            ),
            "fresh_reload_root_forward_calls": self.fresh_reload_root_forward_calls,
            "fresh_reload_model_execution_observed": (
                self.fresh_reload_module_forward_calls > 0
            ),
            "provenance_schema": PROVENANCE_SCHEMA,
            "provenance_directory": _PROVENANCE_DIRECTORY,
            "provenance_filenames": dict(_PROVENANCE_FILENAMES),
            "provenance_plan_sha256": self.provenance_plan_sha256,
            "provenance_run_sha256": self.provenance_run_sha256,
            "provenance_child_graph_sha256": self.provenance_child_graph_sha256,
            "provenance_parent_graphs_sha256": (self.provenance_parent_graphs_sha256),
            "provenance_parent_graphs_fingerprint": (
                self.provenance_parent_graphs_fingerprint
            ),
            "provenance_target_sha256": self.provenance_target_sha256,
            "provenance_target_fingerprint": self.provenance_target_fingerprint,
            "dynamic_target_provenance_present": (
                self.dynamic_target_provenance_present
            ),
            "provenance_tree_sha256": self.provenance_tree_sha256,
            "provenance_documents_embedded": True,
            "loader_path_challenge_performed": (self.loader_path_challenge_performed),
            "loader_path_challenge_exception_type": (
                self.loader_path_challenge_exception_type
            ),
            "loader_path_challenge_passed": self.loader_path_challenge_performed,
            "callable_fingerprint_schema": CALLABLE_FINGERPRINT_SCHEMA,
            "saver_implementation_fingerprint": (self.saver_implementation_fingerprint),
            "saver_resource_fingerprint": self.saver_resource_fingerprint,
            "saver_resource_fingerprint_declared": (
                self.saver_resource_fingerprint_declared
            ),
            "loader_implementation_fingerprint": (
                self.loader_implementation_fingerprint
            ),
            "loader_resource_fingerprint": self.loader_resource_fingerprint,
            "loader_resource_fingerprint_declared": (
                self.loader_resource_fingerprint_declared
            ),
            "fresh_reload_performed": True,
            "exact_budget_match": True,
            "standalone_child_directory": "child",
            "forbidden_named_sidecars_present": False,
            "complete_artifact_tree_hashed": True,
            "standalone_reload_returned_disjoint_model": True,
            "loader_implementation_requires_external_audit": True,
            "best_effort_fold_only_guard_active": True,
            "all_reloaded_parameters_frozen_during_behavior_callback": True,
            "historical_zero_training_proven_for_arbitrary_callbacks": False,
            "callback_implementations_require_external_audit": True,
            "preparation_guard_applied": self.preparation_guard_applied,
            "preparation_optimizer_constructions": (
                self.preparation_optimizer_constructions
            ),
            "preparation_optimizer_steps": self.preparation_optimizer_steps,
            "preparation_backward_calls": self.preparation_backward_calls,
            "preparation_autograd_grad_calls": (self.preparation_autograd_grad_calls),
            "preparation_parent_module_forward_calls": (
                self.preparation_parent_module_forward_calls
            ),
            "preparation_parent_root_forward_calls": (
                self.preparation_parent_root_forward_calls
            ),
            "publication_guard_covers_save_load": (
                self.publication_guard_covers_save_load
            ),
            "publication_optimizer_constructions": (
                self.publication_optimizer_constructions
            ),
            "publication_optimizer_steps": self.publication_optimizer_steps,
            "publication_backward_calls": self.publication_backward_calls,
            "publication_autograd_grad_calls": (self.publication_autograd_grad_calls),
            "publication_parent_module_forward_calls": (
                self.publication_parent_module_forward_calls
            ),
            "publication_parent_root_forward_calls": (
                self.publication_parent_root_forward_calls
            ),
            "fresh_reload_source": "retained-final-destination",
            "final_destination_reload_performed": True,
            "fresh_reload_registered_inference_state_revalidated": (
                self.dynamic_target_provenance_present
            ),
            "fresh_reload_structural_buffer_policy_revalidated": (
                self.dynamic_target_provenance_present
            ),
            "compilation_optimizer_constructions": (
                self.compilation_optimizer_constructions
            ),
            "compilation_optimizer_steps": self.compilation_optimizer_steps,
            "compilation_backward_calls": self.compilation_backward_calls,
            "compilation_autograd_grad_calls": (self.compilation_autograd_grad_calls),
            "fresh_reload_optimizer_constructions": self.optimizer_constructions,
            "fresh_reload_optimizer_steps": self.optimizer_steps,
            "fresh_reload_backward_calls": self.backward_calls,
            "fresh_reload_autograd_grad_calls": self.autograd_grad_calls,
            "optimizer_constructions": self.optimizer_constructions,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "autograd_grad_calls": self.autograd_grad_calls,
        }
        if self.certificate_schema == SCHEMA:
            body["fresh_reload_inference_entrypoint_calls"] = (
                self.fresh_reload_inference_entrypoint_calls
            )
            body["fresh_reload_declared_inference_path_observed"] = (
                self.fresh_reload_inference_entrypoint_calls > 0
            )
        return body

    def as_dict(self) -> dict[str, Any]:
        return {**self.body(), "certificate_sha256": self.certificate_sha256}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicFoldArtifactError(f"{name} must be a nonempty string.")
    return value.strip()


_GUARD_COUNT_NAMES = (
    "optimizer_constructions",
    "optimizer_steps",
    "backward_calls",
    "autograd_grad_calls",
)


def _guard_counts(
    value: Mapping[str, int] | None,
    *,
    name: str,
) -> dict[str, int]:
    source = {} if value is None else dict(value)
    unknown = set(source) - set(_GUARD_COUNT_NAMES)
    if unknown:
        raise DynamicFoldArtifactError(
            f"{name} contains unknown counters: {sorted(unknown)}."
        )
    result = {}
    for key in _GUARD_COUNT_NAMES:
        item = source.get(key, 0)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise DynamicFoldArtifactError(
                f"{name} {key} must be a non-negative integer."
            )
        result[key] = item
    return result


def _forward_counts(
    value: Mapping[str, int] | None,
    *,
    name: str,
) -> dict[str, int]:
    source = {} if value is None else dict(value)
    expected = {"module_forward_calls", "root_forward_calls"}
    if set(source) - expected:
        raise DynamicFoldArtifactError(
            f"{name} contains unknown forward counters: {sorted(set(source) - expected)}."
        )
    result = {}
    for key in sorted(expected):
        item = source.get(key, 0)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise DynamicFoldArtifactError(
                f"{name} {key} must be a non-negative integer."
            )
        result[key] = item
    if result["root_forward_calls"] > result["module_forward_calls"]:
        raise DynamicFoldArtifactError(
            f"{name} root forward calls exceed all module forward calls."
        )
    return result


def _require_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicFoldArtifactError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _stable_resource_value(value: Any, *, depth: int = 0) -> Any:
    """Describe local callable bindings without object IDs or unsafe reprs."""

    if isinstance(value, (str, int, bool, type(None))):
        return value
    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else {"kind": "nonfinite_float", "value": str(value)}
        )
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "type": _qualified_type(value),
            "name": value.name,
        }
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value.expanduser().absolute())}
    if isinstance(value, CodeType):
        return {
            "kind": "code",
            "sha256": hashlib.sha256(marshal.dumps(value)).hexdigest(),
        }
    if depth >= 3:
        return {"kind": "object", "type": _qualified_type(value)}
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        if len(items) > 64:
            return {
                "kind": "mapping",
                "type": _qualified_type(value),
                "length": len(items),
            }
        return {
            "kind": "mapping",
            "items": [
                [
                    str(key),
                    _stable_resource_value(item, depth=depth + 1),
                ]
                for key, item in items
            ],
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 64:
            return {
                "kind": "sequence",
                "type": _qualified_type(value),
                "length": len(value),
            }
        return {
            "kind": "sequence",
            "type": _qualified_type(value),
            "items": [_stable_resource_value(item, depth=depth + 1) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_stable_resource_value(item, depth=depth + 1) for item in value]
        return {
            "kind": "set",
            "type": _qualified_type(value),
            "items": sorted(items, key=canonical_json_bytes),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": _qualified_type(value),
            "fields": {
                descriptor.name: _stable_resource_value(
                    getattr(value, descriptor.name),
                    depth=depth + 1,
                )
                for descriptor in fields(value)
            },
        }
    for attribute in (
        "resource_fingerprint",
        "baseline_fingerprint",
        "fingerprint",
        "cache_key",
        "source_sha256",
    ):
        try:
            identity = getattr(value, attribute, None)
        except Exception:
            continue
        if (
            isinstance(identity, str)
            and len(identity) == 64
            and all(character in "0123456789abcdef" for character in identity)
        ):
            return {
                "kind": "identified_object",
                "type": _qualified_type(value),
                "identity_field": attribute,
                "identity": identity,
            }
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None and dtype is not None:
        try:
            normalized_shape = [int(item) for item in shape]
        except (TypeError, ValueError):
            normalized_shape = []
        return {
            "kind": "tensor_like",
            "type": _qualified_type(value),
            "shape": normalized_shape,
            "dtype": str(dtype),
            "device": str(device),
        }
    if _qualified_type(value) in {
        "torch.device",
        "torch.dtype",
        "torch.layout",
        "torch.memory_format",
    }:
        return {
            "kind": "torch_descriptor",
            "type": _qualified_type(value),
            "value": str(value),
        }
    try:
        attributes = vars(value)
    except TypeError:
        attributes = None
    if isinstance(attributes, Mapping) and attributes:
        items = sorted(attributes.items(), key=lambda item: str(item[0]))
        return {
            "kind": "object",
            "type": _qualified_type(value),
            "attributes": {
                str(key): _stable_resource_value(item, depth=depth + 1)
                for key, item in items[:64]
            },
            "truncated": len(items) > 64,
        }
    return {"kind": "object", "type": _qualified_type(value)}


def _callable_fingerprints(value: Any, *, name: str) -> tuple[str, str, bool]:
    """Bind local implementation code and any declared/captured resources."""

    if not callable(value):
        raise DynamicFoldArtifactError(f"{name} must be callable.")
    try:
        unwrapped = inspect.unwrap(value)
    except (TypeError, ValueError):
        unwrapped = value
    code_owner = unwrapped
    code = getattr(code_owner, "__code__", None)
    if not isinstance(code, CodeType):
        call = getattr(code_owner, "__call__", None)
        code = getattr(call, "__code__", None)
        if isinstance(code, CodeType):
            code_owner = call
    module = str(
        getattr(code_owner, "__module__", None)
        or getattr(type(unwrapped), "__module__", "")
    )
    qualname = str(
        getattr(code_owner, "__qualname__", None)
        or getattr(type(unwrapped), "__qualname__", "")
    )
    source_sha256 = None
    try:
        source = inspect.getsource(code_owner).replace("\r\n", "\n")
    except (OSError, TypeError):
        source = None
    if source is not None:
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    code_sha256 = (
        None
        if not isinstance(code, CodeType)
        else hashlib.sha256(marshal.dumps(code)).hexdigest()
    )
    implementation = canonical_sha256(
        {
            "schema": CALLABLE_FINGERPRINT_SCHEMA,
            "kind": "implementation",
            "module": module,
            "qualname": qualname,
            "callable_type": _qualified_type(unwrapped),
            "source_sha256": source_sha256,
            "code_sha256": code_sha256,
        }
    )

    owners = [unwrapped]
    bound_owner = getattr(unwrapped, "__self__", None)
    if bound_owner is not None:
        owners.append(bound_owner)
    for owner in owners:
        for attribute in ("resource_fingerprint", "fingerprint"):
            try:
                declared = getattr(owner, attribute, None)
            except Exception as exc:
                raise DynamicFoldArtifactError(
                    f"Could not read {name} {attribute}."
                ) from exc
            if callable(declared) or declared is None:
                continue
            return (
                implementation,
                _require_digest(
                    declared,
                    f"{name} {attribute}",
                ),
                True,
            )

    closure_values = []
    closure = getattr(code_owner, "__closure__", None) or ()
    for cell in closure:
        try:
            item = cell.cell_contents
        except ValueError:
            item = None
        closure_values.append(_stable_resource_value(item))
    defaults = _stable_resource_value(getattr(code_owner, "__defaults__", None))
    kwdefaults = _stable_resource_value(getattr(code_owner, "__kwdefaults__", None))
    object_state: dict[str, Any] = {}
    state_owner = bound_owner if bound_owner is not None else unwrapped
    try:
        state_items = vars(state_owner).items()
    except TypeError:
        state_items = ()
    for key, item in sorted(state_items, key=lambda row: str(row[0])):
        if len(object_state) >= 64:
            break
        object_state[str(key)] = _stable_resource_value(item)
    resource = canonical_sha256(
        {
            "schema": CALLABLE_FINGERPRINT_SCHEMA,
            "kind": "implicit_local_binding",
            "implementation_fingerprint": implementation,
            "defaults": defaults,
            "kwdefaults": kwdefaults,
            "closure": closure_values,
            "object_state": object_state,
        }
    )
    return implementation, resource, False


def _storage_identity(value: Any) -> tuple[Any, ...]:
    try:
        storage = value.untyped_storage()
        pointer = int(storage.data_ptr())
        if pointer:
            return (
                str(getattr(value, "device", "")),
                pointer,
                int(storage.nbytes()),
            )
        return ("storage_object", id(storage))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ("object", id(value))


def _assert_disjoint_reload(
    original: Any,
    reloaded: Any,
    *,
    original_label: str = "accepted in-memory child",
) -> None:
    if reloaded is original:
        raise DynamicFoldArtifactError(
            f"Fresh loader returned the {original_label} object."
        )
    original_modules = getattr(original, "modules", None)
    reloaded_modules = getattr(reloaded, "modules", None)
    if not callable(original_modules) or not callable(reloaded_modules):
        raise DynamicFoldArtifactError(
            "Fresh reload models must expose modules(), parameters(), and buffers()."
        )
    original_module_objects = {id(module) for module in original_modules()}
    if any(id(module) in original_module_objects for module in reloaded_modules()):
        raise DynamicFoldArtifactError(
            f"Fresh loader returned a model sharing modules with the {original_label}."
        )
    original_all_tensors: list[Any] = []
    reloaded_all_tensors: list[Any] = []
    for kind in ("parameters", "buffers"):
        original_values = getattr(original, kind, None)
        reloaded_values = getattr(reloaded, kind, None)
        if not callable(original_values) or not callable(reloaded_values):
            raise DynamicFoldArtifactError(
                "Fresh reload models must expose parameters() and buffers()."
            )
        original_objects = {id(value) for value in original_values()}
        original_kind_values = tuple(original_values())
        reloaded_kind_values = tuple(reloaded_values())
        original_all_tensors.extend(original_kind_values)
        reloaded_all_tensors.extend(reloaded_kind_values)
        original_storages = {_storage_identity(value) for value in original_kind_values}
        for value in reloaded_kind_values:
            if id(value) in original_objects or (
                _storage_identity(value) in original_storages
            ):
                raise DynamicFoldArtifactError(
                    "Fresh loader returned a model sharing "
                    f"{kind[:-1]} objects or storage with the {original_label}."
                )
    original_tensor_objects = {id(value) for value in original_all_tensors}
    original_tensor_storages = {
        _storage_identity(value) for value in original_all_tensors
    }
    if any(
        id(value) in original_tensor_objects
        or _storage_identity(value) in original_tensor_storages
        for value in reloaded_all_tensors
    ):
        raise DynamicFoldArtifactError(
            "Fresh loader returned a model sharing cross-kind tensor objects or "
            f"storage with the {original_label}."
        )


_STATE_INVENTORY_COUNT_FIELDS = (
    "serialized_state_elements",
    "serialized_state_bytes",
    "persistent_buffer_elements",
    "persistent_buffer_bytes",
    "runtime_state_elements",
    "runtime_state_bytes",
    "runtime_buffer_elements",
    "runtime_buffer_bytes",
    "nonpersistent_buffer_elements",
    "nonpersistent_buffer_bytes",
    "runtime_named_buffer_count",
    "nonpersistent_named_buffer_count",
    "runtime_unique_buffer_view_count",
)
_STATE_INVENTORY_FINGERPRINT_FIELD = "runtime_tensor_state_fingerprint"


def _validated_state_inventory(
    value: Any,
    *,
    name: str,
) -> dict[str, int | str]:
    if not isinstance(value, Mapping):
        raise DynamicFoldArtifactError(f"{name} must be an inference-state mapping.")
    expected_fields = {
        *_STATE_INVENTORY_COUNT_FIELDS,
        _STATE_INVENTORY_FINGERPRINT_FIELD,
    }
    if set(value) != expected_fields:
        raise DynamicFoldArtifactError(
            f"{name} has missing or unexpected inference-state fields."
        )
    normalized: dict[str, int | str] = {}
    for field_name in _STATE_INVENTORY_COUNT_FIELDS:
        observed = value[field_name]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise DynamicFoldArtifactError(
                f"{name} field {field_name!r} must be a non-negative integer."
            )
        normalized[field_name] = observed
    fingerprint = value[_STATE_INVENTORY_FINGERPRINT_FIELD]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or fingerprint == "0" * 64
    ):
        raise DynamicFoldArtifactError(
            f"{name} has an invalid runtime tensor-state fingerprint."
        )
    normalized[_STATE_INVENTORY_FINGERPRINT_FIELD] = fingerprint
    if (
        normalized["runtime_state_elements"] < normalized["serialized_state_elements"]
        or normalized["runtime_state_bytes"] < normalized["serialized_state_bytes"]
        or normalized["runtime_buffer_elements"]
        < normalized["persistent_buffer_elements"]
        or normalized["runtime_buffer_bytes"] < normalized["persistent_buffer_bytes"]
        or normalized["runtime_buffer_elements"]
        < normalized["nonpersistent_buffer_elements"]
        or normalized["runtime_buffer_bytes"] < normalized["nonpersistent_buffer_bytes"]
        or normalized["runtime_named_buffer_count"]
        < normalized["runtime_unique_buffer_view_count"]
        or normalized["runtime_named_buffer_count"]
        < normalized["nonpersistent_named_buffer_count"]
    ):
        raise DynamicFoldArtifactError(
            f"{name} contains internally inconsistent runtime-state totals."
        )
    return normalized


def _revalidate_dynamic_target_runtime(
    model: Any,
    *,
    dynamic_target: DynamicTargetSynthesisResult | None,
    parent_models: Mapping[str, Any],
    name: str,
) -> None:
    """Reauthenticate complete inference state against the synthesized target."""

    if dynamic_target is None:
        return
    policy = dynamic_target.structural_buffer_policy
    expected_audit = dynamic_target.structural_buffer_audit
    if policy is None or expected_audit is None:
        raise DynamicFoldArtifactError(
            f"{name} lacks compiler-authenticated structural-buffer policy evidence."
        )
    try:
        validate_registered_inference_state(
            model,
            forbidden_models=parent_models,
        )
        observed_audit = validate_structural_runtime_buffers(
            model,
            policy,
            expected_adapter_id=dynamic_target.adapter_id,
            expected_adapter_version=dynamic_target.adapter_version,
        )
    except Exception as exc:
        raise DynamicFoldArtifactError(
            f"{name} failed complete inference-state reauthentication."
        ) from exc
    if observed_audit != expected_audit:
        raise DynamicFoldArtifactError(
            f"{name} structural-buffer audit differs from the exact synthesized target."
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _reference_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token
    )


def _forbidden_content_label(value: str) -> bool:
    tokens = _reference_tokens(value)
    if tokens.intersection(_FORBIDDEN_CONTENT_TOKENS):
        return True
    normalized = "_".join(re.findall(r"[a-z0-9]+", value.casefold()))
    return (
        "cache" in tokens
        and normalized not in _ALLOWED_MODEL_CACHE_KEYS
        and bool(tokens.intersection(_CACHE_REFERENCE_QUALIFIERS))
    )


def _forbidden_path_reference(value: str) -> bool:
    if not (
        "/" in value
        or "\\" in value
        or value.casefold().endswith(
            (
                ".bin",
                ".ckpt",
                ".json",
                ".pt",
                ".pth",
                ".safetensors",
            )
        )
    ):
        return False
    tokens = _reference_tokens(value)
    return bool(tokens.intersection(_FORBIDDEN_CONTENT_TOKENS) or "cache" in tokens)


def _json_forbidden_reference(
    value: Any,
    *,
    path: str,
) -> str | None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            normalized_key = "_".join(re.findall(r"[a-z0-9]+", key.casefold()))
            if normalized_key in _JSON_LITERAL_CONTAINERS:
                continue
            if _forbidden_content_label(key) and (
                not isinstance(child, (int, float, bool, type(None)))
                or len(_reference_tokens(key)) > 1
            ):
                return child_path
            nested = _json_forbidden_reference(child, path=child_path)
            if nested is not None:
                return nested
        return None
    if isinstance(value, list):
        for index, child in enumerate(value):
            nested = _json_forbidden_reference(
                child,
                path=f"{path}[{index}]",
            )
            if nested is not None:
                return nested
        return None
    if isinstance(value, str) and _forbidden_path_reference(value):
        return path
    return None


def _pickle_forbidden_reference(payload: bytes, *, path: str) -> str | None:
    try:
        operations = pickletools.genops(payload)
        for opcode, argument, position in operations:
            if opcode.name not in {
                "BINSTRING",
                "BINUNICODE",
                "BINUNICODE8",
                "SHORT_BINSTRING",
                "SHORT_BINUNICODE",
                "STRING",
                "UNICODE",
            }:
                continue
            if isinstance(argument, bytes):
                try:
                    text = argument.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            elif isinstance(argument, str):
                text = argument
            else:
                continue
            if _forbidden_content_label(text) or _forbidden_path_reference(text):
                return f"{path}@pickle[{position}]"
    except (EOFError, ValueError):
        return None
    return None


def _safetensors_forbidden_reference(path: Path, *, label: str) -> str | None:
    if path.stat().st_size < 8:
        return None
    with path.open("rb") as handle:
        (header_size,) = struct.unpack("<Q", handle.read(8))
        if header_size > _MAX_STRUCTURED_INVENTORY_BYTES:
            raise DynamicFoldArtifactError(
                f"Safetensors inventory is unreasonably large: {label}."
            )
        payload = handle.read(header_size)
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, Mapping):
        return None
    for tensor_name in header:
        if tensor_name == "__metadata__":
            continue
        if _forbidden_content_label(str(tensor_name)):
            return f"{label}::{tensor_name}"
    metadata = header.get("__metadata__")
    if isinstance(metadata, Mapping):
        return _json_forbidden_reference(metadata, path=f"{label}::__metadata__")
    return None


def _artifact_forbidden_content_reference(
    path: Path,
    *,
    relative: str,
) -> str | None:
    suffix = path.suffix.casefold()
    if suffix == ".json" and path.stat().st_size <= _MAX_STRUCTURED_INVENTORY_BYTES:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if payload is not None:
            reference = _json_forbidden_reference(payload, path=relative)
            if reference is not None:
                return reference
    if suffix == ".safetensors":
        reference = _safetensors_forbidden_reference(path, label=relative)
        if reference is not None:
            return reference
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                member = info.filename
                lowered = member.casefold()
                if any(part in lowered for part in _FORBIDDEN_NAME_PARTS):
                    return f"{relative}::{member}"
                if (
                    member.casefold().endswith((".pkl", "/data.pkl"))
                    and info.file_size <= _MAX_STRUCTURED_INVENTORY_BYTES
                ):
                    reference = _pickle_forbidden_reference(
                        archive.read(info),
                        path=f"{relative}::{member}",
                    )
                    if reference is not None:
                        return reference
    return None


def _artifact_tree(root: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DynamicFoldArtifactError(
                "Published child artifacts cannot contain symbolic links."
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if any(part in lowered for part in _FORBIDDEN_NAME_PARTS):
            raise DynamicFoldArtifactError(
                f"Forbidden sidecar-like artifact name: {relative}."
            )
        forbidden_reference = _artifact_forbidden_content_reference(
            path,
            relative=relative,
        )
        if forbidden_reference is not None:
            raise DynamicFoldArtifactError(
                "Forbidden resume/cache/coordinator/checkpoint/pending/admission "
                f"reference in folded child artifact: {forbidden_reference}."
            )
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise DynamicFoldArtifactError("Model saver produced an empty artifact.")
    return tuple(rows)


def _canonical_standalone_file_roles(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise DynamicFoldArtifactError(
            "Standalone folded-child file-role inventory must be nonempty."
        )
    roles: dict[str, str] = {}
    casefolded: set[str] = set()
    for raw_name, raw_role in value.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or raw_name in {".", ".."}
            or "/" in raw_name
            or "\\" in raw_name
            or Path(raw_name).name != raw_name
            or not isinstance(raw_role, str)
            or raw_role not in _STANDALONE_FILE_ROLES
        ):
            raise DynamicFoldArtifactError(
                "Standalone folded-child file-role inventory is not canonical."
            )
        folded = raw_name.casefold()
        if folded in casefolded:
            raise DynamicFoldArtifactError(
                "Standalone folded-child inventory has a case-colliding path."
            )
        casefolded.add(folded)
        if (
            (
                raw_role == "artifact_metadata"
                and raw_name != _LFM2_STANDALONE_ARTIFACT_METADATA
            )
            or (raw_role == "model_config" and raw_name != "config.json")
            or (
                raw_role == "generation_config" and raw_name != "generation_config.json"
            )
            or (
                raw_role == "model_tensor_index"
                and raw_name != "model.safetensors.index.json"
            )
            or (
                raw_role == "model_tensor"
                and raw_name != "model.safetensors"
                and _SAFETENSORS_SHARD.fullmatch(raw_name) is None
            )
            or (
                raw_role == "tokenizer_resource"
                and raw_name not in _CANONICAL_TOKENIZER_FILES
            )
        ):
            raise DynamicFoldArtifactError(
                f"Standalone folded-child role {raw_role!r} cannot authorize "
                f"file {raw_name!r}."
            )
        roles[raw_name] = raw_role
    return dict(sorted(roles.items()))


def _declared_lfm2_file(
    root: Path,
    row: Any,
    *,
    size_field: str,
    role: str,
) -> tuple[str, str]:
    if not isinstance(row, Mapping) or set(row) != {"path", size_field, "sha256"}:
        raise DynamicFoldArtifactError(
            "Standalone LFM2 artifact contains a malformed declared-file row."
        )
    name = row.get("path")
    size = row.get(size_field)
    digest = row.get("sha256")
    if not isinstance(name, str):
        raise DynamicFoldArtifactError(
            "Standalone LFM2 artifact declared-file path is malformed."
        )
    role_row = _canonical_standalone_file_roles({name: role})
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise DynamicFoldArtifactError(
            "Standalone LFM2 artifact declared-file identity is malformed."
        )
    path = root / name
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != size
        or _file_sha256(path) != digest
    ):
        raise DynamicFoldArtifactError(
            f"Standalone LFM2 declared file identity changed: {name}."
        )
    return name, role_row[name]


def canonical_standalone_fold_artifact_inventory(
    root: str | os.PathLike[str],
) -> tuple[dict[str, str], ...]:
    """Derive the exact typed file allowlist from sealed LFM2 checkpoint metadata."""

    raw_root = Path(root).expanduser().absolute()
    if raw_root.is_symlink():
        raise DynamicFoldArtifactError(
            "Standalone folded-child directory cannot be a symbolic link."
        )
    try:
        resolved = raw_root.resolve(strict=True)
    except OSError as exc:
        raise DynamicFoldArtifactError(
            "Standalone folded-child directory is missing."
        ) from exc
    if not resolved.is_dir():
        raise DynamicFoldArtifactError(
            "Standalone folded-child artifact root must be a directory."
        )
    metadata_path = resolved / _LFM2_STANDALONE_ARTIFACT_METADATA
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise DynamicFoldArtifactError(
            "Standalone LFM2 artifact metadata is missing or unsafe."
        )
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicFoldArtifactError(
            "Standalone LFM2 artifact metadata is invalid."
        ) from exc
    if not isinstance(document, Mapping):
        raise DynamicFoldArtifactError(
            "Standalone LFM2 artifact metadata must be one JSON object."
        )
    body = dict(document)
    supplied = body.pop("artifact_sha256", None)
    if body.get(
        "schema"
    ) != _LFM2_STANDALONE_ARTIFACT_SCHEMA or supplied != canonical_sha256(body):
        raise DynamicFoldArtifactError(
            "Standalone LFM2 artifact metadata seal is invalid."
        )
    dense = body.get("dense_safetensor_files")
    tokenizer = body.get("tokenizer_package_files")
    if not isinstance(dense, list) or not isinstance(tokenizer, list):
        raise DynamicFoldArtifactError(
            "Standalone LFM2 metadata omitted its typed checkpoint inventory."
        )
    roles = {
        _LFM2_STANDALONE_ARTIFACT_METADATA: "artifact_metadata",
    }
    for name, role in (
        ("config.json", "model_config"),
        ("generation_config.json", "generation_config"),
        ("model.safetensors.index.json", "model_tensor_index"),
    ):
        path = resolved / name
        if path.is_symlink():
            raise DynamicFoldArtifactError(
                f"Standalone checkpoint file cannot be a symbolic link: {name}."
            )
        if path.is_file():
            roles[name] = role
    for row in dense:
        name, role = _declared_lfm2_file(
            resolved,
            row,
            size_field="size_bytes",
            role="model_tensor",
        )
        if name in roles:
            raise DynamicFoldArtifactError(
                f"Standalone LFM2 inventory repeats file {name!r}."
            )
        roles[name] = role
    for row in tokenizer:
        name, role = _declared_lfm2_file(
            resolved,
            row,
            size_field="bytes",
            role="tokenizer_resource",
        )
        if name in roles:
            raise DynamicFoldArtifactError(
                f"Standalone LFM2 inventory repeats file {name!r}."
            )
        roles[name] = role
    canonical = _canonical_standalone_file_roles(roles)
    return tuple(
        {"path": name, "artifact_role": role} for name, role in canonical.items()
    )


def scan_standalone_fold_artifact_tree(
    root: str | os.PathLike[str],
    *,
    expected_file_roles: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate and hash one exact, typed standalone folded-child directory.

    The no-argument LFM2 path derives its allowlist from the checkpoint's sealed
    weight/tokenizer inventories.  Other family-specific publishers must supply
    an equally explicit canonical file-role map.  Arbitrary extra files are
    rejected even when their names contain no denylisted words.
    """

    raw_root = Path(root).expanduser().absolute()
    if raw_root.is_symlink():
        raise DynamicFoldArtifactError(
            "Standalone folded-child directory cannot be a symbolic link."
        )
    try:
        resolved = raw_root.resolve(strict=True)
    except OSError as exc:
        raise DynamicFoldArtifactError(
            "Standalone folded-child directory is missing."
        ) from exc
    if not resolved.is_dir():
        raise DynamicFoldArtifactError(
            "Standalone folded-child artifact root must be a directory."
        )
    if expected_file_roles is None:
        inventory = canonical_standalone_fold_artifact_inventory(resolved)
        roles = {row["path"]: row["artifact_role"] for row in inventory}
    else:
        roles = _canonical_standalone_file_roles(expected_file_roles)
    try:
        entries = tuple(sorted(resolved.iterdir(), key=lambda path: path.name))
    except OSError as exc:
        raise DynamicFoldArtifactError(
            "Cannot enumerate standalone folded-child directory."
        ) from exc
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise DynamicFoldArtifactError(
                f"Standalone folded-child entry is not an ordinary file: {path.name}."
            )
        details = path.stat()
        if getattr(details, "st_nlink", 1) != 1:
            raise DynamicFoldArtifactError(
                f"Standalone folded-child file cannot be a hard link: {path.name}."
            )
        if path.suffix.casefold() == ".json" and (
            details.st_size > _MAX_STRUCTURED_INVENTORY_BYTES
        ):
            raise DynamicFoldArtifactError(
                f"Standalone folded-child JSON exceeds the scan limit: {path.name}."
            )
    observed = {path.name for path in entries}
    expected = set(roles)
    unexpected = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unexpected or missing:
        raise DynamicFoldArtifactError(
            "Standalone folded-child files differ from the canonical typed "
            f"inventory; unexpected={unexpected}, missing={missing}."
        )
    tree = _artifact_tree(resolved)
    if {row["path"] for row in tree} != expected:
        raise DynamicFoldArtifactError(
            "Standalone folded-child tree differs from its canonical inventory."
        )
    return tree


def _decision(value: FoldDecision | bool) -> FoldDecision:
    if isinstance(value, FoldDecision):
        return value
    if isinstance(value, bool):
        return FoldDecision(value, "")
    raise DynamicFoldArtifactError(
        "Fresh-reload evaluator must return FoldDecision or bool."
    )


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _semantic_fingerprint(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> str:
    value = dict(payload)
    fingerprint = _require_digest(value.pop("fingerprint", None), f"{name} fingerprint")
    if canonical_sha256(value) != fingerprint:
        raise DynamicFoldArtifactError(
            f"{name} semantic fingerprint does not match its contents."
        )
    return fingerprint


def _parent_graphs_document(graphs: Any) -> dict[str, Any]:
    values = tuple(graphs)
    if not values or any(not isinstance(value, ModelCircuitGraph) for value in values):
        raise DynamicFoldArtifactError(
            "Compilation provenance requires concrete parent circuit graphs."
        )
    ordered = sorted(values, key=lambda value: value.model_name)
    if len({value.model_name for value in ordered}) != len(ordered):
        raise DynamicFoldArtifactError(
            "Compilation provenance parent graph names must be unique."
        )
    body = {
        "schema": PROVENANCE_SCHEMA,
        "kind": "parent_graphs",
        "graphs": [value.as_dict() for value in ordered],
    }
    return {**body, "fingerprint": canonical_sha256(body)}


def _no_dynamic_target_document() -> dict[str, Any]:
    """Canonical marker for the generic caller-materialized-child API."""

    return {
        "schema": PROVENANCE_SCHEMA,
        "kind": "no_dynamic_target",
        "dynamic_target_provenance_present": False,
        "reason": "generic-caller-materialized-child",
    }


def _target_provenance_document(
    dynamic_target: DynamicTargetSynthesisResult | None,
) -> tuple[dict[str, Any], bool]:
    if dynamic_target is None:
        return _no_dynamic_target_document(), False
    if not isinstance(dynamic_target, DynamicTargetSynthesisResult):
        raise DynamicFoldArtifactError(
            "dynamic_target must be DynamicTargetSynthesisResult or None."
        )
    try:
        verify_compiler_authenticated_target(dynamic_target)
        document = dynamic_target.as_dict()
    except Exception as exc:
        raise DynamicFoldArtifactError(
            "Dynamic target lacks compiler-authenticated synthesis provenance."
        ) from exc
    if not isinstance(document, Mapping):
        raise DynamicFoldArtifactError(
            "Dynamic target synthesis result did not produce one JSON object."
        )
    return dict(document), True


def _validate_target_provenance_semantics(
    target: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    child_graph: Mapping[str, Any],
    parent_graphs: Mapping[str, Any],
    expected_target_fingerprint: str,
    dynamic_target_provenance_present: bool,
) -> None:
    """Authenticate target synthesis against the compiled graph/plan/parents."""

    target_document = dict(target)
    if canonical_sha256(target_document) != expected_target_fingerprint:
        raise DynamicFoldArtifactError(
            "Target provenance semantic fingerprint differs from its certificate."
        )
    if not dynamic_target_provenance_present:
        if target_document != _no_dynamic_target_document():
            raise DynamicFoldArtifactError(
                "Generic publication has an invalid no-dynamic-target marker."
            )
        return

    if (
        target_document.get("schema") != TARGET_SYNTHESIS_SCHEMA
        or target_document.get("kind") != "materialized_target"
    ):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance schema or kind is invalid."
        )
    request = target_document.get("request")
    target_graph = target_document.get("graph")
    profile = target_document.get("resource_profile")
    blank = target_document.get("blank_provenance")
    declared_parents = target_document.get("parent_graph_fingerprints")
    if not all(
        isinstance(value, Mapping)
        for value in (request, target_graph, profile, blank, declared_parents)
    ):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance lacks request, graph, resource, blank, "
            "or parent bindings."
        )
    request = dict(request)
    target_graph = dict(target_graph)
    profile = dict(profile)
    blank = dict(blank)
    declared_parents = dict(declared_parents)
    if target_graph != dict(child_graph):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance is bound to another child graph."
        )
    child_fingerprint = _semantic_fingerprint(
        child_graph,
        name="Target-bound child graph",
    )
    graph_values = parent_graphs.get("graphs")
    if not isinstance(graph_values, list):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance cannot resolve parent graphs."
        )
    observed_parents = {
        str(value.get("model_name")): _semantic_fingerprint(
            value,
            name="Target-bound parent graph",
        )
        for value in graph_values
        if isinstance(value, Mapping)
    }
    if len(observed_parents) != len(graph_values) or declared_parents != (
        observed_parents
    ):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance is bound to different parent graphs."
        )
    exact_parameters = plan.get("exact_target_unique_parameters")
    child_unique_parameters = child_graph.get("unique_parameter_count")
    child_parameter_bytes = child_graph.get("unique_parameter_bytes")
    request_modalities = request.get("target_modalities")
    profile_modalities = profile.get("modalities")
    interface_values = child_graph.get("interfaces")
    if not isinstance(interface_values, list):
        raise DynamicFoldArtifactError(
            "Dynamic target child graph has invalid modality interfaces."
        )
    graph_modalities = sorted(
        {
            value.get("modality")
            for value in interface_values
            if isinstance(value, Mapping)
        }
    )
    if (
        isinstance(exact_parameters, bool)
        or not isinstance(exact_parameters, int)
        or exact_parameters <= 0
        or exact_parameters != child_unique_parameters
        or request.get("exact_unique_parameters") != exact_parameters
        or profile.get("unique_parameters") != exact_parameters
        or profile.get("parameter_bytes") != child_parameter_bytes
        or request.get("child_name") != child_graph.get("model_name")
        or profile.get("family") != child_graph.get("family")
        or sorted(request_modalities or []) != graph_modalities
        or sorted(profile_modalities or []) != graph_modalities
        or request.get("tokenizer_fingerprint")
        != child_graph.get("tokenizer_fingerprint")
        or target_document.get("blank_state_fingerprint")
        != plan.get("child_state_fingerprint")
        or blank.get("blank_state_fingerprint") != plan.get("child_state_fingerprint")
        or blank.get("unique_parameter_element_count") != exact_parameters
    ):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance differs from the exact child budget, "
            "graph, modalities, tokenizer, or pre-fold blank."
        )
    requested_family = request.get("target_family")
    if requested_family not in {"any", "auto", child_graph.get("family")}:
        raise DynamicFoldArtifactError(
            "Dynamic target provenance changes the requested target family."
        )
    maximum_bytes = request.get("maximum_parameter_bytes")
    if maximum_bytes is not None and (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or child_parameter_bytes > maximum_bytes
    ):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance exceeds its requested byte budget."
        )
    if (
        target_document.get("compiler_created_blank_verified") is not True
        or target_document.get("topology_fingerprint_compiler_derived") is not True
        or target_document.get("exact_budget_match") is not True
        or blank.get("compiler_created_blank") is not True
        or blank.get("parent_child_alias_free") is not True
        or blank.get("all_parameters_zero") is not True
        or blank.get("parameter_gradients_absent") is not True
    ):
        raise DynamicFoldArtifactError(
            "Dynamic target provenance lacks compiler-owned blank/topology proof."
        )
    if canonical_sha256(blank) != target_document.get("blank_provenance_digest"):
        raise DynamicFoldArtifactError(
            "Dynamic target blank-provenance digest is invalid."
        )
    request_fingerprint = canonical_sha256(request)
    expected_topology = canonical_sha256(
        {
            "schema": TARGET_SYNTHESIS_SCHEMA,
            "kind": "compiler_authenticated_target_topology",
            "request_fingerprint": request_fingerprint,
            "adapter_id": target_document.get("adapter_id"),
            "adapter_version": target_document.get("adapter_version"),
            "graph_fingerprint": child_fingerprint,
            "parameter_layout_fingerprint": blank.get("parameter_layout_fingerprint"),
            "buffer_layout_fingerprint": blank.get("buffer_layout_fingerprint"),
            "buffer_state_fingerprint": blank.get("buffer_state_fingerprint_after"),
        }
    )
    if target_document.get("topology_fingerprint") != expected_topology:
        raise DynamicFoldArtifactError(
            "Dynamic target topology seal differs from compiler-observed evidence."
        )
    if plan.get("parent_graph_fingerprints") != sorted(observed_parents.values()):
        raise DynamicFoldArtifactError(
            "Dynamic target parent bindings differ from the fold plan."
        )


def _validate_provenance_semantics(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    expected_plan_fingerprint: str,
    expected_run_fingerprint: str,
    expected_child_graph_fingerprint: str,
    expected_parent_graphs_fingerprint: str,
    expected_target_fingerprint: str,
    dynamic_target_provenance_present: bool,
) -> None:
    plan = dict(documents["plan"])
    run = dict(documents["run"])
    child_graph = dict(documents["child_graph"])
    parent_graphs = dict(documents["parent_graphs"])
    target = dict(documents["target"])
    if _semantic_fingerprint(plan, name="Plan provenance") != (
        expected_plan_fingerprint
    ):
        raise DynamicFoldArtifactError(
            "Plan provenance differs from its certificate fingerprint."
        )
    if _semantic_fingerprint(run, name="Run provenance") != expected_run_fingerprint:
        raise DynamicFoldArtifactError(
            "Run provenance differs from its certificate fingerprint."
        )
    if (
        _semantic_fingerprint(
            child_graph,
            name="Child-graph provenance",
        )
        != expected_child_graph_fingerprint
    ):
        raise DynamicFoldArtifactError(
            "Child-graph provenance differs from its certificate fingerprint."
        )
    if (
        parent_graphs.get("schema") != PROVENANCE_SCHEMA
        or parent_graphs.get("kind") != "parent_graphs"
    ):
        raise DynamicFoldArtifactError(
            "Parent-graph provenance schema or kind is invalid."
        )
    if (
        _semantic_fingerprint(
            parent_graphs,
            name="Parent-graphs provenance",
        )
        != expected_parent_graphs_fingerprint
    ):
        raise DynamicFoldArtifactError(
            "Parent-graph provenance differs from its certificate fingerprint."
        )
    graph_values = parent_graphs.get("graphs")
    if (
        not isinstance(graph_values, list)
        or not graph_values
        or any(not isinstance(value, Mapping) for value in graph_values)
    ):
        raise DynamicFoldArtifactError(
            "Parent-graph provenance must contain full graph documents."
        )
    parent_names = [value.get("model_name") for value in graph_values]
    parent_fingerprints = [
        _semantic_fingerprint(value, name="Parent graph") for value in graph_values
    ]
    if (
        len(set(parent_names)) != len(parent_names)
        or len(set(parent_fingerprints)) != len(parent_fingerprints)
        or parent_names != sorted(parent_names)
    ):
        raise DynamicFoldArtifactError(
            "Parent-graph provenance identities are duplicate or noncanonical."
        )
    if plan.get("child_graph_fingerprint") != expected_child_graph_fingerprint:
        raise DynamicFoldArtifactError(
            "Plan provenance is bound to another child graph."
        )
    if run.get("plan_fingerprint") != expected_plan_fingerprint:
        raise DynamicFoldArtifactError("Run provenance is bound to another plan.")
    declared_parents = plan.get("parent_graph_fingerprints")
    if declared_parents != sorted(parent_fingerprints):
        raise DynamicFoldArtifactError(
            "Plan provenance is bound to different parent graphs."
        )
    _validate_target_provenance_semantics(
        target,
        plan=plan,
        child_graph=child_graph,
        parent_graphs=parent_graphs,
        expected_target_fingerprint=expected_target_fingerprint,
        dynamic_target_provenance_present=dynamic_target_provenance_present,
    )


def _provenance_tree(root: Path) -> tuple[dict[str, Any], ...]:
    expected = set(_PROVENANCE_FILENAMES.values())
    if root.is_symlink() or not root.is_dir():
        raise DynamicFoldArtifactError(
            "Artifact provenance directory is missing or unsafe."
        )
    entries = tuple(root.iterdir())
    if {value.name for value in entries} != expected or any(
        value.is_symlink() or not value.is_file() for value in entries
    ):
        raise DynamicFoldArtifactError(
            "Artifact provenance directory has missing or unexpected entries."
        )
    return tuple(
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(entries, key=lambda value: value.name)
    )


def _write_provenance_documents(
    root: Path,
    compilation: Any,
    *,
    dynamic_target: DynamicTargetSynthesisResult | None = None,
) -> dict[str, Any]:
    provenance = root / _PROVENANCE_DIRECTORY
    provenance.mkdir()
    try:
        target_document, target_present = _target_provenance_document(dynamic_target)
        documents = {
            "plan": compilation.plan.as_dict(),
            "run": compilation.run.as_dict(),
            "child_graph": compilation.child_graph.as_dict(),
            "parent_graphs": _parent_graphs_document(compilation.parent_graphs),
            "target": target_document,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise DynamicFoldArtifactError(
            "Compilation cannot produce complete provenance documents."
        ) from exc
    parent_fingerprint = _require_digest(
        documents["parent_graphs"].get("fingerprint"),
        "parent-graphs provenance fingerprint",
    )
    target_fingerprint = canonical_sha256(documents["target"])
    _validate_provenance_semantics(
        documents,
        expected_plan_fingerprint=compilation.plan.fingerprint,
        expected_run_fingerprint=compilation.run.fingerprint,
        expected_child_graph_fingerprint=compilation.child_graph.fingerprint,
        expected_parent_graphs_fingerprint=parent_fingerprint,
        expected_target_fingerprint=target_fingerprint,
        dynamic_target_provenance_present=target_present,
    )
    for name, filename in _PROVENANCE_FILENAMES.items():
        _write_canonical_json(provenance / filename, documents[name])
    tree = _provenance_tree(provenance)
    hashes = {row["path"]: row["sha256"] for row in tree}
    return {
        "plan_sha256": hashes[_PROVENANCE_FILENAMES["plan"]],
        "run_sha256": hashes[_PROVENANCE_FILENAMES["run"]],
        "child_graph_sha256": hashes[_PROVENANCE_FILENAMES["child_graph"]],
        "parent_graphs_sha256": hashes[_PROVENANCE_FILENAMES["parent_graphs"]],
        "parent_graphs_fingerprint": parent_fingerprint,
        "target_sha256": hashes[_PROVENANCE_FILENAMES["target"]],
        "target_fingerprint": target_fingerprint,
        "dynamic_target_provenance_present": target_present,
        "tree_sha256": canonical_sha256(tree),
    }


def _read_canonical_json_document(path: Path, *, name: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicFoldArtifactError(f"{name} is unreadable.") from exc
    if not isinstance(payload, Mapping):
        raise DynamicFoldArtifactError(f"{name} must be one JSON object.")
    if raw != canonical_json_bytes(payload) + b"\n":
        raise DynamicFoldArtifactError(f"{name} is not canonically encoded.")
    return dict(payload)


def _authenticate_provenance_bundle(
    root: Path,
    *,
    plan_sha256: str,
    run_sha256: str,
    child_graph_sha256: str,
    parent_graphs_sha256: str,
    target_sha256: str,
    tree_sha256: str,
    plan_fingerprint: str,
    run_fingerprint: str,
    child_graph_fingerprint: str,
    parent_graphs_fingerprint: str,
    target_fingerprint: str,
    dynamic_target_provenance_present: bool,
) -> None:
    provenance = root / _PROVENANCE_DIRECTORY
    tree = _provenance_tree(provenance)
    if canonical_sha256(tree) != tree_sha256:
        raise DynamicFoldArtifactError(
            "Artifact provenance tree differs from its certificate."
        )
    hashes = {row["path"]: row["sha256"] for row in tree}
    expected_hashes = {
        _PROVENANCE_FILENAMES["plan"]: plan_sha256,
        _PROVENANCE_FILENAMES["run"]: run_sha256,
        _PROVENANCE_FILENAMES["child_graph"]: child_graph_sha256,
        _PROVENANCE_FILENAMES["parent_graphs"]: parent_graphs_sha256,
        _PROVENANCE_FILENAMES["target"]: target_sha256,
    }
    if hashes != expected_hashes:
        raise DynamicFoldArtifactError(
            "Artifact provenance document digests differ from its certificate."
        )
    documents = {
        name: _read_canonical_json_document(
            provenance / filename,
            name=f"{name.replace('_', '-')} provenance",
        )
        for name, filename in _PROVENANCE_FILENAMES.items()
    }
    _validate_provenance_semantics(
        documents,
        expected_plan_fingerprint=plan_fingerprint,
        expected_run_fingerprint=run_fingerprint,
        expected_child_graph_fingerprint=child_graph_fingerprint,
        expected_parent_graphs_fingerprint=parent_graphs_fingerprint,
        expected_target_fingerprint=target_fingerprint,
        dynamic_target_provenance_present=dynamic_target_provenance_present,
    )


def _authenticate_provenance_documents(
    root: Path,
    certificate: DynamicFoldArtifactCertificate,
) -> None:
    _authenticate_provenance_bundle(
        root,
        plan_sha256=certificate.provenance_plan_sha256,
        run_sha256=certificate.provenance_run_sha256,
        child_graph_sha256=certificate.provenance_child_graph_sha256,
        parent_graphs_sha256=certificate.provenance_parent_graphs_sha256,
        target_sha256=certificate.provenance_target_sha256,
        tree_sha256=certificate.provenance_tree_sha256,
        plan_fingerprint=certificate.plan_fingerprint,
        run_fingerprint=certificate.run_fingerprint,
        child_graph_fingerprint=certificate.child_graph_fingerprint,
        parent_graphs_fingerprint=(certificate.provenance_parent_graphs_fingerprint),
        target_fingerprint=certificate.provenance_target_fingerprint,
        dynamic_target_provenance_present=(
            certificate.dynamic_target_provenance_present
        ),
    )


def _safe_remove_stage(stage: Path, parent: Path) -> None:
    resolved_stage = stage.resolve()
    resolved_parent = parent.resolve()
    if resolved_stage.parent != resolved_parent or not resolved_stage.name.startswith(
        ".dynamic-fold-stage-"
    ):
        raise DynamicFoldArtifactError(
            "Refusing to remove an unverified publication stage."
        )
    shutil.rmtree(resolved_stage)


def _safe_remove_unpublished_destination(destination: Path, parent: Path) -> None:
    """Remove only the exact destination created by this publication call."""

    absolute_destination = destination.expanduser().absolute()
    absolute_parent = parent.expanduser().resolve(strict=True)
    if absolute_destination.parent.resolve(strict=True) != absolute_parent:
        raise DynamicFoldArtifactError(
            "Refusing to remove a failed publication outside its exact parent."
        )
    if absolute_destination.is_symlink():
        absolute_destination.unlink()
        return
    resolved_destination = absolute_destination.resolve(strict=True)
    if resolved_destination.parent != absolute_parent:
        raise DynamicFoldArtifactError(
            "Refusing to remove a failed publication through a redirected path."
        )
    shutil.rmtree(resolved_destination)


def authenticate_dynamic_fold_artifact_tree(
    certificate: DynamicFoldArtifactCertificate,
) -> tuple[dict[str, Any], ...]:
    """Re-hash the complete standalone child tree against its certificate."""

    if not isinstance(certificate, DynamicFoldArtifactCertificate):
        raise DynamicFoldArtifactError(
            "Artifact-tree authentication requires a certificate."
        )
    raw_root = Path(certificate.path).expanduser().absolute()
    raw_child = raw_root / "child"
    if raw_root.is_symlink() or raw_child.is_symlink():
        raise DynamicFoldArtifactError(
            "Certified standalone child path cannot use symbolic links."
        )
    try:
        root = raw_root.resolve(strict=True)
        child = raw_child.resolve(strict=True)
    except OSError as exc:
        raise DynamicFoldArtifactError(
            "Certified standalone child path is missing."
        ) from exc
    if not root.is_dir() or not child.is_dir() or child.parent != root:
        raise DynamicFoldArtifactError("Certified standalone child path is unsafe.")
    tree = _artifact_tree(child)
    if (
        canonical_sha256(tree) != certificate.artifact_tree_sha256
        or len(tree) != certificate.artifact_file_count
        or sum(int(row["size_bytes"]) for row in tree)
        != certificate.artifact_total_bytes
    ):
        raise DynamicFoldArtifactError(
            "Standalone child artifact tree differs from its certificate."
        )
    return tree


def load_dynamic_fold_artifact_certificate(
    destination: str | os.PathLike[str],
) -> DynamicFoldArtifactCertificate:
    """Load, self-authenticate, and re-hash one on-disk certificate tree."""

    root = Path(destination).expanduser().absolute()
    certificate_path = root / "certificate.json"
    if (
        root.is_symlink()
        or certificate_path.is_symlink()
        or not certificate_path.is_file()
    ):
        raise DynamicFoldArtifactError(
            "Artifact certificate path is missing or unsafe."
        )
    try:
        payload = json.loads(certificate_path.read_bytes().decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicFoldArtifactError(
            "Artifact certificate JSON is unreadable."
        ) from exc
    if not isinstance(payload, Mapping):
        raise DynamicFoldArtifactError("Artifact certificate must be one JSON object.")
    certificate_schema = payload.get("schema")
    if certificate_schema not in {LEGACY_SCHEMA, SCHEMA}:
        raise DynamicFoldArtifactError("Artifact certificate schema is unsupported.")
    decision_payload = payload.get("fresh_reload_decision")
    if not isinstance(decision_payload, Mapping):
        raise DynamicFoldArtifactError(
            "Artifact certificate lacks a fresh-reload decision."
        )
    metrics = decision_payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise DynamicFoldArtifactError(
            "Artifact fresh-reload metrics must be an object."
        )
    try:
        decision = FoldDecision(
            accepted=decision_payload["accepted"],
            reason=decision_payload["reason"],
            metrics=tuple(metrics.items()),
        )
        certificate = DynamicFoldArtifactCertificate(
            path=payload["path"],
            plan_fingerprint=payload["plan_fingerprint"],
            run_fingerprint=payload["run_fingerprint"],
            child_graph_fingerprint=payload["child_graph_fingerprint"],
            child_state_fingerprint=payload["child_state_fingerprint"],
            exact_unique_parameters=payload["exact_unique_parameters"],
            artifact_tree_sha256=payload["artifact_tree_sha256"],
            artifact_file_count=payload["artifact_file_count"],
            artifact_total_bytes=payload["artifact_total_bytes"],
            fresh_reload_decision=decision,
            saver_id=payload["saver_id"],
            loader_id=payload["loader_id"],
            compilation_optimizer_constructions=payload[
                "compilation_optimizer_constructions"
            ],
            compilation_optimizer_steps=payload["compilation_optimizer_steps"],
            compilation_backward_calls=payload["compilation_backward_calls"],
            compilation_autograd_grad_calls=payload["compilation_autograd_grad_calls"],
            optimizer_constructions=payload["optimizer_constructions"],
            optimizer_steps=payload["optimizer_steps"],
            backward_calls=payload["backward_calls"],
            autograd_grad_calls=payload["autograd_grad_calls"],
            certificate_sha256=payload["certificate_sha256"],
            evidence_fingerprint=payload["evidence_fingerprint"],
            behavior_gate_id=payload["behavior_gate_id"],
            behavior_gate_version=payload["behavior_gate_version"],
            behavior_baseline_fingerprint=payload["behavior_baseline_fingerprint"],
            behavior_gate_manifest_fingerprint=payload[
                "behavior_gate_manifest_fingerprint"
            ],
            behavior_gate_manifest=payload["behavior_gate_manifest"],
            fresh_reload_state_inventory=_validated_state_inventory(
                payload["fresh_reload_state_inventory"],
                name="artifact fresh-reload state inventory",
            ),
            fresh_reload_module_forward_calls=payload[
                "fresh_reload_module_forward_calls"
            ],
            fresh_reload_root_forward_calls=payload["fresh_reload_root_forward_calls"],
            fresh_reload_inference_entrypoint_calls=(
                payload.get(
                    "fresh_reload_inference_entrypoint_calls",
                    (
                        payload["fresh_reload_root_forward_calls"]
                        if certificate_schema == LEGACY_SCHEMA
                        else None
                    ),
                )
            ),
            preparation_guard_applied=payload["preparation_guard_applied"],
            preparation_optimizer_constructions=payload[
                "preparation_optimizer_constructions"
            ],
            preparation_optimizer_steps=payload["preparation_optimizer_steps"],
            preparation_backward_calls=payload["preparation_backward_calls"],
            preparation_autograd_grad_calls=payload["preparation_autograd_grad_calls"],
            preparation_parent_module_forward_calls=payload[
                "preparation_parent_module_forward_calls"
            ],
            preparation_parent_root_forward_calls=payload[
                "preparation_parent_root_forward_calls"
            ],
            publication_guard_covers_save_load=payload[
                "publication_guard_covers_save_load"
            ],
            publication_optimizer_constructions=payload[
                "publication_optimizer_constructions"
            ],
            publication_optimizer_steps=payload["publication_optimizer_steps"],
            publication_backward_calls=payload["publication_backward_calls"],
            publication_autograd_grad_calls=payload["publication_autograd_grad_calls"],
            publication_parent_module_forward_calls=payload[
                "publication_parent_module_forward_calls"
            ],
            publication_parent_root_forward_calls=payload[
                "publication_parent_root_forward_calls"
            ],
            provenance_plan_sha256=payload["provenance_plan_sha256"],
            provenance_run_sha256=payload["provenance_run_sha256"],
            provenance_child_graph_sha256=payload["provenance_child_graph_sha256"],
            provenance_parent_graphs_sha256=payload["provenance_parent_graphs_sha256"],
            provenance_parent_graphs_fingerprint=payload[
                "provenance_parent_graphs_fingerprint"
            ],
            provenance_target_sha256=payload["provenance_target_sha256"],
            provenance_target_fingerprint=payload["provenance_target_fingerprint"],
            dynamic_target_provenance_present=payload[
                "dynamic_target_provenance_present"
            ],
            provenance_tree_sha256=payload["provenance_tree_sha256"],
            loader_path_challenge_performed=payload["loader_path_challenge_performed"],
            loader_path_challenge_exception_type=payload[
                "loader_path_challenge_exception_type"
            ],
            saver_implementation_fingerprint=payload[
                "saver_implementation_fingerprint"
            ],
            saver_resource_fingerprint=payload["saver_resource_fingerprint"],
            saver_resource_fingerprint_declared=payload[
                "saver_resource_fingerprint_declared"
            ],
            loader_implementation_fingerprint=payload[
                "loader_implementation_fingerprint"
            ],
            loader_resource_fingerprint=payload["loader_resource_fingerprint"],
            loader_resource_fingerprint_declared=payload[
                "loader_resource_fingerprint_declared"
            ],
            certificate_schema=certificate_schema,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise DynamicFoldArtifactError(
            "Artifact certificate fields are incomplete or invalid."
        ) from exc
    if (
        certificate.certificate_sha256 != canonical_sha256(certificate.body())
        or dict(payload) != certificate.as_dict()
    ):
        raise DynamicFoldArtifactError(
            "Artifact certificate does not authenticate its exact body."
        )
    for field_name in (
        "provenance_plan_sha256",
        "provenance_run_sha256",
        "provenance_child_graph_sha256",
        "provenance_parent_graphs_sha256",
        "provenance_parent_graphs_fingerprint",
        "provenance_target_sha256",
        "provenance_target_fingerprint",
        "provenance_tree_sha256",
        "saver_implementation_fingerprint",
        "saver_resource_fingerprint",
        "loader_implementation_fingerprint",
        "loader_resource_fingerprint",
        "behavior_gate_manifest_fingerprint",
    ):
        value = _require_digest(getattr(certificate, field_name), field_name)
        if value == "0" * 64:
            raise DynamicFoldArtifactError(
                f"Artifact certificate has an unset {field_name}."
            )
    if not isinstance(certificate.dynamic_target_provenance_present, bool):
        raise DynamicFoldArtifactError(
            "Artifact certificate dynamic-target marker must be boolean."
        )
    manifest = (
        {}
        if certificate.behavior_gate_manifest is None
        else dict(certificate.behavior_gate_manifest)
    )
    try:
        inference_execution_contract_from_manifest(manifest)
    except DynamicBehaviorExecutionError as exc:
        raise DynamicFoldArtifactError(
            "Artifact behavior-gate inference execution contract is invalid."
        ) from exc
    manifest_body = dict(manifest)
    manifest_seal = manifest_body.pop("manifest_fingerprint", None)
    if (
        manifest_seal != certificate.behavior_gate_manifest_fingerprint
        or canonical_sha256(manifest_body)
        != certificate.behavior_gate_manifest_fingerprint
        or manifest_body.get("gate_id") != certificate.behavior_gate_id
        or manifest_body.get("gate_version") != certificate.behavior_gate_version
        or manifest_body.get("baseline_fingerprint")
        != certificate.behavior_baseline_fingerprint
    ):
        raise DynamicFoldArtifactError(
            "Artifact behavior-gate manifest is invalid or mismatched."
        )
    if (
        isinstance(certificate.fresh_reload_module_forward_calls, bool)
        or not isinstance(certificate.fresh_reload_module_forward_calls, int)
        or certificate.fresh_reload_module_forward_calls <= 0
        or isinstance(certificate.fresh_reload_root_forward_calls, bool)
        or not isinstance(certificate.fresh_reload_root_forward_calls, int)
        or certificate.fresh_reload_root_forward_calls <= 0
        or certificate.fresh_reload_root_forward_calls
        > certificate.fresh_reload_module_forward_calls
        or isinstance(
            certificate.fresh_reload_inference_entrypoint_calls,
            bool,
        )
        or not isinstance(
            certificate.fresh_reload_inference_entrypoint_calls,
            int,
        )
        or certificate.fresh_reload_inference_entrypoint_calls <= 0
    ):
        raise DynamicFoldArtifactError(
            "Artifact certificate does not prove fresh-reloaded model execution."
        )
    for field_name in (
        "preparation_parent_module_forward_calls",
        "preparation_parent_root_forward_calls",
        "publication_parent_module_forward_calls",
        "publication_parent_root_forward_calls",
    ):
        value = getattr(certificate, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise DynamicFoldArtifactError(
                "Artifact certificate reports live-parent execution during "
                "preparation or publication."
            )
    if (
        certificate.loader_path_challenge_performed is not True
        or not isinstance(certificate.loader_path_challenge_exception_type, str)
        or not certificate.loader_path_challenge_exception_type.strip()
        or not isinstance(certificate.saver_resource_fingerprint_declared, bool)
        or not isinstance(certificate.loader_resource_fingerprint_declared, bool)
    ):
        raise DynamicFoldArtifactError(
            "Artifact certificate lacks a valid loader challenge or callable binding."
        )
    try:
        certified_root = Path(certificate.path).expanduser().resolve(strict=True)
        observed_root = root.resolve(strict=True)
    except OSError as exc:
        raise DynamicFoldArtifactError("Artifact certificate root is missing.") from exc
    if certified_root != observed_root:
        raise DynamicFoldArtifactError(
            "Artifact certificate is bound to another destination."
        )
    _authenticate_provenance_documents(observed_root, certificate)
    authenticate_dynamic_fold_artifact_tree(certificate)
    return certificate


def remove_authenticated_dynamic_fold_artifact(
    destination: str | os.PathLike[str],
    *,
    expected_parent: str | os.PathLike[str] | None = None,
    expected_name: str | None = None,
) -> DynamicFoldArtifactCertificate:
    """Remove only an exactly authenticated, caller-owned publication."""

    root = Path(destination).expanduser().absolute()
    certificate = load_dynamic_fold_artifact_certificate(root)
    resolved = root.resolve(strict=True)
    if expected_parent is not None:
        parent = Path(expected_parent).expanduser().resolve(strict=True)
        if resolved.parent != parent:
            raise DynamicFoldArtifactError(
                "Refusing to remove an artifact outside the expected parent."
            )
    if expected_name is not None and resolved.name != expected_name:
        raise DynamicFoldArtifactError(
            "Refusing to remove an artifact with an unexpected name."
        )
    shutil.rmtree(resolved)
    return certificate


def publish_dynamic_fold_artifact(
    *,
    compilation: Any,
    child_model: Any,
    destination: str | os.PathLike[str],
    model_saver: Callable[[Any, Path], None],
    model_loader: Callable[[Path], Any],
    fresh_reload_evaluator: Callable[[Any], FoldDecision | bool],
    saver_id: str,
    loader_id: str,
    behavior_gate_id: str | None = None,
    behavior_gate_version: int | None = None,
    behavior_baseline_fingerprint: str | None = None,
    behavior_gate_manifest_fingerprint: str | None = None,
    behavior_gate_manifest: Mapping[str, Any] | None = None,
    dynamic_target: DynamicTargetSynthesisResult | None = None,
    expected_state_inventory: Mapping[str, int | str] | None = None,
    parent_models: Mapping[str, Any] | None = None,
    preparation_guard_applied: bool = False,
    preparation_guard_counts: Mapping[str, int] | None = None,
    preparation_parent_forward_counts: Mapping[str, int] | None = None,
    inspectors: InspectorRegistry | None = None,
    tokenizer_fingerprint: str | None = None,
    snapshot_config: DynamicFoldSnapshotConfig | None = None,
) -> DynamicFoldArtifactCertificate:
    """Atomically save, freshly load, inspect, and behavior-gate one child."""

    if not bool(getattr(compilation, "complete", False)):
        raise DynamicFoldArtifactError(
            "Only a behavior-accepted complete compilation can be published."
        )
    run = getattr(compilation, "run", None)
    compilation_guard_counts = {
        "compilation_optimizer_constructions": getattr(
            run, "optimizer_constructions", None
        ),
        "compilation_optimizer_steps": getattr(run, "optimizer_steps", None),
        "compilation_backward_calls": getattr(run, "backward_calls", None),
        "compilation_autograd_grad_calls": getattr(run, "autograd_grad_calls", None),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value != 0
        for value in compilation_guard_counts.values()
    ):
        raise DynamicFoldArtifactError(
            "Compilation observed a forbidden training API attempt; "
            "publication is not allowed."
        )
    if not callable(model_saver) or not callable(model_loader):
        raise DynamicFoldArtifactError("model_saver and model_loader must be callable.")
    if not callable(fresh_reload_evaluator):
        raise DynamicFoldArtifactError("fresh_reload_evaluator must be callable.")
    saver_id = _text(saver_id, "saver_id")
    loader_id = _text(loader_id, "loader_id")
    (
        saver_implementation_fingerprint,
        saver_resource_fingerprint,
        saver_resource_fingerprint_declared,
    ) = _callable_fingerprints(model_saver, name="model_saver")
    (
        loader_implementation_fingerprint,
        loader_resource_fingerprint,
        loader_resource_fingerprint_declared,
    ) = _callable_fingerprints(model_loader, name="model_loader")
    if not isinstance(preparation_guard_applied, bool):
        raise DynamicFoldArtifactError("preparation_guard_applied must be boolean.")
    preparation_counts = _guard_counts(
        preparation_guard_counts,
        name="preparation guard",
    )
    if any(preparation_counts.values()):
        raise DynamicFoldArtifactError(
            "Target/stack preparation observed a forbidden training API "
            "attempt; publication is not allowed."
        )
    preparation_parent_counts = _forward_counts(
        preparation_parent_forward_counts,
        name="preparation parent forward observation",
    )
    if any(preparation_parent_counts.values()):
        raise DynamicFoldArtifactError(
            "Target/stack preparation executed a live parent model; cached "
            "evidence is required."
        )
    normalized_gate_id = (
        "unregistered-behavior-gate"
        if behavior_gate_id is None
        else _text(behavior_gate_id, "behavior_gate_id")
    )
    normalized_gate_version = (
        0 if behavior_gate_version is None else behavior_gate_version
    )
    if (
        isinstance(normalized_gate_version, bool)
        or not isinstance(normalized_gate_version, int)
        or normalized_gate_version < 0
    ):
        raise DynamicFoldArtifactError(
            "behavior_gate_version must be a non-negative integer."
        )
    normalized_baseline = (
        "0" * 64
        if behavior_baseline_fingerprint is None
        else behavior_baseline_fingerprint
    )
    if (
        not isinstance(normalized_baseline, str)
        or len(normalized_baseline) != 64
        or any(character not in "0123456789abcdef" for character in normalized_baseline)
    ):
        raise DynamicFoldArtifactError(
            "behavior_baseline_fingerprint must be a lowercase SHA-256 digest."
        )
    if behavior_gate_manifest is None:
        default_execution_contract = (
            BehaviorInferenceExecutionContract.top_level_forward()
        )
        manifest_body = {
            "schema": "neural-forge-unregistered-behavior-gate-manifest-v1",
            "gate_id": normalized_gate_id,
            "gate_version": normalized_gate_version,
            "baseline_fingerprint": normalized_baseline,
            "inference_execution": default_execution_contract.as_dict(),
        }
        normalized_manifest_fingerprint = canonical_sha256(manifest_body)
        normalized_manifest = {
            **manifest_body,
            "manifest_fingerprint": normalized_manifest_fingerprint,
        }
        if (
            behavior_gate_manifest_fingerprint is not None
            and behavior_gate_manifest_fingerprint != normalized_manifest_fingerprint
        ):
            raise DynamicFoldArtifactError(
                "Behavior-gate manifest fingerprint has no matching manifest."
            )
    else:
        normalized_manifest = dict(behavior_gate_manifest)
        manifest_body = dict(normalized_manifest)
        embedded_manifest_fingerprint = manifest_body.pop(
            "manifest_fingerprint",
            None,
        )
        normalized_manifest_fingerprint = (
            embedded_manifest_fingerprint
            if behavior_gate_manifest_fingerprint is None
            else behavior_gate_manifest_fingerprint
        )
        if (
            not isinstance(normalized_manifest_fingerprint, str)
            or canonical_sha256(manifest_body) != normalized_manifest_fingerprint
            or embedded_manifest_fingerprint != normalized_manifest_fingerprint
            or manifest_body.get("gate_id") != normalized_gate_id
            or manifest_body.get("gate_version") != normalized_gate_version
            or manifest_body.get("baseline_fingerprint") != normalized_baseline
        ):
            raise DynamicFoldArtifactError(
                "Behavior-gate manifest is not canonically sealed or differs "
                "from the declared gate."
            )
    try:
        inference_execution_contract = inference_execution_contract_from_manifest(
            normalized_manifest
        )
    except DynamicBehaviorExecutionError as exc:
        raise DynamicFoldArtifactError(
            "Behavior-gate manifest has an invalid inference execution contract."
        ) from exc
    destination_path = Path(destination).expanduser().absolute()
    if destination_path.exists():
        raise DynamicFoldArtifactError(
            f"Publication destination already exists: {destination_path}."
        )
    parent = destination_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".dynamic-fold-stage-", dir=parent)).absolute()
    child_directory = stage / "child"
    child_directory.mkdir()
    selected_inspectors = inspectors or default_inspector_registry()
    in_memory_state = fingerprint_model_state(child_model)
    in_memory_runtime = fingerprint_model_runtime_state(child_model)
    in_memory_inventory = _validated_state_inventory(
        serialized_inference_state_inventory(child_model),
        name="accepted in-memory child state inventory",
    )
    if expected_state_inventory is not None:
        expected_inventory = _validated_state_inventory(
            expected_state_inventory,
            name="compiler target state inventory",
        )
        if in_memory_inventory != expected_inventory:
            raise DynamicFoldArtifactError(
                "Accepted in-memory child state inventory differs from the "
                "compiler target resource profile."
            )
    normalized_parents: tuple[tuple[str, Any], ...] = ()
    if parent_models is not None:
        if not isinstance(parent_models, Mapping):
            raise DynamicFoldArtifactError("parent_models must be a mapping or None.")
        normalized_parents = tuple(
            sorted(
                (
                    (_text(name, "parent model name"), model)
                    for name, model in parent_models.items()
                ),
                key=lambda item: item[0],
            )
        )
        if len({name for name, _model in normalized_parents}) != len(
            normalized_parents
        ):
            raise DynamicFoldArtifactError("parent_models names must be unique.")
    normalized_parent_mapping = dict(normalized_parents)
    _revalidate_dynamic_target_runtime(
        child_model,
        dynamic_target=dynamic_target,
        parent_models=normalized_parent_mapping,
        name="Accepted in-memory child",
    )
    selected_snapshot = snapshot_config or DynamicFoldSnapshotConfig.memory()
    if not isinstance(selected_snapshot, DynamicFoldSnapshotConfig):
        raise DynamicFoldArtifactError(
            "snapshot_config must be DynamicFoldSnapshotConfig or None."
        )
    in_memory_snapshot = _WholeModelSnapshot.capture(
        child_model,
        selected_snapshot,
    )
    publication_guard = _FoldOnlyTorchGuard()
    publication_guard_active = False
    reload_guard: _FoldOnlyTorchGuard | None = None
    final_reload_guard: _FoldOnlyTorchGuard | None = None
    parent_observer_stack = ExitStack()
    parent_observers: list[dict[str, int]] = []
    parent_observer_active = False
    destination_created = False
    try:
        provenance = _write_provenance_documents(
            stage,
            compilation,
            dynamic_target=dynamic_target,
        )
        publication_guard.__enter__()
        publication_guard_active = True
        parent_observer_stack.__enter__()
        parent_observer_active = True
        for _parent_name, parent_model in normalized_parents:
            parent_observers.append(
                parent_observer_stack.enter_context(
                    observe_model_execution(parent_model)
                )
            )
        model_saver(child_model, child_directory)
        if (
            fingerprint_model_state(child_model) != in_memory_state
            or fingerprint_model_runtime_state(child_model) != in_memory_runtime
        ):
            raise DynamicFoldArtifactError(
                "Model saver mutated the accepted in-memory child."
            )
        before_reload_tree = _artifact_tree(child_directory)
        before_reload_tree_sha256 = canonical_sha256(before_reload_tree)
        before_reload_total_bytes = sum(
            int(row["size_bytes"]) for row in before_reload_tree
        )
        challenge_directory = stage / ".loader-path-challenge"
        challenge_directory.mkdir()
        challenge_exception_type = ""
        try:
            challenge_result = model_loader(challenge_directory)
        except Exception as challenge_error:
            challenge_exception_type = _qualified_type(challenge_error)
        else:
            del challenge_result
            raise DynamicFoldArtifactError(
                "Model loader failed the empty path-consumption challenge by "
                "returning without an artifact."
            )
        if any(challenge_directory.iterdir()):
            raise DynamicFoldArtifactError(
                "Model loader mutated the empty path-consumption challenge directory."
            )
        challenge_directory.rmdir()
        if (
            fingerprint_model_state(child_model) != in_memory_state
            or fingerprint_model_runtime_state(child_model) != in_memory_runtime
        ):
            raise DynamicFoldArtifactError(
                "Model loader path-consumption challenge mutated the accepted child."
            )
        reloaded = model_loader(child_directory)
        _assert_disjoint_reload(child_model, reloaded)
        for parent_name, parent_model in normalized_parents:
            _assert_disjoint_reload(
                parent_model,
                reloaded,
                original_label=f"live parent {parent_name!r}",
            )
        _revalidate_dynamic_target_runtime(
            reloaded,
            dynamic_target=dynamic_target,
            parent_models=normalized_parent_mapping,
            name="Stage-path fresh reload",
        )
        reloaded_graph: ModelCircuitGraph = selected_inspectors.inspect(
            reloaded,
            model_name=compilation.child_graph.model_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if reloaded_graph.fingerprint != compilation.child_graph.fingerprint:
            raise DynamicFoldArtifactError(
                "Fresh reload topology differs from the compiled child."
            )
        if (
            reloaded_graph.unique_parameter_count
            != compilation.plan.exact_target_unique_parameters
        ):
            raise DynamicFoldArtifactError(
                "Fresh reload does not match the exact requested budget."
            )
        reloaded_state = fingerprint_model_state(reloaded)
        if reloaded_state != in_memory_state:
            raise DynamicFoldArtifactError(
                "Fresh reload state differs from the accepted in-memory child."
            )
        reloaded_inventory = _validated_state_inventory(
            serialized_inference_state_inventory(reloaded),
            name="fresh-reloaded child state inventory",
        )
        if reloaded_inventory != in_memory_inventory:
            raise DynamicFoldArtifactError(
                "Fresh reload inference-state inventory differs from the "
                "accepted compiler target."
            )
        if any(parameter.grad is not None for parameter in reloaded.parameters()):
            raise DynamicFoldArtifactError(
                "Freshly reloaded child contains parameter gradients."
            )
        reload_runtime = fingerprint_model_runtime_state(reloaded)
        reload_parameters = tuple(
            (parameter, bool(parameter.requires_grad))
            for parameter in reloaded.parameters()
        )
        for parameter, _requires_grad in reload_parameters:
            parameter.requires_grad_(False)
        try:
            with _FoldOnlyTorchGuard() as reload_guard:
                with observe_model_execution(
                    reloaded,
                    contract=inference_execution_contract,
                ) as execution_counts:
                    reload_decision = _decision(fresh_reload_evaluator(reloaded))
        finally:
            for parameter, requires_grad in reload_parameters:
                parameter.grad = None
                parameter.requires_grad_(requires_grad)
        reload_module_forward_calls = int(execution_counts["module_forward_calls"])
        reload_root_forward_calls = int(execution_counts["root_forward_calls"])
        reload_inference_entrypoint_calls = int(
            execution_counts["inference_entrypoint_calls"]
        )
        if reload_decision.accepted and not execution_contract_satisfied(
            execution_counts
        ):
            if (
                inference_execution_contract
                == BehaviorInferenceExecutionContract.top_level_forward()
            ):
                raise DynamicFoldArtifactError(
                    "Fresh-reload evaluator accepted without executing the "
                    "published child root."
                )
            raise DynamicFoldArtifactError(
                "Fresh-reload evaluator accepted without executing the published "
                "child through its declared inference entrypoint and module root."
            )
        if not reload_decision.accepted:
            raise DynamicFoldArtifactError(
                "Freshly reloaded child failed its behavioral gate: "
                + reload_decision.reason
            )
        if (
            fingerprint_model_state(reloaded) != reloaded_state
            or fingerprint_model_runtime_state(reloaded) != reload_runtime
        ):
            raise DynamicFoldArtifactError(
                "Fresh-reload evaluator mutated the published child."
            )
        post_evaluation_graph = selected_inspectors.inspect(
            reloaded,
            model_name=compilation.child_graph.model_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if post_evaluation_graph.fingerprint != reloaded_graph.fingerprint:
            raise DynamicFoldArtifactError(
                "Fresh-reload evaluator changed the child topology."
            )
        after_reload_tree = _artifact_tree(child_directory)
        if after_reload_tree != before_reload_tree:
            raise DynamicFoldArtifactError(
                "Fresh reload mutated the serialized artifact tree."
            )
        _authenticate_provenance_bundle(
            stage,
            plan_sha256=provenance["plan_sha256"],
            run_sha256=provenance["run_sha256"],
            child_graph_sha256=provenance["child_graph_sha256"],
            parent_graphs_sha256=provenance["parent_graphs_sha256"],
            target_sha256=provenance["target_sha256"],
            tree_sha256=provenance["tree_sha256"],
            plan_fingerprint=compilation.plan.fingerprint,
            run_fingerprint=compilation.run.fingerprint,
            child_graph_fingerprint=compilation.child_graph.fingerprint,
            parent_graphs_fingerprint=provenance["parent_graphs_fingerprint"],
            target_fingerprint=provenance["target_fingerprint"],
            dynamic_target_provenance_present=(
                provenance["dynamic_target_provenance_present"]
            ),
        )
        stage.rename(destination_path)
        destination_created = True
        final_child_directory = destination_path / "child"
        final_reloaded = model_loader(final_child_directory)
        _assert_disjoint_reload(child_model, final_reloaded)
        _assert_disjoint_reload(
            reloaded,
            final_reloaded,
            original_label="stage-path fresh reload",
        )
        for parent_name, parent_model in normalized_parents:
            _assert_disjoint_reload(
                parent_model,
                final_reloaded,
                original_label=f"live parent {parent_name!r}",
            )
        _revalidate_dynamic_target_runtime(
            final_reloaded,
            dynamic_target=dynamic_target,
            parent_models=normalized_parent_mapping,
            name="Final-destination fresh reload",
        )
        final_reloaded_graph: ModelCircuitGraph = selected_inspectors.inspect(
            final_reloaded,
            model_name=compilation.child_graph.model_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if final_reloaded_graph.fingerprint != compilation.child_graph.fingerprint:
            raise DynamicFoldArtifactError(
                "Final-destination reload topology differs from the compiled child."
            )
        if (
            final_reloaded_graph.unique_parameter_count
            != compilation.plan.exact_target_unique_parameters
        ):
            raise DynamicFoldArtifactError(
                "Final-destination reload misses the exact requested budget."
            )
        final_reloaded_state = fingerprint_model_state(final_reloaded)
        if final_reloaded_state != in_memory_state:
            raise DynamicFoldArtifactError(
                "Final-destination reload state differs from the accepted child."
            )
        final_reloaded_inventory = _validated_state_inventory(
            serialized_inference_state_inventory(final_reloaded),
            name="final-destination reloaded child state inventory",
        )
        if final_reloaded_inventory != in_memory_inventory:
            raise DynamicFoldArtifactError(
                "Final-destination reload inventory differs from the compiler target."
            )
        if any(parameter.grad is not None for parameter in final_reloaded.parameters()):
            raise DynamicFoldArtifactError(
                "Final-destination reloaded child contains parameter gradients."
            )
        final_reload_runtime = fingerprint_model_runtime_state(final_reloaded)
        final_reload_parameters = tuple(
            (parameter, bool(parameter.requires_grad))
            for parameter in final_reloaded.parameters()
        )
        for parameter, _requires_grad in final_reload_parameters:
            parameter.requires_grad_(False)
        try:
            with _FoldOnlyTorchGuard() as final_reload_guard:
                with observe_model_execution(
                    final_reloaded,
                    contract=inference_execution_contract,
                ) as final_execution_counts:
                    final_reload_decision = _decision(
                        fresh_reload_evaluator(final_reloaded)
                    )
        finally:
            for parameter, requires_grad in final_reload_parameters:
                parameter.grad = None
                parameter.requires_grad_(requires_grad)
        final_reload_module_forward_calls = int(
            final_execution_counts["module_forward_calls"]
        )
        final_reload_root_forward_calls = int(
            final_execution_counts["root_forward_calls"]
        )
        final_reload_inference_entrypoint_calls = int(
            final_execution_counts["inference_entrypoint_calls"]
        )
        if final_reload_decision.accepted and not execution_contract_satisfied(
            final_execution_counts
        ):
            if (
                inference_execution_contract
                == BehaviorInferenceExecutionContract.top_level_forward()
            ):
                raise DynamicFoldArtifactError(
                    "Final-destination evaluator accepted without executing the "
                    "child root."
                )
            raise DynamicFoldArtifactError(
                "Final-destination evaluator accepted without executing the child "
                "through its declared inference entrypoint and module root."
            )
        if not final_reload_decision.accepted:
            raise DynamicFoldArtifactError(
                "Final-destination child failed its behavioral gate: "
                + final_reload_decision.reason
            )
        if (
            fingerprint_model_state(final_reloaded) != final_reloaded_state
            or fingerprint_model_runtime_state(final_reloaded) != final_reload_runtime
        ):
            raise DynamicFoldArtifactError(
                "Final-destination evaluator mutated the retained child."
            )
        final_post_evaluation_graph = selected_inspectors.inspect(
            final_reloaded,
            model_name=compilation.child_graph.model_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if final_post_evaluation_graph.fingerprint != final_reloaded_graph.fingerprint:
            raise DynamicFoldArtifactError(
                "Final-destination evaluator changed the child topology."
            )
        final_tree = _artifact_tree(final_child_directory)
        if final_tree != before_reload_tree:
            raise DynamicFoldArtifactError(
                "Final-destination reload mutated the serialized child tree."
            )
        _authenticate_provenance_bundle(
            destination_path,
            plan_sha256=provenance["plan_sha256"],
            run_sha256=provenance["run_sha256"],
            child_graph_sha256=provenance["child_graph_sha256"],
            parent_graphs_sha256=provenance["parent_graphs_sha256"],
            target_sha256=provenance["target_sha256"],
            tree_sha256=provenance["tree_sha256"],
            plan_fingerprint=compilation.plan.fingerprint,
            run_fingerprint=compilation.run.fingerprint,
            child_graph_fingerprint=compilation.child_graph.fingerprint,
            parent_graphs_fingerprint=provenance["parent_graphs_fingerprint"],
            target_fingerprint=provenance["target_fingerprint"],
            dynamic_target_provenance_present=(
                provenance["dynamic_target_provenance_present"]
            ),
        )
        parent_observer_stack.__exit__(None, None, None)
        parent_observer_active = False
        publication_parent_counts = {
            "module_forward_calls": sum(
                int(value["module_forward_calls"]) for value in parent_observers
            ),
            "root_forward_calls": sum(
                int(value["root_forward_calls"]) for value in parent_observers
            ),
        }
        if any(publication_parent_counts.values()):
            raise DynamicFoldArtifactError(
                "Artifact publication executed a live parent model; cached "
                "evidence is required."
            )
        publication_guard.__exit__(None, None, None)
        publication_guard_active = False
        if reload_guard is None or final_reload_guard is None:
            raise DynamicFoldArtifactError(
                "Stage or final-destination reload training guard did not execute."
            )
        publication_counts = {
            "optimizer_constructions": (publication_guard.optimizer_constructions),
            "optimizer_steps": publication_guard.optimizer_steps,
            "backward_calls": publication_guard.backward_calls,
            "autograd_grad_calls": publication_guard.autograd_grad_calls,
        }
        if any(publication_counts.values()) or any(
            value != 0
            for value in (
                reload_guard.optimizer_constructions,
                reload_guard.optimizer_steps,
                reload_guard.backward_calls,
                reload_guard.autograd_grad_calls,
                final_reload_guard.optimizer_constructions,
                final_reload_guard.optimizer_steps,
                final_reload_guard.backward_calls,
                final_reload_guard.autograd_grad_calls,
            )
        ):
            raise DynamicFoldArtifactError(
                "Artifact publication observed a forbidden training API "
                "attempt; publication is not allowed."
            )
        reloaded_graph = final_reloaded_graph
        reloaded_state = final_reloaded_state
        reloaded_inventory = final_reloaded_inventory
        reload_decision = final_reload_decision
        reload_module_forward_calls = final_reload_module_forward_calls
        reload_root_forward_calls = final_reload_root_forward_calls
        reload_inference_entrypoint_calls = final_reload_inference_entrypoint_calls
        provisional = DynamicFoldArtifactCertificate(
            path=str(destination_path),
            plan_fingerprint=compilation.plan.fingerprint,
            run_fingerprint=compilation.run.fingerprint,
            child_graph_fingerprint=reloaded_graph.fingerprint,
            child_state_fingerprint=reloaded_state,
            exact_unique_parameters=reloaded_graph.unique_parameter_count,
            artifact_tree_sha256=before_reload_tree_sha256,
            artifact_file_count=len(before_reload_tree),
            artifact_total_bytes=before_reload_total_bytes,
            fresh_reload_decision=reload_decision,
            saver_id=saver_id,
            loader_id=loader_id,
            evidence_fingerprint=compilation.plan.evidence_fingerprint,
            behavior_gate_id=normalized_gate_id,
            behavior_gate_version=normalized_gate_version,
            behavior_baseline_fingerprint=normalized_baseline,
            behavior_gate_manifest_fingerprint=(normalized_manifest_fingerprint),
            behavior_gate_manifest=normalized_manifest,
            fresh_reload_state_inventory=reloaded_inventory,
            fresh_reload_module_forward_calls=reload_module_forward_calls,
            fresh_reload_root_forward_calls=reload_root_forward_calls,
            fresh_reload_inference_entrypoint_calls=(reload_inference_entrypoint_calls),
            preparation_guard_applied=preparation_guard_applied,
            preparation_optimizer_constructions=(
                preparation_counts["optimizer_constructions"]
            ),
            preparation_optimizer_steps=(preparation_counts["optimizer_steps"]),
            preparation_backward_calls=(preparation_counts["backward_calls"]),
            preparation_autograd_grad_calls=(preparation_counts["autograd_grad_calls"]),
            preparation_parent_module_forward_calls=(
                preparation_parent_counts["module_forward_calls"]
            ),
            preparation_parent_root_forward_calls=(
                preparation_parent_counts["root_forward_calls"]
            ),
            publication_guard_covers_save_load=True,
            publication_optimizer_constructions=(
                publication_counts["optimizer_constructions"]
            ),
            publication_optimizer_steps=(publication_counts["optimizer_steps"]),
            publication_backward_calls=(publication_counts["backward_calls"]),
            publication_autograd_grad_calls=(publication_counts["autograd_grad_calls"]),
            publication_parent_module_forward_calls=(
                publication_parent_counts["module_forward_calls"]
            ),
            publication_parent_root_forward_calls=(
                publication_parent_counts["root_forward_calls"]
            ),
            provenance_plan_sha256=provenance["plan_sha256"],
            provenance_run_sha256=provenance["run_sha256"],
            provenance_child_graph_sha256=provenance["child_graph_sha256"],
            provenance_parent_graphs_sha256=(provenance["parent_graphs_sha256"]),
            provenance_parent_graphs_fingerprint=(
                provenance["parent_graphs_fingerprint"]
            ),
            provenance_target_sha256=provenance["target_sha256"],
            provenance_target_fingerprint=provenance["target_fingerprint"],
            dynamic_target_provenance_present=(
                provenance["dynamic_target_provenance_present"]
            ),
            provenance_tree_sha256=provenance["tree_sha256"],
            loader_path_challenge_performed=True,
            loader_path_challenge_exception_type=challenge_exception_type,
            saver_implementation_fingerprint=(saver_implementation_fingerprint),
            saver_resource_fingerprint=saver_resource_fingerprint,
            saver_resource_fingerprint_declared=(saver_resource_fingerprint_declared),
            loader_implementation_fingerprint=(loader_implementation_fingerprint),
            loader_resource_fingerprint=loader_resource_fingerprint,
            loader_resource_fingerprint_declared=(loader_resource_fingerprint_declared),
            **compilation_guard_counts,
            optimizer_constructions=(
                reload_guard.optimizer_constructions
                + final_reload_guard.optimizer_constructions
            ),
            optimizer_steps=reload_guard.optimizer_steps
            + final_reload_guard.optimizer_steps,
            backward_calls=reload_guard.backward_calls
            + final_reload_guard.backward_calls,
            autograd_grad_calls=(
                reload_guard.autograd_grad_calls
                + final_reload_guard.autograd_grad_calls
            ),
            certificate_sha256="0" * 64,
        )
        certificate_hash = canonical_sha256(provisional.body())
        certificate = DynamicFoldArtifactCertificate(
            path=provisional.path,
            plan_fingerprint=provisional.plan_fingerprint,
            run_fingerprint=provisional.run_fingerprint,
            child_graph_fingerprint=provisional.child_graph_fingerprint,
            child_state_fingerprint=provisional.child_state_fingerprint,
            exact_unique_parameters=provisional.exact_unique_parameters,
            artifact_tree_sha256=provisional.artifact_tree_sha256,
            artifact_file_count=provisional.artifact_file_count,
            artifact_total_bytes=provisional.artifact_total_bytes,
            fresh_reload_decision=provisional.fresh_reload_decision,
            saver_id=provisional.saver_id,
            loader_id=provisional.loader_id,
            evidence_fingerprint=provisional.evidence_fingerprint,
            behavior_gate_id=provisional.behavior_gate_id,
            behavior_gate_version=provisional.behavior_gate_version,
            behavior_baseline_fingerprint=(provisional.behavior_baseline_fingerprint),
            behavior_gate_manifest_fingerprint=(
                provisional.behavior_gate_manifest_fingerprint
            ),
            behavior_gate_manifest=provisional.behavior_gate_manifest,
            fresh_reload_state_inventory=(provisional.fresh_reload_state_inventory),
            fresh_reload_module_forward_calls=(
                provisional.fresh_reload_module_forward_calls
            ),
            fresh_reload_root_forward_calls=(
                provisional.fresh_reload_root_forward_calls
            ),
            fresh_reload_inference_entrypoint_calls=(
                provisional.fresh_reload_inference_entrypoint_calls
            ),
            preparation_guard_applied=(provisional.preparation_guard_applied),
            preparation_optimizer_constructions=(
                provisional.preparation_optimizer_constructions
            ),
            preparation_optimizer_steps=(provisional.preparation_optimizer_steps),
            preparation_backward_calls=(provisional.preparation_backward_calls),
            preparation_autograd_grad_calls=(
                provisional.preparation_autograd_grad_calls
            ),
            preparation_parent_module_forward_calls=(
                provisional.preparation_parent_module_forward_calls
            ),
            preparation_parent_root_forward_calls=(
                provisional.preparation_parent_root_forward_calls
            ),
            publication_guard_covers_save_load=(
                provisional.publication_guard_covers_save_load
            ),
            publication_optimizer_constructions=(
                provisional.publication_optimizer_constructions
            ),
            publication_optimizer_steps=(provisional.publication_optimizer_steps),
            publication_backward_calls=(provisional.publication_backward_calls),
            publication_autograd_grad_calls=(
                provisional.publication_autograd_grad_calls
            ),
            publication_parent_module_forward_calls=(
                provisional.publication_parent_module_forward_calls
            ),
            publication_parent_root_forward_calls=(
                provisional.publication_parent_root_forward_calls
            ),
            provenance_plan_sha256=provisional.provenance_plan_sha256,
            provenance_run_sha256=provisional.provenance_run_sha256,
            provenance_child_graph_sha256=(provisional.provenance_child_graph_sha256),
            provenance_parent_graphs_sha256=(
                provisional.provenance_parent_graphs_sha256
            ),
            provenance_parent_graphs_fingerprint=(
                provisional.provenance_parent_graphs_fingerprint
            ),
            provenance_target_sha256=provisional.provenance_target_sha256,
            provenance_target_fingerprint=(provisional.provenance_target_fingerprint),
            dynamic_target_provenance_present=(
                provisional.dynamic_target_provenance_present
            ),
            provenance_tree_sha256=provisional.provenance_tree_sha256,
            loader_path_challenge_performed=(
                provisional.loader_path_challenge_performed
            ),
            loader_path_challenge_exception_type=(
                provisional.loader_path_challenge_exception_type
            ),
            saver_implementation_fingerprint=(
                provisional.saver_implementation_fingerprint
            ),
            saver_resource_fingerprint=(provisional.saver_resource_fingerprint),
            saver_resource_fingerprint_declared=(
                provisional.saver_resource_fingerprint_declared
            ),
            loader_implementation_fingerprint=(
                provisional.loader_implementation_fingerprint
            ),
            loader_resource_fingerprint=(provisional.loader_resource_fingerprint),
            loader_resource_fingerprint_declared=(
                provisional.loader_resource_fingerprint_declared
            ),
            compilation_optimizer_constructions=(
                provisional.compilation_optimizer_constructions
            ),
            compilation_optimizer_steps=(provisional.compilation_optimizer_steps),
            compilation_backward_calls=provisional.compilation_backward_calls,
            compilation_autograd_grad_calls=(
                provisional.compilation_autograd_grad_calls
            ),
            optimizer_constructions=provisional.optimizer_constructions,
            optimizer_steps=provisional.optimizer_steps,
            backward_calls=provisional.backward_calls,
            autograd_grad_calls=provisional.autograd_grad_calls,
            certificate_sha256=certificate_hash,
        )
        _write_canonical_json(
            destination_path / "certificate.json",
            certificate.as_dict(),
        )
        authenticated = load_dynamic_fold_artifact_certificate(destination_path)
        if authenticated != certificate:
            raise DynamicFoldArtifactError(
                "Final-destination certificate reload changed the publication seal."
            )
        return authenticated
    except BaseException:
        if parent_observer_active:
            parent_observer_stack.__exit__(None, None, None)
            parent_observer_active = False
        if publication_guard_active:
            publication_guard.__exit__(None, None, None)
            publication_guard_active = False
        try:
            in_memory_snapshot.restore(child_model)
        finally:
            if stage.exists():
                _safe_remove_stage(stage, parent)
            if destination_created and destination_path.exists():
                _safe_remove_unpublished_destination(destination_path, parent)
        raise
    finally:
        in_memory_snapshot.cleanup()


__all__ = [
    "authenticate_dynamic_fold_artifact_tree",
    "DynamicFoldArtifactCertificate",
    "DynamicFoldArtifactError",
    "LEGACY_SCHEMA",
    "load_dynamic_fold_artifact_certificate",
    "remove_authenticated_dynamic_fold_artifact",
    "canonical_standalone_fold_artifact_inventory",
    "scan_standalone_fold_artifact_tree",
    "SCHEMA",
    "STANDALONE_FOLD_ARTIFACT_INVENTORY_SCHEMA",
    "publish_dynamic_fold_artifact",
]
