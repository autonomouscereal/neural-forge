"""Transactional automatic execution for dynamic fold plans.

The executor owns orchestration, not folding mathematics.  Registered
operators install complete circuit candidates, an independent evaluator makes
the behavioral decision, rejected candidates restore exactly, accepted edits
advance the evidence epoch, and every attempt is appended to the novelty
ledger under the exact current child state.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import math
import textwrap
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import CodeType, MappingProxyType
from typing import Any, Protocol

from .dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
    observe_model_execution,
    resolve_behavior_inference_execution_contract,
)
from .dynamic_circuit_ir import JsonScalar, ModelCircuitGraph, canonical_sha256
from .dynamic_fold_planner import (
    DynamicFoldPlan,
    FoldAction,
    validate_fold_action_evidence_channels,
)
from .dynamic_fold_registry import (
    ClosureLevel,
    ExperimentLedger,
    ExperimentSignature,
    LedgerRecord,
    RecordStatus,
)
from .dynamic_fold_snapshot import (
    CapturedModelState,
    DynamicFoldSnapshotConfig,
    capture_model_state,
)


RUN_SCHEMA = "neural-forge-dynamic-fold-run-v1"

_MODULE_HOOK_ATTRIBUTES = (
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_is_full_backward_hook",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
)
_GLOBAL_HOOK_ATTRIBUTES = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_buffer_registration_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_pre_hooks",
    "_global_forward_pre_hooks_with_kwargs",
    "_global_is_full_backward_hook",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
)


class DynamicFoldExecutionError(RuntimeError):
    """The plan, operator, transaction, evaluator, or rollback is invalid."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicFoldExecutionError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _scalar(value: Any, name: str) -> JsonScalar:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise DynamicFoldExecutionError(f"{name} must be a JSON scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise DynamicFoldExecutionError(f"{name} must be finite.")
    return value


def _text_channel(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicFoldExecutionError(
            "Evidence channel names must be nonempty strings."
        )
    return value.strip()


def _metrics(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    result = sorted((str(key), _scalar(item, f"metric {key}")) for key, item in items)
    if len({key for key, _ in result}) != len(result):
        raise DynamicFoldExecutionError("Metric keys must be unique.")
    return tuple(result)


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _explicit_cache_seal(value: Any) -> tuple[str, str] | None:
    """Return a verified external cache seal when the payload exposes one.

    The verifier remains a registered-provider trust boundary: the executor can
    repeatedly invoke it and bind its result, but cannot independently prove an
    arbitrary third-party verifier's implementation.
    """

    verifier = getattr(value, "verify", None)
    if not callable(verifier):
        return None
    seal_name = None
    seal = None
    for candidate in ("cache_key", "fingerprint", "evidence_fingerprint"):
        candidate_value = getattr(value, candidate, None)
        if candidate_value is not None:
            seal_name = candidate
            seal = candidate_value
            break
    if seal_name is None:
        return None
    try:
        verifier()
    except BaseException as exc:
        raise DynamicFoldExecutionError(
            "Explicit evidence cache verification failed."
        ) from exc
    return seal_name, _digest(seal, f"evidence cache {seal_name}")


def _evidence_payload_identity(
    value: Any,
    *,
    active_ids: set[int],
    trust_modes: set[str],
) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency boundary
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        if value.device.type == "meta" or value.layout != torch.strided:
            raise DynamicFoldExecutionError(
                "Evidence tensors must be materialized and dense."
            )
        tensor = value.detach()
        if tensor.is_conj():
            tensor = tensor.resolve_conj()
        if tensor.is_neg():
            tensor = tensor.resolve_neg()
        tensor = tensor.to(device="cpu").contiguous()
        digest_state = hashlib.sha256()
        _update_digest_with_tensor_bytes(digest_state, tensor, torch)
        digest = digest_state.hexdigest()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
            "stride": [int(size) for size in tensor.stride()],
            "requires_grad": bool(value.requires_grad),
            "sha256": digest,
        }
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DynamicFoldExecutionError("Evidence payload floats must be finite.")
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}

    object_id = id(value)
    if object_id in active_ids:
        raise DynamicFoldExecutionError(
            "Evidence payloads cannot contain reference cycles."
        )
    if is_dataclass(value) and not isinstance(value, type):
        active_ids.add(object_id)
        try:
            return {
                "kind": "dataclass",
                "type": _qualified_type_name(value),
                "fields": [
                    {
                        "name": value_field.name,
                        "value": _evidence_payload_identity(
                            getattr(value, value_field.name),
                            active_ids=active_ids,
                            trust_modes=trust_modes,
                        ),
                    }
                    for value_field in fields(value)
                ],
            }
        finally:
            active_ids.remove(object_id)
    if isinstance(value, Mapping):
        active_ids.add(object_id)
        try:
            items = []
            for key, item in value.items():
                key_identity = _evidence_payload_identity(
                    key,
                    active_ids=active_ids,
                    trust_modes=trust_modes,
                )
                items.append(
                    (
                        canonical_sha256(key_identity),
                        key_identity,
                        _evidence_payload_identity(
                            item,
                            active_ids=active_ids,
                            trust_modes=trust_modes,
                        ),
                    )
                )
            items.sort(key=lambda entry: entry[0])
            key_fingerprints = [entry[0] for entry in items]
            if len(set(key_fingerprints)) != len(key_fingerprints):
                raise DynamicFoldExecutionError(
                    "Evidence mapping keys are not canonically unique."
                )
            return {
                "kind": "mapping",
                "type": _qualified_type_name(value),
                "items": [
                    {"key": key_identity, "value": item_identity}
                    for _fingerprint, key_identity, item_identity in items
                ],
            }
        finally:
            active_ids.remove(object_id)
    if isinstance(value, Sequence):
        active_ids.add(object_id)
        try:
            return {
                "kind": "sequence",
                "type": _qualified_type_name(value),
                "items": [
                    _evidence_payload_identity(
                        item,
                        active_ids=active_ids,
                        trust_modes=trust_modes,
                    )
                    for item in value
                ],
            }
        finally:
            active_ids.remove(object_id)
    explicit_seal = _explicit_cache_seal(value)
    if explicit_seal is not None:
        trust_modes.add("explicit-cache-verifier-trust")
        seal_name, seal = explicit_seal
        return {
            "kind": "verified-cache-seal",
            "type": _qualified_type_name(value),
            "seal_attribute": seal_name,
            "seal": seal,
        }
    raise DynamicFoldExecutionError(
        f"Unsupported fold evidence payload type: {_qualified_type_name(value)}."
    )


def _fold_evidence_payload_details(value: Any) -> tuple[str, str]:
    trust_modes: set[str] = set()
    identity = _evidence_payload_identity(
        value,
        active_ids=set(),
        trust_modes=trust_modes,
    )
    mode = (
        "canonical-content"
        if not trust_modes
        else "canonical-content-with-explicit-cache-verifier-trust"
    )
    return (
        canonical_sha256(
            {
                "schema": "neural-forge-fold-evidence-payload-v1",
                "identity": identity,
            }
        ),
        mode,
    )


def fingerprint_fold_evidence_payload(value: Any) -> str:
    """Hash the concrete evidence payload using compiler-owned canonical rules."""

    return _fold_evidence_payload_details(value)[0]


def _update_digest_with_tensor_bytes(
    digest: Any,
    tensor: Any,
    torch: Any,
) -> None:
    """Stream one contiguous tensor into a digest without a duplicate bytes copy."""

    byte_array = tensor.reshape(-1).view(torch.uint8).numpy()
    try:
        digest.update(memoryview(byte_array))
    except (BufferError, TypeError, ValueError) as exc:
        raise DynamicFoldExecutionError(
            "Tensor fingerprinting requires a C-contiguous byte buffer."
        ) from exc


def fingerprint_model_state(model: Any) -> str:
    """Hash every parameter and persistent/non-persistent dense buffer."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicFoldExecutionError(
            "PyTorch is required for default model-state fingerprinting."
        ) from exc
    digest = hashlib.sha256()
    values = _all_named_state_tensors(model)
    if not values:
        raise DynamicFoldExecutionError("The child model state is empty.")
    for name, value in values:
        if not isinstance(value, torch.Tensor):
            raise DynamicFoldExecutionError(
                f"Model state value {name!r} is not a tensor."
            )
        if value.device.type == "meta" or value.layout != torch.strided:
            raise DynamicFoldExecutionError(
                "State fingerprinting requires materialized dense tensors."
            )
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        _update_digest_with_tensor_bytes(digest, tensor, torch)
    return digest.hexdigest()


def fingerprint_model_state_entries(model: Any) -> Mapping[str, str]:
    """Digest every parameter and persistent/non-persistent named buffer."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicFoldExecutionError(
            "PyTorch is required for default model-state fingerprinting."
        ) from exc
    values = _all_named_state_tensors(model)
    if not values:
        raise DynamicFoldExecutionError("The child model state is empty.")
    result: dict[str, str] = {}
    for name, value in values:
        if not isinstance(value, torch.Tensor):
            raise DynamicFoldExecutionError(
                f"Model state value {name!r} is not a tensor."
            )
        if value.device.type == "meta" or value.layout != torch.strided:
            raise DynamicFoldExecutionError(
                "State fingerprinting requires materialized dense tensors."
            )
        tensor = value.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        _update_digest_with_tensor_bytes(digest, tensor, torch)
        result[name] = digest.hexdigest()
    return MappingProxyType(result)


def _named_values(model: Any, method_name: str) -> tuple[tuple[str, Any], ...]:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise DynamicFoldExecutionError(
            f"The child model has no {method_name}() method."
        )
    try:
        return tuple(method(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        return tuple(method())


def _all_named_state_tensors(model: Any) -> tuple[tuple[str, Any], ...]:
    values = _named_values(model, "named_parameters") + _named_values(
        model, "named_buffers"
    )
    if len({name for name, _value in values}) != len(values):
        raise DynamicFoldExecutionError(
            "Parameter and buffer state names must be globally unique."
        )
    return tuple(sorted(values, key=lambda item: item[0]))


def _hook_attribute_names(
    owner: Any,
    required: Sequence[str],
) -> tuple[str, ...]:
    discovered = {str(name) for name in vars(owner) if "hook" in str(name).lower()}
    return tuple(sorted(set(required) | discovered))


def _hook_item_signature(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return (
            "value",
            type(value).__module__,
            type(value).__qualname__,
            value,
        )
    return (
        "identity",
        type(value).__module__,
        type(value).__qualname__,
        id(value),
    )


def _hook_value_signature(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__module__,
            type(value).__qualname__,
            id(value),
            tuple(
                (
                    _hook_item_signature(key),
                    _hook_item_signature(item),
                )
                for key, item in value.items()
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            "sequence",
            type(value).__module__,
            type(value).__qualname__,
            id(value),
            tuple(_hook_item_signature(item) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        return (
            "set",
            type(value).__module__,
            type(value).__qualname__,
            id(value),
            tuple(sorted(_hook_item_signature(item) for item in value)),
        )
    return _hook_item_signature(value)


def _owner_hook_signature(
    owner: Any,
    required: Sequence[str],
) -> tuple[tuple[str, Any], ...]:
    result = []
    for name in _hook_attribute_names(owner, required):
        if not hasattr(owner, name):
            result.append((name, ("absent",)))
            continue
        result.append((name, _hook_value_signature(getattr(owner, name))))
    return tuple(result)


def _hook_state_signature(model: Any) -> tuple[Any, ...]:
    from torch.nn.modules import module as torch_module

    return (
        (
            "global",
            _owner_hook_signature(torch_module, _GLOBAL_HOOK_ATTRIBUTES),
        ),
        tuple(
            (
                id(module),
                _owner_hook_signature(module, _MODULE_HOOK_ATTRIBUTES),
            )
            for module in model.modules()
        ),
    )


def _copy_hook_contents(value: Any) -> tuple[str, Any]:
    if isinstance(value, Mapping):
        return ("mapping", tuple(value.items()))
    if isinstance(value, list):
        return ("list", tuple(value))
    if isinstance(value, set):
        return ("set", frozenset(value))
    return ("value", value)


@dataclass(frozen=True, slots=True)
class _HookBindingSnapshot:
    owner: Any
    name: str
    existed: bool
    original_binding: Any
    saved_contents: tuple[str, Any]


def _capture_owner_hooks(
    owner: Any,
    required: Sequence[str],
) -> tuple[_HookBindingSnapshot, ...]:
    result = []
    for name in _hook_attribute_names(owner, required):
        existed = hasattr(owner, name)
        original = getattr(owner, name, None)
        result.append(
            _HookBindingSnapshot(
                owner=owner,
                name=name,
                existed=existed,
                original_binding=original,
                saved_contents=(
                    _copy_hook_contents(original) if existed else ("absent", None)
                ),
            )
        )
    return tuple(result)


def _capture_hook_bindings(model: Any) -> tuple[_HookBindingSnapshot, ...]:
    from torch.nn.modules import module as torch_module

    result = list(_capture_owner_hooks(torch_module, _GLOBAL_HOOK_ATTRIBUTES))
    for module in model.modules():
        result.extend(_capture_owner_hooks(module, _MODULE_HOOK_ATTRIBUTES))
    return tuple(result)


def _restore_hook_bindings(
    snapshots: Sequence[_HookBindingSnapshot],
) -> None:
    owners: dict[int, tuple[Any, set[str]]] = {}
    for snapshot in snapshots:
        key = id(snapshot.owner)
        if key not in owners:
            owners[key] = (snapshot.owner, set())
        owners[key][1].add(snapshot.name)
    for owner, expected_names in owners.values():
        unexpected = {
            str(name)
            for name in vars(owner)
            if "hook" in str(name).lower() and name not in expected_names
        }
        for name in sorted(unexpected):
            delattr(owner, name)
    for snapshot in snapshots:
        if not snapshot.existed:
            if hasattr(snapshot.owner, snapshot.name):
                delattr(snapshot.owner, snapshot.name)
            continue
        kind, contents = snapshot.saved_contents
        original = snapshot.original_binding
        if kind == "mapping":
            original.clear()
            original.update(contents)
        elif kind == "list":
            original[:] = contents
        elif kind == "set":
            original.clear()
            original.update(contents)
        setattr(snapshot.owner, snapshot.name, original)


def _runtime_invariant_signature(model: Any) -> tuple[Any, ...]:
    parameters = _named_values(model, "named_parameters")
    if any(parameter.grad is not None for _name, parameter in parameters):
        raise DynamicFoldExecutionError(
            "Dynamic folding requires a zero-gradient child state."
        )
    return (
        tuple(
            (
                name,
                id(parameter),
                bool(parameter.requires_grad),
                parameter.grad is None,
            )
            for name, parameter in parameters
        ),
        tuple(
            (name, id(buffer)) for name, buffer in _named_values(model, "named_buffers")
        ),
        tuple((id(module), bool(module.training)) for module in model.modules()),
        _hook_state_signature(model),
    )


def fingerprint_model_runtime_state(model: Any) -> str:
    """Hash identities, modes, gradient flags, and hook registries."""

    return canonical_sha256(_runtime_invariant_signature(model))


@dataclass(frozen=True, slots=True)
class _WholeModelSnapshot:
    state_storage: CapturedModelState
    parameter_bindings: tuple[tuple[str, Any, bool], ...]
    buffer_bindings: tuple[tuple[str, Any], ...]
    module_modes: tuple[tuple[Any, bool], ...]
    hook_bindings: tuple[_HookBindingSnapshot, ...]
    state_fingerprint: str

    @classmethod
    def capture(
        cls,
        model: Any,
        config: DynamicFoldSnapshotConfig | None = None,
    ) -> _WholeModelSnapshot:
        parameters = _named_values(model, "named_parameters")
        if any(parameter.grad is not None for _name, parameter in parameters):
            raise DynamicFoldExecutionError(
                "Dynamic folding cannot snapshot a child with gradients."
            )
        state_storage = capture_model_state(
            _all_named_state_tensors(model),
            config or DynamicFoldSnapshotConfig.memory(),
        )
        try:
            return cls(
                state_storage=state_storage,
                parameter_bindings=tuple(
                    (name, parameter, bool(parameter.requires_grad))
                    for name, parameter in parameters
                ),
                buffer_bindings=_named_values(model, "named_buffers"),
                module_modes=tuple(
                    (module, bool(module.training)) for module in model.modules()
                ),
                hook_bindings=_capture_hook_bindings(model),
                state_fingerprint=fingerprint_model_state(model),
            )
        except BaseException:
            state_storage.cleanup()
            raise

    def restore(self, model: Any) -> None:
        current_parameters = dict(_named_values(model, "named_parameters"))
        current_buffers = dict(_named_values(model, "named_buffers"))
        if set(current_parameters) != {
            name for name, _parameter, _requires_grad in self.parameter_bindings
        }:
            raise DynamicFoldExecutionError(
                "The child parameter topology changed during folding."
            )
        if set(current_buffers) != {name for name, _buffer in self.buffer_bindings}:
            raise DynamicFoldExecutionError(
                "The child buffer topology changed during folding."
            )
        for name, parameter, requires_grad in self.parameter_bindings:
            if current_parameters[name] is not parameter:
                raise DynamicFoldExecutionError(
                    "A fold transaction replaced a model Parameter object."
                )
            parameter.grad = None
            if bool(parameter.requires_grad) != requires_grad:
                parameter.requires_grad_(requires_grad)
        for name, buffer in self.buffer_bindings:
            if current_buffers[name] is not buffer:
                raise DynamicFoldExecutionError(
                    "A fold transaction replaced a model buffer object."
                )
        current_state = dict(_all_named_state_tensors(model))
        if set(current_state) != set(self.state_storage.names):
            raise DynamicFoldExecutionError(
                "The model state topology changed during folding."
            )
        self.state_storage.restore(current_state)
        for module, training in self.module_modes:
            module.training = training
        _restore_hook_bindings(self.hook_bindings)
        if fingerprint_model_state(model) != self.state_fingerprint:
            raise DynamicFoldExecutionError(
                "Whole-run rollback did not restore the exact initial model."
            )

    def cleanup(self) -> None:
        self.state_storage.cleanup()


class _WholeModelSnapshotScope:
    """Own every parent/child snapshot for exactly one executor run."""

    def __init__(self, config: DynamicFoldSnapshotConfig) -> None:
        self.config = config
        self._snapshots: list[_WholeModelSnapshot] = []

    def capture(self, model: Any) -> _WholeModelSnapshot:
        snapshot = _WholeModelSnapshot.capture(model, self.config)
        self._snapshots.append(snapshot)
        return snapshot

    def cleanup(self) -> None:
        failures = []
        for snapshot in reversed(self._snapshots):
            try:
                snapshot.cleanup()
            except BaseException as exc:
                failures.append(exc)
        self._snapshots.clear()
        if failures:
            raise DynamicFoldExecutionError(
                "One or more whole-model snapshots could not be cleaned."
            ) from failures[0]


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    fingerprint: str
    payload: Any = None
    channels: tuple[str, ...] = ()
    parent_forward_count: int = 0
    child_forward_count: int = 0
    root_fingerprint: str | None = None
    payload_fingerprint: str = field(init=False)
    payload_integrity_mode: str = field(init=False)
    binding_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fingerprint", _digest(self.fingerprint, "evidence fingerprint")
        )
        object.__setattr__(
            self,
            "root_fingerprint",
            (
                self.fingerprint
                if self.root_fingerprint is None
                else _digest(
                    self.root_fingerprint,
                    "evidence root fingerprint",
                )
            ),
        )
        channels = tuple(sorted(_text_channel(value) for value in self.channels))
        if len(set(channels)) != len(channels):
            raise DynamicFoldExecutionError("Evidence channel names must be unique.")
        object.__setattr__(self, "channels", channels)
        for name in ("parent_forward_count", "child_forward_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DynamicFoldExecutionError(f"{name} must be non-negative.")
        payload_fingerprint, payload_integrity_mode = _fold_evidence_payload_details(
            self.payload
        )
        object.__setattr__(self, "payload_fingerprint", payload_fingerprint)
        object.__setattr__(self, "payload_integrity_mode", payload_integrity_mode)
        object.__setattr__(
            self,
            "binding_fingerprint",
            canonical_sha256(
                {
                    "schema": "neural-forge-fold-evidence-binding-v1",
                    "evidence_fingerprint": self.fingerprint,
                    "root_fingerprint": self.root_fingerprint,
                    "payload_fingerprint": payload_fingerprint,
                    "payload_integrity_mode": payload_integrity_mode,
                    "channels": list(self.channels),
                    "parent_forward_count": self.parent_forward_count,
                    "child_forward_count": self.child_forward_count,
                }
            ),
        )


def _assert_fold_evidence_payload_unchanged(
    evidence: FoldEvidence,
    phase: str,
) -> None:
    try:
        observed, observed_mode = _fold_evidence_payload_details(evidence.payload)
    except BaseException as exc:
        raise DynamicFoldExecutionError(
            f"{phase} evidence payload integrity verification failed."
        ) from exc
    if (
        observed != evidence.payload_fingerprint
        or observed_mode != evidence.payload_integrity_mode
    ):
        raise DynamicFoldExecutionError(f"{phase} mutated the bound evidence payload.")


def _evidence_derivation_manifest(
    evidence: FoldEvidence,
    action: FoldAction,
    child_state_fingerprint: str,
) -> tuple[Mapping[str, Any], str]:
    manifest = {
        "schema": "neural-forge-fold-evidence-derivation-v1",
        "action_id": action.action_id,
        "action_sha256": action.action_sha256,
        "method_id": action.method_id,
        "child_state_fingerprint": _digest(
            child_state_fingerprint,
            "evidence derivation child state fingerprint",
        ),
        "evidence_fingerprint": evidence.fingerprint,
        "evidence_root_fingerprint": evidence.root_fingerprint,
        "evidence_payload_fingerprint": evidence.payload_fingerprint,
        "evidence_payload_integrity_mode": evidence.payload_integrity_mode,
        "evidence_channels": list(evidence.channels),
        "evidence_binding_fingerprint": evidence.binding_fingerprint,
    }
    return MappingProxyType(manifest), canonical_sha256(manifest)


@dataclass(frozen=True, slots=True)
class FoldDecision:
    accepted: bool
    reason: str
    metrics: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise DynamicFoldExecutionError("Fold decision accepted must be boolean.")
        if not isinstance(self.reason, str):
            raise DynamicFoldExecutionError("Fold decision reason must be a string.")
        object.__setattr__(self, "metrics", _metrics(self.metrics))

    @classmethod
    def accept(
        cls,
        reason: str,
        metrics: Mapping[str, JsonScalar] | None = None,
    ) -> "FoldDecision":
        return cls(True, reason, tuple((metrics or {}).items()))

    @classmethod
    def reject(
        cls,
        reason: str,
        metrics: Mapping[str, JsonScalar] | None = None,
    ) -> "FoldDecision":
        return cls(False, reason, tuple((metrics or {}).items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


class FoldTransaction(Protocol):
    method_id: str
    candidate_state_fingerprint: str
    active: bool

    def commit(self) -> Mapping[str, Any]: ...

    def rollback(self) -> Mapping[str, Any]: ...


class FoldOperator(Protocol):
    method_id: str

    def prepare(
        self,
        action: FoldAction,
        context: "FoldExecutionContext",
        evidence: FoldEvidence,
    ) -> FoldTransaction: ...


class FoldEvidenceProvider(Protocol):
    def prepare(
        self,
        action: FoldAction,
        context: "FoldExecutionContext",
    ) -> FoldEvidence: ...

    def refresh(
        self,
        action: FoldAction,
        context: "FoldExecutionContext",
        transaction_report: Mapping[str, Any],
    ) -> None: ...

    def restore(
        self,
        context: "FoldExecutionContext",
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None: ...


class FoldEvaluator(Protocol):
    def evaluate(self, context: "FoldEvaluationContext") -> FoldDecision | bool: ...


class FoldInitializer(Protocol):
    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: "FoldExecutionContext",
    ) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class FoldExecutionContext:
    parent_models: Mapping[str, Any]
    child_model: Any
    parent_graphs: Mapping[str, ModelCircuitGraph]
    child_graph: ModelCircuitGraph
    state_fingerprint_fn: Callable[[Any], str] = fingerprint_model_state
    parent_state_fingerprint_fn: Callable[[Any], str] = fingerprint_model_state
    runtime: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.parent_models = MappingProxyType(dict(self.parent_models))
        self.parent_graphs = MappingProxyType(dict(self.parent_graphs))
        if set(self.parent_models) != set(self.parent_graphs):
            raise DynamicFoldExecutionError(
                "Parent model and graph names must match exactly."
            )
        if any(
            self.parent_graphs[name].model_name != name for name in self.parent_graphs
        ):
            raise DynamicFoldExecutionError(
                "Parent graph model_name must equal its context key."
            )
        if not isinstance(self.child_graph, ModelCircuitGraph):
            raise DynamicFoldExecutionError("child_graph must be ModelCircuitGraph.")
        if not callable(self.state_fingerprint_fn):
            raise DynamicFoldExecutionError("state_fingerprint_fn must be callable.")
        if not callable(self.parent_state_fingerprint_fn):
            raise DynamicFoldExecutionError(
                "parent_state_fingerprint_fn must be callable."
            )
        self.runtime = {} if self.runtime is None else dict(self.runtime)

    @property
    def child_state_fingerprint(self) -> str:
        return _digest(
            self.state_fingerprint_fn(self.child_model),
            "child state fingerprint callback result",
        )

    @property
    def parent_state_fingerprints(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                name: _digest(
                    self.parent_state_fingerprint_fn(model),
                    f"parent {name} state fingerprint callback result",
                )
                for name, model in self.parent_models.items()
            }
        )


@dataclass(frozen=True, slots=True)
class FoldEvaluationContext:
    action: FoldAction
    execution_signature: ExperimentSignature
    execution_context: FoldExecutionContext
    evidence: FoldEvidence
    transaction: FoldTransaction
    incumbent_state_fingerprint: str
    candidate_state_fingerprint: str


@dataclass(frozen=True, slots=True)
class FoldAttempt:
    action_id: str
    method_id: str
    execution_signature: ExperimentSignature
    evidence_fingerprint: str
    evidence_root_fingerprint: str
    incumbent_state_fingerprint: str
    candidate_state_fingerprint: str
    decision: FoldDecision
    transaction_report: Mapping[str, Any]
    rollback_exact: bool | None
    ledger_record_sha256: str
    evidence_payload_fingerprint: str = "0" * 64
    evidence_payload_integrity_mode: str = "unrecorded"
    evidence_channels: tuple[str, ...] = ()
    evidence_binding_fingerprint: str = "0" * 64
    evidence_derivation_manifest: Mapping[str, Any] = field(default_factory=dict)
    evidence_derivation_fingerprint: str = "0" * 64
    evidence_payload_integrity_reverified: bool = False
    operator_prepare_implementation_fingerprint: str = "0" * 64
    operator_prepare_static_zero_training_audit_passed: bool = False
    transaction_implementation_fingerprint: str = "0" * 64
    transaction_commit_implementation_fingerprint: str = "0" * 64
    transaction_rollback_implementation_fingerprint: str = "0" * 64
    transaction_static_zero_training_audit_passed: bool = False

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "method_id": self.method_id,
            "execution_signature": self.execution_signature.as_dict(),
            "evidence_fingerprint": self.evidence_fingerprint,
            "evidence_root_fingerprint": self.evidence_root_fingerprint,
            "evidence_payload_fingerprint": self.evidence_payload_fingerprint,
            "evidence_payload_integrity_mode": self.evidence_payload_integrity_mode,
            "evidence_channels": list(self.evidence_channels),
            "evidence_binding_fingerprint": self.evidence_binding_fingerprint,
            "evidence_derivation_manifest": dict(self.evidence_derivation_manifest),
            "evidence_derivation_fingerprint": (self.evidence_derivation_fingerprint),
            "evidence_payload_integrity_reverified": (
                self.evidence_payload_integrity_reverified
            ),
            "operator_prepare_implementation_fingerprint": (
                self.operator_prepare_implementation_fingerprint
            ),
            "operator_prepare_static_zero_training_audit_passed": (
                self.operator_prepare_static_zero_training_audit_passed
            ),
            "transaction_implementation_fingerprint": (
                self.transaction_implementation_fingerprint
            ),
            "transaction_commit_implementation_fingerprint": (
                self.transaction_commit_implementation_fingerprint
            ),
            "transaction_rollback_implementation_fingerprint": (
                self.transaction_rollback_implementation_fingerprint
            ),
            "transaction_static_zero_training_audit_passed": (
                self.transaction_static_zero_training_audit_passed
            ),
            "incumbent_state_fingerprint": self.incumbent_state_fingerprint,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "decision": self.decision.as_dict(),
            "transaction_report": dict(self.transaction_report),
            "rollback_exact": self.rollback_exact,
            "ledger_record_sha256": self.ledger_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class DynamicFoldRun:
    plan_fingerprint: str
    status: str
    initial_state_fingerprint: str
    final_state_fingerprint: str
    bootstrap_report: Mapping[str, Any]
    attempts: tuple[FoldAttempt, ...]
    accepted_action_ids: tuple[str, ...]
    completed_target_parameter_paths: tuple[str, ...]
    final_decision: FoldDecision | None
    final_evidence_fingerprint: str | None
    final_evidence_root_fingerprint: str | None
    ledger_fingerprint: str
    final_evidence_payload_fingerprint: str | None = None
    final_evidence_payload_integrity_mode: str | None = None
    final_evidence_channels: tuple[str, ...] = ()
    final_evidence_binding_fingerprint: str | None = None
    final_evidence_derivation_manifest: Mapping[str, Any] | None = None
    final_evidence_derivation_fingerprint: str | None = None
    final_evidence_payload_integrity_reverified: bool = False
    executed_fold_implementation_fingerprint: str = "0" * 64
    executed_fold_static_zero_training_audit_passed: bool = False
    operator_implementation_fingerprint: str = "0" * 64
    operator_static_zero_training_audit_passed: bool = False
    optimizer_constructions: int = 0
    optimizer_steps: int = 0
    backward_calls: int = 0
    autograd_grad_calls: int = 0
    pruning_operations: int = 0
    quantization_operations: int = 0
    whole_model_snapshot_backend: str = "memory"
    observed_parent_module_forward_calls: int = 0
    observed_parent_root_forward_calls: int = 0
    observed_final_child_module_forward_calls: int = 0
    observed_final_child_root_forward_calls: int = 0
    observed_final_child_inference_entrypoint_calls: int = 0

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": RUN_SCHEMA,
            "plan_fingerprint": self.plan_fingerprint,
            "status": self.status,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "final_state_fingerprint": self.final_state_fingerprint,
            "bootstrap_report": dict(self.bootstrap_report),
            "attempts": [value.as_dict() for value in self.attempts],
            "accepted_action_ids": list(self.accepted_action_ids),
            "completed_target_parameter_paths": list(
                self.completed_target_parameter_paths
            ),
            "final_decision": (
                None if self.final_decision is None else self.final_decision.as_dict()
            ),
            "final_evidence_fingerprint": self.final_evidence_fingerprint,
            "final_evidence_root_fingerprint": (self.final_evidence_root_fingerprint),
            "final_evidence_payload_fingerprint": (
                self.final_evidence_payload_fingerprint
            ),
            "final_evidence_payload_integrity_mode": (
                self.final_evidence_payload_integrity_mode
            ),
            "final_evidence_channels": list(self.final_evidence_channels),
            "final_evidence_binding_fingerprint": (
                self.final_evidence_binding_fingerprint
            ),
            "final_evidence_derivation_manifest": (
                None
                if self.final_evidence_derivation_manifest is None
                else dict(self.final_evidence_derivation_manifest)
            ),
            "final_evidence_derivation_fingerprint": (
                self.final_evidence_derivation_fingerprint
            ),
            "final_evidence_payload_integrity_reverified": (
                self.final_evidence_payload_integrity_reverified
            ),
            "ledger_fingerprint": self.ledger_fingerprint,
            "operator_implementation_fingerprint": (
                self.operator_implementation_fingerprint
            ),
            "operator_static_zero_training_audit_passed": (
                self.operator_static_zero_training_audit_passed
            ),
            "executed_fold_implementation_fingerprint": (
                self.executed_fold_implementation_fingerprint
            ),
            "executed_fold_static_zero_training_audit_passed": (
                self.executed_fold_static_zero_training_audit_passed
            ),
            "best_effort_fold_only_guard_active": True,
            "all_parent_and_child_parameters_frozen_during_callbacks": True,
            "all_parameter_gradients_absent_at_exit": True,
            "observed_no_guarded_training_api_attempts": (
                self.optimizer_constructions == 0
                and self.optimizer_steps == 0
                and self.backward_calls == 0
                and self.autograd_grad_calls == 0
                and self.pruning_operations == 0
                and self.quantization_operations == 0
            ),
            "historical_zero_training_proven_for_arbitrary_plugins": False,
            "operator_implementations_require_external_audit": True,
            "static_callable_audit_is_not_arbitrary_plugin_proof": True,
            "optimizer_constructions": self.optimizer_constructions,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "autograd_grad_calls": self.autograd_grad_calls,
            "pruning_operations": self.pruning_operations,
            "quantization_operations": self.quantization_operations,
            "pruning_quantization_api_observation_scope": (
                "process-local public pruning and common public/low-level "
                "quantization bindings installed during the fold guard; "
                "aliases captured before guard entry are excluded"
            ),
            "prebound_pruning_quantization_aliases_runtime_observed": False,
            "whole_model_snapshot_backend": (self.whole_model_snapshot_backend),
            "whole_model_snapshot_cleanup_completed": True,
            "observed_parent_module_forward_calls": (
                self.observed_parent_module_forward_calls
            ),
            "observed_parent_root_forward_calls": (
                self.observed_parent_root_forward_calls
            ),
            "parent_forward_observation_active": True,
            "observed_final_child_module_forward_calls": (
                self.observed_final_child_module_forward_calls
            ),
            "observed_final_child_root_forward_calls": (
                self.observed_final_child_root_forward_calls
            ),
            "observed_final_child_inference_entrypoint_calls": (
                self.observed_final_child_inference_entrypoint_calls
            ),
            "final_child_forward_observation_active": True,
            "evidence_payload_fingerprint_compiler_owned": True,
            "generic_evidence_root_authenticity_requires_registered_provider_trust": (
                True
            ),
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


class StaticEvidenceProvider:
    """Use one immutable parent evidence identity for every action."""

    def __init__(self, fingerprint: str, payload: Any = None) -> None:
        self.fingerprint = _digest(fingerprint, "static evidence fingerprint")
        self.payload = payload
        self.refresh_count = 0
        self.restore_count = 0

    @property
    def base_fingerprint(self) -> str:
        return self.fingerprint

    def prepare(
        self, action: FoldAction, context: FoldExecutionContext
    ) -> FoldEvidence:
        del context
        return FoldEvidence(
            self.fingerprint,
            self.payload,
            channels=action.evidence_requirements,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del context, state_fingerprint, action_ids
        self.restore_count += 1


class NoOpInitializer:
    def initialize(
        self, plan: DynamicFoldPlan, context: FoldExecutionContext
    ) -> Mapping[str, Any]:
        del plan
        return {
            "kind": "no_op",
            "child_state_fingerprint": context.child_state_fingerprint,
            "complete_target_coverage": False,
        }


def _call_expression_name(node: ast.Call) -> str:
    function = node.func
    parts = []
    while isinstance(function, ast.Attribute):
        parts.append(function.attr)
        function = function.value
    if isinstance(function, ast.Name):
        parts.append(function.id)
    return ".".join(reversed(parts))


def _forbidden_training_calls(source: str) -> tuple[str, ...]:
    if not source:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ("<unparseable-source>",)
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _call_expression_name(node)
        if (
            dotted.endswith(".backward")
            or dotted.endswith(".step")
            or dotted
            in {
                "backward",
                "torch.autograd.backward",
                "torch.autograd.grad",
                "torch.enable_grad",
                "torch.set_grad_enabled",
            }
            or dotted.startswith("torch.optim.")
        ):
            forbidden.append(dotted)
    return tuple(sorted(set(forbidden)))


def _code_constant_identity(value: Any) -> Any:
    if isinstance(value, CodeType):
        return _code_identity(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_code_constant_identity(item) for item in value]}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    return {
        "kind": "typed-repr",
        "type": _qualified_type_name(value),
        "repr": repr(value),
    }


def _code_identity(code: CodeType) -> dict[str, Any]:
    return {
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant_identity(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


@dataclass(frozen=True, slots=True)
class _CallableImplementationBinding:
    role: str
    owner_type: str
    implementation_module: str
    implementation_qualname: str
    source_sha256: str
    code_sha256: str
    source_available: bool
    static_forbidden_training_calls: tuple[str, ...]

    @property
    def static_zero_training_audit_passed(self) -> bool:
        return self.source_available and not self.static_forbidden_training_calls

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "role": self.role,
            "owner_type": self.owner_type,
            "implementation_module": self.implementation_module,
            "implementation_qualname": self.implementation_qualname,
            "source_sha256": self.source_sha256,
            "code_sha256": self.code_sha256,
            "source_available": self.source_available,
            "static_forbidden_training_calls": list(
                self.static_forbidden_training_calls
            ),
            "static_zero_training_audit_passed": (
                self.static_zero_training_audit_passed
            ),
            "static_audit_scope": "source-visible-direct-call-ast",
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def _bound_class_implementation(
    instance: Any,
    method_name: str,
    *,
    role: str,
) -> tuple[Callable[..., Any], _CallableImplementationBinding]:
    try:
        instance_attributes = vars(instance)
    except TypeError:
        instance_attributes = {}
    if method_name in instance_attributes:
        raise DynamicFoldExecutionError(
            f"{role} cannot use an instance-level {method_name} override."
        )
    try:
        bound = getattr(instance, method_name)
        declared = inspect.getattr_static(type(instance), method_name)
    except (AttributeError, TypeError) as exc:
        raise DynamicFoldExecutionError(
            f"{role} must expose a class-declared {method_name} method."
        ) from exc
    if isinstance(declared, (staticmethod, classmethod)):
        declared = declared.__func__
    implementation = bound.__func__ if inspect.ismethod(bound) else bound
    if not callable(bound) or not inspect.isfunction(implementation):
        raise DynamicFoldExecutionError(
            f"{role} {method_name} must resolve to a Python class function."
        )
    if implementation is not declared:
        raise DynamicFoldExecutionError(
            f"{role} {method_name} binding differs from its class declaration."
        )
    try:
        source = textwrap.dedent(inspect.getsource(implementation))
    except (OSError, TypeError):
        source = ""
    binding = _CallableImplementationBinding(
        role=role,
        owner_type=_qualified_type_name(instance),
        implementation_module=str(implementation.__module__),
        implementation_qualname=str(implementation.__qualname__),
        source_sha256=(canonical_sha256(source) if source else "0" * 64),
        code_sha256=canonical_sha256(_code_identity(implementation.__code__)),
        source_available=bool(source),
        static_forbidden_training_calls=_forbidden_training_calls(source),
    )
    return bound, binding


def _require_clean_callable(binding: _CallableImplementationBinding) -> None:
    if not binding.static_zero_training_audit_passed:
        detail = (
            ", ".join(binding.static_forbidden_training_calls)
            if binding.static_forbidden_training_calls
            else "source unavailable"
        )
        raise DynamicFoldExecutionError(
            f"{binding.role} failed the static zero-training audit: {detail}."
        )


def _reauthenticate_callable(
    instance: Any,
    method_name: str,
    expected: _CallableImplementationBinding,
) -> Callable[..., Any]:
    callback, observed = _bound_class_implementation(
        instance,
        method_name,
        role=expected.role,
    )
    if observed.fingerprint != expected.fingerprint:
        raise DynamicFoldExecutionError(
            f"{expected.role} implementation changed after authentication."
        )
    _require_clean_callable(observed)
    return callback


@dataclass(frozen=True, slots=True)
class _TransactionImplementationBinding:
    commit: _CallableImplementationBinding
    rollback: _CallableImplementationBinding

    @property
    def static_zero_training_audit_passed(self) -> bool:
        return (
            self.commit.static_zero_training_audit_passed
            and self.rollback.static_zero_training_audit_passed
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "neural-forge-fold-transaction-implementations-v1",
                "commit": self.commit.as_dict(),
                "rollback": self.rollback.as_dict(),
            }
        )


def _bind_transaction_implementations(
    transaction: FoldTransaction,
) -> _TransactionImplementationBinding:
    _commit, commit = _bound_class_implementation(
        transaction,
        "commit",
        role="fold transaction commit",
    )
    _rollback, rollback = _bound_class_implementation(
        transaction,
        "rollback",
        role="fold transaction rollback",
    )
    _require_clean_callable(commit)
    _require_clean_callable(rollback)
    return _TransactionImplementationBinding(commit=commit, rollback=rollback)


class FoldOperatorRegistry:
    def __init__(self, operators: Sequence[FoldOperator] = ()) -> None:
        self._operators: dict[str, FoldOperator] = {}
        self._prepare_bindings: dict[str, _CallableImplementationBinding] = {}
        for operator in operators:
            self.register(operator)

    def register(self, operator: FoldOperator) -> None:
        method_id = getattr(operator, "method_id", None)
        if not isinstance(method_id, str) or not method_id:
            raise DynamicFoldExecutionError("Fold operator method_id is invalid.")
        if method_id in self._operators:
            raise DynamicFoldExecutionError(
                f"Fold operator {method_id!r} is already registered."
            )
        _prepare, prepare_binding = _bound_class_implementation(
            operator,
            "prepare",
            role=f"fold operator {method_id!r} prepare",
        )
        self._operators[method_id] = operator
        self._prepare_bindings[method_id] = prepare_binding

    def get(self, method_id: str) -> FoldOperator:
        try:
            return self._operators[method_id]
        except KeyError as exc:
            raise DynamicFoldExecutionError(
                f"No executable operator is registered for {method_id!r}."
            ) from exc

    def authenticated_prepare(
        self,
        method_id: str,
    ) -> tuple[FoldOperator, Callable[..., Any], _CallableImplementationBinding]:
        operator = self.get(method_id)
        if getattr(operator, "method_id", None) != method_id:
            raise DynamicFoldExecutionError(
                "A registered fold operator changed its method_id."
            )
        expected = self._prepare_bindings[method_id]
        callback = _reauthenticate_callable(
            operator,
            "prepare",
            expected,
        )
        return operator, callback, expected

    @property
    def method_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._operators))

    @staticmethod
    def _operator_descriptor(
        operator: FoldOperator,
        prepare_binding: _CallableImplementationBinding,
    ) -> dict[str, Any]:
        implementation_type = type(operator)
        qualified_type = (
            f"{implementation_type.__module__}.{implementation_type.__qualname__}"
        )
        resource_fingerprint = getattr(
            operator,
            "resource_fingerprint",
            None,
        )
        if resource_fingerprint is not None:
            resource_fingerprint = _digest(
                resource_fingerprint,
                "operator resource_fingerprint",
            )
        return {
            "method_id": str(operator.method_id),
            "implementation_type": qualified_type,
            "source_sha256": prepare_binding.source_sha256,
            "source_available": prepare_binding.source_available,
            "static_forbidden_training_calls": list(
                prepare_binding.static_forbidden_training_calls
            ),
            "prepare_implementation": prepare_binding.as_dict(),
            "prepare_implementation_fingerprint": prepare_binding.fingerprint,
            "prepare_static_zero_training_audit_passed": (
                prepare_binding.static_zero_training_audit_passed
            ),
            "resource_fingerprint": resource_fingerprint,
        }

    @property
    def implementation_manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._operator_descriptor(
                self._operators[method_id],
                self._prepare_bindings[method_id],
            )
            for method_id in sorted(self._operators)
        )

    @property
    def implementation_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": RUN_SCHEMA,
                "kind": "registered_operator_implementation_manifest",
                "operators": self.implementation_manifest,
            }
        )

    @property
    def static_zero_training_audit_passed(self) -> bool:
        return all(
            row["prepare_static_zero_training_audit_passed"]
            for row in self.implementation_manifest
        )


def _decision(value: FoldDecision | bool) -> FoldDecision:
    if isinstance(value, FoldDecision):
        return value
    if isinstance(value, bool):
        return FoldDecision(value, "")
    raise DynamicFoldExecutionError("Fold evaluator must return FoldDecision or bool.")


def _current_signature(
    template: ExperimentSignature,
    *,
    child_state_fingerprint: str,
    evidence_fingerprint: str,
) -> ExperimentSignature:
    return replace(
        template,
        child_state_fingerprint=child_state_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
    )


class _FoldOnlyTorchGuard:
    """Process-local guard that forbids training APIs during fold execution."""

    _lock = threading.RLock()

    def __init__(self) -> None:
        self.optimizer_constructions = 0
        self.optimizer_steps = 0
        self.backward_calls = 0
        self.autograd_grad_calls = 0
        self.pruning_operations = 0
        self.quantization_operations = 0
        self._patches: list[tuple[Any, str, Any, bool]] = []
        self._patched_bindings: set[tuple[int, str]] = set()
        self._no_grad: Any | None = None

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        key = (id(owner), name)
        if key in self._patched_bindings:
            return
        try:
            owner_bindings = vars(owner)
            directly_owned = name in owner_bindings
        except TypeError:
            owner_bindings = {}
            directly_owned = True
        original = inspect.getattr_static(owner, name)
        self._patches.append((owner, name, original, directly_owned))
        self._patched_bindings.add(key)
        setattr(owner, name, replacement)

    def _patch_forbidden_operations(
        self,
        owner: Any,
        names: Sequence[str],
        *,
        counter_name: str,
        operation_label: str,
    ) -> None:
        for name in names:
            if not callable(getattr(owner, name, None)):
                continue

            def forbidden(
                *args: Any,
                _name: str = name,
                **kwargs: Any,
            ) -> None:
                del args, kwargs
                setattr(self, counter_name, getattr(self, counter_name) + 1)
                raise DynamicFoldExecutionError(
                    f"{operation_label} API {_name} is forbidden during "
                    "dynamic folding."
                )

            self._patch(owner, name, forbidden)

    def _patch_forbidden_classmethods(
        self,
        owners: Sequence[type[Any]],
        name: str,
        *,
        counter_name: str,
        operation_label: str,
    ) -> None:
        for owner in owners:
            if not callable(getattr(owner, name, None)):
                continue

            def forbidden(
                _class: type[Any],
                *args: Any,
                _owner_name: str = owner.__qualname__,
                _name: str = name,
                **kwargs: Any,
            ) -> None:
                del _class, args, kwargs
                setattr(self, counter_name, getattr(self, counter_name) + 1)
                raise DynamicFoldExecutionError(
                    f"{operation_label} API {_owner_name}.{_name} is "
                    "forbidden during dynamic folding."
                )

            self._patch(owner, name, classmethod(forbidden))

    def _patch_forbidden_methods(
        self,
        owners: Sequence[type[Any]],
        names: Sequence[str],
        *,
        counter_name: str,
        operation_label: str,
    ) -> None:
        for owner in owners:
            for name in names:
                if not callable(getattr(owner, name, None)):
                    continue

                def forbidden(
                    _instance: Any,
                    *args: Any,
                    _owner_name: str = owner.__qualname__,
                    _name: str = name,
                    **kwargs: Any,
                ) -> None:
                    del _instance, args, kwargs
                    setattr(
                        self,
                        counter_name,
                        getattr(self, counter_name) + 1,
                    )
                    raise DynamicFoldExecutionError(
                        f"{operation_label} API {_owner_name}.{_name} is "
                        "forbidden during dynamic folding."
                    )

                self._patch(owner, name, forbidden)

    def __enter__(self) -> _FoldOnlyTorchGuard:
        self._lock.acquire()
        try:
            import torch

            self._no_grad = torch.no_grad()
            self._no_grad.__enter__()

            def tensor_backward(_tensor: Any, *args: Any, **kwargs: Any) -> None:
                del args, kwargs
                self.backward_calls += 1
                raise DynamicFoldExecutionError(
                    "Backward calls are forbidden during dynamic folding."
                )

            def autograd_backward(*args: Any, **kwargs: Any) -> None:
                del args, kwargs
                self.backward_calls += 1
                raise DynamicFoldExecutionError(
                    "torch.autograd.backward is forbidden during dynamic folding."
                )

            def autograd_grad(*args: Any, **kwargs: Any) -> None:
                del args, kwargs
                self.autograd_grad_calls += 1
                raise DynamicFoldExecutionError(
                    "torch.autograd.grad is forbidden during dynamic folding."
                )

            self._patch(torch.Tensor, "backward", tensor_backward)
            self._patch(torch.autograd, "backward", autograd_backward)
            self._patch(torch.autograd, "grad", autograd_grad)

            original_set_grad_enabled = torch.set_grad_enabled

            def forbidden_enable_grad(*args: Any, **kwargs: Any) -> None:
                del args, kwargs
                self.autograd_grad_calls += 1
                raise DynamicFoldExecutionError(
                    "Enabling autograd is forbidden during dynamic folding."
                )

            def guarded_set_grad_enabled(mode: Any) -> Any:
                if bool(mode):
                    self.autograd_grad_calls += 1
                    raise DynamicFoldExecutionError(
                        "Enabling autograd is forbidden during dynamic folding."
                    )
                return original_set_grad_enabled(False)

            self._patch(torch, "enable_grad", forbidden_enable_grad)
            self._patch(torch, "set_grad_enabled", guarded_set_grad_enabled)

            seen: set[type[Any]] = set()
            for value in vars(torch.optim).values():
                if (
                    not isinstance(value, type)
                    or not issubclass(value, torch.optim.Optimizer)
                    or value in seen
                ):
                    continue
                seen.add(value)
                original_init = value.__init__
                original_step = value.step

                def forbidden_init(
                    _instance: Any,
                    *args: Any,
                    _original: Any = original_init,
                    **kwargs: Any,
                ) -> None:
                    del args, kwargs, _original
                    self.optimizer_constructions += 1
                    raise DynamicFoldExecutionError(
                        "Optimizer construction is forbidden during dynamic folding."
                    )

                def forbidden_step(
                    _instance: Any,
                    *args: Any,
                    _original: Any = original_step,
                    **kwargs: Any,
                ) -> None:
                    del args, kwargs, _original
                    self.optimizer_steps += 1
                    raise DynamicFoldExecutionError(
                        "Optimizer steps are forbidden during dynamic folding."
                    )

                self._patch(value, "__init__", forbidden_init)
                self._patch(value, "step", forbidden_step)

            # These are deliberately scoped to common public PyTorch mutation
            # entrypoints. Dense BF16/FP32 casts remain valid folding math and
            # are not patched. Retained state must be inspected separately.
            from torch.nn.utils import prune as torch_prune

            self._patch_forbidden_operations(
                torch_prune,
                (
                    "identity",
                    "random_unstructured",
                    "l1_unstructured",
                    "random_structured",
                    "ln_structured",
                    "global_unstructured",
                    "custom_from_mask",
                    "remove",
                ),
                counter_name="pruning_operations",
                operation_label="Pruning",
            )
            pruning_method_types = tuple(
                value
                for value in vars(torch_prune).values()
                if isinstance(value, type)
                and issubclass(value, torch_prune.BasePruningMethod)
            )
            self._patch_forbidden_classmethods(
                pruning_method_types,
                "apply",
                counter_name="pruning_operations",
                operation_label="Pruning",
            )
            self._patch_forbidden_methods(
                pruning_method_types,
                ("prune", "apply_mask", "remove"),
                counter_name="pruning_operations",
                operation_label="Pruning",
            )
            quantization_modules = []
            for module_name in (
                "torch.ao.quantization",
                "torch.ao.quantization.quantize",
                "torch.quantization",
                "torch.quantization.quantize",
                "torch.ao.quantization.quantize_fx",
                "torch.ao.quantization.quantize_pt2e",
                "torch.ao.quantization.pt2e",
                "torchao.quantization",
                "torchao.quantization.quantize_",
            ):
                try:
                    quantization_modules.append(importlib.import_module(module_name))
                except ImportError:
                    continue
            quantization_names = (
                "quantize",
                "quantize_dynamic",
                "quantize_qat",
                "prepare",
                "prepare_qat",
                "convert",
                "fuse_modules",
                "fuse_modules_qat",
                "propagate_qconfig_",
                "add_quant_dequant",
                "prepare_fx",
                "prepare_qat_fx",
                "convert_fx",
                "convert_to_reference_fx",
                "prepare_pt2e",
                "prepare_qat_pt2e",
                "convert_pt2e",
                "prepare_jit",
                "prepare_dynamic_jit",
                "convert_jit",
                "convert_dynamic_jit",
                "quantize_jit",
                "quantize_dynamic_jit",
            )
            for module in quantization_modules:
                self._patch_forbidden_operations(
                    module,
                    quantization_names,
                    counter_name="quantization_operations",
                    operation_label="Quantization",
                )
            self._patch_forbidden_operations(
                torch,
                (
                    "quantize_per_tensor",
                    "quantize_per_channel",
                    "fake_quantize_per_tensor_affine",
                    "fake_quantize_per_channel_affine",
                ),
                counter_name="quantization_operations",
                operation_label="Quantization",
            )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        del exc_type, exc, traceback
        for owner, name, original, directly_owned in reversed(self._patches):
            if directly_owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._patches.clear()
        self._patched_bindings.clear()
        if self._no_grad is not None:
            self._no_grad.__exit__(None, None, None)
            self._no_grad = None
        self._lock.release()


@dataclass(frozen=True, slots=True)
class FoldOnlyGuardReport:
    """Observed training-API counters for one guarded non-fold callback phase."""

    phase: str
    optimizer_constructions: int
    optimizer_steps: int
    backward_calls: int
    autograd_grad_calls: int
    pruning_operations: int
    quantization_operations: int
    protected_model_count: int

    @property
    def clean(self) -> bool:
        return all(
            value == 0
            for value in (
                self.optimizer_constructions,
                self.optimizer_steps,
                self.backward_calls,
                self.autograd_grad_calls,
                self.pruning_operations,
                self.quantization_operations,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "optimizer_constructions": self.optimizer_constructions,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "autograd_grad_calls": self.autograd_grad_calls,
            "pruning_operations": self.pruning_operations,
            "quantization_operations": self.quantization_operations,
            "pruning_observation_scope": (
                "process-local public PyTorch pruning module functions and "
                "exported BasePruningMethod class apply/prune/apply_mask/remove "
                "bindings installed during the guard"
            ),
            "quantization_observation_scope": (
                "process-local common public PyTorch/torchao quantization and "
                "low-level torch quantize/fake-quant bindings installed during "
                "the guard"
            ),
            "prebound_aliases_runtime_observed": False,
            "prebound_aliases_require_authenticated_source_audit": True,
            "ordinary_bfloat16_and_float32_casts_blocked": False,
            "protected_model_count": self.protected_model_count,
            "clean": self.clean,
        }


def run_fold_only_callback(
    callback: Callable[[], Any],
    *,
    phase: str,
    protected_models: Sequence[Any] = (),
    freeze_parameters: bool = True,
) -> tuple[Any, FoldOnlyGuardReport]:
    """Run a construction/scoring callback under the same no-training guard.

    Models exposing the normal ``parameters/state_dict/modules`` surface are
    frozen and fingerprinted before and after the callback.  This covers target
    materialization, stack construction, catalog preflight, and portfolio
    scoring without pretending arbitrary plugin implementations are trusted.
    """

    if not callable(callback):
        raise DynamicFoldExecutionError("Fold-only guarded callback must be callable.")
    phase = str(phase).strip()
    if not phase:
        raise DynamicFoldExecutionError(
            "Fold-only guarded callback phase must be nonempty."
        )
    if not isinstance(freeze_parameters, bool):
        raise DynamicFoldExecutionError("freeze_parameters must be boolean.")
    models = []
    seen_models: set[int] = set()
    for model in protected_models:
        if id(model) in seen_models:
            continue
        seen_models.add(id(model))
        if all(
            callable(getattr(model, name, None))
            for name in ("parameters", "state_dict", "modules")
        ):
            models.append(model)

    parameter_rows: list[tuple[Any, bool]] = []
    seen_parameters: set[int] = set()
    before = []
    for model in models:
        before.append(
            (
                model,
                fingerprint_model_state(model),
                fingerprint_model_runtime_state(model),
            )
        )
        for parameter in model.parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            if parameter.grad is not None:
                raise DynamicFoldExecutionError(
                    f"{phase} requires zero-gradient protected models."
                )
            parameter_rows.append((parameter, bool(parameter.requires_grad)))
    if freeze_parameters:
        for parameter, _requires_grad in parameter_rows:
            parameter.requires_grad_(False)

    guard = _FoldOnlyTorchGuard()
    caught: BaseException | None = None
    result: Any = None
    try:
        with guard:
            result = callback()
    except BaseException as exc:
        caught = exc
    finally:
        for parameter, requires_grad in parameter_rows:
            parameter.grad = None
            if freeze_parameters:
                parameter.requires_grad_(requires_grad)

    mutation = None
    for model, state_fingerprint, runtime_fingerprint in before:
        if (
            fingerprint_model_state(model) != state_fingerprint
            or fingerprint_model_runtime_state(model) != runtime_fingerprint
        ):
            mutation = DynamicFoldExecutionError(f"{phase} mutated a protected model.")
            break
    if mutation is not None:
        if caught is not None:
            raise mutation from caught
        raise mutation
    if caught is not None:
        raise caught

    report = FoldOnlyGuardReport(
        phase=phase,
        optimizer_constructions=guard.optimizer_constructions,
        optimizer_steps=guard.optimizer_steps,
        backward_calls=guard.backward_calls,
        autograd_grad_calls=guard.autograd_grad_calls,
        pruning_operations=guard.pruning_operations,
        quantization_operations=guard.quantization_operations,
        protected_model_count=len(models),
    )
    if not report.clean:
        raise DynamicFoldExecutionError(f"{phase} attempted a forbidden training API.")
    return result, report


class _ParentForwardObserver:
    """Observe parent ``nn.Module.__call__`` use throughout fold execution."""

    def __init__(
        self,
        parent_models: Mapping[str, Any],
        *,
        execution_contracts: (
            Mapping[str, BehaviorInferenceExecutionContract] | None
        ) = None,
    ) -> None:
        self.parent_models = dict(parent_models)
        self.execution_contracts = (
            {} if execution_contracts is None else dict(execution_contracts)
        )
        if set(self.execution_contracts) - set(self.parent_models):
            raise DynamicFoldExecutionError(
                "Forward observer execution contracts name unknown models."
            )
        self.module_forward_calls = 0
        self.root_forward_calls = 0
        self.inference_entrypoint_calls = 0
        self._stack: ExitStack | None = None
        self._counts: list[Mapping[str, int]] = []

    def __enter__(self) -> "_ParentForwardObserver":
        stack = ExitStack()
        try:
            for parent_name, model in sorted(self.parent_models.items()):
                self._counts.append(
                    stack.enter_context(
                        observe_model_execution(
                            model,
                            contract=self.execution_contracts.get(parent_name),
                        )
                    )
                )
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._stack is not None:
            self._stack.__exit__(_exc_type, _exc, _traceback)
            self._stack = None
        self.module_forward_calls = sum(
            int(counts["module_forward_calls"]) for counts in self._counts
        )
        self.root_forward_calls = sum(
            int(counts["root_forward_calls"]) for counts in self._counts
        )
        self.inference_entrypoint_calls = sum(
            int(counts["inference_entrypoint_calls"]) for counts in self._counts
        )
        self._counts.clear()


class DynamicFoldExecutor:
    """Execute one plan without family-specific manual oversight."""

    def __init__(
        self,
        *,
        operators: FoldOperatorRegistry,
        evidence_provider: FoldEvidenceProvider,
        evaluator: FoldEvaluator,
        ledger: ExperimentLedger | None = None,
        initializer: FoldInitializer | None = None,
        final_evaluator: FoldEvaluator | None = None,
        stop_on_rejection: bool = True,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        self.operators = operators
        self.evidence_provider = evidence_provider
        self.evaluator = evaluator
        self.ledger = ExperimentLedger() if ledger is None else ledger
        self.initializer = initializer or NoOpInitializer()
        self.final_evaluator = final_evaluator
        self.stop_on_rejection = bool(stop_on_rejection)
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise DynamicFoldExecutionError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    def execute(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> DynamicFoldRun:
        models = (*context.parent_models.values(), context.child_model)
        parameter_rows: list[tuple[Any, bool]] = []
        seen: set[int] = set()
        for model in models:
            for parameter in model.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if parameter.grad is not None:
                    raise DynamicFoldExecutionError(
                        "Dynamic folding requires zero-gradient parents and child."
                    )
                parameter_rows.append((parameter, bool(parameter.requires_grad)))
        for parameter, _requires_grad in parameter_rows:
            parameter.requires_grad_(False)
        snapshot_scope = _WholeModelSnapshotScope(self.snapshot_config)
        parent_forward_observer = _ParentForwardObserver(context.parent_models)
        try:
            try:
                with parent_forward_observer:
                    with _FoldOnlyTorchGuard() as guard:
                        run = self._execute_impl(
                            plan,
                            context,
                            snapshot_scope,
                        )
            finally:
                for parameter, requires_grad in parameter_rows:
                    parameter.grad = None
                    parameter.requires_grad_(requires_grad)
        finally:
            snapshot_scope.cleanup()
        executed_implementation_manifest = [
            {
                "action_id": attempt.action_id,
                "method_id": attempt.method_id,
                "operator_prepare_implementation_fingerprint": (
                    attempt.operator_prepare_implementation_fingerprint
                ),
                "transaction_implementation_fingerprint": (
                    attempt.transaction_implementation_fingerprint
                ),
                "transaction_commit_implementation_fingerprint": (
                    attempt.transaction_commit_implementation_fingerprint
                ),
                "transaction_rollback_implementation_fingerprint": (
                    attempt.transaction_rollback_implementation_fingerprint
                ),
            }
            for attempt in run.attempts
        ]
        return replace(
            run,
            operator_implementation_fingerprint=(
                self.operators.implementation_fingerprint
            ),
            operator_static_zero_training_audit_passed=(
                self.operators.static_zero_training_audit_passed
            ),
            executed_fold_implementation_fingerprint=canonical_sha256(
                {
                    "schema": "neural-forge-executed-fold-implementations-v1",
                    "attempts": executed_implementation_manifest,
                }
            ),
            executed_fold_static_zero_training_audit_passed=(
                bool(run.attempts)
                and all(
                    attempt.operator_prepare_static_zero_training_audit_passed
                    and attempt.transaction_static_zero_training_audit_passed
                    for attempt in run.attempts
                )
            ),
            optimizer_constructions=guard.optimizer_constructions,
            optimizer_steps=guard.optimizer_steps,
            backward_calls=guard.backward_calls,
            autograd_grad_calls=guard.autograd_grad_calls,
            pruning_operations=guard.pruning_operations,
            quantization_operations=guard.quantization_operations,
            whole_model_snapshot_backend=self.snapshot_config.backend,
            observed_parent_module_forward_calls=(
                parent_forward_observer.module_forward_calls
            ),
            observed_parent_root_forward_calls=(
                parent_forward_observer.root_forward_calls
            ),
        )

    def _execute_impl(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
        snapshot_scope: _WholeModelSnapshotScope,
    ) -> DynamicFoldRun:
        if not isinstance(plan, DynamicFoldPlan):
            raise DynamicFoldExecutionError("plan must be DynamicFoldPlan.")
        if not plan.ready:
            raise DynamicFoldExecutionError(
                "The plan has structural gaps or is blocked by prior evidence."
            )
        if plan.child_graph_fingerprint != context.child_graph.fingerprint:
            raise DynamicFoldExecutionError(
                "Execution child graph does not match the plan."
            )
        if set(plan.parent_graph_fingerprints) != {
            value.fingerprint for value in context.parent_graphs.values()
        }:
            raise DynamicFoldExecutionError(
                "Execution parent graphs do not match the plan."
            )
        expected_parent_states = dict(plan.parent_state_fingerprints)
        if dict(context.parent_state_fingerprints) != expected_parent_states:
            raise DynamicFoldExecutionError(
                "A parent state changed after planning or does not match the "
                "evidence-bound checkpoint."
            )
        parent_snapshots = {
            name: snapshot_scope.capture(model)
            for name, model in context.parent_models.items()
        }

        def restore_parent_states() -> None:
            for name, snapshot in parent_snapshots.items():
                snapshot.restore(context.parent_models[name])

        def assert_parents_unchanged(stage: str) -> None:
            if dict(context.parent_state_fingerprints) != expected_parent_states:
                restore_parent_states()
                raise DynamicFoldExecutionError(
                    f"A parent state changed during {stage}."
                )

        initial_state = context.child_state_fingerprint
        if initial_state != plan.child_state_fingerprint:
            raise DynamicFoldExecutionError(
                "The child state changed after planning; replan before execution."
            )
        initial_snapshot = snapshot_scope.capture(context.child_model)
        initial_entries = fingerprint_model_state_entries(context.child_model)
        initial_runtime = _runtime_invariant_signature(context.child_model)
        try:
            bootstrap_report = dict(self.initializer.initialize(plan, context))
            assert_parents_unchanged("initializer execution")
        except BaseException:
            restore_parent_states()
            initial_snapshot.restore(context.child_model)
            raise
        try:
            bootstrap_state = context.child_state_fingerprint
            bootstrap_entries = fingerprint_model_state_entries(context.child_model)
            if _runtime_invariant_signature(context.child_model) != initial_runtime:
                raise DynamicFoldExecutionError(
                    "Initializer changed parameter/buffer identities, gradients, "
                    "requires_grad, or module modes."
                )
            if set(bootstrap_entries) != set(initial_entries):
                raise DynamicFoldExecutionError(
                    "Initializer changed the concrete child state topology."
                )
            bootstrap_changed = tuple(
                sorted(
                    name
                    for name in initial_entries
                    if initial_entries[name] != bootstrap_entries[name]
                )
            )
            unexpected_bootstrap = set(bootstrap_changed) - set(
                context.child_graph.parameter_paths
            )
            if unexpected_bootstrap:
                raise DynamicFoldExecutionError(
                    "Initializer mutated undeclared child state: "
                    + ", ".join(sorted(unexpected_bootstrap))
                )
            declared_bootstrap = bootstrap_report.get("child_state_fingerprint")
            if declared_bootstrap is not None and declared_bootstrap != bootstrap_state:
                raise DynamicFoldExecutionError(
                    "Initializer report does not bind the resulting child state."
                )
            if bootstrap_state != initial_state or bootstrap_changed:
                raise DynamicFoldExecutionError(
                    "Fold initializers must be non-mutating; learned state may "
                    "change only inside a planned fold transaction."
                )
        except BaseException:
            initial_snapshot.restore(context.child_model)
            raise
        bootstrap_report["child_state_fingerprint"] = bootstrap_state
        bootstrap_report["changed_state_paths"] = list(bootstrap_changed)
        bootstrap_report["mutation_scope_validated"] = True
        bootstrap_report["non_mutating_initializer_verified"] = True

        attempts: list[FoldAttempt] = []
        accepted: list[str] = []
        accepted_records: list[LedgerRecord] = []
        transaction_implementation_bindings: dict[
            int, _TransactionImplementationBinding
        ] = {}
        completed_paths: set[str] = set()
        status = "complete"
        declared_evidence_root = getattr(
            self.evidence_provider,
            "base_fingerprint",
            None,
        )
        if declared_evidence_root is not None and _digest(
            declared_evidence_root,
            "evidence provider base_fingerprint",
        ) != _digest(plan.evidence_fingerprint, "plan evidence fingerprint"):
            initial_snapshot.restore(context.child_model)
            raise DynamicFoldExecutionError(
                "Evidence provider base fingerprint differs from the plan root."
            )

        def rollback_whole_run(
            *,
            reason: str,
            metrics: tuple[tuple[str, JsonScalar], ...] = (),
            reject_last: bool,
        ) -> None:
            restore_parent_states()
            initial_snapshot.restore(context.child_model)
            for index, accepted_record in enumerate(accepted_records):
                is_last = index == len(accepted_records) - 1
                closure_status = (
                    RecordStatus.REJECTED
                    if reject_last and is_last
                    else RecordStatus.SUPERSEDED
                )
                closure = LedgerRecord(
                    signature=accepted_record.signature,
                    closure_level=ClosureLevel.EXACT,
                    status=closure_status,
                    reason_class=(
                        "final_behavior_gate_reject"
                        if closure_status is RecordStatus.REJECTED
                        else "global_rollback"
                    ),
                    reason=reason,
                    metrics=metrics,
                    previous_record_sha256=(
                        None
                        if not self.ledger.records
                        else self.ledger.records[-1].record_sha256
                    ),
                )
                self.ledger.append(closure)
            restore = getattr(self.evidence_provider, "restore", None)
            if not callable(restore):
                raise DynamicFoldExecutionError(
                    "A failed run requires evidence_provider.restore()."
                )
            restore_error: BaseException | None = None
            try:
                restore(context, initial_state, tuple(accepted))
            except BaseException as exc:
                restore_error = exc
            finally:
                restore_parent_states()
                initial_snapshot.restore(context.child_model)
            if (
                context.child_state_fingerprint != initial_state
                or fingerprint_model_state_entries(context.child_model)
                != initial_entries
                or _runtime_invariant_signature(context.child_model) != initial_runtime
                or dict(context.parent_state_fingerprints) != expected_parent_states
            ):
                raise DynamicFoldExecutionError(
                    "Whole-run restoration postconditions failed."
                )
            if restore_error is not None:
                raise restore_error
            accepted.clear()
            completed_paths.clear()

        def abort_transaction(
            transaction: FoldTransaction,
            *,
            reason: str,
        ) -> None:
            try:
                active = bool(getattr(transaction, "active", False))
            except BaseException:
                active = False
            rollback_authentication_error: BaseException | None = None
            transaction_binding = transaction_implementation_bindings.get(
                id(transaction)
            )
            if active and transaction_binding is not None:
                try:
                    rollback_callback = _reauthenticate_callable(
                        transaction,
                        "rollback",
                        transaction_binding.rollback,
                    )
                except BaseException as exc:
                    rollback_authentication_error = exc
                else:
                    try:
                        rollback_callback()
                    except BaseException:
                        pass
            rollback_whole_run(reason=reason, reject_last=False)
            if rollback_authentication_error is not None:
                raise DynamicFoldExecutionError(
                    "Transaction rollback authentication failed during abort."
                ) from rollback_authentication_error

        for action in plan.actions:
            if any(value not in accepted for value in action.dependencies):
                status = "dependency_blocked"
                break
            incumbent = context.child_state_fingerprint
            incumbent_entries = fingerprint_model_state_entries(context.child_model)
            incumbent_runtime = _runtime_invariant_signature(context.child_model)
            try:
                evidence = self.evidence_provider.prepare(action, context)
                assert_parents_unchanged("evidence preparation")
                prepared_state = context.child_state_fingerprint
                prepared_entries = fingerprint_model_state_entries(context.child_model)
                prepared_runtime = _runtime_invariant_signature(context.child_model)
            except BaseException:
                rollback_whole_run(
                    reason="Evidence preparation failed; run rolled back.",
                    reject_last=False,
                )
                raise
            if (
                prepared_state != incumbent
                or prepared_entries != incumbent_entries
                or prepared_runtime != incumbent_runtime
            ):
                rollback_whole_run(
                    reason="Evidence preparation mutated the child; run rolled back.",
                    reject_last=False,
                )
                raise DynamicFoldExecutionError(
                    "Evidence preparation mutated the incumbent child."
                )
            try:
                if not isinstance(evidence, FoldEvidence):
                    raise DynamicFoldExecutionError(
                        "Evidence provider must return FoldEvidence."
                    )
                if evidence.root_fingerprint != plan.evidence_fingerprint:
                    raise DynamicFoldExecutionError(
                        "Fold evidence is not rooted in the plan's certified "
                        "evidence fingerprint."
                    )
                try:
                    validate_fold_action_evidence_channels(
                        action,
                        evidence.channels,
                    )
                except ValueError as exc:
                    raise DynamicFoldExecutionError(
                        "Evidence manifest violates the planned method contract."
                    ) from exc
                _assert_fold_evidence_payload_unchanged(
                    evidence,
                    "Evidence binding",
                )
                evidence_derivation, evidence_derivation_fingerprint = (
                    _evidence_derivation_manifest(
                        evidence,
                        action,
                        incumbent,
                    )
                )
                signature = _current_signature(
                    action.signature,
                    child_state_fingerprint=incumbent,
                    evidence_fingerprint=evidence.fingerprint,
                )
                ledger_decision = self.ledger.decision(signature)
            except BaseException:
                rollback_whole_run(
                    reason="Evidence binding failed; run rolled back.",
                    reject_last=False,
                )
                raise
            if not ledger_decision.allowed:
                status = "known_attempt_blocked"
                break
            try:
                (
                    _operator,
                    prepare_callback,
                    operator_prepare_binding,
                ) = self.operators.authenticated_prepare(action.method_id)
                transaction = prepare_callback(action, context, evidence)
                assert_parents_unchanged("fold operator preparation")
                _assert_fold_evidence_payload_unchanged(
                    evidence,
                    "Fold operator",
                )
                transaction_binding = _bind_transaction_implementations(transaction)
                transaction_implementation_bindings[id(transaction)] = (
                    transaction_binding
                )
            except BaseException:
                rollback_whole_run(
                    reason="Fold operator preparation failed; run rolled back.",
                    reject_last=False,
                )
                raise
            try:
                transaction_method_id = transaction.method_id
            except BaseException:
                abort_transaction(
                    transaction,
                    reason="Fold transaction method lookup failed; run rolled back.",
                )
                raise
            if transaction_method_id != action.method_id:
                abort_transaction(
                    transaction,
                    reason="Fold transaction method mismatch; run rolled back.",
                )
                raise DynamicFoldExecutionError(
                    "Fold transaction method_id differs from the planned action."
                )
            try:
                candidate = context.child_state_fingerprint
                candidate_entries = fingerprint_model_state_entries(context.child_model)
                candidate_runtime = _runtime_invariant_signature(context.child_model)
            except BaseException:
                abort_transaction(
                    transaction,
                    reason="Candidate postcondition capture failed; run rolled back.",
                )
                raise
            if set(candidate_entries) != set(incumbent_entries):
                abort_transaction(
                    transaction,
                    reason="Fold operator changed state topology; run rolled back.",
                )
                raise DynamicFoldExecutionError(
                    "Fold operator changed the concrete child state topology."
                )
            changed_paths = tuple(
                sorted(
                    name
                    for name in incumbent_entries
                    if incumbent_entries[name] != candidate_entries[name]
                )
            )
            unexpected_paths = set(changed_paths) - set(action.target_parameter_paths)
            if unexpected_paths:
                abort_transaction(
                    transaction,
                    reason="Fold operator mutated undeclared state; run rolled back.",
                )
                raise DynamicFoldExecutionError(
                    "Fold operator mutated undeclared child state: "
                    + ", ".join(sorted(unexpected_paths))
                )
            verified_noop = bool(getattr(transaction, "verified_noop", False))
            if (
                action.target_parameter_paths
                and not changed_paths
                and not verified_noop
            ):
                abort_transaction(
                    transaction,
                    reason="Unverified no-op transaction; run rolled back.",
                )
                raise DynamicFoldExecutionError(
                    "A parameter-owning fold action installed no state change."
                )
            if candidate_runtime != incumbent_runtime:
                abort_transaction(
                    transaction,
                    reason="Fold operator changed runtime invariants; run rolled back.",
                )
                raise DynamicFoldExecutionError(
                    "Fold operator changed parameter/buffer identities, gradients, "
                    "requires_grad, or module modes."
                )
            try:
                declared_candidate = _digest(
                    transaction.candidate_state_fingerprint,
                    "transaction candidate_state_fingerprint",
                )
            except BaseException:
                abort_transaction(
                    transaction,
                    reason="Candidate fingerprint declaration failed; run rolled back.",
                )
                raise
            if candidate != declared_candidate:
                abort_transaction(
                    transaction,
                    reason="Fold transaction fingerprint mismatch; run rolled back.",
                )
                raise DynamicFoldExecutionError(
                    "Installed transaction does not match its candidate fingerprint."
                )
            evaluation_context = FoldEvaluationContext(
                action=action,
                execution_signature=signature,
                execution_context=context,
                evidence=evidence,
                transaction=transaction,
                incumbent_state_fingerprint=incumbent,
                candidate_state_fingerprint=candidate,
            )
            try:
                decision = _decision(self.evaluator.evaluate(evaluation_context))
                assert_parents_unchanged("transaction evaluation")
                _assert_fold_evidence_payload_unchanged(
                    evidence,
                    "Fold evaluator",
                )
            except BaseException:
                abort_transaction(
                    transaction,
                    reason="Transaction evaluator failed; run rolled back.",
                )
                raise
            try:
                evaluated_state = context.child_state_fingerprint
                evaluated_entries = fingerprint_model_state_entries(context.child_model)
                evaluated_runtime = _runtime_invariant_signature(context.child_model)
            except BaseException:
                abort_transaction(
                    transaction,
                    reason="Evaluator postcondition capture failed; run rolled back.",
                )
                raise
            if (
                evaluated_state != candidate
                or evaluated_entries != candidate_entries
                or evaluated_runtime != candidate_runtime
            ):
                abort_transaction(
                    transaction,
                    reason="Transaction evaluator mutated the child; run rolled back.",
                )
                raise DynamicFoldExecutionError(
                    "Fold evaluator mutated the installed candidate."
                )
            if decision.accepted:
                try:
                    commit_callback = _reauthenticate_callable(
                        transaction,
                        "commit",
                        transaction_binding.commit,
                    )
                    transaction_report = dict(commit_callback())
                    assert_parents_unchanged("transaction commit")
                    _assert_fold_evidence_payload_unchanged(
                        evidence,
                        "Fold transaction commit",
                    )
                except BaseException:
                    abort_transaction(
                        transaction,
                        reason="Transaction commit failed; run rolled back.",
                    )
                    raise
                transaction_report["changed_state_paths"] = list(changed_paths)
                transaction_report["mutation_scope_validated"] = True
                transaction_report["verified_noop"] = verified_noop
                rollback_exact = None
                try:
                    committed_state = context.child_state_fingerprint
                    committed_entries = fingerprint_model_state_entries(
                        context.child_model
                    )
                    committed_runtime = _runtime_invariant_signature(
                        context.child_model
                    )
                except BaseException:
                    rollback_whole_run(
                        reason="Commit postcondition capture failed; run rolled back.",
                        reject_last=False,
                    )
                    raise
                if (
                    committed_state != candidate
                    or committed_entries != candidate_entries
                    or committed_runtime != candidate_runtime
                ):
                    rollback_whole_run(
                        reason="Committed candidate changed unexpectedly; run rolled back.",
                        reject_last=False,
                    )
                    raise DynamicFoldExecutionError(
                        "Committed transaction changed after evaluation."
                    )
                accepted.append(action.action_id)
                completed_paths.update(action.target_parameter_paths)
                record_status = RecordStatus.ACCEPTED
            else:
                try:
                    rollback_callback = _reauthenticate_callable(
                        transaction,
                        "rollback",
                        transaction_binding.rollback,
                    )
                    transaction_report = dict(rollback_callback())
                    assert_parents_unchanged("transaction rollback")
                    _assert_fold_evidence_payload_unchanged(
                        evidence,
                        "Fold transaction rollback",
                    )
                except BaseException:
                    rollback_whole_run(
                        reason="Transaction rollback failed; run restored globally.",
                        reject_last=False,
                    )
                    raise
                transaction_report["changed_state_paths"] = list(changed_paths)
                transaction_report["mutation_scope_validated"] = True
                transaction_report["verified_noop"] = verified_noop
                rollback_exact = bool(transaction_report.get("rollback_exact", False))
                try:
                    rolled_state = context.child_state_fingerprint
                    rolled_entries = fingerprint_model_state_entries(
                        context.child_model
                    )
                    rolled_runtime = _runtime_invariant_signature(context.child_model)
                except BaseException:
                    rollback_whole_run(
                        reason="Rollback postcondition capture failed.",
                        reject_last=False,
                    )
                    raise
                if (
                    not rollback_exact
                    or rolled_state != incumbent
                    or rolled_entries != incumbent_entries
                    or rolled_runtime != incumbent_runtime
                ):
                    rollback_whole_run(
                        reason="Rejected transaction rollback was inexact.",
                        reject_last=False,
                    )
                    raise DynamicFoldExecutionError(
                        "Rejected transaction did not restore the exact incumbent."
                    )
                record_status = RecordStatus.REJECTED
            record = LedgerRecord(
                signature=signature,
                closure_level=ClosureLevel.EXACT,
                status=record_status,
                reason_class=(
                    "behavior_gate_accept"
                    if decision.accepted
                    else "behavior_gate_reject"
                ),
                reason=decision.reason or record_status.value,
                metrics=decision.metrics,
                previous_record_sha256=(
                    None
                    if not self.ledger.records
                    else self.ledger.records[-1].record_sha256
                ),
            )
            try:
                self.ledger.append(record)
            except BaseException:
                rollback_whole_run(
                    reason="Experiment ledger append failed; run rolled back.",
                    reject_last=False,
                )
                raise
            if decision.accepted:
                accepted_records.append(record)
            attempts.append(
                FoldAttempt(
                    action_id=action.action_id,
                    method_id=action.method_id,
                    execution_signature=signature,
                    evidence_fingerprint=evidence.fingerprint,
                    evidence_root_fingerprint=evidence.root_fingerprint,
                    evidence_payload_fingerprint=evidence.payload_fingerprint,
                    evidence_payload_integrity_mode=(evidence.payload_integrity_mode),
                    evidence_channels=evidence.channels,
                    evidence_binding_fingerprint=evidence.binding_fingerprint,
                    evidence_derivation_manifest=evidence_derivation,
                    evidence_derivation_fingerprint=(evidence_derivation_fingerprint),
                    evidence_payload_integrity_reverified=True,
                    operator_prepare_implementation_fingerprint=(
                        operator_prepare_binding.fingerprint
                    ),
                    operator_prepare_static_zero_training_audit_passed=(
                        operator_prepare_binding.static_zero_training_audit_passed
                    ),
                    transaction_implementation_fingerprint=(
                        transaction_binding.fingerprint
                    ),
                    transaction_commit_implementation_fingerprint=(
                        transaction_binding.commit.fingerprint
                    ),
                    transaction_rollback_implementation_fingerprint=(
                        transaction_binding.rollback.fingerprint
                    ),
                    transaction_static_zero_training_audit_passed=(
                        transaction_binding.static_zero_training_audit_passed
                    ),
                    incumbent_state_fingerprint=incumbent,
                    candidate_state_fingerprint=candidate,
                    decision=decision,
                    transaction_report=MappingProxyType(transaction_report),
                    rollback_exact=rollback_exact,
                    ledger_record_sha256=record.record_sha256,
                )
            )
            if decision.accepted:
                try:
                    self.evidence_provider.refresh(
                        action, context, MappingProxyType(transaction_report)
                    )
                    assert_parents_unchanged("accepted evidence refresh")
                    _assert_fold_evidence_payload_unchanged(
                        evidence,
                        "Accepted evidence refresh",
                    )
                    if (
                        context.child_state_fingerprint != candidate
                        or fingerprint_model_state_entries(context.child_model)
                        != candidate_entries
                        or _runtime_invariant_signature(context.child_model)
                        != candidate_runtime
                    ):
                        raise DynamicFoldExecutionError(
                            "Accepted evidence refresh mutated the child."
                        )
                except BaseException:
                    rollback_whole_run(
                        reason="Accepted-only evidence refresh failed; run rolled back.",
                        reject_last=False,
                    )
                    raise
            elif self.stop_on_rejection:
                status = "rejected"
                break

        final_decision = None
        expected_paths = set(context.child_graph.parameter_paths)
        if status == "complete" and completed_paths != expected_paths:
            status = "incomplete_target_coverage"
        if status == "complete" and not any(
            attempt.accepted
            and attempt.candidate_state_fingerprint
            != attempt.incumbent_state_fingerprint
            for attempt in attempts
        ):
            status = "no_effectful_fold_action"

        if status != "complete":
            rollback_whole_run(
                reason=f"Run ended before final acceptance: {status}.",
                reject_last=False,
            )

        final_evidence_fingerprint = None
        final_evidence_root_fingerprint = None
        final_evidence_payload_fingerprint = None
        final_evidence_payload_integrity_mode = None
        final_evidence_channels: tuple[str, ...] = ()
        final_evidence_binding_fingerprint = None
        final_evidence_derivation = None
        final_evidence_derivation_fingerprint = None
        final_evidence_payload_integrity_reverified = False
        final_child_module_forward_calls = 0
        final_child_root_forward_calls = 0
        final_child_inference_entrypoint_calls = 0
        if status == "complete" and self.final_evaluator is not None:
            synthetic_action = plan.actions[-1] if plan.actions else None
            if synthetic_action is None:
                rollback_whole_run(
                    reason="Final evaluator had no fold action context.",
                    reject_last=False,
                )
                raise DynamicFoldExecutionError(
                    "A final evaluator requires at least one fold action."
                )
            state = context.child_state_fingerprint
            final_entries = fingerprint_model_state_entries(context.child_model)
            final_runtime = _runtime_invariant_signature(context.child_model)
            try:
                evidence = self.evidence_provider.prepare(synthetic_action, context)
                assert_parents_unchanged("final evidence preparation")
                final_prepared_state = context.child_state_fingerprint
                final_prepared_entries = fingerprint_model_state_entries(
                    context.child_model
                )
                final_prepared_runtime = _runtime_invariant_signature(
                    context.child_model
                )
            except BaseException:
                rollback_whole_run(
                    reason="Final evidence preparation failed; run rolled back.",
                    reject_last=False,
                )
                raise
            if (
                final_prepared_state != state
                or final_prepared_entries != final_entries
                or final_prepared_runtime != final_runtime
            ):
                rollback_whole_run(
                    reason="Final evidence preparation mutated the child.",
                    reject_last=False,
                )
                raise DynamicFoldExecutionError(
                    "Final evidence preparation mutated the child."
                )
            try:
                if not isinstance(evidence, FoldEvidence):
                    raise DynamicFoldExecutionError(
                        "Final evidence provider must return FoldEvidence."
                    )
                if evidence.root_fingerprint != plan.evidence_fingerprint:
                    raise DynamicFoldExecutionError(
                        "Final fold evidence is not rooted in the plan's "
                        "certified evidence fingerprint."
                    )
                final_evidence_fingerprint = evidence.fingerprint
                final_evidence_root_fingerprint = evidence.root_fingerprint
                final_evidence_payload_fingerprint = evidence.payload_fingerprint
                final_evidence_payload_integrity_mode = evidence.payload_integrity_mode
                final_evidence_channels = evidence.channels
                final_evidence_binding_fingerprint = evidence.binding_fingerprint
                try:
                    validate_fold_action_evidence_channels(
                        synthetic_action,
                        evidence.channels,
                    )
                except ValueError as exc:
                    raise DynamicFoldExecutionError(
                        "Final evidence manifest violates the planned method contract."
                    ) from exc
                _assert_fold_evidence_payload_unchanged(
                    evidence,
                    "Final evidence binding",
                )
                (
                    final_evidence_derivation,
                    final_evidence_derivation_fingerprint,
                ) = _evidence_derivation_manifest(
                    evidence,
                    synthetic_action,
                    state,
                )
                final_signature = _current_signature(
                    synthetic_action.signature,
                    child_state_fingerprint=state,
                    evidence_fingerprint=evidence.fingerprint,
                )
            except BaseException:
                rollback_whole_run(
                    reason="Final evidence binding failed; run rolled back.",
                    reject_last=False,
                )
                raise

            class _FinalTransaction:
                method_id = "final_evaluation"
                candidate_state_fingerprint = state
                active = False

                @staticmethod
                def commit() -> Mapping[str, Any]:
                    raise DynamicFoldExecutionError(
                        "Final evaluation transaction cannot commit."
                    )

                @staticmethod
                def rollback() -> Mapping[str, Any]:
                    raise DynamicFoldExecutionError(
                        "Final evaluation transaction cannot roll back."
                    )

            final_context = FoldEvaluationContext(
                action=synthetic_action,
                execution_signature=final_signature,
                execution_context=context,
                evidence=evidence,
                transaction=_FinalTransaction(),
                incumbent_state_fingerprint=state,
                candidate_state_fingerprint=state,
            )
            try:
                final_execution_contract = (
                    resolve_behavior_inference_execution_contract(
                        self.final_evaluator
                    )
                )
                final_observer = _ParentForwardObserver(
                    {"child": context.child_model},
                    execution_contracts={"child": final_execution_contract},
                )
                with final_observer:
                    final_decision = _decision(
                        self.final_evaluator.evaluate(final_context)
                    )
                final_child_module_forward_calls = final_observer.module_forward_calls
                final_child_root_forward_calls = final_observer.root_forward_calls
                final_child_inference_entrypoint_calls = (
                    final_observer.inference_entrypoint_calls
                )
                assert_parents_unchanged("final evaluation")
                _assert_fold_evidence_payload_unchanged(
                    evidence,
                    "Final fold evaluator",
                )
                final_evidence_payload_integrity_reverified = True
                final_evaluated_state = context.child_state_fingerprint
                final_evaluated_entries = fingerprint_model_state_entries(
                    context.child_model
                )
                final_evaluated_runtime = _runtime_invariant_signature(
                    context.child_model
                )
            except BaseException:
                rollback_whole_run(
                    reason="Final evaluator failed; all provisional commits rolled back.",
                    reject_last=False,
                )
                raise
            if (
                final_evaluated_state != state
                or final_evaluated_entries != final_entries
                or final_evaluated_runtime != final_runtime
            ):
                rollback_whole_run(
                    reason="Final evaluator mutated the child; run rolled back.",
                    reject_last=False,
                )
                raise DynamicFoldExecutionError(
                    "Final fold evaluator mutated the child."
                )
            if not final_decision.accepted:
                rollback_whole_run(
                    reason=final_decision.reason or "final behavior gate rejected",
                    metrics=final_decision.metrics,
                    reject_last=True,
                )
                status = "final_gate_rejected"
        return DynamicFoldRun(
            plan_fingerprint=plan.fingerprint,
            status=status,
            initial_state_fingerprint=initial_state,
            final_state_fingerprint=context.child_state_fingerprint,
            bootstrap_report=MappingProxyType(bootstrap_report),
            attempts=tuple(attempts),
            accepted_action_ids=tuple(accepted),
            completed_target_parameter_paths=tuple(sorted(completed_paths)),
            final_decision=final_decision,
            final_evidence_fingerprint=final_evidence_fingerprint,
            final_evidence_root_fingerprint=final_evidence_root_fingerprint,
            final_evidence_payload_fingerprint=final_evidence_payload_fingerprint,
            final_evidence_payload_integrity_mode=(
                final_evidence_payload_integrity_mode
            ),
            final_evidence_channels=final_evidence_channels,
            final_evidence_binding_fingerprint=final_evidence_binding_fingerprint,
            final_evidence_derivation_manifest=final_evidence_derivation,
            final_evidence_derivation_fingerprint=(
                final_evidence_derivation_fingerprint
            ),
            final_evidence_payload_integrity_reverified=(
                final_evidence_payload_integrity_reverified
            ),
            ledger_fingerprint=self.ledger.fingerprint,
            observed_final_child_module_forward_calls=(
                final_child_module_forward_calls
            ),
            observed_final_child_root_forward_calls=(final_child_root_forward_calls),
            observed_final_child_inference_entrypoint_calls=(
                final_child_inference_entrypoint_calls
            ),
        )


@dataclass(frozen=True, slots=True)
class _ParameterSnapshot:
    path: str
    parameter: Any
    value: Any
    grad: Any | None
    grad_value: Any | None
    requires_grad: bool


class ExactCopyTransaction:
    method_id = "copy_exact"

    def __init__(
        self,
        context: FoldExecutionContext,
        snapshots: tuple[_ParameterSnapshot, ...],
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
    ) -> None:
        self.context = context
        self.snapshots = snapshots
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = candidate_state_fingerprint
        self.verified_noop = incumbent_state_fingerprint == candidate_state_fingerprint
        self.active = True

    def _validate_runtime_identity(self) -> None:
        for snapshot in self.snapshots:
            current = self.context.child_model.get_parameter(snapshot.path)
            if current is not snapshot.parameter:
                raise DynamicFoldExecutionError(
                    "Exact-copy transaction changed a Parameter object."
                )
            if bool(current.requires_grad) != snapshot.requires_grad:
                raise DynamicFoldExecutionError(
                    "Exact-copy transaction changed requires_grad."
                )
            if snapshot.grad is None:
                if current.grad is not None:
                    raise DynamicFoldExecutionError(
                        "Exact-copy transaction created a gradient."
                    )
            elif current.grad is not snapshot.grad:
                raise DynamicFoldExecutionError(
                    "Exact-copy transaction replaced a gradient object."
                )

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicFoldExecutionError("Exact-copy transaction is inactive.")
        self._validate_runtime_identity()
        if self.context.child_state_fingerprint != self.candidate_state_fingerprint:
            raise DynamicFoldExecutionError(
                "Exact-copy candidate changed before commit."
            )
        self.active = False
        return {
            "accepted": True,
            "rollback_exact": None,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "parameter_identities_preserved": True,
            "gradient_state_preserved": True,
            "zero_gradient": True,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicFoldExecutionError("Exact-copy transaction is inactive.")
        import torch

        self._validate_runtime_identity()
        with torch.no_grad():
            for snapshot in self.snapshots:
                snapshot.parameter.copy_(
                    snapshot.value.to(
                        device=snapshot.parameter.device,
                        dtype=snapshot.parameter.dtype,
                    )
                )
                if snapshot.grad is not None and snapshot.grad_value is not None:
                    snapshot.grad.copy_(
                        snapshot.grad_value.to(
                            device=snapshot.grad.device,
                            dtype=snapshot.grad.dtype,
                        )
                    )
        self._validate_runtime_identity()
        rollback_exact = (
            self.context.child_state_fingerprint == self.incumbent_state_fingerprint
        )
        if not rollback_exact:
            raise DynamicFoldExecutionError(
                "Exact-copy rollback did not restore the incumbent."
            )
        self.active = False
        return {
            "accepted": False,
            "rollback_exact": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": self.incumbent_state_fingerprint,
            "parameter_identities_preserved": True,
            "gradient_state_preserved": True,
            "zero_gradient": True,
        }


class ExactCopyOperator:
    """Executable identity/control operator for equal semantic tensor roles."""

    method_id = "copy_exact"

    @staticmethod
    def _node_mapping(
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> tuple[tuple[Any, Any, Any], ...]:
        if len(action.sources) != 1:
            raise DynamicFoldExecutionError(
                "Exact copy accepts exactly one parent source."
            )
        source_ref = action.sources[0]
        parent_graph = context.parent_graphs[source_ref.parent_name]
        source_nodes = {
            node_id: parent_graph.node(node_id) for node_id in source_ref.node_ids
        }
        target_nodes = {
            node_id: context.child_graph.node(node_id)
            for node_id in action.target_node_ids
        }
        if set(source_nodes) != set(target_nodes):
            raise DynamicFoldExecutionError(
                "Exact copy requires stable matching source/target circuit IDs."
            )
        result = []
        for node_id in sorted(target_nodes):
            source_node = source_nodes[node_id]
            target_node = target_nodes[node_id]
            source_parameters = {value.role: value for value in source_node.parameters}
            target_parameters = {value.role: value for value in target_node.parameters}
            if set(source_parameters) != set(target_parameters):
                raise DynamicFoldExecutionError(
                    f"Exact-copy roles differ for node {node_id!r}."
                )
            for role in sorted(target_parameters):
                source = source_parameters[role]
                target = target_parameters[role]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise DynamicFoldExecutionError(
                        f"Exact-copy tensor contract differs for {node_id!r}:{role}."
                    )
                result.append((source_ref.parent_name, source, target))
        return tuple(result)

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> ExactCopyTransaction:
        del evidence
        import torch

        mapping = self._node_mapping(action, context)
        incumbent = context.child_state_fingerprint
        snapshots = []
        grouped: dict[int, list[tuple[Any, Any, Any, Any, Any]]] = {}
        for parent_name, source_spec, target_spec in mapping:
            source = context.parent_models[parent_name].get_parameter(source_spec.path)
            target = context.child_model.get_parameter(target_spec.path)
            grouped.setdefault(id(target), []).append(
                (parent_name, source_spec, target_spec, source, target)
            )
        with torch.no_grad():
            for rows in grouped.values():
                target_objects = {id(row[4]) for row in rows}
                target_storage_groups = {row[2].storage_group for row in rows}
                source_objects = {id(row[3]) for row in rows}
                source_storage_groups = {row[1].storage_group for row in rows}
                if len(target_objects) != 1 or len(target_storage_groups) != 1:
                    raise DynamicFoldExecutionError(
                        "Target storage aliases do not resolve to one Parameter."
                    )
                if len(source_objects) != 1 or len(source_storage_groups) != 1:
                    raise DynamicFoldExecutionError(
                        "A tied target storage group maps to non-aliased parent "
                        "parameters."
                    )
                canonical = min(rows, key=lambda row: row[2].path)
                _parent_name, _source_spec, target_spec, source, target = canonical
                snapshots.append(
                    _ParameterSnapshot(
                        path=target_spec.path,
                        parameter=target,
                        value=target.detach().to(device="cpu").contiguous().clone(),
                        grad=target.grad,
                        grad_value=(
                            None
                            if target.grad is None
                            else target.grad.detach()
                            .to(device="cpu")
                            .contiguous()
                            .clone()
                        ),
                        requires_grad=bool(target.requires_grad),
                    )
                )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        candidate = context.child_state_fingerprint
        transaction = ExactCopyTransaction(
            context,
            tuple(snapshots),
            incumbent,
            candidate,
        )
        transaction._validate_runtime_identity()
        return transaction


__all__ = [
    "DynamicFoldExecutionError",
    "DynamicFoldExecutor",
    "DynamicFoldRun",
    "ExactCopyOperator",
    "ExactCopyTransaction",
    "FoldDecision",
    "FoldEvaluationContext",
    "FoldEvidence",
    "FoldEvidenceProvider",
    "FoldExecutionContext",
    "FoldInitializer",
    "FoldOnlyGuardReport",
    "FoldOperator",
    "FoldOperatorRegistry",
    "FoldTransaction",
    "NoOpInitializer",
    "RUN_SCHEMA",
    "StaticEvidenceProvider",
    "fingerprint_fold_evidence_payload",
    "fingerprint_model_runtime_state",
    "fingerprint_model_state",
    "run_fold_only_callback",
]
