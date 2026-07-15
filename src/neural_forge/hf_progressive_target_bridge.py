"""Exact-budget progressive targets and target-native QKVO publication.

This module connects three already-existing Neural Forge boundaries:

* an explicit, target-sized dense GQA child;
* the exact-size blank-child parent compiler; and
* temporary output-trained Q/K/V/O residual factors that can be baked into
  ordinary target ``nn.Linear`` weights.

The bridge does not train factors and does not score behavior.  It requires an
external behavior-gate receipt before publication, then proves the structural
facts it can establish locally: the adapter registry is gone, every Q/K/V/O
projection is native dense, the state-key topology is restored, and the unique
parameter count is exactly the requested child budget.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import PurePath
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodType,
    ModuleType,
)
from typing import Any, Literal
from weakref import ReferenceType

from torch import Tensor, nn, no_grad, strided, uint8

from neural_forge.hf_arch_layout import (
    DecoderLayout,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAAdapterOnlyOutputTrainingScope,
    DenseGQAGeometry,
    DenseGQAQKVOResidualProjection,
    DenseGQAQKVOResidualTopologyError,
    DenseGQASite,
    InstalledDenseGQAQKVOResidualAdapter,
    SparseDenseGQAQKVOResidualRegistry,
    install_dense_gqa_qkvo_residual_adapter,
)
from neural_forge.hf_dense_lexical_residual_adapter import (
    DenseLexicalResidualError,
    InstalledDenseLexicalResidualAdapter,
    install_dense_lexical_residual_adapter,
)
from neural_forge.hf_factorized_vocab import exact_parameter_report
from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    DenseSwiGLUProjectionResidualTopologyError,
    DenseSwiGLUResidualProjection,
    InstalledDenseSwiGLUProjectionResidualAdapter,
    SparseDenseSwiGLUProjectionResidualRegistry,
    install_dense_swiglu_projection_residual_adapter,
)
from neural_forge.hf_parent_compiler import (
    HFParentCompilerResult,
    fingerprint_hf_child_structure,
)


PROGRESSIVE_TARGET_PLAN_SCHEMA = "neural-forge-progressive-dense-gqa-plan-v1"
NATIVE_TARGET_CERTIFICATE_SCHEMA = "neural-forge-native-dense-gqa-certificate-v1"
EXACT_TARGET_QKVO_INSTALLATION_SCHEMA = "neural-forge-exact-target-qkvo-installation-v1"
EXACT_TARGET_QKVO_PUBLICATION_SCHEMA = "neural-forge-exact-target-qkvo-publication-v1"
EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA = (
    "neural-forge-exact-target-compiler-transaction-installation-v1"
)
EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA = (
    "neural-forge-exact-target-compiler-transaction-publication-v1"
)
EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA = (
    "neural-forge-exact-target-experimental-recovery-materialization-v1"
)
EXPERIMENTAL_RECOVERY_MATERIALIZATION_RECEIPT_SCHEMA = (
    "neural-forge-experimental-recovery-materialization-receipt-v1"
)
EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA = (
    "neural-forge-experimental-recovery-contract-v1"
)
EXACT_TARGET_COMPILER_OUTPUT_TRAINING_SCOPE_SCHEMA = (
    "neural-forge-exact-target-compiler-output-training-scope-v1"
)
EXTERNAL_BEHAVIOR_GATE_SCHEMA = "neural-forge-external-behavior-gate-receipt-v1"

QWEN3_4B_EXACT_PARAMETERS = 4_022_468_096
QWEN3_1P4B_EXACT_PARAMETERS = 1_410_567_168
QWEN3_600M_EXACT_PARAMETERS = 600_628_224
SMOLLM2_135M_EXACT_PARAMETERS = 134_515_008

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProgressiveTargetBridgeError(ValueError):
    """The exact target or publication contract is invalid."""


class ProgressiveTargetPublicationError(ProgressiveTargetBridgeError):
    """A temporary QKVO correction cannot be published safely."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProgressiveTargetBridgeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProgressiveTargetBridgeError(f"{name} must be a nonnegative integer.")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgressiveTargetBridgeError(f"{name} must be bool.")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ProgressiveTargetBridgeError(f"{name} must be a SHA-256 string.")
    normalized = value.removeprefix("sha256:").lower()
    if not _HEX_SHA256.fullmatch(normalized):
        raise ProgressiveTargetBridgeError(f"{name} must be a SHA-256 string.")
    return normalized


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ProgressiveTargetBridgeError(
            "Target provenance must be finite canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_report_bytes(
    value: Mapping[str, Any],
    *,
    hash_field: str,
) -> bytes:
    """Return built-in immutable canonical bytes after self-hash verification."""

    payload = dict(value)
    expected = _sha256(payload.get(hash_field), hash_field)
    preimage = {key: item for key, item in payload.items() if key != hash_field}
    if _canonical_sha256(preimage) != expected:
        raise ProgressiveTargetPublicationError(
            f"Canonical report {hash_field} does not authenticate."
        )
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        decoded = json.loads(canonical)
    except (TypeError, ValueError) as error:
        raise ProgressiveTargetPublicationError(
            "Canonical report is not finite JSON."
        ) from error
    if decoded != payload:
        raise ProgressiveTargetPublicationError(
            "Canonical report JSON round-trip changed its payload."
        )
    decoded_preimage = {
        key: item for key, item in decoded.items() if key != hash_field
    }
    if _canonical_sha256(decoded_preimage) != expected:
        raise ProgressiveTargetPublicationError(
            "Canonical report bytes do not preserve their transaction hash."
        )
    return canonical


def _dense_tensor_sha256(value: Tensor) -> str:
    """Hash one dense tensor in bounded host-memory chunks."""

    if value.layout != strided or value.device.type == "meta":
        raise ProgressiveTargetBridgeError(
            "Dense tensor hashing requires a dense non-meta tensor."
        )
    detached = value.detach()
    if not detached.is_contiguous():
        detached = detached.contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    flat = detached.reshape(-1)
    elements_per_chunk = max(
        1,
        (16 * 1024 * 1024) // max(1, detached.element_size()),
    )
    for start in range(0, int(flat.numel()), elements_per_chunk):
        staged = flat[start : start + elements_per_chunk].to(
            device="cpu"
        ).contiguous()
        digest.update(memoryview(staged.view(uint8).numpy()))
    return digest.hexdigest()


def _dense_model_state_sha256(model: nn.Module) -> str:
    """Hash every directly registered tensor, alias, and byte."""

    digest = hashlib.sha256()
    rows = _registered_model_tensors(model)
    aliases: dict[int, int] = {}
    for kind, name, value, persistent in rows:
        if value.layout != strided or value.device.type == "meta":
            raise ProgressiveTargetBridgeError(
                "Dense model-state hashing requires dense non-meta tensors."
            )
        alias_index = aliases.setdefault(id(value), len(aliases))
        header = json.dumps(
            {
                "alias_index": alias_index,
                "kind": kind,
                "name": name,
                "dtype": str(value.dtype),
                "persistent": persistent,
                "shape": list(value.shape),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(bytes.fromhex(_dense_tensor_sha256(value)))
    if not rows:
        raise ProgressiveTargetBridgeError("Dense model-state hashing found no state.")
    return digest.hexdigest()


def _registered_model_tensors(
    model: nn.Module,
) -> tuple[tuple[str, str, Tensor, bool | None], ...]:
    """Enumerate raw recursive registries without invoking state-dict hooks."""

    values: list[tuple[str, str, Tensor, bool | None]] = []
    active_modules: set[int] = set()

    def visit(module: nn.Module, prefix: str) -> None:
        module_id = id(module)
        if module_id in active_modules:
            raise ProgressiveTargetBridgeError(
                "Registered module graph contains a recursive ownership cycle."
            )
        active_modules.add(module_id)
        try:
            for local_name, parameter in module._parameters.items():
                if parameter is None:
                    continue
                if not isinstance(parameter, nn.Parameter):
                    raise ProgressiveTargetBridgeError(
                        "Registered parameter entry is not an nn.Parameter."
                    )
                name = f"{prefix}.{local_name}" if prefix else str(local_name)
                values.append(("parameter", name, parameter, None))
            for local_name, buffer in module._buffers.items():
                if buffer is None:
                    continue
                if not isinstance(buffer, Tensor):
                    raise ProgressiveTargetBridgeError(
                        "Registered buffer entry is not a Tensor."
                    )
                name = f"{prefix}.{local_name}" if prefix else str(local_name)
                values.append(
                    (
                        "buffer",
                        name,
                        buffer,
                        local_name not in module._non_persistent_buffers_set,
                    )
                )
            for local_name, child in module._modules.items():
                if child is None:
                    continue
                if not isinstance(child, nn.Module):
                    raise ProgressiveTargetBridgeError(
                        "Registered child entry is not an nn.Module."
                    )
                child_prefix = (
                    f"{prefix}.{local_name}" if prefix else str(local_name)
                )
                visit(child, child_prefix)
        finally:
            active_modules.remove(module_id)

    visit(model, "")
    values.sort(key=lambda row: (row[0], row[1]))
    return tuple(values)


def _named_model_tensors(
    model: nn.Module,
) -> tuple[tuple[str, str, Tensor], ...]:
    return tuple(
        (kind, name, value)
        for kind, name, value, _persistent in _registered_model_tensors(model)
    )


@dataclass(frozen=True, slots=True)
class _RegisteredTensorBytesSnapshot:
    tensor_bytes: Tensor
    persistent: bool | None
    alias_index: int


def _snapshot_named_model_tensors(
    model: nn.Module,
) -> dict[tuple[str, str], _RegisteredTensorBytesSnapshot]:
    """Clone raw registered tensors without invoking serialization hooks."""

    snapshot: dict[tuple[str, str], _RegisteredTensorBytesSnapshot] = {}
    aliases: dict[int, int] = {}
    for kind, name, value, persistent in _registered_model_tensors(model):
        if value.device.type == "meta":
            raise ProgressiveTargetBridgeError(
                "Experimental materialization cannot snapshot meta tensors."
            )
        snapshot[(kind, name)] = _RegisteredTensorBytesSnapshot(
            tensor_bytes=value.detach().to(device="cpu").clone(),
            persistent=persistent,
            alias_index=aliases.setdefault(id(value), len(aliases)),
        )
    if not snapshot:
        raise ProgressiveTargetBridgeError(
            "Experimental materialization found no registered model tensors."
        )
    return snapshot


def _restore_named_model_tensors(
    model: nn.Module,
    snapshot: Mapping[tuple[str, str], _RegisteredTensorBytesSnapshot],
) -> None:
    """Restore matching raw registry tensors from the pre-bake snapshot."""

    source_alias_owners: dict[int, int] = {}
    current_alias_sources: dict[int, int] = {}
    with no_grad():
        for kind, name, value, persistent in _registered_model_tensors(model):
            source = snapshot.get((kind, name))
            # Adapter-owned projection names differ while wrapped. Their own
            # authenticated snapshots restore them before this pass. The final
            # direct-registry digest below remains authoritative.
            if source is None:
                continue
            source_bytes = source.tensor_bytes
            current_id = id(value)
            expected_current_id = source_alias_owners.setdefault(
                source.alias_index,
                current_id,
            )
            expected_source_alias = current_alias_sources.setdefault(
                current_id,
                source.alias_index,
            )
            if (
                tuple(source_bytes.shape) != tuple(value.shape)
                or source_bytes.dtype != value.dtype
                or persistent != source.persistent
                or expected_current_id != current_id
                or expected_source_alias != source.alias_index
            ):
                raise ProgressiveTargetPublicationError(
                    "Experimental cleanup registered tensor metadata or aliasing "
                    f"changed at {kind}:{name}."
                )
            value.copy_(source_bytes.to(device=value.device))


def _named_tensor_version_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (kind, name, id(value), int(value._version))
        for kind, name, value in _named_model_tensors(model)
    )


def _capture_native_tensor_guards(
    model: nn.Module,
) -> tuple[tuple[str, str, Tensor, str], ...]:
    """Retain lightweight exact-byte guards over pre-install native tensors."""

    return tuple(
        (kind, name, value, _dense_tensor_sha256(value))
        for kind, name, value in _named_model_tensors(model)
    )


_HOOK_REGISTRY_NAMES = (
    "_backward_pre_hooks",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)
_CONTROLLED_MODULE_ATTRIBUTES = {
    "training",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_modules",
    *_HOOK_REGISTRY_NAMES,
}


@dataclass(frozen=True, slots=True)
class _HookRegistrySnapshot:
    name: str
    existed: bool
    value: Any
    items: tuple[tuple[Any, Any], ...] | None


@dataclass(frozen=True, slots=True)
class _ModuleRuntimeSnapshot:
    module: nn.Module
    modules_registry: dict[str, nn.Module | None]
    module_items: tuple[tuple[str, nn.Module | None], ...]
    parameters_registry: dict[str, nn.Parameter | None]
    parameter_items: tuple[tuple[str, nn.Parameter | None], ...]
    buffers_registry: dict[str, Tensor | None]
    buffer_items: tuple[tuple[str, Tensor | None], ...]
    nonpersistent_registry: set[str]
    nonpersistent_names: frozenset[str]
    training: bool
    hooks: tuple[_HookRegistrySnapshot, ...]
    plain_attributes: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ParameterRuntimeSnapshot:
    parameter: nn.Parameter
    requires_grad: bool
    gradient: Tensor | None
    gradient_bytes: Tensor | None
    gradient_sha256: str | None


@dataclass(frozen=True, slots=True)
class _BufferRuntimeSnapshot:
    buffer: Tensor
    tensor_bytes: Tensor
    tensor_sha256: str


@dataclass(frozen=True, slots=True)
class _PlainListRuntimeSnapshot:
    value: list[Any]
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PlainDictRuntimeSnapshot:
    value: dict[Any, Any]
    items: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True, slots=True)
class _PlainSetRuntimeSnapshot:
    value: set[Any]
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PlainObjectRuntimeSnapshot:
    value: Any
    dictionary: dict[str, Any]
    dictionary_items: tuple[tuple[str, Any], ...]
    slot_items: tuple[tuple[str, bool, Any], ...]


@dataclass(frozen=True, slots=True)
class _PlainTensorRuntimeSnapshot:
    value: Tensor
    tensor_bytes: Tensor
    tensor_sha256: str


@dataclass(frozen=True, slots=True)
class _PlainRuntimeGraphSnapshot:
    nodes: tuple[
        _PlainListRuntimeSnapshot
        | _PlainDictRuntimeSnapshot
        | _PlainSetRuntimeSnapshot
        | _PlainObjectRuntimeSnapshot
        | _PlainTensorRuntimeSnapshot,
        ...,
    ]
    unsupported_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ModelRuntimeSnapshot:
    root: nn.Module
    modules: tuple[_ModuleRuntimeSnapshot, ...]
    parameters: tuple[_ParameterRuntimeSnapshot, ...]
    buffers: tuple[_BufferRuntimeSnapshot, ...]
    plain_runtime: _PlainRuntimeGraphSnapshot


def _identity_items_equal(
    value: Any,
    expected: Sequence[tuple[Any, Any]],
) -> bool:
    if not isinstance(value, Mapping) or len(value) != len(expected):
        return False
    return all(
        actual_key == expected_key and actual_value is expected_value
        for (actual_key, actual_value), (expected_key, expected_value) in zip(
            value.items(), expected, strict=True
        )
    )


def _plain_attributes(module: nn.Module) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted(
            (
                (str(name), value)
                for name, value in module.__dict__.items()
                if name not in _CONTROLLED_MODULE_ATTRIBUTES
            ),
            key=lambda row: row[0],
        )
    )


def _plain_value_matches(actual: Any, expected: Any) -> bool:
    if actual is expected:
        return True
    if isinstance(expected, (type(None), bool, int, float, complex, str, bytes)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        return len(actual) == len(expected) and all(
            _plain_value_matches(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, frozenset) and isinstance(actual, frozenset):
        return actual == expected
    return False


_PLAIN_ATOMIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    range,
    slice,
)
_PLAIN_OPAQUE_TYPES = (
    BuiltinFunctionType,
    ReferenceType,
    re.Pattern,
)


def _plain_slot_names(value: Any) -> tuple[str, ...]:
    names: list[str] = []
    for owner in type(value).__mro__:
        declared = owner.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for raw_name in declared:
            name = str(raw_name)
            if name in {"__dict__", "__weakref__"}:
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{owner.__name__.lstrip('_')}{name}"
            if name not in names:
                names.append(name)
    return tuple(names)


def _capture_plain_runtime_graph(
    modules: Sequence[_ModuleRuntimeSnapshot],
    *,
    name: str,
) -> _PlainRuntimeGraphSnapshot:
    nodes: list[
        _PlainListRuntimeSnapshot
        | _PlainDictRuntimeSnapshot
        | _PlainSetRuntimeSnapshot
        | _PlainObjectRuntimeSnapshot
        | _PlainTensorRuntimeSnapshot
    ] = []
    unsupported: list[str] = []
    observed: set[int] = set()
    registered_module_ids = {id(state.module) for state in modules}

    def unsupported_value(value: Any, path: str) -> None:
        unsupported.append(
            f"{path} ({type(value).__module__}.{type(value).__qualname__})"
        )

    def visit(value: Any, path: str) -> None:
        if (
            isinstance(value, _PLAIN_ATOMIC_TYPES)
            or value is Ellipsis
            or value is NotImplemented
        ):
            return
        if isinstance(value, (Enum, PurePath)) or isinstance(
            value, _PLAIN_OPAQUE_TYPES
        ):
            return
        if isinstance(value, type):
            return
        if isinstance(value, FunctionType):
            if value.__closure__ is not None:
                unsupported_value(value, path)
                return
            if not value.__dict__:
                return
        if isinstance(value, MethodType):
            owner = value.__self__
            if owner is not None:
                visit(owner, f"{path}.__self__")
            return
        if isinstance(value, ModuleType):
            unsupported_value(value, path)
            return
        if isinstance(value, nn.Module):
            if id(value) not in registered_module_ids:
                unsupported_value(value, path)
            return
        if isinstance(value, Tensor):
            if id(value) in observed:
                return
            observed.add(id(value))
            if (
                value.layout != strided
                or value.device.type == "meta"
                or value.requires_grad
                or value.grad_fn is not None
            ):
                unsupported_value(value, path)
                return
            nodes.append(
                _PlainTensorRuntimeSnapshot(
                    value=value,
                    tensor_bytes=value.detach().to(device="cpu").clone(),
                    tensor_sha256=_dense_tensor_sha256(value),
                )
            )
            return
        if id(value) in observed:
            return
        observed.add(id(value))
        if type(value) is list:
            items = tuple(value)
            nodes.append(_PlainListRuntimeSnapshot(value=value, items=items))
            for index, item in enumerate(items):
                visit(item, f"{path}[{index}]")
            return
        if type(value) is dict:
            items = tuple(value.items())
            nodes.append(_PlainDictRuntimeSnapshot(value=value, items=items))
            for index, (key, item) in enumerate(items):
                visit(key, f"{path}.key[{index}]")
                visit(item, f"{path}.value[{index}]")
            return
        if type(value) is set:
            items = tuple(value)
            nodes.append(_PlainSetRuntimeSnapshot(value=value, items=items))
            for index, item in enumerate(items):
                visit(item, f"{path}.set[{index}]")
            return
        if isinstance(value, (list, dict, set)):
            unsupported_value(value, path)
            return
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, frozenset):
            for index, item in enumerate(value):
                visit(item, f"{path}.frozenset[{index}]")
            return
        try:
            dictionary = value.__dict__
        except AttributeError:
            # Known torch scalar descriptors are immutable runtime values.
            if type(value).__module__ == "torch" and type(value).__name__ in {
                "device",
                "dtype",
                "layout",
                "memory_format",
            }:
                return
            unsupported_value(value, path)
            return
        if type(dictionary) is not dict:
            unsupported_value(value, path)
            return
        dictionary_items = tuple(
            (str(attribute), item) for attribute, item in dictionary.items()
        )
        slots = tuple(
            (slot, hasattr(value, slot), getattr(value, slot, None))
            for slot in _plain_slot_names(value)
        )
        nodes.append(
            _PlainObjectRuntimeSnapshot(
                value=value,
                dictionary=dictionary,
                dictionary_items=dictionary_items,
                slot_items=slots,
            )
        )
        for attribute, item in dictionary_items:
            visit(item, f"{path}.{attribute}")
        for slot, existed, item in slots:
            if existed:
                visit(item, f"{path}.{slot}")

    for module_index, state in enumerate(modules):
        for attribute, value in state.plain_attributes:
            visit(value, f"{name}.module[{module_index}].{attribute}")
        for hook in state.hooks:
            if hook.items is None:
                continue
            for hook_index, (key, callback) in enumerate(hook.items):
                visit(
                    key,
                    f"{name}.module[{module_index}].{hook.name}.key[{hook_index}]",
                )
                visit(
                    callback,
                    f"{name}.module[{module_index}].{hook.name}[{hook_index}]",
                )
    return _PlainRuntimeGraphSnapshot(
        nodes=tuple(nodes),
        unsupported_paths=tuple(dict.fromkeys(unsupported)),
    )


def _plain_runtime_graph_matches(snapshot: _PlainRuntimeGraphSnapshot) -> bool:
    def sequence_matches(actual: Sequence[Any], expected: Sequence[Any]) -> bool:
        return len(actual) == len(expected) and all(
            _plain_value_matches(left, right)
            for left, right in zip(actual, expected, strict=True)
        )

    try:
        for state in snapshot.nodes:
            if isinstance(state, _PlainListRuntimeSnapshot):
                if not sequence_matches(state.value, state.items):
                    return False
            elif isinstance(state, _PlainDictRuntimeSnapshot):
                actual_items = tuple(state.value.items())
                if len(actual_items) != len(state.items) or any(
                    not _plain_value_matches(actual_key, expected_key)
                    or not _plain_value_matches(actual_value, expected_value)
                    for (actual_key, actual_value), (
                        expected_key,
                        expected_value,
                    ) in zip(actual_items, state.items, strict=True)
                ):
                    return False
            elif isinstance(state, _PlainSetRuntimeSnapshot):
                unmatched = list(state.value)
                for expected in state.items:
                    for index, actual in enumerate(unmatched):
                        if _plain_value_matches(actual, expected):
                            unmatched.pop(index)
                            break
                    else:
                        return False
                if unmatched:
                    return False
            elif isinstance(state, _PlainObjectRuntimeSnapshot):
                if state.value.__dict__ is not state.dictionary:
                    return False
                actual_items = tuple(state.dictionary.items())
                if len(actual_items) != len(state.dictionary_items) or any(
                    actual_name != expected_name
                    or not _plain_value_matches(actual_value, expected_value)
                    for (actual_name, actual_value), (
                        expected_name,
                        expected_value,
                    ) in zip(actual_items, state.dictionary_items, strict=True)
                ):
                    return False
                for slot, existed, expected in state.slot_items:
                    if hasattr(state.value, slot) != existed:
                        return False
                    if existed and not _plain_value_matches(
                        getattr(state.value, slot), expected
                    ):
                        return False
            elif _dense_tensor_sha256(state.value) != state.tensor_sha256:
                return False
        return True
    except BaseException:
        return False


def _restore_plain_runtime_graph(snapshot: _PlainRuntimeGraphSnapshot) -> None:
    # Restore object attributes first so custom hash/equality state is stable
    # before those objects are reinserted into dictionaries or sets.
    for state in snapshot.nodes:
        if not isinstance(state, _PlainObjectRuntimeSnapshot):
            continue
        if state.value.__dict__ is not state.dictionary:
            object.__setattr__(state.value, "__dict__", state.dictionary)
        state.dictionary.clear()
        state.dictionary.update(state.dictionary_items)
        for slot, existed, expected in state.slot_items:
            if existed:
                object.__setattr__(state.value, slot, expected)
            elif hasattr(state.value, slot):
                object.__delattr__(state.value, slot)
    with no_grad():
        for state in snapshot.nodes:
            if not isinstance(state, _PlainTensorRuntimeSnapshot):
                continue
            value = state.value
            expected = state.tensor_bytes
            if value.dtype != expected.dtype:
                raise ProgressiveTargetPublicationError(
                    "Plain runtime tensor dtype changed and cannot be restored."
                )
            if tuple(value.shape) != tuple(expected.shape):
                value.resize_(tuple(expected.shape))
            value.copy_(expected.to(device=value.device))
    for state in snapshot.nodes:
        if isinstance(state, _PlainListRuntimeSnapshot):
            state.value.clear()
            state.value.extend(state.items)
        elif isinstance(state, _PlainDictRuntimeSnapshot):
            state.value.clear()
            state.value.update(state.items)
        elif isinstance(state, _PlainSetRuntimeSnapshot):
            state.value.clear()
            state.value.update(state.items)


def _capture_model_runtime_snapshot(
    model: nn.Module,
    *,
    name: str,
) -> _ModelRuntimeSnapshot:
    try:
        named_modules = model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - older torch compatibility.
        named_modules = model.named_modules()
    modules: list[_ModuleRuntimeSnapshot] = []
    parameters: list[_ParameterRuntimeSnapshot] = []
    buffers: list[_BufferRuntimeSnapshot] = []
    observed_modules: set[int] = set()
    observed_parameters: set[int] = set()
    observed_buffers: set[int] = set()
    for _module_name, module in named_modules:
        if id(module) in observed_modules:
            continue
        observed_modules.add(id(module))
        hooks: list[_HookRegistrySnapshot] = []
        for hook_name in _HOOK_REGISTRY_NAMES:
            existed = hook_name in module.__dict__
            value = module.__dict__.get(hook_name)
            items = tuple(value.items()) if isinstance(value, Mapping) else None
            hooks.append(
                _HookRegistrySnapshot(
                    name=hook_name,
                    existed=existed,
                    value=value,
                    items=items,
                )
            )
        modules.append(
            _ModuleRuntimeSnapshot(
                module=module,
                modules_registry=module._modules,
                module_items=tuple(module._modules.items()),
                parameters_registry=module._parameters,
                parameter_items=tuple(module._parameters.items()),
                buffers_registry=module._buffers,
                buffer_items=tuple(module._buffers.items()),
                nonpersistent_registry=module._non_persistent_buffers_set,
                nonpersistent_names=frozenset(module._non_persistent_buffers_set),
                training=bool(module.training),
                hooks=tuple(hooks),
                plain_attributes=_plain_attributes(module),
            )
        )
        for parameter in module._parameters.values():
            if parameter is None or id(parameter) in observed_parameters:
                continue
            observed_parameters.add(id(parameter))
            gradient = parameter.grad
            if gradient is not None and (
                gradient.layout != strided or gradient.device.type == "meta"
            ):
                raise ProgressiveTargetPublicationError(
                    f"{name} cannot guard a sparse or meta parameter gradient."
                )
            gradient_bytes = (
                None
                if gradient is None
                else gradient.detach().to(device="cpu").clone()
            )
            parameters.append(
                _ParameterRuntimeSnapshot(
                    parameter=parameter,
                    requires_grad=bool(parameter.requires_grad),
                    gradient=gradient,
                    gradient_bytes=gradient_bytes,
                    gradient_sha256=(
                        None if gradient is None else _dense_tensor_sha256(gradient)
                    ),
                )
            )
        for buffer in module._buffers.values():
            if buffer is None or id(buffer) in observed_buffers:
                continue
            observed_buffers.add(id(buffer))
            if buffer.layout != strided or buffer.device.type == "meta":
                raise ProgressiveTargetPublicationError(
                    f"{name} cannot guard a sparse or meta registered buffer."
                )
            buffers.append(
                _BufferRuntimeSnapshot(
                    buffer=buffer,
                    tensor_bytes=buffer.detach().to(device="cpu").clone(),
                    tensor_sha256=_dense_tensor_sha256(buffer),
                )
            )
    if not modules or not parameters:
        raise ProgressiveTargetPublicationError(
            f"{name} found no module/parameter runtime state to guard."
        )
    module_states = tuple(modules)
    return _ModelRuntimeSnapshot(
        root=model,
        modules=module_states,
        parameters=tuple(parameters),
        buffers=tuple(buffers),
        plain_runtime=_capture_plain_runtime_graph(
            module_states,
            name=name,
        ),
    )


def _model_runtime_snapshot_matches(
    model: nn.Module,
    snapshot: _ModelRuntimeSnapshot,
) -> bool:
    if model is not snapshot.root:
        return False
    try:
        for state in snapshot.modules:
            module = state.module
            if (
                module._modules is not state.modules_registry
                or not _identity_items_equal(module._modules, state.module_items)
                or module._parameters is not state.parameters_registry
                or not _identity_items_equal(module._parameters, state.parameter_items)
                or module._buffers is not state.buffers_registry
                or not _identity_items_equal(module._buffers, state.buffer_items)
                or module._non_persistent_buffers_set
                is not state.nonpersistent_registry
                or frozenset(module._non_persistent_buffers_set)
                != state.nonpersistent_names
                or bool(module.training) != state.training
            ):
                return False
            for hook in state.hooks:
                exists = hook.name in module.__dict__
                current = module.__dict__.get(hook.name)
                if exists != hook.existed or current is not hook.value:
                    return False
                if hook.items is not None and not _identity_items_equal(
                    current, hook.items
                ):
                    return False
            current_plain = _plain_attributes(module)
            if len(current_plain) != len(state.plain_attributes) or any(
                actual_name != expected_name
                or not _plain_value_matches(actual_value, expected_value)
                for (actual_name, actual_value), (expected_name, expected_value) in zip(
                    current_plain, state.plain_attributes, strict=True
                )
            ):
                return False
        for state in snapshot.parameters:
            parameter = state.parameter
            if (
                bool(parameter.requires_grad) != state.requires_grad
                or parameter.grad is not state.gradient
            ):
                return False
            if state.gradient is not None and (
                state.gradient_sha256 is None
                or _dense_tensor_sha256(state.gradient) != state.gradient_sha256
            ):
                return False
        return _plain_runtime_graph_matches(snapshot.plain_runtime) and all(
            _dense_tensor_sha256(state.buffer) == state.tensor_sha256
            for state in snapshot.buffers
        )
    except BaseException:
        return False


def _restore_model_runtime_snapshot(snapshot: _ModelRuntimeSnapshot) -> None:
    for state in snapshot.modules:
        module = state.module
        object.__setattr__(module, "_modules", state.modules_registry)
        state.modules_registry.clear()
        state.modules_registry.update(state.module_items)
        object.__setattr__(module, "_parameters", state.parameters_registry)
        state.parameters_registry.clear()
        state.parameters_registry.update(state.parameter_items)
        object.__setattr__(module, "_buffers", state.buffers_registry)
        state.buffers_registry.clear()
        state.buffers_registry.update(state.buffer_items)
        object.__setattr__(
            module,
            "_non_persistent_buffers_set",
            state.nonpersistent_registry,
        )
        state.nonpersistent_registry.clear()
        state.nonpersistent_registry.update(state.nonpersistent_names)
        module.__dict__["training"] = state.training
        expected_plain = dict(state.plain_attributes)
        for attribute in tuple(module.__dict__):
            if (
                attribute not in _CONTROLLED_MODULE_ATTRIBUTES
                and attribute not in expected_plain
            ):
                module.__dict__.pop(attribute, None)
        module.__dict__.update(expected_plain)
        for hook in state.hooks:
            if not hook.existed:
                module.__dict__.pop(hook.name, None)
                continue
            module.__dict__[hook.name] = hook.value
            if hook.items is not None:
                hook.value.clear()
                hook.value.update(hook.items)
    _restore_plain_runtime_graph(snapshot.plain_runtime)
    with no_grad():
        for state in snapshot.parameters:
            parameter = state.parameter
            parameter.requires_grad_(state.requires_grad)
            if state.gradient is None:
                parameter.grad = None
                continue
            gradient = state.gradient
            expected = state.gradient_bytes
            if expected is None:
                raise ProgressiveTargetPublicationError(
                    "Gradient snapshot bytes are absent during restoration."
                )
            if gradient.dtype != expected.dtype:
                raise ProgressiveTargetPublicationError(
                    "Original gradient dtype changed and cannot be restored."
                )
            if tuple(gradient.shape) != tuple(expected.shape):
                gradient.resize_(tuple(expected.shape))
            gradient.copy_(expected.to(device=gradient.device))
            parameter.grad = gradient
        for state in snapshot.buffers:
            buffer = state.buffer
            expected = state.tensor_bytes
            if buffer.dtype != expected.dtype:
                raise ProgressiveTargetPublicationError(
                    "Original buffer dtype changed and cannot be restored."
                )
            if tuple(buffer.shape) != tuple(expected.shape):
                buffer.resize_(tuple(expected.shape))
            buffer.copy_(expected.to(device=buffer.device))


def _unique_parameter_bytes(model: nn.Module) -> int:
    unique: dict[tuple[Any, ...], int] = {}
    for _, parameter in model.named_parameters(recurse=True, remove_duplicate=False):
        key = (
            parameter.device.type,
            parameter.device.index,
            str(parameter.dtype),
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
        unique[key] = int(parameter.numel()) * int(parameter.element_size())
    return sum(unique.values())


@dataclass(frozen=True, slots=True)
class DenseGQATargetShape:
    """Closed-form storage shape for a native dense GQA/SwiGLU target.

    The formula covers the current Qwen3 and SmolLM2/Llama progressive targets.
    ``query_key_head_norms`` is explicit because Qwen3 stores one Q and K
    RMSNorm vector of ``head_dim`` elements per layer, while SmolLM2 does not.
    """

    family: Literal["qwen3", "llama"]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    tie_word_embeddings: bool
    query_key_head_norms: bool
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        if self.family not in {"qwen3", "llama"}:
            raise ProgressiveTargetBridgeError(
                "DenseGQATargetShape family must be qwen3 or llama."
            )
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "tie_word_embeddings",
            "query_key_head_norms",
            "attention_bias",
            "mlp_bias",
        ):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ProgressiveTargetBridgeError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )

    @property
    def geometry(self) -> DenseGQAGeometry:
        return DenseGQAGeometry(
            logical_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            query_heads=self.num_attention_heads,
            key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )

    def parameter_breakdown(self) -> dict[str, Any]:
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        query_width = self.num_attention_heads * self.head_dim
        key_value_width = self.num_key_value_heads * self.head_dim
        attention_weights = 2 * hidden * (query_width + key_value_width)
        attention_biases = (
            query_width + 2 * key_value_width + hidden if self.attention_bias else 0
        )
        query_key_norms = 2 * self.head_dim if self.query_key_head_norms else 0
        mlp_weights = 3 * hidden * intermediate
        mlp_biases = 2 * intermediate + hidden if self.mlp_bias else 0
        decoder_norms = 2 * hidden
        per_layer = (
            attention_weights
            + attention_biases
            + query_key_norms
            + mlp_weights
            + mlp_biases
            + decoder_norms
        )
        embedding = self.vocab_size * hidden
        output_head_unique = 0 if self.tie_word_embeddings else embedding
        decoder_layers = self.num_hidden_layers * per_layer
        final_norm = hidden
        total = embedding + output_head_unique + decoder_layers + final_norm
        return {
            "family": self.family,
            "shape": self.as_dict(),
            "derived": {
                "query_width": query_width,
                "key_value_width": key_value_width,
                "queries_per_key_value_head": (
                    self.num_attention_heads // self.num_key_value_heads
                ),
            },
            "per_layer": {
                "attention_weights": attention_weights,
                "attention_biases": attention_biases,
                "query_key_head_norms": query_key_norms,
                "gated_mlp_weights": mlp_weights,
                "mlp_biases": mlp_biases,
                "decoder_norms": decoder_norms,
                "total": per_layer,
            },
            "components": {
                "token_embedding": embedding,
                "output_head_unique": output_head_unique,
                "decoder_layers": decoder_layers,
                "final_norm": final_norm,
            },
            "total": total,
        }

    @property
    def unique_parameter_count(self) -> int:
        return int(self.parameter_breakdown()["total"])

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressiveDenseGQAStage:
    """One exact native target in the progressive compiler chain."""

    name: str
    shape: DenseGQATargetShape
    exact_unique_parameters: int
    tokenizer_transition: Literal["preserve", "canonical-byte-transition"]
    purpose: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not self.name.replace("_", "").replace("-", "").isalnum()
        ):
            raise ProgressiveTargetBridgeError("Stage name is not a safe identifier.")
        if not isinstance(self.shape, DenseGQATargetShape):
            raise ProgressiveTargetBridgeError(
                "stage.shape must be DenseGQATargetShape."
            )
        object.__setattr__(
            self,
            "exact_unique_parameters",
            _positive_int(self.exact_unique_parameters, "exact_unique_parameters"),
        )
        if self.tokenizer_transition not in {
            "preserve",
            "canonical-byte-transition",
        }:
            raise ProgressiveTargetBridgeError(
                "Unsupported tokenizer transition policy."
            )
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ProgressiveTargetBridgeError("Stage purpose cannot be empty.")
        calculated = self.shape.unique_parameter_count
        if calculated != self.exact_unique_parameters:
            raise ProgressiveTargetBridgeError(
                f"Stage {self.name!r} declares {self.exact_unique_parameters} "
                f"parameters, but its native shape contains {calculated}."
            )

    @property
    def geometry(self) -> DenseGQAGeometry:
        return self.shape.geometry

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "exact_unique_parameters": self.exact_unique_parameters,
            "tokenizer_transition": self.tokenizer_transition,
            "purpose": self.purpose,
            "shape": self.shape.as_dict(),
            "parameter_breakdown": self.shape.parameter_breakdown(),
            "native_dense_target": True,
            "adapter_parameters_at_publication": 0,
        }
        payload["stage_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ProgressiveDenseGQAPlan:
    """Exact monotone stage plan; behavior promotion remains external."""

    source_name: str
    source_unique_parameters: int
    stages: tuple[ProgressiveDenseGQAStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ProgressiveTargetBridgeError("source_name cannot be empty.")
        object.__setattr__(
            self,
            "source_unique_parameters",
            _positive_int(self.source_unique_parameters, "source_unique_parameters"),
        )
        values = tuple(self.stages)
        if not values or any(
            not isinstance(stage, ProgressiveDenseGQAStage) for stage in values
        ):
            raise ProgressiveTargetBridgeError(
                "stages must contain ProgressiveDenseGQAStage values."
            )
        if len({stage.name for stage in values}) != len(values):
            raise ProgressiveTargetBridgeError(
                "Progressive stage names must be unique."
            )
        previous = self.source_unique_parameters
        for stage in values:
            if stage.exact_unique_parameters >= previous:
                raise ProgressiveTargetBridgeError(
                    "Progressive target counts must be strictly descending."
                )
            previous = stage.exact_unique_parameters
        object.__setattr__(self, "stages", values)

    @property
    def final_unique_parameters(self) -> int:
        return self.stages[-1].exact_unique_parameters

    @property
    def overall_compression_ratio(self) -> float:
        return self.source_unique_parameters / self.final_unique_parameters

    def as_dict(self) -> dict[str, Any]:
        previous_name = self.source_name
        previous_count = self.source_unique_parameters
        stage_records = []
        for stage in self.stages:
            stage_records.append(
                {
                    **stage.as_dict(),
                    "input_name": previous_name,
                    "input_unique_parameters": previous_count,
                    "single_stage_compression_ratio": (
                        previous_count / stage.exact_unique_parameters
                    ),
                    "cumulative_compression_ratio": (
                        self.source_unique_parameters / stage.exact_unique_parameters
                    ),
                    "behavior_gate_required_before_next_stage": True,
                }
            )
            previous_name = stage.name
            previous_count = stage.exact_unique_parameters
        payload = {
            "schema": PROGRESSIVE_TARGET_PLAN_SCHEMA,
            "source": {
                "name": self.source_name,
                "unique_parameters": self.source_unique_parameters,
            },
            "stages": stage_records,
            "final_unique_parameters": self.final_unique_parameters,
            "overall_compression_ratio": self.overall_compression_ratio,
            "minimum_25x_endpoint": self.overall_compression_ratio >= 25.0,
            "every_stage_is_native_target_sized": True,
            "every_stage_requires_parent_relative_behavior_promotion": True,
            "capability_proven_by_plan": False,
        }
        payload["plan_sha256"] = _canonical_sha256(payload)
        return payload


def default_qwen4b_progressive_target_plan() -> ProgressiveDenseGQAPlan:
    """Return the exact 4B -> 1.410B -> 600.628M -> 134.515M plan."""

    return ProgressiveDenseGQAPlan(
        source_name="Qwen3-4B-Base",
        source_unique_parameters=QWEN3_4B_EXACT_PARAMETERS,
        stages=(
            ProgressiveDenseGQAStage(
                name="qwen3_1p410b",
                shape=DenseGQATargetShape(
                    family="qwen3",
                    vocab_size=151_936,
                    hidden_size=1_536,
                    intermediate_size=5_120,
                    num_hidden_layers=30,
                    num_attention_heads=16,
                    num_key_value_heads=4,
                    head_dim=128,
                    tie_word_embeddings=False,
                    query_key_head_norms=True,
                ),
                exact_unique_parameters=QWEN3_1P4B_EXACT_PARAMETERS,
                tokenizer_transition="preserve",
                purpose="first bounded same-tokenizer width/depth fold",
            ),
            ProgressiveDenseGQAStage(
                name="qwen3_600m",
                shape=DenseGQATargetShape(
                    family="qwen3",
                    vocab_size=151_936,
                    hidden_size=1_024,
                    intermediate_size=3_072,
                    num_hidden_layers=24,
                    num_attention_heads=8,
                    num_key_value_heads=2,
                    head_dim=128,
                    tie_word_embeddings=False,
                    query_key_head_norms=True,
                ),
                exact_unique_parameters=QWEN3_600M_EXACT_PARAMETERS,
                tokenizer_transition="preserve",
                purpose="second bounded same-tokenizer width/depth fold",
            ),
            ProgressiveDenseGQAStage(
                name="smollm2_134m",
                shape=DenseGQATargetShape(
                    family="llama",
                    vocab_size=49_152,
                    hidden_size=576,
                    intermediate_size=1_536,
                    num_hidden_layers=30,
                    num_attention_heads=9,
                    num_key_value_heads=3,
                    head_dim=64,
                    tie_word_embeddings=True,
                    query_key_head_norms=False,
                ),
                exact_unique_parameters=SMOLLM2_135M_EXACT_PARAMETERS,
                tokenizer_transition="canonical-byte-transition",
                purpose="29.9x standalone product-size endpoint",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ExternalBehaviorGateReceipt:
    """External output/rollout evidence required to authorize dense baking.

    The bridge validates the receipt's shape, not the referenced evaluation
    file.  Consequently its reports say that behavior was externally attested,
    never that this structural module independently proved capability.
    """

    evidence_sha256: str
    evaluation_contract_sha256: str
    heldout_action_count: int
    free_running_rollout_count: int
    passed: bool
    internal_similarity_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_sha256",
            _sha256(self.evidence_sha256, "evidence_sha256"),
        )
        object.__setattr__(
            self,
            "evaluation_contract_sha256",
            _sha256(
                self.evaluation_contract_sha256,
                "evaluation_contract_sha256",
            ),
        )
        object.__setattr__(
            self,
            "heldout_action_count",
            _nonnegative_int(self.heldout_action_count, "heldout_action_count"),
        )
        object.__setattr__(
            self,
            "free_running_rollout_count",
            _nonnegative_int(
                self.free_running_rollout_count,
                "free_running_rollout_count",
            ),
        )
        object.__setattr__(self, "passed", _strict_bool(self.passed, "passed"))
        object.__setattr__(
            self,
            "internal_similarity_only",
            _strict_bool(self.internal_similarity_only, "internal_similarity_only"),
        )

    def require_publication_approval(self) -> None:
        if not self.passed:
            raise ProgressiveTargetPublicationError(
                "The external behavior gate did not approve publication."
            )
        if self.internal_similarity_only:
            raise ProgressiveTargetPublicationError(
                "Internal numeric similarity cannot authorize publication."
            )
        if self.heldout_action_count <= 0 or self.free_running_rollout_count <= 0:
            raise ProgressiveTargetPublicationError(
                "Publication needs held-out actions and free-running rollouts."
            )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": EXTERNAL_BEHAVIOR_GATE_SCHEMA,
            **asdict(self),
            "behavior_evidence_externally_attested": True,
            "evidence_payload_revalidated_by_bridge": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ExperimentalRecoveryMaterializationReceipt:
    """Bind a folded initialization that is not approved for publication.

    This receipt authorizes one reversible dense materialization for an A/B
    recovery experiment. It cannot attest behavior, consume a sealed split, or
    satisfy :class:`ExternalBehaviorGateReceipt`.
    """

    experiment_contract_sha256: str
    factor_state_sha256: str
    live_factor_state_sha256: str
    native_child_state_sha256: str
    exact_unique_parameters: int
    factor_payload_schema: str

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_contract_sha256",
            "factor_state_sha256",
            "live_factor_state_sha256",
            "native_child_state_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "exact_unique_parameters",
            _positive_int(
                self.exact_unique_parameters,
                "exact_unique_parameters",
            ),
        )
        schema = str(self.factor_payload_schema).strip()
        if re.fullmatch(
            r"neural-forge-target-native-factor-pack-v[1-9][0-9]*",
            schema,
        ) is None:
            raise ProgressiveTargetBridgeError(
                "factor_payload_schema must be a target-native factor-pack schema."
            )
        object.__setattr__(self, "factor_payload_schema", schema)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": EXPERIMENTAL_RECOVERY_MATERIALIZATION_RECEIPT_SCHEMA,
            **asdict(self),
            "status": "experimental-unaccepted",
            "publication_authorized": False,
            "sealed_data_accessed": False,
            "behavior_capability_proven": False,
            "recovery_training_required": True,
            "external_experiment_contract_revalidated_by_bridge": False,
            "external_factor_pack_revalidated_by_bridge": False,
            "live_factor_tensor_state_revalidated_by_bridge": True,
            "native_child_state_revalidated_by_bridge": True,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class NativeDenseGQACertificate:
    exact_unique_parameters: int
    logical_parameters: int
    unique_learned_weight_bytes: int
    structure_sha256: str
    state_key_sha256: str
    state_key_count: int
    dense_qkvo_projection_count: int
    dense_swiglu_projection_count: int
    geometry: DenseGQAGeometry
    projection_names: Mapping[str, str]
    swiglu_projection_names: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": NATIVE_TARGET_CERTIFICATE_SCHEMA,
            "exact_unique_parameters": self.exact_unique_parameters,
            "logical_parameters": self.logical_parameters,
            "unique_learned_weight_bytes": self.unique_learned_weight_bytes,
            "structure_sha256": self.structure_sha256,
            "state_key_sha256": self.state_key_sha256,
            "state_key_count": self.state_key_count,
            "dense_qkvo_projection_count": self.dense_qkvo_projection_count,
            "dense_swiglu_projection_count": self.dense_swiglu_projection_count,
            "geometry": self.geometry.as_dict(),
            "projection_names": dict(self.projection_names),
            "swiglu_projection_names": dict(self.swiglu_projection_names),
            "all_qkvo_projections_are_exact_nn_linear": True,
            "all_swiglu_projections_are_exact_nn_linear": True,
            "adapter_registry_count": 0,
            "adapter_projection_wrapper_count": 0,
            "adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_parameter_count": 0,
            "capability_proven": False,
        }
        payload["certificate_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class _DenseGQATopology:
    layout: DecoderLayout
    geometry: DenseGQAGeometry
    attention_modules: Mapping[int, nn.Module]
    projection_names: Mapping[str, str]
    projection_paths: tuple[str, ...]
    mlp_modules: Mapping[int, nn.Module]
    swiglu_projection_names: Mapping[str, str]
    swiglu_projection_paths: tuple[str, ...]


def _direct_projection_name(module_path: str, projection_path: str) -> str:
    prefix = f"{module_path}."
    if not projection_path.startswith(prefix):
        raise ProgressiveTargetBridgeError(
            "Projection path is not owned by its declared module."
        )
    name = projection_path[len(prefix) :]
    if not name.isidentifier():
        raise ProgressiveTargetBridgeError(
            "Native publication requires direct module projection fields."
        )
    return name


def _resolve_dense_gqa_topology(
    model: nn.Module,
    layout: DecoderLayout | None,
    *,
    require_dense_swiglu: bool = False,
) -> _DenseGQATopology:
    resolved = build_decoder_layout(model) if layout is None else layout
    if not isinstance(resolved, DecoderLayout):
        raise ProgressiveTargetBridgeError("layout must be DecoderLayout.")
    if not resolved.blocks:
        raise ProgressiveTargetBridgeError("Target decoder has no layers.")
    if tuple(block.index for block in resolved.blocks) != tuple(
        range(resolved.layer_count)
    ):
        raise ProgressiveTargetBridgeError(
            "Target decoder block indices must be contiguous from zero."
        )
    first = resolved.blocks[0].attention
    if first.projection_style != "split_qkv":
        raise ProgressiveTargetBridgeError(
            "Target-native QKVO baking requires split dense Q/K/V projections."
        )
    geometry = DenseGQAGeometry(
        logical_layers=resolved.layer_count,
        hidden_size=resolved.hidden_size,
        query_heads=first.query_heads,
        key_value_heads=first.key_value_heads,
        head_dim=first.head_dim,
    )
    modules: dict[int, nn.Module] = {}
    mlp_modules: dict[int, nn.Module] = {}
    paths: list[str] = []
    mlp_paths: list[str] = []
    names: dict[str, str] | None = None
    mlp_names: dict[str, str] | None = None
    for block in resolved.blocks:
        attention = block.attention
        if (
            attention.projection_style != "split_qkv"
            or attention.hidden_size != geometry.hidden_size
            or attention.query_heads != geometry.query_heads
            or attention.key_value_heads != geometry.key_value_heads
            or attention.head_dim != geometry.head_dim
            or attention.projection_weight_layout != "output_input"
            or attention.query_path is None
            or attention.key_path is None
            or attention.value_path is None
        ):
            raise ProgressiveTargetBridgeError(
                "Every target layer must share one dense split-GQA geometry."
            )
        module = resolve_module(model, attention.module_path, family=resolved.family)
        modules[block.index] = module
        layer_paths = {
            "q": attention.query_path,
            "k": attention.key_path,
            "v": attention.value_path,
            "o": attention.output_path,
        }
        layer_names = {
            role: _direct_projection_name(attention.module_path, path)
            for role, path in layer_paths.items()
        }
        if names is None:
            names = layer_names
        elif names != layer_names:
            raise ProgressiveTargetBridgeError(
                "Target layers use inconsistent QKVO projection attribute names."
            )
        for path in layer_paths.values():
            projection = resolve_module(model, path, family=resolved.family)
            if type(projection) is not nn.Linear:
                raise ProgressiveTargetBridgeError(
                    f"Target projection {path!r} is not an ordinary nn.Linear."
                )
            paths.append(path)
        if require_dense_swiglu:
            mlp = block.mlp
            if (
                mlp.kind != "gated"
                or mlp.gate_projection_path is None
                or mlp.up_projection_path is None
                or mlp.down_projection_path is None
                or mlp.projection_weight_layout != "output_input"
                or mlp.hidden_size != resolved.hidden_size
                or mlp.intermediate_size != resolved.intermediate_size
            ):
                raise ProgressiveTargetBridgeError(
                    "Every target layer must expose one native dense gated MLP."
                )
            mlp_module = resolve_module(model, mlp.module_path, family=resolved.family)
            mlp_modules[block.index] = mlp_module
            layer_mlp_paths = {
                "gate": mlp.gate_projection_path,
                "up": mlp.up_projection_path,
                "down": mlp.down_projection_path,
            }
            layer_mlp_names = {
                role: _direct_projection_name(mlp.module_path, path)
                for role, path in layer_mlp_paths.items()
            }
            if mlp_names is None:
                mlp_names = layer_mlp_names
            elif mlp_names != layer_mlp_names:
                raise ProgressiveTargetBridgeError(
                    "Target layers use inconsistent Gate/Up/Down attribute names."
                )
            for path in layer_mlp_paths.values():
                projection = resolve_module(model, path, family=resolved.family)
                if type(projection) is not nn.Linear:
                    raise ProgressiveTargetBridgeError(
                        f"Target projection {path!r} is not an ordinary nn.Linear."
                    )
                mlp_paths.append(path)
    assert names is not None
    if require_dense_swiglu:
        assert mlp_names is not None
    return _DenseGQATopology(
        layout=resolved,
        geometry=geometry,
        attention_modules=modules,
        projection_names=names,
        projection_paths=tuple(paths),
        mlp_modules=mlp_modules,
        swiglu_projection_names={} if mlp_names is None else mlp_names,
        swiglu_projection_paths=tuple(mlp_paths),
    )


def _validate_stage_matches_layout(
    stage: ProgressiveDenseGQAStage,
    topology: _DenseGQATopology,
) -> None:
    shape = stage.shape
    layout = topology.layout
    first_attention = layout.blocks[0].attention
    query_norm_present = first_attention.query_norm is not None
    key_norm_present = first_attention.key_norm is not None
    if query_norm_present != key_norm_present:
        raise ProgressiveTargetBridgeError(
            "Target must expose both Q/K head norms or neither."
        )
    checks = {
        "family": (layout.family, shape.family),
        "vocab_size": (layout.embeddings.vocab_size, shape.vocab_size),
        "hidden_size": (layout.hidden_size, shape.hidden_size),
        "intermediate_size": (layout.intermediate_size, shape.intermediate_size),
        "num_hidden_layers": (layout.layer_count, shape.num_hidden_layers),
        "num_attention_heads": (
            first_attention.query_heads,
            shape.num_attention_heads,
        ),
        "num_key_value_heads": (
            first_attention.key_value_heads,
            shape.num_key_value_heads,
        ),
        "head_dim": (first_attention.head_dim, shape.head_dim),
        "tie_word_embeddings": (
            layout.tie_word_embeddings,
            shape.tie_word_embeddings,
        ),
        "attention_bias": (
            first_attention.projection_bias,
            shape.attention_bias,
        ),
        "mlp_bias": (layout.blocks[0].mlp.projection_bias, shape.mlp_bias),
        "query_key_head_norms": (
            query_norm_present,
            shape.query_key_head_norms,
        ),
    }
    mismatches = [
        f"{name}: live={actual!r}, planned={planned!r}"
        for name, (actual, planned) in checks.items()
        if actual != planned
    ]
    if mismatches:
        raise ProgressiveTargetBridgeError(
            "Live child does not match its progressive stage: " + "; ".join(mismatches)
        )


def certify_native_dense_gqa_target(
    model: nn.Module,
    *,
    exact_unique_parameters: int,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    require_dense_swiglu: bool = False,
) -> NativeDenseGQACertificate:
    """Certify a loaded target has exact storage and no QKVO sidecar modules."""

    if not isinstance(model, nn.Module):
        raise ProgressiveTargetBridgeError("model must be torch.nn.Module.")
    expected = _positive_int(exact_unique_parameters, "exact_unique_parameters")
    topology = _resolve_dense_gqa_topology(
        model,
        layout,
        require_dense_swiglu=require_dense_swiglu,
    )
    if stage is not None:
        if not isinstance(stage, ProgressiveDenseGQAStage):
            raise ProgressiveTargetBridgeError(
                "stage must be ProgressiveDenseGQAStage."
            )
        if stage.exact_unique_parameters != expected:
            raise ProgressiveTargetBridgeError(
                "Stage count and requested exact count disagree."
            )
        _validate_stage_matches_layout(stage, topology)
    forbidden_registry = sum(
        isinstance(module, SparseDenseGQAQKVOResidualRegistry)
        for module in model.modules()
    )
    forbidden_wrappers = sum(
        isinstance(module, DenseGQAQKVOResidualProjection) for module in model.modules()
    )
    forbidden_mlp_registry = sum(
        isinstance(module, SparseDenseSwiGLUProjectionResidualRegistry)
        for module in model.modules()
    )
    forbidden_mlp_wrappers = sum(
        isinstance(module, DenseSwiGLUResidualProjection) for module in model.modules()
    )
    if (
        forbidden_registry
        or forbidden_wrappers
        or forbidden_mlp_registry
        or forbidden_mlp_wrappers
    ):
        raise ProgressiveTargetBridgeError(
            "Target still contains an adapter registry or projection wrapper."
        )
    counts = exact_parameter_report(model)
    actual = int(counts["unique_parameter_count"])
    if actual != expected:
        raise ProgressiveTargetBridgeError(
            f"Native target contains {actual} unique parameters, expected {expected}."
        )
    registered_rows = _registered_model_tensors(model)
    aliases: dict[int, int] = {}
    keys = tuple(
        {
            "alias_index": aliases.setdefault(id(value), len(aliases)),
            "kind": kind,
            "name": name,
            "persistent": persistent,
        }
        for kind, name, value, persistent in registered_rows
    )
    return NativeDenseGQACertificate(
        exact_unique_parameters=actual,
        logical_parameters=int(counts["logical_parameter_count"]),
        unique_learned_weight_bytes=_unique_parameter_bytes(model),
        structure_sha256=fingerprint_hf_child_structure(
            model,
            layout=topology.layout,
        ),
        state_key_sha256=_canonical_sha256(keys),
        state_key_count=len(keys),
        dense_qkvo_projection_count=len(topology.projection_paths),
        dense_swiglu_projection_count=len(topology.swiglu_projection_paths),
        geometry=topology.geometry,
        projection_names=topology.projection_names,
        swiglu_projection_names=topology.swiglu_projection_names,
    )


class ExactTargetQKVOPublicationSession:
    """One temporary adapter transaction around an exact native child."""

    def __init__(
        self,
        *,
        model: nn.Module,
        exact_unique_parameters: int,
        topology: _DenseGQATopology,
        native_before: NativeDenseGQACertificate,
        handle: InstalledDenseGQAQKVOResidualAdapter,
        stage: ProgressiveDenseGQAStage | None,
        parent_compiler_report_sha256: str | None,
    ) -> None:
        self.model = model
        self.exact_unique_parameters = exact_unique_parameters
        self.topology = topology
        self.native_before = native_before
        self.handle = handle
        self.stage = stage
        self.parent_compiler_report_sha256 = parent_compiler_report_sha256
        self.publication_report: dict[str, Any] | None = None

    @property
    def is_live(self) -> bool:
        return self.handle.is_applied

    def output_training_scope(self) -> DenseGQAAdapterOnlyOutputTrainingScope:
        return self.handle.output_training_scope()

    def rollback(self) -> NativeDenseGQACertificate:
        self.handle.rollback()
        restored = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
        )
        if restored.as_dict() != self.native_before.as_dict():
            raise ProgressiveTargetPublicationError(
                "QKVO rejection did not restore the exact native target."
            )
        return restored

    def installation_report(self) -> dict[str, Any]:
        payload = {
            "schema": EXACT_TARGET_QKVO_INSTALLATION_SCHEMA,
            "exact_unique_parameters_at_publication": self.exact_unique_parameters,
            "temporary_installed_parameters": (
                self.handle.report.model_parameter_count_installed
            ),
            "temporary_adapter_parameters": (
                self.handle.report.adapter_parameter_count
            ),
            "native_before": self.native_before.as_dict(),
            "dense_qkvo_installation": self.handle.report.as_dict(),
            "stage": None if self.stage is None else self.stage.as_dict(),
            "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
            "temporary_factors_may_be_output_trained": True,
            "internal_numeric_matching_required": False,
            "checkpoint_publication_allowed_while_installed": False,
            "capability_proven": False,
        }
        payload["installation_bridge_sha256"] = _canonical_sha256(payload)
        return payload

    def bake_for_publication(
        self,
        behavior_gate: ExternalBehaviorGateReceipt,
    ) -> dict[str, Any]:
        if not isinstance(behavior_gate, ExternalBehaviorGateReceipt):
            raise ProgressiveTargetPublicationError(
                "behavior_gate must be ExternalBehaviorGateReceipt."
            )
        behavior_gate.require_publication_approval()
        bake = self.handle.bake_for_publication()
        native_after = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
        )
        if (
            native_after.structure_sha256 != self.native_before.structure_sha256
            or native_after.state_key_sha256 != self.native_before.state_key_sha256
            or native_after.state_key_count != self.native_before.state_key_count
        ):
            raise ProgressiveTargetPublicationError(
                "Dense QKVO bake did not restore the native target topology."
            )
        payload = {
            "schema": EXACT_TARGET_QKVO_PUBLICATION_SCHEMA,
            "exact_unique_parameters": self.exact_unique_parameters,
            "exact_budget_match": True,
            "native_before": self.native_before.as_dict(),
            "native_after": native_after.as_dict(),
            "dense_bake": bake.as_dict(),
            "external_behavior_gate": behavior_gate.as_dict(),
            "stage": None if self.stage is None else self.stage.as_dict(),
            "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
            "published_adapter_parameter_count": 0,
            "published_adapter_registry_count": 0,
            "published_projection_wrapper_count": 0,
            "published_adapter_flops_per_token": 0,
            "published_parent_runtime_reference_count": 0,
            "published_source_sized_sidecar_parameter_count": 0,
            "native_state_key_topology_restored": True,
            "behavior_gate_externally_attested": True,
            "behavior_evidence_revalidated_by_this_module": False,
            "capability_proven_by_this_module": False,
            "save_reload_artifact_gate_still_required": True,
        }
        payload["publication_sha256"] = _canonical_sha256(payload)
        self.publication_report = payload
        return payload

    def __enter__(self) -> ExactTargetQKVOPublicationSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self.handle.is_applied:
            self.rollback()


class ExactTargetCompilerOutputTrainingScope:
    """Expose only combined QKVO, Gate/Up/Down, and lexical factors."""

    def __init__(self, transaction: ExactTargetCompilerTransaction) -> None:
        self.transaction = transaction
        self._base: tuple[tuple[nn.Parameter, bool, Tensor | None, int, int], ...] = ()
        self._trainable: tuple[tuple[nn.Parameter, bool, Tensor | None], ...] = ()
        self._modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise ProgressiveTargetBridgeError(
                "Compiler output-training scope cannot be re-entered."
            )
        self.transaction.assert_live()
        trainable = self.transaction.output_trainable_parameters()
        trainable_ids = {id(value) for value in trainable}
        if len(trainable_ids) != len(trainable):
            raise ProgressiveTargetBridgeError(
                "Temporary compiler factors unexpectedly share parameters."
            )
        self._base = tuple(
            (
                value,
                bool(value.requires_grad),
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in self.transaction.model.parameters()
            if id(value) not in trainable_ids
        )
        self._trainable = tuple(
            (value, bool(value.requires_grad), value.grad) for value in trainable
        )
        self._modes = tuple(
            (module, bool(module.training))
            for module in self.transaction.model.modules()
        )
        for value, *_rest in self._base:
            value.requires_grad_(False)
        for value, _requires_grad, _gradient in self._trainable:
            value.requires_grad_(True)
        self.transaction.model.eval()
        self._entered = True
        return trainable

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        failed = exc_type is not None
        del exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._base:
            value.requires_grad_(requires_grad)
            value.grad = gradient
        for value, requires_grad, _gradient in self._trainable:
            value.requires_grad_(requires_grad)
        for module, training in self._modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            for value, _requires_grad, gradient, version, pointer in self._base
        ):
            raise ProgressiveTargetBridgeError(
                "A native child parameter changed during factor-only training."
            )
        # Preserve the original training exception even when a disposable
        # factor was poisoned.  The trainer can now restore its exact captured
        # factor state before the usual full finite/live assertion.  A clean
        # scope exit still enforces the complete invariant here.
        if not failed:
            self.transaction.assert_live()

    def as_dict(self) -> dict[str, Any]:
        self.transaction.assert_live()
        payload = {
            "schema": EXACT_TARGET_COMPILER_OUTPUT_TRAINING_SCOPE_SCHEMA,
            "trainable_qkvo_factor_parameters": (
                self.transaction.qkvo_handle.report.adapter_parameter_count
            ),
            "trainable_gate_up_down_factor_parameters": (
                self.transaction.swiglu_handle.report.adapter_parameter_count
            ),
            "base_parameter_updates_allowed": False,
            "accepted_objective_domain": (
                "causal-model-output-actions-logits-and-free-rollouts"
            ),
            "weight_matching_allowed": False,
            "activation_matching_allowed": False,
            "hidden_state_matching_allowed": False,
            "parent_runtime_reference_count": 0,
            "optimizer_state_is_temporary": True,
            "optimizer_state_published": False,
        }
        if self.transaction.lexical_handle is not None:
            payload.update(
                {
                    "trainable_lexical_factor_parameters": (
                        self.transaction.lexical_handle.report.adapter_parameter_count
                    ),
                    "trainable_lexical_factor_tensor_count": 4,
                    "role_complete_lexical_boundary": True,
                }
            )
        payload["training_scope_sha256"] = _canonical_sha256(payload)
        return payload


class ExactTargetCompilerTransaction:
    """Atomic target-native QKVO, Gate/Up/Down, and optional lexical transaction."""

    def __init__(
        self,
        *,
        model: nn.Module,
        exact_unique_parameters: int,
        topology: _DenseGQATopology,
        native_before: NativeDenseGQACertificate,
        native_child_state_sha256: str,
        native_tensor_guards: tuple[tuple[str, str, Tensor, str], ...],
        qkvo_handle: InstalledDenseGQAQKVOResidualAdapter,
        swiglu_handle: InstalledDenseSwiGLUProjectionResidualAdapter,
        stage: ProgressiveDenseGQAStage | None,
        parent_compiler_report_sha256: str | None,
        lexical_handle: InstalledDenseLexicalResidualAdapter | None = None,
        native_runtime_snapshot: _ModelRuntimeSnapshot | None = None,
    ) -> None:
        self.model = model
        self.exact_unique_parameters = exact_unique_parameters
        self.topology = topology
        self.native_before = native_before
        self._native_runtime_snapshot = native_runtime_snapshot
        self.native_child_state_sha256 = _sha256(
            native_child_state_sha256,
            "native_child_state_sha256",
        )
        self._native_tensor_guards = tuple(native_tensor_guards)
        if not self._native_tensor_guards:
            raise ProgressiveTargetBridgeError(
                "Compiler transaction has no native tensor guards."
            )
        self.qkvo_handle = qkvo_handle
        self.swiglu_handle = swiglu_handle
        self.lexical_handle = lexical_handle
        self.stage = stage
        self.parent_compiler_report_sha256 = parent_compiler_report_sha256
        self.publication_report: dict[str, Any] | None = None
        self._zero_step_identity_at_installation = (
            qkvo_handle.adapter_registry.starts_as_exact_zero_correction
            and swiglu_handle.registry.starts_as_exact_zero_correction
            and (
                lexical_handle is None
                or lexical_handle.registry.starts_as_exact_zero_correction
            )
        )
        if not self._zero_step_identity_at_installation:
            raise ProgressiveTargetBridgeError(
                "Compiler transaction did not start as an exact identity."
            )

    @property
    def is_live(self) -> bool:
        return (
            self.qkvo_handle.is_applied
            and self.swiglu_handle.is_applied
            and (self.lexical_handle is None or self.lexical_handle.is_applied)
        )

    @property
    def starts_as_zero_step_identity(self) -> bool:
        return self._zero_step_identity_at_installation

    def assert_live(self) -> None:
        self.qkvo_handle.assert_live()
        self.swiglu_handle.assert_live()
        if self.lexical_handle is not None:
            self.lexical_handle.assert_live()

    def assert_structurally_live(self) -> None:
        """Validate transaction ownership while permitting poisoned factors."""

        self.qkvo_handle.assert_structurally_live()
        self.swiglu_handle.assert_structurally_live()
        if self.lexical_handle is not None:
            self.lexical_handle.assert_structurally_live()

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        values = (
            *self.qkvo_handle.output_trainable_parameters(),
            *self.swiglu_handle.output_trainable_parameters(),
        )
        if self.lexical_handle is not None:
            values = (*values, *self.lexical_handle.output_trainable_parameters())
        return values

    def output_training_scope(self) -> ExactTargetCompilerOutputTrainingScope:
        return ExactTargetCompilerOutputTrainingScope(self)

    def _live_factor_state_sha256(self) -> str:
        """Hash every live factor tensor by its registered target-side name."""

        self.assert_live()
        factors = self.output_trainable_parameters()
        factor_ids = {id(value) for value in factors}
        rows: list[dict[str, Any]] = []
        first_name_by_id: dict[int, str] = {}
        observed_ids: set[int] = set()
        for _kind, name, value in _named_model_tensors(self.model):
            identity = id(value)
            if identity not in factor_ids:
                continue
            observed_ids.add(identity)
            first_name = first_name_by_id.setdefault(identity, name)
            rows.append(
                {
                    "name": name,
                    "alias_of": first_name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "tensor_sha256": _dense_tensor_sha256(value),
                }
            )
        if observed_ids != factor_ids or not rows:
            raise ProgressiveTargetPublicationError(
                "Live compiler factors are not fully registered on the child."
            )
        return _canonical_sha256(
            {
                "schema": "neural-forge-live-target-factor-state-v1",
                "unique_factor_tensor_count": len(factor_ids),
                "named_factor_tensor_count": len(rows),
                "factors": rows,
            }
        )

    def _assert_native_tensor_guards(self) -> None:
        """Fail closed if frozen native child bytes or ownership changed."""

        registered_ids = {
            id(value) for _kind, _name, value in _named_model_tensors(self.model)
        }
        for kind, name, value, digest in self._native_tensor_guards:
            if id(value) not in registered_ids or _dense_tensor_sha256(value) != digest:
                raise ProgressiveTargetPublicationError(
                    "Native child tensor changed before experimental materialization "
                    f"at {kind}:{name}."
                )

    def experimental_recovery_receipt(
        self,
        *,
        experiment_contract: Mapping[str, Any],
    ) -> ExperimentalRecoveryMaterializationReceipt:
        """Create a receipt bound to this live transaction and native child."""

        if (
            not isinstance(experiment_contract, Mapping)
            or experiment_contract.get("schema")
            != EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA
            or experiment_contract.get("publication_authorized") is not False
            or experiment_contract.get("sealed_data_accessed") is not False
            or experiment_contract.get("blank_control_state_sha256")
            != self.native_child_state_sha256
            or experiment_contract.get("native_child_state_sha256")
            != self.native_child_state_sha256
            or experiment_contract.get("exact_unique_parameters")
            != self.exact_unique_parameters
        ):
            raise ProgressiveTargetPublicationError(
                "Experimental recovery contract does not bind this native child."
            )
        self._assert_native_tensor_guards()
        return ExperimentalRecoveryMaterializationReceipt(
            experiment_contract_sha256=_canonical_sha256(experiment_contract),
            factor_state_sha256=_sha256(
                experiment_contract.get("factor_state_sha256"),
                "experiment factor_state_sha256",
            ),
            live_factor_state_sha256=self._live_factor_state_sha256(),
            native_child_state_sha256=self.native_child_state_sha256,
            exact_unique_parameters=self.exact_unique_parameters,
            factor_payload_schema=str(
                experiment_contract.get("factor_payload_schema", "")
            ),
        )

    def installation_report(self) -> dict[str, Any]:
        self.assert_live()
        qkvo = self.qkvo_handle.report
        swiglu = self.swiglu_handle.report
        lexical = None if self.lexical_handle is None else self.lexical_handle.report
        temporary = (
            qkvo.adapter_parameter_count
            + swiglu.adapter_parameter_count
            + (0 if lexical is None else lexical.adapter_parameter_count)
        )
        installed = exact_parameter_report(self.model)["unique_parameter_count"]
        if installed != self.exact_unique_parameters + temporary:
            raise ProgressiveTargetBridgeError(
                "Combined temporary factor accounting does not match the live model."
            )
        payload = {
            "schema": EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA,
            "exact_native_parameters": self.exact_unique_parameters,
            "temporary_qkvo_factor_parameters": qkvo.adapter_parameter_count,
            "temporary_gate_up_down_factor_parameters": (
                swiglu.adapter_parameter_count
            ),
            "temporary_factor_parameters": temporary,
            "temporary_installed_parameters": installed,
            "published_parameters_if_promoted": self.exact_unique_parameters,
            "published_temporary_factor_parameters": 0,
            "zero_step_identity": self.starts_as_zero_step_identity,
            "native_before": self.native_before.as_dict(),
            "dense_qkvo_installation": qkvo.as_dict(),
            "dense_gate_up_down_installation": swiglu.as_dict(),
            "stage": None if self.stage is None else self.stage.as_dict(),
            "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
            "role_complete_attention_qkvo": True,
            "role_complete_mlp_gate_up_down": True,
            "target_native_factors_only": True,
            "output_and_free_rollout_training_only": True,
            "internal_numeric_matching_required": False,
            "checkpoint_publication_allowed_while_installed": False,
            "capability_proven": False,
        }
        if lexical is not None:
            payload.update(
                {
                    "temporary_lexical_factor_parameters": (
                        lexical.adapter_parameter_count
                    ),
                    "dense_lexical_installation": lexical.as_dict(),
                    "role_complete_lexical_boundary": True,
                    "lexical_embedding_and_lm_head_untied": True,
                }
            )
        payload["installation_transaction_sha256"] = _canonical_sha256(payload)
        return payload

    @staticmethod
    def _enter_guided_owner_suspensions(
        stack: ExitStack,
        handles: Sequence[tuple[str, Any]],
    ) -> tuple[str, ...]:
        """Hide later guided owners while an older transaction restores its baseline."""

        entered: list[str] = []
        identities: set[int] = set()
        for name, handle in handles:
            if id(handle) in identities or not bool(
                getattr(handle, "is_applied", False)
            ):
                continue
            identities.add(id(handle))
            suspend = getattr(
                handle,
                "suspend_guided_registry_owner_for_nested_publication",
                None,
            )
            if callable(suspend):
                stack.enter_context(suspend())
                entered.append(name)
        return tuple(entered)

    def rollback(self) -> NativeDenseGQACertificate:
        if self.lexical_handle is not None and self.lexical_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (
                        ("qkvo", self.qkvo_handle),
                        ("gate-up-down", self.swiglu_handle),
                    ),
                )
                self.lexical_handle.rollback()
        if self.qkvo_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (("gate-up-down", self.swiglu_handle),),
                )
                self.qkvo_handle.rollback()
        if self.swiglu_handle.is_applied:
            self.swiglu_handle.rollback()
        restored = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
            require_dense_swiglu=True,
        )
        if restored.as_dict() != self.native_before.as_dict():
            raise ProgressiveTargetPublicationError(
                "Compiler rollback did not restore the exact native child."
            )
        return restored

    def _restore_failed_publication(self) -> NativeDenseGQACertificate:
        # A lexical bake is first on the success path.  If it never completed,
        # reject its live wrapper before the older transactions whose state-key
        # topology formed its installation baseline.
        if self.lexical_handle is not None and self.lexical_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (
                        ("qkvo", self.qkvo_handle),
                        ("gate-up-down", self.swiglu_handle),
                    ),
                )
                self.lexical_handle.rollback()
        # QKVO is baked first.  If it is still live, reject it before removing
        # the MLP factors because its installation baseline includes them.
        if self.qkvo_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (("gate-up-down", self.swiglu_handle),),
                )
                self.qkvo_handle.rollback()
            if self.swiglu_handle.is_applied:
                self.swiglu_handle.rollback()
        else:
            if self.swiglu_handle.is_baked:
                self.swiglu_handle.restore_prepublication_native()
            elif self.swiglu_handle.is_applied:
                self.swiglu_handle.rollback()
            if self.qkvo_handle.is_baked:
                self.qkvo_handle.restore_prepublication_native(
                    expected_model_parameter_count=self.exact_unique_parameters
                )
        # If lexical publication completed, restore it only after the older
        # temporary operators have returned the model to its final native
        # budget.  Its own installation baseline included both of them.
        if self.lexical_handle is not None and self.lexical_handle.is_baked:
            self.lexical_handle.restore_prepublication_native(
                expected_model_parameter_count=self.exact_unique_parameters
            )
        restored = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
            require_dense_swiglu=True,
        )
        if restored.as_dict() != self.native_before.as_dict():
            raise ProgressiveTargetPublicationError(
                "Failed publication could not restore the exact native child."
            )
        return restored

    def restore_prepublication_native(self) -> NativeDenseGQACertificate:
        """Undo live or already-baked factors after a later publication failure."""

        restored = self._restore_failed_publication()
        self.publication_report = None
        return restored

    @contextmanager
    def experimental_recovery_materialization(
        self,
        receipt: ExperimentalRecoveryMaterializationReceipt,
        *,
        experiment_contract: Mapping[str, Any],
    ) -> Iterator[bytes]:
        """Temporarily expose a dense, exact-size, unaccepted recovery init.

        ``self.model`` is dense while canonical built-in report bytes are
        yielded. The context is for serialization only: recovery must freshly
        reload the serialized child and train that copy. Exit restores every
        guarded native tensor and supported runtime object from private
        pre-bake snapshots.
        """

        if not isinstance(receipt, ExperimentalRecoveryMaterializationReceipt):
            raise ProgressiveTargetPublicationError(
                "receipt must be ExperimentalRecoveryMaterializationReceipt."
            )
        if receipt.exact_unique_parameters != self.exact_unique_parameters:
            raise ProgressiveTargetPublicationError(
                "Experimental receipt does not match the exact child budget."
            )
        if (
            not isinstance(experiment_contract, Mapping)
            or experiment_contract.get("schema")
            != EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA
            or _canonical_sha256(experiment_contract)
            != receipt.experiment_contract_sha256
            or experiment_contract.get("factor_state_sha256")
            != receipt.factor_state_sha256
            or experiment_contract.get("factor_payload_schema")
            != receipt.factor_payload_schema
            or experiment_contract.get("native_child_state_sha256")
            != receipt.native_child_state_sha256
            or experiment_contract.get("blank_control_state_sha256")
            != receipt.native_child_state_sha256
            or experiment_contract.get("exact_unique_parameters")
            != receipt.exact_unique_parameters
            or experiment_contract.get("publication_authorized") is not False
            or experiment_contract.get("sealed_data_accessed") is not False
        ):
            raise ProgressiveTargetPublicationError(
                "Experimental receipt does not authenticate its recovery contract."
            )
        if receipt.native_child_state_sha256 != self.native_child_state_sha256:
            raise ProgressiveTargetPublicationError(
                "Experimental receipt belongs to another native child state."
            )
        native_runtime_snapshot = self._native_runtime_snapshot
        if (
            native_runtime_snapshot is None
            or native_runtime_snapshot.root is not self.model
        ):
            raise ProgressiveTargetPublicationError(
                "Experimental materialization lacks its pre-install native runtime "
                "snapshot."
            )
        if native_runtime_snapshot.plain_runtime.unsupported_paths:
            raise ProgressiveTargetPublicationError(
                "Experimental materialization cannot guard unsupported native "
                "runtime state at "
                f"{native_runtime_snapshot.plain_runtime.unsupported_paths[0]}."
            )
        self.assert_live()
        self._assert_native_tensor_guards()
        live_factor_state_sha256 = self._live_factor_state_sha256()
        if receipt.live_factor_state_sha256 != live_factor_state_sha256:
            raise ProgressiveTargetPublicationError(
                "Experimental receipt does not bind the live factor tensors."
            )
        try:
            contract_payload = json.loads(
                json.dumps(
                    experiment_contract,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as error:
            raise ProgressiveTargetPublicationError(
                "Experimental recovery contract is not canonical JSON."
            ) from error
        baseline_snapshot = _snapshot_named_model_tensors(self.model)
        materialized_state_sha256: str | None = None
        materialized_runtime_snapshot: _ModelRuntimeSnapshot | None = None
        tensor_versions: tuple[tuple[Any, ...], ...] | None = None
        mutated = False
        try:
            if self.lexical_handle is None:
                lexical_bake = None
                lexical_suspensions: tuple[str, ...] = ()
            else:
                with ExitStack() as stack:
                    lexical_suspensions = self._enter_guided_owner_suspensions(
                        stack,
                        (
                            ("qkvo", self.qkvo_handle),
                            ("gate-up-down", self.swiglu_handle),
                        ),
                    )
                    lexical_bake = self.lexical_handle.bake_for_publication()
            with ExitStack() as stack:
                qkvo_suspensions = self._enter_guided_owner_suspensions(
                    stack,
                    (("gate-up-down", self.swiglu_handle),),
                )
                qkvo_bake = self.qkvo_handle.bake_for_publication()
            swiglu_bake = self.swiglu_handle.bake_for_publication()
            native_after = certify_native_dense_gqa_target(
                self.model,
                exact_unique_parameters=self.exact_unique_parameters,
                layout=self.topology.layout,
                stage=self.stage,
                require_dense_swiglu=True,
            )
            if (
                native_after.structure_sha256 != self.native_before.structure_sha256
                or native_after.state_key_sha256 != self.native_before.state_key_sha256
                or native_after.state_key_count != self.native_before.state_key_count
            ):
                raise ProgressiveTargetPublicationError(
                    "Experimental dense bake changed native state-key topology."
                )
            payload = {
                "schema": EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA,
                "status": "experimental-unaccepted",
                "exact_unique_parameters": self.exact_unique_parameters,
                "exact_budget_match": True,
                "native_before": self.native_before.as_dict(),
                "native_materialized": native_after.as_dict(),
                "dense_qkvo_bake": qkvo_bake.as_dict(),
                "dense_gate_up_down_bake": swiglu_bake.as_dict(),
                "experimental_recovery_contract": contract_payload,
                "experimental_recovery_receipt": receipt.as_dict(),
                "stage": None if self.stage is None else self.stage.as_dict(),
                "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
                "live_factor_state_sha256": live_factor_state_sha256,
                "native_child_state_sha256": self.native_child_state_sha256,
                "materialized_registered_adapter_parameter_count": 0,
                "materialized_optimizer_parameter_count": 0,
                "materialized_registered_source_sized_sidecar_parameter_count": 0,
                "arbitrary_python_runtime_reference_audit_performed": False,
                "recursive_module_training_state_guarded": True,
                "parameter_requires_grad_state_guarded": True,
                "parameter_gradient_identity_and_bytes_guarded": True,
                "parameter_and_buffer_owner_alias_identity_guarded": True,
                "module_hook_registry_identity_guarded": True,
                "plain_python_runtime_reference_ownership_guarded": True,
                "supported_plain_python_mutable_graph_guarded": True,
                "supported_plain_python_aliases_and_cycles_guarded": True,
                "unsupported_plain_python_mutable_state_fails_closed": True,
                "callable_closure_and_opaque_runtime_internal_state_audited": False,
                "plain_python_runtime_mutation_proven_absent": False,
                "native_state_key_topology_restored": True,
                "native_registered_tensor_topology_directly_enumerated": True,
                "authoritative_snapshot_uses_state_dict": False,
                "authoritative_hash_uses_state_dict": False,
                "standalone_structure_materialized": True,
                "serialization_only_context": True,
                "state_dict_hook_output_completeness_proven": False,
                "serialization_api_output_revalidated_by_bridge": False,
                "caller_must_serialize_all_registered_tensors": True,
                "mutating_or_training_in_context_forbidden": True,
                "endpoint_tensor_version_and_byte_guard": True,
                "transient_mutation_proven_absent": False,
                "serialized_state_must_match_materialized_state_sha256": True,
                "fresh_reload_and_recovery_required": True,
                "publication_authorized": False,
                "accepted_artifact": False,
                "sealed_data_accessed": False,
                "behavior_capability_proven": False,
                "recovery_training_required": True,
                "external_factor_pack_revalidated_by_bridge": False,
                "external_factor_pack_receipt_bound": False,
                "contract_factor_state_sha256_is_external_label_only": True,
                "live_factor_state_sha256_is_bridge_authenticated": True,
            }
            if lexical_bake is not None:
                payload.update(
                    {
                        "dense_lexical_bake": lexical_bake.as_dict(),
                        "materialized_lexical_adapter_parameter_count": 0,
                        "native_lexical_state_key_topology_restored": True,
                        "guided_registry_suspended_for_lexical_bake": (
                            "qkvo" in lexical_suspensions
                        ),
                    }
                )
            if "gate-up-down" in (*lexical_suspensions, *qkvo_suspensions):
                payload.update(
                    {
                        "guided_gate_up_down_registry_suspended_for_lexical_bake": (
                            "gate-up-down" in lexical_suspensions
                        ),
                        "guided_gate_up_down_registry_suspended_for_qkvo_bake": (
                            "gate-up-down" in qkvo_suspensions
                        ),
                        "nested_materialization_bake_order": [
                            "lexical",
                            "qkvo-inherited-then-guided",
                            "gate-up-down-inherited-then-guided",
                        ],
                    }
                )
            materialized_state_sha256 = _dense_model_state_sha256(self.model)
            payload["materialized_native_state_sha256"] = (
                materialized_state_sha256
            )
            materialized_runtime_snapshot = _capture_model_runtime_snapshot(
                self.model,
                name="materialized recovery runtime",
            )
            if materialized_runtime_snapshot.plain_runtime.unsupported_paths:
                raise ProgressiveTargetPublicationError(
                    "Experimental materialization produced unsupported runtime "
                    "state at "
                    f"{materialized_runtime_snapshot.plain_runtime.unsupported_paths[0]}."
                )
            payload["materialization_transaction_sha256"] = _canonical_sha256(
                payload
            )
            canonical_report = _canonical_json_report_bytes(
                payload,
                hash_field="materialization_transaction_sha256",
            )
            tensor_versions = _named_tensor_version_signature(self.model)
            yield canonical_report
        finally:
            active_error = sys.exc_info()[1]
            if tensor_versions is not None:
                try:
                    mutated = (
                        tensor_versions != _named_tensor_version_signature(self.model)
                        or materialized_state_sha256 is None
                        or _dense_model_state_sha256(self.model)
                        != materialized_state_sha256
                        or materialized_runtime_snapshot is None
                        or not _model_runtime_snapshot_matches(
                            self.model,
                            materialized_runtime_snapshot,
                        )
                    )
                except BaseException:
                    mutated = True
            cleanup_errors: list[BaseException] = []
            cleanup_steps = []
            if materialized_runtime_snapshot is not None:
                cleanup_steps.append(
                    lambda: _restore_model_runtime_snapshot(
                        materialized_runtime_snapshot
                    )
                )
            cleanup_steps.extend(
                (
                    self._restore_failed_publication,
                    lambda: _restore_named_model_tensors(
                        self.model,
                        baseline_snapshot,
                    ),
                    lambda: _restore_model_runtime_snapshot(
                        native_runtime_snapshot
                    ),
                )
            )
            for cleanup_step in cleanup_steps:
                try:
                    cleanup_step()
                except BaseException as error:
                    cleanup_errors.append(error)
            try:
                if (
                    _dense_model_state_sha256(self.model)
                    != self.native_child_state_sha256
                ):
                    raise ProgressiveTargetPublicationError(
                        "Experimental cleanup did not restore native child bytes."
                    )
                if not _model_runtime_snapshot_matches(
                    self.model,
                    native_runtime_snapshot,
                ):
                    raise ProgressiveTargetPublicationError(
                        "Experimental cleanup did not restore native runtime ownership."
                    )
            except BaseException as error:
                cleanup_errors.append(error)
            self.publication_report = None
            if cleanup_errors:
                cleanup_error = cleanup_errors[0]
                for secondary_error in cleanup_errors[1:]:
                    cleanup_error.add_note(
                        "Additional experimental cleanup failure: "
                        f"{secondary_error!r}"
                    )
                if active_error is not None:
                    active_error.add_note(
                        "Experimental materialization cleanup also failed: "
                        f"{cleanup_error!r}"
                    )
                    raise cleanup_error from active_error
                raise cleanup_error
            if mutated:
                if active_error is not None:
                    active_error.add_note(
                        "Experimental materialization was also mutated before cleanup."
                    )
                else:
                    raise ProgressiveTargetPublicationError(
                        "Experimental materialization was mutated instead of serialized."
                    )

    def bake_for_publication(
        self,
        behavior_gate: ExternalBehaviorGateReceipt,
    ) -> dict[str, Any]:
        if not isinstance(behavior_gate, ExternalBehaviorGateReceipt):
            raise ProgressiveTargetPublicationError(
                "behavior_gate must be ExternalBehaviorGateReceipt."
            )
        behavior_gate.require_publication_approval()
        self.assert_live()
        try:
            if self.lexical_handle is None:
                lexical_bake = None
                lexical_suspensions: tuple[str, ...] = ()
            else:
                with ExitStack() as stack:
                    lexical_suspensions = self._enter_guided_owner_suspensions(
                        stack,
                        (
                            ("qkvo", self.qkvo_handle),
                            ("gate-up-down", self.swiglu_handle),
                        ),
                    )
                    lexical_bake = self.lexical_handle.bake_for_publication()
            with ExitStack() as stack:
                qkvo_suspensions = self._enter_guided_owner_suspensions(
                    stack,
                    (("gate-up-down", self.swiglu_handle),),
                )
                qkvo_bake = self.qkvo_handle.bake_for_publication()
            swiglu_bake = self.swiglu_handle.bake_for_publication()
            native_after = certify_native_dense_gqa_target(
                self.model,
                exact_unique_parameters=self.exact_unique_parameters,
                layout=self.topology.layout,
                stage=self.stage,
                require_dense_swiglu=True,
            )
            if (
                native_after.structure_sha256 != self.native_before.structure_sha256
                or native_after.state_key_sha256 != self.native_before.state_key_sha256
                or native_after.state_key_count != self.native_before.state_key_count
            ):
                raise ProgressiveTargetPublicationError(
                    "Combined bake did not restore native state-key topology."
                )
        except Exception:
            self._restore_failed_publication()
            raise
        try:
            payload = {
                "schema": EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA,
                "exact_unique_parameters": self.exact_unique_parameters,
                "exact_budget_match": True,
                "native_before": self.native_before.as_dict(),
                "native_after": native_after.as_dict(),
                "dense_qkvo_bake": qkvo_bake.as_dict(),
                "dense_gate_up_down_bake": swiglu_bake.as_dict(),
                "external_behavior_gate": behavior_gate.as_dict(),
                "stage": None if self.stage is None else self.stage.as_dict(),
                "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
                "published_adapter_parameter_count": 0,
                "published_optimizer_parameter_count": 0,
                "published_parent_runtime_reference_count": 0,
                "published_source_sized_sidecar_parameter_count": 0,
                "native_state_key_topology_restored": True,
                "output_action_gate_required": True,
                "free_running_rollout_gate_required": True,
                "weight_or_activation_similarity_can_promote": False,
                "behavior_gate_externally_attested": True,
                "behavior_evidence_revalidated_by_this_module": False,
                "capability_proven_by_this_module": False,
                "save_reload_artifact_gate_still_required": True,
            }
            if lexical_bake is not None:
                payload.update(
                    {
                        "dense_lexical_bake": lexical_bake.as_dict(),
                        "published_lexical_adapter_parameter_count": 0,
                        "native_lexical_state_key_topology_restored": True,
                        "role_complete_lexical_boundary": True,
                        "guided_registry_suspended_for_lexical_bake": (
                            "qkvo" in lexical_suspensions
                        ),
                    }
                )
            if "gate-up-down" in (*lexical_suspensions, *qkvo_suspensions):
                payload.update(
                    {
                        "guided_gate_up_down_registry_suspended_for_lexical_bake": (
                            "gate-up-down" in lexical_suspensions
                        ),
                        "guided_gate_up_down_registry_suspended_for_qkvo_bake": (
                            "gate-up-down" in qkvo_suspensions
                        ),
                        "nested_publication_bake_order": [
                            "lexical",
                            "qkvo-inherited-then-guided",
                            "gate-up-down-inherited-then-guided",
                        ],
                    }
                )
            payload["publication_transaction_sha256"] = _canonical_sha256(payload)
        except Exception:
            self._restore_failed_publication()
            raise
        self.publication_report = payload
        return payload

    def __enter__(self) -> ExactTargetCompilerTransaction:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        failed = exc_type is not None
        del exc_type, exc, traceback
        if self.is_live:
            self.rollback()
        elif failed and (
            self.qkvo_handle.is_baked
            or self.swiglu_handle.is_baked
            or (self.lexical_handle is not None and self.lexical_handle.is_baked)
        ):
            self.restore_prepublication_native()


def install_exact_target_qkvo_bridge(
    model: nn.Module,
    *,
    exact_unique_parameters: int,
    sites: Sequence[DenseGQASite],
    rank: int,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    owner_name: str = "dense_gqa_qkvo_residual_registry",
    initialization_seed: int = 0,
    parent_compiler_report_sha256: str | None = None,
) -> ExactTargetQKVOPublicationSession:
    """Install temporary QKVO factors only after exact target certification."""

    expected = _positive_int(exact_unique_parameters, "exact_unique_parameters")
    if parent_compiler_report_sha256 is not None:
        parent_compiler_report_sha256 = _sha256(
            parent_compiler_report_sha256,
            "parent_compiler_report_sha256",
        )
    topology = _resolve_dense_gqa_topology(model, layout)
    native = certify_native_dense_gqa_target(
        model,
        exact_unique_parameters=expected,
        layout=topology.layout,
        stage=stage,
    )
    try:
        handle = install_dense_gqa_qkvo_residual_adapter(
            model,
            topology.attention_modules,
            geometry=topology.geometry,
            sites=sites,
            rank=rank,
            projection_names=topology.projection_names,
            owner_name=owner_name,
            initialization_seed=initialization_seed,
        )
    except DenseGQAQKVOResidualTopologyError as error:
        raise ProgressiveTargetBridgeError(str(error)) from error
    if handle.report.model_parameter_count_before != expected:
        handle.rollback()
        raise ProgressiveTargetBridgeError(
            "Dense adapter installation disagrees with exact target accounting."
        )
    return ExactTargetQKVOPublicationSession(
        model=model,
        exact_unique_parameters=expected,
        topology=topology,
        native_before=native,
        handle=handle,
        stage=stage,
        parent_compiler_report_sha256=parent_compiler_report_sha256,
    )


def install_exact_target_compiler_transaction(
    model: nn.Module,
    *,
    exact_unique_parameters: int,
    qkvo_sites: Sequence[DenseGQASite],
    qkvo_rank: int,
    swiglu_layer_indices: Sequence[int],
    swiglu_rank: int,
    lexical_rank: int | None = None,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    qkvo_owner_name: str = "dense_gqa_qkvo_residual_registry",
    swiglu_owner_name: str = "dense_swiglu_projection_residual_registry",
    lexical_owner_name: str = "dense_lexical_residual_registry",
    initialization_seed: int = 0,
    parent_compiler_report_sha256: str | None = None,
) -> ExactTargetCompilerTransaction:
    """Open one exact-budget target-native compiler-factor transaction."""

    expected = _positive_int(exact_unique_parameters, "exact_unique_parameters")
    if parent_compiler_report_sha256 is not None:
        parent_compiler_report_sha256 = _sha256(
            parent_compiler_report_sha256,
            "parent_compiler_report_sha256",
        )
    topology = _resolve_dense_gqa_topology(
        model,
        layout,
        require_dense_swiglu=True,
    )
    native_child_state_sha256 = _dense_model_state_sha256(model)
    native_tensor_guards = _capture_native_tensor_guards(model)
    native_runtime_snapshot = _capture_model_runtime_snapshot(
        model,
        name="native child runtime",
    )
    native = certify_native_dense_gqa_target(
        model,
        exact_unique_parameters=expected,
        layout=topology.layout,
        stage=stage,
        require_dense_swiglu=True,
    )
    try:
        swiglu_handle = install_dense_swiglu_projection_residual_adapter(
            model,
            topology.mlp_modules,
            layer_indices=swiglu_layer_indices,
            hidden_size=topology.layout.hidden_size,
            intermediate_size=topology.layout.intermediate_size,
            rank=swiglu_rank,
            projection_names=topology.swiglu_projection_names,
            owner_name=swiglu_owner_name,
            initialization_seed=initialization_seed,
        )
    except DenseSwiGLUProjectionResidualTopologyError as error:
        raise ProgressiveTargetBridgeError(str(error)) from error
    try:
        qkvo_handle = install_dense_gqa_qkvo_residual_adapter(
            model,
            topology.attention_modules,
            geometry=topology.geometry,
            sites=qkvo_sites,
            rank=qkvo_rank,
            projection_names=topology.projection_names,
            owner_name=qkvo_owner_name,
            initialization_seed=initialization_seed + 1,
        )
        if (
            swiglu_handle.report.model_parameter_count_before != expected
            or qkvo_handle.report.model_parameter_count_before
            != swiglu_handle.report.model_parameter_count_installed
        ):
            raise ProgressiveTargetBridgeError(
                "Combined adapter installation disagrees with exact target accounting."
            )
        lexical_handle = None
        if lexical_rank is not None:
            lexical_handle = install_dense_lexical_residual_adapter(
                model,
                rank=lexical_rank,
                owner_name=lexical_owner_name,
                initialization_seed=initialization_seed + 2,
            )
            if (
                lexical_handle.report.model_parameter_count_before
                != qkvo_handle.report.model_parameter_count_installed
                or lexical_handle.report.model_parameter_count_installed
                != (
                    qkvo_handle.report.model_parameter_count_installed
                    + lexical_handle.report.adapter_parameter_count
                )
            ):
                raise ProgressiveTargetBridgeError(
                    "Lexical adapter installation disagrees with combined accounting."
                )
    except DenseLexicalResidualError as error:
        if "lexical_handle" in locals() and lexical_handle is not None:
            if lexical_handle.is_applied:
                lexical_handle.rollback()
        if "qkvo_handle" in locals() and qkvo_handle.is_applied:
            qkvo_handle.rollback()
        if swiglu_handle.is_applied:
            swiglu_handle.rollback()
        raise ProgressiveTargetBridgeError(str(error)) from error
    except Exception:
        if (
            "lexical_handle" in locals()
            and lexical_handle is not None
            and lexical_handle.is_applied
        ):
            lexical_handle.rollback()
        if "qkvo_handle" in locals() and qkvo_handle.is_applied:
            qkvo_handle.rollback()
        if swiglu_handle.is_applied:
            swiglu_handle.rollback()
        raise
    transaction = ExactTargetCompilerTransaction(
        model=model,
        exact_unique_parameters=expected,
        topology=topology,
        native_before=native,
        native_child_state_sha256=native_child_state_sha256,
        native_tensor_guards=native_tensor_guards,
        native_runtime_snapshot=native_runtime_snapshot,
        qkvo_handle=qkvo_handle,
        swiglu_handle=swiglu_handle,
        lexical_handle=lexical_handle,
        stage=stage,
        parent_compiler_report_sha256=parent_compiler_report_sha256,
    )
    try:
        transaction.installation_report()
        return transaction
    except Exception:
        if transaction.is_live:
            transaction.rollback()
        raise


def install_parent_compiler_result_qkvo_bridge(
    result: HFParentCompilerResult,
    *,
    sites: Sequence[DenseGQASite],
    rank: int,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    owner_name: str = "dense_gqa_qkvo_residual_registry",
    initialization_seed: int = 0,
) -> ExactTargetQKVOPublicationSession:
    """Attach the native QKVO transaction to an accepted parent compilation."""

    if not isinstance(result, HFParentCompilerResult):
        raise ProgressiveTargetBridgeError("result must be HFParentCompilerResult.")
    report = result.report
    if (
        not report.target_budget_exact
        or not report.bootstrap_accepted
        or report.target_unique_parameter_count != report.target_parameter_budget
    ):
        raise ProgressiveTargetBridgeError(
            "Parent compiler result is not an accepted exact-budget child."
        )
    report_payload = report.as_dict()
    return install_exact_target_qkvo_bridge(
        result.child_model,
        exact_unique_parameters=report.target_unique_parameter_count,
        sites=sites,
        rank=rank,
        layout=layout,
        stage=stage,
        owner_name=owner_name,
        initialization_seed=initialization_seed,
        parent_compiler_report_sha256=_canonical_sha256(report_payload),
    )


def install_parent_compiler_result_target_native_transaction(
    result: HFParentCompilerResult,
    *,
    qkvo_sites: Sequence[DenseGQASite],
    qkvo_rank: int,
    swiglu_layer_indices: Sequence[int],
    swiglu_rank: int,
    lexical_rank: int | None = None,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    qkvo_owner_name: str = "dense_gqa_qkvo_residual_registry",
    swiglu_owner_name: str = "dense_swiglu_projection_residual_registry",
    lexical_owner_name: str = "dense_lexical_residual_registry",
    initialization_seed: int = 0,
) -> ExactTargetCompilerTransaction:
    """Attach the full target-native transaction to an accepted compilation."""

    if not isinstance(result, HFParentCompilerResult):
        raise ProgressiveTargetBridgeError("result must be HFParentCompilerResult.")
    report = result.report
    if (
        not report.target_budget_exact
        or not report.bootstrap_accepted
        or report.target_unique_parameter_count != report.target_parameter_budget
    ):
        raise ProgressiveTargetBridgeError(
            "Parent compiler result is not an accepted exact-budget child."
        )
    return install_exact_target_compiler_transaction(
        result.child_model,
        exact_unique_parameters=report.target_unique_parameter_count,
        qkvo_sites=qkvo_sites,
        qkvo_rank=qkvo_rank,
        swiglu_layer_indices=swiglu_layer_indices,
        swiglu_rank=swiglu_rank,
        lexical_rank=lexical_rank,
        layout=layout,
        stage=stage,
        qkvo_owner_name=qkvo_owner_name,
        swiglu_owner_name=swiglu_owner_name,
        lexical_owner_name=lexical_owner_name,
        initialization_seed=initialization_seed,
        parent_compiler_report_sha256=_canonical_sha256(report.as_dict()),
    )


__all__ = [
    "EXACT_TARGET_QKVO_INSTALLATION_SCHEMA",
    "EXACT_TARGET_QKVO_PUBLICATION_SCHEMA",
    "EXACT_TARGET_COMPILER_OUTPUT_TRAINING_SCOPE_SCHEMA",
    "EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA",
    "EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA",
    "EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA",
    "EXTERNAL_BEHAVIOR_GATE_SCHEMA",
    "EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA",
    "EXPERIMENTAL_RECOVERY_MATERIALIZATION_RECEIPT_SCHEMA",
    "NATIVE_TARGET_CERTIFICATE_SCHEMA",
    "PROGRESSIVE_TARGET_PLAN_SCHEMA",
    "QWEN3_1P4B_EXACT_PARAMETERS",
    "QWEN3_4B_EXACT_PARAMETERS",
    "QWEN3_600M_EXACT_PARAMETERS",
    "SMOLLM2_135M_EXACT_PARAMETERS",
    "DenseGQATargetShape",
    "ExactTargetCompilerOutputTrainingScope",
    "ExactTargetCompilerTransaction",
    "ExactTargetQKVOPublicationSession",
    "ExternalBehaviorGateReceipt",
    "ExperimentalRecoveryMaterializationReceipt",
    "NativeDenseGQACertificate",
    "ProgressiveDenseGQAPlan",
    "ProgressiveDenseGQAStage",
    "ProgressiveTargetBridgeError",
    "ProgressiveTargetPublicationError",
    "certify_native_dense_gqa_target",
    "default_qwen4b_progressive_target_plan",
    "install_exact_target_qkvo_bridge",
    "install_exact_target_compiler_transaction",
    "install_parent_compiler_result_qkvo_bridge",
    "install_parent_compiler_result_target_native_transaction",
]
