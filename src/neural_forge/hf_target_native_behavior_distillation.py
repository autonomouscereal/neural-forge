"""Output-only distillation for an exact target-native compiler transaction.

The trainer in this module is deliberately blind to model internals.  Its only
teacher signal is the parent's causal output distribution and greedy action on
two closed-loop lanes: prefixes produced by the parent and prefixes produced
on-policy by the child.  Candidate selection likewise uses only action and
free-running token behavior.  Weights, hidden states, activations, attention
maps, and cache values are neither requested nor accepted as targets.

Temporary QKVO and Gate/Up/Down factors are selected on a disjoint calibration
domain with a step-zero rollback candidate.  The sealed domain is inaccessible
until a preregistered output-behavior gate passes.  The v3 continuation keeps
the inherited QKVO product and an authenticated output-Jacobian-guided product
separate, trains them in three fixed stages, and still selects only on causal
output behavior.
"""

from __future__ import annotations

import hashlib
import contextvars
import dis
import functools
import inspect
import json
import logging
import math
import operator
import sys
import types
import typing
import _thread
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from neural_forge.hf_evaluation_provenance import token_sequence_sha256
from neural_forge.hf_functional_decode_trace import (
    FunctionalDecodeStep,
    FunctionalDecodeTrace,
    trace_mapping,
)
from neural_forge.hf_progressive_target_bridge import (
    ExactTargetCompilerTransaction,
)
from neural_forge.hf_residual_attention_functional import (
    ResidualAttentionFunctionalReport,
    score_residual_attention_function,
)


TARGET_NATIVE_BEHAVIOR_DISTILLATION_SCHEMA = (
    "neural-forge-target-native-behavior-distillation-v1"
)
BEHAVIOR_DOMAIN_SCHEMA = "neural-forge-behavior-domain-v1"
BEHAVIOR_FACTOR_PACK_SCHEMA = "neural-forge-target-native-factor-pack-v1"
BEHAVIOR_FACTOR_PACK_V2_SCHEMA = "neural-forge-target-native-factor-pack-v2"
BEHAVIOR_FACTOR_PACK_V3_SCHEMA = "neural-forge-target-native-factor-pack-v3"
BEHAVIOR_FACTOR_PACK_V4_SCHEMA = "neural-forge-target-native-factor-pack-v4"
BEHAVIOR_CHECKPOINT_SCHEMA = "neural-forge-behavior-checkpoint-v1"
DURABLE_BEHAVIOR_CHECKPOINT_SCHEMA = "neural-forge-durable-behavior-checkpoint-v1"
PARENT_BEHAVIOR_CACHE_SCHEMA = "neural-forge-parent-behavior-cache-v1"
PARENT_BEHAVIOR_CACHE_BINDING_SCHEMA = (
    "neural-forge-parent-behavior-cache-binding-v1"
)
PARENT_BEHAVIOR_CACHE_AUDIT_SCHEMA = "neural-forge-parent-behavior-cache-audit-v1"
PARENT_TENSOR_STATE_SCHEMA = "neural-forge-parent-tensor-state-v1"
PARENT_EXECUTION_CONTRACT_SCHEMA = "neural-forge-parent-execution-contract-v1"

_PARENT_TENSOR_HASH_CHUNK_BYTES = 32 * 1024 * 1024

GUIDED_QKVO_PARAMETER_COUNT = 7_127_040
GUIDED_QKVO_FACTOR_TENSOR_COUNT = 8
GUIDED_COMBINED_FACTOR_TENSOR_COUNT = 26
GUIDED_GUD_FACTOR_TENSOR_COUNT = 6
GUIDED_GUD_COMBINED_FACTOR_TENSOR_COUNT = 32
GUIDED_LEFT_ONLY_END_STEP = 8
GUIDED_ONLY_END_STEP = 24
GUIDED_JOINT_START_STEP = 25
GUIDED_CONTINUATION_STEPS = 64

_QKVO_GUD_FACTOR_NAMES = frozenset(
    {
        *(
            f"qkvo.{role}_{side}"
            for role in ("q", "k", "v", "o")
            for side in ("left", "right")
        ),
        *(
            f"gud.{role}_{side}"
            for role in ("gate", "up", "down")
            for side in ("left", "right")
        ),
    }
)
_LEXICAL_FACTOR_NAMES = frozenset(
    f"lexical.{role}_{side}"
    for role in ("embedding", "lm_head")
    for side in ("left", "right")
)
_GUIDED_QKVO_FACTOR_NAMES = frozenset(
    f"guided_qkvo.{role}_{side}"
    for role in ("q", "k", "v", "o")
    for side in ("left", "right")
)
_GUIDED_GUD_FACTOR_NAMES = frozenset(
    f"guided_gud.{role}_{side}"
    for role in ("gate", "up", "down")
    for side in ("left", "right")
)
BehaviorEvaluationSplit = Literal[
    "calibration-selection",
    "calibration-exploratory",
    "heldout-evaluation",
]
BehaviorSelectionPolicy = Literal[
    "strict-nonregression",
    "rollout-first-acquisition",
]
BehaviorSealedEvaluationPolicy = Literal["candidate-gated", "disabled"]
GuidedTrainingStage = Literal["guided-left", "guided", "joint"]


class TargetNativeBehaviorDistillationError(RuntimeError):
    """An output-only training, provenance, or selection contract failed."""


def _canonical_json_default(value: Any) -> Any:
    """Expose immutable Mapping implementations without weakening JSON checks."""

    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            default=_canonical_json_default,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TargetNativeBehaviorDistillationError(
            "Behavior evidence is not canonical JSON."
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise TargetNativeBehaviorDistillationError(f"{name} must be a SHA-256.")
    return result


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise TargetNativeBehaviorDistillationError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be an integer."
        ) from error
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise TargetNativeBehaviorDistillationError(f"{name} must be {qualifier}.")
    return result


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be finite."
        ) from error
    if not math.isfinite(result):
        raise TargetNativeBehaviorDistillationError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be at least {minimum}."
        )
    if maximum is not None and result > maximum:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be at most {maximum}."
        )
    return result


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, Tensor):
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TargetNativeBehaviorDistillationError(
                f"{name} must contain integer token IDs."
            )
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be a non-empty token sequence."
        )
    result: list[int] = []
    for index, item in enumerate(value):
        token_id = _index(item, f"{name}[{index}]")
        result.append(token_id)
    if not result:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be a non-empty token sequence."
        )
    return tuple(result)


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    return _raw_tensor_sha256(
        dtype=str(detached.dtype),
        shape=tuple(detached.shape),
        raw_bytes=detached.reshape(-1).view(torch.uint8).numpy().tobytes(),
    )


def _raw_tensor_sha256(
    *,
    dtype: str,
    shape: Sequence[int],
    raw_bytes: bytes,
) -> str:
    """Hash immutable tensor bytes in the legacy factor-pack hash domain."""

    digest = hashlib.sha256()
    digest.update(str(dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def _output_jacobian_tensor_sha256(value: Tensor) -> str:
    """Hash one audit-basis tensor in the producer's immutable hash domain.

    Factor-pack tensors intentionally use :func:`_tensor_sha256` above.  The
    output-Jacobian audit predates that domain and binds its FP32 right bases as
    ``dtype + compact JSON shape + raw bytes``.  Keeping a separate helper is
    essential: changing the legacy factor-pack digest would invalidate every
    authenticated v1/v2 continuation, while comparing an audit digest with the
    factor-pack digest can never succeed even when the tensor bytes are equal.
    """

    if not isinstance(value, Tensor):
        raise TargetNativeBehaviorDistillationError(
            "Output-Jacobian basis hashing requires a tensor."
        )
    detached = value.detach()
    if detached.layout != torch.strided or detached.device.type == "meta":
        raise TargetNativeBehaviorDistillationError(
            "Output-Jacobian basis hashing requires a dense non-meta tensor."
        )
    if not detached.is_contiguous():
        detached = detached.contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii")
    )
    flat = detached.reshape(-1)
    elements_per_chunk = max(1, (4 * 1024 * 1024) // detached.element_size())
    for start in range(0, int(flat.numel()), elements_per_chunk):
        staged = flat[start : start + elements_per_chunk].to(device="cpu").contiguous()
        digest.update(memoryview(staged.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _guided_gud_basis_tensor_sha256(value: Tensor) -> str:
    """Hash a GUD right basis in the SwiGLU installer's provenance domain."""

    if not isinstance(value, Tensor):
        raise TargetNativeBehaviorDistillationError(
            "Guided-GUD basis hashing requires a tensor."
        )
    contiguous = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(
        header + b"\0" + contiguous.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorPrompt:
    """One pre-tokenized prompt with a content-derived immutable identity."""

    token_ids: tuple[int, ...]
    prompt_id: str = field(init=False)
    token_ids_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        tokens = _token_ids(self.token_ids, "token_ids")
        digest = token_sequence_sha256(tokens)
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "token_ids_sha256", digest)
        object.__setattr__(self, "prompt_id", f"prompt-{digest}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "token_ids_sha256": self.token_ids_sha256,
            "token_count": len(self.token_ids),
        }


@dataclass(frozen=True, slots=True)
class BehaviorDomainSuite:
    """A bounded, hash-bound prompt domain used in exactly one split role."""

    split_name: str
    prompts: tuple[BehaviorPrompt, ...]
    artifact_sha256: str
    domain_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        name = str(self.split_name).strip().lower()
        if not name:
            raise TargetNativeBehaviorDistillationError("split_name cannot be blank.")
        prompts = tuple(self.prompts)
        if not prompts or any(
            not isinstance(value, BehaviorPrompt) for value in prompts
        ):
            raise TargetNativeBehaviorDistillationError(
                "prompts must be a non-empty BehaviorPrompt sequence."
            )
        ids = [value.prompt_id for value in prompts]
        if len(set(ids)) != len(ids):
            raise TargetNativeBehaviorDistillationError(
                "A behavior domain cannot contain duplicate prompts."
            )
        artifact = _sha256(self.artifact_sha256, "artifact_sha256")
        payload = {
            "schema": BEHAVIOR_DOMAIN_SCHEMA,
            "split_name": name,
            "artifact_sha256": artifact,
            "prompts": [value.as_dict() for value in prompts],
        }
        object.__setattr__(self, "split_name", name)
        object.__setattr__(self, "prompts", prompts)
        object.__setattr__(self, "artifact_sha256", artifact)
        object.__setattr__(self, "domain_sha256", _canonical_sha256(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BEHAVIOR_DOMAIN_SCHEMA,
            "split_name": self.split_name,
            "artifact_sha256": self.artifact_sha256,
            "prompt_count": len(self.prompts),
            "prompts": [value.as_dict() for value in self.prompts],
            "domain_sha256": self.domain_sha256,
            "raw_prompt_text_retained": False,
        }


def _named_parent_tensors(model: nn.Module) -> tuple[tuple[str, str, Tensor], ...]:
    """Return every named parameter/buffer, preserving alias topology."""

    values: list[tuple[str, str, Tensor]] = []
    for kind, accessor in (
        ("parameter", model.named_parameters),
        ("buffer", model.named_buffers),
    ):
        try:
            named_values = accessor(remove_duplicate=False)
        except TypeError:  # pragma: no cover - compatibility with older torch.
            named_values = accessor()
        for name, value in named_values:
            if not isinstance(value, Tensor):
                raise TargetNativeBehaviorDistillationError(
                    "Parent parameters and buffers must be tensors."
                )
            values.append((kind, str(name), value))
    values.sort(key=lambda row: (row[0], row[1]))
    if not values:
        raise TargetNativeBehaviorDistillationError(
            "A parent behavior cache requires model parameters or buffers."
        )
    return tuple(values)


def _parent_runtime_signature(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Return a cheap in-process guard over tensor ownership and versions."""

    values: list[tuple[Any, ...]] = []
    for kind, name, value in _named_parent_tensors(model):
        try:
            version = int(value._version)
        except RuntimeError as error:
            raise TargetNativeBehaviorDistillationError(
                "Parent cache tensors must expose a version counter."
            ) from error
        values.append(
            (
                kind,
                name,
                id(value),
                int(value.data_ptr()),
                version,
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        )
    return tuple(values)


def _chunked_parent_tensor_sha256(value: Tensor) -> str:
    """Hash one dense contiguous tensor without staging it whole on the CPU."""

    detached = value.detach()
    if (
        detached.layout != torch.strided
        or detached.device.type == "meta"
        or not detached.is_contiguous()
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent cache authentication requires dense contiguous non-meta tensors."
        )
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    flat = detached.reshape(-1)
    elements_per_chunk = max(
        1,
        _PARENT_TENSOR_HASH_CHUNK_BYTES // max(1, detached.element_size()),
    )
    for start in range(0, int(flat.numel()), elements_per_chunk):
        staged = flat[start : start + elements_per_chunk].to(device="cpu").contiguous()
        digest.update(memoryview(staged.view(torch.uint8).numpy()))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ParentTensorState:
    state_sha256: str
    tensor_count: int
    tensor_bytes: int


def _parent_tensor_state(model: nn.Module) -> _ParentTensorState:
    """Bind all parent tensor names, aliases, metadata, and exact bytes."""

    tensor_hash_by_id: dict[int, str] = {}
    first_name_by_id: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for kind, name, value in _named_parent_tensors(model):
        identity = id(value)
        qualified_name = f"{kind}:{name}"
        tensor_hash = tensor_hash_by_id.get(identity)
        if tensor_hash is None:
            tensor_hash = _chunked_parent_tensor_sha256(value)
            tensor_hash_by_id[identity] = tensor_hash
            first_name_by_id[identity] = qualified_name
            total_bytes += int(value.numel() * value.element_size())
        rows.append(
            {
                "kind": kind,
                "name": name,
                "alias_of": first_name_by_id[identity],
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "requires_grad": bool(value.requires_grad),
                "tensor_sha256": tensor_hash,
            }
        )
    payload = {
        "schema": PARENT_TENSOR_STATE_SCHEMA,
        "named_tensor_count": len(rows),
        "unique_tensor_count": len(tensor_hash_by_id),
        "unique_tensor_bytes": total_bytes,
        "tensors": rows,
    }
    return _ParentTensorState(
        state_sha256=_canonical_sha256(payload),
        tensor_count=len(rows),
        tensor_bytes=total_bytes,
    )


_JSON_SIMPLE_UNSUPPORTED = object()
_MODULE_ATTRIBUTE_EXCLUSIONS = frozenset(
    {
        "training",
        "config",
        "_parameters",
        "_buffers",
        "_non_persistent_buffers_set",
        "_modules",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_is_full_backward_hook",
    }
)


def _stable_code_constant(value: Any) -> Any:
    """Canonicalize one Python code constant without adaptive-bytecode state."""

    if isinstance(value, types.CodeType):
        return {"nested_code_v1": _stable_code_payload(value)}
    if value is None or value is Ellipsis or isinstance(value, (bool, int, str)):
        return {
            "constant_type": type(value).__name__,
            "value": "Ellipsis" if value is Ellipsis else value,
        }
    if isinstance(value, float):
        return {"constant_type": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {
            "constant_type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if isinstance(value, bytes):
        return {"constant_type": "bytes", "value_hex": value.hex()}
    if isinstance(value, tuple):
        return {
            "constant_type": "tuple",
            "items": [_stable_code_constant(item) for item in value],
        }
    if isinstance(value, frozenset):
        items = [_stable_code_constant(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return {"constant_type": "frozenset", "items": items}
    return {
        "constant_type_module": type(value).__module__,
        "constant_type_qualname": type(value).__qualname__,
        "repr": repr(value),
    }


def _stable_code_payload(code: types.CodeType) -> dict[str, Any]:
    """Describe executable code using stable public fields only."""

    return {
        "schema": "neural-forge-python-code-v1",
        "argcount": int(code.co_argcount),
        "posonlyargcount": int(code.co_posonlyargcount),
        "kwonlyargcount": int(code.co_kwonlyargcount),
        "nlocals": int(code.co_nlocals),
        "stacksize": int(code.co_stacksize),
        "flags": int(code.co_flags),
        "bytecode_hex": code.co_code.hex(),
        "exception_table_hex": code.co_exceptiontable.hex(),
        "constants": [_stable_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "name": code.co_name,
        "qualname": code.co_qualname,
    }


def _stable_code_sha256(code: types.CodeType) -> str:
    return _canonical_sha256(_stable_code_payload(code))


def _json_simple_value(value: Any, seen: set[int] | None = None) -> Any:
    """Return typed canonical runtime state or an unsupported sentinel."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TargetNativeBehaviorDistillationError(
                "Parent module scalar attributes must be finite."
            )
        return float(value)
    generic_origin = typing.get_origin(value)
    if generic_origin is not None:
        normalized_origin = _runtime_object_state(generic_origin, seen or set())
        normalized_arguments = _json_simple_value(typing.get_args(value), seen)
        if (
            normalized_origin is _JSON_SIMPLE_UNSUPPORTED
            or normalized_arguments is _JSON_SIMPLE_UNSUPPORTED
        ):
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "runtime_generic_alias_v1": {
                "origin": normalized_origin,
                "arguments": normalized_arguments,
            }
        }
    if isinstance(value, typing.TypeVar):
        constraints = _json_simple_value(value.__constraints__, seen)
        bound = _json_simple_value(value.__bound__, seen)
        if (
            constraints is _JSON_SIMPLE_UNSUPPORTED
            or bound is _JSON_SIMPLE_UNSUPPORTED
        ):
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "runtime_type_var_v1": {
                "name": value.__name__,
                "constraints": constraints,
                "bound": bound,
                "covariant": bool(value.__covariant__),
                "contravariant": bool(value.__contravariant__),
            }
        }
    if seen is None:
        seen = set()
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return _JSON_SIMPLE_UNSUPPORTED
        seen.add(identity)
        result = [_json_simple_value(item, seen) for item in value]
        seen.remove(identity)
        if any(item is _JSON_SIMPLE_UNSUPPORTED for item in result):
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "sequence_type": "tuple" if isinstance(value, tuple) else "list",
            "items": result,
        }
    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in seen:
            return _JSON_SIMPLE_UNSUPPORTED
        seen.add(identity)
        result = [_json_simple_value(item, seen) for item in value]
        seen.remove(identity)
        if any(item is _JSON_SIMPLE_UNSUPPORTED for item in result):
            return _JSON_SIMPLE_UNSUPPORTED
        result.sort(
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return {
            "set_type": "frozenset" if isinstance(value, frozenset) else "set",
            "items": result,
        }
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return _JSON_SIMPLE_UNSUPPORTED
        seen.add(identity)
        result: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(key, str):
                normalized_key = {"type": "str", "value": key}
            elif isinstance(key, bool):
                normalized_key = {"type": "bool", "value": key}
            elif isinstance(key, int):
                normalized_key = {"type": "int", "value": int(key)}
            elif isinstance(key, float) and math.isfinite(key):
                normalized_key = {"type": "float", "value": float(key)}
            elif key is None:
                normalized_key = {"type": "none", "value": None}
            else:
                seen.remove(identity)
                return _JSON_SIMPLE_UNSUPPORTED
            normalized = _json_simple_value(item, seen)
            if normalized is _JSON_SIMPLE_UNSUPPORTED:
                seen.remove(identity)
                return _JSON_SIMPLE_UNSUPPORTED
            result.append({"key": normalized_key, "value": normalized})
        seen.remove(identity)
        result.sort(
            key=lambda row: json.dumps(
                row["key"],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        # Always wrap mappings, including string-keyed mappings, so a real
        # mapping cannot collide with the typed-key representation itself.
        return {"typed_mapping_items_v1": result}
    if isinstance(value, Tensor):
        return _JSON_SIMPLE_UNSUPPORTED
    if isinstance(value, (staticmethod, classmethod)):
        normalized_descriptor = _runtime_callable_fingerprint(
            value.__func__,
            seen=seen,
            include_referenced_globals=False,
        )
        if normalized_descriptor is _JSON_SIMPLE_UNSUPPORTED:
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "callable_descriptor_type": type(value).__name__,
            "callable": normalized_descriptor,
        }
    if callable(value):
        return _runtime_callable_fingerprint(
            value,
            seen=seen,
            include_referenced_globals=False,
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        identity = id(value)
        if identity in seen:
            return _JSON_SIMPLE_UNSUPPORTED
        seen.add(identity)
        try:
            payload = to_dict()
        except Exception:
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
        normalized = _json_simple_value(payload, seen)
        seen.remove(identity)
        if normalized is _JSON_SIMPLE_UNSUPPORTED:
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "config_like_type": {
                "module": type(value).__module__,
                "qualname": type(value).__qualname__,
            },
            "to_dict": normalized,
        }
    return _JSON_SIMPLE_UNSUPPORTED


def _runtime_object_state(
    value: Any,
    seen: set[int],
    *,
    bind_helper_class: bool = False,
) -> Any:
    """Fingerprint one non-tensor helper object used by executable code."""

    if isinstance(value, Tensor):
        return _JSON_SIMPLE_UNSUPPORTED
    if isinstance(value, logging.Logger):
        return {
            "runtime_logger_v1": {
                "name": value.name,
                "level": int(value.level),
                "disabled": bool(value.disabled),
                "propagate": bool(value.propagate),
                "class_module": type(value).__module__,
                "class_qualname": type(value).__qualname__,
            }
        }
    if isinstance(value, _thread.LockType):
        return {
            "runtime_thread_lock_v1": {
                "locked": bool(value.locked()),
            }
        }
    if isinstance(value, contextvars.ContextVar):
        try:
            current = value.get()
        except LookupError:
            current = {"context_value_missing": True}
        normalized_current = _json_simple_value(current, seen)
        if normalized_current is _JSON_SIMPLE_UNSUPPORTED:
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "runtime_context_var_v1": {
                "name": value.name,
                "current": normalized_current,
            }
        }
    if isinstance(value, types.GeneratorType):
        identity = id(value)
        if identity in seen:
            return _JSON_SIMPLE_UNSUPPORTED
        seen.add(identity)
        frame = value.gi_frame
        local_state = _json_simple_value(
            {} if frame is None else dict(frame.f_locals),
            seen,
        )
        seen.remove(identity)
        if local_state is _JSON_SIMPLE_UNSUPPORTED:
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "runtime_generator_v1": {
                "code_sha256": _stable_code_sha256(value.gi_code),
                "state": inspect.getgeneratorstate(value),
                "locals": local_state,
            }
        }
    if isinstance(value, types.ModuleType):
        version = getattr(value, "__version__", None)
        normalized_version = _json_simple_value(version, seen)
        if normalized_version is _JSON_SIMPLE_UNSUPPORTED:
            normalized_version = None
        return {
            "runtime_module_v1": {
                "name": str(getattr(value, "__name__", "")),
                "file": str(getattr(value, "__file__", "")),
                "version": normalized_version,
            }
        }
    if isinstance(value, type):
        return {
            "runtime_type_v1": {
                "module": value.__module__,
                "qualname": value.__qualname__,
            }
        }
    normalized = _json_simple_value(value, seen)
    if normalized is not _JSON_SIMPLE_UNSUPPORTED:
        return normalized
    try:
        state = vars(value)
    except TypeError:
        return _JSON_SIMPLE_UNSUPPORTED
    identity = id(value)
    if identity in seen:
        return _JSON_SIMPLE_UNSUPPORTED
    seen.add(identity)
    rows: dict[str, Any] = {}
    for name, item in sorted(state.items()):
        if _contains_runtime_tensor(item):
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
        item_state = _runtime_object_state(
            item,
            seen,
            bind_helper_class=bind_helper_class,
        )
        if item_state is _JSON_SIMPLE_UNSUPPORTED:
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
        rows[str(name)] = item_state
    seen.remove(identity)
    class_contract = None
    if bind_helper_class:
        class_contract = _runtime_helper_class_contract(type(value), seen)
        if class_contract is _JSON_SIMPLE_UNSUPPORTED:
            return _JSON_SIMPLE_UNSUPPORTED
    return {
        "runtime_object_v1": {
            "module": type(value).__module__,
            "qualname": type(value).__qualname__,
            "state": rows,
            "class_execution_contract": class_contract,
        }
    }


def _runtime_bound_object_state(value: Any, seen: set[int]) -> Any:
    """Bind callable owners without recursively treating the owner as callable."""

    if isinstance(value, (types.ModuleType, type)):
        return _runtime_object_state(value, seen)
    identity = id(value)
    if identity in seen:
        return _JSON_SIMPLE_UNSUPPORTED
    try:
        state = vars(value)
    except TypeError:
        return {
            "bound_runtime_object_v1": {
                "module": type(value).__module__,
                "qualname": type(value).__qualname__,
                "state": None,
            }
        }
    seen.add(identity)
    rows: dict[str, Any] = {}
    for name, item in sorted(state.items()):
        if _contains_runtime_tensor(item):
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
        normalized = _runtime_object_state(item, seen)
        if normalized is _JSON_SIMPLE_UNSUPPORTED:
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
        rows[str(name)] = normalized
    seen.remove(identity)
    return {
        "bound_runtime_object_v1": {
            "module": type(value).__module__,
            "qualname": type(value).__qualname__,
            "state": rows,
        }
    }


def _runtime_callable_fingerprint(
    value: Any,
    *,
    seen: set[int],
    include_referenced_globals: bool,
    relevant_registry_keys: frozenset[str] = frozenset(),
) -> Any:
    """Bind callable implementation/default/closure state without executing it."""

    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if isinstance(value, type):
        return _runtime_object_state(value, seen)
    if isinstance(value, functools.partial):
        identity = id(value)
        if identity in seen:
            return _JSON_SIMPLE_UNSUPPORTED
        seen.add(identity)
        function_state = _runtime_callable_fingerprint(
            value.func,
            seen=seen,
            include_referenced_globals=include_referenced_globals,
            relevant_registry_keys=relevant_registry_keys,
        )
        arguments = _json_simple_value(value.args, seen)
        keywords = _json_simple_value(value.keywords or {}, seen)
        seen.remove(identity)
        if any(
            item is _JSON_SIMPLE_UNSUPPORTED
            for item in (function_state, arguments, keywords)
        ):
            return _JSON_SIMPLE_UNSUPPORTED
        return {
            "runtime_partial_v1": {
                "function": function_state,
                "arguments": arguments,
                "keywords": keywords,
            }
        }

    target = value
    bound_self: Any = None
    if inspect.ismethod(value):
        target = value.__func__
        bound_self = value.__self__
    elif inspect.isbuiltin(value):
        bound_self = getattr(value, "__self__", None)
    elif not inspect.isfunction(value):
        target = inspect.getattr_static(type(value), "__call__", None)
        bound_self = value
    if target is None:
        return _JSON_SIMPLE_UNSUPPORTED

    identity = id(value)
    if identity in seen:
        return _JSON_SIMPLE_UNSUPPORTED
    seen.add(identity)
    code = getattr(target, "__code__", None)
    defaults = _json_simple_value(getattr(target, "__defaults__", None), seen)
    keyword_defaults = _json_simple_value(
        getattr(target, "__kwdefaults__", None),
        seen,
    )
    if (
        defaults is _JSON_SIMPLE_UNSUPPORTED
        or keyword_defaults is _JSON_SIMPLE_UNSUPPORTED
    ):
        seen.remove(identity)
        return _JSON_SIMPLE_UNSUPPORTED
    closure_rows: list[Any] = []
    for cell in getattr(target, "__closure__", None) or ():
        try:
            cell_value = cell.cell_contents
        except ValueError:
            closure_rows.append({"empty_cell": True})
            continue
        if id(cell_value) in seen:
            closure_rows.append(
                {
                    "recursive_runtime_reference_v1": {
                        "module": str(getattr(cell_value, "__module__", "")),
                        "qualname": str(getattr(cell_value, "__qualname__", "")),
                        "type_module": type(cell_value).__module__,
                        "type_qualname": type(cell_value).__qualname__,
                    }
                }
            )
            continue
        normalized_cell = _runtime_object_state(cell_value, seen)
        if normalized_cell is _JSON_SIMPLE_UNSUPPORTED:
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
        closure_rows.append(normalized_cell)
    bound_state = None
    if bound_self is not None:
        bound_state = _runtime_bound_object_state(bound_self, seen)
        if bound_state is _JSON_SIMPLE_UNSUPPORTED:
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
    global_rows: dict[str, Any] = {}
    if include_referenced_globals and code is not None:
        global_values = getattr(target, "__globals__", {})
        referenced_names = {
            str(instruction.argval)
            for instruction in dis.get_instructions(code)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
            and isinstance(instruction.argval, str)
        }
        for name in sorted(referenced_names):
            if name not in global_values:
                continue
            item = global_values[name]
            if isinstance(item, Tensor):
                seen.remove(identity)
                return _JSON_SIMPLE_UNSUPPORTED
            if name == "_CAN_RECORD_REGISTRY" and isinstance(item, Mapping):
                registry_rows: list[dict[str, Any]] = []
                for registry_key in sorted(relevant_registry_keys):
                    if registry_key not in item:
                        registry_rows.append(
                            {"key": registry_key, "entry_missing": True}
                        )
                        continue
                    registry_value = _runtime_object_state(
                        item[registry_key],
                        seen,
                    )
                    if registry_value is _JSON_SIMPLE_UNSUPPORTED:
                        seen.remove(identity)
                        return _JSON_SIMPLE_UNSUPPORTED
                    registry_rows.append(
                        {"key": registry_key, "value": registry_value}
                    )
                normalized_item = {
                    "transformers_relevant_output_registry_v1": registry_rows
                }
            elif callable(item):
                if id(item) in seen:
                    normalized_item = {
                        "recursive_runtime_reference_v1": {
                            "module": str(getattr(item, "__module__", "")),
                            "qualname": str(getattr(item, "__qualname__", "")),
                            "type_module": type(item).__module__,
                            "type_qualname": type(item).__qualname__,
                        }
                    }
                else:
                    normalized_item = _runtime_callable_fingerprint(
                        item,
                        seen=seen,
                        include_referenced_globals=True,
                        relevant_registry_keys=relevant_registry_keys,
                    )
            else:
                normalized_item = _runtime_object_state(item, seen)
            if normalized_item is _JSON_SIMPLE_UNSUPPORTED:
                seen.remove(identity)
                return _JSON_SIMPLE_UNSUPPORTED
            global_rows[name] = normalized_item
    wrapped_state = None
    wrapped = getattr(target, "__wrapped__", None)
    if callable(wrapped) and wrapped is not target:
        wrapped_state = _runtime_callable_fingerprint(
            wrapped,
            seen=seen,
            include_referenced_globals=include_referenced_globals,
            relevant_registry_keys=relevant_registry_keys,
        )
        if wrapped_state is _JSON_SIMPLE_UNSUPPORTED:
            seen.remove(identity)
            return _JSON_SIMPLE_UNSUPPORTED
    seen.remove(identity)
    return {
        "runtime_callable_v1": {
            "type_module": type(value).__module__,
            "type_qualname": type(value).__qualname__,
            "module": str(getattr(target, "__module__", "")),
            "qualname": str(getattr(target, "__qualname__", "")),
            "code_sha256": (
                _stable_code_sha256(code)
                if code is not None
                else None
            ),
            "defaults": defaults,
            "keyword_defaults": keyword_defaults,
            "closure": closure_rows,
            "bound_self": bound_state,
            "referenced_globals": global_rows,
            "wrapped": wrapped_state,
        }
    }


def _runtime_helper_class_contract(
    helper_type: type[Any],
    seen: set[int],
) -> Any:
    """Bind executable class state for an arbitrary nested helper object."""

    identity = id(helper_type)
    if identity in seen:
        return {
            "recursive_helper_class_reference_v1": {
                "module": helper_type.__module__,
                "qualname": helper_type.__qualname__,
            }
        }
    seen.add(identity)
    owners: list[dict[str, Any]] = []
    for owner in helper_type.__mro__:
        if owner is object:
            continue
        attributes: dict[str, Any] = {}
        for name, value in sorted(vars(owner).items()):
            if name in _CLASS_ATTRIBUTE_EXCLUSIONS:
                continue
            if _contains_runtime_tensor(value):
                seen.remove(identity)
                return _JSON_SIMPLE_UNSUPPORTED
            if isinstance(value, property):
                property_state: dict[str, Any] = {}
                for role, accessor in (
                    ("get", value.fget),
                    ("set", value.fset),
                    ("delete", value.fdel),
                ):
                    if accessor is None:
                        property_state[role] = None
                        continue
                    normalized_accessor = _runtime_callable_fingerprint(
                        accessor,
                        seen=seen,
                        include_referenced_globals=True,
                    )
                    if normalized_accessor is _JSON_SIMPLE_UNSUPPORTED:
                        seen.remove(identity)
                        return _JSON_SIMPLE_UNSUPPORTED
                    property_state[role] = normalized_accessor
                attributes[name] = {"runtime_property_v1": property_state}
                continue
            descriptor = value.__func__ if isinstance(
                value, (staticmethod, classmethod)
            ) else value
            if callable(descriptor):
                normalized = _runtime_callable_fingerprint(
                    descriptor,
                    seen=seen,
                    include_referenced_globals=True,
                )
            else:
                normalized = _runtime_object_state(descriptor, seen)
            if normalized is _JSON_SIMPLE_UNSUPPORTED:
                seen.remove(identity)
                return _JSON_SIMPLE_UNSUPPORTED
            attributes[name] = normalized
        owners.append(
            {
                "owner_module": owner.__module__,
                "owner_qualname": owner.__qualname__,
                "attributes": attributes,
            }
        )
    seen.remove(identity)
    return {
        "runtime_helper_class_v1": {
            "module": helper_type.__module__,
            "qualname": helper_type.__qualname__,
            "owners": owners,
        }
    }


def _contains_runtime_tensor(value: Any, seen: set[int] | None = None) -> bool:
    """Find unregistered tensors nested in ordinary runtime attributes."""

    if isinstance(value, Tensor):
        return True
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return False
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    if isinstance(value, Mapping):
        seen.add(identity)
        result = any(
            _contains_runtime_tensor(key, seen)
            or _contains_runtime_tensor(item, seen)
            for key, item in value.items()
        )
        seen.remove(identity)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(identity)
        result = any(_contains_runtime_tensor(item, seen) for item in value)
        seen.remove(identity)
        return result
    if isinstance(value, functools.partial):
        seen.add(identity)
        result = any(
            (
                _contains_runtime_tensor(value.func, seen),
                _contains_runtime_tensor(value.args, seen),
                _contains_runtime_tensor(value.keywords or {}, seen),
            )
        )
        seen.remove(identity)
        return result
    callable_target = value
    bound_self: Any = None
    if inspect.ismethod(value):
        callable_target = value.__func__
        bound_self = value.__self__
    if inspect.isfunction(callable_target):
        seen.add(identity)
        values: list[Any] = [
            getattr(callable_target, "__defaults__", None),
            getattr(callable_target, "__kwdefaults__", None),
        ]
        for cell in getattr(callable_target, "__closure__", None) or ():
            try:
                values.append(cell.cell_contents)
            except ValueError:
                pass
        if bound_self is not None:
            values.append(bound_self)
        result = any(_contains_runtime_tensor(item, seen) for item in values)
        seen.remove(identity)
        return result
    if isinstance(value, types.GeneratorType):
        seen.add(identity)
        frame = value.gi_frame
        result = frame is not None and _contains_runtime_tensor(
            dict(frame.f_locals),
            seen,
        )
        seen.remove(identity)
        return result
    if isinstance(
        value,
        (types.ModuleType, type, logging.Logger, contextvars.ContextVar),
    ):
        return False
    try:
        state = vars(value)
    except TypeError:
        return False
    seen.add(identity)
    result = any(_contains_runtime_tensor(item, seen) for item in state.values())
    seen.remove(identity)
    return result


def _reject_parent_forward_hooks(model: nn.Module) -> None:
    """Reject output-intercepting state that cannot be fingerprinted exactly."""

    global_module = torch.nn.modules.module
    global_hook_names = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
    )
    if any(bool(getattr(global_module, name, {})) for name in global_hook_names):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior caching forbids global forward/pre-forward hooks."
        )
    for path, module in model.named_modules():
        module_state = vars(module)
        if bool(module_state.get("_forward_pre_hooks")) or bool(
            module_state.get("_forward_hooks")
        ):
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior caching forbids local forward/pre-forward hooks "
                f"at {path or '<root>'}."
            )
        if "forward" in module_state:
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior caching forbids instance-level forward overrides "
                f"at {path or '<root>'}."
            )
        if module_state.get("_hf_hook") is not None:
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior caching forbids Accelerate _hf_hook dispatch state "
                f"at {path or '<root>'}."
            )


def _forward_code_sha256(module: nn.Module) -> str:
    implementation = inspect.getattr_static(type(module), "forward", None)
    if not callable(implementation):
        raise TargetNativeBehaviorDistillationError(
            "Parent module forward implementation is not callable."
        )
    payload = _runtime_callable_fingerprint(
        implementation,
        seen=set(),
        include_referenced_globals=True,
        relevant_registry_keys=frozenset({str(type(module))}),
    )
    if payload is _JSON_SIMPLE_UNSUPPORTED:
        raise TargetNativeBehaviorDistillationError(
            "Parent module forward implementation has unsupported executable state."
        )
    return _canonical_sha256(payload)


def _descriptor_callables(value: Any) -> tuple[Callable[..., Any], ...]:
    if isinstance(value, (staticmethod, classmethod)):
        return (value.__func__,)
    if isinstance(value, property):
        return tuple(
            accessor
            for accessor in (value.fget, value.fset, value.fdel)
            if accessor is not None
        )
    return (value,) if callable(value) else ()


def _callable_code_objects(value: Callable[..., Any]) -> tuple[types.CodeType, ...]:
    """Return implementation/wrapper code objects used by one callable."""

    pending: list[Any] = [value]
    seen: set[int] = set()
    codes: list[types.CodeType] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if inspect.ismethod(current):
            current = current.__func__
        code = getattr(current, "__code__", None)
        if isinstance(code, types.CodeType):
            codes.append(code)
        wrapped = getattr(current, "__wrapped__", None)
        if callable(wrapped):
            pending.append(wrapped)
        for cell in getattr(current, "__closure__", None) or ():
            try:
                cell_value = cell.cell_contents
            except ValueError:
                continue
            if callable(cell_value):
                pending.append(cell_value)
    return tuple(codes)


def _reachable_class_method_contract(module_type: type[nn.Module]) -> list[dict[str, Any]]:
    """Bind reachable method implementations across the concrete class MRO."""

    def definitions(
        name: str,
    ) -> list[tuple[type[Any], str, Callable[..., Any]]]:
        values: list[tuple[type[Any], str, Callable[..., Any]]] = []
        for owner in module_type.__mro__:
            if owner is object or name not in vars(owner):
                continue
            values.extend(
                (owner, name, callable_value)
                for callable_value in _descriptor_callables(vars(owner)[name])
            )
        return values

    forward_definitions = definitions("forward")
    if not forward_definitions:
        raise TargetNativeBehaviorDistillationError(
            "Parent module class has no resolvable forward implementation."
        )
    resolved_forward_owner = forward_definitions[0][0]
    pending = [
        value
        for value in forward_definitions
        if value[0] is resolved_forward_owner
    ]
    root_keys = {
        (id(owner), name, id(implementation))
        for owner, name, implementation in pending
    }
    seen_implementations: set[tuple[int, str, int]] = set()
    rows: list[dict[str, Any]] = []
    registry_keys = frozenset({str(module_type)})
    while pending:
        owner, name, implementation = pending.pop()
        implementation_key = (id(owner), name, id(implementation))
        if implementation_key in seen_implementations:
            continue
        seen_implementations.add(implementation_key)
        if implementation_key not in root_keys:
            fingerprint = _runtime_callable_fingerprint(
                implementation,
                seen=set(),
                include_referenced_globals=True,
                relevant_registry_keys=registry_keys,
            )
            if fingerprint is _JSON_SIMPLE_UNSUPPORTED:
                raise TargetNativeBehaviorDistillationError(
                    "A forward-reachable parent class method has unsupported "
                    f"execution state: {owner.__module__}."
                    f"{owner.__qualname__}.{name}."
                )
            rows.append(
                {
                    "owner_module": owner.__module__,
                    "owner_qualname": owner.__qualname__,
                    "name": name,
                    "callable": fingerprint,
                }
            )
        referenced_names: set[str] = set()
        for code in _callable_code_objects(implementation):
            referenced_names.update(
                str(instruction.argval)
                for instruction in dis.get_instructions(code)
                if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
                and isinstance(instruction.argval, str)
            )
        for referenced_name in sorted(referenced_names):
            pending.extend(definitions(referenced_name))
    rows.sort(
        key=lambda row: (
            row["owner_module"],
            row["owner_qualname"],
            row["name"],
            _canonical_sha256(row["callable"]),
        )
    )
    return rows


_CLASS_ATTRIBUTE_EXCLUSIONS = frozenset(
    {
        "__annotations__",
        "__dict__",
        "__doc__",
        "__module__",
        "__weakref__",
    }
)


def _class_execution_attributes(
    module_type: type[nn.Module],
    owner_cache: dict[type[Any], dict[str, Any]],
) -> list[dict[str, str]]:
    """Bind scalar/callable class state across the module's full MRO."""

    rows: list[dict[str, str]] = []
    for owner in module_type.__mro__:
        if owner is object:
            continue
        if owner not in owner_cache:
            attributes: dict[str, Any] = {}
            for name, value in sorted(vars(owner).items()):
                if name in _CLASS_ATTRIBUTE_EXCLUSIONS:
                    continue
                if _contains_runtime_tensor(value):
                    raise TargetNativeBehaviorDistillationError(
                        "Parent behavior caching forbids unregistered class tensors at "
                        f"{owner.__module__}.{owner.__qualname__}.{name}."
                    )
                if isinstance(value, property):
                    property_state: dict[str, Any] = {}
                    for role, accessor in (
                        ("get", value.fget),
                        ("set", value.fset),
                        ("delete", value.fdel),
                    ):
                        if accessor is None:
                            property_state[role] = None
                            continue
                        normalized_accessor = _runtime_callable_fingerprint(
                            accessor,
                            seen=set(),
                            include_referenced_globals=False,
                        )
                        if normalized_accessor is _JSON_SIMPLE_UNSUPPORTED:
                            raise TargetNativeBehaviorDistillationError(
                                "Parent class property has unsupported callable state "
                                f"at {owner.__module__}.{owner.__qualname__}.{name}."
                            )
                        property_state[role] = normalized_accessor
                    attributes[name] = {"runtime_property_v1": property_state}
                    continue
                normalized = _runtime_object_state(value, set())
                if normalized is not _JSON_SIMPLE_UNSUPPORTED:
                    attributes[name] = normalized
                else:
                    raise TargetNativeBehaviorDistillationError(
                        "Parent class attribute has unsupported executable state at "
                        f"{owner.__module__}.{owner.__qualname__}.{name}."
                    )
            owner_cache[owner] = {
                "owner_module": owner.__module__,
                "owner_qualname": owner.__qualname__,
                "attributes": attributes,
            }
        rows.append(
            {
                "owner_module": owner.__module__,
                "owner_qualname": owner.__qualname__,
            }
        )
    return rows


def _parent_execution_contract_sha256(model: nn.Module) -> str:
    """Bind covered hook-free code, topology, config, and runtime state.

    This is deliberately a bounded in-process contract, not a proof over every
    mutable Python interpreter object.  Terminal cache authentication separately
    rehashes cached output rows; it does not rerun live parent outputs.
    """

    _reject_parent_forward_hooks(model)
    config = getattr(model, "config", None)
    config_payload: Any = None
    if config is not None:
        to_dict = getattr(config, "to_dict", None)
        if not callable(to_dict):
            raise TargetNativeBehaviorDistillationError(
                "Parent model config must expose to_dict for behavior caching."
            )
        config_payload = _json_simple_value(to_dict())
        if config_payload is _JSON_SIMPLE_UNSUPPORTED:
            raise TargetNativeBehaviorDistillationError(
                "Parent model config is not canonical JSON-simple data."
            )
    modules: list[dict[str, Any]] = []
    try:
        named_modules = model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        named_modules = model.named_modules()
    code_hash_by_class: dict[type[nn.Module], str] = {}
    class_attributes_by_class: dict[type[nn.Module], list[dict[str, str]]] = {}
    reachable_methods_by_class: dict[type[nn.Module], list[dict[str, Any]]] = {}
    class_owner_attribute_cache: dict[type[Any], dict[str, Any]] = {}
    for path, module in named_modules:
        module_type = type(module)
        code_hash = code_hash_by_class.get(module_type)
        if code_hash is None:
            code_hash = _forward_code_sha256(module)
            code_hash_by_class[module_type] = code_hash
            class_attributes_by_class[module_type] = _class_execution_attributes(
                module_type,
                class_owner_attribute_cache,
            )
            reachable_methods_by_class[module_type] = (
                _reachable_class_method_contract(module_type)
            )
        scalar_attributes: dict[str, Any] = {}
        for name, value in sorted(vars(module).items()):
            if name in _MODULE_ATTRIBUTE_EXCLUSIONS:
                continue
            if _contains_runtime_tensor(value):
                raise TargetNativeBehaviorDistillationError(
                    "Parent behavior caching forbids unregistered tensor attributes "
                    f"at {path or '<root>'}.{name}."
                )
            normalized = _runtime_object_state(
                value,
                set(),
                bind_helper_class=True,
            )
            if normalized is not _JSON_SIMPLE_UNSUPPORTED:
                scalar_attributes[name] = normalized
            else:
                raise TargetNativeBehaviorDistillationError(
                    "Parent module attribute has unsupported executable state at "
                    f"{path or '<root>'}.{name}."
                )
        modules.append(
            {
                "path": str(path),
                "class_module": module_type.__module__,
                "class_qualname": module_type.__qualname__,
                "json_simple_scalar_attributes": scalar_attributes,
            }
        )
    module_class_contracts = [
        {
            "class_module": module_type.__module__,
            "class_qualname": module_type.__qualname__,
            "forward_code_sha256": code_hash_by_class[module_type],
            "mro": class_attributes_by_class[module_type],
            "forward_reachable_class_methods": (
                reachable_methods_by_class[module_type]
            ),
        }
        for module_type in sorted(
            code_hash_by_class,
            key=lambda value: (value.__module__, value.__qualname__),
        )
    ]
    class_owner_contracts = sorted(
        class_owner_attribute_cache.values(),
        key=lambda row: (row["owner_module"], row["owner_qualname"]),
    )
    runtime = {
        "torch_version": str(torch.__version__),
        "python_version": str(sys.version),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    return _canonical_sha256(
        {
            "schema": PARENT_EXECUTION_CONTRACT_SCHEMA,
            "hook_policy": "no-local-or-global-forward-hooks",
            "runtime": runtime,
            "model_config": config_payload,
            "module_class_contracts": module_class_contracts,
            "class_owner_contracts": class_owner_contracts,
            "modules": modules,
        }
    )


@dataclass(frozen=True, slots=True)
class _ParentAuthentication:
    runtime_signature: tuple[tuple[Any, ...], ...]
    tensor_state_sha256: str
    tensor_count: int
    tensor_bytes: int
    execution_contract_sha256: str


def _capture_parent_authentication(model: nn.Module) -> _ParentAuthentication:
    runtime_signature = _parent_runtime_signature(model)
    execution_contract_sha256 = _parent_execution_contract_sha256(model)
    tensor_state = _parent_tensor_state(model)
    if _parent_runtime_signature(model) != runtime_signature:
        raise TargetNativeBehaviorDistillationError(
            "Parent runtime tensors changed while their bytes were authenticated."
        )
    if _parent_execution_contract_sha256(model) != execution_contract_sha256:
        raise TargetNativeBehaviorDistillationError(
            "Parent execution state changed while it was authenticated."
        )
    return _ParentAuthentication(
        runtime_signature=runtime_signature,
        tensor_state_sha256=tensor_state.state_sha256,
        tensor_count=tensor_state.tensor_count,
        tensor_bytes=tensor_state.tensor_bytes,
        execution_contract_sha256=execution_contract_sha256,
    )


def _assert_parent_authentication_matches(
    expected: _ParentAuthentication,
    observed: _ParentAuthentication,
) -> None:
    if observed.runtime_signature != expected.runtime_signature:
        raise TargetNativeBehaviorDistillationError(
            "Parent runtime tensor ownership or versions changed."
        )
    if observed.tensor_state_sha256 != expected.tensor_state_sha256:
        raise TargetNativeBehaviorDistillationError(
            "Parent tensor-state bytes or topology changed."
        )
    if (
        observed.tensor_count != expected.tensor_count
        or observed.tensor_bytes != expected.tensor_bytes
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent tensor-state cardinality changed."
        )
    if (
        observed.execution_contract_sha256
        != expected.execution_contract_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent execution contract changed."
        )


def _assert_parent_runtime_execution_unchanged(
    model: nn.Module,
    expected: _ParentAuthentication,
) -> None:
    """Use cheap guards during training; terminal byte hashing remains authoritative."""

    if _parent_runtime_signature(model) != expected.runtime_signature:
        raise TargetNativeBehaviorDistillationError(
            "Parent runtime tensor ownership or versions changed."
        )
    if (
        _parent_execution_contract_sha256(model)
        != expected.execution_contract_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent execution contract changed."
        )


def _full_prefix_parent_trace_manifest(
    trace: FunctionalDecodeTrace,
    *,
    horizon: int,
) -> dict[str, Any]:
    """Authenticate one full-prefix parent trace and return its cache manifest."""

    if not isinstance(trace, FunctionalDecodeTrace):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache traces must be FunctionalDecodeTrace values."
        )
    horizon_value = _index(horizon, "parent cache horizon", positive=True)
    token_hash = token_sequence_sha256(trace.prompt_token_ids)
    if (
        trace.lane != "greedy"
        or not trace.sequence_id
        or len(trace.steps) != horizon_value
        or len(trace.selected_token_ids) != horizon_value
        or trace.consumed_token_ids != trace.selected_token_ids[:-1]
        or trace.cache_class != "full-prefix-functional-decode"
        or trace.model_class != "output-only-causal-model"
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache contains a non-canonical full-prefix trace."
        )
    row_hashes: list[str] = []
    row_shapes: list[list[int]] = []
    row_dtypes: list[str] = []
    logit_bytes = 0
    for index, (selected, step) in enumerate(
        zip(trace.selected_token_ids, trace.steps, strict=True)
    ):
        row = step.action_logits
        if (
            not isinstance(row, Tensor)
            or row.device.type != "cpu"
            or not row.is_contiguous()
            or not row.dtype.is_floating_point
            or tuple(row.shape) != (1, trace.vocabulary_size)
            or not bool(torch.isfinite(row).all())
            or step.step_index != index
            or step.prefix_length != len(trace.prompt_token_ids) + index
            or step.cache_length != step.prefix_length
            or step.selected_token_id != selected
            or step.greedy_token_id != selected
        ):
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior cache contains an invalid output-logit row."
            )
        digest = _tensor_sha256(row)
        if digest != step.action_logits_sha256:
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior cache output-logit row hash changed."
            )
        row_hashes.append(digest)
        row_shapes.append(list(row.shape))
        row_dtypes.append(str(row.dtype))
        logit_bytes += int(row.numel() * row.element_size())
    trace_payload = {
        "sequence_id": trace.sequence_id,
        "lane": "greedy",
        "prompt_token_ids_sha256": token_hash,
        "selected_token_ids": list(trace.selected_token_ids),
        "action_logits_sha256s": row_hashes,
    }
    if (
        trace.trace_sha256 != _canonical_sha256(trace_payload)
        or trace.logit_bytes != logit_bytes
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache trace digest or byte count changed."
        )
    return {
        "prompt_id": trace.sequence_id,
        "token_ids_sha256": token_hash,
        "token_count": len(trace.prompt_token_ids),
        "trace_sha256": trace.trace_sha256,
        "selected_token_ids": list(trace.selected_token_ids),
        "row_tensor_sha256s": row_hashes,
        "row_shapes": row_shapes,
        "row_dtypes": row_dtypes,
    }


def _parent_behavior_cache_manifest(
    *,
    split_name: str,
    artifact_sha256: str,
    domain_sha256: str,
    parent_tensor_state_sha256: str,
    parent_tensor_count: int,
    parent_tensor_bytes: int,
    parent_execution_contract_sha256: str,
    horizon: int,
    traces: Sequence[FunctionalDecodeTrace],
) -> dict[str, Any]:
    prompt_rows = [
        _full_prefix_parent_trace_manifest(trace, horizon=horizon) for trace in traces
    ]
    return {
        "schema": PARENT_BEHAVIOR_CACHE_SCHEMA,
        "split_name": str(split_name),
        "artifact_sha256": _sha256(artifact_sha256, "cache artifact_sha256"),
        "domain_sha256": _sha256(domain_sha256, "cache domain_sha256"),
        "parent_tensor_state_sha256": _sha256(
            parent_tensor_state_sha256,
            "cache parent tensor-state SHA-256",
        ),
        "parent_tensor_count": _index(
            parent_tensor_count,
            "cache parent tensor count",
            positive=True,
        ),
        "parent_tensor_bytes": _index(
            parent_tensor_bytes,
            "cache parent tensor bytes",
            positive=True,
        ),
        "parent_execution_contract_sha256": _sha256(
            parent_execution_contract_sha256,
            "cache parent execution-contract SHA-256",
        ),
        "horizon": _index(horizon, "cache horizon", positive=True),
        "prompt_count": len(prompt_rows),
        "prompts": prompt_rows,
        "cache_build_parent_forward_count": len(prompt_rows) * horizon,
        "collector": "full-prefix-use-cache-false",
        "output_function_only": True,
        "parent_weights_stored": False,
    }


@dataclass(frozen=True, slots=True)
class ParentBehaviorCache:
    """Hash-bound in-process cache of parent greedy output traces.

    Public row tensors remain ordinary mutable tensors, so every use authenticates
    them and hands consumers private clones.  The cache object itself is not a
    Python security boundary.
    """

    split_name: str
    artifact_sha256: str
    domain_sha256: str
    parent_tensor_state_sha256: str
    parent_tensor_count: int
    parent_tensor_bytes: int
    parent_execution_contract_sha256: str
    horizon: int
    traces: tuple[FunctionalDecodeTrace, ...]
    _parent_model: nn.Module = field(repr=False, compare=False)
    _parent_runtime_signature: tuple[tuple[Any, ...], ...] = field(
        repr=False,
        compare=False,
    )
    cache_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        name = str(self.split_name).strip().lower()
        traces = tuple(self.traces)
        if not name or not traces or not isinstance(self._parent_model, nn.Module):
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior cache metadata is incomplete."
            )
        runtime_signature = tuple(self._parent_runtime_signature)
        if not runtime_signature:
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior cache has no runtime state guard."
            )
        object.__setattr__(self, "split_name", name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "cache artifact_sha256"),
        )
        object.__setattr__(
            self,
            "domain_sha256",
            _sha256(self.domain_sha256, "cache domain_sha256"),
        )
        object.__setattr__(
            self,
            "parent_tensor_state_sha256",
            _sha256(
                self.parent_tensor_state_sha256,
                "cache parent tensor-state SHA-256",
            ),
        )
        object.__setattr__(
            self,
            "parent_tensor_count",
            _index(
                self.parent_tensor_count,
                "cache parent tensor count",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "parent_tensor_bytes",
            _index(
                self.parent_tensor_bytes,
                "cache parent tensor bytes",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "parent_execution_contract_sha256",
            _sha256(
                self.parent_execution_contract_sha256,
                "cache parent execution-contract SHA-256",
            ),
        )
        object.__setattr__(
            self,
            "horizon",
            _index(self.horizon, "cache horizon", positive=True),
        )
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "_parent_runtime_signature", runtime_signature)
        object.__setattr__(self, "cache_sha256", _canonical_sha256(self.manifest()))

    def manifest(self) -> dict[str, Any]:
        return _parent_behavior_cache_manifest(
            split_name=self.split_name,
            artifact_sha256=self.artifact_sha256,
            domain_sha256=self.domain_sha256,
            parent_tensor_state_sha256=self.parent_tensor_state_sha256,
            parent_tensor_count=self.parent_tensor_count,
            parent_tensor_bytes=self.parent_tensor_bytes,
            parent_execution_contract_sha256=(
                self.parent_execution_contract_sha256
            ),
            horizon=self.horizon,
            traces=self.traces,
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self.manifest(), "cache_sha256": self.cache_sha256}


@dataclass(frozen=True, slots=True)
class ParentBehaviorCacheBinding:
    """One authenticated horizon-specific hit against a parent behavior cache."""

    cache_sha256: str
    domain_sha256: str
    split_name: str
    parent_tensor_state_sha256: str
    parent_execution_contract_sha256: str
    cache_horizon: int
    requested_horizon: int
    traces: tuple[FunctionalDecodeTrace, ...] = field(repr=False)
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        traces = tuple(self.traces)
        if not traces:
            raise TargetNativeBehaviorDistillationError(
                "A parent behavior cache binding cannot be empty."
            )
        payload = {
            "schema": PARENT_BEHAVIOR_CACHE_BINDING_SCHEMA,
            "cache_sha256": _sha256(self.cache_sha256, "parent cache SHA-256"),
            "domain_sha256": _sha256(self.domain_sha256, "parent cache domain"),
            "split_name": str(self.split_name),
            "parent_tensor_state_sha256": _sha256(
                self.parent_tensor_state_sha256,
                "parent cache tensor-state SHA-256",
            ),
            "parent_execution_contract_sha256": _sha256(
                self.parent_execution_contract_sha256,
                "parent cache execution-contract SHA-256",
            ),
            "cache_horizon": _index(
                self.cache_horizon, "parent cache horizon", positive=True
            ),
            "requested_horizon": _index(
                self.requested_horizon, "requested cache horizon", positive=True
            ),
            "prompt_count": len(traces),
            "trace_sha256s": [value.trace_sha256 for value in traces],
            "row_tensor_sha256s": [
                [step.action_logits_sha256 for step in value.steps]
                for value in traces
            ],
        }
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "binding_sha256", _canonical_sha256(payload))

    @property
    def parent_forward_count_avoided(self) -> int:
        return len(self.traces) * self.requested_horizon

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PARENT_BEHAVIOR_CACHE_BINDING_SCHEMA,
            "cache_hit": True,
            "cache_sha256": self.cache_sha256,
            "binding_sha256": self.binding_sha256,
            "domain_sha256": self.domain_sha256,
            "split_name": self.split_name,
            "parent_tensor_state_sha256": self.parent_tensor_state_sha256,
            "parent_execution_contract_sha256": (
                self.parent_execution_contract_sha256
            ),
            "cache_horizon": self.cache_horizon,
            "requested_horizon": self.requested_horizon,
            "prompt_count": len(self.traces),
            "gross_parent_forward_count_avoided": (
                self.parent_forward_count_avoided
            ),
            "full_prefix_parent_rows_reused": True,
            "parent_tensor_state_authenticated_at_binding": True,
            "covered_parent_execution_state_authenticated_at_binding": True,
            "arbitrary_python_runtime_state_authenticated_at_binding": False,
            "live_parent_outputs_executed_at_binding": False,
        }


def load_behavior_domain_suite(
    split_name: str,
    payload: Mapping[str, Any],
    *,
    artifact_sha256: str,
) -> BehaviorDomainSuite:
    """Load one pre-tokenized domain and verify any supplied prompt bindings."""

    if not isinstance(payload, Mapping):
        raise TargetNativeBehaviorDistillationError("payload must be a mapping.")
    raw_prompts = payload.get("prompts")
    if isinstance(raw_prompts, (str, bytes)) or not isinstance(raw_prompts, Sequence):
        raise TargetNativeBehaviorDistillationError(
            "payload.prompts must be a non-empty sequence."
        )
    prompts: list[BehaviorPrompt] = []
    for index, raw in enumerate(raw_prompts):
        supplied_id: Any = None
        supplied_hash: Any = None
        if isinstance(raw, Mapping):
            tokens = raw.get("token_ids", raw.get("input_ids"))
            supplied_id = raw.get("prompt_id")
            supplied_hash = raw.get("token_ids_sha256")
        else:
            tokens = raw
        prompt = BehaviorPrompt(_token_ids(tokens, f"prompts[{index}].token_ids"))
        if supplied_id is not None and str(supplied_id) != prompt.prompt_id:
            raise TargetNativeBehaviorDistillationError(
                f"prompts[{index}].prompt_id does not match its token content."
            )
        if supplied_hash is not None and str(supplied_hash) != prompt.token_ids_sha256:
            raise TargetNativeBehaviorDistillationError(
                f"prompts[{index}].token_ids_sha256 does not match its content."
            )
        prompts.append(prompt)
    declared_artifact = payload.get("artifact_sha256")
    if declared_artifact is not None and _sha256(
        declared_artifact, "payload.artifact_sha256"
    ) != _sha256(artifact_sha256, "artifact_sha256"):
        raise TargetNativeBehaviorDistillationError(
            "payload artifact binding disagrees with artifact_sha256."
        )
    return BehaviorDomainSuite(
        split_name=str(split_name),
        prompts=tuple(prompts),
        artifact_sha256=artifact_sha256,
    )


@dataclass(frozen=True, slots=True)
class BehaviorDistillationSchedule:
    """Preregistered finite work and checkpoint bounds."""

    steps: int
    learning_rate: float
    checkpoint_steps: tuple[int, ...] = ()
    fit_horizon: int = 4
    evaluation_horizon: int = 8
    fit_prompts_per_step: int = 1
    temperature: float = 1.0
    action_loss_weight: float = 1.0
    distribution_loss_weight: float = 1.0
    action_margin_loss_weight: float = 0.0
    action_margin: float = 1.0
    repetition_excess_loss_weight: float = 0.0
    repetition_window: int = 8
    parent_lane_loss_weight: float = 1.0
    child_lane_loss_weight: float = 1.0
    parent_only_steps: int = 0
    warmup_fit_horizon: int = 0
    late_position_loss_multiplier: float = 1.0
    solved_prefix_loss_multiplier: float = 1.0
    divergence_loss_multiplier: float = 1.0
    post_divergence_loss_multiplier: float = 1.0
    post_divergence_positions: int = 2
    child_lane_starts_at_divergence: bool = False
    max_child_lane_prefixes: int = 0
    child_repeat_stop_run_length: int = 0
    qk_learning_rate_scale: float = 1.0
    vo_learning_rate_scale: float = 1.0
    gud_learning_rate_scale: float = 1.0
    lexical_embedding_learning_rate_scale: float = 1.0
    lexical_head_learning_rate_scale: float = 1.0
    right_factor_learning_rate_scale: float = 1.0
    lexical_right_factor_learning_rate_scale: float = 1.0
    guided_qk_learning_rate_scale: float = 1.0
    guided_vo_learning_rate_scale: float = 1.0
    guided_right_factor_learning_rate_scale: float = 1.0
    guided_joint_inherited_learning_rate_scale: float = 0.25
    guided_continuation_enabled: bool = False
    learning_rate_decay_step_1: int = 0
    learning_rate_decay_scale_1: float = 1.0
    learning_rate_decay_step_2: int = 0
    learning_rate_decay_scale_2: float = 1.0
    learning_rate_decay_step_3: int = 0
    learning_rate_decay_scale_3: float = 1.0
    qkvo_frozen_steps: int = 0
    gud_frozen_steps: int = 0
    lexical_embedding_frozen_steps: int = 0
    lexical_head_frozen_steps: int = 0
    guided_gud_unlock_step: int = 0
    guided_lexical_embedding_unlock_step: int = 0
    guided_lexical_head_unlock_step: int = 0
    top_k: int = 5
    gradient_clip_norm: float = 1.0
    max_prompt_tokens: int = 512
    max_train_prefixes: int = 100_000
    max_evaluation_prefixes: int = 1_000_000
    max_checkpoints: int = 16
    selection_policy: BehaviorSelectionPolicy = "strict-nonregression"
    sealed_evaluation_policy: BehaviorSealedEvaluationPolicy = "candidate-gated"

    def __post_init__(self) -> None:
        if not isinstance(self.guided_continuation_enabled, bool):
            raise TargetNativeBehaviorDistillationError(
                "guided_continuation_enabled must be boolean."
            )
        policy = str(self.selection_policy).strip().lower()
        if policy not in {"strict-nonregression", "rollout-first-acquisition"}:
            raise TargetNativeBehaviorDistillationError(
                "selection_policy must be strict-nonregression or "
                "rollout-first-acquisition."
            )
        object.__setattr__(self, "selection_policy", policy)
        sealed_policy = str(self.sealed_evaluation_policy).strip().lower()
        if sealed_policy not in {"candidate-gated", "disabled"}:
            raise TargetNativeBehaviorDistillationError(
                "sealed_evaluation_policy must be candidate-gated or disabled."
            )
        object.__setattr__(self, "sealed_evaluation_policy", sealed_policy)
        if not isinstance(self.child_lane_starts_at_divergence, bool):
            raise TargetNativeBehaviorDistillationError(
                "child_lane_starts_at_divergence must be boolean."
            )
        integer_fields = (
            "steps",
            "fit_horizon",
            "evaluation_horizon",
            "fit_prompts_per_step",
            "repetition_window",
            "top_k",
            "max_prompt_tokens",
            "max_train_prefixes",
            "max_evaluation_prefixes",
            "max_checkpoints",
        )
        for name in integer_fields:
            object.__setattr__(
                self, name, _index(getattr(self, name), name, positive=True)
            )
        if self.steps > 10_000:
            raise TargetNativeBehaviorDistillationError("steps exceeds 10,000.")
        if self.fit_horizon > 128 or self.evaluation_horizon > 128:
            raise TargetNativeBehaviorDistillationError("horizon exceeds 128 tokens.")
        parent_only = _index(self.parent_only_steps, "parent_only_steps")
        if parent_only >= self.steps:
            raise TargetNativeBehaviorDistillationError(
                "parent_only_steps must leave at least one child-on-policy step."
            )
        object.__setattr__(self, "parent_only_steps", parent_only)
        warmup_horizon = _index(self.warmup_fit_horizon, "warmup_fit_horizon")
        if warmup_horizon > self.fit_horizon:
            raise TargetNativeBehaviorDistillationError(
                "warmup_fit_horizon cannot exceed fit_horizon."
            )
        object.__setattr__(self, "warmup_fit_horizon", warmup_horizon)
        post_positions = _index(
            self.post_divergence_positions, "post_divergence_positions"
        )
        object.__setattr__(self, "post_divergence_positions", post_positions)
        for name in ("max_child_lane_prefixes", "child_repeat_stop_run_length"):
            value = _index(getattr(self, name), name)
            object.__setattr__(self, name, value)
        for name in (
            "learning_rate_decay_step_1",
            "learning_rate_decay_step_2",
            "learning_rate_decay_step_3",
            "qkvo_frozen_steps",
            "gud_frozen_steps",
            "lexical_embedding_frozen_steps",
            "lexical_head_frozen_steps",
            "guided_gud_unlock_step",
            "guided_lexical_embedding_unlock_step",
            "guided_lexical_head_unlock_step",
        ):
            value = _index(getattr(self, name), name)
            if value > self.steps:
                raise TargetNativeBehaviorDistillationError(
                    f"{name} cannot exceed steps."
                )
            object.__setattr__(self, name, value)
        for name in (
            "guided_gud_unlock_step",
            "guided_lexical_embedding_unlock_step",
            "guided_lexical_head_unlock_step",
        ):
            value = getattr(self, name)
            if value and value < GUIDED_JOINT_START_STEP:
                raise TargetNativeBehaviorDistillationError(
                    f"{name} cannot unlock before the joint guided phase at step "
                    f"{GUIDED_JOINT_START_STEP}."
                )
        if (
            self.learning_rate_decay_step_2
            and self.learning_rate_decay_step_2 <= self.learning_rate_decay_step_1
        ):
            raise TargetNativeBehaviorDistillationError(
                "learning-rate decay steps must be strictly increasing."
            )
        if self.learning_rate_decay_step_3 and (
            not self.learning_rate_decay_step_2
            or self.learning_rate_decay_step_3 <= self.learning_rate_decay_step_2
        ):
            raise TargetNativeBehaviorDistillationError(
                "The third learning-rate decay requires a smaller second decay step."
            )
        for name in (
            "learning_rate",
            "temperature",
            "action_loss_weight",
            "distribution_loss_weight",
            "action_margin_loss_weight",
            "action_margin",
            "repetition_excess_loss_weight",
            "parent_lane_loss_weight",
            "child_lane_loss_weight",
            "qk_learning_rate_scale",
            "vo_learning_rate_scale",
            "gud_learning_rate_scale",
            "lexical_embedding_learning_rate_scale",
            "lexical_head_learning_rate_scale",
            "right_factor_learning_rate_scale",
            "lexical_right_factor_learning_rate_scale",
            "guided_qk_learning_rate_scale",
            "guided_vo_learning_rate_scale",
            "guided_right_factor_learning_rate_scale",
            "guided_joint_inherited_learning_rate_scale",
            "learning_rate_decay_scale_1",
            "learning_rate_decay_scale_2",
            "learning_rate_decay_scale_3",
            "gradient_clip_norm",
        ):
            minimum = (
                0.0
                if name.endswith("loss_weight") or name == "action_margin"
                else 1e-12
            )
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=minimum),
            )
        object.__setattr__(
            self,
            "late_position_loss_multiplier",
            _finite_float(
                self.late_position_loss_multiplier,
                "late_position_loss_multiplier",
                minimum=1.0,
            ),
        )
        for name in (
            "solved_prefix_loss_multiplier",
            "divergence_loss_multiplier",
            "post_divergence_loss_multiplier",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=1e-12),
            )
        if self.action_loss_weight + self.distribution_loss_weight <= 0.0:
            raise TargetNativeBehaviorDistillationError(
                "At least one output loss weight must be positive."
            )
        if self.parent_lane_loss_weight <= 0.0:
            raise TargetNativeBehaviorDistillationError(
                "parent_lane_loss_weight must be positive."
            )
        steps = tuple(
            _index(value, "checkpoint_step") for value in self.checkpoint_steps
        )
        if any(value > self.steps for value in steps):
            raise TargetNativeBehaviorDistillationError(
                "checkpoint_steps cannot exceed steps."
            )
        object.__setattr__(self, "checkpoint_steps", tuple(sorted(set(steps))))
        if len(self.normalized_checkpoint_steps) > self.max_checkpoints:
            raise TargetNativeBehaviorDistillationError(
                "Checkpoint count exceeds max_checkpoints."
            )
        guided_defaults = {
            "guided_qk_learning_rate_scale": 1.0,
            "guided_vo_learning_rate_scale": 1.0,
            "guided_right_factor_learning_rate_scale": 1.0,
            "guided_joint_inherited_learning_rate_scale": 0.25,
            "guided_gud_unlock_step": 0,
            "guided_lexical_embedding_unlock_step": 0,
            "guided_lexical_head_unlock_step": 0,
        }
        if not self.guided_continuation_enabled and any(
            getattr(self, name) != default for name, default in guided_defaults.items()
        ):
            raise TargetNativeBehaviorDistillationError(
                "Guided schedule fields require guided_continuation_enabled=True."
            )

    @property
    def normalized_checkpoint_steps(self) -> tuple[int, ...]:
        return tuple(sorted({0, self.steps, *self.checkpoint_steps}))

    def as_dict(self) -> dict[str, Any]:
        guided_fields = {
            "guided_qk_learning_rate_scale",
            "guided_vo_learning_rate_scale",
            "guided_right_factor_learning_rate_scale",
            "guided_joint_inherited_learning_rate_scale",
            "guided_continuation_enabled",
            "guided_gud_unlock_step",
            "guided_lexical_embedding_unlock_step",
            "guided_lexical_head_unlock_step",
        }
        return {
            field_name: (
                list(self.normalized_checkpoint_steps)
                if field_name == "checkpoint_steps"
                else getattr(self, field_name)
            )
            for field_name in self.__dataclass_fields__
            if (
                (self.guided_continuation_enabled or field_name not in guided_fields)
                and (
                    field_name != "sealed_evaluation_policy"
                    or self.sealed_evaluation_policy != "candidate-gated"
                )
            )
        }


@dataclass(frozen=True, slots=True)
class CandidateBehaviorThresholds:
    """Preregistered calibration gate that alone can unlock sealed evidence."""

    minimum_action_accuracy: float = 0.0
    minimum_exact_rollout_fraction: float = 0.0
    minimum_token_match_fraction: float = 0.0
    minimum_lcp_fraction: float = 0.0
    minimum_action_count: int = 1
    minimum_rollout_count: int = 1
    maximum_repeat_fraction_excess: float = 1.0
    maximum_adjacent_repeat_fraction_excess: float = 1.0
    maximum_longest_run_excess: int = 2_147_483_647

    def __post_init__(self) -> None:
        for name in (
            "minimum_action_accuracy",
            "minimum_exact_rollout_fraction",
            "minimum_token_match_fraction",
            "minimum_lcp_fraction",
            "maximum_repeat_fraction_excess",
            "maximum_adjacent_repeat_fraction_excess",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        for name in (
            "minimum_action_count",
            "minimum_rollout_count",
            "maximum_longest_run_excess",
        ):
            object.__setattr__(
                self,
                name,
                _index(
                    getattr(self, name),
                    name,
                    positive=name != "maximum_longest_run_excess",
                ),
            )

    def passes(self, metrics: CausalBehaviorMetrics) -> bool:
        if not isinstance(metrics, CausalBehaviorMetrics):
            raise TargetNativeBehaviorDistillationError(
                "metrics must be CausalBehaviorMetrics."
            )
        return (
            metrics.action_count >= self.minimum_action_count
            and metrics.rollout_count >= self.minimum_rollout_count
            and metrics.action_accuracy >= self.minimum_action_accuracy
            and metrics.exact_rollout_fraction >= self.minimum_exact_rollout_fraction
            and metrics.token_match_fraction >= self.minimum_token_match_fraction
            and metrics.lcp_fraction >= self.minimum_lcp_fraction
            and metrics.repeat_fraction_excess <= self.maximum_repeat_fraction_excess
            and metrics.adjacent_repeat_fraction_excess
            <= self.maximum_adjacent_repeat_fraction_excess
            and metrics.longest_run_excess <= self.maximum_longest_run_excess
        )

    def as_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CausalBehaviorMetrics:
    """Action and free-rollout metrics derived from output traces only."""

    split: BehaviorEvaluationSplit
    action_count: int
    action_agreement_count: int
    action_accuracy: float
    parent_action_mean_logprob: float
    parent_action_topk_fraction: float
    parent_action_mean_rank: float
    parent_action_worst_rank: int
    rollout_count: int
    exact_rollout_count: int
    exact_rollout_fraction: float
    rollout_token_count: int
    token_match_count: int
    token_match_fraction: float
    lcp_token_count: int
    lcp_fraction: float
    lcp_at_least_4_count: int
    repeat_fraction_excess: float
    adjacent_repeat_fraction_excess: float
    longest_run_excess: int
    report_sha256: str

    @classmethod
    def from_report(
        cls, report: ResidualAttentionFunctionalReport
    ) -> CausalBehaviorMetrics:
        forced = report.forced
        greedy = report.greedy
        return cls(
            split=report.split,
            action_count=forced.action_count,
            action_agreement_count=forced.candidate_parent_top1_agreement_count,
            action_accuracy=forced.candidate_parent_top1_agreement_rate,
            parent_action_mean_logprob=forced.candidate_parent_action_mean_logprob,
            parent_action_topk_fraction=forced.candidate_parent_action_topk_rate,
            parent_action_mean_rank=forced.candidate_parent_action_mean_rank,
            parent_action_worst_rank=forced.candidate_parent_action_worst_rank,
            rollout_count=greedy.sequence_count,
            exact_rollout_count=greedy.exact_rollout_match_count,
            exact_rollout_fraction=greedy.exact_rollout_match_rate,
            rollout_token_count=greedy.parent_token_count,
            token_match_count=greedy.positional_match_count,
            token_match_fraction=greedy.exact_token_retention_rate,
            lcp_token_count=greedy.longest_common_prefix_tokens,
            lcp_fraction=greedy.longest_common_prefix_rate,
            lcp_at_least_4_count=sum(
                value.longest_common_prefix >= 4 for value in greedy.sequences
            ),
            repeat_fraction_excess=greedy.candidate_repeat_fraction_excess,
            adjacent_repeat_fraction_excess=(
                greedy.candidate_adjacent_repeat_fraction_excess
            ),
            longest_run_excess=greedy.candidate_longest_run_excess,
            report_sha256=report.report_sha256,
        )

    @property
    def pareto_objectives(self) -> tuple[float, ...]:
        return (
            self.action_accuracy,
            self.parent_action_mean_logprob,
            self.parent_action_topk_fraction,
            -self.parent_action_mean_rank,
            -float(self.parent_action_worst_rank),
            self.token_match_fraction,
            self.lcp_fraction,
            self.exact_rollout_fraction,
            -self.repeat_fraction_excess,
            -self.adjacent_repeat_fraction_excess,
            -float(self.longest_run_excess),
        )

    @property
    def selection_key(self) -> tuple[float, ...]:
        return tuple(-value for value in self.pareto_objectives)

    @property
    def rollout_first_acquisition_key(self) -> tuple[int, ...]:
        """Ascending key for acquiring behavior in an initially blank stage."""

        return (
            -self.exact_rollout_count,
            -self.lcp_at_least_4_count,
            -self.lcp_token_count,
            -self.token_match_count,
            round(self.repeat_fraction_excess * 1_000_000),
            round(self.adjacent_repeat_fraction_excess * 1_000_000),
            self.longest_run_excess,
            -self.action_agreement_count,
        )

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class GuidedFactorSourceBinding:
    """Immutable byte provenance for one v2 + guided-basis continuation."""

    source_v2_factor_state_sha256: str
    source_v2_tensor_sha256s: tuple[tuple[str, str], ...]
    audit_report_sha256: str
    basis_envelope_file_sha256: str
    basis_payload_sha256: str
    basis_manifest_sha256: str
    source_basis_tensor_sha256s: tuple[tuple[str, str], ...]
    inherited_installation_sha256: str
    inherited_factor_state_sha256_at_install: str
    adapter_site_indices_sha256: str
    audit_site_indices_sha256: str
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_v2_factor_state_sha256",
            _sha256(
                self.source_v2_factor_state_sha256,
                "source_v2_factor_state_sha256",
            ),
        )
        for name in (
            "audit_report_sha256",
            "basis_envelope_file_sha256",
            "basis_payload_sha256",
            "basis_manifest_sha256",
            "inherited_installation_sha256",
            "inherited_factor_state_sha256_at_install",
            "adapter_site_indices_sha256",
            "audit_site_indices_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        v2_hashes = self._normalize_hash_rows(
            self.source_v2_tensor_sha256s,
            "source_v2_tensor_sha256s",
            expected_names=_QKVO_GUD_FACTOR_NAMES | _LEXICAL_FACTOR_NAMES,
        )
        basis_hashes = self._normalize_hash_rows(
            self.source_basis_tensor_sha256s,
            "source_basis_tensor_sha256s",
            expected_names={
                "q",
                "k",
                "v",
                "o",
            },
        )
        object.__setattr__(self, "source_v2_tensor_sha256s", v2_hashes)
        object.__setattr__(self, "source_basis_tensor_sha256s", basis_hashes)
        object.__setattr__(self, "binding_sha256", _canonical_sha256(self.as_dict()))

    @staticmethod
    def _normalize_hash_rows(
        values: Any,
        name: str,
        *,
        expected_names: set[str] | frozenset[str],
    ) -> tuple[tuple[str, str], ...]:
        if isinstance(values, Mapping):
            rows = tuple(values.items())
        elif not isinstance(values, (str, bytes)) and isinstance(values, Sequence):
            rows = tuple(values)
        else:
            raise TargetNativeBehaviorDistillationError(
                f"{name} must be a name-to-SHA mapping."
            )
        normalized: dict[str, str] = {}
        for index, row in enumerate(rows):
            if (
                isinstance(row, (str, bytes))
                or not isinstance(row, Sequence)
                or len(row) != 2
            ):
                raise TargetNativeBehaviorDistillationError(
                    f"{name}[{index}] must be a name/SHA pair."
                )
            factor_name = str(row[0])
            if factor_name in normalized:
                raise TargetNativeBehaviorDistillationError(
                    f"{name} contains duplicate factor {factor_name!r}."
                )
            normalized[factor_name] = _sha256(row[1], f"{name}[{factor_name!r}]")
        if set(normalized) != set(expected_names):
            raise TargetNativeBehaviorDistillationError(
                f"{name} does not contain exactly the required factor names."
            )
        return tuple(sorted(normalized.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_v2_factor_state_sha256": self.source_v2_factor_state_sha256,
            "source_v2_tensor_sha256s": dict(self.source_v2_tensor_sha256s),
            "audit_report_sha256": self.audit_report_sha256,
            "basis_envelope_file_sha256": self.basis_envelope_file_sha256,
            "basis_payload_sha256": self.basis_payload_sha256,
            "basis_manifest_sha256": self.basis_manifest_sha256,
            "source_basis_tensor_sha256s": dict(self.source_basis_tensor_sha256s),
            "inherited_installation_sha256": self.inherited_installation_sha256,
            "inherited_factor_state_sha256_at_install": (
                self.inherited_factor_state_sha256_at_install
            ),
            "adapter_site_indices_sha256": self.adapter_site_indices_sha256,
            "audit_site_indices_sha256": self.audit_site_indices_sha256,
        }


def _portable_guided_factor_source_binding(
    binding: GuidedFactorSourceBinding,
) -> dict[str, Any]:
    """Return the content-bearing v3 source fields used by nested v4 identity."""

    if not isinstance(binding, GuidedFactorSourceBinding):
        raise TargetNativeBehaviorDistillationError(
            "Portable guided source identity requires an authenticated binding."
        )
    payload = binding.as_dict()
    payload.pop("inherited_installation_sha256")
    return payload


@dataclass(frozen=True, slots=True)
class GuidedGUDFactorSourceBinding:
    """Immutable v3 + output-Jacobian GUD provenance for one v4 continuation.

    The legacy v3 digest and nested QKVO binding retain their live-installation
    receipts for exact rollback.  ``source_v3_content_sha256`` separately binds
    the same source tensor manifest and the portable QKVO provenance so recovery
    can prove identical content across processes without treating object/storage
    identity as model content.
    """

    source_v3_factor_state_sha256: str
    source_v3_factor_manifest_sha256: str
    source_v3_tensor_sha256s: tuple[tuple[str, str], ...]
    source_v3_lexical_rank: int
    source_guided_qkvo_binding: GuidedFactorSourceBinding
    audit_report_sha256: str
    basis_envelope_file_sha256: str
    basis_payload_sha256: str
    basis_manifest_sha256: str
    source_basis_tensor_sha256s: tuple[tuple[str, str], ...]
    inherited_installation_sha256: str
    inherited_factor_state_sha256_at_install: str
    adapter_site_indices_sha256: str
    audit_site_indices_sha256: str
    source_v3_content_sha256: str = field(init=False)
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_v3_factor_state_sha256",
            "source_v3_factor_manifest_sha256",
            "audit_report_sha256",
            "basis_envelope_file_sha256",
            "basis_payload_sha256",
            "basis_manifest_sha256",
            "inherited_installation_sha256",
            "inherited_factor_state_sha256_at_install",
            "adapter_site_indices_sha256",
            "audit_site_indices_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        lexical_rank = _index(
            self.source_v3_lexical_rank,
            "source_v3_lexical_rank",
            positive=True,
        )
        object.__setattr__(self, "source_v3_lexical_rank", lexical_rank)
        if not isinstance(self.source_guided_qkvo_binding, GuidedFactorSourceBinding):
            raise TargetNativeBehaviorDistillationError(
                "A guided-GUD source requires the authenticated source QKVO binding."
            )
        source_hashes = GuidedFactorSourceBinding._normalize_hash_rows(
            self.source_v3_tensor_sha256s,
            "source_v3_tensor_sha256s",
            expected_names=(
                _QKVO_GUD_FACTOR_NAMES
                | _LEXICAL_FACTOR_NAMES
                | _GUIDED_QKVO_FACTOR_NAMES
            ),
        )
        basis_hashes = GuidedFactorSourceBinding._normalize_hash_rows(
            self.source_basis_tensor_sha256s,
            "source_basis_tensor_sha256s",
            expected_names={"gate", "up", "down"},
        )
        object.__setattr__(self, "source_v3_tensor_sha256s", source_hashes)
        object.__setattr__(self, "source_basis_tensor_sha256s", basis_hashes)
        content_sha = _canonical_sha256(
            {
                "schema": "neural-forge-guided-gud-source-content-v1",
                "source_v3_factor_manifest_sha256": (
                    self.source_v3_factor_manifest_sha256
                ),
                "source_v3_tensor_sha256s": dict(source_hashes),
                "source_v3_lexical_rank": lexical_rank,
                "portable_source_guided_qkvo_binding": (
                    _portable_guided_factor_source_binding(
                        self.source_guided_qkvo_binding
                    )
                ),
            }
        )
        object.__setattr__(self, "source_v3_content_sha256", content_sha)
        object.__setattr__(self, "binding_sha256", _canonical_sha256(self.as_dict()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_v3_factor_state_sha256": self.source_v3_factor_state_sha256,
            "source_v3_factor_manifest_sha256": (self.source_v3_factor_manifest_sha256),
            "source_v3_tensor_sha256s": dict(self.source_v3_tensor_sha256s),
            "source_v3_lexical_rank": self.source_v3_lexical_rank,
            "source_guided_qkvo_binding": self.source_guided_qkvo_binding.as_dict(),
            "source_guided_qkvo_binding_sha256": (
                self.source_guided_qkvo_binding.binding_sha256
            ),
            "source_v3_content_sha256": self.source_v3_content_sha256,
            "audit_report_sha256": self.audit_report_sha256,
            "basis_envelope_file_sha256": self.basis_envelope_file_sha256,
            "basis_payload_sha256": self.basis_payload_sha256,
            "basis_manifest_sha256": self.basis_manifest_sha256,
            "source_basis_tensor_sha256s": dict(self.source_basis_tensor_sha256s),
            "inherited_installation_sha256": self.inherited_installation_sha256,
            "inherited_factor_state_sha256_at_install": (
                self.inherited_factor_state_sha256_at_install
            ),
            "adapter_site_indices_sha256": self.adapter_site_indices_sha256,
            "audit_site_indices_sha256": self.audit_site_indices_sha256,
        }


def _guided_gud_source_binding_from_payload(
    value: Any,
) -> GuidedGUDFactorSourceBinding:
    expected_keys = {
        "source_v3_factor_state_sha256",
        "source_v3_factor_manifest_sha256",
        "source_v3_tensor_sha256s",
        "source_v3_lexical_rank",
        "source_guided_qkvo_binding",
        "source_guided_qkvo_binding_sha256",
        "source_v3_content_sha256",
        "audit_report_sha256",
        "basis_envelope_file_sha256",
        "basis_payload_sha256",
        "basis_manifest_sha256",
        "source_basis_tensor_sha256s",
        "inherited_installation_sha256",
        "inherited_factor_state_sha256_at_install",
        "adapter_site_indices_sha256",
        "audit_site_indices_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise TargetNativeBehaviorDistillationError(
            "Persisted v4 guided-GUD source binding has an unsupported schema."
        )
    raw = dict(value)
    nested_raw = raw.pop("source_guided_qkvo_binding")
    nested_sha = raw.pop("source_guided_qkvo_binding_sha256")
    declared_content_sha = raw.pop("source_v3_content_sha256")
    if not isinstance(nested_raw, Mapping):
        raise TargetNativeBehaviorDistillationError(
            "Persisted v4 guided-GUD source has no nested QKVO binding."
        )
    try:
        nested = GuidedFactorSourceBinding(**dict(nested_raw))
        result = GuidedGUDFactorSourceBinding(
            source_guided_qkvo_binding=nested,
            **raw,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise TargetNativeBehaviorDistillationError(
            "Persisted v4 guided-GUD source binding failed authentication."
        ) from error
    if (
        nested_sha != nested.binding_sha256
        or declared_content_sha != result.source_v3_content_sha256
        or _canonical_sha256(nested_raw) != nested.binding_sha256
        or _canonical_sha256(value) != result.binding_sha256
        or dict(value) != result.as_dict()
    ):
        raise TargetNativeBehaviorDistillationError(
            "Persisted v4 guided-GUD source binding hash does not authenticate."
        )
    return result


@dataclass(frozen=True, slots=True)
class TargetNativeFactorState:
    """A CPU clone of only the temporary target-native factor tensors."""

    names: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    tensor_sha256s: tuple[str, ...]
    factor_state_sha256: str
    parameter_ids: tuple[int, ...] = ()
    parameter_data_ptrs: tuple[int, ...] = ()
    schema: str = BEHAVIOR_FACTOR_PACK_SCHEMA
    lexical_rank: int | None = None
    guided_source_binding: GuidedFactorSourceBinding | None = None
    guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None

    def as_manifest(self) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "factor_count": len(self.names),
            "factors": [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "tensor_sha256": digest,
                }
                for name, tensor, digest in zip(
                    self.names, self.tensors, self.tensor_sha256s, strict=True
                )
            ],
            "factor_state_sha256": self.factor_state_sha256,
            "parent_state_included": False,
            "native_child_state_included": False,
            "optimizer_state_included": False,
        }
        if self.schema in {
            BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }:
            result["lexical_rank"] = _index(
                self.lexical_rank, "lexical_rank", positive=True
            )
        if self.schema in {
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }:
            if not isinstance(self.guided_source_binding, GuidedFactorSourceBinding):
                raise TargetNativeBehaviorDistillationError(
                    "A v3 factor state requires authenticated guided sources."
                )
            result["guided_source_binding"] = self.guided_source_binding.as_dict()
            result["guided_source_binding_sha256"] = (
                self.guided_source_binding.binding_sha256
            )
            result["inherited_product_regrouped"] = False
            result["guided_product_regrouped"] = False
        if self.schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA:
            if not isinstance(
                self.guided_gud_source_binding, GuidedGUDFactorSourceBinding
            ):
                raise TargetNativeBehaviorDistillationError(
                    "A v4 factor state requires authenticated guided-GUD sources."
                )
            if (
                self.guided_gud_source_binding.source_guided_qkvo_binding
                != self.guided_source_binding
            ):
                raise TargetNativeBehaviorDistillationError(
                    "A v4 factor state's nested and top-level QKVO bindings differ."
                )
            result["guided_gud_source_binding"] = (
                self.guided_gud_source_binding.as_dict()
            )
            result["guided_gud_source_binding_sha256"] = (
                self.guided_gud_source_binding.binding_sha256
            )
            result["guided_gud_product_regrouped"] = False
        return result

    def as_payload(self) -> dict[str, Any]:
        return {
            **self.as_manifest(),
            "factor_tensors": {
                name: tensor.detach().cpu().clone()
                for name, tensor in zip(self.names, self.tensors, strict=True)
            },
        }


def _factor_manifest_rows(
    names: Sequence[str],
    tensors: Sequence[Tensor],
    tensor_sha256s: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": digest,
        }
        for name, value, digest in zip(names, tensors, tensor_sha256s, strict=True)
    ]


def _factor_state_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: str,
    lexical_rank: int | None,
    guided_source_binding: GuidedFactorSourceBinding | None = None,
    guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None,
) -> str:
    if schema == BEHAVIOR_FACTOR_PACK_SCHEMA:
        return _canonical_sha256(list(rows))
    if schema == BEHAVIOR_FACTOR_PACK_V2_SCHEMA:
        return _canonical_sha256(
            {
                "schema": schema,
                "lexical_rank": _index(lexical_rank, "lexical_rank", positive=True),
                "factors": list(rows),
            }
        )
    if schema == BEHAVIOR_FACTOR_PACK_V3_SCHEMA:
        if not isinstance(guided_source_binding, GuidedFactorSourceBinding):
            raise TargetNativeBehaviorDistillationError(
                "A v3 factor-state hash requires authenticated guided sources."
            )
        return _canonical_sha256(
            {
                "schema": schema,
                "lexical_rank": _index(lexical_rank, "lexical_rank", positive=True),
                "factors": list(rows),
                "guided_source_binding": guided_source_binding.as_dict(),
                "guided_source_binding_sha256": (guided_source_binding.binding_sha256),
                "inherited_product_regrouped": False,
                "guided_product_regrouped": False,
            }
        )
    if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA:
        if not isinstance(guided_source_binding, GuidedFactorSourceBinding) or not (
            isinstance(guided_gud_source_binding, GuidedGUDFactorSourceBinding)
        ):
            raise TargetNativeBehaviorDistillationError(
                "A v4 factor-state hash requires both authenticated guided source "
                "bindings."
            )
        return _canonical_sha256(
            {
                "schema": schema,
                "lexical_rank": _index(lexical_rank, "lexical_rank", positive=True),
                "factors": list(rows),
                "guided_source_binding": guided_source_binding.as_dict(),
                "guided_source_binding_sha256": (guided_source_binding.binding_sha256),
                "guided_gud_source_binding": guided_gud_source_binding.as_dict(),
                "guided_gud_source_binding_sha256": (
                    guided_gud_source_binding.binding_sha256
                ),
                "inherited_product_regrouped": False,
                "guided_product_regrouped": False,
                "guided_gud_product_regrouped": False,
            }
        )
    raise TargetNativeBehaviorDistillationError(
        f"Unsupported target-native factor-state schema {schema!r}."
    )


def _lexical_handle(
    transaction: ExactTargetCompilerTransaction,
    *,
    require_finite: bool,
) -> Any | None:
    handle = getattr(transaction, "lexical_handle", None)
    if handle is None:
        return None
    assertion_name = "assert_live" if require_finite else "assert_structurally_live"
    assertion = getattr(handle, assertion_name, None)
    if not callable(assertion):
        raise TargetNativeBehaviorDistillationError(
            "The optional lexical handle does not expose the required live-state "
            "assertion."
        )
    assertion()
    return handle


def _lexical_registry(handle: Any) -> nn.Module:
    registry = getattr(handle, "registry", None)
    if registry is None:
        registry = getattr(handle, "adapter_registry", None)
    if not isinstance(registry, nn.Module):
        raise TargetNativeBehaviorDistillationError(
            "The optional lexical handle has no factor registry."
        )
    return registry


def _lexical_rank(transaction: ExactTargetCompilerTransaction) -> int | None:
    handle = getattr(transaction, "lexical_handle", None)
    if handle is None:
        return None
    registry = _lexical_registry(handle)
    rank = getattr(registry, "rank", None)
    return _index(rank, "lexical_rank", positive=True)


def _factor_pack_contract(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any | None = None,
    guided_gud_tranche: Any | None = None,
) -> tuple[str, int | None]:
    rank = _lexical_rank(transaction)
    if guided_gud_tranche is not None:
        if guided_tranche is None or rank is None:
            raise TargetNativeBehaviorDistillationError(
                "A guided-GUD v4 continuation requires the complete v3 QKVO "
                "continuation and inherited lexical transaction."
            )
        return BEHAVIOR_FACTOR_PACK_V4_SCHEMA, rank
    if guided_tranche is not None:
        if rank is None:
            raise TargetNativeBehaviorDistillationError(
                "A guided v3 continuation requires the complete inherited v2 "
                "lexical transaction."
            )
        return BEHAVIOR_FACTOR_PACK_V3_SCHEMA, rank
    if rank is None:
        return BEHAVIOR_FACTOR_PACK_SCHEMA, None
    return BEHAVIOR_FACTOR_PACK_V2_SCHEMA, rank


def _factor_parameters(
    transaction: ExactTargetCompilerTransaction,
    *,
    require_finite: bool = True,
    guided_tranche: Any | None = None,
    guided_gud_tranche: Any | None = None,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if not isinstance(transaction, ExactTargetCompilerTransaction):
        raise TargetNativeBehaviorDistillationError(
            "transaction must be ExactTargetCompilerTransaction."
        )
    if not transaction.is_live:
        raise TargetNativeBehaviorDistillationError(
            "Compiler transaction is no longer structurally live."
        )
    if require_finite:
        transaction.assert_live()
    else:
        transaction.assert_structurally_live()
    entries = tuple(
        (f"qkvo.{name}", value)
        for name, value in transaction.qkvo_handle.adapter_registry.named_parameters(
            recurse=False
        )
    ) + tuple(
        (f"gud.{name}", value)
        for name, value in transaction.swiglu_handle.registry.named_parameters(
            recurse=False
        )
    )
    lexical = _lexical_handle(transaction, require_finite=require_finite)
    expected_names = set(_QKVO_GUD_FACTOR_NAMES)
    if lexical is not None:
        registry = _lexical_registry(lexical)
        entries += tuple(
            (f"lexical.{name}", value)
            for name, value in registry.named_parameters(recurse=False)
        )
        expected_names.update(_LEXICAL_FACTOR_NAMES)
    if {name for name, _value in entries} != expected_names:
        raise TargetNativeBehaviorDistillationError(
            "Compiler transaction does not expose one complete target-native "
            "factor set."
        )
    # The composite accessor includes the optional lexical registry.  Factor
    # enumeration above remains structural so emergency restoration can use
    # the non-finite-tolerant transaction assertion before overwriting a
    # poisoned disposable factor.
    trainable = (
        transaction.output_trainable_parameters()
        if require_finite
        else tuple(value for _name, value in entries)
    )
    if {id(value) for _name, value in entries} != {id(value) for value in trainable}:
        raise TargetNativeBehaviorDistillationError(
            "Named factors disagree with the compiler training scope."
        )
    if guided_gud_tranche is not None and guided_tranche is None:
        raise TargetNativeBehaviorDistillationError(
            "guided_gud_tranche requires the recovered guided QKVO tranche."
        )
    if guided_tranche is not None:
        assertion_name = "assert_live" if require_finite else "assert_structurally_live"
        assertion = getattr(guided_tranche, assertion_name, None)
        if not callable(assertion):
            raise TargetNativeBehaviorDistillationError(
                "guided_tranche does not expose the required live-state assertion."
            )
        assertion()
        if getattr(guided_tranche, "inherited_handle", None) is not (
            transaction.qkvo_handle
        ):
            raise TargetNativeBehaviorDistillationError(
                "guided_tranche is not bound to this transaction's inherited QKVO "
                "product."
            )
        registry = getattr(guided_tranche, "guided_registry", None)
        if not isinstance(registry, nn.Module):
            raise TargetNativeBehaviorDistillationError(
                "guided_tranche has no guided factor registry."
            )
        guided_entries = tuple(
            (f"guided_qkvo.{name}", value)
            for name, value in registry.named_parameters(recurse=False)
        )
        if {name for name, _value in guided_entries} != _GUIDED_QKVO_FACTOR_NAMES:
            raise TargetNativeBehaviorDistillationError(
                "guided_tranche does not expose exactly eight separate Q/K/V/O "
                "left/right factors."
            )
        if any(not isinstance(value, nn.Parameter) for _name, value in guided_entries):
            raise TargetNativeBehaviorDistillationError(
                "Every guided factor must be an nn.Parameter."
            )
        if sum(int(value.numel()) for _name, value in guided_entries) != (
            GUIDED_QKVO_PARAMETER_COUNT
        ):
            raise TargetNativeBehaviorDistillationError(
                "The guided tranche must contain exactly 7,127,040 FP32 parameters."
            )
        if {value.dtype for _name, value in guided_entries} != {torch.float32}:
            raise TargetNativeBehaviorDistillationError(
                "Every guided factor must use FP32."
            )
        if require_finite and any(
            not bool(torch.isfinite(value.detach()).all())
            for _name, value in guided_entries
        ):
            raise TargetNativeBehaviorDistillationError(
                "A guided factor contains NaN or infinity."
            )
        if {id(value) for _name, value in entries} & {
            id(value) for _name, value in guided_entries
        }:
            raise TargetNativeBehaviorDistillationError(
                "Inherited and guided products unexpectedly share parameters."
            )
        entries += guided_entries
        expected_combined_names = (
            _QKVO_GUD_FACTOR_NAMES | _LEXICAL_FACTOR_NAMES | _GUIDED_QKVO_FACTOR_NAMES
        )
        if {name for name, _value in entries} != expected_combined_names:
            raise TargetNativeBehaviorDistillationError(
                "The guided continuation must expose exactly 26 factor tensors."
            )
    if guided_gud_tranche is not None:
        assertion_name = "assert_live" if require_finite else "assert_structurally_live"
        assertion = getattr(guided_gud_tranche, assertion_name, None)
        if not callable(assertion):
            raise TargetNativeBehaviorDistillationError(
                "guided_gud_tranche does not expose the required live-state assertion."
            )
        assertion()
        inherited_handle = getattr(guided_gud_tranche, "inherited_handle", None)
        if inherited_handle is not transaction.swiglu_handle:
            raise TargetNativeBehaviorDistillationError(
                "guided_gud_tranche is not bound to this transaction's inherited "
                "Gate/Up/Down product."
            )
        registry = getattr(guided_gud_tranche, "guided_registry", None)
        if registry is None:
            registry = getattr(guided_gud_tranche, "registry", None)
        if not isinstance(registry, nn.Module):
            raise TargetNativeBehaviorDistillationError(
                "guided_gud_tranche has no guided factor registry."
            )
        guided_gud_entries = tuple(
            (f"guided_gud.{name}", value)
            for name, value in registry.named_parameters(recurse=False)
        )
        if {name for name, _value in guided_gud_entries} != _GUIDED_GUD_FACTOR_NAMES:
            raise TargetNativeBehaviorDistillationError(
                "guided_gud_tranche does not expose exactly six separate "
                "Gate/Up/Down left/right factors."
            )
        if any(
            not isinstance(value, nn.Parameter) for _name, value in guided_gud_entries
        ):
            raise TargetNativeBehaviorDistillationError(
                "Every guided-GUD factor must be an nn.Parameter."
            )
        if {value.dtype for _name, value in guided_gud_entries} != {torch.float32}:
            raise TargetNativeBehaviorDistillationError(
                "Every guided-GUD factor must use FP32."
            )
        if any(int(value.numel()) <= 0 for _name, value in guided_gud_entries):
            raise TargetNativeBehaviorDistillationError(
                "Every guided-GUD factor must be non-empty."
            )
        if require_finite and any(
            not bool(torch.isfinite(value.detach()).all())
            for _name, value in guided_gud_entries
        ):
            raise TargetNativeBehaviorDistillationError(
                "A guided-GUD factor contains NaN or infinity."
            )
        if {id(value) for _name, value in entries} & {
            id(value) for _name, value in guided_gud_entries
        }:
            raise TargetNativeBehaviorDistillationError(
                "The v3 source and guided-GUD product unexpectedly share parameters."
            )
        entries += guided_gud_entries
        expected_v4_names = (
            _QKVO_GUD_FACTOR_NAMES
            | _LEXICAL_FACTOR_NAMES
            | _GUIDED_QKVO_FACTOR_NAMES
            | _GUIDED_GUD_FACTOR_NAMES
        )
        if {name for name, _value in entries} != expected_v4_names or len(
            entries
        ) != GUIDED_GUD_COMBINED_FACTOR_TENSOR_COUNT:
            raise TargetNativeBehaviorDistillationError(
                "The guided-GUD continuation must expose exactly 32 factor tensors."
            )
    return tuple(sorted(entries, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class _LiveFactorBinding:
    """Private ownership/storage guard around one trusted callback invocation."""

    qualified_name: str
    registry: nn.Module
    parameter_name: str
    parameter: nn.Parameter
    storage_alias: Tensor
    data_ptr: int
    requires_grad: bool


def _live_factor_bindings(
    transaction: ExactTargetCompilerTransaction,
    *,
    guided_tranche: Any | None,
    guided_gud_tranche: Any | None = None,
) -> tuple[_LiveFactorBinding, ...]:
    """Capture live factor owners without exposing them to the callback event."""

    registries: list[tuple[str, nn.Module]] = [
        ("qkvo", transaction.qkvo_handle.adapter_registry),
        ("gud", transaction.swiglu_handle.registry),
    ]
    lexical = _lexical_handle(transaction, require_finite=True)
    if lexical is not None:
        registries.append(("lexical", _lexical_registry(lexical)))
    if guided_tranche is not None:
        guided_registry = getattr(guided_tranche, "guided_registry", None)
        if not isinstance(guided_registry, nn.Module):
            raise TargetNativeBehaviorDistillationError(
                "guided_tranche has no guided factor registry."
            )
        registries.append(("guided_qkvo", guided_registry))
    if guided_gud_tranche is not None:
        guided_gud_registry = getattr(guided_gud_tranche, "guided_registry", None)
        if guided_gud_registry is None:
            guided_gud_registry = getattr(guided_gud_tranche, "registry", None)
        if not isinstance(guided_gud_registry, nn.Module):
            raise TargetNativeBehaviorDistillationError(
                "guided_gud_tranche has no guided factor registry."
            )
        registries.append(("guided_gud", guided_gud_registry))

    owners: dict[str, tuple[nn.Module, str, nn.Parameter]] = {}
    for prefix, registry in registries:
        for parameter_name, parameter in registry.named_parameters(recurse=False):
            qualified_name = f"{prefix}.{parameter_name}"
            if qualified_name in owners:
                raise TargetNativeBehaviorDistillationError(
                    "Live factor ownership contains a duplicate name."
                )
            owners[qualified_name] = (registry, parameter_name, parameter)

    entries = _factor_parameters(
        transaction,
        guided_tranche=guided_tranche,
        guided_gud_tranche=guided_gud_tranche,
    )
    if set(owners) != {name for name, _parameter in entries}:
        raise TargetNativeBehaviorDistillationError(
            "Live factor ownership does not match the canonical factor set."
        )
    result: list[_LiveFactorBinding] = []
    for qualified_name, parameter in entries:
        registry, parameter_name, owned_parameter = owners[qualified_name]
        if owned_parameter is not parameter:
            raise TargetNativeBehaviorDistillationError(
                "Live factor enumeration disagrees with registry ownership."
            )
        result.append(
            _LiveFactorBinding(
                qualified_name=qualified_name,
                registry=registry,
                parameter_name=parameter_name,
                parameter=parameter,
                # Keep the original storage alive if trusted sink code
                # accidentally replaces ``Parameter.data``.
                storage_alias=parameter.detach(),
                data_ptr=int(parameter.data_ptr()),
                requires_grad=bool(parameter.requires_grad),
            )
        )
    return tuple(result)


def _repair_live_factor_bindings(
    bindings: Sequence[_LiveFactorBinding],
) -> bool:
    """Repair callback-time owner/storage replacement and report any mutation."""

    changed = False
    for binding in bindings:
        current = binding.registry._parameters.get(binding.parameter_name)
        if current is not binding.parameter:
            binding.registry._parameters[binding.parameter_name] = binding.parameter
            changed = True
        if int(binding.parameter.data_ptr()) != binding.data_ptr:
            binding.parameter.data = binding.storage_alias
            changed = True
        if bool(binding.parameter.requires_grad) != binding.requires_grad:
            binding.parameter.requires_grad_(binding.requires_grad)
            changed = True
    return changed


def authenticate_guided_factor_sources(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    source_v2_state: TargetNativeFactorState,
) -> GuidedFactorSourceBinding:
    """Bind all inherited v2 bytes and installed audit-basis bytes before step 1."""

    if (
        not isinstance(source_v2_state, TargetNativeFactorState)
        or source_v2_state.schema != BEHAVIOR_FACTOR_PACK_V2_SCHEMA
        or len(source_v2_state.names) != 18
        or len(source_v2_state.tensors) != 18
        or len(source_v2_state.tensor_sha256s) != 18
        or set(source_v2_state.names) != _QKVO_GUD_FACTOR_NAMES | _LEXICAL_FACTOR_NAMES
    ):
        raise TargetNativeBehaviorDistillationError(
            "A guided continuation requires one authenticated complete v2 source "
            "factor state."
        )
    source_v2_rows = _factor_manifest_rows(
        source_v2_state.names,
        source_v2_state.tensors,
        source_v2_state.tensor_sha256s,
    )
    if (
        any(
            _tensor_sha256(tensor) != digest
            for tensor, digest in zip(
                source_v2_state.tensors,
                source_v2_state.tensor_sha256s,
                strict=True,
            )
        )
        or _factor_state_sha256(
            source_v2_rows,
            schema=BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            lexical_rank=source_v2_state.lexical_rank,
        )
        != source_v2_state.factor_state_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "The inherited v2 source state failed byte authentication."
        )
    entries = _factor_parameters(
        transaction,
        guided_tranche=guided_tranche,
    )
    if len(entries) != GUIDED_COMBINED_FACTOR_TENSOR_COUNT:
        raise TargetNativeBehaviorDistillationError(
            "A guided continuation must expose exactly 26 factor tensors."
        )
    source_accessor = getattr(guided_tranche, "factor_source_binding", None)
    if not callable(source_accessor):
        raise TargetNativeBehaviorDistillationError(
            "guided_tranche does not expose authenticated factor-source binding."
        )
    raw = source_accessor()
    expected_keys = {
        "audit_report_sha256",
        "basis_envelope_file_sha256",
        "basis_payload_sha256",
        "basis_manifest_sha256",
        "source_basis_tensor_sha256s",
        "inherited_installation_sha256",
        "inherited_factor_state_sha256_at_install",
        "inherited_handle_identity",
        "inherited_registry_identity",
        "adapter_site_indices_sha256",
        "audit_site_indices_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise TargetNativeBehaviorDistillationError(
            "guided_tranche factor-source binding has an unsupported schema."
        )
    if raw.get("inherited_handle_identity") != id(transaction.qkvo_handle) or raw.get(
        "inherited_registry_identity"
    ) != id(transaction.qkvo_handle.adapter_registry):
        raise TargetNativeBehaviorDistillationError(
            "guided_tranche source binding does not own this inherited QKVO product."
        )
    basis_hashes = GuidedFactorSourceBinding._normalize_hash_rows(
        raw.get("source_basis_tensor_sha256s"),
        "source_basis_tensor_sha256s",
        expected_names={
            "q",
            "k",
            "v",
            "o",
        },
    )
    basis_by_name = dict(basis_hashes)
    entry_by_name = dict(entries)
    for role in ("q", "k", "v", "o"):
        factor = entry_by_name[f"guided_qkvo.{role}_right"]
        if _output_jacobian_tensor_sha256(factor) != basis_by_name[role]:
            raise TargetNativeBehaviorDistillationError(
                f"Installed guided {role.upper()} right factor does not byte-match "
                "its authenticated audit basis."
            )
        left = entry_by_name[f"guided_qkvo.{role}_left"]
        if int(torch.count_nonzero(left.detach()).item()) != 0:
            raise TargetNativeBehaviorDistillationError(
                "Guided left factors must remain exact zero until source "
                "authentication completes."
            )
    source_v2_hashes = tuple(
        zip(
            source_v2_state.names,
            source_v2_state.tensor_sha256s,
            strict=True,
        )
    )
    return GuidedFactorSourceBinding(
        source_v2_factor_state_sha256=source_v2_state.factor_state_sha256,
        source_v2_tensor_sha256s=source_v2_hashes,
        audit_report_sha256=raw.get("audit_report_sha256"),
        basis_envelope_file_sha256=raw.get("basis_envelope_file_sha256"),
        basis_payload_sha256=raw.get("basis_payload_sha256"),
        basis_manifest_sha256=raw.get("basis_manifest_sha256"),
        source_basis_tensor_sha256s=basis_hashes,
        inherited_installation_sha256=raw.get("inherited_installation_sha256"),
        inherited_factor_state_sha256_at_install=raw.get(
            "inherited_factor_state_sha256_at_install"
        ),
        adapter_site_indices_sha256=raw.get("adapter_site_indices_sha256"),
        audit_site_indices_sha256=raw.get("audit_site_indices_sha256"),
    )


def authenticate_guided_gud_factor_sources(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    guided_gud_tranche: Any,
    source_v3_state: TargetNativeFactorState,
) -> GuidedGUDFactorSourceBinding:
    """Bind a complete recovered v3 state and installed GUD right bases."""

    expected_v3_names = (
        _QKVO_GUD_FACTOR_NAMES | _LEXICAL_FACTOR_NAMES | _GUIDED_QKVO_FACTOR_NAMES
    )
    source_binding = getattr(source_v3_state, "guided_source_binding", None)
    if (
        not isinstance(source_v3_state, TargetNativeFactorState)
        or source_v3_state.schema != BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        or len(source_v3_state.names) != GUIDED_COMBINED_FACTOR_TENSOR_COUNT
        or len(source_v3_state.tensors) != GUIDED_COMBINED_FACTOR_TENSOR_COUNT
        or len(source_v3_state.tensor_sha256s) != GUIDED_COMBINED_FACTOR_TENSOR_COUNT
        or set(source_v3_state.names) != expected_v3_names
        or not isinstance(source_binding, GuidedFactorSourceBinding)
    ):
        raise TargetNativeBehaviorDistillationError(
            "A guided-GUD continuation requires one authenticated complete v3 "
            "source factor state."
        )
    source_rows = _factor_manifest_rows(
        source_v3_state.names,
        source_v3_state.tensors,
        source_v3_state.tensor_sha256s,
    )
    if (
        any(
            _tensor_sha256(tensor) != digest
            for tensor, digest in zip(
                source_v3_state.tensors,
                source_v3_state.tensor_sha256s,
                strict=True,
            )
        )
        or _factor_state_sha256(
            source_rows,
            schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=source_v3_state.lexical_rank,
            guided_source_binding=source_binding,
        )
        != source_v3_state.factor_state_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "The recovered v3 source state failed byte authentication."
        )
    entries = _factor_parameters(
        transaction,
        guided_tranche=guided_tranche,
        guided_gud_tranche=guided_gud_tranche,
    )
    entry_by_name = dict(entries)
    source_hash_by_name = dict(
        zip(
            source_v3_state.names,
            source_v3_state.tensor_sha256s,
            strict=True,
        )
    )
    if any(
        _tensor_sha256(entry_by_name[name]) != digest
        for name, digest in source_hash_by_name.items()
    ):
        raise TargetNativeBehaviorDistillationError(
            "The live recovered v3 factors differ from the guided-GUD source snapshot."
        )
    source_accessor = getattr(guided_gud_tranche, "factor_source_binding", None)
    if not callable(source_accessor):
        source_accessor = getattr(
            guided_gud_tranche, "guided_factor_source_binding", None
        )
    if not callable(source_accessor):
        raise TargetNativeBehaviorDistillationError(
            "guided_gud_tranche does not expose authenticated factor-source binding."
        )
    raw = source_accessor()
    expected_keys = {
        "audit_report_sha256",
        "basis_envelope_file_sha256",
        "basis_payload_sha256",
        "basis_manifest_sha256",
        "source_basis_tensor_sha256s",
        "inherited_installation_sha256",
        "inherited_factor_state_sha256_at_install",
        "inherited_handle_identity",
        "inherited_registry_identity",
        "adapter_site_indices_sha256",
        "audit_site_indices_sha256",
    }
    production_expected_keys = expected_keys - {
        "adapter_site_indices_sha256",
        "audit_site_indices_sha256",
    } | {"adapter_layer_indices_sha256"}
    if not isinstance(raw, Mapping) or set(raw) not in {
        frozenset(expected_keys),
        frozenset(production_expected_keys),
    }:
        raise TargetNativeBehaviorDistillationError(
            "guided_gud_tranche factor-source binding has an unsupported schema."
        )
    inherited_registry = getattr(transaction.swiglu_handle, "registry", None)
    if raw.get("inherited_handle_identity") != id(transaction.swiglu_handle) or raw.get(
        "inherited_registry_identity"
    ) != id(inherited_registry):
        raise TargetNativeBehaviorDistillationError(
            "guided_gud_tranche source binding does not own this inherited GUD product."
        )
    basis_hashes = GuidedFactorSourceBinding._normalize_hash_rows(
        raw.get("source_basis_tensor_sha256s"),
        "guided_gud source_basis_tensor_sha256s",
        expected_names={"gate", "up", "down"},
    )
    basis_by_name = dict(basis_hashes)
    for role in ("gate", "up", "down"):
        right = entry_by_name[f"guided_gud.{role}_right"]
        if _guided_gud_basis_tensor_sha256(right) != basis_by_name[role]:
            raise TargetNativeBehaviorDistillationError(
                f"Installed guided-GUD {role} right factor does not byte-match "
                "its authenticated audit basis."
            )
        left = entry_by_name[f"guided_gud.{role}_left"]
        if int(torch.count_nonzero(left.detach()).item()) != 0:
            raise TargetNativeBehaviorDistillationError(
                "Guided-GUD left factors must remain exact zero until source "
                "authentication completes."
            )
    adapter_indices_sha256 = raw.get(
        "adapter_site_indices_sha256",
        raw.get("adapter_layer_indices_sha256"),
    )
    audit_indices_sha256 = raw.get(
        "audit_site_indices_sha256",
        adapter_indices_sha256,
    )
    return GuidedGUDFactorSourceBinding(
        source_v3_factor_state_sha256=source_v3_state.factor_state_sha256,
        source_v3_factor_manifest_sha256=_canonical_sha256(source_rows),
        source_v3_tensor_sha256s=tuple(
            zip(
                source_v3_state.names,
                source_v3_state.tensor_sha256s,
                strict=True,
            )
        ),
        source_v3_lexical_rank=_index(
            source_v3_state.lexical_rank,
            "source_v3_lexical_rank",
            positive=True,
        ),
        source_guided_qkvo_binding=source_binding,
        audit_report_sha256=raw.get("audit_report_sha256"),
        basis_envelope_file_sha256=raw.get("basis_envelope_file_sha256"),
        basis_payload_sha256=raw.get("basis_payload_sha256"),
        basis_manifest_sha256=raw.get("basis_manifest_sha256"),
        source_basis_tensor_sha256s=basis_hashes,
        inherited_installation_sha256=raw.get("inherited_installation_sha256"),
        inherited_factor_state_sha256_at_install=raw.get(
            "inherited_factor_state_sha256_at_install"
        ),
        adapter_site_indices_sha256=adapter_indices_sha256,
        audit_site_indices_sha256=audit_indices_sha256,
    )


def capture_target_native_factor_state(
    transaction: ExactTargetCompilerTransaction,
    *,
    guided_tranche: Any | None = None,
    guided_source_binding: GuidedFactorSourceBinding | None = None,
    guided_gud_tranche: Any | None = None,
    guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None,
) -> TargetNativeFactorState:
    entries = _factor_parameters(
        transaction,
        guided_tranche=guided_tranche,
        guided_gud_tranche=guided_gud_tranche,
    )
    tensors = tuple(
        value.detach().cpu().contiguous().clone() for _name, value in entries
    )
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise TargetNativeBehaviorDistillationError(
            "Cannot capture non-finite target-native factors."
        )
    names = tuple(name for name, _value in entries)
    hashes = tuple(_tensor_sha256(value) for value in tensors)
    schema, lexical_rank = _factor_pack_contract(
        transaction, guided_tranche, guided_gud_tranche
    )
    if schema in {
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    } and not isinstance(guided_source_binding, GuidedFactorSourceBinding):
        raise TargetNativeBehaviorDistillationError(
            "Capturing a v3/v4 state requires authenticated guided source binding."
        )
    if (
        schema
        not in {
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }
        and guided_source_binding is not None
    ):
        raise TargetNativeBehaviorDistillationError(
            "Guided source binding is only valid for a v3/v4 factor state."
        )
    if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA and not isinstance(
        guided_gud_source_binding, GuidedGUDFactorSourceBinding
    ):
        raise TargetNativeBehaviorDistillationError(
            "Capturing a v4 state requires authenticated guided-GUD source binding."
        )
    if schema != BEHAVIOR_FACTOR_PACK_V4_SCHEMA and (
        guided_gud_source_binding is not None
    ):
        raise TargetNativeBehaviorDistillationError(
            "Guided-GUD source binding is only valid for a v4 factor state."
        )
    payload = _factor_manifest_rows(names, tensors, hashes)
    return TargetNativeFactorState(
        names=names,
        tensors=tensors,
        tensor_sha256s=hashes,
        factor_state_sha256=_factor_state_sha256(
            payload,
            schema=schema,
            lexical_rank=lexical_rank,
            guided_source_binding=guided_source_binding,
            guided_gud_source_binding=guided_gud_source_binding,
        ),
        parameter_ids=tuple(id(value) for _name, value in entries),
        parameter_data_ptrs=tuple(int(value.data_ptr()) for _name, value in entries),
        schema=schema,
        lexical_rank=lexical_rank,
        guided_source_binding=guided_source_binding,
        guided_gud_source_binding=guided_gud_source_binding,
    )


def restore_target_native_factor_state(
    transaction: ExactTargetCompilerTransaction,
    state: TargetNativeFactorState,
    *,
    guided_tranche: Any | None = None,
    guided_gud_tranche: Any | None = None,
) -> None:
    if not isinstance(state, TargetNativeFactorState):
        raise TargetNativeBehaviorDistillationError(
            "state must be TargetNativeFactorState."
        )
    entries = _factor_parameters(
        transaction,
        require_finite=False,
        guided_tranche=guided_tranche,
        guided_gud_tranche=guided_gud_tranche,
    )
    if tuple(name for name, _value in entries) != state.names:
        raise TargetNativeBehaviorDistillationError(
            "Factor-state names do not match the live transaction."
        )
    expected_schema, expected_lexical_rank = _factor_pack_contract(
        transaction, guided_tranche, guided_gud_tranche
    )
    if state.schema != expected_schema or state.lexical_rank != expected_lexical_rank:
        raise TargetNativeBehaviorDistillationError(
            "Factor-state schema or lexical rank does not match the live transaction."
        )
    if (
        tuple(id(value) for _name, value in entries) != state.parameter_ids
        or tuple(int(value.data_ptr()) for _name, value in entries)
        != state.parameter_data_ptrs
    ):
        raise TargetNativeBehaviorDistillationError(
            "Factor-state parameter ownership or storage changed before restore."
        )
    for (name, parameter), value, digest in zip(
        entries, state.tensors, state.tensor_sha256s, strict=True
    ):
        if (
            tuple(parameter.shape) != tuple(value.shape)
            or parameter.dtype != value.dtype
            or _tensor_sha256(value) != digest
        ):
            raise TargetNativeBehaviorDistillationError(
                f"Factor state {name!r} has invalid shape, dtype, or digest."
            )
    with torch.no_grad():
        for (_name, parameter), value in zip(entries, state.tensors, strict=True):
            parameter.copy_(value.to(device=parameter.device))
    transaction.assert_live()
    if guided_tranche is not None:
        guided_tranche.assert_live()
    if guided_gud_tranche is not None:
        guided_gud_tranche.assert_live()
    if (
        capture_target_native_factor_state(
            transaction,
            guided_tranche=guided_tranche,
            guided_source_binding=state.guided_source_binding,
            guided_gud_tranche=guided_gud_tranche,
            guided_gud_source_binding=state.guided_gud_source_binding,
        ).factor_state_sha256
        != state.factor_state_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Factor-state restore did not verify."
        )


@dataclass(frozen=True, slots=True)
class _FactorPayloadRestore:
    source_factor_state_sha256: str
    source_schema: str
    combined_state: TargetNativeFactorState
    restore_mode: Literal[
        "full",
        "qkvo-gud-partial",
        "guided-v3-full",
        "guided-v3-to-v4",
        "guided-v4-full",
    ]
    lexical_initialization_state_sha256: str | None
    lexical_initialization_unchanged: bool | None


def _lexical_initialization_sha256(
    entries: Sequence[tuple[str, nn.Parameter]],
    *,
    lexical_rank: int,
) -> str:
    if {name for name, _value in entries} != _LEXICAL_FACTOR_NAMES:
        raise TargetNativeBehaviorDistillationError(
            "The live lexical factor set is incomplete."
        )
    for name, value in entries:
        count = int(torch.count_nonzero(value.detach()).item())
        if (name.endswith("_left") and count != 0) or (
            name.endswith("_right") and count == 0
        ):
            raise TargetNativeBehaviorDistillationError(
                "A v1 continuation requires untouched exact-zero lexical "
                "initialization."
            )
    tensors = tuple(value.detach().cpu().contiguous() for _name, value in entries)
    hashes = tuple(_tensor_sha256(value) for value in tensors)
    return _canonical_sha256(
        {
            "schema": "neural-forge-lexical-factor-initialization-v1",
            "lexical_rank": _index(lexical_rank, "lexical_rank", positive=True),
            "factors": _factor_manifest_rows(
                tuple(name for name, _value in entries), tensors, hashes
            ),
        }
    )


def _authenticate_factor_payload_state(
    payload: Mapping[str, Any],
    entries: Sequence[tuple[str, nn.Parameter]],
    *,
    schema: str,
    lexical_rank: int | None,
    expected_guided_source_binding: GuidedFactorSourceBinding | None = None,
    expected_guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None,
) -> TargetNativeFactorState:
    expected_payload_keys = {
        "schema",
        "factor_count",
        "factors",
        "factor_state_sha256",
        "parent_state_included",
        "native_child_state_included",
        "optimizer_state_included",
        "factor_tensors",
    }
    if schema in {
        BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    }:
        expected_payload_keys.add("lexical_rank")
    guided_source_binding = None
    if schema in {
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    }:
        expected_payload_keys.update(
            {
                "guided_source_binding",
                "guided_source_binding_sha256",
                "inherited_product_regrouped",
                "guided_product_regrouped",
            }
        )
    guided_gud_source_binding = None
    if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA:
        expected_payload_keys.update(
            {
                "guided_gud_source_binding",
                "guided_gud_source_binding_sha256",
                "guided_gud_product_regrouped",
            }
        )
    if set(payload) != expected_payload_keys:
        raise TargetNativeBehaviorDistillationError(
            "Persisted factor payload does not contain exactly the schema fields."
        )
    if any(
        payload.get(name) is not False
        for name in (
            "parent_state_included",
            "native_child_state_included",
            "optimizer_state_included",
        )
    ):
        raise TargetNativeBehaviorDistillationError(
            "Continuation payload must contain factors only."
        )
    if schema in {
        BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    } and payload.get("lexical_rank") != _index(
        lexical_rank, "lexical_rank", positive=True
    ):
        raise TargetNativeBehaviorDistillationError(
            "Persisted lexical rank does not match the live transaction."
        )
    if schema in {
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    }:
        raw_binding = payload.get("guided_source_binding")
        expected_binding_keys = {
            "source_v2_factor_state_sha256",
            "source_v2_tensor_sha256s",
            "audit_report_sha256",
            "basis_envelope_file_sha256",
            "basis_payload_sha256",
            "basis_manifest_sha256",
            "source_basis_tensor_sha256s",
            "inherited_installation_sha256",
            "inherited_factor_state_sha256_at_install",
            "adapter_site_indices_sha256",
            "audit_site_indices_sha256",
        }
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != (
            expected_binding_keys
        ):
            raise TargetNativeBehaviorDistillationError(
                "Persisted v3/v4 guided source binding has an unsupported schema."
            )
        guided_source_binding = GuidedFactorSourceBinding(
            **dict(raw_binding),
        )
        if payload.get("guided_source_binding_sha256") != (
            guided_source_binding.binding_sha256
        ):
            raise TargetNativeBehaviorDistillationError(
                "Persisted v3/v4 guided source binding hash does not authenticate."
            )
        if (
            payload.get("inherited_product_regrouped") is not False
            or payload.get("guided_product_regrouped") is not False
        ):
            raise TargetNativeBehaviorDistillationError(
                "A v3/v4 pack may not regroup inherited and guided products."
            )
        if (
            expected_guided_source_binding is not None
            and guided_source_binding != expected_guided_source_binding
        ):
            raise TargetNativeBehaviorDistillationError(
                "Persisted v3/v4 guided source binding disagrees with the "
                "preregistered sources."
            )
    elif expected_guided_source_binding is not None:
        raise TargetNativeBehaviorDistillationError(
            "Expected guided source binding is only valid for a v3/v4 payload."
        )
    if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA:
        guided_gud_source_binding = _guided_gud_source_binding_from_payload(
            payload.get("guided_gud_source_binding")
        )
        if payload.get("guided_gud_source_binding_sha256") != (
            guided_gud_source_binding.binding_sha256
        ):
            raise TargetNativeBehaviorDistillationError(
                "Persisted v4 guided-GUD source binding hash does not authenticate."
            )
        if payload.get("guided_gud_product_regrouped") is not False:
            raise TargetNativeBehaviorDistillationError(
                "A v4 pack may not regroup the guided-GUD product."
            )
        if (
            guided_gud_source_binding.source_guided_qkvo_binding
            != guided_source_binding
        ):
            raise TargetNativeBehaviorDistillationError(
                "Persisted v4 nested QKVO binding differs from its top-level "
                "guided binding."
            )
        if (
            expected_guided_gud_source_binding is not None
            and guided_gud_source_binding != expected_guided_gud_source_binding
        ):
            raise TargetNativeBehaviorDistillationError(
                "Persisted v4 guided-GUD source binding disagrees with the "
                "preregistered sources."
            )
    elif expected_guided_gud_source_binding is not None:
        raise TargetNativeBehaviorDistillationError(
            "Expected guided-GUD source binding is only valid for a v4 payload."
        )
    rows = payload.get("factors")
    tensors = payload.get("factor_tensors")
    declared_count = payload.get("factor_count")
    declared_state_sha = _sha256(
        payload.get("factor_state_sha256"), "factor_state_sha256"
    )
    if (
        not isinstance(rows, list)
        or not isinstance(tensors, Mapping)
        or isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(entries)
        or len(rows) != len(entries)
    ):
        raise TargetNativeBehaviorDistillationError(
            "Persisted factor count or manifest is malformed."
        )
    expected_names = tuple(name for name, _value in entries)
    row_by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "shape",
            "dtype",
            "tensor_sha256",
        }:
            raise TargetNativeBehaviorDistillationError(
                "Persisted factor manifest rows must contain exactly the v1 fields."
            )
        name = row.get("name")
        if not isinstance(name, str) or name in row_by_name:
            raise TargetNativeBehaviorDistillationError(
                "Persisted factor names must be unique strings."
            )
        row_by_name[name] = row
    if set(row_by_name) != set(expected_names) or set(tensors) != set(expected_names):
        raise TargetNativeBehaviorDistillationError(
            "Persisted factor names do not match the selected live factor set."
        )
    ordered_tensors: list[Tensor] = []
    ordered_hashes: list[str] = []
    canonical_rows: list[dict[str, Any]] = []
    for name, parameter in entries:
        row = row_by_name[name]
        value = tensors[name]
        if not isinstance(value, Tensor):
            raise TargetNativeBehaviorDistillationError(
                f"Persisted factor {name!r} is not a tensor."
            )
        detached = value.detach().cpu().contiguous().clone()
        digest = _tensor_sha256(detached)
        expected_row = {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "tensor_sha256": digest,
        }
        if (
            dict(row) != expected_row
            or tuple(detached.shape) != tuple(parameter.shape)
            or detached.dtype != parameter.dtype
        ):
            raise TargetNativeBehaviorDistillationError(
                f"Persisted factor {name!r} failed shape/dtype/hash authentication."
            )
        if not bool(torch.isfinite(detached).all()):
            raise TargetNativeBehaviorDistillationError(
                f"Persisted factor {name!r} contains NaN or infinity."
            )
        ordered_tensors.append(detached)
        ordered_hashes.append(digest)
        canonical_rows.append(expected_row)
    if (
        _factor_state_sha256(
            canonical_rows,
            schema=schema,
            lexical_rank=lexical_rank,
            guided_source_binding=guided_source_binding,
            guided_gud_source_binding=guided_gud_source_binding,
        )
        != declared_state_sha
    ):
        raise TargetNativeBehaviorDistillationError(
            "Persisted factor-state hash does not authenticate its manifest."
        )
    return TargetNativeFactorState(
        names=expected_names,
        tensors=tuple(ordered_tensors),
        tensor_sha256s=tuple(ordered_hashes),
        factor_state_sha256=declared_state_sha,
        parameter_ids=tuple(id(value) for _name, value in entries),
        parameter_data_ptrs=tuple(int(value.data_ptr()) for _name, value in entries),
        schema=schema,
        lexical_rank=lexical_rank,
        guided_source_binding=guided_source_binding,
        guided_gud_source_binding=guided_gud_source_binding,
    )


def _copy_authenticated_factor_state(
    transaction: ExactTargetCompilerTransaction,
    entries: Sequence[tuple[str, nn.Parameter]],
    state: TargetNativeFactorState,
) -> None:
    if (
        tuple(name for name, _value in entries) != state.names
        or tuple(id(value) for _name, value in entries) != state.parameter_ids
        or tuple(int(value.data_ptr()) for _name, value in entries)
        != state.parameter_data_ptrs
    ):
        raise TargetNativeBehaviorDistillationError(
            "Persisted factor ownership changed before restore."
        )
    with torch.no_grad():
        for (_name, parameter), value in zip(entries, state.tensors, strict=True):
            parameter.copy_(value.to(device=parameter.device))
    transaction.assert_live()


def _restore_target_native_factor_payload(
    transaction: ExactTargetCompilerTransaction,
    payload: Mapping[str, Any],
    *,
    guided_tranche: Any | None = None,
    expected_guided_source_binding: GuidedFactorSourceBinding | None = None,
    guided_gud_tranche: Any | None = None,
    expected_guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None,
) -> _FactorPayloadRestore:
    if not isinstance(payload, Mapping):
        raise TargetNativeBehaviorDistillationError(
            "Persisted target-native factor payload must be a mapping."
        )
    schema = payload.get("schema")
    if schema not in {
        BEHAVIOR_FACTOR_PACK_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    }:
        raise TargetNativeBehaviorDistillationError(
            "Persisted target-native factor payload has an unsupported schema."
        )
    if schema == BEHAVIOR_FACTOR_PACK_V3_SCHEMA and guided_tranche is None:
        raise TargetNativeBehaviorDistillationError(
            "Restoring a v3 factor pack requires its live guided tranche."
        )
    if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA and (
        guided_tranche is None or guided_gud_tranche is None
    ):
        raise TargetNativeBehaviorDistillationError(
            "Restoring a v4 factor pack requires both live guided tranches."
        )
    if schema == BEHAVIOR_FACTOR_PACK_V3_SCHEMA and not isinstance(
        expected_guided_source_binding, GuidedFactorSourceBinding
    ):
        raise TargetNativeBehaviorDistillationError(
            "Restoring a v3 factor pack requires its preregistered guided source "
            "binding."
        )
    if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA and (
        not isinstance(expected_guided_source_binding, GuidedFactorSourceBinding)
        or not isinstance(
            expected_guided_gud_source_binding, GuidedGUDFactorSourceBinding
        )
    ):
        raise TargetNativeBehaviorDistillationError(
            "Restoring a v4 factor pack requires both preregistered guided source "
            "bindings."
        )
    if schema not in {
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    } and (expected_guided_source_binding is not None):
        raise TargetNativeBehaviorDistillationError(
            "Expected guided source binding is only valid for a v3/v4 payload."
        )
    if schema != BEHAVIOR_FACTOR_PACK_V4_SCHEMA and (
        expected_guided_gud_source_binding is not None
    ):
        raise TargetNativeBehaviorDistillationError(
            "Expected guided-GUD source binding is only valid for a v4 payload."
        )
    entries = _factor_parameters(
        transaction,
        guided_tranche=(
            guided_tranche
            if schema
            in {BEHAVIOR_FACTOR_PACK_V3_SCHEMA, BEHAVIOR_FACTOR_PACK_V4_SCHEMA}
            else None
        ),
        guided_gud_tranche=(
            guided_gud_tranche if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA else None
        ),
    )
    live_names = {name for name, _value in entries}
    live_lexical_rank = _lexical_rank(transaction)
    if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA:
        expected_names = (
            _QKVO_GUD_FACTOR_NAMES
            | _LEXICAL_FACTOR_NAMES
            | _GUIDED_QKVO_FACTOR_NAMES
            | _GUIDED_GUD_FACTOR_NAMES
        )
        if live_names != expected_names or len(entries) != (
            GUIDED_GUD_COMBINED_FACTOR_TENSOR_COUNT
        ):
            raise TargetNativeBehaviorDistillationError(
                "A v4 factor pack requires the complete separate 32-factor "
                "guided-GUD transaction."
            )
        state = _authenticate_factor_payload_state(
            payload,
            entries,
            schema=schema,
            lexical_rank=live_lexical_rank,
            expected_guided_source_binding=expected_guided_source_binding,
            expected_guided_gud_source_binding=(expected_guided_gud_source_binding),
        )
        for tranche, name in (
            (guided_tranche, "guided_tranche"),
            (guided_gud_tranche, "guided_gud_tranche"),
        ):
            set_stage = getattr(tranche, "set_training_stage", None)
            if not callable(set_stage):
                raise TargetNativeBehaviorDistillationError(
                    f"{name} does not expose its authenticated joint-stage transition."
                )
            set_stage("joint")
        restore_target_native_factor_state(
            transaction,
            state,
            guided_tranche=guided_tranche,
            guided_gud_tranche=guided_gud_tranche,
        )
        combined = capture_target_native_factor_state(
            transaction,
            guided_tranche=guided_tranche,
            guided_source_binding=state.guided_source_binding,
            guided_gud_tranche=guided_gud_tranche,
            guided_gud_source_binding=state.guided_gud_source_binding,
        )
        if combined.factor_state_sha256 != state.factor_state_sha256:
            raise TargetNativeBehaviorDistillationError(
                "Persisted v4 factor payload restore did not verify."
            )
        return _FactorPayloadRestore(
            source_factor_state_sha256=state.factor_state_sha256,
            source_schema=schema,
            combined_state=combined,
            restore_mode="guided-v4-full",
            lexical_initialization_state_sha256=None,
            lexical_initialization_unchanged=None,
        )
    if schema == BEHAVIOR_FACTOR_PACK_V3_SCHEMA:
        expected_names = (
            _QKVO_GUD_FACTOR_NAMES | _LEXICAL_FACTOR_NAMES | _GUIDED_QKVO_FACTOR_NAMES
        )
        if live_names != expected_names or len(entries) != (
            GUIDED_COMBINED_FACTOR_TENSOR_COUNT
        ):
            raise TargetNativeBehaviorDistillationError(
                "A v3 factor pack requires the complete separate 26-factor "
                "guided transaction."
            )
        state = _authenticate_factor_payload_state(
            payload,
            entries,
            schema=schema,
            lexical_rank=live_lexical_rank,
            expected_guided_source_binding=expected_guided_source_binding,
        )
        restore_target_native_factor_state(
            transaction,
            state,
            guided_tranche=guided_tranche,
        )
        combined = capture_target_native_factor_state(
            transaction,
            guided_tranche=guided_tranche,
            guided_source_binding=state.guided_source_binding,
        )
        if combined.factor_state_sha256 != state.factor_state_sha256:
            raise TargetNativeBehaviorDistillationError(
                "Persisted v3 factor payload restore did not verify."
            )
        return _FactorPayloadRestore(
            source_factor_state_sha256=state.factor_state_sha256,
            source_schema=schema,
            combined_state=combined,
            restore_mode="guided-v3-full",
            lexical_initialization_state_sha256=None,
            lexical_initialization_unchanged=None,
        )
    if schema == BEHAVIOR_FACTOR_PACK_V2_SCHEMA:
        if live_names != _QKVO_GUD_FACTOR_NAMES | _LEXICAL_FACTOR_NAMES:
            raise TargetNativeBehaviorDistillationError(
                "A v2 factor pack requires the complete 18-factor lexical transaction."
            )
        state = _authenticate_factor_payload_state(
            payload,
            entries,
            schema=schema,
            lexical_rank=live_lexical_rank,
        )
        restore_target_native_factor_state(transaction, state)
        combined = capture_target_native_factor_state(transaction)
        if combined.factor_state_sha256 != state.factor_state_sha256:
            raise TargetNativeBehaviorDistillationError(
                "Persisted v2 factor payload restore did not verify."
            )
        return _FactorPayloadRestore(
            source_factor_state_sha256=state.factor_state_sha256,
            source_schema=schema,
            combined_state=combined,
            restore_mode="full",
            lexical_initialization_state_sha256=None,
            lexical_initialization_unchanged=None,
        )

    base_entries = tuple(
        (name, value) for name, value in entries if name in _QKVO_GUD_FACTOR_NAMES
    )
    if {name for name, _value in base_entries} != _QKVO_GUD_FACTOR_NAMES:
        raise TargetNativeBehaviorDistillationError(
            "A v1 continuation requires the complete 14-factor QKVO/GUD set."
        )
    state = _authenticate_factor_payload_state(
        payload,
        base_entries,
        schema=BEHAVIOR_FACTOR_PACK_SCHEMA,
        lexical_rank=None,
    )
    if live_lexical_rank is None:
        restore_target_native_factor_state(transaction, state)
        combined = capture_target_native_factor_state(transaction)
        return _FactorPayloadRestore(
            source_factor_state_sha256=state.factor_state_sha256,
            source_schema=schema,
            combined_state=combined,
            restore_mode="full",
            lexical_initialization_state_sha256=None,
            lexical_initialization_unchanged=None,
        )

    lexical_entries = tuple(
        (name, value) for name, value in entries if name in _LEXICAL_FACTOR_NAMES
    )
    lexical_before = _lexical_initialization_sha256(
        lexical_entries, lexical_rank=live_lexical_rank
    )
    _copy_authenticated_factor_state(transaction, base_entries, state)
    lexical_after = _lexical_initialization_sha256(
        lexical_entries, lexical_rank=live_lexical_rank
    )
    if lexical_after != lexical_before:
        raise TargetNativeBehaviorDistillationError(
            "Partial v1 restore changed lexical initialization."
        )
    combined = capture_target_native_factor_state(transaction)
    return _FactorPayloadRestore(
        source_factor_state_sha256=state.factor_state_sha256,
        source_schema=schema,
        combined_state=combined,
        restore_mode="qkvo-gud-partial",
        lexical_initialization_state_sha256=lexical_before,
        lexical_initialization_unchanged=True,
    )


def restore_target_native_factor_payload(
    transaction: ExactTargetCompilerTransaction,
    payload: Mapping[str, Any],
    *,
    guided_tranche: Any | None = None,
    expected_guided_source_binding: GuidedFactorSourceBinding | None = None,
    guided_gud_tranche: Any | None = None,
    expected_guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None,
) -> TargetNativeFactorState:
    """Authenticate and restore a v1-v4 separate-product continuation."""

    return _restore_target_native_factor_payload(
        transaction,
        payload,
        guided_tranche=guided_tranche,
        expected_guided_source_binding=expected_guided_source_binding,
        guided_gud_tranche=guided_gud_tranche,
        expected_guided_gud_source_binding=expected_guided_gud_source_binding,
    ).combined_state


def _authenticate_retained_guided_source_snapshots(
    *,
    source_v2_state: TargetNativeFactorState,
    basis_snapshots: Sequence[tuple[str, Tensor, str]],
    expected_binding: GuidedFactorSourceBinding,
    guided_tranche: Any,
    source_accessor_sha256: str,
) -> None:
    """Reauthenticate pre-restore v2/basis clones after a full v3 restore."""

    rows = _factor_manifest_rows(
        source_v2_state.names,
        source_v2_state.tensors,
        source_v2_state.tensor_sha256s,
    )
    if (
        source_v2_state.schema != BEHAVIOR_FACTOR_PACK_V2_SCHEMA
        or any(
            _tensor_sha256(tensor) != digest
            for tensor, digest in zip(
                source_v2_state.tensors,
                source_v2_state.tensor_sha256s,
                strict=True,
            )
        )
        or _factor_state_sha256(
            rows,
            schema=BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            lexical_rank=source_v2_state.lexical_rank,
        )
        != expected_binding.source_v2_factor_state_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Retained inherited v2 source bytes changed during v3 restoration."
        )
    expected_basis_hashes = dict(expected_binding.source_basis_tensor_sha256s)
    if {role for role, _tensor, _digest in basis_snapshots} != set(
        expected_basis_hashes
    ) or any(
        digest != expected_basis_hashes[role]
        or _output_jacobian_tensor_sha256(tensor) != digest
        for role, tensor, digest in basis_snapshots
    ):
        raise TargetNativeBehaviorDistillationError(
            "Retained guided audit-basis bytes changed during v3 restoration."
        )
    source_accessor = getattr(guided_tranche, "factor_source_binding", None)
    if not callable(source_accessor) or _canonical_sha256(source_accessor()) != _sha256(
        source_accessor_sha256, "source_accessor_sha256"
    ):
        raise TargetNativeBehaviorDistillationError(
            "Guided source accessor changed during v3 restoration."
        )


def _restore_guided_v3_initial_factor_payload(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    payload: Mapping[str, Any],
) -> tuple[_FactorPayloadRestore, GuidedFactorSourceBinding]:
    """Authenticate zero-installed sources, then restore one complete v3 pack."""

    source_v2_state = capture_target_native_factor_state(transaction)
    expected_binding = authenticate_guided_factor_sources(
        transaction,
        guided_tranche,
        source_v2_state,
    )
    zero_state = capture_target_native_factor_state(
        transaction,
        guided_tranche=guided_tranche,
        guided_source_binding=expected_binding,
    )
    entries = dict(_factor_parameters(transaction, guided_tranche=guided_tranche))
    expected_basis_hashes = dict(expected_binding.source_basis_tensor_sha256s)
    basis_snapshots = tuple(
        (
            role,
            entries[f"guided_qkvo.{role}_right"].detach().cpu().contiguous().clone(),
            expected_basis_hashes[role],
        )
        for role in ("q", "k", "v", "o")
    )
    source_accessor = getattr(guided_tranche, "factor_source_binding")
    source_accessor_sha256 = _canonical_sha256(source_accessor())
    try:
        # A recovered full pack may contain trained inherited and right-guided
        # factors.  The installed tranche only permits those bytes once its
        # monotonic runtime state has reached ``joint``.  Authenticate the
        # payload while the zero installation is still guarded, then advance
        # the runtime immediately before the existing full restore path.
        origin_state = _authenticate_factor_payload_state(
            payload,
            tuple(entries.items()),
            schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=_lexical_rank(transaction),
        )
        origin_binding = origin_state.guided_source_binding
        if not isinstance(origin_binding, GuidedFactorSourceBinding):
            raise TargetNativeBehaviorDistillationError(
                "Persisted v3 pack has no authenticated origin binding."
            )
        origin_portable = origin_binding.as_dict()
        runtime_portable = expected_binding.as_dict()
        origin_portable.pop("inherited_installation_sha256")
        runtime_portable.pop("inherited_installation_sha256")
        if origin_portable != runtime_portable:
            raise TargetNativeBehaviorDistillationError(
                "Persisted v3 guided source binding disagrees with the "
                "preregistered sources outside its runtime installation receipt."
            )
        runtime_payload = dict(payload)
        runtime_payload["guided_source_binding"] = expected_binding.as_dict()
        runtime_payload["guided_source_binding_sha256"] = (
            expected_binding.binding_sha256
        )
        runtime_payload["factor_state_sha256"] = _factor_state_sha256(
            payload["factors"],
            schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=_lexical_rank(transaction),
            guided_source_binding=expected_binding,
        )
        _authenticate_factor_payload_state(
            runtime_payload,
            tuple(entries.items()),
            schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=_lexical_rank(transaction),
            expected_guided_source_binding=expected_binding,
        )
        set_training_stage = getattr(guided_tranche, "set_training_stage", None)
        if not callable(set_training_stage):
            raise TargetNativeBehaviorDistillationError(
                "guided_tranche does not expose its authenticated training-stage "
                "transition."
            )
        set_training_stage("joint")
        restored = _restore_target_native_factor_payload(
            transaction,
            runtime_payload,
            guided_tranche=guided_tranche,
            expected_guided_source_binding=expected_binding,
        )
        if (
            restored.source_schema != BEHAVIOR_FACTOR_PACK_V3_SCHEMA
            or restored.restore_mode != "guided-v3-full"
            or restored.combined_state.factor_state_sha256
            != _sha256(
                runtime_payload.get("factor_state_sha256"),
                "runtime factor_state_sha256",
            )
            or restored.combined_state.guided_source_binding != expected_binding
        ):
            raise TargetNativeBehaviorDistillationError(
                "Guided v3 initial factor restore did not recover its exact step zero."
            )
        _authenticate_retained_guided_source_snapshots(
            source_v2_state=source_v2_state,
            basis_snapshots=basis_snapshots,
            expected_binding=expected_binding,
            guided_tranche=guided_tranche,
            source_accessor_sha256=source_accessor_sha256,
        )
        return restored, expected_binding
    except BaseException as error:
        try:
            restore_target_native_factor_state(
                transaction,
                zero_state,
                guided_tranche=guided_tranche,
            )
        except Exception as restore_error:
            error.add_note(
                "Exact zero-installed guided restoration also failed: "
                f"{restore_error!r}"
            )
            raise error from restore_error
        raise


def _restore_guided_v3_to_v4_initial_factor_payload(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    guided_gud_tranche: Any,
    payload: Mapping[str, Any],
) -> tuple[
    _FactorPayloadRestore,
    GuidedFactorSourceBinding,
    GuidedGUDFactorSourceBinding,
]:
    """Restore the retained v3 source, authenticate GUD zero install, form v4."""

    if getattr(guided_tranche, "training_stage", None) == "joint":
        restored, guided_source_binding = (
            _authenticate_prerestored_guided_v3_initial_factor_payload(
                transaction,
                guided_tranche,
                payload,
            )
        )
        guided_gud_source_binding = authenticate_guided_gud_factor_sources(
            transaction,
            guided_tranche,
            guided_gud_tranche,
            restored.combined_state,
        )
        combined = capture_target_native_factor_state(
            transaction,
            guided_tranche=guided_tranche,
            guided_source_binding=guided_source_binding,
            guided_gud_tranche=guided_gud_tranche,
            guided_gud_source_binding=guided_gud_source_binding,
        )
        return (
            replace(restored, combined_state=combined),
            guided_source_binding,
            guided_gud_source_binding,
        )
    if getattr(guided_gud_tranche, "report", None) is not None:
        raise TargetNativeBehaviorDistillationError(
            "A production guided-GUD continuation requires the v3 payload to be "
            "authenticated and restored before guided-GUD installation."
        )

    source_v2_state = capture_target_native_factor_state(transaction)
    zero_qkvo_binding = authenticate_guided_factor_sources(
        transaction,
        guided_tranche,
        source_v2_state,
    )
    zero_v3_state = capture_target_native_factor_state(
        transaction,
        guided_tranche=guided_tranche,
        guided_source_binding=zero_qkvo_binding,
    )
    try:
        restored, guided_source_binding = _restore_guided_v3_initial_factor_payload(
            transaction,
            guided_tranche,
            payload,
        )
        source_v3_state = restored.combined_state
        guided_gud_source_binding = authenticate_guided_gud_factor_sources(
            transaction,
            guided_tranche,
            guided_gud_tranche,
            source_v3_state,
        )
        combined = capture_target_native_factor_state(
            transaction,
            guided_tranche=guided_tranche,
            guided_source_binding=guided_source_binding,
            guided_gud_tranche=guided_gud_tranche,
            guided_gud_source_binding=guided_gud_source_binding,
        )
        combined_hashes = dict(
            zip(combined.names, combined.tensor_sha256s, strict=True)
        )
        if any(
            combined_hashes.get(name) != digest
            for name, digest in guided_gud_source_binding.source_v3_tensor_sha256s
        ):
            raise TargetNativeBehaviorDistillationError(
                "Installing the guided-GUD product changed its authenticated v3 "
                "source bytes."
            )
        return (
            _FactorPayloadRestore(
                source_factor_state_sha256=restored.source_factor_state_sha256,
                source_schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
                combined_state=combined,
                restore_mode="guided-v3-to-v4",
                lexical_initialization_state_sha256=None,
                lexical_initialization_unchanged=None,
            ),
            guided_source_binding,
            guided_gud_source_binding,
        )
    except BaseException as error:
        try:
            restore_target_native_factor_state(
                transaction,
                zero_v3_state,
                guided_tranche=guided_tranche,
            )
        except Exception as restore_error:
            error.add_note(
                "Exact zero-installed QKVO restoration after v4 setup also "
                f"failed: {restore_error!r}"
            )
            raise error from restore_error
        raise


def _authenticate_prerestored_guided_v3_initial_factor_payload(
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any,
    payload: Mapping[str, Any],
) -> tuple[_FactorPayloadRestore, GuidedFactorSourceBinding]:
    """Prove a caller-restored v3 handoff without mutating one live factor."""

    if not isinstance(payload, Mapping) or payload.get("schema") != (
        BEHAVIOR_FACTOR_PACK_V3_SCHEMA
    ):
        raise TargetNativeBehaviorDistillationError(
            "A pre-restored guided-GUD handoff requires one complete v3 payload."
        )
    if getattr(guided_tranche, "training_stage", None) != "joint":
        raise TargetNativeBehaviorDistillationError(
            "A pre-restored v3 handoff requires the live QKVO joint stage."
        )
    entries = _factor_parameters(transaction, guided_tranche=guided_tranche)
    authenticated = _authenticate_factor_payload_state(
        payload,
        entries,
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=_lexical_rank(transaction),
    )
    binding = authenticated.guided_source_binding
    if not isinstance(binding, GuidedFactorSourceBinding):
        raise TargetNativeBehaviorDistillationError(
            "The pre-restored v3 payload has no authenticated QKVO source binding."
        )
    source_accessor = getattr(guided_tranche, "factor_source_binding", None)
    if not callable(source_accessor):
        raise TargetNativeBehaviorDistillationError(
            "The live guided QKVO tranche exposes no source binding."
        )
    raw = source_accessor()
    expected_raw = {
        "audit_report_sha256": binding.audit_report_sha256,
        "basis_envelope_file_sha256": binding.basis_envelope_file_sha256,
        "basis_payload_sha256": binding.basis_payload_sha256,
        "basis_manifest_sha256": binding.basis_manifest_sha256,
        "source_basis_tensor_sha256s": dict(binding.source_basis_tensor_sha256s),
        "inherited_installation_sha256": binding.inherited_installation_sha256,
        "inherited_factor_state_sha256_at_install": (
            binding.inherited_factor_state_sha256_at_install
        ),
        "inherited_handle_identity": id(transaction.qkvo_handle),
        "inherited_registry_identity": id(transaction.qkvo_handle.adapter_registry),
        "adapter_site_indices_sha256": binding.adapter_site_indices_sha256,
        "audit_site_indices_sha256": binding.audit_site_indices_sha256,
    }
    if not isinstance(raw, Mapping) or _canonical_sha256(raw) != _canonical_sha256(
        expected_raw
    ):
        raise TargetNativeBehaviorDistillationError(
            "The live guided QKVO source binding differs from the supplied v3 "
            "runtime binding."
        )
    live = capture_target_native_factor_state(
        transaction,
        guided_tranche=guided_tranche,
        guided_source_binding=binding,
    )
    if (
        live.names != authenticated.names
        or live.tensor_sha256s != authenticated.tensor_sha256s
        or live.factor_state_sha256 != authenticated.factor_state_sha256
        or live.parameter_ids != authenticated.parameter_ids
        or live.parameter_data_ptrs != authenticated.parameter_data_ptrs
    ):
        raise TargetNativeBehaviorDistillationError(
            "The live 26-factor v3 state does not exactly equal the supplied "
            "authenticated payload."
        )
    return (
        _FactorPayloadRestore(
            source_factor_state_sha256=authenticated.factor_state_sha256,
            source_schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            combined_state=live,
            restore_mode="guided-v3-to-v4",
            lexical_initialization_state_sha256=None,
            lexical_initialization_unchanged=None,
        ),
        binding,
    )


def _assert_guided_gud_frozen_sources(
    factor_entries: Sequence[tuple[str, nn.Parameter]],
    binding: GuidedGUDFactorSourceBinding,
    *,
    step: int,
) -> None:
    """Prove right bases/source-v3 bytes stay exact until their unlock steps."""

    entry_by_name = dict(factor_entries)
    if step <= GUIDED_LEFT_ONLY_END_STEP:
        for role, digest in binding.source_basis_tensor_sha256s:
            if (
                _guided_gud_basis_tensor_sha256(
                    entry_by_name[f"guided_gud.{role}_right"]
                )
                != digest
            ):
                raise TargetNativeBehaviorDistillationError(
                    "A guided-GUD right basis changed before its step-9 unlock."
                )
    if step <= GUIDED_ONLY_END_STEP:
        for name, digest in binding.source_v3_tensor_sha256s:
            if _tensor_sha256(entry_by_name[name]) != digest:
                raise TargetNativeBehaviorDistillationError(
                    "A recovered v3 source factor changed before its step-25 "
                    "joint unlock."
                )


@dataclass(frozen=True, slots=True)
class GradientActivationAudit:
    factor_names: tuple[str, ...]
    finite_gradient_factor_names: tuple[str, ...]
    nonzero_gradient_factor_names: tuple[str, ...]
    backward_observation_count: int
    authorized_factor_names: tuple[str, ...] = ()
    stage_authorized_factor_names: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def complete(self) -> bool:
        expected = set(self.authorized_factor_names or self.factor_names)
        return expected.issubset(
            self.finite_gradient_factor_names
        ) and expected.issubset(self.nonzero_gradient_factor_names)

    @property
    def qkvo_gud_complete(self) -> bool:
        expected = set(self.factor_names) & _QKVO_GUD_FACTOR_NAMES
        return (
            expected == _QKVO_GUD_FACTOR_NAMES
            and expected.issubset(self.finite_gradient_factor_names)
            and expected.issubset(self.nonzero_gradient_factor_names)
        )

    @property
    def lexical_complete(self) -> bool | None:
        expected = set(self.factor_names) & _LEXICAL_FACTOR_NAMES
        if not expected:
            return None
        return (
            expected == _LEXICAL_FACTOR_NAMES
            and expected.issubset(self.finite_gradient_factor_names)
            and expected.issubset(self.nonzero_gradient_factor_names)
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "factor_names": list(self.factor_names),
            "finite_gradient_factor_names": list(self.finite_gradient_factor_names),
            "nonzero_gradient_factor_names": list(self.nonzero_gradient_factor_names),
            "backward_observation_count": self.backward_observation_count,
            # Retained verbatim for legacy report consumers.
            "every_qkvo_gud_factor_gradient_activated": self.qkvo_gud_complete,
            "every_lexical_factor_gradient_activated": self.lexical_complete,
            "every_combined_factor_gradient_activated": self.complete,
            "combined_factor_count": len(self.factor_names),
        }
        if self.stage_authorized_factor_names:
            result.update(
                {
                    "authorized_factor_names": list(
                        self.authorized_factor_names or self.factor_names
                    ),
                    "intentionally_frozen_factor_names": sorted(
                        set(self.factor_names)
                        - set(self.authorized_factor_names or self.factor_names)
                    ),
                    "stage_authorized_factor_names": {
                        stage: list(names)
                        for stage, names in self.stage_authorized_factor_names
                    },
                    "every_authorized_factor_gradient_activated": self.complete,
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class BehaviorTrainingAudit:
    optimizer_steps: int
    optimizer_class: str
    parent_greedy_prefix_count: int
    child_on_policy_prefix_count: int
    parent_output_forward_count: int
    child_output_forward_count: int
    parent_greedy_cache_prompt_count: int
    parent_greedy_cache_forward_count: int
    gradient_activation: GradientActivationAudit
    parent_behavior_cache_audit: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "optimizer_steps": self.optimizer_steps,
            "optimizer_class": self.optimizer_class,
            "parent_greedy_prefix_count": self.parent_greedy_prefix_count,
            "child_on_policy_prefix_count": self.child_on_policy_prefix_count,
            "parent_output_forward_count": self.parent_output_forward_count,
            "child_output_forward_count": self.child_output_forward_count,
            "parent_greedy_cache_prompt_count": (self.parent_greedy_cache_prompt_count),
            "parent_greedy_cache_forward_count": (
                self.parent_greedy_cache_forward_count
            ),
            "invariant_parent_greedy_rows_cached_once": True,
            "gradient_activation": self.gradient_activation.as_dict(),
            "teacher_signal": "parent-causal-output-logits-and-actions",
            "parent_greedy_prefix_training": True,
            "child_greedy_on_policy_prefix_training": True,
            "weight_targets_used": False,
            "activation_targets_used": False,
            "hidden_state_targets_used": False,
            "attention_map_targets_used": False,
            "cache_value_targets_used": False,
        }
        if self.parent_behavior_cache_audit is not None:
            result["parent_behavior_cache"] = dict(
                self.parent_behavior_cache_audit
            )
        return result


@dataclass(frozen=True, slots=True)
class BehaviorCheckpoint:
    step: int
    metrics: CausalBehaviorMetrics
    factor_state_sha256: str
    pareto_safe_vs_step0: bool
    pareto_frontier: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BEHAVIOR_CHECKPOINT_SCHEMA,
            "step": self.step,
            "metrics": self.metrics.as_dict(),
            "factor_state_sha256": self.factor_state_sha256,
            "pareto_safe_vs_step0": self.pareto_safe_vs_step0,
            "pareto_frontier": self.pareto_frontier,
            "strict_selection_key": list(self.metrics.selection_key),
            "rollout_first_acquisition_key": list(
                self.metrics.rollout_first_acquisition_key
            ),
        }


@dataclass(frozen=True, slots=True)
class DurableFactorTensorSnapshot:
    """One immutable factor tensor for an external content-addressed sink."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    tensor_sha256: str
    raw_bytes: bytes

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise TargetNativeBehaviorDistillationError(
                "Durable factor tensor name cannot be blank."
            )
        shape = tuple(_index(value, "factor dimension") for value in self.shape)
        dtype = str(self.dtype)
        if not dtype.startswith("torch."):
            raise TargetNativeBehaviorDistillationError(
                "Durable factor tensor dtype must be a torch dtype name."
            )
        torch_dtype = getattr(torch, dtype.removeprefix("torch."), None)
        if not isinstance(torch_dtype, torch.dtype):
            raise TargetNativeBehaviorDistillationError(
                f"Unsupported durable tensor dtype {dtype!r}."
            )
        raw_bytes = bytes(self.raw_bytes)
        try:
            element_size = int(torch.empty((), dtype=torch_dtype).element_size())
        except (RuntimeError, TypeError) as error:
            raise TargetNativeBehaviorDistillationError(
                f"Unsupported durable tensor dtype {dtype!r}."
            ) from error
        if len(raw_bytes) != math.prod(shape) * element_size:
            raise TargetNativeBehaviorDistillationError(
                f"Durable factor tensor {name!r} has an invalid byte count."
            )
        try:
            finite_view = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype)
            finite = bool(torch.isfinite(finite_view).all())
        except (RuntimeError, TypeError) as error:
            raise TargetNativeBehaviorDistillationError(
                f"Unsupported durable tensor dtype {dtype!r}."
            ) from error
        if not finite:
            raise TargetNativeBehaviorDistillationError(
                f"Durable factor tensor {name!r} contains NaN or infinity."
            )
        digest = _sha256(self.tensor_sha256, "tensor_sha256")
        if (
            _raw_tensor_sha256(
                dtype=dtype,
                shape=shape,
                raw_bytes=raw_bytes,
            )
            != digest
        ):
            raise TargetNativeBehaviorDistillationError(
                f"Durable factor tensor {name!r} failed byte authentication."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "tensor_sha256", digest)
        object.__setattr__(self, "raw_bytes", raw_bytes)

    def as_manifest_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "tensor_sha256": self.tensor_sha256,
            "byte_count": len(self.raw_bytes),
        }

    def materialize_tensor(self) -> Tensor:
        """Return a new mutable CPU tensor without aliasing the stored bytes."""

        dtype = getattr(torch, self.dtype.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise TargetNativeBehaviorDistillationError(
                f"Unsupported durable tensor dtype {self.dtype!r}."
            )
        value = torch.frombuffer(bytearray(self.raw_bytes), dtype=dtype).clone()
        expected_numel = math.prod(self.shape)
        if int(value.numel()) != expected_numel:
            raise TargetNativeBehaviorDistillationError(
                f"Durable factor tensor {self.name!r} has an invalid byte count."
            )
        value = value.reshape(self.shape)
        if _tensor_sha256(value) != self.tensor_sha256:
            raise TargetNativeBehaviorDistillationError(
                f"Materialized durable tensor {self.name!r} failed authentication."
            )
        return value


@dataclass(frozen=True, slots=True)
class DurableFactorStateSnapshot:
    """Immutable, authenticated full factor state with no live tensor aliases."""

    schema: str
    tensors: tuple[DurableFactorTensorSnapshot, ...]
    factor_state_sha256: str
    lexical_rank: int | None = None
    guided_source_binding: GuidedFactorSourceBinding | None = None
    guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None

    def __post_init__(self) -> None:
        schema = str(self.schema)
        if schema not in {
            BEHAVIOR_FACTOR_PACK_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }:
            raise TargetNativeBehaviorDistillationError(
                f"Unsupported durable factor-state schema {schema!r}."
            )
        tensors = tuple(self.tensors)
        if not tensors or any(
            not isinstance(value, DurableFactorTensorSnapshot) for value in tensors
        ):
            raise TargetNativeBehaviorDistillationError(
                "Durable factor state requires immutable tensor snapshots."
            )
        names = tuple(value.name for value in tensors)
        if len(set(names)) != len(names) or names != tuple(sorted(names)):
            raise TargetNativeBehaviorDistillationError(
                "Durable factor tensors must have unique sorted names."
            )
        expected_names = set(_QKVO_GUD_FACTOR_NAMES)
        if schema in {
            BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }:
            expected_names.update(_LEXICAL_FACTOR_NAMES)
        if schema in {
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }:
            expected_names.update(_GUIDED_QKVO_FACTOR_NAMES)
        if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA:
            expected_names.update(_GUIDED_GUD_FACTOR_NAMES)
        if set(names) != expected_names:
            raise TargetNativeBehaviorDistillationError(
                "Durable factor names/count do not match the factor-pack schema."
            )
        lexical_rank = self.lexical_rank
        if schema == BEHAVIOR_FACTOR_PACK_SCHEMA:
            if (
                lexical_rank is not None
                or self.guided_source_binding is not None
                or self.guided_gud_source_binding is not None
            ):
                raise TargetNativeBehaviorDistillationError(
                    "A v1 durable factor state cannot carry lexical or guided metadata."
                )
        else:
            lexical_rank = _index(lexical_rank, "lexical_rank", positive=True)
        if schema in {
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }:
            if not isinstance(self.guided_source_binding, GuidedFactorSourceBinding):
                raise TargetNativeBehaviorDistillationError(
                    "A v3/v4 durable factor state requires guided source binding."
                )
        elif self.guided_source_binding is not None:
            raise TargetNativeBehaviorDistillationError(
                "Guided source binding is valid only for a v3/v4 durable factor state."
            )
        if schema == BEHAVIOR_FACTOR_PACK_V4_SCHEMA:
            if not isinstance(
                self.guided_gud_source_binding, GuidedGUDFactorSourceBinding
            ):
                raise TargetNativeBehaviorDistillationError(
                    "A v4 durable factor state requires guided-GUD source binding."
                )
            if (
                self.guided_gud_source_binding.source_guided_qkvo_binding
                != self.guided_source_binding
            ):
                raise TargetNativeBehaviorDistillationError(
                    "A v4 durable factor state's nested and top-level QKVO "
                    "bindings differ."
                )
        elif self.guided_gud_source_binding is not None:
            raise TargetNativeBehaviorDistillationError(
                "Guided-GUD source binding is valid only for a v4 durable factor state."
            )
        rows = [
            {
                "name": value.name,
                "shape": list(value.shape),
                "dtype": value.dtype,
                "tensor_sha256": value.tensor_sha256,
            }
            for value in tensors
        ]
        expected = _factor_state_sha256(
            rows,
            schema=schema,
            lexical_rank=lexical_rank,
            guided_source_binding=self.guided_source_binding,
            guided_gud_source_binding=self.guided_gud_source_binding,
        )
        digest = _sha256(self.factor_state_sha256, "factor_state_sha256")
        if expected != digest:
            raise TargetNativeBehaviorDistillationError(
                "Durable factor-state manifest failed authentication."
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "factor_state_sha256", digest)
        object.__setattr__(self, "lexical_rank", lexical_rank)

    def as_manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "factor_count": len(self.tensors),
            "factors": [value.as_manifest_row() for value in self.tensors],
            "factor_state_sha256": self.factor_state_sha256,
            "parent_state_included": False,
            "native_child_state_included": False,
            "optimizer_state_included": False,
        }
        if self.schema in {
            BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        }:
            result["lexical_rank"] = self.lexical_rank
        if self.guided_source_binding is not None:
            result["guided_source_binding"] = self.guided_source_binding.as_dict()
            result["guided_source_binding_sha256"] = (
                self.guided_source_binding.binding_sha256
            )
            result["inherited_product_regrouped"] = False
            result["guided_product_regrouped"] = False
        if self.guided_gud_source_binding is not None:
            result["guided_gud_source_binding"] = (
                self.guided_gud_source_binding.as_dict()
            )
            result["guided_gud_source_binding_sha256"] = (
                self.guided_gud_source_binding.binding_sha256
            )
            result["guided_gud_product_regrouped"] = False
        return result

    def materialize_factor_state(self) -> TargetNativeFactorState:
        """Materialize an authenticated, storage-only factor-state value.

        The returned state deliberately has no live parameter ownership.  Use
        :meth:`as_factor_payload` with ``restore_target_native_factor_payload``
        to authenticate it into a live compiler transaction.
        """

        tensors = tuple(value.materialize_tensor() for value in self.tensors)
        state = TargetNativeFactorState(
            names=tuple(value.name for value in self.tensors),
            tensors=tensors,
            tensor_sha256s=tuple(value.tensor_sha256 for value in self.tensors),
            factor_state_sha256=self.factor_state_sha256,
            schema=self.schema,
            lexical_rank=self.lexical_rank,
            guided_source_binding=self.guided_source_binding,
            guided_gud_source_binding=self.guided_gud_source_binding,
        )
        if state.as_manifest()["factor_state_sha256"] != self.factor_state_sha256:
            raise TargetNativeBehaviorDistillationError(
                "Materialized durable factor state failed authentication."
            )
        return state

    def as_factor_payload(self) -> dict[str, Any]:
        """Return the public payload accepted by factor-payload restoration."""

        return self.materialize_factor_state().as_payload()


@dataclass(frozen=True, slots=True)
class DurableBehaviorCheckpoint:
    """Immutable trainer-to-sink event emitted after full checkpoint capture."""

    step: int
    metrics: CausalBehaviorMetrics
    factor_state: DurableFactorStateSnapshot
    pareto_safe_vs_step0: bool
    checkpoint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _index(self.step, "checkpoint step"))
        if not isinstance(self.metrics, CausalBehaviorMetrics):
            raise TargetNativeBehaviorDistillationError(
                "Durable checkpoint metrics must be CausalBehaviorMetrics."
            )
        if not isinstance(self.factor_state, DurableFactorStateSnapshot):
            raise TargetNativeBehaviorDistillationError(
                "Durable checkpoint requires an authenticated factor state."
            )
        if not isinstance(self.pareto_safe_vs_step0, bool):
            raise TargetNativeBehaviorDistillationError(
                "pareto_safe_vs_step0 must be boolean."
            )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _canonical_sha256(self.as_manifest()),
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema": DURABLE_BEHAVIOR_CHECKPOINT_SCHEMA,
            "step": self.step,
            "metrics": self.metrics.as_dict(),
            "factor_state": self.factor_state.as_manifest(),
            "pareto_safe_vs_step0": self.pareto_safe_vs_step0,
            "behavior_metrics_complete": True,
            "selection_complete": False,
            "callback_controls_selection": False,
            "callback_controls_optimizer": False,
            "callback_controls_sealed_access": False,
            "callback_controls_numeric_promotion": False,
            "callback_controls_promotion": False,
        }


DurableBehaviorCheckpointCallback = Callable[[DurableBehaviorCheckpoint], None]


def _durable_behavior_checkpoint(
    *,
    step: int,
    metrics: CausalBehaviorMetrics,
    state: TargetNativeFactorState,
    pareto_safe_vs_step0: bool,
) -> DurableBehaviorCheckpoint:
    """Detach a factor state into immutable bytes before invoking user code."""

    tensors: list[DurableFactorTensorSnapshot] = []
    for name, tensor, digest in zip(
        state.names, state.tensors, state.tensor_sha256s, strict=True
    ):
        detached = tensor.detach().cpu().contiguous()
        tensors.append(
            DurableFactorTensorSnapshot(
                name=name,
                shape=tuple(detached.shape),
                dtype=str(detached.dtype),
                tensor_sha256=digest,
                raw_bytes=(detached.reshape(-1).view(torch.uint8).numpy().tobytes()),
            )
        )
    return DurableBehaviorCheckpoint(
        step=step,
        metrics=metrics,
        factor_state=DurableFactorStateSnapshot(
            schema=state.schema,
            tensors=tuple(tensors),
            factor_state_sha256=state.factor_state_sha256,
            lexical_rank=state.lexical_rank,
            guided_source_binding=state.guided_source_binding,
            guided_gud_source_binding=state.guided_gud_source_binding,
        ),
        pareto_safe_vs_step0=pareto_safe_vs_step0,
    )


def _invoke_durable_checkpoint_callback(
    callback: DurableBehaviorCheckpointCallback | None,
    *,
    transaction: ExactTargetCompilerTransaction,
    guided_tranche: Any | None,
    guided_source_binding: GuidedFactorSourceBinding | None,
    guided_gud_tranche: Any | None,
    guided_gud_source_binding: GuidedGUDFactorSourceBinding | None,
    step: int,
    metrics: CausalBehaviorMetrics,
    state: TargetNativeFactorState,
    pareto_safe_vs_step0: bool,
) -> None:
    if callback is None:
        return
    event = _durable_behavior_checkpoint(
        step=step,
        metrics=metrics,
        state=state,
        pareto_safe_vs_step0=pareto_safe_vs_step0,
    )
    # The event gets immutable bytes only and the return value is ignored.  The
    # callable itself remains trusted in-process code: arbitrary closure side
    # effects cannot be sandboxed here and belong in a separate process.  We do
    # guard and repair every factor's registry owner and storage across the call.
    bindings = _live_factor_bindings(
        transaction,
        guided_tranche=guided_tranche,
        guided_gud_tranche=guided_gud_tranche,
    )
    callback_error: BaseException | None = None
    try:
        callback(event)
    except BaseException as error:
        callback_error = error
    ownership_changed = _repair_live_factor_bindings(bindings)
    if callback_error is not None:
        if ownership_changed:
            callback_error.add_note(
                "Durable callback factor ownership/storage was repaired before "
                "exact step-zero restoration."
            )
        raise callback_error
    live_state = capture_target_native_factor_state(
        transaction,
        guided_tranche=guided_tranche,
        guided_source_binding=guided_source_binding,
        guided_gud_tranche=guided_gud_tranche,
        guided_gud_source_binding=guided_gud_source_binding,
    )
    if (
        ownership_changed
        or live_state.names != state.names
        or live_state.schema != state.schema
        or live_state.lexical_rank != state.lexical_rank
        or live_state.guided_source_binding != state.guided_source_binding
        or live_state.guided_gud_source_binding != state.guided_gud_source_binding
        or live_state.parameter_ids != state.parameter_ids
        or live_state.parameter_data_ptrs != state.parameter_data_ptrs
        or live_state.factor_state_sha256 != state.factor_state_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Durable checkpoint callback altered live compiler factors: value, "
            "ownership, or storage changed."
        )


@dataclass(frozen=True, slots=True)
class GuidedCandidateGateAudit:
    """Guided-only methodological gate evaluated before sealed access."""

    selected_step: int
    selected_step_is_promoted: bool
    minimum_lcp_at_least_4_count: int
    observed_lcp_at_least_4_count: int
    baseline_repeat_fraction_excess: float
    candidate_repeat_fraction_excess: float
    repeat_fraction_nonregression: bool
    baseline_adjacent_repeat_fraction_excess: float
    candidate_adjacent_repeat_fraction_excess: float
    adjacent_repeat_fraction_nonregression: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _guided_candidate_gate(
    *,
    selected_step: int,
    step0_metrics: CausalBehaviorMetrics,
    candidate_metrics: CausalBehaviorMetrics,
) -> GuidedCandidateGateAudit:
    step = _index(selected_step, "selected_step")
    if not isinstance(step0_metrics, CausalBehaviorMetrics) or not isinstance(
        candidate_metrics, CausalBehaviorMetrics
    ):
        raise TargetNativeBehaviorDistillationError(
            "Guided candidate gate requires causal behavior metrics."
        )
    promoted = step > 0
    lcp_passed = candidate_metrics.lcp_at_least_4_count >= 1
    repeat_passed = candidate_metrics.repeat_fraction_excess <= (
        step0_metrics.repeat_fraction_excess
    )
    adjacent_passed = candidate_metrics.adjacent_repeat_fraction_excess <= (
        step0_metrics.adjacent_repeat_fraction_excess
    )
    return GuidedCandidateGateAudit(
        selected_step=step,
        selected_step_is_promoted=promoted,
        minimum_lcp_at_least_4_count=1,
        observed_lcp_at_least_4_count=candidate_metrics.lcp_at_least_4_count,
        baseline_repeat_fraction_excess=step0_metrics.repeat_fraction_excess,
        candidate_repeat_fraction_excess=candidate_metrics.repeat_fraction_excess,
        repeat_fraction_nonregression=repeat_passed,
        baseline_adjacent_repeat_fraction_excess=(
            step0_metrics.adjacent_repeat_fraction_excess
        ),
        candidate_adjacent_repeat_fraction_excess=(
            candidate_metrics.adjacent_repeat_fraction_excess
        ),
        adjacent_repeat_fraction_nonregression=adjacent_passed,
        passed=promoted and lcp_passed and repeat_passed and adjacent_passed,
    )


@dataclass(frozen=True, slots=True)
class BehaviorDistillationResult:
    seed: int
    selection_policy: BehaviorSelectionPolicy
    selected_step: int
    checkpoints: tuple[BehaviorCheckpoint, ...]
    pareto_frontier_steps: tuple[int, ...]
    fit_metrics: CausalBehaviorMetrics
    candidate_metrics: CausalBehaviorMetrics
    sealed_metrics: CausalBehaviorMetrics | None
    candidate_thresholds: CandidateBehaviorThresholds
    candidate_gate_passed: bool
    sealed_evaluation_unlocked: bool
    training_audit: BehaviorTrainingAudit
    factor_pack: TargetNativeFactorState
    initial_factor_state_sha256: str | None
    initial_factor_payload_schema: str | None
    initial_factor_restore_mode: (
        Literal[
            "full",
            "qkvo-gud-partial",
            "guided-v3-full",
            "guided-v3-to-v4",
            "guided-v4-full",
        ]
        | None
    )
    combined_step0_factor_state_sha256: str
    lexical_initialization_state_sha256: str | None
    lexical_initialization_unchanged: bool | None
    domain_binding_sha256: str
    schedule_sha256: str
    guided_source_binding: GuidedFactorSourceBinding | None = None
    guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None
    guided_candidate_gate: GuidedCandidateGateAudit | None = None
    sealed_evaluation_policy: BehaviorSealedEvaluationPolicy = "candidate-gated"
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.guided_source_binding is None) != (self.guided_candidate_gate is None):
            raise TargetNativeBehaviorDistillationError(
                "Guided source binding and candidate gate evidence must be "
                "present together."
            )
        if self.guided_gud_source_binding is not None:
            if self.guided_source_binding is None or (
                self.guided_gud_source_binding.source_guided_qkvo_binding
                != self.guided_source_binding
            ):
                raise TargetNativeBehaviorDistillationError(
                    "Guided-GUD evidence requires its identical nested QKVO source "
                    "binding."
                )
        if self.sealed_evaluation_policy not in {"candidate-gated", "disabled"}:
            raise TargetNativeBehaviorDistillationError(
                "Result has an unsupported sealed-evaluation policy."
            )
        if self.sealed_evaluation_policy == "disabled" and (
            self.sealed_metrics is not None or self.sealed_evaluation_unlocked
        ):
            raise TargetNativeBehaviorDistillationError(
                "A candidate-only result cannot contain unlocked sealed evidence."
            )
        object.__setattr__(
            self,
            "evidence_sha256",
            _canonical_sha256(self.evidence_payload()),
        )

    @property
    def calibration_metrics(self) -> CausalBehaviorMetrics:
        return self.candidate_metrics

    def factor_pack_payload(self) -> dict[str, Any]:
        return self.factor_pack.as_payload()

    def evidence_payload(self) -> dict[str, Any]:
        """Return the exact canonical JSON preimage of ``evidence_sha256``."""

        payload = {
            "schema": TARGET_NATIVE_BEHAVIOR_DISTILLATION_SCHEMA,
            "seed": self.seed,
            "selection_policy": self.selection_policy,
            "selected_step": self.selected_step,
            "checkpoints": [value.as_dict() for value in self.checkpoints],
            "pareto_frontier_steps": list(self.pareto_frontier_steps),
            "fit_metrics": self.fit_metrics.as_dict(),
            "candidate_metrics": self.candidate_metrics.as_dict(),
            "sealed_metrics": (
                None if self.sealed_metrics is None else self.sealed_metrics.as_dict()
            ),
            "candidate_thresholds": self.candidate_thresholds.as_dict(),
            "candidate_gate_passed": self.candidate_gate_passed,
            "sealed_evaluation_unlocked": self.sealed_evaluation_unlocked,
            "training_audit": self.training_audit.as_dict(),
            "factor_pack_manifest": self.factor_pack.as_manifest(),
            "continued_from_factor_pack": self.initial_factor_state_sha256 is not None,
            "source_factor_state_sha256": self.initial_factor_state_sha256,
            "initial_factor_state_sha256": self.initial_factor_state_sha256,
            "initial_factor_payload_schema": self.initial_factor_payload_schema,
            "initial_factor_restore_mode": self.initial_factor_restore_mode,
            "combined_step0_factor_state_sha256": (
                self.combined_step0_factor_state_sha256
            ),
            "lexical_initialization_state_sha256": (
                self.lexical_initialization_state_sha256
            ),
            "lexical_initialization_unchanged": (self.lexical_initialization_unchanged),
            "domain_binding_sha256": self.domain_binding_sha256,
            "schedule_sha256": self.schedule_sha256,
            "selection_uses_internal_numeric_targets": False,
            "internal_numeric_matching_used": False,
        }
        if self.guided_source_binding is not None:
            payload.update(
                {
                    "guided_source_binding": (self.guided_source_binding.as_dict()),
                    "guided_source_binding_sha256": (
                        self.guided_source_binding.binding_sha256
                    ),
                    "guided_candidate_gate": self.guided_candidate_gate.as_dict(),
                    "selected_factor_pack_retained_on_acceptance": True,
                    "guided_schedule": {
                        "steps_1_through_8": "guided-left-only",
                        "steps_9_through_24": "guided-left-and-right-only",
                        "steps_25_through_64": ("joint-inherited-and-guided-qkvo"),
                        "gud_unlock_step": "bound-in-schedule_sha256",
                        "lexical_unlock_steps": "bound-in-schedule_sha256",
                        "inherited_product_regrouped": False,
                        "guided_product_regrouped": False,
                    },
                    "guided_lifecycle": {
                        "transaction_qkvo_handle_during_training": "inherited",
                        "guided_tranche_passed_separately": True,
                        "trainer_swapped_publication_handle": False,
                        "trainer_baked_candidate": False,
                        "trainer_rolled_back_candidate": False,
                        "reject_order_owned_by_runner": [
                            "guided-only-rollback",
                            "inherited-transaction-rollback",
                        ],
                        "accepted_publication_owned_by_runner": True,
                    },
                }
            )
        if self.guided_gud_source_binding is not None:
            payload.update(
                {
                    "guided_gud_source_binding": (
                        self.guided_gud_source_binding.as_dict()
                    ),
                    "guided_gud_source_binding_sha256": (
                        self.guided_gud_source_binding.binding_sha256
                    ),
                    "guided_gud_schedule": {
                        "steps_1_through_8": "guided-gud-left-only",
                        "steps_9_through_24": "guided-gud-left-and-right-only",
                        "steps_25_through_64": (
                            "guided-gud-plus-retained-inherited-and-guided-qkvo"
                        ),
                        "authorized_tensor_counts": [3, 6, 22],
                        "legacy_gud_frozen": True,
                        "lexical_frozen": True,
                        "guided_gud_product_regrouped": False,
                    },
                }
            )
        if self.sealed_evaluation_policy == "disabled":
            payload.update(
                {
                    "sealed_evaluation_policy": "disabled",
                    "candidate_only_evaluation": True,
                    "sealed_suite_supplied": False,
                    "sealed_evaluation_attempted": False,
                }
            )
        return payload

    def as_dict(self) -> dict[str, Any]:
        evidence = self.evidence_payload()
        return {
            **evidence,
            "step0_retained": bool(self.checkpoints and self.checkpoints[0].step == 0),
            "evidence_sha256": self.evidence_sha256,
            "evidence_payload": evidence,
            "selection_inputs": "calibration-output-actions-and-free-rollouts-only",
            "selection_policy_semantics": (
                "every causal-output diagnostic must weakly preserve step0 and at "
                "least one must improve"
                if self.selection_policy == "strict-nonregression"
                else "lexicographically maximize exact rollouts, rollouts with at "
                "least four LCP tokens, total LCP tokens, positional token matches; "
                "then minimize repetition excess before maximizing action agreements"
            ),
            "sealed_used_for_selection": False,
            "internal_numeric_matching_used": False,
            "optimizer_state_retained": False,
            "parent_runtime_reference_retained": False,
        }


def _model_device(model: nn.Module) -> torch.device:
    embeddings = getattr(model, "get_input_embeddings", None)
    if callable(embeddings):
        module = embeddings()
        weight = getattr(module, "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    for value in (*model.parameters(), *model.buffers()):
        return value.device
    raise TargetNativeBehaviorDistillationError("Model has no parameter device.")


def _forward_kwargs(model: nn.Module, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return values
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {
        name: value for name, value in values.items() if name in signature.parameters
    }


def _last_causal_logits(model: nn.Module, prefix: Sequence[int]) -> Tensor:
    device = _model_device(model)
    ids = torch.tensor([list(prefix)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    values = {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "logits_to_keep": 1,
    }
    kwargs = _forward_kwargs(model, values)
    output = model(**kwargs) if "input_ids" in kwargs else model(ids)
    logits: Any
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or int(logits.shape[0]) != 1
        or int(logits.shape[1]) <= 0
        or int(logits.shape[2]) <= 1
        or not logits.dtype.is_floating_point
    ):
        raise TargetNativeBehaviorDistillationError(
            "Causal model output must contain [1,sequence,vocab] floating logits."
        )
    action = logits[:, -1, :]
    if not bool(torch.isfinite(action.detach()).all()):
        raise TargetNativeBehaviorDistillationError(
            "Causal output logits are non-finite."
        )
    return action


def _greedy_path(
    model: nn.Module,
    prompt: Sequence[int],
    horizon: int,
) -> tuple[tuple[int, ...], tuple[Tensor, ...]]:
    prefix = tuple(prompt)
    selected: list[int] = []
    rows: list[Tensor] = []
    with torch.no_grad():
        for _step in range(horizon):
            logits = (
                _last_causal_logits(model, prefix).detach().cpu().contiguous().clone()
            )
            token = int(torch.argmax(logits, dim=-1).item())
            rows.append(logits)
            selected.append(token)
            prefix = (*prefix, token)
    return tuple(selected), tuple(rows)


def _forced_path(
    model: nn.Module,
    prompt: Sequence[int],
    selected: Sequence[int],
) -> tuple[Tensor, ...]:
    prefix = tuple(prompt)
    rows: list[Tensor] = []
    with torch.no_grad():
        for token in selected:
            rows.append(
                _last_causal_logits(model, prefix).detach().cpu().contiguous().clone()
            )
            prefix = (*prefix, int(token))
    return tuple(rows)


def _functional_trace(
    *,
    prompt: BehaviorPrompt,
    lane: Literal["forced", "greedy"],
    selected: Sequence[int],
    rows: Sequence[Tensor],
) -> FunctionalDecodeTrace:
    if len(selected) != len(rows) or not rows:
        raise TargetNativeBehaviorDistillationError(
            "A functional trace needs one output row per selected action."
        )
    steps: list[FunctionalDecodeStep] = []
    for index, (token, row) in enumerate(zip(selected, rows, strict=True)):
        greedy = int(torch.argmax(row, dim=-1).item())
        steps.append(
            FunctionalDecodeStep(
                step_index=index,
                prefix_length=len(prompt.token_ids) + index,
                selected_token_id=int(token),
                greedy_token_id=greedy,
                cache_length=len(prompt.token_ids) + index,
                action_logits=row.detach().cpu().contiguous().clone(),
                action_logits_sha256=_tensor_sha256(row),
            )
        )
    selected_tuple = tuple(int(value) for value in selected)
    payload = {
        "sequence_id": prompt.prompt_id,
        "lane": lane,
        "prompt_token_ids_sha256": prompt.token_ids_sha256,
        "selected_token_ids": list(selected_tuple),
        "action_logits_sha256s": [value.action_logits_sha256 for value in steps],
    }
    return FunctionalDecodeTrace(
        sequence_id=prompt.prompt_id,
        lane=lane,
        prompt_token_ids=prompt.token_ids,
        consumed_token_ids=selected_tuple[:-1],
        selected_token_ids=selected_tuple,
        steps=tuple(steps),
        vocabulary_size=int(rows[0].shape[-1]),
        logit_bytes=sum(int(value.numel() * value.element_size()) for value in rows),
        model_class="output-only-causal-model",
        cache_class="full-prefix-functional-decode",
        trace_sha256=_canonical_sha256(payload),
    )


def _parent_trace_at_horizon(
    trace: FunctionalDecodeTrace,
    horizon: int,
    *,
    clone_rows: bool = False,
) -> FunctionalDecodeTrace:
    """Return an authenticated prefix, optionally with private cloned rows."""

    horizon_value = _index(horizon, "parent trace horizon", positive=True)
    if horizon_value > len(trace.steps):
        raise TargetNativeBehaviorDistillationError(
            "Requested parent trace horizon exceeds its cached rows."
        )
    if horizon_value == len(trace.steps) and not clone_rows:
        return trace
    source_steps = trace.steps[:horizon_value]
    steps = (
        tuple(
            replace(
                step,
                action_logits=(
                    step.action_logits.detach().cpu().contiguous().clone()
                ),
            )
            for step in source_steps
        )
        if clone_rows
        else source_steps
    )
    selected = trace.selected_token_ids[:horizon_value]
    payload = {
        "sequence_id": trace.sequence_id,
        "lane": "greedy",
        "prompt_token_ids_sha256": token_sequence_sha256(trace.prompt_token_ids),
        "selected_token_ids": list(selected),
        "action_logits_sha256s": [value.action_logits_sha256 for value in steps],
    }
    result = FunctionalDecodeTrace(
        sequence_id=trace.sequence_id,
        lane="greedy",
        prompt_token_ids=trace.prompt_token_ids,
        consumed_token_ids=selected[:-1],
        selected_token_ids=selected,
        steps=steps,
        vocabulary_size=trace.vocabulary_size,
        logit_bytes=sum(
            int(value.action_logits.numel() * value.action_logits.element_size())
            for value in steps
        ),
        model_class=trace.model_class,
        cache_class=trace.cache_class,
        trace_sha256=_canonical_sha256(payload),
    )
    if clone_rows:
        # Authenticate the private copy itself.  This closes the interval
        # between validating the public cache and copying rows out of it.
        _full_prefix_parent_trace_manifest(result, horizon=horizon_value)
    return result


def build_parent_behavior_cache(
    parent_model: nn.Module,
    domain: BehaviorDomainSuite,
    *,
    horizon: int,
) -> ParentBehaviorCache:
    """Collect one hash-bound full-prefix parent cache for reuse across arms."""

    if not isinstance(parent_model, nn.Module):
        raise TargetNativeBehaviorDistillationError(
            "parent_model must be a torch module."
        )
    if not isinstance(domain, BehaviorDomainSuite):
        raise TargetNativeBehaviorDistillationError(
            "domain must be BehaviorDomainSuite."
        )
    horizon_value = _index(horizon, "parent cache horizon", positive=True)
    parent_authentication = _capture_parent_authentication(parent_model)
    modes = tuple((module, bool(module.training)) for module in parent_model.modules())
    traces: list[FunctionalDecodeTrace] = []
    try:
        parent_model.eval()
        for prompt in domain.prompts:
            tokens, rows = _greedy_path(
                parent_model,
                prompt.token_ids,
                horizon_value,
            )
            traces.append(
                _functional_trace(
                    prompt=prompt,
                    lane="greedy",
                    selected=tokens,
                    rows=rows,
                )
            )
    finally:
        for module, mode in modes:
            module.training = mode
    terminal_authentication = _capture_parent_authentication(parent_model)
    _assert_parent_authentication_matches(
        parent_authentication,
        terminal_authentication,
    )
    return ParentBehaviorCache(
        split_name=domain.split_name,
        artifact_sha256=domain.artifact_sha256,
        domain_sha256=domain.domain_sha256,
        parent_tensor_state_sha256=(
            parent_authentication.tensor_state_sha256
        ),
        parent_tensor_count=parent_authentication.tensor_count,
        parent_tensor_bytes=parent_authentication.tensor_bytes,
        parent_execution_contract_sha256=(
            parent_authentication.execution_contract_sha256
        ),
        horizon=horizon_value,
        traces=tuple(traces),
        _parent_model=parent_model,
        _parent_runtime_signature=parent_authentication.runtime_signature,
    )


def _authenticate_parent_behavior_cache_with_parent(
    parent_model: nn.Module,
    domain: BehaviorDomainSuite,
    cache: ParentBehaviorCache,
    *,
    horizon: int,
    parent_authentication: _ParentAuthentication,
) -> ParentBehaviorCacheBinding:
    """Authenticate one cache against an already-hashed shared parent state."""

    if not isinstance(parent_model, nn.Module) or not isinstance(
        domain, BehaviorDomainSuite
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent cache authentication requires a model and behavior domain."
        )
    if not isinstance(cache, ParentBehaviorCache):
        raise TargetNativeBehaviorDistillationError(
            "parent_behavior_cache must be ParentBehaviorCache."
        )
    requested_horizon = _index(horizon, "requested parent cache horizon", positive=True)
    if requested_horizon > cache.horizon:
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache horizon is shorter than requested."
        )
    if cache._parent_model is not parent_model:
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache belongs to another model object."
        )
    if (
        parent_authentication.runtime_signature
        != cache._parent_runtime_signature
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache runtime tensor ownership or versions changed."
        )
    if (
        parent_authentication.tensor_state_sha256
        != cache.parent_tensor_state_sha256
        or parent_authentication.tensor_count != cache.parent_tensor_count
        or parent_authentication.tensor_bytes != cache.parent_tensor_bytes
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache tensor-state bytes or topology changed."
        )
    if (
        parent_authentication.execution_contract_sha256
        != cache.parent_execution_contract_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache execution contract changed."
        )
    manifest = cache.manifest()
    if _canonical_sha256(manifest) != cache.cache_sha256:
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache canonical digest changed."
        )
    if (
        cache.split_name != domain.split_name
        or cache.artifact_sha256 != domain.artifact_sha256
        or cache.domain_sha256 != domain.domain_sha256
        or len(cache.traces) != len(domain.prompts)
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior cache does not bind the requested domain."
        )
    for prompt, trace, prompt_manifest in zip(
        domain.prompts,
        cache.traces,
        manifest["prompts"],
        strict=True,
    ):
        if (
            trace.sequence_id != prompt.prompt_id
            or trace.prompt_token_ids != prompt.token_ids
            or prompt_manifest["prompt_id"] != prompt.prompt_id
            or prompt_manifest["token_ids_sha256"] != prompt.token_ids_sha256
        ):
            raise TargetNativeBehaviorDistillationError(
                "Parent behavior cache prompt identity or token hash changed."
            )
    return ParentBehaviorCacheBinding(
        cache_sha256=cache.cache_sha256,
        domain_sha256=domain.domain_sha256,
        split_name=domain.split_name,
        parent_tensor_state_sha256=cache.parent_tensor_state_sha256,
        parent_execution_contract_sha256=(
            cache.parent_execution_contract_sha256
        ),
        cache_horizon=cache.horizon,
        requested_horizon=requested_horizon,
        traces=tuple(
            _parent_trace_at_horizon(
                trace,
                requested_horizon,
                clone_rows=True,
            )
            for trace in cache.traces
        ),
    )


def authenticate_parent_behavior_cache(
    parent_model: nn.Module,
    domain: BehaviorDomainSuite,
    cache: ParentBehaviorCache,
    *,
    horizon: int,
) -> ParentBehaviorCacheBinding:
    """Authenticate cached rows and the explicitly covered parent state."""

    if not isinstance(parent_model, nn.Module):
        raise TargetNativeBehaviorDistillationError(
            "Parent cache authentication requires a torch module."
        )
    parent_authentication = _capture_parent_authentication(parent_model)
    return _authenticate_parent_behavior_cache_with_parent(
        parent_model,
        domain,
        cache,
        horizon=horizon,
        parent_authentication=parent_authentication,
    )


def _parent_cache_binding_at_horizon(
    binding: ParentBehaviorCacheBinding,
    horizon: int,
) -> ParentBehaviorCacheBinding:
    """Derive a shorter binding without rehashing an already-authenticated cache."""

    requested_horizon = _index(horizon, "requested parent cache horizon", positive=True)
    if requested_horizon > binding.requested_horizon:
        raise TargetNativeBehaviorDistillationError(
            "Derived parent cache horizon exceeds its authenticated binding."
        )
    if requested_horizon == binding.requested_horizon:
        return binding
    return ParentBehaviorCacheBinding(
        cache_sha256=binding.cache_sha256,
        domain_sha256=binding.domain_sha256,
        split_name=binding.split_name,
        parent_tensor_state_sha256=binding.parent_tensor_state_sha256,
        parent_execution_contract_sha256=(
            binding.parent_execution_contract_sha256
        ),
        cache_horizon=binding.cache_horizon,
        requested_horizon=requested_horizon,
        traces=tuple(
            _parent_trace_at_horizon(trace, requested_horizon)
            for trace in binding.traces
        ),
    )


def evaluate_behavior_domain(
    parent_model: nn.Module,
    child_model: nn.Module,
    domain: BehaviorDomainSuite,
    *,
    horizon: int,
    top_k: int,
    split: BehaviorEvaluationSplit,
    parent_behavior_cache: ParentBehaviorCache | None = None,
) -> CausalBehaviorMetrics:
    """Evaluate fixed-parent actions plus independent free-running rollouts."""

    if parent_behavior_cache is None:
        return _evaluate_behavior_domain_impl(
            parent_model,
            child_model,
            domain,
            horizon=horizon,
            top_k=top_k,
            split=split,
            parent_cache_binding=None,
        )
    parent_authentication = _capture_parent_authentication(parent_model)
    parent_cache_binding = _authenticate_parent_behavior_cache_with_parent(
        parent_model,
        domain,
        parent_behavior_cache,
        horizon=horizon,
        parent_authentication=parent_authentication,
    )
    result = _evaluate_behavior_domain_impl(
        parent_model,
        child_model,
        domain,
        horizon=horizon,
        top_k=top_k,
        split=split,
        parent_cache_binding=parent_cache_binding,
    )
    terminal_authentication = _capture_parent_authentication(parent_model)
    _assert_parent_authentication_matches(
        parent_authentication,
        terminal_authentication,
    )
    _authenticate_parent_behavior_cache_with_parent(
        parent_model,
        domain,
        parent_behavior_cache,
        horizon=horizon,
        parent_authentication=terminal_authentication,
    )
    return result


def _evaluate_behavior_domain_impl(
    parent_model: nn.Module,
    child_model: nn.Module,
    domain: BehaviorDomainSuite,
    *,
    horizon: int,
    top_k: int,
    split: BehaviorEvaluationSplit,
    parent_cache_binding: ParentBehaviorCacheBinding | None,
) -> CausalBehaviorMetrics:
    """Evaluate behavior with a preauthenticated optional parent cache binding."""

    if not isinstance(parent_model, nn.Module) or not isinstance(
        child_model, nn.Module
    ):
        raise TargetNativeBehaviorDistillationError(
            "parent_model and child_model must be torch modules."
        )
    if parent_model is child_model:
        raise TargetNativeBehaviorDistillationError(
            "Parent and child must be distinct model objects."
        )
    if not isinstance(domain, BehaviorDomainSuite):
        raise TargetNativeBehaviorDistillationError(
            "domain must be BehaviorDomainSuite."
        )
    expected_splits: dict[str, BehaviorEvaluationSplit] = {
        "fit": "calibration-exploratory",
        "calibration": "calibration-selection",
        "candidate": "calibration-selection",
        "sealed": "heldout-evaluation",
    }
    expected_split = expected_splits.get(domain.split_name)
    if expected_split is not None and split != expected_split:
        raise TargetNativeBehaviorDistillationError(
            f"The {domain.split_name} domain must use split={expected_split!r}."
        )
    horizon_value = _index(horizon, "horizon", positive=True)
    top_k_value = _index(top_k, "top_k", positive=True)
    if parent_cache_binding is not None and (
        parent_cache_binding.domain_sha256 != domain.domain_sha256
        or parent_cache_binding.requested_horizon != horizon_value
        or len(parent_cache_binding.traces) != len(domain.prompts)
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent cache binding does not match this evaluation request."
        )
    parent_traces: list[FunctionalDecodeTrace] = []
    child_traces: list[FunctionalDecodeTrace] = []
    parent_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    child_modes = tuple(
        (module, bool(module.training)) for module in child_model.modules()
    )
    try:
        parent_model.eval()
        child_model.eval()
        for prompt_index, prompt in enumerate(domain.prompts):
            cached_parent_trace: FunctionalDecodeTrace | None = None
            if parent_cache_binding is None:
                parent_tokens, parent_rows = _greedy_path(
                    parent_model, prompt.token_ids, horizon_value
                )
            else:
                cached_parent_trace = parent_cache_binding.traces[prompt_index]
                if (
                    cached_parent_trace.sequence_id != prompt.prompt_id
                    or cached_parent_trace.prompt_token_ids != prompt.token_ids
                ):
                    raise TargetNativeBehaviorDistillationError(
                        "Parent cache trace order changed after authentication."
                    )
                parent_tokens = cached_parent_trace.selected_token_ids
                parent_rows = tuple(
                    value.action_logits for value in cached_parent_trace.steps
                )
            child_forced_rows = _forced_path(
                child_model, prompt.token_ids, parent_tokens
            )
            child_tokens, child_rows = _greedy_path(
                child_model, prompt.token_ids, horizon_value
            )
            parent_traces.extend(
                (
                    _functional_trace(
                        prompt=prompt,
                        lane="forced",
                        selected=parent_tokens,
                        rows=parent_rows,
                    ),
                    cached_parent_trace
                    if cached_parent_trace is not None
                    else _functional_trace(
                        prompt=prompt,
                        lane="greedy",
                        selected=parent_tokens,
                        rows=parent_rows,
                    ),
                )
            )
            child_traces.extend(
                (
                    _functional_trace(
                        prompt=prompt,
                        lane="forced",
                        selected=parent_tokens,
                        rows=child_forced_rows,
                    ),
                    _functional_trace(
                        prompt=prompt,
                        lane="greedy",
                        selected=child_tokens,
                        rows=child_rows,
                    ),
                )
            )
    finally:
        for module, mode in parent_modes:
            module.training = mode
        for module, mode in child_modes:
            module.training = mode
    report = score_residual_attention_function(
        trace_mapping(parent_traces),
        trace_mapping(child_traces),
        candidate_id=f"target-native-{domain.domain_sha256}",
        split=split,
        top_k=top_k_value,
    )
    return CausalBehaviorMetrics.from_report(report)


def _normalize_domains(
    domains: Mapping[str, BehaviorDomainSuite] | Sequence[BehaviorDomainSuite],
    *,
    sealed_evaluation_policy: BehaviorSealedEvaluationPolicy = "candidate-gated",
) -> Mapping[str, BehaviorDomainSuite]:
    sealed_enabled = sealed_evaluation_policy == "candidate-gated"
    expected_roles = (
        ("fit", "calibration", "sealed") if sealed_enabled else ("fit", "calibration")
    )
    if isinstance(domains, Mapping):
        values = dict(domains)
    elif not isinstance(domains, (str, bytes)) and isinstance(domains, Sequence):
        if len(domains) != len(expected_roles):
            raise TargetNativeBehaviorDistillationError(
                "Domain sequence does not match the sealed-evaluation policy."
            )
        values = dict(zip(expected_roles, domains, strict=True))
    else:
        raise TargetNativeBehaviorDistillationError(
            "domains must be a mapping or ordered policy-matched suite sequence."
        )
    if "candidate" in values and "calibration" not in values:
        values["calibration"] = values.pop("candidate")
    if set(values) != set(expected_roles):
        expected = ", ".join(expected_roles)
        raise TargetNativeBehaviorDistillationError(
            f"domains must contain exactly {expected} under the selected policy."
        )
    if any(not isinstance(value, BehaviorDomainSuite) for value in values.values()):
        raise TargetNativeBehaviorDistillationError(
            "Every domain value must be BehaviorDomainSuite."
        )
    for role, suite in values.items():
        allowed_names = {role, "candidate"} if role == "calibration" else {role}
        if suite.split_name not in allowed_names:
            raise TargetNativeBehaviorDistillationError(
                f"The {role} suite has split_name={suite.split_name!r}."
            )
    seen_ids: dict[str, str] = {}
    meaningful_prefixes: dict[str, str] = {}
    for role in expected_roles:
        for prompt in values[role].prompts:
            previous = seen_ids.get(prompt.prompt_id)
            if previous is not None:
                raise TargetNativeBehaviorDistillationError(
                    f"Prompt content overlaps {previous} and {role}."
                )
            seen_ids[prompt.prompt_id] = role
            for length in range(8, len(prompt.token_ids) + 1):
                prefix = token_sequence_sha256(prompt.token_ids[:length])
                previous = meaningful_prefixes.get(prefix)
                if previous is not None and previous != role:
                    raise TargetNativeBehaviorDistillationError(
                        f"A meaningful token prefix overlaps {previous} and {role}."
                    )
                meaningful_prefixes[prefix] = role
    return MappingProxyType(values)


def _normalize_parent_behavior_caches(
    caches: Mapping[str, ParentBehaviorCache] | None,
) -> Mapping[str, ParentBehaviorCache]:
    """Normalize optional fit/candidate cache roles without widening domains."""

    if caches is None:
        return MappingProxyType({})
    if not isinstance(caches, Mapping):
        raise TargetNativeBehaviorDistillationError(
            "parent_behavior_caches must be a mapping or None."
        )
    values = dict(caches)
    if "candidate" in values:
        if "calibration" in values:
            raise TargetNativeBehaviorDistillationError(
                "Provide only one of candidate or calibration parent caches."
            )
        values["calibration"] = values.pop("candidate")
    if not set(values).issubset({"fit", "calibration"}) or any(
        not isinstance(value, ParentBehaviorCache) for value in values.values()
    ):
        raise TargetNativeBehaviorDistillationError(
            "Parent behavior caches may contain only fit and candidate/calibration."
        )
    return MappingProxyType(values)


def _dominates(left: CausalBehaviorMetrics, right: CausalBehaviorMetrics) -> bool:
    pairs = tuple(zip(left.pareto_objectives, right.pareto_objectives, strict=True))
    return all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)


def _pareto_frontier(checkpoints: Sequence[BehaviorCheckpoint]) -> tuple[int, ...]:
    eligible = tuple(value for value in checkpoints if value.pareto_safe_vs_step0)
    return tuple(
        value.step
        for index, value in enumerate(eligible)
        if not any(
            other_index != index and _dominates(other.metrics, value.metrics)
            for other_index, other in enumerate(eligible)
        )
    )


def select_behavior_checkpoint(
    checkpoints: Sequence[BehaviorCheckpoint],
    *,
    selection_policy: BehaviorSelectionPolicy,
) -> BehaviorCheckpoint:
    """Select one calibration checkpoint under a preregistered output policy.

    ``strict-nonregression`` is appropriate when step zero already has useful
    behavior: a candidate must dominate it across every causal diagnostic.
    ``rollout-first-acquisition`` is appropriate for a blank stage: it ignores
    non-promoting diagnostics and lexicographically maximizes exact rollouts,
    rollouts reaching at least four LCP tokens, total LCP tokens, and positional
    matches; it then minimizes three repetition diagnostics before maximizing
    parent-action agreements.  Any repetition regression from step zero is
    ineligible, even when exact rollouts improve.  In either policy, a metric tie
    retains the earlier checkpoint, including step zero.
    """

    values = tuple(checkpoints)
    if (
        not values
        or any(not isinstance(value, BehaviorCheckpoint) for value in values)
        or values[0].step != 0
        or tuple(value.step for value in values)
        != tuple(sorted({value.step for value in values}))
    ):
        raise TargetNativeBehaviorDistillationError(
            "checkpoints must be unique, ordered, and retain step zero first."
        )
    policy = str(selection_policy).strip().lower()
    if policy not in {"strict-nonregression", "rollout-first-acquisition"}:
        raise TargetNativeBehaviorDistillationError(
            "selection_policy must be strict-nonregression or "
            "rollout-first-acquisition."
        )
    baseline = values[0]
    selected = baseline
    for candidate in values[1:]:
        if policy == "strict-nonregression":
            if not _dominates(candidate.metrics, baseline.metrics):
                continue
            key = (candidate.metrics.selection_key, candidate.step)
            selected_key = (selected.metrics.selection_key, selected.step)
        else:
            if (
                candidate.metrics.repeat_fraction_excess
                > baseline.metrics.repeat_fraction_excess
                or candidate.metrics.adjacent_repeat_fraction_excess
                > baseline.metrics.adjacent_repeat_fraction_excess
                or candidate.metrics.longest_run_excess
                > baseline.metrics.longest_run_excess
            ):
                continue
            key = (candidate.metrics.rollout_first_acquisition_key, candidate.step)
            selected_key = (
                selected.metrics.rollout_first_acquisition_key,
                selected.step,
            )
        if key < selected_key:
            selected = candidate
    return selected


def _training_loss(
    parent_logits: Tensor,
    child_logits: Tensor,
    schedule: BehaviorDistillationSchedule,
    *,
    prefix: Sequence[int],
    repetition_token_ids: Sequence[int] = (),
) -> Tensor:
    if tuple(parent_logits.shape) != tuple(child_logits.shape):
        raise TargetNativeBehaviorDistillationError(
            "Parent and child output vocabularies must match exactly."
        )
    teacher = parent_logits.detach().to(device=child_logits.device, dtype=torch.float32)
    student = child_logits.to(dtype=torch.float32)
    temperature = schedule.temperature
    teacher_probability = F.softmax(teacher / temperature, dim=-1)
    student_log_probability = F.log_softmax(student / temperature, dim=-1)
    distribution = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * (temperature * temperature)
    teacher_action = torch.argmax(teacher, dim=-1)
    action = F.cross_entropy(student, teacher_action)
    weighted = (
        schedule.distribution_loss_weight * distribution
        + schedule.action_loss_weight * action
    )
    total_weight = schedule.distribution_loss_weight + schedule.action_loss_weight
    if schedule.action_margin_loss_weight > 0.0:
        top_values, top_indices = torch.topk(student, k=2, dim=-1)
        best_other = torch.where(
            top_indices[:, 0] == teacher_action,
            top_values[:, 1],
            top_values[:, 0],
        )
        action_value = student.gather(1, teacher_action.unsqueeze(1)).squeeze(1)
        margin = F.relu(schedule.action_margin - (action_value - best_other)).mean()
        weighted = weighted + schedule.action_margin_loss_weight * margin
        total_weight += schedule.action_margin_loss_weight
    if schedule.repetition_excess_loss_weight > 0.0 and repetition_token_ids:
        recent = tuple(
            sorted(
                {
                    int(value)
                    for value in repetition_token_ids
                    if 0 <= int(value) < int(student.shape[-1])
                    and int(value) != int(teacher_action.item())
                }
            )
        )
        if recent:
            indices = torch.tensor(recent, dtype=torch.long, device=student.device)
            student_probability = student_log_probability.exp()
            student_mass = student_probability.index_select(-1, indices).sum(dim=-1)
            teacher_mass = teacher_probability.index_select(-1, indices).sum(dim=-1)
            repetition_excess = F.relu(student_mass - teacher_mass).mean()
            weighted = (
                weighted + schedule.repetition_excess_loss_weight * repetition_excess
            )
            total_weight += schedule.repetition_excess_loss_weight
    return weighted / total_weight


def _factor_learning_rate(
    name: str,
    schedule: BehaviorDistillationSchedule,
    *,
    step: int,
    guided_continuation: bool = False,
    guided_gud_continuation: bool = False,
) -> float:
    if name.startswith(("guided_qkvo.q_", "guided_qkvo.k_")):
        scale = schedule.guided_qk_learning_rate_scale
    elif name.startswith(("guided_qkvo.v_", "guided_qkvo.o_")):
        scale = schedule.guided_vo_learning_rate_scale
    elif name.startswith(("qkvo.q_", "qkvo.k_")):
        scale = schedule.qk_learning_rate_scale
    elif name.startswith(("qkvo.v_", "qkvo.o_")):
        scale = schedule.vo_learning_rate_scale
    elif name.startswith(("gud.", "guided_gud.")):
        scale = schedule.gud_learning_rate_scale
    elif name.startswith("lexical.embedding_"):
        scale = schedule.lexical_embedding_learning_rate_scale
    elif name.startswith("lexical.lm_head_"):
        scale = schedule.lexical_head_learning_rate_scale
    else:
        raise TargetNativeBehaviorDistillationError(
            f"No preregistered learning-rate group exists for factor {name!r}."
        )
    if name.endswith("_right"):
        if name.startswith("lexical."):
            scale *= schedule.lexical_right_factor_learning_rate_scale
        elif name.startswith("guided_qkvo."):
            scale *= schedule.guided_right_factor_learning_rate_scale
        elif name.startswith("guided_gud."):
            scale *= schedule.guided_right_factor_learning_rate_scale
        else:
            scale *= schedule.right_factor_learning_rate_scale
    if (
        guided_continuation
        and name.startswith("qkvo.")
        and step >= GUIDED_JOINT_START_STEP
    ):
        scale *= schedule.guided_joint_inherited_learning_rate_scale
    if schedule.learning_rate_decay_step_1 and step > (
        schedule.learning_rate_decay_step_1
    ):
        scale *= schedule.learning_rate_decay_scale_1
    if schedule.learning_rate_decay_step_2 and step > (
        schedule.learning_rate_decay_step_2
    ):
        scale *= schedule.learning_rate_decay_scale_2
    if schedule.learning_rate_decay_step_3 and step > (
        schedule.learning_rate_decay_step_3
    ):
        scale *= schedule.learning_rate_decay_scale_3
    return schedule.learning_rate * scale


def _factor_frozen(
    name: str,
    schedule: BehaviorDistillationSchedule,
    *,
    step: int,
    guided_continuation: bool = False,
    guided_gud_continuation: bool = False,
) -> bool:
    if guided_gud_continuation:
        if not guided_continuation:
            raise TargetNativeBehaviorDistillationError(
                "Guided-GUD freezing requires guided continuation mode."
            )
        if name.startswith("guided_gud."):
            return name.endswith("_right") and step <= GUIDED_LEFT_ONLY_END_STEP
        if name.startswith(("qkvo.", "guided_qkvo.")):
            return step <= GUIDED_ONLY_END_STEP
        if name.startswith(("gud.", "lexical.")):
            return True
        raise TargetNativeBehaviorDistillationError(
            f"No guided-GUD freeze group exists for factor {name!r}."
        )
    if guided_continuation:
        if name.startswith("guided_qkvo."):
            return name.endswith("_right") and step <= GUIDED_LEFT_ONLY_END_STEP
        if name.startswith("qkvo."):
            return step <= GUIDED_ONLY_END_STEP
        unlock_step = 0
        if name.startswith("gud."):
            unlock_step = schedule.guided_gud_unlock_step
        elif name.startswith("lexical.embedding_"):
            unlock_step = schedule.guided_lexical_embedding_unlock_step
        elif name.startswith("lexical.lm_head_"):
            unlock_step = schedule.guided_lexical_head_unlock_step
        else:
            raise TargetNativeBehaviorDistillationError(
                f"No guided freeze group exists for factor {name!r}."
            )
        return unlock_step == 0 or step < unlock_step
    if name.startswith("qkvo."):
        frozen_steps = schedule.qkvo_frozen_steps
    elif name.startswith("gud."):
        frozen_steps = schedule.gud_frozen_steps
    elif name.startswith("lexical.embedding_"):
        frozen_steps = schedule.lexical_embedding_frozen_steps
    elif name.startswith("lexical.lm_head_"):
        frozen_steps = schedule.lexical_head_frozen_steps
    else:
        raise TargetNativeBehaviorDistillationError(
            f"No preregistered freeze group exists for factor {name!r}."
        )
    return step <= frozen_steps


def _guided_training_stage(step: int) -> GuidedTrainingStage:
    value = _index(step, "step", positive=True)
    if value > GUIDED_CONTINUATION_STEPS:
        raise TargetNativeBehaviorDistillationError(
            "Guided continuation step exceeds the fixed 64-step schedule."
        )
    if value <= GUIDED_LEFT_ONLY_END_STEP:
        return "guided-left"
    if value <= GUIDED_ONLY_END_STEP:
        return "guided"
    return "joint"


def _authorized_factor_names(
    factor_entries: Sequence[tuple[str, nn.Parameter]],
    schedule: BehaviorDistillationSchedule,
    *,
    step: int,
    guided_continuation: bool,
    guided_gud_continuation: bool = False,
) -> tuple[str, ...]:
    return tuple(
        name
        for name, _value in factor_entries
        if not _factor_frozen(
            name,
            schedule,
            step=step,
            guided_continuation=guided_continuation,
            guided_gud_continuation=guided_gud_continuation,
        )
    )


class _GuidedCombinedOutputTrainingScope:
    """Freeze native weights while exposing the complete v3 or v4 factor set."""

    def __init__(
        self,
        transaction: ExactTargetCompilerTransaction,
        guided_tranche: Any,
        factor_entries: Sequence[tuple[str, nn.Parameter]],
        guided_gud_tranche: Any | None = None,
    ) -> None:
        self.transaction = transaction
        self.guided_tranche = guided_tranche
        self.guided_gud_tranche = guided_gud_tranche
        self.factor_entries = tuple(factor_entries)
        self._base: tuple[tuple[nn.Parameter, bool, Tensor | None, int, int], ...] = ()
        self._factors: tuple[tuple[nn.Parameter, bool, Tensor | None], ...] = ()
        self._modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise TargetNativeBehaviorDistillationError(
                "Guided combined output-training scope cannot be re-entered."
            )
        self.transaction.assert_live()
        self.guided_tranche.assert_live()
        if self.guided_gud_tranche is not None:
            self.guided_gud_tranche.assert_live()
        current = _factor_parameters(
            self.transaction,
            guided_tranche=self.guided_tranche,
            guided_gud_tranche=self.guided_gud_tranche,
        )
        if tuple((name, id(value)) for name, value in current) != tuple(
            (name, id(value)) for name, value in self.factor_entries
        ):
            raise TargetNativeBehaviorDistillationError(
                "Guided factor topology changed before entering training scope."
            )
        factors = tuple(value for _name, value in self.factor_entries)
        factor_ids = {id(value) for value in factors}
        model_parameters = tuple(self.transaction.model.parameters())
        expected_count = (
            GUIDED_GUD_COMBINED_FACTOR_TENSOR_COUNT
            if self.guided_gud_tranche is not None
            else GUIDED_COMBINED_FACTOR_TENSOR_COUNT
        )
        if len(factor_ids) != expected_count or not (
            factor_ids.issubset({id(value) for value in model_parameters})
        ):
            raise TargetNativeBehaviorDistillationError(
                f"The model does not own exactly all {expected_count} guided "
                "continuation "
                "factor tensors."
            )
        self._base = tuple(
            (
                value,
                bool(value.requires_grad),
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in model_parameters
            if id(value) not in factor_ids
        )
        self._factors = tuple(
            (value, bool(value.requires_grad), value.grad) for value in factors
        )
        self._modes = tuple(
            (module, bool(module.training))
            for module in self.transaction.model.modules()
        )
        for value, *_rest in self._base:
            value.requires_grad_(False)
        for value, _requires_grad, _gradient in self._factors:
            value.requires_grad_(True)
        self.transaction.model.eval()
        self._entered = True
        return factors

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        failed = exc_type is not None
        del exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._base:
            value.requires_grad_(requires_grad)
            value.grad = gradient
        for value, requires_grad, _gradient in self._factors:
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
            raise TargetNativeBehaviorDistillationError(
                "A native child parameter changed during guided factor-only training."
            )
        if not failed:
            self.transaction.assert_live()
            final_stage = getattr(self.guided_tranche, "training_stage", None)
            set_stage = getattr(self.guided_tranche, "set_training_stage", None)
            output_parameters = getattr(
                self.guided_tranche, "output_trainable_parameters", None
            )
            if (
                final_stage != "joint"
                or not callable(set_stage)
                or not callable(output_parameters)
            ):
                raise TargetNativeBehaviorDistillationError(
                    "A clean 64-step guided run must exit in the joint stage."
                )
            set_stage(final_stage)
            final_trainable = tuple(output_parameters())
            self.guided_tranche.assert_live()
            expected_qkvo_ids = {
                id(value)
                for name, value in self.factor_entries
                if name.startswith(("qkvo.", "guided_qkvo."))
            }
            if {id(value) for value in final_trainable} != expected_qkvo_ids:
                raise TargetNativeBehaviorDistillationError(
                    "The post-training guided composite is not live with exactly "
                    "the joint inherited/guided QKVO factors trainable."
                )
            if self.guided_gud_tranche is not None:
                final_gud_stage = getattr(
                    self.guided_gud_tranche, "training_stage", None
                )
                set_gud_stage = getattr(
                    self.guided_gud_tranche, "set_training_stage", None
                )
                gud_output_parameters = getattr(
                    self.guided_gud_tranche, "output_trainable_parameters", None
                )
                if (
                    final_gud_stage != "joint"
                    or not callable(set_gud_stage)
                    or not callable(gud_output_parameters)
                ):
                    raise TargetNativeBehaviorDistillationError(
                        "A clean 64-step guided-GUD run must exit in the joint stage."
                    )
                set_gud_stage(final_gud_stage)
                final_gud_trainable = tuple(gud_output_parameters())
                self.guided_gud_tranche.assert_live()
                guided_gud_ids = {
                    id(value)
                    for name, value in self.factor_entries
                    if name.startswith("guided_gud.")
                }
                allowed_gud_ids = {
                    id(value)
                    for name, value in self.factor_entries
                    if name.startswith(("gud.", "guided_gud."))
                }
                observed_gud_ids = {id(value) for value in final_gud_trainable}
                if not guided_gud_ids.issubset(observed_gud_ids) or not (
                    observed_gud_ids.issubset(allowed_gud_ids)
                ):
                    raise TargetNativeBehaviorDistillationError(
                        "The post-training guided-GUD composite does not expose its "
                        "complete separate guided product."
                    )


def _first_divergence(parent_tokens: Sequence[int], child_tokens: Sequence[int]) -> int:
    if len(parent_tokens) != len(child_tokens):
        raise TargetNativeBehaviorDistillationError(
            "Parent and child training rollouts have different horizons."
        )
    return next(
        (
            index
            for index, (parent, child) in enumerate(
                zip(parent_tokens, child_tokens, strict=True)
            )
            if int(parent) != int(child)
        ),
        len(parent_tokens),
    )


def _divergence_position_weights(
    horizon: int,
    divergence: int,
    schedule: BehaviorDistillationSchedule,
) -> tuple[float, ...]:
    values: list[float] = []
    for index in range(horizon):
        if index < divergence:
            focus = schedule.solved_prefix_loss_multiplier
        elif index == divergence:
            focus = schedule.divergence_loss_multiplier
        elif index <= divergence + schedule.post_divergence_positions:
            focus = schedule.post_divergence_loss_multiplier
        else:
            focus = 1.0
        late = (
            1.0
            if horizon == 1
            else 1.0
            + (schedule.late_position_loss_multiplier - 1.0) * index / (horizon - 1)
        )
        values.append(focus * late)
    return tuple(values)


def _trailing_run_length(values: Sequence[int]) -> int:
    if not values:
        return 0
    last = int(values[-1])
    count = 0
    for value in reversed(values):
        if int(value) != last:
            break
        count += 1
    return count


def _child_lane_indices(
    child_tokens: Sequence[int],
    divergence: int,
    schedule: BehaviorDistillationSchedule,
) -> tuple[int, ...]:
    selected: list[int] = []
    start = divergence if schedule.child_lane_starts_at_divergence else 0
    for index in range(start, len(child_tokens)):
        if (
            schedule.max_child_lane_prefixes
            and len(selected) >= schedule.max_child_lane_prefixes
        ):
            break
        if (
            schedule.child_repeat_stop_run_length
            and _trailing_run_length(child_tokens[:index])
            >= schedule.child_repeat_stop_run_length
        ):
            break
        selected.append(index)
    return tuple(selected)


def train_target_native_behavior_distillation(
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    domains: Mapping[str, BehaviorDomainSuite] | Sequence[BehaviorDomainSuite],
    schedule: BehaviorDistillationSchedule,
    seed: int = 0,
    *,
    candidate_thresholds: CandidateBehaviorThresholds | None = None,
    optimizer_factory: Callable[[Iterable[nn.Parameter]], Optimizer] | None = None,
    initial_factor_payload: Mapping[str, Any] | None = None,
    guided_tranche: Any | None = None,
    guided_gud_tranche: Any | None = None,
    durable_checkpoint_callback: DurableBehaviorCheckpointCallback | None = None,
    post_optimizer_step_callback: Callable[[int], None] | None = None,
    parent_behavior_caches: Mapping[str, ParentBehaviorCache] | None = None,
) -> BehaviorDistillationResult:
    """Train, Pareto-select, restore, and optionally unlock sealed evaluation.

    ``durable_checkpoint_callback`` receives an immutable authenticated snapshot
    after step zero and every normalized scheduled checkpoint.  Its return value
    is ignored and factor value/owner/storage mutations are detected and rolled
    back.  The callable is trusted in-process code; isolate an untrusted sink in
    another process because Python closure side effects cannot be sandboxed.

    ``post_optimizer_step_callback`` is a trusted in-process invariant hook.  It
    receives each completed one-based optimizer step immediately after finite
    factor validation.  Its return value is ignored, and it has no selection,
    checkpoint-promotion, or sealed-evaluation control channel.
    """

    if not isinstance(transaction, ExactTargetCompilerTransaction):
        raise TargetNativeBehaviorDistillationError(
            "transaction must be ExactTargetCompilerTransaction."
        )
    transaction.assert_live()
    if not isinstance(parent_model, nn.Module) or parent_model is transaction.model:
        raise TargetNativeBehaviorDistillationError(
            "parent_model must be a distinct torch module."
        )
    if not isinstance(schedule, BehaviorDistillationSchedule):
        raise TargetNativeBehaviorDistillationError(
            "schedule must be BehaviorDistillationSchedule."
        )
    if durable_checkpoint_callback is not None and not callable(
        durable_checkpoint_callback
    ):
        raise TargetNativeBehaviorDistillationError(
            "durable_checkpoint_callback must be callable or None."
        )
    if post_optimizer_step_callback is not None and not callable(
        post_optimizer_step_callback
    ):
        raise TargetNativeBehaviorDistillationError(
            "post_optimizer_step_callback must be callable or None."
        )
    guided_continuation = guided_tranche is not None
    guided_gud_continuation = guided_gud_tranche is not None
    if guided_gud_continuation and not guided_continuation:
        raise TargetNativeBehaviorDistillationError(
            "guided_gud_tranche requires guided_tranche."
        )
    if guided_continuation != schedule.guided_continuation_enabled:
        raise TargetNativeBehaviorDistillationError(
            "guided_tranche and guided_continuation_enabled must be supplied together."
        )
    if guided_continuation:
        if schedule.steps != GUIDED_CONTINUATION_STEPS:
            raise TargetNativeBehaviorDistillationError(
                "A guided continuation must use exactly 64 optimizer steps."
            )
        if not isinstance(
            initial_factor_payload, Mapping
        ) or initial_factor_payload.get("schema") not in {
            BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        }:
            raise TargetNativeBehaviorDistillationError(
                "A guided continuation must begin from one authenticated complete "
                "v2 or v3 factor payload."
            )
        if guided_gud_continuation and initial_factor_payload.get("schema") != (
            BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        ):
            raise TargetNativeBehaviorDistillationError(
                "A guided-GUD v4 continuation must begin from one authenticated "
                "complete v3 factor payload."
            )
    threshold = candidate_thresholds or CandidateBehaviorThresholds()
    if not isinstance(threshold, CandidateBehaviorThresholds):
        raise TargetNativeBehaviorDistillationError(
            "candidate_thresholds must be CandidateBehaviorThresholds."
        )
    seed_value = _index(seed, "seed")
    suites = _normalize_domains(
        domains,
        sealed_evaluation_policy=schedule.sealed_evaluation_policy,
    )
    fit = suites["fit"]
    calibration = suites["calibration"]
    sealed = suites.get("sealed")
    if schedule.fit_prompts_per_step > len(fit.prompts):
        raise TargetNativeBehaviorDistillationError(
            "fit_prompts_per_step exceeds the fit domain."
        )
    if any(
        len(prompt.token_ids) > schedule.max_prompt_tokens
        for suite in suites.values()
        for prompt in suite.prompts
    ):
        raise TargetNativeBehaviorDistillationError(
            "A prompt exceeds max_prompt_tokens."
        )
    worst_train = (
        schedule.steps * schedule.fit_prompts_per_step * 2 * schedule.fit_horizon
    )
    if worst_train > schedule.max_train_prefixes:
        raise TargetNativeBehaviorDistillationError(
            "Preregistered training work exceeds max_train_prefixes."
        )
    worst_evaluation = (
        4
        * schedule.evaluation_horizon
        * (
            len(schedule.normalized_checkpoint_steps) * len(calibration.prompts)
            + len(fit.prompts)
            + (0 if sealed is None else len(sealed.prompts))
        )
    )
    if worst_evaluation > schedule.max_evaluation_prefixes:
        raise TargetNativeBehaviorDistillationError(
            "Preregistered evaluation work exceeds max_evaluation_prefixes."
        )

    parent_caches = _normalize_parent_behavior_caches(parent_behavior_caches)
    fit_training_cache_binding: ParentBehaviorCacheBinding | None = None
    fit_evaluation_cache_binding: ParentBehaviorCacheBinding | None = None
    calibration_cache_binding: ParentBehaviorCacheBinding | None = None
    parent_authentication_before = (
        _capture_parent_authentication(parent_model) if parent_caches else None
    )
    parent_authentication_after: _ParentAuthentication | None = None
    if "fit" in parent_caches:
        if parent_authentication_before is None:
            raise TargetNativeBehaviorDistillationError(
                "Parent cache authentication state disappeared."
            )
        fit_full_binding = _authenticate_parent_behavior_cache_with_parent(
            parent_model,
            fit,
            parent_caches["fit"],
            horizon=max(schedule.fit_horizon, schedule.evaluation_horizon),
            parent_authentication=parent_authentication_before,
        )
        fit_training_cache_binding = _parent_cache_binding_at_horizon(
            fit_full_binding,
            schedule.fit_horizon,
        )
        fit_evaluation_cache_binding = _parent_cache_binding_at_horizon(
            fit_full_binding,
            schedule.evaluation_horizon,
        )
    if "calibration" in parent_caches:
        if parent_authentication_before is None:
            raise TargetNativeBehaviorDistillationError(
                "Parent cache authentication state disappeared."
            )
        calibration_cache_binding = _authenticate_parent_behavior_cache_with_parent(
            parent_model,
            calibration,
            parent_caches["calibration"],
            horizon=schedule.evaluation_horizon,
            parent_authentication=parent_authentication_before,
        )

    domain_payload = {role: suite.as_dict() for role, suite in suites.items()}
    domain_binding_sha256 = _canonical_sha256(domain_payload)
    schedule_sha256 = _canonical_sha256(schedule.as_dict())
    initial_factor_state_sha256 = None
    initial_factor_payload_schema = None
    initial_factor_restore_mode = None
    lexical_initialization_state_sha256 = None
    lexical_initialization_unchanged = None
    restored_initial_state: TargetNativeFactorState | None = None
    guided_source_binding: GuidedFactorSourceBinding | None = None
    guided_gud_source_binding: GuidedGUDFactorSourceBinding | None = None
    if initial_factor_payload is not None:
        if guided_gud_continuation:
            (
                restore_result,
                guided_source_binding,
                guided_gud_source_binding,
            ) = _restore_guided_v3_to_v4_initial_factor_payload(
                transaction,
                guided_tranche,
                guided_gud_tranche,
                initial_factor_payload,
            )
        elif (
            guided_continuation
            and initial_factor_payload.get("schema") == BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        ):
            restore_result, guided_source_binding = (
                _restore_guided_v3_initial_factor_payload(
                    transaction,
                    guided_tranche,
                    initial_factor_payload,
                )
            )
        else:
            restore_result = _restore_target_native_factor_payload(
                transaction, initial_factor_payload
            )
        initial_factor_state_sha256 = restore_result.source_factor_state_sha256
        initial_factor_payload_schema = restore_result.source_schema
        initial_factor_restore_mode = restore_result.restore_mode
        lexical_initialization_state_sha256 = (
            restore_result.lexical_initialization_state_sha256
        )
        lexical_initialization_unchanged = (
            restore_result.lexical_initialization_unchanged
        )
        restored_initial_state = restore_result.combined_state
        if guided_continuation and guided_source_binding is None:
            guided_source_binding = authenticate_guided_factor_sources(
                transaction,
                guided_tranche,
                restored_initial_state,
            )
    factor_entries = _factor_parameters(
        transaction,
        guided_tranche=guided_tranche,
        guided_gud_tranche=guided_gud_tranche,
    )
    factor_names = tuple(name for name, _value in factor_entries)
    finite_gradients: set[str] = set()
    nonzero_gradients: set[str] = set()
    authorized_gradients: set[str] = set()
    stage_authorized: dict[str, tuple[str, ...]] = {}
    gradient_observations = 0
    optimizer_class = ""
    parent_prefixes = 0
    child_prefixes = 0
    parent_forwards = 0
    child_forwards = 0
    parent_greedy_cache_forwards = 0

    step0_state = capture_target_native_factor_state(
        transaction,
        guided_tranche=guided_tranche,
        guided_source_binding=guided_source_binding,
        guided_gud_tranche=guided_gud_tranche,
        guided_gud_source_binding=guided_gud_source_binding,
    )
    if (
        restored_initial_state is not None
        and not guided_continuation
        and step0_state.factor_state_sha256
        != restored_initial_state.factor_state_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Combined step-zero capture differs from the authenticated restored "
            "continuation."
        )
    if guided_continuation and restored_initial_state is not None:
        step0_hash_by_name = dict(
            zip(step0_state.names, step0_state.tensor_sha256s, strict=True)
        )
        if any(
            step0_hash_by_name.get(name) != digest
            for name, digest in zip(
                restored_initial_state.names,
                restored_initial_state.tensor_sha256s,
                strict=True,
            )
        ):
            raise TargetNativeBehaviorDistillationError(
                "The inherited v2 tensor subset changed while the guided product "
                "was installed."
            )
        if (
            not guided_gud_continuation
            and initial_factor_payload_schema == BEHAVIOR_FACTOR_PACK_V3_SCHEMA
            and step0_state.factor_state_sha256
            != restored_initial_state.factor_state_sha256
        ):
            raise TargetNativeBehaviorDistillationError(
                "Combined guided step zero differs from the authenticated restored "
                "v3 continuation."
            )
    if parent_authentication_before is not None:
        _assert_parent_runtime_execution_unchanged(
            parent_model,
            parent_authentication_before,
        )
    step0_metrics = _evaluate_behavior_domain_impl(
        parent_model,
        transaction.model,
        calibration,
        horizon=schedule.evaluation_horizon,
        top_k=schedule.top_k,
        split="calibration-selection",
        parent_cache_binding=calibration_cache_binding,
    )
    checkpoints: list[BehaviorCheckpoint] = [
        BehaviorCheckpoint(
            step=0,
            metrics=step0_metrics,
            factor_state_sha256=step0_state.factor_state_sha256,
            pareto_safe_vs_step0=True,
        )
    ]
    selected_step = 0
    selected_state = step0_state
    selected_metrics = step0_metrics

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_value)
    order = torch.randperm(len(fit.prompts), generator=generator).tolist()
    if optimizer_factory is None:

        def default_optimizer_factory(
            parameters: Iterable[nn.Parameter],
        ) -> Optimizer:
            parameter_ids = {id(value) for value in parameters}
            groups: list[dict[str, Any]] = []
            for name, value in factor_entries:
                if id(value) not in parameter_ids:
                    raise TargetNativeBehaviorDistillationError(
                        "Default optimizer received the wrong factor set."
                    )
                groups.append(
                    {
                        "params": [value],
                        "lr": _factor_learning_rate(
                            name,
                            schedule,
                            step=1,
                            guided_continuation=guided_continuation,
                            guided_gud_continuation=guided_gud_continuation,
                        ),
                    }
                )
            return torch.optim.AdamW(
                groups,
                lr=schedule.learning_rate,
                weight_decay=0.0,
            )

        optimizer_factory = default_optimizer_factory

    parent_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    try:
        parent_model.eval()
        _invoke_durable_checkpoint_callback(
            durable_checkpoint_callback,
            transaction=transaction,
            guided_tranche=guided_tranche,
            guided_source_binding=guided_source_binding,
            guided_gud_tranche=guided_gud_tranche,
            guided_gud_source_binding=guided_gud_source_binding,
            step=0,
            metrics=step0_metrics,
            state=step0_state,
            pareto_safe_vs_step0=True,
        )
        parent_greedy_cache: dict[
            str, tuple[tuple[int, ...], tuple[Tensor, ...]]
        ] = {}
        for prompt_index, prompt in enumerate(fit.prompts):
            if fit_training_cache_binding is None:
                tokens, rows = _greedy_path(
                    parent_model, prompt.token_ids, schedule.fit_horizon
                )
                parent_forwards += schedule.fit_horizon
                parent_greedy_cache_forwards += schedule.fit_horizon
            else:
                trace = fit_training_cache_binding.traces[prompt_index]
                if (
                    trace.sequence_id != prompt.prompt_id
                    or trace.prompt_token_ids != prompt.token_ids
                ):
                    raise TargetNativeBehaviorDistillationError(
                        "Fit parent cache trace order changed after authentication."
                    )
                tokens = trace.selected_token_ids
                rows = tuple(value.action_logits for value in trace.steps)
            if len(tokens) != schedule.fit_horizon or len(rows) != len(tokens):
                raise TargetNativeBehaviorDistillationError(
                    "Invariant parent greedy cache has an invalid horizon."
                )
            parent_greedy_cache[prompt.prompt_id] = (tokens, rows)
        training_scope = (
            _GuidedCombinedOutputTrainingScope(
                transaction,
                guided_tranche,
                factor_entries,
                guided_gud_tranche,
            )
            if guided_continuation
            else transaction.output_training_scope()
        )
        with training_scope as trainable:
            optimizer = optimizer_factory(trainable)
            if not isinstance(optimizer, Optimizer):
                raise TargetNativeBehaviorDistillationError(
                    "optimizer_factory must return torch.optim.Optimizer."
                )
            optimizer_class = (
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            )
            optimizer_parameters = tuple(
                value for group in optimizer.param_groups for value in group["params"]
            )
            if len(optimizer_parameters) != len(trainable) or {
                id(value) for value in optimizer_parameters
            } != {id(value) for value in trainable}:
                raise TargetNativeBehaviorDistillationError(
                    "Optimizer must contain exactly the temporary compiler factors."
                )
            optimizer_rate_by_parameter = {
                id(value): _finite_float(
                    group.get("lr"), "optimizer learning rate", minimum=0.0
                )
                for group in optimizer.param_groups
                for value in group["params"]
            }
            if any(
                optimizer_rate_by_parameter[id(value)]
                != _factor_learning_rate(
                    name,
                    schedule,
                    step=1,
                    guided_continuation=guided_continuation,
                    guided_gud_continuation=guided_gud_continuation,
                )
                for name, value in factor_entries
            ):
                raise TargetNativeBehaviorDistillationError(
                    "Optimizer factor-group learning rates must match the "
                    "preregistered schedule."
                )
            factor_name_by_id = {id(value): name for name, value in factor_entries}
            for step in range(1, schedule.steps + 1):
                if parent_authentication_before is not None:
                    _assert_parent_runtime_execution_unchanged(
                        parent_model,
                        parent_authentication_before,
                    )
                for group in optimizer.param_groups:
                    expected_rates = {
                        _factor_learning_rate(
                            factor_name_by_id[id(value)],
                            schedule,
                            step=step,
                            guided_continuation=guided_continuation,
                            guided_gud_continuation=guided_gud_continuation,
                        )
                        for value in group["params"]
                    }
                    if len(expected_rates) != 1:
                        raise TargetNativeBehaviorDistillationError(
                            "One optimizer group spans incompatible preregistered "
                            "learning-rate schedules."
                        )
                    group["lr"] = expected_rates.pop()
                optimizer.zero_grad(set_to_none=True)
                if guided_continuation:
                    stage = _guided_training_stage(step)
                    runtime_stage = (
                        "joint"
                        if initial_factor_payload_schema
                        == BEHAVIOR_FACTOR_PACK_V3_SCHEMA
                        else stage
                    )
                    guided_tranche.set_training_stage(runtime_stage)
                    if guided_gud_continuation:
                        guided_gud_tranche.set_training_stage(stage)
                    authorized_names = _authorized_factor_names(
                        factor_entries,
                        schedule,
                        step=step,
                        guided_continuation=True,
                        guided_gud_continuation=guided_gud_continuation,
                    )
                    if (
                        not stage_authorized
                        or next(reversed(stage_authorized.values())) != authorized_names
                    ):
                        stage_authorized[f"{stage}-from-step-{step}"] = authorized_names
                    authorized_gradients.update(authorized_names)
                    authorized_set = set(authorized_names)
                    for name, parameter in factor_entries:
                        parameter.requires_grad_(name in authorized_set)
                selected_prompts = tuple(
                    fit.prompts[
                        order[
                            ((step - 1) * schedule.fit_prompts_per_step + offset)
                            % len(order)
                        ]
                    ]
                    for offset in range(schedule.fit_prompts_per_step)
                )
                active_horizon = (
                    schedule.warmup_fit_horizon
                    if step <= schedule.parent_only_steps
                    and schedule.warmup_fit_horizon
                    else schedule.fit_horizon
                )
                training_plans: list[
                    tuple[
                        BehaviorPrompt,
                        tuple[int, ...],
                        tuple[Tensor, ...],
                        tuple[int, ...],
                        tuple[float, ...],
                        tuple[int, ...],
                    ]
                ] = []
                divisor = 0.0
                for prompt in selected_prompts:
                    cached_parent_tokens, cached_parent_rows = parent_greedy_cache[
                        prompt.prompt_id
                    ]
                    parent_tokens = cached_parent_tokens[:active_horizon]
                    parent_rows = cached_parent_rows[:active_horizon]
                    child_tokens, _child_rows = _greedy_path(
                        transaction.model, prompt.token_ids, active_horizon
                    )
                    child_forwards += active_horizon
                    divergence = _first_divergence(parent_tokens, child_tokens)
                    position_weights = _divergence_position_weights(
                        active_horizon, divergence, schedule
                    )
                    child_indices: tuple[int, ...] = ()
                    if step > schedule.parent_only_steps:
                        child_indices = _child_lane_indices(
                            child_tokens, divergence, schedule
                        )
                    divisor += schedule.parent_lane_loss_weight * sum(position_weights)
                    divisor += schedule.child_lane_loss_weight * sum(
                        position_weights[index] for index in child_indices
                    )
                    training_plans.append(
                        (
                            prompt,
                            parent_tokens,
                            parent_rows,
                            child_tokens,
                            position_weights,
                            child_indices,
                        )
                    )
                if divisor <= 0.0 or not math.isfinite(divisor):
                    raise TargetNativeBehaviorDistillationError(
                        "Divergence curriculum produced an invalid loss divisor."
                    )
                for (
                    prompt,
                    parent_tokens,
                    parent_rows,
                    child_tokens,
                    position_weights,
                    child_indices,
                ) in training_plans:
                    lane_specs = (
                        (
                            0,
                            parent_tokens,
                            tuple(range(len(parent_tokens))),
                            schedule.parent_lane_loss_weight,
                        ),
                        (
                            1,
                            child_tokens,
                            child_indices,
                            schedule.child_lane_loss_weight,
                        ),
                    )
                    for (
                        lane_index,
                        lane_tokens,
                        active_indices,
                        lane_weight,
                    ) in lane_specs:
                        if lane_weight == 0.0:
                            continue
                        active = set(active_indices)
                        prefix = prompt.token_ids
                        for lane_step, token in enumerate(lane_tokens):
                            if lane_step not in active:
                                prefix = (*prefix, int(token))
                                continue
                            if lane_index == 0:
                                parent_logits = parent_rows[lane_step]
                            else:
                                with torch.no_grad():
                                    parent_logits = _last_causal_logits(
                                        parent_model, prefix
                                    )
                                parent_forwards += 1
                            child_logits = _last_causal_logits(
                                transaction.model, prefix
                            )
                            child_forwards += 1
                            repetition_tokens: tuple[int, ...] = ()
                            if lane_index == 1:
                                recent_child = lane_tokens[
                                    max(
                                        0, lane_step - schedule.repetition_window
                                    ) : lane_step
                                ]
                                repetition_tokens = tuple(
                                    sorted(
                                        {
                                            int(value)
                                            for value in recent_child
                                            if recent_child.count(value) >= 2
                                        }
                                    )
                                )
                            (
                                _training_loss(
                                    parent_logits,
                                    child_logits,
                                    schedule,
                                    prefix=prefix,
                                    repetition_token_ids=repetition_tokens,
                                )
                                * lane_weight
                                * position_weights[lane_step]
                                / divisor
                            ).backward()
                            if lane_index == 0:
                                parent_prefixes += 1
                            else:
                                child_prefixes += 1
                            prefix = (*prefix, int(token))
                gradient_observations += 1
                invalid_gradients: list[str] = []
                active_names = (
                    set(
                        _authorized_factor_names(
                            factor_entries,
                            schedule,
                            step=step,
                            guided_continuation=True,
                            guided_gud_continuation=guided_gud_continuation,
                        )
                    )
                    if guided_continuation
                    else set(factor_names)
                )
                for name, parameter in factor_entries:
                    gradient = parameter.grad
                    if name not in active_names:
                        if gradient is not None:
                            invalid_gradients.append(name)
                        continue
                    if gradient is None or not bool(torch.isfinite(gradient).all()):
                        invalid_gradients.append(name)
                        continue
                    finite_gradients.add(name)
                    if int(torch.count_nonzero(gradient).item()) > 0:
                        nonzero_gradients.add(name)
                if invalid_gradients:
                    raise TargetNativeBehaviorDistillationError(
                        "Output loss produced missing or non-finite compiler-factor "
                        f"gradients before the optimizer step: {invalid_gradients}."
                    )
                for name, parameter in factor_entries:
                    if (
                        _factor_frozen(
                            name,
                            schedule,
                            step=step,
                            guided_continuation=guided_continuation,
                            guided_gud_continuation=guided_gud_continuation,
                        )
                        and parameter.grad is not None
                    ):
                        # ``None`` makes AdamW skip both the update and its
                        # per-parameter step counter.  A zero gradient would
                        # silently age moments/bias correction while frozen.
                        parameter.grad = None
                try:
                    total_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        tuple(value for value in trainable if value.grad is not None),
                        schedule.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as error:
                    raise TargetNativeBehaviorDistillationError(
                        "Compiler-factor gradient norm became non-finite before the "
                        "optimizer step."
                    ) from error
                if not bool(torch.isfinite(total_gradient_norm).all()):
                    raise TargetNativeBehaviorDistillationError(
                        "Compiler-factor gradient norm became non-finite before the "
                        "optimizer step."
                    )
                optimizer.step()
                if any(
                    not bool(torch.isfinite(value.detach()).all())
                    for value in trainable
                ):
                    raise TargetNativeBehaviorDistillationError(
                        "Optimizer produced non-finite compiler factors."
                    )
                if post_optimizer_step_callback is not None:
                    post_optimizer_step_callback(step)
                if parent_authentication_before is not None:
                    _assert_parent_runtime_execution_unchanged(
                        parent_model,
                        parent_authentication_before,
                    )
                if guided_gud_continuation:
                    _assert_guided_gud_frozen_sources(
                        factor_entries,
                        guided_gud_source_binding,
                        step=step,
                    )
                if step in schedule.normalized_checkpoint_steps:
                    metrics = _evaluate_behavior_domain_impl(
                        parent_model,
                        transaction.model,
                        calibration,
                        horizon=schedule.evaluation_horizon,
                        top_k=schedule.top_k,
                        split="calibration-selection",
                        parent_cache_binding=calibration_cache_binding,
                    )
                    state = capture_target_native_factor_state(
                        transaction,
                        guided_tranche=guided_tranche,
                        guided_source_binding=guided_source_binding,
                        guided_gud_tranche=guided_gud_tranche,
                        guided_gud_source_binding=guided_gud_source_binding,
                    )
                    safe = (
                        metrics.pareto_objectives == step0_metrics.pareto_objectives
                        or _dominates(metrics, step0_metrics)
                    )
                    _invoke_durable_checkpoint_callback(
                        durable_checkpoint_callback,
                        transaction=transaction,
                        guided_tranche=guided_tranche,
                        guided_source_binding=guided_source_binding,
                        guided_gud_tranche=guided_gud_tranche,
                        guided_gud_source_binding=guided_gud_source_binding,
                        step=step,
                        metrics=metrics,
                        state=state,
                        pareto_safe_vs_step0=safe,
                    )
                    checkpoints.append(
                        BehaviorCheckpoint(
                            step=step,
                            metrics=metrics,
                            factor_state_sha256=state.factor_state_sha256,
                            pareto_safe_vs_step0=safe,
                        )
                    )
                    policy_winner = select_behavior_checkpoint(
                        checkpoints,
                        selection_policy=schedule.selection_policy,
                    )
                    if policy_winner.step == step:
                        selected_step = step
                        selected_state = state
                        selected_metrics = metrics
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
    except BaseException as training_error:
        if transaction.is_live:
            try:
                restore_target_native_factor_state(
                    transaction,
                    step0_state,
                    guided_tranche=guided_tranche,
                    guided_gud_tranche=guided_gud_tranche,
                )
            except Exception as restore_error:
                training_error.add_note(
                    f"Exact step-zero factor restoration also failed: {restore_error!r}"
                )
                raise training_error from restore_error
        raise
    finally:
        for module, mode in parent_modes:
            module.training = mode

    gradient_audit = GradientActivationAudit(
        factor_names=factor_names,
        finite_gradient_factor_names=tuple(sorted(finite_gradients)),
        nonzero_gradient_factor_names=tuple(sorted(nonzero_gradients)),
        backward_observation_count=gradient_observations,
        authorized_factor_names=tuple(
            sorted(authorized_gradients if guided_continuation else factor_names)
        ),
        stage_authorized_factor_names=tuple(stage_authorized.items()),
    )
    if not gradient_audit.complete:
        expected_gradients = (
            authorized_gradients if guided_continuation else set(factor_names)
        )
        missing = sorted(expected_gradients - nonzero_gradients)
        restore_target_native_factor_state(
            transaction,
            step0_state,
            guided_tranche=guided_tranche,
            guided_gud_tranche=guided_gud_tranche,
        )
        raise TargetNativeBehaviorDistillationError(
            "Not every target-native factor received a finite nonzero output-loss "
            f"gradient; inactive={missing}."
        )

    frontier_steps = _pareto_frontier(checkpoints)
    checkpoints = [
        replace(value, pareto_frontier=value.step in frontier_steps)
        for value in checkpoints
    ]
    guided_candidate_gate: GuidedCandidateGateAudit | None = None
    try:
        if parent_authentication_before is not None:
            _assert_parent_runtime_execution_unchanged(
                parent_model,
                parent_authentication_before,
            )
        restore_target_native_factor_state(
            transaction,
            selected_state,
            guided_tranche=guided_tranche,
            guided_gud_tranche=guided_gud_tranche,
        )
        factor_pack = capture_target_native_factor_state(
            transaction,
            guided_tranche=guided_tranche,
            guided_source_binding=guided_source_binding,
            guided_gud_tranche=guided_gud_tranche,
            guided_gud_source_binding=guided_gud_source_binding,
        )
        if factor_pack.factor_state_sha256 != selected_state.factor_state_sha256:
            raise TargetNativeBehaviorDistillationError(
                "Selected factor pack does not match the restored checkpoint."
            )
        fit_metrics = _evaluate_behavior_domain_impl(
            parent_model,
            transaction.model,
            fit,
            horizon=schedule.evaluation_horizon,
            top_k=schedule.top_k,
            split="calibration-exploratory",
            parent_cache_binding=fit_evaluation_cache_binding,
        )
        static_thresholds_passed = threshold.passes(selected_metrics)
        if guided_continuation:
            guided_candidate_gate = _guided_candidate_gate(
                selected_step=selected_step,
                step0_metrics=step0_metrics,
                candidate_metrics=selected_metrics,
            )
        candidate_passed = static_thresholds_passed and (
            guided_candidate_gate is None or guided_candidate_gate.passed
        )
        sealed_metrics = None
        if candidate_passed and schedule.sealed_evaluation_policy == "candidate-gated":
            if sealed is None:
                raise TargetNativeBehaviorDistillationError(
                    "Candidate-gated sealed evaluation requires a sealed suite."
                )
            sealed_metrics = evaluate_behavior_domain(
                parent_model,
                transaction.model,
                sealed,
                horizon=schedule.evaluation_horizon,
                top_k=schedule.top_k,
                split="heldout-evaluation",
            )
        if parent_authentication_before is not None:
            parent_authentication_after = _capture_parent_authentication(
                parent_model
            )
            _assert_parent_authentication_matches(
                parent_authentication_before,
                parent_authentication_after,
            )
            if "fit" in parent_caches:
                _authenticate_parent_behavior_cache_with_parent(
                    parent_model,
                    fit,
                    parent_caches["fit"],
                    horizon=max(
                        schedule.fit_horizon,
                        schedule.evaluation_horizon,
                    ),
                    parent_authentication=parent_authentication_after,
                )
            if "calibration" in parent_caches:
                _authenticate_parent_behavior_cache_with_parent(
                    parent_model,
                    calibration,
                    parent_caches["calibration"],
                    horizon=schedule.evaluation_horizon,
                    parent_authentication=parent_authentication_after,
                )
    except Exception:
        if transaction.is_live:
            restore_target_native_factor_state(
                transaction,
                step0_state,
                guided_tranche=guided_tranche,
                guided_gud_tranche=guided_gud_tranche,
            )
        raise

    parent_cache_audit: Mapping[str, Any] | None = None
    if fit_training_cache_binding is not None or calibration_cache_binding is not None:
        if (
            parent_authentication_before is None
            or parent_authentication_after is None
        ):
            raise TargetNativeBehaviorDistillationError(
                "Terminal parent cache authentication evidence disappeared."
            )
        cache_roles: dict[str, Any] = {}
        gross_avoided = 0
        cache_build_forwards = 0
        if fit_training_cache_binding is not None:
            if fit_evaluation_cache_binding is None:
                raise TargetNativeBehaviorDistillationError(
                    "Fit cache evaluation binding disappeared."
                )
            fit_avoided = (
                fit_training_cache_binding.parent_forward_count_avoided
                + fit_evaluation_cache_binding.parent_forward_count_avoided
            )
            gross_avoided += fit_avoided
            fit_cache = parent_caches["fit"]
            fit_build_forwards = len(fit_cache.traces) * fit_cache.horizon
            cache_build_forwards += fit_build_forwards
            cache_roles["fit"] = {
                "cache_sha256_before": fit_training_cache_binding.cache_sha256,
                "cache_sha256_after": fit_training_cache_binding.cache_sha256,
                "training_binding": fit_training_cache_binding.as_dict(),
                "exploratory_evaluation_binding": (
                    fit_evaluation_cache_binding.as_dict()
                ),
                "training_cache_hit": True,
                "exploratory_evaluation_cache_hit": True,
                "gross_parent_forward_count_avoided": fit_avoided,
                "one_time_cache_build_parent_forward_count": (
                    fit_build_forwards
                ),
                "terminal_cache_rows_reauthenticated": True,
            }
        if calibration_cache_binding is not None:
            evaluation_hit_count = len(checkpoints)
            calibration_avoided = (
                calibration_cache_binding.parent_forward_count_avoided
                * evaluation_hit_count
            )
            gross_avoided += calibration_avoided
            calibration_cache = parent_caches["calibration"]
            calibration_build_forwards = (
                len(calibration_cache.traces) * calibration_cache.horizon
            )
            cache_build_forwards += calibration_build_forwards
            cache_roles["calibration"] = {
                "cache_sha256_before": calibration_cache_binding.cache_sha256,
                "cache_sha256_after": calibration_cache_binding.cache_sha256,
                "evaluation_binding": calibration_cache_binding.as_dict(),
                "evaluation_cache_hit_count": evaluation_hit_count,
                "gross_parent_forward_count_avoided": calibration_avoided,
                "one_time_cache_build_parent_forward_count": (
                    calibration_build_forwards
                ),
                "terminal_cache_rows_reauthenticated": True,
            }
        cache_payload = {
            "schema": PARENT_BEHAVIOR_CACHE_AUDIT_SCHEMA,
            "cache_hit": True,
            "roles": cache_roles,
            "gross_parent_forward_count_avoided": gross_avoided,
            "one_time_cache_build_parent_forward_count": cache_build_forwards,
            "single_consumer_net_parent_forward_count_avoided": (
                gross_avoided - cache_build_forwards
            ),
            "cache_build_cost_not_charged_to_training_forward_counters": True,
            "parent_cache_rows_authenticated_before_training": True,
            "parent_cache_rows_reauthenticated_after_training": True,
            "parent_tensor_state_sha256_before": (
                parent_authentication_before.tensor_state_sha256
            ),
            "parent_tensor_state_sha256_after": (
                parent_authentication_after.tensor_state_sha256
            ),
            "parent_tensor_count": parent_authentication_before.tensor_count,
            "parent_tensor_bytes": parent_authentication_before.tensor_bytes,
            "parent_execution_contract_sha256_before": (
                parent_authentication_before.execution_contract_sha256
            ),
            "parent_execution_contract_sha256_after": (
                parent_authentication_after.execution_contract_sha256
            ),
            "covered_parent_execution_state_authenticated": True,
            "arbitrary_python_runtime_state_authenticated": False,
            "live_parent_outputs_reexecuted_at_terminal": False,
            "terminal_authentication_scope": (
                "cached-row-bytes-and-covered-execution-state"
            ),
        }
        parent_cache_audit = MappingProxyType(
            {
                **cache_payload,
                "audit_sha256": _canonical_sha256(cache_payload),
            }
        )

    training_audit = BehaviorTrainingAudit(
        optimizer_steps=schedule.steps,
        optimizer_class=optimizer_class,
        parent_greedy_prefix_count=parent_prefixes,
        child_on_policy_prefix_count=child_prefixes,
        parent_output_forward_count=parent_forwards,
        child_output_forward_count=child_forwards,
        parent_greedy_cache_prompt_count=len(parent_greedy_cache),
        parent_greedy_cache_forward_count=parent_greedy_cache_forwards,
        gradient_activation=gradient_audit,
        parent_behavior_cache_audit=parent_cache_audit,
    )
    return BehaviorDistillationResult(
        seed=seed_value,
        selection_policy=schedule.selection_policy,
        selected_step=selected_step,
        checkpoints=tuple(checkpoints),
        pareto_frontier_steps=frontier_steps,
        fit_metrics=fit_metrics,
        candidate_metrics=selected_metrics,
        sealed_metrics=sealed_metrics,
        candidate_thresholds=threshold,
        candidate_gate_passed=candidate_passed,
        sealed_evaluation_unlocked=(
            candidate_passed and schedule.sealed_evaluation_policy == "candidate-gated"
        ),
        training_audit=training_audit,
        factor_pack=factor_pack,
        initial_factor_state_sha256=initial_factor_state_sha256,
        initial_factor_payload_schema=initial_factor_payload_schema,
        initial_factor_restore_mode=initial_factor_restore_mode,
        combined_step0_factor_state_sha256=step0_state.factor_state_sha256,
        lexical_initialization_state_sha256=(lexical_initialization_state_sha256),
        lexical_initialization_unchanged=lexical_initialization_unchanged,
        domain_binding_sha256=domain_binding_sha256,
        schedule_sha256=schedule_sha256,
        guided_source_binding=guided_source_binding,
        guided_gud_source_binding=guided_gud_source_binding,
        guided_candidate_gate=guided_candidate_gate,
        sealed_evaluation_policy=schedule.sealed_evaluation_policy,
    )


__all__ = [
    "BEHAVIOR_CHECKPOINT_SCHEMA",
    "BEHAVIOR_DOMAIN_SCHEMA",
    "BEHAVIOR_FACTOR_PACK_SCHEMA",
    "BEHAVIOR_FACTOR_PACK_V2_SCHEMA",
    "BEHAVIOR_FACTOR_PACK_V3_SCHEMA",
    "BEHAVIOR_FACTOR_PACK_V4_SCHEMA",
    "DURABLE_BEHAVIOR_CHECKPOINT_SCHEMA",
    "GUIDED_COMBINED_FACTOR_TENSOR_COUNT",
    "GUIDED_GUD_COMBINED_FACTOR_TENSOR_COUNT",
    "GUIDED_GUD_FACTOR_TENSOR_COUNT",
    "GUIDED_CONTINUATION_STEPS",
    "GUIDED_JOINT_START_STEP",
    "GUIDED_LEFT_ONLY_END_STEP",
    "GUIDED_ONLY_END_STEP",
    "GUIDED_QKVO_FACTOR_TENSOR_COUNT",
    "GUIDED_QKVO_PARAMETER_COUNT",
    "PARENT_BEHAVIOR_CACHE_AUDIT_SCHEMA",
    "PARENT_BEHAVIOR_CACHE_BINDING_SCHEMA",
    "PARENT_BEHAVIOR_CACHE_SCHEMA",
    "PARENT_EXECUTION_CONTRACT_SCHEMA",
    "PARENT_TENSOR_STATE_SCHEMA",
    "TARGET_NATIVE_BEHAVIOR_DISTILLATION_SCHEMA",
    "BehaviorCheckpoint",
    "BehaviorDistillationResult",
    "BehaviorDistillationSchedule",
    "BehaviorDomainSuite",
    "BehaviorPrompt",
    "BehaviorSealedEvaluationPolicy",
    "BehaviorSelectionPolicy",
    "BehaviorTrainingAudit",
    "CandidateBehaviorThresholds",
    "CausalBehaviorMetrics",
    "DurableBehaviorCheckpoint",
    "DurableBehaviorCheckpointCallback",
    "DurableFactorStateSnapshot",
    "DurableFactorTensorSnapshot",
    "GradientActivationAudit",
    "GuidedCandidateGateAudit",
    "GuidedFactorSourceBinding",
    "GuidedGUDFactorSourceBinding",
    "GuidedTrainingStage",
    "ParentBehaviorCache",
    "ParentBehaviorCacheBinding",
    "TargetNativeBehaviorDistillationError",
    "TargetNativeFactorState",
    "authenticate_guided_factor_sources",
    "authenticate_guided_gud_factor_sources",
    "authenticate_parent_behavior_cache",
    "build_parent_behavior_cache",
    "capture_target_native_factor_state",
    "evaluate_behavior_domain",
    "load_behavior_domain_suite",
    "restore_target_native_factor_state",
    "restore_target_native_factor_payload",
    "select_behavior_checkpoint",
    "train_target_native_behavior_distillation",
]
