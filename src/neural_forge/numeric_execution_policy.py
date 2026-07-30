"""Authenticated numeric-domain policy for analytic model folding.

The folding system has several numeric domains which must not be conflated:
checkpoint storage, model forward execution, analytic accumulation, closed-form
solves, matrix factorizations, target publication, high-precision verification,
and verification in the target's real runtime dtype.  This module resolves and
seals those domains without executing a model or mutating a checkpoint.

Only explicitly typed backend failures may retry on the deterministic CPU/FP64
fallback.  Numerical invalidity, rank failure, authentication failure, shape
failure, and arbitrary ``RuntimeError`` instances are intentionally outside the
fallback hierarchy.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import torch

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256


NUMERIC_EXECUTION_POLICY_SCHEMA = "neural-forge-numeric-execution-policy-v1"
NUMERIC_MODEL_INVENTORY_SCHEMA = "neural-forge-numeric-model-inventory-v1"
NUMERIC_CAST_LEDGER_SCHEMA = "neural-forge-numeric-cast-ledger-v1"
NUMERIC_EXECUTION_RECEIPT_SCHEMA = "neural-forge-numeric-execution-receipt-v1"

SUPPORTED_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
_SUPPORTED_FLOAT_DTYPE_SET = frozenset(SUPPORTED_FLOAT_DTYPES)
_DTYPE_TO_TORCH = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_DTYPE_ELEMENT_SIZE = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
_DTYPE_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "torch.float32": "float32",
    "float64": "float64",
    "fp64": "float64",
    "double": "float64",
    "torch.float64": "float64",
}
_STORAGE_SENTINEL = "inventory-native"
_MODEL_NATIVE_SENTINEL = "model-native"
_TARGET_NATIVE_SENTINEL = "target-native"
_MIXED_SENTINEL = "mixed"
_PLACEMENT_SENTINELS = frozenset(
    {
        _STORAGE_SENTINEL,
        _MODEL_NATIVE_SENTINEL,
        _TARGET_NATIVE_SENTINEL,
        _MIXED_SENTINEL,
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NumericDomain(str, Enum):
    SOURCE_STORAGE = "source_storage"
    TARGET_STORAGE = "target_storage"
    SOURCE_FORWARD = "source_forward"
    TARGET_FORWARD = "target_forward"
    ACCUMULATION = "accumulation"
    SOLVE = "solve"
    FACTORIZATION = "factorization"
    PUBLICATION = "publication"
    VERIFICATION = "verification"
    TARGET_VERIFICATION = "target_verification"


_DOMAIN_ORDER = tuple(domain.value for domain in NumericDomain)
_FALLBACK_DOMAINS = frozenset(
    {NumericDomain.SOLVE.value, NumericDomain.FACTORIZATION.value}
)
_DEFAULT_REQUIRED_OPERATIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        NumericDomain.SOURCE_STORAGE.value: ("storage",),
        NumericDomain.TARGET_STORAGE.value: ("storage",),
        NumericDomain.SOURCE_FORWARD.value: ("model_forward",),
        NumericDomain.TARGET_FORWARD.value: ("model_forward",),
        NumericDomain.ACCUMULATION.value: ("matmul", "reduction"),
        NumericDomain.SOLVE.value: ("linear_solve",),
        NumericDomain.FACTORIZATION.value: ("svd", "cholesky"),
        NumericDomain.PUBLICATION.value: ("cast",),
        NumericDomain.VERIFICATION.value: ("matmul", "reduction"),
        NumericDomain.TARGET_VERIFICATION.value: ("model_forward",),
    }
)
_FACTORIZATION_OPERATIONS = frozenset(
    {
        "cholesky",
        "eigh",
        "eigvalsh",
        "linear_solve",
        "lstsq",
        "pinv",
        "svd",
        "svdvals",
    }
)
_OPERATION_AUDIT = MappingProxyType(
    {
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "filesystem_writes": 0,
        "network_access": False,
    }
)


class NumericExecutionPolicyError(ValueError):
    """The requested numeric policy is invalid or cannot be resolved."""


class NumericExecutionError(RuntimeError):
    """Base error for an operation governed by a numeric policy."""

    reason_code = "numeric-execution-error"


class NumericFallbackEligibleError(NumericExecutionError):
    """Base class for the only failures allowed to retry before publication."""


class NumericUnsupportedBackendError(NumericFallbackEligibleError):
    reason_code = "unsupported-backend"


class NumericDeviceLossError(NumericFallbackEligibleError):
    reason_code = "device-loss"


class NumericPrepublicationOutOfMemoryError(NumericFallbackEligibleError):
    reason_code = "prepublication-out-of-memory"


class NumericNonFiniteError(NumericExecutionError):
    reason_code = "nonfinite"


class NumericRankError(NumericExecutionError):
    reason_code = "rank-failure"


class NumericAuthenticationError(NumericExecutionError):
    reason_code = "authentication-failure"


class NumericShapeError(NumericExecutionError):
    reason_code = "shape-failure"


class NumericFallbackExhaustedError(NumericExecutionError):
    """All authenticated execution candidates failed for eligible reasons."""

    reason_code = "fallback-exhausted"

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = verify_numeric_execution_receipt(receipt)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise NumericExecutionPolicyError(
        f"Numeric receipt contains a non-JSON value: {type(value).__name__}."
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value.strip()):
        raise NumericExecutionPolicyError(
            f"{label} must be a nonempty stable identifier."
        )
    return value.strip()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise NumericExecutionPolicyError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def normalize_numeric_dtype(value: Any) -> str:
    """Return the canonical name of one supported floating dtype."""

    if isinstance(value, torch.dtype):
        value = str(value)
    if not isinstance(value, str):
        raise NumericExecutionPolicyError(
            "Numeric dtype must be FP16, BF16, FP32, or FP64."
        )
    result = _DTYPE_ALIASES.get(value.strip().lower())
    if result is None:
        raise NumericExecutionPolicyError(
            f"Unsupported floating dtype {value!r}; expected one of "
            f"{SUPPORTED_FLOAT_DTYPES}."
        )
    return result


def torch_numeric_dtype(value: Any) -> torch.dtype:
    return _DTYPE_TO_TORCH[normalize_numeric_dtype(value)]


def _normalize_device(value: Any, *, allow_sentinel: bool = False) -> str:
    if allow_sentinel and isinstance(value, str) and value in _PLACEMENT_SENTINELS:
        return value
    if not isinstance(value, (str, torch.device)):
        raise NumericExecutionPolicyError(
            "Numeric device must be exactly 'cpu' or an indexed 'cuda:N'."
        )
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise NumericExecutionPolicyError(
            "Numeric device must be exactly 'cpu' or an indexed 'cuda:N'."
        ) from exc
    if device.type == "cpu" and device.index is None:
        return "cpu"
    if device.type != "cuda" or device.index is None or device.index < 0:
        raise NumericExecutionPolicyError(
            "Numeric device must be exactly 'cpu' or an indexed 'cuda:N'."
        )
    return f"cuda:{device.index}"


def _dtype_inventory_name(value: torch.dtype) -> str:
    raw = str(value)
    if raw in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[raw]
    return raw.removeprefix("torch.")


def _named_tensors(model: Any, method_name: str) -> tuple[tuple[str, Any], ...]:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise NumericExecutionPolicyError(
            f"Model inventory requires a callable {method_name}()."
        )
    try:
        values = tuple(method(remove_duplicate=False))
    except TypeError:
        values = tuple(method())
    result: list[tuple[str, Any]] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise NumericExecutionPolicyError(
                f"{method_name}() returned an invalid named tensor entry."
            )
        name, tensor = item
        if not isinstance(name, str) or not name:
            raise NumericExecutionPolicyError(
                f"{method_name}() returned an unnamed tensor."
            )
        if not isinstance(tensor, torch.Tensor):
            raise NumericExecutionPolicyError(
                f"{method_name}() entry {name!r} is not a tensor."
            )
        result.append((name, tensor))
    return tuple(result)


_INVENTORY_KEYS = frozenset(
    {
        "schema",
        "role",
        "entries",
        "entry_count",
        "parameter_binding_count",
        "buffer_binding_count",
        "unique_tensor_object_count",
        "floating_binding_count",
        "floating_parameter_dtype_counts",
        "floating_buffer_dtype_counts",
        "floating_dtype_counts",
        "floating_device_counts",
        "mixed_floating_parameter_dtypes",
        "mixed_floating_storage_dtypes",
        "operation_audit",
        "inventory_sha256",
    }
)
_INVENTORY_ENTRY_KEYS = frozenset(
    {
        "name",
        "kind",
        "shape",
        "numel",
        "dtype",
        "device",
        "floating",
        "requires_grad",
    }
)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NumericExecutionPolicyError(f"{label} must be a non-negative integer.")
    return value


def _validate_operation_audit(value: Any) -> None:
    if value != dict(_OPERATION_AUDIT):
        raise NumericExecutionPolicyError("Numeric operation audit is invalid.")


def _validate_inventory_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _plain_json(value)
    if set(receipt) != _INVENTORY_KEYS:
        raise NumericExecutionPolicyError(
            "Numeric model inventory fields are incomplete or unknown."
        )
    if receipt["schema"] != NUMERIC_MODEL_INVENTORY_SCHEMA:
        raise NumericExecutionPolicyError("Numeric model inventory schema is invalid.")
    _identifier(receipt["role"], "inventory role")
    entries = receipt["entries"]
    if not isinstance(entries, list) or not entries:
        raise NumericExecutionPolicyError("Numeric model inventory is empty.")
    if receipt["entry_count"] != len(entries):
        raise NumericExecutionPolicyError("Numeric model inventory count is invalid.")
    names_by_kind: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _INVENTORY_ENTRY_KEYS:
            raise NumericExecutionPolicyError("Numeric model inventory entry is invalid.")
        name = entry["name"]
        kind = entry["kind"]
        if not isinstance(name, str) or not name or kind not in {"parameter", "buffer"}:
            raise NumericExecutionPolicyError("Numeric model inventory binding is invalid.")
        if (kind, name) in names_by_kind:
            raise NumericExecutionPolicyError("Numeric model inventory binding is duplicated.")
        names_by_kind.add((kind, name))
        shape = entry["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise NumericExecutionPolicyError("Numeric model inventory shape is invalid.")
        expected_numel = math.prod(shape)
        if _nonnegative_int(entry["numel"], "inventory numel") != expected_numel:
            raise NumericExecutionPolicyError("Numeric model inventory numel is invalid.")
        if not isinstance(entry["dtype"], str) or not entry["dtype"]:
            raise NumericExecutionPolicyError("Numeric model inventory dtype is invalid.")
        if not isinstance(entry["device"], str) or not entry["device"]:
            raise NumericExecutionPolicyError("Numeric model inventory device is invalid.")
        if not isinstance(entry["floating"], bool) or not isinstance(
            entry["requires_grad"], bool
        ):
            raise NumericExecutionPolicyError("Numeric model inventory flags are invalid.")
        if entry.get("floating") and entry.get("dtype") not in _SUPPORTED_FLOAT_DTYPE_SET:
            raise NumericExecutionPolicyError(
                "Numeric model inventory contains an unsupported floating dtype."
            )
    parameter_count = sum(entry["kind"] == "parameter" for entry in entries)
    buffer_count = sum(entry["kind"] == "buffer" for entry in entries)
    floating_entries = [entry for entry in entries if entry["floating"]]
    if (
        _nonnegative_int(
            receipt["parameter_binding_count"], "parameter_binding_count"
        )
        != parameter_count
        or _nonnegative_int(receipt["buffer_binding_count"], "buffer_binding_count")
        != buffer_count
        or _nonnegative_int(receipt["floating_binding_count"], "floating_binding_count")
        != len(floating_entries)
    ):
        raise NumericExecutionPolicyError("Numeric model inventory totals are invalid.")
    unique_count = _nonnegative_int(
        receipt["unique_tensor_object_count"], "unique_tensor_object_count"
    )
    if unique_count <= 0 or unique_count > len(entries):
        raise NumericExecutionPolicyError(
            "Numeric model inventory unique tensor count is invalid."
        )
    for key in (
        "floating_parameter_dtype_counts",
        "floating_buffer_dtype_counts",
        "floating_dtype_counts",
        "floating_device_counts",
    ):
        counts = receipt[key]
        if not isinstance(counts, Mapping) or any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for name, count in counts.items()
        ):
            raise NumericExecutionPolicyError(
                f"Numeric model inventory {key} is invalid."
            )
    def expected_counts(values: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in values:
            label = str(entry[key])
            result[label] = result.get(label, 0) + 1
        return dict(sorted(result.items()))

    floating_parameters = [
        entry for entry in floating_entries if entry["kind"] == "parameter"
    ]
    floating_buffers = [
        entry for entry in floating_entries if entry["kind"] == "buffer"
    ]
    expected_count_maps = {
        "floating_parameter_dtype_counts": expected_counts(
            floating_parameters, "dtype"
        ),
        "floating_buffer_dtype_counts": expected_counts(floating_buffers, "dtype"),
        "floating_dtype_counts": expected_counts(floating_entries, "dtype"),
        "floating_device_counts": expected_counts(floating_entries, "device"),
    }
    if any(receipt[key] != expected for key, expected in expected_count_maps.items()):
        raise NumericExecutionPolicyError(
            "Numeric model inventory dtype/device counts are inconsistent."
        )
    expected_mixed_flags = {
        "mixed_floating_parameter_dtypes": (
            len({entry["dtype"] for entry in floating_parameters}) > 1
        ),
        "mixed_floating_storage_dtypes": (
            len({entry["dtype"] for entry in floating_entries}) > 1
        ),
    }
    if any(receipt[key] is not expected for key, expected in expected_mixed_flags.items()):
        raise NumericExecutionPolicyError(
            "Numeric model inventory mixed-dtype flags are inconsistent."
        )
    _validate_operation_audit(receipt["operation_audit"])
    supplied = _digest(receipt["inventory_sha256"], "inventory_sha256")
    body = {key: item for key, item in receipt.items() if key != "inventory_sha256"}
    if supplied != canonical_sha256(body):
        raise NumericExecutionPolicyError("Numeric model inventory seal is invalid.")
    return receipt


@dataclass(frozen=True, slots=True)
class ModelNumericInventory:
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt",
            _deep_freeze(_validate_inventory_receipt(self.receipt)),
        )

    @property
    def role(self) -> str:
        return str(self.receipt["role"])

    @property
    def inventory_sha256(self) -> str:
        return str(self.receipt["inventory_sha256"])

    @property
    def mixed_floating_parameter_dtypes(self) -> bool:
        return bool(self.receipt["mixed_floating_parameter_dtypes"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def inventory_model_numeric_state(model: Any, *, role: str) -> ModelNumericInventory:
    """Inventory model tensors without moving, casting, or executing the model."""

    role = _identifier(role, "inventory role")
    named_parameters = _named_tensors(model, "named_parameters")
    named_buffers = _named_tensors(model, "named_buffers")
    entries: list[dict[str, Any]] = []
    unique_ids: set[int] = set()
    for kind, values in (("parameter", named_parameters), ("buffer", named_buffers)):
        for name, tensor in values:
            unique_ids.add(id(tensor))
            floating = bool(tensor.is_floating_point())
            dtype = _dtype_inventory_name(tensor.dtype)
            if floating and dtype not in _SUPPORTED_FLOAT_DTYPE_SET:
                raise NumericExecutionPolicyError(
                    f"{role} tensor {name!r} uses unsupported floating dtype {dtype!r}."
                )
            entries.append(
                {
                    "name": name,
                    "kind": kind,
                    "shape": [int(item) for item in tensor.shape],
                    "numel": int(tensor.numel()),
                    "dtype": dtype,
                    "device": str(tensor.device),
                    "floating": floating,
                    "requires_grad": bool(getattr(tensor, "requires_grad", False)),
                }
            )
    entries.sort(key=lambda item: (item["kind"], item["name"]))
    if not entries:
        raise NumericExecutionPolicyError(f"{role} model has no tensor inventory.")
    floating_entries = [entry for entry in entries if entry["floating"]]
    if not floating_entries:
        raise NumericExecutionPolicyError(f"{role} model has no floating tensors.")

    def counts(values: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in values:
            label = str(entry[key])
            result[label] = result.get(label, 0) + 1
        return dict(sorted(result.items()))

    parameter_floating = [
        entry
        for entry in floating_entries
        if entry["kind"] == "parameter"
    ]
    buffer_floating = [
        entry for entry in floating_entries if entry["kind"] == "buffer"
    ]
    body: dict[str, Any] = {
        "schema": NUMERIC_MODEL_INVENTORY_SCHEMA,
        "role": role,
        "entries": entries,
        "entry_count": len(entries),
        "parameter_binding_count": len(named_parameters),
        "buffer_binding_count": len(named_buffers),
        "unique_tensor_object_count": len(unique_ids),
        "floating_binding_count": len(floating_entries),
        "floating_parameter_dtype_counts": counts(parameter_floating, "dtype"),
        "floating_buffer_dtype_counts": counts(buffer_floating, "dtype"),
        "floating_dtype_counts": counts(floating_entries, "dtype"),
        "floating_device_counts": counts(floating_entries, "device"),
        "mixed_floating_parameter_dtypes": (
            len({entry["dtype"] for entry in parameter_floating}) > 1
        ),
        "mixed_floating_storage_dtypes": (
            len({entry["dtype"] for entry in floating_entries}) > 1
        ),
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    return ModelNumericInventory(
        {**body, "inventory_sha256": canonical_sha256(body)}
    )


@dataclass(frozen=True, slots=True)
class NumericExecutionRequest:
    source_forward_device: Any | None = None
    source_forward_dtype: Any | None = None
    target_forward_device: Any | None = None
    target_forward_dtype: Any | None = None
    accumulation_device: Any | None = None
    accumulation_dtype: Any = "float32"
    solve_device: Any = "cpu"
    solve_dtype: Any = "float64"
    factorization_device: Any | None = None
    factorization_dtype: Any | None = None
    publication_device: Any | None = None
    publication_dtype: Any | None = None
    verification_device: Any = "cpu"
    verification_dtype: Any = "float64"
    target_verification_device: Any | None = None
    target_verification_dtype: Any | None = None
    allow_cpu_fp64_fallback: bool = True
    deterministic_algorithms_required: bool = True
    cpu_thread_count: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "allow_cpu_fp64_fallback",
            "deterministic_algorithms_required",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise NumericExecutionPolicyError(f"{field_name} must be boolean.")
        if not self.deterministic_algorithms_required:
            raise NumericExecutionPolicyError(
                "Analytic numeric policy requires deterministic algorithms."
            )
        if (
            isinstance(self.cpu_thread_count, bool)
            or not isinstance(self.cpu_thread_count, int)
            or self.cpu_thread_count <= 0
        ):
            raise NumericExecutionPolicyError("cpu_thread_count must be positive.")


@dataclass(frozen=True, slots=True)
class NumericConcretePlacement:
    device: str
    dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _normalize_device(self.device))
        object.__setattr__(self, "dtype", normalize_numeric_dtype(self.dtype))

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch_numeric_dtype(self.dtype)

    def as_dict(self) -> dict[str, str]:
        return {"device": self.device, "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class NumericDomainPlacement:
    domain: str
    requested_device: str
    requested_dtype: str
    resolved_device: str
    resolved_dtype: str
    resolution_reason: str
    resolution_fallback_used: bool
    inventory_sha256s: tuple[str, ...]
    execution_candidates: tuple[NumericConcretePlacement, ...]
    capability_records: tuple[Mapping[str, Any], ...]

    @property
    def is_concrete(self) -> bool:
        return (
            self.resolved_device not in _PLACEMENT_SENTINELS
            and self.resolved_dtype not in _PLACEMENT_SENTINELS
        )

    @property
    def concrete(self) -> NumericConcretePlacement:
        if not self.is_concrete:
            raise NumericExecutionPolicyError(
                f"{self.domain} placement is model-native rather than concrete."
            )
        return NumericConcretePlacement(self.resolved_device, self.resolved_dtype)

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "requested_device": self.requested_device,
            "requested_dtype": self.requested_dtype,
            "resolved_device": self.resolved_device,
            "resolved_dtype": self.resolved_dtype,
            "resolution_reason": self.resolution_reason,
            "resolution_fallback_used": self.resolution_fallback_used,
            "inventory_sha256s": list(self.inventory_sha256s),
            "execution_candidates": [
                candidate.as_dict() for candidate in self.execution_candidates
            ],
            "capability_records": [
                _plain_json(record) for record in self.capability_records
            ],
        }


def _cuda_environment() -> dict[str, Any]:
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        if count < 0:
            raise RuntimeError("negative CUDA device count")
    except Exception:  # pragma: no cover - backend failures vary by runtime
        return {
            "cuda_runtime_available": False,
            "cuda_device_count": 0,
            "cuda_probe_failed": True,
        }
    return {
        "cuda_runtime_available": available,
        "cuda_device_count": count,
        "cuda_probe_failed": False,
    }


def numeric_capability_key(operation: str, device: Any, dtype: Any) -> str:
    return "|".join(
        (
            _identifier(operation, "numeric operation"),
            _normalize_device(device),
            normalize_numeric_dtype(dtype),
        )
    )


def _normalize_capability_overrides(
    value: Mapping[Any, Any] | None,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise NumericExecutionPolicyError("capability_overrides must be a mapping.")
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_key, tuple) and len(raw_key) == 3:
            key = numeric_capability_key(*raw_key)
        elif isinstance(raw_key, str):
            parts = raw_key.split("|")
            if len(parts) != 3:
                raise NumericExecutionPolicyError(
                    "Capability override keys must be operation|device|dtype."
                )
            key = numeric_capability_key(*parts)
        else:
            raise NumericExecutionPolicyError(
                "Capability override keys must be triples or canonical strings."
            )
        if isinstance(raw_value, bool):
            body = {
                "supported": raw_value,
                "reason": "explicit-supported" if raw_value else "explicit-unsupported",
                "runtime_preflight_required": False,
            }
        elif isinstance(raw_value, Mapping):
            if not isinstance(raw_value.get("supported"), bool):
                raise NumericExecutionPolicyError(
                    "Capability override supported flag must be boolean."
                )
            reason = raw_value.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise NumericExecutionPolicyError(
                    "Capability override reason must be nonempty text."
                )
            preflight = raw_value.get("runtime_preflight_required", False)
            if not isinstance(preflight, bool):
                raise NumericExecutionPolicyError(
                    "runtime_preflight_required must be boolean."
                )
            body = {
                "supported": raw_value["supported"],
                "reason": reason.strip(),
                "runtime_preflight_required": preflight,
            }
        else:
            raise NumericExecutionPolicyError(
                "Capability override values must be booleans or mappings."
            )
        if key in result:
            raise NumericExecutionPolicyError(f"Duplicate capability override {key!r}.")
        result[key] = body
    return result


def _resolve_capability(
    operation: str,
    placement: NumericConcretePlacement,
    *,
    environment: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    operation = _identifier(operation, "numeric operation")
    key = numeric_capability_key(operation, placement.device, placement.dtype)
    override = overrides.get(key)
    if override is not None:
        return {
            "key": key,
            "operation": operation,
            **placement.as_dict(),
            "supported": bool(override["supported"]),
            "source": "explicit-override",
            "reason": str(override["reason"]),
            "runtime_preflight_required": bool(
                override["runtime_preflight_required"]
            ),
        }
    device = torch.device(placement.device)
    if device.type == "cuda":
        cuda_available = bool(environment["cuda_runtime_available"])
        cuda_count = int(environment["cuda_device_count"])
        if not cuda_available:
            supported = False
            reason = (
                "cuda-probe-failed"
                if environment["cuda_probe_failed"]
                else "cuda-runtime-unavailable"
            )
        elif device.index is None or device.index >= cuda_count:
            supported = False
            reason = "cuda-index-unavailable"
        else:
            supported = True
            reason = "device-visible"
    else:
        supported = True
        reason = "cpu-available"
    if supported and operation in _FACTORIZATION_OPERATIONS:
        supported = placement.dtype in {"float32", "float64"}
        reason = (
            "factorization-dtype-supported"
            if supported
            else "low-precision-factorization-disallowed"
        )
    runtime_preflight = operation in {"model_forward", "matmul", "reduction"}
    if placement.dtype in {"float16", "bfloat16"}:
        runtime_preflight = True
    return {
        "key": key,
        "operation": operation,
        **placement.as_dict(),
        "supported": supported,
        "source": "builtin-conservative-capability-table",
        "reason": reason,
        "runtime_preflight_required": runtime_preflight,
    }


def _normalized_required_operations(
    overrides: Mapping[Any, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    result = {
        domain: tuple(sorted(set(operations)))
        for domain, operations in _DEFAULT_REQUIRED_OPERATIONS.items()
    }
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping):
        raise NumericExecutionPolicyError("required_operations must be a mapping.")
    for raw_domain, raw_operations in overrides.items():
        domain = _domain_name(raw_domain)
        if isinstance(raw_operations, str) or not isinstance(raw_operations, Sequence):
            raise NumericExecutionPolicyError(
                f"Required operations for {domain} must be a nonempty sequence."
            )
        operations = tuple(
            sorted({_identifier(item, "numeric operation") for item in raw_operations})
        )
        if not operations:
            raise NumericExecutionPolicyError(
                f"Required operations for {domain} must not be empty."
            )
        result[domain] = operations
    return result


def _domain_name(value: NumericDomain | str) -> str:
    result = value.value if isinstance(value, NumericDomain) else value
    if result not in _DOMAIN_ORDER:
        raise NumericExecutionPolicyError(f"Unknown numeric domain {value!r}.")
    return str(result)


def _inventory_entries(
    inventories: Sequence[ModelNumericInventory],
    *,
    parameters_preferred: bool,
) -> list[dict[str, Any]]:
    entries = [
        dict(entry)
        for inventory in inventories
        for entry in inventory.as_dict()["entries"]
        if entry["floating"]
    ]
    if parameters_preferred:
        parameters = [entry for entry in entries if entry["kind"] == "parameter"]
        if parameters:
            entries = parameters
    if not entries:
        raise NumericExecutionPolicyError("Numeric placement has no floating inventory.")
    return entries


def _summary(values: set[str]) -> str:
    if not values:
        raise NumericExecutionPolicyError("Cannot summarize an empty placement.")
    return next(iter(values)) if len(values) == 1 else _MIXED_SENTINEL


def _model_placement(
    domain: str,
    inventories: Sequence[ModelNumericInventory],
    *,
    requested_device: Any | None,
    requested_dtype: Any | None,
    operations: Sequence[str],
    environment: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]],
    storage: bool = False,
    target_native: bool = False,
) -> dict[str, Any]:
    entries = _inventory_entries(inventories, parameters_preferred=not storage)
    inventory_sha256s = tuple(
        sorted(inventory.inventory_sha256 for inventory in inventories)
    )
    native_pairs = sorted({(entry["device"], entry["dtype"]) for entry in entries})
    if storage:
        requested_device_name = _STORAGE_SENTINEL
        requested_dtype_name = _STORAGE_SENTINEL
        resolved_device = _summary({device for device, _dtype in native_pairs})
        resolved_dtype = _summary({dtype for _device, dtype in native_pairs})
        concrete_pairs = native_pairs
        reason = "authenticated-model-storage-inventory"
    else:
        native_sentinel = _TARGET_NATIVE_SENTINEL if target_native else _MODEL_NATIVE_SENTINEL
        requested_device_name = (
            native_sentinel
            if requested_device is None
            else _normalize_device(requested_device)
        )
        requested_dtype_name = (
            native_sentinel
            if requested_dtype is None
            else normalize_numeric_dtype(requested_dtype)
        )
        devices = {device for device, _dtype in native_pairs}
        dtypes = {dtype for _device, dtype in native_pairs}
        resolved_device = (
            _summary(devices)
            if requested_device is None and len(devices) == 1
            else requested_device_name
        )
        resolved_dtype = (
            _summary(dtypes)
            if requested_dtype is None and len(dtypes) == 1
            else requested_dtype_name
        )
        concrete_pairs = sorted(
            {
                (
                    device if requested_device is None else requested_device_name,
                    dtype if requested_dtype is None else requested_dtype_name,
                )
                for device, dtype in native_pairs
            }
        )
        reason = (
            "explicit-model-execution-placement"
            if requested_device is not None or requested_dtype is not None
            else "authenticated-model-native-placement"
        )
    capabilities: list[dict[str, Any]] = []
    for device, dtype in concrete_pairs:
        concrete = NumericConcretePlacement(device, dtype)
        for operation in operations:
            capabilities.append(
                _resolve_capability(
                    operation,
                    concrete,
                    environment=environment,
                    overrides=overrides,
                )
            )
    unsupported = [record for record in capabilities if not record["supported"]]
    if unsupported:
        keys = ", ".join(record["key"] for record in unsupported)
        raise NumericExecutionPolicyError(
            f"{domain} has unsupported numeric capabilities: {keys}."
        )
    return {
        "domain": domain,
        "requested_device": requested_device_name,
        "requested_dtype": requested_dtype_name,
        "resolved_device": resolved_device,
        "resolved_dtype": resolved_dtype,
        "resolution_reason": reason,
        "resolution_fallback_used": False,
        "inventory_sha256s": list(inventory_sha256s),
        "execution_candidates": [],
        "capability_records": capabilities,
    }


def _concrete_placement(
    domain: str,
    *,
    requested_device: Any,
    requested_dtype: Any,
    operations: Sequence[str],
    environment: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]],
    allow_cpu_fp64_fallback: bool,
) -> dict[str, Any]:
    requested = NumericConcretePlacement(
        _normalize_device(requested_device),
        normalize_numeric_dtype(requested_dtype),
    )
    candidates = [requested]
    if domain in _FALLBACK_DOMAINS and allow_cpu_fp64_fallback:
        fallback = NumericConcretePlacement("cpu", "float64")
        if fallback != requested:
            candidates.append(fallback)
    capability_records: list[dict[str, Any]] = []
    supported_candidates: list[NumericConcretePlacement] = []
    candidate_support: list[bool] = []
    for candidate in candidates:
        records = [
            _resolve_capability(
                operation,
                candidate,
                environment=environment,
                overrides=overrides,
            )
            for operation in operations
        ]
        capability_records.extend(records)
        supported = all(record["supported"] for record in records)
        candidate_support.append(supported)
        if supported:
            supported_candidates.append(candidate)
    if not supported_candidates:
        keys = ", ".join(
            record["key"]
            for record in capability_records
            if not record["supported"]
        )
        raise NumericExecutionPolicyError(
            f"No supported {domain} numeric placement; unavailable: {keys}."
        )
    selected_index = next(index for index, item in enumerate(candidate_support) if item)
    selected = candidates[selected_index]
    execution_candidates = [
        candidate
        for index, candidate in enumerate(candidates)
        if index >= selected_index and candidate_support[index]
    ]
    return {
        "domain": domain,
        "requested_device": requested.device,
        "requested_dtype": requested.dtype,
        "resolved_device": selected.device,
        "resolved_dtype": selected.dtype,
        "resolution_reason": (
            "requested-placement-supported"
            if selected_index == 0
            else "cpu-float64-capability-fallback"
        ),
        "resolution_fallback_used": selected_index != 0,
        "inventory_sha256s": [],
        "execution_candidates": [
            candidate.as_dict() for candidate in execution_candidates
        ],
        "capability_records": capability_records,
    }


_PLACEMENT_KEYS = frozenset(
    {
        "domain",
        "requested_device",
        "requested_dtype",
        "resolved_device",
        "resolved_dtype",
        "resolution_reason",
        "resolution_fallback_used",
        "inventory_sha256s",
        "execution_candidates",
        "capability_records",
    }
)
_CAPABILITY_RECORD_KEYS = frozenset(
    {
        "key",
        "operation",
        "device",
        "dtype",
        "supported",
        "source",
        "reason",
        "runtime_preflight_required",
    }
)
_CAPABILITY_ENVIRONMENT_KEYS = frozenset(
    {"cuda_runtime_available", "cuda_device_count", "cuda_probe_failed"}
)
_DETERMINISTIC_SETTING_KEYS = frozenset(
    {
        "deterministic_algorithms_required",
        "autocast_enabled",
        "cuda_tf32_allowed",
        "cudnn_benchmark_allowed",
        "cpu_thread_count",
    }
)
_FALLBACK_POLICY_KEYS = frozenset(
    {
        "enabled",
        "domains",
        "ordered_chain",
        "eligible_failure_classes",
        "ineligible_failure_classes",
        "raw_runtime_error_message_classification",
        "fallback_requires_immutable_prepublication_inputs",
    }
)
_POLICY_KEYS = frozenset(
    {
        "schema",
        "supported_float_dtypes",
        "domains",
        "source_inventories",
        "target_inventory",
        "placements",
        "required_operations",
        "capability_environment",
        "deterministic_settings",
        "fallback_policy",
        "caller_must_apply_placements",
        "cast_ledger_required_for_lossy_boundaries",
        "ordinary_dtype_cast_is_quantization",
        "operation_audit",
        "policy_sha256",
    }
)


def _validate_placement(value: Mapping[str, Any], expected_domain: str) -> dict[str, Any]:
    result = _plain_json(value)
    if set(result) != _PLACEMENT_KEYS or result["domain"] != expected_domain:
        raise NumericExecutionPolicyError(
            f"Numeric placement {expected_domain!r} is incomplete or invalid."
        )
    if not isinstance(result["resolution_fallback_used"], bool):
        raise NumericExecutionPolicyError("Placement fallback flag must be boolean.")
    for key in ("requested_device", "requested_dtype", "resolved_device", "resolved_dtype"):
        if not isinstance(result[key], str) or not result[key]:
            raise NumericExecutionPolicyError(f"Placement {key} is invalid.")
        if key.endswith("device") and result[key] not in _PLACEMENT_SENTINELS:
            _normalize_device(result[key])
        if key.endswith("dtype") and result[key] not in _PLACEMENT_SENTINELS:
            normalize_numeric_dtype(result[key])
    if not isinstance(result["resolution_reason"], str) or not result[
        "resolution_reason"
    ].strip():
        raise NumericExecutionPolicyError("Placement resolution reason is invalid.")
    if not isinstance(result["inventory_sha256s"], list) or result[
        "inventory_sha256s"
    ] != sorted(set(result["inventory_sha256s"])):
        raise NumericExecutionPolicyError("Placement inventory digests are invalid.")
    for digest in result["inventory_sha256s"]:
        _digest(digest, "placement inventory digest")
    if not isinstance(result["execution_candidates"], list):
        raise NumericExecutionPolicyError("Execution candidates must be a list.")
    for candidate in result["execution_candidates"]:
        if not isinstance(candidate, Mapping) or set(candidate) != {"device", "dtype"}:
            raise NumericExecutionPolicyError("Execution candidate is invalid.")
        NumericConcretePlacement(candidate["device"], candidate["dtype"])
    if not isinstance(result["capability_records"], list):
        raise NumericExecutionPolicyError("Capability records must be a list.")
    for record in result["capability_records"]:
        if not isinstance(record, Mapping) or set(record) != _CAPABILITY_RECORD_KEYS:
            raise NumericExecutionPolicyError("Numeric capability record is invalid.")
        operation = _identifier(record["operation"], "capability operation")
        concrete = NumericConcretePlacement(record["device"], record["dtype"])
        if record["key"] != numeric_capability_key(
            operation, concrete.device, concrete.dtype
        ):
            raise NumericExecutionPolicyError("Numeric capability key is invalid.")
        if not isinstance(record["supported"], bool) or not isinstance(
            record["runtime_preflight_required"], bool
        ):
            raise NumericExecutionPolicyError("Numeric capability flags are invalid.")
        for key in ("source", "reason"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise NumericExecutionPolicyError(
                    f"Numeric capability {key} is invalid."
                )
    if expected_domain in _FALLBACK_DOMAINS:
        if not result["execution_candidates"]:
            raise NumericExecutionPolicyError(
                f"Fallback domain {expected_domain} has no execution candidates."
            )
        first = result["execution_candidates"][0]
        if (
            first["device"] != result["resolved_device"]
            or first["dtype"] != result["resolved_dtype"]
        ):
            raise NumericExecutionPolicyError(
                "Resolved fallback placement is not the first execution candidate."
            )
        if len(result["execution_candidates"]) > 2 or any(
            candidate != {"device": "cpu", "dtype": "float64"}
            for candidate in result["execution_candidates"][1:]
        ):
            raise NumericExecutionPolicyError(
                "Runtime fallback chain must end with only CPU FP64."
            )
        if result["resolution_fallback_used"] and (
            result["resolved_device"] != "cpu"
            or result["resolved_dtype"] != "float64"
        ):
            raise NumericExecutionPolicyError(
                "Resolution fallback must resolve to CPU FP64."
            )
    elif result["resolution_fallback_used"]:
        raise NumericExecutionPolicyError(
            "Only solve/factorization domains may resolve through fallback."
        )
    return result


def _validate_policy_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _plain_json(value)
    if set(receipt) != _POLICY_KEYS:
        raise NumericExecutionPolicyError(
            "Numeric execution policy fields are incomplete or unknown."
        )
    if receipt["schema"] != NUMERIC_EXECUTION_POLICY_SCHEMA:
        raise NumericExecutionPolicyError("Numeric execution policy schema is invalid.")
    if receipt["supported_float_dtypes"] != list(SUPPORTED_FLOAT_DTYPES):
        raise NumericExecutionPolicyError("Supported numeric dtype set is invalid.")
    if receipt["domains"] != list(_DOMAIN_ORDER):
        raise NumericExecutionPolicyError("Numeric domain set is invalid.")
    sources = receipt["source_inventories"]
    if not isinstance(sources, Mapping) or not sources:
        raise NumericExecutionPolicyError("Numeric policy has no source inventories.")
    source_inventory_digests: list[str] = []
    for name, inventory in sources.items():
        _identifier(name, "source model name")
        authenticated = ModelNumericInventory(inventory)
        if authenticated.role != f"source:{name}":
            raise NumericExecutionPolicyError(
                "Source numeric inventory role does not match its model binding."
            )
        source_inventory_digests.append(authenticated.inventory_sha256)
    target_inventory = ModelNumericInventory(receipt["target_inventory"])
    if target_inventory.role != "target":
        raise NumericExecutionPolicyError("Target numeric inventory role is invalid.")
    placements = receipt["placements"]
    if not isinstance(placements, Mapping) or set(placements) != set(_DOMAIN_ORDER):
        raise NumericExecutionPolicyError("Numeric policy placements are incomplete.")
    for domain in _DOMAIN_ORDER:
        placements[domain] = _validate_placement(placements[domain], domain)
    expected_inventory_bindings = {
        NumericDomain.SOURCE_STORAGE.value: sorted(source_inventory_digests),
        NumericDomain.SOURCE_FORWARD.value: sorted(source_inventory_digests),
        NumericDomain.TARGET_STORAGE.value: [target_inventory.inventory_sha256],
        NumericDomain.TARGET_FORWARD.value: [target_inventory.inventory_sha256],
        NumericDomain.PUBLICATION.value: (
            [target_inventory.inventory_sha256]
            if placements[NumericDomain.PUBLICATION.value]["inventory_sha256s"]
            else []
        ),
        NumericDomain.TARGET_VERIFICATION.value: (
            [target_inventory.inventory_sha256]
            if placements[NumericDomain.TARGET_VERIFICATION.value][
                "inventory_sha256s"
            ]
            else []
        ),
    }
    for domain, expected in expected_inventory_bindings.items():
        if placements[domain]["inventory_sha256s"] != expected:
            raise NumericExecutionPolicyError(
                f"Numeric placement {domain} is bound to the wrong inventory."
            )
    for domain in (
        NumericDomain.ACCUMULATION.value,
        NumericDomain.SOLVE.value,
        NumericDomain.FACTORIZATION.value,
        NumericDomain.VERIFICATION.value,
    ):
        if placements[domain]["inventory_sha256s"]:
            raise NumericExecutionPolicyError(
                f"Concrete numeric placement {domain} cannot bind model inventory."
            )
    required = receipt["required_operations"]
    if not isinstance(required, Mapping) or set(required) != set(_DOMAIN_ORDER):
        raise NumericExecutionPolicyError("Required operation map is incomplete.")
    for domain, domain_operations in required.items():
        if (
            not isinstance(domain_operations, list)
            or not domain_operations
            or domain_operations != sorted(set(domain_operations))
        ):
            raise NumericExecutionPolicyError(
                f"Required operations for {domain} are invalid."
            )
        for operation in domain_operations:
            _identifier(operation, f"required {domain} operation")
        for capability in placements[domain]["capability_records"]:
            if capability["operation"] not in domain_operations:
                raise NumericExecutionPolicyError(
                    f"Placement {domain} has an unauthenticated capability operation."
                )
    capability_environment = receipt["capability_environment"]
    if (
        not isinstance(capability_environment, Mapping)
        or set(capability_environment) != _CAPABILITY_ENVIRONMENT_KEYS
        or not isinstance(capability_environment["cuda_runtime_available"], bool)
        or not isinstance(capability_environment["cuda_probe_failed"], bool)
    ):
        raise NumericExecutionPolicyError("Numeric capability environment is invalid.")
    _nonnegative_int(
        capability_environment["cuda_device_count"], "cuda_device_count"
    )
    deterministic = receipt["deterministic_settings"]
    if not isinstance(deterministic, Mapping) or set(
        deterministic
    ) != _DETERMINISTIC_SETTING_KEYS or deterministic != {
        "deterministic_algorithms_required": True,
        "autocast_enabled": False,
        "cuda_tf32_allowed": False,
        "cudnn_benchmark_allowed": False,
        "cpu_thread_count": deterministic.get("cpu_thread_count"),
    }:
        raise NumericExecutionPolicyError("Deterministic numeric settings are invalid.")
    if (
        isinstance(deterministic["cpu_thread_count"], bool)
        or not isinstance(deterministic["cpu_thread_count"], int)
        or deterministic["cpu_thread_count"] <= 0
    ):
        raise NumericExecutionPolicyError("Numeric CPU thread policy is invalid.")
    fallback = receipt["fallback_policy"]
    if not isinstance(fallback, Mapping) or set(fallback) != _FALLBACK_POLICY_KEYS:
        raise NumericExecutionPolicyError("Numeric fallback policy fields are invalid.")
    if fallback.get("eligible_failure_classes") != [
        "unsupported-backend",
        "device-loss",
        "prepublication-out-of-memory",
    ] or fallback.get("ineligible_failure_classes") != [
        "nonfinite",
        "rank-failure",
        "authentication-failure",
        "shape-failure",
        "untyped-runtime-error",
    ]:
        raise NumericExecutionPolicyError("Numeric fallback class policy is invalid.")
    if (
        not isinstance(fallback["enabled"], bool)
        or fallback["domains"] != sorted(_FALLBACK_DOMAINS)
        or fallback["ordered_chain"]
        != (
            ["requested-placement", "cpu:float64"]
            if fallback["enabled"]
            else ["requested-placement"]
        )
        or fallback["raw_runtime_error_message_classification"] is not False
        or fallback["fallback_requires_immutable_prepublication_inputs"] is not True
    ):
        raise NumericExecutionPolicyError("Numeric fallback behavior is invalid.")
    if not isinstance(receipt["cast_ledger_required_for_lossy_boundaries"], bool):
        raise NumericExecutionPolicyError("Numeric cast ledger policy is invalid.")
    if receipt["caller_must_apply_placements"] is not True:
        raise NumericExecutionPolicyError("Caller placement obligation is invalid.")
    if receipt["ordinary_dtype_cast_is_quantization"] is not False:
        raise NumericExecutionPolicyError("Numeric policy cannot classify casts as quantization.")
    _validate_operation_audit(receipt["operation_audit"])
    supplied = _digest(receipt["policy_sha256"], "policy_sha256")
    body = {key: item for key, item in receipt.items() if key != "policy_sha256"}
    if supplied != canonical_sha256(body):
        raise NumericExecutionPolicyError("Numeric execution policy seal is invalid.")
    return receipt


@dataclass(frozen=True, slots=True)
class NumericExecutionPolicy:
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt",
            _deep_freeze(_validate_policy_receipt(self.receipt)),
        )

    @property
    def policy_sha256(self) -> str:
        return str(self.receipt["policy_sha256"])

    def placement(self, domain: NumericDomain | str) -> NumericDomainPlacement:
        name = _domain_name(domain)
        value = self.receipt["placements"][name]
        return NumericDomainPlacement(
            domain=name,
            requested_device=str(value["requested_device"]),
            requested_dtype=str(value["requested_dtype"]),
            resolved_device=str(value["resolved_device"]),
            resolved_dtype=str(value["resolved_dtype"]),
            resolution_reason=str(value["resolution_reason"]),
            resolution_fallback_used=bool(value["resolution_fallback_used"]),
            inventory_sha256s=tuple(value["inventory_sha256s"]),
            execution_candidates=tuple(
                NumericConcretePlacement(item["device"], item["dtype"])
                for item in value["execution_candidates"]
            ),
            capability_records=tuple(
                _deep_freeze(_plain_json(item))
                for item in value["capability_records"]
            ),
        )

    def required_operations(self, domain: NumericDomain | str) -> tuple[str, ...]:
        return tuple(self.receipt["required_operations"][_domain_name(domain)])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def resolve_numeric_execution_policy(
    *,
    source_models: Mapping[str, Any],
    target_model: Any,
    request: NumericExecutionRequest | None = None,
    required_operations: Mapping[Any, Sequence[str]] | None = None,
    capability_overrides: Mapping[Any, Any] | None = None,
) -> NumericExecutionPolicy:
    """Resolve one immutable policy without model execution or mutation."""

    if not isinstance(source_models, Mapping) or not source_models:
        raise NumericExecutionPolicyError("source_models must be a nonempty mapping.")
    active_request = request or NumericExecutionRequest()
    if not isinstance(active_request, NumericExecutionRequest):
        raise NumericExecutionPolicyError("request must be NumericExecutionRequest.")
    operations = _normalized_required_operations(required_operations)
    overrides = _normalize_capability_overrides(capability_overrides)
    environment = _cuda_environment()
    sources: dict[str, ModelNumericInventory] = {}
    for raw_name, model in source_models.items():
        name = _identifier(raw_name, "source model name")
        if name in sources:
            raise NumericExecutionPolicyError(f"Duplicate source model name {name!r}.")
        sources[name] = inventory_model_numeric_state(model, role=f"source:{name}")
    target = inventory_model_numeric_state(target_model, role="target")
    source_values = tuple(sources.values())
    target_values = (target,)

    placements: dict[str, dict[str, Any]] = {}
    placements[NumericDomain.SOURCE_STORAGE.value] = _model_placement(
        NumericDomain.SOURCE_STORAGE.value,
        source_values,
        requested_device=None,
        requested_dtype=None,
        operations=operations[NumericDomain.SOURCE_STORAGE.value],
        environment=environment,
        overrides=overrides,
        storage=True,
    )
    placements[NumericDomain.TARGET_STORAGE.value] = _model_placement(
        NumericDomain.TARGET_STORAGE.value,
        target_values,
        requested_device=None,
        requested_dtype=None,
        operations=operations[NumericDomain.TARGET_STORAGE.value],
        environment=environment,
        overrides=overrides,
        storage=True,
    )
    placements[NumericDomain.SOURCE_FORWARD.value] = _model_placement(
        NumericDomain.SOURCE_FORWARD.value,
        source_values,
        requested_device=active_request.source_forward_device,
        requested_dtype=active_request.source_forward_dtype,
        operations=operations[NumericDomain.SOURCE_FORWARD.value],
        environment=environment,
        overrides=overrides,
    )
    placements[NumericDomain.TARGET_FORWARD.value] = _model_placement(
        NumericDomain.TARGET_FORWARD.value,
        target_values,
        requested_device=active_request.target_forward_device,
        requested_dtype=active_request.target_forward_dtype,
        operations=operations[NumericDomain.TARGET_FORWARD.value],
        environment=environment,
        overrides=overrides,
        target_native=True,
    )
    target_forward = placements[NumericDomain.TARGET_FORWARD.value]
    accumulation_device = active_request.accumulation_device
    if accumulation_device is None:
        resolved = target_forward["resolved_device"]
        accumulation_device = (
            resolved if resolved not in _PLACEMENT_SENTINELS else "cpu"
        )
    placements[NumericDomain.ACCUMULATION.value] = _concrete_placement(
        NumericDomain.ACCUMULATION.value,
        requested_device=accumulation_device,
        requested_dtype=active_request.accumulation_dtype,
        operations=operations[NumericDomain.ACCUMULATION.value],
        environment=environment,
        overrides=overrides,
        allow_cpu_fp64_fallback=False,
    )
    placements[NumericDomain.SOLVE.value] = _concrete_placement(
        NumericDomain.SOLVE.value,
        requested_device=active_request.solve_device,
        requested_dtype=active_request.solve_dtype,
        operations=operations[NumericDomain.SOLVE.value],
        environment=environment,
        overrides=overrides,
        allow_cpu_fp64_fallback=active_request.allow_cpu_fp64_fallback,
    )
    placements[NumericDomain.FACTORIZATION.value] = _concrete_placement(
        NumericDomain.FACTORIZATION.value,
        requested_device=(
            active_request.solve_device
            if active_request.factorization_device is None
            else active_request.factorization_device
        ),
        requested_dtype=(
            active_request.solve_dtype
            if active_request.factorization_dtype is None
            else active_request.factorization_dtype
        ),
        operations=operations[NumericDomain.FACTORIZATION.value],
        environment=environment,
        overrides=overrides,
        allow_cpu_fp64_fallback=active_request.allow_cpu_fp64_fallback,
    )
    if (
        active_request.publication_device is None
        or active_request.publication_dtype is None
    ):
        if not (
            active_request.publication_device is None
            and active_request.publication_dtype is None
        ):
            raise NumericExecutionPolicyError(
                "Publication device and dtype must both be explicit or both target-native."
            )
        placements[NumericDomain.PUBLICATION.value] = _model_placement(
            NumericDomain.PUBLICATION.value,
            target_values,
            requested_device=None,
            requested_dtype=None,
            operations=operations[NumericDomain.PUBLICATION.value],
            environment=environment,
            overrides=overrides,
            target_native=True,
        )
    else:
        placements[NumericDomain.PUBLICATION.value] = _concrete_placement(
            NumericDomain.PUBLICATION.value,
            requested_device=active_request.publication_device,
            requested_dtype=active_request.publication_dtype,
            operations=operations[NumericDomain.PUBLICATION.value],
            environment=environment,
            overrides=overrides,
            allow_cpu_fp64_fallback=False,
        )
    placements[NumericDomain.VERIFICATION.value] = _concrete_placement(
        NumericDomain.VERIFICATION.value,
        requested_device=active_request.verification_device,
        requested_dtype=active_request.verification_dtype,
        operations=operations[NumericDomain.VERIFICATION.value],
        environment=environment,
        overrides=overrides,
        allow_cpu_fp64_fallback=False,
    )
    target_verification_device = active_request.target_verification_device
    target_verification_dtype = active_request.target_verification_dtype
    if target_verification_device is None and target_verification_dtype is None:
        placements[NumericDomain.TARGET_VERIFICATION.value] = _model_placement(
            NumericDomain.TARGET_VERIFICATION.value,
            target_values,
            requested_device=active_request.target_forward_device,
            requested_dtype=active_request.target_forward_dtype,
            operations=operations[NumericDomain.TARGET_VERIFICATION.value],
            environment=environment,
            overrides=overrides,
            target_native=True,
        )
    elif target_verification_device is None or target_verification_dtype is None:
        raise NumericExecutionPolicyError(
            "Target verification device and dtype must both be explicit or target-native."
        )
    else:
        placements[NumericDomain.TARGET_VERIFICATION.value] = _concrete_placement(
            NumericDomain.TARGET_VERIFICATION.value,
            requested_device=target_verification_device,
            requested_dtype=target_verification_dtype,
            operations=operations[NumericDomain.TARGET_VERIFICATION.value],
            environment=environment,
            overrides=overrides,
            allow_cpu_fp64_fallback=False,
        )

    fallback_chain = ["requested-placement"]
    if active_request.allow_cpu_fp64_fallback:
        fallback_chain.append("cpu:float64")
    body: dict[str, Any] = {
        "schema": NUMERIC_EXECUTION_POLICY_SCHEMA,
        "supported_float_dtypes": list(SUPPORTED_FLOAT_DTYPES),
        "domains": list(_DOMAIN_ORDER),
        "source_inventories": {
            name: inventory.as_dict() for name, inventory in sorted(sources.items())
        },
        "target_inventory": target.as_dict(),
        "placements": placements,
        "required_operations": {
            domain: list(operations[domain]) for domain in _DOMAIN_ORDER
        },
        "capability_environment": dict(environment),
        "deterministic_settings": {
            "deterministic_algorithms_required": True,
            "autocast_enabled": False,
            "cuda_tf32_allowed": False,
            "cudnn_benchmark_allowed": False,
            "cpu_thread_count": active_request.cpu_thread_count,
        },
        "fallback_policy": {
            "enabled": active_request.allow_cpu_fp64_fallback,
            "domains": sorted(_FALLBACK_DOMAINS),
            "ordered_chain": fallback_chain,
            "eligible_failure_classes": [
                "unsupported-backend",
                "device-loss",
                "prepublication-out-of-memory",
            ],
            "ineligible_failure_classes": [
                "nonfinite",
                "rank-failure",
                "authentication-failure",
                "shape-failure",
                "untyped-runtime-error",
            ],
            "raw_runtime_error_message_classification": False,
            "fallback_requires_immutable_prepublication_inputs": True,
        },
        "caller_must_apply_placements": True,
        "cast_ledger_required_for_lossy_boundaries": True,
        "ordinary_dtype_cast_is_quantization": False,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    return NumericExecutionPolicy(
        {**body, "policy_sha256": canonical_sha256(body)}
    )


def require_finite_tensor(value: Any, label: str = "numeric tensor") -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise NumericShapeError(f"{label} must be one floating tensor.")
    if not bool(torch.isfinite(value).all().item()):
        raise NumericNonFiniteError(f"{label} contains NaN or infinity.")


def require_tensor_shape(
    value: Any,
    expected_shape: Sequence[int],
    label: str = "numeric tensor",
) -> None:
    if not isinstance(value, torch.Tensor):
        raise NumericShapeError(f"{label} must be a tensor.")
    expected = tuple(int(item) for item in expected_shape)
    if tuple(int(item) for item in value.shape) != expected:
        raise NumericShapeError(
            f"{label} shape {tuple(value.shape)} does not equal {expected}."
        )


def require_authentication(condition: Any, label: str = "numeric evidence") -> None:
    if condition is not True:
        raise NumericAuthenticationError(f"{label} failed authentication.")


def require_full_rank(observed_rank: Any, expected_rank: Any, label: str = "matrix") -> None:
    if (
        isinstance(observed_rank, bool)
        or not isinstance(observed_rank, int)
        or isinstance(expected_rank, bool)
        or not isinstance(expected_rank, int)
        or observed_rank < 0
        or expected_rank < 0
    ):
        raise NumericRankError(f"{label} rank metadata is invalid.")
    if observed_rank < expected_rank:
        raise NumericRankError(
            f"{label} rank {observed_rank} is below required rank {expected_rank}."
        )


def _require_finite_result(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        require_finite_tensor(value, label)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_result(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite_result(item, f"{label}[{index}]")


T = TypeVar("T")


_EXECUTION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "policy_sha256",
        "domain",
        "operation",
        "requested_device",
        "requested_dtype",
        "resolved_device",
        "resolved_dtype",
        "resolution_fallback_used",
        "runtime_fallback_used",
        "fallback_used",
        "attempts",
        "status",
        "operation_audit",
        "execution_sha256",
    }
)
_EXECUTION_ATTEMPT_KEYS = frozenset(
    {
        "attempt_index",
        "device",
        "dtype",
        "status",
        "failure_class",
        "exception_type",
    }
)


def verify_numeric_execution_receipt(
    value: Mapping[str, Any],
    *,
    expected_policy_sha256: str | None = None,
) -> Mapping[str, Any]:
    receipt = _plain_json(value)
    if set(receipt) != _EXECUTION_RECEIPT_KEYS:
        raise NumericExecutionPolicyError(
            "Numeric execution receipt fields are incomplete or unknown."
        )
    if receipt["schema"] != NUMERIC_EXECUTION_RECEIPT_SCHEMA:
        raise NumericExecutionPolicyError("Numeric execution receipt schema is invalid.")
    policy_sha = _digest(receipt["policy_sha256"], "execution policy_sha256")
    if expected_policy_sha256 is not None and policy_sha != _digest(
        expected_policy_sha256, "expected policy_sha256"
    ):
        raise NumericExecutionPolicyError(
            "Numeric execution receipt belongs to another policy."
        )
    domain = _domain_name(receipt["domain"])
    if domain not in _FALLBACK_DOMAINS:
        raise NumericExecutionPolicyError("Execution receipt domain cannot fallback.")
    _identifier(receipt["operation"], "execution operation")
    NumericConcretePlacement(receipt["requested_device"], receipt["requested_dtype"])
    for key in (
        "resolution_fallback_used",
        "runtime_fallback_used",
        "fallback_used",
    ):
        if not isinstance(receipt[key], bool):
            raise NumericExecutionPolicyError(f"Execution receipt {key} is invalid.")
    attempts = receipt["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise NumericExecutionPolicyError("Execution receipt has no attempts.")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != _EXECUTION_ATTEMPT_KEYS:
            raise NumericExecutionPolicyError("Numeric execution attempt is invalid.")
        if attempt["attempt_index"] != index:
            raise NumericExecutionPolicyError("Numeric execution attempt order is invalid.")
        NumericConcretePlacement(attempt["device"], attempt["dtype"])
        if attempt["status"] == "succeeded":
            if attempt["failure_class"] is not None or attempt["exception_type"] is not None:
                raise NumericExecutionPolicyError(
                    "Successful numeric execution attempt carries a failure."
                )
        elif attempt["status"] == "eligible-backend-failure":
            if attempt["failure_class"] not in {
                "unsupported-backend",
                "device-loss",
                "prepublication-out-of-memory",
            } or not isinstance(attempt["exception_type"], str):
                raise NumericExecutionPolicyError(
                    "Numeric execution attempt has an ineligible failure."
                )
        else:
            raise NumericExecutionPolicyError("Numeric execution attempt status is invalid.")
    if receipt["runtime_fallback_used"] != (len(attempts) > 1):
        raise NumericExecutionPolicyError("Runtime fallback accounting is invalid.")
    if receipt["fallback_used"] != (
        receipt["resolution_fallback_used"] or receipt["runtime_fallback_used"]
    ):
        raise NumericExecutionPolicyError("Combined fallback accounting is invalid.")
    status = receipt["status"]
    if status == "succeeded":
        last = attempts[-1]
        if last["status"] != "succeeded":
            raise NumericExecutionPolicyError("Successful execution has no final success.")
        concrete = NumericConcretePlacement(
            receipt["resolved_device"], receipt["resolved_dtype"]
        )
        if concrete.device != last["device"] or concrete.dtype != last["dtype"]:
            raise NumericExecutionPolicyError("Execution resolution is inconsistent.")
    elif status == "failed":
        if (
            receipt["resolved_device"] is not None
            or receipt["resolved_dtype"] is not None
            or attempts[-1]["status"] != "eligible-backend-failure"
        ):
            raise NumericExecutionPolicyError("Failed execution resolution is invalid.")
    else:
        raise NumericExecutionPolicyError("Numeric execution status is invalid.")
    _validate_operation_audit(receipt["operation_audit"])
    supplied = _digest(receipt["execution_sha256"], "execution_sha256")
    body = {key: item for key, item in receipt.items() if key != "execution_sha256"}
    if supplied != canonical_sha256(body):
        raise NumericExecutionPolicyError("Numeric execution receipt seal is invalid.")
    return _deep_freeze(receipt)


@dataclass(frozen=True, slots=True)
class NumericExecutionResult(Generic[T]):
    value: T
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt",
            verify_numeric_execution_receipt(self.receipt),
        )

    def receipt_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def _execution_receipt(
    *,
    policy: NumericExecutionPolicy,
    placement: NumericDomainPlacement,
    operation: str,
    attempts: Sequence[Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    body = {
        "schema": NUMERIC_EXECUTION_RECEIPT_SCHEMA,
        "policy_sha256": policy.policy_sha256,
        "domain": placement.domain,
        "operation": operation,
        "requested_device": placement.requested_device,
        "requested_dtype": placement.requested_dtype,
        "resolved_device": (
            None if status != "succeeded" else attempts[-1]["device"]
        ),
        "resolved_dtype": (
            None if status != "succeeded" else attempts[-1]["dtype"]
        ),
        "resolution_fallback_used": placement.resolution_fallback_used,
        "runtime_fallback_used": len(attempts) > 1,
        "fallback_used": placement.resolution_fallback_used or len(attempts) > 1,
        "attempts": [_plain_json(item) for item in attempts],
        "status": status,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    return {**body, "execution_sha256": canonical_sha256(body)}


def run_with_numeric_fallback(
    policy: NumericExecutionPolicy,
    *,
    domain: NumericDomain | str,
    operation: str,
    execute: Callable[[NumericConcretePlacement], T],
    validate_finite_result: bool = True,
) -> NumericExecutionResult[T]:
    """Execute one pure analytic operation with the sealed fallback chain.

    ``execute`` must operate on immutable inputs and must not publish model
    state.  Callers wrap raw backend failures in one of the three explicitly
    eligible exception types only when that precondition is true.
    """

    if not isinstance(policy, NumericExecutionPolicy):
        raise NumericExecutionPolicyError("policy must be NumericExecutionPolicy.")
    name = _domain_name(domain)
    if name not in _FALLBACK_DOMAINS:
        raise NumericExecutionPolicyError(
            "Runtime numeric fallback is restricted to solve and factorization."
        )
    operation = _identifier(operation, "numeric operation")
    if operation not in policy.required_operations(name):
        raise NumericExecutionPolicyError(
            f"Operation {operation!r} is not authenticated for {name}."
        )
    if not callable(execute):
        raise NumericExecutionPolicyError("execute must be callable.")
    if not isinstance(validate_finite_result, bool):
        raise NumericExecutionPolicyError("validate_finite_result must be boolean.")
    placement = policy.placement(name)
    candidates = placement.execution_candidates
    if not candidates:
        raise NumericExecutionPolicyError(
            f"Numeric placement {name} has no executable candidates."
        )
    attempts: list[dict[str, Any]] = []
    eligible = (
        NumericUnsupportedBackendError,
        NumericDeviceLossError,
        NumericPrepublicationOutOfMemoryError,
    )
    for index, candidate in enumerate(candidates):
        try:
            result = execute(candidate)
        except eligible as exc:
            attempts.append(
                {
                    "attempt_index": index,
                    **candidate.as_dict(),
                    "status": "eligible-backend-failure",
                    "failure_class": exc.reason_code,
                    "exception_type": type(exc).__name__,
                }
            )
            if index + 1 < len(candidates):
                continue
            receipt = _execution_receipt(
                policy=policy,
                placement=placement,
                operation=operation,
                attempts=attempts,
                status="failed",
            )
            raise NumericFallbackExhaustedError(
                f"Authenticated {name} fallback candidates were exhausted.",
                receipt,
            ) from exc
        if validate_finite_result:
            _require_finite_result(result, f"{name} result")
        attempts.append(
            {
                "attempt_index": index,
                **candidate.as_dict(),
                "status": "succeeded",
                "failure_class": None,
                "exception_type": None,
            }
        )
        receipt = _execution_receipt(
            policy=policy,
            placement=placement,
            operation=operation,
            attempts=attempts,
            status="succeeded",
        )
        return NumericExecutionResult(result, receipt)
    raise AssertionError("Numeric fallback candidate loop ended unexpectedly.")


def _tensor_sha256(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    metadata = {
        "dtype": _dtype_inventory_name(detached.dtype),
        "shape": [int(item) for item in detached.shape],
    }
    raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(raw)
    return digest.hexdigest()


_CAST_LEDGER_KEYS = frozenset(
    {
        "schema",
        "policy_sha256",
        "entries",
        "entry_count",
        "source_bytes",
        "target_bytes",
        "lossy_cast_count",
        "changed_coordinate_count",
        "underflow_coordinate_count",
        "ordinary_dtype_cast_is_quantization",
        "operation_audit",
        "ledger_sha256",
    }
)
_CAST_LEDGER_ENTRY_KEYS = frozenset(
    {
        "entry_index",
        "tensor_role",
        "domain",
        "shape",
        "numel",
        "source_dtype",
        "source_device",
        "target_dtype",
        "target_device",
        "source_bytes",
        "target_bytes",
        "source_sha256",
        "target_sha256",
        "dtype_changed",
        "device_changed",
        "lossy_cast",
        "changed_coordinate_count",
        "underflow_coordinate_count",
        "overflow_or_nonfinite_coordinate_count",
        "source_finite",
        "target_finite",
        "ordinary_dtype_cast_is_quantization",
    }
)


def _validate_cast_ledger_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _plain_json(value)
    if set(receipt) != _CAST_LEDGER_KEYS:
        raise NumericExecutionPolicyError(
            "Numeric cast ledger fields are incomplete or unknown."
        )
    if receipt["schema"] != NUMERIC_CAST_LEDGER_SCHEMA:
        raise NumericExecutionPolicyError("Numeric cast ledger schema is invalid.")
    _digest(receipt["policy_sha256"], "cast ledger policy_sha256")
    if receipt["entry_count"] != len(receipt["entries"]):
        raise NumericExecutionPolicyError("Numeric cast ledger count is invalid.")
    for index, entry in enumerate(receipt["entries"]):
        if not isinstance(entry, Mapping) or set(entry) != _CAST_LEDGER_ENTRY_KEYS:
            raise NumericExecutionPolicyError("Numeric cast ledger entry is invalid.")
        if entry["entry_index"] != index:
            raise NumericExecutionPolicyError("Numeric cast ledger order is invalid.")
        _identifier(entry["tensor_role"], "cast tensor role")
        _domain_name(entry["domain"])
        shape = entry["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise NumericExecutionPolicyError("Numeric cast shape is invalid.")
        numel = _nonnegative_int(entry["numel"], "cast numel")
        if numel != math.prod(shape):
            raise NumericExecutionPolicyError("Numeric cast numel is invalid.")
        source_dtype = normalize_numeric_dtype(entry["source_dtype"])
        target_dtype = normalize_numeric_dtype(entry["target_dtype"])
        _normalize_device(entry["source_device"])
        _normalize_device(entry["target_device"])
        if (
            entry["source_bytes"] != numel * _DTYPE_ELEMENT_SIZE[source_dtype]
            or entry["target_bytes"] != numel * _DTYPE_ELEMENT_SIZE[target_dtype]
        ):
            raise NumericExecutionPolicyError("Numeric cast byte accounting is invalid.")
        for key in ("source_sha256", "target_sha256"):
            _digest(entry[key], f"cast {key}")
        for key in (
            "dtype_changed",
            "device_changed",
            "lossy_cast",
            "source_finite",
            "target_finite",
        ):
            if not isinstance(entry[key], bool):
                raise NumericExecutionPolicyError(f"Numeric cast {key} is invalid.")
        if (
            entry["dtype_changed"] != (source_dtype != target_dtype)
            or entry["device_changed"]
            != (entry["source_device"] != entry["target_device"])
            or entry["source_finite"] is not True
            or entry["target_finite"] is not True
            or entry["ordinary_dtype_cast_is_quantization"] is not False
        ):
            raise NumericExecutionPolicyError("Numeric cast semantic flags are invalid.")
        changed = _nonnegative_int(
            entry["changed_coordinate_count"], "changed_coordinate_count"
        )
        underflow = _nonnegative_int(
            entry["underflow_coordinate_count"], "underflow_coordinate_count"
        )
        overflow = _nonnegative_int(
            entry["overflow_or_nonfinite_coordinate_count"],
            "overflow_or_nonfinite_coordinate_count",
        )
        if (
            changed > numel
            or underflow > changed
            or overflow != 0
            or entry["lossy_cast"] != (changed > 0)
        ):
            raise NumericExecutionPolicyError("Numeric cast loss accounting is invalid.")
    aggregate_fields = {
        "source_bytes": sum(entry["source_bytes"] for entry in receipt["entries"]),
        "target_bytes": sum(entry["target_bytes"] for entry in receipt["entries"]),
        "lossy_cast_count": sum(
            int(entry["lossy_cast"]) for entry in receipt["entries"]
        ),
        "changed_coordinate_count": sum(
            entry["changed_coordinate_count"] for entry in receipt["entries"]
        ),
        "underflow_coordinate_count": sum(
            entry["underflow_coordinate_count"] for entry in receipt["entries"]
        ),
    }
    if any(receipt[key] != expected for key, expected in aggregate_fields.items()):
        raise NumericExecutionPolicyError("Numeric cast ledger totals are invalid.")
    if receipt["ordinary_dtype_cast_is_quantization"] is not False:
        raise NumericExecutionPolicyError("Numeric cast ledger mislabeled quantization.")
    _validate_operation_audit(receipt["operation_audit"])
    supplied = _digest(receipt["ledger_sha256"], "ledger_sha256")
    body = {key: item for key, item in receipt.items() if key != "ledger_sha256"}
    if supplied != canonical_sha256(body):
        raise NumericExecutionPolicyError("Numeric cast ledger seal is invalid.")
    return receipt


@dataclass(frozen=True, slots=True)
class NumericCastLedgerReceipt:
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt",
            _deep_freeze(_validate_cast_ledger_receipt(self.receipt)),
        )

    @property
    def ledger_sha256(self) -> str:
        return str(self.receipt["ledger_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


class NumericCastLedger:
    """Mutable cast recorder which becomes immutable when sealed."""

    def __init__(self, policy: NumericExecutionPolicy) -> None:
        if not isinstance(policy, NumericExecutionPolicy):
            raise NumericExecutionPolicyError("policy must be NumericExecutionPolicy.")
        self._policy = policy
        self._entries: list[dict[str, Any]] = []
        self._sealed = False

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _target(
        self,
        domain: str,
        *,
        target_dtype: Any | None,
        target_device: Any | None,
        target_like: Any | None,
    ) -> NumericConcretePlacement:
        if target_like is not None:
            if target_dtype is not None or target_device is not None:
                raise NumericExecutionPolicyError(
                    "target_like cannot be combined with explicit dtype/device."
                )
            if not isinstance(target_like, torch.Tensor):
                raise NumericShapeError("target_like must be a tensor.")
            target_dtype = target_like.dtype
            target_device = target_like.device
        placement = self._policy.placement(domain)
        if target_dtype is None and target_device is None and placement.is_concrete:
            return placement.concrete
        if target_dtype is None or target_device is None:
            raise NumericExecutionPolicyError(
                f"{domain} is model-native; concrete target dtype/device are required."
            )
        return NumericConcretePlacement(
            _normalize_device(target_device), normalize_numeric_dtype(target_dtype)
        )

    def record_cast(
        self,
        source: Any,
        target: Any,
        *,
        tensor_role: str,
        domain: NumericDomain | str,
    ) -> None:
        if self._sealed:
            raise NumericExecutionPolicyError("Numeric cast ledger is already sealed.")
        role = _identifier(tensor_role, "tensor role")
        domain_name = _domain_name(domain)
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise NumericShapeError("Cast ledger values must be tensors.")
        if not source.is_floating_point() or not target.is_floating_point():
            raise NumericShapeError("Cast ledger values must be floating tensors.")
        source_dtype = normalize_numeric_dtype(source.dtype)
        target_dtype = normalize_numeric_dtype(target.dtype)
        require_tensor_shape(target, source.shape, f"cast target {role}")
        require_finite_tensor(source, f"cast source {role}")
        require_finite_tensor(target, f"cast target {role}")
        source64 = source.detach().to(device="cpu", dtype=torch.float64)
        target64 = target.detach().to(device="cpu", dtype=torch.float64)
        changed = int((source64 != target64).sum().item())
        underflow = int(((source64 != 0.0) & (target64 == 0.0)).sum().item())
        entry = {
            "entry_index": len(self._entries),
            "tensor_role": role,
            "domain": domain_name,
            "shape": [int(item) for item in source.shape],
            "numel": int(source.numel()),
            "source_dtype": source_dtype,
            "source_device": str(source.device),
            "target_dtype": target_dtype,
            "target_device": str(target.device),
            "source_bytes": int(source.numel() * source.element_size()),
            "target_bytes": int(target.numel() * target.element_size()),
            "source_sha256": _tensor_sha256(source),
            "target_sha256": _tensor_sha256(target),
            "dtype_changed": source_dtype != target_dtype,
            "device_changed": str(source.device) != str(target.device),
            "lossy_cast": changed > 0,
            "changed_coordinate_count": changed,
            "underflow_coordinate_count": underflow,
            "overflow_or_nonfinite_coordinate_count": 0,
            "source_finite": True,
            "target_finite": True,
            "ordinary_dtype_cast_is_quantization": False,
        }
        self._entries.append(entry)

    def cast(
        self,
        value: Any,
        *,
        tensor_role: str,
        domain: NumericDomain | str,
        target_dtype: Any | None = None,
        target_device: Any | None = None,
        target_like: Any | None = None,
    ) -> torch.Tensor:
        if self._sealed:
            raise NumericExecutionPolicyError("Numeric cast ledger is already sealed.")
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise NumericShapeError("Numeric cast source must be a floating tensor.")
        require_finite_tensor(value, f"cast source {tensor_role}")
        domain_name = _domain_name(domain)
        target = self._target(
            domain_name,
            target_dtype=target_dtype,
            target_device=target_device,
            target_like=target_like,
        )
        with torch.no_grad():
            result = (
                value.detach()
                .to(device=target.torch_device, dtype=target.torch_dtype)
                .contiguous()
                .clone()
            )
        require_finite_tensor(result, f"cast target {tensor_role}")
        self.record_cast(
            value,
            result,
            tensor_role=tensor_role,
            domain=domain_name,
        )
        return result

    def seal(self) -> NumericCastLedgerReceipt:
        if self._sealed:
            raise NumericExecutionPolicyError("Numeric cast ledger is already sealed.")
        self._sealed = True
        body = {
            "schema": NUMERIC_CAST_LEDGER_SCHEMA,
            "policy_sha256": self._policy.policy_sha256,
            "entries": list(self._entries),
            "entry_count": len(self._entries),
            "source_bytes": sum(entry["source_bytes"] for entry in self._entries),
            "target_bytes": sum(entry["target_bytes"] for entry in self._entries),
            "lossy_cast_count": sum(
                int(entry["lossy_cast"]) for entry in self._entries
            ),
            "changed_coordinate_count": sum(
                entry["changed_coordinate_count"] for entry in self._entries
            ),
            "underflow_coordinate_count": sum(
                entry["underflow_coordinate_count"] for entry in self._entries
            ),
            "ordinary_dtype_cast_is_quantization": False,
            "operation_audit": dict(_OPERATION_AUDIT),
        }
        return NumericCastLedgerReceipt(
            {**body, "ledger_sha256": canonical_sha256(body)}
        )


__all__ = [
    "NUMERIC_CAST_LEDGER_SCHEMA",
    "NUMERIC_EXECUTION_POLICY_SCHEMA",
    "NUMERIC_EXECUTION_RECEIPT_SCHEMA",
    "NUMERIC_MODEL_INVENTORY_SCHEMA",
    "SUPPORTED_FLOAT_DTYPES",
    "ModelNumericInventory",
    "NumericAuthenticationError",
    "NumericCastLedger",
    "NumericCastLedgerReceipt",
    "NumericConcretePlacement",
    "NumericDeviceLossError",
    "NumericDomain",
    "NumericDomainPlacement",
    "NumericExecutionError",
    "NumericExecutionPolicy",
    "NumericExecutionPolicyError",
    "NumericExecutionRequest",
    "NumericExecutionResult",
    "NumericFallbackEligibleError",
    "NumericFallbackExhaustedError",
    "NumericNonFiniteError",
    "NumericPrepublicationOutOfMemoryError",
    "NumericRankError",
    "NumericShapeError",
    "NumericUnsupportedBackendError",
    "inventory_model_numeric_state",
    "normalize_numeric_dtype",
    "numeric_capability_key",
    "require_authentication",
    "require_finite_tensor",
    "require_full_rank",
    "require_tensor_shape",
    "resolve_numeric_execution_policy",
    "run_with_numeric_fallback",
    "torch_numeric_dtype",
    "verify_numeric_execution_receipt",
]
