"""Deterministic, fail-closed scheduling for read-only replay work.

The folding solvers deliberately remain outside this module.  A caller gives
the scheduler independent replay units, two authenticated replicas of one
installed state, and read-only CPU/CUDA replay callbacks.  The scheduler:

* requires exactly two homogeneous CUDA devices by default;
* assigns canonical shard keys by deterministic round robin;
* proves CPU/CUDA parity before releasing any scheduled result;
* stages CPU tensor archives through two bounded pinned-host slots per device;
* restores canonical result order after concurrent execution; and
* reauthenticates backend settings and installed-state bytes after replay.

This is execution infrastructure only.  It does not fit, select, mutate, or
reinterpret model parameters, candidate states, replay cases, or outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol


DYNAMIC_REPLAY_DEVICE_SCHEMA = "neural-forge-dynamic-replay-device-v1"
DYNAMIC_REPLAY_INSTALLED_STATE_SCHEMA = (
    "neural-forge-dynamic-replay-installed-state-v1"
)
DYNAMIC_REPLAY_PLAN_SCHEMA = "neural-forge-dynamic-replay-plan-v1"
DYNAMIC_REPLAY_PARITY_SCHEMA = "neural-forge-dynamic-replay-parity-v1"
DYNAMIC_REPLAY_BUFFER_SCHEMA = "neural-forge-dynamic-replay-buffer-v1"
DYNAMIC_REPLAY_EXECUTION_SCHEMA = "neural-forge-dynamic-replay-execution-v1"
_SHA256_LENGTH = 64
_ALLOWED_CUBLAS_WORKSPACE_CONFIGS = frozenset({":16:8", ":4096:8"})


class DynamicReplaySchedulingError(RuntimeError):
    """Replay scheduling or authentication failed before result publication."""


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
        raise DynamicReplaySchedulingError(
            "Replay metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicReplaySchedulingError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicReplaySchedulingError(f"{name} must be nonempty text.")
    return value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicReplaySchedulingError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise DynamicReplaySchedulingError(f"{name} must be positive.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise DynamicReplaySchedulingError(
            f"{name} must be finite and nonnegative."
        )
    return float(value)


def _torch_or_none() -> Any | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional native dependency
        return None
    return torch


def _is_tensor(value: Any) -> bool:
    torch = _torch_or_none()
    return torch is not None and isinstance(value, torch.Tensor)


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_bytes(value: Any, *, require_cpu: bool = False) -> tuple[dict[str, Any], bytes]:
    torch = _torch_or_none()
    if torch is None or not isinstance(value, torch.Tensor):
        raise DynamicReplaySchedulingError(
            "Tensor authentication requires PyTorch."
        )
    if (
        value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or torch.is_complex(value)
    ):
        raise DynamicReplaySchedulingError(
            "Replay tensors must be materialized dense detached real tensors."
        )
    if require_cpu and value.device.type != "cpu":
        raise DynamicReplaySchedulingError(
            "Replay input archives must be CPU tensors."
        )
    detached = value.detach().to(device="cpu", non_blocking=False).contiguous()
    if (
        torch.is_floating_point(detached)
        and not bool(torch.isfinite(detached).all().item())
    ):
        raise DynamicReplaySchedulingError(
            "Replay tensors may not contain NaN or infinity."
        )
    descriptor = {
        "kind": "tensor",
        "dtype": _dtype_name(detached),
        "shape": [int(size) for size in detached.shape],
        "stride": [int(size) for size in detached.stride()],
        "layout": str(detached.layout),
    }
    raw = detached.view(torch.uint8).numpy().tobytes(order="C")
    return descriptor, raw


def _update_value_digest(
    digest: Any,
    value: Any,
    *,
    require_cpu_tensors: bool,
    path: str,
) -> None:
    if _is_tensor(value):
        descriptor, raw = _tensor_bytes(
            value,
            require_cpu=require_cpu_tensors,
        )
        digest.update(_canonical_json({"path": path, **descriptor}))
        digest.update(raw)
        return
    if value is None or isinstance(value, (bool, int, str)):
        digest.update(
            _canonical_json(
                {
                    "path": path,
                    "kind": type(value).__name__,
                    "value": value,
                }
            )
        )
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DynamicReplaySchedulingError(
                f"Replay value {path} is not finite."
            )
        digest.update(
            _canonical_json(
                {
                    "path": path,
                    "kind": "float",
                    "value": value,
                }
            )
        )
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DynamicReplaySchedulingError(
                f"Replay mapping {path} must use string keys."
            )
        digest.update(
            _canonical_json(
                {
                    "path": path,
                    "kind": "mapping",
                    "keys": sorted(value),
                }
            )
        )
        for key in sorted(value):
            _update_value_digest(
                digest,
                value[key],
                require_cpu_tensors=require_cpu_tensors,
                path=f"{path}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        digest.update(
            _canonical_json(
                {
                    "path": path,
                    "kind": type(value).__name__,
                    "length": len(value),
                }
            )
        )
        for index, child in enumerate(value):
            _update_value_digest(
                digest,
                child,
                require_cpu_tensors=require_cpu_tensors,
                path=f"{path}[{index}]",
            )
        return
    raise DynamicReplaySchedulingError(
        f"Replay value {path} has unsupported type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def fingerprint_replay_value(
    value: Any,
    *,
    require_cpu_tensors: bool = False,
) -> str:
    """Hash one canonical tensor/scalar tree without changing its values."""

    digest = hashlib.sha256()
    digest.update(b"neural-forge-dynamic-replay-value-v1\0")
    _update_value_digest(
        digest,
        value,
        require_cpu_tensors=require_cpu_tensors,
        path="$",
    )
    return digest.hexdigest()


def _archive_replay_value(value: Any, *, path: str = "$") -> Any:
    if _is_tensor(value):
        torch = _torch_or_none()
        if torch is None:  # pragma: no cover - guarded by _is_tensor
            raise DynamicReplaySchedulingError("PyTorch disappeared.")
        _tensor_bytes(value)
        return value.detach().to(
            device="cpu",
            non_blocking=False,
            copy=True,
            memory_format=torch.contiguous_format,
        )
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DynamicReplaySchedulingError(
                f"Replay output {path} is not finite."
            )
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DynamicReplaySchedulingError(
                f"Replay output mapping {path} must use string keys."
            )
        return MappingProxyType(
            {
                key: _archive_replay_value(value[key], path=f"{path}.{key}")
                for key in sorted(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(
            _archive_replay_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if isinstance(value, list):
        return tuple(
            _archive_replay_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    raise DynamicReplaySchedulingError(
        f"Replay output {path} has unsupported type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


@dataclass(frozen=True, slots=True)
class CudaReplayDeviceReceipt:
    """Authenticated CUDA hardware and deterministic backend settings."""

    device_index: int
    device_uuid: str
    device_name: str
    pci_domain_id: int
    pci_bus_id: int
    pci_device_id: int
    capability: tuple[int, int]
    total_memory_bytes: int
    multiprocessor_count: int
    torch_version: str
    cuda_runtime_version: str
    cudnn_version: int
    deterministic_algorithms_enabled: bool
    deterministic_debug_mode: int
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cudnn_allow_tf32: bool
    cuda_matmul_allow_tf32: bool
    cuda_matmul_allow_bf16_reduced_precision_reduction: bool
    cuda_matmul_allow_fp16_reduced_precision_reduction: bool
    float32_matmul_precision: str
    cublas_workspace_config: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        body = self._body()
        if self.receipt_sha256 != _sha(body):
            raise DynamicReplaySchedulingError(
                "CUDA device receipt seal is invalid."
            )

    @classmethod
    def create(
        cls,
        *,
        device_index: int,
        device_uuid: str,
        device_name: str,
        pci_domain_id: int,
        pci_bus_id: int,
        pci_device_id: int,
        capability: tuple[int, int],
        total_memory_bytes: int,
        multiprocessor_count: int,
        torch_version: str,
        cuda_runtime_version: str,
        cudnn_version: int,
        deterministic_algorithms_enabled: bool,
        deterministic_debug_mode: int,
        cudnn_deterministic: bool,
        cudnn_benchmark: bool,
        cudnn_allow_tf32: bool,
        cuda_matmul_allow_tf32: bool,
        cuda_matmul_allow_bf16_reduced_precision_reduction: bool,
        cuda_matmul_allow_fp16_reduced_precision_reduction: bool,
        float32_matmul_precision: str,
        cublas_workspace_config: str,
    ) -> CudaReplayDeviceReceipt:
        values = {
            "device_index": device_index,
            "device_uuid": device_uuid,
            "device_name": device_name,
            "pci_domain_id": pci_domain_id,
            "pci_bus_id": pci_bus_id,
            "pci_device_id": pci_device_id,
            "capability": capability,
            "total_memory_bytes": total_memory_bytes,
            "multiprocessor_count": multiprocessor_count,
            "torch_version": torch_version,
            "cuda_runtime_version": cuda_runtime_version,
            "cudnn_version": cudnn_version,
            "deterministic_algorithms_enabled": deterministic_algorithms_enabled,
            "deterministic_debug_mode": deterministic_debug_mode,
            "cudnn_deterministic": cudnn_deterministic,
            "cudnn_benchmark": cudnn_benchmark,
            "cudnn_allow_tf32": cudnn_allow_tf32,
            "cuda_matmul_allow_tf32": cuda_matmul_allow_tf32,
            "cuda_matmul_allow_bf16_reduced_precision_reduction": (
                cuda_matmul_allow_bf16_reduced_precision_reduction
            ),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": (
                cuda_matmul_allow_fp16_reduced_precision_reduction
            ),
            "float32_matmul_precision": float32_matmul_precision,
            "cublas_workspace_config": cublas_workspace_config,
        }
        temporary = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(temporary, name, value)
        body = temporary._body()
        return cls(**values, receipt_sha256=_sha(body))

    def _body(self) -> dict[str, Any]:
        index = _nonnegative_int(self.device_index, "device_index")
        uuid = _text(self.device_uuid, "device_uuid")
        name = _text(self.device_name, "device_name")
        pci_domain_id = _nonnegative_int(
            self.pci_domain_id,
            "pci_domain_id",
        )
        pci_bus_id = _nonnegative_int(self.pci_bus_id, "pci_bus_id")
        pci_device_id = _nonnegative_int(
            self.pci_device_id,
            "pci_device_id",
        )
        if (
            not isinstance(self.capability, tuple)
            or len(self.capability) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in self.capability
            )
        ):
            raise DynamicReplaySchedulingError(
                "capability must contain two nonnegative integers."
            )
        total_memory = _positive_int(
            self.total_memory_bytes,
            "total_memory_bytes",
        )
        multiprocessors = _positive_int(
            self.multiprocessor_count,
            "multiprocessor_count",
        )
        cudnn_version = _positive_int(self.cudnn_version, "cudnn_version")
        debug_mode = _nonnegative_int(
            self.deterministic_debug_mode,
            "deterministic_debug_mode",
        )
        boolean_fields = (
            "deterministic_algorithms_enabled",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cudnn_allow_tf32",
            "cuda_matmul_allow_tf32",
            "cuda_matmul_allow_bf16_reduced_precision_reduction",
            "cuda_matmul_allow_fp16_reduced_precision_reduction",
        )
        for field in boolean_fields:
            if not isinstance(getattr(self, field), bool):
                raise DynamicReplaySchedulingError(
                    f"{field} must be boolean."
                )
        body = {
            "schema": DYNAMIC_REPLAY_DEVICE_SCHEMA,
            "backend": "cuda",
            "device_index": index,
            "device_uuid": uuid,
            "device_name": name,
            "pci_domain_id": pci_domain_id,
            "pci_bus_id": pci_bus_id,
            "pci_device_id": pci_device_id,
            "capability": list(self.capability),
            "total_memory_bytes": total_memory,
            "multiprocessor_count": multiprocessors,
            "torch_version": _text(self.torch_version, "torch_version"),
            "cuda_runtime_version": _text(
                self.cuda_runtime_version,
                "cuda_runtime_version",
            ),
            "cudnn_version": cudnn_version,
            "deterministic_algorithms_enabled": (
                self.deterministic_algorithms_enabled
            ),
            "deterministic_debug_mode": debug_mode,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
            "cuda_matmul_allow_tf32": self.cuda_matmul_allow_tf32,
            "cuda_matmul_allow_bf16_reduced_precision_reduction": (
                self.cuda_matmul_allow_bf16_reduced_precision_reduction
            ),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": (
                self.cuda_matmul_allow_fp16_reduced_precision_reduction
            ),
            "float32_matmul_precision": _text(
                self.float32_matmul_precision,
                "float32_matmul_precision",
            ),
            "cublas_workspace_config": _text(
                self.cublas_workspace_config,
                "cublas_workspace_config",
            ),
        }
        return body

    @property
    def hardware_profile_sha256(self) -> str:
        body = self._body()
        return _sha(
            {
                name: body[name]
                for name in (
                    "backend",
                    "device_name",
                    "capability",
                    "total_memory_bytes",
                    "multiprocessor_count",
                )
            }
        )

    @property
    def runtime_profile_sha256(self) -> str:
        body = self._body()
        return _sha(
            {
                name: value
                for name, value in body.items()
                if name
                not in {
                    "schema",
                    "device_index",
                    "device_uuid",
                    "device_name",
                    "pci_domain_id",
                    "pci_bus_id",
                    "pci_device_id",
                    "capability",
                    "total_memory_bytes",
                    "multiprocessor_count",
                }
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}


def capture_cuda_replay_device_receipt(
    device_index: int,
) -> CudaReplayDeviceReceipt:
    """Capture one CUDA device and the backend flags that affect replay."""

    torch = _torch_or_none()
    if torch is None:
        raise DynamicReplaySchedulingError(
            "PyTorch is required to capture a CUDA replay receipt."
        )
    index = _nonnegative_int(device_index, "device_index")
    if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
        raise DynamicReplaySchedulingError(
            f"CUDA device {index} is unavailable."
        )
    properties = torch.cuda.get_device_properties(index)
    cudnn_version = torch.backends.cudnn.version()
    cuda_runtime = getattr(torch.version, "cuda", None)
    if cudnn_version is None or cuda_runtime is None:
        raise DynamicReplaySchedulingError(
            "CUDA/cuDNN runtime identity is incomplete."
        )
    matmul = torch.backends.cuda.matmul
    return CudaReplayDeviceReceipt.create(
        device_index=index,
        device_uuid=str(properties.uuid),
        device_name=str(properties.name),
        pci_domain_id=int(properties.pci_domain_id),
        pci_bus_id=int(properties.pci_bus_id),
        pci_device_id=int(properties.pci_device_id),
        capability=(int(properties.major), int(properties.minor)),
        total_memory_bytes=int(properties.total_memory),
        multiprocessor_count=int(properties.multi_processor_count),
        torch_version=str(torch.__version__),
        cuda_runtime_version=str(cuda_runtime),
        cudnn_version=int(cudnn_version),
        deterministic_algorithms_enabled=bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        deterministic_debug_mode=int(torch.get_deterministic_debug_mode()),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cuda_matmul_allow_tf32=bool(matmul.allow_tf32),
        cuda_matmul_allow_bf16_reduced_precision_reduction=bool(
            matmul.allow_bf16_reduced_precision_reduction
        ),
        cuda_matmul_allow_fp16_reduced_precision_reduction=bool(
            matmul.allow_fp16_reduced_precision_reduction
        ),
        float32_matmul_precision=str(torch.get_float32_matmul_precision()),
        cublas_workspace_config=str(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        ),
    )


@dataclass(frozen=True, slots=True)
class InstalledReplayStateFingerprint:
    """Content identity for one installed replay replica."""

    state_sha256: str
    tensor_count: int
    logical_bytes: int
    fingerprint_sha256: str

    def __post_init__(self) -> None:
        body = self._body()
        if self.fingerprint_sha256 != _sha(body):
            raise DynamicReplaySchedulingError(
                "Installed-state fingerprint seal is invalid."
            )

    @classmethod
    def create(
        cls,
        *,
        state_sha256: str,
        tensor_count: int,
        logical_bytes: int,
    ) -> InstalledReplayStateFingerprint:
        state = _require_sha256(state_sha256, "state_sha256")
        count = _positive_int(tensor_count, "tensor_count")
        byte_count = _positive_int(logical_bytes, "logical_bytes")
        body = {
            "schema": DYNAMIC_REPLAY_INSTALLED_STATE_SCHEMA,
            "state_sha256": state,
            "tensor_count": count,
            "logical_bytes": byte_count,
        }
        return cls(
            state_sha256=state,
            tensor_count=count,
            logical_bytes=byte_count,
            fingerprint_sha256=_sha(body),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": DYNAMIC_REPLAY_INSTALLED_STATE_SCHEMA,
            "state_sha256": _require_sha256(
                self.state_sha256,
                "state_sha256",
            ),
            "tensor_count": _positive_int(
                self.tensor_count,
                "tensor_count",
            ),
            "logical_bytes": _positive_int(
                self.logical_bytes,
                "logical_bytes",
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "fingerprint_sha256": self.fingerprint_sha256,
        }


def fingerprint_installed_replay_state(
    named_tensors: Mapping[str, Any],
) -> InstalledReplayStateFingerprint:
    """Hash every named installed-state tensor in canonical name order."""

    torch = _torch_or_none()
    if torch is None:
        raise DynamicReplaySchedulingError(
            "PyTorch is required to fingerprint installed replay state."
        )
    if (
        not isinstance(named_tensors, Mapping)
        or not named_tensors
        or any(not isinstance(name, str) or not name for name in named_tensors)
    ):
        raise DynamicReplaySchedulingError(
            "Installed replay state requires nonempty named tensors."
        )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-installed-replay-state-v1\0")
    logical_bytes = 0
    object_aliases: dict[int, list[str]] = {}
    storage_aliases: dict[tuple[Any, ...], list[str]] = {}
    for name in sorted(named_tensors):
        value = named_tensors[name]
        if not isinstance(value, torch.Tensor):
            raise DynamicReplaySchedulingError(
                f"Installed state {name!r} is not a tensor."
            )
        descriptor, raw = _tensor_bytes(value)
        digest.update(
            _canonical_json(
                {
                    "name": name,
                    **descriptor,
                }
            )
        )
        digest.update(raw)
        logical_bytes += len(raw)
        object_aliases.setdefault(id(value), []).append(name)
        storage_key = (
            ("empty", id(value))
            if value.numel() == 0
            else (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
        )
        storage_aliases.setdefault(storage_key, []).append(name)
    alias_manifest = {
        "object_alias_groups": sorted(
            (sorted(names) for names in object_aliases.values()),
            key=lambda names: tuple(names),
        ),
        "storage_alias_groups": sorted(
            (sorted(names) for names in storage_aliases.values()),
            key=lambda names: tuple(names),
        ),
    }
    digest.update(_canonical_json(alias_manifest))
    return InstalledReplayStateFingerprint.create(
        state_sha256=digest.hexdigest(),
        tensor_count=len(named_tensors),
        logical_bytes=logical_bytes,
    )


@dataclass(frozen=True, slots=True)
class ReplayReplicaReceipt:
    """One device receipt bound to one installed-state fingerprint."""

    device: CudaReplayDeviceReceipt
    installed_state: InstalledReplayStateFingerprint
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.device, CudaReplayDeviceReceipt):
            raise TypeError("device must be CudaReplayDeviceReceipt.")
        if not isinstance(
            self.installed_state,
            InstalledReplayStateFingerprint,
        ):
            raise TypeError(
                "installed_state must be InstalledReplayStateFingerprint."
            )
        if self.receipt_sha256 != _sha(self._body()):
            raise DynamicReplaySchedulingError(
                "Replay replica receipt seal is invalid."
            )

    @classmethod
    def create(
        cls,
        *,
        device: CudaReplayDeviceReceipt,
        installed_state: InstalledReplayStateFingerprint,
    ) -> ReplayReplicaReceipt:
        body = {
            "device_receipt_sha256": device.receipt_sha256,
            "installed_state_fingerprint_sha256": (
                installed_state.fingerprint_sha256
            ),
        }
        return cls(
            device=device,
            installed_state=installed_state,
            receipt_sha256=_sha(body),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "device_receipt_sha256": self.device.receipt_sha256,
            "installed_state_fingerprint_sha256": (
                self.installed_state.fingerprint_sha256
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "device": self.device.as_dict(),
            "installed_state": self.installed_state.as_dict(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReplaySchedulerPolicy:
    """Predeclared deterministic scheduler and numeric parity policy."""

    expected_device_count: int = 2
    expected_device_name: str = "NVIDIA GeForce RTX 3090"
    pinned_buffer_slots_per_device: int = 2
    maximum_pinned_bytes_per_slot: int = 1 << 30
    parity_absolute_tolerance: float = 0.0
    parity_relative_tolerance: float = 0.0
    parity_maximum_relative_rmse: float = 0.0
    parity_minimum_cosine_similarity: float = 1.0
    require_deterministic_debug_mode: bool = True

    def __post_init__(self) -> None:
        if _positive_int(
            self.expected_device_count,
            "expected_device_count",
        ) != 2:
            raise DynamicReplaySchedulingError(
                "The current production scheduler requires exactly two devices."
            )
        _text(self.expected_device_name, "expected_device_name")
        if self.pinned_buffer_slots_per_device != 2:
            raise DynamicReplaySchedulingError(
                "Pinned-host staging must use exactly two slots per device."
            )
        _positive_int(
            self.maximum_pinned_bytes_per_slot,
            "maximum_pinned_bytes_per_slot",
        )
        _finite_nonnegative(
            self.parity_absolute_tolerance,
            "parity_absolute_tolerance",
        )
        _finite_nonnegative(
            self.parity_relative_tolerance,
            "parity_relative_tolerance",
        )
        _finite_nonnegative(
            self.parity_maximum_relative_rmse,
            "parity_maximum_relative_rmse",
        )
        cosine = float(self.parity_minimum_cosine_similarity)
        if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
            raise DynamicReplaySchedulingError(
                "parity_minimum_cosine_similarity must be within [-1, 1]."
            )
        if not isinstance(self.require_deterministic_debug_mode, bool):
            raise DynamicReplaySchedulingError(
                "require_deterministic_debug_mode must be boolean."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_device_count": self.expected_device_count,
            "expected_device_name": self.expected_device_name,
            "assignment": "canonical_round_robin_by_first_shard_key",
            "pinned_buffer_slots_per_device": (
                self.pinned_buffer_slots_per_device
            ),
            "maximum_pinned_bytes_per_slot": (
                self.maximum_pinned_bytes_per_slot
            ),
            "parity_absolute_tolerance": self.parity_absolute_tolerance,
            "parity_relative_tolerance": self.parity_relative_tolerance,
            "parity_maximum_relative_rmse": (
                self.parity_maximum_relative_rmse
            ),
            "parity_minimum_cosine_similarity": (
                self.parity_minimum_cosine_similarity
            ),
            "require_deterministic_debug_mode": (
                self.require_deterministic_debug_mode
            ),
            "cpu_cuda_parity_required_before_scheduled_work": True,
            "runtime_reauthenticated_after_parity_and_work": True,
            "installed_state_reauthenticated_after_parity_and_work": True,
            "folding_math_changed": False,
        }


def _validate_device_contract(
    replicas: Sequence[ReplayReplicaReceipt],
    policy: ReplaySchedulerPolicy,
) -> tuple[ReplayReplicaReceipt, ...]:
    ordered = tuple(sorted(replicas, key=lambda value: value.device.device_index))
    if len(ordered) != policy.expected_device_count:
        raise DynamicReplaySchedulingError(
            f"Expected exactly {policy.expected_device_count} replay replicas."
        )
    indices = tuple(replica.device.device_index for replica in ordered)
    if len(set(indices)) != len(indices):
        raise DynamicReplaySchedulingError(
            "Replay replicas contain duplicate CUDA device indices."
        )
    hardware = {replica.device.hardware_profile_sha256 for replica in ordered}
    runtimes = {replica.device.runtime_profile_sha256 for replica in ordered}
    states = {
        replica.installed_state.fingerprint_sha256 for replica in ordered
    }
    if len(hardware) != 1 or len(runtimes) != 1:
        raise DynamicReplaySchedulingError(
            "Replay replicas are not hardware/runtime homogeneous."
        )
    if len(states) != 1:
        raise DynamicReplaySchedulingError(
            "Replay replicas do not contain byte-identical installed state."
        )
    for replica in ordered:
        device = replica.device
        if device.device_name != policy.expected_device_name:
            raise DynamicReplaySchedulingError(
                f"CUDA device {device.device_index} is {device.device_name!r}, "
                f"not {policy.expected_device_name!r}."
            )
        if (
            device.deterministic_algorithms_enabled is not True
            or device.cudnn_deterministic is not True
            or device.cudnn_benchmark is not False
            or device.cudnn_allow_tf32 is not False
            or device.cuda_matmul_allow_tf32 is not False
            or device.cuda_matmul_allow_bf16_reduced_precision_reduction
            is not False
            or device.cuda_matmul_allow_fp16_reduced_precision_reduction
            is not False
            or device.float32_matmul_precision != "highest"
            or device.cublas_workspace_config
            not in _ALLOWED_CUBLAS_WORKSPACE_CONFIGS
            or (
                policy.require_deterministic_debug_mode
                and device.deterministic_debug_mode < 1
            )
        ):
            raise DynamicReplaySchedulingError(
                f"CUDA device {device.device_index} is not pinned to the "
                "required deterministic backend contract."
            )
    return ordered


@dataclass(frozen=True, slots=True)
class ReplayWorkItem:
    """One immutable independent phase or candidate replay unit."""

    ordinal: int
    work_id: str
    shard_key: str
    payload: Any
    payload_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.ordinal, "ordinal")
        _text(self.work_id, "work_id")
        _text(self.shard_key, "shard_key")
        claimed = _require_sha256(self.payload_sha256, "payload_sha256")
        observed = fingerprint_replay_value(
            self.payload,
            require_cpu_tensors=True,
        )
        if claimed != observed:
            raise DynamicReplaySchedulingError(
                f"Replay work item {self.work_id!r} payload seal is invalid."
            )

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        work_id: str,
        shard_key: str,
        payload: Any,
    ) -> ReplayWorkItem:
        return cls(
            ordinal=ordinal,
            work_id=work_id,
            shard_key=shard_key,
            payload=payload,
            payload_sha256=fingerprint_replay_value(
                payload,
                require_cpu_tensors=True,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "work_id": self.work_id,
            "shard_key": self.shard_key,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReplayAssignment:
    ordinal: int
    work_id: str
    shard_key: str
    device_index: int
    device_receipt_sha256: str
    installed_state_sha256: str
    input_payload_sha256: str
    assignment_sha256: str

    def __post_init__(self) -> None:
        if self.assignment_sha256 != _sha(self._body()):
            raise DynamicReplaySchedulingError(
                "Replay assignment seal is invalid."
            )

    @classmethod
    def create(
        cls,
        *,
        item: ReplayWorkItem,
        replica: ReplayReplicaReceipt,
    ) -> ReplayAssignment:
        body = {
            "ordinal": item.ordinal,
            "work_id": item.work_id,
            "shard_key": item.shard_key,
            "device_index": replica.device.device_index,
            "device_receipt_sha256": replica.device.receipt_sha256,
            "installed_state_sha256": (
                replica.installed_state.state_sha256
            ),
            "input_payload_sha256": item.payload_sha256,
        }
        return cls(**body, assignment_sha256=_sha(body))

    def _body(self) -> dict[str, Any]:
        return {
            "ordinal": _nonnegative_int(self.ordinal, "ordinal"),
            "work_id": _text(self.work_id, "work_id"),
            "shard_key": _text(self.shard_key, "shard_key"),
            "device_index": _nonnegative_int(
                self.device_index,
                "device_index",
            ),
            "device_receipt_sha256": _require_sha256(
                self.device_receipt_sha256,
                "device_receipt_sha256",
            ),
            "installed_state_sha256": _require_sha256(
                self.installed_state_sha256,
                "installed_state_sha256",
            ),
            "input_payload_sha256": _require_sha256(
                self.input_payload_sha256,
                "input_payload_sha256",
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "assignment_sha256": self.assignment_sha256}


@dataclass(frozen=True, slots=True)
class ReplaySchedulePlan:
    assignments: tuple[ReplayAssignment, ...]
    policy: ReplaySchedulerPolicy
    replicas: tuple[ReplayReplicaReceipt, ...]
    plan_sha256: str

    def __post_init__(self) -> None:
        body = self._body()
        if self.plan_sha256 != _sha(body):
            raise DynamicReplaySchedulingError("Replay plan seal is invalid.")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": DYNAMIC_REPLAY_PLAN_SCHEMA,
            "policy": self.policy.as_dict(),
            "replica_receipt_sha256": [
                replica.receipt_sha256 for replica in self.replicas
            ],
            "assignments": [
                assignment.as_dict() for assignment in self.assignments
            ],
            "canonical_output_order": [
                assignment.work_id for assignment in self.assignments
            ],
            "dynamic_work_stealing": False,
            "folding_math_changed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "plan_sha256": self.plan_sha256}


def build_replay_schedule_plan(
    items: Sequence[ReplayWorkItem],
    *,
    replicas: Sequence[ReplayReplicaReceipt],
    policy: ReplaySchedulerPolicy,
) -> ReplaySchedulePlan:
    """Build one canonical two-device assignment independent of caller order."""

    if not isinstance(policy, ReplaySchedulerPolicy):
        raise TypeError("policy must be ReplaySchedulerPolicy.")
    ordered_replicas = _validate_device_contract(replicas, policy)
    ordered_items = tuple(sorted(items, key=lambda value: value.ordinal))
    if not ordered_items:
        raise DynamicReplaySchedulingError("Replay plan cannot be empty.")
    if any(not isinstance(item, ReplayWorkItem) for item in ordered_items):
        raise TypeError("items must contain ReplayWorkItem values.")
    if tuple(item.ordinal for item in ordered_items) != tuple(
        range(len(ordered_items))
    ):
        raise DynamicReplaySchedulingError(
            "Replay ordinals must be contiguous from zero."
        )
    if len({item.work_id for item in ordered_items}) != len(ordered_items):
        raise DynamicReplaySchedulingError(
            "Replay work IDs must be unique."
        )
    shard_devices: dict[str, ReplayReplicaReceipt] = {}
    assignments = []
    for item in ordered_items:
        replica = shard_devices.get(item.shard_key)
        if replica is None:
            replica = ordered_replicas[
                len(shard_devices) % len(ordered_replicas)
            ]
            shard_devices[item.shard_key] = replica
        assignments.append(
            ReplayAssignment.create(item=item, replica=replica)
        )
    temporary = object.__new__(ReplaySchedulePlan)
    object.__setattr__(temporary, "assignments", tuple(assignments))
    object.__setattr__(temporary, "policy", policy)
    object.__setattr__(temporary, "replicas", ordered_replicas)
    body = temporary._body()
    return ReplaySchedulePlan(
        assignments=tuple(assignments),
        policy=policy,
        replicas=ordered_replicas,
        plan_sha256=_sha(body),
    )


def _tensor_leaves(
    value: Any,
    *,
    path: str = "$",
) -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    if _is_tensor(value):
        leaves[path] = value
        return leaves
    if isinstance(value, Mapping):
        for key in sorted(value):
            leaves.update(_tensor_leaves(value[key], path=f"{path}.{key}"))
        return leaves
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            leaves.update(_tensor_leaves(child, path=f"{path}[{index}]"))
        return leaves
    if value is None or isinstance(value, (bool, int, float, str)):
        return leaves
    raise DynamicReplaySchedulingError(
        f"Parity output {path} has an unsupported type."
    )


def _structure_descriptor(value: Any, *, path: str = "$") -> Any:
    if _is_tensor(value):
        return {
            "path": path,
            "kind": "tensor",
            "dtype": _dtype_name(value),
            "shape": [int(size) for size in value.shape],
        }
    if isinstance(value, Mapping):
        return {
            "path": path,
            "kind": "mapping",
            "children": [
                _structure_descriptor(value[key], path=f"{path}.{key}")
                for key in sorted(value)
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "path": path,
            "kind": "sequence",
            "children": [
                _structure_descriptor(child, path=f"{path}[{index}]")
                for index, child in enumerate(value)
            ],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {
            "path": path,
            "kind": type(value).__name__,
            "value": value,
        }
    raise DynamicReplaySchedulingError(
        f"Parity output {path} has an unsupported type."
    )


def compare_cpu_cuda_replay(
    cpu_output: Any,
    cuda_output: Any,
    *,
    policy: ReplaySchedulerPolicy,
    device_index: int,
) -> Mapping[str, Any]:
    """Compare canonical CPU/CUDA output trees under a preregistered gate."""

    torch = _torch_or_none()
    cpu = _archive_replay_value(cpu_output)
    cuda = _archive_replay_value(cuda_output)
    cpu_structure = _structure_descriptor(cpu)
    cuda_structure = _structure_descriptor(cuda)
    if cpu_structure != cuda_structure:
        raise DynamicReplaySchedulingError(
            f"CPU/CUDA parity structure differs on cuda:{device_index}."
        )
    cpu_leaves = _tensor_leaves(cpu)
    cuda_leaves = _tensor_leaves(cuda)
    tensor_reports = []
    passed = True
    for path in sorted(cpu_leaves):
        left = cpu_leaves[path]
        right = cuda_leaves[path]
        if torch is None:  # pragma: no cover - tensor leaves require torch
            raise DynamicReplaySchedulingError("PyTorch disappeared.")
        if torch.is_floating_point(left) or torch.is_complex(left):
            if left.numel() == 0:
                leaf_passed = True
                tensor_reports.append(
                    {
                        "path": path,
                        "dtype": _dtype_name(left),
                        "shape": [int(size) for size in left.shape],
                        "empty_tensor_exact_geometry_match": True,
                        "passed": True,
                    }
                )
                passed = passed and leaf_passed
                continue
            left64 = left.to(torch.float64)
            right64 = right.to(torch.float64)
            delta = right64 - left64
            maximum_absolute_error = float(delta.abs().max().item())
            reference_amplitude = float(left64.abs().max().item())
            allowed_absolute_error = (
                policy.parity_absolute_tolerance
                + policy.parity_relative_tolerance * reference_amplitude
            )
            reference_rms = float(left64.square().mean().sqrt().item())
            error_rms = float(delta.square().mean().sqrt().item())
            relative_rmse = error_rms / max(
                reference_rms,
                torch.finfo(torch.float64).tiny,
            )
            flattened_left = left64.reshape(-1)
            flattened_right = right64.reshape(-1)
            denominator = float(
                (
                    torch.linalg.vector_norm(flattened_left)
                    * torch.linalg.vector_norm(flattened_right)
                ).item()
            )
            if denominator == 0.0:
                cosine = 1.0 if bool(torch.equal(left, right)) else 0.0
            else:
                cosine = float(
                    torch.dot(flattened_left, flattened_right).item()
                    / denominator
                )
            leaf_passed = bool(
                math.isfinite(maximum_absolute_error)
                and math.isfinite(relative_rmse)
                and math.isfinite(cosine)
                and maximum_absolute_error <= allowed_absolute_error
                and relative_rmse
                <= policy.parity_maximum_relative_rmse
                and cosine >= policy.parity_minimum_cosine_similarity
            )
            tensor_reports.append(
                {
                    "path": path,
                    "dtype": _dtype_name(left),
                    "shape": [int(size) for size in left.shape],
                    "maximum_absolute_error": maximum_absolute_error,
                    "allowed_absolute_error": allowed_absolute_error,
                    "relative_rmse": relative_rmse,
                    "maximum_relative_rmse": (
                        policy.parity_maximum_relative_rmse
                    ),
                    "cosine_similarity": cosine,
                    "minimum_cosine_similarity": (
                        policy.parity_minimum_cosine_similarity
                    ),
                    "passed": leaf_passed,
                }
            )
        else:
            leaf_passed = bool(torch.equal(left, right))
            tensor_reports.append(
                {
                    "path": path,
                    "dtype": _dtype_name(left),
                    "shape": [int(size) for size in left.shape],
                    "exact_match_required": True,
                    "passed": leaf_passed,
                }
            )
        passed = passed and leaf_passed
    cpu_scalar_sha = fingerprint_replay_value(cpu)
    cuda_scalar_sha = fingerprint_replay_value(cuda)
    if not cpu_leaves and cpu_scalar_sha != cuda_scalar_sha:
        passed = False
    body = {
        "schema": DYNAMIC_REPLAY_PARITY_SCHEMA,
        "device_index": _nonnegative_int(device_index, "device_index"),
        "cpu_output_sha256": cpu_scalar_sha,
        "cuda_output_sha256": cuda_scalar_sha,
        "tensor_reports": tensor_reports,
        "tensor_leaf_count": len(tensor_reports),
        "structure_sha256": _sha(cpu_structure),
        "non_tensor_output_exact_match": (
            cpu_scalar_sha == cuda_scalar_sha if not tensor_reports else True
        ),
        "passed": passed,
    }
    report = {**body, "report_sha256": _sha(body)}
    if not passed:
        raise DynamicReplaySchedulingError(
            f"CPU/CUDA replay parity failed on cuda:{device_index}."
        )
    return MappingProxyType(report)


@dataclass(frozen=True, slots=True)
class StagedReplayPayload:
    value: Any
    slot_index: int
    source_payload_sha256: str
    pinned_payload_sha256: str
    tensor_bytes: int


class ReplayHostBuffer(Protocol):
    """Interface accepted by :class:`DeterministicReplayScheduler`."""

    slot_count: int
    maximum_bytes_per_slot: int
    pinned_host_memory: bool
    asynchronous_h2d: bool

    def stage(self, item: ReplayWorkItem, slot_index: int) -> StagedReplayPayload: ...

    def wait_ready(self, slot_index: int) -> None: ...

    def mark_consumed(self, slot_index: int) -> None: ...

    def as_dict(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _PinnedSlot:
    host_tensors: dict[str, Any]
    gpu_value: Any | None = None
    transfer_event: Any | None = None
    consumption_event: Any | None = None
    tensor_bytes: int = 0


class PinnedHostDoubleBuffer:
    """Two-slot reusable pinned CPU staging for one CUDA replay replica."""

    slot_count = 2
    pinned_host_memory = True
    asynchronous_h2d = True

    def __init__(
        self,
        replica: ReplayReplicaReceipt,
        *,
        maximum_bytes_per_slot: int,
    ) -> None:
        torch = _torch_or_none()
        if torch is None:
            raise DynamicReplaySchedulingError(
                "PyTorch is required for pinned CUDA staging."
            )
        if not torch.cuda.is_available():
            raise DynamicReplaySchedulingError(
                "Pinned CUDA staging requires an available CUDA runtime."
            )
        self.replica = replica
        self.maximum_bytes_per_slot = _positive_int(
            maximum_bytes_per_slot,
            "maximum_bytes_per_slot",
        )
        self._torch = torch
        self._device = torch.device(
            "cuda",
            replica.device.device_index,
        )
        self._transfer_stream = torch.cuda.Stream(device=self._device)
        self._slots = (
            _PinnedSlot(host_tensors={}),
            _PinnedSlot(host_tensors={}),
        )
        self._stage_count = 0
        self._maximum_observed_slot_bytes = 0
        self._closed = False

    def _payload_tensor_bytes(self, value: Any, *, path: str) -> int:
        torch = self._torch
        if isinstance(value, torch.Tensor):
            _tensor_bytes(value, require_cpu=True)
            return int(value.numel()) * int(value.element_size())
        if isinstance(value, Mapping):
            return sum(
                self._payload_tensor_bytes(
                    value[key],
                    path=f"{path}.{key}",
                )
                for key in sorted(value)
            )
        if isinstance(value, (list, tuple)):
            return sum(
                self._payload_tensor_bytes(
                    child,
                    path=f"{path}[{index}]",
                )
                for index, child in enumerate(value)
            )
        if value is None or isinstance(value, (bool, int, float, str)):
            return 0
        raise DynamicReplaySchedulingError(
            f"Staging payload {path} has an unsupported type."
        )

    def _stage_tree(
        self,
        value: Any,
        *,
        path: str,
        slot: _PinnedSlot,
        active_paths: set[str],
    ) -> tuple[Any, int]:
        torch = self._torch
        if isinstance(value, torch.Tensor):
            _tensor_bytes(value, require_cpu=True)
            active_paths.add(path)
            byte_count = int(value.numel()) * int(value.element_size())
            host = slot.host_tensors.get(path)
            if (
                host is None
                or tuple(host.shape) != tuple(value.shape)
                or host.dtype != value.dtype
            ):
                if host is not None:
                    del slot.host_tensors[path]
                host = torch.empty_like(
                    value,
                    device="cpu",
                    memory_format=torch.contiguous_format,
                    pin_memory=True,
                )
                slot.host_tensors[path] = host
            host.copy_(value, non_blocking=False)
            if not bool(host.is_pinned()):
                raise DynamicReplaySchedulingError(
                    f"Pinned staging allocation {path} is not pinned."
                )
            return (
                host.to(
                    device=self._device,
                    non_blocking=True,
                ),
                byte_count,
            )
        if isinstance(value, Mapping):
            staged = {}
            total = 0
            for key in sorted(value):
                child, child_bytes = self._stage_tree(
                    value[key],
                    path=f"{path}.{key}",
                    slot=slot,
                    active_paths=active_paths,
                )
                staged[key] = child
                total += child_bytes
            return MappingProxyType(staged), total
        if isinstance(value, tuple):
            children = []
            total = 0
            for index, child_value in enumerate(value):
                child, child_bytes = self._stage_tree(
                    child_value,
                    path=f"{path}[{index}]",
                    slot=slot,
                    active_paths=active_paths,
                )
                children.append(child)
                total += child_bytes
            return tuple(children), total
        if isinstance(value, list):
            children = []
            total = 0
            for index, child_value in enumerate(value):
                child, child_bytes = self._stage_tree(
                    child_value,
                    path=f"{path}[{index}]",
                    slot=slot,
                    active_paths=active_paths,
                )
                children.append(child)
                total += child_bytes
            return tuple(children), total
        if value is None or isinstance(value, (bool, int, float, str)):
            return value, 0
        raise DynamicReplaySchedulingError(
            f"Staging payload {path} has an unsupported type."
        )

    def stage(
        self,
        item: ReplayWorkItem,
        slot_index: int,
    ) -> StagedReplayPayload:
        if self._closed:
            raise DynamicReplaySchedulingError("Pinned buffer is closed.")
        if slot_index not in {0, 1}:
            raise DynamicReplaySchedulingError(
                "Pinned double-buffer slot must be zero or one."
            )
        observed = fingerprint_replay_value(
            item.payload,
            require_cpu_tensors=True,
        )
        if observed != item.payload_sha256:
            raise DynamicReplaySchedulingError(
                f"Replay input {item.work_id!r} changed before staging."
            )
        required_bytes = self._payload_tensor_bytes(item.payload, path="$")
        if required_bytes > self.maximum_bytes_per_slot:
            raise DynamicReplaySchedulingError(
                f"Replay input {item.work_id!r} requires {required_bytes} "
                f"pinned bytes, above the per-slot limit "
                f"{self.maximum_bytes_per_slot}."
            )
        torch = self._torch
        slot = self._slots[slot_index]
        if slot.consumption_event is not None:
            slot.consumption_event.synchronize()
        expected_paths: set[str] = set()
        self._collect_tensor_paths(
            item.payload,
            path="$",
            paths=expected_paths,
        )
        for path in tuple(slot.host_tensors):
            if path not in expected_paths:
                del slot.host_tensors[path]
        active_paths: set[str] = set()
        with torch.cuda.device(self._device), torch.cuda.stream(
            self._transfer_stream
        ):
            gpu_value, tensor_bytes = self._stage_tree(
                item.payload,
                path="$",
                slot=slot,
                active_paths=active_paths,
            )
            event = torch.cuda.Event(enable_timing=False, blocking=False)
            event.record(self._transfer_stream)
        for path in tuple(slot.host_tensors):
            if path not in active_paths:
                del slot.host_tensors[path]
        if tensor_bytes != required_bytes:
            raise DynamicReplaySchedulingError(
                f"Staged tensor-byte accounting changed for {item.work_id!r}."
            )
        pinned_tree = self._pinned_tree(item.payload, path="$", slot=slot)
        pinned_sha = fingerprint_replay_value(
            pinned_tree,
            require_cpu_tensors=True,
        )
        if pinned_sha != item.payload_sha256:
            raise DynamicReplaySchedulingError(
                f"Pinned staging bytes changed for {item.work_id!r}."
            )
        slot.gpu_value = gpu_value
        slot.transfer_event = event
        slot.consumption_event = None
        slot.tensor_bytes = tensor_bytes
        self._stage_count += 1
        self._maximum_observed_slot_bytes = max(
            self._maximum_observed_slot_bytes,
            tensor_bytes,
        )
        return StagedReplayPayload(
            value=gpu_value,
            slot_index=slot_index,
            source_payload_sha256=observed,
            pinned_payload_sha256=pinned_sha,
            tensor_bytes=tensor_bytes,
        )

    def _collect_tensor_paths(
        self,
        value: Any,
        *,
        path: str,
        paths: set[str],
    ) -> None:
        if _is_tensor(value):
            paths.add(path)
            return
        if isinstance(value, Mapping):
            for key in sorted(value):
                self._collect_tensor_paths(
                    value[key],
                    path=f"{path}.{key}",
                    paths=paths,
                )
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                self._collect_tensor_paths(
                    child,
                    path=f"{path}[{index}]",
                    paths=paths,
                )

    def _pinned_tree(
        self,
        value: Any,
        *,
        path: str,
        slot: _PinnedSlot,
    ) -> Any:
        if _is_tensor(value):
            return slot.host_tensors[path]
        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    key: self._pinned_tree(
                        value[key],
                        path=f"{path}.{key}",
                        slot=slot,
                    )
                    for key in sorted(value)
                }
            )
        if isinstance(value, (list, tuple)):
            return tuple(
                self._pinned_tree(
                    child,
                    path=f"{path}[{index}]",
                    slot=slot,
                )
                for index, child in enumerate(value)
            )
        return value

    def wait_ready(self, slot_index: int) -> None:
        if self._closed or slot_index not in {0, 1}:
            raise DynamicReplaySchedulingError(
                "Pinned buffer slot is unavailable."
            )
        event = self._slots[slot_index].transfer_event
        if event is None:
            raise DynamicReplaySchedulingError(
                "Pinned buffer slot was not staged."
            )
        torch = self._torch
        with torch.cuda.device(self._device):
            torch.cuda.current_stream(self._device).wait_event(event)

    def mark_consumed(self, slot_index: int) -> None:
        if self._closed or slot_index not in {0, 1}:
            raise DynamicReplaySchedulingError(
                "Pinned buffer slot is unavailable."
            )
        slot = self._slots[slot_index]
        if slot.transfer_event is None:
            raise DynamicReplaySchedulingError(
                "Cannot consume an unstaged pinned slot."
            )
        torch = self._torch
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        with torch.cuda.device(self._device):
            event.record(torch.cuda.current_stream(self._device))
        slot.consumption_event = event

    def as_dict(self) -> Mapping[str, Any]:
        body = {
            "schema": DYNAMIC_REPLAY_BUFFER_SCHEMA,
            "device_index": self.replica.device.device_index,
            "device_receipt_sha256": self.replica.device.receipt_sha256,
            "slot_count": self.slot_count,
            "maximum_bytes_per_slot": self.maximum_bytes_per_slot,
            "maximum_total_pinned_bytes": (
                self.slot_count * self.maximum_bytes_per_slot
            ),
            "maximum_observed_slot_bytes": self._maximum_observed_slot_bytes,
            "stage_count": self._stage_count,
            "pinned_host_memory": True,
            "asynchronous_h2d": True,
            "canonical_pinned_payload_hash_verified": True,
        }
        return MappingProxyType({**body, "report_sha256": _sha(body)})

    def close(self) -> None:
        if self._closed:
            return
        for slot in self._slots:
            if slot.consumption_event is not None:
                slot.consumption_event.synchronize()
            elif slot.transfer_event is not None:
                slot.transfer_event.synchronize()
            slot.gpu_value = None
            slot.host_tensors.clear()
        self._closed = True


ReplayWorker = Callable[[ReplayWorkItem, ReplayReplicaReceipt, Any], Any]
CpuParityWorker = Callable[[ReplayWorkItem, Any], Any]
DeviceReceiptProbe = Callable[[int], CudaReplayDeviceReceipt]
InstalledStateProbe = Callable[
    [ReplayReplicaReceipt],
    InstalledReplayStateFingerprint,
]
CpuInstalledStateProbe = Callable[[], InstalledReplayStateFingerprint]
HostBufferFactory = Callable[[ReplayReplicaReceipt], ReplayHostBuffer]


@dataclass(frozen=True, slots=True)
class ReplayExecutionResult:
    outputs: tuple[Any, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        report = dict(self.report)
        body = dict(report)
        claimed = body.pop("report_sha256", None)
        if claimed != _sha(body):
            raise DynamicReplaySchedulingError(
                "Replay execution report seal is invalid."
            )
        output_hashes = [
            fingerprint_replay_value(value) for value in self.outputs
        ]
        if report.get("canonical_output_sha256") != output_hashes:
            raise DynamicReplaySchedulingError(
                "Replay execution outputs differ from their receipt."
            )
        object.__setattr__(self, "report", MappingProxyType(report))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


class DeterministicReplayScheduler:
    """Execute authenticated replay units on two homogeneous CUDA replicas."""

    def __init__(
        self,
        *,
        replicas: Sequence[ReplayReplicaReceipt],
        policy: ReplaySchedulerPolicy,
        gpu_worker: ReplayWorker,
        cpu_parity_worker: CpuParityWorker,
        device_receipt_probe: DeviceReceiptProbe,
        installed_state_probe: InstalledStateProbe,
        cpu_installed_state_probe: CpuInstalledStateProbe,
        host_buffer_factory: HostBufferFactory | None = None,
    ) -> None:
        self.policy = policy
        self.replicas = _validate_device_contract(replicas, policy)
        self.gpu_worker = gpu_worker
        self.cpu_parity_worker = cpu_parity_worker
        self.device_receipt_probe = device_receipt_probe
        self.installed_state_probe = installed_state_probe
        self.cpu_installed_state_probe = cpu_installed_state_probe
        if host_buffer_factory is None:
            self.host_buffer_factory = lambda replica: PinnedHostDoubleBuffer(
                replica,
                maximum_bytes_per_slot=(
                    self.policy.maximum_pinned_bytes_per_slot
                ),
            )
        else:
            self.host_buffer_factory = host_buffer_factory

    def _reauthenticate(self, stage: str) -> None:
        expected_state = self.replicas[0].installed_state
        cpu_state = self.cpu_installed_state_probe()
        if cpu_state != expected_state:
            raise DynamicReplaySchedulingError(
                f"CPU installed state differs at {stage}."
            )
        for replica in self.replicas:
            current_device = self.device_receipt_probe(
                replica.device.device_index
            )
            if current_device != replica.device:
                raise DynamicReplaySchedulingError(
                    f"CUDA runtime/device receipt changed at {stage} on "
                    f"cuda:{replica.device.device_index}."
                )
            current_state = self.installed_state_probe(replica)
            if current_state != replica.installed_state:
                raise DynamicReplaySchedulingError(
                    f"Installed replay state changed at {stage} on "
                    f"cuda:{replica.device.device_index}."
                )

    def _validate_buffer(
        self,
        replica: ReplayReplicaReceipt,
        buffer: ReplayHostBuffer,
    ) -> None:
        if (
            buffer.slot_count != self.policy.pinned_buffer_slots_per_device
            or buffer.maximum_bytes_per_slot
            != self.policy.maximum_pinned_bytes_per_slot
            or buffer.pinned_host_memory is not True
            or buffer.asynchronous_h2d is not True
        ):
            raise DynamicReplaySchedulingError(
                f"Host buffer for cuda:{replica.device.device_index} does not "
                "satisfy the bounded pinned double-buffer contract."
            )

    def _run_gpu_one(
        self,
        *,
        item: ReplayWorkItem,
        replica: ReplayReplicaReceipt,
        buffer: ReplayHostBuffer,
        slot_index: int,
        staged: StagedReplayPayload | None = None,
    ) -> Any:
        staged_value = (
            buffer.stage(item, slot_index)
            if staged is None
            else staged
        )
        if (
            staged_value.slot_index != slot_index
            or staged_value.source_payload_sha256 != item.payload_sha256
            or staged_value.pinned_payload_sha256 != item.payload_sha256
            or staged_value.tensor_bytes
            > self.policy.maximum_pinned_bytes_per_slot
        ):
            raise DynamicReplaySchedulingError(
                f"Staging receipt changed for {item.work_id!r}."
            )
        buffer.wait_ready(slot_index)
        torch = _torch_or_none()
        context = nullcontext() if torch is None else torch.no_grad()
        with context:
            raw_output = self.gpu_worker(
                item,
                replica,
                staged_value.value,
            )
        output = _archive_replay_value(raw_output)
        buffer.mark_consumed(slot_index)
        if fingerprint_replay_value(
            item.payload,
            require_cpu_tensors=True,
        ) != item.payload_sha256:
            raise DynamicReplaySchedulingError(
                f"Replay input {item.work_id!r} changed during GPU replay."
            )
        return output

    def _execute_shard(
        self,
        *,
        replica: ReplayReplicaReceipt,
        items: Sequence[ReplayWorkItem],
        buffer: ReplayHostBuffer,
    ) -> list[tuple[int, Any, dict[str, Any]]]:
        if not items:
            return []
        prepared: dict[int, StagedReplayPayload] = {}
        first_count = min(2, len(items))
        for index in range(first_count):
            prepared[index] = buffer.stage(items[index], index % 2)
        results = []
        for index, item in enumerate(items):
            slot_index = index % 2
            staged = prepared.pop(index)
            output = self._run_gpu_one(
                item=item,
                replica=replica,
                buffer=buffer,
                slot_index=slot_index,
                staged=staged,
            )
            future_index = index + 2
            if future_index < len(items):
                prepared[future_index] = buffer.stage(
                    items[future_index],
                    slot_index,
                )
            output_sha = fingerprint_replay_value(output)
            record_body = {
                "ordinal": item.ordinal,
                "work_id": item.work_id,
                "shard_key": item.shard_key,
                "device_index": replica.device.device_index,
                "device_receipt_sha256": replica.device.receipt_sha256,
                "installed_state_sha256": (
                    replica.installed_state.state_sha256
                ),
                "input_payload_sha256": item.payload_sha256,
                "output_sha256": output_sha,
                "slot_index": slot_index,
                "folding_math_changed": False,
            }
            results.append(
                (
                    item.ordinal,
                    output,
                    {
                        **record_body,
                        "record_sha256": _sha(record_body),
                    },
                )
            )
        return results

    def execute(
        self,
        items: Sequence[ReplayWorkItem],
        *,
        parity_item: ReplayWorkItem,
    ) -> ReplayExecutionResult:
        """Run parity, then scheduled work, returning only canonical results."""

        if not isinstance(parity_item, ReplayWorkItem):
            raise TypeError("parity_item must be ReplayWorkItem.")
        plan = build_replay_schedule_plan(
            items,
            replicas=self.replicas,
            policy=self.policy,
        )
        item_by_id = {item.work_id: item for item in items}
        canonical_items = tuple(
            item_by_id[assignment.work_id]
            for assignment in plan.assignments
        )
        for item in (*canonical_items, parity_item):
            if fingerprint_replay_value(
                item.payload,
                require_cpu_tensors=True,
            ) != item.payload_sha256:
                raise DynamicReplaySchedulingError(
                    f"Replay input {item.work_id!r} changed before execution."
                )

        buffers: dict[int, ReplayHostBuffer] = {}
        try:
            for replica in self.replicas:
                buffer = self.host_buffer_factory(replica)
                self._validate_buffer(replica, buffer)
                buffers[replica.device.device_index] = buffer
            self._reauthenticate("pre_parity")

            torch = _torch_or_none()
            context = nullcontext() if torch is None else torch.no_grad()
            with context:
                cpu_output = _archive_replay_value(
                    self.cpu_parity_worker(
                        parity_item,
                        parity_item.payload,
                    )
                )
            if fingerprint_replay_value(
                parity_item.payload,
                require_cpu_tensors=True,
            ) != parity_item.payload_sha256:
                raise DynamicReplaySchedulingError(
                    "CPU parity replay changed its input payload."
                )
            parity_reports = []
            for replica in self.replicas:
                gpu_output = self._run_gpu_one(
                    item=parity_item,
                    replica=replica,
                    buffer=buffers[replica.device.device_index],
                    slot_index=0,
                )
                parity_reports.append(
                    dict(
                        compare_cpu_cuda_replay(
                            cpu_output,
                            gpu_output,
                            policy=self.policy,
                            device_index=replica.device.device_index,
                        )
                    )
                )
            self._reauthenticate("post_parity")

            assignment_by_id = {
                assignment.work_id: assignment
                for assignment in plan.assignments
            }
            shards: dict[int, list[ReplayWorkItem]] = {
                replica.device.device_index: []
                for replica in self.replicas
            }
            for item in canonical_items:
                shards[assignment_by_id[item.work_id].device_index].append(
                    item
                )
            replica_by_index = {
                replica.device.device_index: replica
                for replica in self.replicas
            }
            scheduled_results: list[
                tuple[int, Any, dict[str, Any]]
            ] = []
            with ThreadPoolExecutor(
                max_workers=len(self.replicas),
                thread_name_prefix="neural-forge-replay",
            ) as executor:
                futures = {
                    executor.submit(
                        self._execute_shard,
                        replica=replica_by_index[device_index],
                        items=tuple(shard),
                        buffer=buffers[device_index],
                    ): device_index
                    for device_index, shard in shards.items()
                }
                for future in as_completed(futures):
                    try:
                        scheduled_results.extend(future.result())
                    except BaseException as exc:
                        for sibling in futures:
                            sibling.cancel()
                        raise DynamicReplaySchedulingError(
                            "A replay shard failed; no scheduled outputs were "
                            "published."
                        ) from exc
            self._reauthenticate("post_work")

            scheduled_results.sort(key=lambda value: value[0])
            if tuple(row[0] for row in scheduled_results) != tuple(
                range(len(canonical_items))
            ):
                raise DynamicReplaySchedulingError(
                    "Scheduled results are missing or noncanonical."
                )
            outputs = tuple(row[1] for row in scheduled_results)
            records = [row[2] for row in scheduled_results]
            for item in (*canonical_items, parity_item):
                if fingerprint_replay_value(
                    item.payload,
                    require_cpu_tensors=True,
                ) != item.payload_sha256:
                    raise DynamicReplaySchedulingError(
                        f"Replay input {item.work_id!r} changed during execution."
                    )
            buffer_reports = [
                dict(buffers[index].as_dict())
                for index in sorted(buffers)
            ]
            body = {
                "schema": DYNAMIC_REPLAY_EXECUTION_SCHEMA,
                "plan_sha256": plan.plan_sha256,
                "plan": plan.as_dict(),
                "parity_work_id": parity_item.work_id,
                "parity_payload_sha256": parity_item.payload_sha256,
                "parity_reports": parity_reports,
                "parity_passed_before_scheduled_work": True,
                "scheduled_records": records,
                "canonical_output_work_ids": [
                    item.work_id for item in canonical_items
                ],
                "canonical_output_sha256": [
                    fingerprint_replay_value(value) for value in outputs
                ],
                "buffer_reports": buffer_reports,
                "backend_receipts_reauthenticated": True,
                "installed_state_reauthenticated": True,
                "cpu_reference_state_reauthenticated": True,
                "dynamic_work_stealing": False,
                "canonical_result_order_restored": True,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "folding_math_changed": False,
            }
            report = {**body, "report_sha256": _sha(body)}
            return ReplayExecutionResult(outputs=outputs, report=report)
        finally:
            for buffer in buffers.values():
                buffer.close()


__all__ = [
    "DYNAMIC_REPLAY_BUFFER_SCHEMA",
    "DYNAMIC_REPLAY_DEVICE_SCHEMA",
    "DYNAMIC_REPLAY_EXECUTION_SCHEMA",
    "DYNAMIC_REPLAY_INSTALLED_STATE_SCHEMA",
    "DYNAMIC_REPLAY_PARITY_SCHEMA",
    "DYNAMIC_REPLAY_PLAN_SCHEMA",
    "CudaReplayDeviceReceipt",
    "DeterministicReplayScheduler",
    "DynamicReplaySchedulingError",
    "InstalledReplayStateFingerprint",
    "PinnedHostDoubleBuffer",
    "ReplayAssignment",
    "ReplayExecutionResult",
    "ReplayHostBuffer",
    "ReplayReplicaReceipt",
    "ReplaySchedulePlan",
    "ReplaySchedulerPolicy",
    "ReplayWorkItem",
    "StagedReplayPayload",
    "build_replay_schedule_plan",
    "capture_cuda_replay_device_receipt",
    "compare_cpu_cuda_replay",
    "fingerprint_installed_replay_state",
    "fingerprint_replay_value",
]
