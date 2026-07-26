"""Sealed device policy for deterministic analytic folding solves.

The profile produced here is a plan, not an execution context.  It records the
requested and resolved device, the only permitted numeric dtype, deterministic
runtime requirements, and any explicit CUDA-to-CPU fallback.  Callers can
authenticate that plan before applying its settings around a numerical solve.

Resolving a profile performs no tensor allocation, model execution, trace
capture, autograd, training, optimizer step, or filesystem write.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256


ANALYTIC_EXECUTION_PROFILE_SCHEMA = "neural-forge-analytic-execution-profile-v1"

_CPU_THREAD_POLICY = "single-thread-under-process-global-numerical-lock"
_NO_FALLBACK_REASON = "none"
_CUDA_RUNTIME_UNAVAILABLE = "cuda-runtime-unavailable"
_CUDA_INDEX_UNAVAILABLE = "requested-cuda-index-unavailable"
_CUDA_PROBE_FAILED = "cuda-availability-probe-failed"
_EXPECTED_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "requested_device",
        "resolved_device",
        "requested_device_type",
        "resolved_device_type",
        "requested_cuda_index",
        "cuda_runtime_available",
        "cuda_device_count",
        "require_cuda",
        "allow_cpu_fallback",
        "cpu_fallback_used",
        "cpu_fallback_reason",
        "compute_dtype",
        "cpu_thread_policy",
        "cpu_thread_count",
        "deterministic_algorithms_required",
        "cuda_tf32_allowed",
        "cudnn_benchmark_allowed",
        "execution_settings_must_be_applied_by_caller",
        "profile_resolution_model_forward_calls",
        "profile_resolution_training_steps",
        "profile_resolution_backward_calls",
        "profile_resolution_optimizer_steps",
        "profile_resolution_filesystem_writes",
        "result_sha256",
    }
)


class AnalyticExecutionProfileError(ValueError):
    """A requested analytic execution policy is invalid or unavailable."""


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AnalyticExecutionProfileError(f"{label} must be boolean.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalyticExecutionProfileError(f"{label} must be a non-negative integer.")
    return value


def _normalized_requested_device(value: Any) -> torch.device:
    if not isinstance(value, (str, torch.device)):
        raise AnalyticExecutionProfileError(
            "requested_device must be an explicit CPU or indexed CUDA device."
        )
    if isinstance(value, str) and not value.strip():
        raise AnalyticExecutionProfileError("requested_device must not be empty.")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AnalyticExecutionProfileError(
            "requested_device must be an explicit CPU or indexed CUDA device."
        ) from exc
    if device.type == "cpu":
        if device.index is not None:
            raise AnalyticExecutionProfileError(
                "CPU analytic execution must be requested as exactly 'cpu'."
            )
        return torch.device("cpu")
    if device.type != "cuda" or device.index is None:
        raise AnalyticExecutionProfileError(
            "Analytic execution supports only 'cpu' or an explicit 'cuda:N'."
        )
    if device.index < 0:
        raise AnalyticExecutionProfileError(
            "The requested CUDA device index must be non-negative."
        )
    return torch.device("cuda", device.index)


def _cuda_availability() -> tuple[bool, int, bool]:
    """Return runtime availability, visible count, and whether probing failed."""

    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - backend failures vary by build/driver
        return False, 0, True
    if count < 0:  # pragma: no cover - defensive backend contract
        return False, 0, True
    return available, count, False


def _validated_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalyticExecutionProfileError("receipt must be a mapping.")
    receipt = dict(value)
    if set(receipt) != _EXPECTED_RECEIPT_KEYS:
        raise AnalyticExecutionProfileError(
            "Analytic execution profile receipt fields are incomplete or unknown."
        )
    if receipt.get("schema") != ANALYTIC_EXECUTION_PROFILE_SCHEMA:
        raise AnalyticExecutionProfileError(
            "Analytic execution profile receipt schema is invalid."
        )
    requested = _normalized_requested_device(receipt.get("requested_device"))
    resolved = _normalized_requested_device(receipt.get("resolved_device"))
    requested_type = receipt.get("requested_device_type")
    resolved_type = receipt.get("resolved_device_type")
    if requested_type != requested.type or resolved_type != resolved.type:
        raise AnalyticExecutionProfileError(
            "Analytic execution profile device types disagree with its devices."
        )
    requested_cuda_index = receipt.get("requested_cuda_index")
    if requested.type == "cuda":
        if requested_cuda_index != requested.index:
            raise AnalyticExecutionProfileError(
                "Analytic execution profile CUDA index is inconsistent."
            )
    elif requested_cuda_index is not None:
        raise AnalyticExecutionProfileError(
            "A CPU request cannot carry a requested CUDA index."
        )

    cuda_available = _boolean(
        receipt.get("cuda_runtime_available"),
        "cuda_runtime_available",
    )
    cuda_count = _nonnegative_int(
        receipt.get("cuda_device_count"),
        "cuda_device_count",
    )
    require_cuda = _boolean(receipt.get("require_cuda"), "require_cuda")
    allow_fallback = _boolean(
        receipt.get("allow_cpu_fallback"),
        "allow_cpu_fallback",
    )
    fallback_used = _boolean(
        receipt.get("cpu_fallback_used"),
        "cpu_fallback_used",
    )
    fallback_reason = receipt.get("cpu_fallback_reason")
    if require_cuda and allow_fallback:
        raise AnalyticExecutionProfileError(
            "require_cuda and allow_cpu_fallback are mutually exclusive."
        )
    if require_cuda and (
        requested.type != "cuda" or resolved.type != "cuda" or fallback_used
    ):
        raise AnalyticExecutionProfileError(
            "A require_cuda profile must resolve its CUDA request without fallback."
        )
    if requested.type == "cpu":
        if resolved.type != "cpu" or fallback_used:
            raise AnalyticExecutionProfileError(
                "An explicit CPU request cannot be represented as a fallback."
            )
    elif resolved.type == "cuda":
        if (
            fallback_used
            or fallback_reason != _NO_FALLBACK_REASON
            or not cuda_available
            or requested.index is None
            or requested.index >= cuda_count
            or resolved != requested
        ):
            raise AnalyticExecutionProfileError(
                "A resolved CUDA profile is inconsistent with CUDA availability."
            )
    elif (
        resolved.type != "cpu"
        or not fallback_used
        or not allow_fallback
        or require_cuda
        or fallback_reason
        not in {
            _CUDA_RUNTIME_UNAVAILABLE,
            _CUDA_INDEX_UNAVAILABLE,
            _CUDA_PROBE_FAILED,
        }
    ):
        raise AnalyticExecutionProfileError(
            "A CUDA-to-CPU fallback profile is internally inconsistent."
        )
    if not fallback_used and fallback_reason != _NO_FALLBACK_REASON:
        raise AnalyticExecutionProfileError(
            "A non-fallback profile must use the 'none' fallback reason."
        )

    fixed_values = {
        "compute_dtype": "float64",
        "cpu_thread_policy": _CPU_THREAD_POLICY,
        "cpu_thread_count": 1,
        "deterministic_algorithms_required": True,
        "cuda_tf32_allowed": False,
        "cudnn_benchmark_allowed": False,
        "execution_settings_must_be_applied_by_caller": True,
        "profile_resolution_model_forward_calls": 0,
        "profile_resolution_training_steps": 0,
        "profile_resolution_backward_calls": 0,
        "profile_resolution_optimizer_steps": 0,
        "profile_resolution_filesystem_writes": 0,
    }
    if any(receipt.get(key) != expected for key, expected in fixed_values.items()):
        raise AnalyticExecutionProfileError(
            "Analytic execution profile deterministic settings are invalid."
        )
    expected_result = receipt.get("result_sha256")
    body = {key: item for key, item in receipt.items() if key != "result_sha256"}
    if expected_result != canonical_sha256(body):
        raise AnalyticExecutionProfileError(
            "Analytic execution profile result seal is invalid."
        )
    return receipt


@dataclass(frozen=True, slots=True)
class AnalyticExecutionProfile:
    """An immutable, self-authenticating analytic device policy."""

    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt",
            MappingProxyType(_validated_receipt(self.receipt)),
        )

    @property
    def requested_device(self) -> str:
        return str(self.receipt["requested_device"])

    @property
    def resolved_device(self) -> str:
        return str(self.receipt["resolved_device"])

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.resolved_device)

    @property
    def require_cuda(self) -> bool:
        return bool(self.receipt["require_cuda"])

    @property
    def allow_cpu_fallback(self) -> bool:
        return bool(self.receipt["allow_cpu_fallback"])

    @property
    def cpu_fallback_used(self) -> bool:
        return bool(self.receipt["cpu_fallback_used"])

    @property
    def result_sha256(self) -> str:
        return str(self.receipt["result_sha256"])

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the sealed receipt."""

        return dict(self.receipt)


def resolve_analytic_execution_profile(
    requested_device: Any,
    *,
    require_cuda: bool = False,
    allow_cpu_fallback: bool = False,
) -> AnalyticExecutionProfile:
    """Resolve and seal one explicit CPU or indexed-CUDA analytic policy."""

    require_cuda = _boolean(require_cuda, "require_cuda")
    allow_cpu_fallback = _boolean(
        allow_cpu_fallback,
        "allow_cpu_fallback",
    )
    if require_cuda and allow_cpu_fallback:
        raise AnalyticExecutionProfileError(
            "require_cuda and allow_cpu_fallback are mutually exclusive."
        )
    requested = _normalized_requested_device(requested_device)
    if require_cuda and requested.type != "cuda":
        raise AnalyticExecutionProfileError(
            "require_cuda needs an explicit indexed CUDA request."
        )

    cuda_available, cuda_count, probe_failed = _cuda_availability()
    resolved = requested
    fallback_used = False
    fallback_reason = _NO_FALLBACK_REASON
    if requested.type == "cuda":
        if probe_failed:
            unavailable_reason = _CUDA_PROBE_FAILED
        elif not cuda_available:
            unavailable_reason = _CUDA_RUNTIME_UNAVAILABLE
        elif requested.index is None or requested.index >= cuda_count:
            unavailable_reason = _CUDA_INDEX_UNAVAILABLE
        else:
            unavailable_reason = None
        if unavailable_reason is not None:
            if require_cuda or not allow_cpu_fallback:
                raise AnalyticExecutionProfileError(
                    "The requested CUDA analytic device is unavailable and CPU "
                    "fallback was not explicitly permitted."
                )
            resolved = torch.device("cpu")
            fallback_used = True
            fallback_reason = unavailable_reason

    body: dict[str, Any] = {
        "schema": ANALYTIC_EXECUTION_PROFILE_SCHEMA,
        "requested_device": str(requested),
        "resolved_device": str(resolved),
        "requested_device_type": requested.type,
        "resolved_device_type": resolved.type,
        "requested_cuda_index": (requested.index if requested.type == "cuda" else None),
        "cuda_runtime_available": cuda_available,
        "cuda_device_count": cuda_count,
        "require_cuda": require_cuda,
        "allow_cpu_fallback": allow_cpu_fallback,
        "cpu_fallback_used": fallback_used,
        "cpu_fallback_reason": fallback_reason,
        "compute_dtype": "float64",
        "cpu_thread_policy": _CPU_THREAD_POLICY,
        "cpu_thread_count": 1,
        "deterministic_algorithms_required": True,
        "cuda_tf32_allowed": False,
        "cudnn_benchmark_allowed": False,
        "execution_settings_must_be_applied_by_caller": True,
        "profile_resolution_model_forward_calls": 0,
        "profile_resolution_training_steps": 0,
        "profile_resolution_backward_calls": 0,
        "profile_resolution_optimizer_steps": 0,
        "profile_resolution_filesystem_writes": 0,
    }
    return AnalyticExecutionProfile(
        receipt={**body, "result_sha256": canonical_sha256(body)}
    )


__all__ = [
    "ANALYTIC_EXECUTION_PROFILE_SCHEMA",
    "AnalyticExecutionProfile",
    "AnalyticExecutionProfileError",
    "resolve_analytic_execution_profile",
]
