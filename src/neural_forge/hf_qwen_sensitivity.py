"""Bounded held-out sensitivity evidence for Qwen3 model surgery.

The collector in this module performs inference-only interventions.  It does
not fit weights, update assignments, run backward, or perform recovery.  A
neuron intervention zeros one real post-SwiGLU channel.  An attention
intervention zeros the matching Q, K, V, and O slices for one GQA group in the
same forward transaction.

The reported residual and logit vectors are count sketches of directional
differences between an untouched calibration forward and a zero-ablation
forward. They are generic intervention evidence, never JVPs, VJPs, Jacobians,
duals, or claims about infinitesimal derivatives.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import operator
import os
import platform
import sys
import threading
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neural_forge.adaptive_surgery_graph import AnchorSpec
from neural_forge.hf_neuron_codebook import NeuronCodebookLayerView
from neural_forge.hf_qwen_adaptive_surgery import Qwen3AdaptiveSurgeryFeature


SENSITIVITY_SCHEMA = "neural-forge-qwen3-sensitivity-v3"
DIRECTIONAL_DIFFERENCE_METHOD = "zero_ablation_directional_difference_count_sketch"
FINITE_DIFFERENCE_METHOD = DIRECTIONAL_DIFFERENCE_METHOD
_ATTENTION_ROLES = ("q", "k", "v", "o")
_MEBIBYTE = 1024 * 1024
_RSS_SAMPLE_INTERVAL_SECONDS = 0.001
_MLP_KERNEL_WORKSPACE_BOUND_BYTES = 32 * _MEBIBYTE
_ORIGINAL_TENSOR_UNTYPED_STORAGE = Tensor.untyped_storage


def _tensor_storage(value: Tensor) -> Any:
    """Use the captured read-only storage accessor inside guarded callbacks."""

    return _ORIGINAL_TENSOR_UNTYPED_STORAGE(value)


def _tensor_version(value: Tensor) -> int:
    """Return a stable sentinel for inference tensors without version counters."""

    try:
        return int(value._version)
    except RuntimeError:
        return -1


class Qwen3SensitivityError(ValueError):
    """Base error for invalid or unsafe sensitivity collection."""


class Qwen3SensitivityShapeError(Qwen3SensitivityError):
    """A model, batch, candidate, or hook result has an invalid shape."""


class Qwen3SensitivityLimitError(Qwen3SensitivityError):
    """A hard row, device, candidate, or collector-memory cap was exceeded."""


class Qwen3SensitivitySplitError(Qwen3SensitivityError):
    """Calibration and held-out gate rows are not strictly disjoint."""


class Qwen3SensitivityStateError(Qwen3SensitivityError):
    """A forward intervention did not leave the model hook state intact."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3SensitivityShapeError(f"{name} must be a positive integer.")
    try:
        normalized = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3SensitivityShapeError(f"{name} must be a positive integer.") from exc
    if normalized <= 0:
        raise Qwen3SensitivityShapeError(f"{name} must be a positive integer.")
    return normalized


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3SensitivityShapeError(f"{name} must be a non-negative integer.")
    try:
        normalized = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3SensitivityShapeError(
            f"{name} must be a non-negative integer."
        ) from exc
    if normalized < 0:
        raise Qwen3SensitivityShapeError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityLimits:
    """Hard bounds for collector-owned work and materialized forward rows."""

    max_calibration_rows: int = 32
    max_gate_rows: int = 256
    max_sequence_length: int = 4096
    max_score_rows: int = 128
    max_neuron_candidates: int = 512
    max_attention_candidates: int = 256
    candidate_chunk_size: int = 8
    max_candidate_forward_rows: int = 256
    max_model_devices: int = 1
    max_collector_bytes: int = 512 * _MEBIBYTE
    max_fingerprint_chunk_bytes: int = 8 * _MEBIBYTE
    max_sketch_work_bytes: int = 8 * _MEBIBYTE
    max_sketch_dim: int = 64
    max_codebook_channels_per_chunk: int = 128
    max_process_rss_increase_bytes: int | None = None
    max_cuda_allocated_increase_bytes: int | None = None
    max_cuda_reserved_increase_bytes: int | None = None
    allowed_device_types: tuple[str, ...] = ("cpu", "cuda")

    def __post_init__(self) -> None:
        integer_fields = (
            "max_calibration_rows",
            "max_gate_rows",
            "max_sequence_length",
            "max_score_rows",
            "max_neuron_candidates",
            "max_attention_candidates",
            "candidate_chunk_size",
            "max_candidate_forward_rows",
            "max_model_devices",
            "max_collector_bytes",
            "max_fingerprint_chunk_bytes",
            "max_sketch_work_bytes",
            "max_sketch_dim",
            "max_codebook_channels_per_chunk",
        )
        for name in integer_fields:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        device_types = tuple(
            str(item).strip().lower() for item in self.allowed_device_types
        )
        if not device_types or any(not item for item in device_types):
            raise Qwen3SensitivityShapeError(
                "allowed_device_types must contain at least one device type."
            )
        object.__setattr__(self, "allowed_device_types", device_types)
        if self.max_fingerprint_chunk_bytes > self.max_collector_bytes:
            raise Qwen3SensitivityLimitError(
                "max_fingerprint_chunk_bytes cannot exceed max_collector_bytes."
            )
        if self.max_sketch_work_bytes > self.max_collector_bytes:
            raise Qwen3SensitivityLimitError(
                "max_sketch_work_bytes cannot exceed max_collector_bytes."
            )
        for name in (
            "max_process_rss_increase_bytes",
            "max_cuda_allocated_increase_bytes",
            "max_cuda_reserved_increase_bytes",
        ):
            value = getattr(self, name)
            if value is None:
                value = self.max_collector_bytes
            object.__setattr__(self, name, _positive_int(value, name))


@dataclass(frozen=True, order=True, slots=True)
class Qwen3NeuronRef:
    """One logical post-SwiGLU neuron execution."""

    layer_index: int
    neuron_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_index",
            _nonnegative_int(self.layer_index, "layer_index"),
        )
        object.__setattr__(
            self,
            "neuron_index",
            _nonnegative_int(self.neuron_index, "neuron_index"),
        )


@dataclass(frozen=True, order=True, slots=True)
class Qwen3AttentionGroupRef:
    """One logical coupled Q/K/V/O GQA-group execution."""

    layer_index: int
    group_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_index",
            _nonnegative_int(self.layer_index, "layer_index"),
        )
        object.__setattr__(
            self,
            "group_index",
            _nonnegative_int(self.group_index, "group_index"),
        )


@dataclass(frozen=True, slots=True)
class Qwen3StateTensorFingerprint:
    """Exact tensor digest plus same-process mutation provenance."""

    name: str
    sha256: str
    dtype: str
    shape: tuple[int, ...]
    device: str
    data_ptr: int
    version: int
    stride: tuple[int, ...]
    storage_offset: int
    storage_nbytes: int
    requires_grad: bool


@dataclass(frozen=True, slots=True)
class Qwen3ModelFingerprintManifest:
    """Merkle-style exact state manifest supporting bounded accepted refreshes."""

    root: str
    config_sha256: str
    tensors: tuple[Qwen3StateTensorFingerprint, ...]
    mode: Literal["full", "incremental"]
    parent_root: str | None = None
    accepted_transaction_sha256: str | None = None
    changed_state_prefixes: tuple[str, ...] = ()
    tensors_hashed: int = 0
    tensors_reused: int = 0


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityFingerprints:
    """Exact model and positioned-token fingerprints for cache isolation."""

    model: str
    calibration: str
    gate: str
    runtime: str
    measurement: str
    cache: str
    calibration_row_fingerprints: tuple[str, ...]
    gate_row_fingerprints: tuple[str, ...]
    model_manifest: Qwen3ModelFingerprintManifest
    model_hash_method: str = "sha256_merkle_exact_tensor_state_v4_full_rehash"
    row_hash_method: str = "sha256_active_token_ids_and_position_ids"
    runtime_hash_method: str = "sha256_fail_closed_execution_environment_v1"
    sketch_basis: str = ""
    sketch_basis_hash_method: str = "sha256_semantic_measurement_basis_v1"


@dataclass(frozen=True, slots=True)
class Qwen3NeuronSensitivity:
    """Measured held-out damage from zeroing one post-SwiGLU channel."""

    node: Qwen3NeuronRef
    activation_energy: float
    residual_mse: float
    residual_relative_l2: float
    logit_mse: float
    logit_relative_l2: float
    logit_kl: float
    top1_disagreement: float
    damage_score: float
    residual_ablation_difference_sketch: tuple[float, ...]
    logit_ablation_difference_sketch: tuple[float, ...]
    evidence_method: str = DIRECTIONAL_DIFFERENCE_METHOD


@dataclass(frozen=True, slots=True)
class Qwen3AttentionGroupSensitivity:
    """Measured damage from one atomic coupled Q/K/V/O group ablation."""

    node: Qwen3AttentionGroupRef
    role_activation_energy: tuple[float, float, float, float]
    residual_mse: float
    residual_relative_l2: float
    logit_mse: float
    logit_relative_l2: float
    logit_kl: float
    top1_disagreement: float
    damage_score: float
    residual_ablation_difference_sketch: tuple[float, ...]
    logit_ablation_difference_sketch: tuple[float, ...]
    atomic_projection_roles: tuple[str, str, str, str] = _ATTENTION_ROLES
    role_ablation_call_counts: tuple[int, int, int, int] = (1, 1, 1, 1)
    evidence_method: str = DIRECTIONAL_DIFFERENCE_METHOD


@dataclass(frozen=True, slots=True)
class Qwen3ExactAnchorMasks:
    """Immutable exact-anchor masks over the complete logical Qwen layout."""

    neuron_mask: tuple[tuple[bool, ...], ...]
    attention_group_mask: tuple[tuple[bool, ...], ...]
    explicit_neurons: frozenset[Qwen3NeuronRef]
    measured_high_damage_neurons: frozenset[Qwen3NeuronRef]
    explicit_attention_groups: frozenset[Qwen3AttentionGroupRef]
    measured_high_damage_attention_groups: frozenset[Qwen3AttentionGroupRef]

    @property
    def exact_neurons(self) -> frozenset[Qwen3NeuronRef]:
        return self.explicit_neurons | self.measured_high_damage_neurons

    @property
    def exact_attention_groups(self) -> frozenset[Qwen3AttentionGroupRef]:
        return (
            self.explicit_attention_groups | self.measured_high_damage_attention_groups
        )

    def is_neuron_exact(self, layer_index: int, neuron_index: int) -> bool:
        return self.neuron_mask[layer_index][neuron_index]

    def is_attention_group_exact(self, layer_index: int, group_index: int) -> bool:
        return self.attention_group_mask[layer_index][group_index]


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityReport:
    """Audit record proving bounded inference-only collection."""

    schema: str
    logical_layers: int
    intermediate_width: int
    hidden_size: int
    key_value_groups: int
    calibration_rows: int
    gate_rows: int
    sequence_length: int
    scored_positions_per_row: int
    neuron_candidates: int
    attention_group_candidates: int
    baseline_forward_passes: int
    candidate_forward_passes: int
    maximum_candidate_chunk: int
    tracked_peak_collector_bytes: int
    collector_byte_limit: int
    process_rss_start_bytes: int
    process_rss_peak_bytes: int
    process_rss_peak_increase_bytes: int
    process_rss_increase_limit: int
    process_rss_sample_interval_seconds: float
    process_rss_sampler_stopped: bool
    cuda_allocated_start_bytes: int
    cuda_allocated_peak_bytes: int
    cuda_allocated_peak_increase_bytes: int
    cuda_allocated_increase_limit: int
    cuda_reserved_start_bytes: int
    cuda_reserved_peak_bytes: int
    cuda_reserved_peak_increase_bytes: int
    cuda_reserved_increase_limit: int
    resource_checkpoints: int
    attention_backend: str
    attention_workspace_bound_bytes: int
    dense_mlp_forward_bound_bytes: int
    codebook_materialization_bound_bytes: int
    mlp_forward_workspace_bound_bytes: int
    model_devices: tuple[str, ...]
    mlp_capture_methods: tuple[str, ...]
    evidence_method: str
    exact_autodiff_used: bool
    directional_difference_reason: str
    model_fingerprint_mode: str
    model_tensors_hashed: int
    model_tensors_reused: int
    runtime_fingerprint: str
    backward_calls: int
    optimizer_steps: int
    training_steps: int
    recovery_steps: int
    model_weights_mutated: bool
    hooks_restored: bool


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityEvidence:
    """Immutable evidence bundle consumed by selective surgery planning."""

    fingerprints: Qwen3SensitivityFingerprints
    neuron_activation_energy: tuple[tuple[float, ...], ...]
    attention_role_activation_energy: tuple[
        tuple[tuple[float, float, float, float], ...], ...
    ]
    neurons: tuple[Qwen3NeuronSensitivity, ...]
    attention_groups: tuple[Qwen3AttentionGroupSensitivity, ...]
    anchors: Qwen3ExactAnchorMasks
    report: Qwen3SensitivityReport

    def to_adaptive_surgery_features(
        self,
    ) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
        """Translate measured neuron evidence into the existing Qwen bridge API."""

        result: list[Qwen3AdaptiveSurgeryFeature] = []
        for item in self.neurons:
            layer_context = tuple(
                value
                for group in self.attention_role_activation_energy[
                    item.node.layer_index
                ]
                for value in group
            )
            result.append(
                Qwen3AdaptiveSurgeryFeature(
                    item.node.layer_index,
                    item.node.neuron_index,
                    activation=[
                        item.activation_energy,
                        item.residual_relative_l2,
                        item.logit_relative_l2,
                        item.logit_kl,
                        item.top1_disagreement,
                    ],
                    jacobian=None,
                    dual=None,
                    domain_features=(
                        layer_context
                        + item.residual_ablation_difference_sketch
                        + item.logit_ablation_difference_sketch
                    )
                    or None,
                    importance=max(item.damage_score, 1e-12),
                    anchor=(
                        AnchorSpec.pinned()
                        if self.anchors.is_neuron_exact(
                            item.node.layer_index,
                            item.node.neuron_index,
                        )
                        else None
                    ),
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class _Batch:
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    row_fingerprints: tuple[str, ...]
    content_fingerprints: tuple[str, ...]
    fingerprint: str

    @property
    def rows(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass(frozen=True, slots=True)
class _Layout:
    layers: tuple[nn.Module, ...]
    mlp_modes: tuple[Literal["dense", "codebook"], ...]
    intermediate_width: int
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    vocabulary_size: int
    input_device: torch.device
    activation_element_bytes: int
    model_devices: tuple[str, ...]
    attention_backend: str

    @property
    def query_rows_per_group(self) -> int:
        return (self.query_heads // self.key_value_heads) * self.head_dim


@dataclass(slots=True)
class _Baseline:
    logits: Tensor
    residuals: dict[int, Tensor]
    neuron_energy: tuple[tuple[float, ...], ...]
    attention_energy: tuple[tuple[tuple[float, float, float, float], ...], ...]
    capture_methods: tuple[str, ...]


class _HookScope:
    def __init__(self) -> None:
        self._handles: list[Any] = []

    def add(self, handle: Any) -> None:
        self._handles.append(handle)

    def __enter__(self) -> _HookScope:
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_process_memory() -> tuple[int, int]:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise Qwen3SensitivityStateError("GetProcessMemoryInfo failed.")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _process_rss_bytes() -> int:
    """Return current resident bytes without making psutil a hard dependency."""

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        pass
    if os.name == "posix" and os.path.exists("/proc/self/statm"):
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            raise Qwen3SensitivityStateError("/proc/self/statm lacks an RSS field.")
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    if os.name == "nt":
        return _windows_process_memory()[0]
    raise Qwen3SensitivityStateError(
        "Current process RSS is unavailable; collection fails closed."
    )


def _process_peak_rss_bytes() -> int:
    """Return the operating system's process-lifetime peak resident bytes."""

    if os.name == "nt":
        return _windows_process_memory()[1]
    if os.name == "posix" and os.path.exists("/proc/self/status"):
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return int(fields[1]) * 1024
    if os.name == "posix":
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    raise Qwen3SensitivityStateError(
        "Peak process RSS is unavailable; collection fails closed."
    )


def _cuda_devices(model_devices: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(torch.device(item))
                for item in model_devices
                if torch.device(item).type == "cuda"
            }
        )
    )


def _reset_cuda_peak_stats(devices: Sequence[str]) -> None:
    for name in devices:
        torch.cuda.reset_peak_memory_stats(torch.device(name))


def _cuda_memory_snapshot(
    devices: Sequence[str],
) -> tuple[int, int, int, int]:
    allocated = reserved = allocated_peak = reserved_peak = 0
    for name in devices:
        device = torch.device(name)
        allocated += int(torch.cuda.memory_allocated(device))
        reserved += int(torch.cuda.memory_reserved(device))
        allocated_peak += int(torch.cuda.max_memory_allocated(device))
        reserved_peak += int(torch.cuda.max_memory_reserved(device))
    return allocated, reserved, allocated_peak, reserved_peak


class _ResourceGuard:
    """Continuously sample process RSS and retain allocator-reported CUDA peaks."""

    def __init__(
        self,
        limits: Qwen3SensitivityLimits,
        model_devices: Sequence[str] | None = None,
    ) -> None:
        self.limits = limits
        self.cuda_devices: tuple[str, ...] = ()
        self._cuda_bound = False
        self.process_rss_start = _process_rss_bytes()
        self.process_lifetime_peak_start = _process_peak_rss_bytes()
        self.process_rss_peak = self.process_rss_start
        self.sample_interval_seconds = _RSS_SAMPLE_INTERVAL_SECONDS
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler_error: BaseException | None = None
        self._process_violation: str | None = None
        self._cuda_violation: str | None = None
        self.stopped = False
        self.cuda_allocated_start = 0
        self.cuda_reserved_start = 0
        self.cuda_allocated_peak = 0
        self.cuda_reserved_peak = 0
        self.checkpoints = 0
        self._sampler_thread = threading.Thread(
            target=self._sample_loop,
            name="qwen3-sensitivity-rss-sampler",
            daemon=True,
        )
        self._sampler_thread.start()
        try:
            self.checkpoint("resource_guard_start")
            if model_devices is not None:
                self.bind_cuda_devices(model_devices)
        except BaseException:
            self._stop_sampler()
            raise

    def bind_cuda_devices(self, model_devices: Sequence[str]) -> None:
        """Bind CUDA accounting after the process sampler is already live."""

        if self.stopped:
            raise Qwen3SensitivityStateError(
                "Cannot bind CUDA devices after the resource guard has stopped."
            )
        if self._cuda_bound:
            raise Qwen3SensitivityStateError(
                "CUDA devices can be bound to a resource guard only once."
            )
        self.cuda_devices = _cuda_devices(model_devices)
        if self.cuda_devices:
            _reset_cuda_peak_stats(self.cuda_devices)
            (
                self.cuda_allocated_start,
                self.cuda_reserved_start,
                self.cuda_allocated_peak,
                self.cuda_reserved_peak,
            ) = _cuda_memory_snapshot(self.cuda_devices)
        self._cuda_bound = True
        self.checkpoint("resource_guard_cuda_bound")

    @property
    def process_rss_peak_increase(self) -> int:
        return max(0, self.process_rss_peak - self.process_rss_start)

    @property
    def cuda_allocated_peak_increase(self) -> int:
        return max(0, self.cuda_allocated_peak - self.cuda_allocated_start)

    @property
    def cuda_reserved_peak_increase(self) -> int:
        return max(0, self.cuda_reserved_peak - self.cuda_reserved_start)

    def _sample_process(self) -> None:
        rss = _process_rss_bytes()
        lifetime_peak = _process_peak_rss_bytes()
        new_lifetime_peak = (
            lifetime_peak if lifetime_peak > self.process_lifetime_peak_start else 0
        )
        with self._lock:
            self.process_rss_peak = max(
                self.process_rss_peak,
                rss,
                new_lifetime_peak,
            )
            increase = self.process_rss_peak_increase
            maximum = self.limits.max_process_rss_increase_bytes
            if increase > maximum and self._process_violation is None:
                self._process_violation = (
                    f"process RSS increase {increase} exceeded hard cap {maximum}"
                )

    def _sample_loop(self) -> None:
        try:
            while not self._stop_event.wait(self.sample_interval_seconds):
                self._sample_process()
        except BaseException as exc:  # pragma: no cover - platform failure path
            with self._lock:
                self._sampler_error = exc

    def _raise_guard_failure(self, label: str) -> None:
        with self._lock:
            sampler_error = self._sampler_error
            process_violation = self._process_violation
            cuda_violation = self._cuda_violation
        if sampler_error is not None:
            raise Qwen3SensitivityStateError(
                f"Process RSS sampler failed at {label!r}."
            ) from sampler_error
        violation = process_violation or cuda_violation
        if violation is not None:
            raise Qwen3SensitivityLimitError(f"{violation} at checkpoint {label!r}.")

    def checkpoint(self, label: str) -> None:
        self.checkpoints += 1
        self._sample_process()
        self._raise_guard_failure(label)
        if self.cuda_devices:
            current_allocated, current_reserved, peak_allocated, peak_reserved = (
                _cuda_memory_snapshot(self.cuda_devices)
            )
        else:
            current_allocated = current_reserved = 0
            peak_allocated = peak_reserved = 0
        self.cuda_allocated_peak = max(
            self.cuda_allocated_peak,
            current_allocated,
            peak_allocated,
        )
        self.cuda_reserved_peak = max(
            self.cuda_reserved_peak,
            current_reserved,
            peak_reserved,
        )
        checks = (
            (
                "CUDA allocated",
                self.cuda_allocated_peak_increase,
                self.limits.max_cuda_allocated_increase_bytes,
            ),
            (
                "CUDA reserved",
                self.cuda_reserved_peak_increase,
                self.limits.max_cuda_reserved_increase_bytes,
            ),
        )
        for resource, increase, maximum in checks:
            if increase > maximum:
                with self._lock:
                    if self._cuda_violation is None:
                        self._cuda_violation = (
                            f"{resource} increase {increase} exceeded hard cap "
                            f"{maximum}"
                        )
                self._raise_guard_failure(label)

    def _stop_sampler(self) -> None:
        if self.stopped:
            return
        self._stop_event.set()
        self._sampler_thread.join(timeout=2.0)
        if self._sampler_thread.is_alive():
            raise Qwen3SensitivityStateError(
                "Process RSS sampler did not stop within its lifecycle bound."
            )
        self.stopped = True

    def close(self) -> None:
        if self.stopped:
            self._raise_guard_failure("resource_guard_close")
            return
        self._stop_sampler()
        self._sample_process()
        self.checkpoint("resource_guard_close")

    def __enter__(self) -> _ResourceGuard:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash_tensor(
    digest: Any,
    value: Tensor,
    chunk_bytes: int,
    checkpoint: Any = None,
) -> None:
    if value.device.type == "meta":
        raise Qwen3SensitivityShapeError("Model state cannot contain meta tensors.")
    if value.layout != torch.strided:
        raise Qwen3SensitivityShapeError(
            "Exact model fingerprints require strided state tensors."
        )
    element_size = max(1, int(value.element_size()))
    chunk_elements = max(1, chunk_bytes // element_size)

    def update(chunk: Tensor) -> None:
        cpu = chunk.detach().to("cpu").contiguous()
        byte_view = cpu.view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
        del byte_view, cpu
        if checkpoint is not None:
            checkpoint("model_fingerprint_chunk")

    if value.is_contiguous():
        flat = value.detach().reshape(-1)
        for start in range(0, int(flat.numel()), chunk_elements):
            update(flat[start : start + chunk_elements])
        return

    def update_noncontiguous(chunk: Tensor) -> None:
        if int(chunk.numel()) <= chunk_elements or chunk.ndim == 0:
            update(chunk)
            return
        trailing = math.prod(int(item) for item in chunk.shape[1:])
        if trailing <= chunk_elements:
            rows = max(1, chunk_elements // max(1, trailing))
            for start in range(0, int(chunk.shape[0]), rows):
                update(chunk[start : start + rows])
            return
        for row in chunk.unbind(0):
            update_noncontiguous(row)

    update_noncontiguous(value)


def _sha256_string(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Qwen3SensitivityShapeError(
            f"{name} must be one lowercase SHA-256 hex digest."
        )
    return normalized


def _model_state_tensors(model: nn.Module) -> dict[str, Tensor]:
    """Read registered tensor state directly, without state-dict hook rewriting."""

    state: dict[str, Tensor] = {}

    def visit(module: nn.Module, prefix: str, ancestors: frozenset[int]) -> None:
        identity = id(module)
        if identity in ancestors:
            raise Qwen3SensitivityStateError(
                "Model module registration contains a cycle."
            )
        lineage = ancestors | {identity}
        for name, value in module._parameters.items():
            if value is None:
                continue
            qualified = f"{prefix}{name}"
            if qualified in state or not isinstance(value, Tensor):
                raise Qwen3SensitivityStateError(
                    f"Invalid or duplicate registered parameter {qualified!r}."
                )
            state[qualified] = value
        non_persistent = module._non_persistent_buffers_set
        for name, value in module._buffers.items():
            if value is None or name in non_persistent:
                continue
            qualified = f"{prefix}{name}"
            if qualified in state or not isinstance(value, Tensor):
                raise Qwen3SensitivityStateError(
                    f"Invalid or duplicate registered buffer {qualified!r}."
                )
            state[qualified] = value
        for name, child in module._modules.items():
            if child is not None:
                visit(child, f"{prefix}{name}.", lineage)

    visit(model, "", frozenset())
    if not state:
        raise Qwen3SensitivityShapeError(
            "Exact Qwen fingerprints require registered tensor state."
        )
    return state


def _state_tensor_fingerprint(
    name: str,
    value: Tensor,
    chunk_bytes: int,
    checkpoint: Any,
) -> Qwen3StateTensorFingerprint:
    digest = hashlib.sha256()
    _hash_tensor(digest, value, chunk_bytes, checkpoint)
    return Qwen3StateTensorFingerprint(
        name=name,
        sha256=digest.hexdigest(),
        dtype=str(value.dtype),
        shape=tuple(int(item) for item in value.shape),
        device=str(value.device),
        data_ptr=int(_tensor_storage(value).data_ptr()) if int(value.numel()) else 0,
        version=_tensor_version(value),
        stride=tuple(int(item) for item in value.stride()),
        storage_offset=int(value.storage_offset()),
        storage_nbytes=int(_tensor_storage(value).nbytes()),
        requires_grad=bool(value.requires_grad),
    )


def _model_fingerprint_root(
    config_sha256: str,
    tensors: Sequence[Qwen3StateTensorFingerprint],
) -> str:
    return _sha256_json(
        {
            "schema": SENSITIVITY_SCHEMA,
            "config_sha256": config_sha256,
            "tensors": [
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "dtype": item.dtype,
                    "shape": item.shape,
                    "stride": item.stride,
                    "storage_offset": item.storage_offset,
                    "storage_nbytes": item.storage_nbytes,
                    "requires_grad": item.requires_grad,
                }
                for item in tensors
            ],
        }
    )


def fingerprint_qwen3_model_manifest(
    model: nn.Module,
    *,
    max_chunk_bytes: int = 8 * _MEBIBYTE,
    parent: Qwen3ModelFingerprintManifest | None = None,
    changed_state_prefixes: Sequence[str] = (),
    accepted_transaction_sha256: str | None = None,
    checkpoint: Any = None,
) -> Qwen3ModelFingerprintManifest:
    """Build a full exact manifest or an accepted-change incremental manifest.

    Incremental mode is deliberately same-process only. Every tensor is hashed
    again: declared scopes may change, while every undeclared digest and storage
    identity must remain exact. Pointer and ``Tensor._version`` checks are only
    supplemental provenance because raw-storage writes can bypass both.
    """

    chunk_bytes = _positive_int(max_chunk_bytes, "max_chunk_bytes")
    config = getattr(model, "config", None)
    if config is None:
        raise Qwen3SensitivityShapeError(
            "Exact Qwen fingerprints require model.config."
        )
    config_payload = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    config_sha256 = hashlib.sha256(_canonical_json(config_payload)).hexdigest()
    state = _model_state_tensors(model)
    names = tuple(sorted(state))
    if any(not isinstance(state[name], Tensor) for name in names):
        raise Qwen3SensitivityShapeError("Every model state entry must be a tensor.")

    if parent is None:
        if changed_state_prefixes or accepted_transaction_sha256 is not None:
            raise Qwen3SensitivityShapeError(
                "Changed scopes and transaction provenance require a parent manifest."
            )
        tensors = tuple(
            _state_tensor_fingerprint(name, state[name], chunk_bytes, checkpoint)
            for name in names
        )
        return Qwen3ModelFingerprintManifest(
            root=_model_fingerprint_root(config_sha256, tensors),
            config_sha256=config_sha256,
            tensors=tensors,
            mode="full",
            tensors_hashed=len(tensors),
        )

    if not isinstance(parent, Qwen3ModelFingerprintManifest):
        raise Qwen3SensitivityShapeError(
            "parent must be a Qwen3ModelFingerprintManifest."
        )
    _sha256_string(parent.root, "parent.root")
    _sha256_string(parent.config_sha256, "parent.config_sha256")
    for item in parent.tensors:
        _sha256_string(item.sha256, f"parent tensor {item.name!r} sha256")
    if _model_fingerprint_root(parent.config_sha256, parent.tensors) != parent.root:
        raise Qwen3SensitivityStateError(
            "Incremental fingerprint refused an invalid parent manifest root."
        )
    transaction = _sha256_string(
        accepted_transaction_sha256,
        "accepted_transaction_sha256",
    )
    prefixes = tuple(sorted({str(item).strip() for item in changed_state_prefixes}))
    if not prefixes or any(not item or "*" in item for item in prefixes):
        raise Qwen3SensitivityShapeError(
            "Incremental fingerprints require non-empty literal state prefixes."
        )
    if config_sha256 != parent.config_sha256:
        raise Qwen3SensitivityStateError(
            "Incremental fingerprint refused a changed model config."
        )
    previous = {item.name: item for item in parent.tensors}
    if len(previous) != len(parent.tensors) or tuple(sorted(previous)) != names:
        raise Qwen3SensitivityStateError(
            "Incremental fingerprint refused a changed or duplicate state manifest."
        )
    matched = {prefix: 0 for prefix in prefixes}
    directly_selected: set[str] = set()
    for name in names:
        for prefix in prefixes:
            if name.startswith(prefix):
                matched[prefix] += 1
                directly_selected.add(name)
    selected_storage = {
        int(_tensor_storage(state[name]).data_ptr())
        for name in directly_selected
        if int(state[name].numel())
    }
    tensors_list: list[Qwen3StateTensorFingerprint] = []
    hashed = 0
    reused = 0
    for name in names:
        value = state[name]
        data_ptr = int(_tensor_storage(value).data_ptr()) if int(value.numel()) else 0
        selected = name in directly_selected or (
            bool(data_ptr) and data_ptr in selected_storage
        )
        old = previous[name]
        current_fingerprint = _state_tensor_fingerprint(
            name,
            value,
            chunk_bytes,
            checkpoint,
        )
        hashed += 1
        if selected:
            tensors_list.append(current_fingerprint)
            continue
        if current_fingerprint != old:
            raise Qwen3SensitivityStateError(
                f"Undeclared state mutation or replacement detected for {name!r}."
            )
        tensors_list.append(current_fingerprint)
        reused += 1
    unmatched = [prefix for prefix, count in matched.items() if count == 0]
    if unmatched:
        raise Qwen3SensitivityShapeError(
            f"Changed state prefixes matched no tensors: {unmatched!r}."
        )
    tensors = tuple(tensors_list)
    return Qwen3ModelFingerprintManifest(
        root=_model_fingerprint_root(config_sha256, tensors),
        config_sha256=config_sha256,
        tensors=tensors,
        mode="incremental",
        parent_root=parent.root,
        accepted_transaction_sha256=transaction,
        changed_state_prefixes=prefixes,
        tensors_hashed=hashed,
        tensors_reused=reused,
    )


def fingerprint_qwen3_model(
    model: nn.Module,
    *,
    max_chunk_bytes: int = 8 * _MEBIBYTE,
) -> str:
    """Return the full exact Merkle root for Qwen config and tensor state."""

    return fingerprint_qwen3_model_manifest(
        model,
        max_chunk_bytes=max_chunk_bytes,
    ).root


def _verify_model_state_provenance(
    model: nn.Module,
    manifest: Qwen3ModelFingerprintManifest,
    *,
    max_chunk_bytes: int,
    checkpoint: Any = None,
) -> None:
    final = fingerprint_qwen3_model_manifest(
        model,
        max_chunk_bytes=max_chunk_bytes,
        checkpoint=checkpoint,
    )
    if final.root != manifest.root or final.tensors != manifest.tensors:
        raise Qwen3SensitivityStateError(
            "Sensitivity collection mutated or replaced model state; the exact "
            "post-forward tensor rehash differs from the initial manifest."
        )


def _finalize_model_integrity(
    model: nn.Module,
    manifest: Qwen3ModelFingerprintManifest,
    inventory: tuple[tuple[str, str, int, int, str], ...],
    *,
    max_chunk_bytes: int,
    guard: _ResourceGuard,
) -> bool:
    """Rehash every tensor and verify exact hook identity on every exit path."""

    provenance_error: Exception | None = None
    try:
        _verify_model_state_provenance(
            model,
            manifest,
            max_chunk_bytes=max_chunk_bytes,
            checkpoint=guard.checkpoint,
        )
    except Exception as exc:
        provenance_error = exc
    final_inventory = _hook_inventory(model)
    hooks_restored = final_inventory == inventory
    if not hooks_restored:
        removed = tuple(item for item in inventory if item not in final_inventory)
        added = tuple(item for item in final_inventory if item not in inventory)
        message = (
            "Sensitivity collection changed the model hook inventory; "
            f"removed={removed!r}, added={added!r}."
        )
        if provenance_error is not None:
            message = (
                "Sensitivity collection changed both exact model state and the "
                f"model hook inventory; removed={removed!r}, added={added!r}."
            )
        raise Qwen3SensitivityStateError(message) from provenance_error
    if provenance_error is not None:
        raise provenance_error
    guard.checkpoint("final_model_rehash_complete")
    return True


def _tensor_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to("cpu").contiguous()
    return memoryview(cpu.view(torch.uint8).numpy()).tobytes()


def _batch(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    position_ids: Tensor | None,
    *,
    split_name: str,
    max_rows: int,
    limits: Qwen3SensitivityLimits,
) -> _Batch:
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise Qwen3SensitivityShapeError(
            f"{split_name}_input_ids must be a rank-2 tensor."
        )
    if input_ids.dtype == torch.bool or input_ids.is_floating_point():
        raise Qwen3SensitivityShapeError(
            f"{split_name}_input_ids must use an integer dtype."
        )
    ids = input_ids.detach().to(device="cpu", dtype=torch.long).clone()
    rows, sequence_length = map(int, ids.shape)
    if rows <= 0 or sequence_length <= 0:
        raise Qwen3SensitivityShapeError(f"{split_name} cannot be empty.")
    if rows > max_rows:
        raise Qwen3SensitivityLimitError(
            f"{split_name} has {rows} rows, exceeding its hard cap {max_rows}."
        )
    if sequence_length > limits.max_sequence_length:
        raise Qwen3SensitivityLimitError(
            f"{split_name} sequence length {sequence_length} exceeds "
            f"{limits.max_sequence_length}."
        )
    if bool((ids < 0).any()):
        raise Qwen3SensitivityShapeError(
            f"{split_name}_input_ids cannot contain negative token IDs."
        )
    if attention_mask is None:
        mask = torch.ones_like(ids, dtype=torch.bool)
    else:
        if not isinstance(attention_mask, Tensor) or tuple(attention_mask.shape) != (
            rows,
            sequence_length,
        ):
            raise Qwen3SensitivityShapeError(
                f"{split_name}_attention_mask must match input_ids."
            )
        mask = attention_mask.detach().to(device="cpu", dtype=torch.bool).clone()
    if bool((mask.sum(dim=1) == 0).any()):
        raise Qwen3SensitivityShapeError(
            f"Every {split_name} row must contain at least one active token."
        )
    if position_ids is None:
        positions = mask.to(torch.long).cumsum(dim=1).sub(1).clamp_min(0)
    else:
        if not isinstance(position_ids, Tensor) or tuple(position_ids.shape) != (
            rows,
            sequence_length,
        ):
            raise Qwen3SensitivityShapeError(
                f"{split_name}_position_ids must match input_ids."
            )
        if position_ids.dtype == torch.bool or position_ids.is_floating_point():
            raise Qwen3SensitivityShapeError(
                f"{split_name}_position_ids must use an integer dtype."
            )
        positions = position_ids.detach().to(device="cpu", dtype=torch.long).clone()
    if bool((positions[mask] < 0).any()):
        raise Qwen3SensitivityShapeError(
            f"Active {split_name} position IDs cannot be negative."
        )

    row_hashes: list[str] = []
    content_hashes: list[str] = []
    for row in range(rows):
        active_ids = ids[row][mask[row]]
        active_positions = positions[row][mask[row]]
        content = hashlib.sha256()
        content.update(_tensor_bytes(active_ids))
        content_hashes.append(content.hexdigest())
        positioned = hashlib.sha256()
        positioned.update(_tensor_bytes(active_ids))
        positioned.update(_tensor_bytes(active_positions))
        row_hashes.append(positioned.hexdigest())
    fingerprint = _sha256_json(
        {
            "split": split_name,
            "shape": [rows, sequence_length],
            "rows": row_hashes,
            "mask": hashlib.sha256(_tensor_bytes(mask)).hexdigest(),
        }
    )
    return _Batch(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        row_fingerprints=tuple(row_hashes),
        content_fingerprints=tuple(content_hashes),
        fingerprint=fingerprint,
    )


def _strictly_disjoint(calibration: _Batch, gate: _Batch) -> None:
    positioned_overlap = set(calibration.row_fingerprints) & set(gate.row_fingerprints)
    content_overlap = set(calibration.content_fingerprints) & set(
        gate.content_fingerprints
    )
    if positioned_overlap or content_overlap:
        raise Qwen3SensitivitySplitError(
            "Calibration and gate rows must be disjoint by active token content "
            "and by token/position pairs."
        )


def _layout(model: nn.Module, limits: Qwen3SensitivityLimits) -> _Layout:
    if not isinstance(model, nn.Module):
        raise Qwen3SensitivityShapeError("model must be a torch module.")
    if model.training:
        raise Qwen3SensitivityStateError(
            "Sensitivity collection requires model.eval(); training mode is refused."
        )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise Qwen3SensitivityShapeError("model.config.model_type must be 'qwen3'.")
    backbone = getattr(model, "model", None)
    layers_value = getattr(backbone, "layers", None)
    if not isinstance(layers_value, (nn.ModuleList, list, tuple)) or not layers_value:
        raise Qwen3SensitivityShapeError(
            "Qwen3 model.model.layers must be a non-empty layer sequence."
        )
    layers = tuple(layers_value)
    hidden_size = _positive_int(getattr(config, "hidden_size", None), "hidden_size")
    query_heads = _positive_int(
        getattr(config, "num_attention_heads", None),
        "num_attention_heads",
    )
    key_value_heads = _positive_int(
        getattr(config, "num_key_value_heads", None),
        "num_key_value_heads",
    )
    head_dim = _positive_int(getattr(config, "head_dim", None), "head_dim")
    if query_heads % key_value_heads:
        raise Qwen3SensitivityShapeError(
            "num_attention_heads must be divisible by num_key_value_heads."
        )

    modes: list[Literal["dense", "codebook"]] = []
    widths: set[int] = set()
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, NeuronCodebookLayerView):
            modes.append("codebook")
            widths.add(int(mlp.intermediate_features))
        else:
            down = getattr(mlp, "down_proj", None)
            if not isinstance(down, nn.Module) or not hasattr(down, "in_features"):
                raise Qwen3SensitivityShapeError(
                    f"Layer {layer_index} lacks a dense or codebook Qwen MLP."
                )
            if not callable(getattr(mlp, "act_fn", None)):
                raise Qwen3SensitivityShapeError(
                    f"Layer {layer_index} dense MLP lacks act_fn."
                )
            modes.append("dense")
            widths.add(int(down.in_features))
        attention = getattr(layer, "self_attn", None)
        for role in _ATTENTION_ROLES:
            projection = getattr(attention, f"{role}_proj", None)
            if not isinstance(projection, nn.Module):
                raise Qwen3SensitivityShapeError(
                    f"Layer {layer_index} lacks self_attn.{role}_proj."
                )
        for role in ("q", "k"):
            normalization = getattr(attention, f"{role}_norm", None)
            if not isinstance(normalization, nn.Module):
                raise Qwen3SensitivityShapeError(
                    f"Layer {layer_index} lacks self_attn.{role}_norm."
                )
    if len(widths) != 1:
        raise Qwen3SensitivityShapeError(
            "All Qwen layers must share one intermediate width."
        )
    intermediate = widths.pop()
    output = model.get_output_embeddings()
    vocabulary_size = int(getattr(output, "out_features", 0))
    if vocabulary_size <= 0:
        vocabulary_size = _positive_int(
            getattr(config, "vocab_size", None),
            "vocab_size",
        )
    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, Tensor) or weight.device.type == "meta":
        raise Qwen3SensitivityShapeError(
            "Qwen input embedding must expose a materialized weight tensor."
        )
    devices = {
        str(value.device)
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
        if value.device.type != "meta"
    }
    if not devices:
        devices = {str(weight.device)}
    device_types = {torch.device(item).type for item in devices}
    unsupported = device_types - set(limits.allowed_device_types)
    if unsupported:
        raise Qwen3SensitivityLimitError(
            f"Model uses forbidden device types {sorted(unsupported)!r}."
        )
    if len(devices) > limits.max_model_devices:
        raise Qwen3SensitivityLimitError(
            f"Model spans {len(devices)} devices, exceeding hard cap "
            f"{limits.max_model_devices}."
        )
    attention_backend = (
        str(getattr(config, "_attn_implementation", None) or "eager").strip().lower()
    )
    return _Layout(
        layers=layers,
        mlp_modes=tuple(modes),
        intermediate_width=intermediate,
        hidden_size=hidden_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        vocabulary_size=vocabulary_size,
        input_device=weight.device,
        activation_element_bytes=max(1, int(weight.element_size())),
        model_devices=tuple(sorted(devices)),
        attention_backend=attention_backend,
    )


def _ref(value: Any, kind: Literal["neuron", "attention"]) -> Any:
    expected = Qwen3NeuronRef if kind == "neuron" else Qwen3AttentionGroupRef
    if isinstance(value, expected):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            return expected(value[0], value[1])
    label = "(layer, neuron)" if kind == "neuron" else "(layer, GQA group)"
    raise Qwen3SensitivityShapeError(f"Candidate must be {label}.")


def _refs(
    values: Sequence[Any] | None,
    *,
    kind: Literal["neuron", "attention"],
    layout: _Layout,
    maximum: int,
) -> tuple[Any, ...]:
    if values is None:
        total = len(layout.layers) * (
            layout.intermediate_width if kind == "neuron" else layout.key_value_heads
        )
        if total > maximum:
            raise Qwen3SensitivityLimitError(
                f"Implicit all-{kind} collection would create {total} candidates; "
                f"provide an explicit bounded candidate list (cap {maximum})."
            )
        width = (
            layout.intermediate_width if kind == "neuron" else layout.key_value_heads
        )
        values = [
            (layer, index)
            for layer in range(len(layout.layers))
            for index in range(width)
        ]
    result = tuple(sorted({_ref(value, kind) for value in values}))
    if len(result) > maximum:
        raise Qwen3SensitivityLimitError(
            f"{kind} candidate count {len(result)} exceeds hard cap {maximum}."
        )
    width = layout.intermediate_width if kind == "neuron" else layout.key_value_heads
    for item in result:
        index = item.neuron_index if kind == "neuron" else item.group_index
        if item.layer_index >= len(layout.layers) or index >= width:
            raise Qwen3SensitivityShapeError(f"Out-of-range {kind} candidate {item!r}.")
    return result


def _score_indices(
    values: Sequence[int] | None,
    batch: _Batch,
    limits: Qwen3SensitivityLimits,
) -> tuple[int, ...]:
    requested = (-1,) if values is None else tuple(values)
    if not requested:
        raise Qwen3SensitivityShapeError("score_positions cannot be empty.")
    normalized: list[int] = []
    for value in requested:
        if isinstance(value, bool):
            raise Qwen3SensitivityShapeError("score positions must be integers.")
        try:
            index = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Qwen3SensitivityShapeError(
                "score positions must be integers."
            ) from exc
        if index < 0:
            index += batch.sequence_length
        if not 0 <= index < batch.sequence_length:
            raise Qwen3SensitivityShapeError(
                f"Score position {value!r} is outside the calibration sequence."
            )
        normalized.append(index)
    result = tuple(sorted(set(normalized)))
    if batch.rows * len(result) > limits.max_score_rows:
        raise Qwen3SensitivityLimitError(
            f"Scored logit rows {batch.rows * len(result)} exceed hard cap "
            f"{limits.max_score_rows}."
        )
    if not bool(batch.attention_mask[:, list(result)].all()):
        raise Qwen3SensitivityShapeError(
            "Every score position must be active in every calibration row."
        )
    return result


_HOOK_MAPPING_NAMES = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)
_HOOK_OPTION_MAPPING_NAMES = (
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks_with_kwargs",
)
_GLOBAL_HOOK_MAPPING_NAMES = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_hooks",
)


def _hook_inventory(
    model: nn.Module,
) -> tuple[tuple[str, str, int, int, str], ...]:
    def static_callable_sha256(hook: Any) -> str:
        identity = _hook_callable_identity(hook)
        return _sha256_json(
            {
                name: identity[name]
                for name in ("type", "module", "qualname", "code_sha256")
            }
        )

    result: list[tuple[str, str, int, int, str]] = []
    for module_name, module in model.named_modules():
        for mapping_name in _HOOK_MAPPING_NAMES:
            mapping = getattr(module, mapping_name, {})
            for key, hook in mapping.items():
                result.append(
                    (
                        module_name,
                        mapping_name,
                        int(key),
                        id(hook),
                        static_callable_sha256(hook),
                    )
                )
        for mapping_name in _HOOK_OPTION_MAPPING_NAMES:
            mapping = getattr(module, mapping_name, {})
            for key, value in mapping.items():
                result.append(
                    (
                        module_name,
                        mapping_name,
                        int(key),
                        id(value),
                        hashlib.sha256(repr(value).encode()).hexdigest(),
                    )
                )
    module_runtime = torch.nn.modules.module
    for mapping_name in _GLOBAL_HOOK_MAPPING_NAMES:
        mapping = getattr(module_runtime, mapping_name, {})
        for key, hook in mapping.items():
            result.append(
                (
                    "<global>",
                    mapping_name,
                    int(key),
                    id(hook),
                    static_callable_sha256(hook),
                )
            )
    return tuple(sorted(result))


def _code_constant_identity(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return {"code": _code_identity_payload(value)}
    if isinstance(value, tuple):
        return {"tuple": [_code_constant_identity(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_code_constant_identity(item) for item in value]
        return {"frozenset": sorted(items, key=_canonical_json)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if (
        value is None
        or value is Ellipsis
        or isinstance(
            value,
            (bool, int, float, complex, str),
        )
    ):
        return {
            "type": type(value).__qualname__,
            "repr": repr(value),
        }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _code_identity_payload(code: types.CodeType) -> dict[str, Any]:
    return {
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant_identity(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
    }


def _hook_callable_identity(value: Any) -> dict[str, Any]:
    def opaque(item: Any) -> dict[str, Any]:
        if item is None or isinstance(item, (bool, int, float, str)):
            rendered = repr(item)
        elif isinstance(item, Tensor):
            rendered = repr(
                (
                    tuple(int(part) for part in item.shape),
                    str(item.dtype),
                    str(item.device),
                    _tensor_version(item),
                )
            )
        else:
            rendered = f"object_id={id(item)}"
        return {
            "type": f"{type(item).__module__}.{type(item).__qualname__}",
            "value_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }

    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    code_sha256 = None
    if code is not None:
        code_sha256 = _sha256_json(_code_identity_payload(code))
    defaults = getattr(function, "__defaults__", None)
    keyword_defaults = getattr(function, "__kwdefaults__", None)
    closure = getattr(function, "__closure__", None)
    bound_self = getattr(value, "__self__", None)
    opaque_state = _sha256_json(
        {
            "defaults": [opaque(item) for item in defaults or ()],
            "keyword_defaults": [
                [name, opaque(item)]
                for name, item in sorted((keyword_defaults or {}).items())
            ],
            "closure": (
                [opaque(cell.cell_contents) for cell in closure]
                if closure is not None
                else None
            ),
            "bound_self": opaque(bound_self) if bound_self is not None else None,
        }
    )
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "code_sha256": code_sha256,
        "opaque_state_sha256": opaque_state,
    }


def _runtime_hook_identity(model: nn.Module) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for module_name, module in model.named_modules():
        for mapping_name in _HOOK_MAPPING_NAMES:
            mapping = getattr(module, mapping_name, {})
            for key, hook in mapping.items():
                records.append(
                    {
                        "module": module_name,
                        "kind": mapping_name,
                        "key": int(key),
                        "callable": _hook_callable_identity(hook),
                    }
                )
        for mapping_name in _HOOK_OPTION_MAPPING_NAMES:
            for key, value in getattr(module, mapping_name, {}).items():
                records.append(
                    {
                        "module": module_name,
                        "kind": mapping_name,
                        "key": int(key),
                        "value": repr(value),
                    }
                )
    module_runtime = torch.nn.modules.module
    for mapping_name in _GLOBAL_HOOK_MAPPING_NAMES:
        for key, hook in getattr(module_runtime, mapping_name, {}).items():
            records.append(
                {
                    "module": "<global>",
                    "kind": mapping_name,
                    "key": int(key),
                    "callable": _hook_callable_identity(hook),
                }
            )
    ordered = sorted(records, key=lambda item: _canonical_json(item))
    return {
        "records": ordered,
        "cross_runtime_reuse": not records,
        "cache_nonce": os.urandom(32).hex() if records else None,
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _autocast_identity(device_type: str) -> dict[str, Any]:
    try:
        enabled = bool(torch.is_autocast_enabled(device_type))
        dtype = str(torch.get_autocast_dtype(device_type))
    except (AttributeError, RuntimeError, TypeError):
        enabled = bool(torch.is_autocast_enabled()) if device_type == "cuda" else False
        dtype = None
    return {"enabled": enabled, "dtype": dtype}


def _optional_backend_flag(owner: Any, name: str) -> bool | None:
    value = getattr(owner, name, None)
    if value is None:
        return None
    try:
        return bool(value() if callable(value) else value)
    except (AttributeError, RuntimeError, TypeError):
        return None


def _cuda_driver_version() -> int | None:
    """Query the loaded CUDA driver directly without invoking a subprocess."""

    try:
        library = (
            ctypes.WinDLL("nvcuda.dll", use_last_error=True)
            if os.name == "nt"
            else ctypes.CDLL("libcuda.so.1")
        )
        query = library.cuDriverGetVersion
        query.argtypes = (ctypes.POINTER(ctypes.c_int),)
        query.restype = ctypes.c_int
        version = ctypes.c_int()
        if int(query(ctypes.byref(version))) != 0:
            return None
        return int(version.value)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _execution_implementation_identity(model: nn.Module, layout: _Layout) -> str:
    callables: dict[str, Any] = {
        "collector_forward": _forward,
        "baseline": _baseline,
        "measure_neurons": _measure_neurons,
        "measure_attention": _measure_attention,
        "model_forward": type(model).forward,
    }
    for index, layer in enumerate(layout.layers):
        callables[f"layer_{index}_forward"] = type(layer).forward
        callables[f"layer_{index}_attention_forward"] = type(layer.self_attn).forward
        callables[f"layer_{index}_mlp_forward"] = type(layer.mlp).forward
        callables[f"layer_{index}_q_norm_forward"] = type(
            layer.self_attn.q_norm
        ).forward
        callables[f"layer_{index}_k_norm_forward"] = type(
            layer.self_attn.k_norm
        ).forward
    return _sha256_json(
        {
            name: _hook_callable_identity(value)
            for name, value in sorted(callables.items())
        }
    )


def _runtime_identity(
    model: nn.Module,
    layout: _Layout,
    limits: Qwen3SensitivityLimits,
) -> dict[str, Any]:
    state = _model_state_tensors(model)
    state_layout = sorted(
        {
            (str(value.device), str(value.dtype))
            for value in state.values()
            if isinstance(value, Tensor)
        }
    )
    cuda_devices: list[dict[str, Any]] = []
    for name in layout.model_devices:
        device = torch.device(name)
        if device.type != "cuda":
            continue
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        cuda_devices.append(
            {
                "logical_device": name,
                "index": int(index),
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(properties.total_memory),
                "multi_processor_count": int(properties.multi_processor_count),
                "uuid": str(getattr(properties, "uuid", "")) or None,
            }
        )
    uses_cuda = bool(cuda_devices)
    cuda_driver_version = _cuda_driver_version() if uses_cuda else None
    cudnn_version = (
        int(torch.backends.cudnn.version())
        if uses_cuda and torch.backends.cudnn.is_available()
        else None
    )
    hooks = _runtime_hook_identity(model)
    packages = {
        "torch": str(torch.__version__),
        "transformers": _distribution_version("transformers"),
        "numpy": _distribution_version("numpy"),
        "flash_attn": _distribution_version("flash-attn"),
        "triton": _distribution_version("triton"),
    }
    fail_closed_reasons: list[str] = []
    if hooks["records"]:
        fail_closed_reasons.append("active_hooks")
    if packages["transformers"] is None:
        fail_closed_reasons.append("unknown_transformers_version")
    if uses_cuda and cuda_driver_version is None:
        fail_closed_reasons.append("unknown_cuda_driver_version")
    fail_closed_nonce = (
        hooks["cache_nonce"] or os.urandom(32).hex() if fail_closed_reasons else None
    )
    sdp_flags = {
        name: _optional_backend_flag(torch.backends.cuda, name) if uses_cuda else None
        for name in (
            "flash_sdp_enabled",
            "mem_efficient_sdp_enabled",
            "math_sdp_enabled",
            "cudnn_sdp_enabled",
        )
    }
    implementation_sha256 = _execution_implementation_identity(model, layout)
    return {
        "schema": "neural-forge-qwen3-sensitivity-runtime-v2",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "torch": {
            "version": str(torch.__version__),
            "git_version": getattr(torch.version, "git_version", None),
            "debug": bool(getattr(torch.version, "debug", False)),
            "build_config_sha256": hashlib.sha256(
                torch.__config__.show().encode("utf-8")
            ).hexdigest(),
            "cuda_version": torch.version.cuda,
            "cuda_driver_version": cuda_driver_version,
            "cudnn_version": cudnn_version,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": (
                bool(torch.backends.cuda.matmul.allow_tf32) if uses_cuda else None
            ),
            "cudnn_allow_tf32": (
                bool(torch.backends.cudnn.allow_tf32) if uses_cuda else None
            ),
            "cudnn_benchmark": (
                bool(torch.backends.cudnn.benchmark) if uses_cuda else None
            ),
            "cudnn_deterministic": (
                bool(torch.backends.cudnn.deterministic) if uses_cuda else None
            ),
            "cuda_sdp_backends": sdp_flags,
            "cuda_matmul_allow_fp16_reduced_precision_reduction": (
                bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                if uses_cuda
                else None
            ),
            "cuda_matmul_allow_bf16_reduced_precision_reduction": (
                bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                if uses_cuda
                else None
            ),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "mkldnn_deterministic": bool(
                getattr(torch.backends.mkldnn, "deterministic", False)
            ),
            "default_dtype": str(torch.get_default_dtype()),
            "default_device": str(
                torch.get_default_device()
                if hasattr(torch, "get_default_device")
                else "cpu"
            ),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "autocast_cpu": _autocast_identity("cpu"),
            "autocast_cuda": _autocast_identity("cuda") if uses_cuda else None,
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_TF32_OVERRIDE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "TORCHINDUCTOR_FX_GRAPH_CACHE",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "attention_backend": layout.attention_backend,
        "attention_module_types": [
            f"{type(layer.self_attn).__module__}.{type(layer.self_attn).__qualname__}"
            for layer in layout.layers
        ],
        "model_devices": list(layout.model_devices),
        "state_device_dtypes": [list(item) for item in state_layout],
        "state_tensor_placements": [
            [name, str(value.device), str(value.dtype)]
            for name, value in sorted(state.items())
        ],
        "input_device": str(layout.input_device),
        "activation_element_bytes": layout.activation_element_bytes,
        "cuda_devices": cuda_devices,
        "module_training_modes": [
            [name, bool(module.training)] for name, module in model.named_modules()
        ],
        "hooks": hooks,
        "execution_implementation_sha256": implementation_sha256,
        "cross_runtime_cache_reuse": {
            "allowed": not fail_closed_reasons,
            "fail_closed_reasons": fail_closed_reasons,
            "fail_closed_nonce": fail_closed_nonce,
        },
        "execution": {
            "inference_mode": True,
            "use_cache": False,
            "logits_to_keep": "explicit_position_tensor",
            "grad_enabled_at_entry": bool(torch.is_grad_enabled()),
            "codebook_channels_per_chunk": limits.max_codebook_channels_per_chunk,
            "candidate_chunk_size_cap": limits.candidate_chunk_size,
            "candidate_forward_row_cap": limits.max_candidate_forward_rows,
            "fingerprint_chunk_bytes": limits.max_fingerprint_chunk_bytes,
            "sketch_work_bytes": limits.max_sketch_work_bytes,
            "rss_sample_interval_seconds": _RSS_SAMPLE_INTERVAL_SECONDS,
        },
    }


def _activation(inputs: Tensor, name: str) -> Tensor:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "linear": "identity",
        "swish": "silu",
        "gelu_new": "gelu_tanh",
        "gelu_pytorch_tanh": "gelu_tanh",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "identity":
        return inputs
    if normalized == "silu":
        return F.silu(inputs)
    if normalized == "gelu":
        return F.gelu(inputs, approximate="none")
    if normalized == "gelu_tanh":
        return F.gelu(inputs, approximate="tanh")
    if normalized == "relu":
        return F.relu(inputs)
    raise Qwen3SensitivityShapeError(f"Unsupported codebook activation {name!r}.")


def _codebook_channels(
    module: NeuronCodebookLayerView,
    layer_index: int,
    logical_indices: Tensor,
    *,
    include_down: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Gather only requested logical channels directly from shared prototypes."""

    registry = module.registry
    logical = logical_indices.to(device=registry.assignments.device, dtype=torch.long)
    prototype_indices = registry.assignments[layer_index].index_select(0, logical)
    prototype_indices = prototype_indices.to(
        device=registry.prototype_gate_weight.device,
        dtype=torch.long,
    )
    gate = registry.prototype_gate_weight.index_select(0, prototype_indices)
    up = registry.prototype_up_weight.index_select(0, prototype_indices)
    down = (
        registry.prototype_down_weight.index_select(1, prototype_indices)
        if include_down
        else None
    )
    gate_bias = (
        registry.prototype_gate_bias.index_select(0, prototype_indices)
        if registry.prototype_gate_bias is not None
        else None
    )
    up_bias = (
        registry.prototype_up_bias.index_select(0, prototype_indices)
        if registry.prototype_up_bias is not None
        else None
    )
    if registry.neuron_scales is not None:
        scales = registry.neuron_scales[layer_index].index_select(
            0,
            logical.to(device=registry.neuron_scales.device),
        )
        gate = gate * scales[:, 0].unsqueeze(1)
        up = up * scales[:, 1].unsqueeze(1)
        if down is not None:
            down = down * scales[:, 2].unsqueeze(0)
        if gate_bias is not None:
            gate_bias = gate_bias * scales[:, 0]
        if up_bias is not None:
            up_bias = up_bias * scales[:, 1]
    return gate, up, down, gate_bias, up_bias


def _streamed_codebook_neuron_energy(
    module: NeuronCodebookLayerView,
    layer_index: int,
    inputs: Tensor,
    mask: Tensor,
    *,
    channels_per_chunk: int,
    guard: _ResourceGuard,
) -> tuple[float, ...]:
    """Measure every logical channel without a full dense layer reconstruction."""

    result: list[float] = []
    width = int(module.intermediate_features)
    for start in range(0, width, channels_per_chunk):
        stop = min(width, start + channels_per_chunk)
        logical = torch.arange(start, stop, device=inputs.device, dtype=torch.long)
        gate, up, down, gate_bias, up_bias = _codebook_channels(
            module,
            layer_index,
            logical,
            include_down=False,
        )
        hidden = _activation(
            F.linear(inputs, gate, gate_bias),
            module.registry.activation,
        ) * F.linear(inputs, up, up_bias)
        result.extend(_masked_channel_energy(hidden, mask))
        if down is not None:
            raise Qwen3SensitivityStateError(
                "Codebook energy unexpectedly materialized down-projection columns."
            )
        del gate, up, down, gate_bias, up_bias, hidden
        guard.checkpoint(f"codebook_energy_layer_{layer_index}_channels_{start}_{stop}")
    return tuple(result)


def _masked_channel_energy(values: Tensor, mask: Tensor) -> tuple[float, ...]:
    if values.ndim != 3 or tuple(values.shape[:2]) != tuple(mask.shape):
        raise Qwen3SensitivityShapeError(
            "Captured post-SwiGLU activations do not match calibration rows."
        )
    active = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denominator = max(1, int(mask.sum().item()))
    energy = values.detach().float().square().mul(active.float()).sum((0, 1))
    energy = energy.div(float(denominator)).to("cpu")
    return tuple(float(item) for item in energy.tolist())


def _masked_group_energy(
    values: Tensor,
    mask: Tensor,
    groups: int,
) -> tuple[float, ...]:
    if values.ndim < 3 or tuple(values.shape[:2]) != tuple(mask.shape):
        raise Qwen3SensitivityShapeError(
            "Captured attention activations do not match calibration rows."
        )
    width = math.prod(int(item) for item in values.shape[2:])
    if width % groups:
        raise Qwen3SensitivityShapeError(
            "Attention projection width is not divisible by GQA groups."
        )
    per_group = width // groups
    shaped = values.detach().reshape(*values.shape[:2], groups, per_group)
    active = (
        mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1).unsqueeze(-1)
    )
    denominator = max(1, int(mask.sum().item()) * per_group)
    energy = shaped.float().square().mul(active.float()).sum((0, 1, 3))
    energy = energy.div(float(denominator)).to("cpu")
    return tuple(float(item) for item in energy.tolist())


def _forward(
    model: nn.Module,
    batch: _Batch,
    score_positions: tuple[int, ...],
    *,
    repeats: int = 1,
) -> Tensor:
    device = next(model.get_input_embeddings().parameters()).device
    ids = batch.input_ids.to(device).repeat((repeats, 1))
    mask = batch.attention_mask.to(device).repeat((repeats, 1))
    positions = batch.position_ids.to(device).repeat((repeats, 1))
    score = torch.tensor(score_positions, device=device, dtype=torch.long)
    output = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        logits_to_keep=score,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise Qwen3SensitivityShapeError("Qwen forward did not return rank-3 logits.")
    expected_rows = batch.rows * repeats
    if tuple(logits.shape[:2]) != (expected_rows, len(score_positions)):
        raise Qwen3SensitivityShapeError(
            "Qwen logits do not match requested held-out score rows."
        )
    return logits


def _baseline(
    model: nn.Module,
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    residual_layers: frozenset[int],
    limits: Qwen3SensitivityLimits,
    guard: _ResourceGuard,
) -> _Baseline:
    mask = batch.attention_mask
    residuals: dict[int, Tensor] = {}
    neuron_energy: list[tuple[float, ...] | None] = [None] * len(layout.layers)
    capture_methods: list[str] = [""] * len(layout.layers)
    role_energy: list[dict[str, tuple[float, ...]]] = [{} for _ in layout.layers]
    with _HookScope() as hooks:
        for layer_index, layer in enumerate(layout.layers):

            def residual_hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, Tensor) or output.ndim != 3:
                    raise Qwen3SensitivityShapeError(
                        "Qwen decoder layer did not return rank-3 residual state."
                    )
                residuals[index] = output.detach().to("cpu", dtype=torch.float32)

            if layer_index in residual_layers:
                hooks.add(layer.register_forward_hook(residual_hook))
            mlp = layer.mlp
            if layout.mlp_modes[layer_index] == "dense":

                def dense_energy_hook(
                    _module: nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    index: int = layer_index,
                ) -> None:
                    if not inputs or not isinstance(inputs[0], Tensor):
                        raise Qwen3SensitivityShapeError(
                            "Qwen down_proj did not receive post-SwiGLU activations."
                        )
                    neuron_energy[index] = _masked_channel_energy(inputs[0], mask)

                hooks.add(mlp.down_proj.register_forward_pre_hook(dense_energy_hook))
                capture_methods[layer_index] = "real_down_proj_pre_hook"
            else:

                def codebook_energy_hook(
                    module: nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    index: int = layer_index,
                ) -> None:
                    if not isinstance(module, NeuronCodebookLayerView):
                        raise Qwen3SensitivityShapeError(
                            "Expected a NeuronCodebookLayerView."
                        )
                    if not inputs or not isinstance(inputs[0], Tensor):
                        raise Qwen3SensitivityShapeError(
                            "Codebook MLP did not receive hidden states."
                        )
                    neuron_energy[index] = _streamed_codebook_neuron_energy(
                        module,
                        index,
                        inputs[0],
                        mask,
                        channels_per_chunk=limits.max_codebook_channels_per_chunk,
                        guard=guard,
                    )

                hooks.add(mlp.register_forward_pre_hook(codebook_energy_hook))
                capture_methods[layer_index] = (
                    "streamed_direct_codebook_prototype_channels"
                )

            attention = layer.self_attn
            for role in ("q", "k"):
                normalization = getattr(attention, f"{role}_norm")

                def attention_output_hook(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    index: int = layer_index,
                    role_name: str = role,
                ) -> None:
                    if not isinstance(output, Tensor):
                        raise Qwen3SensitivityShapeError(
                            f"Qwen {role_name}_norm did not return a tensor."
                        )
                    role_energy[index][role_name] = _masked_group_energy(
                        output,
                        mask,
                        layout.key_value_heads,
                    )

                hooks.add(normalization.register_forward_hook(attention_output_hook))

            def attention_v_hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Qwen v_proj did not return a tensor."
                    )
                role_energy[index]["v"] = _masked_group_energy(
                    output,
                    mask,
                    layout.key_value_heads,
                )

            hooks.add(attention.v_proj.register_forward_hook(attention_v_hook))

            def attention_o_hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                index: int = layer_index,
            ) -> None:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Qwen o_proj did not receive attention contexts."
                    )
                role_energy[index]["o"] = _masked_group_energy(
                    inputs[0],
                    mask,
                    layout.key_value_heads,
                )

            hooks.add(attention.o_proj.register_forward_pre_hook(attention_o_hook))
        logits = _forward(model, batch, score_positions)
        guard.checkpoint("baseline_forward")

    if set(residuals) != set(residual_layers):
        raise Qwen3SensitivityStateError(
            "At least one requested decoder residual hook did not execute."
        )
    if any(value is None for value in neuron_energy):
        raise Qwen3SensitivityStateError(
            "At least one post-SwiGLU activation hook did not execute."
        )
    attention_result: list[tuple[tuple[float, float, float, float], ...]] = []
    for layer_index, values in enumerate(role_energy):
        if set(values) != set(_ATTENTION_ROLES):
            raise Qwen3SensitivityStateError(
                f"Layer {layer_index} did not expose all Q/K/V/O activations."
            )
        attention_result.append(
            tuple(
                tuple(values[role][group] for role in _ATTENTION_ROLES)
                for group in range(layout.key_value_heads)
            )
        )
    return _Baseline(
        logits=logits.detach().to("cpu", dtype=torch.float32),
        residuals=residuals,
        neuron_energy=tuple(value for value in neuron_energy if value is not None),
        attention_energy=tuple(attention_result),
        capture_methods=tuple(capture_methods),
    )


def _candidate_residual_hooks(
    hooks: _HookScope,
    layout: _Layout,
    chunk: Sequence[Any],
    rows: int,
    result: dict[int, Tensor],
) -> None:
    by_layer: dict[int, list[int]] = {}
    for local_index, candidate in enumerate(chunk):
        by_layer.setdefault(candidate.layer_index, []).append(local_index)
    for layer_index, local_indices in by_layer.items():

        def capture(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            indices: tuple[int, ...] = tuple(local_indices),
        ) -> None:
            if not isinstance(output, Tensor) or output.ndim != 3:
                raise Qwen3SensitivityShapeError(
                    "Qwen decoder layer did not return rank-3 residual state."
                )
            for local in indices:
                start = local * rows
                result[local] = (
                    output[start : start + rows].detach().to("cpu", dtype=torch.float32)
                )

        hooks.add(layout.layers[layer_index].register_forward_hook(capture))


def _install_neuron_ablation_hooks(
    hooks: _HookScope,
    layout: _Layout,
    chunk: Sequence[Qwen3NeuronRef],
    rows: int,
    call_counts: dict[int, int],
) -> None:
    by_layer: dict[int, list[tuple[int, Qwen3NeuronRef]]] = {}
    for local, candidate in enumerate(chunk):
        by_layer.setdefault(candidate.layer_index, []).append((local, candidate))
    for layer_index, entries in by_layer.items():
        mlp = layout.layers[layer_index].mlp
        if layout.mlp_modes[layer_index] == "dense":

            def dense_ablation(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                selected: tuple[tuple[int, Qwen3NeuronRef], ...] = tuple(entries),
            ) -> tuple[Any, ...]:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Qwen down_proj did not receive a tensor."
                    )
                changed = inputs[0].clone()
                for local, candidate in selected:
                    start = local * rows
                    changed[
                        start : start + rows,
                        :,
                        candidate.neuron_index,
                    ] = 0
                    call_counts[local] = call_counts.get(local, 0) + 1
                return (changed, *inputs[1:])

            hooks.add(mlp.down_proj.register_forward_pre_hook(dense_ablation))
        else:

            def codebook_ablation(
                module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                selected: tuple[tuple[int, Qwen3NeuronRef], ...] = tuple(entries),
            ) -> Tensor:
                if not isinstance(module, NeuronCodebookLayerView):
                    raise Qwen3SensitivityShapeError(
                        "Expected a NeuronCodebookLayerView."
                    )
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Codebook MLP did not receive a tensor."
                    )
                if not isinstance(output, Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Codebook MLP did not return a tensor."
                    )
                changed = output.clone()
                for local, candidate in selected:
                    start = local * rows
                    hidden = inputs[0][start : start + rows]
                    neuron = candidate.neuron_index
                    logical = torch.tensor(
                        [neuron],
                        device=hidden.device,
                        dtype=torch.long,
                    )
                    gate, up, down, gate_bias, up_bias = _codebook_channels(
                        module,
                        index,
                        logical,
                    )
                    if down is None:
                        raise Qwen3SensitivityStateError(
                            "Codebook ablation requires one down-projection column."
                        )
                    post = _activation(
                        F.linear(hidden, gate, gate_bias),
                        module.registry.activation,
                    ) * F.linear(hidden, up, up_bias)
                    contribution = F.linear(post, down)
                    changed[start : start + rows] -= contribution
                    call_counts[local] = call_counts.get(local, 0) + 1
                return changed

            hooks.add(mlp.register_forward_hook(codebook_ablation))


def _install_attention_ablation_hooks(
    hooks: _HookScope,
    layout: _Layout,
    chunk: Sequence[Qwen3AttentionGroupRef],
    rows: int,
    call_counts: dict[int, dict[str, int]],
) -> None:
    by_layer: dict[int, list[tuple[int, Qwen3AttentionGroupRef]]] = {}
    for local, candidate in enumerate(chunk):
        by_layer.setdefault(candidate.layer_index, []).append((local, candidate))
    for layer_index, entries in by_layer.items():
        attention = layout.layers[layer_index].self_attn
        for role in ("q", "k", "v"):
            projection = getattr(attention, f"{role}_proj")
            group_width = (
                layout.query_rows_per_group if role == "q" else layout.head_dim
            )

            def output_ablation(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                role_name: str = role,
                width: int = group_width,
                selected: tuple[tuple[int, Qwen3AttentionGroupRef], ...] = tuple(
                    entries
                ),
            ) -> Tensor:
                if not isinstance(output, Tensor):
                    raise Qwen3SensitivityShapeError(
                        f"Qwen {role_name}_proj did not return a tensor."
                    )
                changed = output.clone()
                for local, candidate in selected:
                    row_start = local * rows
                    column_start = candidate.group_index * width
                    changed[
                        row_start : row_start + rows,
                        :,
                        column_start : column_start + width,
                    ] = 0
                    counts = call_counts.setdefault(local, {})
                    counts[role_name] = counts.get(role_name, 0) + 1
                return changed

            hooks.add(projection.register_forward_hook(output_ablation))

        def o_ablation(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            selected: tuple[tuple[int, Qwen3AttentionGroupRef], ...] = tuple(entries),
        ) -> tuple[Any, ...]:
            if not inputs or not isinstance(inputs[0], Tensor):
                raise Qwen3SensitivityShapeError(
                    "Qwen o_proj did not receive a tensor."
                )
            changed = inputs[0].clone()
            width = layout.query_rows_per_group
            for local, candidate in selected:
                row_start = local * rows
                column_start = candidate.group_index * width
                changed[
                    row_start : row_start + rows,
                    :,
                    column_start : column_start + width,
                ] = 0
                counts = call_counts.setdefault(local, {})
                counts["o"] = counts.get("o", 0) + 1
            return (changed, *inputs[1:])

        hooks.add(attention.o_proj.register_forward_pre_hook(o_ablation))


def _count_sketch(
    values: Tensor,
    dimension: int,
    seed: int,
    max_work_bytes: int,
) -> tuple[float, ...]:
    flat = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not int(flat.numel()):
        return tuple(0.0 for _ in range(dimension))
    result = torch.zeros(dimension, dtype=torch.float32)
    result_bytes = dimension * result.element_size()
    available = max_work_bytes - result_bytes
    if available < 32:
        raise Qwen3SensitivityLimitError(
            "max_sketch_work_bytes is too small for the requested sketch."
        )
    chunk_elements = max(1, available // 32)
    seed_value = int(seed % 2_147_483_647)
    for start in range(0, int(flat.numel()), chunk_elements):
        end = min(int(flat.numel()), start + chunk_elements)
        indices = torch.arange(start, end, dtype=torch.int64)
        buckets = torch.remainder(
            indices * 1_103_515_245 + seed_value,
            dimension,
        )
        signs = torch.where(
            torch.remainder(
                indices * 2_654_435_761 + seed_value * 17 + 13,
                2,
            )
            == 0,
            1.0,
            -1.0,
        )
        result.scatter_add_(0, buckets, flat[start:end] * signs)
    result.div_(math.sqrt(float(flat.numel())))
    return tuple(float(item) for item in result.tolist())


def _damage(
    baseline_residual: Tensor,
    candidate_residual: Tensor,
    baseline_logits: Tensor,
    candidate_logits: Tensor,
    mask: Tensor,
    *,
    sketch_dim: int,
    residual_seed: int,
    logit_seed: int,
    sketch_work_bytes: int,
) -> dict[str, Any]:
    residual_delta = candidate_residual.float() - baseline_residual.float()
    active = mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(residual_delta)
    active_delta = residual_delta[active]
    active_baseline = baseline_residual.float()[active]
    residual_square = float(active_delta.square().sum().item())
    residual_base_square = float(active_baseline.square().sum().item())
    residual_mse = residual_square / max(1, int(active_delta.numel()))
    residual_relative = math.sqrt(
        residual_square / max(residual_base_square, torch.finfo(torch.float32).tiny)
    )

    logit_delta = candidate_logits.float() - baseline_logits.float()
    logit_square = float(logit_delta.square().sum().item())
    logit_base_square = float(baseline_logits.float().square().sum().item())
    logit_mse = logit_square / max(1, int(logit_delta.numel()))
    logit_relative = math.sqrt(
        logit_square / max(logit_base_square, torch.finfo(torch.float32).tiny)
    )
    baseline_log_probability = F.log_softmax(baseline_logits.float(), dim=-1)
    candidate_log_probability = F.log_softmax(candidate_logits.float(), dim=-1)
    baseline_probability = baseline_log_probability.exp()
    kl = float(
        (baseline_probability * (baseline_log_probability - candidate_log_probability))
        .sum(dim=-1)
        .mean()
        .item()
    )
    kl = max(0.0, kl)
    top1 = float(
        (baseline_logits.argmax(dim=-1) != candidate_logits.argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    score = residual_relative + logit_relative + math.sqrt(kl) + top1
    return {
        "residual_mse": residual_mse,
        "residual_relative_l2": residual_relative,
        "logit_mse": logit_mse,
        "logit_relative_l2": logit_relative,
        "logit_kl": kl,
        "top1_disagreement": top1,
        "damage_score": score,
        "residual_ablation_difference_sketch": _count_sketch(
            residual_delta * mask.unsqueeze(-1),
            sketch_dim,
            residual_seed,
            sketch_work_bytes,
        ),
        "logit_ablation_difference_sketch": _count_sketch(
            logit_delta,
            sketch_dim,
            logit_seed,
            sketch_work_bytes,
        ),
    }


def _attention_workspace_bound(
    layout: _Layout,
    batch: _Batch,
    repeats: int,
) -> int:
    """Conservative backend-aware bound for attention forward workspace."""

    expanded_rows = batch.rows * repeats
    sequence = batch.sequence_length
    dtype_bytes = layout.activation_element_bytes
    qkv_elements = (
        expanded_rows
        * sequence
        * (layout.query_heads + 2 * layout.key_value_heads)
        * layout.head_dim
    )
    qkv_and_context = qkv_elements * dtype_bytes * 2
    backend = layout.attention_backend
    if "flash" in backend:
        logsumexp = expanded_rows * layout.query_heads * sequence * 4
        return qkv_and_context + logsumexp
    score_elements = expanded_rows * layout.query_heads * sequence * sequence
    score_and_probability = score_elements * (dtype_bytes + 4)
    additive_mask = expanded_rows * sequence * sequence * max(1, dtype_bytes)
    return qkv_and_context + score_and_probability + additive_mask


def _mlp_forward_workspace_bound(
    layout: _Layout,
    batch: _Batch,
    repeats: int,
) -> tuple[int, int, int]:
    """Bound ordinary MLP temporaries and prepared-codebook materialization.

    The activation term charges all ordinary dense MLP live values plus a
    bounded GEMM workspace. The codebook term additionally charges complete
    logical gate/up/down matrices. When scales are enabled, both every gathered
    matrix and every scaled replacement are charged rather than assuming that
    allocator reuse immediately returns the superseded storage.
    """

    expanded_tokens = batch.rows * repeats * batch.sequence_length
    dense_element_bytes = layout.activation_element_bytes
    codebook_element_bytes = layout.activation_element_bytes
    codebook_matrix_copies = 0
    codebook_bias_copies = 0
    for mode, layer in zip(layout.mlp_modes, layout.layers, strict=True):
        if mode == "dense":
            for projection_name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(layer.mlp, projection_name)
                weight = getattr(projection, "weight", None)
                if isinstance(weight, Tensor):
                    dense_element_bytes = max(
                        dense_element_bytes,
                        int(weight.element_size()),
                    )
            continue
        registry = layer.mlp.registry
        materialized_tensors = (
            registry.prototype_gate_weight,
            registry.prototype_up_weight,
            registry.prototype_down_weight,
            registry.neuron_scales,
        )
        codebook_element_bytes = max(
            codebook_element_bytes,
            *(
                int(value.element_size())
                for value in materialized_tensors
                if isinstance(value, Tensor)
            ),
        )
        scaled = registry.neuron_scales is not None
        codebook_matrix_copies = max(codebook_matrix_copies, 6 if scaled else 3)
        bias_count = int(registry.prototype_gate_bias is not None) + int(
            registry.prototype_up_bias is not None
        )
        codebook_bias_copies = max(
            codebook_bias_copies,
            bias_count * (2 if scaled else 1),
        )

    live_elements = expanded_tokens * (
        5 * layout.intermediate_width + 3 * layout.hidden_size
    )
    dense_activation_workspace = live_elements * max(4, dense_element_bytes)
    dense_workspace = (
        dense_activation_workspace + _MLP_KERNEL_WORKSPACE_BOUND_BYTES
        if "dense" in layout.mlp_modes
        else 0
    )
    codebook_materialization = 0
    codebook_workspace = 0
    if "codebook" in layout.mlp_modes:
        matrix_elements = (
            codebook_matrix_copies * layout.hidden_size * layout.intermediate_width
        )
        bias_elements = codebook_bias_copies * layout.intermediate_width
        index_elements = 2 * layout.intermediate_width
        codebook_materialization = (
            matrix_elements + bias_elements
        ) * codebook_element_bytes + index_elements * 8
        codebook_activation_workspace = live_elements * max(
            4,
            codebook_element_bytes,
        )
        codebook_workspace = (
            codebook_materialization
            + codebook_activation_workspace
            + _MLP_KERNEL_WORKSPACE_BOUND_BYTES
        )
    return (
        dense_workspace,
        codebook_materialization,
        max(
            dense_workspace,
            codebook_workspace,
        ),
    )


def _chunk_capacity(
    *,
    kind: Literal["neuron", "attention"],
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    limits: Qwen3SensitivityLimits,
    persistent_bytes: int,
) -> tuple[int, int, int]:
    rows = batch.rows
    sequence = batch.sequence_length
    score_rows = len(score_positions)
    dtype_bytes = layout.activation_element_bytes
    residual = rows * sequence * layout.hidden_size * 4
    logits = rows * score_rows * layout.vocabulary_size * (4 + dtype_bytes)
    expanded_inputs = rows * sequence * (8 + 1 + 8)
    if kind == "neuron":
        intervention_width = (
            layout.intermediate_width
            if "dense" in layout.mlp_modes
            else layout.hidden_size + 2
        )
    else:
        query_width = layout.query_heads * layout.head_dim
        key_value_width = layout.key_value_heads * layout.head_dim
        intervention_width = 2 * query_width + 2 * key_value_width
    intervention = rows * sequence * intervention_width * dtype_bytes
    per_candidate = residual + logits + expanded_inputs + intervention
    analysis_scratch = (
        rows * sequence * layout.hidden_size * 4 * 3
        + rows * score_rows * layout.vocabulary_size * 4 * 5
        + limits.max_sketch_work_bytes
    )
    capacity = min(
        limits.candidate_chunk_size,
        limits.max_candidate_forward_rows // rows,
    )
    while capacity > 0:
        attention_workspace = _attention_workspace_bound(layout, batch, capacity)
        _, _, mlp_workspace = _mlp_forward_workspace_bound(layout, batch, capacity)
        peak = (
            persistent_bytes
            + capacity * per_candidate
            + analysis_scratch
            + attention_workspace
            + mlp_workspace
        )
        if peak <= limits.max_collector_bytes:
            return capacity, peak, attention_workspace
        capacity -= 1
    raise Qwen3SensitivityLimitError(
        f"A single {kind} candidate would exceed max_collector_bytes="
        f"{limits.max_collector_bytes}."
    )


def _enforce_forward_resource_preflight(
    peak_bytes: int,
    layout: _Layout,
    limits: Qwen3SensitivityLimits,
) -> None:
    """Reject a predicted forward peak before any model execution begins."""

    checks = [
        (
            "process RSS",
            limits.max_process_rss_increase_bytes,
        )
    ]
    if any(torch.device(name).type == "cuda" for name in layout.model_devices):
        checks.extend(
            (
                ("CUDA allocated", limits.max_cuda_allocated_increase_bytes),
                ("CUDA reserved", limits.max_cuda_reserved_increase_bytes),
            )
        )
    for resource, maximum in checks:
        if peak_bytes > maximum:
            raise Qwen3SensitivityLimitError(
                f"Predicted forward {resource} peak {peak_bytes} exceeds hard cap "
                f"{maximum}; collection was refused before model execution."
            )


def _measure_neurons(
    model: nn.Module,
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    candidates: tuple[Qwen3NeuronRef, ...],
    baseline: _Baseline,
    *,
    chunk_size: int,
    sketch_dim: int,
    residual_seed: int,
    logit_seed: int,
    sketch_work_bytes: int,
    guard: _ResourceGuard,
) -> tuple[tuple[Qwen3NeuronSensitivity, ...], int]:
    result: list[Qwen3NeuronSensitivity] = []
    forward_passes = 0
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        residuals: dict[int, Tensor] = {}
        calls: dict[int, int] = {}
        with _HookScope() as hooks:
            _install_neuron_ablation_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                calls,
            )
            _candidate_residual_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                residuals,
            )
            logits = (
                _forward(
                    model,
                    batch,
                    score_positions,
                    repeats=len(chunk),
                )
                .detach()
                .to("cpu", dtype=torch.float32)
            )
        guard.checkpoint(f"neuron_candidate_forward_{start}")
        forward_passes += 1
        shaped_logits = logits.reshape(
            len(chunk),
            batch.rows,
            len(score_positions),
            layout.vocabulary_size,
        )
        for local, candidate in enumerate(chunk):
            if calls.get(local) != 1 or local not in residuals:
                raise Qwen3SensitivityStateError(
                    f"Neuron ablation {candidate!r} did not execute exactly once."
                )
            metrics = _damage(
                baseline.residuals[candidate.layer_index],
                residuals[local],
                baseline.logits,
                shaped_logits[local],
                batch.attention_mask,
                sketch_dim=sketch_dim,
                residual_seed=residual_seed,
                logit_seed=logit_seed,
                sketch_work_bytes=sketch_work_bytes,
            )
            result.append(
                Qwen3NeuronSensitivity(
                    node=candidate,
                    activation_energy=baseline.neuron_energy[candidate.layer_index][
                        candidate.neuron_index
                    ],
                    **metrics,
                )
            )
        guard.checkpoint(f"neuron_candidate_analysis_{start}")
    return tuple(result), forward_passes


def _measure_attention(
    model: nn.Module,
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    candidates: tuple[Qwen3AttentionGroupRef, ...],
    baseline: _Baseline,
    *,
    chunk_size: int,
    sketch_dim: int,
    residual_seed: int,
    logit_seed: int,
    sketch_work_bytes: int,
    guard: _ResourceGuard,
) -> tuple[tuple[Qwen3AttentionGroupSensitivity, ...], int]:
    result: list[Qwen3AttentionGroupSensitivity] = []
    forward_passes = 0
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        residuals: dict[int, Tensor] = {}
        calls: dict[int, dict[str, int]] = {}
        with _HookScope() as hooks:
            _install_attention_ablation_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                calls,
            )
            _candidate_residual_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                residuals,
            )
            logits = (
                _forward(
                    model,
                    batch,
                    score_positions,
                    repeats=len(chunk),
                )
                .detach()
                .to("cpu", dtype=torch.float32)
            )
        guard.checkpoint(f"attention_candidate_forward_{start}")
        forward_passes += 1
        shaped_logits = logits.reshape(
            len(chunk),
            batch.rows,
            len(score_positions),
            layout.vocabulary_size,
        )
        for local, candidate in enumerate(chunk):
            counts = calls.get(local, {})
            role_counts = tuple(counts.get(role, 0) for role in _ATTENTION_ROLES)
            if role_counts != (1, 1, 1, 1) or local not in residuals:
                raise Qwen3SensitivityStateError(
                    f"Atomic GQA ablation {candidate!r} did not cover Q/K/V/O once."
                )
            metrics = _damage(
                baseline.residuals[candidate.layer_index],
                residuals[local],
                baseline.logits,
                shaped_logits[local],
                batch.attention_mask,
                sketch_dim=sketch_dim,
                residual_seed=residual_seed,
                logit_seed=logit_seed,
                sketch_work_bytes=sketch_work_bytes,
            )
            result.append(
                Qwen3AttentionGroupSensitivity(
                    node=candidate,
                    role_activation_energy=baseline.attention_energy[
                        candidate.layer_index
                    ][candidate.group_index],
                    role_ablation_call_counts=role_counts,
                    **metrics,
                )
            )
        guard.checkpoint(f"attention_candidate_analysis_{start}")
    return tuple(result), forward_passes


def _highest_damage(items: Sequence[Any], count: int) -> frozenset[Any]:
    if count <= 0:
        return frozenset()
    ranked = sorted(
        items,
        key=lambda item: (
            -item.damage_score,
            item.node.layer_index,
            getattr(item.node, "neuron_index", getattr(item.node, "group_index", 0)),
        ),
    )
    return frozenset(item.node for item in ranked[:count])


def _anchors(
    layout: _Layout,
    neurons: tuple[Qwen3NeuronSensitivity, ...],
    attention: tuple[Qwen3AttentionGroupSensitivity, ...],
    *,
    explicit_neurons: tuple[Qwen3NeuronRef, ...],
    explicit_attention: tuple[Qwen3AttentionGroupRef, ...],
    neuron_top_k: int,
    attention_top_k: int,
) -> Qwen3ExactAnchorMasks:
    measured_neurons = _highest_damage(neurons, neuron_top_k)
    measured_attention = _highest_damage(attention, attention_top_k)
    exact_neurons = frozenset(explicit_neurons) | measured_neurons
    exact_attention = frozenset(explicit_attention) | measured_attention
    neuron_mask = tuple(
        tuple(
            Qwen3NeuronRef(layer, neuron) in exact_neurons
            for neuron in range(layout.intermediate_width)
        )
        for layer in range(len(layout.layers))
    )
    attention_mask = tuple(
        tuple(
            Qwen3AttentionGroupRef(layer, group) in exact_attention
            for group in range(layout.key_value_heads)
        )
        for layer in range(len(layout.layers))
    )
    return Qwen3ExactAnchorMasks(
        neuron_mask=neuron_mask,
        attention_group_mask=attention_mask,
        explicit_neurons=frozenset(explicit_neurons),
        measured_high_damage_neurons=measured_neurons,
        explicit_attention_groups=frozenset(explicit_attention),
        measured_high_damage_attention_groups=measured_attention,
    )


def _collect_qwen3_sensitivity_evidence_impl(
    model: nn.Module,
    calibration_input_ids: Tensor,
    gate_input_ids: Tensor,
    *,
    calibration_attention_mask: Tensor | None = None,
    gate_attention_mask: Tensor | None = None,
    calibration_position_ids: Tensor | None = None,
    gate_position_ids: Tensor | None = None,
    score_positions: Sequence[int] | None = None,
    neuron_candidates: Sequence[Qwen3NeuronRef | tuple[int, int]] | None = None,
    attention_group_candidates: Sequence[Qwen3AttentionGroupRef | tuple[int, int]]
    | None = None,
    explicit_neuron_anchors: Sequence[Qwen3NeuronRef | tuple[int, int]] = (),
    explicit_attention_group_anchors: Sequence[
        Qwen3AttentionGroupRef | tuple[int, int]
    ] = (),
    neuron_high_damage_anchor_count: int = 1,
    attention_high_damage_anchor_count: int = 1,
    sketch_dim: int = 16,
    limits: Qwen3SensitivityLimits | None = None,
    incremental_parent_manifest: Qwen3ModelFingerprintManifest | None = None,
    changed_state_prefixes: Sequence[str] = (),
    accepted_transaction_sha256: str | None = None,
    _guard: _ResourceGuard,
) -> Qwen3SensitivityEvidence:
    """Collect deterministic real-forward evidence without training or recovery.

    ``gate_input_ids`` are fingerprinted and checked for strict disjointness but
    are intentionally not evaluated here.  They remain untouched for the
    external accept/reject gate used after a proposed surgery transaction.
    """

    active_limits = limits or Qwen3SensitivityLimits()
    if not isinstance(active_limits, Qwen3SensitivityLimits):
        raise Qwen3SensitivityShapeError(
            "limits must be a Qwen3SensitivityLimits instance."
        )
    guard = _guard
    if not isinstance(guard, _ResourceGuard) or guard.stopped:
        raise Qwen3SensitivityStateError("A live resource guard is required.")
    guard.checkpoint("collector_entry")
    dimension = _positive_int(sketch_dim, "sketch_dim")
    if dimension > active_limits.max_sketch_dim:
        raise Qwen3SensitivityLimitError(
            f"sketch_dim {dimension} exceeds hard cap {active_limits.max_sketch_dim}."
        )
    neuron_anchor_count = _nonnegative_int(
        neuron_high_damage_anchor_count,
        "neuron_high_damage_anchor_count",
    )
    attention_anchor_count = _nonnegative_int(
        attention_high_damage_anchor_count,
        "attention_high_damage_anchor_count",
    )
    layout = _layout(model, active_limits)
    guard.checkpoint("layout_discovery_complete")
    calibration = _batch(
        calibration_input_ids,
        calibration_attention_mask,
        calibration_position_ids,
        split_name="calibration",
        max_rows=active_limits.max_calibration_rows,
        limits=active_limits,
    )
    gate = _batch(
        gate_input_ids,
        gate_attention_mask,
        gate_position_ids,
        split_name="gate",
        max_rows=active_limits.max_gate_rows,
        limits=active_limits,
    )
    guard.checkpoint("batch_preparation_complete")
    _strictly_disjoint(calibration, gate)
    positions = _score_indices(score_positions, calibration, active_limits)
    neurons = _refs(
        neuron_candidates,
        kind="neuron",
        layout=layout,
        maximum=active_limits.max_neuron_candidates,
    )
    attention = _refs(
        attention_group_candidates,
        kind="attention",
        layout=layout,
        maximum=active_limits.max_attention_candidates,
    )
    explicit_neurons = _refs(
        explicit_neuron_anchors,
        kind="neuron",
        layout=layout,
        maximum=active_limits.max_neuron_candidates,
    )
    explicit_attention = _refs(
        explicit_attention_group_anchors,
        kind="attention",
        layout=layout,
        maximum=active_limits.max_attention_candidates,
    )
    residual_layers = frozenset(item.layer_index for item in (*neurons, *attention))

    input_bytes = calibration.rows * calibration.sequence_length * (
        8 + 1 + 8
    ) + gate.rows * gate.sequence_length * (8 + 1 + 8)
    baseline_bytes = (
        len(residual_layers)
        * calibration.rows
        * calibration.sequence_length
        * layout.hidden_size
        * 4
        + calibration.rows * len(positions) * layout.vocabulary_size * 4
        + len(layout.layers) * layout.intermediate_width * 8
        + len(layout.layers) * layout.key_value_heads * 4 * 8
    )
    persistent_bytes = input_bytes + baseline_bytes
    activation_rows = calibration.rows * calibration.sequence_length
    (
        dense_mlp_forward_bound,
        codebook_materialization_bound,
        baseline_mlp_forward_bound,
    ) = _mlp_forward_workspace_bound(layout, calibration, 1)
    codebook_channels = min(
        layout.intermediate_width,
        active_limits.max_codebook_channels_per_chunk,
    )
    codebook_mlp_scratch = (
        (
            activation_rows * codebook_channels * 3
            + codebook_channels * layout.hidden_size * 4
            + codebook_channels * 4
        )
        * max(4, layout.activation_element_bytes)
        if "codebook" in layout.mlp_modes
        else 0
    )
    attention_capture_scratch = (
        activation_rows
        * (layout.query_heads + 2 * layout.key_value_heads)
        * layout.head_dim
        * max(4, layout.activation_element_bytes)
    )
    baseline_activation_scratch = max(
        codebook_mlp_scratch,
        attention_capture_scratch,
        baseline_mlp_forward_bound,
    )
    baseline_logit_scratch = (
        calibration.rows
        * len(positions)
        * layout.vocabulary_size
        * (4 + layout.activation_element_bytes)
    )
    baseline_attention_workspace = _attention_workspace_bound(layout, calibration, 1)
    baseline_peak = (
        persistent_bytes
        + max(baseline_activation_scratch, baseline_logit_scratch)
        + baseline_attention_workspace
    )
    if baseline_peak > active_limits.max_collector_bytes:
        raise Qwen3SensitivityLimitError(
            f"Baseline collector peak {baseline_peak} exceeds "
            f"max_collector_bytes={active_limits.max_collector_bytes}."
        )
    if neurons:
        neuron_chunk, neuron_peak, neuron_attention_workspace = _chunk_capacity(
            kind="neuron",
            layout=layout,
            batch=calibration,
            score_positions=positions,
            limits=active_limits,
            persistent_bytes=persistent_bytes,
        )
    else:
        neuron_chunk, neuron_peak = 1, baseline_peak
        neuron_attention_workspace = baseline_attention_workspace
    if attention:
        attention_chunk, attention_peak, candidate_attention_workspace = (
            _chunk_capacity(
                kind="attention",
                layout=layout,
                batch=calibration,
                score_positions=positions,
                limits=active_limits,
                persistent_bytes=persistent_bytes,
            )
        )
    else:
        attention_chunk, attention_peak = 1, baseline_peak
        candidate_attention_workspace = baseline_attention_workspace

    _, _, neuron_mlp_forward_bound = _mlp_forward_workspace_bound(
        layout,
        calibration,
        neuron_chunk if neurons else 1,
    )
    _, _, attention_mlp_forward_bound = _mlp_forward_workspace_bound(
        layout,
        calibration,
        attention_chunk if attention else 1,
    )
    predicted_forward_peak = max(
        baseline_peak,
        neuron_peak if neurons else baseline_peak,
        attention_peak if attention else baseline_peak,
    )
    _enforce_forward_resource_preflight(
        predicted_forward_peak,
        layout,
        active_limits,
    )
    guard.checkpoint("forward_resource_preflight_complete")
    inventory = _hook_inventory(model)
    runtime_payload = _runtime_identity(model, layout, active_limits)
    runtime_fingerprint = _sha256_json(runtime_payload)
    guard.checkpoint("runtime_identity_complete")
    model_manifest = fingerprint_qwen3_model_manifest(
        model,
        max_chunk_bytes=active_limits.max_fingerprint_chunk_bytes,
        parent=incremental_parent_manifest,
        changed_state_prefixes=changed_state_prefixes,
        accepted_transaction_sha256=accepted_transaction_sha256,
        checkpoint=guard.checkpoint,
    )
    guard.checkpoint("model_fingerprint_complete")
    model_fingerprint = model_manifest.root
    measurement_payload = {
        "schema": SENSITIVITY_SCHEMA,
        "model": model_fingerprint,
        "runtime": runtime_fingerprint,
        "calibration": calibration.fingerprint,
        "gate": gate.fingerprint,
        "score_positions": positions,
        "neurons": [[item.layer_index, item.neuron_index] for item in neurons],
        "attention": [[item.layer_index, item.group_index] for item in attention],
        "sketch_dim": dimension,
        "neuron_chunk": neuron_chunk,
        "attention_chunk": attention_chunk,
        "model_fingerprint_mode": model_manifest.mode,
        "parent_model": model_manifest.parent_root,
        "accepted_transaction_sha256": model_manifest.accepted_transaction_sha256,
        "changed_state_prefixes": model_manifest.changed_state_prefixes,
    }
    measurement_fingerprint = _sha256_json(measurement_payload)
    # Random count-sketch signs are part of the mathematical measurement, not
    # part of cache isolation.  Deriving them from ``runtime_fingerprint`` made
    # harmless process/source provenance changes silently rotate the feature
    # basis and therefore change candidate identities.  Keep the fail-closed
    # runtime hash in ``measurement_fingerprint`` and the cache key, while
    # deriving the sketch basis solely from semantic model/input/work content.
    sketch_basis_fingerprint = _sha256_json(
        {
            "schema": "neural-forge-qwen3-sensitivity-sketch-basis-v1",
            "model": model_fingerprint,
            "calibration": calibration.fingerprint,
            "score_positions": positions,
            "neurons": [
                [item.layer_index, item.neuron_index] for item in neurons
            ],
            "attention": [
                [item.layer_index, item.group_index] for item in attention
            ],
            "sketch_dim": dimension,
        }
    )
    cache_fingerprint = _sha256_json(
        {
            "measurement": measurement_fingerprint,
            "explicit_neuron_anchors": [
                [item.layer_index, item.neuron_index] for item in explicit_neurons
            ],
            "explicit_attention_anchors": [
                [item.layer_index, item.group_index] for item in explicit_attention
            ],
            "neuron_high_damage_anchor_count": neuron_anchor_count,
            "attention_high_damage_anchor_count": attention_anchor_count,
        }
    )
    fingerprints = Qwen3SensitivityFingerprints(
        model=model_fingerprint,
        calibration=calibration.fingerprint,
        gate=gate.fingerprint,
        runtime=runtime_fingerprint,
        measurement=measurement_fingerprint,
        cache=cache_fingerprint,
        calibration_row_fingerprints=calibration.row_fingerprints,
        gate_row_fingerprints=gate.row_fingerprints,
        model_manifest=model_manifest,
        sketch_basis=sketch_basis_fingerprint,
    )
    residual_seed = int(
        hashlib.sha256(
            f"{sketch_basis_fingerprint}:residual".encode()
        ).hexdigest()[:15],
        16,
    )
    logit_seed = int(
        hashlib.sha256(
            f"{sketch_basis_fingerprint}:logit".encode()
        ).hexdigest()[:15],
        16,
    )

    hooks_restored = False
    try:
        with torch.inference_mode():
            baseline = _baseline(
                model,
                layout,
                calibration,
                positions,
                residual_layers,
                active_limits,
                guard,
            )
            neuron_evidence, neuron_forwards = _measure_neurons(
                model,
                layout,
                calibration,
                positions,
                neurons,
                baseline,
                chunk_size=neuron_chunk,
                sketch_dim=dimension,
                residual_seed=residual_seed,
                logit_seed=logit_seed,
                sketch_work_bytes=active_limits.max_sketch_work_bytes,
                guard=guard,
            )
            attention_evidence, attention_forwards = _measure_attention(
                model,
                layout,
                calibration,
                positions,
                attention,
                baseline,
                chunk_size=attention_chunk,
                sketch_dim=dimension,
                residual_seed=residual_seed,
                logit_seed=logit_seed,
                sketch_work_bytes=active_limits.max_sketch_work_bytes,
                guard=guard,
            )
        guard.checkpoint("collection_complete")
    finally:
        hooks_restored = _finalize_model_integrity(
            model,
            model_manifest,
            inventory,
            max_chunk_bytes=active_limits.max_fingerprint_chunk_bytes,
            guard=guard,
        )
    anchor_masks = _anchors(
        layout,
        neuron_evidence,
        attention_evidence,
        explicit_neurons=explicit_neurons,
        explicit_attention=explicit_attention,
        neuron_top_k=neuron_anchor_count,
        attention_top_k=attention_anchor_count,
    )
    guard.close()
    report = Qwen3SensitivityReport(
        schema=SENSITIVITY_SCHEMA,
        logical_layers=len(layout.layers),
        intermediate_width=layout.intermediate_width,
        hidden_size=layout.hidden_size,
        key_value_groups=layout.key_value_heads,
        calibration_rows=calibration.rows,
        gate_rows=gate.rows,
        sequence_length=calibration.sequence_length,
        scored_positions_per_row=len(positions),
        neuron_candidates=len(neurons),
        attention_group_candidates=len(attention),
        baseline_forward_passes=1,
        candidate_forward_passes=neuron_forwards + attention_forwards,
        maximum_candidate_chunk=max(
            min(neuron_chunk, len(neurons)) if neurons else 0,
            min(attention_chunk, len(attention)) if attention else 0,
        ),
        tracked_peak_collector_bytes=max(
            baseline_peak,
            neuron_peak if neurons else persistent_bytes,
            attention_peak if attention else persistent_bytes,
        ),
        collector_byte_limit=active_limits.max_collector_bytes,
        process_rss_start_bytes=guard.process_rss_start,
        process_rss_peak_bytes=guard.process_rss_peak,
        process_rss_peak_increase_bytes=guard.process_rss_peak_increase,
        process_rss_increase_limit=active_limits.max_process_rss_increase_bytes,
        process_rss_sample_interval_seconds=guard.sample_interval_seconds,
        process_rss_sampler_stopped=guard.stopped,
        cuda_allocated_start_bytes=guard.cuda_allocated_start,
        cuda_allocated_peak_bytes=guard.cuda_allocated_peak,
        cuda_allocated_peak_increase_bytes=guard.cuda_allocated_peak_increase,
        cuda_allocated_increase_limit=(active_limits.max_cuda_allocated_increase_bytes),
        cuda_reserved_start_bytes=guard.cuda_reserved_start,
        cuda_reserved_peak_bytes=guard.cuda_reserved_peak,
        cuda_reserved_peak_increase_bytes=guard.cuda_reserved_peak_increase,
        cuda_reserved_increase_limit=active_limits.max_cuda_reserved_increase_bytes,
        resource_checkpoints=guard.checkpoints,
        attention_backend=layout.attention_backend,
        attention_workspace_bound_bytes=max(
            baseline_attention_workspace,
            neuron_attention_workspace,
            candidate_attention_workspace,
        ),
        dense_mlp_forward_bound_bytes=dense_mlp_forward_bound,
        codebook_materialization_bound_bytes=codebook_materialization_bound,
        mlp_forward_workspace_bound_bytes=max(
            baseline_mlp_forward_bound,
            neuron_mlp_forward_bound,
            attention_mlp_forward_bound,
        ),
        model_devices=layout.model_devices,
        mlp_capture_methods=baseline.capture_methods,
        evidence_method=DIRECTIONAL_DIFFERENCE_METHOD,
        exact_autodiff_used=False,
        directional_difference_reason=(
            "count sketches summarize finite zero-ablation output differences; "
            "they are generic evidence and are not JVP, VJP, Jacobian, or dual data"
        ),
        model_fingerprint_mode=model_manifest.mode,
        model_tensors_hashed=model_manifest.tensors_hashed,
        model_tensors_reused=model_manifest.tensors_reused,
        runtime_fingerprint=runtime_fingerprint,
        backward_calls=0,
        optimizer_steps=0,
        training_steps=0,
        recovery_steps=0,
        model_weights_mutated=False,
        hooks_restored=hooks_restored,
    )
    return Qwen3SensitivityEvidence(
        fingerprints=fingerprints,
        neuron_activation_energy=baseline.neuron_energy,
        attention_role_activation_energy=baseline.attention_energy,
        neurons=neuron_evidence,
        attention_groups=attention_evidence,
        anchors=anchor_masks,
        report=report,
    )


def _guard_model_devices(model: Any) -> tuple[str, ...]:
    if not isinstance(model, nn.Module):
        return ()
    devices: set[str] = set()
    for value in model.parameters():
        if value.device.type != "meta":
            devices.add(str(value.device))
    for value in model.buffers():
        if value.device.type != "meta":
            devices.add(str(value.device))
    return tuple(sorted(devices))


def collect_qwen3_sensitivity_evidence(
    model: nn.Module,
    calibration_input_ids: Tensor,
    gate_input_ids: Tensor,
    *,
    calibration_attention_mask: Tensor | None = None,
    gate_attention_mask: Tensor | None = None,
    calibration_position_ids: Tensor | None = None,
    gate_position_ids: Tensor | None = None,
    score_positions: Sequence[int] | None = None,
    neuron_candidates: Sequence[Qwen3NeuronRef | tuple[int, int]] | None = None,
    attention_group_candidates: Sequence[Qwen3AttentionGroupRef | tuple[int, int]]
    | None = None,
    explicit_neuron_anchors: Sequence[Qwen3NeuronRef | tuple[int, int]] = (),
    explicit_attention_group_anchors: Sequence[
        Qwen3AttentionGroupRef | tuple[int, int]
    ] = (),
    neuron_high_damage_anchor_count: int = 1,
    attention_high_damage_anchor_count: int = 1,
    sketch_dim: int = 16,
    limits: Qwen3SensitivityLimits | None = None,
    incremental_parent_manifest: Qwen3ModelFingerprintManifest | None = None,
    changed_state_prefixes: Sequence[str] = (),
    accepted_transaction_sha256: str | None = None,
) -> Qwen3SensitivityEvidence:
    """Collect bounded evidence under a guard that precedes every input clone."""

    active_limits = limits or Qwen3SensitivityLimits()
    if not isinstance(active_limits, Qwen3SensitivityLimits):
        raise Qwen3SensitivityShapeError(
            "limits must be a Qwen3SensitivityLimits instance."
        )
    guard = _ResourceGuard(active_limits)
    try:
        guard.bind_cuda_devices(_guard_model_devices(model))
        return _collect_qwen3_sensitivity_evidence_impl(
            model,
            calibration_input_ids,
            gate_input_ids,
            calibration_attention_mask=calibration_attention_mask,
            gate_attention_mask=gate_attention_mask,
            calibration_position_ids=calibration_position_ids,
            gate_position_ids=gate_position_ids,
            score_positions=score_positions,
            neuron_candidates=neuron_candidates,
            attention_group_candidates=attention_group_candidates,
            explicit_neuron_anchors=explicit_neuron_anchors,
            explicit_attention_group_anchors=explicit_attention_group_anchors,
            neuron_high_damage_anchor_count=neuron_high_damage_anchor_count,
            attention_high_damage_anchor_count=attention_high_damage_anchor_count,
            sketch_dim=sketch_dim,
            limits=active_limits,
            incremental_parent_manifest=incremental_parent_manifest,
            changed_state_prefixes=changed_state_prefixes,
            accepted_transaction_sha256=accepted_transaction_sha256,
            _guard=guard,
        )
    finally:
        guard.close()


__all__ = [
    "DIRECTIONAL_DIFFERENCE_METHOD",
    "FINITE_DIFFERENCE_METHOD",
    "Qwen3AttentionGroupRef",
    "Qwen3AttentionGroupSensitivity",
    "Qwen3ExactAnchorMasks",
    "Qwen3NeuronRef",
    "Qwen3NeuronSensitivity",
    "Qwen3ModelFingerprintManifest",
    "Qwen3SensitivityError",
    "Qwen3SensitivityEvidence",
    "Qwen3SensitivityFingerprints",
    "Qwen3SensitivityLimitError",
    "Qwen3SensitivityLimits",
    "Qwen3SensitivityReport",
    "Qwen3SensitivityShapeError",
    "Qwen3SensitivitySplitError",
    "Qwen3SensitivityStateError",
    "Qwen3StateTensorFingerprint",
    "collect_qwen3_sensitivity_evidence",
    "fingerprint_qwen3_model",
    "fingerprint_qwen3_model_manifest",
]
