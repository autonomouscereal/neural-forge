"""Unified zero-recovery allocation across Qwen3 MLP and attention surgery.

The allocator coordinates the existing neuron-codebook and coupled Q/K/V/O
bridges.  Candidate scores are damage/risk estimates, not claims of behavioral
improvement.  A mixed batch is installed tentatively and evaluated once; any
rejection or exception restores both registries and every bridge/allocator
bookkeeping field captured by the outer transaction.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
import os
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

import neural_forge.hf_qwen_adaptive_surgery as neuron_surgery
import neural_forge.hf_qwen_attention_surgery as attention_surgery
from neural_forge.adaptive_surgery_graph import (
    EvaluationDecision,
    MergeCandidate,
    SurgerySearchExhausted,
    SurgeryShapeError,
)
from neural_forge.hf_qwen_adaptive_surgery import (
    Qwen3ActionAudit,
    Qwen3AdaptiveSurgeryBridge,
    Qwen3AdaptiveSurgeryFeature,
    Qwen3AdaptiveSurgeryTransaction,
    Qwen3ProjectionCoefficients,
    Qwen3RuntimeAudit,
    Qwen3SurgeryEvidenceRefresh,
)
from neural_forge.hf_qwen_attention_surgery import (
    Qwen3AttentionEvidenceRefresh,
    Qwen3AttentionProjection,
    Qwen3AttentionRuntimeAudit,
    Qwen3AttentionSurgeryBridge,
    Qwen3AttentionSurgeryFeature,
    Qwen3AttentionSurgeryRefreshContext,
    Qwen3AttentionSurgeryTransaction,
)
from neural_forge.hf_qwen_codebook_model import (
    Qwen3CodebookParameterReport,
    qwen3_codebook_parameter_report,
)
from neural_forge.hf_qwen_sensitivity import (
    SENSITIVITY_SCHEMA,
    Qwen3ModelFingerprintManifest,
    Qwen3SensitivityEvidence,
    Qwen3StateTensorFingerprint,
)


SCHEMA_VERSION = 1
_REPORT_ATTRIBUTE = "neural_forge_codebook_report"
_METADATA_LIMIT = 4 * 1024 * 1024
_GIB = 1024**3
_HARD_LIMITS = {
    "max_proposals_per_modality": 4_096,
    "max_explicit_pairs_per_modality": 1_024,
    "max_batch_size": 256,
    "max_knapsack_states": 65_536,
    "max_expansion_work": 2_000_000,
    "max_attempts": 4_096,
    "history_limit": 4_096,
    "max_metadata_bytes": 64 * 1024 * 1024,
    "max_overshoot_bytes": 8 * _GIB,
    "max_manifest_transfer_bytes": 8 * _GIB,
    "max_process_rss_increase_bytes": 96 * _GIB,
    "max_cuda_allocated_increase_bytes": 96 * _GIB,
    "max_cuda_reserved_increase_bytes": 128 * _GIB,
}
Evaluator = Callable[..., bool | EvaluationDecision | Mapping[str, Any]]
RefreshEvidence = Callable[..., Qwen3SensitivityEvidence]
NeuronRefresh = Callable[..., Any]
AttentionRefresh = Callable[..., Any]
ProposalKind = Literal["neuron", "attention"]


class Qwen3SurgeryAllocatorError(ValueError):
    """Base error for unified Qwen3 surgery allocation."""


class Qwen3SurgeryAllocatorShapeError(Qwen3SurgeryAllocatorError):
    """An allocator limit, proposal, callback, or target is malformed."""


class Qwen3SurgeryAllocatorStateError(Qwen3SurgeryAllocatorError):
    """A mixed transaction could not preserve the prepared runtime."""


class Qwen3SurgeryAllocatorTargetError(Qwen3SurgeryAllocatorError):
    """No bounded positive-byte mixed batch can reach the requested target."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3SurgeryAllocatorShapeError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3SurgeryAllocatorShapeError(f"{name} must be positive.") from exc
    if result <= 0:
        raise Qwen3SurgeryAllocatorShapeError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3SurgeryAllocatorShapeError(f"{name} must be non-negative.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3SurgeryAllocatorShapeError(f"{name} must be non-negative.") from exc
    if result < 0:
        raise Qwen3SurgeryAllocatorShapeError(f"{name} must be non-negative.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Qwen3SurgeryAllocatorShapeError(
            f"{name} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Qwen3SurgeryAllocatorShapeError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Qwen3SurgeryAllocatorShapeError(
            f"{name} must be finite and non-negative."
        )
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_bytes(value: Tensor | None) -> int:
    return 0 if value is None else int(value.numel()) * int(value.element_size())


def _process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return 0


def _storage_identity(value: Tensor) -> tuple[str, int]:
    if value.device.type == "meta":
        return "meta", id(value)
    storage = value.untyped_storage()
    return str(value.device), int(getattr(storage, "_cdata", storage.data_ptr()))


def _tensor_version(value: Tensor) -> int | None:
    try:
        return int(value._version)
    except RuntimeError:
        return None


def _manifest_tensor_version(value: Tensor) -> int:
    version = _tensor_version(value)
    return -1 if version is None else version


def _walk_tensors(value: Any) -> tuple[Tensor, ...]:
    result: list[Tensor] = []
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Tensor):
            result.append(current)
        elif isinstance(current, Mapping):
            if id(current) in seen:
                continue
            seen.add(id(current))
            pending.extend(current.values())
        elif isinstance(current, (tuple, list)):
            if id(current) in seen:
                continue
            seen.add(id(current))
            pending.extend(current)
    return tuple(result)


class _ProtectedTensorMutationMode(TorchDispatchMode):
    def __init__(self, protected: frozenset[tuple[str, int]]) -> None:
        super().__init__()
        self.protected = protected
        self.attempted = False

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        keyword = dict(kwargs or {})
        schema = getattr(func, "_schema", None)
        if schema is not None and bool(getattr(schema, "is_mutable", False)):
            for index, argument in enumerate(schema.arguments):
                alias = getattr(argument, "alias_info", None)
                if alias is None or not bool(getattr(alias, "is_write", False)):
                    continue
                if index < len(args):
                    value = args[index]
                elif argument.name in keyword:
                    value = keyword[argument.name]
                else:
                    continue
                if any(
                    _storage_identity(tensor) in self.protected
                    for tensor in _walk_tensors(value)
                ):
                    self.attempted = True
                    raise Qwen3SurgeryAllocatorStateError(
                        "Evaluator/refresh callbacks cannot mutate live model tensors."
                    )
        return func(*args, **keyword)


def _capture_runtime_bindings(runtime: nn.Module) -> _RuntimeBindingCheckpoint:
    modules = tuple(runtime.modules())
    module_bindings = tuple(
        _ModuleBinding(
            module=module,
            training=bool(module.training),
            parameters=tuple(module._parameters.items()),
            buffers=tuple(module._buffers.items()),
            modules=tuple(module._modules.items()),
        )
        for module in modules
    )
    tensors: list[_TensorBinding] = []
    seen: set[int] = set()
    for module in modules:
        for value in (*module._parameters.values(), *module._buffers.values()):
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            tensors.append(
                _TensorBinding(
                    tensor=value,
                    data_alias=value.detach(),
                    version=_tensor_version(value),
                    requires_grad=bool(value.requires_grad),
                )
            )
    return _RuntimeBindingCheckpoint(tuple(tensors), module_bindings)


def _restore_runtime_bindings(checkpoint: _RuntimeBindingCheckpoint) -> bool:
    changed = False
    for item in checkpoint.modules:
        parameter_items = tuple(item.module._parameters.items())
        buffer_items = tuple(item.module._buffers.items())
        module_items = tuple(item.module._modules.items())
        parameters_match = len(parameter_items) == len(item.parameters) and all(
            left_name == right_name and left_value is right_value
            for (left_name, left_value), (right_name, right_value) in zip(
                parameter_items, item.parameters, strict=True
            )
        )
        buffers_match = len(buffer_items) == len(item.buffers) and all(
            left_name == right_name and left_value is right_value
            for (left_name, left_value), (right_name, right_value) in zip(
                buffer_items, item.buffers, strict=True
            )
        )
        modules_match = len(module_items) == len(item.modules) and all(
            left_name == right_name and left_value is right_value
            for (left_name, left_value), (right_name, right_value) in zip(
                module_items, item.modules, strict=True
            )
        )
        if (
            bool(item.module.training) != item.training
            or not parameters_match
            or not buffers_match
            or not modules_match
        ):
            changed = True
        item.module._parameters.clear()
        item.module._parameters.update(item.parameters)
        item.module._buffers.clear()
        item.module._buffers.update(item.buffers)
        item.module._modules.clear()
        item.module._modules.update(item.modules)
        item.module.training = item.training
    with torch.no_grad():
        for item in checkpoint.tensors:
            tensor = item.tensor
            if (
                _storage_identity(tensor) != _storage_identity(item.data_alias)
                or tuple(tensor.shape) != tuple(item.data_alias.shape)
                or tensor.dtype != item.data_alias.dtype
                or (
                    item.version is not None and _tensor_version(tensor) != item.version
                )
                or bool(tensor.requires_grad) != item.requires_grad
            ):
                changed = True
            if _storage_identity(tensor) != _storage_identity(item.data_alias):
                tensor.data = item.data_alias  # noqa: PGH003 - exact pointer restore
            if bool(tensor.requires_grad) != item.requires_grad:
                tensor.requires_grad_(item.requires_grad)
    return changed


class _GuardedInferenceRuntime:
    __slots__ = ("__forward", "__generate", "config")

    def __init__(self, runtime: nn.Module) -> None:
        self.__forward = runtime.__call__
        generate = getattr(runtime, "generate", None)
        self.__generate = generate if callable(generate) else None
        config = getattr(runtime, "config", None)
        payload = config.to_dict() if hasattr(config, "to_dict") else {}
        self.config = MappingProxyType(payload)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.__forward(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.__generate is None:
            raise Qwen3SurgeryAllocatorStateError(
                "The guarded runtime has no generation method."
            )
        return self.__generate(*args, **kwargs)


@contextmanager
def _isolated_callback_runtime(runtime: nn.Module):
    checkpoint = _capture_runtime_bindings(runtime)
    protected = frozenset(_storage_identity(item.tensor) for item in checkpoint.tensors)
    mutation_mode = _ProtectedTensorMutationMode(protected)
    python_state = random.getstate()
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except ImportError:
        np = None  # type: ignore[assignment]
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    callback_error: BaseException | None = None
    changed = False
    try:
        with torch.random.fork_rng(devices=devices, enabled=True), mutation_mode:
            yield _GuardedInferenceRuntime(runtime)
    except BaseException as error:
        callback_error = error
        raise
    finally:
        random.setstate(python_state)
        if numpy_state is not None and np is not None:
            np.random.set_state(numpy_state)
        changed = _restore_runtime_bindings(checkpoint)
        if callback_error is None and (changed or mutation_mode.attempted):
            raise Qwen3SurgeryAllocatorStateError(
                "Evaluator/refresh callback mutation was restored and rejected."
            )


def _unique_named_tensors(
    runtime: nn.Module,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    parameters = tuple(runtime.parameters())
    parameter_ids = {id(value) for value in parameters}
    buffers = tuple(
        value for value in runtime.buffers() if id(value) not in parameter_ids
    )
    return parameters, buffers


def _runtime_version_sha256(runtime: nn.Module) -> str:
    state = runtime.state_dict(keep_vars=True)
    return _sha256(
        {
            "tensors": [
                {
                    "name": name,
                    "shape": tuple(int(item) for item in state[name].shape),
                    "dtype": str(state[name].dtype),
                    "device": str(state[name].device),
                    "storage": _storage_identity(state[name]),
                    "version": _tensor_version(state[name]),
                }
                for name in sorted(state)
            ],
            "module_modes": [bool(module.training) for module in runtime.modules()],
        }
    )


def _logical_tensor_sha256(value: Tensor, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if value.device.type == "meta" or value.layout != torch.strided:
        raise Qwen3SurgeryAllocatorStateError(
            "Exact evidence verification requires materialized strided tensors."
        )
    digest = hashlib.sha256()
    # Non-contiguous tensors use int64 logical indices plus one value chunk.
    elements = max(1, chunk_bytes // max(int(value.element_size()), 8))
    detached = value.detach()
    if detached.is_contiguous():
        flat = detached.reshape(-1)
        for start in range(0, int(flat.numel()), elements):
            chunk = flat[start : start + elements].to("cpu").contiguous()
            digest.update(memoryview(chunk.view(torch.uint8).numpy()))
    else:
        total = int(detached.numel())
        for start in range(0, total, elements):
            indices = torch.arange(
                start,
                min(start + elements, total),
                device=detached.device,
                dtype=torch.int64,
            )
            chunk = torch.take(detached, indices).to("cpu").contiguous()
            digest.update(memoryview(chunk.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _logical_model_manifest(runtime: nn.Module) -> Qwen3ModelFingerprintManifest:
    config = getattr(runtime, "config", None)
    if config is None:
        raise Qwen3SurgeryAllocatorStateError(
            "Exact evidence verification requires runtime.config."
        )
    payload = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    config_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
    state = runtime.state_dict(keep_vars=True)
    tensors = tuple(
        Qwen3StateTensorFingerprint(
            name=name,
            sha256=_logical_tensor_sha256(state[name]),
            dtype=str(state[name].dtype),
            shape=tuple(int(item) for item in state[name].shape),
            device=str(state[name].device),
            data_ptr=(
                int(state[name].untyped_storage().data_ptr())
                if int(state[name].numel())
                else 0
            ),
            version=_manifest_tensor_version(state[name]),
        )
        for name in sorted(state)
    )
    root = _sha256(
        {
            "schema": SENSITIVITY_SCHEMA,
            "config_sha256": config_sha256,
            "tensors": [
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "dtype": item.dtype,
                    "shape": item.shape,
                }
                for item in tensors
            ],
        }
    )
    return Qwen3ModelFingerprintManifest(
        root=root,
        config_sha256=config_sha256,
        tensors=tensors,
        mode="full",
        tensors_hashed=len(tensors),
    )


@dataclass(frozen=True, slots=True)
class Qwen3SurgeryAllocatorLimits:
    """Hard bounds for discovery, mixed knapsack work, and retained history."""

    max_proposals_per_modality: int = 64
    max_explicit_pairs_per_modality: int = 64
    max_batch_size: int = 8
    max_knapsack_states: int = 4096
    max_expansion_work: int = 262_144
    max_attempts: int = 256
    history_limit: int = 64
    max_metadata_bytes: int = _METADATA_LIMIT
    max_overshoot_bytes: int = 64 * 1024 * 1024
    max_overshoot_ratio: float = 1.0
    max_manifest_transfer_bytes: int = 512 * 1024 * 1024
    max_process_rss_increase_bytes: int = 24 * _GIB
    max_cuda_allocated_increase_bytes: int = 24 * _GIB
    max_cuda_reserved_increase_bytes: int = 32 * _GIB

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "max_overshoot_ratio":
                value = _finite_nonnegative(getattr(self, name), name)
                if value > 16.0:
                    raise Qwen3SurgeryAllocatorShapeError(
                        "max_overshoot_ratio exceeds the immutable hard limit 16.0."
                    )
            else:
                value = _positive_int(getattr(self, name), name)
                hard_limit = _HARD_LIMITS.get(name)
                if hard_limit is not None and value > hard_limit:
                    raise Qwen3SurgeryAllocatorShapeError(
                        f"{name} exceeds immutable hard limit {hard_limit}."
                    )
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class Qwen3UnifiedSurgeryAccounting:
    """Exact current parameter and physical byte accounting."""

    source_unique_parameters: int
    child_unique_parameters: int
    parameter_tensor_bytes: int
    buffer_tensor_bytes: int
    tensor_bytes: int
    neuron_registry_bytes: int
    attention_registry_bytes: int
    index_bytes: int
    scale_bytes: int
    metadata_bytes: int
    deployable_bytes: int
    parameter_report_sha256: str
    neuron_registry_sha256: str
    attention_registry_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Qwen3CandidateValidationDamage:
    """Parent-relative marginal damage measured on untouched validation rows."""

    kind: ProposalKind
    candidate_sha256: str
    parent_state_sha256: str
    discovery_fingerprint: str
    validation_fingerprint: str
    discovery_row_fingerprints: tuple[str, ...]
    validation_row_fingerprints: tuple[str, ...]
    marginal_damage: float
    metric_name: str = "parent_relative_candidate_marginal_damage"
    parent_relative: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {"neuron", "attention"}:
            raise Qwen3SurgeryAllocatorShapeError(
                "Candidate validation kind must be neuron or attention."
            )
        for name in (
            "candidate_sha256",
            "parent_state_sha256",
            "discovery_fingerprint",
            "validation_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise Qwen3SurgeryAllocatorShapeError(
                    f"{name} must be a lowercase SHA-256 digest."
                )
        for name in ("discovery_row_fingerprints", "validation_row_fingerprints"):
            rows = tuple(getattr(self, name))
            if not rows or len(rows) != len(set(rows)):
                raise Qwen3SurgeryAllocatorShapeError(
                    f"{name} must contain unique row fingerprints."
                )
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in rows
            ):
                raise Qwen3SurgeryAllocatorShapeError(
                    f"{name} must contain lowercase SHA-256 digests."
                )
            object.__setattr__(self, name, rows)
        if set(self.discovery_row_fingerprints) & set(self.validation_row_fingerprints):
            raise Qwen3SurgeryAllocatorShapeError(
                "Discovery and validation rows must be provenance-disjoint."
            )
        if self.discovery_fingerprint == self.validation_fingerprint:
            raise Qwen3SurgeryAllocatorShapeError(
                "Discovery and validation fingerprints must differ."
            )
        if self.parent_relative is not True:
            raise Qwen3SurgeryAllocatorShapeError(
                "Candidate validation must be explicitly parent-relative."
            )
        object.__setattr__(
            self,
            "marginal_damage",
            _finite_nonnegative(self.marginal_damage, "marginal_damage"),
        )
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise Qwen3SurgeryAllocatorShapeError("metric_name must be non-empty.")

    @property
    def provenance_sha256(self) -> str:
        return _sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "candidate_sha256": self.candidate_sha256,
            "parent_state_sha256": self.parent_state_sha256,
            "discovery_fingerprint": self.discovery_fingerprint,
            "validation_fingerprint": self.validation_fingerprint,
            "discovery_row_fingerprints": list(self.discovery_row_fingerprints),
            "validation_row_fingerprints": list(self.validation_row_fingerprints),
            "marginal_damage": self.marginal_damage,
            "metric_name": self.metric_name,
            "parent_relative": self.parent_relative,
        }


@dataclass(frozen=True, slots=True)
class Qwen3UnifiedSurgeryProposal:
    """One bounded neuron or complete-attention merge candidate."""

    kind: ProposalKind
    candidate: MergeCandidate
    weighted_damage: float
    measured_heldout_damage: float | None
    projection_damage: float
    risk: float
    prototype_bytes_saved: int
    scale_setup_bytes: int
    individual_physical_bytes_saved: int
    risk_per_individual_physical_byte: float
    explicit_pair: bool
    protected_anchor: tuple[Any, ...] | None
    validation_parent_state_sha256: str | None = None
    validation_provenance_sha256: str | None = None

    @property
    def measured_validation_damage(self) -> float | None:
        return self.measured_heldout_damage

    @property
    def has_candidate_marginal_validation(self) -> bool:
        return (
            self.measured_heldout_damage is not None
            and self.validation_parent_state_sha256 is not None
            and self.validation_provenance_sha256 is not None
        )

    @property
    def proposal_sha256(self) -> str:
        return _sha256(self.as_dict())

    @property
    def source_nodes(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(tuple(item) for item in self.candidate.member_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "candidate_sha256": self.candidate.candidate_sha256,
            "weighted_damage": self.weighted_damage,
            "measured_heldout_damage": self.measured_heldout_damage,
            "projection_damage": self.projection_damage,
            "risk": self.risk,
            "prototype_bytes_saved": self.prototype_bytes_saved,
            "scale_setup_bytes": self.scale_setup_bytes,
            "individual_physical_bytes_saved": (self.individual_physical_bytes_saved),
            "risk_per_individual_physical_byte": (
                self.risk_per_individual_physical_byte
            ),
            "explicit_pair": self.explicit_pair,
            "protected_anchor": (
                None if self.protected_anchor is None else list(self.protected_anchor)
            ),
            "validation_parent_state_sha256": self.validation_parent_state_sha256,
            "validation_provenance_sha256": self.validation_provenance_sha256,
            "risk_is_candidate_marginal_validation": (
                self.has_candidate_marginal_validation
            ),
            "score_is_behavioral_improvement": False,
        }


@dataclass(frozen=True, slots=True)
class Qwen3MixedSurgeryBatch:
    """A bounded mixed knapsack selection against one exact parent state."""

    proposals: tuple[Qwen3UnifiedSurgeryProposal, ...]
    parent_state_sha256: str
    requested_physical_bytes: int
    prototype_bytes_saved: int
    scale_setup_bytes: int
    physical_bytes_saved: int
    total_risk: float
    proposal_work: int
    knapsack_states_considered: int

    @property
    def neuron_candidates(self) -> tuple[MergeCandidate, ...]:
        return tuple(item.candidate for item in self.proposals if item.kind == "neuron")

    @property
    def attention_candidates(self) -> tuple[MergeCandidate, ...]:
        return tuple(
            item.candidate for item in self.proposals if item.kind == "attention"
        )

    @property
    def batch_sha256(self) -> str:
        return _sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_sha256s": [item.proposal_sha256 for item in self.proposals],
            "parent_state_sha256": self.parent_state_sha256,
            "requested_physical_bytes": self.requested_physical_bytes,
            "prototype_bytes_saved": self.prototype_bytes_saved,
            "scale_setup_bytes": self.scale_setup_bytes,
            "physical_bytes_saved": self.physical_bytes_saved,
            "total_risk": self.total_risk,
            "risk_per_physical_byte": self.total_risk
            / max(self.physical_bytes_saved, 1),
            "proposal_work": self.proposal_work,
            "knapsack_states_considered": self.knapsack_states_considered,
            "score_is_behavioral_improvement": False,
        }


@dataclass(frozen=True, slots=True)
class Qwen3UnifiedSurgeryEvaluationContext:
    """Immutable tentative-state evidence plus a guarded inference-only runtime."""

    runtime: Any
    transaction_index: int
    evidence_epoch: int
    parent_state_sha256: str
    batch: Qwen3MixedSurgeryBatch
    parent_accounting: Qwen3UnifiedSurgeryAccounting
    proposed_accounting: Qwen3UnifiedSurgeryAccounting
    parent_neuron_snapshot: Any
    proposed_neuron_snapshot: Any
    parent_attention_snapshot: Any
    proposed_attention_snapshot: Any


@dataclass(frozen=True, slots=True)
class Qwen3UnifiedSurgeryRefreshContext:
    """Accepted-only context for replacing measured sensitivity evidence."""

    runtime: Any
    transaction_index: int
    next_evidence_epoch: int
    parent_state_sha256: str
    batch: Qwen3MixedSurgeryBatch
    proposed_accounting: Qwen3UnifiedSurgeryAccounting


@dataclass(frozen=True, slots=True)
class Qwen3UnifiedSurgeryTransaction:
    """One accepted commit or an ephemeral exactly restored rejection."""

    transaction_index: int
    evidence_epoch_before: int
    evidence_epoch_after: int
    batch_sha256: str
    proposal_sha256s: tuple[str, ...]
    accepted: bool
    decision: EvaluationDecision
    parent_accounting: Qwen3UnifiedSurgeryAccounting
    proposed_accounting: Qwen3UnifiedSurgeryAccounting
    committed_accounting: Qwen3UnifiedSurgeryAccounting
    rollback_exact: bool | None
    measured_behavioral_improvement_claim: bool = False
    optimizer_steps: int = 0
    backward_calls: int = 0
    process_rss_peak_increase_bytes: int = 0
    cuda_allocated_peak_increase_bytes: int = 0
    cuda_reserved_peak_increase_bytes: int = 0
    manifest_transfer_bytes: int = 0

    @property
    def zero_recovery(self) -> bool:
        return self.optimizer_steps == 0 and self.backward_calls == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_index": self.transaction_index,
            "evidence_epoch_before": self.evidence_epoch_before,
            "evidence_epoch_after": self.evidence_epoch_after,
            "batch_sha256": self.batch_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "accepted": self.accepted,
            "decision": self.decision.as_dict(),
            "parent_accounting": self.parent_accounting.as_dict(),
            "proposed_accounting": self.proposed_accounting.as_dict(),
            "committed_accounting": self.committed_accounting.as_dict(),
            "rollback_exact": self.rollback_exact,
            "measured_behavioral_improvement_claim": False,
            "zero_recovery": self.zero_recovery,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "process_rss_peak_increase_bytes": self.process_rss_peak_increase_bytes,
            "cuda_allocated_peak_increase_bytes": (
                self.cuda_allocated_peak_increase_bytes
            ),
            "cuda_reserved_peak_increase_bytes": (
                self.cuda_reserved_peak_increase_bytes
            ),
            "manifest_transfer_bytes": self.manifest_transfer_bytes,
        }


@dataclass(frozen=True, slots=True)
class Qwen3UnifiedSurgeryDiagnostics:
    accepted_transactions: int
    evidence_epoch: int
    proposal_rounds: int
    proposal_work: int
    knapsack_states_considered: int
    retained_history: int
    history_limit: int
    stale_explicit_pairs_skipped: int = 0
    candidate_validation_records: int = 0
    manifest_transfer_bytes: int = 0
    process_rss_peak_increase_bytes: int = 0
    cuda_allocated_peak_increase_bytes: int = 0
    cuda_reserved_peak_increase_bytes: int = 0
    optimizer_steps: int = 0
    backward_calls: int = 0

    @property
    def zero_recovery(self) -> bool:
        return self.optimizer_steps == 0 and self.backward_calls == 0


@dataclass(frozen=True, slots=True)
class Qwen3UnifiedSurgeryResult:
    runtime: nn.Module
    target_deployable_bytes: int
    initial_accounting: Qwen3UnifiedSurgeryAccounting
    final_accounting: Qwen3UnifiedSurgeryAccounting
    transactions: tuple[Qwen3UnifiedSurgeryTransaction, ...]
    diagnostics: Qwen3UnifiedSurgeryDiagnostics
    target_reached: bool

    @property
    def deployable_bytes_saved(self) -> int:
        return (
            self.initial_accounting.deployable_bytes
            - self.final_accounting.deployable_bytes
        )


@dataclass(frozen=True, slots=True)
class _KnapsackState:
    selected: tuple[Qwen3UnifiedSurgeryProposal, ...]
    occupied_neurons: frozenset[tuple[Any, ...]]
    occupied_attention: frozenset[tuple[Any, ...]]
    prototype_bytes: int
    neuron_scale_setup: int
    attention_scale_setup: int
    risk: float

    @property
    def setup_bytes(self) -> int:
        return self.neuron_scale_setup + self.attention_scale_setup

    @property
    def physical_bytes(self) -> int:
        return self.prototype_bytes - self.setup_bytes


@dataclass(frozen=True, slots=True)
class _NeuronBridgeCheckpoint:
    installed: Any
    graph: Any
    graph_checkpoint: Any
    features: tuple[Qwen3AdaptiveSurgeryFeature, ...]
    projections: tuple[Qwen3ProjectionCoefficients, ...]
    transactions: tuple[Any, ...]
    counters: tuple[int, ...]
    has_run: bool


@dataclass(frozen=True, slots=True)
class _AllocatorCheckpoint:
    neuron: _NeuronBridgeCheckpoint
    attention: Any
    attention_has_run: bool
    report: Qwen3CodebookParameterReport
    history: tuple[Qwen3UnifiedSurgeryTransaction, ...]
    counters: tuple[int, ...]
    evidence: Qwen3SensitivityEvidence | None
    candidate_validation: tuple[Qwen3CandidateValidationDamage, ...]
    accounting_cache: Qwen3UnifiedSurgeryAccounting | None


@dataclass(frozen=True, slots=True)
class _TensorBinding:
    tensor: Tensor
    data_alias: Tensor
    version: int | None
    requires_grad: bool


@dataclass(frozen=True, slots=True)
class _ModuleBinding:
    module: nn.Module
    training: bool
    parameters: tuple[tuple[str, Tensor | None], ...]
    buffers: tuple[tuple[str, Tensor | None], ...]
    modules: tuple[tuple[str, nn.Module | None], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeBindingCheckpoint:
    tensors: tuple[_TensorBinding, ...]
    modules: tuple[_ModuleBinding, ...]


@dataclass(slots=True)
class _ResourceGuard:
    limits: Qwen3SurgeryAllocatorLimits
    cuda_devices: tuple[torch.device, ...]
    process_rss_start: int
    process_rss_peak: int
    cuda_allocated_start: int
    cuda_reserved_start: int
    cuda_allocated_peak: int
    cuda_reserved_peak: int

    @classmethod
    def start(
        cls, runtime: nn.Module, limits: Qwen3SurgeryAllocatorLimits
    ) -> _ResourceGuard:
        devices = tuple(
            sorted(
                {
                    value.device
                    for value in (*runtime.parameters(), *runtime.buffers())
                    if value.device.type == "cuda"
                },
                key=str,
            )
        )
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        allocated = sum(int(torch.cuda.memory_allocated(item)) for item in devices)
        reserved = sum(int(torch.cuda.memory_reserved(item)) for item in devices)
        rss = _process_rss_bytes()
        return cls(
            limits,
            devices,
            rss,
            rss,
            allocated,
            reserved,
            allocated,
            reserved,
        )

    @property
    def process_rss_peak_increase(self) -> int:
        return max(0, self.process_rss_peak - self.process_rss_start)

    @property
    def cuda_allocated_peak_increase(self) -> int:
        return max(0, self.cuda_allocated_peak - self.cuda_allocated_start)

    @property
    def cuda_reserved_peak_increase(self) -> int:
        return max(0, self.cuda_reserved_peak - self.cuda_reserved_start)

    def sample(self) -> None:
        self.process_rss_peak = max(self.process_rss_peak, _process_rss_bytes())
        self.cuda_allocated_peak = max(
            self.cuda_allocated_peak,
            sum(
                int(torch.cuda.max_memory_allocated(item)) for item in self.cuda_devices
            ),
        )
        self.cuda_reserved_peak = max(
            self.cuda_reserved_peak,
            sum(
                int(torch.cuda.max_memory_reserved(item)) for item in self.cuda_devices
            ),
        )
        values = (
            (
                self.process_rss_peak_increase,
                self.limits.max_process_rss_increase_bytes,
                "process RSS",
            ),
            (
                self.cuda_allocated_peak_increase,
                self.limits.max_cuda_allocated_increase_bytes,
                "CUDA allocated memory",
            ),
            (
                self.cuda_reserved_peak_increase,
                self.limits.max_cuda_reserved_increase_bytes,
                "CUDA reserved memory",
            ),
        )
        for measured, limit, name in values:
            if measured > limit:
                raise Qwen3SurgeryAllocatorStateError(
                    f"{name} peak increase {measured} exceeded hard limit {limit}."
                )


def _decision(value: Any) -> EvaluationDecision:
    if isinstance(value, EvaluationDecision):
        return value
    if isinstance(value, bool):
        return EvaluationDecision(value)
    if not isinstance(value, Mapping):
        raise Qwen3SurgeryAllocatorShapeError(
            "evaluator must return bool, EvaluationDecision, or a mapping."
        )
    accepted = value.get("accepted", value.get("accept"))
    if not isinstance(accepted, bool):
        raise Qwen3SurgeryAllocatorShapeError(
            "evaluator mapping must contain boolean accepted or accept."
        )
    reason = value.get("reason", "")
    metrics = {
        str(key): item
        for key, item in value.items()
        if key not in {"accepted", "accept", "reason"}
        and isinstance(item, (str, int, float, bool, type(None)))
        and not (isinstance(item, float) and not math.isfinite(item))
    }
    return EvaluationDecision(accepted, str(reason), metrics)


def _invoke_callback(callback: Callable[..., Any], context: Any) -> Any:
    callable_value = getattr(callback, "evaluate", callback)
    if not callable(callable_value):
        raise Qwen3SurgeryAllocatorShapeError("callback must be callable.")
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError):
        return callable_value(context)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if not positional:
        return callable_value()
    if len(positional) == 1:
        return callable_value(context)
    return callable_value(context.runtime, context)


def _sync_parameter_report(runtime: nn.Module) -> Qwen3CodebookParameterReport:
    previous = qwen3_codebook_parameter_report(runtime)
    neuron = runtime.neuron_codebook.report
    attention = runtime.attention_codebook.report
    parameters, _ = _unique_named_tensors(runtime)
    actual_unique = sum(int(value.numel()) for value in parameters)
    registered = (
        previous.registered_logical_parameter_count
        - previous.neuron_unique_parameter_count
        - previous.attention_unique_parameter_count
        + neuron.unique_learned_parameters
        + attention.unique_learned_parameters
    )
    report = replace(
        previous,
        attention_prototypes_per_group=(attention.prototypes_per_group or 0),
        neuron_prototypes_per_neuron=(neuron.prototypes_per_neuron or 0),
        neuron_prototype_count=neuron.prototype_count,
        unique_parameter_count=actual_unique,
        registered_logical_parameter_count=registered,
        attention_unique_parameter_count=attention.unique_learned_parameters,
        neuron_unique_parameter_count=neuron.unique_learned_parameters,
        attention_index_bytes=attention.index_buffer_bytes,
        neuron_index_bytes=neuron.index_buffer_bytes,
        index_buffer_bytes=(
            attention.index_buffer_bytes
            + neuron.index_buffer_bytes
            + previous.token_row_index_bytes
        ),
    )
    setattr(runtime, _REPORT_ATTRIBUTE, report)
    return report


def _neuron_features_from_evidence(
    runtime: nn.Module, evidence: Qwen3SensitivityEvidence
) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
    measured = {
        (item.node.layer_index, item.node.neuron_index): item
        for item in evidence.neurons
    }
    converted = {
        (item.layer_index, item.neuron_index): item
        for item in evidence.to_adaptive_surgery_features()
    }
    snapshot = neuron_surgery.qwen3_runtime_assignment_snapshot(runtime)
    result: list[Qwen3AdaptiveSurgeryFeature] = []
    for node in snapshot.prototype_source_ids:
        layer, neuron = int(node[2]), int(node[3])
        if (layer, neuron) in measured:
            result.append(converted[(layer, neuron)])
            continue
        energy = evidence.neuron_activation_energy[layer][neuron]
        result.append(
            Qwen3AdaptiveSurgeryFeature(
                layer,
                neuron,
                activation=[energy],
                jacobian=[0.0],
                dual=[0.0],
                importance=1.0,
                anchor=(
                    "exact" if evidence.anchors.is_neuron_exact(layer, neuron) else None
                ),
            )
        )
    return tuple(result)


def _attention_features_from_evidence(
    runtime: nn.Module, evidence: Qwen3SensitivityEvidence
) -> tuple[Qwen3AttentionSurgeryFeature, ...]:
    measured = {
        (item.node.layer_index, item.node.group_index): item
        for item in evidence.attention_groups
    }
    snapshot = attention_surgery.qwen3_attention_runtime_snapshot(runtime)
    result: list[Qwen3AttentionSurgeryFeature] = []
    for node in snapshot.active_node_ids:
        layer, group = int(node[2]), int(node[3])
        item = measured.get((layer, group))
        roles = evidence.attention_role_activation_energy[layer][group]
        if item is None:
            activation = roles
            jvp: Sequence[float] = (0.0,)
            output: Sequence[float] = (0.0,)
            importance = 1.0
        else:
            activation = (
                *roles,
                item.residual_relative_l2,
                item.logit_relative_l2,
                item.logit_kl,
                item.top1_disagreement,
            )
            jvp = getattr(
                item,
                "finite_difference_controllability",
                getattr(item, "residual_directional_difference", (0.0,)),
            )
            output = getattr(
                item,
                "finite_difference_observability",
                getattr(item, "logit_directional_difference", (0.0,)),
            )
            importance = max(item.damage_score, 1e-12)
        result.append(
            Qwen3AttentionSurgeryFeature(
                layer,
                group,
                activation=activation,
                jvp=jvp,
                output=output,
                importance=importance,
                anchor=(
                    "exact"
                    if evidence.anchors.is_attention_group_exact(layer, group)
                    else None
                ),
            )
        )
    return tuple(result)


class Qwen3UnifiedSurgeryAllocator:
    """Allocate and atomically apply mixed Qwen3 codebook surgery."""

    def __init__(
        self,
        neuron_bridge: Qwen3AdaptiveSurgeryBridge,
        attention_bridge: Qwen3AttentionSurgeryBridge,
        *,
        sensitivity_evidence: Qwen3SensitivityEvidence | None = None,
        limits: Qwen3SurgeryAllocatorLimits | None = None,
    ) -> None:
        if not isinstance(neuron_bridge, Qwen3AdaptiveSurgeryBridge):
            raise Qwen3SurgeryAllocatorShapeError(
                "neuron_bridge must be Qwen3AdaptiveSurgeryBridge."
            )
        if not isinstance(attention_bridge, Qwen3AttentionSurgeryBridge):
            raise Qwen3SurgeryAllocatorShapeError(
                "attention_bridge must be Qwen3AttentionSurgeryBridge."
            )
        if neuron_bridge.runtime is not attention_bridge.runtime:
            raise Qwen3SurgeryAllocatorStateError(
                "Both bridges must own the same prepared Qwen3 runtime."
            )
        self.runtime = neuron_bridge.runtime
        self.neuron_bridge = neuron_bridge
        self.attention_bridge = attention_bridge
        self.limits = limits or Qwen3SurgeryAllocatorLimits()
        if not isinstance(self.limits, Qwen3SurgeryAllocatorLimits):
            raise Qwen3SurgeryAllocatorShapeError(
                "limits must be Qwen3SurgeryAllocatorLimits."
            )
        self._history: list[Qwen3UnifiedSurgeryTransaction] = []
        self._accepted_transactions = 0
        self._evidence_epoch = 0
        self._proposal_rounds = 0
        self._proposal_work = 0
        self._knapsack_states_considered = 0
        self._stale_explicit_pairs_skipped = 0
        self._manifest_transfer_bytes = 0
        self._resource_peaks = (0, 0, 0)
        self._candidate_validation: dict[
            tuple[ProposalKind, str], Qwen3CandidateValidationDamage
        ] = {}
        self._accounting_cache: Qwen3UnifiedSurgeryAccounting | None = None
        self._evidence = sensitivity_evidence
        if sensitivity_evidence is not None:
            self._validate_evidence_fingerprint(sensitivity_evidence)
        _sync_parameter_report(self.runtime)

    @classmethod
    def from_evidence(
        cls,
        runtime: nn.Module,
        evidence: Qwen3SensitivityEvidence,
        *,
        limits: Qwen3SurgeryAllocatorLimits | None = None,
        neuron_bridge_options: Mapping[str, Any] | None = None,
        attention_bridge_options: Mapping[str, Any] | None = None,
    ) -> Qwen3UnifiedSurgeryAllocator:
        if not isinstance(evidence, Qwen3SensitivityEvidence):
            raise Qwen3SurgeryAllocatorShapeError(
                "evidence must be Qwen3SensitivityEvidence."
            )
        neuron_options = dict(neuron_bridge_options or {})
        attention_options = dict(attention_bridge_options or {})
        return cls(
            Qwen3AdaptiveSurgeryBridge(
                runtime,
                _neuron_features_from_evidence(runtime, evidence),
                **neuron_options,
            ),
            Qwen3AttentionSurgeryBridge(
                runtime,
                _attention_features_from_evidence(runtime, evidence),
                **attention_options,
            ),
            sensitivity_evidence=evidence,
            limits=limits,
        )

    @property
    def history(self) -> tuple[Qwen3UnifiedSurgeryTransaction, ...]:
        return tuple(self._history)

    @property
    def evidence_epoch(self) -> int:
        return self._evidence_epoch

    @property
    def diagnostics(self) -> Qwen3UnifiedSurgeryDiagnostics:
        return Qwen3UnifiedSurgeryDiagnostics(
            accepted_transactions=self._accepted_transactions,
            evidence_epoch=self._evidence_epoch,
            proposal_rounds=self._proposal_rounds,
            proposal_work=self._proposal_work,
            knapsack_states_considered=self._knapsack_states_considered,
            retained_history=len(self._history),
            history_limit=self.limits.history_limit,
            stale_explicit_pairs_skipped=self._stale_explicit_pairs_skipped,
            candidate_validation_records=len(self._candidate_validation),
            manifest_transfer_bytes=self._manifest_transfer_bytes,
            process_rss_peak_increase_bytes=self._resource_peaks[0],
            cuda_allocated_peak_increase_bytes=self._resource_peaks[1],
            cuda_reserved_peak_increase_bytes=self._resource_peaks[2],
        )

    def _validate_evidence_fingerprint(
        self, evidence: Qwen3SensitivityEvidence
    ) -> None:
        if not isinstance(evidence, Qwen3SensitivityEvidence):
            raise Qwen3SurgeryAllocatorShapeError(
                "refresh_evidence must return Qwen3SensitivityEvidence."
            )
        expected = evidence.fingerprints.model_manifest
        if (
            expected.root != evidence.fingerprints.model
            or expected.root != evidence.fingerprints.model_manifest.root
        ):
            raise Qwen3SurgeryAllocatorStateError(
                "Sensitivity evidence carries inconsistent model fingerprints."
            )
        config = getattr(self.runtime, "config", None)
        payload = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        if (
            hashlib.sha256(_canonical_json(payload)).hexdigest()
            != expected.config_sha256
        ):
            raise Qwen3SurgeryAllocatorStateError(
                "Sensitivity evidence config fingerprint does not match runtime."
            )
        current = self.runtime.state_dict(keep_vars=True)
        expected_by_name = {item.name: item for item in expected.tensors}
        if set(current) != set(expected_by_name):
            raise Qwen3SurgeryAllocatorStateError(
                "Sensitivity evidence tensor manifest does not match runtime names."
            )
        for name in sorted(current):
            value = current[name]
            item = expected_by_name[name]
            if (
                tuple(int(part) for part in value.shape) != item.shape
                or str(value.dtype) != item.dtype
                or str(value.device) != item.device
            ):
                raise Qwen3SurgeryAllocatorStateError(
                    f"Sensitivity evidence tensor metadata changed for {name}."
                )
            pointer = int(value.untyped_storage().data_ptr()) if value.numel() else 0
            if (
                pointer == item.data_ptr
                and _manifest_tensor_version(value) == item.version
            ):
                continue
            transfer = _tensor_bytes(value)
            if (
                self._manifest_transfer_bytes + transfer
                > self.limits.max_manifest_transfer_bytes
            ):
                raise Qwen3SurgeryAllocatorStateError(
                    "Incremental evidence verification exceeded its transfer ceiling."
                )
            self._manifest_transfer_bytes += transfer
            if _logical_tensor_sha256(value) != item.sha256:
                raise Qwen3SurgeryAllocatorStateError(
                    f"Sensitivity evidence tensor fingerprint changed for {name}."
                )

    def _compute_accounting(
        self,
        neuron_snapshot: Any | None = None,
        attention_snapshot: Any | None = None,
    ) -> Qwen3UnifiedSurgeryAccounting:
        report = _sync_parameter_report(self.runtime)
        neuron = (
            neuron_surgery.qwen3_runtime_assignment_snapshot(self.runtime)
            if neuron_snapshot is None
            else neuron_snapshot
        )
        attention = (
            attention_surgery.qwen3_attention_runtime_snapshot(self.runtime)
            if attention_snapshot is None
            else attention_snapshot
        )
        parameters, buffers = _unique_named_tensors(self.runtime)
        parameter_bytes = sum(_tensor_bytes(value) for value in parameters)
        buffer_bytes = sum(_tensor_bytes(value) for value in buffers)
        neuron_registry = self.runtime.neuron_codebook.registry
        attention_registry = self.runtime.attention_codebook.registry
        scale_bytes = _tensor_bytes(neuron_registry.neuron_scales) + _tensor_bytes(
            attention_registry.role_scales
        )
        metadata_payload = {
            "schema": "neural-forge-qwen3-unified-surgery-v1",
            "parameter_report": report.as_dict(),
            "neuron_registry_sha256": neuron.registry_sha256,
            "attention_registry_sha256": attention.registry_sha256,
            "evidence_epoch": self._evidence_epoch,
            "evidence_fingerprint": (
                None if self._evidence is None else self._evidence.fingerprints.cache
            ),
        }
        metadata_bytes = len(_canonical_json(metadata_payload))
        if metadata_bytes > self.limits.max_metadata_bytes:
            raise Qwen3SurgeryAllocatorStateError(
                "Bounded deployable metadata limit was exceeded."
            )
        tensor_bytes = parameter_bytes + buffer_bytes
        return Qwen3UnifiedSurgeryAccounting(
            source_unique_parameters=report.source_unique_parameter_count,
            child_unique_parameters=report.unique_parameter_count,
            parameter_tensor_bytes=parameter_bytes,
            buffer_tensor_bytes=buffer_bytes,
            tensor_bytes=tensor_bytes,
            neuron_registry_bytes=neuron.tensor_bytes,
            attention_registry_bytes=attention.tensor_bytes,
            index_bytes=report.index_buffer_bytes,
            scale_bytes=scale_bytes,
            metadata_bytes=metadata_bytes,
            deployable_bytes=tensor_bytes + metadata_bytes,
            parameter_report_sha256=_sha256(report.as_dict()),
            neuron_registry_sha256=neuron.registry_sha256,
            attention_registry_sha256=attention.registry_sha256,
        )

    def accounting(self) -> Qwen3UnifiedSurgeryAccounting:
        if self._accounting_cache is None:
            self._accounting_cache = self._compute_accounting()
        return self._accounting_cache

    def _state_sha256(self) -> str:
        accounting = self.accounting()
        return _sha256(
            {
                "accounting": accounting.as_dict(),
                "runtime_version_manifest": _runtime_version_sha256(self.runtime),
                "neuron_graph": self.neuron_bridge.graph.snapshot.snapshot_sha256,
                "attention_graph": self.attention_bridge.graph.snapshot.snapshot_sha256,
                "neuron_history": [
                    item.as_dict() for item in self.neuron_bridge.transactions
                ],
                "attention_history": [
                    item.as_dict() for item in self.attention_bridge.transactions
                ],
                "neuron_counters": (
                    self.neuron_bridge._attempt_count,
                    self.neuron_bridge._accepted_transaction_count,
                    self.neuron_bridge._rejected_transaction_count,
                    self.neuron_bridge._feature_recomputations,
                    self.neuron_bridge._graph_rounds,
                ),
                "attention_diagnostics": self.attention_bridge.diagnostics.as_dict(),
                "allocator": (
                    self._accepted_transactions,
                    self._evidence_epoch,
                    len(self._history),
                ),
            }
        )

    def _neuron_checkpoint(self) -> _NeuronBridgeCheckpoint:
        bridge = self.neuron_bridge
        return _NeuronBridgeCheckpoint(
            installed=neuron_surgery._capture_installed(self.runtime),
            graph=bridge._graph,
            graph_checkpoint=bridge._graph_checkpoint(),
            features=bridge._features,
            projections=bridge._projection_coefficients,
            transactions=tuple(bridge._transactions),
            counters=(
                bridge._attempt_count,
                bridge._accepted_transaction_count,
                bridge._rejected_transaction_count,
                bridge._feature_recomputations,
                bridge._graph_rounds,
            ),
            has_run=bridge._has_run,
        )

    def _checkpoint(self) -> _AllocatorCheckpoint:
        return _AllocatorCheckpoint(
            neuron=self._neuron_checkpoint(),
            attention=self.attention_bridge._checkpoint(),
            attention_has_run=self.attention_bridge._has_run,
            report=qwen3_codebook_parameter_report(self.runtime),
            history=tuple(self._history),
            counters=(
                self._accepted_transactions,
                self._evidence_epoch,
                self._proposal_rounds,
                self._proposal_work,
                self._knapsack_states_considered,
            ),
            evidence=self._evidence,
            candidate_validation=tuple(self._candidate_validation.values()),
            accounting_cache=self._accounting_cache,
        )

    def _restore_checkpoint(self, checkpoint: _AllocatorCheckpoint) -> None:
        neuron = checkpoint.neuron
        setattr(self.runtime, _REPORT_ATTRIBUTE, checkpoint.report)
        neuron_surgery._install_codebook(self.runtime, neuron.installed.codebook)
        attention_surgery._force_install_attention_codebook(
            self.runtime, checkpoint.attention.installed.codebook
        )
        setattr(self.runtime, _REPORT_ATTRIBUTE, checkpoint.report)
        if (
            neuron_surgery.qwen3_runtime_assignment_snapshot(self.runtime)
            != neuron.installed.public
        ):
            raise Qwen3SurgeryAllocatorStateError(
                "Outer rollback failed to restore the neuron registry."
            )
        if (
            attention_surgery.qwen3_attention_runtime_snapshot(self.runtime)
            != checkpoint.attention.installed.snapshot
        ):
            raise Qwen3SurgeryAllocatorStateError(
                "Outer rollback failed to restore the attention registry."
            )
        self.attention_bridge._restore_checkpoint(checkpoint.attention)
        self.attention_bridge._has_run = checkpoint.attention_has_run
        bridge = self.neuron_bridge
        bridge._graph = neuron.graph
        bridge._restore_graph_checkpoint(neuron.graph_checkpoint)
        bridge._features = neuron.features
        bridge._projection_coefficients = neuron.projections
        bridge._transactions = list(neuron.transactions)
        (
            bridge._attempt_count,
            bridge._accepted_transaction_count,
            bridge._rejected_transaction_count,
            bridge._feature_recomputations,
            bridge._graph_rounds,
        ) = neuron.counters
        bridge._has_run = neuron.has_run
        setattr(self.runtime, _REPORT_ATTRIBUTE, checkpoint.report)
        self._history = list(checkpoint.history)
        (
            self._accepted_transactions,
            self._evidence_epoch,
            self._proposal_rounds,
            self._proposal_work,
            self._knapsack_states_considered,
        ) = checkpoint.counters
        self._evidence = checkpoint.evidence
        self._candidate_validation = {
            (item.kind, item.candidate_sha256): item
            for item in checkpoint.candidate_validation
        }
        self._accounting_cache = checkpoint.accounting_cache
        if qwen3_codebook_parameter_report(self.runtime) != checkpoint.report:
            raise Qwen3SurgeryAllocatorStateError(
                "Outer rollback failed to restore the parameter report."
            )

    def install_candidate_validation(
        self, records: Sequence[Qwen3CandidateValidationDamage]
    ) -> None:
        if isinstance(records, (str, bytes)):
            raise Qwen3SurgeryAllocatorShapeError(
                "Candidate validation records must be a sequence."
            )
        values = tuple(records)
        if len(values) > 2 * self.limits.max_proposals_per_modality:
            raise Qwen3SurgeryAllocatorShapeError(
                "Candidate validation records exceed bounded proposal work."
            )
        parent_state = self._state_sha256()
        expected_rows: frozenset[str] | None = None
        expected_discovery: str | None = None
        if self._evidence is not None:
            expected_discovery = self._evidence.fingerprints.cache
            expected_rows = frozenset(
                (
                    *self._evidence.fingerprints.calibration_row_fingerprints,
                    *self._evidence.fingerprints.gate_row_fingerprints,
                )
            )
        installed = dict(self._candidate_validation)
        for item in values:
            if not isinstance(item, Qwen3CandidateValidationDamage):
                raise Qwen3SurgeryAllocatorShapeError(
                    "Every validation record must be Qwen3CandidateValidationDamage."
                )
            if item.parent_state_sha256 != parent_state:
                raise Qwen3SurgeryAllocatorStateError(
                    "Candidate validation was measured against a stale parent state."
                )
            if expected_discovery is not None and (
                item.discovery_fingerprint != expected_discovery
                or frozenset(item.discovery_row_fingerprints) != expected_rows
            ):
                raise Qwen3SurgeryAllocatorStateError(
                    "Candidate validation discovery provenance does not match evidence."
                )
            key = (item.kind, item.candidate_sha256)
            prior = installed.get(key)
            if prior is not None and prior != item:
                raise Qwen3SurgeryAllocatorStateError(
                    "Conflicting validation records exist for one candidate."
                )
            installed[key] = item
        if len(installed) > 2 * self.limits.max_proposals_per_modality:
            raise Qwen3SurgeryAllocatorShapeError(
                "Accumulated candidate validation exceeds bounded proposal work."
            )
        self._candidate_validation = installed

    def _validation_for(
        self,
        kind: ProposalKind,
        candidate: MergeCandidate,
        parent_state_sha256: str,
    ) -> Qwen3CandidateValidationDamage | None:
        item = self._candidate_validation.get((kind, candidate.candidate_sha256))
        if item is None:
            return None
        if item.parent_state_sha256 != parent_state_sha256:
            return None
        if self._evidence is None:
            if item.discovery_fingerprint != candidate.evidence_sha256:
                return None
        elif item.discovery_fingerprint != self._evidence.fingerprints.cache:
            return None
        return item

    def _proposal_validation_is_current(
        self,
        proposal: Qwen3UnifiedSurgeryProposal,
        parent_state_sha256: str,
    ) -> bool:
        record = self._candidate_validation.get(
            (proposal.kind, proposal.candidate.candidate_sha256)
        )
        return (
            record is not None
            and record.parent_state_sha256 == parent_state_sha256
            and proposal.has_candidate_marginal_validation
            and proposal.validation_parent_state_sha256 == parent_state_sha256
            and proposal.validation_provenance_sha256 == record.provenance_sha256
            and proposal.measured_validation_damage == record.marginal_damage
            and proposal.risk == record.marginal_damage
        )

    def _neuron_proposal(
        self,
        candidate: MergeCandidate,
        *,
        explicit: bool,
        parent_state_sha256: str,
    ) -> Qwen3UnifiedSurgeryProposal:
        base = candidate.score.parameter_bytes_saved
        physical = self.neuron_bridge._marginal_bytes_saved((candidate,))
        setup = base - physical
        validation = self._validation_for("neuron", candidate, parent_state_sha256)
        measured = None if validation is None else validation.marginal_damage
        risk = measured if measured is not None else candidate.score.weighted_damage
        return Qwen3UnifiedSurgeryProposal(
            kind="neuron",
            candidate=candidate,
            weighted_damage=candidate.score.weighted_damage,
            measured_heldout_damage=measured,
            projection_damage=0.0,
            risk=risk,
            prototype_bytes_saved=base,
            scale_setup_bytes=max(setup, 0),
            individual_physical_bytes_saved=physical,
            risk_per_individual_physical_byte=(
                risk / physical if physical > 0 else risk + setup + 1.0
            ),
            explicit_pair=explicit,
            protected_anchor=(
                None
                if candidate.exact_anchor is None
                else tuple(candidate.exact_anchor)
            ),
            validation_parent_state_sha256=(
                None if validation is None else validation.parent_state_sha256
            ),
            validation_provenance_sha256=(
                None if validation is None else validation.provenance_sha256
            ),
        )

    def _attention_proposal(
        self,
        candidate: MergeCandidate,
        *,
        explicit: bool,
        parent_state_sha256: str,
    ) -> Qwen3UnifiedSurgeryProposal:
        scored = self.attention_bridge.candidate_score(candidate)
        validation = self._validation_for("attention", candidate, parent_state_sha256)
        measured = None if validation is None else validation.marginal_damage
        risk = (
            measured
            if measured is not None
            else candidate.score.weighted_damage + max(scored.projection_damage, 0.0)
        )
        return Qwen3UnifiedSurgeryProposal(
            kind="attention",
            candidate=candidate,
            weighted_damage=candidate.score.weighted_damage,
            measured_heldout_damage=measured,
            projection_damage=scored.projection_damage,
            risk=risk,
            prototype_bytes_saved=scored.prototype_bytes_saved,
            scale_setup_bytes=scored.scale_allocation_bytes,
            individual_physical_bytes_saved=scored.physical_bytes_saved,
            risk_per_individual_physical_byte=(
                risk / scored.physical_bytes_saved
                if scored.physical_bytes_saved > 0
                else risk + scored.scale_allocation_bytes + 1.0
            ),
            explicit_pair=explicit,
            protected_anchor=(
                None
                if candidate.exact_anchor is None
                else tuple(candidate.exact_anchor)
            ),
            validation_parent_state_sha256=(
                None if validation is None else validation.parent_state_sha256
            ),
            validation_provenance_sha256=(
                None if validation is None else validation.provenance_sha256
            ),
        )

    @staticmethod
    def _pairs(values: Sequence[Any], limit: int, name: str) -> tuple[Any, ...]:
        if isinstance(values, (str, bytes)):
            raise Qwen3SurgeryAllocatorShapeError(f"{name} must be a sequence.")
        result = tuple(values)
        if len(result) > limit:
            raise Qwen3SurgeryAllocatorShapeError(
                f"{name} exceeds its explicit-pair work limit."
            )
        if any(
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
            for item in result
        ):
            raise Qwen3SurgeryAllocatorShapeError(
                f"Every {name} entry must contain two node selectors."
            )
        return result

    def _exact_neuron_nodes(self) -> frozenset[tuple[Any, ...]]:
        return frozenset(
            tuple(item.node_id)
            for item in self.neuron_bridge.features
            if item.anchor is not None and getattr(item.anchor, "mode", None) == "exact"
        )

    def _exact_attention_nodes(self) -> frozenset[tuple[Any, ...]]:
        return frozenset(
            tuple(item.node_id)
            for item in self.attention_bridge.features
            if item.anchor is not None and getattr(item.anchor, "mode", None) == "exact"
        )

    @staticmethod
    def _candidate_touches_exact(
        candidate: MergeCandidate, exact_nodes: frozenset[tuple[Any, ...]]
    ) -> bool:
        return bool({tuple(node) for node in candidate.member_ids} & exact_nodes) or (
            candidate.exact_anchor is not None
            and tuple(candidate.exact_anchor) in exact_nodes
        )

    def discover_proposals(
        self,
        *,
        explicit_neuron_pairs: Sequence[Any] = (),
        explicit_attention_pairs: Sequence[Any] = (),
        excluded_sha256s: frozenset[str] = frozenset(),
    ) -> tuple[Qwen3UnifiedSurgeryProposal, ...]:
        neuron_pairs = self._pairs(
            explicit_neuron_pairs,
            self.limits.max_explicit_pairs_per_modality,
            "explicit_neuron_pairs",
        )
        attention_pairs = self._pairs(
            explicit_attention_pairs,
            self.limits.max_explicit_pairs_per_modality,
            "explicit_attention_pairs",
        )
        proposals: list[Qwen3UnifiedSurgeryProposal] = []
        parent_state_sha256 = self._state_sha256()
        exact_neurons = self._exact_neuron_nodes()
        exact_attention = self._exact_attention_nodes()

        neuron_checkpoint = self.neuron_bridge._graph_checkpoint()
        try:
            explicit_hashes: set[str] = set()
            neuron_candidates: list[MergeCandidate] = []
            for pair in neuron_pairs:
                left = neuron_surgery._node_ref(pair[0])
                right = neuron_surgery._node_ref(pair[1])
                if tuple(left) in exact_neurons or tuple(right) in exact_neurons:
                    raise Qwen3SurgeryAllocatorShapeError(
                        "Explicit neuron pairs cannot touch exact anchors."
                    )
                try:
                    candidate = self.neuron_bridge.graph.propose_candidate_for_pair(
                        left, right
                    )
                except SurgeryShapeError as error:
                    if "Unknown node ID" not in str(error):
                        raise
                    self._stale_explicit_pairs_skipped += 1
                    continue
                neuron_candidates.append(candidate)
                explicit_hashes.add(candidate.candidate_sha256)
            try:
                neuron_candidates.extend(self.neuron_bridge.graph.propose_candidates())
            except SurgerySearchExhausted:
                pass
            seen: set[str] = set()
            for candidate in neuron_candidates:
                if self._candidate_touches_exact(candidate, exact_neurons):
                    continue
                if candidate.candidate_sha256 in seen:
                    continue
                seen.add(candidate.candidate_sha256)
                proposals.append(
                    self._neuron_proposal(
                        candidate,
                        explicit=candidate.candidate_sha256 in explicit_hashes,
                        parent_state_sha256=parent_state_sha256,
                    )
                )
                if len(seen) >= self.limits.max_proposals_per_modality:
                    break
        finally:
            self.neuron_bridge._restore_graph_checkpoint(neuron_checkpoint)

        attention_checkpoint = self.attention_bridge._graph_checkpoint()
        try:
            explicit_hashes = set()
            attention_candidates: list[MergeCandidate] = []
            for pair in attention_pairs:
                left = attention_surgery._node_ref(pair[0])
                right = attention_surgery._node_ref(pair[1])
                if tuple(left) in exact_attention or tuple(right) in exact_attention:
                    raise Qwen3SurgeryAllocatorShapeError(
                        "Explicit attention pairs cannot touch exact anchors."
                    )
                try:
                    candidate = self.attention_bridge.graph.propose_candidate_for_pair(
                        left, right
                    )
                except SurgeryShapeError as error:
                    if "Unknown node ID" not in str(error):
                        raise
                    self._stale_explicit_pairs_skipped += 1
                    continue
                attention_candidates.append(candidate)
                explicit_hashes.add(candidate.candidate_sha256)
            try:
                attention_candidates.extend(
                    self.attention_bridge.graph.propose_candidates()
                )
            except SurgerySearchExhausted:
                pass
            seen = set()
            for candidate in attention_candidates:
                if self._candidate_touches_exact(candidate, exact_attention):
                    continue
                if candidate.candidate_sha256 in seen:
                    continue
                seen.add(candidate.candidate_sha256)
                proposals.append(
                    self._attention_proposal(
                        candidate,
                        explicit=candidate.candidate_sha256 in explicit_hashes,
                        parent_state_sha256=parent_state_sha256,
                    )
                )
                if len(seen) >= self.limits.max_proposals_per_modality:
                    break
        finally:
            self.attention_bridge._restore_graph(
                self.attention_bridge.graph, attention_checkpoint
            )

        proposals = [
            item
            for item in proposals
            if item.proposal_sha256 not in excluded_sha256s
            and item.candidate.candidate_sha256 not in excluded_sha256s
            and item.prototype_bytes_saved > 0
        ]
        proposals.sort(
            key=lambda item: (
                item.risk_per_individual_physical_byte,
                item.risk,
                -item.prototype_bytes_saved,
                item.kind,
                item.proposal_sha256,
            )
        )
        self._proposal_rounds += 1
        self._proposal_work += len(proposals)
        return tuple(proposals)

    @staticmethod
    def _conflicts(
        state: _KnapsackState, proposal: Qwen3UnifiedSurgeryProposal
    ) -> bool:
        nodes = frozenset(proposal.source_nodes)
        occupied = (
            state.occupied_neurons
            if proposal.kind == "neuron"
            else state.occupied_attention
        )
        return bool(nodes & occupied)

    def _extend_state(
        self, state: _KnapsackState, proposal: Qwen3UnifiedSurgeryProposal
    ) -> _KnapsackState:
        nodes = frozenset(proposal.source_nodes)
        if proposal.kind == "neuron":
            neurons = state.occupied_neurons | nodes
            attention = state.occupied_attention
            neuron_setup = max(state.neuron_scale_setup, proposal.scale_setup_bytes)
            attention_setup = state.attention_scale_setup
        else:
            neurons = state.occupied_neurons
            attention = state.occupied_attention | nodes
            neuron_setup = state.neuron_scale_setup
            attention_setup = max(
                state.attention_scale_setup, proposal.scale_setup_bytes
            )
        return _KnapsackState(
            selected=state.selected + (proposal,),
            occupied_neurons=neurons,
            occupied_attention=attention,
            prototype_bytes=state.prototype_bytes + proposal.prototype_bytes_saved,
            neuron_scale_setup=neuron_setup,
            attention_scale_setup=attention_setup,
            risk=state.risk + proposal.risk,
        )

    @staticmethod
    def _dominates(left: _KnapsackState, right: _KnapsackState) -> bool:
        return (
            left.physical_bytes >= right.physical_bytes
            and left.risk <= right.risk
            and left.setup_bytes <= right.setup_bytes
            and left.occupied_neurons.issubset(right.occupied_neurons)
            and left.occupied_attention.issubset(right.occupied_attention)
            and (
                left.physical_bytes > right.physical_bytes
                or left.risk < right.risk
                or left.setup_bytes < right.setup_bytes
                or left.occupied_neurons != right.occupied_neurons
                or left.occupied_attention != right.occupied_attention
            )
        )

    def _capped_pareto(
        self,
        states: Sequence[_KnapsackState],
        target: int,
        work: list[int],
    ) -> list[_KnapsackState]:
        ordered = sorted(
            states,
            key=lambda item: (
                item.risk,
                item.setup_bytes,
                -min(max(item.physical_bytes, 0), target),
                abs(item.physical_bytes - target),
                len(item.selected),
                tuple(proposal.proposal_sha256 for proposal in item.selected),
            ),
        )
        frontier: list[_KnapsackState] = []
        for candidate in ordered:
            dominated = False
            survivors: list[_KnapsackState] = []
            for prior in frontier:
                work[0] += 1
                if work[0] > self.limits.max_expansion_work:
                    raise Qwen3SurgeryAllocatorTargetError(
                        "Pareto expansion exceeded immutable work ceiling."
                    )
                if self._dominates(prior, candidate):
                    dominated = True
                    break
                if not self._dominates(candidate, prior):
                    survivors.append(prior)
            if not dominated:
                survivors.append(candidate)
                frontier = survivors
            if len(frontier) > self.limits.max_knapsack_states:
                frontier.sort(
                    key=lambda item: (
                        item.risk,
                        max(item.physical_bytes - target, 0),
                        -min(item.physical_bytes, target),
                        item.setup_bytes,
                        len(item.selected),
                    )
                )
                frontier = frontier[: self.limits.max_knapsack_states]
        return frontier

    def select_batch(
        self,
        requested_physical_bytes: int,
        *,
        explicit_neuron_pairs: Sequence[Any] = (),
        explicit_attention_pairs: Sequence[Any] = (),
        excluded_sha256s: frozenset[str] = frozenset(),
        unsafe_allow_unvalidated_candidates: bool = False,
    ) -> Qwen3MixedSurgeryBatch:
        target = _positive_int(requested_physical_bytes, "requested_physical_bytes")
        parent_state = self._state_sha256()
        proposals = self.discover_proposals(
            explicit_neuron_pairs=explicit_neuron_pairs,
            explicit_attention_pairs=explicit_attention_pairs,
            excluded_sha256s=excluded_sha256s,
        )
        if not unsafe_allow_unvalidated_candidates:
            proposals = tuple(
                item
                for item in proposals
                if self._proposal_validation_is_current(item, parent_state)
            )
        if not proposals:
            raise Qwen3SurgeryAllocatorTargetError(
                "No bounded, parent-state candidate-marginal validation remains."
            )
        empty = _KnapsackState((), frozenset(), frozenset(), 0, 0, 0, 0.0)
        states = [empty]
        considered = 1
        expansion_work = [0]
        for proposal in proposals:
            additions = []
            for state in states:
                expansion_work[0] += 1
                if expansion_work[0] > self.limits.max_expansion_work:
                    raise Qwen3SurgeryAllocatorTargetError(
                        "Knapsack expansion exceeded immutable work ceiling."
                    )
                if len(state.selected) >= self.limits.max_batch_size:
                    continue
                if self._conflicts(state, proposal):
                    continue
                additions.append(self._extend_state(state, proposal))
            considered += len(additions)
            states.extend(additions)
            states = self._capped_pareto(states, target, expansion_work)
        viable = [item for item in states if item.selected and item.physical_bytes > 0]
        if not viable:
            raise Qwen3SurgeryAllocatorTargetError(
                "All bounded batches have non-positive true physical savings."
            )
        allowed_overshoot = max(
            self.limits.max_overshoot_bytes,
            int(math.ceil(target * self.limits.max_overshoot_ratio)),
        )
        meeting = [
            item
            for item in viable
            if target <= item.physical_bytes <= target + allowed_overshoot
        ]
        partial = [item for item in viable if item.physical_bytes < target]
        if not meeting and not partial:
            raise Qwen3SurgeryAllocatorTargetError(
                "Every target-reaching batch exceeds the configured overshoot ceiling."
            )
        selected = min(
            meeting or partial,
            key=(
                (
                    lambda item: (
                        item.risk,
                        item.physical_bytes - target,
                        item.setup_bytes,
                        len(item.selected),
                        tuple(proposal.proposal_sha256 for proposal in item.selected),
                    )
                )
                if meeting
                else lambda item: (
                    -item.physical_bytes,
                    item.risk,
                    item.setup_bytes,
                    len(item.selected),
                )
            ),
        )
        self._knapsack_states_considered += considered
        return Qwen3MixedSurgeryBatch(
            proposals=selected.selected,
            parent_state_sha256=parent_state,
            requested_physical_bytes=target,
            prototype_bytes_saved=selected.prototype_bytes,
            scale_setup_bytes=selected.setup_bytes,
            physical_bytes_saved=selected.physical_bytes,
            total_risk=selected.risk,
            proposal_work=len(proposals),
            knapsack_states_considered=considered,
        )

    def _assert_zero_gradient(self) -> None:
        if any(parameter.grad is not None for parameter in self.runtime.parameters()):
            raise Qwen3SurgeryAllocatorStateError(
                "Zero-recovery surgery forbids backward/gradient state."
            )

    @staticmethod
    def _assert_parent_relative_gate(
        decision: EvaluationDecision,
        parent_state_sha256: str,
        *,
        unsafe_allow_ungated_commit: bool,
    ) -> None:
        if not decision.accepted or unsafe_allow_ungated_commit:
            return
        metrics = dict(decision.metrics)
        fingerprint = metrics.get("evaluation_fingerprint")
        if (
            metrics.get("parent_relative") is not True
            or metrics.get("parent_state_sha256") != parent_state_sha256
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise Qwen3SurgeryAllocatorStateError(
                "Accepted surgery requires an auditable parent-relative behavioral "
                "gate with parent_state_sha256 and evaluation_fingerprint."
            )

    def _record_resource_peaks(self, guard: _ResourceGuard) -> None:
        guard.sample()
        self._resource_peaks = tuple(
            max(old, new)
            for old, new in zip(
                self._resource_peaks,
                (
                    guard.process_rss_peak_increase,
                    guard.cuda_allocated_peak_increase,
                    guard.cuda_reserved_peak_increase,
                ),
                strict=True,
            )
        )

    @staticmethod
    def _preserve_neuron_policy(
        old: Sequence[Qwen3AdaptiveSurgeryFeature],
        fresh: Sequence[Qwen3AdaptiveSurgeryFeature],
    ) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
        old_by_node = {item.node_id: item for item in old}
        result = []
        for item in fresh:
            prior = old_by_node.get(item.node_id)
            if prior is None:
                raise Qwen3SurgeryAllocatorStateError(
                    "Evidence refresh introduced a non-active neuron node."
                )
            result.append(
                Qwen3AdaptiveSurgeryFeature(
                    item.layer_index,
                    item.neuron_index,
                    weight_features=item.weight_features,
                    activation_features=item.activation_features,
                    rollout_jvp_features=item.jacobian_features,
                    output_dual_features=item.dual_features,
                    domain_features=item.domain_features,
                    importance=prior.importance,
                    anchor=prior.anchor,
                )
            )
        return tuple(result)

    @staticmethod
    def _preserve_attention_policy(
        old: Sequence[Qwen3AttentionSurgeryFeature],
        fresh: Sequence[Qwen3AttentionSurgeryFeature],
    ) -> tuple[Qwen3AttentionSurgeryFeature, ...]:
        old_by_node = {item.node_id: item for item in old}
        result = []
        for item in fresh:
            prior = old_by_node.get(item.node_id)
            if prior is None:
                raise Qwen3SurgeryAllocatorStateError(
                    "Evidence refresh introduced a non-active attention node."
                )
            result.append(
                Qwen3AttentionSurgeryFeature(
                    item.layer_index,
                    item.group_index,
                    weight_features=item.weight_features,
                    activation_features=item.activation_features,
                    rollout_jvp_features=item.jvp_features,
                    output_dual_features=item.output_features,
                    domain_features=item.domain_features,
                    importance=prior.importance,
                    anchor=prior.anchor,
                )
            )
        return tuple(result)

    def _combined_refresh(
        self,
        callback: RefreshEvidence,
        context: Qwen3UnifiedSurgeryRefreshContext,
    ) -> tuple[Qwen3SurgeryEvidenceRefresh, Qwen3AttentionEvidenceRefresh]:
        evidence = _invoke_callback(callback, context)
        self._validate_evidence_fingerprint(evidence)
        neuron_features = self._preserve_neuron_policy(
            self.neuron_bridge.features,
            _neuron_features_from_evidence(self.runtime, evidence),
        )
        attention_features = self._preserve_attention_policy(
            self.attention_bridge.features,
            _attention_features_from_evidence(self.runtime, evidence),
        )
        self._evidence = evidence
        return (
            Qwen3SurgeryEvidenceRefresh(
                neuron_features, self.neuron_bridge.projection_coefficients
            ),
            Qwen3AttentionEvidenceRefresh(
                attention_features, self.attention_bridge.projections
            ),
        )

    def _commit_neuron(
        self,
        candidates: tuple[MergeCandidate, ...],
        proposal: Any,
        parent: Any,
        proposed: Any,
        decision: EvaluationDecision,
        refresh: Qwen3SurgeryEvidenceRefresh,
    ) -> None:
        bridge = self.neuron_bridge
        changes = neuron_surgery._assignment_changes(parent.public, proposed.public)
        round_index = bridge._graph_rounds
        outcome = bridge._graph.commit(proposal, decision)
        next_graph = bridge._build_graph(refresh.features)
        used_pairs = {
            (
                neuron_surgery._node_ref(binding.source),
                neuron_surgery._node_ref(binding.target),
            )
            for binding in outcome.transaction.projection_bindings
        }
        transaction = Qwen3AdaptiveSurgeryTransaction(
            transaction_index=bridge._attempt_count + 1,
            round_index=round_index,
            accepted=True,
            decision=decision,
            actions=tuple(
                Qwen3ActionAudit(action.kind, action.action_sha256)
                for action in outcome.transaction.actions
            ),
            graph_transaction_sha256=outcome.transaction.transaction_sha256,
            parent_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(parent.public),
            proposed_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(proposed.public),
            committed_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(proposed.public),
            assignment_change_count=len(changes),
            assignment_changes_sha256=neuron_surgery._sha256(
                [item.as_dict() for item in changes]
            ),
            projection_coefficients=tuple(
                item
                for item in bridge.projection_coefficients
                if (item.source, item.target) in used_pairs
            ),
            rollback_exact=None,
        )
        bridge._features = refresh.features
        bridge._projection_coefficients = refresh.projection_coefficients
        bridge._graph = next_graph
        bridge._graph_rounds += 1
        bridge._append_transaction(transaction)

    def _commit_attention(
        self,
        candidates: tuple[MergeCandidate, ...],
        proposal: Any,
        parent: Any,
        proposed: Any,
        used_projections: tuple[Qwen3AttentionProjection, ...],
        decision: EvaluationDecision,
        refresh: Qwen3AttentionEvidenceRefresh,
    ) -> None:
        bridge = self.attention_bridge
        active_nodes = set(proposed.snapshot.active_node_ids)
        bridge._validate_refresh(refresh, active_nodes)
        changes = attention_surgery._assignment_changes(
            parent.snapshot, proposed.snapshot
        )
        round_index = bridge._graph_rounds
        epoch_before = bridge._evidence_epoch
        outcome = bridge._graph.commit(proposal, decision)
        next_graph = bridge._build_graph(refresh.features, refresh.projections)
        bridge._attempt_count += 1
        bridge._accepted_count += 1
        bridge._candidate_batches += 1
        bridge._evidence_epoch += 1
        bridge._features = refresh.features
        bridge._projections = refresh.projections
        bridge._graph = next_graph
        bridge._graph_rounds += 1
        bridge._append_transaction(
            Qwen3AttentionSurgeryTransaction(
                transaction_index=bridge._attempt_count,
                round_index=round_index,
                evidence_epoch_before=epoch_before,
                evidence_epoch_after=bridge._evidence_epoch,
                accepted=True,
                decision=decision,
                graph_transaction_sha256=outcome.transaction.transaction_sha256,
                action_sha256s=tuple(
                    action.action_sha256 for action in outcome.transaction.actions
                ),
                candidate_sha256s=tuple(
                    candidate.candidate_sha256 for candidate in candidates
                ),
                parent_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                    parent.snapshot
                ),
                proposed_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                    proposed.snapshot
                ),
                committed_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                    proposed.snapshot
                ),
                assignment_change_count=len(changes),
                assignment_changes_sha256=attention_surgery._sha256(changes),
                projections=used_projections,
                physical_bytes_saved=(
                    parent.snapshot.tensor_bytes - proposed.snapshot.tensor_bytes
                ),
                rollback_exact=None,
                measured_improvement_claim=False,
            )
        )

    def _refresh_without_neuron_edit(
        self, refresh: Qwen3SurgeryEvidenceRefresh
    ) -> None:
        active = set(self.neuron_bridge.graph.snapshot.prototype_ids)
        if {item.node_id for item in refresh.features} != active:
            raise Qwen3SurgeryAllocatorStateError(
                "Neuron refresh did not preserve every active node."
            )
        self.neuron_bridge._features = refresh.features
        self.neuron_bridge._projection_coefficients = refresh.projection_coefficients
        self.neuron_bridge._graph = self.neuron_bridge._build_graph(refresh.features)
        self.neuron_bridge._feature_recomputations += 1
        self.neuron_bridge._graph_rounds += 1

    def _refresh_without_attention_edit(
        self, refresh: Qwen3AttentionEvidenceRefresh
    ) -> None:
        active = set(self.attention_bridge.runtime_snapshot.active_node_ids)
        self.attention_bridge._validate_refresh(refresh, active)
        self.attention_bridge._features = refresh.features
        self.attention_bridge._projections = refresh.projections
        self.attention_bridge._graph = self.attention_bridge._build_graph(
            refresh.features, refresh.projections
        )
        self.attention_bridge._feature_refreshes += 1
        self.attention_bridge._evidence_epoch += 1
        self.attention_bridge._graph_rounds += 1

    def apply_batch(
        self,
        batch: Qwen3MixedSurgeryBatch,
        *,
        evaluator: Evaluator | None,
        refresh_evidence: RefreshEvidence | None = None,
        refresh_neuron_features: NeuronRefresh | None = None,
        refresh_attention_features: AttentionRefresh | None = None,
        unsafe_allow_ungated_commit: bool = False,
        unsafe_allow_unvalidated_candidates: bool = False,
    ) -> Qwen3UnifiedSurgeryTransaction:
        if not isinstance(batch, Qwen3MixedSurgeryBatch) or not batch.proposals:
            raise Qwen3SurgeryAllocatorShapeError(
                "batch must be a non-empty Qwen3MixedSurgeryBatch."
            )
        if refresh_evidence is not None and (
            refresh_neuron_features is not None
            or refresh_attention_features is not None
        ):
            raise Qwen3SurgeryAllocatorShapeError(
                "Use refresh_evidence or modality refresh callbacks, not both."
            )
        if evaluator is None and not unsafe_allow_ungated_commit:
            raise Qwen3SurgeryAllocatorStateError(
                "A parent-relative behavioral evaluator is required for commit."
            )
        if batch.parent_state_sha256 != self._state_sha256():
            raise Qwen3SurgeryAllocatorStateError("Cannot apply a stale mixed batch.")
        if not unsafe_allow_unvalidated_candidates and any(
            not self._proposal_validation_is_current(item, batch.parent_state_sha256)
            for item in batch.proposals
        ):
            raise Qwen3SurgeryAllocatorStateError(
                "Every committed proposal requires parent-state candidate-marginal "
                "validation evidence."
            )
        exact_by_kind = {
            "neuron": self._exact_neuron_nodes(),
            "attention": self._exact_attention_nodes(),
        }
        if any(
            self._candidate_touches_exact(item.candidate, exact_by_kind[item.kind])
            for item in batch.proposals
        ):
            raise Qwen3SurgeryAllocatorStateError(
                "Exact anchors cannot be members or medoids of a surgery batch."
            )
        self._assert_zero_gradient()
        resource_guard = _ResourceGuard.start(self.runtime, self.limits)
        manifest_transfer_start = self._manifest_transfer_bytes
        checkpoint = self._checkpoint()
        parent_digest = self._state_sha256()
        parent_accounting = self.accounting()
        parent_neuron = checkpoint.neuron.installed
        parent_attention = checkpoint.attention.installed
        neuron_candidates = batch.neuron_candidates
        attention_candidates = batch.attention_candidates
        neuron_proposal = None
        attention_proposal = None
        used_attention_projections: tuple[Qwen3AttentionProjection, ...] = ()
        try:
            resource_guard.sample()
            if neuron_candidates:
                neuron_proposal = self.neuron_bridge.graph.preview(neuron_candidates)
                with torch.inference_mode():
                    codebook = neuron_surgery._compacted_codebook(
                        self.runtime,
                        neuron_proposal.snapshot,
                        neuron_proposal.transaction.actions,
                        projection_coefficients=(
                            self.neuron_bridge.projection_coefficients
                        ),
                        projection_bindings=(
                            neuron_proposal.transaction.projection_bindings
                        ),
                    )
                    neuron_surgery._install_codebook(self.runtime, codebook)
                self._accounting_cache = None
                _sync_parameter_report(self.runtime)
                resource_guard.sample()
            if attention_candidates:
                attention_proposal = self.attention_bridge.graph.preview(
                    attention_candidates
                )

                def projection_for_pair(source, target):
                    return self.attention_bridge._projection_for_pair_from(
                        source, target, self.attention_bridge.projections
                    )

                with torch.inference_mode():
                    codebook, used_attention_projections = (
                        attention_surgery._compacted_attention_codebook(
                            self.runtime,
                            attention_proposal.snapshot,
                            projection_for_pair,
                        )
                    )
                    attention_surgery._install_attention_codebook(
                        self.runtime, codebook
                    )
                self._accounting_cache = None
                _sync_parameter_report(self.runtime)
                resource_guard.sample()
            proposed_neuron = neuron_surgery._capture_installed(self.runtime)
            proposed_attention = attention_surgery._capture_installed(self.runtime)
            neuron_saved = (
                parent_neuron.public.tensor_bytes - proposed_neuron.public.tensor_bytes
            )
            attention_saved = (
                parent_attention.snapshot.tensor_bytes
                - proposed_attention.snapshot.tensor_bytes
            )
            if neuron_saved + attention_saved != batch.physical_bytes_saved:
                raise Qwen3SurgeryAllocatorStateError(
                    "Mixed knapsack bytes did not match installed physical tensors."
                )
            if attention_candidates:
                self.attention_bridge._assert_exact_anchors(
                    parent_attention.snapshot, proposed_attention.snapshot
                )
            exact_neurons = {
                item.node_id
                for item in self.neuron_bridge.features
                if item.anchor is not None and item.anchor.mode == "exact"
            }
            if not exact_neurons.issubset(
                set(proposed_neuron.public.prototype_source_ids)
            ):
                raise Qwen3SurgeryAllocatorStateError(
                    "A mixed proposal removed an immutable exact neuron anchor."
                )
            self._accounting_cache = self._compute_accounting(
                proposed_neuron.public, proposed_attention.snapshot
            )
            proposed_accounting = self.accounting()
            with _isolated_callback_runtime(self.runtime) as guarded_runtime:
                context = Qwen3UnifiedSurgeryEvaluationContext(
                    runtime=guarded_runtime,
                    transaction_index=self._accepted_transactions + 1,
                    evidence_epoch=self._evidence_epoch,
                    parent_state_sha256=parent_digest,
                    batch=batch,
                    parent_accounting=parent_accounting,
                    proposed_accounting=proposed_accounting,
                    parent_neuron_snapshot=parent_neuron.public,
                    proposed_neuron_snapshot=proposed_neuron.public,
                    parent_attention_snapshot=parent_attention.snapshot,
                    proposed_attention_snapshot=proposed_attention.snapshot,
                )
                with torch.inference_mode():
                    decision = (
                        EvaluationDecision.accept(
                            "unsafe ungated commit",
                            {
                                "unsafe_debug_override": True,
                                "parent_state_sha256": parent_digest,
                            },
                        )
                        if evaluator is None
                        else _decision(_invoke_callback(evaluator, context))
                    )
            self._assert_parent_relative_gate(
                decision,
                parent_digest,
                unsafe_allow_ungated_commit=unsafe_allow_ungated_commit,
            )
            resource_guard.sample()
            self._assert_zero_gradient()
            if self.accounting() != proposed_accounting:
                raise Qwen3SurgeryAllocatorStateError(
                    "The external evaluator mutated the tentative surgery state."
                )
            if not decision.accepted:
                self._restore_checkpoint(checkpoint)
                self._record_resource_peaks(resource_guard)
                if self._state_sha256() != parent_digest:
                    raise Qwen3SurgeryAllocatorStateError(
                        "Rejected mixed transaction did not restore exact state."
                    )
                return Qwen3UnifiedSurgeryTransaction(
                    transaction_index=self._accepted_transactions + 1,
                    evidence_epoch_before=self._evidence_epoch,
                    evidence_epoch_after=self._evidence_epoch,
                    batch_sha256=batch.batch_sha256,
                    proposal_sha256s=tuple(
                        item.proposal_sha256 for item in batch.proposals
                    ),
                    accepted=False,
                    decision=decision,
                    parent_accounting=parent_accounting,
                    proposed_accounting=proposed_accounting,
                    committed_accounting=parent_accounting,
                    rollback_exact=True,
                    process_rss_peak_increase_bytes=(
                        resource_guard.process_rss_peak_increase
                    ),
                    cuda_allocated_peak_increase_bytes=(
                        resource_guard.cuda_allocated_peak_increase
                    ),
                    cuda_reserved_peak_increase_bytes=(
                        resource_guard.cuda_reserved_peak_increase
                    ),
                    manifest_transfer_bytes=(
                        self._manifest_transfer_bytes - manifest_transfer_start
                    ),
                )

            combined_refresh = None
            if refresh_evidence is not None:
                with _isolated_callback_runtime(self.runtime) as guarded_runtime:
                    refresh_context = Qwen3UnifiedSurgeryRefreshContext(
                        runtime=guarded_runtime,
                        transaction_index=self._accepted_transactions + 1,
                        next_evidence_epoch=self._evidence_epoch + 1,
                        parent_state_sha256=parent_digest,
                        batch=batch,
                        proposed_accounting=proposed_accounting,
                    )
                    combined_refresh = self._combined_refresh(
                        refresh_evidence, refresh_context
                    )
                resource_guard.sample()

            if neuron_candidates:
                assert neuron_proposal is not None
                if combined_refresh is not None:
                    neuron_refresh = combined_refresh[0]
                elif refresh_neuron_features is None:
                    neuron_refresh = self.neuron_bridge._next_features(
                        neuron_proposal.snapshot,
                        neuron_proposal.transaction,
                        proposed_neuron.public.tensor_bytes,
                        None,
                    )
                else:
                    with _isolated_callback_runtime(self.runtime):
                        neuron_refresh = self.neuron_bridge._next_features(
                            neuron_proposal.snapshot,
                            neuron_proposal.transaction,
                            proposed_neuron.public.tensor_bytes,
                            refresh_neuron_features,
                        )
                    resource_guard.sample()
                if combined_refresh is not None:
                    self.neuron_bridge._feature_recomputations += 1
                self._commit_neuron(
                    neuron_candidates,
                    neuron_proposal,
                    parent_neuron,
                    proposed_neuron,
                    decision,
                    neuron_refresh,
                )
            elif combined_refresh is not None:
                self._refresh_without_neuron_edit(combined_refresh[0])

            if attention_candidates:
                assert attention_proposal is not None
                active = set(proposed_attention.snapshot.active_node_ids)
                attention_context = Qwen3AttentionSurgeryRefreshContext(
                    runtime=self.runtime,
                    round_index=self.attention_bridge._graph_rounds,
                    accepted_transaction_index=(
                        self.attention_bridge._accepted_count + 1
                    ),
                    evidence_epoch=(self.attention_bridge._evidence_epoch + 1),
                    target_bytes=proposed_attention.snapshot.tensor_bytes,
                    current_bytes=proposed_attention.snapshot.tensor_bytes,
                    active_node_ids=tuple(sorted(active)),
                    graph_transaction=attention_proposal.transaction,
                )
                if combined_refresh is not None:
                    attention_refresh = combined_refresh[1]
                elif refresh_attention_features is None:
                    attention_refresh = self.attention_bridge._invoke_refresh(
                        None,
                        attention_context,
                        active,
                    )
                else:
                    with _isolated_callback_runtime(self.runtime):
                        attention_refresh = self.attention_bridge._invoke_refresh(
                            refresh_attention_features,
                            attention_context,
                            active,
                        )
                    resource_guard.sample()
                if combined_refresh is not None:
                    self.attention_bridge._feature_refreshes += 1
                self._commit_attention(
                    attention_candidates,
                    attention_proposal,
                    parent_attention,
                    proposed_attention,
                    used_attention_projections,
                    decision,
                    attention_refresh,
                )
            elif combined_refresh is not None:
                self._refresh_without_attention_edit(combined_refresh[1])

            if combined_refresh is None:
                self._evidence = None
            self._candidate_validation.clear()
            _sync_parameter_report(self.runtime)
            if (
                neuron_surgery.qwen3_runtime_assignment_snapshot(self.runtime)
                != proposed_neuron.public
                or attention_surgery.qwen3_attention_runtime_snapshot(self.runtime)
                != proposed_attention.snapshot
            ):
                raise Qwen3SurgeryAllocatorStateError(
                    "Commit bookkeeping changed the evaluator-approved tensors."
                )
            self._accepted_transactions += 1
            epoch_before = self._evidence_epoch
            self._evidence_epoch += 1
            self._accounting_cache = self._compute_accounting(
                proposed_neuron.public, proposed_attention.snapshot
            )
            committed = self.accounting()
            self._record_resource_peaks(resource_guard)
            transaction = Qwen3UnifiedSurgeryTransaction(
                transaction_index=self._accepted_transactions,
                evidence_epoch_before=epoch_before,
                evidence_epoch_after=self._evidence_epoch,
                batch_sha256=batch.batch_sha256,
                proposal_sha256s=tuple(
                    item.proposal_sha256 for item in batch.proposals
                ),
                accepted=True,
                decision=decision,
                parent_accounting=parent_accounting,
                proposed_accounting=proposed_accounting,
                committed_accounting=committed,
                rollback_exact=None,
                process_rss_peak_increase_bytes=(
                    resource_guard.process_rss_peak_increase
                ),
                cuda_allocated_peak_increase_bytes=(
                    resource_guard.cuda_allocated_peak_increase
                ),
                cuda_reserved_peak_increase_bytes=(
                    resource_guard.cuda_reserved_peak_increase
                ),
                manifest_transfer_bytes=(
                    self._manifest_transfer_bytes - manifest_transfer_start
                ),
            )
            self._history.append(transaction)
            overflow = len(self._history) - self.limits.history_limit
            if overflow > 0:
                del self._history[:overflow]
            return transaction
        except Exception:
            self._restore_checkpoint(checkpoint)
            self._record_resource_peaks(resource_guard)
            if self._state_sha256() != parent_digest:
                raise Qwen3SurgeryAllocatorStateError(
                    "Mixed transaction exception rollback was not exact."
                )
            raise

    def run(
        self,
        target_deployable_bytes: int,
        *,
        evaluator: Evaluator | None,
        refresh_evidence: RefreshEvidence | None = None,
        refresh_neuron_features: NeuronRefresh | None = None,
        refresh_attention_features: AttentionRefresh | None = None,
        explicit_neuron_pairs: Sequence[Any] = (),
        explicit_attention_pairs: Sequence[Any] = (),
        max_attempts: int | None = None,
        unsafe_allow_ungated_commit: bool = False,
        unsafe_allow_unvalidated_candidates: bool = False,
    ) -> Qwen3UnifiedSurgeryResult:
        target = _nonnegative_int(target_deployable_bytes, "target_deployable_bytes")
        attempt_limit = (
            self.limits.max_attempts
            if max_attempts is None
            else min(
                _positive_int(max_attempts, "max_attempts"),
                self.limits.max_attempts,
            )
        )
        initial = self.accounting()
        if initial.deployable_bytes <= target:
            return Qwen3UnifiedSurgeryResult(
                self.runtime,
                target,
                initial,
                initial,
                (),
                self.diagnostics,
                True,
            )
        transactions: list[Qwen3UnifiedSurgeryTransaction] = []
        excluded: set[str] = set()
        attempts = 0
        while self.accounting().deployable_bytes > target:
            if attempts >= attempt_limit:
                raise Qwen3SurgeryAllocatorTargetError(
                    f"max_attempts={attempt_limit} reached before target."
                )
            current = self.accounting()
            requested = current.deployable_bytes - target
            batch = self.select_batch(
                requested,
                explicit_neuron_pairs=explicit_neuron_pairs,
                explicit_attention_pairs=explicit_attention_pairs,
                excluded_sha256s=frozenset(excluded),
                unsafe_allow_unvalidated_candidates=(
                    unsafe_allow_unvalidated_candidates
                ),
            )
            transaction = self.apply_batch(
                batch,
                evaluator=evaluator,
                refresh_evidence=refresh_evidence,
                refresh_neuron_features=refresh_neuron_features,
                refresh_attention_features=refresh_attention_features,
                unsafe_allow_ungated_commit=unsafe_allow_ungated_commit,
                unsafe_allow_unvalidated_candidates=(
                    unsafe_allow_unvalidated_candidates
                ),
            )
            attempts += 1
            if transaction.accepted:
                transactions.append(transaction)
                excluded.clear()
            else:
                excluded.update(transaction.proposal_sha256s)
                excluded.update(
                    item.candidate.candidate_sha256 for item in batch.proposals
                )
        final = self.accounting()
        return Qwen3UnifiedSurgeryResult(
            runtime=self.runtime,
            target_deployable_bytes=target,
            initial_accounting=initial,
            final_accounting=final,
            transactions=tuple(transactions),
            diagnostics=self.diagnostics,
            target_reached=final.deployable_bytes <= target,
        )


__all__ = [
    "Qwen3CandidateValidationDamage",
    "Qwen3MixedSurgeryBatch",
    "Qwen3SurgeryAllocatorError",
    "Qwen3SurgeryAllocatorLimits",
    "Qwen3SurgeryAllocatorShapeError",
    "Qwen3SurgeryAllocatorStateError",
    "Qwen3SurgeryAllocatorTargetError",
    "Qwen3UnifiedSurgeryAccounting",
    "Qwen3UnifiedSurgeryAllocator",
    "Qwen3UnifiedSurgeryDiagnostics",
    "Qwen3UnifiedSurgeryEvaluationContext",
    "Qwen3UnifiedSurgeryProposal",
    "Qwen3UnifiedSurgeryRefreshContext",
    "Qwen3UnifiedSurgeryResult",
    "Qwen3UnifiedSurgeryTransaction",
]
