"""Deterministic zero-gradient orchestration for on-policy HF refolding.

The orchestrator alternates exact child-prefix corpus construction with a
callback-owned parent/child trace refold.  It does not load models, resolve
remote resources, construct an optimizer, or call backward.  Every candidate
is evaluated on a caller-supplied free-running benchmark and is committed only
when all configured parent-relative gates pass; rejected candidates are
restored exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.hf_on_policy_probe import (
    GreedyProbeSettings,
    OnPolicyProbeCorpus,
    build_on_policy_probe_corpus,
)


REPORT_SCHEMA = "neural-forge-hf-on-policy-refold-v1"
CORPUS_IDENTITY_SCHEMA = "neural-forge-hf-on-policy-corpus-identity-v1"
FREE_RUN_EVALUATION_SCHEMA = "neural-forge-hf-free-run-evaluation-v1"
TRACE_REFOLD_RESULT_SCHEMA = "neural-forge-hf-trace-refold-result-v1"

_SHA256_LENGTH = 64


class OnPolicyRefoldError(RuntimeError):
    """Base error for invalid or failed on-policy refold transactions."""


class OnPolicyRefoldContractError(OnPolicyRefoldError):
    """Raised when a callback violates the zero-gradient evidence contract."""


class OnPolicyRefoldIntegrityError(OnPolicyRefoldError):
    """Raised when an immutable identity or protected model changes."""


class OnPolicyRefoldRollbackError(OnPolicyRefoldIntegrityError):
    """Raised when a failed or rejected transaction cannot be restored exactly."""


def _canonical_value(value: Any, name: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OnPolicyRefoldContractError(f"{name} must contain finite JSON values")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OnPolicyRefoldContractError(f"{name} keys must be strings")
            if key in result:
                raise OnPolicyRefoldContractError(f"{name} contains duplicate keys")
            result[key] = _canonical_value(item, f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise OnPolicyRefoldContractError(
        f"{name} must contain only canonical JSON-compatible values"
    )


def _canonical_json(value: Any) -> bytes:
    normalized = _canonical_value(value)
    try:
        return json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - normalized above
        raise OnPolicyRefoldContractError(
            "value cannot be encoded as canonical JSON"
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OnPolicyRefoldContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(result)


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True, slots=True)
class ParentRelativeMetricGate:
    """One required free-run metric and its promotion thresholds."""

    name: str
    direction: Literal["higher", "lower"] = "higher"
    min_parent_retention: float = 0.0
    min_incumbent_gain: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("metric gate name must be a nonempty string")
        if self.name != self.name.strip():
            raise ValueError("metric gate name cannot contain surrounding whitespace")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("metric gate direction must be 'higher' or 'lower'")
        object.__setattr__(
            self,
            "min_parent_retention",
            _nonnegative_float(self.min_parent_retention, "min_parent_retention"),
        )
        object.__setattr__(
            self,
            "min_incumbent_gain",
            _nonnegative_float(self.min_incumbent_gain, "min_incumbent_gain"),
        )


@dataclass(frozen=True, slots=True)
class OnPolicyRefoldConfig:
    """Deterministic transaction and free-run gate policy."""

    rounds: int = 1
    seed: int = 0
    metric_gates: tuple[ParentRelativeMetricGate, ...] = field(
        default_factory=lambda: (ParentRelativeMetricGate("score"),)
    )
    require_any_metric_improvement: bool = True
    require_child_mutation: bool = True
    stop_on_rejection: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rounds", _positive_int(self.rounds, "rounds"))
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        gates = tuple(self.metric_gates)
        if not gates or any(
            not isinstance(gate, ParentRelativeMetricGate) for gate in gates
        ):
            raise TypeError(
                "metric_gates must be a nonempty sequence of "
                "ParentRelativeMetricGate values"
            )
        names = [gate.name for gate in gates]
        if len(set(names)) != len(names):
            raise ValueError("metric_gates must use unique names")
        object.__setattr__(self, "metric_gates", gates)
        for name in (
            "require_any_metric_improvement",
            "require_child_mutation",
            "stop_on_rejection",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class FreeRunEvaluation:
    """Immutable free-running evaluation bound to a model and benchmark."""

    role: str
    model_sha256: str
    benchmark_sha256: str
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise OnPolicyRefoldContractError(
                "free-run evaluation role must be a nonempty string"
            )
        _validate_sha256(self.model_sha256, "model_sha256")
        _validate_sha256(self.benchmark_sha256, "benchmark_sha256")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise OnPolicyRefoldContractError(
                "free-run evaluation metrics must be a nonempty mapping"
            )
        metrics: dict[str, float] = {}
        for name, value in sorted(self.metrics.items()):
            if not isinstance(name, str) or not name:
                raise OnPolicyRefoldContractError(
                    "free-run metric names must be nonempty strings"
                )
            try:
                metrics[name] = _nonnegative_float(value, f"metric {name!r}")
            except ValueError as exc:
                raise OnPolicyRefoldContractError(str(exc)) from exc
        metadata = _canonical_value(self.metadata, "evaluation metadata")
        if not isinstance(metadata, dict):
            raise OnPolicyRefoldContractError(
                "free-run evaluation metadata must be a mapping"
            )
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "metadata", _deep_freeze(metadata))
        object.__setattr__(self, "evaluation_sha256", _json_sha256(self.core()))

    def core(self) -> dict[str, Any]:
        return {
            "schema": FREE_RUN_EVALUATION_SCHEMA,
            "role": self.role,
            "model_sha256": self.model_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "free_running": True,
            "metrics": dict(self.metrics),
            "metadata": _canonical_value(self.metadata),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.core(), "evaluation_sha256": self.evaluation_sha256}


@dataclass(frozen=True, slots=True)
class CorpusIdentity:
    """Content identity for one immutable child-prefix probe corpus."""

    corpus_sha256: str
    materialized_sha256: str
    probe_report_sha256: str
    generation_request_sha256: str
    generation_cache_sha256: str
    generation_cache_hit: bool
    row_count: int
    tokenizer_batch_sha256: Mapping[str, tuple[str, ...]]
    identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "corpus_sha256",
            "materialized_sha256",
            "probe_report_sha256",
            "generation_request_sha256",
            "generation_cache_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not isinstance(self.generation_cache_hit, bool):
            raise OnPolicyRefoldContractError("generation_cache_hit must be a bool")
        _positive_int(self.row_count, "row_count")
        batches: dict[str, tuple[str, ...]] = {}
        for name, values in sorted(self.tokenizer_batch_sha256.items()):
            if not isinstance(name, str) or not name:
                raise OnPolicyRefoldContractError(
                    "tokenizer names must be nonempty strings"
                )
            values = tuple(values)
            if not values:
                raise OnPolicyRefoldContractError(
                    f"tokenizer {name!r} must have trace batches"
                )
            batches[name] = tuple(
                _validate_sha256(value, f"{name} batch identity") for value in values
            )
        object.__setattr__(self, "tokenizer_batch_sha256", MappingProxyType(batches))
        object.__setattr__(self, "identity_sha256", _json_sha256(self.identity_core()))

    def identity_core(self) -> dict[str, Any]:
        """Return content fields only; cache-hit telemetry is invocation-local."""

        return {
            "schema": CORPUS_IDENTITY_SCHEMA,
            "corpus_sha256": self.corpus_sha256,
            "materialized_sha256": self.materialized_sha256,
            "generation_request_sha256": self.generation_request_sha256,
            "generation_cache_sha256": self.generation_cache_sha256,
            "row_count": self.row_count,
            "tokenizer_batch_sha256": {
                name: list(values)
                for name, values in self.tokenizer_batch_sha256.items()
            },
        }

    def core(self) -> dict[str, Any]:
        return {
            **self.identity_core(),
            "probe_report_sha256": self.probe_report_sha256,
            "generation_cache_hit": self.generation_cache_hit,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.core(), "identity_sha256": self.identity_sha256}


@dataclass(frozen=True, slots=True)
class TraceRefoldResult:
    """Sealed evidence returned by the callback that mutates the child."""

    corpus_identity_sha256: str
    parent_model_sha256: str
    child_model_sha256: str
    parent_trace_cache_sha256: str
    child_trace_cache_sha256: str
    parent_trace_cache_hit: bool = False
    child_trace_cache_hit: bool = False
    optimizer_constructed: bool = False
    optimizer_steps: int = 0
    backward_calls: int = 0
    training_steps: int = 0
    network_calls: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)
    result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "corpus_identity_sha256",
            "parent_model_sha256",
            "child_model_sha256",
            "parent_trace_cache_sha256",
            "child_trace_cache_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "parent_trace_cache_hit",
            "child_trace_cache_hit",
            "optimizer_constructed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise OnPolicyRefoldContractError(f"{name} must be a bool")
        counters = {}
        for name in (
            "optimizer_steps",
            "backward_calls",
            "training_steps",
            "network_calls",
        ):
            try:
                counters[name] = _nonnegative_int(getattr(self, name), name)
            except ValueError as exc:
                raise OnPolicyRefoldContractError(str(exc)) from exc
            object.__setattr__(self, name, counters[name])
        details = _canonical_value(self.details, "trace refold details")
        if not isinstance(details, dict):
            raise OnPolicyRefoldContractError("trace refold details must be a mapping")
        object.__setattr__(self, "details", _deep_freeze(details))
        if self.optimizer_constructed or any(counters.values()):
            raise OnPolicyRefoldContractError(
                "trace refold callbacks must report zero optimizer, backward, "
                "training, and network activity"
            )
        object.__setattr__(self, "result_sha256", _json_sha256(self.core()))

    def core(self) -> dict[str, Any]:
        return {
            "schema": TRACE_REFOLD_RESULT_SCHEMA,
            "corpus_identity_sha256": self.corpus_identity_sha256,
            "parent_model_sha256": self.parent_model_sha256,
            "child_model_sha256": self.child_model_sha256,
            "parent_trace_cache_sha256": self.parent_trace_cache_sha256,
            "child_trace_cache_sha256": self.child_trace_cache_sha256,
            "parent_trace_cache_hit": self.parent_trace_cache_hit,
            "child_trace_cache_hit": self.child_trace_cache_hit,
            "optimizer_constructed": self.optimizer_constructed,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "training_steps": self.training_steps,
            "network_calls": self.network_calls,
            "details": _canonical_value(self.details),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.core(), "result_sha256": self.result_sha256}


@dataclass(frozen=True, slots=True)
class FreeRunEvaluationContext:
    """Read-only invocation context for a free-run evaluation callback."""

    role: str
    model: Any
    model_sha256: str
    benchmark_sha256: str
    round_index: int | None
    seed: int

    def seal(
        self,
        metrics: Mapping[str, float],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> FreeRunEvaluation:
        return FreeRunEvaluation(
            role=self.role,
            model_sha256=self.model_sha256,
            benchmark_sha256=self.benchmark_sha256,
            metrics=metrics,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class OnPolicyRefoldContext:
    """One child-prefix trace/refold transaction exposed to the fold callback."""

    round_index: int
    seed: int
    parent_model: Any
    child_model: Any
    parent_model_sha256: str
    child_model_sha256: str
    corpus: OnPolicyProbeCorpus
    corpus_identity: CorpusIdentity
    parent_trace_batches: tuple[Any, ...]
    child_trace_batches: tuple[Any, ...]
    trace_cache_root: Path | None

    def seal_result(
        self,
        *,
        parent_trace_cache_sha256: str,
        child_trace_cache_sha256: str,
        parent_trace_cache_hit: bool = False,
        child_trace_cache_hit: bool = False,
        details: Mapping[str, Any] | None = None,
        optimizer_constructed: bool = False,
        optimizer_steps: int = 0,
        backward_calls: int = 0,
        training_steps: int = 0,
        network_calls: int = 0,
    ) -> TraceRefoldResult:
        return TraceRefoldResult(
            corpus_identity_sha256=self.corpus_identity.identity_sha256,
            parent_model_sha256=self.parent_model_sha256,
            child_model_sha256=self.child_model_sha256,
            parent_trace_cache_sha256=parent_trace_cache_sha256,
            child_trace_cache_sha256=child_trace_cache_sha256,
            parent_trace_cache_hit=parent_trace_cache_hit,
            child_trace_cache_hit=child_trace_cache_hit,
            optimizer_constructed=optimizer_constructed,
            optimizer_steps=optimizer_steps,
            backward_calls=backward_calls,
            training_steps=training_steps,
            network_calls=network_calls,
            details={} if details is None else details,
        )


@dataclass(frozen=True, slots=True)
class OnPolicyRefoldResult:
    """Final immutable orchestration result and its structured report."""

    initial_child_sha256: str
    final_child_sha256: str
    accepted_rounds: int
    rejected_rounds: int
    report: Mapping[str, Any]

    @property
    def changed(self) -> bool:
        return self.initial_child_sha256 != self.final_child_sha256

    @property
    def accepted(self) -> bool:
        return self.accepted_rounds > 0


@dataclass(slots=True)
class _TorchModelSnapshot:
    state: dict[str, Any]
    parameters: tuple[tuple[str, Any, bool, Any | None, Any | None], ...]
    module_modes: tuple[tuple[Any, bool], ...]


@dataclass(frozen=True, slots=True)
class _ModelRuntime:
    parameter_state: tuple[tuple[str, int, bool, int | None, str | None], ...]
    module_modes: tuple[tuple[int, bool], ...]


class _ModelAccess:
    def __init__(
        self,
        torch: Any,
        fingerprint_fn: Callable[[Any], str | Mapping[str, Any]] | None,
        snapshot_fn: Callable[[Any], Any] | None,
        restore_fn: Callable[[Any, Any], None] | None,
    ) -> None:
        if (snapshot_fn is None) != (restore_fn is None):
            raise TypeError("snapshot_fn and restore_fn must be supplied together")
        self.torch = torch
        self.fingerprint_fn = fingerprint_fn
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn

    def fingerprint(self, model: Any) -> str:
        if self.fingerprint_fn is None:
            return _default_model_fingerprint(self.torch, model)
        value = self.fingerprint_fn(model)
        if isinstance(value, str):
            return _validate_sha256(value, "model fingerprint callback result")
        if isinstance(value, Mapping):
            return _json_sha256(value)
        raise OnPolicyRefoldContractError(
            "model fingerprint callback must return a SHA-256 string or mapping"
        )

    def runtime(self, model: Any) -> _ModelRuntime:
        return _model_runtime(self.torch, model)

    def snapshot(self, model: Any) -> Any:
        if self.snapshot_fn is not None:
            return self.snapshot_fn(model)
        return _default_snapshot(self.torch, model)

    def restore(self, model: Any, snapshot: Any) -> None:
        if self.restore_fn is not None:
            result = self.restore_fn(model, snapshot)
            if result is not None:
                raise OnPolicyRefoldContractError(
                    "restore_fn must restore in place and return None"
                )
            return
        _default_restore(self.torch, model, snapshot)


def _torch(value: Any | None) -> Any:
    if value is not None:
        return value
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "on-policy refolding requires the 'hf' dependencies"
        ) from exc
    return torch


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
        raise OnPolicyRefoldContractError(
            "model state and gradients must contain dense tensors"
        )
    if tensor.device.type == "meta":
        raise OnPolicyRefoldContractError("model tensors must be materialized")
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-hf-on-policy-refold-tensor-v1\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in value.shape]))
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _default_model_fingerprint(torch: Any, model: Any) -> str:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise OnPolicyRefoldContractError(
            "models must expose state_dict() or use fingerprint_fn"
        )
    state = state_dict()
    if not isinstance(state, Mapping):
        raise OnPolicyRefoldContractError("state_dict() must return a mapping")
    tensors = []
    for name in sorted(state):
        if not isinstance(name, str):
            raise OnPolicyRefoldContractError("model state names must be strings")
        tensors.append(
            {
                "name": name,
                "tensor_sha256": _tensor_sha256(torch, state[name]),
            }
        )
    config_value = None
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        config_value = _canonical_value(to_dict(), "model config")
    return _json_sha256(
        {
            "schema": "neural-forge-hf-on-policy-model-fingerprint-v1",
            "model_class": (f"{type(model).__module__}.{type(model).__qualname__}"),
            "state": tensors,
            "config": config_value,
        }
    )


def _named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return ()
    return tuple((str(name), parameter) for name, parameter in named_parameters())


def _module_modes(model: Any) -> tuple[tuple[Any, bool], ...]:
    modules = getattr(model, "modules", None)
    if callable(modules):
        return tuple(
            (module, bool(module.training))
            for module in modules()
            if isinstance(getattr(module, "training", None), bool)
        )
    if isinstance(getattr(model, "training", None), bool):
        return ((model, bool(model.training)),)
    return ()


def _model_runtime(torch: Any, model: Any) -> _ModelRuntime:
    parameters = []
    for name, parameter in _named_parameters(model):
        grad = getattr(parameter, "grad", None)
        parameters.append(
            (
                name,
                id(parameter),
                bool(getattr(parameter, "requires_grad", False)),
                None if grad is None else id(grad),
                None if grad is None else _tensor_sha256(torch, grad),
            )
        )
    return _ModelRuntime(
        parameter_state=tuple(parameters),
        module_modes=tuple((id(module), mode) for module, mode in _module_modes(model)),
    )


def _default_snapshot(torch: Any, model: Any) -> _TorchModelSnapshot:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise OnPolicyRefoldContractError(
            "models must expose state_dict() or use snapshot_fn/restore_fn"
        )
    state = {
        str(name): tensor.detach().to(device="cpu").clone()
        for name, tensor in state_dict().items()
    }
    parameters = []
    for name, parameter in _named_parameters(model):
        grad = getattr(parameter, "grad", None)
        parameters.append(
            (
                name,
                parameter,
                bool(getattr(parameter, "requires_grad", False)),
                grad,
                None if grad is None else grad.detach().to(device="cpu").clone(),
            )
        )
    return _TorchModelSnapshot(
        state=state,
        parameters=tuple(parameters),
        module_modes=_module_modes(model),
    )


def _default_restore(torch: Any, model: Any, snapshot: Any) -> None:
    if not isinstance(snapshot, _TorchModelSnapshot):
        raise OnPolicyRefoldRollbackError("invalid default model snapshot")
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise OnPolicyRefoldRollbackError("model cannot restore state_dict()")
    current_parameters = dict(_named_parameters(model))
    expected_names = {name for name, *_rest in snapshot.parameters}
    if set(current_parameters) != expected_names:
        raise OnPolicyRefoldRollbackError(
            "model parameter inventory changed during transaction"
        )
    for name, parameter, _requires_grad, _grad, _grad_value in snapshot.parameters:
        if current_parameters[name] is not parameter:
            raise OnPolicyRefoldRollbackError(
                f"model parameter identity changed for {name!r}"
            )
    with torch.no_grad():
        load_state_dict(snapshot.state, strict=True)
        for name, parameter, requires_grad, grad, grad_value in snapshot.parameters:
            if bool(parameter.requires_grad) != requires_grad:
                parameter.requires_grad_(requires_grad)
            if grad is None:
                parameter.grad = None
            else:
                grad.copy_(grad_value.to(device=grad.device, dtype=grad.dtype))
                parameter.grad = grad
        for module, training in snapshot.module_modes:
            module.training = training


def _cuda_devices(torch: Any, models: Sequence[Any]) -> list[int]:
    devices = set()
    for model in models:
        for _name, parameter in _named_parameters(model):
            device = getattr(parameter, "device", None)
            if device is not None and device.type == "cuda":
                devices.add(device.index if device.index is not None else 0)
    return sorted(devices)


@contextmanager
def _deterministic_no_grad(
    torch: Any,
    models: Sequence[Any],
    seed: int,
    *,
    inference: bool,
):
    modes = []
    seen_modules: set[int] = set()
    for model in models:
        for module, training in _module_modes(model):
            if id(module) not in seen_modules:
                seen_modules.add(id(module))
                modes.append((module, training))
    algorithms_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    devices = _cuda_devices(torch, models)
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.manual_seed(seed)
            for model in models:
                evaluate = getattr(model, "eval", None)
                if callable(evaluate):
                    evaluate()
            grad_context = (
                getattr(torch, "inference_mode", torch.no_grad)()
                if inference
                else torch.no_grad()
            )
            with grad_context:
                yield
    finally:
        torch.use_deterministic_algorithms(
            algorithms_before, warn_only=warn_only_before
        )
        for module, training in modes:
            module.training = training


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise OnPolicyRefoldContractError("tokenizer must expose get_vocab()")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise OnPolicyRefoldContractError("tokenizer vocabulary must be nonempty")
    normalized: dict[str, int] = {}
    for token, token_id in vocabulary.items():
        if not isinstance(token, str) or isinstance(token_id, bool):
            raise OnPolicyRefoldContractError("tokenizer vocabulary is invalid")
        try:
            normalized[token] = int(operator.index(token_id))
        except TypeError as exc:
            raise OnPolicyRefoldContractError(
                "tokenizer vocabulary IDs must be integers"
            ) from exc
    return _json_sha256(
        {
            "tokenizer_class": (
                f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
            ),
            "length": int(len(tokenizer)),
            "vocabulary": normalized,
            "special_token_ids": {
                name: getattr(tokenizer, name, None)
                for name in ("bos_token_id", "eos_token_id", "pad_token_id")
            },
        }
    )


def _generation_cache_path(
    root: str | Path | None,
    *,
    child_model_sha256: str,
    child_tokenizer: Any,
    seed_prompts: Sequence[str],
    probe_settings: GreedyProbeSettings,
) -> tuple[Path | None, str]:
    request_identity = _json_sha256(
        {
            "schema": "neural-forge-hf-on-policy-generation-cache-route-v1",
            "child_model_sha256": child_model_sha256,
            "child_tokenizer_sha256": _tokenizer_fingerprint(child_tokenizer),
            "seed_prompts": list(seed_prompts),
            "generation": probe_settings.generation_contract(),
        }
    )
    if root is None:
        return None, request_identity
    if isinstance(root, str) and "://" in root:
        raise OnPolicyRefoldContractError(
            "generation_cache_root must be a local filesystem path"
        )
    cache_root = Path(root).expanduser().absolute()
    return cache_root / "generation" / f"{request_identity}.json", request_identity


def _materialized_corpus(corpus: OnPolicyProbeCorpus) -> dict[str, Any]:
    return {
        "rows": [
            {
                "row_id": row.row_id,
                "utf8_sha256": row.utf8_sha256,
                "text": row.text,
                "utf8_hex": row.utf8_bytes.hex(),
                "origins": [asdict(origin) for origin in row.origins],
            }
            for row in corpus.rows
        ],
        "tokenizations": {
            name: [
                {
                    "row_id": row.row_id,
                    "utf8_sha256": row.utf8_sha256,
                    "token_ids": list(row.token_ids),
                    "token_ids_sha256": row.token_ids_sha256,
                }
                for row in rows
            ]
            for name, rows in sorted(corpus.tokenizations.items())
        },
        "trace_batches": {
            name: [
                {
                    "tokenizer_name": batch.tokenizer_name,
                    "start": batch.start,
                    "row_ids": list(batch.row_ids),
                    "input_ids": [list(row) for row in batch.input_ids],
                    "attention_mask": [list(row) for row in batch.attention_mask],
                    "sequence_lengths": list(batch.sequence_lengths),
                    "pad_token_id": batch.pad_token_id,
                    "input_sha256": batch.input_sha256,
                }
                for batch in batches
            ]
            for name, batches in sorted(corpus.trace_batches.items())
        },
    }


def _corpus_identity(corpus: OnPolicyProbeCorpus) -> CorpusIdentity:
    if not isinstance(corpus, OnPolicyProbeCorpus):
        raise OnPolicyRefoldContractError(
            "probe builder must return OnPolicyProbeCorpus"
        )
    report = _canonical_value(corpus.report, "probe report")
    if not isinstance(report, dict):  # pragma: no cover - dataclass contract
        raise OnPolicyRefoldIntegrityError("probe report must be a mapping")
    supplied_report_sha = report.pop("report_sha256", None)
    _validate_sha256(supplied_report_sha, "probe report_sha256")
    if _json_sha256(report) != supplied_report_sha:
        raise OnPolicyRefoldIntegrityError("probe report fingerprint is invalid")
    generation_cache = report.get("generation_cache")
    if not isinstance(generation_cache, Mapping):
        raise OnPolicyRefoldIntegrityError(
            "probe report generation cache evidence is missing"
        )
    tokenizer_batches = {
        name: tuple(batch.input_sha256 for batch in batches)
        for name, batches in corpus.trace_batches.items()
    }
    return CorpusIdentity(
        corpus_sha256=_validate_sha256(
            report.get("corpus_sha256"), "probe corpus_sha256"
        ),
        materialized_sha256=_json_sha256(_materialized_corpus(corpus)),
        probe_report_sha256=supplied_report_sha,
        generation_request_sha256=_validate_sha256(
            report.get("request_sha256"), "probe request_sha256"
        ),
        generation_cache_sha256=_validate_sha256(
            generation_cache.get("cache_sha256"),
            "probe generation cache_sha256",
        ),
        generation_cache_hit=bool(generation_cache.get("hit")),
        row_count=len(corpus.rows),
        tokenizer_batch_sha256=tokenizer_batches,
    )


def _assert_corpus_unchanged(
    corpus: OnPolicyProbeCorpus, expected: CorpusIdentity
) -> None:
    actual = _corpus_identity(corpus)
    if actual != expected:
        raise OnPolicyRefoldIntegrityError(
            "trace refold callback mutated the immutable probe corpus"
        )


def _normalize_evaluation(
    value: FreeRunEvaluation | Mapping[str, Any],
    context: FreeRunEvaluationContext,
) -> FreeRunEvaluation:
    supplied_sha = None
    if isinstance(value, FreeRunEvaluation):
        evaluation = value
    elif isinstance(value, Mapping):
        if "metrics" in value:
            if value.get("schema", FREE_RUN_EVALUATION_SCHEMA) != (
                FREE_RUN_EVALUATION_SCHEMA
            ):
                raise OnPolicyRefoldIntegrityError(
                    "evaluation callback schema identity is invalid"
                )
            metrics = value.get("metrics")
            if not isinstance(metrics, Mapping):
                raise OnPolicyRefoldContractError(
                    "evaluation callback metrics must be a mapping"
                )
            if value.get("free_running", True) is not True:
                raise OnPolicyRefoldContractError(
                    "evaluation callback must report free-running metrics"
                )
            for name, expected in (
                ("role", context.role),
                ("model_sha256", context.model_sha256),
                ("benchmark_sha256", context.benchmark_sha256),
            ):
                if name in value and value[name] != expected:
                    raise OnPolicyRefoldIntegrityError(
                        f"evaluation callback returned a mismatched {name}"
                    )
            supplied_sha = value.get("evaluation_sha256")
            evaluation = context.seal(
                metrics,
                metadata=value.get("metadata", {}),
            )
        else:
            evaluation = context.seal(value)
    else:
        raise OnPolicyRefoldContractError(
            "evaluation callback must return FreeRunEvaluation or a metric mapping"
        )
    if evaluation.role != context.role:
        raise OnPolicyRefoldIntegrityError("evaluation role identity is invalid")
    if evaluation.model_sha256 != context.model_sha256:
        raise OnPolicyRefoldIntegrityError("evaluation model identity is invalid")
    if evaluation.benchmark_sha256 != context.benchmark_sha256:
        raise OnPolicyRefoldIntegrityError("evaluation benchmark identity is invalid")
    if supplied_sha is not None and supplied_sha != evaluation.evaluation_sha256:
        raise OnPolicyRefoldIntegrityError("evaluation callback fingerprint is invalid")
    return evaluation


def _normalize_refold_result(
    value: TraceRefoldResult | Mapping[str, Any],
    context: OnPolicyRefoldContext,
) -> TraceRefoldResult:
    supplied_sha = None
    if isinstance(value, TraceRefoldResult):
        result = value
    elif isinstance(value, Mapping):
        if value.get("schema", TRACE_REFOLD_RESULT_SCHEMA) != (
            TRACE_REFOLD_RESULT_SCHEMA
        ):
            raise OnPolicyRefoldIntegrityError(
                "trace refold callback schema identity is invalid"
            )
        supplied_sha = value.get("result_sha256")
        result = TraceRefoldResult(
            corpus_identity_sha256=value.get(
                "corpus_identity_sha256", context.corpus_identity.identity_sha256
            ),
            parent_model_sha256=value.get(
                "parent_model_sha256", context.parent_model_sha256
            ),
            child_model_sha256=value.get(
                "child_model_sha256", context.child_model_sha256
            ),
            parent_trace_cache_sha256=value.get("parent_trace_cache_sha256"),
            child_trace_cache_sha256=value.get("child_trace_cache_sha256"),
            parent_trace_cache_hit=value.get("parent_trace_cache_hit", False),
            child_trace_cache_hit=value.get("child_trace_cache_hit", False),
            optimizer_constructed=value.get("optimizer_constructed", False),
            optimizer_steps=value.get("optimizer_steps", 0),
            backward_calls=value.get("backward_calls", 0),
            training_steps=value.get("training_steps", 0),
            network_calls=value.get("network_calls", 0),
            details=value.get("details", {}),
        )
    else:
        raise OnPolicyRefoldContractError(
            "trace refold callback must return TraceRefoldResult or a mapping"
        )
    expected = {
        "corpus_identity_sha256": context.corpus_identity.identity_sha256,
        "parent_model_sha256": context.parent_model_sha256,
        "child_model_sha256": context.child_model_sha256,
    }
    for name, expected_value in expected.items():
        if getattr(result, name) != expected_value:
            raise OnPolicyRefoldIntegrityError(
                f"trace refold callback returned a mismatched {name}"
            )
    if supplied_sha is not None and supplied_sha != result.result_sha256:
        raise OnPolicyRefoldIntegrityError(
            "trace refold callback result fingerprint is invalid"
        )
    return result


def _parent_retention(parent: float, candidate: float, direction: str) -> float:
    if direction == "higher":
        if parent == 0.0:
            return 1.0
        return candidate / parent
    if parent == 0.0:
        return 1.0 if candidate == 0.0 else 0.0
    if candidate == 0.0:
        return math.inf
    return parent / candidate


def _gate_candidate(
    parent: FreeRunEvaluation,
    incumbent: FreeRunEvaluation,
    candidate: FreeRunEvaluation,
    config: OnPolicyRefoldConfig,
) -> dict[str, Any]:
    gates = []
    improved = []
    for policy in config.metric_gates:
        missing = [
            role
            for role, evaluation in (
                ("parent", parent),
                ("incumbent", incumbent),
                ("candidate", candidate),
            )
            if policy.name not in evaluation.metrics
        ]
        if missing:
            raise OnPolicyRefoldContractError(
                f"metric {policy.name!r} is missing from {', '.join(missing)} evaluation"
            )
        parent_value = parent.metrics[policy.name]
        incumbent_value = incumbent.metrics[policy.name]
        candidate_value = candidate.metrics[policy.name]
        gain = (
            candidate_value - incumbent_value
            if policy.direction == "higher"
            else incumbent_value - candidate_value
        )
        retention = _parent_retention(parent_value, candidate_value, policy.direction)
        retention_passed = retention >= policy.min_parent_retention
        gain_passed = gain >= policy.min_incumbent_gain
        strictly_improved = gain > 0.0
        improved.append(strictly_improved)
        gates.append(
            {
                "name": policy.name,
                "direction": policy.direction,
                "parent": parent_value,
                "incumbent": incumbent_value,
                "candidate": candidate_value,
                "parent_retention": retention,
                "min_parent_retention": policy.min_parent_retention,
                "incumbent_gain": gain,
                "min_incumbent_gain": policy.min_incumbent_gain,
                "retention_passed": retention_passed,
                "gain_passed": gain_passed,
                "strictly_improved": strictly_improved,
                "passed": retention_passed and gain_passed,
            }
        )
    any_improved = any(improved)
    accepted = all(gate["passed"] for gate in gates) and (
        any_improved or not config.require_any_metric_improvement
    )
    return {
        "policy": "parent-relative-free-run-and-incumbent-non-regression",
        "accepted": accepted,
        "all_metric_gates_passed": all(gate["passed"] for gate in gates),
        "any_metric_strictly_improved": any_improved,
        "require_any_metric_improvement": config.require_any_metric_improvement,
        "metric_gates": gates,
        "parent_evaluation_sha256": parent.evaluation_sha256,
        "incumbent_evaluation_sha256": incumbent.evaluation_sha256,
        "candidate_evaluation_sha256": candidate.evaluation_sha256,
    }


def _restore_exact(
    access: _ModelAccess,
    models: Sequence[tuple[str, Any, Any, str, _ModelRuntime]],
) -> None:
    problems = []
    for name, model, snapshot, expected_sha, expected_runtime in reversed(models):
        try:
            access.restore(model, snapshot)
            actual_sha = access.fingerprint(model)
            actual_runtime = access.runtime(model)
            if actual_sha != expected_sha:
                problems.append(f"{name} state fingerprint differs")
            if actual_runtime != expected_runtime:
                problems.append(f"{name} runtime or gradients differ")
        except BaseException as exc:  # rollback must aggregate both models
            problems.append(f"{name} restore failed: {exc}")
    if problems:
        raise OnPolicyRefoldRollbackError(
            "transaction rollback was not exact: " + "; ".join(problems)
        )


def _guarded_evaluation(
    *,
    torch: Any,
    access: _ModelAccess,
    callback: Callable[[FreeRunEvaluationContext], Any],
    context: FreeRunEvaluationContext,
    parent_model: Any,
    child_model: Any,
) -> FreeRunEvaluation:
    protected = []
    for name, model in (("parent", parent_model), ("child", child_model)):
        protected.append(
            (
                name,
                model,
                access.snapshot(model),
                access.fingerprint(model),
                access.runtime(model),
            )
        )
    try:
        with _deterministic_no_grad(
            torch,
            (parent_model, child_model),
            context.seed,
            inference=True,
        ):
            value = callback(context)
        evaluation = _normalize_evaluation(value, context)
        for name, model, _snapshot, expected_sha, expected_runtime in protected:
            if access.fingerprint(model) != expected_sha:
                raise OnPolicyRefoldIntegrityError(
                    f"free-run evaluation mutated the protected {name} model"
                )
            if access.runtime(model) != expected_runtime:
                raise OnPolicyRefoldIntegrityError(
                    f"free-run evaluation changed {name} parameters, gradients, or modes"
                )
        return evaluation
    except BaseException as exc:
        try:
            _restore_exact(access, protected)
        except OnPolicyRefoldRollbackError as rollback_exc:
            raise rollback_exc from exc
        raise


def _config_report(config: OnPolicyRefoldConfig) -> dict[str, Any]:
    return {
        "rounds": config.rounds,
        "seed": config.seed,
        "metric_gates": [asdict(gate) for gate in config.metric_gates],
        "require_any_metric_improvement": config.require_any_metric_improvement,
        "require_child_mutation": config.require_child_mutation,
        "stop_on_rejection": config.stop_on_rejection,
    }


def run_on_policy_refold(
    *,
    parent_model: Any,
    child_model: Any,
    parent_tokenizer: Any,
    child_tokenizer: Any,
    seed_prompts: Sequence[str],
    benchmark_sha256: str,
    trace_refold_fn: Callable[[OnPolicyRefoldContext], Any],
    free_run_eval_fn: Callable[[FreeRunEvaluationContext], Any],
    cached_parent_evaluation: FreeRunEvaluation | Mapping[str, Any] | None = None,
    config: OnPolicyRefoldConfig | None = None,
    probe_settings: GreedyProbeSettings | None = None,
    generation_cache_root: str | Path | None = None,
    trace_cache_root: str | Path | None = None,
    parent_tokenizer_name: str = "parent",
    fingerprint_fn: Callable[[Any], str | Mapping[str, Any]] | None = None,
    snapshot_fn: Callable[[Any], Any] | None = None,
    restore_fn: Callable[[Any, Any], None] | None = None,
    torch_module: Any | None = None,
) -> OnPolicyRefoldResult:
    """Run bounded on-policy refold transactions on already-loaded models.

    ``trace_refold_fn`` owns trace-cache construction/reuse and the closed-form
    child mutation.  It receives exact parent and child trace batches and must
    return sealed cache identities through :class:`TraceRefoldResult`.
    ``free_run_eval_fn`` receives a parent, incumbent, or candidate evaluation
    context and must execute the same local, free-running benchmark each time.
    """

    if parent_model is child_model:
        raise ValueError("parent_model and child_model must be distinct objects")
    if not callable(trace_refold_fn):
        raise TypeError("trace_refold_fn must be callable")
    if not callable(free_run_eval_fn):
        raise TypeError("free_run_eval_fn must be callable")
    benchmark_sha256 = _validate_sha256(benchmark_sha256, "benchmark_sha256")
    if not isinstance(parent_tokenizer_name, str) or not parent_tokenizer_name.strip():
        raise ValueError("parent_tokenizer_name must be a nonempty string")
    parent_tokenizer_name = parent_tokenizer_name.strip()
    if parent_tokenizer_name == "child":
        raise ValueError("parent_tokenizer_name cannot use the reserved name 'child'")
    selected_config = config or OnPolicyRefoldConfig()
    if not isinstance(selected_config, OnPolicyRefoldConfig):
        raise TypeError("config must be OnPolicyRefoldConfig or None")
    selected_probe = probe_settings or GreedyProbeSettings()
    if not isinstance(selected_probe, GreedyProbeSettings):
        raise TypeError("probe_settings must be GreedyProbeSettings or None")
    prompts = tuple(seed_prompts)
    if isinstance(seed_prompts, (str, bytes)) or not prompts:
        raise ValueError("seed_prompts must be a nonempty sequence of strings")
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("seed_prompts must contain nonempty strings")
    if trace_cache_root is not None:
        if isinstance(trace_cache_root, str) and "://" in trace_cache_root:
            raise OnPolicyRefoldContractError(
                "trace_cache_root must be a local filesystem path"
            )
        resolved_trace_root = Path(trace_cache_root).expanduser().absolute()
    else:
        resolved_trace_root = None

    torch = _torch(torch_module)
    access = _ModelAccess(
        torch,
        fingerprint_fn=fingerprint_fn,
        snapshot_fn=snapshot_fn,
        restore_fn=restore_fn,
    )
    initial_parent_sha = access.fingerprint(parent_model)
    initial_child_sha = access.fingerprint(child_model)
    initial_parent_runtime = access.runtime(parent_model)
    initial_child_runtime = access.runtime(child_model)

    parent_context = FreeRunEvaluationContext(
        role="parent",
        model=parent_model,
        model_sha256=initial_parent_sha,
        benchmark_sha256=benchmark_sha256,
        round_index=None,
        seed=selected_config.seed,
    )
    if cached_parent_evaluation is None:
        parent_evaluation = _guarded_evaluation(
            torch=torch,
            access=access,
            callback=free_run_eval_fn,
            context=parent_context,
            parent_model=parent_model,
            child_model=child_model,
        )
        parent_evaluation_source = "callback"
    else:
        parent_evaluation = _normalize_evaluation(
            cached_parent_evaluation,
            parent_context,
        )
        parent_evaluation_source = "verified-cache"
    initial_child_evaluation = _guarded_evaluation(
        torch=torch,
        access=access,
        callback=free_run_eval_fn,
        context=FreeRunEvaluationContext(
            role="incumbent",
            model=child_model,
            model_sha256=initial_child_sha,
            benchmark_sha256=benchmark_sha256,
            round_index=None,
            seed=selected_config.seed,
        ),
        parent_model=parent_model,
        child_model=child_model,
    )
    incumbent_evaluation = initial_child_evaluation

    round_reports = []
    accepted_rounds = 0
    rejected_rounds = 0
    for round_index in range(selected_config.rounds):
        parent_before = access.fingerprint(parent_model)
        child_before = access.fingerprint(child_model)
        parent_runtime_before = access.runtime(parent_model)
        child_runtime_before = access.runtime(child_model)
        protected = [
            (
                "parent",
                parent_model,
                access.snapshot(parent_model),
                parent_before,
                parent_runtime_before,
            ),
            (
                "child",
                child_model,
                access.snapshot(child_model),
                child_before,
                child_runtime_before,
            ),
        ]
        try:
            cache_path, cache_route_sha = _generation_cache_path(
                generation_cache_root,
                child_model_sha256=child_before,
                child_tokenizer=child_tokenizer,
                seed_prompts=prompts,
                probe_settings=selected_probe,
            )
            corpus = build_on_policy_probe_corpus(
                child_model,
                child_tokenizer,
                {parent_tokenizer_name: parent_tokenizer},
                prompts,
                settings=selected_probe,
                generation_cache_path=cache_path,
                torch_module=torch,
            )
            corpus_identity = _corpus_identity(corpus)
            if access.fingerprint(parent_model) != parent_before:
                raise OnPolicyRefoldIntegrityError(
                    "child-prefix generation mutated the parent model"
                )
            if access.fingerprint(child_model) != child_before:
                raise OnPolicyRefoldIntegrityError(
                    "child-prefix generation mutated the child model"
                )
            if access.runtime(parent_model) != parent_runtime_before:
                raise OnPolicyRefoldIntegrityError(
                    "child-prefix generation changed parent runtime state"
                )
            if access.runtime(child_model) != child_runtime_before:
                raise OnPolicyRefoldIntegrityError(
                    "child-prefix generation changed child gradients or modes"
                )

            context = OnPolicyRefoldContext(
                round_index=round_index,
                seed=selected_config.seed + round_index,
                parent_model=parent_model,
                child_model=child_model,
                parent_model_sha256=parent_before,
                child_model_sha256=child_before,
                corpus=corpus,
                corpus_identity=corpus_identity,
                parent_trace_batches=corpus.batches_for(parent_tokenizer_name),
                child_trace_batches=corpus.batches_for("child"),
                trace_cache_root=resolved_trace_root,
            )
            with _deterministic_no_grad(
                torch,
                (parent_model, child_model),
                context.seed,
                inference=False,
            ):
                callback_value = trace_refold_fn(context)
            refold_result = _normalize_refold_result(callback_value, context)
            _assert_corpus_unchanged(corpus, corpus_identity)

            parent_candidate_sha = access.fingerprint(parent_model)
            child_candidate_sha = access.fingerprint(child_model)
            parent_runtime_candidate = access.runtime(parent_model)
            child_runtime_candidate = access.runtime(child_model)
            if parent_candidate_sha != parent_before:
                raise OnPolicyRefoldIntegrityError(
                    "trace refold callback mutated the protected parent model"
                )
            if parent_runtime_candidate != parent_runtime_before:
                raise OnPolicyRefoldIntegrityError(
                    "trace refold callback changed parent parameters, gradients, or modes"
                )
            if child_runtime_candidate != child_runtime_before:
                raise OnPolicyRefoldIntegrityError(
                    "trace refold callback changed child parameter identities, gradients, or modes"
                )
            child_changed = child_candidate_sha != child_before
            if selected_config.require_child_mutation and not child_changed:
                raise OnPolicyRefoldContractError(
                    "trace refold callback did not mutate the child model"
                )

            candidate_evaluation = _guarded_evaluation(
                torch=torch,
                access=access,
                callback=free_run_eval_fn,
                context=FreeRunEvaluationContext(
                    role="candidate",
                    model=child_model,
                    model_sha256=child_candidate_sha,
                    benchmark_sha256=benchmark_sha256,
                    round_index=round_index,
                    seed=selected_config.seed + round_index,
                ),
                parent_model=parent_model,
                child_model=child_model,
            )
            _assert_corpus_unchanged(corpus, corpus_identity)
            gate = _gate_candidate(
                parent_evaluation,
                incumbent_evaluation,
                candidate_evaluation,
                selected_config,
            )
            accepted = bool(gate["accepted"])
            rollback = {"performed": False, "exact": None}
            if accepted:
                accepted_rounds += 1
                incumbent_evaluation = candidate_evaluation
                final_round_child_sha = child_candidate_sha
            else:
                _restore_exact(access, protected)
                rejected_rounds += 1
                rollback = {"performed": True, "exact": True}
                final_round_child_sha = child_before

            round_reports.append(
                {
                    "round": round_index,
                    "phase_order": [
                        "child-prefix-generation",
                        "parent-child-trace-refold",
                        "parent-relative-free-run-gate",
                        "commit" if accepted else "rollback",
                    ],
                    "generation_cache_route_sha256": cache_route_sha,
                    "generation_cache_path": (
                        None if cache_path is None else str(cache_path)
                    ),
                    "corpus_identity": corpus_identity.as_dict(),
                    "trace_refold": refold_result.as_dict(),
                    "candidate_evaluation": candidate_evaluation.as_dict(),
                    "gate": gate,
                    "accepted": accepted,
                    "mutation_audit": {
                        "parent_before_sha256": parent_before,
                        "parent_after_sha256": parent_candidate_sha,
                        "parent_unchanged": True,
                        "child_before_sha256": child_before,
                        "child_candidate_sha256": child_candidate_sha,
                        "child_final_sha256": final_round_child_sha,
                        "child_changed_by_refold": child_changed,
                        "parameter_identities_unchanged": True,
                        "gradient_references_and_values_unchanged": True,
                        "corpus_unchanged": True,
                    },
                    "rollback": rollback,
                }
            )
            if not accepted and selected_config.stop_on_rejection:
                break
        except BaseException as exc:
            try:
                _restore_exact(access, protected)
            except OnPolicyRefoldRollbackError as rollback_exc:
                raise rollback_exc from exc
            raise

    final_parent_sha = access.fingerprint(parent_model)
    final_child_sha = access.fingerprint(child_model)
    if (
        final_parent_sha != initial_parent_sha
        or access.runtime(parent_model) != initial_parent_runtime
    ):
        raise OnPolicyRefoldIntegrityError(
            "parent model changed across on-policy refold orchestration"
        )
    if access.runtime(child_model) != initial_child_runtime:
        raise OnPolicyRefoldIntegrityError(
            "child parameter identities, gradients, or modes changed outside values"
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "benchmark_sha256": benchmark_sha256,
        "settings": _config_report(selected_config),
        "probe_settings": asdict(selected_probe),
        "baseline": {
            "parent": parent_evaluation.as_dict(),
            "parent_source": parent_evaluation_source,
            "initial_child": initial_child_evaluation.as_dict(),
        },
        "rounds": round_reports,
        "accepted_rounds": accepted_rounds,
        "rejected_rounds": rejected_rounds,
        "completed_rounds": len(round_reports),
        "accepted": accepted_rounds > 0,
        "initial_parent_sha256": initial_parent_sha,
        "final_parent_sha256": final_parent_sha,
        "initial_child_sha256": initial_child_sha,
        "final_child_sha256": final_child_sha,
        "child_changed": initial_child_sha != final_child_sha,
        "execution": {
            "deterministic": True,
            "zero_gradient": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_steps": 0,
            "network_resources_called_by_orchestrator": False,
            "loaded_models_supplied_by_caller": True,
            "parent_evaluation_callback_calls": (
                0 if cached_parent_evaluation is not None else 1
            ),
            "parent_mutations": 0,
            "rejected_child_mutations_restored": rejected_rounds,
        },
    }
    report["report_sha256"] = _json_sha256(report)
    return OnPolicyRefoldResult(
        initial_child_sha256=initial_child_sha,
        final_child_sha256=final_child_sha,
        accepted_rounds=accepted_rounds,
        rejected_rounds=rejected_rounds,
        report=_deep_freeze(report),
    )


# Descriptive aliases for callers that prefer the module or orchestration name.
run_hf_on_policy_refold = run_on_policy_refold
orchestrate_on_policy_refold = run_on_policy_refold
OnPolicyRefoldSettings = OnPolicyRefoldConfig
RefoldCallbackResult = TraceRefoldResult


__all__ = [
    "CORPUS_IDENTITY_SCHEMA",
    "FREE_RUN_EVALUATION_SCHEMA",
    "REPORT_SCHEMA",
    "TRACE_REFOLD_RESULT_SCHEMA",
    "CorpusIdentity",
    "FreeRunEvaluation",
    "FreeRunEvaluationContext",
    "OnPolicyRefoldConfig",
    "OnPolicyRefoldContext",
    "OnPolicyRefoldContractError",
    "OnPolicyRefoldError",
    "OnPolicyRefoldIntegrityError",
    "OnPolicyRefoldResult",
    "OnPolicyRefoldRollbackError",
    "OnPolicyRefoldSettings",
    "ParentRelativeMetricGate",
    "RefoldCallbackResult",
    "TraceRefoldResult",
    "orchestrate_on_policy_refold",
    "run_hf_on_policy_refold",
    "run_on_policy_refold",
]
