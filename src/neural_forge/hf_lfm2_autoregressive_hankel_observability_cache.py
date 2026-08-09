"""Immutable aggregate cache for autoregressive Hankel observability.

This cache sits above the independently reusable per-batch Hankel phase
cache.  Its key binds the exact *ordered* shard identities and payloads to the
future-causal solver implementation, boundary topology, numerical ABI, and
regularization policy.  A cold entry runs the closed-form observability solver
once.  A warm entry authenticates every current source shard and reconstructs
the solver result directly from CPU float64 SafeTensors; it never re-aggregates
moments or repeats an eigensolve.

The public surface deliberately accepts no model, tokenizer, logits, device,
or optimizer.  This is fold-only compiler infrastructure, not training.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from neural_forge import hf_lfm2_autoregressive_hankel_observability as _solver
from neural_forge import hf_lfm2_autoregressive_hankel_phase_cache as _phase
from neural_forge.dynamic_balanced_circuit_basis import (
    PrecomputedDownstreamObservability,
    _tensor_sha256,
)
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256


LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-cache-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_IDENTITY_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-cache-identity-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-cache-receipt-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_PROVENANCE_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-cache-provenance-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_SAFETENSORS_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-cache-safetensors-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_NUMERIC_ABI_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-numeric-abi-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PLAN_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-plan-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-progress-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROGRESS_SAFETENSORS_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-progress-safetensors-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CURRENT_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-observability-current-v1"
)

_RECEIPT_FILENAME = "receipt.json"
_PROVENANCE_FILENAME = "provenance.json"
_REPORT_FILENAME = "report.json"
_TENSOR_FILENAME = "observability.safetensors"
_PROGRESS_RECEIPT_FILENAME = "progress.json"
_PROGRESS_TENSOR_FILENAME = "pooled-statistics.safetensors"
_CURRENT_FILENAME = "current.json"
_EXACT_FILE_INVENTORY = frozenset(
    {
        _RECEIPT_FILENAME,
        _PROVENANCE_FILENAME,
        _REPORT_FILENAME,
        _TENSOR_FILENAME,
    }
)
_GLOBAL_TENSOR_NAME = "global_gram"
_BOUNDARY_TENSOR_PREFIX = "boundary_gram."
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ZERO_EXECUTION_AUDIT = {
    "model_objects_constructed": 0,
    "tokenizer_objects_constructed": 0,
    "parent_forward_calls": 0,
    "child_forward_calls": 0,
    "generate_calls": 0,
    "trace_regeneration_calls": 0,
    "statistics_aggregation_calls_on_cache_hit": 0,
    "eigensolves_on_cache_hit": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "network_accesses": 0,
    "gradients_used": False,
}


class Lfm2AutoregressiveHankelObservabilityCacheError(RuntimeError):
    """Base error for the immutable aggregate observability cache."""


class Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
    Lfm2AutoregressiveHankelObservabilityCacheError
):
    """A source shard or persisted aggregate failed closed authentication."""


@dataclass(frozen=True, slots=True)
class Lfm2AutoregressiveHankelObservabilityCacheIdentity:
    """Canonical identity of one aggregate future-causal solve."""

    body: Mapping[str, Any]
    aggregate_key_sha256: str
    boundary_names: tuple[str, ...]
    state_ridge_fraction: float
    future_ridge_fraction: float
    maximum_condition_number: float


@dataclass(frozen=True, slots=True)
class Lfm2AutoregressiveHankelObservabilityPlan:
    """A tensor-free ordered plan used to address resumable pooled work."""

    body: Mapping[str, Any]
    plan_key_sha256: str
    planned_phase_keys: tuple[str, ...]
    boundary_names: tuple[str, ...]
    state_ridge_fraction: float
    future_ridge_fraction: float
    maximum_condition_number: float


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2AutoregressiveHankelObservability:
    """A verified aggregate result loaded from or published to the cache."""

    result: _solver.Lfm2AutoregressiveHankelObservabilityResult
    receipt: Mapping[str, Any]
    provenance: Mapping[str, Any]
    path: Path
    aggregate_key_sha256: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class _AuthenticatedSourceShard:
    statistics: Any
    description: Mapping[str, Any]
    boundary_topology: Mapping[str, Any]
    boundary_names: tuple[str, ...]
    future_lags: tuple[int, ...]


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _json_copy(value: Any, name: str) -> Any:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, list | tuple):
            return [thaw(child) for child in item]
        return item

    try:
        return json.loads(canonical_json_bytes(thaw(value)).decode("ascii"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} is not canonical JSON data."
        ) from exc


def _sealed_body(value: Any, seal_name: str, name: str) -> dict[str, Any]:
    document = _json_copy(value, name)
    if not isinstance(document, dict):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} must be a JSON object."
        )
    seal = document.pop(seal_name, None)
    if _digest(seal, f"{name}.{seal_name}") != canonical_sha256(document):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} self-seal changed."
        )
    return document


def _canonical_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "boundary_names must be an ordered sequence of canonical names."
        )
    names: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                "boundary_names must contain exact nonempty canonical text."
            )
        names.append(value)
    result = tuple(names)
    if not result or len(set(result)) != len(result):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "boundary_names must uniquely name every boundary in order."
        )
    return result


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} must be a finite positive number."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} must be a finite positive number."
        )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"Required regular file is absent or unsafe: {path}."
        )
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _read_canonical_json(path: Path) -> dict[str, Any]:
    before = _file_identity(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"Cache document is unreadable: {path}."
        ) from exc
    if _file_identity(path) != before:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"Cache document changed while it was read: {path}."
        )
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"Cache document is not exact canonical JSON: {path}."
        )
    return value


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _source_sha256(module: Any, name: str) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} has no auditable source file."
        )
    path = Path(source).resolve()
    if not path.is_file() or path.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} source file is absent or unsafe."
        )
    return _file_sha256(path)


def _numeric_abi() -> dict[str, Any]:
    try:
        import safetensors
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Persistent LFM2 Hankel observability requires the 'hf' extra."
        ) from exc
    torch_build = torch.__config__.show()
    return {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_NUMERIC_ABI_SCHEMA,
        "arithmetic_device": "cpu",
        "solver_dtype": "float64",
        "tensor_layout": "strided-contiguous",
        "eigensolver": "torch.linalg.eigh",
        "symmetric_projection": "exact-average-with-transpose",
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "torch_build_sha256": hashlib.sha256(torch_build.encode("utf-8")).hexdigest(),
        "safetensors_version": str(safetensors.__version__),
    }


def _moment_numeric_abi() -> dict[str, Any]:
    try:
        import safetensors
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Persistent LFM2 Hankel observability requires the 'hf' extra."
        ) from exc
    return {
        "schema": "neural-forge-lfm2-ahb-pooled-moment-numeric-abi-v1",
        "arithmetic_device": "cpu",
        "accumulation_dtype": "float64",
        "count_dtype": "int64",
        "pooling_algorithm": "ordered-in-place-raw-moment-sum-v1",
        "tensor_layout": "strided-contiguous",
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "safetensors_version": str(safetensors.__version__),
    }


def _tensor_contract(value: Any, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.ndim != 2
        or int(value.shape[0]) != int(value.shape[1])
        or value.numel() == 0
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        or not torch.allclose(value, value.T, rtol=0.0, atol=0.0)
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} violated the exact CPU float64 symmetric Gram contract."
        )
    if float(torch.trace(value).item()) <= 0.0:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} has no positive trace energy."
        )
    return value


def _authenticate_source_shard(
    shard: Any,
    *,
    ordinal: int,
) -> _AuthenticatedSourceShard:
    if not isinstance(shard, _phase.AuthenticatedLfm2AutoregressiveHankelShard):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] is not an authenticated phase-cache shard."
        )
    phase_key_sha256 = _digest(
        shard.phase_key_sha256,
        f"shards[{ordinal}].phase_key_sha256",
    )
    receipt = _json_copy(shard.receipt, f"shards[{ordinal}] receipt")
    receipt_body = _sealed_body(
        receipt,
        "receipt_sha256",
        f"shards[{ordinal}] receipt",
    )
    phase_key = receipt_body.get("phase_key")
    if not isinstance(phase_key, dict):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] has no canonical phase key."
        )
    operation_audit = receipt_body.get("operation_audit")
    if (
        receipt_body.get("schema")
        != _phase.LFM2_AUTOREGRESSIVE_HANKEL_PHASE_RECEIPT_SCHEMA
        or receipt_body.get("cache_schema")
        != _phase.LFM2_AUTOREGRESSIVE_HANKEL_PHASE_CACHE_SCHEMA
        or receipt_body.get("phase_key_sha256") != phase_key_sha256
        or canonical_sha256(phase_key) != phase_key_sha256
        or receipt_body.get("immutable_by_contract") is not True
        or not isinstance(operation_audit, Mapping)
        or any(item not in (0, False) for item in operation_audit.values())
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] phase identity or zero-execution audit changed."
        )

    boundary_names = _canonical_names(receipt_body.get("boundary_names"))
    topology = phase_key.get("boundary_topology")
    if not isinstance(topology, dict):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] has no boundary topology."
        )
    topology_sha256 = _digest(
        phase_key.get("boundary_topology_sha256"),
        f"shards[{ordinal}] boundary_topology_sha256",
    )
    if (
        canonical_sha256(topology) != topology_sha256
        or topology.get("boundary_names") != list(boundary_names)
        or topology.get("boundary_count") != len(boundary_names)
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] boundary topology changed."
        )
    future_lags_raw = phase_key.get("future_lags")
    if not isinstance(future_lags_raw, list):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] has no exact future-lag schedule."
        )
    try:
        future_lags = tuple(int(value) for value in future_lags_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] future-lag schedule is malformed."
        ) from exc
    if (
        not future_lags
        or list(future_lags) != future_lags_raw
        or canonical_sha256(
            {
                "schema": "neural-forge-lfm2-ahb-future-lags-v1",
                "lags": list(future_lags),
            }
        )
        != phase_key.get("future_lags_sha256")
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] future-lag identity changed."
        )

    statistics_receipt = receipt_body.get("statistics_receipt")
    if not isinstance(statistics_receipt, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] has no statistics receipt."
        )
    try:
        verified_statistics = _solver._verify_statistics(
            shard.statistics,
            index=ordinal,
        )
    except Exception as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] in-memory statistics failed authentication."
        ) from exc
    verified_receipt = _json_copy(
        verified_statistics.receipt,
        f"shards[{ordinal}] verified statistics receipt",
    )
    if (
        _json_copy(statistics_receipt, "phase statistics receipt") != verified_receipt
        or verified_statistics.boundary_count != len(boundary_names)
        or verified_statistics.hidden_size != topology.get("hidden_size")
        or verified_statistics.future_lags != future_lags
        or receipt_body.get("source_batch_receipt_sha256")
        != verified_receipt.get("source_receipt_sha256")
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] statistics are not bound to its phase receipt."
        )

    path = Path(shard.path).expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] cache directory is absent or unsafe."
        )
    if {item.name for item in path.iterdir()} != {
        _phase._RECEIPT_FILENAME,
        _phase._TENSOR_FILENAME,
    }:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] exact file inventory changed."
        )
    disk_receipt = _read_canonical_json(path / _phase._RECEIPT_FILENAME)
    if disk_receipt != receipt:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] disk and in-memory receipts differ."
        )
    tensor_file = receipt_body.get("tensor_file")
    if (
        not isinstance(tensor_file, Mapping)
        or tensor_file.get("filename") != _phase._TENSOR_FILENAME
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] tensor-file receipt is malformed."
        )
    tensor_path = path / _phase._TENSOR_FILENAME
    before = _file_identity(tensor_path)
    tensor_file_sha256 = _file_sha256(tensor_path)
    if (
        tensor_file.get("size_bytes") != before[2]
        or tensor_file.get("file_sha256") != tensor_file_sha256
        or _file_identity(tensor_path) != before
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] SafeTensors bytes changed."
        )
    phase_identity = _phase.Lfm2AutoregressiveHankelPhaseIdentity(
        body=MappingProxyType(phase_key),
        phase_key_sha256=phase_key_sha256,
        session=None,
        reference=None,
        future_lags=future_lags,
        boundary_names=boundary_names,
    )
    expected_metadata = _phase._safetensors_metadata(
        phase_identity,
        statistics_receipt,
    )
    try:
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if handle.metadata() != expected_metadata:
                raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                    f"shards[{ordinal}] SafeTensors metadata changed."
                )
            if tuple(sorted(handle.keys())) != _phase._TENSOR_NAMES:
                raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                    f"shards[{ordinal}] tensor inventory changed."
                )
    except Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError:
        raise
    except Exception as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] SafeTensors header failed authentication."
        ) from exc
    if _file_identity(tensor_path) != before:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] SafeTensors changed during authentication."
        )

    payload = verified_receipt.get("tensor_payload")
    if not isinstance(payload, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"shards[{ordinal}] has no tensor payload receipt."
        )
    description = {
        "schema": "neural-forge-lfm2-ahb-observability-source-shard-v1",
        "ordinal": ordinal,
        "phase_key_sha256": phase_key_sha256,
        "phase_receipt_sha256": receipt["receipt_sha256"],
        "statistics_receipt_sha256": verified_receipt["receipt_sha256"],
        "source_batch_receipt_sha256": verified_receipt["source_receipt_sha256"],
        "tensor_payload_sha256": payload["payload_sha256"],
        "phase_tensor_file_sha256": tensor_file_sha256,
        "parent_identity_sha256": phase_key.get("parent_identity_sha256"),
        "member_identity_sha256": phase_key.get("member_identity_sha256"),
        "member_session_sha256": phase_key.get("member_session_sha256"),
        "member_batch_index": phase_key.get("member_batch_index"),
        "trace_input_sha256": phase_key.get("trace_input_sha256"),
        "trace_record_sha256": phase_key.get("trace_record_sha256"),
        "selected_payload_sha256": phase_key.get("selected_payload_sha256"),
        "boundary_topology_sha256": topology_sha256,
        "future_lags_sha256": phase_key.get("future_lags_sha256"),
        "phase_key": phase_key,
    }
    for name in (
        "phase_receipt_sha256",
        "statistics_receipt_sha256",
        "source_batch_receipt_sha256",
        "tensor_payload_sha256",
        "phase_tensor_file_sha256",
        "parent_identity_sha256",
        "member_identity_sha256",
        "member_session_sha256",
        "trace_input_sha256",
        "trace_record_sha256",
        "selected_payload_sha256",
        "future_lags_sha256",
    ):
        _digest(description[name], f"shards[{ordinal}].{name}")
    return _AuthenticatedSourceShard(
        statistics=shard.statistics,
        description=MappingProxyType(description),
        boundary_topology=MappingProxyType(topology),
        boundary_names=boundary_names,
        future_lags=future_lags,
    )


def _numeric_policy(
    *,
    state_ridge_fraction: float,
    future_ridge_fraction: float,
    maximum_condition_number: float,
) -> tuple[float, float, float, dict[str, float]]:
    state_ridge = _finite_positive(state_ridge_fraction, "state_ridge_fraction")
    future_ridge = _finite_positive(
        future_ridge_fraction,
        "future_ridge_fraction",
    )
    condition_limit = _finite_positive(
        maximum_condition_number,
        "maximum_condition_number",
    )
    if condition_limit <= 1.0:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "maximum_condition_number must be greater than one."
        )
    policy = {
        "state_ridge_fraction": state_ridge,
        "future_ridge_fraction": future_ridge,
        "maximum_condition_number": condition_limit,
    }
    return state_ridge, future_ridge, condition_limit, policy


def describe_lfm2_autoregressive_hankel_observability_plan(
    planned_phase_keys: Sequence[str],
    *,
    boundary_names: Sequence[str],
    state_ridge_fraction: float = 1e-8,
    future_ridge_fraction: float = 1e-8,
    maximum_condition_number: float = 1e10,
) -> Lfm2AutoregressiveHankelObservabilityPlan:
    """Build a tensor-free plan before any multi-gigabyte shard is loaded."""

    if (
        isinstance(planned_phase_keys, (str, bytes))
        or not isinstance(planned_phase_keys, Sequence)
        or not planned_phase_keys
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "planned_phase_keys must be a nonempty ordered sequence."
        )
    phase_keys = tuple(
        _digest(value, f"planned_phase_keys[{index}]")
        for index, value in enumerate(planned_phase_keys)
    )
    if len(set(phase_keys)) != len(phase_keys):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "planned_phase_keys must not repeat a phase shard."
        )
    names = _canonical_names(boundary_names)
    state_ridge, future_ridge, condition_limit, _policy = _numeric_policy(
        state_ridge_fraction=state_ridge_fraction,
        future_ridge_fraction=future_ridge_fraction,
        maximum_condition_number=maximum_condition_number,
    )
    moment_numeric_abi = _moment_numeric_abi()
    body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PLAN_SCHEMA,
        "planned_phase_keys": list(phase_keys),
        "source_shard_count": len(phase_keys),
        "source_order_is_semantic": True,
        "boundary_names": list(names),
        "boundary_names_sha256": canonical_sha256(
            {
                "schema": "neural-forge-lfm2-ahb-boundary-order-v1",
                "boundary_names": list(names),
            }
        ),
        "moment_numeric_abi": moment_numeric_abi,
        "moment_numeric_abi_sha256": canonical_sha256(moment_numeric_abi),
        "rolling_pooled_raw_moments": True,
        "peak_shard_tensor_residency": 1,
        "completed_prefixes_skippable_before_shard_load": True,
        "cache_hit_repeats_eigensolves": False,
        "cache_hit_repeats_statistics_aggregation": False,
    }
    return Lfm2AutoregressiveHankelObservabilityPlan(
        body=MappingProxyType(_json_copy(body, "AHB observability plan")),
        plan_key_sha256=canonical_sha256(body),
        planned_phase_keys=phase_keys,
        boundary_names=names,
        state_ridge_fraction=state_ridge,
        future_ridge_fraction=future_ridge,
        maximum_condition_number=condition_limit,
    )


def _aggregate_identity(
    plan: Lfm2AutoregressiveHankelObservabilityPlan,
    *,
    source_descriptions: Sequence[Mapping[str, Any]],
    boundary_topology: Mapping[str, Any],
    future_lags: Sequence[int],
    pooled_statistics_receipt: Mapping[str, Any],
) -> Lfm2AutoregressiveHankelObservabilityCacheIdentity:
    sources = [
        _json_copy(value, "source shard description") for value in source_descriptions
    ]
    if len(sources) != len(plan.planned_phase_keys) or [
        value.get("phase_key_sha256") for value in sources
    ] != list(plan.planned_phase_keys):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "The completed source ledger differs from its ordered plan."
        )
    topology = _json_copy(boundary_topology, "boundary topology")
    if (
        not isinstance(topology, dict)
        or topology.get("boundary_names") != list(plan.boundary_names)
        or topology.get("boundary_count") != len(plan.boundary_names)
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "The completed boundary topology differs from its plan."
        )
    lags = tuple(int(value) for value in future_lags)
    pooled_receipt = _json_copy(
        pooled_statistics_receipt,
        "pooled statistics receipt",
    )
    _sealed_body(
        pooled_receipt,
        "receipt_sha256",
        "pooled statistics receipt",
    )
    pooled_payload = pooled_receipt.get("tensor_payload")
    if not isinstance(pooled_payload, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "The pooled statistics receipt has no tensor payload."
        )
    solver_policy = {
        "state_ridge_fraction": plan.state_ridge_fraction,
        "future_ridge_fraction": plan.future_ridge_fraction,
        "maximum_condition_number": plan.maximum_condition_number,
    }
    solver_numeric_abi = _numeric_abi()
    body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_IDENTITY_SCHEMA,
        "plan_key_sha256": plan.plan_key_sha256,
        "ordered_source_shards": sources,
        "source_shard_count": len(sources),
        "source_order_is_semantic": True,
        "boundary_names": list(plan.boundary_names),
        "boundary_names_sha256": plan.body["boundary_names_sha256"],
        "boundary_topology": topology,
        "boundary_topology_sha256": canonical_sha256(topology),
        "future_lags": list(lags),
        "pooled_statistics_receipt_sha256": pooled_receipt["receipt_sha256"],
        "pooled_source_ledger_sha256": pooled_receipt["source_receipt_sha256"],
        "pooled_tensor_payload_sha256": pooled_payload["payload_sha256"],
        "moment_numeric_abi_sha256": plan.body["moment_numeric_abi_sha256"],
        "solver_numeric_policy": solver_policy,
        "solver_numeric_policy_sha256": canonical_sha256(solver_policy),
        "numeric_abi": solver_numeric_abi,
        "numeric_abi_sha256": canonical_sha256(solver_numeric_abi),
        "solver_implementation_sha256": _source_sha256(
            _solver,
            "autoregressive Hankel observability solver",
        ),
        "cache_implementation_sha256": _source_sha256(
            sys.modules[__name__],
            "autoregressive Hankel observability cache",
        ),
        "rolling_pooled_raw_moments": True,
        "aggregate_moment_reuse": True,
        "cache_hit_repeats_eigensolves": False,
        "cache_hit_repeats_statistics_aggregation": False,
    }
    frozen = MappingProxyType(_json_copy(body, "aggregate cache identity"))
    return Lfm2AutoregressiveHankelObservabilityCacheIdentity(
        body=frozen,
        aggregate_key_sha256=canonical_sha256(body),
        boundary_names=plan.boundary_names,
        state_ridge_fraction=plan.state_ridge_fraction,
        future_ridge_fraction=plan.future_ridge_fraction,
        maximum_condition_number=plan.maximum_condition_number,
    )


def _cache_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheError(
            f"Aggregate observability cache root is absent or unsafe: {root}."
        )
    entries = root / "aggregates"
    entries.mkdir(exist_ok=True)
    if not entries.is_dir() or entries.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheError(
            f"Aggregate observability entry root is unsafe: {entries}."
        )
    work = root / "work"
    work.mkdir(exist_ok=True)
    if not work.is_dir() or work.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheError(
            f"Aggregate observability work root is unsafe: {work}."
        )
    return root


def _entry_path(root: Path, aggregate_key_sha256: str) -> Path:
    return root / "aggregates" / aggregate_key_sha256[:2] / aggregate_key_sha256


def _tensor_name(index: int) -> str:
    return f"{_BOUNDARY_TENSOR_PREFIX}{index:04d}"


def _result_tensors(
    result: _solver.Lfm2AutoregressiveHankelObservabilityResult,
) -> dict[str, torch.Tensor]:
    if not isinstance(
        result,
        _solver.Lfm2AutoregressiveHankelObservabilityResult,
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "The cold solver returned the wrong result type."
        )
    tensors = {_GLOBAL_TENSOR_NAME: _tensor_contract(result.gram, "global Gram")}
    for index, value in enumerate(result.boundary_grams):
        tensors[_tensor_name(index)] = _tensor_contract(
            value,
            f"boundary Gram {index}",
        )
    return tensors


def _tensor_payload(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    records = []
    for name in sorted(tensors):
        value = _tensor_contract(tensors[name], name)
        records.append(
            {
                "name": name,
                "dtype": "float64",
                "shape": [int(size) for size in value.shape],
                "tensor_sha256": _tensor_sha256(value),
            }
        )
    body = {
        "schema": "neural-forge-lfm2-ahb-observability-tensor-payload-v1",
        "tensors": records,
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def _safetensors_metadata(
    identity: Lfm2AutoregressiveHankelObservabilityCacheIdentity,
    *,
    tensor_payload: Mapping[str, Any],
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    evidence_sha256: str,
) -> dict[str, str]:
    return {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_SAFETENSORS_SCHEMA,
        "aggregate_key_sha256": identity.aggregate_key_sha256,
        "tensor_payload_sha256": str(tensor_payload["payload_sha256"]),
        "report_sha256": str(report["report_sha256"]),
        "provenance_sha256": str(provenance["provenance_sha256"]),
        "evidence_sha256": evidence_sha256,
        "boundary_topology_sha256": str(identity.body["boundary_topology_sha256"]),
        "numeric_abi_sha256": str(identity.body["numeric_abi_sha256"]),
    }


def _file_record(path: Path) -> dict[str, Any]:
    before = _file_identity(path)
    digest = _file_sha256(path)
    if _file_identity(path) != before:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"Published file changed while it was hashed: {path}."
        )
    return {
        "filename": path.name,
        "size_bytes": before[2],
        "file_sha256": digest,
    }


def _validate_result_documents(
    identity: Lfm2AutoregressiveHankelObservabilityCacheIdentity,
    *,
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> PrecomputedDownstreamObservability:
    report_body = _sealed_body(report, "report_sha256", "AHB solver report")
    provenance_body = _sealed_body(
        provenance,
        "provenance_sha256",
        "AHB solver provenance",
    )
    source_descriptions = identity.body["ordered_source_shards"]
    expected_statistics = [
        value["statistics_receipt_sha256"] for value in source_descriptions
    ]
    expected_sources = [
        value["source_batch_receipt_sha256"] for value in source_descriptions
    ]
    expected_payloads = [
        value["tensor_payload_sha256"] for value in source_descriptions
    ]
    if (
        report_body.get("schema")
        != _solver.LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_SCHEMA
        or report_body.get("source_trajectory_count") != len(source_descriptions)
        or report_body.get("boundary_names") != list(identity.boundary_names)
        or report_body.get("future_lags") != identity.body["future_lags"]
        or report_body.get("configuration")
        != {
            **dict(identity.body["solver_numeric_policy"]),
            "pooled_pair_centering": True,
            "cross_trajectory_pairs_created": False,
            "lag_zero_used_for_selection": False,
            "nonzero_lags": [lag for lag in identity.body["future_lags"] if lag > 0],
            "equal_horizon_weighting": True,
            "horizon_trace_normalization": False,
            "equal_boundary_weighting": True,
            "boundary_trace_normalization": False,
        }
        or provenance_body.get("source_statistics_receipt_sha256s")
        != expected_statistics
        or provenance_body.get("source_batch_receipt_sha256s") != expected_sources
        or provenance_body.get("source_tensor_payload_sha256s") != expected_payloads
        or provenance_body.get("boundary_names") != list(identity.boundary_names)
        or provenance_body.get("future_lags") != identity.body["future_lags"]
        or provenance_body.get("configuration") != report_body.get("configuration")
        or report_body.get("streaming_raw_moment_pooling") is not True
        or provenance_body.get("streaming_raw_moment_pooling") is not True
        or report_body.get("pooled_statistics_receipt_sha256")
        != identity.body["pooled_statistics_receipt_sha256"]
        or provenance_body.get("pooled_statistics_receipt_sha256")
        != identity.body["pooled_statistics_receipt_sha256"]
        or report_body.get("pooled_source_ledger_sha256")
        != identity.body["pooled_source_ledger_sha256"]
        or provenance_body.get("pooled_source_ledger_sha256")
        != identity.body["pooled_source_ledger_sha256"]
        or report_body.get("pooled_tensor_payload_sha256")
        != identity.body["pooled_tensor_payload_sha256"]
        or provenance_body.get("pooled_tensor_payload_sha256")
        != identity.body["pooled_tensor_payload_sha256"]
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Persisted AHB report or provenance differs from its aggregate key."
        )

    boundary_names = identity.boundary_names
    if len(tensors) != len(boundary_names) + 1:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Persisted AHB tensor count differs from its boundary topology."
        )
    global_gram = _tensor_contract(tensors[_GLOBAL_TENSOR_NAME], "global Gram")
    boundary_grams = tuple(
        _tensor_contract(tensors[_tensor_name(index)], f"boundary Gram {index}")
        for index in range(len(boundary_names))
    )
    global_sha256 = _tensor_sha256(global_gram)
    boundary_sha256s = [_tensor_sha256(value) for value in boundary_grams]
    boundaries = report_body.get("boundaries")
    if (
        report_body.get("global_gram_sha256") != global_sha256
        or provenance_body.get("global_gram_sha256") != global_sha256
        or provenance_body.get("boundary_gram_sha256s") != boundary_sha256s
        or not isinstance(boundaries, list)
        or len(boundaries) != len(boundary_names)
        or any(
            not isinstance(value, Mapping)
            or value.get("boundary_index") != index
            or value.get("boundary_name") != boundary_names[index]
            or value.get("gram_sha256") != boundary_sha256s[index]
            for index, value in enumerate(boundaries)
        )
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Persisted AHB Grams differ from their report or provenance."
        )

    descriptor = report_body.get("precomputed_observability")
    if not isinstance(descriptor, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Persisted AHB report has no observability descriptor."
        )
    evidence = PrecomputedDownstreamObservability(
        evidence_id=descriptor.get("evidence_id"),
        node_id=descriptor.get("node_id"),
        circuit_kind=descriptor.get("circuit_kind"),
        role=descriptor.get("role"),
        gram=global_gram,
        provenance={
            **provenance_body,
            "provenance_sha256": provenance["provenance_sha256"],
        },
        interface_ids=tuple(descriptor.get("interface_ids", ())),
        gram_weight=descriptor.get("gram_weight", 1.0),
    )
    if evidence.as_dict() != dict(descriptor):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Persisted AHB observability descriptor changed."
        )
    return evidence


def _rebind_pooled_solver_result(
    result: _solver.Lfm2AutoregressiveHankelObservabilityResult,
    identity: Lfm2AutoregressiveHankelObservabilityCacheIdentity,
    pooled_statistics: Any,
) -> _solver.Lfm2AutoregressiveHankelObservabilityResult:
    """Rebind one mathematically equivalent pooled solve to every source."""

    sources = identity.body["ordered_source_shards"]
    pooled_receipt = _json_copy(
        pooled_statistics.receipt,
        "pooled statistics receipt",
    )
    pooled_payload = pooled_receipt.get("tensor_payload")
    if not isinstance(pooled_payload, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Pooled statistics have no authenticated tensor payload."
        )
    original_provenance = _sealed_body(
        result.evidence.provenance,
        "provenance_sha256",
        "pooled solver provenance",
    )
    provenance_body = {
        **original_provenance,
        "source_statistics_receipt_sha256s": [
            value["statistics_receipt_sha256"] for value in sources
        ],
        "source_batch_receipt_sha256s": [
            value["source_batch_receipt_sha256"] for value in sources
        ],
        "source_tensor_payload_sha256s": [
            value["tensor_payload_sha256"] for value in sources
        ],
        "streaming_raw_moment_pooling": True,
        "pooled_statistics_receipt_sha256": pooled_receipt["receipt_sha256"],
        "pooled_source_ledger_sha256": pooled_receipt["source_receipt_sha256"],
        "pooled_tensor_payload_sha256": pooled_payload["payload_sha256"],
        "pooled_source_trajectory_count": len(sources),
        "peak_source_shards_resident": 1,
    }
    provenance = {
        **provenance_body,
        "provenance_sha256": canonical_sha256(provenance_body),
    }
    evidence = PrecomputedDownstreamObservability(
        evidence_id=(
            "interface.text.logits.autoregressive_hankel."
            + provenance["provenance_sha256"][:16]
        ),
        node_id=result.evidence.node_id,
        circuit_kind=result.evidence.circuit_kind,
        role=result.evidence.role,
        gram=result.gram,
        provenance=provenance,
        interface_ids=result.evidence.interface_ids,
        gram_weight=result.evidence.gram_weight,
    )
    original_report = _sealed_body(
        result.report,
        "report_sha256",
        "pooled solver report",
    )
    report_body = {
        **original_report,
        "source_trajectory_count": len(sources),
        "precomputed_observability": evidence.as_dict(),
        "streaming_raw_moment_pooling": True,
        "pooled_statistics_receipt_sha256": pooled_receipt["receipt_sha256"],
        "pooled_source_ledger_sha256": pooled_receipt["source_receipt_sha256"],
        "pooled_tensor_payload_sha256": pooled_payload["payload_sha256"],
        "peak_source_shards_resident": 1,
        "source_shards_may_be_pruned_after_durable_absorption": True,
    }
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    return _solver.Lfm2AutoregressiveHankelObservabilityResult(
        evidence=evidence,
        boundary_grams=result.boundary_grams,
        report=report,
    )


def _write_entry(
    temporary: Path,
    identity: Lfm2AutoregressiveHankelObservabilityCacheIdentity,
    result: _solver.Lfm2AutoregressiveHankelObservabilityResult,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Persistent LFM2 Hankel observability requires safetensors."
        ) from exc
    tensors = _result_tensors(result)
    report = _json_copy(result.report, "cold AHB solver report")
    provenance = _json_copy(result.evidence.provenance, "cold AHB provenance")
    evidence = _validate_result_documents(
        identity,
        report=report,
        provenance=provenance,
        tensors=tensors,
    )
    if evidence.as_dict() != result.evidence.as_dict():
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Cold AHB evidence changed before publication."
        )
    payload = _tensor_payload(tensors)
    _write_canonical_json(temporary / _PROVENANCE_FILENAME, provenance)
    _write_canonical_json(temporary / _REPORT_FILENAME, report)
    tensor_path = temporary / _TENSOR_FILENAME
    save_file(
        {name: value for name, value in sorted(tensors.items())},
        tensor_path,
        metadata=_safetensors_metadata(
            identity,
            tensor_payload=payload,
            report=report,
            provenance=provenance,
            evidence_sha256=evidence.evidence_sha256,
        ),
    )
    with tensor_path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())

    cache_provenance_body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_PROVENANCE_SCHEMA,
        "aggregate_key_sha256": identity.aggregate_key_sha256,
        "ordered_source_phase_key_sha256s": [
            value["phase_key_sha256"]
            for value in identity.body["ordered_source_shards"]
        ],
        "solver_implementation_sha256": identity.body["solver_implementation_sha256"],
        "numeric_abi_sha256": identity.body["numeric_abi_sha256"],
        "solver_numeric_policy_sha256": identity.body["solver_numeric_policy_sha256"],
        "solver_report_sha256": report["report_sha256"],
        "solver_provenance_sha256": provenance["provenance_sha256"],
        "tensor_payload_sha256": payload["payload_sha256"],
        "fold_only_closed_form": True,
        "model_or_tokenizer_accepted": False,
        "training_used": False,
    }
    cache_provenance = {
        **cache_provenance_body,
        "provenance_sha256": canonical_sha256(cache_provenance_body),
    }
    receipt_body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_RECEIPT_SCHEMA,
        "cache_schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_SCHEMA,
        "aggregate_key": dict(identity.body),
        "aggregate_key_sha256": identity.aggregate_key_sha256,
        "cache_provenance": cache_provenance,
        "solver_report_sha256": report["report_sha256"],
        "solver_provenance_sha256": provenance["provenance_sha256"],
        "precomputed_observability_evidence_sha256": evidence.evidence_sha256,
        "tensor_payload": payload,
        "files": {
            filename: _file_record(temporary / filename)
            for filename in (
                _PROVENANCE_FILENAME,
                _REPORT_FILENAME,
                _TENSOR_FILENAME,
            )
        },
        "operation_audit": dict(_ZERO_EXECUTION_AUDIT),
        "cold_solver_calls": 1,
        "cache_hit_solver_calls": 0,
        "atomic_directory_publication": True,
        "immutable_by_contract": True,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    _write_canonical_json(temporary / _RECEIPT_FILENAME, receipt)


def _verify_file_record(path: Path, record: Any, name: str) -> None:
    if not isinstance(record, Mapping) or record.get("filename") != path.name:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} file receipt is malformed."
        )
    before = _file_identity(path)
    if (
        record.get("size_bytes") != before[2]
        or record.get("file_sha256") != _file_sha256(path)
        or _file_identity(path) != before
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"{name} bytes changed."
        )


def _load_entry(
    path: Path,
    identity: Lfm2AutoregressiveHankelObservabilityCacheIdentity,
    *,
    cache_hit: bool,
) -> AuthenticatedLfm2AutoregressiveHankelObservability:
    if not path.is_dir() or path.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            f"Aggregate observability entry is absent or unsafe: {path}."
        )
    if frozenset(item.name for item in path.iterdir()) != _EXACT_FILE_INVENTORY:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate observability exact file inventory changed."
        )
    receipt = _read_canonical_json(path / _RECEIPT_FILENAME)
    receipt_body = _sealed_body(
        receipt,
        "receipt_sha256",
        "aggregate observability receipt",
    )
    if (
        receipt_body.get("schema")
        != LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_RECEIPT_SCHEMA
        or receipt_body.get("cache_schema")
        != LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_SCHEMA
        or receipt_body.get("aggregate_key") != dict(identity.body)
        or receipt_body.get("aggregate_key_sha256") != identity.aggregate_key_sha256
        or receipt_body.get("operation_audit") != _ZERO_EXECUTION_AUDIT
        or receipt_body.get("cold_solver_calls") != 1
        or receipt_body.get("cache_hit_solver_calls") != 0
        or receipt_body.get("atomic_directory_publication") is not True
        or receipt_body.get("immutable_by_contract") is not True
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate observability receipt differs from its requested identity."
        )
    cache_provenance = receipt_body.get("cache_provenance")
    cache_provenance_body = _sealed_body(
        cache_provenance,
        "provenance_sha256",
        "aggregate cache provenance",
    )
    if (
        cache_provenance_body.get("schema")
        != LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_PROVENANCE_SCHEMA
        or cache_provenance_body.get("aggregate_key_sha256")
        != identity.aggregate_key_sha256
        or cache_provenance_body.get("solver_implementation_sha256")
        != identity.body["solver_implementation_sha256"]
        or cache_provenance_body.get("numeric_abi_sha256")
        != identity.body["numeric_abi_sha256"]
        or cache_provenance_body.get("solver_numeric_policy_sha256")
        != identity.body["solver_numeric_policy_sha256"]
        or cache_provenance_body.get("fold_only_closed_form") is not True
        or cache_provenance_body.get("model_or_tokenizer_accepted") is not False
        or cache_provenance_body.get("training_used") is not False
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate cache provenance differs from its requested identity."
        )

    files = receipt_body.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        _PROVENANCE_FILENAME,
        _REPORT_FILENAME,
        _TENSOR_FILENAME,
    }:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate observability file receipts changed."
        )
    for filename, record in files.items():
        _verify_file_record(path / filename, record, filename)
    provenance = _read_canonical_json(path / _PROVENANCE_FILENAME)
    report = _read_canonical_json(path / _REPORT_FILENAME)
    if (
        provenance.get("provenance_sha256")
        != receipt_body.get("solver_provenance_sha256")
        or report.get("report_sha256") != receipt_body.get("solver_report_sha256")
        or cache_provenance_body.get("solver_provenance_sha256")
        != provenance.get("provenance_sha256")
        or cache_provenance_body.get("solver_report_sha256")
        != report.get("report_sha256")
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate documents differ from their receipt."
        )

    payload = receipt_body.get("tensor_payload")
    if not isinstance(payload, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate tensor payload receipt is absent."
        )
    descriptor = report.get("precomputed_observability")
    if not isinstance(descriptor, Mapping):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate solver report has no observability descriptor."
        )
    expected_metadata = _safetensors_metadata(
        identity,
        tensor_payload=payload,
        report=report,
        provenance=provenance,
        evidence_sha256=str(receipt_body["precomputed_observability_evidence_sha256"]),
    )
    tensor_path = path / _TENSOR_FILENAME
    before = _file_identity(tensor_path)
    try:
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if handle.metadata() != expected_metadata:
                raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                    "Aggregate SafeTensors metadata changed."
                )
            expected_names = tuple(
                sorted(
                    (_GLOBAL_TENSOR_NAME,)
                    + tuple(
                        _tensor_name(index)
                        for index in range(len(identity.boundary_names))
                    )
                )
            )
            if tuple(sorted(handle.keys())) != expected_names:
                raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                    "Aggregate SafeTensors tensor inventory changed."
                )
            tensors = {
                name: handle.get_tensor(name)
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
                .clone()
                for name in expected_names
            }
    except Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError:
        raise
    except Exception as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate SafeTensors could not be authenticated."
        ) from exc
    if _file_identity(tensor_path) != before:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate SafeTensors changed while it was loaded."
        )
    if _tensor_payload(tensors) != dict(payload):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate tensor payload changed."
        )
    evidence = _validate_result_documents(
        identity,
        report=report,
        provenance=provenance,
        tensors=tensors,
    )
    if evidence.evidence_sha256 != receipt_body.get(
        "precomputed_observability_evidence_sha256"
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Aggregate evidence seal changed."
        )
    result = _solver.Lfm2AutoregressiveHankelObservabilityResult(
        evidence=evidence,
        boundary_grams=tuple(
            tensors[_tensor_name(index)]
            for index in range(len(identity.boundary_names))
        ),
        report=report,
    )
    if result.as_dict() != report or not result.evidence.verify():
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Reconstructed aggregate AHB result failed verification."
        )
    return AuthenticatedLfm2AutoregressiveHankelObservability(
        result=result,
        receipt=MappingProxyType(receipt),
        provenance=MappingProxyType(provenance),
        path=path,
        aggregate_key_sha256=identity.aggregate_key_sha256,
        cache_hit=cache_hit,
    )


def _pooled_statistics_tensors(statistics: Any) -> dict[str, torch.Tensor]:
    try:
        tensors = _phase._statistics_tensors(statistics)
    except Exception as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Pooled Hankel statistics violated their tensor contract."
        ) from exc
    return {name: tensors[name] for name in _phase._TENSOR_NAMES}


def _pooled_statistics_receipt(
    tensors: Mapping[str, torch.Tensor],
    *,
    source_descriptions: Sequence[Mapping[str, Any]],
    boundary_topology: Mapping[str, Any],
    future_lags: Sequence[int],
) -> dict[str, Any]:
    lags = tuple(int(value) for value in future_lags)
    ledger_body = {
        "schema": "neural-forge-lfm2-ahb-pooled-source-ledger-v1",
        "ordered_source_shards": [
            _json_copy(value, "pooled source description")
            for value in source_descriptions
        ],
    }
    source_digest = canonical_sha256(ledger_body)
    payload = _phase._statistics._tensor_payload_receipt(tensors)
    lag_pair_counts = tensors["lag_pair_counts"].tolist()
    right_censor_counts = tensors["right_censor_counts"].tolist()
    topology = _json_copy(boundary_topology, "pooled boundary topology")
    body = {
        "schema": _phase._statistics.LFM2_AUTOREGRESSIVE_HANKEL_STATISTICS_SCHEMA,
        "source_receipt_sha256": source_digest,
        "boundary_count": int(topology["boundary_count"]),
        "trajectory_row_count": int(lag_pair_counts[0]),
        "hidden_size": int(topology["hidden_size"]),
        "future_lags": list(lags),
        "future_lag_count": len(lags),
        "maximum_horizon": lags[-1],
        "dense_future_lag_schedule": lags == tuple(range(lags[-1] + 1)),
        "lag_pair_counts": lag_pair_counts,
        "right_censor_counts": right_censor_counts,
        "arithmetic_device": "cpu",
        "accumulation_dtype": "float64",
        "terminal_rows_zero_padded": 0,
        "contains_nonzero_future_lags": True,
        "model_wide_residual_boundaries_required": True,
        "static_gram_only_method": False,
        "short_conv_only_fhvr_method": False,
        "raw_uncentered_moments": True,
        "streaming_pooled_source_count": len(source_descriptions),
        "cross_trajectory_pairs_created": False,
        "tensor_payload": payload,
        "operation_audit": _phase._statistics._zero_execution_audit(),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _statistics_from_pool(
    tensors: Mapping[str, torch.Tensor],
    *,
    source_descriptions: Sequence[Mapping[str, Any]],
    boundary_topology: Mapping[str, Any],
    future_lags: Sequence[int],
) -> Any:
    receipt = _pooled_statistics_receipt(
        tensors,
        source_descriptions=source_descriptions,
        boundary_topology=boundary_topology,
        future_lags=future_lags,
    )
    statistics = _phase._statistics_from_tensors(tensors, receipt)
    try:
        _solver._verify_statistics(statistics, index=0)
    except Exception as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "Pooled Hankel statistics failed closed verification."
        ) from exc
    return statistics


def _clone_first_pool(item: _AuthenticatedSourceShard) -> Any:
    source = _pooled_statistics_tensors(item.statistics)
    with torch.inference_mode():
        tensors = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in source.items()
        }
    return _statistics_from_pool(
        tensors,
        source_descriptions=(item.description,),
        boundary_topology=item.boundary_topology,
        future_lags=item.future_lags,
    )


def _add_to_pool(
    pooled: Any,
    item: _AuthenticatedSourceShard,
    *,
    source_descriptions: Sequence[Mapping[str, Any]],
    boundary_topology: Mapping[str, Any],
    future_lags: Sequence[int],
) -> Any:
    pooled_tensors = _pooled_statistics_tensors(pooled)
    source_tensors = _pooled_statistics_tensors(item.statistics)
    if {
        name: (value.dtype, tuple(value.shape))
        for name, value in pooled_tensors.items()
    } != {
        name: (value.dtype, tuple(value.shape))
        for name, value in source_tensors.items()
    }:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "A streamed shard differs from the pooled moment geometry."
        )
    with torch.inference_mode():
        for name in _phase._TENSOR_NAMES:
            pooled_tensors[name].add_(source_tensors[name])
    return _statistics_from_pool(
        pooled_tensors,
        source_descriptions=source_descriptions,
        boundary_topology=boundary_topology,
        future_lags=future_lags,
    )


def _work_path(
    root: Path,
    plan: Lfm2AutoregressiveHankelObservabilityPlan,
) -> Path:
    return root / "work" / plan.plan_key_sha256[:2] / plan.plan_key_sha256


def _atomic_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_canonical_json(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _progress_metadata(
    plan: Lfm2AutoregressiveHankelObservabilityPlan,
    *,
    prefix_key_sha256: str,
    statistics_receipt: Mapping[str, Any],
    boundary_topology_sha256: str,
) -> dict[str, str]:
    return {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROGRESS_SAFETENSORS_SCHEMA,
        "plan_key_sha256": plan.plan_key_sha256,
        "prefix_key_sha256": prefix_key_sha256,
        "statistics_receipt_sha256": str(statistics_receipt["receipt_sha256"]),
        "tensor_payload_sha256": str(
            statistics_receipt["tensor_payload"]["payload_sha256"]
        ),
        "boundary_topology_sha256": boundary_topology_sha256,
        "moment_numeric_abi_sha256": str(plan.body["moment_numeric_abi_sha256"]),
    }


def _publish_progress(
    work: Path,
    plan: Lfm2AutoregressiveHankelObservabilityPlan,
    *,
    pooled_statistics: Any,
    source_descriptions: Sequence[Mapping[str, Any]],
    boundary_topology: Mapping[str, Any],
    future_lags: Sequence[int],
    previous_prefix: Path | None,
) -> tuple[Path, Mapping[str, Any]]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Persistent LFM2 Hankel observability requires safetensors."
        ) from exc
    tensors = _pooled_statistics_tensors(pooled_statistics)
    statistics_receipt = _json_copy(
        pooled_statistics.receipt,
        "pooled progress statistics receipt",
    )
    topology = _json_copy(boundary_topology, "progress boundary topology")
    prefix_body = {
        "schema": "neural-forge-lfm2-ahb-pooled-prefix-key-v1",
        "plan_key_sha256": plan.plan_key_sha256,
        "ordered_source_shards": [
            _json_copy(value, "progress source shard") for value in source_descriptions
        ],
        "boundary_topology_sha256": canonical_sha256(topology),
        "future_lags": list(future_lags),
        "pooled_statistics_receipt_sha256": statistics_receipt["receipt_sha256"],
        "pooled_tensor_payload_sha256": statistics_receipt["tensor_payload"][
            "payload_sha256"
        ],
    }
    prefix_key = canonical_sha256(prefix_body)
    prefixes = work / "prefixes"
    prefixes.mkdir(parents=True, exist_ok=True)
    if not prefixes.is_dir() or prefixes.is_symlink():
        raise Lfm2AutoregressiveHankelObservabilityCacheError(
            f"AHB progress prefix root is unsafe: {prefixes}."
        )
    prefix_name = f"prefix-{len(source_descriptions):08d}-{prefix_key}"
    prefix_path = prefixes / prefix_name
    temporary = prefixes / f".{prefix_name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        tensor_path = temporary / _PROGRESS_TENSOR_FILENAME
        save_file(
            {name: tensors[name] for name in _phase._TENSOR_NAMES},
            tensor_path,
            metadata=_progress_metadata(
                plan,
                prefix_key_sha256=prefix_key,
                statistics_receipt=statistics_receipt,
                boundary_topology_sha256=canonical_sha256(topology),
            ),
        )
        with tensor_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        tensor_record = _file_record(tensor_path)
        progress_body = {
            "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROGRESS_SCHEMA,
            "plan": dict(plan.body),
            "plan_key_sha256": plan.plan_key_sha256,
            "prefix_key_sha256": prefix_key,
            "processed_source_count": len(source_descriptions),
            "ordered_source_shards": [
                _json_copy(value, "progress source shard")
                for value in source_descriptions
            ],
            "boundary_topology": topology,
            "boundary_topology_sha256": canonical_sha256(topology),
            "future_lags": list(future_lags),
            "statistics_receipt": statistics_receipt,
            "tensor_file": tensor_record,
            "rolling_checkpoint_count": 1,
            "peak_source_shards_resident": 1,
            "all_processed_phase_shards_prunable": True,
            "operation_audit": dict(_ZERO_EXECUTION_AUDIT),
            "immutable_prefix_by_contract": True,
        }
        progress = {
            **progress_body,
            "receipt_sha256": canonical_sha256(progress_body),
        }
        _write_canonical_json(temporary / _PROGRESS_RECEIPT_FILENAME, progress)
        try:
            temporary.rename(prefix_path)
        except FileExistsError:
            pass
        except OSError:
            if not prefix_path.exists():
                raise
        if temporary.exists():
            shutil.rmtree(temporary)

        disk_progress = _read_canonical_json(prefix_path / _PROGRESS_RECEIPT_FILENAME)
        if disk_progress != progress:
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                "Published AHB progress receipt differs from memory."
            )
        before = _file_identity(prefix_path / _PROGRESS_TENSOR_FILENAME)
        if (
            before[2] != tensor_record["size_bytes"]
            or _file_sha256(prefix_path / _PROGRESS_TENSOR_FILENAME)
            != tensor_record["file_sha256"]
            or _file_identity(prefix_path / _PROGRESS_TENSOR_FILENAME) != before
        ):
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                "Published AHB pooled tensor bytes failed authentication."
            )
        with safe_open(
            prefix_path / _PROGRESS_TENSOR_FILENAME,
            framework="pt",
            device="cpu",
        ) as handle:
            if (
                handle.metadata()
                != _progress_metadata(
                    plan,
                    prefix_key_sha256=prefix_key,
                    statistics_receipt=statistics_receipt,
                    boundary_topology_sha256=canonical_sha256(topology),
                )
                or tuple(sorted(handle.keys())) != _phase._TENSOR_NAMES
            ):
                raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                    "Published AHB pooled SafeTensors metadata changed."
                )

        current_body = {
            "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CURRENT_SCHEMA,
            "plan_key_sha256": plan.plan_key_sha256,
            "prefix_directory": prefix_name,
            "prefix_key_sha256": prefix_key,
            "prefix_receipt_sha256": progress["receipt_sha256"],
            "processed_source_count": len(source_descriptions),
        }
        current = {
            **current_body,
            "current_sha256": canonical_sha256(current_body),
        }
        _atomic_json_replace(work / _CURRENT_FILENAME, current)
        if _read_canonical_json(work / _CURRENT_FILENAME) != current:
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                "AHB current-prefix pointer failed atomic authentication."
            )
        if (
            previous_prefix is not None
            and previous_prefix != prefix_path
            and previous_prefix.is_dir()
            and not previous_prefix.is_symlink()
        ):
            shutil.rmtree(previous_prefix)
        return prefix_path, MappingProxyType(progress)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_progress(
    work: Path,
    plan: Lfm2AutoregressiveHankelObservabilityPlan,
    *,
    load_tensors: bool,
) -> (
    tuple[
        Any,
        list[dict[str, Any]],
        dict[str, Any],
        tuple[int, ...],
        Path,
        Mapping[str, Any],
    ]
    | None
):
    current_path = work / _CURRENT_FILENAME
    if not current_path.exists():
        return None
    current = _read_canonical_json(current_path)
    current_body = _sealed_body(current, "current_sha256", "AHB current pointer")
    prefix_name = current_body.get("prefix_directory")
    if (
        current_body.get("schema")
        != LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CURRENT_SCHEMA
        or current_body.get("plan_key_sha256") != plan.plan_key_sha256
        or not isinstance(prefix_name, str)
        or re.fullmatch(r"prefix-[0-9]{8}-[0-9a-f]{64}", prefix_name) is None
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB current-prefix pointer differs from its plan."
        )
    prefix_path = work / "prefixes" / prefix_name
    if (
        not prefix_path.is_dir()
        or prefix_path.is_symlink()
        or {item.name for item in prefix_path.iterdir()}
        != {_PROGRESS_RECEIPT_FILENAME, _PROGRESS_TENSOR_FILENAME}
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB current prefix is absent or has an unsafe inventory."
        )
    progress = _read_canonical_json(prefix_path / _PROGRESS_RECEIPT_FILENAME)
    progress_body = _sealed_body(progress, "receipt_sha256", "AHB progress receipt")
    sources = progress_body.get("ordered_source_shards")
    topology = progress_body.get("boundary_topology")
    future_lags_raw = progress_body.get("future_lags")
    if (
        progress_body.get("schema")
        != LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROGRESS_SCHEMA
        or progress_body.get("plan") != dict(plan.body)
        or progress_body.get("plan_key_sha256") != plan.plan_key_sha256
        or progress_body.get("prefix_key_sha256")
        != current_body.get("prefix_key_sha256")
        or progress.get("receipt_sha256") != current_body.get("prefix_receipt_sha256")
        or not isinstance(sources, list)
        or progress_body.get("processed_source_count") != len(sources)
        or len(sources) != current_body.get("processed_source_count")
        or [value.get("phase_key_sha256") for value in sources]
        != list(plan.planned_phase_keys[: len(sources)])
        or not isinstance(topology, dict)
        or topology.get("boundary_names") != list(plan.boundary_names)
        or not isinstance(future_lags_raw, list)
        or progress_body.get("all_processed_phase_shards_prunable") is not True
        or progress_body.get("operation_audit") != _ZERO_EXECUTION_AUDIT
        or progress_body.get("immutable_prefix_by_contract") is not True
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB progress receipt differs from its plan or current pointer."
        )
    statistics_receipt = progress_body.get("statistics_receipt")
    tensor_record = progress_body.get("tensor_file")
    if not isinstance(statistics_receipt, Mapping) or not isinstance(
        tensor_record, Mapping
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB progress tensor receipts are absent."
        )
    tensor_path = prefix_path / _PROGRESS_TENSOR_FILENAME
    expected_metadata = _progress_metadata(
        plan,
        prefix_key_sha256=str(progress_body["prefix_key_sha256"]),
        statistics_receipt=statistics_receipt,
        boundary_topology_sha256=str(progress_body["boundary_topology_sha256"]),
    )
    if (
        tensor_record.get("filename") != _PROGRESS_TENSOR_FILENAME
        or not tensor_path.is_file()
        or tensor_path.is_symlink()
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB progress pooled tensor file is absent or unsafe."
        )
    before = _file_identity(tensor_path)
    if load_tensors:
        _verify_file_record(tensor_path, tensor_record, "pooled progress")
    try:
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if (
                handle.metadata() != expected_metadata
                or tuple(sorted(handle.keys())) != _phase._TENSOR_NAMES
            ):
                raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                    "AHB progress SafeTensors metadata changed."
                )
            tensors = (
                {
                    name: handle.get_tensor(name)
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                    .clone()
                    for name in _phase._TENSOR_NAMES
                }
                if load_tensors
                else None
            )
    except Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError:
        raise
    except Exception as exc:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB progress SafeTensors could not be loaded."
        ) from exc
    if _file_identity(tensor_path) != before:
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB progress tensors changed while loading."
        )
    if load_tensors and (
        not isinstance(tensors, Mapping)
        or _phase._statistics._tensor_payload_receipt(tensors)
        != statistics_receipt.get("tensor_payload")
    ):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "AHB progress pooled tensor payload changed."
        )
    future_lags = tuple(int(value) for value in future_lags_raw)
    statistics = None
    if load_tensors:
        statistics = _phase._statistics_from_tensors(tensors, statistics_receipt)
        try:
            _solver._verify_statistics(statistics, index=0)
        except Exception as exc:
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                "AHB progress pooled statistics failed verification."
            ) from exc
    return (
        statistics,
        [_json_copy(value, "resumed source description") for value in sources],
        _json_copy(topology, "resumed boundary topology"),
        future_lags,
        prefix_path,
        MappingProxyType(progress),
    )


def _absorption_receipt(
    *,
    plan: Lfm2AutoregressiveHankelObservabilityPlan,
    ordinal: int,
    shard: _phase.AuthenticatedLfm2AutoregressiveHankelShard,
    progress: Mapping[str, Any],
) -> Mapping[str, Any]:
    body = {
        "schema": "neural-forge-lfm2-ahb-observability-absorption-v1",
        "plan_key_sha256": plan.plan_key_sha256,
        "source_ordinal": ordinal,
        "phase_key_sha256": shard.phase_key_sha256,
        "phase_shard_path": str(Path(shard.path).absolute()),
        "durable_progress_receipt_sha256": progress["receipt_sha256"],
        "durable_processed_source_count": progress["processed_source_count"],
        "phase_shard_prunable_after_commit": True,
        "pooled_checkpoint_authenticated_before_prunable": True,
        "source_tensor_references_retained_by_accumulator": 0,
    }
    return MappingProxyType({**body, "receipt_sha256": canonical_sha256(body)})


class Lfm2AutoregressiveHankelObservabilityAccumulator:
    """Bounded-memory, resumable raw-moment pool for one ordered plan."""

    def __init__(
        self,
        root: Path,
        plan: Lfm2AutoregressiveHankelObservabilityPlan,
    ) -> None:
        self.root = root
        self.plan = plan
        self.work = _work_path(root, plan)
        self.work.mkdir(parents=True, exist_ok=True)
        if not self.work.is_dir() or self.work.is_symlink():
            raise Lfm2AutoregressiveHankelObservabilityCacheError(
                f"AHB accumulator work directory is unsafe: {self.work}."
            )
        self._pooled_statistics: Any | None = None
        self._sources: list[dict[str, Any]] = []
        self._boundary_topology: dict[str, Any] | None = None
        self._future_lags: tuple[int, ...] | None = None
        self._prefix_path: Path | None = None
        self._progress: Mapping[str, Any] | None = None
        self._warm_identity: (
            Lfm2AutoregressiveHankelObservabilityCacheIdentity | None
        ) = None
        self._poisoned = False
        self._restore()

    def _restore(self) -> None:
        metadata = _load_progress(self.work, self.plan, load_tensors=False)
        if metadata is None:
            return
        (
            _unused_statistics,
            self._sources,
            self._boundary_topology,
            self._future_lags,
            self._prefix_path,
            self._progress,
        ) = metadata
        if self.processed_source_count == len(self.plan.planned_phase_keys):
            identity = _aggregate_identity(
                self.plan,
                source_descriptions=self._sources,
                boundary_topology=self._boundary_topology,
                future_lags=self._future_lags,
                pooled_statistics_receipt=self._progress["statistics_receipt"],
            )
            if _entry_path(self.root, identity.aggregate_key_sha256).exists():
                self._warm_identity = identity
                return
        restored = _load_progress(self.work, self.plan, load_tensors=True)
        if restored is not None:
            (
                self._pooled_statistics,
                self._sources,
                self._boundary_topology,
                self._future_lags,
                self._prefix_path,
                self._progress,
            ) = restored

    @property
    def processed_source_count(self) -> int:
        return len(self._sources)

    @property
    def next_expected_phase_key(self) -> str | None:
        if self.processed_source_count >= len(self.plan.planned_phase_keys):
            return None
        return self.plan.planned_phase_keys[self.processed_source_count]

    @property
    def committed_phase_keys(self) -> tuple[str, ...]:
        return tuple(str(value["phase_key_sha256"]) for value in self._sources)

    @property
    def completed(self) -> bool:
        return self.processed_source_count == len(self.plan.planned_phase_keys)

    def absorb(
        self,
        shard: _phase.AuthenticatedLfm2AutoregressiveHankelShard,
    ) -> Mapping[str, Any]:
        if self._poisoned:
            raise Lfm2AutoregressiveHankelObservabilityCacheError(
                "The accumulator must be reopened from its last durable prefix."
            )
        ordinal = self.processed_source_count
        expected = self.next_expected_phase_key
        if expected is None:
            raise Lfm2AutoregressiveHankelObservabilityCacheError(
                "The ordered AHB source plan is already complete."
            )
        if getattr(shard, "phase_key_sha256", None) != expected:
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                f"Expected phase key {expected}, not {getattr(shard, 'phase_key_sha256', None)}."
            )
        item = _authenticate_source_shard(shard, ordinal=ordinal)
        if item.boundary_names != self.plan.boundary_names:
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                "The streamed shard boundary order differs from its plan."
            )
        if self._boundary_topology is None:
            self._boundary_topology = dict(item.boundary_topology)
            self._future_lags = item.future_lags
        elif (
            dict(item.boundary_topology) != self._boundary_topology
            or item.future_lags != self._future_lags
        ):
            raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
                "The streamed shard geometry differs from the durable prefix."
            )
        candidate_sources = [*self._sources, dict(item.description)]
        try:
            if self._pooled_statistics is None:
                pooled = _clone_first_pool(item)
            else:
                pooled = _add_to_pool(
                    self._pooled_statistics,
                    item,
                    source_descriptions=candidate_sources,
                    boundary_topology=self._boundary_topology,
                    future_lags=self._future_lags,
                )
            prefix_path, progress = _publish_progress(
                self.work,
                self.plan,
                pooled_statistics=pooled,
                source_descriptions=candidate_sources,
                boundary_topology=self._boundary_topology,
                future_lags=self._future_lags,
                previous_prefix=self._prefix_path,
            )
        except Exception:
            self._poisoned = True
            raise
        self._pooled_statistics = pooled
        self._sources = candidate_sources
        self._prefix_path = prefix_path
        self._progress = progress
        return _absorption_receipt(
            plan=self.plan,
            ordinal=ordinal,
            shard=shard,
            progress=progress,
        )

    def finalize(self) -> AuthenticatedLfm2AutoregressiveHankelObservability:
        if self._poisoned:
            raise Lfm2AutoregressiveHankelObservabilityCacheError(
                "The accumulator must be reopened from its last durable prefix."
            )
        if self._warm_identity is not None:
            path = _entry_path(self.root, self._warm_identity.aggregate_key_sha256)
            return _load_entry(path, self._warm_identity, cache_hit=True)
        if self.processed_source_count != len(self.plan.planned_phase_keys):
            raise Lfm2AutoregressiveHankelObservabilityCacheError(
                "Cannot finalize before every planned source shard is durably absorbed."
            )
        if (
            self._pooled_statistics is None
            or self._boundary_topology is None
            or self._future_lags is None
            or self._progress is None
        ):
            raise Lfm2AutoregressiveHankelObservabilityCacheError(
                "The completed source plan has no pooled statistics."
            )
        identity = _aggregate_identity(
            self.plan,
            source_descriptions=self._sources,
            boundary_topology=self._boundary_topology,
            future_lags=self._future_lags,
            pooled_statistics_receipt=self._progress["statistics_receipt"],
        )
        path = _entry_path(self.root, identity.aggregate_key_sha256)
        if path.exists():
            loaded = _load_entry(path, identity, cache_hit=True)
            self._warm_identity = identity
            self._pooled_statistics = None
            return loaded
        raw_result = _solver.build_lfm2_autoregressive_hankel_observability(
            (self._pooled_statistics,),
            boundary_names=identity.boundary_names,
            state_ridge_fraction=identity.state_ridge_fraction,
            future_ridge_fraction=identity.future_ridge_fraction,
            maximum_condition_number=identity.maximum_condition_number,
        )
        result = _rebind_pooled_solver_result(
            raw_result,
            identity,
            self._pooled_statistics,
        )
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir() or parent.is_symlink():
            raise Lfm2AutoregressiveHankelObservabilityCacheError(
                f"Aggregate observability fanout is unsafe: {parent}."
            )
        temporary = parent / f".{identity.aggregate_key_sha256}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            _write_entry(temporary, identity, result)
            try:
                temporary.rename(path)
            except FileExistsError:
                pass
            except OSError:
                if not path.exists():
                    raise
            if temporary.exists():
                shutil.rmtree(temporary)
            loaded = _load_entry(path, identity, cache_hit=False)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self._warm_identity = identity
        self._pooled_statistics = None
        return loaded


def open_lfm2_autoregressive_hankel_observability_accumulator(
    cache_root: str | os.PathLike[str],
    *,
    planned_phase_keys: Sequence[str],
    boundary_names: Sequence[str],
    state_ridge_fraction: float = 1e-8,
    future_ridge_fraction: float = 1e-8,
    maximum_condition_number: float = 1e10,
) -> Lfm2AutoregressiveHankelObservabilityAccumulator:
    """Open or resume a bounded-memory ordered observability accumulator."""

    root = _cache_root(cache_root)
    plan = describe_lfm2_autoregressive_hankel_observability_plan(
        planned_phase_keys,
        boundary_names=boundary_names,
        state_ridge_fraction=state_ridge_fraction,
        future_ridge_fraction=future_ridge_fraction,
        maximum_condition_number=maximum_condition_number,
    )
    return Lfm2AutoregressiveHankelObservabilityAccumulator(root, plan)


def load_or_build_lfm2_autoregressive_hankel_observability(
    cache_root: str | os.PathLike[str],
    *,
    planned_phase_keys: Sequence[str],
    shards: Iterable[_phase.AuthenticatedLfm2AutoregressiveHankelShard],
    boundary_names: Sequence[str],
    state_ridge_fraction: float = 1e-8,
    future_ridge_fraction: float = 1e-8,
    maximum_condition_number: float = 1e10,
) -> AuthenticatedLfm2AutoregressiveHankelObservability:
    """Stream remaining shards, then load or publish the final aggregate.

    This convenience wrapper never materializes ``shards`` as a sequence.  A
    production runner that must avoid even loading an already-completed shard
    should open the accumulator directly, query ``next_expected_phase_key``,
    construct exactly that one shard, call ``absorb``, and repeat.
    """

    if isinstance(shards, (str, bytes)) or not isinstance(shards, Iterable):
        raise Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError(
            "shards must be a one-at-a-time iterable."
        )
    accumulator = open_lfm2_autoregressive_hankel_observability_accumulator(
        cache_root,
        planned_phase_keys=planned_phase_keys,
        boundary_names=boundary_names,
        state_ridge_fraction=state_ridge_fraction,
        future_ridge_fraction=future_ridge_fraction,
        maximum_condition_number=maximum_condition_number,
    )
    for shard in shards:
        accumulator.absorb(shard)
    return accumulator.finalize()


__all__ = [
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_IDENTITY_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_PROVENANCE_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_RECEIPT_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_SAFETENSORS_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CACHE_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_CURRENT_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_NUMERIC_ABI_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PLAN_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROGRESS_SAFETENSORS_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROGRESS_SCHEMA",
    "AuthenticatedLfm2AutoregressiveHankelObservability",
    "Lfm2AutoregressiveHankelObservabilityAccumulator",
    "Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError",
    "Lfm2AutoregressiveHankelObservabilityCacheError",
    "Lfm2AutoregressiveHankelObservabilityCacheIdentity",
    "Lfm2AutoregressiveHankelObservabilityPlan",
    "describe_lfm2_autoregressive_hankel_observability_plan",
    "load_or_build_lfm2_autoregressive_hankel_observability",
    "open_lfm2_autoregressive_hankel_observability_accumulator",
]
