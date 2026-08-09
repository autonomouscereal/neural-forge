"""Immutable per-batch cache for autoregressive Hankel sufficient statistics.

The cache is intentionally below any whole-corpus aggregate.  One immutable
shard represents one already-authenticated trace-cache member batch and one
exact sparse future-lag schedule.  Publishing after every batch means an
interruption can discard at most the batch currently being accumulated, while
base evidence remains reusable when independent extension members are added.

Cache hits use only authenticated tensor-free session metadata.  They never
load a parent trace tensor and never call the trajectory extractor or moment
accumulator.  Cache misses call those two existing functions exactly once,
then persist the seven raw, uncentered moments as CPU SafeTensors.  There are no
model or tokenizer objects, logits, forwards, gradients, optimizers, training,
pruning, or quantization in this module.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge import hf_lfm2_autoregressive_hankel_statistics as _statistics
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import (
    SELECTIVE_BATCH_PAYLOAD_SCHEMA,
)


LFM2_AUTOREGRESSIVE_HANKEL_PHASE_CACHE_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-phase-cache-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_PHASE_IDENTITY_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-phase-identity-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_PHASE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-phase-receipt-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_PHASE_SAFETENSORS_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-phase-safetensors-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_NUMERIC_ABI_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-numeric-abi-v1"
)

_RECEIPT_FILENAME = "receipt.json"
_TENSOR_FILENAME = "statistics.safetensors"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TENSOR_NAMES = (
    "future_second_moment",
    "future_sum",
    "lag_pair_counts",
    "right_censor_counts",
    "state_future_cross",
    "state_second_moment",
    "state_sum",
)
_FLOAT_TENSOR_NAMES = frozenset(
    {
        "future_second_moment",
        "future_sum",
        "state_future_cross",
        "state_second_moment",
        "state_sum",
    }
)
_COUNT_TENSOR_NAMES = frozenset({"lag_pair_counts", "right_censor_counts"})
_ZERO_EXECUTION_AUDIT = {
    "model_objects_constructed": 0,
    "tokenizer_objects_constructed": 0,
    "parent_forward_calls": 0,
    "child_forward_calls": 0,
    "generate_calls": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "network_accesses": 0,
    "logit_tensors_loaded": 0,
    "gradients_used": False,
}


class Lfm2AutoregressiveHankelPhaseCacheError(RuntimeError):
    """Base error for an immutable Hankel sufficient-statistics shard."""


class Lfm2AutoregressiveHankelPhaseAuthenticationError(
    Lfm2AutoregressiveHankelPhaseCacheError
):
    """A source identity or persisted shard failed closed authentication."""


@dataclass(frozen=True, slots=True)
class Lfm2AutoregressiveHankelPhaseIdentity:
    """Canonical identity of one independently reusable member-batch shard."""

    body: Mapping[str, Any]
    phase_key_sha256: str
    session: Any
    reference: Any
    future_lags: tuple[int, ...]
    boundary_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2AutoregressiveHankelShard:
    """A byte-authenticated persisted sufficient-statistics shard."""

    statistics: _statistics.Lfm2AutoregressiveHankelStatistics
    receipt: Mapping[str, Any]
    path: Path
    phase_key_sha256: str
    cache_hit: bool


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"{name} must be a nonnegative integer."
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
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"{name} is not canonical JSON data."
        ) from exc


def _sealed_body(value: Mapping[str, Any], seal_name: str, name: str) -> dict[str, Any]:
    document = _json_copy(value, name)
    if not isinstance(document, dict):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"{name} must be a JSON object."
        )
    sealed = document.pop(seal_name, None)
    if _digest(sealed, f"{name}.{seal_name}") != canonical_sha256(document):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"{name} self-seal changed."
        )
    return document


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
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
    _file_identity(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"Cache document is unreadable: {path}."
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"Cache document is not exact canonical JSON: {path}."
        )
    return value


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _module_source_sha256(module: Any, name: str) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"{name} has no auditable source file."
        )
    path = Path(source).resolve()
    if not path.is_file() or path.is_symlink():
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"{name} source file is absent or unsafe."
        )
    return _file_sha256(path)


def _numeric_abi() -> dict[str, Any]:
    try:
        import safetensors
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Persistent LFM2 Hankel statistics require the 'hf' extra."
        ) from exc
    config = torch.__config__.show()
    return {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_NUMERIC_ABI_SCHEMA,
        "arithmetic_device": "cpu",
        "accumulation_dtype": "float64",
        "count_dtype": "int64",
        "tensor_layout": "strided-contiguous",
        "moment_algorithm": "raw-uncentered-lag-separable-einsum-v1",
        "terminal_policy": "right-censor-no-zero-padding",
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "torch_build_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
        "safetensors_version": str(safetensors.__version__),
    }


def _same_reference(left: Any, right: Any) -> bool:
    return (
        getattr(left, "index", None) == getattr(right, "index", None)
        and getattr(left, "input_sha256", None) == getattr(right, "input_sha256", None)
        and getattr(left, "record_sha256", None)
        == getattr(right, "record_sha256", None)
        and tuple(getattr(left, "ordered_tensor_names", ()))
        == tuple(getattr(right, "ordered_tensor_names", ()))
        and _json_copy(getattr(left, "record", None), "left reference record")
        == _json_copy(getattr(right, "record", None), "right reference record")
    )


def _normalize_member_source(session: Any, reference: Any) -> tuple[Any, Any]:
    """Map a merged coordinate back to the member that owns its bytes."""

    merged_index = _nonnegative_int(
        getattr(reference, "index", None), "reference index"
    )
    evidence_set = getattr(session, "evidence_set", None)
    if evidence_set is not None:
        try:
            provenance = evidence_set.batch(merged_index)
            members = evidence_set.members
            member = members[int(provenance.member_ordinal)]
            member_session = member.session
            member_index = int(provenance.member_batch_index)
            member_reference = member_session.batch_references[member_index]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                "Merged evidence cannot be resolved to its authenticated member."
            ) from exc
        if member_session is None or (
            getattr(provenance, "input_sha256", None)
            != getattr(reference, "input_sha256", None)
            or getattr(provenance, "input_sha256", None)
            != getattr(member_reference, "input_sha256", None)
        ):
            raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                "Merged evidence provenance differs from its member reference."
            )
        session = member_session
        reference = member_reference

    if getattr(session, "closed", False):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The selective authenticated session is closed."
        )
    references = tuple(getattr(session, "batch_references", ()))
    index = _nonnegative_int(getattr(reference, "index", None), "member batch index")
    if index >= len(references) or not _same_reference(reference, references[index]):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The reference is not bound to the authenticated member session."
        )
    return session, references[index]


def _source_description(session: Any, reference: Any) -> dict[str, Any]:
    manifest = _json_copy(getattr(session, "manifest", None), "trace manifest")
    if not isinstance(manifest, dict):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The authenticated session has no trace manifest."
        )
    parent = manifest.get("parent")
    topology = manifest.get("layer_topology")
    corpus = manifest.get("corpus")
    implementation = manifest.get("implementation")
    if not all(
        isinstance(value, Mapping)
        for value in (parent, topology, corpus, implementation)
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The trace manifest lacks parent, topology, corpus, or implementation."
        )
    parent_sha256 = _digest(
        parent.get("identity_sha256"), "trace parent identity_sha256"
    )
    layer_topology_sha256 = _digest(
        topology.get("layer_topology_sha256"),
        "trace layer_topology_sha256",
    )
    corpus_sha256 = _digest(corpus.get("corpus_sha256"), "trace corpus_sha256")
    trace_implementation_sha256 = _digest(
        implementation.get("implementation_sha256"),
        "trace implementation_sha256",
    )
    cache_key = _digest(getattr(session, "cache_key", None), "member cache_key")
    session_sha256 = _digest(
        getattr(session, "evidence_sha256", None), "member session_sha256"
    )

    batch_index = _nonnegative_int(
        getattr(reference, "index", None), "member batch index"
    )
    input_sha256 = _digest(
        getattr(reference, "input_sha256", None), "trace input_sha256"
    )
    record_sha256 = _digest(
        getattr(reference, "record_sha256", None), "trace record_sha256"
    )
    record = _json_copy(getattr(reference, "record", None), "trace batch record")
    if (
        not isinstance(record, dict)
        or record.get("cache_key") not in (None, cache_key)
        or record.get("batch_index") not in (None, batch_index)
        or record.get("input_sha256") not in (None, input_sha256)
        or record.get("record_sha256") not in (None, record_sha256)
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The trace record differs from its authenticated reference."
        )
    capture = record.get("capture")
    inventory = record.get("tensors")
    if not isinstance(capture, Mapping) or not isinstance(inventory, Mapping):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The trace record lacks capture metadata or a tensor inventory."
        )
    inventory_names = tuple(getattr(reference, "ordered_tensor_names", ()))
    if inventory_names != tuple(sorted(inventory)):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The trace reference tensor inventory changed."
        )

    decode_steps = _nonnegative_int(capture.get("decode_steps"), "decode_steps")
    layers = _statistics._layer_indices(inventory_names)
    phases = _statistics._phase_names(decode_steps)
    selected_names = _statistics._required_tensor_names(
        phases=phases,
        layer_indices=layers,
    )
    if any(name not in inventory for name in selected_names):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "The trace inventory lacks a required Hankel endpoint."
        )
    logical_bytes = 0
    hidden_size: int | None = None
    for name in selected_names:
        descriptor = inventory[name]
        if not isinstance(descriptor, Mapping):
            raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                f"Trace tensor descriptor {name!r} is malformed."
            )
        size_bytes = descriptor.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                f"Trace tensor descriptor {name!r} has invalid size."
            )
        logical_bytes += size_bytes
        if name.endswith("residual.pre") or name.endswith("residual.post_layer"):
            shape = descriptor.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or isinstance(shape[2], bool)
                or not isinstance(shape[2], int)
                or shape[2] <= 0
            ):
                raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                    f"Trace endpoint {name!r} has invalid hidden geometry."
                )
            hidden_size = shape[2] if hidden_size is None else hidden_size
            if shape[2] != hidden_size:
                raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                    "Trace residual endpoint hidden widths disagree."
                )

    selection_body = {
        "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
        "session_sha256": session_sha256,
        "cache_key": cache_key,
        "batch_index": batch_index,
        "input_sha256": input_sha256,
        "record_sha256": record_sha256,
        "ordered_tensor_names": list(selected_names),
        "ordered_tensor_descriptors": [
            dict(inventory[name]) for name in selected_names
        ],
        "logical_bytes": logical_bytes,
    }
    selected_payload_sha256 = canonical_sha256(selection_body)
    boundary_names = (
        "layer.000.residual.pre",
        *(f"layer.{layer:03d}.residual.post_layer" for layer in layers),
    )
    boundary_topology_body = {
        "schema": "neural-forge-lfm2-ahb-boundary-topology-v1",
        "trace_layer_topology_sha256": layer_topology_sha256,
        "layer_indices": list(layers),
        "boundary_names": list(boundary_names),
        "boundary_count": len(boundary_names),
        "hidden_size": hidden_size,
    }
    boundary_topology_sha256 = canonical_sha256(boundary_topology_body)
    member_body = {
        "schema": "neural-forge-lfm2-ahb-evidence-member-v1",
        "parent_identity_sha256": parent_sha256,
        "member_cache_key": cache_key,
        "member_corpus_sha256": corpus_sha256,
        "trace_implementation_sha256": trace_implementation_sha256,
        "trace_layer_topology_sha256": layer_topology_sha256,
    }
    member_identity_sha256 = canonical_sha256(member_body)
    return {
        "parent_identity_sha256": parent_sha256,
        "member_cache_key": cache_key,
        "member_corpus_sha256": corpus_sha256,
        "member_identity_sha256": member_identity_sha256,
        "member_session_sha256": session_sha256,
        "member_batch_index": batch_index,
        "trace_input_sha256": input_sha256,
        "trace_record_sha256": record_sha256,
        "selected_payload_sha256": selected_payload_sha256,
        "selected_tensor_names": list(selected_names),
        "selected_tensor_count": len(selected_names),
        "trace_implementation_sha256": trace_implementation_sha256,
        "boundary_topology": boundary_topology_body,
        "boundary_topology_sha256": boundary_topology_sha256,
        "boundary_names": boundary_names,
    }


def describe_lfm2_autoregressive_hankel_phase(
    *,
    session: Any,
    reference: Any,
    future_lags: Sequence[int],
) -> Lfm2AutoregressiveHankelPhaseIdentity:
    """Describe a shard from authenticated metadata without loading tensors."""

    member_session, member_reference = _normalize_member_source(session, reference)
    lags = _statistics._resolve_future_lags(
        max_horizon=None,
        future_lags=future_lags,
    )
    source = _source_description(member_session, member_reference)
    numeric_abi = _numeric_abi()
    body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_PHASE_IDENTITY_SCHEMA,
        "parent_identity_sha256": source["parent_identity_sha256"],
        "member_cache_key": source["member_cache_key"],
        "member_corpus_sha256": source["member_corpus_sha256"],
        "member_identity_sha256": source["member_identity_sha256"],
        "member_session_sha256": source["member_session_sha256"],
        "member_batch_index": source["member_batch_index"],
        "trace_input_sha256": source["trace_input_sha256"],
        "trace_record_sha256": source["trace_record_sha256"],
        "selected_payload_sha256": source["selected_payload_sha256"],
        "selected_tensor_names": source["selected_tensor_names"],
        "selected_tensor_count": source["selected_tensor_count"],
        "trace_implementation_sha256": source["trace_implementation_sha256"],
        "future_lags": list(lags),
        "future_lags_sha256": canonical_sha256(
            {"schema": "neural-forge-lfm2-ahb-future-lags-v1", "lags": list(lags)}
        ),
        "boundary_topology": source["boundary_topology"],
        "boundary_topology_sha256": source["boundary_topology_sha256"],
        "numeric_abi": numeric_abi,
        "numeric_abi_sha256": canonical_sha256(numeric_abi),
        "statistics_source_sha256": _module_source_sha256(
            _statistics,
            "autoregressive Hankel statistics implementation",
        ),
        "phase_cache_source_sha256": _module_source_sha256(
            sys.modules[__name__],
            "autoregressive Hankel phase-cache implementation",
        ),
        "checkpoint_granularity_batches": 1,
        "raw_uncentered_moments": True,
        "merged_evidence_identity_in_key": False,
        "member_shards_independently_reusable": True,
    }
    frozen = MappingProxyType(_json_copy(body, "Hankel phase identity"))
    return Lfm2AutoregressiveHankelPhaseIdentity(
        body=frozen,
        phase_key_sha256=canonical_sha256(body),
        session=member_session,
        reference=member_reference,
        future_lags=lags,
        boundary_names=tuple(source["boundary_names"]),
    )


def _entry_path(root: Path, phase_key_sha256: str) -> Path:
    return root / "shards" / phase_key_sha256[:2] / phase_key_sha256


def _cache_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise Lfm2AutoregressiveHankelPhaseCacheError(
            f"Hankel phase cache root is absent or unsafe: {root}."
        )
    shard_root = root / "shards"
    shard_root.mkdir(exist_ok=True)
    if not shard_root.is_dir() or shard_root.is_symlink():
        raise Lfm2AutoregressiveHankelPhaseCacheError(
            f"Hankel phase shard root is absent or unsafe: {shard_root}."
        )
    return root


def _statistics_tensors(
    statistics: _statistics.Lfm2AutoregressiveHankelStatistics,
) -> dict[str, Any]:
    import torch

    tensors = {name: getattr(statistics, name) for name in _TENSOR_NAMES}
    for name, tensor in tensors.items():
        expected_dtype = torch.float64 if name in _FLOAT_TENSOR_NAMES else torch.int64
        if (
            not torch.is_tensor(tensor)
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.dtype != expected_dtype
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or (
                name in _FLOAT_TENSOR_NAMES
                and not bool(torch.isfinite(tensor).all().item())
            )
        ):
            raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                f"Hankel statistic {name!r} violated the exact CPU dtype contract."
            )
    return tensors


def _statistics_from_tensors(
    tensors: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> _statistics.Lfm2AutoregressiveHankelStatistics:
    return _statistics.Lfm2AutoregressiveHankelStatistics(
        lag_pair_counts=tensors["lag_pair_counts"],
        right_censor_counts=tensors["right_censor_counts"],
        state_sum=tensors["state_sum"],
        future_sum=tensors["future_sum"],
        state_second_moment=tensors["state_second_moment"],
        future_second_moment=tensors["future_second_moment"],
        state_future_cross=tensors["state_future_cross"],
        receipt=MappingProxyType(_json_copy(receipt, "statistics receipt")),
    )


def _safetensors_metadata(
    identity: Lfm2AutoregressiveHankelPhaseIdentity,
    statistics_receipt: Mapping[str, Any],
) -> dict[str, str]:
    payload = statistics_receipt["tensor_payload"]
    return {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_PHASE_SAFETENSORS_SCHEMA,
        "phase_key_sha256": identity.phase_key_sha256,
        "statistics_receipt_sha256": str(statistics_receipt["receipt_sha256"]),
        "tensor_payload_sha256": str(payload["payload_sha256"]),
        "boundary_topology_sha256": str(identity.body["boundary_topology_sha256"]),
        "future_lags_sha256": str(identity.body["future_lags_sha256"]),
    }


def _validate_statistics_receipt(
    tensors: Mapping[str, Any],
    statistics_receipt: Mapping[str, Any],
    identity: Lfm2AutoregressiveHankelPhaseIdentity,
) -> None:
    body = _sealed_body(
        statistics_receipt,
        "receipt_sha256",
        "Hankel statistics receipt",
    )
    if (
        body.get("schema") != _statistics.LFM2_AUTOREGRESSIVE_HANKEL_STATISTICS_SCHEMA
        or body.get("source_receipt_sha256") is None
        or body.get("future_lags") != list(identity.future_lags)
        or body.get("future_lag_count") != len(identity.future_lags)
        or body.get("boundary_count") != len(identity.boundary_names)
        or body.get("hidden_size") != identity.body["boundary_topology"]["hidden_size"]
        or body.get("arithmetic_device") != "cpu"
        or body.get("accumulation_dtype") != "float64"
        or body.get("terminal_rows_zero_padded") != 0
        or body.get("contains_nonzero_future_lags") is not True
        or body.get("raw_uncentered_moments", True) is not True
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel statistics receipt differs from its phase identity."
        )
    actual_payload = _statistics._tensor_payload_receipt(tensors)
    if body.get("tensor_payload") != actual_payload:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel statistics tensor descriptors changed."
        )
    audit = body.get("operation_audit")
    if not isinstance(audit, Mapping) or any(
        value not in (0, False) for value in audit.values()
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel statistics receipt reports model execution or mutation."
        )


def _write_shard(
    temporary: Path,
    identity: Lfm2AutoregressiveHankelPhaseIdentity,
    statistics: _statistics.Lfm2AutoregressiveHankelStatistics,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Persistent LFM2 Hankel statistics require safetensors."
        ) from exc
    tensors = _statistics_tensors(statistics)
    statistics_receipt = _json_copy(
        statistics.receipt,
        "cold Hankel statistics receipt",
    )
    _validate_statistics_receipt(tensors, statistics_receipt, identity)
    tensor_path = temporary / _TENSOR_FILENAME
    save_file(
        {name: tensors[name] for name in _TENSOR_NAMES},
        tensor_path,
        metadata=_safetensors_metadata(identity, statistics_receipt),
    )
    with tensor_path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    tensor_size = tensor_path.stat().st_size
    tensor_sha256 = _file_sha256(tensor_path)
    receipt_body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_PHASE_RECEIPT_SCHEMA,
        "cache_schema": LFM2_AUTOREGRESSIVE_HANKEL_PHASE_CACHE_SCHEMA,
        "phase_key": dict(identity.body),
        "phase_key_sha256": identity.phase_key_sha256,
        "source_batch_receipt_sha256": statistics_receipt["source_receipt_sha256"],
        "boundary_names": list(identity.boundary_names),
        "raw_uncentered_moments": True,
        "globally_center_only_after_aggregate": True,
        "checkpoint_granularity_batches": 1,
        "tensor_file": {
            "filename": _TENSOR_FILENAME,
            "size_bytes": tensor_size,
            "file_sha256": tensor_sha256,
        },
        "statistics_receipt": statistics_receipt,
        "operation_audit": dict(_ZERO_EXECUTION_AUDIT),
        "immutable_by_contract": True,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    _write_canonical_json(temporary / _RECEIPT_FILENAME, receipt)


def _load_shard(
    path: Path,
    identity: Lfm2AutoregressiveHankelPhaseIdentity,
    *,
    cache_hit: bool,
) -> AuthenticatedLfm2AutoregressiveHankelShard:
    if not path.is_dir() or path.is_symlink():
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            f"Hankel phase shard is absent or unsafe: {path}."
        )
    if {item.name for item in path.iterdir()} != {
        _RECEIPT_FILENAME,
        _TENSOR_FILENAME,
    }:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel phase shard file inventory changed."
        )
    receipt = _read_canonical_json(path / _RECEIPT_FILENAME)
    receipt_body = _sealed_body(
        receipt,
        "receipt_sha256",
        "Hankel phase receipt",
    )
    if (
        receipt_body.get("schema") != LFM2_AUTOREGRESSIVE_HANKEL_PHASE_RECEIPT_SCHEMA
        or receipt_body.get("cache_schema")
        != LFM2_AUTOREGRESSIVE_HANKEL_PHASE_CACHE_SCHEMA
        or receipt_body.get("phase_key") != dict(identity.body)
        or receipt_body.get("phase_key_sha256") != identity.phase_key_sha256
        or receipt_body.get("boundary_names") != list(identity.boundary_names)
        or receipt_body.get("raw_uncentered_moments") is not True
        or receipt_body.get("globally_center_only_after_aggregate") is not True
        or receipt_body.get("checkpoint_granularity_batches") != 1
        or receipt_body.get("operation_audit") != _ZERO_EXECUTION_AUDIT
        or receipt_body.get("immutable_by_contract") is not True
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel phase receipt differs from its requested identity."
        )
    tensor_file = receipt_body.get("tensor_file")
    if (
        not isinstance(tensor_file, Mapping)
        or tensor_file.get("filename") != _TENSOR_FILENAME
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel phase tensor-file receipt is malformed."
        )
    tensor_path = path / _TENSOR_FILENAME
    before = _file_identity(tensor_path)
    if (
        tensor_file.get("size_bytes") != before[2]
        or _file_sha256(tensor_path) != tensor_file.get("file_sha256")
        or _file_identity(tensor_path) != before
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel phase SafeTensors bytes changed."
        )

    statistics_receipt = receipt_body.get("statistics_receipt")
    if not isinstance(statistics_receipt, Mapping):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel statistics receipt is absent."
        )
    expected_metadata = _safetensors_metadata(identity, statistics_receipt)
    try:
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if handle.metadata() != expected_metadata:
                raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                    "Hankel phase SafeTensors metadata changed."
                )
            if tuple(sorted(handle.keys())) != _TENSOR_NAMES:
                raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
                    "Hankel phase SafeTensors tensor inventory changed."
                )
            tensors = {
                name: handle.get_tensor(name)
                .detach()
                .to(device="cpu")
                .contiguous()
                .clone()
                for name in _TENSOR_NAMES
            }
    except Lfm2AutoregressiveHankelPhaseAuthenticationError:
        raise
    except Exception as exc:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel phase SafeTensors could not be authenticated."
        ) from exc
    if _file_identity(tensor_path) != before:
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel phase SafeTensors changed while it was loaded."
        )
    _statistics_tensors(_statistics_from_tensors(tensors, statistics_receipt))
    _validate_statistics_receipt(tensors, statistics_receipt, identity)
    if receipt_body.get("source_batch_receipt_sha256") != statistics_receipt.get(
        "source_receipt_sha256"
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Hankel phase source receipt binding changed."
        )
    statistics = _statistics_from_tensors(tensors, statistics_receipt)
    return AuthenticatedLfm2AutoregressiveHankelShard(
        statistics=statistics,
        receipt=MappingProxyType(receipt),
        path=path,
        phase_key_sha256=identity.phase_key_sha256,
        cache_hit=cache_hit,
    )


def load_or_build_lfm2_autoregressive_hankel_statistics(
    cache_root: str | os.PathLike[str],
    *,
    session: Any,
    reference: Any,
    future_lags: Sequence[int],
) -> AuthenticatedLfm2AutoregressiveHankelShard:
    """Load or checkpoint one member-batch's exact raw Hankel moments.

    The public surface deliberately has no model, tokenizer, logits, device,
    or optimizer argument.  ``session`` may be a member selective session or a
    merged evidence session; merged coordinates are normalized to the owning
    member before keying so adding another extension cannot invalidate prior
    base shards.
    """

    root = _cache_root(cache_root)
    identity = describe_lfm2_autoregressive_hankel_phase(
        session=session,
        reference=reference,
        future_lags=future_lags,
    )
    path = _entry_path(root, identity.phase_key_sha256)
    if path.exists():
        return _load_shard(path, identity, cache_hit=True)

    batch = _statistics.extract_lfm2_autoregressive_hankel_batch(
        session=identity.session,
        reference=identity.reference,
    )
    if (
        batch.receipt.get("trace_cache_key") != identity.body["member_cache_key"]
        or batch.receipt.get("trace_session_sha256")
        != identity.body["member_session_sha256"]
        or batch.receipt.get("trace_input_sha256")
        != identity.body["trace_input_sha256"]
        or batch.receipt.get("trace_record_sha256")
        != identity.body["trace_record_sha256"]
        or batch.receipt.get("selected_payload_sha256")
        != identity.body["selected_payload_sha256"]
        or batch.receipt.get("boundary_names") != list(identity.boundary_names)
    ):
        raise Lfm2AutoregressiveHankelPhaseAuthenticationError(
            "Extracted Hankel batch differs from its precomputed phase identity."
        )
    statistics = _statistics.accumulate_lfm2_ahb_trajectory(
        batch.boundary_rows,
        batch.behavior_rows,
        future_lags=identity.future_lags,
        source_receipt_sha256=batch.receipt["receipt_sha256"],
    )

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise Lfm2AutoregressiveHankelPhaseCacheError(
            f"Hankel shard fanout directory is unsafe: {parent}."
        )
    temporary = parent / f".{identity.phase_key_sha256}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_shard(temporary, identity, statistics)
        try:
            temporary.rename(path)
        except FileExistsError:
            # Another producer published the identical immutable batch first.
            pass
        except OSError:
            if not path.exists():
                raise
        if temporary.exists():
            shutil.rmtree(temporary)
        return _load_shard(path, identity, cache_hit=False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "LFM2_AUTOREGRESSIVE_HANKEL_NUMERIC_ABI_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_PHASE_CACHE_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_PHASE_IDENTITY_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_PHASE_RECEIPT_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_PHASE_SAFETENSORS_SCHEMA",
    "AuthenticatedLfm2AutoregressiveHankelShard",
    "Lfm2AutoregressiveHankelPhaseAuthenticationError",
    "Lfm2AutoregressiveHankelPhaseCacheError",
    "Lfm2AutoregressiveHankelPhaseIdentity",
    "describe_lfm2_autoregressive_hankel_phase",
    "load_or_build_lfm2_autoregressive_hankel_statistics",
]
