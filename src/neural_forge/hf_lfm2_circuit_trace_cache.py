"""Immutable circuit traces for a loaded Transformers LFM2 causal parent.

The cache is deliberately architecture-specific.  It validates every decoder
layer with :mod:`neural_forge.hf_lfm2_canonical` and captures one uncached
full-prefix prefill.  The default mode consumes caller-supplied forced tokens
through true one-token cached forwards and verifies every cached logit row
against a fresh ``use_cache=False`` full-prefix forward.  The explicit
``full_prefix_canonical`` mode instead captures every forced-decode trace from
the uncached full prefix alone; it never executes or accepts cached logits.

Attention layers retain sampled Q/K/V, pre-O context, O delta, and residual
endpoints.  The opt-in causal-circuit evidence contract additionally retains
pre-RoPE normalized Q/K, rotary terms, exact causal QK scores/probabilities,
and the complete K/V history seen by each sampled query.  Short-convolution
layers retain their actual ``B, C, x`` gated FIR endpoints under a separate
namespace.  The same opt-in contract adds the exact FIR kernel, pre-update
state, and functionally reconstructed filtered Bx path.  They are never
exposed as Q/K/V.  Feed-forward hooks retain gate, up, exact post-SwiGLU,
down, and residual endpoints for every layer.  The opt-in contract also seals
the final RMSNorm input and output.

The parent, ordered inputs, implementation sources, capture controls, and exact
layer topology are content-addressed.  A completed cache is never rewritten.
Opening or reusing a modified artifact fails closed.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import stat
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_canonical import (
    Lfm2AttentionContract,
    Lfm2HeterogeneousLayerContract,
    Lfm2HeterogeneousShortConvContract,
    Lfm2LayerContract,
    Lfm2ShortConvContract,
    inspect_lfm2_heterogeneous_layer,
    inspect_lfm2_layer,
)
from neural_forge.hf_lfm2_circuit_trace_cache_compatibility import (
    permits_explicit_implementation_compatibility,
)
from neural_forge.numeric_execution_policy import (
    NumericExecutionPolicy,
    NumericExecutionPolicyError,
    NumericExecutionRequest,
    resolve_numeric_execution_policy,
)


CACHE_SCHEMA = "neural-forge-lfm2-parent-circuit-trace-cache-v1"
BATCH_SCHEMA = "neural-forge-lfm2-parent-circuit-trace-batch-v1"
COMPLETION_SCHEMA = "neural-forge-lfm2-parent-circuit-trace-completion-v1"
CHILD_REPLAY_SCHEMA = "neural-forge-lfm2-child-circuit-replay-v2"
CHILD_REPLAY_BATCH_SCHEMA = "neural-forge-lfm2-child-circuit-replay-batch-v2"
CHILD_EVIDENCE_PROFILE_SCHEMA = "neural-forge-lfm2-child-evidence-profile-v1"
CHILD_EMBEDDING_REPLAY_SCHEMA = "neural-forge-lfm2-child-embedding-circuit-replay-v1"
AUTHENTICATED_SESSION_SCHEMA = (
    "neural-forge-lfm2-circuit-trace-authenticated-session-v1"
)
SELECTIVE_AUTHENTICATED_SESSION_SCHEMA = (
    "neural-forge-lfm2-circuit-trace-selective-session-v1"
)
SELECTIVE_BATCH_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-circuit-trace-selective-batch-payload-v1"
)
MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "complete.json"
TRACE_FILENAME = "trace.safetensors"
BATCH_RECORD_FILENAME = "record.json"
CACHED_VERIFIED_TRACE_MODE = "cached_verified"
FULL_PREFIX_CANONICAL_TRACE_MODE = "full_prefix_canonical"
CHILD_REPLAY_PREFIX_BY_PREFIX_MODE = "prefix_by_prefix_canonical"
CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE = "single_pass_causal"
CHILD_REPLAY_LOGIT_ERROR_MODEL = "strict-allclose-or-ieee-dot-product-forward-error-v1"
CHILD_REPLAY_NON_LOGIT_ERROR_MODEL = (
    "strict-allclose-or-recorded-ieee-cross-kernel-path-forward-error-v1"
)
CHILD_REPLAY_NUMERIC_POLICY_SCHEMA = (
    "neural-forge-lfm2-child-replay-numeric-certification-policy-v1"
)
CHILD_REPLAY_NON_LOGIT_ENVELOPE_SCHEMA = (
    "neural-forge-lfm2-child-replay-non-logit-envelope-policy-v1"
)


Lfm2RuntimeLayerContract = Lfm2LayerContract | Lfm2HeterogeneousLayerContract
Lfm2RuntimeShortConvContract = (
    Lfm2ShortConvContract | Lfm2HeterogeneousShortConvContract
)

_FORCED_IDS = "forced_decode_ids"
_ALLOWED_BATCH_FIELDS = frozenset(
    {"input_ids", "attention_mask", "position_ids", _FORCED_IDS}
)
_SHA256_LENGTH = 64
_SINGLE_PASS_CAUSAL_RUNTIME_ALLOWLIST = frozenset(
    {
        (
            "5.12.1",
            "2ec8be8af6b65482f54d29e52da9765b863ce37b65846423dfbfda72f944f150",
            "aa525d800f824b25cbe9542ca7cc5b35e948e7cfa3178e2c0f2356a15ff32df2",
        ),
        (
            "5.13.0",
            "766e026caefee6e985470d92b36b597cec5938e183d9afc414bf28ac38100277",
            "ad7b312bc3830894a198760520cc1baf02e35d327d216d41f00b0c61e9604334",
        ),
    }
)
_HETEROGENEOUS_SINGLE_PASS_MATERIALIZER_SOURCE_ALLOWLIST = frozenset(
    {
        "ef757359d9376fd66fb1c7578ab1409c93fa8d2c64c54f07cee49bfb87fa7927",
    }
)
_TORCH_LINEAR_FORWARD_RUNTIME_ALLOWLIST = frozenset(
    {
        (
            "2.11.0+cu128",
            "cpython-312",
            "0da76648c1aaff74b782e6e5a81694e0f6f94c5e4972f921ece130f4df8fe32b",
            "45aa464d373ca22cc2f063e25df64ed78f37bbf3ba4203e8220bd545920c45dd",
        ),
        (
            "2.13.0+cpu",
            "cpython-311",
            "be8f64fc618ac59af36edcd1124aec19ff29402b5dfb1792e51346a84f3e8f34",
            "5559b5aa67c9baa896ecbba6ff8abd71abaa77ad23cfdf283c5ba6001c4dbf1f",
        ),
    }
)

# Sealed metadata is unaffected by any of the following: they only change when
# already determined work happens, never what is recorded.
_TENSOR_HASH_WORKER_CEILING = 32
_TENSOR_HASH_WORKER_VARIABLE = "NEURAL_FORGE_LFM2_TRACE_HASH_WORKERS"
# Fan-out is only profitable once there are enough bytes to amortize the
# dispatch, so small inventories keep the original serial pass.  Buffers are
# prepared in bounded windows so hashing a multi-gigabyte parent never holds
# every host copy resident at once.
_TENSOR_HASH_MINIMUM_PARALLEL_BYTES = 8 * 1024 * 1024
_TENSOR_HASH_CHUNK_TARGET_BYTES = 2 * 1024 * 1024
_TENSOR_HASH_WINDOW_BYTES = 256 * 1024 * 1024
_VERIFIED_BATCH_GUARD_LIMIT = 256
_CAPTURE_TIMING_SCHEMA = "neural-forge-lfm2-capture-timing-v1"
_SEAL_PIPELINE_ENABLED = True

_tensor_hash_executor_lock = threading.Lock()
_tensor_hash_executor: concurrent.futures.ThreadPoolExecutor | None = None


class Lfm2CircuitTraceCacheError(RuntimeError):
    """Base error for LFM2 circuit-trace capture and persistence."""


class Lfm2CircuitTraceCaptureError(Lfm2CircuitTraceCacheError):
    """The parent or requested trace could not be captured safely."""


class Lfm2CircuitTraceCorruptionError(Lfm2CircuitTraceCacheError):
    """An immutable committed cache failed strict verification."""


class Lfm2CircuitTraceConflictError(Lfm2CircuitTraceCacheError):
    """Existing immutable contents conflict with the requested identity."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "LFM2 circuit tracing requires the 'hf' optional dependencies."
        ) from exc
    return torch


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "LFM2 circuit tracing requires the 'hf' optional dependencies."
        ) from exc
    return safe_open, save_file


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2CircuitTraceCacheError(
            "Circuit-trace metadata must be canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cannot hash cache file {path}."
        ) from exc
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache metadata cannot be a symlink: {path}."
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cannot parse cache metadata {path}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache metadata {path} must contain an object."
        )
    if raw != _canonical_json(value) + b"\n":
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache metadata {path} is not canonical or was modified."
        )
    return value


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(dict(value)) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise Lfm2CircuitTraceConflictError(
            f"Immutable cache metadata already exists at {path}."
        ) from exc
    except OSError as exc:
        raise Lfm2CircuitTraceCacheError(
            f"Could not publish cache metadata {path}."
        ) from exc


def _sealed_document(
    body: Mapping[str, Any],
    seal_name: str,
) -> dict[str, Any]:
    if seal_name in body:
        raise Lfm2CircuitTraceCacheError(
            f"Document body already contains seal {seal_name!r}."
        )
    result = dict(body)
    result[seal_name] = _json_sha256(result)
    return result


def _verify_seal(
    value: Mapping[str, Any],
    seal_name: str,
    document_name: str,
) -> None:
    seal = value.get(seal_name)
    if not isinstance(seal, str) or len(seal) != _SHA256_LENGTH:
        raise Lfm2CircuitTraceCorruptionError(
            f"{document_name} has an invalid {seal_name}."
        )
    body = dict(value)
    body.pop(seal_name, None)
    if _json_sha256(body) != seal:
        raise Lfm2CircuitTraceCorruptionError(
            f"{document_name} self-hash does not match its contents."
        )


def _tensor_hash_inputs(torch: Any, value: Any) -> tuple[bytes, Any, Any]:
    """Split one tensor digest into its header and its own byte buffer.

    The header is the exact byte sequence the serial collector fed to
    ``digest.update`` before the payload, concatenated: SHA-256 is a stream,
    so one update of the joined header hashes identically to five.  The buffer
    aliases the tensor's storage instead of materializing an intermediate
    ``bytes`` copy, so the returned tensor must be retained alongside it.

    Splitting the work this way also keeps every Torch dispatch on the calling
    thread.  Worker threads then run pure :mod:`hashlib`, which releases the
    GIL and therefore actually scales.
    """

    tensor = value.detach().to(device="cpu").contiguous()
    header = b"".join(
        (
            b"neural-forge-lfm2-circuit-tensor-v1\0",
            str(value.dtype).replace("torch.", "").encode("ascii"),
            b"\0",
            _canonical_json([int(size) for size in value.shape]),
            b"\0",
        )
    )
    return header, tensor, tensor.reshape(-1).view(torch.uint8).numpy()


def _tensor_bytes(torch: Any, value: Any) -> bytes:
    """Materialize one tensor's contiguous bytes.

    Digest paths hash the zero-copy buffer directly.  This accessor is part of
    the reader surface fingerprinted by the global-topology phase cache, so it
    is kept as the single explicit place that pays for a byte copy.
    """

    _header, _tensor, buffer = _tensor_hash_inputs(torch, value)
    return buffer.tobytes()


def _sha256_from_parts(header: bytes, buffer: Any) -> str:
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(buffer)
    return digest.hexdigest()


def _tensor_sha256(torch: Any, value: Any) -> str:
    header, _tensor, buffer = _tensor_hash_inputs(torch, value)
    return _sha256_from_parts(header, buffer)


def _tensor_hash_worker_count() -> int:
    """Bound the shared hashing pool without ever changing what is hashed."""

    configured = os.environ.get(_TENSOR_HASH_WORKER_VARIABLE, "")
    if configured.strip().isdigit():
        requested = int(configured.strip())
        if requested > 0:
            return min(requested, _TENSOR_HASH_WORKER_CEILING)
    return max(1, min(_TENSOR_HASH_WORKER_CEILING, os.cpu_count() or 1))


def _tensor_hash_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the process-wide bounded pool used for tensor SHA-256 work.

    ``hashlib`` releases the GIL for the buffer sizes captured here, so the
    pool converts one serial inventory pass into a parallel one.  Nothing
    submitted to this pool may itself wait on the pool.
    """

    global _tensor_hash_executor
    with _tensor_hash_executor_lock:
        if _tensor_hash_executor is None:
            _tensor_hash_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_tensor_hash_worker_count(),
                thread_name_prefix="lfm2-trace-hash",
            )
        return _tensor_hash_executor


def _hash_prepared_chunk(prepared: Sequence[tuple[bytes, Any, Any]]) -> list[str]:
    return [_sha256_from_parts(header, buffer) for header, _tensor, buffer in prepared]


def _byte_balanced_chunks(
    sizes: Sequence[int],
    chunk_count: int,
) -> list[tuple[int, int]]:
    """Split an ordered inventory into contiguous, roughly equal-byte ranges."""

    total = sum(sizes)
    target = total / chunk_count
    chunks: list[tuple[int, int]] = []
    start = 0
    running = 0
    for index, size in enumerate(sizes):
        running += size
        if len(chunks) < chunk_count - 1 and running >= target * (len(chunks) + 1):
            chunks.append((start, index + 1))
            start = index + 1
    if start < len(sizes):
        chunks.append((start, len(sizes)))
    return chunks


def _hash_prepared_window(prepared: Sequence[tuple[bytes, Any, Any]]) -> list[str]:
    """Hash one prepared window, fanning out only when the bytes justify it."""

    sizes = [int(buffer.nbytes) for _header, _tensor, buffer in prepared]
    total = sum(sizes)
    workers = _tensor_hash_worker_count()
    if (
        workers < 2
        or len(prepared) < 2
        or total < _TENSOR_HASH_MINIMUM_PARALLEL_BYTES
    ):
        return _hash_prepared_chunk(prepared)
    chunk_count = max(
        2,
        min(workers, len(prepared), total // _TENSOR_HASH_CHUNK_TARGET_BYTES),
    )
    chunks = _byte_balanced_chunks(sizes, chunk_count)
    if len(chunks) < 2:
        return _hash_prepared_chunk(prepared)
    pool = _tensor_hash_pool()
    futures = [
        pool.submit(_hash_prepared_chunk, prepared[start:stop])
        for start, stop in chunks
    ]
    digests: list[str] = []
    for future in futures:
        digests.extend(future.result())
    return digests


def _parallel_tensor_sha256(torch: Any, values: Sequence[Any]) -> tuple[str, ...]:
    """Hash tensors concurrently and return the digests in the caller's order.

    Buffers are prepared on the calling thread in bounded byte windows, then
    each window's pure hashing is handed to the pool in contiguous,
    byte-balanced chunks.  Windowing caps how many host copies are resident at
    once when a device-resident parent is fingerprinted; chunking keeps the
    dispatch cost far below the digest it parallelizes.  Small inventories
    never leave the calling thread.  Every path produces the same digests, in
    the same order, as the serial pass.
    """

    if len(values) < 2:
        return tuple(_tensor_sha256(torch, value) for value in values)
    digests: list[str] = []
    window: list[tuple[bytes, Any, Any]] = []
    window_bytes = 0
    for value in values:
        prepared = _tensor_hash_inputs(torch, value)
        window.append(prepared)
        window_bytes += int(prepared[2].nbytes)
        if window_bytes >= _TENSOR_HASH_WINDOW_BYTES:
            digests.extend(_hash_prepared_window(window))
            window = []
            window_bytes = 0
    if window:
        digests.extend(_hash_prepared_window(window))
    return tuple(digests)


def _tensor_descriptor(
    torch: Any,
    value: Any,
    *,
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype).replace("torch.", ""),
        "shape": [int(size) for size in value.shape],
        "numel": int(value.numel()),
        "size_bytes": int(value.numel() * value.element_size()),
        "sha256": _tensor_sha256(torch, value) if sha256 is None else sha256,
    }


def _ordered_tensor_descriptors(
    torch: Any,
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one name-sorted inventory whose digests are computed in parallel.

    The result is byte-identical to the serial
    ``{name: _tensor_descriptor(torch, value) for name, value in sorted(...)}``
    comprehension it replaces; only the digest computation is concurrent, and
    the futures are collected in submission order.
    """

    names = sorted(tensors)
    digests = _parallel_tensor_sha256(torch, [tensors[name] for name in names])
    return {
        name: _tensor_descriptor(torch, tensors[name], sha256=digest)
        for name, digest in zip(names, digests)
    }


def _validated_capture_source(torch: Any, value: Any, name: str) -> Any:
    """Reject anything the capture contract forbids before any host copy."""

    if not torch.is_tensor(value):
        raise Lfm2CircuitTraceCaptureError(f"{name} must be a tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise Lfm2CircuitTraceCaptureError(
            f"{name} must be a materialized dense tensor."
        )
    return value.detach()


def _validate_captured_host_tensor(torch: Any, result: Any, name: str) -> None:
    """Apply the fail-closed gradient and finiteness contract to one endpoint.

    This is the exact check the serial collector ran inline.  Splitting it out
    lets a staged capture defer it to a batched pass without weakening it: a
    non-finite endpoint still aborts the capture with
    :class:`Lfm2CircuitTraceCaptureError` and the same message before anything
    is validated, retained, hashed, or sealed.
    """

    if result.requires_grad or result.grad_fn is not None:
        raise Lfm2CircuitTraceCaptureError(f"{name} retained a gradient graph.")
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise Lfm2CircuitTraceCaptureError(f"{name} contains NaN or infinity.")


def _cpu_tensor(
    torch: Any,
    value: Any,
    name: str,
) -> Any:
    source = _validated_capture_source(torch, value, name)
    result = source.to(device="cpu").contiguous().clone()
    _validate_captured_host_tensor(torch, result, name)
    return result


def _sampled_sequence_view(
    torch: Any,
    value: Any,
    name: str,
    *,
    sequence_axis: int,
    sample_last_tokens: int,
) -> Any:
    """Narrow one endpoint to its retained trailing tokens without copying."""

    if not torch.is_tensor(value) or value.ndim <= sequence_axis:
        raise Lfm2CircuitTraceCaptureError(
            f"{name} does not expose sequence axis {sequence_axis}."
        )
    sequence_length = int(value.shape[sequence_axis])
    if sequence_length <= 0:
        raise Lfm2CircuitTraceCaptureError(f"{name} has an empty sequence axis.")
    count = min(sequence_length, sample_last_tokens)
    return value.narrow(sequence_axis, sequence_length - count, count)


def _sample_sequence_tensor(
    torch: Any,
    value: Any,
    name: str,
    *,
    sequence_axis: int,
    sample_last_tokens: int,
) -> Any:
    sampled = _sampled_sequence_view(
        torch,
        value,
        name,
        sequence_axis=sequence_axis,
        sample_last_tokens=sample_last_tokens,
    )
    return _cpu_tensor(torch, sampled, name)


@dataclass(slots=True)
class _PendingHostTensor:
    """One staged device-to-host copy awaiting its batched finiteness verdict."""

    name: str
    tensor: Any = None
    pinned: Any = None
    source: Any = None
    shape: tuple[int, ...] = ()


class _HostTransferStage:
    """Batch capture host copies and their fail-closed finiteness proofs.

    The serial collector issued one blocking ``.to(device="cpu")`` and one
    ``torch.isfinite(...).item()`` per captured endpoint.  Against an
    accelerator that is a full round trip per tensor, and one capture point
    emits tens of thousands of them.  This stage changes only *when* that work
    happens.  A CUDA source is copied into pooled pinned host memory with
    ``non_blocking=True`` and the whole phase is landed by a single device
    synchronization followed by one host-side validation pass in submission
    order.  A CPU source never touches the pinned pool: it is cloned exactly
    as before and validated in that same deferred pass, so a CPU-only run
    records byte-identical tensors.
    """

    __slots__ = (
        "torch",
        "_device",
        "_direct_transfers",
        "_outstanding_device_copies",
        "_pending",
        "_pinned_pool",
        "_staged_transfers",
        "_synchronizations",
    )

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self._pending: list[_PendingHostTensor] = []
        self._pinned_pool: dict[tuple[Any, int], list[Any]] = {}
        self._device: Any | None = None
        self._outstanding_device_copies = 0
        self._synchronizations = 0
        self._staged_transfers = 0
        self._direct_transfers = 0

    @property
    def counters(self) -> dict[str, int]:
        return {
            "device_synchronizations": self._synchronizations,
            "staged_host_transfers": self._staged_transfers,
            "direct_host_transfers": self._direct_transfers,
        }

    def reset(self) -> None:
        """Drop any copies staged by an aborted phase; keep the pinned pool."""

        for item in self._pending:
            self._recycle(item)
        self._pending = []
        self._outstanding_device_copies = 0

    def _acquire_pinned(self, dtype: Any, numel: int) -> Any:
        free = self._pinned_pool.get((dtype, numel))
        if free:
            return free.pop()
        return self.torch.empty(numel, dtype=dtype, pin_memory=True)

    def _recycle(self, item: _PendingHostTensor) -> None:
        if item.pinned is not None:
            key = (item.pinned.dtype, int(item.pinned.numel()))
            self._pinned_pool.setdefault(key, []).append(item.pinned)
            item.pinned = None
        item.source = None

    def stage(self, value: Any, name: str) -> _PendingHostTensor:
        torch = self.torch
        source = _validated_capture_source(torch, value, name)
        if source.device.type != "cuda" or int(source.numel()) <= 0:
            self._direct_transfers += 1
            item = _PendingHostTensor(
                name=name,
                tensor=source.to(device="cpu").contiguous().clone(),
            )
            self._pending.append(item)
            return item
        contiguous = source.contiguous()
        shape = tuple(int(size) for size in contiguous.shape)
        pinned = self._acquire_pinned(contiguous.dtype, int(contiguous.numel()))
        pinned.view(shape).copy_(contiguous, non_blocking=True)
        if self._device is None:
            self._device = contiguous.device
        self._outstanding_device_copies += 1
        self._staged_transfers += 1
        # ``contiguous`` is retained until the copy lands so the stream-ordered
        # device allocation backing it cannot be recycled underneath the copy.
        item = _PendingHostTensor(
            name=name,
            pinned=pinned,
            source=contiguous,
            shape=shape,
        )
        self._pending.append(item)
        return item

    def resolve(self) -> None:
        """Land every staged copy, then validate the batch in staged order."""

        torch = self.torch
        pending = self._pending
        if not pending:
            return
        self._pending = []
        if self._outstanding_device_copies:
            torch.cuda.synchronize(self._device)
            self._synchronizations += 1
            self._outstanding_device_copies = 0
        for item in pending:
            if item.tensor is None:
                staged = item.pinned.view(item.shape)
                result = torch.empty(item.shape, dtype=staged.dtype)
                result.copy_(staged)
                item.tensor = result
                self._recycle(item)
            _validate_captured_host_tensor(torch, item.tensor, item.name)


def _materialize_staged_endpoints(target: dict[str, Any]) -> None:
    """Replace resolved staging handles with the tensors they now hold."""

    for name, value in list(target.items()):
        if isinstance(value, _PendingHostTensor):
            target[name] = value.tensor


@dataclass(slots=True)
class _CaptureTelemetry:
    """Non-sealed wall-clock accounting for one capture invocation.

    This never enters a manifest, record, or sealed document.  It is returned
    beside the capture so an operator can see where a capture point spent its
    time without perturbing any content-addressed identity.
    """

    replay_seconds: float = 0.0
    hash_and_seal_seconds: float = 0.0
    fingerprint_seconds: float = 0.0
    device_synchronizations: int = 0
    staged_host_transfers: int = 0
    direct_host_transfers: int = 0
    sealed_tensor_count: int = 0
    batches: list[dict[str, Any]] = field(default_factory=list)

    def absorb_stage(self, counters: Mapping[str, int]) -> None:
        self.device_synchronizations += int(counters["device_synchronizations"])
        self.staged_host_transfers += int(counters["staged_host_transfers"])
        self.direct_host_transfers += int(counters["direct_host_transfers"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _CAPTURE_TIMING_SCHEMA,
            "sealed_in_metadata": False,
            "batch_count": len(self.batches),
            "replay_seconds": round(self.replay_seconds, 6),
            "hash_and_seal_seconds": round(self.hash_and_seal_seconds, 6),
            "fingerprint_seconds": round(self.fingerprint_seconds, 6),
            "device_synchronizations": self.device_synchronizations,
            "staged_host_transfers": self.staged_host_transfers,
            "direct_host_transfers": self.direct_host_transfers,
            "sealed_tensor_count": self.sealed_tensor_count,
            "tensor_hash_workers": _tensor_hash_worker_count(),
            "seal_pipeline_enabled": bool(_SEAL_PIPELINE_ENABLED),
            "per_batch": [dict(entry) for entry in self.batches],
        }


def _qualified_class(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_file_sha256(path: Path, name: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise Lfm2CircuitTraceCaptureError(
            f"{name} source is not a regular local file: {path}."
        )
    return _file_sha256(path)


def implementation_identity() -> dict[str, Any]:
    """Hash the collector, canonical contract, and installed LFM2 runtime."""

    try:
        import neural_forge.hf_lfm2_canonical as canonical
        from transformers.models.lfm2 import modeling_lfm2
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "LFM2 circuit tracing requires Transformers LFM2 support."
        ) from exc

    collector_path = Path(__file__).resolve()
    canonical_path = Path(canonical.__file__).resolve()
    runtime_path = Path(modeling_lfm2.__file__).resolve()
    body = {
        "schema": "neural-forge-lfm2-circuit-implementation-v1",
        "collector_source_sha256": _source_file_sha256(collector_path, "collector"),
        "canonical_contract_source_sha256": _source_file_sha256(
            canonical_path, "canonical LFM2 contract"
        ),
        "transformers_lfm2_source_sha256": _source_file_sha256(
            runtime_path, "Transformers LFM2"
        ),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "safetensors_version": _package_version("safetensors"),
    }
    return {**body, "implementation_sha256": _json_sha256(body)}


def _implementation_identity_is_internally_consistent(
    value: Mapping[str, Any],
) -> bool:
    body = dict(value)
    implementation_sha256 = body.pop("implementation_sha256", None)
    if (
        not isinstance(implementation_sha256, str)
        or len(implementation_sha256) != _SHA256_LENGTH
    ):
        return False
    try:
        return _json_sha256(body) == implementation_sha256
    except Lfm2CircuitTraceCacheError:
        return False


def _compatible_implementation_identity(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Accept identical identities or one exact directional policy receipt."""

    if not _implementation_identity_is_internally_consistent(
        stored
    ) or not _implementation_identity_is_internally_consistent(current):
        return False
    if stored.get("implementation_sha256") == current.get("implementation_sha256"):
        return True
    try:
        actual_current_source_sha256 = _source_file_sha256(
            Path(__file__).resolve(),
            "collector",
        )
    except Lfm2CircuitTraceCacheError:
        return False
    if current.get("collector_source_sha256") != actual_current_source_sha256:
        return False
    return permits_explicit_implementation_compatibility(stored, current)


@dataclass(frozen=True, slots=True)
class Lfm2ParentFingerprint:
    """Hash-bound identity of the loaded LFM2 parent."""

    model_class: str
    config_sha256: str
    state_dict_sha256: str
    state_tensor_count: int
    runtime_buffer_sha256: str
    runtime_buffer_count: int
    identity_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_class": self.model_class,
            "config_sha256": self.config_sha256,
            "state_dict_sha256": self.state_dict_sha256,
            "state_tensor_count": self.state_tensor_count,
            "runtime_buffer_sha256": self.runtime_buffer_sha256,
            "runtime_buffer_count": self.runtime_buffer_count,
            "identity_sha256": self.identity_sha256,
        }


def _model_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise Lfm2CircuitTraceCaptureError("The parent must expose config.to_dict().")
    try:
        value = to_dict()
    except Exception as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Could not serialize the parent config."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2CircuitTraceCaptureError(
            "Parent config.to_dict() must return a mapping."
        )
    result = dict(value)
    if result.get("model_type") != "lfm2":
        raise Lfm2CircuitTraceCaptureError("The parent config.model_type must be lfm2.")
    # Loading the same local bytes from a different directory rewrites this
    # provenance-only field.  It does not affect forward execution.
    result.pop("_name_or_path", None)
    # ``block_ff_dim`` is a serialized compatibility alias for the operative
    # ``intermediate_size`` field.  Transformers may either consume and drop
    # the alias or a strict local loader may restore it for byte-stable
    # save/reload.  Those representations describe the same instantiated
    # circuit, so exclude a redundant alias from the semantic identity while
    # still rejecting a contradictory value.
    if "block_ff_dim" in result:
        block_ff_dim = result.pop("block_ff_dim")
        intermediate_size = result.get("intermediate_size")
        if block_ff_dim is not None and (
            isinstance(block_ff_dim, bool)
            or not isinstance(block_ff_dim, int)
            or block_ff_dim <= 0
            or isinstance(intermediate_size, bool)
            or not isinstance(intermediate_size, int)
            or intermediate_size <= 0
            or block_ff_dim != intermediate_size
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Parent config.block_ff_dim must be absent, null, or equal to "
                "config.intermediate_size."
            )
    _canonical_json(result)
    return result


def fingerprint_lfm2_parent(model: Any) -> Lfm2ParentFingerprint:
    """Hash parameters, persistent state, runtime buffers, and configuration."""

    torch = _torch()
    config_sha256 = _json_sha256(_model_config(model))
    try:
        state = model.state_dict()
    except Exception as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Could not read the parent state dict."
        ) from exc
    if not isinstance(state, Mapping) or not state:
        raise Lfm2CircuitTraceCaptureError("The parent state dict must be nonempty.")
    # Validate in canonical order first, then hash the same ordered tensors
    # concurrently.  The digest feed order, and therefore the fingerprint, is
    # unchanged; only the per-parameter SHA-256 work is parallel.
    state_names = sorted(state)
    for name in state_names:
        value = state[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise Lfm2CircuitTraceCaptureError(
                "Parent state entries must be string-named tensors."
            )
        if value.layout != torch.strided or value.device.type == "meta":
            raise Lfm2CircuitTraceCaptureError(
                f"Parent state tensor {name!r} is not materialized and dense."
            )
    state_digests = _parallel_tensor_sha256(
        torch,
        [state[name] for name in state_names],
    )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-parent-state-v1\0")
    for name, tensor_sha256 in zip(state_names, state_digests):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256.encode("ascii"))
        digest.update(b"\0")
    state_sha256 = digest.hexdigest()
    try:
        buffers = tuple(model.named_buffers())
    except Exception as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Could not read the parent runtime buffers."
        ) from exc
    ordered_buffers = sorted(buffers, key=lambda item: item[0])
    for name, value in ordered_buffers:
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise Lfm2CircuitTraceCaptureError(
                "Parent runtime buffers must be string-named tensors."
            )
        if value.layout != torch.strided or value.device.type == "meta":
            raise Lfm2CircuitTraceCaptureError(
                f"Parent runtime buffer {name!r} is not materialized and dense."
            )
    buffer_digests = _parallel_tensor_sha256(
        torch,
        [value for _name, value in ordered_buffers],
    )
    buffer_digest = hashlib.sha256()
    buffer_digest.update(b"neural-forge-lfm2-parent-runtime-buffers-v1\0")
    for (name, _value), tensor_sha256 in zip(ordered_buffers, buffer_digests):
        buffer_digest.update(name.encode("utf-8"))
        buffer_digest.update(b"\0")
        buffer_digest.update(tensor_sha256.encode("ascii"))
        buffer_digest.update(b"\0")
    runtime_buffer_sha256 = buffer_digest.hexdigest()
    body = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": _qualified_class(model),
        "config_sha256": config_sha256,
        "state_dict_sha256": state_sha256,
        "state_tensor_count": len(state),
        "runtime_buffer_sha256": runtime_buffer_sha256,
        "runtime_buffer_count": len(buffers),
    }
    return Lfm2ParentFingerprint(
        model_class=body["model_class"],
        config_sha256=config_sha256,
        state_dict_sha256=state_sha256,
        state_tensor_count=len(state),
        runtime_buffer_sha256=runtime_buffer_sha256,
        runtime_buffer_count=len(buffers),
        identity_sha256=_json_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CircuitTraceCaptureConfig:
    """Bounded trace selection and cached/full-prefix verification controls."""

    sample_last_tokens: int = 8
    max_decode_steps: int = 32
    max_capture_bytes: int = 512 * 1024 * 1024
    logit_rtol: float = 1e-5
    logit_atol: float = 1e-6
    require_top1_agreement: bool = True
    allow_top1_disagreement_within_numeric_error_envelope: bool = False
    decode_trace_mode: str = CACHED_VERIFIED_TRACE_MODE
    capture_causal_circuit_evidence: bool = False

    def __post_init__(self) -> None:
        for name in (
            "sample_last_tokens",
            "max_decode_steps",
            "max_capture_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("logit_rtol", "logit_atol"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a nonnegative finite float.")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a nonnegative finite float.") from exc
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be a nonnegative finite float.")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.require_top1_agreement, bool):
            raise TypeError("require_top1_agreement must be bool.")
        if not isinstance(
            self.allow_top1_disagreement_within_numeric_error_envelope,
            bool,
        ):
            raise TypeError(
                "allow_top1_disagreement_within_numeric_error_envelope must be bool."
            )
        if not isinstance(self.capture_causal_circuit_evidence, bool):
            raise TypeError("capture_causal_circuit_evidence must be bool.")
        if self.decode_trace_mode not in {
            CACHED_VERIFIED_TRACE_MODE,
            FULL_PREFIX_CANONICAL_TRACE_MODE,
        }:
            raise ValueError(
                "decode_trace_mode must be 'cached_verified' or "
                "'full_prefix_canonical'."
            )
        if (
            self.capture_causal_circuit_evidence
            and self.decode_trace_mode != CACHED_VERIFIED_TRACE_MODE
        ):
            raise ValueError(
                "Causal-circuit installability evidence requires "
                "decode_trace_mode='cached_verified' so full causal histories "
                "come from an exact cache whose logits are independently "
                "verified against use_cache=False full-prefix forwards."
            )

    def as_dict(self) -> dict[str, Any]:
        full_prefix_only = self.decode_trace_mode == FULL_PREFIX_CANONICAL_TRACE_MODE
        body = {
            "sample_last_tokens": self.sample_last_tokens,
            "max_decode_steps": self.max_decode_steps,
            "max_capture_bytes": self.max_capture_bytes,
            "logit_rtol": self.logit_rtol,
            "logit_atol": self.logit_atol,
            "require_top1_agreement": self.require_top1_agreement,
            "allow_top1_disagreement_within_numeric_error_envelope": (
                self.allow_top1_disagreement_within_numeric_error_envelope
            ),
            "inference_mode": True,
            "model_eval_mode": True,
            "gradients_used": False,
            "training_steps": 0,
            "prefill_use_cache": False,
            "cached_prefill_use_cache": not full_prefix_only,
            "forced_decode_use_cache": not full_prefix_only,
            "full_prefix_verification_use_cache": False,
        }
        if full_prefix_only:
            body.update(
                {
                    "decode_trace_mode": FULL_PREFIX_CANONICAL_TRACE_MODE,
                    "canonical_trace_source": "use_cache_false_full_prefix",
                    "cached_logits_accepted": False,
                }
            )
        if self.capture_causal_circuit_evidence:
            body.update(
                {
                    "capture_causal_circuit_evidence": True,
                    "attention_pre_rope_qk": True,
                    "attention_exact_causal_scores_and_probabilities": True,
                    "attention_complete_kv_history": True,
                    "short_conv_exact_fir_state_and_response": True,
                    "final_normalization_boundary": True,
                }
            )
        return {**body, "selection_sha256": _json_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2ChildEvidenceProfile:
    """Exact immutable tensor-retention contract for one child replay point.

    Hooks still collect every endpoint needed by the existing LFM2 algebra
    validators.  This profile controls only which already-validated tensors
    survive ``finish()`` and can therefore enter a replay batch descriptor.
    """

    profile_role: str
    layer_suffixes: tuple[tuple[int, tuple[str, ...]], ...]
    retain_logits: bool = False
    retain_lm_head_input: bool = False
    retain_phase_forced_token_ids: bool = False
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.profile_role, str) or not self.profile_role:
            raise ValueError("profile_role must be a nonempty string.")
        if not isinstance(self.layer_suffixes, tuple) or not self.layer_suffixes:
            raise ValueError("layer_suffixes must be a nonempty tuple.")
        previous_index = -1
        normalized: list[tuple[int, tuple[str, ...]]] = []
        for entry in self.layer_suffixes:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    "layer_suffixes entries must be (layer_index, suffixes) tuples."
                )
            layer_index, suffixes = entry
            if (
                isinstance(layer_index, bool)
                or not isinstance(layer_index, int)
                or layer_index <= previous_index
            ):
                raise ValueError(
                    "layer_suffixes indices must be nonnegative, unique, and ordered."
                )
            if (
                not isinstance(suffixes, tuple)
                or not suffixes
                or suffixes != tuple(sorted(set(suffixes)))
                or any(not isinstance(value, str) or not value for value in suffixes)
            ):
                raise ValueError(
                    "Each layer suffix inventory must be nonempty, unique, and ordered."
                )
            normalized.append((layer_index, suffixes))
            previous_index = layer_index
        for name in (
            "retain_logits",
            "retain_lm_head_input",
            "retain_phase_forced_token_ids",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        if self.retain_lm_head_input and not self.retain_logits:
            raise ValueError("LM-head input retention requires logits retention.")
        object.__setattr__(self, "layer_suffixes", tuple(normalized))
        body = self._body()
        expected = _json_sha256(body)
        if self.profile_sha256 not in {"", expected}:
            raise ValueError("Child evidence profile SHA-256 is invalid.")
        object.__setattr__(self, "profile_sha256", expected)

    @property
    def selected_layer_indices(self) -> tuple[int, ...]:
        return tuple(index for index, _suffixes in self.layer_suffixes)

    def suffixes_for_layer(self, layer_index: int) -> tuple[str, ...]:
        for candidate, suffixes in self.layer_suffixes:
            if candidate == layer_index:
                return suffixes
        raise Lfm2CircuitTraceCaptureError(
            f"Child evidence profile has no layer {layer_index}."
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": CHILD_EVIDENCE_PROFILE_SCHEMA,
            "profile_role": self.profile_role,
            "layers": [
                {
                    "layer_index": layer_index,
                    "retained_suffixes": list(suffixes),
                }
                for layer_index, suffixes in self.layer_suffixes
            ],
            "retain_logits": self.retain_logits,
            "retain_lm_head_input": self.retain_lm_head_input,
            "retain_phase_forced_token_ids": (
                self.retain_phase_forced_token_ids
            ),
            "full_layer_algebra_validation_before_filter": True,
            "inputs_always_retained": True,
            "phase_order_and_rows_always_retained": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "profile_sha256": self.profile_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Lfm2ChildEvidenceProfile:
        if not isinstance(value, Mapping):
            raise TypeError("Child evidence profile must be a mapping.")
        layers = value.get("layers")
        if not isinstance(layers, list):
            raise ValueError("Child evidence profile has no layer inventory.")
        try:
            profile = cls(
                profile_role=value["profile_role"],
                layer_suffixes=tuple(
                    (
                        entry["layer_index"],
                        tuple(entry["retained_suffixes"]),
                    )
                    for entry in layers
                ),
                retain_logits=value["retain_logits"],
                retain_lm_head_input=value["retain_lm_head_input"],
                retain_phase_forced_token_ids=value[
                    "retain_phase_forced_token_ids"
                ],
                profile_sha256=value["profile_sha256"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Child evidence profile body is malformed.") from exc
        if profile.as_dict() != dict(value):
            raise ValueError("Child evidence profile body or seal changed.")
        return profile


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    index: int
    input_ids: Any
    attention_mask: Any
    position_ids: Any | None
    forced_decode_ids: Any
    input_sha256: str


def _is_integer_dtype(torch: Any, value: Any) -> bool:
    return value.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _prepare_batch(
    torch: Any,
    value: Mapping[str, Any],
    index: int,
    config: Lfm2CircuitTraceCaptureConfig,
) -> _PreparedBatch:
    if not isinstance(value, Mapping):
        raise Lfm2CircuitTraceCaptureError(f"Batch {index} must be a mapping.")
    unexpected = sorted(set(value).difference(_ALLOWED_BATCH_FIELDS))
    if unexpected:
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} has unsupported or reserved fields: {unexpected}."
        )
    if "input_ids" not in value or _FORCED_IDS not in value:
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} must contain input_ids and {_FORCED_IDS}."
        )
    input_ids = _cpu_tensor(torch, value["input_ids"], f"batch {index} input_ids")
    forced = _cpu_tensor(torch, value[_FORCED_IDS], f"batch {index} {_FORCED_IDS}")
    if (
        input_ids.ndim != 2
        or forced.ndim != 2
        or min(int(size) for size in input_ids.shape) <= 0
        or min(int(size) for size in forced.shape) <= 0
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} input_ids and {_FORCED_IDS} must be nonempty rank-2 tensors."
        )
    if int(input_ids.shape[0]) != 1:
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} must contain exactly one unpadded sequence."
        )
    if int(input_ids.shape[1]) < 3:
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} prompt must contain at least three tokens."
        )
    if not _is_integer_dtype(torch, input_ids) or not _is_integer_dtype(torch, forced):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} token IDs must use integer dtypes."
        )
    if int(input_ids.shape[0]) != int(forced.shape[0]):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} prompt and forced decode batch sizes differ."
        )
    if int(forced.shape[1]) > config.max_decode_steps:
        raise Lfm2CircuitTraceCaptureError(f"Batch {index} exceeds max_decode_steps.")
    if bool((input_ids < 0).any().item()) or bool((forced < 0).any().item()):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} token IDs must be nonnegative."
        )

    attention_value = value.get("attention_mask")
    attention_mask = (
        torch.ones_like(input_ids, dtype=torch.long)
        if attention_value is None
        else _cpu_tensor(torch, attention_value, f"batch {index} attention_mask")
    )
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} attention_mask must match input_ids."
        )
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} attention_mask must be binary."
        )
    if not bool((attention_mask == 1).all().item()):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {index} must be unpadded; attention_mask must be all ones."
        )

    position_value = value.get("position_ids")
    position_ids = (
        None
        if position_value is None
        else _cpu_tensor(torch, position_value, f"batch {index} position_ids")
    )
    if position_ids is not None:
        if (
            tuple(position_ids.shape) != tuple(input_ids.shape)
            or not _is_integer_dtype(torch, position_ids)
            or bool((position_ids < 0).any().item())
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Batch {index} position_ids must be nonnegative integer IDs matching input_ids."
            )

    named = {
        "attention_mask": attention_mask,
        _FORCED_IDS: forced,
        "input_ids": input_ids,
    }
    if position_ids is not None:
        named["position_ids"] = position_ids
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-circuit-input-v1\0")
    for name in sorted(named):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(torch, named[name]).encode("ascii"))
        digest.update(b"\0")
    return _PreparedBatch(
        index=index,
        input_ids=input_ids.to(torch.long),
        attention_mask=attention_mask.to(torch.long),
        position_ids=(None if position_ids is None else position_ids.to(torch.long)),
        forced_decode_ids=forced.to(torch.long),
        input_sha256=digest.hexdigest(),
    )


def _prepare_batches(
    torch: Any,
    values: Iterable[Mapping[str, Any]],
    config: Lfm2CircuitTraceCaptureConfig,
) -> tuple[_PreparedBatch, ...]:
    try:
        prepared = tuple(
            _prepare_batch(torch, value, index, config)
            for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise Lfm2CircuitTraceCaptureError(
            "calibration_batches must be iterable."
        ) from exc
    if not prepared:
        raise Lfm2CircuitTraceCaptureError(
            "At least one calibration batch is required."
        )
    return prepared


def _decoder_layers(model: Any) -> tuple[Any, ...]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    ]
    for value in candidates:
        if value is None or isinstance(value, (str, bytes)):
            continue
        try:
            layers = tuple(value)
        except TypeError:
            continue
        if layers:
            return layers
    raise Lfm2CircuitTraceCaptureError(
        "Could not locate the LFM2 decoder layer container."
    )


def _layer_topology(
    model: Any,
) -> tuple[
    tuple[Any, ...],
    tuple[Lfm2RuntimeLayerContract, ...],
    dict[str, Any],
]:
    layers = _decoder_layers(model)
    contracts: list[Lfm2RuntimeLayerContract] = []
    entries: list[dict[str, Any]] = []
    config = getattr(model, "config", None)
    heterogeneous_layout = None
    if config is not None and hasattr(
        config,
        "neural_forge_dynamic_blank_config_schema",
    ):
        try:
            from .hf_lfm2_dynamic_blank_materializer import (
                extract_lfm2_dynamic_physical_layout_from_config,
            )
            from .hf_lfm2_dynamic_physical_layout import (
                Lfm2HeterogeneousPhysicalLayoutDescriptor,
            )

            recovered_layout = extract_lfm2_dynamic_physical_layout_from_config(config)
        except Exception as exc:
            raise Lfm2CircuitTraceCaptureError(
                "Dynamic LFM2 child layout authentication failed before replay."
            ) from exc
        if type(recovered_layout) is not Lfm2HeterogeneousPhysicalLayoutDescriptor:
            raise Lfm2CircuitTraceCaptureError(
                "Child replay currently accepts only the sealed heterogeneous "
                "dynamic LFM2 layout."
            )
        heterogeneous_layout = recovered_layout
    for index, layer in enumerate(layers):
        try:
            contract = (
                inspect_lfm2_layer(layer)
                if heterogeneous_layout is None
                else inspect_lfm2_heterogeneous_layer(
                    layer,
                    heterogeneous_layout,
                )
            )
        except Exception as exc:
            raise Lfm2CircuitTraceCaptureError(
                f"LFM2 layer {index} failed canonical topology validation."
            ) from exc
        if contract.layer_index != index:
            raise Lfm2CircuitTraceCaptureError(
                "LFM2 layer container order disagrees with operator layer_idx."
            )
        contracts.append(contract)
        operator = layer.self_attn if contract.kind == "full_attention" else layer.conv
        entry = {
            "container_index": index,
            "layer_class": _qualified_class(layer),
            "operator_class": _qualified_class(operator),
            "feed_forward_class": _qualified_class(layer.feed_forward),
            "contract": contract.as_dict(),
        }
        if contract.kind == "short_conv":
            operator_dict = entry["contract"]["operator"]
            if (
                operator_dict.get("input_projection_chunk_order") != ["b", "c", "x"]
                or operator_dict.get("not_attention_qkv") is not True
            ):
                raise Lfm2CircuitTraceCaptureError(
                    "Short-conv topology did not certify B/C/x semantics."
                )
        entries.append(entry)
    body = {
        "schema": "neural-forge-lfm2-layer-topology-v1",
        "layer_count": len(entries),
        "layers": entries,
    }
    return (
        layers,
        tuple(contracts),
        {
            **body,
            "layer_topology_sha256": _json_sha256(body),
        },
    )


def _resolve_input_device(torch: Any, model: Any, explicit: Any | None) -> Any:
    if explicit is not None:
        return torch.device(explicit)
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if torch.is_tensor(weight) and weight.device.type != "meta":
            return weight.device
    for value in model.parameters():
        if value.device.type != "meta":
            return value.device
    raise Lfm2CircuitTraceCaptureError(
        "Could not resolve a materialized parent input device."
    )


def _corpus_identity(prepared: Sequence[_PreparedBatch]) -> dict[str, Any]:
    body = {
        "schema": "neural-forge-lfm2-circuit-corpus-v1",
        "batch_count": len(prepared),
        "ordered_batch_input_sha256": [value.input_sha256 for value in prepared],
        "prompt_token_counts": [int(value.input_ids.numel()) for value in prepared],
        "forced_decode_token_counts": [
            int(value.forced_decode_ids.numel()) for value in prepared
        ],
    }
    return {**body, "corpus_sha256": _json_sha256(body)}


def _selected_layer_indices(
    layer_count: int,
    selected: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return a unique, canonical layer selection without changing execution."""

    if selected is None:
        return tuple(range(layer_count))
    if isinstance(selected, (str, bytes)):
        raise TypeError("selected_layer_indices must be a sequence of integers.")
    try:
        values = tuple(selected)
    except TypeError as exc:
        raise TypeError(
            "selected_layer_indices must be a sequence of integers."
        ) from exc
    if not values:
        raise ValueError("selected_layer_indices must not be empty.")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("selected_layer_indices must contain only integers.")
        if value < 0 or value >= layer_count:
            raise IndexError(
                f"selected layer index must be in [0, {layer_count}), got {value}."
            )
    if len(set(values)) != len(values):
        raise ValueError("selected_layer_indices must not contain duplicates.")
    return tuple(sorted(values))


def _manifest_for(
    parent: Lfm2ParentFingerprint,
    corpus: Mapping[str, Any],
    implementation: Mapping[str, Any],
    topology: Mapping[str, Any],
    config: Lfm2CircuitTraceCaptureConfig,
    device: Any,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    capture = config.as_dict()
    runtime = {
        "input_device": str(device),
        "network_access_performed": False,
        "local_loaded_parent_only": True,
    }
    identity = {
        "schema": CACHE_SCHEMA,
        "parent_identity_sha256": parent.identity_sha256,
        "corpus_sha256": corpus["corpus_sha256"],
        "implementation_sha256": implementation["implementation_sha256"],
        "layer_topology_sha256": topology["layer_topology_sha256"],
        "selection_sha256": capture["selection_sha256"],
        "execution_sha256": execution["execution_sha256"],
        "runtime": runtime,
    }
    cache_key = _json_sha256(identity)
    body = {
        "schema": CACHE_SCHEMA,
        "cache_key": cache_key,
        "parent": parent.as_dict(),
        "corpus": dict(corpus),
        "implementation": dict(implementation),
        "layer_topology": dict(topology),
        "capture": capture,
        "runtime": runtime,
        "execution": dict(execution),
        "storage": {
            "format": "canonical-json-plus-safetensors",
            "content_addressed": True,
            "committed_cache_immutable": True,
            "committed_batches_immutable": True,
        },
    }
    return _sealed_document(body, "manifest_sha256")


def _execution_identity(model: Any, device: Any) -> dict[str, Any]:
    """Bind runtime branches that change LFM2 attention or short-conv math."""

    try:
        from transformers.models.lfm2 import modeling_lfm2
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "LFM2 circuit tracing requires Transformers LFM2 support."
        ) from exc
    config = getattr(model, "config", None)
    attention_backend = getattr(config, "_attn_implementation", None)
    if not isinstance(attention_backend, str) or not attention_backend:
        attention_backend = "eager"
    fast_conv_available = bool(getattr(modeling_lfm2, "is_fast_path_available", False))
    device_type = str(getattr(device, "type", device))
    heterogeneous_runtime = bool(
        config is not None
        and hasattr(config, "neural_forge_dynamic_blank_config_schema")
    )
    body = {
        "schema": "neural-forge-lfm2-circuit-execution-v1",
        "attention_backend": attention_backend,
        "fast_short_conv_available": fast_conv_available,
        "fast_short_conv_selected": bool(
            fast_conv_available and device_type == "cuda" and not heterogeneous_runtime
        ),
        "input_device_type": device_type,
    }
    if heterogeneous_runtime:
        body["sealed_heterogeneous_dynamic_runtime"] = True
    return {**body, "execution_sha256": _json_sha256(body)}


def _first_tensor(torch: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
    value = kwargs.get("hidden_states")
    if torch.is_tensor(value):
        return value
    for item in args:
        if torch.is_tensor(item):
            return item
    raise Lfm2CircuitTraceCaptureError(
        "Hooked LFM2 module did not receive hidden states."
    )


def _output_tensor(torch: Any, output: Any, name: str) -> Any:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise Lfm2CircuitTraceCaptureError(f"Hooked {name} did not return a tensor output.")


def _rotate_half(torch: Any, value: Any) -> Any:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _reconstruct_attention_causal_path(
    torch: Any,
    *,
    query: Any,
    key_history: Any,
    value_history: Any,
    additive_mask: Any | None,
    groups: int,
    sample_count: int,
    head_dim: int,
) -> dict[str, Any]:
    """Replay eager QK/softmax/V without changing execution devices."""

    operands = [query, key_history, value_history]
    if additive_mask is not None:
        operands.append(additive_mask)
    if any(not torch.is_tensor(value) for value in operands):
        raise Lfm2CircuitTraceCaptureError(
            "Attention causal replay operands must be tensors."
        )
    if len({str(value.device) for value in operands}) != 1:
        raise Lfm2CircuitTraceCaptureError(
            "Attention causal replay operands must remain on one execution device."
        )
    if (
        query.ndim != 4
        or key_history.ndim != 4
        or value_history.ndim != 4
        or tuple(key_history.shape) != tuple(value_history.shape)
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Attention causal replay received invalid Q/K/V geometry."
        )
    if (
        isinstance(groups, bool)
        or not isinstance(groups, int)
        or groups <= 0
        or int(key_history.shape[1]) * groups != int(query.shape[1])
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Attention causal replay received invalid grouped-query geometry."
        )
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or sample_count > int(query.shape[2])
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Attention causal replay received an invalid sample count."
        )
    if (
        isinstance(head_dim, bool)
        or not isinstance(head_dim, int)
        or head_dim <= 0
        or int(query.shape[3]) != head_dim
        or int(key_history.shape[3]) != head_dim
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Attention causal replay received an invalid head dimension."
        )

    repeated_key = key_history.repeat_interleave(groups, dim=1)
    repeated_value = value_history.repeat_interleave(groups, dim=1)
    sampled_query = query[:, :, -sample_count:, :]
    qk_scores = torch.matmul(
        sampled_query,
        repeated_key.transpose(2, 3),
    ) * float(head_dim**-0.5)
    history_count = int(key_history.shape[2])
    if additive_mask is None:
        sampled_mask = torch.zeros(
            (
                int(query.shape[0]),
                1,
                sample_count,
                history_count,
            ),
            dtype=qk_scores.dtype,
            device=qk_scores.device,
        )
    else:
        if additive_mask.ndim != 4:
            raise Lfm2CircuitTraceCaptureError(
                "Attention causal replay additive mask is not rank 4."
            )
        sampled_mask = additive_mask[
            ...,
            -sample_count:,
            :history_count,
        ].to(dtype=qk_scores.dtype)
        if (
            int(sampled_mask.shape[-2]) != sample_count
            or int(sampled_mask.shape[-1]) != history_count
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Attention causal replay additive mask does not cover the "
                "sampled queries and complete key history."
            )
    probabilities = torch.nn.functional.softmax(
        qk_scores + sampled_mask,
        dim=-1,
        dtype=torch.float32,
    ).to(sampled_query.dtype)
    reconstructed_context = torch.matmul(
        probabilities,
        repeated_value,
    )
    return {
        "qk_scores": qk_scores,
        "sampled_mask": sampled_mask,
        "probabilities": probabilities,
        "reconstructed_context": reconstructed_context,
    }


class _Lfm2TraceHooks:
    def __init__(
        self,
        torch: Any,
        model: Any,
        layers: Sequence[Any],
        contracts: Sequence[Lfm2RuntimeLayerContract],
        config: Lfm2CircuitTraceCaptureConfig,
        selected_layer_indices: Sequence[int] | None = None,
        *,
        capture_replay_lm_head_input: bool = False,
        child_evidence_profile: Lfm2ChildEvidenceProfile | None = None,
    ) -> None:
        self.torch = torch
        self.model = model
        self.layers = tuple(layers)
        self.contracts = tuple(contracts)
        self.config = config
        if not isinstance(capture_replay_lm_head_input, bool):
            raise TypeError("capture_replay_lm_head_input must be bool.")
        if child_evidence_profile is not None and not isinstance(
            child_evidence_profile,
            Lfm2ChildEvidenceProfile,
        ):
            raise TypeError(
                "child_evidence_profile must be Lfm2ChildEvidenceProfile or None."
            )
        self.capture_replay_lm_head_input = capture_replay_lm_head_input
        self.child_evidence_profile = child_evidence_profile
        model_config = getattr(model, "config", None)
        self._heterogeneous_dynamic_runtime = bool(
            model_config is not None
            and hasattr(
                model_config,
                "neural_forge_dynamic_blank_config_schema",
            )
        )
        self.selected_layer_indices = _selected_layer_indices(
            len(self.layers),
            selected_layer_indices,
        )
        self._selected_layer_index_set = frozenset(self.selected_layer_indices)
        if (
            child_evidence_profile is not None
            and child_evidence_profile.selected_layer_indices
            != self.selected_layer_indices
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Child evidence profile differs from the selected hook layers."
            )
        if (
            child_evidence_profile is not None
            and child_evidence_profile.retain_lm_head_input
            != capture_replay_lm_head_input
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Child evidence profile and LM-head hook selection differ."
            )
        self.handles: list[Any] = []
        self.active_phase: str | None = None
        self.active_sample_last_tokens: int | None = None
        self.raw: dict[int, dict[str, Any]] = {}
        self.global_raw: dict[str, Any] = {}
        self.execution_raw: dict[int, dict[str, Any]] = {}
        self.last_finish_accounting: Mapping[str, Any] = MappingProxyType({})
        self.stage = _HostTransferStage(torch)
        self._register()

    @property
    def stage_counters(self) -> dict[str, int]:
        return self.stage.counters

    def _register_pre(self, module: Any, callback: Any) -> None:
        self.handles.append(
            module.register_forward_pre_hook(callback, with_kwargs=True)
        )

    def _register_post(self, module: Any, callback: Any) -> None:
        self.handles.append(module.register_forward_hook(callback, with_kwargs=True))

    def _register(self) -> None:
        for index, (layer, contract) in enumerate(zip(self.layers, self.contracts)):
            if index not in self._selected_layer_index_set:
                continue
            self._register_pre(
                layer,
                lambda module, args, kwargs, i=index: self._pre_hidden(
                    i, "residual_pre", args, kwargs
                ),
            )
            self._register_post(
                layer,
                lambda module, args, kwargs, output, i=index: self._post_output(
                    i, "layer_output", output, "decoder layer"
                ),
            )
            self._register_post(
                layer.operator_norm,
                lambda module, args, kwargs, output, i=index: self._post_output(
                    i, "operator_input", output, "operator norm"
                ),
            )
            self._register_pre(
                layer.ffn_norm,
                lambda module, args, kwargs, i=index: self._pre_hidden(
                    i, "residual_post_operator", args, kwargs
                ),
            )
            self._register_post(
                layer.ffn_norm,
                lambda module, args, kwargs, output, i=index: self._post_output(
                    i, "mlp_input", output, "FFN norm"
                ),
            )
            mlp = layer.feed_forward
            self._register_post(
                mlp.w1,
                lambda module, args, kwargs, output, i=index: self._post_output(
                    i, "mlp_gate", output, "MLP gate"
                ),
            )
            self._register_post(
                mlp.w3,
                lambda module, args, kwargs, output, i=index: self._post_output(
                    i, "mlp_up", output, "MLP up"
                ),
            )
            self._register_pre(
                mlp.w2,
                lambda module, args, kwargs, i=index: self._pre_hidden(
                    i, "mlp_post_swiglu", args, kwargs
                ),
            )
            self._register_post(
                mlp.w2,
                lambda module, args, kwargs, output, i=index: self._post_output(
                    i, "mlp_down", output, "MLP down"
                ),
            )

            if contract.kind == "full_attention":
                attention = layer.self_attn
                self._register_pre(
                    attention,
                    lambda module, args, kwargs, i=index: self._attention_pre(
                        i, module, args, kwargs
                    ),
                )
                self._register_post(
                    attention,
                    lambda module, args, kwargs, output, i=index: self._post_output(
                        i, "operator_output", output, "attention"
                    ),
                )
                for module_name, raw_name in (
                    ("q_proj", "attention_q_linear"),
                    ("k_proj", "attention_k_linear"),
                    ("v_proj", "attention_v_linear"),
                    ("q_layernorm", "attention_q_normalized"),
                    ("k_layernorm", "attention_k_normalized"),
                ):
                    self._register_post(
                        getattr(attention, module_name),
                        lambda module, args, kwargs, output, i=index, n=raw_name: (
                            self._attention_projection_post(
                                i,
                                n,
                                output,
                                n,
                            )
                        ),
                    )
                self._register_pre(
                    attention.out_proj,
                    lambda module, args, kwargs, i=index: self._attention_context_pre(
                        i, args, kwargs
                    ),
                )
                self._register_post(
                    attention.out_proj,
                    lambda module, args, kwargs, output, i=index: self._post_output(
                        i, "attention_o", output, "attention out projection"
                    ),
                )
            else:
                operator = layer.conv
                self._register_pre(
                    operator,
                    lambda module, args, kwargs, i=index: self._short_conv_pre(
                        i, args, kwargs
                    ),
                )
                self._register_post(
                    operator,
                    lambda module, args, kwargs, output, i=index: self._post_output(
                        i, "operator_output", output, "short convolution"
                    ),
                )
                self._register_post(
                    operator.in_proj,
                    lambda module, args, kwargs, output, i=index: self._post_output(
                        i, "short_conv_bcx", output, "short-conv B/C/x projection"
                    ),
                )
                self._register_pre(
                    operator.out_proj,
                    lambda module, args, kwargs, i=index: self._pre_hidden(
                        i, "short_conv_context", args, kwargs
                    ),
                )
                self._register_post(
                    operator.out_proj,
                    lambda module, args, kwargs, output, i=index: self._post_output(
                        i, "short_conv_o", output, "short-conv out projection"
                    ),
                )
                if self.config.capture_causal_circuit_evidence:
                    self._register_post(
                        operator.in_proj,
                        lambda module, args, kwargs, output, i=index: self._post_output(
                            i,
                            "short_conv_bcx_history",
                            output,
                            "short-conv B/C/x projection history",
                            retain_full_sequence=True,
                        ),
                    )

        if self.config.capture_causal_circuit_evidence:
            inner = getattr(self.model, "model", None)
            final_norm = getattr(inner, "embedding_norm", None)
            lm_head = getattr(self.model, "lm_head", None)
            if final_norm is None or lm_head is None:
                raise Lfm2CircuitTraceCaptureError(
                    "Causal-circuit evidence requires model.embedding_norm "
                    "and lm_head modules."
                )
            self._register_pre(
                final_norm,
                lambda module, args, kwargs: self._pre_global(
                    "output_final_norm_input", args, kwargs
                ),
            )
            self._register_post(
                final_norm,
                lambda module, args, kwargs, output: self._post_global(
                    "output_final_norm_output",
                    output,
                    "final normalization",
                ),
            )
            self._register_pre(
                lm_head,
                lambda module, args, kwargs: self._pre_global(
                    "output_lm_head_input", args, kwargs
                ),
            )
        elif self.capture_replay_lm_head_input:
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None:
                raise Lfm2CircuitTraceCaptureError(
                    "Replay output-boundary evidence requires model.lm_head."
                )
            self._register_pre(
                lm_head,
                lambda module, args, kwargs: self._pre_global(
                    "output_lm_head_input", args, kwargs
                ),
            )

    def begin(
        self,
        phase: str,
        *,
        sample_last_tokens: int | None = None,
    ) -> None:
        if self.active_phase is not None:
            raise Lfm2CircuitTraceCaptureError("A trace phase is already active.")
        if sample_last_tokens is not None and (
            isinstance(sample_last_tokens, bool)
            or not isinstance(sample_last_tokens, int)
            or sample_last_tokens <= 0
        ):
            raise Lfm2CircuitTraceCaptureError(
                "sample_last_tokens override must be a positive integer."
            )
        self.active_phase = phase
        self.active_sample_last_tokens = (
            self.config.sample_last_tokens
            if sample_last_tokens is None
            else sample_last_tokens
        )
        self.raw = {index: {} for index in self.selected_layer_indices}
        self.global_raw = {}
        self.execution_raw = {index: {} for index in self.selected_layer_indices}
        self.last_finish_accounting = MappingProxyType({})
        self.stage.reset()

    def _ensure_inference(self) -> None:
        if self.active_phase is not None and self.torch.is_grad_enabled():
            raise Lfm2CircuitTraceCaptureError(
                "LFM2 hooks executed with gradients enabled."
            )

    def _store(
        self,
        layer_index: int,
        name: str,
        value: Any,
        *,
        axis: int = 1,
        retain_full_sequence: bool = False,
    ) -> None:
        if self.active_phase is None:
            return
        self._ensure_inference()
        target = self.raw[layer_index]
        if name in target:
            raise Lfm2CircuitTraceCaptureError(
                f"Layer {layer_index} emitted duplicate hook endpoint {name!r}."
            )
        endpoint_name = f"{self.active_phase} layer {layer_index} {name}"
        target[name] = self.stage.stage(
            value
            if retain_full_sequence
            else _sampled_sequence_view(
                self.torch,
                value,
                endpoint_name,
                sequence_axis=axis,
                sample_last_tokens=self.active_sample_last_tokens,
            ),
            endpoint_name,
        )

    def _store_exact(self, layer_index: int, name: str, value: Any) -> None:
        if self.active_phase is None:
            return
        self._ensure_inference()
        target = self.raw[layer_index]
        if name in target:
            raise Lfm2CircuitTraceCaptureError(
                f"Layer {layer_index} emitted duplicate hook endpoint {name!r}."
            )
        target[name] = self.stage.stage(
            value,
            f"{self.active_phase} layer {layer_index} {name}",
        )

    def _store_execution(
        self,
        layer_index: int,
        name: str,
        value: Any,
        *,
        clone: bool = False,
    ) -> None:
        if self.active_phase is None:
            return
        self._ensure_inference()
        if not self.torch.is_tensor(value):
            raise Lfm2CircuitTraceCaptureError(
                f"{self.active_phase} layer {layer_index} {name} "
                "must be an execution tensor."
            )
        target = self.execution_raw[layer_index]
        if name in target:
            raise Lfm2CircuitTraceCaptureError(
                f"Layer {layer_index} emitted duplicate execution endpoint {name!r}."
            )
        retained = value.detach()
        if clone:
            retained = retained.clone()
        if retained.requires_grad or retained.grad_fn is not None:
            raise Lfm2CircuitTraceCaptureError(
                f"{self.active_phase} layer {layer_index} {name} "
                "retained a gradient graph."
            )
        target[name] = retained

    def _store_global(self, name: str, value: Any) -> None:
        if self.active_phase is None:
            return
        self._ensure_inference()
        if name in self.global_raw:
            raise Lfm2CircuitTraceCaptureError(
                f"Trace phase emitted duplicate global endpoint {name!r}."
            )
        endpoint_name = f"{self.active_phase} {name}"
        self.global_raw[name] = self.stage.stage(
            _sampled_sequence_view(
                self.torch,
                value,
                endpoint_name,
                sequence_axis=1,
                sample_last_tokens=self.active_sample_last_tokens,
            ),
            endpoint_name,
        )

    def _pre_hidden(
        self,
        layer_index: int,
        name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        if self.active_phase is None:
            return
        self._store(
            layer_index,
            name,
            _first_tensor(self.torch, args, kwargs),
        )

    def _post_output(
        self,
        layer_index: int,
        name: str,
        output: Any,
        endpoint_name: str,
        *,
        retain_full_sequence: bool = False,
    ) -> None:
        if self.active_phase is None:
            return
        self._store(
            layer_index,
            name,
            _output_tensor(self.torch, output, endpoint_name),
            retain_full_sequence=retain_full_sequence,
        )

    def _attention_projection_post(
        self,
        layer_index: int,
        name: str,
        output: Any,
        endpoint_name: str,
    ) -> None:
        if self.active_phase is None:
            return
        value = _output_tensor(self.torch, output, endpoint_name)
        if self.config.capture_causal_circuit_evidence:
            self._store_execution(layer_index, name, value)
        self._store(
            layer_index,
            name,
            value,
            retain_full_sequence=self.config.capture_causal_circuit_evidence,
        )

    def _pre_global(
        self,
        name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        if self.active_phase is None:
            return
        self._store_global(name, _first_tensor(self.torch, args, kwargs))

    def _post_global(
        self,
        name: str,
        output: Any,
        endpoint_name: str,
    ) -> None:
        if self.active_phase is None:
            return
        self._store_global(
            name,
            _output_tensor(self.torch, output, endpoint_name),
        )

    def _attention_pre(
        self,
        layer_index: int,
        module: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        if self.active_phase is None:
            return
        position_embeddings = kwargs.get("position_embeddings")
        if self._heterogeneous_dynamic_runtime:
            local_position_embeddings = getattr(
                module,
                "_position_embeddings",
                None,
            )
            if not callable(local_position_embeddings):
                raise Lfm2CircuitTraceCaptureError(
                    "Dynamic LFM2 attention has no descriptor-local RoPE "
                    "implementation."
                )
            try:
                position_embeddings = local_position_embeddings(
                    _first_tensor(self.torch, args, kwargs),
                    kwargs.get("position_ids"),
                )
            except Exception as exc:
                raise Lfm2CircuitTraceCaptureError(
                    "Dynamic LFM2 attention could not reproduce its effective "
                    "descriptor-local rotary position embeddings."
                ) from exc
        if (
            not isinstance(position_embeddings, (tuple, list))
            or len(position_embeddings) != 2
        ):
            raise Lfm2CircuitTraceCaptureError(
                "LFM2 attention did not receive rotary position embeddings."
            )
        if self.config.capture_causal_circuit_evidence:
            self._store_execution(
                layer_index,
                "attention_cos",
                position_embeddings[0],
            )
            self._store_execution(
                layer_index,
                "attention_sin",
                position_embeddings[1],
            )
        self._store(
            layer_index,
            "attention_cos",
            position_embeddings[0],
            retain_full_sequence=self.config.capture_causal_circuit_evidence,
        )
        self._store(
            layer_index,
            "attention_sin",
            position_embeddings[1],
            retain_full_sequence=self.config.capture_causal_circuit_evidence,
        )
        if not self.config.capture_causal_circuit_evidence:
            return
        attention_mask = kwargs.get("attention_mask")
        if self.torch.is_tensor(attention_mask):
            self._store_execution(
                layer_index,
                "attention_additive_mask",
                attention_mask,
            )
            self._store_exact(
                layer_index,
                "attention_additive_mask",
                attention_mask,
            )
        past = kwargs.get("past_key_values")
        cache_layer = self._cache_layer(past, layer_index)
        if cache_layer is None:
            return
        keys = getattr(cache_layer, "keys", None)
        values = getattr(cache_layer, "values", None)
        if self.torch.is_tensor(keys) and int(keys.numel()) > 0:
            self._store_execution(
                layer_index,
                "attention_pre_key_cache",
                keys,
                clone=True,
            )
            self._store_exact(layer_index, "attention_pre_key_cache", keys)
        if self.torch.is_tensor(values) and int(values.numel()) > 0:
            self._store_execution(
                layer_index,
                "attention_pre_value_cache",
                values,
                clone=True,
            )
            self._store_exact(layer_index, "attention_pre_value_cache", values)

    def _attention_context_pre(
        self,
        layer_index: int,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        if self.active_phase is None:
            return
        context = _first_tensor(self.torch, args, kwargs)
        try:
            if self.config.capture_causal_circuit_evidence:
                self._authenticate_attention_execution_context(
                    layer_index,
                    context,
                )
            self._store(
                layer_index,
                "attention_context",
                context,
            )
        finally:
            self.execution_raw[layer_index] = {}

    def _authenticate_attention_execution_context(
        self,
        layer_index: int,
        context: Any,
    ) -> None:
        torch = self.torch
        execution = self.execution_raw[layer_index]
        required = {
            "attention_q_normalized",
            "attention_k_normalized",
            "attention_v_linear",
            "attention_cos",
            "attention_sin",
        }
        missing = sorted(required.difference(execution))
        if missing:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {layer_index} is missing execution-device "
                f"endpoints: {missing}."
            )
        contract = self.contracts[layer_index]
        operator = contract.operator
        if not isinstance(operator, Lfm2AttentionContract):
            raise Lfm2CircuitTraceCaptureError(
                "Internal LFM2 attention execution contract mismatch."
            )
        q_heads = operator.query_heads
        kv_heads = operator.key_value_heads
        head_dim = operator.head_dim
        groups = q_heads // kv_heads
        q_normalized = execution["attention_q_normalized"]
        k_normalized = execution["attention_k_normalized"]
        v_linear = execution["attention_v_linear"]
        cos = execution["attention_cos"].unsqueeze(2)
        sin = execution["attention_sin"].unsqueeze(2)
        batch = int(q_normalized.shape[0])
        sequence = int(q_normalized.shape[1])
        sample_count = min(
            sequence,
            int(self.active_sample_last_tokens or self.config.sample_last_tokens),
        )
        q_rotary = q_normalized * cos + _rotate_half(torch, q_normalized) * sin
        k_rotary = k_normalized * cos + _rotate_half(torch, k_normalized) * sin
        query = q_rotary.transpose(1, 2).contiguous()
        current_key = k_rotary.transpose(1, 2).contiguous()
        current_value = v_linear.view(
            batch,
            sequence,
            kv_heads,
            head_dim,
        ).transpose(1, 2)
        pre_key = execution.get("attention_pre_key_cache")
        pre_value = execution.get("attention_pre_value_cache")
        if (pre_key is None) is not (pre_value is None):
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {layer_index} retained only one execution "
                "side of its K/V cache."
            )
        if pre_key is not None:
            key_history = torch.cat((pre_key, current_key), dim=2)
            value_history = torch.cat((pre_value, current_value), dim=2)
        else:
            key_history = current_key
            value_history = current_value
        replay = _reconstruct_attention_causal_path(
            torch,
            query=query,
            key_history=key_history,
            value_history=value_history,
            additive_mask=execution.get("attention_additive_mask"),
            groups=groups,
            sample_count=sample_count,
            head_dim=head_dim,
        )
        expected_context_shape = (
            batch,
            sample_count,
            q_heads * head_dim,
        )
        sampled_context = context[:, -sample_count:, :]
        if tuple(sampled_context.shape) != expected_context_shape:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {layer_index} execution context shape changed."
            )
        captured_context = sampled_context.view(
            batch,
            sample_count,
            q_heads,
            head_dim,
        ).transpose(1, 2)
        reconstructed_context = replay["reconstructed_context"]
        maximum_error = float(
            (
                reconstructed_context.to(torch.float32)
                - captured_context.to(torch.float32)
            )
            .abs()
            .max()
            .item()
        )
        if not torch.allclose(
            reconstructed_context,
            captured_context,
            rtol=2e-2,
            atol=2e-3,
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {layer_index} execution-device causal "
                "score/probability reconstruction does not reproduce its "
                f"pre-O context (max error {maximum_error:.6g})."
            )

        serialized = {
            "attention_execution_q_rotary": q_rotary,
            "attention_execution_k_rotary": k_rotary,
            "attention_execution_key_history": key_history,
            "attention_execution_value_history": value_history,
            "attention_execution_qk_scores": replay["qk_scores"],
            "attention_execution_sampled_mask": replay["sampled_mask"],
            "attention_execution_probabilities": replay["probabilities"],
            "attention_execution_reconstructed_context": reconstructed_context,
        }
        for name, value in serialized.items():
            self._store_exact(layer_index, name, value)
        target = self.raw[layer_index]
        target["attention_execution_device"] = str(query.device)
        target["attention_execution_device_type"] = query.device.type
        target["attention_execution_reconstruction_max_abs_error"] = maximum_error
        target["attention_execution_reconstruction_verified"] = True

    @staticmethod
    def _cache_layer(past: Any, layer_index: int) -> Any | None:
        if past is None:
            return None
        layers = getattr(past, "layers", None)
        if layers is None:
            return None
        try:
            return layers[layer_index]
        except (IndexError, TypeError):
            return None

    def _short_conv_pre(
        self,
        layer_index: int,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        if self.active_phase is None or not self.config.capture_causal_circuit_evidence:
            return
        cache_layer = self._cache_layer(kwargs.get("past_key_values"), layer_index)
        if cache_layer is None:
            return
        state = getattr(cache_layer, "conv_states", None)
        if self.torch.is_tensor(state) and int(state.numel()) > 0:
            self._store_exact(layer_index, "short_conv_pre_state", state)

    def finish(self) -> tuple[dict[str, Any], dict[str, Any]]:
        phase = self.active_phase
        if phase is None:
            raise Lfm2CircuitTraceCaptureError("No trace phase is active.")
        try:
            self._resolve_staged_endpoints()
            tensors: dict[str, Any] = {}
            diagnostics: dict[str, Any] = {}
            transient_raw_bytes = sum(
                int(value.numel() * value.element_size())
                for raw in self.raw.values()
                for value in raw.values()
                if self.torch.is_tensor(value)
            ) + sum(
                int(value.numel() * value.element_size())
                for value in self.global_raw.values()
                if self.torch.is_tensor(value)
            )
            full_validated_bytes = 0
            retained_layer_bytes = 0
            for index in self.selected_layer_indices:
                contract = self.contracts[index]
                full_layer_tensors, layer_diagnostics = self._finalize_layer(
                    index, contract, self.raw[index]
                )
                full_validated_bytes += sum(
                    int(value.numel() * value.element_size())
                    for value in full_layer_tensors.values()
                )
                if self.child_evidence_profile is None:
                    layer_tensors = full_layer_tensors
                else:
                    retained_suffixes = self.child_evidence_profile.suffixes_for_layer(
                        index
                    )
                    missing = sorted(
                        set(retained_suffixes).difference(full_layer_tensors)
                    )
                    if missing:
                        raise Lfm2CircuitTraceCaptureError(
                            "Child evidence profile requests unavailable validated "
                            f"layer {index} endpoints: {missing}."
                        )
                    layer_tensors = {
                        name: full_layer_tensors[name]
                        for name in retained_suffixes
                    }
                    layer_diagnostics = {
                        **layer_diagnostics,
                        "full_layer_algebra_validated_before_filter": True,
                        "retained_suffixes": list(retained_suffixes),
                        "retained_tensor_count": len(layer_tensors),
                        "transient_validated_tensor_count": len(
                            full_layer_tensors
                        ),
                    }
                retained_layer_bytes += sum(
                    int(value.numel() * value.element_size())
                    for value in layer_tensors.values()
                )
                prefix = f"{phase}.layer.{index:03d}."
                tensors.update(
                    {prefix + name: value for name, value in layer_tensors.items()}
                )
                diagnostics[f"layer.{index:03d}"] = layer_diagnostics
            if self.config.capture_causal_circuit_evidence:
                output_tensors, output_diagnostics = self._finalize_output_boundary()
                tensors.update(
                    {f"{phase}.{name}": value for name, value in output_tensors.items()}
                )
                diagnostics["output"] = output_diagnostics
            elif self.capture_replay_lm_head_input:
                head_input = self.global_raw.get("output_lm_head_input")
                if head_input is None:
                    raise Lfm2CircuitTraceCaptureError(
                        "Replay output boundary is missing the LM-head input."
                    )
                tensors[f"{phase}.output.lm_head_input"] = head_input
            retained_finish_bytes = sum(
                int(value.numel() * value.element_size())
                for value in tensors.values()
            )
            self.last_finish_accounting = MappingProxyType(
                {
                    "profile_sha256": (
                        None
                        if self.child_evidence_profile is None
                        else self.child_evidence_profile.profile_sha256
                    ),
                    "full_layer_algebra_validation_completed": True,
                    "transient_hook_endpoint_bytes": transient_raw_bytes,
                    "full_finalized_validation_bytes": full_validated_bytes,
                    "retained_layer_bytes": retained_layer_bytes,
                    "retained_finish_bytes": retained_finish_bytes,
                    "estimated_transient_validation_peak_bytes": (
                        transient_raw_bytes + full_validated_bytes
                    ),
                }
            )
            return tensors, diagnostics
        finally:
            self.active_phase = None
            self.active_sample_last_tokens = None
            self.raw = {}
            self.global_raw = {}
            self.execution_raw = {}

    def _resolve_staged_endpoints(self) -> None:
        """Land every deferred host copy before any layer algebra runs.

        Resolution performs the batched finiteness pass, so a non-finite
        endpoint aborts this phase with the original error before residual,
        SwiGLU, attention, or short-conv reconstruction can misreport it as an
        algebra mismatch, and long before anything is hashed or sealed.
        """

        self.stage.resolve()
        for raw in self.raw.values():
            _materialize_staged_endpoints(raw)
        _materialize_staged_endpoints(self.global_raw)

    def _finalize_output_boundary(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        required = {
            "output_final_norm_input",
            "output_final_norm_output",
            "output_lm_head_input",
        }
        missing = sorted(required.difference(self.global_raw))
        if missing:
            raise Lfm2CircuitTraceCaptureError(
                f"Output boundary is missing hook endpoints: {missing}."
            )
        final_output = self.global_raw["output_final_norm_output"]
        head_input = self.global_raw["output_lm_head_input"]
        if not self.torch.allclose(
            final_output,
            head_input,
            rtol=0.0,
            atol=0.0,
        ):
            raise Lfm2CircuitTraceCaptureError(
                "The LM head input does not exactly equal final RMSNorm output."
            )
        return (
            {
                "output.final_norm_input": self.global_raw["output_final_norm_input"],
                "output.final_norm_output": final_output,
                "output.lm_head_input": head_input,
            },
            {
                "final_norm_boundary_captured": True,
                "lm_head_input_exactly_verified": True,
            },
        )

    def _finalize_layer(
        self,
        index: int,
        contract: Lfm2RuntimeLayerContract,
        raw: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self.torch
        common = {
            "residual_pre",
            "operator_input",
            "operator_output",
            "residual_post_operator",
            "mlp_input",
            "mlp_gate",
            "mlp_up",
            "mlp_post_swiglu",
            "mlp_down",
            "layer_output",
        }
        missing = sorted(common.difference(raw))
        if missing:
            raise Lfm2CircuitTraceCaptureError(
                f"Layer {index} is missing hook endpoints: {missing}."
            )
        residual_pre = raw["residual_pre"]
        operator_output = raw["operator_output"]
        post_operator = raw["residual_post_operator"]
        post_operator_expected = residual_pre + operator_output
        if not torch.allclose(
            post_operator_expected,
            post_operator,
            rtol=1e-5,
            atol=1e-7,
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Layer {index} operator residual hook does not equal residual + operator output."
            )
        # Reproduce the actual LFM2 runtime equation in the runtime dtype.  In
        # bf16, recomputing SiLU in float64 and comparing against the bf16
        # tensor is not an algebraic check; it measures precision differences.
        mlp_expected = torch.nn.functional.silu(raw["mlp_gate"]) * raw["mlp_up"]
        mlp_actual = raw["mlp_post_swiglu"]
        if not torch.allclose(mlp_expected, mlp_actual, rtol=1e-5, atol=1e-7):
            raise Lfm2CircuitTraceCaptureError(
                f"Layer {index} post-SwiGLU hook does not equal silu(gate) * up."
            )
        layer_expected = post_operator + raw["mlp_down"]
        layer_actual = raw["layer_output"]
        if not torch.allclose(layer_expected, layer_actual, rtol=1e-5, atol=1e-7):
            raise Lfm2CircuitTraceCaptureError(
                f"Layer {index} residual endpoints do not reproduce its output."
            )
        result = {
            "residual.pre": residual_pre,
            "operator.input": raw["operator_input"],
            "operator.output": operator_output,
            "residual.post_operator": post_operator,
            "mlp.input": raw["mlp_input"],
            "mlp.gate": raw["mlp_gate"],
            "mlp.up": raw["mlp_up"],
            "mlp.post_swiglu": raw["mlp_post_swiglu"],
            "mlp.down": raw["mlp_down"],
            "residual.post_layer": raw["layer_output"],
        }
        diagnostics = {
            "kind": contract.kind,
            "post_swiglu_verified": True,
            "operator_residual_verified": True,
            "layer_residual_verified": True,
        }
        if contract.kind == "full_attention":
            attention_result, attention_diagnostics = self._finalize_attention(
                index, contract, raw
            )
            result.update(attention_result)
            diagnostics["attention_qkvo_context_captured"] = True
            diagnostics["short_conv_bcx_captured"] = False
            diagnostics.update(attention_diagnostics)
        else:
            conv_result, conv_diagnostics = self._finalize_short_conv(
                index, contract, raw
            )
            result.update(conv_result)
            diagnostics["attention_qkvo_context_captured"] = False
            diagnostics["short_conv_bcx_captured"] = True
            diagnostics["short_conv_not_treated_as_qkv"] = True
            diagnostics.update(conv_diagnostics)
        return result, diagnostics

    def _finalize_attention(
        self,
        index: int,
        contract: Lfm2RuntimeLayerContract,
        raw: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self.torch
        required = {
            "attention_q_linear",
            "attention_k_linear",
            "attention_v_linear",
            "attention_q_normalized",
            "attention_k_normalized",
            "attention_cos",
            "attention_sin",
            "attention_context",
            "attention_o",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} is missing endpoints: {missing}."
            )
        operator = contract.operator
        if not isinstance(operator, Lfm2AttentionContract):
            raise Lfm2CircuitTraceCaptureError(
                "Internal LFM2 attention contract mismatch."
            )
        q_heads = operator.query_heads
        kv_heads = operator.key_value_heads
        head_dim = operator.head_dim

        q_linear_full = raw["attention_q_linear"]
        k_linear_full = raw["attention_k_linear"]
        v_linear_full = raw["attention_v_linear"]
        batch, sequence = (
            int(q_linear_full.shape[0]),
            int(q_linear_full.shape[1]),
        )
        expected_projection_shapes = {
            "q": (batch, sequence, q_heads * head_dim),
            "k": (batch, sequence, kv_heads * head_dim),
            "v": (batch, sequence, kv_heads * head_dim),
        }
        if tuple(q_linear_full.shape) != expected_projection_shapes["q"]:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} Q projection shape changed."
            )
        if tuple(k_linear_full.shape) != expected_projection_shapes["k"]:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} K projection shape changed."
            )
        if tuple(v_linear_full.shape) != expected_projection_shapes["v"]:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} V projection shape changed."
            )
        sample_count = min(
            sequence,
            int(self.active_sample_last_tokens or self.config.sample_last_tokens),
        )
        expected_context_shape = (
            batch,
            sample_count,
            q_heads * head_dim,
        )
        if tuple(raw["attention_context"].shape) != expected_context_shape:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} pre-O context shape changed."
            )
        q_normalized = raw["attention_q_normalized"]
        k_normalized = raw["attention_k_normalized"]
        if tuple(q_normalized.shape) != (batch, sequence, q_heads, head_dim):
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} normalized Q shape changed."
            )
        if tuple(k_normalized.shape) != (batch, sequence, kv_heads, head_dim):
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} normalized K shape changed."
            )
        if self.config.capture_causal_circuit_evidence:
            execution_required = {
                "attention_execution_q_rotary",
                "attention_execution_k_rotary",
                "attention_execution_key_history",
                "attention_execution_value_history",
                "attention_execution_qk_scores",
                "attention_execution_sampled_mask",
                "attention_execution_probabilities",
                "attention_execution_reconstructed_context",
                "attention_execution_device",
                "attention_execution_device_type",
                "attention_execution_reconstruction_max_abs_error",
                "attention_execution_reconstruction_verified",
            }
            execution_missing = sorted(execution_required.difference(raw))
            if execution_missing:
                raise Lfm2CircuitTraceCaptureError(
                    f"Attention layer {index} is missing authenticated "
                    f"execution-device evidence: {execution_missing}."
                )
            q_rotary = raw["attention_execution_q_rotary"]
            k_rotary = raw["attention_execution_k_rotary"]
        else:
            cos = raw["attention_cos"].unsqueeze(2)
            sin = raw["attention_sin"].unsqueeze(2)
            q_rotary = q_normalized * cos + _rotate_half(torch, q_normalized) * sin
            k_rotary = k_normalized * cos + _rotate_half(torch, k_normalized) * sin
        context = raw["attention_context"].view(
            batch,
            sample_count,
            q_heads,
            head_dim,
        )
        q_linear = q_linear_full[:, -sample_count:, :]
        k_linear = k_linear_full[:, -sample_count:, :]
        v_linear = v_linear_full[:, -sample_count:, :]
        q_sample = q_rotary[:, -sample_count:, :, :]
        k_sample = k_rotary[:, -sample_count:, :, :]
        result = {
            "attention.q_linear": q_linear.view(batch, sample_count, q_heads, head_dim)
            .transpose(1, 2)
            .contiguous(),
            "attention.q": q_sample.transpose(1, 2).contiguous(),
            "attention.k_linear": k_linear.view(batch, sample_count, kv_heads, head_dim)
            .transpose(1, 2)
            .contiguous(),
            "attention.k": k_sample.transpose(1, 2).contiguous(),
            "attention.v": v_linear.view(batch, sample_count, kv_heads, head_dim)
            .transpose(1, 2)
            .contiguous(),
            "attention.context": context.transpose(1, 2).contiguous(),
            "attention.o": raw["attention_o"],
        }
        diagnostics = {
            "attention_causal_circuit_evidence_captured": False,
        }
        if not self.config.capture_causal_circuit_evidence:
            return result, diagnostics

        groups = q_heads // kv_heads
        if groups <= 0 or groups * kv_heads != q_heads:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} has invalid grouped-query geometry."
            )
        key_history = raw["attention_execution_key_history"]
        value_history = raw["attention_execution_value_history"]
        qk_scores = raw["attention_execution_qk_scores"]
        sampled_mask = raw["attention_execution_sampled_mask"]
        probabilities = raw["attention_execution_probabilities"]
        reconstructed_context = raw["attention_execution_reconstructed_context"]
        captured_context = context.transpose(1, 2)
        serialized_maximum_error = float(
            (
                reconstructed_context.to(torch.float32)
                - captured_context.to(torch.float32)
            )
            .abs()
            .max()
            .item()
        )
        if not torch.allclose(
            reconstructed_context,
            captured_context,
            rtol=2e-2,
            atol=2e-3,
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} causal score/probability reconstruction "
                "changed during CPU serialization "
                f"(max error {serialized_maximum_error:.6g})."
            )
        maximum_error = float(raw["attention_execution_reconstruction_max_abs_error"])
        if raw["attention_execution_reconstruction_verified"] is not True:
            raise Lfm2CircuitTraceCaptureError(
                f"Attention layer {index} lacks a successful execution-device "
                "causal reconstruction decision."
            )
        result.update(
            {
                "attention.q_normalized_pre_rope": q_normalized.transpose(
                    1, 2
                ).contiguous(),
                "attention.k_normalized_pre_rope": k_normalized.transpose(
                    1, 2
                ).contiguous(),
                "attention.rope_cos": raw["attention_cos"],
                "attention.rope_sin": raw["attention_sin"],
                "attention.key_history": key_history.clone(),
                "attention.value_history": value_history.clone(),
                "attention.qk_scores": qk_scores,
                "attention.additive_mask": sampled_mask,
                "attention.probabilities": probabilities,
                "attention.context_from_probabilities": reconstructed_context,
            }
        )
        diagnostics.update(
            {
                "attention_causal_circuit_evidence_captured": True,
                "attention_context_reconstructed_from_probabilities": True,
                "attention_context_reconstruction_max_abs_error": maximum_error,
                "attention_context_serialization_max_abs_error": (
                    serialized_maximum_error
                ),
                "attention_context_authenticated_on_execution_device": True,
                "attention_evidence_serialized_after_device_authentication": True,
                "attention_reconstruction_execution_device": raw[
                    "attention_execution_device"
                ],
                "attention_reconstruction_execution_device_type": raw[
                    "attention_execution_device_type"
                ],
                "attention_query_count": sample_count,
                "attention_key_history_count": int(key_history.shape[2]),
            }
        )
        return result, diagnostics

    def _finalize_short_conv(
        self,
        index: int,
        contract: Lfm2RuntimeLayerContract,
        raw: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self.torch
        required = {
            "short_conv_bcx",
            "short_conv_context",
            "short_conv_o",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise Lfm2CircuitTraceCaptureError(
                f"Short-conv layer {index} is missing endpoints: {missing}."
            )
        operator = contract.operator
        if not isinstance(operator, Lfm2RuntimeShortConvContract):
            raise Lfm2CircuitTraceCaptureError(
                "Internal LFM2 short-conv contract mismatch."
            )
        internal_width = (
            operator.internal_channel_width
            if isinstance(operator, Lfm2HeterogeneousShortConvContract)
            else operator.hidden_size
        )
        bcx = raw["short_conv_bcx"]
        if int(bcx.shape[-1]) != 3 * internal_width:
            raise Lfm2CircuitTraceCaptureError(
                f"Short-conv layer {index} B/C/x projection shape changed."
            )
        b_value, c_value, x_value = bcx.chunk(3, dim=-1)
        result = {
            "short_conv.b": b_value.contiguous(),
            "short_conv.c": c_value.contiguous(),
            "short_conv.x": x_value.contiguous(),
            "short_conv.bx": (b_value * x_value).contiguous(),
            "short_conv.context": raw["short_conv_context"],
            "short_conv.o": raw["short_conv_o"],
        }
        diagnostics = {
            "short_conv_causal_circuit_evidence_captured": False,
        }
        if not self.config.capture_causal_circuit_evidence:
            return result, diagnostics

        history = raw.get("short_conv_bcx_history")
        if history is None or int(history.shape[-1]) != 3 * internal_width:
            raise Lfm2CircuitTraceCaptureError(
                f"Short-conv layer {index} is missing its complete B/C/x history."
            )
        b_history, c_history, x_history = history.chunk(3, dim=-1)
        bx_history = b_history * x_history
        batch = int(history.shape[0])
        sequence = int(history.shape[1])
        hidden = internal_width
        fir_length = operator.fir_length
        conv_module = getattr(getattr(self.layers[index], "conv", None), "conv", None)
        kernel_value = getattr(conv_module, "weight", None)
        if not torch.is_tensor(kernel_value) or tuple(kernel_value.shape) != (
            hidden,
            1,
            fir_length,
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Short-conv layer {index} FIR kernel shape changed."
            )
        kernel = _cpu_tensor(
            torch,
            kernel_value[:, 0, :],
            f"{self.active_phase} layer {index} FIR kernel",
        )
        bias_value = getattr(conv_module, "bias", None)
        if bias_value is None:
            bias = torch.zeros(hidden, dtype=kernel.dtype)
            has_bias = False
        else:
            if not torch.is_tensor(bias_value) or tuple(bias_value.shape) != (hidden,):
                raise Lfm2CircuitTraceCaptureError(
                    f"Short-conv layer {index} FIR bias shape changed."
                )
            bias = _cpu_tensor(
                torch,
                bias_value,
                f"{self.active_phase} layer {index} FIR bias",
            )
            has_bias = True

        pre_state = raw.get("short_conv_pre_state")
        bx_channels = bx_history.transpose(1, 2).contiguous()
        if pre_state is None:
            pre_state = torch.zeros(
                (batch, hidden, fir_length),
                dtype=bx_channels.dtype,
            )
            filtered = torch.nn.functional.conv1d(
                bx_channels.to(torch.float32),
                kernel[:, None, :].to(torch.float32),
                bias=bias.to(torch.float32),
                padding=fir_length - 1,
                groups=hidden,
            )[..., :sequence]
        else:
            if tuple(pre_state.shape) != (batch, hidden, fir_length) or sequence != 1:
                raise Lfm2CircuitTraceCaptureError(
                    f"Short-conv layer {index} pre-update state shape changed."
                )
            updated_state = torch.cat(
                (pre_state[..., 1:], bx_channels),
                dim=-1,
            )
            filtered = (
                updated_state.to(torch.float32) * kernel.to(torch.float32)[None, :, :]
            ).sum(dim=-1, keepdim=True)
            filtered = filtered + bias.to(torch.float32)[None, :, None]

        filtered_tokens = filtered.transpose(1, 2).contiguous()
        gated_context = c_history.to(torch.float32) * filtered_tokens
        sample_count = int(raw["short_conv_context"].shape[1])
        captured_context = raw["short_conv_context"].to(torch.float32)
        reconstructed_sample = gated_context[:, -sample_count:, :]
        maximum_error = float(
            (reconstructed_sample - captured_context).abs().max().item()
        )
        if not torch.allclose(
            reconstructed_sample,
            captured_context,
            rtol=2e-2,
            atol=2e-3,
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Short-conv layer {index} FIR reconstruction does not reproduce "
                f"its pre-output context (max error {maximum_error:.6g})."
            )
        result.update(
            {
                "short_conv.b_history": b_history,
                "short_conv.c_history": c_history,
                "short_conv.x_history": x_history,
                "short_conv.bx_history": bx_history,
                "short_conv.pre_bx_state": pre_state,
                "short_conv.fir_kernel": kernel,
                "short_conv.fir_bias": bias,
                "short_conv.fir_has_bias": torch.tensor(
                    [int(has_bias)],
                    dtype=torch.int8,
                ),
                "short_conv.filtered_bx": filtered_tokens,
                "short_conv.context_from_fir": gated_context,
            }
        )
        diagnostics.update(
            {
                "short_conv_causal_circuit_evidence_captured": True,
                "short_conv_context_reconstructed_from_fir": True,
                "short_conv_context_reconstruction_max_abs_error": maximum_error,
                "short_conv_history_count": sequence,
                "short_conv_fir_length": fir_length,
            }
        )
        return result, diagnostics

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.active_phase = None
        self.active_sample_last_tokens = None
        self.raw = {}
        self.global_raw = {}
        self.execution_raw = {}
        self.stage.reset()


def _output_field(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _output_logits(torch: Any, output: Any) -> Any:
    value = _output_field(output, "logits")
    if not torch.is_tensor(value) or value.ndim != 3:
        raise Lfm2CircuitTraceCaptureError(
            "The LFM2 causal parent did not return rank-3 logits."
        )
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise Lfm2CircuitTraceCaptureError(
            "The LFM2 causal parent returned non-finite logits."
        )
    return value


def _forward(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any | None,
    use_cache: bool,
    past_key_values: Any | None,
) -> Any:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
        "return_dict": True,
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    try:
        return model(**kwargs)
    except Exception as exc:
        raise Lfm2CircuitTraceCaptureError(
            "The LFM2 parent forward failed during circuit tracing."
        ) from exc


def _compare_logits(
    torch: Any,
    cached: Any,
    full_prefix: Any,
    *,
    config: Lfm2CircuitTraceCaptureConfig,
    name: str,
) -> dict[str, Any]:
    cached_cpu = _cpu_tensor(torch, cached, f"{name} cached logits")
    full_cpu = _cpu_tensor(torch, full_prefix, f"{name} full-prefix logits")
    if tuple(cached_cpu.shape) != tuple(full_cpu.shape):
        raise Lfm2CircuitTraceCaptureError(
            f"{name} cached and full-prefix logits have different shapes."
        )
    difference = cached_cpu.to(torch.float64) - full_cpu.to(torch.float64)
    max_abs_error = float(difference.abs().max().item())
    cached_rows = cached_cpu.to(torch.float64).reshape(-1, int(cached_cpu.shape[-1]))
    full_rows = full_cpu.to(torch.float64).reshape(-1, int(full_cpu.shape[-1]))
    cached_top1 = torch.argmax(cached_rows, dim=-1)
    full_top1 = torch.argmax(full_rows, dim=-1)
    agreements = cached_top1 == full_top1
    disagreement_indices = torch.nonzero(~agreements, as_tuple=False).reshape(-1)
    cached_top2 = torch.topk(cached_rows, k=2, dim=-1).values
    full_top2 = torch.topk(full_rows, k=2, dim=-1).values
    cached_margins = cached_top2[:, 0] - cached_top2[:, 1]
    full_margins = full_top2[:, 0] - full_top2[:, 1]
    error_envelope_verified = True
    maximum_cached_preference_gap = 0.0
    maximum_full_preference_gap = 0.0
    first_disagreement: dict[str, Any] | None = None
    for raw_index in disagreement_indices.tolist():
        row_index = int(raw_index)
        cached_id = int(cached_top1[row_index].item())
        full_id = int(full_top1[row_index].item())
        row_difference = difference.reshape_as(cached_rows)[row_index]
        row_error = float(row_difference.abs().max().item())
        pair_error_envelope = float(
            (row_difference[cached_id].abs() + row_difference[full_id].abs()).item()
        )
        cached_gap = float(
            (cached_rows[row_index, cached_id] - cached_rows[row_index, full_id]).item()
        )
        full_gap = float(
            (full_rows[row_index, full_id] - full_rows[row_index, cached_id]).item()
        )
        maximum_cached_preference_gap = max(maximum_cached_preference_gap, cached_gap)
        maximum_full_preference_gap = max(maximum_full_preference_gap, full_gap)
        within_envelope = (
            cached_gap <= pair_error_envelope + 1e-12
            and full_gap <= pair_error_envelope + 1e-12
        )
        error_envelope_verified = error_envelope_verified and within_envelope
        if first_disagreement is None:
            first_disagreement = {
                "row_index": row_index,
                "cached_token_id": cached_id,
                "full_prefix_token_id": full_id,
                "cached_winner_margin": float(cached_margins[row_index].item()),
                "full_prefix_winner_margin": float(full_margins[row_index].item()),
                "cached_preference_gap": cached_gap,
                "full_prefix_preference_gap": full_gap,
                "row_max_abs_error": row_error,
                "winner_pair_error_envelope": pair_error_envelope,
                "within_numeric_error_envelope": within_envelope,
            }
    numerically_close = bool(
        torch.allclose(
            cached_cpu.to(torch.float64),
            full_cpu.to(torch.float64),
            rtol=config.logit_rtol,
            atol=config.logit_atol,
        )
    )
    top1_agreement = bool(bool(agreements.all().item()))
    top1_decision_verified = top1_agreement or (
        config.allow_top1_disagreement_within_numeric_error_envelope
        and error_envelope_verified
    )
    if not numerically_close:
        raise Lfm2CircuitTraceCaptureError(
            f"{name} cached logits differ from use_cache=False full-prefix logits "
            f"(maximum error {max_abs_error:.6g})."
        )
    if config.require_top1_agreement and not top1_decision_verified:
        assert first_disagreement is not None
        raise Lfm2CircuitTraceCaptureError(
            f"{name} cached and full-prefix logits choose different tokens; "
            f"first mismatch cached={first_disagreement['cached_token_id']} "
            f"full={first_disagreement['full_prefix_token_id']}, "
            f"cached_margin={first_disagreement['cached_winner_margin']:.6g}, "
            f"full_margin={first_disagreement['full_prefix_winner_margin']:.6g}, "
            f"row_error={first_disagreement['row_max_abs_error']:.6g}, "
            "within_error_envelope="
            f"{first_disagreement['within_numeric_error_envelope']}."
        )
    return {
        "max_abs_error": max_abs_error,
        "numerically_close": numerically_close,
        "top1_agreement": top1_agreement,
        "top1_decision_verified": top1_decision_verified,
        "top1_exact_or_numeric_tie_required": (config.require_top1_agreement),
        "numeric_top1_tie_acceptance_enabled": (
            config.allow_top1_disagreement_within_numeric_error_envelope
        ),
        "top1_disagreement_count": int(disagreement_indices.numel()),
        "top1_disagreement_within_numeric_error_envelope": (error_envelope_verified),
        "maximum_cached_preference_gap_on_disagreement": (
            maximum_cached_preference_gap
        ),
        "maximum_full_prefix_preference_gap_on_disagreement": (
            maximum_full_preference_gap
        ),
        "minimum_cached_top1_margin": float(cached_margins.min().item()),
        "minimum_full_prefix_top1_margin": float(full_margins.min().item()),
        "cached_top1_sha256": _tensor_sha256(torch, cached_top1),
        "full_prefix_top1_sha256": _tensor_sha256(torch, full_top1),
        "first_top1_disagreement": first_disagreement,
    }


def _cache_snapshot(
    torch: Any,
    past: Any,
    contracts: Sequence[Lfm2RuntimeLayerContract],
    *,
    prefix: str,
    sample_last_tokens: int,
    selected_layer_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    layers = getattr(past, "layers", None)
    if layers is None:
        raise Lfm2CircuitTraceCaptureError(
            "The returned LFM2 cache does not match the decoder topology."
        )
    try:
        cache_layers = tuple(layers)
    except TypeError as exc:
        raise Lfm2CircuitTraceCaptureError(
            "The returned LFM2 cache layers are not iterable."
        ) from exc
    if len(cache_layers) != len(contracts):
        raise Lfm2CircuitTraceCaptureError(
            "The returned LFM2 cache does not match the decoder topology."
        )
    selected = frozenset(
        _selected_layer_indices(len(contracts), selected_layer_indices)
    )
    result: dict[str, Any] = {}
    for index, (cache_layer, contract) in enumerate(zip(cache_layers, contracts)):
        if index not in selected:
            continue
        base = f"{prefix}.layer.{index:03d}."
        if contract.kind == "full_attention":
            keys = getattr(cache_layer, "keys", None)
            values = getattr(cache_layer, "values", None)
            if not torch.is_tensor(keys) or not torch.is_tensor(values):
                raise Lfm2CircuitTraceCaptureError(
                    f"Attention cache layer {index} has no K/V tensors."
                )
            result[base + "attention.key_cache"] = _sample_sequence_tensor(
                torch,
                keys,
                f"{prefix} layer {index} key cache",
                sequence_axis=2,
                sample_last_tokens=sample_last_tokens,
            )
            result[base + "attention.value_cache"] = _sample_sequence_tensor(
                torch,
                values,
                f"{prefix} layer {index} value cache",
                sequence_axis=2,
                sample_last_tokens=sample_last_tokens,
            )
        else:
            conv_state = getattr(cache_layer, "conv_states", None)
            if not torch.is_tensor(conv_state):
                raise Lfm2CircuitTraceCaptureError(
                    f"Short-conv cache layer {index} has no convolution state."
                )
            result[base + "short_conv.conv_state"] = _cpu_tensor(
                torch,
                conv_state,
                f"{prefix} layer {index} convolution state",
            )
            flag_names = (
                "is_conv_states_initialized",
                "is_recurrent_states_initialized",
                "has_previous_state",
            )
            flags: list[int] = []
            for flag_name in flag_names:
                flag = getattr(cache_layer, flag_name, None)
                if not isinstance(flag, bool):
                    raise Lfm2CircuitTraceCaptureError(
                        f"Short-conv cache layer {index} has invalid {flag_name}."
                    )
                flags.append(int(flag))
            result[base + "short_conv.state_flags"] = torch.tensor(
                flags,
                dtype=torch.int8,
            )
            kernel_size = getattr(cache_layer, "conv_kernel_size", None)
            if (
                isinstance(kernel_size, bool)
                or not isinstance(kernel_size, int)
                or kernel_size <= 0
            ):
                raise Lfm2CircuitTraceCaptureError(
                    f"Short-conv cache layer {index} has invalid conv_kernel_size."
                )
            result[base + "short_conv.kernel_size"] = torch.tensor(
                [kernel_size],
                dtype=torch.int64,
            )
            recurrent = getattr(cache_layer, "recurrent_states", None)
            if torch.is_tensor(recurrent):
                result[base + "short_conv.recurrent_state"] = _cpu_tensor(
                    torch,
                    recurrent,
                    f"{prefix} layer {index} recurrent state",
                )
    return result


def _move_to_device(
    value: Any | None,
    device: Any,
) -> Any | None:
    if value is None:
        return None
    return value.to(device=device)


def _capture_batch(
    torch: Any,
    model: Any,
    layers: Sequence[Any],
    contracts: Sequence[Lfm2RuntimeLayerContract],
    batch: _PreparedBatch,
    config: Lfm2CircuitTraceCaptureConfig,
    device: Any,
    *,
    selected_layer_indices: Sequence[int] | None = None,
    capture_replay_lm_head_input: bool = False,
    child_evidence_profile: Lfm2ChildEvidenceProfile | None = None,
    telemetry: _CaptureTelemetry | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_ids = batch.input_ids.to(device=device)
    attention_mask = batch.attention_mask.to(device=device)
    position_ids = _move_to_device(batch.position_ids, device)
    forced_ids = batch.forced_decode_ids.to(device=device)
    vocabulary_size = int(getattr(getattr(model, "config", None), "vocab_size", 0))
    if vocabulary_size <= 0 or bool((input_ids >= vocabulary_size).any().item()):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {batch.index} prompt contains out-of-vocabulary tokens."
        )
    if bool((forced_ids >= vocabulary_size).any().item()):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {batch.index} forced decode contains out-of-vocabulary tokens."
        )

    module_modes = [(module, bool(module.training)) for module in model.modules()]
    selected = _selected_layer_indices(
        len(layers),
        selected_layer_indices,
    )
    retain_logits = (
        child_evidence_profile is None or child_evidence_profile.retain_logits
    )
    retain_phase_forced_token_ids = (
        child_evidence_profile is None
        or child_evidence_profile.retain_phase_forced_token_ids
    )
    if child_evidence_profile is not None:
        capture_replay_lm_head_input = (
            child_evidence_profile.retain_lm_head_input
        )
    hooks = _Lfm2TraceHooks(
        torch,
        model,
        layers,
        contracts,
        config,
        selected,
        capture_replay_lm_head_input=capture_replay_lm_head_input,
        child_evidence_profile=child_evidence_profile,
    )
    tensors: dict[str, Any] = {
        "input.input_ids": batch.input_ids,
        "input.attention_mask": batch.attention_mask,
        f"input.{_FORCED_IDS}": batch.forced_decode_ids,
    }
    if batch.position_ids is not None:
        tensors["input.position_ids"] = batch.position_ids
    phase_diagnostics: dict[str, Any] = {}
    phase_evidence_accounting: dict[str, Any] = {}
    verifications: list[dict[str, Any]] = []
    full_prefix_only = config.decode_trace_mode == FULL_PREFIX_CANONICAL_TRACE_MODE
    try:
        model.eval()
        with torch.inference_mode():
            hooks.begin("prefill")
            full_prefill = _forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                past_key_values=None,
            )
            prefill_trace, prefill_diagnostics = hooks.finish()
            phase_evidence_accounting["prefill"] = dict(
                hooks.last_finish_accounting
            )
            if _output_field(full_prefill, "past_key_values") is not None:
                raise Lfm2CircuitTraceCaptureError(
                    "The uncached prefill returned past_key_values."
                )
            tensors.update(prefill_trace)
            phase_diagnostics["prefill"] = prefill_diagnostics
            full_prefill_logits = None
            if retain_logits or not full_prefix_only:
                full_prefill_logits = _sample_sequence_tensor(
                    torch,
                    _output_logits(torch, full_prefill),
                    "prefill logits",
                    sequence_axis=1,
                    sample_last_tokens=config.sample_last_tokens,
                )
            if retain_logits:
                tensors["prefill.logits"] = full_prefill_logits

            past = None
            if not full_prefix_only:
                cached_prefill = _forward(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=None,
                )
                past = _output_field(cached_prefill, "past_key_values")
                if past is None:
                    raise Lfm2CircuitTraceCaptureError(
                        "The cached prefill did not return past_key_values."
                    )
                cached_prefill_logits = _sample_sequence_tensor(
                    torch,
                    _output_logits(torch, cached_prefill),
                    "cached prefill logits",
                    sequence_axis=1,
                    sample_last_tokens=config.sample_last_tokens,
                )
                tensors["cached_prefill.logits"] = cached_prefill_logits
                tensors.update(
                    _cache_snapshot(
                        torch,
                        past,
                        contracts,
                        prefix="cached_prefill",
                        sample_last_tokens=config.sample_last_tokens,
                        selected_layer_indices=selected,
                    )
                )
                prefill_verification = _compare_logits(
                    torch,
                    cached_prefill_logits,
                    full_prefill_logits,
                    config=config,
                    name="cached prefill",
                )
                verifications.append(
                    {
                        "phase": "cached_prefill",
                        "prefix_length": int(input_ids.shape[1]),
                        **prefill_verification,
                    }
                )

            prefix_ids = input_ids
            prefix_mask = attention_mask
            prefix_positions = position_ids
            for step_index in range(int(forced_ids.shape[1])):
                token = forced_ids[:, step_index : step_index + 1]
                prefix_ids = torch.cat((prefix_ids, token), dim=1)
                prefix_mask = torch.cat(
                    (
                        prefix_mask,
                        torch.ones(
                            (int(prefix_mask.shape[0]), 1),
                            dtype=prefix_mask.dtype,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
                step_positions = None
                if prefix_positions is not None:
                    step_positions = prefix_positions[:, -1:] + 1
                    prefix_positions = torch.cat(
                        (prefix_positions, step_positions), dim=1
                    )

                phase = f"decode.{step_index:03d}"
                if full_prefix_only:
                    hooks.begin(phase, sample_last_tokens=1)
                    full_step = _forward(
                        model,
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        position_ids=prefix_positions,
                        use_cache=False,
                        past_key_values=None,
                    )
                    step_trace, step_diagnostics = hooks.finish()
                    phase_evidence_accounting[phase] = dict(
                        hooks.last_finish_accounting
                    )
                    tensors.update(step_trace)
                    phase_diagnostics[phase] = step_diagnostics
                    if _output_field(full_step, "past_key_values") is not None:
                        raise Lfm2CircuitTraceCaptureError(
                            f"Canonical full-prefix step {step_index} returned a cache."
                        )
                    if retain_logits:
                        tensors[f"{phase}.logits"] = _cpu_tensor(
                            torch,
                            _output_logits(torch, full_step)[:, -1:, :],
                            f"{phase} canonical full-prefix logits",
                        )
                else:
                    hooks.begin(phase)
                    cached_step = _forward(
                        model,
                        input_ids=token,
                        attention_mask=prefix_mask,
                        position_ids=step_positions,
                        use_cache=True,
                        past_key_values=past,
                    )
                    step_trace, step_diagnostics = hooks.finish()
                    phase_evidence_accounting[phase] = dict(
                        hooks.last_finish_accounting
                    )
                    tensors.update(step_trace)
                    phase_diagnostics[phase] = step_diagnostics
                    past = _output_field(cached_step, "past_key_values")
                    if past is None:
                        raise Lfm2CircuitTraceCaptureError(
                            f"Cached decode step {step_index} lost past_key_values."
                        )
                    tensors.update(
                        _cache_snapshot(
                            torch,
                            past,
                            contracts,
                            prefix=phase,
                            sample_last_tokens=config.sample_last_tokens,
                            selected_layer_indices=selected,
                        )
                    )
                    cached_logits = _cpu_tensor(
                        torch,
                        _output_logits(torch, cached_step)[:, -1:, :],
                        f"{phase} logits",
                    )
                    tensors[f"{phase}.logits"] = cached_logits

                    full_step = _forward(
                        model,
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        position_ids=prefix_positions,
                        use_cache=False,
                        past_key_values=None,
                    )
                    if _output_field(full_step, "past_key_values") is not None:
                        raise Lfm2CircuitTraceCaptureError(
                            f"Full-prefix verification step {step_index} "
                            "returned a cache."
                        )
                    full_logits = _cpu_tensor(
                        torch,
                        _output_logits(torch, full_step)[:, -1:, :],
                        f"{phase} full-prefix logits",
                    )
                    tensors[f"{phase}.full_prefix_logits"] = full_logits
                    verification = _compare_logits(
                        torch,
                        cached_logits,
                        full_logits,
                        config=config,
                        name=phase,
                    )
                    verifications.append(
                        {
                            "phase": phase,
                            "prefix_length": int(prefix_ids.shape[1]),
                            "forced_token_sha256": _tensor_sha256(torch, token),
                            **verification,
                        }
                    )
                if retain_phase_forced_token_ids:
                    tensors[f"{phase}.forced_token_ids"] = _cpu_tensor(
                        torch, token, f"{phase} forced token IDs"
                    )
    finally:
        if telemetry is not None:
            telemetry.absorb_stage(hooks.stage_counters)
        hooks.remove()
        for module, was_training in module_modes:
            module.training = was_training

    capture_bytes = sum(
        int(value.numel() * value.element_size()) for value in tensors.values()
    )
    if capture_bytes > config.max_capture_bytes:
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {batch.index} trace exceeded max_capture_bytes."
        )
    maximum_transient_validation_peak_bytes = max(
        (
            int(value["estimated_transient_validation_peak_bytes"])
            for value in phase_evidence_accounting.values()
        ),
        default=0,
    )
    capture = {
        "capture_bytes": capture_bytes,
        "decode_steps": int(batch.forced_decode_ids.shape[1]),
        "prefill_sample_start": max(
            0, int(batch.input_ids.shape[1]) - config.sample_last_tokens
        ),
        "prefill_sample_count": min(
            int(batch.input_ids.shape[1]), config.sample_last_tokens
        ),
        "phase_diagnostics": phase_diagnostics,
        "cached_full_prefix_verifications": verifications,
        "all_cached_logits_verified": (
            None
            if full_prefix_only
            else all(bool(value["numerically_close"]) for value in verifications)
        ),
        "all_cached_top1_verified": (
            None
            if full_prefix_only
            else all(bool(value["top1_agreement"]) for value in verifications)
        ),
        "all_cached_top1_decisions_verified": (
            None
            if full_prefix_only
            else all(bool(value["top1_decision_verified"]) for value in verifications)
        ),
        "decode_trace_mode": config.decode_trace_mode,
        "canonical_trace_source": (
            "use_cache_false_full_prefix"
            if full_prefix_only
            else "verified_one_token_cache"
        ),
        "cached_logits_accepted": not full_prefix_only,
        "cached_forward_count": 0 if full_prefix_only else 1 + len(verifications) - 1,
        "gradients_used": False,
        "training_steps": 0,
    }
    if child_evidence_profile is not None:
        capture.update(
            {
                "retained_capture_bytes": capture_bytes,
                "transient_validation_peak_bytes": (
                    maximum_transient_validation_peak_bytes
                ),
                "estimated_peak_capture_bytes": (
                    capture_bytes + maximum_transient_validation_peak_bytes
                ),
                "phase_evidence_accounting": phase_evidence_accounting,
                "child_evidence_profile": child_evidence_profile.as_dict(),
            }
        )
    return tensors, capture


def _record_for(
    torch: Any,
    manifest: Mapping[str, Any],
    batch: _PreparedBatch,
    tensors: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = _ordered_tensor_descriptors(torch, tensors)
    body = {
        "schema": BATCH_SCHEMA,
        "cache_key": manifest["cache_key"],
        "batch_index": batch.index,
        "input_sha256": batch.input_sha256,
        "tensor_count": len(inventory),
        "tensors": inventory,
        "capture": dict(capture),
    }
    return _sealed_document(body, "record_sha256")


def _publish_batch(
    torch: Any,
    batch_dir: Path,
    manifest: Mapping[str, Any],
    batch: _PreparedBatch,
    tensors: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    _safe_open, save_file = _safetensors()
    batch_dir.mkdir()
    trace_path = batch_dir / TRACE_FILENAME
    record_path = batch_dir / BATCH_RECORD_FILENAME
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(trace_path),
        metadata={
            "schema": BATCH_SCHEMA,
            "cache_key": str(manifest["cache_key"]),
            "batch_index": str(batch.index),
            "input_sha256": batch.input_sha256,
        },
    )
    with trace_path.open("r+b") as handle:
        os.fsync(handle.fileno())
    record = _record_for(torch, manifest, batch, tensors, capture)
    _write_canonical_json(record_path, record)
    return {
        "batch_index": batch.index,
        "input_sha256": batch.input_sha256,
        "trace_file_sha256": _file_sha256(trace_path),
        "record_file_sha256": _file_sha256(record_path),
        "record_sha256": record["record_sha256"],
    }


def _completion_for(
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema": COMPLETION_SCHEMA,
        "cache_key": manifest["cache_key"],
        "manifest_sha256": manifest["manifest_sha256"],
        "batch_count": len(entries),
        "batches": [dict(value) for value in entries],
    }
    return _sealed_document(body, "completion_sha256")


def _validate_manifest(value: Mapping[str, Any]) -> None:
    _verify_seal(value, "manifest_sha256", "cache manifest")
    if value.get("schema") != CACHE_SCHEMA:
        raise Lfm2CircuitTraceCorruptionError("Cache manifest schema is unsupported.")
    cache_key = value.get("cache_key")
    if not isinstance(cache_key, str) or len(cache_key) != _SHA256_LENGTH:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache manifest has an invalid cache key."
        )
    for section in (
        "parent",
        "corpus",
        "implementation",
        "layer_topology",
        "capture",
        "runtime",
        "execution",
        "storage",
    ):
        if not isinstance(value.get(section), Mapping):
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache manifest section {section!r} is invalid."
            )
    implementation_body = dict(value["implementation"])
    implementation_hash = implementation_body.pop("implementation_sha256", None)
    if _json_sha256(implementation_body) != implementation_hash:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache implementation identity is internally inconsistent."
        )
    current_implementation = implementation_identity()
    if not _compatible_implementation_identity(
        value["implementation"],
        current_implementation,
    ):
        raise Lfm2CircuitTraceCorruptionError(
            "Cache implementation differs from the current collector/runtime "
            "and is not an explicitly compatible sealed revision."
        )
    execution_body = dict(value["execution"])
    execution_hash = execution_body.pop("execution_sha256", None)
    if _json_sha256(execution_body) != execution_hash:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache execution identity is internally inconsistent."
        )
    try:
        from transformers.models.lfm2 import modeling_lfm2
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise Lfm2CircuitTraceCorruptionError(
            "Current runtime cannot import Transformers LFM2."
        ) from exc
    if execution_body.get("fast_short_conv_available") is not bool(
        getattr(modeling_lfm2, "is_fast_path_available", False)
    ):
        raise Lfm2CircuitTraceCorruptionError(
            "Cache fast short-conv availability differs from the current runtime."
        )
    corpus_body = dict(value["corpus"])
    corpus_hash = corpus_body.pop("corpus_sha256", None)
    if _json_sha256(corpus_body) != corpus_hash:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache corpus identity is internally inconsistent."
        )
    capture_body = dict(value["capture"])
    selection_hash = capture_body.pop("selection_sha256", None)
    if _json_sha256(capture_body) != selection_hash:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache capture identity is internally inconsistent."
        )
    topology = value["layer_topology"]
    topology_body = dict(topology)
    topology_hash = topology_body.pop("layer_topology_sha256", None)
    if _json_sha256(topology_body) != topology_hash:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache layer topology hash is internally inconsistent."
        )
    layers = topology.get("layers")
    if not isinstance(layers, list) or not layers:
        raise Lfm2CircuitTraceCorruptionError("Cache layer topology is empty.")
    for entry in layers:
        if not isinstance(entry, Mapping):
            raise Lfm2CircuitTraceCorruptionError(
                "Cache layer topology entry is invalid."
            )
        contract = entry.get("contract")
        if not isinstance(contract, Mapping):
            raise Lfm2CircuitTraceCorruptionError("Cache layer contract is invalid.")
        if contract.get("kind") == "short_conv":
            operator = contract.get("operator")
            if (
                not isinstance(operator, Mapping)
                or operator.get("input_projection_chunk_order") != ["b", "c", "x"]
                or operator.get("not_attention_qkv") is not True
            ):
                raise Lfm2CircuitTraceCorruptionError(
                    "Cached short-conv topology does not preserve B/C/x semantics."
                )
    parent = value["parent"]
    parent_body = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": parent.get("model_class"),
        "config_sha256": parent.get("config_sha256"),
        "state_dict_sha256": parent.get("state_dict_sha256"),
        "state_tensor_count": parent.get("state_tensor_count"),
        "runtime_buffer_sha256": parent.get("runtime_buffer_sha256"),
        "runtime_buffer_count": parent.get("runtime_buffer_count"),
    }
    if _json_sha256(parent_body) != parent.get("identity_sha256"):
        raise Lfm2CircuitTraceCorruptionError(
            "Cache parent identity is internally inconsistent."
        )
    identity = {
        "schema": CACHE_SCHEMA,
        "parent_identity_sha256": parent.get("identity_sha256"),
        "corpus_sha256": corpus_hash,
        "implementation_sha256": implementation_hash,
        "layer_topology_sha256": topology_hash,
        "selection_sha256": selection_hash,
        "execution_sha256": execution_hash,
        "runtime": dict(value["runtime"]),
    }
    if _json_sha256(identity) != cache_key:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache key does not bind its declared identities."
        )


def _validate_completion(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    _verify_seal(value, "completion_sha256", "cache completion")
    if (
        value.get("schema") != COMPLETION_SCHEMA
        or value.get("cache_key") != manifest["cache_key"]
        or value.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise Lfm2CircuitTraceCorruptionError(
            "Cache completion does not bind the manifest."
        )
    batches = value.get("batches")
    expected_count = manifest["corpus"].get("batch_count")
    if (
        not isinstance(expected_count, int)
        or not isinstance(batches, list)
        or value.get("batch_count") != expected_count
        or len(batches) != expected_count
    ):
        raise Lfm2CircuitTraceCorruptionError(
            "Cache completion batch count is invalid."
        )
    for index, entry in enumerate(batches):
        if (
            not isinstance(entry, Mapping)
            or entry.get("batch_index") != index
            or entry.get("input_sha256")
            != manifest["corpus"]["ordered_batch_input_sha256"][index]
        ):
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache completion batch {index} is invalid."
            )
        for name in (
            "trace_file_sha256",
            "record_file_sha256",
            "record_sha256",
        ):
            digest = entry.get(name)
            if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH:
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache completion batch {index} has an invalid {name}."
                )
    return tuple(batches)


def _verified_batch_paths(cache_path: Path, index: int) -> tuple[Path, Path]:
    """Authenticate one batch directory shape and return its two artifacts."""

    batch_dir = cache_path / "batches" / f"batch-{index:06d}"
    if not batch_dir.is_dir() or batch_dir.is_symlink():
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch directory {index} is missing or unsafe."
        )
    names = {path.name for path in batch_dir.iterdir()}
    if names != {TRACE_FILENAME, BATCH_RECORD_FILENAME}:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} contains unexpected artifacts."
        )
    trace_path = batch_dir / TRACE_FILENAME
    record_path = batch_dir / BATCH_RECORD_FILENAME
    if trace_path.is_symlink() or record_path.is_symlink():
        raise Lfm2CircuitTraceCorruptionError(f"Cache batch {index} contains symlinks.")
    return trace_path, record_path


def _verified_batch_record(
    record_path: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    """Hash, parse, seal-check, and rebind one committed batch record."""

    if _file_sha256(record_path) != entry["record_file_sha256"]:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} record file hash changed."
        )
    record = _read_canonical_json(record_path)
    _verify_seal(record, "record_sha256", f"cache batch {index} record")
    if (
        record.get("schema") != BATCH_SCHEMA
        or record.get("cache_key") != manifest["cache_key"]
        or record.get("batch_index") != index
        or record.get("input_sha256") != entry["input_sha256"]
        or record.get("record_sha256") != entry["record_sha256"]
    ):
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} record binding is invalid."
        )
    expected_inventory = record.get("tensors")
    if not isinstance(expected_inventory, Mapping) or not expected_inventory:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} tensor inventory is invalid."
        )
    return record


def _read_batch_trace_tensors(
    torch: Any,
    trace_path: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    index: int,
    expected_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Map the committed trace file and clone its declared tensor set."""

    safe_open, _save_file = _safetensors()
    try:
        with safe_open(str(trace_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata != {
                "schema": BATCH_SCHEMA,
                "cache_key": str(manifest["cache_key"]),
                "batch_index": str(index),
                "input_sha256": str(entry["input_sha256"]),
            }:
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} safetensors metadata is invalid."
                )
            keys = tuple(sorted(handle.keys()))
            if keys != tuple(sorted(expected_inventory)):
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} safetensors keys changed."
                )
            return {
                name: _cpu_tensor(
                    torch,
                    handle.get_tensor(name),
                    f"cache batch {index} tensor {name}",
                )
                for name in keys
            }
    except Lfm2CircuitTraceCorruptionError:
        raise
    except Exception as exc:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} safetensors file is corrupt."
        ) from exc


def _verify_batch_tensor_inventory(
    torch: Any,
    tensors: Mapping[str, Any],
    expected_inventory: Mapping[str, Any],
    index: int,
) -> None:
    """Recompute every descriptor in canonical order using the hashing pool."""

    names = sorted(tensors)
    digests = _parallel_tensor_sha256(torch, [tensors[name] for name in names])
    for name, digest in zip(names, digests):
        if (
            _tensor_descriptor(torch, tensors[name], sha256=digest)
            != expected_inventory[name]
        ):
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} tensor {name!r} changed."
            )


def _load_verified_batch(
    cache_path: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    index = int(entry["batch_index"])
    trace_path, record_path = _verified_batch_paths(cache_path, index)
    if _file_sha256(trace_path) != entry["trace_file_sha256"]:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} trace file hash changed."
        )
    record = _verified_batch_record(record_path, manifest, entry, index)
    expected_inventory = record["tensors"]
    tensors = _read_batch_trace_tensors(
        torch,
        trace_path,
        manifest,
        entry,
        index,
        expected_inventory,
    )
    _verify_batch_tensor_inventory(torch, tensors, expected_inventory, index)
    return tensors, record


def _verify_cache_layout(cache_path: Path, batch_count: int) -> None:
    """Authenticate the immutable directory shape without reading payloads."""

    expected_root = {
        MANIFEST_FILENAME,
        COMPLETION_FILENAME,
        "batches",
    }
    if {path.name for path in cache_path.iterdir()} != expected_root:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache root contains unexpected artifacts."
        )
    batches_path = cache_path / "batches"
    if not batches_path.is_dir() or batches_path.is_symlink():
        raise Lfm2CircuitTraceCorruptionError(
            "Cache batches directory is missing or unsafe."
        )
    expected_batches = {f"batch-{index:06d}" for index in range(batch_count)}
    if {path.name for path in batches_path.iterdir()} != expected_batches:
        raise Lfm2CircuitTraceCorruptionError(
            "Cache batch directory set is incomplete or modified."
        )


def _fresh_verified_cache_state(
    cache: Lfm2CircuitTraceCache,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Reauthenticate cache metadata and layout against an opened handle."""

    fresh_manifest = _read_canonical_json(cache.path / MANIFEST_FILENAME)
    fresh_completion = _read_canonical_json(cache.path / COMPLETION_FILENAME)
    _validate_manifest(fresh_manifest)
    entries = _validate_completion(fresh_completion, fresh_manifest)
    if (
        fresh_manifest["manifest_sha256"] != cache._manifest["manifest_sha256"]
        or fresh_completion["completion_sha256"]
        != cache._completion["completion_sha256"]
    ):
        raise Lfm2CircuitTraceCorruptionError(
            "Cache metadata changed after the cache was opened."
        )
    _verify_cache_layout(cache.path, cache.batch_count)
    return fresh_manifest, entries


@dataclass(frozen=True, slots=True)
class _Lfm2CircuitTraceFileIdentity:
    """Stable regular-file identity used around selective mmap reads."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(
        cls,
        value: os.stat_result,
        *,
        path: Path,
    ) -> _Lfm2CircuitTraceFileIdentity:
        if not stat.S_ISREG(value.st_mode):
            raise Lfm2CircuitTraceCorruptionError(
                f"Selective cache artifact is not a regular file: {path}."
            )
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


def _path_file_identity(path: Path) -> _Lfm2CircuitTraceFileIdentity:
    if path.is_symlink():
        raise Lfm2CircuitTraceCorruptionError(
            f"Selective cache artifact cannot be a symlink: {path}."
        )
    try:
        value = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cannot stat selective cache artifact {path}."
        ) from exc
    return _Lfm2CircuitTraceFileIdentity.from_stat(value, path=path)


def _handle_file_identity(
    handle: Any,
    *,
    path: Path,
) -> _Lfm2CircuitTraceFileIdentity:
    try:
        value = os.fstat(handle.fileno())
    except (OSError, ValueError) as exc:
        raise Lfm2CircuitTraceCorruptionError(
            f"Cannot stat authenticated cache handle {path}."
        ) from exc
    return _Lfm2CircuitTraceFileIdentity.from_stat(value, path=path)


def _optional_path_file_identity(
    path: Path,
) -> _Lfm2CircuitTraceFileIdentity | None:
    """Return a stable file identity, or ``None`` when one cannot be taken."""

    try:
        return _path_file_identity(path)
    except Lfm2CircuitTraceCorruptionError:
        return None


@dataclass(slots=True)
class _Lfm2VerifiedBatchGuard:
    """One already content-hashed trace file pinned by an open descriptor.

    The guard is established only after :func:`_load_verified_batch` has
    hashed the trace file and every sealed tensor descriptor, and only if the
    regular-file identity taken before that hashing still equals the identity
    of both the path and the pinned descriptor afterwards.  While it holds,
    repeated loads of the same batch reread the payload but skip rehashing
    bytes an open authenticated descriptor already proves immutable.
    """

    index: int
    entry: Mapping[str, Any]
    record: Mapping[str, Any]
    trace_path: Path
    trace_identity: _Lfm2CircuitTraceFileIdentity
    trace_guard: Any

    def verify(self) -> None:
        if (
            _path_file_identity(self.trace_path) != self.trace_identity
            or _handle_file_identity(self.trace_guard, path=self.trace_path)
            != self.trace_identity
        ):
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {self.index} trace identity changed after it was "
                "verified in this session."
            )

    def close(self) -> None:
        self.trace_guard.close()


def _establish_verified_batch_guard(
    cache_path: Path,
    entry: Mapping[str, Any],
    record: Mapping[str, Any],
    identity: _Lfm2CircuitTraceFileIdentity,
) -> _Lfm2VerifiedBatchGuard | None:
    """Pin a freshly verified trace file, or decline so the next load rehashes.

    Declining is the conservative outcome: it costs a redundant hash pass and
    never lets an unproven file reach the reuse path.
    """

    index = int(entry["batch_index"])
    trace_path = cache_path / "batches" / f"batch-{index:06d}" / TRACE_FILENAME
    guard: Any = None
    try:
        guard = trace_path.open("rb", buffering=0)
        if (
            _handle_file_identity(guard, path=trace_path) != identity
            or _path_file_identity(trace_path) != identity
        ):
            guard.close()
            return None
    except (OSError, ValueError, Lfm2CircuitTraceCorruptionError):
        if guard is not None:
            guard.close()
        return None
    return _Lfm2VerifiedBatchGuard(
        index=index,
        entry=MappingProxyType(dict(entry)),
        record=MappingProxyType(dict(record)),
        trace_path=trace_path,
        trace_identity=identity,
        trace_guard=guard,
    )


def _load_guarded_batch(
    cache_path: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    guard: _Lfm2VerifiedBatchGuard,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reread one batch whose bytes an open descriptor proves are unchanged.

    Only redundant hashing is skipped.  The small record document is still
    reread, rehashed, seal-checked, rebound, and required to equal the record
    verified when the guard was established.  The pinned trace identity is
    rechecked immediately before and after the safetensors mapping, exactly
    like the selective session, and fresh tensors are cloned on every call so
    callers never share retained payload storage.
    """

    torch = _torch()
    index = int(entry["batch_index"])
    _trace_path, record_path = _verified_batch_paths(cache_path, index)
    guard.verify()
    record = _verified_batch_record(record_path, manifest, entry, index)
    if dict(record) != dict(guard.record):
        raise Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} record changed after it was verified in "
            "this session."
        )
    tensors = _read_batch_trace_tensors(
        torch,
        guard.trace_path,
        manifest,
        entry,
        index,
        record["tensors"],
    )
    guard.verify()
    return tensors, record


@dataclass(frozen=True, slots=True)
class Lfm2CircuitTraceSelectiveBatchReference:
    """Tensor-free record and inventory for one authenticated cache batch."""

    index: int
    input_sha256: str
    record_sha256: str
    tensor_count: int
    ordered_tensor_names: tuple[str, ...]
    logical_bytes: int
    _record: Mapping[str, Any]

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._record)))


class Lfm2CircuitTraceSelectedBatchPayload:
    """One bounded selected-tensor view with deterministic reference release."""

    __slots__ = (
        "batch_index",
        "input_sha256",
        "logical_bytes",
        "ordered_tensor_names",
        "selection_sha256",
        "session_sha256",
        "_released",
        "_tensors",
    )

    def __init__(
        self,
        *,
        batch_index: int,
        input_sha256: str,
        ordered_tensor_names: tuple[str, ...],
        logical_bytes: int,
        session_sha256: str,
        selection_sha256: str,
        tensors: Mapping[str, Any],
    ) -> None:
        self.batch_index = batch_index
        self.input_sha256 = input_sha256
        self.ordered_tensor_names = ordered_tensor_names
        self.logical_bytes = logical_bytes
        self.session_sha256 = session_sha256
        self.selection_sha256 = selection_sha256
        self._tensors: dict[str, Any] | None = dict(tensors)
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def tensors(self) -> Mapping[str, Any]:
        if self._tensors is None:
            raise Lfm2CircuitTraceCacheError(
                "Selected cache tensor payload was released."
            )
        return MappingProxyType(self._tensors)

    def _release(self) -> None:
        if self._released:
            return
        if self._tensors is not None:
            self._tensors.clear()
        self._tensors = None
        self._released = True


@dataclass(frozen=True, slots=True)
class Lfm2CircuitTraceBatch:
    """One verified, detached CPU circuit trace."""

    index: int
    input_sha256: str
    _tensors: Mapping[str, Any]
    _record: Mapping[str, Any]

    @property
    def tensors(self) -> Mapping[str, Any]:
        return self._tensors

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._record)))

    @property
    def input_ids(self) -> Any:
        return self._tensors["input.input_ids"]

    @property
    def attention_mask(self) -> Any:
        return self._tensors["input.attention_mask"]

    @property
    def forced_decode_ids(self) -> Any:
        return self._tensors[f"input.{_FORCED_IDS}"]


@dataclass(frozen=True, slots=True)
class Lfm2ChildReplayBatch:
    """One exact, detached in-memory replay captured from the live child."""

    index: int
    input_sha256: str
    _tensors: Mapping[str, Any]
    _record: Mapping[str, Any]

    @property
    def tensors(self) -> Mapping[str, Any]:
        return self._tensors

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._record)))

    @property
    def capture(self) -> dict[str, Any]:
        return self.record["capture"]

    @property
    def phase_order(self) -> tuple[str, ...]:
        return tuple(self._record["phase_order"])

    @property
    def selected_layer_indices(self) -> tuple[int, ...]:
        return tuple(self._record["selected_layer_indices"])

    @property
    def child_evidence_profile(self) -> Lfm2ChildEvidenceProfile | None:
        value = self._record.get("child_evidence_profile")
        return None if value is None else Lfm2ChildEvidenceProfile.from_dict(value)

    @property
    def input_ids(self) -> Any:
        return self._tensors["input.input_ids"]

    @property
    def attention_mask(self) -> Any:
        return self._tensors["input.attention_mask"]

    @property
    def position_ids(self) -> Any | None:
        return self._tensors.get("input.position_ids")

    @property
    def forced_decode_ids(self) -> Any:
        return self._tensors[f"input.{_FORCED_IDS}"]


@dataclass(frozen=True, slots=True)
class Lfm2ChildEmbeddingReplay:
    """One sealed donor-time ``inputs_embeds`` trace from a live child."""

    _tensors: Mapping[str, Any]
    _record: Mapping[str, Any]

    @property
    def tensors(self) -> Mapping[str, Any]:
        return self._tensors

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._record)))

    @property
    def replay_sha256(self) -> str:
        return str(self._record["embedding_replay_sha256"])

    @property
    def child_fingerprint(self) -> str:
        return str(self._record["child"]["identity_sha256"])

    @property
    def selected_layer_indices(self) -> tuple[int, ...]:
        return tuple(self._record["selected_layer_indices"])

    def authenticate_for_inputs(
        self,
        inputs_embeds: Any,
        *,
        attention_mask: Any,
        position_ids: Any | None,
    ) -> dict[str, Any]:
        """Bind the replay to embeddings and the exact causal input geometry."""

        torch = _torch()
        record = self.record
        if record.get("schema") != CHILD_EMBEDDING_REPLAY_SCHEMA:
            raise Lfm2CircuitTraceCorruptionError(
                "Child embedding replay schema is unsupported."
            )
        _verify_seal(
            record,
            "embedding_replay_sha256",
            "child embedding replay",
        )
        inventory = record.get("tensors")
        live_names = tuple(sorted(self._tensors))
        if (
            not isinstance(inventory, Mapping)
            or record.get("tensor_count") != len(live_names)
            or tuple(sorted(inventory)) != live_names
        ):
            raise Lfm2CircuitTraceCorruptionError(
                "Child embedding replay tensor inventory differs from its "
                "sealed evidence."
            )
        for name in live_names:
            tensor = self._tensors[name]
            descriptor = inventory[name]
            if (
                not torch.is_tensor(tensor)
                or not isinstance(descriptor, Mapping)
                or _tensor_descriptor(torch, tensor) != dict(descriptor)
            ):
                raise Lfm2CircuitTraceCorruptionError(
                    "Child embedding replay tensor inventory differs from its "
                    "sealed evidence."
                )
        embeddings = _cpu_tensor(
            torch,
            inputs_embeds,
            "authenticated child trajectory embeddings",
        )
        mask = _cpu_tensor(
            torch,
            attention_mask,
            "authenticated child trajectory attention mask",
        )
        positions = (
            None
            if position_ids is None
            else _cpu_tensor(
                torch,
                position_ids,
                "authenticated child trajectory position IDs",
            )
        )
        inputs = record.get("inputs")
        if (
            embeddings.ndim != 3
            or not torch.is_floating_point(embeddings)
            or embeddings.is_complex()
            or mask.ndim != 2
            or tuple(mask.shape) != tuple(embeddings.shape[:2])
            or not _is_integer_dtype(torch, mask)
            or not bool((mask == 1).all().item())
            or (
                positions is not None
                and (
                    positions.ndim != 2
                    or tuple(positions.shape) != tuple(mask.shape)
                    or not _is_integer_dtype(torch, positions)
                    or bool((positions < 0).any().item())
                )
            )
            or not isinstance(inputs, Mapping)
            or inputs.get("inputs_embeds_sha256") != _tensor_sha256(torch, embeddings)
            or inputs.get("attention_mask_sha256") != _tensor_sha256(torch, mask)
            or inputs.get("position_ids_sha256")
            != (None if positions is None else _tensor_sha256(torch, positions))
            or inputs.get("batch_size") != int(embeddings.shape[0])
            or inputs.get("trajectory_length") != int(embeddings.shape[1])
            or inputs.get("hidden_size") != int(embeddings.shape[2])
            or inputs.get("runtime_dtype")
            != str(embeddings.dtype).removeprefix("torch.")
        ):
            raise Lfm2CircuitTraceCorruptionError(
                "Child embedding replay belongs to a different donor trajectory "
                "or causal input geometry."
            )
        return record


@dataclass(frozen=True, slots=True)
class Lfm2ChildReplay:
    """Hash-bound selected-layer traces held only in process memory."""

    _batches: tuple[Lfm2ChildReplayBatch, ...]
    _metadata: Mapping[str, Any]
    # Wall-clock accounting for the capture that produced this replay.  It is
    # deliberately outside the sealed metadata: identical inputs must produce
    # identical hashes and canonical JSON regardless of how long they took.
    _capture_timing: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
    )

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._metadata)))

    @property
    def capture_timing(self) -> dict[str, Any]:
        """Non-sealed replay/hash/seal seconds and host-transfer counters."""

        return json.loads(json.dumps(dict(self._capture_timing)))

    @property
    def replay_sha256(self) -> str:
        return str(self._metadata["replay_sha256"])

    @property
    def child_fingerprint(self) -> str:
        return str(self._metadata["child"]["identity_sha256"])

    @property
    def corpus_fingerprint(self) -> str:
        return str(self._metadata["corpus"]["corpus_sha256"])

    @property
    def layer_topology_fingerprint(self) -> str:
        return str(self._metadata["layer_topology"]["layer_topology_sha256"])

    @property
    def selected_layer_indices(self) -> tuple[int, ...]:
        return tuple(self._metadata["selection"]["selected_layer_indices"])

    @property
    def child_evidence_profile(self) -> Lfm2ChildEvidenceProfile | None:
        value = self._metadata["selection"].get("child_evidence_profile")
        return None if value is None else Lfm2ChildEvidenceProfile.from_dict(value)

    @property
    def batch_count(self) -> int:
        return len(self._batches)

    @property
    def batches(self) -> tuple[Lfm2ChildReplayBatch, ...]:
        return self._batches

    def __len__(self) -> int:
        return self.batch_count

    def load_batch(self, index: int) -> Lfm2ChildReplayBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("batch index must be an integer.")
        if index < 0 or index >= self.batch_count:
            raise IndexError(
                f"batch index must be in [0, {self.batch_count}), got {index}."
            )
        return self._batches[index]

    def iter_batches(self) -> Iterator[Lfm2ChildReplayBatch]:
        return iter(self._batches)


@dataclass(frozen=True, slots=True)
class Lfm2CircuitTraceCache:
    """Read-only handle for a verified immutable LFM2 circuit cache."""

    path: Path
    _manifest: Mapping[str, Any]
    _completion: Mapping[str, Any]
    # Per-handle reuse state.  It holds no cache content, so it is excluded
    # from equality and repr and never reaches any sealed document.
    _verified_batch_guards: dict[int, _Lfm2VerifiedBatchGuard] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        verify: bool = True,
    ) -> Lfm2CircuitTraceCache:
        cache_path = Path(path).expanduser().absolute()
        if not cache_path.is_dir() or cache_path.is_symlink():
            raise Lfm2CircuitTraceCorruptionError(
                f"Trace cache directory does not exist or is unsafe: {cache_path}."
            )
        manifest = _read_canonical_json(cache_path / MANIFEST_FILENAME)
        completion = _read_canonical_json(cache_path / COMPLETION_FILENAME)
        _validate_manifest(manifest)
        _validate_completion(completion, manifest)
        result = cls(
            path=cache_path,
            _manifest=MappingProxyType(manifest),
            _completion=MappingProxyType(completion),
        )
        if verify:
            result.verify()
        return result

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._manifest)))

    @property
    def cache_key(self) -> str:
        return str(self._manifest["cache_key"])

    @property
    def parent_fingerprint(self) -> str:
        return str(self._manifest["parent"]["identity_sha256"])

    @property
    def corpus_fingerprint(self) -> str:
        return str(self._manifest["corpus"]["corpus_sha256"])

    @property
    def implementation_fingerprint(self) -> str:
        return str(self._manifest["implementation"]["implementation_sha256"])

    @property
    def layer_topology_fingerprint(self) -> str:
        return str(self._manifest["layer_topology"]["layer_topology_sha256"])

    @property
    def batch_count(self) -> int:
        return int(self._manifest["corpus"]["batch_count"])

    def __len__(self) -> int:
        return self.batch_count

    @property
    def verified_batch_guard_count(self) -> int:
        """Number of batches this handle can currently reload without rehashing."""

        return len(self._verified_batch_guards)

    def release_verified_batch_guards(self) -> None:
        """Close every descriptor pinned by verified-once batch reuse."""

        for guard in tuple(self._verified_batch_guards.values()):
            guard.close()
        self._verified_batch_guards.clear()

    def _drop_verified_batch_guard(self, index: int) -> None:
        guard = self._verified_batch_guards.pop(index, None)
        if guard is not None:
            guard.close()

    def _retain_verified_batch_guard(
        self,
        guard: _Lfm2VerifiedBatchGuard,
    ) -> None:
        self._drop_verified_batch_guard(guard.index)
        self._verified_batch_guards[guard.index] = guard
        while len(self._verified_batch_guards) > _VERIFIED_BATCH_GUARD_LIMIT:
            oldest = next(iter(self._verified_batch_guards))
            self._verified_batch_guards.pop(oldest).close()

    def verify(self) -> Lfm2CircuitTraceCache:
        fresh_manifest, entries = _fresh_verified_cache_state(self)
        for entry in entries:
            _load_verified_batch(self.path, fresh_manifest, entry)
        return self

    def load_batch(self, index: int) -> Lfm2CircuitTraceBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("batch index must be an integer.")
        if index < 0 or index >= self.batch_count:
            raise IndexError(
                f"batch index must be in [0, {self.batch_count}), got {index}."
            )
        fresh_completion = _read_canonical_json(self.path / COMPLETION_FILENAME)
        entries = _validate_completion(fresh_completion, self._manifest)
        entry = entries[index]
        guard = self._verified_batch_guards.get(index)
        if guard is not None and dict(guard.entry) != dict(entry):
            self._drop_verified_batch_guard(index)
            guard = None
        if guard is None:
            trace_path = self.path / "batches" / f"batch-{index:06d}" / TRACE_FILENAME
            identity = _optional_path_file_identity(trace_path)
            tensors, record = _load_verified_batch(self.path, self._manifest, entry)
            if identity is not None:
                established = _establish_verified_batch_guard(
                    self.path,
                    entry,
                    record,
                    identity,
                )
                if established is not None:
                    self._retain_verified_batch_guard(established)
        else:
            tensors, record = _load_guarded_batch(
                self.path,
                self._manifest,
                entry,
                guard,
            )
        return Lfm2CircuitTraceBatch(
            index=index,
            input_sha256=str(record["input_sha256"]),
            _tensors=MappingProxyType(tensors),
            _record=MappingProxyType(record),
        )

    def iter_batches(self) -> Iterator[Lfm2CircuitTraceBatch]:
        for index in range(self.batch_count):
            yield self.load_batch(index)

    def authenticated_session(
        self,
    ) -> Lfm2CircuitTraceAuthenticatedSession:
        """Open a one-pass payload-verifying view of this sealed cache."""

        return Lfm2CircuitTraceAuthenticatedSession(self)

    def selective_authenticated_session(
        self,
    ) -> Lfm2CircuitTraceSelectiveAuthenticatedSession:
        """Hash-authenticate each file once, then permit selected tensor reads."""

        return Lfm2CircuitTraceSelectiveAuthenticatedSession(self)


class Lfm2CircuitTraceAuthenticatedSession:
    """Verify one sealed cache exactly as its payloads are consumed.

    A normal :meth:`Lfm2CircuitTraceCache.verify` intentionally scans every
    batch independently of later consumers.  That remains the conservative
    compatibility API.  Production folding instead needs the authenticated
    tensors themselves, so scanning an 85-GB cache and then loading it again
    is pure duplicate I/O.

    This session authenticates the same manifest, completion document,
    directory layout, trace-file hash, record-file hash, safetensors metadata,
    tensor inventory, and per-tensor descriptors.  The only difference is
    timing: each payload is verified once, in canonical batch order, at the
    moment it is returned to the caller.  The final batch reauthenticates the
    metadata and layout and seals a deterministic session receipt.  Consumed
    batches cannot be loaded again through the session, preventing accidental
    duplicate payload scans while allowing downstream code to retain the
    returned detached tensors.
    """

    __slots__ = (
        "_cache",
        "_entries",
        "_evidence",
        "_next_batch_index",
        "_ordered_verified_batches",
    )

    def __init__(self, cache: Lfm2CircuitTraceCache) -> None:
        if not isinstance(cache, Lfm2CircuitTraceCache):
            raise TypeError("cache must be Lfm2CircuitTraceCache.")
        _fresh_manifest, entries = _fresh_verified_cache_state(cache)
        self._cache = cache
        self._entries = tuple(MappingProxyType(dict(entry)) for entry in entries)
        self._next_batch_index = 0
        self._ordered_verified_batches: list[dict[str, Any]] = []
        self._evidence: Mapping[str, Any] | None = None

    @property
    def path(self) -> Path:
        return self._cache.path

    @property
    def manifest(self) -> dict[str, Any]:
        return self._cache.manifest

    @property
    def cache_key(self) -> str:
        return self._cache.cache_key

    @property
    def parent_fingerprint(self) -> str:
        return self._cache.parent_fingerprint

    @property
    def corpus_fingerprint(self) -> str:
        return self._cache.corpus_fingerprint

    @property
    def implementation_fingerprint(self) -> str:
        return self._cache.implementation_fingerprint

    @property
    def layer_topology_fingerprint(self) -> str:
        return self._cache.layer_topology_fingerprint

    @property
    def batch_count(self) -> int:
        return self._cache.batch_count

    @property
    def verified_batch_count(self) -> int:
        return self._next_batch_index

    @property
    def complete(self) -> bool:
        return self._evidence is not None

    @property
    def evidence_sha256(self) -> str:
        if self._evidence is None:
            raise Lfm2CircuitTraceCacheError(
                "Authenticated cache session is not complete."
            )
        return str(self._evidence["session_sha256"])

    def __len__(self) -> int:
        return self.batch_count

    def _completion_entries(self) -> tuple[Mapping[str, Any], ...]:
        fresh_completion = _read_canonical_json(self.path / COMPLETION_FILENAME)
        entries = _validate_completion(
            fresh_completion,
            self._cache._manifest,
        )
        if (
            fresh_completion["completion_sha256"]
            != self._cache._completion["completion_sha256"]
        ):
            raise Lfm2CircuitTraceCorruptionError(
                "Cache completion changed during authenticated consumption."
            )
        return entries

    def load_batch(self, index: int) -> Lfm2CircuitTraceBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("batch index must be an integer.")
        if index < 0 or index >= self.batch_count:
            raise IndexError(
                f"batch index must be in [0, {self.batch_count}), got {index}."
            )
        if index != self._next_batch_index:
            raise Lfm2CircuitTraceCacheError(
                "Authenticated cache sessions require each batch exactly "
                "once in canonical order; expected batch "
                f"{self._next_batch_index}, got {index}."
            )
        fresh_entries = self._completion_entries()
        entry = fresh_entries[index]
        if dict(entry) != dict(self._entries[index]):
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} completion binding changed."
            )
        tensors, record = _load_verified_batch(
            self.path,
            self._cache._manifest,
            entry,
        )
        batch = Lfm2CircuitTraceBatch(
            index=index,
            input_sha256=str(record["input_sha256"]),
            _tensors=MappingProxyType(tensors),
            _record=MappingProxyType(record),
        )
        self._ordered_verified_batches.append(
            {
                "batch_index": index,
                "input_sha256": str(entry["input_sha256"]),
                "trace_file_sha256": str(entry["trace_file_sha256"]),
                "record_file_sha256": str(entry["record_file_sha256"]),
                "record_sha256": str(entry["record_sha256"]),
            }
        )
        self._next_batch_index += 1
        if self._next_batch_index == self.batch_count:
            self._finalize()
        return batch

    def iter_batches(self) -> Iterator[Lfm2CircuitTraceBatch]:
        while self._next_batch_index < self.batch_count:
            yield self.load_batch(self._next_batch_index)

    def _finalize(self) -> None:
        if self._evidence is not None:
            return
        if self._next_batch_index != self.batch_count:
            raise Lfm2CircuitTraceCacheError(
                "Authenticated cache session cannot finalize before every "
                "batch has been consumed."
            )
        _fresh_manifest, entries = _fresh_verified_cache_state(self._cache)
        if tuple(dict(entry) for entry in entries) != tuple(
            dict(entry) for entry in self._entries
        ):
            raise Lfm2CircuitTraceCorruptionError(
                "Cache completion entries changed during authenticated consumption."
            )
        body = {
            "schema": AUTHENTICATED_SESSION_SCHEMA,
            "cache_key": self.cache_key,
            "manifest_sha256": str(self._cache._manifest["manifest_sha256"]),
            "completion_sha256": str(self._cache._completion["completion_sha256"]),
            "batch_count": self.batch_count,
            "ordered_verified_batches": list(self._ordered_verified_batches),
            "authentication_contract": {
                "manifest_verified_before_payload_consumption": True,
                "completion_verified_before_each_payload": True,
                "directory_layout_verified_before_and_after": True,
                "trace_file_sha256_verified_at_load": True,
                "record_file_sha256_verified_at_load": True,
                "record_self_hash_verified_at_load": True,
                "safetensors_metadata_verified_at_load": True,
                "tensor_inventory_verified_at_load": True,
                "tensor_descriptors_verified_at_load": True,
                "canonical_batch_order_required": True,
                "duplicate_session_payload_loads_permitted": False,
                "parent_forward_calls": 0,
                "training_steps": 0,
            },
        }
        self._evidence = MappingProxyType(
            {
                **body,
                "session_sha256": _json_sha256(body),
            }
        )

    def verify(self) -> Lfm2CircuitTraceAuthenticatedSession:
        """Finalize an already consumed session without rereading payloads."""

        self._finalize()
        return self

    def as_dict(self) -> dict[str, Any]:
        if self._evidence is None:
            raise Lfm2CircuitTraceCacheError(
                "Authenticated cache session is not complete."
            )
        return json.loads(_canonical_json(dict(self._evidence)))


@dataclass(slots=True)
class _Lfm2SelectiveBatchState:
    entry: Mapping[str, Any]
    record: Mapping[str, Any]
    reference: Lfm2CircuitTraceSelectiveBatchReference
    trace_path: Path
    record_path: Path
    trace_identity: _Lfm2CircuitTraceFileIdentity
    record_identity: _Lfm2CircuitTraceFileIdentity
    trace_guard: Any


class Lfm2CircuitTraceSelectiveAuthenticatedSession:
    """One immutable cache authentication with repeated bounded selections.

    Every committed trace and record file is content-hashed exactly once while
    this session opens.  The corresponding authenticated regular file
    descriptor remains open for the session lifetime.  A safetensors mapping
    is opened only while a caller-selected tensor set is cloned into bounded
    CPU storage, then closed immediately so released layers do not remain
    physically resident through pinned mappings.  Every selected tensor is
    compared to its sealed descriptor.  Stable path and handle identities are
    checked before and after every selection, so replacement or in-place
    mutation fails closed without rescanning unrelated payload bytes.

    The cross-platform TOCTOU boundary is explicit: ``safe_open`` accepts a
    path rather than an existing descriptor.  This implementation requires
    the path identity to equal the already-hashed guard identity immediately
    before and after every ``safe_open`` while keeping the authenticated raw
    descriptor pinned.  It detects normal concurrent replacement and
    mutation.  It does not claim protection from a privileged hostile
    filesystem that swaps and restores the same path entirely inside that
    narrow open/stat interval.
    """

    __slots__ = (
        "_batch_states",
        "_cache",
        "_closed",
        "_evidence",
        "_lock",
        "_manifest_identity",
        "_completion_identity",
    )

    def __init__(self, cache: Lfm2CircuitTraceCache) -> None:
        if not isinstance(cache, Lfm2CircuitTraceCache):
            raise TypeError("cache must be Lfm2CircuitTraceCache.")
        self._cache = cache
        self._closed = False
        self._lock = threading.RLock()
        self._batch_states: tuple[_Lfm2SelectiveBatchState, ...] = ()
        self._evidence: Mapping[str, Any] | None = None
        manifest_path = cache.path / MANIFEST_FILENAME
        completion_path = cache.path / COMPLETION_FILENAME
        self._manifest_identity = _path_file_identity(manifest_path)
        self._completion_identity = _path_file_identity(completion_path)
        fresh_manifest, entries = _fresh_verified_cache_state(cache)
        states: list[_Lfm2SelectiveBatchState] = []
        try:
            for entry in entries:
                states.append(
                    self._open_batch_state(
                        fresh_manifest,
                        entry,
                    )
                )
            self._batch_states = tuple(states)
            self._verify_immutable_metadata(entries)
            self._seal_evidence()
        except BaseException:
            for state in reversed(states):
                self._close_batch_state(state)
            raise

    @property
    def path(self) -> Path:
        return self._cache.path

    @property
    def manifest(self) -> dict[str, Any]:
        return self._cache.manifest

    @property
    def cache_key(self) -> str:
        return self._cache.cache_key

    @property
    def batch_count(self) -> int:
        return self._cache.batch_count

    @property
    def evidence_sha256(self) -> str:
        if self._evidence is None:  # pragma: no cover - constructor seals
            raise Lfm2CircuitTraceCacheError(
                "Selective authenticated session is not sealed."
            )
        return str(self._evidence["session_sha256"])

    @property
    def batch_references(
        self,
    ) -> tuple[Lfm2CircuitTraceSelectiveBatchReference, ...]:
        return tuple(state.reference for state in self._batch_states)

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        return self.batch_count

    def _open_batch_state(
        self,
        manifest: Mapping[str, Any],
        entry: Mapping[str, Any],
    ) -> _Lfm2SelectiveBatchState:
        safe_open, _save_file = _safetensors()
        index = int(entry["batch_index"])
        batch_dir = self.path / "batches" / f"batch-{index:06d}"
        if not batch_dir.is_dir() or batch_dir.is_symlink():
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch directory {index} is missing or unsafe."
            )
        if {path.name for path in batch_dir.iterdir()} != {
            TRACE_FILENAME,
            BATCH_RECORD_FILENAME,
        }:
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} contains unexpected artifacts."
            )
        trace_path = batch_dir / TRACE_FILENAME
        record_path = batch_dir / BATCH_RECORD_FILENAME
        trace_identity = _path_file_identity(trace_path)
        record_identity = _path_file_identity(record_path)
        if _file_sha256(trace_path) != entry["trace_file_sha256"]:
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} trace file hash changed."
            )
        if _path_file_identity(trace_path) != trace_identity:
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} trace changed while it was hashed."
            )
        if _file_sha256(record_path) != entry["record_file_sha256"]:
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} record file hash changed."
            )
        if _path_file_identity(record_path) != record_identity:
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} record changed while it was hashed."
            )
        record = _read_canonical_json(record_path)
        _verify_seal(record, "record_sha256", f"cache batch {index} record")
        if (
            record.get("schema") != BATCH_SCHEMA
            or record.get("cache_key") != manifest["cache_key"]
            or record.get("batch_index") != index
            or record.get("input_sha256") != entry["input_sha256"]
            or record.get("record_sha256") != entry["record_sha256"]
        ):
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} record binding is invalid."
            )
        inventory = record.get("tensors")
        if not isinstance(inventory, Mapping) or not inventory:
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} tensor inventory is invalid."
            )
        expected_names = tuple(sorted(inventory))
        guard = None
        try:
            guard = trace_path.open("rb", buffering=0)
            if _handle_file_identity(guard, path=trace_path) != trace_identity:
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} trace guard changed identity."
                )
            if _path_file_identity(trace_path) != trace_identity:
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} trace changed before safetensors open."
                )
            with safe_open(
                str(trace_path),
                framework="pt",
                device="cpu",
            ) as handle:
                metadata = handle.metadata() or {}
                if metadata != {
                    "schema": BATCH_SCHEMA,
                    "cache_key": str(manifest["cache_key"]),
                    "batch_index": str(index),
                    "input_sha256": str(entry["input_sha256"]),
                }:
                    raise Lfm2CircuitTraceCorruptionError(
                        f"Cache batch {index} safetensors metadata is invalid."
                    )
                if tuple(sorted(handle.keys())) != expected_names:
                    raise Lfm2CircuitTraceCorruptionError(
                        f"Cache batch {index} safetensors keys changed."
                    )
            if _path_file_identity(trace_path) != trace_identity:
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} trace changed during safetensors open."
                )
        except BaseException:
            if guard is not None:
                guard.close()
            raise
        try:
            logical_bytes = 0
            for name in expected_names:
                descriptor = inventory[name]
                if not isinstance(descriptor, Mapping):
                    raise Lfm2CircuitTraceCorruptionError(
                        f"Cache batch {index} tensor descriptor {name!r} is invalid."
                    )
                size_bytes = descriptor.get("size_bytes")
                if (
                    isinstance(size_bytes, bool)
                    or not isinstance(size_bytes, int)
                    or size_bytes <= 0
                ):
                    raise Lfm2CircuitTraceCorruptionError(
                        f"Cache batch {index} tensor {name!r} has invalid bytes."
                    )
                logical_bytes += size_bytes
        except BaseException:
            guard.close()
            raise
        reference = Lfm2CircuitTraceSelectiveBatchReference(
            index=index,
            input_sha256=str(entry["input_sha256"]),
            record_sha256=str(entry["record_sha256"]),
            tensor_count=len(expected_names),
            ordered_tensor_names=expected_names,
            logical_bytes=logical_bytes,
            _record=MappingProxyType(record),
        )
        return _Lfm2SelectiveBatchState(
            entry=MappingProxyType(dict(entry)),
            record=MappingProxyType(record),
            reference=reference,
            trace_path=trace_path,
            record_path=record_path,
            trace_identity=trace_identity,
            record_identity=record_identity,
            trace_guard=guard,
        )

    @staticmethod
    def _close_batch_state(state: _Lfm2SelectiveBatchState) -> None:
        state.trace_guard.close()

    def _verify_batch_identity(
        self,
        state: _Lfm2SelectiveBatchState,
    ) -> None:
        index = state.reference.index
        if (
            _path_file_identity(state.trace_path) != state.trace_identity
            or _handle_file_identity(
                state.trace_guard,
                path=state.trace_path,
            )
            != state.trace_identity
        ):
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} trace identity changed."
            )
        if _path_file_identity(state.record_path) != state.record_identity:
            raise Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} record identity changed."
            )

    def _verify_immutable_metadata(
        self,
        expected_entries: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if (
            _path_file_identity(self.path / MANIFEST_FILENAME)
            != self._manifest_identity
            or _path_file_identity(self.path / COMPLETION_FILENAME)
            != self._completion_identity
        ):
            raise Lfm2CircuitTraceCorruptionError(
                "Cache metadata identity changed during selective session."
            )
        _fresh_manifest, entries = _fresh_verified_cache_state(self._cache)
        expected = (
            tuple(state.entry for state in self._batch_states)
            if expected_entries is None
            else tuple(expected_entries)
        )
        if tuple(dict(entry) for entry in entries) != tuple(
            dict(entry) for entry in expected
        ):
            raise Lfm2CircuitTraceCorruptionError(
                "Cache completion entries changed during selective session."
            )
        for state in self._batch_states:
            self._verify_batch_identity(state)

    def _seal_evidence(self) -> None:
        body = {
            "schema": SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
            "cache_key": self.cache_key,
            "manifest_sha256": str(self._cache._manifest["manifest_sha256"]),
            "completion_sha256": str(self._cache._completion["completion_sha256"]),
            "batch_count": self.batch_count,
            "ordered_authenticated_batches": [
                {
                    "batch_index": state.reference.index,
                    "input_sha256": state.reference.input_sha256,
                    "trace_file_sha256": str(state.entry["trace_file_sha256"]),
                    "record_file_sha256": str(state.entry["record_file_sha256"]),
                    "record_sha256": state.reference.record_sha256,
                    "tensor_count": state.reference.tensor_count,
                    "logical_bytes": state.reference.logical_bytes,
                }
                for state in self._batch_states
            ],
            "authentication_contract": {
                "committed_local_cache_only": True,
                "manifest_completion_and_layout_verified": True,
                "trace_file_sha256_passes_per_session": 1,
                "record_file_sha256_passes_per_session": 1,
                "record_self_hash_verified": True,
                "safetensors_metadata_verified": True,
                "complete_tensor_key_inventory_verified": True,
                "selected_tensor_descriptors_verified_per_load": True,
                "stable_file_identity_checked_before_and_after_load": True,
                "authenticated_trace_guard_held_open": True,
                "safetensors_mapping_held_open": False,
                "safetensors_mapping_opened_per_selection_and_closed": True,
                "unselected_tensor_payloads_loaded": False,
                "parent_forward_calls": 0,
                "training_steps": 0,
                "toctou_boundary": (
                    "privileged hostile filesystem replacement entirely "
                    "inside the cross-platform safe_open path/stat interval"
                ),
            },
        }
        self._evidence = MappingProxyType(
            {
                **body,
                "session_sha256": _json_sha256(body),
            }
        )

    def _state(self, index: int) -> _Lfm2SelectiveBatchState:
        if self._closed:
            raise Lfm2CircuitTraceCacheError(
                "Selective authenticated session is closed."
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("batch index must be an integer.")
        if index < 0 or index >= self.batch_count:
            raise IndexError(
                f"batch index must be in [0, {self.batch_count}), got {index}."
            )
        return self._batch_states[index]

    def load_selected_tensors(
        self,
        index: int,
        tensor_names: Sequence[str],
    ) -> Lfm2CircuitTraceSelectedBatchPayload:
        """Load exactly one canonical selected tensor set from one batch."""

        if isinstance(tensor_names, (str, bytes)) or not isinstance(
            tensor_names,
            Sequence,
        ):
            raise TypeError("tensor_names must be a sequence of strings.")
        names = tuple(tensor_names)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or names != tuple(sorted(set(names)))
        ):
            raise Lfm2CircuitTraceCacheError(
                "Selected tensor names must be nonempty, unique, and in "
                "canonical sorted order."
            )
        with self._lock:
            state = self._state(index)
            inventory = state.record["tensors"]
            if any(name not in inventory for name in names):
                missing = next(name for name in names if name not in inventory)
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} has no sealed tensor {missing!r}."
                )
            self._verify_batch_identity(state)
            torch = _torch()
            safe_open, _save_file = _safetensors()
            tensors: dict[str, Any] = {}
            try:
                with safe_open(
                    str(state.trace_path),
                    framework="pt",
                    device="cpu",
                ) as handle:
                    metadata = handle.metadata() or {}
                    if metadata != {
                        "schema": BATCH_SCHEMA,
                        "cache_key": self.cache_key,
                        "batch_index": str(index),
                        "input_sha256": state.reference.input_sha256,
                    }:
                        raise Lfm2CircuitTraceCorruptionError(
                            f"Cache batch {index} safetensors metadata changed."
                        )
                    if (
                        tuple(sorted(handle.keys()))
                        != state.reference.ordered_tensor_names
                    ):
                        raise Lfm2CircuitTraceCorruptionError(
                            f"Cache batch {index} safetensors keys changed."
                        )
                    self._verify_batch_identity(state)
                    for name in names:
                        tensor = _cpu_tensor(
                            torch,
                            handle.get_tensor(name),
                            f"cache batch {index} selected tensor {name}",
                        )
                        if _tensor_descriptor(torch, tensor) != inventory[name]:
                            raise Lfm2CircuitTraceCorruptionError(
                                f"Cache batch {index} selected tensor {name!r} changed."
                            )
                        tensors[name] = tensor
                self._verify_batch_identity(state)
            except BaseException:
                tensors.clear()
                raise
            logical_bytes = sum(
                int(value.numel() * value.element_size()) for value in tensors.values()
            )
            expected_bytes = sum(int(inventory[name]["size_bytes"]) for name in names)
            if logical_bytes != expected_bytes:
                tensors.clear()
                raise Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} selected tensor byte count changed."
                )
            body = {
                "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
                "session_sha256": self.evidence_sha256,
                "cache_key": self.cache_key,
                "batch_index": index,
                "input_sha256": state.reference.input_sha256,
                "record_sha256": state.reference.record_sha256,
                "ordered_tensor_names": list(names),
                "ordered_tensor_descriptors": [dict(inventory[name]) for name in names],
                "logical_bytes": logical_bytes,
            }
            return Lfm2CircuitTraceSelectedBatchPayload(
                batch_index=index,
                input_sha256=state.reference.input_sha256,
                ordered_tensor_names=names,
                logical_bytes=logical_bytes,
                session_sha256=self.evidence_sha256,
                selection_sha256=_json_sha256(body),
                tensors=tensors,
            )

    def release_selected_tensors(
        self,
        payload: Lfm2CircuitTraceSelectedBatchPayload,
    ) -> None:
        if not isinstance(payload, Lfm2CircuitTraceSelectedBatchPayload):
            raise TypeError("payload must be Lfm2CircuitTraceSelectedBatchPayload.")
        if payload.session_sha256 != self.evidence_sha256:
            raise Lfm2CircuitTraceCacheError(
                "Selected payload belongs to another authenticated session."
            )
        payload._release()

    def verify(
        self,
    ) -> Lfm2CircuitTraceSelectiveAuthenticatedSession:
        """Recheck identities and metadata without rereading trace contents."""

        with self._lock:
            if self._closed:
                raise Lfm2CircuitTraceCacheError(
                    "Selective authenticated session is closed."
                )
            self._verify_immutable_metadata()
        return self

    def as_dict(self) -> dict[str, Any]:
        if self._evidence is None:  # pragma: no cover - constructor seals
            raise Lfm2CircuitTraceCacheError(
                "Selective authenticated session is not sealed."
            )
        return json.loads(_canonical_json(dict(self._evidence)))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            error: BaseException | None = None
            try:
                self._verify_immutable_metadata()
            except BaseException as exc:
                error = exc
            for state in reversed(self._batch_states):
                try:
                    self._close_batch_state(state)
                except BaseException as exc:
                    if error is None:
                        error = exc
                    else:
                        error.add_note(
                            "Selective session close also failed: "
                            f"{type(exc).__qualname__}: {exc}"
                        )
            self._closed = True
            if error is not None:
                raise error

    def __enter__(
        self,
    ) -> Lfm2CircuitTraceSelectiveAuthenticatedSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()


def open_lfm2_circuit_trace_cache(
    path: str | os.PathLike[str],
    *,
    verify: bool = True,
) -> Lfm2CircuitTraceCache:
    """Open a completed local LFM2 circuit cache without its parent."""

    return Lfm2CircuitTraceCache.open(path, verify=verify)


def _safe_remove_stage(stage: Path, root: Path) -> None:
    try:
        resolved_stage = stage.resolve()
        resolved_root = root.resolve()
    except OSError:
        return
    if (
        resolved_stage.parent == resolved_root
        and stage.name.startswith(".lfm2-circuit-stage-")
        and stage.is_dir()
        and not stage.is_symlink()
    ):
        shutil.rmtree(stage)


def _same_capture_request(
    stored: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> bool:
    for section in (
        "parent",
        "corpus",
        "layer_topology",
        "capture",
        "runtime",
        "execution",
        "storage",
    ):
        if stored.get(section) != requested.get(section):
            return False
    stored_implementation = stored.get("implementation")
    requested_implementation = requested.get("implementation")
    return (
        isinstance(stored_implementation, Mapping)
        and isinstance(requested_implementation, Mapping)
        and _compatible_implementation_identity(
            stored_implementation,
            requested_implementation,
        )
    )


def _find_compatible_committed_cache(
    root: Path,
    requested_manifest: Mapping[str, Any],
) -> Lfm2CircuitTraceCache | None:
    """Reuse a sealed cache only through an exact directional policy receipt."""

    for candidate in sorted(root.iterdir(), key=lambda value: value.name):
        if (
            not candidate.is_dir()
            or candidate.is_symlink()
            or candidate.name.startswith(".")
        ):
            continue
        manifest_path = candidate / MANIFEST_FILENAME
        completion_path = candidate / COMPLETION_FILENAME
        if not manifest_path.is_file() or not completion_path.is_file():
            continue
        try:
            stored_manifest = _read_canonical_json(manifest_path)
        except Lfm2CircuitTraceCacheError:
            continue
        if not _same_capture_request(stored_manifest, requested_manifest):
            continue
        return open_lfm2_circuit_trace_cache(candidate, verify=True)
    return None


def _exact_replay_batch_mappings(
    values: Iterable[Mapping[str, Any] | Lfm2CircuitTraceBatch | Lfm2ChildReplayBatch],
) -> Iterator[Mapping[str, Any]]:
    """Adapt already-tokenized trace batches without consulting a tokenizer."""

    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            yield value
            continue
        if not isinstance(value, (Lfm2CircuitTraceBatch, Lfm2ChildReplayBatch)):
            raise Lfm2CircuitTraceCaptureError(
                f"Replay batch {index} must be a token mapping or an LFM2 "
                "circuit-trace batch."
            )
        mapping = {
            "input_ids": value.input_ids,
            "attention_mask": value.attention_mask,
            _FORCED_IDS: value.forced_decode_ids,
        }
        position_ids = value.tensors.get("input.position_ids")
        if position_ids is not None:
            mapping["position_ids"] = position_ids
        yield mapping


def _child_replay_phase_rows(
    batch: _PreparedBatch,
    config: Lfm2CircuitTraceCaptureConfig,
) -> tuple[dict[str, Any], ...]:
    prompt_length = int(batch.input_ids.shape[1])
    rows: list[dict[str, Any]] = [
        {
            "phase": "prefill",
            "prefix_length": prompt_length,
            "sample_start": max(0, prompt_length - config.sample_last_tokens),
            "sample_count": min(prompt_length, config.sample_last_tokens),
            "forced_decode_step": None,
        }
    ]
    for step_index in range(int(batch.forced_decode_ids.shape[1])):
        prefix_length = prompt_length + step_index + 1
        rows.append(
            {
                "phase": f"decode.{step_index:03d}",
                "prefix_length": prefix_length,
                "sample_start": prefix_length - 1,
                "sample_count": 1,
                "forced_decode_step": step_index,
            }
        )
    return tuple(rows)


def _validate_child_replay_mode(value: str) -> str:
    if value not in {
        CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    }:
        raise ValueError(
            "replay_capture_mode must be 'prefix_by_prefix_canonical' or "
            "'single_pass_causal'."
        )
    return value


def _heterogeneous_single_pass_runtime_binding(
    model: Any,
    execution: Mapping[str, Any],
    floating_dtypes: Sequence[str],
    floating_parameter_dtypes: Sequence[str],
) -> dict[str, Any] | None:
    """Authenticate the exact descriptor-native heterogeneous runtime.

    Heterogeneous children retain the canonical Transformers root, decoder,
    causal mask, residual, and MLP/attention/ShortConv mathematics, while the
    physically variable operators are exact native-derived subclasses.  This
    binding admits only the materializer-owned classes after its complete
    physical audit has reauthenticated the live child.  It also pins the
    descriptor, materializer source, method identities, FP32 execution, and
    receipt so an arbitrary look-alike subclass cannot enter the causal fast
    path.
    """

    if execution.get("sealed_heterogeneous_dynamic_runtime") is not True:
        return None
    if tuple(floating_parameter_dtypes) != ("float32",):
        raise Lfm2CircuitTraceCaptureError(
            "Heterogeneous single-pass replay requires the authenticated "
            "dense FP32 materialization dtype."
        )
    try:
        from . import hf_lfm2_dynamic_blank_materializer as materializer
        from .hf_lfm2_dynamic_physical_layout import (
            Lfm2HeterogeneousPhysicalLayoutDescriptor,
        )

        layout = materializer.extract_lfm2_dynamic_physical_layout_from_config(
            model.config
        )
    except Exception as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Heterogeneous single-pass replay could not authenticate its "
            "embedded physical descriptor."
        ) from exc
    if type(layout) is not Lfm2HeterogeneousPhysicalLayoutDescriptor:
        raise Lfm2CircuitTraceCaptureError(
            "Heterogeneous single-pass replay requires the exact sealed "
            "heterogeneous physical descriptor."
        )

    source_path = Path(materializer.__file__).resolve()
    source_sha256 = _source_file_sha256(
        source_path,
        "Neural Forge heterogeneous materializer",
    )
    if source_sha256 not in _HETEROGENEOUS_SINGLE_PASS_MATERIALIZER_SOURCE_ALLOWLIST:
        raise Lfm2CircuitTraceCaptureError(
            "Heterogeneous single-pass replay has not certified this Neural "
            "Forge materializer source revision."
        )
    try:
        receipt = dict(
            materializer.materialize_lfm2_dynamic_blank_child(
                model,
                model.config,
                layout,
                expected_descriptor_sha256=layout.descriptor_sha256,
            )
        )
        dynamic_mlp_class = materializer._dynamic_mlp_class()
        dynamic_short_conv_class = materializer._dynamic_short_conv_class()
        dynamic_attention_class = materializer._dynamic_attention_class()
    except Exception as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Heterogeneous single-pass replay physical runtime authentication failed."
        ) from exc

    receipt_body = dict(receipt)
    receipt_sha256 = receipt_body.pop("receipt_sha256", None)
    audit = receipt.get("physical_materialization_audit")
    if not isinstance(audit, Mapping):
        raise Lfm2CircuitTraceCaptureError(
            "Heterogeneous single-pass replay omitted its physical audit."
        )
    audit_body = dict(audit)
    audit_sha256 = audit_body.pop("result_sha256", None)
    if (
        not isinstance(receipt_sha256, str)
        or _json_sha256(receipt_body) != receipt_sha256
        or receipt.get("descriptor_sha256") != layout.descriptor_sha256
        or receipt.get("status")
        != "authenticated-dense-fp32-blank-child-materialization"
        or not isinstance(audit_sha256, str)
        or _json_sha256(audit_body) != audit_sha256
        or audit.get("descriptor_sha256") != layout.descriptor_sha256
        or audit.get("status") != "dense-fp32-physical-layout-authenticated"
        or audit.get("all_parameters_frozen") is not True
        or audit.get("parameters_quantized") != 0
        or audit.get("sparse_parameter_storage") is not False
        or audit.get("cross_layer_parameter_sharing") is not False
        or audit.get("runtime_sidecar_parameter_elements") != 0
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Heterogeneous single-pass replay materialization receipt is not "
            "an exact dense FP32 physical authentication."
        )

    class_binding = {
        "feed_forward": (
            f"{dynamic_mlp_class.__module__}.{dynamic_mlp_class.__qualname__}"
        ),
        "short_conv": (
            f"{dynamic_short_conv_class.__module__}."
            f"{dynamic_short_conv_class.__qualname__}"
        ),
        "attention": (
            f"{dynamic_attention_class.__module__}."
            f"{dynamic_attention_class.__qualname__}"
        ),
    }
    body = {
        "schema": ("neural-forge-lfm2-heterogeneous-single-pass-runtime-binding-v1"),
        "descriptor_sha256": layout.descriptor_sha256,
        "descriptor_layer_count": layout.layer_count,
        "materializer_source_sha256": source_sha256,
        "materializer_source_allowlisted": True,
        "materialization_receipt_sha256": receipt_sha256,
        "physical_materialization_audit_sha256": audit_sha256,
        "exact_runtime_classes": class_binding,
        "exact_runtime_class_methods_reauthenticated": True,
        "floating_dtypes": list(floating_parameter_dtypes),
        "floating_parameter_and_buffer_dtypes": list(floating_dtypes),
        "parameters_quantized": 0,
        "sparse_parameter_storage": False,
        "parameter_sharing": False,
        "runtime_sidecar_parameter_elements": 0,
        "execution_sha256": execution["execution_sha256"],
    }
    return {
        "binding": {**body, "binding_sha256": _json_sha256(body)},
        "mlp_class": dynamic_mlp_class,
        "short_conv_class": dynamic_short_conv_class,
        "attention_class": dynamic_attention_class,
    }


def _stable_runtime_code_constant(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {
            "kind": "code",
            "sha256": _stable_runtime_code_sha256(value),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_stable_runtime_code_constant(item) for item in value],
        }
    return {
        "kind": "constant",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _stable_runtime_code_sha256(code: CodeType) -> str:
    """Hash executable semantics without installation-path or line-table drift."""

    return _json_sha256(
        {
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
            "constants": [
                _stable_runtime_code_constant(value) for value in code.co_consts
            ],
            "names": list(code.co_names),
            "variable_names": list(code.co_varnames),
            "free_variables": list(code.co_freevars),
            "cell_variables": list(code.co_cellvars),
            "positional_argument_count": int(code.co_argcount),
            "positional_only_argument_count": int(code.co_posonlyargcount),
            "keyword_only_argument_count": int(code.co_kwonlyargcount),
            "flags": int(code.co_flags),
        }
    )


def _authenticated_torch_linear_forward(torch: Any) -> dict[str, Any]:
    """Bind ``Linear.forward`` independently of its mutable class attribute."""

    forward = getattr(getattr(torch.nn, "Linear", None), "forward", None)
    code = getattr(forward, "__code__", None)
    linear_module = sys.modules.get("torch.nn.modules.linear")
    source_value = getattr(linear_module, "__file__", None)
    runtime_version = getattr(torch, "__version__", None)
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    if (
        not isinstance(code, CodeType)
        or linear_module is None
        or not isinstance(source_value, str)
        or not isinstance(runtime_version, str)
        or not isinstance(cache_tag, str)
        or getattr(forward, "__module__", None) != "torch.nn.modules.linear"
        or getattr(forward, "__qualname__", None) != "Linear.forward"
        or getattr(forward, "__globals__", None) is not vars(linear_module)
        or getattr(forward, "__closure__", None) is not None
        or getattr(forward, "__defaults__", None) is not None
        or getattr(forward, "__kwdefaults__", None) is not None
        or forward.__globals__.get("F") is not torch.nn.functional
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires an authenticated canonical "
            "torch.nn.Linear.forward implementation."
        )
    source_path = Path(source_value).resolve()
    try:
        filename_matches = Path(code.co_filename).resolve() == source_path
        source_sha256 = _source_file_sha256(
            source_path,
            "torch.nn.Linear.forward",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay cannot authenticate "
            "torch.nn.Linear.forward source identity."
        ) from exc
    code_sha256 = _stable_runtime_code_sha256(code)
    runtime_identity = (
        runtime_version,
        cache_tag,
        code_sha256,
        source_sha256,
    )
    if not filename_matches or runtime_identity not in (
        _TORCH_LINEAR_FORWARD_RUNTIME_ALLOWLIST
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay rejects this unauthenticated "
            "torch.nn.Linear.forward runtime."
        )
    return {
        "torch_version": runtime_version,
        "python_cache_tag": cache_tag,
        "module": "torch.nn.modules.linear",
        "qualname": "Linear.forward",
        "defining_file_authenticated": True,
        "defining_globals_authenticated": True,
        "functional_namespace_authenticated": True,
        "code_sha256": code_sha256,
        "source_sha256": source_sha256,
    }


def _single_pass_lm_head_runtime_binding(
    torch: Any,
    model: Any,
) -> tuple[Any, dict[str, Any]]:
    """Authenticate the exact linear operation assumed by the logit bound."""

    lm_head = getattr(model, "lm_head", None)
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    vocabulary_size = getattr(config, "vocab_size", None)
    linear_forward_binding = _authenticated_torch_linear_forward(torch)
    native_linear = getattr(
        getattr(getattr(torch, "_C", None), "_nn", None),
        "linear",
        None,
    )
    if (
        type(lm_head) is not torch.nn.Linear
        or type(lm_head).forward is not torch.nn.Linear.forward
        or type(lm_head).__call__ is not torch.nn.Module.__call__
        or not callable(native_linear)
        or torch.nn.functional.linear is not native_linear
        or "forward" in vars(lm_head)
        or "__call__" in vars(lm_head)
        or isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
        or isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or vocabulary_size <= 0
        or type(lm_head.in_features) is not int
        or lm_head.in_features != hidden_size
        or type(lm_head.out_features) is not int
        or lm_head.out_features != vocabulary_size
        or lm_head.bias is not None
        or not torch.is_tensor(lm_head.weight)
        or lm_head.weight.layout != torch.strided
        or lm_head.weight.device.type == "meta"
        or type(lm_head.weight) is not torch.nn.Parameter
        or tuple(lm_head.weight.shape) != (vocabulary_size, hidden_size)
        or lm_head.weight.is_complex()
        or not lm_head.weight.is_floating_point()
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires an exact bias-free "
            "torch.nn.Linear LM head with canonical class methods and no "
            "instance forward/__call__ overrides."
        )
    body = {
        "schema": "neural-forge-lfm2-single-pass-lm-head-runtime-v1",
        "module_class": _qualified_class(lm_head),
        "forward_method": (
            f"{torch.nn.Linear.forward.__module__}."
            f"{torch.nn.Linear.forward.__qualname__}"
        ),
        "forward_runtime_binding": linear_forward_binding,
        "call_method": (
            f"{torch.nn.Module.__call__.__module__}."
            f"{torch.nn.Module.__call__.__qualname__}"
        ),
        "functional_linear_primitive": (
            f"{native_linear.__module__}.{native_linear.__qualname__}"
        ),
        "functional_linear_native": True,
        "instance_forward_override": False,
        "instance_call_override": False,
        "bias": False,
        "input_features": hidden_size,
        "output_features": vocabulary_size,
        "weight_dtype": str(lm_head.weight.dtype).replace("torch.", ""),
    }
    return lm_head, {**body, "binding_sha256": _json_sha256(body)}


def _optional_fp32_backend_precision(value: Any, name: str) -> str | None:
    try:
        precision = getattr(value, "fp32_precision")
    except (AttributeError, RuntimeError):
        return None
    if precision is None:
        return None
    if not isinstance(precision, str) or precision not in {
        "none",
        "ieee",
        "tf32",
    }:
        raise Lfm2CircuitTraceCaptureError(
            f"Single-pass child replay found an unsupported {name} FP32 precision mode."
        )
    return precision


def _ieee_fp32_matmul_runtime_binding(
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Fail closed unless FP32 linear algebra uses IEEE-width products."""

    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    if not callable(get_precision):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay cannot authenticate FP32 matmul precision."
        )
    precision = get_precision()
    if precision != "highest":
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires float32 matmul precision "
            "'highest'; reduced FP32/TF32 math is not certified."
        )
    device_type = str(getattr(device, "type", device))
    if device_type not in {"cpu", "cuda"}:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay has not certified IEEE FP32 matmul on "
            f"device type {device_type!r}."
        )
    backends = getattr(torch, "backends", None)
    global_backend_precision = _optional_fp32_backend_precision(
        backends,
        "global backend",
    )
    if global_backend_precision == "tf32":
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay rejects globally enabled TF32 math."
        )

    capability: list[int] | None = None
    tf32_hardware_supported = False
    legacy_allow_tf32: bool | None = None
    cuda_backend_precision: str | None = None
    if device_type == "cuda":
        cuda = getattr(torch, "cuda", None)
        get_capability = getattr(cuda, "get_device_capability", None)
        if not callable(get_capability):
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass child replay cannot authenticate CUDA capability."
            )
        raw_capability = get_capability(device)
        if (
            not isinstance(raw_capability, (tuple, list))
            or len(raw_capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in raw_capability
            )
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass child replay received an invalid CUDA capability."
            )
        capability = [int(value) for value in raw_capability]
        tf32_hardware_supported = capability[0] >= 8
        cuda_backend = getattr(backends, "cuda", None)
        matmul_backend = getattr(cuda_backend, "matmul", None)
        cuda_backend_precision = _optional_fp32_backend_precision(
            matmul_backend,
            "CUDA matmul backend",
        )
        try:
            raw_allow_tf32 = getattr(matmul_backend, "allow_tf32")
        except (AttributeError, RuntimeError):
            raw_allow_tf32 = None
        if raw_allow_tf32 is not None:
            if not isinstance(raw_allow_tf32, bool):
                raise Lfm2CircuitTraceCaptureError(
                    "Single-pass child replay received an invalid CUDA TF32 flag."
                )
            legacy_allow_tf32 = raw_allow_tf32
        if tf32_hardware_supported and (
            legacy_allow_tf32 is not False or cuda_backend_precision == "tf32"
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass child replay rejects TF32-capable CUDA matmul "
                "unless its runtime explicitly disables TF32."
            )

    body = {
        "schema": "neural-forge-lfm2-ieee-fp32-matmul-runtime-v1",
        "device_type": device_type,
        "torch_float32_matmul_precision": precision,
        "global_backend_fp32_precision": global_backend_precision,
        "cuda_matmul_fp32_precision": cuda_backend_precision,
        "cuda_compute_capability": capability,
        "tf32_hardware_supported": tf32_hardware_supported,
        "legacy_cuda_allow_tf32": legacy_allow_tf32,
        "ieee_fp32_products_proven": True,
    }
    return {**body, "binding_sha256": _json_sha256(body)}


def _indexed_numeric_execution_device(torch: Any, device: Any) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu" and resolved.index is None:
        return "cpu"
    if resolved.type != "cuda":
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric certification supports only CPU or CUDA."
        )
    index = resolved.index
    if index is None:
        current_device = getattr(getattr(torch, "cuda", None), "current_device", None)
        if not callable(current_device):
            raise Lfm2CircuitTraceCaptureError(
                "Child replay cannot resolve the active CUDA device index."
            )
        index = int(current_device())
    return f"cuda:{index}"


def _model_layer_fan_in_widths(
    numeric_policy: Mapping[str, Any],
    layer_count: int,
    hidden_size: int,
) -> tuple[int, tuple[int, ...]]:
    target_inventory = numeric_policy.get("target_inventory")
    entries = (
        target_inventory.get("entries")
        if isinstance(target_inventory, Mapping)
        else None
    )
    if not isinstance(entries, list):
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy omitted its target tensor inventory."
        )
    parameter_widths: list[tuple[str, int]] = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or entry.get("kind") != "parameter"
            or entry.get("floating") is not True
        ):
            continue
        shape = entry.get("shape")
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or not isinstance(shape, list)
            or len(shape) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            continue
        parameter_widths.append((name, int(shape[-1])))
    if not parameter_widths:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy has no matrix fan-in inventory."
        )
    overall = max(hidden_size, *(width for _name, width in parameter_widths))
    per_layer: list[int] = []
    for layer_index in range(layer_count):
        marker = f".layers.{layer_index}."
        widths = [width for name, width in parameter_widths if marker in f".{name}"]
        per_layer.append(max(hidden_size, *widths))
    return overall, tuple(per_layer)


def compile_lfm2_child_replay_numeric_policy(
    child_model: Any,
    *,
    input_device: Any | None = None,
) -> dict[str, Any]:
    """Seal the actual child inventory and its two replay execution roles.

    The canonical prefix oracle and the full-trajectory runtime execute the
    same immutable child, but they are deliberately represented as distinct
    forward domains.  Comparison is performed on detached CPU FP64 values;
    this does not cast or move the live model.
    """

    torch = _torch()
    device = _resolve_input_device(torch, child_model, input_device)
    numeric_device = _indexed_numeric_execution_device(torch, device)
    try:
        policy = resolve_numeric_execution_policy(
            source_models={"canonical_prefix_reference_child": child_model},
            target_model=child_model,
            request=NumericExecutionRequest(
                source_forward_device=numeric_device,
                target_forward_device=numeric_device,
                accumulation_device="cpu",
                accumulation_dtype="float64",
                verification_device="cpu",
                verification_dtype="float64",
            ),
        )
    except NumericExecutionPolicyError as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy could not authenticate the live "
            "parameter-and-buffer inventory."
        ) from exc
    numeric_receipt = policy.as_dict()
    target_inventory = numeric_receipt["target_inventory"]
    floating_dtype_counts = target_inventory["floating_dtype_counts"]
    floating_dtypes = sorted(floating_dtype_counts)
    if not floating_dtypes:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy found no floating runtime state."
        )
    unit_roundoffs = {
        dtype: _ieee_unit_roundoff(
            torch,
            getattr(torch, dtype),
        )
        for dtype in floating_dtypes
    }
    maximum_unit_roundoff = max(unit_roundoffs.values())
    strict_rounding_terms = 8
    strict_product = strict_rounding_terms * maximum_unit_roundoff
    if strict_product >= 1.0:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay strict comparison has no finite IEEE gamma bound."
        )
    strict_rtol = strict_product / (1.0 - strict_product)

    inner_model = getattr(child_model, "model", None)
    raw_layers = getattr(inner_model, "layers", None)
    try:
        layers = tuple(raw_layers)
    except TypeError as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy requires an iterable decoder stack."
        ) from exc
    if not layers:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy requires a nonempty decoder stack."
        )
    lm_head = getattr(child_model, "lm_head", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    if not torch.is_tensor(lm_head_weight) or int(lm_head_weight.ndim) != 2:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy requires a rank-2 LM-head weight."
        )
    hidden_size = int(lm_head_weight.shape[1])
    maximum_fan_in, layer_fan_ins = _model_layer_fan_in_widths(
        numeric_receipt,
        len(layers),
        hidden_size,
    )
    source_inventory = numeric_receipt["source_inventories"][
        "canonical_prefix_reference_child"
    ]
    reference_role = {
        "role": "canonical_prefix_reference",
        "numeric_domain": "source_forward",
        "execution_device": numeric_device,
        "inventory_sha256": source_inventory["inventory_sha256"],
    }
    runtime_role = {
        "role": "single_pass_causal_runtime",
        "numeric_domain": "target_forward",
        "execution_device": numeric_device,
        "inventory_sha256": target_inventory["inventory_sha256"],
    }
    envelope_body = {
        "schema": CHILD_REPLAY_NON_LOGIT_ENVELOPE_SCHEMA,
        "error_model": CHILD_REPLAY_NON_LOGIT_ERROR_MODEL,
        "numeric_policy_sha256": policy.policy_sha256,
        "reference_execution_role": reference_role,
        "runtime_execution_role": runtime_role,
        "comparison_device": "cpu",
        "comparison_dtype": "float64",
        "floating_dtype_counts": dict(floating_dtype_counts),
        "unit_roundoffs": unit_roundoffs,
        "model_layer_count": len(layers),
        "hidden_size": hidden_size,
        "maximum_parameter_fan_in": maximum_fan_in,
        "layer_parameter_fan_ins": list(layer_fan_ins),
        "recorded_residual_rounding_stages_per_layer": 2,
        "recorded_final_normalization_rounding_stages": 1,
        "path_depth_rule": (
            "two accumulated residual branches per completed decoder layer "
            "plus the producing normalization/branch"
        ),
        "reduction_width_rule": (
            "maximum actual matrix fan-in on the recorded producer path"
        ),
        "forward_error_formula": (
            "2*gamma_(path_depth*reduction_width)*"
            "max(abs(reference),abs(runtime),row_rms(reference),"
            "row_rms(runtime))+underflow"
        ),
        "strict_comparison_rounding_terms": strict_rounding_terms,
        "strict_rtol": strict_rtol,
        "strict_atol": 0.0,
        "top1_required_for_logits": True,
        "tensor_shape_dtype_device_hash_and_finiteness_required": True,
    }
    envelope = {
        **envelope_body,
        "envelope_sha256": _json_sha256(envelope_body),
    }
    body = {
        "schema": CHILD_REPLAY_NUMERIC_POLICY_SCHEMA,
        "numeric_execution_policy": numeric_receipt,
        "numeric_policy_sha256": policy.policy_sha256,
        "reference_execution_role": reference_role,
        "runtime_execution_role": runtime_role,
        "strict_rtol": strict_rtol,
        "strict_atol": 0.0,
        "logit_error_model": CHILD_REPLAY_LOGIT_ERROR_MODEL,
        "non_logit_error_model": CHILD_REPLAY_NON_LOGIT_ERROR_MODEL,
        "non_logit_envelope_policy": envelope,
        "non_logit_envelope_sha256": envelope["envelope_sha256"],
        "byte_exact_required": False,
        "top1_required": True,
    }
    return {
        **body,
        "replay_numeric_policy_sha256": _json_sha256(body),
    }


def _authenticate_child_replay_numeric_policy(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    policy = dict(value)
    claimed = policy.get("replay_numeric_policy_sha256")
    if (
        policy.get("schema") != CHILD_REPLAY_NUMERIC_POLICY_SCHEMA
        or not isinstance(claimed, str)
        or len(claimed) != _SHA256_LENGTH
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric certification policy is malformed."
        )
    body = dict(policy)
    body.pop("replay_numeric_policy_sha256", None)
    if _json_sha256(body) != claimed:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric certification policy seal is invalid."
        )
    try:
        authenticated_numeric = NumericExecutionPolicy(
            policy["numeric_execution_policy"]
        )
    except (KeyError, NumericExecutionPolicyError) as exc:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric execution policy is invalid."
        ) from exc
    if authenticated_numeric.policy_sha256 != policy.get("numeric_policy_sha256"):
        raise Lfm2CircuitTraceCaptureError(
            "Child replay numeric policy digest binding is invalid."
        )
    envelope = policy.get("non_logit_envelope_policy")
    if not isinstance(envelope, Mapping):
        raise Lfm2CircuitTraceCaptureError(
            "Child replay non-logit envelope policy is missing."
        )
    envelope_body = dict(envelope)
    envelope_claimed = envelope_body.pop("envelope_sha256", None)
    if (
        envelope.get("schema") != CHILD_REPLAY_NON_LOGIT_ENVELOPE_SCHEMA
        or envelope.get("numeric_policy_sha256") != authenticated_numeric.policy_sha256
        or envelope_claimed != policy.get("non_logit_envelope_sha256")
        or not isinstance(envelope_claimed, str)
        or _json_sha256(envelope_body) != envelope_claimed
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Child replay non-logit envelope policy seal is invalid."
        )
    return policy


def _single_pass_causal_contract(
    model: Any,
    layers: Sequence[Any],
    contracts: Sequence[Lfm2RuntimeLayerContract],
    execution: Mapping[str, Any],
    device: Any,
    prepared: Sequence[_PreparedBatch],
    numeric_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the loaded runtime has known causal LFM2 semantics.

    One full-trajectory forward can replace repeated prefix forwards only when
    every retained position is independent of future tokens.  The fast path is
    intentionally limited to the canonical Transformers LFM2 implementation
    and eager attention.  Its exact source and execution identity are already
    content-addressed in replay metadata.
    """

    torch = _torch()
    autocast_enabled = bool(torch.is_autocast_enabled())
    try:
        autocast_enabled = autocast_enabled or bool(torch.is_autocast_enabled("cpu"))
    except TypeError:  # pragma: no cover - older supported PyTorch surface
        pass
    if autocast_enabled:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires autocast to be disabled."
        )
    hooked_modules = []
    for name, module in model.named_modules():
        if bool(getattr(module, "_forward_pre_hooks", {})) or bool(
            getattr(module, "_forward_hooks", {})
        ):
            hooked_modules.append(name or "<root>")
    if hooked_modules:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires an unhooked runtime; existing "
            f"forward hooks were found on {sorted(hooked_modules)}."
        )
    authenticated_numeric_policy = _authenticate_child_replay_numeric_policy(
        numeric_policy
    )
    target_numeric_inventory = authenticated_numeric_policy["numeric_execution_policy"][
        "target_inventory"
    ]
    floating_dtypes = sorted(target_numeric_inventory["floating_dtype_counts"])
    floating_parameter_dtypes = sorted(
        target_numeric_inventory["floating_parameter_dtype_counts"]
    )
    if not floating_dtypes or not set(floating_dtypes).issubset({"float32", "float64"}):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay has not certified low-precision model "
            f"dtypes {floating_dtypes}; use prefix_by_prefix_canonical."
        )
    try:
        from transformers import masking_utils
        from transformers.models.lfm2 import modeling_lfm2
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Single-pass child replay requires Transformers LFM2 support."
        ) from exc
    _lm_head, lm_head_runtime_binding = _single_pass_lm_head_runtime_binding(
        torch,
        model,
    )
    ieee_fp32_matmul_runtime_binding = (
        _ieee_fp32_matmul_runtime_binding(torch, device)
        if "float32" in floating_dtypes
        else None
    )
    inner_model = getattr(model, "model", None)
    heterogeneous_runtime = _heterogeneous_single_pass_runtime_binding(
        model,
        execution,
        floating_dtypes,
        floating_parameter_dtypes,
    )
    if (
        type(model) is not modeling_lfm2.Lfm2ForCausalLM
        or type(inner_model) is not modeling_lfm2.Lfm2Model
        or "forward" in vars(model)
        or "forward" in vars(inner_model)
        or type(model).forward is not modeling_lfm2.Lfm2ForCausalLM.forward
        or type(inner_model).forward is not modeling_lfm2.Lfm2Model.forward
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires the canonical Transformers "
            "LFM2ForCausalLM/Lfm2Model forward runtime without instance or "
            "class monkeypatches."
        )
    for layer, contract in zip(layers, contracts):
        if heterogeneous_runtime is None:
            expected_operator_type = (
                modeling_lfm2.Lfm2Attention
                if contract.kind == "full_attention"
                else modeling_lfm2.Lfm2ShortConv
            )
            expected_feed_forward_type = modeling_lfm2.Lfm2MLP
        else:
            expected_operator_type = (
                heterogeneous_runtime["attention_class"]
                if contract.kind == "full_attention"
                else heterogeneous_runtime["short_conv_class"]
            )
            expected_feed_forward_type = heterogeneous_runtime["mlp_class"]
        operator = layer.self_attn if contract.kind == "full_attention" else layer.conv
        if (
            type(layer) is not modeling_lfm2.Lfm2DecoderLayer
            or type(operator) is not expected_operator_type
            or type(layer.feed_forward) is not expected_feed_forward_type
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass child replay requires exact canonical LFM2 "
                "decoder/operator/feed-forward runtime classes."
            )
        if (
            contract.kind == "full_attention"
            and getattr(operator, "is_causal", None) is not True
        ):
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass child replay requires every attention operator "
                "to certify is_causal=True."
            )
    if execution.get("attention_backend") not in {"eager", "sdpa"}:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay is certified only for eager or SDPA "
            "LFM2 attention; use prefix_by_prefix_canonical for this runtime."
        )
    if execution.get("fast_short_conv_selected") is not False:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay has not certified the external fast "
            "short-convolution kernel; use prefix_by_prefix_canonical."
        )
    config = getattr(model, "config", None)
    if getattr(config, "is_causal", True) is not True:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires config.is_causal=True."
        )
    rope_parameters = getattr(config, "rope_parameters", None)
    if (
        not isinstance(rope_parameters, Mapping)
        or rope_parameters.get("rope_type") != "default"
        or getattr(getattr(inner_model, "rotary_emb", None), "rope_type", None)
        != "default"
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay is certified only for sequence-length "
            "independent default RoPE."
        )
    max_positions = getattr(config, "max_position_embeddings", None)
    if (
        isinstance(max_positions, bool)
        or not isinstance(max_positions, int)
        or max_positions <= 0
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay requires a positive model context limit."
        )
    for batch in prepared:
        trajectory_length = int(
            batch.input_ids.shape[1] + batch.forced_decode_ids.shape[1]
        )
        if trajectory_length > max_positions:
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass child replay trajectory exceeds the model context limit."
            )
        if batch.position_ids is not None:
            last_position = int(batch.position_ids[:, -1:].max().item())
            last_position += int(batch.forced_decode_ids.shape[1])
            if last_position >= max_positions:
                raise Lfm2CircuitTraceCaptureError(
                    "Single-pass child replay positions exceed the model context limit."
                )
    runtime_path = Path(modeling_lfm2.__file__).resolve()
    masking_path = Path(masking_utils.__file__).resolve()
    transformers_version = _package_version("transformers")
    runtime_source_sha256 = _source_file_sha256(
        runtime_path,
        "Transformers LFM2",
    )
    masking_source_sha256 = _source_file_sha256(
        masking_path,
        "Transformers causal masking",
    )
    runtime_identity = (
        transformers_version,
        runtime_source_sha256,
        masking_source_sha256,
    )
    if runtime_identity not in _SINGLE_PASS_CAUSAL_RUNTIME_ALLOWLIST:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay has not certified this Transformers "
            "LFM2/masking runtime revision; use prefix_by_prefix_canonical."
        )
    for batch in prepared:
        positions = batch.position_ids
        if positions is None or int(positions.shape[1]) <= 1:
            continue
        increments = positions[:, 1:] - positions[:, :-1]
        if not bool((increments == 1).all().item()):
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass child replay requires contiguous position_ids; "
                "use prefix_by_prefix_canonical for custom position layouts."
            )
    body = {
        "schema": "neural-forge-lfm2-single-pass-causal-contract-v4",
        "runtime_family": "transformers.models.lfm2.modeling_lfm2",
        "runtime_variant": (
            "canonical_transformers_lfm2"
            if heterogeneous_runtime is None
            else "sealed_heterogeneous_dynamic_lfm2"
        ),
        "transformers_version": transformers_version,
        "lfm2_runtime_source_sha256": runtime_source_sha256,
        "causal_mask_source_sha256": masking_source_sha256,
        "canonical_topology_validated": True,
        "exact_runtime_classes_validated": True,
        "unmodified_model_forwards_validated": True,
        "floating_dtypes": floating_dtypes,
        "floating_parameter_dtypes": floating_parameter_dtypes,
        "floating_parameter_and_buffer_inventory_authenticated": True,
        "numeric_policy_sha256": authenticated_numeric_policy["numeric_policy_sha256"],
        "replay_numeric_policy_sha256": authenticated_numeric_policy[
            "replay_numeric_policy_sha256"
        ],
        "non_logit_envelope_sha256": authenticated_numeric_policy[
            "non_logit_envelope_sha256"
        ],
        "reference_execution_role": authenticated_numeric_policy[
            "reference_execution_role"
        ],
        "runtime_execution_role": authenticated_numeric_policy[
            "runtime_execution_role"
        ],
        "low_precision_execution": False,
        "lm_head_runtime_binding": lm_head_runtime_binding,
        "ieee_fp32_matmul_runtime_binding": ieee_fp32_matmul_runtime_binding,
        "attention_backend": execution["attention_backend"],
        "attention_is_causal": True,
        "short_conv_is_causal": True,
        "external_fast_short_conv_used": False,
        "rope_type": "default",
        "max_position_embeddings": max_positions,
        "all_trajectories_within_context": True,
        "unpadding_required": True,
        "contiguous_positions_required": True,
        "trajectory_use_cache": False,
        "execution_sha256": execution["execution_sha256"],
        "heterogeneous_runtime_binding": (
            None if heterogeneous_runtime is None else heterogeneous_runtime["binding"]
        ),
    }
    return {**body, "contract_sha256": _json_sha256(body)}


def _trajectory_inputs(
    torch: Any,
    batch: _PreparedBatch,
    device: Any,
) -> tuple[Any, Any, Any | None, Any]:
    input_ids = batch.input_ids.to(device=device)
    forced_ids = batch.forced_decode_ids.to(device=device)
    attention_mask = batch.attention_mask.to(device=device)
    forced_mask = torch.ones(
        (int(attention_mask.shape[0]), int(forced_ids.shape[1])),
        dtype=attention_mask.dtype,
        device=device,
    )
    trajectory_mask = torch.cat((attention_mask, forced_mask), dim=1)
    positions = _move_to_device(batch.position_ids, device)
    trajectory_positions = None
    if positions is not None:
        offsets = torch.arange(
            1,
            int(forced_ids.shape[1]) + 1,
            dtype=positions.dtype,
            device=device,
        ).unsqueeze(0)
        trajectory_positions = torch.cat(
            (positions, positions[:, -1:] + offsets),
            dim=1,
        )
    return (
        torch.cat((input_ids, forced_ids), dim=1),
        trajectory_mask,
        trajectory_positions,
        forced_ids,
    )


def _trajectory_tensor_sequence_axis(name: str, value: Any) -> int:
    if ".attention." in name and int(value.ndim) == 4:
        return 2
    if int(value.ndim) >= 2:
        return 1
    raise Lfm2CircuitTraceCaptureError(
        f"Full-trajectory endpoint {name!r} has no sequence axis."
    )


def _slice_child_trajectory_phase(
    torch: Any,
    trajectory_trace: Mapping[str, Any],
    trajectory_logits: Any,
    trajectory_diagnostics: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = str(row["phase"])
    sample_start = int(row["sample_start"])
    sample_count = int(row["sample_count"])
    result: dict[str, Any] = {}
    expected_prefix = "trajectory."
    for name, value in trajectory_trace.items():
        if not name.startswith(expected_prefix):
            raise Lfm2CircuitTraceCaptureError(
                f"Unexpected full-trajectory endpoint {name!r}."
            )
        axis = _trajectory_tensor_sequence_axis(name, value)
        sequence_length = int(value.shape[axis])
        if (
            sample_start < 0
            or sample_count <= 0
            or sample_start + sample_count > sequence_length
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Cannot slice {phase!r} from full-trajectory endpoint {name!r}."
            )
        sliced = value.narrow(axis, sample_start, sample_count)
        result[phase + name[len("trajectory") :]] = _cpu_tensor(
            torch,
            sliced,
            f"{phase} full-trajectory slice {name}",
        )
    if sample_start + sample_count > int(trajectory_logits.shape[1]):
        raise Lfm2CircuitTraceCaptureError(
            f"Cannot slice {phase!r} logits from the full trajectory."
        )
    result[f"{phase}.logits"] = _cpu_tensor(
        torch,
        trajectory_logits[:, sample_start : sample_start + sample_count, :],
        f"{phase} full-trajectory logits",
    )
    diagnostics = json.loads(_canonical_json(dict(trajectory_diagnostics)))
    return result, diagnostics


def _single_pass_peak_capture_bytes(
    model: Any,
    layers: Sequence[Any],
    contracts: Sequence[Lfm2RuntimeLayerContract],
    batch: _PreparedBatch,
    config: Lfm2CircuitTraceCaptureConfig,
    selected_layer_indices: Sequence[int],
) -> int:
    torch = _torch()
    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0))
    vocabulary_size = int(getattr(getattr(model, "config", None), "vocab_size", 0))
    if hidden_size <= 0 or vocabulary_size <= 0:
        raise Lfm2CircuitTraceCaptureError(
            "Cannot estimate single-pass trace memory from the child config."
        )
    # Logits plus the final normalized hidden state retained to derive the
    # cancellation-safe LM-head replay error envelope.
    per_token_elements = vocabulary_size + hidden_size
    for index in selected_layer_indices:
        layer = layers[index]
        contract = contracts[index]
        intermediate_size = int(layer.feed_forward.w1.out_features)
        # Seven hidden-width common endpoints and three SwiGLU-width endpoints.
        per_token_elements += 7 * hidden_size + 3 * intermediate_size
        if contract.kind == "full_attention":
            operator = contract.operator
            if not isinstance(operator, Lfm2AttentionContract):
                raise Lfm2CircuitTraceCaptureError(
                    "Internal LFM2 attention contract mismatch."
                )
            query_width = operator.query_heads * operator.head_dim
            key_value_width = operator.key_value_heads * operator.head_dim
            # Q-linear, Q-rotary, context; K-linear, K-rotary, V; O.
            per_token_elements += 3 * query_width + 3 * key_value_width + hidden_size
        else:
            # B, C, x, B*x, pre-O context, O.
            operator = contract.operator
            if not isinstance(operator, Lfm2RuntimeShortConvContract):
                raise Lfm2CircuitTraceCaptureError(
                    "Internal LFM2 short-conv contract mismatch."
                )
            internal_width = (
                operator.internal_channel_width
                if isinstance(operator, Lfm2HeterogeneousShortConvContract)
                else operator.hidden_size
            )
            per_token_elements += 5 * internal_width + hidden_size

    embeddings = model.get_input_embeddings()
    element_size = int(embeddings.weight.element_size())
    trajectory_tokens = int(batch.input_ids.shape[1] + batch.forced_decode_ids.shape[1])
    retained_tokens = min(
        int(batch.input_ids.shape[1]),
        config.sample_last_tokens,
    ) + int(batch.forced_decode_ids.shape[1])
    peak = per_token_elements * (trajectory_tokens + retained_tokens) * element_size
    input_tensors = [
        batch.input_ids,
        batch.attention_mask,
        batch.forced_decode_ids,
    ]
    if batch.position_ids is not None:
        input_tensors.append(batch.position_ids)
    peak += sum(int(value.numel() * value.element_size()) for value in input_tensors)
    if peak <= 0 or peak > torch.iinfo(torch.int64).max:
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass trace memory estimate is invalid."
        )
    return int(peak)


def _capture_child_batch_single_pass(
    torch: Any,
    model: Any,
    layers: Sequence[Any],
    contracts: Sequence[Lfm2RuntimeLayerContract],
    batch: _PreparedBatch,
    config: Lfm2CircuitTraceCaptureConfig,
    device: Any,
    selected_layer_indices: Sequence[int],
    causal_contract: Mapping[str, Any],
    telemetry: _CaptureTelemetry | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    estimated_peak_capture_bytes = _single_pass_peak_capture_bytes(
        model,
        layers,
        contracts,
        batch,
        config,
        selected_layer_indices,
    )
    if estimated_peak_capture_bytes > config.max_capture_bytes:
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {batch.index} single-pass trajectory would exceed "
            "max_capture_bytes before forwarding."
        )
    trajectory_ids, trajectory_mask, trajectory_positions, forced_ids = (
        _trajectory_inputs(torch, batch, device)
    )
    vocabulary_size = int(getattr(getattr(model, "config", None), "vocab_size", 0))
    if vocabulary_size <= 0 or bool((trajectory_ids >= vocabulary_size).any().item()):
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {batch.index} trajectory contains out-of-vocabulary tokens."
        )

    selected = _selected_layer_indices(
        len(layers),
        selected_layer_indices,
    )
    module_modes = [(module, bool(module.training)) for module in model.modules()]
    hooks = _Lfm2TraceHooks(
        torch,
        model,
        layers,
        contracts,
        config,
        selected,
        capture_replay_lm_head_input=True,
    )
    tensors: dict[str, Any] = {
        "input.input_ids": batch.input_ids,
        "input.attention_mask": batch.attention_mask,
        f"input.{_FORCED_IDS}": batch.forced_decode_ids,
    }
    if batch.position_ids is not None:
        tensors["input.position_ids"] = batch.position_ids
    phase_rows = _child_replay_phase_rows(batch, config)
    phase_diagnostics: dict[str, Any] = {}
    try:
        model.eval()
        with torch.inference_mode():
            hooks.begin(
                "trajectory",
                sample_last_tokens=int(trajectory_ids.shape[1]),
            )
            output = _forward(
                model,
                input_ids=trajectory_ids,
                attention_mask=trajectory_mask,
                position_ids=trajectory_positions,
                use_cache=False,
                past_key_values=None,
            )
            trajectory_trace, trajectory_diagnostics = hooks.finish()
            if _output_field(output, "past_key_values") is not None:
                raise Lfm2CircuitTraceCaptureError(
                    "The uncached full-trajectory child forward returned a cache."
                )
            trajectory_logits = _output_logits(torch, output)
            if int(trajectory_logits.shape[1]) != int(trajectory_ids.shape[1]):
                raise Lfm2CircuitTraceCaptureError(
                    "The full-trajectory child logits changed sequence length."
                )
            for row in phase_rows:
                phase_trace, diagnostics = _slice_child_trajectory_phase(
                    torch,
                    trajectory_trace,
                    trajectory_logits,
                    trajectory_diagnostics,
                    row,
                )
                tensors.update(phase_trace)
                phase = str(row["phase"])
                phase_diagnostics[phase] = diagnostics
                step_index = row["forced_decode_step"]
                if step_index is not None:
                    token = forced_ids[:, int(step_index) : int(step_index) + 1]
                    tensors[f"{phase}.forced_token_ids"] = _cpu_tensor(
                        torch,
                        token,
                        f"{phase} forced token IDs",
                    )
    finally:
        if telemetry is not None:
            telemetry.absorb_stage(hooks.stage_counters)
        hooks.remove()
        for module, was_training in module_modes:
            module.training = was_training

    capture_bytes = sum(
        int(value.numel() * value.element_size()) for value in tensors.values()
    )
    if capture_bytes > config.max_capture_bytes:
        raise Lfm2CircuitTraceCaptureError(
            f"Batch {batch.index} trace exceeded max_capture_bytes."
        )
    return tensors, {
        "capture_bytes": capture_bytes,
        "estimated_peak_capture_bytes": estimated_peak_capture_bytes,
        "decode_steps": int(batch.forced_decode_ids.shape[1]),
        "prefill_sample_start": max(
            0, int(batch.input_ids.shape[1]) - config.sample_last_tokens
        ),
        "prefill_sample_count": min(
            int(batch.input_ids.shape[1]), config.sample_last_tokens
        ),
        "phase_diagnostics": phase_diagnostics,
        "cached_full_prefix_verifications": [],
        "all_cached_logits_verified": None,
        "all_cached_top1_verified": None,
        "decode_trace_mode": FULL_PREFIX_CANONICAL_TRACE_MODE,
        "canonical_trace_source": "use_cache_false_full_trajectory_slices",
        "cached_logits_accepted": False,
        "cached_forward_count": 0,
        "child_replay_capture_mode": CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        "child_forward_count": 1,
        "canonical_prefix_forward_count": len(phase_rows),
        "causal_equivalence_contract": dict(causal_contract),
        "lm_head_input_captured_for_replay_equivalence": True,
        "gradients_used": False,
        "training_steps": 0,
    }


def _ieee_unit_roundoff(torch: Any, dtype: Any) -> float:
    unit_roundoff = {
        torch.float64: 2.0**-53,
        torch.float32: 2.0**-24,
        torch.float16: 2.0**-11,
        torch.bfloat16: 2.0**-8,
    }.get(dtype)
    if unit_roundoff is None:
        raise Lfm2CircuitTraceCaptureError(
            f"No IEEE replay error model is defined for dtype {dtype}."
        )
    return unit_roundoff


def _lm_head_logit_forward_error_check(
    torch: Any,
    *,
    name: str,
    reference_logits: Any,
    candidate_logits: Any,
    reference_head_input: Any,
    candidate_head_input: Any,
    lm_head_weight: Any,
    lm_head_weight64: Any | None = None,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Check only strict-allclose outliers against an IEEE dot-product bound.

    A full-prefix GEMM and a longer full-trajectory GEMM can use different
    legal reduction trees.  Relative-to-output tolerances are not sufficient
    near a cancellation zero.  For each strict-allclose outlier this computes

        |W| |h_ref - h_candidate|
        + gamma_k (|W| |h_ref| + |W| |h_candidate|),

    where ``gamma_k = k*u/(1-k*u)`` and ``u`` is the unit roundoff of the
    executed dtype.  The calculation is performed in float64 on the exact
    captured LM-head operands.  It therefore admits reduction-order drift but
    does not grant a tensor-wide absolute tolerance.
    """

    if reference_logits.is_complex() or not reference_logits.is_floating_point():
        raise Lfm2CircuitTraceCaptureError(
            f"Replay logits {name!r} must use a real floating dtype."
        )
    if (
        reference_head_input.is_complex()
        or not reference_head_input.is_floating_point()
        or lm_head_weight.is_complex()
        or not lm_head_weight.is_floating_point()
    ):
        raise Lfm2CircuitTraceCaptureError(
            "LM-head replay error evidence must use real floating tensors."
        )
    if (
        reference_logits.dtype != candidate_logits.dtype
        or reference_logits.dtype != reference_head_input.dtype
        or reference_logits.dtype != candidate_head_input.dtype
        or reference_logits.dtype != lm_head_weight.dtype
    ):
        raise Lfm2CircuitTraceCaptureError(
            "LM-head replay error evidence changed floating dtype."
        )
    if int(lm_head_weight.ndim) != 2:
        raise Lfm2CircuitTraceCaptureError("The child LM-head weight must be rank 2.")
    vocabulary_size, hidden_size = (int(value) for value in lm_head_weight.shape)
    if (
        int(reference_logits.ndim) < 2
        or int(reference_head_input.ndim) != int(reference_logits.ndim)
        or tuple(reference_head_input.shape[:-1]) != tuple(reference_logits.shape[:-1])
        or tuple(candidate_head_input.shape) != tuple(reference_head_input.shape)
        or int(reference_logits.shape[-1]) != vocabulary_size
        or int(reference_head_input.shape[-1]) != hidden_size
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Replay logits {name!r} do not align with their LM-head operands."
        )

    reference64 = reference_logits.to(torch.float64)
    candidate64 = candidate_logits.to(torch.float64)
    absolute_error = (reference64 - candidate64).abs()
    strict_limit = atol + rtol * candidate64.abs()
    strict_outliers = absolute_error > strict_limit
    outlier_count = int(strict_outliers.sum().item())
    if outlier_count == 0:
        return {
            "strict_allclose": True,
            "operation_envelope_used": False,
            "operation_envelope_value_count": 0,
            "maximum_operation_envelope_ratio": 0.0,
            "maximum_operation_envelope_bound": 0.0,
            "unit_roundoff": _ieee_unit_roundoff(torch, reference_logits.dtype),
            "reduction_width": hidden_size,
        }

    unit_roundoff = _ieee_unit_roundoff(torch, reference_logits.dtype)
    reduction_product = hidden_size * unit_roundoff
    if reduction_product >= 1.0:
        raise Lfm2CircuitTraceCaptureError(
            "The LM-head reduction width has no finite IEEE gamma bound."
        )
    gamma = reduction_product / (1.0 - reduction_product)
    float64_reduction_product = hidden_size * (2.0**-53)
    float64_gamma = float64_reduction_product / (1.0 - float64_reduction_product)
    envelope_evaluation_inflation = 1.0 / (1.0 - float64_gamma)
    tiny = float(torch.finfo(reference_logits.dtype).tiny)
    underflow_allowance = 2.0 * (hidden_size + 1) * tiny

    flattened_outliers = torch.nonzero(
        strict_outliers.reshape(-1),
        as_tuple=False,
    ).reshape(-1)
    flattened_reference_input = reference_head_input.to(torch.float64).reshape(
        -1, hidden_size
    )
    flattened_candidate_input = candidate_head_input.to(torch.float64).reshape(
        -1, hidden_size
    )
    weight64 = (
        lm_head_weight.to(device="cpu", dtype=torch.float64)
        if lm_head_weight64 is None
        else lm_head_weight64
    )
    if (
        weight64.device.type != "cpu"
        or weight64.dtype != torch.float64
        or tuple(weight64.shape) != tuple(lm_head_weight.shape)
    ):
        raise Lfm2CircuitTraceCaptureError(
            "The precomputed LM-head replay bound weight is invalid."
        )
    flattened_error = absolute_error.reshape(-1)
    maximum_ratio = 0.0
    maximum_bound = 0.0
    maximum_rejected_error = 0.0
    maximum_rejected_bound = 0.0
    chunk_size = 2048
    for start in range(0, outlier_count, chunk_size):
        indices = flattened_outliers[start : start + chunk_size]
        row_indices = torch.div(indices, vocabulary_size, rounding_mode="floor")
        vocabulary_indices = indices.remainder(vocabulary_size)
        reference_input = flattened_reference_input.index_select(0, row_indices)
        candidate_input = flattened_candidate_input.index_select(0, row_indices)
        weight_rows = weight64.index_select(0, vocabulary_indices).abs()
        input_transport = (weight_rows * (reference_input - candidate_input).abs()).sum(
            dim=-1
        )
        reduction_magnitude = (weight_rows * reference_input.abs()).sum(dim=-1) + (
            weight_rows * candidate_input.abs()
        ).sum(dim=-1)
        bound = (
            input_transport + gamma * reduction_magnitude
        ) * envelope_evaluation_inflation
        bound = bound + underflow_allowance
        observed = flattened_error.index_select(0, indices)
        rejected = observed > bound
        maximum_bound = max(maximum_bound, float(bound.max().item()))
        positive_floor = max(
            underflow_allowance, float(torch.finfo(torch.float64).tiny)
        )
        maximum_ratio = max(
            maximum_ratio,
            float((observed / bound.clamp_min(positive_floor)).max().item()),
        )
        if bool(rejected.any().item()):
            maximum_rejected_error = max(
                maximum_rejected_error,
                float(observed[rejected].max().item()),
            )
            maximum_rejected_bound = max(
                maximum_rejected_bound,
                float(bound[rejected].max().item()),
            )

    if maximum_rejected_error > 0.0:
        raise Lfm2CircuitTraceCaptureError(
            f"Single-pass replay tensor {name!r} differs from the canonical "
            "prefix trace beyond its IEEE LM-head dot-product error envelope "
            f"(maximum rejected error {maximum_rejected_error:.6g}, "
            f"maximum corresponding bound {maximum_rejected_bound:.6g})."
        )
    return {
        "strict_allclose": False,
        "operation_envelope_used": True,
        "operation_envelope_value_count": outlier_count,
        "maximum_operation_envelope_ratio": maximum_ratio,
        "maximum_operation_envelope_bound": maximum_bound,
        "unit_roundoff": unit_roundoff,
        "reduction_width": hidden_size,
    }


def _authenticate_child_replay_batch_record(
    batch: Lfm2ChildReplayBatch,
) -> dict[str, Any]:
    record = batch.record
    try:
        _verify_seal(
            record,
            "replay_batch_sha256",
            f"child replay batch {batch.index}",
        )
    except Lfm2CircuitTraceCorruptionError as exc:
        raise Lfm2CircuitTraceCaptureError(
            f"Child replay batch {batch.index} authentication failed."
        ) from exc
    inventory = record.get("tensors")
    if (
        record.get("schema") != CHILD_REPLAY_BATCH_SCHEMA
        or record.get("batch_index") != batch.index
        or record.get("input_sha256") != batch.input_sha256
        or not isinstance(inventory, Mapping)
        or record.get("tensor_count") != len(inventory)
        or set(inventory) != set(batch.tensors)
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Child replay batch {batch.index} record binding is invalid."
        )
    profile_body = record.get("child_evidence_profile")
    profile_sha256 = record.get("child_evidence_profile_sha256")
    if profile_body is None:
        if profile_sha256 is not None:
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay batch {batch.index} has an orphaned profile seal."
            )
    else:
        try:
            profile = Lfm2ChildEvidenceProfile.from_dict(profile_body)
        except (TypeError, ValueError) as exc:
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay batch {batch.index} profile authentication failed."
            ) from exc
        if (
            profile.profile_sha256 != profile_sha256
            or profile.selected_layer_indices
            != tuple(record.get("selected_layer_indices", ()))
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay batch {batch.index} profile binding is invalid."
            )
    return record


def _authenticate_child_replay_tensor_binding(
    torch: Any,
    batch: Lfm2ChildReplayBatch,
    name: str,
    tensor: Any,
    *,
    authenticated_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not torch.is_tensor(tensor):
        raise Lfm2CircuitTraceCaptureError(
            f"Child replay tensor {name!r} is not a tensor."
        )
    if tensor.layout != torch.strided or tensor.device.type != "cpu":
        raise Lfm2CircuitTraceCaptureError(
            f"Child replay tensor {name!r} is not a detached CPU strided tensor."
        )
    record = (
        _authenticate_child_replay_batch_record(batch)
        if authenticated_record is None
        else authenticated_record
    )
    inventory = record.get("tensors")
    expected = inventory.get(name) if isinstance(inventory, Mapping) else None
    actual = _tensor_descriptor(torch, tensor)
    if not isinstance(expected, Mapping) or dict(expected) != actual:
        raise Lfm2CircuitTraceCaptureError(
            f"Child replay tensor {name!r} shape/dtype/hash binding failed."
        )
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
        torch.isfinite(tensor).all().item()
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Child replay tensor {name!r} contains NaN or infinity."
        )
    return actual


def _recorded_non_logit_operation_profile(
    name: str,
    tensor: Any,
    envelope_policy: Mapping[str, Any],
) -> tuple[str, int, int]:
    layer_count = int(envelope_policy["model_layer_count"])
    layer_fan_ins = tuple(
        int(value) for value in envelope_policy["layer_parameter_fan_ins"]
    )
    if len(layer_fan_ins) != layer_count:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay non-logit envelope layer inventory is invalid."
        )
    if name.endswith(".output.lm_head_input"):
        operation_class = "complete-residual-stack-plus-final-rmsnorm"
        path_depth = 2 * layer_count + 1
        recorded_width = int(envelope_policy["maximum_parameter_fan_in"])
    elif ".layer." in name:
        _phase, layer_suffix = name.split(".layer.", 1)
        try:
            layer_text, endpoint = layer_suffix.split(".", 1)
            layer_index = int(layer_text)
        except (TypeError, ValueError) as exc:
            raise Lfm2CircuitTraceCaptureError(
                f"Replay tensor {name!r} has no recorded producer path."
            ) from exc
        if layer_index < 0 or layer_index >= layer_count:
            raise Lfm2CircuitTraceCaptureError(
                f"Replay tensor {name!r} has an invalid producer layer."
            )
        recorded_width = layer_fan_ins[layer_index]
        if endpoint == "residual.pre":
            operation_class = "residual-stack-entry"
            path_depth = 2 * layer_index
        elif (
            endpoint.startswith("operator.")
            or endpoint.startswith("attention.")
            or endpoint.startswith("short_conv.")
            or endpoint == "residual.post_operator"
        ):
            operation_class = "residual-stack-through-operator-branch"
            path_depth = 2 * layer_index + 1
        elif endpoint.startswith("mlp.") or endpoint == "residual.post_layer":
            operation_class = "residual-stack-through-mlp-branch"
            path_depth = 2 * layer_index + 2
        else:
            raise Lfm2CircuitTraceCaptureError(
                f"Replay tensor {name!r} has no recorded producer class."
            )
    else:
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} has no recorded non-logit operation path."
        )
    tensor_width = max((int(value) for value in tensor.shape), default=1)
    return operation_class, path_depth, max(recorded_width, tensor_width)


def _cross_kernel_non_logit_forward_error_check(
    torch: Any,
    *,
    name: str,
    reference_tensor: Any,
    candidate_tensor: Any,
    envelope_policy: Mapping[str, Any],
    reference_descriptor: Mapping[str, Any] | None = None,
    candidate_descriptor: Mapping[str, Any] | None = None,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Certify strict outliers against their recorded FP operation path.

    The envelope is not a tensor-wide magic epsilon.  Its gamma term is
    derived from the producing residual path depth, the actual parameter
    fan-in inventory, and the executed tensor dtype.  The factor of two
    covers the independent legal reduction trees used by the prefix oracle
    and the full-trajectory kernel.
    """

    if not torch.is_tensor(reference_tensor) or not torch.is_tensor(candidate_tensor):
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} comparison requires two tensors."
        )
    if (
        reference_tensor.layout != torch.strided
        or candidate_tensor.layout != torch.strided
        or reference_tensor.device.type != "cpu"
        or candidate_tensor.device.type != "cpu"
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} comparison changed recorded CPU device/layout."
        )
    if tuple(reference_tensor.shape) != tuple(candidate_tensor.shape):
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} comparison changed shape."
        )
    if reference_tensor.dtype != candidate_tensor.dtype:
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} comparison changed dtype."
        )
    if reference_tensor.is_complex() or not reference_tensor.is_floating_point():
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} envelope requires a real floating dtype."
        )
    if not bool(torch.isfinite(reference_tensor).all().item()) or not bool(
        torch.isfinite(candidate_tensor).all().item()
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} contains NaN or infinity."
        )
    actual_reference_descriptor = _tensor_descriptor(torch, reference_tensor)
    actual_candidate_descriptor = _tensor_descriptor(torch, candidate_tensor)
    if (
        reference_descriptor is not None
        and dict(reference_descriptor) != actual_reference_descriptor
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} reference hash binding failed."
        )
    if (
        candidate_descriptor is not None
        and dict(candidate_descriptor) != actual_candidate_descriptor
    ):
        raise Lfm2CircuitTraceCaptureError(
            f"Replay tensor {name!r} runtime hash binding failed."
        )
    if envelope_policy.get("error_model") != CHILD_REPLAY_NON_LOGIT_ERROR_MODEL:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay non-logit error model is invalid."
        )

    operation_class, path_depth, reduction_width = (
        _recorded_non_logit_operation_profile(
            name,
            reference_tensor,
            envelope_policy,
        )
    )
    reference64 = reference_tensor.to(torch.float64)
    candidate64 = candidate_tensor.to(torch.float64)
    absolute_error = (reference64 - candidate64).abs()
    strict_limit = atol + rtol * candidate64.abs()
    strict_outliers = absolute_error > strict_limit
    outlier_count = int(strict_outliers.sum().item())
    maximum_error = float(absolute_error.max().item())
    unit_roundoff = _ieee_unit_roundoff(torch, reference_tensor.dtype)
    effective_reduction_terms = path_depth * reduction_width
    operation_envelope_used = outlier_count > 0
    maximum_ratio = 0.0
    maximum_bound = 0.0
    if outlier_count > 0:
        if effective_reduction_terms <= 0:
            raise Lfm2CircuitTraceCaptureError(
                f"Single-pass replay tensor {name!r} differs before any "
                "recorded cross-kernel floating operation."
            )
        reduction_product = effective_reduction_terms * unit_roundoff
        if reduction_product >= 1.0:
            raise Lfm2CircuitTraceCaptureError(
                f"Replay tensor {name!r} has no finite IEEE path gamma bound."
            )
        gamma = reduction_product / (1.0 - reduction_product)
        evaluation_product = effective_reduction_terms * (2.0**-53)
        evaluation_inflation = 1.0 / (1.0 - evaluation_product)
        if reference64.ndim == 0:
            reference_rms = reference64.abs()
            candidate_rms = candidate64.abs()
        else:
            reference_rms = (
                reference64.square()
                .mean(
                    dim=-1,
                    keepdim=True,
                )
                .sqrt()
            )
            candidate_rms = (
                candidate64.square()
                .mean(
                    dim=-1,
                    keepdim=True,
                )
                .sqrt()
            )
        scale = torch.maximum(reference64.abs(), candidate64.abs())
        scale = torch.maximum(scale, reference_rms)
        scale = torch.maximum(scale, candidate_rms)
        underflow_allowance = (
            2.0
            * (effective_reduction_terms + 1)
            * float(torch.finfo(reference_tensor.dtype).tiny)
        )
        bound = 2.0 * gamma * scale * evaluation_inflation + underflow_allowance
        rejected = strict_outliers & (absolute_error > bound)
        outlier_bounds = bound[strict_outliers]
        outlier_errors = absolute_error[strict_outliers]
        maximum_bound = float(outlier_bounds.max().item())
        positive_floor = max(
            underflow_allowance,
            float(torch.finfo(torch.float64).tiny),
        )
        maximum_ratio = float(
            (outlier_errors / outlier_bounds.clamp_min(positive_floor)).max().item()
        )
        if bool(rejected.any().item()):
            maximum_rejected_error = float(absolute_error[rejected].max().item())
            maximum_rejected_bound = float(bound[rejected].max().item())
            raise Lfm2CircuitTraceCaptureError(
                f"Single-pass replay tensor {name!r} differs beyond its "
                "recorded IEEE cross-kernel operation envelope (maximum "
                f"rejected error {maximum_rejected_error:.6g}, maximum "
                f"corresponding bound {maximum_rejected_bound:.6g})."
            )

    receipt_body = {
        "schema": ("neural-forge-lfm2-child-replay-non-logit-envelope-receipt-v1"),
        "tensor_name": name,
        "reference_tensor_sha256": actual_reference_descriptor["sha256"],
        "runtime_tensor_sha256": actual_candidate_descriptor["sha256"],
        "shape": actual_reference_descriptor["shape"],
        "dtype": actual_reference_descriptor["dtype"],
        "device": "cpu",
        "numeric_policy_sha256": envelope_policy["numeric_policy_sha256"],
        "envelope_sha256": envelope_policy["envelope_sha256"],
        "operation_class": operation_class,
        "path_depth": path_depth,
        "reduction_width": reduction_width,
        "effective_reduction_terms": effective_reduction_terms,
        "unit_roundoff": unit_roundoff,
        "strict_rtol": rtol,
        "strict_atol": atol,
        "strict_allclose": outlier_count == 0,
        "operation_envelope_used": operation_envelope_used,
        "operation_envelope_value_count": outlier_count,
        "maximum_abs_error": maximum_error,
        "maximum_operation_envelope_ratio": maximum_ratio,
        "maximum_operation_envelope_bound": maximum_bound,
        "finite": True,
        "shape_dtype_device_hash_authenticated": True,
    }
    return {
        **receipt_body,
        "receipt_sha256": _json_sha256(receipt_body),
    }


def _verify_child_replay_tensor_equivalence(
    torch: Any,
    child_model: Any,
    reference: Lfm2ChildReplay,
    candidate_batches: Sequence[Lfm2ChildReplayBatch],
    *,
    child_fingerprint: str,
    corpus_fingerprint: str,
    selected_layer_indices: Sequence[int],
    config: Lfm2CircuitTraceCaptureConfig,
    execution: Mapping[str, Any],
    implementation: Mapping[str, Any],
    numeric_policy: Mapping[str, Any],
) -> dict[str, Any]:
    authenticated_numeric_policy = _authenticate_child_replay_numeric_policy(
        numeric_policy
    )
    non_logit_envelope_policy = authenticated_numeric_policy[
        "non_logit_envelope_policy"
    ]
    reference_metadata = reference.metadata
    reference_selection = reference_metadata.get("selection")
    if (
        not isinstance(reference_selection, Mapping)
        or reference_selection.get("replay_capture_mode")
        != CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    ):
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference must be a canonical "
            "prefix-by-prefix child replay."
        )
    if reference_selection.get("capture") != config.as_dict():
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference uses a different capture config."
        )
    if reference_metadata.get("execution") != dict(execution):
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference uses a different execution runtime."
        )
    if reference_metadata.get("implementation") != dict(implementation):
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference uses a different implementation."
        )
    if reference_metadata.get("numeric_execution_policy") != (
        authenticated_numeric_policy
    ):
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference uses a different numeric policy."
        )
    if reference.child_fingerprint != child_fingerprint:
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference belongs to a different child state."
        )
    if reference.corpus_fingerprint != corpus_fingerprint:
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference uses a different token corpus."
        )
    if reference.selected_layer_indices != tuple(selected_layer_indices):
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference uses a different layer selection."
        )
    if len(reference) != len(candidate_batches):
        raise Lfm2CircuitTraceCaptureError(
            "The causal-equivalence reference batch count changed."
        )

    lm_head, _lm_head_runtime_binding = _single_pass_lm_head_runtime_binding(
        torch,
        child_model,
    )
    lm_head_weight_value = lm_head.weight
    lm_head_weight = _cpu_tensor(
        torch,
        lm_head_weight_value,
        "child replay LM-head weight",
    )
    lm_head_weight64 = lm_head_weight.to(torch.float64)

    max_abs_error = 0.0
    compared_tensors = 0
    compared_logits = 0
    operation_envelope_logit_count = 0
    operation_envelope_value_count = 0
    maximum_operation_envelope_ratio = 0.0
    maximum_operation_envelope_bound = 0.0
    non_logit_operation_envelope_tensor_count = 0
    non_logit_operation_envelope_value_count = 0
    maximum_non_logit_operation_envelope_ratio = 0.0
    maximum_non_logit_operation_envelope_bound = 0.0
    non_logit_envelope_receipt_sha256s: list[str] = []
    for index, candidate in enumerate(candidate_batches):
        expected = reference.load_batch(index)
        expected_record = _authenticate_child_replay_batch_record(expected)
        candidate_record = _authenticate_child_replay_batch_record(candidate)
        candidate_capture = candidate_record.get("capture")
        candidate_contract = (
            candidate_capture.get("causal_equivalence_contract")
            if isinstance(candidate_capture, Mapping)
            else None
        )
        if (
            not isinstance(candidate_contract, Mapping)
            or candidate_contract.get("numeric_policy_sha256")
            != authenticated_numeric_policy["numeric_policy_sha256"]
            or candidate_contract.get("replay_numeric_policy_sha256")
            != authenticated_numeric_policy["replay_numeric_policy_sha256"]
            or candidate_contract.get("non_logit_envelope_sha256")
            != authenticated_numeric_policy["non_logit_envelope_sha256"]
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay batch {index} changed numeric contract binding."
            )
        if expected.phase_order != candidate.phase_order:
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay phase order differs in batch {index}."
            )
        if set(expected.tensors) != set(candidate.tensors):
            missing = sorted(set(expected.tensors).difference(candidate.tensors))
            extra = sorted(set(candidate.tensors).difference(expected.tensors))
            raise Lfm2CircuitTraceCaptureError(
                "Single-pass replay tensor schema differs from its canonical "
                f"reference in batch {index}; missing={missing}, extra={extra}."
            )
        for name in sorted(expected.tensors):
            left = expected.tensors[name]
            right = candidate.tensors[name]
            left_descriptor = _authenticate_child_replay_tensor_binding(
                torch,
                expected,
                name,
                left,
                authenticated_record=expected_record,
            )
            right_descriptor = _authenticate_child_replay_tensor_binding(
                torch,
                candidate,
                name,
                right,
                authenticated_record=candidate_record,
            )
            if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
                raise Lfm2CircuitTraceCaptureError(
                    f"Single-pass replay tensor {name!r} changed shape or dtype."
                )
            if left.is_floating_point() or left.is_complex():
                if left.is_complex():
                    raise Lfm2CircuitTraceCaptureError(
                        f"Single-pass replay tensor {name!r} is complex."
                    )
                difference = left.to(torch.float64) - right.to(torch.float64)
                tensor_error = float(difference.abs().max().item())
                max_abs_error = max(max_abs_error, tensor_error)
                if name.endswith(".logits"):
                    phase = name[: -len(".logits")]
                    head_input_name = f"{phase}.output.lm_head_input"
                    left_head_input = expected.tensors.get(head_input_name)
                    right_head_input = candidate.tensors.get(head_input_name)
                    if not torch.is_tensor(left_head_input) or not torch.is_tensor(
                        right_head_input
                    ):
                        raise Lfm2CircuitTraceCaptureError(
                            f"Replay logits {name!r} have no sealed LM-head operands."
                        )
                    logit_check = _lm_head_logit_forward_error_check(
                        torch,
                        name=name,
                        reference_logits=left,
                        candidate_logits=right,
                        reference_head_input=left_head_input,
                        candidate_head_input=right_head_input,
                        lm_head_weight=lm_head_weight,
                        lm_head_weight64=lm_head_weight64,
                        rtol=config.logit_rtol,
                        atol=config.logit_atol,
                    )
                    if logit_check["operation_envelope_used"]:
                        operation_envelope_logit_count += 1
                    operation_envelope_value_count += int(
                        logit_check["operation_envelope_value_count"]
                    )
                    maximum_operation_envelope_ratio = max(
                        maximum_operation_envelope_ratio,
                        float(logit_check["maximum_operation_envelope_ratio"]),
                    )
                    maximum_operation_envelope_bound = max(
                        maximum_operation_envelope_bound,
                        float(logit_check["maximum_operation_envelope_bound"]),
                    )
                else:
                    non_logit_check = _cross_kernel_non_logit_forward_error_check(
                        torch,
                        name=name,
                        reference_tensor=left,
                        candidate_tensor=right,
                        envelope_policy=non_logit_envelope_policy,
                        reference_descriptor=left_descriptor,
                        candidate_descriptor=right_descriptor,
                        rtol=config.logit_rtol,
                        atol=config.logit_atol,
                    )
                    non_logit_envelope_receipt_sha256s.append(
                        non_logit_check["receipt_sha256"]
                    )
                    if non_logit_check["operation_envelope_used"]:
                        non_logit_operation_envelope_tensor_count += 1
                    non_logit_operation_envelope_value_count += int(
                        non_logit_check["operation_envelope_value_count"]
                    )
                    maximum_non_logit_operation_envelope_ratio = max(
                        maximum_non_logit_operation_envelope_ratio,
                        float(non_logit_check["maximum_operation_envelope_ratio"]),
                    )
                    maximum_non_logit_operation_envelope_bound = max(
                        maximum_non_logit_operation_envelope_bound,
                        float(non_logit_check["maximum_operation_envelope_bound"]),
                    )
            elif not bool(torch.equal(left, right)):
                raise Lfm2CircuitTraceCaptureError(
                    f"Single-pass replay tensor {name!r} differs exactly from "
                    "the canonical prefix trace."
                )
            if name.endswith(".logits"):
                compared_logits += 1
                if config.require_top1_agreement and not bool(
                    torch.equal(
                        torch.argmax(left, dim=-1),
                        torch.argmax(right, dim=-1),
                    )
                ):
                    raise Lfm2CircuitTraceCaptureError(
                        f"Single-pass replay logits {name!r} choose a different "
                        "top-1 token than the canonical prefix trace."
                    )
            compared_tensors += 1
    return {
        "reference_supplied": True,
        "tolerance_equivalent": True,
        "compared_tensors": compared_tensors,
        "compared_logits": compared_logits,
        "max_abs_error": max_abs_error,
        "rtol": config.logit_rtol,
        "atol": config.logit_atol,
        "top1_required": config.require_top1_agreement,
        "logit_error_model": CHILD_REPLAY_LOGIT_ERROR_MODEL,
        "non_logit_error_model": CHILD_REPLAY_NON_LOGIT_ERROR_MODEL,
        "non_logit_tolerance_unchanged": False,
        "numeric_policy_sha256": authenticated_numeric_policy["numeric_policy_sha256"],
        "replay_numeric_policy_sha256": authenticated_numeric_policy[
            "replay_numeric_policy_sha256"
        ],
        "non_logit_envelope_sha256": authenticated_numeric_policy[
            "non_logit_envelope_sha256"
        ],
        "operation_envelope_logit_count": operation_envelope_logit_count,
        "operation_envelope_value_count": operation_envelope_value_count,
        "maximum_operation_envelope_ratio": maximum_operation_envelope_ratio,
        "maximum_operation_envelope_bound": maximum_operation_envelope_bound,
        "non_logit_operation_envelope_tensor_count": (
            non_logit_operation_envelope_tensor_count
        ),
        "non_logit_operation_envelope_value_count": (
            non_logit_operation_envelope_value_count
        ),
        "maximum_non_logit_operation_envelope_ratio": (
            maximum_non_logit_operation_envelope_ratio
        ),
        "maximum_non_logit_operation_envelope_bound": (
            maximum_non_logit_operation_envelope_bound
        ),
        "non_logit_envelope_receipts_sha256": _json_sha256(
            non_logit_envelope_receipt_sha256s
        ),
        "shape_dtype_device_hash_and_finiteness_authenticated": True,
    }


def _validate_child_replay_capture(
    tensors: Mapping[str, Any],
    capture: Mapping[str, Any],
    selected_layer_indices: Sequence[int],
    phase_order: Sequence[str],
    replay_capture_mode: str,
    child_evidence_profile: Lfm2ChildEvidenceProfile | None,
) -> None:
    expected_source = (
        "use_cache_false_full_trajectory_slices"
        if replay_capture_mode == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
        else "use_cache_false_full_prefix"
    )
    if (
        capture.get("decode_trace_mode") != FULL_PREFIX_CANONICAL_TRACE_MODE
        or capture.get("canonical_trace_source") != expected_source
        or capture.get("cached_logits_accepted") is not False
        or capture.get("cached_forward_count") != 0
        or capture.get("lm_head_input_captured_for_replay_equivalence")
        is not (
            True
            if child_evidence_profile is None
            else child_evidence_profile.retain_lm_head_input
        )
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Child replay did not use the canonical full-prefix-only path."
        )
    diagnostics = capture.get("phase_diagnostics")
    if not isinstance(diagnostics, Mapping) or tuple(diagnostics) != tuple(phase_order):
        raise Lfm2CircuitTraceCaptureError(
            "Child replay phase order changed during capture."
        )
    expected_layers = tuple(f"layer.{index:03d}" for index in selected_layer_indices)
    profile_body = capture.get("child_evidence_profile")
    if child_evidence_profile is None:
        if profile_body is not None:
            raise Lfm2CircuitTraceCaptureError(
                "Legacy child replay unexpectedly carries a retention profile."
            )
    elif profile_body != child_evidence_profile.as_dict():
        raise Lfm2CircuitTraceCaptureError(
            "Child replay evidence profile changed during capture."
        )
    for phase in phase_order:
        phase_diagnostics = diagnostics.get(phase)
        if (
            not isinstance(phase_diagnostics, Mapping)
            or tuple(phase_diagnostics) != expected_layers
        ):
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay layer order changed in phase {phase!r}."
            )
        logits_present = f"{phase}.logits" in tensors
        expected_logits = (
            True
            if child_evidence_profile is None
            else child_evidence_profile.retain_logits
        )
        if logits_present is not expected_logits:
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay phase {phase!r} changed logits retention."
            )
        head_present = f"{phase}.output.lm_head_input" in tensors
        expected_head = (
            True
            if child_evidence_profile is None
            else child_evidence_profile.retain_lm_head_input
        )
        if head_present is not expected_head:
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay phase {phase!r} changed LM-head retention."
            )
        for layer_index in selected_layer_indices:
            residual_name = f"{phase}.layer.{layer_index:03d}.residual.pre"
            if residual_name not in tensors:
                raise Lfm2CircuitTraceCaptureError(
                    f"Child replay phase {phase!r} did not capture selected "
                    f"layer {layer_index}."
                )
            if child_evidence_profile is not None:
                prefix = f"{phase}.layer.{layer_index:03d}."
                observed_suffixes = tuple(
                    sorted(
                        name[len(prefix) :]
                        for name in tensors
                        if name.startswith(prefix)
                    )
                )
                if observed_suffixes != child_evidence_profile.suffixes_for_layer(
                    layer_index
                ):
                    raise Lfm2CircuitTraceCaptureError(
                        f"Child replay phase {phase!r} layer {layer_index} "
                        "differs from its exact retained suffix inventory."
                    )
        if phase.startswith("decode."):
            forced_present = f"{phase}.forced_token_ids" in tensors
            expected_forced = (
                True
                if child_evidence_profile is None
                else child_evidence_profile.retain_phase_forced_token_ids
            )
            if forced_present is not expected_forced:
                raise Lfm2CircuitTraceCaptureError(
                    f"Child replay phase {phase!r} changed forced-token retention."
                )

    allowed_layers = frozenset(selected_layer_indices)
    observed_layers: set[int] = set()
    for name in tensors:
        if name.startswith("cached_prefill.") or name.endswith(".full_prefix_logits"):
            raise Lfm2CircuitTraceCaptureError(
                "Child replay retained a cached or verification-only tensor."
            )
        marker = ".layer."
        if marker not in name:
            continue
        _phase, suffix = name.split(marker, 1)
        layer_text = suffix.split(".", 1)[0]
        try:
            layer_index = int(layer_text)
        except ValueError as exc:
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay tensor {name!r} has an invalid layer index."
            ) from exc
        if layer_index not in allowed_layers:
            raise Lfm2CircuitTraceCaptureError(
                f"Child replay captured unselected layer {layer_index}."
            )
        observed_layers.add(layer_index)
    if observed_layers != allowed_layers:
        raise Lfm2CircuitTraceCaptureError(
            "Child replay did not capture every selected layer."
        )


class _Lfm2ChildReplaySealPipeline:
    """Overlap batch N's inventory hashing and sealing with batch N+1's replay.

    Sealing reads nothing but already detached CPU tensors, so it is handed to
    one background worker while the next batch replays on the calling thread.
    Results are collected strictly in submission order, and the sealed record
    body is built exactly as the serial path built it, so every record hash,
    canonical JSON document, and ordered batch entry stays byte-identical.
    """

    __slots__ = ("_executor", "_futures", "_telemetry", "_torch")

    def __init__(
        self,
        torch: Any,
        *,
        telemetry: _CaptureTelemetry | None = None,
        enabled: bool = True,
    ) -> None:
        self._torch = torch
        self._telemetry = telemetry
        self._futures: list[concurrent.futures.Future[Lfm2ChildReplayBatch]] = []
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="lfm2-replay-seal",
            )
            if enabled
            else None
        )

    def submit(self, body: Mapping[str, Any], tensors: Mapping[str, Any]) -> None:
        if self._executor is None:
            completed: concurrent.futures.Future[Lfm2ChildReplayBatch] = (
                concurrent.futures.Future()
            )
            try:
                completed.set_result(self._seal(dict(body), tensors))
            except Exception as exc:
                completed.set_exception(exc)
            self._futures.append(completed)
            return
        self._futures.append(self._executor.submit(self._seal, dict(body), tensors))

    def _seal(
        self,
        body: dict[str, Any],
        tensors: Mapping[str, Any],
    ) -> Lfm2ChildReplayBatch:
        started = time.perf_counter()
        inventory = _ordered_tensor_descriptors(self._torch, tensors)
        record = _sealed_document(
            {
                **body,
                "tensor_count": len(inventory),
                "tensors": inventory,
            },
            "replay_batch_sha256",
        )
        elapsed = time.perf_counter() - started
        if self._telemetry is not None:
            self._telemetry.hash_and_seal_seconds += elapsed
            self._telemetry.sealed_tensor_count += len(inventory)
            self._telemetry.batches.append(
                {
                    "batch_index": int(body["batch_index"]),
                    "tensor_count": len(inventory),
                    "hash_and_seal_seconds": round(elapsed, 6),
                }
            )
        return Lfm2ChildReplayBatch(
            index=int(body["batch_index"]),
            input_sha256=str(body["input_sha256"]),
            _tensors=MappingProxyType(dict(tensors)),
            _record=MappingProxyType(record),
        )

    def results(self) -> list[Lfm2ChildReplayBatch]:
        return [future.result() for future in self._futures]

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)


def capture_lfm2_child_replay(
    child_model: Any,
    replay_batches: Iterable[
        Mapping[str, Any] | Lfm2CircuitTraceBatch | Lfm2ChildReplayBatch
    ],
    *,
    selected_layer_indices: Sequence[int],
    config: Lfm2CircuitTraceCaptureConfig | None = None,
    input_device: Any | None = None,
    replay_capture_mode: str = CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    causal_equivalence_reference: Lfm2ChildReplay | None = None,
    child_evidence_profile: Lfm2ChildEvidenceProfile | None = None,
) -> Lfm2ChildReplay:
    """Capture selected live-child layers from exact token batches in memory.

    This API deliberately has no tokenizer, parent model, or filesystem
    argument. The default mode keeps the reference behavior: every phase is an
    uncached full-prefix child forward. The opt-in ``single_pass_causal`` mode
    performs one uncached full-trajectory forward per batch and slices the
    exact same causal token rows into the existing phase schema. Layer
    selection limits retained hooks only; all preceding child layers still
    execute and therefore feed their current installed state into each
    selected layer.
    """

    torch = _torch()
    selected_replay_mode = _validate_child_replay_mode(replay_capture_mode)
    selected_config = (
        Lfm2CircuitTraceCaptureConfig(
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE
        )
        if config is None
        else config
    )
    if not isinstance(selected_config, Lfm2CircuitTraceCaptureConfig):
        raise TypeError("config must be Lfm2CircuitTraceCaptureConfig or None.")
    if child_evidence_profile is not None and not isinstance(
        child_evidence_profile,
        Lfm2ChildEvidenceProfile,
    ):
        raise TypeError(
            "child_evidence_profile must be Lfm2ChildEvidenceProfile or None."
        )
    if (
        child_evidence_profile is not None
        and selected_replay_mode != CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Point-specific child evidence profiles currently require "
            "prefix_by_prefix_canonical replay."
        )
    if (
        child_evidence_profile is not None
        and selected_config.capture_causal_circuit_evidence
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Point-specific child evidence profiles do not yet authenticate "
            "causal-circuit output-boundary globals."
        )
    if (
        selected_replay_mode == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
        and selected_config.capture_causal_circuit_evidence
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Single-pass child replay does not yet support causal-circuit "
            "evidence: attention histories require prefix truncation, "
            "attention scores/probabilities require query-and-key-axis "
            "slicing, and ShortConv histories require prefix semantics. Use "
            "canonical replay for this evidence schema."
        )
    if selected_config.decode_trace_mode != FULL_PREFIX_CANONICAL_TRACE_MODE:
        raise ValueError(
            "Child replay requires decode_trace_mode='full_prefix_canonical'."
        )

    prepared = _prepare_batches(
        torch,
        _exact_replay_batch_mappings(replay_batches),
        selected_config,
    )
    layers, contracts, topology = _layer_topology(child_model)
    selected_indices = _selected_layer_indices(
        len(layers),
        selected_layer_indices,
    )
    if (
        child_evidence_profile is not None
        and child_evidence_profile.selected_layer_indices != selected_indices
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Child evidence profile differs from selected_layer_indices."
        )
    device = _resolve_input_device(torch, child_model, input_device)
    telemetry = _CaptureTelemetry()
    fingerprint_started = time.perf_counter()
    child_before = fingerprint_lfm2_parent(child_model)
    telemetry.fingerprint_seconds += time.perf_counter() - fingerprint_started
    corpus = _corpus_identity(prepared)
    execution = _execution_identity(child_model, device)
    implementation = implementation_identity()
    replay_numeric_policy = compile_lfm2_child_replay_numeric_policy(
        child_model,
        input_device=device,
    )
    causal_contract = None
    if selected_replay_mode == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE:
        causal_contract = _single_pass_causal_contract(
            child_model,
            layers,
            contracts,
            execution,
            device,
            prepared,
            replay_numeric_policy,
        )

    child_forward_count = 0
    canonical_prefix_forward_count = 0
    seal_pipeline = _Lfm2ChildReplaySealPipeline(
        torch,
        telemetry=telemetry,
        enabled=_SEAL_PIPELINE_ENABLED,
    )
    try:
        for batch in prepared:
            phase_rows = _child_replay_phase_rows(batch, selected_config)
            phase_order = tuple(row["phase"] for row in phase_rows)
            replay_started = time.perf_counter()
            if selected_replay_mode == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE:
                if causal_contract is None:  # pragma: no cover - guarded above
                    raise AssertionError("missing single-pass causal contract")
                tensors, capture = _capture_child_batch_single_pass(
                    torch,
                    child_model,
                    layers,
                    contracts,
                    batch,
                    selected_config,
                    device,
                    selected_indices,
                    causal_contract,
                    telemetry=telemetry,
                )
                child_forward_count += 1
            else:
                tensors, capture = _capture_batch(
                    torch,
                    child_model,
                    layers,
                    contracts,
                    batch,
                    selected_config,
                    device,
                    selected_layer_indices=selected_indices,
                    capture_replay_lm_head_input=(
                        True
                        if child_evidence_profile is None
                        else child_evidence_profile.retain_lm_head_input
                    ),
                    child_evidence_profile=child_evidence_profile,
                    telemetry=telemetry,
                )
                capture = {
                    **capture,
                    "lm_head_input_captured_for_replay_equivalence": (
                        True
                        if child_evidence_profile is None
                        else child_evidence_profile.retain_lm_head_input
                    ),
                    "child_replay_capture_mode": CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                    "child_forward_count": len(phase_rows),
                    "canonical_prefix_forward_count": len(phase_rows),
                }
                child_forward_count += len(phase_rows)
            telemetry.replay_seconds += time.perf_counter() - replay_started
            canonical_prefix_forward_count += len(phase_rows)
            _validate_child_replay_capture(
                tensors,
                capture,
                selected_indices,
                phase_order,
                selected_replay_mode,
                child_evidence_profile,
            )
            # Batch N's inventory hashing and sealing runs on the pipeline
            # worker while batch N+1 replays.  Results are collected in
            # submission order below, so the sealed bytes are unchanged.
            seal_pipeline.submit(
                {
                    "schema": CHILD_REPLAY_BATCH_SCHEMA,
                    "batch_index": batch.index,
                    "input_sha256": batch.input_sha256,
                    "selected_layer_indices": list(selected_indices),
                    "phase_order": list(phase_order),
                    "phase_rows": list(phase_rows),
                    "child_evidence_profile": (
                        None
                        if child_evidence_profile is None
                        else child_evidence_profile.as_dict()
                    ),
                    "child_evidence_profile_sha256": (
                        None
                        if child_evidence_profile is None
                        else child_evidence_profile.profile_sha256
                    ),
                    "capture": dict(capture),
                },
                tensors,
            )
        memory_batches = seal_pipeline.results()
    finally:
        seal_pipeline.close()

    reference_verification = {
        "reference_supplied": False,
        "tolerance_equivalent": None,
    }
    if causal_equivalence_reference is not None:
        if selected_replay_mode != CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE:
            raise ValueError(
                "causal_equivalence_reference is only valid with "
                "replay_capture_mode='single_pass_causal'."
            )
        if not isinstance(causal_equivalence_reference, Lfm2ChildReplay):
            raise TypeError(
                "causal_equivalence_reference must be Lfm2ChildReplay or None."
            )
        reference_verification = _verify_child_replay_tensor_equivalence(
            torch,
            child_model,
            causal_equivalence_reference,
            memory_batches,
            child_fingerprint=child_before.identity_sha256,
            corpus_fingerprint=corpus["corpus_sha256"],
            selected_layer_indices=selected_indices,
            config=selected_config,
            execution=execution,
            implementation=implementation,
            numeric_policy=replay_numeric_policy,
        )

    # The after fingerprint is an integrity proof, not redundant work: replay
    # runs under ``torch.inference_mode()``, which suppresses version-counter
    # bumps, so no cheap local signal can prove a frozen parameter was not
    # mutated in place inside that window.  It is therefore recomputed, with
    # the per-parameter hashing parallelized inside fingerprint_lfm2_parent.
    fingerprint_started = time.perf_counter()
    child_after = fingerprint_lfm2_parent(child_model)
    telemetry.fingerprint_seconds += time.perf_counter() - fingerprint_started
    if child_after != child_before:
        raise Lfm2CircuitTraceCaptureError(
            "Child state or configuration changed during in-memory replay."
        )

    selected_topology_body = {
        "schema": "neural-forge-lfm2-child-selected-topology-v1",
        "model_layer_count": len(layers),
        "selected_layer_indices": list(selected_indices),
        "layers": [topology["layers"][index] for index in selected_indices],
    }
    selected_topology = {
        **selected_topology_body,
        "selected_topology_sha256": _json_sha256(selected_topology_body),
    }
    capture_config = selected_config.as_dict()
    selection_body = {
        "schema": "neural-forge-lfm2-child-replay-selection-v2",
        "selected_layer_indices": list(selected_indices),
        "capture": capture_config,
        "replay_capture_mode": selected_replay_mode,
        "selected_topology": selected_topology,
        "child_evidence_profile": (
            None
            if child_evidence_profile is None
            else child_evidence_profile.as_dict()
        ),
        "child_evidence_profile_sha256": (
            None
            if child_evidence_profile is None
            else child_evidence_profile.profile_sha256
        ),
    }
    selection = {
        **selection_body,
        "selection_sha256": _json_sha256(selection_body),
    }
    metadata = _sealed_document(
        {
            "schema": CHILD_REPLAY_SCHEMA,
            "child": child_before.as_dict(),
            "corpus": corpus,
            "implementation": implementation,
            "layer_topology": topology,
            "selection": selection,
            "execution": execution,
            "numeric_execution_policy": replay_numeric_policy,
            "batch_count": len(memory_batches),
            "ordered_batches": [
                {
                    "batch_index": batch.index,
                    "input_sha256": batch.input_sha256,
                    "replay_batch_sha256": batch._record["replay_batch_sha256"],
                    "phase_order": list(batch.phase_order),
                    "child_evidence_profile_sha256": batch._record.get(
                        "child_evidence_profile_sha256"
                    ),
                }
                for batch in memory_batches
            ],
            "forward_counts": {
                "child": child_forward_count,
                "parent": 0,
            },
            "replay_benchmark": {
                "batch_count": len(memory_batches),
                "phase_count": canonical_prefix_forward_count,
                "actual_child_forwards": child_forward_count,
                "canonical_prefix_child_forwards": canonical_prefix_forward_count,
                "saved_child_forwards": (
                    canonical_prefix_forward_count - child_forward_count
                ),
            },
            "causal_equivalence": reference_verification,
            "runtime": {
                "tokenizer_accessed": False,
                "parent_model_supplied": False,
                "parent_forward_count": 0,
                "disk_artifacts_written": 0,
                "intermediate_cache": "memory_only",
                "child_state_preserved": True,
            },
        },
        "replay_sha256",
    )
    return Lfm2ChildReplay(
        _batches=tuple(memory_batches),
        _metadata=MappingProxyType(metadata),
        _capture_timing=MappingProxyType(telemetry.as_dict()),
    )


def capture_lfm2_child_embedding_replay(
    child_model: Any,
    inputs_embeds: Any,
    *,
    attention_mask: Any,
    selected_layer_indices: Sequence[int],
    sample_last_tokens: int,
    position_ids: Any | None = None,
    max_capture_bytes: int = 512 * 1024 * 1024,
    input_device: Any | None = None,
) -> Lfm2ChildEmbeddingReplay:
    """Capture one bounded donor-time child trajectory from supplied embeddings.

    This is deliberately narrower than :func:`capture_lfm2_child_replay`.
    It accepts no tokenizer, token IDs, parent, cache path, or optimization
    callback.  The caller has already reconstructed the donor-token trajectory
    through an authenticated C_in route.  One complete uncached child forward
    is observed without retaining logits or changing model state.
    """

    torch = _torch()
    if (
        isinstance(sample_last_tokens, bool)
        or not isinstance(sample_last_tokens, int)
        or sample_last_tokens <= 0
    ):
        raise ValueError("sample_last_tokens must be a positive integer.")
    if (
        isinstance(max_capture_bytes, bool)
        or not isinstance(max_capture_bytes, int)
        or max_capture_bytes <= 0
    ):
        raise ValueError("max_capture_bytes must be a positive integer.")
    embeddings = _cpu_tensor(torch, inputs_embeds, "child trajectory embeddings")
    mask = _cpu_tensor(torch, attention_mask, "child trajectory attention mask")
    positions = (
        None
        if position_ids is None
        else _cpu_tensor(torch, position_ids, "child trajectory position IDs")
    )
    if (
        embeddings.ndim != 3
        or not torch.is_floating_point(embeddings)
        or embeddings.is_complex()
        or not bool(torch.isfinite(embeddings).all().item())
        or mask.ndim != 2
        or tuple(mask.shape) != tuple(embeddings.shape[:2])
        or not _is_integer_dtype(torch, mask)
        or not bool((mask == 1).all().item())
        or sample_last_tokens > int(embeddings.shape[1])
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Embedding replay requires finite rank-3 embeddings, one matching "
            "unpadded integer attention mask, and a bounded retained suffix."
        )
    if positions is not None and (
        positions.ndim != 2
        or tuple(positions.shape) != tuple(mask.shape)
        or not _is_integer_dtype(torch, positions)
        or bool((positions < 0).any().item())
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Embedding replay position IDs must be nonnegative integers matching "
            "the attention mask."
        )

    layers, contracts, topology = _layer_topology(child_model)
    selected = _selected_layer_indices(len(layers), selected_layer_indices)
    device = _resolve_input_device(torch, child_model, input_device)
    input_layer = child_model.get_input_embeddings()
    input_weight = getattr(input_layer, "weight", None)
    if (
        not torch.is_tensor(input_weight)
        or input_weight.device != device
        or int(embeddings.shape[-1]) != int(input_weight.shape[-1])
        or embeddings.dtype != input_weight.dtype
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Embedding replay must use the live child's exact hidden width and "
            "floating dtype on its selected runtime device."
        )
    embeddings_runtime = embeddings.to(device=device)
    mask_runtime = mask.to(device=device, dtype=torch.long)
    positions_runtime = (
        None if positions is None else positions.to(device=device, dtype=torch.long)
    )
    input_identity = {
        "inputs_embeds_sha256": _tensor_sha256(torch, embeddings),
        "attention_mask_sha256": _tensor_sha256(torch, mask),
        "position_ids_sha256": (
            None if positions is None else _tensor_sha256(torch, positions)
        ),
    }
    child_before = fingerprint_lfm2_parent(child_model)
    module_modes = [(module, bool(module.training)) for module in child_model.modules()]
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=sample_last_tokens,
        max_capture_bytes=max_capture_bytes,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        capture_causal_circuit_evidence=False,
    )
    hooks = _Lfm2TraceHooks(
        torch,
        child_model,
        layers,
        contracts,
        config,
        selected,
        capture_replay_lm_head_input=True,
    )
    trace: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    try:
        child_model.eval()
        with torch.inference_mode():
            hooks.begin("trajectory", sample_last_tokens=sample_last_tokens)
            kwargs: dict[str, Any] = {
                "inputs_embeds": embeddings_runtime,
                "attention_mask": mask_runtime,
                "use_cache": False,
                "return_dict": True,
            }
            if positions_runtime is not None:
                kwargs["position_ids"] = positions_runtime
            try:
                output = child_model(**kwargs)
            except Exception as exc:
                raise Lfm2CircuitTraceCaptureError(
                    "The LFM2 child forward failed during embedding replay."
                ) from exc
            trace, diagnostics = hooks.finish()
            if _output_field(output, "past_key_values") is not None:
                raise Lfm2CircuitTraceCaptureError(
                    "The uncached embedding replay unexpectedly returned a cache."
                )
            logits = _output_logits(torch, output)
            if tuple(logits.shape[:2]) != tuple(embeddings_runtime.shape[:2]):
                raise Lfm2CircuitTraceCaptureError(
                    "Embedding replay logits changed the donor-time trajectory length."
                )
    finally:
        hooks.remove()
        for module, was_training in module_modes:
            module.training = was_training

    if trace is None or diagnostics is None:  # pragma: no cover - guarded above
        raise AssertionError("embedding replay produced no trace")
    capture_bytes = sum(
        int(value.numel() * value.element_size()) for value in trace.values()
    )
    if capture_bytes > max_capture_bytes:
        trace.clear()
        raise Lfm2CircuitTraceCaptureError(
            "Embedding replay exceeded max_capture_bytes."
        )
    child_after = fingerprint_lfm2_parent(child_model)
    if child_after != child_before:
        trace.clear()
        raise Lfm2CircuitTraceCaptureError(
            "Child state or configuration changed during embedding replay."
        )
    if (
        input_identity["inputs_embeds_sha256"] != _tensor_sha256(torch, embeddings)
        or input_identity["attention_mask_sha256"] != _tensor_sha256(torch, mask)
        or input_identity["position_ids_sha256"]
        != (None if positions is None else _tensor_sha256(torch, positions))
    ):
        trace.clear()
        raise Lfm2CircuitTraceCaptureError(
            "Embedding replay inputs changed during the child forward."
        )

    inventory = _ordered_tensor_descriptors(torch, trace)
    selected_topology_body = {
        "schema": "neural-forge-lfm2-child-selected-topology-v1",
        "model_layer_count": len(layers),
        "selected_layer_indices": list(selected),
        "layers": [topology["layers"][index] for index in selected],
    }
    selected_topology = {
        **selected_topology_body,
        "selected_topology_sha256": _json_sha256(selected_topology_body),
    }
    body = {
        "schema": CHILD_EMBEDDING_REPLAY_SCHEMA,
        "child": child_before.as_dict(),
        "inputs": {
            **input_identity,
            "batch_size": int(embeddings.shape[0]),
            "trajectory_length": int(embeddings.shape[1]),
            "hidden_size": int(embeddings.shape[2]),
            "runtime_device": str(device),
            "runtime_dtype": str(embeddings.dtype).removeprefix("torch."),
        },
        "selected_layer_indices": list(selected),
        "sample_last_tokens": sample_last_tokens,
        "layer_topology": selected_topology,
        "tensor_count": len(inventory),
        "capture_bytes": capture_bytes,
        "tensors": inventory,
        "diagnostics": diagnostics,
        "operation_audit": {
            "parent_forward_calls": 0,
            "child_forward_calls": 1,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "model_mutations": 0,
            "tokenizer_calls": 0,
            "network_calls": 0,
            "use_cache": False,
            "inputs_embeds_used": True,
            "input_ids_used": False,
        },
    }
    record = _sealed_document(body, "embedding_replay_sha256")
    return Lfm2ChildEmbeddingReplay(
        _tensors=MappingProxyType(dict(trace)),
        _record=MappingProxyType(record),
    )


def build_lfm2_circuit_trace_cache(
    cache_root: str | os.PathLike[str],
    model: Any,
    calibration_batches: Iterable[Mapping[str, Any]],
    *,
    config: Lfm2CircuitTraceCaptureConfig | None = None,
    input_device: Any | None = None,
) -> Lfm2CircuitTraceCache:
    """Build or reuse an immutable content-addressed LFM2 parent trace cache."""

    torch = _torch()
    _safetensors()
    selected = Lfm2CircuitTraceCaptureConfig() if config is None else config
    if not isinstance(selected, Lfm2CircuitTraceCaptureConfig):
        raise TypeError("config must be Lfm2CircuitTraceCaptureConfig or None.")
    prepared = _prepare_batches(torch, calibration_batches, selected)
    layers, contracts, topology = _layer_topology(model)
    parent_before = fingerprint_lfm2_parent(model)
    implementation = implementation_identity()
    corpus = _corpus_identity(prepared)
    device = _resolve_input_device(torch, model, input_device)
    execution = _execution_identity(model, device)
    if (
        selected.capture_causal_circuit_evidence
        and execution.get("attention_backend") != "eager"
    ):
        raise Lfm2CircuitTraceCaptureError(
            "Causal-circuit installability evidence currently requires the "
            "eager attention backend so the reconstructed score/probability "
            "operator is identical to the executed parent operator."
        )
    manifest = _manifest_for(
        parent_before,
        corpus,
        implementation,
        topology,
        selected,
        device,
        execution,
    )

    root = Path(cache_root).expanduser().absolute()
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise Lfm2CircuitTraceCacheError(
            f"cache_root must be a regular local directory: {root}."
        )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(manifest["cache_key"])
    if destination.exists():
        cache = open_lfm2_circuit_trace_cache(destination, verify=True)
        if cache.manifest != manifest:
            raise Lfm2CircuitTraceConflictError(
                "Existing content-addressed cache manifest conflicts with the request."
            )
        return cache
    compatible_cache = _find_compatible_committed_cache(root, manifest)
    if compatible_cache is not None:
        return compatible_cache

    stage = root / f".lfm2-circuit-stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        _write_canonical_json(stage / MANIFEST_FILENAME, manifest)
        batches_path = stage / "batches"
        batches_path.mkdir()
        entries: list[dict[str, Any]] = []
        for batch in prepared:
            tensors, capture = _capture_batch(
                torch,
                model,
                layers,
                contracts,
                batch,
                selected,
                device,
            )
            entries.append(
                _publish_batch(
                    torch,
                    batches_path / f"batch-{batch.index:06d}",
                    manifest,
                    batch,
                    tensors,
                    capture,
                )
            )
        parent_after = fingerprint_lfm2_parent(model)
        if parent_after != parent_before:
            raise Lfm2CircuitTraceCaptureError(
                "Parent state or configuration changed during cache capture."
            )
        completion = _completion_for(manifest, entries)
        _write_canonical_json(stage / COMPLETION_FILENAME, completion)

        staged = Lfm2CircuitTraceCache.open(stage, verify=True)
        if destination.exists():
            existing = open_lfm2_circuit_trace_cache(destination, verify=True)
            if existing.manifest != staged.manifest:
                raise Lfm2CircuitTraceConflictError(
                    "A concurrent cache publication has a conflicting manifest."
                )
            return existing
        try:
            os.rename(stage, destination)
        except FileExistsError:
            existing = open_lfm2_circuit_trace_cache(destination, verify=True)
            if existing.manifest != manifest:
                raise Lfm2CircuitTraceConflictError(
                    "A concurrent cache publication conflicts with the request."
                )
            return existing
        return open_lfm2_circuit_trace_cache(destination, verify=True)
    finally:
        if stage.exists():
            _safe_remove_stage(stage, root)


capture_lfm2_circuit_trace_cache = build_lfm2_circuit_trace_cache


__all__ = [
    "AUTHENTICATED_SESSION_SCHEMA",
    "BATCH_RECORD_FILENAME",
    "BATCH_SCHEMA",
    "CACHE_SCHEMA",
    "CACHED_VERIFIED_TRACE_MODE",
    "CHILD_REPLAY_PREFIX_BY_PREFIX_MODE",
    "CHILD_REPLAY_LOGIT_ERROR_MODEL",
    "CHILD_REPLAY_NON_LOGIT_ERROR_MODEL",
    "CHILD_REPLAY_NON_LOGIT_ENVELOPE_SCHEMA",
    "CHILD_REPLAY_NUMERIC_POLICY_SCHEMA",
    "CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE",
    "CHILD_REPLAY_BATCH_SCHEMA",
    "CHILD_EVIDENCE_PROFILE_SCHEMA",
    "CHILD_EMBEDDING_REPLAY_SCHEMA",
    "CHILD_REPLAY_SCHEMA",
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "FULL_PREFIX_CANONICAL_TRACE_MODE",
    "MANIFEST_FILENAME",
    "TRACE_FILENAME",
    "Lfm2ChildReplay",
    "Lfm2ChildReplayBatch",
    "Lfm2ChildEvidenceProfile",
    "Lfm2ChildEmbeddingReplay",
    "Lfm2CircuitTraceAuthenticatedSession",
    "Lfm2CircuitTraceBatch",
    "Lfm2CircuitTraceCache",
    "Lfm2CircuitTraceCacheError",
    "Lfm2CircuitTraceCaptureConfig",
    "Lfm2CircuitTraceCaptureError",
    "Lfm2CircuitTraceConflictError",
    "Lfm2CircuitTraceCorruptionError",
    "Lfm2CircuitTraceSelectedBatchPayload",
    "Lfm2CircuitTraceSelectiveAuthenticatedSession",
    "Lfm2CircuitTraceSelectiveBatchReference",
    "Lfm2ParentFingerprint",
    "SELECTIVE_AUTHENTICATED_SESSION_SCHEMA",
    "SELECTIVE_BATCH_PAYLOAD_SCHEMA",
    "build_lfm2_circuit_trace_cache",
    "compile_lfm2_child_replay_numeric_policy",
    "capture_lfm2_child_embedding_replay",
    "capture_lfm2_child_replay",
    "capture_lfm2_circuit_trace_cache",
    "fingerprint_lfm2_parent",
    "implementation_identity",
    "open_lfm2_circuit_trace_cache",
]
