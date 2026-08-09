"""Authenticated row-sliced reads for sealed LFM2 circuit-trace caches.

This reader is intentionally separate from :mod:`hf_lfm2_circuit_trace_cache`.
The collector module's complete source hash is part of every sealed cache
identity, so adding a derived reader there would unnecessarily invalidate the
existing 256-batch parent cache.  This module consumes the already
authenticated selective session without changing collector, manifest, record,
or session evidence identities.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import hf_lfm2_circuit_trace_cache as trace_cache


SELECTIVE_ROW_BATCH_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-circuit-trace-selective-row-batch-payload-v1"
)
_READER_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()
).hexdigest()


def _canonical_flat_row_indices(
    value: Any,
    *,
    row_count: int,
    name: str,
) -> tuple[int, ...]:
    """Validate one deterministic, nonempty flattened-row selection."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise trace_cache.Lfm2CircuitTraceCacheError(
            f"Selected row indices for {name!r} must be a sequence."
        )
    indices = tuple(value)
    if (
        not indices
        or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        )
        or indices != tuple(sorted(set(indices)))
        or indices[0] < 0
        or indices[-1] >= row_count
    ):
        raise trace_cache.Lfm2CircuitTraceCacheError(
            f"Selected row indices for {name!r} must be nonempty, unique, "
            f"canonical, and within [0, {row_count})."
        )
    return indices


def _unravel_flat_index(index: int, shape: Sequence[int]) -> tuple[int, ...]:
    coordinates = [0] * len(shape)
    remainder = index
    for axis in range(len(shape) - 1, -1, -1):
        coordinates[axis] = remainder % int(shape[axis])
        remainder //= int(shape[axis])
    if remainder != 0:  # pragma: no cover - caller bounds prove this
        raise AssertionError("flat row index exceeds its leading shape")
    return tuple(coordinates)


def _selected_row_runs(
    indices: Sequence[int],
    leading_shape: Sequence[int],
) -> tuple[tuple[tuple[int, ...], int, int], ...]:
    """Group canonical flat rows into slices along the last leading axis."""

    final_axis = int(leading_shape[-1])
    runs: list[tuple[tuple[int, ...], int, int]] = []
    start = int(indices[0])
    previous = start
    for raw in (*indices[1:], None):
        current = None if raw is None else int(raw)
        same_prefix = (
            current is not None
            and current == previous + 1
            and current // final_axis == start // final_axis
        )
        if same_prefix:
            previous = current
            continue
        prefix = _unravel_flat_index(
            start // final_axis,
            leading_shape[:-1],
        )
        runs.append(
            (
                prefix,
                start % final_axis,
                previous % final_axis + 1,
            )
        )
        if current is not None:
            start = previous = current
    return tuple(runs)


def _copy_selected_safetensor_rows(
    torch: Any,
    tensor_slice: Any,
    *,
    shape: Sequence[int],
    indices: Sequence[int],
    expected_dtype: str,
    name: str,
) -> tuple[Any, int, int]:
    """Copy only selected flattened rows from an authenticated safe slice."""

    normalized_shape = tuple(int(size) for size in shape)
    if len(normalized_shape) < 2 or any(size <= 0 for size in normalized_shape):
        raise trace_cache.Lfm2CircuitTraceCorruptionError(
            f"Authenticated tensor {name!r} has an invalid row shape."
        )
    observed_shape = tuple(int(size) for size in tensor_slice.get_shape())
    if observed_shape != normalized_shape:
        raise trace_cache.Lfm2CircuitTraceCorruptionError(
            f"Authenticated tensor {name!r} changed shape before row selection."
        )
    width = normalized_shape[-1]
    leading_shape = normalized_shape[:-1]
    runs = _selected_row_runs(indices, leading_shape)
    result = None
    cursor = 0
    maximum_materialized_source_rows = 0
    for prefix, start, end in runs:
        selector = (*prefix, slice(start, end), slice(None))
        piece = tensor_slice[selector]
        if (
            not torch.is_tensor(piece)
            or piece.layout != torch.strided
            or piece.device.type != "cpu"
            or tuple(piece.shape) != (end - start, width)
            or str(piece.dtype).replace("torch.", "") != expected_dtype
        ):
            raise trace_cache.Lfm2CircuitTraceCorruptionError(
                f"Authenticated tensor {name!r} returned an invalid row slice."
            )
        if result is None:
            result = torch.empty(
                (len(indices), width),
                dtype=piece.dtype,
                device="cpu",
            )
        count = end - start
        result[cursor : cursor + count].copy_(piece)
        cursor += count
        maximum_materialized_source_rows = max(
            maximum_materialized_source_rows,
            count,
        )
    if result is None or cursor != len(indices):  # pragma: no cover
        raise AssertionError("authenticated row selection did not materialize its plan")
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise trace_cache.Lfm2CircuitTraceCorruptionError(
            f"Authenticated tensor {name!r} selected NaN or infinity."
        )
    return result.contiguous(), maximum_materialized_source_rows, len(runs)


def _verify_authenticated_record_state(state: Any, *, index: int) -> None:
    """Recheck mutable nested record metadata against its original seal."""

    trace_cache._verify_seal(
        state.record,
        "record_sha256",
        f"cache batch {index} record",
    )
    record_sha256 = state.record.get("record_sha256")
    if (
        record_sha256 != state.reference.record_sha256
        or record_sha256 != state.entry.get("record_sha256")
    ):
        raise trace_cache.Lfm2CircuitTraceCorruptionError(
            f"Cache batch {index} record seal changed during row selection."
        )


class Lfm2CircuitTraceSelectedRowsPayload:
    """Owned source-dtype rows copied from authenticated tensor slices."""

    __slots__ = (
        "batch_index",
        "coalesced_slice_count",
        "input_sha256",
        "logical_bytes",
        "maximum_materialized_source_rows",
        "ordered_tensor_names",
        "row_indices",
        "safetensors_get_slice_calls",
        "selected_row_count",
        "selection_sha256",
        "session_sha256",
        "source_logical_bytes",
        "source_row_count",
        "source_tensor_count",
        "unselected_rows_loaded",
        "whole_tensor_cpu_clone_calls",
        "_released",
        "_tensors",
    )

    def __init__(
        self,
        *,
        batch_index: int,
        input_sha256: str,
        ordered_tensor_names: tuple[str, ...],
        row_indices: Mapping[str, tuple[int, ...]],
        logical_bytes: int,
        source_logical_bytes: int,
        selected_row_count: int,
        source_row_count: int,
        source_tensor_count: int,
        safetensors_get_slice_calls: int,
        coalesced_slice_count: int,
        maximum_materialized_source_rows: int,
        session_sha256: str,
        selection_sha256: str,
        tensors: Mapping[str, Any],
    ) -> None:
        self.batch_index = batch_index
        self.input_sha256 = input_sha256
        self.ordered_tensor_names = ordered_tensor_names
        self.row_indices = MappingProxyType(dict(row_indices))
        self.logical_bytes = logical_bytes
        self.source_logical_bytes = source_logical_bytes
        self.selected_row_count = selected_row_count
        self.source_row_count = source_row_count
        self.source_tensor_count = source_tensor_count
        self.safetensors_get_slice_calls = safetensors_get_slice_calls
        self.coalesced_slice_count = coalesced_slice_count
        self.maximum_materialized_source_rows = maximum_materialized_source_rows
        self.whole_tensor_cpu_clone_calls = 0
        self.unselected_rows_loaded = False
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
            raise trace_cache.Lfm2CircuitTraceCacheError(
                "Selected cache row payload was released."
            )
        return MappingProxyType(self._tensors)

    def _release(self) -> None:
        if self._released:
            return
        if self._tensors is not None:
            self._tensors.clear()
        self._tensors = None
        self._released = True


def lfm2_circuit_trace_selected_rows_payload_sha256(
    *,
    session_sha256: str,
    cache_key: str,
    batch_index: int,
    reference: Any,
    payload: Lfm2CircuitTraceSelectedRowsPayload,
) -> str:
    """Recompute a selected-row payload seal in one session's coordinates."""

    if not isinstance(payload, Lfm2CircuitTraceSelectedRowsPayload):
        raise TypeError("payload must be Lfm2CircuitTraceSelectedRowsPayload.")
    record = getattr(reference, "_record", None)
    inventory = None if not isinstance(record, Mapping) else record.get("tensors")
    names = payload.ordered_tensor_names
    if (
        not isinstance(inventory, Mapping)
        or not names
        or names != tuple(sorted(set(names)))
        or set(payload.row_indices) != set(names)
        or any(name not in inventory for name in names)
    ):
        raise trace_cache.Lfm2CircuitTraceCacheError(
            "Selected row payload cannot be sealed from its authenticated inventory."
        )
    tensors = payload.tensors
    if set(tensors) != set(names):
        raise trace_cache.Lfm2CircuitTraceCacheError(
            "Selected row payload tensor inventory changed."
        )
    torch = trace_cache._torch()
    source_logical_bytes = 0
    source_row_count = 0
    selected_row_count = 0
    selected_logical_bytes = 0
    ordered_row_selections = []
    for name in names:
        descriptor = inventory[name]
        shape = descriptor.get("shape") if isinstance(descriptor, Mapping) else None
        if (
            not isinstance(shape, list)
            or len(shape) < 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in shape
            )
        ):
            raise trace_cache.Lfm2CircuitTraceCacheError(
                "Selected row payload source descriptor changed."
            )
        row_count = math.prod(shape[:-1])
        indices = _canonical_flat_row_indices(
            payload.row_indices[name],
            row_count=row_count,
            name=name,
        )
        tensor = tensors[name]
        selected_descriptor = trace_cache._tensor_descriptor(torch, tensor)
        source_logical_bytes += int(descriptor["size_bytes"])
        source_row_count += row_count
        selected_row_count += len(indices)
        selected_logical_bytes += int(tensor.numel() * tensor.element_size())
        ordered_row_selections.append(
            {
                "tensor_name": name,
                "flat_row_indices": list(indices),
                "source_descriptor": dict(descriptor),
                "selected_descriptor": selected_descriptor,
            }
        )
    if (
        payload.input_sha256 != reference.input_sha256
        or payload.logical_bytes != selected_logical_bytes
        or payload.source_logical_bytes != source_logical_bytes
        or payload.selected_row_count != selected_row_count
        or payload.source_row_count != source_row_count
        or payload.source_tensor_count != len(names)
        or payload.safetensors_get_slice_calls != len(names)
        or payload.whole_tensor_cpu_clone_calls != 0
        or payload.unselected_rows_loaded is not False
    ):
        raise trace_cache.Lfm2CircuitTraceCacheError(
            "Selected row payload accounting changed."
        )
    body = {
        "schema": SELECTIVE_ROW_BATCH_PAYLOAD_SCHEMA,
        "session_sha256": session_sha256,
        "cache_key": cache_key,
        "batch_index": batch_index,
        "input_sha256": reference.input_sha256,
        "record_sha256": reference.record_sha256,
        "ordered_row_selections": ordered_row_selections,
        "source_logical_bytes": source_logical_bytes,
        "selected_logical_bytes": selected_logical_bytes,
        "source_tensor_count": len(names),
        "source_row_count": source_row_count,
        "selected_row_count": selected_row_count,
        "safetensors_get_slice_calls": len(names),
        "coalesced_slice_count": payload.coalesced_slice_count,
        "maximum_materialized_source_rows": (payload.maximum_materialized_source_rows),
        "row_selection_precedes_whole_tensor_cpu_clone": True,
        "whole_tensor_cpu_clone_calls": 0,
        "unselected_rows_loaded": False,
        "collector_source_or_implementation_identity_changed": False,
        "row_selection_reader_source_sha256": _READER_SOURCE_SHA256,
    }
    return trace_cache._json_sha256(body)


def load_lfm2_circuit_trace_selected_rows(
    session: trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
    index: int,
    row_indices: Mapping[str, Sequence[int]],
) -> Lfm2CircuitTraceSelectedRowsPayload:
    """Read authenticated canonical rows without changing collector identity."""

    if not isinstance(
        session,
        trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise TypeError(
            "session must be an Lfm2CircuitTraceSelectiveAuthenticatedSession."
        )
    if not isinstance(row_indices, Mapping) or not row_indices:
        raise TypeError("row_indices must be a nonempty tensor-name mapping.")
    names = tuple(sorted(row_indices))
    if any(not isinstance(name, str) or not name for name in names):
        raise trace_cache.Lfm2CircuitTraceCacheError(
            "Selected row tensor names must be nonempty strings."
        )
    with session._lock:
        state = session._state(index)
        _verify_authenticated_record_state(state, index=index)
        inventory = state.record["tensors"]
        if any(name not in inventory for name in names):
            missing = next(name for name in names if name not in inventory)
            raise trace_cache.Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} has no sealed tensor {missing!r}."
            )
        normalized_indices: dict[str, tuple[int, ...]] = {}
        source_logical_bytes = 0
        source_row_count = 0
        selected_row_count = 0
        expected_selected_bytes = 0
        for name in names:
            descriptor = inventory[name]
            shape = descriptor.get("shape") if isinstance(descriptor, Mapping) else None
            numel = descriptor.get("numel") if isinstance(descriptor, Mapping) else None
            size_bytes = (
                descriptor.get("size_bytes")
                if isinstance(descriptor, Mapping)
                else None
            )
            dtype = descriptor.get("dtype") if isinstance(descriptor, Mapping) else None
            if (
                not isinstance(shape, list)
                or len(shape) < 2
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                )
                or isinstance(numel, bool)
                or not isinstance(numel, int)
                or numel != math.prod(shape)
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes <= 0
                or size_bytes % numel != 0
                or not isinstance(dtype, str)
                or not dtype
            ):
                raise trace_cache.Lfm2CircuitTraceCorruptionError(
                    f"Cache batch {index} tensor {name!r} has an invalid "
                    "sealed row descriptor."
                )
            row_count = math.prod(shape[:-1])
            selected = _canonical_flat_row_indices(
                row_indices[name],
                row_count=row_count,
                name=name,
            )
            normalized_indices[name] = selected
            element_size = size_bytes // numel
            source_logical_bytes += size_bytes
            source_row_count += row_count
            selected_row_count += len(selected)
            expected_selected_bytes += len(selected) * shape[-1] * element_size

        session._verify_batch_identity(state)
        torch = trace_cache._torch()
        safe_open, _save_file = trace_cache._safetensors()
        tensors: dict[str, Any] = {}
        maximum_materialized_source_rows = 0
        coalesced_slice_count = 0
        try:
            with safe_open(
                str(state.trace_path),
                framework="pt",
                device="cpu",
            ) as handle:
                metadata = handle.metadata() or {}
                if metadata != {
                    "schema": trace_cache.BATCH_SCHEMA,
                    "cache_key": session.cache_key,
                    "batch_index": str(index),
                    "input_sha256": state.reference.input_sha256,
                }:
                    raise trace_cache.Lfm2CircuitTraceCorruptionError(
                        f"Cache batch {index} safetensors metadata changed."
                    )
                if tuple(sorted(handle.keys())) != state.reference.ordered_tensor_names:
                    raise trace_cache.Lfm2CircuitTraceCorruptionError(
                        f"Cache batch {index} safetensors keys changed."
                    )
                get_slice = getattr(handle, "get_slice", None)
                if not callable(get_slice):
                    raise trace_cache.Lfm2CircuitTraceCacheError(
                        "Installed safetensors runtime does not expose "
                        "authenticated row slicing; whole-tensor fallback "
                        "is forbidden."
                    )
                session._verify_batch_identity(state)
                for name in names:
                    descriptor = inventory[name]
                    (
                        tensor,
                        materialized_rows,
                        tensor_slice_count,
                    ) = _copy_selected_safetensor_rows(
                        torch,
                        get_slice(name),
                        shape=descriptor["shape"],
                        indices=normalized_indices[name],
                        expected_dtype=str(descriptor["dtype"]),
                        name=f"cache batch {index} selected rows {name}",
                    )
                    tensors[name] = tensor
                    maximum_materialized_source_rows = max(
                        maximum_materialized_source_rows,
                        materialized_rows,
                    )
                    coalesced_slice_count += tensor_slice_count
            session._verify_batch_identity(state)
            _verify_authenticated_record_state(state, index=index)
        except BaseException:
            tensors.clear()
            raise
        logical_bytes = sum(
            int(value.numel() * value.element_size()) for value in tensors.values()
        )
        if logical_bytes != expected_selected_bytes:
            tensors.clear()
            raise trace_cache.Lfm2CircuitTraceCorruptionError(
                f"Cache batch {index} selected row byte count changed."
            )
        body = {
            "schema": SELECTIVE_ROW_BATCH_PAYLOAD_SCHEMA,
            "session_sha256": session.evidence_sha256,
            "cache_key": session.cache_key,
            "batch_index": index,
            "input_sha256": state.reference.input_sha256,
            "record_sha256": state.reference.record_sha256,
            "ordered_row_selections": [
                {
                    "tensor_name": name,
                    "flat_row_indices": list(normalized_indices[name]),
                    "source_descriptor": dict(inventory[name]),
                    "selected_descriptor": trace_cache._tensor_descriptor(
                        torch,
                        tensors[name],
                    ),
                }
                for name in names
            ],
            "source_logical_bytes": source_logical_bytes,
            "selected_logical_bytes": logical_bytes,
            "source_tensor_count": len(names),
            "source_row_count": source_row_count,
            "selected_row_count": selected_row_count,
            "safetensors_get_slice_calls": len(names),
            "coalesced_slice_count": coalesced_slice_count,
            "maximum_materialized_source_rows": maximum_materialized_source_rows,
            "row_selection_precedes_whole_tensor_cpu_clone": True,
            "whole_tensor_cpu_clone_calls": 0,
            "unselected_rows_loaded": False,
            "collector_source_or_implementation_identity_changed": False,
            "row_selection_reader_source_sha256": _READER_SOURCE_SHA256,
        }
        return Lfm2CircuitTraceSelectedRowsPayload(
            batch_index=index,
            input_sha256=state.reference.input_sha256,
            ordered_tensor_names=names,
            row_indices=normalized_indices,
            logical_bytes=logical_bytes,
            source_logical_bytes=source_logical_bytes,
            selected_row_count=selected_row_count,
            source_row_count=source_row_count,
            source_tensor_count=len(names),
            safetensors_get_slice_calls=len(names),
            coalesced_slice_count=coalesced_slice_count,
            maximum_materialized_source_rows=maximum_materialized_source_rows,
            session_sha256=session.evidence_sha256,
            selection_sha256=trace_cache._json_sha256(body),
            tensors=tensors,
        )


def release_lfm2_circuit_trace_selected_rows(
    session: trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
    payload: Lfm2CircuitTraceSelectedRowsPayload,
) -> None:
    """Deterministically release one authenticated row payload."""

    if not isinstance(
        session,
        trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise TypeError(
            "session must be an Lfm2CircuitTraceSelectiveAuthenticatedSession."
        )
    if not isinstance(payload, Lfm2CircuitTraceSelectedRowsPayload):
        raise TypeError("payload must be Lfm2CircuitTraceSelectedRowsPayload.")
    if payload.session_sha256 != session.evidence_sha256:
        raise trace_cache.Lfm2CircuitTraceCacheError(
            "Selected row payload belongs to another authenticated session."
        )
    payload._release()


__all__ = [
    "Lfm2CircuitTraceSelectedRowsPayload",
    "SELECTIVE_ROW_BATCH_PAYLOAD_SCHEMA",
    "lfm2_circuit_trace_selected_rows_payload_sha256",
    "load_lfm2_circuit_trace_selected_rows",
    "release_lfm2_circuit_trace_selected_rows",
]
