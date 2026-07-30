"""Authenticated zero-copy access to already materialized LFM2 trace rows.

The production runner materializes a compact row index once for the fold.  The
global topology cold path may reuse those exact tensors instead of reopening
the same safetensors payloads.  This adapter binds the row index back to the
authenticated per-batch record descriptors and reconstructs the original
rank-three/rank-four tensor views without changing values or dtype.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TRACE_ROWS_SCHEMA = "neural-forge-lfm2-circuit-rows-v1"
_HASH_CHUNK_BYTES = 8 * 1024**2
_TENSOR_NAME = re.compile(
    r"^(prefill|decode\.\d{3})\.layer\.(\d{3})\.(.+)$"
)
_COMMON_FIELDS = frozenset(
    {
        "mlp.gate",
        "mlp.post_swiglu",
        "mlp.up",
        "residual.post_layer",
        "residual.post_operator",
        "residual.pre",
    }
)
_ATTENTION_FIELDS = frozenset(
    {
        "attention.context",
        "attention.k",
        "attention.q",
        "attention.v",
    }
)
_OPTIONAL_ATTENTION_FIELDS = frozenset(
    {
        "attention.k_linear",
        "attention.k_normalized_pre_rope",
        "attention.q_linear",
        "attention.q_normalized_pre_rope",
    }
)


class Lfm2RetainedTraceRowsError(ValueError):
    """Materialized rows do not exactly match their authenticated cache."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Retained LFM2 trace rows require PyTorch.") from exc
    return torch


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
        raise Lfm2RetainedTraceRowsError(
            "Retained-row identity is not canonical JSON."
        ) from exc


def _update_tensor_bytes(digest: Any, tensor: Any) -> None:
    """Hash logical contiguous tensor bytes without one unbounded byte copy."""

    torch = _torch()
    value = tensor.detach()
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise Lfm2RetainedTraceRowsError(
            "Retained-row tensor hashing requires CPU strided storage."
        )

    def update_contiguous(piece: Any) -> None:
        raw = piece.contiguous().view(torch.uint8).reshape(-1)
        for start in range(0, int(raw.numel()), _HASH_CHUNK_BYTES):
            block = raw[start : start + _HASH_CHUNK_BYTES]
            digest.update(block.numpy().tobytes())

    if value.is_contiguous():
        update_contiguous(value)
        return
    if value.ndim == 4:
        for batch_index in range(int(value.shape[0])):
            for head_index in range(int(value.shape[1])):
                update_contiguous(value[batch_index, head_index])
        return
    update_contiguous(value)


def _cache_tensor_sha256(tensor: Any) -> str:
    """Reproduce the immutable circuit-cache tensor descriptor SHA exactly."""

    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-circuit-tensor-v1\0")
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    _update_tensor_bytes(digest, tensor)
    return digest.hexdigest()


def _trace_row_tensor_sha256(tensor: Any) -> str:
    """Reproduce the TraceRowIndex field SHA exactly."""

    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "dtype": str(tensor.dtype),
                "shape": [int(size) for size in tensor.shape],
            }
        )
    )
    _update_tensor_bytes(digest, tensor)
    return digest.hexdigest()


def _restore_source_tensor(tensor: Any, field: str) -> Any:
    if field.startswith("attention."):
        return tensor.permute(1, 0, 2).unsqueeze(0)
    return tensor.unsqueeze(0)


def _reference_identity(reference: Any) -> tuple[Any, ...]:
    return (
        getattr(reference, "index", None),
        getattr(reference, "input_sha256", None),
        getattr(reference, "record_sha256", None),
        getattr(reference, "ordered_tensor_names", None),
    )


def _descriptor(
    record: Mapping[str, Any],
    batch_index: int,
    name: str,
) -> Mapping[str, Any]:
    inventory = record.get("tensors")
    value = inventory.get(name) if isinstance(inventory, Mapping) else None
    if not isinstance(value, Mapping):
        raise Lfm2RetainedTraceRowsError(
            f"Authenticated batch {batch_index} lacks descriptor {name!r}."
        )
    shape = value.get("shape")
    dtype = value.get("dtype")
    if (
        not isinstance(shape, list)
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in shape
        )
        or not isinstance(dtype, str)
        or not dtype
        or value.get("numel") != math.prod(shape)
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] <= 0
        or not isinstance(value.get("sha256"), str)
        or _SHA256.fullmatch(value["sha256"]) is None
    ):
        raise Lfm2RetainedTraceRowsError(
            f"Authenticated descriptor {name!r} is malformed."
        )
    return value


@dataclass(frozen=True, slots=True)
class Lfm2RetainedTraceRowAdapter:
    """Strict batch/phase views over one materialized TraceRowIndex."""

    rows: Any
    batch_references: tuple[Any, ...]
    phases_by_batch: tuple[tuple[str, ...], ...]
    expected_layer_kinds: tuple[str, ...]
    _phase_slices: Mapping[tuple[int, str], slice]
    _available_names_by_batch: Mapping[int, frozenset[str]]
    _records_by_batch: Mapping[int, Mapping[str, Any]]
    _tensor_versions: Mapping[tuple[int, str], int]

    @classmethod
    def build(
        cls,
        rows: Any,
        *,
        cache_key: str,
        corpus_fingerprint: str,
        batch_references: Sequence[Any],
        phases_by_batch: Sequence[Sequence[str]],
        expected_layer_kinds: Sequence[str],
    ) -> Lfm2RetainedTraceRowAdapter:
        references = tuple(batch_references)
        phase_inventory = tuple(tuple(value) for value in phases_by_batch)
        layer_kinds = tuple(expected_layer_kinds)
        if isinstance(rows, cls):
            rows._verify_reuse_context(
                cache_key=cache_key,
                corpus_fingerprint=corpus_fingerprint,
                batch_references=references,
                phases_by_batch=phase_inventory,
                expected_layer_kinds=layer_kinds,
            )
            return rows
        records = tuple(reference.record for reference in references)
        if (
            not isinstance(cache_key, str)
            or not _SHA256.fullmatch(cache_key)
            or not isinstance(corpus_fingerprint, str)
            or not _SHA256.fullmatch(corpus_fingerprint)
        ):
            raise Lfm2RetainedTraceRowsError(
                "Retained-row cache identities must be SHA-256 digests."
            )
        if (
            getattr(rows, "cache_key", None) != cache_key
            or getattr(rows, "corpus_fingerprint", None) != corpus_fingerprint
            or getattr(rows, "layer_kinds", None) != layer_kinds
        ):
            raise Lfm2RetainedTraceRowsError(
                "Retained rows are bound to a different cache, corpus, or topology."
            )
        trace_rows_sha256 = getattr(rows, "trace_rows_sha256", None)
        if not isinstance(trace_rows_sha256, str) or not _SHA256.fullmatch(
            trace_rows_sha256
        ):
            raise Lfm2RetainedTraceRowsError(
                "Retained rows lack a canonical trace-row identity."
            )
        if len(references) != len(phase_inventory) or any(
            getattr(reference, "index", None) != index
            for index, reference in enumerate(references)
        ):
            raise Lfm2RetainedTraceRowsError(
                "Retained-row batch references are not canonical."
            )
        paths = tuple(getattr(rows, "paths", ()))
        path_splits = tuple(getattr(rows, "path_splits", ()))
        if not paths or len(path_splits) != len(paths):
            raise Lfm2RetainedTraceRowsError(
                "Retained rows have missing or unaligned causal paths."
            )

        expected_paths: list[str] = []
        phase_slices: dict[tuple[int, str], slice] = {}
        cursor = 0
        for reference, record, phases in zip(
            references,
            records,
            phase_inventory,
            strict=True,
        ):
            if not phases or len(set(phases)) != len(phases):
                raise Lfm2RetainedTraceRowsError(
                    f"Batch {reference.index} has a malformed phase inventory."
                )
            for phase in phases:
                anchor = (
                    f"{phase}.layer.000.residual.pre"
                )
                descriptor = _descriptor(record, reference.index, anchor)
                shape = descriptor["shape"]
                if len(shape) != 3 or shape[0] != 1:
                    raise Lfm2RetainedTraceRowsError(
                        f"Retained-row anchor {anchor!r} changed shape."
                    )
                row_count = int(shape[1])
                start = cursor
                cursor += row_count
                phase_slices[(reference.index, phase)] = slice(start, cursor)
                expected_paths.extend(
                    f"batch.{reference.index:06d}/{phase}/row.{row:04d}"
                    for row in range(row_count)
                )
                splits = path_splits[start:cursor]
                if not splits or len(set(splits)) != 1 or not isinstance(
                    splits[0], str
                ):
                    raise Lfm2RetainedTraceRowsError(
                        f"Retained-row split lineage changed for batch "
                        f"{reference.index}, phase {phase}."
                    )
        if tuple(expected_paths) != paths:
            raise Lfm2RetainedTraceRowsError(
                "Retained causal paths are missing, foreign, or out of order."
            )

        layer_fields = getattr(rows, "layer_fields", None)
        if not isinstance(layer_fields, Mapping) or tuple(layer_fields) != tuple(
            range(len(layer_kinds))
        ):
            raise Lfm2RetainedTraceRowsError(
                "Retained rows do not cover each source layer exactly once."
            )
        torch = _torch()
        available_by_batch: dict[int, frozenset[str]] = {}
        trace_descriptors: list[dict[str, Any]] = []
        tensor_versions: dict[tuple[int, str], int] = {}
        for layer_index, layer_kind in enumerate(layer_kinds):
            fields = layer_fields[layer_index]
            if not isinstance(fields, Mapping):
                raise Lfm2RetainedTraceRowsError(
                    f"Retained layer {layer_index} fields are not a mapping."
                )
            required = set(_COMMON_FIELDS)
            allowed = set(_COMMON_FIELDS)
            if layer_kind == "full_attention":
                required.update(_ATTENTION_FIELDS)
                allowed.update(_ATTENTION_FIELDS)
                allowed.update(_OPTIONAL_ATTENTION_FIELDS)
            elif layer_kind != "short_conv":
                raise Lfm2RetainedTraceRowsError(
                    f"Unsupported retained LFM2 layer kind {layer_kind!r}."
                )
            actual = set(fields)
            if not required.issubset(actual) or not actual.issubset(allowed):
                raise Lfm2RetainedTraceRowsError(
                    f"Retained layer {layer_index} has missing or foreign endpoints."
            )
            for field in sorted(actual):
                tensor = fields[field]
                expected_ndim = 3 if field.startswith("attention.") else 2
                if (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_floating_point()
                    or tensor.is_complex()
                    or tensor.device.type != "cpu"
                    or tensor.ndim != expected_ndim
                    or int(tensor.shape[0]) != len(paths)
                    or not tensor.is_contiguous()
                ):
                    raise Lfm2RetainedTraceRowsError(
                        f"Retained layer {layer_index} endpoint {field!r} changed "
                        "shape, dtype, device, or storage order."
                    )
                version_before = int(tensor._version)
                trace_descriptors.append(
                    {
                        "layer": layer_index,
                        "field": field,
                        "shape": [int(size) for size in tensor.shape],
                        "dtype": str(tensor.dtype),
                        "sha256": _trace_row_tensor_sha256(tensor),
                    }
                )
                for reference, record, phases in zip(
                    references,
                    records,
                    phase_inventory,
                    strict=True,
                ):
                    for phase in phases:
                        name = f"{phase}.layer.{layer_index:03d}.{field}"
                        descriptor = _descriptor(record, reference.index, name)
                        shape = descriptor["shape"]
                        descriptor_ndim = 4 if field.startswith("attention.") else 3
                        expected_tail = (
                            (int(shape[1]), int(shape[3]))
                            if descriptor_ndim == 4 and len(shape) == 4
                            else (
                                (int(shape[2]),)
                                if descriptor_ndim == 3 and len(shape) == 3
                                else None
                            )
                        )
                        if (
                            expected_tail is None
                            or tuple(tensor.shape[1:]) != expected_tail
                            or str(tensor.dtype).removeprefix("torch.")
                            != descriptor["dtype"]
                            or descriptor["size_bytes"]
                            != math.prod(shape) * int(tensor.element_size())
                        ):
                            raise Lfm2RetainedTraceRowsError(
                                f"Retained endpoint {name!r} differs from its "
                                "authenticated descriptor."
                            )
                        retained_phase = tensor[
                            phase_slices[(reference.index, phase)]
                        ]
                        source_rows = int(shape[2] if field.startswith("attention.") else shape[1])
                        if int(retained_phase.shape[0]) == source_rows:
                            source_tensor = _restore_source_tensor(
                                retained_phase,
                                field,
                            )
                            if (
                                _cache_tensor_sha256(source_tensor)
                                != descriptor["sha256"]
                            ):
                                raise Lfm2RetainedTraceRowsError(
                                    f"Retained endpoint {name!r} differs from its "
                                    "authenticated tensor SHA."
                                )
                        elif field not in _OPTIONAL_ATTENTION_FIELDS:
                            raise Lfm2RetainedTraceRowsError(
                                f"Retained endpoint {name!r} is an unauthenticated "
                                "partial source tensor."
                            )
                if int(tensor._version) != version_before:
                    raise Lfm2RetainedTraceRowsError(
                        f"Retained layer {layer_index} endpoint {field!r} changed "
                        "during authentication."
                    )
                tensor_versions[(layer_index, field)] = version_before
        trace_identity = {
            "schema": _TRACE_ROWS_SCHEMA,
            "cache_key": cache_key,
            "corpus_fingerprint": corpus_fingerprint,
            "paths": list(paths),
            "path_splits": list(path_splits),
            "tensors": sorted(
                trace_descriptors,
                key=lambda row: (row["layer"], row["field"]),
            ),
        }
        if hashlib.sha256(_canonical_json(trace_identity)).hexdigest() != (
            trace_rows_sha256
        ):
            raise Lfm2RetainedTraceRowsError(
                "Retained rows differ from their canonical trace-row identity."
            )
        for reference, phases in zip(references, phase_inventory, strict=True):
            names = {
                f"{phase}.layer.{layer_index:03d}.{field}"
                for phase in phases
                for layer_index in range(len(layer_kinds))
                for field in layer_fields[layer_index]
            }
            available_by_batch[reference.index] = frozenset(names)
        return cls(
            rows=rows,
            batch_references=references,
            phases_by_batch=phase_inventory,
            expected_layer_kinds=layer_kinds,
            _phase_slices=MappingProxyType(phase_slices),
            _available_names_by_batch=MappingProxyType(available_by_batch),
            _records_by_batch=MappingProxyType(
                {
                    reference.index: MappingProxyType(record)
                    for reference, record in zip(references, records, strict=True)
                }
            ),
            _tensor_versions=MappingProxyType(tensor_versions),
        )

    def _verify_reuse_context(
        self,
        *,
        cache_key: str,
        corpus_fingerprint: str,
        batch_references: Sequence[Any],
        phases_by_batch: Sequence[Sequence[str]],
        expected_layer_kinds: Sequence[str],
    ) -> None:
        references = tuple(batch_references)
        if (
            getattr(self.rows, "cache_key", None) != cache_key
            or getattr(self.rows, "corpus_fingerprint", None) != corpus_fingerprint
            or self.phases_by_batch != tuple(tuple(value) for value in phases_by_batch)
            or self.expected_layer_kinds != tuple(expected_layer_kinds)
            or tuple(_reference_identity(value) for value in self.batch_references)
            != tuple(_reference_identity(value) for value in references)
        ):
            raise Lfm2RetainedTraceRowsError(
                "Authenticated retained-row adapter was reused in another context."
            )
        for (layer_index, field), version in self._tensor_versions.items():
            tensor = self.rows.layer_fields[layer_index][field]
            if int(tensor._version) != version:
                raise Lfm2RetainedTraceRowsError(
                    f"Retained layer {layer_index} endpoint {field!r} changed "
                    "after authentication."
                )

    def available_tensor_names(self, batch_index: int) -> frozenset[str]:
        try:
            return self._available_names_by_batch[batch_index]
        except KeyError as exc:
            raise Lfm2RetainedTraceRowsError(
                f"Unknown retained-row batch {batch_index}."
            ) from exc

    def named_tensor(self, reference: Any, name: str) -> Any:
        match = _TENSOR_NAME.fullmatch(name)
        if match is None:
            raise Lfm2RetainedTraceRowsError(
                f"Retained tensor name {name!r} is not canonical."
            )
        phase = match.group(1)
        layer_index = int(match.group(2))
        field = match.group(3)
        batch_index = getattr(reference, "index", None)
        if (
            not isinstance(batch_index, int)
            or name not in self.available_tensor_names(batch_index)
            or layer_index < 0
            or layer_index >= len(self.expected_layer_kinds)
        ):
            raise Lfm2RetainedTraceRowsError(
                f"Retained tensor {name!r} is foreign to batch {batch_index}."
            )
        try:
            row_slice = self._phase_slices[(batch_index, phase)]
            source = self.rows.layer_fields[layer_index][field]
            tensor = source[row_slice]
        except KeyError as exc:
            raise Lfm2RetainedTraceRowsError(
                f"Retained tensor {name!r} is missing."
            ) from exc
        if int(source._version) != self._tensor_versions[(layer_index, field)]:
            raise Lfm2RetainedTraceRowsError(
                f"Retained tensor {name!r} changed after authentication."
            )
        descriptor = _descriptor(
            self._records_by_batch[batch_index],
            batch_index,
            name,
        )
        torch = _torch()
        if not bool(torch.isfinite(tensor).all().item()):
            raise Lfm2RetainedTraceRowsError(
                f"Retained tensor {name!r} contains non-finite values."
            )
        restored = _restore_source_tensor(tensor, field)
        if (
            tuple(restored.shape) != tuple(descriptor["shape"])
            or str(restored.dtype).removeprefix("torch.")
            != descriptor["dtype"]
        ):
            raise Lfm2RetainedTraceRowsError(
                f"Retained tensor {name!r} cannot reproduce its source shape "
                "and dtype."
            )
        return restored

    def selected_tensors(
        self,
        reference: Any,
        names: Sequence[str],
    ) -> Mapping[str, Any]:
        selected = tuple(names)
        if not selected or selected != tuple(sorted(set(selected))):
            raise Lfm2RetainedTraceRowsError(
                "Retained tensor selections must be nonempty, unique, and sorted."
            )
        return MappingProxyType(
            {name: self.named_tensor(reference, name) for name in selected}
        )


__all__ = [
    "Lfm2RetainedTraceRowAdapter",
    "Lfm2RetainedTraceRowsError",
]
