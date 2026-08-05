"""LFM2 adapter for family-neutral balanced circuit coordinates.

This adapter binds the generic balanced-circuit basis builder to one
authenticated LFM2 parent and one sealed circuit-trace cache.  It performs no
model execution: parent weights are inventoried structurally, circuit IR is
inspected from modules, and fit/held-out residual snapshots are read from the
already materialized cache.

The cache split is by whole input record, never by rows.  Every accepted cache
record must expose the same complete phase/layer/endpoint residual lattice, so
held-out records cannot leak into basis selection and no traced location can
silently disappear.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_balanced_circuit_basis import (
    BalancedCircuitBasisResult,
    DownstreamObservability,
    EvidenceSplit,
    PrecomputedResidualSnapshotGram,
    ResidualSnapshotBlock,
    _NUMERICAL_LOCK,
    _seal_residual_snapshot_contribution_manifest,
    _snapshot_block_report_and_receipt,
    _symmetrized_gram_contribution,
    _tensor_sha256 as _basis_tensor_sha256,
    build_balanced_circuit_basis,
)
from .dynamic_circuit_ir import CircuitKind, ModelCircuitGraph, canonical_sha256
from .dynamic_model_inspector import inspect_model_circuits
from .dynamic_weight_complete_basis import (
    FeatureOrientation,
    ResidualFacingOperator,
)
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    fingerprint_lfm2_parent,
)
from .hf_lfm2_circuit_trace_row_selection import (
    load_lfm2_circuit_trace_selected_rows,
    release_lfm2_circuit_trace_selected_rows,
)
from .hf_lfm2_sealed_evidence_set import Lfm2MergedSealedEvidenceSession
from .hf_lfm2_weight_complete_basis import (
    Lfm2WeightCompleteInventory,
    inventory_lfm2_weight_complete_operators,
)


LFM2_CIRCUIT_BALANCED_BASIS_SCHEMA = (
    "neural-forge-lfm2-circuit-balanced-basis-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESIDUAL_NAME = re.compile(
    r"^(?P<phase>.+)\.layer\.(?P<layer>[0-9]{3})\."
    r"residual\.(?P<endpoint>pre|post_operator|post_layer)\Z"
)
_DECODE_PHASE = re.compile(r"decode\.(?P<index>[0-9]{3})\Z")
_ENDPOINTS = ("pre", "post_operator", "post_layer")
_VOCABULARY_MARGIN_ROLE = "vocabulary_margin_geometry"
_VOCABULARY_MARGIN_SCHEMA = (
    "neural-forge-lfm2-vocabulary-margin-observability-v1"
)
_STREAMING_ROW_SELECTION_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-balanced-basis-row-selection-progress-v1"
)
_STREAMING_SNAPSHOT_MEMORY_BOUND_SCHEMA = (
    "neural-forge-lfm2-balanced-basis-snapshot-memory-bound-v1"
)
_LOGGER = logging.getLogger(__name__)


class Lfm2CircuitBalancedBasisError(ValueError):
    """The LFM2/cache/IR evidence cannot prove a balanced structural map."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CircuitBalancedBasisError(
            f"{name} must be a positive integer."
        )
    return value


def _unit_interval(
    value: Any,
    name: str,
    *,
    include_zero: bool,
    include_one: bool,
) -> float:
    if isinstance(value, bool):
        raise Lfm2CircuitBalancedBasisError(
            f"{name} must be a finite fraction."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CircuitBalancedBasisError(
            f"{name} must be a finite fraction."
        ) from exc
    lower = result >= 0.0 if include_zero else result > 0.0
    upper = result <= 1.0 if include_one else result < 1.0
    if not math.isfinite(result) or not lower or not upper:
        left = "[" if include_zero else "("
        right = "]" if include_one else ")"
        raise Lfm2CircuitBalancedBasisError(
            f"{name} must be within {left}0, 1{right}."
        )
    return result


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CircuitBalancedBasisError(
            f"{name} must be a nonempty string."
        )
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise Lfm2CircuitBalancedBasisError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _phase_key(value: str) -> tuple[int, int, str]:
    if value == "prefill":
        return (0, 0, value)
    if value.startswith("decode."):
        try:
            return (1, int(value.split(".", 1)[1]), value)
        except ValueError:
            pass
    return (2, 0, value)


def _bounded_rows(value: torch.Tensor, maximum: int) -> torch.Tensor:
    if int(value.shape[0]) <= maximum:
        return value.contiguous()
    indices = (
        torch.arange(maximum, dtype=torch.long)
        * int(value.shape[0])
        // maximum
    )
    return value.index_select(0, indices).contiguous()


def _residual_rows(
    value: Any,
    *,
    source_width: int,
    name: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim < 2
        or int(value.shape[-1]) != source_width
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.numel() == 0
    ):
        raise Lfm2CircuitBalancedBasisError(
            f"Cached residual {name!r} must be a dense floating tensor with "
            f"last dimension {source_width}."
        )
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, source_width)
        .contiguous()
    )
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CircuitBalancedBasisError(
            f"Cached residual {name!r} is non-finite."
        )
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        canonical_sha256([int(size) for size in tensor.shape]).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CachedVocabularyMarginPhase:
    phase: str
    token_ids: torch.Tensor
    logit_values: torch.Tensor

    def __post_init__(self) -> None:
        phase = _nonempty_text(self.phase, "vocabulary-margin phase")
        token_ids = self.token_ids
        logit_values = self.logit_values
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.layout != torch.strided
            or token_ids.device.type == "meta"
            or token_ids.ndim != 2
            or token_ids.dtype != torch.long
            or token_ids.numel() < 2
            or not isinstance(logit_values, torch.Tensor)
            or logit_values.layout != torch.strided
            or logit_values.device.type == "meta"
            or logit_values.ndim != 2
            or tuple(logit_values.shape) != tuple(token_ids.shape)
            or not torch.is_floating_point(logit_values)
            or torch.is_complex(logit_values)
            or not bool(torch.isfinite(logit_values).all().item())
            or int(token_ids.shape[1]) < 2
            or not bool(
                (
                    logit_values[:, :1] >= logit_values[:, 1:]
                ).all().item()
            )
        ):
            raise Lfm2CircuitBalancedBasisError(
                f"Cached logits for phase {phase!r} did not produce a valid "
                "top-1/competitor table."
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "token_ids",
            token_ids.detach().to(device="cpu").contiguous().clone(),
        )
        object.__setattr__(
            self,
            "logit_values",
            logit_values.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone(),
        )

    @property
    def row_count(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.token_ids.shape[1]) - 1

    @property
    def margins(self) -> torch.Tensor:
        return (
            self.logit_values[:, :1] - self.logit_values[:, 1:]
        ).contiguous()


def _deterministic_top_tokens(
    logits: Any,
    *,
    vocabulary_size: int,
    top_k: int,
    phase: str,
) -> _CachedVocabularyMarginPhase:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.layout != torch.strided
        or logits.device.type == "meta"
        or logits.ndim < 2
        or int(logits.shape[-1]) != vocabulary_size
        or not torch.is_floating_point(logits)
        or torch.is_complex(logits)
        or logits.numel() == 0
    ):
        raise Lfm2CircuitBalancedBasisError(
            f"Cached {phase!r} logits must be a nonempty dense floating "
            f"tensor with vocabulary width {vocabulary_size}."
        )
    rows = (
        logits.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, vocabulary_size)
        .contiguous()
    )
    if not bool(torch.isfinite(rows).all().item()):
        raise Lfm2CircuitBalancedBasisError(
            f"Cached {phase!r} logits are non-finite."
        )
    candidate_count = top_k + 1
    token_rows: list[list[int]] = []
    value_rows: list[list[float]] = []
    for row in rows:
        # ``topk`` is used only to discover the exact kth-value threshold.
        # All values above it and the lowest token IDs tied at the boundary
        # are then selected explicitly.  This makes ties canonical without a
        # full-vocabulary sort or implementation-dependent topk indices.
        threshold = torch.topk(
            row,
            k=candidate_count,
            largest=True,
            sorted=False,
        ).values.min()
        above = torch.nonzero(row > threshold, as_tuple=False).flatten()
        tied = torch.nonzero(row == threshold, as_tuple=False).flatten()
        remaining = candidate_count - int(above.numel())
        if remaining <= 0 or int(tied.numel()) < remaining:
            raise Lfm2CircuitBalancedBasisError(
                f"Could not canonicalize cached {phase!r} logit ties."
            )
        chosen = torch.cat((above, tied[:remaining])).tolist()
        ordered = sorted(
            (int(token_id) for token_id in chosen),
            key=lambda token_id: (-float(row[token_id].item()), token_id),
        )
        token_rows.append(ordered)
        value_rows.append(
            [float(row[token_id].item()) for token_id in ordered]
        )
    return _CachedVocabularyMarginPhase(
        phase=phase,
        token_ids=torch.tensor(token_rows, dtype=torch.long),
        logit_values=torch.tensor(value_rows, dtype=torch.float64),
    )


@dataclass(frozen=True, slots=True)
class _CacheRecord:
    index: int
    record_id: str
    locations: Mapping[tuple[str, int, str], torch.Tensor]
    vocabulary_margins: Mapping[str, _CachedVocabularyMarginPhase]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise Lfm2CircuitBalancedBasisError(
                "Cache record index must be an integer."
            )
        record_id = _sha256(self.record_id, "cache record input_sha256")
        locations = dict(self.locations)
        vocabulary_margins = dict(self.vocabulary_margins)
        if not locations:
            raise Lfm2CircuitBalancedBasisError(
                "A cache record must contain residual snapshots."
            )
        if any(
            phase != value.phase
            for phase, value in vocabulary_margins.items()
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Vocabulary-margin phases are not canonically keyed."
            )
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(
            self,
            "locations",
            MappingProxyType(locations),
        )
        object.__setattr__(
            self,
            "vocabulary_margins",
            MappingProxyType(vocabulary_margins),
        )

    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {phase for phase, _layer, _endpoint in self.locations},
                key=_phase_key,
            )
        )

    @property
    def phase_count(self) -> int:
        return len(self.phases)

    @property
    def decode_phase_count(self) -> int:
        return self.phase_count - 1

    @property
    def location_count(self) -> int:
        return len(self.locations)


@dataclass(frozen=True, slots=True)
class _CacheRecordSummary:
    """Tensor-free authenticated shape summary for one cache record."""

    index: int
    record_id: str
    phases: tuple[str, ...]
    location_row_counts: Mapping[tuple[str, int, str], int]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise Lfm2CircuitBalancedBasisError(
                "Cache record index must be an integer."
            )
        record_id = _sha256(self.record_id, "cache record input_sha256")
        phases = tuple(self.phases)
        row_counts = dict(self.location_row_counts)
        if (
            not phases
            or phases != tuple(sorted(set(phases), key=_phase_key))
            or not row_counts
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                for count in row_counts.values()
            )
            or {phase for phase, _layer, _endpoint in row_counts}
            != set(phases)
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Cache record tensor-free row metadata is inconsistent."
            )
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(
            self,
            "location_row_counts",
            MappingProxyType(row_counts),
        )

    @property
    def phase_count(self) -> int:
        return len(self.phases)

    @property
    def decode_phase_count(self) -> int:
        return self.phase_count - 1

    @property
    def location_count(self) -> int:
        return len(self.location_row_counts)


@dataclass(frozen=True, slots=True)
class _SelectiveRecordMetadata:
    """Authenticated reference plus the residual names it describes."""

    reference: Any
    summary: _CacheRecordSummary
    tensor_names: Mapping[tuple[str, int, str], str]

    def __post_init__(self) -> None:
        names = dict(self.tensor_names)
        if set(names) != set(self.summary.location_row_counts):
            raise Lfm2CircuitBalancedBasisError(
                "Selective cache tensor names and row metadata disagree."
            )
        object.__setattr__(self, "tensor_names", MappingProxyType(names))


@dataclass(frozen=True, slots=True)
class _StreamingSplitMemoryStats:
    maximum_phase_buffer_fp64_bytes: int
    maximum_selected_source_payload_bytes: int
    maximum_source_slice_or_hash_overhead_bytes: int
    maximum_fp64_conversion_bytes: int
    maximum_fp64_finiteness_scratch_bytes: int
    maximum_block_fp64_bytes: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.maximum_phase_buffer_fp64_bytes,
                self.maximum_selected_source_payload_bytes,
                self.maximum_source_slice_or_hash_overhead_bytes,
                self.maximum_fp64_conversion_bytes,
                self.maximum_fp64_finiteness_scratch_bytes,
                self.maximum_block_fp64_bytes,
            )
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Streaming split memory statistics are invalid."
            )


@dataclass(frozen=True, slots=True)
class _StreamingSnapshotLoad:
    """Bounded sufficient statistics plus tensor-free cache reports."""

    records: tuple[_CacheRecordSummary, ...]
    fit_records: tuple[_CacheRecordSummary, ...]
    heldout_records: tuple[_CacheRecordSummary, ...]
    fit_snapshot_gram: PrecomputedResidualSnapshotGram
    heldout_snapshot_gram: PrecomputedResidualSnapshotGram
    fit_block_count: int
    heldout_block_count: int
    split_report: Mapping[str, Any]
    source_row_count: int
    selected_snapshot_row_count: int
    maximum_resident_fp64_snapshot_rows: int
    maximum_resident_snapshot_logical_bytes: int
    memory_bound_report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not self.records
            or not self.fit_records
            or not self.heldout_records
            or not isinstance(
                self.fit_snapshot_gram,
                PrecomputedResidualSnapshotGram,
            )
            or not isinstance(
                self.heldout_snapshot_gram,
                PrecomputedResidualSnapshotGram,
            )
            or self.fit_block_count <= 0
            or self.heldout_block_count <= 0
            or not isinstance(self.split_report, Mapping)
            or self.source_row_count <= 0
            or self.selected_snapshot_row_count <= 0
            or self.selected_snapshot_row_count > self.source_row_count
            or self.maximum_resident_fp64_snapshot_rows <= 0
            or self.maximum_resident_fp64_snapshot_rows
            > self.selected_snapshot_row_count
            or self.maximum_resident_snapshot_logical_bytes <= 0
            or not isinstance(self.memory_bound_report, Mapping)
            or self.memory_bound_report.get("schema")
            != _STREAMING_SNAPSHOT_MEMORY_BOUND_SCHEMA
            or self.memory_bound_report.get(
                "maximum_resident_snapshot_logical_bytes"
            )
            != self.maximum_resident_snapshot_logical_bytes
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Streaming snapshot-load invariants are invalid."
            )
        object.__setattr__(
            self,
            "split_report",
            MappingProxyType(dict(self.split_report)),
        )
        object.__setattr__(
            self,
            "memory_bound_report",
            MappingProxyType(dict(self.memory_bound_report)),
        )


def _cache_record_from_tensors(
    tensors: Mapping[str, Any],
    *,
    index: int,
    record_id: str,
    source_width: int,
    layer_count: int,
    vocabulary_size: int,
    vocabulary_margin_top_k: int | None,
) -> _CacheRecord:
    if not tensors:
        raise Lfm2CircuitBalancedBasisError(
            "Every verified cache batch must expose a tensor mapping."
        )
    locations: dict[tuple[str, int, str], torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or name.startswith("cached_prefill."):
            continue
        match = _RESIDUAL_NAME.fullmatch(name)
        if match is None:
            continue
        phase = match.group("phase")
        layer = int(match.group("layer"))
        endpoint = match.group("endpoint")
        location = (phase, layer, endpoint)
        if location in locations:
            raise Lfm2CircuitBalancedBasisError(
                f"Cache batch {index} repeats residual location {name!r}."
            )
        locations[location] = _residual_rows(
            tensor,
            source_width=source_width,
            name=name,
        )
    phases = sorted(
        {phase for phase, _layer, _endpoint in locations},
        key=_phase_key,
    )
    if not phases:
        raise Lfm2CircuitBalancedBasisError(
            f"Cache batch {index} has no canonical residual phases."
        )
    decode_indices = []
    unexpected_phases = []
    for phase in phases:
        if phase == "prefill":
            continue
        match = _DECODE_PHASE.fullmatch(phase)
        if match is None:
            unexpected_phases.append(phase)
        else:
            decode_indices.append(int(match.group("index")))
    if (
        phases.count("prefill") != 1
        or not decode_indices
        or unexpected_phases
        or decode_indices != list(range(len(decode_indices)))
    ):
        raise Lfm2CircuitBalancedBasisError(
            f"Cache batch {index} phases must be exactly prefill plus "
            "one or more contiguous decode.000...decode.NNN phases; "
            f"observed={phases}."
        )
    expected = {
        (phase, layer, endpoint)
        for phase in phases
        for layer in range(layer_count)
        for endpoint in _ENDPOINTS
    }
    observed = set(locations)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise Lfm2CircuitBalancedBasisError(
            f"Cache batch {index} does not expose the complete "
            "phase/layer/residual-endpoint lattice; "
            f"missing={missing[:8]}, extra={extra[:8]}."
        )
    vocabulary_margins: dict[str, _CachedVocabularyMarginPhase] = {}
    if vocabulary_margin_top_k is not None:
        for phase in phases:
            logits_name = f"{phase}.logits"
            if logits_name not in tensors:
                raise Lfm2CircuitBalancedBasisError(
                    "Vocabulary-margin observability requires full parent "
                    f"logits at authenticated cache phase {phase!r}."
                )
            margin_phase = _deterministic_top_tokens(
                tensors[logits_name],
                vocabulary_size=vocabulary_size,
                top_k=vocabulary_margin_top_k,
                phase=phase,
            )
            final_rows = locations[(phase, layer_count - 1, "post_layer")]
            if margin_phase.row_count != int(final_rows.shape[0]):
                raise Lfm2CircuitBalancedBasisError(
                    f"Cached {phase!r} logits and final residual rows do not align."
                )
            vocabulary_margins[phase] = margin_phase
    return _CacheRecord(
        index=index,
        record_id=record_id,
        locations=locations,
        vocabulary_margins=vocabulary_margins,
    )


def _load_cache_records(
    trace_cache: Any,
    *,
    source_width: int,
    layer_count: int,
    vocabulary_size: int,
    vocabulary_margin_top_k: int | None,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
) -> tuple[_CacheRecord, ...]:
    records: list[_CacheRecord] = []
    if selective_session is None:
        iterator = getattr(trace_cache, "iter_batches", None)
        if not callable(iterator):
            raise Lfm2CircuitBalancedBasisError(
                "trace_cache must expose iter_batches()."
            )
        for fallback_index, batch in enumerate(iterator()):
            tensors = getattr(batch, "tensors", None)
            if not isinstance(tensors, Mapping):
                raise Lfm2CircuitBalancedBasisError(
                    "Every verified cache batch must expose a tensor mapping."
                )
            records.append(
                _cache_record_from_tensors(
                    tensors,
                    index=getattr(batch, "index", fallback_index),
                    record_id=getattr(batch, "input_sha256", None),
                    source_width=source_width,
                    layer_count=layer_count,
                    vocabulary_size=vocabulary_size,
                    vocabulary_margin_top_k=vocabulary_margin_top_k,
                )
            )
    else:
        for reference in selective_session.batch_references:
            residual_names = tuple(
                name
                for name in reference.ordered_tensor_names
                if _RESIDUAL_NAME.fullmatch(name) is not None
                and not name.startswith("cached_prefill.")
            )
            phases = {
                match.group("phase")
                for name in residual_names
                if (match := _RESIDUAL_NAME.fullmatch(name)) is not None
            }
            selected_names = set(residual_names)
            if vocabulary_margin_top_k is not None:
                selected_names.update(f"{phase}.logits" for phase in phases)
            ordered_names = tuple(sorted(selected_names))
            payload = selective_session.load_selected_tensors(
                reference.index,
                ordered_names,
            )
            try:
                records.append(
                    _cache_record_from_tensors(
                        payload.tensors,
                        index=reference.index,
                        record_id=reference.input_sha256,
                        source_width=source_width,
                        layer_count=layer_count,
                        vocabulary_size=vocabulary_size,
                        vocabulary_margin_top_k=vocabulary_margin_top_k,
                    )
                )
            finally:
                selective_session.release_selected_tensors(payload)
    if len(records) < 2:
        raise Lfm2CircuitBalancedBasisError(
            "At least two sealed cache records are required for disjoint "
            "fit and held-out evidence."
        )
    if len({value.index for value in records}) != len(records):
        raise Lfm2CircuitBalancedBasisError(
            "Cache batch indices must be unique."
        )
    if len({value.record_id for value in records}) != len(records):
        raise Lfm2CircuitBalancedBasisError(
            "Cache input_sha256 record IDs must be unique."
        )
    return tuple(sorted(records, key=lambda value: value.index))


def _extension_batch_indices(handle: Any) -> frozenset[int]:
    """Merged batch indices on-policy extensions contributed to one handle.

    A single sealed cache or session has no extension members and exposes no
    such attribute, so this is empty for every single-cache solve -- which is
    what keeps the split below byte-identical there.
    """

    value = getattr(handle, "extension_batch_indices", None)
    if value is None:
        return frozenset()
    try:
        indices = frozenset(int(index) for index in value)
    except (TypeError, ValueError) as exc:
        raise Lfm2CircuitBalancedBasisError(
            "Sealed evidence extension batch indices must be integers."
        ) from exc
    if any(index < 0 for index in indices):
        raise Lfm2CircuitBalancedBasisError(
            "Sealed evidence extension batch indices must be non-negative."
        )
    return indices


def _resolved_extension_batch_indices(
    trace_cache: Any,
    selective_session: Any,
) -> frozenset[int]:
    """The one set of extension batches both handles agree this solve reads."""

    cache_indices = _extension_batch_indices(trace_cache)
    if selective_session is None:
        return cache_indices
    session_indices = _extension_batch_indices(selective_session)
    if cache_indices != session_indices:
        raise Lfm2CircuitBalancedBasisError(
            "Sealed cache and session disagree on which batches on-policy "
            "extensions contributed."
        )
    return cache_indices


def _split_records(
    records: Sequence[_CacheRecord],
    *,
    cache_key: str,
    validation_fraction: float,
    split_seed: str,
    extension_batch_indices: frozenset[int] = frozenset(),
) -> tuple[tuple[_CacheRecord, ...], tuple[_CacheRecord, ...], dict[str, Any]]:
    count = len(records)
    heldout_count = max(1, min(count - 1, math.ceil(count * validation_fraction)))
    eligible = records
    if extension_batch_indices:
        # Every row an on-policy extension contributes is a fit row.  The
        # sealed evidence set already refuses any other split label
        # (Lfm2SealedEvidenceRequest: "validation would stop being held out"),
        # and the same rule has to hold for the basis's own fit/held-out
        # ranking: a child-visited state in the held-out split would have the
        # projection gate measure the child partly on its own outputs.  The
        # ranking is unchanged; only the pool it draws the held-out quota from
        # is narrowed to the base members.
        present = {int(value.index) for value in records}
        if not extension_batch_indices <= present:
            raise Lfm2CircuitBalancedBasisError(
                "Sealed evidence extension batches are not among the records "
                "this split covers."
            )
        eligible = tuple(
            value
            for value in records
            if int(value.index) not in extension_batch_indices
        )
        if len(eligible) < 2:
            raise Lfm2CircuitBalancedBasisError(
                "A merged sealed evidence split needs at least two base "
                "records for disjoint fit and held-out evidence."
            )
        heldout_count = max(1, min(len(eligible) - 1, heldout_count))
    ranked = sorted(
        eligible,
        key=lambda value: (
            hashlib.sha256(
                (
                    split_seed
                    + "\0"
                    + cache_key
                    + "\0"
                    + value.record_id
                ).encode("utf-8")
            ).hexdigest(),
            value.index,
        ),
    )
    heldout_ids = {value.record_id for value in ranked[:heldout_count]}
    fit = tuple(value for value in records if value.record_id not in heldout_ids)
    heldout = tuple(
        value for value in records if value.record_id in heldout_ids
    )
    report = {
        "method": "whole-record-seeded-sha256-ranking",
        "split_seed_sha256": hashlib.sha256(
            split_seed.encode("utf-8")
        ).hexdigest(),
        "requested_validation_fraction": validation_fraction,
        "record_count": count,
        "fit_record_count": len(fit),
        "heldout_record_count": len(heldout),
        "actual_heldout_fraction": len(heldout) / count,
        "fit_record_ids_sha256": canonical_sha256(
            sorted(value.record_id for value in fit)
        ),
        "heldout_record_ids_sha256": canonical_sha256(
            sorted(value.record_id for value in heldout)
        ),
        "whole_records_only": True,
        "record_ids_disjoint": True,
        "heldout_excluded_from_selection": True,
        # Present only for merged evidence, so a single-cache split report --
        # and therefore every basis report digest it enters -- is unchanged.
        **(
            {}
            if not extension_batch_indices
            else {
                "on_policy_extension_record_count": count - len(eligible),
                "heldout_eligible_record_count": len(eligible),
                "heldout_extension_record_count": 0,
                "extension_records_are_never_heldout": True,
                "heldout_pool": "base-members-only",
            }
        ),
    }
    return fit, heldout, report


def _snapshot_blocks(
    records: Sequence[_CacheRecord],
    *,
    split: EvidenceSplit,
    max_rows_per_block: int,
) -> tuple[ResidualSnapshotBlock, ...]:
    if not records:
        raise Lfm2CircuitBalancedBasisError(
            f"{split.value} cache record split is empty."
        )
    locations = sorted(
        {
            location
            for record in records
            for location in record.locations
        },
        key=lambda value: (
            _phase_key(value[0]),
            value[1],
            _ENDPOINTS.index(value[2]),
        ),
    )
    blocks = []
    for phase, layer_index, endpoint in locations:
        contributors = tuple(
            value
            for value in records
            if (phase, layer_index, endpoint) in value.locations
        )
        if not contributors:  # pragma: no cover - union construction proves it
            raise AssertionError("snapshot location has no contributing record")
        rows = _bounded_rows(
            torch.cat(
                [
                    value.locations[(phase, layer_index, endpoint)]
                    for value in contributors
                ],
                dim=0,
            ),
            max_rows_per_block,
        )
        layer = f"layer.{layer_index:03d}"
        blocks.append(
            ResidualSnapshotBlock(
                block_id=(
                    f"{split.value}.{phase}.{layer}.residual.{endpoint}"
                ),
                split=split,
                phase=phase,
                layer=layer,
                endpoint=f"residual.{endpoint}",
                rows=rows,
                record_ids=tuple(
                    value.record_id for value in contributors
                ),
            )
        )
    return tuple(blocks)


def _selective_record_metadata(
    selective_session: Any,
    *,
    source_width: int,
    layer_count: int,
) -> tuple[_SelectiveRecordMetadata, ...]:
    """Read the authenticated tensor inventory without loading payload rows."""

    references = getattr(selective_session, "batch_references", None)
    if not isinstance(references, tuple) or not references:
        raise Lfm2CircuitBalancedBasisError(
            "Selective trace session must expose authenticated batch references."
        )
    metadata: list[_SelectiveRecordMetadata] = []
    for reference in references:
        index = getattr(reference, "index", None)
        record_id = getattr(reference, "input_sha256", None)
        ordered_names = getattr(reference, "ordered_tensor_names", None)
        record = getattr(reference, "record", None)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(ordered_names, tuple)
            or ordered_names != tuple(sorted(set(ordered_names)))
            or not isinstance(record, Mapping)
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Selective trace reference metadata is invalid."
            )
        inventory = record.get("tensors")
        if not isinstance(inventory, Mapping):
            raise Lfm2CircuitBalancedBasisError(
                f"Cache batch {index} has no authenticated tensor inventory."
            )
        row_counts: dict[tuple[str, int, str], int] = {}
        tensor_names: dict[tuple[str, int, str], str] = {}
        for name in ordered_names:
            if not isinstance(name, str) or name.startswith("cached_prefill."):
                continue
            match = _RESIDUAL_NAME.fullmatch(name)
            if match is None:
                continue
            descriptor = inventory.get(name)
            if not isinstance(descriptor, Mapping):
                raise Lfm2CircuitBalancedBasisError(
                    f"Cache batch {index} residual {name!r} has no descriptor."
                )
            shape = descriptor.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) < 2
                or any(
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size <= 0
                    for size in shape
                )
                or shape[-1] != source_width
                or descriptor.get("numel") != math.prod(shape)
                or not isinstance(descriptor.get("dtype"), str)
            ):
                raise Lfm2CircuitBalancedBasisError(
                    f"Cache batch {index} residual {name!r} descriptor is invalid."
                )
            location = (
                match.group("phase"),
                int(match.group("layer")),
                match.group("endpoint"),
            )
            if location in row_counts:
                raise Lfm2CircuitBalancedBasisError(
                    f"Cache batch {index} repeats residual location {name!r}."
                )
            row_counts[location] = math.prod(shape[:-1])
            tensor_names[location] = name
        phases = tuple(
            sorted(
                {phase for phase, _layer, _endpoint in row_counts},
                key=_phase_key,
            )
        )
        decode_indices: list[int] = []
        unexpected_phases: list[str] = []
        for phase in phases:
            if phase == "prefill":
                continue
            match = _DECODE_PHASE.fullmatch(phase)
            if match is None:
                unexpected_phases.append(phase)
            else:
                decode_indices.append(int(match.group("index")))
        if (
            phases.count("prefill") != 1
            or not decode_indices
            or unexpected_phases
            or decode_indices != list(range(len(decode_indices)))
        ):
            raise Lfm2CircuitBalancedBasisError(
                f"Cache batch {index} phases must be exactly prefill plus "
                "one or more contiguous decode.000...decode.NNN phases; "
                f"observed={list(phases)}."
            )
        expected = {
            (phase, layer, endpoint)
            for phase in phases
            for layer in range(layer_count)
            for endpoint in _ENDPOINTS
        }
        observed = set(row_counts)
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extra = sorted(observed.difference(expected))
            raise Lfm2CircuitBalancedBasisError(
                f"Cache batch {index} does not expose the complete "
                "phase/layer/residual-endpoint lattice; "
                f"missing={missing[:8]}, extra={extra[:8]}."
            )
        summary = _CacheRecordSummary(
            index=index,
            record_id=record_id,
            phases=phases,
            location_row_counts=row_counts,
        )
        metadata.append(
            _SelectiveRecordMetadata(
                reference=reference,
                summary=summary,
                tensor_names=tensor_names,
            )
        )
    ordered = tuple(sorted(metadata, key=lambda value: value.summary.index))
    if len(ordered) < 2:
        raise Lfm2CircuitBalancedBasisError(
            "At least two sealed cache records are required for disjoint "
            "fit and held-out evidence."
        )
    if len({value.summary.index for value in ordered}) != len(ordered):
        raise Lfm2CircuitBalancedBasisError(
            "Cache batch indices must be unique."
        )
    if len({value.summary.record_id for value in ordered}) != len(ordered):
        raise Lfm2CircuitBalancedBasisError(
            "Cache input_sha256 record IDs must be unique."
        )
    return ordered


def _global_row_selection_plan(
    records: Sequence[_CacheRecordSummary],
    *,
    max_rows_per_block: int,
) -> tuple[
    dict[int, dict[tuple[str, int, str], tuple[int, ...]]],
    int,
    int,
]:
    """Resolve the legacy post-concatenation indices into local row indices."""

    if not records:
        raise Lfm2CircuitBalancedBasisError(
            "A streaming cache split cannot be empty."
        )
    plans: dict[int, dict[tuple[str, int, str], tuple[int, ...]]] = {
        record.index: {} for record in records
    }
    locations = sorted(
        {
            location
            for record in records
            for location in record.location_row_counts
        },
        key=lambda value: (
            _phase_key(value[0]),
            value[1],
            _ENDPOINTS.index(value[2]),
        ),
    )
    source_row_count = 0
    selected_row_count = 0
    for location in locations:
        contributors = tuple(
            record
            for record in records
            if location in record.location_row_counts
        )
        total = sum(
            record.location_row_counts[location]
            for record in contributors
        )
        source_row_count += total
        if total <= max_rows_per_block:
            global_indices = tuple(range(total))
        else:
            global_indices = tuple(
                int(value)
                for value in (
                    torch.arange(max_rows_per_block, dtype=torch.long)
                    * total
                    // max_rows_per_block
                ).tolist()
            )
        selected_row_count += len(global_indices)
        cursor = 0
        offset = 0
        for record in contributors:
            count = record.location_row_counts[location]
            end = offset + count
            local: list[int] = []
            while (
                cursor < len(global_indices)
                and global_indices[cursor] < end
            ):
                if global_indices[cursor] < offset:  # pragma: no cover
                    raise AssertionError("global row selection moved backwards")
                local.append(global_indices[cursor] - offset)
                cursor += 1
            plans[record.index][location] = tuple(local)
            offset = end
        if cursor != len(global_indices) or offset != total:  # pragma: no cover
            raise AssertionError("global row selection did not cover its source")
    return plans, source_row_count, selected_row_count


def _selected_residual_rows(
    value: Any,
    indices: tuple[int, ...],
    *,
    source_width: int,
    expected_row_count: int,
    name: str,
) -> torch.Tensor:
    """Select authenticated source rows before their CPU-float64 conversion."""

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim < 2
        or int(value.shape[-1]) != source_width
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.numel() == 0
    ):
        raise Lfm2CircuitBalancedBasisError(
            f"Cached residual {name!r} must be a dense floating tensor with "
            f"last dimension {source_width}."
        )
    source = value.detach().reshape(-1, source_width)
    if (
        int(source.shape[0]) != expected_row_count
        or not bool(torch.isfinite(source).all().item())
        or not indices
        or indices != tuple(sorted(set(indices)))
        or indices[0] < 0
        or indices[-1] >= expected_row_count
    ):
        raise Lfm2CircuitBalancedBasisError(
            f"Cached residual {name!r} cannot satisfy its authenticated "
            "streaming row selection."
        )
    selected = source.index_select(
        0,
        torch.tensor(indices, dtype=torch.long, device=source.device),
    )
    result = selected.to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):  # pragma: no cover
        raise Lfm2CircuitBalancedBasisError(
            f"Cached residual {name!r} selected non-finite rows."
        )
    return result


def _preselected_residual_rows(
    value: Any,
    indices: tuple[int, ...],
    *,
    source_width: int,
    name: str,
) -> torch.Tensor:
    """Convert already selected source-dtype rows with legacy arithmetic."""

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or tuple(value.shape) != (len(indices), source_width)
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.numel() == 0
        or not indices
        or indices != tuple(sorted(set(indices)))
    ):
        raise Lfm2CircuitBalancedBasisError(
            f"Authenticated residual rows {name!r} do not match their "
            "canonical selection plan."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CircuitBalancedBasisError(
            f"Authenticated residual rows {name!r} selected non-finite rows."
        )
    return result


def _emit_streaming_row_selection_progress(event: Mapping[str, Any]) -> None:
    """Emit out-of-band progress without changing authenticated basis hashes."""

    payload = dict(event)
    if payload.get("schema") != _STREAMING_ROW_SELECTION_PROGRESS_SCHEMA:
        raise Lfm2CircuitBalancedBasisError(
            "Streaming row-selection progress has the wrong schema."
        )
    _LOGGER.info(
        "balanced-basis row selection batch %d/%d: %d/%d rows, %d/%d bytes",
        payload["completed_batch_count"],
        payload["batch_count"],
        payload["selected_row_count"],
        payload["source_row_count"],
        payload["selected_logical_bytes"],
        payload["source_logical_bytes"],
        extra={"neural_forge_progress": payload},
    )


def _stream_selective_cache_snapshots(
    selective_session: Any,
    *,
    source_width: int,
    layer_count: int,
    cache_key: str,
    validation_fraction: float,
    split_seed: str,
    max_rows_per_block: int,
    extension_batch_indices: frozenset[int] | None = None,
) -> _StreamingSnapshotLoad:
    """Compatibility alias for bounded snapshot-Gram streaming."""

    return _stream_selective_cache_snapshot_grams(
        selective_session,
        source_width=source_width,
        layer_count=layer_count,
        cache_key=cache_key,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        max_rows_per_block=max_rows_per_block,
        extension_batch_indices=extension_batch_indices,
    )


def _authenticated_row_api(selective_session: Any) -> tuple[Any, Any]:
    """Resolve bounded row reads without permitting a production fallback."""

    load = getattr(selective_session, "load_selected_tensor_rows", None)
    release = getattr(selective_session, "release_selected_tensor_rows", None)
    if callable(load) and callable(release):
        return load, release
    if callable(load) != callable(release):
        raise Lfm2CircuitBalancedBasisError(
            "Selective trace session exposes an incomplete row-slice API."
        )
    if not isinstance(
        selective_session,
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Selective trace session cannot authenticate bounded row reads."
        )
    return (
        lambda index, rows: load_lfm2_circuit_trace_selected_rows(
            selective_session,
            index,
            rows,
        ),
        lambda payload: release_lfm2_circuit_trace_selected_rows(
            selective_session,
            payload,
        ),
    )


def _snapshot_gram_location_key(
    location: tuple[str, int, str],
    split: EvidenceSplit,
) -> tuple[str, str, str, str]:
    phase, layer_index, endpoint = location
    layer = f"layer.{layer_index:03d}"
    endpoint_name = f"residual.{endpoint}"
    block_id = f"{split.value}.{phase}.{layer}.{endpoint_name}"
    return phase, layer, endpoint_name, block_id


def _conservative_python_container_bytes(value: Any) -> int:
    """Measure owned container overhead, deliberately double-counting aliases."""

    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        mapping_table_bound = 64 + 128 * len(value)
        return size + mapping_table_bound + sum(
            _conservative_python_container_bytes(key)
            + _conservative_python_container_bytes(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(
            _conservative_python_container_bytes(item) for item in value
        )
    return size


def _streaming_metadata_overhead_bound_bytes(
    metadata: Sequence[_SelectiveRecordMetadata],
) -> int:
    total = sys.getsizeof(metadata)
    for value in metadata:
        total += sys.getsizeof(value)
        total += sys.getsizeof(value.summary)
        total += _conservative_python_container_bytes(value.summary.phases)
        total += _conservative_python_container_bytes(
            value.summary.location_row_counts
        )
        total += _conservative_python_container_bytes(value.tensor_names)
        total += sys.getsizeof(value.summary.index)
        total += sys.getsizeof(value.summary.record_id)
    return total


def _stream_snapshot_gram_split(
    metadata: Sequence[_SelectiveRecordMetadata],
    records: Sequence[_CacheRecordSummary],
    plan: Mapping[int, Mapping[tuple[str, int, str], tuple[int, ...]]],
    *,
    split: EvidenceSplit,
    source_width: int,
    load_selected_rows: Any,
    release_selected_rows: Any,
) -> tuple[PrecomputedResidualSnapshotGram, _StreamingSplitMemoryStats]:
    """Accumulate one exact legacy snapshot Gram in bounded phase windows."""

    record_ids = {value.record_id for value in records}
    locations = tuple(
        sorted(
            {
                location
                for record in records
                for location in record.location_row_counts
            },
            key=lambda location: _snapshot_gram_location_key(location, split),
        )
    )
    if not locations:
        raise Lfm2CircuitBalancedBasisError(
            f"{split.value} snapshot locations are empty."
        )
    phases = tuple(sorted({location[0] for location in locations}))
    block_count = len(locations)
    raw = torch.zeros((source_width, source_width), dtype=torch.float64)
    balanced = torch.zeros((source_width, source_width), dtype=torch.float64)
    reports: list[dict[str, Any]] = []
    contribution_receipts: list[Any] = []
    block_manifest: list[str] = []
    observed_record_ids: set[str] = set()
    observed_layers: set[str] = set()
    observed_endpoints: set[str] = set()
    observed_locations: set[str] = set()
    total_row_count = 0
    maximum_phase_buffer_bytes = 0
    maximum_source_payload_bytes = 0
    maximum_source_overhead_bytes = 0
    maximum_conversion_bytes = 0
    maximum_finiteness_scratch_bytes = 0
    maximum_block_bytes = 0
    split_metadata = tuple(
        value for value in metadata if value.summary.record_id in record_ids
    )
    batch_work_count = sum(
        phase in value.summary.phases
        for phase in phases
        for value in split_metadata
    )
    completed_batch_work = 0

    for phase in phases:
        phase_locations = tuple(
            location for location in locations if location[0] == phase
        )
        selected_counts = {
            location: sum(
                len(plan[record.index].get(location, ()))
                for record in records
                if location in record.location_row_counts
            )
            for location in phase_locations
        }
        if any(count <= 0 for count in selected_counts.values()):
            raise Lfm2CircuitBalancedBasisError(
                f"{split.value} phase {phase!r} has an empty selected block."
            )
        buffers = {
            location: torch.empty(
                (selected_counts[location], source_width),
                dtype=torch.float64,
            )
            for location in phase_locations
        }
        cursors = {location: 0 for location in phase_locations}
        phase_row_count = sum(selected_counts.values())
        phase_buffer_bytes = phase_row_count * source_width * 8
        maximum_phase_buffer_bytes = max(
            maximum_phase_buffer_bytes,
            phase_buffer_bytes,
        )
        phase_maximum_block_bytes = (
            max(selected_counts.values()) * source_width * 8
        )
        maximum_block_bytes = max(
            maximum_block_bytes,
            phase_maximum_block_bytes,
        )

        for value in split_metadata:
            summary = value.summary
            if phase not in summary.phases:
                continue
            local_indices = plan[summary.index]
            row_selections = {
                value.tensor_names[location]: local_indices[location]
                for location in phase_locations
                if location in summary.location_row_counts
                and local_indices.get(location)
            }
            selected_names = tuple(sorted(row_selections))
            payload = None
            tensors: Mapping[str, Any] = MappingProxyType({})
            try:
                if selected_names:
                    payload = load_selected_rows(
                        summary.index,
                        row_selections,
                    )
                    tensors = getattr(payload, "tensors", None)
                    if (
                        getattr(payload, "batch_index", None) != summary.index
                        or getattr(payload, "input_sha256", None)
                        != summary.record_id
                        or not isinstance(tensors, Mapping)
                        or tuple(sorted(tensors)) != selected_names
                        or dict(getattr(payload, "row_indices", {}))
                        != row_selections
                    ):
                        raise Lfm2CircuitBalancedBasisError(
                            "Selective trace payload is not bound to its row plan."
                        )
                    payload_bytes = sum(
                        int(tensor.numel() * tensor.element_size())
                        for tensor in tensors.values()
                    )
                    maximum_tensor_bytes = max(
                        int(tensor.numel() * tensor.element_size())
                        for tensor in tensors.values()
                    )
                    maximum_element_size = max(
                        int(tensor.element_size()) for tensor in tensors.values()
                    )
                    source_slice_bytes = (
                        int(
                            getattr(
                                payload,
                                "maximum_materialized_source_rows",
                                0,
                            )
                        )
                        * source_width
                        * maximum_element_size
                    )
                    source_overhead_bytes = max(
                        maximum_tensor_bytes,
                        source_slice_bytes,
                    )
                    conversion_bytes = max(
                        len(indices) * source_width * 8
                        for indices in row_selections.values()
                    )
                    finiteness_scratch_bytes = conversion_bytes // 8
                    maximum_source_payload_bytes = max(
                        maximum_source_payload_bytes,
                        payload_bytes,
                    )
                    maximum_source_overhead_bytes = max(
                        maximum_source_overhead_bytes,
                        source_overhead_bytes,
                    )
                    maximum_conversion_bytes = max(
                        maximum_conversion_bytes,
                        conversion_bytes,
                    )
                    maximum_finiteness_scratch_bytes = max(
                        maximum_finiteness_scratch_bytes,
                        finiteness_scratch_bytes,
                    )
                    for location in phase_locations:
                        indices = local_indices.get(location, ())
                        if not indices:
                            continue
                        name = value.tensor_names[location]
                        selected = _preselected_residual_rows(
                            tensors[name],
                            indices,
                            source_width=source_width,
                            name=name,
                        )
                        cursor = cursors[location]
                        end = cursor + len(indices)
                        buffers[location][cursor:end].copy_(selected)
                        cursors[location] = end
                        del selected
            finally:
                if payload is not None:
                    release_selected_rows(payload)
            completed_batch_work += 1
            _emit_streaming_row_selection_progress(
                {
                    "schema": _STREAMING_ROW_SELECTION_PROGRESS_SCHEMA,
                    "split": split.value,
                    "phase": phase,
                    "batch_index": summary.index,
                    "completed_batch_count": completed_batch_work,
                    "batch_count": batch_work_count,
                    "selected_tensor_count": len(selected_names),
                    "selected_row_count": sum(
                        len(indices) for indices in row_selections.values()
                    ),
                    "retained_fp64_logical_bytes": (
                        phase_buffer_bytes
                    ),
                    "phase_buffer_fp64_bytes": phase_buffer_bytes,
                    "selected_source_payload_bytes": int(
                        getattr(payload, "logical_bytes", 0)
                    ),
                    "source_slice_or_hash_overhead_bytes": (
                        0 if payload is None else source_overhead_bytes
                    ),
                    "maximum_fp64_conversion_bytes": (
                        0 if payload is None else conversion_bytes
                    ),
                    "maximum_fp64_finiteness_scratch_bytes": (
                        0 if payload is None else finiteness_scratch_bytes
                    ),
                    "maximum_block_fp64_bytes": phase_maximum_block_bytes,
                    "block_owned_clone_count": 1,
                    "block_hash_clone_count": 1,
                    "block_hash_tobytes_copy_count": 1,
                    "source_row_count": sum(
                        summary.location_row_counts[location]
                        for location in phase_locations
                        if location in summary.location_row_counts
                    ),
                    "selected_logical_bytes": int(
                        getattr(payload, "logical_bytes", 0)
                    ),
                    "source_logical_bytes": int(
                        getattr(payload, "source_logical_bytes", 0)
                    ),
                    "maximum_materialized_source_rows": int(
                        getattr(
                            payload,
                            "maximum_materialized_source_rows",
                            0,
                        )
                    ),
                    "safetensors_get_slice_calls": int(
                        getattr(payload, "safetensors_get_slice_calls", 0)
                    ),
                    "coalesced_slice_count": int(
                        getattr(payload, "coalesced_slice_count", 0)
                    ),
                    "authenticated_row_slice_path": True,
                    "whole_tensor_cpu_clone_preceded_row_selection": False,
                    "payload_released": bool(
                        payload is None or getattr(payload, "released", False)
                    ),
                    "phase_windowed_fp64_retention": True,
                    "all_snapshot_blocks_retained": False,
                    "telemetry_excluded_from_basis_evidence_hashes": True,
                }
            )

        if any(cursors[key] != selected_counts[key] for key in phase_locations):
            raise Lfm2CircuitBalancedBasisError(
                f"{split.value} phase {phase!r} did not fill its row plan."
            )
        for phase_value, layer_index, endpoint in phase_locations:
            rows = buffers.pop((phase_value, layer_index, endpoint))
            layer = f"layer.{layer_index:03d}"
            endpoint_name = f"residual.{endpoint}"
            contributors = tuple(
                value
                for value in records
                if (phase_value, layer_index, endpoint)
                in value.location_row_counts
            )
            block = ResidualSnapshotBlock(
                block_id=(
                    f"{split.value}.{phase_value}.{layer}.{endpoint_name}"
                ),
                split=split,
                phase=phase_value,
                layer=layer,
                endpoint=endpoint_name,
                rows=rows,
                record_ids=tuple(value.record_id for value in contributors),
            )
            del rows
            contribution = _symmetrized_gram_contribution(
                block.rows.T @ block.rows * block.gram_weight
            )
            block_report, contribution_receipt, trace = (
                _snapshot_block_report_and_receipt(
                    block,
                    contribution,
                    block_count=block_count,
                )
            )
            raw = raw + contribution
            balanced = balanced + contribution / (block_count * trace)
            reports.append(block_report)
            contribution_receipts.append(contribution_receipt)
            block_manifest.append(block.block_sha256)
            observed_record_ids.update(block.record_ids)
            observed_layers.add(block.layer)
            observed_endpoints.add(block.endpoint)
            observed_locations.add(block.location)
            total_row_count += int(block.rows.shape[0])
            del contribution, block

    raw = ((raw + raw.T) * 0.5).contiguous()
    balanced = ((balanced + balanced.T) * 0.5).contiguous()
    report = {
        "split": split.value,
        "source_width": source_width,
        "block_count": block_count,
        "row_count": total_row_count,
        "record_count": len(observed_record_ids),
        "record_ids_sha256": canonical_sha256(sorted(observed_record_ids)),
        "phases": sorted(phases),
        "layers": sorted(observed_layers),
        "endpoints": sorted(observed_endpoints),
        "locations": sorted(observed_locations),
        "all_supplied_locations_contribute_equally_after_trace_normalization": (
            True
        ),
        "blocks": reports,
        "raw_trace_energy": float(torch.trace(raw).item()),
        "raw_gram_sha256": _basis_tensor_sha256(raw),
        "balanced_gram_sha256": _basis_tensor_sha256(balanced),
        "block_manifest_sha256": canonical_sha256(block_manifest),
    }
    contribution_manifest = _seal_residual_snapshot_contribution_manifest(
        contribution_receipts,
        split=split,
        report=report,
        raw=raw,
        balanced=balanced,
    )
    return (
        PrecomputedResidualSnapshotGram(
            split=split,
            raw=raw,
            balanced=balanced,
            report=report,
            contribution_manifest=contribution_manifest,
            record_ids=tuple(sorted(observed_record_ids)),
        ),
        _StreamingSplitMemoryStats(
            maximum_phase_buffer_fp64_bytes=maximum_phase_buffer_bytes,
            maximum_selected_source_payload_bytes=(
                maximum_source_payload_bytes
            ),
            maximum_source_slice_or_hash_overhead_bytes=(
                maximum_source_overhead_bytes
            ),
            maximum_fp64_conversion_bytes=maximum_conversion_bytes,
            maximum_fp64_finiteness_scratch_bytes=(
                maximum_finiteness_scratch_bytes
            ),
            maximum_block_fp64_bytes=maximum_block_bytes,
        ),
    )


def _stream_selective_cache_snapshot_grams(
    selective_session: Any,
    *,
    source_width: int,
    layer_count: int,
    cache_key: str,
    validation_fraction: float,
    split_seed: str,
    max_rows_per_block: int,
    extension_batch_indices: frozenset[int] | None = None,
) -> _StreamingSnapshotLoad:
    """Build exact snapshot sufficient statistics without retaining blocks."""

    if extension_batch_indices is None:
        extension_batch_indices = _extension_batch_indices(selective_session)
    metadata = _selective_record_metadata(
        selective_session,
        source_width=source_width,
        layer_count=layer_count,
    )
    records = tuple(value.summary for value in metadata)
    fit_records, heldout_records, split_report = _split_records(
        records,
        cache_key=cache_key,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        extension_batch_indices=extension_batch_indices,
    )
    fit_plan, fit_source_rows, fit_selected_rows = _global_row_selection_plan(
        fit_records,
        max_rows_per_block=max_rows_per_block,
    )
    heldout_plan, heldout_source_rows, heldout_selected_rows = (
        _global_row_selection_plan(
            heldout_records,
            max_rows_per_block=max_rows_per_block,
        )
    )
    load_selected_rows, release_selected_rows = _authenticated_row_api(
        selective_session
    )
    original_threads = torch.get_num_threads()
    with _NUMERICAL_LOCK:
        try:
            torch.set_num_threads(1)
            with torch.no_grad():
                fit_gram, fit_memory = _stream_snapshot_gram_split(
                    metadata,
                    fit_records,
                    fit_plan,
                    split=EvidenceSplit.FIT,
                    source_width=source_width,
                    load_selected_rows=load_selected_rows,
                    release_selected_rows=release_selected_rows,
                )
                heldout_gram, heldout_memory = _stream_snapshot_gram_split(
                    metadata,
                    heldout_records,
                    heldout_plan,
                    split=EvidenceSplit.HELDOUT,
                    source_width=source_width,
                    load_selected_rows=load_selected_rows,
                    release_selected_rows=release_selected_rows,
                )
        finally:
            torch.set_num_threads(original_threads)
    maximum_phase_buffer_bytes = max(
        fit_memory.maximum_phase_buffer_fp64_bytes,
        heldout_memory.maximum_phase_buffer_fp64_bytes,
    )
    maximum_source_payload_bytes = max(
        fit_memory.maximum_selected_source_payload_bytes,
        heldout_memory.maximum_selected_source_payload_bytes,
    )
    maximum_source_overhead_bytes = max(
        fit_memory.maximum_source_slice_or_hash_overhead_bytes,
        heldout_memory.maximum_source_slice_or_hash_overhead_bytes,
    )
    maximum_conversion_bytes = max(
        fit_memory.maximum_fp64_conversion_bytes,
        heldout_memory.maximum_fp64_conversion_bytes,
    )
    maximum_finiteness_scratch_bytes = max(
        fit_memory.maximum_fp64_finiteness_scratch_bytes,
        heldout_memory.maximum_fp64_finiteness_scratch_bytes,
    )
    maximum_block_bytes = max(
        fit_memory.maximum_block_fp64_bytes,
        heldout_memory.maximum_block_fp64_bytes,
    )
    block_owned_clone_bytes = maximum_block_bytes
    block_hash_clone_bytes = maximum_block_bytes
    block_hash_tobytes_bytes = maximum_block_bytes
    phase_and_source_read_bound = (
        maximum_phase_buffer_bytes
        + maximum_source_payload_bytes
        + maximum_source_overhead_bytes
    )
    phase_and_conversion_bound = (
        maximum_phase_buffer_bytes
        + maximum_source_payload_bytes
        + maximum_conversion_bytes
        + maximum_finiteness_scratch_bytes
    )
    phase_and_block_seal_bound = (
        maximum_phase_buffer_bytes
        + block_owned_clone_bytes
        + block_hash_clone_bytes
        + block_hash_tobytes_bytes
    )
    gram_matrix_bytes = source_width * source_width * 8
    maximum_concurrent_gram_matrix_count = 8
    gram_allocation_bound_bytes = (
        maximum_concurrent_gram_matrix_count * gram_matrix_bytes
    )
    python_plan_single_copy_overhead_bytes = _conservative_python_container_bytes(
        (fit_plan, heldout_plan)
    )
    python_plan_live_copy_count_bound = 2
    python_plan_overhead_bound_bytes = (
        python_plan_live_copy_count_bound
        * python_plan_single_copy_overhead_bytes
    )
    python_metadata_overhead_bound_bytes = (
        _streaming_metadata_overhead_bound_bytes(metadata)
    )
    python_report_single_copy_overhead_bytes = (
        _conservative_python_container_bytes(
            (
                fit_gram.report,
                heldout_gram.report,
                fit_gram.contribution_manifest.as_dict(),
                heldout_gram.contribution_manifest.as_dict(),
            )
        )
    )
    python_report_live_copy_count_bound = 4
    python_report_overhead_bound_bytes = (
        python_report_live_copy_count_bound
        * python_report_single_copy_overhead_bytes
    )
    known_python_overhead_bound_bytes = (
        python_plan_overhead_bound_bytes
        + python_metadata_overhead_bound_bytes
        + python_report_overhead_bound_bytes
    )
    maximum_row_path_bound_bytes = max(
        phase_and_source_read_bound,
        phase_and_conversion_bound,
        phase_and_block_seal_bound,
    )
    maximum_resident_snapshot_logical_bytes = (
        maximum_row_path_bound_bytes
        + gram_allocation_bound_bytes
        + known_python_overhead_bound_bytes
    )
    fp64_row_bytes = source_width * 8
    maximum_resident_fp64_snapshot_rows = (
        maximum_phase_buffer_bytes
        + block_owned_clone_bytes
        + block_hash_clone_bytes
        + block_hash_tobytes_bytes
    ) // fp64_row_bytes
    memory_bound_report = {
        "schema": _STREAMING_SNAPSHOT_MEMORY_BOUND_SCHEMA,
        "source_width": source_width,
        "residual_endpoint_count_per_layer": len(_ENDPOINTS),
        "residual_endpoints": [f"residual.{value}" for value in _ENDPOINTS],
        "maximum_phase_buffer_fp64_bytes": maximum_phase_buffer_bytes,
        "maximum_selected_source_payload_bytes": maximum_source_payload_bytes,
        "maximum_source_slice_or_hash_overhead_bytes": (
            maximum_source_overhead_bytes
        ),
        "maximum_fp64_conversion_bytes": maximum_conversion_bytes,
        "maximum_fp64_finiteness_scratch_bytes": (
            maximum_finiteness_scratch_bytes
        ),
        "maximum_block_fp64_bytes": maximum_block_bytes,
        "block_owned_clone_count": 1,
        "block_owned_clone_bytes": block_owned_clone_bytes,
        "block_hash_clone_count": 1,
        "block_hash_clone_bytes": block_hash_clone_bytes,
        "block_hash_tobytes_copy_count": 1,
        "block_hash_tobytes_copy_bytes": block_hash_tobytes_bytes,
        "phase_and_source_read_bound_bytes": phase_and_source_read_bound,
        "phase_and_conversion_bound_bytes": phase_and_conversion_bound,
        "phase_and_block_seal_bound_bytes": phase_and_block_seal_bound,
        "maximum_row_path_bound_bytes": maximum_row_path_bound_bytes,
        "gram_matrix_fp64_bytes": gram_matrix_bytes,
        "maximum_concurrent_gram_matrix_count": (
            maximum_concurrent_gram_matrix_count
        ),
        "gram_allocation_bound_bytes": gram_allocation_bound_bytes,
        "python_plan_single_copy_overhead_bytes": (
            python_plan_single_copy_overhead_bytes
        ),
        "python_plan_live_copy_count_bound": python_plan_live_copy_count_bound,
        "python_plan_overhead_bound_bytes": python_plan_overhead_bound_bytes,
        "python_metadata_overhead_bound_bytes": (
            python_metadata_overhead_bound_bytes
        ),
        "python_report_single_copy_overhead_bytes": (
            python_report_single_copy_overhead_bytes
        ),
        "python_report_live_copy_count_bound": (
            python_report_live_copy_count_bound
        ),
        "python_report_overhead_bound_bytes": python_report_overhead_bound_bytes,
        "known_python_overhead_bound_bytes": known_python_overhead_bound_bytes,
        "maximum_resident_fp64_snapshot_rows": (
            maximum_resident_fp64_snapshot_rows
        ),
        "maximum_resident_snapshot_logical_bytes": (
            maximum_resident_snapshot_logical_bytes
        ),
        "source_payload_included": True,
        "block_clone_hash_and_tobytes_copies_included": True,
        "precomputed_fit_gram_residency_during_heldout_included": True,
        "known_python_container_overhead_included": True,
        "python_container_aliases_deliberately_double_counted": True,
        "allocator_fragmentation_and_native_runtime_overhead_included": False,
        "safetensors_file_mapping_virtual_bytes_included": False,
        "bound_scope": "owned-logical-tensor-bytes-and-known-python-containers",
        "telemetry_excluded_from_basis_evidence_hashes": True,
    }
    return _StreamingSnapshotLoad(
        records=records,
        fit_records=fit_records,
        heldout_records=heldout_records,
        fit_snapshot_gram=fit_gram,
        heldout_snapshot_gram=heldout_gram,
        fit_block_count=int(fit_gram.report["block_count"]),
        heldout_block_count=int(heldout_gram.report["block_count"]),
        split_report=split_report,
        source_row_count=fit_source_rows + heldout_source_rows,
        selected_snapshot_row_count=fit_selected_rows + heldout_selected_rows,
        maximum_resident_fp64_snapshot_rows=(
            maximum_resident_fp64_snapshot_rows
        ),
        maximum_resident_snapshot_logical_bytes=(
            maximum_resident_snapshot_logical_bytes
        ),
        memory_bound_report=memory_bound_report,
    )


def _layer_index(operator_name: str) -> int:
    fields = operator_name.split(".")
    if len(fields) < 3 or fields[0] != "layer":
        raise Lfm2CircuitBalancedBasisError(
            f"Operator {operator_name!r} has no stable LFM2 layer binding."
        )
    try:
        return int(fields[1])
    except ValueError as exc:
        raise Lfm2CircuitBalancedBasisError(
            f"Operator {operator_name!r} has an invalid layer binding."
        ) from exc


@dataclass(frozen=True, slots=True)
class _VocabularyMarginDirectionSet:
    operator: ResidualFacingOperator
    gram: torch.Tensor
    effective_directions: torch.Tensor
    residual_rows: torch.Tensor
    cached_margins: torch.Tensor
    row_groups: torch.Tensor
    competitor_ranks: torch.Tensor
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        pair_count = int(self.cached_margins.numel())
        source_width = self.operator.source_width
        if (
            self.operator.role != _VOCABULARY_MARGIN_ROLE
            or tuple(self.gram.shape) != (source_width, source_width)
            or tuple(self.effective_directions.shape)
            != (pair_count, source_width)
            or tuple(self.residual_rows.shape) != (pair_count, source_width)
            or tuple(self.row_groups.shape) != (pair_count,)
            or tuple(self.competitor_ranks.shape) != (pair_count,)
            or pair_count <= 0
            or not isinstance(self.report, Mapping)
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Vocabulary-margin direction evidence is inconsistent."
            )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))


def lfm2_vocabulary_margin_vulnerability(
    margins: torch.Tensor,
    *,
    logit_values: torch.Tensor | None = None,
    margin_floor_fraction: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the unnormalized inverse-margin vulnerability of cached margins.

    This is the exact per-pair emphasis
    :func:`_build_vocabulary_margin_direction_set` applies to its top-1 versus
    competitor margin directions, lifted to a public, side-effect-free function
    so a body-layer fit can weight its rows by the same quantity the
    observability Gram already weights its directions by::

        margin_scale = median(positive margins)     (value scale if none)
        margin_floor = max(margin_scale * margin_floor_fraction,
                           eps * value_scale * 16)
        vulnerability = 1 / (margin + margin_floor)

    The caller normalizes.  Inside the balanced basis the normalization is
    hierarchical (per phase, then globally); a row-weighted ridge fit instead
    wants a unit-mean profile, which is why normalization is deliberately not
    performed here.
    """

    if (
        not isinstance(margins, torch.Tensor)
        or margins.layout != torch.strided
        or margins.device.type == "meta"
        or margins.numel() == 0
        or not torch.is_floating_point(margins)
        or torch.is_complex(margins)
        or not bool(torch.isfinite(margins).all().item())
        or bool((margins < 0.0).any().item())
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Vocabulary margins must be a nonempty finite non-negative real "
            "tensor."
        )
    fraction = _unit_interval(
        margin_floor_fraction,
        "margin_floor_fraction",
        include_zero=False,
        include_one=True,
    )
    table = margins.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if logit_values is None:
        value_scale = max(1.0, float(table.abs().max().item()))
    else:
        if (
            not isinstance(logit_values, torch.Tensor)
            or logit_values.numel() == 0
            or not torch.is_floating_point(logit_values)
            or not bool(torch.isfinite(logit_values).all().item())
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Vocabulary logit values must be a nonempty finite real tensor."
            )
        value_scale = max(1.0, float(logit_values.abs().max().item()))
    positive = table[table > 0.0]
    margin_scale = (
        float(torch.median(positive).item())
        if int(positive.numel()) > 0
        else value_scale
    )
    numerical_epsilon = torch.finfo(torch.float64).eps
    margin_floor = max(
        margin_scale * fraction,
        numerical_epsilon * value_scale * 16.0,
    )
    vulnerability = (1.0 / (table + margin_floor)).contiguous()
    report = {
        "schema": "neural-forge-lfm2-vocabulary-margin-vulnerability-v1",
        "margin_floor_fraction": fraction,
        "margin_floor": margin_floor,
        "margin_scale": margin_scale,
        "value_scale": value_scale,
        "logit_values_supplied": logit_values is not None,
        "margin_count": int(table.numel()),
        "minimum_cached_margin": float(table.min().item()),
        "median_cached_margin": float(torch.median(table).item()),
        "maximum_cached_margin": float(table.max().item()),
        "normalized": False,
    }
    return vulnerability, report


def _find_inventory_operator(
    inventory: Lfm2WeightCompleteInventory,
    *,
    names: frozenset[str],
    label: str,
) -> ResidualFacingOperator:
    matches = [
        value for value in inventory.operators if value.name in names
    ]
    if len(matches) != 1:
        raise Lfm2CircuitBalancedBasisError(
            f"Exactly one {label} operator is required."
        )
    return matches[0]


def _build_vocabulary_margin_direction_set(
    records: Sequence[_CacheRecord],
    *,
    split: EvidenceSplit,
    lexical_head: ResidualFacingOperator,
    final_norm: ResidualFacingOperator,
    layer_count: int,
    top_k: int,
    margin_floor_fraction: float,
) -> _VocabularyMarginDirectionSet:
    if not records:
        raise Lfm2CircuitBalancedBasisError(
            f"{split.value} vocabulary-margin records are empty."
        )
    head = lexical_head.tensor
    norm = final_norm.tensor
    source_width = lexical_head.source_width
    if (
        head.ndim != 2
        or tuple(head.shape)[1] != source_width
        or not final_norm.diagonal
        or tuple(norm.shape) != (source_width,)
    ):
        raise Lfm2CircuitBalancedBasisError(
            "The LFM2 head/final-norm inventory cannot form residual-facing "
            "vocabulary margins."
        )

    unit_blocks: list[torch.Tensor] = []
    effective_blocks: list[torch.Tensor] = []
    residual_blocks: list[torch.Tensor] = []
    margin_blocks: list[torch.Tensor] = []
    weight_blocks: list[torch.Tensor] = []
    group_blocks: list[torch.Tensor] = []
    rank_blocks: list[torch.Tensor] = []
    phase_reports: list[dict[str, Any]] = []
    skipped_zero_directions = 0
    row_group_offset = 0
    record_weight = 1.0 / len(records)
    numerical_epsilon = torch.finfo(torch.float64).eps

    for record in sorted(records, key=lambda value: value.index):
        if set(record.vocabulary_margins) != set(record.phases):
            raise Lfm2CircuitBalancedBasisError(
                f"Cache record {record.index} does not expose vocabulary "
                "margins for every authenticated phase."
            )
        phase_weight = record_weight / record.phase_count
        for phase in record.phases:
            evidence = record.vocabulary_margins[phase]
            if evidence.top_k != top_k:
                raise Lfm2CircuitBalancedBasisError(
                    "Vocabulary-margin top-k changed across cache phases."
                )
            token_ids = evidence.token_ids
            top_ids = token_ids[:, :1].expand(-1, top_k).reshape(-1)
            competitor_ids = token_ids[:, 1:].reshape(-1)
            raw_directions = (
                head.index_select(0, top_ids)
                - head.index_select(0, competitor_ids)
            )
            effective_directions = (
                raw_directions * norm.unsqueeze(0)
            ).contiguous()
            direction_norms = torch.linalg.vector_norm(
                effective_directions,
                dim=1,
            )
            direction_floor = (
                numerical_epsilon * max(1, source_width) * 16.0
            )
            valid = direction_norms > direction_floor
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                raise Lfm2CircuitBalancedBasisError(
                    f"All {phase!r} top-token embedding margin directions "
                    "are numerically zero."
                )
            skipped = int(valid.numel()) - valid_count
            skipped_zero_directions += skipped
            effective_directions = effective_directions[valid]
            direction_norms = direction_norms[valid]
            units = (
                effective_directions / direction_norms.unsqueeze(1)
            ).contiguous()

            margins = evidence.margins.reshape(-1)[valid]
            values_scale = max(
                1.0,
                float(evidence.logit_values.abs().max().item()),
            )
            positive_margins = margins[margins > 0.0]
            if int(positive_margins.numel()) > 0:
                margin_scale = float(
                    torch.median(positive_margins).item()
                )
            else:
                margin_scale = values_scale
            margin_floor = max(
                margin_scale * margin_floor_fraction,
                numerical_epsilon * values_scale * 16.0,
            )
            vulnerability = 1.0 / (margins + margin_floor)
            vulnerability = (
                vulnerability / vulnerability.sum() * phase_weight
            ).contiguous()

            residual = record.locations[
                (phase, layer_count - 1, "post_layer")
            ]
            repeated_residual = residual.repeat_interleave(top_k, dim=0)[
                valid
            ]
            local_groups = (
                torch.arange(
                    evidence.row_count,
                    dtype=torch.long,
                )
                .repeat_interleave(top_k)[valid]
                + row_group_offset
            )
            local_ranks = (
                torch.arange(1, top_k + 1, dtype=torch.long)
                .repeat(evidence.row_count)[valid]
            )

            unit_blocks.append(units)
            effective_blocks.append(effective_directions)
            residual_blocks.append(repeated_residual)
            margin_blocks.append(margins)
            weight_blocks.append(vulnerability)
            group_blocks.append(local_groups)
            rank_blocks.append(local_ranks)
            phase_reports.append(
                {
                    "record_index": record.index,
                    "record_id": record.record_id,
                    "phase": phase,
                    "logit_row_count": evidence.row_count,
                    "top_k_competitor_count": top_k,
                    "direction_count": valid_count,
                    "zero_direction_count_skipped": skipped,
                    "hierarchical_phase_weight": phase_weight,
                    "margin_floor": margin_floor,
                    "minimum_cached_margin": float(margins.min().item()),
                    "median_cached_margin": float(
                        torch.median(margins).item()
                    ),
                    "maximum_cached_margin": float(margins.max().item()),
                }
            )
            row_group_offset += evidence.row_count

    units = torch.cat(unit_blocks, dim=0).contiguous()
    effective = torch.cat(effective_blocks, dim=0).contiguous()
    residual_rows = torch.cat(residual_blocks, dim=0).contiguous()
    cached_margins = torch.cat(margin_blocks, dim=0).contiguous()
    weights = torch.cat(weight_blocks, dim=0).contiguous()
    row_groups = torch.cat(group_blocks, dim=0).contiguous()
    competitor_ranks = torch.cat(rank_blocks, dim=0).contiguous()
    weights = (weights / weights.sum()).contiguous()

    # Center the normalized directions, but append their weighted mean as a
    # separate operator row.  Consequently C + mu*mu.T equals the exact
    # weighted second moment: centering is numerically helpful and loses no
    # common decision direction.
    weighted_mean = (weights.unsqueeze(1) * units).sum(dim=0)
    centered = units - weighted_mean.unsqueeze(0)
    operator_rows = torch.cat(
        (
            centered * torch.sqrt(weights).unsqueeze(1),
            weighted_mean.unsqueeze(0),
        ),
        dim=0,
    ).contiguous()
    gram = (operator_rows.T @ operator_rows).contiguous()
    direct_gram = (
        units.T @ (units * weights.unsqueeze(1))
    ).contiguous()
    if not torch.allclose(
        gram,
        direct_gram,
        rtol=1e-11,
        atol=1e-12,
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Centered vocabulary-margin operator did not preserve its "
            "weighted second moment."
        )
    operator = ResidualFacingOperator(
        role=_VOCABULARY_MARGIN_ROLE,
        name=(
            f"cache.{split.value}.top1_vs_top{top_k}."
            "effective_embedding_margins"
        ),
        tensor=operator_rows,
        orientation=FeatureOrientation.INPUT,
        feature_axis=1,
    )
    report = {
        "schema": _VOCABULARY_MARGIN_SCHEMA,
        "split": split.value,
        "record_count": len(records),
        "record_ids_sha256": canonical_sha256(
            sorted(value.record_id for value in records)
        ),
        "phase_count": len(phase_reports),
        "logit_row_count": row_group_offset,
        "top_k": top_k,
        "direction_count": int(units.shape[0]),
        "zero_direction_count_skipped": skipped_zero_directions,
        "embedding_row_differences_used": True,
        "final_norm_diagonal_applied": True,
        "directions_l2_normalized": True,
        "vulnerability_weight": (
            "inverse(cached_top1_margin+deterministic_phase_floor)"
        ),
        "equal_record_weight": True,
        "equal_phase_weight_within_record": True,
        "weights_sum": float(weights.sum().item()),
        "margin_floor_fraction": margin_floor_fraction,
        "weighted_centering": True,
        "weighted_mean_reinserted": True,
        "centering_preserves_exact_second_moment": True,
        "operator": operator.as_dict(),
        "gram_trace_energy": float(torch.trace(gram).item()),
        "gram_sha256": _tensor_sha256(gram),
        "unit_directions_sha256": _tensor_sha256(units),
        "weights_sha256": _tensor_sha256(weights),
        "phases": phase_reports,
    }
    report["evidence_sha256"] = canonical_sha256(report)
    return _VocabularyMarginDirectionSet(
        operator=operator,
        gram=gram,
        effective_directions=effective,
        residual_rows=residual_rows,
        cached_margins=cached_margins,
        row_groups=row_groups,
        competitor_ranks=competitor_ranks,
        report=report,
    )


def _heldout_vocabulary_margin_diagnostics(
    basis: torch.Tensor,
    evidence: _VocabularyMarginDirectionSet,
) -> dict[str, Any]:
    numerator = float(torch.trace(basis.T @ evidence.gram @ basis).item())
    denominator = float(torch.trace(evidence.gram).item())
    retained = numerator / max(denominator, torch.finfo(torch.float64).tiny)

    original_margins = (
        evidence.residual_rows * evidence.effective_directions
    ).sum(dim=1)
    projected_margins = (
        (evidence.residual_rows @ basis)
        * (evidence.effective_directions @ basis)
    ).sum(dim=1)
    scale = torch.maximum(
        torch.ones_like(original_margins),
        torch.maximum(
            original_margins.abs(),
            projected_margins.abs(),
        ),
    )
    tolerance = (
        torch.finfo(torch.float64).eps
        * int(basis.shape[0])
        * 64.0
        * scale
    )

    def signs(values: torch.Tensor) -> torch.Tensor:
        return torch.where(
            values > tolerance,
            torch.ones_like(values, dtype=torch.int8),
            torch.where(
                values < -tolerance,
                -torch.ones_like(values, dtype=torch.int8),
                torch.zeros_like(values, dtype=torch.int8),
            ),
        )

    cached_signs = torch.where(
        evidence.cached_margins > tolerance,
        torch.ones_like(evidence.cached_margins, dtype=torch.int8),
        torch.zeros_like(evidence.cached_margins, dtype=torch.int8),
    )
    original_signs = signs(original_margins)
    projected_signs = signs(projected_margins)
    projected_nonnegative = projected_margins >= -tolerance
    rank_one = evidence.competitor_ranks == 1
    group_success = []
    for group in torch.unique(evidence.row_groups, sorted=True):
        group_success.append(
            bool(projected_nonnegative[evidence.row_groups == group].all())
        )

    diagnostics = {
        "schema": _VOCABULARY_MARGIN_SCHEMA,
        "selection_split": False,
        "used_for_basis_selection": False,
        "retained_top_k_margin_direction_energy_fraction": retained,
        "pair_count": int(evidence.cached_margins.numel()),
        "logit_row_count": len(group_success),
        "cached_top1_margin_nonnegative_fraction": float(
            (evidence.cached_margins >= 0.0)
            .to(dtype=torch.float64)
            .mean()
            .item()
        ),
        "closed_form_unprojected_vs_cached_sign_agreement_fraction": float(
            (original_signs == cached_signs)
            .to(dtype=torch.float64)
            .mean()
            .item()
        ),
        "projected_vs_cached_sign_agreement_fraction": float(
            (projected_signs == cached_signs)
            .to(dtype=torch.float64)
            .mean()
            .item()
        ),
        "projected_top1_vs_top_competitor_nonnegative_fraction": float(
            projected_nonnegative[rank_one]
            .to(dtype=torch.float64)
            .mean()
            .item()
        ),
        "projected_top1_vs_all_top_k_nonnegative_fraction": float(
            sum(group_success) / len(group_success)
        ),
        "projected_margin_minimum": float(projected_margins.min().item()),
        "projected_margin_median": float(
            torch.median(projected_margins).item()
        ),
        "projected_margin_maximum": float(projected_margins.max().item()),
        "projection_diagnostics_require_parent_execution": False,
        "source": (
            "sealed_logits-plus-final-layer-residuals-plus-head-and-final-norm"
        ),
        "heldout_evidence_sha256": evidence.report["evidence_sha256"],
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return diagnostics


@dataclass(frozen=True, slots=True)
class Lfm2VocabularyMarginTransactionGate:
    """Reusable held-out vocabulary-boundary gate with sealed parent evidence."""

    vocabulary_size: int
    top_k: int
    _phase_tables: Mapping[
        tuple[str, str], _CachedVocabularyMarginPhase
    ]
    _projection_evidence: _VocabularyMarginDirectionSet
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        vocabulary_size = _positive_int(
            self.vocabulary_size,
            "vocabulary_size",
        )
        top_k = _positive_int(self.top_k, "top_k")
        tables = dict(self._phase_tables)
        if (
            top_k >= vocabulary_size
            or not tables
            or any(
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(value, _CachedVocabularyMarginPhase)
                or value.top_k != top_k
                for key, value in tables.items()
            )
            or not isinstance(
                self._projection_evidence,
                _VocabularyMarginDirectionSet,
            )
            or not isinstance(self.report, Mapping)
        ):
            raise Lfm2CircuitBalancedBasisError(
                "Vocabulary-margin transaction gate evidence is invalid."
            )
        object.__setattr__(self, "vocabulary_size", vocabulary_size)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(
            self,
            "_phase_tables",
            MappingProxyType(tables),
        )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def heldout_record_ids(self) -> tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._phase_tables}))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)

    @staticmethod
    def _threshold(value: Any, name: str) -> float:
        return _unit_interval(
            value,
            name,
            include_zero=True,
            include_one=True,
        )

    def evaluate_projection(
        self,
        basis: torch.Tensor,
        *,
        minimum_retained_energy_fraction: float,
        minimum_sign_agreement_fraction: float,
        minimum_top_competitor_nonnegative_fraction: float,
        minimum_all_top_k_nonnegative_fraction: float,
    ) -> dict[str, Any]:
        """Gate a proposed source-to-child subspace without model execution."""

        if (
            not isinstance(basis, torch.Tensor)
            or basis.layout != torch.strided
            or basis.device.type == "meta"
            or basis.ndim != 2
            or int(basis.shape[0])
            != self._projection_evidence.operator.source_width
            or int(basis.shape[1]) <= 0
            or not torch.is_floating_point(basis)
            or torch.is_complex(basis)
            or not bool(torch.isfinite(basis).all().item())
        ):
            raise Lfm2CircuitBalancedBasisError(
                "projection basis must be a finite dense source-by-target "
                "matrix."
            )
        basis64 = basis.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        thresholds = {
            "retained_top_k_margin_direction_energy_fraction": self._threshold(
                minimum_retained_energy_fraction,
                "minimum_retained_energy_fraction",
            ),
            "projected_vs_cached_sign_agreement_fraction": self._threshold(
                minimum_sign_agreement_fraction,
                "minimum_sign_agreement_fraction",
            ),
            "projected_top1_vs_top_competitor_nonnegative_fraction": (
                self._threshold(
                    minimum_top_competitor_nonnegative_fraction,
                    "minimum_top_competitor_nonnegative_fraction",
                )
            ),
            "projected_top1_vs_all_top_k_nonnegative_fraction": (
                self._threshold(
                    minimum_all_top_k_nonnegative_fraction,
                    "minimum_all_top_k_nonnegative_fraction",
                )
            ),
        }
        metrics = _heldout_vocabulary_margin_diagnostics(
            basis64,
            self._projection_evidence,
        )
        checks = {
            name: float(metrics[name]) + 1e-15 >= minimum
            for name, minimum in thresholds.items()
        }
        result = {
            "schema": _VOCABULARY_MARGIN_SCHEMA,
            "gate_kind": "heldout_projection_top1_vs_top_k",
            "metrics": metrics,
            "minimums": thresholds,
            "checks": checks,
            "gate_passed": all(checks.values()),
            "heldout_used_for_selection": False,
            "parent_forward_calls": 0,
            "candidate_forward_calls": 0,
            "training_steps": 0,
        }
        result["gate_sha256"] = canonical_sha256(result)
        return result

    def evaluate_candidate_replay(
        self,
        candidate_replay: Any,
        *,
        minimum_sign_agreement_fraction: float,
        minimum_top_competitor_nonnegative_fraction: float,
        minimum_all_top_k_nonnegative_fraction: float,
        minimum_exact_top1_agreement_fraction: float,
    ) -> dict[str, Any]:
        """Gate already-captured child logits at held-out parent boundaries."""

        iterator = getattr(candidate_replay, "iter_batches", None)
        if not callable(iterator):
            raise Lfm2CircuitBalancedBasisError(
                "candidate_replay must expose iter_batches()."
            )
        batches: dict[str, Any] = {}
        for batch in iterator():
            record_id = _sha256(
                getattr(batch, "input_sha256", None),
                "candidate replay input_sha256",
            )
            if record_id in batches:
                raise Lfm2CircuitBalancedBasisError(
                    "Candidate replay repeats a held-out record ID."
                )
            batches[record_id] = batch
        missing_records = sorted(
            set(self.heldout_record_ids).difference(batches)
        )
        if missing_records:
            raise Lfm2CircuitBalancedBasisError(
                "Candidate replay omits held-out vocabulary-margin records: "
                + ", ".join(missing_records)
                + "."
            )

        candidate_margin_blocks: list[torch.Tensor] = []
        cached_margin_blocks: list[torch.Tensor] = []
        exact_top1_blocks: list[torch.Tensor] = []
        selected_logit_blocks: list[torch.Tensor] = []
        phase_reports: list[dict[str, Any]] = []
        for (record_id, phase), parent in sorted(
            self._phase_tables.items(),
            key=lambda value: (value[0][0], _phase_key(value[0][1])),
        ):
            tensors = getattr(batches[record_id], "tensors", None)
            logits_name = f"{phase}.logits"
            if not isinstance(tensors, Mapping) or logits_name not in tensors:
                raise Lfm2CircuitBalancedBasisError(
                    f"Candidate replay omits held-out logits {logits_name!r}."
                )
            logits = tensors[logits_name]
            if (
                not isinstance(logits, torch.Tensor)
                or logits.layout != torch.strided
                or logits.device.type == "meta"
                or logits.ndim < 2
                or int(logits.shape[-1]) != self.vocabulary_size
                or not torch.is_floating_point(logits)
                or torch.is_complex(logits)
                or logits.numel() == 0
            ):
                raise Lfm2CircuitBalancedBasisError(
                    f"Candidate {logits_name!r} is not a compatible full "
                    "vocabulary logit tensor."
                )
            rows = (
                logits.detach()
                .to(device="cpu", dtype=torch.float64)
                .reshape(-1, self.vocabulary_size)
                .contiguous()
            )
            if (
                int(rows.shape[0]) != parent.row_count
                or not bool(torch.isfinite(rows).all().item())
            ):
                raise Lfm2CircuitBalancedBasisError(
                    f"Candidate {logits_name!r} rows do not align with the "
                    "sealed held-out parent."
                )
            selected = rows.gather(1, parent.token_ids)
            candidate_margins = (
                selected[:, :1] - selected[:, 1:]
            ).contiguous()
            candidate_top1 = _deterministic_top_tokens(
                logits,
                vocabulary_size=self.vocabulary_size,
                top_k=1,
                phase=phase,
            ).token_ids[:, 0]
            exact_top1 = candidate_top1 == parent.token_ids[:, 0]
            candidate_margin_blocks.append(candidate_margins)
            cached_margin_blocks.append(parent.margins)
            exact_top1_blocks.append(exact_top1)
            selected_logit_blocks.append(selected)
            phase_reports.append(
                {
                    "record_id": record_id,
                    "phase": phase,
                    "row_count": parent.row_count,
                    "candidate_top1_exact_count": int(
                        exact_top1.sum().item()
                    ),
                    "candidate_top1_boundary_count": int(
                        (candidate_margins >= 0.0).sum().item()
                    ),
                }
            )

        candidate_margins = torch.cat(
            candidate_margin_blocks,
            dim=0,
        )
        cached_margins = torch.cat(cached_margin_blocks, dim=0)
        exact_top1 = torch.cat(exact_top1_blocks, dim=0)
        selected_logits = torch.cat(selected_logit_blocks, dim=0)
        scale = torch.maximum(
            torch.ones_like(candidate_margins),
            torch.maximum(candidate_margins.abs(), cached_margins.abs()),
        )
        tolerance = (
            torch.finfo(torch.float64).eps
            * self.vocabulary_size
            * 16.0
            * scale
        )
        candidate_signs = torch.where(
            candidate_margins > tolerance,
            torch.ones_like(candidate_margins, dtype=torch.int8),
            torch.where(
                candidate_margins < -tolerance,
                -torch.ones_like(candidate_margins, dtype=torch.int8),
                torch.zeros_like(candidate_margins, dtype=torch.int8),
            ),
        )
        cached_signs = torch.where(
            cached_margins > tolerance,
            torch.ones_like(cached_margins, dtype=torch.int8),
            torch.zeros_like(cached_margins, dtype=torch.int8),
        )
        nonnegative = candidate_margins >= -tolerance
        metrics = {
            "candidate_vs_cached_sign_agreement_fraction": float(
                (candidate_signs == cached_signs)
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
            "candidate_top1_vs_top_competitor_nonnegative_fraction": float(
                nonnegative[:, 0]
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
            "candidate_top1_vs_all_top_k_nonnegative_fraction": float(
                nonnegative.all(dim=1)
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
            "candidate_exact_parent_top1_agreement_fraction": float(
                exact_top1.to(dtype=torch.float64).mean().item()
            ),
            "candidate_margin_minimum": float(
                candidate_margins.min().item()
            ),
            "candidate_margin_median": float(
                torch.median(candidate_margins).item()
            ),
            "candidate_margin_maximum": float(
                candidate_margins.max().item()
            ),
        }
        thresholds = {
            "candidate_vs_cached_sign_agreement_fraction": self._threshold(
                minimum_sign_agreement_fraction,
                "minimum_sign_agreement_fraction",
            ),
            "candidate_top1_vs_top_competitor_nonnegative_fraction": (
                self._threshold(
                    minimum_top_competitor_nonnegative_fraction,
                    "minimum_top_competitor_nonnegative_fraction",
                )
            ),
            "candidate_top1_vs_all_top_k_nonnegative_fraction": (
                self._threshold(
                    minimum_all_top_k_nonnegative_fraction,
                    "minimum_all_top_k_nonnegative_fraction",
                )
            ),
            "candidate_exact_parent_top1_agreement_fraction": self._threshold(
                minimum_exact_top1_agreement_fraction,
                "minimum_exact_top1_agreement_fraction",
            ),
        }
        checks = {
            name: metrics[name] + 1e-15 >= minimum
            for name, minimum in thresholds.items()
        }
        result = {
            "schema": _VOCABULARY_MARGIN_SCHEMA,
            "gate_kind": "heldout_candidate_replay_top1_vs_top_k",
            "heldout_record_count": len(self.heldout_record_ids),
            "phase_count": len(phase_reports),
            "top_k": self.top_k,
            "metrics": metrics,
            "minimums": thresholds,
            "checks": checks,
            "gate_passed": all(checks.values()),
            "phases": phase_reports,
            "candidate_selected_logits_sha256": _tensor_sha256(
                selected_logits
            ),
            "heldout_used_for_selection": False,
            "parent_forward_calls": 0,
            "candidate_forward_calls": 0,
            "trace_capture_calls": 0,
            "training_steps": 0,
        }
        result["gate_sha256"] = canonical_sha256(result)
        return result


def _build_vocabulary_margin_transaction_gate(
    records: Sequence[_CacheRecord],
    *,
    vocabulary_size: int,
    top_k: int,
    projection_evidence: _VocabularyMarginDirectionSet,
) -> Lfm2VocabularyMarginTransactionGate:
    tables = {
        (record.record_id, phase): record.vocabulary_margins[phase]
        for record in records
        for phase in record.phases
    }
    table_manifest = [
        {
            "record_id": record_id,
            "phase": phase,
            "token_ids_sha256": _tensor_sha256(value.token_ids),
            "logit_values_sha256": _tensor_sha256(value.logit_values),
        }
        for (record_id, phase), value in sorted(
            tables.items(),
            key=lambda item: (item[0][0], _phase_key(item[0][1])),
        )
    ]
    report = {
        "schema": _VOCABULARY_MARGIN_SCHEMA,
        "gate_kind": "reusable-heldout-top1-vs-top-k",
        "vocabulary_size": vocabulary_size,
        "top_k": top_k,
        "heldout_record_count": len(records),
        "heldout_record_ids_sha256": canonical_sha256(
            sorted(value.record_id for value in records)
        ),
        "phase_count": len(tables),
        "phase_table_manifest_sha256": canonical_sha256(table_manifest),
        "supports_projection_gate": True,
        "supports_candidate_replay_gate": True,
        "parent_forward_calls_per_evaluation": 0,
        "candidate_forward_calls_per_evaluation": 0,
        "training_steps_per_evaluation": 0,
    }
    report["gate_evidence_sha256"] = canonical_sha256(report)
    return Lfm2VocabularyMarginTransactionGate(
        vocabulary_size=vocabulary_size,
        top_k=top_k,
        _phase_tables=tables,
        _projection_evidence=projection_evidence,
        report=report,
    )


def _build_observability(
    inventory: Lfm2WeightCompleteInventory,
    graph: ModelCircuitGraph,
    *,
    vocabulary_margin_operator: ResidualFacingOperator | None = None,
) -> tuple[DownstreamObservability, ...]:
    nodes = {value.node_id: value for value in graph.nodes}
    result: list[DownstreamObservability] = []
    role_bindings = {
        "attention_output_span": ("operator", CircuitKind.ATTENTION.value),
        "short_conv_output_span": (
            "operator",
            CircuitKind.SHORT_CONV.value,
        ),
        "swiglu_output_span": ("mlp", CircuitKind.GATED_MLP.value),
    }
    for operator in inventory.operators:
        binding = role_bindings.get(operator.role)
        if binding is None:
            continue
        suffix, expected_kind = binding
        layer_index = _layer_index(operator.name)
        node_id = f"layer.{layer_index:03d}.{suffix}"
        node = nodes.get(node_id)
        if node is None or node.kind != expected_kind:
            raise Lfm2CircuitBalancedBasisError(
                f"Weight operator {operator.name!r} cannot be bound to "
                f"{node_id!r}/{expected_kind!r} in circuit IR."
            )
        result.append(
            DownstreamObservability(
                evidence_id=f"writer.{operator.name}",
                node_id=node_id,
                circuit_kind=expected_kind,
                operator=operator,
            )
        )

    lexical_candidates = [
        value
        for value in inventory.operators
        if value.name
        in {
            "lexical.tied_embedding_head.weight",
            "lexical.lm_head.weight",
        }
    ]
    if len(lexical_candidates) != 1:
        raise Lfm2CircuitBalancedBasisError(
            "Exactly one LFM2 output-head weight must supply logits "
            "observability."
        )
    head_node = nodes.get("output.lm_head")
    if (
        head_node is None
        or head_node.kind != CircuitKind.OUTPUT_HEAD.value
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Circuit IR does not expose the LFM2 output head."
        )
    result.append(
        DownstreamObservability(
            evidence_id="interface.text.logits.output",
            node_id="output.lm_head",
            circuit_kind=CircuitKind.OUTPUT_HEAD,
            operator=lexical_candidates[0],
            interface_ids=("text.logits.output",),
        )
    )
    if vocabulary_margin_operator is not None:
        if vocabulary_margin_operator.role != _VOCABULARY_MARGIN_ROLE:
            raise Lfm2CircuitBalancedBasisError(
                "Vocabulary-margin operator has the wrong observability role."
            )
        result.append(
            DownstreamObservability(
                evidence_id="interface.text.logits.vocabulary_margin",
                node_id="output.lm_head",
                circuit_kind=CircuitKind.OUTPUT_HEAD,
                operator=vocabulary_margin_operator,
                interface_ids=("text.logits.output",),
            )
        )
    if not result:
        raise Lfm2CircuitBalancedBasisError(
            "No LFM2 downstream observability operators were discovered."
        )
    return tuple(
        sorted(
            result,
            key=lambda value: (value.node_id, value.evidence_id),
        )
    )


@dataclass(frozen=True, slots=True)
class Lfm2CircuitBalancedBasisResult:
    """Balanced structural map bound to its LFM2 cache and complete weights."""

    inventory: Lfm2WeightCompleteInventory
    circuit_graph: ModelCircuitGraph
    basis_result: BalancedCircuitBasisResult
    report: Mapping[str, Any]
    vocabulary_margin_gate: Any | None = None
    reuse_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.inventory, Lfm2WeightCompleteInventory)
            or not isinstance(self.circuit_graph, ModelCircuitGraph)
            or not isinstance(self.basis_result, BalancedCircuitBasisResult)
            or self.inventory.source_width != self.basis_result.source_width
            or not isinstance(self.report, Mapping)
            or (
                self.reuse_provenance is not None
                and not isinstance(self.reuse_provenance, Mapping)
            )
            or (
                self.vocabulary_margin_gate is not None
                and not (
                    hasattr(self.vocabulary_margin_gate, "heldout_record_ids")
                    and callable(
                        getattr(
                            self.vocabulary_margin_gate,
                            "evaluate_projection",
                            None,
                        )
                    )
                    and callable(
                        getattr(
                            self.vocabulary_margin_gate,
                            "evaluate_candidate_replay",
                            None,
                        )
                    )
                )
            )
        ):
            raise Lfm2CircuitBalancedBasisError(
                "The LFM2 circuit-balanced result is inconsistent."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )
        if self.reuse_provenance is not None:
            object.__setattr__(
                self,
                "reuse_provenance",
                MappingProxyType(dict(self.reuse_provenance)),
            )

    @property
    def basis(self) -> torch.Tensor:
        return self.basis_result.basis

    @property
    def balanced_encoder(self) -> torch.Tensor:
        return self.basis_result.balanced_encoder

    @property
    def balanced_decoder(self) -> torch.Tensor:
        return self.basis_result.balanced_decoder

    @property
    def full_weight_gram(self) -> torch.Tensor:
        return self.basis_result.full_weight_gram

    @property
    def controllability_gram(self) -> torch.Tensor:
        return self.basis_result.controllability_gram

    @property
    def observability_gram(self) -> torch.Tensor:
        return self.basis_result.observability_gram

    @property
    def heldout_gram(self) -> torch.Tensor:
        return self.basis_result.heldout_gram

    @property
    def source_width(self) -> int:
        return self.basis_result.source_width

    @property
    def target_width(self) -> int:
        return self.basis_result.target_width

    @property
    def map_sha256(self) -> str:
        return self.basis_result.map_sha256

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.encode(rows)

    def decode(self, rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.decode(rows)

    def balanced_encode(self, rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.balanced_encode(rows)

    def balanced_decode(self, rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.balanced_decode(rows)

    def apply(self, rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.apply(rows)

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result(rows)

    def fold_consumer_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self.basis_result.fold_consumer_weight(weight)

    def fold_producer_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self.basis_result.fold_producer_weight(weight)

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: torch.dtype | None = None,
        device: Any = "cpu",
    ) -> torch.Tensor:
        return self.basis_result.materialize_dense(
            max_elements=max_elements,
            dtype=dtype,
            device=device,
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)

    def evaluate_vocabulary_margin_projection_gate(
        self,
        basis: torch.Tensor,
        *,
        minimum_retained_energy_fraction: float,
        minimum_sign_agreement_fraction: float,
        minimum_top_competitor_nonnegative_fraction: float,
        minimum_all_top_k_nonnegative_fraction: float,
    ) -> dict[str, Any]:
        if self.vocabulary_margin_gate is None:
            raise Lfm2CircuitBalancedBasisError(
                "This basis was built without vocabulary-margin evidence."
            )
        return self.vocabulary_margin_gate.evaluate_projection(
            basis,
            minimum_retained_energy_fraction=(
                minimum_retained_energy_fraction
            ),
            minimum_sign_agreement_fraction=(
                minimum_sign_agreement_fraction
            ),
            minimum_top_competitor_nonnegative_fraction=(
                minimum_top_competitor_nonnegative_fraction
            ),
            minimum_all_top_k_nonnegative_fraction=(
                minimum_all_top_k_nonnegative_fraction
            ),
        )

    def evaluate_vocabulary_margin_candidate_gate(
        self,
        candidate_replay: Any,
        *,
        minimum_sign_agreement_fraction: float,
        minimum_top_competitor_nonnegative_fraction: float,
        minimum_all_top_k_nonnegative_fraction: float,
        minimum_exact_top1_agreement_fraction: float,
    ) -> dict[str, Any]:
        if self.vocabulary_margin_gate is None:
            raise Lfm2CircuitBalancedBasisError(
                "This basis was built without vocabulary-margin evidence."
            )
        return self.vocabulary_margin_gate.evaluate_candidate_replay(
            candidate_replay,
            minimum_sign_agreement_fraction=(
                minimum_sign_agreement_fraction
            ),
            minimum_top_competitor_nonnegative_fraction=(
                minimum_top_competitor_nonnegative_fraction
            ),
            minimum_all_top_k_nonnegative_fraction=(
                minimum_all_top_k_nonnegative_fraction
            ),
            minimum_exact_top1_agreement_fraction=(
                minimum_exact_top1_agreement_fraction
            ),
        )


def build_lfm2_circuit_balanced_basis(
    parent_model: torch.nn.Module,
    trace_cache: Any,
    target_width: int,
    *,
    validation_fraction: float = 0.25,
    split_seed: str = "lfm2-circuit-balanced-v1",
    max_rows_per_block: int = 512,
    structural_controllability_weight: float = 0.35,
    cached_controllability_weight: float = 0.65,
    gram_regularization: float = 1e-10,
    rank_tolerance: float = 1e-12,
    max_condition_number: float = 1e12,
    minimum_heldout_retained_energy_fraction: float = 0.0,
    circuit_graph: ModelCircuitGraph | None = None,
    vocabulary_margin_observability: bool = False,
    vocabulary_margin_top_k: int = 8,
    vocabulary_margin_floor_fraction: float = 0.05,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
    streaming_vocabulary_margin_v2: bool = False,
    streaming_vocabulary_margin_pair_chunk_size: int = 256,
    target_tokenizer_vocabulary_margin_binding: Any | None = None,
) -> Lfm2CircuitBalancedBasisResult:
    """Build an LFM2 residual map from weights, sealed cache, and circuit IR."""

    if not isinstance(parent_model, torch.nn.Module):
        raise Lfm2CircuitBalancedBasisError(
            "parent_model must be a torch.nn.Module."
        )
    target_width = _positive_int(target_width, "target_width")
    validation_fraction = _unit_interval(
        validation_fraction,
        "validation_fraction",
        include_zero=False,
        include_one=False,
    )
    split_seed = _nonempty_text(split_seed, "split_seed")
    max_rows_per_block = _positive_int(
        max_rows_per_block,
        "max_rows_per_block",
    )
    minimum_heldout = _unit_interval(
        minimum_heldout_retained_energy_fraction,
        "minimum_heldout_retained_energy_fraction",
        include_zero=True,
        include_one=True,
    )
    if not isinstance(vocabulary_margin_observability, bool):
        raise Lfm2CircuitBalancedBasisError(
            "vocabulary_margin_observability must be boolean."
        )
    vocabulary_margin_top_k = _positive_int(
        vocabulary_margin_top_k,
        "vocabulary_margin_top_k",
    )
    vocabulary_margin_floor_fraction = _unit_interval(
        vocabulary_margin_floor_fraction,
        "vocabulary_margin_floor_fraction",
        include_zero=False,
        include_one=True,
    )
    if not isinstance(streaming_vocabulary_margin_v2, bool):
        raise Lfm2CircuitBalancedBasisError(
            "streaming_vocabulary_margin_v2 must be boolean."
        )
    streaming_vocabulary_margin_pair_chunk_size = _positive_int(
        streaming_vocabulary_margin_pair_chunk_size,
        "streaming_vocabulary_margin_pair_chunk_size",
    )
    if streaming_vocabulary_margin_v2 and not vocabulary_margin_observability:
        raise Lfm2CircuitBalancedBasisError(
            "streaming_vocabulary_margin_v2 requires "
            "vocabulary_margin_observability=True."
        )
    if streaming_vocabulary_margin_v2 and selective_session is None:
        raise Lfm2CircuitBalancedBasisError(
            "streaming_vocabulary_margin_v2 requires a selective authenticated "
            "cache session."
        )
    if streaming_vocabulary_margin_v2 and not isinstance(
        selective_session,
        (
            Lfm2CircuitTraceSelectiveAuthenticatedSession,
            Lfm2MergedSealedEvidenceSession,
        ),
    ):
        # Fail-closed dispatch, named where the configuration is composed: the
        # streaming margin adapter authenticates one sealed cache's manifest
        # digest, or every member digest of one merged sealed evidence set, and
        # nothing else.  Both names are read from the module namespace for the
        # same substitution reason the session check below states.
        raise Lfm2CircuitBalancedBasisError(
            "The streaming vocabulary-margin adapter authenticates one sealed "
            "cache session or one merged sealed evidence session."
        )
    if (
        target_tokenizer_vocabulary_margin_binding is not None
        and not streaming_vocabulary_margin_v2
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Target tokenizer vocabulary-margin binding requires streaming v2."
        )
    if selective_session is None:
        verify = getattr(trace_cache, "verify", None)
        if not callable(verify):
            raise Lfm2CircuitBalancedBasisError("trace_cache must expose verify().")
        verified = verify()
        if verified is not None and verified is not trace_cache:
            raise Lfm2CircuitBalancedBasisError(
                "trace_cache.verify() returned a different cache handle."
            )
    else:
        # One sealed cache session, or one merged session over a base plus
        # on-policy extensions; both authenticate every byte they hand back
        # through the member that sealed it.  The two names are read from the
        # module namespace rather than a prebuilt tuple so the accepted session
        # type stays substitutable exactly as it already was.
        if not isinstance(
            selective_session,
            (
                Lfm2CircuitTraceSelectiveAuthenticatedSession,
                Lfm2MergedSealedEvidenceSession,
            ),
        ):
            raise TypeError(
                "selective_session must be an "
                "Lfm2CircuitTraceSelectiveAuthenticatedSession, one merged "
                "sealed evidence session, or None."
            )
        selective_session.verify()
        if (
            selective_session.path != getattr(trace_cache, "path", None)
            or selective_session.cache_key != getattr(trace_cache, "cache_key", None)
            or selective_session.batch_count
            != getattr(trace_cache, "batch_count", None)
            or selective_session.manifest != getattr(trace_cache, "manifest", None)
        ):
            raise Lfm2CircuitBalancedBasisError(
                "selective_session is not bound to trace_cache."
            )
    cache_key = _sha256(
        getattr(trace_cache, "cache_key", None),
        "trace_cache.cache_key",
    )
    corpus_fingerprint = _sha256(
        getattr(trace_cache, "corpus_fingerprint", None),
        "trace_cache.corpus_fingerprint",
    )
    parent_fingerprint = fingerprint_lfm2_parent(parent_model)
    cached_parent = _sha256(
        getattr(trace_cache, "parent_fingerprint", None),
        "trace_cache.parent_fingerprint",
    )
    if cached_parent != parent_fingerprint.identity_sha256:
        raise Lfm2CircuitBalancedBasisError(
            "The sealed trace cache is not bound to the supplied parent."
        )

    inventory = inventory_lfm2_weight_complete_operators(parent_model)
    config = getattr(parent_model, "config", None)
    layer_count = _positive_int(
        getattr(config, "num_hidden_layers", None),
        "config.num_hidden_layers",
    )
    vocabulary_size = _positive_int(
        getattr(config, "vocab_size", None),
        "config.vocab_size",
    )
    if (
        vocabulary_margin_observability
        and vocabulary_margin_top_k >= vocabulary_size
    ):
        raise Lfm2CircuitBalancedBasisError(
            "vocabulary_margin_top_k must be smaller than config.vocab_size."
        )
    graph = (
        inspect_model_circuits(
            parent_model,
            model_name=(
                "lfm2.circuit-balanced."
                + parent_fingerprint.identity_sha256[:16]
            ),
        )
        if circuit_graph is None
        else circuit_graph
    )
    if not isinstance(graph, ModelCircuitGraph) or graph.family != "lfm2":
        raise Lfm2CircuitBalancedBasisError(
            "circuit_graph must be the canonical LFM2 ModelCircuitGraph."
        )
    # One set both handles agree on, resolved once: every fit/held-out split
    # this solve makes has to know which batches a child actually visited.
    extension_batch_indices = _resolved_extension_batch_indices(
        trace_cache,
        selective_session,
    )
    if selective_session is not None and (
        not vocabulary_margin_observability or streaming_vocabulary_margin_v2
    ):
        streaming = _stream_selective_cache_snapshot_grams(
            selective_session,
            source_width=inventory.source_width,
            layer_count=layer_count,
            cache_key=cache_key,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
            max_rows_per_block=max_rows_per_block,
            extension_batch_indices=extension_batch_indices,
        )
        records = streaming.records
        fit_records = streaming.fit_records
        heldout_records = streaming.heldout_records
        split_report = dict(streaming.split_report)
        fit_snapshots: tuple[ResidualSnapshotBlock, ...] = ()
        heldout_snapshots: tuple[ResidualSnapshotBlock, ...] = ()
        precomputed_fit_snapshot_gram = streaming.fit_snapshot_gram
        precomputed_heldout_snapshot_gram = streaming.heldout_snapshot_gram
        fit_snapshot_block_count = streaming.fit_block_count
        heldout_snapshot_block_count = streaming.heldout_block_count
    else:
        records = _load_cache_records(
            trace_cache,
            source_width=inventory.source_width,
            layer_count=layer_count,
            vocabulary_size=vocabulary_size,
            vocabulary_margin_top_k=(
                vocabulary_margin_top_k
                if vocabulary_margin_observability
                else None
            ),
            selective_session=selective_session,
        )
        fit_records, heldout_records, split_report = _split_records(
            records,
            cache_key=cache_key,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
            extension_batch_indices=extension_batch_indices,
        )
        fit_snapshots = _snapshot_blocks(
            fit_records,
            split=EvidenceSplit.FIT,
            max_rows_per_block=max_rows_per_block,
        )
        heldout_snapshots = _snapshot_blocks(
            heldout_records,
            split=EvidenceSplit.HELDOUT,
            max_rows_per_block=max_rows_per_block,
        )
        precomputed_fit_snapshot_gram = None
        precomputed_heldout_snapshot_gram = None
        fit_snapshot_block_count = len(fit_snapshots)
        heldout_snapshot_block_count = len(heldout_snapshots)
    if selective_session is not None:
        selective_session.verify()
    declared_batch_count = getattr(trace_cache, "batch_count", len(records))
    if (
        isinstance(declared_batch_count, bool)
        or not isinstance(declared_batch_count, int)
        or declared_batch_count != len(records)
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Verified cache batch_count disagrees with iter_batches()."
        )
    fit_expected_block_count = (
        max(value.phase_count for value in fit_records)
        * layer_count
        * len(_ENDPOINTS)
    )
    heldout_expected_block_count = (
        max(value.phase_count for value in heldout_records)
        * layer_count
        * len(_ENDPOINTS)
    )
    if (
        fit_snapshot_block_count != fit_expected_block_count
        or heldout_snapshot_block_count != heldout_expected_block_count
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Variable-horizon snapshot union omitted authenticated locations."
        )
    fit_margin_evidence = None
    heldout_margin_evidence = None
    vocabulary_margin_gate = None
    streaming_margin_adapter = None
    if vocabulary_margin_observability:
        lexical_head = _find_inventory_operator(
            inventory,
            names=frozenset(
                {
                    "lexical.tied_embedding_head.weight",
                    "lexical.lm_head.weight",
                }
            ),
            label="LFM2 lexical head",
        )
        final_norm = _find_inventory_operator(
            inventory,
            names=frozenset({"output.final_norm.weight"}),
            label="LFM2 final norm",
        )
        if streaming_vocabulary_margin_v2:
            from .hf_lfm2_streaming_vocabulary_margin_adapter import (
                Lfm2TargetTokenizerVocabularyMarginBinding,
                build_lfm2_streaming_vocabulary_margin_adapter_v2,
            )

            if (
                target_tokenizer_vocabulary_margin_binding is not None
                and not isinstance(
                    target_tokenizer_vocabulary_margin_binding,
                    Lfm2TargetTokenizerVocabularyMarginBinding,
                )
            ):
                raise Lfm2CircuitBalancedBasisError(
                    "Target tokenizer vocabulary-margin binding has the wrong "
                    "type."
                )

            streaming_margin_adapter = (
                build_lfm2_streaming_vocabulary_margin_adapter_v2(
                    trace_cache,
                    selective_session,
                    fit_record_ids=tuple(
                        value.record_id for value in fit_records
                    ),
                    heldout_record_ids=tuple(
                        value.record_id for value in heldout_records
                    ),
                    layer_count=layer_count,
                    source_width=inventory.source_width,
                    lexical_head=lexical_head.tensor,
                    final_norm=final_norm.tensor,
                    top_k=vocabulary_margin_top_k,
                    margin_floor_fraction=(vocabulary_margin_floor_fraction),
                    pair_chunk_size=(
                        streaming_vocabulary_margin_pair_chunk_size
                    ),
                    target_tokenizer_binding=(
                        target_tokenizer_vocabulary_margin_binding
                    ),
                )
            )
            vocabulary_margin_gate = streaming_margin_adapter.gate
        else:
            fit_margin_evidence = _build_vocabulary_margin_direction_set(
                fit_records,
                split=EvidenceSplit.FIT,
                lexical_head=lexical_head,
                final_norm=final_norm,
                layer_count=layer_count,
                top_k=vocabulary_margin_top_k,
                margin_floor_fraction=vocabulary_margin_floor_fraction,
            )
            heldout_margin_evidence = _build_vocabulary_margin_direction_set(
                heldout_records,
                split=EvidenceSplit.HELDOUT,
                lexical_head=lexical_head,
                final_norm=final_norm,
                layer_count=layer_count,
                top_k=vocabulary_margin_top_k,
                margin_floor_fraction=vocabulary_margin_floor_fraction,
            )
            vocabulary_margin_gate = _build_vocabulary_margin_transaction_gate(
                heldout_records,
                vocabulary_size=vocabulary_size,
                top_k=vocabulary_margin_top_k,
                projection_evidence=heldout_margin_evidence,
            )
    observability = _build_observability(
        inventory,
        graph,
        vocabulary_margin_operator=(
            None
            if fit_margin_evidence is None
            else fit_margin_evidence.operator
        ),
    )
    required_kinds = tuple(
        sorted({value.circuit_kind for value in observability})
    )
    basis_result = build_balanced_circuit_basis(
        inventory.operators,
        fit_snapshots,
        heldout_snapshots,
        observability,
        target_width,
        circuit_graph=graph,
        precomputed_fit_snapshot_gram=precomputed_fit_snapshot_gram,
        precomputed_heldout_snapshot_gram=(
            precomputed_heldout_snapshot_gram
        ),
        precomputed_downstream_observability=(
            ()
            if streaming_margin_adapter is None
            else (streaming_margin_adapter.precomputed_observability,)
        ),
        structural_controllability_weight=(
            structural_controllability_weight
        ),
        cached_controllability_weight=cached_controllability_weight,
        gram_regularization=gram_regularization,
        rank_tolerance=rank_tolerance,
        max_condition_number=max_condition_number,
        required_observability_kinds=required_kinds,
        require_all_ir_writers=True,
        minimum_heldout_retained_energy_fraction=minimum_heldout,
    )
    margin_report = None
    if (
        fit_margin_evidence is not None
        and heldout_margin_evidence is not None
    ):
        margin_report = {
            "schema": _VOCABULARY_MARGIN_SCHEMA,
            "enabled": True,
            "observability_role": _VOCABULARY_MARGIN_ROLE,
            "fit": dict(fit_margin_evidence.report),
            "heldout": dict(heldout_margin_evidence.report),
            "heldout_diagnostics": (
                _heldout_vocabulary_margin_diagnostics(
                    basis_result.basis,
                    heldout_margin_evidence,
                )
            ),
            "transaction_gate": vocabulary_margin_gate.as_dict(),
            "fit_used_for_selection": True,
            "heldout_used_for_selection": False,
            "full_parent_logits_read_from_sealed_cache": True,
            "parent_forward_calls": 0,
            "trace_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
        margin_report["report_sha256"] = canonical_sha256(margin_report)
    elif streaming_margin_adapter is not None:
        projection_gate = vocabulary_margin_gate.evaluate_projection(
            basis_result.basis,
            minimum_retained_energy_fraction=0.0,
            minimum_sign_agreement_fraction=0.0,
            minimum_top_competitor_nonnegative_fraction=0.0,
            minimum_all_top_k_nonnegative_fraction=0.0,
        )
        adapter_report = streaming_margin_adapter.as_dict()
        margin_report = {
            "schema": _VOCABULARY_MARGIN_SCHEMA,
            "enabled": True,
            "streaming_v2": True,
            "opt_in_only": True,
            "observability_role": _VOCABULARY_MARGIN_ROLE,
            "fit": adapter_report["fit"]["statistics"],
            "heldout": adapter_report["heldout"]["manifest"],
            "heldout_diagnostics": projection_gate["metrics"],
            "heldout_projection_gate": projection_gate,
            "transaction_gate": vocabulary_margin_gate.as_dict(),
            "adapter": adapter_report,
            "fit_used_for_selection": True,
            "heldout_used_for_selection": False,
            "full_parent_logits_read_from_sealed_cache": True,
            "all_declared_fit_rows_consumed": True,
            "row_cap_or_subsample_applied": False,
            "full_sealed_cache_integrity_authenticated": True,
            "selected_margin_tensor_subset_only": True,
            "parent_forward_calls": 0,
            "trace_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
        margin_report["report_sha256"] = canonical_sha256(margin_report)
    fit_ids = {value.record_id for value in fit_records}
    record_horizons = [
        {
            "batch_index": value.index,
            "record_id": value.record_id,
            "split": "fit" if value.record_id in fit_ids else "heldout",
            "phase_count": value.phase_count,
            "decode_phase_count": value.decode_phase_count,
            "last_decode_phase": (
                f"decode.{value.decode_phase_count - 1:03d}"
            ),
            "residual_location_count": value.location_count,
        }
        for value in records
    ]
    phase_counts = [value.phase_count for value in records]
    decode_phase_counts = [value.decode_phase_count for value in records]
    location_counts = [value.location_count for value in records]
    report: dict[str, Any] = {
        "schema": LFM2_CIRCUIT_BALANCED_BASIS_SCHEMA,
        "family": "lfm2",
        "method": (
            "complete-parent-weights-plus-sealed-residual-"
            "controllability-plus-ir-observability"
            + (
                "-plus-vocabulary-margin-observability"
                if vocabulary_margin_observability
                else ""
            )
            + ("-streaming-v2" if streaming_vocabulary_margin_v2 else "")
            + "-balanced-pod"
        ),
        "parent": parent_fingerprint.as_dict(),
        "cache": {
            "cache_key": cache_key,
            "corpus_sha256": corpus_fingerprint,
            "parent_identity_sha256": cached_parent,
            "batch_count": len(records),
            "variable_decode_horizons_supported": True,
            "phase_count_minimum": min(phase_counts),
            "phase_count_maximum": max(phase_counts),
            "decode_phase_count_minimum": min(decode_phase_counts),
            "decode_phase_count_maximum": max(decode_phase_counts),
            "residual_location_count_minimum": min(location_counts),
            "residual_location_count_maximum": max(location_counts),
            "record_horizons": record_horizons,
            "fit_and_heldout_read_from_existing_sealed_cache": True,
            "trace_regeneration_calls": 0,
        },
        "split": split_report,
        "snapshot_adapter": {
            "max_rows_per_block": max_rows_per_block,
            "fit_block_count": fit_snapshot_block_count,
            "heldout_block_count": heldout_snapshot_block_count,
            "fit_union_phase_count": max(
                value.phase_count for value in fit_records
            ),
            "heldout_union_phase_count": max(
                value.phase_count for value in heldout_records
            ),
            "all_phases_layers_endpoints_preserved": True,
            "all_record_locations_consumed_by_own_split": True,
            "common_horizon_truncation_used": False,
            "endpoints": [f"residual.{value}" for value in _ENDPOINTS],
            "row_bounding_occurs_after_whole_record_split": True,
        },
        "inventory": inventory.as_dict(),
        "circuit_ir": {
            "fingerprint": graph.fingerprint,
            "family": graph.family,
            "writer_and_output_interface_coverage_required": True,
        },
        "balanced_basis": basis_result.as_dict(),
        "prompt_independent_whole_weight_objective": True,
        "cached_residual_controllability": True,
        "downstream_observability_balancing": True,
        "model_forward_calls": 0,
        "parent_trace_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
        "hashes": {
            "basis_sha256": basis_result.map_sha256,
            "inventory_sha256": inventory.manifest["inventory_sha256"],
            "circuit_graph_sha256": graph.fingerprint,
        },
    }
    if margin_report is not None:
        report["vocabulary_margin_observability"] = margin_report
    report["result_sha256"] = canonical_sha256(report)
    return Lfm2CircuitBalancedBasisResult(
        inventory=inventory,
        circuit_graph=graph,
        basis_result=basis_result,
        report=report,
        vocabulary_margin_gate=vocabulary_margin_gate,
    )


__all__ = [
    "LFM2_CIRCUIT_BALANCED_BASIS_SCHEMA",
    "Lfm2CircuitBalancedBasisError",
    "Lfm2CircuitBalancedBasisResult",
    "Lfm2VocabularyMarginTransactionGate",
    "build_lfm2_circuit_balanced_basis",
    "lfm2_vocabulary_margin_vulnerability",
]
