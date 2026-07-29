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
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_balanced_circuit_basis import (
    BalancedCircuitBasisResult,
    DownstreamObservability,
    EvidenceSplit,
    ResidualSnapshotBlock,
    build_balanced_circuit_basis,
)
from .dynamic_circuit_ir import CircuitKind, ModelCircuitGraph, canonical_sha256
from .dynamic_model_inspector import inspect_model_circuits
from .dynamic_weight_complete_basis import (
    FeatureOrientation,
    ResidualFacingOperator,
)
from .hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
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


def _load_cache_records(
    trace_cache: Any,
    *,
    source_width: int,
    layer_count: int,
    vocabulary_size: int,
    vocabulary_margin_top_k: int | None,
) -> tuple[_CacheRecord, ...]:
    iterator = getattr(trace_cache, "iter_batches", None)
    if not callable(iterator):
        raise Lfm2CircuitBalancedBasisError(
            "trace_cache must expose iter_batches()."
        )
    records: list[_CacheRecord] = []
    for fallback_index, batch in enumerate(iterator()):
        tensors = getattr(batch, "tensors", None)
        if not isinstance(tensors, Mapping) or not tensors:
            raise Lfm2CircuitBalancedBasisError(
                "Every verified cache batch must expose a tensor mapping."
            )
        index = getattr(batch, "index", fallback_index)
        record_id = getattr(batch, "input_sha256", None)
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
        vocabulary_margins: dict[
            str, _CachedVocabularyMarginPhase
        ] = {}
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
                final_rows = locations[
                    (phase, layer_count - 1, "post_layer")
                ]
                if margin_phase.row_count != int(final_rows.shape[0]):
                    raise Lfm2CircuitBalancedBasisError(
                        f"Cached {phase!r} logits and final residual rows do "
                        "not align."
                    )
                vocabulary_margins[phase] = margin_phase
        records.append(
            _CacheRecord(
                index=index,
                record_id=record_id,
                locations=locations,
                vocabulary_margins=vocabulary_margins,
            )
        )
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


def _split_records(
    records: Sequence[_CacheRecord],
    *,
    cache_key: str,
    validation_fraction: float,
    split_seed: str,
) -> tuple[tuple[_CacheRecord, ...], tuple[_CacheRecord, ...], dict[str, Any]]:
    count = len(records)
    heldout_count = max(1, min(count - 1, math.ceil(count * validation_fraction)))
    ranked = sorted(
        records,
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
    vocabulary_margin_gate: (
        Lfm2VocabularyMarginTransactionGate | None
    ) = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.inventory, Lfm2WeightCompleteInventory)
            or not isinstance(self.circuit_graph, ModelCircuitGraph)
            or not isinstance(self.basis_result, BalancedCircuitBasisResult)
            or self.inventory.source_width != self.basis_result.source_width
            or not isinstance(self.report, Mapping)
            or (
                self.vocabulary_margin_gate is not None
                and not isinstance(
                    self.vocabulary_margin_gate,
                    Lfm2VocabularyMarginTransactionGate,
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
    verify = getattr(trace_cache, "verify", None)
    if not callable(verify):
        raise Lfm2CircuitBalancedBasisError(
            "trace_cache must expose verify()."
        )
    verified = verify()
    if verified is not None and verified is not trace_cache:
        raise Lfm2CircuitBalancedBasisError(
            "trace_cache.verify() returned a different cache handle."
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
    )
    declared_batch_count = getattr(trace_cache, "batch_count", len(records))
    if (
        isinstance(declared_batch_count, bool)
        or not isinstance(declared_batch_count, int)
        or declared_batch_count != len(records)
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Verified cache batch_count disagrees with iter_batches()."
        )
    fit_records, heldout_records, split_report = _split_records(
        records,
        cache_key=cache_key,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
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
        len(fit_snapshots) != fit_expected_block_count
        or len(heldout_snapshots) != heldout_expected_block_count
    ):
        raise Lfm2CircuitBalancedBasisError(
            "Variable-horizon snapshot union omitted authenticated locations."
        )
    fit_margin_evidence = None
    heldout_margin_evidence = None
    vocabulary_margin_gate = None
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
            "fit_block_count": len(fit_snapshots),
            "heldout_block_count": len(heldout_snapshots),
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
]
