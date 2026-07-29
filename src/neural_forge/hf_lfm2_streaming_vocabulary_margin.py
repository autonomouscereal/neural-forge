"""Opt-in manifest-declared-row vocabulary-margin sufficient statistics.

Version one of the LFM2 vocabulary-margin path materializes every normalized
top-token direction, every repeated residual row, and the centered operator
matrix.  That is useful as a direct reference implementation, but its memory
grows with ``records * phases * rows * top_k * hidden_width``.

This version-two core preserves the semantic evidence definition while making
its numerical state independent of the total number of manifest-declared rows.
A caller supplies a tensor-free canonical phase manifest and then yields
exactly one matching, tensor-authenticated phase payload at a time.  Every
declared row is consumed; no cap is a selection or subsampling control.
``maximum_phase_rows`` is only a fail-closed resource contract for one phase.
The core does not claim that the manifest is a complete inventory of an
external sealed cache; that binding belongs to the later cache adapter.

The arithmetic is deliberately separately versioned.  Pair contributions are
formed and summed in fixed-size canonical chunks with one CPU FP64 thread;
whole-phase sufficient statistics are then accumulated with tensor-valued
Kahan sums.  Consequently v2 is expected to be numerically close to the
materialized v1 definition, not bitwise identical to it.

This module stops at reusable sufficient statistics.  It does not change the
default basis builder or the existing transaction gate.  A later adapter can
bind ``second_moment_gram`` as precomputed observability and can lazily rescan
held-out authenticated phases for sign/candidate diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_balanced_circuit_basis import EvidenceSplit


STREAMING_VOCABULARY_MARGIN_ARITHMETIC_SCHEMA = (
    "neural-forge-lfm2-streaming-vocabulary-margin-arithmetic-v2"
)
STREAMING_VOCABULARY_MARGIN_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-streaming-vocabulary-margin-manifest-v2"
)
STREAMING_VOCABULARY_MARGIN_STATISTICS_SCHEMA = (
    "neural-forge-lfm2-streaming-vocabulary-margin-statistics-v2"
)
STREAMING_VOCABULARY_MARGIN_TOP_TOKEN_SCHEMA = (
    "neural-forge-lfm2-streaming-vocabulary-margin-top-token-table-v2"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECODE_PHASE = re.compile(r"decode\.(?P<index>[0-9]{3})\Z")
_NUMERICAL_LOCK = threading.RLock()
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class Lfm2StreamingVocabularyMarginError(ValueError):
    """The v2 manifest, payload, or deterministic arithmetic is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2StreamingVocabularyMarginError(
            f"{name} must be a positive integer."
        )
    return value


def _fraction(
    value: Any,
    name: str,
    *,
    include_zero: bool,
) -> float:
    if isinstance(value, bool):
        raise Lfm2StreamingVocabularyMarginError(
            f"{name} must be a finite fraction."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2StreamingVocabularyMarginError(
            f"{name} must be a finite fraction."
        ) from exc
    lower = result >= 0.0 if include_zero else result > 0.0
    if not math.isfinite(result) or not lower or result > 1.0:
        boundary = "[0, 1]" if include_zero else "(0, 1]"
        raise Lfm2StreamingVocabularyMarginError(
            f"{name} must be within {boundary}."
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2StreamingVocabularyMarginError(
            f"{name} must be a nonempty string."
        )
    return value.strip()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2StreamingVocabularyMarginError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _mutable_report_copy(value: Any) -> Any:
    """Copy a JSON report while rejecting mutable or opaque leaf objects."""

    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Lfm2StreamingVocabularyMarginError(
                    "Streaming report keys must be strings."
                )
            copied[key] = _mutable_report_copy(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_mutable_report_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Lfm2StreamingVocabularyMarginError(
        "Streaming reports must contain finite JSON-compatible values."
    )


def _freeze_report(value: Any) -> Any:
    """Recursively freeze a validated mutable report copy."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_report(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_report(item) for item in value)
    return value


def _phase_key(value: str) -> tuple[int, int, str]:
    if value == "prefill":
        return (0, 0, value)
    match = _DECODE_PHASE.fullmatch(value)
    if match is not None:
        return (1, int(match.group("index")), value)
    return (2, 0, value)


def _tensor_sha256(value: torch.Tensor) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() <= 0
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "Hashed values must be materialized dense tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if (
        (torch.is_floating_point(tensor) or torch.is_complex(tensor))
        and not bool(torch.isfinite(tensor).all().item())
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "Hashed values must be finite."
        )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-streaming-vocabulary-margin-tensor-v2\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    view = memoryview(
        tensor.reshape(-1).view(torch.uint8).numpy()
    ).cast("B")
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[start : start + _HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _authenticated_trace_tensor_sha256(value: torch.Tensor) -> str:
    """Reproduce the sealed LFM2 trace-cache tensor descriptor hash."""

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() <= 0
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "Authenticated values must be materialized dense tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if (
        (torch.is_floating_point(tensor) or torch.is_complex(tensor))
        and not bool(torch.isfinite(tensor).all().item())
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "Authenticated values must be finite."
        )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-circuit-tensor-v1\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    view = memoryview(
        tensor.reshape(-1).view(torch.uint8).numpy()
    ).cast("B")
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[start : start + _HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _source_sha256() -> str:
    path = Path(__file__).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise Lfm2StreamingVocabularyMarginError(
            "Streaming vocabulary-margin source must be a regular file."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginConfig:
    """Explicit v2 resource and semantic controls."""

    top_k: int = 8
    margin_floor_fraction: float = 0.05
    pair_chunk_size: int = 256
    maximum_phase_rows: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_k", _positive_int(self.top_k, "top_k"))
        object.__setattr__(
            self,
            "margin_floor_fraction",
            _fraction(
                self.margin_floor_fraction,
                "margin_floor_fraction",
                include_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "pair_chunk_size",
            _positive_int(self.pair_chunk_size, "pair_chunk_size"),
        )
        object.__setattr__(
            self,
            "maximum_phase_rows",
            _positive_int(self.maximum_phase_rows, "maximum_phase_rows"),
        )

    def arithmetic_contract(self) -> dict[str, Any]:
        body = {
            "schema": STREAMING_VOCABULARY_MARGIN_ARITHMETIC_SCHEMA,
            "algorithm": (
                "manifest-declared-row-phase-local-topk-fixed-pair-chunks-"
                "tensor-kahan-second-moment-v2"
            ),
            "top_k": self.top_k,
            "margin_floor_fraction": self.margin_floor_fraction,
            "pair_chunk_size": self.pair_chunk_size,
            "maximum_phase_rows": self.maximum_phase_rows,
            "maximum_phase_rows_is_fail_closed_not_subsampling": True,
            "all_manifest_declared_rows_required": True,
            "phase_vulnerability_weight": (
                "inverse(cached_top1_margin+deterministic_phase_floor)"
            ),
            "record_weight": "one_over_record_count",
            "phase_weight": "record_weight_over_record_phase_count",
            "phase_weight_normalization": "torch-float64-sum",
            "global_weight_normalization": "tensor-kahan-mass",
            "phase_chunk_reduction": (
                "canonical-fixed-chunk-ordered-torch-add"
            ),
            "cross_phase_reduction": "tensor-kahan",
            "mean_reduction": (
                "phase-local-fixed-chunk-then-cross-phase-tensor-kahan"
            ),
            "second_moment_reduction": (
                "phase-local-fixed-chunk-then-cross-phase-tensor-kahan"
            ),
            "gram_symmetrization": "(G+G.T)*0.5",
            "compute_device": "cpu",
            "compute_dtype": "float64",
            "torch_intraop_threads": 1,
            "canonical_manifest_order": True,
            "not_bitwise_compatible_with_materialized_v1": True,
            "source_sha256": _source_sha256(),
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "byteorder": sys.byteorder,
                "torch_config_sha256": hashlib.sha256(
                    torch.__config__.show().encode("utf-8")
                ).hexdigest(),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "MKL_CBWR",
                    )
                },
            },
        }
        return {
            **body,
            "arithmetic_contract_sha256": canonical_sha256(body),
        }


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginPhaseDescriptor:
    """Tensor-free authenticated identity and row count for one phase."""

    record_index: int
    record_id: str
    phase: str
    row_count: int
    residual_tensor_sha256: str
    logits_tensor_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.record_index, bool) or not isinstance(
            self.record_index, int
        ):
            raise Lfm2StreamingVocabularyMarginError(
                "record_index must be an integer."
            )
        object.__setattr__(
            self,
            "record_id",
            _digest(self.record_id, "record_id"),
        )
        phase = _text(self.phase, "phase")
        if phase != "prefill" and _DECODE_PHASE.fullmatch(phase) is None:
            raise Lfm2StreamingVocabularyMarginError(
                "phase must be prefill or decode.NNN."
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "row_count",
            _positive_int(self.row_count, "row_count"),
        )
        object.__setattr__(
            self,
            "residual_tensor_sha256",
            _digest(
                self.residual_tensor_sha256,
                "residual_tensor_sha256",
            ),
        )
        object.__setattr__(
            self,
            "logits_tensor_sha256",
            _digest(self.logits_tensor_sha256, "logits_tensor_sha256"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "record_id": self.record_id,
            "phase": self.phase,
            "row_count": self.row_count,
            "residual_tensor_sha256": self.residual_tensor_sha256,
            "logits_tensor_sha256": self.logits_tensor_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginManifest:
    """Canonical caller-declared phase lattice for one isolated split."""

    split: EvidenceSplit
    phases: tuple[Lfm2StreamingVocabularyMarginPhaseDescriptor, ...]
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.split, EvidenceSplit):
            raise Lfm2StreamingVocabularyMarginError(
                "split must be EvidenceSplit.FIT or EvidenceSplit.HELDOUT."
            )
        phases = tuple(self.phases)
        if not phases or any(
            not isinstance(
                phase,
                Lfm2StreamingVocabularyMarginPhaseDescriptor,
            )
            for phase in phases
        ):
            raise Lfm2StreamingVocabularyMarginError(
                "phases must contain typed phase descriptors."
            )
        expected_order = tuple(
            sorted(
                phases,
                key=lambda value: (
                    value.record_index,
                    _phase_key(value.phase),
                ),
            )
        )
        if phases != expected_order or len(
            {(value.record_index, value.phase) for value in phases}
        ) != len(phases):
            raise Lfm2StreamingVocabularyMarginError(
                "Phase descriptors must be unique and canonically ordered."
            )
        by_record: dict[int, list[Lfm2StreamingVocabularyMarginPhaseDescriptor]] = {}
        for phase in phases:
            by_record.setdefault(phase.record_index, []).append(phase)
        record_ids: set[str] = set()
        for record_index, values in by_record.items():
            ids = {value.record_id for value in values}
            ordered_phases = [value.phase for value in values]
            decode_indices = [
                int(match.group("index"))
                for value in ordered_phases
                if (match := _DECODE_PHASE.fullmatch(value)) is not None
            ]
            if (
                len(ids) != 1
                or next(iter(ids)) in record_ids
                or ordered_phases.count("prefill") != 1
                or not decode_indices
                or decode_indices != list(range(len(decode_indices)))
            ):
                raise Lfm2StreamingVocabularyMarginError(
                    f"Record {record_index} has an invalid phase horizon."
                )
            record_ids.update(ids)
        body = {
            "schema": STREAMING_VOCABULARY_MARGIN_MANIFEST_SCHEMA,
            "authenticated_tensor_hash_schema": (
                "neural-forge-lfm2-circuit-tensor-v1"
            ),
            "split": self.split.value,
            "record_count": len(by_record),
            "phase_count": len(phases),
            "record_ids_sha256": canonical_sha256(sorted(record_ids)),
            "phases": [value.as_dict() for value in phases],
            "all_rows_declared_before_payload_iteration": True,
            "sealed_cache_completeness_authenticated": False,
            "authenticated_source_receipt_required_from_later_adapter": True,
            "fit_heldout_isolation_by_separate_manifest": True,
        }
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "manifest_sha256", canonical_sha256(body))

    @property
    def record_count(self) -> int:
        return len({value.record_id for value in self.phases})

    @property
    def phase_count_by_record(self) -> Mapping[int, int]:
        counts: dict[int, int] = {}
        for phase in self.phases:
            counts[phase.record_index] = counts.get(phase.record_index, 0) + 1
        return MappingProxyType(counts)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": STREAMING_VOCABULARY_MARGIN_MANIFEST_SCHEMA,
            "authenticated_tensor_hash_schema": (
                "neural-forge-lfm2-circuit-tensor-v1"
            ),
            "split": self.split.value,
            "record_count": self.record_count,
            "phase_count": len(self.phases),
            "record_ids_sha256": canonical_sha256(
                sorted({value.record_id for value in self.phases})
            ),
            "phases": [value.as_dict() for value in self.phases],
            "all_rows_declared_before_payload_iteration": True,
            "sealed_cache_completeness_authenticated": False,
            "authenticated_source_receipt_required_from_later_adapter": True,
            "fit_heldout_isolation_by_separate_manifest": True,
        }
        return {**body, "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginPhaseBatch:
    """One authenticated phase payload, released by its yielding adapter."""

    descriptor: Lfm2StreamingVocabularyMarginPhaseDescriptor
    final_residual: torch.Tensor
    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(
            self.descriptor,
            Lfm2StreamingVocabularyMarginPhaseDescriptor,
        ):
            raise Lfm2StreamingVocabularyMarginError(
                "descriptor has the wrong type."
            )


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginTopTokenTable:
    """Deterministic top-1 plus top-k competitor rows for one phase."""

    token_ids: torch.Tensor
    logit_values: torch.Tensor

    def __post_init__(self) -> None:
        token_ids = self.token_ids
        logit_values = self.logit_values
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.dtype != torch.long
            or token_ids.device.type != "cpu"
            or token_ids.ndim != 2
            or token_ids.numel() < 2
            or not isinstance(logit_values, torch.Tensor)
            or logit_values.device.type != "cpu"
            or logit_values.dtype != torch.float64
            or tuple(logit_values.shape) != tuple(token_ids.shape)
            or not bool(torch.isfinite(logit_values).all().item())
            or not bool(
                (logit_values[:, :1] >= logit_values[:, 1:]).all().item()
            )
        ):
            raise Lfm2StreamingVocabularyMarginError(
                "Top-token table invariants are invalid."
            )
        object.__setattr__(self, "token_ids", token_ids.contiguous())
        object.__setattr__(self, "logit_values", logit_values.contiguous())

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

    def hashes(self) -> dict[str, str]:
        body = {
            "schema": STREAMING_VOCABULARY_MARGIN_TOP_TOKEN_SCHEMA,
            "row_count": self.row_count,
            "top_k": self.top_k,
            "token_ids_sha256": _tensor_sha256(self.token_ids),
            "logit_values_sha256": _tensor_sha256(self.logit_values),
        }
        return {**body, "table_sha256": canonical_sha256(body)}


def deterministic_top_token_table_v2(
    logits: torch.Tensor,
    *,
    top_k: int,
) -> Lfm2StreamingVocabularyMarginTopTokenTable:
    """Match v1's value ordering and token-ID tie break for one phase."""

    top_k = _positive_int(top_k, "top_k")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.layout != torch.strided
        or logits.device.type == "meta"
        or logits.ndim < 2
        or not torch.is_floating_point(logits)
        or torch.is_complex(logits)
        or logits.numel() == 0
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "logits must be a nonempty dense floating tensor."
        )
    vocabulary_size = int(logits.shape[-1])
    if top_k >= vocabulary_size:
        raise Lfm2StreamingVocabularyMarginError(
            "top_k must be smaller than the vocabulary size."
        )
    rows = (
        logits.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, vocabulary_size)
        .contiguous()
    )
    if not bool(torch.isfinite(rows).all().item()):
        raise Lfm2StreamingVocabularyMarginError("logits contain NaN or infinity.")
    candidate_count = top_k + 1
    token_rows: list[list[int]] = []
    value_rows: list[list[float]] = []
    for row in rows:
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
            raise Lfm2StreamingVocabularyMarginError(
                "Could not canonicalize top-token ties."
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
    return Lfm2StreamingVocabularyMarginTopTokenTable(
        token_ids=torch.tensor(token_rows, dtype=torch.long),
        logit_values=torch.tensor(value_rows, dtype=torch.float64),
    )


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginStatistics:
    """Declared-row v2 second moment with no retained per-row tensors."""

    second_moment_gram: torch.Tensor
    weighted_mean: torch.Tensor
    report: Mapping[str, Any]

    @staticmethod
    def _validated_report_copy(
        gram: torch.Tensor,
        mean: torch.Tensor,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            not isinstance(gram, torch.Tensor)
            or gram.dtype != torch.float64
            or gram.device.type != "cpu"
            or gram.ndim != 2
            or int(gram.shape[0]) != int(gram.shape[1])
            or not isinstance(mean, torch.Tensor)
            or mean.dtype != torch.float64
            or mean.device.type != "cpu"
            or tuple(mean.shape) != (int(gram.shape[0]),)
            or not bool(torch.isfinite(gram).all().item())
            or not bool(torch.isfinite(mean).all().item())
            or not torch.allclose(
                gram,
                gram.T,
                rtol=0.0,
                atol=0.0,
            )
            or not isinstance(report, Mapping)
        ):
            raise Lfm2StreamingVocabularyMarginError(
                "Streaming vocabulary-margin statistics are invalid."
            )
        normalized_report = _mutable_report_copy(report)
        if not isinstance(normalized_report, dict):  # pragma: no cover
            raise AssertionError("report copy must remain a dictionary")
        claimed = normalized_report.get("result_sha256")
        report_body = dict(normalized_report)
        report_body.pop("result_sha256", None)
        if claimed != canonical_sha256(report_body):
            raise Lfm2StreamingVocabularyMarginError(
                "Streaming vocabulary-margin report seal is invalid."
            )
        hashes = normalized_report.get("hashes")
        if (
            normalized_report.get("schema")
            != STREAMING_VOCABULARY_MARGIN_STATISTICS_SCHEMA
            or not isinstance(hashes, dict)
            or hashes.get("gram_sha256") != _tensor_sha256(gram)
            or hashes.get("weighted_mean_sha256") != _tensor_sha256(mean)
        ):
            raise Lfm2StreamingVocabularyMarginError(
                "Streaming vocabulary-margin tensor hashes are invalid."
            )
        return normalized_report

    def __post_init__(self) -> None:
        gram = self.second_moment_gram
        mean = self.weighted_mean
        normalized_report = self._validated_report_copy(
            gram,
            mean,
            self.report,
        )
        object.__setattr__(
            self,
            "second_moment_gram",
            gram.detach().contiguous().clone(),
        )
        object.__setattr__(
            self,
            "weighted_mean",
            mean.detach().contiguous().clone(),
        )
        object.__setattr__(
            self,
            "report",
            _freeze_report(normalized_report),
        )

    def _authenticate_state(self) -> dict[str, Any]:
        """Re-authenticate mutable public tensors before every observation."""

        return self._validated_report_copy(
            self.second_moment_gram,
            self.weighted_mean,
            self.report,
        )

    @property
    def source_width(self) -> int:
        self._authenticate_state()
        return int(self.second_moment_gram.shape[0])

    @property
    def retained_tensor_numel(self) -> int:
        self._authenticate_state()
        return int(self.second_moment_gram.numel() + self.weighted_mean.numel())

    def retained_energy_fraction(self, basis: torch.Tensor) -> float:
        self._authenticate_state()
        source_width = int(self.second_moment_gram.shape[0])
        if (
            not isinstance(basis, torch.Tensor)
            or basis.layout != torch.strided
            or basis.device.type == "meta"
            or basis.ndim != 2
            or int(basis.shape[0]) != source_width
            or int(basis.shape[1]) <= 0
            or not torch.is_floating_point(basis)
            or torch.is_complex(basis)
            or not bool(torch.isfinite(basis).all().item())
        ):
            raise Lfm2StreamingVocabularyMarginError(
                "basis must be a finite source-by-target matrix."
            )
        basis64 = basis.detach().to(device="cpu", dtype=torch.float64)
        denominator = float(torch.trace(self.second_moment_gram).item())
        numerator = float(
            torch.trace(
                basis64.T @ self.second_moment_gram @ basis64
            ).item()
        )
        if denominator <= 0.0 or not math.isfinite(numerator):
            raise Lfm2StreamingVocabularyMarginError(
                "Vocabulary-margin Gram has no finite positive energy."
            )
        return min(1.0, max(0.0, numerator / denominator))

    def as_dict(self) -> dict[str, Any]:
        return self._authenticate_state()


def _phase_tensors(
    batch: Lfm2StreamingVocabularyMarginPhaseBatch,
    *,
    source_width: int,
    vocabulary_size: int,
    maximum_phase_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    descriptor = batch.descriptor
    residual = batch.final_residual
    logits = batch.logits
    if (
        not isinstance(residual, torch.Tensor)
        or residual.layout != torch.strided
        or residual.device.type == "meta"
        or residual.ndim < 2
        or int(residual.shape[-1]) != source_width
        or not torch.is_floating_point(residual)
        or torch.is_complex(residual)
        or residual.numel() == 0
        or not isinstance(logits, torch.Tensor)
        or logits.layout != torch.strided
        or logits.device.type == "meta"
        or logits.ndim < 2
        or int(logits.shape[-1]) != vocabulary_size
        or not torch.is_floating_point(logits)
        or torch.is_complex(logits)
        or logits.numel() == 0
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "Phase residual/logit payloads have incompatible tensor shapes."
        )
    residual64 = (
        residual.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, source_width)
        .contiguous()
    )
    logits64 = (
        logits.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, vocabulary_size)
        .contiguous()
    )
    row_count = int(residual64.shape[0])
    if (
        row_count != descriptor.row_count
        or int(logits64.shape[0]) != row_count
        or row_count > maximum_phase_rows
        or not bool(torch.isfinite(residual64).all().item())
        or not bool(torch.isfinite(logits64).all().item())
        or _authenticated_trace_tensor_sha256(residual)
        != descriptor.residual_tensor_sha256
        or _authenticated_trace_tensor_sha256(logits)
        != descriptor.logits_tensor_sha256
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "Phase residual/logit rows are not aligned with their manifest."
        )
    return residual64, logits64


def _kahan_add(
    total: torch.Tensor,
    compensation: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjusted = value - compensation
    updated = total + adjusted
    next_compensation = (updated - total) - adjusted
    return updated, next_compensation


def build_streaming_vocabulary_margin_statistics_v2(
    manifest: Lfm2StreamingVocabularyMarginManifest,
    phase_batches: Iterable[Lfm2StreamingVocabularyMarginPhaseBatch],
    *,
    lexical_head: torch.Tensor,
    final_norm: torch.Tensor,
    config: Lfm2StreamingVocabularyMarginConfig | None = None,
) -> Lfm2StreamingVocabularyMarginStatistics:
    """Consume every declared phase once and retain only v2 sufficient stats."""

    if not isinstance(manifest, Lfm2StreamingVocabularyMarginManifest):
        raise TypeError("manifest must be Lfm2StreamingVocabularyMarginManifest.")
    config = (
        Lfm2StreamingVocabularyMarginConfig()
        if config is None
        else config
    )
    if not isinstance(config, Lfm2StreamingVocabularyMarginConfig):
        raise TypeError("config must be Lfm2StreamingVocabularyMarginConfig.")
    if (
        not isinstance(lexical_head, torch.Tensor)
        or lexical_head.layout != torch.strided
        or lexical_head.device.type == "meta"
        or lexical_head.ndim != 2
        or not torch.is_floating_point(lexical_head)
        or torch.is_complex(lexical_head)
        or not isinstance(final_norm, torch.Tensor)
        or final_norm.layout != torch.strided
        or final_norm.device.type == "meta"
        or final_norm.ndim != 1
        or not torch.is_floating_point(final_norm)
        or torch.is_complex(final_norm)
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "lexical_head/final_norm must be dense floating tensors."
        )
    vocabulary_size, source_width = map(int, lexical_head.shape)
    if (
        tuple(final_norm.shape) != (source_width,)
        or config.top_k >= vocabulary_size
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "lexical_head/final_norm dimensions are incompatible with config."
        )
    head = lexical_head.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    norm = final_norm.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(head).all().item()) or not bool(
        torch.isfinite(norm).all().item()
    ):
        raise Lfm2StreamingVocabularyMarginError(
            "lexical_head/final_norm contain NaN or infinity."
        )

    arithmetic = config.arithmetic_contract()
    phase_counts = manifest.phase_count_by_record
    record_weight = 1.0 / manifest.record_count
    iterator = iter(phase_batches)
    original_threads: int | None = None
    with _NUMERICAL_LOCK:
        try:
            original_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            with torch.no_grad():
                mean_total = torch.zeros(source_width, dtype=torch.float64)
                mean_compensation = torch.zeros_like(mean_total)
                gram_total = torch.zeros(
                    (source_width, source_width),
                    dtype=torch.float64,
                )
                gram_compensation = torch.zeros_like(gram_total)
                mass_total = torch.zeros((), dtype=torch.float64)
                mass_compensation = torch.zeros_like(mass_total)
                phase_reports: list[dict[str, Any]] = []
                total_rows = 0
                total_directions = 0
                total_zero_directions = 0

                for expected in manifest.phases:
                    try:
                        batch = next(iterator)
                    except StopIteration as exc:
                        raise Lfm2StreamingVocabularyMarginError(
                            "Phase payload iterator ended before its manifest."
                        ) from exc
                    if (
                        not isinstance(
                            batch,
                            Lfm2StreamingVocabularyMarginPhaseBatch,
                        )
                        or batch.descriptor != expected
                    ):
                        raise Lfm2StreamingVocabularyMarginError(
                            "Phase payload order or identity differs from manifest."
                        )
                    residual, logits = _phase_tensors(
                        batch,
                        source_width=source_width,
                        vocabulary_size=vocabulary_size,
                        maximum_phase_rows=config.maximum_phase_rows,
                    )
                    table = deterministic_top_token_table_v2(
                        logits,
                        top_k=config.top_k,
                    )
                    if table.row_count != expected.row_count:
                        raise Lfm2StreamingVocabularyMarginError(
                            "Top-token rows do not align with final residual rows."
                        )
                    top_ids = (
                        table.token_ids[:, :1]
                        .expand(-1, config.top_k)
                        .reshape(-1)
                    )
                    competitor_ids = table.token_ids[:, 1:].reshape(-1)
                    pair_count = int(top_ids.numel())
                    valid = torch.zeros(pair_count, dtype=torch.bool)
                    direction_floor = (
                        torch.finfo(torch.float64).eps
                        * max(1, source_width)
                        * 16.0
                    )
                    for start in range(0, pair_count, config.pair_chunk_size):
                        end = min(pair_count, start + config.pair_chunk_size)
                        effective = (
                            (
                                head.index_select(0, top_ids[start:end])
                                - head.index_select(
                                    0,
                                    competitor_ids[start:end],
                                )
                            )
                            * norm.unsqueeze(0)
                        )
                        valid[start:end] = (
                            torch.linalg.vector_norm(effective, dim=1)
                            > direction_floor
                        )
                    valid_count = int(valid.sum().item())
                    if valid_count <= 0:
                        raise Lfm2StreamingVocabularyMarginError(
                            f"All {expected.phase!r} top-token directions "
                            "are numerically zero."
                        )
                    margins_full = table.margins.reshape(-1)
                    margins = margins_full[valid]
                    values_scale = max(
                        1.0,
                        float(table.logit_values.abs().max().item()),
                    )
                    positive_margins = margins[margins > 0.0]
                    margin_scale = (
                        float(torch.median(positive_margins).item())
                        if int(positive_margins.numel()) > 0
                        else values_scale
                    )
                    margin_floor = max(
                        margin_scale * config.margin_floor_fraction,
                        torch.finfo(torch.float64).eps
                        * values_scale
                        * 16.0,
                    )
                    vulnerability = 1.0 / (margins + margin_floor)
                    phase_weight = (
                        record_weight / phase_counts[expected.record_index]
                    )
                    valid_weights = (
                        vulnerability
                        / vulnerability.sum()
                        * phase_weight
                    ).contiguous()
                    weights = torch.zeros(pair_count, dtype=torch.float64)
                    weights[valid] = valid_weights
                    phase_mass = valid_weights.sum()
                    phase_mean = torch.zeros(
                        source_width,
                        dtype=torch.float64,
                    )
                    phase_gram = torch.zeros(
                        (source_width, source_width),
                        dtype=torch.float64,
                    )
                    valid_cursor = 0
                    for start in range(0, pair_count, config.pair_chunk_size):
                        end = min(pair_count, start + config.pair_chunk_size)
                        chunk_valid = valid[start:end]
                        chunk_count = int(chunk_valid.sum().item())
                        if chunk_count <= 0:
                            continue
                        effective = (
                            (
                                head.index_select(0, top_ids[start:end])
                                - head.index_select(
                                    0,
                                    competitor_ids[start:end],
                                )
                            )
                            * norm.unsqueeze(0)
                        )[chunk_valid]
                        norms = torch.linalg.vector_norm(effective, dim=1)
                        units = effective / norms.unsqueeze(1)
                        chunk_weights = weights[start:end][chunk_valid]
                        phase_mean = phase_mean + (
                            units * chunk_weights.unsqueeze(1)
                        ).sum(dim=0)
                        phase_gram = phase_gram + (
                            units.T
                            @ (units * chunk_weights.unsqueeze(1))
                        )
                        valid_cursor += chunk_count
                    if valid_cursor != valid_count:  # pragma: no cover
                        raise AssertionError("valid direction cursor disagrees")
                    mass_total, mass_compensation = _kahan_add(
                        mass_total,
                        mass_compensation,
                        phase_mass,
                    )
                    mean_total, mean_compensation = _kahan_add(
                        mean_total,
                        mean_compensation,
                        phase_mean,
                    )
                    gram_total, gram_compensation = _kahan_add(
                        gram_total,
                        gram_compensation,
                        phase_gram,
                    )
                    skipped = pair_count - valid_count
                    total_rows += expected.row_count
                    total_directions += valid_count
                    total_zero_directions += skipped
                    table_hashes = table.hashes()
                    phase_body = {
                        "record_index": expected.record_index,
                        "record_id": expected.record_id,
                        "phase": expected.phase,
                        "logit_row_count": expected.row_count,
                        "top_k_competitor_count": config.top_k,
                        "direction_count": valid_count,
                        "zero_direction_count_skipped": skipped,
                        "hierarchical_phase_weight": phase_weight,
                        "phase_weight_sum_before_global_normalization": float(
                            phase_mass.item()
                        ),
                        "margin_floor": margin_floor,
                        "minimum_cached_margin": float(margins.min().item()),
                        "median_cached_margin": float(
                            torch.median(margins).item()
                        ),
                        "maximum_cached_margin": float(margins.max().item()),
                        "top_token_table": table_hashes,
                        "all_declared_rows_consumed": True,
                    }
                    phase_reports.append(
                        {
                            **phase_body,
                            "phase_statistics_sha256": canonical_sha256(
                                phase_body
                            ),
                        }
                    )
                iterator_exhausted = object()
                extra = next(iterator, iterator_exhausted)
                if extra is not iterator_exhausted:
                    raise Lfm2StreamingVocabularyMarginError(
                        "Phase payload iterator contains undeclared evidence."
                    )
                total_mass = float(mass_total.item())
                if not math.isfinite(total_mass) or total_mass <= 0.0:
                    raise Lfm2StreamingVocabularyMarginError(
                        "Streaming vocabulary-margin weights have no mass."
                    )
                weighted_mean = (mean_total / total_mass).contiguous()
                gram = (gram_total / total_mass).contiguous()
                gram = ((gram + gram.T) * 0.5).contiguous()
                trace_energy = float(torch.trace(gram).item())
                if (
                    not bool(torch.isfinite(weighted_mean).all().item())
                    or not bool(torch.isfinite(gram).all().item())
                    or not math.isfinite(trace_energy)
                    or trace_energy <= 0.0
                ):
                    raise Lfm2StreamingVocabularyMarginError(
                        "Streaming vocabulary-margin statistics are non-finite."
                    )
        finally:
            if original_threads is not None:
                torch.set_num_threads(original_threads)

    report: dict[str, Any] = {
        "schema": STREAMING_VOCABULARY_MARGIN_STATISTICS_SCHEMA,
        "split": manifest.split.value,
        "method": (
            "manifest-declared-row-streaming-top1-vs-topk-second-moment-v2"
        ),
        "manifest": manifest.as_dict(),
        "arithmetic_contract": arithmetic,
        "source_width": source_width,
        "vocabulary_size": vocabulary_size,
        "record_count": manifest.record_count,
        "phase_count": len(manifest.phases),
        "logit_row_count": total_rows,
        "top_k": config.top_k,
        "direction_count": total_directions,
        "zero_direction_count_skipped": total_zero_directions,
        "embedding_row_differences_used": True,
        "final_norm_diagonal_applied": True,
        "directions_l2_normalized": True,
        "vulnerability_weight": (
            "inverse(cached_top1_margin+deterministic_phase_floor)"
        ),
        "equal_record_weight": True,
        "equal_phase_weight_within_record": True,
        "weights_sum_before_global_normalization": total_mass,
        "weights_normalized_to_one_in_sufficient_statistics": True,
        "coverage_scope": "manifest-declared-phase-payloads-only",
        "all_manifest_declared_rows_consumed": total_rows
        == sum(value.row_count for value in manifest.phases),
        "sealed_cache_completeness_authenticated": False,
        "row_cap_or_subsample_applied": False,
        "fit_heldout_isolation": {
            "split": manifest.split.value,
            "separate_manifest_required": True,
            "cross_split_rows_consumed": 0,
        },
        "retention": {
            "full_direction_rows_retained": 0,
            "full_residual_rows_retained": 0,
            "full_top_token_tables_retained": 0,
            "retained_tensor_numel": int(gram.numel() + weighted_mean.numel()),
            "retained_state": ["second_moment_gram", "weighted_mean"],
            "retained_numeric_tensor_ram_scales_with_total_rows": False,
            "manifest_and_report_metadata_scale_with_phase_count": True,
        },
        "input_hashes": {
            "lexical_head_sha256": _tensor_sha256(head),
            "final_norm_sha256": _tensor_sha256(norm),
            "manifest_sha256": manifest.manifest_sha256,
        },
        "hashes": {
            "gram_sha256": _tensor_sha256(gram),
            "weighted_mean_sha256": _tensor_sha256(weighted_mean),
            "phase_statistics_sha256": canonical_sha256(phase_reports),
        },
        "phases": phase_reports,
        "execution": {
            "parent_forward_calls": 0,
            "trace_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "autograd_used": False,
            "filesystem_writes": 0,
        },
        "integration_status": {
            "default_basis_builder_changed": False,
            "sealed_cache_completeness_adapter_required": True,
            "precomputed_observability_adapter_required": True,
            "lazy_projection_gate_rescan_required": True,
            "lazy_candidate_gate_rescan_required": True,
        },
    }
    report["result_sha256"] = canonical_sha256(report)
    return Lfm2StreamingVocabularyMarginStatistics(
        second_moment_gram=gram,
        weighted_mean=weighted_mean,
        report=report,
    )


__all__ = [
    "STREAMING_VOCABULARY_MARGIN_ARITHMETIC_SCHEMA",
    "STREAMING_VOCABULARY_MARGIN_MANIFEST_SCHEMA",
    "STREAMING_VOCABULARY_MARGIN_STATISTICS_SCHEMA",
    "STREAMING_VOCABULARY_MARGIN_TOP_TOKEN_SCHEMA",
    "Lfm2StreamingVocabularyMarginConfig",
    "Lfm2StreamingVocabularyMarginError",
    "Lfm2StreamingVocabularyMarginManifest",
    "Lfm2StreamingVocabularyMarginPhaseBatch",
    "Lfm2StreamingVocabularyMarginPhaseDescriptor",
    "Lfm2StreamingVocabularyMarginStatistics",
    "Lfm2StreamingVocabularyMarginTopTokenTable",
    "build_streaming_vocabulary_margin_statistics_v2",
    "deterministic_top_token_table_v2",
]
