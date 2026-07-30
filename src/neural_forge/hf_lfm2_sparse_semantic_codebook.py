"""Local sparse semantic codes over an already-planted LFM2 anchor table.

Raw-token spans and semantic body inputs are different channels.  The existing
span adapter remains the exact external-emission channel.  This module builds
only the semantic channel: for every defined donor row it stores a sparse row
of coefficients over the target model's planted anchor table.

The construction is analytic and local.  Direct strict-bridge anchors are
fixed to exact one-hot rows.  Other donor rows discover a bounded cosine-near
candidate neighborhood, solve a small ridge problem, retain the strongest
``k`` coordinates, and refit only that retained neighborhood.  Candidate
discovery is a separately sealed artifact so several ``k`` or ridge variants
can reuse it without repeating normalization or similarity computation.

No gradient, optimizer, training, pruning, quantization, global map, parent
forward, or network operation is performed.  Integer CSR offsets/IDs are
structural routing metadata.  Every published FP32 coefficient is folded child
state and is counted explicitly in the receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from neural_forge.dynamic_circuit_ir import canonical_sha256


LFM2_SPARSE_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-candidate-inventory-v1"
)
LFM2_SPARSE_SEMANTIC_CANDIDATE_CHUNK_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-candidate-chunk-v1"
)
LFM2_SPARSE_SEMANTIC_CANDIDATE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-candidate-request-v1"
)
LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-codebook-v1"
)
LFM2_SPARSE_SEMANTIC_COEFFICIENT_CHUNK_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-coefficient-chunk-v1"
)
LFM2_SPARSE_SEMANTIC_COEFFICIENT_REQUEST_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-coefficient-request-v1"
)
LFM2_SPARSE_SEMANTIC_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-progress-v1"
)
LFM2_SPARSE_SEMANTIC_PERSISTENCE_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-persistence-v1"
)

_SUPPORTED_INPUT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)
_SUPPORTED_COMPUTE_DTYPES = frozenset({torch.float32, torch.float64})
_SEMANTIC_ALGORITHM = (
    "bounded-absolute-cosine-neighbors-local-ridge-active-absolute-k-refit-v1"
)
_CANDIDATE_ALGORITHM = (
    "bounded-absolute-cosine-neighbors-stable-target-id-ties-v1"
)
_TENSOR_HASH_DOMAIN = b"neural-forge-lfm2-sparse-semantic-tensor-v1\0"
_PROHIBITED_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "model_forward_calls": 0,
    "global_map_solve_calls": 0,
    "global_factorization_calls": 0,
    "pruned_parameters": 0,
    "quantized_parameters": 0,
    "network_access": False,
}


class Lfm2SparseSemanticCodebookError(RuntimeError):
    """Sparse semantic-code inputs, checkpoints, or seals are invalid."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise Lfm2SparseSemanticCodebookError(
                "Receipts cannot contain non-finite numbers."
            )
        return value
    raise Lfm2SparseSemanticCodebookError(
        f"Receipt contains non-JSON value {type(value).__name__}."
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _sealed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = _plain_json(body)
    return {**value, field: canonical_sha256(value)}


def _verify_sealed(
    value: Mapping[str, Any],
    *,
    schema: str,
    field: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SparseSemanticCodebookError(f"{name} must be a mapping.")
    result = _plain_json(value)
    supplied = result.pop(field, None)
    if supplied != canonical_sha256(result):
        raise Lfm2SparseSemanticCodebookError(f"{name} seal is invalid.")
    if result.get("schema") != schema:
        raise Lfm2SparseSemanticCodebookError(f"{name} schema is unsupported.")
    return {**result, field: supplied}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SparseSemanticCodebookError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SparseSemanticCodebookError(f"{name} must be nonnegative.")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise Lfm2SparseSemanticCodebookError(
            f"{name} must be finite and positive."
        )
    return float(value)


def _dtype_name(value: Tensor | torch.dtype) -> str:
    dtype = value.dtype if isinstance(value, Tensor) else value
    return str(dtype).removeprefix("torch.")


def _compute_dtype(value: str | torch.dtype, name: str) -> torch.dtype:
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    dtype = value if isinstance(value, torch.dtype) else aliases.get(str(value).lower())
    if dtype not in _SUPPORTED_COMPUTE_DTYPES:
        raise Lfm2SparseSemanticCodebookError(
            f"{name} must be explicit FP32 or FP64."
        )
    assert isinstance(dtype, torch.dtype)
    return dtype


def _compute_device(value: str | torch.device, name: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2SparseSemanticCodebookError(f"{name} is invalid.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise Lfm2SparseSemanticCodebookError(
            f"{name} must be a CPU or CUDA device."
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Lfm2SparseSemanticCodebookError(
                f"{name} requests CUDA, but CUDA is unavailable."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise Lfm2SparseSemanticCodebookError(
                f"{name} requests an unavailable CUDA device."
            )
        return torch.device("cuda", index)
    return torch.device("cpu")


def _validated_table(value: Tensor, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) <= 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype not in _SUPPORTED_INPUT_DTYPES
    ):
        raise Lfm2SparseSemanticCodebookError(
            f"{name} must be one nonempty dense FP16/BF16/FP32/FP64 matrix."
        )
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise Lfm2SparseSemanticCodebookError(f"{name} must be finite.")
    return detached


def _require_finite_compute(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise Lfm2SparseSemanticCodebookError(
            f"{name} became non-finite at the explicit compute placement; "
            "select FP64 or a numerically safer device explicitly."
        )


def _validated_mapping(
    value: Tensor | Sequence[int],
    *,
    target_row_count: int,
    donor_row_count: int,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.ndim != 1 or value.device.type == "meta":
            raise Lfm2SparseSemanticCodebookError(
                "target_to_donor must be one integer vector."
            )
        if value.dtype == torch.bool or torch.is_floating_point(value) or value.is_complex():
            raise Lfm2SparseSemanticCodebookError(
                "target_to_donor must be one integer vector."
            )
        mapping = value.detach().to(device="cpu", dtype=torch.int64).contiguous()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise Lfm2SparseSemanticCodebookError(
                "target_to_donor must be an integer tensor or sequence."
            )
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise Lfm2SparseSemanticCodebookError(
                "target_to_donor contains a non-integer value."
            )
        mapping = torch.tensor(tuple(value), dtype=torch.int64)
    if int(mapping.numel()) != target_row_count:
        raise Lfm2SparseSemanticCodebookError(
            "target_to_donor must contain one donor ID per target anchor row."
        )
    if bool((mapping < 0).any()) or bool((mapping >= donor_row_count).any()):
        raise Lfm2SparseSemanticCodebookError(
            "target_to_donor leaves the defined donor table."
        )
    if int(torch.unique(mapping).numel()) != target_row_count:
        raise Lfm2SparseSemanticCodebookError(
            "target_to_donor must be injective."
        )
    return mapping


def _tensor_sha256(value: Tensor, *, rows_per_chunk: int = 4096) -> str:
    tensor = value.detach()
    chunk_rows = _positive_int(rows_per_chunk, "hash rows_per_chunk")
    digest = hashlib.sha256()
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(_dtype_name(tensor).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    if tensor.ndim == 0:
        blocks = (tensor.reshape(1),)
    else:
        blocks = (
            tensor[start : min(start + chunk_rows, int(tensor.shape[0]))]
            for start in range(0, int(tensor.shape[0]), chunk_rows)
        )
    for block in blocks:
        cpu = block.to(device="cpu").contiguous()
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _csr_checks(
    offsets: Tensor,
    ids: Tensor,
    *,
    row_count: int,
    target_row_count: int,
    name: str,
) -> None:
    if (
        offsets.device.type != "cpu"
        or offsets.dtype != torch.int64
        or offsets.ndim != 1
        or int(offsets.numel()) != row_count + 1
        or ids.device.type != "cpu"
        or ids.dtype != torch.int64
        or ids.ndim != 1
    ):
        raise Lfm2SparseSemanticCodebookError(
            f"{name} CSR tensors must be CPU int64 vectors."
        )
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != int(ids.numel())
        or bool((offsets[1:] < offsets[:-1]).any())
        or bool((ids < 0).any())
        or bool((ids >= target_row_count).any())
    ):
        raise Lfm2SparseSemanticCodebookError(f"{name} CSR bounds are invalid.")
    for row in range(row_count):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        selected = ids[start:stop]
        if int(torch.unique(selected).numel()) != int(selected.numel()):
            raise Lfm2SparseSemanticCodebookError(
                f"{name} row {row} contains duplicate target IDs."
            )


def _direct_inverse(mapping: Tensor, donor_row_count: int) -> Tensor:
    inverse = torch.full((donor_row_count,), -1, dtype=torch.int64)
    inverse[mapping] = torch.arange(int(mapping.numel()), dtype=torch.int64)
    return inverse


def _progress_emitter(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    phase: str,
) -> tuple[Callable[..., None], Callable[[], tuple[int, str | None]]]:
    if callback is not None and not callable(callback):
        raise TypeError("progress_callback must be callable or None.")
    state: dict[str, Any] = {"index": 0, "previous": None}

    def emit(event: str, **fields: Any) -> None:
        body = {
            "schema": LFM2_SPARSE_SEMANTIC_PROGRESS_SCHEMA,
            "phase": phase,
            "event": event,
            "progress_index": state["index"],
            "previous_progress_sha256": state["previous"],
            **_plain_json(fields),
        }
        record = _sealed(body, "progress_sha256")
        state["index"] += 1
        state["previous"] = record["progress_sha256"]
        if callback is not None:
            callback(MappingProxyType(record))

    def status() -> tuple[int, str | None]:
        return int(state["index"]), state["previous"]

    return emit, status


@dataclass(frozen=True, slots=True)
class Lfm2SparseSemanticCandidateChunkCheckpoint:
    candidate_offsets: Tensor
    candidate_target_ids: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SPARSE_SEMANTIC_CANDIDATE_CHUNK_SCHEMA,
            field="checkpoint_sha256",
            name="candidate chunk checkpoint",
        )
        row_start = _nonnegative_int(value.get("donor_row_start"), "donor_row_start")
        row_stop = _positive_int(value.get("donor_row_stop"), "donor_row_stop")
        if row_stop <= row_start:
            raise Lfm2SparseSemanticCodebookError("Candidate chunk row bounds are invalid.")
        offsets = self.candidate_offsets.detach().to(device="cpu").contiguous()
        ids = self.candidate_target_ids.detach().to(device="cpu").contiguous()
        _csr_checks(
            offsets,
            ids,
            row_count=row_stop - row_start,
            target_row_count=_positive_int(value.get("target_row_count"), "target_row_count"),
            name="candidate chunk",
        )
        if (
            value.get("candidate_offsets_sha256") != _tensor_sha256(offsets)
            or value.get("candidate_target_ids_sha256") != _tensor_sha256(ids)
            or value.get("candidate_entry_count") != int(ids.numel())
            or value.get("operation_audit") != _PROHIBITED_OPERATION_AUDIT
        ):
            raise Lfm2SparseSemanticCodebookError(
                "Candidate chunk checkpoint tensor binding is invalid."
            )
        object.__setattr__(self, "candidate_offsets", offsets.clone())
        object.__setattr__(self, "candidate_target_ids", ids.clone())
        object.__setattr__(self, "receipt", _deep_freeze(value))

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2SparseSemanticCandidateInventory:
    candidate_offsets: Tensor
    candidate_target_ids: Tensor
    target_to_donor: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SPARSE_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA,
            field="artifact_sha256",
            name="candidate inventory",
        )
        donor_count = _positive_int(value.get("donor_row_count"), "donor_row_count")
        target_count = _positive_int(value.get("target_row_count"), "target_row_count")
        offsets = self.candidate_offsets.detach().to(device="cpu").contiguous()
        ids = self.candidate_target_ids.detach().to(device="cpu").contiguous()
        mapping = self.target_to_donor.detach().to(device="cpu").contiguous()
        _csr_checks(
            offsets,
            ids,
            row_count=donor_count,
            target_row_count=target_count,
            name="candidate inventory",
        )
        if mapping.dtype != torch.int64 or mapping.ndim != 1:
            raise Lfm2SparseSemanticCodebookError(
                "Candidate inventory mapping must be CPU int64."
            )
        mapping = _validated_mapping(
            mapping,
            target_row_count=target_count,
            donor_row_count=donor_count,
        )
        row_diagnostics = value.get("row_diagnostics")
        row_diagnostics_sha = (
            canonical_sha256(row_diagnostics)
            if isinstance(row_diagnostics, list)
            else None
        )
        expected_content_sha = canonical_sha256(
            {
                "algorithm": value.get("algorithm"),
                "phase_cache_key": value.get("phase_cache_key"),
                "target_to_donor_sha256": _tensor_sha256(mapping),
                "candidate_offsets_sha256": _tensor_sha256(offsets),
                "candidate_target_ids_sha256": _tensor_sha256(ids),
                "row_diagnostics_sha256": row_diagnostics_sha,
            }
        )
        if (
            value.get("candidate_offsets_sha256") != _tensor_sha256(offsets)
            or value.get("candidate_target_ids_sha256") != _tensor_sha256(ids)
            or value.get("target_to_donor_sha256") != _tensor_sha256(mapping)
            or value.get("candidate_entry_count") != int(ids.numel())
            or value.get("row_diagnostics_sha256") != row_diagnostics_sha
            or value.get("candidate_content_sha256") != expected_content_sha
            or value.get("operation_audit", {}).get("training_steps") != 0
            or value.get("operation_audit", {}).get("global_map_solve_calls") != 0
        ):
            raise Lfm2SparseSemanticCodebookError(
                "Candidate inventory tensor binding is invalid."
            )
        inverse = _direct_inverse(mapping, donor_count)
        for donor_id in mapping.tolist():
            start, stop = int(offsets[donor_id]), int(offsets[donor_id + 1])
            if stop - start != 1 or int(ids[start]) != int(inverse[donor_id]):
                raise Lfm2SparseSemanticCodebookError(
                    "Direct donor candidate rows must be exact one-hot identities."
                )
        object.__setattr__(self, "candidate_offsets", offsets.clone())
        object.__setattr__(self, "candidate_target_ids", ids.clone())
        object.__setattr__(self, "target_to_donor", mapping.clone())
        object.__setattr__(self, "receipt", _deep_freeze(value))

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    @property
    def content_sha256(self) -> str:
        return str(self.receipt["candidate_content_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2SparseSemanticCoefficientChunkCheckpoint:
    semantic_code_offsets: Tensor
    semantic_code_target_ids: Tensor
    semantic_code_coefficients: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SPARSE_SEMANTIC_COEFFICIENT_CHUNK_SCHEMA,
            field="checkpoint_sha256",
            name="coefficient chunk checkpoint",
        )
        row_start = _nonnegative_int(value.get("donor_row_start"), "donor_row_start")
        row_stop = _positive_int(value.get("donor_row_stop"), "donor_row_stop")
        target_count = _positive_int(value.get("target_row_count"), "target_row_count")
        if row_stop <= row_start:
            raise Lfm2SparseSemanticCodebookError("Coefficient chunk row bounds are invalid.")
        offsets = self.semantic_code_offsets.detach().to(device="cpu").contiguous()
        ids = self.semantic_code_target_ids.detach().to(device="cpu").contiguous()
        coefficients = self.semantic_code_coefficients.detach().to(device="cpu").contiguous()
        _csr_checks(
            offsets,
            ids,
            row_count=row_stop - row_start,
            target_row_count=target_count,
            name="coefficient chunk",
        )
        if (
            coefficients.dtype != torch.float32
            or coefficients.ndim != 1
            or int(coefficients.numel()) != int(ids.numel())
            or coefficients.requires_grad
            or not bool(torch.isfinite(coefficients).all())
            or value.get("semantic_code_offsets_sha256") != _tensor_sha256(offsets)
            or value.get("semantic_code_target_ids_sha256") != _tensor_sha256(ids)
            or value.get("semantic_code_coefficients_sha256")
            != _tensor_sha256(coefficients)
            or value.get("continuous_fold_coefficient_count") != int(coefficients.numel())
            or value.get("operation_audit") != _PROHIBITED_OPERATION_AUDIT
        ):
            raise Lfm2SparseSemanticCodebookError(
                "Coefficient chunk checkpoint tensor binding is invalid."
            )
        object.__setattr__(self, "semantic_code_offsets", offsets.clone())
        object.__setattr__(self, "semantic_code_target_ids", ids.clone())
        object.__setattr__(self, "semantic_code_coefficients", coefficients.clone())
        object.__setattr__(self, "receipt", _deep_freeze(value))

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2SparseSemanticCodebook:
    semantic_code_offsets: Tensor
    semantic_code_target_ids: Tensor
    semantic_code_coefficients: Tensor
    target_to_donor: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA,
            field="artifact_sha256",
            name="sparse semantic codebook",
        )
        donor_count = _positive_int(value.get("donor_row_count"), "donor_row_count")
        target_count = _positive_int(value.get("target_row_count"), "target_row_count")
        offsets = self.semantic_code_offsets.detach().to(device="cpu").contiguous()
        ids = self.semantic_code_target_ids.detach().to(device="cpu").contiguous()
        coefficients = self.semantic_code_coefficients.detach().to(device="cpu").contiguous()
        mapping = _validated_mapping(
            self.target_to_donor.detach().to(device="cpu").contiguous(),
            target_row_count=target_count,
            donor_row_count=donor_count,
        )
        _csr_checks(
            offsets,
            ids,
            row_count=donor_count,
            target_row_count=target_count,
            name="sparse semantic codebook",
        )
        if (
            coefficients.dtype != torch.float32
            or coefficients.ndim != 1
            or int(coefficients.numel()) != int(ids.numel())
            or coefficients.requires_grad
            or not bool(torch.isfinite(coefficients).all())
            or value.get("semantic_code_offsets_sha256") != _tensor_sha256(offsets)
            or value.get("semantic_code_target_ids_sha256") != _tensor_sha256(ids)
            or value.get("semantic_code_coefficients_sha256")
            != _tensor_sha256(coefficients)
            or value.get("target_to_donor_sha256") != _tensor_sha256(mapping)
            or value.get("continuous_fold_coefficient_count") != int(coefficients.numel())
            or value.get("continuous_fold_coefficient_bytes") != int(coefficients.numel()) * 4
            or value.get("integer_routing_values_are_structural_metadata") is not True
            or value.get("continuous_coefficients_are_folded_child_state") is not True
            or value.get("operation_audit", {}).get("training_steps") != 0
            or value.get("operation_audit", {}).get("global_map_solve_calls") != 0
        ):
            raise Lfm2SparseSemanticCodebookError(
                "Sparse semantic codebook tensor binding is invalid."
            )
        inverse = _direct_inverse(mapping, donor_count)
        for donor_id in mapping.tolist():
            start, stop = int(offsets[donor_id]), int(offsets[donor_id + 1])
            if (
                stop - start != 1
                or int(ids[start]) != int(inverse[donor_id])
                or coefficients[start].view(torch.int32).item()
                != torch.tensor(1.0, dtype=torch.float32).view(torch.int32).item()
            ):
                raise Lfm2SparseSemanticCodebookError(
                    "Direct donor semantic rows must be exact one-hot identities."
                )
        object.__setattr__(self, "semantic_code_offsets", offsets.clone())
        object.__setattr__(self, "semantic_code_target_ids", ids.clone())
        object.__setattr__(self, "semantic_code_coefficients", coefficients.clone())
        object.__setattr__(self, "target_to_donor", mapping.clone())
        object.__setattr__(self, "receipt", _deep_freeze(value))

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)

    def reconstruct(
        self,
        anchor_table: Tensor,
        donor_ids: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        """Reconstruct selected semantic rows without a dense donor-by-target map."""

        anchors = _validated_table(anchor_table, "anchor_table")
        target_count = int(self.receipt["target_row_count"])
        if int(anchors.shape[0]) != target_count:
            raise Lfm2SparseSemanticCodebookError(
                "anchor_table row count differs from this codebook."
            )
        donor_count = int(self.receipt["donor_row_count"])
        if donor_ids is None:
            selected = tuple(range(donor_count))
        elif isinstance(donor_ids, Tensor):
            if donor_ids.ndim != 1 or torch.is_floating_point(donor_ids):
                raise Lfm2SparseSemanticCodebookError("donor_ids must be integers.")
            selected = tuple(int(value) for value in donor_ids.to(device="cpu").tolist())
        else:
            if isinstance(donor_ids, (str, bytes)) or not isinstance(donor_ids, Sequence):
                raise Lfm2SparseSemanticCodebookError("donor_ids must be integers.")
            selected = tuple(donor_ids)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= donor_count
            for value in selected
        ):
            raise Lfm2SparseSemanticCodebookError("donor_ids leave the codebook.")
        output = torch.zeros(
            (len(selected), int(anchors.shape[1])),
            device=anchors.device,
            dtype=anchors.dtype,
        )
        accumulation_dtype = (
            torch.float32
            if anchors.dtype in {torch.float16, torch.bfloat16}
            else anchors.dtype
        )
        with torch.no_grad():
            for output_row, donor_id in enumerate(selected):
                start = int(self.semantic_code_offsets[donor_id])
                stop = int(self.semantic_code_offsets[donor_id + 1])
                if start == stop:
                    continue
                ids = self.semantic_code_target_ids[start:stop].to(device=anchors.device)
                coefficients = self.semantic_code_coefficients[start:stop].to(
                    device=anchors.device,
                    dtype=accumulation_dtype,
                )
                value = (
                    anchors.index_select(0, ids).to(dtype=accumulation_dtype)
                    * coefficients.unsqueeze(1)
                ).sum(dim=0)
                output[output_row].copy_(value.to(dtype=anchors.dtype))
        return output


def _candidate_phase_body(
    *,
    donor_table_sha256: str,
    anchor_table_sha256: str,
    target_to_donor_sha256: str,
    donor_row_count: int,
    target_row_count: int,
    hidden_size: int,
    maximum_candidates: int,
    donor_rows_per_chunk: int,
    selection_device: torch.device,
    selection_dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "algorithm": _CANDIDATE_ALGORITHM,
        "donor_table_sha256": donor_table_sha256,
        "anchor_table_sha256": anchor_table_sha256,
        "target_to_donor_sha256": target_to_donor_sha256,
        "donor_row_count": donor_row_count,
        "target_row_count": target_row_count,
        "hidden_size": hidden_size,
        "maximum_candidates": maximum_candidates,
        "donor_rows_per_chunk": donor_rows_per_chunk,
        "selection_placement": {
            "device": str(selection_device),
            "dtype": _dtype_name(selection_dtype),
        },
        "cosine_definition": "absolute-dot-of-safe-l2-normalized-rows",
        "signed_cosines_retained_for_row_diagnostics": True,
        "zero_norm_policy": "normalized-zero-row-has-zero-similarity",
        "candidate_tie_break": "higher-absolute-cosine-then-lower-target-id",
        "direct_anchor_policy": "explicit-target-to-donor-exact-one-candidate",
    }


def discover_lfm2_sparse_semantic_candidates(
    donor_table: Tensor,
    anchor_table: Tensor,
    target_to_donor: Tensor | Sequence[int],
    *,
    maximum_candidates: int = 32,
    donor_rows_per_chunk: int = 256,
    selection_device: str | torch.device = "cpu",
    selection_dtype: str | torch.dtype = torch.float32,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    candidate_chunk_loader: (
        Callable[
            [Mapping[str, Any]],
            Lfm2SparseSemanticCandidateChunkCheckpoint | None,
        ]
        | None
    ) = None,
    candidate_chunk_publisher: (
        Callable[[Lfm2SparseSemanticCandidateChunkCheckpoint], None] | None
    ) = None,
) -> Lfm2SparseSemanticCandidateInventory:
    """Discover and seal bounded cosine neighborhoods independent of active k."""

    donor = _validated_table(donor_table, "donor_table")
    anchors = _validated_table(anchor_table, "anchor_table")
    donor_count, hidden_size = map(int, donor.shape)
    target_count, anchor_hidden = map(int, anchors.shape)
    if anchor_hidden != hidden_size:
        raise Lfm2SparseSemanticCodebookError(
            "donor_table and anchor_table hidden sizes differ."
        )
    mapping = _validated_mapping(
        target_to_donor,
        target_row_count=target_count,
        donor_row_count=donor_count,
    )
    candidate_count = min(
        _positive_int(maximum_candidates, "maximum_candidates"), target_count
    )
    row_chunk = min(
        _positive_int(donor_rows_per_chunk, "donor_rows_per_chunk"), donor_count
    )
    select_device = _compute_device(selection_device, "selection_device")
    select_dtype = _compute_dtype(selection_dtype, "selection_dtype")
    if candidate_chunk_loader is not None and not callable(candidate_chunk_loader):
        raise TypeError("candidate_chunk_loader must be callable or None.")
    if candidate_chunk_publisher is not None and not callable(candidate_chunk_publisher):
        raise TypeError("candidate_chunk_publisher must be callable or None.")

    donor_sha = _tensor_sha256(donor, rows_per_chunk=row_chunk)
    anchor_sha = _tensor_sha256(anchors, rows_per_chunk=row_chunk)
    mapping_sha = _tensor_sha256(mapping, rows_per_chunk=row_chunk)
    phase_body = _candidate_phase_body(
        donor_table_sha256=donor_sha,
        anchor_table_sha256=anchor_sha,
        target_to_donor_sha256=mapping_sha,
        donor_row_count=donor_count,
        target_row_count=target_count,
        hidden_size=hidden_size,
        maximum_candidates=candidate_count,
        donor_rows_per_chunk=row_chunk,
        selection_device=select_device,
        selection_dtype=select_dtype,
    )
    phase_cache_key = canonical_sha256(phase_body)
    emit, progress_status = _progress_emitter(
        progress_callback, phase="candidate-discovery"
    )
    emit(
        "phase-start",
        phase_cache_key=phase_cache_key,
        donor_row_count=donor_count,
        target_row_count=target_count,
        maximum_candidates=candidate_count,
        donor_rows_per_chunk=row_chunk,
        selection_device=str(select_device),
        selection_dtype=_dtype_name(select_dtype),
    )

    direct = _direct_inverse(mapping, donor_count)
    offsets = [0]
    all_ids: list[int] = []
    row_diagnostics: list[dict[str, Any]] = []
    computed_chunks = 0
    reused_chunks = 0
    loader_calls = 0
    publisher_calls = 0
    similarity_matmul_calls = 0
    donor_zero_norm_rows = 0
    anchor_zero_norm_rows = 0
    with torch.no_grad():
        anchor_compute = anchors.to(device=select_device, dtype=select_dtype)
        _require_finite_compute(anchor_compute, "anchor table selection cast")
        anchor_norms = torch.linalg.vector_norm(anchor_compute, dim=1)
        _require_finite_compute(anchor_norms, "anchor cosine norms")
        anchor_zero_norm_rows = int((anchor_norms == 0).sum().item())
        safe_anchor_norms = torch.where(
            anchor_norms > 0, anchor_norms, torch.ones_like(anchor_norms)
        )
        normalized_anchors = anchor_compute / safe_anchor_norms.unsqueeze(1)
        for row_start in range(0, donor_count, row_chunk):
            row_stop = min(row_start + row_chunk, donor_count)
            request_body = {
                "schema": LFM2_SPARSE_SEMANTIC_CANDIDATE_REQUEST_SCHEMA,
                "phase_cache_key": phase_cache_key,
                "donor_row_start": row_start,
                "donor_row_stop": row_stop,
                "donor_table_sha256": donor_sha,
                "anchor_table_sha256": anchor_sha,
                "target_to_donor_sha256": mapping_sha,
                "maximum_candidates": candidate_count,
                "selection_device": str(select_device),
                "selection_dtype": _dtype_name(select_dtype),
            }
            request = _sealed(request_body, "request_sha256")
            checkpoint = None
            if candidate_chunk_loader is not None:
                loader_calls += 1
                checkpoint = candidate_chunk_loader(MappingProxyType(request))
                if checkpoint is not None and not isinstance(
                    checkpoint, Lfm2SparseSemanticCandidateChunkCheckpoint
                ):
                    raise Lfm2SparseSemanticCodebookError(
                        "candidate_chunk_loader returned an invalid checkpoint type."
                    )
            if checkpoint is not None:
                checkpoint_receipt = checkpoint.as_dict()
                if (
                    checkpoint_receipt.get("request_sha256")
                    != request["request_sha256"]
                    or checkpoint_receipt.get("phase_cache_key") != phase_cache_key
                    or checkpoint_receipt.get("donor_row_start") != row_start
                    or checkpoint_receipt.get("donor_row_stop") != row_stop
                ):
                    raise Lfm2SparseSemanticCodebookError(
                        "Loaded candidate chunk does not match the exact request."
                    )
                local_offsets = checkpoint.candidate_offsets
                local_ids = checkpoint.candidate_target_ids
                diagnostics = _plain_json(checkpoint_receipt["row_diagnostics"])
                reused_chunks += 1
                source = "reused-authenticated-checkpoint"
            else:
                chunk_donor = donor[row_start:row_stop].to(
                    device=select_device, dtype=select_dtype
                )
                _require_finite_compute(chunk_donor, "donor selection chunk cast")
                norms = torch.linalg.vector_norm(chunk_donor, dim=1)
                _require_finite_compute(norms, "donor cosine norms")
                donor_zero_norm_rows += int((norms == 0).sum().item())
                safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
                normalized_donor = chunk_donor / safe_norms.unsqueeze(1)
                similarities = normalized_donor @ normalized_anchors.transpose(0, 1)
                _require_finite_compute(similarities, "candidate cosine similarities")
                candidate_scores = similarities.abs()
                similarity_matmul_calls += 1
                # Target IDs enter the matrix in ascending order.  Stable sort
                # therefore makes exact cosine ties choose lower target IDs.
                ranked = torch.argsort(
                    candidate_scores, dim=1, descending=True, stable=True
                )[:, :candidate_count]
                selected_signed_cpu = torch.gather(
                    similarities, 1, ranked
                ).to(device="cpu")
                selected_absolute_cpu = torch.gather(
                    candidate_scores, 1, ranked
                ).to(device="cpu")
                ranked_cpu = ranked.to(device="cpu")
                zero_norm_cpu = (norms == 0).to(device="cpu")
                local_offsets_values = [0]
                local_id_values: list[int] = []
                diagnostics = []
                for local_row, donor_id in enumerate(range(row_start, row_stop)):
                    direct_target = int(direct[donor_id])
                    if direct_target >= 0:
                        selected = [direct_target]
                        diagnostic = {
                            "donor_id": donor_id,
                            "direct_anchor": True,
                            "candidate_count": 1,
                            "nearest_candidate_cosine": 1.0,
                            "candidate_cutoff_cosine": 1.0,
                            "nearest_candidate_absolute_cosine": 1.0,
                            "candidate_cutoff_absolute_cosine": 1.0,
                            "donor_zero_norm": bool(zero_norm_cpu[local_row].item()),
                        }
                    else:
                        selected_tensor = ranked_cpu[local_row]
                        selected = [int(value) for value in selected_tensor.tolist()]
                        selected_cosines = selected_signed_cpu[local_row]
                        selected_scores = selected_absolute_cpu[local_row]
                        diagnostic = {
                            "donor_id": donor_id,
                            "direct_anchor": False,
                            "candidate_count": len(selected),
                            "nearest_candidate_cosine": float(
                                selected_cosines[0].item()
                            ),
                            "candidate_cutoff_cosine": float(
                                selected_cosines[-1].item()
                            ),
                            "nearest_candidate_absolute_cosine": float(
                                selected_scores[0].item()
                            ),
                            "candidate_cutoff_absolute_cosine": float(
                                selected_scores[-1].item()
                            ),
                            "donor_zero_norm": bool(zero_norm_cpu[local_row].item()),
                        }
                    local_id_values.extend(selected)
                    local_offsets_values.append(len(local_id_values))
                    diagnostics.append(diagnostic)
                local_offsets = torch.tensor(local_offsets_values, dtype=torch.int64)
                local_ids = torch.tensor(local_id_values, dtype=torch.int64)
                checkpoint_body = {
                    "schema": LFM2_SPARSE_SEMANTIC_CANDIDATE_CHUNK_SCHEMA,
                    "phase_cache_key": phase_cache_key,
                    "request_sha256": request["request_sha256"],
                    "donor_row_start": row_start,
                    "donor_row_stop": row_stop,
                    "target_row_count": target_count,
                    "candidate_entry_count": int(local_ids.numel()),
                    "candidate_offsets_sha256": _tensor_sha256(local_offsets),
                    "candidate_target_ids_sha256": _tensor_sha256(local_ids),
                    "row_diagnostics": diagnostics,
                    "row_diagnostics_sha256": canonical_sha256(diagnostics),
                    "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
                }
                checkpoint = Lfm2SparseSemanticCandidateChunkCheckpoint(
                    local_offsets,
                    local_ids,
                    _sealed(checkpoint_body, "checkpoint_sha256"),
                )
                computed_chunks += 1
                source = "computed-cosine-neighborhood"
                if candidate_chunk_publisher is not None:
                    candidate_chunk_publisher(checkpoint)
                    publisher_calls += 1
            base = len(all_ids)
            all_ids.extend(int(value) for value in local_ids.tolist())
            offsets.extend(base + int(value) for value in local_offsets[1:].tolist())
            row_diagnostics.extend(diagnostics)
            emit(
                "row-chunk-complete",
                phase_cache_key=phase_cache_key,
                donor_row_start=row_start,
                donor_row_stop=row_stop,
                completed_donor_rows=row_stop,
                total_donor_rows=donor_count,
                chunk_source=source,
                candidate_entry_count=int(local_ids.numel()),
                candidate_offsets_sha256=_tensor_sha256(local_offsets),
                candidate_target_ids_sha256=_tensor_sha256(local_ids),
            )
    candidate_offsets = torch.tensor(offsets, dtype=torch.int64)
    candidate_ids = torch.tensor(all_ids, dtype=torch.int64)
    progress_count, progress_final = progress_status()
    candidate_offsets_sha = _tensor_sha256(candidate_offsets)
    candidate_ids_sha = _tensor_sha256(candidate_ids)
    row_diagnostics_sha = canonical_sha256(row_diagnostics)
    candidate_content_sha = canonical_sha256(
        {
            "algorithm": _CANDIDATE_ALGORITHM,
            "phase_cache_key": phase_cache_key,
            "target_to_donor_sha256": mapping_sha,
            "candidate_offsets_sha256": candidate_offsets_sha,
            "candidate_target_ids_sha256": candidate_ids_sha,
            "row_diagnostics_sha256": row_diagnostics_sha,
        }
    )
    analytic_audit = {
        "anchor_normalization_calls": 1,
        "donor_normalization_chunk_calls": computed_chunks,
        "candidate_similarity_chunk_matmul_calls": similarity_matmul_calls,
        "candidate_chunk_loader_calls": loader_calls,
        "candidate_chunk_publisher_calls": publisher_calls,
    }
    body = {
        "schema": LFM2_SPARSE_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA,
        "status": "complete-k-independent-bounded-local-neighborhoods",
        "algorithm": _CANDIDATE_ALGORITHM,
        "phase_cache_key": phase_cache_key,
        "donor_table_sha256": donor_sha,
        "anchor_table_sha256": anchor_sha,
        "target_to_donor_sha256": mapping_sha,
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "direct_anchor_count": target_count,
        "non_direct_donor_row_count": donor_count - target_count,
        "maximum_candidates": candidate_count,
        "donor_rows_per_chunk": row_chunk,
        "candidate_content_sha256": candidate_content_sha,
        "candidate_offsets_sha256": candidate_offsets_sha,
        "candidate_target_ids_sha256": candidate_ids_sha,
        "candidate_entry_count": int(candidate_ids.numel()),
        "candidate_offset_count": int(candidate_offsets.numel()),
        "candidate_offset_bytes": int(candidate_offsets.numel()) * 8,
        "candidate_target_id_bytes": int(candidate_ids.numel()) * 8,
        "target_to_donor_bytes": int(mapping.numel()) * 8,
        "integer_candidate_values_are_reusable_structural_evidence": True,
        "candidate_phase_is_independent_of_active_k_and_ridges": True,
        "selection_placement": {
            "device": str(select_device),
            "dtype": _dtype_name(select_dtype),
        },
        "cosine_definition": "absolute-dot-of-safe-l2-normalized-rows",
        "signed_cosines_retained_for_row_diagnostics": True,
        "candidate_tie_break": "higher-absolute-cosine-then-lower-target-id",
        "zero_norm_policy": "normalized-zero-row-has-zero-similarity",
        "donor_zero_norm_row_count_computed_chunks": donor_zero_norm_rows,
        "anchor_zero_norm_row_count": anchor_zero_norm_rows,
        "row_diagnostics": row_diagnostics,
        "row_diagnostics_sha256": row_diagnostics_sha,
        "computed_chunk_count": computed_chunks,
        "checkpoint_reused_chunk_count": reused_chunks,
        "checkpoint_loader_supplied": candidate_chunk_loader is not None,
        "checkpoint_loader_invocation_count": loader_calls,
        "checkpoint_publisher_supplied": candidate_chunk_publisher is not None,
        "checkpoint_publisher_invocation_count": publisher_calls,
        "progress_record_count": progress_count,
        "progress_final_sha256": progress_final,
        "progress_callback_supplied": progress_callback is not None,
        "analytic_operation_audit": analytic_audit,
        "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
    }
    return Lfm2SparseSemanticCandidateInventory(
        candidate_offsets,
        candidate_ids,
        mapping,
        _sealed(body, "artifact_sha256"),
    )


def _ridge_solve_batched(
    basis: Tensor,
    targets: Tensor,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    """Solve a local ridge objective for each row in one bounded batch."""

    batch, width, _hidden = map(int, basis.shape)
    gram = torch.bmm(basis, basis.transpose(1, 2))
    diagonal = torch.arange(width, device=basis.device)
    gram[:, diagonal, diagonal] += ridge
    rhs = torch.bmm(basis, targets.unsqueeze(2)).squeeze(2)
    _require_finite_compute(gram, "local regularized Gram")
    _require_finite_compute(rhs, "local ridge right-hand side")
    try:
        solution = torch.linalg.solve(gram, rhs.unsqueeze(2)).squeeze(2)
    except RuntimeError as exc:
        raise Lfm2SparseSemanticCodebookError(
            "A bounded local ridge solve failed at the requested placement."
        ) from exc
    _require_finite_compute(solution, "local ridge solution")
    # This inexpensive SPD diagonal-ratio diagnostic is explicit; it is not
    # mislabeled as a spectral condition number.
    gram_diagonal = torch.diagonal(gram, dim1=1, dim2=2)
    condition_proxy = gram_diagonal.max(dim=1).values / gram_diagonal.min(
        dim=1
    ).values.clamp_min(torch.finfo(gram.dtype).tiny)
    _require_finite_compute(condition_proxy, "local Gram condition proxy")
    if solution.shape != (batch, width):
        raise AssertionError("Unexpected local ridge solution shape.")
    return solution, condition_proxy


def _coefficient_phase_body(
    *,
    candidate_content_sha256: str,
    donor_table_sha256: str,
    anchor_table_sha256: str,
    target_to_donor_sha256: str,
    active_k: int,
    candidate_ridge: float,
    refit_ridge: float,
    donor_rows_per_chunk: int,
    solve_device: torch.device,
    solve_dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "algorithm": _SEMANTIC_ALGORITHM,
        "candidate_content_sha256": candidate_content_sha256,
        "donor_table_sha256": donor_table_sha256,
        "anchor_table_sha256": anchor_table_sha256,
        "target_to_donor_sha256": target_to_donor_sha256,
        "active_k": active_k,
        "candidate_ridge": candidate_ridge,
        "refit_ridge": refit_ridge,
        "donor_rows_per_chunk": donor_rows_per_chunk,
        "solve_placement": {
            "device": str(solve_device),
            "dtype": _dtype_name(solve_dtype),
        },
        "publication_placement": {"device": "cpu", "dtype": "float32"},
        "active_tie_break": "larger-absolute-preliminary-weight-then-lower-target-id",
        "published_row_order": "ascending-target-id",
    }


def compile_lfm2_sparse_semantic_codebook(
    donor_table: Tensor,
    anchor_table: Tensor,
    target_to_donor: Tensor | Sequence[int],
    *,
    candidate_inventory: Lfm2SparseSemanticCandidateInventory | None = None,
    maximum_candidates: int = 32,
    active_k: int = 8,
    candidate_ridge: float = 1e-5,
    refit_ridge: float = 1e-5,
    donor_rows_per_chunk: int = 256,
    selection_device: str | torch.device = "cpu",
    selection_dtype: str | torch.dtype = torch.float32,
    solve_device: str | torch.device = "cpu",
    solve_dtype: str | torch.dtype = torch.float64,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    candidate_chunk_loader: (
        Callable[
            [Mapping[str, Any]],
            Lfm2SparseSemanticCandidateChunkCheckpoint | None,
        ]
        | None
    ) = None,
    candidate_chunk_publisher: (
        Callable[[Lfm2SparseSemanticCandidateChunkCheckpoint], None] | None
    ) = None,
    coefficient_chunk_loader: (
        Callable[
            [Mapping[str, Any]],
            Lfm2SparseSemanticCoefficientChunkCheckpoint | None,
        ]
        | None
    ) = None,
    coefficient_chunk_publisher: (
        Callable[[Lfm2SparseSemanticCoefficientChunkCheckpoint], None] | None
    ) = None,
) -> Lfm2SparseSemanticCodebook:
    """Compile the sparse semantic channel over a planted anchor table.

    Candidate discovery and coefficient fitting have independent placements.
    In particular, discovery may run on CUDA FP32 while the small batched ridge
    systems run on CPU FP64.  Nothing implicitly selects CUDA FP64.
    """

    donor = _validated_table(donor_table, "donor_table")
    anchors = _validated_table(anchor_table, "anchor_table")
    donor_count, hidden_size = map(int, donor.shape)
    target_count, anchor_hidden = map(int, anchors.shape)
    if hidden_size != anchor_hidden:
        raise Lfm2SparseSemanticCodebookError(
            "donor_table and anchor_table hidden sizes differ."
        )
    mapping = _validated_mapping(
        target_to_donor,
        target_row_count=target_count,
        donor_row_count=donor_count,
    )
    requested_candidates = min(
        _positive_int(maximum_candidates, "maximum_candidates"), target_count
    )
    retained_k = _positive_int(active_k, "active_k")
    if retained_k > requested_candidates:
        raise Lfm2SparseSemanticCodebookError(
            "active_k cannot exceed maximum_candidates."
        )
    first_ridge = _positive_float(candidate_ridge, "candidate_ridge")
    second_ridge = _positive_float(refit_ridge, "refit_ridge")
    row_chunk = min(
        _positive_int(donor_rows_per_chunk, "donor_rows_per_chunk"), donor_count
    )
    fit_device = _compute_device(solve_device, "solve_device")
    fit_dtype = _compute_dtype(solve_dtype, "solve_dtype")
    if coefficient_chunk_loader is not None and not callable(coefficient_chunk_loader):
        raise TypeError("coefficient_chunk_loader must be callable or None.")
    if coefficient_chunk_publisher is not None and not callable(
        coefficient_chunk_publisher
    ):
        raise TypeError("coefficient_chunk_publisher must be callable or None.")
    if candidate_inventory is not None and (
        candidate_chunk_loader is not None or candidate_chunk_publisher is not None
    ):
        raise Lfm2SparseSemanticCodebookError(
            "Candidate chunk callbacks cannot accompany an already-built inventory."
        )

    donor_sha = _tensor_sha256(donor, rows_per_chunk=row_chunk)
    anchor_sha = _tensor_sha256(anchors, rows_per_chunk=row_chunk)
    mapping_sha = _tensor_sha256(mapping, rows_per_chunk=row_chunk)
    if candidate_inventory is None:
        inventory = discover_lfm2_sparse_semantic_candidates(
            donor,
            anchors,
            mapping,
            maximum_candidates=requested_candidates,
            donor_rows_per_chunk=row_chunk,
            selection_device=selection_device,
            selection_dtype=selection_dtype,
            progress_callback=progress_callback,
            candidate_chunk_loader=candidate_chunk_loader,
            candidate_chunk_publisher=candidate_chunk_publisher,
        )
    elif not isinstance(candidate_inventory, Lfm2SparseSemanticCandidateInventory):
        raise TypeError(
            "candidate_inventory must be Lfm2SparseSemanticCandidateInventory or None."
        )
    else:
        inventory = candidate_inventory
    inventory_receipt = inventory.as_dict()
    if (
        inventory_receipt.get("donor_table_sha256") != donor_sha
        or inventory_receipt.get("anchor_table_sha256") != anchor_sha
        or inventory_receipt.get("target_to_donor_sha256") != mapping_sha
        or inventory_receipt.get("donor_row_count") != donor_count
        or inventory_receipt.get("target_row_count") != target_count
        or inventory_receipt.get("hidden_size") != hidden_size
        or inventory_receipt.get("maximum_candidates") != requested_candidates
        or not torch.equal(inventory.target_to_donor, mapping)
    ):
        raise Lfm2SparseSemanticCodebookError(
            "Candidate inventory does not authenticate these exact tables and mapping."
        )
    candidate_diagnostics = inventory_receipt.get("row_diagnostics")
    if not isinstance(candidate_diagnostics, list) or len(candidate_diagnostics) != donor_count:
        raise Lfm2SparseSemanticCodebookError(
            "Candidate inventory lacks complete row diagnostics."
        )
    direct_donor_ids = frozenset(int(value) for value in mapping.tolist())
    for donor_id in range(donor_count):
        start = int(inventory.candidate_offsets[donor_id])
        stop = int(inventory.candidate_offsets[donor_id + 1])
        expected = 1 if donor_id in direct_donor_ids else requested_candidates
        if stop - start != expected:
            raise Lfm2SparseSemanticCodebookError(
                "Candidate inventory row width differs from its sealed policy."
            )

    phase_body = _coefficient_phase_body(
        candidate_content_sha256=str(inventory_receipt["candidate_content_sha256"]),
        donor_table_sha256=donor_sha,
        anchor_table_sha256=anchor_sha,
        target_to_donor_sha256=mapping_sha,
        active_k=retained_k,
        candidate_ridge=first_ridge,
        refit_ridge=second_ridge,
        donor_rows_per_chunk=row_chunk,
        solve_device=fit_device,
        solve_dtype=fit_dtype,
    )
    phase_cache_key = canonical_sha256(phase_body)
    emit, progress_status = _progress_emitter(
        progress_callback, phase="active-k-local-ridge-refit"
    )
    emit(
        "phase-start",
        phase_cache_key=phase_cache_key,
        candidate_artifact_sha256=inventory.artifact_sha256,
        candidate_content_sha256=inventory_receipt["candidate_content_sha256"],
        active_k=retained_k,
        candidate_ridge=first_ridge,
        refit_ridge=second_ridge,
        solve_device=str(fit_device),
        solve_dtype=_dtype_name(fit_dtype),
        publication_device="cpu",
        publication_dtype="float32",
    )

    direct = _direct_inverse(mapping, donor_count)
    offsets = [0]
    all_ids: list[int] = []
    all_coefficients: list[float] = []
    row_diagnostics: list[dict[str, Any]] = []
    computed_chunks = 0
    reused_chunks = 0
    loader_calls = 0
    publisher_calls = 0
    preliminary_solve_calls = 0
    active_refit_solve_calls = 0
    residual_sum_squares = 0.0
    source_sum_squares = 0.0
    maximum_row_residual_l2 = 0.0
    maximum_absolute_residual = 0.0
    cosine_sum = 0.0

    with torch.no_grad():
        anchor_compute = anchors.to(device=fit_device, dtype=fit_dtype)
        _require_finite_compute(anchor_compute, "anchor table solve cast")
        for row_start in range(0, donor_count, row_chunk):
            row_stop = min(row_start + row_chunk, donor_count)
            request_body = {
                "schema": LFM2_SPARSE_SEMANTIC_COEFFICIENT_REQUEST_SCHEMA,
                "phase_cache_key": phase_cache_key,
                "candidate_content_sha256": inventory_receipt[
                    "candidate_content_sha256"
                ],
                "donor_row_start": row_start,
                "donor_row_stop": row_stop,
                "donor_table_sha256": donor_sha,
                "anchor_table_sha256": anchor_sha,
                "target_to_donor_sha256": mapping_sha,
                "active_k": retained_k,
                "candidate_ridge": first_ridge,
                "refit_ridge": second_ridge,
                "solve_device": str(fit_device),
                "solve_dtype": _dtype_name(fit_dtype),
                "publication_device": "cpu",
                "publication_dtype": "float32",
            }
            request = _sealed(request_body, "request_sha256")
            checkpoint = None
            if coefficient_chunk_loader is not None:
                loader_calls += 1
                checkpoint = coefficient_chunk_loader(MappingProxyType(request))
                if checkpoint is not None and not isinstance(
                    checkpoint, Lfm2SparseSemanticCoefficientChunkCheckpoint
                ):
                    raise Lfm2SparseSemanticCodebookError(
                        "coefficient_chunk_loader returned an invalid checkpoint type."
                    )
            if checkpoint is not None:
                checkpoint_receipt = checkpoint.as_dict()
                if (
                    checkpoint_receipt.get("request_sha256")
                    != request["request_sha256"]
                    or checkpoint_receipt.get("phase_cache_key") != phase_cache_key
                    or checkpoint_receipt.get("donor_row_start") != row_start
                    or checkpoint_receipt.get("donor_row_stop") != row_stop
                ):
                    raise Lfm2SparseSemanticCodebookError(
                        "Loaded coefficient chunk does not match the exact request."
                    )
                local_offsets = checkpoint.semantic_code_offsets
                local_ids = checkpoint.semantic_code_target_ids
                local_coefficients = checkpoint.semantic_code_coefficients
                diagnostics = _plain_json(checkpoint_receipt["row_diagnostics"])
                chunk_residual_sum_squares = float(
                    checkpoint_receipt["reconstruction_residual_sum_squares"]
                )
                chunk_source_sum_squares = float(
                    checkpoint_receipt["source_sum_squares"]
                )
                chunk_max_row_residual = float(
                    checkpoint_receipt["maximum_row_reconstruction_l2_error"]
                )
                chunk_max_absolute = float(
                    checkpoint_receipt["maximum_absolute_reconstruction_residual"]
                )
                chunk_cosine_sum = float(
                    checkpoint_receipt["sum_reconstruction_cosine"]
                )
                reused_chunks += 1
                source = "reused-authenticated-checkpoint"
            else:
                local_offsets_values = [0]
                local_ids_values: list[int] = []
                local_coefficient_values: list[float] = []
                diagnostics_by_donor: dict[int, dict[str, Any]] = {}
                direct_ids_in_chunk = [
                    donor_id
                    for donor_id in range(row_start, row_stop)
                    if donor_id in direct_donor_ids
                ]
                if direct_ids_in_chunk:
                    direct_target_ids = torch.tensor(
                        [int(direct[donor_id]) for donor_id in direct_ids_in_chunk],
                        device=fit_device,
                        dtype=torch.int64,
                    )
                    direct_observed = donor[
                        torch.tensor(direct_ids_in_chunk, device=donor.device)
                    ].to(device=fit_device, dtype=fit_dtype)
                    direct_reconstructed = anchor_compute.index_select(
                        0, direct_target_ids
                    )
                    direct_residual = direct_reconstructed - direct_observed
                    direct_residual_norm = torch.linalg.vector_norm(
                        direct_residual, dim=1
                    )
                    direct_observed_norm = torch.linalg.vector_norm(
                        direct_observed, dim=1
                    )
                    direct_reconstructed_norm = torch.linalg.vector_norm(
                        direct_reconstructed, dim=1
                    )
                    direct_denominator = (
                        direct_observed_norm * direct_reconstructed_norm
                    )
                    direct_cosine = torch.where(
                        direct_denominator > 0,
                        (direct_observed * direct_reconstructed).sum(dim=1)
                        / direct_denominator,
                        torch.where(
                            (direct_observed_norm == 0)
                            & (direct_reconstructed_norm == 0),
                            torch.ones_like(direct_denominator),
                            torch.zeros_like(direct_denominator),
                        ),
                    )
                    direct_metrics_cpu = torch.stack(
                        (
                            direct_residual_norm,
                            direct_observed_norm,
                            direct_cosine,
                            direct_residual.square().sum(dim=1),
                            direct_observed.square().sum(dim=1),
                            direct_residual.abs().max(dim=1).values,
                        ),
                        dim=1,
                    ).to(device="cpu")
                    for local_index, donor_id in enumerate(direct_ids_in_chunk):
                        target_id = int(direct[donor_id])
                        values = direct_metrics_cpu[local_index]
                        residual_norm_value = float(values[0].item())
                        observed_norm_value = float(values[1].item())
                        diagnostics_by_donor[donor_id] = {
                            "donor_id": donor_id,
                            "direct_anchor": True,
                            "nearest_candidate_cosine": 1.0,
                            "candidate_cutoff_cosine": 1.0,
                            "nearest_candidate_absolute_cosine": 1.0,
                            "candidate_cutoff_absolute_cosine": 1.0,
                            "preliminary_reconstruction_l2_error": (
                                residual_norm_value
                            ),
                            "published_reconstruction_l2_error": (
                                residual_norm_value
                            ),
                            "published_relative_reconstruction_l2_error": (
                                residual_norm_value
                                / max(
                                    observed_norm_value,
                                    torch.finfo(fit_dtype).tiny,
                                )
                            ),
                            "published_reconstruction_cosine": float(
                                values[2].item()
                            ),
                            "preliminary_gram_diagonal_condition_proxy": 1.0,
                            "refit_gram_diagonal_condition_proxy": 1.0,
                            "active_target_ids": [target_id],
                            "active_coefficient_count": 1,
                            "_coefficients": [1.0],
                            "_residual_square": float(values[3].item()),
                            "_source_square": float(values[4].item()),
                            "_maximum_absolute": float(values[5].item()),
                        }
                non_direct_ids = [
                    donor_id
                    for donor_id in range(row_start, row_stop)
                    if donor_id not in direct_donor_ids
                ]
                if non_direct_ids:
                    candidate_rows = []
                    for donor_id in non_direct_ids:
                        start = int(inventory.candidate_offsets[donor_id])
                        stop = int(inventory.candidate_offsets[donor_id + 1])
                        candidate_rows.append(
                            inventory.candidate_target_ids[start:stop]
                        )
                    candidate_ids = torch.stack(candidate_rows).to(
                        device=fit_device, dtype=torch.int64
                    )
                    basis = anchor_compute.index_select(
                        0, candidate_ids.reshape(-1)
                    ).reshape(len(non_direct_ids), requested_candidates, hidden_size)
                    target_rows = donor[
                        torch.tensor(non_direct_ids, device=donor.device)
                    ].to(device=fit_device, dtype=fit_dtype)
                    preliminary, preliminary_condition = _ridge_solve_batched(
                        basis, target_rows, ridge=first_ridge
                    )
                    preliminary_solve_calls += len(non_direct_ids)
                    preliminary_reconstruction = torch.bmm(
                        preliminary.unsqueeze(1), basis
                    ).squeeze(1)
                    preliminary_residual = preliminary_reconstruction - target_rows

                    # Stable two-key order: first establish ascending target ID,
                    # then stable-sort absolute coefficient magnitude descending.
                    target_order = torch.argsort(candidate_ids, dim=1, stable=True)
                    ids_by_target = candidate_ids.gather(1, target_order)
                    preliminary_by_target = preliminary.gather(1, target_order)
                    magnitude_order = torch.argsort(
                        preliminary_by_target.abs(),
                        dim=1,
                        descending=True,
                        stable=True,
                    )[:, :retained_k]
                    active_ids = ids_by_target.gather(1, magnitude_order)
                    # Canonical published CSR rows are ascending by target ID.
                    canonical_order = torch.argsort(active_ids, dim=1, stable=True)
                    active_ids = active_ids.gather(1, canonical_order)
                    active_basis = anchor_compute.index_select(
                        0, active_ids.reshape(-1)
                    ).reshape(len(non_direct_ids), retained_k, hidden_size)
                    refitted, refit_condition = _ridge_solve_batched(
                        active_basis, target_rows, ridge=second_ridge
                    )
                    active_refit_solve_calls += len(non_direct_ids)
                    published_coefficients = refitted.to(dtype=torch.float32)
                    _require_finite_compute(
                        published_coefficients, "FP32 coefficient publication"
                    )
                    # Metrics authenticate the *published* FP32 state, widened
                    # back to the solve dtype, not an unpublishable FP64 ideal.
                    published_solve = published_coefficients.to(dtype=fit_dtype)
                    reconstruction = torch.bmm(
                        published_solve.unsqueeze(1), active_basis
                    ).squeeze(1)
                    residual = reconstruction - target_rows
                    source_norm = torch.linalg.vector_norm(target_rows, dim=1)
                    residual_norm = torch.linalg.vector_norm(residual, dim=1)
                    reconstruction_norm = torch.linalg.vector_norm(
                        reconstruction, dim=1
                    )
                    dot = (reconstruction * target_rows).sum(dim=1)
                    denominator = reconstruction_norm * source_norm
                    cosine = torch.where(
                        denominator > 0,
                        dot / denominator,
                        torch.where(
                            (reconstruction_norm == 0) & (source_norm == 0),
                            torch.ones_like(denominator),
                            torch.zeros_like(denominator),
                        ),
                    )
                    diagnostic_metrics_cpu = torch.stack(
                        (
                            torch.linalg.vector_norm(
                                preliminary_residual, dim=1
                            ),
                            residual_norm,
                            source_norm,
                            cosine,
                            preliminary_condition,
                            refit_condition,
                            residual.square().sum(dim=1),
                            target_rows.square().sum(dim=1),
                            residual.abs().max(dim=1).values,
                        ),
                        dim=1,
                    ).to(device="cpu")
                    active_ids_cpu = active_ids.to(device="cpu")
                    published_coefficients_cpu = published_coefficients.to(
                        device="cpu"
                    )
                    for local_index, donor_id in enumerate(non_direct_ids):
                        metrics = diagnostic_metrics_cpu[local_index]
                        ids_row = [
                            int(value)
                            for value in active_ids_cpu[local_index].tolist()
                        ]
                        coefficient_row = [
                            float(value)
                            for value in published_coefficients_cpu[
                                local_index
                            ].tolist()
                        ]
                        diagnostics_by_donor[donor_id] = {
                            "donor_id": donor_id,
                            "direct_anchor": False,
                            "nearest_candidate_cosine": float(
                                candidate_diagnostics[donor_id][
                                    "nearest_candidate_cosine"
                                ]
                            ),
                            "candidate_cutoff_cosine": float(
                                candidate_diagnostics[donor_id][
                                    "candidate_cutoff_cosine"
                                ]
                            ),
                            "nearest_candidate_absolute_cosine": float(
                                candidate_diagnostics[donor_id][
                                    "nearest_candidate_absolute_cosine"
                                ]
                            ),
                            "candidate_cutoff_absolute_cosine": float(
                                candidate_diagnostics[donor_id][
                                    "candidate_cutoff_absolute_cosine"
                                ]
                            ),
                            "preliminary_reconstruction_l2_error": float(
                                metrics[0].item()
                            ),
                            "published_reconstruction_l2_error": float(
                                metrics[1].item()
                            ),
                            "published_relative_reconstruction_l2_error": float(
                                (
                                    metrics[1]
                                    / metrics[2].clamp_min(
                                        torch.finfo(fit_dtype).tiny
                                    )
                                ).item()
                            ),
                            "published_reconstruction_cosine": float(
                                metrics[3].item()
                            ),
                            "preliminary_gram_diagonal_condition_proxy": float(
                                metrics[4].item()
                            ),
                            "refit_gram_diagonal_condition_proxy": float(
                                metrics[5].item()
                            ),
                            "active_target_ids": ids_row,
                            "active_coefficient_count": retained_k,
                        }
                        diagnostics_by_donor[donor_id]["_coefficients"] = coefficient_row
                        diagnostics_by_donor[donor_id]["_residual_square"] = float(
                            metrics[6].item()
                        )
                        diagnostics_by_donor[donor_id]["_source_square"] = float(
                            metrics[7].item()
                        )
                        diagnostics_by_donor[donor_id]["_maximum_absolute"] = float(
                            metrics[8].item()
                        )
                diagnostics = []
                chunk_residual_sum_squares = 0.0
                chunk_source_sum_squares = 0.0
                chunk_max_row_residual = 0.0
                chunk_max_absolute = 0.0
                chunk_cosine_sum = 0.0
                for donor_id in range(row_start, row_stop):
                    diagnostic = diagnostics_by_donor[donor_id]
                    coefficient_row = diagnostic.pop("_coefficients")
                    ids_row = list(diagnostic["active_target_ids"])
                    residual_norm_value = float(
                        diagnostic["published_reconstruction_l2_error"]
                    )
                    residual_square = float(diagnostic.pop("_residual_square"))
                    source_square = float(diagnostic.pop("_source_square"))
                    max_absolute = float(diagnostic.pop("_maximum_absolute"))
                    local_ids_values.extend(ids_row)
                    local_coefficient_values.extend(coefficient_row)
                    local_offsets_values.append(len(local_ids_values))
                    diagnostics.append(diagnostic)
                    chunk_residual_sum_squares += residual_square
                    chunk_source_sum_squares += source_square
                    chunk_max_row_residual = max(
                        chunk_max_row_residual, residual_norm_value
                    )
                    chunk_max_absolute = max(chunk_max_absolute, max_absolute)
                    chunk_cosine_sum += float(
                        diagnostic["published_reconstruction_cosine"]
                    )
                local_offsets = torch.tensor(local_offsets_values, dtype=torch.int64)
                local_ids = torch.tensor(local_ids_values, dtype=torch.int64)
                local_coefficients = torch.tensor(
                    local_coefficient_values, dtype=torch.float32
                )
                checkpoint_body = {
                    "schema": LFM2_SPARSE_SEMANTIC_COEFFICIENT_CHUNK_SCHEMA,
                    "phase_cache_key": phase_cache_key,
                    "request_sha256": request["request_sha256"],
                    "candidate_content_sha256": inventory_receipt[
                        "candidate_content_sha256"
                    ],
                    "donor_row_start": row_start,
                    "donor_row_stop": row_stop,
                    "target_row_count": target_count,
                    "semantic_code_offsets_sha256": _tensor_sha256(local_offsets),
                    "semantic_code_target_ids_sha256": _tensor_sha256(local_ids),
                    "semantic_code_coefficients_sha256": _tensor_sha256(
                        local_coefficients
                    ),
                    "continuous_fold_coefficient_count": int(
                        local_coefficients.numel()
                    ),
                    "reconstruction_residual_sum_squares": chunk_residual_sum_squares,
                    "source_sum_squares": chunk_source_sum_squares,
                    "maximum_row_reconstruction_l2_error": chunk_max_row_residual,
                    "maximum_absolute_reconstruction_residual": chunk_max_absolute,
                    "sum_reconstruction_cosine": chunk_cosine_sum,
                    "row_diagnostics": diagnostics,
                    "row_diagnostics_sha256": canonical_sha256(diagnostics),
                    "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
                }
                checkpoint = Lfm2SparseSemanticCoefficientChunkCheckpoint(
                    local_offsets,
                    local_ids,
                    local_coefficients,
                    _sealed(checkpoint_body, "checkpoint_sha256"),
                )
                computed_chunks += 1
                source = "computed-local-ridge-active-k-refit"
                if coefficient_chunk_publisher is not None:
                    coefficient_chunk_publisher(checkpoint)
                    publisher_calls += 1
            base = len(all_ids)
            all_ids.extend(int(value) for value in local_ids.tolist())
            all_coefficients.extend(float(value) for value in local_coefficients.tolist())
            offsets.extend(base + int(value) for value in local_offsets[1:].tolist())
            row_diagnostics.extend(diagnostics)
            residual_sum_squares += chunk_residual_sum_squares
            source_sum_squares += chunk_source_sum_squares
            maximum_row_residual_l2 = max(
                maximum_row_residual_l2, chunk_max_row_residual
            )
            maximum_absolute_residual = max(
                maximum_absolute_residual, chunk_max_absolute
            )
            cosine_sum += chunk_cosine_sum
            emit(
                "row-chunk-complete",
                phase_cache_key=phase_cache_key,
                donor_row_start=row_start,
                donor_row_stop=row_stop,
                completed_donor_rows=row_stop,
                total_donor_rows=donor_count,
                chunk_source=source,
                continuous_fold_coefficient_count=int(local_coefficients.numel()),
                semantic_code_coefficients_sha256=_tensor_sha256(local_coefficients),
            )

    semantic_offsets = torch.tensor(offsets, dtype=torch.int64)
    semantic_ids = torch.tensor(all_ids, dtype=torch.int64)
    semantic_coefficients = torch.tensor(all_coefficients, dtype=torch.float32)
    progress_count, progress_final = progress_status()
    coefficient_count = int(semantic_coefficients.numel())
    expected_coefficient_count = target_count + (
        donor_count - target_count
    ) * retained_k
    if coefficient_count != expected_coefficient_count:
        raise Lfm2SparseSemanticCodebookError(
            "Published coefficient count violates direct/active-k accounting."
        )
    analytic_audit = {
        "candidate_discovery_artifact_reused": candidate_inventory is not None,
        "local_preliminary_ridge_solve_row_count": preliminary_solve_calls,
        "local_active_refit_ridge_solve_row_count": active_refit_solve_calls,
        "coefficient_chunk_loader_calls": loader_calls,
        "coefficient_chunk_publisher_calls": publisher_calls,
        "dense_global_donor_by_target_map_materializations": 0,
    }
    body = {
        "schema": LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA,
        "status": "complete-local-sparse-semantic-codebook",
        "algorithm": _SEMANTIC_ALGORITHM,
        "phase_cache_key": phase_cache_key,
        "candidate_artifact_sha256": inventory.artifact_sha256,
        "candidate_content_sha256": inventory_receipt[
            "candidate_content_sha256"
        ],
        "candidate_phase_cache_key": inventory_receipt["phase_cache_key"],
        "candidate_phase_is_reusable_across_k_and_ridge_variants": True,
        "donor_table_sha256": donor_sha,
        "anchor_table_sha256": anchor_sha,
        "target_to_donor_sha256": mapping_sha,
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "direct_anchor_count": target_count,
        "every_direct_anchor_preserved_as_exact_one_hot": True,
        "non_direct_donor_row_count": donor_count - target_count,
        "maximum_candidates": requested_candidates,
        "active_k": retained_k,
        "candidate_ridge": first_ridge,
        "refit_ridge": second_ridge,
        "donor_rows_per_chunk": row_chunk,
        "solve_placement": {
            "device": str(fit_device),
            "dtype": _dtype_name(fit_dtype),
        },
        "publication_placement": {"device": "cpu", "dtype": "float32"},
        "selection_placement": inventory_receipt["selection_placement"],
        "selection_and_solve_placements_are_independent": True,
        "cuda_fp64_never_selected_implicitly": True,
        "semantic_code_offsets_sha256": _tensor_sha256(semantic_offsets),
        "semantic_code_target_ids_sha256": _tensor_sha256(semantic_ids),
        "semantic_code_coefficients_sha256": _tensor_sha256(
            semantic_coefficients
        ),
        "semantic_code_offset_count": int(semantic_offsets.numel()),
        "semantic_code_offset_bytes": int(semantic_offsets.numel()) * 8,
        "semantic_code_target_id_count": int(semantic_ids.numel()),
        "semantic_code_target_id_bytes": int(semantic_ids.numel()) * 8,
        "target_to_donor_bytes": int(mapping.numel()) * 8,
        "structural_integer_routing_bytes": (
            int(semantic_offsets.numel()) * 8
            + int(semantic_ids.numel()) * 8
            + int(mapping.numel()) * 8
        ),
        "integer_routing_values_are_structural_metadata": True,
        "continuous_fold_coefficient_count": coefficient_count,
        "continuous_fold_coefficient_bytes": coefficient_count * 4,
        "continuous_coefficients_are_folded_child_state": True,
        "continuous_coefficients_count_toward_effective_child_parameters": True,
        "effective_child_parameter_addition_from_semantic_bridge": coefficient_count,
        "representation_is_fixed_bridge_state_not_model_sparsification": True,
        "candidate_rows_are_local_not_one_global_residual_map": True,
        "active_tie_break": (
            "larger-absolute-preliminary-weight-then-lower-target-id"
        ),
        "published_row_order": "ascending-target-id",
        "reconstruction_root_mean_square_error": math.sqrt(
            residual_sum_squares / (donor_count * hidden_size)
        ),
        "relative_reconstruction_l2_error": math.sqrt(
            residual_sum_squares
            / max(source_sum_squares, torch.finfo(torch.float64).tiny)
        ),
        "mean_row_reconstruction_cosine": cosine_sum / donor_count,
        "maximum_row_reconstruction_l2_error": maximum_row_residual_l2,
        "maximum_absolute_reconstruction_residual": maximum_absolute_residual,
        "metrics_authenticate_published_fp32_coefficients": True,
        "condition_diagnostic_kind": "regularized-gram-diagonal-ratio-proxy",
        "condition_diagnostics_are_not_spectral_condition_numbers": True,
        "row_diagnostics": row_diagnostics,
        "row_diagnostics_sha256": canonical_sha256(row_diagnostics),
        "computed_chunk_count": computed_chunks,
        "checkpoint_reused_chunk_count": reused_chunks,
        "checkpoint_loader_supplied": coefficient_chunk_loader is not None,
        "checkpoint_loader_invocation_count": loader_calls,
        "checkpoint_publisher_supplied": coefficient_chunk_publisher is not None,
        "checkpoint_publisher_invocation_count": publisher_calls,
        "progress_record_count": progress_count,
        "progress_final_sha256": progress_final,
        "progress_callback_supplied": progress_callback is not None,
        "raw_token_span_emission_channel_modified": False,
        "semantic_code_channel_is_separate_from_external_emission": True,
        "analytic_operation_audit": analytic_audit,
        "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
    }
    return Lfm2SparseSemanticCodebook(
        semantic_offsets,
        semantic_ids,
        semantic_coefficients,
        mapping,
        _sealed(body, "artifact_sha256"),
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain_json(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_binary(value: Tensor, dtype: str) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    numpy_dtype = {"int64": "<i8", "float32": "<f4"}[dtype]
    return cpu.numpy().astype(numpy_dtype, copy=False).tobytes(order="C")


def _prepare_empty_directory(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(path))
    if path.exists():
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise Lfm2SparseSemanticCodebookError(
                "Persistence destination must be absent or an empty ordinary directory."
            )
    else:
        path.mkdir(parents=True)
    return path


def _save_tensor_bundle(
    *,
    artifact_kind: str,
    seal_field: str,
    receipt: Mapping[str, Any],
    tensors: Mapping[str, tuple[Tensor, str]],
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    root = _prepare_empty_directory(directory)
    receipt_value = _plain_json(receipt)
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(_canonical_json_bytes(receipt_value))
    file_records = [
        {
            "path": "receipt.json",
            "size_bytes": receipt_path.stat().st_size,
            "sha256": _file_sha256(receipt_path),
        }
    ]
    tensor_records = []
    for name, (tensor, dtype) in sorted(tensors.items()):
        filename = f"{name}.bin"
        path = root / filename
        path.write_bytes(_tensor_binary(tensor, dtype))
        record = {
            "name": name,
            "path": filename,
            "dtype": dtype,
            "shape": [int(size) for size in tensor.shape],
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        tensor_records.append(record)
        file_records.append(
            {
                "path": filename,
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    body = {
        "schema": LFM2_SPARSE_SEMANTIC_PERSISTENCE_SCHEMA,
        "artifact_kind": artifact_kind,
        "artifact_seal_field": seal_field,
        "artifact_sha256": receipt_value[seal_field],
        "files": sorted(file_records, key=lambda item: item["path"]),
        "tensors": tensor_records,
    }
    manifest = _sealed(body, "manifest_sha256")
    (root / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return manifest


def _load_tensor_bundle(
    directory: str | os.PathLike[str],
    *,
    artifact_kind: str,
    seal_field: str,
    expected_artifact_sha256: str | None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    root = Path(directory).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = Path(os.path.abspath(root))
    if not root.is_dir() or root.is_symlink():
        raise Lfm2SparseSemanticCodebookError(
            "Persistence source must be an ordinary directory."
        )
    try:
        manifest_raw = json.loads((root / "manifest.json").read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SparseSemanticCodebookError(
            "Cannot read sparse semantic persistence manifest."
        ) from exc
    manifest = _verify_sealed(
        manifest_raw,
        schema=LFM2_SPARSE_SEMANTIC_PERSISTENCE_SCHEMA,
        field="manifest_sha256",
        name="persistence manifest",
    )
    if (
        manifest.get("artifact_kind") != artifact_kind
        or manifest.get("artifact_seal_field") != seal_field
        or (
            expected_artifact_sha256 is not None
            and manifest.get("artifact_sha256") != expected_artifact_sha256
        )
    ):
        raise Lfm2SparseSemanticCodebookError(
            "Persistence manifest does not match the requested artifact."
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise Lfm2SparseSemanticCodebookError("Persistence file inventory is invalid.")
    expected_names = {"manifest.json"}
    for record in files:
        if not isinstance(record, Mapping):
            raise Lfm2SparseSemanticCodebookError("Persistence file record is invalid.")
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).name != relative
            or relative in expected_names
        ):
            raise Lfm2SparseSemanticCodebookError("Persistence file path is invalid.")
        expected_names.add(relative)
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or _file_sha256(path) != record.get("sha256")
        ):
            raise Lfm2SparseSemanticCodebookError(
                f"Persistence file {relative!r} failed authentication."
            )
    observed_names = {path.name for path in root.iterdir()}
    if observed_names != expected_names:
        raise Lfm2SparseSemanticCodebookError(
            "Persistence directory contains missing or unsealed extra files."
        )
    try:
        receipt = json.loads((root / "receipt.json").read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SparseSemanticCodebookError(
            "Cannot read persisted artifact receipt."
        ) from exc
    if receipt.get(seal_field) != manifest.get("artifact_sha256"):
        raise Lfm2SparseSemanticCodebookError(
            "Persisted receipt seal differs from the manifest."
        )
    tensors: dict[str, Tensor] = {}
    records = manifest.get("tensors")
    if not isinstance(records, list):
        raise Lfm2SparseSemanticCodebookError("Persistence tensor inventory is invalid.")
    for record in records:
        name = record.get("name")
        dtype = record.get("dtype")
        shape = record.get("shape")
        if (
            not isinstance(name, str)
            or dtype not in {"int64", "float32"}
            or not isinstance(shape, list)
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape)
        ):
            raise Lfm2SparseSemanticCodebookError(
                "Persistence tensor metadata is invalid."
            )
        numpy_dtype = {"int64": "<i8", "float32": "<f4"}[dtype]
        raw = (root / record["path"]).read_bytes()
        array = np.frombuffer(raw, dtype=numpy_dtype)
        expected_elements = math.prod(shape)
        if int(array.size) != expected_elements:
            raise Lfm2SparseSemanticCodebookError(
                "Persisted tensor byte length differs from its shape."
            )
        tensors[name] = torch.from_numpy(array.copy().reshape(shape)).contiguous()
    return tensors, receipt


def save_lfm2_sparse_semantic_candidate_inventory(
    artifact: Lfm2SparseSemanticCandidateInventory,
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    if not isinstance(artifact, Lfm2SparseSemanticCandidateInventory):
        raise TypeError("artifact must be Lfm2SparseSemanticCandidateInventory.")
    return _save_tensor_bundle(
        artifact_kind="candidate-inventory",
        seal_field="artifact_sha256",
        receipt=artifact.as_dict(),
        tensors={
            "candidate_offsets": (artifact.candidate_offsets, "int64"),
            "candidate_target_ids": (artifact.candidate_target_ids, "int64"),
            "target_to_donor": (artifact.target_to_donor, "int64"),
        },
        directory=directory,
    )


def load_lfm2_sparse_semantic_candidate_inventory(
    directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
) -> Lfm2SparseSemanticCandidateInventory:
    tensors, receipt = _load_tensor_bundle(
        directory,
        artifact_kind="candidate-inventory",
        seal_field="artifact_sha256",
        expected_artifact_sha256=expected_artifact_sha256,
    )
    if set(tensors) != {
        "candidate_offsets",
        "candidate_target_ids",
        "target_to_donor",
    }:
        raise Lfm2SparseSemanticCodebookError(
            "Candidate persistence tensor set is invalid."
        )
    return Lfm2SparseSemanticCandidateInventory(
        tensors["candidate_offsets"],
        tensors["candidate_target_ids"],
        tensors["target_to_donor"],
        receipt,
    )


def save_lfm2_sparse_semantic_codebook(
    artifact: Lfm2SparseSemanticCodebook,
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    if not isinstance(artifact, Lfm2SparseSemanticCodebook):
        raise TypeError("artifact must be Lfm2SparseSemanticCodebook.")
    return _save_tensor_bundle(
        artifact_kind="semantic-codebook",
        seal_field="artifact_sha256",
        receipt=artifact.as_dict(),
        tensors={
            "semantic_code_offsets": (artifact.semantic_code_offsets, "int64"),
            "semantic_code_target_ids": (
                artifact.semantic_code_target_ids,
                "int64",
            ),
            "semantic_code_coefficients": (
                artifact.semantic_code_coefficients,
                "float32",
            ),
            "target_to_donor": (artifact.target_to_donor, "int64"),
        },
        directory=directory,
    )


def load_lfm2_sparse_semantic_codebook(
    directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
) -> Lfm2SparseSemanticCodebook:
    tensors, receipt = _load_tensor_bundle(
        directory,
        artifact_kind="semantic-codebook",
        seal_field="artifact_sha256",
        expected_artifact_sha256=expected_artifact_sha256,
    )
    if set(tensors) != {
        "semantic_code_offsets",
        "semantic_code_target_ids",
        "semantic_code_coefficients",
        "target_to_donor",
    }:
        raise Lfm2SparseSemanticCodebookError(
            "Semantic codebook persistence tensor set is invalid."
        )
    return Lfm2SparseSemanticCodebook(
        tensors["semantic_code_offsets"],
        tensors["semantic_code_target_ids"],
        tensors["semantic_code_coefficients"],
        tensors["target_to_donor"],
        receipt,
    )


def save_lfm2_sparse_semantic_candidate_chunk(
    checkpoint: Lfm2SparseSemanticCandidateChunkCheckpoint,
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    if not isinstance(checkpoint, Lfm2SparseSemanticCandidateChunkCheckpoint):
        raise TypeError("checkpoint must be a candidate chunk checkpoint.")
    return _save_tensor_bundle(
        artifact_kind="candidate-chunk",
        seal_field="checkpoint_sha256",
        receipt=checkpoint.as_dict(),
        tensors={
            "candidate_offsets": (checkpoint.candidate_offsets, "int64"),
            "candidate_target_ids": (checkpoint.candidate_target_ids, "int64"),
        },
        directory=directory,
    )


def load_lfm2_sparse_semantic_candidate_chunk(
    directory: str | os.PathLike[str],
    *,
    expected_checkpoint_sha256: str | None = None,
) -> Lfm2SparseSemanticCandidateChunkCheckpoint:
    tensors, receipt = _load_tensor_bundle(
        directory,
        artifact_kind="candidate-chunk",
        seal_field="checkpoint_sha256",
        expected_artifact_sha256=expected_checkpoint_sha256,
    )
    if set(tensors) != {"candidate_offsets", "candidate_target_ids"}:
        raise Lfm2SparseSemanticCodebookError(
            "Candidate chunk persistence tensor set is invalid."
        )
    return Lfm2SparseSemanticCandidateChunkCheckpoint(
        tensors["candidate_offsets"],
        tensors["candidate_target_ids"],
        receipt,
    )


def save_lfm2_sparse_semantic_coefficient_chunk(
    checkpoint: Lfm2SparseSemanticCoefficientChunkCheckpoint,
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    if not isinstance(checkpoint, Lfm2SparseSemanticCoefficientChunkCheckpoint):
        raise TypeError("checkpoint must be a coefficient chunk checkpoint.")
    return _save_tensor_bundle(
        artifact_kind="coefficient-chunk",
        seal_field="checkpoint_sha256",
        receipt=checkpoint.as_dict(),
        tensors={
            "semantic_code_offsets": (checkpoint.semantic_code_offsets, "int64"),
            "semantic_code_target_ids": (
                checkpoint.semantic_code_target_ids,
                "int64",
            ),
            "semantic_code_coefficients": (
                checkpoint.semantic_code_coefficients,
                "float32",
            ),
        },
        directory=directory,
    )


def load_lfm2_sparse_semantic_coefficient_chunk(
    directory: str | os.PathLike[str],
    *,
    expected_checkpoint_sha256: str | None = None,
) -> Lfm2SparseSemanticCoefficientChunkCheckpoint:
    tensors, receipt = _load_tensor_bundle(
        directory,
        artifact_kind="coefficient-chunk",
        seal_field="checkpoint_sha256",
        expected_artifact_sha256=expected_checkpoint_sha256,
    )
    if set(tensors) != {
        "semantic_code_offsets",
        "semantic_code_target_ids",
        "semantic_code_coefficients",
    }:
        raise Lfm2SparseSemanticCodebookError(
            "Coefficient chunk persistence tensor set is invalid."
        )
    return Lfm2SparseSemanticCoefficientChunkCheckpoint(
        tensors["semantic_code_offsets"],
        tensors["semantic_code_target_ids"],
        tensors["semantic_code_coefficients"],
        receipt,
    )


__all__ = [
    "LFM2_SPARSE_SEMANTIC_CANDIDATE_CHUNK_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CANDIDATE_REQUEST_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_COEFFICIENT_CHUNK_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_COEFFICIENT_REQUEST_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_PERSISTENCE_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_PROGRESS_SCHEMA",
    "Lfm2SparseSemanticCandidateChunkCheckpoint",
    "Lfm2SparseSemanticCandidateInventory",
    "Lfm2SparseSemanticCodebook",
    "Lfm2SparseSemanticCodebookError",
    "Lfm2SparseSemanticCoefficientChunkCheckpoint",
    "compile_lfm2_sparse_semantic_codebook",
    "discover_lfm2_sparse_semantic_candidates",
    "load_lfm2_sparse_semantic_candidate_chunk",
    "load_lfm2_sparse_semantic_candidate_inventory",
    "load_lfm2_sparse_semantic_codebook",
    "load_lfm2_sparse_semantic_coefficient_chunk",
    "save_lfm2_sparse_semantic_candidate_chunk",
    "save_lfm2_sparse_semantic_candidate_inventory",
    "save_lfm2_sparse_semantic_codebook",
    "save_lfm2_sparse_semantic_coefficient_chunk",
]
