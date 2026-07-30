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
import platform
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
LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA = (
    "neural-forge-lfm2-residual-omp-semantic-candidate-inventory-v1"
)
LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_CHUNK_SCHEMA = (
    "neural-forge-lfm2-residual-omp-semantic-candidate-chunk-v1"
)
LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-residual-omp-semantic-candidate-request-v1"
)
LFM2_RESIDUAL_OMP_NUMERIC_BACKEND_SCHEMA = (
    "neural-forge-lfm2-residual-omp-numeric-backend-v1"
)
LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA = "neural-forge-lfm2-sparse-semantic-codebook-v1"
LFM2_SPARSE_SEMANTIC_COEFFICIENT_CHUNK_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-coefficient-chunk-v1"
)
LFM2_SPARSE_SEMANTIC_COEFFICIENT_REQUEST_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-coefficient-request-v1"
)
LFM2_SPARSE_SEMANTIC_PROGRESS_SCHEMA = "neural-forge-lfm2-sparse-semantic-progress-v1"
LFM2_SPARSE_SEMANTIC_NORM_RECLOSURE_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-norm-reclosure-v1"
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
_CANDIDATE_ALGORITHM = "bounded-absolute-cosine-neighbors-stable-target-id-ties-v1"
_RESIDUAL_OMP_CANDIDATE_ALGORITHM = (
    "full-anchor-normalized-residual-omp-double-mgs-stable-target-id-ties-v2"
)
_NORM_RECLOSURE_ALGORITHM = (
    "positive-row-norm-ratio-power-scale-baked-into-existing-coefficients-v1"
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
        raise Lfm2SparseSemanticCodebookError(f"{name} must be finite and positive.")
    return float(value)


def _finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Lfm2SparseSemanticCodebookError(f"{name} must be finite.")
    return float(value)


def _distribution_summary(value: Tensor, name: str) -> dict[str, Any]:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2SparseSemanticCodebookError(
            f"{name} distribution must be a finite floating vector."
        )
    ordered = torch.sort(value.detach().to(device="cpu", dtype=torch.float64)).values
    count = int(ordered.numel())
    if count == 0:
        return {
            "count": 0,
            "empty_population": True,
            "minimum": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p99": None,
            "maximum": None,
            "mean": None,
        }

    def quantile(fraction: float) -> float:
        index = int(math.floor(fraction * (count - 1)))
        return float(ordered[index].item())

    return {
        "count": count,
        "empty_population": False,
        "minimum": float(ordered[0].item()),
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p99": quantile(0.99),
        "maximum": float(ordered[-1].item()),
        "mean": float(ordered.mean().item()),
    }


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
        raise Lfm2SparseSemanticCodebookError(f"{name} must be explicit FP32 or FP64.")
    assert isinstance(dtype, torch.dtype)
    return dtype


def _compute_device(value: str | torch.device, name: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2SparseSemanticCodebookError(f"{name} is invalid.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise Lfm2SparseSemanticCodebookError(f"{name} must be a CPU or CUDA device.")
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


def build_lfm2_residual_omp_numeric_backend_receipt(
    selection_device: str | torch.device,
    selection_dtype: str | torch.dtype,
) -> dict[str, Any]:
    """Fingerprint the concrete numeric backend used by OMP selection.

    The receipt intentionally includes conservative global CUDA policies even
    for a CPU selection placement.  Invalidating an otherwise reusable CPU
    prefix is cheaper and safer than allowing a caller to move the same sealed
    request to CUDA without its numeric policy already being authenticated.
    """

    device = _compute_device(selection_device, "selection_device")
    dtype = _compute_dtype(selection_dtype, "selection_dtype")
    torch_build_configuration = str(torch.__config__.show())
    cuda_matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    cpu_backend = torch.backends.cpu
    cpu_capability = (
        str(cpu_backend.get_cpu_capability())
        if callable(getattr(cpu_backend, "get_cpu_capability", None))
        else "unavailable"
    )
    body: dict[str, Any] = {
        "schema": LFM2_RESIDUAL_OMP_NUMERIC_BACKEND_SCHEMA,
        "selection_placement": {
            "device": str(device),
            "device_type": device.type,
            "device_index": device.index,
            "dtype": _dtype_name(dtype),
        },
        "torch": {
            "version": str(torch.__version__),
            "git_version": (
                None
                if getattr(torch.version, "git_version", None) is None
                else str(torch.version.git_version)
            ),
            "cuda_runtime_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "hip_runtime_version": (
                None
                if getattr(torch.version, "hip", None) is None
                else str(torch.version.hip)
            ),
            "cudnn_version": (
                None if cudnn.version() is None else int(cudnn.version())
            ),
            "build_configuration_sha256": hashlib.sha256(
                torch_build_configuration.encode("utf-8")
            ).hexdigest(),
        },
        "float32_and_determinism_policy": {
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "autocast_explicitly_disabled_by_omp": True,
            "cuda_matmul_allow_tf32": bool(cuda_matmul.allow_tf32),
            "cudnn_allow_tf32": bool(cudnn.allow_tf32),
            "cudnn_deterministic": bool(cudnn.deterministic),
            "cudnn_benchmark": bool(cudnn.benchmark),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
                cuda_matmul.allow_fp16_reduced_precision_reduction
            ),
            "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
                cuda_matmul.allow_bf16_reduced_precision_reduction
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "cpu_backend": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "capability": cpu_capability,
            "intraop_thread_count": int(torch.get_num_threads()),
            "interop_thread_count": int(torch.get_num_interop_threads()),
            "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "mkl_available": bool(torch.backends.mkl.is_available()),
            "openmp_available": bool(torch.backends.openmp.is_available()),
        },
        "selected_accelerator": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        body["selected_accelerator"] = {
            "name": str(torch.cuda.get_device_name(device)),
            "compute_capability": [int(capability[0]), int(capability[1])],
            "multi_processor_count": int(properties.multi_processor_count),
            "total_memory_bytes": int(properties.total_memory),
            "compiled_arch_list": [str(value) for value in torch.cuda.get_arch_list()],
        }
    return _sealed(body, "numeric_backend_receipt_sha256")


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
        if (
            value.dtype == torch.bool
            or torch.is_floating_point(value)
            or value.is_complex()
        ):
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
        raise Lfm2SparseSemanticCodebookError("target_to_donor must be injective.")
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


def _validated_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise Lfm2SparseSemanticCodebookError(
            f"{name} must be exactly 64 hexadecimal characters."
        )
    return value.lower()


def _resolved_implementation_sha256(
    supplied: str | None,
) -> tuple[str, str]:
    if supplied is not None:
        return _validated_sha256(supplied, "implementation_sha256"), "explicit"
    try:
        source_path = Path(__file__).resolve(strict=True)
        source_sha = _file_sha256(source_path)
    except OSError as exc:
        raise Lfm2SparseSemanticCodebookError(
            "The loaded core source cannot be authenticated automatically; "
            "supply implementation_sha256 explicitly."
        ) from exc
    return (
        _validated_sha256(source_sha, "automatic implementation source SHA-256"),
        "automatic-loaded-core-source-file",
    )


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
        schema = (
            self.receipt.get("schema") if isinstance(self.receipt, Mapping) else None
        )
        if schema not in {
            LFM2_SPARSE_SEMANTIC_CANDIDATE_CHUNK_SCHEMA,
            LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_CHUNK_SCHEMA,
        }:
            raise Lfm2SparseSemanticCodebookError(
                "candidate chunk checkpoint schema is unsupported."
            )
        value = _verify_sealed(
            self.receipt,
            schema=str(schema),
            field="checkpoint_sha256",
            name="candidate chunk checkpoint",
        )
        if schema == LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_CHUNK_SCHEMA:
            _validated_sha256(
                value.get("implementation_sha256"),
                "candidate chunk implementation_sha256",
            )
            _validated_sha256(
                value.get("numeric_backend_receipt_sha256"),
                "candidate chunk numeric_backend_receipt_sha256",
            )
        row_start = _nonnegative_int(value.get("donor_row_start"), "donor_row_start")
        row_stop = _positive_int(value.get("donor_row_stop"), "donor_row_stop")
        if row_stop <= row_start:
            raise Lfm2SparseSemanticCodebookError(
                "Candidate chunk row bounds are invalid."
            )
        offsets = self.candidate_offsets.detach().to(device="cpu").contiguous()
        ids = self.candidate_target_ids.detach().to(device="cpu").contiguous()
        _csr_checks(
            offsets,
            ids,
            row_count=row_stop - row_start,
            target_row_count=_positive_int(
                value.get("target_row_count"), "target_row_count"
            ),
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
        schema = (
            self.receipt.get("schema") if isinstance(self.receipt, Mapping) else None
        )
        if schema not in {
            LFM2_SPARSE_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA,
            LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA,
        }:
            raise Lfm2SparseSemanticCodebookError(
                "candidate inventory schema is unsupported."
            )
        value = _verify_sealed(
            self.receipt,
            schema=str(schema),
            field="artifact_sha256",
            name="candidate inventory",
        )
        if schema == LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA:
            _validated_sha256(
                value.get("implementation_sha256"),
                "candidate inventory implementation_sha256",
            )
            numeric_backend = _verify_sealed(
                value.get("numeric_backend_receipt"),
                schema=LFM2_RESIDUAL_OMP_NUMERIC_BACKEND_SCHEMA,
                field="numeric_backend_receipt_sha256",
                name="candidate inventory numeric backend receipt",
            )
            if (
                value.get("numeric_backend_receipt_sha256")
                != numeric_backend["numeric_backend_receipt_sha256"]
            ):
                raise Lfm2SparseSemanticCodebookError(
                    "Candidate inventory numeric backend binding is invalid."
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
            raise Lfm2SparseSemanticCodebookError(
                "Coefficient chunk row bounds are invalid."
            )
        offsets = self.semantic_code_offsets.detach().to(device="cpu").contiguous()
        ids = self.semantic_code_target_ids.detach().to(device="cpu").contiguous()
        coefficients = (
            self.semantic_code_coefficients.detach().to(device="cpu").contiguous()
        )
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
            or value.get("continuous_fold_coefficient_count")
            != int(coefficients.numel())
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
        coefficients = (
            self.semantic_code_coefficients.detach().to(device="cpu").contiguous()
        )
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
            or value.get("continuous_fold_coefficient_count")
            != int(coefficients.numel())
            or value.get("continuous_fold_coefficient_bytes")
            != int(coefficients.numel()) * 4
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
            selected = tuple(
                int(value) for value in donor_ids.to(device="cpu").tolist()
            )
        else:
            if isinstance(donor_ids, (str, bytes)) or not isinstance(
                donor_ids, Sequence
            ):
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
                ids = self.semantic_code_target_ids[start:stop].to(
                    device=anchors.device
                )
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
    if candidate_chunk_publisher is not None and not callable(
        candidate_chunk_publisher
    ):
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
                selected_signed_cpu = torch.gather(similarities, 1, ranked).to(
                    device="cpu"
                )
                selected_absolute_cpu = torch.gather(candidate_scores, 1, ranked).to(
                    device="cpu"
                )
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


def _safe_unit_rows_to_compute(
    value: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> tuple[Tensor, Tensor]:
    """Normalize finite rows in CPU FP64 before any narrower device upload."""

    cpu64 = value.detach().to(device="cpu", dtype=torch.float64)
    _require_finite_compute(cpu64, f"{name} CPU FP64 normalization input")
    maximum = cpu64.abs().amax(dim=1)
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    scaled = cpu64 / safe_maximum.unsqueeze(1)
    norms = torch.linalg.vector_norm(scaled, dim=1)
    _require_finite_compute(norms, f"{name} CPU FP64 scaled norms")
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    normalized_cpu64 = scaled / safe_norms.unsqueeze(1)
    normalized = normalized_cpu64.to(device=device, dtype=dtype)
    _require_finite_compute(normalized, f"{name} normalized compute upload")
    return normalized, norms == 0


def _omp_step_summary(
    *,
    step: int,
    signed_cosines: Tensor,
    residual_ratios_before: Tensor,
    residual_ratios_after: Tensor,
    selected_target_ids: Tensor,
) -> dict[str, Any]:
    signed_cpu = signed_cosines.detach().to(device="cpu", dtype=torch.float64)
    before_cpu = residual_ratios_before.detach().to(device="cpu", dtype=torch.float64)
    after_cpu = residual_ratios_after.detach().to(device="cpu", dtype=torch.float64)
    target_cpu = selected_target_ids.detach().to(device="cpu", dtype=torch.int64)
    tolerance = torch.finfo(torch.float64).eps * 64.0
    return {
        "omp_step": step,
        "row_count": int(signed_cpu.numel()),
        "selected_absolute_residual_cosine": _distribution_summary(
            signed_cpu.abs(), "OMP selected absolute residual cosine"
        ),
        "selected_signed_residual_cosine": _distribution_summary(
            signed_cpu, "OMP selected signed residual cosine"
        ),
        "relative_residual_l2_before": _distribution_summary(
            before_cpu, "OMP relative residual before update"
        ),
        "relative_residual_l2_after": _distribution_summary(
            after_cpu, "OMP relative residual after update"
        ),
        "strictly_improved_row_count": int((after_cpu < before_cpu).sum().item()),
        "nonincreasing_with_tolerance_row_count": int(
            (after_cpu <= before_cpu + tolerance).sum().item()
        ),
        "selected_target_id_minimum": (
            int(target_cpu.min().item()) if target_cpu.numel() else None
        ),
        "selected_target_id_maximum": (
            int(target_cpu.max().item()) if target_cpu.numel() else None
        ),
    }


def _omp_orthogonalize_atom(
    q_basis: Tensor,
    atoms: Tensor,
    *,
    orthogonality_tolerance: float,
) -> tuple[Tensor, Tensor]:
    """Double-MGS one batched atom and report numerical dependence."""

    candidate = atoms.clone()
    for _pass in range(2):
        if int(q_basis.shape[1]) > 0:
            components = torch.bmm(q_basis, candidate.unsqueeze(2)).squeeze(2)
            candidate = candidate - torch.bmm(components.unsqueeze(1), q_basis).squeeze(
                1
            )
    candidate_norms = torch.linalg.vector_norm(candidate, dim=1)
    _require_finite_compute(candidate_norms, "OMP double-MGS candidate norms")
    nondegenerate = candidate_norms > orthogonality_tolerance
    safe_norms = torch.where(
        nondegenerate, candidate_norms, torch.ones_like(candidate_norms)
    )
    q_new = torch.where(
        nondegenerate.unsqueeze(1),
        candidate / safe_norms.unsqueeze(1),
        torch.zeros_like(candidate),
    )
    return q_new, ~nondegenerate


def _omp_append_orthonormal_atom(
    q_basis: Tensor,
    atoms: Tensor,
    residual: Tensor,
    *,
    orthogonality_tolerance: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append one batched atom with double MGS and reproject the residual."""

    q_new, degenerate = _omp_orthogonalize_atom(
        q_basis,
        atoms,
        orthogonality_tolerance=orthogonality_tolerance,
    )
    q_basis = torch.cat((q_basis, q_new.unsqueeze(1)), dim=1)
    # Two projection passes clean up finite-precision leakage against the full
    # current Q basis.  This is bounded row-local QR math, never a global map.
    updated_residual = residual
    for _pass in range(2):
        components = torch.bmm(q_basis, updated_residual.unsqueeze(2)).squeeze(2)
        updated_residual = updated_residual - torch.bmm(
            components.unsqueeze(1), q_basis
        ).squeeze(1)
    _require_finite_compute(updated_residual, "OMP orthogonal residual update")
    return q_basis, updated_residual, degenerate


def _residual_omp_prefix_phase_body(
    *,
    donor_table_sha256: str,
    anchor_table_sha256: str,
    target_to_donor_sha256: str,
    donor_row_count: int,
    target_row_count: int,
    hidden_size: int,
    implementation_sha256: str,
    numeric_backend_receipt_sha256: str,
    orthogonality_tolerance: float,
    donor_rows_per_chunk: int,
    selection_device: torch.device,
    selection_dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "schema": LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_REQUEST_SCHEMA,
        "algorithm": _RESIDUAL_OMP_CANDIDATE_ALGORITHM,
        "donor_table_sha256": donor_table_sha256,
        "anchor_table_sha256": anchor_table_sha256,
        "target_to_donor_sha256": target_to_donor_sha256,
        "donor_row_count": donor_row_count,
        "target_row_count": target_row_count,
        "hidden_size": hidden_size,
        "implementation_sha256": implementation_sha256,
        "numeric_backend_receipt_sha256": numeric_backend_receipt_sha256,
        "orthogonality_tolerance": orthogonality_tolerance,
        "donor_rows_per_chunk": donor_rows_per_chunk,
        "selection_placement": {
            "device": str(selection_device),
            "dtype": _dtype_name(selection_dtype),
        },
        "pre_upload_direction_normalization": (
            "cpu-fp64-row-max-abs-scaling-then-safe-unit-norm"
        ),
        "autocast_policy": "explicitly-disabled-during-selection",
        "cuda_tf32_allowed_at_request": bool(
            selection_device.type == "cuda"
            and selection_dtype == torch.float32
            and torch.backends.cuda.matmul.allow_tf32
        ),
        "cuda_tf32_policy": (
            "intentionally-use-and-seal-request-time-framework-policy"
        ),
        "active_k_intentionally_excluded_from_base_identity": True,
        "dictionary_search_policy": "all-target-anchors-at-every-new-step",
        "selection_atom_policy": "safe-l2-normalized-target-anchor-rows",
        "residual_update_policy": "incremental-double-mgs-qr-orthogonal-projection",
        "candidate_tie_break": "higher-absolute-residual-cosine-then-lower-target-id",
        "direct_anchor_policy": "explicit-target-to-donor-exact-one-candidate",
    }


def discover_lfm2_residual_omp_semantic_candidates(
    donor_embedding_table: Tensor,
    target_anchor_table: Tensor,
    donor_to_target_token_id: Tensor | Sequence[int],
    *,
    active_k: int,
    implementation_sha256: str | None = None,
    orthogonality_tolerance: float | None = None,
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
    """Discover full-dictionary residual-OMP supports with reusable prefixes.

    Direct bridge rows remain exact one-candidate identities.  Every other row
    searches all target anchors against its current orthogonal residual.  Only
    cumulative integer support prefixes are checkpointed; extending k replays
    cheap bounded QR updates from those IDs and skips the already-completed
    full-dictionary searches.  Final coefficients are still solved by the
    existing explicit codebook compile stage.
    """

    donor = _validated_table(donor_embedding_table, "donor_embedding_table")
    anchors = _validated_table(target_anchor_table, "target_anchor_table")
    donor_count, hidden_size = map(int, donor.shape)
    target_count, anchor_hidden = map(int, anchors.shape)
    if hidden_size != anchor_hidden:
        raise Lfm2SparseSemanticCodebookError(
            "donor_embedding_table and target_anchor_table hidden sizes differ."
        )
    mapping = _validated_mapping(
        donor_to_target_token_id,
        target_row_count=target_count,
        donor_row_count=donor_count,
    )
    retained_k = _positive_int(active_k, "active_k")
    if retained_k > target_count:
        raise Lfm2SparseSemanticCodebookError(
            "active_k cannot exceed the target anchor row count."
        )
    row_chunk = min(
        _positive_int(donor_rows_per_chunk, "donor_rows_per_chunk"), donor_count
    )
    select_device = _compute_device(selection_device, "selection_device")
    select_dtype = _compute_dtype(selection_dtype, "selection_dtype")
    numeric_backend_receipt = build_lfm2_residual_omp_numeric_backend_receipt(
        select_device,
        select_dtype,
    )
    numeric_backend_sha = str(
        numeric_backend_receipt["numeric_backend_receipt_sha256"]
    )
    if orthogonality_tolerance is None:
        tolerance = float(torch.finfo(select_dtype).eps * 64.0)
    else:
        tolerance = _positive_float(orthogonality_tolerance, "orthogonality_tolerance")
    implementation_sha, implementation_sha_resolution = _resolved_implementation_sha256(
        implementation_sha256
    )
    if candidate_chunk_loader is not None and not callable(candidate_chunk_loader):
        raise TypeError("candidate_chunk_loader must be callable or None.")
    if candidate_chunk_publisher is not None and not callable(
        candidate_chunk_publisher
    ):
        raise TypeError("candidate_chunk_publisher must be callable or None.")

    donor_sha = _tensor_sha256(donor, rows_per_chunk=row_chunk)
    anchor_sha = _tensor_sha256(anchors, rows_per_chunk=row_chunk)
    mapping_sha = _tensor_sha256(mapping, rows_per_chunk=row_chunk)
    base_phase_body = _residual_omp_prefix_phase_body(
        donor_table_sha256=donor_sha,
        anchor_table_sha256=anchor_sha,
        target_to_donor_sha256=mapping_sha,
        donor_row_count=donor_count,
        target_row_count=target_count,
        hidden_size=hidden_size,
        implementation_sha256=implementation_sha,
        numeric_backend_receipt_sha256=numeric_backend_sha,
        orthogonality_tolerance=tolerance,
        donor_rows_per_chunk=row_chunk,
        selection_device=select_device,
        selection_dtype=select_dtype,
    )
    base_phase_cache_key = canonical_sha256(base_phase_body)
    artifact_phase_cache_key = canonical_sha256(
        {
            "schema": LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA,
            "base_phase_cache_key": base_phase_cache_key,
            "active_k": retained_k,
        }
    )
    emit, progress_status = _progress_emitter(
        progress_callback, phase="residual-omp-candidate-discovery"
    )
    emit(
        "phase-start",
        base_phase_cache_key=base_phase_cache_key,
        artifact_phase_cache_key=artifact_phase_cache_key,
        donor_row_count=donor_count,
        target_row_count=target_count,
        requested_active_k=retained_k,
        implementation_sha256=implementation_sha,
        numeric_backend_receipt_sha256=numeric_backend_sha,
        orthogonality_tolerance=tolerance,
        donor_rows_per_chunk=row_chunk,
        selection_device=str(select_device),
        selection_dtype=_dtype_name(select_dtype),
        support_prefix_cache_reusable_across_k=True,
        full_target_anchor_search_each_new_step=True,
    )

    direct = _direct_inverse(mapping, donor_count)
    offsets = [0]
    all_ids: list[int] = []
    row_diagnostics: list[dict[str, Any]] = []
    global_step_signed: list[list[float]] = [[] for _ in range(retained_k)]
    global_step_before: list[list[float]] = [[] for _ in range(retained_k)]
    global_step_after: list[list[float]] = [[] for _ in range(retained_k)]
    global_step_targets: list[list[int]] = [[] for _ in range(retained_k)]
    computed_chunks = 0
    fully_reused_chunks = 0
    prefix_extended_chunks = 0
    loader_calls = 0
    publisher_calls = 0
    loaded_prefix_steps = 0
    newly_searched_steps = 0
    full_dictionary_matmul_calls = 0
    replayed_qr_step_calls = 0
    new_qr_step_calls = 0
    dependent_atom_retry_count = 0

    with torch.no_grad(), torch.autocast(device_type=select_device.type, enabled=False):
        normalized_anchors, anchor_zero_norm = _safe_unit_rows_to_compute(
            anchors,
            device=select_device,
            dtype=select_dtype,
            name="OMP anchor table",
        )
        target_id_axis = torch.arange(
            target_count, device=select_device, dtype=torch.int64
        ).unsqueeze(0)

        for row_start in range(0, donor_count, row_chunk):
            row_stop = min(row_start + row_chunk, donor_count)
            base_request_body = {
                "schema": LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_REQUEST_SCHEMA,
                "algorithm": _RESIDUAL_OMP_CANDIDATE_ALGORITHM,
                "base_phase_cache_key": base_phase_cache_key,
                "donor_row_start": row_start,
                "donor_row_stop": row_stop,
                "donor_table_sha256": donor_sha,
                "anchor_table_sha256": anchor_sha,
                "target_to_donor_sha256": mapping_sha,
                "implementation_sha256": implementation_sha,
                "numeric_backend_receipt_sha256": numeric_backend_sha,
                "orthogonality_tolerance": tolerance,
                "selection_device": str(select_device),
                "selection_dtype": _dtype_name(select_dtype),
                "final_active_k_intentionally_excluded": True,
            }
            base_request_sha = canonical_sha256(base_request_body)

            def prefix_request(depth: int) -> dict[str, Any]:
                return _sealed(
                    {
                        **base_request_body,
                        "base_request_sha256": base_request_sha,
                        "support_depth": depth,
                    },
                    "request_sha256",
                )

            loaded_checkpoint = None
            loaded_depth = 0
            if candidate_chunk_loader is not None:
                for depth in range(retained_k, 0, -1):
                    loader_calls += 1
                    request = prefix_request(depth)
                    candidate = candidate_chunk_loader(MappingProxyType(request))
                    if candidate is not None and not isinstance(
                        candidate, Lfm2SparseSemanticCandidateChunkCheckpoint
                    ):
                        raise Lfm2SparseSemanticCodebookError(
                            "candidate_chunk_loader returned an invalid checkpoint type."
                        )
                    if candidate is None:
                        continue
                    receipt = candidate.as_dict()
                    if (
                        receipt.get("schema")
                        != LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_CHUNK_SCHEMA
                        or receipt.get("request_sha256") != request["request_sha256"]
                        or receipt.get("base_phase_cache_key") != base_phase_cache_key
                        or receipt.get("base_request_sha256") != base_request_sha
                        or receipt.get("implementation_sha256") != implementation_sha
                        or receipt.get("numeric_backend_receipt_sha256")
                        != numeric_backend_sha
                        or receipt.get("support_depth") != depth
                        or receipt.get("donor_row_start") != row_start
                        or receipt.get("donor_row_stop") != row_stop
                    ):
                        raise Lfm2SparseSemanticCodebookError(
                            "Loaded residual OMP prefix does not match the exact request."
                        )
                    loaded_checkpoint = candidate
                    loaded_depth = depth
                    break

            if loaded_checkpoint is not None:
                loaded_receipt = loaded_checkpoint.as_dict()
                local_offsets = loaded_checkpoint.candidate_offsets
                local_ids = loaded_checkpoint.candidate_target_ids
                loaded_rejection_values = loaded_receipt.get(
                    "dependent_atom_rejection_counts"
                )
                non_direct_rows_in_chunk = int(
                    (direct[row_start:row_stop] < 0).sum().item()
                )
                if (
                    loaded_receipt.get("payload_kind") != "integer-support-prefix-only"
                    or loaded_receipt.get("cached_continuous_value_count") != 0
                    or loaded_receipt.get("cached_float_diagnostic_value_count") != 0
                    or "row_diagnostics" in loaded_receipt
                    or "omp_step_diagnostics" in loaded_receipt
                    or not isinstance(loaded_rejection_values, list)
                    or len(loaded_rejection_values)
                    != non_direct_rows_in_chunk * loaded_depth
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in loaded_rejection_values
                    )
                ):
                    raise Lfm2SparseSemanticCodebookError(
                        "Loaded residual OMP prefix payload is not integer-only."
                    )
                for local_row, donor_id in enumerate(range(row_start, row_stop)):
                    start = int(local_offsets[local_row])
                    stop = int(local_offsets[local_row + 1])
                    direct_target = int(direct[donor_id])
                    expected = 1 if direct_target >= 0 else loaded_depth
                    if stop - start != expected or (
                        direct_target >= 0 and int(local_ids[start]) != direct_target
                    ):
                        raise Lfm2SparseSemanticCodebookError(
                            "Loaded residual OMP prefix row policy is invalid."
                        )
                loaded_prefix_steps += loaded_depth
            else:
                local_offsets = torch.tensor(
                    [0] * (row_stop - row_start + 1), dtype=torch.int64
                )
                local_ids = torch.empty((0,), dtype=torch.int64)
                loaded_rejection_values = []

            chunk_donor, chunk_donor_zero_norm = _safe_unit_rows_to_compute(
                donor[row_start:row_stop],
                device=select_device,
                dtype=select_dtype,
                name="OMP donor selection chunk",
            )
            local_direct = direct[row_start:row_stop]
            non_direct_local_cpu = torch.nonzero(
                local_direct < 0, as_tuple=False
            ).flatten()
            non_direct_local = non_direct_local_cpu.to(device=select_device)
            target_rows = chunk_donor.index_select(0, non_direct_local)
            source_norms = torch.linalg.vector_norm(target_rows, dim=1)
            _require_finite_compute(source_norms, "OMP donor source norms")
            safe_source_norms = torch.where(
                source_norms > 0, source_norms, torch.ones_like(source_norms)
            )
            normalized_sources = target_rows / safe_source_norms.unsqueeze(1)
            non_direct_count = int(non_direct_local.numel())
            omp_row_by_local = {
                int(local_row.item()): omp_row
                for omp_row, local_row in enumerate(non_direct_local_cpu)
            }
            q_basis = torch.empty(
                (non_direct_count, 0, hidden_size),
                device=select_device,
                dtype=select_dtype,
            )
            residual = target_rows.clone()
            chunk_step_diagnostics: list[dict[str, Any]] = []

            if loaded_depth:
                selected_rows = []
                for local_row_tensor in non_direct_local_cpu:
                    local_row = int(local_row_tensor.item())
                    start = int(local_offsets[local_row])
                    stop = int(local_offsets[local_row + 1])
                    selected_rows.append(local_ids[start:stop])
                selected_ids = (
                    torch.stack(selected_rows).to(
                        device=select_device, dtype=torch.int64
                    )
                    if selected_rows
                    else torch.empty(
                        (0, loaded_depth),
                        device=select_device,
                        dtype=torch.int64,
                    )
                )
                rejection_matrix = torch.tensor(
                    loaded_rejection_values,
                    device=select_device,
                    dtype=torch.int64,
                ).reshape(non_direct_count, loaded_depth)
                signed_history = []
                ratio_history = []
                degenerate_history = []
                rejection_history = []
                for step in range(loaded_depth):
                    residual_norms_before = torch.linalg.vector_norm(residual, dim=1)
                    selected_atoms = normalized_anchors.index_select(
                        0, selected_ids[:, step]
                    )
                    selected_correlations = (residual * selected_atoms).sum(dim=1)
                    signed_cosines = torch.where(
                        residual_norms_before > 0,
                        selected_correlations / residual_norms_before,
                        torch.zeros_like(selected_correlations),
                    )
                    ratios_before = (
                        ratio_history[-1]
                        if ratio_history
                        else torch.where(
                            source_norms > 0,
                            torch.ones_like(source_norms),
                            torch.zeros_like(source_norms),
                        )
                    )
                    atoms = normalized_anchors.index_select(0, selected_ids[:, step])
                    q_basis, residual, replay_degenerate = _omp_append_orthonormal_atom(
                        q_basis,
                        atoms,
                        residual,
                        orthogonality_tolerance=tolerance,
                    )
                    replayed_qr_step_calls += non_direct_count
                    if bool(replay_degenerate.any()):
                        raise Lfm2SparseSemanticCodebookError(
                            "Replayed residual OMP prefix contains a dependent atom."
                        )
                    residual_norms_after = torch.linalg.vector_norm(residual, dim=1)
                    ratios_after = torch.where(
                        source_norms > 0,
                        residual_norms_after / safe_source_norms,
                        torch.zeros_like(residual_norms_after),
                    )
                    signed_history.append(signed_cosines)
                    ratio_history.append(ratios_after)
                    degenerate_history.append(replay_degenerate)
                    rejection_history.append(rejection_matrix[:, step])
                    step_diagnostic = _omp_step_summary(
                        step=step,
                        signed_cosines=signed_cosines,
                        residual_ratios_before=ratios_before,
                        residual_ratios_after=ratios_after,
                        selected_target_ids=selected_ids[:, step],
                    )
                    chunk_step_diagnostics.append(step_diagnostic)
                    emit(
                        "omp-prefix-step-replayed",
                        base_phase_cache_key=base_phase_cache_key,
                        donor_row_start=row_start,
                        donor_row_stop=row_stop,
                        loaded_support_depth=loaded_depth,
                        **step_diagnostic,
                    )
            else:
                selected_ids = torch.empty(
                    (non_direct_count, 0),
                    device=select_device,
                    dtype=torch.int64,
                )
                signed_history = []
                ratio_history = []
                degenerate_history = []
                rejection_history = []

            direct_diagnostics: dict[int, dict[str, Any]] = {}
            for local_row, donor_id in enumerate(range(row_start, row_stop)):
                direct_target = int(local_direct[local_row])
                if direct_target < 0:
                    continue
                observed = chunk_donor[local_row]
                direct_residual_l2 = float(
                    torch.linalg.vector_norm(
                        observed - normalized_anchors[direct_target]
                    ).item()
                )
                direct_diagnostics[local_row] = {
                    "donor_id": donor_id,
                    "direct_anchor": True,
                    "candidate_count": 1,
                    "nearest_candidate_cosine": 1.0,
                    "candidate_cutoff_cosine": 1.0,
                    "nearest_candidate_absolute_cosine": 1.0,
                    "candidate_cutoff_absolute_cosine": 1.0,
                    "donor_zero_norm": bool(chunk_donor_zero_norm[local_row].item()),
                    "omp_selection_step_count": 0,
                    "omp_selected_residual_cosines": [],
                    "omp_relative_residual_l2_after_steps": [],
                    "omp_degenerate_qr_append": [],
                    "omp_dependent_atom_rejections": [],
                    "omp_final_residual_l2": direct_residual_l2,
                    "direct_anchor_was_not_greedily_reselected": True,
                }

            def materialize_structural_prefix(depth: int) -> tuple[Tensor, Tensor]:
                selected_cpu = selected_ids[:, :depth].to(device="cpu")
                prefix_offsets = [0]
                prefix_ids: list[int] = []
                for local_row in range(row_stop - row_start):
                    if local_row in direct_diagnostics:
                        row_ids = [int(local_direct[local_row])]
                    else:
                        row_ids = [
                            int(value)
                            for value in selected_cpu[
                                omp_row_by_local[local_row]
                            ].tolist()
                        ]
                    prefix_ids.extend(row_ids)
                    prefix_offsets.append(len(prefix_ids))
                return (
                    torch.tensor(prefix_offsets, dtype=torch.int64),
                    torch.tensor(prefix_ids, dtype=torch.int64),
                )

            def materialize_prefix(
                depth: int,
            ) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
                source_norms_cpu = source_norms.to(device="cpu")
                residual_norms_cpu = torch.linalg.vector_norm(residual, dim=1).to(
                    device="cpu"
                )
                source_cosines_cpu = torch.empty(
                    (non_direct_count, depth), dtype=torch.float64
                )
                if non_direct_count and depth:
                    selected_atoms = normalized_anchors.index_select(
                        0, selected_ids[:, :depth].reshape(-1)
                    ).reshape(non_direct_count, depth, hidden_size)
                    source_cosines_cpu = (
                        (selected_atoms * normalized_sources.unsqueeze(1))
                        .sum(dim=2)
                        .to(device="cpu", dtype=torch.float64)
                    )
                diagnostics_by_local = dict(direct_diagnostics)
                for omp_row, local_row_tensor in enumerate(non_direct_local_cpu):
                    local_row = int(local_row_tensor.item())
                    source_cosines = [
                        float(value) for value in source_cosines_cpu[omp_row].tolist()
                    ]
                    residual_cosines = [
                        float(signed_history[step][omp_row].item())
                        for step in range(depth)
                    ]
                    residual_ratios = [
                        float(ratio_history[step][omp_row].item())
                        for step in range(depth)
                    ]
                    degeneracies = [
                        bool(degenerate_history[step][omp_row].item())
                        for step in range(depth)
                    ]
                    rejection_counts = [
                        int(rejection_history[step][omp_row].item())
                        for step in range(depth)
                    ]
                    diagnostics_by_local[local_row] = {
                        "donor_id": row_start + local_row,
                        "direct_anchor": False,
                        "candidate_count": depth,
                        "nearest_candidate_cosine": source_cosines[0],
                        "candidate_cutoff_cosine": source_cosines[-1],
                        "nearest_candidate_absolute_cosine": abs(source_cosines[0]),
                        "candidate_cutoff_absolute_cosine": abs(source_cosines[-1]),
                        "donor_zero_norm": bool(
                            source_norms_cpu[omp_row].item() == 0.0
                        ),
                        "omp_selection_step_count": depth,
                        "omp_selected_residual_cosines": residual_cosines,
                        "omp_relative_residual_l2_after_steps": residual_ratios,
                        "omp_degenerate_qr_append": degeneracies,
                        "omp_dependent_atom_rejections": rejection_counts,
                        "omp_final_residual_l2": float(
                            residual_norms_cpu[omp_row].item()
                        ),
                        "omp_final_relative_residual_l2": residual_ratios[-1],
                        "omp_full_target_anchor_search_each_step": True,
                    }
                diagnostics_result = []
                for local_row in range(row_stop - row_start):
                    diagnostics_result.append(diagnostics_by_local[local_row])
                prefix_offsets, prefix_ids = materialize_structural_prefix(depth)
                return prefix_offsets, prefix_ids, diagnostics_result

            for step in range(loaded_depth, retained_k):
                if non_direct_count:
                    residual_norms_before = torch.linalg.vector_norm(residual, dim=1)
                    _require_finite_compute(
                        residual_norms_before, "OMP residual norms before search"
                    )
                    correlations = residual @ normalized_anchors.transpose(0, 1)
                    _require_finite_compute(
                        correlations, "OMP full-dictionary residual correlations"
                    )
                    full_dictionary_matmul_calls += 1
                    scores = correlations.abs()
                    if step:
                        scores.scatter_(1, selected_ids, -torch.inf)
                    working_scores = scores
                    unresolved = torch.ones(
                        (non_direct_count,),
                        device=select_device,
                        dtype=torch.bool,
                    )
                    next_ids = torch.full(
                        (non_direct_count,),
                        -1,
                        device=select_device,
                        dtype=torch.int64,
                    )
                    accepted_q = torch.zeros_like(target_rows)
                    rejection_counts = torch.zeros(
                        (non_direct_count,),
                        device=select_device,
                        dtype=torch.int64,
                    )
                    for _retry in range(target_count - step):
                        best_scores = working_scores.max(dim=1, keepdim=True).values
                        if bool(
                            unresolved.logical_and(
                                ~torch.isfinite(best_scores.squeeze(1))
                            ).any()
                        ):
                            break
                        tied_target_ids = torch.where(
                            working_scores == best_scores,
                            target_id_axis.expand_as(working_scores),
                            torch.full_like(
                                working_scores,
                                target_count,
                                dtype=torch.int64,
                            ),
                        )
                        proposed_ids = tied_target_ids.min(dim=1).values
                        proposed_q, proposed_degenerate = _omp_orthogonalize_atom(
                            q_basis,
                            normalized_anchors.index_select(0, proposed_ids),
                            orthogonality_tolerance=tolerance,
                        )
                        accepted = unresolved & ~proposed_degenerate
                        next_ids = torch.where(accepted, proposed_ids, next_ids)
                        accepted_q = torch.where(
                            accepted.unsqueeze(1), proposed_q, accepted_q
                        )
                        rejected = unresolved & proposed_degenerate
                        rejection_counts += rejected.to(dtype=torch.int64)
                        rows = torch.nonzero(unresolved, as_tuple=False).flatten()
                        working_scores[
                            rows, proposed_ids.index_select(0, rows)
                        ] = -torch.inf
                        unresolved = rejected
                        if not bool(unresolved.any()):
                            break
                    failed_rows = unresolved | (next_ids < 0)
                    if bool(failed_rows.any()):
                        first_row = int(
                            torch.nonzero(failed_rows, as_tuple=False)[0].item()
                        )
                        raise Lfm2SparseSemanticCodebookError(
                            "Residual OMP reached deterministic numerical rank "
                            f"saturation at donor row "
                            f"{row_start + int(non_direct_local_cpu[first_row])} "
                            f"before support depth {step + 1}."
                        )
                    selected_correlations = correlations.gather(
                        1, next_ids.unsqueeze(1)
                    ).squeeze(1)
                    signed_cosines = torch.where(
                        residual_norms_before > 0,
                        selected_correlations / residual_norms_before,
                        torch.zeros_like(selected_correlations),
                    )
                    selected_ids = torch.cat(
                        (selected_ids, next_ids.unsqueeze(1)), dim=1
                    )
                    q_basis = torch.cat((q_basis, accepted_q.unsqueeze(1)), dim=1)
                    for _pass in range(2):
                        components = torch.bmm(q_basis, residual.unsqueeze(2)).squeeze(
                            2
                        )
                        residual = residual - torch.bmm(
                            components.unsqueeze(1), q_basis
                        ).squeeze(1)
                    _require_finite_compute(residual, "OMP orthogonal residual update")
                    new_qr_step_calls += non_direct_count
                    dependent_atom_retry_count += int(rejection_counts.sum().item())
                    degenerate = torch.zeros(
                        (non_direct_count,),
                        device=select_device,
                        dtype=torch.bool,
                    )
                    residual_norms_after = torch.linalg.vector_norm(residual, dim=1)
                    _require_finite_compute(
                        residual_norms_after, "OMP residual norms after projection"
                    )
                    ratios_after = torch.where(
                        source_norms > 0,
                        residual_norms_after / safe_source_norms,
                        torch.zeros_like(residual_norms_after),
                    )
                    ratios_before = (
                        ratio_history[-1]
                        if ratio_history
                        else torch.where(
                            source_norms > 0,
                            torch.ones_like(source_norms),
                            torch.zeros_like(source_norms),
                        )
                    )
                else:
                    next_ids = torch.empty(
                        (0,), device=select_device, dtype=torch.int64
                    )
                    selected_ids = torch.cat(
                        (selected_ids, next_ids.unsqueeze(1)), dim=1
                    )
                    signed_cosines = torch.empty(
                        (0,), device=select_device, dtype=select_dtype
                    )
                    ratios_before = signed_cosines
                    ratios_after = signed_cosines
                    degenerate = torch.empty(
                        (0,), device=select_device, dtype=torch.bool
                    )
                    rejection_counts = torch.empty(
                        (0,), device=select_device, dtype=torch.int64
                    )
                signed_history.append(signed_cosines)
                ratio_history.append(ratios_after)
                degenerate_history.append(degenerate)
                rejection_history.append(rejection_counts)
                newly_searched_steps += 1
                step_diagnostic = _omp_step_summary(
                    step=step,
                    signed_cosines=signed_cosines,
                    residual_ratios_before=ratios_before,
                    residual_ratios_after=ratios_after,
                    selected_target_ids=next_ids,
                )
                chunk_step_diagnostics.append(step_diagnostic)
                emit(
                    "omp-step-complete",
                    base_phase_cache_key=base_phase_cache_key,
                    donor_row_start=row_start,
                    donor_row_stop=row_stop,
                    support_depth=step + 1,
                    chunk_source="computed-full-anchor-double-mgs-omp",
                    **step_diagnostic,
                )
                local_offsets, local_ids = materialize_structural_prefix(step + 1)
                request = prefix_request(step + 1)
                checkpoint_body = {
                    "schema": (LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_CHUNK_SCHEMA),
                    "algorithm": _RESIDUAL_OMP_CANDIDATE_ALGORITHM,
                    "base_phase_cache_key": base_phase_cache_key,
                    "base_request_sha256": base_request_sha,
                    "request_sha256": request["request_sha256"],
                    "support_depth": step + 1,
                    "donor_row_start": row_start,
                    "donor_row_stop": row_stop,
                    "target_row_count": target_count,
                    "implementation_sha256": implementation_sha,
                    "numeric_backend_receipt_sha256": numeric_backend_sha,
                    "orthogonality_tolerance": tolerance,
                    "candidate_entry_count": int(local_ids.numel()),
                    "candidate_offsets_sha256": _tensor_sha256(local_offsets),
                    "candidate_target_ids_sha256": _tensor_sha256(local_ids),
                    "payload_kind": "integer-support-prefix-only",
                    "dependent_atom_rejection_counts": (
                        torch.stack(rejection_history, dim=1)
                        .to(device="cpu", dtype=torch.int64)
                        .reshape(-1)
                        .tolist()
                        if non_direct_count
                        else []
                    ),
                    "cached_continuous_value_count": 0,
                    "cached_float_diagnostic_value_count": 0,
                    "row_diagnostics_intentionally_omitted": True,
                    "step_diagnostics_intentionally_omitted": True,
                    "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
                }
                checkpoint = Lfm2SparseSemanticCandidateChunkCheckpoint(
                    local_offsets,
                    local_ids,
                    _sealed(checkpoint_body, "checkpoint_sha256"),
                )
                if candidate_chunk_publisher is not None:
                    candidate_chunk_publisher(checkpoint)
                    publisher_calls += 1

            # Rebuild all published floating diagnostics from the final integer
            # support sequence through one common QR path.  Fresh discovery and
            # resumed-prefix runs therefore seal identical semantic content.
            q_basis = torch.empty(
                (non_direct_count, 0, hidden_size),
                device=select_device,
                dtype=select_dtype,
            )
            residual = target_rows.clone()
            signed_history = []
            ratio_history = []
            degenerate_history = []
            chunk_step_diagnostics = []
            for step in range(retained_k):
                residual_norms_before = torch.linalg.vector_norm(residual, dim=1)
                selected_atoms = normalized_anchors.index_select(
                    0, selected_ids[:, step]
                )
                selected_correlations = (residual * selected_atoms).sum(dim=1)
                signed_cosines = torch.where(
                    residual_norms_before > 0,
                    selected_correlations / residual_norms_before,
                    torch.zeros_like(selected_correlations),
                )
                ratios_before = (
                    ratio_history[-1]
                    if ratio_history
                    else torch.where(
                        source_norms > 0,
                        torch.ones_like(source_norms),
                        torch.zeros_like(source_norms),
                    )
                )
                q_basis, residual, replay_degenerate = _omp_append_orthonormal_atom(
                    q_basis,
                    selected_atoms,
                    residual,
                    orthogonality_tolerance=tolerance,
                )
                replayed_qr_step_calls += non_direct_count
                if bool(replay_degenerate.any()):
                    raise Lfm2SparseSemanticCodebookError(
                        "Final residual OMP support contains a dependent atom."
                    )
                residual_norms_after = torch.linalg.vector_norm(residual, dim=1)
                ratios_after = torch.where(
                    source_norms > 0,
                    residual_norms_after / safe_source_norms,
                    torch.zeros_like(residual_norms_after),
                )
                signed_history.append(signed_cosines)
                ratio_history.append(ratios_after)
                degenerate_history.append(replay_degenerate)
                chunk_step_diagnostics.append(
                    _omp_step_summary(
                        step=step,
                        signed_cosines=signed_cosines,
                        residual_ratios_before=ratios_before,
                        residual_ratios_after=ratios_after,
                        selected_target_ids=selected_ids[:, step],
                    )
                )

            local_offsets, local_ids, diagnostics = materialize_prefix(retained_k)
            if loaded_depth == retained_k:
                fully_reused_chunks += 1
                source = "reused-complete-authenticated-omp-prefix"
            else:
                computed_chunks += 1
                if loaded_depth:
                    prefix_extended_chunks += 1
                    source = "extended-authenticated-omp-prefix"
                else:
                    source = "computed-full-anchor-double-mgs-omp"

            for local_row, diagnostic in enumerate(diagnostics):
                if diagnostic["direct_anchor"]:
                    continue
                start = int(local_offsets[local_row])
                stop = int(local_offsets[local_row + 1])
                selected = local_ids[start:stop].tolist()
                for step in range(retained_k):
                    cosine = float(diagnostic["omp_selected_residual_cosines"][step])
                    after = float(
                        diagnostic["omp_relative_residual_l2_after_steps"][step]
                    )
                    before = (
                        0.0
                        if diagnostic["donor_zero_norm"]
                        else (
                            1.0
                            if step == 0
                            else float(
                                diagnostic["omp_relative_residual_l2_after_steps"][
                                    step - 1
                                ]
                            )
                        )
                    )
                    global_step_signed[step].append(cosine)
                    global_step_before[step].append(before)
                    global_step_after[step].append(after)
                    global_step_targets[step].append(int(selected[step]))

            base = len(all_ids)
            all_ids.extend(int(value) for value in local_ids.tolist())
            offsets.extend(base + int(value) for value in local_offsets[1:].tolist())
            row_diagnostics.extend(diagnostics)
            emit(
                "row-chunk-complete",
                base_phase_cache_key=base_phase_cache_key,
                donor_row_start=row_start,
                donor_row_stop=row_stop,
                completed_donor_rows=row_stop,
                total_donor_rows=donor_count,
                chunk_source=source,
                loaded_support_depth=loaded_depth,
                final_support_depth=retained_k,
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
            "algorithm": _RESIDUAL_OMP_CANDIDATE_ALGORITHM,
            "phase_cache_key": artifact_phase_cache_key,
            "target_to_donor_sha256": mapping_sha,
            "candidate_offsets_sha256": candidate_offsets_sha,
            "candidate_target_ids_sha256": candidate_ids_sha,
            "row_diagnostics_sha256": row_diagnostics_sha,
        }
    )
    omp_step_diagnostics = [
        _omp_step_summary(
            step=step,
            signed_cosines=torch.tensor(global_step_signed[step], dtype=torch.float64),
            residual_ratios_before=torch.tensor(
                global_step_before[step], dtype=torch.float64
            ),
            residual_ratios_after=torch.tensor(
                global_step_after[step], dtype=torch.float64
            ),
            selected_target_ids=torch.tensor(
                global_step_targets[step], dtype=torch.int64
            ),
        )
        for step in range(retained_k)
    ]
    analytic_audit = {
        "anchor_normalization_calls": 1,
        "full_target_dictionary_residual_matmul_calls": full_dictionary_matmul_calls,
        "loaded_prefix_steps_skipping_dictionary_search": loaded_prefix_steps,
        "newly_searched_support_steps": newly_searched_steps,
        "replayed_prefix_double_mgs_qr_row_step_calls": replayed_qr_step_calls,
        "new_double_mgs_qr_row_step_calls": new_qr_step_calls,
        "dependent_atom_retry_count": dependent_atom_retry_count,
        "dependent_atom_published_count": 0,
        "candidate_chunk_loader_calls": loader_calls,
        "candidate_chunk_publisher_calls": publisher_calls,
        "dense_global_donor_by_target_map_materializations": 0,
        "continuous_projection_state_checkpoint_values": 0,
    }
    body = {
        "schema": LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA,
        "status": "complete-residual-aware-full-dictionary-omp-supports",
        "algorithm": _RESIDUAL_OMP_CANDIDATE_ALGORITHM,
        "phase_cache_key": artifact_phase_cache_key,
        "artifact_phase_cache_key": artifact_phase_cache_key,
        "base_phase_cache_key": base_phase_cache_key,
        "donor_table_sha256": donor_sha,
        "anchor_table_sha256": anchor_sha,
        "target_to_donor_sha256": mapping_sha,
        "implementation_sha256": implementation_sha,
        "implementation_sha256_resolution": implementation_sha_resolution,
        "numeric_backend_receipt": numeric_backend_receipt,
        "numeric_backend_receipt_sha256": numeric_backend_sha,
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "direct_anchor_count": target_count,
        "non_direct_donor_row_count": donor_count - target_count,
        "maximum_candidates": retained_k,
        "active_k": retained_k,
        "orthogonality_tolerance": tolerance,
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
        "candidate_phase_is_independent_of_active_k_and_ridges": False,
        "support_prefix_cache_reusable_across_active_k": True,
        "active_k_excluded_from_base_phase_and_row_request_identity": True,
        "selection_placement": {
            "device": str(select_device),
            "dtype": _dtype_name(select_dtype),
        },
        "pre_upload_direction_normalization": (
            "cpu-fp64-row-max-abs-scaling-then-safe-unit-norm"
        ),
        "autocast_policy": "explicitly-disabled-during-selection",
        "cuda_tf32_allowed_at_request": bool(
            select_device.type == "cuda"
            and select_dtype == torch.float32
            and torch.backends.cuda.matmul.allow_tf32
        ),
        "cuda_tf32_policy": (
            "intentionally-use-and-seal-request-time-framework-policy"
        ),
        "dictionary_search_policy": "all-target-anchors-at-every-new-step",
        "full_target_anchor_search_each_new_step": True,
        "selection_atom_policy": "safe-l2-normalized-target-anchor-rows",
        "residual_update_policy": "incremental-double-mgs-qr-orthogonal-projection",
        "final_continuous_coefficients_stored_in_candidate_artifact": 0,
        "cached_continuous_projection_state_count": 0,
        "final_coefficient_policy": "deferred-to-existing-explicit-codebook-compile",
        "candidate_tie_break": ("higher-absolute-residual-cosine-then-lower-target-id"),
        "cosine_definition": (
            "signed-dot-of-residual-with-safe-l2-normalized-anchor-"
            "divided-by-safe-residual-l2; selection-uses-absolute-value"
        ),
        "zero_norm_policy": (
            "normalized-zero-anchor-and-zero-residual-have-zero-correlation"
        ),
        "donor_zero_norm_row_count": sum(
            bool(row["donor_zero_norm"]) for row in row_diagnostics
        ),
        "anchor_zero_norm_row_count": int(anchor_zero_norm.sum().item()),
        "row_diagnostics": row_diagnostics,
        "row_diagnostics_sha256": row_diagnostics_sha,
        "omp_step_diagnostics": omp_step_diagnostics,
        "omp_step_diagnostics_sha256": canonical_sha256(omp_step_diagnostics),
        "computed_chunk_count": computed_chunks,
        "checkpoint_fully_reused_chunk_count": fully_reused_chunks,
        "checkpoint_prefix_extended_chunk_count": prefix_extended_chunks,
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
    if (
        not isinstance(candidate_diagnostics, list)
        or len(candidate_diagnostics) != donor_count
    ):
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
                            "published_reconstruction_l2_error": (residual_norm_value),
                            "published_relative_reconstruction_l2_error": (
                                residual_norm_value
                                / max(
                                    observed_norm_value,
                                    torch.finfo(fit_dtype).tiny,
                                )
                            ),
                            "published_reconstruction_cosine": float(values[2].item()),
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
                            torch.linalg.vector_norm(preliminary_residual, dim=1),
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
                    published_coefficients_cpu = published_coefficients.to(device="cpu")
                    for local_index, donor_id in enumerate(non_direct_ids):
                        metrics = diagnostic_metrics_cpu[local_index]
                        ids_row = [
                            int(value) for value in active_ids_cpu[local_index].tolist()
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
                                    / metrics[2].clamp_min(torch.finfo(fit_dtype).tiny)
                                ).item()
                            ),
                            "published_reconstruction_cosine": float(metrics[3].item()),
                            "preliminary_gram_diagonal_condition_proxy": float(
                                metrics[4].item()
                            ),
                            "refit_gram_diagonal_condition_proxy": float(
                                metrics[5].item()
                            ),
                            "active_target_ids": ids_row,
                            "active_coefficient_count": retained_k,
                        }
                        diagnostics_by_donor[donor_id]["_coefficients"] = (
                            coefficient_row
                        )
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
            all_coefficients.extend(
                float(value) for value in local_coefficients.tolist()
            )
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
    expected_coefficient_count = (
        target_count + (donor_count - target_count) * retained_k
    )
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
        "candidate_content_sha256": inventory_receipt["candidate_content_sha256"],
        "candidate_phase_cache_key": inventory_receipt["phase_cache_key"],
        "candidate_discovery_algorithm": inventory_receipt["algorithm"],
        "candidate_phase_is_reusable_across_k_and_ridge_variants": bool(
            inventory_receipt.get(
                "candidate_phase_is_independent_of_active_k_and_ridges", False
            )
        ),
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
        "semantic_code_coefficients_sha256": _tensor_sha256(semantic_coefficients),
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
        "active_tie_break": ("larger-absolute-preliminary-weight-then-lower-target-id"),
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


def reclose_lfm2_sparse_semantic_codebook_norms(
    base_codebook: Lfm2SparseSemanticCodebook,
    donor_table: Tensor,
    anchor_table: Tensor,
    *,
    gamma: float,
    max_scale: float,
    minimum_cosine_to_amplify: float | None = None,
    donor_rows_per_chunk: int = 256,
    compute_device: str | torch.device = "cpu",
    compute_dtype: str | torch.dtype = torch.float64,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Lfm2SparseSemanticCodebook:
    """Bake a bounded source-norm reclosure into an existing sparse codebook.

    For each non-direct donor row, this computes ``rho = ||C_i A|| / ||D_i||``
    and multiplies its existing coefficients by
    ``min(max_scale, rho ** -gamma)``.  A configured minimum cosine leaves
    low-cosine rows unamplified.  Direct mapped rows remain bit-exact one-hot.
    The function changes no CSR edge, adds no coefficient, and performs no
    solve, gradient, optimizer, training, pruning, quantization, or model call.
    """

    if not isinstance(base_codebook, Lfm2SparseSemanticCodebook):
        raise TypeError("base_codebook must be Lfm2SparseSemanticCodebook.")
    donor = _validated_table(donor_table, "donor_table")
    anchors = _validated_table(anchor_table, "anchor_table")
    donor_count, hidden_size = map(int, donor.shape)
    target_count, anchor_hidden = map(int, anchors.shape)
    base_receipt = base_codebook.as_dict()
    if (
        hidden_size != anchor_hidden
        or base_receipt.get("donor_row_count") != donor_count
        or base_receipt.get("target_row_count") != target_count
        or base_receipt.get("hidden_size") != hidden_size
    ):
        raise Lfm2SparseSemanticCodebookError(
            "Norm reclosure tables do not match the base codebook shape."
        )
    mapping = _validated_mapping(
        base_codebook.target_to_donor,
        target_row_count=target_count,
        donor_row_count=donor_count,
    )
    gamma_value = _positive_float(gamma, "gamma")
    max_scale_value = _finite_float(max_scale, "max_scale")
    if max_scale_value < 1.0:
        raise Lfm2SparseSemanticCodebookError(
            "max_scale must be at least 1 so it bounds amplification."
        )
    if minimum_cosine_to_amplify is None:
        minimum_cosine = None
    else:
        minimum_cosine = _finite_float(
            minimum_cosine_to_amplify, "minimum_cosine_to_amplify"
        )
        if minimum_cosine < -1.0 or minimum_cosine > 1.0:
            raise Lfm2SparseSemanticCodebookError(
                "minimum_cosine_to_amplify must lie in [-1, 1]."
            )
    row_chunk = min(
        _positive_int(donor_rows_per_chunk, "donor_rows_per_chunk"), donor_count
    )
    device = _compute_device(compute_device, "compute_device")
    dtype = _compute_dtype(compute_dtype, "compute_dtype")

    donor_sha = _tensor_sha256(donor, rows_per_chunk=row_chunk)
    anchor_sha = _tensor_sha256(anchors, rows_per_chunk=row_chunk)
    mapping_sha = _tensor_sha256(mapping, rows_per_chunk=row_chunk)
    if (
        base_receipt.get("donor_table_sha256") != donor_sha
        or base_receipt.get("anchor_table_sha256") != anchor_sha
        or base_receipt.get("target_to_donor_sha256") != mapping_sha
    ):
        raise Lfm2SparseSemanticCodebookError(
            "Norm reclosure requires the exact tables authenticated by the base codebook."
        )
    offsets = base_codebook.semantic_code_offsets
    target_ids = base_codebook.semantic_code_target_ids
    base_coefficients = base_codebook.semantic_code_coefficients
    base_content_sha = canonical_sha256(
        {
            "donor_row_count": donor_count,
            "target_row_count": target_count,
            "hidden_size": hidden_size,
            "semantic_code_offsets_sha256": _tensor_sha256(offsets),
            "semantic_code_target_ids_sha256": _tensor_sha256(target_ids),
            "semantic_code_coefficients_sha256": _tensor_sha256(base_coefficients),
            "target_to_donor_sha256": mapping_sha,
        }
    )
    phase_body = {
        "algorithm": _NORM_RECLOSURE_ALGORITHM,
        "base_semantic_code_content_sha256": base_content_sha,
        "donor_table_sha256": donor_sha,
        "anchor_table_sha256": anchor_sha,
        "target_to_donor_sha256": mapping_sha,
        "gamma": gamma_value,
        "max_scale": max_scale_value,
        "minimum_cosine_to_amplify": minimum_cosine,
        "donor_rows_per_chunk": row_chunk,
        "compute_placement": {
            "device": str(device),
            "dtype": _dtype_name(dtype),
        },
        "publication_placement": {"device": "cpu", "dtype": "float32"},
    }
    phase_cache_key = canonical_sha256(phase_body)
    emit, progress_status = _progress_emitter(
        progress_callback, phase="coefficient-only-norm-reclosure"
    )
    emit(
        "phase-start",
        phase_cache_key=phase_cache_key,
        base_codebook_artifact_sha256=base_codebook.artifact_sha256,
        base_semantic_code_content_sha256=base_content_sha,
        gamma=gamma_value,
        max_scale=max_scale_value,
        minimum_cosine_to_amplify=minimum_cosine,
        donor_row_count=donor_count,
        donor_rows_per_chunk=row_chunk,
        compute_device=str(device),
        compute_dtype=_dtype_name(dtype),
    )

    direct_inverse = _direct_inverse(mapping, donor_count)
    published_coefficients = base_coefficients.clone()
    before_ratios: list[Tensor] = []
    after_ratios: list[Tensor] = []
    applied_scales: list[Tensor] = []
    before_cosines: list[Tensor] = []
    after_cosines: list[Tensor] = []
    before_relative_errors: list[Tensor] = []
    after_relative_errors: list[Tensor] = []
    row_reclosure_diagnostics: list[dict[str, Any]] = []
    scale_cap_count = 0
    amplified_count = 0
    unresolved_count = 0
    l2_worsened_count = 0
    l2_improved_count = 0
    l2_unchanged_count = 0
    before_residual_sum_squares = 0.0
    after_residual_sum_squares = 0.0
    source_sum_squares = 0.0
    before_maximum_absolute_residual = 0.0
    after_maximum_absolute_residual = 0.0
    before_maximum_row_residual = 0.0
    after_maximum_row_residual = 0.0
    cosine_clamp_count = 0
    maximum_absolute_cosine_clamp = 0.0
    after_cosine_sum_all_rows = 0.0

    with torch.no_grad():
        anchor_compute = anchors.to(device=device, dtype=dtype)
        _require_finite_compute(anchor_compute, "norm reclosure anchor cast")
        for row_start in range(0, donor_count, row_chunk):
            row_stop = min(row_start + row_chunk, donor_count)
            edge_start = int(offsets[row_start])
            edge_stop = int(offsets[row_stop])
            local_lengths_cpu = (
                offsets[row_start + 1 : row_stop + 1] - offsets[row_start:row_stop]
            )
            local_lengths = local_lengths_cpu.to(device=device)
            edge_rows = torch.repeat_interleave(
                torch.arange(
                    row_stop - row_start,
                    device=device,
                    dtype=torch.int64,
                ),
                local_lengths,
            )
            local_target_ids = target_ids[edge_start:edge_stop].to(device=device)
            local_base_coefficients = base_coefficients[edge_start:edge_stop].to(
                device=device, dtype=dtype
            )
            source = donor[row_start:row_stop].to(device=device, dtype=dtype)
            _require_finite_compute(source, "norm reclosure donor chunk cast")
            before = source.new_zeros((row_stop - row_start, hidden_size))
            before.index_add_(
                0,
                edge_rows,
                anchor_compute.index_select(0, local_target_ids)
                * local_base_coefficients.unsqueeze(1),
            )
            _require_finite_compute(before, "base semantic reconstruction")
            source_norm = torch.linalg.vector_norm(source, dim=1)
            before_norm = torch.linalg.vector_norm(before, dim=1)
            zero_source = source_norm == 0
            zero_reconstruction = before_norm == 0
            if bool(zero_source.any()) or bool(zero_reconstruction.any()):
                bad_local = torch.nonzero(
                    zero_source | zero_reconstruction, as_tuple=False
                ).reshape(-1)
                bad_donor_ids = [
                    row_start + int(value)
                    for value in bad_local[:16].to(device="cpu").tolist()
                ]
                raise Lfm2SparseSemanticCodebookError(
                    "Norm reclosure fails closed on zero source or reconstructed "
                    f"row norms; donor IDs include {bad_donor_ids}."
                )
            _require_finite_compute(source_norm, "source row norms")
            _require_finite_compute(before_norm, "base reconstruction row norms")
            before_ratio = before_norm / source_norm
            raw_before_cosine = (before * source).sum(dim=1) / (
                before_norm * source_norm
            )
            _require_finite_compute(before_ratio, "base reconstruction norm ratios")
            _require_finite_compute(raw_before_cosine, "base reconstruction cosines")
            before_cosine = raw_before_cosine.clamp(min=-1.0, max=1.0)
            before_clamp_delta = (before_cosine - raw_before_cosine).abs()
            cosine_clamp_count += int((before_clamp_delta > 0).sum().item())
            maximum_absolute_cosine_clamp = max(
                maximum_absolute_cosine_clamp,
                float(before_clamp_delta.max().item()),
            )
            raw_scale = torch.pow(before_ratio, -gamma_value)
            _require_finite_compute(raw_scale, "raw norm reclosure scales")
            scale = torch.minimum(
                raw_scale,
                torch.tensor(max_scale_value, device=device, dtype=dtype),
            )
            direct_mask = (direct_inverse[row_start:row_stop] >= 0).to(device=device)
            non_direct_mask = ~direct_mask
            if minimum_cosine is None:
                unresolved = torch.zeros_like(non_direct_mask)
            else:
                unresolved = (
                    non_direct_mask
                    & (raw_scale > 1.0)
                    & (before_cosine < minimum_cosine)
                )
                scale = torch.where(unresolved, torch.ones_like(scale), scale)
            scale = torch.where(direct_mask, torch.ones_like(scale), scale)
            capped = non_direct_mask & ~unresolved & (raw_scale > max_scale_value)
            amplified = non_direct_mask & (scale > 1.0)
            edge_direct = direct_mask.index_select(0, edge_rows)
            edge_scale = scale.index_select(0, edge_rows)
            scaled = (local_base_coefficients * edge_scale).to(dtype=torch.float32)
            local_published = torch.where(
                edge_direct,
                base_coefficients[edge_start:edge_stop].to(device=device),
                scaled,
            )
            _require_finite_compute(
                local_published, "published norm-reclosed coefficients"
            )
            published_coefficients[edge_start:edge_stop].copy_(
                local_published.to(device="cpu")
            )
            after = source.new_zeros((row_stop - row_start, hidden_size))
            after.index_add_(
                0,
                edge_rows,
                anchor_compute.index_select(0, local_target_ids)
                * local_published.to(dtype=dtype).unsqueeze(1),
            )
            _require_finite_compute(after, "norm-reclosed semantic reconstruction")
            after_norm = torch.linalg.vector_norm(after, dim=1)
            if bool((after_norm == 0).any()):
                raise Lfm2SparseSemanticCodebookError(
                    "Norm reclosure produced a zero reconstructed row norm."
                )
            after_ratio = after_norm / source_norm
            raw_after_cosine = (after * source).sum(dim=1) / (after_norm * source_norm)
            _require_finite_compute(raw_after_cosine, "norm-reclosed raw cosines")
            after_cosine = raw_after_cosine.clamp(min=-1.0, max=1.0)
            after_clamp_delta = (after_cosine - raw_after_cosine).abs()
            cosine_clamp_count += int((after_clamp_delta > 0).sum().item())
            maximum_absolute_cosine_clamp = max(
                maximum_absolute_cosine_clamp,
                float(after_clamp_delta.max().item()),
            )
            before_residual = before - source
            after_residual = after - source
            before_row_error = torch.linalg.vector_norm(before_residual, dim=1)
            after_row_error = torch.linalg.vector_norm(after_residual, dim=1)
            before_relative = before_row_error / source_norm
            after_relative = after_row_error / source_norm
            for value, name in (
                (after_ratio, "norm-reclosed norm ratios"),
                (after_cosine, "norm-reclosed cosines"),
                (before_relative, "base relative reconstruction errors"),
                (after_relative, "norm-reclosed relative reconstruction errors"),
            ):
                _require_finite_compute(value, name)

            selected = non_direct_mask
            selected_local_ids = (
                torch.nonzero(selected, as_tuple=False).reshape(-1).to(device="cpu")
            )
            selected_before_ratio = before_ratio[selected].to(device="cpu")
            selected_after_ratio = after_ratio[selected].to(device="cpu")
            selected_scale = scale[selected].to(device="cpu")
            selected_before_cosine = before_cosine[selected].to(device="cpu")
            selected_after_cosine = after_cosine[selected].to(device="cpu")
            selected_before_relative = before_relative[selected].to(device="cpu")
            selected_after_relative = after_relative[selected].to(device="cpu")
            selected_capped = capped[selected].to(device="cpu")
            selected_amplified = amplified[selected].to(device="cpu")
            selected_unresolved = unresolved[selected].to(device="cpu")
            before_ratios.append(selected_before_ratio)
            after_ratios.append(selected_after_ratio)
            applied_scales.append(selected_scale)
            before_cosines.append(selected_before_cosine)
            after_cosines.append(selected_after_cosine)
            before_relative_errors.append(selected_before_relative)
            after_relative_errors.append(selected_after_relative)
            for selected_index, local_id_value in enumerate(
                selected_local_ids.tolist()
            ):
                local_id = int(local_id_value)
                row_reclosure_diagnostics.append(
                    {
                        "donor_id": row_start + local_id,
                        "semantic_code_edge_count": int(
                            local_lengths_cpu[local_id].item()
                        ),
                        "before_norm_ratio": float(
                            selected_before_ratio[selected_index].item()
                        ),
                        "after_norm_ratio": float(
                            selected_after_ratio[selected_index].item()
                        ),
                        "applied_scale": float(selected_scale[selected_index].item()),
                        "before_cosine": float(
                            selected_before_cosine[selected_index].item()
                        ),
                        "after_cosine": float(
                            selected_after_cosine[selected_index].item()
                        ),
                        "before_relative_l2_error": float(
                            selected_before_relative[selected_index].item()
                        ),
                        "after_relative_l2_error": float(
                            selected_after_relative[selected_index].item()
                        ),
                        "amplified": bool(selected_amplified[selected_index].item()),
                        "scale_cap_applied": bool(
                            selected_capped[selected_index].item()
                        ),
                        "unresolved_low_cosine": bool(
                            selected_unresolved[selected_index].item()
                        ),
                    }
                )
            scale_cap_count += int(capped.sum().item())
            amplified_count += int(amplified.sum().item())
            unresolved_count += int(unresolved.sum().item())
            l2_worsened_count += int(
                (selected & (after_relative > before_relative)).sum().item()
            )
            l2_improved_count += int(
                (selected & (after_relative < before_relative)).sum().item()
            )
            l2_unchanged_count += int(
                (selected & (after_relative == before_relative)).sum().item()
            )
            before_residual_sum_squares += float(before_residual.square().sum().item())
            after_residual_sum_squares += float(after_residual.square().sum().item())
            source_sum_squares += float(source.square().sum().item())
            before_maximum_absolute_residual = max(
                before_maximum_absolute_residual,
                float(before_residual.abs().max().item()),
            )
            after_maximum_absolute_residual = max(
                after_maximum_absolute_residual,
                float(after_residual.abs().max().item()),
            )
            before_maximum_row_residual = max(
                before_maximum_row_residual,
                float(before_row_error.max().item()),
            )
            after_maximum_row_residual = max(
                after_maximum_row_residual,
                float(after_row_error.max().item()),
            )
            after_cosine_sum_all_rows += float(after_cosine.sum().item())
            emit(
                "row-chunk-complete",
                phase_cache_key=phase_cache_key,
                donor_row_start=row_start,
                donor_row_stop=row_stop,
                completed_donor_rows=row_stop,
                total_donor_rows=donor_count,
                amplified_non_direct_row_count=int(amplified.sum().item()),
                capped_non_direct_row_count=int(capped.sum().item()),
                unresolved_non_direct_row_count=int(unresolved.sum().item()),
            )

    if not torch.equal(base_codebook.semantic_code_offsets, offsets):
        raise AssertionError("Norm reclosure unexpectedly changed offsets.")
    if not torch.equal(base_codebook.semantic_code_target_ids, target_ids):
        raise AssertionError("Norm reclosure unexpectedly changed target IDs.")
    for donor_id in mapping.tolist():
        start, stop = int(offsets[donor_id]), int(offsets[donor_id + 1])
        if (
            stop - start != 1
            or not torch.equal(
                published_coefficients[start:stop],
                base_coefficients[start:stop],
            )
            or published_coefficients[start].view(torch.int32).item()
            != torch.tensor(1.0, dtype=torch.float32).view(torch.int32).item()
        ):
            raise Lfm2SparseSemanticCodebookError(
                "Norm reclosure changed a protected direct one-hot row."
            )
    if int(published_coefficients.numel()) != int(base_coefficients.numel()):
        raise AssertionError("Norm reclosure changed the coefficient count.")

    before_ratio_values = torch.cat(before_ratios)
    after_ratio_values = torch.cat(after_ratios)
    scale_values = torch.cat(applied_scales)
    before_cosine_values = torch.cat(before_cosines)
    after_cosine_values = torch.cat(after_cosines)
    before_relative_values = torch.cat(before_relative_errors)
    after_relative_values = torch.cat(after_relative_errors)
    cosine_delta = (after_cosine_values - before_cosine_values).abs()
    if int(cosine_delta.numel()) == 0:
        maximum_cosine_delta = 0.0
        mean_cosine_delta = 0.0
    else:
        maximum_cosine_delta = float(cosine_delta.max().item())
        mean_cosine_delta = float(cosine_delta.mean().item())
    progress_count, progress_final = progress_status()
    coefficient_sha = _tensor_sha256(published_coefficients)
    reclosure_body = {
        "schema": LFM2_SPARSE_SEMANTIC_NORM_RECLOSURE_SCHEMA,
        "status": "complete-coefficient-only-bounded-row-norm-reclosure",
        "algorithm": _NORM_RECLOSURE_ALGORITHM,
        "phase_cache_key": phase_cache_key,
        "base_codebook_artifact_sha256": base_codebook.artifact_sha256,
        "base_semantic_code_content_sha256": base_content_sha,
        "base_semantic_code_coefficients_sha256": _tensor_sha256(base_coefficients),
        "published_semantic_code_coefficients_sha256": coefficient_sha,
        "donor_table_sha256": donor_sha,
        "anchor_table_sha256": anchor_sha,
        "target_to_donor_sha256": mapping_sha,
        "gamma": gamma_value,
        "max_scale": max_scale_value,
        "minimum_cosine_to_amplify": minimum_cosine,
        "minimum_cosine_threshold_enabled": minimum_cosine is not None,
        "formula": "scale=min(max_scale,norm_ratio**-gamma)",
        "threshold_policy": (
            "leave-low-cosine-amplification-at-one"
            if minimum_cosine is not None
            else "no-cosine-threshold"
        ),
        "diagnostic_population": "non-direct-donor-rows",
        "row_reclosure_diagnostics": row_reclosure_diagnostics,
        "row_reclosure_diagnostics_sha256": canonical_sha256(row_reclosure_diagnostics),
        "before_norm_ratio_quantiles": _distribution_summary(
            before_ratio_values, "before norm ratio"
        ),
        "after_norm_ratio_quantiles": _distribution_summary(
            after_ratio_values, "after norm ratio"
        ),
        "applied_scale_quantiles": _distribution_summary(scale_values, "applied scale"),
        "before_cosine_quantiles": _distribution_summary(
            before_cosine_values, "before cosine"
        ),
        "after_cosine_quantiles": _distribution_summary(
            after_cosine_values, "after cosine"
        ),
        "maximum_absolute_cosine_change": maximum_cosine_delta,
        "mean_absolute_cosine_change": mean_cosine_delta,
        "cosine_direction_is_analytically_invariant_under_positive_scaling": True,
        "measured_cosine_change_is_publication_rounding_only": True,
        "cosine_roundoff_clamp_count": cosine_clamp_count,
        "maximum_absolute_cosine_roundoff_clamp": (maximum_absolute_cosine_clamp),
        "reported_cosines_clamped_to_closed_unit_interval": True,
        "before_relative_l2_error_quantiles": _distribution_summary(
            before_relative_values, "before relative L2 error"
        ),
        "after_relative_l2_error_quantiles": _distribution_summary(
            after_relative_values, "after relative L2 error"
        ),
        "l2_worsened_non_direct_row_count": l2_worsened_count,
        "l2_improved_non_direct_row_count": l2_improved_count,
        "l2_unchanged_non_direct_row_count": l2_unchanged_count,
        "l2_worsening_is_reported_not_hidden": True,
        "amplified_non_direct_row_count": amplified_count,
        "scale_cap_applied_non_direct_row_count": scale_cap_count,
        "unresolved_non_direct_row_count": unresolved_count,
        "unresolved_definition": (
            "low-cosine-row-left-unamplified-by-configured-threshold"
            if minimum_cosine is not None
            else "none-no-threshold-configured"
        ),
        "zero_source_norm_row_count": 0,
        "zero_base_reconstruction_norm_row_count": 0,
        "nonfinite_intermediate_count": 0,
        "zero_or_nonfinite_rows_fail_closed_before_publication": True,
        "direct_anchor_count": target_count,
        "protected_direct_rows_preserved_bit_exact": True,
        "csr_offsets_changed": False,
        "csr_target_ids_changed": False,
        "coefficient_count_before": int(base_coefficients.numel()),
        "coefficient_count_after": int(published_coefficients.numel()),
        "continuous_coefficient_count_added": 0,
        "effective_child_parameter_count_added": 0,
        "routing_metadata_values_added": 0,
        "donor_rows_per_chunk": row_chunk,
        "compute_placement": {"device": str(device), "dtype": _dtype_name(dtype)},
        "publication_placement": {"device": "cpu", "dtype": "float32"},
        "progress_record_count": progress_count,
        "progress_final_sha256": progress_final,
        "progress_callback_supplied": progress_callback is not None,
        "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
    }
    norm_reclosure = _sealed(reclosure_body, "norm_reclosure_sha256")
    body = {
        key: value
        for key, value in base_receipt.items()
        if key
        not in {
            "artifact_sha256",
            "row_diagnostics",
            "row_diagnostics_sha256",
        }
    }
    body.update(
        {
            "schema": LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA,
            "status": "complete-coefficient-only-norm-reclosed-codebook",
            "phase_cache_key": phase_cache_key,
            "base_codebook_artifact_sha256": base_codebook.artifact_sha256,
            "base_semantic_code_content_sha256": base_content_sha,
            "semantic_code_coefficients_sha256": coefficient_sha,
            "reconstruction_root_mean_square_error": math.sqrt(
                after_residual_sum_squares / (donor_count * hidden_size)
            ),
            "relative_reconstruction_l2_error": math.sqrt(
                after_residual_sum_squares
                / max(source_sum_squares, torch.finfo(torch.float64).tiny)
            ),
            "mean_row_reconstruction_cosine": float(
                after_cosine_sum_all_rows / donor_count
            ),
            "maximum_row_reconstruction_l2_error": after_maximum_row_residual,
            "maximum_absolute_reconstruction_residual": (
                after_maximum_absolute_residual
            ),
            "metrics_authenticate_published_fp32_coefficients": True,
            "norm_reclosure": norm_reclosure,
            "norm_reclosure_sha256": norm_reclosure["norm_reclosure_sha256"],
            "norm_reclosure_before_global_root_mean_square_error": math.sqrt(
                before_residual_sum_squares / (donor_count * hidden_size)
            ),
            "norm_reclosure_before_global_relative_l2_error": math.sqrt(
                before_residual_sum_squares
                / max(source_sum_squares, torch.finfo(torch.float64).tiny)
            ),
            "norm_reclosure_before_maximum_row_l2_error": (before_maximum_row_residual),
            "norm_reclosure_before_maximum_absolute_residual": (
                before_maximum_absolute_residual
            ),
            "computed_chunk_count": math.ceil(donor_count / row_chunk),
            "checkpoint_reused_chunk_count": 0,
            "checkpoint_loader_supplied": False,
            "checkpoint_loader_invocation_count": 0,
            "checkpoint_publisher_supplied": False,
            "checkpoint_publisher_invocation_count": 0,
            "progress_record_count": progress_count,
            "progress_final_sha256": progress_final,
            "progress_callback_supplied": progress_callback is not None,
            "analytic_operation_audit": {
                "base_codebook_reused": True,
                "candidate_discovery_calls": 0,
                "coefficient_solve_calls": 0,
                "coefficient_refit_calls": 0,
                "row_norm_reclosure_chunk_count": math.ceil(donor_count / row_chunk),
                "continuous_coefficient_count_added": 0,
            },
            "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
        }
    )
    return Lfm2SparseSemanticCodebook(
        offsets,
        target_ids,
        published_coefficients,
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
        raise Lfm2SparseSemanticCodebookError(
            "Persistence tensor inventory is invalid."
        )
    for record in records:
        name = record.get("name")
        dtype = record.get("dtype")
        shape = record.get("shape")
        if (
            not isinstance(name, str)
            or dtype not in {"int64", "float32"}
            or not isinstance(shape, list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            )
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
    "LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_CHUNK_SCHEMA",
    "LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA",
    "LFM2_RESIDUAL_OMP_SEMANTIC_CANDIDATE_REQUEST_SCHEMA",
    "LFM2_RESIDUAL_OMP_NUMERIC_BACKEND_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CANDIDATE_CHUNK_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CANDIDATE_INVENTORY_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CANDIDATE_REQUEST_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_COEFFICIENT_CHUNK_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_COEFFICIENT_REQUEST_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_NORM_RECLOSURE_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_PERSISTENCE_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_PROGRESS_SCHEMA",
    "Lfm2SparseSemanticCandidateChunkCheckpoint",
    "Lfm2SparseSemanticCandidateInventory",
    "Lfm2SparseSemanticCodebook",
    "Lfm2SparseSemanticCodebookError",
    "Lfm2SparseSemanticCoefficientChunkCheckpoint",
    "build_lfm2_residual_omp_numeric_backend_receipt",
    "compile_lfm2_sparse_semantic_codebook",
    "discover_lfm2_residual_omp_semantic_candidates",
    "discover_lfm2_sparse_semantic_candidates",
    "load_lfm2_sparse_semantic_candidate_chunk",
    "load_lfm2_sparse_semantic_candidate_inventory",
    "load_lfm2_sparse_semantic_codebook",
    "load_lfm2_sparse_semantic_coefficient_chunk",
    "reclose_lfm2_sparse_semantic_codebook_norms",
    "save_lfm2_sparse_semantic_candidate_chunk",
    "save_lfm2_sparse_semantic_candidate_inventory",
    "save_lfm2_sparse_semantic_codebook",
    "save_lfm2_sparse_semantic_coefficient_chunk",
]
