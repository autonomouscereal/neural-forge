"""Closed-form functional refit of an LFM2 sparse semantic output channel.

The sparse semantic codebook is the input route (``C_in``).  This module keeps
that authenticated artifact immutable and refits only a second coefficient
vector (``C_out``) on the same integer supports.  Support widths are derived
from the authenticated CSR layout; there is no special case for k16, k32, or
any other positive width.

The refit consumes tables and a metric assembled from an already-cached trace.
It never constructs or executes a model and performs no gradient, optimizer,
training, pruning, quantization, candidate search, or global-map solve.  Every
solve is a bounded closed-form ridge system local to one donor row.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    Lfm2SparseSemanticCodebook,
)


LFM2_FUNCTIONAL_OUTPUT_REFIT_REQUEST_SCHEMA = (
    "neural-forge-lfm2-functional-output-refit-request-v1"
)
LFM2_FUNCTIONAL_OUTPUT_REFIT_RESULT_SCHEMA = (
    "neural-forge-lfm2-functional-output-refit-result-v1"
)
LFM2_FUNCTIONAL_OUTPUT_REFIT_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-functional-output-refit-progress-v1"
)
LFM2_FUNCTIONAL_OUTPUT_REFIT_ALGORITHM = (
    "fixed-authenticated-support-local-functional-ridge-v1"
)

_SUPPORTED_TABLE_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)
_SUPPORTED_SOLVE_DTYPES = frozenset({torch.float32, torch.float64})
_TENSOR_HASH_DOMAIN = b"neural-forge-lfm2-functional-output-refit-tensor-v1\0"
_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "model_objects_constructed": 0,
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "candidate_search_calls": 0,
    "support_mutations": 0,
    "global_map_solve_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2FunctionalOutputRefitError(RuntimeError):
    """Authenticated refit inputs or closed-form solve results are invalid."""


ProgressSink = Callable[[Mapping[str, Any]], None]


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise Lfm2FunctionalOutputRefitError(
                "Receipts cannot contain non-finite values."
            )
        return value
    raise Lfm2FunctionalOutputRefitError(
        f"Receipt contains unsupported {type(value).__name__}."
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _resolve_solve_dtype(value: str | torch.dtype) -> torch.dtype:
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    dtype = value if isinstance(value, torch.dtype) else aliases.get(value.lower())
    if dtype not in _SUPPORTED_SOLVE_DTYPES:
        raise Lfm2FunctionalOutputRefitError(
            "solve_dtype must be explicit FP32 or FP64."
        )
    assert isinstance(dtype, torch.dtype)
    return dtype


def _resolve_solve_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2FunctionalOutputRefitError("solve_device is invalid.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise Lfm2FunctionalOutputRefitError(
            "solve_device must be a CPU or CUDA device."
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Lfm2FunctionalOutputRefitError(
                "solve_device requests CUDA, but CUDA is unavailable."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise Lfm2FunctionalOutputRefitError(
                "solve_device requests an unavailable CUDA device."
            )
        return torch.device("cuda", index)
    return torch.device("cpu")


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2FunctionalOutputRefitError(
            f"{name} must be finite and nonnegative."
        )
    return result


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2FunctionalOutputRefitError(
            f"{name} must be finite and positive."
        )
    return result


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2FunctionalOutputRefitError(f"{name} must be positive.")
    return int(value)


def _validated_table(value: Tensor, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.dtype not in _SUPPORTED_TABLE_DTYPES
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2FunctionalOutputRefitError(
            f"{name} must be a finite, detached FP16/BF16/FP32/FP64 matrix."
        )
    return value.detach()


def _validated_metric(value: Tensor, hidden_size: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or tuple(value.shape) != (hidden_size, hidden_size)
        or value.dtype not in _SUPPORTED_TABLE_DTYPES
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2FunctionalOutputRefitError(
            "functional_metric must be a finite detached square floating matrix "
            "matching the table hidden size."
        )
    return value.detach()


def _support_layout(
    codebook: Lfm2SparseSemanticCodebook,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    offsets = codebook.semantic_code_offsets
    donor_count = int(codebook.receipt["donor_row_count"])
    target_count = int(codebook.receipt["target_row_count"])
    lengths = offsets[1:] - offsets[:-1]
    if int(lengths.numel()) != donor_count or bool((lengths <= 0).any()):
        raise Lfm2FunctionalOutputRefitError(
            "Authenticated codebook contains an empty or missing donor route."
        )
    direct_mask = torch.zeros(donor_count, dtype=torch.bool)
    direct_mask[codebook.target_to_donor] = True
    if int(direct_mask.sum().item()) != target_count:
        raise Lfm2FunctionalOutputRefitError(
            "Authenticated direct-route inventory is inconsistent."
        )
    if not bool(torch.all(lengths[direct_mask] == 1)):
        raise Lfm2FunctionalOutputRefitError(
            "Authenticated direct routes are not one-entry identities."
        )
    non_direct = torch.nonzero(~direct_mask, as_tuple=False).reshape(-1)
    histogram: dict[str, int] = {}
    for width, count in zip(*torch.unique(lengths, return_counts=True), strict=True):
        histogram[str(int(width))] = int(count)
    return lengths, non_direct, histogram


def _emit(
    progress: ProgressSink | None,
    *,
    event: str,
    stage: str,
    cache_key: str,
    **details: Any,
) -> None:
    if progress is None:
        return
    progress(
        _plain_json(
            {
                "schema": LFM2_FUNCTIONAL_OUTPUT_REFIT_PROGRESS_SCHEMA,
                "event": event,
                "stage": stage,
                "cache_key": cache_key,
                **details,
            }
        )
    )


def _validated_request_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate a request while excluding execution-only controls from math."""

    if not isinstance(value, Mapping):
        raise Lfm2FunctionalOutputRefitError("Output refit request must be a mapping.")
    request = _plain_json(value)
    supplied_cache_key = request.pop("cache_key", None)
    execution_contract = request.pop("execution_contract", None)
    if (
        request.get("schema") != LFM2_FUNCTIONAL_OUTPUT_REFIT_REQUEST_SCHEMA
        or supplied_cache_key != canonical_sha256(request)
    ):
        raise Lfm2FunctionalOutputRefitError(
            "Output refit request math identity is invalid."
        )
    if not isinstance(execution_contract, Mapping):
        raise Lfm2FunctionalOutputRefitError(
            "Output refit execution contract is missing."
        )
    execution = _plain_json(execution_contract)
    _positive_int(execution.get("rows_per_chunk"), "rows_per_chunk")
    if execution.get("progress_logging_changes_math_identity") is not False:
        raise Lfm2FunctionalOutputRefitError(
            "Output refit execution contract does not isolate logging from math."
        )
    return {
        **request,
        "cache_key": supplied_cache_key,
        "execution_contract": execution,
    }


def lfm2_functional_output_refit_request(
    *,
    codebook: Lfm2SparseSemanticCodebook,
    donor_table: Tensor,
    target_table: Tensor,
    functional_metric: Tensor,
    functional_weight: float,
    solve_ridge: float,
    solve_device: str | torch.device = "cpu",
    solve_dtype: str | torch.dtype = torch.float64,
    rows_per_chunk: int = 512,
) -> dict[str, Any]:
    """Build the content-addressed identity for one closed-form refit."""

    if not isinstance(codebook, Lfm2SparseSemanticCodebook):
        raise TypeError("codebook must be an authenticated Lfm2SparseSemanticCodebook.")
    donor = _validated_table(donor_table, "donor_table")
    target = _validated_table(target_table, "target_table")
    donor_count = int(codebook.receipt["donor_row_count"])
    target_count = int(codebook.receipt["target_row_count"])
    if int(donor.shape[0]) != donor_count or int(target.shape[0]) != target_count:
        raise Lfm2FunctionalOutputRefitError(
            "Table row counts differ from the authenticated codebook."
        )
    if int(donor.shape[1]) != int(target.shape[1]):
        raise Lfm2FunctionalOutputRefitError(
            "Donor and target table hidden sizes differ."
        )
    hidden_size = int(donor.shape[1])
    metric = _validated_metric(functional_metric, hidden_size)
    weight = _finite_nonnegative(functional_weight, "functional_weight")
    ridge = _positive(solve_ridge, "solve_ridge")
    device = _resolve_solve_device(solve_device)
    dtype = _resolve_solve_dtype(solve_dtype)
    chunk_size = _positive_int(rows_per_chunk, "rows_per_chunk")
    lengths, non_direct, histogram = _support_layout(codebook)
    body = {
        "schema": LFM2_FUNCTIONAL_OUTPUT_REFIT_REQUEST_SCHEMA,
        "algorithm": LFM2_FUNCTIONAL_OUTPUT_REFIT_ALGORITHM,
        "input_codebook_artifact_sha256": codebook.artifact_sha256,
        "input_semantic_code_coefficients_sha256": _tensor_sha256(
            codebook.semantic_code_coefficients
        ),
        "semantic_code_offsets_sha256": _tensor_sha256(
            codebook.semantic_code_offsets
        ),
        "semantic_code_target_ids_sha256": _tensor_sha256(
            codebook.semantic_code_target_ids
        ),
        "target_to_donor_sha256": _tensor_sha256(codebook.target_to_donor),
        "donor_table_sha256": _tensor_sha256(donor),
        "target_table_sha256": _tensor_sha256(target),
        "functional_metric_sha256": _tensor_sha256(metric),
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "continuous_coefficient_count": int(
            codebook.semantic_code_coefficients.numel()
        ),
        "direct_row_count": target_count,
        "non_direct_row_count": int(non_direct.numel()),
        "maximum_support_width": int(lengths.max().item()),
        "support_width_histogram": histogram,
        "functional_weight": weight,
        "solve_ridge": ridge,
        "dtype_contract": {
            "donor_table_storage_dtype": _dtype_name(donor.dtype),
            "target_table_storage_dtype": _dtype_name(target.dtype),
            "functional_metric_storage_dtype": _dtype_name(metric.dtype),
            "solve_dtype": _dtype_name(dtype),
            "publication_dtype": "float32",
            "solve_device": str(device),
            "half_and_bfloat_inputs_promoted_before_linear_algebra": True,
        },
        "fixed_authenticated_supports_reused": True,
        "input_output_channels_separate": True,
        "input_coefficients_mutated": False,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    # ``body`` is the mathematical/numeric identity.  Chunking and progress
    # plumbing are execution choices: record them for diagnosis, but never
    # throw away a reusable result merely because orchestration changed.
    return {
        **body,
        "cache_key": canonical_sha256(body),
        "execution_contract": {
            "rows_per_chunk": chunk_size,
            "progress_logging_changes_math_identity": False,
        },
    }


@dataclass(frozen=True, slots=True)
class Lfm2FunctionalOutputRefitResult:
    """Immutable folded ``C_out`` coefficients and their sealed receipt."""

    output_semantic_code_coefficients: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        coefficients = (
            self.output_semantic_code_coefficients.detach()
            .to(device="cpu")
            .contiguous()
        )
        if (
            coefficients.dtype != torch.float32
            or coefficients.ndim != 1
            or coefficients.requires_grad
            or not bool(torch.isfinite(coefficients).all())
        ):
            raise Lfm2FunctionalOutputRefitError(
                "Published output coefficients must be a finite detached FP32 vector."
            )
        value = _plain_json(self.receipt)
        supplied = value.pop("result_sha256", None)
        if supplied != canonical_sha256(value):
            raise Lfm2FunctionalOutputRefitError("Output refit receipt seal is invalid.")
        request = _validated_request_receipt(value.get("request"))
        if (
            value.get("schema") != LFM2_FUNCTIONAL_OUTPUT_REFIT_RESULT_SCHEMA
            or value.get("cache_key") != request["cache_key"]
            or value.get("input_codebook_artifact_sha256")
            != request["input_codebook_artifact_sha256"]
            or value.get("input_semantic_code_coefficients_sha256_before")
            != request["input_semantic_code_coefficients_sha256"]
            or value.get("input_semantic_code_coefficients_sha256_after")
            != request["input_semantic_code_coefficients_sha256"]
            or value.get("dtype_contract") != request["dtype_contract"]
            or value.get("continuous_coefficient_count")
            != request["continuous_coefficient_count"]
            or value.get("direct_row_count") != request["direct_row_count"]
            or value.get("non_direct_row_count")
            != request["non_direct_row_count"]
            or value.get("maximum_support_width")
            != request["maximum_support_width"]
            or value.get("support_width_histogram")
            != request["support_width_histogram"]
            or value.get("fixed_authenticated_supports_reused") is not True
            or value.get("input_output_channels_separate") is not True
            or value.get("input_coefficients_mutated") is not False
            or value.get("output_coefficients_are_folded_child_state") is not True
            or value.get("output_semantic_code_coefficients_sha256")
            != _tensor_sha256(coefficients)
            or value.get("continuous_coefficient_count")
            != int(coefficients.numel())
            or value.get("operation_audit") != _OPERATION_AUDIT
        ):
            raise Lfm2FunctionalOutputRefitError(
                "Output refit tensor binding is invalid."
            )
        object.__setattr__(
            self, "output_semantic_code_coefficients", coefficients.clone()
        )
        object.__setattr__(
            self, "receipt", _deep_freeze({**value, "result_sha256": supplied})
        )

    @property
    def cache_key(self) -> str:
        return str(self.receipt["cache_key"])

    @property
    def result_sha256(self) -> str:
        return str(self.receipt["result_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def authenticate_lfm2_functional_output_refit_result(
    *,
    output_semantic_code_coefficients: Tensor,
    receipt: Mapping[str, Any],
    expected_cache_key: str,
) -> Lfm2FunctionalOutputRefitResult:
    """Admit loaded result values only for the caller's expected math key.

    Persistence code may choose its own container, but it must pass loaded
    values through this helper before cache reuse.  The result class validates
    the receipt seal, nested request identity, tensor binding, and operation
    audit; this helper additionally prevents a valid result for different math
    from being accepted merely because it occupied the requested cache path.
    """

    if (
        not isinstance(expected_cache_key, str)
        or len(expected_cache_key) != 64
        or any(character not in "0123456789abcdef" for character in expected_cache_key)
    ):
        raise Lfm2FunctionalOutputRefitError(
            "expected_cache_key must be a lowercase SHA-256 digest."
        )
    result = Lfm2FunctionalOutputRefitResult(
        output_semantic_code_coefficients, receipt
    )
    if result.cache_key != expected_cache_key:
        raise Lfm2FunctionalOutputRefitError(
            "Loaded output refit result belongs to a different math cache key."
        )
    return result


def refit_lfm2_functional_output_coefficients(
    *,
    codebook: Lfm2SparseSemanticCodebook,
    donor_table: Tensor,
    target_table: Tensor,
    functional_metric: Tensor,
    functional_weight: float,
    solve_ridge: float,
    solve_device: str | torch.device = "cpu",
    solve_dtype: str | torch.dtype = torch.float64,
    rows_per_chunk: int = 512,
    progress: ProgressSink | None = None,
) -> Lfm2FunctionalOutputRefitResult:
    """Refit ``C_out`` on fixed authenticated supports with local solves."""

    request = lfm2_functional_output_refit_request(
        codebook=codebook,
        donor_table=donor_table,
        target_table=target_table,
        functional_metric=functional_metric,
        functional_weight=functional_weight,
        solve_ridge=solve_ridge,
        solve_device=solve_device,
        solve_dtype=solve_dtype,
        rows_per_chunk=rows_per_chunk,
    )
    cache_key = str(request["cache_key"])
    dtype_contract = request["dtype_contract"]
    _emit(
        progress,
        event="refit-start",
        stage="preflight",
        cache_key=cache_key,
        input_codebook_artifact_sha256=codebook.artifact_sha256,
        support_width_histogram=request["support_width_histogram"],
        dtype_contract=dtype_contract,
    )
    _emit(
        progress,
        event="cache-identity-ready",
        stage="cache",
        cache_key=cache_key,
        cache_status="caller-may-reuse-sealed-result-by-cache-key",
    )

    device = _resolve_solve_device(solve_device)
    dtype = _resolve_solve_dtype(solve_dtype)
    chunk_size = _positive_int(rows_per_chunk, "rows_per_chunk")
    donor = donor_table.detach().to(device=device, dtype=dtype)
    target = target_table.detach().to(device=device, dtype=dtype)
    metric = functional_metric.detach().to(device=device, dtype=dtype)
    metric = (metric + metric.transpose(0, 1)) * 0.5
    identity = torch.eye(int(metric.shape[0]), device=device, dtype=dtype)
    combined_metric = metric + float(functional_weight) * identity
    target_metric = target @ combined_metric

    offsets = codebook.semantic_code_offsets
    target_ids = codebook.semantic_code_target_ids
    input_coefficients = codebook.semantic_code_coefficients
    input_coefficients_sha256 = _tensor_sha256(input_coefficients)
    output_coefficients = input_coefficients.clone()
    lengths, non_direct, histogram = _support_layout(codebook)
    solve_failure_count = 0
    solve_chunk_count = 0
    maximum_solve_residual = 0.0
    width_groups = sorted(
        int(value)
        for value in torch.unique(lengths.index_select(0, non_direct)).tolist()
    )

    with torch.no_grad():
        for width in width_groups:
            group_rows = non_direct[lengths.index_select(0, non_direct).eq(width)]
            _emit(
                progress,
                event="support-group-start",
                stage="solve",
                cache_key=cache_key,
                support_width=width,
                group_row_count=int(group_rows.numel()),
            )
            column = torch.arange(width, dtype=torch.int64).unsqueeze(0)
            for chunk_index, start in enumerate(
                range(0, int(group_rows.numel()), chunk_size)
            ):
                stop = min(start + chunk_size, int(group_rows.numel()))
                donor_ids_cpu = group_rows[start:stop]
                positions = offsets.index_select(0, donor_ids_cpu).unsqueeze(1) + column
                support_ids = target_ids.index_select(
                    0, positions.reshape(-1)
                ).reshape(stop - start, width)
                donor_ids = donor_ids_cpu.to(device=device)
                support = support_ids.to(device=device)
                basis = target[support]
                basis_metric = target_metric[support]
                gram = torch.bmm(basis_metric, basis.transpose(1, 2))
                gram.diagonal(dim1=-2, dim2=-1).add_(float(solve_ridge))
                source = donor.index_select(0, donor_ids)
                rhs = torch.bmm(basis_metric, source.unsqueeze(2))
                base = input_coefficients.index_select(
                    0, positions.reshape(-1)
                ).reshape(stop - start, width, 1).to(device=device, dtype=dtype)
                rhs.add_(base, alpha=float(solve_ridge))
                solved, info = torch.linalg.solve_ex(gram, rhs, check_errors=False)
                failed = info.ne(0) | ~torch.isfinite(solved).all(dim=(1, 2))
                if bool(failed.any()):
                    solve_failure_count += int(failed.sum().item())
                    solved[failed] = torch.linalg.lstsq(
                        gram[failed], rhs[failed]
                    ).solution
                residual = torch.bmm(gram, solved) - rhs
                maximum_solve_residual = max(
                    maximum_solve_residual,
                    float(
                        torch.linalg.vector_norm(residual, dim=(1, 2)).max().item()
                    ),
                )
                published = solved.squeeze(2).to(device="cpu", dtype=torch.float32)
                output_coefficients.index_copy_(
                    0, positions.reshape(-1), published.reshape(-1)
                )
                solve_chunk_count += 1
                _emit(
                    progress,
                    event="solve-chunk-complete",
                    stage="solve",
                    cache_key=cache_key,
                    support_width=width,
                    chunk_index=chunk_index,
                    chunk_row_start=start,
                    chunk_row_stop=stop,
                    group_row_count=int(group_rows.numel()),
                    solve_failure_count=solve_failure_count,
                    solve_dtype=_dtype_name(dtype),
                    solve_device=str(device),
                )
            _emit(
                progress,
                event="support-group-complete",
                stage="solve",
                cache_key=cache_key,
                support_width=width,
                group_row_count=int(group_rows.numel()),
            )

    if not bool(torch.isfinite(output_coefficients).all()):
        raise Lfm2FunctionalOutputRefitError(
            "Closed-form output coefficients contain non-finite values."
        )
    if _tensor_sha256(input_coefficients) != input_coefficients_sha256:
        raise Lfm2FunctionalOutputRefitError(
            "Input semantic coefficients changed during output refit."
        )
    direct_positions = offsets.index_select(0, codebook.target_to_donor)
    if not torch.equal(
        output_coefficients.index_select(0, direct_positions),
        input_coefficients.index_select(0, direct_positions),
    ):
        raise Lfm2FunctionalOutputRefitError(
            "Direct one-hot semantic routes changed during output refit."
        )

    body = {
        "schema": LFM2_FUNCTIONAL_OUTPUT_REFIT_RESULT_SCHEMA,
        "algorithm": LFM2_FUNCTIONAL_OUTPUT_REFIT_ALGORITHM,
        "cache_key": cache_key,
        "request": request,
        "input_codebook_artifact_sha256": codebook.artifact_sha256,
        "input_semantic_code_coefficients_sha256_before": (
            input_coefficients_sha256
        ),
        "input_semantic_code_coefficients_sha256_after": _tensor_sha256(
            input_coefficients
        ),
        "output_semantic_code_coefficients_sha256": _tensor_sha256(
            output_coefficients
        ),
        "continuous_coefficient_count": int(output_coefficients.numel()),
        "direct_row_count": int(codebook.target_to_donor.numel()),
        "non_direct_row_count": int(non_direct.numel()),
        "maximum_support_width": int(lengths.max().item()),
        "support_width_histogram": histogram,
        "solve_chunk_count": solve_chunk_count,
        "solve_failure_count": solve_failure_count,
        "maximum_solve_residual": maximum_solve_residual,
        "coefficient_abs_maximum": float(output_coefficients.abs().max().item()),
        "dtype_contract": dtype_contract,
        "fixed_authenticated_supports_reused": True,
        "input_output_channels_separate": True,
        "input_coefficients_mutated": False,
        "output_coefficients_are_folded_child_state": True,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    receipt = {**body, "result_sha256": canonical_sha256(body)}
    result = Lfm2FunctionalOutputRefitResult(output_coefficients, receipt)
    _emit(
        progress,
        event="refit-complete",
        stage="publish",
        cache_key=cache_key,
        result_sha256=result.result_sha256,
        solve_chunk_count=solve_chunk_count,
        solve_failure_count=solve_failure_count,
        output_semantic_code_coefficients_sha256=receipt[
            "output_semantic_code_coefficients_sha256"
        ],
        dtype_contract=dtype_contract,
    )
    return result


def build_lfm2_functional_output_sparse_design(
    codebook: Lfm2SparseSemanticCodebook,
    output_coefficients: Tensor,
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Materialize only a sparse diagnostic design; never a dense global map."""

    if not isinstance(codebook, Lfm2SparseSemanticCodebook):
        raise TypeError("codebook must be an authenticated Lfm2SparseSemanticCodebook.")
    coefficients = output_coefficients.detach().to(device="cpu").contiguous()
    if (
        coefficients.dtype != torch.float32
        or coefficients.ndim != 1
        or int(coefficients.numel())
        != int(codebook.semantic_code_target_ids.numel())
        or coefficients.requires_grad
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise Lfm2FunctionalOutputRefitError(
            "output_coefficients do not match the authenticated sparse layout."
        )
    target_device = _resolve_solve_device(device)
    donor_count = int(codebook.receipt["donor_row_count"])
    target_count = int(codebook.receipt["target_row_count"])
    lengths = (
        codebook.semantic_code_offsets[1:]
        - codebook.semantic_code_offsets[:-1]
    )
    rows = torch.repeat_interleave(
        torch.arange(donor_count, dtype=torch.int64), lengths
    )
    design = torch.sparse_coo_tensor(
        torch.stack((rows, codebook.semantic_code_target_ids)),
        coefficients,
        size=(donor_count, target_count),
        dtype=torch.float32,
        device="cpu",
        check_invariants=True,
    ).coalesce()
    return design.to(device=target_device)


def json_line_functional_output_refit_progress(
    details: Mapping[str, Any],
) -> None:
    """Default flush-immediate JSON-lines sink for long refit orchestration."""

    print(
        json.dumps(
            _plain_json(details),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
