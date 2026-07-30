"""Analytic semantic-step folding for tokenizer-changing LFM2 children.

The target tokenizer may spell one donor action with several target tokens.
Advancing the transformer body once per spelling token changes the semantic
clock and is not equivalent to advancing it once per donor action.  This module
keeps those concerns separate:

* an authenticated donor-to-target span inventory remains the text-emission
  route ``S``;
* legacy hard-anchor folds may use the positional operator ``A_span`` and one
  deterministic ridge solve ``W_donor ~= A_span @ W_target``;
* dual-channel runtimes instead use an independently authenticated sparse
  semantic code ``C`` for embeddings and logits; and
* the child body receives exactly one reconstructed semantic embedding per
  chosen donor action, regardless of the target span length.

No parent/model forward, trace recapture, gradient, optimizer, training,
quantization, pruning, or network operation occurs here.  The sparse operator
is compilation/runtime metadata and is never counted as learned parameters.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.dynamic_circuit_ir import canonical_sha256

try:  # The fold remains usable in minimal installs without policy plumbing.
    from neural_forge.numeric_execution_policy import (
        NumericDomain,
        NumericExecutionPolicy,
        NumericExecutionPolicyError,
        torch_numeric_dtype,
    )
except ImportError:  # pragma: no cover - repository installs include the policy
    NumericDomain = None  # type: ignore[assignment]
    NumericExecutionPolicy = None  # type: ignore[assignment]
    NumericExecutionPolicyError = ValueError  # type: ignore[assignment]
    torch_numeric_dtype = None  # type: ignore[assignment]


LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-semantic-step-span-evidence-v1"
)
LFM2_SEMANTIC_STEP_TOKEN_FOLD_SCHEMA = "neural-forge-lfm2-semantic-step-token-fold-v1"
LFM2_SEMANTIC_STEP_PROGRESS_SCHEMA = "neural-forge-lfm2-semantic-step-progress-v1"
LFM2_SEMANTIC_STEP_CODEC_RESULT_SCHEMA = (
    "neural-forge-lfm2-semantic-step-codec-result-v1"
)
LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA = (
    "neural-forge-lfm2-semantic-step-routing-runtime-v1"
)
LFM2_SPARSE_SEMANTIC_CODE_SCHEMA = "neural-forge-lfm2-sparse-semantic-code-v1"
LFM2_SEPARATE_SPARSE_SEMANTIC_CODE_PAIR_SCHEMA = (
    "neural-forge-lfm2-separate-sparse-semantic-code-pair-v1"
)
LFM2_SEMANTIC_STEP_LIVE_TABLE_SESSION_SCHEMA = (
    "neural-forge-lfm2-semantic-step-live-table-session-v1"
)
LFM2_SEMANTIC_STEP_CHUNK_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-semantic-step-solved-chunk-checkpoint-v1"
)
LFM2_SEMANTIC_STEP_CHUNK_REQUEST_SCHEMA = (
    "neural-forge-lfm2-semantic-step-solved-chunk-request-v1"
)

POSITIONAL_CHANNEL_DESIGNS = (
    "uniform",
    "linear-forward",
    "linear-reverse",
    "geometric-forward",
)

_SUPPORTED_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_LIMITATION_KINDS = frozenset(
    {
        "suffix-rewrite",
        "zero-width",
        "unmapped-control",
        "nonappend-only",
        "roundtrip-mismatch",
        "unresolved",
        "unsupported",
    }
)
_LIMITATION_REASONS = frozenset(
    {
        "target-suffix-rewrite",
        "zero-width-target-transition",
        "unmapped-or-ambiguous-donor-control",
        "donor-text-nonappend-only",
        "target-roundtrip-mismatch",
        "unresolved",
        "unsupported",
    }
)
_MODEL_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "model_forward_calls": 0,
    "quantized_parameters": 0,
    "pruned_parameters": 0,
    "network_access": False,
}


class Lfm2SemanticStepTokenFoldError(RuntimeError):
    """Semantic-step evidence, solve, or runtime state is invalid."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise Lfm2SemanticStepTokenFoldError(
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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SemanticStepTokenFoldError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SemanticStepTokenFoldError(f"{name} must be nonnegative.")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise Lfm2SemanticStepTokenFoldError(f"{name} must be finite and positive.")
    return float(value)


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticStepTokenFoldError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _dtype_name(value: Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def _routing_accumulation_dtype(table_dtype: torch.dtype) -> torch.dtype:
    if table_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    if table_dtype in {torch.float32, torch.float64}:
        return table_dtype
    raise Lfm2SemanticStepTokenFoldError(
        "Live child table has no supported routing accumulation dtype."
    )


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-semantic-step-tensor-v1\0")
    digest.update(_dtype_name(tensor).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sealed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = _plain_json(body)
    return {**value, field: canonical_sha256(value)}


def _verify_sealed(
    value: Mapping[str, Any],
    *,
    field: str,
    schema: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticStepTokenFoldError(f"{name} must be a mapping.")
    result = _plain_json(value)
    supplied = result.pop(field, None)
    if supplied != canonical_sha256(result):
        raise Lfm2SemanticStepTokenFoldError(f"{name} seal is invalid.")
    if result.get("schema") != schema:
        raise Lfm2SemanticStepTokenFoldError(f"{name} schema is unsupported.")
    assert isinstance(supplied, str)
    return {**result, field: supplied}


def _integer_span(value: Any, *, target_row_count: int, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2SemanticStepTokenFoldError(
            f"{name} must be a nonempty integer token span."
        )
    result = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item >= target_row_count
        ):
            raise Lfm2SemanticStepTokenFoldError(
                f"{name} leaves the target vocabulary."
            )
        result.append(int(item))
    if not result:
        raise Lfm2SemanticStepTokenFoldError(
            f"{name} is zero-width; semantic-step folding fails closed."
        )
    return tuple(result)


def seal_lfm2_semantic_step_span_evidence(
    donor_target_spans: Sequence[Sequence[int] | Mapping[str, Any]],
    *,
    target_row_count: int,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one exact, nonempty target span for every donor row.

    Limitation records from contextual tokenization are deliberately rejected.
    A suffix rewrite is not a span append, and a zero-width or unresolved row
    cannot be used to claim semantic-step equivalence.
    """

    target_count = _positive_int(target_row_count, "target_row_count")
    if (
        isinstance(donor_target_spans, (str, bytes))
        or not isinstance(donor_target_spans, Sequence)
        or not donor_target_spans
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "donor_target_spans must cover at least one donor row."
        )
    records: list[dict[str, Any]] = []
    for donor_id, raw in enumerate(donor_target_spans):
        if isinstance(raw, Mapping):
            observed_id = raw.get("donor_id", donor_id)
            if observed_id != donor_id:
                raise Lfm2SemanticStepTokenFoldError(
                    "Span records must be in exact donor-row order."
                )
            kind = raw.get("kind", raw.get("transition_kind", "span"))
            reason = raw.get("reason")
            if kind in _LIMITATION_KINDS or reason in _LIMITATION_REASONS:
                raise Lfm2SemanticStepTokenFoldError(
                    f"Donor row {donor_id} is {kind or reason!r}; only exact "
                    "nonempty target spans are accepted."
                )
            if kind not in {
                "span",
                "append",
                "canonical-contextual-append",
                "authenticated-special-control",
            }:
                raise Lfm2SemanticStepTokenFoldError(
                    f"Donor row {donor_id} transition kind is unsupported."
                )
            raw_span = raw.get(
                "target_token_ids",
                raw.get("target_append_token_ids"),
            )
        else:
            raw_span = raw
        span = _integer_span(
            raw_span,
            target_row_count=target_count,
            name=f"donor row {donor_id} target span",
        )
        records.append({"donor_id": donor_id, "target_token_ids": list(span)})
    lineage_value = {} if lineage is None else _plain_json(lineage)
    body = {
        "schema": LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA,
        "status": "complete-every-donor-nonempty-target-spans-sealed",
        "lineage": lineage_value,
        "donor_row_count": len(records),
        "target_row_count": target_count,
        "span_token_count": sum(len(record["target_token_ids"]) for record in records),
        "maximum_span_length": max(
            len(record["target_token_ids"]) for record in records
        ),
        "spans": records,
        "every_donor_row_classified_exactly_once": True,
        "all_spans_nonempty": True,
        "suffix_rewrites_accepted": False,
        "zero_width_rows_accepted": False,
        "unresolved_rows_accepted": False,
        "routing_metadata_is_integer_only": True,
        "operation_audit": dict(_MODEL_OPERATION_AUDIT),
    }
    return _sealed(body, "evidence_sha256")


def _verified_span_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _verify_sealed(
        value,
        field="evidence_sha256",
        schema=LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA,
        name="semantic-step span evidence",
    )
    expected_keys = {
        "schema",
        "status",
        "lineage",
        "donor_row_count",
        "target_row_count",
        "span_token_count",
        "maximum_span_length",
        "spans",
        "every_donor_row_classified_exactly_once",
        "all_spans_nonempty",
        "suffix_rewrites_accepted",
        "zero_width_rows_accepted",
        "unresolved_rows_accepted",
        "routing_metadata_is_integer_only",
        "operation_audit",
        "evidence_sha256",
    }
    if set(evidence) != expected_keys:
        raise Lfm2SemanticStepTokenFoldError(
            "Semantic-step span evidence fields are incomplete or unknown."
        )
    donor_count = _positive_int(evidence["donor_row_count"], "donor_row_count")
    target_count = _positive_int(evidence["target_row_count"], "target_row_count")
    raw_spans = evidence["spans"]
    if (
        not isinstance(raw_spans, list)
        or len(raw_spans) != donor_count
        or evidence["every_donor_row_classified_exactly_once"] is not True
        or evidence["all_spans_nonempty"] is not True
        or evidence["suffix_rewrites_accepted"] is not False
        or evidence["zero_width_rows_accepted"] is not False
        or evidence["unresolved_rows_accepted"] is not False
        or evidence["routing_metadata_is_integer_only"] is not True
        or evidence["operation_audit"] != _MODEL_OPERATION_AUDIT
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Semantic-step span evidence completeness contract is invalid."
        )
    normalized = []
    for donor_id, record in enumerate(raw_spans):
        if not isinstance(record, Mapping) or set(record) != {
            "donor_id",
            "target_token_ids",
        }:
            raise Lfm2SemanticStepTokenFoldError(
                f"Semantic-step span row {donor_id} is invalid."
            )
        if record["donor_id"] != donor_id:
            raise Lfm2SemanticStepTokenFoldError(
                "Semantic-step spans are not in donor-row order."
            )
        normalized.append(
            _integer_span(
                record["target_token_ids"],
                target_row_count=target_count,
                name=f"semantic-step span row {donor_id}",
            )
        )
    if evidence["span_token_count"] != sum(map(len, normalized)) or evidence[
        "maximum_span_length"
    ] != max(map(len, normalized)):
        raise Lfm2SemanticStepTokenFoldError(
            "Semantic-step span evidence counts are invalid."
        )
    return evidence


def _span_tuples(evidence: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in record["target_token_ids"])
        for record in evidence["spans"]
    )


def _routing_tensors(
    spans: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    offsets = [0]
    target_ids: list[int] = []
    for span in spans:
        target_ids.extend(int(value) for value in span)
        offsets.append(len(target_ids))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(target_ids, dtype=torch.int64),
    )


def _validated_sparse_semantic_code(
    *,
    semantic_code_offsets: Tensor,
    semantic_code_target_ids: Tensor,
    semantic_code_coefficients: Tensor,
    donor_row_count: int,
    target_row_count: int,
    target_to_donor: Sequence[int] | Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Validate one canonical FP32 CSR semantic code independently of text spans."""

    if (
        not isinstance(semantic_code_offsets, Tensor)
        or not isinstance(semantic_code_target_ids, Tensor)
        or not isinstance(semantic_code_coefficients, Tensor)
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Sparse semantic code offsets, target IDs, and coefficients must be tensors."
        )
    if (
        semantic_code_offsets.ndim != 1
        or semantic_code_target_ids.ndim != 1
        or semantic_code_coefficients.ndim != 1
        or semantic_code_offsets.dtype not in {torch.int32, torch.int64}
        or semantic_code_target_ids.dtype not in {torch.int32, torch.int64}
        or semantic_code_coefficients.dtype != torch.float32
        or semantic_code_coefficients.device.type == "meta"
        or semantic_code_coefficients.layout != torch.strided
        or semantic_code_coefficients.requires_grad
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Sparse semantic code must be one-dimensional integer CSR indices "
            "plus detached dense FP32 coefficients."
        )
    offsets = (
        semantic_code_offsets.detach()
        .to(device="cpu", dtype=torch.int64)
        .clone()
        .contiguous()
    )
    target_ids = (
        semantic_code_target_ids.detach()
        .to(device="cpu", dtype=torch.int64)
        .clone()
        .contiguous()
    )
    coefficients = (
        semantic_code_coefficients.detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
        .contiguous()
    )
    if (
        int(offsets.numel()) != donor_row_count + 1
        or int(offsets[0].item()) != 0
        or int(offsets[-1].item()) != int(target_ids.numel())
        or int(target_ids.numel()) != int(coefficients.numel())
        or bool((offsets[1:] <= offsets[:-1]).any())
        or bool((target_ids < 0).any())
        or bool((target_ids >= target_row_count).any())
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Sparse semantic code CSR structure, bounds, or coefficients are invalid."
        )
    if isinstance(target_to_donor, Tensor):
        if target_to_donor.ndim != 1 or target_to_donor.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise Lfm2SemanticStepTokenFoldError(
                "target_to_donor must be one integer vector."
            )
        protected = (
            target_to_donor.detach()
            .to(device="cpu", dtype=torch.int64)
            .clone()
            .contiguous()
        )
    elif isinstance(target_to_donor, Sequence) and not isinstance(
        target_to_donor, (str, bytes)
    ):
        protected_values = tuple(
            _nonnegative_int(value, "target_to_donor donor ID")
            for value in target_to_donor
        )
        protected = torch.tensor(protected_values, dtype=torch.int64)
    else:
        raise Lfm2SemanticStepTokenFoldError(
            "target_to_donor must be one integer sequence."
        )
    if (
        int(protected.numel()) != target_row_count
        or bool((protected < 0).any())
        or bool((protected >= donor_row_count).any())
        or int(torch.unique(protected).numel()) != target_row_count
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "target_to_donor must injectively protect every target anchor."
        )
    for donor_id in range(donor_row_count):
        start = int(offsets[donor_id].item())
        stop = int(offsets[donor_id + 1].item())
        row_ids = target_ids[start:stop]
        if int(row_ids.numel()) > 1 and bool((row_ids[1:] <= row_ids[:-1]).any()):
            raise Lfm2SemanticStepTokenFoldError(
                f"Sparse semantic code row {donor_id} target IDs are not unique "
                "ascending values."
            )
    for target_id, donor_id in enumerate(protected.tolist()):
        start = int(offsets[donor_id].item())
        stop = int(offsets[donor_id + 1].item())
        if (
            stop - start != 1
            or int(target_ids[start].item()) != target_id
            or float(coefficients[start].item()) != 1.0
        ):
            raise Lfm2SemanticStepTokenFoldError(
                f"Protected donor anchor row {donor_id} must remain exact one-hot "
                f"at target row {target_id}."
            )
    return offsets, target_ids, coefficients, protected


def _sparse_semantic_code_matrix(
    offsets: Tensor,
    target_ids: Tensor,
    coefficients: Tensor,
    *,
    donor_row_count: int,
    target_row_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    offsets_device = offsets.to(device=device, dtype=torch.int64)
    lengths = offsets_device[1:] - offsets_device[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(donor_row_count, device=device, dtype=torch.int64),
        lengths,
    )
    indices = torch.stack(
        (
            row_ids,
            target_ids.to(device=device, dtype=torch.int64),
        ),
        dim=0,
    )
    return torch.sparse_coo_tensor(
        indices,
        coefficients.to(device=device, dtype=dtype),
        size=(donor_row_count, target_row_count),
        device=device,
        dtype=dtype,
        check_invariants=True,
    ).coalesce()


def _positional_values(
    lengths: Tensor,
    positions: Tensor,
    *,
    design: str,
    dtype: torch.dtype,
) -> Tensor:
    if design not in POSITIONAL_CHANNEL_DESIGNS:
        raise Lfm2SemanticStepTokenFoldError(
            f"positional_design must be one of {POSITIONAL_CHANNEL_DESIGNS}."
        )
    length_values = lengths.to(dtype=dtype)
    position_values = positions.to(dtype=dtype)
    if design == "uniform":
        return torch.ones_like(position_values) / length_values
    if design == "linear-forward":
        denominator = length_values * (length_values + 1.0) * 0.5
        return (position_values + 1.0) / denominator
    if design == "linear-reverse":
        denominator = length_values * (length_values + 1.0) * 0.5
        return (length_values - position_values) / denominator
    # Shifted powers avoid overflow for long spans.  The denominator is the
    # exact finite geometric sum in the same deterministic arithmetic domain.
    shifted = torch.pow(
        torch.tensor(2.0, device=positions.device, dtype=dtype),
        position_values - (length_values - 1.0),
    )
    denominator = 2.0 - torch.pow(
        torch.tensor(2.0, device=positions.device, dtype=dtype),
        1.0 - length_values,
    )
    return shifted / denominator


def _sparse_design_matrix(
    offsets: Tensor,
    target_ids: Tensor,
    *,
    donor_row_count: int,
    target_row_count: int,
    design: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    offsets_device = offsets.to(device=device)
    lengths = offsets_device[1:] - offsets_device[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(donor_row_count, device=device, dtype=torch.int64),
        lengths,
    )
    flat_positions = torch.arange(
        int(target_ids.numel()), device=device, dtype=torch.int64
    )
    positions = flat_positions - offsets_device[:-1].index_select(0, row_ids)
    edge_lengths = lengths.index_select(0, row_ids)
    values = _positional_values(
        edge_lengths,
        positions,
        design=design,
        dtype=dtype,
    )
    indices = torch.stack(
        (row_ids, target_ids.to(device=device, dtype=torch.int64)), dim=0
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(donor_row_count, target_row_count),
        device=device,
        dtype=dtype,
        check_invariants=True,
    ).coalesce()


def _sparse_design_row_blocks(
    design: Tensor,
    *,
    maximum_rows_per_block: int,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Split A by donor rows so ``A @ X`` is never donor-table sized."""

    coalesced = design.coalesce()
    donor_count, target_count = map(int, coalesced.shape)
    block_rows = min(
        _positive_int(maximum_rows_per_block, "maximum_rows_per_block"),
        donor_count,
    )
    if donor_count > 1 and block_rows == donor_count:
        block_rows = donor_count - 1
    row_ids = coalesced.indices()[0]
    blocks = []
    for start in range(0, donor_count, block_rows):
        stop = min(start + block_rows, donor_count)
        mask = (row_ids >= start) & (row_ids < stop)
        selected_indices = coalesced.indices()[:, mask].clone()
        selected_indices[0].sub_(start)
        block = torch.sparse_coo_tensor(
            selected_indices,
            coalesced.values()[mask],
            size=(stop - start, target_count),
            device=coalesced.device,
            dtype=coalesced.dtype,
            check_invariants=True,
        ).coalesce()
        blocks.append((block, block.transpose(0, 1).coalesce()))
    return tuple(blocks)


def materialize_lfm2_semantic_step_design_matrix(
    span_evidence: Mapping[str, Any],
    *,
    positional_design: str = "uniform",
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    maximum_elements: int = 1_000_000,
) -> Tensor:
    """Materialize a bounded diagnostic ``A_span`` (never donor weights)."""

    evidence = _verified_span_evidence(span_evidence)
    maximum = _positive_int(maximum_elements, "maximum_elements")
    donor_count = int(evidence["donor_row_count"])
    target_count = int(evidence["target_row_count"])
    if donor_count * target_count > maximum:
        raise Lfm2SemanticStepTokenFoldError(
            "Diagnostic dense design matrix exceeds maximum_elements."
        )
    if dtype not in _SUPPORTED_DTYPES.values():
        raise Lfm2SemanticStepTokenFoldError(
            "Diagnostic design dtype must be FP16, BF16, FP32, or FP64."
        )
    offsets, ids = _routing_tensors(_span_tuples(evidence))
    return _sparse_design_matrix(
        offsets,
        ids,
        donor_row_count=donor_count,
        target_row_count=target_count,
        design=positional_design,
        device=torch.device(device),
        dtype=dtype,
    ).to_dense()


def _validated_source_table(value: Tensor) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.layout != torch.strided
        or value.device.type == "meta"
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) <= 0
        or not bool(torch.isfinite(value.detach()).all())
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "donor_table must be one finite dense floating matrix."
        )
    if value.dtype not in _SUPPORTED_DTYPES.values():
        raise Lfm2SemanticStepTokenFoldError(
            "donor_table dtype must be FP16, BF16, FP32, or FP64."
        )
    return value.detach()


def _validated_donor_row_weights(
    value: Tensor | Sequence[float] | None,
    *,
    donor_row_count: int,
) -> tuple[Tensor, str]:
    if value is None:
        return torch.ones(donor_row_count, dtype=torch.float64), "uniform-default"
    if isinstance(value, Tensor):
        if (
            value.ndim != 1
            or int(value.numel()) != donor_row_count
            or not torch.is_floating_point(value)
            or value.is_complex()
            or value.layout != torch.strided
            or value.device.type == "meta"
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "donor_row_weights tensor must be one floating donor-row vector."
            )
        weights = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise Lfm2SemanticStepTokenFoldError(
                "donor_row_weights must be a tensor, numeric sequence, or None."
            )
        try:
            weights = torch.tensor(tuple(value), dtype=torch.float64)
        except (TypeError, ValueError) as exc:
            raise Lfm2SemanticStepTokenFoldError(
                "donor_row_weights contains a nonnumeric value."
            ) from exc
        if weights.ndim != 1 or int(weights.numel()) != donor_row_count:
            raise Lfm2SemanticStepTokenFoldError(
                "donor_row_weights must cover every donor row."
            )
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise Lfm2SemanticStepTokenFoldError(
            "Every donor row weight must be finite and strictly positive."
        )
    return weights, "caller-supplied-positive-weights"


def _resolved_dtype(value: Any, name: str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value not in _SUPPORTED_DTYPES.values():
            raise Lfm2SemanticStepTokenFoldError(f"{name} dtype is unsupported.")
        return value
    if torch_numeric_dtype is not None:
        try:
            return torch_numeric_dtype(value)
        except NumericExecutionPolicyError as exc:
            raise Lfm2SemanticStepTokenFoldError(
                f"{name} dtype is unsupported."
            ) from exc
    if not isinstance(value, str) or value.lower() not in _SUPPORTED_DTYPES:
        raise Lfm2SemanticStepTokenFoldError(f"{name} dtype is unsupported.")
    return _SUPPORTED_DTYPES[value.lower()]


def _resolve_numeric_placements(
    donor_table: Tensor,
    policy: Any,
    *,
    publication_device: str | torch.device | None,
    publication_dtype: str | torch.dtype | None,
) -> tuple[torch.device, torch.dtype, torch.device, torch.dtype, dict[str, Any]]:
    if policy is None:
        solve_device = torch.device("cpu")
        solve_dtype = torch.float64
        publish_device = (
            donor_table.device
            if publication_device is None
            else torch.device(publication_device)
        )
        publish_dtype = (
            donor_table.dtype
            if publication_dtype is None
            else _resolved_dtype(publication_dtype, "publication")
        )
        receipt = {
            "numeric_execution_policy_sha256": None,
            "solve_placement": {
                "device": str(solve_device),
                "dtype": str(solve_dtype).removeprefix("torch."),
            },
            "publication_placement": {
                "device": str(publish_device),
                "dtype": str(publish_dtype).removeprefix("torch."),
            },
            "placement_source": "deterministic-default-cpu-fp64-solve",
        }
        return (
            solve_device,
            solve_dtype,
            publish_device,
            publish_dtype,
            receipt,
        )
    if NumericExecutionPolicy is None or not isinstance(policy, NumericExecutionPolicy):
        raise TypeError("numeric_execution_policy must be NumericExecutionPolicy.")
    try:
        solve = policy.placement(NumericDomain.SOLVE).concrete
        publication = policy.placement(NumericDomain.PUBLICATION)
    except NumericExecutionPolicyError as exc:
        raise Lfm2SemanticStepTokenFoldError(
            "Numeric execution policy has no concrete solve placement."
        ) from exc
    solve_device = solve.torch_device
    solve_dtype = solve.torch_dtype
    if publication.is_concrete:
        concrete_publication = publication.concrete
        publish_device = concrete_publication.torch_device
        publish_dtype = concrete_publication.torch_dtype
    else:
        publish_device = donor_table.device
        publish_dtype = donor_table.dtype
    if publication_device is not None and torch.device(publication_device) != (
        publish_device
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Explicit publication_device conflicts with numeric policy."
        )
    if (
        publication_dtype is not None
        and _resolved_dtype(publication_dtype, "publication") != publish_dtype
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Explicit publication_dtype conflicts with numeric policy."
        )
    receipt = {
        "numeric_execution_policy_sha256": policy.policy_sha256,
        "solve_placement": solve.as_dict(),
        "publication_placement": {
            "device": str(publish_device),
            "dtype": str(publish_dtype).removeprefix("torch."),
        },
        "placement_source": "numeric-execution-policy",
    }
    return solve_device, solve_dtype, publish_device, publish_dtype, receipt


def _pcg_ridge_solve(
    design: Tensor,
    design_row_blocks: Sequence[tuple[Tensor, Tensor]],
    rhs: Tensor,
    *,
    ridge: float,
    maximum_iterations: int,
    tolerance: float,
    iteration_callback: (
        Callable[[int, float, float, bool, float | None], None] | None
    ) = None,
) -> tuple[Tensor, dict[str, Any]]:
    target_count, hidden_columns = (int(rhs.shape[0]), int(rhs.shape[1]))
    coalesced = design.coalesce()
    column_ids = coalesced.indices()[1]
    diagonal = torch.full(
        (target_count,),
        ridge,
        device=rhs.device,
        dtype=rhs.dtype,
    )
    diagonal.index_add_(0, column_ids, coalesced.values().square())

    def operator(value: Tensor) -> Tensor:
        result = ridge * value
        for design_block, design_block_transpose in design_row_blocks:
            result = result + torch.sparse.mm(
                design_block_transpose,
                torch.sparse.mm(design_block, value),
            )
        return result

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual / diagonal.unsqueeze(1)
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum(dim=0)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=0)
    threshold = tolerance * torch.maximum(rhs_norm, torch.ones_like(rhs_norm))
    active = torch.linalg.vector_norm(residual, dim=0) > threshold
    iterations = 0
    if iteration_callback is not None:
        iteration_callback(
            0,
            float(torch.linalg.vector_norm(residual, dim=0).max().item()),
            float(threshold.max().item()),
            not bool(active.any()),
            None,
        )
    for iteration in range(maximum_iterations):
        if not bool(active.any()):
            break
        applied = operator(direction)
        denominator = (direction * applied).sum(dim=0)
        if bool((denominator[active] <= 0).any()) or not bool(
            torch.isfinite(denominator[active]).all()
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Sparse ridge PCG lost its positive-definite denominator."
            )
        alpha = torch.zeros_like(rz)
        alpha[active] = rz[active] / denominator[active]
        solution.add_(direction * alpha.unsqueeze(0))
        residual.sub_(applied * alpha.unsqueeze(0))
        iterations = iteration + 1
        residual_norm = torch.linalg.vector_norm(residual, dim=0)
        next_active = residual_norm > threshold
        final_heartbeat = (
            not bool(next_active.any()) or iteration + 1 == maximum_iterations
        )
        if iteration_callback is not None and (
            iteration == 0 or (iteration + 1) % 8 == 0 or final_heartbeat
        ):
            iteration_callback(
                iteration + 1,
                float(residual_norm.max().item()),
                float(threshold.max().item()),
                final_heartbeat,
                None,
            )
        if not bool(next_active.any()):
            active = next_active
            break
        next_preconditioned = residual / diagonal.unsqueeze(1)
        next_rz = (residual * next_preconditioned).sum(dim=0)
        beta = torch.zeros_like(rz)
        retained = next_active & active & rz.ne(0)
        beta[retained] = next_rz[retained] / rz[retained]
        direction = next_preconditioned + direction * beta.unsqueeze(0)
        direction[:, ~next_active] = 0
        preconditioned = next_preconditioned
        del preconditioned
        rz = next_rz
        active = next_active
    residual_norm = torch.linalg.vector_norm(residual, dim=0)
    converged = residual_norm <= threshold
    if not bool(converged.all()):
        raise Lfm2SemanticStepTokenFoldError(
            "Sparse ridge PCG did not converge within maximum_solver_iterations."
        )
    if not bool(torch.isfinite(solution).all()):
        raise Lfm2SemanticStepTokenFoldError(
            "Sparse ridge solve produced nonfinite target rows."
        )
    return solution, {
        "hidden_column_count": hidden_columns,
        "iterations": iterations,
        "maximum_absolute_normal_residual": float(residual.abs().max().item()),
        "maximum_l2_normal_residual": float(residual_norm.max().item()),
        "all_hidden_columns_converged": True,
    }


def _runtime_weights_for_spans(
    spans: Sequence[Sequence[int]],
    *,
    design: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    lengths = torch.tensor(
        [len(span) for span in spans], device=device, dtype=torch.int64
    )
    local_rows = torch.repeat_interleave(
        torch.arange(len(spans), device=device, dtype=torch.int64), lengths
    )
    target_ids = torch.tensor(
        [token_id for span in spans for token_id in span],
        device=device,
        dtype=torch.int64,
    )
    offsets = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int64),
            lengths.cumsum(dim=0),
        )
    )
    positions = torch.arange(
        int(target_ids.numel()), device=device, dtype=torch.int64
    ) - offsets[:-1].index_select(0, local_rows)
    weights = _positional_values(
        lengths.index_select(0, local_rows),
        positions,
        design=design,
        dtype=dtype,
    )
    return local_rows, target_ids, weights


def _verified_fold_receipt(
    receipt: Mapping[str, Any],
    *,
    target_table: Tensor,
    span_offsets: Tensor,
    span_target_ids: Tensor,
) -> dict[str, Any]:
    value = _verify_sealed(
        receipt,
        field="result_sha256",
        schema=LFM2_SEMANTIC_STEP_TOKEN_FOLD_SCHEMA,
        name="semantic-step token fold",
    )
    donor_count = int(span_offsets.numel()) - 1
    target_count, hidden_size = map(int, target_table.shape)
    checks = {
        "target_table_sha256": value.get("target_table_sha256")
        == _tensor_sha256(target_table),
        "span_offsets_sha256": value.get("span_offsets_sha256")
        == _tensor_sha256(span_offsets),
        "span_target_ids_sha256": value.get("span_target_ids_sha256")
        == _tensor_sha256(span_target_ids),
        "donor_row_count": value.get("donor_row_count") == donor_count,
        "donor_model_row_capacity": isinstance(
            value.get("donor_model_row_capacity"), int
        )
        and value["donor_model_row_capacity"] >= donor_count,
        "undefined_donor_rows_excluded": value.get(
            "undefined_donor_rows_excluded_from_solve_and_runtime_scores"
        )
        is True,
        "target_row_count": value.get("target_row_count") == target_count,
        "hidden_size": value.get("hidden_size") == hidden_size,
        "target_table_shape": value.get("target_table_shape")
        == [target_count, hidden_size],
        "learned_parameter_count": value.get("learned_parameter_count")
        == target_count * hidden_size,
        "learned_parameter_tensor_count": value.get("learned_parameter_tensor_count")
        == 1,
        "routing_parameter_count": value.get("routing_parameter_count") == 0,
        "routing_metadata_is_integer_only": value.get(
            "routing_metadata_is_integer_only"
        )
        is True,
        "donor_row_weights_shape": value.get("donor_row_weights_shape")
        == [donor_count],
        "positive_weighted_donor_row_count": value.get(
            "positive_weighted_donor_row_count"
        )
        == donor_count,
        "all_donor_rows_retained_in_weighted_solve": value.get(
            "all_donor_rows_retained_in_weighted_solve"
        )
        is True,
        "donor_row_weights_are_learned_parameters": value.get(
            "donor_row_weights_are_learned_parameters"
        )
        is False,
        "sqrt_weights_design_rhs": value.get(
            "sqrt_donor_row_weights_applied_to_design_and_rhs"
        )
        is True,
        "sqrt_weights_verification": value.get(
            "sqrt_donor_row_weights_applied_to_verification_residual"
        )
        is True,
        "no_full_donor_table": value.get("full_reconstructed_donor_table_persisted")
        is False,
        "no_full_donor_table_materializations": value.get(
            "full_reconstructed_donor_table_materializations"
        )
        == 0,
        "matrix_free_operator_blocks": isinstance(
            value.get("matrix_free_operator_donor_row_block_count"), int
        )
        and value["matrix_free_operator_donor_row_block_count"] >= 1,
        "strict_partial_donor_row_blocks": value.get(
            "pcg_operator_uses_strict_partial_donor_row_blocks"
        )
        is True,
        "bounded_matrix_free_operator_rows": isinstance(
            value.get("maximum_matrix_free_operator_donor_rows_per_block"),
            int,
        )
        and value["maximum_matrix_free_operator_donor_rows_per_block"] >= 1
        and (
            donor_count <= 1
            or value["maximum_matrix_free_operator_donor_rows_per_block"] < donor_count
        ),
        "operation_audit": value.get("operation_audit") == _MODEL_OPERATION_AUDIT,
        "positional_design": value.get("positional_design")
        in POSITIONAL_CHANNEL_DESIGNS,
        "solver_chunk_accounting": isinstance(
            value.get("solver_computed_chunk_count"), int
        )
        and isinstance(value.get("checkpoint_reused_chunk_count"), int)
        and value["solver_computed_chunk_count"]
        + value["checkpoint_reused_chunk_count"]
        == value.get("solver_hidden_chunk_count"),
    }
    failures = sorted(name for name, valid in checks.items() if not valid)
    if failures:
        raise Lfm2SemanticStepTokenFoldError(
            "Semantic-step token fold receipt or tensor binding is invalid: "
            + ", ".join(failures)
        )
    _digest(value.get("span_evidence_sha256"), "span_evidence_sha256")
    _digest(value.get("donor_row_weights_sha256"), "donor_row_weights_sha256")
    return value


@dataclass(frozen=True, slots=True)
class Lfm2SemanticStepSolvedChunkCheckpoint:
    """One sealed, resumable target-table hidden-column chunk."""

    target_chunk: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_chunk, Tensor)
            or self.target_chunk.ndim != 2
            or not torch.is_floating_point(self.target_chunk)
            or self.target_chunk.is_complex()
            or self.target_chunk.layout != torch.strided
            or self.target_chunk.device.type == "meta"
            or self.target_chunk.requires_grad
            or not bool(torch.isfinite(self.target_chunk).all())
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Solved chunk checkpoint tensor must be detached finite floats."
            )
        value = _verify_sealed(
            self.receipt,
            field="checkpoint_sha256",
            schema=LFM2_SEMANTIC_STEP_CHUNK_CHECKPOINT_SCHEMA,
            name="semantic-step solved chunk checkpoint",
        )
        rows, columns = map(int, self.target_chunk.shape)
        hidden_start = _nonnegative_int(value.get("hidden_start"), "hidden_start")
        hidden_stop = _positive_int(value.get("hidden_stop"), "hidden_stop")
        if (
            hidden_stop <= hidden_start
            or columns != hidden_stop - hidden_start
            or value.get("target_row_count") != rows
            or value.get("target_chunk_shape") != [rows, columns]
            or value.get("target_chunk_dtype") != _dtype_name(self.target_chunk)
            or value.get("target_chunk_sha256") != _tensor_sha256(self.target_chunk)
            or not isinstance(value.get("solver_report"), Mapping)
            or value["solver_report"].get("all_hidden_columns_converged") is not True
            or value.get("operation_audit") != _MODEL_OPERATION_AUDIT
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Solved chunk checkpoint receipt or tensor binding is invalid."
            )
        _digest(value.get("phase_cache_key"), "phase_cache_key")
        _digest(value.get("chunk_input_sha256"), "chunk_input_sha256")
        object.__setattr__(
            self,
            "target_chunk",
            self.target_chunk.detach().to(device="cpu").clone().contiguous(),
        )
        object.__setattr__(self, "receipt", _deep_freeze(value))

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticStepCodecResult:
    donor_id: int
    target_token_ids: tuple[int, ...]
    semantic_embedding: Tensor
    body_output: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            field="result_sha256",
            schema=LFM2_SEMANTIC_STEP_CODEC_RESULT_SCHEMA,
            name="semantic-step codec result",
        )
        if (
            value.get("donor_id") != self.donor_id
            or value.get("target_token_ids") != list(self.target_token_ids)
            or value.get("semantic_embedding_sha256")
            != _tensor_sha256(self.semantic_embedding)
            or value.get("body_step_callback_invocations") != 1
            or value.get("body_semantic_steps_advanced") != 1
            or value.get("target_span_tokens_used_as_body_steps") != 0
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Semantic-step codec result receipt is invalid."
            )
        object.__setattr__(
            self,
            "semantic_embedding",
            self.semantic_embedding.detach().clone(),
        )
        object.__setattr__(self, "receipt", _deep_freeze(value))

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def _verified_runtime_receipt(
    receipt: Mapping[str, Any],
    *,
    span_offsets: Tensor,
    span_target_ids: Tensor,
    semantic_code_offsets: Tensor | None = None,
    semantic_code_target_ids: Tensor | None = None,
    semantic_code_coefficients: Tensor | None = None,
    output_semantic_code_offsets: Tensor | None = None,
    output_semantic_code_target_ids: Tensor | None = None,
    output_semantic_code_coefficients: Tensor | None = None,
) -> dict[str, Any]:
    value = _verify_sealed(
        receipt,
        field="runtime_sha256",
        schema=LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA,
        name="semantic-step routing runtime",
    )
    donor_count = int(span_offsets.numel()) - 1
    code_values = (
        semantic_code_offsets,
        semantic_code_target_ids,
        semantic_code_coefficients,
    )
    semantic_code_supplied = all(item is not None for item in code_values)
    if semantic_code_supplied != any(item is not None for item in code_values):
        raise Lfm2SemanticStepTokenFoldError(
            "Sparse semantic code tensors must be supplied together."
        )
    output_code_values = (
        output_semantic_code_offsets,
        output_semantic_code_target_ids,
        output_semantic_code_coefficients,
    )
    output_code_supplied = all(item is not None for item in output_code_values)
    if output_code_supplied != any(item is not None for item in output_code_values):
        raise Lfm2SemanticStepTokenFoldError(
            "Output sparse semantic code tensors must be supplied together."
        )
    if output_code_supplied and not semantic_code_supplied:
        raise Lfm2SemanticStepTokenFoldError(
            "A separate output sparse semantic code requires an input code."
        )
    if (
        value.get("donor_row_count") != donor_count
        or value.get("span_offsets_sha256") != _tensor_sha256(span_offsets)
        or value.get("span_target_ids_sha256") != _tensor_sha256(span_target_ids)
        or value.get("positional_design") not in POSITIONAL_CHANNEL_DESIGNS
        or value.get("owned_learned_parameter_count") != 0
        or value.get("owns_or_registers_target_table") is not False
        or value.get("live_child_table_borrowed_per_call") is not True
        or value.get("live_child_table_hash_verified_per_call") is not True
        or value.get("parameter_alias_count") != 0
        or value.get("table_origin")
        not in {
            "analytic-sparse-ridge-solved-table",
            "hard-direct-anchor-table",
            "hard-direct-anchor-table-with-sparse-semantic-code",
            "hard-direct-anchor-table-with-separate-input-output-sparse-semantic-codes",
        }
        or value.get("operation_audit") != _MODEL_OPERATION_AUDIT
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Semantic-step routing runtime receipt is invalid."
        )
    _digest(value.get("source_artifact_sha256"), "source_artifact_sha256")
    _digest(
        value.get("expected_target_table_sha256"),
        "expected_target_table_sha256",
    )
    target_count = _positive_int(value.get("target_row_count"), "target_row_count")
    hidden_size = _positive_int(value.get("hidden_size"), "hidden_size")
    if value.get("expected_target_table_shape") != [target_count, hidden_size]:
        raise Lfm2SemanticStepTokenFoldError(
            "Semantic-step runtime target shape binding is invalid."
        )
    if semantic_code_supplied and output_code_supplied:
        assert semantic_code_offsets is not None
        assert semantic_code_target_ids is not None
        assert semantic_code_coefficients is not None
        assert output_semantic_code_offsets is not None
        assert output_semantic_code_target_ids is not None
        assert output_semantic_code_coefficients is not None
        input_count = int(semantic_code_coefficients.numel())
        output_count = int(output_semantic_code_coefficients.numel())
        pair_body = {
            "schema": LFM2_SEPARATE_SPARSE_SEMANTIC_CODE_PAIR_SCHEMA,
            "input_semantic_code_offsets_sha256": _tensor_sha256(
                semantic_code_offsets
            ),
            "input_semantic_code_target_ids_sha256": _tensor_sha256(
                semantic_code_target_ids
            ),
            "input_semantic_code_coefficients_sha256": _tensor_sha256(
                semantic_code_coefficients
            ),
            "output_semantic_code_offsets_sha256": _tensor_sha256(
                output_semantic_code_offsets
            ),
            "output_semantic_code_target_ids_sha256": _tensor_sha256(
                output_semantic_code_target_ids
            ),
            "output_semantic_code_coefficients_sha256": _tensor_sha256(
                output_semantic_code_coefficients
            ),
            "input_continuous_semantic_code_coefficient_count": input_count,
            "output_continuous_semantic_code_coefficient_count": output_count,
            "continuous_semantic_code_coefficient_count": input_count
            + output_count,
            "input_output_coefficient_storage_shared": False,
        }
        if (
            value.get("routing_metadata_is_integer_only") is not False
            or value.get("semantic_routing_mode")
            != "emission-plus-separate-input-output-sparse-semantic-codes"
            or value.get("semantic_code_pair_schema")
            != LFM2_SEPARATE_SPARSE_SEMANTIC_CODE_PAIR_SCHEMA
            or value.get("semantic_code_pair_sha256")
            != canonical_sha256(pair_body)
            or any(
                value.get(name) != expected
                for name, expected in pair_body.items()
                if name != "schema"
            )
            or value.get("semantic_code_coefficient_dtype") != "float32"
            or value.get("owned_fixed_continuous_state_count")
            != input_count + output_count
            or value.get("semantic_code_coefficients_counted_as_child_state")
            is not True
            or value.get("emission_spans_are_semantic_coefficients") is not False
            or value.get("direct_anchor_rows_are_exact_one_hot") is not True
            or value.get("prepared_semantic_code_cache_scope")
            != "route-role-code-hash-runtime-device-table-dtype-table-hash"
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Separate input/output sparse semantic runtime receipt is invalid."
            )
        _digest(value.get("semantic_code_pair_sha256"), "semantic_code_pair_sha256")
    elif semantic_code_supplied:
        assert semantic_code_offsets is not None
        assert semantic_code_target_ids is not None
        assert semantic_code_coefficients is not None
        coefficient_count = int(semantic_code_coefficients.numel())
        if (
            value.get("routing_metadata_is_integer_only") is not False
            or value.get("semantic_routing_mode") != "dual-channel-sparse-semantic-code"
            or value.get("semantic_code_schema") != LFM2_SPARSE_SEMANTIC_CODE_SCHEMA
            or value.get("semantic_code_offsets_sha256")
            != _tensor_sha256(semantic_code_offsets)
            or value.get("semantic_code_target_ids_sha256")
            != _tensor_sha256(semantic_code_target_ids)
            or value.get("semantic_code_coefficients_sha256")
            != _tensor_sha256(semantic_code_coefficients)
            or value.get("semantic_code_coefficient_dtype") != "float32"
            or value.get("continuous_semantic_code_coefficient_count")
            != coefficient_count
            or value.get("owned_fixed_continuous_state_count") != coefficient_count
            or value.get("semantic_code_coefficients_counted_as_child_state")
            is not True
            or value.get("emission_spans_are_semantic_coefficients") is not False
            or value.get("direct_anchor_rows_are_exact_one_hot") is not True
            or value.get("prepared_semantic_code_cache_scope")
            != "runtime-device-table-dtype-table-hash"
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Dual-channel sparse semantic runtime receipt is invalid."
            )
        _digest(value.get("semantic_code_sha256"), "semantic_code_sha256")
    elif (
        value.get("routing_metadata_is_integer_only") is not True
        or value.get("owned_fixed_continuous_state_count", 0) != 0
        or value.get("continuous_semantic_code_coefficient_count", 0) != 0
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Legacy span routing receipt contains unauthenticated continuous state."
        )
    return value


class Lfm2SemanticStepLiveTableSession:
    """Hash-once bounded borrow of one live child table for a generation."""

    __slots__ = (
        "_closing_table_sha256",
        "_full_hash_verification_count",
        "_opening_table_sha256",
        "_opening_version",
        "_output_routing_design",
        "_output_routing_edge_donor_ids",
        "_output_routing_row_offsets",
        "_output_routing_target_ids",
        "_output_routing_weights",
        "_routing_accumulation_dtype",
        "_routing_cache_build_count",
        "_routing_cache_hit",
        "_routing_cache_persisted_after_close",
        "_routing_cache_use_count",
        "_routing_design",
        "_routing_edge_donor_ids",
        "_routing_row_offsets",
        "_routing_target_ids",
        "_routing_tensor_rebuild_count",
        "_routing_weights",
        "_receipt",
        "_runtime",
        "_state",
        "_table",
        "_table_identity",
        "_typed_fallback_reasons",
    )

    def __init__(
        self,
        runtime: Lfm2SemanticStepRoutingRuntime,
        live_child_table: Tensor,
    ) -> None:
        self._runtime = runtime
        self._table: Tensor | None = live_child_table
        self._state = "new"
        self._opening_version: int | None = None
        self._opening_table_sha256: str | None = None
        self._closing_table_sha256: str | None = None
        self._full_hash_verification_count = 0
        self._table_identity: tuple[Any, ...] | None = None
        self._receipt: Mapping[str, Any] | None = None
        self._routing_design: Tensor | None = None
        self._routing_edge_donor_ids: Tensor | None = None
        self._routing_row_offsets: Tensor | None = None
        self._routing_target_ids: Tensor | None = None
        self._routing_weights: Tensor | None = None
        self._output_routing_design: Tensor | None = None
        self._output_routing_edge_donor_ids: Tensor | None = None
        self._output_routing_row_offsets: Tensor | None = None
        self._output_routing_target_ids: Tensor | None = None
        self._output_routing_weights: Tensor | None = None
        self._routing_accumulation_dtype: torch.dtype | None = None
        self._routing_cache_build_count = 0
        self._routing_cache_hit = False
        self._routing_cache_persisted_after_close = False
        self._routing_cache_use_count = 0
        self._routing_tensor_rebuild_count = 0
        self._typed_fallback_reasons: list[str] = []

    @property
    def full_hash_verification_count(self) -> int:
        return self._full_hash_verification_count

    def __enter__(self) -> Lfm2SemanticStepLiveTableSession:
        if self._state != "new" or self._table is None:
            raise Lfm2SemanticStepTokenFoldError(
                "Live-table session cannot be entered more than once."
            )
        self._runtime._full_authenticate_live_table(self._table)
        self._opening_table_sha256 = str(
            self._runtime._receipt["expected_target_table_sha256"]
        )
        self._full_hash_verification_count = 1
        raw = self._table
        self._opening_version = int(raw._version)
        self._table_identity = (
            id(raw),
            tuple(int(size) for size in raw.shape),
            tuple(int(stride) for stride in raw.stride()),
            str(raw.dtype),
            str(raw.device),
            int(raw.data_ptr()),
        )
        accumulation_dtype = _routing_accumulation_dtype(raw.dtype)
        try:
            if self._runtime.has_sparse_semantic_code:
                prepared, cache_hit = self._runtime._prepared_semantic_routing(
                    route_role=(
                        "input"
                        if self._runtime.has_separate_output_semantic_code
                        else "shared"
                    ),
                    device=raw.device,
                    table_dtype=raw.dtype,
                    accumulation_dtype=accumulation_dtype,
                )
                (
                    self._routing_design,
                    self._routing_edge_donor_ids,
                    self._routing_row_offsets,
                    self._routing_target_ids,
                    self._routing_weights,
                ) = prepared
                self._routing_cache_hit = cache_hit
                self._routing_cache_build_count = int(not cache_hit)
                self._routing_cache_persisted_after_close = True
                if self._runtime.has_separate_output_semantic_code:
                    output_prepared, output_cache_hit = (
                        self._runtime._prepared_semantic_routing(
                            route_role="output",
                            device=raw.device,
                            table_dtype=raw.dtype,
                            accumulation_dtype=accumulation_dtype,
                        )
                    )
                    (
                        self._output_routing_design,
                        self._output_routing_edge_donor_ids,
                        self._output_routing_row_offsets,
                        self._output_routing_target_ids,
                        self._output_routing_weights,
                    ) = output_prepared
                    self._routing_cache_hit = cache_hit and output_cache_hit
                    self._routing_cache_build_count += int(not output_cache_hit)
            else:
                routing_design = _sparse_design_matrix(
                    self._runtime._span_offsets,
                    self._runtime._span_target_ids,
                    donor_row_count=self._runtime.donor_row_count,
                    target_row_count=self._runtime.target_row_count,
                    design=self._runtime.positional_design,
                    device=raw.device,
                    dtype=accumulation_dtype,
                ).coalesce()
                self._routing_design = routing_design
                self._routing_edge_donor_ids = routing_design.indices()[0]
                routing_row_counts = torch.bincount(
                    self._routing_edge_donor_ids,
                    minlength=self._runtime.donor_row_count,
                )
                self._routing_row_offsets = torch.cat(
                    (
                        routing_row_counts.new_zeros(1),
                        routing_row_counts.cumsum(dim=0),
                    )
                )
                self._routing_target_ids = routing_design.indices()[1]
                self._routing_weights = routing_design.values()
                self._routing_cache_build_count = 1
        except (RuntimeError, TypeError) as exc:
            raise Lfm2SemanticStepTokenFoldError(
                "Sparse semantic routing is unsupported at the live table placement."
            ) from exc
        self._routing_accumulation_dtype = accumulation_dtype
        self._state = "open"
        return self

    def _borrow(
        self,
        runtime: Lfm2SemanticStepRoutingRuntime,
    ) -> Tensor:
        if self._state != "open" or self._table is None:
            raise Lfm2SemanticStepTokenFoldError("Live-table session is not open.")
        if runtime is not self._runtime:
            raise Lfm2SemanticStepTokenFoldError(
                "Live-table session belongs to a different routing runtime."
            )
        raw = self._table
        observed_identity = (
            id(raw),
            tuple(int(size) for size in raw.shape),
            tuple(int(stride) for stride in raw.stride()),
            str(raw.dtype),
            str(raw.device),
            int(raw.data_ptr()),
        )
        if (
            observed_identity != self._table_identity
            or int(raw._version) != self._opening_version
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Borrowed live child table identity, placement, or version changed."
            )
        return raw.detach()

    def _borrow_routing(
        self,
        runtime: Lfm2SemanticStepRoutingRuntime,
        *,
        route_role: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, torch.dtype]:
        self._borrow(runtime)
        if route_role not in {"input", "output"}:
            raise Lfm2SemanticStepTokenFoldError(
                "Borrowed sparse semantic route must be input or output."
            )
        if route_role == "output" and runtime.has_separate_output_semantic_code:
            design = self._output_routing_design
            edge_donor_ids = self._output_routing_edge_donor_ids
            row_offsets = self._output_routing_row_offsets
            target_ids = self._output_routing_target_ids
            weights = self._output_routing_weights
        else:
            design = self._routing_design
            edge_donor_ids = self._routing_edge_donor_ids
            row_offsets = self._routing_row_offsets
            target_ids = self._routing_target_ids
            weights = self._routing_weights
        if (
            design is None
            or edge_donor_ids is None
            or row_offsets is None
            or target_ids is None
            or weights is None
            or self._routing_accumulation_dtype is None
            or (
                not self._routing_cache_persisted_after_close
                and self._routing_cache_build_count != 1
            )
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Session-scoped sparse routing cache is unavailable."
            )
        self._routing_cache_use_count += 1
        return (
            design,
            edge_donor_ids,
            row_offsets,
            target_ids,
            weights,
            self._routing_accumulation_dtype,
        )

    def _record_typed_fallback(self, reason: str) -> None:
        if self._state != "open":
            raise Lfm2SemanticStepTokenFoldError(
                "Typed fallback can only be recorded in an open session."
            )
        self._typed_fallback_reasons.append(str(reason))

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: Any,
    ) -> None:
        if self._state != "open" or self._table is None:
            raise Lfm2SemanticStepTokenFoldError("Live-table session is not open.")
        try:
            table = self._borrow(self._runtime)
            if (
                self._routing_design is None
                or self._routing_accumulation_dtype is None
                or (
                    self._runtime.has_separate_output_semantic_code
                    and self._output_routing_design is None
                )
            ):
                raise Lfm2SemanticStepTokenFoldError(
                    "Session routing cache disappeared before close."
                )
            output_routing_design = (
                self._output_routing_design
                if self._runtime.has_separate_output_semantic_code
                else self._routing_design
            )
            assert output_routing_design is not None
            self._closing_table_sha256 = _tensor_sha256(table)
            self._full_hash_verification_count += 1
            if self._closing_table_sha256 != self._opening_table_sha256:
                raise Lfm2SemanticStepTokenFoldError(
                    "Live child table hash changed during the bounded session."
                )
            body = {
                "schema": LFM2_SEMANTIC_STEP_LIVE_TABLE_SESSION_SCHEMA,
                "runtime_sha256": self._runtime.as_dict()["runtime_sha256"],
                "opening_target_table_sha256": self._opening_table_sha256,
                "closing_target_table_sha256": self._closing_table_sha256,
                "opening_tensor_version": self._opening_version,
                "closing_tensor_version": int(self._table._version),
                "full_hash_verification_count": self._full_hash_verification_count,
                "interior_calls_use_identity_version_shape_dtype_device_only": True,
                "live_table_cloned_or_registered": False,
                "owned_learned_parameter_count": 0,
                "routing_cache_build_count": self._routing_cache_build_count,
                "routing_cache_hit": self._routing_cache_hit,
                "routing_cache_use_count": self._routing_cache_use_count,
                "routing_tensor_rebuild_count_inside_session": (
                    self._routing_tensor_rebuild_count
                ),
                "routing_cache_released_on_close": (
                    not self._routing_cache_persisted_after_close
                ),
                "prepared_semantic_code_cache_persisted_after_close": (
                    self._routing_cache_persisted_after_close
                ),
                "runtime_prepared_semantic_code_cache_build_count": (
                    self._runtime.prepared_semantic_code_cache_build_count
                ),
                "runtime_prepared_semantic_code_cache_hit_count": (
                    self._runtime.prepared_semantic_code_cache_hit_count
                ),
                "runtime_prepared_semantic_code_cache_use_count": (
                    self._runtime.prepared_semantic_code_cache_use_count
                ),
                "semantic_routing_mode": self._runtime.semantic_routing_mode,
                "physical_sparse_route_count": (
                    2 if self._runtime.has_separate_output_semantic_code else 1
                ),
                "routing_cache_layout": str(self._routing_design.layout).removeprefix(
                    "torch."
                ),
                "routing_cache_device": str(self._routing_design.device),
                "routing_sparse_nonzero_count": int(self._routing_design._nnz())
                + (
                    int(output_routing_design._nnz())
                    if self._runtime.has_separate_output_semantic_code
                    else 0
                ),
                "routing_sparse_nonzero_count_semantics": (
                    "unique-physical-route-nnz-total"
                ),
                "input_routing_sparse_nonzero_count": int(
                    self._routing_design._nnz()
                ),
                "output_routing_sparse_nonzero_count": int(
                    output_routing_design._nnz()
                ),
                "routing_index_storage_dtype": "int64",
                "routing_weight_storage_dtype": str(
                    self._routing_accumulation_dtype
                ).removeprefix("torch."),
                "routing_accumulation_dtype": str(
                    self._routing_accumulation_dtype
                ).removeprefix("torch."),
                "routing_dtype_policy": (
                    "widen-float16-bfloat16-to-float32;preserve-float32-float64"
                ),
                "live_table_dtype": _dtype_name(table),
                "semantic_embedding_publication_dtype": _dtype_name(table),
                "donor_logits_publication_dtype": str(
                    self._routing_accumulation_dtype
                ).removeprefix("torch."),
                "low_precision_routing_accumulation_widened_to_fp32": (
                    table.dtype in {torch.float16, torch.bfloat16}
                    and self._routing_accumulation_dtype == torch.float32
                ),
                "typed_fallback_count": len(self._typed_fallback_reasons),
                "typed_fallback_reasons": list(self._typed_fallback_reasons),
                "operation_audit": dict(_MODEL_OPERATION_AUDIT),
            }
            self._receipt = _deep_freeze(_sealed(body, "session_sha256"))
        finally:
            self._state = "closed"
            self._routing_design = None
            self._routing_edge_donor_ids = None
            self._routing_row_offsets = None
            self._routing_target_ids = None
            self._routing_weights = None
            self._output_routing_design = None
            self._output_routing_edge_donor_ids = None
            self._output_routing_row_offsets = None
            self._output_routing_target_ids = None
            self._output_routing_weights = None
            self._routing_accumulation_dtype = None
            self._table = None

    def as_dict(self) -> dict[str, Any]:
        if self._state != "closed" or self._receipt is None:
            raise Lfm2SemanticStepTokenFoldError(
                "Live-table session receipt is available only after close."
            )
        return _plain_json(self._receipt)


class Lfm2SemanticStepRoutingRuntime:
    """Parameterless runtime borrowing the child's existing tied table.

    This intentionally is not an ``nn.Module``.  It owns only authenticated
    CPU routing metadata plus, for the dual-channel contract, fixed FP32
    semantic coefficients.  It never clones, registers, aliases, or retains a
    live child parameter.  Every operation receives and authenticates the
    current child table by reference.
    """

    __slots__ = (
        "_prepared_semantic_code_cache",
        "_prepared_semantic_code_cache_build_count",
        "_prepared_semantic_code_cache_hit_count",
        "_prepared_semantic_code_cache_lock",
        "_prepared_semantic_code_cache_use_count",
        "_receipt",
        "_semantic_code_coefficients",
        "_semantic_code_offsets",
        "_semantic_code_target_ids",
        "_output_semantic_code_coefficients",
        "_output_semantic_code_offsets",
        "_output_semantic_code_target_ids",
        "_span_offsets",
        "_span_target_ids",
    )

    def __init__(
        self,
        span_offsets: Tensor,
        span_target_ids: Tensor,
        receipt: Mapping[str, Any],
        *,
        semantic_code_offsets: Tensor | None = None,
        semantic_code_target_ids: Tensor | None = None,
        semantic_code_coefficients: Tensor | None = None,
        output_semantic_code_offsets: Tensor | None = None,
        output_semantic_code_target_ids: Tensor | None = None,
        output_semantic_code_coefficients: Tensor | None = None,
    ) -> None:
        offsets = span_offsets.detach().to(device="cpu", dtype=torch.int64).clone()
        ids = span_target_ids.detach().to(device="cpu", dtype=torch.int64).clone()
        if (
            offsets.ndim != 1
            or offsets.numel() < 2
            or ids.ndim != 1
            or int(offsets[0].item()) != 0
            or int(offsets[-1].item()) != int(ids.numel())
            or bool((offsets[1:] <= offsets[:-1]).any())
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Parameterless routing metadata is invalid."
            )
        verified = _verified_runtime_receipt(
            receipt,
            span_offsets=offsets,
            span_target_ids=ids,
            semantic_code_offsets=semantic_code_offsets,
            semantic_code_target_ids=semantic_code_target_ids,
            semantic_code_coefficients=semantic_code_coefficients,
            output_semantic_code_offsets=output_semantic_code_offsets,
            output_semantic_code_target_ids=output_semantic_code_target_ids,
            output_semantic_code_coefficients=output_semantic_code_coefficients,
        )
        self._span_offsets = offsets
        self._span_target_ids = ids
        self._semantic_code_offsets = (
            None
            if semantic_code_offsets is None
            else semantic_code_offsets.detach()
            .to(device="cpu", dtype=torch.int64)
            .clone()
            .contiguous()
        )
        self._semantic_code_target_ids = (
            None
            if semantic_code_target_ids is None
            else semantic_code_target_ids.detach()
            .to(device="cpu", dtype=torch.int64)
            .clone()
            .contiguous()
        )
        self._semantic_code_coefficients = (
            None
            if semantic_code_coefficients is None
            else semantic_code_coefficients.detach()
            .to(device="cpu", dtype=torch.float32)
            .clone()
            .contiguous()
        )
        self._output_semantic_code_offsets = (
            None
            if output_semantic_code_offsets is None
            else output_semantic_code_offsets.detach()
            .to(device="cpu", dtype=torch.int64)
            .clone()
            .contiguous()
        )
        self._output_semantic_code_target_ids = (
            None
            if output_semantic_code_target_ids is None
            else output_semantic_code_target_ids.detach()
            .to(device="cpu", dtype=torch.int64)
            .clone()
            .contiguous()
        )
        self._output_semantic_code_coefficients = (
            None
            if output_semantic_code_coefficients is None
            else output_semantic_code_coefficients.detach()
            .to(device="cpu", dtype=torch.float32)
            .clone()
            .contiguous()
        )
        self._prepared_semantic_code_cache: dict[
            tuple[str, str, str, str, str, str],
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        ] = {}
        self._prepared_semantic_code_cache_build_count = 0
        self._prepared_semantic_code_cache_hit_count = 0
        self._prepared_semantic_code_cache_use_count = 0
        self._prepared_semantic_code_cache_lock = threading.Lock()
        self._receipt = _deep_freeze(verified)

    @property
    def donor_row_count(self) -> int:
        return int(self._receipt["donor_row_count"])

    @property
    def target_row_count(self) -> int:
        return int(self._receipt["target_row_count"])

    @property
    def hidden_size(self) -> int:
        return int(self._receipt["hidden_size"])

    @property
    def positional_design(self) -> str:
        return str(self._receipt["positional_design"])

    @property
    def semantic_routing_mode(self) -> str:
        return str(
            self._receipt.get(
                "semantic_routing_mode",
                "legacy-emission-span-positional-code",
            )
        )

    @property
    def has_sparse_semantic_code(self) -> bool:
        return self._semantic_code_coefficients is not None

    @property
    def has_separate_output_semantic_code(self) -> bool:
        return self._output_semantic_code_coefficients is not None

    @property
    def continuous_semantic_code_coefficient_count(self) -> int:
        return int(
            self._receipt.get(
                "continuous_semantic_code_coefficient_count",
                0,
            )
        )

    @property
    def prepared_semantic_code_cache_build_count(self) -> int:
        return self._prepared_semantic_code_cache_build_count

    @property
    def prepared_semantic_code_cache_hit_count(self) -> int:
        return self._prepared_semantic_code_cache_hit_count

    @property
    def prepared_semantic_code_cache_use_count(self) -> int:
        return self._prepared_semantic_code_cache_use_count

    @property
    def source_artifact_sha256(self) -> str:
        return str(self._receipt["source_artifact_sha256"])

    @property
    def table_origin(self) -> str:
        return str(self._receipt["table_origin"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self._receipt)

    def authenticated_live_table_session(
        self,
        live_child_table: Tensor,
    ) -> Lfm2SemanticStepLiveTableSession:
        return Lfm2SemanticStepLiveTableSession(self, live_child_table)

    def _prepared_semantic_routing(
        self,
        *,
        route_role: str,
        device: torch.device,
        table_dtype: torch.dtype,
        accumulation_dtype: torch.dtype,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor], bool]:
        if route_role not in {"shared", "input", "output"}:
            raise Lfm2SemanticStepTokenFoldError(
                "Sparse semantic route role must be shared, input, or output."
            )
        if route_role == "output":
            offsets = self._output_semantic_code_offsets
            target_ids = self._output_semantic_code_target_ids
            coefficients = self._output_semantic_code_coefficients
            code_sha256 = self._receipt.get(
                "output_semantic_code_coefficients_sha256"
            )
        else:
            offsets = self._semantic_code_offsets
            target_ids = self._semantic_code_target_ids
            coefficients = self._semantic_code_coefficients
            code_sha256 = self._receipt.get(
                (
                    "input_semantic_code_coefficients_sha256"
                    if self.has_separate_output_semantic_code
                    else "semantic_code_coefficients_sha256"
                )
            )
        if offsets is None or target_ids is None or coefficients is None:
            raise Lfm2SemanticStepTokenFoldError(
                f"Runtime has no {route_role} sparse semantic code to prepare."
            )
        key = (
            route_role,
            str(code_sha256),
            str(device),
            str(table_dtype),
            str(accumulation_dtype),
            str(self._receipt["expected_target_table_sha256"]),
        )
        with self._prepared_semantic_code_cache_lock:
            self._prepared_semantic_code_cache_use_count += 1
            cached = self._prepared_semantic_code_cache.get(key)
            if cached is not None:
                self._prepared_semantic_code_cache_hit_count += 1
                return cached, True
            design = _sparse_semantic_code_matrix(
                offsets,
                target_ids,
                coefficients,
                donor_row_count=self.donor_row_count,
                target_row_count=self.target_row_count,
                device=device,
                dtype=accumulation_dtype,
            )
            edge_donor_ids = design.indices()[0]
            row_counts = torch.bincount(
                edge_donor_ids,
                minlength=self.donor_row_count,
            )
            row_offsets = torch.cat((row_counts.new_zeros(1), row_counts.cumsum(dim=0)))
            prepared = (
                design,
                edge_donor_ids,
                row_offsets,
                design.indices()[1],
                design.values(),
            )
            self._prepared_semantic_code_cache[key] = prepared
            self._prepared_semantic_code_cache_build_count += 1
            return prepared, False

    def _full_authenticate_live_table(self, live_child_table: Tensor) -> Tensor:
        if (
            not isinstance(live_child_table, Tensor)
            or live_child_table.ndim != 2
            or tuple(int(size) for size in live_child_table.shape)
            != (self.target_row_count, self.hidden_size)
            or not torch.is_floating_point(live_child_table)
            or live_child_table.is_complex()
            or live_child_table.layout != torch.strided
            or live_child_table.device.type == "meta"
            or not bool(torch.isfinite(live_child_table.detach()).all())
            or _tensor_sha256(live_child_table)
            != self._receipt["expected_target_table_sha256"]
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Live child table differs from the compiled semantic-step table."
            )
        # A detached view is returned only for this call.  It is never retained.
        return live_child_table.detach()

    def _authenticated_live_table(
        self,
        live_child_table: Tensor | Lfm2SemanticStepLiveTableSession,
    ) -> Tensor:
        if isinstance(live_child_table, Lfm2SemanticStepLiveTableSession):
            return live_child_table._borrow(self)
        return self._full_authenticate_live_table(live_child_table)

    def span_for_donor(self, donor_id: int) -> tuple[int, ...]:
        index = _nonnegative_int(donor_id, "donor_id")
        if index >= self.donor_row_count:
            raise Lfm2SemanticStepTokenFoldError(
                "donor_id leaves semantic-step routing."
            )
        start = int(self._span_offsets[index].item())
        stop = int(self._span_offsets[index + 1].item())
        return tuple(int(value) for value in self._span_target_ids[start:stop].tolist())

    def reconstruct_selected_donor_embeddings(
        self,
        live_child_table: Tensor | Lfm2SemanticStepLiveTableSession,
        donor_ids: Sequence[int] | Tensor,
        *,
        maximum_rows_per_chunk: int = 1024,
    ) -> Tensor:
        if isinstance(live_child_table, Tensor):
            with self.authenticated_live_table_session(live_child_table) as session:
                return self.reconstruct_selected_donor_embeddings(
                    session,
                    donor_ids,
                    maximum_rows_per_chunk=maximum_rows_per_chunk,
                )
        table = self._authenticated_live_table(live_child_table)
        (
            _routing_design,
            _edge_donor_ids,
            row_offsets,
            cached_target_ids,
            cached_weights,
            accumulation_dtype,
        ) = live_child_table._borrow_routing(self, route_role="input")
        chunk_size = _positive_int(maximum_rows_per_chunk, "maximum_rows_per_chunk")
        if isinstance(donor_ids, Tensor):
            if donor_ids.ndim != 1 or donor_ids.dtype not in {
                torch.int32,
                torch.int64,
            }:
                raise Lfm2SemanticStepTokenFoldError(
                    "donor_ids tensor must be one integer vector."
                )
            values = tuple(int(value) for value in donor_ids.detach().cpu().tolist())
        elif isinstance(donor_ids, Sequence) and not isinstance(
            donor_ids, (str, bytes)
        ):
            values = tuple(
                _nonnegative_int(value, "selected donor_id") for value in donor_ids
            )
        else:
            raise Lfm2SemanticStepTokenFoldError(
                "donor_ids must be one integer sequence."
            )
        if any(value >= self.donor_row_count for value in values):
            raise Lfm2SemanticStepTokenFoldError(
                "Selected donor ID leaves semantic-step routing."
            )
        if not values:
            return table.new_empty((0, self.hidden_size))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), chunk_size):
                selected = values[start : start + chunk_size]
                selected_ids = torch.tensor(
                    selected,
                    device=row_offsets.device,
                    dtype=torch.int64,
                )
                edge_starts = row_offsets.index_select(0, selected_ids)
                edge_stops = row_offsets.index_select(0, selected_ids + 1)
                edge_lengths = edge_stops - edge_starts
                packed_starts = torch.cat(
                    (
                        edge_lengths.new_zeros(1),
                        edge_lengths.cumsum(dim=0)[:-1],
                    )
                )
                local_rows = torch.repeat_interleave(
                    torch.arange(
                        len(selected),
                        device=row_offsets.device,
                        dtype=torch.int64,
                    ),
                    edge_lengths,
                )
                edge_positions = torch.arange(
                    int(edge_lengths.sum().item()),
                    device=row_offsets.device,
                    dtype=torch.int64,
                ) + torch.repeat_interleave(
                    edge_starts - packed_starts,
                    edge_lengths,
                )
                target_ids = cached_target_ids.index_select(0, edge_positions)
                weights = cached_weights.index_select(0, edge_positions)
                result = torch.zeros(
                    (len(selected), self.hidden_size),
                    device=table.device,
                    dtype=accumulation_dtype,
                )
                result.index_add_(
                    0,
                    local_rows,
                    table.index_select(0, target_ids).to(dtype=accumulation_dtype)
                    * weights.unsqueeze(1),
                )
                outputs.append(result.to(dtype=table.dtype))
        return torch.cat(outputs, dim=0)

    def score_target_tokens(
        self,
        live_child_table: Tensor | Lfm2SemanticStepLiveTableSession,
        hidden_states: Tensor,
    ) -> Tensor:
        table = self._authenticated_live_table(live_child_table)
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.ndim < 1
            or int(hidden_states.shape[-1]) != self.hidden_size
            or not torch.is_floating_point(hidden_states)
            or hidden_states.is_complex()
            or not bool(torch.isfinite(hidden_states.detach()).all())
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "hidden_states must end in the semantic hidden size."
            )
        projection_dtype = table.dtype
        if table.dtype == torch.bfloat16 and table.device.type == "cuda":
            try:
                native_bfloat16 = torch.cuda.get_device_capability(table.device)[0] >= 8
            except (RuntimeError, AssertionError):
                native_bfloat16 = False
            if not native_bfloat16:
                projection_dtype = torch.float32
                if isinstance(live_child_table, Lfm2SemanticStepLiveTableSession):
                    live_child_table._record_typed_fallback(
                        "cuda-device-lacks-native-bfloat16-dense-projection;"
                        "used-float32"
                    )
        with torch.no_grad():
            return hidden_states.detach().to(
                device=table.device, dtype=projection_dtype
            ) @ table.to(dtype=projection_dtype).transpose(0, 1)

    def compute_all_donor_logits_from_target_scores(
        self,
        live_child_table: Tensor | Lfm2SemanticStepLiveTableSession,
        target_scores: Tensor,
        *,
        maximum_donor_rows_per_chunk: int = 4096,
    ) -> Tensor:
        if isinstance(live_child_table, Tensor):
            with self.authenticated_live_table_session(live_child_table) as session:
                return self.compute_all_donor_logits_from_target_scores(
                    session,
                    target_scores,
                    maximum_donor_rows_per_chunk=(maximum_donor_rows_per_chunk),
                )
        # Keep the public bound validated for API compatibility.  The cached
        # sparse operator evaluates all donor rows in one SpMM and therefore
        # does not rebuild row chunks at each generated semantic step.
        _positive_int(
            maximum_donor_rows_per_chunk,
            "maximum_donor_rows_per_chunk",
        )
        self._authenticated_live_table(live_child_table)
        (
            routing_design,
            _edge_donor_ids,
            _row_offsets,
            _cached_target_ids,
            _cached_weights,
            accumulation_dtype,
        ) = live_child_table._borrow_routing(self, route_role="output")
        if (
            not isinstance(target_scores, Tensor)
            or target_scores.ndim < 1
            or int(target_scores.shape[-1]) != self.target_row_count
            or not torch.is_floating_point(target_scores)
            or target_scores.is_complex()
            or not bool(torch.isfinite(target_scores.detach()).all())
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "target_scores must end in target_row_count finite floats."
            )
        scores = target_scores.detach().to(
            device=routing_design.device,
            dtype=accumulation_dtype,
        )
        leading = tuple(int(size) for size in scores.shape[:-1])
        flat = scores.reshape(-1, self.target_row_count)
        with torch.no_grad():
            donor_logits = torch.sparse.mm(
                routing_design,
                flat.transpose(0, 1),
            ).transpose(0, 1)
        return donor_logits.reshape((*leading, self.donor_row_count))

    def semantic_step(
        self,
        live_child_table: Tensor | Lfm2SemanticStepLiveTableSession,
        body_step: Callable[[Tensor], Any],
        *,
        donor_id: int | None = None,
        target_scores: Tensor | None = None,
        target_span_emitter: Callable[[tuple[int, ...]], None] | None = None,
        maximum_donor_rows_per_chunk: int = 4096,
    ) -> Lfm2SemanticStepCodecResult:
        if isinstance(live_child_table, Tensor):
            with self.authenticated_live_table_session(live_child_table) as session:
                return self.semantic_step(
                    session,
                    body_step,
                    donor_id=donor_id,
                    target_scores=target_scores,
                    target_span_emitter=target_span_emitter,
                    maximum_donor_rows_per_chunk=(maximum_donor_rows_per_chunk),
                )
        self._authenticated_live_table(live_child_table)
        if not callable(body_step):
            raise TypeError("body_step must be callable.")
        if target_span_emitter is not None and not callable(target_span_emitter):
            raise TypeError("target_span_emitter must be callable or None.")
        if (donor_id is None) == (target_scores is None):
            raise Lfm2SemanticStepTokenFoldError(
                "Supply exactly one of donor_id or target_scores."
            )
        if donor_id is None:
            assert target_scores is not None
            if target_scores.ndim != 1:
                raise Lfm2SemanticStepTokenFoldError(
                    "Codec target_scores selection requires one score row."
                )
            donor_logits = self.compute_all_donor_logits_from_target_scores(
                live_child_table,
                target_scores,
                maximum_donor_rows_per_chunk=maximum_donor_rows_per_chunk,
            )
            selected_id = int(torch.argmax(donor_logits).item())
            selection = "deterministic-argmax-from-target-scores"
        else:
            selected_id = _nonnegative_int(donor_id, "donor_id")
            if selected_id >= self.donor_row_count:
                raise Lfm2SemanticStepTokenFoldError(
                    "donor_id leaves semantic-step routing."
                )
            selection = "caller-supplied-donor-id"
        span = self.span_for_donor(selected_id)
        embedding = self.reconstruct_selected_donor_embeddings(
            live_child_table, [selected_id]
        )[0]
        emitter_calls = 0
        if target_span_emitter is not None:
            target_span_emitter(span)
            emitter_calls = 1
        with torch.no_grad():
            body_output = body_step(embedding.detach())
        body = {
            "schema": LFM2_SEMANTIC_STEP_CODEC_RESULT_SCHEMA,
            "fold_sha256": self.source_artifact_sha256,
            "selection": selection,
            "donor_id": selected_id,
            "target_token_ids": list(span),
            "target_span_length": len(span),
            "semantic_embedding_sha256": _tensor_sha256(embedding),
            "target_span_emitter_invocations": emitter_calls,
            "body_step_callback_invocations": 1,
            "body_semantic_steps_advanced": 1,
            "target_span_tokens_used_as_body_steps": 0,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "network_access": False,
        }
        return Lfm2SemanticStepCodecResult(
            donor_id=selected_id,
            target_token_ids=span,
            semantic_embedding=embedding,
            body_output=body_output,
            receipt=_sealed(body, "result_sha256"),
        )


class Lfm2SemanticStepTokenFold(nn.Module):
    """One dense target table plus authenticated integer span routing."""

    def __init__(
        self,
        target_table: Tensor,
        span_offsets: Tensor,
        span_target_ids: Tensor,
        receipt: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if (
            not isinstance(target_table, Tensor)
            or target_table.ndim != 2
            or not torch.is_floating_point(target_table)
            or target_table.is_complex()
            or target_table.layout != torch.strided
            or target_table.device.type == "meta"
            or target_table.requires_grad
            or not bool(torch.isfinite(target_table).all())
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "target_table must be one detached finite dense floating matrix."
            )
        offsets = span_offsets.detach().to(device="cpu", dtype=torch.int64).contiguous()
        ids = span_target_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        if (
            offsets.ndim != 1
            or offsets.numel() < 2
            or ids.ndim != 1
            or int(offsets[0].item()) != 0
            or int(offsets[-1].item()) != int(ids.numel())
            or bool((offsets[1:] <= offsets[:-1]).any())
            or (ids.numel() and int(ids.min().item()) < 0)
            or (ids.numel() and int(ids.max().item()) >= int(target_table.shape[0]))
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Span routing buffers must encode nonempty in-range integer spans."
            )
        table = target_table.detach().clone().contiguous()
        verified = _verified_fold_receipt(
            receipt,
            target_table=table,
            span_offsets=offsets,
            span_target_ids=ids,
        )
        self.target_table = nn.Parameter(table, requires_grad=False)
        self.register_buffer("span_offsets", offsets, persistent=True)
        self.register_buffer("span_target_ids", ids, persistent=True)
        self._receipt = _deep_freeze(verified)
        if tuple(self.named_parameters(remove_duplicate=False)) != (
            ("target_table", self.target_table),
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "Semantic-step fold must own exactly one unaliased parameter tensor."
            )

    @property
    def donor_row_count(self) -> int:
        return int(self.span_offsets.numel()) - 1

    @property
    def target_row_count(self) -> int:
        return int(self.target_table.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.target_table.shape[1])

    @property
    def positional_design(self) -> str:
        return str(self._receipt["positional_design"])

    @property
    def fold_sha256(self) -> str:
        return str(self._receipt["result_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self._receipt)

    def to_parameterless_runtime(self) -> Lfm2SemanticStepRoutingRuntime:
        """Bind integer routing to the table expected in the live child model."""

        body = {
            "schema": LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA,
            "source_artifact_sha256": self.fold_sha256,
            "table_origin": "analytic-sparse-ridge-solved-table",
            "donor_row_count": self.donor_row_count,
            "target_row_count": self.target_row_count,
            "hidden_size": self.hidden_size,
            "positional_design": self.positional_design,
            "expected_target_table_shape": [
                self.target_row_count,
                self.hidden_size,
            ],
            "expected_target_table_sha256": _tensor_sha256(self.target_table),
            "span_offsets_sha256": _tensor_sha256(self.span_offsets),
            "span_target_ids_sha256": _tensor_sha256(self.span_target_ids),
            "owned_learned_parameter_count": 0,
            "owns_or_registers_target_table": False,
            "live_child_table_borrowed_per_call": True,
            "live_child_table_hash_verified_per_call": True,
            "routing_metadata_is_integer_only": True,
            "parameter_alias_count": 0,
            "operation_audit": dict(_MODEL_OPERATION_AUDIT),
        }
        return Lfm2SemanticStepRoutingRuntime(
            self.span_offsets,
            self.span_target_ids,
            _sealed(body, "runtime_sha256"),
        )

    def span_for_donor(self, donor_id: int) -> tuple[int, ...]:
        index = _nonnegative_int(donor_id, "donor_id")
        if index >= self.donor_row_count:
            raise Lfm2SemanticStepTokenFoldError(
                "donor_id leaves semantic-step routing."
            )
        start = int(self.span_offsets[index].item())
        stop = int(self.span_offsets[index + 1].item())
        return tuple(int(value) for value in self.span_target_ids[start:stop].tolist())

    def reconstruct_selected_donor_embeddings(
        self,
        donor_ids: Sequence[int] | Tensor,
        *,
        maximum_rows_per_chunk: int = 1024,
    ) -> Tensor:
        """Reconstruct only requested donor rows, in bounded chunks."""

        chunk_size = _positive_int(maximum_rows_per_chunk, "maximum_rows_per_chunk")
        if isinstance(donor_ids, Tensor):
            if donor_ids.ndim != 1 or donor_ids.dtype not in {
                torch.int32,
                torch.int64,
            }:
                raise Lfm2SemanticStepTokenFoldError(
                    "donor_ids tensor must be one integer vector."
                )
            values = tuple(int(value) for value in donor_ids.detach().cpu().tolist())
        else:
            if isinstance(donor_ids, (str, bytes)) or not isinstance(
                donor_ids, Sequence
            ):
                raise Lfm2SemanticStepTokenFoldError(
                    "donor_ids must be one integer sequence."
                )
            values = tuple(
                _nonnegative_int(value, "selected donor_id") for value in donor_ids
            )
        if any(value >= self.donor_row_count for value in values):
            raise Lfm2SemanticStepTokenFoldError(
                "Selected donor ID leaves semantic-step routing."
            )
        if not values:
            return self.target_table.new_empty((0, self.hidden_size))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), chunk_size):
                selected = values[start : start + chunk_size]
                spans = [self.span_for_donor(value) for value in selected]
                local_rows, target_ids, weights = _runtime_weights_for_spans(
                    spans,
                    design=self.positional_design,
                    device=self.target_table.device,
                    dtype=self.target_table.dtype,
                )
                result = self.target_table.new_zeros((len(selected), self.hidden_size))
                result.index_add_(
                    0,
                    local_rows,
                    self.target_table.index_select(0, target_ids)
                    * weights.unsqueeze(1),
                )
                outputs.append(result)
        return torch.cat(outputs, dim=0)

    def score_target_tokens(self, hidden_states: Tensor) -> Tensor:
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.ndim < 1
            or int(hidden_states.shape[-1]) != self.hidden_size
            or not torch.is_floating_point(hidden_states)
            or hidden_states.is_complex()
            or not bool(torch.isfinite(hidden_states.detach()).all())
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "hidden_states must end in the semantic hidden size."
            )
        with torch.no_grad():
            return hidden_states.detach().to(
                device=self.target_table.device,
                dtype=self.target_table.dtype,
            ) @ self.target_table.transpose(0, 1)

    def compute_all_donor_logits_from_target_scores(
        self,
        target_scores: Tensor,
        *,
        maximum_donor_rows_per_chunk: int = 4096,
    ) -> Tensor:
        """Apply ``A_span`` to target scores without donor output weights."""

        chunk_size = _positive_int(
            maximum_donor_rows_per_chunk,
            "maximum_donor_rows_per_chunk",
        )
        if (
            not isinstance(target_scores, Tensor)
            or target_scores.ndim < 1
            or int(target_scores.shape[-1]) != self.target_row_count
            or not torch.is_floating_point(target_scores)
            or target_scores.is_complex()
            or not bool(torch.isfinite(target_scores.detach()).all())
        ):
            raise Lfm2SemanticStepTokenFoldError(
                "target_scores must end in target_row_count finite floats."
            )
        scores = target_scores.detach()
        leading = tuple(int(size) for size in scores.shape[:-1])
        flat = scores.reshape(-1, self.target_row_count)
        outputs = []
        with torch.no_grad():
            for start in range(0, self.donor_row_count, chunk_size):
                stop = min(start + chunk_size, self.donor_row_count)
                spans = [self.span_for_donor(value) for value in range(start, stop)]
                local_rows, target_ids, weights = _runtime_weights_for_spans(
                    spans,
                    design=self.positional_design,
                    device=scores.device,
                    dtype=scores.dtype,
                )
                selected = flat.index_select(1, target_ids) * weights.unsqueeze(0)
                result = scores.new_zeros((int(flat.shape[0]), stop - start))
                result.index_add_(1, local_rows, selected)
                outputs.append(result)
        return torch.cat(outputs, dim=1).reshape((*leading, self.donor_row_count))

    def semantic_step(
        self,
        body_step: Callable[[Tensor], Any],
        *,
        donor_id: int | None = None,
        target_scores: Tensor | None = None,
        target_span_emitter: Callable[[tuple[int, ...]], None] | None = None,
        maximum_donor_rows_per_chunk: int = 4096,
    ) -> Lfm2SemanticStepCodecResult:
        """Emit one target spelling and advance the semantic body exactly once."""

        if not callable(body_step):
            raise TypeError("body_step must be callable.")
        if target_span_emitter is not None and not callable(target_span_emitter):
            raise TypeError("target_span_emitter must be callable or None.")
        if (donor_id is None) == (target_scores is None):
            raise Lfm2SemanticStepTokenFoldError(
                "Supply exactly one of donor_id or target_scores."
            )
        if donor_id is None:
            assert target_scores is not None
            if target_scores.ndim != 1:
                raise Lfm2SemanticStepTokenFoldError(
                    "Codec target_scores selection requires one target-score row."
                )
            donor_logits = self.compute_all_donor_logits_from_target_scores(
                target_scores,
                maximum_donor_rows_per_chunk=maximum_donor_rows_per_chunk,
            )
            selected_id = int(torch.argmax(donor_logits).item())
            selection = "deterministic-argmax-from-target-scores"
        else:
            selected_id = _nonnegative_int(donor_id, "donor_id")
            if selected_id >= self.donor_row_count:
                raise Lfm2SemanticStepTokenFoldError(
                    "donor_id leaves semantic-step routing."
                )
            selection = "caller-supplied-donor-id"
        span = self.span_for_donor(selected_id)
        embedding = self.reconstruct_selected_donor_embeddings([selected_id])[0]
        emitter_calls = 0
        if target_span_emitter is not None:
            target_span_emitter(span)
            emitter_calls = 1
        with torch.no_grad():
            body_output = body_step(embedding.detach())
        body = {
            "schema": LFM2_SEMANTIC_STEP_CODEC_RESULT_SCHEMA,
            "fold_sha256": self.fold_sha256,
            "selection": selection,
            "donor_id": selected_id,
            "target_token_ids": list(span),
            "target_span_length": len(span),
            "semantic_embedding_sha256": _tensor_sha256(embedding),
            "target_span_emitter_invocations": emitter_calls,
            "body_step_callback_invocations": 1,
            "body_semantic_steps_advanced": 1,
            "target_span_tokens_used_as_body_steps": 0,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "network_access": False,
        }
        return Lfm2SemanticStepCodecResult(
            donor_id=selected_id,
            target_token_ids=span,
            semantic_embedding=embedding,
            body_output=body_output,
            receipt=_sealed(body, "result_sha256"),
        )


def bind_lfm2_semantic_step_hard_anchor_runtime(
    span_evidence: Mapping[str, Any],
    live_child_table: Tensor,
    *,
    positional_design: str = "uniform",
) -> Lfm2SemanticStepRoutingRuntime:
    """Bind routing directly to an already-installed complete anchor table.

    No least-squares solve, source donor table, model execution, or learned
    parameter is created.  This is the cheapest isolation test for whether
    multi-token spelling was incorrectly advancing the semantic body clock.
    """

    evidence = _verified_span_evidence(span_evidence)
    table = _validated_source_table(live_child_table)
    target_count = int(evidence["target_row_count"])
    if int(table.shape[0]) != target_count:
        raise Lfm2SemanticStepTokenFoldError(
            "Live hard-anchor table rows differ from span target vocabulary."
        )
    if positional_design not in POSITIONAL_CHANNEL_DESIGNS:
        raise Lfm2SemanticStepTokenFoldError(
            f"positional_design must be one of {POSITIONAL_CHANNEL_DESIGNS}."
        )
    spans = _span_tuples(evidence)
    offsets, ids = _routing_tensors(spans)
    hidden_size = int(table.shape[1])
    body = {
        "schema": LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA,
        "source_artifact_sha256": evidence["evidence_sha256"],
        "table_origin": "hard-direct-anchor-table",
        "donor_row_count": int(evidence["donor_row_count"]),
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "positional_design": positional_design,
        "expected_target_table_shape": [target_count, hidden_size],
        "expected_target_table_sha256": _tensor_sha256(table),
        "span_offsets_sha256": _tensor_sha256(offsets),
        "span_target_ids_sha256": _tensor_sha256(ids),
        "owned_learned_parameter_count": 0,
        "owns_or_registers_target_table": False,
        "live_child_table_borrowed_per_call": True,
        "live_child_table_hash_verified_per_call": True,
        "routing_metadata_is_integer_only": True,
        "parameter_alias_count": 0,
        "least_squares_or_ridge_solve_calls": 0,
        "hard_direct_anchor_binding": True,
        "operation_audit": dict(_MODEL_OPERATION_AUDIT),
    }
    return Lfm2SemanticStepRoutingRuntime(
        offsets,
        ids,
        _sealed(body, "runtime_sha256"),
    )


def bind_lfm2_sparse_semantic_code_runtime(
    span_evidence: Mapping[str, Any],
    live_child_table: Tensor,
    *,
    semantic_code_offsets: Tensor,
    semantic_code_target_ids: Tensor,
    semantic_code_coefficients: Tensor,
    target_to_donor: Sequence[int] | Tensor,
) -> Lfm2SemanticStepRoutingRuntime:
    """Bind orthographic emission spans and an independent sparse semantic code.

    ``span_evidence`` remains the exact donor-to-target spelling channel exposed
    by :meth:`span_for_donor`.  The FP32 CSR code is the only channel used to
    reconstruct donor embeddings and donor logits.  Its continuous coefficients
    are authenticated and explicitly counted as child state.
    """

    evidence = _verified_span_evidence(span_evidence)
    table = _validated_source_table(live_child_table)
    donor_count = int(evidence["donor_row_count"])
    target_count = int(evidence["target_row_count"])
    if int(table.shape[0]) != target_count:
        raise Lfm2SemanticStepTokenFoldError(
            "Live semantic-code table rows differ from span target vocabulary."
        )
    spans = _span_tuples(evidence)
    span_offsets, span_target_ids = _routing_tensors(spans)
    (
        code_offsets,
        code_target_ids,
        code_coefficients,
        protected_target_to_donor,
    ) = _validated_sparse_semantic_code(
        semantic_code_offsets=semantic_code_offsets,
        semantic_code_target_ids=semantic_code_target_ids,
        semantic_code_coefficients=semantic_code_coefficients,
        donor_row_count=donor_count,
        target_row_count=target_count,
        target_to_donor=target_to_donor,
    )
    code_body = {
        "schema": LFM2_SPARSE_SEMANTIC_CODE_SCHEMA,
        "emission_span_evidence_sha256": evidence["evidence_sha256"],
        "expected_target_table_sha256": _tensor_sha256(table),
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "semantic_code_offsets_sha256": _tensor_sha256(code_offsets),
        "semantic_code_target_ids_sha256": _tensor_sha256(code_target_ids),
        "semantic_code_coefficients_sha256": _tensor_sha256(code_coefficients),
        "target_to_donor_sha256": _tensor_sha256(protected_target_to_donor),
        "semantic_code_coefficient_dtype": "float32",
        "continuous_semantic_code_coefficient_count": int(code_coefficients.numel()),
        "direct_anchor_count": int(protected_target_to_donor.numel()),
        "direct_anchor_rows_are_exact_one_hot": True,
        "all_semantic_code_rows_nonempty": True,
        "semantic_code_rows_have_unique_target_ids": True,
    }
    semantic_code_sha256 = canonical_sha256(code_body)
    coefficient_count = int(code_coefficients.numel())
    hidden_size = int(table.shape[1])
    body = {
        "schema": LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA,
        "source_artifact_sha256": semantic_code_sha256,
        "semantic_code_sha256": semantic_code_sha256,
        "semantic_code_schema": LFM2_SPARSE_SEMANTIC_CODE_SCHEMA,
        "emission_span_evidence_sha256": evidence["evidence_sha256"],
        "table_origin": "hard-direct-anchor-table-with-sparse-semantic-code",
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        # Retained for the backward-compatible runtime surface.  Positional
        # spelling weights are not consulted by the dual-channel path.
        "positional_design": "uniform",
        "positional_design_applies_to_semantic_code": False,
        "semantic_routing_mode": "dual-channel-sparse-semantic-code",
        "expected_target_table_shape": [target_count, hidden_size],
        "expected_target_table_sha256": _tensor_sha256(table),
        "span_offsets_sha256": _tensor_sha256(span_offsets),
        "span_target_ids_sha256": _tensor_sha256(span_target_ids),
        "semantic_code_offsets_sha256": _tensor_sha256(code_offsets),
        "semantic_code_target_ids_sha256": _tensor_sha256(code_target_ids),
        "semantic_code_coefficients_sha256": _tensor_sha256(code_coefficients),
        "target_to_donor_sha256": _tensor_sha256(protected_target_to_donor),
        "semantic_code_coefficient_dtype": "float32",
        "continuous_semantic_code_coefficient_count": coefficient_count,
        "owned_fixed_continuous_state_count": coefficient_count,
        "semantic_code_continuous_state_bytes": coefficient_count * 4,
        "semantic_code_integer_metadata_bytes": int(
            (
                code_offsets.numel()
                + code_target_ids.numel()
                + protected_target_to_donor.numel()
            )
            * torch.tensor([], dtype=torch.int64).element_size()
        ),
        "semantic_code_coefficients_counted_as_child_state": True,
        "emission_spans_are_semantic_coefficients": False,
        "direct_anchor_count": int(protected_target_to_donor.numel()),
        "direct_anchor_rows_are_exact_one_hot": True,
        "owned_learned_parameter_count": 0,
        "owns_or_registers_target_table": False,
        "live_child_table_borrowed_per_call": True,
        "live_child_table_hash_verified_per_call": True,
        "routing_metadata_is_integer_only": False,
        "routing_metadata_contains_fixed_float_coefficients": True,
        "prepared_semantic_code_cache_scope": ("runtime-device-table-dtype-table-hash"),
        "parameter_alias_count": 0,
        "least_squares_or_ridge_solve_calls": 0,
        "hard_direct_anchor_binding": True,
        "operation_audit": dict(_MODEL_OPERATION_AUDIT),
    }
    return Lfm2SemanticStepRoutingRuntime(
        span_offsets,
        span_target_ids,
        _sealed(body, "runtime_sha256"),
        semantic_code_offsets=code_offsets,
        semantic_code_target_ids=code_target_ids,
        semantic_code_coefficients=code_coefficients,
    )


def bind_lfm2_separate_sparse_semantic_code_runtime(
    span_evidence: Mapping[str, Any],
    live_child_table: Tensor,
    *,
    input_semantic_code_offsets: Tensor,
    input_semantic_code_target_ids: Tensor,
    input_semantic_code_coefficients: Tensor,
    output_semantic_code_offsets: Tensor,
    output_semantic_code_target_ids: Tensor,
    output_semantic_code_coefficients: Tensor,
    target_to_donor: Sequence[int] | Tensor,
) -> Lfm2SemanticStepRoutingRuntime:
    """Bind separate counted sparse operators for body input and readout.

    The exact spelling channel remains independent.  ``C_in`` reconstructs
    embeddings supplied to the body while ``C_out`` maps target-table scores
    back to donor logits.  Both detached FP32 coefficient vectors are fixed
    folded child state and every scalar in both vectors is counted.
    """

    evidence = _verified_span_evidence(span_evidence)
    table = _validated_source_table(live_child_table)
    donor_count = int(evidence["donor_row_count"])
    target_count = int(evidence["target_row_count"])
    if int(table.shape[0]) != target_count:
        raise Lfm2SemanticStepTokenFoldError(
            "Live separate-code table rows differ from span target vocabulary."
        )
    spans = _span_tuples(evidence)
    span_offsets, span_target_ids = _routing_tensors(spans)
    (
        input_offsets,
        input_target_ids,
        input_coefficients,
        input_protected,
    ) = _validated_sparse_semantic_code(
        semantic_code_offsets=input_semantic_code_offsets,
        semantic_code_target_ids=input_semantic_code_target_ids,
        semantic_code_coefficients=input_semantic_code_coefficients,
        donor_row_count=donor_count,
        target_row_count=target_count,
        target_to_donor=target_to_donor,
    )
    (
        output_offsets,
        output_target_ids,
        output_coefficients,
        output_protected,
    ) = _validated_sparse_semantic_code(
        semantic_code_offsets=output_semantic_code_offsets,
        semantic_code_target_ids=output_semantic_code_target_ids,
        semantic_code_coefficients=output_semantic_code_coefficients,
        donor_row_count=donor_count,
        target_row_count=target_count,
        target_to_donor=target_to_donor,
    )
    if not torch.equal(input_protected, output_protected):
        raise Lfm2SemanticStepTokenFoldError(
            "Separate input/output codes authenticate different protected anchors."
        )
    input_count = int(input_coefficients.numel())
    output_count = int(output_coefficients.numel())
    total_count = input_count + output_count
    pair_body = {
        "schema": LFM2_SEPARATE_SPARSE_SEMANTIC_CODE_PAIR_SCHEMA,
        "input_semantic_code_offsets_sha256": _tensor_sha256(input_offsets),
        "input_semantic_code_target_ids_sha256": _tensor_sha256(input_target_ids),
        "input_semantic_code_coefficients_sha256": _tensor_sha256(
            input_coefficients
        ),
        "output_semantic_code_offsets_sha256": _tensor_sha256(output_offsets),
        "output_semantic_code_target_ids_sha256": _tensor_sha256(
            output_target_ids
        ),
        "output_semantic_code_coefficients_sha256": _tensor_sha256(
            output_coefficients
        ),
        "input_continuous_semantic_code_coefficient_count": input_count,
        "output_continuous_semantic_code_coefficient_count": output_count,
        "continuous_semantic_code_coefficient_count": total_count,
        "input_output_coefficient_storage_shared": False,
    }
    pair_sha256 = canonical_sha256(pair_body)
    table_sha256 = _tensor_sha256(table)
    protected_sha256 = _tensor_sha256(input_protected)
    source_artifact_sha256 = canonical_sha256(
        {
            "semantic_code_pair_sha256": pair_sha256,
            "emission_span_evidence_sha256": evidence["evidence_sha256"],
            "expected_target_table_sha256": table_sha256,
            "target_to_donor_sha256": protected_sha256,
        }
    )
    hidden_size = int(table.shape[1])
    body = {
        "schema": LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA,
        "source_artifact_sha256": source_artifact_sha256,
        "semantic_code_pair_schema": LFM2_SEPARATE_SPARSE_SEMANTIC_CODE_PAIR_SCHEMA,
        "semantic_code_pair_sha256": pair_sha256,
        **{name: value for name, value in pair_body.items() if name != "schema"},
        "input_semantic_code_schema": LFM2_SPARSE_SEMANTIC_CODE_SCHEMA,
        "output_semantic_code_schema": LFM2_SPARSE_SEMANTIC_CODE_SCHEMA,
        "emission_span_evidence_sha256": evidence["evidence_sha256"],
        "table_origin": (
            "hard-direct-anchor-table-with-separate-input-output-sparse-semantic-codes"
        ),
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "positional_design": "uniform",
        "positional_design_applies_to_semantic_code": False,
        "semantic_routing_mode": (
            "emission-plus-separate-input-output-sparse-semantic-codes"
        ),
        "expected_target_table_shape": [target_count, hidden_size],
        "expected_target_table_sha256": table_sha256,
        "span_offsets_sha256": _tensor_sha256(span_offsets),
        "span_target_ids_sha256": _tensor_sha256(span_target_ids),
        "target_to_donor_sha256": protected_sha256,
        "semantic_code_coefficient_dtype": "float32",
        "owned_fixed_continuous_state_count": total_count,
        "semantic_code_continuous_state_bytes": total_count * 4,
        "semantic_code_integer_metadata_bytes": int(
            (
                input_offsets.numel()
                + input_target_ids.numel()
                + output_offsets.numel()
                + output_target_ids.numel()
                + input_protected.numel()
            )
            * torch.tensor([], dtype=torch.int64).element_size()
        ),
        "semantic_code_coefficients_counted_as_child_state": True,
        "emission_spans_are_semantic_coefficients": False,
        "direct_anchor_count": int(input_protected.numel()),
        "direct_anchor_rows_are_exact_one_hot": True,
        "owned_learned_parameter_count": 0,
        "owns_or_registers_target_table": False,
        "live_child_table_borrowed_per_call": True,
        "live_child_table_hash_verified_per_call": True,
        "routing_metadata_is_integer_only": False,
        "routing_metadata_contains_fixed_float_coefficients": True,
        "prepared_semantic_code_cache_scope": (
            "route-role-code-hash-runtime-device-table-dtype-table-hash"
        ),
        "parameter_alias_count": 0,
        "least_squares_or_ridge_solve_calls": 0,
        "hard_direct_anchor_binding": True,
        "operation_audit": dict(_MODEL_OPERATION_AUDIT),
    }
    return Lfm2SemanticStepRoutingRuntime(
        span_offsets,
        span_target_ids,
        _sealed(body, "runtime_sha256"),
        semantic_code_offsets=input_offsets,
        semantic_code_target_ids=input_target_ids,
        semantic_code_coefficients=input_coefficients,
        output_semantic_code_offsets=output_offsets,
        output_semantic_code_target_ids=output_target_ids,
        output_semantic_code_coefficients=output_coefficients,
    )


def compile_lfm2_semantic_step_token_fold(
    donor_table: Tensor,
    span_evidence: Mapping[str, Any],
    *,
    target_vocabulary_size: int | None = None,
    source_table_is_tied: bool = True,
    donor_row_weights: Tensor | Sequence[float] | None = None,
    positional_design: str = "uniform",
    ridge: float = 1e-5,
    maximum_solver_iterations: int = 256,
    solver_tolerance: float = 1e-10,
    hidden_columns_per_chunk: int = 128,
    verification_donor_rows_per_chunk: int = 4096,
    numeric_execution_policy: Any | None = None,
    publication_device: str | torch.device | None = None,
    publication_dtype: str | torch.dtype | None = None,
    phase_cache_key: str | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    solved_chunk_loader: (
        Callable[
            [Mapping[str, Any]],
            Lfm2SemanticStepSolvedChunkCheckpoint | None,
        ]
        | None
    ) = None,
    solved_chunk_publisher: (
        Callable[[Lfm2SemanticStepSolvedChunkCheckpoint], None] | None
    ) = None,
) -> Lfm2SemanticStepTokenFold:
    """Fit exactly one target-vocabulary-by-hidden table with sparse PCG."""

    source_capacity_table = _validated_source_table(donor_table)
    evidence = _verified_span_evidence(span_evidence)
    if source_table_is_tied is not True:
        raise Lfm2SemanticStepTokenFoldError(
            "Semantic-step folding requires one caller-authenticated tied "
            "input/output donor table."
        )
    donor_capacity, hidden_size = map(int, source_capacity_table.shape)
    donor_count = int(evidence["donor_row_count"])
    if donor_capacity < donor_count:
        raise Lfm2SemanticStepTokenFoldError(
            "Donor table capacity is smaller than defined span evidence."
        )
    source = source_capacity_table[:donor_count]
    target_count = int(evidence["target_row_count"])
    row_weights, weights_source = _validated_donor_row_weights(
        donor_row_weights,
        donor_row_count=donor_count,
    )
    weights_sha256 = _tensor_sha256(row_weights)
    if (
        target_vocabulary_size is not None
        and _positive_int(target_vocabulary_size, "target_vocabulary_size")
        != target_count
    ):
        raise Lfm2SemanticStepTokenFoldError(
            "Requested target vocabulary differs from authenticated spans."
        )
    if positional_design not in POSITIONAL_CHANNEL_DESIGNS:
        raise Lfm2SemanticStepTokenFoldError(
            f"positional_design must be one of {POSITIONAL_CHANNEL_DESIGNS}."
        )
    ridge_value = _positive_float(ridge, "ridge")
    iterations = _positive_int(maximum_solver_iterations, "maximum_solver_iterations")
    tolerance = _positive_float(solver_tolerance, "solver_tolerance")
    requested_hidden_chunk = _positive_int(
        hidden_columns_per_chunk, "hidden_columns_per_chunk"
    )
    verification_chunk = _positive_int(
        verification_donor_rows_per_chunk,
        "verification_donor_rows_per_chunk",
    )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    if solved_chunk_loader is not None and not callable(solved_chunk_loader):
        raise TypeError("solved_chunk_loader must be callable or None.")
    if solved_chunk_publisher is not None and not callable(solved_chunk_publisher):
        raise TypeError("solved_chunk_publisher must be callable or None.")
    (
        solve_device,
        solve_dtype,
        publish_device,
        publish_dtype,
        placement_receipt,
    ) = _resolve_numeric_placements(
        source,
        numeric_execution_policy,
        publication_device=publication_device,
        publication_dtype=publication_dtype,
    )
    effective_tolerance = max(
        tolerance,
        float(torch.finfo(solve_dtype).eps) * 32.0,
    )
    # A reconstructed operator block is always narrower than the complete
    # donor table for nondegenerate hidden sizes.
    effective_hidden_chunk = min(requested_hidden_chunk, hidden_size)
    if hidden_size > 1 and effective_hidden_chunk == hidden_size:
        effective_hidden_chunk = hidden_size - 1

    spans = _span_tuples(evidence)
    span_offsets, span_target_ids = _routing_tensors(spans)
    source_capacity_sha256 = _tensor_sha256(source_capacity_table)
    source_sha256 = _tensor_sha256(source)
    phase_body = {
        "span_evidence_sha256": evidence["evidence_sha256"],
        "source_donor_table_sha256": source_sha256,
        "source_donor_capacity_table_sha256": source_capacity_sha256,
        "source_shape": [donor_count, hidden_size],
        "source_capacity_shape": [donor_capacity, hidden_size],
        "target_shape": [target_count, hidden_size],
        "donor_row_weights_sha256": weights_sha256,
        "positional_design": positional_design,
        "ridge": ridge_value,
        "maximum_solver_iterations": iterations,
        "effective_solver_tolerance": effective_tolerance,
        "effective_hidden_columns_per_chunk": effective_hidden_chunk,
        "solve_placement": placement_receipt["solve_placement"],
    }
    effective_phase_cache_key = (
        canonical_sha256(phase_body)
        if phase_cache_key is None
        else _digest(phase_cache_key, "phase_cache_key")
    )
    progress_count = 0
    progress_previous: str | None = None
    progress_callback_invocations = 0

    def emit(stage: str, **details: Any) -> None:
        nonlocal progress_count, progress_previous, progress_callback_invocations
        body = {
            "schema": LFM2_SEMANTIC_STEP_PROGRESS_SCHEMA,
            "phase": "semantic-step-token-fold",
            "phase_cache_key": effective_phase_cache_key,
            "ordinal": progress_count,
            "previous_progress_sha256": progress_previous,
            "stage": stage,
            **details,
        }
        seal = canonical_sha256(body)
        record = MappingProxyType({**body, "progress_sha256": seal})
        progress_count += 1
        progress_previous = seal
        if progress_callback is not None:
            progress_callback(record)
            progress_callback_invocations += 1

    emit(
        "start",
        completed_hidden_columns=0,
        total_hidden_columns=hidden_size,
        source_donor_table_sha256=source_sha256,
        span_evidence_sha256=evidence["evidence_sha256"],
        donor_row_weights_sha256=weights_sha256,
    )
    try:
        semantic_design = _sparse_design_matrix(
            span_offsets,
            span_target_ids,
            donor_row_count=donor_count,
            target_row_count=target_count,
            design=positional_design,
            device=solve_device,
            dtype=solve_dtype,
        )
        sqrt_weights = row_weights.to(
            device=solve_device,
            dtype=solve_dtype,
        ).sqrt()
        semantic_design = semantic_design.coalesce()
        semantic_indices = semantic_design.indices()
        design = torch.sparse_coo_tensor(
            semantic_indices,
            semantic_design.values()
            * sqrt_weights.index_select(0, semantic_indices[0]),
            size=semantic_design.shape,
            device=solve_device,
            dtype=solve_dtype,
            check_invariants=True,
        ).coalesce()
        design_transpose = design.transpose(0, 1).coalesce()
        design_row_blocks = _sparse_design_row_blocks(
            design,
            maximum_rows_per_block=4_096,
        )
    except (RuntimeError, TypeError) as exc:
        raise Lfm2SemanticStepTokenFoldError(
            "Sparse positional operator is unsupported at the solve placement."
        ) from exc
    emit(
        "sparse-operator-ready",
        completed_hidden_columns=0,
        total_hidden_columns=hidden_size,
        sparse_nonzero_count=int(design._nnz()),
        matrix_free_operator_donor_row_block_count=len(design_row_blocks),
        maximum_matrix_free_operator_donor_rows_per_block=max(
            int(block.shape[0]) for block, _ in design_row_blocks
        ),
        donor_row_weights_sha256=weights_sha256,
    )

    solved = torch.empty(
        (target_count, hidden_size),
        device=solve_device,
        dtype=solve_dtype,
    )
    chunk_reports: list[dict[str, Any]] = []
    residual_sum_squares = 0.0
    unweighted_residual_sum_squares = 0.0
    source_sum_squares = 0.0
    maximum_absolute_reconstruction_residual = 0.0
    maximum_absolute_weighted_reconstruction_residual = 0.0
    solver_computed_chunk_count = 0
    checkpoint_reused_chunk_count = 0
    checkpoint_loader_invocation_count = 0
    checkpoint_publisher_invocation_count = 0
    with torch.no_grad():
        for hidden_start in range(0, hidden_size, effective_hidden_chunk):
            hidden_stop = min(hidden_start + effective_hidden_chunk, hidden_size)
            source_columns = source[:, hidden_start:hidden_stop].to(
                device=solve_device,
                dtype=solve_dtype,
            )
            chunk_input_body = {
                "phase_cache_key": effective_phase_cache_key,
                "hidden_start": hidden_start,
                "hidden_stop": hidden_stop,
                "target_row_count": target_count,
                "solve_dtype": str(solve_dtype).removeprefix("torch."),
                "source_donor_table_sha256": source_sha256,
                "span_evidence_sha256": evidence["evidence_sha256"],
                "donor_row_weights_sha256": weights_sha256,
                "positional_design": positional_design,
                "ridge": ridge_value,
                "maximum_solver_iterations": iterations,
                "effective_solver_tolerance": effective_tolerance,
            }
            chunk_input_sha256 = canonical_sha256(chunk_input_body)
            request_body = {
                "schema": LFM2_SEMANTIC_STEP_CHUNK_REQUEST_SCHEMA,
                **chunk_input_body,
                "chunk_input_sha256": chunk_input_sha256,
                "target_chunk_shape": [
                    target_count,
                    hidden_stop - hidden_start,
                ],
            }
            request = _sealed(request_body, "request_sha256")
            checkpoint = None
            if solved_chunk_loader is not None:
                checkpoint_loader_invocation_count += 1
                checkpoint = solved_chunk_loader(MappingProxyType(dict(request)))
                if checkpoint is not None and not isinstance(
                    checkpoint,
                    Lfm2SemanticStepSolvedChunkCheckpoint,
                ):
                    raise Lfm2SemanticStepTokenFoldError(
                        "solved_chunk_loader returned an invalid checkpoint type."
                    )
            if checkpoint is not None:
                checkpoint_receipt = checkpoint.as_dict()
                if (
                    checkpoint_receipt.get("request_sha256")
                    != request["request_sha256"]
                    or checkpoint_receipt.get("phase_cache_key")
                    != effective_phase_cache_key
                    or checkpoint_receipt.get("chunk_input_sha256")
                    != chunk_input_sha256
                    or checkpoint_receipt.get("hidden_start") != hidden_start
                    or checkpoint_receipt.get("hidden_stop") != hidden_stop
                    or checkpoint_receipt.get("target_row_count") != target_count
                    or checkpoint.target_chunk.dtype != solve_dtype
                ):
                    raise Lfm2SemanticStepTokenFoldError(
                        "Loaded solved chunk does not match the exact phase request."
                    )
                solution = checkpoint.target_chunk.to(
                    device=solve_device,
                    dtype=solve_dtype,
                )
                solve_report = _plain_json(checkpoint_receipt["solver_report"])
                checkpoint_reused_chunk_count += 1
                chunk_source = "reused-authenticated-checkpoint"
                emit(
                    "hidden-chunk-reused",
                    completed_hidden_columns=hidden_stop,
                    total_hidden_columns=hidden_size,
                    hidden_start=hidden_start,
                    hidden_stop=hidden_stop,
                    checkpoint_sha256=checkpoint_receipt["checkpoint_sha256"],
                    target_chunk_sha256=checkpoint_receipt["target_chunk_sha256"],
                )
            else:
                weighted_source_columns = source_columns * sqrt_weights.unsqueeze(1)
                try:
                    rhs = torch.sparse.mm(
                        design_transpose,
                        weighted_source_columns,
                    )
                    solution, solve_report = _pcg_ridge_solve(
                        design,
                        design_row_blocks,
                        rhs,
                        ridge=ridge_value,
                        maximum_iterations=iterations,
                        tolerance=effective_tolerance,
                        iteration_callback=(
                            lambda iteration_index, residual_norm, residual_tolerance, final, elapsed_seconds, *, hidden_start=hidden_start, hidden_stop=hidden_stop: (
                                emit(
                                    "solver-heartbeat",
                                    completed_hidden_columns=hidden_start,
                                    total_hidden_columns=hidden_size,
                                    hidden_start=hidden_start,
                                    hidden_stop=hidden_stop,
                                    solver_iteration=iteration_index,
                                    maximum_solver_iterations=iterations,
                                    maximum_l2_normal_residual=residual_norm,
                                    maximum_l2_normal_residual_tolerance=(
                                        residual_tolerance
                                    ),
                                    final_heartbeat=final,
                                    elapsed_seconds=elapsed_seconds,
                                    elapsed_timing_collected=False,
                                )
                            )
                        ),
                    )
                except RuntimeError as exc:
                    raise Lfm2SemanticStepTokenFoldError(
                        "Sparse ridge solve failed at the numeric placement."
                    ) from exc
                solver_computed_chunk_count += 1
                chunk_source = "computed-sparse-ridge"
                if solved_chunk_publisher is not None:
                    checkpoint_body = {
                        "schema": LFM2_SEMANTIC_STEP_CHUNK_CHECKPOINT_SCHEMA,
                        "phase_cache_key": effective_phase_cache_key,
                        "request_sha256": request["request_sha256"],
                        "chunk_input_sha256": chunk_input_sha256,
                        "hidden_start": hidden_start,
                        "hidden_stop": hidden_stop,
                        "target_row_count": target_count,
                        "target_chunk_shape": [
                            target_count,
                            hidden_stop - hidden_start,
                        ],
                        "target_chunk_dtype": _dtype_name(solution),
                        "target_chunk_sha256": _tensor_sha256(solution),
                        "solver_report": solve_report,
                        "operation_audit": dict(_MODEL_OPERATION_AUDIT),
                    }
                    published_checkpoint = Lfm2SemanticStepSolvedChunkCheckpoint(
                        solution.detach().to(device="cpu"),
                        _sealed(checkpoint_body, "checkpoint_sha256"),
                    )
                    solved_chunk_publisher(published_checkpoint)
                    checkpoint_publisher_invocation_count += 1
                    emit(
                        "hidden-chunk-checkpoint-published",
                        completed_hidden_columns=hidden_stop,
                        total_hidden_columns=hidden_size,
                        hidden_start=hidden_start,
                        hidden_stop=hidden_stop,
                        checkpoint_sha256=published_checkpoint.as_dict()[
                            "checkpoint_sha256"
                        ],
                        target_chunk_sha256=published_checkpoint.as_dict()[
                            "target_chunk_sha256"
                        ],
                    )
            solved[:, hidden_start:hidden_stop].copy_(solution)
            # Verification materializes only this partial-width transient, never
            # a persistent donor-vocabulary-by-hidden reconstructed table.
            for donor_start in range(0, donor_count, verification_chunk):
                donor_stop = min(donor_start + verification_chunk, donor_count)
                selected_spans = spans[donor_start:donor_stop]
                local_rows, target_ids, span_weights = _runtime_weights_for_spans(
                    selected_spans,
                    design=positional_design,
                    device=solve_device,
                    dtype=solve_dtype,
                )
                reconstructed = solution.new_zeros(
                    (donor_stop - donor_start, hidden_stop - hidden_start)
                )
                reconstructed.index_add_(
                    0,
                    local_rows,
                    solution.index_select(0, target_ids) * span_weights.unsqueeze(1),
                )
                observed = source_columns[donor_start:donor_stop]
                residual = reconstructed - observed
                weighted_residual = residual * sqrt_weights[
                    donor_start:donor_stop
                ].unsqueeze(1)
                residual_sum_squares += float(weighted_residual.square().sum().item())
                unweighted_residual_sum_squares += float(residual.square().sum().item())
                source_sum_squares += float(
                    (observed * sqrt_weights[donor_start:donor_stop].unsqueeze(1))
                    .square()
                    .sum()
                    .item()
                )
                maximum_absolute_reconstruction_residual = max(
                    maximum_absolute_reconstruction_residual,
                    float(residual.abs().max().item()),
                )
                maximum_absolute_weighted_reconstruction_residual = max(
                    maximum_absolute_weighted_reconstruction_residual,
                    float(weighted_residual.abs().max().item()),
                )
            report = {
                "hidden_start": hidden_start,
                "hidden_stop": hidden_stop,
                "chunk_source": chunk_source,
                **solve_report,
                "target_chunk_sha256": _tensor_sha256(solution),
            }
            chunk_reports.append(report)
            emit(
                "hidden-chunk-complete",
                completed_hidden_columns=hidden_stop,
                total_hidden_columns=hidden_size,
                hidden_start=hidden_start,
                hidden_stop=hidden_stop,
                target_chunk_sha256=report["target_chunk_sha256"],
                solver_iterations=solve_report["iterations"],
            )
        published = (
            solved.to(
                device=publish_device,
                dtype=publish_dtype,
            )
            .detach()
            .clone()
            .contiguous()
        )
    if not bool(torch.isfinite(published).all()):
        raise Lfm2SemanticStepTokenFoldError(
            "Published semantic-step target table is nonfinite."
        )
    emit(
        "publication-complete",
        completed_hidden_columns=hidden_size,
        total_hidden_columns=hidden_size,
        target_table_sha256=_tensor_sha256(published),
    )
    learned_parameter_count = target_count * hidden_size
    body = {
        "schema": LFM2_SEMANTIC_STEP_TOKEN_FOLD_SCHEMA,
        "status": "analytic-sparse-ridge-semantic-step-table-compiled",
        "span_evidence_sha256": evidence["evidence_sha256"],
        "source_donor_table_sha256": source_sha256,
        "source_donor_capacity_table_sha256": source_capacity_sha256,
        "source_table_role": "caller-authenticated-tied-input-output",
        "source_dtype": _dtype_name(source),
        "source_device_type": source.device.type,
        "donor_row_count": donor_count,
        "donor_model_row_capacity": donor_capacity,
        "undefined_donor_row_count": donor_capacity - donor_count,
        "undefined_donor_rows_excluded_from_solve_and_runtime_scores": True,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "target_table_shape": [target_count, hidden_size],
        "target_table_dtype": _dtype_name(published),
        "target_table_device": str(published.device),
        "target_table_sha256": _tensor_sha256(published),
        "span_offsets_sha256": _tensor_sha256(span_offsets),
        "span_target_ids_sha256": _tensor_sha256(span_target_ids),
        "span_token_count": int(span_target_ids.numel()),
        "positional_design": positional_design,
        "positional_channels_are_fixed_analytic_metadata": True,
        "positional_order_is_representable": positional_design != "uniform",
        "donor_row_weights_sha256": weights_sha256,
        "donor_row_weights_shape": [donor_count],
        "donor_row_weights_source": weights_source,
        "minimum_donor_row_weight": float(row_weights.min().item()),
        "maximum_donor_row_weight": float(row_weights.max().item()),
        "sum_donor_row_weights": float(row_weights.sum().item()),
        "positive_weighted_donor_row_count": int((row_weights > 0).sum().item()),
        "all_donor_rows_retained_in_weighted_solve": True,
        "donor_row_weights_are_learned_parameters": False,
        "sqrt_donor_row_weights_applied_to_design_and_rhs": True,
        "sqrt_donor_row_weights_applied_to_verification_residual": True,
        "ridge": ridge_value,
        "solver": "matrix-free-preconditioned-conjugate-gradient",
        "requested_solver_tolerance": tolerance,
        "effective_solver_tolerance": effective_tolerance,
        "maximum_solver_iterations": iterations,
        "requested_hidden_columns_per_chunk": requested_hidden_chunk,
        "effective_hidden_columns_per_chunk": effective_hidden_chunk,
        "verification_donor_rows_per_chunk": verification_chunk,
        "solver_chunk_reports": chunk_reports,
        "solver_hidden_chunk_count": len(chunk_reports),
        "solver_computed_chunk_count": solver_computed_chunk_count,
        "checkpoint_reused_chunk_count": checkpoint_reused_chunk_count,
        "checkpoint_loader_supplied": solved_chunk_loader is not None,
        "checkpoint_loader_invocation_count": (checkpoint_loader_invocation_count),
        "checkpoint_publisher_supplied": solved_chunk_publisher is not None,
        "checkpoint_publisher_invocation_count": (
            checkpoint_publisher_invocation_count
        ),
        "solved_chunks_are_resumable_by_exact_phase_and_input_hash": True,
        "all_solver_chunks_converged": True,
        "reconstruction_root_mean_square_error": math.sqrt(
            unweighted_residual_sum_squares / (donor_count * hidden_size)
        ),
        "weighted_reconstruction_root_mean_square_error": math.sqrt(
            residual_sum_squares / (donor_count * hidden_size)
        ),
        "unweighted_reconstruction_root_mean_square_error": math.sqrt(
            unweighted_residual_sum_squares / (donor_count * hidden_size)
        ),
        "relative_reconstruction_l2_error": math.sqrt(
            residual_sum_squares
            / max(source_sum_squares, torch.finfo(torch.float64).tiny)
        ),
        "maximum_absolute_reconstruction_residual": (
            maximum_absolute_reconstruction_residual
        ),
        "maximum_absolute_weighted_reconstruction_residual": (
            maximum_absolute_weighted_reconstruction_residual
        ),
        "diagnostic_metrics_are_not_equivalence_claims": True,
        "learned_parameter_tensor_count": 1,
        "learned_parameter_count": learned_parameter_count,
        "routing_parameter_count": 0,
        "routing_metadata_is_integer_only": True,
        "parameter_alias_count": 0,
        "full_reconstructed_donor_table_materializations": 0,
        "full_reconstructed_donor_table_persisted": False,
        "matrix_free_operator_donor_row_block_count": len(design_row_blocks),
        "maximum_matrix_free_operator_donor_rows_per_block": max(
            int(block.shape[0]) for block, _ in design_row_blocks
        ),
        "pcg_operator_uses_strict_partial_donor_row_blocks": (
            donor_count <= 1
            or all(int(block.shape[0]) < donor_count for block, _ in design_row_blocks)
        ),
        "partial_width_reconstruction_blocks_are_transient": True,
        "parent_or_replay_head_required_at_runtime": False,
        "target_span_emission_is_external_to_body_clock": True,
        "semantic_body_steps_per_donor_action": 1,
        "suffix_zero_width_or_unresolved_rows_accepted": False,
        "phase_cache_key": effective_phase_cache_key,
        "progress_record_count": progress_count,
        "progress_final_sha256": progress_previous,
        "progress_callback_supplied": progress_callback is not None,
        "progress_callback_invocation_count": progress_callback_invocations,
        **placement_receipt,
        "operation_audit": dict(_MODEL_OPERATION_AUDIT),
    }
    return Lfm2SemanticStepTokenFold(
        published,
        span_offsets,
        span_target_ids,
        _sealed(body, "result_sha256"),
    )


__all__ = [
    "LFM2_SEMANTIC_STEP_CHUNK_CHECKPOINT_SCHEMA",
    "LFM2_SEMANTIC_STEP_CHUNK_REQUEST_SCHEMA",
    "LFM2_SEMANTIC_STEP_CODEC_RESULT_SCHEMA",
    "LFM2_SEMANTIC_STEP_LIVE_TABLE_SESSION_SCHEMA",
    "LFM2_SEMANTIC_STEP_PROGRESS_SCHEMA",
    "LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA",
    "LFM2_SEPARATE_SPARSE_SEMANTIC_CODE_PAIR_SCHEMA",
    "LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA",
    "LFM2_SEMANTIC_STEP_TOKEN_FOLD_SCHEMA",
    "LFM2_SPARSE_SEMANTIC_CODE_SCHEMA",
    "POSITIONAL_CHANNEL_DESIGNS",
    "Lfm2SemanticStepCodecResult",
    "Lfm2SemanticStepLiveTableSession",
    "Lfm2SemanticStepRoutingRuntime",
    "Lfm2SemanticStepSolvedChunkCheckpoint",
    "Lfm2SemanticStepTokenFold",
    "Lfm2SemanticStepTokenFoldError",
    "bind_lfm2_semantic_step_hard_anchor_runtime",
    "bind_lfm2_separate_sparse_semantic_code_runtime",
    "bind_lfm2_sparse_semantic_code_runtime",
    "compile_lfm2_semantic_step_token_fold",
    "materialize_lfm2_semantic_step_design_matrix",
    "seal_lfm2_semantic_step_span_evidence",
]
