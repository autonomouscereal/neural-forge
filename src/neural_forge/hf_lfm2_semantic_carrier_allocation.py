"""Deterministic carrier allocation for the LFM2 sparse semantic bridge.

Carrier allocation is a local, analytic permutation of an existing semantic
codebook.  A demanded non-direct donor is planted into an eligible target row,
the direct donor previously protected by that row is evicted, and the caller
supplies independently computed replacement routes for that evicted donor in
both semantic channels.

This module deliberately does not load a model, capture prompts, run a parent,
train, prune, quantize, or solve a global residual map.  Selection evidence
must be sealed as fold-fit evidence; evaluation evidence is rejected.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from neural_forge.dynamic_circuit_ir import canonical_sha256


LFM2_SEMANTIC_CARRIER_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-semantic-carrier-demand-evidence-v1"
)
LFM2_SEMANTIC_CARRIER_INVENTORY_SCHEMA = (
    "neural-forge-lfm2-semantic-carrier-inventory-v1"
)
LFM2_SEMANTIC_CARRIER_SELECTION_SCHEMA = (
    "neural-forge-lfm2-semantic-carrier-selection-v1"
)
LFM2_SEMANTIC_CARRIER_ALLOCATION_SCHEMA = (
    "neural-forge-lfm2-semantic-carrier-allocation-v1"
)
LFM2_SEMANTIC_CARRIER_MATERIALIZATION_SCHEMA = (
    "neural-forge-lfm2-semantic-carrier-materialization-v1"
)
LFM2_SEMANTIC_CARRIER_SOLVER_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-semantic-carrier-replacement-solver-v1"
)

_SUPPORTED_TABLE_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)
_TENSOR_HASH_DOMAIN = b"neural-forge-lfm2-semantic-carrier-tensor-v1\0"
_ZERO_OPERATION_AUDIT = {
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


class Lfm2SemanticCarrierAllocationError(RuntimeError):
    """Carrier evidence, selection, routing, or tensor bindings are invalid."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise Lfm2SemanticCarrierAllocationError(
                "Receipts cannot contain non-finite values."
            )
        return value
    raise Lfm2SemanticCarrierAllocationError(
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
    plain = _plain_json(body)
    return {**plain, field: canonical_sha256(plain)}


def _verify_sealed(
    value: Mapping[str, Any],
    *,
    schema: str,
    field: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticCarrierAllocationError(f"{name} must be a mapping.")
    plain = _plain_json(value)
    supplied = plain.pop(field, None)
    if supplied != canonical_sha256(plain):
        raise Lfm2SemanticCarrierAllocationError(f"{name} seal is invalid.")
    if plain.get("schema") != schema:
        raise Lfm2SemanticCarrierAllocationError(f"{name} schema is unsupported.")
    return {**plain, field: supplied}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SemanticCarrierAllocationError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SemanticCarrierAllocationError(f"{name} must be nonnegative.")
    return int(value)


def _validated_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{name} must be exactly 64 hexadecimal characters."
        )
    return value.lower()


def _dtype_name(value: Tensor | torch.dtype) -> str:
    dtype = value.dtype if isinstance(value, Tensor) else value
    return str(dtype).removeprefix("torch.")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach()
    digest = hashlib.sha256()
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(_dtype_name(tensor).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    cpu = tensor.to(device="cpu").contiguous()
    digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _byte_equal(left: Tensor, right: Tensor) -> bool:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    left_bytes = left.detach().to(device="cpu").contiguous().view(torch.uint8)
    right_bytes = right.detach().to(device="cpu").contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _validated_table(value: Tensor, name: str, *, copy: bool = True) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) <= 0
        or value.dtype not in _SUPPORTED_TABLE_DTYPES
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{name} must be one detached finite dense FP16/BF16/FP32/FP64 matrix."
        )
    detached = value.detach()
    return detached.clone().contiguous() if copy else detached


def _validated_int64_vector(value: Tensor, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.device.type != "cpu"
        or value.dtype != torch.int64
        or value.requires_grad
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{name} must be one detached CPU int64 vector."
        )
    return value.detach().clone().contiguous()


def _validated_count_vector(value: Tensor, name: str) -> Tensor:
    result = _validated_int64_vector(value, name)
    if bool((result < 0).any()):
        raise Lfm2SemanticCarrierAllocationError(f"{name} cannot be negative.")
    return result


def _validated_float64_vector(value: Tensor, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{name} must be one detached finite CPU float64 vector."
        )
    result = value.detach().clone().contiguous()
    if bool((result < 0).any()):
        raise Lfm2SemanticCarrierAllocationError(f"{name} cannot be negative.")
    return result


def _validated_mapping(
    value: Tensor,
    *,
    target_row_count: int,
    donor_row_count: int,
) -> Tensor:
    mapping = _validated_int64_vector(value, "target_to_donor")
    if int(mapping.numel()) != target_row_count:
        raise Lfm2SemanticCarrierAllocationError(
            "target_to_donor must contain one donor per target row."
        )
    if bool((mapping < 0).any()) or bool((mapping >= donor_row_count).any()):
        raise Lfm2SemanticCarrierAllocationError(
            "target_to_donor leaves the defined donor table."
        )
    if int(torch.unique(mapping).numel()) != target_row_count:
        raise Lfm2SemanticCarrierAllocationError("target_to_donor must be injective.")
    return mapping


def _validated_csr(
    offsets: Tensor,
    target_ids: Tensor,
    coefficients: Tensor,
    *,
    row_count: int,
    target_row_count: int,
    name: str,
) -> tuple[Tensor, Tensor, Tensor]:
    checked_offsets = _validated_int64_vector(offsets, f"{name} offsets")
    checked_ids = _validated_int64_vector(target_ids, f"{name} target IDs")
    if (
        not isinstance(coefficients, Tensor)
        or coefficients.ndim != 1
        or coefficients.device.type != "cpu"
        or coefficients.dtype != torch.float32
        or coefficients.requires_grad
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{name} coefficients must be one detached finite CPU FP32 vector."
        )
    checked_coefficients = coefficients.detach().clone().contiguous()
    if (
        int(checked_offsets.numel()) != row_count + 1
        or int(checked_offsets[0]) != 0
        or int(checked_offsets[-1]) != int(checked_ids.numel())
        or int(checked_ids.numel()) != int(checked_coefficients.numel())
        or bool((checked_offsets[1:] <= checked_offsets[:-1]).any())
        or bool((checked_ids < 0).any())
        or bool((checked_ids >= target_row_count).any())
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{name} CSR structure or bounds are invalid."
        )
    for row_id in range(row_count):
        start = int(checked_offsets[row_id])
        stop = int(checked_offsets[row_id + 1])
        row = checked_ids[start:stop]
        if int(row.numel()) > 1 and bool((row[1:] <= row[:-1]).any()):
            raise Lfm2SemanticCarrierAllocationError(
                f"{name} row {row_id} target IDs must be unique and ascending."
            )
    return checked_offsets, checked_ids, checked_coefficients


def _validated_reverse_incidence_csr(
    offsets: Tensor,
    donor_ids: Tensor,
    *,
    target_row_count: int,
    donor_row_count: int,
    name: str,
) -> tuple[Tensor, Tensor]:
    """Validate target -> unique non-direct-donor incidence, including empty rows."""

    checked_offsets = _validated_int64_vector(offsets, f"{name} offsets")
    checked_donor_ids = _validated_int64_vector(donor_ids, f"{name} donor IDs")
    if (
        int(checked_offsets.numel()) != target_row_count + 1
        or int(checked_offsets[0]) != 0
        or int(checked_offsets[-1]) != int(checked_donor_ids.numel())
        or bool((checked_offsets[1:] < checked_offsets[:-1]).any())
        or bool((checked_donor_ids < 0).any())
        or bool((checked_donor_ids >= donor_row_count).any())
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{name} reverse CSR structure or bounds are invalid."
        )
    for target_id in range(target_row_count):
        start = int(checked_offsets[target_id])
        stop = int(checked_offsets[target_id + 1])
        row = checked_donor_ids[start:stop]
        if int(row.numel()) > 1 and bool((row[1:] <= row[:-1]).any()):
            raise Lfm2SemanticCarrierAllocationError(
                f"{name} target {target_id} donor IDs must be unique and ascending."
            )
    return checked_offsets, checked_donor_ids


def _direct_inverse(mapping: Tensor, donor_row_count: int) -> Tensor:
    inverse = torch.full((donor_row_count,), -1, dtype=torch.int64)
    inverse[mapping] = torch.arange(int(mapping.numel()), dtype=torch.int64)
    return inverse


def _require_exact_direct_rows(
    offsets: Tensor,
    target_ids: Tensor,
    coefficients: Tensor,
    mapping: Tensor,
    *,
    name: str,
) -> None:
    one_bits = int(torch.tensor(1.0, dtype=torch.float32).view(torch.int32).item())
    for target_id, donor_id in enumerate(mapping.tolist()):
        start = int(offsets[donor_id])
        stop = int(offsets[donor_id + 1])
        if (
            stop - start != 1
            or int(target_ids[start]) != target_id
            or int(coefficients[start].view(torch.int32).item()) != one_bits
        ):
            raise Lfm2SemanticCarrierAllocationError(
                f"{name} direct donor {donor_id} must be exact one-hot at "
                f"target row {target_id}."
            )


def _validated_base_codes(
    *,
    input_offsets: Tensor,
    input_target_ids: Tensor,
    input_coefficients: Tensor,
    output_offsets: Tensor,
    output_target_ids: Tensor,
    output_coefficients: Tensor,
    target_to_donor: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(input_offsets, Tensor) or input_offsets.ndim != 1:
        raise Lfm2SemanticCarrierAllocationError(
            "input offsets must define the donor row count."
        )
    donor_count = int(input_offsets.numel()) - 1
    if donor_count <= 0:
        raise Lfm2SemanticCarrierAllocationError(
            "The semantic code must contain donor rows."
        )
    if not isinstance(target_to_donor, Tensor) or target_to_donor.ndim != 1:
        raise Lfm2SemanticCarrierAllocationError(
            "target_to_donor must define the target row count."
        )
    target_count = int(target_to_donor.numel())
    mapping = _validated_mapping(
        target_to_donor,
        target_row_count=target_count,
        donor_row_count=donor_count,
    )
    input_code = _validated_csr(
        input_offsets,
        input_target_ids,
        input_coefficients,
        row_count=donor_count,
        target_row_count=target_count,
        name="input semantic code",
    )
    output_code = _validated_csr(
        output_offsets,
        output_target_ids,
        output_coefficients,
        row_count=donor_count,
        target_row_count=target_count,
        name="output semantic code",
    )
    _require_exact_direct_rows(*input_code, mapping, name="input semantic code")
    _require_exact_direct_rows(*output_code, mapping, name="output semantic code")
    return (*input_code, *output_code, mapping)


def _base_binding(
    input_offsets: Tensor,
    input_target_ids: Tensor,
    input_coefficients: Tensor,
    output_offsets: Tensor,
    output_target_ids: Tensor,
    output_coefficients: Tensor,
    mapping: Tensor,
) -> dict[str, Any]:
    return {
        "input_offsets_sha256": _tensor_sha256(input_offsets),
        "input_target_ids_sha256": _tensor_sha256(input_target_ids),
        "input_coefficients_sha256": _tensor_sha256(input_coefficients),
        "output_offsets_sha256": _tensor_sha256(output_offsets),
        "output_target_ids_sha256": _tensor_sha256(output_target_ids),
        "output_coefficients_sha256": _tensor_sha256(output_coefficients),
        "target_to_donor_sha256": _tensor_sha256(mapping),
        "input_coefficient_count": int(input_coefficients.numel()),
        "output_coefficient_count": int(output_coefficients.numel()),
    }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticCarrierDemandEvidence:
    donor_occurrence_counts: Tensor
    donor_winner_counts: Tensor
    donor_reconstruction_errors: Tensor
    donor_eviction_costs: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SEMANTIC_CARRIER_EVIDENCE_SCHEMA,
            field="artifact_sha256",
            name="carrier demand evidence",
        )
        occurrence = _validated_count_vector(
            self.donor_occurrence_counts, "donor_occurrence_counts"
        )
        winners = _validated_count_vector(
            self.donor_winner_counts, "donor_winner_counts"
        )
        errors = _validated_float64_vector(
            self.donor_reconstruction_errors, "donor_reconstruction_errors"
        )
        costs = _validated_float64_vector(
            self.donor_eviction_costs, "donor_eviction_costs"
        )
        donor_count = _positive_int(value.get("donor_row_count"), "donor_row_count")
        if any(
            int(tensor.numel()) != donor_count
            for tensor in (occurrence, winners, errors, costs)
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier demand evidence vectors must cover every donor row."
            )
        if (
            value.get("partition_role") != "fold-fit"
            or value.get("selection_allowed") is not True
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier selection accepts sealed fold-fit evidence only."
            )
        _validated_sha256(value.get("source_artifact_sha256"), "source_artifact_sha256")
        _validated_sha256(value.get("source_corpus_sha256"), "source_corpus_sha256")
        _validated_sha256(value.get("parent_identity_sha256"), "parent_identity_sha256")
        _validated_sha256(
            value.get("fit_partition_manifest_sha256"),
            "fit_partition_manifest_sha256",
        )
        _validated_sha256(
            value.get("producer_implementation_sha256"),
            "producer_implementation_sha256",
        )
        _validated_sha256(
            value.get("numeric_contract_sha256"), "numeric_contract_sha256"
        )
        _validated_sha256(
            value.get("base_semantic_binding_sha256"),
            "base_semantic_binding_sha256",
        )
        expected_hashes = {
            "donor_occurrence_counts_sha256": _tensor_sha256(occurrence),
            "donor_winner_counts_sha256": _tensor_sha256(winners),
            "donor_reconstruction_errors_sha256": _tensor_sha256(errors),
            "donor_eviction_costs_sha256": _tensor_sha256(costs),
        }
        if (
            any(value.get(key) != item for key, item in expected_hashes.items())
            or value.get("review_or_evaluation_evidence_allowed") is not False
            or value.get("occurrence_and_winner_counts_are_structural_evidence")
            is not True
            or value.get(
                "reconstruction_errors_and_eviction_costs_are_float64_diagnostics"
            )
            is not True
            or value.get("operation_audit") != _ZERO_OPERATION_AUDIT
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier demand evidence tensor binding is invalid."
            )
        object.__setattr__(self, "donor_occurrence_counts", occurrence)
        object.__setattr__(self, "donor_winner_counts", winners)
        object.__setattr__(self, "donor_reconstruction_errors", errors)
        object.__setattr__(self, "donor_eviction_costs", costs)
        object.__setattr__(self, "receipt", _deep_freeze(value))

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def build_lfm2_semantic_carrier_demand_evidence(
    donor_occurrence_counts: Tensor,
    donor_winner_counts: Tensor,
    donor_reconstruction_errors: Tensor,
    donor_eviction_costs: Tensor,
    *,
    source_artifact_sha256: str,
    source_corpus_sha256: str,
    parent_identity_sha256: str,
    fit_partition_manifest_sha256: str,
    producer_implementation_sha256: str,
    numeric_contract_sha256: str,
    base_semantic_binding_sha256: str,
    partition_role: str,
    selection_allowed: bool,
) -> Lfm2SemanticCarrierDemandEvidence:
    """Seal donor ranking evidence while rejecting review/evaluation provenance."""

    if partition_role != "fold-fit" or selection_allowed is not True:
        raise Lfm2SemanticCarrierAllocationError(
            "Carrier selection evidence must have partition_role='fold-fit' "
            "and selection_allowed=True."
        )
    occurrence = _validated_count_vector(
        donor_occurrence_counts, "donor_occurrence_counts"
    )
    winners = _validated_count_vector(donor_winner_counts, "donor_winner_counts")
    errors = _validated_float64_vector(
        donor_reconstruction_errors, "donor_reconstruction_errors"
    )
    costs = _validated_float64_vector(donor_eviction_costs, "donor_eviction_costs")
    donor_count = int(occurrence.numel())
    if donor_count <= 0 or any(
        int(tensor.numel()) != donor_count for tensor in (winners, errors, costs)
    ):
        raise Lfm2SemanticCarrierAllocationError(
            "Carrier demand evidence vectors must be nonempty and equally sized."
        )
    body = {
        "schema": LFM2_SEMANTIC_CARRIER_EVIDENCE_SCHEMA,
        "partition_role": "fold-fit",
        "selection_allowed": True,
        "review_or_evaluation_evidence_allowed": False,
        "source_artifact_sha256": _validated_sha256(
            source_artifact_sha256, "source_artifact_sha256"
        ),
        "source_corpus_sha256": _validated_sha256(
            source_corpus_sha256, "source_corpus_sha256"
        ),
        "parent_identity_sha256": _validated_sha256(
            parent_identity_sha256, "parent_identity_sha256"
        ),
        "fit_partition_manifest_sha256": _validated_sha256(
            fit_partition_manifest_sha256, "fit_partition_manifest_sha256"
        ),
        "producer_implementation_sha256": _validated_sha256(
            producer_implementation_sha256, "producer_implementation_sha256"
        ),
        "numeric_contract_sha256": _validated_sha256(
            numeric_contract_sha256, "numeric_contract_sha256"
        ),
        "base_semantic_binding_sha256": _validated_sha256(
            base_semantic_binding_sha256, "base_semantic_binding_sha256"
        ),
        "donor_row_count": donor_count,
        "donor_occurrence_counts_sha256": _tensor_sha256(occurrence),
        "donor_winner_counts_sha256": _tensor_sha256(winners),
        "donor_reconstruction_errors_sha256": _tensor_sha256(errors),
        "donor_eviction_costs_sha256": _tensor_sha256(costs),
        "occurrence_and_winner_counts_are_structural_evidence": True,
        "reconstruction_errors_and_eviction_costs_are_float64_diagnostics": True,
        "operation_audit": _ZERO_OPERATION_AUDIT,
    }
    return Lfm2SemanticCarrierDemandEvidence(
        occurrence,
        winners,
        errors,
        costs,
        _sealed(body, "artifact_sha256"),
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticCarrierInventory:
    target_to_donor: Tensor
    input_non_direct_incidence: Tensor
    output_non_direct_incidence: Tensor
    union_non_direct_incidence: Tensor
    input_non_direct_leverage: Tensor
    output_non_direct_leverage: Tensor
    union_non_direct_leverage: Tensor
    union_non_direct_target_offsets: Tensor
    union_non_direct_donor_ids: Tensor
    eligible_target_ids: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SEMANTIC_CARRIER_INVENTORY_SCHEMA,
            field="artifact_sha256",
            name="carrier inventory",
        )
        donor_count = _positive_int(value.get("donor_row_count"), "donor_row_count")
        target_count = _positive_int(value.get("target_row_count"), "target_row_count")
        if not isinstance(value.get("base_binding"), Mapping):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier inventory base_binding must be a mapping."
            )
        mapping = _validated_mapping(
            self.target_to_donor,
            target_row_count=target_count,
            donor_row_count=donor_count,
        )
        input_incidence = _validated_count_vector(
            self.input_non_direct_incidence, "input_non_direct_incidence"
        )
        output_incidence = _validated_count_vector(
            self.output_non_direct_incidence, "output_non_direct_incidence"
        )
        union_incidence = _validated_count_vector(
            self.union_non_direct_incidence, "union_non_direct_incidence"
        )
        input_leverage = _validated_float64_vector(
            self.input_non_direct_leverage, "input_non_direct_leverage"
        )
        output_leverage = _validated_float64_vector(
            self.output_non_direct_leverage, "output_non_direct_leverage"
        )
        union_leverage = _validated_float64_vector(
            self.union_non_direct_leverage, "union_non_direct_leverage"
        )
        reverse_offsets, reverse_donor_ids = _validated_reverse_incidence_csr(
            self.union_non_direct_target_offsets,
            self.union_non_direct_donor_ids,
            target_row_count=target_count,
            donor_row_count=donor_count,
            name="union non-direct incidence",
        )
        eligible = _validated_int64_vector(
            self.eligible_target_ids, "eligible_target_ids"
        )
        if any(
            int(tensor.numel()) != target_count
            for tensor in (
                input_incidence,
                output_incidence,
                union_incidence,
                input_leverage,
                output_leverage,
                union_leverage,
            )
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier inventory vectors must cover every target row."
            )
        expected_eligible = torch.nonzero(
            (union_incidence == 0) & (union_leverage == 0.0), as_tuple=False
        ).reshape(-1)
        if (
            not torch.equal(union_leverage, input_leverage + output_leverage)
            or bool(
                (
                    union_incidence < torch.maximum(input_incidence, output_incidence)
                ).any()
            )
            or bool((union_incidence > input_incidence + output_incidence).any())
            or not torch.equal(
                union_incidence,
                reverse_offsets[1:] - reverse_offsets[:-1],
            )
            or bool(torch.isin(reverse_donor_ids, mapping).any())
            or not torch.equal(eligible, expected_eligible)
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier inventory union incidence or leverage is inconsistent."
            )
        tensors = {
            "target_to_donor_sha256": mapping,
            "input_non_direct_incidence_sha256": input_incidence,
            "output_non_direct_incidence_sha256": output_incidence,
            "union_non_direct_incidence_sha256": union_incidence,
            "input_non_direct_leverage_sha256": input_leverage,
            "output_non_direct_leverage_sha256": output_leverage,
            "union_non_direct_leverage_sha256": union_leverage,
            "union_non_direct_target_offsets_sha256": reverse_offsets,
            "union_non_direct_donor_ids_sha256": reverse_donor_ids,
            "eligible_target_ids_sha256": eligible,
        }
        if (
            any(
                value.get(key) != _tensor_sha256(tensor)
                for key, tensor in tensors.items()
            )
            or value.get("eligible_carrier_target_count") != int(eligible.numel())
            or value.get("direct_donor_rows_excluded_from_incidence_and_leverage")
            is not True
            or value.get("reverse_union_incidence_is_complete_and_unique") is not True
            or value.get(
                "carrier_requires_zero_union_incidence_and_zero_union_leverage"
            )
            is not True
            or value.get("operation_audit") != _ZERO_OPERATION_AUDIT
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier inventory tensor binding is invalid."
            )
        object.__setattr__(self, "target_to_donor", mapping)
        object.__setattr__(self, "input_non_direct_incidence", input_incidence)
        object.__setattr__(self, "output_non_direct_incidence", output_incidence)
        object.__setattr__(self, "union_non_direct_incidence", union_incidence)
        object.__setattr__(self, "input_non_direct_leverage", input_leverage)
        object.__setattr__(self, "output_non_direct_leverage", output_leverage)
        object.__setattr__(self, "union_non_direct_leverage", union_leverage)
        object.__setattr__(self, "union_non_direct_target_offsets", reverse_offsets)
        object.__setattr__(self, "union_non_direct_donor_ids", reverse_donor_ids)
        object.__setattr__(self, "eligible_target_ids", eligible)
        object.__setattr__(self, "receipt", _deep_freeze(value))

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def build_lfm2_semantic_carrier_inventory(
    *,
    input_offsets: Tensor,
    input_target_ids: Tensor,
    input_coefficients: Tensor,
    output_offsets: Tensor,
    output_target_ids: Tensor,
    output_coefficients: Tensor,
    target_to_donor: Tensor,
) -> Lfm2SemanticCarrierInventory:
    """Measure target-row use by all non-direct donors in both routes."""

    (
        checked_input_offsets,
        checked_input_ids,
        checked_input_coefficients,
        checked_output_offsets,
        checked_output_ids,
        checked_output_coefficients,
        mapping,
    ) = _validated_base_codes(
        input_offsets=input_offsets,
        input_target_ids=input_target_ids,
        input_coefficients=input_coefficients,
        output_offsets=output_offsets,
        output_target_ids=output_target_ids,
        output_coefficients=output_coefficients,
        target_to_donor=target_to_donor,
    )
    donor_count = int(checked_input_offsets.numel()) - 1
    target_count = int(mapping.numel())
    direct_inverse = _direct_inverse(mapping, donor_count)
    input_incidence = torch.zeros(target_count, dtype=torch.int64)
    output_incidence = torch.zeros(target_count, dtype=torch.int64)
    union_incidence = torch.zeros(target_count, dtype=torch.int64)
    input_leverage = torch.zeros(target_count, dtype=torch.float64)
    output_leverage = torch.zeros(target_count, dtype=torch.float64)
    reverse_donors: list[list[int]] = [[] for _ in range(target_count)]

    for donor_id in range(donor_count):
        if int(direct_inverse[donor_id]) >= 0:
            continue
        input_start = int(checked_input_offsets[donor_id])
        input_stop = int(checked_input_offsets[donor_id + 1])
        output_start = int(checked_output_offsets[donor_id])
        output_stop = int(checked_output_offsets[donor_id + 1])
        input_row_ids = checked_input_ids[input_start:input_stop]
        output_row_ids = checked_output_ids[output_start:output_stop]
        input_incidence.index_add_(
            0, input_row_ids, torch.ones_like(input_row_ids, dtype=torch.int64)
        )
        output_incidence.index_add_(
            0, output_row_ids, torch.ones_like(output_row_ids, dtype=torch.int64)
        )
        input_leverage.index_add_(
            0,
            input_row_ids,
            checked_input_coefficients[input_start:input_stop]
            .abs()
            .to(dtype=torch.float64),
        )
        output_leverage.index_add_(
            0,
            output_row_ids,
            checked_output_coefficients[output_start:output_stop]
            .abs()
            .to(dtype=torch.float64),
        )
        union_ids = torch.unique(
            torch.cat((input_row_ids, output_row_ids)), sorted=True
        )
        union_incidence.index_add_(
            0, union_ids, torch.ones_like(union_ids, dtype=torch.int64)
        )
        for target_id in union_ids.tolist():
            reverse_donors[int(target_id)].append(donor_id)

    union_leverage = input_leverage + output_leverage
    reverse_offsets_values = [0]
    reverse_donor_values: list[int] = []
    for donor_ids in reverse_donors:
        reverse_donor_values.extend(donor_ids)
        reverse_offsets_values.append(len(reverse_donor_values))
    reverse_offsets = torch.tensor(reverse_offsets_values, dtype=torch.int64)
    reverse_donor_ids = torch.tensor(reverse_donor_values, dtype=torch.int64)
    eligible = torch.nonzero(
        (union_incidence == 0) & (union_leverage == 0.0), as_tuple=False
    ).reshape(-1)
    base_binding = _base_binding(
        checked_input_offsets,
        checked_input_ids,
        checked_input_coefficients,
        checked_output_offsets,
        checked_output_ids,
        checked_output_coefficients,
        mapping,
    )
    body = {
        "schema": LFM2_SEMANTIC_CARRIER_INVENTORY_SCHEMA,
        "algorithm": "union-input-output-non-direct-incidence-and-l1-leverage-v1",
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "base_binding": base_binding,
        "target_to_donor_sha256": _tensor_sha256(mapping),
        "input_non_direct_incidence_sha256": _tensor_sha256(input_incidence),
        "output_non_direct_incidence_sha256": _tensor_sha256(output_incidence),
        "union_non_direct_incidence_sha256": _tensor_sha256(union_incidence),
        "input_non_direct_leverage_sha256": _tensor_sha256(input_leverage),
        "output_non_direct_leverage_sha256": _tensor_sha256(output_leverage),
        "union_non_direct_leverage_sha256": _tensor_sha256(union_leverage),
        "union_non_direct_target_offsets_sha256": _tensor_sha256(reverse_offsets),
        "union_non_direct_donor_ids_sha256": _tensor_sha256(reverse_donor_ids),
        "eligible_target_ids_sha256": _tensor_sha256(eligible),
        "eligible_carrier_target_count": int(eligible.numel()),
        "direct_donor_rows_excluded_from_incidence_and_leverage": True,
        "reverse_union_incidence_is_complete_and_unique": True,
        "carrier_requires_zero_union_incidence_and_zero_union_leverage": True,
        "operation_audit": _ZERO_OPERATION_AUDIT,
    }
    return Lfm2SemanticCarrierInventory(
        mapping,
        input_incidence,
        output_incidence,
        union_incidence,
        input_leverage,
        output_leverage,
        union_leverage,
        reverse_offsets,
        reverse_donor_ids,
        eligible,
        _sealed(body, "artifact_sha256"),
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticCarrierSelection:
    carrier_target_ids: Tensor
    promoted_donor_ids: Tensor
    evicted_donor_ids: Tensor
    required_rerouted_donor_ids: Tensor
    promotion_benefit_scores: Tensor
    eviction_cost_scores: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SEMANTIC_CARRIER_SELECTION_SCHEMA,
            field="artifact_sha256",
            name="carrier selection",
        )
        carriers = _validated_int64_vector(
            self.carrier_target_ids, "carrier_target_ids"
        )
        promoted = _validated_int64_vector(
            self.promoted_donor_ids, "promoted_donor_ids"
        )
        evicted = _validated_int64_vector(self.evicted_donor_ids, "evicted_donor_ids")
        rerouted = _validated_int64_vector(
            self.required_rerouted_donor_ids, "required_rerouted_donor_ids"
        )
        benefits = _validated_float64_vector(
            self.promotion_benefit_scores, "promotion_benefit_scores"
        )
        costs = _validated_float64_vector(
            self.eviction_cost_scores, "eviction_cost_scores"
        )
        count = _positive_int(value.get("allocation_count"), "allocation_count")
        if any(
            int(tensor.numel()) != count
            for tensor in (carriers, promoted, evicted, benefits, costs)
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier selection vectors must have allocation_count entries."
            )
        if (
            int(torch.unique(carriers).numel()) != count
            or int(torch.unique(promoted).numel()) != count
            or int(torch.unique(evicted).numel()) != count
            or bool(torch.isin(promoted, evicted).any())
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier, promoted, and evicted IDs must define a unique permutation."
            )
        if (
            int(rerouted.numel()) <= 0
            or not torch.equal(rerouted, torch.unique(rerouted, sorted=True))
            or bool(torch.isin(rerouted, promoted).any())
            or not bool(torch.isin(evicted, rerouted).all())
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Required rerouted donors must be sorted, unique, include every "
                "evicted direct donor, and exclude promoted exact donors."
            )
        donor_count = _positive_int(value.get("donor_row_count"), "donor_row_count")
        target_count = _positive_int(value.get("target_row_count"), "target_row_count")
        _validated_sha256(
            value.get("inventory_artifact_sha256"), "inventory_artifact_sha256"
        )
        _validated_sha256(
            value.get("evidence_artifact_sha256"), "evidence_artifact_sha256"
        )
        _validated_sha256(
            value.get("evidence_numeric_contract_sha256"),
            "evidence_numeric_contract_sha256",
        )
        if not isinstance(value.get("base_binding"), Mapping):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier selection base_binding must be a mapping."
            )
        _validated_sha256(
            value.get("protection_artifact_sha256"),
            "protection_artifact_sha256",
        )
        protected_values = value.get("protected_donor_ids")
        if (
            not isinstance(protected_values, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in protected_values
            )
            or protected_values != sorted(set(protected_values))
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier selection protected donor IDs are invalid."
            )
        protected = torch.tensor(protected_values, dtype=torch.int64)
        if (
            bool((protected < 0).any())
            or bool((protected >= donor_count).any())
            or value.get("protected_donor_ids_sha256") != _tensor_sha256(protected)
            or bool(torch.isin(promoted, protected).any())
            or bool(torch.isin(evicted, protected).any())
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier selection violates its protected donor set."
            )
        if (
            bool((carriers < 0).any())
            or bool((carriers >= target_count).any())
            or bool((promoted < 0).any())
            or bool((promoted >= donor_count).any())
            or bool((evicted < 0).any())
            or bool((evicted >= donor_count).any())
            or bool((rerouted < 0).any())
            or bool((rerouted >= donor_count).any())
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier selection IDs leave the declared tables."
            )
        tensors = {
            "carrier_target_ids_sha256": carriers,
            "promoted_donor_ids_sha256": promoted,
            "evicted_donor_ids_sha256": evicted,
            "required_rerouted_donor_ids_sha256": rerouted,
            "promotion_benefit_scores_sha256": benefits,
            "eviction_cost_scores_sha256": costs,
        }
        if (
            any(
                value.get(key) != _tensor_sha256(tensor)
                for key, tensor in tensors.items()
            )
            or value.get("evidence_partition_role") != "fold-fit"
            or value.get("required_rerouted_donor_count") != int(rerouted.numel())
            or value.get("required_rerouted_donor_ids") != rerouted.tolist()
            or value.get(
                "rerouted_set_is_reverse_union_plus_evictions_minus_promotions"
            )
            is not True
            or not isinstance(value.get("positive_incidence_target_rows_allowed"), bool)
            or value.get("review_or_evaluation_evidence_allowed") is not False
            or value.get("operation_audit") != _ZERO_OPERATION_AUDIT
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier selection tensor binding is invalid."
            )
        object.__setattr__(self, "carrier_target_ids", carriers)
        object.__setattr__(self, "promoted_donor_ids", promoted)
        object.__setattr__(self, "evicted_donor_ids", evicted)
        object.__setattr__(self, "required_rerouted_donor_ids", rerouted)
        object.__setattr__(self, "promotion_benefit_scores", benefits)
        object.__setattr__(self, "eviction_cost_scores", costs)
        object.__setattr__(self, "receipt", _deep_freeze(value))

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def select_lfm2_semantic_carriers(
    inventory: Lfm2SemanticCarrierInventory,
    evidence: Lfm2SemanticCarrierDemandEvidence,
    *,
    promotion_count: int,
    protected_donor_ids: Sequence[int],
    protection_artifact_sha256: str,
    winner_weight: int = 8,
    allow_positive_incidence: bool = False,
) -> Lfm2SemanticCarrierSelection:
    """Select demanded promotions and the least disruptive target-row carriers.

    The default preserves the original zero-incidence allocator.  When
    ``allow_positive_incidence`` is true, every unprotected target row is a
    candidate and ranking first minimizes unique C_in/C_out donor incidence,
    then combined L1 leverage, then the evicted direct donor's demand/cost.
    """

    if not isinstance(inventory, Lfm2SemanticCarrierInventory):
        raise Lfm2SemanticCarrierAllocationError(
            "inventory must be a validated carrier inventory."
        )
    if not isinstance(evidence, Lfm2SemanticCarrierDemandEvidence):
        raise Lfm2SemanticCarrierAllocationError(
            "evidence must be validated fold-fit carrier demand evidence."
        )
    count = _positive_int(promotion_count, "promotion_count")
    weight = _nonnegative_int(winner_weight, "winner_weight")
    if not isinstance(allow_positive_incidence, bool):
        raise Lfm2SemanticCarrierAllocationError(
            "allow_positive_incidence must be a boolean."
        )
    donor_count = int(inventory.receipt["donor_row_count"])
    target_count = int(inventory.receipt["target_row_count"])
    if int(evidence.receipt["donor_row_count"]) != donor_count:
        raise Lfm2SemanticCarrierAllocationError(
            "Carrier evidence donor count differs from the inventory."
        )
    base_binding_sha256 = canonical_sha256(
        _plain_json(inventory.receipt["base_binding"])
    )
    if evidence.receipt.get("base_semantic_binding_sha256") != base_binding_sha256:
        raise Lfm2SemanticCarrierAllocationError(
            "Carrier evidence was produced for a different semantic base binding."
        )
    if (
        evidence.receipt.get("partition_role") != "fold-fit"
        or evidence.receipt.get("selection_allowed") is not True
    ):
        raise Lfm2SemanticCarrierAllocationError(
            "Evaluation or review evidence cannot drive carrier selection."
        )
    if isinstance(protected_donor_ids, (str, bytes)) or not isinstance(
        protected_donor_ids, Sequence
    ):
        raise Lfm2SemanticCarrierAllocationError(
            "protected_donor_ids must be an integer sequence."
        )
    protected_values: list[int] = []
    for item in protected_donor_ids:
        if isinstance(item, bool) or not isinstance(item, int):
            raise Lfm2SemanticCarrierAllocationError(
                "protected_donor_ids must contain integers."
            )
        donor_id = int(item)
        if donor_id < 0 or donor_id >= donor_count:
            raise Lfm2SemanticCarrierAllocationError(
                "protected_donor_ids leaves the donor table."
            )
        protected_values.append(donor_id)
    protected = torch.tensor(sorted(set(protected_values)), dtype=torch.int64)
    protection_sha256 = _validated_sha256(
        protection_artifact_sha256, "protection_artifact_sha256"
    )
    protected_set = set(protected.tolist())
    mapping = inventory.target_to_donor
    direct_set = set(mapping.tolist())

    promotion_candidates: list[tuple[float, int, int, int]] = []
    for donor_id in range(donor_count):
        if donor_id in direct_set or donor_id in protected_set:
            continue
        occurrence = int(evidence.donor_occurrence_counts[donor_id])
        winners = int(evidence.donor_winner_counts[donor_id])
        if occurrence == 0 and winners == 0:
            continue
        benefit = float(evidence.donor_reconstruction_errors[donor_id]) * float(
            occurrence + weight * winners
        )
        promotion_candidates.append((-benefit, -winners, -occurrence, donor_id))
    promotion_candidates.sort()

    eviction_candidates: list[tuple[int, float, int, int, float, int, int]] = []
    candidate_target_ids = (
        range(target_count)
        if allow_positive_incidence
        else inventory.eligible_target_ids.tolist()
    )
    for target_id in candidate_target_ids:
        evicted_id = int(mapping[target_id])
        if evicted_id in protected_set:
            continue
        eviction_candidates.append(
            (
                int(inventory.union_non_direct_incidence[target_id]),
                float(inventory.union_non_direct_leverage[target_id]),
                int(evidence.donor_winner_counts[evicted_id]),
                int(evidence.donor_occurrence_counts[evicted_id]),
                float(evidence.donor_eviction_costs[evicted_id]),
                evicted_id,
                int(target_id),
            )
        )
    eviction_candidates.sort()
    if len(promotion_candidates) < count:
        raise Lfm2SemanticCarrierAllocationError(
            "The requested carrier count exceeds demanded non-direct donors."
        )
    if len(eviction_candidates) < count:
        raise Lfm2SemanticCarrierAllocationError(
            "The requested carrier count exceeds eligible unprotected target rows."
        )

    promoted_values = [item[3] for item in promotion_candidates[:count]]
    carrier_values = [item[6] for item in eviction_candidates[:count]]
    evicted_values = [item[5] for item in eviction_candidates[:count]]
    benefit_values = [-item[0] for item in promotion_candidates[:count]]
    cost_values = [item[4] for item in eviction_candidates[:count]]
    carriers = torch.tensor(carrier_values, dtype=torch.int64)
    promoted = torch.tensor(promoted_values, dtype=torch.int64)
    evicted = torch.tensor(evicted_values, dtype=torch.int64)
    benefits = torch.tensor(benefit_values, dtype=torch.float64)
    costs = torch.tensor(cost_values, dtype=torch.float64)
    rerouted_values = set(evicted_values)
    for carrier_id in carrier_values:
        start = int(inventory.union_non_direct_target_offsets[carrier_id])
        stop = int(inventory.union_non_direct_target_offsets[carrier_id + 1])
        rerouted_values.update(
            inventory.union_non_direct_donor_ids[start:stop].tolist()
        )
    rerouted_values.difference_update(promoted_values)
    rerouted = torch.tensor(sorted(rerouted_values), dtype=torch.int64)
    body = {
        "schema": LFM2_SEMANTIC_CARRIER_SELECTION_SCHEMA,
        "algorithm": (
            "fit-demand-times-reconstruction-error-promotion-then-"
            "low-demand-low-refit-cost-eviction-stable-id-ties-v1"
        ),
        "inventory_artifact_sha256": inventory.artifact_sha256,
        "evidence_artifact_sha256": evidence.artifact_sha256,
        "evidence_partition_role": "fold-fit",
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "allocation_count": count,
        "winner_weight": weight,
        "positive_incidence_target_rows_allowed": allow_positive_incidence,
        "protection_artifact_sha256": protection_sha256,
        "protected_donor_ids": protected.tolist(),
        "protected_donor_ids_sha256": _tensor_sha256(protected),
        "base_binding": _plain_json(inventory.receipt["base_binding"]),
        "carrier_target_ids_sha256": _tensor_sha256(carriers),
        "promoted_donor_ids_sha256": _tensor_sha256(promoted),
        "evicted_donor_ids_sha256": _tensor_sha256(evicted),
        "required_rerouted_donor_ids": rerouted.tolist(),
        "required_rerouted_donor_ids_sha256": _tensor_sha256(rerouted),
        "required_rerouted_donor_count": int(rerouted.numel()),
        "rerouted_set_is_reverse_union_plus_evictions_minus_promotions": True,
        "evidence_numeric_contract_sha256": evidence.receipt["numeric_contract_sha256"],
        "promotion_benefit_scores_sha256": _tensor_sha256(benefits),
        "eviction_cost_scores_sha256": _tensor_sha256(costs),
        "selection_rows": [
            {
                "rank": rank,
                "carrier_target_id": int(carriers[rank]),
                "promoted_donor_id": int(promoted[rank]),
                "evicted_donor_id": int(evicted[rank]),
                "union_non_direct_incidence": int(
                    inventory.union_non_direct_incidence[carriers[rank]]
                ),
                "union_non_direct_l1_leverage": float(
                    inventory.union_non_direct_leverage[carriers[rank]]
                ),
                "promotion_benefit_score": float(benefits[rank]),
                "eviction_cost_score": float(costs[rank]),
            }
            for rank in range(count)
        ],
        "stable_tie_breaks": [
            "promotion: benefit-score-desc, winner-count-desc, "
            "occurrence-count-desc, donor-id-asc",
            "eviction: unique-union-incidence-asc, union-l1-leverage-asc, "
            "winner-count-asc, occurrence-count-asc, refit-cost-asc, "
            "donor-id-asc, target-id-asc",
        ],
        "review_or_evaluation_evidence_allowed": False,
        "operation_audit": _ZERO_OPERATION_AUDIT,
    }
    return Lfm2SemanticCarrierSelection(
        carriers,
        promoted,
        evicted,
        rerouted,
        benefits,
        costs,
        _sealed(body, "artifact_sha256"),
    )


def _build_replacement_solver_receipt(
    *,
    channel: str,
    source_artifact_sha256: str,
    post_replant_target_table_sha256: str,
    rerouted_donor_ids: Tensor,
    offsets: Tensor,
    target_ids: Tensor,
    coefficients: Tensor,
    solver_math_abi_sha256: str,
    numeric_contract_sha256: str,
) -> dict[str, Any]:
    support_binding = {
        "offsets_sha256": _tensor_sha256(offsets),
        "target_ids_sha256": _tensor_sha256(target_ids),
    }
    body = {
        "schema": LFM2_SEMANTIC_CARRIER_SOLVER_RECEIPT_SCHEMA,
        "algorithm": "post-replant-complete-local-csr-reclosure-v1",
        "channel": channel,
        "source_artifact_sha256": _validated_sha256(
            source_artifact_sha256, f"{channel} source_artifact_sha256"
        ),
        "post_replant_target_table_sha256": _validated_sha256(
            post_replant_target_table_sha256,
            "post_replant_target_table_sha256",
        ),
        "rerouted_donor_ids": rerouted_donor_ids.tolist(),
        "rerouted_donor_ids_sha256": _tensor_sha256(rerouted_donor_ids),
        "rerouted_donor_count": int(rerouted_donor_ids.numel()),
        "support_binding": support_binding,
        "support_binding_sha256": canonical_sha256(support_binding),
        "offsets_sha256": _tensor_sha256(offsets),
        "target_ids_sha256": _tensor_sha256(target_ids),
        "coefficients_sha256": _tensor_sha256(coefficients),
        "coefficient_count": int(coefficients.numel()),
        "solver_math_abi_sha256": _validated_sha256(
            solver_math_abi_sha256, f"{channel} solver_math_abi_sha256"
        ),
        "numeric_contract_sha256": _validated_sha256(
            numeric_contract_sha256, "numeric_contract_sha256"
        ),
        "routes_are_complete_for_exact_rerouted_set": True,
        "routes_were_solved_against_post_replant_target_table": True,
        "operation_audit": _ZERO_OPERATION_AUDIT,
    }
    return _sealed(body, "artifact_sha256")


def _validate_replacement_solver_receipt(
    value: Mapping[str, Any],
    *,
    channel: str,
    post_replant_target_table_sha256: str,
    rerouted_donor_ids: Tensor,
    offsets: Tensor,
    target_ids: Tensor,
    coefficients: Tensor,
    numeric_contract_sha256: str,
) -> dict[str, Any]:
    receipt = _verify_sealed(
        value,
        schema=LFM2_SEMANTIC_CARRIER_SOLVER_RECEIPT_SCHEMA,
        field="artifact_sha256",
        name=f"{channel} replacement solver receipt",
    )
    _validated_sha256(
        receipt.get("source_artifact_sha256"),
        f"{channel} source_artifact_sha256",
    )
    _validated_sha256(
        receipt.get("solver_math_abi_sha256"),
        f"{channel} solver_math_abi_sha256",
    )
    expected_support = {
        "offsets_sha256": _tensor_sha256(offsets),
        "target_ids_sha256": _tensor_sha256(target_ids),
    }
    if (
        receipt.get("channel") != channel
        or receipt.get("post_replant_target_table_sha256")
        != post_replant_target_table_sha256
        or receipt.get("rerouted_donor_ids") != rerouted_donor_ids.tolist()
        or receipt.get("rerouted_donor_ids_sha256")
        != _tensor_sha256(rerouted_donor_ids)
        or receipt.get("rerouted_donor_count") != int(rerouted_donor_ids.numel())
        or receipt.get("support_binding") != expected_support
        or receipt.get("support_binding_sha256") != canonical_sha256(expected_support)
        or receipt.get("offsets_sha256") != _tensor_sha256(offsets)
        or receipt.get("target_ids_sha256") != _tensor_sha256(target_ids)
        or receipt.get("coefficients_sha256") != _tensor_sha256(coefficients)
        or receipt.get("coefficient_count") != int(coefficients.numel())
        or receipt.get("numeric_contract_sha256") != numeric_contract_sha256
        or receipt.get("routes_are_complete_for_exact_rerouted_set") is not True
        or receipt.get("routes_were_solved_against_post_replant_target_table")
        is not True
        or receipt.get("operation_audit") != _ZERO_OPERATION_AUDIT
    ):
        raise Lfm2SemanticCarrierAllocationError(
            f"{channel} replacement solver receipt binding is invalid."
        )
    return receipt


@dataclass(frozen=True, slots=True)
class Lfm2SemanticCarrierAllocation:
    carrier_target_ids: Tensor
    promoted_donor_ids: Tensor
    evicted_donor_ids: Tensor
    required_rerouted_donor_ids: Tensor
    replacement_target_rows: Tensor
    input_evicted_offsets: Tensor
    input_evicted_target_ids: Tensor
    input_evicted_coefficients: Tensor
    output_evicted_offsets: Tensor
    output_evicted_target_ids: Tensor
    output_evicted_coefficients: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SEMANTIC_CARRIER_ALLOCATION_SCHEMA,
            field="artifact_sha256",
            name="carrier allocation",
        )
        count = _positive_int(value.get("allocation_count"), "allocation_count")
        donor_count = _positive_int(value.get("donor_row_count"), "donor_row_count")
        target_count = _positive_int(value.get("target_row_count"), "target_row_count")
        width = _positive_int(value.get("embedding_width"), "embedding_width")
        _validated_sha256(
            value.get("selection_artifact_sha256"), "selection_artifact_sha256"
        )
        _validated_sha256(
            value.get("inventory_artifact_sha256"), "inventory_artifact_sha256"
        )
        _validated_sha256(
            value.get("base_target_table_sha256"), "base_target_table_sha256"
        )
        _validated_sha256(value.get("parent_table_sha256"), "parent_table_sha256")
        post_replant_target_table_sha256 = _validated_sha256(
            value.get("post_replant_target_table_sha256"),
            "post_replant_target_table_sha256",
        )
        numeric_contract_sha256 = _validated_sha256(
            value.get("numeric_contract_sha256"), "numeric_contract_sha256"
        )
        _validated_sha256(
            value.get("input_replacement_artifact_sha256"),
            "input_replacement_artifact_sha256",
        )
        _validated_sha256(
            value.get("output_replacement_artifact_sha256"),
            "output_replacement_artifact_sha256",
        )
        if not isinstance(value.get("base_binding"), Mapping):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier allocation base_binding must be a mapping."
            )
        carriers = _validated_int64_vector(
            self.carrier_target_ids, "carrier_target_ids"
        )
        promoted = _validated_int64_vector(
            self.promoted_donor_ids, "promoted_donor_ids"
        )
        evicted = _validated_int64_vector(self.evicted_donor_ids, "evicted_donor_ids")
        rerouted = _validated_int64_vector(
            self.required_rerouted_donor_ids, "required_rerouted_donor_ids"
        )
        rows = _validated_table(self.replacement_target_rows, "replacement_target_rows")
        rerouted_count = _positive_int(
            value.get("required_rerouted_donor_count"),
            "required_rerouted_donor_count",
        )
        input_routes = _validated_csr(
            self.input_evicted_offsets,
            self.input_evicted_target_ids,
            self.input_evicted_coefficients,
            row_count=rerouted_count,
            target_row_count=target_count,
            name="input evicted routes",
        )
        output_routes = _validated_csr(
            self.output_evicted_offsets,
            self.output_evicted_target_ids,
            self.output_evicted_coefficients,
            row_count=rerouted_count,
            target_row_count=target_count,
            name="output evicted routes",
        )
        if (
            tuple(rows.shape) != (count, width)
            or _dtype_name(rows) != value.get("target_table_dtype")
            or any(
                int(tensor.numel()) != count for tensor in (carriers, promoted, evicted)
            )
            or int(torch.unique(carriers).numel()) != count
            or int(torch.unique(promoted).numel()) != count
            or int(torch.unique(evicted).numel()) != count
            or int(rerouted.numel()) != rerouted_count
            or not torch.equal(rerouted, torch.unique(rerouted, sorted=True))
            or not bool(torch.isin(evicted, rerouted).all())
            or bool(torch.isin(rerouted, promoted).any())
            or bool(torch.isin(promoted, evicted).any())
            or bool((carriers < 0).any())
            or bool((carriers >= target_count).any())
            or bool((promoted < 0).any())
            or bool((promoted >= donor_count).any())
            or bool((evicted < 0).any())
            or bool((evicted >= donor_count).any())
            or bool((rerouted < 0).any())
            or bool((rerouted >= donor_count).any())
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier allocation tensor shapes, dtypes, or IDs are invalid."
            )
        tensors = {
            "carrier_target_ids_sha256": carriers,
            "promoted_donor_ids_sha256": promoted,
            "evicted_donor_ids_sha256": evicted,
            "required_rerouted_donor_ids_sha256": rerouted,
            "replacement_target_rows_sha256": rows,
            "input_evicted_offsets_sha256": input_routes[0],
            "input_evicted_target_ids_sha256": input_routes[1],
            "input_evicted_coefficients_sha256": input_routes[2],
            "output_evicted_offsets_sha256": output_routes[0],
            "output_evicted_target_ids_sha256": output_routes[1],
            "output_evicted_coefficients_sha256": output_routes[2],
        }
        input_solver_receipt_value = value.get("input_replacement_solver_receipt")
        output_solver_receipt_value = value.get("output_replacement_solver_receipt")
        if not isinstance(input_solver_receipt_value, Mapping) or not isinstance(
            output_solver_receipt_value, Mapping
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier allocation requires independent input and output solver receipts."
            )
        input_solver_receipt = _validate_replacement_solver_receipt(
            input_solver_receipt_value,
            channel="input",
            post_replant_target_table_sha256=post_replant_target_table_sha256,
            rerouted_donor_ids=rerouted,
            offsets=input_routes[0],
            target_ids=input_routes[1],
            coefficients=input_routes[2],
            numeric_contract_sha256=numeric_contract_sha256,
        )
        output_solver_receipt = _validate_replacement_solver_receipt(
            output_solver_receipt_value,
            channel="output",
            post_replant_target_table_sha256=post_replant_target_table_sha256,
            rerouted_donor_ids=rerouted,
            offsets=output_routes[0],
            target_ids=output_routes[1],
            coefficients=output_routes[2],
            numeric_contract_sha256=numeric_contract_sha256,
        )
        evicted_set = set(evicted.tolist())
        evicted_coefficient_count = sum(
            int(input_routes[0][row_index + 1] - input_routes[0][row_index])
            + int(output_routes[0][row_index + 1] - output_routes[0][row_index])
            for row_index, donor_id in enumerate(rerouted.tolist())
            if donor_id in evicted_set
        )
        rerouted_coefficient_count = int(
            input_routes[2].numel() + output_routes[2].numel()
        )
        if (
            any(
                value.get(key) != _tensor_sha256(tensor)
                for key, tensor in tensors.items()
            )
            or value.get("stored_evicted_coefficient_count")
            != evicted_coefficient_count
            or value.get("stored_evicted_coefficient_bytes")
            != evicted_coefficient_count * 4
            or value.get("stored_rerouted_coefficient_count")
            != rerouted_coefficient_count
            or value.get("stored_rerouted_coefficient_bytes")
            != rerouted_coefficient_count * 4
            or value.get("replacement_target_parameter_count") != int(rows.numel())
            or value.get("target_table_replacement_parameter_addition") != 0
            or value.get("target_table_dtype_is_preserved") is not True
            or value.get("replacement_target_rows_replace_existing_parameters")
            is not True
            or value.get("promoted_input_and_output_routes_are_implicit_exact_one_hot")
            is not True
            or value.get("evicted_input_and_output_routes_are_independent") is not True
            or value.get("replacement_routes_cover_exact_required_rerouted_set")
            is not True
            or value.get("required_rerouted_donor_ids") != rerouted.tolist()
            or value.get("input_replacement_solver_receipt_sha256")
            != input_solver_receipt["artifact_sha256"]
            or value.get("output_replacement_solver_receipt_sha256")
            != output_solver_receipt["artifact_sha256"]
            or input_solver_receipt.get("source_artifact_sha256")
            != value.get("input_replacement_artifact_sha256")
            or output_solver_receipt.get("source_artifact_sha256")
            != value.get("output_replacement_artifact_sha256")
            or not isinstance(value.get("positive_incidence_target_rows_allowed"), bool)
            or value.get("indices_are_cpu_int64") is not True
            or value.get("coefficients_are_cpu_float32") is not True
            or value.get("operation_audit") != _ZERO_OPERATION_AUDIT
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier allocation tensor binding is invalid."
            )
        object.__setattr__(self, "carrier_target_ids", carriers)
        object.__setattr__(self, "promoted_donor_ids", promoted)
        object.__setattr__(self, "evicted_donor_ids", evicted)
        object.__setattr__(self, "required_rerouted_donor_ids", rerouted)
        object.__setattr__(self, "replacement_target_rows", rows)
        object.__setattr__(self, "input_evicted_offsets", input_routes[0])
        object.__setattr__(self, "input_evicted_target_ids", input_routes[1])
        object.__setattr__(self, "input_evicted_coefficients", input_routes[2])
        object.__setattr__(self, "output_evicted_offsets", output_routes[0])
        object.__setattr__(self, "output_evicted_target_ids", output_routes[1])
        object.__setattr__(self, "output_evicted_coefficients", output_routes[2])
        object.__setattr__(self, "receipt", _deep_freeze(value))

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def compile_lfm2_semantic_carrier_allocation(
    selection: Lfm2SemanticCarrierSelection,
    parent_table: Tensor,
    target_table: Tensor,
    *,
    input_evicted_offsets: Tensor,
    input_evicted_target_ids: Tensor,
    input_evicted_coefficients: Tensor,
    output_evicted_offsets: Tensor,
    output_evicted_target_ids: Tensor,
    output_evicted_coefficients: Tensor,
    input_replacement_artifact_sha256: str,
    output_replacement_artifact_sha256: str,
    input_solver_math_abi_sha256: str | None = None,
    output_solver_math_abi_sha256: str | None = None,
    numeric_contract_sha256: str | None = None,
) -> Lfm2SemanticCarrierAllocation:
    """Compile a sealed delta after local replacement routes have been solved."""

    if not isinstance(selection, Lfm2SemanticCarrierSelection):
        raise Lfm2SemanticCarrierAllocationError(
            "selection must be a validated carrier selection."
        )
    parent = _validated_table(parent_table, "parent_table", copy=False)
    target = _validated_table(target_table, "target_table", copy=False)
    donor_count = int(selection.receipt["donor_row_count"])
    target_count = int(selection.receipt["target_row_count"])
    count = int(selection.receipt["allocation_count"])
    rerouted = selection.required_rerouted_donor_ids
    rerouted_count = int(rerouted.numel())
    if (
        int(parent.shape[0]) != donor_count
        or int(target.shape[0]) != target_count
        or int(parent.shape[1]) != int(target.shape[1])
    ):
        raise Lfm2SemanticCarrierAllocationError(
            "Parent and target tables do not match the selected donor/target ABI."
        )
    input_routes = _validated_csr(
        input_evicted_offsets,
        input_evicted_target_ids,
        input_evicted_coefficients,
        row_count=rerouted_count,
        target_row_count=target_count,
        name="input evicted routes",
    )
    output_routes = _validated_csr(
        output_evicted_offsets,
        output_evicted_target_ids,
        output_evicted_coefficients,
        row_count=rerouted_count,
        target_row_count=target_count,
        name="output evicted routes",
    )
    replacement_rows = (
        parent.index_select(0, selection.promoted_donor_ids.to(device=parent.device))
        .to(device=target.device, dtype=target.dtype)
        .clone()
        .contiguous()
    )
    if not bool(torch.isfinite(replacement_rows).all()):
        raise Lfm2SemanticCarrierAllocationError(
            "Promoted parent rows became non-finite at the target table dtype."
        )
    post_replant_table = target.clone()
    post_replant_table.index_copy_(
        0,
        selection.carrier_target_ids.to(device=post_replant_table.device),
        replacement_rows,
    )
    post_replant_target_table_sha256 = _tensor_sha256(post_replant_table)
    input_source_sha256 = _validated_sha256(
        input_replacement_artifact_sha256,
        "input_replacement_artifact_sha256",
    )
    output_source_sha256 = _validated_sha256(
        output_replacement_artifact_sha256,
        "output_replacement_artifact_sha256",
    )
    numeric_sha256 = _validated_sha256(
        numeric_contract_sha256
        if numeric_contract_sha256 is not None
        else selection.receipt["evidence_numeric_contract_sha256"],
        "numeric_contract_sha256",
    )
    input_math_sha256 = _validated_sha256(
        input_solver_math_abi_sha256
        if input_solver_math_abi_sha256 is not None
        else input_source_sha256,
        "input_solver_math_abi_sha256",
    )
    output_math_sha256 = _validated_sha256(
        output_solver_math_abi_sha256
        if output_solver_math_abi_sha256 is not None
        else output_source_sha256,
        "output_solver_math_abi_sha256",
    )
    input_solver_receipt = _build_replacement_solver_receipt(
        channel="input",
        source_artifact_sha256=input_source_sha256,
        post_replant_target_table_sha256=post_replant_target_table_sha256,
        rerouted_donor_ids=rerouted,
        offsets=input_routes[0],
        target_ids=input_routes[1],
        coefficients=input_routes[2],
        solver_math_abi_sha256=input_math_sha256,
        numeric_contract_sha256=numeric_sha256,
    )
    output_solver_receipt = _build_replacement_solver_receipt(
        channel="output",
        source_artifact_sha256=output_source_sha256,
        post_replant_target_table_sha256=post_replant_target_table_sha256,
        rerouted_donor_ids=rerouted,
        offsets=output_routes[0],
        target_ids=output_routes[1],
        coefficients=output_routes[2],
        solver_math_abi_sha256=output_math_sha256,
        numeric_contract_sha256=numeric_sha256,
    )
    evicted_set = set(selection.evicted_donor_ids.tolist())
    evicted_coefficient_count = sum(
        int(input_routes[0][row_index + 1] - input_routes[0][row_index])
        + int(output_routes[0][row_index + 1] - output_routes[0][row_index])
        for row_index, donor_id in enumerate(rerouted.tolist())
        if donor_id in evicted_set
    )
    rerouted_coefficient_count = int(input_routes[2].numel() + output_routes[2].numel())
    body = {
        "schema": LFM2_SEMANTIC_CARRIER_ALLOCATION_SCHEMA,
        "algorithm": "authenticated-carrier-eviction-promotion-delta-v1",
        "selection_artifact_sha256": selection.artifact_sha256,
        "inventory_artifact_sha256": selection.receipt["inventory_artifact_sha256"],
        "base_binding": _plain_json(selection.receipt["base_binding"]),
        "base_target_table_sha256": _tensor_sha256(target),
        "post_replant_target_table_sha256": post_replant_target_table_sha256,
        "parent_table_sha256": _tensor_sha256(parent),
        "parent_table_dtype": _dtype_name(parent),
        "target_table_dtype": _dtype_name(target),
        "input_replacement_artifact_sha256": input_source_sha256,
        "output_replacement_artifact_sha256": output_source_sha256,
        "numeric_contract_sha256": numeric_sha256,
        "input_replacement_solver_receipt": input_solver_receipt,
        "output_replacement_solver_receipt": output_solver_receipt,
        "input_replacement_solver_receipt_sha256": input_solver_receipt[
            "artifact_sha256"
        ],
        "output_replacement_solver_receipt_sha256": output_solver_receipt[
            "artifact_sha256"
        ],
        "target_table_dtype_is_preserved": True,
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "embedding_width": int(target.shape[1]),
        "allocation_count": count,
        "positive_incidence_target_rows_allowed": selection.receipt[
            "positive_incidence_target_rows_allowed"
        ],
        "carrier_target_ids_sha256": _tensor_sha256(selection.carrier_target_ids),
        "promoted_donor_ids_sha256": _tensor_sha256(selection.promoted_donor_ids),
        "evicted_donor_ids_sha256": _tensor_sha256(selection.evicted_donor_ids),
        "required_rerouted_donor_ids": rerouted.tolist(),
        "required_rerouted_donor_ids_sha256": _tensor_sha256(rerouted),
        "required_rerouted_donor_count": rerouted_count,
        "replacement_target_rows_sha256": _tensor_sha256(replacement_rows),
        "input_evicted_offsets_sha256": _tensor_sha256(input_routes[0]),
        "input_evicted_target_ids_sha256": _tensor_sha256(input_routes[1]),
        "input_evicted_coefficients_sha256": _tensor_sha256(input_routes[2]),
        "output_evicted_offsets_sha256": _tensor_sha256(output_routes[0]),
        "output_evicted_target_ids_sha256": _tensor_sha256(output_routes[1]),
        "output_evicted_coefficients_sha256": _tensor_sha256(output_routes[2]),
        "stored_evicted_coefficient_count": evicted_coefficient_count,
        "stored_evicted_coefficient_bytes": evicted_coefficient_count * 4,
        "stored_rerouted_coefficient_count": rerouted_coefficient_count,
        "stored_rerouted_coefficient_bytes": rerouted_coefficient_count * 4,
        "replacement_target_parameter_count": int(replacement_rows.numel()),
        "target_table_replacement_parameter_addition": 0,
        "replacement_target_rows_replace_existing_parameters": True,
        "promoted_input_and_output_routes_are_implicit_exact_one_hot": True,
        "evicted_input_and_output_routes_are_independent": True,
        "replacement_routes_cover_exact_required_rerouted_set": True,
        "indices_are_cpu_int64": True,
        "coefficients_are_cpu_float32": True,
        "operation_audit": _ZERO_OPERATION_AUDIT,
    }
    return Lfm2SemanticCarrierAllocation(
        selection.carrier_target_ids,
        selection.promoted_donor_ids,
        selection.evicted_donor_ids,
        rerouted,
        replacement_rows,
        input_routes[0],
        input_routes[1],
        input_routes[2],
        output_routes[0],
        output_routes[1],
        output_routes[2],
        _sealed(body, "artifact_sha256"),
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticCarrierMaterialization:
    target_table: Tensor
    target_to_donor: Tensor
    input_offsets: Tensor
    input_target_ids: Tensor
    input_coefficients: Tensor
    output_offsets: Tensor
    output_target_ids: Tensor
    output_coefficients: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _verify_sealed(
            self.receipt,
            schema=LFM2_SEMANTIC_CARRIER_MATERIALIZATION_SCHEMA,
            field="artifact_sha256",
            name="carrier materialization",
        )
        table = _validated_table(self.target_table, "target_table")
        donor_count = _positive_int(value.get("donor_row_count"), "donor_row_count")
        target_count = _positive_int(value.get("target_row_count"), "target_row_count")
        _validated_sha256(
            value.get("allocation_artifact_sha256"), "allocation_artifact_sha256"
        )
        mapping = _validated_mapping(
            self.target_to_donor,
            target_row_count=target_count,
            donor_row_count=donor_count,
        )
        input_code = _validated_csr(
            self.input_offsets,
            self.input_target_ids,
            self.input_coefficients,
            row_count=donor_count,
            target_row_count=target_count,
            name="materialized input semantic code",
        )
        output_code = _validated_csr(
            self.output_offsets,
            self.output_target_ids,
            self.output_coefficients,
            row_count=donor_count,
            target_row_count=target_count,
            name="materialized output semantic code",
        )
        _require_exact_direct_rows(*input_code, mapping, name="materialized input code")
        _require_exact_direct_rows(
            *output_code, mapping, name="materialized output code"
        )
        tensors = {
            "target_table_sha256": table,
            "target_to_donor_sha256": mapping,
            "input_offsets_sha256": input_code[0],
            "input_target_ids_sha256": input_code[1],
            "input_coefficients_sha256": input_code[2],
            "output_offsets_sha256": output_code[0],
            "output_target_ids_sha256": output_code[1],
            "output_coefficients_sha256": output_code[2],
        }
        coefficient_count = int(input_code[2].numel() + output_code[2].numel())
        rerouted_count = _positive_int(
            value.get("required_rerouted_donor_count"),
            "required_rerouted_donor_count",
        )
        _validated_sha256(
            value.get("required_rerouted_donor_ids_sha256"),
            "required_rerouted_donor_ids_sha256",
        )
        if (
            tuple(table.shape)
            != (
                target_count,
                _positive_int(value.get("embedding_width"), "embedding_width"),
            )
            or _dtype_name(table) != value.get("target_table_dtype")
            or any(
                value.get(key) != _tensor_sha256(tensor)
                for key, tensor in tensors.items()
            )
            or value.get("continuous_fold_coefficient_count") != coefficient_count
            or value.get("continuous_fold_coefficient_bytes") != coefficient_count * 4
            or value.get("physical_target_table_parameter_count") != int(table.numel())
            or value.get("target_table_replacement_parameter_addition") != 0
            or value.get("effective_child_parameter_addition_from_semantic_bridge")
            != coefficient_count
            or value.get("complete_materialized_continuous_state_count")
            != int(table.numel()) + coefficient_count
            or value.get("target_table_counted_once_in_complete_continuous_state")
            is not True
            or value.get("input_and_output_coefficients_counted_independently")
            is not True
            or rerouted_count > donor_count
            or value.get("target_table_dtype_preserved") is not True
            or value.get("integer_routing_values_are_structural_metadata") is not True
            or value.get("indices_are_cpu_int64") is not True
            or value.get("coefficients_are_cpu_float32") is not True
            or value.get("every_final_direct_donor_is_exact_one_hot_in_both_routes")
            is not True
            or value.get("unaffected_target_rows_are_byte_identical") is not True
            or value.get("unaffected_input_rows_are_byte_identical") is not True
            or value.get("unaffected_output_rows_are_byte_identical") is not True
            or value.get("mapping_is_injective") is not True
            or value.get("operation_audit") != _ZERO_OPERATION_AUDIT
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Carrier materialization tensor binding or state count is invalid."
            )
        object.__setattr__(self, "target_table", table)
        object.__setattr__(self, "target_to_donor", mapping)
        object.__setattr__(self, "input_offsets", input_code[0])
        object.__setattr__(self, "input_target_ids", input_code[1])
        object.__setattr__(self, "input_coefficients", input_code[2])
        object.__setattr__(self, "output_offsets", output_code[0])
        object.__setattr__(self, "output_target_ids", output_code[1])
        object.__setattr__(self, "output_coefficients", output_code[2])
        object.__setattr__(self, "receipt", _deep_freeze(value))

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def _replacement_rows_by_donor(
    *,
    donor_ids: Tensor,
    offsets: Tensor,
    target_ids: Tensor,
    coefficients: Tensor,
) -> dict[int, tuple[Tensor, Tensor]]:
    rows: dict[int, tuple[Tensor, Tensor]] = {}
    for row_index, donor_id in enumerate(donor_ids.tolist()):
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        rows[int(donor_id)] = (
            target_ids[start:stop].clone(),
            coefficients[start:stop].clone(),
        )
    return rows


def _materialize_route(
    *,
    base_offsets: Tensor,
    base_target_ids: Tensor,
    base_coefficients: Tensor,
    carrier_target_ids: Tensor,
    promoted_donor_ids: Tensor,
    evicted_donor_ids: Tensor,
    evicted_offsets: Tensor,
    evicted_target_ids: Tensor,
    evicted_coefficients: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    promoted = {
        int(donor_id): int(carrier_id)
        for donor_id, carrier_id in zip(
            promoted_donor_ids.tolist(), carrier_target_ids.tolist(), strict=True
        )
    }
    replacements = _replacement_rows_by_donor(
        donor_ids=evicted_donor_ids,
        offsets=evicted_offsets,
        target_ids=evicted_target_ids,
        coefficients=evicted_coefficients,
    )
    donor_count = int(base_offsets.numel()) - 1
    new_offsets = [0]
    new_ids: list[Tensor] = []
    new_coefficients: list[Tensor] = []
    for donor_id in range(donor_count):
        if donor_id in promoted:
            ids = torch.tensor([promoted[donor_id]], dtype=torch.int64)
            coefficients = torch.tensor([1.0], dtype=torch.float32)
        elif donor_id in replacements:
            ids, coefficients = replacements[donor_id]
        else:
            start = int(base_offsets[donor_id])
            stop = int(base_offsets[donor_id + 1])
            ids = base_target_ids[start:stop].clone()
            coefficients = base_coefficients[start:stop].clone()
        new_ids.append(ids)
        new_coefficients.append(coefficients)
        new_offsets.append(new_offsets[-1] + int(ids.numel()))
    return (
        torch.tensor(new_offsets, dtype=torch.int64),
        torch.cat(new_ids),
        torch.cat(new_coefficients),
    )


def materialize_lfm2_semantic_carrier_allocation(
    allocation: Lfm2SemanticCarrierAllocation,
    target_table: Tensor,
    *,
    target_to_donor: Tensor,
    input_offsets: Tensor,
    input_target_ids: Tensor,
    input_coefficients: Tensor,
    output_offsets: Tensor,
    output_target_ids: Tensor,
    output_coefficients: Tensor,
) -> Lfm2SemanticCarrierMaterialization:
    """Apply a validated carrier delta and republish complete canonical CSR routes."""

    if not isinstance(allocation, Lfm2SemanticCarrierAllocation):
        raise Lfm2SemanticCarrierAllocationError(
            "allocation must be a validated carrier allocation."
        )
    base_table = _validated_table(target_table, "target_table", copy=False)
    (
        base_input_offsets,
        base_input_ids,
        base_input_coefficients,
        base_output_offsets,
        base_output_ids,
        base_output_coefficients,
        base_mapping,
    ) = _validated_base_codes(
        input_offsets=input_offsets,
        input_target_ids=input_target_ids,
        input_coefficients=input_coefficients,
        output_offsets=output_offsets,
        output_target_ids=output_target_ids,
        output_coefficients=output_coefficients,
        target_to_donor=target_to_donor,
    )
    donor_count = int(base_input_offsets.numel()) - 1
    target_count = int(base_mapping.numel())
    if (
        donor_count != int(allocation.receipt["donor_row_count"])
        or target_count != int(allocation.receipt["target_row_count"])
        or tuple(base_table.shape)
        != (target_count, int(allocation.receipt["embedding_width"]))
        or _dtype_name(base_table) != allocation.receipt["target_table_dtype"]
        or _tensor_sha256(base_table) != allocation.receipt["base_target_table_sha256"]
    ):
        raise Lfm2SemanticCarrierAllocationError(
            "The base target table does not match the carrier allocation."
        )
    actual_binding = _base_binding(
        base_input_offsets,
        base_input_ids,
        base_input_coefficients,
        base_output_offsets,
        base_output_ids,
        base_output_coefficients,
        base_mapping,
    )
    if actual_binding != _plain_json(allocation.receipt["base_binding"]):
        raise Lfm2SemanticCarrierAllocationError(
            "The base semantic routes do not match the carrier allocation."
        )
    base_inventory = build_lfm2_semantic_carrier_inventory(
        input_offsets=base_input_offsets,
        input_target_ids=base_input_ids,
        input_coefficients=base_input_coefficients,
        output_offsets=base_output_offsets,
        output_target_ids=base_output_ids,
        output_coefficients=base_output_coefficients,
        target_to_donor=base_mapping,
    )
    if (
        base_inventory.artifact_sha256
        != allocation.receipt["inventory_artifact_sha256"]
    ):
        raise Lfm2SemanticCarrierAllocationError(
            "The recomputed carrier inventory does not match the allocation."
        )

    carriers = allocation.carrier_target_ids
    promoted = allocation.promoted_donor_ids
    evicted = allocation.evicted_donor_ids
    rerouted = allocation.required_rerouted_donor_ids
    direct_set = set(base_mapping.tolist())
    eligible_set = set(base_inventory.eligible_target_ids.tolist())
    for carrier_id, promoted_id, evicted_id in zip(
        carriers.tolist(), promoted.tolist(), evicted.tolist(), strict=True
    ):
        if (
            not allocation.receipt["positive_incidence_target_rows_allowed"]
            and carrier_id not in eligible_set
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "Each carrier must have zero non-direct incidence and leverage "
                "across both semantic routes."
            )
        if int(base_mapping[carrier_id]) != evicted_id:
            raise Lfm2SemanticCarrierAllocationError(
                "Each carrier must evict the donor protected by that target row."
            )
        if promoted_id in direct_set:
            raise Lfm2SemanticCarrierAllocationError(
                "Promoted donors must be non-direct in the base mapping."
            )

    expected_rerouted = set(evicted.tolist())
    for carrier_id in carriers.tolist():
        start = int(base_inventory.union_non_direct_target_offsets[carrier_id])
        stop = int(base_inventory.union_non_direct_target_offsets[carrier_id + 1])
        expected_rerouted.update(
            base_inventory.union_non_direct_donor_ids[start:stop].tolist()
        )
    expected_rerouted.difference_update(promoted.tolist())
    if rerouted.tolist() != sorted(expected_rerouted):
        raise Lfm2SemanticCarrierAllocationError(
            "The allocation omits or adds donors relative to authenticated reverse "
            "C_in/C_out incidence."
        )

    new_table = base_table.clone()
    new_table.index_copy_(
        0,
        carriers.to(device=new_table.device),
        allocation.replacement_target_rows.to(device=new_table.device),
    )
    if new_table.dtype != base_table.dtype:
        raise Lfm2SemanticCarrierAllocationError(
            "Carrier materialization changed the target table dtype."
        )
    if (
        _tensor_sha256(new_table)
        != allocation.receipt["post_replant_target_table_sha256"]
    ):
        raise Lfm2SemanticCarrierAllocationError(
            "The materialized target table does not match the solver-bound "
            "post-replant table."
        )
    new_mapping = base_mapping.clone()
    new_mapping[carriers] = promoted
    if int(torch.unique(new_mapping).numel()) != target_count:
        raise Lfm2SemanticCarrierAllocationError(
            "Carrier promotion did not produce an injective mapping."
        )
    new_input = _materialize_route(
        base_offsets=base_input_offsets,
        base_target_ids=base_input_ids,
        base_coefficients=base_input_coefficients,
        carrier_target_ids=carriers,
        promoted_donor_ids=promoted,
        evicted_donor_ids=rerouted,
        evicted_offsets=allocation.input_evicted_offsets,
        evicted_target_ids=allocation.input_evicted_target_ids,
        evicted_coefficients=allocation.input_evicted_coefficients,
    )
    new_output = _materialize_route(
        base_offsets=base_output_offsets,
        base_target_ids=base_output_ids,
        base_coefficients=base_output_coefficients,
        carrier_target_ids=carriers,
        promoted_donor_ids=promoted,
        evicted_donor_ids=rerouted,
        evicted_offsets=allocation.output_evicted_offsets,
        evicted_target_ids=allocation.output_evicted_target_ids,
        evicted_coefficients=allocation.output_evicted_coefficients,
    )
    _validated_csr(
        *new_input,
        row_count=donor_count,
        target_row_count=target_count,
        name="materialized input semantic code",
    )
    _validated_csr(
        *new_output,
        row_count=donor_count,
        target_row_count=target_count,
        name="materialized output semantic code",
    )
    _require_exact_direct_rows(*new_input, new_mapping, name="materialized input code")
    _require_exact_direct_rows(
        *new_output, new_mapping, name="materialized output code"
    )

    changed_donors = set(promoted.tolist()) | set(rerouted.tolist())
    for donor_id in range(donor_count):
        if donor_id in changed_donors:
            continue
        for base_route, new_route in (
            (
                (base_input_offsets, base_input_ids, base_input_coefficients),
                new_input,
            ),
            (
                (base_output_offsets, base_output_ids, base_output_coefficients),
                new_output,
            ),
        ):
            base_start = int(base_route[0][donor_id])
            base_stop = int(base_route[0][donor_id + 1])
            new_start = int(new_route[0][donor_id])
            new_stop = int(new_route[0][donor_id + 1])
            if not _byte_equal(
                base_route[1][base_start:base_stop],
                new_route[1][new_start:new_stop],
            ) or not _byte_equal(
                base_route[2][base_start:base_stop],
                new_route[2][new_start:new_stop],
            ):
                raise Lfm2SemanticCarrierAllocationError(
                    "An unaffected semantic row changed during materialization."
                )
    changed_targets = set(carriers.tolist())
    for target_id in range(target_count):
        if target_id not in changed_targets and not _byte_equal(
            base_table[target_id], new_table[target_id]
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "An unaffected target-table row changed during materialization."
            )
        if target_id not in changed_targets and int(base_mapping[target_id]) != int(
            new_mapping[target_id]
        ):
            raise Lfm2SemanticCarrierAllocationError(
                "An unaffected direct mapping changed during materialization."
            )

    coefficient_count = int(new_input[2].numel() + new_output[2].numel())
    body = {
        "schema": LFM2_SEMANTIC_CARRIER_MATERIALIZATION_SCHEMA,
        "algorithm": "pure-complete-csr-carrier-materialization-v1",
        "allocation_artifact_sha256": allocation.artifact_sha256,
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "embedding_width": int(new_table.shape[1]),
        "target_table_dtype": _dtype_name(new_table),
        "target_table_dtype_preserved": True,
        "target_table_sha256": _tensor_sha256(new_table),
        "target_to_donor_sha256": _tensor_sha256(new_mapping),
        "input_offsets_sha256": _tensor_sha256(new_input[0]),
        "input_target_ids_sha256": _tensor_sha256(new_input[1]),
        "input_coefficients_sha256": _tensor_sha256(new_input[2]),
        "output_offsets_sha256": _tensor_sha256(new_output[0]),
        "output_target_ids_sha256": _tensor_sha256(new_output[1]),
        "output_coefficients_sha256": _tensor_sha256(new_output[2]),
        "continuous_fold_coefficient_count": coefficient_count,
        "continuous_fold_coefficient_bytes": coefficient_count * 4,
        "physical_target_table_parameter_count": int(new_table.numel()),
        "target_table_replacement_parameter_addition": 0,
        "effective_child_parameter_addition_from_semantic_bridge": coefficient_count,
        "complete_materialized_continuous_state_count": int(new_table.numel())
        + coefficient_count,
        "target_table_counted_once_in_complete_continuous_state": True,
        "input_and_output_coefficients_counted_independently": True,
        "required_rerouted_donor_count": int(rerouted.numel()),
        "required_rerouted_donor_ids_sha256": _tensor_sha256(rerouted),
        "integer_routing_values_are_structural_metadata": True,
        "indices_are_cpu_int64": True,
        "coefficients_are_cpu_float32": True,
        "every_final_direct_donor_is_exact_one_hot_in_both_routes": True,
        "unaffected_target_rows_are_byte_identical": True,
        "unaffected_input_rows_are_byte_identical": True,
        "unaffected_output_rows_are_byte_identical": True,
        "mapping_is_injective": True,
        "operation_audit": _ZERO_OPERATION_AUDIT,
    }
    return Lfm2SemanticCarrierMaterialization(
        new_table,
        new_mapping,
        *new_input,
        *new_output,
        _sealed(body, "artifact_sha256"),
    )
