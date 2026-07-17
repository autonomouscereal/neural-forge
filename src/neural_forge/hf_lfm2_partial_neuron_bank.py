"""Zero-training partial coupled-neuron sharing for LFM2 SwiGLU FFNs.

PyTorch cannot physically share individual rows or columns inside otherwise
dense ``Linear.weight`` Parameters.  This primitive therefore installs a real
indexed runtime:

* each selected coupled SwiGLU unit is ``w1[row]``, ``w3[row]``, and
  ``w2[:, column]``;
* one three-tensor bank entry is shared by two selected layer/unit
  occurrences;
* every unselected unit is copied byte-for-byte into layer-private tensors;
* each layer evaluates its private units plus indexed shared-bank units.

One paired bank entry removes exactly ``3 * hidden_size`` unique learned
scalars.  Integer index buffers are counted as metadata, and this version has
no per-layer or per-unit scalar compensation.  Planning uses sealed
post-SwiGLU activations and hashes of the exact live coupled weights.  No
forward capture, optimizer, backward pass, gradient write, or training step is
performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from neural_forge.hf_lfm2_canonical import (
    Lfm2CanonicalError,
    inspect_lfm2_layer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_parameter_sharing import (
    DEFAULT_CORRELATION_CHUNK_SIZE,
    DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    derive_neuron_permutation,
)


EVIDENCE_SCHEMA = "neural-forge-lfm2-partial-neuron-bank-evidence-v1"
PLAN_SCHEMA = "neural-forge-lfm2-partial-neuron-bank-plan-v1"
REPORT_SCHEMA = "neural-forge-lfm2-partial-neuron-bank-report-v1"
SELECTION_METHOD = (
    "activation-permutation-plus-coupled-weight-cosine-greedy-pairs"
)
_PLAN_SEAL_SCHEMA = "neural-forge-lfm2-partial-neuron-bank-seal-v1"
_ROLES = ("gate", "up", "down")
_PROJECTION_NAMES = ("w1", "w3", "w2")
_SHA256_LENGTH = 64
_TRANSACTION_LOCK = threading.RLock()


class Lfm2PartialNeuronBankError(RuntimeError):
    """Base error for partial LFM2 coupled-neuron sharing."""


class Lfm2PartialNeuronBankEvidenceError(Lfm2PartialNeuronBankError):
    """Activation or weight evidence is missing, stale, or modified."""


class Lfm2PartialNeuronBankTopologyError(Lfm2PartialNeuronBankError):
    """The model or requested partial-sharing topology is unsupported."""


class Lfm2PartialNeuronBankTransactionError(Lfm2PartialNeuronBankError):
    """Installation, accounting, sealing, or rollback failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 partial neuron banking."
        ) from exc
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
        raise Lfm2PartialNeuronBankEvidenceError(
            "Partial-neuron evidence must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _tensor_record(
    torch: Any,
    value: Any,
    name: str,
    *,
    require_floating: bool = True,
) -> dict[str, Any]:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or (require_floating and not value.is_floating_point())
    ):
        raise Lfm2PartialNeuronBankEvidenceError(
            f"{name} must be a materialized strided tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if require_floating and not bool(torch.isfinite(tensor).all()):
        raise Lfm2PartialNeuronBankEvidenceError(
            f"{name} contains non-finite values."
        )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-partial-neuron-tensor-v1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(size) for size in tensor.shape],
        "sha256": digest.hexdigest(),
    }


def _clone_cpu(torch: Any, value: Any, name: str) -> Any:
    _tensor_record(torch, value, name)
    return value.detach().to(device="cpu").contiguous().clone()


def _locate_layers(model: Any) -> tuple[Any, ...]:
    inner = getattr(model, "model", None)
    nested = getattr(inner, "layers", None)
    direct = getattr(model, "layers", None)
    candidates = [
        value for value in (nested, direct) if value is not None
    ]
    if len(candidates) != 1:
        raise Lfm2PartialNeuronBankTopologyError(
            "The model must expose exactly one decoder stack."
        )
    try:
        layers = tuple(candidates[0])
    except TypeError as exc:
        raise Lfm2PartialNeuronBankTopologyError(
            "The LFM2 decoder stack is not iterable."
        ) from exc
    if len(layers) < 2:
        raise Lfm2PartialNeuronBankTopologyError(
            "Partial sharing requires at least two decoder layers."
        )
    return layers


def _storage_identity(value: Any) -> tuple[str, int]:
    try:
        pointer = int(value.untyped_storage().data_ptr())
    except (AttributeError, RuntimeError):
        pointer = int(value.data_ptr())
    return str(value.device), pointer


def _inspect_dense_parent(
    model: Any,
) -> tuple[tuple[Any, ...], int, int]:
    torch = _torch()
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "lfm2":
        raise Lfm2PartialNeuronBankTopologyError(
            "The source config.model_type must be 'lfm2'."
        )
    layers = _locate_layers(model)
    contracts = []
    parameter_ids: set[int] = set()
    storage_ids: set[tuple[str, int]] = set()
    try:
        for layer in layers:
            contract = inspect_lfm2_layer(layer)
            contracts.append(contract)
            for role in _PROJECTION_NAMES:
                projection = getattr(layer.feed_forward, role, None)
                weight = getattr(projection, "weight", None)
                if (
                    not isinstance(projection, torch.nn.Linear)
                    or projection.bias is not None
                    or not isinstance(weight, torch.nn.Parameter)
                ):
                    raise Lfm2PartialNeuronBankTopologyError(
                        "Every source FFN must contain independent bias-free "
                        "w1/w3/w2 Linear projections."
                    )
                identity = id(weight)
                storage = _storage_identity(weight)
                if identity in parameter_ids or storage in storage_ids:
                    raise Lfm2PartialNeuronBankTopologyError(
                        "The source already contains shared or overlapping FFN "
                        "Parameter storage."
                    )
                parameter_ids.add(identity)
                storage_ids.add(storage)
    except Lfm2CanonicalError as exc:
        raise Lfm2PartialNeuronBankTopologyError(
            "The source is not a supported Transformers LFM2 layout."
        ) from exc
    first = contracts[0]
    if any(
        contract.hidden_size != first.hidden_size
        or contract.ffn_intermediate_size != first.ffn_intermediate_size
        for contract in contracts
    ):
        raise Lfm2PartialNeuronBankTopologyError(
            "Every LFM2 layer must use one hidden/intermediate FFN shape."
        )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise Lfm2PartialNeuronBankTopologyError(
            "The source contains gradients; zero-training evidence is required."
        )
    return layers, first.hidden_size, first.ffn_intermediate_size


def _parent_identity(model: Any) -> str:
    try:
        return fingerprint_lfm2_parent(model).identity_sha256
    except Exception as exc:
        raise Lfm2PartialNeuronBankTopologyError(
            "Could not authenticate the exact loaded LFM2 parent."
        ) from exc


def _parameter_counts(model: Any) -> tuple[int, int, int, int]:
    try:
        named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - current torch supports this
        raise Lfm2PartialNeuronBankTopologyError(
            "PyTorch must expose non-deduplicated Parameter traversal."
        ) from exc
    logical = sum(int(parameter.numel()) for _, parameter in named)
    unique = {id(parameter): parameter for _, parameter in named}
    return (
        logical,
        sum(int(parameter.numel()) for parameter in unique.values()),
        len(named),
        len(unique),
    )


def _layer_weight_records(
    torch: Any,
    layer: Any,
    layer_index: int,
) -> dict[str, Any]:
    return {
        "gate": _tensor_record(
            torch,
            layer.feed_forward.w1.weight,
            f"layer.{layer_index}.weight.gate",
        ),
        "up": _tensor_record(
            torch,
            layer.feed_forward.w3.weight,
            f"layer.{layer_index}.weight.up",
        ),
        "down": _tensor_record(
            torch,
            layer.feed_forward.w2.weight,
            f"layer.{layer_index}.weight.down",
        ),
    }


@dataclass(frozen=True, slots=True)
class Lfm2PartialNeuronBankEvidence:
    """Sealed post-SwiGLU rows plus exact live weight payload bindings."""

    schema: str
    cache_key: str
    heldout_corpus_sha256: str
    parent_identity_sha256: str
    layer_rows: tuple[Any, ...]
    layer_activation_records: tuple[Mapping[str, Any], ...]
    layer_weight_records: tuple[Mapping[str, Any], ...]
    row_count: int
    layer_count: int
    hidden_size: int
    intermediate_size: int
    evidence_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cache_key": self.cache_key,
            "heldout_corpus_sha256": self.heldout_corpus_sha256,
            "parent_identity_sha256": self.parent_identity_sha256,
            "layer_activation_records": [
                dict(value) for value in self.layer_activation_records
            ],
            "layer_weight_records": [
                dict(value) for value in self.layer_weight_records
            ],
            "row_count": self.row_count,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "post_swiglu_verified": True,
            "weight_payloads_verified": True,
            "evidence_sha256": self.evidence_sha256,
        }


def _evidence_body(
    evidence: Lfm2PartialNeuronBankEvidence,
) -> dict[str, Any]:
    body = evidence.as_dict()
    body.pop("evidence_sha256", None)
    return body


def _normalize_activation_rows(
    torch: Any,
    activation_rows: Mapping[int, Any] | Sequence[Any],
    *,
    layer_count: int,
    intermediate_size: int,
) -> tuple[Any, ...]:
    if isinstance(activation_rows, Mapping):
        if set(activation_rows) != set(range(layer_count)):
            raise Lfm2PartialNeuronBankEvidenceError(
                "Activation rows must cover every layer exactly once."
            )
        values = tuple(activation_rows[index] for index in range(layer_count))
    elif isinstance(activation_rows, Sequence) and not isinstance(
        activation_rows,
        (str, bytes),
    ):
        values = tuple(activation_rows)
    else:
        raise Lfm2PartialNeuronBankEvidenceError(
            "activation_rows must be a layer mapping or sequence."
        )
    if len(values) != layer_count:
        raise Lfm2PartialNeuronBankEvidenceError(
            "Activation row count differs from the decoder layer count."
        )
    result = []
    row_count = None
    for layer_index, value in enumerate(values):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or int(value.shape[1]) != intermediate_size
        ):
            raise Lfm2PartialNeuronBankEvidenceError(
                "Every post-SwiGLU activation must be [rows, intermediate]."
            )
        current_rows = int(value.shape[0])
        row_count = current_rows if row_count is None else row_count
        if current_rows != row_count or current_rows < 2:
            raise Lfm2PartialNeuronBankEvidenceError(
                "Post-SwiGLU activations must be path-aligned with at least two rows."
            )
        result.append(
            _clone_cpu(
                torch,
                value,
                f"layer.{layer_index}.activation.post_swiglu",
            )
        )
    return tuple(result)


def seal_lfm2_partial_neuron_bank_evidence(
    model: Any,
    activation_rows: Mapping[int, Any] | Sequence[Any],
    *,
    cache_key: str,
    heldout_corpus_sha256: str,
) -> Lfm2PartialNeuronBankEvidence:
    """Seal cached activations and hashes of every coupled FFN weight payload."""

    torch = _torch()
    with _TRANSACTION_LOCK, torch.no_grad():
        if not _valid_sha256(cache_key) or not _valid_sha256(
            heldout_corpus_sha256
        ):
            raise Lfm2PartialNeuronBankEvidenceError(
                "cache_key and heldout_corpus_sha256 must be lowercase SHA-256."
            )
        layers, hidden, intermediate = _inspect_dense_parent(model)
        rows = _normalize_activation_rows(
            torch,
            activation_rows,
            layer_count=len(layers),
            intermediate_size=intermediate,
        )
        activation_records = tuple(
            _tensor_record(
                torch,
                value,
                f"layer.{index}.activation.post_swiglu",
            )
            for index, value in enumerate(rows)
        )
        weight_records = tuple(
            _layer_weight_records(torch, layer, index)
            for index, layer in enumerate(layers)
        )
        evidence = Lfm2PartialNeuronBankEvidence(
            schema=EVIDENCE_SCHEMA,
            cache_key=cache_key,
            heldout_corpus_sha256=heldout_corpus_sha256,
            parent_identity_sha256=_parent_identity(model),
            layer_rows=rows,
            layer_activation_records=activation_records,
            layer_weight_records=weight_records,
            row_count=int(rows[0].shape[0]),
            layer_count=len(layers),
            hidden_size=hidden,
            intermediate_size=intermediate,
            evidence_sha256="",
        )
        return replace(
            evidence,
            evidence_sha256=_json_sha256(_evidence_body(evidence)),
        )


def _validate_evidence(
    model: Any,
    evidence: Lfm2PartialNeuronBankEvidence,
) -> tuple[tuple[Any, ...], int, int]:
    torch = _torch()
    layers, hidden, intermediate = _inspect_dense_parent(model)
    if (
        not isinstance(evidence, Lfm2PartialNeuronBankEvidence)
        or evidence.schema != EVIDENCE_SCHEMA
        or not _valid_sha256(evidence.evidence_sha256)
        or evidence.layer_count != len(layers)
        or evidence.hidden_size != hidden
        or evidence.intermediate_size != intermediate
        or evidence.parent_identity_sha256 != _parent_identity(model)
        or len(evidence.layer_rows) != len(layers)
        or len(evidence.layer_activation_records) != len(layers)
        or len(evidence.layer_weight_records) != len(layers)
    ):
        raise Lfm2PartialNeuronBankEvidenceError(
            "Partial-neuron evidence is stale or does not match this parent."
        )
    for index, rows in enumerate(evidence.layer_rows):
        if (
            tuple(rows.shape) != (evidence.row_count, intermediate)
            or _tensor_record(
                torch,
                rows,
                f"layer.{index}.activation.post_swiglu",
            )
            != dict(evidence.layer_activation_records[index])
            or _layer_weight_records(torch, layers[index], index)
            != dict(evidence.layer_weight_records[index])
        ):
            raise Lfm2PartialNeuronBankEvidenceError(
                "Activation or coupled-weight evidence changed after sealing."
            )
    if _json_sha256(_evidence_body(evidence)) != evidence.evidence_sha256:
        raise Lfm2PartialNeuronBankEvidenceError(
            "Partial-neuron evidence metadata changed after sealing."
        )
    return layers, hidden, intermediate


@dataclass(frozen=True, slots=True)
class Lfm2CoupledUnitPair:
    """One bank entry shared by two disjoint layer/unit occurrences."""

    bank_index: int
    left_layer: int
    left_unit: int
    right_layer: int
    right_unit: int
    activation_similarity: float
    weight_similarity: float
    combined_similarity: float

    @property
    def occurrences(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (self.left_layer, self.left_unit),
            (self.right_layer, self.right_unit),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "bank_index": self.bank_index,
            "left_layer": self.left_layer,
            "left_unit": self.left_unit,
            "right_layer": self.right_layer,
            "right_unit": self.right_unit,
            "activation_similarity": self.activation_similarity,
            "weight_similarity": self.weight_similarity,
            "combined_similarity": self.combined_similarity,
            "coupled_roles": ["w1.row", "w3.row", "w2.column"],
        }


@dataclass(frozen=True, slots=True)
class Lfm2PartialNeuronLayerPlan:
    """Private payload and shared-bank indices for one replaced FFN."""

    layer_index: int
    private_unit_indices: tuple[int, ...]
    shared_source_unit_indices: tuple[int, ...]
    bank_indices: tuple[int, ...]
    private_gate: Any
    private_up: Any
    private_down: Any
    private_requires_grad: tuple[bool, bool, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "private_unit_indices": list(self.private_unit_indices),
            "shared_source_unit_indices": list(
                self.shared_source_unit_indices
            ),
            "bank_indices": list(self.bank_indices),
            "private_unit_count": len(self.private_unit_indices),
            "shared_unit_count": len(self.bank_indices),
            "private_requires_grad": list(self.private_requires_grad),
        }


@dataclass(frozen=True, slots=True)
class Lfm2PartialNeuronBankPlan:
    """Canonical sealed model-level partial coupled-neuron sharing plan."""

    schema: str
    parent_identity_sha256: str
    evidence_sha256: str
    target_parameters_saved: int
    target_unique_parameter_budget: int | None
    unit_pair_count: int
    unit_storage_parameters: int
    selected_pairs: tuple[Lfm2CoupledUnitPair, ...]
    layer_plans: tuple[Lfm2PartialNeuronLayerPlan, ...]
    bank_gate: Any
    bank_up: Any
    bank_down: Any
    bank_requires_grad: tuple[bool, bool, bool]
    activation_weight: float
    coupled_weight_weight: float
    logical_parameters_before: int
    unique_parameters_before: int
    projected_unique_parameters_after: int
    hidden_size: int
    intermediate_size: int
    layer_count: int
    plan_sha256: str = ""
    training_steps: int = 0
    gradients_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_identity_sha256": self.parent_identity_sha256,
            "evidence_sha256": self.evidence_sha256,
            "selection_method": SELECTION_METHOD,
            "target_parameters_saved": self.target_parameters_saved,
            "target_unique_parameter_budget": (
                self.target_unique_parameter_budget
            ),
            "unit_pair_count": self.unit_pair_count,
            "unit_storage_parameters": self.unit_storage_parameters,
            "selected_pairs": [
                value.as_dict() for value in self.selected_pairs
            ],
            "layer_plans": [
                value.as_dict() for value in self.layer_plans
            ],
            "bank_requires_grad": list(self.bank_requires_grad),
            "activation_weight": self.activation_weight,
            "coupled_weight_weight": self.coupled_weight_weight,
            "logical_parameters_before": self.logical_parameters_before,
            "unique_parameters_before": self.unique_parameters_before,
            "projected_unique_parameters_after": (
                self.projected_unique_parameters_after
            ),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "layer_count": self.layer_count,
            "physical_runtime": "shared-bank-plus-indexed-private-tensors",
            "compensation_parameters_per_occurrence": 0,
            "compensation_parameters_total": 0,
            "integer_index_buffers_are_parameters": False,
            "unshared_units_preserved_exactly": True,
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
            "plan_sha256": self.plan_sha256,
        }


def partial_neuron_bank_pair_savings(hidden_size: int, pair_count: int) -> int:
    """Return exact unique learned-scalar savings for paired coupled units."""

    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
        or isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or pair_count < 0
    ):
        raise Lfm2PartialNeuronBankTopologyError(
            "hidden_size must be positive and pair_count non-negative."
        )
    return 3 * hidden_size * pair_count


def _normalized_score_weights(
    activation_weight: float,
    coupled_weight_weight: float,
) -> tuple[float, float]:
    values = (activation_weight, coupled_weight_weight)
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise Lfm2PartialNeuronBankTopologyError(
            "Similarity weights must be finite non-negative numbers."
        ) from exc
    total = sum(normalized)
    if (
        any(not math.isfinite(value) or value < 0.0 for value in normalized)
        or total <= 0.0
    ):
        raise Lfm2PartialNeuronBankTopologyError(
            "Similarity weights must be finite, non-negative, and nonzero."
        )
    return normalized[0] / total, normalized[1] / total


def _normalize_layer_pairs(
    layer_pairs: Sequence[Sequence[int]] | None,
    layer_count: int,
) -> tuple[tuple[int, int], ...]:
    if layer_pairs is None:
        return tuple(combinations(range(layer_count), 2))
    if isinstance(layer_pairs, (str, bytes)) or not isinstance(
        layer_pairs,
        Sequence,
    ):
        raise Lfm2PartialNeuronBankTopologyError(
            "layer_pairs must be a sequence of two-index pairs."
        )
    result = []
    for value in layer_pairs:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 2
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in value
            )
        ):
            raise Lfm2PartialNeuronBankTopologyError(
                "Each layer pair must contain two integer indices."
            )
        left, right = sorted((int(value[0]), int(value[1])))
        if left == right or left < 0 or right >= layer_count:
            raise Lfm2PartialNeuronBankTopologyError(
                "Layer pairs must contain distinct in-range indices."
            )
        result.append((left, right))
    if not result or len(set(result)) != len(result):
        raise Lfm2PartialNeuronBankTopologyError(
            "layer_pairs must be nonempty and unique."
        )
    return tuple(sorted(result))


def _coupled_rows(layer: Any) -> tuple[Any, Any, Any]:
    return (
        layer.feed_forward.w1.weight.detach(),
        layer.feed_forward.w3.weight.detach(),
        layer.feed_forward.w2.weight.detach().transpose(0, 1),
    )


def _coupled_weight_similarity(
    torch: Any,
    left_layer: Any,
    right_layer: Any,
    permutation: Any,
) -> Any:
    left = tuple(
        value.to(device="cpu", dtype=torch.float64).contiguous()
        for value in _coupled_rows(left_layer)
    )
    right = tuple(
        value.to(device="cpu", dtype=torch.float64)
        .index_select(0, permutation)
        .contiguous()
        for value in _coupled_rows(right_layer)
    )
    numerator = sum(
        (left_value * right_value).sum(dim=1)
        for left_value, right_value in zip(left, right)
    )
    left_norm = torch.sqrt(
        sum(value.square().sum(dim=1) for value in left)
    )
    right_norm = torch.sqrt(
        sum(value.square().sum(dim=1) for value in right)
    )
    result = numerator / (left_norm * right_norm).clamp_min(1e-30)
    if not bool(torch.isfinite(result).all()):
        raise Lfm2PartialNeuronBankEvidenceError(
            "Coupled weight similarities are non-finite."
        )
    return result.clamp(-1.0, 1.0)


def _target_pair_count(
    *,
    unique_before: int,
    unit_storage: int,
    target_parameters_saved: int | None,
    target_unique_parameter_budget: int | None,
) -> tuple[int, int]:
    if (target_parameters_saved is None) == (
        target_unique_parameter_budget is None
    ):
        raise Lfm2PartialNeuronBankTopologyError(
            "Specify exactly one target_parameters_saved or "
            "target_unique_parameter_budget."
        )
    if target_parameters_saved is not None:
        if (
            isinstance(target_parameters_saved, bool)
            or not isinstance(target_parameters_saved, int)
            or target_parameters_saved <= 0
            or target_parameters_saved % unit_storage
        ):
            raise Lfm2PartialNeuronBankTopologyError(
                "target_parameters_saved must be a positive exact multiple of "
                "one coupled-unit pair."
            )
        return target_parameters_saved // unit_storage, target_parameters_saved
    assert target_unique_parameter_budget is not None
    if (
        isinstance(target_unique_parameter_budget, bool)
        or not isinstance(target_unique_parameter_budget, int)
        or target_unique_parameter_budget <= 0
        or target_unique_parameter_budget >= unique_before
    ):
        raise Lfm2PartialNeuronBankTopologyError(
            "target_unique_parameter_budget must be positive and below the parent."
        )
    required = unique_before - target_unique_parameter_budget
    pair_count = (required + unit_storage - 1) // unit_storage
    return pair_count, pair_count * unit_storage


def _average_unit_rows(
    torch: Any,
    left: Any,
    right: Any,
) -> Any:
    dtype = left.dtype
    result = (
        (
            left.detach().to(device="cpu", dtype=torch.float64)
            + right.detach().to(device="cpu", dtype=torch.float64)
        )
        * 0.5
    ).to(dtype=dtype)
    if not bool(torch.isfinite(result).all()):
        raise Lfm2PartialNeuronBankEvidenceError(
            "A shared bank unit became non-finite."
        )
    return result.contiguous()


def _plan_seal_body(
    torch: Any,
    plan: Lfm2PartialNeuronBankPlan,
) -> dict[str, Any]:
    metadata = plan.as_dict()
    metadata.pop("plan_sha256", None)
    bank = {
        role: _tensor_record(
            torch,
            getattr(plan, f"bank_{role}"),
            f"plan.bank.{role}",
        )
        for role in _ROLES
    }
    layers = []
    for layer_plan in plan.layer_plans:
        layers.append(
            {
                "layer_index": layer_plan.layer_index,
                "private_gate": _tensor_record(
                    torch,
                    layer_plan.private_gate,
                    f"plan.layer.{layer_plan.layer_index}.private.gate",
                ),
                "private_up": _tensor_record(
                    torch,
                    layer_plan.private_up,
                    f"plan.layer.{layer_plan.layer_index}.private.up",
                ),
                "private_down": _tensor_record(
                    torch,
                    layer_plan.private_down,
                    f"plan.layer.{layer_plan.layer_index}.private.down",
                ),
            }
        )
    return {
        "schema": _PLAN_SEAL_SCHEMA,
        "metadata": metadata,
        "bank_payloads": bank,
        "private_payloads": layers,
    }


def _seal_plan(
    torch: Any,
    plan: Lfm2PartialNeuronBankPlan,
) -> Lfm2PartialNeuronBankPlan:
    return replace(
        plan,
        plan_sha256=_json_sha256(_plan_seal_body(torch, plan)),
    )


def _validate_plan_seal(
    torch: Any,
    plan: Lfm2PartialNeuronBankPlan,
) -> None:
    try:
        expected = _json_sha256(_plan_seal_body(torch, plan))
    except Lfm2PartialNeuronBankEvidenceError as exc:
        raise Lfm2PartialNeuronBankTransactionError(
            "The sealed partial-neuron bank plan payload is invalid."
        ) from exc
    if not _valid_sha256(plan.plan_sha256) or plan.plan_sha256 != expected:
        raise Lfm2PartialNeuronBankTransactionError(
            "The sealed partial-neuron bank plan changed after planning."
        )


def plan_lfm2_partial_neuron_bank(
    model: Any,
    evidence: Lfm2PartialNeuronBankEvidence,
    *,
    target_parameters_saved: int | None = None,
    target_unique_parameter_budget: int | None = None,
    layer_pairs: Sequence[Sequence[int]] | None = None,
    activation_weight: float = 0.5,
    coupled_weight_weight: float = 0.5,
    max_shared_units_per_layer: int | None = None,
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    correlation_chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
) -> Lfm2PartialNeuronBankPlan:
    """Select disjoint coupled-unit pairs and stage a physical indexed runtime."""

    torch = _torch()
    with _TRANSACTION_LOCK, torch.no_grad():
        layers, hidden, intermediate = _validate_evidence(model, evidence)
        logical_before, unique_before, _logical_objects, _unique_objects = (
            _parameter_counts(model)
        )
        unit_storage = 3 * hidden
        pair_count, exact_savings = _target_pair_count(
            unique_before=unique_before,
            unit_storage=unit_storage,
            target_parameters_saved=target_parameters_saved,
            target_unique_parameter_budget=target_unique_parameter_budget,
        )
        if pair_count > len(layers) * intermediate // 2:
            raise Lfm2PartialNeuronBankTopologyError(
                "The target requires more disjoint unit occurrences than exist."
            )
        maximum_per_layer = (
            intermediate
            if max_shared_units_per_layer is None
            else max_shared_units_per_layer
        )
        if (
            isinstance(maximum_per_layer, bool)
            or not isinstance(maximum_per_layer, int)
            or maximum_per_layer <= 0
            or maximum_per_layer > intermediate
        ):
            raise Lfm2PartialNeuronBankTopologyError(
                "max_shared_units_per_layer must be in [1, intermediate]."
            )
        activation_weight, coupled_weight_weight = (
            _normalized_score_weights(
                activation_weight,
                coupled_weight_weight,
            )
        )
        candidate_layer_pairs = _normalize_layer_pairs(
            layer_pairs,
            len(layers),
        )
        candidates = []
        for left_layer, right_layer in candidate_layer_pairs:
            assignment = derive_neuron_permutation(
                evidence.layer_rows[left_layer],
                evidence.layer_rows[right_layer],
                exact_assignment_limit=exact_assignment_limit,
                chunk_size=correlation_chunk_size,
                variance_floor=variance_floor,
            )
            permutation = assignment.permutation.to(
                device="cpu",
                dtype=torch.long,
            )
            activation_scores = assignment.assigned_correlations.to(
                device="cpu",
                dtype=torch.float64,
            )
            weight_scores = _coupled_weight_similarity(
                torch,
                layers[left_layer],
                layers[right_layer],
                permutation,
            )
            combined = (
                activation_weight * activation_scores
                + coupled_weight_weight * weight_scores
            )
            if not bool(torch.isfinite(combined).all()):
                raise Lfm2PartialNeuronBankEvidenceError(
                    "Combined unit similarities are non-finite."
                )
            for left_unit in range(intermediate):
                candidates.append(
                    (
                        float(combined[left_unit].item()),
                        float(activation_scores[left_unit].item()),
                        float(weight_scores[left_unit].item()),
                        left_layer,
                        left_unit,
                        right_layer,
                        int(permutation[left_unit].item()),
                    )
                )
        candidates.sort(
            key=lambda value: (
                -value[0],
                -value[1],
                -value[2],
                value[3],
                value[4],
                value[5],
                value[6],
            )
        )
        used: set[tuple[int, int]] = set()
        layer_counts = [0] * len(layers)
        selected_rows = []
        for candidate in candidates:
            (
                combined,
                activation,
                weight,
                left_layer,
                left_unit,
                right_layer,
                right_unit,
            ) = candidate
            occurrences = (
                (left_layer, left_unit),
                (right_layer, right_unit),
            )
            if (
                occurrences[0] in used
                or occurrences[1] in used
                or layer_counts[left_layer] >= maximum_per_layer
                or layer_counts[right_layer] >= maximum_per_layer
            ):
                continue
            bank_index = len(selected_rows)
            selected_rows.append(
                Lfm2CoupledUnitPair(
                    bank_index=bank_index,
                    left_layer=left_layer,
                    left_unit=left_unit,
                    right_layer=right_layer,
                    right_unit=right_unit,
                    activation_similarity=activation,
                    weight_similarity=weight,
                    combined_similarity=combined,
                )
            )
            used.update(occurrences)
            layer_counts[left_layer] += 1
            layer_counts[right_layer] += 1
            if len(selected_rows) == pair_count:
                break
        if len(selected_rows) != pair_count:
            raise Lfm2PartialNeuronBankTopologyError(
                "Deterministic selection could not meet the requested exact target."
            )

        bank_gate = []
        bank_up = []
        bank_down = []
        bank_requires_grad: tuple[bool, bool, bool] | None = None
        occurrences_by_layer: dict[int, list[tuple[int, int]]] = {}
        for selected in selected_rows:
            left_weights = _coupled_rows(layers[selected.left_layer])
            right_weights = _coupled_rows(layers[selected.right_layer])
            role_requires = tuple(
                bool(
                    getattr(
                        layers[selected.left_layer].feed_forward,
                        projection_name,
                    ).weight.requires_grad
                )
                for projection_name in _PROJECTION_NAMES
            )
            right_requires = tuple(
                bool(
                    getattr(
                        layers[selected.right_layer].feed_forward,
                        projection_name,
                    ).weight.requires_grad
                )
                for projection_name in _PROJECTION_NAMES
            )
            if role_requires != right_requires:
                raise Lfm2PartialNeuronBankTopologyError(
                    "Paired units disagree on requires_grad."
                )
            bank_requires_grad = (
                role_requires
                if bank_requires_grad is None
                else bank_requires_grad
            )
            if bank_requires_grad != role_requires:
                raise Lfm2PartialNeuronBankTopologyError(
                    "All bank entries must share one role-wise requires_grad policy."
                )
            averaged = tuple(
                _average_unit_rows(
                    torch,
                    left_value[selected.left_unit],
                    right_value[selected.right_unit],
                )
                for left_value, right_value in zip(
                    left_weights,
                    right_weights,
                )
            )
            bank_gate.append(averaged[0])
            bank_up.append(averaged[1])
            bank_down.append(averaged[2])
            occurrences_by_layer.setdefault(
                selected.left_layer,
                [],
            ).append((selected.left_unit, selected.bank_index))
            occurrences_by_layer.setdefault(
                selected.right_layer,
                [],
            ).append((selected.right_unit, selected.bank_index))
        assert bank_requires_grad is not None
        bank_tensors = (
            torch.stack(bank_gate).contiguous(),
            torch.stack(bank_up).contiguous(),
            torch.stack(bank_down).contiguous(),
        )
        layer_plans = []
        all_units = set(range(intermediate))
        for layer_index in sorted(occurrences_by_layer):
            occurrences = tuple(sorted(occurrences_by_layer[layer_index]))
            selected_units = tuple(value[0] for value in occurrences)
            bank_indices = tuple(value[1] for value in occurrences)
            private_units = tuple(sorted(all_units - set(selected_units)))
            index = torch.tensor(private_units, dtype=torch.long)
            gate, up, down = _coupled_rows(layers[layer_index])
            private_gate = (
                gate.detach().to(device="cpu").index_select(0, index).contiguous()
            )
            private_up = (
                up.detach().to(device="cpu").index_select(0, index).contiguous()
            )
            private_down = (
                down.detach().to(device="cpu").index_select(0, index).contiguous()
            )
            requires_grad = tuple(
                bool(
                    getattr(
                        layers[layer_index].feed_forward,
                        projection_name,
                    ).weight.requires_grad
                )
                for projection_name in _PROJECTION_NAMES
            )
            layer_plans.append(
                Lfm2PartialNeuronLayerPlan(
                    layer_index=layer_index,
                    private_unit_indices=private_units,
                    shared_source_unit_indices=selected_units,
                    bank_indices=bank_indices,
                    private_gate=private_gate,
                    private_up=private_up,
                    private_down=private_down,
                    private_requires_grad=requires_grad,
                )
            )
        plan = Lfm2PartialNeuronBankPlan(
            schema=PLAN_SCHEMA,
            parent_identity_sha256=evidence.parent_identity_sha256,
            evidence_sha256=evidence.evidence_sha256,
            target_parameters_saved=exact_savings,
            target_unique_parameter_budget=target_unique_parameter_budget,
            unit_pair_count=pair_count,
            unit_storage_parameters=unit_storage,
            selected_pairs=tuple(selected_rows),
            layer_plans=tuple(layer_plans),
            bank_gate=bank_tensors[0],
            bank_up=bank_tensors[1],
            bank_down=bank_tensors[2],
            bank_requires_grad=bank_requires_grad,
            activation_weight=activation_weight,
            coupled_weight_weight=coupled_weight_weight,
            logical_parameters_before=logical_before,
            unique_parameters_before=unique_before,
            projected_unique_parameters_after=unique_before - exact_savings,
            hidden_size=hidden,
            intermediate_size=intermediate,
            layer_count=len(layers),
        )
        if (
            plan.projected_unique_parameters_after <= 0
            or (
                target_unique_parameter_budget is not None
                and plan.projected_unique_parameters_after
                > target_unique_parameter_budget
            )
        ):
            raise Lfm2PartialNeuronBankTransactionError(
                "Projected unique-parameter accounting is invalid."
            )
        return _seal_plan(torch, plan)


class Lfm2CoupledNeuronBank(_torch().nn.Module):
    """Three shared row-oriented tensors for coupled SwiGLU units."""

    def __init__(
        self,
        gate: Any,
        up: Any,
        down: Any,
        *,
        requires_grad: tuple[bool, bool, bool],
    ) -> None:
        super().__init__()
        torch = _torch()
        if (
            gate.ndim != 2
            or up.shape != gate.shape
            or down.shape != gate.shape
            or not bool(torch.isfinite(gate).all())
            or not bool(torch.isfinite(up).all())
            or not bool(torch.isfinite(down).all())
        ):
            raise Lfm2PartialNeuronBankTopologyError(
                "Shared bank tensors must be finite equal-shaped matrices."
            )
        self.gate = torch.nn.Parameter(
            gate.detach().clone(),
            requires_grad=requires_grad[0],
        )
        self.up = torch.nn.Parameter(
            up.detach().clone(),
            requires_grad=requires_grad[1],
        )
        self.down = torch.nn.Parameter(
            down.detach().clone(),
            requires_grad=requires_grad[2],
        )

    @property
    def unit_count(self) -> int:
        return int(self.gate.shape[0])


class Lfm2PartialNeuronBankMLP(_torch().nn.Module):
    """One LFM2 FFN split into exact private units and indexed bank units."""

    def __init__(
        self,
        bank: Lfm2CoupledNeuronBank,
        *,
        private_gate: Any,
        private_up: Any,
        private_down: Any,
        bank_indices: Sequence[int],
        private_source_units: Sequence[int],
        shared_source_units: Sequence[int],
        requires_grad: tuple[bool, bool, bool],
    ) -> None:
        super().__init__()
        torch = _torch()
        if (
            not isinstance(bank, Lfm2CoupledNeuronBank)
            or private_gate.ndim != 2
            or private_up.shape != private_gate.shape
            or private_down.shape != private_gate.shape
            or private_gate.shape[1] != bank.gate.shape[1]
            or not bool(torch.isfinite(private_gate).all())
            or not bool(torch.isfinite(private_up).all())
            or not bool(torch.isfinite(private_down).all())
        ):
            raise Lfm2PartialNeuronBankTopologyError(
                "Private coupled-unit tensors are invalid."
            )
        indices = torch.tensor(
            tuple(bank_indices),
            dtype=torch.long,
            device=bank.gate.device,
        )
        if (
            indices.ndim != 1
            or int(indices.numel()) == 0
            or bool(((indices < 0) | (indices >= bank.unit_count)).any())
            or len(set(int(value) for value in indices.tolist()))
            != int(indices.numel())
        ):
            raise Lfm2PartialNeuronBankTopologyError(
                "Layer bank indices must be nonempty, unique, and in range."
            )
        self.bank = bank
        self.private_gate = torch.nn.Parameter(
            private_gate.detach().clone(),
            requires_grad=requires_grad[0],
        )
        self.private_up = torch.nn.Parameter(
            private_up.detach().clone(),
            requires_grad=requires_grad[1],
        )
        self.private_down = torch.nn.Parameter(
            private_down.detach().clone(),
            requires_grad=requires_grad[2],
        )
        self.register_buffer("bank_indices", indices, persistent=True)
        self.private_source_units = tuple(int(value) for value in private_source_units)
        self.shared_source_units = tuple(int(value) for value in shared_source_units)
        self.hidden_size = int(bank.gate.shape[1])
        self.intermediate_size = int(
            self.private_gate.shape[0] + self.bank_indices.numel()
        )

    def forward(self, inputs: Any) -> Any:
        torch = _torch()
        functional = torch.nn.functional
        private_gate = functional.linear(inputs, self.private_gate)
        private_up = functional.linear(inputs, self.private_up)
        private_activation = functional.silu(private_gate) * private_up
        private_output = private_activation.matmul(self.private_down)
        bank_gate = self.bank.gate.index_select(0, self.bank_indices)
        bank_up = self.bank.up.index_select(0, self.bank_indices)
        bank_down = self.bank.down.index_select(0, self.bank_indices)
        shared_gate = functional.linear(inputs, bank_gate)
        shared_up = functional.linear(inputs, bank_up)
        shared_activation = functional.silu(shared_gate) * shared_up
        return private_output + shared_activation.matmul(bank_down)


@dataclass(frozen=True, slots=True)
class Lfm2PartialNeuronBankReport:
    schema: str
    parent_identity_before: str
    candidate_state_sha256: str
    evidence_sha256: str
    plan_sha256: str
    selected_pairs: tuple[Lfm2CoupledUnitPair, ...]
    replaced_layers: tuple[int, ...]
    logical_parameters_before: int
    logical_parameters_after: int
    unique_parameters_before: int
    unique_parameters_after: int
    parameters_saved: int
    logical_parameter_objects_after: int
    unique_parameter_objects_after: int
    bank_parameter_count: int
    private_parameter_count: int
    integer_index_buffer_count: int
    integer_index_buffer_bytes: int
    compensation_parameter_count: int
    unshared_units_preserved_exactly: bool

    @property
    def compression_fraction(self) -> float:
        return self.parameters_saved / self.unique_parameters_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_identity_before": self.parent_identity_before,
            "candidate_state_sha256": self.candidate_state_sha256,
            "evidence_sha256": self.evidence_sha256,
            "plan_sha256": self.plan_sha256,
            "selected_pairs": [
                value.as_dict() for value in self.selected_pairs
            ],
            "replaced_layers": list(self.replaced_layers),
            "logical_parameters_before": self.logical_parameters_before,
            "logical_parameters_after": self.logical_parameters_after,
            "unique_parameters_before": self.unique_parameters_before,
            "unique_parameters_after": self.unique_parameters_after,
            "parameters_saved": self.parameters_saved,
            "compression_fraction": self.compression_fraction,
            "logical_parameter_objects_after": (
                self.logical_parameter_objects_after
            ),
            "unique_parameter_objects_after": (
                self.unique_parameter_objects_after
            ),
            "bank_parameter_count": self.bank_parameter_count,
            "private_parameter_count": self.private_parameter_count,
            "integer_index_buffer_count": self.integer_index_buffer_count,
            "integer_index_buffer_bytes": self.integer_index_buffer_bytes,
            "integer_index_buffers_are_parameters": False,
            "compensation_parameter_count": self.compensation_parameter_count,
            "unshared_units_preserved_exactly": (
                self.unshared_units_preserved_exactly
            ),
            "physical_runtime": "shared-bank-plus-indexed-private-tensors",
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
        }


@dataclass(frozen=True, slots=True)
class _FfnBinding:
    layer_index: int
    layer: Any
    original: Any
    installed: Lfm2PartialNeuronBankMLP


def _runtime_state_sha256(model: Any) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-partial-neuron-runtime-v1\0")
    for category, values in (
        (
            "parameter",
            model.named_parameters(remove_duplicate=False),
        ),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, value in values:
            digest.update(category.encode("ascii"))
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                _canonical_json(
                    _tensor_record(
                        torch,
                        value,
                        f"runtime.{category}.{name}",
                        require_floating=value.is_floating_point(),
                    )
                )
            )
    return digest.hexdigest()


def _assign_feed_forward(layer: Any, module: Any) -> None:
    layer.feed_forward = module


class InstalledLfm2PartialNeuronBank:
    """Live partial-bank candidate with explicit commit or exact rollback."""

    def __init__(
        self,
        *,
        model: Any,
        report: Lfm2PartialNeuronBankReport,
        bindings: tuple[_FfnBinding, ...],
        parent_parameter_objects: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.report = report
        self._bindings = bindings
        self._parent_parameter_objects = dict(parent_parameter_objects)
        self._active = True
        self._committed = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def committed(self) -> bool:
        return self._committed

    def rollback(self) -> bool:
        with _TRANSACTION_LOCK:
            if not self._active:
                return False
            for binding in self._bindings:
                if binding.layer.feed_forward is not binding.installed:
                    raise Lfm2PartialNeuronBankTransactionError(
                        "An installed partial-bank FFN changed before rollback."
                    )
            for binding in self._bindings:
                _assign_feed_forward(binding.layer, binding.original)
            restored = dict(
                self.model.named_parameters(remove_duplicate=False)
            )
            exact = bool(
                set(restored) == set(self._parent_parameter_objects)
                and all(
                    restored[name] is parameter
                    for name, parameter in self._parent_parameter_objects.items()
                )
                and _parent_identity(self.model)
                == self.report.parent_identity_before
            )
            if not exact:
                raise Lfm2PartialNeuronBankTransactionError(
                    "Partial-bank rollback did not restore exact parent objects/state."
                )
            self._bindings = ()
            self._parent_parameter_objects = {}
            self._active = False
            return True

    def commit(self) -> Lfm2PartialNeuronBankReport:
        with _TRANSACTION_LOCK:
            if not self._active:
                raise Lfm2PartialNeuronBankTransactionError(
                    "The partial-bank transaction is no longer active."
                )
            if any(
                binding.layer.feed_forward is not binding.installed
                for binding in self._bindings
            ):
                raise Lfm2PartialNeuronBankTransactionError(
                    "An installed partial-bank FFN changed before commit."
                )
            counts = _parameter_counts(self.model)
            if counts[1] != self.report.unique_parameters_after:
                raise Lfm2PartialNeuronBankTransactionError(
                    "Partial-bank unique storage changed before commit."
                )
            self._bindings = ()
            self._parent_parameter_objects = {}
            self._active = False
            self._committed = True
            return self.report

    def __enter__(self) -> InstalledLfm2PartialNeuronBank:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active:
            self.rollback()


def _validate_plan_structure(
    plan: Lfm2PartialNeuronBankPlan,
    *,
    layer_count: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    if (
        not isinstance(plan, Lfm2PartialNeuronBankPlan)
        or plan.schema != PLAN_SCHEMA
        or plan.layer_count != layer_count
        or plan.hidden_size != hidden_size
        or plan.intermediate_size != intermediate_size
        or len(plan.selected_pairs) != plan.unit_pair_count
        or plan.target_parameters_saved
        != partial_neuron_bank_pair_savings(hidden_size, plan.unit_pair_count)
        or plan.projected_unique_parameters_after
        != plan.unique_parameters_before - plan.target_parameters_saved
        or tuple(value.bank_index for value in plan.selected_pairs)
        != tuple(range(plan.unit_pair_count))
    ):
        raise Lfm2PartialNeuronBankTransactionError(
            "The partial-bank plan structure or accounting is invalid."
        )
    occurrences = [
        occurrence
        for pair in plan.selected_pairs
        for occurrence in pair.occurrences
    ]
    if len(set(occurrences)) != len(occurrences):
        raise Lfm2PartialNeuronBankTransactionError(
            "The partial-bank plan reuses a layer/unit occurrence."
        )
    planned_layers = {value.layer_index for value in plan.layer_plans}
    if planned_layers != {value[0] for value in occurrences}:
        raise Lfm2PartialNeuronBankTransactionError(
            "Layer plans do not exactly cover selected occurrences."
        )
    expected_by_layer: dict[int, list[tuple[int, int]]] = {}
    for pair in plan.selected_pairs:
        expected_by_layer.setdefault(pair.left_layer, []).append(
            (pair.left_unit, pair.bank_index)
        )
        expected_by_layer.setdefault(pair.right_layer, []).append(
            (pair.right_unit, pair.bank_index)
        )
    for layer_plan in plan.layer_plans:
        observed_occurrences = tuple(
            sorted(
                zip(
                    layer_plan.shared_source_unit_indices,
                    layer_plan.bank_indices,
                )
            )
        )
        if (
            len(layer_plan.shared_source_unit_indices)
            != len(layer_plan.bank_indices)
            or observed_occurrences
            != tuple(sorted(expected_by_layer[layer_plan.layer_index]))
            or len(
                set(
                    layer_plan.private_unit_indices
                    + layer_plan.shared_source_unit_indices
                )
            )
            != intermediate_size
            or set(layer_plan.private_unit_indices).intersection(
                layer_plan.shared_source_unit_indices
            )
            or set(
                layer_plan.private_unit_indices
                + layer_plan.shared_source_unit_indices
            )
            != set(range(intermediate_size))
        ):
            raise Lfm2PartialNeuronBankTransactionError(
                "A layer plan does not partition every coupled unit exactly once."
            )
    torch = _torch()
    expected_bank_shape = (plan.unit_pair_count, hidden_size)
    if any(
        tuple(getattr(plan, f"bank_{role}").shape) != expected_bank_shape
        for role in _ROLES
    ) or any(
        tuple(getattr(layer_plan, f"private_{role}").shape)
        != (len(layer_plan.private_unit_indices), hidden_size)
        for layer_plan in plan.layer_plans
        for role in _ROLES
    ):
        raise Lfm2PartialNeuronBankTransactionError(
            "Bank or private tensor shapes differ from exact unit accounting."
        )
    if any(
        not bool(
            torch.isfinite(getattr(plan, f"bank_{role}")).all()
        )
        for role in _ROLES
    ):
        raise Lfm2PartialNeuronBankTransactionError(
            "The partial-bank plan contains non-finite bank payloads."
        )


def install_lfm2_partial_neuron_bank_plan(
    model: Any,
    evidence: Lfm2PartialNeuronBankEvidence,
    plan: Lfm2PartialNeuronBankPlan,
) -> InstalledLfm2PartialNeuronBank:
    """Atomically replace selected dense FFNs with a real partial-bank runtime."""

    torch = _torch()
    with _TRANSACTION_LOCK, torch.no_grad():
        layers, hidden, intermediate = _validate_evidence(model, evidence)
        _validate_plan_structure(
            plan,
            layer_count=len(layers),
            hidden_size=hidden,
            intermediate_size=intermediate,
        )
        if (
            plan.parent_identity_sha256 != evidence.parent_identity_sha256
            or plan.evidence_sha256 != evidence.evidence_sha256
        ):
            raise Lfm2PartialNeuronBankTransactionError(
                "The partial-bank plan is stale for this model/evidence."
            )
        logical_before, unique_before, _logical_objects, _unique_objects = (
            _parameter_counts(model)
        )
        if (
            logical_before != plan.logical_parameters_before
            or unique_before != plan.unique_parameters_before
        ):
            raise Lfm2PartialNeuronBankTransactionError(
                "Model Parameter counts changed after partial-bank planning."
            )
        parent_parameters = dict(
            model.named_parameters(remove_duplicate=False)
        )
        first_weight = layers[plan.layer_plans[0].layer_index].feed_forward.w1.weight
        device = first_weight.device
        dtype = first_weight.dtype
        selected_specs = {
            (
                getattr(
                    layers[layer_plan.layer_index].feed_forward,
                    projection_name,
                ).weight.device,
                getattr(
                    layers[layer_plan.layer_index].feed_forward,
                    projection_name,
                ).weight.dtype,
            )
            for layer_plan in plan.layer_plans
            for projection_name in _PROJECTION_NAMES
        }
        if selected_specs != {(device, dtype)}:
            raise Lfm2PartialNeuronBankTopologyError(
                "All selected FFN projections must share one device and dtype "
                "for one physical bank."
            )
        bank = Lfm2CoupledNeuronBank(
            plan.bank_gate.to(device=device, dtype=dtype),
            plan.bank_up.to(device=device, dtype=dtype),
            plan.bank_down.to(device=device, dtype=dtype),
            requires_grad=plan.bank_requires_grad,
        )
        bindings = []
        for layer_plan in plan.layer_plans:
            layer = layers[layer_plan.layer_index]
            installed = Lfm2PartialNeuronBankMLP(
                bank,
                private_gate=layer_plan.private_gate.to(
                    device=device,
                    dtype=dtype,
                ),
                private_up=layer_plan.private_up.to(
                    device=device,
                    dtype=dtype,
                ),
                private_down=layer_plan.private_down.to(
                    device=device,
                    dtype=dtype,
                ),
                bank_indices=layer_plan.bank_indices,
                private_source_units=layer_plan.private_unit_indices,
                shared_source_units=(
                    layer_plan.shared_source_unit_indices
                ),
                requires_grad=layer_plan.private_requires_grad,
            )
            installed.training = bool(layer.feed_forward.training)
            bank.training = bool(layer.feed_forward.training)
            bindings.append(
                _FfnBinding(
                    layer_index=layer_plan.layer_index,
                    layer=layer,
                    original=layer.feed_forward,
                    installed=installed,
                )
            )

        # Tensor-bearing plans are mutable despite frozen dataclasses.  Keep
        # this as the final operation before the first model mutation.
        _validate_plan_seal(torch, plan)
        try:
            for binding in bindings:
                _assign_feed_forward(binding.layer, binding.installed)
            if any(
                binding.layer.feed_forward is not binding.installed
                for binding in bindings
            ):
                raise Lfm2PartialNeuronBankTransactionError(
                    "A selected layer did not install its indexed partial bank."
                )
            counts_after = _parameter_counts(model)
            observed_savings = unique_before - counts_after[1]
            if (
                counts_after[1] != plan.projected_unique_parameters_after
                or observed_savings != plan.target_parameters_saved
                or any(
                    parameter.grad is not None
                    for parameter in model.parameters()
                )
            ):
                raise Lfm2PartialNeuronBankTransactionError(
                    "Live unique Parameter accounting or zero-gradient audit failed."
                )
            bank_parameter_count = sum(
                int(parameter.numel()) for parameter in bank.parameters()
            )
            private_parameter_count = sum(
                int(value.numel())
                for binding in bindings
                for value in (
                    binding.installed.private_gate,
                    binding.installed.private_up,
                    binding.installed.private_down,
                )
            )
            report = Lfm2PartialNeuronBankReport(
                schema=REPORT_SCHEMA,
                parent_identity_before=evidence.parent_identity_sha256,
                candidate_state_sha256=_runtime_state_sha256(model),
                evidence_sha256=evidence.evidence_sha256,
                plan_sha256=plan.plan_sha256,
                selected_pairs=plan.selected_pairs,
                replaced_layers=tuple(
                    binding.layer_index for binding in bindings
                ),
                logical_parameters_before=logical_before,
                logical_parameters_after=counts_after[0],
                unique_parameters_before=unique_before,
                unique_parameters_after=counts_after[1],
                parameters_saved=observed_savings,
                logical_parameter_objects_after=counts_after[2],
                unique_parameter_objects_after=counts_after[3],
                bank_parameter_count=bank_parameter_count,
                private_parameter_count=private_parameter_count,
                integer_index_buffer_count=sum(
                    int(binding.installed.bank_indices.numel())
                    for binding in bindings
                ),
                integer_index_buffer_bytes=sum(
                    int(
                        binding.installed.bank_indices.numel()
                        * binding.installed.bank_indices.element_size()
                    )
                    for binding in bindings
                ),
                compensation_parameter_count=0,
                unshared_units_preserved_exactly=True,
            )
            return InstalledLfm2PartialNeuronBank(
                model=model,
                report=report,
                bindings=tuple(bindings),
                parent_parameter_objects=parent_parameters,
            )
        except BaseException as exc:
            for binding in bindings:
                _assign_feed_forward(binding.layer, binding.original)
            try:
                restored_parameters = dict(
                    model.named_parameters(remove_duplicate=False)
                )
                exact = bool(
                    set(restored_parameters) == set(parent_parameters)
                    and all(
                        restored_parameters[name] is parameter
                        for name, parameter in parent_parameters.items()
                    )
                    and _parent_identity(model)
                    == evidence.parent_identity_sha256
                )
            except BaseException as rollback_exc:
                raise Lfm2PartialNeuronBankTransactionError(
                    "Partial-bank installation failed and rollback could not be "
                    "authenticated."
                ) from rollback_exc
            if not exact:
                raise Lfm2PartialNeuronBankTransactionError(
                    "Partial-bank installation failed and exact rollback failed."
                ) from exc
            raise


def install_lfm2_partial_neuron_bank(
    model: Any,
    evidence: Lfm2PartialNeuronBankEvidence,
    **plan_kwargs: Any,
) -> InstalledLfm2PartialNeuronBank:
    """Plan and atomically install a partial coupled-neuron bank."""

    plan = plan_lfm2_partial_neuron_bank(
        model,
        evidence,
        **plan_kwargs,
    )
    return install_lfm2_partial_neuron_bank_plan(model, evidence, plan)


build_lfm2_partial_neuron_bank_plan = plan_lfm2_partial_neuron_bank


__all__ = [
    "EVIDENCE_SCHEMA",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "InstalledLfm2PartialNeuronBank",
    "Lfm2CoupledNeuronBank",
    "Lfm2CoupledUnitPair",
    "Lfm2PartialNeuronBankError",
    "Lfm2PartialNeuronBankEvidence",
    "Lfm2PartialNeuronBankEvidenceError",
    "Lfm2PartialNeuronBankMLP",
    "Lfm2PartialNeuronBankPlan",
    "Lfm2PartialNeuronBankReport",
    "Lfm2PartialNeuronBankTopologyError",
    "Lfm2PartialNeuronBankTransactionError",
    "Lfm2PartialNeuronLayerPlan",
    "build_lfm2_partial_neuron_bank_plan",
    "install_lfm2_partial_neuron_bank",
    "install_lfm2_partial_neuron_bank_plan",
    "partial_neuron_bank_pair_savings",
    "plan_lfm2_partial_neuron_bank",
    "seal_lfm2_partial_neuron_bank_evidence",
]
