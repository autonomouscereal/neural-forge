"""Dynamic zero-training, per-layer variable-width folding for LFM2.

Uniform FFN contraction makes every layer pay the same width reduction even
when sealed validation evidence says their sensitivities differ sharply.  This
module instead:

* seals aligned gate/up/post-SwiGLU rows and per-layer validation curves;
* allocates an exact global parameter-removal budget across fourteen layers;
* plants real parent medoid producers into bias-free ``w1``/``w3`` matrices;
* aggregates the coupled ``w2`` consumer without reading blank child values;
* preserves every attention, short-convolution, norm, embedding, and head
  tensor exactly from the parent; and
* exposes one in-memory commit/byte-exact rollback boundary.

The optional residual corrector is called only after the complete parent-
derived candidate is installed.  It may observe the live candidate and return
one replacement ``w2`` matrix per layer, but it may not mutate either model.
No optimizer, backward call, gradient, or training step is used.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from neural_forge.hf_activation_cluster import (
    build_gated_mlp_activation_cluster_plan,
)
from neural_forge.hf_lfm2_canonical import (
    Lfm2CanonicalError,
    inspect_lfm2_layer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent


LFM2_VARIABLE_WIDTH_LAYER_COUNT = 14
EVIDENCE_SCHEMA = "neural-forge-lfm2-variable-width-evidence-v1"
ALLOCATION_SCHEMA = "neural-forge-lfm2-variable-width-allocation-v1"
PLAN_SCHEMA = "neural-forge-lfm2-variable-width-plan-v1"
REPORT_SCHEMA = "neural-forge-lfm2-variable-width-report-v1"
FIXED_WIDTH_BUNDLE_SCHEMA = "neural-forge-lfm2-variable-width-fixed-plan-bundle-v1"
LAYER_BINDING_SCHEMA = "neural-forge-lfm2-variable-width-layer-binding-v1"
METHOD_ID = "lfm2_sensitivity_allocated_variable_width_medoid_fold"
_MARKER = "_neural_forge_variable_width_fold_transaction"
_FFN_PROJECTIONS = ("w1", "w3", "w2")
_FORWARD_HOOK_ATTRIBUTES = (
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
)
_TRANSACTION_LOCK = threading.RLock()


class Lfm2VariableWidthFoldError(RuntimeError):
    """Base failure for variable-width LFM2 folding."""


class Lfm2VariableWidthEvidenceError(Lfm2VariableWidthFoldError):
    """Sealed sensitivity or activation evidence is invalid or stale."""


class Lfm2VariableWidthAllocationError(Lfm2VariableWidthFoldError):
    """The requested global budget has no supported exact allocation."""


class Lfm2VariableWidthTopologyError(Lfm2VariableWidthFoldError):
    """The parent or blank child topology cannot support this fold."""


class Lfm2VariableWidthTransactionError(Lfm2VariableWidthFoldError):
    """Plan installation, accounting, commit, or rollback failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 variable-width folding."
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
        raise Lfm2VariableWidthEvidenceError(
            "Variable-width provenance must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2VariableWidthEvidenceError(f"{name} must be a positive integer.")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2VariableWidthEvidenceError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2VariableWidthEvidenceError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2VariableWidthEvidenceError(f"{name} must be finite and non-negative.")
    return result


def _tensor_record(
    value: Any,
    name: str,
    *,
    require_floating: bool = True,
) -> dict[str, Any]:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or (require_floating and not value.is_floating_point())
    ):
        raise Lfm2VariableWidthEvidenceError(
            f"{name} must be a materialized strided tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if require_floating and not bool(torch.isfinite(tensor).all().item()):
        raise Lfm2VariableWidthEvidenceError(f"{name} contains NaN or infinity.")
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-variable-width-tensor-v1\0")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(size) for size in tensor.shape],
        "sha256": digest.hexdigest(),
    }


def _clone_cpu(value: Any, name: str) -> Any:
    _tensor_record(value, name)
    return value.detach().to(device="cpu").contiguous().clone()


def _layers(model: Any) -> tuple[Any, ...]:
    body = getattr(model, "model", None)
    values = getattr(body, "layers", None)
    try:
        layers = tuple(values)
    except TypeError as exc:
        raise Lfm2VariableWidthTopologyError(
            "Expected model.model.layers on an LFM2 causal model."
        ) from exc
    if len(layers) != LFM2_VARIABLE_WIDTH_LAYER_COUNT:
        raise Lfm2VariableWidthTopologyError(
            "This proof requires the concrete fourteen-layer LFM2 topology."
        )
    return layers


def _inspect_dense_model(
    model: Any,
) -> tuple[tuple[Any, ...], int, int, tuple[str, ...]]:
    torch = _torch()
    if getattr(getattr(model, "config", None), "model_type", None) != "lfm2":
        raise Lfm2VariableWidthTopologyError(
            "The model config.model_type must be 'lfm2'."
        )
    layers = _layers(model)
    contracts = []
    try:
        for layer in layers:
            contract = inspect_lfm2_layer(layer)
            contracts.append(contract)
            ffn_state = layer.feed_forward.state_dict()
            if set(ffn_state) != {
                "w1.weight",
                "w3.weight",
                "w2.weight",
            }:
                raise Lfm2VariableWidthTopologyError(
                    "Every LFM2 FFN must contain only bias-free w1/w3/w2 "
                    "projection state."
                )
            for projection_name in _FFN_PROJECTIONS:
                projection = getattr(layer.feed_forward, projection_name, None)
                if (
                    not isinstance(projection, torch.nn.Linear)
                    or projection.bias is not None
                    or not isinstance(projection.weight, torch.nn.Parameter)
                ):
                    raise Lfm2VariableWidthTopologyError(
                        "Every LFM2 FFN projection must be a bias-free Linear."
                    )
    except Lfm2CanonicalError as exc:
        raise Lfm2VariableWidthTopologyError(
            "The model is not a supported Transformers LFM2 layout."
        ) from exc
    first = contracts[0]
    if any(
        contract.hidden_size != first.hidden_size
        or contract.ffn_intermediate_size != first.ffn_intermediate_size
        for contract in contracts
    ):
        raise Lfm2VariableWidthTopologyError(
            "The dense source must use one hidden/intermediate shape."
        )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise Lfm2VariableWidthTopologyError(
            "The model contains gradients; zero-training state is required."
        )
    return (
        layers,
        first.hidden_size,
        first.ffn_intermediate_size,
        tuple(contract.kind for contract in contracts),
    )


def _parent_identity(model: Any) -> str:
    try:
        identity = fingerprint_lfm2_parent(model).identity_sha256
    except Exception as exc:
        raise Lfm2VariableWidthTopologyError(
            "Could not authenticate the exact loaded LFM2 parent."
        ) from exc
    if not _valid_sha256(identity):  # pragma: no cover - upstream invariant
        raise Lfm2VariableWidthTopologyError(
            "The loaded LFM2 parent identity is malformed."
        )
    return identity


def _unique_parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _weight_records(
    layers: Sequence[Any],
) -> tuple[Mapping[str, Any], ...]:
    result = []
    for index, layer in enumerate(layers):
        result.append(
            MappingProxyType(
                {
                    "gate": _tensor_record(
                        layer.feed_forward.w1.weight,
                        f"layer.{index}.parent.gate",
                    ),
                    "up": _tensor_record(
                        layer.feed_forward.w3.weight,
                        f"layer.{index}.parent.up",
                    ),
                    "down": _tensor_record(
                        layer.feed_forward.w2.weight,
                        f"layer.{index}.parent.down",
                    ),
                }
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Lfm2WidthCostPoint:
    """One cumulative held-out degradation measurement at a candidate width."""

    width: int
    whole_trajectory_cost: float
    residual_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "width",
            _positive_int(self.width, "width"),
        )
        object.__setattr__(
            self,
            "whole_trajectory_cost",
            _finite_nonnegative(
                self.whole_trajectory_cost,
                "whole_trajectory_cost",
            ),
        )
        object.__setattr__(
            self,
            "residual_cost",
            _finite_nonnegative(self.residual_cost, "residual_cost"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "whole_trajectory_cost": self.whole_trajectory_cost,
            "residual_cost": self.residual_cost,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LayerWidthCurve:
    """Ordered source-to-narrower validation measurements for one layer."""

    layer_index: int
    points: tuple[Lfm2WidthCostPoint, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < LFM2_VARIABLE_WIDTH_LAYER_COUNT
        ):
            raise Lfm2VariableWidthEvidenceError(
                "layer_index must be in the fourteen-layer LFM2 range."
            )
        points = tuple(self.points)
        if not points or any(
            not isinstance(point, Lfm2WidthCostPoint) for point in points
        ):
            raise Lfm2VariableWidthEvidenceError(
                "A sensitivity curve needs typed cost points."
            )
        widths = tuple(point.width for point in points)
        if any(left <= right for left, right in zip(widths, widths[1:])):
            raise Lfm2VariableWidthEvidenceError(
                "Curve widths must be unique and strictly descending."
            )
        object.__setattr__(self, "points", points)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthLayerActivations:
    """Immutable aligned gate/up/post-SwiGLU rows for one parent layer."""

    gate: Any
    up: Any
    post_swiglu: Any


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthEvidence:
    """Hash-bound activation rows, sensitivity curves, and parent weights."""

    schema: str
    cache_key: str
    heldout_corpus_sha256: str
    parent_identity_sha256: str
    layer_activations: tuple[Lfm2VariableWidthLayerActivations, ...]
    activation_records: tuple[Mapping[str, Any], ...]
    parent_weight_records: tuple[Mapping[str, Any], ...]
    curves: tuple[Lfm2LayerWidthCurve, ...]
    row_count: int
    hidden_size: int
    source_intermediate_width: int
    layer_kinds: tuple[str, ...]
    evidence_builder_id: str
    evidence_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cache_key": self.cache_key,
            "heldout_corpus_sha256": self.heldout_corpus_sha256,
            "parent_identity_sha256": self.parent_identity_sha256,
            "activation_records": [dict(record) for record in self.activation_records],
            "parent_weight_records": [
                dict(record) for record in self.parent_weight_records
            ],
            "curves": [curve.as_dict() for curve in self.curves],
            "row_count": self.row_count,
            "layer_count": len(self.layer_activations),
            "hidden_size": self.hidden_size,
            "source_intermediate_width": self.source_intermediate_width,
            "layer_kinds": list(self.layer_kinds),
            "evidence_builder_id": self.evidence_builder_id,
            "activation_roles": ["gate", "up", "post_swiglu"],
            "curve_costs_are_cumulative": True,
            "evidence_sha256": self.evidence_sha256,
        }


def _evidence_body(evidence: Lfm2VariableWidthEvidence) -> dict[str, Any]:
    result = evidence.as_dict()
    result.pop("evidence_sha256", None)
    return result


def _coerce_layer_activations(
    activation_rows: Mapping[int, Any] | Sequence[Any],
    *,
    intermediate_width: int,
) -> tuple[Lfm2VariableWidthLayerActivations, ...]:
    torch = _torch()
    if isinstance(activation_rows, Mapping):
        if set(activation_rows) != set(range(LFM2_VARIABLE_WIDTH_LAYER_COUNT)):
            raise Lfm2VariableWidthEvidenceError(
                "Activation rows must cover all fourteen layers exactly once."
            )
        raw = tuple(
            activation_rows[index] for index in range(LFM2_VARIABLE_WIDTH_LAYER_COUNT)
        )
    elif isinstance(activation_rows, Sequence) and not isinstance(
        activation_rows,
        (str, bytes),
    ):
        raw = tuple(activation_rows)
    else:
        raise Lfm2VariableWidthEvidenceError(
            "activation_rows must be a layer mapping or sequence."
        )
    if len(raw) != LFM2_VARIABLE_WIDTH_LAYER_COUNT:
        raise Lfm2VariableWidthEvidenceError(
            "Activation rows must contain exactly fourteen layers."
        )
    result = []
    common_rows: int | None = None
    for index, item in enumerate(raw):
        if isinstance(item, Lfm2VariableWidthLayerActivations):
            values = {
                "gate": item.gate,
                "up": item.up,
                "post_swiglu": item.post_swiglu,
            }
        elif isinstance(item, Mapping):
            if set(item) != {"gate", "up", "post_swiglu"}:
                raise Lfm2VariableWidthEvidenceError(
                    "Each activation payload needs gate/up/post_swiglu."
                )
            values = dict(item)
        else:
            raise Lfm2VariableWidthEvidenceError(
                "Each layer activation payload must be typed or mapped."
            )
        shapes = []
        cloned = {}
        for role, value in values.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or int(value.shape[1]) != intermediate_width
            ):
                raise Lfm2VariableWidthEvidenceError(
                    "Every activation role must be [rows, source_width]."
                )
            cloned[role] = _clone_cpu(
                value,
                f"layer.{index}.activation.{role}",
            )
            shapes.append(tuple(int(size) for size in value.shape))
        if len(set(shapes)) != 1:
            raise Lfm2VariableWidthEvidenceError(
                "A layer's gate/up/post-SwiGLU rows are not aligned."
            )
        rows = shapes[0][0]
        common_rows = rows if common_rows is None else common_rows
        if rows < 2 or rows != common_rows:
            raise Lfm2VariableWidthEvidenceError(
                "All layer activations must share at least two aligned rows."
            )
        result.append(Lfm2VariableWidthLayerActivations(**cloned))
    return tuple(result)


def _activation_records(
    values: Sequence[Lfm2VariableWidthLayerActivations],
) -> tuple[Mapping[str, Any], ...]:
    result = []
    for index, item in enumerate(values):
        result.append(
            MappingProxyType(
                {
                    role: _tensor_record(
                        getattr(item, role),
                        f"layer.{index}.activation.{role}",
                    )
                    for role in ("gate", "up", "post_swiglu")
                }
            )
        )
    return tuple(result)


def seal_lfm2_variable_width_evidence(
    parent: Any,
    activation_rows: Mapping[int, Any] | Sequence[Any],
    curves: Sequence[Lfm2LayerWidthCurve],
    *,
    cache_key: str,
    heldout_corpus_sha256: str,
    evidence_builder_id: str,
) -> Lfm2VariableWidthEvidence:
    """Seal one cached, parent-bound width-sensitivity envelope."""

    torch = _torch()
    with _TRANSACTION_LOCK, torch.no_grad():
        if not _valid_sha256(cache_key) or not _valid_sha256(heldout_corpus_sha256):
            raise Lfm2VariableWidthEvidenceError(
                "cache_key and heldout_corpus_sha256 must be SHA-256 digests."
            )
        if not isinstance(evidence_builder_id, str) or not evidence_builder_id:
            raise Lfm2VariableWidthEvidenceError(
                "evidence_builder_id must be a nonempty string."
            )
        layers, hidden, intermediate, kinds = _inspect_dense_model(parent)
        rows = _coerce_layer_activations(
            activation_rows,
            intermediate_width=intermediate,
        )
        ordered_curves = tuple(curves)
        if (
            len(ordered_curves) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
            or any(
                not isinstance(curve, Lfm2LayerWidthCurve) for curve in ordered_curves
            )
            or tuple(curve.layer_index for curve in ordered_curves)
            != tuple(range(LFM2_VARIABLE_WIDTH_LAYER_COUNT))
            or any(curve.points[0].width != intermediate for curve in ordered_curves)
            or any(
                curve.points[0].whole_trajectory_cost != 0.0
                or curve.points[0].residual_cost != 0.0
                for curve in ordered_curves
            )
        ):
            raise Lfm2VariableWidthEvidenceError(
                "Curves must cover layers 0..13 in order and start at the "
                "zero-cost source width."
            )
        evidence = Lfm2VariableWidthEvidence(
            schema=EVIDENCE_SCHEMA,
            cache_key=cache_key,
            heldout_corpus_sha256=heldout_corpus_sha256,
            parent_identity_sha256=_parent_identity(parent),
            layer_activations=rows,
            activation_records=_activation_records(rows),
            parent_weight_records=_weight_records(layers),
            curves=ordered_curves,
            row_count=int(rows[0].gate.shape[0]),
            hidden_size=hidden,
            source_intermediate_width=intermediate,
            layer_kinds=kinds,
            evidence_builder_id=evidence_builder_id,
            evidence_sha256="",
        )
        return replace(
            evidence,
            evidence_sha256=_json_sha256(_evidence_body(evidence)),
        )


def _validate_evidence_seal(evidence: Lfm2VariableWidthEvidence) -> None:
    if (
        not isinstance(evidence, Lfm2VariableWidthEvidence)
        or evidence.schema != EVIDENCE_SCHEMA
        or not _valid_sha256(evidence.evidence_sha256)
        or len(evidence.layer_activations) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
        or len(evidence.activation_records) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
        or len(evidence.parent_weight_records) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
        or len(evidence.curves) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
    ):
        raise Lfm2VariableWidthEvidenceError(
            "The variable-width evidence structure is invalid."
        )
    for index, rows in enumerate(evidence.layer_activations):
        expected = {
            role: _tensor_record(
                getattr(rows, role),
                f"layer.{index}.activation.{role}",
            )
            for role in ("gate", "up", "post_swiglu")
        }
        if expected != dict(evidence.activation_records[index]):
            raise Lfm2VariableWidthEvidenceError(
                "Activation evidence changed after sealing."
            )
    if _json_sha256(_evidence_body(evidence)) != evidence.evidence_sha256:
        raise Lfm2VariableWidthEvidenceError(
            "Sensitivity evidence metadata changed after sealing."
        )


def _validate_live_evidence(
    parent: Any,
    evidence: Lfm2VariableWidthEvidence,
) -> tuple[tuple[Any, ...], int, int, tuple[str, ...]]:
    _validate_evidence_seal(evidence)
    layers, hidden, intermediate, kinds = _inspect_dense_model(parent)
    if (
        evidence.parent_identity_sha256 != _parent_identity(parent)
        or evidence.hidden_size != hidden
        or evidence.source_intermediate_width != intermediate
        or evidence.layer_kinds != kinds
        or evidence.parent_weight_records != _weight_records(layers)
    ):
        raise Lfm2VariableWidthEvidenceError(
            "The sealed evidence is stale or belongs to another exact parent."
        )
    return layers, hidden, intermediate, kinds


def verify_lfm2_variable_width_evidence(
    evidence: Lfm2VariableWidthEvidence,
    *,
    parent: Any | None = None,
) -> Lfm2VariableWidthEvidence:
    """Reauthenticate a cached envelope, optionally against its live parent."""

    if parent is None:
        _validate_evidence_seal(evidence)
    else:
        _validate_live_evidence(parent, evidence)
    return evidence


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthAllocation:
    """Exact sensitivity-minimizing allocation under one global budget."""

    schema: str
    evidence_sha256: str
    source_widths: tuple[int, ...]
    target_widths: tuple[int, ...]
    chosen_costs: tuple[float, ...]
    weighted_validation_cost: float
    parameters_saved: int
    minimum_width: int
    maximum_width_step: int
    whole_trajectory_weight: float
    residual_weight: float
    allocation_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_sha256": self.evidence_sha256,
            "source_widths": list(self.source_widths),
            "target_widths": list(self.target_widths),
            "chosen_costs": list(self.chosen_costs),
            "weighted_validation_cost": self.weighted_validation_cost,
            "parameters_saved": self.parameters_saved,
            "minimum_width": self.minimum_width,
            "maximum_width_step": self.maximum_width_step,
            "whole_trajectory_weight": self.whole_trajectory_weight,
            "residual_weight": self.residual_weight,
            "allocator": "exact-budget-dynamic-program-over-sealed-curves",
            "allocation_sha256": self.allocation_sha256,
        }


def _allocation_body(
    allocation: Lfm2VariableWidthAllocation,
) -> dict[str, Any]:
    result = allocation.as_dict()
    result.pop("allocation_sha256", None)
    return result


def allocate_lfm2_variable_widths(
    evidence: Lfm2VariableWidthEvidence,
    *,
    target_parameters_saved: int,
    minimum_width: int,
    maximum_width_step: int,
    whole_trajectory_weight: float = 1.0,
    residual_weight: float = 1.0,
) -> Lfm2VariableWidthAllocation:
    """Allocate exact FFN removals to the least-cost supported layers."""

    _validate_evidence_seal(evidence)
    target = _positive_int(
        target_parameters_saved,
        "target_parameters_saved",
    )
    minimum = _positive_int(minimum_width, "minimum_width")
    maximum_step = _positive_int(maximum_width_step, "maximum_width_step")
    trajectory_weight = _finite_nonnegative(
        whole_trajectory_weight,
        "whole_trajectory_weight",
    )
    residual_cost_weight = _finite_nonnegative(
        residual_weight,
        "residual_weight",
    )
    if trajectory_weight + residual_cost_weight <= 0.0:
        raise Lfm2VariableWidthAllocationError(
            "At least one sensitivity cost weight must be positive."
        )
    source = evidence.source_intermediate_width
    if minimum > source:
        raise Lfm2VariableWidthAllocationError(
            "minimum_width cannot exceed the source width."
        )

    choices: list[tuple[tuple[int, int, float], ...]] = []
    for curve in evidence.curves:
        eligible = tuple(point for point in curve.points if point.width >= minimum)
        if not eligible or eligible[0].width != source:
            raise Lfm2VariableWidthAllocationError(
                "Each curve must retain its source-width point."
            )
        if any(
            left.width - right.width > maximum_step
            for left, right in zip(eligible, eligible[1:])
        ):
            raise Lfm2VariableWidthAllocationError(
                "A sensitivity-curve transition exceeds maximum_width_step."
            )
        layer_choices = []
        for point in eligible:
            saved = (source - point.width) * 3 * evidence.hidden_size
            cost = (
                trajectory_weight * point.whole_trajectory_cost
                + residual_cost_weight * point.residual_cost
            )
            layer_choices.append((saved, point.width, cost))
        choices.append(tuple(layer_choices))

    # saved -> (total cost, widths, individual chosen costs)
    states: dict[int, tuple[float, tuple[int, ...], tuple[float, ...]]] = {
        0: (0.0, (), ())
    }
    for layer_choices in choices:
        updated: dict[
            int,
            tuple[float, tuple[int, ...], tuple[float, ...]],
        ] = {}
        for saved_before, (cost_before, widths_before, costs_before) in states.items():
            for saved, width, cost in layer_choices:
                total_saved = saved_before + saved
                if total_saved > target:
                    continue
                candidate = (
                    cost_before + cost,
                    widths_before + (width,),
                    costs_before + (cost,),
                )
                current = updated.get(total_saved)
                if current is None or (candidate[0], candidate[1]) < (
                    current[0],
                    current[1],
                ):
                    updated[total_saved] = candidate
        states = updated
    if target not in states:
        raise Lfm2VariableWidthAllocationError(
            "The sealed curves cannot realize the requested parameter budget "
            "exactly at this width granularity."
        )
    total_cost, target_widths, chosen_costs = states[target]
    allocation = Lfm2VariableWidthAllocation(
        schema=ALLOCATION_SCHEMA,
        evidence_sha256=evidence.evidence_sha256,
        source_widths=(source,) * LFM2_VARIABLE_WIDTH_LAYER_COUNT,
        target_widths=target_widths,
        chosen_costs=chosen_costs,
        weighted_validation_cost=total_cost,
        parameters_saved=target,
        minimum_width=minimum,
        maximum_width_step=maximum_step,
        whole_trajectory_weight=trajectory_weight,
        residual_weight=residual_cost_weight,
        allocation_sha256="",
    )
    return replace(
        allocation,
        allocation_sha256=_json_sha256(_allocation_body(allocation)),
    )


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthLayerPlan:
    """One source-derived coupled SwiGLU payload."""

    layer_index: int
    source_width: int
    target_width: int
    medoid_source_neurons: tuple[int, ...]
    source_to_target: tuple[int, ...]
    consumer_coefficients: tuple[float, ...]
    gate_weight: Any
    up_weight: Any
    down_weight: Any
    requires_grad: tuple[bool, bool, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "medoid_source_neurons": list(self.medoid_source_neurons),
            "source_to_target": list(self.source_to_target),
            "consumer_coefficients": list(self.consumer_coefficients),
            "gate_weight": _tensor_record(
                self.gate_weight,
                f"plan.layer.{self.layer_index}.gate",
            ),
            "up_weight": _tensor_record(
                self.up_weight,
                f"plan.layer.{self.layer_index}.up",
            ),
            "down_weight": _tensor_record(
                self.down_weight,
                f"plan.layer.{self.layer_index}.down",
            ),
            "requires_grad": list(self.requires_grad),
            "real_parent_medoid_producers": True,
            "blank_values_read": False,
        }


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthFoldPlan:
    """Canonical, tensor-sealed nonuniform fourteen-layer fold plan."""

    schema: str
    method_id: str
    parent_identity_sha256: str
    evidence_sha256: str
    allocation: Lfm2VariableWidthAllocation
    layer_plans: tuple[Lfm2VariableWidthLayerPlan, ...]
    source_unique_parameters: int
    target_unique_parameters: int
    parameters_saved: int
    hidden_size: int
    source_intermediate_width: int
    layer_kinds: tuple[str, ...]
    cluster_ridge: float
    cluster_max_iterations: int
    cluster_distance_chunk_size: int
    plan_sha256: str

    @property
    def target_widths(self) -> tuple[int, ...]:
        return self.allocation.target_widths

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "parent_identity_sha256": self.parent_identity_sha256,
            "evidence_sha256": self.evidence_sha256,
            "allocation": self.allocation.as_dict(),
            "layer_plans": [value.as_dict() for value in self.layer_plans],
            "source_unique_parameters": self.source_unique_parameters,
            "target_unique_parameters": self.target_unique_parameters,
            "parameters_saved": self.parameters_saved,
            "hidden_size": self.hidden_size,
            "source_intermediate_width": self.source_intermediate_width,
            "target_widths": list(self.target_widths),
            "layer_kinds": list(self.layer_kinds),
            "cluster_ridge": self.cluster_ridge,
            "cluster_max_iterations": self.cluster_max_iterations,
            "cluster_distance_chunk_size": self.cluster_distance_chunk_size,
            "plan_sha256": self.plan_sha256,
            "full_depth": True,
            "variable_width": True,
            "checkpoint_serialization_supported": False,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def fingerprint_lfm2_variable_width_layer_plan(
    layer_plan: Lfm2VariableWidthLayerPlan,
) -> str:
    """Bind a causal measurement to one exact physical FFN layer payload."""

    if not isinstance(layer_plan, Lfm2VariableWidthLayerPlan):
        raise Lfm2VariableWidthTransactionError(
            "layer_plan must be Lfm2VariableWidthLayerPlan."
        )
    return _json_sha256(
        {
            "schema": LAYER_BINDING_SCHEMA,
            "layer_plan": layer_plan.as_dict(),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthFixedPlanBundle:
    """The fixed-width evidence view required to install a fixed plan."""

    schema: str
    source_evidence_sha256: str
    fixed_evidence: Lfm2VariableWidthEvidence
    plan: Lfm2VariableWidthFoldPlan
    layer_binding_sha256: tuple[str, ...]
    bundle_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_evidence_sha256": self.source_evidence_sha256,
            "fixed_evidence_sha256": self.fixed_evidence.evidence_sha256,
            "plan": self.plan.as_dict(),
            "layer_binding_sha256": list(self.layer_binding_sha256),
            "bundle_sha256": self.bundle_sha256,
        }


def _fixed_bundle_body(
    bundle: Lfm2VariableWidthFixedPlanBundle,
) -> dict[str, Any]:
    result = bundle.as_dict()
    result.pop("bundle_sha256", None)
    return result


def verify_lfm2_variable_width_fixed_plan_bundle(
    bundle: Lfm2VariableWidthFixedPlanBundle,
    *,
    parent: Any | None = None,
) -> Lfm2VariableWidthFixedPlanBundle:
    """Reauthenticate a fixed-width evidence/plan/install binding."""

    if (
        not isinstance(bundle, Lfm2VariableWidthFixedPlanBundle)
        or bundle.schema != FIXED_WIDTH_BUNDLE_SCHEMA
        or not _valid_sha256(bundle.source_evidence_sha256)
        or not _valid_sha256(bundle.bundle_sha256)
        or len(bundle.layer_binding_sha256) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
    ):
        raise Lfm2VariableWidthTransactionError(
            "The fixed-width plan bundle structure is invalid."
        )
    verify_lfm2_variable_width_evidence(
        bundle.fixed_evidence,
        parent=parent,
    )
    _validate_plan_seal(bundle.plan)
    if (
        bundle.plan.evidence_sha256 != bundle.fixed_evidence.evidence_sha256
        or bundle.layer_binding_sha256
        != tuple(
            fingerprint_lfm2_variable_width_layer_plan(layer_plan)
            for layer_plan in bundle.plan.layer_plans
        )
        or _json_sha256(_fixed_bundle_body(bundle)) != bundle.bundle_sha256
    ):
        raise Lfm2VariableWidthTransactionError(
            "The fixed-width evidence, plan, or layer binding changed."
        )
    return bundle


def _plan_body(plan: Lfm2VariableWidthFoldPlan) -> dict[str, Any]:
    result = plan.as_dict()
    result.pop("plan_sha256", None)
    return result


def _validate_allocation_seal(
    allocation: Lfm2VariableWidthAllocation,
) -> None:
    if (
        not isinstance(allocation, Lfm2VariableWidthAllocation)
        or allocation.schema != ALLOCATION_SCHEMA
        or not _valid_sha256(allocation.allocation_sha256)
        or _json_sha256(_allocation_body(allocation)) != allocation.allocation_sha256
        or len(allocation.source_widths) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
        or len(allocation.target_widths) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
    ):
        raise Lfm2VariableWidthTransactionError(
            "The sealed width allocation changed after planning."
        )


def _validate_plan_seal(plan: Lfm2VariableWidthFoldPlan) -> None:
    if not isinstance(plan, Lfm2VariableWidthFoldPlan):
        raise Lfm2VariableWidthTransactionError(
            "plan must be Lfm2VariableWidthFoldPlan."
        )
    _validate_allocation_seal(plan.allocation)
    if (
        plan.schema != PLAN_SCHEMA
        or plan.method_id != METHOD_ID
        or not _valid_sha256(plan.plan_sha256)
        or _json_sha256(_plan_body(plan)) != plan.plan_sha256
        or len(plan.layer_plans) != LFM2_VARIABLE_WIDTH_LAYER_COUNT
        or tuple(value.layer_index for value in plan.layer_plans)
        != tuple(range(LFM2_VARIABLE_WIDTH_LAYER_COUNT))
        or plan.target_unique_parameters
        != plan.source_unique_parameters - plan.parameters_saved
        or plan.parameters_saved != plan.allocation.parameters_saved
        or plan.target_widths != tuple(value.target_width for value in plan.layer_plans)
    ):
        raise Lfm2VariableWidthTransactionError(
            "The sealed variable-width plan structure or payload changed."
        )
    expected_saved = 0
    for layer_plan in plan.layer_plans:
        expected_saved += (
            (layer_plan.source_width - layer_plan.target_width) * 3 * plan.hidden_size
        )
        if (
            layer_plan.source_width != plan.source_intermediate_width
            or not 0 < layer_plan.target_width <= layer_plan.source_width
            or len(layer_plan.medoid_source_neurons) != layer_plan.target_width
            or len(set(layer_plan.medoid_source_neurons)) != layer_plan.target_width
            or any(
                not 0 <= value < layer_plan.source_width
                for value in layer_plan.medoid_source_neurons
            )
            or len(layer_plan.source_to_target) != layer_plan.source_width
            or any(
                not 0 <= value < layer_plan.target_width
                for value in layer_plan.source_to_target
            )
            or len(layer_plan.consumer_coefficients) != layer_plan.source_width
            or tuple(layer_plan.gate_weight.shape)
            != (layer_plan.target_width, plan.hidden_size)
            or tuple(layer_plan.up_weight.shape)
            != (layer_plan.target_width, plan.hidden_size)
            or tuple(layer_plan.down_weight.shape)
            != (plan.hidden_size, layer_plan.target_width)
        ):
            raise Lfm2VariableWidthTransactionError(
                "A variable-width layer payload is structurally invalid."
            )
    if expected_saved != plan.parameters_saved:
        raise Lfm2VariableWidthTransactionError(
            "Variable-width parameter accounting is inconsistent."
        )


def plan_lfm2_variable_width_fold(
    parent: Any,
    evidence: Lfm2VariableWidthEvidence,
    *,
    target_parameters_saved: int,
    minimum_width: int,
    maximum_width_step: int,
    whole_trajectory_weight: float = 1.0,
    residual_weight: float = 1.0,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
) -> Lfm2VariableWidthFoldPlan:
    """Build an immutable parent-only plan without touching a blank child."""

    torch = _torch()
    ridge = _finite_nonnegative(cluster_ridge, "cluster_ridge")
    iterations = _positive_int(
        cluster_max_iterations,
        "cluster_max_iterations",
    )
    chunk_size = _positive_int(
        cluster_distance_chunk_size,
        "cluster_distance_chunk_size",
    )
    with _TRANSACTION_LOCK, torch.no_grad():
        layers, hidden, source_width, kinds = _validate_live_evidence(
            parent,
            evidence,
        )
        parent_before = _parent_identity(parent)
        allocation = allocate_lfm2_variable_widths(
            evidence,
            target_parameters_saved=target_parameters_saved,
            minimum_width=minimum_width,
            maximum_width_step=maximum_width_step,
            whole_trajectory_weight=whole_trajectory_weight,
            residual_weight=residual_weight,
        )
        layer_plans = []
        for index, (layer, target_width) in enumerate(
            zip(layers, allocation.target_widths, strict=True)
        ):
            gate_source = layer.feed_forward.w1.weight
            up_source = layer.feed_forward.w3.weight
            down_source = layer.feed_forward.w2.weight
            if target_width == source_width:
                medoids = tuple(range(source_width))
                assignment = tuple(range(source_width))
                coefficients = (1.0,) * source_width
                gate = _clone_cpu(
                    gate_source,
                    f"plan.layer.{index}.gate",
                )
                up = _clone_cpu(
                    up_source,
                    f"plan.layer.{index}.up",
                )
                down = _clone_cpu(
                    down_source,
                    f"plan.layer.{index}.down",
                )
            else:
                rows = evidence.layer_activations[index]
                source_down = down_source.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
                contribution = rows.post_swiglu.to(torch.float64) * (
                    torch.linalg.vector_norm(source_down, dim=0).unsqueeze(0)
                )
                cluster = build_gated_mlp_activation_cluster_plan(
                    rows.gate,
                    rows.up,
                    rows.post_swiglu,
                    contribution,
                    target_neurons=target_width,
                    source_neurons=source_width,
                    ridge=ridge,
                    max_iterations=iterations,
                    distance_chunk_size=chunk_size,
                )
                medoids = tuple(cluster.medoid_source_neurons)
                assignment = tuple(cluster.source_to_target)
                coefficients = tuple(cluster.consumer_coefficients)
                gate = _clone_cpu(
                    cluster.select_gate_producer(gate_source, axis=0),
                    f"plan.layer.{index}.gate",
                )
                up = _clone_cpu(
                    cluster.select_up_producer(up_source, axis=0),
                    f"plan.layer.{index}.up",
                )
                down = _clone_cpu(
                    cluster.aggregate_down_consumer(down_source, axis=1),
                    f"plan.layer.{index}.down",
                )
            layer_plans.append(
                Lfm2VariableWidthLayerPlan(
                    layer_index=index,
                    source_width=source_width,
                    target_width=target_width,
                    medoid_source_neurons=medoids,
                    source_to_target=assignment,
                    consumer_coefficients=coefficients,
                    gate_weight=gate,
                    up_weight=up,
                    down_weight=down,
                    requires_grad=(
                        bool(gate_source.requires_grad),
                        bool(up_source.requires_grad),
                        bool(down_source.requires_grad),
                    ),
                )
            )
        source_parameters = _unique_parameter_count(parent)
        plan = Lfm2VariableWidthFoldPlan(
            schema=PLAN_SCHEMA,
            method_id=METHOD_ID,
            parent_identity_sha256=evidence.parent_identity_sha256,
            evidence_sha256=evidence.evidence_sha256,
            allocation=allocation,
            layer_plans=tuple(layer_plans),
            source_unique_parameters=source_parameters,
            target_unique_parameters=(source_parameters - allocation.parameters_saved),
            parameters_saved=allocation.parameters_saved,
            hidden_size=hidden,
            source_intermediate_width=source_width,
            layer_kinds=kinds,
            cluster_ridge=ridge,
            cluster_max_iterations=iterations,
            cluster_distance_chunk_size=chunk_size,
            plan_sha256="",
        )
        sealed = replace(plan, plan_sha256=_json_sha256(_plan_body(plan)))
        _validate_plan_seal(sealed)
        if _parent_identity(parent) != parent_before:
            raise Lfm2VariableWidthTransactionError(
                "Variable-width planning mutated the parent."
            )
        return sealed


build_lfm2_variable_width_fold_plan = plan_lfm2_variable_width_fold


def plan_lfm2_variable_width_fold_for_widths(
    parent: Any,
    evidence: Lfm2VariableWidthEvidence,
    target_widths: Sequence[int],
    *,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
) -> Lfm2VariableWidthFixedPlanBundle:
    """Build the existing physical FFN transaction for one exact width vector.

    The source evidence must already contain every requested layer/width point.
    A sealed two-point view (source width and requested width) makes the
    existing exact allocator choose the supplied vector without introducing a
    second implementation of medoid selection, payload construction, parameter
    accounting, or transaction installation.
    """

    _validate_live_evidence(parent, evidence)
    widths = tuple(target_widths)
    if len(widths) != LFM2_VARIABLE_WIDTH_LAYER_COUNT or any(
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or width > evidence.source_intermediate_width
        for width in widths
    ):
        raise Lfm2VariableWidthAllocationError(
            "target_widths must contain fourteen positive contractions."
        )
    fixed_curves = []
    for layer_index, (curve, target_width) in enumerate(
        zip(evidence.curves, widths, strict=True)
    ):
        by_width = {point.width: point for point in curve.points}
        if target_width not in by_width:
            raise Lfm2VariableWidthAllocationError(
                f"Layer {layer_index} has no sealed causal point at width "
                f"{target_width}."
            )
        source_point = by_width[evidence.source_intermediate_width]
        points = (
            (source_point,)
            if target_width == evidence.source_intermediate_width
            else (source_point, by_width[target_width])
        )
        fixed_curves.append(
            Lfm2LayerWidthCurve(
                layer_index=layer_index,
                points=points,
            )
        )
    widths_sha256 = _json_sha256(
        {
            "source_evidence_sha256": evidence.evidence_sha256,
            "target_widths": list(widths),
        }
    )
    fixed = replace(
        evidence,
        curves=tuple(fixed_curves),
        evidence_builder_id=(
            f"{evidence.evidence_builder_id}:fixed-widths:{widths_sha256}"
        ),
        evidence_sha256="",
    )
    fixed = replace(
        fixed,
        evidence_sha256=_json_sha256(_evidence_body(fixed)),
    )
    _validate_live_evidence(parent, fixed)
    parameters_saved = sum(
        (evidence.source_intermediate_width - width) * 3 * evidence.hidden_size
        for width in widths
    )
    if parameters_saved <= 0:
        raise Lfm2VariableWidthAllocationError(
            "A fixed-width fold must remove at least one parameter."
        )
    maximum_step = max(
        1,
        max(evidence.source_intermediate_width - width for width in widths),
    )
    plan = plan_lfm2_variable_width_fold(
        parent,
        fixed,
        target_parameters_saved=parameters_saved,
        minimum_width=min(widths),
        maximum_width_step=maximum_step,
        whole_trajectory_weight=1.0,
        residual_weight=1.0,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
    )
    if plan.target_widths != widths:
        raise Lfm2VariableWidthAllocationError(
            "The fixed evidence view did not reproduce the requested widths."
        )
    layer_bindings = tuple(
        fingerprint_lfm2_variable_width_layer_plan(layer_plan)
        for layer_plan in plan.layer_plans
    )
    bundle = Lfm2VariableWidthFixedPlanBundle(
        schema=FIXED_WIDTH_BUNDLE_SCHEMA,
        source_evidence_sha256=evidence.evidence_sha256,
        fixed_evidence=fixed,
        plan=plan,
        layer_binding_sha256=layer_bindings,
        bundle_sha256="",
    )
    return replace(
        bundle,
        bundle_sha256=_json_sha256(_fixed_bundle_body(bundle)),
    )


def rebuild_lfm2_variable_width_fixed_plan_bundle_from_layer_payload(
    parent: Any,
    evidence: Lfm2VariableWidthEvidence,
    target_widths: Sequence[int],
    canonical_layer_plan: Lfm2VariableWidthLayerPlan,
    *,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    expected_plan_sha256: str | None = None,
    expected_layer_binding_sha256: str | None = None,
    expected_bundle_sha256: str | None = None,
) -> Lfm2VariableWidthFixedPlanBundle:
    """Rebuild one isolated plan from an exact canonical changed-layer payload.

    No clustering, medoid selection, source assignment, coefficient fitting,
    or consumer aggregation is performed here.  The one contracted layer is
    supplied byte-exactly by a previously sealed canonical producer.  Every
    unchanged layer is cloned from the authenticated parent, so the resulting
    full plan and bundle must reproduce the canonical seals exactly.
    """

    torch = _torch()
    ridge = _finite_nonnegative(cluster_ridge, "cluster_ridge")
    iterations = _positive_int(
        cluster_max_iterations,
        "cluster_max_iterations",
    )
    chunk_size = _positive_int(
        cluster_distance_chunk_size,
        "cluster_distance_chunk_size",
    )
    widths = tuple(target_widths)
    _validate_live_evidence(parent, evidence)
    if len(widths) != LFM2_VARIABLE_WIDTH_LAYER_COUNT or any(
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or width > evidence.source_intermediate_width
        for width in widths
    ):
        raise Lfm2VariableWidthAllocationError(
            "target_widths must contain fourteen positive contractions."
        )
    contracted = tuple(
        index
        for index, width in enumerate(widths)
        if width != evidence.source_intermediate_width
    )
    if (
        len(contracted) != 1
        or not isinstance(canonical_layer_plan, Lfm2VariableWidthLayerPlan)
        or canonical_layer_plan.layer_index != contracted[0]
        or canonical_layer_plan.source_width != evidence.source_intermediate_width
        or canonical_layer_plan.target_width != widths[contracted[0]]
    ):
        raise Lfm2VariableWidthTransactionError(
            "Canonical reconstruction requires exactly one matching "
            "contracted-layer payload."
        )

    fixed_curves = []
    for layer_index, (curve, target_width) in enumerate(
        zip(evidence.curves, widths, strict=True)
    ):
        by_width = {point.width: point for point in curve.points}
        if target_width not in by_width:
            raise Lfm2VariableWidthAllocationError(
                f"Layer {layer_index} has no sealed causal point at width "
                f"{target_width}."
            )
        source_point = by_width[evidence.source_intermediate_width]
        points = (
            (source_point,)
            if target_width == evidence.source_intermediate_width
            else (source_point, by_width[target_width])
        )
        fixed_curves.append(
            Lfm2LayerWidthCurve(
                layer_index=layer_index,
                points=points,
            )
        )
    widths_sha256 = _json_sha256(
        {
            "source_evidence_sha256": evidence.evidence_sha256,
            "target_widths": list(widths),
        }
    )
    fixed = replace(
        evidence,
        curves=tuple(fixed_curves),
        evidence_builder_id=(
            f"{evidence.evidence_builder_id}:fixed-widths:{widths_sha256}"
        ),
        evidence_sha256="",
    )
    fixed = replace(
        fixed,
        evidence_sha256=_json_sha256(_evidence_body(fixed)),
    )

    with _TRANSACTION_LOCK, torch.no_grad():
        layers, hidden, source_width, kinds = _validate_live_evidence(
            parent,
            fixed,
        )
        parent_before = _parent_identity(parent)
        parameters_saved = sum((source_width - width) * 3 * hidden for width in widths)
        maximum_step = max(1, max(source_width - width for width in widths))
        allocation = allocate_lfm2_variable_widths(
            fixed,
            target_parameters_saved=parameters_saved,
            minimum_width=min(widths),
            maximum_width_step=maximum_step,
            whole_trajectory_weight=1.0,
            residual_weight=1.0,
        )
        if allocation.target_widths != widths:
            raise Lfm2VariableWidthTransactionError(
                "Canonical fixed evidence did not reproduce its width vector."
            )
        layer_plans = []
        for index, (layer, target_width) in enumerate(zip(layers, widths, strict=True)):
            gate_source = layer.feed_forward.w1.weight
            up_source = layer.feed_forward.w3.weight
            down_source = layer.feed_forward.w2.weight
            requires_grad = (
                bool(gate_source.requires_grad),
                bool(up_source.requires_grad),
                bool(down_source.requires_grad),
            )
            if index == canonical_layer_plan.layer_index:
                if canonical_layer_plan.requires_grad != requires_grad:
                    raise Lfm2VariableWidthTransactionError(
                        "Canonical payload gradient metadata differs from parent."
                    )
                layer_plan = Lfm2VariableWidthLayerPlan(
                    layer_index=index,
                    source_width=source_width,
                    target_width=target_width,
                    medoid_source_neurons=tuple(
                        canonical_layer_plan.medoid_source_neurons
                    ),
                    source_to_target=tuple(canonical_layer_plan.source_to_target),
                    consumer_coefficients=tuple(
                        canonical_layer_plan.consumer_coefficients
                    ),
                    gate_weight=_clone_cpu(
                        canonical_layer_plan.gate_weight,
                        f"canonical.layer.{index}.gate",
                    ),
                    up_weight=_clone_cpu(
                        canonical_layer_plan.up_weight,
                        f"canonical.layer.{index}.up",
                    ),
                    down_weight=_clone_cpu(
                        canonical_layer_plan.down_weight,
                        f"canonical.layer.{index}.down",
                    ),
                    requires_grad=requires_grad,
                )
            else:
                if target_width != source_width:
                    raise Lfm2VariableWidthTransactionError(
                        "Canonical payload omitted a contracted layer."
                    )
                layer_plan = Lfm2VariableWidthLayerPlan(
                    layer_index=index,
                    source_width=source_width,
                    target_width=source_width,
                    medoid_source_neurons=tuple(range(source_width)),
                    source_to_target=tuple(range(source_width)),
                    consumer_coefficients=(1.0,) * source_width,
                    gate_weight=_clone_cpu(
                        gate_source,
                        f"plan.layer.{index}.gate",
                    ),
                    up_weight=_clone_cpu(
                        up_source,
                        f"plan.layer.{index}.up",
                    ),
                    down_weight=_clone_cpu(
                        down_source,
                        f"plan.layer.{index}.down",
                    ),
                    requires_grad=requires_grad,
                )
            layer_plans.append(layer_plan)

        source_parameters = _unique_parameter_count(parent)
        plan = Lfm2VariableWidthFoldPlan(
            schema=PLAN_SCHEMA,
            method_id=METHOD_ID,
            parent_identity_sha256=evidence.parent_identity_sha256,
            evidence_sha256=fixed.evidence_sha256,
            allocation=allocation,
            layer_plans=tuple(layer_plans),
            source_unique_parameters=source_parameters,
            target_unique_parameters=source_parameters - parameters_saved,
            parameters_saved=parameters_saved,
            hidden_size=hidden,
            source_intermediate_width=source_width,
            layer_kinds=kinds,
            cluster_ridge=ridge,
            cluster_max_iterations=iterations,
            cluster_distance_chunk_size=chunk_size,
            plan_sha256="",
        )
        plan = replace(plan, plan_sha256=_json_sha256(_plan_body(plan)))
        _validate_plan_seal(plan)
        layer_bindings = tuple(
            fingerprint_lfm2_variable_width_layer_plan(layer_plan)
            for layer_plan in plan.layer_plans
        )
        bundle = Lfm2VariableWidthFixedPlanBundle(
            schema=FIXED_WIDTH_BUNDLE_SCHEMA,
            source_evidence_sha256=evidence.evidence_sha256,
            fixed_evidence=fixed,
            plan=plan,
            layer_binding_sha256=layer_bindings,
            bundle_sha256="",
        )
        bundle = replace(
            bundle,
            bundle_sha256=_json_sha256(_fixed_bundle_body(bundle)),
        )
        verify_lfm2_variable_width_fixed_plan_bundle(
            bundle,
            parent=parent,
        )
        expected_layer = layer_bindings[canonical_layer_plan.layer_index]
        if (
            (
                expected_plan_sha256 is not None
                and plan.plan_sha256 != expected_plan_sha256
            )
            or (
                expected_layer_binding_sha256 is not None
                and expected_layer != expected_layer_binding_sha256
            )
            or (
                expected_bundle_sha256 is not None
                and bundle.bundle_sha256 != expected_bundle_sha256
            )
        ):
            raise Lfm2VariableWidthTransactionError(
                "Canonical layer payload did not reproduce the sealed plan."
            )
        if _parent_identity(parent) != parent_before:
            raise Lfm2VariableWidthTransactionError(
                "Canonical plan reconstruction mutated the parent."
            )
        return bundle


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-variable-width-state-v1\0")
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            _canonical_json(
                _tensor_record(
                    value,
                    f"state.{name}",
                    require_floating=value.is_floating_point(),
                )
            )
        )
    return digest.hexdigest()


def _named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - supported PyTorch versions expose it
        return tuple(model.named_parameters())


def _named_buffers(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(model.named_buffers(remove_duplicate=False))
    except TypeError:  # pragma: no cover
        return tuple(model.named_buffers())


def _runtime_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-variable-width-runtime-v1\0")
    for category, values in (
        ("parameter", _named_parameters(model)),
        ("buffer", _named_buffers(model)),
    ):
        for name, value in values:
            digest.update(category.encode("ascii"))
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                _canonical_json(
                    _tensor_record(
                        value,
                        f"runtime.{category}.{name}",
                        require_floating=value.is_floating_point(),
                    )
                )
            )
    return digest.hexdigest()


@dataclass(slots=True)
class _ChildSnapshot:
    state: Mapping[str, Any]
    buffers: Mapping[str, Any]
    feed_forwards: tuple[Any, ...]
    parameter_ids: tuple[tuple[str, int], ...]
    buffer_ids: tuple[tuple[str, int], ...]
    state_sha256: str
    runtime_sha256: str


def _snapshot_child(child: Any, layers: Sequence[Any]) -> _ChildSnapshot:
    state = MappingProxyType(
        {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in child.state_dict().items()
        }
    )
    buffers = MappingProxyType(
        {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in _named_buffers(child)
        }
    )
    return _ChildSnapshot(
        state=state,
        buffers=buffers,
        feed_forwards=tuple(layer.feed_forward for layer in layers),
        parameter_ids=tuple(
            (name, id(value)) for name, value in _named_parameters(child)
        ),
        buffer_ids=tuple((name, id(value)) for name, value in _named_buffers(child)),
        state_sha256=_state_sha256(child),
        runtime_sha256=_runtime_sha256(child),
    )


def _restore_child(
    child: Any,
    layers: Sequence[Any],
    snapshot: _ChildSnapshot,
) -> None:
    torch = _torch()
    for layer, feed_forward in zip(
        layers,
        snapshot.feed_forwards,
        strict=True,
    ):
        layer.feed_forward = feed_forward
    state = child.state_dict()
    buffers = dict(_named_buffers(child))
    if set(state) != set(snapshot.state) or set(buffers) != set(snapshot.buffers):
        raise Lfm2VariableWidthTransactionError(
            "Rollback topology no longer matches the blank child."
        )
    with torch.no_grad():
        for name, target in state.items():
            source = snapshot.state[name]
            if (
                tuple(target.shape) != tuple(source.shape)
                or target.dtype != source.dtype
            ):
                raise Lfm2VariableWidthTransactionError(
                    f"Rollback state contract changed for {name!r}."
                )
            target.copy_(source.to(device=target.device))
        for name, target in buffers.items():
            source = snapshot.buffers[name]
            if (
                tuple(target.shape) != tuple(source.shape)
                or target.dtype != source.dtype
            ):
                raise Lfm2VariableWidthTransactionError(
                    f"Rollback buffer contract changed for {name!r}."
                )
            target.copy_(source.to(device=target.device))
    if (
        tuple((name, id(value)) for name, value in _named_parameters(child))
        != snapshot.parameter_ids
        or tuple((name, id(value)) for name, value in _named_buffers(child))
        != snapshot.buffer_ids
        or _state_sha256(child) != snapshot.state_sha256
        or _runtime_sha256(child) != snapshot.runtime_sha256
    ):
        raise Lfm2VariableWidthTransactionError(
            "Rollback did not restore the exact blank objects and bytes."
        )


def _is_folded_ffn_weight(name: str) -> bool:
    return any(
        name.endswith(f".feed_forward.{projection}.weight")
        for projection in _FFN_PROJECTIONS
    )


def _make_linear_from_payload(
    source_linear: Any,
    payload: Any,
    *,
    in_features: int,
    out_features: int,
    device: Any,
    dtype: Any,
    requires_grad: bool,
) -> Any:
    torch = _torch()
    result = copy.deepcopy(source_linear)
    _clear_copied_forward_hooks(result)
    result.in_features = in_features
    result.out_features = out_features
    result.bias = None
    result.weight = torch.nn.Parameter(
        payload.detach().to(device=device, dtype=dtype).contiguous().clone(),
        requires_grad=requires_grad,
    )
    return result


def _clear_copied_forward_hooks(module: Any) -> None:
    """Keep external parent instrumentation out of physical child modules."""

    for nested in module.modules():
        for name in _FORWARD_HOOK_ATTRIBUTES:
            hooks = getattr(nested, name, None)
            if hooks is not None and hasattr(hooks, "clear"):
                hooks.clear()


def _build_replacement_ffn(
    source_ffn: Any,
    blank_ffn: Any,
    layer_plan: Lfm2VariableWidthLayerPlan,
) -> Any:
    result = copy.deepcopy(source_ffn)
    _clear_copied_forward_hooks(result)
    target_device = blank_ffn.w1.weight.device
    target_dtype = blank_ffn.w1.weight.dtype
    if any(
        getattr(blank_ffn, projection).weight.device != target_device
        or getattr(blank_ffn, projection).weight.dtype != target_dtype
        for projection in _FFN_PROJECTIONS
    ):
        raise Lfm2VariableWidthTopologyError(
            "A blank FFN's projections must share one device and dtype."
        )
    result.w1 = _make_linear_from_payload(
        source_ffn.w1,
        layer_plan.gate_weight,
        in_features=int(layer_plan.gate_weight.shape[1]),
        out_features=int(layer_plan.gate_weight.shape[0]),
        device=target_device,
        dtype=target_dtype,
        requires_grad=layer_plan.requires_grad[0],
    )
    result.w3 = _make_linear_from_payload(
        source_ffn.w3,
        layer_plan.up_weight,
        in_features=int(layer_plan.up_weight.shape[1]),
        out_features=int(layer_plan.up_weight.shape[0]),
        device=target_device,
        dtype=target_dtype,
        requires_grad=layer_plan.requires_grad[1],
    )
    result.w2 = _make_linear_from_payload(
        source_ffn.w2,
        layer_plan.down_weight,
        in_features=int(layer_plan.down_weight.shape[1]),
        out_features=int(layer_plan.down_weight.shape[0]),
        device=target_device,
        dtype=target_dtype,
        requires_grad=layer_plan.requires_grad[2],
    )
    result.training = bool(blank_ffn.training)
    return result


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthCorrectionContext:
    """Read-only metadata plus the complete live candidate for one correction."""

    parent: Any
    child: Any
    layer_index: int
    source_width: int
    target_width: int
    medoid_source_neurons: tuple[int, ...]
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthResidualCorrection:
    """One externally derived, sealed residual endpoint correction for ``w2``."""

    down_weight: Any
    evidence_sha256: str
    method_id: str


@dataclass(frozen=True, slots=True)
class Lfm2VariableWidthFoldReport:
    """Authenticated live accounting for an installed candidate."""

    schema: str
    method_id: str
    parent_identity_sha256: str
    evidence_sha256: str
    plan_sha256: str
    source_widths: tuple[int, ...]
    target_widths: tuple[int, ...]
    unique_parameters_before: int
    unique_parameters_after: int
    parameters_saved: int
    compression_fraction: float
    candidate_runtime_sha256: str
    correction_reports: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "parent_identity_sha256": self.parent_identity_sha256,
            "evidence_sha256": self.evidence_sha256,
            "plan_sha256": self.plan_sha256,
            "source_widths": list(self.source_widths),
            "target_widths": list(self.target_widths),
            "unique_parameters_before": self.unique_parameters_before,
            "unique_parameters_after": self.unique_parameters_after,
            "parameters_saved": self.parameters_saved,
            "compression_fraction": self.compression_fraction,
            "candidate_runtime_sha256": self.candidate_runtime_sha256,
            "correction_reports": [dict(value) for value in self.correction_reports],
            "same_shape_parent_tensors_preserved_exactly": True,
            "real_parent_medoid_producers": True,
            "blank_values_used_for_candidate": False,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "checkpoint_serialization_supported": False,
        }


class InstalledLfm2VariableWidthFold:
    """Live nonuniform candidate with explicit commit or exact rollback."""

    def __init__(
        self,
        *,
        child: Any,
        layers: tuple[Any, ...],
        snapshot: _ChildSnapshot,
        report: Lfm2VariableWidthFoldReport,
    ) -> None:
        self.child = child
        self.report = report
        self._layers = layers
        self._snapshot = snapshot
        self._active = True
        self._committed = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def committed(self) -> bool:
        return self._committed

    def commit(self) -> Lfm2VariableWidthFoldReport:
        with _TRANSACTION_LOCK:
            if not self._active:
                raise Lfm2VariableWidthTransactionError(
                    "The variable-width transaction is no longer active."
                )
            if _runtime_sha256(self.child) != self.report.candidate_runtime_sha256:
                raise Lfm2VariableWidthTransactionError(
                    "The variable-width candidate changed before commit."
                )
            if hasattr(self.child, _MARKER):
                delattr(self.child, _MARKER)
            self._active = False
            self._committed = True
            return self.report

    def rollback(self) -> bool:
        with _TRANSACTION_LOCK:
            if not self._active:
                return False
            _restore_child(self.child, self._layers, self._snapshot)
            if hasattr(self.child, _MARKER):
                delattr(self.child, _MARKER)
            self._active = False
            return True

    def __enter__(self) -> InstalledLfm2VariableWidthFold:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active:
            self.rollback()


def _validate_blank_child(
    parent: Any,
    child: Any,
    plan: Lfm2VariableWidthFoldPlan,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if parent is child:
        raise Lfm2VariableWidthTopologyError(
            "Parent and blank child must be distinct models."
        )
    if hasattr(child, _MARKER):
        raise Lfm2VariableWidthTransactionError(
            "The child already has an active variable-width transaction."
        )
    parent_layers, parent_hidden, parent_width, parent_kinds = _inspect_dense_model(
        parent
    )
    child_layers, child_hidden, child_width, child_kinds = _inspect_dense_model(child)
    if (
        parent_hidden != child_hidden
        or parent_width != child_width
        or parent_kinds != child_kinds
        or parent_hidden != plan.hidden_size
        or parent_width != plan.source_intermediate_width
        or parent_kinds != plan.layer_kinds
        or _unique_parameter_count(child) != plan.source_unique_parameters
    ):
        raise Lfm2VariableWidthTopologyError(
            "The blank child does not match the plan's full-depth source shell."
        )
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    if set(parent_state) != set(child_state):
        raise Lfm2VariableWidthTopologyError(
            "Parent and blank child state keys differ."
        )
    for name, target in child_state.items():
        source = parent_state[name]
        if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
            raise Lfm2VariableWidthTopologyError(
                f"Parent/blank state contract differs for {name!r}."
            )
    return parent_layers, child_layers


def _validate_correction(
    correction: Lfm2VariableWidthResidualCorrection,
    *,
    layer_index: int,
    hidden_size: int,
    target_width: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(correction, Lfm2VariableWidthResidualCorrection)
        or not _valid_sha256(correction.evidence_sha256)
        or not isinstance(correction.method_id, str)
        or not correction.method_id
        or not isinstance(correction.down_weight, torch.Tensor)
        or tuple(correction.down_weight.shape) != (hidden_size, target_width)
        or not correction.down_weight.is_floating_point()
        or not bool(torch.isfinite(correction.down_weight).all().item())
    ):
        raise Lfm2VariableWidthTransactionError(
            f"Residual correction for layer {layer_index} is invalid."
        )
    return correction.down_weight


def install_lfm2_variable_width_fold_plan(
    parent: Any,
    child: Any,
    evidence: Lfm2VariableWidthEvidence,
    plan: Lfm2VariableWidthFoldPlan,
    *,
    residual_corrector: (
        Callable[
            [Lfm2VariableWidthCorrectionContext],
            Lfm2VariableWidthResidualCorrection | None,
        ]
        | None
    ) = None,
) -> InstalledLfm2VariableWidthFold:
    """Atomically install one source-only nonuniform candidate in memory."""

    torch = _torch()
    if residual_corrector is not None and not callable(residual_corrector):
        raise TypeError("residual_corrector must be callable or None.")
    with _TRANSACTION_LOCK, torch.no_grad():
        _validate_live_evidence(parent, evidence)
        _validate_plan_seal(plan)
        if (
            plan.parent_identity_sha256 != evidence.parent_identity_sha256
            or plan.evidence_sha256 != evidence.evidence_sha256
            or plan.allocation.evidence_sha256 != evidence.evidence_sha256
        ):
            raise Lfm2VariableWidthTransactionError(
                "The variable-width plan is stale for this evidence."
            )
        parent_layers, child_layers = _validate_blank_child(
            parent,
            child,
            plan,
        )
        snapshot = _snapshot_child(child, child_layers)
        parent_before = _parent_identity(parent)
        replacements = tuple(
            _build_replacement_ffn(
                parent_layers[layer_plan.layer_index].feed_forward,
                child_layers[layer_plan.layer_index].feed_forward,
                layer_plan,
            )
            for layer_plan in plan.layer_plans
        )
        # Tensor-bearing frozen plans remain mutable through their tensors.
        # Recheck immediately before the first possible child write.
        _validate_plan_seal(plan)
        mutated = False
        try:
            parent_state = parent.state_dict()
            child_state = child.state_dict()
            # The very next operation is the first child write.  Mark the
            # transaction dirty before it so even a mid-copy BaseException
            # restores the exact blank.
            mutated = True
            for name, target in child_state.items():
                if _is_folded_ffn_weight(name):
                    continue
                source = parent_state[name]
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            for layer, replacement in zip(
                child_layers,
                replacements,
                strict=True,
            ):
                layer.feed_forward = replacement
            setattr(child, _MARKER, plan.plan_sha256)

            candidate_state = child.state_dict()
            if set(candidate_state) != set(parent_state):
                raise Lfm2VariableWidthTransactionError(
                    "The nonuniform candidate changed the model state-key set."
                )
            for name, target in candidate_state.items():
                if _is_folded_ffn_weight(name):
                    continue
                source = parent_state[name]
                if not torch.equal(
                    target.detach().to(device="cpu"),
                    source.detach().to(device="cpu"),
                ):
                    raise Lfm2VariableWidthTransactionError(
                        f"Same-shape parent tensor {name!r} was not preserved."
                    )
            observed_after_base = _unique_parameter_count(child)
            if observed_after_base != plan.target_unique_parameters:
                raise Lfm2VariableWidthTransactionError(
                    "Live parameter accounting differs from the sealed plan."
                )

            correction_reports = []
            if residual_corrector is not None:
                for layer_plan in plan.layer_plans:
                    before_callback = _runtime_sha256(child)
                    correction = residual_corrector(
                        Lfm2VariableWidthCorrectionContext(
                            parent=parent,
                            child=child,
                            layer_index=layer_plan.layer_index,
                            source_width=layer_plan.source_width,
                            target_width=layer_plan.target_width,
                            medoid_source_neurons=(layer_plan.medoid_source_neurons),
                            plan_sha256=plan.plan_sha256,
                        )
                    )
                    if _runtime_sha256(child) != before_callback:
                        raise Lfm2VariableWidthTransactionError(
                            "A residual corrector mutated a model directly."
                        )
                    if correction is None:
                        correction_reports.append(
                            MappingProxyType(
                                {
                                    "layer_index": layer_plan.layer_index,
                                    "supplied": False,
                                }
                            )
                        )
                        continue
                    down = _validate_correction(
                        correction,
                        layer_index=layer_plan.layer_index,
                        hidden_size=plan.hidden_size,
                        target_width=layer_plan.target_width,
                    )
                    target = child_layers[layer_plan.layer_index].feed_forward.w2.weight
                    target.copy_(
                        down.detach().to(
                            device=target.device,
                            dtype=target.dtype,
                        )
                    )
                    correction_reports.append(
                        MappingProxyType(
                            {
                                "layer_index": layer_plan.layer_index,
                                "supplied": True,
                                "method_id": correction.method_id,
                                "evidence_sha256": (correction.evidence_sha256),
                                "down_weight": _tensor_record(
                                    target,
                                    (
                                        "installed.correction."
                                        f"{layer_plan.layer_index}.down"
                                    ),
                                ),
                            }
                        )
                    )

            if (
                _unique_parameter_count(child) != plan.target_unique_parameters
                or any(parameter.grad is not None for parameter in child.parameters())
                or _parent_identity(parent) != parent_before
            ):
                raise Lfm2VariableWidthTransactionError(
                    "Final accounting, zero-gradient, or parent-identity audit failed."
                )
            candidate_runtime = _runtime_sha256(child)
            report = Lfm2VariableWidthFoldReport(
                schema=REPORT_SCHEMA,
                method_id=METHOD_ID,
                parent_identity_sha256=parent_before,
                evidence_sha256=evidence.evidence_sha256,
                plan_sha256=plan.plan_sha256,
                source_widths=plan.allocation.source_widths,
                target_widths=plan.target_widths,
                unique_parameters_before=plan.source_unique_parameters,
                unique_parameters_after=plan.target_unique_parameters,
                parameters_saved=plan.parameters_saved,
                compression_fraction=(
                    plan.parameters_saved / plan.source_unique_parameters
                ),
                candidate_runtime_sha256=candidate_runtime,
                correction_reports=tuple(correction_reports),
            )
            return InstalledLfm2VariableWidthFold(
                child=child,
                layers=child_layers,
                snapshot=snapshot,
                report=report,
            )
        except BaseException:
            if mutated or hasattr(child, _MARKER):
                try:
                    _restore_child(child, child_layers, snapshot)
                    if hasattr(child, _MARKER):
                        delattr(child, _MARKER)
                except BaseException as rollback_exc:
                    raise Lfm2VariableWidthTransactionError(
                        "Variable-width installation failed and exact rollback "
                        "could not be authenticated."
                    ) from rollback_exc
            raise


def install_lfm2_variable_width_fold(
    parent: Any,
    child: Any,
    evidence: Lfm2VariableWidthEvidence,
    *,
    residual_corrector: (
        Callable[
            [Lfm2VariableWidthCorrectionContext],
            Lfm2VariableWidthResidualCorrection | None,
        ]
        | None
    ) = None,
    **plan_kwargs: Any,
) -> InstalledLfm2VariableWidthFold:
    """Plan and install one dynamic variable-width LFM2 candidate."""

    plan = plan_lfm2_variable_width_fold(
        parent,
        evidence,
        **plan_kwargs,
    )
    return install_lfm2_variable_width_fold_plan(
        parent,
        child,
        evidence,
        plan,
        residual_corrector=residual_corrector,
    )


__all__ = [
    "ALLOCATION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "FIXED_WIDTH_BUNDLE_SCHEMA",
    "LAYER_BINDING_SCHEMA",
    "LFM2_VARIABLE_WIDTH_LAYER_COUNT",
    "METHOD_ID",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "InstalledLfm2VariableWidthFold",
    "Lfm2LayerWidthCurve",
    "Lfm2VariableWidthAllocation",
    "Lfm2VariableWidthAllocationError",
    "Lfm2VariableWidthCorrectionContext",
    "Lfm2VariableWidthEvidence",
    "Lfm2VariableWidthEvidenceError",
    "Lfm2VariableWidthFoldError",
    "Lfm2VariableWidthFoldPlan",
    "Lfm2VariableWidthFoldReport",
    "Lfm2VariableWidthFixedPlanBundle",
    "Lfm2VariableWidthLayerActivations",
    "Lfm2VariableWidthLayerPlan",
    "Lfm2VariableWidthResidualCorrection",
    "Lfm2VariableWidthTopologyError",
    "Lfm2VariableWidthTransactionError",
    "Lfm2WidthCostPoint",
    "allocate_lfm2_variable_widths",
    "build_lfm2_variable_width_fold_plan",
    "install_lfm2_variable_width_fold",
    "install_lfm2_variable_width_fold_plan",
    "fingerprint_lfm2_variable_width_layer_plan",
    "plan_lfm2_variable_width_fold",
    "plan_lfm2_variable_width_fold_for_widths",
    "rebuild_lfm2_variable_width_fixed_plan_bundle_from_layer_payload",
    "seal_lfm2_variable_width_evidence",
    "verify_lfm2_variable_width_fixed_plan_bundle",
    "verify_lfm2_variable_width_evidence",
]
