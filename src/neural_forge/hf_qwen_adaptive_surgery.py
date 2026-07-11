"""Transactional adaptive surgery for prepared Qwen3 neuron codebooks.

This module is intentionally Qwen3-specific.  It translates bounded
post-SwiGLU evidence for real Qwen3 MLP neuron bundles, plus optional bounded
GQA context evidence, into :class:`AdaptiveSurgeryGraph` nodes and applies
accepted graph assignments to the prepared Qwen3 *neuron* codebook runtime.
It does not rewrite Qwen3 attention/GQA codebook assignments: GQA evidence is
context for scoring MLP-bundle surgery, not a claim of attention surgery.  A
tentative change is installed as a complete replacement codebook; the previous
registry is never mutated, so a rejected rollout gate can restore the exact
prior registry and all parameterless layer views by identity.

All feature handling, prototype selection, scalar anchor projection, and
registry rebuilding runs on detached tensors under ``torch.inference_mode``.
There is no optimizer, autograd graph, or backward pass in this bridge.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from neural_forge.adaptive_surgery_graph import (
    ActionRecord,
    AdaptiveSurgeryGraph,
    AnchorSpec,
    EvaluationDecision,
    MergeCandidate,
    ProjectionBinding,
    ScoreWeights,
    SurgeryNode,
    SurgerySearchExhausted,
    SurgerySnapshot,
    SurgeryTransaction,
)
from neural_forge.hf_neuron_codebook import (
    GlobalNeuronCodebook,
    NeuronCodebookRegistry,
)
from neural_forge.hf_qwen_codebook_model import (
    Qwen3CodebookError,
    Qwen3CodebookStateError,
    qwen3_codebook_parameter_report,
)


SCHEMA_VERSION = 1
QWEN3_NODE_KIND = "mlp-neuron"
Qwen3NodeId = tuple[str, str, int, int]
Gate = Callable[..., bool | EvaluationDecision | Mapping[str, Any]]
FeatureRecompute = Callable[..., Any]
CandidateSelector = Callable[..., Any]


class Qwen3AdaptiveSurgeryError(Qwen3CodebookError):
    """Base error for Qwen3 adaptive-surgery validation and execution."""


class Qwen3AdaptiveSurgeryShapeError(Qwen3AdaptiveSurgeryError):
    """A Qwen3 feature, node reference, or runtime tensor has a bad shape."""


class Qwen3AdaptiveSurgeryStateError(
    Qwen3AdaptiveSurgeryError, Qwen3CodebookStateError
):
    """The prepared Qwen3 runtime cannot support an exact transaction."""


class Qwen3AdaptiveSurgeryTargetError(Qwen3AdaptiveSurgeryError):
    """The requested physical neuron-codebook byte target cannot be reached."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a non-negative integer."
        )
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a non-negative integer."
        ) from error
    if result < 0:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a non-negative integer."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result <= 0:
        raise Qwen3AdaptiveSurgeryShapeError(f"{name} must be positive.")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a finite positive number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a finite positive number."
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a finite positive number."
        )
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Qwen3AdaptiveSurgeryShapeError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a finite number."
        ) from error
    if not math.isfinite(result):
        raise Qwen3AdaptiveSurgeryShapeError(f"{name} must be finite.")
    return result


def _feature_vector(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        raise Qwen3AdaptiveSurgeryShapeError(f"{name} is required.")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            value = numpy_method()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must be a numeric feature vector."
        ) from error
    if array.ndim == 0 or array.size == 0:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must contain at least one value."
        )
    flat = np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
    if not bool(np.isfinite(flat).all()):
        raise Qwen3AdaptiveSurgeryShapeError(
            f"{name} must contain only finite values."
        )
    return tuple(float(item) for item in flat)


def _optional_feature_vector(value: Any, name: str) -> tuple[float, ...] | None:
    return None if value is None else _feature_vector(value, name)


def _neutral_optional_feature(value: Any, name: str) -> tuple[float, ...]:
    """Represent unavailable JVP/output evidence with a bounded neutral channel."""

    return (0.0,) if value is None else _feature_vector(value, name)


def _coalesce(primary: Any, alias: Any, primary_name: str, alias_name: str) -> Any:
    if primary is not None and alias is not None:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"Provide only one of {primary_name} and {alias_name}."
        )
    return primary if primary is not None else alias


def _anchor_spec(value: AnchorSpec | str | None) -> AnchorSpec | None:
    if value is None or isinstance(value, AnchorSpec):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"exact", "pinned", "hard"}:
        return AnchorSpec.pinned()
    if normalized in {"projected", "projection"}:
        return AnchorSpec.projected()
    if normalized == "soft":
        return AnchorSpec.soft()
    raise Qwen3AdaptiveSurgeryShapeError(f"Unknown anchor mode {value!r}.")


def qwen3_neuron_node_id(layer_index: Any, neuron_index: Any) -> Qwen3NodeId:
    """Return the canonical node ID for one Qwen3 MLP neuron execution."""

    return (
        "qwen3",
        QWEN3_NODE_KIND,
        _nonnegative_int(layer_index, "layer_index"),
        _nonnegative_int(neuron_index, "neuron_index"),
    )


def _node_ref(value: Any, name: str = "node") -> Qwen3NodeId:
    if isinstance(value, Qwen3AdaptiveSurgeryFeature):
        return value.node_id
    if isinstance(value, tuple) and len(value) == 2:
        return qwen3_neuron_node_id(value[0], value[1])
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and value[:2] == ("qwen3", QWEN3_NODE_KIND)
    ):
        return qwen3_neuron_node_id(value[2], value[3])
    raise Qwen3AdaptiveSurgeryShapeError(
        f"{name} must be (layer, neuron) or a canonical Qwen3 node ID."
    )


@dataclass(frozen=True, slots=True, init=False)
class Qwen3AdaptiveSurgeryFeature:
    """Bounded evidence for one real Qwen3 gate/up/down neuron bundle.

    ``activation`` is real post-SwiGLU evidence.  ``gqa_features`` is optional
    bounded attention/GQA context used only in graph scoring; it never changes
    Qwen3 attention-codebook assignments.  Omitted JVP or output/dual evidence
    becomes a shared neutral scalar, so its score contribution is zero.
    """

    layer_index: int
    neuron_index: int
    weight_features: tuple[float, ...] | None
    activation_features: tuple[float, ...]
    jacobian_features: tuple[float, ...]
    dual_features: tuple[float, ...]
    domain_features: tuple[float, ...] | None
    importance: float
    anchor: AnchorSpec | None

    def __init__(
        self,
        layer_index: Any,
        neuron_index: Any = None,
        weight: Any = None,
        activation: Any = None,
        jacobian: Any = None,
        dual: Any = None,
        importance: float = 1.0,
        anchor: AnchorSpec | str | None = None,
        *,
        feature_index: Any = None,
        weight_features: Any = None,
        activation_features: Any = None,
        jacobian_features: Any = None,
        rollout_jvp_features: Any = None,
        dual_features: Any = None,
        output_dual_features: Any = None,
        domain_features: Any = None,
        gqa_features: Any = None,
    ) -> None:
        neuron = _coalesce(
            neuron_index, feature_index, "neuron_index", "feature_index"
        )
        if neuron is None:
            raise Qwen3AdaptiveSurgeryShapeError(
                "neuron_index or feature_index is required."
            )
        weight_value = _coalesce(
            weight, weight_features, "weight", "weight_features"
        )
        activation_value = _coalesce(
            activation,
            activation_features,
            "activation",
            "activation_features",
        )
        jacobian_alias = _coalesce(
            jacobian_features,
            rollout_jvp_features,
            "jacobian_features",
            "rollout_jvp_features",
        )
        jacobian_value = _coalesce(
            jacobian, jacobian_alias, "jacobian", "jacobian feature alias"
        )
        dual_alias = _coalesce(
            dual_features,
            output_dual_features,
            "dual_features",
            "output_dual_features",
        )
        dual_value = _coalesce(dual, dual_alias, "dual", "dual feature alias")
        context_value = _coalesce(
            domain_features, gqa_features, "domain_features", "gqa_features"
        )

        object.__setattr__(
            self, "layer_index", _nonnegative_int(layer_index, "layer_index")
        )
        object.__setattr__(
            self, "neuron_index", _nonnegative_int(neuron, "neuron_index")
        )
        object.__setattr__(
            self,
            "weight_features",
            _optional_feature_vector(weight_value, "weight features"),
        )
        object.__setattr__(
            self,
            "activation_features",
            _feature_vector(activation_value, "activation features"),
        )
        object.__setattr__(
            self,
            "jacobian_features",
            _neutral_optional_feature(jacobian_value, "Jacobian/rollout features"),
        )
        object.__setattr__(
            self,
            "dual_features",
            _neutral_optional_feature(dual_value, "output/dual features"),
        )
        object.__setattr__(
            self,
            "domain_features",
            _optional_feature_vector(context_value, "GQA context features"),
        )
        object.__setattr__(
            self, "importance", _positive_float(importance, "importance")
        )
        object.__setattr__(self, "anchor", _anchor_spec(anchor))

    @property
    def node_id(self) -> Qwen3NodeId:
        return qwen3_neuron_node_id(self.layer_index, self.neuron_index)

    @property
    def feature_index(self) -> int:
        return self.neuron_index

    @property
    def rollout_jvp_features(self) -> tuple[float, ...]:
        return self.jacobian_features

    @property
    def output_dual_features(self) -> tuple[float, ...]:
        return self.dual_features

    @property
    def gqa_features(self) -> tuple[float, ...] | None:
        return self.domain_features


Qwen3SurgeryFeature = Qwen3AdaptiveSurgeryFeature
Qwen3FeatureBundle = Qwen3AdaptiveSurgeryFeature


@dataclass(frozen=True, slots=True, init=False)
class Qwen3ProjectionCoefficients:
    """Executable per-role projection from one real Qwen3 bundle to another.

    The prepared neuron registry can represent one scalar for each of the
    gate, up, and down roles at every logical neuron.  Each value is the
    multiplier installed on the target role, after any existing logical-row
    scale.  This is the full projection surface this bridge claims; arbitrary
    projection matrices are deliberately not represented.
    """

    source: Qwen3NodeId
    target: Qwen3NodeId
    gate: float
    up: float
    down: float

    def __init__(
        self,
        source: Any,
        target: Any,
        gate: Any,
        up: Any,
        down: Any,
    ) -> None:
        object.__setattr__(self, "source", _node_ref(source, "projection source"))
        object.__setattr__(self, "target", _node_ref(target, "projection target"))
        if self.source == self.target:
            raise Qwen3AdaptiveSurgeryShapeError(
                "Projection source and target must be different real bundles."
            )
        object.__setattr__(self, "gate", _finite_float(gate, "gate coefficient"))
        object.__setattr__(self, "up", _finite_float(up, "up coefficient"))
        object.__setattr__(self, "down", _finite_float(down, "down coefficient"))

    @property
    def values(self) -> tuple[float, float, float]:
        return self.gate, self.up, self.down

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": list(self.source),
            "target": list(self.target),
            "gate": self.gate,
            "up": self.up,
            "down": self.down,
        }


@dataclass(frozen=True, slots=True, init=False)
class Qwen3SurgeryEvidenceRefresh:
    """Real post-commit node evidence and executable projection coefficients."""

    features: tuple[Qwen3AdaptiveSurgeryFeature, ...]
    projection_coefficients: tuple[Qwen3ProjectionCoefficients, ...]

    def __init__(
        self,
        features: Sequence[Qwen3AdaptiveSurgeryFeature],
        projection_coefficients: (
            Sequence[Qwen3ProjectionCoefficients] | Mapping[Any, Any]
        ) = (),
    ) -> None:
        object.__setattr__(self, "features", _feature_sequence(features))
        object.__setattr__(
            self,
            "projection_coefficients",
            _projection_sequence(projection_coefficients),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_ids": [list(item.node_id) for item in self.features],
            "projection_coefficients": [
                item.as_dict() for item in self.projection_coefficients
            ],
        }


@dataclass(frozen=True, slots=True)
class Qwen3FeatureBounds:
    """Explicit memory-independent bounds for supplied feature evidence."""

    max_nodes: int = 1_000_000
    max_weight_dimensions: int = 256
    max_activation_dimensions: int = 4_096
    max_jacobian_dimensions: int = 4_096
    max_dual_dimensions: int = 4_096
    max_domain_dimensions: int = 4_096
    max_total_values: int = 64_000_000
    derived_weight_dimensions: int = 32

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Qwen3RuntimeAssignmentSnapshot:
    """Immutable value snapshot of one installed neuron-codebook assignment state."""

    logical_layers: int
    intermediate_width: int
    prototype_count: int
    assignments: tuple[tuple[int, ...], ...]
    prototype_source_indices: tuple[int, ...]
    assignment_source_indices: tuple[tuple[int, ...], ...]
    tensor_bytes: int
    registry_sha256: str

    @property
    def deployable_neuron_codebook_bytes(self) -> int:
        """Exact tensor bytes needed by the installed neuron-codebook registry."""

        return self.tensor_bytes

    @property
    def prototype_source_ids(self) -> tuple[Qwen3NodeId, ...]:
        width = self.intermediate_width
        return tuple(
            qwen3_neuron_node_id(*divmod(source, width))
            for source in self.prototype_source_indices
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_layers": self.logical_layers,
            "intermediate_width": self.intermediate_width,
            "prototype_count": self.prototype_count,
            "assignments": [list(row) for row in self.assignments],
            "prototype_source_indices": list(self.prototype_source_indices),
            "assignment_source_indices": [
                list(row) for row in self.assignment_source_indices
            ],
            "tensor_bytes": self.tensor_bytes,
            "registry_sha256": self.registry_sha256,
        }


@dataclass(frozen=True, slots=True)
class Qwen3RuntimeAssignmentChange:
    """One logical Qwen3 neuron execution selecting a different real bundle."""

    logical_node: Qwen3NodeId
    before_prototype: Qwen3NodeId
    after_prototype: Qwen3NodeId

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_node": list(self.logical_node),
            "before_prototype": list(self.before_prototype),
            "after_prototype": list(self.after_prototype),
        }


@dataclass(frozen=True, slots=True)
class Qwen3RuntimeAudit:
    """Constant-size runtime evidence retained in transaction history."""

    prototype_count: int
    tensor_bytes: int
    registry_sha256: str

    @classmethod
    def from_snapshot(cls, snapshot: Qwen3RuntimeAssignmentSnapshot) -> Qwen3RuntimeAudit:
        return cls(snapshot.prototype_count, snapshot.tensor_bytes, snapshot.registry_sha256)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prototype_count": self.prototype_count,
            "tensor_bytes": self.tensor_bytes,
            "registry_sha256": self.registry_sha256,
        }


@dataclass(frozen=True, slots=True)
class Qwen3ActionAudit:
    """Constant-size action provenance retained after a gate resolution."""

    kind: str
    action_sha256: str


@dataclass(frozen=True, slots=True)
class Qwen3SurgeryEvaluationContext:
    """Tentative installed runtime passed to a parent-relative rollout gate."""

    runtime: nn.Module
    round_index: int
    target_bytes: int
    parent_graph_snapshot: SurgerySnapshot
    proposed_graph_snapshot: SurgerySnapshot
    graph_transaction: SurgeryTransaction
    parent_runtime_snapshot: Qwen3RuntimeAssignmentSnapshot
    proposed_runtime_snapshot: Qwen3RuntimeAssignmentSnapshot


@dataclass(frozen=True, slots=True)
class Qwen3SurgeryRecomputeContext:
    """Accepted-only context for recomputing rollout-sensitive node evidence."""

    runtime: nn.Module
    round_index: int
    accepted_transaction_index: int
    target_bytes: int
    current_bytes: int
    active_node_ids: tuple[Qwen3NodeId, ...]
    graph_transaction: SurgeryTransaction


@dataclass(frozen=True, slots=True)
class Qwen3SurgeryCandidateSelectionContext:
    """Bounded current-graph input for an optional candidate selection policy.

    A selector may return ranked discovered ``MergeCandidate`` objects, explicit
    ``(left_node_id, right_node_id)`` pairs, or a mapping with ``candidates``
    and/or ``pairs``.  Explicit pairs are scored through the graph and are
    capped by ``max_nonconflicting_batch_size``; this keeps callers such as a
    meta-compiler decoupled from this Qwen-specific bridge.
    """

    graph: AdaptiveSurgeryGraph
    discovered_candidates: tuple[MergeCandidate, ...]
    target_bytes: int
    current_bytes: int
    max_nonconflicting_batch_size: int
    projection_coefficients: tuple[Qwen3ProjectionCoefficients, ...]
    projection_role_scales: Mapping[Qwen3NodeId, tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class Qwen3AdaptiveSurgeryTransaction:
    """One accepted commit or exactly restored skip attempt."""

    transaction_index: int
    round_index: int
    accepted: bool
    decision: EvaluationDecision
    actions: tuple[Qwen3ActionAudit, ...]
    graph_transaction_sha256: str
    parent_runtime_snapshot: Qwen3RuntimeAudit
    proposed_runtime_snapshot: Qwen3RuntimeAudit
    committed_runtime_snapshot: Qwen3RuntimeAudit
    assignment_change_count: int
    assignment_changes_sha256: str
    projection_coefficients: tuple[Qwen3ProjectionCoefficients, ...]
    rollback_exact: bool | None

    @property
    def reverted(self) -> bool:
        return not self.accepted

    @property
    def byte_delta(self) -> int:
        return (
            self.proposed_runtime_snapshot.tensor_bytes
            - self.parent_runtime_snapshot.tensor_bytes
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_index": self.transaction_index,
            "round_index": self.round_index,
            "accepted": self.accepted,
            "reverted": self.reverted,
            "rollback_exact": self.rollback_exact,
            "decision": self.decision.as_dict(),
            "actions": [
                {"kind": action.kind, "action_sha256": action.action_sha256}
                for action in self.actions
            ],
            "action_sha256s": [action.action_sha256 for action in self.actions],
            "graph_transaction_sha256": self.graph_transaction_sha256,
            "parent_runtime": self.parent_runtime_snapshot.as_dict(),
            "proposed_runtime": self.proposed_runtime_snapshot.as_dict(),
            "committed_runtime": self.committed_runtime_snapshot.as_dict(),
            "assignment_change_count": self.assignment_change_count,
            "assignment_changes_sha256": self.assignment_changes_sha256,
            "projection_coefficients": [
                item.as_dict() for item in self.projection_coefficients
            ],
            "byte_delta": self.byte_delta,
        }


@dataclass(frozen=True, slots=True)
class Qwen3AdaptiveSurgeryResult:
    """Completed byte-target surgery report and the same mutated Qwen3 runtime."""

    runtime: nn.Module
    target_bytes: int
    initial_runtime_snapshot: Qwen3RuntimeAssignmentSnapshot
    final_runtime_snapshot: Qwen3RuntimeAssignmentSnapshot
    transactions: tuple[Qwen3AdaptiveSurgeryTransaction, ...]
    feature_recomputations: int
    graph_rounds: int
    target_reached: bool
    accepted_transaction_count: int = 0
    rejected_transaction_count: int = 0
    optimizer_steps: int = 0
    backward_calls: int = 0

    @property
    def actions(self) -> tuple[Qwen3ActionAudit, ...]:
        return tuple(action for item in self.transactions for action in item.actions)

    @property
    def accepted_transactions(self) -> int:
        return self.accepted_transaction_count

    @property
    def rejected_transactions(self) -> int:
        return self.rejected_transaction_count

    @property
    def zero_gradient(self) -> bool:
        return self.optimizer_steps == 0 and self.backward_calls == 0

    @property
    def immutable_rollback_verified(self) -> bool:
        return all(
            item.accepted or item.rollback_exact is True for item in self.transactions
        )

    @property
    def bytes_saved(self) -> int:
        return (
            self.initial_runtime_snapshot.tensor_bytes
            - self.final_runtime_snapshot.tensor_bytes
        )

    @property
    def target_deployable_neuron_codebook_bytes(self) -> int:
        """The explicit deployed-registry byte target requested by the caller."""

        return self.target_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "qwen3-adaptive-neuron-codebook-surgery",
            "qwen_specific": True,
            "target_bytes": self.target_bytes,
            "target_deployable_neuron_codebook_bytes": self.target_bytes,
            "target_reached": self.target_reached,
            "initial_runtime": self.initial_runtime_snapshot.as_dict(),
            "final_runtime": self.final_runtime_snapshot.as_dict(),
            "bytes_saved": self.bytes_saved,
            "transactions": [item.as_dict() for item in self.transactions],
            "accepted_transactions": self.accepted_transactions,
            "rejected_transactions": self.rejected_transactions,
            "feature_recomputations": self.feature_recomputations,
            "graph_rounds": self.graph_rounds,
            "immutable_rollback_verified": self.immutable_rollback_verified,
            "zero_gradient": self.zero_gradient,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
        }

    @property
    def report(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True, slots=True)
class _InstalledRuntime:
    codebook: GlobalNeuronCodebook
    public: Qwen3RuntimeAssignmentSnapshot


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "dtype": str(detached.dtype),
                "shape": [int(item) for item in detached.shape],
            }
        )
    )
    byte_view = detached.view(torch.uint8).reshape(-1)
    for start in range(0, int(byte_view.numel()), 4 * 1024 * 1024):
        digest.update(byte_view[start : start + 4 * 1024 * 1024].numpy().tobytes())
    return digest.hexdigest()


def _registry_snapshot(
    registry: NeuronCodebookRegistry,
) -> Qwen3RuntimeAssignmentSnapshot:
    state_hashes: dict[str, str] = {}
    tensor_bytes = 0
    for name, value in sorted(registry.state_dict().items()):
        tensor_bytes += int(value.numel()) * int(value.element_size())
        state_hashes[name] = _tensor_sha256(value)
    assignments = tuple(
        tuple(int(item) for item in row)
        for row in registry.assignments.detach().to("cpu", torch.int64).tolist()
    )
    sources = tuple(
        int(item)
        for item in registry.prototype_source_indices.detach()
        .to("cpu", torch.int64)
        .tolist()
    )
    assignment_sources = tuple(
        tuple(sources[slot] for slot in row) for row in assignments
    )
    payload = {
        "logical_layers": registry.logical_layers,
        "intermediate_width": registry.intermediate_width,
        "prototype_count": registry.prototype_count,
        "assignments": assignments,
        "prototype_source_indices": sources,
        "state_hashes": state_hashes,
        "tensor_bytes": tensor_bytes,
    }
    return Qwen3RuntimeAssignmentSnapshot(
        logical_layers=registry.logical_layers,
        intermediate_width=registry.intermediate_width,
        prototype_count=registry.prototype_count,
        assignments=assignments,
        prototype_source_indices=sources,
        assignment_source_indices=assignment_sources,
        tensor_bytes=tensor_bytes,
        registry_sha256=_sha256(payload),
    )


def qwen3_neuron_codebook_storage_bytes(runtime: nn.Module) -> int:
    """Count physical tensors owned by the prepared Qwen3 neuron registry."""

    _, registry = _validated_runtime(runtime)
    return _registry_snapshot(registry).tensor_bytes


def qwen3_runtime_assignment_snapshot(
    runtime: nn.Module,
) -> Qwen3RuntimeAssignmentSnapshot:
    """Capture an immutable Qwen3 neuron-codebook value snapshot."""

    _, registry = _validated_runtime(runtime)
    return _registry_snapshot(registry)


def _validated_runtime(
    runtime: nn.Module,
) -> tuple[GlobalNeuronCodebook, NeuronCodebookRegistry]:
    if not isinstance(runtime, nn.Module):
        raise Qwen3AdaptiveSurgeryShapeError("runtime must be a torch nn.Module.")
    try:
        report = qwen3_codebook_parameter_report(runtime)
    except Qwen3CodebookError as error:
        raise Qwen3AdaptiveSurgeryStateError(
            "runtime must be prepared by prepare_qwen3_codebook_model()."
        ) from error
    if getattr(getattr(runtime, "config", None), "model_type", None) != "qwen3":
        raise Qwen3AdaptiveSurgeryStateError(
            "Adaptive surgery only supports a prepared Qwen3 runtime."
        )
    if not report.full_logical_depth:
        raise Qwen3AdaptiveSurgeryStateError(
            "Prepared Qwen3 runtime must retain full logical depth."
        )
    codebook = getattr(runtime, "neuron_codebook", None)
    if not isinstance(codebook, GlobalNeuronCodebook):
        raise Qwen3AdaptiveSurgeryStateError(
            "runtime.neuron_codebook must be a GlobalNeuronCodebook."
        )
    registry = codebook.registry
    if not isinstance(registry, NeuronCodebookRegistry):
        raise Qwen3AdaptiveSurgeryStateError(
            "Qwen3 neuron codebook has no NeuronCodebookRegistry."
        )
    backbone = getattr(runtime, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != len(codebook):
        raise Qwen3AdaptiveSurgeryStateError(
            "Qwen3 decoder layers do not match the neuron codebook."
        )
    if any(layer.mlp is not codebook[index] for index, layer in enumerate(layers)):
        raise Qwen3AdaptiveSurgeryStateError(
            "Every Qwen3 layer.mlp must be its parameterless codebook view."
        )
    assignments = registry.assignments.detach().to("cpu", torch.int64)
    used = set(int(item) for item in assignments.reshape(-1).tolist())
    expected = set(range(registry.prototype_count))
    if used != expected:
        raise Qwen3AdaptiveSurgeryStateError(
            "Every physical Qwen3 neuron prototype must be actively assigned."
        )
    if any(parameter.grad is not None for parameter in registry.parameters()):
        raise Qwen3AdaptiveSurgeryStateError(
            "Qwen3 adaptive surgery requires a zero-gradient neuron registry."
        )
    return codebook, registry


def _install_codebook(runtime: nn.Module, codebook: GlobalNeuronCodebook) -> None:
    setattr(runtime, "neuron_codebook", codebook)
    layers = runtime.model.layers
    if len(layers) != len(codebook):
        raise Qwen3AdaptiveSurgeryStateError(
            "Replacement codebook depth does not match Qwen3 decoder depth."
        )
    for layer_index, layer in enumerate(layers):
        layer.mlp = codebook[layer_index]
    if any(layer.mlp is not codebook[index] for index, layer in enumerate(layers)):
        raise Qwen3AdaptiveSurgeryStateError(
            "Failed to install all replacement Qwen3 codebook views."
        )


def _capture_installed(runtime: nn.Module) -> _InstalledRuntime:
    codebook, registry = _validated_runtime(runtime)
    return _InstalledRuntime(codebook=codebook, public=_registry_snapshot(registry))


def _restore_installed(runtime: nn.Module, installed: _InstalledRuntime) -> bool:
    _install_codebook(runtime, installed.codebook)
    codebook, registry = _validated_runtime(runtime)
    restored = _registry_snapshot(registry)
    return codebook is installed.codebook and restored == installed.public


def _prototype_bytes(registry: NeuronCodebookRegistry) -> int:
    result = (
        int(registry.prototype_gate_weight[0].numel())
        * registry.prototype_gate_weight.element_size()
        + int(registry.prototype_up_weight[0].numel())
        * registry.prototype_up_weight.element_size()
        + int(registry.prototype_down_weight[:, 0].numel())
        * registry.prototype_down_weight.element_size()
        + registry.prototype_source_indices.element_size()
    )
    if registry.prototype_gate_bias is not None:
        result += registry.prototype_gate_bias.element_size()
    if registry.prototype_up_bias is not None:
        result += registry.prototype_up_bias.element_size()
    return int(result)


def _derived_weight_features(
    registry: NeuronCodebookRegistry,
    prototype_slot: int,
    dimensions: int,
) -> tuple[float, ...]:
    pieces = [
        registry.prototype_gate_weight[prototype_slot].detach().reshape(-1),
        registry.prototype_up_weight[prototype_slot].detach().reshape(-1),
        registry.prototype_down_weight[:, prototype_slot].detach().reshape(-1),
    ]
    if registry.prototype_gate_bias is not None:
        pieces.append(registry.prototype_gate_bias[prototype_slot].detach().reshape(-1))
    if registry.prototype_up_bias is not None:
        pieces.append(registry.prototype_up_bias[prototype_slot].detach().reshape(-1))
    vector = torch.cat(pieces).to(device="cpu", dtype=torch.float64)
    count = min(_positive_int(dimensions, "derived_weight_dimensions"), vector.numel())
    if count == vector.numel():
        selected = vector
    else:
        indices = torch.div(
            torch.arange(count, dtype=torch.int64) * vector.numel(),
            count,
            rounding_mode="floor",
        )
        selected = vector.index_select(0, indices)
    return tuple(float(item) for item in selected.tolist())


def _feature_sequence(
    values: Sequence[Qwen3AdaptiveSurgeryFeature] | Iterable[Any],
) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
    if isinstance(values, (str, bytes)):
        raise Qwen3AdaptiveSurgeryShapeError(
            "features must be an iterable of Qwen3AdaptiveSurgeryFeature values."
        )
    try:
        result = tuple(values)
    except TypeError as error:
        raise Qwen3AdaptiveSurgeryShapeError(
            "features must be an iterable of Qwen3AdaptiveSurgeryFeature values."
        ) from error
    if not result:
        raise Qwen3AdaptiveSurgeryShapeError("features cannot be empty.")
    if not all(isinstance(item, Qwen3AdaptiveSurgeryFeature) for item in result):
        raise Qwen3AdaptiveSurgeryShapeError(
            "Every feature must be a Qwen3AdaptiveSurgeryFeature."
        )
    return result


def _projection_sequence(
    values: Sequence[Qwen3ProjectionCoefficients] | Mapping[Any, Any],
) -> tuple[Qwen3ProjectionCoefficients, ...]:
    result: list[Qwen3ProjectionCoefficients] = []
    if isinstance(values, Mapping):
        entries = values.items()
        for pair, coefficients in entries:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise Qwen3AdaptiveSurgeryShapeError(
                    "Projection mapping keys must be (source, target) pairs."
                )
            if isinstance(coefficients, Qwen3ProjectionCoefficients):
                item = coefficients
                if (item.source, item.target) != (
                    _node_ref(pair[0]),
                    _node_ref(pair[1]),
                ):
                    raise Qwen3AdaptiveSurgeryShapeError(
                        "Projection mapping key does not match its coefficient value."
                    )
            elif isinstance(coefficients, Mapping):
                item = Qwen3ProjectionCoefficients(
                    pair[0],
                    pair[1],
                    coefficients.get("gate"),
                    coefficients.get("up"),
                    coefficients.get("down"),
                )
            else:
                try:
                    gate, up, down = coefficients
                except (TypeError, ValueError) as error:
                    raise Qwen3AdaptiveSurgeryShapeError(
                        "Projection values must contain gate, up, and down scalars."
                    ) from error
                item = Qwen3ProjectionCoefficients(
                    pair[0], pair[1], gate, up, down
                )
            result.append(item)
    else:
        if isinstance(values, (str, bytes)):
            raise Qwen3AdaptiveSurgeryShapeError(
                "projection_coefficients must be a sequence or mapping."
            )
        try:
            entries = tuple(values)
        except TypeError as error:
            raise Qwen3AdaptiveSurgeryShapeError(
                "projection_coefficients must be a sequence or mapping."
            ) from error
        if not all(isinstance(item, Qwen3ProjectionCoefficients) for item in entries):
            raise Qwen3AdaptiveSurgeryShapeError(
                "Every projection coefficient must be Qwen3ProjectionCoefficients."
            )
        result.extend(entries)
    by_pair: dict[
        tuple[Qwen3NodeId, Qwen3NodeId], Qwen3ProjectionCoefficients
    ] = {}
    for item in result:
        pair = (item.source, item.target)
        if pair in by_pair:
            raise Qwen3AdaptiveSurgeryShapeError(
                f"Duplicate projection coefficients for {pair!r}."
            )
        by_pair[pair] = item
    return tuple(by_pair[pair] for pair in sorted(by_pair))


def _projection_by_pair(
    values: Sequence[Qwen3ProjectionCoefficients] | Mapping[Any, Any],
) -> dict[tuple[Qwen3NodeId, Qwen3NodeId], Qwen3ProjectionCoefficients]:
    return {
        (item.source, item.target): item for item in _projection_sequence(values)
    }


def _features_to_nodes(
    runtime: nn.Module,
    features: Sequence[Qwen3AdaptiveSurgeryFeature],
    bounds: Qwen3FeatureBounds,
) -> tuple[SurgeryNode, ...]:
    _, registry = _validated_runtime(runtime)
    if len(features) > bounds.max_nodes:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"features exceeds max_nodes={bounds.max_nodes}."
        )
    width = registry.intermediate_width
    active_by_source = {
        int(source): slot
        for slot, source in enumerate(
            registry.prototype_source_indices.detach()
            .to("cpu", torch.int64)
            .tolist()
        )
    }
    seen: set[Qwen3NodeId] = set()
    nodes: list[SurgeryNode] = []
    total_values = 0
    channel_dimensions: dict[str, int] = {}
    for feature in sorted(features, key=lambda item: item.node_id):
        node_id = feature.node_id
        if node_id in seen:
            raise Qwen3AdaptiveSurgeryShapeError(
                f"Duplicate feature node {node_id!r}."
            )
        seen.add(node_id)
        if feature.layer_index >= registry.logical_layers:
            raise Qwen3AdaptiveSurgeryShapeError(
                f"Feature layer {feature.layer_index} is outside the Qwen3 runtime."
            )
        if feature.neuron_index >= width:
            raise Qwen3AdaptiveSurgeryShapeError(
                f"Feature neuron {feature.neuron_index} is outside the Qwen3 runtime."
            )
        source_index = feature.layer_index * width + feature.neuron_index
        if source_index not in active_by_source:
            raise Qwen3AdaptiveSurgeryStateError(
                f"Feature {node_id!r} is not an active real codebook prototype."
            )
        slot = active_by_source[source_index]
        weight = feature.weight_features or _derived_weight_features(
            registry, slot, bounds.derived_weight_dimensions
        )
        channels = {
            "weight": weight,
            "activation": feature.activation_features,
            "jacobian": feature.jacobian_features,
            "dual": feature.dual_features,
        }
        limits = {
            "weight": bounds.max_weight_dimensions,
            "activation": bounds.max_activation_dimensions,
            "jacobian": bounds.max_jacobian_dimensions,
            "dual": bounds.max_dual_dimensions,
        }
        if feature.domain_features is not None:
            channels["domain"] = feature.domain_features
            limits["domain"] = bounds.max_domain_dimensions
        for name, values in channels.items():
            if len(values) > limits[name]:
                raise Qwen3AdaptiveSurgeryShapeError(
                    f"{node_id!r} {name} feature dimension {len(values)} exceeds "
                    f"the bound {limits[name]}."
                )
            expected = channel_dimensions.setdefault(name, len(values))
            if expected != len(values):
                raise Qwen3AdaptiveSurgeryShapeError(
                    f"All {name} feature vectors must have dimension {expected}."
                )
            total_values += len(values)
        nodes.append(
            SurgeryNode(
                node_id=node_id,
                weight=weight,
                activation=feature.activation_features,
                rollout_jvp=feature.jacobian_features,
                output_dual=feature.dual_features,
                domain=feature.domain_features,
                importance=feature.importance,
                parameter_bytes=_prototype_bytes(registry),
                anchor=feature.anchor,
            )
        )
    if total_values > bounds.max_total_values:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"Feature payload has {total_values} values, exceeding "
            f"max_total_values={bounds.max_total_values}."
        )
    return tuple(nodes)


def qwen3_adaptive_features_from_channels(
    *,
    activation_features: Mapping[Any, Any],
    jacobian_features: Mapping[Any, Any] | None = None,
    dual_features: Mapping[Any, Any] | None = None,
    weight_features: Mapping[Any, Any] | None = None,
    gqa_features: Mapping[Any, Any] | None = None,
    anchors: Mapping[Any, AnchorSpec | str | None] | None = None,
    importance: Mapping[Any, float] | None = None,
) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
    """Build Qwen3 feature records from channel mappings keyed by node."""

    def normalized(mapping: Mapping[Any, Any], name: str) -> dict[Qwen3NodeId, Any]:
        if not isinstance(mapping, Mapping):
            raise Qwen3AdaptiveSurgeryShapeError(f"{name} must be a mapping.")
        result: dict[Qwen3NodeId, Any] = {}
        for key, value in mapping.items():
            node = _node_ref(key, f"{name} key")
            if node in result:
                raise Qwen3AdaptiveSurgeryShapeError(
                    f"{name} contains duplicate normalized node {node!r}."
                )
            result[node] = value
        return result

    activation = normalized(activation_features, "activation_features")
    jacobian = (
        {} if jacobian_features is None else normalized(jacobian_features, "jacobian_features")
    )
    dual = {} if dual_features is None else normalized(dual_features, "dual_features")
    if not activation:
        raise Qwen3AdaptiveSurgeryShapeError("Feature channel mappings cannot be empty.")
    if jacobian and set(jacobian) != set(activation):
        raise Qwen3AdaptiveSurgeryShapeError(
            "jacobian_features must be omitted or have every feature node key."
        )
    if dual and set(dual) != set(activation):
        raise Qwen3AdaptiveSurgeryShapeError(
            "dual_features must be omitted or have every feature node key."
        )
    weight = (
        {} if weight_features is None else normalized(weight_features, "weight_features")
    )
    if weight and set(weight) != set(activation):
        raise Qwen3AdaptiveSurgeryShapeError(
            "weight_features must be omitted or have every feature node key."
        )
    gqa = {} if gqa_features is None else normalized(gqa_features, "gqa_features")
    if gqa and set(gqa) != set(activation):
        raise Qwen3AdaptiveSurgeryShapeError(
            "gqa_features must be omitted or have every feature node key."
        )
    anchor_map = {} if anchors is None else normalized(anchors, "anchors")
    importance_map = (
        {} if importance is None else normalized(importance, "importance")
    )
    unknown = (set(anchor_map) | set(importance_map)).difference(activation)
    if unknown:
        raise Qwen3AdaptiveSurgeryShapeError(
            f"Anchor or importance mapping contains unknown nodes {sorted(unknown)!r}."
        )
    return tuple(
        Qwen3AdaptiveSurgeryFeature(
            layer_index=node[2],
            neuron_index=node[3],
            weight=weight.get(node),
            activation=activation[node],
            jacobian=jacobian.get(node),
            dual=dual.get(node),
            gqa_features=gqa.get(node),
            anchor=anchor_map.get(node),
            importance=importance_map.get(node, 1.0),
        )
        for node in sorted(activation)
    )


def build_qwen3_adaptive_surgery_graph(
    runtime: nn.Module,
    features: Sequence[Qwen3AdaptiveSurgeryFeature],
    *,
    feature_bounds: Qwen3FeatureBounds | None = None,
    score_weights: ScoreWeights | None = None,
    projection_count: int = 8,
    neighbor_window: int = 4,
    max_candidates: int = 64,
    medoid_candidate_limit: int = 32,
) -> AdaptiveSurgeryGraph:
    """Create a graph over active real prototypes in a prepared Qwen3 runtime."""

    bounds = feature_bounds or Qwen3FeatureBounds()
    if not isinstance(bounds, Qwen3FeatureBounds):
        raise Qwen3AdaptiveSurgeryShapeError(
            "feature_bounds must be a Qwen3FeatureBounds instance."
        )
    feature_values = _feature_sequence(features)
    nodes = _features_to_nodes(runtime, feature_values, bounds)
    return AdaptiveSurgeryGraph(
        nodes,
        score_weights=score_weights,
        projection_count=projection_count,
        neighbor_window=neighbor_window,
        max_candidates=max_candidates,
        medoid_candidate_limit=medoid_candidate_limit,
    )


def _least_squares_scale(source: Tensor, target: Tensor) -> float:
    source64 = source.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    target64 = target.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    denominator = float(torch.dot(target64, target64).item())
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if bool(torch.equal(source64, target64)) else 0.0
    result = float(torch.dot(source64, target64).item()) / denominator
    if not math.isfinite(result):
        raise Qwen3AdaptiveSurgeryStateError(
            "Projected anchor produced a non-finite scalar."
        )
    return result


def _producer_vector(
    weight: Tensor,
    bias: Tensor | None,
    prototype_slot: int,
    scale: Tensor,
) -> Tensor:
    value = weight[prototype_slot].detach() * scale
    if bias is None:
        return value
    return torch.cat((value.reshape(-1), (bias[prototype_slot].detach() * scale).view(1)))


def _least_squares_projection_roles(
    registry: NeuronCodebookRegistry,
    source_slot: int,
    target_slot: int,
    source_scale: Tensor,
) -> tuple[float, float, float]:
    """Fit detached scalar target-role multipliers for one logical Qwen row."""

    gate_source = _producer_vector(
        registry.prototype_gate_weight,
        registry.prototype_gate_bias,
        source_slot,
        source_scale[0],
    )
    gate_target = _producer_vector(
        registry.prototype_gate_weight,
        registry.prototype_gate_bias,
        target_slot,
        torch.ones_like(source_scale[0]),
    )
    up_source = _producer_vector(
        registry.prototype_up_weight,
        registry.prototype_up_bias,
        source_slot,
        source_scale[1],
    )
    up_target = _producer_vector(
        registry.prototype_up_weight,
        registry.prototype_up_bias,
        target_slot,
        torch.ones_like(source_scale[1]),
    )
    return (
        _least_squares_scale(gate_source, gate_target),
        _least_squares_scale(up_source, up_target),
        _least_squares_scale(
            registry.prototype_down_weight[:, source_slot].detach() * source_scale[2],
            registry.prototype_down_weight[:, target_slot].detach(),
        ),
    )


def _projected_scales(
    registry: NeuronCodebookRegistry,
    redirects: Mapping[int, int],
    projection_roles_by_slot: Mapping[int, tuple[float, float, float] | None],
) -> Tensor | None:
    existing = registry.neuron_scales
    if not projection_roles_by_slot:
        return None if existing is None else existing.detach().clone()
    if existing is None:
        scales = torch.ones(
            (registry.logical_layers, registry.intermediate_width, 3),
            device=registry.prototype_gate_weight.device,
            dtype=registry.prototype_gate_weight.dtype,
        )
    else:
        scales = existing.detach().clone()
    old_assignments = registry.assignments.detach().to("cpu", torch.int64)
    for layer_index in range(registry.logical_layers):
        for neuron_index in range(registry.intermediate_width):
            old_slot = int(old_assignments[layer_index, neuron_index])
            if old_slot not in projection_roles_by_slot:
                continue
            target_slot = redirects[old_slot]
            old_scale = (
                torch.ones(
                    3,
                    device=registry.prototype_gate_weight.device,
                    dtype=registry.prototype_gate_weight.dtype,
                )
                if existing is None
                else existing[layer_index, neuron_index].detach()
            )
            coefficients = projection_roles_by_slot[old_slot]
            if coefficients is None:
                coefficients = _least_squares_projection_roles(
                    registry, old_slot, target_slot, old_scale
                )
                scales[layer_index, neuron_index] = torch.tensor(
                    coefficients, device=scales.device, dtype=scales.dtype
                )
            else:
                scales[layer_index, neuron_index] = old_scale * torch.tensor(
                    coefficients, device=scales.device, dtype=scales.dtype
                )
    return scales


def _copy_requires_grad(
    source: NeuronCodebookRegistry,
    target: NeuronCodebookRegistry,
    *,
    scales_created: bool,
) -> None:
    source_parameters = dict(source.named_parameters())
    for name, parameter in target.named_parameters():
        source_parameter = source_parameters.get(name)
        requires_grad = (
            False
            if name == "neuron_scales" and scales_created
            else True
            if source_parameter is None
            else source_parameter.requires_grad
        )
        parameter.requires_grad_(requires_grad)


def _compacted_codebook(
    runtime: nn.Module,
    graph_snapshot: SurgerySnapshot,
    actions: Sequence[ActionRecord],
    *,
    projection_coefficients: (
        Sequence[Qwen3ProjectionCoefficients] | Mapping[Any, Any]
    ) = (),
    projection_bindings: Sequence[ProjectionBinding] = (),
) -> GlobalNeuronCodebook:
    current_codebook, registry = _validated_runtime(runtime)
    width = registry.intermediate_width
    source_values = (
        registry.prototype_source_indices.detach().to("cpu", torch.int64).tolist()
    )
    slot_by_node = {
        qwen3_neuron_node_id(*divmod(int(source), width)): slot
        for slot, source in enumerate(source_values)
    }
    redirects = {slot: slot for slot in range(registry.prototype_count)}
    for source_node, target_node in graph_snapshot.assignment_pairs:
        source_id = _node_ref(source_node, "graph assignment source")
        target_id = _node_ref(target_node, "graph assignment target")
        if source_id not in slot_by_node or target_id not in slot_by_node:
            raise Qwen3AdaptiveSurgeryStateError(
                "Graph assignment references a non-active Qwen3 prototype."
            )
        redirects[slot_by_node[source_id]] = slot_by_node[target_id]
    for slot, target in redirects.items():
        if redirects[target] != target:
            raise Qwen3AdaptiveSurgeryStateError(
                f"Runtime redirect {slot}->{target} does not end at a real medoid."
            )
    survivor_slots = tuple(sorted(set(redirects.values())))
    if len(survivor_slots) >= registry.prototype_count:
        raise Qwen3AdaptiveSurgeryStateError(
            "Adaptive graph proposal did not remove a physical Qwen3 prototype."
        )
    new_slot = {old: index for index, old in enumerate(survivor_slots)}
    old_assignments = registry.assignments.detach().to("cpu", torch.int64)
    flattened = [
        new_slot[redirects[int(slot)]] for slot in old_assignments.reshape(-1).tolist()
    ]
    assignments = torch.tensor(flattened, dtype=torch.int64).reshape(
        registry.logical_layers, registry.intermediate_width
    )
    index_device = registry.prototype_gate_weight.device
    selected = torch.tensor(survivor_slots, device=index_device, dtype=torch.long)
    explicit_by_pair = _projection_by_pair(projection_coefficients)
    generic_by_pair = {
        (_node_ref(item.source, "projection binding source"), _node_ref(item.target, "projection binding target")):
        _finite_float(item.coefficient, "projection binding coefficient")
        for item in projection_bindings
    }
    projection_roles_by_slot: dict[int, tuple[float, float, float] | None] = {}
    for action in actions:
        if action.kind != "project" or len(action.sources) != 1:
            continue
        source = _node_ref(action.sources[0], "project action source")
        target = _node_ref(action.target, "project action target")
        source_slot = slot_by_node.get(source)
        target_slot = slot_by_node.get(target)
        if source_slot is None or target_slot is None or redirects[source_slot] != target_slot:
            raise Qwen3AdaptiveSurgeryStateError(
                "Projected Qwen3 action does not match the proposed real-medoid redirect."
            )
        explicit = explicit_by_pair.get((source, target))
        if explicit is not None:
            projection_roles_by_slot[source_slot] = explicit.values
            continue
        scalar = generic_by_pair.get((source, target))
        if scalar is not None:
            projection_roles_by_slot[source_slot] = (scalar, scalar, scalar)
            continue
        # A projected anchor without a supplied executable binding falls back
        # to detached least-squares scalars for the three Qwen MLP roles.
        projection_roles_by_slot[source_slot] = None
    scales = _projected_scales(registry, redirects, projection_roles_by_slot)
    scales_created = registry.neuron_scales is None and scales is not None
    new_registry = NeuronCodebookRegistry(
        registry.prototype_gate_weight.index_select(0, selected),
        registry.prototype_up_weight.index_select(0, selected),
        registry.prototype_down_weight.index_select(1, selected),
        assignments,
        registry.prototype_source_indices.detach()
        .to(device=index_device)
        .index_select(0, selected),
        prototype_gate_bias=(
            None
            if registry.prototype_gate_bias is None
            else registry.prototype_gate_bias.index_select(0, selected)
        ),
        prototype_up_bias=(
            None
            if registry.prototype_up_bias is None
            else registry.prototype_up_bias.index_select(0, selected)
        ),
        layer_down_bias=registry.layer_down_bias,
        neuron_scales=scales,
        activation=registry.activation,
        diagnostics=None,
    )
    _copy_requires_grad(registry, new_registry, scales_created=scales_created)
    new_registry.train(registry.training)
    result = GlobalNeuronCodebook(new_registry)
    result.train(current_codebook.training)
    return result


def _assignment_changes(
    before: Qwen3RuntimeAssignmentSnapshot,
    after: Qwen3RuntimeAssignmentSnapshot,
) -> tuple[Qwen3RuntimeAssignmentChange, ...]:
    if (
        before.logical_layers != after.logical_layers
        or before.intermediate_width != after.intermediate_width
    ):
        raise Qwen3AdaptiveSurgeryStateError(
            "A surgery transaction changed the logical Qwen3 neuron layout."
        )
    width = before.intermediate_width
    result: list[Qwen3RuntimeAssignmentChange] = []
    for layer_index in range(before.logical_layers):
        for neuron_index in range(width):
            source = before.assignment_source_indices[layer_index][neuron_index]
            target = after.assignment_source_indices[layer_index][neuron_index]
            if source == target:
                continue
            result.append(
                Qwen3RuntimeAssignmentChange(
                    logical_node=qwen3_neuron_node_id(layer_index, neuron_index),
                    before_prototype=qwen3_neuron_node_id(*divmod(source, width)),
                    after_prototype=qwen3_neuron_node_id(*divmod(target, width)),
                )
            )
    return tuple(result)


def _decision(value: Any) -> EvaluationDecision:
    if isinstance(value, EvaluationDecision):
        return value
    if isinstance(value, bool):
        return EvaluationDecision(value)
    if not isinstance(value, Mapping):
        raise Qwen3AdaptiveSurgeryShapeError(
            "gate must return bool, EvaluationDecision, or a mapping."
        )
    accepted = value.get("accepted", value.get("accept"))
    if not isinstance(accepted, bool):
        raise Qwen3AdaptiveSurgeryShapeError(
            "gate mapping must contain a boolean accepted or accept field."
        )
    reason_value = value.get("reason", value.get("accepted_reason", ""))
    reason = "" if reason_value is None else str(reason_value)
    metrics = {
        str(key): item
        for key, item in value.items()
        if key not in {"accepted", "accept", "reason", "accepted_reason"}
        and isinstance(item, (str, int, float, bool, type(None)))
        and not (isinstance(item, float) and not math.isfinite(item))
    }
    return EvaluationDecision(accepted, reason=reason, metrics=metrics)


def _invoke_gate(gate: Gate | None, context: Qwen3SurgeryEvaluationContext) -> EvaluationDecision:
    if gate is None:
        return EvaluationDecision.accept(reason="no external rollout gate supplied")
    callable_value = getattr(gate, "evaluate", gate)
    if not callable(callable_value):
        raise Qwen3AdaptiveSurgeryShapeError("gate must be callable or expose evaluate().")
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError):
        return _decision(callable_value(context))
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if has_varargs or len(positional) >= 4:
        value = callable_value(
            context.runtime,
            context.parent_graph_snapshot,
            context.proposed_graph_snapshot,
            context.graph_transaction,
        )
    elif len(positional) == 3:
        value = callable_value(
            context.parent_graph_snapshot,
            context.proposed_graph_snapshot,
            context.graph_transaction,
        )
    elif len(positional) == 2:
        value = callable_value(context.runtime, context)
    elif len(positional) == 1:
        value = callable_value(context)
    else:
        value = callable_value()
    return _decision(value)


def _invoke_recompute(
    callback: FeatureRecompute,
    context: Qwen3SurgeryRecomputeContext,
) -> Qwen3SurgeryEvidenceRefresh:
    if not callable(callback):
        raise Qwen3AdaptiveSurgeryShapeError("recompute_features must be callable.")
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        value = callback(context)
        return _normalize_refresh(value)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if has_varargs or len(positional) >= 2:
        value = callback(context.runtime, context)
    elif len(positional) == 1:
        value = callback(context)
    else:
        value = callback()
    return _normalize_refresh(value)


def _normalize_refresh(value: Any) -> Qwen3SurgeryEvidenceRefresh:
    if isinstance(value, Qwen3SurgeryEvidenceRefresh):
        return value
    return Qwen3SurgeryEvidenceRefresh(_feature_sequence(value))


class Qwen3AdaptiveSurgeryBridge:
    """Transactional batched surgery over one prepared Qwen3 neuron runtime."""

    def __init__(
        self,
        runtime: nn.Module,
        features: Sequence[Qwen3AdaptiveSurgeryFeature],
        *,
        feature_bounds: Qwen3FeatureBounds | None = None,
        score_weights: ScoreWeights | None = None,
        projection_count: int = 8,
        neighbor_window: int = 4,
        max_candidates: int = 64,
        medoid_candidate_limit: int = 32,
        max_nonconflicting_batch_size: int = 1,
        transaction_history_limit: int = 32,
        candidate_selector: CandidateSelector | None = None,
        candidate_policy: CandidateSelector | None = None,
        projection_coefficients: (
            Sequence[Qwen3ProjectionCoefficients] | Mapping[Any, Any]
        ) = (),
    ) -> None:
        _validated_runtime(runtime)
        self.runtime = runtime
        self.feature_bounds = feature_bounds or Qwen3FeatureBounds()
        if not isinstance(self.feature_bounds, Qwen3FeatureBounds):
            raise Qwen3AdaptiveSurgeryShapeError(
                "feature_bounds must be a Qwen3FeatureBounds instance."
            )
        self.score_weights = score_weights
        self.projection_count = _positive_int(projection_count, "projection_count")
        self.neighbor_window = _positive_int(neighbor_window, "neighbor_window")
        self.max_candidates = _positive_int(max_candidates, "max_candidates")
        self.medoid_candidate_limit = _positive_int(
            medoid_candidate_limit, "medoid_candidate_limit"
        )
        self.max_nonconflicting_batch_size = _positive_int(
            max_nonconflicting_batch_size, "max_nonconflicting_batch_size"
        )
        self.transaction_history_limit = _positive_int(
            transaction_history_limit, "transaction_history_limit"
        )
        if candidate_selector is not None and candidate_policy is not None:
            raise Qwen3AdaptiveSurgeryShapeError(
                "Provide candidate_selector or candidate_policy, not both."
            )
        self._candidate_selector = (
            candidate_selector if candidate_selector is not None else candidate_policy
        )
        if self._candidate_selector is not None and not callable(
            getattr(self._candidate_selector, "select", self._candidate_selector)
        ):
            raise Qwen3AdaptiveSurgeryShapeError(
                "candidate_selector must be callable or expose select()."
            )
        self._features = _feature_sequence(features)
        self._projection_coefficients = _projection_sequence(projection_coefficients)
        self._graph = self._build_graph(self._features)
        self._transactions: list[Qwen3AdaptiveSurgeryTransaction] = []
        self._attempt_count = 0
        self._accepted_transaction_count = 0
        self._rejected_transaction_count = 0
        self._feature_recomputations = 0
        self._graph_rounds = 1
        self._has_run = False

    @property
    def graph(self) -> AdaptiveSurgeryGraph:
        return self._graph

    @property
    def features(self) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
        return self._features

    @property
    def transactions(self) -> tuple[Qwen3AdaptiveSurgeryTransaction, ...]:
        return tuple(self._transactions)

    @property
    def projection_coefficients(self) -> tuple[Qwen3ProjectionCoefficients, ...]:
        return self._projection_coefficients

    @property
    def runtime_snapshot(self) -> Qwen3RuntimeAssignmentSnapshot:
        return qwen3_runtime_assignment_snapshot(self.runtime)

    def _build_graph(
        self, features: Sequence[Qwen3AdaptiveSurgeryFeature]
    ) -> AdaptiveSurgeryGraph:
        return build_qwen3_adaptive_surgery_graph(
            self.runtime,
            features,
            feature_bounds=self.feature_bounds,
            score_weights=self.score_weights,
            projection_count=self.projection_count,
            neighbor_window=self.neighbor_window,
            max_candidates=self.max_candidates,
            medoid_candidate_limit=self.medoid_candidate_limit,
        )

    def _next_features(
        self,
        proposal_snapshot: SurgerySnapshot,
        graph_transaction: SurgeryTransaction,
        target_bytes: int,
        recompute_features: FeatureRecompute | None,
    ) -> Qwen3SurgeryEvidenceRefresh:
        active = tuple(_node_ref(item) for item in proposal_snapshot.prototype_ids)
        if recompute_features is None:
            by_node = {item.node_id: item for item in self._features}
            return Qwen3SurgeryEvidenceRefresh(
                tuple(by_node[node] for node in active),
                self._projection_coefficients,
            )
        context = Qwen3SurgeryRecomputeContext(
            runtime=self.runtime,
            round_index=self._graph_rounds,
            accepted_transaction_index=len(self._transactions) + 1,
            target_bytes=target_bytes,
            current_bytes=self.runtime_snapshot.tensor_bytes,
            active_node_ids=active,
            graph_transaction=graph_transaction,
        )
        before = self.runtime_snapshot
        with torch.inference_mode():
            refreshed = _invoke_recompute(recompute_features, context)
        after = self.runtime_snapshot
        if after != before:
            raise Qwen3AdaptiveSurgeryStateError(
                "Feature recomputation mutated the installed Qwen3 codebook."
            )
        refreshed_ids = {item.node_id for item in refreshed.features}
        if refreshed_ids != set(active) or len(refreshed.features) != len(active):
            raise Qwen3AdaptiveSurgeryShapeError(
                "Feature recomputation must return exactly one record for every "
                "remaining active surgery node."
            )
        previous = {item.node_id: item for item in self._features}
        for item in refreshed.features:
            original = previous[item.node_id]
            if item.anchor != original.anchor or item.importance != original.importance:
                raise Qwen3AdaptiveSurgeryShapeError(
                    "Feature recomputation may replace evidence only; anchors and "
                    "importance are immutable."
                )
        self._feature_recomputations += 1
        return refreshed

    def _projection_role_scales(self) -> dict[Qwen3NodeId, tuple[float, float, float]]:
        _, registry = _validated_runtime(self.runtime)
        if registry.neuron_scales is None:
            return {}
        values = registry.neuron_scales.detach().to("cpu", torch.float64)
        return {
            qwen3_neuron_node_id(layer, neuron): tuple(
                float(value) for value in values[layer, neuron].tolist()
            )
            for layer in range(registry.logical_layers)
            for neuron in range(registry.intermediate_width)
        }

    def _projection_rank_adjustment(self, candidate: MergeCandidate) -> float:
        source, target = (_node_ref(item) for item in candidate.sources)
        explicit = _projection_by_pair(self._projection_coefficients).get((source, target))
        if explicit is not None:
            return -sum(abs(value - 1.0) for value in explicit.values)
        scales = self._projection_role_scales().get(source)
        return 0.0 if scales is None else -sum(abs(value - 1.0) for value in scales)

    def _marginal_bytes_saved(self, candidates: Sequence[MergeCandidate]) -> int:
        _, registry = _validated_runtime(self.runtime)
        saved = sum(candidate.score.parameter_bytes_saved for candidate in candidates)
        if registry.neuron_scales is not None:
            return saved
        by_node = {item.node_id: item for item in self._features}
        needs_scales = any(
            by_node[node].anchor is not None and by_node[node].anchor.mode == "projected"
            for candidate in candidates
            for node in candidate.member_ids
        )
        allocation = (
            registry.logical_layers
            * registry.intermediate_width
            * 3
            * registry.prototype_gate_weight.element_size()
            if needs_scales
            else 0
        )
        return saved - allocation

    def _append_transaction(self, transaction: Qwen3AdaptiveSurgeryTransaction) -> None:
        self._transactions.append(transaction)
        self._attempt_count += 1
        if transaction.accepted:
            self._accepted_transaction_count += 1
        else:
            self._rejected_transaction_count += 1
        overflow = len(self._transactions) - self.transaction_history_limit
        if overflow > 0:
            del self._transactions[:overflow]

    def _graph_checkpoint(self) -> Any:
        checkpoint = getattr(self._graph, "_checkpoint", None)
        if not callable(checkpoint):
            raise Qwen3AdaptiveSurgeryStateError(
                "Adaptive surgery graph does not expose transactional checkpointing."
            )
        return checkpoint()

    def _restore_graph_checkpoint(self, checkpoint: Any) -> None:
        restore = getattr(self._graph, "_restore_checkpoint", None)
        if not callable(restore):
            raise Qwen3AdaptiveSurgeryStateError(
                "Adaptive surgery graph does not expose transactional checkpoint restore."
            )
        restore(checkpoint)

    @staticmethod
    def _is_explicit_pair(value: Any) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
            and all(isinstance(item, tuple) for item in value)
        )

    def _invoke_candidate_selector(
        self,
        selector: CandidateSelector,
        context: Qwen3SurgeryCandidateSelectionContext,
    ) -> Any:
        callable_value = getattr(selector, "select", selector)
        if not callable(callable_value):
            raise Qwen3AdaptiveSurgeryShapeError(
                "candidate_selector must be callable or expose select()."
            )
        try:
            signature = inspect.signature(callable_value)
        except (TypeError, ValueError):
            return callable_value(context)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        has_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        if has_varargs or len(positional) >= 3:
            return callable_value(
                context.graph, context.discovered_candidates, context
            )
        if len(positional) == 2:
            if "graph" in positional[0].name.lower():
                return callable_value(context.graph, context.discovered_candidates)
            return callable_value(context.discovered_candidates, context)
        if len(positional) == 1:
            name = positional[0].name.lower()
            if "candidate" in name or "discover" in name:
                return callable_value(context.discovered_candidates)
            if "graph" in name:
                return callable_value(context.graph)
            return callable_value(context)
        return callable_value()

    def _selected_candidates(
        self,
        *,
        target_bytes: int,
        current_bytes: int,
        selector: CandidateSelector | None,
    ) -> tuple[MergeCandidate, ...]:
        checkpoint = self._graph_checkpoint()
        try:
            discovered = tuple(
                sorted(
                    self._graph.propose_candidates(),
                    key=lambda item: (
                        -(
                            item.score.total_score
                            + self._projection_rank_adjustment(item)
                        ),
                        item.candidate_sha256,
                    ),
                )
            )
            if selector is None:
                ranked: tuple[Any, ...] = tuple(
                    self._graph.propose_nonconflicting_batch(
                        self.max_nonconflicting_batch_size, discovered
                    )
                )
            else:
                selected = self._invoke_candidate_selector(
                    selector,
                    Qwen3SurgeryCandidateSelectionContext(
                        graph=self._graph,
                        discovered_candidates=discovered,
                        target_bytes=target_bytes,
                        current_bytes=current_bytes,
                        max_nonconflicting_batch_size=self.max_nonconflicting_batch_size,
                        projection_coefficients=self._projection_coefficients,
                        projection_role_scales=self._projection_role_scales(),
                    ),
                )
            if selector is not None and isinstance(selected, Mapping):
                ranked = tuple(selected.get("candidates", ())) + tuple(
                    selected.get("pairs", selected.get("explicit_pairs", ()))
                )
            elif selector is not None and (
                isinstance(selected, MergeCandidate) or self._is_explicit_pair(selected)
            ):
                ranked = (selected,)
            elif selector is not None and isinstance(selected, Sequence) and not isinstance(
                selected, (str, bytes)
            ):
                ranked = tuple(selected)
            elif selector is not None:
                raise Qwen3AdaptiveSurgeryShapeError(
                    "candidate_selector must return candidates, explicit pairs, "
                    "or a mapping containing candidates and/or pairs."
                )

            selected_candidates: list[MergeCandidate] = []
            occupied: set[Qwen3NodeId] = set()
            explicit_pairs_scored = 0
            for item in ranked:
                if isinstance(item, MergeCandidate):
                    candidate = item
                elif self._is_explicit_pair(item):
                    if explicit_pairs_scored >= self.max_nonconflicting_batch_size:
                        break
                    explicit_pairs_scored += 1
                    candidate = self._graph.propose_candidate_for_pair(item)
                else:
                    raise Qwen3AdaptiveSurgeryShapeError(
                        "candidate_selector items must be MergeCandidate objects or "
                        "two-node explicit pairs."
                    )
                if candidate.parent_snapshot_sha256 != self._graph.snapshot.snapshot_sha256:
                    raise Qwen3AdaptiveSurgeryStateError(
                    "candidate_selector returned a stale graph candidate."
                    )
                members = set(candidate.member_ids)
                if members & occupied:
                    continue
                selected_candidates.append(candidate)
                occupied.update(members)
                if len(selected_candidates) >= self.max_nonconflicting_batch_size:
                    break

            remaining_bytes = current_bytes - target_bytes
            batch: list[MergeCandidate] = []
            for candidate in selected_candidates:
                batch.append(candidate)
                if self._marginal_bytes_saved(batch) >= remaining_bytes:
                    break
            result = tuple(batch)
        except Exception:
            self._restore_graph_checkpoint(checkpoint)
            raise
        self._restore_graph_checkpoint(checkpoint)
        return result

    def _rescore_batch(
        self, candidates: Sequence[MergeCandidate]
    ) -> tuple[MergeCandidate, ...]:
        return tuple(
            self._graph.propose_candidate_for_pair(candidate.sources)
            for candidate in candidates
        )

    def run(
        self,
        target_bytes: int,
        *,
        gate: Gate | None = None,
        evaluator: Gate | None = None,
        recompute_features: FeatureRecompute | None = None,
        max_attempts: int | None = None,
        candidate_selector: CandidateSelector | None = None,
        candidate_policy: CandidateSelector | None = None,
    ) -> Qwen3AdaptiveSurgeryResult:
        """Apply batched gated transactions until physical registry bytes meet target."""

        if self._has_run:
            raise Qwen3AdaptiveSurgeryStateError(
                "A Qwen3AdaptiveSurgeryBridge instance can run only once."
            )
        self._has_run = True
        target = _nonnegative_int(target_bytes, "target_bytes")
        if gate is not None and evaluator is not None:
            raise Qwen3AdaptiveSurgeryShapeError(
                "Provide gate or evaluator, not both."
            )
        rollout_gate = gate if gate is not None else evaluator
        if candidate_selector is not None and candidate_policy is not None:
            raise Qwen3AdaptiveSurgeryShapeError(
                "Provide candidate_selector or candidate_policy, not both."
            )
        selector = (
            candidate_selector
            if candidate_selector is not None
            else candidate_policy
            if candidate_policy is not None
            else self._candidate_selector
        )
        if selector is not None and not callable(getattr(selector, "select", selector)):
            raise Qwen3AdaptiveSurgeryShapeError(
                "candidate_selector must be callable or expose select()."
            )
        attempt_limit = (
            None if max_attempts is None else _positive_int(max_attempts, "max_attempts")
        )
        initial = _capture_installed(self.runtime)
        current = initial.public
        if target >= current.tensor_bytes:
            return Qwen3AdaptiveSurgeryResult(
                runtime=self.runtime,
                target_bytes=target,
                initial_runtime_snapshot=initial.public,
                final_runtime_snapshot=current,
                transactions=(),
                feature_recomputations=0,
                graph_rounds=self._graph_rounds,
                target_reached=True,
            )

        def attempt_batch(
            candidates: tuple[MergeCandidate, ...], graph_checkpoint: Any
        ) -> bool:
            nonlocal current
            if attempt_limit is not None and self._attempt_count >= attempt_limit:
                raise Qwen3AdaptiveSurgeryTargetError(
                    f"max_attempts={attempt_limit} was reached at "
                    f"{current.tensor_bytes} bytes; target is {target}."
                )
            proposal = self._graph.preview(candidates)
            parent = _capture_installed(self.runtime)
            try:
                with torch.inference_mode():
                    tentative_codebook = _compacted_codebook(
                        self.runtime,
                        proposal.snapshot,
                        proposal.transaction.actions,
                        projection_coefficients=self._projection_coefficients,
                        projection_bindings=proposal.transaction.projection_bindings,
                    )
                    _install_codebook(self.runtime, tentative_codebook)
                proposed = _capture_installed(self.runtime)
                context = Qwen3SurgeryEvaluationContext(
                    runtime=self.runtime,
                    round_index=self._graph_rounds,
                    target_bytes=target,
                    parent_graph_snapshot=proposal.parent,
                    proposed_graph_snapshot=proposal.snapshot,
                    graph_transaction=proposal.transaction,
                    parent_runtime_snapshot=parent.public,
                    proposed_runtime_snapshot=proposed.public,
                )
                with torch.inference_mode():
                    decision = _invoke_gate(rollout_gate, context)
                if self.runtime_snapshot != proposed.public:
                    raise Qwen3AdaptiveSurgeryStateError(
                        "Rollout gate mutated the tentative Qwen3 codebook."
                    )
            except Exception:
                if not _restore_installed(self.runtime, parent):
                    raise Qwen3AdaptiveSurgeryStateError(
                        "Tentative Qwen3 surgery failed and exact rollback failed."
                    )
                self._restore_graph_checkpoint(graph_checkpoint)
                raise

            changes = _assignment_changes(parent.public, proposed.public)
            transaction_index = self._attempt_count + 1
            if not decision.accepted:
                if not _restore_installed(self.runtime, parent):
                    raise Qwen3AdaptiveSurgeryStateError(
                        "Rejected Qwen3 surgery did not restore the exact parent."
                    )
                outcome = self._graph.reject(proposal, decision)
                current = self.runtime_snapshot
                self._append_transaction(
                    Qwen3AdaptiveSurgeryTransaction(
                        transaction_index=transaction_index,
                        round_index=self._graph_rounds,
                        accepted=False,
                        decision=decision,
                        actions=tuple(
                            Qwen3ActionAudit(action.kind, action.action_sha256)
                            for action in outcome.attempt.actions
                        ),
                        graph_transaction_sha256=proposal.transaction.transaction_sha256,
                        parent_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(parent.public),
                        proposed_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(proposed.public),
                        committed_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(current),
                        assignment_change_count=len(changes),
                        assignment_changes_sha256=_sha256(
                            [change.as_dict() for change in changes]
                        ),
                        projection_coefficients=(),
                        rollback_exact=True,
                    )
                )
                return False

            recomputations_before = self._feature_recomputations
            try:
                # Recompute before mutating graph state so callback failures are atomic.
                refresh = self._next_features(
                    proposal.snapshot,
                    proposal.transaction,
                    target,
                    recompute_features,
                )
                next_graph = (
                    self._build_graph(refresh.features)
                    if proposed.public.tensor_bytes > target
                    else None
                )
                outcome = self._graph.commit(proposal, decision)
            except Exception:
                self._feature_recomputations = recomputations_before
                if not _restore_installed(self.runtime, parent):
                    raise Qwen3AdaptiveSurgeryStateError(
                        "Accepted Qwen3 proposal failed and exact rollback failed."
                    )
                self._restore_graph_checkpoint(graph_checkpoint)
                raise

            current = proposed.public
            used_projection_pairs = {
                (_node_ref(binding.source), _node_ref(binding.target))
                for binding in outcome.transaction.projection_bindings
            }
            self._append_transaction(
                Qwen3AdaptiveSurgeryTransaction(
                    transaction_index=transaction_index,
                    round_index=self._graph_rounds,
                    accepted=True,
                    decision=decision,
                    actions=tuple(
                        Qwen3ActionAudit(action.kind, action.action_sha256)
                        for action in outcome.transaction.actions
                    ),
                    graph_transaction_sha256=outcome.transaction.transaction_sha256,
                    parent_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(parent.public),
                    proposed_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(proposed.public),
                    committed_runtime_snapshot=Qwen3RuntimeAudit.from_snapshot(proposed.public),
                    assignment_change_count=len(changes),
                    assignment_changes_sha256=_sha256(
                        [change.as_dict() for change in changes]
                    ),
                    projection_coefficients=tuple(
                        item
                        for item in self._projection_coefficients
                        if (item.source, item.target) in used_projection_pairs
                    ),
                    rollback_exact=None,
                )
            )
            self._features = refresh.features
            self._projection_coefficients = refresh.projection_coefficients
            if next_graph is not None:
                self._graph = next_graph
                self._graph_rounds += 1
            return True

        def resolve_rejected_batch(candidates: tuple[MergeCandidate, ...]) -> None:
            graph_checkpoint = self._graph_checkpoint()
            try:
                fresh = self._rescore_batch(candidates)
            except Exception:
                self._restore_graph_checkpoint(graph_checkpoint)
                raise
            if attempt_batch(fresh, graph_checkpoint) or len(fresh) == 1:
                return
            midpoint = len(fresh) // 2
            resolve_rejected_batch(fresh[:midpoint])
            resolve_rejected_batch(fresh[midpoint:])

        while current.tensor_bytes > target:
            if attempt_limit is not None and self._attempt_count >= attempt_limit:
                raise Qwen3AdaptiveSurgeryTargetError(
                    f"max_attempts={attempt_limit} was reached at "
                    f"{current.tensor_bytes} bytes; target is {target}."
                )
            try:
                candidates = self._selected_candidates(
                    target_bytes=target,
                    current_bytes=current.tensor_bytes,
                    selector=selector,
                )
            except SurgerySearchExhausted as error:
                raise Qwen3AdaptiveSurgeryTargetError(str(error)) from error
            if not candidates:
                raise Qwen3AdaptiveSurgeryTargetError(
                    f"No Qwen3 surgery candidate remains at {current.tensor_bytes} "
                    f"bytes; target is {target}."
                )
            resolve_rejected_batch(candidates)

        final = self.runtime_snapshot
        return Qwen3AdaptiveSurgeryResult(
            runtime=self.runtime,
            target_bytes=target,
            initial_runtime_snapshot=initial.public,
            final_runtime_snapshot=final,
            transactions=tuple(self._transactions),
            feature_recomputations=self._feature_recomputations,
            graph_rounds=self._graph_rounds,
            target_reached=final.tensor_bytes <= target,
            accepted_transaction_count=self._accepted_transaction_count,
            rejected_transaction_count=self._rejected_transaction_count,
        )


def apply_qwen3_adaptive_surgery(
    runtime: nn.Module,
    features: Sequence[Qwen3AdaptiveSurgeryFeature],
    *,
    target_bytes: int,
    gate: Gate | None = None,
    evaluator: Gate | None = None,
    recompute_features: FeatureRecompute | None = None,
    projection_coefficients: (
        Sequence[Qwen3ProjectionCoefficients] | Mapping[Any, Any]
    ) = (),
    feature_bounds: Qwen3FeatureBounds | None = None,
    score_weights: ScoreWeights | None = None,
    projection_count: int = 8,
    neighbor_window: int = 4,
    max_candidates: int = 64,
    medoid_candidate_limit: int = 32,
    max_nonconflicting_batch_size: int = 1,
    transaction_history_limit: int = 32,
    max_attempts: int | None = None,
    candidate_selector: CandidateSelector | None = None,
    candidate_policy: CandidateSelector | None = None,
) -> Qwen3AdaptiveSurgeryResult:
    """Build a Qwen3-specific bridge and run it to a physical byte target."""

    bridge = Qwen3AdaptiveSurgeryBridge(
        runtime,
        features,
        feature_bounds=feature_bounds,
        score_weights=score_weights,
        projection_count=projection_count,
        neighbor_window=neighbor_window,
        max_candidates=max_candidates,
        medoid_candidate_limit=medoid_candidate_limit,
        max_nonconflicting_batch_size=max_nonconflicting_batch_size,
        transaction_history_limit=transaction_history_limit,
        candidate_selector=candidate_selector,
        candidate_policy=candidate_policy,
        projection_coefficients=projection_coefficients,
    )
    return bridge.run(
        target_bytes,
        gate=gate,
        evaluator=evaluator,
        recompute_features=recompute_features,
        max_attempts=max_attempts,
    )


surgically_compress_qwen3_codebook = apply_qwen3_adaptive_surgery


__all__ = [
    "QWEN3_NODE_KIND",
    "SCHEMA_VERSION",
    "Qwen3AdaptiveSurgeryBridge",
    "Qwen3AdaptiveSurgeryError",
    "Qwen3AdaptiveSurgeryFeature",
    "Qwen3AdaptiveSurgeryResult",
    "Qwen3AdaptiveSurgeryShapeError",
    "Qwen3AdaptiveSurgeryStateError",
    "Qwen3AdaptiveSurgeryTargetError",
    "Qwen3AdaptiveSurgeryTransaction",
    "Qwen3ActionAudit",
    "Qwen3FeatureBounds",
    "Qwen3FeatureBundle",
    "Qwen3ProjectionCoefficients",
    "Qwen3RuntimeAssignmentChange",
    "Qwen3RuntimeAssignmentSnapshot",
    "Qwen3RuntimeAudit",
    "Qwen3SurgeryCandidateSelectionContext",
    "Qwen3SurgeryEvaluationContext",
    "Qwen3SurgeryEvidenceRefresh",
    "Qwen3SurgeryFeature",
    "Qwen3SurgeryRecomputeContext",
    "apply_qwen3_adaptive_surgery",
    "build_qwen3_adaptive_surgery_graph",
    "qwen3_adaptive_features_from_channels",
    "qwen3_neuron_codebook_storage_bytes",
    "qwen3_neuron_node_id",
    "qwen3_runtime_assignment_snapshot",
    "surgically_compress_qwen3_codebook",
]
