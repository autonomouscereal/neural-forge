"""Deterministic adaptive prototype surgery over arbitrary logical nodes.

The graph starts with one real prototype per node.  Candidate edges are found
from feature geometry using a bounded collection of projection-order neighbor
windows; logical identifiers are never interpreted as coordinates.  Accepted
transactions merge prototype groups around a real member medoid and return a
new immutable snapshot.  No optimizer, autograd operation, or dense pairwise
distance matrix is used.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, overload

import numpy as np


SCHEMA_VERSION = 1
ACTION_KINDS = ("preserve", "share", "merge", "project", "compose", "skip")
ANCHOR_MODES = ("exact", "projected", "soft")
_EPSILON = np.finfo(np.float64).eps
_UNSET = object()

NodeId = tuple[Any, ...]


class AdaptiveSurgeryError(ValueError):
    """Base class for adaptive-surgery validation and state failures."""


class SurgeryShapeError(AdaptiveSurgeryError):
    """A node, feature vector, candidate, or target has an invalid shape."""


class SurgeryFiniteError(AdaptiveSurgeryError):
    """A feature or score control contains a non-finite value."""


class SurgeryStateError(AdaptiveSurgeryError):
    """A stale or foreign proposal cannot be applied to the current graph."""


class SurgeryTargetError(AdaptiveSurgeryError):
    """A requested prototype target is invalid or anchor-infeasible."""


class SurgerySearchExhausted(AdaptiveSurgeryError):
    """No unevaluated valid candidate can reach the requested target."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _id_payload(value: Any) -> Any:
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SurgeryFiniteError(
                "Node identifiers cannot contain non-finite floats."
            )
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_id_payload(item) for item in value]}
    raise SurgeryShapeError(
        "Node IDs must be tuples containing only stable scalar values or tuples; "
        f"got {type(value).__name__}."
    )


def _public_id(node_id: NodeId | None) -> list[Any] | None:
    if node_id is None:
        return None

    def convert(value: Any) -> Any:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        return value

    return convert(node_id)


def _normalize_node_id(value: Any, name: str = "node_id") -> NodeId:
    if not isinstance(value, tuple) or not value:
        raise SurgeryShapeError(f"{name} must be a non-empty tuple.")
    _id_payload(value)
    try:
        hash(value)
    except TypeError as error:
        raise SurgeryShapeError(f"{name} must be hashable.") from error
    return value


def _id_sort_key(node_id: NodeId) -> bytes:
    return _canonical_json_bytes(_id_payload(node_id))


def _as_vector(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        raise SurgeryShapeError(f"{name} is required.")
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
        raise SurgeryShapeError(f"{name} must be a numeric feature vector.") from error
    if array.ndim == 0 or array.size == 0:
        raise SurgeryShapeError(f"{name} must contain at least one feature value.")
    flat = np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
    if not bool(np.isfinite(flat).all()):
        raise SurgeryFiniteError(f"{name} must contain only finite values.")
    return tuple(float(item) for item in flat)


def _optional_vector(value: Any, name: str) -> tuple[float, ...] | None:
    return None if value is None else _as_vector(value, name)


def _vector_sha256(value: tuple[float, ...] | None) -> str | None:
    if value is None:
        return None
    array = np.asarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(len(value).to_bytes(8, "little", signed=False))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SurgeryShapeError(f"{name} must be a finite non-negative number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise SurgeryShapeError(
            f"{name} must be a finite non-negative number."
        ) from error
    if not math.isfinite(normalized):
        raise SurgeryFiniteError(f"{name} must be finite.")
    if normalized < 0.0:
        raise SurgeryShapeError(f"{name} must be non-negative.")
    return normalized


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SurgeryShapeError(f"{name} must be a finite number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise SurgeryShapeError(f"{name} must be a finite number.") from error
    if not math.isfinite(normalized):
        raise SurgeryFiniteError(f"{name} must be finite.")
    return normalized


def _positive_float(value: Any, name: str) -> float:
    normalized = _finite_nonnegative(value, name)
    if normalized <= 0.0:
        raise SurgeryShapeError(f"{name} must be positive.")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SurgeryShapeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SurgeryShapeError(f"{name} must be a non-negative integer.")
    return value


def _signed_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SurgeryShapeError(f"{name} must be an integer.")
    return value


def _coalesce_feature(
    primary: Any,
    alias: Any,
    primary_name: str,
    alias_name: str,
) -> Any:
    if primary is not None and alias is not None:
        raise SurgeryShapeError(f"Provide only one of {primary_name} and {alias_name}.")
    return primary if primary is not None else alias


@dataclass(frozen=True, slots=True)
class AnchorSpec:
    """Constraint applied when a node stops being its own real prototype."""

    mode: Literal["exact", "projected", "soft"] = "exact"
    penalty: float = 1.0
    max_damage: float | None = None
    projection_coefficient: float | None = None

    def __post_init__(self) -> None:
        normalized = str(self.mode).strip().lower().replace("-", "_")
        aliases = {"pinned": "exact", "hard": "exact", "projection": "projected"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in ANCHOR_MODES:
            raise SurgeryShapeError(f"anchor mode must be one of {ANCHOR_MODES!r}.")
        object.__setattr__(self, "mode", normalized)
        object.__setattr__(
            self,
            "penalty",
            _finite_nonnegative(self.penalty, "anchor penalty"),
        )
        if self.max_damage is not None:
            object.__setattr__(
                self,
                "max_damage",
                _finite_nonnegative(self.max_damage, "anchor max_damage"),
            )
        if normalized == "projected":
            coefficient = (
                1.0
                if self.projection_coefficient is None
                else self.projection_coefficient
            )
            object.__setattr__(
                self,
                "projection_coefficient",
                _finite_float(coefficient, "projection coefficient"),
            )
        elif self.projection_coefficient is not None:
            raise SurgeryShapeError(
                "projection_coefficient is only valid for projected anchors."
            )

    @classmethod
    def pinned(cls) -> AnchorSpec:
        return cls(mode="exact", penalty=0.0)

    @classmethod
    def projected(
        cls,
        penalty: float = 1.0,
        *,
        max_damage: float | None = None,
        coefficient: float = 1.0,
        projection_coefficient: float | None = None,
    ) -> AnchorSpec:
        if projection_coefficient is not None:
            if coefficient != 1.0:
                raise SurgeryShapeError(
                    "Provide only one of coefficient and projection_coefficient."
                )
            coefficient = projection_coefficient
        return cls(
            mode="projected",
            penalty=penalty,
            max_damage=max_damage,
            projection_coefficient=coefficient,
        )

    @classmethod
    def soft(
        cls,
        penalty: float = 0.25,
        *,
        max_damage: float | None = None,
    ) -> AnchorSpec:
        return cls(mode="soft", penalty=penalty, max_damage=max_damage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "penalty": self.penalty,
            "max_damage": self.max_damage,
            "projection_coefficient": self.projection_coefficient,
        }


def _coerce_anchor(value: AnchorSpec | str | None) -> AnchorSpec | None:
    if value is None:
        return None
    if isinstance(value, AnchorSpec):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"exact", "pinned", "hard"}:
        return AnchorSpec.pinned()
    if normalized in {"projected", "projection"}:
        return AnchorSpec.projected()
    if normalized == "soft":
        return AnchorSpec.soft()
    raise SurgeryShapeError(f"Unknown anchor mode {value!r}.")


@dataclass(frozen=True, slots=True, init=False)
class SurgeryNode:
    """One arbitrary logical node with independent surgery feature channels."""

    node_id: NodeId
    weight_features: tuple[float, ...]
    activation_features: tuple[float, ...]
    rollout_features: tuple[float, ...]
    output_features: tuple[float, ...]
    domain_features: tuple[float, ...] | None
    importance: float
    parameter_bytes: int
    anchor: AnchorSpec | None

    def __init__(
        self,
        node_id: NodeId,
        weight: Any = None,
        activation: Any = None,
        rollout: Any = None,
        output: Any = None,
        domain: Any = None,
        importance: float = 1.0,
        parameter_bytes: int = 0,
        anchor: AnchorSpec | str | None = None,
        *,
        weight_features: Any = None,
        activation_features: Any = None,
        rollout_features: Any = None,
        rollout_jvp: Any = None,
        rollout_jvp_features: Any = None,
        output_features: Any = None,
        output_dual: Any = None,
        output_dual_features: Any = None,
        domain_features: Any = None,
        parameter_byte_savings: int | None = None,
        savings_bytes: int | None = None,
    ) -> None:
        weight_value = _coalesce_feature(
            weight, weight_features, "weight", "weight_features"
        )
        activation_value = _coalesce_feature(
            activation,
            activation_features,
            "activation",
            "activation_features",
        )
        rollout_feature_alias = _coalesce_feature(
            rollout_features,
            rollout_jvp_features,
            "rollout_features",
            "rollout_jvp_features",
        )
        rollout_alias = _coalesce_feature(
            rollout_feature_alias,
            rollout_jvp,
            "rollout feature alias",
            "rollout_jvp",
        )
        rollout_value = _coalesce_feature(
            rollout, rollout_alias, "rollout", "rollout feature alias"
        )
        output_feature_alias = _coalesce_feature(
            output_features,
            output_dual_features,
            "output_features",
            "output_dual_features",
        )
        output_alias = _coalesce_feature(
            output_feature_alias,
            output_dual,
            "output feature alias",
            "output_dual",
        )
        output_value = _coalesce_feature(
            output, output_alias, "output", "output feature alias"
        )
        domain_value = _coalesce_feature(
            domain, domain_features, "domain", "domain_features"
        )
        if parameter_byte_savings is not None and savings_bytes is not None:
            raise SurgeryShapeError(
                "Provide only one of parameter_byte_savings and savings_bytes."
            )
        savings_alias = (
            parameter_byte_savings
            if parameter_byte_savings is not None
            else savings_bytes
        )
        if savings_alias is not None:
            if parameter_bytes != 0:
                raise SurgeryShapeError(
                    "Provide only one byte-savings alias and parameter_bytes."
                )
            parameter_bytes = savings_alias

        object.__setattr__(self, "node_id", _normalize_node_id(node_id))
        object.__setattr__(
            self, "weight_features", _as_vector(weight_value, "weight features")
        )
        object.__setattr__(
            self,
            "activation_features",
            _as_vector(activation_value, "activation features"),
        )
        object.__setattr__(
            self,
            "rollout_features",
            _as_vector(rollout_value, "rollout/JVP features"),
        )
        object.__setattr__(
            self,
            "output_features",
            _as_vector(output_value, "output/dual features"),
        )
        object.__setattr__(
            self,
            "domain_features",
            _optional_vector(domain_value, "domain features"),
        )
        object.__setattr__(
            self, "importance", _positive_float(importance, "importance")
        )
        object.__setattr__(
            self,
            "parameter_bytes",
            _nonnegative_int(parameter_bytes, "parameter_bytes"),
        )
        object.__setattr__(self, "anchor", _coerce_anchor(anchor))

    @property
    def weight(self) -> tuple[float, ...]:
        return self.weight_features

    @property
    def activation(self) -> tuple[float, ...]:
        return self.activation_features

    @property
    def rollout_jvp(self) -> tuple[float, ...]:
        return self.rollout_features

    @property
    def output_dual(self) -> tuple[float, ...]:
        return self.output_features

    @property
    def domain(self) -> tuple[float, ...] | None:
        return self.domain_features

    @property
    def parameter_byte_savings(self) -> int:
        return self.parameter_bytes

    @property
    def input_sha256(self) -> str:
        return _sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": _id_payload(self.node_id),
            "feature_hashes": {
                "weight": _vector_sha256(self.weight_features),
                "activation": _vector_sha256(self.activation_features),
                "rollout_jvp": _vector_sha256(self.rollout_features),
                "output_dual": _vector_sha256(self.output_features),
                "domain": _vector_sha256(self.domain_features),
            },
            "feature_dimensions": {
                "weight": len(self.weight_features),
                "activation": len(self.activation_features),
                "rollout_jvp": len(self.rollout_features),
                "output_dual": len(self.output_features),
                "domain": 0
                if self.domain_features is None
                else len(self.domain_features),
            },
            "importance": self.importance,
            "parameter_bytes": self.parameter_bytes,
            "anchor": None if self.anchor is None else self.anchor.as_dict(),
        }


# Descriptive aliases let callers use either graph-centric or prototype-centric names.
AdaptiveSurgeryNode = SurgeryNode
PrototypeSurgeryNode = SurgeryNode


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Explicit weights for every candidate score component."""

    weight: float = 0.25
    activation: float = 1.0
    rollout_jvp: float = 1.0
    output_dual: float = 1.0
    domain: float = 0.5
    anchor: float = 1.0
    savings: float = 1.0
    savings_byte_scale: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "weight",
            "activation",
            "rollout_jvp",
            "output_dual",
            "domain",
            "anchor",
            "savings",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), f"{name} score weight"),
            )
        if self.savings_byte_scale is not None:
            object.__setattr__(
                self,
                "savings_byte_scale",
                _positive_float(self.savings_byte_scale, "savings_byte_scale"),
            )

    def as_dict(self) -> dict[str, float | None]:
        return {
            "weight": self.weight,
            "activation": self.activation,
            "rollout_jvp": self.rollout_jvp,
            "output_dual": self.output_dual,
            "domain": self.domain,
            "anchor": self.anchor,
            "savings": self.savings,
            "savings_byte_scale": self.savings_byte_scale,
        }


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Parent-relative damage, savings, and utility for one proposed merge."""

    weight_damage: float
    activation_damage: float
    rollout_jvp_damage: float
    output_dual_damage: float
    domain_damage: float
    anchor_damage: float
    weighted_damage: float
    parameter_bytes_saved: int
    savings_reward: float
    total_score: float

    def __post_init__(self) -> None:
        for name in (
            "weight_damage",
            "activation_damage",
            "rollout_jvp_damage",
            "output_dual_damage",
            "domain_damage",
            "anchor_damage",
            "weighted_damage",
            "savings_reward",
            "total_score",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SurgeryShapeError(f"{name} must be a finite number.")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise SurgeryFiniteError(f"{name} must be finite.")
            object.__setattr__(self, name, normalized)
        object.__setattr__(
            self,
            "parameter_bytes_saved",
            _signed_int(self.parameter_bytes_saved, "parameter_bytes_saved"),
        )

    @property
    def rollout_damage(self) -> float:
        return self.rollout_jvp_damage

    @property
    def output_damage(self) -> float:
        return self.output_dual_damage

    @property
    def savings_bytes(self) -> int:
        return self.parameter_bytes_saved

    @property
    def score_sha256(self) -> str:
        return _sha256(self.as_dict())

    def as_dict(self) -> dict[str, float | int]:
        return {
            "weight_damage": self.weight_damage,
            "activation_damage": self.activation_damage,
            "rollout_jvp_damage": self.rollout_jvp_damage,
            "output_dual_damage": self.output_dual_damage,
            "domain_damage": self.domain_damage,
            "anchor_damage": self.anchor_damage,
            "weighted_damage": self.weighted_damage,
            "parameter_bytes_saved": self.parameter_bytes_saved,
            "savings_reward": self.savings_reward,
            "total_score": self.total_score,
        }


MetadataScalar = str | int | float | bool | None


def _freeze_metadata(
    value: Mapping[str, MetadataScalar] | Iterable[tuple[str, MetadataScalar]] | None,
    name: str,
) -> tuple[tuple[str, MetadataScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalized: list[tuple[str, MetadataScalar]] = []
    for key, item in items:
        if not isinstance(key, str):
            raise SurgeryShapeError(f"{name} keys must be strings.")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise SurgeryShapeError(f"{name} values must be JSON scalar values.")
        if isinstance(item, float) and not math.isfinite(item):
            raise SurgeryFiniteError(f"{name} values must be finite.")
        normalized.append((key, item))
    normalized.sort(key=lambda pair: pair[0])
    if len({key for key, _ in normalized}) != len(normalized):
        raise SurgeryShapeError(f"{name} keys must be unique.")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One immutable preserve/share/merge/project/compose/skip action."""

    kind: str
    sources: tuple[NodeId, ...] = ()
    target: NodeId | None = None
    reason: str = ""
    anchor_mode: str | None = None
    parent_action_sha256s: tuple[str, ...] = ()
    details: tuple[tuple[str, MetadataScalar], ...] = ()

    def __post_init__(self) -> None:
        normalized = str(self.kind).strip().lower().replace("-", "_")
        if normalized not in ACTION_KINDS:
            raise SurgeryShapeError(f"action kind must be one of {ACTION_KINDS!r}.")
        object.__setattr__(self, "kind", normalized)
        object.__setattr__(
            self,
            "sources",
            tuple(_normalize_node_id(item, "action source") for item in self.sources),
        )
        if self.target is not None:
            object.__setattr__(
                self, "target", _normalize_node_id(self.target, "action target")
            )
        if not isinstance(self.reason, str):
            raise SurgeryShapeError("action reason must be a string.")
        if self.anchor_mode is not None and self.anchor_mode not in ANCHOR_MODES:
            raise SurgeryShapeError(f"anchor_mode must be one of {ANCHOR_MODES!r}.")
        for digest in self.parent_action_sha256s:
            _validate_sha256(digest, "parent action hash")
        object.__setattr__(
            self, "details", _freeze_metadata(self.details, "action details")
        )

    @property
    def action_sha256(self) -> str:
        return _sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sources": [_id_payload(item) for item in self.sources],
            "target": None if self.target is None else _id_payload(self.target),
            "reason": self.reason,
            "anchor_mode": self.anchor_mode,
            "parent_action_sha256s": list(self.parent_action_sha256s),
            "details": dict(self.details),
        }


SurgeryAction = ActionRecord


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SurgeryShapeError(f"{name} must be a lowercase SHA-256 digest.")


@dataclass(frozen=True, slots=True)
class MergeCandidate:
    """One geometry-derived edge between two current prototype groups."""

    parent_snapshot_sha256: str
    evidence_sha256: str
    left_prototype: NodeId
    right_prototype: NodeId
    left_members: tuple[NodeId, ...]
    right_members: tuple[NodeId, ...]
    member_ids: tuple[NodeId, ...]
    resulting_medoid: NodeId
    medoid_candidates_considered: tuple[NodeId, ...]
    score: CandidateScore
    exact_anchor: NodeId | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.parent_snapshot_sha256, "parent_snapshot_sha256")
        _validate_sha256(self.evidence_sha256, "evidence_sha256")
        for name in (
            "left_prototype",
            "right_prototype",
            "resulting_medoid",
        ):
            object.__setattr__(
                self, name, _normalize_node_id(getattr(self, name), name)
            )
        for name in (
            "left_members",
            "right_members",
            "member_ids",
            "medoid_candidates_considered",
        ):
            object.__setattr__(
                self,
                name,
                tuple(_normalize_node_id(item, name) for item in getattr(self, name)),
            )
        if self.exact_anchor is not None:
            object.__setattr__(
                self,
                "exact_anchor",
                _normalize_node_id(self.exact_anchor, "exact_anchor"),
            )
        if self.left_prototype == self.right_prototype:
            raise SurgeryShapeError("A merge candidate needs two distinct prototypes.")
        if set(self.left_members) & set(self.right_members):
            raise SurgeryShapeError("Candidate prototype groups must be disjoint.")
        if set(self.member_ids) != set(self.left_members) | set(self.right_members):
            raise SurgeryShapeError("member_ids must be the union of both groups.")
        if self.resulting_medoid not in self.member_ids:
            raise SurgeryShapeError("resulting_medoid must be a real candidate member.")
        if self.resulting_medoid not in self.medoid_candidates_considered:
            raise SurgeryShapeError(
                "resulting_medoid must appear in medoid_candidates_considered."
            )
        if self.exact_anchor is not None and self.resulting_medoid != self.exact_anchor:
            raise SurgeryShapeError("An exact anchor must remain the resulting medoid.")

    @property
    def candidate_sha256(self) -> str:
        return _sha256(self.as_dict())

    @property
    def operation_sha256(self) -> str:
        """Hash only the executable cluster rewrite, not discovery evidence.

        ``candidate_sha256`` remains the exact identity of the scored candidate:
        it binds the parent snapshot, evidence, medoid search, and every score.
        This second digest is deliberately narrower.  It lets audit code prove
        that two independently discovered candidates describe the same cluster
        rewrite without pretending that their evidence or scores are equal.
        Callers must still verify ``candidate_sha256`` whenever exact discovery
        identity is required.
        """

        source_groups = [
            {
                "prototype": _id_payload(prototype),
                "members": [
                    _id_payload(item)
                    for item in sorted(members, key=_id_sort_key)
                ],
            }
            for prototype, members in (
                (self.left_prototype, self.left_members),
                (self.right_prototype, self.right_members),
            )
        ]
        source_groups.sort(key=_canonical_json_bytes)
        return _sha256(
            {
                "kind": "real-medoid-merge-operation-v1",
                "source_groups": source_groups,
                "member_ids": [
                    _id_payload(item)
                    for item in sorted(self.member_ids, key=_id_sort_key)
                ],
                "resulting_medoid": _id_payload(self.resulting_medoid),
                "exact_anchor": (
                    None
                    if self.exact_anchor is None
                    else _id_payload(self.exact_anchor)
                ),
            }
        )

    @property
    def sources(self) -> tuple[NodeId, NodeId]:
        return self.left_prototype, self.right_prototype

    def conflicts_with(self, other: MergeCandidate) -> bool:
        return bool(set(self.member_ids) & set(other.member_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "real-medoid-merge-candidate",
            "parent_snapshot_sha256": self.parent_snapshot_sha256,
            "evidence_sha256": self.evidence_sha256,
            "left_prototype": _id_payload(self.left_prototype),
            "right_prototype": _id_payload(self.right_prototype),
            "left_members": [_id_payload(item) for item in self.left_members],
            "right_members": [_id_payload(item) for item in self.right_members],
            "member_ids": [_id_payload(item) for item in self.member_ids],
            "resulting_medoid": _id_payload(self.resulting_medoid),
            "medoid_candidates_considered": [
                _id_payload(item) for item in self.medoid_candidates_considered
            ],
            "exact_anchor": (
                None if self.exact_anchor is None else _id_payload(self.exact_anchor)
            ),
            "score": self.score.as_dict(),
            "score_sha256": self.score.score_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateSearchDiagnostics:
    """Bounded-work evidence for one candidate recomputation."""

    search_round: int
    active_prototypes: int
    logical_nodes: int
    geometry_dimensions: int
    projection_count: int
    neighbor_window: int
    max_candidates: int
    medoid_candidate_limit: int
    projection_neighbor_comparisons: int
    projection_neighbor_comparison_bound: int
    unique_pair_count: int
    candidate_pairs_scored: int
    candidates_returned: int
    anchor_pairs_skipped: int
    rejected_candidates_filtered: int
    medoid_candidates_evaluated: int
    assignment_distance_evaluations: int
    assignment_distance_evaluation_bound: int
    distance_coordinate_operations: int
    distance_coordinate_operation_bound: int
    full_pairwise_matrix_elements: int
    materialized_full_pairwise_matrix: bool
    candidate_buffer_bound: int
    deterministic: bool = True
    bounded_work: bool = True
    optimizer_used: bool = False
    optimizer_steps: int = 0
    backward_used: bool = False
    backward_operations: int = 0

    @property
    def no_full_pairwise_matrix(self) -> bool:
        return not self.materialized_full_pairwise_matrix

    @property
    def no_training(self) -> bool:
        return (
            not self.optimizer_used
            and self.optimizer_steps == 0
            and not self.backward_used
            and self.backward_operations == 0
        )

    @property
    def pair_work_bound_formula(self) -> str:
        return "projection_count * active_prototypes * neighbor_window"

    @property
    def medoid_work_bound_formula(self) -> str:
        return (
            "logical_nodes + candidate_buffer_bound * "
            "medoid_candidate_limit * logical_nodes"
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {field: getattr(self, field) for field in self.__dataclass_fields__}
        payload.update(
            {
                "no_full_pairwise_matrix": self.no_full_pairwise_matrix,
                "no_training": self.no_training,
                "pair_work_bound_formula": self.pair_work_bound_formula,
                "medoid_work_bound_formula": self.medoid_work_bound_formula,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class CandidateSet(Sequence[MergeCandidate]):
    """Sequence of ranked candidates plus proof of bounded discovery work."""

    candidates: tuple[MergeCandidate, ...]
    diagnostics: CandidateSearchDiagnostics

    @overload
    def __getitem__(self, index: int) -> MergeCandidate: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[MergeCandidate, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> MergeCandidate | tuple[MergeCandidate, ...]:
        return self.candidates[index]

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[MergeCandidate]:
        return iter(self.candidates)


@dataclass(frozen=True, slots=True)
class AssignmentChange:
    """One logical node changing its assigned real prototype."""

    source: NodeId
    before: NodeId
    after: NodeId

    def __post_init__(self) -> None:
        for name in ("source", "before", "after"):
            object.__setattr__(
                self, name, _normalize_node_id(getattr(self, name), name)
            )

    @property
    def change_sha256(self) -> str:
        return _sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": _id_payload(self.source),
            "before": _id_payload(self.before),
            "after": _id_payload(self.after),
        }


@dataclass(frozen=True, slots=True)
class ProjectionBinding:
    """Executable scalar projection retained for one projected anchor."""

    source: NodeId
    target: NodeId
    coefficient: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", _normalize_node_id(self.source, "projection source")
        )
        object.__setattr__(
            self, "target", _normalize_node_id(self.target, "projection target")
        )
        if self.source == self.target:
            raise SurgeryShapeError(
                "A projection binding requires distinct source and target nodes."
            )
        object.__setattr__(
            self,
            "coefficient",
            _finite_float(self.coefficient, "projection coefficient"),
        )

    def apply(self, value: Any) -> float | tuple[float, ...]:
        """Apply the retained coefficient to a scalar or numeric vector."""

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return self.coefficient * _finite_float(value, "projection input")
        vector = _as_vector(value, "projection input")
        return tuple(self.coefficient * item for item in vector)

    @property
    def binding_sha256(self) -> str:
        return _sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": _id_payload(self.source),
            "target": _id_payload(self.target),
            "coefficient": self.coefficient,
        }


@dataclass(frozen=True, slots=True)
class SurgeryTransaction:
    """Full accepted-or-previewed transaction provenance."""

    schema_version: int
    transaction_index: int
    parent_snapshot_sha256: str
    parent_state_sha256: str
    candidate_sha256s: tuple[str, ...]
    candidates: tuple[MergeCandidate, ...]
    assignment_changes: tuple[AssignmentChange, ...]
    projection_bindings: tuple[ProjectionBinding, ...]
    actions: tuple[ActionRecord, ...]
    prototype_count_before: int
    prototype_count_after: int
    parameter_bytes_saved_increment: int
    parameter_bytes_saved_total: int
    assignments_sha256: str
    registry_sha256: str
    state_sha256: str
    transaction_sha256: str
    provenance_sha256: str

    @property
    def action_sha256s(self) -> tuple[str, ...]:
        return tuple(action.action_sha256 for action in self.actions)

    @property
    def total_score(self) -> float:
        return math.fsum(candidate.score.total_score for candidate in self.candidates)

    @property
    def real_medoid_assignments(self) -> tuple[tuple[NodeId, NodeId], ...]:
        return tuple(
            (change.source, change.after) for change in self.assignment_changes
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "adaptive-prototype-surgery-transaction",
            "transaction_index": self.transaction_index,
            "parent_snapshot_sha256": self.parent_snapshot_sha256,
            "parent_state_sha256": self.parent_state_sha256,
            "candidate_sha256s": list(self.candidate_sha256s),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "assignment_changes": [
                change.as_dict() for change in self.assignment_changes
            ],
            "projection_bindings": [
                binding.as_dict() for binding in self.projection_bindings
            ],
            "projection_binding_sha256s": [
                binding.binding_sha256 for binding in self.projection_bindings
            ],
            "actions": [action.as_dict() for action in self.actions],
            "action_sha256s": list(self.action_sha256s),
            "prototype_count_before": self.prototype_count_before,
            "prototype_count_after": self.prototype_count_after,
            "parameter_bytes_saved_increment": self.parameter_bytes_saved_increment,
            "parameter_bytes_saved_total": self.parameter_bytes_saved_total,
            "total_score": self.total_score,
            "assignments_sha256": self.assignments_sha256,
            "registry_sha256": self.registry_sha256,
            "state_sha256": self.state_sha256,
            "transaction_sha256": self.transaction_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True, slots=True)
class SurgerySnapshot:
    """Immutable assignments and provenance at one accepted graph state."""

    schema_version: int
    graph_sha256: str
    node_ids: tuple[NodeId, ...]
    assignment_pairs: tuple[tuple[NodeId, NodeId], ...]
    prototype_ids: tuple[NodeId, ...]
    assignment_indices: tuple[int, ...]
    prototype_source_indices: tuple[int, ...]
    parameter_bytes_saved: int
    projection_bindings: tuple[ProjectionBinding, ...]
    transactions: tuple[SurgeryTransaction, ...]
    transaction_count_total: int
    assignments_sha256: str
    registry_sha256: str
    state_sha256: str
    provenance_sha256: str
    snapshot_sha256: str

    @property
    def prototype_count(self) -> int:
        return len(self.prototype_ids)

    @property
    def transaction_count(self) -> int:
        return self.transaction_count_total

    @property
    def retained_transaction_count(self) -> int:
        return len(self.transactions)

    @property
    def is_exact(self) -> bool:
        return all(source == medoid for source, medoid in self.assignment_pairs)

    @property
    def assignments(self) -> Mapping[NodeId, NodeId]:
        return MappingProxyType(dict(self.assignment_pairs))

    @property
    def real_medoid_assignments(self) -> Mapping[NodeId, NodeId]:
        return self.assignments

    @property
    def prototype_source_ids(self) -> tuple[NodeId, ...]:
        return self.prototype_ids

    @property
    def flat_assignments(self) -> tuple[int, ...]:
        return self.assignment_indices

    @property
    def registry_assignments(self) -> tuple[int, ...]:
        return self.assignment_indices

    @property
    def actions(self) -> tuple[ActionRecord, ...]:
        return tuple(
            action
            for transaction in self.transactions
            for action in transaction.actions
        )

    def registry_payload(self) -> dict[str, Any]:
        """Return flat index buffers accepted by neuron/attention registries."""

        return {
            "node_ids": [_public_id(item) for item in self.node_ids],
            "assignments": list(self.assignment_indices),
            "prototype_source_indices": list(self.prototype_source_indices),
            "prototype_source_ids": [_public_id(item) for item in self.prototype_ids],
            "projection_bindings": [
                binding.as_dict() for binding in self.projection_bindings
            ],
            "assignments_sha256": self.assignments_sha256,
            "registry_sha256": self.registry_sha256,
            "provenance_sha256": self.provenance_sha256,
            "snapshot_sha256": self.snapshot_sha256,
        }

    def registry_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return detached, read-only int64 copies of the registry buffers."""

        assignments = np.asarray(self.assignment_indices, dtype=np.int64).copy()
        sources = np.asarray(self.prototype_source_indices, dtype=np.int64).copy()
        assignments.flags.writeable = False
        sources.flags.writeable = False
        return assignments, sources

    def registry_tensors(self) -> tuple[Any, Any]:
        """Return CPU int64 Torch buffers when Torch is installed."""

        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional project extra
            raise AdaptiveSurgeryError(
                "registry_tensors requires the optional torch dependency."
            ) from error
        return (
            torch.tensor(self.assignment_indices, dtype=torch.int64),
            torch.tensor(self.prototype_source_indices, dtype=torch.int64),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "adaptive-prototype-surgery-snapshot",
            "graph_sha256": self.graph_sha256,
            "node_ids": [_id_payload(item) for item in self.node_ids],
            "assignments": [
                {
                    "source": _id_payload(source),
                    "prototype": _id_payload(prototype),
                }
                for source, prototype in self.assignment_pairs
            ],
            "prototype_ids": [_id_payload(item) for item in self.prototype_ids],
            "assignment_indices": list(self.assignment_indices),
            "prototype_source_indices": list(self.prototype_source_indices),
            "prototype_count": self.prototype_count,
            "parameter_bytes_saved": self.parameter_bytes_saved,
            "projection_bindings": [
                binding.as_dict() for binding in self.projection_bindings
            ],
            "transactions": [
                transaction.as_dict() for transaction in self.transactions
            ],
            "transaction_count_total": self.transaction_count_total,
            "retained_transaction_count": self.retained_transaction_count,
            "hashes": {
                "assignments": self.assignments_sha256,
                "registry": self.registry_sha256,
                "state": self.state_sha256,
                "provenance": self.provenance_sha256,
                "snapshot": self.snapshot_sha256,
            },
            "is_exact": self.is_exact,
        }


@dataclass(frozen=True, slots=True, init=False)
class EvaluationDecision:
    """External parent-relative evaluator result."""

    accepted: bool
    reason: str
    metrics: tuple[tuple[str, MetadataScalar], ...]

    def __init__(
        self,
        accepted: bool,
        reason: str = "",
        metrics: (
            Mapping[str, MetadataScalar] | Iterable[tuple[str, MetadataScalar]] | None
        ) = None,
    ) -> None:
        if not isinstance(accepted, bool):
            raise SurgeryShapeError("EvaluationDecision.accepted must be boolean.")
        if not isinstance(reason, str):
            raise SurgeryShapeError("EvaluationDecision.reason must be a string.")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metrics", _freeze_metadata(metrics, "metrics"))

    @classmethod
    def accept(
        cls,
        reason: str = "",
        metrics: Mapping[str, MetadataScalar] | None = None,
    ) -> EvaluationDecision:
        return cls(True, reason=reason, metrics=metrics)

    @classmethod
    def reject(
        cls,
        reason: str = "",
        metrics: Mapping[str, MetadataScalar] | None = None,
    ) -> EvaluationDecision:
        return cls(False, reason=reason, metrics=metrics)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class SurgeryProposal:
    """Immutable parent/proposed pair passed to an external evaluator."""

    parent: SurgerySnapshot
    snapshot: SurgerySnapshot
    transaction: SurgeryTransaction

    @property
    def proposal_sha256(self) -> str:
        return _sha256(
            {
                "parent_snapshot_sha256": self.parent.snapshot_sha256,
                "proposed_snapshot_sha256": self.snapshot.snapshot_sha256,
                "transaction_sha256": self.transaction.transaction_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Accepted or rejected evaluator resolution, including skip provenance."""

    attempt_index: int
    parent_snapshot_sha256: str
    proposed_snapshot_sha256: str
    transaction_sha256: str
    accepted: bool
    decision: EvaluationDecision
    actions: tuple[ActionRecord, ...]
    attempt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "parent_snapshot_sha256": self.parent_snapshot_sha256,
            "proposed_snapshot_sha256": self.proposed_snapshot_sha256,
            "transaction_sha256": self.transaction_sha256,
            "accepted": self.accepted,
            "decision": self.decision.as_dict(),
            "actions": [action.as_dict() for action in self.actions],
            "action_sha256s": [action.action_sha256 for action in self.actions],
            "attempt_sha256": self.attempt_sha256,
        }


@dataclass(frozen=True, slots=True)
class SurgeryOutcome:
    """Resolution of one proposal; rejection returns the unchanged parent."""

    accepted: bool
    snapshot: SurgerySnapshot
    proposed_snapshot: SurgerySnapshot
    transaction: SurgeryTransaction
    decision: EvaluationDecision
    attempt: AttemptRecord

    @property
    def rolled_back(self) -> bool:
        return not self.accepted


@dataclass(frozen=True, slots=True)
class GraphWorkDiagnostics:
    """Cumulative work and execution evidence for the mutable orchestrator."""

    candidate_recomputations: int
    accepted_transactions: int
    rejected_transactions: int
    projection_neighbor_comparisons: int
    candidate_pairs_scored: int
    assignment_distance_evaluations: int
    distance_coordinate_operations: int
    candidate_pair_buffer_limit: int
    history_limit: int
    retained_snapshots: int
    retained_attempts: int
    retained_transactions: int
    pending_proposals: int
    singleton_rejections: int
    materialized_full_pairwise_matrix: bool = False
    optimizer_used: bool = False
    optimizer_steps: int = 0
    backward_used: bool = False
    backward_operations: int = 0

    @property
    def no_training(self) -> bool:
        return (
            not self.optimizer_used
            and self.optimizer_steps == 0
            and not self.backward_used
            and self.backward_operations == 0
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {field: getattr(self, field) for field in self.__dataclass_fields__}
        payload["no_training"] = self.no_training
        return payload


@dataclass(frozen=True, slots=True)
class _Cluster:
    medoid: int
    members: tuple[int, ...]


@dataclass(slots=True)
class _SearchWork:
    projection_comparisons: int = 0
    anchor_pairs_skipped: int = 0
    rejected_filtered: int = 0
    medoid_candidates: int = 0
    assignment_evaluations: int = 0
    coordinate_operations: int = 0


@dataclass(frozen=True, slots=True)
class _Cost:
    weight: float
    activation: float
    rollout_jvp: float
    output_dual: float
    domain: float
    anchor: float

    def subtract(self, other: _Cost) -> _Cost:
        return _Cost(
            weight=self.weight - other.weight,
            activation=self.activation - other.activation,
            rollout_jvp=self.rollout_jvp - other.rollout_jvp,
            output_dual=self.output_dual - other.output_dual,
            domain=self.domain - other.domain,
            anchor=self.anchor - other.anchor,
        )

    def add(self, other: _Cost) -> _Cost:
        return _Cost(
            weight=self.weight + other.weight,
            activation=self.activation + other.activation,
            rollout_jvp=self.rollout_jvp + other.rollout_jvp,
            output_dual=self.output_dual + other.output_dual,
            domain=self.domain + other.domain,
            anchor=self.anchor + other.anchor,
        )


_ZERO_COST = _Cost(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


Evaluator = Callable[..., bool | EvaluationDecision | Mapping[str, Any]]
EvidenceRefresh = Callable[..., Sequence[SurgeryNode] | Mapping[Any, Any]]
ProjectionDamageScorer = Callable[
    [ProjectionBinding, SurgeryNode, SurgeryNode, tuple[float, float, float, float, float]],
    float,
]


@dataclass(frozen=True, slots=True)
class _EvidenceState:
    """Evidence and ranked candidates valid for one accepted snapshot epoch."""

    nodes: tuple[SurgeryNode, ...]
    evidence_sha256: str
    features: dict[str, np.ndarray]
    feature_scales: dict[str, float]
    domain_present: np.ndarray
    feature_dimension_total: int
    normalized_geometry: np.ndarray
    candidate_cache: CandidateSet | None


@dataclass(frozen=True, slots=True)
class _GraphCheckpoint:
    nodes: tuple[SurgeryNode, ...]
    evidence_sha256: str
    features: dict[str, np.ndarray]
    feature_scales: dict[str, float]
    domain_present: np.ndarray
    feature_dimension_total: int
    normalized_geometry: np.ndarray
    snapshot: SurgerySnapshot
    known_snapshots: tuple[tuple[str, SurgerySnapshot], ...]
    evidence_states: tuple[tuple[str, _EvidenceState], ...]
    candidate_cache: CandidateSet | None
    rejected_by_parent: tuple[tuple[str, tuple[str, ...]], ...]
    pending_proposals: tuple[tuple[str, SurgeryProposal], ...]
    attempts: tuple[AttemptRecord, ...]
    attempt_count: int
    counters: tuple[int, int, int, int, int, int, int]


class AdaptiveSurgeryGraph:
    """Sequential, deterministic real-medoid prototype folding graph."""

    def __init__(
        self,
        nodes: Sequence[SurgeryNode],
        *,
        score_weights: ScoreWeights | None = None,
        pinned_anchors: Iterable[NodeId] = (),
        projected_anchors: (
            Mapping[NodeId, float | AnchorSpec] | Iterable[NodeId]
        ) = (),
        soft_anchors: Mapping[NodeId, float | AnchorSpec] | Iterable[NodeId] = (),
        projection_count: int = 8,
        neighbor_window: int = 4,
        max_candidates: int = 64,
        medoid_candidate_limit: int = 32,
        candidate_pair_limit: int | None = None,
        history_limit: int = 32,
        evidence_refresh: EvidenceRefresh | None = None,
        evidence_refresh_callback: EvidenceRefresh | None = None,
        projection_damage_callback: ProjectionDamageScorer | None = None,
    ) -> None:
        if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Sequence):
            raise SurgeryShapeError("nodes must be a non-empty sequence.")
        if not nodes:
            raise SurgeryShapeError("nodes must be a non-empty sequence.")
        if not all(isinstance(node, SurgeryNode) for node in nodes):
            raise SurgeryShapeError("Every node must be a SurgeryNode.")
        self._nodes = tuple(nodes)
        self._node_ids = tuple(node.node_id for node in nodes)
        if len(set(self._node_ids)) != len(self._node_ids):
            raise SurgeryShapeError("node_id values must be unique.")
        self._index = {node_id: index for index, node_id in enumerate(self._node_ids)}
        self._evidence_sha256 = _sha256([node.as_dict() for node in self._nodes])
        self._score_weights = score_weights or ScoreWeights()
        if not isinstance(self._score_weights, ScoreWeights):
            raise SurgeryShapeError("score_weights must be a ScoreWeights instance.")
        self._projection_count = _positive_int(projection_count, "projection_count")
        self._neighbor_window = _positive_int(neighbor_window, "neighbor_window")
        self._max_candidates = _positive_int(max_candidates, "max_candidates")
        self._medoid_candidate_limit = _positive_int(
            medoid_candidate_limit, "medoid_candidate_limit"
        )
        if candidate_pair_limit is None:
            self._candidate_pair_limit = max(64, self._max_candidates * 4)
        else:
            self._candidate_pair_limit = _positive_int(
                candidate_pair_limit, "candidate_pair_limit"
            )
        self._history_limit = _positive_int(history_limit, "history_limit")
        if evidence_refresh is not None and evidence_refresh_callback is not None:
            raise SurgeryShapeError(
                "Provide only one of evidence_refresh and evidence_refresh_callback."
            )
        self._evidence_refresh = (
            evidence_refresh
            if evidence_refresh is not None
            else evidence_refresh_callback
        )
        if self._evidence_refresh is not None:
            refresh_callable = getattr(
                self._evidence_refresh, "refresh", self._evidence_refresh
            )
            if not callable(refresh_callable):
                raise SurgeryShapeError(
                    "evidence_refresh must be callable or expose refresh()."
                )
        if projection_damage_callback is not None and not callable(
            projection_damage_callback
        ):
            raise SurgeryShapeError("projection_damage_callback must be callable.")
        self._projection_damage_callback = projection_damage_callback

        self._features, self._feature_scales, self._domain_present = (
            self._build_feature_matrices()
        )
        self._feature_dimension_total = sum(
            matrix.shape[1] for matrix in self._features.values()
        )
        self._anchors = self._resolve_anchors(
            pinned_anchors, projected_anchors, soft_anchors
        )
        positive_bytes = sorted(
            node.parameter_bytes for node in self._nodes if node.parameter_bytes > 0
        )
        automatic_byte_scale = (
            float(positive_bytes[(len(positive_bytes) - 1) // 2])
            if positive_bytes
            else 1.0
        )
        self._savings_byte_scale = (
            automatic_byte_scale
            if self._score_weights.savings_byte_scale is None
            else self._score_weights.savings_byte_scale
        )
        self._normalized_geometry = self._build_normalized_geometry()

        self._graph_sha256 = _sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "adaptive-prototype-surgery-graph",
                "nodes": [node.as_dict() for node in self._nodes],
                "anchors": [
                    None if anchor is None else anchor.as_dict()
                    for anchor in self._anchors
                ],
                "score_weights": self._score_weights.as_dict(),
                "savings_byte_scale": self._savings_byte_scale,
                "search": {
                    "projection_count": self._projection_count,
                    "neighbor_window": self._neighbor_window,
                    "max_candidates": self._max_candidates,
                    "medoid_candidate_limit": self._medoid_candidate_limit,
                    "candidate_pair_limit": self._candidate_pair_limit,
                    "history_limit": self._history_limit,
                },
            }
        )
        self._root_provenance_sha256 = _sha256(
            {
                "kind": "adaptive-prototype-surgery-root",
                "graph_sha256": self._graph_sha256,
            }
        )
        exact_assignments = tuple(range(len(self._nodes)))
        self._snapshot = self._initial_snapshot(exact_assignments)
        self._known_snapshots: OrderedDict[str, SurgerySnapshot] = OrderedDict()
        self._evidence_states: OrderedDict[str, _EvidenceState] = OrderedDict()
        self._candidate_cache: CandidateSet | None = None
        self._remember_snapshot(self._snapshot)
        self._rejected_by_parent: OrderedDict[str, OrderedDict[str, None]] = (
            OrderedDict()
        )
        self._pending_proposals: OrderedDict[str, SurgeryProposal] = OrderedDict()
        self._attempts: list[AttemptRecord] = []
        self._attempt_count = 0
        self._candidate_recomputations = 0
        self._accepted_transactions = 0
        self._rejected_transactions = 0
        self._projection_neighbor_comparisons = 0
        self._candidate_pairs_scored = 0
        self._assignment_distance_evaluations = 0
        self._distance_coordinate_operations = 0

    @property
    def nodes(self) -> tuple[SurgeryNode, ...]:
        return self._nodes

    @property
    def snapshot(self) -> SurgerySnapshot:
        return self._snapshot

    @property
    def current_snapshot(self) -> SurgerySnapshot:
        return self._snapshot

    @property
    def graph_sha256(self) -> str:
        return self._graph_sha256

    @property
    def evidence_sha256(self) -> str:
        return self._evidence_sha256

    @property
    def attempts(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._attempts)

    @property
    def snapshot_history(self) -> tuple[SurgerySnapshot, ...]:
        """Return the bounded tail of restorable immutable snapshots."""

        return tuple(self._known_snapshots.values())

    @property
    def history_limit(self) -> int:
        return self._history_limit

    @property
    def candidate_pair_limit(self) -> int:
        return self._candidate_pair_limit

    @property
    def last_search_diagnostics(self) -> CandidateSearchDiagnostics | None:
        return (
            None if self._candidate_cache is None else self._candidate_cache.diagnostics
        )

    @property
    def work_diagnostics(self) -> GraphWorkDiagnostics:
        return GraphWorkDiagnostics(
            candidate_recomputations=self._candidate_recomputations,
            accepted_transactions=self._accepted_transactions,
            rejected_transactions=self._rejected_transactions,
            projection_neighbor_comparisons=self._projection_neighbor_comparisons,
            candidate_pairs_scored=self._candidate_pairs_scored,
            assignment_distance_evaluations=self._assignment_distance_evaluations,
            distance_coordinate_operations=self._distance_coordinate_operations,
            candidate_pair_buffer_limit=self._candidate_pair_limit,
            history_limit=self._history_limit,
            retained_snapshots=len(self._known_snapshots),
            retained_attempts=len(self._attempts),
            retained_transactions=len(self._snapshot.transactions),
            pending_proposals=len(self._pending_proposals),
            singleton_rejections=sum(
                len(rejections) for rejections in self._rejected_by_parent.values()
            ),
        )

    @property
    def diagnostics(self) -> GraphWorkDiagnostics:
        return self.work_diagnostics

    def _build_feature_matrices(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
        feature_names = {
            "weight": "weight_features",
            "activation": "activation_features",
            "rollout_jvp": "rollout_features",
            "output_dual": "output_features",
        }
        matrices: dict[str, np.ndarray] = {}
        scales: dict[str, float] = {}
        for feature_name, attribute in feature_names.items():
            rows = [getattr(node, attribute) for node in self._nodes]
            dimensions = {len(row) for row in rows}
            if len(dimensions) != 1:
                raise SurgeryShapeError(
                    f"All {feature_name} feature vectors must have the same length."
                )
            matrix = np.asarray(rows, dtype=np.float64)
            matrices[feature_name] = matrix
            scales[feature_name] = self._feature_scale(matrix)

        domain_rows = [node.domain_features for node in self._nodes]
        present_dimensions = {len(row) for row in domain_rows if row is not None}
        if len(present_dimensions) > 1:
            raise SurgeryShapeError(
                "All provided domain feature vectors must have the same length."
            )
        domain_present = np.asarray(
            [row is not None for row in domain_rows], dtype=np.bool_
        )
        if present_dimensions:
            domain_dimension = next(iter(present_dimensions))
            domain_matrix = np.zeros(
                (len(self._nodes), domain_dimension), dtype=np.float64
            )
            for index, row in enumerate(domain_rows):
                if row is not None:
                    domain_matrix[index] = row
        else:
            domain_matrix = np.zeros((len(self._nodes), 0), dtype=np.float64)
        matrices["domain"] = domain_matrix
        scales["domain"] = self._feature_scale(domain_matrix)
        return matrices, scales, domain_present

    @staticmethod
    def _feature_scale(matrix: np.ndarray) -> float:
        if matrix.shape[1] == 0:
            return 1.0
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        scale = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
        return scale if scale > _EPSILON else 1.0

    def _resolve_anchors(
        self,
        pinned_anchors: Iterable[NodeId],
        projected_anchors: Mapping[NodeId, float | AnchorSpec] | Iterable[NodeId],
        soft_anchors: Mapping[NodeId, float | AnchorSpec] | Iterable[NodeId],
    ) -> tuple[AnchorSpec | None, ...]:
        anchors = [node.anchor for node in self._nodes]

        def assign(node_id: NodeId, anchor: AnchorSpec) -> None:
            normalized_id = _normalize_node_id(node_id, "anchor node ID")
            if normalized_id not in self._index:
                raise SurgeryShapeError(f"Unknown anchor node ID {normalized_id!r}.")
            index = self._index[normalized_id]
            current = anchors[index]
            if current is not None and current != anchor:
                raise SurgeryShapeError(
                    f"Node {normalized_id!r} has conflicting anchor specifications."
                )
            anchors[index] = anchor

        for node_id in pinned_anchors:
            assign(node_id, AnchorSpec.pinned())
        self._assign_soft_anchor_collection(projected_anchors, "projected", assign)
        self._assign_soft_anchor_collection(soft_anchors, "soft", assign)
        return tuple(anchors)

    @staticmethod
    def _assign_soft_anchor_collection(
        collection: Mapping[NodeId, float | AnchorSpec] | Iterable[NodeId],
        mode: Literal["projected", "soft"],
        assign: Callable[[NodeId, AnchorSpec], None],
    ) -> None:
        if isinstance(collection, Mapping):
            items: Iterable[tuple[NodeId, float | AnchorSpec]] = collection.items()
        else:
            default = (
                AnchorSpec.projected() if mode == "projected" else AnchorSpec.soft()
            )
            items = ((node_id, default) for node_id in collection)
        for node_id, value in items:
            if isinstance(value, AnchorSpec):
                anchor = value
                if anchor.mode != mode:
                    raise SurgeryShapeError(
                        f"{mode}_anchors contains an AnchorSpec with mode {anchor.mode!r}."
                    )
            elif mode == "projected":
                anchor = AnchorSpec.projected(float(value))
            else:
                anchor = AnchorSpec.soft(float(value))
            assign(node_id, anchor)

    def _build_normalized_geometry(self) -> np.ndarray:
        components: list[np.ndarray] = []
        geometry_weights = {
            "weight": self._score_weights.weight,
            "activation": self._score_weights.activation,
            "rollout_jvp": self._score_weights.rollout_jvp,
            "output_dual": self._score_weights.output_dual,
            "domain": self._score_weights.domain,
        }
        for name in ("weight", "activation", "rollout_jvp", "output_dual"):
            matrix = self._features[name]
            coefficient = math.sqrt(geometry_weights[name])
            if coefficient > 0.0:
                components.append(matrix * (coefficient / self._feature_scales[name]))
        domain = self._features["domain"]
        domain_coefficient = math.sqrt(geometry_weights["domain"])
        if domain_coefficient > 0.0:
            if domain.shape[1] > 0:
                components.append(
                    domain * (domain_coefficient / self._feature_scales["domain"])
                )
            components.append(
                self._domain_present.astype(np.float64)[:, None] * domain_coefficient
            )
        if not components:
            return np.zeros((len(self._nodes), 1), dtype=np.float64)
        return np.ascontiguousarray(np.concatenate(components, axis=1))

    def _projection_bindings(
        self, medoid_by_node: tuple[int, ...]
    ) -> tuple[ProjectionBinding, ...]:
        bindings: list[ProjectionBinding] = []
        for source, medoid in enumerate(medoid_by_node):
            anchor = self._anchors[source]
            if anchor is None or anchor.mode != "projected" or source == medoid:
                continue
            if anchor.projection_coefficient is None:  # pragma: no cover - validated
                raise SurgeryStateError("Projected anchor has no coefficient.")
            bindings.append(
                ProjectionBinding(
                    source=self._node_ids[source],
                    target=self._node_ids[medoid],
                    coefficient=anchor.projection_coefficient,
                )
            )
        return tuple(bindings)

    def _state_material(
        self, medoid_by_node: tuple[int, ...]
    ) -> tuple[
        tuple[tuple[NodeId, NodeId], ...],
        tuple[NodeId, ...],
        tuple[int, ...],
        tuple[int, ...],
        int,
        str,
        str,
        str,
        tuple[ProjectionBinding, ...],
    ]:
        node_count = len(self._nodes)
        if len(medoid_by_node) != node_count:
            raise SurgeryStateError("Internal assignment length mismatch.")
        prototypes = tuple(sorted(set(medoid_by_node)))
        if any(index < 0 or index >= node_count for index in prototypes):
            raise SurgeryStateError("Internal assignment contains an invalid medoid.")
        if any(medoid_by_node[medoid] != medoid for medoid in prototypes):
            raise SurgeryStateError("Every real medoid must assign to itself.")
        slot_by_medoid = {medoid: slot for slot, medoid in enumerate(prototypes)}
        assignment_indices = tuple(slot_by_medoid[medoid] for medoid in medoid_by_node)
        prototype_ids = tuple(self._node_ids[index] for index in prototypes)
        assignment_pairs = tuple(
            (self._node_ids[source], self._node_ids[medoid])
            for source, medoid in enumerate(medoid_by_node)
        )
        parameter_bytes_saved = sum(
            self._nodes[index].parameter_bytes
            for index, medoid in enumerate(medoid_by_node)
            if index != medoid
        )
        assignments_sha256 = _sha256(
            [
                {
                    "source": _id_payload(source),
                    "prototype": _id_payload(prototype),
                }
                for source, prototype in assignment_pairs
            ]
        )
        projection_bindings = self._projection_bindings(medoid_by_node)
        registry_sha256 = _sha256(
            {
                "assignments": list(assignment_indices),
                "prototype_source_indices": list(prototypes),
                "node_ids": [_id_payload(item) for item in self._node_ids],
                "projection_bindings": [
                    binding.as_dict() for binding in projection_bindings
                ],
            }
        )
        state_sha256 = _sha256(
            {
                "graph_sha256": self._graph_sha256,
                "assignments_sha256": assignments_sha256,
                "registry_sha256": registry_sha256,
                "parameter_bytes_saved": parameter_bytes_saved,
                "projection_binding_sha256s": [
                    binding.binding_sha256 for binding in projection_bindings
                ],
            }
        )
        return (
            assignment_pairs,
            prototype_ids,
            assignment_indices,
            prototypes,
            parameter_bytes_saved,
            assignments_sha256,
            registry_sha256,
            state_sha256,
            projection_bindings,
        )

    def _initial_snapshot(self, medoid_by_node: tuple[int, ...]) -> SurgerySnapshot:
        material = self._state_material(medoid_by_node)
        snapshot_sha256 = _sha256(
            {
                "state_sha256": material[7],
                "provenance_sha256": self._root_provenance_sha256,
                "transaction_count": 0,
            }
        )
        return SurgerySnapshot(
            schema_version=SCHEMA_VERSION,
            graph_sha256=self._graph_sha256,
            node_ids=self._node_ids,
            assignment_pairs=material[0],
            prototype_ids=material[1],
            assignment_indices=material[2],
            prototype_source_indices=material[3],
            parameter_bytes_saved=material[4],
            projection_bindings=material[8],
            transactions=(),
            transaction_count_total=0,
            assignments_sha256=material[5],
            registry_sha256=material[6],
            state_sha256=material[7],
            provenance_sha256=self._root_provenance_sha256,
            snapshot_sha256=snapshot_sha256,
        )

    def _medoid_indices(self, snapshot: SurgerySnapshot) -> tuple[int, ...]:
        return tuple(
            snapshot.prototype_source_indices[slot]
            for slot in snapshot.assignment_indices
        )

    def _clusters(self, snapshot: SurgerySnapshot) -> tuple[_Cluster, ...]:
        medoid_indices = self._medoid_indices(snapshot)
        members_by_medoid: dict[int, list[int]] = {}
        for member, medoid in enumerate(medoid_indices):
            members_by_medoid.setdefault(medoid, []).append(member)
        return tuple(
            _Cluster(medoid=medoid, members=tuple(members_by_medoid[medoid]))
            for medoid in sorted(members_by_medoid)
        )

    def _cluster_centroid(self, cluster: _Cluster) -> np.ndarray:
        member_array = np.asarray(cluster.members, dtype=np.int64)
        importance = np.asarray(
            [self._nodes[index].importance for index in cluster.members],
            dtype=np.float64,
        )
        return np.average(
            self._normalized_geometry[member_array],
            axis=0,
            weights=importance,
        )

    @staticmethod
    def _projection_direction(dimension: int, projection: int) -> np.ndarray:
        direction = np.empty(dimension, dtype=np.float64)
        for coordinate in range(dimension):
            digest = hashlib.sha256(
                f"adaptive-surgery:{projection}:{coordinate}".encode("ascii")
            ).digest()
            direction[coordinate] = 1.0 if digest[0] & 1 else -1.0
        norm = float(np.linalg.norm(direction))
        return direction / norm

    def _candidate_pairs(
        self,
        clusters: tuple[_Cluster, ...],
        work: _SearchWork,
    ) -> set[tuple[int, int]]:
        if len(clusters) < 2:
            return set()
        centroids = np.vstack([self._cluster_centroid(cluster) for cluster in clusters])
        pairs: dict[tuple[int, int], tuple[float, bytes, bytes]] = {}
        for projection in range(self._projection_count):
            direction = self._projection_direction(centroids.shape[1], projection)
            values = centroids @ direction
            order = sorted(
                range(len(clusters)),
                key=lambda index: (
                    float(values[index]),
                    _id_sort_key(self._node_ids[clusters[index].medoid]),
                ),
            )
            for position, left in enumerate(order):
                stop = min(len(order), position + self._neighbor_window + 1)
                for right_position in range(position + 1, stop):
                    right = order[right_position]
                    work.projection_comparisons += 1
                    pair = (min(left, right), max(left, right))
                    if pair in pairs:
                        continue
                    left_id = self._node_ids[clusters[pair[0]].medoid]
                    right_id = self._node_ids[clusters[pair[1]].medoid]
                    key = (
                        float(
                            np.sum(
                                np.square(centroids[pair[0]] - centroids[pair[1]]),
                                dtype=np.float64,
                            )
                        ),
                        _id_sort_key(left_id),
                        _id_sort_key(right_id),
                    )
                    if len(pairs) < self._candidate_pair_limit:
                        pairs[pair] = key
                        continue
                    worst_pair, worst_key = max(pairs.items(), key=lambda item: item[1])
                    if key < worst_key:
                        del pairs[worst_pair]
                        pairs[pair] = key
        return set(pairs)

    def _feature_distances(
        self,
        source: int,
        medoid: int,
        work: _SearchWork,
    ) -> tuple[float, float, float, float, float]:
        values: list[float] = []
        for name in ("weight", "activation", "rollout_jvp", "output_dual"):
            matrix = self._features[name]
            difference = matrix[source] - matrix[medoid]
            work.coordinate_operations += int(matrix.shape[1])
            values.append(
                float(np.mean(np.square(difference), dtype=np.float64))
                / (self._feature_scales[name] ** 2)
            )

        domain_matrix = self._features["domain"]
        source_present = bool(self._domain_present[source])
        medoid_present = bool(self._domain_present[medoid])
        if source_present and medoid_present and domain_matrix.shape[1] > 0:
            difference = domain_matrix[source] - domain_matrix[medoid]
            work.coordinate_operations += int(domain_matrix.shape[1])
            domain_distance = float(
                np.mean(np.square(difference), dtype=np.float64)
            ) / (self._feature_scales["domain"] ** 2)
        elif source_present != medoid_present:
            domain_distance = 1.0
        else:
            domain_distance = 0.0
        values.append(domain_distance)
        work.assignment_evaluations += 1
        return tuple(values)  # type: ignore[return-value]

    def _anchor_damage(
        self,
        source: int,
        medoid: int,
        distances: tuple[float, float, float, float, float],
    ) -> float:
        anchor = self._anchors[source]
        if anchor is None or source == medoid:
            return 0.0
        if anchor.mode == "exact":
            return math.inf
        if anchor.mode == "projected":
            if anchor.projection_coefficient is None:  # pragma: no cover - validated
                raise SurgeryStateError("Projected anchor has no coefficient.")
            binding = ProjectionBinding(
                source=self._node_ids[source],
                target=self._node_ids[medoid],
                coefficient=anchor.projection_coefficient,
            )
            if self._projection_damage_callback is None:
                # The generic graph cannot execute domain-specific operators, but
                # this scalar reflects the executable binding deterministically.
                projection_loss = abs(binding.coefficient) * (
                    1.0 + math.fsum(distances)
                )
            else:
                projection_loss = _finite_nonnegative(
                    self._projection_damage_callback(
                        binding,
                        self._nodes[source],
                        self._nodes[medoid],
                        distances,
                    ),
                    "projection damage callback result",
                )
            damage = anchor.penalty * self._nodes[source].importance * projection_loss
        else:
            damage = (
                anchor.penalty
                * self._nodes[source].importance
                * (1.0 + math.fsum(distances))
            )
        if anchor.max_damage is not None and damage > anchor.max_damage:
            return math.inf
        return damage

    def _cluster_cost(
        self,
        members: tuple[int, ...],
        medoid: int,
        work: _SearchWork,
    ) -> _Cost | None:
        cost = _ZERO_COST
        for source in members:
            distances = self._feature_distances(source, medoid, work)
            anchor_damage = self._anchor_damage(source, medoid, distances)
            if not math.isfinite(anchor_damage):
                return None
            importance = self._nodes[source].importance
            cost = cost.add(
                _Cost(
                    weight=importance * distances[0],
                    activation=importance * distances[1],
                    rollout_jvp=importance * distances[2],
                    output_dual=importance * distances[3],
                    domain=importance * distances[4],
                    anchor=anchor_damage,
                )
            )
        return cost

    def _weighted_damage(self, cost: _Cost) -> float:
        weights = self._score_weights
        return math.fsum(
            (
                weights.weight * cost.weight,
                weights.activation * cost.activation,
                weights.rollout_jvp * cost.rollout_jvp,
                weights.output_dual * cost.output_dual,
                weights.domain * cost.domain,
                weights.anchor * cost.anchor,
            )
        )

    def _cluster_savings(self, members: tuple[int, ...], medoid: int) -> int:
        return sum(
            self._nodes[index].parameter_bytes for index in members if index != medoid
        )

    def _medoid_candidates(
        self,
        members: tuple[int, ...],
        current_medoids: tuple[int, int],
    ) -> tuple[int, ...]:
        exact = [
            index
            for index in members
            if self._anchors[index] is not None and self._anchors[index].mode == "exact"
        ]
        if len(exact) > 1:
            return ()
        if exact:
            return (exact[0],)
        if len(members) <= self._medoid_candidate_limit:
            return members

        priority: list[int] = []
        for index in current_medoids:
            if index not in priority:
                priority.append(index)
        for index in members:
            if self._anchors[index] is not None and index not in priority:
                priority.append(index)
        member_array = np.asarray(members, dtype=np.int64)
        importance = np.asarray(
            [self._nodes[index].importance for index in members], dtype=np.float64
        )
        centroid = np.average(
            self._normalized_geometry[member_array],
            axis=0,
            weights=importance,
        )
        centroid_distances = np.sum(
            np.square(self._normalized_geometry[member_array] - centroid), axis=1
        )
        member_positions = {member: position for position, member in enumerate(members)}
        by_centroid = sorted(
            members,
            key=lambda index: (
                float(centroid_distances[member_positions[index]]),
                _id_sort_key(self._node_ids[index]),
            ),
        )
        for index in by_centroid:
            if index not in priority:
                priority.append(index)
            if len(priority) >= self._medoid_candidate_limit:
                break
        return tuple(priority[: self._medoid_candidate_limit])

    @staticmethod
    def _clean_delta(value: float) -> float:
        return 0.0 if abs(value) <= 1e-14 else value

    def _candidate_for_pair(
        self,
        snapshot: SurgerySnapshot,
        left: _Cluster,
        right: _Cluster,
        work: _SearchWork,
        baseline_costs: Mapping[int, _Cost],
    ) -> MergeCandidate | None:
        members = tuple(sorted((*left.members, *right.members)))
        medoid_candidates = self._medoid_candidates(
            members, (left.medoid, right.medoid)
        )
        if not medoid_candidates:
            work.anchor_pairs_skipped += 1
            return None
        work.medoid_candidates += len(medoid_candidates)
        baseline_cost = baseline_costs[left.medoid].add(baseline_costs[right.medoid])
        baseline_savings = self._cluster_savings(left.members, left.medoid) + (
            self._cluster_savings(right.members, right.medoid)
        )
        ranked: list[tuple[float, float, int, bytes, int, CandidateScore]] = []
        for medoid in medoid_candidates:
            proposed_cost = self._cluster_cost(members, medoid, work)
            if proposed_cost is None:
                continue
            delta = proposed_cost.subtract(baseline_cost)
            delta = _Cost(
                weight=self._clean_delta(delta.weight),
                activation=self._clean_delta(delta.activation),
                rollout_jvp=self._clean_delta(delta.rollout_jvp),
                output_dual=self._clean_delta(delta.output_dual),
                domain=self._clean_delta(delta.domain),
                anchor=self._clean_delta(delta.anchor),
            )
            weighted_damage = self._weighted_damage(delta)
            savings = self._cluster_savings(members, medoid) - baseline_savings
            savings_reward = (
                self._score_weights.savings * savings / self._savings_byte_scale
            )
            total_score = savings_reward - weighted_damage
            score = CandidateScore(
                weight_damage=delta.weight,
                activation_damage=delta.activation,
                rollout_jvp_damage=delta.rollout_jvp,
                output_dual_damage=delta.output_dual,
                domain_damage=delta.domain,
                anchor_damage=delta.anchor,
                weighted_damage=weighted_damage,
                parameter_bytes_saved=savings,
                savings_reward=savings_reward,
                total_score=total_score,
            )
            ranked.append(
                (
                    -score.total_score,
                    score.weighted_damage,
                    -score.parameter_bytes_saved,
                    _id_sort_key(self._node_ids[medoid]),
                    medoid,
                    score,
                )
            )
        if not ranked:
            work.anchor_pairs_skipped += 1
            return None
        ranked.sort(key=lambda item: item[:4])
        _, _, _, _, medoid, score = ranked[0]
        exact_anchor_index = next(
            (
                index
                for index in members
                if self._anchors[index] is not None
                and self._anchors[index].mode == "exact"
            ),
            None,
        )
        return MergeCandidate(
            parent_snapshot_sha256=snapshot.snapshot_sha256,
            evidence_sha256=self._evidence_sha256,
            left_prototype=self._node_ids[left.medoid],
            right_prototype=self._node_ids[right.medoid],
            left_members=tuple(self._node_ids[index] for index in left.members),
            right_members=tuple(self._node_ids[index] for index in right.members),
            member_ids=tuple(self._node_ids[index] for index in members),
            resulting_medoid=self._node_ids[medoid],
            medoid_candidates_considered=tuple(
                self._node_ids[index] for index in medoid_candidates
            ),
            score=score,
            exact_anchor=(
                None
                if exact_anchor_index is None
                else self._node_ids[exact_anchor_index]
            ),
        )

    def _compute_candidate_set(self) -> CandidateSet:
        snapshot = self._snapshot
        clusters = self._clusters(snapshot)
        work = _SearchWork()
        self._candidate_recomputations += 1
        pairs = self._candidate_pairs(clusters, work)
        baseline_costs: dict[int, _Cost] = {}
        needed_cluster_indices = sorted(
            {cluster_index for pair in pairs for cluster_index in pair}
        )
        for cluster_index in needed_cluster_indices:
            cluster = clusters[cluster_index]
            cost = self._cluster_cost(cluster.members, cluster.medoid, work)
            if cost is None:
                raise SurgeryStateError(
                    "The current snapshot violates an exact anchor constraint."
                )
            baseline_costs[cluster.medoid] = cost

        rejected = self._rejected_by_parent.get(snapshot.snapshot_sha256, OrderedDict())
        candidates: list[MergeCandidate] = []
        candidate_pairs_scored = 0
        for left_index, right_index in sorted(pairs):
            candidate_pairs_scored += 1
            candidate = self._candidate_for_pair(
                snapshot,
                clusters[left_index],
                clusters[right_index],
                work,
                baseline_costs,
            )
            if candidate is None:
                continue
            if candidate.candidate_sha256 in rejected:
                work.rejected_filtered += 1
                continue
            candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (
                -candidate.score.total_score,
                candidate.score.weighted_damage,
                _id_sort_key(candidate.left_prototype),
                _id_sort_key(candidate.right_prototype),
                _id_sort_key(candidate.resulting_medoid),
            )
        )
        selected = tuple(candidates[: self._max_candidates])
        pair_bound = self._projection_count * len(clusters) * self._neighbor_window
        assignment_bound = len(self._nodes) + (
            self._candidate_pair_limit * self._medoid_candidate_limit * len(self._nodes)
        )
        coordinate_bound = assignment_bound * self._feature_dimension_total
        bounded = (
            work.projection_comparisons <= pair_bound
            and len(pairs) <= self._candidate_pair_limit
            and len(selected) <= self._max_candidates
            and work.assignment_evaluations <= assignment_bound
            and work.coordinate_operations <= coordinate_bound
        )
        diagnostics = CandidateSearchDiagnostics(
            search_round=self._candidate_recomputations,
            active_prototypes=len(clusters),
            logical_nodes=len(self._nodes),
            geometry_dimensions=int(self._normalized_geometry.shape[1]),
            projection_count=self._projection_count,
            neighbor_window=self._neighbor_window,
            max_candidates=self._max_candidates,
            medoid_candidate_limit=self._medoid_candidate_limit,
            projection_neighbor_comparisons=work.projection_comparisons,
            projection_neighbor_comparison_bound=pair_bound,
            unique_pair_count=len(pairs),
            candidate_pairs_scored=candidate_pairs_scored,
            candidates_returned=len(selected),
            anchor_pairs_skipped=work.anchor_pairs_skipped,
            rejected_candidates_filtered=work.rejected_filtered,
            medoid_candidates_evaluated=work.medoid_candidates,
            assignment_distance_evaluations=work.assignment_evaluations,
            assignment_distance_evaluation_bound=assignment_bound,
            distance_coordinate_operations=work.coordinate_operations,
            distance_coordinate_operation_bound=coordinate_bound,
            full_pairwise_matrix_elements=len(clusters) * len(clusters),
            materialized_full_pairwise_matrix=False,
            candidate_buffer_bound=self._candidate_pair_limit,
            bounded_work=bounded,
        )
        self._projection_neighbor_comparisons += work.projection_comparisons
        self._candidate_pairs_scored += candidate_pairs_scored
        self._assignment_distance_evaluations += work.assignment_evaluations
        self._distance_coordinate_operations += work.coordinate_operations
        candidate_set = CandidateSet(selected, diagnostics)
        self._candidate_cache = candidate_set
        if snapshot.snapshot_sha256 in self._known_snapshots:
            self._evidence_states[snapshot.snapshot_sha256] = (
                self._capture_evidence_state()
            )
        return candidate_set

    def propose_candidates(self, limit: int | None = None) -> CandidateSet:
        """Return deterministic ranked candidates for the current snapshot."""

        if limit is not None:
            limit = _positive_int(limit, "limit")
        candidates = (
            self._candidate_cache
            if self._candidate_cache is not None
            else self._compute_candidate_set()
        )
        if limit is None or limit >= len(candidates):
            return candidates
        diagnostics_payload = candidates.diagnostics.as_dict()
        for derived in (
            "no_full_pairwise_matrix",
            "no_training",
            "pair_work_bound_formula",
            "medoid_work_bound_formula",
        ):
            diagnostics_payload.pop(derived)
        diagnostics_payload["candidates_returned"] = limit
        return CandidateSet(
            candidates.candidates[:limit],
            CandidateSearchDiagnostics(**diagnostics_payload),
        )

    def _active_cluster_for_selector(self, selector: Any) -> _Cluster:
        if isinstance(selector, int) and not isinstance(selector, bool):
            if selector < 0 or selector >= len(self._node_ids):
                raise SurgeryShapeError(
                    f"Active prototype index {selector} is out of range."
                )
            node_id = self._node_ids[selector]
        else:
            node_id = _normalize_node_id(selector, "active prototype")
        for cluster in self._clusters(self._snapshot):
            if self._node_ids[cluster.medoid] == node_id:
                return cluster
        if node_id not in self._index:
            raise SurgeryShapeError(f"Unknown node ID {node_id!r}.")
        raise SurgeryStateError(f"Node {node_id!r} is not an active prototype.")

    def _explicit_pair_selectors(
        self, pair_or_left: Any, right_prototype: Any | None
    ) -> tuple[Any, Any]:
        if right_prototype is not None:
            return pair_or_left, right_prototype
        if (
            isinstance(pair_or_left, Sequence)
            and not isinstance(pair_or_left, (str, bytes))
            and len(pair_or_left) == 2
        ):
            return pair_or_left[0], pair_or_left[1]
        raise SurgeryShapeError(
            "Provide two active prototypes or one two-item pair/tuple."
        )

    def _candidate_for_explicit_selectors(
        self, pair_or_left: Any, right_prototype: Any | None
    ) -> tuple[MergeCandidate, _SearchWork]:
        left_selector, right_selector = self._explicit_pair_selectors(
            pair_or_left, right_prototype
        )
        left = self._active_cluster_for_selector(left_selector)
        right = self._active_cluster_for_selector(right_selector)
        if left.medoid == right.medoid:
            raise SurgeryShapeError("An explicit pair needs two active prototypes.")
        work = _SearchWork()
        baseline_costs: dict[int, _Cost] = {}
        for cluster in (left, right):
            cost = self._cluster_cost(cluster.members, cluster.medoid, work)
            if cost is None:
                raise SurgeryStateError(
                    "The current snapshot violates an exact anchor constraint."
                )
            baseline_costs[cluster.medoid] = cost
        candidate = self._candidate_for_pair(
            self._snapshot, left, right, work, baseline_costs
        )
        if candidate is None:
            raise SurgeryTargetError(
                "The explicitly supplied active pair is anchor-infeasible."
            )
        return candidate, work

    def propose_candidate_for_pair(
        self, pair_or_left: Any, right_prototype: Any | None = None
    ) -> MergeCandidate:
        """Score one explicit active pair with bounded work outside discovery."""

        candidate, work = self._candidate_for_explicit_selectors(
            pair_or_left, right_prototype
        )
        rejected = self._rejected_by_parent.get(
            self._snapshot.snapshot_sha256, OrderedDict()
        )
        if candidate.candidate_sha256 in rejected:
            raise SurgeryStateError("Cannot repropose a rejected singleton candidate.")
        self._candidate_pairs_scored += 1
        self._assignment_distance_evaluations += work.assignment_evaluations
        self._distance_coordinate_operations += work.coordinate_operations
        return candidate

    def propose_pair(
        self, pair_or_left: Any, right_prototype: Any | None = None
    ) -> MergeCandidate:
        """Alias for :meth:`propose_candidate_for_pair`."""

        return self.propose_candidate_for_pair(pair_or_left, right_prototype)

    def score_pair(
        self, pair_or_left: Any, right_prototype: Any | None = None
    ) -> CandidateScore:
        """Return the bounded on-demand score for one explicit active pair."""

        return self.propose_candidate_for_pair(pair_or_left, right_prototype).score

    def propose_nonconflicting_batch(
        self,
        max_size: int,
        candidates: Sequence[MergeCandidate] | None = None,
    ) -> tuple[MergeCandidate, ...]:
        """Greedily select a ranked batch whose prototype groups do not overlap."""

        max_size = _positive_int(max_size, "max_size")
        source = self.propose_candidates() if candidates is None else candidates
        selected: list[MergeCandidate] = []
        occupied: set[NodeId] = set()
        for candidate in source:
            if candidate.parent_snapshot_sha256 != self._snapshot.snapshot_sha256:
                raise SurgeryStateError("Cannot batch a stale merge candidate.")
            member_set = set(candidate.member_ids)
            if member_set & occupied:
                continue
            selected.append(candidate)
            occupied.update(member_set)
            if len(selected) >= max_size:
                break
        return tuple(selected)

    def _validate_candidates(
        self, candidates: Sequence[MergeCandidate]
    ) -> tuple[MergeCandidate, ...]:
        if not candidates:
            raise SurgeryShapeError("At least one merge candidate is required.")
        normalized = tuple(candidates)
        if len(normalized) > self._candidate_pair_limit:
            raise SurgeryShapeError(
                "Candidate batch exceeds the configured candidate_pair_limit."
            )
        clusters = self._clusters(self._snapshot)
        current_clusters = {
            self._node_ids[cluster.medoid]: tuple(
                self._node_ids[index] for index in cluster.members
            )
            for cluster in clusters
        }
        occupied: set[NodeId] = set()
        rejected = self._rejected_by_parent.get(
            self._snapshot.snapshot_sha256, OrderedDict()
        )
        clusters_by_id = {
            self._node_ids[cluster.medoid]: cluster for cluster in clusters
        }
        for candidate in normalized:
            if not isinstance(candidate, MergeCandidate):
                raise SurgeryShapeError("Every candidate must be a MergeCandidate.")
            if candidate.parent_snapshot_sha256 != self._snapshot.snapshot_sha256:
                raise SurgeryStateError("Cannot apply a stale merge candidate.")
            if candidate.candidate_sha256 in rejected:
                raise SurgeryStateError("Cannot reapply a rejected merge candidate.")
            if current_clusters.get(candidate.left_prototype) != candidate.left_members:
                raise SurgeryStateError("Candidate left prototype group is stale.")
            if (
                current_clusters.get(candidate.right_prototype)
                != candidate.right_members
            ):
                raise SurgeryStateError("Candidate right prototype group is stale.")
            left = clusters_by_id[candidate.left_prototype]
            right = clusters_by_id[candidate.right_prototype]
            validation_work = _SearchWork()
            baseline_costs: dict[int, _Cost] = {}
            for cluster in (left, right):
                cost = self._cluster_cost(
                    cluster.members, cluster.medoid, validation_work
                )
                if cost is None:
                    raise SurgeryStateError(
                        "The current snapshot violates an exact anchor constraint."
                    )
                baseline_costs[cluster.medoid] = cost
            expected = self._candidate_for_pair(
                self._snapshot, left, right, validation_work, baseline_costs
            )
            if expected != candidate:
                raise SurgeryStateError(
                    "Candidate evidence or score does not match the current graph."
                )
            members = set(candidate.member_ids)
            if members & occupied:
                raise SurgeryShapeError("Batch candidates must be nonconflicting.")
            occupied.update(members)
        return normalized

    def _actions_for_candidate(
        self,
        candidate: MergeCandidate,
        medoid_before: tuple[int, ...],
    ) -> tuple[ActionRecord, ...]:
        actions: list[ActionRecord] = []
        composed_actions: list[ActionRecord] = []
        if len(candidate.left_members) > 1 or len(candidate.right_members) > 1:
            candidate_members = set(candidate.member_ids)
            prior_action_hashes = tuple(
                action.action_sha256
                for transaction in self._snapshot.transactions
                for action in transaction.actions
                if (
                    action.target in candidate_members
                    or bool(set(action.sources) & candidate_members)
                )
            )
            composed_actions.append(
                ActionRecord(
                    kind="compose",
                    sources=(candidate.left_prototype, candidate.right_prototype),
                    target=candidate.resulting_medoid,
                    reason="compose previously formed prototype groups",
                    parent_action_sha256s=prior_action_hashes,
                    details=(("candidate_sha256", candidate.candidate_sha256),),
                )
            )
        actions.extend(composed_actions)
        merge_action = ActionRecord(
            kind="merge",
            sources=(candidate.left_prototype, candidate.right_prototype),
            target=candidate.resulting_medoid,
            reason="bounded feature-geometry real-medoid merge",
            parent_action_sha256s=tuple(
                action.action_sha256 for action in composed_actions
            ),
            details=(
                ("candidate_sha256", candidate.candidate_sha256),
                ("parameter_bytes_saved", candidate.score.parameter_bytes_saved),
                ("score", candidate.score.total_score),
            ),
        )
        actions.append(merge_action)
        actions.append(
            ActionRecord(
                kind="preserve",
                sources=(candidate.resulting_medoid,),
                target=candidate.resulting_medoid,
                reason="retain a real source node as the merged prototype",
                parent_action_sha256s=(merge_action.action_sha256,),
            )
        )
        resulting_index = self._index[candidate.resulting_medoid]
        for member_id in candidate.member_ids:
            member = self._index[member_id]
            if member == resulting_index or medoid_before[member] == resulting_index:
                continue
            anchor = self._anchors[member]
            share_action = ActionRecord(
                kind="share",
                sources=(member_id,),
                target=candidate.resulting_medoid,
                reason="logical node shares the selected real prototype",
                anchor_mode=(
                    "soft" if anchor is not None and anchor.mode == "soft" else None
                ),
                parent_action_sha256s=(merge_action.action_sha256,),
                details=(
                    (("anchor_penalty", anchor.penalty),)
                    if anchor is not None and anchor.mode == "soft"
                    else ()
                ),
            )
            actions.append(share_action)
            if anchor is not None and anchor.mode == "projected":
                actions.append(
                    ActionRecord(
                        kind="project",
                        sources=(member_id,),
                        target=candidate.resulting_medoid,
                        reason="apply retained projected-anchor coefficient binding",
                        anchor_mode="projected",
                        parent_action_sha256s=(share_action.action_sha256,),
                        details=(
                            ("anchor_penalty", anchor.penalty),
                            (
                                "projection_coefficient",
                                anchor.projection_coefficient,
                            ),
                        ),
                    )
                )
        return tuple(actions)

    def preview(
        self,
        candidate_or_batch: MergeCandidate | Sequence[MergeCandidate],
    ) -> SurgeryProposal:
        """Build an immutable tentative snapshot without mutating graph state."""

        if isinstance(candidate_or_batch, MergeCandidate):
            candidates = (candidate_or_batch,)
        else:
            candidates = tuple(candidate_or_batch)
        candidates = self._validate_candidates(candidates)
        parent = self._snapshot
        medoid_before = self._medoid_indices(parent)
        medoid_after = list(medoid_before)
        actions: list[ActionRecord] = []
        for candidate in candidates:
            resulting_index = self._index[candidate.resulting_medoid]
            for member_id in candidate.member_ids:
                medoid_after[self._index[member_id]] = resulting_index
            medoid_after[resulting_index] = resulting_index
            actions.extend(self._actions_for_candidate(candidate, medoid_before))

        medoid_after_tuple = tuple(medoid_after)
        material = self._state_material(medoid_after_tuple)
        assignment_changes = tuple(
            AssignmentChange(
                source=self._node_ids[index],
                before=self._node_ids[before],
                after=self._node_ids[after],
            )
            for index, (before, after) in enumerate(
                zip(medoid_before, medoid_after_tuple)
            )
            if before != after
        )
        transaction_index = parent.transaction_count + 1
        transaction_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "adaptive-prototype-surgery-transaction",
            "transaction_index": transaction_index,
            "parent_snapshot_sha256": parent.snapshot_sha256,
            "parent_state_sha256": parent.state_sha256,
            "candidate_sha256s": [
                candidate.candidate_sha256 for candidate in candidates
            ],
            "assignment_changes": [change.as_dict() for change in assignment_changes],
            "projection_bindings": [binding.as_dict() for binding in material[8]],
            "projection_binding_sha256s": [
                binding.binding_sha256 for binding in material[8]
            ],
            "action_sha256s": [action.action_sha256 for action in actions],
            "prototype_count_before": parent.prototype_count,
            "prototype_count_after": len(material[1]),
            "parameter_bytes_saved_increment": (
                material[4] - parent.parameter_bytes_saved
            ),
            "parameter_bytes_saved_total": material[4],
            "assignments_sha256": material[5],
            "registry_sha256": material[6],
            "state_sha256": material[7],
        }
        transaction_sha256 = _sha256(transaction_payload)
        provenance_sha256 = _sha256(
            {
                "parent_provenance_sha256": parent.provenance_sha256,
                "transaction_sha256": transaction_sha256,
            }
        )
        transaction = SurgeryTransaction(
            schema_version=SCHEMA_VERSION,
            transaction_index=transaction_index,
            parent_snapshot_sha256=parent.snapshot_sha256,
            parent_state_sha256=parent.state_sha256,
            candidate_sha256s=tuple(
                candidate.candidate_sha256 for candidate in candidates
            ),
            candidates=candidates,
            assignment_changes=assignment_changes,
            projection_bindings=material[8],
            actions=tuple(actions),
            prototype_count_before=parent.prototype_count,
            prototype_count_after=len(material[1]),
            parameter_bytes_saved_increment=(
                material[4] - parent.parameter_bytes_saved
            ),
            parameter_bytes_saved_total=material[4],
            assignments_sha256=material[5],
            registry_sha256=material[6],
            state_sha256=material[7],
            transaction_sha256=transaction_sha256,
            provenance_sha256=provenance_sha256,
        )
        transactions = (*parent.transactions, transaction)[-self._history_limit :]
        snapshot_sha256 = _sha256(
            {
                "state_sha256": material[7],
                "provenance_sha256": provenance_sha256,
                "transaction_count": transaction_index,
            }
        )
        snapshot = SurgerySnapshot(
            schema_version=SCHEMA_VERSION,
            graph_sha256=self._graph_sha256,
            node_ids=self._node_ids,
            assignment_pairs=material[0],
            prototype_ids=material[1],
            assignment_indices=material[2],
            prototype_source_indices=material[3],
            parameter_bytes_saved=material[4],
            projection_bindings=material[8],
            transactions=transactions,
            transaction_count_total=transaction_index,
            assignments_sha256=material[5],
            registry_sha256=material[6],
            state_sha256=material[7],
            provenance_sha256=provenance_sha256,
            snapshot_sha256=snapshot_sha256,
        )
        proposal = SurgeryProposal(
            parent=parent, snapshot=snapshot, transaction=transaction
        )
        self._pending_proposals[proposal.proposal_sha256] = proposal
        self._pending_proposals.move_to_end(proposal.proposal_sha256)
        while len(self._pending_proposals) > self._candidate_pair_limit:
            self._pending_proposals.popitem(last=False)
        return proposal

    @staticmethod
    def _normalize_decision(value: Any) -> EvaluationDecision:
        if isinstance(value, EvaluationDecision):
            return value
        if isinstance(value, bool):
            return EvaluationDecision(value)
        if isinstance(value, Mapping):
            accepted = value.get("accepted", value.get("accept", _UNSET))
            if accepted is _UNSET:
                raise SurgeryShapeError(
                    "Evaluator mappings must contain 'accepted' or 'accept'."
                )
            reason = value.get("reason", "")
            metrics = value.get("metrics")
            return EvaluationDecision(accepted, reason=reason, metrics=metrics)
        raise SurgeryShapeError(
            "Evaluator must return bool, EvaluationDecision, or an acceptance mapping."
        )

    @staticmethod
    def _invoke_evaluator(
        evaluator: Evaluator,
        proposal: SurgeryProposal,
    ) -> EvaluationDecision:
        callable_value = getattr(evaluator, "evaluate", evaluator)
        if not callable(callable_value):
            raise SurgeryShapeError("evaluator must be callable or expose evaluate().")
        try:
            signature = inspect.signature(callable_value)
        except (TypeError, ValueError):
            result = callable_value(
                proposal.parent, proposal.snapshot, proposal.transaction
            )
            return AdaptiveSurgeryGraph._normalize_decision(result)
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
            result = callable_value(
                proposal.parent, proposal.snapshot, proposal.transaction
            )
        elif len(positional) == 2:
            result = callable_value(proposal.parent, proposal.snapshot)
        elif len(positional) == 1:
            result = callable_value(proposal.snapshot)
        else:
            result = callable_value()
        return AdaptiveSurgeryGraph._normalize_decision(result)

    def _attempt_record(
        self,
        proposal: SurgeryProposal,
        decision: EvaluationDecision,
        actions: tuple[ActionRecord, ...],
    ) -> AttemptRecord:
        attempt_index = self._attempt_count + 1
        payload = {
            "attempt_index": attempt_index,
            "parent_snapshot_sha256": proposal.parent.snapshot_sha256,
            "proposed_snapshot_sha256": proposal.snapshot.snapshot_sha256,
            "transaction_sha256": proposal.transaction.transaction_sha256,
            "accepted": decision.accepted,
            "decision": decision.as_dict(),
            "action_sha256s": [action.action_sha256 for action in actions],
        }
        return AttemptRecord(
            attempt_index=attempt_index,
            parent_snapshot_sha256=proposal.parent.snapshot_sha256,
            proposed_snapshot_sha256=proposal.snapshot.snapshot_sha256,
            transaction_sha256=proposal.transaction.transaction_sha256,
            accepted=decision.accepted,
            decision=decision,
            actions=actions,
            attempt_sha256=_sha256(payload),
        )

    def _validate_proposal_parent(self, proposal: SurgeryProposal) -> None:
        if not isinstance(proposal, SurgeryProposal):
            raise SurgeryShapeError("proposal must be a SurgeryProposal.")
        if proposal.parent.snapshot_sha256 != self._snapshot.snapshot_sha256:
            raise SurgeryStateError("Cannot resolve a proposal from a stale parent.")
        if proposal.parent.graph_sha256 != self._graph_sha256:
            raise SurgeryStateError("Cannot resolve a proposal from another graph.")
        if self._pending_proposals.get(proposal.proposal_sha256) != proposal:
            raise SurgeryStateError("Proposal was not issued by the current graph.")

    def _checkpoint(self) -> _GraphCheckpoint:
        return _GraphCheckpoint(
            nodes=self._nodes,
            evidence_sha256=self._evidence_sha256,
            features=self._features,
            feature_scales=self._feature_scales,
            domain_present=self._domain_present,
            feature_dimension_total=self._feature_dimension_total,
            normalized_geometry=self._normalized_geometry,
            snapshot=self._snapshot,
            known_snapshots=tuple(self._known_snapshots.items()),
            evidence_states=tuple(self._evidence_states.items()),
            candidate_cache=self._candidate_cache,
            rejected_by_parent=tuple(
                (parent, tuple(rejections))
                for parent, rejections in self._rejected_by_parent.items()
            ),
            pending_proposals=tuple(self._pending_proposals.items()),
            attempts=tuple(self._attempts),
            attempt_count=self._attempt_count,
            counters=(
                self._candidate_recomputations,
                self._accepted_transactions,
                self._rejected_transactions,
                self._projection_neighbor_comparisons,
                self._candidate_pairs_scored,
                self._assignment_distance_evaluations,
                self._distance_coordinate_operations,
            ),
        )

    def _restore_checkpoint(self, checkpoint: _GraphCheckpoint) -> None:
        self._nodes = checkpoint.nodes
        self._evidence_sha256 = checkpoint.evidence_sha256
        self._features = checkpoint.features
        self._feature_scales = checkpoint.feature_scales
        self._domain_present = checkpoint.domain_present
        self._feature_dimension_total = checkpoint.feature_dimension_total
        self._normalized_geometry = checkpoint.normalized_geometry
        self._snapshot = checkpoint.snapshot
        self._known_snapshots = OrderedDict(checkpoint.known_snapshots)
        self._evidence_states = OrderedDict(checkpoint.evidence_states)
        self._candidate_cache = checkpoint.candidate_cache
        self._rejected_by_parent = OrderedDict(
            (parent, OrderedDict((digest, None) for digest in rejections))
            for parent, rejections in checkpoint.rejected_by_parent
        )
        self._pending_proposals = OrderedDict(checkpoint.pending_proposals)
        self._attempts = list(checkpoint.attempts)
        self._attempt_count = checkpoint.attempt_count
        (
            self._candidate_recomputations,
            self._accepted_transactions,
            self._rejected_transactions,
            self._projection_neighbor_comparisons,
            self._candidate_pairs_scored,
            self._assignment_distance_evaluations,
            self._distance_coordinate_operations,
        ) = checkpoint.counters

    def _capture_evidence_state(self) -> _EvidenceState:
        return _EvidenceState(
            nodes=self._nodes,
            evidence_sha256=self._evidence_sha256,
            features=self._features,
            feature_scales=self._feature_scales,
            domain_present=self._domain_present,
            feature_dimension_total=self._feature_dimension_total,
            normalized_geometry=self._normalized_geometry,
            candidate_cache=self._candidate_cache,
        )

    def _install_evidence_state(self, state: _EvidenceState) -> None:
        self._nodes = state.nodes
        self._evidence_sha256 = state.evidence_sha256
        self._features = state.features
        self._feature_scales = state.feature_scales
        self._domain_present = state.domain_present
        self._feature_dimension_total = state.feature_dimension_total
        self._normalized_geometry = state.normalized_geometry
        self._candidate_cache = state.candidate_cache

    def _remember_snapshot(self, snapshot: SurgerySnapshot) -> None:
        self._known_snapshots[snapshot.snapshot_sha256] = snapshot
        self._known_snapshots.move_to_end(snapshot.snapshot_sha256)
        self._evidence_states[snapshot.snapshot_sha256] = self._capture_evidence_state()
        self._evidence_states.move_to_end(snapshot.snapshot_sha256)
        while len(self._known_snapshots) > self._history_limit:
            evicted_snapshot_sha256, _ = self._known_snapshots.popitem(last=False)
            self._evidence_states.pop(evicted_snapshot_sha256, None)
        while len(self._evidence_states) > self._history_limit:
            self._evidence_states.popitem(last=False)

    def _append_attempt(self, attempt: AttemptRecord) -> None:
        self._attempt_count += 1
        self._attempts.append(attempt)
        if len(self._attempts) > self._history_limit:
            del self._attempts[: len(self._attempts) - self._history_limit]

    def _remember_singleton_rejection(
        self, parent_snapshot_sha256: str, candidate_sha256: str
    ) -> None:
        rejected = self._rejected_by_parent.setdefault(
            parent_snapshot_sha256, OrderedDict()
        )
        rejected[candidate_sha256] = None
        rejected.move_to_end(candidate_sha256)
        while len(rejected) > self._candidate_pair_limit:
            rejected.popitem(last=False)
        self._rejected_by_parent.move_to_end(parent_snapshot_sha256)
        while len(self._rejected_by_parent) > self._history_limit:
            self._rejected_by_parent.popitem(last=False)

    @staticmethod
    def _normalize_refreshed_nodes(
        value: Any, current_nodes: tuple[SurgeryNode, ...]
    ) -> tuple[SurgeryNode, ...]:
        if isinstance(value, Mapping):
            node_payload = value.get("nodes", _UNSET)
            if node_payload is _UNSET:
                try:
                    nodes = tuple(value[node.node_id] for node in current_nodes)
                except (KeyError, TypeError) as error:
                    raise SurgeryShapeError(
                        "Evidence refresh mappings must contain 'nodes' or one "
                        "SurgeryNode per existing node ID."
                    ) from error
            else:
                nodes = tuple(node_payload)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            nodes = tuple(value)
        else:
            raise SurgeryShapeError(
                "Evidence refresh must return an immutable SurgeryNode sequence "
                "or node mapping."
            )
        if len(nodes) != len(current_nodes) or not all(
            isinstance(node, SurgeryNode) for node in nodes
        ):
            raise SurgeryShapeError(
                "Evidence refresh must return exactly one SurgeryNode per graph node."
            )
        refreshed = tuple(nodes)
        if tuple(node.node_id for node in refreshed) != tuple(
            node.node_id for node in current_nodes
        ):
            raise SurgeryShapeError(
                "Evidence refresh cannot reorder or replace logical node IDs."
            )
        for old, new in zip(current_nodes, refreshed):
            if (
                old.importance != new.importance
                or old.parameter_bytes != new.parameter_bytes
                or old.anchor != new.anchor
            ):
                raise SurgeryShapeError(
                    "Evidence refresh may replace feature evidence only; importance, "
                    "parameter bytes, and node anchors are immutable."
                )
        return refreshed

    @staticmethod
    def _invoke_evidence_refresh(
        callback: EvidenceRefresh,
        proposal: SurgeryProposal,
        current_nodes: tuple[SurgeryNode, ...],
    ) -> tuple[SurgeryNode, ...]:
        callable_value = getattr(callback, "refresh", callback)
        if not callable(callable_value):
            raise SurgeryShapeError(
                "evidence_refresh must be callable or expose refresh()."
            )
        try:
            signature = inspect.signature(callable_value)
        except (TypeError, ValueError):
            result = callable_value(
                proposal.parent,
                proposal.snapshot,
                proposal.transaction,
                current_nodes,
            )
            return AdaptiveSurgeryGraph._normalize_refreshed_nodes(
                result, current_nodes
            )
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
            result = callable_value(
                proposal.parent,
                proposal.snapshot,
                proposal.transaction,
                current_nodes,
            )
        else:
            fallbacks: dict[int, tuple[Any, ...]] = {
                0: (),
                1: (proposal.snapshot,),
                2: (proposal.parent, proposal.snapshot),
                3: (proposal.parent, proposal.snapshot, proposal.transaction),
            }
            arguments = list(fallbacks[len(positional)])
            for index, parameter in enumerate(positional):
                name = parameter.name.lower()
                if "node" in name or "evidence" in name:
                    arguments[index] = current_nodes
                elif "transaction" in name or name in {"tx", "txn"}:
                    arguments[index] = proposal.transaction
                elif "proposal" in name:
                    arguments[index] = proposal
                elif "parent" in name or "before" in name or name.startswith("old"):
                    arguments[index] = proposal.parent
                elif (
                    "snapshot" in name
                    or "accepted" in name
                    or "proposed" in name
                    or name.startswith("new")
                ):
                    arguments[index] = proposal.snapshot
            result = callable_value(*arguments)
        return AdaptiveSurgeryGraph._normalize_refreshed_nodes(result, current_nodes)

    def _install_refreshed_nodes(self, nodes: tuple[SurgeryNode, ...]) -> None:
        self._nodes = nodes
        self._features, self._feature_scales, self._domain_present = (
            self._build_feature_matrices()
        )
        self._feature_dimension_total = sum(
            matrix.shape[1] for matrix in self._features.values()
        )
        self._normalized_geometry = self._build_normalized_geometry()
        self._evidence_sha256 = _sha256([node.as_dict() for node in self._nodes])

    def commit(
        self,
        proposal: SurgeryProposal,
        decision: EvaluationDecision | None = None,
        *,
        evidence_refresh: EvidenceRefresh | None = None,
    ) -> SurgeryOutcome:
        """Accept a preview, advance state, then recompute all candidates."""

        self._validate_proposal_parent(proposal)
        accepted_decision = decision or EvaluationDecision.accept()
        if not accepted_decision.accepted:
            raise SurgeryShapeError("commit requires an accepting decision.")
        attempt = self._attempt_record(
            proposal, accepted_decision, proposal.transaction.actions
        )
        checkpoint = self._checkpoint()
        try:
            self._snapshot = proposal.snapshot
            refresh_callback = (
                self._evidence_refresh if evidence_refresh is None else evidence_refresh
            )
            if refresh_callback is not None:
                refreshed_nodes = self._invoke_evidence_refresh(
                    refresh_callback, proposal, self._nodes
                )
                self._install_refreshed_nodes(refreshed_nodes)
            self._append_attempt(attempt)
            self._accepted_transactions += 1
            self._pending_proposals.clear()
            self._candidate_cache = None
            self._compute_candidate_set()
            self._remember_snapshot(self._snapshot)
        except Exception:
            self._restore_checkpoint(checkpoint)
            raise
        return SurgeryOutcome(
            accepted=True,
            snapshot=self._snapshot,
            proposed_snapshot=proposal.snapshot,
            transaction=proposal.transaction,
            decision=accepted_decision,
            attempt=attempt,
        )

    def reject(
        self,
        proposal: SurgeryProposal,
        decision: EvaluationDecision | None = None,
    ) -> SurgeryOutcome:
        """Reject a preview, retain its parent, and expose the next alternative."""

        self._validate_proposal_parent(proposal)
        rejected_decision = decision or EvaluationDecision.reject()
        if rejected_decision.accepted:
            raise SurgeryShapeError("reject requires a rejecting decision.")
        merge_action_hashes = {
            dict(action.details)["candidate_sha256"]: action.action_sha256
            for action in proposal.transaction.actions
            if action.kind == "merge" and "candidate_sha256" in dict(action.details)
        }
        skip_actions = tuple(
            ActionRecord(
                kind="skip",
                sources=(candidate.left_prototype, candidate.right_prototype),
                target=candidate.resulting_medoid,
                reason=rejected_decision.reason
                or "external evaluator rejected proposal",
                parent_action_sha256s=(
                    merge_action_hashes[candidate.candidate_sha256],
                ),
                details=(("candidate_sha256", candidate.candidate_sha256),),
            )
            for candidate in proposal.transaction.candidates
        )
        attempt = self._attempt_record(proposal, rejected_decision, skip_actions)
        checkpoint = self._checkpoint()
        try:
            if len(proposal.transaction.candidates) == 1:
                self._remember_singleton_rejection(
                    self._snapshot.snapshot_sha256,
                    proposal.transaction.candidate_sha256s[0],
                )
            self._append_attempt(attempt)
            self._rejected_transactions += 1
            self._pending_proposals.clear()
            self._candidate_cache = None
            self._compute_candidate_set()
        except Exception:
            self._restore_checkpoint(checkpoint)
            raise
        return SurgeryOutcome(
            accepted=False,
            snapshot=self._snapshot,
            proposed_snapshot=proposal.snapshot,
            transaction=proposal.transaction,
            decision=rejected_decision,
            attempt=attempt,
        )

    def apply(
        self,
        candidate_or_batch: MergeCandidate | Sequence[MergeCandidate],
        evaluator: Evaluator | None = None,
        *,
        evidence_refresh: EvidenceRefresh | None = None,
    ) -> SurgeryOutcome:
        """Preview and resolve one candidate or nonconflicting batch."""

        proposal = self.preview(candidate_or_batch)
        decision = (
            EvaluationDecision.accept()
            if evaluator is None
            else self._invoke_evaluator(evaluator, proposal)
        )
        return (
            self.commit(proposal, decision, evidence_refresh=evidence_refresh)
            if decision.accepted
            else self.reject(proposal, decision)
        )

    def apply_one(
        self,
        candidate: MergeCandidate | None = None,
        evaluator: Evaluator | None = None,
        *,
        evidence_refresh: EvidenceRefresh | None = None,
    ) -> SurgeryOutcome:
        """Apply the supplied candidate, or the highest-ranked current candidate."""

        if candidate is None:
            candidates = self.propose_candidates(limit=1)
            if not candidates:
                raise SurgerySearchExhausted("No valid merge candidate remains.")
            candidate = candidates[0]
        return self.apply(
            candidate,
            evaluator=evaluator,
            evidence_refresh=evidence_refresh,
        )

    def apply_batch(
        self,
        candidates: Sequence[MergeCandidate],
        evaluator: Evaluator | None = None,
        *,
        evidence_refresh: EvidenceRefresh | None = None,
    ) -> SurgeryOutcome:
        """Apply one atomically evaluated nonconflicting candidate batch."""

        return self.apply(
            tuple(candidates),
            evaluator=evaluator,
            evidence_refresh=evidence_refresh,
        )

    def apply_batch_with_retry(
        self,
        candidates: Sequence[MergeCandidate],
        evaluator: Evaluator | None = None,
        *,
        evidence_refresh: EvidenceRefresh | None = None,
    ) -> tuple[SurgeryOutcome, ...]:
        """Bisect a rejected batch so its constituents receive isolated retries."""

        batch = tuple(candidates)
        outcome = self.apply_batch(
            batch,
            evaluator=evaluator,
            evidence_refresh=evidence_refresh,
        )
        outcomes = [outcome]
        if outcome.accepted or len(batch) == 1:
            return tuple(outcomes)
        midpoint = len(batch) // 2
        for part in (batch[:midpoint], batch[midpoint:]):
            rescored = tuple(
                self.propose_candidate_for_pair(candidate.sources) for candidate in part
            )
            outcomes.extend(
                self.apply_batch_with_retry(
                    rescored,
                    evaluator=evaluator,
                    evidence_refresh=evidence_refresh,
                )
            )
        return tuple(outcomes)

    def rollback(
        self,
        proposal_or_snapshot: SurgeryProposal | SurgerySnapshot,
        *,
        reason: str = "external evaluator requested rollback",
        metrics: Mapping[str, MetadataScalar] | None = None,
    ) -> SurgeryOutcome | SurgerySnapshot:
        """Reject a preview or restore a previously accepted immutable snapshot."""

        if isinstance(proposal_or_snapshot, SurgeryProposal):
            return self.reject(
                proposal_or_snapshot,
                EvaluationDecision.reject(reason=reason, metrics=metrics),
            )
        return self.restore(proposal_or_snapshot)

    def restore(self, snapshot: SurgerySnapshot) -> SurgerySnapshot:
        """Restore an initial or accepted snapshot and recompute its candidates."""

        if not isinstance(snapshot, SurgerySnapshot):
            raise SurgeryShapeError("snapshot must be a SurgerySnapshot.")
        snapshot_sha256 = snapshot.snapshot_sha256
        known = self._known_snapshots.get(snapshot_sha256)
        evidence_state = self._evidence_states.get(snapshot_sha256)
        if (
            known != snapshot
            or evidence_state is None
            or snapshot.graph_sha256 != self._graph_sha256
        ):
            raise SurgeryStateError("Cannot restore an unknown or foreign snapshot.")
        if snapshot_sha256 == self._snapshot.snapshot_sha256:
            return self._snapshot
        checkpoint = self._checkpoint()
        try:
            self._install_evidence_state(evidence_state)
            self._snapshot = snapshot
            self._known_snapshots.move_to_end(snapshot_sha256)
            self._evidence_states.move_to_end(snapshot_sha256)
            self._pending_proposals.clear()
        except Exception:
            self._restore_checkpoint(checkpoint)
            raise
        return self._snapshot

    def fold_to(
        self,
        target_prototypes: int,
        *,
        evaluator: Evaluator | None = None,
        batch_size: int = 1,
        evidence_refresh: EvidenceRefresh | None = None,
    ) -> SurgerySnapshot:
        """Sequentially fold until exactly ``target_prototypes`` remain."""

        target = _positive_int(target_prototypes, "target_prototypes")
        batch_size = _positive_int(batch_size, "batch_size")
        current = self._snapshot.prototype_count
        if target > current:
            raise SurgeryTargetError(
                "target_prototypes cannot exceed the current prototype count."
            )
        exact_anchor_count = sum(
            anchor is not None and anchor.mode == "exact" for anchor in self._anchors
        )
        if target < exact_anchor_count:
            raise SurgeryTargetError(
                "target_prototypes is smaller than the number of exact anchors."
            )
        if target == current:
            return self._snapshot

        while self._snapshot.prototype_count > target:
            candidates = self.propose_candidates()
            if not candidates:
                raise SurgerySearchExhausted(
                    f"No valid candidate remains at {self._snapshot.prototype_count} "
                    f"prototypes; target is {target}."
                )
            remaining = self._snapshot.prototype_count - target
            batch = self.propose_nonconflicting_batch(
                min(batch_size, remaining), candidates
            )
            if not batch:
                raise SurgerySearchExhausted(
                    "No nonconflicting candidate batch remains."
                )
            if len(batch) == 1:
                self.apply(
                    batch,
                    evaluator=evaluator,
                    evidence_refresh=evidence_refresh,
                )
            else:
                self.apply_batch_with_retry(
                    batch,
                    evaluator=evaluator,
                    evidence_refresh=evidence_refresh,
                )
        return self._snapshot

    def as_dict(self) -> dict[str, Any]:
        """Return current state, attempts, and zero-training work diagnostics."""

        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "adaptive-prototype-surgery-graph",
            "graph_sha256": self._graph_sha256,
            "evidence_sha256": self._evidence_sha256,
            "snapshot": self._snapshot.as_dict(),
            "attempts": [attempt.as_dict() for attempt in self._attempts],
            "history": {
                "limit": self._history_limit,
                "retained_snapshots": len(self._known_snapshots),
                "retained_attempts": len(self._attempts),
                "retained_transactions": len(self._snapshot.transactions),
            },
            "last_search": (
                None
                if self.last_search_diagnostics is None
                else self.last_search_diagnostics.as_dict()
            ),
            "work": self.work_diagnostics.as_dict(),
        }


def fold_adaptive_prototypes(
    nodes: Sequence[SurgeryNode],
    target_prototypes: int,
    *,
    evaluator: Evaluator | None = None,
    batch_size: int = 1,
    **graph_options: Any,
) -> SurgerySnapshot:
    """Build a graph and return its exact-target folded snapshot."""

    graph = AdaptiveSurgeryGraph(nodes, **graph_options)
    return graph.fold_to(
        target_prototypes,
        evaluator=evaluator,
        batch_size=batch_size,
    )


__all__ = [
    "ACTION_KINDS",
    "ANCHOR_MODES",
    "SCHEMA_VERSION",
    "ActionRecord",
    "AdaptiveSurgeryError",
    "AdaptiveSurgeryGraph",
    "AdaptiveSurgeryNode",
    "AnchorSpec",
    "AssignmentChange",
    "AttemptRecord",
    "CandidateScore",
    "CandidateSearchDiagnostics",
    "CandidateSet",
    "EvaluationDecision",
    "EvidenceRefresh",
    "GraphWorkDiagnostics",
    "MergeCandidate",
    "PrototypeSurgeryNode",
    "ProjectionBinding",
    "ProjectionDamageScorer",
    "ScoreWeights",
    "SurgeryAction",
    "SurgeryFiniteError",
    "SurgeryNode",
    "SurgeryOutcome",
    "SurgeryProposal",
    "SurgerySearchExhausted",
    "SurgeryShapeError",
    "SurgerySnapshot",
    "SurgeryStateError",
    "SurgeryTargetError",
    "SurgeryTransaction",
    "fold_adaptive_prototypes",
]
