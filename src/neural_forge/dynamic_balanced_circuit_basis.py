"""Family-neutral balanced residual coordinates for structural folding.

This module joins three evidence lanes that were previously isolated:

* complete, prompt-independent residual-facing weights, balanced by semantic
  role rather than raw tensor size;
* cached residual snapshots from every supplied phase/layer location, used as
  an empirical controllability Gram on a fit-only split; and
* downstream residual consumers, attention/MLP writers, and output/modality
  interfaces, used as an explicit observability approximation.

The fit uses regularized balanced POD, not activation PCA, raw weight SVD, or
independent local regressions.  Classical biorthogonal encoder/decoder modes
are retained for audit.  Their selected state-space span is canonicalized into
one dense orthonormal ``B[source, target]`` so existing structural folding can
transport consumers as ``W @ B`` and producers as ``B.T @ W``.

Only already-materialized tensors are accepted.  This module cannot run a
model, capture a trace, invoke autograd/backward, construct an optimizer, or
write a checkpoint.  Held-out snapshots are disjoint provenance and are used
only after the basis has been fixed.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch

from .balanced_krylov import (
    BalancedKrylovConditionError,
    BalancedKrylovError,
    BalancedKrylovRankError,
    build_balanced_krylov_basis,
)
from .dynamic_circuit_ir import (
    CircuitKind,
    ModelCircuitGraph,
    PortKind,
    canonical_sha256,
)
from .dynamic_weight_complete_basis import (
    FeatureOrientation,
    ResidualFacingOperator,
)


BALANCED_CIRCUIT_BASIS_SCHEMA = (
    "neural-forge-dynamic-balanced-circuit-basis-v1"
)
RESIDUAL_SNAPSHOT_SCHEMA = (
    "neural-forge-dynamic-balanced-residual-snapshot-v1"
)
OBSERVABILITY_EVIDENCE_SCHEMA = (
    "neural-forge-dynamic-downstream-observability-v1"
)
_NUMERICAL_LOCK = threading.RLock()


class BalancedCircuitBasisError(ValueError):
    """Base error for invalid evidence or unsafe balanced coordinates."""


class BalancedCircuitBasisEvidenceError(BalancedCircuitBasisError):
    """Cached fit/held-out or IR evidence is incomplete or overlapping."""


class BalancedCircuitBasisRankError(BalancedCircuitBasisError):
    """The controllable/observable intersection cannot support the target."""


class EvidenceSplit(str, Enum):
    """Whether cached residual rows may select the basis."""

    FIT = "fit"
    HELDOUT = "heldout"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BalancedCircuitBasisEvidenceError(
            f"{name} must be a nonempty string."
        )
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BalancedCircuitBasisError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BalancedCircuitBasisError(
            f"{name} must be finite and strictly positive."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BalancedCircuitBasisError(
            f"{name} must be finite and strictly positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise BalancedCircuitBasisError(
            f"{name} must be finite and strictly positive."
        )
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BalancedCircuitBasisError(
            f"{name} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BalancedCircuitBasisError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise BalancedCircuitBasisError(
            f"{name} must be finite and non-negative."
        )
    return result


def _finite_tensor(value: Any, name: str, *, rank: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != rank
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.numel() == 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise BalancedCircuitBasisEvidenceError(
            f"{name} must be a nonempty finite dense real rank-{rank} tensor."
        )
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )


def _finite_runtime_rows(value: Any, name: str) -> torch.Tensor:
    """Validate runtime row tensors while permitting empty leading axes.

    Basis construction evidence remains subject to :func:`_finite_tensor`,
    including its nonempty requirement.  The runtime coordinate maps also
    receive sequence-shaped producer cases and legitimate empty cases such as
    ``(batch, 0, width)``. Rejecting either would make a mathematically total
    last-axis linear map partial for no numerical reason.
    """

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim < 2
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise BalancedCircuitBasisEvidenceError(
            f"{name} must be a finite dense real tensor with rank at least 2."
        )
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = _finite_tensor(value, "hashed tensor", rank=value.ndim)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_strings(
    values: Sequence[str],
    name: str,
    *,
    required: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BalancedCircuitBasisEvidenceError(
            f"{name} must be a sequence of stable strings."
        )
    result = tuple(sorted(_text(value, name) for value in values))
    if required and not result:
        raise BalancedCircuitBasisEvidenceError(
            f"{name} must not be empty."
        )
    if len(set(result)) != len(result):
        raise BalancedCircuitBasisEvidenceError(
            f"{name} must not contain duplicates."
        )
    return result


@dataclass(frozen=True, slots=True)
class ResidualSnapshotBlock:
    """One sealed cache block at a stable phase/layer residual endpoint."""

    block_id: str
    split: EvidenceSplit
    phase: str
    layer: str
    endpoint: str
    rows: torch.Tensor
    record_ids: tuple[str, ...]
    gram_weight: float = 1.0
    block_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        block_id = _text(self.block_id, "block_id")
        if not isinstance(self.split, EvidenceSplit):
            raise BalancedCircuitBasisEvidenceError(
                "split must be EvidenceSplit.FIT or EvidenceSplit.HELDOUT."
            )
        phase = _text(self.phase, "phase")
        layer = _text(self.layer, "layer")
        endpoint = _text(self.endpoint, "endpoint")
        rows = _finite_tensor(self.rows, f"snapshot {block_id!r}", rank=2)
        record_ids = _canonical_strings(
            self.record_ids,
            "record_ids",
            required=True,
        )
        gram_weight = _positive_float(self.gram_weight, "gram_weight")
        body = {
            "schema": RESIDUAL_SNAPSHOT_SCHEMA,
            "block_id": block_id,
            "split": self.split.value,
            "phase": phase,
            "layer": layer,
            "endpoint": endpoint,
            "rows_sha256": _tensor_sha256(rows),
            "row_count": int(rows.shape[0]),
            "source_width": int(rows.shape[1]),
            "record_ids": list(record_ids),
            "gram_weight": gram_weight,
        }
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "record_ids", record_ids)
        object.__setattr__(self, "gram_weight", gram_weight)
        object.__setattr__(self, "block_sha256", canonical_sha256(body))

    @property
    def source_width(self) -> int:
        return int(self.rows.shape[1])

    @property
    def location(self) -> str:
        return f"{self.phase}/{self.layer}/{self.endpoint}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_SNAPSHOT_SCHEMA,
            "block_id": self.block_id,
            "split": self.split.value,
            "phase": self.phase,
            "layer": self.layer,
            "endpoint": self.endpoint,
            "location": self.location,
            "row_count": int(self.rows.shape[0]),
            "source_width": self.source_width,
            "record_ids": list(self.record_ids),
            "gram_weight": self.gram_weight,
            "rows_sha256": _tensor_sha256(self.rows),
            "block_sha256": self.block_sha256,
        }


@dataclass(frozen=True, slots=True)
class DownstreamObservability:
    """One role-tagged downstream sensitivity/writer bound to circuit IR."""

    evidence_id: str
    node_id: str
    circuit_kind: CircuitKind | str
    operator: ResidualFacingOperator
    interface_ids: tuple[str, ...] = ()
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        evidence_id = _text(self.evidence_id, "evidence_id")
        node_id = _text(self.node_id, "node_id")
        kind = (
            self.circuit_kind.value
            if isinstance(self.circuit_kind, CircuitKind)
            else _text(self.circuit_kind, "circuit_kind")
        )
        if not isinstance(self.operator, ResidualFacingOperator):
            raise BalancedCircuitBasisEvidenceError(
                "operator must be a ResidualFacingOperator."
            )
        interface_ids = _canonical_strings(
            self.interface_ids,
            "interface_ids",
            required=False,
        )
        body = {
            "schema": OBSERVABILITY_EVIDENCE_SCHEMA,
            "evidence_id": evidence_id,
            "node_id": node_id,
            "circuit_kind": kind,
            "operator_sha256": self.operator.operator_sha256,
            "interface_ids": list(interface_ids),
        }
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "circuit_kind", kind)
        object.__setattr__(self, "interface_ids", interface_ids)
        object.__setattr__(
            self,
            "evidence_sha256",
            canonical_sha256(body),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVABILITY_EVIDENCE_SCHEMA,
            "evidence_id": self.evidence_id,
            "node_id": self.node_id,
            "circuit_kind": self.circuit_kind,
            "operator": self.operator.as_dict(),
            "interface_ids": list(self.interface_ids),
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class _RoleGram:
    raw: torch.Tensor
    balanced: torch.Tensor
    roles: Mapping[str, torch.Tensor]
    report: Mapping[str, Any]


def _operator_gram(operator: ResidualFacingOperator) -> torch.Tensor:
    tensor = operator.tensor
    width = operator.source_width
    if operator.diagonal:
        gram = torch.diag(tensor.square())
    elif operator.orientation is FeatureOrientation.INPUT:
        rows = tensor.movedim(operator.feature_axis, -1).reshape(-1, width)
        gram = rows.T @ rows
    else:
        rows = tensor.movedim(operator.feature_axis, 0).reshape(width, -1)
        gram = rows @ rows.T
    return (
        (gram + gram.T) * (0.5 * operator.gram_weight)
    ).contiguous()


def _role_balanced_operator_gram(
    values: Sequence[tuple[str, ResidualFacingOperator]],
    *,
    label: str,
) -> _RoleGram:
    ordered = tuple(
        sorted(
            values,
            key=lambda value: (
                value[1].role,
                value[0],
                value[1].name,
                value[1].operator_sha256,
            ),
        )
    )
    if not ordered:
        raise BalancedCircuitBasisEvidenceError(
            f"{label} must contain residual-facing operators."
        )
    widths = {operator.source_width for _key, operator in ordered}
    if len(widths) != 1:
        raise BalancedCircuitBasisEvidenceError(
            f"{label} operators do not share one residual width."
        )
    source_width = next(iter(widths))
    role_grams: dict[str, torch.Tensor] = {}
    operator_reports = []
    for binding, operator in ordered:
        contribution = _operator_gram(operator)
        role_grams.setdefault(
            operator.role,
            torch.zeros(
                (source_width, source_width),
                dtype=torch.float64,
            ),
        ).add_(contribution)
        operator_reports.append(
            {
                "binding": binding,
                "role": operator.role,
                "name": operator.name,
                "operator_sha256": operator.operator_sha256,
                "gram_sha256": _tensor_sha256(contribution),
                "trace_energy": float(torch.trace(contribution).item()),
            }
        )
    roles = tuple(sorted(role_grams))
    energies = {
        role: float(torch.trace(role_grams[role]).item())
        for role in roles
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in energies.values()
    ):
        raise BalancedCircuitBasisEvidenceError(
            f"Every {label} role must have finite positive Gram energy."
        )
    raw = sum(
        (role_grams[role] for role in roles),
        torch.zeros((source_width, source_width), dtype=torch.float64),
    )
    balanced = sum(
        (
            role_grams[role] / (len(roles) * energies[role])
            for role in roles
        ),
        torch.zeros((source_width, source_width), dtype=torch.float64),
    )
    raw = ((raw + raw.T) * 0.5).contiguous()
    balanced = ((balanced + balanced.T) * 0.5).contiguous()
    report = {
        "label": label,
        "source_width": source_width,
        "operator_count": len(ordered),
        "role_count": len(roles),
        "role_order": list(roles),
        "role_trace_balanced": True,
        "raw_trace_energy": float(torch.trace(raw).item()),
        "roles": {
            role: {
                "raw_trace_energy": energies[role],
                "balanced_trace_contribution": float(
                    torch.trace(
                        role_grams[role]
                        / (len(roles) * energies[role])
                    ).item()
                ),
                "gram_sha256": _tensor_sha256(role_grams[role]),
            }
            for role in roles
        },
        "operators": operator_reports,
        "raw_gram_sha256": _tensor_sha256(raw),
        "balanced_gram_sha256": _tensor_sha256(balanced),
        "large_tensor_roles_cannot_dominate_by_row_count": True,
    }
    return _RoleGram(
        raw=raw,
        balanced=balanced,
        roles=MappingProxyType(role_grams),
        report=MappingProxyType(report),
    )


@dataclass(frozen=True, slots=True)
class _SnapshotGram:
    raw: torch.Tensor
    balanced: torch.Tensor
    report: Mapping[str, Any]
    record_ids: frozenset[str]


def _snapshot_gram(
    blocks: Sequence[ResidualSnapshotBlock],
    *,
    split: EvidenceSplit,
) -> _SnapshotGram:
    ordered = tuple(
        sorted(
            blocks,
            key=lambda value: (
                value.phase,
                value.layer,
                value.endpoint,
                value.block_id,
                value.block_sha256,
            ),
        )
    )
    if not ordered or any(value.split is not split for value in ordered):
        raise BalancedCircuitBasisEvidenceError(
            f"{split.value} snapshots are empty or carry the wrong split."
        )
    if len({value.block_id for value in ordered}) != len(ordered):
        raise BalancedCircuitBasisEvidenceError(
            f"{split.value} snapshot block IDs must be unique."
        )
    widths = {value.source_width for value in ordered}
    if len(widths) != 1:
        raise BalancedCircuitBasisEvidenceError(
            f"{split.value} snapshots do not share one residual width."
        )
    source_width = next(iter(widths))
    contributions = []
    reports = []
    for block in ordered:
        contribution = (
            block.rows.T @ block.rows * block.gram_weight
        ).contiguous()
        trace = float(torch.trace(contribution).item())
        if not math.isfinite(trace) or trace <= 0.0:
            raise BalancedCircuitBasisEvidenceError(
                f"Snapshot block {block.block_id!r} has no positive energy."
            )
        contributions.append((contribution, trace))
        reports.append(
            {
                **block.as_dict(),
                "raw_trace_energy": trace,
                "balanced_trace_contribution": 1.0 / len(ordered),
                "gram_sha256": _tensor_sha256(contribution),
            }
        )
    raw = sum(
        (value for value, _trace in contributions),
        torch.zeros((source_width, source_width), dtype=torch.float64),
    )
    balanced = sum(
        (
            value / (len(contributions) * trace)
            for value, trace in contributions
        ),
        torch.zeros((source_width, source_width), dtype=torch.float64),
    )
    raw = ((raw + raw.T) * 0.5).contiguous()
    balanced = ((balanced + balanced.T) * 0.5).contiguous()
    record_ids = frozenset(
        record_id
        for block in ordered
        for record_id in block.record_ids
    )
    report = {
        "split": split.value,
        "source_width": source_width,
        "block_count": len(ordered),
        "row_count": sum(int(value.rows.shape[0]) for value in ordered),
        "record_count": len(record_ids),
        "record_ids_sha256": canonical_sha256(sorted(record_ids)),
        "phases": sorted({value.phase for value in ordered}),
        "layers": sorted({value.layer for value in ordered}),
        "endpoints": sorted({value.endpoint for value in ordered}),
        "locations": sorted({value.location for value in ordered}),
        "all_supplied_locations_contribute_equally_after_trace_normalization": (
            True
        ),
        "blocks": reports,
        "raw_trace_energy": float(torch.trace(raw).item()),
        "raw_gram_sha256": _tensor_sha256(raw),
        "balanced_gram_sha256": _tensor_sha256(balanced),
        "block_manifest_sha256": canonical_sha256(
            [value.block_sha256 for value in ordered]
        ),
    }
    return _SnapshotGram(
        raw=raw,
        balanced=balanced,
        report=MappingProxyType(report),
        record_ids=record_ids,
    )


def _psd_sqrt_and_rank(
    gram: torch.Tensor,
    *,
    name: str,
    rank_tolerance: float,
) -> tuple[torch.Tensor, int, tuple[float, ...]]:
    symmetric = ((gram + gram.T) * 0.5).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = max(1.0, float(eigenvalues.abs().max().item()))
    negative_tolerance = (
        max(
            rank_tolerance,
            torch.finfo(torch.float64).eps * int(gram.shape[0]) * 16,
        )
        * scale
    )
    minimum = float(eigenvalues[0].item())
    if minimum < -negative_tolerance:
        raise BalancedCircuitBasisEvidenceError(
            f"{name} is not positive semidefinite."
        )
    clipped = eigenvalues.clamp_min(0.0)
    threshold = (
        max(
            rank_tolerance,
            torch.finfo(torch.float64).eps * int(gram.shape[0]),
        )
        * max(float(clipped[-1].item()), torch.finfo(torch.float64).tiny)
    )
    rank = int((clipped > threshold).sum().item())
    square_root = (
        eigenvectors
        @ torch.diag(torch.sqrt(clipped))
        @ eigenvectors.T
    )
    square_root = ((square_root + square_root.T) * 0.5).contiguous()
    if not bool(torch.isfinite(square_root).all().item()):
        raise BalancedCircuitBasisEvidenceError(
            f"{name} square root is non-finite."
        )
    return (
        square_root,
        rank,
        tuple(float(value) for value in reversed(clipped.tolist())),
    )


def _effective_rank(
    values: torch.Tensor,
    *,
    rank_tolerance: float,
) -> int:
    if values.numel() == 0:
        return 0
    maximum = float(values.max().item())
    if maximum <= 0.0:
        return 0
    threshold = maximum * max(
        rank_tolerance,
        torch.finfo(torch.float64).eps * max(values.shape),
    )
    return int((values > threshold).sum().item())


def _canonical_subspace_basis(
    decoder: torch.Tensor,
    target_width: int,
) -> torch.Tensor:
    gram = decoder.T @ decoder
    try:
        projection = decoder @ torch.linalg.solve(gram, decoder.T)
    except RuntimeError as exc:
        raise BalancedCircuitBasisRankError(
            "Balanced decoder span is singular."
        ) from exc
    projection = ((projection + projection.T) * 0.5).contiguous()
    tolerance = (
        torch.finfo(torch.float64).eps
        * int(projection.shape[0])
        * 512.0
    )
    selected: list[torch.Tensor] = []
    for axis in range(int(projection.shape[0])):
        candidate = projection[:, axis].clone()
        for _pass in range(2):
            for existing in selected:
                candidate -= existing * torch.dot(existing, candidate)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(torch.argmax(candidate.abs()).item())
        if float(candidate[pivot].item()) < 0.0:
            candidate = -candidate
        selected.append(candidate)
        if len(selected) == target_width:
            break
    if len(selected) != target_width:
        raise BalancedCircuitBasisRankError(
            "Could not canonicalize the selected balanced subspace."
        )
    basis = torch.stack(selected, dim=1).contiguous()
    if not torch.allclose(
        basis.T @ basis,
        torch.eye(target_width, dtype=torch.float64),
        rtol=1e-11,
        atol=1e-12,
    ):
        raise BalancedCircuitBasisRankError(
            "Canonical balanced structural basis is not orthonormal."
        )
    return basis


def _retained_energy(basis: torch.Tensor, gram: torch.Tensor) -> float:
    denominator = float(torch.trace(gram).item())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise BalancedCircuitBasisEvidenceError(
            "Retention Gram must have finite positive trace."
        )
    numerator = float(torch.trace(basis.T @ gram @ basis).item())
    value = numerator / denominator
    if not math.isfinite(value):
        raise BalancedCircuitBasisEvidenceError(
            "Retained-energy diagnostic is non-finite."
        )
    return min(1.0, max(0.0, value))


def _validate_ir_bindings(
    graph: ModelCircuitGraph | None,
    observability: Sequence[DownstreamObservability],
    *,
    source_width: int,
    require_all_ir_writers: bool,
) -> dict[str, Any]:
    actual_kinds = sorted({value.circuit_kind for value in observability})
    if graph is None:
        if require_all_ir_writers:
            raise BalancedCircuitBasisEvidenceError(
                "require_all_ir_writers needs a ModelCircuitGraph."
            )
        return {
            "graph_supplied": False,
            "graph_fingerprint": None,
            "residual_width_verified": False,
            "all_writer_nodes_covered": None,
            "all_output_interfaces_covered": None,
            "observability_circuit_kinds": actual_kinds,
        }
    if not isinstance(graph, ModelCircuitGraph):
        raise BalancedCircuitBasisEvidenceError(
            "circuit_graph must be a ModelCircuitGraph or None."
        )
    residual_port_kinds = {
        PortKind.RESIDUAL.value,
        PortKind.NORMALIZED_RESIDUAL.value,
        PortKind.BRANCH_DELTA.value,
    }
    residual_widths = {
        port.dimensions[-1]
        for node in graph.nodes
        for port in (*node.inputs, *node.outputs)
        if port.kind in residual_port_kinds
        and port.dimensions
        and isinstance(port.dimensions[-1], int)
    }
    if residual_widths != {source_width}:
        raise BalancedCircuitBasisEvidenceError(
            "Circuit IR residual ports do not prove the evidence width."
        )
    graph_nodes = {value.node_id: value for value in graph.nodes}
    for value in observability:
        node = graph_nodes.get(value.node_id)
        if node is None or node.kind != value.circuit_kind:
            raise BalancedCircuitBasisEvidenceError(
                f"Observability evidence {value.evidence_id!r} is not bound "
                "to its declared circuit node/kind."
            )
    writer_kinds = {
        CircuitKind.ATTENTION.value,
        CircuitKind.SHORT_CONV.value,
        CircuitKind.RECURRENT_MIXER.value,
        CircuitKind.DENSE_MLP.value,
        CircuitKind.GATED_MLP.value,
    }
    required_writers = {
        node.node_id for node in graph.nodes if node.kind in writer_kinds
    }
    covered_nodes = {value.node_id for value in observability}
    writers_covered = required_writers.issubset(covered_nodes)
    output_interfaces = {
        value.interface_id
        for value in graph.interfaces
        if value.direction == "output"
    }
    covered_interfaces = {
        interface
        for value in observability
        for interface in value.interface_ids
    }
    interfaces_covered = output_interfaces.issubset(covered_interfaces)
    if require_all_ir_writers and not writers_covered:
        missing = sorted(required_writers.difference(covered_nodes))
        raise BalancedCircuitBasisEvidenceError(
            "Downstream observability omits IR writer nodes: "
            + ", ".join(missing)
            + "."
        )
    if output_interfaces and not interfaces_covered:
        missing = sorted(output_interfaces.difference(covered_interfaces))
        raise BalancedCircuitBasisEvidenceError(
            "Downstream observability omits output interfaces: "
            + ", ".join(missing)
            + "."
        )
    return {
        "graph_supplied": True,
        "graph_fingerprint": graph.fingerprint,
        "family": graph.family,
        "residual_width_verified": True,
        "required_writer_node_count": len(required_writers),
        "covered_writer_node_count": len(
            required_writers.intersection(covered_nodes)
        ),
        "all_writer_nodes_covered": writers_covered,
        "output_interface_count": len(output_interfaces),
        "covered_output_interface_count": len(
            output_interfaces.intersection(covered_interfaces)
        ),
        "all_output_interfaces_covered": interfaces_covered,
        "observability_circuit_kinds": actual_kinds,
    }


@dataclass(frozen=True, slots=True)
class BalancedCircuitBasisResult:
    """Dense structural map plus its classical balanced realization."""

    basis: torch.Tensor
    balanced_encoder: torch.Tensor
    balanced_decoder: torch.Tensor
    full_weight_gram: torch.Tensor
    controllability_gram: torch.Tensor
    observability_gram: torch.Tensor
    heldout_gram: torch.Tensor
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        basis = _finite_tensor(self.basis, "basis", rank=2)
        encoder = _finite_tensor(
            self.balanced_encoder,
            "balanced_encoder",
            rank=2,
        )
        decoder = _finite_tensor(
            self.balanced_decoder,
            "balanced_decoder",
            rank=2,
        )
        source_width, target_width = map(int, basis.shape)
        grams = tuple(
            _finite_tensor(value, name, rank=2)
            for name, value in (
                ("full_weight_gram", self.full_weight_gram),
                ("controllability_gram", self.controllability_gram),
                ("observability_gram", self.observability_gram),
                ("heldout_gram", self.heldout_gram),
            )
        )
        if (
            tuple(encoder.shape) != (target_width, source_width)
            or tuple(decoder.shape) != (source_width, target_width)
            or any(
                tuple(value.shape) != (source_width, source_width)
                for value in grams
            )
            or not torch.allclose(
                basis.T @ basis,
                torch.eye(target_width, dtype=torch.float64),
                rtol=1e-11,
                atol=1e-12,
            )
            or not torch.allclose(
                encoder @ decoder,
                torch.eye(target_width, dtype=torch.float64),
                rtol=1e-9,
                atol=1e-10,
            )
            or not isinstance(self.report, Mapping)
        ):
            raise BalancedCircuitBasisError(
                "BalancedCircuitBasisResult invariants are invalid."
            )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "balanced_encoder", encoder)
        object.__setattr__(self, "balanced_decoder", decoder)
        object.__setattr__(self, "full_weight_gram", grams[0])
        object.__setattr__(self, "controllability_gram", grams[1])
        object.__setattr__(self, "observability_gram", grams[2])
        object.__setattr__(self, "heldout_gram", grams[3])
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def source_width(self) -> int:
        return int(self.basis.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.basis.shape[1])

    @property
    def map_sha256(self) -> str:
        value = self.report.get("hashes", {}).get("basis_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise BalancedCircuitBasisError("Basis report hash is invalid.")
        return value

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        value = _finite_runtime_rows(rows, "rows")
        if int(value.shape[-1]) != self.source_width:
            raise BalancedCircuitBasisError(
                "rows do not match the source residual width."
            )
        leading_shape = tuple(int(size) for size in value.shape[:-1])
        flat = value.reshape(math.prod(leading_shape), self.source_width)
        return (flat @ self.basis).reshape(
            *leading_shape, self.target_width
        ).contiguous()

    def decode(self, rows: torch.Tensor) -> torch.Tensor:
        value = _finite_runtime_rows(rows, "rows")
        if int(value.shape[-1]) != self.target_width:
            raise BalancedCircuitBasisError(
                "rows do not match the target residual width."
            )
        leading_shape = tuple(int(size) for size in value.shape[:-1])
        flat = value.reshape(math.prod(leading_shape), self.target_width)
        return (flat @ self.basis.T).reshape(
            *leading_shape, self.source_width
        ).contiguous()

    def balanced_encode(self, rows: torch.Tensor) -> torch.Tensor:
        value = _finite_runtime_rows(rows, "rows")
        if int(value.shape[-1]) != self.source_width:
            raise BalancedCircuitBasisError(
                "rows do not match the source residual width."
            )
        leading_shape = tuple(int(size) for size in value.shape[:-1])
        flat = value.reshape(math.prod(leading_shape), self.source_width)
        return (flat @ self.balanced_encoder.T).reshape(
            *leading_shape, self.target_width
        ).contiguous()

    def balanced_decode(self, rows: torch.Tensor) -> torch.Tensor:
        value = _finite_runtime_rows(rows, "rows")
        if int(value.shape[-1]) != self.target_width:
            raise BalancedCircuitBasisError(
                "rows do not match the target residual width."
            )
        leading_shape = tuple(int(size) for size in value.shape[:-1])
        flat = value.reshape(math.prod(leading_shape), self.target_width)
        return (flat @ self.balanced_decoder.T).reshape(
            *leading_shape, self.source_width
        ).contiguous()

    def apply(self, rows: torch.Tensor) -> torch.Tensor:
        return self.encode(rows)

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        return self.encode(rows)

    def fold_consumer_weight(self, weight: torch.Tensor) -> torch.Tensor:
        value = _finite_tensor(weight, "consumer weight", rank=2)
        if int(value.shape[1]) != self.source_width:
            raise BalancedCircuitBasisError(
                "Consumer weight does not match source_width."
            )
        return (value @ self.basis).contiguous()

    def fold_producer_weight(self, weight: torch.Tensor) -> torch.Tensor:
        value = _finite_tensor(weight, "producer weight", rank=2)
        if int(value.shape[0]) != self.source_width:
            raise BalancedCircuitBasisError(
                "Producer weight does not match source_width."
            )
        return (self.basis.T @ value).contiguous()

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: torch.dtype | None = None,
        device: Any = "cpu",
    ) -> torch.Tensor:
        maximum = _positive_int(max_elements, "max_elements")
        if int(self.basis.numel()) > maximum:
            raise BalancedCircuitBasisError(
                "Dense basis exceeds the materialization bound."
            )
        target_dtype = self.basis.dtype if dtype is None else dtype
        if not isinstance(target_dtype, torch.dtype):
            raise BalancedCircuitBasisError(
                "dtype must be a torch dtype or None."
            )
        result = self.basis.to(
            device=device,
            dtype=target_dtype,
        ).contiguous()
        if not bool(torch.isfinite(result).all().item()):
            raise BalancedCircuitBasisError(
                "Materialized basis is non-finite."
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def build_balanced_circuit_basis(
    structural_operators: Sequence[ResidualFacingOperator],
    fit_snapshots: Sequence[ResidualSnapshotBlock],
    heldout_snapshots: Sequence[ResidualSnapshotBlock],
    downstream_observability: Sequence[DownstreamObservability],
    target_width: int,
    *,
    circuit_graph: ModelCircuitGraph | None = None,
    structural_controllability_weight: float = 0.35,
    cached_controllability_weight: float = 0.65,
    gram_regularization: float = 1e-10,
    rank_tolerance: float = 1e-12,
    max_condition_number: float = 1e12,
    required_observability_kinds: Sequence[CircuitKind | str] = (),
    require_all_ir_writers: bool = False,
    minimum_heldout_retained_energy_fraction: float = 0.0,
) -> BalancedCircuitBasisResult:
    """Build a deterministic global balanced residual basis.

    Selection consumes every supplied structural operator, every fit snapshot
    location, and every downstream observability operator.  Held-out snapshots
    are required, provenance-disjoint, and excluded from every selection Gram.
    """

    structural = tuple(structural_operators)
    fit = tuple(fit_snapshots)
    heldout = tuple(heldout_snapshots)
    observability = tuple(downstream_observability)
    if not structural or any(
        not isinstance(value, ResidualFacingOperator)
        for value in structural
    ):
        raise BalancedCircuitBasisEvidenceError(
            "structural_operators must contain ResidualFacingOperator values."
        )
    if not observability or any(
        not isinstance(value, DownstreamObservability)
        for value in observability
    ):
        raise BalancedCircuitBasisEvidenceError(
            "downstream_observability must contain bound evidence."
        )
    if len({value.operator_sha256 for value in structural}) != len(
        structural
    ):
        raise BalancedCircuitBasisEvidenceError(
            "Structural operators must not contain duplicate tensors/roles."
        )
    if len({value.evidence_id for value in observability}) != len(
        observability
    ):
        raise BalancedCircuitBasisEvidenceError(
            "Observability evidence IDs must be unique."
        )
    target_width = _positive_int(target_width, "target_width")
    structural_weight = _positive_float(
        structural_controllability_weight,
        "structural_controllability_weight",
    )
    cached_weight = _positive_float(
        cached_controllability_weight,
        "cached_controllability_weight",
    )
    weight_total = structural_weight + cached_weight
    structural_weight /= weight_total
    cached_weight /= weight_total
    regularization = _positive_float(
        gram_regularization,
        "gram_regularization",
    )
    rank_tolerance = _positive_float(
        rank_tolerance,
        "rank_tolerance",
    )
    max_condition_number = _positive_float(
        max_condition_number,
        "max_condition_number",
    )
    if max_condition_number < 1.0:
        raise BalancedCircuitBasisError(
            "max_condition_number must be at least one."
        )
    minimum_heldout = _nonnegative_float(
        minimum_heldout_retained_energy_fraction,
        "minimum_heldout_retained_energy_fraction",
    )
    if minimum_heldout > 1.0:
        raise BalancedCircuitBasisError(
            "minimum_heldout_retained_energy_fraction cannot exceed one."
        )
    if not isinstance(require_all_ir_writers, bool):
        raise BalancedCircuitBasisError(
            "require_all_ir_writers must be boolean."
        )
    required_kinds = tuple(
        sorted(
            {
                value.value
                if isinstance(value, CircuitKind)
                else _text(value, "required_observability_kind")
                for value in required_observability_kinds
            }
        )
    )
    actual_kinds = {value.circuit_kind for value in observability}
    missing_kinds = sorted(set(required_kinds).difference(actual_kinds))
    if missing_kinds:
        raise BalancedCircuitBasisEvidenceError(
            "Missing required observability circuit kinds: "
            + ", ".join(missing_kinds)
            + "."
        )

    original_threads = torch.get_num_threads()
    with _NUMERICAL_LOCK:
        try:
            torch.set_num_threads(1)
            with torch.no_grad():
                structural_gram = _role_balanced_operator_gram(
                    tuple(
                        (value.name, value)
                        for value in structural
                    ),
                    label="whole_weight_structural",
                )
                fit_gram = _snapshot_gram(
                    fit,
                    split=EvidenceSplit.FIT,
                )
                heldout_gram = _snapshot_gram(
                    heldout,
                    split=EvidenceSplit.HELDOUT,
                )
                observable_gram = _role_balanced_operator_gram(
                    tuple(
                        (value.evidence_id, value.operator)
                        for value in observability
                    ),
                    label="downstream_observability",
                )
                widths = {
                    int(structural_gram.raw.shape[0]),
                    int(fit_gram.raw.shape[0]),
                    int(heldout_gram.raw.shape[0]),
                    int(observable_gram.raw.shape[0]),
                }
                if len(widths) != 1:
                    raise BalancedCircuitBasisEvidenceError(
                        "Weight, fit, held-out, and observability evidence "
                        "do not share one residual width."
                    )
                source_width = next(iter(widths))
                if target_width > source_width:
                    raise BalancedCircuitBasisError(
                        "target_width cannot exceed source_width."
                    )
                overlap = fit_gram.record_ids.intersection(
                    heldout_gram.record_ids
                )
                if overlap:
                    raise BalancedCircuitBasisEvidenceError(
                        "Fit and held-out residual evidence share record IDs: "
                        + ", ".join(sorted(overlap))
                        + "."
                    )
                ir_report = _validate_ir_bindings(
                    circuit_graph,
                    observability,
                    source_width=source_width,
                    require_all_ir_writers=require_all_ir_writers,
                )

                controllability_unregularized = (
                    structural_weight * structural_gram.balanced
                    + cached_weight * fit_gram.balanced
                )
                controllability_unregularized = (
                    (
                        controllability_unregularized
                        + controllability_unregularized.T
                    )
                    * 0.5
                ).contiguous()
                observability_unregularized = (
                    observable_gram.balanced
                )
                controllability_sqrt_raw, controllability_rank, _ = (
                    _psd_sqrt_and_rank(
                        controllability_unregularized,
                        name="unregularized controllability Gram",
                        rank_tolerance=rank_tolerance,
                    )
                )
                observability_sqrt_raw, observability_rank, _ = (
                    _psd_sqrt_and_rank(
                        observability_unregularized,
                        name="unregularized observability Gram",
                        rank_tolerance=rank_tolerance,
                    )
                )
                raw_hankel = (
                    observability_sqrt_raw.T
                    @ controllability_sqrt_raw
                )
                raw_hankel_singular = torch.linalg.svdvals(raw_hankel)
                raw_hankel_rank = _effective_rank(
                    raw_hankel_singular,
                    rank_tolerance=rank_tolerance,
                )
                if (
                    controllability_rank < target_width
                    or observability_rank < target_width
                    or raw_hankel_rank < target_width
                ):
                    raise BalancedCircuitBasisRankError(
                        "Unregularized controllable/observable intersection "
                        f"ranks are C={controllability_rank}, "
                        f"O={observability_rank}, H={raw_hankel_rank}; "
                        f"target_width={target_width}."
                    )

                identity = torch.eye(
                    source_width,
                    dtype=torch.float64,
                )
                ridge = identity * (
                    regularization / source_width
                )
                controllability = (
                    controllability_unregularized + ridge
                ).contiguous()
                observable = (
                    observability_unregularized + ridge
                ).contiguous()
                controllability_sqrt, _, _ = _psd_sqrt_and_rank(
                    controllability,
                    name="regularized controllability Gram",
                    rank_tolerance=rank_tolerance,
                )
                observability_sqrt, _, _ = _psd_sqrt_and_rank(
                    observable,
                    name="regularized observability Gram",
                    rank_tolerance=rank_tolerance,
                )
                try:
                    balanced = build_balanced_krylov_basis(
                        controllability_sqrt.numpy(),
                        target_width,
                        output_sensitivities=(
                            observability_sqrt.numpy()
                        ),
                        center=False,
                        svd_atol=0.0,
                        svd_rtol=rank_tolerance,
                        max_condition_number=max_condition_number,
                        allow_svd_fallback=False,
                    )
                except (
                    BalancedKrylovRankError,
                    BalancedKrylovConditionError,
                    BalancedKrylovError,
                ) as exc:
                    raise BalancedCircuitBasisRankError(
                        "Regularized balanced POD failed closed."
                    ) from exc
                encoder = torch.from_numpy(
                    balanced.encoder.copy()
                ).to(dtype=torch.float64)
                decoder = torch.from_numpy(
                    balanced.decoder.copy()
                ).to(dtype=torch.float64)
                basis = _canonical_subspace_basis(
                    decoder,
                    target_width,
                )
                projection = basis @ basis.T

                heldout_retained = _retained_energy(
                    basis,
                    heldout_gram.raw,
                )
                heldout_gate_passed = (
                    heldout_retained + 1e-15 >= minimum_heldout
                )
                if not heldout_gate_passed:
                    raise BalancedCircuitBasisEvidenceError(
                        "Balanced basis failed held-out retained-energy gate."
                    )
                raw_hankel_values = tuple(
                    float(value)
                    for value in raw_hankel_singular.tolist()
                )
                raw_hankel_total = sum(
                    value * value for value in raw_hankel_values
                )
                captured_hankel = (
                    sum(
                        value * value
                        for value in raw_hankel_values[:target_width]
                    )
                    / max(
                        raw_hankel_total,
                        torch.finfo(torch.float64).tiny,
                    )
                )
                structural_role_retention = {
                    role: _retained_energy(basis, gram)
                    for role, gram in sorted(
                        structural_gram.roles.items()
                    )
                }
                observable_role_retention = {
                    role: _retained_energy(basis, gram)
                    for role, gram in sorted(
                        observable_gram.roles.items()
                    )
                }
                selection_payload = {
                    "schema": BALANCED_CIRCUIT_BASIS_SCHEMA,
                    "source_width": source_width,
                    "target_width": target_width,
                    "whole_weight_manifest_sha256": canonical_sha256(
                        [
                            value.operator_sha256
                            for value in sorted(
                                structural,
                                key=lambda item: (
                                    item.role,
                                    item.name,
                                    item.operator_sha256,
                                ),
                            )
                        ]
                    ),
                    "fit_manifest_sha256": fit_gram.report[
                        "block_manifest_sha256"
                    ],
                    "observability_manifest_sha256": canonical_sha256(
                        [
                            value.evidence_sha256
                            for value in sorted(
                                observability,
                                key=lambda item: (
                                    item.evidence_id,
                                    item.evidence_sha256,
                                ),
                            )
                        ]
                    ),
                    "circuit_graph_fingerprint": ir_report[
                        "graph_fingerprint"
                    ],
                    "structural_controllability_weight": (
                        structural_weight
                    ),
                    "cached_controllability_weight": cached_weight,
                    "gram_regularization": regularization,
                    "rank_tolerance": rank_tolerance,
                    "max_condition_number": max_condition_number,
                    "required_observability_kinds": list(
                        required_kinds
                    ),
                    "basis_sha256": _tensor_sha256(basis),
                    "balanced_encoder_sha256": _tensor_sha256(
                        encoder
                    ),
                    "balanced_decoder_sha256": _tensor_sha256(
                        decoder
                    ),
                }
                selection_sha256 = canonical_sha256(
                    selection_payload
                )
                heldout_payload = {
                    "manifest_sha256": heldout_gram.report[
                        "block_manifest_sha256"
                    ],
                    "record_ids_sha256": heldout_gram.report[
                        "record_ids_sha256"
                    ],
                    "retained_energy_fraction": heldout_retained,
                    "minimum_retained_energy_fraction": (
                        minimum_heldout
                    ),
                    "gate_passed": heldout_gate_passed,
                    "used_for_selection": False,
                }
                report: dict[str, Any] = {
                    "schema": BALANCED_CIRCUIT_BASIS_SCHEMA,
                    "method": (
                        "regularized-role-balanced-weight-cache-"
                        "observability-balanced-pod"
                    ),
                    "source_width": source_width,
                    "target_width": target_width,
                    "aggressive_structural_fold_coordinate_map": True,
                    "not_activation_pca": True,
                    "not_raw_weight_svd": True,
                    "not_independent_local_regression": True,
                    "whole_weight": dict(structural_gram.report),
                    "cached_controllability": {
                        **dict(fit_gram.report),
                        "selection_split": True,
                    },
                    "heldout": {
                        **dict(heldout_gram.report),
                        **heldout_payload,
                    },
                    "downstream_observability": {
                        **dict(observable_gram.report),
                        "evidence": [
                            value.as_dict()
                            for value in sorted(
                                observability,
                                key=lambda item: (
                                    item.evidence_id,
                                    item.evidence_sha256,
                                ),
                            )
                        ],
                        "required_circuit_kinds": list(required_kinds),
                        "required_circuit_kinds_present": True,
                        "writer_spans_are_observability_approximations": (
                            True
                        ),
                        "output_interfaces_are_observability_approximations": (
                            True
                        ),
                    },
                    "circuit_ir": ir_report,
                    "fit_heldout_isolation": {
                        "fit_record_count": len(fit_gram.record_ids),
                        "heldout_record_count": len(
                            heldout_gram.record_ids
                        ),
                        "shared_record_count": 0,
                        "record_ids_disjoint": True,
                        "heldout_used_for_selection": False,
                        "selection_sha256_excludes_heldout": True,
                    },
                    "controllability_blend": {
                        "whole_weight_role_balanced_weight": (
                            structural_weight
                        ),
                        "cached_residual_weight": cached_weight,
                        "weights_normalized_to_one": True,
                        "unregularized_gram_sha256": _tensor_sha256(
                            controllability_unregularized
                        ),
                        "regularized_gram_sha256": _tensor_sha256(
                            controllability
                        ),
                    },
                    "regularized_balanced_truncation": {
                        "enabled": True,
                        "gram_regularization": regularization,
                        "regularization_isotropic_trace": (
                            regularization
                        ),
                        "unregularized_controllability_rank": (
                            controllability_rank
                        ),
                        "unregularized_observability_rank": (
                            observability_rank
                        ),
                        "unregularized_hankel_rank": raw_hankel_rank,
                        "unregularized_hankel_singular_values": list(
                            raw_hankel_values
                        ),
                        "captured_hankel_squared_energy_fraction": (
                            captured_hankel
                        ),
                        "regularized_balanced_pod_report": (
                            balanced.report.as_dict()
                        ),
                        "svd_fallback_used": False,
                    },
                    "retention": {
                        "whole_weight_raw_energy_fraction": (
                            _retained_energy(
                                basis,
                                structural_gram.raw,
                            )
                        ),
                        "whole_weight_role_balanced_fraction": (
                            _retained_energy(
                                basis,
                                structural_gram.balanced,
                            )
                        ),
                        "fit_cached_residual_fraction": (
                            _retained_energy(
                                basis,
                                fit_gram.raw,
                            )
                        ),
                        "heldout_cached_residual_fraction": (
                            heldout_retained
                        ),
                        "downstream_observability_fraction": (
                            _retained_energy(
                                basis,
                                observable_gram.raw,
                            )
                        ),
                        "structural_roles": (
                            structural_role_retention
                        ),
                        "observability_roles": (
                            observable_role_retention
                        ),
                    },
                    "basis": {
                        "dense": True,
                        "orthonormal_structural_map": True,
                        "balanced_span_preserved": True,
                        "consumer_transport": "W_source @ B",
                        "producer_transport": "B.T @ W_source",
                        "balanced_encoder_decoder_retained_for_audit": (
                            True
                        ),
                        "biorthogonality_max_abs_error": float(
                            (
                                encoder @ decoder
                                - torch.eye(
                                    target_width,
                                    dtype=torch.float64,
                                )
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                        "orthonormality_max_abs_error": float(
                            (
                                basis.T @ basis
                                - torch.eye(
                                    target_width,
                                    dtype=torch.float64,
                                )
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    },
                    "determinism": {
                        "cpu_float64": True,
                        "internal_torch_threads": 1,
                        "canonical_input_order": True,
                        "canonical_balanced_subspace_projector": True,
                        "canonical_coordinate_axis_order": True,
                    },
                    "hashes": {
                        "basis_sha256": _tensor_sha256(basis),
                        "projection_sha256": _tensor_sha256(
                            projection
                        ),
                        "balanced_encoder_sha256": _tensor_sha256(
                            encoder
                        ),
                        "balanced_decoder_sha256": _tensor_sha256(
                            decoder
                        ),
                        "full_weight_gram_sha256": _tensor_sha256(
                            structural_gram.raw
                        ),
                        "controllability_gram_sha256": _tensor_sha256(
                            controllability
                        ),
                        "observability_gram_sha256": _tensor_sha256(
                            observable
                        ),
                        "heldout_gram_sha256": _tensor_sha256(
                            heldout_gram.raw
                        ),
                        "selection_sha256": selection_sha256,
                        "heldout_evaluation_sha256": canonical_sha256(
                            heldout_payload
                        ),
                    },
                    "execution": {
                        "precomputed_parent_cache_only": True,
                        "model_forward_calls": 0,
                        "parent_forward_calls": 0,
                        "trace_capture_calls": 0,
                        "training_steps": 0,
                        "backward_calls": 0,
                        "optimizer_steps": 0,
                        "autograd_used": False,
                        "filesystem_writes": 0,
                    },
                }
                report["run_sha256"] = canonical_sha256(report)
                return BalancedCircuitBasisResult(
                    basis=basis,
                    balanced_encoder=encoder,
                    balanced_decoder=decoder,
                    full_weight_gram=structural_gram.raw,
                    controllability_gram=controllability,
                    observability_gram=observable,
                    heldout_gram=heldout_gram.raw,
                    report=report,
                )
        finally:
            torch.set_num_threads(original_threads)


__all__ = [
    "BALANCED_CIRCUIT_BASIS_SCHEMA",
    "BalancedCircuitBasisError",
    "BalancedCircuitBasisEvidenceError",
    "BalancedCircuitBasisRankError",
    "BalancedCircuitBasisResult",
    "DownstreamObservability",
    "EvidenceSplit",
    "ResidualSnapshotBlock",
    "build_balanced_circuit_basis",
]
