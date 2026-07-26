"""Analytic full-rank completion of trace-identified coordinate maps.

A cached trace fit identifies only the row space of its design matrix.  A
minimum-norm decoder therefore says nothing about target-coordinate directions
orthogonal to that row space.  Treating those zero rows as learned structure is
one direct route to a collapsed child.

This module keeps the identified action of a fitted map and completes only its
unidentifiable target-coordinate complement.  Completion directions come from
the joint Gram geometry of immutable parent operators.  An optional cached
Fisher Gram can supplement, but never replace, that structural objective.  An
optional bridge prior supplies a rectangular orthogonal-Procrustes orientation.

The compiler is deliberately architecture neutral: target and source widths
may differ in either direction, and parent operators use the public
``ResidualFacingOperator`` feature-axis contract.  It performs no model
execution, trace capture, training, optimization, pruning, quantization, or
filesystem writes.  Candidate order and numeric diagnostics are navigation
evidence only; they are never quality gates or semantic acceptance decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_weight_complete_basis import (
    FeatureOrientation,
    ResidualFacingOperator,
)


ANALYTIC_COORDINATE_COMPLETION_EVIDENCE_SCHEMA = (
    "neural-forge-analytic-coordinate-completion-evidence-v1"
)
ANALYTIC_COORDINATE_COMPLETION_CANDIDATE_SCHEMA = (
    "neural-forge-analytic-coordinate-completion-candidate-v1"
)
ANALYTIC_COORDINATE_COMPLETION_FRONTIER_SCHEMA = (
    "neural-forge-analytic-coordinate-completion-frontier-v1"
)

_NUMERICAL_LOCK = threading.RLock()
_HEX_DIGITS = frozenset("0123456789abcdef")


class AnalyticCoordinateCompletionError(ValueError):
    """Coordinate evidence or analytic completion is invalid."""


class AnalyticCoordinateCompletionRankError(AnalyticCoordinateCompletionError):
    """Exact fitted-subspace preservation makes full rank impossible."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = MappingProxyType(dict(report))
        super().__init__(
            "The trace-identified action and target nullspace cannot support "
            "a full-rank target-to-source coordinate map without changing "
            "the fitted subspace."
        )


@dataclass(frozen=True, slots=True)
class AnalyticCoordinateCompletionCandidate:
    """One standalone dense coordinate map and its immutable receipt."""

    recipe: str
    coordinate_map: torch.Tensor
    nullspace_update: torch.Tensor
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        coordinate_map = _matrix(self.coordinate_map, "coordinate_map")
        update = _matrix(self.nullspace_update, "nullspace_update")
        if tuple(update.shape) != tuple(coordinate_map.shape):
            raise AnalyticCoordinateCompletionError(
                "nullspace_update must match coordinate_map shape."
            )
        if not isinstance(self.recipe, str) or not self.recipe.strip():
            raise AnalyticCoordinateCompletionError("recipe must be non-empty.")
        if not isinstance(self.report, Mapping):
            raise AnalyticCoordinateCompletionError("report must be a mapping.")
        object.__setattr__(self, "recipe", self.recipe.strip())
        object.__setattr__(self, "coordinate_map", coordinate_map)
        object.__setattr__(self, "nullspace_update", update)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def target_width(self) -> int:
        return int(self.coordinate_map.shape[0])

    @property
    def source_width(self) -> int:
        return int(self.coordinate_map.shape[1])

    @property
    def map_sha256(self) -> str:
        value = self.report.get("coordinate_map_sha256")
        return _digest(value, "candidate coordinate_map_sha256")

    def decode(self, target_rows: torch.Tensor) -> torch.Tensor:
        """Map target-coordinate rows into the parent/source chart."""

        rows = _floating_tensor(target_rows, "target_rows")
        if int(rows.shape[-1]) != self.target_width:
            raise AnalyticCoordinateCompletionError(
                "target_rows final axis does not match target_width."
            )
        return (rows @ self.coordinate_map).contiguous()

    def materialize(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: Any = "cpu",
    ) -> torch.Tensor:
        """Return a detached dense copy for a later family-specific installer."""

        if not isinstance(dtype, torch.dtype):
            raise AnalyticCoordinateCompletionError("dtype must be a torch dtype.")
        result = self.coordinate_map.to(device=device, dtype=dtype).contiguous()
        if not bool(torch.isfinite(result).all().item()):
            raise AnalyticCoordinateCompletionError(
                "Materialized coordinate map is not finite."
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class AnalyticCoordinateCompletionFrontier:
    """Trace decomposition, structural geometry, and distinct completions."""

    identified_map: torch.Tensor
    identified_target_basis: torch.Tensor
    target_null_basis: torch.Tensor
    occupied_source_basis: torch.Tensor
    completion_source_basis: torch.Tensor
    candidates: tuple[AnalyticCoordinateCompletionCandidate, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        identified = _matrix(self.identified_map, "identified_map")
        target_width, source_width = map(int, identified.shape)
        identified_basis = _matrix_allow_empty_columns(
            self.identified_target_basis,
            "identified_target_basis",
            rows=target_width,
        )
        null_basis = _matrix_allow_empty_columns(
            self.target_null_basis,
            "target_null_basis",
            rows=target_width,
        )
        occupied = _matrix_allow_empty_columns(
            self.occupied_source_basis,
            "occupied_source_basis",
            rows=source_width,
        )
        completion = _matrix_allow_empty_columns(
            self.completion_source_basis,
            "completion_source_basis",
            rows=source_width,
        )
        candidates = tuple(self.candidates)
        if not candidates or any(
            not isinstance(value, AnalyticCoordinateCompletionCandidate)
            for value in candidates
        ):
            raise AnalyticCoordinateCompletionError(
                "A completion frontier needs candidate objects."
            )
        if any(
            tuple(value.coordinate_map.shape) != tuple(identified.shape)
            for value in candidates
        ):
            raise AnalyticCoordinateCompletionError(
                "Every candidate must match the identified map shape."
            )
        if not isinstance(self.report, Mapping):
            raise AnalyticCoordinateCompletionError("report must be a mapping.")
        object.__setattr__(self, "identified_map", identified)
        object.__setattr__(self, "identified_target_basis", identified_basis)
        object.__setattr__(self, "target_null_basis", null_basis)
        object.__setattr__(self, "occupied_source_basis", occupied)
        object.__setattr__(self, "completion_source_basis", completion)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def candidate(self, recipe: str) -> AnalyticCoordinateCompletionCandidate:
        for value in self.candidates:
            if value.recipe == recipe:
                return value
        raise KeyError(recipe)

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _floating_tensor(value: Any, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.ndim < 1
        or any(int(size) <= 0 for size in value.shape)
    ):
        raise AnalyticCoordinateCompletionError(
            f"{label} must be one materialized dense real floating tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise AnalyticCoordinateCompletionError(f"{label} contains non-finite values.")
    return result


def _matrix(value: Any, label: str) -> torch.Tensor:
    result = _floating_tensor(value, label)
    if result.ndim != 2:
        raise AnalyticCoordinateCompletionError(f"{label} must be rank two.")
    return result


def _matrix_allow_empty_columns(
    value: Any,
    label: str,
    *,
    rows: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.ndim != 2
        or tuple(value.shape)[0] != rows
        or int(value.shape[1]) < 0
    ):
        raise AnalyticCoordinateCompletionError(
            f"{label} must be a dense floating matrix with {rows} rows."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise AnalyticCoordinateCompletionError(f"{label} contains non-finite values.")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise AnalyticCoordinateCompletionError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _unit_interval(value: Any, label: str, *, upper_open: bool = False) -> float:
    if isinstance(value, bool):
        raise AnalyticCoordinateCompletionError(f"{label} must be within [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalyticCoordinateCompletionError(
            f"{label} must be within [0, 1]."
        ) from exc
    upper_valid = result < 1.0 if upper_open else result <= 1.0
    if not math.isfinite(result) or result < 0.0 or not upper_valid:
        boundary = "[0, 1)" if upper_open else "[0, 1]"
        raise AnalyticCoordinateCompletionError(f"{label} must be within {boundary}.")
    return result


def _positive_float_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AnalyticCoordinateCompletionError(
            f"{label} must be a positive finite number or None."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalyticCoordinateCompletionError(
            f"{label} must be a positive finite number or None."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise AnalyticCoordinateCompletionError(
            f"{label} must be a positive finite number or None."
        )
    return result


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
    gram = gram * operator.gram_weight
    return ((gram + gram.T) * 0.5).contiguous()


def _operator_manifest(
    operators: Sequence[ResidualFacingOperator],
) -> tuple[ResidualFacingOperator, ...]:
    values = tuple(operators)
    if not values or any(not isinstance(value, ResidualFacingOperator) for value in values):
        raise AnalyticCoordinateCompletionError(
            "parent_operators must contain ResidualFacingOperator values."
        )
    # ``ResidualFacingOperator`` takes an owning clone at construction time,
    # but a torch tensor itself remains mutable even when stored on a frozen
    # dataclass.  Do not trust the cached ``operator_sha256`` and then consume
    # the exposed tensor later: that would let an in-place mutation change the
    # geometry without changing the authenticated evidence.  Reconstruct each
    # operator once.  The reconstructed values are the immutable-by-ownership
    # snapshots used by both the evidence digest and the numeric solve.
    snapshots: list[ResidualFacingOperator] = []
    for value in values:
        try:
            snapshot = ResidualFacingOperator(
                role=value.role,
                name=value.name,
                tensor=value.tensor,
                orientation=value.orientation,
                feature_axis=value.feature_axis,
                gram_weight=value.gram_weight,
                diagonal=value.diagonal,
            )
        except ValueError as exc:
            raise AnalyticCoordinateCompletionError(
                "A residual-facing parent operator is no longer a valid "
                "authenticated tensor."
            ) from exc
        if snapshot.operator_sha256 != value.operator_sha256:
            raise AnalyticCoordinateCompletionError(
                "A residual-facing parent operator changed after its "
                "authenticated receipt was constructed."
            )
        snapshots.append(snapshot)
    ordered = tuple(
        sorted(
            snapshots,
            key=lambda value: (
                value.role,
                value.name,
                value.orientation.value,
                value.feature_axis,
                value.operator_sha256,
            ),
        )
    )
    if len({value.operator_sha256 for value in ordered}) != len(ordered):
        raise AnalyticCoordinateCompletionError(
            "Duplicate residual-facing parent operators are not allowed."
        )
    if len({value.source_width for value in ordered}) != 1:
        raise AnalyticCoordinateCompletionError(
            "Every parent operator must share one source width."
        )
    return ordered


def _coordinate_completion_evidence_sha256_from_snapshots(
    design: torch.Tensor,
    fitted: torch.Tensor,
    operators: Sequence[ResidualFacingOperator],
    *,
    bridge: torch.Tensor | None,
    fisher: torch.Tensor | None,
    cached_source_fisher_receipt_sha256: str | None,
) -> str:
    """Hash the exact owning snapshots later consumed by the solve."""

    body = {
        "schema": ANALYTIC_COORDINATE_COMPLETION_EVIDENCE_SCHEMA,
        "trace_design_sha256": _tensor_sha256(design),
        "fitted_map_sha256": _tensor_sha256(fitted),
        "bridge_prior_sha256": None if bridge is None else _tensor_sha256(bridge),
        "parent_operator_sha256s": [value.operator_sha256 for value in operators],
        "cached_source_fisher_gram_sha256": (
            None if fisher is None else _tensor_sha256(fisher)
        ),
        "cached_source_fisher_receipt_sha256": (
            cached_source_fisher_receipt_sha256
        ),
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "trace_capture_calls": 0,
    }
    return canonical_sha256(body)


def _validate_coordinate_completion_snapshots(
    design: torch.Tensor,
    fitted: torch.Tensor,
    operators: Sequence[ResidualFacingOperator],
    *,
    bridge: torch.Tensor | None,
    fisher: torch.Tensor | None,
    cached_source_fisher_receipt_sha256: str | None,
) -> None:
    """Validate shape and receipt relations on the exact consumed snapshots."""

    target_width, source_width = map(int, fitted.shape)
    if int(design.shape[1]) != target_width:
        raise AnalyticCoordinateCompletionError(
            "trace_design width must match fitted_map target width."
        )
    if operators[0].source_width != source_width:
        raise AnalyticCoordinateCompletionError(
            "Parent operator width must match fitted_map source width."
        )
    if bridge is not None and tuple(bridge.shape) != tuple(fitted.shape):
        raise AnalyticCoordinateCompletionError(
            "bridge_prior must match fitted_map shape."
        )
    if fisher is None:
        if cached_source_fisher_receipt_sha256 is not None:
            raise AnalyticCoordinateCompletionError(
                "A Fisher receipt cannot be supplied without its cached Gram."
            )
    else:
        if tuple(fisher.shape) != (source_width, source_width):
            raise AnalyticCoordinateCompletionError(
                "cached_source_fisher_gram must be source-width square."
            )
        _digest(
            cached_source_fisher_receipt_sha256,
            "cached_source_fisher_receipt_sha256",
        )


def analytic_coordinate_completion_evidence_sha256(
    trace_design: torch.Tensor,
    fitted_map: torch.Tensor,
    parent_operators: Sequence[ResidualFacingOperator],
    *,
    bridge_prior: torch.Tensor | None = None,
    cached_source_fisher_gram: torch.Tensor | None = None,
    cached_source_fisher_receipt_sha256: str | None = None,
) -> str:
    """Seal all cached tensors consumed by one completion invocation."""

    design = _matrix(trace_design, "trace_design")
    fitted = _matrix(fitted_map, "fitted_map")
    operators = _operator_manifest(parent_operators)
    bridge = None if bridge_prior is None else _matrix(bridge_prior, "bridge_prior")
    fisher = (
        None
        if cached_source_fisher_gram is None
        else _matrix(cached_source_fisher_gram, "cached_source_fisher_gram")
    )
    _validate_coordinate_completion_snapshots(
        design,
        fitted,
        operators,
        bridge=bridge,
        fisher=fisher,
        cached_source_fisher_receipt_sha256=(
            cached_source_fisher_receipt_sha256
        ),
    )
    return _coordinate_completion_evidence_sha256_from_snapshots(
        design,
        fitted,
        operators,
        bridge=bridge,
        fisher=fisher,
        cached_source_fisher_receipt_sha256=(
            cached_source_fisher_receipt_sha256
        ),
    )


def _rank_from_singular_values(
    singular_values: torch.Tensor,
    *,
    rows: int,
    columns: int,
    relative_tolerance: float,
    absolute_tolerance: float = 0.0,
) -> tuple[int, float]:
    if singular_values.numel() == 0:
        return 0, 0.0
    maximum = float(singular_values.max().item())
    if maximum == 0.0:
        return 0, 0.0
    threshold = max(
        absolute_tolerance,
        maximum
        * max(
            relative_tolerance,
            torch.finfo(torch.float64).eps * max(rows, columns),
        ),
    )
    return int((singular_values > threshold).sum().item()), threshold


def _sign_canonicalize(vector: torch.Tensor) -> torch.Tensor:
    pivot = int(torch.argmax(vector.abs()).item())
    if float(vector[pivot].item()) < 0.0:
        return -vector
    return vector


def _orthogonal_residual(
    vector: torch.Tensor,
    accepted: Sequence[torch.Tensor],
) -> torch.Tensor:
    result = vector.clone()
    for _pass in range(2):
        for basis in accepted:
            result = result - basis * torch.dot(basis, result)
    return result


def _pivoted_projector_basis(
    projector: torch.Tensor,
    dimension: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Canonical orthogonal/Q-DEIM-style basis from coordinate pivots."""

    width = int(projector.shape[0])
    if dimension == 0:
        return torch.empty((width, 0), dtype=torch.float64), ()
    tolerance = max(1e-14, torch.finfo(torch.float64).eps * width * 32)
    accepted: list[torch.Tensor] = []
    pivots: list[int] = []
    remaining = set(range(width))
    while len(accepted) < dimension:
        scored: list[tuple[float, int, torch.Tensor]] = []
        for axis in sorted(remaining):
            candidate = _orthogonal_residual(projector[:, axis], accepted)
            norm = float(torch.linalg.vector_norm(candidate).item())
            scored.append((norm, axis, candidate))
        norm, axis, candidate = max(scored, key=lambda value: (value[0], -value[1]))
        if norm <= tolerance:
            raise AnalyticCoordinateCompletionError(
                "A numerical projector did not expose its declared dimension."
            )
        candidate = _sign_canonicalize(candidate / norm)
        accepted.append(candidate)
        pivots.append(axis)
        remaining.remove(axis)
    basis = torch.stack(accepted, dim=1).contiguous()
    return basis, tuple(pivots)


def _subspace_from_rows(
    rows: torch.Tensor,
    *,
    relative_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float, torch.Tensor]:
    _left, singular_values, right_h = torch.linalg.svd(rows, full_matrices=False)
    rank, threshold = _rank_from_singular_values(
        singular_values,
        rows=int(rows.shape[0]),
        columns=int(rows.shape[1]),
        relative_tolerance=relative_tolerance,
    )
    width = int(rows.shape[1])
    if rank:
        preliminary = right_h[:rank].T.contiguous()
        projector = ((preliminary @ preliminary.T) + (preliminary @ preliminary.T).T) * 0.5
        basis, _pivots = _pivoted_projector_basis(projector, rank)
        projector = (basis @ basis.T).contiguous()
    else:
        basis = torch.empty((width, 0), dtype=torch.float64)
        projector = torch.zeros((width, width), dtype=torch.float64)
    return basis, projector, rank, threshold, singular_values


def _canonical_eigenbasis(
    gram: torch.Tensor,
    count: int,
    *,
    degeneracy_relative_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[dict[str, Any], ...]]:
    if count == 0:
        width = int(gram.shape[0])
        return (
            torch.empty((width, 0), dtype=torch.float64),
            torch.empty((0,), dtype=torch.float64),
            (),
        )
    values, vectors = torch.linalg.eigh(gram)
    order = torch.arange(values.numel() - 1, -1, -1, dtype=torch.long)
    values = values.index_select(0, order)
    vectors = vectors.index_select(1, order)
    group_tolerance = max(
        1e-13,
        degeneracy_relative_tolerance * max(1.0, float(values.abs().max().item())),
    )
    groups: list[tuple[int, int]] = []
    start = 0
    while start < int(values.numel()):
        stop = start + 1
        while stop < int(values.numel()) and abs(
            float(values[stop].item() - values[start].item())
        ) <= group_tolerance:
            stop += 1
        groups.append((start, stop))
        start = stop

    accepted: list[torch.Tensor] = []
    selected_values: list[float] = []
    reports: list[dict[str, Any]] = []
    for start, stop in groups:
        if len(accepted) >= count:
            break
        group_vectors = vectors[:, start:stop]
        projector = (group_vectors @ group_vectors.T).contiguous()
        canonical, pivots = _pivoted_projector_basis(projector, stop - start)
        take = min(count - len(accepted), int(canonical.shape[1]))
        for index in range(take):
            candidate = _orthogonal_residual(canonical[:, index], accepted)
            candidate = _sign_canonicalize(
                candidate / torch.linalg.vector_norm(candidate)
            )
            accepted.append(candidate)
            selected_values.append(float(values[start + index].item()))
        reports.append(
            {
                "start": start,
                "stop": stop,
                "multiplicity": stop - start,
                "selected": take,
                "coordinate_pivots": list(pivots),
                "eigenvalue_max": float(values[start].item()),
                "eigenvalue_min": float(values[stop - 1].item()),
                "projector_sha256": _tensor_sha256(projector),
            }
        )
    if len(accepted) != count:
        raise AnalyticCoordinateCompletionError(
            "The source geometry did not yield the requested completion basis."
        )
    return (
        torch.stack(accepted, dim=1).contiguous(),
        torch.tensor(selected_values, dtype=torch.float64),
        tuple(reports),
    )


def _joint_parent_geometry(
    operators: Sequence[ResidualFacingOperator],
    *,
    cached_fisher: torch.Tensor | None,
    cached_fisher_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source_width = operators[0].source_width
    role_grams: dict[str, torch.Tensor] = {}
    operator_reports: list[dict[str, Any]] = []
    for operator in operators:
        contribution = _operator_gram(operator)
        role_grams.setdefault(
            operator.role,
            torch.zeros((source_width, source_width), dtype=torch.float64),
        ).add_(contribution)
        operator_reports.append(
            {
                **operator.as_dict(),
                "gram_sha256": _tensor_sha256(contribution),
                "gram_trace": float(torch.trace(contribution).item()),
            }
        )
    role_energies = {
        role: float(torch.trace(value).item()) for role, value in role_grams.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in role_energies.values()):
        raise AnalyticCoordinateCompletionError(
            "Every parent operator role must carry finite positive Gram energy."
        )
    roles = tuple(sorted(role_grams))
    structural = sum(
        (role_grams[role] / (len(roles) * role_energies[role]) for role in roles),
        torch.zeros((source_width, source_width), dtype=torch.float64),
    )
    structural = ((structural + structural.T) * 0.5).contiguous()
    raw = sum(
        (role_grams[role] for role in roles),
        torch.zeros((source_width, source_width), dtype=torch.float64),
    )
    raw = ((raw + raw.T) * 0.5).contiguous()
    if cached_fisher is None:
        geometry = structural.clone()
        fisher_report: dict[str, Any] = {
            "supplied": False,
            "weight": 0.0,
            "gram_sha256": None,
        }
    else:
        symmetry_scale = max(1.0, float(cached_fisher.abs().max().item()))
        symmetry_error = float((cached_fisher - cached_fisher.T).abs().max().item())
        if symmetry_error > 1e-10 * symmetry_scale:
            raise AnalyticCoordinateCompletionError(
                "cached_source_fisher_gram must be symmetric."
            )
        fisher = ((cached_fisher + cached_fisher.T) * 0.5).contiguous()
        eigenvalues = torch.linalg.eigvalsh(fisher)
        eigen_scale = max(1.0, float(eigenvalues.abs().max().item()))
        minimum = float(eigenvalues.min().item())
        if minimum < -1e-10 * eigen_scale:
            raise AnalyticCoordinateCompletionError(
                "cached_source_fisher_gram must be positive semidefinite."
            )
        shift = max(0.0, -minimum)
        if shift:
            fisher = fisher + torch.eye(source_width, dtype=torch.float64) * shift
        energy = float(torch.trace(fisher).item())
        if not math.isfinite(energy) or energy <= 0.0:
            raise AnalyticCoordinateCompletionError(
                "cached_source_fisher_gram must carry positive finite energy."
            )
        fisher_normalized = (fisher / energy).contiguous()
        geometry = (
            (1.0 - cached_fisher_weight) * structural
            + cached_fisher_weight * fisher_normalized
        ).contiguous()
        fisher_report = {
            "supplied": True,
            "weight": cached_fisher_weight,
            "structural_weight": 1.0 - cached_fisher_weight,
            "structural_geometry_cannot_be_replaced": True,
            "positive_semidefinite_shift": shift,
            "gram_sha256": _tensor_sha256(fisher),
            "normalized_gram_sha256": _tensor_sha256(fisher_normalized),
        }
    geometry = ((geometry + geometry.T) * 0.5).contiguous()
    report = {
        "source_width": source_width,
        "operator_count": len(operators),
        "role_count": len(roles),
        "roles": {
            role: {
                "raw_gram_trace": role_energies[role],
                "balanced_trace_contribution": 1.0 / len(roles),
                "gram_sha256": _tensor_sha256(role_grams[role]),
            }
            for role in roles
        },
        "operators": operator_reports,
        "role_frobenius_balancing": True,
        "raw_joint_gram_sha256": _tensor_sha256(raw),
        "balanced_structural_gram_sha256": _tensor_sha256(structural),
        "completion_geometry_sha256": _tensor_sha256(geometry),
        "cached_fisher": fisher_report,
    }
    return geometry, report


def _rectangular_polar_orientation(
    value: torch.Tensor,
    *,
    relative_tolerance: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Canonical rectangular polar factor, including a rank-deficient fill."""

    rows, columns = map(int, value.shape)
    if rows < columns:
        raise AnalyticCoordinateCompletionError(
            "Rectangular Procrustes orientation requires rows >= columns."
        )
    left, singular_values, right_h = torch.linalg.svd(value, full_matrices=False)
    rank, threshold = _rank_from_singular_values(
        singular_values,
        rows=rows,
        columns=columns,
        relative_tolerance=relative_tolerance,
    )
    if rank:
        left_identified = left[:, :rank]
        right_identified = right_h[:rank].T
        partial = left_identified @ right_identified.T
    else:
        left_identified = torch.empty((rows, 0), dtype=torch.float64)
        right_identified = torch.empty((columns, 0), dtype=torch.float64)
        partial = torch.zeros((rows, columns), dtype=torch.float64)
    missing = columns - rank
    if missing:
        left_projector = torch.eye(rows, dtype=torch.float64) - (
            left_identified @ left_identified.T
        )
        right_projector = torch.eye(columns, dtype=torch.float64) - (
            right_identified @ right_identified.T
        )
        left_fill, left_pivots = _pivoted_projector_basis(left_projector, missing)
        right_fill, right_pivots = _pivoted_projector_basis(right_projector, missing)
        orientation = partial + left_fill @ right_fill.T
    else:
        left_pivots = ()
        right_pivots = ()
        orientation = partial
    orientation = orientation.contiguous()
    identity = torch.eye(columns, dtype=torch.float64)
    if not torch.allclose(
        orientation.T @ orientation,
        identity,
        rtol=1e-11,
        atol=1e-12,
    ):
        raise AnalyticCoordinateCompletionError(
            "Canonical rectangular polar orientation is not isometric."
        )
    return orientation, {
        "input_rank": rank,
        "input_rank_threshold": threshold,
        "rank_deficient_nullspace_completed_canonically": missing > 0,
        "left_completion_pivots": list(left_pivots),
        "right_completion_pivots": list(right_pivots),
        "orientation_sha256": _tensor_sha256(orientation),
    }


def _reference_scale(
    identified_singular_values: torch.Tensor,
    identified_rank: int,
    bridge_prior: torch.Tensor | None,
    explicit: float | None,
) -> tuple[float, str]:
    if explicit is not None:
        return explicit, "explicit"
    if identified_rank:
        values = identified_singular_values[:identified_rank]
        return float(values.median().item()), "identified-map-median-singular-value"
    if bridge_prior is not None:
        scale = float(torch.linalg.vector_norm(bridge_prior).item()) / math.sqrt(
            min(map(int, bridge_prior.shape))
        )
        if scale > 0.0 and math.isfinite(scale):
            return scale, "bridge-prior-rms-singular-scale"
    return 1.0, "unit-structural-fallback"


def _candidate_receipt(
    *,
    recipe: str,
    candidate: torch.Tensor,
    update: torch.Tensor,
    identified_map: torch.Tensor,
    trace_design: torch.Tensor,
    fitted_map: torch.Tensor,
    identified_basis: torch.Tensor,
    target_basis: torch.Tensor,
    source_basis: torch.Tensor,
    scales: torch.Tensor,
    structural_rank: int,
    full_rank: int,
    evidence_sha256: str,
    recipe_details: Mapping[str, Any],
) -> dict[str, Any]:
    identified_delta = (
        identified_basis.T @ (candidate - fitted_map)
        if int(identified_basis.shape[1])
        else torch.empty((0, candidate.shape[1]), dtype=torch.float64)
    )
    prediction_delta = trace_design @ (candidate - fitted_map)
    source_orthogonality = identified_map @ source_basis
    target_orthogonality = identified_basis.T @ target_basis
    scale = max(
        1.0,
        float(fitted_map.abs().max().item()),
        float(candidate.abs().max().item()),
    )
    coordinate_roundoff_bound = (
        torch.finfo(torch.float64).eps
        * max(map(int, candidate.shape))
        * scale
        * 256
    )
    prediction_roundoff_bound = (
        coordinate_roundoff_bound
        * max(1.0, float(trace_design.abs().max().item()))
        * max(1, int(trace_design.shape[1]))
    )
    identified_error = (
        0.0 if identified_delta.numel() == 0 else float(identified_delta.abs().max().item())
    )
    prediction_error = float(prediction_delta.abs().max().item())
    if (
        identified_error > coordinate_roundoff_bound
        or prediction_error > prediction_roundoff_bound
    ):
        raise AnalyticCoordinateCompletionError(
            "A completion changed the trace-identified fitted subspace."
        )
    body: dict[str, Any] = {
        "schema": ANALYTIC_COORDINATE_COMPLETION_CANDIDATE_SCHEMA,
        "recipe": recipe,
        "evidence_sha256": evidence_sha256,
        "coordinate_map_shape": [int(value) for value in candidate.shape],
        "coordinate_map_sha256": _tensor_sha256(candidate),
        "nullspace_update_sha256": _tensor_sha256(update),
        "target_completion_basis_sha256": _tensor_sha256(target_basis),
        "source_completion_basis_sha256": _tensor_sha256(source_basis),
        "completion_scales": [float(value) for value in scales.tolist()],
        "completion_scales_sha256": _tensor_sha256(scales),
        "identified_map_sha256": _tensor_sha256(identified_map),
        "structural_rank": structural_rank,
        "materialized_rank": full_rank,
        "full_rank_materialized": full_rank == structural_rank,
        "identified_subspace_update_is_zero_by_construction": True,
        "fitted_subspace_preserved_to_fp64_roundoff": True,
        "trace_predictions_preserved_to_fp64_roundoff": True,
        "maximum_identified_coordinate_error": identified_error,
        "maximum_trace_prediction_error": prediction_error,
        "identified_coordinate_fp64_roundoff_bound": (
            coordinate_roundoff_bound
        ),
        "trace_prediction_fp64_roundoff_bound": prediction_roundoff_bound,
        "maximum_identified_to_completion_source_dot": (
            0.0
            if source_orthogonality.numel() == 0
            else float(source_orthogonality.abs().max().item())
        ),
        "maximum_identified_to_completion_target_dot": (
            0.0
            if target_orthogonality.numel() == 0
            else float(target_orthogonality.abs().max().item())
        ),
        "recipe_details": _deep_thaw(recipe_details),
        "numeric_diagnostics_are_navigation_only": True,
        "semantic_acceptance_decision": None,
        "quality_score_gate_applied": False,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "trace_capture_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "random_completion_used": False,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def compile_analytic_coordinate_completion_frontier(
    trace_design: torch.Tensor,
    fitted_map: torch.Tensor,
    parent_operators: Sequence[ResidualFacingOperator],
    *,
    expected_evidence_sha256: str,
    bridge_prior: torch.Tensor | None = None,
    cached_source_fisher_gram: torch.Tensor | None = None,
    cached_source_fisher_receipt_sha256: str | None = None,
    cached_source_fisher_weight: float = 0.25,
    reference_completion_scale: float | None = None,
    rank_relative_tolerance: float = 1e-12,
    degeneracy_relative_tolerance: float = 1e-11,
) -> AnalyticCoordinateCompletionFrontier:
    """Compile deterministic full-rank maps without changing cached trace fit.

    This is the narrow integration seam for a family-specific folding runner.
    ``trace_design`` is expressed in target coordinates and ``fitted_map`` has
    shape ``[target_width, source_width]``.  Every returned candidate can
    replace that decoder directly; all source consumer weights can then be
    folded as ``W_source @ candidate.T`` exactly as before.
    """

    expected_evidence_sha256 = _digest(
        expected_evidence_sha256,
        "expected_evidence_sha256",
    )
    fisher_weight = _unit_interval(
        cached_source_fisher_weight,
        "cached_source_fisher_weight",
        upper_open=True,
    )
    reference_completion_scale = _positive_float_or_none(
        reference_completion_scale,
        "reference_completion_scale",
    )
    rank_relative_tolerance = _unit_interval(
        rank_relative_tolerance,
        "rank_relative_tolerance",
        upper_open=True,
    )
    degeneracy_relative_tolerance = _unit_interval(
        degeneracy_relative_tolerance,
        "degeneracy_relative_tolerance",
        upper_open=True,
    )
    # Snapshot every caller-owned input once, then authenticate and consume
    # those same objects.  In particular, never hash one clone and solve from a
    # second clone: caller mutation between those operations is a TOCTOU hole.
    design = _matrix(trace_design, "trace_design")
    fitted = _matrix(fitted_map, "fitted_map")
    target_width, source_width = map(int, fitted.shape)
    operators = _operator_manifest(parent_operators)
    bridge = None if bridge_prior is None else _matrix(bridge_prior, "bridge_prior")
    fisher = (
        None
        if cached_source_fisher_gram is None
        else _matrix(cached_source_fisher_gram, "cached_source_fisher_gram")
    )
    _validate_coordinate_completion_snapshots(
        design,
        fitted,
        operators,
        bridge=bridge,
        fisher=fisher,
        cached_source_fisher_receipt_sha256=(
            cached_source_fisher_receipt_sha256
        ),
    )
    actual_evidence_sha256 = _coordinate_completion_evidence_sha256_from_snapshots(
        design,
        fitted,
        operators,
        bridge=bridge,
        fisher=fisher,
        cached_source_fisher_receipt_sha256=(
            cached_source_fisher_receipt_sha256
        ),
    )
    if actual_evidence_sha256 != expected_evidence_sha256:
        raise AnalyticCoordinateCompletionError(
            "Cached coordinate-completion evidence authentication failed."
        )

    with _NUMERICAL_LOCK:
        # Thread count is process-global.  Capture it only after taking the
        # same lock that guards the temporary single-thread setting; otherwise
        # a waiter can observe ``1`` and restore that stale value permanently.
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.no_grad():
                (
                    identified_target_basis,
                    identified_target_projector,
                    trace_rank,
                    trace_rank_threshold,
                    trace_singular_values,
                ) = _subspace_from_rows(
                    design,
                    relative_tolerance=rank_relative_tolerance,
                )
                identified_map = (
                    identified_target_projector @ fitted
                ).contiguous()
                (
                    _identified_left,
                    identified_singular_values,
                    identified_right_h,
                ) = torch.linalg.svd(identified_map, full_matrices=False)
                fitted_norm = float(torch.linalg.vector_norm(fitted).item())
                if not math.isfinite(fitted_norm):
                    raise AnalyticCoordinateCompletionError(
                        "fitted_map norm overflowed the FP64 completion solve."
                    )
                # Projecting an arbitrary fitted null component through a
                # numerically constructed row-space projector leaves O(eps)
                # residue.  A purely relative rank test would promote any such
                # nonzero dust to rank one (and could then call an almost-zero
                # map 'full rank').  Anchor the identified rank to a forward
                # roundoff floor of the original fit as well.
                identified_projection_roundoff_floor = (
                    torch.finfo(torch.float64).eps
                    * max(1, target_width, trace_rank)
                    * fitted_norm
                    * 32
                )
                identified_map_rank, identified_map_rank_threshold = (
                    _rank_from_singular_values(
                        identified_singular_values,
                        rows=target_width,
                        columns=source_width,
                        relative_tolerance=rank_relative_tolerance,
                        absolute_tolerance=(
                            identified_projection_roundoff_floor
                        ),
                    )
                )
                if identified_map_rank:
                    occupied_preliminary = (
                        identified_right_h[:identified_map_rank].T.contiguous()
                    )
                    occupied_projector = (
                        occupied_preliminary @ occupied_preliminary.T
                    ).contiguous()
                    occupied_source_basis, occupied_source_pivots = (
                        _pivoted_projector_basis(
                            occupied_projector,
                            identified_map_rank,
                        )
                    )
                    occupied_projector = (
                        occupied_source_basis @ occupied_source_basis.T
                    ).contiguous()
                else:
                    occupied_source_basis = torch.empty(
                        (source_width, 0), dtype=torch.float64
                    )
                    occupied_source_pivots = ()
                    occupied_projector = torch.zeros(
                        (source_width, source_width), dtype=torch.float64
                    )

                target_null_dimension = target_width - trace_rank
                structural_rank = min(target_width, source_width)
                completion_rank = structural_rank - identified_map_rank
                if completion_rank > target_null_dimension:
                    rank_report = {
                        "schema": ANALYTIC_COORDINATE_COMPLETION_FRONTIER_SCHEMA,
                        "status": "full-rank-impossible-with-exact-fit-preservation",
                        "target_width": target_width,
                        "source_width": source_width,
                        "trace_identified_rank": trace_rank,
                        "identified_map_rank": identified_map_rank,
                        "identified_map_rank_threshold": (
                            identified_map_rank_threshold
                        ),
                        "identified_map_projection_roundoff_floor": (
                            identified_projection_roundoff_floor
                        ),
                        "target_null_dimension": target_null_dimension,
                        "required_completion_rank": completion_rank,
                        "maximum_achievable_rank": (
                            identified_map_rank + target_null_dimension
                        ),
                        "required_full_rank": structural_rank,
                        "fitted_subspace_mutation_permitted": False,
                        "quality_score_gate_applied": False,
                        "result_sha256": "",
                    }
                    rank_report["result_sha256"] = canonical_sha256(
                        {key: value for key, value in rank_report.items() if key != "result_sha256"}
                    )
                    raise AnalyticCoordinateCompletionRankError(rank_report)

                target_null_projector = (
                    torch.eye(target_width, dtype=torch.float64)
                    - identified_target_projector
                ).contiguous()
                target_null_basis, target_null_pivots = _pivoted_projector_basis(
                    target_null_projector,
                    target_null_dimension,
                )
                source_null_dimension = source_width - identified_map_rank
                source_null_projector = (
                    torch.eye(source_width, dtype=torch.float64)
                    - occupied_projector
                ).contiguous()
                source_null_basis, source_null_pivots = _pivoted_projector_basis(
                    source_null_projector,
                    source_null_dimension,
                )

                parent_geometry, geometry_report = _joint_parent_geometry(
                    operators,
                    cached_fisher=fisher,
                    cached_fisher_weight=fisher_weight if fisher is not None else 0.0,
                )
                # Operator snapshots can dominate memory for a many-role
                # parent.  Their authenticated Gram/report is complete now;
                # release the owning tensor copies before the source-square
                # eigensolve and dense candidate materializations overlap.
                del operators, fisher
                reduced_geometry = (
                    source_null_basis.T @ parent_geometry @ source_null_basis
                ).contiguous()
                reduced_geometry = (
                    (reduced_geometry + reduced_geometry.T) * 0.5
                ).contiguous()
                reduced_completion_basis, selected_geometry_values, eigen_report = (
                    _canonical_eigenbasis(
                        reduced_geometry,
                        completion_rank,
                        degeneracy_relative_tolerance=(
                            degeneracy_relative_tolerance
                        ),
                    )
                )
                completion_source_basis = (
                    source_null_basis @ reduced_completion_basis
                ).contiguous()
                base_target_basis = target_null_basis[:, :completion_rank].contiguous()
                scale, scale_source = _reference_scale(
                    identified_singular_values,
                    identified_map_rank,
                    bridge,
                    reference_completion_scale,
                )
                if not math.isfinite(scale) or scale <= 0.0:
                    raise AnalyticCoordinateCompletionError(
                        "The analytic completion reference scale is invalid."
                    )

                recipes: list[
                    tuple[str, torch.Tensor, torch.Tensor, dict[str, Any]]
                ] = []
                isometric_scales = torch.full(
                    (completion_rank,), scale, dtype=torch.float64
                )
                if completion_rank == 0:
                    recipes.append(
                        (
                            "identified-map-already-full-rank",
                            base_target_basis,
                            isometric_scales,
                            {
                                "target_orientation": "no-nullspace-update",
                                "source_orientation": "no-nullspace-update",
                            },
                        )
                    )
                else:
                    recipes.append(
                        (
                            "operator-sensitive-isometric",
                            base_target_basis,
                            isometric_scales,
                            {
                                "target_orientation": (
                                    "orthogonal-coordinate-pivot-qdeim-null-basis"
                                ),
                                "source_orientation": (
                                    "top-joint-parent-operator-geometry"
                                ),
                            },
                        )
                    )
                    reflected_basis = base_target_basis.clone()
                    reflected_basis[:, -1].neg_()
                    recipes.append(
                        (
                            "operator-sensitive-reflected",
                            reflected_basis,
                            isometric_scales,
                            {
                                "target_orientation": (
                                    "deterministic-last-axis-reflection"
                                ),
                                "source_orientation": (
                                    "top-joint-parent-operator-geometry"
                                ),
                                "navigation_alternative_not_quality_ranking": (
                                    True
                                ),
                            },
                        )
                    )

                positive_geometry = selected_geometry_values.clamp_min(0.0)
                positive = positive_geometry[positive_geometry > 0.0]
                if positive.numel():
                    reference_geometry = float(positive.median().item())
                    floor = max(
                        torch.finfo(torch.float64).tiny,
                        reference_geometry * 1e-12,
                    )
                    balanced_scales = scale * torch.sqrt(
                        torch.tensor(reference_geometry, dtype=torch.float64)
                        / positive_geometry.clamp_min(floor)
                    )
                    recipes.append(
                        (
                            "operator-fisher-balanced",
                            base_target_basis,
                            balanced_scales.contiguous(),
                            {
                                "target_orientation": (
                                    "orthogonal-coordinate-pivot-qdeim-null-basis"
                                ),
                                "source_orientation": (
                                    "top-joint-parent-operator-geometry"
                                ),
                                "scale_rule": (
                                    "equalized-completion-geometry-response"
                                ),
                                "geometry_reference": reference_geometry,
                            },
                        )
                    )

                if bridge is not None and completion_rank:
                    bridge_coordinates = (
                        target_null_basis.T @ bridge @ completion_source_basis
                    ).contiguous()
                    orientation, procrustes_report = (
                        _rectangular_polar_orientation(
                            bridge_coordinates,
                            relative_tolerance=rank_relative_tolerance,
                        )
                    )
                    procrustes_target_basis = (
                        target_null_basis @ orientation
                    ).contiguous()
                    recipes.append(
                        (
                            "bridge-procrustes-isometric",
                            procrustes_target_basis,
                            isometric_scales,
                            {
                                "target_orientation": (
                                    "bridge-prior-rectangular-polar-factor"
                                ),
                                "source_orientation": (
                                    "top-joint-parent-operator-geometry"
                                ),
                                "procrustes": procrustes_report,
                            },
                        )
                    )

                candidates: list[AnalyticCoordinateCompletionCandidate] = []
                duplicate_recipes: list[dict[str, str]] = []
                rejected_optional_recipes: list[dict[str, Any]] = []
                map_hash_to_recipe: dict[str, str] = {}
                for recipe, target_basis, scales, details in recipes:
                    update = (
                        (target_basis * scales.unsqueeze(0))
                        @ completion_source_basis.T
                    ).contiguous()
                    if trace_rank:
                        update = (
                            update
                            - identified_target_basis
                            @ (identified_target_basis.T @ update)
                        ).contiguous()
                    candidate = (identified_map + update).contiguous()
                    candidate_singular_values = torch.linalg.svdvals(candidate)
                    candidate_rank, candidate_rank_threshold = (
                        _rank_from_singular_values(
                            candidate_singular_values,
                            rows=target_width,
                            columns=source_width,
                            relative_tolerance=rank_relative_tolerance,
                        )
                    )
                    if candidate_rank != structural_rank:
                        if recipe in {
                            "identified-map-already-full-rank",
                            "operator-sensitive-isometric",
                            "operator-sensitive-reflected",
                        }:
                            raise AnalyticCoordinateCompletionError(
                                "A required analytic completion did not "
                                "materialize full rank."
                            )
                        rejected_optional_recipes.append(
                            {
                                "recipe": recipe,
                                "reason": (
                                    "optional-scale-or-orientation-lost-"
                                    "numerical-full-rank"
                                ),
                                "coordinate_map_sha256": _tensor_sha256(
                                    candidate
                                ),
                                "materialized_rank": candidate_rank,
                                "required_full_rank": structural_rank,
                                "rank_threshold": candidate_rank_threshold,
                                "quality_rejection": False,
                            }
                        )
                        continue
                    receipt = _candidate_receipt(
                        recipe=recipe,
                        candidate=candidate,
                        update=update,
                        identified_map=identified_map,
                        trace_design=design,
                        fitted_map=fitted,
                        identified_basis=identified_target_basis,
                        target_basis=target_basis,
                        source_basis=completion_source_basis,
                        scales=scales,
                        structural_rank=structural_rank,
                        full_rank=candidate_rank,
                        evidence_sha256=actual_evidence_sha256,
                        recipe_details=details,
                    )
                    map_sha256 = receipt["coordinate_map_sha256"]
                    if map_sha256 in map_hash_to_recipe:
                        duplicate_recipes.append(
                            {
                                "recipe": recipe,
                                "duplicates": map_hash_to_recipe[map_sha256],
                            }
                        )
                        continue
                    map_hash_to_recipe[map_sha256] = recipe
                    candidates.append(
                        AnalyticCoordinateCompletionCandidate(
                            recipe=recipe,
                            coordinate_map=candidate,
                            nullspace_update=update,
                            report=receipt,
                        )
                    )
                minimum_candidate_count = 2 if completion_rank else 1
                if len(candidates) < minimum_candidate_count:
                    raise AnalyticCoordinateCompletionError(
                        "The completion frontier is not functionally distinct."
                    )

                body: dict[str, Any] = {
                    "schema": ANALYTIC_COORDINATE_COMPLETION_FRONTIER_SCHEMA,
                    "status": "analytic-full-rank-completion-materialized",
                    "evidence_sha256": actual_evidence_sha256,
                    "target_width": target_width,
                    "source_width": source_width,
                    "required_full_rank": structural_rank,
                    "trace_row_count": int(design.shape[0]),
                    "trace_identified_rank": trace_rank,
                    "trace_rank_threshold": trace_rank_threshold,
                    "trace_singular_values": [
                        float(value) for value in trace_singular_values.tolist()
                    ],
                    "identified_map_rank": identified_map_rank,
                    "identified_map_rank_threshold": (
                        identified_map_rank_threshold
                    ),
                    "identified_map_projection_roundoff_floor": (
                        identified_projection_roundoff_floor
                    ),
                    "target_null_dimension": target_null_dimension,
                    "source_null_dimension": source_null_dimension,
                    "completion_rank": completion_rank,
                    "identified_target_coordinate_pivots": (
                        list(
                            _pivoted_projector_basis(
                                identified_target_projector,
                                trace_rank,
                            )[1]
                        )
                    ),
                    "target_null_coordinate_pivots": list(target_null_pivots),
                    "occupied_source_coordinate_pivots": list(
                        occupied_source_pivots
                    ),
                    "source_null_coordinate_pivots": list(source_null_pivots),
                    "parent_geometry": geometry_report,
                    "completion_geometry": {
                        "reduced_gram_sha256": _tensor_sha256(
                            reduced_geometry
                        ),
                        "selected_eigenvalues": [
                            float(value)
                            for value in selected_geometry_values.tolist()
                        ],
                        "degenerate_eigenspaces": list(eigen_report),
                        "completion_source_basis_sha256": _tensor_sha256(
                            completion_source_basis
                        ),
                    },
                    "reference_completion_scale": scale,
                    "reference_completion_scale_source": scale_source,
                    "identified_map_sha256": _tensor_sha256(identified_map),
                    "identified_target_basis_sha256": _tensor_sha256(
                        identified_target_basis
                    ),
                    "target_null_basis_sha256": _tensor_sha256(
                        target_null_basis
                    ),
                    "occupied_source_basis_sha256": _tensor_sha256(
                        occupied_source_basis
                    ),
                    "candidate_count": len(candidates),
                    "candidate_recipes": [value.recipe for value in candidates],
                    "candidate_result_sha256s": [
                        value.report["result_sha256"] for value in candidates
                    ],
                    "duplicate_recipe_materializations": duplicate_recipes,
                    "rejected_optional_recipe_materializations": (
                        rejected_optional_recipes
                    ),
                    "candidate_order_is_quality_ranking": False,
                    "numeric_diagnostics_are_navigation_only": True,
                    "semantic_acceptance_requires_decoded_behavior_review": True,
                    "quality_score_gate_applied": False,
                    "completion_algorithms": [
                        "trace-row-space-orthogonal-decomposition",
                        "role-balanced-joint-parent-operator-gram",
                        "optional-cached-parent-fisher-gram",
                        "orthogonal-coordinate-pivot-qdeim-null-basis",
                        "source-complement-geometry-eigendecomposition",
                        "optional-bridge-rectangular-procrustes",
                    ],
                    "evidence_policy": {
                        "cached_trace_and_fit_only": True,
                        "cached_parent_operator_tensors_only": True,
                        "cached_fisher_only_when_supplied": True,
                        "authenticated_bridge_asset_only_when_supplied": True,
                        "live_parent_access_permitted": False,
                        "parent_trace_recapture_permitted": False,
                    },
                    "identified_subspace_preserved_exactly_by_decomposition": True,
                    "unidentifiable_complement_is_only_modified_subspace": True,
                    "model_forward_calls": 0,
                    "model_generate_calls": 0,
                    "trace_capture_calls": 0,
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "autograd_used": False,
                    "pruning_used": False,
                    "quantization_used": False,
                    "random_completion_used": False,
                    "filesystem_writes": 0,
                }
                report = {**body, "result_sha256": canonical_sha256(body)}
                return AnalyticCoordinateCompletionFrontier(
                    identified_map=identified_map,
                    identified_target_basis=identified_target_basis,
                    target_null_basis=target_null_basis,
                    occupied_source_basis=occupied_source_basis,
                    completion_source_basis=completion_source_basis,
                    candidates=tuple(candidates),
                    report=report,
                )
        finally:
            torch.set_num_threads(original_threads)


__all__ = [
    "ANALYTIC_COORDINATE_COMPLETION_CANDIDATE_SCHEMA",
    "ANALYTIC_COORDINATE_COMPLETION_EVIDENCE_SCHEMA",
    "ANALYTIC_COORDINATE_COMPLETION_FRONTIER_SCHEMA",
    "AnalyticCoordinateCompletionCandidate",
    "AnalyticCoordinateCompletionError",
    "AnalyticCoordinateCompletionFrontier",
    "AnalyticCoordinateCompletionRankError",
    "analytic_coordinate_completion_evidence_sha256",
    "compile_analytic_coordinate_completion_frontier",
]
