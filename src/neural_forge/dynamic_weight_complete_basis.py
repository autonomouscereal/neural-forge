"""Prompt-independent, weight-complete residual coordinate bases.

Residual-facing operators contribute source-width Gram energy directly:

* consumer/input orientation: ``W.T @ W``;
* producer/output orientation: ``W @ W.T``; and
* diagonal operators: ``diag(d ** 2)``.

Rows from unrelated operators are never concatenated.  Operators are grouped
by semantic role, each role is normalized by its total Frobenius energy, and
the normalized roles are averaged.  This prevents a vocabulary-sized
embedding or another unusually large matrix from dominating the coordinate
basis merely because it has more rows.

An optional externally supplied activation Gram may be blended at a strictly
bounded weight.  The structural weight objective always retains coefficient
``1 - activation_weight`` and is evaluated independently in the report.

The eigensolve is performed on CPU float64 with one numerical thread under a
process-local lock.  Degenerate eigenspaces are canonicalized from their
projectors against ordered source axes, and every vector receives a stable
sign.  The resulting basis is deterministic across caller thread settings.
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


BASIS_SCHEMA = "neural-forge-dynamic-weight-complete-basis-v1"
OPERATOR_SCHEMA = "neural-forge-residual-facing-operator-v1"
_NUMERICAL_LOCK = threading.RLock()


class WeightCompleteBasisError(ValueError):
    """Operator geometry, numerical state, or basis construction is invalid."""


class WeightCompleteBasisCoverageError(WeightCompleteBasisError):
    """The requested basis failed a rank, retention, or source-coverage gate."""


class FeatureOrientation(str, Enum):
    """Which side of an operator is expressed in residual coordinates."""

    INPUT = "feature_axis_input"
    OUTPUT = "feature_axis_output"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WeightCompleteBasisError(f"{name} must be a nonempty string.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WeightCompleteBasisError(f"{name} must be a positive integer.")
    return value


def _unit_interval(value: Any, name: str, *, upper_open: bool = False) -> float:
    if isinstance(value, bool):
        raise WeightCompleteBasisError(f"{name} must be finite within [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WeightCompleteBasisError(
            f"{name} must be finite within [0, 1]."
        ) from exc
    upper_ok = result < 1.0 if upper_open else result <= 1.0
    if not math.isfinite(result) or result < 0.0 or not upper_ok:
        boundary = "[0, 1)" if upper_open else "[0, 1]"
        raise WeightCompleteBasisError(f"{name} must be within {boundary}.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise WeightCompleteBasisError(
            f"{name} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WeightCompleteBasisError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise WeightCompleteBasisError(
            f"{name} must be finite and non-negative."
        )
    return result


def _finite_dense_tensor(value: Any, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.ndim < 1
        or not bool(torch.isfinite(value.detach()).all())
    ):
        raise WeightCompleteBasisError(
            f"{name} must be a finite dense real floating tensor."
        )
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = _finite_dense_tensor(value, "hashed tensor")
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


@dataclass(frozen=True, slots=True)
class ResidualFacingOperator:
    """One immutable role-tagged operator with one residual feature axis."""

    role: str
    name: str
    tensor: torch.Tensor
    orientation: FeatureOrientation
    feature_axis: int | None = None
    gram_weight: float = 1.0
    diagonal: bool = False
    operator_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        role = _text(self.role, "role")
        name = _text(self.name, "name")
        tensor = _finite_dense_tensor(self.tensor, f"operator {name!r}")
        if not isinstance(self.orientation, FeatureOrientation):
            raise WeightCompleteBasisError(
                "orientation must be FeatureOrientation.INPUT or OUTPUT."
            )
        if not isinstance(self.diagonal, bool):
            raise WeightCompleteBasisError("diagonal must be boolean.")
        if self.diagonal:
            if tensor.ndim != 1:
                raise WeightCompleteBasisError(
                    "A diagonal residual operator must be a vector."
                )
            feature_axis = 0
        else:
            if tensor.ndim < 2:
                raise WeightCompleteBasisError(
                    "A non-diagonal residual operator needs rank at least two."
                )
            default_axis = (
                tensor.ndim - 1
                if self.orientation is FeatureOrientation.INPUT
                else 0
            )
            feature_axis = (
                default_axis if self.feature_axis is None else self.feature_axis
            )
            if (
                isinstance(feature_axis, bool)
                or not isinstance(feature_axis, int)
                or not -tensor.ndim <= feature_axis < tensor.ndim
            ):
                raise WeightCompleteBasisError(
                    "feature_axis is outside the operator tensor rank."
                )
            feature_axis %= tensor.ndim
        gram_weight = _nonnegative_float(self.gram_weight, "gram_weight")
        if gram_weight == 0.0:
            raise WeightCompleteBasisError(
                "gram_weight must be strictly positive."
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "feature_axis", feature_axis)
        object.__setattr__(self, "gram_weight", gram_weight)
        object.__setattr__(
            self,
            "operator_sha256",
            hashlib.sha256(
                json.dumps(
                    {
                        "schema": OPERATOR_SCHEMA,
                        "role": role,
                        "name": name,
                        "tensor_sha256": _tensor_sha256(tensor),
                        "orientation": self.orientation.value,
                        "feature_axis": feature_axis,
                        "gram_weight": gram_weight,
                        "diagonal": self.diagonal,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest(),
        )

    @property
    def source_width(self) -> int:
        return int(self.tensor.shape[self.feature_axis])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OPERATOR_SCHEMA,
            "role": self.role,
            "name": self.name,
            "orientation": self.orientation.value,
            "feature_axis": self.feature_axis,
            "diagonal": self.diagonal,
            "gram_weight": self.gram_weight,
            "source_width": self.source_width,
            "tensor_shape": [int(value) for value in self.tensor.shape],
            "tensor_sha256": _tensor_sha256(self.tensor),
            "operator_sha256": self.operator_sha256,
        }


def residual_facing_operator(
    tensor: torch.Tensor,
    *,
    role: str,
    name: str,
    orientation: FeatureOrientation,
    feature_axis: int | None = None,
    gram_weight: float = 1.0,
) -> ResidualFacingOperator:
    """Create one generic dense residual-facing operator descriptor."""

    return ResidualFacingOperator(
        role=role,
        name=name,
        tensor=tensor,
        orientation=orientation,
        feature_axis=feature_axis,
        gram_weight=gram_weight,
    )


def qkv_input_sensitivity_operators(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    role: str = "qkv_input_sensitivity",
) -> tuple[ResidualFacingOperator, ...]:
    """Compose Q/K/V consumer sensitivity as three ``W.T @ W`` terms."""

    return tuple(
        ResidualFacingOperator(
            role=role,
            name=name,
            tensor=value,
            orientation=FeatureOrientation.INPUT,
            feature_axis=value.ndim - 1,
        )
        for name, value in (
            ("query", query_weight),
            ("key", key_weight),
            ("value", value_weight),
        )
    )


def output_mlp_span_operators(
    output_projection: torch.Tensor,
    mlp_down_projection: torch.Tensor,
    *,
    output_role: str = "operator_output_span",
    mlp_role: str = "mlp_output_span",
) -> tuple[ResidualFacingOperator, ...]:
    """Compose output and MLP producer spans as ``W @ W.T`` terms."""

    return (
        ResidualFacingOperator(
            role=output_role,
            name="output_projection",
            tensor=output_projection,
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=0,
        ),
        ResidualFacingOperator(
            role=mlp_role,
            name="mlp_down_projection",
            tensor=mlp_down_projection,
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=0,
        ),
    )


def embedding_head_operators(
    embedding_weight: torch.Tensor,
    output_head_weight: torch.Tensor | None = None,
    *,
    embedding_role: str = "embedding_interface",
    head_role: str = "output_head_interface",
) -> tuple[ResidualFacingOperator, ...]:
    """Compose embedding/head residual sensitivity without vocabulary rows."""

    values = [
        ResidualFacingOperator(
            role=embedding_role,
            name="embedding",
            tensor=embedding_weight,
            orientation=FeatureOrientation.INPUT,
            feature_axis=embedding_weight.ndim - 1,
        )
    ]
    if output_head_weight is not None:
        values.append(
            ResidualFacingOperator(
                role=head_role,
                name="output_head",
                tensor=output_head_weight,
                orientation=FeatureOrientation.INPUT,
                feature_axis=output_head_weight.ndim - 1,
            )
        )
    return tuple(values)


def norm_operators(
    weights: Sequence[torch.Tensor] | torch.Tensor,
    *,
    role: str = "normalization",
    names: Sequence[str] | None = None,
) -> tuple[ResidualFacingOperator, ...]:
    """Compose diagonal normalization sensitivity as ``diag(d ** 2)``."""

    values = (weights,) if isinstance(weights, torch.Tensor) else tuple(weights)
    if not values:
        raise WeightCompleteBasisError("At least one norm weight is required.")
    actual_names = (
        tuple(f"norm_{index}" for index in range(len(values)))
        if names is None
        else tuple(names)
    )
    if len(actual_names) != len(values):
        raise WeightCompleteBasisError(
            "Norm names must match the number of norm weights."
        )
    return tuple(
        ResidualFacingOperator(
            role=role,
            name=_text(name, "norm name"),
            tensor=value,
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=0,
            diagonal=True,
        )
        for name, value in zip(actual_names, values)
    )


def conv_state_operators(
    convolution_operators: Sequence[torch.Tensor] | torch.Tensor,
    state_operators: Sequence[torch.Tensor] | torch.Tensor = (),
    *,
    convolution_feature_axis: int = 0,
    state_feature_axis: int = 0,
    state_orientation: FeatureOrientation = FeatureOrientation.OUTPUT,
    convolution_role: str = "convolution_output_span",
    state_role: str = "state_operator_span",
) -> tuple[ResidualFacingOperator, ...]:
    """Compose convolution kernels and recurrent/state residual operators."""

    convolutions = (
        (convolution_operators,)
        if isinstance(convolution_operators, torch.Tensor)
        else tuple(convolution_operators)
    )
    states = (
        (state_operators,)
        if isinstance(state_operators, torch.Tensor)
        else tuple(state_operators)
    )
    if not convolutions and not states:
        raise WeightCompleteBasisError(
            "At least one convolution or state operator is required."
        )
    result = [
        ResidualFacingOperator(
            role=convolution_role,
            name=f"convolution_{index}",
            tensor=value,
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=convolution_feature_axis,
        )
        for index, value in enumerate(convolutions)
    ]
    result.extend(
        ResidualFacingOperator(
            role=state_role,
            name=f"state_{index}",
            tensor=value,
            orientation=state_orientation,
            feature_axis=state_feature_axis,
        )
        for index, value in enumerate(states)
    )
    return tuple(result)


def _operator_gram(value: ResidualFacingOperator) -> torch.Tensor:
    tensor = value.tensor
    width = value.source_width
    if value.diagonal:
        gram = torch.diag(tensor.square())
    elif value.orientation is FeatureOrientation.INPUT:
        rows = tensor.movedim(value.feature_axis, -1).reshape(-1, width)
        gram = rows.T @ rows
    else:
        rows = tensor.movedim(value.feature_axis, 0).reshape(width, -1)
        gram = rows @ rows.T
    gram = gram * value.gram_weight
    return ((gram + gram.T) * 0.5).contiguous()


def _rank(
    eigenvalues: torch.Tensor,
    *,
    width: int,
    tolerance: float,
) -> int:
    maximum = float(eigenvalues.abs().max().item())
    if maximum == 0.0:
        return 0
    threshold = maximum * max(
        tolerance,
        torch.finfo(torch.float64).eps * width,
    )
    return int((eigenvalues > threshold).sum().item())


def _sign_canonicalize(vector: torch.Tensor) -> torch.Tensor:
    absolute = vector.abs()
    pivot = int(torch.argmax(absolute).item())
    if float(vector[pivot].item()) < 0.0:
        vector = -vector
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


def _canonical_projector_basis(
    projector: torch.Tensor,
    dimension: int,
    *,
    tolerance: float,
    fallbacks: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    width = int(projector.shape[0])
    accepted: list[torch.Tensor] = []
    for axis in range(width):
        candidate = _orthogonal_residual(projector[:, axis], accepted)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= tolerance:
            continue
        candidate = _sign_canonicalize(candidate / norm)
        accepted.append(candidate)
        if len(accepted) == dimension:
            break
    if len(accepted) < dimension:
        fallback_values = sorted(
            (
                int(torch.argmax(fallbacks[:, index].abs()).item()),
                index,
                _sign_canonicalize(fallbacks[:, index].clone()),
            )
            for index in range(int(fallbacks.shape[1]))
        )
        for _pivot, _index, value in fallback_values:
            candidate = _orthogonal_residual(value, accepted)
            norm = float(torch.linalg.vector_norm(candidate).item())
            if norm <= tolerance:
                continue
            accepted.append(_sign_canonicalize(candidate / norm))
            if len(accepted) == dimension:
                break
    if len(accepted) != dimension:
        raise WeightCompleteBasisError(
            "Could not canonicalize a degenerate eigenspace."
        )
    return tuple(accepted)


def _canonical_top_k(
    gram: torch.Tensor,
    target_width: int,
    *,
    degeneracy_relative_tolerance: float,
    degeneracy_absolute_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[dict[str, Any], ...]]:
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.arange(
        int(eigenvalues.numel()) - 1,
        -1,
        -1,
        dtype=torch.long,
    )
    values = eigenvalues.index_select(0, order)
    vectors = eigenvectors.index_select(1, order)
    maximum = max(1.0, float(values.abs().max().item()))
    group_tolerance = max(
        degeneracy_absolute_tolerance,
        degeneracy_relative_tolerance * maximum,
    )
    vector_tolerance = max(
        1e-14,
        torch.finfo(torch.float64).eps * int(gram.shape[0]) * 16,
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

    selected: list[torch.Tensor] = []
    selected_values: list[float] = []
    group_reports: list[dict[str, Any]] = []
    for start, stop in groups:
        if len(selected) >= target_width:
            break
        group_vectors = vectors[:, start:stop]
        projector = (group_vectors @ group_vectors.T).contiguous()
        canonical = _canonical_projector_basis(
            projector,
            stop - start,
            tolerance=vector_tolerance,
            fallbacks=group_vectors,
        )
        take = min(target_width - len(selected), len(canonical))
        for index in range(take):
            candidate = _orthogonal_residual(canonical[index], selected)
            norm = float(torch.linalg.vector_norm(candidate).item())
            if norm <= vector_tolerance:
                raise WeightCompleteBasisError(
                    "Canonical eigenspaces lost global orthogonality."
                )
            selected.append(_sign_canonicalize(candidate / norm))
            selected_values.append(float(values[start + index].item()))
        group_reports.append(
            {
                "start": start,
                "stop": stop,
                "multiplicity": stop - start,
                "selected": take,
                "eigenvalue_max": float(values[start].item()),
                "eigenvalue_min": float(values[stop - 1].item()),
                "projector_sha256": _tensor_sha256(projector),
                "canonicalized_from_projector": True,
            }
        )
    if len(selected) != target_width:
        raise WeightCompleteBasisError(
            "Eigensolver did not return the requested target width."
        )
    basis = torch.stack(selected, dim=1).contiguous()
    selected_eigenvalues = torch.tensor(
        selected_values,
        dtype=torch.float64,
    )
    identity = torch.eye(target_width, dtype=torch.float64)
    if not torch.allclose(
        basis.T @ basis,
        identity,
        rtol=1e-11,
        atol=1e-12,
    ):
        raise WeightCompleteBasisError(
            "Canonical top-k basis is not orthonormal."
        )
    return basis, selected_eigenvalues, tuple(group_reports)


@dataclass(frozen=True, slots=True)
class WeightCompleteBasisResult:
    """Materialized basis, objectives, and auditable coverage report."""

    basis: torch.Tensor
    full_weight_gram: torch.Tensor
    balanced_structural_gram: torch.Tensor
    combined_gram: torch.Tensor
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        basis = _finite_dense_tensor(self.basis, "basis")
        full_weight = _finite_dense_tensor(
            self.full_weight_gram, "full_weight_gram"
        )
        balanced = _finite_dense_tensor(
            self.balanced_structural_gram,
            "balanced_structural_gram",
        )
        combined = _finite_dense_tensor(self.combined_gram, "combined_gram")
        source_width, target_width = map(int, basis.shape)
        if (
            basis.ndim != 2
            or tuple(full_weight.shape) != (source_width, source_width)
            or tuple(balanced.shape) != tuple(full_weight.shape)
            or tuple(combined.shape) != tuple(full_weight.shape)
            or not torch.allclose(
                basis.T @ basis,
                torch.eye(target_width, dtype=torch.float64),
                rtol=1e-11,
                atol=1e-12,
            )
            or not isinstance(self.report, Mapping)
        ):
            raise WeightCompleteBasisError(
                "WeightCompleteBasisResult invariants are invalid."
            )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "full_weight_gram", full_weight)
        object.__setattr__(self, "balanced_structural_gram", balanced)
        object.__setattr__(self, "combined_gram", combined)
        object.__setattr__(
            self, "report", MappingProxyType(dict(self.report))
        )

    @property
    def source_width(self) -> int:
        return int(self.basis.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.basis.shape[1])

    @property
    def map_sha256(self) -> str:
        hashes = self.report.get("hashes")
        if not isinstance(hashes, Mapping):
            raise WeightCompleteBasisError(
                "Weight-complete report has no hash contract."
            )
        digest = hashes.get("basis_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise WeightCompleteBasisError(
                "Weight-complete basis SHA-256 is invalid."
            )
        return digest

    def encode(self, source_rows: torch.Tensor) -> torch.Tensor:
        source = _finite_dense_tensor(source_rows, "source_rows")
        if int(source.shape[-1]) != self.source_width:
            raise WeightCompleteBasisError(
                "source_rows final axis does not match source_width."
            )
        return source @ self.basis

    def decode(self, target_rows: torch.Tensor) -> torch.Tensor:
        target = _finite_dense_tensor(target_rows, "target_rows")
        if int(target.shape[-1]) != self.target_width:
            raise WeightCompleteBasisError(
                "target_rows final axis does not match target_width."
            )
        return target @ self.basis.T

    def apply(self, source_rows: torch.Tensor) -> torch.Tensor:
        return self.encode(source_rows)

    def __call__(self, source_rows: torch.Tensor) -> torch.Tensor:
        return self.encode(source_rows)

    def fold_consumer_weight(
        self,
        source_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Transport a residual-space consumer as ``W_source @ B``."""

        source = _finite_dense_tensor(
            source_weight,
            "source consumer weight",
        )
        if source.ndim != 2 or int(source.shape[1]) != self.source_width:
            raise WeightCompleteBasisError(
                "A source consumer weight must be a matrix with source_width "
                "columns."
            )
        return (source @ self.basis).contiguous()

    def fold_producer_weight(
        self,
        source_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Transport a residual-space producer as ``B.T @ W_source``."""

        source = _finite_dense_tensor(
            source_weight,
            "source producer weight",
        )
        if source.ndim != 2 or int(source.shape[0]) != self.source_width:
            raise WeightCompleteBasisError(
                "A source producer weight must be a matrix with source_width "
                "rows."
            )
        return (self.basis.T @ source).contiguous()

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: torch.dtype | None = None,
        device: Any = "cpu",
    ) -> torch.Tensor:
        """Materialize ``B[source, target]`` under an explicit memory bound."""

        maximum = _positive_int(max_elements, "max_elements")
        if int(self.basis.numel()) > maximum:
            raise WeightCompleteBasisError(
                "Dense basis exceeds the requested materialization bound."
            )
        target_dtype = self.basis.dtype if dtype is None else dtype
        if not isinstance(target_dtype, torch.dtype):
            raise WeightCompleteBasisError("dtype must be a torch dtype or None.")
        result = self.basis.to(device=device, dtype=target_dtype).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise WeightCompleteBasisError(
                "Materialized dense basis contains NaN or infinity."
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def build_weight_complete_basis(
    operators: Sequence[ResidualFacingOperator],
    target_width: int,
    *,
    activation_gram: torch.Tensor | None = None,
    activation_weight: float = 0.0,
    maximum_activation_weight: float = 0.25,
    minimum_full_weight_retained_energy_fraction: float = 0.0,
    require_all_source_axis_loading: bool = True,
    source_axis_loading_tolerance: float = 1e-18,
    rank_tolerance: float = 1e-12,
    degeneracy_relative_tolerance: float = 1e-11,
    degeneracy_absolute_tolerance: float = 1e-13,
    enforce_coverage_gate: bool = True,
) -> WeightCompleteBasisResult:
    """Build one deterministic top-k basis from complete structural weights."""

    operators = tuple(operators)
    if not operators or any(
        not isinstance(value, ResidualFacingOperator) for value in operators
    ):
        raise WeightCompleteBasisError(
            "operators must contain ResidualFacingOperator values."
        )
    ordered = tuple(
        sorted(
            operators,
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
        raise WeightCompleteBasisError(
            "Duplicate residual-facing operators are not allowed."
        )
    source_widths = {value.source_width for value in ordered}
    if len(source_widths) != 1:
        raise WeightCompleteBasisError(
            "Every residual-facing operator must share one source width."
        )
    source_width = next(iter(source_widths))
    target_width = _positive_int(target_width, "target_width")
    if target_width > source_width:
        raise WeightCompleteBasisError(
            "target_width cannot exceed source_width."
        )
    activation_weight = _unit_interval(
        activation_weight,
        "activation_weight",
        upper_open=True,
    )
    maximum_activation_weight = _unit_interval(
        maximum_activation_weight,
        "maximum_activation_weight",
        upper_open=True,
    )
    if activation_weight > maximum_activation_weight:
        raise WeightCompleteBasisError(
            "activation_weight exceeds maximum_activation_weight."
        )
    minimum_retained = _unit_interval(
        minimum_full_weight_retained_energy_fraction,
        "minimum_full_weight_retained_energy_fraction",
    )
    loading_tolerance = _nonnegative_float(
        source_axis_loading_tolerance,
        "source_axis_loading_tolerance",
    )
    rank_tolerance = _nonnegative_float(rank_tolerance, "rank_tolerance")
    degeneracy_relative_tolerance = _nonnegative_float(
        degeneracy_relative_tolerance,
        "degeneracy_relative_tolerance",
    )
    degeneracy_absolute_tolerance = _nonnegative_float(
        degeneracy_absolute_tolerance,
        "degeneracy_absolute_tolerance",
    )
    for name, value in (
        ("require_all_source_axis_loading", require_all_source_axis_loading),
        ("enforce_coverage_gate", enforce_coverage_gate),
    ):
        if not isinstance(value, bool):
            raise WeightCompleteBasisError(f"{name} must be boolean.")

    original_threads = torch.get_num_threads()
    with _NUMERICAL_LOCK:
        try:
            torch.set_num_threads(1)
            with torch.no_grad():
                role_grams: dict[str, torch.Tensor] = {}
                operator_reports: list[dict[str, Any]] = []
                for operator in ordered:
                    contribution = _operator_gram(operator)
                    role_grams.setdefault(
                        operator.role,
                        torch.zeros(
                            (source_width, source_width),
                            dtype=torch.float64,
                        ),
                    )
                    role_grams[operator.role].add_(contribution)
                    operator_reports.append(
                        {
                            **operator.as_dict(),
                            "gram_sha256": _tensor_sha256(contribution),
                            "gram_trace_frobenius_energy": float(
                                torch.trace(contribution).item()
                            ),
                            "concatenated_rows_materialized": False,
                        }
                    )
                roles = tuple(sorted(role_grams))
                raw_role_energies = {
                    role: float(torch.trace(role_grams[role]).item())
                    for role in roles
                }
                if any(
                    not math.isfinite(value) or value <= 0.0
                    for value in raw_role_energies.values()
                ):
                    raise WeightCompleteBasisError(
                        "Every structural role must carry finite positive "
                        "Frobenius energy."
                    )
                role_count = len(roles)
                role_scales = {
                    role: 1.0 / (role_count * raw_role_energies[role])
                    for role in roles
                }
                balanced_structural = sum(
                    (
                        role_grams[role] * role_scales[role]
                        for role in roles
                    ),
                    torch.zeros(
                        (source_width, source_width),
                        dtype=torch.float64,
                    ),
                )
                balanced_structural = (
                    (balanced_structural + balanced_structural.T) * 0.5
                ).contiguous()
                full_weight_gram = sum(
                    (role_grams[role] for role in roles),
                    torch.zeros(
                        (source_width, source_width),
                        dtype=torch.float64,
                    ),
                )
                full_weight_gram = (
                    (full_weight_gram + full_weight_gram.T) * 0.5
                ).contiguous()

                activation_normalized = None
                activation_report: dict[str, Any]
                if activation_gram is None:
                    if activation_weight != 0.0:
                        raise WeightCompleteBasisError(
                            "A positive activation_weight requires "
                            "activation_gram."
                        )
                    combined = balanced_structural.clone()
                    activation_report = {
                        "supplied": False,
                        "requested_weight": activation_weight,
                        "maximum_weight": maximum_activation_weight,
                        "actual_weight": 0.0,
                        "gram_sha256": None,
                    }
                else:
                    activation = _finite_dense_tensor(
                        activation_gram, "activation_gram"
                    )
                    if tuple(activation.shape) != (
                        source_width,
                        source_width,
                    ):
                        raise WeightCompleteBasisError(
                            "activation_gram must have source-width square "
                            "shape."
                        )
                    symmetry_error = float(
                        (activation - activation.T).abs().max().item()
                    )
                    symmetry_scale = max(
                        1.0, float(activation.abs().max().item())
                    )
                    if symmetry_error > 1e-10 * symmetry_scale:
                        raise WeightCompleteBasisError(
                            "activation_gram is not symmetric."
                        )
                    activation = ((activation + activation.T) * 0.5).contiguous()
                    activation_eigenvalues = torch.linalg.eigvalsh(activation)
                    activation_scale = max(
                        torch.finfo(torch.float64).tiny,
                        float(activation_eigenvalues.abs().max().item()),
                    )
                    minimum_activation_eigenvalue = float(
                        activation_eigenvalues.min().item()
                    )
                    if minimum_activation_eigenvalue < (
                        -1e-10 * activation_scale
                    ):
                        raise WeightCompleteBasisError(
                            "activation_gram is not positive semidefinite."
                        )
                    activation_psd_shift = max(
                        0.0, -minimum_activation_eigenvalue
                    )
                    if activation_psd_shift:
                        activation = (
                            activation
                            + torch.eye(
                                source_width, dtype=torch.float64
                            )
                            * activation_psd_shift
                        ).contiguous()
                    activation_energy = float(torch.trace(activation).item())
                    if not math.isfinite(activation_energy) or activation_energy <= 0.0:
                        raise WeightCompleteBasisError(
                            "activation_gram must have finite positive trace."
                        )
                    activation_normalized = (
                        activation / activation_energy
                    ).contiguous()
                    combined = (
                        (1.0 - activation_weight) * balanced_structural
                        + activation_weight * activation_normalized
                    )
                    combined = ((combined + combined.T) * 0.5).contiguous()
                    activation_report = {
                        "supplied": True,
                        "requested_weight": activation_weight,
                        "maximum_weight": maximum_activation_weight,
                        "actual_weight": activation_weight,
                        "raw_trace_energy": activation_energy,
                        "positive_semidefinite_diagonal_shift": (
                            activation_psd_shift
                        ),
                        "gram_sha256": _tensor_sha256(activation),
                        "normalized_gram_sha256": _tensor_sha256(
                            activation_normalized
                        ),
                    }

                combined_eigenvalues = torch.linalg.eigvalsh(combined)
                combined_rank = _rank(
                    combined_eigenvalues,
                    width=source_width,
                    tolerance=rank_tolerance,
                )
                if combined_rank < target_width:
                    raise WeightCompleteBasisCoverageError(
                        "Combined structural/activation objective rank is below "
                        "target_width."
                    )
                basis, selected_eigenvalues, degeneracy_groups = (
                    _canonical_top_k(
                        combined,
                        target_width,
                        degeneracy_relative_tolerance=(
                            degeneracy_relative_tolerance
                        ),
                        degeneracy_absolute_tolerance=(
                            degeneracy_absolute_tolerance
                        ),
                    )
                )

                projection = basis @ basis.T
                role_reports: dict[str, dict[str, Any]] = {}
                for role in roles:
                    raw_energy = raw_role_energies[role]
                    projected_energy = float(
                        torch.trace(basis.T @ role_grams[role] @ basis).item()
                    )
                    retained = projected_energy / raw_energy
                    role_reports[role] = {
                        "operator_count": sum(
                            value.role == role for value in ordered
                        ),
                        "raw_trace_frobenius_energy": raw_energy,
                        "balanced_objective_scale": role_scales[role],
                        "balanced_trace_contribution": float(
                            torch.trace(
                                role_grams[role] * role_scales[role]
                            ).item()
                        ),
                        "projected_raw_energy": projected_energy,
                        "retained_energy_fraction": retained,
                        "relative_reconstruction_error": math.sqrt(
                            max(0.0, 1.0 - retained)
                        ),
                        "gram_sha256": _tensor_sha256(role_grams[role]),
                    }
                full_weight_energy = float(
                    torch.trace(full_weight_gram).item()
                )
                full_weight_projected = float(
                    torch.trace(
                        basis.T @ full_weight_gram @ basis
                    ).item()
                )
                full_weight_retained = (
                    full_weight_projected / full_weight_energy
                )
                balanced_energy = float(
                    torch.trace(balanced_structural).item()
                )
                balanced_projected = float(
                    torch.trace(
                        basis.T @ balanced_structural @ basis
                    ).item()
                )
                balanced_retained = balanced_projected / balanced_energy
                raw_full_eigenvalues = torch.linalg.eigvalsh(
                    full_weight_gram
                )
                full_weight_rank = _rank(
                    raw_full_eigenvalues,
                    width=source_width,
                    tolerance=rank_tolerance,
                )
                basis_rank = _rank(
                    torch.linalg.eigvalsh(basis.T @ basis),
                    width=target_width,
                    tolerance=rank_tolerance,
                )
                structural_diagonal = torch.diagonal(full_weight_gram)
                maximum_structural_axis_energy = float(
                    structural_diagonal.abs().max().item()
                )
                structural_axis_threshold = (
                    maximum_structural_axis_energy
                    * max(
                        rank_tolerance,
                        torch.finfo(torch.float64).eps * source_width,
                    )
                )
                structurally_covered = (
                    structural_diagonal > structural_axis_threshold
                )
                row_loadings = basis.square().sum(dim=1)
                loaded = row_loadings > loading_tolerance
                structurally_covered_count = int(
                    structurally_covered.sum().item()
                )
                loaded_count = int(loaded.sum().item())
                covered_and_loaded_count = int(
                    (structurally_covered & loaded).sum().item()
                )
                all_structural_axes_present = (
                    structurally_covered_count == source_width
                    and covered_and_loaded_count == source_width
                )
                retention_gate_passed = (
                    full_weight_retained + 1e-15 >= minimum_retained
                )
                rank_gate_passed = (
                    full_weight_rank >= target_width
                    and combined_rank >= target_width
                    and basis_rank == target_width
                )
                axis_gate_passed = (
                    all_structural_axes_present
                    if require_all_source_axis_loading
                    else True
                )
                coverage_gate_passed = (
                    retention_gate_passed
                    and rank_gate_passed
                    and axis_gate_passed
                )
                if enforce_coverage_gate and not coverage_gate_passed:
                    failures = []
                    if not retention_gate_passed:
                        failures.append("full-weight retained-energy")
                    if not rank_gate_passed:
                        failures.append("rank")
                    if not axis_gate_passed:
                        failures.append("all-source-axis loading")
                    raise WeightCompleteBasisCoverageError(
                        "Weight-complete basis failed coverage gate(s): "
                        + ", ".join(failures)
                        + "."
                    )

                report: dict[str, Any] = {
                    "schema": BASIS_SCHEMA,
                    "source_width": source_width,
                    "target_width": target_width,
                    "operator_count": len(ordered),
                    "role_count": role_count,
                    "roles": role_reports,
                    "operators": operator_reports,
                    "role_frobenius_balancing": {
                        "enabled": True,
                        "algorithm": (
                            "equal-role-trace-after-raw-frobenius-energy-"
                            "normalization"
                        ),
                        "role_order": list(roles),
                        "balanced_role_trace_target": 1.0 / role_count,
                        "large_row_count_roles_cannot_dominate_by_size": True,
                    },
                    "activation_supplement": {
                        **activation_report,
                        "structural_objective_weight": (
                            1.0 - activation_weight
                        ),
                        "activation_cannot_replace_structural_objective": True,
                        "structural_direction_energy_lower_bound_factor": (
                            1.0 - activation_weight
                        ),
                    },
                    "per_role_retained_energy": {
                        role: role_reports[role][
                            "retained_energy_fraction"
                        ]
                        for role in roles
                    },
                    "full_weight_raw_trace_energy": full_weight_energy,
                    "full_weight_projected_energy": full_weight_projected,
                    "full_weight_retained_energy_fraction": (
                        full_weight_retained
                    ),
                    "full_weight_relative_reconstruction_error": math.sqrt(
                        max(0.0, 1.0 - full_weight_retained)
                    ),
                    "balanced_structural_retained_energy_fraction": (
                        balanced_retained
                    ),
                    "full_weight_rank": full_weight_rank,
                    "combined_objective_rank": combined_rank,
                    "basis_rank": basis_rank,
                    "selected_eigenvalues": [
                        float(value)
                        for value in selected_eigenvalues.tolist()
                    ],
                    "degenerate_eigenspaces": list(degeneracy_groups),
                    "determinism": {
                        "cpu_float64": True,
                        "internal_numerical_threads": 1,
                        "caller_thread_count_independent": True,
                        "operator_order_canonicalized": True,
                        "degenerate_subspaces_canonicalized_from_projectors": True,
                        "vector_signs_canonicalized": True,
                    },
                    "source_axis_loading": {
                        "tolerance": loading_tolerance,
                        "structurally_covered_axis_count": (
                            structurally_covered_count
                        ),
                        "basis_loaded_axis_count": loaded_count,
                        "structurally_covered_and_loaded_axis_count": (
                            covered_and_loaded_count
                        ),
                        "all_source_axes_structurally_covered_and_loaded": (
                            all_structural_axes_present
                        ),
                        "minimum_basis_row_loading": float(
                            row_loadings.min().item()
                        ),
                        "maximum_basis_row_loading": float(
                            row_loadings.max().item()
                        ),
                    },
                    "coverage_gate": {
                        "enforced": enforce_coverage_gate,
                        "passed": coverage_gate_passed,
                        "minimum_full_weight_retained_energy_fraction": (
                            minimum_retained
                        ),
                        "full_weight_retained_energy_gate_passed": (
                            retention_gate_passed
                        ),
                        "rank_gate_passed": rank_gate_passed,
                        "require_all_source_axis_loading": (
                            require_all_source_axis_loading
                        ),
                        "all_source_axis_loading_gate_passed": (
                            axis_gate_passed
                        ),
                    },
                    "hashes": {
                        "operator_manifest_sha256": hashlib.sha256(
                            json.dumps(
                                [
                                    value.operator_sha256
                                    for value in ordered
                                ],
                                separators=(",", ":"),
                            ).encode("ascii")
                        ).hexdigest(),
                        "full_weight_gram_sha256": _tensor_sha256(
                            full_weight_gram
                        ),
                        "balanced_structural_gram_sha256": _tensor_sha256(
                            balanced_structural
                        ),
                        "combined_gram_sha256": _tensor_sha256(combined),
                        "basis_sha256": _tensor_sha256(basis),
                        "projection_sha256": _tensor_sha256(projection),
                    },
                    "prompt_independent_weight_objective": True,
                    "model_forward_calls": 0,
                    "trace_capture_calls": 0,
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "autograd_used": False,
                    "filesystem_writes": 0,
                    "concatenated_operator_rows_materialized": False,
                }
                report["run_sha256"] = hashlib.sha256(
                    json.dumps(
                        report,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest()
                return WeightCompleteBasisResult(
                    basis=basis,
                    full_weight_gram=full_weight_gram,
                    balanced_structural_gram=balanced_structural,
                    combined_gram=combined,
                    report=report,
                )
        finally:
            torch.set_num_threads(original_threads)


__all__ = [
    "BASIS_SCHEMA",
    "FeatureOrientation",
    "ResidualFacingOperator",
    "WeightCompleteBasisCoverageError",
    "WeightCompleteBasisError",
    "WeightCompleteBasisResult",
    "build_weight_complete_basis",
    "conv_state_operators",
    "embedding_head_operators",
    "norm_operators",
    "output_mlp_span_operators",
    "qkv_input_sensitivity_operators",
    "residual_facing_operator",
]
