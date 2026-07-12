"""Output-Jacobian-guided directions orthogonal to inherited low-rank state.

This module is intentionally model independent.  It consumes a finite matrix
``G`` and an inherited factorization ``L0 @ R0`` only; it does not load a
model, run autograd, construct an optimizer, or mutate any caller-owned
tensor.  The inherited column and row spaces are removed from ``G`` as

``Gnormal = (I - U U.T) G (I - V V.T)``

using projection products rather than materializing either identity matrix.
The leading right-singular subspace of ``Gnormal`` is the proposed next right
factor tranche.  An equal-rank, seed-bound random basis in the same inherited
row-space complement is evaluated as a preregistered control.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "output-jacobian-guided-orthogonal-rank-tranche"
MINIMUM_NORMAL_ENERGY_FRACTION = 0.20
MINIMUM_GUIDED_RANDOM_CAPTURE_RATIO = 1.50
_ORTHOGONALITY_ATOL = 1e-10
_SPECTRAL_TOLERANCE_MULTIPLIER = 64.0


class OutputJacobianOrthogonalBasisError(ValueError):
    """Base class for validation or basis-construction failures."""


class OutputJacobianShapeError(OutputJacobianOrthogonalBasisError):
    """An input tensor, factor relation, seed, or rank has an invalid shape."""


class OutputJacobianFiniteError(OutputJacobianOrthogonalBasisError):
    """An input, intermediate, metric, or output contains NaN or infinity."""


class OutputJacobianDegenerateError(OutputJacobianOrthogonalBasisError):
    """The requested tranche is mathematically undefined or unsupported."""


@dataclass(frozen=True, slots=True)
class OutputJacobianOrthogonalBasisReport:
    """Canonical evidence for one guided basis and its random control."""

    output_rows: int
    input_columns: int
    inherited_factor_rank: int
    inherited_column_rank: int
    inherited_row_rank: int
    requested_rank: int
    normal_effective_rank: int
    random_seed: int
    total_energy: float
    normal_energy: float
    normal_energy_fraction: float
    guided_capture_energy: float
    guided_capture_fraction: float
    random_control_capture_energy: float
    random_control_capture_fraction: float
    guided_random_capture_ratio: float
    minimum_normal_energy_fraction: float
    minimum_guided_random_capture_ratio: float
    normal_energy_threshold_passed: bool
    guided_random_ratio_threshold_passed: bool
    preregistration_passed: bool
    normal_singular_values: tuple[float, ...]
    selected_singular_values: tuple[float, ...]
    guided_canonical_pivot_indices: tuple[int, ...]
    random_canonical_pivot_indices: tuple[int, ...]
    inherited_column_orthogonality_max_abs_error: float
    inherited_row_orthogonality_max_abs_error: float
    guided_orthogonality_max_abs_error: float
    random_orthogonality_max_abs_error: float
    normal_left_leakage_max_abs: float
    normal_right_leakage_max_abs: float
    guided_inherited_row_overlap_max_abs: float
    random_inherited_row_overlap_max_abs: float
    output_jacobian_sha256: str
    inherited_left_factor_sha256: str
    inherited_right_factor_sha256: str
    inherited_column_basis_sha256: str
    inherited_row_basis_sha256: str
    normal_jacobian_sha256: str
    guided_basis_sha256: str
    right_factor_rows_sha256: str
    random_control_basis_sha256: str
    finite: bool = True
    deterministic: bool = True
    input_mutation: bool = False
    backward_used: bool = False
    optimizer_used: bool = False

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "orientation": {
                "guided_basis": "[input_columns, requested_rank] orthonormal columns",
                "right_factor_rows": "[requested_rank, input_columns]",
                "random_control_basis": (
                    "[input_columns, requested_rank] orthonormal columns"
                ),
            },
            "shape": {
                "output_rows": self.output_rows,
                "input_columns": self.input_columns,
                "inherited_factor_rank": self.inherited_factor_rank,
                "inherited_column_rank": self.inherited_column_rank,
                "inherited_row_rank": self.inherited_row_rank,
                "requested_rank": self.requested_rank,
                "normal_effective_rank": self.normal_effective_rank,
            },
            "random_control": {"seed": self.random_seed, "equal_rank": True},
            "energy": {
                "total": self.total_energy,
                "normal": self.normal_energy,
                "normal_fraction": self.normal_energy_fraction,
                "guided_capture": self.guided_capture_energy,
                "guided_capture_fraction": self.guided_capture_fraction,
                "random_control_capture": self.random_control_capture_energy,
                "random_control_capture_fraction": (
                    self.random_control_capture_fraction
                ),
                "guided_random_capture_ratio": self.guided_random_capture_ratio,
                "capture_fraction_denominator": "normal_energy",
            },
            "preregistration": {
                "minimum_normal_energy_fraction": (
                    self.minimum_normal_energy_fraction
                ),
                "minimum_guided_random_capture_ratio": (
                    self.minimum_guided_random_capture_ratio
                ),
                "normal_energy_threshold_passed": (
                    self.normal_energy_threshold_passed
                ),
                "guided_random_ratio_threshold_passed": (
                    self.guided_random_ratio_threshold_passed
                ),
                "passed": self.preregistration_passed,
            },
            "spectrum": {
                "normal_singular_values": list(self.normal_singular_values),
                "selected_singular_values": list(self.selected_singular_values),
            },
            "canonical_signs": {
                "guided_pivot_indices": list(
                    self.guided_canonical_pivot_indices
                ),
                "random_pivot_indices": list(self.random_canonical_pivot_indices),
                "pivot_rule": "lowest-index maximum-absolute coordinate is positive",
            },
            "numerical_validation": {
                "inherited_column_orthogonality_max_abs_error": (
                    self.inherited_column_orthogonality_max_abs_error
                ),
                "inherited_row_orthogonality_max_abs_error": (
                    self.inherited_row_orthogonality_max_abs_error
                ),
                "guided_orthogonality_max_abs_error": (
                    self.guided_orthogonality_max_abs_error
                ),
                "random_orthogonality_max_abs_error": (
                    self.random_orthogonality_max_abs_error
                ),
                "normal_left_leakage_max_abs": self.normal_left_leakage_max_abs,
                "normal_right_leakage_max_abs": self.normal_right_leakage_max_abs,
                "guided_inherited_row_overlap_max_abs": (
                    self.guided_inherited_row_overlap_max_abs
                ),
                "random_inherited_row_overlap_max_abs": (
                    self.random_inherited_row_overlap_max_abs
                ),
            },
            "hashes": {
                "output_jacobian": self.output_jacobian_sha256,
                "inherited_left_factor": self.inherited_left_factor_sha256,
                "inherited_right_factor": self.inherited_right_factor_sha256,
                "inherited_column_basis": self.inherited_column_basis_sha256,
                "inherited_row_basis": self.inherited_row_basis_sha256,
                "normal_jacobian": self.normal_jacobian_sha256,
                "guided_basis": self.guided_basis_sha256,
                "right_factor_rows": self.right_factor_rows_sha256,
                "random_control_basis": self.random_control_basis_sha256,
            },
            "execution": {
                "finite": self.finite,
                "deterministic": self.deterministic,
                "input_mutation": self.input_mutation,
                "backward_used": self.backward_used,
                "optimizer_used": self.optimizer_used,
                "torch_only": True,
                "working_dtype": "torch.float64",
                "working_device": "cpu",
            },
        }

    @property
    def report_sha256(self) -> str:
        """SHA-256 of the canonical report payload (excluding this hash)."""

        return hashlib.sha256(self.canonical_payload_json().encode("utf-8")).hexdigest()

    def canonical_payload_json(self) -> str:
        """Return the exact canonical JSON used as the report hash basis."""

        return json.dumps(
            self._hash_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a canonical, JSON-compatible report including its hash."""

        payload = self._hash_payload()
        payload["report_sha256"] = self.report_sha256
        return payload

    def canonical_json(self) -> str:
        """Return canonical JSON for the complete externally stored report."""

        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class OutputJacobianOrthogonalBasis:
    """Detached CPU-float64 matrices and evidence for the proposed tranche."""

    inherited_column_basis: Tensor
    inherited_row_basis: Tensor
    normal_jacobian: Tensor
    guided_basis: Tensor
    right_factor_rows: Tensor
    random_control_basis: Tensor
    report: OutputJacobianOrthogonalBasisReport

    @property
    def rank(self) -> int:
        return self.report.requested_rank

    def as_dict(self) -> dict[str, Any]:
        """Return the report without embedding dense tensor payloads."""

        return self.report.as_dict()


def _finite_matrix(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise OutputJacobianShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise OutputJacobianShapeError(f"{name} must be a dense non-meta tensor.")
    if value.ndim != 2:
        raise OutputJacobianShapeError(f"{name} must be two-dimensional.")
    if int(value.shape[0]) <= 0 or int(value.shape[1]) <= 0:
        raise OutputJacobianShapeError(f"{name} cannot have an empty dimension.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise OutputJacobianShapeError(f"{name} must have a real floating dtype.")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise OutputJacobianFiniteError(f"{name} contains NaN or infinity.")
    try:
        result = detached.to(device="cpu", dtype=torch.float64).contiguous().clone()
    except (RuntimeError, TypeError) as error:
        raise OutputJacobianShapeError(
            f"{name} could not be copied into canonical CPU float64 storage."
        ) from error
    if not bool(torch.isfinite(result).all().item()):
        raise OutputJacobianFiniteError(
            f"{name} became non-finite in canonical float64 storage."
        )
    return result


def _positive_rank(value: Any) -> int:
    if isinstance(value, bool):
        raise OutputJacobianShapeError("rank must be a positive integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise OutputJacobianShapeError("rank must be a positive integer.") from error
    if result <= 0:
        raise OutputJacobianShapeError("rank must be a positive integer.")
    return result


def _random_seed(value: Any) -> int:
    if isinstance(value, bool):
        raise OutputJacobianShapeError(
            "random_seed must be an integer in [0, 2**63 - 1]."
        )
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise OutputJacobianShapeError(
            "random_seed must be an integer in [0, 2**63 - 1]."
        ) from error
    if result < 0 or result > 2**63 - 1:
        raise OutputJacobianShapeError(
            "random_seed must be an integer in [0, 2**63 - 1]."
        )
    return result


def _safe_svd(value: Tensor, name: str) -> tuple[Tensor, Tensor, Tensor]:
    try:
        left, singular_values, right_t = torch.linalg.svd(
            value,
            full_matrices=False,
        )
    except RuntimeError as error:
        raise OutputJacobianDegenerateError(f"SVD failed for {name}.") from error
    if not all(
        bool(torch.isfinite(item).all().item())
        for item in (left, singular_values, right_t)
    ):
        raise OutputJacobianFiniteError(
            f"SVD for {name} produced NaN or infinity."
        )
    return left, singular_values, right_t


def _spectral_tolerance(singular_values: Tensor, shape: tuple[int, int]) -> float:
    if singular_values.numel() == 0:
        return 0.0
    leading = float(singular_values[0].item())
    return (
        _SPECTRAL_TOLERANCE_MULTIPLIER
        * torch.finfo(torch.float64).eps
        * max(shape)
        * leading
    )


def _effective_rank(singular_values: Tensor, shape: tuple[int, int]) -> int:
    tolerance = _spectral_tolerance(singular_values, shape)
    return int(torch.count_nonzero(singular_values > tolerance).item())


def _orthogonality_error(value: Tensor) -> float:
    if int(value.shape[1]) == 0:
        return 0.0
    gram_error = value.T @ value
    gram_error.diagonal().sub_(1.0)
    return float(torch.max(torch.abs(gram_error)).item())


def _maximum_absolute(value: Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(value)).item())


def _canonicalize_signs(value: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    result = value.clone()
    pivots: list[int] = []
    for column_index in range(int(result.shape[1])):
        column = result[:, column_index]
        pivot = int(torch.argmax(torch.abs(column)).item())
        pivot_value = float(column[pivot].item())
        if not math.isfinite(pivot_value) or pivot_value == 0.0:
            raise OutputJacobianDegenerateError(
                "An orthonormal basis column has no finite nonzero pivot."
            )
        if pivot_value < 0.0:
            result[:, column_index].neg_()
        pivots.append(pivot)
    return result, tuple(pivots)


def _canonical_subspace(value: Tensor, width: int, name: str) -> Tensor:
    """Construct a coordinate-ordered basis from an orthonormal span."""

    if width <= 0 or tuple(value.shape)[1] != width:
        raise OutputJacobianDegenerateError(
            f"{name} does not contain the requested subspace width."
        )
    selected: list[Tensor] = []
    tolerance = (
        _SPECTRAL_TOLERANCE_MULTIPLIER
        * torch.finfo(torch.float64).eps
        * max(value.shape)
    )
    for coordinate in range(int(value.shape[0])):
        candidate = value @ value[coordinate, :]
        for _pass in range(2):
            for previous in selected:
                candidate = candidate - previous * torch.dot(previous, candidate)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm > tolerance:
            selected.append(candidate / norm)
            if len(selected) == width:
                break
    if len(selected) != width:
        raise OutputJacobianDegenerateError(
            f"Could not construct a deterministic basis for {name}."
        )
    result = torch.stack(selected, dim=1)
    result, _pivots = _canonicalize_signs(result)
    if _orthogonality_error(result) > _ORTHOGONALITY_ATOL:
        raise OutputJacobianDegenerateError(
            f"Canonical basis for {name} is not numerically orthonormal."
        )
    return result


def _full_rank_span(value: Tensor, *, columns: bool, name: str) -> Tensor:
    matrix = value if columns else value.T
    left, singular_values, _right_t = _safe_svd(matrix, name)
    expected_rank = int(matrix.shape[1])
    actual_rank = _effective_rank(singular_values, tuple(matrix.shape))
    if actual_rank != expected_rank:
        raise OutputJacobianDegenerateError(
            f"{name} has effective rank {actual_rank}, expected {expected_rank}."
        )
    basis = _canonical_subspace(left[:, :expected_rank], expected_rank, name)
    return basis


def _degenerate_blocks(
    singular_values: Tensor,
    selected_rank: int,
    tolerance: float,
) -> tuple[tuple[int, int], ...]:
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < selected_rank:
        stop = start + 1
        while stop < selected_rank and abs(
            float(singular_values[stop - 1].item())
            - float(singular_values[stop].item())
        ) <= tolerance:
            stop += 1
        blocks.append((start, stop))
        start = stop
    return tuple(blocks)


def _guided_right_basis(
    right_t: Tensor,
    singular_values: Tensor,
    rank: int,
    normal_shape: tuple[int, int],
) -> tuple[Tensor, tuple[int, ...]]:
    tolerance = _spectral_tolerance(singular_values, normal_shape)
    if rank < int(singular_values.numel()) and abs(
        float(singular_values[rank - 1].item())
        - float(singular_values[rank].item())
    ) <= tolerance:
        raise OutputJacobianDegenerateError(
            "The requested rank cuts through a degenerate singular-value block."
        )

    basis = right_t[:rank, :].T.contiguous()
    for start, stop in _degenerate_blocks(singular_values, rank, tolerance):
        if stop - start > 1:
            basis[:, start:stop] = _canonical_subspace(
                basis[:, start:stop],
                stop - start,
                "selected right-singular subspace",
            )
    basis, pivots = _canonicalize_signs(basis)
    if _orthogonality_error(basis) > _ORTHOGONALITY_ATOL:
        raise OutputJacobianDegenerateError(
            "The guided right-singular basis is not numerically orthonormal."
        )
    return basis, pivots


def _random_control_basis(
    input_columns: int,
    rank: int,
    inherited_row_basis: Tensor,
    seed: int,
) -> tuple[Tensor, tuple[int, ...]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_columns = torch.randn(
        input_columns,
        rank,
        dtype=torch.float64,
        generator=generator,
    )
    random_columns = random_columns - inherited_row_basis @ (
        inherited_row_basis.T @ random_columns
    )
    try:
        basis, triangular = torch.linalg.qr(random_columns, mode="reduced")
    except RuntimeError as error:
        raise OutputJacobianDegenerateError(
            "QR failed for the random orthogonal control."
        ) from error
    if not all(
        bool(torch.isfinite(item).all().item()) for item in (basis, triangular)
    ):
        raise OutputJacobianFiniteError(
            "Random-control QR produced NaN or infinity."
        )
    diagonal = torch.abs(torch.diagonal(triangular))
    # QR diagonals are not ordered singular values.  Use their actual maximum
    # as the rank-test scale rather than assuming the first diagonal is the
    # leading one (the spectral helper's contract).
    leading = float(torch.max(diagonal).item()) if diagonal.numel() else 0.0
    tolerance = (
        _SPECTRAL_TOLERANCE_MULTIPLIER
        * torch.finfo(torch.float64).eps
        * max(random_columns.shape)
        * leading
    )
    if int(torch.count_nonzero(diagonal > tolerance).item()) != rank:
        raise OutputJacobianDegenerateError(
            "The seed-bound random control is rank deficient after projection."
        )
    basis, pivots = _canonicalize_signs(basis)
    if _orthogonality_error(basis) > _ORTHOGONALITY_ATOL:
        raise OutputJacobianDegenerateError(
            "The random control basis is not numerically orthonormal."
        )
    return basis, pivots


def _energy(value: Tensor, name: str) -> float:
    result = float(torch.sum(torch.square(value)).item())
    if not math.isfinite(result):
        raise OutputJacobianFiniteError(f"{name} is NaN or infinity.")
    return result


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(b"torch.float64-le\0")
    digest.update(
        json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    # Flatten first because PyTorch may regard a size-one trailing dimension
    # as contiguous while retaining a non-unit (semantically irrelevant)
    # stride that cannot be reinterpreted as bytes.
    raw = canonical.reshape(-1).contiguous().view(torch.uint8).clone()
    payload = bytes(raw.untyped_storage())
    if sys.byteorder != "little":
        raise OutputJacobianShapeError(
            "Canonical tensor hashing requires a little-endian host."
        )
    digest.update(payload)
    return digest.hexdigest()


def _validate_finite_metrics(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise OutputJacobianFiniteError("A reported metric is NaN or infinity.")


def build_output_jacobian_orthogonal_basis(
    G: Any,
    L0: Any,
    R0: Any,
    rank: Any,
    *,
    random_seed: Any = 0,
) -> OutputJacobianOrthogonalBasis:
    """Build a deterministic next-rank right basis and random control.

    ``G`` has shape ``[output_rows, input_columns]``.  The inherited factors
    must have shapes ``L0=[output_rows, inherited_factor_rank]`` and
    ``R0=[inherited_factor_rank, input_columns]``.  Returned ``guided_basis``
    has orthonormal columns with shape ``[input_columns, rank]``;
    ``right_factor_rows`` is its contiguous transpose with shape
    ``[rank, input_columns]``.
    """

    output_jacobian = _finite_matrix(G, "G")
    left_factor = _finite_matrix(L0, "L0")
    right_factor = _finite_matrix(R0, "R0")
    requested_rank = _positive_rank(rank)
    seed = _random_seed(random_seed)

    output_rows, input_columns = map(int, output_jacobian.shape)
    inherited_factor_rank = int(left_factor.shape[1])
    if int(left_factor.shape[0]) != output_rows:
        raise OutputJacobianShapeError(
            "L0 rows must equal G output rows."
        )
    if int(right_factor.shape[0]) != inherited_factor_rank:
        raise OutputJacobianShapeError(
            "L0 columns must equal R0 rows."
        )
    if int(right_factor.shape[1]) != input_columns:
        raise OutputJacobianShapeError(
            "R0 columns must equal G input columns."
        )
    if inherited_factor_rank >= min(output_rows, input_columns):
        raise OutputJacobianDegenerateError(
            "Inherited factors leave no two-sided complement for another tranche."
        )

    inherited_column_basis = _full_rank_span(
        left_factor,
        columns=True,
        name="L0 column space",
    )
    inherited_row_basis = _full_rank_span(
        right_factor,
        columns=False,
        name="R0 row space",
    )
    inherited_column_rank = int(inherited_column_basis.shape[1])
    inherited_row_rank = int(inherited_row_basis.shape[1])
    maximum_rank = min(
        output_rows - inherited_column_rank,
        input_columns - inherited_row_rank,
    )
    if requested_rank > maximum_rank:
        raise OutputJacobianShapeError(
            f"rank cannot exceed the two-sided complement dimension {maximum_rank}."
        )

    # Apply both orthogonal complements without constructing either identity.
    normal_jacobian = output_jacobian - inherited_column_basis @ (
        inherited_column_basis.T @ output_jacobian
    )
    normal_jacobian = normal_jacobian - (
        normal_jacobian @ inherited_row_basis
    ) @ inherited_row_basis.T
    if not bool(torch.isfinite(normal_jacobian).all().item()):
        raise OutputJacobianFiniteError("Gnormal contains NaN or infinity.")

    total_energy = _energy(output_jacobian, "total energy")
    normal_energy = _energy(normal_jacobian, "normal energy")
    if total_energy <= 0.0:
        raise OutputJacobianDegenerateError("G has zero total energy.")
    if normal_energy <= 0.0:
        raise OutputJacobianDegenerateError(
            "G has zero energy outside the inherited column and row spaces."
        )
    energy_tolerance = (
        _SPECTRAL_TOLERANCE_MULTIPLIER
        * torch.finfo(torch.float64).eps
        * max(output_jacobian.shape)
        * total_energy
    )
    if normal_energy > total_energy + energy_tolerance:
        raise OutputJacobianFiniteError(
            "Projected normal energy exceeds total energy beyond tolerance."
        )

    _left, singular_values, right_t = _safe_svd(
        normal_jacobian,
        "Gnormal",
    )
    normal_effective_rank = _effective_rank(
        singular_values,
        tuple(normal_jacobian.shape),
    )
    if normal_effective_rank < requested_rank:
        raise OutputJacobianDegenerateError(
            "Gnormal has effective rank "
            f"{normal_effective_rank}, below requested rank {requested_rank}."
        )
    guided_basis, guided_pivots = _guided_right_basis(
        right_t,
        singular_values,
        requested_rank,
        tuple(normal_jacobian.shape),
    )
    random_basis, random_pivots = _random_control_basis(
        input_columns,
        requested_rank,
        inherited_row_basis,
        seed,
    )

    guided_projection = normal_jacobian @ guided_basis
    random_projection = normal_jacobian @ random_basis
    guided_capture_energy = _energy(guided_projection, "guided capture energy")
    random_capture_energy = _energy(
        random_projection,
        "random-control capture energy",
    )
    if random_capture_energy <= 0.0:
        raise OutputJacobianDegenerateError(
            "The seed-bound random control captured zero normal energy."
        )
    normal_energy_fraction = normal_energy / total_energy
    guided_capture_fraction = guided_capture_energy / normal_energy
    random_capture_fraction = random_capture_energy / normal_energy
    # Both fractions use the same normal-energy denominator, so the exact
    # control comparison is the direct captured-energy ratio.  Computing it
    # directly also avoids introducing two unnecessary rounding operations.
    guided_random_ratio = guided_capture_energy / random_capture_energy
    _validate_finite_metrics(
        total_energy,
        normal_energy,
        normal_energy_fraction,
        guided_capture_energy,
        guided_capture_fraction,
        random_capture_energy,
        random_capture_fraction,
        guided_random_ratio,
    )

    column_orthogonality_error = _orthogonality_error(inherited_column_basis)
    row_orthogonality_error = _orthogonality_error(inherited_row_basis)
    guided_orthogonality_error = _orthogonality_error(guided_basis)
    random_orthogonality_error = _orthogonality_error(random_basis)
    normal_left_leakage = _maximum_absolute(
        inherited_column_basis.T @ normal_jacobian
    )
    normal_right_leakage = _maximum_absolute(
        normal_jacobian @ inherited_row_basis
    )
    guided_row_overlap = _maximum_absolute(inherited_row_basis.T @ guided_basis)
    random_row_overlap = _maximum_absolute(inherited_row_basis.T @ random_basis)
    validation_scale = max(1.0, math.sqrt(normal_energy))
    if max(
        column_orthogonality_error,
        row_orthogonality_error,
        guided_orthogonality_error,
        random_orthogonality_error,
        guided_row_overlap,
        random_row_overlap,
        normal_left_leakage / validation_scale,
        normal_right_leakage / validation_scale,
    ) > _ORTHOGONALITY_ATOL:
        raise OutputJacobianDegenerateError(
            "An orthogonality or inherited-space leakage check failed."
        )

    right_factor_rows = guided_basis.T.contiguous()
    normal_threshold_passed = (
        normal_energy_fraction >= MINIMUM_NORMAL_ENERGY_FRACTION
    )
    ratio_threshold_passed = (
        guided_random_ratio >= MINIMUM_GUIDED_RANDOM_CAPTURE_RATIO
    )
    preregistration_passed = normal_threshold_passed and ratio_threshold_passed

    report = OutputJacobianOrthogonalBasisReport(
        output_rows=output_rows,
        input_columns=input_columns,
        inherited_factor_rank=inherited_factor_rank,
        inherited_column_rank=inherited_column_rank,
        inherited_row_rank=inherited_row_rank,
        requested_rank=requested_rank,
        normal_effective_rank=normal_effective_rank,
        random_seed=seed,
        total_energy=total_energy,
        normal_energy=normal_energy,
        normal_energy_fraction=normal_energy_fraction,
        guided_capture_energy=guided_capture_energy,
        guided_capture_fraction=guided_capture_fraction,
        random_control_capture_energy=random_capture_energy,
        random_control_capture_fraction=random_capture_fraction,
        guided_random_capture_ratio=guided_random_ratio,
        minimum_normal_energy_fraction=MINIMUM_NORMAL_ENERGY_FRACTION,
        minimum_guided_random_capture_ratio=(
            MINIMUM_GUIDED_RANDOM_CAPTURE_RATIO
        ),
        normal_energy_threshold_passed=normal_threshold_passed,
        guided_random_ratio_threshold_passed=ratio_threshold_passed,
        preregistration_passed=preregistration_passed,
        normal_singular_values=tuple(float(value) for value in singular_values),
        selected_singular_values=tuple(
            float(value) for value in singular_values[:requested_rank]
        ),
        guided_canonical_pivot_indices=guided_pivots,
        random_canonical_pivot_indices=random_pivots,
        inherited_column_orthogonality_max_abs_error=(
            column_orthogonality_error
        ),
        inherited_row_orthogonality_max_abs_error=row_orthogonality_error,
        guided_orthogonality_max_abs_error=guided_orthogonality_error,
        random_orthogonality_max_abs_error=random_orthogonality_error,
        normal_left_leakage_max_abs=normal_left_leakage,
        normal_right_leakage_max_abs=normal_right_leakage,
        guided_inherited_row_overlap_max_abs=guided_row_overlap,
        random_inherited_row_overlap_max_abs=random_row_overlap,
        output_jacobian_sha256=_tensor_sha256(output_jacobian),
        inherited_left_factor_sha256=_tensor_sha256(left_factor),
        inherited_right_factor_sha256=_tensor_sha256(right_factor),
        inherited_column_basis_sha256=_tensor_sha256(inherited_column_basis),
        inherited_row_basis_sha256=_tensor_sha256(inherited_row_basis),
        normal_jacobian_sha256=_tensor_sha256(normal_jacobian),
        guided_basis_sha256=_tensor_sha256(guided_basis),
        right_factor_rows_sha256=_tensor_sha256(right_factor_rows),
        random_control_basis_sha256=_tensor_sha256(random_basis),
    )

    # Return independent tensors so no output aliases an input or another output.
    return OutputJacobianOrthogonalBasis(
        inherited_column_basis=inherited_column_basis.clone(),
        inherited_row_basis=inherited_row_basis.clone(),
        normal_jacobian=normal_jacobian.clone(),
        guided_basis=guided_basis.clone(),
        right_factor_rows=right_factor_rows.clone(),
        random_control_basis=random_basis.clone(),
        report=report,
    )


def build_output_jacobian_guided_orthogonal_basis(
    G: Any,
    L0: Any,
    R0: Any,
    rank: Any,
    *,
    random_seed: Any = 0,
) -> OutputJacobianOrthogonalBasis:
    """Descriptive alias for :func:`build_output_jacobian_orthogonal_basis`."""

    return build_output_jacobian_orthogonal_basis(
        G,
        L0,
        R0,
        rank,
        random_seed=random_seed,
    )


__all__ = [
    "MINIMUM_GUIDED_RANDOM_CAPTURE_RATIO",
    "MINIMUM_NORMAL_ENERGY_FRACTION",
    "OutputJacobianDegenerateError",
    "OutputJacobianFiniteError",
    "OutputJacobianOrthogonalBasis",
    "OutputJacobianOrthogonalBasisError",
    "OutputJacobianOrthogonalBasisReport",
    "OutputJacobianShapeError",
    "build_output_jacobian_guided_orthogonal_basis",
    "build_output_jacobian_orthogonal_basis",
]
