"""Deterministic no-gradient mathematics for blank-target weight transport.

The coordinate convention follows the representation-space interpretation in
Transport and Merge (arXiv 2602.05495): a parent-space column vector is encoded
as ``B.T @ x`` and decoded as ``B @ z``.  Consequently, a source linear map is
transported as ``B_out.T @ W_source @ B_in``.  The parent-induced bases and the
uniform entropic transport plan follow the activation-space direction in
DOTResize (arXiv 2507.04517).

This module only transforms already-collected tensors.  It never invokes a
model or constructs an optimization or autograd path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


MAX_SINKHORN_ITERATIONS = 10_000
_COORDINATE_ATOL = 1e-10


class TransportMathError(ValueError):
    """Base class for fail-closed transport-math validation failures."""


class ShapeValidationError(TransportMathError):
    """A tensor or coordinate map has an inadmissible shape or type."""


class RankValidationError(TransportMathError):
    """Activation data cannot support the requested target rank."""


class FiniteValidationError(TransportMathError):
    """An input, intermediate, or result contains a non-finite value."""


class PathConsistencyError(TransportMathError):
    """Adjacent operators do not share one valid boundary coordinate map."""


class SinkhornConvergenceError(TransportMathError):
    """Bounded Sinkhorn iterations did not meet the marginal tolerance."""

    def __init__(
        self,
        message: str,
        diagnostics: SinkhornDiagnostics,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class PseudoinverseDiagnostics:
    """Moore-Penrose and projection checks for an orthonormal basis map."""

    rank: int
    expected_rank: int
    decode_pseudoinverse_max_abs_error: float
    encode_pseudoinverse_max_abs_error: float
    target_identity_max_abs_error: float
    source_projector_symmetry_max_abs_error: float
    source_projector_idempotence_max_abs_error: float
    finite: bool


@dataclass(frozen=True, slots=True)
class CoordinateMap:
    """Explicit parent/source and blank-target coordinate maps.

    ``decode`` is the semi-orthogonal basis ``B`` with shape
    ``(source_dim, target_dim)``.  ``encode`` is ``B.T``.  For row-major
    activations, encoding is ``source_rows @ decode`` and decoding is
    ``target_rows @ encode``.
    """

    source_dim: int
    target_dim: int
    activation_row_count: int
    centered_for_fit: bool
    effective_rank: int
    rank_tolerance: float
    explained_variance_fraction: float
    fit_mean: Any
    singular_values: Any
    encode: Any
    decode: Any
    source_projector: Any
    diagnostics: PseudoinverseDiagnostics

    @property
    def basis(self) -> Any:
        """Return ``B``, the target-to-source decoding basis."""

        return self.decode


@dataclass(frozen=True, slots=True)
class LinearTransportDiagnostics:
    """Shape, rank, and finite evidence for one transported linear map."""

    source_weight_shape: tuple[int, int]
    target_weight_shape: tuple[int, int]
    source_bias_present: bool
    target_bias_shape: tuple[int, ...] | None
    input_source_dim: int
    input_target_dim: int
    output_source_dim: int
    output_target_dim: int
    finite: bool


@dataclass(frozen=True, slots=True)
class LinearTransportResult:
    """A source operator represented in explicit target coordinates."""

    weight: Any
    bias: Any | None
    input_coordinates: CoordinateMap
    output_coordinates: CoordinateMap
    diagnostics: LinearTransportDiagnostics


@dataclass(frozen=True, slots=True)
class PathCompositionDiagnostics:
    """Evidence that two adjacent transported linears compose consistently."""

    shared_boundary_map: bool
    boundary_source_dim: int
    boundary_target_dim: int
    boundary_projector_rank: int
    boundary_is_square: bool
    boundary_map_max_abs_error: float
    producer_formula_max_abs_error: float
    consumer_formula_max_abs_error: float
    composed_weight_max_abs_error: float
    composed_bias_max_abs_error: float
    finite: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class SinkhornDiagnostics:
    """Convergence, marginal, total-mass, and finite Sinkhorn evidence."""

    iterations: int
    max_iterations: int
    converged: bool
    finite: bool
    epsilon: float
    tolerance: float
    total_mass: float
    mass_error: float
    max_source_marginal_error: float
    max_target_marginal_error: float
    cost_min: float
    cost_max: float


@dataclass(frozen=True, slots=True)
class SinkhornResult:
    """A balanced source-by-target entropic transport plan."""

    plan: Any
    cost_matrix: Any
    source_marginal: Any
    target_marginal: Any
    diagnostics: SinkhornDiagnostics

    @property
    def transport_plan(self) -> Any:
        """Descriptive alias for ``plan``."""

        return self.plan


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without HF extra
        raise RuntimeError(
            "PyTorch is required for neural_forge.hf_transport_math."
        ) from exc
    return torch


def _validate_positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShapeValidationError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ShapeValidationError(f"{name} cannot exceed {maximum}.")
    return value


def _validate_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ShapeValidationError(f"{name} must be a finite non-negative float.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ShapeValidationError(
            f"{name} must be a finite non-negative float."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ShapeValidationError(f"{name} must be a finite non-negative float.")
    return result


def _validate_positive_float(value: Any, name: str) -> float:
    result = _validate_nonnegative_float(value, name)
    if result == 0.0:
        raise ShapeValidationError(f"{name} must be greater than zero.")
    return result


def _as_float64_matrix(torch, value: Any, name: str):
    if not isinstance(value, torch.Tensor):
        raise ShapeValidationError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise ShapeValidationError(f"{name} must be a dense strided tensor.")
    if value.ndim != 2:
        raise ShapeValidationError(f"{name} must be two-dimensional.")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ShapeValidationError(f"{name} cannot have an empty dimension.")
    if not torch.is_floating_point(value):
        raise ShapeValidationError(f"{name} must have a real floating-point dtype.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise FiniteValidationError(f"{name} contains NaN or infinity.")
    return result


def _as_float64_vector(torch, value: Any, name: str, *, length: int):
    if not isinstance(value, torch.Tensor):
        raise ShapeValidationError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise ShapeValidationError(f"{name} must be a dense strided tensor.")
    if value.ndim != 1 or value.shape[0] != length:
        raise ShapeValidationError(f"{name} must have shape ({length},).")
    if not torch.is_floating_point(value):
        raise ShapeValidationError(f"{name} must have a real floating-point dtype.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise FiniteValidationError(f"{name} contains NaN or infinity.")
    return result


def _max_abs(torch, value: Any) -> float:
    return float(torch.max(torch.abs(value)).item())


def _max_abs_error(torch, left: Any, right: Any) -> float:
    return _max_abs(torch, left - right)


def _canonicalize_basis_signs(torch, basis: Any):
    column_indices = torch.arange(basis.shape[1], dtype=torch.long)
    pivot_rows = torch.argmax(torch.abs(basis), dim=0)
    pivot_values = basis[pivot_rows, column_indices]
    signs = torch.where(
        pivot_values < 0.0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return (basis * signs.unsqueeze(0)).contiguous()


def _coordinate_diagnostics(torch, decode: Any, encode: Any, projector: Any):
    target_dim = int(decode.shape[1])
    identity = torch.eye(target_dim, dtype=torch.float64)
    decode_pseudoinverse = torch.linalg.pinv(decode)
    encode_pseudoinverse = torch.linalg.pinv(encode)
    values = (
        decode,
        encode,
        projector,
        decode_pseudoinverse,
        encode_pseudoinverse,
    )
    finite = all(bool(torch.isfinite(value).all()) for value in values)
    return PseudoinverseDiagnostics(
        rank=int(torch.linalg.matrix_rank(decode).item()),
        expected_rank=target_dim,
        decode_pseudoinverse_max_abs_error=_max_abs_error(
            torch, decode_pseudoinverse, encode
        ),
        encode_pseudoinverse_max_abs_error=_max_abs_error(
            torch, encode_pseudoinverse, decode
        ),
        target_identity_max_abs_error=_max_abs_error(torch, encode @ decode, identity),
        source_projector_symmetry_max_abs_error=_max_abs_error(
            torch, projector, projector.T
        ),
        source_projector_idempotence_max_abs_error=_max_abs_error(
            torch, projector @ projector, projector
        ),
        finite=finite,
    )


def build_parent_residual_basis(
    activation_rows: Any,
    target_dim: int,
    *,
    center: bool = True,
    rank_atol: float = 0.0,
    rank_rtol: float | None = None,
) -> CoordinateMap:
    """Fit a deterministic target basis from parent activation rows.

    The fit is always performed on detached CPU float64 data.  Right singular
    vectors are ordered by singular value, and each vector's sign is fixed by
    requiring its largest-absolute loading (first on ties) to be positive.
    Rank-deficient requests are rejected instead of padded with arbitrary axes.
    """

    torch = _require_torch()
    with torch.no_grad():
        rows = _as_float64_matrix(torch, activation_rows, "activation_rows")
        target_dim = _validate_positive_int(target_dim, "target_dim")
        source_dim = int(rows.shape[1])
        if target_dim > source_dim:
            raise ShapeValidationError(
                "target_dim cannot exceed the parent activation width."
            )
        if not isinstance(center, bool):
            raise ShapeValidationError("center must be a boolean.")
        rank_atol = _validate_nonnegative_float(rank_atol, "rank_atol")
        if rank_rtol is None:
            rank_rtol = max(rows.shape) * torch.finfo(torch.float64).eps
        else:
            rank_rtol = _validate_nonnegative_float(rank_rtol, "rank_rtol")

        fit_mean = (
            rows.mean(dim=0) if center else torch.zeros(source_dim, dtype=torch.float64)
        )
        if not bool(torch.isfinite(fit_mean).all()):
            raise FiniteValidationError("The activation mean is non-finite.")
        fitted_rows = rows - fit_mean if center else rows
        if not bool(torch.isfinite(fitted_rows).all()):
            raise FiniteValidationError("Centered activation rows are non-finite.")

        try:
            _, singular_values, right_vectors_t = torch.linalg.svd(
                fitted_rows,
                full_matrices=False,
            )
        except RuntimeError as exc:
            raise RankValidationError("Float64 activation SVD failed.") from exc
        if not bool(torch.isfinite(singular_values).all()) or not bool(
            torch.isfinite(right_vectors_t).all()
        ):
            raise FiniteValidationError("Activation SVD produced non-finite values.")

        largest = float(singular_values[0].item())
        rank_tolerance = max(rank_atol, largest * rank_rtol)
        if not math.isfinite(rank_tolerance):
            raise FiniteValidationError("The computed rank tolerance is non-finite.")
        effective_rank = int((singular_values > rank_tolerance).sum().item())
        if effective_rank < target_dim:
            raise RankValidationError(
                "Parent activations have effective rank "
                f"{effective_rank}, below requested target_dim {target_dim} "
                f"at tolerance {rank_tolerance:.6g}."
            )

        decode = _canonicalize_basis_signs(
            torch,
            right_vectors_t[:target_dim].T.contiguous(),
        )
        encode = decode.T.contiguous()
        projector = (decode @ encode).contiguous()
        diagnostics = _coordinate_diagnostics(torch, decode, encode, projector)
        if not diagnostics.finite:
            raise FiniteValidationError(
                "Coordinate-map pseudoinverse diagnostics are non-finite."
            )
        if diagnostics.rank != target_dim:
            raise RankValidationError(
                "The fitted decoding basis lost full target-column rank."
            )
        if diagnostics.target_identity_max_abs_error > _COORDINATE_ATOL:
            raise RankValidationError(
                "The fitted decoding basis is not orthonormal within tolerance."
            )

        scaled = singular_values / singular_values[0]
        scaled_energy = scaled.square()
        explained_variance_fraction = float(
            (scaled_energy[:target_dim].sum() / scaled_energy.sum()).item()
        )
        if not math.isfinite(explained_variance_fraction):
            raise FiniteValidationError("Explained variance is non-finite.")

        return CoordinateMap(
            source_dim=source_dim,
            target_dim=target_dim,
            activation_row_count=int(rows.shape[0]),
            centered_for_fit=center,
            effective_rank=effective_rank,
            rank_tolerance=rank_tolerance,
            explained_variance_fraction=explained_variance_fraction,
            fit_mean=fit_mean.detach().contiguous(),
            singular_values=singular_values.detach().contiguous(),
            encode=encode.detach().contiguous(),
            decode=decode.detach().contiguous(),
            source_projector=projector.detach().contiguous(),
            diagnostics=diagnostics,
        )


def _validated_coordinate_tensors(torch, coordinates: Any, name: str):
    if not isinstance(coordinates, CoordinateMap):
        raise ShapeValidationError(f"{name} must be a CoordinateMap.")
    source_dim = _validate_positive_int(coordinates.source_dim, f"{name}.source_dim")
    target_dim = _validate_positive_int(coordinates.target_dim, f"{name}.target_dim")
    if target_dim > source_dim:
        raise ShapeValidationError(
            f"{name}.target_dim cannot exceed {name}.source_dim."
        )
    decode = _as_float64_matrix(torch, coordinates.decode, f"{name}.decode")
    encode = _as_float64_matrix(torch, coordinates.encode, f"{name}.encode")
    projector = _as_float64_matrix(
        torch,
        coordinates.source_projector,
        f"{name}.source_projector",
    )
    if tuple(decode.shape) != (source_dim, target_dim):
        raise ShapeValidationError(
            f"{name}.decode must have shape ({source_dim}, {target_dim})."
        )
    if tuple(encode.shape) != (target_dim, source_dim):
        raise ShapeValidationError(
            f"{name}.encode must have shape ({target_dim}, {source_dim})."
        )
    if tuple(projector.shape) != (source_dim, source_dim):
        raise ShapeValidationError(
            f"{name}.source_projector must have shape ({source_dim}, {source_dim})."
        )
    if not torch.equal(encode, decode.T.contiguous()):
        raise PathConsistencyError(f"{name}.encode must equal {name}.decode.T exactly.")
    expected_projector = decode @ encode
    if _max_abs_error(torch, projector, expected_projector) > _COORDINATE_ATOL:
        raise PathConsistencyError(
            f"{name}.source_projector does not match decode @ encode."
        )
    identity = torch.eye(target_dim, dtype=torch.float64)
    if _max_abs_error(torch, encode @ decode, identity) > _COORDINATE_ATOL:
        raise RankValidationError(f"{name}.decode does not have orthonormal columns.")
    if int(torch.linalg.matrix_rank(decode).item()) != target_dim:
        raise RankValidationError(f"{name}.decode is rank deficient.")
    if not coordinates.diagnostics.finite:
        raise FiniteValidationError(f"{name} records non-finite diagnostics.")
    return decode, encode, projector


def encode_coordinates(source_rows: Any, coordinates: CoordinateMap):
    """Encode row-major source activations into target coordinates."""

    torch = _require_torch()
    with torch.no_grad():
        rows = _as_float64_matrix(torch, source_rows, "source_rows")
        decode, _, _ = _validated_coordinate_tensors(torch, coordinates, "coordinates")
        if rows.shape[1] != coordinates.source_dim:
            raise ShapeValidationError(
                "source_rows width does not match coordinates.source_dim."
            )
        result = rows @ decode
        if not bool(torch.isfinite(result).all()):
            raise FiniteValidationError("Encoded coordinates are non-finite.")
        return result.detach().contiguous()


def decode_coordinates(target_rows: Any, coordinates: CoordinateMap):
    """Decode row-major target coordinates into the parent/source space."""

    torch = _require_torch()
    with torch.no_grad():
        rows = _as_float64_matrix(torch, target_rows, "target_rows")
        _, encode, _ = _validated_coordinate_tensors(torch, coordinates, "coordinates")
        if rows.shape[1] != coordinates.target_dim:
            raise ShapeValidationError(
                "target_rows width does not match coordinates.target_dim."
            )
        result = rows @ encode
        if not bool(torch.isfinite(result).all()):
            raise FiniteValidationError("Decoded coordinates are non-finite.")
        return result.detach().contiguous()


def _transport_formula(torch, weight: Any, input_decode: Any, output_encode: Any):
    result = output_encode @ weight @ input_decode
    if not bool(torch.isfinite(result).all()):
        raise FiniteValidationError("Transported linear weight is non-finite.")
    return result.detach().contiguous()


def transport_linear_operator(
    source_weight: Any,
    input_coordinates: CoordinateMap,
    output_coordinates: CoordinateMap,
    *,
    source_bias: Any | None = None,
) -> LinearTransportResult:
    """Represent a source linear operator in target input/output coordinates."""

    torch = _require_torch()
    with torch.no_grad():
        weight = _as_float64_matrix(torch, source_weight, "source_weight")
        input_decode, _, _ = _validated_coordinate_tensors(
            torch,
            input_coordinates,
            "input_coordinates",
        )
        _, output_encode, _ = _validated_coordinate_tensors(
            torch,
            output_coordinates,
            "output_coordinates",
        )
        expected_shape = (
            output_coordinates.source_dim,
            input_coordinates.source_dim,
        )
        if tuple(weight.shape) != expected_shape:
            raise ShapeValidationError(
                f"source_weight must have shape {expected_shape}, got "
                f"{tuple(weight.shape)}."
            )

        target_weight = _transport_formula(
            torch,
            weight,
            input_decode,
            output_encode,
        )
        target_bias = None
        if source_bias is not None:
            bias = _as_float64_vector(
                torch,
                source_bias,
                "source_bias",
                length=output_coordinates.source_dim,
            )
            target_bias = (output_encode @ bias).detach().contiguous()
            if not bool(torch.isfinite(target_bias).all()):
                raise FiniteValidationError("Transported linear bias is non-finite.")

        diagnostics = LinearTransportDiagnostics(
            source_weight_shape=expected_shape,
            target_weight_shape=(
                output_coordinates.target_dim,
                input_coordinates.target_dim,
            ),
            source_bias_present=source_bias is not None,
            target_bias_shape=(
                None if target_bias is None else tuple(target_bias.shape)
            ),
            input_source_dim=input_coordinates.source_dim,
            input_target_dim=input_coordinates.target_dim,
            output_source_dim=output_coordinates.source_dim,
            output_target_dim=output_coordinates.target_dim,
            finite=True,
        )
        return LinearTransportResult(
            weight=target_weight,
            bias=target_bias,
            input_coordinates=input_coordinates,
            output_coordinates=output_coordinates,
            diagnostics=diagnostics,
        )


def _validated_linear_result(torch, result: Any, name: str):
    if not isinstance(result, LinearTransportResult):
        raise ShapeValidationError(f"{name} must be a LinearTransportResult.")
    input_tensors = _validated_coordinate_tensors(
        torch,
        result.input_coordinates,
        f"{name}.input_coordinates",
    )
    output_tensors = _validated_coordinate_tensors(
        torch,
        result.output_coordinates,
        f"{name}.output_coordinates",
    )
    weight = _as_float64_matrix(torch, result.weight, f"{name}.weight")
    expected_shape = (
        result.output_coordinates.target_dim,
        result.input_coordinates.target_dim,
    )
    if tuple(weight.shape) != expected_shape:
        raise ShapeValidationError(f"{name}.weight must have shape {expected_shape}.")
    bias = None
    if result.bias is not None:
        bias = _as_float64_vector(
            torch,
            result.bias,
            f"{name}.bias",
            length=result.output_coordinates.target_dim,
        )
    if not result.diagnostics.finite:
        raise FiniteValidationError(f"{name} records non-finite diagnostics.")
    return weight, bias, input_tensors, output_tensors


def _assert_formula_close(
    torch,
    actual: Any,
    expected: Any,
    name: str,
    *,
    atol: float,
    rtol: float,
) -> float:
    error = _max_abs_error(torch, actual, expected)
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise PathConsistencyError(
            f"{name} violates the shared-coordinate transport formula; "
            f"max absolute error is {error:.6g}."
        )
    return error


def check_path_composition(
    producer: LinearTransportResult,
    consumer: LinearTransportResult,
    *,
    producer_source_weight: Any,
    consumer_source_weight: Any,
    producer_source_bias: Any | None = None,
    consumer_source_bias: Any | None = None,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> PathCompositionDiagnostics:
    """Verify two transported linears reuse one boundary map and compose.

    Boundary sharing is defined by exact equality of the explicit encode,
    decode, and projector tensors.  This remains valid after serialization and
    rejects independently fitted, merely shape-compatible boundary maps.
    """

    torch = _require_torch()
    with torch.no_grad():
        atol = _validate_nonnegative_float(atol, "atol")
        rtol = _validate_nonnegative_float(rtol, "rtol")
        (
            producer_weight,
            producer_bias,
            producer_input_tensors,
            producer_output_tensors,
        ) = _validated_linear_result(torch, producer, "producer")
        (
            consumer_weight,
            consumer_bias,
            consumer_input_tensors,
            consumer_output_tensors,
        ) = _validated_linear_result(torch, consumer, "consumer")

        producer_input_decode, _, _ = producer_input_tensors
        (
            producer_boundary_decode,
            producer_boundary_encode,
            producer_boundary_projector,
        ) = producer_output_tensors
        (
            consumer_boundary_decode,
            consumer_boundary_encode,
            consumer_boundary_projector,
        ) = consumer_input_tensors
        _, consumer_output_encode, _ = consumer_output_tensors

        producer_boundary = producer.output_coordinates
        consumer_boundary = consumer.input_coordinates
        same_shape = (
            producer_boundary.source_dim == consumer_boundary.source_dim
            and producer_boundary.target_dim == consumer_boundary.target_dim
        )
        if same_shape:
            boundary_map_error = max(
                _max_abs_error(
                    torch,
                    producer_boundary_decode,
                    consumer_boundary_decode,
                ),
                _max_abs_error(
                    torch,
                    producer_boundary_encode,
                    consumer_boundary_encode,
                ),
                _max_abs_error(
                    torch,
                    producer_boundary_projector,
                    consumer_boundary_projector,
                ),
            )
            shared_boundary_map = (
                torch.equal(
                    producer_boundary_decode,
                    consumer_boundary_decode,
                )
                and torch.equal(
                    producer_boundary_encode,
                    consumer_boundary_encode,
                )
                and torch.equal(
                    producer_boundary_projector,
                    consumer_boundary_projector,
                )
            )
        else:
            boundary_map_error = math.inf
            shared_boundary_map = False
        if not shared_boundary_map:
            raise PathConsistencyError(
                "Producer output and consumer input must reuse one shared "
                "boundary coordinate map; independently fitted maps are rejected."
            )

        first_source_weight = _as_float64_matrix(
            torch,
            producer_source_weight,
            "producer_source_weight",
        )
        second_source_weight = _as_float64_matrix(
            torch,
            consumer_source_weight,
            "consumer_source_weight",
        )
        first_expected_shape = (
            producer.output_coordinates.source_dim,
            producer.input_coordinates.source_dim,
        )
        second_expected_shape = (
            consumer.output_coordinates.source_dim,
            consumer.input_coordinates.source_dim,
        )
        if tuple(first_source_weight.shape) != first_expected_shape:
            raise ShapeValidationError(
                f"producer_source_weight must have shape {first_expected_shape}."
            )
        if tuple(second_source_weight.shape) != second_expected_shape:
            raise ShapeValidationError(
                f"consumer_source_weight must have shape {second_expected_shape}."
            )

        expected_producer_weight = (
            producer_boundary_encode @ first_source_weight @ producer_input_decode
        )
        expected_consumer_weight = (
            consumer_output_encode @ second_source_weight @ consumer_boundary_decode
        )
        producer_formula_error = _assert_formula_close(
            torch,
            producer_weight,
            expected_producer_weight,
            "producer.weight",
            atol=atol,
            rtol=rtol,
        )
        consumer_formula_error = _assert_formula_close(
            torch,
            consumer_weight,
            expected_consumer_weight,
            "consumer.weight",
            atol=atol,
            rtol=rtol,
        )

        if (producer_source_bias is None) != (producer_bias is None):
            raise PathConsistencyError(
                "producer source/transported bias presence does not match."
            )
        if (consumer_source_bias is None) != (consumer_bias is None):
            raise PathConsistencyError(
                "consumer source/transported bias presence does not match."
            )
        first_source_bias = torch.zeros(
            producer.output_coordinates.source_dim,
            dtype=torch.float64,
        )
        second_source_bias = torch.zeros(
            consumer.output_coordinates.source_dim,
            dtype=torch.float64,
        )
        if producer_source_bias is not None:
            first_source_bias = _as_float64_vector(
                torch,
                producer_source_bias,
                "producer_source_bias",
                length=producer.output_coordinates.source_dim,
            )
            expected_producer_bias = producer_boundary_encode @ first_source_bias
            _assert_formula_close(
                torch,
                producer_bias,
                expected_producer_bias,
                "producer.bias",
                atol=atol,
                rtol=rtol,
            )
        if consumer_source_bias is not None:
            second_source_bias = _as_float64_vector(
                torch,
                consumer_source_bias,
                "consumer_source_bias",
                length=consumer.output_coordinates.source_dim,
            )
            expected_consumer_bias = consumer_output_encode @ second_source_bias
            _assert_formula_close(
                torch,
                consumer_bias,
                expected_consumer_bias,
                "consumer.bias",
                atol=atol,
                rtol=rtol,
            )

        boundary_projector = producer_boundary_projector
        expected_composed_weight = (
            consumer_output_encode
            @ second_source_weight
            @ boundary_projector
            @ first_source_weight
            @ producer_input_decode
        )
        actual_composed_weight = consumer_weight @ producer_weight
        composed_weight_error = _assert_formula_close(
            torch,
            actual_composed_weight,
            expected_composed_weight,
            "composed weight",
            atol=atol,
            rtol=rtol,
        )

        actual_composed_bias = torch.zeros(
            consumer.output_coordinates.target_dim,
            dtype=torch.float64,
        )
        if producer_bias is not None:
            actual_composed_bias = (
                actual_composed_bias + consumer_weight @ producer_bias
            )
        if consumer_bias is not None:
            actual_composed_bias = actual_composed_bias + consumer_bias
        expected_composed_bias = consumer_output_encode @ (
            second_source_weight @ (boundary_projector @ first_source_bias)
            + second_source_bias
        )
        composed_bias_error = _assert_formula_close(
            torch,
            actual_composed_bias,
            expected_composed_bias,
            "composed bias",
            atol=atol,
            rtol=rtol,
        )

        finite = all(
            bool(torch.isfinite(value).all())
            for value in (
                actual_composed_weight,
                expected_composed_weight,
                actual_composed_bias,
                expected_composed_bias,
            )
        )
        if not finite:
            raise FiniteValidationError("Path composition produced non-finite values.")
        return PathCompositionDiagnostics(
            shared_boundary_map=True,
            boundary_source_dim=producer_boundary.source_dim,
            boundary_target_dim=producer_boundary.target_dim,
            boundary_projector_rank=int(
                torch.linalg.matrix_rank(boundary_projector).item()
            ),
            boundary_is_square=(
                producer_boundary.source_dim == producer_boundary.target_dim
            ),
            boundary_map_max_abs_error=boundary_map_error,
            producer_formula_max_abs_error=producer_formula_error,
            consumer_formula_max_abs_error=consumer_formula_error,
            composed_weight_max_abs_error=composed_weight_error,
            composed_bias_max_abs_error=composed_bias_error,
            finite=True,
            passed=True,
        )


def _sinkhorn_diagnostics(
    *,
    iterations: int,
    max_iterations: int,
    converged: bool,
    finite: bool,
    epsilon: float,
    tolerance: float,
    total_mass: float,
    source_error: float,
    target_error: float,
    cost_min: float,
    cost_max: float,
) -> SinkhornDiagnostics:
    return SinkhornDiagnostics(
        iterations=iterations,
        max_iterations=max_iterations,
        converged=converged,
        finite=finite,
        epsilon=epsilon,
        tolerance=tolerance,
        total_mass=total_mass,
        mass_error=abs(total_mass - 1.0),
        max_source_marginal_error=source_error,
        max_target_marginal_error=target_error,
        cost_min=cost_min,
        cost_max=cost_max,
    )


def uniform_entropic_sinkhorn(
    source_signatures: Any,
    target_signatures: Any,
    *,
    epsilon: float = 0.1,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
) -> SinkhornResult:
    """Solve balanced uniform entropic OT in the log domain.

    Rows of each input are activation signatures.  The cost is Euclidean
    distance, the plan shape is ``(source_count, target_count)``, and both
    marginals are uniform.  Non-convergence after the explicit bounded iteration
    budget raises ``SinkhornConvergenceError`` with the final diagnostics.
    """

    torch = _require_torch()
    with torch.no_grad():
        source = _as_float64_matrix(
            torch,
            source_signatures,
            "source_signatures",
        )
        target = _as_float64_matrix(
            torch,
            target_signatures,
            "target_signatures",
        )
        if source.shape[1] != target.shape[1]:
            raise ShapeValidationError(
                "Source and target signatures must have the same feature width."
            )
        epsilon = _validate_positive_float(epsilon, "epsilon")
        tolerance = _validate_positive_float(tolerance, "tolerance")
        max_iterations = _validate_positive_int(
            max_iterations,
            "max_iterations",
            maximum=MAX_SINKHORN_ITERATIONS,
        )

        cost = torch.cdist(source, target, p=2).contiguous()
        if not bool(torch.isfinite(cost).all()):
            raise FiniteValidationError("Sinkhorn cost matrix is non-finite.")
        source_count = int(source.shape[0])
        target_count = int(target.shape[0])
        source_marginal = torch.full(
            (source_count,),
            1.0 / source_count,
            dtype=torch.float64,
        )
        target_marginal = torch.full(
            (target_count,),
            1.0 / target_count,
            dtype=torch.float64,
        )
        log_source = torch.full(
            (source_count,),
            -math.log(source_count),
            dtype=torch.float64,
        )
        log_target = torch.full(
            (target_count,),
            -math.log(target_count),
            dtype=torch.float64,
        )
        log_kernel = -cost / epsilon
        if not bool(torch.isfinite(log_kernel).all()):
            raise FiniteValidationError("Sinkhorn log-kernel is non-finite.")

        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)
        plan = torch.empty_like(cost)
        source_error = math.inf
        target_error = math.inf
        total_mass = math.nan
        converged = False
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(0),
                dim=1,
            )
            log_v = log_target - torch.logsumexp(
                log_kernel + log_u.unsqueeze(1),
                dim=0,
            )
            log_plan = log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
            if not bool(torch.isfinite(log_plan).all()):
                raise FiniteValidationError(
                    "Sinkhorn dual updates produced non-finite values."
                )
            plan = torch.exp(log_plan)
            if not bool(torch.isfinite(plan).all()):
                raise FiniteValidationError("Sinkhorn plan is non-finite.")
            actual_source = plan.sum(dim=1)
            actual_target = plan.sum(dim=0)
            source_error = _max_abs_error(
                torch,
                actual_source,
                source_marginal,
            )
            target_error = _max_abs_error(
                torch,
                actual_target,
                target_marginal,
            )
            total_mass = float(plan.sum().item())
            if max(source_error, target_error) <= tolerance:
                converged = True
                break

        finite = all(
            math.isfinite(value)
            for value in (
                total_mass,
                source_error,
                target_error,
            )
        ) and bool(torch.isfinite(plan).all())
        diagnostics = _sinkhorn_diagnostics(
            iterations=iterations,
            max_iterations=max_iterations,
            converged=converged,
            finite=finite,
            epsilon=epsilon,
            tolerance=tolerance,
            total_mass=total_mass,
            source_error=source_error,
            target_error=target_error,
            cost_min=float(cost.min().item()),
            cost_max=float(cost.max().item()),
        )
        if not diagnostics.finite:
            raise FiniteValidationError(
                "Sinkhorn terminated with non-finite mass diagnostics."
            )
        if not diagnostics.converged:
            raise SinkhornConvergenceError(
                "Sinkhorn did not satisfy both uniform marginals within "
                f"{max_iterations} iterations.",
                diagnostics,
            )
        return SinkhornResult(
            plan=plan.detach().contiguous(),
            cost_matrix=cost.detach().contiguous(),
            source_marginal=source_marginal.detach().contiguous(),
            target_marginal=target_marginal.detach().contiguous(),
            diagnostics=diagnostics,
        )


__all__ = [
    "CoordinateMap",
    "FiniteValidationError",
    "LinearTransportDiagnostics",
    "LinearTransportResult",
    "MAX_SINKHORN_ITERATIONS",
    "PathCompositionDiagnostics",
    "PathConsistencyError",
    "PseudoinverseDiagnostics",
    "RankValidationError",
    "ShapeValidationError",
    "SinkhornConvergenceError",
    "SinkhornDiagnostics",
    "SinkhornResult",
    "TransportMathError",
    "build_parent_residual_basis",
    "check_path_composition",
    "decode_coordinates",
    "encode_coordinates",
    "transport_linear_operator",
    "uniform_entropic_sinkhorn",
]
