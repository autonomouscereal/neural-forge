"""Deterministic balanced coordinates for externally sampled rollout dynamics.

The builder consumes only matrices already collected by a parent model.  It
does not run a model, construct an optimizer, invoke backward, or retain an
autograd graph.  Activation and forward-response rows form an empirical
controllability sketch.  Output/dual sensitivity rows form an observability
sketch.  When both sketches support the requested rank, the implementation
uses snapshot balanced POD; otherwise it can fall back to a deterministic,
orthonormal SVD basis over the available feature-space sketches.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


REPORT_SCHEMA_VERSION = 1
_FLOAT64_EPS = np.finfo(np.float64).eps
_BIORTHOGONALITY_ATOL = 1e-9


class BalancedKrylovError(ValueError):
    """Base class for balanced-basis construction failures."""


class BalancedKrylovShapeError(BalancedKrylovError):
    """An input matrix or control has an invalid shape or type."""


class BalancedKrylovFiniteError(BalancedKrylovError):
    """An input, decomposition, or diagnostic is non-finite."""


class BalancedKrylovRankError(BalancedKrylovError):
    """The supplied sketches cannot support the requested rank."""


class BalancedKrylovConditionError(BalancedKrylovError):
    """A balanced realization is too ill-conditioned to use safely."""


# Descriptive aliases make the validation categories easy to consume without
# coupling callers to the module prefix.
ShapeValidationError = BalancedKrylovShapeError
FiniteValidationError = BalancedKrylovFiniteError
RankValidationError = BalancedKrylovRankError
ConditionValidationError = BalancedKrylovConditionError


@dataclass(frozen=True, slots=True)
class RolloutErrorReport:
    """Finite aggregate and per-step errors for one reduced linear rollout."""

    transition_count: int
    evaluated_step_count: int
    include_initial: bool
    relative_l2_error: float
    root_mean_squared_error: float
    maximum_absolute_error: float
    reference_l2_norm: float
    per_step_relative_l2_error: tuple[float, ...]
    finite: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible rollout diagnostics."""

        return {
            "transition_count": self.transition_count,
            "evaluated_step_count": self.evaluated_step_count,
            "include_initial": self.include_initial,
            "relative_l2_error": self.relative_l2_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "reference_l2_norm": self.reference_l2_norm,
            "per_step_relative_l2_error": list(self.per_step_relative_l2_error),
            "finite": self.finite,
        }


@dataclass(frozen=True, slots=True)
class BalancedKrylovReport:
    """Auditable evidence for one deterministic basis build."""

    schema_version: int
    method: str
    state_dimension: int
    rank: int
    activation_row_count: int
    forward_response_row_count: int
    output_sensitivity_row_count: int
    centered_activations: bool
    controllability_rank: int
    observability_rank: int
    hankel_rank: int
    controllability_rank_tolerance: float
    observability_rank_tolerance: float | None
    hankel_rank_tolerance: float | None
    hankel_singular_values: tuple[float, ...]
    selected_hankel_condition_number: float | None
    decoder_condition_number: float
    encoder_condition_number: float
    biorthogonality_max_abs_error: float
    projection_idempotence_max_abs_error: float
    captured_hankel_energy_fraction: float | None
    captured_controllability_energy_fraction: float
    captured_observability_energy_fraction: float | None
    activation_relative_reconstruction_error: float
    controllability_relative_reconstruction_error: float
    observability_relative_reconstruction_error: float | None
    fallback_used: bool
    fallback_reason: str | None
    canonical_pivot_indices: tuple[int, ...]
    canonical_pivot_values: tuple[float, ...]
    activation_sha256: str
    forward_response_sha256: str | None
    output_sensitivity_sha256: str | None
    controllability_sketch_sha256: str
    observability_sketch_sha256: str | None
    encoder_sha256: str
    decoder_sha256: str
    basis_sha256: str
    provenance_sha256: str
    finite: bool = True
    shape_checks_passed: bool = True
    deterministic: bool = True
    backward_used: bool = False
    backward_operations: int = 0
    optimizer_used: bool = False
    optimizer_steps: int = 0

    @property
    def zero_gradient(self) -> bool:
        """Whether construction was explicitly free of backward operations."""

        return not self.backward_used and self.backward_operations == 0

    @property
    def no_training(self) -> bool:
        """Whether no optimizer or backward operation occurred."""

        return (
            self.zero_gradient and not self.optimizer_used and self.optimizer_steps == 0
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible report."""

        return {
            "schema_version": self.schema_version,
            "kind": "balanced-krylov-basis",
            "method": self.method,
            "state_dimension": self.state_dimension,
            "rank": self.rank,
            "row_counts": {
                "activation": self.activation_row_count,
                "forward_response": self.forward_response_row_count,
                "output_sensitivity": self.output_sensitivity_row_count,
            },
            "centered_activations": self.centered_activations,
            "ranks": {
                "controllability": self.controllability_rank,
                "observability": self.observability_rank,
                "hankel": self.hankel_rank,
            },
            "rank_tolerances": {
                "controllability": self.controllability_rank_tolerance,
                "observability": self.observability_rank_tolerance,
                "hankel": self.hankel_rank_tolerance,
            },
            "hankel_singular_values": list(self.hankel_singular_values),
            "conditioning": {
                "selected_hankel": self.selected_hankel_condition_number,
                "decoder": self.decoder_condition_number,
                "encoder": self.encoder_condition_number,
                "biorthogonality_max_abs_error": (self.biorthogonality_max_abs_error),
                "projection_idempotence_max_abs_error": (
                    self.projection_idempotence_max_abs_error
                ),
            },
            "captured_energy": {
                "hankel": self.captured_hankel_energy_fraction,
                "controllability": (self.captured_controllability_energy_fraction),
                "observability": self.captured_observability_energy_fraction,
            },
            "relative_reconstruction_error": {
                "activation": self.activation_relative_reconstruction_error,
                "controllability": (self.controllability_relative_reconstruction_error),
                "observability": (self.observability_relative_reconstruction_error),
            },
            "fallback": {
                "used": self.fallback_used,
                "reason": self.fallback_reason,
            },
            "canonical_signs": {
                "pivot_indices": list(self.canonical_pivot_indices),
                "pivot_values": list(self.canonical_pivot_values),
            },
            "hashes": {
                "activation": self.activation_sha256,
                "forward_response": self.forward_response_sha256,
                "output_sensitivity": self.output_sensitivity_sha256,
                "controllability_sketch": self.controllability_sketch_sha256,
                "observability_sketch": self.observability_sketch_sha256,
                "encoder": self.encoder_sha256,
                "decoder": self.decoder_sha256,
                "basis": self.basis_sha256,
                "provenance": self.provenance_sha256,
            },
            "validation": {
                "finite": self.finite,
                "shape_checks_passed": self.shape_checks_passed,
            },
            "execution": {
                "deterministic": self.deterministic,
                "backward_used": self.backward_used,
                "backward_operations": self.backward_operations,
                "optimizer_used": self.optimizer_used,
                "optimizer_steps": self.optimizer_steps,
                "zero_gradient": self.zero_gradient,
                "no_training": self.no_training,
            },
            "finite": self.finite,
            "deterministic": self.deterministic,
            "backward_used": self.backward_used,
            "optimizer_used": self.optimizer_used,
            "zero_gradient": self.zero_gradient,
            "no_training": self.no_training,
        }


@dataclass(frozen=True, slots=True)
class BalancedKrylovBasis:
    """A parent-state encoder/decoder pair with ``encoder @ decoder == I``.

    ``encoder`` has shape ``[rank, state_dimension]`` and ``decoder`` has
    shape ``[state_dimension, rank]``.  For row-major states, encoding is
    ``(states - mean) @ encoder.T`` and decoding is
    ``coordinates @ decoder.T + mean``.
    """

    encoder: np.ndarray
    decoder: np.ndarray
    mean: np.ndarray
    report: BalancedKrylovReport

    @property
    def rank(self) -> int:
        return self.report.rank

    @property
    def state_dimension(self) -> int:
        return self.report.state_dimension

    @property
    def basis(self) -> np.ndarray:
        """Return the state-space decoding basis."""

        return self.decoder

    @property
    def projector(self) -> np.ndarray:
        """Return the generally oblique state-space projector."""

        return self.decoder @ self.encoder

    def encode(self, states: Any) -> np.ndarray:
        """Encode one state or row-major state matrix without autograd."""

        rows, was_vector = _runtime_rows(
            states,
            "states",
            width=self.state_dimension,
        )
        encoded = (rows - self.mean) @ self.encoder.T
        _require_finite(encoded, "encoded states")
        return encoded[0] if was_vector else encoded

    def decode(self, coordinates: Any) -> np.ndarray:
        """Decode one coordinate vector or row-major coordinate matrix."""

        rows, was_vector = _runtime_rows(
            coordinates,
            "coordinates",
            width=self.rank,
        )
        decoded = rows @ self.decoder.T + self.mean
        _require_finite(decoded, "decoded states")
        return decoded[0] if was_vector else decoded

    def reconstruct(self, states: Any) -> np.ndarray:
        """Project states through the encoder and decoder."""

        return self.decode(self.encode(states))

    def as_dict(self) -> dict[str, Any]:
        """Return diagnostics without serializing the dense basis matrices."""

        return self.report.as_dict()


def canonical_array_sha256(value: Any) -> str:
    """Hash a finite floating array in canonical little-endian float64 form."""

    array = _as_float64_array(value, "value")
    return _hash_array(array)


def build_balanced_krylov_basis(
    activation_snapshots: Any,
    rank: int,
    *,
    forward_responses: Any | None = None,
    forward_jvp_responses: Any | None = None,
    krylov_responses: Any | None = None,
    output_sensitivities: Any | None = None,
    dual_sensitivities: Any | None = None,
    center: bool = True,
    activation_weight: float = 1.0,
    forward_weight: float = 1.0,
    sensitivity_weight: float = 1.0,
    svd_atol: float = 0.0,
    svd_rtol: float | None = None,
    max_condition_number: float = 1e10,
    allow_svd_fallback: bool = True,
) -> BalancedKrylovBasis:
    """Build balanced rank-``rank`` coordinates from parent-owned snapshots.

    Matrix inputs are row-major ``[samples, state_dimension]`` arrays or Torch
    tensors.  A non-empty sequence of such matrices is also accepted.  Forward
    aliases are concatenated into one controllability response block; output
    and dual aliases are concatenated into one observability block.

    Balanced POD is computed from compact controllability/observability Gramian
    factors, avoiding a potentially enormous all-pairs snapshot Hankel matrix.
    If output sensitivities are absent, or the Hankel factor is rank-deficient
    or unsafe, the optional fallback uses a deterministic feature-space SVD.
    """

    activation_matrices = _matrix_group(
        activation_snapshots,
        "activation_snapshots",
        required=True,
    )
    forward_matrices = _combined_groups(
        (
            (forward_responses, "forward_responses"),
            (forward_jvp_responses, "forward_jvp_responses"),
            (krylov_responses, "krylov_responses"),
        )
    )
    sensitivity_matrices = _combined_groups(
        (
            (output_sensitivities, "output_sensitivities"),
            (dual_sensitivities, "dual_sensitivities"),
        )
    )

    state_dimension = int(activation_matrices[0].shape[1])
    _require_common_width(
        activation_matrices,
        state_dimension,
        "activation_snapshots",
    )
    _require_common_width(
        forward_matrices,
        state_dimension,
        "forward responses",
    )
    _require_common_width(
        sensitivity_matrices,
        state_dimension,
        "output/dual sensitivities",
    )
    rank = _positive_int(rank, "rank", maximum=state_dimension)
    if not isinstance(center, bool):
        raise BalancedKrylovShapeError("center must be a boolean.")
    if not isinstance(allow_svd_fallback, bool):
        raise BalancedKrylovShapeError("allow_svd_fallback must be a boolean.")
    activation_weight = _positive_float(
        activation_weight,
        "activation_weight",
    )
    forward_weight = _positive_float(forward_weight, "forward_weight")
    sensitivity_weight = _positive_float(
        sensitivity_weight,
        "sensitivity_weight",
    )
    svd_atol = _nonnegative_float(svd_atol, "svd_atol")
    if svd_rtol is not None:
        svd_rtol = _nonnegative_float(svd_rtol, "svd_rtol")
    max_condition_number = _finite_float(
        max_condition_number,
        "max_condition_number",
    )
    if max_condition_number < 1.0:
        raise BalancedKrylovShapeError("max_condition_number must be at least one.")

    activations = np.vstack(activation_matrices)
    mean = (
        np.mean(activations, axis=0, dtype=np.float64)
        if center
        else np.zeros(state_dimension, dtype=np.float64)
    )
    fitted_activations = activations - mean if center else activations.copy()
    _require_finite(fitted_activations, "centered activation snapshots")

    controllability_blocks = [_empirical_block(fitted_activations, activation_weight)]
    if forward_matrices:
        forward_rows = np.vstack(forward_matrices)
        controllability_blocks.append(_empirical_block(forward_rows, forward_weight))
    else:
        forward_rows = None
    controllability = np.ascontiguousarray(
        np.vstack(controllability_blocks),
        dtype=np.float64,
    )

    if sensitivity_matrices:
        sensitivity_rows = np.vstack(sensitivity_matrices)
        observability = np.ascontiguousarray(
            _empirical_block(sensitivity_rows, sensitivity_weight),
            dtype=np.float64,
        )
    else:
        sensitivity_rows = None
        observability = None

    controllability_svd = _safe_svd(
        controllability,
        "controllability sketch",
    )
    controllability_tolerance = _rank_tolerance(
        controllability_svd[1],
        controllability.shape,
        atol=svd_atol,
        rtol=svd_rtol,
    )
    controllability_rank = _effective_rank(
        controllability_svd[1],
        controllability_tolerance,
    )
    if controllability_rank < rank:
        raise BalancedKrylovRankError(
            "Controllability sketch has effective rank "
            f"{controllability_rank}, below requested rank {rank}."
        )

    method: str
    fallback_used = False
    fallback_reason: str | None = None
    observability_rank = 0
    observability_tolerance: float | None = None
    hankel_rank = 0
    hankel_tolerance: float | None = None
    hankel_singular_values: np.ndarray = np.empty(0, dtype=np.float64)
    selected_hankel_condition: float | None = None
    balanced_candidate: tuple[np.ndarray, np.ndarray] | None = None

    if observability is not None:
        observability_svd = _safe_svd(
            observability,
            "observability sketch",
        )
        observability_tolerance = _rank_tolerance(
            observability_svd[1],
            observability.shape,
            atol=svd_atol,
            rtol=svd_rtol,
        )
        observability_rank = _effective_rank(
            observability_svd[1],
            observability_tolerance,
        )
        if observability_rank > 0:
            primal_factor = _gramian_factor(
                controllability_svd,
                controllability_rank,
            )
            dual_factor = _gramian_factor(
                observability_svd,
                observability_rank,
            )
            hankel_factor = dual_factor.T @ primal_factor
            hankel_u, hankel_singular_values, hankel_vt = _safe_svd(
                hankel_factor,
                "compact Hankel factor",
            )
            hankel_tolerance = _rank_tolerance(
                hankel_singular_values,
                hankel_factor.shape,
                atol=svd_atol,
                rtol=svd_rtol,
            )
            hankel_rank = _effective_rank(
                hankel_singular_values,
                hankel_tolerance,
            )
            if hankel_rank >= rank:
                selected_hankel_condition = float(
                    hankel_singular_values[0] / hankel_singular_values[rank - 1]
                )
                if selected_hankel_condition <= max_condition_number:
                    try:
                        balanced_candidate = _balanced_modes(
                            primal_factor,
                            dual_factor,
                            hankel_u,
                            hankel_singular_values,
                            hankel_vt,
                            hankel_rank=hankel_rank,
                            rank=rank,
                            degeneracy_atol=svd_atol,
                        )
                    except (
                        BalancedKrylovConditionError,
                        BalancedKrylovRankError,
                    ) as exc:
                        fallback_reason = str(exc)
                else:
                    fallback_reason = (
                        "selected Hankel condition number "
                        f"{selected_hankel_condition:.6g} exceeds limit "
                        f"{max_condition_number:.6g}"
                    )
            else:
                fallback_reason = (
                    "compact Hankel factor has effective rank "
                    f"{hankel_rank}, below requested rank {rank}"
                )
        else:
            fallback_reason = "observability sketch has zero effective rank"
    else:
        fallback_reason = "output/dual sensitivities were not supplied"

    if balanced_candidate is not None:
        decoder, encoder = balanced_candidate
        candidate_reason = _unsafe_basis_reason(
            decoder,
            encoder,
            max_condition_number=max_condition_number,
        )
        if candidate_reason is None:
            method = "balanced_pod"
        else:
            fallback_reason = candidate_reason
            balanced_candidate = None

    if balanced_candidate is None:
        if not allow_svd_fallback:
            raise BalancedKrylovConditionError(
                "Balanced basis could not be constructed safely and SVD "
                f"fallback is disabled: {fallback_reason}."
            )
        fallback_used = True
        decoder, encoder, method = _fallback_basis(
            controllability,
            observability,
            rank=rank,
            atol=svd_atol,
            rtol=svd_rtol,
        )

    gram = encoder @ decoder
    if not np.all(np.isfinite(gram)):
        raise BalancedKrylovFiniteError(
            "Encoder/decoder product contains NaN or infinity."
        )
    if _max_abs(gram - np.eye(rank, dtype=np.float64)) > _BIORTHOGONALITY_ATOL:
        try:
            encoder = np.linalg.solve(gram, encoder)
        except np.linalg.LinAlgError as exc:
            raise BalancedKrylovConditionError(
                "Could not re-biorthogonalize the selected basis."
            ) from exc
    decoder, encoder, pivot_indices, pivot_values = _canonicalize_signs(
        decoder,
        encoder,
    )
    _validate_basis(decoder, encoder, rank, state_dimension)

    projection = decoder @ encoder
    biorthogonality_error = _max_abs(encoder @ decoder - np.eye(rank, dtype=np.float64))
    idempotence_error = _max_abs(projection @ projection - projection)
    decoder_condition = _matrix_condition_number(decoder)
    encoder_condition = _matrix_condition_number(encoder)
    if not all(
        math.isfinite(value)
        for value in (
            biorthogonality_error,
            idempotence_error,
            decoder_condition,
            encoder_condition,
        )
    ):
        raise BalancedKrylovFiniteError(
            "Basis conditioning diagnostics contain NaN or infinity."
        )

    activation_error = _relative_projection_error(
        fitted_activations,
        projection.T,
    )
    controllability_error = _relative_projection_error(
        controllability,
        projection.T,
    )
    observability_error = (
        _relative_projection_error(observability, projection)
        if observability is not None
        else None
    )
    controllability_capture = _subspace_energy_fraction(
        controllability,
        decoder,
    )
    observability_capture = (
        _subspace_energy_fraction(observability, encoder.T)
        if observability is not None
        else None
    )
    hankel_capture = _captured_spectral_energy(
        hankel_singular_values,
        rank,
    )

    activation_hash = _hash_group(activation_matrices)
    forward_hash = _hash_group(forward_matrices) if forward_matrices else None
    sensitivity_hash = (
        _hash_group(sensitivity_matrices) if sensitivity_matrices else None
    )
    controllability_hash = _hash_array(controllability)
    observability_hash = (
        _hash_array(observability) if observability is not None else None
    )
    encoder = _readonly(encoder)
    decoder = _readonly(decoder)
    mean = _readonly(mean)
    encoder_hash = _hash_array(encoder)
    decoder_hash = _hash_array(decoder)
    basis_hash = _hash_parts(
        "balanced-krylov-basis-v1",
        encoder_hash,
        decoder_hash,
        _hash_array(mean),
    )
    provenance_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "method": method,
        "rank": rank,
        "state_dimension": state_dimension,
        "center": center,
        "activation_weight": activation_weight,
        "forward_weight": forward_weight,
        "sensitivity_weight": sensitivity_weight,
        "svd_atol": svd_atol,
        "svd_rtol": svd_rtol,
        "max_condition_number": max_condition_number,
        "activation_sha256": activation_hash,
        "forward_response_sha256": forward_hash,
        "output_sensitivity_sha256": sensitivity_hash,
        "basis_sha256": basis_hash,
    }
    provenance_hash = _hash_parts(
        json.dumps(
            provenance_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )

    report = BalancedKrylovReport(
        schema_version=REPORT_SCHEMA_VERSION,
        method=method,
        state_dimension=state_dimension,
        rank=rank,
        activation_row_count=int(activations.shape[0]),
        forward_response_row_count=(
            int(forward_rows.shape[0]) if forward_rows is not None else 0
        ),
        output_sensitivity_row_count=(
            int(sensitivity_rows.shape[0]) if sensitivity_rows is not None else 0
        ),
        centered_activations=center,
        controllability_rank=controllability_rank,
        observability_rank=observability_rank,
        hankel_rank=hankel_rank,
        controllability_rank_tolerance=controllability_tolerance,
        observability_rank_tolerance=observability_tolerance,
        hankel_rank_tolerance=hankel_tolerance,
        hankel_singular_values=tuple(float(value) for value in hankel_singular_values),
        selected_hankel_condition_number=selected_hankel_condition,
        decoder_condition_number=decoder_condition,
        encoder_condition_number=encoder_condition,
        biorthogonality_max_abs_error=biorthogonality_error,
        projection_idempotence_max_abs_error=idempotence_error,
        captured_hankel_energy_fraction=hankel_capture,
        captured_controllability_energy_fraction=controllability_capture,
        captured_observability_energy_fraction=observability_capture,
        activation_relative_reconstruction_error=activation_error,
        controllability_relative_reconstruction_error=controllability_error,
        observability_relative_reconstruction_error=observability_error,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason if fallback_used else None,
        canonical_pivot_indices=pivot_indices,
        canonical_pivot_values=pivot_values,
        activation_sha256=activation_hash,
        forward_response_sha256=forward_hash,
        output_sensitivity_sha256=sensitivity_hash,
        controllability_sketch_sha256=controllability_hash,
        observability_sketch_sha256=observability_hash,
        encoder_sha256=encoder_hash,
        decoder_sha256=decoder_hash,
        basis_sha256=basis_hash,
        provenance_sha256=provenance_hash,
    )
    return BalancedKrylovBasis(
        encoder=encoder,
        decoder=decoder,
        mean=mean,
        report=report,
    )


def relative_reconstruction_error(
    snapshots: Any,
    basis: BalancedKrylovBasis,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return relative L2 error after encoding and decoding snapshots."""

    if not isinstance(basis, BalancedKrylovBasis):
        raise BalancedKrylovShapeError("basis must be a BalancedKrylovBasis.")
    epsilon = _positive_float(epsilon, "epsilon")
    rows, _ = _runtime_rows(
        snapshots,
        "snapshots",
        width=basis.state_dimension,
    )
    reference = rows - basis.mean
    reconstructed = reference @ basis.projector.T
    return _relative_array_error(reference, reconstructed, epsilon=epsilon)


def reconstruction_error(
    snapshots: Any,
    basis: BalancedKrylovBasis,
    *,
    relative: bool = True,
    epsilon: float = 1e-12,
) -> float:
    """Return relative L2 error, or absolute RMS error, for snapshots."""

    if not isinstance(relative, bool):
        raise BalancedKrylovShapeError("relative must be a boolean.")
    rows, _ = _runtime_rows(
        snapshots,
        "snapshots",
        width=basis.state_dimension,
    )
    reconstructed = basis.reconstruct(rows)
    if relative:
        return _relative_array_error(rows, reconstructed, epsilon=epsilon)
    return float(np.sqrt(np.mean(np.square(rows - reconstructed))))


def linear_rollout(
    transition_matrix: Any,
    initial_states: Any,
    steps: int,
    *,
    output_matrix: Any | None = None,
) -> np.ndarray:
    """Roll out a full row-major linear system for ``steps`` transitions."""

    transition = _square_matrix(transition_matrix, "transition_matrix")
    steps = _nonnegative_int(steps, "steps")
    states, was_vector = _runtime_rows(
        initial_states,
        "initial_states",
        width=int(transition.shape[0]),
    )
    output = _output_matrix(
        output_matrix,
        state_dimension=int(transition.shape[0]),
    )
    trajectory: list[np.ndarray] = []
    current = states.copy()
    for _ in range(steps + 1):
        trajectory.append(current @ output.T)
        current = current @ transition.T
        _require_finite(current, "linear rollout state")
    stacked = np.stack(trajectory, axis=0)
    return stacked[:, 0, :] if was_vector else stacked


def reduced_linear_rollout(
    transition_matrix: Any,
    initial_states: Any,
    basis: BalancedKrylovBasis,
    steps: int,
    *,
    output_matrix: Any | None = None,
) -> np.ndarray:
    """Roll out the Petrov-Galerkin system induced by ``basis``."""

    if not isinstance(basis, BalancedKrylovBasis):
        raise BalancedKrylovShapeError("basis must be a BalancedKrylovBasis.")
    transition = _square_matrix(
        transition_matrix,
        "transition_matrix",
        width=basis.state_dimension,
    )
    steps = _nonnegative_int(steps, "steps")
    states, was_vector = _runtime_rows(
        initial_states,
        "initial_states",
        width=basis.state_dimension,
    )
    output = _output_matrix(
        output_matrix,
        state_dimension=basis.state_dimension,
    )
    reduced_transition = basis.encoder @ transition @ basis.decoder
    affine_offset = basis.encoder @ (transition @ basis.mean - basis.mean)
    coordinates = (states - basis.mean) @ basis.encoder.T
    trajectory: list[np.ndarray] = []
    for _ in range(steps + 1):
        decoded = coordinates @ basis.decoder.T + basis.mean
        trajectory.append(decoded @ output.T)
        coordinates = coordinates @ reduced_transition.T + affine_offset
        _require_finite(coordinates, "reduced rollout coordinates")
    stacked = np.stack(trajectory, axis=0)
    return stacked[:, 0, :] if was_vector else stacked


def rollout_error_report(
    reference_rollout: Any,
    approximate_rollout: Any,
    *,
    include_initial: bool = False,
    epsilon: float = 1e-12,
) -> RolloutErrorReport:
    """Compare already-materialized rollout arrays along their first axis."""

    if not isinstance(include_initial, bool):
        raise BalancedKrylovShapeError("include_initial must be a boolean.")
    epsilon = _positive_float(epsilon, "epsilon")
    reference = _as_float64_array(reference_rollout, "reference_rollout")
    approximate = _as_float64_array(
        approximate_rollout,
        "approximate_rollout",
    )
    if reference.shape != approximate.shape:
        raise BalancedKrylovShapeError(
            "reference_rollout and approximate_rollout must have the same shape."
        )
    if reference.ndim < 2 or reference.shape[0] < 1:
        raise BalancedKrylovShapeError(
            "rollouts must have shape [steps, ..., outputs]."
        )
    transition_count = int(reference.shape[0] - 1)
    start = 0 if include_initial else 1
    if start >= int(reference.shape[0]):
        raise BalancedKrylovShapeError(
            "At least one transition is required when include_initial is false."
        )
    selected_reference = reference[start:]
    selected_approximate = approximate[start:]
    difference = selected_approximate - selected_reference
    relative_error = _relative_array_error(
        selected_reference,
        selected_approximate,
        epsilon=epsilon,
    )
    per_step = tuple(
        _relative_array_error(
            selected_reference[index],
            selected_approximate[index],
            epsilon=epsilon,
        )
        for index in range(int(selected_reference.shape[0]))
    )
    rms = float(np.sqrt(np.mean(np.square(difference))))
    maximum = float(np.max(np.abs(difference)))
    reference_norm = float(np.linalg.norm(selected_reference.reshape(-1)))
    values = (relative_error, rms, maximum, reference_norm, *per_step)
    if not all(math.isfinite(value) for value in values):
        raise BalancedKrylovFiniteError(
            "Rollout error diagnostics contain NaN or infinity."
        )
    return RolloutErrorReport(
        transition_count=transition_count,
        evaluated_step_count=int(selected_reference.shape[0]),
        include_initial=include_initial,
        relative_l2_error=relative_error,
        root_mean_squared_error=rms,
        maximum_absolute_error=maximum,
        reference_l2_norm=reference_norm,
        per_step_relative_l2_error=per_step,
    )


def multi_step_rollout_error_report(
    transition_matrix: Any,
    initial_states: Any,
    basis: BalancedKrylovBasis,
    steps: int,
    *,
    output_matrix: Any | None = None,
    include_initial: bool = False,
    epsilon: float = 1e-12,
) -> RolloutErrorReport:
    """Compare full and reduced parent-owned linear rollouts."""

    reference = linear_rollout(
        transition_matrix,
        initial_states,
        steps,
        output_matrix=output_matrix,
    )
    approximate = reduced_linear_rollout(
        transition_matrix,
        initial_states,
        basis,
        steps,
        output_matrix=output_matrix,
    )
    return rollout_error_report(
        reference,
        approximate,
        include_initial=include_initial,
        epsilon=epsilon,
    )


def multi_step_rollout_error(
    transition_matrix: Any,
    initial_states: Any,
    basis: BalancedKrylovBasis,
    steps: int,
    *,
    output_matrix: Any | None = None,
    include_initial: bool = False,
    epsilon: float = 1e-12,
) -> float:
    """Return relative L2 output error over a reduced multi-step rollout."""

    return multi_step_rollout_error_report(
        transition_matrix,
        initial_states,
        basis,
        steps,
        output_matrix=output_matrix,
        include_initial=include_initial,
        epsilon=epsilon,
    ).relative_l2_error


# A concise descriptive alias for callers concerned specifically with outputs.
multi_step_output_error = multi_step_rollout_error


def _matrix_group(value: Any, name: str, *, required: bool) -> tuple[np.ndarray, ...]:
    if value is None:
        if required:
            raise BalancedKrylovShapeError(f"{name} is required.")
        return ()
    if _is_matrix_object(value):
        return (_as_float64_matrix(value, name),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            candidate = np.asarray(value)
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate.ndim == 2:
            return (_as_float64_matrix(candidate, name),)
        if len(value) == 0:
            raise BalancedKrylovShapeError(f"{name} cannot be empty.")
        return tuple(
            _as_float64_matrix(matrix, f"{name}[{index}]")
            for index, matrix in enumerate(value)
        )
    return (_as_float64_matrix(value, name),)


def _combined_groups(
    values: Sequence[tuple[Any | None, str]],
) -> tuple[np.ndarray, ...]:
    matrices: list[np.ndarray] = []
    for value, name in values:
        matrices.extend(_matrix_group(value, name, required=False))
    return tuple(matrices)


def _is_matrix_object(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _as_float64_matrix(value: Any, name: str) -> np.ndarray:
    array = _as_float64_array(value, name)
    if array.ndim != 2:
        raise BalancedKrylovShapeError(f"{name} must be two-dimensional.")
    if 0 in array.shape:
        raise BalancedKrylovShapeError(f"{name} cannot have an empty dimension.")
    return array


def _as_float64_array(value: Any, name: str) -> np.ndarray:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by input type
            raise BalancedKrylovShapeError(
                f"{name} is a Torch tensor but PyTorch is unavailable."
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise BalancedKrylovShapeError(
                f"{name} must be a NumPy array, Torch tensor, or array-like."
            )
        if value.layout != torch.strided:
            raise BalancedKrylovShapeError(f"{name} must be a dense strided tensor.")
        if not torch.is_floating_point(value):
            raise BalancedKrylovShapeError(
                f"{name} must have a real floating-point dtype."
            )
        array = (
            value.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .numpy()
            .copy()
        )
    else:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise BalancedKrylovShapeError(
                f"{name} must be a NumPy array, Torch tensor, or array-like."
            ) from exc
        if raw.dtype.kind != "f":
            raise BalancedKrylovShapeError(
                f"{name} must have a real floating-point dtype."
            )
        array = np.array(raw, dtype=np.float64, order="C", copy=True)
    if array.ndim == 0:
        raise BalancedKrylovShapeError(f"{name} must have at least one dimension.")
    _require_finite(array, name)
    return np.ascontiguousarray(array, dtype=np.float64)


def _runtime_rows(value: Any, name: str, *, width: int) -> tuple[np.ndarray, bool]:
    array = _as_float64_array(value, name)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    if array.ndim != 2 or int(array.shape[1]) != width:
        raise BalancedKrylovShapeError(
            f"{name} must have shape ({width},) or [rows, {width}]."
        )
    return array, was_vector


def _require_common_width(
    matrices: Sequence[np.ndarray],
    width: int,
    name: str,
) -> None:
    for index, matrix in enumerate(matrices):
        if int(matrix.shape[1]) != width:
            raise BalancedKrylovShapeError(
                f"{name}[{index}] width {matrix.shape[1]} does not match "
                f"activation width {width}."
            )


def _require_finite(value: np.ndarray, name: str) -> None:
    if not bool(np.all(np.isfinite(value))):
        raise BalancedKrylovFiniteError(f"{name} contains NaN or infinity.")


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise BalancedKrylovShapeError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BalancedKrylovShapeError(f"{name} must be a positive integer.") from exc
    if result <= 0 or result != value:
        raise BalancedKrylovShapeError(f"{name} must be a positive integer.")
    if maximum is not None and result > maximum:
        raise BalancedKrylovShapeError(f"{name} cannot exceed {maximum}.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise BalancedKrylovShapeError(f"{name} must be a non-negative integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BalancedKrylovShapeError(
            f"{name} must be a non-negative integer."
        ) from exc
    if result < 0 or result != value:
        raise BalancedKrylovShapeError(f"{name} must be a non-negative integer.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise BalancedKrylovShapeError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BalancedKrylovShapeError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise BalancedKrylovShapeError(f"{name} must be finite.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise BalancedKrylovShapeError(f"{name} must be greater than zero.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise BalancedKrylovShapeError(f"{name} must be non-negative.")
    return result


def _empirical_block(rows: np.ndarray, weight: float) -> np.ndarray:
    scale = math.sqrt(weight / int(rows.shape[0]))
    block = np.ascontiguousarray(rows * scale, dtype=np.float64)
    _require_finite(block, "weighted empirical sketch block")
    return block


def _safe_svd(
    matrix: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_finite(matrix, name)
    try:
        left, singular_values, right_t = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        raise BalancedKrylovRankError(f"SVD failed for {name}.") from exc
    for value, value_name in (
        (left, f"{name} left singular vectors"),
        (singular_values, f"{name} singular values"),
        (right_t, f"{name} right singular vectors"),
    ):
        _require_finite(value, value_name)
    return left, singular_values, right_t


def _rank_tolerance(
    singular_values: np.ndarray,
    shape: tuple[int, ...],
    *,
    atol: float,
    rtol: float | None,
) -> float:
    if singular_values.size == 0:
        return float(atol)
    relative = rtol if rtol is not None else _FLOAT64_EPS * max(shape) * 64.0
    tolerance = max(atol, float(singular_values[0]) * relative)
    if not math.isfinite(tolerance):
        raise BalancedKrylovFiniteError("Computed SVD rank tolerance is non-finite.")
    return tolerance


def _effective_rank(singular_values: np.ndarray, tolerance: float) -> int:
    return int(np.count_nonzero(singular_values > tolerance))


def _gramian_factor(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray],
    effective_rank: int,
) -> np.ndarray:
    _, singular_values, right_t = decomposition
    return np.ascontiguousarray(
        right_t[:effective_rank].T * singular_values[:effective_rank],
        dtype=np.float64,
    )


def _balanced_modes(
    primal_factor: np.ndarray,
    dual_factor: np.ndarray,
    hankel_u: np.ndarray,
    hankel_singular_values: np.ndarray,
    hankel_vt: np.ndarray,
    *,
    hankel_rank: int,
    rank: int,
    degeneracy_atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    inverse_roots = 1.0 / np.sqrt(hankel_singular_values[:hankel_rank])
    decoder_all = primal_factor @ hankel_vt[:hankel_rank].T * inverse_roots[None, :]
    encoder_all = inverse_roots[:, None] * (hankel_u[:, :hankel_rank].T @ dual_factor.T)
    _require_finite(decoder_all, "balanced decoder modes")
    _require_finite(encoder_all, "balanced encoder modes")

    tolerance = max(
        degeneracy_atol,
        _FLOAT64_EPS
        * max(primal_factor.shape[0], hankel_rank)
        * max(1.0, float(hankel_singular_values[0]))
        * 128.0,
    )
    decoder_parts: list[np.ndarray] = []
    encoder_parts: list[np.ndarray] = []
    selected = 0
    start = 0
    while start < hankel_rank and selected < rank:
        end = start + 1
        while (
            end < hankel_rank
            and abs(float(hankel_singular_values[end] - hankel_singular_values[start]))
            <= tolerance
        ):
            end += 1
        take = min(end - start, rank - selected)
        decoder_group = decoder_all[:, start:end]
        encoder_group = encoder_all[start:end, :]
        if end - start == 1:
            decoder_parts.append(decoder_group[:, :take])
            encoder_parts.append(encoder_group[:take, :])
        else:
            try:
                projector = decoder_group @ np.linalg.pinv(decoder_group)
            except np.linalg.LinAlgError as exc:
                raise BalancedKrylovConditionError(
                    "Could not canonicalize degenerate balanced modes."
                ) from exc
            canonical = _canonical_subspace(projector, end - start)
            group_projection = decoder_group @ encoder_group
            canonical_encoder = canonical.T @ group_projection
            decoder_parts.append(canonical[:, :take])
            encoder_parts.append(canonical_encoder[:take, :])
        selected += take
        start = end

    decoder = np.ascontiguousarray(np.column_stack(decoder_parts))
    encoder = np.ascontiguousarray(np.vstack(encoder_parts))
    gram = encoder @ decoder
    try:
        encoder = np.linalg.solve(gram, encoder)
    except np.linalg.LinAlgError as exc:
        raise BalancedKrylovConditionError(
            "Balanced modes could not be made biorthogonal."
        ) from exc
    return decoder, np.ascontiguousarray(encoder)


def _canonical_subspace(projector: np.ndarray, width: int) -> np.ndarray:
    dimension = int(projector.shape[0])
    acceptance = _FLOAT64_EPS * max(1, dimension) * 512.0
    selected: list[np.ndarray] = []
    for axis in range(dimension):
        candidate = projector[:, axis].copy()
        for _ in range(2):
            for existing in selected:
                candidate -= float(existing @ candidate) * existing
        norm = float(np.linalg.norm(candidate))
        if norm <= acceptance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0.0:
            candidate *= -1.0
        selected.append(candidate)
        if len(selected) == width:
            break
    if len(selected) != width:
        raise BalancedKrylovRankError(
            "Could not construct a deterministic basis for a degenerate subspace."
        )
    return np.ascontiguousarray(np.column_stack(selected))


def _unsafe_basis_reason(
    decoder: np.ndarray,
    encoder: np.ndarray,
    *,
    max_condition_number: float,
) -> str | None:
    if not bool(np.all(np.isfinite(decoder))) or not bool(np.all(np.isfinite(encoder))):
        return "balanced modes contain NaN or infinity"
    decoder_condition = _matrix_condition_number(decoder)
    encoder_condition = _matrix_condition_number(encoder)
    if not math.isfinite(decoder_condition) or not math.isfinite(encoder_condition):
        return "balanced mode condition number is non-finite"
    if decoder_condition > max_condition_number:
        return (
            f"decoder condition number {decoder_condition:.6g} exceeds limit "
            f"{max_condition_number:.6g}"
        )
    if encoder_condition > max_condition_number:
        return (
            f"encoder condition number {encoder_condition:.6g} exceeds limit "
            f"{max_condition_number:.6g}"
        )
    error = _max_abs(encoder @ decoder - np.eye(decoder.shape[1], dtype=np.float64))
    if error > _BIORTHOGONALITY_ATOL:
        return f"biorthogonality error {error:.6g} exceeds tolerance"
    return None


def _fallback_basis(
    controllability: np.ndarray,
    observability: np.ndarray | None,
    *,
    rank: int,
    atol: float,
    rtol: float | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    controllability_norm = float(np.linalg.norm(controllability))
    if controllability_norm == 0.0:
        raise BalancedKrylovRankError(
            "Controllability sketch has zero spectral energy."
        )
    rows = [controllability / controllability_norm]
    method = "controllability_svd_fallback"
    if observability is not None:
        observability_norm = float(np.linalg.norm(observability))
        if observability_norm > 0.0:
            rows.append(observability / observability_norm)
            method = "joint_sketch_svd_fallback"
    fallback_rows = np.vstack(rows)
    _, singular_values, right_t = _safe_svd(
        fallback_rows,
        "fallback feature sketch",
    )
    tolerance = _rank_tolerance(
        singular_values,
        fallback_rows.shape,
        atol=atol,
        rtol=rtol,
    )
    effective_rank = _effective_rank(singular_values, tolerance)
    if effective_rank < rank:
        raise BalancedKrylovRankError(
            "Fallback feature sketch has effective rank "
            f"{effective_rank}, below requested rank {rank}."
        )

    degeneracy_tolerance = max(
        atol,
        _FLOAT64_EPS
        * max(fallback_rows.shape)
        * max(1.0, float(singular_values[0]))
        * 128.0,
    )
    columns: list[np.ndarray] = []
    selected = 0
    start = 0
    while start < effective_rank and selected < rank:
        end = start + 1
        while (
            end < effective_rank
            and abs(float(singular_values[end] - singular_values[start]))
            <= degeneracy_tolerance
        ):
            end += 1
        take = min(end - start, rank - selected)
        group = right_t[start:end].T
        if end - start > 1:
            group = _canonical_subspace(group @ group.T, end - start)
        columns.append(group[:, :take])
        selected += take
        start = end
    decoder = np.ascontiguousarray(np.column_stack(columns))
    encoder = np.ascontiguousarray(decoder.T)
    return decoder, encoder, method


def _canonicalize_signs(
    decoder: np.ndarray,
    encoder: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[float, ...]]:
    decoder = np.array(decoder, dtype=np.float64, order="C", copy=True)
    encoder = np.array(encoder, dtype=np.float64, order="C", copy=True)
    pivots: list[int] = []
    values: list[float] = []
    for index in range(int(decoder.shape[1])):
        pivot = int(np.argmax(np.abs(decoder[:, index])))
        if decoder[pivot, index] < 0.0:
            decoder[:, index] *= -1.0
            encoder[index, :] *= -1.0
        pivots.append(pivot)
        values.append(float(decoder[pivot, index]))
    return decoder, encoder, tuple(pivots), tuple(values)


def _validate_basis(
    decoder: np.ndarray,
    encoder: np.ndarray,
    rank: int,
    state_dimension: int,
) -> None:
    if decoder.shape != (state_dimension, rank):
        raise BalancedKrylovShapeError(
            "Decoder has an invalid shape after decomposition."
        )
    if encoder.shape != (rank, state_dimension):
        raise BalancedKrylovShapeError(
            "Encoder has an invalid shape after decomposition."
        )
    _require_finite(decoder, "decoder")
    _require_finite(encoder, "encoder")
    if np.linalg.matrix_rank(decoder) != rank:
        raise BalancedKrylovRankError("Decoder lost full column rank.")
    if np.linalg.matrix_rank(encoder) != rank:
        raise BalancedKrylovRankError("Encoder lost full row rank.")
    error = _max_abs(encoder @ decoder - np.eye(rank, dtype=np.float64))
    if error > _BIORTHOGONALITY_ATOL:
        raise BalancedKrylovConditionError(
            "Encoder and decoder are not biorthogonal within tolerance; "
            f"maximum error is {error:.6g}."
        )


def _matrix_condition_number(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] == 0.0:
        return math.inf
    return float(singular_values[0] / singular_values[-1])


def _relative_projection_error(rows: np.ndarray, operator: np.ndarray) -> float:
    reconstructed = rows @ operator
    return _relative_array_error(rows, reconstructed, epsilon=1e-15)


def _relative_array_error(
    reference: np.ndarray,
    approximate: np.ndarray,
    *,
    epsilon: float,
) -> float:
    epsilon = _positive_float(epsilon, "epsilon")
    difference_norm = float(np.linalg.norm((approximate - reference).reshape(-1)))
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    if reference_norm <= epsilon:
        result = 0.0 if difference_norm <= epsilon else difference_norm / epsilon
    else:
        result = difference_norm / reference_norm
    if not math.isfinite(result):
        raise BalancedKrylovFiniteError(
            "Relative error calculation produced NaN or infinity."
        )
    return result


def _subspace_energy_fraction(rows: np.ndarray, columns: np.ndarray) -> float:
    orthonormal, _ = np.linalg.qr(columns, mode="reduced")
    total = float(np.sum(np.square(rows)))
    if total == 0.0:
        return 1.0
    captured = float(np.sum(np.square(rows @ orthonormal)))
    result = min(1.0, max(0.0, captured / total))
    if not math.isfinite(result):
        raise BalancedKrylovFiniteError(
            "Captured-energy calculation produced NaN or infinity."
        )
    return result


def _captured_spectral_energy(
    singular_values: np.ndarray,
    rank: int,
) -> float | None:
    if singular_values.size == 0:
        return None
    total = float(np.sum(np.square(singular_values)))
    if total == 0.0:
        return 0.0
    captured = float(np.sum(np.square(singular_values[:rank])))
    return min(1.0, max(0.0, captured / total))


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _hash_array(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"neural-forge-canonical-array-v1\0")
    digest.update(str(canonical.ndim).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _hash_group(matrices: Sequence[np.ndarray]) -> str:
    parts = ["balanced-krylov-matrix-group-v1", str(len(matrices))]
    parts.extend(_hash_array(matrix) for matrix in matrices)
    return _hash_parts(*parts)


def _hash_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def _square_matrix(
    value: Any,
    name: str,
    *,
    width: int | None = None,
) -> np.ndarray:
    matrix = _as_float64_matrix(value, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise BalancedKrylovShapeError(f"{name} must be square.")
    if width is not None and matrix.shape != (width, width):
        raise BalancedKrylovShapeError(f"{name} must have shape ({width}, {width}).")
    return matrix


def _output_matrix(value: Any | None, *, state_dimension: int) -> np.ndarray:
    if value is None:
        return np.eye(state_dimension, dtype=np.float64)
    matrix = _as_float64_array(value, "output_matrix")
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or int(matrix.shape[1]) != state_dimension:
        raise BalancedKrylovShapeError(
            "output_matrix must have shape [outputs, state_dimension]."
        )
    if matrix.shape[0] == 0:
        raise BalancedKrylovShapeError(
            "output_matrix cannot have an empty output dimension."
        )
    return matrix


__all__ = [
    "BalancedKrylovBasis",
    "BalancedKrylovConditionError",
    "BalancedKrylovError",
    "BalancedKrylovFiniteError",
    "BalancedKrylovRankError",
    "BalancedKrylovReport",
    "BalancedKrylovShapeError",
    "ConditionValidationError",
    "FiniteValidationError",
    "RankValidationError",
    "RolloutErrorReport",
    "ShapeValidationError",
    "build_balanced_krylov_basis",
    "canonical_array_sha256",
    "linear_rollout",
    "multi_step_output_error",
    "multi_step_rollout_error",
    "multi_step_rollout_error_report",
    "reconstruction_error",
    "reduced_linear_rollout",
    "relative_reconstruction_error",
    "rollout_error_report",
]
