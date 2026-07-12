"""Deterministic dual-aware residual coordinates and no-model oracle metrics.

The builder balances energy in parent residual activations with energy in the
dual embedding/head rows and teacher decision margins.  It operates only on
already-collected tensors, always on detached CPU float64 values, and never
constructs an autograd or optimizer path.
"""

from __future__ import annotations

import math
from typing import Any

from neural_forge.hf_transport_math import (
    CoordinateMap,
    FiniteValidationError,
    PseudoinverseDiagnostics,
    RankValidationError,
    ShapeValidationError,
)


REPORT_SCHEMA_VERSION = 1
_ORTHONORMAL_ATOL = 1e-10


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Dual-aware Hugging Face basis operations require PyTorch."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ShapeValidationError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ShapeValidationError(f"{name} must be a positive integer.") from exc
    if result <= 0 or result != value:
        raise ShapeValidationError(f"{name} must be a positive integer.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ShapeValidationError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ShapeValidationError(f"{name} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ShapeValidationError(f"{name} must be finite and non-negative.")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise ShapeValidationError(f"{name} must be greater than zero.")
    return result


def _float64_matrix(torch, value: Any, name: str):
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


def _float64_vector(torch, value: Any, name: str, *, length: int):
    if not isinstance(value, torch.Tensor):
        raise ShapeValidationError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise ShapeValidationError(f"{name} must be a dense strided tensor.")
    if value.ndim != 1 or int(value.shape[0]) != length:
        raise ShapeValidationError(f"{name} must have shape ({length},).")
    if not torch.is_floating_point(value):
        raise ShapeValidationError(f"{name} must have a real floating-point dtype.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise FiniteValidationError(f"{name} contains NaN or infinity.")
    return result


def _max_abs(torch, value: Any) -> float:
    return float(torch.max(torch.abs(value)).item())


def _trace_normalized_covariance(
    torch,
    rows: Any,
    name: str,
    *,
    active: bool,
    row_weights: Any | None = None,
):
    if row_weights is None:
        weighted_rows = rows
    else:
        weighted_rows = rows * torch.sqrt(row_weights).unsqueeze(1)
        if not bool(torch.isfinite(weighted_rows).all()):
            raise FiniteValidationError(
                f"Weighted {name} rows contain NaN or infinity."
            )
    covariance = (weighted_rows.T @ weighted_rows).contiguous()
    covariance = ((covariance + covariance.T) * 0.5).contiguous()
    if not bool(torch.isfinite(covariance).all()):
        raise FiniteValidationError(f"{name} covariance contains NaN or infinity.")
    trace = float(torch.trace(covariance).item())
    if not math.isfinite(trace):
        raise FiniteValidationError(f"{name} covariance trace is non-finite.")
    if trace <= 0.0:
        if active:
            raise RankValidationError(
                f"{name} covariance has zero trace while its weight is active."
            )
        return covariance.zero_(), trace, False
    normalized = (covariance / trace).contiguous()
    if not bool(torch.isfinite(normalized).all()):
        raise FiniteValidationError(
            f"Trace-normalized {name} covariance is non-finite."
        )
    return normalized, trace, True


def _canonical_subspace(torch, projector: Any, width: int):
    dimension = int(projector.shape[0])
    acceptance = torch.finfo(torch.float64).eps * max(1, dimension) * 512.0
    selected: list[Any] = []
    for axis in range(dimension):
        candidate = projector[:, axis].clone()
        for _ in range(2):
            for existing in selected:
                candidate.sub_(torch.dot(existing, candidate) * existing)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= acceptance:
            continue
        candidate.div_(norm)
        pivot = int(torch.argmax(torch.abs(candidate)).item())
        if float(candidate[pivot].item()) < 0.0:
            candidate.neg_()
        selected.append(candidate)
        if len(selected) == width:
            break
    if len(selected) != width:
        raise RankValidationError(
            "Could not construct a deterministic basis for a degenerate "
            "covariance eigenspace."
        )
    return torch.stack(selected, dim=1).contiguous()


def _descending_eigh(torch, covariance: Any, *, target_dim: int):
    covariance = ((covariance + covariance.T) * 0.5).contiguous()
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    except RuntimeError as exc:
        raise RankValidationError(
            "CPU float64 covariance eigendecomposition failed."
        ) from exc
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(
        torch.isfinite(eigenvectors).all()
    ):
        raise FiniteValidationError(
            "Covariance eigendecomposition produced NaN or infinity."
        )
    eigenvalues = torch.flip(eigenvalues, dims=(0,)).contiguous()
    eigenvectors = torch.flip(eigenvectors, dims=(1,)).contiguous()
    scale = max(1.0, float(torch.abs(eigenvalues[0]).item()))
    tolerance = (
        torch.finfo(torch.float64).eps * scale * max(1, int(covariance.shape[0])) * 64.0
    )
    minimum = float(eigenvalues[-1].item())
    if minimum < -tolerance:
        raise RankValidationError(
            "Combined covariance is not positive semidefinite within float64 "
            f"tolerance; minimum eigenvalue is {minimum:.6g}."
        )
    eigenvalues.clamp_(min=0.0)

    degenerate_groups = 0
    start = 0
    while start < target_dim:
        end = start + 1
        while (
            end < int(eigenvalues.numel())
            and abs(float(eigenvalues[end].item() - eigenvalues[start].item()))
            <= tolerance
        ):
            end += 1
        selected_width = min(end, target_dim) - start
        if end - start > 1:
            block = eigenvectors[:, start:end]
            projector = (block @ block.T).contiguous()
            eigenvectors[:, start : start + selected_width] = _canonical_subspace(
                torch,
                projector,
                selected_width,
            )
            degenerate_groups += 1
        else:
            vector = eigenvectors[:, start]
            pivot = int(torch.argmax(torch.abs(vector)).item())
            if float(vector[pivot].item()) < 0.0:
                vector.neg_()
        start = end
    return eigenvalues, eigenvectors, degenerate_groups, float(tolerance)


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
        decode_pseudoinverse_max_abs_error=_max_abs(
            torch, decode_pseudoinverse - encode
        ),
        encode_pseudoinverse_max_abs_error=_max_abs(
            torch, encode_pseudoinverse - decode
        ),
        target_identity_max_abs_error=_max_abs(torch, encode @ decode - identity),
        source_projector_symmetry_max_abs_error=_max_abs(
            torch, projector - projector.T
        ),
        source_projector_idempotence_max_abs_error=_max_abs(
            torch, projector @ projector - projector
        ),
        finite=finite,
    )


def _covariance_rank(torch, covariance: Any) -> int:
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    largest = float(eigenvalues[-1].item())
    if largest == 0.0:
        return 0
    tolerance = (
        max(covariance.shape) * torch.finfo(torch.float64).eps * math.sqrt(largest)
    )
    return int((torch.sqrt(eigenvalues) > tolerance).sum().item())


def build_dual_aware_residual_basis(
    residual_activation_rows: Any,
    sampled_tied_embedding_rows: Any,
    teacher_margin_directions: Any,
    target_dim: int,
    *,
    margin_weights: Any | None = None,
    teacher_margin_weights: Any | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    center_residual: bool = True,
    rank_atol: float = 0.0,
    rank_rtol: float | None = None,
) -> tuple[CoordinateMap, dict[str, Any]]:
    """Build a trace-normalized primal/dual residual basis.

    ``alpha`` weights centered residual activation covariance, ``beta`` weights
    sampled tied embedding/head row covariance, and ``gamma`` weights the
    covariance of teacher winner-minus-competitor embedding differences.
    Margin row weights enter as statistical weights, not squared multipliers.
    """

    torch = _require_torch()
    with torch.no_grad():
        residual = _float64_matrix(
            torch, residual_activation_rows, "residual_activation_rows"
        )
        embeddings = _float64_matrix(
            torch,
            sampled_tied_embedding_rows,
            "sampled_tied_embedding_rows",
        )
        margins = _float64_matrix(
            torch, teacher_margin_directions, "teacher_margin_directions"
        )
        source_dim = int(residual.shape[1])
        for name, rows in (
            ("sampled_tied_embedding_rows", embeddings),
            ("teacher_margin_directions", margins),
        ):
            if int(rows.shape[1]) != source_dim:
                raise ShapeValidationError(
                    f"{name} width must equal residual activation width {source_dim}."
                )
        target_dim = _positive_int(target_dim, "target_dim")
        if target_dim > source_dim:
            raise ShapeValidationError(
                "target_dim cannot exceed the parent residual width."
            )
        if not isinstance(center_residual, bool):
            raise ShapeValidationError("center_residual must be a boolean.")
        alpha = _finite_nonnegative(alpha, "alpha")
        beta = _finite_nonnegative(beta, "beta")
        gamma = _finite_nonnegative(gamma, "gamma")
        if alpha + beta + gamma == 0.0:
            raise ShapeValidationError(
                "At least one of alpha, beta, and gamma must be positive."
            )
        rank_atol = _finite_nonnegative(rank_atol, "rank_atol")
        if rank_rtol is None:
            rank_rtol = (
                max(
                    source_dim,
                    int(residual.shape[0] + embeddings.shape[0] + margins.shape[0]),
                )
                * torch.finfo(torch.float64).eps
            )
        else:
            rank_rtol = _finite_nonnegative(rank_rtol, "rank_rtol")

        if margin_weights is not None and teacher_margin_weights is not None:
            raise ShapeValidationError(
                "Use margin_weights or teacher_margin_weights, not both."
            )
        supplied_margin_weights = (
            teacher_margin_weights
            if teacher_margin_weights is not None
            else margin_weights
        )
        if supplied_margin_weights is None:
            weights = torch.ones(int(margins.shape[0]), dtype=torch.float64)
            margin_weights_provided = False
        else:
            weights = _float64_vector(
                torch,
                supplied_margin_weights,
                "margin_weights",
                length=int(margins.shape[0]),
            )
            margin_weights_provided = True
        if bool((weights < 0.0).any()):
            raise ShapeValidationError("margin_weights cannot contain negatives.")
        weight_sum = float(weights.sum().item())
        if not math.isfinite(weight_sum):
            raise FiniteValidationError("margin_weights sum is non-finite.")

        residual_mean = (
            residual.mean(dim=0)
            if center_residual
            else torch.zeros(source_dim, dtype=torch.float64)
        )
        fitted_residual = residual - residual_mean if center_residual else residual
        if not bool(torch.isfinite(fitted_residual).all()):
            raise FiniteValidationError(
                "Centered residual activation rows are non-finite."
            )

        residual_covariance, residual_trace, residual_normalized = (
            _trace_normalized_covariance(
                torch,
                fitted_residual,
                "residual activation",
                active=alpha > 0.0,
            )
        )
        embedding_covariance, embedding_trace, embedding_normalized = (
            _trace_normalized_covariance(
                torch,
                embeddings,
                "tied embedding",
                active=beta > 0.0,
            )
        )
        margin_covariance, margin_trace, margin_normalized = (
            _trace_normalized_covariance(
                torch,
                margins,
                "teacher margin",
                active=gamma > 0.0,
                row_weights=weights,
            )
        )
        combined = (
            alpha * residual_covariance
            + beta * embedding_covariance
            + gamma * margin_covariance
        ).contiguous()
        combined = ((combined + combined.T) * 0.5).contiguous()
        if not bool(torch.isfinite(combined).all()):
            raise FiniteValidationError("Combined covariance is non-finite.")

        eigenvalues, eigenvectors, degenerate_groups, eigenspace_tolerance = (
            _descending_eigh(torch, combined, target_dim=target_dim)
        )
        singular_values = torch.sqrt(eigenvalues).contiguous()
        largest_singular = float(singular_values[0].item())
        rank_tolerance = max(rank_atol, largest_singular * rank_rtol)
        if not math.isfinite(rank_tolerance):
            raise FiniteValidationError("Computed rank tolerance is non-finite.")
        effective_rank = int((singular_values > rank_tolerance).sum().item())
        if effective_rank < target_dim:
            raise RankValidationError(
                "Combined dual-aware covariance has effective rank "
                f"{effective_rank}, below requested target_dim {target_dim} "
                f"at singular-value tolerance {rank_tolerance:.6g}."
            )

        decode = eigenvectors[:, :target_dim].clone().contiguous()
        column_indices = torch.arange(target_dim, dtype=torch.long)
        pivot_rows = torch.argmax(torch.abs(decode), dim=0)
        pivot_values = decode[pivot_rows, column_indices]
        signs = torch.where(
            pivot_values < 0.0,
            -torch.ones_like(pivot_values),
            torch.ones_like(pivot_values),
        )
        decode.mul_(signs.unsqueeze(0))
        encode = decode.T.contiguous()
        projector = (decode @ encode).contiguous()
        diagnostics = _coordinate_diagnostics(torch, decode, encode, projector)
        if not diagnostics.finite:
            raise FiniteValidationError(
                "Coordinate-map diagnostics contain NaN or infinity."
            )
        if diagnostics.rank != target_dim:
            raise RankValidationError(
                "The dual-aware basis lost full target-column rank."
            )
        if diagnostics.target_identity_max_abs_error > _ORTHONORMAL_ATOL:
            raise RankValidationError(
                "The dual-aware basis is not column-orthonormal within "
                f"tolerance {_ORTHONORMAL_ATOL:.6g}."
            )

        total_eigenvalue = float(eigenvalues.sum().item())
        selected_eigenvalue = float(eigenvalues[:target_dim].sum().item())
        if not math.isfinite(total_eigenvalue) or total_eigenvalue <= 0.0:
            raise RankValidationError(
                "Combined covariance has no positive finite spectral energy."
            )
        explained_variance_fraction = selected_eigenvalue / total_eigenvalue

        component_covariances = {
            "residual": residual_covariance,
            "embedding": embedding_covariance,
            "margin": margin_covariance,
        }
        captured_fractions = {
            name: float(torch.trace(encode @ covariance @ decode).item())
            if normalized
            else 0.0
            for (name, covariance), normalized in zip(
                component_covariances.items(),
                (
                    residual_normalized,
                    embedding_normalized,
                    margin_normalized,
                ),
                strict=True,
            )
        }
        normalized_traces = {
            name: float(torch.trace(covariance).item())
            for name, covariance in component_covariances.items()
        }
        covariance_ranks = {
            name: _covariance_rank(torch, covariance)
            for name, covariance in component_covariances.items()
        }
        pivot_rows = torch.argmax(torch.abs(decode), dim=0)
        pivot_values = decode[pivot_rows, column_indices]

        coordinates = CoordinateMap(
            source_dim=source_dim,
            target_dim=target_dim,
            activation_row_count=int(residual.shape[0]),
            centered_for_fit=center_residual,
            effective_rank=effective_rank,
            rank_tolerance=rank_tolerance,
            explained_variance_fraction=explained_variance_fraction,
            fit_mean=residual_mean.detach().contiguous(),
            singular_values=singular_values.detach().contiguous(),
            encode=encode.detach().contiguous(),
            decode=decode.detach().contiguous(),
            source_projector=projector.detach().contiguous(),
            diagnostics=diagnostics,
        )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": "dual-aware-trace-normalized-residual-basis",
            "source_dim": source_dim,
            "target_dim": target_dim,
            "device": "cpu",
            "dtype": "float64",
            "gradient_free": True,
            "optimizer_used": False,
            "centered_residual_activations": center_residual,
            "row_counts": {
                "residual": int(residual.shape[0]),
                "embedding": int(embeddings.shape[0]),
                "margin": int(margins.shape[0]),
            },
            "covariance_weights": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            },
            "raw_covariance_traces": {
                "residual": residual_trace,
                "embedding": embedding_trace,
                "margin": margin_trace,
            },
            "normalized_covariance_traces": normalized_traces,
            "covariance_was_normalized": {
                "residual": residual_normalized,
                "embedding": embedding_normalized,
                "margin": margin_normalized,
            },
            "covariance_ranks": covariance_ranks,
            "combined_covariance_trace": float(torch.trace(combined).item()),
            "eigenvalues": [float(value) for value in eigenvalues.tolist()],
            "singular_values": [float(value) for value in singular_values.tolist()],
            "effective_rank": effective_rank,
            "rank_tolerance": rank_tolerance,
            "eigenspace_tolerance": eigenspace_tolerance,
            "degenerate_selected_eigenspace_count": degenerate_groups,
            "explained_variance_fraction": explained_variance_fraction,
            "captured_component_fractions": captured_fractions,
            "margin_weights": {
                "provided": margin_weights_provided,
                "sum": weight_sum,
                "minimum": float(weights.min().item()),
                "maximum": float(weights.max().item()),
                "positive_count": int((weights > 0.0).sum().item()),
            },
            "canonical_pivot_rows": [int(value) for value in pivot_rows.tolist()],
            "canonical_pivot_values": [float(value) for value in pivot_values.tolist()],
            "orthonormality_max_abs_error": (diagnostics.target_identity_max_abs_error),
            "projector_symmetry_max_abs_error": (
                diagnostics.source_projector_symmetry_max_abs_error
            ),
            "projector_idempotence_max_abs_error": (
                diagnostics.source_projector_idempotence_max_abs_error
            ),
            "finite": True,
        }
        return coordinates, report


def _candidate_decode(torch, candidate_basis: Any):
    if isinstance(candidate_basis, CoordinateMap):
        decode = _float64_matrix(
            torch, candidate_basis.decode, "candidate_basis.decode"
        )
        if tuple(decode.shape) != (
            candidate_basis.source_dim,
            candidate_basis.target_dim,
        ):
            raise ShapeValidationError(
                "candidate_basis.decode shape does not match its CoordinateMap "
                "dimensions."
            )
        encode = _float64_matrix(
            torch, candidate_basis.encode, "candidate_basis.encode"
        )
        projector = _float64_matrix(
            torch,
            candidate_basis.source_projector,
            "candidate_basis.source_projector",
        )
        if tuple(encode.shape) != (decode.shape[1], decode.shape[0]):
            raise ShapeValidationError(
                "candidate_basis.encode has an incompatible shape."
            )
        if tuple(projector.shape) != (decode.shape[0], decode.shape[0]):
            raise ShapeValidationError(
                "candidate_basis.source_projector has an incompatible shape."
            )
        if _max_abs(torch, encode - decode.T) > _ORTHONORMAL_ATOL:
            raise ShapeValidationError(
                "candidate_basis.encode must equal candidate_basis.decode.T."
            )
        if _max_abs(torch, projector - decode @ decode.T) > _ORTHONORMAL_ATOL:
            raise ShapeValidationError(
                "candidate_basis.source_projector must equal decode @ decode.T."
            )
        kind = "CoordinateMap"
    else:
        decode = _float64_matrix(torch, candidate_basis, "candidate_basis")
        kind = "tensor"
    identity = torch.eye(int(decode.shape[1]), dtype=torch.float64)
    orthonormality_error = _max_abs(torch, decode.T @ decode - identity)
    if orthonormality_error > _ORTHONORMAL_ATOL:
        raise RankValidationError(
            "candidate_basis columns must be orthonormal; maximum identity "
            f"error is {orthonormality_error:.6g}."
        )
    if int(torch.linalg.matrix_rank(decode).item()) != int(decode.shape[1]):
        raise RankValidationError("candidate_basis is rank deficient.")
    return decode, kind, orthonormality_error


def _index_vector(torch, value: Any, name: str):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.layout != torch.strided or tensor.ndim != 1 or tensor.numel() == 0:
        raise ShapeValidationError(f"{name} must be a non-empty index vector.")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if tensor.dtype not in integer_dtypes:
        raise ShapeValidationError(f"{name} must contain integer indices.")
    result = tensor.detach().to(device="cpu", dtype=torch.long).contiguous()
    if bool((result < 0).any()):
        raise ShapeValidationError(f"{name} cannot contain negative indices.")
    if int(torch.unique(result).numel()) != int(result.numel()):
        raise ShapeValidationError(f"{name} must contain unique indices.")
    return result


def _select_vocab_axis(torch, value: Any, indices: Any, name: str, axis: int):
    size = int(value.shape[axis])
    if size == int(indices.numel()):
        return value
    largest = int(indices.max().item())
    if largest >= size:
        raise ShapeValidationError(
            f"{name} cannot select vocabulary index {largest} from size {size}."
        )
    return torch.index_select(value, axis, indices).contiguous()


def evaluate_dual_basis_oracle(
    candidate_basis: Any,
    heldout_hidden_rows: Any,
    vocab_embedding_or_head_rows: Any,
    teacher_logits: Any,
    *,
    vocab_indices: Any | None = None,
    selected_vocab_indices: Any | None = None,
    head_bias: Any | None = None,
    temperature: float = 1.0,
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Evaluate coordinate logits from cached tensors without invoking a model.

    With no vocabulary indices, head rows and teacher logits must already share
    one vocabulary axis.  With indices, either input may be the full vocabulary
    or an already-selected tensor whose vocabulary size equals the index count.
    """

    torch = _require_torch()
    with torch.no_grad():
        decode, basis_kind, orthonormality_error = _candidate_decode(
            torch, candidate_basis
        )
        hidden = _float64_matrix(torch, heldout_hidden_rows, "heldout_hidden_rows")
        vocab_rows = _float64_matrix(
            torch,
            vocab_embedding_or_head_rows,
            "vocab_embedding_or_head_rows",
        )
        teacher = _float64_matrix(torch, teacher_logits, "teacher_logits")
        source_dim = int(decode.shape[0])
        if int(hidden.shape[1]) != source_dim:
            raise ShapeValidationError(
                "heldout_hidden_rows width must match candidate basis source width."
            )
        if int(vocab_rows.shape[1]) != source_dim:
            raise ShapeValidationError(
                "vocab_embedding_or_head_rows width must match candidate basis "
                "source width."
            )
        if int(teacher.shape[0]) != int(hidden.shape[0]):
            raise ShapeValidationError(
                "teacher_logits row count must match heldout_hidden_rows."
            )
        if vocab_indices is not None and selected_vocab_indices is not None:
            raise ShapeValidationError(
                "Use vocab_indices or selected_vocab_indices, not both."
            )
        supplied_indices = (
            selected_vocab_indices
            if selected_vocab_indices is not None
            else vocab_indices
        )
        indices = None
        if supplied_indices is None:
            if int(vocab_rows.shape[0]) != int(teacher.shape[1]):
                raise ShapeValidationError(
                    "Vocabulary row count must equal teacher_logits width when "
                    "vocab_indices are omitted."
                )
            vocabulary_scope = "full-or-preselected"
        else:
            indices = _index_vector(torch, supplied_indices, "vocab_indices")
            vocab_rows = _select_vocab_axis(
                torch, vocab_rows, indices, "vocab_embedding_or_head_rows", 0
            )
            teacher = _select_vocab_axis(torch, teacher, indices, "teacher_logits", 1)
            if int(vocab_rows.shape[0]) != int(teacher.shape[1]):
                raise ShapeValidationError(
                    "Selected vocabulary rows and teacher logits are not aligned."
                )
            vocabulary_scope = "selected"
        vocab_size = int(vocab_rows.shape[0])
        if vocab_size < 2:
            raise ShapeValidationError(
                "At least two vocabulary rows are required for margin evaluation."
            )

        aligned_bias = None
        if head_bias is not None:
            if not isinstance(head_bias, torch.Tensor):
                raise ShapeValidationError("head_bias must be a torch.Tensor.")
            if head_bias.ndim != 1 or not torch.is_floating_point(head_bias):
                raise ShapeValidationError(
                    "head_bias must be a real floating-point vector."
                )
            aligned_bias = (
                head_bias.detach().to(device="cpu", dtype=torch.float64).contiguous()
            )
            if not bool(torch.isfinite(aligned_bias).all()):
                raise FiniteValidationError("head_bias contains NaN or infinity.")
            if indices is not None:
                aligned_bias = _select_vocab_axis(
                    torch, aligned_bias, indices, "head_bias", 0
                )
            if int(aligned_bias.shape[0]) != vocab_size:
                raise ShapeValidationError(
                    "head_bias length must match the aligned vocabulary size."
                )

        temperature = _finite_positive(temperature, "temperature")
        tie_tolerance = _finite_nonnegative(tie_tolerance, "tie_tolerance")
        hidden_coordinates = (hidden @ decode).contiguous()
        head_coordinates = (vocab_rows @ decode).contiguous()
        coordinate_logits = (hidden_coordinates @ head_coordinates.T).contiguous()
        if aligned_bias is not None:
            coordinate_logits.add_(aligned_bias.unsqueeze(0))
        if not bool(torch.isfinite(coordinate_logits).all()):
            raise FiniteValidationError("Coordinate logits contain NaN or infinity.")

        teacher_top1 = teacher.argmax(dim=1)
        candidate_top1 = coordinate_logits.argmax(dim=1)
        top1 = teacher_top1.eq(candidate_top1).to(torch.float64)
        effective_top5 = min(5, vocab_size)
        teacher_top5 = torch.topk(teacher, k=effective_top5, dim=1).indices
        candidate_top5 = torch.topk(coordinate_logits, k=effective_top5, dim=1).indices
        top5_overlap = teacher_top5.unsqueeze(2).eq(candidate_top5.unsqueeze(1)).any(
            dim=2
        ).sum(dim=1).to(torch.float64) / float(effective_top5)
        top5_exact = top5_overlap.eq(1.0).to(torch.float64)
        teacher_top1_in_candidate_top5 = (
            candidate_top5.eq(teacher_top1.unsqueeze(1)).any(dim=1).to(torch.float64)
        )

        teacher_pair = torch.topk(teacher, k=2, dim=1).indices
        winner = teacher_pair[:, :1]
        competitor = teacher_pair[:, 1:2]
        teacher_margin = teacher.gather(1, winner).squeeze(1) - teacher.gather(
            1, competitor
        ).squeeze(1)
        candidate_margin = coordinate_logits.gather(1, winner).squeeze(
            1
        ) - coordinate_logits.gather(1, competitor).squeeze(1)
        teacher_sign = torch.sign(teacher_margin).where(
            teacher_margin.abs() > tie_tolerance,
            teacher_margin.new_zeros(()),
        )
        candidate_sign = torch.sign(candidate_margin).where(
            candidate_margin.abs() > tie_tolerance,
            candidate_margin.new_zeros(()),
        )
        margin_retention = teacher_sign.eq(candidate_sign).to(torch.float64)

        teacher_log_probability = torch.log_softmax(teacher / temperature, dim=1)
        candidate_log_probability = torch.log_softmax(
            coordinate_logits / temperature, dim=1
        )
        kl_per_row = (
            (
                teacher_log_probability.exp()
                * (teacher_log_probability - candidate_log_probability)
            )
            .sum(dim=1)
            .clamp_min(0.0)
        )

        error = coordinate_logits - teacher
        centered_error = (
            coordinate_logits - coordinate_logits.mean(dim=1, keepdim=True)
        ) - (teacher - teacher.mean(dim=1, keepdim=True))
        coordinate_logit_rmse = float(torch.sqrt(error.square().mean()).item())
        centered_coordinate_logit_rmse = float(
            torch.sqrt(centered_error.square().mean()).item()
        )
        teacher_centered_rms = float(
            torch.sqrt(
                (teacher - teacher.mean(dim=1, keepdim=True)).square().mean()
            ).item()
        )
        relative_centered_error = centered_coordinate_logit_rmse / max(
            teacher_centered_rms,
            torch.finfo(torch.float64).tiny,
        )
        scalar_values = (
            coordinate_logit_rmse,
            centered_coordinate_logit_rmse,
            teacher_centered_rms,
            relative_centered_error,
            float(kl_per_row.mean().item()),
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise FiniteValidationError("Oracle metrics contain NaN or infinity.")

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": "dual-basis-no-model-oracle",
            "basis_kind": basis_kind,
            "source_dim": source_dim,
            "target_dim": int(decode.shape[1]),
            "heldout_row_count": int(hidden.shape[0]),
            "vocab_row_count": vocab_size,
            "vocabulary_scope": vocabulary_scope,
            "selected_vocab_indices": (
                None if indices is None else [int(value) for value in indices.tolist()]
            ),
            "top1_agreement": float(top1.mean().item()),
            "top5_agreement": float(top5_overlap.mean().item()),
            "top5_exact_agreement": float(top5_exact.mean().item()),
            "teacher_top1_in_candidate_top5": float(
                teacher_top1_in_candidate_top5.mean().item()
            ),
            "top5_effective_k": effective_top5,
            "teacher_margin_sign_retention": float(margin_retention.mean().item()),
            "kl": float(kl_per_row.mean().item()),
            "kl_teacher_candidate": float(kl_per_row.mean().item()),
            "temperature": temperature,
            "tie_tolerance": tie_tolerance,
            "coordinate_logit_error": coordinate_logit_rmse,
            "coordinate_logit_rmse": coordinate_logit_rmse,
            "coordinate_logit_mae": float(error.abs().mean().item()),
            "coordinate_logit_max_abs_error": _max_abs(torch, error),
            "centered_coordinate_logit_rmse": centered_coordinate_logit_rmse,
            "relative_centered_coordinate_logit_rmse": relative_centered_error,
            "teacher_centered_logit_rms": teacher_centered_rms,
            "orthonormality_max_abs_error": orthonormality_error,
            "device": "cpu",
            "dtype": "float64",
            "gradient_free": True,
            "optimizer_used": False,
            "finite": True,
        }


build_dual_residual_basis = build_dual_aware_residual_basis
build_dual_basis = build_dual_aware_residual_basis
evaluate_candidate_basis_oracle = evaluate_dual_basis_oracle
evaluate_basis_oracle = evaluate_dual_basis_oracle


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_dual_aware_residual_basis",
    "build_dual_basis",
    "build_dual_residual_basis",
    "evaluate_basis_oracle",
    "evaluate_candidate_basis_oracle",
    "evaluate_dual_basis_oracle",
]
