"""Scalable deterministic activation-aware feature folding.

The represented encoding matrix ``M`` has shape ``[source_width, target_width]``
and exactly one nonzero per source row.  It is stored as a target assignment and
a coefficient for each source channel, never as a dense projector.  Coefficients
are normalized per target column, so ``M.T @ M == I`` and ``M.T`` is the
Moore-Penrose inverse of ``M``.

For row-major activations, ``encode(x)`` computes ``x @ M`` and ``decode(z)``
computes ``z @ M.T``.  A producer weight in canonical ``[out, in]`` form is
folded with ``M.T @ weight``; a consumer is folded with ``weight @ M``.  The
decoded rectangular path is therefore the source-space projection ``M @ M.T``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_MAX_DENSE_ELEMENTS = 1_000_000


class FeatureFoldError(ValueError):
    """Base class for feature-fold validation failures."""


class FeatureFoldShapeError(FeatureFoldError):
    """A width, activation, map, or runtime tensor has an invalid shape."""


class FeatureFoldFiniteError(FeatureFoldError):
    """An input, coefficient, intermediate, or output is non-finite."""


class FeatureFoldRankError(FeatureFoldError):
    """A requested fold cannot have full target-column rank."""


class FeatureFoldMemoryError(FeatureFoldError):
    """Explicit dense materialization exceeds its caller-provided guard."""


@dataclass(frozen=True, slots=True)
class FeatureFoldReport:
    """Coverage, balance, storage, and approximation evidence for a fold."""

    source_width: int
    target_width: int
    source_channels_covered: int
    coverage_fraction: float
    dropped_channels: int
    group_sizes: tuple[int, ...]
    group_loads: tuple[float, ...]
    coefficient_column_norms: tuple[float, ...]
    calibration_matrix_count: int
    calibration_rows: tuple[int, ...]
    calibration_total_energy: float
    sketch_size_per_matrix: int
    signature_width: int
    sketch_input_energy: float
    sketch_folded_energy: float
    sketch_retained_energy_fraction: float
    sketch_relative_reconstruction_error: float
    sketch_energy_relative_error: float
    chunk_size: int
    map_stored_elements: int
    map_stored_bytes: int
    dense_map_elements_avoided: int
    peak_activation_chunk_elements: int
    peak_signature_chunk_elements: int
    peak_similarity_chunk_elements: int
    persistent_builder_elements_upper_bound: int
    full_target_rank: bool
    columns_normalized: bool
    deterministic: bool
    sign_flips_allowed: bool
    negative_coefficient_count: int
    dense_materialized: bool = False

    @property
    def full_source_coverage(self) -> bool:
        return self.source_channels_covered == self.source_width

    @property
    def zero_dropped_channels(self) -> bool:
        return self.dropped_channels == 0

    @property
    def max_group_size(self) -> int:
        return max(self.group_sizes)

    @property
    def min_group_size(self) -> int:
        return min(self.group_sizes)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostics dictionary."""

        return {
            "source_width": self.source_width,
            "target_width": self.target_width,
            "source_channels_covered": self.source_channels_covered,
            "coverage_fraction": self.coverage_fraction,
            "full_source_coverage": self.full_source_coverage,
            "dropped_channels": self.dropped_channels,
            "zero_dropped_channels": self.zero_dropped_channels,
            "group_sizes": list(self.group_sizes),
            "group_loads": list(self.group_loads),
            "min_group_size": self.min_group_size,
            "max_group_size": self.max_group_size,
            "coefficient_column_norms": list(
                self.coefficient_column_norms
            ),
            "calibration_matrix_count": self.calibration_matrix_count,
            "calibration_rows": list(self.calibration_rows),
            "calibration_total_energy": self.calibration_total_energy,
            "sketch_size_per_matrix": self.sketch_size_per_matrix,
            "signature_width": self.signature_width,
            "sketch_input_energy": self.sketch_input_energy,
            "sketch_folded_energy": self.sketch_folded_energy,
            "sketch_retained_energy_fraction": (
                self.sketch_retained_energy_fraction
            ),
            "sketch_relative_reconstruction_error": (
                self.sketch_relative_reconstruction_error
            ),
            "sketch_energy_relative_error": (
                self.sketch_energy_relative_error
            ),
            "chunk_size": self.chunk_size,
            "memory_bounds": {
                "map_stored_elements": self.map_stored_elements,
                "map_stored_bytes": self.map_stored_bytes,
                "dense_map_elements_avoided": (
                    self.dense_map_elements_avoided
                ),
                "peak_activation_chunk_elements": (
                    self.peak_activation_chunk_elements
                ),
                "peak_signature_chunk_elements": (
                    self.peak_signature_chunk_elements
                ),
                "peak_similarity_chunk_elements": (
                    self.peak_similarity_chunk_elements
                ),
                "persistent_builder_elements_upper_bound": (
                    self.persistent_builder_elements_upper_bound
                ),
                "caller_calibration_storage_excluded": True,
            },
            "full_target_rank": self.full_target_rank,
            "columns_normalized": self.columns_normalized,
            "deterministic": self.deterministic,
            "sign_flips_allowed": self.sign_flips_allowed,
            "negative_coefficient_count": self.negative_coefficient_count,
            "dense_materialized": self.dense_materialized,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for neural_forge.hf_feature_fold."
        ) from exc
    return torch


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FeatureFoldShapeError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise FeatureFoldShapeError(f"{name} cannot exceed {maximum}.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise FeatureFoldShapeError(
            f"{name} must be a finite non-negative float."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureFoldShapeError(
            f"{name} must be a finite non-negative float."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise FeatureFoldShapeError(
            f"{name} must be a finite non-negative float."
        )
    return result


def _real_tensor(torch, value: Any, name: str, *, dimensions: int):
    if not isinstance(value, torch.Tensor):
        raise FeatureFoldShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise FeatureFoldShapeError(
            f"{name} must be a dense strided non-meta tensor."
        )
    if value.ndim != dimensions:
        raise FeatureFoldShapeError(
            f"{name} must be {dimensions}-dimensional."
        )
    if any(int(size) <= 0 for size in value.shape):
        raise FeatureFoldShapeError(f"{name} cannot have an empty dimension.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise FeatureFoldShapeError(
            f"{name} must have a real floating-point dtype."
        )
    return value


def _finite_runtime_tensor(torch, value: Any, name: str, *, width: int):
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise FeatureFoldShapeError(
            f"{name} must be a non-scalar torch.Tensor."
        )
    value = _real_tensor(torch, value, name, dimensions=value.ndim)
    if int(value.shape[-1]) != width:
        raise FeatureFoldShapeError(
            f"{name} last dimension must equal {width}."
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise FeatureFoldFiniteError(f"{name} contains NaN or infinity.")
    return value


class FeatureFoldMap:
    """One-nonzero-per-source sparse representation of a feature fold."""

    __slots__ = ("assignment", "coefficients", "report", "target_width")

    def __init__(
        self,
        assignment: Any,
        coefficients: Any,
        target_width: int,
        *,
        report: FeatureFoldReport | None = None,
    ) -> None:
        torch = _require_torch()
        target_width = _positive_int(target_width, "target_width")
        if not isinstance(assignment, torch.Tensor) or assignment.ndim != 1:
            raise FeatureFoldShapeError(
                "assignment must be a one-dimensional integer torch.Tensor."
            )
        if assignment.layout != torch.strided or assignment.device.type == "meta":
            raise FeatureFoldShapeError(
                "assignment must be a dense strided non-meta tensor."
            )
        if (
            assignment.dtype == torch.bool
            or torch.is_floating_point(assignment)
            or torch.is_complex(assignment)
        ):
            raise FeatureFoldShapeError("assignment must use an integer dtype.")
        source_width = int(assignment.numel())
        if source_width <= 0 or target_width > source_width:
            raise FeatureFoldShapeError(
                "Widths must satisfy 0 < target_width <= source_width."
            )
        if not isinstance(coefficients, torch.Tensor) or coefficients.ndim != 1:
            raise FeatureFoldShapeError(
                "coefficients must be a one-dimensional torch.Tensor."
            )
        if coefficients.layout != torch.strided or coefficients.device.type == "meta":
            raise FeatureFoldShapeError(
                "coefficients must be a dense strided non-meta tensor."
            )
        if int(coefficients.numel()) != source_width:
            raise FeatureFoldShapeError(
                "coefficients must contain one value per source channel."
            )
        if not torch.is_floating_point(coefficients) or torch.is_complex(
            coefficients
        ):
            raise FeatureFoldShapeError(
                "coefficients must have a real floating-point dtype."
            )

        mapping = assignment.detach().to(device="cpu", dtype=torch.long).clone()
        values = (
            coefficients.detach()
            .to(device="cpu", dtype=torch.float64)
            .clone()
        )
        if bool(((mapping < 0) | (mapping >= target_width)).any()):
            raise FeatureFoldShapeError(
                "assignment contains an out-of-range target index."
            )
        if not bool(torch.isfinite(values).all()):
            raise FeatureFoldFiniteError(
                "coefficients contain NaN or infinity."
            )
        if bool((values == 0.0).any()):
            raise FeatureFoldRankError(
                "Every source channel must have a nonzero coefficient."
            )

        counts = torch.bincount(mapping, minlength=target_width)
        if bool((counts == 0).any()):
            raise FeatureFoldRankError(
                "Every target column must have a nonempty source group."
            )
        squared_norms = torch.zeros(target_width, dtype=torch.float64)
        squared_norms.index_add_(0, mapping, values.square())
        if not bool(torch.isfinite(squared_norms).all()) or bool(
            (squared_norms <= 0.0).any()
        ):
            raise FeatureFoldRankError(
                "Coefficient columns do not have finite positive norm."
            )
        values = (values / torch.sqrt(squared_norms[mapping])).contiguous()
        if not bool(torch.isfinite(values).all()) or bool((values == 0.0).any()):
            raise FeatureFoldFiniteError(
                "Column normalization produced invalid coefficients."
            )

        self.assignment = mapping.contiguous()
        self.coefficients = values
        self.target_width = target_width
        self.report = report

    @property
    def source_width(self) -> int:
        return int(self.assignment.numel())

    @property
    def source_to_target(self):
        """Descriptive alias for the one-dimensional assignment tensor."""

        return self.assignment

    @property
    def coefficient(self):
        """Descriptive singular alias for per-source coefficients."""

        return self.coefficients

    @property
    def diagnostics(self) -> FeatureFoldReport | None:
        return self.report

    @property
    def stored_elements(self) -> int:
        return int(self.assignment.numel() + self.coefficients.numel())

    @property
    def group_sizes(self) -> tuple[int, ...]:
        torch = _require_torch()
        return tuple(
            int(value)
            for value in torch.bincount(
                self.assignment, minlength=self.target_width
            ).tolist()
        )

    def _runtime_mapping(self, inputs: Any):
        return (
            self.assignment.to(device=inputs.device),
            self.coefficients.to(device=inputs.device, dtype=inputs.dtype),
        )

    def encode(self, source_rows: Any):
        """Encode row-major ``[..., source_width]`` activations with ``x @ M``."""

        torch = _require_torch()
        source_rows = _finite_runtime_tensor(
            torch,
            source_rows,
            "source_rows",
            width=self.source_width,
        )
        mapping, coefficients = self._runtime_mapping(source_rows)
        flat = source_rows.reshape(-1, self.source_width)
        weighted = flat * coefficients.unsqueeze(0)
        folded = flat.new_zeros((flat.shape[0], self.target_width))
        folded = folded.index_add(1, mapping, weighted)
        result = folded.reshape(*source_rows.shape[:-1], self.target_width)
        if not bool(torch.isfinite(result.detach()).all()):
            raise FeatureFoldFiniteError("Encoding produced NaN or infinity.")
        return result

    def apply(self, source_rows: Any):
        """Alias for :meth:`encode`."""

        return self.encode(source_rows)

    def __call__(self, source_rows: Any):
        return self.encode(source_rows)

    def decode(self, target_rows: Any):
        """Apply ``M.T``, the pseudoinverse, to row-major target activations."""

        torch = _require_torch()
        target_rows = _finite_runtime_tensor(
            torch,
            target_rows,
            "target_rows",
            width=self.target_width,
        )
        mapping, coefficients = self._runtime_mapping(target_rows)
        flat = target_rows.reshape(-1, self.target_width)
        decoded = flat.index_select(1, mapping) * coefficients.unsqueeze(0)
        result = decoded.reshape(*target_rows.shape[:-1], self.source_width)
        if not bool(torch.isfinite(result.detach()).all()):
            raise FeatureFoldFiniteError("Decoding produced NaN or infinity.")
        return result

    def pseudoinverse_decode(self, target_rows: Any):
        """Descriptive alias for :meth:`decode`."""

        return self.decode(target_rows)

    def fold_producer_weight(self, source_weight: Any):
        """Return ``M.T @ W`` for producer ``W[out=source_width, in]``."""

        torch = _require_torch()
        source_weight = _real_tensor(
            torch,
            source_weight,
            "source_weight",
            dimensions=2,
        )
        if int(source_weight.shape[0]) != self.source_width:
            raise FeatureFoldShapeError(
                f"source_weight rows must equal {self.source_width}."
            )
        if not bool(torch.isfinite(source_weight.detach()).all()):
            raise FeatureFoldFiniteError(
                "source_weight contains NaN or infinity."
            )
        mapping, coefficients = self._runtime_mapping(source_weight)
        result = source_weight.new_zeros(
            (self.target_width, int(source_weight.shape[1]))
        )
        result = result.index_add(
            0,
            mapping,
            source_weight * coefficients.unsqueeze(1),
        )
        if not bool(torch.isfinite(result.detach()).all()):
            raise FeatureFoldFiniteError(
                "Producer folding produced NaN or infinity."
            )
        return result

    def fold_consumer_weight(self, source_weight: Any):
        """Return ``W @ M`` for consumer ``W[out, in=source_width]``."""

        torch = _require_torch()
        source_weight = _real_tensor(
            torch,
            source_weight,
            "source_weight",
            dimensions=2,
        )
        if int(source_weight.shape[1]) != self.source_width:
            raise FeatureFoldShapeError(
                f"source_weight columns must equal {self.source_width}."
            )
        return self.encode(source_weight)

    def materialize_dense(
        self,
        *,
        max_elements: int = DEFAULT_MAX_DENSE_ELEMENTS,
        dtype: Any | None = None,
        device: Any = "cpu",
    ):
        """Explicitly materialize ``M`` when a small-test guard permits it."""

        torch = _require_torch()
        max_elements = _positive_int(max_elements, "max_elements")
        elements = self.source_width * self.target_width
        if elements > max_elements:
            raise FeatureFoldMemoryError(
                f"Dense map needs {elements} elements, exceeding "
                f"max_elements={max_elements}."
            )
        if dtype is None:
            dtype = torch.float64
        probe = torch.empty((), dtype=dtype)
        if not torch.is_floating_point(probe) or torch.is_complex(probe):
            raise FeatureFoldShapeError("dtype must be a real floating dtype.")
        mapping = self.assignment.to(device=device)
        coefficients = self.coefficients.to(device=device, dtype=dtype)
        dense = torch.zeros(
            (self.source_width, self.target_width),
            dtype=dtype,
            device=device,
        )
        dense[
            torch.arange(self.source_width, device=device),
            mapping,
        ] = coefficients
        return dense

    def to_dense(self, **kwargs: Any):
        """Alias for the explicitly guarded dense helper."""

        return self.materialize_dense(**kwargs)


def _activation_matrices(torch, values: Any) -> tuple[Any, ...]:
    if isinstance(values, torch.Tensor):
        matrices = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        matrices = tuple(values)
    else:
        raise FeatureFoldShapeError(
            "activation_matrices must be a tensor or nonempty tensor sequence."
        )
    if not matrices:
        raise FeatureFoldShapeError(
            "activation_matrices must contain at least one matrix."
        )
    checked = []
    source_width = None
    for index, matrix in enumerate(matrices):
        matrix = _real_tensor(
            torch,
            matrix,
            f"activation_matrices[{index}]",
            dimensions=2,
        )
        if source_width is None:
            source_width = int(matrix.shape[1])
        elif int(matrix.shape[1]) != source_width:
            raise FeatureFoldShapeError(
                "All activation matrices must have the same source width."
            )
        checked.append(matrix)
    return tuple(checked)


def _column_chunk(torch, matrix: Any, indices: Any):
    device_indices = indices.to(device=matrix.device)
    return (
        matrix.detach()
        .index_select(1, device_indices)
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )


def _countsketch_signatures(
    torch,
    matrices: tuple[Any, ...],
    indices: Any,
    sketch_size: int,
):
    parts = []
    for matrix_index, matrix in enumerate(matrices):
        chunk = _column_chunk(torch, matrix, indices)
        if not bool(torch.isfinite(chunk).all()):
            raise FeatureFoldFiniteError(
                f"activation_matrices[{matrix_index}] contains NaN or infinity."
            )
        rows = torch.arange(int(matrix.shape[0]), dtype=torch.long)
        buckets = (rows * 37 + matrix_index * 17) % sketch_size
        parity = (rows * 104_729 + matrix_index * 8_191) % 2
        signs = parity.to(dtype=torch.float64).mul_(2.0).sub_(1.0)
        sketch = torch.zeros(
            (sketch_size, int(indices.numel())),
            dtype=torch.float64,
        )
        sketch.index_add_(0, buckets, chunk * signs.unsqueeze(1))
        parts.append(sketch.T)
    result = torch.cat(parts, dim=1).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise FeatureFoldFiniteError(
            "Activation sketching produced NaN or infinity."
        )
    return result


def _normalized_rows(torch, rows: Any):
    norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    return torch.where(norms > 0.0, rows / norms.clamp_min(1e-300), rows)


def build_feature_fold_map(
    activation_matrices: Any,
    target_width: int,
    *,
    source_width: int | None = None,
    chunk_size: int = 256,
    sketch_size: int = 32,
    balance_strength: float = 0.25,
    allow_sign_flip: bool = True,
) -> FeatureFoldMap:
    """Build a coupled sparse fold from one or more calibration matrices.

    Each matrix is interpreted as row-major ``[observations, source_width]``.
    Multiple matrices share one assignment, which keeps related paths such as
    Q/K or gate/up in identical target coordinates.  Source-channel chunks are
    copied to CPU float64, sketched, and released; no source-square or complete
    source-by-target array is built.
    """

    torch = _require_torch()
    matrices = _activation_matrices(torch, activation_matrices)
    inferred_source_width = int(matrices[0].shape[1])
    if source_width is not None:
        source_width = _positive_int(source_width, "source_width")
        if source_width != inferred_source_width:
            raise FeatureFoldShapeError(
                "source_width does not match the calibration matrices."
            )
    else:
        source_width = inferred_source_width
    target_width = _positive_int(
        target_width,
        "target_width",
        maximum=source_width,
    )
    chunk_size = _positive_int(chunk_size, "chunk_size")
    sketch_size = _positive_int(sketch_size, "sketch_size")
    balance_strength = _nonnegative_float(
        balance_strength,
        "balance_strength",
    )
    if not isinstance(allow_sign_flip, bool):
        raise FeatureFoldShapeError("allow_sign_flip must be boolean.")
    effective_chunk = min(chunk_size, source_width)
    signature_width = sketch_size * len(matrices)

    energies = torch.zeros(source_width, dtype=torch.float64)
    for start in range(0, source_width, effective_chunk):
        stop = min(start + effective_chunk, source_width)
        indices = torch.arange(start, stop, dtype=torch.long)
        for matrix_index, matrix in enumerate(matrices):
            chunk = _column_chunk(torch, matrix, indices)
            if not bool(torch.isfinite(chunk).all()):
                raise FeatureFoldFiniteError(
                    f"activation_matrices[{matrix_index}] contains NaN or infinity."
                )
            energies[start:stop] += chunk.square().sum(dim=0)
    if not bool(torch.isfinite(energies).all()):
        raise FeatureFoldFiniteError(
            "Calibration activation energies contain NaN or infinity."
        )

    order = torch.argsort(energies, descending=True, stable=True)
    seed_indices = order[:target_width]
    assignment = torch.full((source_width,), -1, dtype=torch.long)
    orientation = torch.ones(source_width, dtype=torch.float64)
    assignment[seed_indices] = torch.arange(target_width, dtype=torch.long)
    group_sizes = torch.ones(target_width, dtype=torch.long)
    group_loads = energies.index_select(0, seed_indices).clone()

    seed_signatures = _countsketch_signatures(
        torch,
        matrices,
        seed_indices,
        sketch_size,
    )
    seed_scales = torch.sqrt(energies.index_select(0, seed_indices))
    seed_scales = torch.where(
        seed_scales > 0.0,
        seed_scales,
        torch.ones_like(seed_scales),
    )
    prototype_sums = seed_signatures * seed_scales.unsqueeze(1)
    prototypes = _normalized_rows(torch, prototype_sums)

    base_capacity, extra_capacity = divmod(source_width, target_width)
    capacities = torch.full(
        (target_width,),
        base_capacity,
        dtype=torch.long,
    )
    capacities[:extra_capacity] += 1
    ideal_energy = float(energies.sum().item()) / target_width
    remaining = order[target_width:]
    for start in range(0, int(remaining.numel()), effective_chunk):
        indices = remaining[start : start + effective_chunk]
        signatures = _countsketch_signatures(
            torch,
            matrices,
            indices,
            sketch_size,
        )
        normalized = _normalized_rows(torch, signatures)
        similarities = normalized @ prototypes.T
        chosen_targets = torch.empty(int(indices.numel()), dtype=torch.long)
        chosen_signs = torch.empty(int(indices.numel()), dtype=torch.float64)

        for local_index, source_index_tensor in enumerate(indices):
            source_index = int(source_index_tensor.item())
            admissible = group_sizes < capacities
            if not bool(admissible.any()):
                raise FeatureFoldRankError(
                    "Balanced grouping exhausted all target capacities."
                )
            load_pressure = (
                group_sizes.to(torch.float64)
                / capacities.to(torch.float64)
            )
            if ideal_energy > 0.0:
                energy_pressure = group_loads / ideal_energy
            else:
                energy_pressure = load_pressure
            penalty = balance_strength * (
                0.5 * load_pressure + 0.5 * energy_pressure
            )
            similarity_scores = similarities[local_index]
            if allow_sign_flip:
                similarity_scores = similarity_scores.abs()
            scores = similarity_scores - penalty
            scores = torch.where(
                admissible,
                scores,
                torch.full_like(scores, -torch.inf),
            )
            target = int(torch.argmax(scores).item())
            sign = 1.0
            if allow_sign_flip and float(similarities[local_index, target]) < 0.0:
                sign = -1.0
            assignment[source_index] = target
            orientation[source_index] = sign
            chosen_targets[local_index] = target
            chosen_signs[local_index] = sign
            group_sizes[target] += 1
            group_loads[target] += energies[source_index]

        update_scales = torch.sqrt(energies.index_select(0, indices))
        update_scales = torch.where(
            update_scales > 0.0,
            update_scales,
            torch.ones_like(update_scales),
        )
        updates = (
            signatures
            * chosen_signs.unsqueeze(1)
            * update_scales.unsqueeze(1)
        )
        prototype_sums.index_add_(0, chosen_targets, updates)
        prototypes = _normalized_rows(torch, prototype_sums)

    if bool((assignment < 0).any()):
        raise FeatureFoldRankError(
            "Grouping failed to assign every source channel."
        )

    positive_energies = energies[energies > 0.0]
    if int(positive_energies.numel()) > 0:
        energy_floor = max(
            float(positive_energies.min().item()) * 1e-12,
            torch.finfo(torch.float64).tiny,
        )
        coefficient_scales = torch.sqrt(energies.clamp_min(energy_floor))
    else:
        coefficient_scales = torch.ones_like(energies)
    raw_coefficients = orientation * coefficient_scales
    squared_norms = torch.zeros(target_width, dtype=torch.float64)
    squared_norms.index_add_(
        0,
        assignment,
        raw_coefficients.square(),
    )
    if bool((squared_norms <= 0.0).any()) or not bool(
        torch.isfinite(squared_norms).all()
    ):
        raise FeatureFoldRankError(
            "A target group has no finite coefficient norm."
        )
    coefficients = raw_coefficients / torch.sqrt(squared_norms[assignment])
    if bool((coefficients == 0.0).any()) or not bool(
        torch.isfinite(coefficients).all()
    ):
        raise FeatureFoldFiniteError(
            "Coefficient normalization produced NaN, infinity, or zero."
        )

    folded_sketches = torch.zeros(
        (target_width, signature_width),
        dtype=torch.float64,
    )
    sketch_input_energy = 0.0
    for start in range(0, source_width, effective_chunk):
        stop = min(start + effective_chunk, source_width)
        indices = torch.arange(start, stop, dtype=torch.long)
        signatures = _countsketch_signatures(
            torch,
            matrices,
            indices,
            sketch_size,
        )
        sketch_input_energy += float(signatures.square().sum().item())
        folded_sketches.index_add_(
            0,
            assignment[start:stop],
            signatures * coefficients[start:stop].unsqueeze(1),
        )
    sketch_folded_energy = float(folded_sketches.square().sum().item())
    if not math.isfinite(sketch_input_energy) or not math.isfinite(
        sketch_folded_energy
    ):
        raise FeatureFoldFiniteError(
            "Sketch diagnostics contain NaN or infinity."
        )
    if sketch_input_energy > 0.0:
        retained = min(
            1.0,
            max(0.0, sketch_folded_energy / sketch_input_energy),
        )
        sketch_error = math.sqrt(max(0.0, 1.0 - retained))
    else:
        retained = 1.0
        sketch_error = 0.0
    total_energy = float(energies.sum().item())
    if total_energy > 0.0:
        sketch_energy_error = abs(sketch_input_energy - total_energy) / total_energy
    else:
        sketch_energy_error = 0.0

    exact_group_loads = torch.zeros(target_width, dtype=torch.float64)
    exact_group_loads.index_add_(0, assignment, energies)
    exact_squared_norms = torch.zeros(target_width, dtype=torch.float64)
    exact_squared_norms.index_add_(0, assignment, coefficients.square())
    column_norms = torch.sqrt(exact_squared_norms)
    rows = tuple(int(matrix.shape[0]) for matrix in matrices)
    report = FeatureFoldReport(
        source_width=source_width,
        target_width=target_width,
        source_channels_covered=int((assignment >= 0).sum().item()),
        coverage_fraction=float((assignment >= 0).sum().item()) / source_width,
        dropped_channels=int((assignment < 0).sum().item()),
        group_sizes=tuple(int(value) for value in group_sizes.tolist()),
        group_loads=tuple(float(value) for value in exact_group_loads.tolist()),
        coefficient_column_norms=tuple(
            float(value) for value in column_norms.tolist()
        ),
        calibration_matrix_count=len(matrices),
        calibration_rows=rows,
        calibration_total_energy=total_energy,
        sketch_size_per_matrix=sketch_size,
        signature_width=signature_width,
        sketch_input_energy=sketch_input_energy,
        sketch_folded_energy=sketch_folded_energy,
        sketch_retained_energy_fraction=retained,
        sketch_relative_reconstruction_error=sketch_error,
        sketch_energy_relative_error=sketch_energy_error,
        chunk_size=effective_chunk,
        map_stored_elements=source_width * 2,
        map_stored_bytes=source_width * 16,
        dense_map_elements_avoided=source_width * target_width,
        peak_activation_chunk_elements=max(rows) * effective_chunk,
        peak_signature_chunk_elements=signature_width * effective_chunk,
        peak_similarity_chunk_elements=target_width * effective_chunk,
        persistent_builder_elements_upper_bound=(
            source_width * 6
            + target_width * (signature_width * 3 + 8)
        ),
        full_target_rank=bool((group_sizes > 0).all()),
        columns_normalized=bool(
            torch.allclose(
                column_norms,
                torch.ones_like(column_norms),
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        deterministic=True,
        sign_flips_allowed=allow_sign_flip,
        negative_coefficient_count=int((coefficients < 0.0).sum().item()),
    )
    if not report.full_source_coverage or not report.full_target_rank:
        raise FeatureFoldRankError(
            "Fold diagnostics do not prove full source coverage and target rank."
        )
    if not report.columns_normalized:
        raise FeatureFoldRankError(
            "Fold coefficient columns failed normalization."
        )
    return FeatureFoldMap(
        assignment,
        coefficients,
        target_width,
        report=report,
    )


def build_coupled_feature_fold_map(
    activation_matrices: Any,
    target_width: int,
    **kwargs: Any,
) -> FeatureFoldMap:
    """Build one shared map for multiple same-width activation paths."""

    return build_feature_fold_map(
        activation_matrices,
        target_width,
        **kwargs,
    )


build_activation_aware_feature_fold = build_feature_fold_map


__all__ = [
    "DEFAULT_MAX_DENSE_ELEMENTS",
    "FeatureFoldError",
    "FeatureFoldFiniteError",
    "FeatureFoldMap",
    "FeatureFoldMemoryError",
    "FeatureFoldRankError",
    "FeatureFoldReport",
    "FeatureFoldShapeError",
    "build_activation_aware_feature_fold",
    "build_coupled_feature_fold_map",
    "build_feature_fold_map",
]
