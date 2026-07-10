"""Deterministic depth-basis representations of same-shaped linear layers.

The fitting path is deliberately detached.  It streams canonical ``[out, in]``
features through a bounded CPU-float64 layer Gram matrix, then writes only the
selected basis rows into final dtype/device storage.  The resulting module
parameters are ordinary trainable PyTorch parameters, so gradients flow during
later runtime use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_FEATURE_CHUNK_SIZE = 65_536


class DepthBasisLinearError(ValueError):
    """Base class for depth-basis validation and decomposition failures."""


class DepthBasisShapeError(DepthBasisLinearError):
    """An operator, factor, or logical-layer mapping has an invalid shape."""


class DepthBasisFiniteError(DepthBasisLinearError):
    """An input, intermediate, or result contains NaN or infinity."""


class DepthBasisRankError(DepthBasisLinearError):
    """A rank or energy selector is invalid."""


class DepthBasisBudgetError(DepthBasisLinearError):
    """A parameter budget cannot represent even one basis component."""


@dataclass(frozen=True, slots=True)
class DepthBasisReport:
    """Selection, approximation, and physical-storage evidence."""

    logical_layers: int
    coefficient_rows: int
    in_features: int
    out_features: int
    has_bias: bool
    rank: int
    max_rank: int
    selection: str
    requested_energy: float | None
    retained_energy_fraction: float
    singular_values: tuple[float, ...]
    parameter_budget: int | None
    budget_headroom: int | None
    dense_equivalent_parameters: int
    basis_parameters: int
    coefficient_parameters: int
    unique_parameters: int
    squared_reconstruction_error: float
    relative_frobenius_error: float
    full_rank_reconstruction: bool
    algorithm: str
    feature_count: int
    requested_feature_chunk_size: int
    feature_chunk_size: int
    feature_chunks: int
    gram_rows: int
    gram_elements: int
    estimated_peak_float64_elements: int
    estimated_peak_float64_bytes: int
    fitting_passes: int
    materialized_operator_stack: bool
    materialized_full_right_singular_matrix: bool
    numerical_rank: int
    degenerate_eigenspaces: int
    deterministic: bool = True

    @property
    def logical_parameter_count(self) -> int:
        """Number of parameters in independent logical ``nn.Linear`` layers."""

        return self.dense_equivalent_parameters

    @property
    def unique_parameter_count(self) -> int:
        """Number of physically stored basis and coefficient parameters."""

        return self.unique_parameters

    @property
    def parameters_saved(self) -> int:
        return self.dense_equivalent_parameters - self.unique_parameters

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return {
            "logical_layers": self.logical_layers,
            "coefficient_rows": self.coefficient_rows,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.has_bias,
            "rank": self.rank,
            "max_rank": self.max_rank,
            "selection": self.selection,
            "requested_energy": self.requested_energy,
            "retained_energy_fraction": self.retained_energy_fraction,
            "singular_values": list(self.singular_values),
            "parameter_budget": self.parameter_budget,
            "budget_headroom": self.budget_headroom,
            "logical_parameter_count": self.logical_parameter_count,
            "dense_equivalent_parameters": self.dense_equivalent_parameters,
            "basis_parameters": self.basis_parameters,
            "coefficient_parameters": self.coefficient_parameters,
            "unique_parameter_count": self.unique_parameter_count,
            "parameters_saved": self.parameters_saved,
            "squared_reconstruction_error": self.squared_reconstruction_error,
            "relative_frobenius_error": self.relative_frobenius_error,
            "full_rank_reconstruction": self.full_rank_reconstruction,
            "algorithm": self.algorithm,
            "feature_count": self.feature_count,
            "requested_feature_chunk_size": self.requested_feature_chunk_size,
            "feature_chunk_size": self.feature_chunk_size,
            "feature_chunks": self.feature_chunks,
            "gram_rows": self.gram_rows,
            "gram_elements": self.gram_elements,
            "estimated_peak_float64_elements": (
                self.estimated_peak_float64_elements
            ),
            "estimated_peak_float64_bytes": self.estimated_peak_float64_bytes,
            "fitting_passes": self.fitting_passes,
            "materialized_operator_stack": self.materialized_operator_stack,
            "materialized_full_right_singular_matrix": (
                self.materialized_full_right_singular_matrix
            ),
            "numerical_rank": self.numerical_rank,
            "degenerate_eigenspaces": self.degenerate_eigenspaces,
            "deterministic": self.deterministic,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _finite_dense_tensor(value: Any, name: str, *, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise DepthBasisShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise DepthBasisShapeError(f"{name} must be a dense non-meta tensor.")
    if value.ndim != dimensions:
        raise DepthBasisShapeError(
            f"{name} must be {dimensions}-dimensional, got {value.ndim}."
        )
    if any(int(size) <= 0 for size in value.shape):
        raise DepthBasisShapeError(f"{name} cannot have an empty dimension.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise DepthBasisShapeError(f"{name} must have a real floating dtype.")
    if not bool(torch.isfinite(value.detach()).all()):
        raise DepthBasisFiniteError(f"{name} contains NaN or infinity.")
    return value


def _validate_selectors(
    rank: int | None,
    energy: float | None,
    parameter_budget: int | None,
) -> None:
    if sum(value is not None for value in (rank, energy, parameter_budget)) > 1:
        raise DepthBasisLinearError(
            "Specify at most one of rank, energy, or parameter_budget."
        )


def _positive_rank(value: Any, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DepthBasisRankError("rank must be a positive integer.")
    if value > maximum:
        raise DepthBasisRankError(f"rank cannot exceed {maximum}.")
    return value


def _energy_target(value: Any) -> float:
    if isinstance(value, bool):
        raise DepthBasisRankError("energy must be in the interval (0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DepthBasisRankError(
            "energy must be in the interval (0, 1]."
        ) from exc
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise DepthBasisRankError("energy must be in the interval (0, 1].")
    return result


def _positive_budget(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DepthBasisBudgetError(
            "parameter_budget must be a positive integer."
        )
    return value


def _positive_feature_chunk_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DepthBasisShapeError(
            "feature_chunk_size must be a positive integer."
        )
    return value


def _normalized_coefficient_rows(
    value: Sequence[int] | Tensor | None,
    logical_layers: int,
) -> tuple[tuple[int, ...], int]:
    if value is None:
        rows = tuple(range(logical_layers))
        return rows, logical_layers
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise DepthBasisShapeError(
                "tied_coefficient_rows must be one-dimensional."
            )
        raw = value.detach().to(device="cpu").tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise DepthBasisShapeError(
            "tied_coefficient_rows must be an integer sequence."
        )
    if len(raw) != logical_layers:
        raise DepthBasisShapeError(
            "tied_coefficient_rows must contain one value per logical layer."
        )

    normalized: list[int] = []
    labels: dict[int, int] = {}
    for label in raw:
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise DepthBasisShapeError(
                "tied_coefficient_rows values must be non-negative integers."
            )
        if label not in labels:
            labels[label] = len(labels)
        normalized.append(labels[label])
    return tuple(normalized), len(labels)


def _validate_linears(
    linears: Sequence[nn.Linear],
) -> tuple[tuple[nn.Linear, ...], int, int, bool, torch.dtype, torch.device]:
    if not isinstance(linears, Sequence) or isinstance(linears, (str, bytes)):
        raise DepthBasisShapeError("linears must be a sequence of nn.Linear modules.")
    modules = tuple(linears)
    if not modules:
        raise DepthBasisShapeError("linears must contain at least one module.")
    if not all(isinstance(module, nn.Linear) for module in modules):
        raise DepthBasisShapeError("Every operator must be an nn.Linear module.")

    first = modules[0]
    in_features = int(first.in_features)
    out_features = int(first.out_features)
    has_bias = first.bias is not None
    dtype = first.weight.dtype
    device = first.weight.device
    for index, module in enumerate(modules):
        prefix = f"linears[{index}]"
        weight = _finite_dense_tensor(module.weight, f"{prefix}.weight", dimensions=2)
        if tuple(weight.shape) != (out_features, in_features):
            raise DepthBasisShapeError(
                "All linear weights must have the same canonical [out, in] shape."
            )
        if weight.dtype != dtype or weight.device != device:
            raise DepthBasisShapeError(
                "All linear weights must have the same dtype and device."
            )
        if (module.bias is not None) != has_bias:
            raise DepthBasisShapeError(
                "All linear operators must use the same optional-bias topology."
            )
        if module.bias is not None:
            bias = _finite_dense_tensor(
                module.bias,
                f"{prefix}.bias",
                dimensions=1,
            )
            if tuple(bias.shape) != (out_features,):
                raise DepthBasisShapeError(
                    f"{prefix}.bias must have shape ({out_features},)."
                )
            if bias.dtype != dtype or bias.device != device:
                raise DepthBasisShapeError(
                    "Every bias must match its weight dtype and device."
                )
    return modules, in_features, out_features, has_bias, dtype, device


def stack_linear_operators(linears: Sequence[nn.Linear]) -> Tensor:
    """Return dense canonical rows for tests and small diagnostic helpers.

    Production fitting does not call this function because its allocation is
    proportional to ``logical_layers * flattened_operator_features``.
    """

    modules, _, _, has_bias, _, _ = _validate_linears(linears)
    with torch.no_grad():
        rows = []
        for module in modules:
            parts = [module.weight.detach().to(device="cpu", dtype=torch.float64).reshape(-1)]
            if has_bias and module.bias is not None:
                parts.append(
                    module.bias.detach().to(device="cpu", dtype=torch.float64)
                )
            rows.append(torch.cat(parts))
        result = torch.stack(rows).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise DepthBasisFiniteError(
                "Canonical operator stack contains NaN or infinity."
            )
        return result


def _canonical_feature_chunk(
    modules: tuple[nn.Linear, ...],
    *,
    start: int,
    end: int,
    weight_features: int,
    has_bias: bool,
) -> Tensor:
    """Copy one canonical feature interval to detached CPU float64 storage."""

    chunk = torch.empty((len(modules), end - start), dtype=torch.float64)
    weight_end = min(end, weight_features)
    for layer_index, module in enumerate(modules):
        if start < weight_end:
            weight = module.weight.detach()
            if weight.is_contiguous():
                source = weight.view(-1)[start:weight_end]
                chunk[layer_index, : weight_end - start].copy_(source)
            else:
                position = start
                while position < weight_end:
                    row, column = divmod(position, int(module.in_features))
                    take = min(
                        weight_end - position,
                        int(module.in_features) - column,
                    )
                    destination = position - start
                    chunk[
                        layer_index,
                        destination : destination + take,
                    ].copy_(weight[row, column : column + take])
                    position += take
        if end > weight_features:
            if not has_bias or module.bias is None:  # pragma: no cover - validated
                raise DepthBasisShapeError("A required linear bias is missing.")
            bias_start = max(start, weight_features) - weight_features
            bias_end = end - weight_features
            destination_start = max(start, weight_features) - start
            chunk[layer_index, destination_start:].copy_(
                module.bias.detach()[bias_start:bias_end]
            )
    if not bool(torch.isfinite(chunk).all()):
        raise DepthBasisFiniteError(
            f"Canonical feature chunk [{start}, {end}) contains NaN or infinity."
        )
    return chunk


def _weighted_group_chunk(
    source_chunk: Tensor,
    mapping: Tensor,
    inverse_sqrt_counts: Tensor,
) -> Tensor:
    rows = int(inverse_sqrt_counts.numel())
    grouped = torch.zeros(
        (rows, int(source_chunk.shape[1])),
        dtype=torch.float64,
    )
    grouped.index_add_(0, mapping, source_chunk)
    grouped.mul_(inverse_sqrt_counts.unsqueeze(1))
    if not bool(torch.isfinite(grouped).all()):
        raise DepthBasisFiniteError(
            "Weighted coefficient-row feature chunk is non-finite."
        )
    return grouped


def _canonical_subspace(projector: Tensor, width: int) -> Tensor:
    """Choose a deterministic orthonormal basis from a small projector."""

    dimension = int(projector.shape[0])
    acceptance = torch.finfo(torch.float64).eps * max(1, dimension) * 512.0
    selected: list[Tensor] = []
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
        raise DepthBasisRankError(
            "Could not canonicalize a degenerate Gram eigenspace."
        )
    return torch.stack(selected, dim=1).contiguous()


def _canonical_eigh(
    gram: Tensor,
    *,
    feature_count: int,
) -> tuple[Tensor, Tensor, int, float]:
    """Eigendecompose a layer Gram matrix with deterministic subspace bases."""

    gram = ((gram + gram.T) * 0.5).contiguous()
    if not bool(torch.isfinite(gram).all()):
        raise DepthBasisFiniteError("Layer Gram matrix contains NaN or infinity.")
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    except RuntimeError as exc:
        raise DepthBasisRankError(
            "Deterministic float64 layer-Gram eigendecomposition failed."
        ) from exc
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(
        torch.isfinite(eigenvectors).all()
    ):
        raise DepthBasisFiniteError(
            "Layer-Gram eigendecomposition produced NaN or infinity."
        )

    eigenvalues = torch.flip(eigenvalues, dims=(0,)).contiguous()
    eigenvectors = torch.flip(eigenvectors, dims=(1,)).contiguous()
    scale = max(1.0, float(torch.abs(eigenvalues[0]).item()))
    accumulation_scale = max(int(gram.shape[0]), feature_count)
    tolerance = (
        torch.finfo(torch.float64).eps * scale * accumulation_scale * 64.0
    )
    minimum = float(eigenvalues[-1].item())
    if minimum < -tolerance:
        raise DepthBasisRankError(
            "Layer Gram matrix is not positive semidefinite within float64 "
            f"tolerance: minimum eigenvalue {minimum:.6g}."
        )
    eigenvalues.clamp_(min=0.0)

    degenerate = 0
    start = 0
    while start < int(eigenvalues.numel()):
        end = start + 1
        while end < int(eigenvalues.numel()) and (
            abs(float(eigenvalues[end].item() - eigenvalues[start].item()))
            <= tolerance
        ):
            end += 1
        width = end - start
        if width > 1:
            projector = (
                eigenvectors[:, start:end] @ eigenvectors[:, start:end].T
            )
            eigenvectors[:, start:end] = _canonical_subspace(projector, width)
            degenerate += 1
        else:
            vector = eigenvectors[:, start]
            pivot = int(torch.argmax(torch.abs(vector)).item())
            if float(vector[pivot].item()) < 0.0:
                eigenvectors[:, start].neg_()
        start = end

    component_energy = torch.sum(
        eigenvectors * (gram @ eigenvectors),
        dim=0,
    )
    if float(component_energy.min().item()) < -tolerance:
        raise DepthBasisRankError(
            "Canonical Gram component energy is materially negative."
        )
    component_energy.clamp_(min=0.0)
    component_energy[component_energy <= tolerance] = 0.0
    singular_values = torch.sqrt(component_energy).contiguous()
    return singular_values, eigenvectors.contiguous(), degenerate, tolerance


def _rank_for_energy(singular_values: Tensor, energy: float) -> int:
    if float(singular_values[0].item()) == 0.0:
        return 1
    if energy == 1.0:
        return int(singular_values.numel())
    scaled = (singular_values / singular_values[0]).square()
    cumulative = torch.cumsum(scaled, dim=0) / scaled.sum()
    return min(
        int(torch.searchsorted(cumulative, energy, right=False).item()) + 1,
        int(singular_values.numel()),
    )


class DepthBasisLinear(nn.Module):
    """Execute logical linear layer ``i`` from coefficient-weighted bases."""

    basis_weight: nn.Parameter
    basis_bias: nn.Parameter | None
    coefficients: nn.Parameter
    layer_to_coefficient: Tensor

    def __init__(
        self,
        basis_weight: Tensor,
        coefficients: Tensor,
        basis_bias: Tensor | None = None,
        *,
        layer_to_coefficient: Sequence[int] | Tensor | None = None,
        report: DepthBasisReport | None = None,
    ) -> None:
        super().__init__()
        basis_weight = _finite_dense_tensor(
            basis_weight,
            "basis_weight",
            dimensions=3,
        )
        coefficients = _finite_dense_tensor(
            coefficients,
            "coefficients",
            dimensions=2,
        )
        rank, out_features, in_features = map(int, basis_weight.shape)
        if tuple(coefficients.shape[1:]) != (rank,):
            raise DepthBasisShapeError(
                "coefficients columns must equal the number of bases."
            )
        if coefficients.dtype != basis_weight.dtype or (
            coefficients.device != basis_weight.device
        ):
            raise DepthBasisShapeError(
                "basis_weight and coefficients must share dtype and device."
            )
        if basis_bias is not None:
            basis_bias = _finite_dense_tensor(
                basis_bias,
                "basis_bias",
                dimensions=2,
            )
            if tuple(basis_bias.shape) != (rank, out_features):
                raise DepthBasisShapeError(
                    f"basis_bias must have shape ({rank}, {out_features})."
                )
            if basis_bias.dtype != basis_weight.dtype or (
                basis_bias.device != basis_weight.device
            ):
                raise DepthBasisShapeError(
                    "basis_bias must match basis_weight dtype and device."
                )

        coefficient_rows = int(coefficients.shape[0])
        if layer_to_coefficient is None:
            mapping = tuple(range(coefficient_rows))
            logical_layers = coefficient_rows
        else:
            if isinstance(layer_to_coefficient, Tensor):
                if layer_to_coefficient.ndim != 1:
                    raise DepthBasisShapeError(
                        "layer_to_coefficient must be one-dimensional."
                    )
                raw_mapping = layer_to_coefficient.detach().cpu().tolist()
            elif isinstance(layer_to_coefficient, Sequence) and not isinstance(
                layer_to_coefficient,
                (str, bytes),
            ):
                raw_mapping = list(layer_to_coefficient)
            else:
                raise DepthBasisShapeError(
                    "layer_to_coefficient must be an integer sequence."
                )
            if not raw_mapping:
                raise DepthBasisShapeError("At least one logical layer is required.")
            if any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < coefficient_rows
                for row in raw_mapping
            ):
                raise DepthBasisShapeError(
                    "layer_to_coefficient contains an invalid coefficient row."
                )
            mapping = tuple(raw_mapping)
            logical_layers = len(mapping)
            if set(mapping) != set(range(coefficient_rows)):
                raise DepthBasisShapeError(
                    "Every stored coefficient row must be used by a logical layer."
                )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.logical_layers = logical_layers
        self.coefficient_rows = coefficient_rows
        self.basis_weight = nn.Parameter(basis_weight.detach().clone().contiguous())
        self.coefficients = nn.Parameter(coefficients.detach().clone().contiguous())
        if basis_bias is None:
            self.register_parameter("basis_bias", None)
        else:
            self.basis_bias = nn.Parameter(basis_bias.detach().clone().contiguous())
        self.register_buffer(
            "layer_to_coefficient",
            torch.tensor(mapping, dtype=torch.long, device=basis_weight.device),
        )
        self.report = report

    @property
    def has_bias(self) -> bool:
        return self.basis_bias is not None

    @classmethod
    def from_linears(
        cls,
        linears: Sequence[nn.Linear],
        *,
        rank: int | None = None,
        energy: float | None = None,
        parameter_budget: int | None = None,
        tied_coefficient_rows: Sequence[int] | Tensor | None = None,
        feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
    ) -> DepthBasisLinear:
        """Fit a detached, bounded-memory basis from linear modules."""

        return factorize_linears(
            linears,
            rank=rank,
            energy=energy,
            parameter_budget=parameter_budget,
            tied_coefficient_rows=tied_coefficient_rows,
            feature_chunk_size=feature_chunk_size,
        )

    def _coefficient_row(self, layer_index: int) -> int:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.logical_layers
        ):
            raise DepthBasisShapeError(
                f"layer_index must be an integer in [0, {self.logical_layers})."
            )
        return int(self.layer_to_coefficient[layer_index].item())

    def materialize_weight(self, layer_index: int) -> tuple[Tensor, Tensor | None]:
        """Return one logical weight and optional bias without detaching gradients."""

        row = self._coefficient_row(layer_index)
        coefficient = self.coefficients[row]
        weight = torch.sum(
            coefficient[:, None, None] * self.basis_weight,
            dim=0,
        )
        bias = None
        if self.basis_bias is not None:
            bias = torch.sum(coefficient[:, None] * self.basis_bias, dim=0)
        return weight, bias

    def forward(self, inputs: Tensor, layer_index: int) -> Tensor:
        if not isinstance(inputs, Tensor) or inputs.ndim < 1:
            raise DepthBasisShapeError("inputs must be a non-scalar torch.Tensor.")
        if int(inputs.shape[-1]) != self.in_features:
            raise DepthBasisShapeError(
                f"inputs last dimension must equal {self.in_features}."
            )
        if inputs.dtype != self.basis_weight.dtype or (
            inputs.device != self.basis_weight.device
        ):
            raise DepthBasisShapeError(
                "inputs must match the basis parameter dtype and device."
            )
        weight, bias = self.materialize_weight(layer_index)
        return F.linear(inputs, weight, bias)

    def materialize(self) -> list[nn.Linear]:
        """Expand every logical layer into an independent trainable ``nn.Linear``."""

        modules: list[nn.Linear] = []
        with torch.no_grad():
            for layer_index in range(self.logical_layers):
                weight, bias = self.materialize_weight(layer_index)
                linear = nn.Linear(
                    self.in_features,
                    self.out_features,
                    bias=bias is not None,
                    device=weight.device,
                    dtype=weight.dtype,
                )
                linear.weight.copy_(weight.detach())
                if bias is not None and linear.bias is not None:
                    linear.bias.copy_(bias.detach())
                linear.train(self.training)
                modules.append(linear)
        return modules

    def to_linears(self) -> list[nn.Linear]:
        return self.materialize()

    def extra_repr(self) -> str:
        return (
            f"logical_layers={self.logical_layers}, in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={self.rank}, "
            f"coefficient_rows={self.coefficient_rows}, bias={self.has_bias}"
        )


def factorize_linears(
    linears: Sequence[nn.Linear],
    *,
    rank: int | None = None,
    energy: float | None = None,
    parameter_budget: int | None = None,
    tied_coefficient_rows: Sequence[int] | Tensor | None = None,
    feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
) -> DepthBasisLinear:
    """Fit a deterministic basis through bounded layer-Gram accumulation."""

    _validate_selectors(rank, energy, parameter_budget)
    requested_chunk_size = _positive_feature_chunk_size(feature_chunk_size)
    (
        modules,
        in_features,
        out_features,
        has_bias,
        dtype,
        device,
    ) = _validate_linears(linears)
    mapping, coefficient_row_count = _normalized_coefficient_rows(
        tied_coefficient_rows,
        len(modules),
    )

    with torch.no_grad():
        weight_features = out_features * in_features
        feature_count = weight_features + (out_features if has_bias else 0)
        effective_chunk_size = min(requested_chunk_size, feature_count)
        feature_chunks = math.ceil(feature_count / effective_chunk_size)
        mapping_tensor = torch.tensor(mapping, dtype=torch.long)
        counts = torch.bincount(
            mapping_tensor,
            minlength=coefficient_row_count,
        ).to(torch.float64)
        inverse_sqrt_counts = torch.rsqrt(counts)
        gram = torch.zeros(
            (coefficient_row_count, coefficient_row_count),
            dtype=torch.float64,
        )
        source_energy = 0.0
        for start in range(0, feature_count, effective_chunk_size):
            end = min(start + effective_chunk_size, feature_count)
            source_chunk = _canonical_feature_chunk(
                modules,
                start=start,
                end=end,
                weight_features=weight_features,
                has_bias=has_bias,
            )
            flattened_chunk = source_chunk.view(-1)
            chunk_energy = float(
                torch.dot(flattened_chunk, flattened_chunk).item()
            )
            if not math.isfinite(chunk_energy):
                raise DepthBasisFiniteError(
                    "Canonical feature energy overflowed float64."
                )
            source_energy += chunk_energy
            if not math.isfinite(source_energy):
                raise DepthBasisFiniteError(
                    "Accumulated source energy overflowed float64."
                )
            grouped_chunk = _weighted_group_chunk(
                source_chunk,
                mapping_tensor,
                inverse_sqrt_counts,
            )
            gram.add_(grouped_chunk @ grouped_chunk.T)

        (
            all_singular_values,
            all_left_vectors,
            degenerate_eigenspaces,
            gram_tolerance,
        ) = _canonical_eigh(gram, feature_count=feature_count)
        max_rank = min(coefficient_row_count, feature_count)
        singular_values = all_singular_values[:max_rank]
        left_vectors = all_left_vectors[:, :max_rank]

        requested_energy = None
        budget = None
        if rank is not None:
            selected_rank = _positive_rank(rank, max_rank)
            selection = "rank"
        elif energy is not None:
            requested_energy = _energy_target(energy)
            selected_rank = _rank_for_energy(singular_values, requested_energy)
            selection = "energy"
        elif parameter_budget is not None:
            budget = _positive_budget(parameter_budget)
            per_rank = feature_count + coefficient_row_count
            selected_rank = min(max_rank, budget // per_rank)
            if selected_rank < 1:
                raise DepthBasisBudgetError(
                    f"parameter_budget must be at least {per_rank} for rank 1."
                )
            selection = "parameter_budget"
        else:
            selected_rank = max_rank
            selection = "full_rank"

        selected_singular = singular_values[:selected_rank]
        selected_left = left_vectors[:, :selected_rank].contiguous()
        coefficient_rows = selected_left * selected_singular.unsqueeze(0)
        coefficient_rows.mul_(inverse_sqrt_counts.unsqueeze(1))
        captured_energy = float(selected_singular.square().sum().item())
        raw_squared_error = source_energy - captured_energy
        error_tolerance = (
            torch.finfo(torch.float64).eps
            * max(1.0, source_energy)
            * max(len(modules), feature_count)
            * 128.0
        )
        if raw_squared_error < -max(error_tolerance, gram_tolerance):
            raise DepthBasisRankError(
                "Captured Gram energy exceeds source energy beyond float64 "
                "accumulation tolerance."
            )
        squared_error = max(0.0, raw_squared_error)
        if source_energy == 0.0:
            retained_energy = 1.0
            relative_error = 0.0
        else:
            retained_energy = min(1.0, max(0.0, 1.0 - squared_error / source_energy))
            relative_error = math.sqrt(max(0.0, squared_error / source_energy))

        basis_parameters = selected_rank * feature_count
        coefficient_parameters = selected_rank * coefficient_row_count
        unique_parameters = basis_parameters + coefficient_parameters
        dense_parameters = len(modules) * feature_count
        streaming_workspace = (
            effective_chunk_size
            * (len(modules) + coefficient_row_count + 2 * selected_rank)
            + 2 * coefficient_row_count * coefficient_row_count
        )
        eig_workspace = (
            4 * coefficient_row_count * coefficient_row_count
            + 3 * coefficient_row_count
        )
        estimated_peak_elements = max(streaming_workspace, eig_workspace)
        report = DepthBasisReport(
            logical_layers=len(modules),
            coefficient_rows=coefficient_row_count,
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            rank=selected_rank,
            max_rank=max_rank,
            selection=selection,
            requested_energy=requested_energy,
            retained_energy_fraction=retained_energy,
            singular_values=tuple(float(value) for value in singular_values.tolist()),
            parameter_budget=budget,
            budget_headroom=(
                None if budget is None else budget - unique_parameters
            ),
            dense_equivalent_parameters=dense_parameters,
            basis_parameters=basis_parameters,
            coefficient_parameters=coefficient_parameters,
            unique_parameters=unique_parameters,
            squared_reconstruction_error=squared_error,
            relative_frobenius_error=relative_error,
            full_rank_reconstruction=(
                selected_rank == max_rank
                and squared_error <= max(error_tolerance, gram_tolerance)
            ),
            algorithm="two_pass_chunked_layer_gram_eigh",
            feature_count=feature_count,
            requested_feature_chunk_size=requested_chunk_size,
            feature_chunk_size=effective_chunk_size,
            feature_chunks=feature_chunks,
            gram_rows=coefficient_row_count,
            gram_elements=coefficient_row_count * coefficient_row_count,
            estimated_peak_float64_elements=estimated_peak_elements,
            estimated_peak_float64_bytes=estimated_peak_elements * 8,
            fitting_passes=2,
            materialized_operator_stack=False,
            materialized_full_right_singular_matrix=False,
            numerical_rank=min(
                max_rank,
                int(torch.count_nonzero(singular_values).item()),
            ),
            degenerate_eigenspaces=degenerate_eigenspaces,
        )

        coefficient_rows = coefficient_rows.to(
            device=device,
            dtype=dtype,
        ).contiguous()
        basis_weight = torch.zeros(
            selected_rank,
            out_features,
            in_features,
            device=device,
            dtype=dtype,
        )
        basis_bias = (
            torch.zeros(
                selected_rank,
                out_features,
                device=device,
                dtype=dtype,
            )
            if has_bias
            else None
        )
        result = DepthBasisLinear(
            basis_weight,
            coefficient_rows,
            basis_bias,
            layer_to_coefficient=mapping,
            report=report,
        )
        del basis_weight, basis_bias, coefficient_rows
        weight_storage = result.basis_weight.view(selected_rank, weight_features)
        pivot_absolute = torch.zeros(selected_rank, dtype=torch.float64)
        pivot_values = torch.zeros(selected_rank, dtype=torch.float64)
        positive = selected_singular > 0.0
        for start in range(0, feature_count, effective_chunk_size):
            end = min(start + effective_chunk_size, feature_count)
            source_chunk = _canonical_feature_chunk(
                modules,
                start=start,
                end=end,
                weight_features=weight_features,
                has_bias=has_bias,
            )
            grouped_chunk = _weighted_group_chunk(
                source_chunk,
                mapping_tensor,
                inverse_sqrt_counts,
            )
            basis_chunk = selected_left.T @ grouped_chunk
            if bool(positive.any()):
                basis_chunk[positive] = basis_chunk[positive] / selected_singular[
                    positive
                ].unsqueeze(1)
            if bool((~positive).any()):
                basis_chunk[~positive] = 0.0
            if not bool(torch.isfinite(basis_chunk).all()):
                raise DepthBasisFiniteError(
                    "Selected basis feature chunk contains NaN or infinity."
                )

            chunk_absolute, local_pivots = torch.max(
                torch.abs(basis_chunk),
                dim=1,
            )
            update = chunk_absolute > pivot_absolute
            if bool(update.any()):
                rows = torch.nonzero(update, as_tuple=False).flatten()
                pivot_absolute[rows] = chunk_absolute[rows]
                pivot_values[rows] = basis_chunk[rows, local_pivots[rows]]

            if start < weight_features:
                weight_end = min(end, weight_features)
                weight_storage[:, start:weight_end].copy_(
                    basis_chunk[:, : weight_end - start].to(
                        device=device,
                        dtype=dtype,
                    )
                )
            if end > weight_features:
                if result.basis_bias is None:  # pragma: no cover - validated
                    raise DepthBasisShapeError("A required basis bias is missing.")
                bias_start = max(start, weight_features) - weight_features
                bias_end = end - weight_features
                destination_start = max(start, weight_features) - start
                result.basis_bias[:, bias_start:bias_end].copy_(
                    basis_chunk[:, destination_start:].to(
                        device=device,
                        dtype=dtype,
                    )
                )

        signs = torch.where(
            pivot_values < 0.0,
            -torch.ones_like(pivot_values),
            torch.ones_like(pivot_values),
        ).to(device=device, dtype=dtype)
        result.basis_weight.mul_(signs[:, None, None])
        result.coefficients.mul_(signs[None, :])
        if result.basis_bias is not None:
            result.basis_bias.mul_(signs[:, None])
        if not all(
            bool(torch.isfinite(parameter).all())
            for parameter in result.parameters()
        ):
            raise DepthBasisFiniteError(
                "Casting depth-basis parameters produced NaN or infinity."
            )
        result.train(modules[0].training)
        return result


def materialize_linears(representation: DepthBasisLinear) -> list[nn.Linear]:
    """Expand a depth-basis representation into independent linear modules."""

    if not isinstance(representation, DepthBasisLinear):
        raise DepthBasisShapeError("representation must be a DepthBasisLinear.")
    return representation.materialize()


fit_depth_basis = factorize_linears
build_depth_basis_linear = factorize_linears


__all__ = [
    "DEFAULT_FEATURE_CHUNK_SIZE",
    "DepthBasisBudgetError",
    "DepthBasisFiniteError",
    "DepthBasisLinear",
    "DepthBasisLinearError",
    "DepthBasisRankError",
    "DepthBasisReport",
    "DepthBasisShapeError",
    "build_depth_basis_linear",
    "factorize_linears",
    "fit_depth_basis",
    "materialize_linears",
    "stack_linear_operators",
]
