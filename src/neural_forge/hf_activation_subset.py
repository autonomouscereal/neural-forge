"""Deterministic activation-column subset plans for severe width folding.

Unlike activation clustering, this module does not assign every source circuit
to a representative.  It selects a rank-covering subset of real source
producers, then jointly refits their linear consumer against the complete
source residual delta.  Producer identities remain coupled (gate/up for
SwiGLU and Q/K/V group topology for GQA), while the fitted consumer is an
explicit finite-trace approximation.

Selection is zero-gradient and CPU deterministic.  A deterministic row
CountSketch bounds the feature dimension, regularized statistical leverage
scores reward independent activation directions, and realized contribution
energy keeps the subset focused on behaviorally active circuits.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_SCHEMA = "neural-forge-activation-subset-v1"
_METHOD = "deterministic-countsketch-leverage-energy-column-subset"
_CLASSIFICATION = "approximate"
_DEFAULT_SKETCH_ROWS = 32
_TENSOR_HASH_CHUNK_BYTES = 1 << 20


class ActivationSubsetError(ValueError):
    """Base class for fail-closed activation-subset errors."""


class ActivationSubsetShapeError(ActivationSubsetError):
    """Raised when activation or topology shapes cannot prove the contract."""


class ActivationSubsetFiniteError(ActivationSubsetError):
    """Raised when inputs or computed plan values are non-finite."""


class ActivationSubsetInvariantError(ActivationSubsetError):
    """Raised when a constructed plan does not satisfy its invariants."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for activation-column subset planning."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActivationSubsetShapeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ActivationSubsetShapeError(
            f"{name} must be a finite non-negative number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ActivationSubsetShapeError(
            f"{name} must be a finite non-negative number."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ActivationSubsetShapeError(
            f"{name} must be a finite non-negative number."
        )
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result > 1.0:
        raise ActivationSubsetShapeError(f"{name} must be in [0, 1].")
    return result


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ActivationSubsetShapeError(f"{name} must be a bool.")
    return value


def _as_float64_cpu(value: Any, name: str, *, ndim: int) -> Any:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor):
        raise ActivationSubsetShapeError(f"{name} must be a torch.Tensor.")
    if value.ndim != ndim:
        raise ActivationSubsetShapeError(
            f"{name} must have rank {ndim}; received shape {tuple(value.shape)}."
        )
    if any(int(size) <= 0 for size in value.shape):
        raise ActivationSubsetShapeError(f"{name} must have no empty dimensions.")
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ActivationSubsetFiniteError(f"{name} contains non-finite values.")
    return tensor


def _tensor_sha256(tensor: Any) -> str:
    torch = _require_torch()
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(struct.pack("<I", value.ndim))
    for size in value.shape:
        digest.update(struct.pack("<Q", int(size)))
    raw = value.view(torch.uint8).reshape(-1).numpy()
    raw_view = memoryview(raw)
    for start in range(0, int(raw.size), _TENSOR_HASH_CHUNK_BYTES):
        digest.update(raw_view[start : start + _TENSOR_HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matrix_tuple(tensor: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in tensor.tolist())


def _tensor_from_matrix_tuple(value: Sequence[Sequence[float]]) -> Any:
    torch = _require_torch()
    return torch.tensor(value, dtype=torch.float64)


@dataclass(frozen=True, slots=True)
class ReconstructionDiagnostics:
    """Joint residual reconstruction diagnostics for one immutable split."""

    observation_rows: int
    residual_width: int
    target_l2: float
    residual_l2: float
    squared_error: float
    rmse: float
    relative_l2_error: float
    explained_energy: float
    max_absolute_error: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "observation_rows": self.observation_rows,
            "residual_width": self.residual_width,
            "target_l2": self.target_l2,
            "residual_l2": self.residual_l2,
            "squared_error": self.squared_error,
            "rmse": self.rmse,
            "relative_l2_error": self.relative_l2_error,
            "explained_energy": self.explained_energy,
            "max_absolute_error": self.max_absolute_error,
        }


@dataclass(frozen=True, slots=True)
class SubsetSelectionDiagnostics:
    """Rank/energy evidence emitted by deterministic item selection."""

    item_count: int
    selected_count: int
    sketch_rows_per_feature: int
    sketched_feature_width: int
    estimated_feature_rank: int
    leverage_weight: float
    contribution_energy_weight: float
    selected_min_score: float
    selected_max_score: float
    selected_leverage_sum: float
    selected_contribution_energy_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "item_count": self.item_count,
            "selected_count": self.selected_count,
            "sketch_rows_per_feature": self.sketch_rows_per_feature,
            "sketched_feature_width": self.sketched_feature_width,
            "estimated_feature_rank": self.estimated_feature_rank,
            "leverage_weight": self.leverage_weight,
            "contribution_energy_weight": self.contribution_energy_weight,
            "selected_min_score": self.selected_min_score,
            "selected_max_score": self.selected_max_score,
            "selected_leverage_sum": self.selected_leverage_sum,
            "selected_contribution_energy_fraction": (
                self.selected_contribution_energy_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class GatedMLPActivationSubsetReport:
    """Auditable evidence for a coupled real-neuron SwiGLU subset."""

    source_neurons: int
    target_neurons: int
    observation_rows: int
    residual_width: int
    selected_source_neurons: tuple[int, ...]
    selection: SubsetSelectionDiagnostics
    ridge: float
    effective_ridge: float
    calibration: ReconstructionDiagnostics
    validation: ReconstructionDiagnostics | None
    input_sha256: str
    fitted_consumer_sha256: str
    plan_sha256: str

    @property
    def classification(self) -> str:
        return _CLASSIFICATION

    @property
    def zero_gradient(self) -> bool:
        return True

    @property
    def coupled_gate_up_down_identities(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "kind": "gated-mlp",
            "method": _METHOD,
            "classification": self.classification,
            "source_neurons": self.source_neurons,
            "target_neurons": self.target_neurons,
            "observation_rows": self.observation_rows,
            "residual_width": self.residual_width,
            "selected_source_neurons": list(self.selected_source_neurons),
            "selection": self.selection.as_dict(),
            "joint_consumer_fit": True,
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "calibration": self.calibration.as_dict(),
            "validation": (
                None if self.validation is None else self.validation.as_dict()
            ),
            "producer_aggregation": "none-real-source-neurons-only",
            "coupled_gate_up_down_identities": (
                self.coupled_gate_up_down_identities
            ),
            "zero_gradient": self.zero_gradient,
            "input_sha256": self.input_sha256,
            "fitted_consumer_sha256": self.fitted_consumer_sha256,
            "plan_sha256": self.plan_sha256,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class GatedMLPActivationSubsetPlan:
    """Selected real SwiGLU producers plus a jointly fitted down consumer."""

    selected_source_neurons: tuple[int, ...]
    fitted_residual_directions: tuple[tuple[float, ...], ...]
    report: GatedMLPActivationSubsetReport

    def __post_init__(self) -> None:
        if self.selected_source_neurons != self.report.selected_source_neurons:
            raise ActivationSubsetInvariantError(
                "Plan selections do not match the gated-MLP report."
            )
        if len(self.fitted_residual_directions) != self.report.target_neurons:
            raise ActivationSubsetInvariantError(
                "Fitted gated-MLP consumer has the wrong neuron count."
            )
        if any(
            len(row) != self.report.residual_width
            for row in self.fitted_residual_directions
        ):
            raise ActivationSubsetInvariantError(
                "Fitted gated-MLP consumer has the wrong residual width."
            )

    def _select_producer(self, weight: Any, *, axis: int) -> Any:
        return _select_indices(
            weight,
            self.selected_source_neurons,
            axis=axis,
            expected=self.report.source_neurons,
            name="gated-MLP producer",
        )

    def select_gate_producer(self, weight: Any, *, axis: int = 0) -> Any:
        """Select the exact same real neuron rows from the gate producer."""

        return self._select_producer(weight, axis=axis)

    def select_up_producer(self, weight: Any, *, axis: int = 0) -> Any:
        """Select the exact same real neuron rows from the up producer."""

        return self._select_producer(weight, axis=axis)

    def fitted_down_consumer(self, *, dtype: Any | None = None) -> Any:
        """Return the jointly fitted conventional ``[residual, neuron]`` matrix."""

        directions = _tensor_from_matrix_tuple(self.fitted_residual_directions)
        consumer = directions.transpose(0, 1).contiguous()
        return consumer if dtype is None else consumer.to(dtype=dtype)

    def reconstruct(self, post_swiglu_activations: Any) -> Any:
        post = _as_float64_cpu(
            post_swiglu_activations,
            "post_swiglu_activations",
            ndim=2,
        )
        if int(post.shape[1]) != self.report.source_neurons:
            raise ActivationSubsetShapeError(
                "post_swiglu_activations has the wrong source-neuron width."
            )
        directions = _tensor_from_matrix_tuple(self.fitted_residual_directions)
        return post[:, self.selected_source_neurons] @ directions

    def evaluate_reconstruction(
        self,
        post_swiglu_activations: Any,
        *,
        total_residual_delta: Any | None = None,
        residual_contribution_directions: Any | None = None,
    ) -> ReconstructionDiagnostics:
        post = _as_float64_cpu(
            post_swiglu_activations,
            "post_swiglu_activations",
            ndim=2,
        )
        if int(post.shape[1]) != self.report.source_neurons:
            raise ActivationSubsetShapeError(
                "post_swiglu_activations has the wrong source-neuron width."
            )
        target = _resolve_mlp_total_delta(
            post,
            total_residual_delta=total_residual_delta,
            residual_contribution_directions=residual_contribution_directions,
            source_neurons=self.report.source_neurons,
            residual_width=self.report.residual_width,
        )
        return _reconstruction_diagnostics(target, self.reconstruct(post))


@dataclass(frozen=True, slots=True)
class GQAActivationSubsetReport:
    """Auditable evidence for a topology-preserving real-head GQA subset."""

    source_query_heads: int
    source_key_value_heads: int
    target_query_heads: int
    target_key_value_heads: int
    source_query_heads_per_group: int
    target_query_heads_per_group: int
    head_dim: int
    observation_rows: int
    residual_width: int
    selected_source_groups: tuple[int, ...]
    selected_query_offsets: tuple[int, ...]
    selected_source_query_heads: tuple[int, ...]
    group_selection: SubsetSelectionDiagnostics
    query_offset_selection: SubsetSelectionDiagnostics
    ridge: float
    effective_ridge: float
    calibration: ReconstructionDiagnostics
    validation: ReconstructionDiagnostics | None
    input_sha256: str
    fitted_consumer_sha256: str
    plan_sha256: str

    @property
    def classification(self) -> str:
        return _CLASSIFICATION

    @property
    def zero_gradient(self) -> bool:
        return True

    @property
    def complete_real_heads(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "kind": "grouped-query-attention",
            "method": _METHOD,
            "classification": self.classification,
            "source_query_heads": self.source_query_heads,
            "source_key_value_heads": self.source_key_value_heads,
            "target_query_heads": self.target_query_heads,
            "target_key_value_heads": self.target_key_value_heads,
            "source_query_heads_per_group": self.source_query_heads_per_group,
            "target_query_heads_per_group": self.target_query_heads_per_group,
            "head_dim": self.head_dim,
            "observation_rows": self.observation_rows,
            "residual_width": self.residual_width,
            "selected_source_groups": list(self.selected_source_groups),
            "selected_query_offsets": list(self.selected_query_offsets),
            "selected_source_query_heads": list(
                self.selected_source_query_heads
            ),
            "group_selection": self.group_selection.as_dict(),
            "query_offset_selection": self.query_offset_selection.as_dict(),
            "joint_consumer_fit": True,
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "calibration": self.calibration.as_dict(),
            "validation": (
                None if self.validation is None else self.validation.as_dict()
            ),
            "producer_aggregation": "none-real-complete-heads-only",
            "gqa_topology_preserved": True,
            "head_dim_preserved": True,
            "complete_real_heads": self.complete_real_heads,
            "zero_gradient": self.zero_gradient,
            "input_sha256": self.input_sha256,
            "fitted_consumer_sha256": self.fitted_consumer_sha256,
            "plan_sha256": self.plan_sha256,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class GQAActivationSubsetPlan:
    """Selected real GQA groups/heads plus a jointly fitted output consumer."""

    selected_source_groups: tuple[int, ...]
    selected_query_offsets: tuple[int, ...]
    selected_source_query_heads: tuple[int, ...]
    fitted_output_consumer: tuple[tuple[float, ...], ...]
    report: GQAActivationSubsetReport

    def __post_init__(self) -> None:
        if self.selected_source_groups != self.report.selected_source_groups:
            raise ActivationSubsetInvariantError(
                "Plan groups do not match the GQA subset report."
            )
        if self.selected_query_offsets != self.report.selected_query_offsets:
            raise ActivationSubsetInvariantError(
                "Plan offsets do not match the GQA subset report."
            )
        if (
            self.selected_source_query_heads
            != self.report.selected_source_query_heads
        ):
            raise ActivationSubsetInvariantError(
                "Plan query heads do not match the GQA subset report."
            )
        expected_columns = self.report.target_query_heads * self.report.head_dim
        if len(self.fitted_output_consumer) != self.report.residual_width:
            raise ActivationSubsetInvariantError(
                "Fitted GQA output consumer has the wrong residual width."
            )
        if any(
            len(row) != expected_columns for row in self.fitted_output_consumer
        ):
            raise ActivationSubsetInvariantError(
                "Fitted GQA output consumer has the wrong head-coordinate width."
            )

    def select_query_producer(self, weight: Any, *, axis: int = 0) -> Any:
        return _select_blocks(
            weight,
            self.selected_source_query_heads,
            block_width=self.report.head_dim,
            axis=axis,
            expected_blocks=self.report.source_query_heads,
            name="query producer",
        )

    def select_key_producer(self, weight: Any, *, axis: int = 0) -> Any:
        return _select_blocks(
            weight,
            self.selected_source_groups,
            block_width=self.report.head_dim,
            axis=axis,
            expected_blocks=self.report.source_key_value_heads,
            name="key producer",
        )

    def select_value_producer(self, weight: Any, *, axis: int = 0) -> Any:
        return _select_blocks(
            weight,
            self.selected_source_groups,
            block_width=self.report.head_dim,
            axis=axis,
            expected_blocks=self.report.source_key_value_heads,
            name="value producer",
        )

    def output_consumer(self, *, dtype: Any | None = None) -> Any:
        consumer = _tensor_from_matrix_tuple(self.fitted_output_consumer)
        return consumer if dtype is None else consumer.to(dtype=dtype)

    def reconstruct(self, context_trajectories: Any) -> Any:
        context = _as_float64_cpu(
            context_trajectories,
            "context_trajectories",
            ndim=3,
        )
        _validate_context_shape(context, self.report)
        selected = context[:, self.selected_source_query_heads, :].reshape(
            int(context.shape[0]),
            self.report.target_query_heads * self.report.head_dim,
        )
        return selected @ self.output_consumer().transpose(0, 1)

    def evaluate_reconstruction(
        self,
        context_trajectories: Any,
        *,
        total_residual_delta: Any | None = None,
        contribution_trajectories: Any | None = None,
    ) -> ReconstructionDiagnostics:
        context = _as_float64_cpu(
            context_trajectories,
            "context_trajectories",
            ndim=3,
        )
        _validate_context_shape(context, self.report)
        target = _resolve_attention_total_delta(
            context,
            total_residual_delta=total_residual_delta,
            contribution_trajectories=contribution_trajectories,
            source_query_heads=self.report.source_query_heads,
            residual_width=self.report.residual_width,
        )
        return _reconstruction_diagnostics(target, self.reconstruct(context))


@dataclass(frozen=True, slots=True)
class _SelectionResult:
    selected: tuple[int, ...]
    diagnostics: SubsetSelectionDiagnostics


def _countsketch_rows(matrix: Any, rows: int, *, seed: int) -> Any:
    torch = _require_torch()
    source_rows, columns = (int(matrix.shape[0]), int(matrix.shape[1]))
    sketch_rows = min(rows, source_rows)
    indices = torch.arange(source_rows, dtype=torch.int64)
    buckets = (
        indices * 1_103_515_245 + int(seed) * 2_246_822_519 + 12_345
    ).remainder(sketch_rows)
    sign_bits = (
        indices * 2_654_435_761 + int(seed) * 3_266_489_917 + 97
    ).bitwise_and(1)
    signs = torch.where(
        sign_bits == 0,
        torch.ones(source_rows, dtype=torch.float64),
        -torch.ones(source_rows, dtype=torch.float64),
    )
    result = torch.zeros((sketch_rows, columns), dtype=torch.float64)
    result.index_add_(0, buckets, matrix * signs[:, None])
    return result


def _normalized_feature_part(part: Any) -> Any:
    torch = _require_torch()
    rms = torch.sqrt(torch.mean(part.square()))
    scale = float(rms.item())
    if not math.isfinite(scale):
        raise ActivationSubsetFiniteError("Feature scaling produced non-finite RMS.")
    return part if scale <= torch.finfo(torch.float64).tiny else part / scale


def _select_rank_energy_items(
    feature_parts: Sequence[Any],
    contribution_energy: Any,
    *,
    target_items: int,
    sketch_rows: int,
    leverage_weight: float,
    seed: int,
) -> _SelectionResult:
    torch = _require_torch()
    if not feature_parts:
        raise ActivationSubsetShapeError("At least one feature trajectory is required.")
    item_count = int(feature_parts[0].shape[1])
    if any(part.ndim != 2 or int(part.shape[1]) != item_count for part in feature_parts):
        raise ActivationSubsetShapeError(
            "All selection features must be rank-2 with the same item count."
        )
    if contribution_energy.ndim != 1 or int(contribution_energy.shape[0]) != item_count:
        raise ActivationSubsetShapeError(
            "Contribution energy must contain one value per selectable item."
        )
    if target_items > item_count:
        raise ActivationSubsetShapeError(
            "The requested subset cannot exceed the source item count."
        )

    sketched = []
    for part_index, part in enumerate(feature_parts):
        normalized = _normalized_feature_part(part)
        sketched.append(
            _countsketch_rows(
                normalized,
                sketch_rows,
                seed=seed + part_index * 104_729,
            )
        )
    features = torch.cat(sketched, dim=0)
    gram = features @ features.transpose(0, 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    maximum_eigenvalue = float(eigenvalues[-1].item()) if eigenvalues.numel() else 0.0
    rank_tolerance = (
        maximum_eigenvalue
        * max(int(features.shape[0]), item_count)
        * torch.finfo(torch.float64).eps
    )
    estimated_rank = int((eigenvalues > rank_tolerance).sum().item())
    regularizer = max(
        maximum_eigenvalue * 1e-10,
        torch.finfo(torch.float64).tiny,
    )
    inverse_features = torch.linalg.solve(
        gram + regularizer * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        features,
    )
    leverage = (features * inverse_features).sum(dim=0).clamp_min(0.0)
    energy = contribution_energy.clamp_min(0.0)
    if not bool(torch.isfinite(leverage).all().item()):
        raise ActivationSubsetFiniteError("Leverage computation is non-finite.")
    if not bool(torch.isfinite(energy).all().item()):
        raise ActivationSubsetFiniteError("Contribution energy is non-finite.")

    tiny = torch.finfo(torch.float64).eps
    leverage_scale = max(float(leverage.max().item()), tiny)
    energy_scale = max(float(energy.max().item()), tiny)
    normalized_leverage = leverage / leverage_scale
    normalized_energy = energy / energy_scale
    energy_weight = 1.0 - leverage_weight
    scores = (normalized_leverage + tiny).pow(leverage_weight) * (
        normalized_energy + tiny
    ).pow(energy_weight)
    order = sorted(
        range(item_count),
        key=lambda index: (-float(scores[index].item()), index),
    )
    selected = tuple(sorted(order[:target_items]))
    selected_tensor = torch.tensor(selected, dtype=torch.int64)
    selected_scores = scores[selected_tensor]
    total_energy = float(energy.sum().item())
    selected_energy = float(energy[selected_tensor].sum().item())
    diagnostics = SubsetSelectionDiagnostics(
        item_count=item_count,
        selected_count=target_items,
        sketch_rows_per_feature=sketch_rows,
        sketched_feature_width=int(features.shape[0]),
        estimated_feature_rank=estimated_rank,
        leverage_weight=leverage_weight,
        contribution_energy_weight=energy_weight,
        selected_min_score=float(selected_scores.min().item()),
        selected_max_score=float(selected_scores.max().item()),
        selected_leverage_sum=float(leverage[selected_tensor].sum().item()),
        selected_contribution_energy_fraction=(
            selected_energy / total_energy if total_energy > 0.0 else 0.0
        ),
    )
    return _SelectionResult(selected=selected, diagnostics=diagnostics)


def _ridge_fit(design: Any, target: Any, *, ridge: float) -> tuple[Any, float]:
    torch = _require_torch()
    gram = design.transpose(0, 1) @ design
    diagonal_mean = float(torch.diagonal(gram).mean().item())
    effective_ridge = ridge * max(
        diagonal_mean,
        torch.finfo(torch.float64).eps,
    )
    regularized = gram + effective_ridge * torch.eye(
        int(gram.shape[0]), dtype=torch.float64
    )
    coefficients = torch.linalg.solve(
        regularized,
        design.transpose(0, 1) @ target,
    )
    if not bool(torch.isfinite(coefficients).all().item()):
        raise ActivationSubsetFiniteError(
            "Joint ridge consumer fit produced non-finite coefficients."
        )
    return coefficients, effective_ridge


def _reconstruction_diagnostics(
    target: Any,
    prediction: Any,
) -> ReconstructionDiagnostics:
    torch = _require_torch()
    if target.shape != prediction.shape:
        raise ActivationSubsetShapeError(
            "Residual target and prediction must have identical shapes."
        )
    residual = target - prediction
    squared_error = float(residual.square().sum().item())
    target_energy = float(target.square().sum().item())
    target_l2 = math.sqrt(target_energy)
    residual_l2 = math.sqrt(squared_error)
    denominator = max(target_l2, torch.finfo(torch.float64).eps)
    relative_error = residual_l2 / denominator
    explained = 1.0 - (squared_error / max(target_energy, torch.finfo(torch.float64).eps))
    result = ReconstructionDiagnostics(
        observation_rows=int(target.shape[0]),
        residual_width=int(target.shape[1]),
        target_l2=target_l2,
        residual_l2=residual_l2,
        squared_error=squared_error,
        rmse=math.sqrt(squared_error / int(target.numel())),
        relative_l2_error=relative_error,
        explained_energy=explained,
        max_absolute_error=float(residual.abs().max().item()),
    )
    if not all(
        math.isfinite(value)
        for value in (
            result.target_l2,
            result.residual_l2,
            result.squared_error,
            result.rmse,
            result.relative_l2_error,
            result.explained_energy,
            result.max_absolute_error,
        )
    ):
        raise ActivationSubsetFiniteError(
            "Residual reconstruction diagnostics are non-finite."
        )
    return result


def _resolve_mlp_total_delta(
    post: Any,
    *,
    total_residual_delta: Any | None,
    residual_contribution_directions: Any | None,
    source_neurons: int,
    residual_width: int,
) -> Any:
    if (total_residual_delta is None) == (
        residual_contribution_directions is None
    ):
        raise ActivationSubsetShapeError(
            "Provide exactly one of total_residual_delta or "
            "residual_contribution_directions."
        )
    if residual_contribution_directions is not None:
        directions = _as_float64_cpu(
            residual_contribution_directions,
            "residual_contribution_directions",
            ndim=2,
        )
        if tuple(directions.shape) != (source_neurons, residual_width):
            raise ActivationSubsetShapeError(
                "residual_contribution_directions has the wrong shape."
            )
        return post @ directions
    target = _as_float64_cpu(
        total_residual_delta,
        "total_residual_delta",
        ndim=2,
    )
    if tuple(target.shape) != (int(post.shape[0]), residual_width):
        raise ActivationSubsetShapeError("total_residual_delta has the wrong shape.")
    return target


def _resolve_attention_total_delta(
    context: Any,
    *,
    total_residual_delta: Any | None,
    contribution_trajectories: Any | None,
    source_query_heads: int,
    residual_width: int,
) -> Any:
    if (total_residual_delta is None) == (contribution_trajectories is None):
        raise ActivationSubsetShapeError(
            "Provide exactly one of total_residual_delta or "
            "contribution_trajectories."
        )
    if contribution_trajectories is not None:
        contribution = _as_float64_cpu(
            contribution_trajectories,
            "contribution_trajectories",
            ndim=3,
        )
        expected = (int(context.shape[0]), source_query_heads, residual_width)
        if tuple(contribution.shape) != expected:
            raise ActivationSubsetShapeError(
                "contribution_trajectories has the wrong shape."
            )
        return contribution.sum(dim=1)
    target = _as_float64_cpu(
        total_residual_delta,
        "total_residual_delta",
        ndim=2,
    )
    if tuple(target.shape) != (int(context.shape[0]), residual_width):
        raise ActivationSubsetShapeError("total_residual_delta has the wrong shape.")
    return target


def _select_indices(
    tensor: Any,
    indices: tuple[int, ...],
    *,
    axis: int,
    expected: int,
    name: str,
) -> Any:
    torch = _require_torch()
    if not isinstance(tensor, torch.Tensor):
        raise ActivationSubsetShapeError(f"{name} must be a torch.Tensor.")
    normalized_axis = axis if axis >= 0 else tensor.ndim + axis
    if normalized_axis < 0 or normalized_axis >= tensor.ndim:
        raise ActivationSubsetShapeError(f"{name} axis is out of range.")
    if int(tensor.shape[normalized_axis]) != expected:
        raise ActivationSubsetShapeError(f"{name} has the wrong source width.")
    index = torch.tensor(indices, dtype=torch.int64, device=tensor.device)
    return torch.index_select(tensor, normalized_axis, index)


def _select_blocks(
    tensor: Any,
    blocks: tuple[int, ...],
    *,
    block_width: int,
    axis: int,
    expected_blocks: int,
    name: str,
) -> Any:
    expanded = tuple(
        block * block_width + offset
        for block in blocks
        for offset in range(block_width)
    )
    return _select_indices(
        tensor,
        expanded,
        axis=axis,
        expected=expected_blocks * block_width,
        name=name,
    )


def _item_feature_matrix(tensor: Any, *, item_axis: int) -> Any:
    return tensor.movedim(item_axis, 0).reshape(int(tensor.shape[item_axis]), -1).T


def _validate_observation_rows(
    observation_rows: int,
    target_items: int,
    *,
    allow_insufficient_rows: bool,
) -> None:
    if observation_rows < target_items and not allow_insufficient_rows:
        raise ActivationSubsetShapeError(
            f"observation_rows ({observation_rows}) must be at least the target "
            f"subset width ({target_items}); pass allow_insufficient_rows=True "
            "only to explicitly accept an underdetermined fit."
        )


def build_gated_mlp_activation_subset_plan(
    gate_activations: Any,
    up_activations: Any,
    post_swiglu_activations: Any,
    residual_contribution_directions: Any,
    *,
    target_neurons: int,
    source_neurons: int | None = None,
    ridge: float = 1e-6,
    sketch_rows: int = _DEFAULT_SKETCH_ROWS,
    leverage_weight: float = 0.80,
    countsketch_seed: int = 0,
    allow_insufficient_rows: bool = False,
    validation_post_swiglu_activations: Any | None = None,
    validation_total_residual_delta: Any | None = None,
) -> GatedMLPActivationSubsetPlan:
    """Select real coupled SwiGLU neurons and jointly fit their down consumer.

    ``gate_activations``, ``up_activations``, and
    ``post_swiglu_activations`` are full token-row matrices with shape
    ``[observation_rows, source_neurons]``.  Contribution directions have
    shape ``[source_neurons, residual_width]`` and are the source down-consumer
    columns.  The fit target is therefore the complete residual delta
    ``post_swiglu_activations @ residual_contribution_directions``.
    """

    torch = _require_torch()
    target_neurons = _positive_int(target_neurons, "target_neurons")
    sketch_rows = _positive_int(sketch_rows, "sketch_rows")
    ridge = _nonnegative_float(ridge, "ridge")
    if ridge == 0.0:
        raise ActivationSubsetShapeError("ridge must be greater than zero.")
    leverage_weight = _unit_interval(leverage_weight, "leverage_weight")
    countsketch_seed = int(countsketch_seed)
    allow_insufficient_rows = _require_bool(
        allow_insufficient_rows, "allow_insufficient_rows"
    )

    with torch.no_grad():
        gate = _as_float64_cpu(gate_activations, "gate_activations", ndim=2)
        up = _as_float64_cpu(up_activations, "up_activations", ndim=2)
        post = _as_float64_cpu(
            post_swiglu_activations,
            "post_swiglu_activations",
            ndim=2,
        )
        if gate.shape != up.shape or gate.shape != post.shape:
            raise ActivationSubsetShapeError(
                "Gate, up, and post-SwiGLU activations must have identical shapes."
            )
        observed_source_neurons = int(post.shape[1])
        if source_neurons is not None:
            source_neurons = _positive_int(source_neurons, "source_neurons")
            if source_neurons != observed_source_neurons:
                raise ActivationSubsetShapeError(
                    "source_neurons does not match the activation width."
                )
        source_neurons = observed_source_neurons
        if target_neurons > source_neurons:
            raise ActivationSubsetShapeError(
                "target_neurons cannot exceed source_neurons."
            )
        observation_rows = int(post.shape[0])
        _validate_observation_rows(
            observation_rows,
            target_neurons,
            allow_insufficient_rows=allow_insufficient_rows,
        )
        directions = _as_float64_cpu(
            residual_contribution_directions,
            "residual_contribution_directions",
            ndim=2,
        )
        if int(directions.shape[0]) != source_neurons:
            raise ActivationSubsetShapeError(
                "Contribution directions must have one row per source neuron."
            )
        residual_width = int(directions.shape[1])
        contribution_energy = torch.sqrt(
            post.square().sum(dim=0) * directions.square().sum(dim=1)
        )
        selection = _select_rank_energy_items(
            (gate, up, post),
            contribution_energy,
            target_items=target_neurons,
            sketch_rows=sketch_rows,
            leverage_weight=leverage_weight,
            seed=countsketch_seed,
        )
        target_delta = post @ directions
        design = post[:, selection.selected]
        fitted_directions, effective_ridge = _ridge_fit(
            design,
            target_delta,
            ridge=ridge,
        )
        calibration = _reconstruction_diagnostics(
            target_delta,
            design @ fitted_directions,
        )

        validation = None
        validation_post = None
        validation_target = None
        if validation_post_swiglu_activations is not None:
            validation_post = _as_float64_cpu(
                validation_post_swiglu_activations,
                "validation_post_swiglu_activations",
                ndim=2,
            )
            if int(validation_post.shape[1]) != source_neurons:
                raise ActivationSubsetShapeError(
                    "Validation post-SwiGLU width does not match the source."
                )
            if validation_total_residual_delta is None:
                validation_target = validation_post @ directions
            else:
                validation_target = _as_float64_cpu(
                    validation_total_residual_delta,
                    "validation_total_residual_delta",
                    ndim=2,
                )
                if tuple(validation_target.shape) != (
                    int(validation_post.shape[0]),
                    residual_width,
                ):
                    raise ActivationSubsetShapeError(
                        "validation_total_residual_delta has the wrong shape."
                    )
            validation = _reconstruction_diagnostics(
                validation_target,
                validation_post[:, selection.selected] @ fitted_directions,
            )
        elif validation_total_residual_delta is not None:
            raise ActivationSubsetShapeError(
                "validation_total_residual_delta requires validation activations."
            )

        input_body: dict[str, Any] = {
            "schema": _SCHEMA,
            "kind": "gated-mlp",
            "gate_sha256": _tensor_sha256(gate),
            "up_sha256": _tensor_sha256(up),
            "post_swiglu_sha256": _tensor_sha256(post),
            "contribution_directions_sha256": _tensor_sha256(directions),
            "target_neurons": target_neurons,
            "ridge": ridge,
            "sketch_rows": sketch_rows,
            "leverage_weight": leverage_weight,
            "countsketch_seed": countsketch_seed,
            "allow_insufficient_rows": allow_insufficient_rows,
        }
        if validation_post is not None and validation_target is not None:
            input_body["validation_post_swiglu_sha256"] = _tensor_sha256(
                validation_post
            )
            input_body["validation_target_sha256"] = _tensor_sha256(
                validation_target
            )
        input_sha256 = _canonical_sha256(input_body)
        consumer_sha256 = _tensor_sha256(fitted_directions)
        plan_body = {
            "input_sha256": input_sha256,
            "selected_source_neurons": list(selection.selected),
            "fitted_consumer_sha256": consumer_sha256,
            "calibration": calibration.as_dict(),
            "validation": None if validation is None else validation.as_dict(),
            "classification": _CLASSIFICATION,
        }
        plan_sha256 = _canonical_sha256(plan_body)
        report = GatedMLPActivationSubsetReport(
            source_neurons=source_neurons,
            target_neurons=target_neurons,
            observation_rows=observation_rows,
            residual_width=residual_width,
            selected_source_neurons=selection.selected,
            selection=selection.diagnostics,
            ridge=ridge,
            effective_ridge=effective_ridge,
            calibration=calibration,
            validation=validation,
            input_sha256=input_sha256,
            fitted_consumer_sha256=consumer_sha256,
            plan_sha256=plan_sha256,
        )
        return GatedMLPActivationSubsetPlan(
            selected_source_neurons=selection.selected,
            fitted_residual_directions=_matrix_tuple(fitted_directions),
            report=report,
        )


def _resolve_gqa_head_dim(
    observed: int,
    *,
    head_dim: int | None,
    source_head_dim: int | None,
    target_head_dim: int | None,
) -> int:
    supplied = {
        name: _positive_int(value, name)
        for name, value in (
            ("head_dim", head_dim),
            ("source_head_dim", source_head_dim),
            ("target_head_dim", target_head_dim),
        )
        if value is not None
    }
    if any(value != observed for value in supplied.values()):
        raise ActivationSubsetShapeError(
            "Source and target head_dim must exactly match trajectory head_dim."
        )
    if observed % 2 != 0:
        raise ActivationSubsetShapeError(
            "GQA head_dim must be even so RoPE coordinate pairs stay complete."
        )
    return observed


def _validate_context_shape(context: Any, report: GQAActivationSubsetReport) -> None:
    expected = (report.source_query_heads, report.head_dim)
    if tuple(context.shape[1:]) != expected:
        raise ActivationSubsetShapeError(
            "context_trajectories does not match the source GQA topology."
        )


def build_gqa_activation_subset_plan(
    query_activations: Any,
    key_activations: Any,
    value_activations: Any,
    context_trajectories: Any,
    contribution_trajectories: Any,
    *,
    source_query_heads: int,
    source_key_value_heads: int,
    target_query_heads: int,
    target_key_value_heads: int,
    head_dim: int | None = None,
    source_head_dim: int | None = None,
    target_head_dim: int | None = None,
    ridge: float = 1e-6,
    sketch_rows: int = _DEFAULT_SKETCH_ROWS,
    leverage_weight: float = 0.80,
    countsketch_seed: int = 0,
    allow_insufficient_rows: bool = False,
    validation_context_trajectories: Any | None = None,
    validation_contribution_trajectories: Any | None = None,
    validation_total_residual_delta: Any | None = None,
) -> GQAActivationSubsetPlan:
    """Select complete real GQA heads and jointly fit an output consumer.

    All activation/context tensors use ``[token_rows, heads, head_dim]``.
    Per-query-head residual contributions use
    ``[token_rows, source_query_heads, residual_width]``.  Selected KV groups
    and query offsets form a group-major Cartesian product, preserving GQA
    grouping and the complete source ``head_dim`` coordinate block.
    """

    torch = _require_torch()
    source_query_heads = _positive_int(source_query_heads, "source_query_heads")
    source_key_value_heads = _positive_int(
        source_key_value_heads, "source_key_value_heads"
    )
    target_query_heads = _positive_int(target_query_heads, "target_query_heads")
    target_key_value_heads = _positive_int(
        target_key_value_heads, "target_key_value_heads"
    )
    sketch_rows = _positive_int(sketch_rows, "sketch_rows")
    ridge = _nonnegative_float(ridge, "ridge")
    if ridge == 0.0:
        raise ActivationSubsetShapeError("ridge must be greater than zero.")
    leverage_weight = _unit_interval(leverage_weight, "leverage_weight")
    countsketch_seed = int(countsketch_seed)
    allow_insufficient_rows = _require_bool(
        allow_insufficient_rows, "allow_insufficient_rows"
    )
    if source_query_heads % source_key_value_heads != 0:
        raise ActivationSubsetShapeError(
            "source_query_heads must divide exactly into source GQA groups."
        )
    if target_query_heads % target_key_value_heads != 0:
        raise ActivationSubsetShapeError(
            "target_query_heads must divide exactly into target GQA groups."
        )
    source_offsets = source_query_heads // source_key_value_heads
    target_offsets = target_query_heads // target_key_value_heads
    if target_key_value_heads > source_key_value_heads:
        raise ActivationSubsetShapeError(
            "Target KV-group count cannot exceed the source count."
        )
    if target_offsets > source_offsets:
        raise ActivationSubsetShapeError(
            "Target query-head offsets per group cannot exceed the source."
        )

    with torch.no_grad():
        query = _as_float64_cpu(
            query_activations, "query_activations", ndim=3
        )
        key = _as_float64_cpu(key_activations, "key_activations", ndim=3)
        value = _as_float64_cpu(value_activations, "value_activations", ndim=3)
        context = _as_float64_cpu(
            context_trajectories, "context_trajectories", ndim=3
        )
        contribution = _as_float64_cpu(
            contribution_trajectories,
            "contribution_trajectories",
            ndim=3,
        )
        observation_rows = int(query.shape[0])
        if any(
            int(tensor.shape[0]) != observation_rows
            for tensor in (key, value, context, contribution)
        ):
            raise ActivationSubsetShapeError(
                "All GQA trajectories must use identical token rows."
            )
        observed_head_dim = int(query.shape[2])
        resolved_head_dim = _resolve_gqa_head_dim(
            observed_head_dim,
            head_dim=head_dim,
            source_head_dim=source_head_dim,
            target_head_dim=target_head_dim,
        )
        if tuple(query.shape[1:]) != (source_query_heads, resolved_head_dim):
            raise ActivationSubsetShapeError(
                "query_activations does not match the declared source topology."
            )
        expected_kv_shape = (source_key_value_heads, resolved_head_dim)
        if tuple(key.shape[1:]) != expected_kv_shape:
            raise ActivationSubsetShapeError(
                "key_activations does not match the declared source topology."
            )
        if tuple(value.shape[1:]) != expected_kv_shape:
            raise ActivationSubsetShapeError(
                "value_activations does not match the declared source topology."
            )
        if tuple(context.shape[1:]) != (
            source_query_heads,
            resolved_head_dim,
        ):
            raise ActivationSubsetShapeError(
                "context_trajectories does not match the declared source topology."
            )
        if int(contribution.shape[1]) != source_query_heads:
            raise ActivationSubsetShapeError(
                "Contribution trajectories need one entry per source query head."
            )
        residual_width = int(contribution.shape[2])
        _validate_observation_rows(
            observation_rows,
            target_query_heads,
            allow_insufficient_rows=allow_insufficient_rows,
        )

        query_grouped = query.reshape(
            observation_rows,
            source_key_value_heads,
            source_offsets,
            resolved_head_dim,
        )
        context_grouped = context.reshape(
            observation_rows,
            source_key_value_heads,
            source_offsets,
            resolved_head_dim,
        )
        contribution_grouped = contribution.reshape(
            observation_rows,
            source_key_value_heads,
            source_offsets,
            residual_width,
        )
        group_energy = torch.sqrt(
            contribution_grouped.square().sum(dim=(0, 2, 3))
        )
        group_selection = _select_rank_energy_items(
            (
                _item_feature_matrix(key, item_axis=1),
                _item_feature_matrix(value, item_axis=1),
                _item_feature_matrix(query_grouped, item_axis=1),
                _item_feature_matrix(context_grouped, item_axis=1),
                _item_feature_matrix(contribution_grouped, item_axis=1),
            ),
            group_energy,
            target_items=target_key_value_heads,
            sketch_rows=sketch_rows,
            leverage_weight=leverage_weight,
            seed=countsketch_seed,
        )
        offset_energy = torch.sqrt(
            contribution_grouped.square().sum(dim=(0, 1, 3))
        )
        offset_selection = _select_rank_energy_items(
            (
                _item_feature_matrix(query_grouped, item_axis=2),
                _item_feature_matrix(context_grouped, item_axis=2),
                _item_feature_matrix(contribution_grouped, item_axis=2),
            ),
            offset_energy,
            target_items=target_offsets,
            sketch_rows=sketch_rows,
            leverage_weight=leverage_weight,
            seed=countsketch_seed + 1_000_003,
        )
        selected_heads = tuple(
            group * source_offsets + offset
            for group in group_selection.selected
            for offset in offset_selection.selected
        )
        if len(selected_heads) != target_query_heads:
            raise ActivationSubsetInvariantError(
                "Selected GQA groups and offsets do not produce target_query_heads."
            )
        design = context[:, selected_heads, :].reshape(
            observation_rows,
            target_query_heads * resolved_head_dim,
        )
        target_delta = contribution.sum(dim=1)
        fitted_directions, effective_ridge = _ridge_fit(
            design,
            target_delta,
            ridge=ridge,
        )
        fitted_output = fitted_directions.transpose(0, 1).contiguous()
        calibration = _reconstruction_diagnostics(
            target_delta,
            design @ fitted_directions,
        )

        validation = None
        validation_context = None
        validation_target = None
        if validation_context_trajectories is not None:
            validation_context = _as_float64_cpu(
                validation_context_trajectories,
                "validation_context_trajectories",
                ndim=3,
            )
            if tuple(validation_context.shape[1:]) != (
                source_query_heads,
                resolved_head_dim,
            ):
                raise ActivationSubsetShapeError(
                    "Validation context does not match the source GQA topology."
                )
            if (
                validation_contribution_trajectories is None
            ) == (validation_total_residual_delta is None):
                raise ActivationSubsetShapeError(
                    "Validation context requires exactly one validation target "
                    "source: contribution trajectories or total residual delta."
                )
            validation_target = _resolve_attention_total_delta(
                validation_context,
                total_residual_delta=validation_total_residual_delta,
                contribution_trajectories=validation_contribution_trajectories,
                source_query_heads=source_query_heads,
                residual_width=residual_width,
            )
            validation_design = validation_context[:, selected_heads, :].reshape(
                int(validation_context.shape[0]),
                target_query_heads * resolved_head_dim,
            )
            validation = _reconstruction_diagnostics(
                validation_target,
                validation_design @ fitted_directions,
            )
        elif (
            validation_contribution_trajectories is not None
            or validation_total_residual_delta is not None
        ):
            raise ActivationSubsetShapeError(
                "Validation target trajectories require validation context."
            )

        input_body: dict[str, Any] = {
            "schema": _SCHEMA,
            "kind": "grouped-query-attention",
            "query_sha256": _tensor_sha256(query),
            "key_sha256": _tensor_sha256(key),
            "value_sha256": _tensor_sha256(value),
            "context_sha256": _tensor_sha256(context),
            "contribution_sha256": _tensor_sha256(contribution),
            "source_query_heads": source_query_heads,
            "source_key_value_heads": source_key_value_heads,
            "target_query_heads": target_query_heads,
            "target_key_value_heads": target_key_value_heads,
            "head_dim": resolved_head_dim,
            "ridge": ridge,
            "sketch_rows": sketch_rows,
            "leverage_weight": leverage_weight,
            "countsketch_seed": countsketch_seed,
            "allow_insufficient_rows": allow_insufficient_rows,
        }
        if validation_context is not None and validation_target is not None:
            input_body["validation_context_sha256"] = _tensor_sha256(
                validation_context
            )
            input_body["validation_target_sha256"] = _tensor_sha256(
                validation_target
            )
        input_sha256 = _canonical_sha256(input_body)
        consumer_sha256 = _tensor_sha256(fitted_output)
        plan_body = {
            "input_sha256": input_sha256,
            "selected_source_groups": list(group_selection.selected),
            "selected_query_offsets": list(offset_selection.selected),
            "selected_source_query_heads": list(selected_heads),
            "fitted_consumer_sha256": consumer_sha256,
            "calibration": calibration.as_dict(),
            "validation": None if validation is None else validation.as_dict(),
            "classification": _CLASSIFICATION,
        }
        plan_sha256 = _canonical_sha256(plan_body)
        report = GQAActivationSubsetReport(
            source_query_heads=source_query_heads,
            source_key_value_heads=source_key_value_heads,
            target_query_heads=target_query_heads,
            target_key_value_heads=target_key_value_heads,
            source_query_heads_per_group=source_offsets,
            target_query_heads_per_group=target_offsets,
            head_dim=resolved_head_dim,
            observation_rows=observation_rows,
            residual_width=residual_width,
            selected_source_groups=group_selection.selected,
            selected_query_offsets=offset_selection.selected,
            selected_source_query_heads=selected_heads,
            group_selection=group_selection.diagnostics,
            query_offset_selection=offset_selection.diagnostics,
            ridge=ridge,
            effective_ridge=effective_ridge,
            calibration=calibration,
            validation=validation,
            input_sha256=input_sha256,
            fitted_consumer_sha256=consumer_sha256,
            plan_sha256=plan_sha256,
        )
        return GQAActivationSubsetPlan(
            selected_source_groups=group_selection.selected,
            selected_query_offsets=offset_selection.selected,
            selected_source_query_heads=selected_heads,
            fitted_output_consumer=_matrix_tuple(fitted_output),
            report=report,
        )


__all__ = [
    "ActivationSubsetError",
    "ActivationSubsetFiniteError",
    "ActivationSubsetInvariantError",
    "ActivationSubsetShapeError",
    "GatedMLPActivationSubsetPlan",
    "GatedMLPActivationSubsetReport",
    "GQAActivationSubsetPlan",
    "GQAActivationSubsetReport",
    "ReconstructionDiagnostics",
    "SubsetSelectionDiagnostics",
    "build_gated_mlp_activation_subset_plan",
    "build_gqa_activation_subset_plan",
]
