"""Sparse multi-anchor activation folding for Qwen-style SwiGLU neurons.

Each removed intermediate neuron is reconstructed from several surviving real
neurons.  The fitted coefficients are absorbed into the surviving down
projection columns, so the installed model has no adapter or auxiliary path.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from neural_forge.hf_qwen_compensated_neuron_fold import (
    Qwen3CompensatedFoldShapeError,
    Qwen3CompensatedFoldStateError,
    _exact_tensor_fingerprint,
    _finite_matrix,
    _realized_layer_relative_l2,
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3CompensatedFoldShapeError(f"{name} must be a positive integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3CompensatedFoldShapeError(
            f"{name} must be a positive integer."
        ) from exc
    if result <= 0:
        raise Qwen3CompensatedFoldShapeError(f"{name} must be positive.")
    return result


def _provenance(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Qwen3CompensatedFoldShapeError(f"{name} must be a non-empty string.")
    return value.strip()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MultiAnchorNeuronMerge:
    dropped: int
    anchors: tuple[int, ...]
    coefficients: tuple[float, ...]
    calibration_path_relative_l2: float
    validation_path_relative_l2: float
    calibration_layer_marginal_relative_l2: float
    validation_layer_marginal_relative_l2: float
    score: float

    def __post_init__(self) -> None:
        if self.dropped < 0 or not self.anchors:
            raise Qwen3CompensatedFoldShapeError("A merge needs a dropped path and anchors.")
        if len(self.anchors) != len(self.coefficients):
            raise Qwen3CompensatedFoldShapeError(
                "Anchor and coefficient counts must match."
            )
        if len(set(self.anchors)) != len(self.anchors) or self.dropped in self.anchors:
            raise Qwen3CompensatedFoldShapeError(
                "A merge must contain distinct source paths."
            )
        values = (*self.coefficients, self.calibration_path_relative_l2,
                  self.validation_path_relative_l2,
                  self.calibration_layer_marginal_relative_l2,
                  self.validation_layer_marginal_relative_l2, self.score)
        if any(not math.isfinite(float(value)) for value in values):
            raise Qwen3CompensatedFoldShapeError("Merge values must be finite.")
        if min(values[-5:]) < 0.0:
            raise Qwen3CompensatedFoldShapeError("Merge errors cannot be negative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "dropped": self.dropped,
            "anchors": list(self.anchors),
            "coefficients": list(self.coefficients),
            "calibration_path_relative_l2": self.calibration_path_relative_l2,
            "validation_path_relative_l2": self.validation_path_relative_l2,
            "calibration_layer_marginal_relative_l2": (
                self.calibration_layer_marginal_relative_l2
            ),
            "validation_layer_marginal_relative_l2": (
                self.validation_layer_marginal_relative_l2
            ),
            "score": self.score,
        }


@dataclass(frozen=True)
class MultiAnchorFoldDiagnostics:
    source_width: int
    target_width: int
    calibration_rows: int
    validation_rows: int
    calibration_provenance: str
    validation_provenance: str
    calibration_activation_sha256: str
    validation_activation_sha256: str
    candidate_pool_size: int
    survivor_pool_size: int
    max_anchors: int
    similarity_products: int
    peak_similarity_elements: int
    similarity_chunk_size: int
    projection_fit_count: int
    max_projection_fits: int
    selected_merge_count: int
    protected_count: int
    aggregate_calibration_relative_l2: float
    aggregate_validation_relative_l2: float
    realized_dtype: str
    materialized_full_pairwise_matrix: bool = False
    optimizer_steps: int = 0
    backward_called: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class MultiAnchorNeuronFoldWeights:
    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    kept_source_indices: tuple[int, ...]
    merges: tuple[MultiAnchorNeuronMerge, ...]
    protected_indices: tuple[int, ...]
    source_weight_fingerprints: tuple[str, str, str]
    result_weight_fingerprints: tuple[str, str, str]
    diagnostics: MultiAnchorFoldDiagnostics

    @property
    def source_width(self) -> int:
        return self.diagnostics.source_width

    @property
    def target_width(self) -> int:
        return self.diagnostics.target_width

    @property
    def parameter_savings(self) -> int:
        hidden = int(self.gate_weight.shape[1])
        return (self.source_width - self.target_width) * hidden * 3

    def assert_integrity(self) -> None:
        fingerprints = tuple(
            _exact_tensor_fingerprint(value)
            for value in (self.gate_weight, self.up_weight, self.down_weight)
        )
        if fingerprints != self.result_weight_fingerprints:
            raise Qwen3CompensatedFoldStateError(
                "Multi-anchor fold tensors changed after fitting."
            )

    @property
    def fold_sha256(self) -> str:
        self.assert_integrity()
        return _canonical_sha256(
            {
                "kept_source_indices": list(self.kept_source_indices),
                "merges": [merge.as_dict() for merge in self.merges],
                "protected_indices": list(self.protected_indices),
                "source_weight_fingerprints": list(self.source_weight_fingerprints),
                "result_weight_fingerprints": list(self.result_weight_fingerprints),
                "diagnostics": self.diagnostics.as_dict(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "target_width": self.target_width,
            "parameter_savings": self.parameter_savings,
            "kept_source_indices": list(self.kept_source_indices),
            "merges": [merge.as_dict() for merge in self.merges],
            "protected_indices": list(self.protected_indices),
            "source_weight_fingerprints": list(self.source_weight_fingerprints),
            "result_weight_fingerprints": list(self.result_weight_fingerprints),
            "diagnostics": self.diagnostics.as_dict(),
            "fold_sha256": self.fold_sha256,
        }


def _validate_weights(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    calibration_activations: Tensor,
    validation_activations: Tensor,
    target_width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    gate = _finite_matrix(gate_weight, "gate_weight")
    up = _finite_matrix(up_weight, "up_weight")
    down = _finite_matrix(down_weight, "down_weight")
    calibration = _finite_matrix(calibration_activations, "calibration_activations")
    validation = _finite_matrix(validation_activations, "validation_activations")
    if gate.shape != up.shape:
        raise Qwen3CompensatedFoldShapeError("Gate and up weights must match.")
    width, hidden = map(int, gate.shape)
    if tuple(down.shape) != (hidden, width):
        raise Qwen3CompensatedFoldShapeError(
            "Down weight must have shape [hidden, intermediate]."
        )
    if int(calibration.shape[1]) != width or int(validation.shape[1]) != width:
        raise Qwen3CompensatedFoldShapeError(
            "Activation width must equal the MLP intermediate width."
        )
    target = _positive_int(target_width, "target_width")
    if target >= width:
        raise Qwen3CompensatedFoldShapeError(
            "target_width must be smaller than the source width."
        )
    return gate, up, down, calibration, validation, target


def _path_energy(activations: Tensor, down: Tensor) -> Tensor:
    activation_norm = torch.linalg.vector_norm(activations, dim=0)
    consumer_norm = torch.linalg.vector_norm(down, dim=0)
    return activation_norm * consumer_norm


def _candidate_neighbors(
    calibration: Tensor,
    candidate_indices: Tensor,
    survivor_indices: Tensor,
    *,
    shortlist_size: int,
    chunk_size: int,
    max_similarity_products: int,
) -> tuple[dict[int, tuple[int, ...]], int, int]:
    products = int(candidate_indices.numel()) * int(survivor_indices.numel())
    if products > max_similarity_products:
        raise Qwen3CompensatedFoldShapeError(
            "Candidate/survivor similarity work exceeds max_similarity_products."
        )
    work = calibration.to(device="cpu", dtype=torch.float32)
    norms = torch.linalg.vector_norm(work, dim=0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized = work / norms
    survivors = normalized.index_select(1, survivor_indices)
    result: dict[int, tuple[int, ...]] = {}
    peak = 0
    for start in range(0, int(candidate_indices.numel()), chunk_size):
        piece = candidate_indices[start : start + chunk_size]
        similarity = normalized.index_select(1, piece).T @ survivors
        peak = max(peak, int(similarity.numel()))
        count = min(shortlist_size, int(survivor_indices.numel()))
        local = torch.topk(similarity.abs(), count, dim=1, sorted=True).indices
        for row, dropped in enumerate(piece.tolist()):
            result[int(dropped)] = tuple(
                int(survivor_indices[index]) for index in local[row].tolist()
            )
    return result, products, peak


def _fit_one_merge(
    dropped: int,
    anchor_shortlist: tuple[int, ...],
    calibration: Tensor,
    validation: Tensor,
    down: Tensor,
    *,
    max_anchors: int,
    ridge: float,
    max_coefficient_l1: float,
    calibration_layer_norm: Tensor,
    validation_layer_norm: Tensor,
) -> tuple[MultiAnchorNeuronMerge, int]:
    tiny = torch.finfo(torch.float64).tiny
    cal_target = calibration[:, dropped]
    val_target = validation[:, dropped]
    consumer_norm = torch.linalg.vector_norm(down[:, dropped]).to(torch.float64)
    proposals: list[MultiAnchorNeuronMerge] = []
    selected_anchors: list[int] = []
    remaining = list(anchor_shortlist)
    max_selected = min(len(remaining), max_anchors)
    fit_count = 0
    for _ in range(max_selected):
        best_step: tuple[
            tuple[float, int],
            int,
            tuple[int, ...],
            Tensor,
            Tensor,
            Tensor,
        ] | None = None
        for candidate in remaining:
            fit_count += 1
            anchors = tuple((*selected_anchors, candidate))
            count = len(anchors)
            index = torch.tensor(anchors, dtype=torch.long)
            design = calibration.index_select(1, index)
            gram = design.T @ design
            scale = (torch.trace(gram) / count).clamp_min(tiny)
            regularized = gram + torch.eye(count, dtype=torch.float64) * (
                ridge * scale
            )
            coefficients = torch.linalg.solve(regularized, design.T @ cal_target)
            if float(coefficients.abs().sum().item()) > max_coefficient_l1:
                continue
            residual = cal_target - design @ coefficients
            item = (
                (float(torch.linalg.vector_norm(residual).item()), candidate),
                candidate,
                anchors,
                design,
                coefficients,
                residual,
            )
            if best_step is None or item[0] < best_step[0]:
                best_step = item
        if best_step is None:
            break
        _, selected, anchors, design, coefficients, residual = best_step
        selected_anchors.append(selected)
        remaining.remove(selected)
        if float(coefficients.abs().sum().item()) > max_coefficient_l1:
            continue
        cal_residual = residual
        val_design = validation.index_select(1, index)
        val_residual = val_target - val_design @ coefficients
        cal_path = torch.linalg.vector_norm(cal_residual) / torch.linalg.vector_norm(
            cal_target
        ).clamp_min(tiny)
        val_path = torch.linalg.vector_norm(val_residual) / torch.linalg.vector_norm(
            val_target
        ).clamp_min(tiny)
        cal_marginal = (
            torch.linalg.vector_norm(cal_residual) * consumer_norm
            / calibration_layer_norm
        )
        val_marginal = (
            torch.linalg.vector_norm(val_residual) * consumer_norm
            / validation_layer_norm
        )
        score = torch.maximum(cal_marginal, val_marginal)
        proposals.append(
            MultiAnchorNeuronMerge(
                dropped=dropped,
                anchors=anchors,
                coefficients=tuple(float(value) for value in coefficients.tolist()),
                calibration_path_relative_l2=float(cal_path.item()),
                validation_path_relative_l2=float(val_path.item()),
                calibration_layer_marginal_relative_l2=float(cal_marginal.item()),
                validation_layer_marginal_relative_l2=float(val_marginal.item()),
                score=float(score.item()),
            )
        )
    if not proposals:
        raise Qwen3CompensatedFoldShapeError(
            f"No numerically bounded anchor fit exists for neuron {dropped}."
        )
    return (
        min(
            proposals,
            key=lambda item: (
                item.score,
                item.validation_path_relative_l2,
                len(item.anchors),
                item.anchors,
            ),
        ),
        fit_count,
    )


def fit_multi_anchor_neuron_fold(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    calibration_activations: Tensor,
    *,
    validation_activations: Tensor,
    target_width: int,
    calibration_provenance: str,
    validation_provenance: str,
    protected_indices: tuple[int, ...] | list[int] = (),
    candidate_pool_size: int = 512,
    max_anchors: int = 4,
    similarity_chunk_size: int = 64,
    max_similarity_products: int = 16_000_000,
    max_projection_fits: int = 131_072,
    ridge: float = 1e-6,
    max_coefficient_l1: float = 16.0,
) -> MultiAnchorNeuronFoldWeights:
    """Fold low-contribution paths into nonlocal survivors without training."""

    gate, up, down, calibration, validation, target = _validate_weights(
        gate_weight,
        up_weight,
        down_weight,
        calibration_activations,
        validation_activations,
        target_width,
    )
    cal_source = _provenance(calibration_provenance, "calibration_provenance")
    val_source = _provenance(validation_provenance, "validation_provenance")
    if cal_source == val_source:
        raise Qwen3CompensatedFoldShapeError(
            "Calibration and validation provenance must be disjoint."
        )
    width = int(gate.shape[0])
    required = width - target
    pool_limit = _positive_int(candidate_pool_size, "candidate_pool_size")
    anchor_limit = _positive_int(max_anchors, "max_anchors")
    chunk = _positive_int(similarity_chunk_size, "similarity_chunk_size")
    product_limit = _positive_int(
        max_similarity_products, "max_similarity_products"
    )
    projection_fit_limit = _positive_int(max_projection_fits, "max_projection_fits")
    if anchor_limit > 8:
        raise Qwen3CompensatedFoldShapeError("max_anchors exceeds hard cap 8.")
    if chunk > 1_024:
        raise Qwen3CompensatedFoldShapeError(
            "similarity_chunk_size exceeds hard cap 1024."
        )
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise Qwen3CompensatedFoldShapeError("ridge must be finite and positive.")
    if not math.isfinite(max_coefficient_l1) or max_coefficient_l1 <= 0.0:
        raise Qwen3CompensatedFoldShapeError(
            "max_coefficient_l1 must be finite and positive."
        )
    protected = tuple(sorted({int(index) for index in protected_indices}))
    if any(index < 0 or index >= width for index in protected):
        raise Qwen3CompensatedFoldShapeError("A protected index is out of range.")
    available = width - len(protected)
    if required >= available:
        raise Qwen3CompensatedFoldShapeError(
            "At least one unprotected survivor is required."
        )
    pool_count = min(pool_limit, available - 1)
    if pool_count < required:
        raise Qwen3CompensatedFoldShapeError(
            "candidate_pool_size is smaller than the required merge count."
        )

    cal = calibration.to(device="cpu", dtype=torch.float64)
    val = validation.to(device="cpu", dtype=torch.float64)
    consumer = down.detach().to(device="cpu", dtype=torch.float64)
    protected_set = frozenset(protected)
    eligible = torch.tensor(
        [index for index in range(width) if index not in protected_set],
        dtype=torch.long,
    )
    energy = _path_energy(cal, consumer).index_select(0, eligible)
    pool_local = torch.topk(energy, pool_count, largest=False, sorted=True).indices
    candidate_indices = eligible.index_select(0, pool_local)
    candidate_set = frozenset(int(index) for index in candidate_indices.tolist())
    survivor_indices = torch.tensor(
        [
            index
            for index in range(width)
            if index not in protected_set and index not in candidate_set
        ],
        dtype=torch.long,
    )
    shortlist_size = min(32, max(anchor_limit * 4, anchor_limit))
    actual_shortlist = min(shortlist_size, int(survivor_indices.numel()))
    fit_bound_per_candidate = sum(
        max(0, actual_shortlist - depth)
        for depth in range(min(anchor_limit, actual_shortlist))
    )
    if pool_count * fit_bound_per_candidate > projection_fit_limit:
        raise Qwen3CompensatedFoldShapeError(
            "Multi-anchor projection work exceeds max_projection_fits."
        )
    neighbors, similarity_products, peak_similarity_elements = _candidate_neighbors(
        cal,
        candidate_indices,
        survivor_indices,
        shortlist_size=shortlist_size,
        chunk_size=chunk,
        max_similarity_products=product_limit,
    )
    calibration_layer_norm = torch.linalg.vector_norm(cal @ consumer.T).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    validation_layer_norm = torch.linalg.vector_norm(val @ consumer.T).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    proposals: list[MultiAnchorNeuronMerge] = []
    projection_fit_count = 0
    for dropped in candidate_indices.tolist():
        try:
            proposal, fit_count = _fit_one_merge(
                    int(dropped),
                    neighbors[int(dropped)],
                    cal,
                    val,
                    consumer,
                    max_anchors=anchor_limit,
                    ridge=float(ridge),
                    max_coefficient_l1=float(max_coefficient_l1),
                    calibration_layer_norm=calibration_layer_norm,
                    validation_layer_norm=validation_layer_norm,
                )
            proposals.append(proposal)
            projection_fit_count += fit_count
        except Qwen3CompensatedFoldShapeError:
            continue
    proposals.sort(
        key=lambda item: (
            item.score,
            item.validation_path_relative_l2,
            item.dropped,
            item.anchors,
        )
    )
    selected = tuple(proposals[:required])
    if len(selected) != required:
        raise Qwen3CompensatedFoldShapeError(
            f"Only {len(selected)} bounded multi-anchor merges were found; "
            f"{required} are required."
        )

    down_work = down.detach().clone()
    dropped_indices = {merge.dropped for merge in selected}
    for merge in selected:
        source_column = down[:, merge.dropped]
        for anchor, coefficient in zip(
            merge.anchors, merge.coefficients, strict=True
        ):
            down_work[:, anchor] = (
                down_work[:, anchor] + float(coefficient) * source_column
            )
    kept = tuple(index for index in range(width) if index not in dropped_indices)
    kept_gate = torch.tensor(kept, dtype=torch.long, device=gate.device)
    kept_down = kept_gate.to(device=down_work.device)
    folded_gate = gate.index_select(0, kept_gate).contiguous()
    folded_up = up.index_select(0, kept_gate).contiguous()
    folded_down = down_work.index_select(1, kept_down).contiguous()
    diagnostics = MultiAnchorFoldDiagnostics(
        source_width=width,
        target_width=target,
        calibration_rows=int(cal.shape[0]),
        validation_rows=int(val.shape[0]),
        calibration_provenance=cal_source,
        validation_provenance=val_source,
        calibration_activation_sha256=_exact_tensor_fingerprint(calibration),
        validation_activation_sha256=_exact_tensor_fingerprint(validation),
        candidate_pool_size=pool_count,
        survivor_pool_size=int(survivor_indices.numel()),
        max_anchors=anchor_limit,
        similarity_products=similarity_products,
        peak_similarity_elements=peak_similarity_elements,
        similarity_chunk_size=chunk,
        projection_fit_count=projection_fit_count,
        max_projection_fits=projection_fit_limit,
        selected_merge_count=len(selected),
        protected_count=len(protected),
        aggregate_calibration_relative_l2=_realized_layer_relative_l2(
            calibration, down, folded_down, kept
        ),
        aggregate_validation_relative_l2=_realized_layer_relative_l2(
            validation, down, folded_down, kept
        ),
        realized_dtype=str(folded_down.dtype),
    )
    return MultiAnchorNeuronFoldWeights(
        gate_weight=folded_gate,
        up_weight=folded_up,
        down_weight=folded_down,
        kept_source_indices=kept,
        merges=selected,
        protected_indices=protected,
        source_weight_fingerprints=tuple(
            _exact_tensor_fingerprint(value) for value in (gate, up, down)
        ),
        result_weight_fingerprints=tuple(
            _exact_tensor_fingerprint(value)
            for value in (folded_gate, folded_up, folded_down)
        ),
        diagnostics=diagnostics,
    )


__all__ = [
    "MultiAnchorFoldDiagnostics",
    "MultiAnchorNeuronFoldWeights",
    "MultiAnchorNeuronMerge",
    "fit_multi_anchor_neuron_fold",
]
