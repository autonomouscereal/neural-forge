"""Closed-form, activation-compensated Qwen MLP neuron merging.

The fold keeps one real gate/up producer from each selected pair and solves a
new down-projection column for the sum of both parent contributions.  It is a
structured width reduction: no optimizer, backward pass, recovery data, or
shape-only promotion signal is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


class Qwen3CompensatedFoldError(ValueError):
    """Base error for compensated neuron folding."""


class Qwen3CompensatedFoldShapeError(Qwen3CompensatedFoldError):
    """A weight, activation, or requested width has an invalid shape."""


class Qwen3CompensatedFoldStateError(Qwen3CompensatedFoldError):
    """The runtime does not match the fitted fold transaction."""


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


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3CompensatedFoldShapeError(
            f"{name} must be a non-negative integer."
        )
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3CompensatedFoldShapeError(
            f"{name} must be a non-negative integer."
        ) from exc
    if result < 0:
        raise Qwen3CompensatedFoldShapeError(f"{name} must be non-negative.")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _exact_tensor_fingerprint(
    value: Tensor,
    *,
    chunk_elements: int = 1_048_576,
) -> str:
    """Stream every tensor element through SHA-256 without a full CPU copy."""

    chunk = _positive_int(chunk_elements, "chunk_elements")
    tensor = value.detach().reshape(-1)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    for start in range(0, int(tensor.numel()), chunk):
        piece = tensor[start : start + chunk].to(device="cpu").contiguous()
        digest.update(piece.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_matrix(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise Qwen3CompensatedFoldShapeError(f"{name} must be a rank-2 tensor.")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise Qwen3CompensatedFoldShapeError(
            f"{name} must contain finite floating-point values."
        )
    return value.detach()


@dataclass(frozen=True)
class CompensatedNeuronMerge:
    left: int
    right: int
    representative: int
    dropped: int
    alpha: float
    calibration_relative_error: float
    validation_relative_error: float
    score: float

    def __post_init__(self) -> None:
        if self.left < 0 or self.right <= self.left:
            raise Qwen3CompensatedFoldShapeError("Merge indices must be ordered.")
        if {self.representative, self.dropped} != {self.left, self.right}:
            raise Qwen3CompensatedFoldShapeError(
                "Representative and dropped indices must cover the pair."
            )
        for name in (
            "alpha",
            "calibration_relative_error",
            "validation_relative_error",
            "score",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise Qwen3CompensatedFoldShapeError(f"{name} must be finite.")
        if min(
            self.calibration_relative_error,
            self.validation_relative_error,
            self.score,
        ) < 0.0:
            raise Qwen3CompensatedFoldShapeError("Fold errors cannot be negative.")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "left": self.left,
            "right": self.right,
            "representative": self.representative,
            "dropped": self.dropped,
            "alpha": self.alpha,
            "calibration_relative_error": self.calibration_relative_error,
            "validation_relative_error": self.validation_relative_error,
            "score": self.score,
        }


@dataclass(frozen=True)
class CompensatedNeuronFoldDiagnostics:
    source_width: int
    target_width: int
    calibration_rows: int
    validation_rows: int
    calibration_activation_sha256: str
    validation_activation_sha256: str
    projection_count: int
    neighbor_window: int
    max_generated_pairs: int
    generated_pair_count: int
    scored_pair_count: int
    retained_scored_pair_count: int
    score_chunk_size: int
    selected_pair_count: int
    protected_count: int
    aggregate_calibration_relative_l2: float
    aggregate_validation_relative_l2: float
    realized_dtype: str
    materialized_full_pairwise_matrix: bool = False
    optimizer_steps: int = 0
    backward_called: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass
class CompensatedNeuronFoldWeights:
    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    kept_source_indices: tuple[int, ...]
    merges: tuple[CompensatedNeuronMerge, ...]
    protected_indices: tuple[int, ...]
    source_weight_fingerprints: tuple[str, str, str]
    result_weight_fingerprints: tuple[str, str, str]
    diagnostics: CompensatedNeuronFoldDiagnostics

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
        current = (
            _exact_tensor_fingerprint(self.gate_weight),
            _exact_tensor_fingerprint(self.up_weight),
            _exact_tensor_fingerprint(self.down_weight),
        )
        if current != self.result_weight_fingerprints:
            raise Qwen3CompensatedFoldStateError(
                "Fitted fold tensors changed after their exact hash was recorded."
            )

    @property
    def fold_sha256(self) -> str:
        self.assert_integrity()
        return _canonical_sha256(
            {
                "kept_source_indices": list(self.kept_source_indices),
                "merges": [item.as_dict() for item in self.merges],
                "protected_indices": list(self.protected_indices),
                "source_weight_fingerprints": list(self.source_weight_fingerprints),
                "result_weight_fingerprints": list(
                    self.result_weight_fingerprints
                ),
                "diagnostics": self.diagnostics.as_dict(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "target_width": self.target_width,
            "parameter_savings": self.parameter_savings,
            "kept_source_indices": list(self.kept_source_indices),
            "merges": [item.as_dict() for item in self.merges],
            "protected_indices": list(self.protected_indices),
            "source_weight_fingerprints": list(self.source_weight_fingerprints),
            "result_weight_fingerprints": list(self.result_weight_fingerprints),
            "diagnostics": self.diagnostics.as_dict(),
            "fold_sha256": self.fold_sha256,
        }


def _validate_fold_inputs(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    calibration_activations: Tensor,
    validation_activations: Tensor,
    target_width: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int]:
    gate = _finite_matrix(gate_weight, "gate_weight")
    up = _finite_matrix(up_weight, "up_weight")
    down = _finite_matrix(down_weight, "down_weight")
    calibration = _finite_matrix(calibration_activations, "calibration_activations")
    validation = _finite_matrix(
        validation_activations, "validation_activations"
    )
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
    if _exact_tensor_fingerprint(calibration) == _exact_tensor_fingerprint(validation):
        raise Qwen3CompensatedFoldShapeError(
            "Calibration and validation activations must be provenance-disjoint."
        )
    requested = _positive_int(target_width, "target_width")
    if requested >= width:
        raise Qwen3CompensatedFoldShapeError(
            "target_width must be smaller than the source width."
        )
    return gate, up, down, calibration, validation, width, hidden


def _candidate_pairs(
    calibration: Tensor,
    protected: frozenset[int],
    *,
    projection_count: int,
    neighbor_window: int,
    max_pair_candidates: int,
    max_generated_pairs: int,
    seed: int,
) -> tuple[list[tuple[int, int]], int]:
    work = calibration.to(device="cpu", dtype=torch.float32)
    norms = torch.linalg.vector_norm(work, dim=0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized = work / norms
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_nonnegative_int(seed, "seed"))
    directions = torch.randn(
        int(work.shape[0]),
        projection_count,
        generator=generator,
        dtype=torch.float32,
    )
    sketches = normalized.T @ directions
    pairs: set[tuple[int, int]] = set()
    exhausted = False
    for projection in range(projection_count):
        orders = (
            torch.argsort(sketches[:, projection], stable=True).tolist(),
            torch.argsort(sketches[:, projection].abs(), stable=True).tolist(),
        )
        for order in orders:
            for offset in range(1, neighbor_window + 1):
                for left, right in zip(order[:-offset], order[offset:], strict=True):
                    if left == right or left in protected or right in protected:
                        continue
                    pairs.add((min(left, right), max(left, right)))
                    if len(pairs) >= max_generated_pairs:
                        exhausted = True
                        break
                if exhausted:
                    break
            if exhausted:
                break
        if exhausted:
            break
    generated = len(pairs)
    if not pairs:
        return [], generated
    ordered = sorted(pairs)
    left = torch.tensor([item[0] for item in ordered], dtype=torch.long)
    right = torch.tensor([item[1] for item in ordered], dtype=torch.long)
    similarity = (normalized[:, left] * normalized[:, right]).sum(dim=0).abs()
    keep = min(max_pair_candidates, len(ordered))
    selected = torch.topk(similarity, keep, largest=True, sorted=True).indices.tolist()
    return [ordered[index] for index in selected], generated


def _pair_scores(
    down: Tensor,
    calibration: Tensor,
    validation: Tensor,
    pairs: list[tuple[int, int]],
    *,
    keep_limit: int,
    chunk_size: int,
) -> list[CompensatedNeuronMerge]:
    if not pairs:
        return []
    device = torch.device("cpu")
    a = calibration.to(device=device, dtype=torch.float64)
    h = validation.to(device=device, dtype=torch.float64)
    d = down.detach().to(device=device, dtype=torch.float32)
    tiny = torch.finfo(torch.float64).tiny
    retained: list[CompensatedNeuronMerge] = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        left = torch.tensor([item[0] for item in chunk], dtype=torch.long)
        right = torch.tensor([item[1] for item in chunk], dtype=torch.long)
        ai = a.index_select(1, left)
        aj = a.index_select(1, right)
        ni = ai.square().sum(dim=0).clamp_min(tiny)
        nj = aj.square().sum(dim=0).clamp_min(tiny)
        cross = (ai * aj).sum(dim=0)
        di = d.index_select(1, left).to(dtype=torch.float64)
        dj = d.index_select(1, right).to(dtype=torch.float64)
        di2 = di.square().sum(dim=0)
        dj2 = dj.square().sum(dim=0)
        didj = (di * dj).sum(dim=0)
        original_energy = (
            ni * di2 + nj * dj2 + 2.0 * cross * didj
        ).clamp_min(tiny)

        alpha_left = cross / ni
        alpha_right = cross / nj
        residual_left = (
            (aj - ai * alpha_left).square().sum(dim=0) * dj2
        ).clamp_min(0.0)
        residual_right = (
            (ai - aj * alpha_right).square().sum(dim=0) * di2
        ).clamp_min(0.0)
        cal_left = torch.sqrt(residual_left / original_energy)
        cal_right = torch.sqrt(residual_right / original_energy)

        hi = h.index_select(1, left)
        hj = h.index_select(1, right)
        hni = hi.square().sum(dim=0)
        hnj = hj.square().sum(dim=0)
        hcross = (hi * hj).sum(dim=0)
        validation_energy = (
            hni * di2 + hnj * dj2 + 2.0 * hcross * didj
        ).clamp_min(tiny)
        validation_left = torch.sqrt(
            ((hj - hi * alpha_left).square().sum(dim=0) * dj2).clamp_min(0.0)
            / validation_energy
        )
        validation_right = torch.sqrt(
            ((hi - hj * alpha_right).square().sum(dim=0) * di2).clamp_min(0.0)
            / validation_energy
        )
        score_left = torch.maximum(cal_left, validation_left)
        score_right = torch.maximum(cal_right, validation_right)
        keep_left = score_left <= score_right

        for index, (left_index, right_index) in enumerate(chunk):
            choose_left = bool(keep_left[index].item())
            retained.append(
                CompensatedNeuronMerge(
                    left=left_index,
                    right=right_index,
                    representative=left_index if choose_left else right_index,
                    dropped=right_index if choose_left else left_index,
                    alpha=float(
                        alpha_left[index].item()
                        if choose_left
                        else alpha_right[index].item()
                    ),
                    calibration_relative_error=float(
                        cal_left[index].item()
                        if choose_left
                        else cal_right[index].item()
                    ),
                    validation_relative_error=float(
                        validation_left[index].item()
                        if choose_left
                        else validation_right[index].item()
                    ),
                    score=float(
                        score_left[index].item()
                        if choose_left
                        else score_right[index].item()
                    ),
                )
            )
        if len(retained) > 2 * keep_limit:
            retained.sort(
                key=lambda item: (
                    item.score,
                    item.left,
                    item.right,
                    item.representative,
                )
            )
            del retained[keep_limit:]
    retained.sort(
        key=lambda item: (item.score, item.left, item.right, item.representative)
    )
    return retained[:keep_limit]


def _realized_layer_relative_l2(
    activations: Tensor,
    source_down: Tensor,
    folded_down: Tensor,
    kept_indices: tuple[int, ...],
) -> float:
    values = activations.detach().to(device="cpu", dtype=torch.float64)
    source = source_down.detach().to(device="cpu", dtype=torch.float64)
    folded = folded_down.detach().to(device="cpu", dtype=torch.float64)
    kept = torch.tensor(kept_indices, dtype=torch.long)
    expected = values @ source.T
    actual = values.index_select(1, kept) @ folded.T
    denominator = torch.linalg.vector_norm(expected).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return float((torch.linalg.vector_norm(expected - actual) / denominator).item())


def fit_compensated_neuron_pairs(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    calibration_activations: Tensor,
    *,
    target_width: int,
    validation_activations: Tensor,
    protected_indices: tuple[int, ...] | list[int] = (),
    projection_count: int = 8,
    neighbor_window: int = 4,
    max_pair_candidates: int = 131_072,
    max_generated_pairs: int = 524_288,
    score_chunk_size: int = 1_024,
    seed: int = 0,
) -> CompensatedNeuronFoldWeights:
    """Fit on calibration, select on validation, then await an external gate."""

    gate, up, down, calibration, validation, width, _hidden = _validate_fold_inputs(
        gate_weight,
        up_weight,
        down_weight,
        calibration_activations,
        validation_activations,
        target_width,
    )
    projections = _positive_int(projection_count, "projection_count")
    window = _positive_int(neighbor_window, "neighbor_window")
    pair_limit = _positive_int(max_pair_candidates, "max_pair_candidates")
    generated_limit = _positive_int(max_generated_pairs, "max_generated_pairs")
    scoring_chunk = _positive_int(score_chunk_size, "score_chunk_size")
    if projections > 64:
        raise Qwen3CompensatedFoldShapeError("projection_count exceeds hard cap 64.")
    if window > 64 or window >= width:
        raise Qwen3CompensatedFoldShapeError(
            "neighbor_window must be below the width and hard cap 64."
        )
    if pair_limit > generated_limit:
        raise Qwen3CompensatedFoldShapeError(
            "max_pair_candidates cannot exceed max_generated_pairs."
        )
    protected_values = tuple(
        sorted({_nonnegative_int(index, "protected index") for index in protected_indices})
    )
    if any(index >= width for index in protected_values):
        raise Qwen3CompensatedFoldShapeError("A protected index is out of range.")
    merges_required = width - int(target_width)
    if 2 * merges_required > width - len(protected_values):
        raise Qwen3CompensatedFoldShapeError(
            "This disjoint-pair pass cannot reach the requested width."
        )

    pairs, generated_count = _candidate_pairs(
        calibration,
        frozenset(protected_values),
        projection_count=projections,
        neighbor_window=window,
        max_pair_candidates=pair_limit,
        max_generated_pairs=generated_limit,
        seed=seed,
    )
    retained_score_limit = min(
        len(pairs),
        max(4_096, merges_required * 32),
    )
    scored = _pair_scores(
        down,
        calibration,
        validation,
        pairs,
        keep_limit=retained_score_limit,
        chunk_size=scoring_chunk,
    )
    selected: list[CompensatedNeuronMerge] = []
    occupied: set[int] = set(protected_values)
    for candidate in scored:
        if candidate.left in occupied or candidate.right in occupied:
            continue
        selected.append(candidate)
        occupied.update((candidate.left, candidate.right))
        if len(selected) == merges_required:
            break
    if len(selected) != merges_required:
        raise Qwen3CompensatedFoldShapeError(
            f"Bounded candidate search found {len(selected)} disjoint merges; "
            f"{merges_required} are required."
        )

    down_work = down.detach().clone()
    dropped: set[int] = set()
    for merge in selected:
        down_work[:, merge.representative] = (
            down_work[:, merge.representative]
            + float(merge.alpha) * down_work[:, merge.dropped]
        )
        dropped.add(merge.dropped)
    kept = tuple(index for index in range(width) if index not in dropped)
    kept_tensor = torch.tensor(kept, dtype=torch.long, device=gate.device)
    down_kept_tensor = kept_tensor.to(device=down_work.device)
    folded_gate = gate.index_select(0, kept_tensor).contiguous()
    folded_up = up.index_select(0, kept_tensor).contiguous()
    folded_down = down_work.index_select(1, down_kept_tensor).contiguous()
    calibration_fingerprint = _exact_tensor_fingerprint(calibration)
    validation_fingerprint = _exact_tensor_fingerprint(validation)
    diagnostics = CompensatedNeuronFoldDiagnostics(
        source_width=width,
        target_width=int(target_width),
        calibration_rows=int(calibration.shape[0]),
        validation_rows=int(validation.shape[0]),
        calibration_activation_sha256=calibration_fingerprint,
        validation_activation_sha256=validation_fingerprint,
        projection_count=projections,
        neighbor_window=window,
        max_generated_pairs=generated_limit,
        generated_pair_count=generated_count,
        scored_pair_count=len(pairs),
        retained_scored_pair_count=len(scored),
        score_chunk_size=scoring_chunk,
        selected_pair_count=len(selected),
        protected_count=len(protected_values),
        aggregate_calibration_relative_l2=_realized_layer_relative_l2(
            calibration,
            down,
            folded_down,
            kept,
        ),
        aggregate_validation_relative_l2=_realized_layer_relative_l2(
            validation,
            down,
            folded_down,
            kept,
        ),
        realized_dtype=str(folded_down.dtype),
    )
    return CompensatedNeuronFoldWeights(
        gate_weight=folded_gate,
        up_weight=folded_up,
        down_weight=folded_down,
        kept_source_indices=kept,
        merges=tuple(selected),
        protected_indices=protected_values,
        source_weight_fingerprints=(
            _exact_tensor_fingerprint(gate),
            _exact_tensor_fingerprint(up),
            _exact_tensor_fingerprint(down),
        ),
        result_weight_fingerprints=(
            _exact_tensor_fingerprint(folded_gate),
            _exact_tensor_fingerprint(folded_up),
            _exact_tensor_fingerprint(folded_down),
        ),
        diagnostics=diagnostics,
    )


class Qwen3CompensatedMLP(nn.Module):
    """A stock-compatible Qwen SwiGLU MLP with a physically smaller width."""

    def __init__(self, hidden_size: int, intermediate_size: int, act_fn: nn.Module | Any):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


@dataclass
class Qwen3NeuronFoldHandle:
    model: nn.Module
    layer_index: int
    fold: CompensatedNeuronFoldWeights
    original_mlp: nn.Module | None
    installed_mlp: nn.Module
    committed: bool = False
    reverted: bool = False

    def commit(self) -> None:
        if self.reverted:
            raise Qwen3CompensatedFoldStateError("A reverted fold cannot be committed.")
        if self.committed:
            return
        self.original_mlp = None
        self.committed = True

    def rollback(self) -> None:
        if self.committed:
            raise Qwen3CompensatedFoldStateError("A committed fold cannot be rolled back.")
        if self.reverted:
            return
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if not isinstance(layers, nn.ModuleList) or self.original_mlp is None:
            raise Qwen3CompensatedFoldStateError("The original Qwen MLP is unavailable.")
        if layers[self.layer_index].mlp is not self.installed_mlp:
            raise Qwen3CompensatedFoldStateError("The installed Qwen MLP changed externally.")
        layers[self.layer_index].mlp = self.original_mlp
        self.reverted = True


def install_qwen3_compensated_fold(
    model: nn.Module,
    layer_index: int,
    fold: CompensatedNeuronFoldWeights,
) -> Qwen3NeuronFoldHandle:
    """Install one fitted layer transaction; caller must commit or roll it back."""

    fold.assert_integrity()
    layers = getattr(getattr(model, "model", None), "layers", None)
    index = _nonnegative_int(layer_index, "layer_index")
    if not isinstance(layers, nn.ModuleList) or index >= len(layers):
        raise Qwen3CompensatedFoldShapeError("model is not a compatible Qwen runtime.")
    original = layers[index].mlp
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(original, name):
            raise Qwen3CompensatedFoldShapeError(f"Qwen MLP lacks {name}.")
    current = (
        original.gate_proj.weight,
        original.up_proj.weight,
        original.down_proj.weight,
    )
    fingerprints = tuple(_exact_tensor_fingerprint(value) for value in current)
    if fingerprints != fold.source_weight_fingerprints:
        raise Qwen3CompensatedFoldStateError(
            "The fitted source weights do not match the selected Qwen layer."
        )
    device = current[0].device
    dtype = current[0].dtype
    installed = Qwen3CompensatedMLP(
        int(current[0].shape[1]),
        fold.target_width,
        original.act_fn,
    ).to(device=device, dtype=dtype)
    with torch.inference_mode():
        installed.gate_proj.weight.copy_(fold.gate_weight.to(device=device, dtype=dtype))
        installed.up_proj.weight.copy_(fold.up_weight.to(device=device, dtype=dtype))
        installed.down_proj.weight.copy_(fold.down_weight.to(device=device, dtype=dtype))
    installed_fingerprints = tuple(
        _exact_tensor_fingerprint(value)
        for value in (
            installed.gate_proj.weight,
            installed.up_proj.weight,
            installed.down_proj.weight,
        )
    )
    if installed_fingerprints != fold.result_weight_fingerprints:
        raise Qwen3CompensatedFoldStateError(
            "Installed Qwen fold weights differ after destination-dtype copy."
        )
    installed.train(original.training)
    layers[index].mlp = installed
    return Qwen3NeuronFoldHandle(
        model=model,
        layer_index=index,
        fold=fold,
        original_mlp=original,
        installed_mlp=installed,
    )


def synchronize_qwen3_intermediate_size(model: nn.Module) -> int | None:
    """Update config only when every installed Qwen MLP has the same width."""

    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise Qwen3CompensatedFoldShapeError("model is not a compatible Qwen runtime.")
    widths = {int(layer.mlp.gate_proj.weight.shape[0]) for layer in layers}
    if len(widths) != 1:
        return None
    width = next(iter(widths))
    model.config.intermediate_size = width
    inner_config = getattr(getattr(model, "model", None), "config", None)
    if inner_config is not None:
        inner_config.intermediate_size = width
    return width


__all__ = [
    "CompensatedNeuronFoldDiagnostics",
    "CompensatedNeuronFoldWeights",
    "CompensatedNeuronMerge",
    "Qwen3CompensatedFoldError",
    "Qwen3CompensatedFoldShapeError",
    "Qwen3CompensatedFoldStateError",
    "Qwen3CompensatedMLP",
    "Qwen3NeuronFoldHandle",
    "fit_compensated_neuron_pairs",
    "install_qwen3_compensated_fold",
    "synchronize_qwen3_intermediate_size",
]
