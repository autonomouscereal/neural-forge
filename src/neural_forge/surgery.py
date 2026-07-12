"""Neuron-level shape surgery for dense checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neural_forge.checkpoint import DenseModel, param_count_for_widths


@dataclass(frozen=True)
class WidthPlan:
    input_width: int
    hidden_widths: list[int]
    output_width: int
    target_params: int
    achieved_params: int
    exact_or_under: bool
    minimum_reached: bool

    @property
    def widths(self) -> list[int]:
        return [self.input_width] + list(self.hidden_widths) + [self.output_width]

    def as_dict(self) -> dict[str, object]:
        return {
            "widths": self.widths,
            "hidden_widths": self.hidden_widths,
            "target_params": self.target_params,
            "achieved_params": self.achieved_params,
            "exact_or_under": self.exact_or_under,
            "minimum_reached": self.minimum_reached,
        }


def parse_widths(text: str) -> list[int]:
    widths = [int(part.strip()) for part in text.split(",") if part.strip()]
    if any(width <= 0 for width in widths):
        raise ValueError("Widths must all be positive integers.")
    return widths


def plan_widths_for_target_params(model: DenseModel, target_params: int, min_width: int = 1) -> WidthPlan:
    if target_params <= 0:
        raise ValueError("target_params must be positive.")
    if min_width <= 0:
        raise ValueError("min_width must be positive.")
    input_width = model.input_dim
    output_width = model.output_dim
    hidden = list(model.hidden_widths)
    if not hidden:
        achieved = param_count_for_widths([input_width, output_width])
        return WidthPlan(input_width, [], output_width, target_params, achieved, achieved <= target_params, True)

    def count() -> int:
        return param_count_for_widths([input_width] + hidden + [output_width])

    current = count()
    best_hidden = list(hidden)
    best_count = current
    minimum_reached = False
    if target_params < current:
        while True:
            current = count()
            impacts = _unit_impacts([input_width] + hidden + [output_width])
            candidates = [(impact, idx) for idx, impact in enumerate(impacts) if hidden[idx] > min_width]
            if not candidates:
                minimum_reached = True
                break
            best_move = min(
                ((abs((current - impact) - target_params), impact, idx) for impact, idx in candidates),
                key=lambda item: (item[0], -item[1]),
            )
            if best_move[0] >= abs(best_count - target_params):
                break
            _, _, chosen = best_move
            hidden[chosen] -= 1
            current = count()
            if abs(current - target_params) < abs(best_count - target_params):
                best_hidden = list(hidden)
                best_count = current
            if current <= target_params:
                break
    elif target_params > current:
        while True:
            current = count()
            widths = [input_width] + hidden + [output_width]
            impacts = _unit_impacts(widths)
            best_move = min(
                ((abs((current + impact) - target_params), impact, idx) for idx, impact in enumerate(impacts)),
                key=lambda item: (item[0], item[1]),
            )
            if best_move[0] >= abs(best_count - target_params):
                break
            _, _, chosen = best_move
            hidden[chosen] += 1
            current = count()
            if abs(current - target_params) < abs(best_count - target_params):
                best_hidden = list(hidden)
                best_count = current
            if current >= target_params:
                break
    hidden = best_hidden
    achieved = best_count
    return WidthPlan(
        input_width=input_width,
        hidden_widths=hidden,
        output_width=output_width,
        target_params=target_params,
        achieved_params=achieved,
        exact_or_under=achieved <= target_params,
        minimum_reached=minimum_reached,
    )


def resize_to_widths(model: DenseModel, hidden_widths: Sequence[int], seed: int = 0, noise_std: float = 0.0) -> DenseModel:
    if len(hidden_widths) != model.hidden_depth:
        raise ValueError(
            f"Expected {model.hidden_depth} hidden widths for this model, got {len(hidden_widths)}."
        )
    resized = model.copy()
    rng = np.random.default_rng(seed)
    for idx, target in enumerate(hidden_widths):
        target_width = int(target)
        if target_width <= 0:
            raise ValueError("Hidden widths must be positive.")
        current = resized.weights[idx].shape[0]
        if target_width < current:
            resized = shrink_hidden_layer(resized, idx, target_width)
        elif target_width > current:
            resized = expand_hidden_layer(resized, idx, target_width, rng=rng, noise_std=noise_std)
    return resized


def shrink_hidden_layer(model: DenseModel, hidden_index: int, new_width: int) -> DenseModel:
    _check_hidden_index(model, hidden_index)
    old_width = model.weights[hidden_index].shape[0]
    if new_width <= 0:
        raise ValueError("new_width must be positive.")
    if new_width > old_width:
        raise ValueError("new_width must be <= the current width.")
    if new_width == old_width:
        return model.copy()

    features = _neuron_features(model, hidden_index)
    assignments = _cluster_assignments(features, new_width)
    clusters = _clusters_from_assignments(assignments, new_width)

    output = model.copy()
    incoming = output.weights[hidden_index]
    bias = output.biases[hidden_index]
    outgoing = output.weights[hidden_index + 1]

    new_incoming = np.zeros((new_width, incoming.shape[1]), dtype=np.float64)
    new_bias = np.zeros((new_width,), dtype=np.float64)
    new_outgoing = np.zeros((outgoing.shape[0], new_width), dtype=np.float64)
    for new_idx, members in enumerate(clusters):
        member_array = np.array(members, dtype=np.int64)
        new_incoming[new_idx] = incoming[member_array].mean(axis=0)
        new_bias[new_idx] = bias[member_array].mean(axis=0)
        new_outgoing[:, new_idx] = outgoing[:, member_array].sum(axis=1)

    output.weights[hidden_index] = new_incoming
    output.biases[hidden_index] = new_bias
    output.weights[hidden_index + 1] = new_outgoing
    output.validate()
    return output


def expand_hidden_layer(
    model: DenseModel,
    hidden_index: int,
    new_width: int,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.0,
) -> DenseModel:
    _check_hidden_index(model, hidden_index)
    old_width = model.weights[hidden_index].shape[0]
    if new_width <= old_width:
        if new_width == old_width:
            return model.copy()
        raise ValueError("new_width must be >= the current width.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    generator = np.random.default_rng(0) if rng is None else rng
    incoming = model.weights[hidden_index]
    bias = model.biases[hidden_index]
    outgoing = model.weights[hidden_index + 1]
    scores = np.linalg.norm(outgoing, axis=0) + np.linalg.norm(incoming, axis=1)
    order = list(np.argsort(-scores))

    counts = np.ones(old_width, dtype=np.int64)
    for extra_idx in range(new_width - old_width):
        counts[order[extra_idx % old_width]] += 1

    new_incoming = []
    new_bias = []
    new_outgoing = []
    for old_idx, copies in enumerate(counts):
        for _ in range(int(copies)):
            row = incoming[old_idx].copy()
            if noise_std:
                row = row + generator.normal(0.0, noise_std, size=row.shape)
            new_incoming.append(row)
            new_bias.append(float(bias[old_idx]))
            new_outgoing.append(outgoing[:, old_idx] / float(copies))

    output = model.copy()
    output.weights[hidden_index] = np.vstack(new_incoming)
    output.biases[hidden_index] = np.array(new_bias, dtype=np.float64)
    output.weights[hidden_index + 1] = np.stack(new_outgoing, axis=1)
    output.validate()
    return output


def expand_depth_identity(model: DenseModel, insertions: int = 1) -> DenseModel:
    if insertions < 0:
        raise ValueError("insertions must be non-negative.")
    if insertions == 0:
        return model.copy()
    if model.activation != "relu":
        raise ValueError("Identity depth expansion is only function-preserving for relu in this MVP.")
    if model.hidden_depth == 0:
        raise ValueError("Need at least one hidden layer to insert a relu identity block.")

    output = model.copy()
    for insertion_idx in range(insertions):
        hidden_index = min(output.hidden_depth - 1, insertion_idx % output.hidden_depth)
        width = output.weights[hidden_index].shape[0]
        insert_at = hidden_index + 1
        output.weights.insert(insert_at, np.eye(width, dtype=np.float64))
        output.biases.insert(insert_at, np.zeros((width,), dtype=np.float64))
    output.validate()
    return output


def merge_models(
    models: Sequence[DenseModel],
    target_hidden_widths: Sequence[int] | None = None,
    weights: Sequence[float] | None = None,
    method: str = "average",
    align: bool = True,
    seed: int = 0,
    noise_std: float = 0.0,
) -> DenseModel:
    if not models:
        raise ValueError("Need at least one model to merge.")
    method = method.lower()
    if method not in {"average", "slerp"}:
        raise ValueError("method must be average or slerp.")
    _validate_merge_compatibility(models)

    if target_hidden_widths is None:
        hidden_depth = models[0].hidden_depth
        target_hidden_widths = [max(model.hidden_widths[idx] for model in models) for idx in range(hidden_depth)]
    target_hidden_widths = [int(width) for width in target_hidden_widths]

    resized: list[DenseModel] = []
    for idx, model in enumerate(models):
        resized.append(resize_to_widths(model, target_hidden_widths, seed=seed + idx, noise_std=noise_std))

    if align and len(resized) > 1:
        reference = resized[0]
        resized = [reference] + [align_model_to_reference(model, reference) for model in resized[1:]]

    merge_weights = _normalized_merge_weights(len(resized), weights)
    if method == "slerp":
        if len(resized) != 2:
            raise ValueError("slerp requires exactly two models.")
        merged = _slerp_models(resized[0], resized[1], merge_weights[1])
    else:
        merged_weights = []
        merged_biases = []
        for layer_idx in range(resized[0].depth):
            merged_weights.append(
                sum(weight * model.weights[layer_idx] for weight, model in zip(merge_weights, resized))
            )
            merged_biases.append(
                sum(weight * model.biases[layer_idx] for weight, model in zip(merge_weights, resized))
            )
        merged = DenseModel(
            weights=merged_weights,
            biases=merged_biases,
            activation=resized[0].activation,
            name="merged",
        )
    merged.validate()
    return merged


def align_model_to_reference(model: DenseModel, reference: DenseModel) -> DenseModel:
    if model.widths != reference.widths:
        raise ValueError("Alignment requires matching widths.")
    aligned = model.copy()
    for hidden_index in range(aligned.hidden_depth):
        ref_features = _neuron_features(reference, hidden_index)
        src_features = _neuron_features(aligned, hidden_index)
        order = _greedy_match_order(ref_features, src_features)
        aligned = _permute_hidden_layer(aligned, hidden_index, order)
    return aligned


def _unit_impacts(widths: Sequence[int]) -> list[int]:
    return [int(widths[idx] + 1 + widths[idx + 2]) for idx in range(len(widths) - 2)]


def _check_hidden_index(model: DenseModel, hidden_index: int) -> None:
    if hidden_index < 0 or hidden_index >= model.hidden_depth:
        raise IndexError(f"hidden_index must be in [0, {model.hidden_depth - 1}], got {hidden_index}.")


def _neuron_features(model: DenseModel, hidden_index: int) -> np.ndarray:
    incoming = model.weights[hidden_index]
    bias = model.biases[hidden_index][:, None]
    outgoing = model.weights[hidden_index + 1].T
    feature = np.concatenate([incoming, bias, outgoing], axis=1)
    return _row_normalize(feature)


def _row_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _cluster_assignments(features: np.ndarray, clusters: int, iterations: int = 40) -> np.ndarray:
    count = features.shape[0]
    if clusters >= count:
        return np.arange(count, dtype=np.int64)

    centroids = _farthest_point_centroids(features, clusters)
    assignments = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        distances = _squared_distances(features, centroids)
        next_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for cluster_idx in range(clusters):
            members = features[assignments == cluster_idx]
            if len(members):
                centroids[cluster_idx] = members.mean(axis=0)
    return assignments


def _farthest_point_centroids(features: np.ndarray, clusters: int) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1)
    first = int(np.argmax(norms))
    chosen = [first]
    while len(chosen) < clusters:
        distances = _squared_distances(features, features[chosen])
        min_distance = np.min(distances, axis=1)
        min_distance[chosen] = -1.0
        chosen.append(int(np.argmax(min_distance)))
    return features[chosen].copy()


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)


def _clusters_from_assignments(assignments: np.ndarray, clusters: int) -> list[list[int]]:
    grouped: list[list[int]] = [[] for _ in range(clusters)]
    for old_idx, cluster_idx in enumerate(assignments):
        grouped[int(cluster_idx)].append(old_idx)
    empty = [idx for idx, members in enumerate(grouped) if not members]
    if empty:
        raise RuntimeError(f"Internal clustering error: empty clusters {empty}.")
    grouped.sort(key=lambda members: min(members))
    return grouped


def _validate_merge_compatibility(models: Sequence[DenseModel]) -> None:
    first = models[0]
    for idx, model in enumerate(models[1:], start=1):
        if model.input_dim != first.input_dim or model.output_dim != first.output_dim:
            raise ValueError(f"Model {idx} input/output dims do not match the first model.")
        if model.hidden_depth != first.hidden_depth:
            raise ValueError(f"Model {idx} hidden depth does not match the first model.")
        if model.activation != first.activation:
            raise ValueError(f"Model {idx} activation does not match the first model.")


def _normalized_merge_weights(count: int, weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} merge weights, got {len(weights)}.")
    values = [float(weight) for weight in weights]
    total = sum(values)
    if total == 0:
        raise ValueError("Merge weights must not sum to zero.")
    return [weight / total for weight in values]


def _greedy_match_order(reference_features: np.ndarray, source_features: np.ndarray) -> list[int]:
    similarity = reference_features @ source_features.T
    unused = set(range(source_features.shape[0]))
    order: list[int] = []
    for ref_idx in range(reference_features.shape[0]):
        chosen = max(unused, key=lambda src_idx: similarity[ref_idx, src_idx])
        order.append(int(chosen))
        unused.remove(chosen)
    return order


def _permute_hidden_layer(model: DenseModel, hidden_index: int, order: Sequence[int]) -> DenseModel:
    output = model.copy()
    index = np.array(order, dtype=np.int64)
    output.weights[hidden_index] = output.weights[hidden_index][index]
    output.biases[hidden_index] = output.biases[hidden_index][index]
    output.weights[hidden_index + 1] = output.weights[hidden_index + 1][:, index]
    output.validate()
    return output


def _flatten_model(model: DenseModel) -> np.ndarray:
    parts = []
    for weight, bias in zip(model.weights, model.biases):
        parts.append(weight.reshape(-1))
        parts.append(bias.reshape(-1))
    return np.concatenate(parts)


def _unflatten_like(vector: np.ndarray, template: DenseModel, name: str) -> DenseModel:
    offset = 0
    weights = []
    biases = []
    for weight, bias in zip(template.weights, template.biases):
        weight_size = weight.size
        weights.append(vector[offset : offset + weight_size].reshape(weight.shape))
        offset += weight_size
        bias_size = bias.size
        biases.append(vector[offset : offset + bias_size].reshape(bias.shape))
        offset += bias_size
    return DenseModel(weights=weights, biases=biases, activation=template.activation, name=name)


def _slerp_models(a: DenseModel, b: DenseModel, t: float) -> DenseModel:
    vec_a = _flatten_model(a)
    vec_b = _flatten_model(b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return _unflatten_like((1.0 - t) * vec_a + t * vec_b, a, "merged-slerp")
    unit_a = vec_a / norm_a
    unit_b = vec_b / norm_b
    dot = float(np.clip(unit_a @ unit_b, -1.0, 1.0))
    omega = float(np.arccos(dot))
    if abs(omega) < 1e-8:
        merged = (1.0 - t) * vec_a + t * vec_b
    else:
        merged_unit = (np.sin((1.0 - t) * omega) * unit_a + np.sin(t * omega) * unit_b) / np.sin(omega)
        merged_norm = (1.0 - t) * norm_a + t * norm_b
        merged = merged_unit * merged_norm
    return _unflatten_like(merged, a, "merged-slerp")
